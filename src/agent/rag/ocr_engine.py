"""RapidOCR engine construction + image pre-processing.

Two things live here:

1. **Execution provider selection** — RapidOCR runs its three ONNX models
   (det / cls / rec) through ``onnxruntime.InferenceSession``.  Its bundled
   :class:`OrtInferSession` hardcodes the EP list to CUDA / DirectML / CPU —
   **CoreML is never considered**, so on Apple Silicon every OCR run burns
   CPU only while the Neural Engine and GPU sit idle.  We patch the EP list
   to put ``CoreMLExecutionProvider`` first when (and only when) the machine
   really is Apple Silicon and ``onnxruntime`` is new enough.

2. **Pre-processing / normalisation** — scanned pages vary wildly in DPI,
   contrast and noise.  Feeding the raw render to the detector loses small
   glyphs (the detector downsamples the short side to 736px by default).
   We normalise scale and enhance contrast before inference.

Everything is best-effort: any failure falls back to the stock behaviour
(CPU EP / raw image) instead of breaking ingestion.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Apple Silicon detection
# ---------------------------------------------------------------------------

# CoreML EP requires onnxruntime >= 1.31.0 (below that the CoreML EP is either
# absent or too buggy to trust for these models).
_ORT_MIN_VERSION_FOR_COREML: Tuple[int, int, int] = (1, 31, 0)

_COREML_EP = "CoreMLExecutionProvider"
_CPU_EP = "CPUExecutionProvider"

# Compute units "ALL" lets CoreML schedule on CPU + GPU + ANE (Neural Engine),
# which is what actually makes OCR several times faster on M-series Macs.
# ModelFormat MLProgram is the modern CoreML format (coremltools >= 7).
_COREML_EP_OPTIONS: Dict[str, Any] = {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "ALL",
}

_apple_silicon_cache: Optional[bool] = None
_apple_silicon_lock = threading.Lock()


def is_apple_silicon() -> bool:
    """True when running on an Apple M-series Mac.

    ``platform.machine()`` reports ``arm64`` for native arm64 builds, but a
    Rosetta-translated interpreter reports ``x86_64`` on an M-series machine —
    so it alone is not enough.  We fall back to ``sysctl`` which always reports
    the *real* CPU ("Apple M1 Pro", "Apple M3 Max", …).
    """
    global _apple_silicon_cache
    if _apple_silicon_cache is not None:
        return _apple_silicon_cache
    with _apple_silicon_lock:
        if _apple_silicon_cache is not None:
            return _apple_silicon_cache
        _apple_silicon_cache = _detect_apple_silicon()
        return _apple_silicon_cache


def _detect_apple_silicon() -> bool:
    if platform.system() != "Darwin":
        return False
    if platform.machine() in ("arm64", "aarch64"):
        return True
    # Rosetta: machine() lies, ask the kernel for the real CPU name.
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=3,
        )
        brand = (out.stdout or "").strip()
    except Exception:  # pragma: no cover - sysctl missing/timeout
        brand = ""
    if re.match(r"^Apple\s+M\d", brand):
        return True
    # Very old Apple Silicon report "Apple processor"/virtualized CPUs.
    return bool(re.match(r"^Apple\s+", brand)) and "Intel" not in brand


def ort_version() -> Optional[Tuple[int, int, int]]:
    """Installed onnxruntime version as a tuple, or None if unavailable."""
    try:
        import onnxruntime as ort
    except Exception:
        return None
    raw = str(getattr(ort, "__version__", "") or "")
    nums: List[int] = []
    for part in raw.split("."):
        m = re.match(r"^(\d+)", part.strip())
        if not m:
            break
        nums.append(int(m.group(1)))
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def coreml_reason() -> Tuple[bool, str]:
    """Explain whether CoreML EP *can* be used, and why (not).

    Returns ``(usable, human_readable_reason)``.  Purely diagnostic — no
    side effects, safe to call for logging/UI.
    """
    if not is_apple_silicon():
        return False, f"非 Apple Silicon（{platform.system()} {platform.machine()}），跳过 CoreML EP"
    ver = ort_version()
    if ver is None:
        return False, "onnxruntime 未安装"
    if ver < _ORT_MIN_VERSION_FOR_COREML:
        return False, (
            f"onnxruntime {'.'.join(str(v) for v in ver)} < "
            f"{'.'.join(str(v) for v in _ORT_MIN_VERSION_FOR_COREML)}，"
            f"请升级: pip install -U 'onnxruntime>={'.'.join(str(v) for v in _ORT_MIN_VERSION_FOR_COREML)}'"
        )
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception as e:
        return False, f"无法读取 providers: {e}"
    if _COREML_EP not in providers:
        return False, f"当前 onnxruntime 未编译 CoreML EP（可用: {providers}）"
    return True, f"Apple Silicon + onnxruntime {'.'.join(str(v) for v in ver)} + CoreML EP 可用"


# ---------------------------------------------------------------------------
# CoreML EP injection
# ---------------------------------------------------------------------------

_patch_lock = threading.Lock()


def _ort_infer_session_class():
    """Locate RapidOCR's ``OrtInferSession`` across package layouts.

    Different rapidocr_onnxruntime releases place it in different modules;
    return ``None`` when it cannot be found so callers can bail out instead
    of guessing at internals.
    """
    candidates = (
        "rapidocr_onnxruntime.utils.infer_engine",
        "rapidocr_onnxruntime.inference_engine.ort_inference_engine",
        "rapidocr_onnxruntime.utils.ort_infer_engine",
    )
    import importlib
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        cls = getattr(mod, "OrtInferSession", None)
        if cls is not None and hasattr(cls, "_get_ep_list"):
            return cls
    return None


def _patch_coreml_ep(session_cls) -> bool:
    """Prepend CoreMLExecutionProvider to the EP list RapidOCR builds.

    ``OrtInferSession._get_ep_list()`` returns CPU (+CUDA/DirectML when asked).
    We wrap it so CoreML comes first and CPU stays last — ONNX Runtime then
    automatically falls back to CPU for any subgraph CoreML cannot handle, so
    an unsupported op degrades to the old speed instead of failing.
    """
    original = getattr(session_cls, "_get_ep_list", None)
    if original is None:
        return False
    if getattr(original, "_coreml_patched", False):
        return True

    def _get_ep_list(self):  # type: ignore[no-untyped-def]
        try:
            ep_list = list(original(self))
        except Exception:
            ep_list = [(_CPU_EP, {})]
        # Drop any pre-existing CoreML entry so ordering is deterministic.
        ep_list = [ep for ep in ep_list
                   if (ep[0] if isinstance(ep, (list, tuple)) else ep) != _COREML_EP]
        ep_list.insert(0, (_COREML_EP, dict(_COREML_EP_OPTIONS)))
        return ep_list

    _get_ep_list._coreml_patched = True          # type: ignore[attr-defined]
    _get_ep_list._coreml_original = original     # type: ignore[attr-defined]
    session_cls._get_ep_list = _get_ep_list      # type: ignore[assignment]
    return True


def _unpatch_coreml_ep(session_cls) -> None:
    """Restore the stock EP list (used when CoreML session creation fails)."""
    patched = getattr(session_cls, "_get_ep_list", None)
    original = getattr(patched, "_coreml_original", None)
    if original is not None:
        session_cls._get_ep_list = original      # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

# Detector input resolution.  RapidOCR's default (736, limit_type="min")
# downsamples an A4 page rendered at 200dpi to ~736px short side — small
# glyphs get destroyed.  1024 keeps the text legible at a modest cost.
_DET_LIMIT_SIDE_LEN = 1024


def _engine_kwargs(**overrides) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "det_limit_side_len": _DET_LIMIT_SIDE_LEN,
        "det_limit_type": "min",
    }
    kwargs.update(overrides)
    return kwargs


def create_ocr_engine(use_coreml: bool = True, **kwargs):
    """Build a RapidOCR engine, preferring CoreML EP on Apple Silicon.

    Falls back to the stock CPU engine if CoreML is unavailable *or* if the
    CoreML session fails to build (some ops are unsupported on certain ORT
    builds) — the patch is rolled back before retrying so a later construction
    is not poisoned by the failed attempt.
    """
    from rapidocr_onnxruntime import RapidOCR

    engine_kwargs = _engine_kwargs(**kwargs)
    if not use_coreml:
        logger.info("OCR: CoreML EP 已按配置关闭，使用默认 EP")
        return RapidOCR(**engine_kwargs)

    usable, reason = coreml_reason()
    if not usable:
        logger.info("OCR: 未启用 CoreML EP — %s", reason)
        return RapidOCR(**engine_kwargs)

    with _patch_lock:
        session_cls = _ort_infer_session_class()
        if session_cls is None:
            logger.warning(
                "OCR: 找不到 rapidocr 的 OrtInferSession，无法注入 CoreML EP，回退默认 EP"
            )
            return RapidOCR(**engine_kwargs)
        if not _patch_coreml_ep(session_cls):
            return RapidOCR(**engine_kwargs)
        try:
            engine = RapidOCR(**engine_kwargs)
            logger.info("OCR: 已启用 CoreML EP（MLComputeUnits=ALL，CPU+GPU+ANE）— %s", reason)
            return engine
        except Exception as e:
            # CoreML session creation failed (unsupported op / MLProgram
            # unavailable).  Roll the patch back and retry on CPU so OCR still
            # works — a failed EP must never break ingestion.
            _unpatch_coreml_ep(session_cls)
            logger.warning("OCR: CoreML EP 初始化失败，回退 CPU EP: %s", e)
            return RapidOCR(**engine_kwargs)


# ---------------------------------------------------------------------------
# Image pre-processing / normalisation
# ---------------------------------------------------------------------------

# Target short side (px) for a page before it reaches the detector.  Below
# this the text is upscaled (small print becomes legible); far above it we
# downscale so a 600dpi scan doesn't take minutes per page.
_TARGET_SHORT_SIDE = 1600
_MAX_LONG_SIDE = 4000
_MIN_SHORT_SIDE = 900


def render_dpi_for_page(page, target_short_side: int = _TARGET_SHORT_SIDE) -> int:
    """Pick a render DPI that normalises the page's short side to ~target.

    PDFs ship at wildly different physical sizes (A4 vs letter vs a tiny
    cropped figure).  Rendering everything at a fixed 200dpi makes the
    detector see very different glyph sizes.  Solving for DPI first means the
    detector always works on roughly the same scale.
    """
    try:
        rect = page.rect
        width_in = max(float(rect.width), 1e-6) / 72.0
        height_in = max(float(rect.height), 1e-6) / 72.0
        short_in = min(width_in, height_in)
        dpi = int(round(target_short_side / short_in))
    except Exception:
        return 200
    # Clamp: below 150 the render itself loses strokes; above 400 the pixmap
    # (and OCR time) explodes for no recognition gain.
    return max(150, min(dpi, 400))


def preprocess_for_ocr(
    img: "Any",
    enhance: bool = True,
    binarize: bool = False,
) -> "Any":
    """Normalise + enhance a BGR image before OCR.

    Steps (all best-effort; any failure returns the input untouched):

    1. **Scale normalisation** — short side < ``_MIN_SHORT_SIDE`` is upscaled,
       long side > ``_MAX_LONG_SIDE`` is downscaled, so the detector always
       sees text at a sane size.
    2. **Grayscale + CLAHE** — adaptive histogram equalisation lifts faint
       scans and evens out lighting gradients that break text detection.
    3. **Unsharp mask** — restores edge definition lost to the resample.
    4. *(optional)* **Adaptive binarisation** — helps clean black-and-white
       scans, but harms screenshots / colour figures, hence off by default.

    Returns a **BGR** ``ndarray`` (RapidOCR treats a 3-channel ndarray as BGR
    and does no conversion).
    """
    if img is None:
        return img
    try:
        import cv2
        import numpy as np
    except Exception:
        return img
    try:
        arr = np.asarray(img)
        if arr.ndim == 2:
            work = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            work = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            work = arr
        else:
            return img
        if work.size == 0:
            return img

        # ---- 1. scale normalisation ----
        h, w = work.shape[:2]
        short, long_ = min(h, w), max(h, w)
        scale = 1.0
        if short < _MIN_SHORT_SIDE:
            scale = _MIN_SHORT_SIDE / float(short)
        if long_ * scale > _MAX_LONG_SIDE:
            scale = _MAX_LONG_SIDE / float(long_)
        if abs(scale - 1.0) > 0.01:
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=interp)

        if not enhance:
            return work

        # ---- 2. grayscale + CLAHE ----
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            # CLAHE unavailable on very old OpenCV builds.
            pass

        if binarize:
            # Adaptive threshold: only correct for clean B/W scans.
            try:
                gray = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 10,
                )
            except Exception:
                pass

        # ---- 3. unsharp mask ----
        try:
            blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
            sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        except Exception:
            sharpened = gray

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        logger.debug("OCR 预处理失败，使用原图: %s", e)
        return img


def decode_image(img_bytes: bytes) -> "Any":
    """PNG/JPEG bytes → BGR ndarray (``None`` when decoding fails)."""
    try:
        import cv2
        import numpy as np
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return arr
    except Exception as e:
        logger.debug("OCR 图像解码失败: %s", e)
        return None
