"""OvisOCR3 多模态 OCR 引擎——docling / marker 之外的第三方案。

定位（2026-10 需求）：
  * PDF / 图像类文件的**兜底解析方案**，取代原先 parsers.py 的
    PyMuPDF + ocrmac/rapidocr 组合；
  * 同时作为 ``rag_pdf_engine="ovis"`` 时可显式选中的主引擎。

管线（PyMuPDF 取图 + OvisOCR3 VLM 转 markdown）：
  * 文本 PDF：PyMuPDF 直取每页文本；页面检测到表格时，把表格区域裁剪
    渲染成图 → 归一化 → 送 VLM 转 markdown → 填回原表格所在位置；
  * 图片 PDF（无文本层）：逐页渲染整页 → 归一化 → 送 VLM 转 markdown；
  * 图像文件（png/jpg/...）：整图归一化后送 VLM。

模型默认 ``models/OvisOCR3-MLX-4bit-mixed46``（Qwen3.5-VL 架构的 853M OCR
微调模型，4bit MLX 量化，实测比 docling OCR 路径快 2-3 倍、内存 0.86GB）。
后端下拉提供 MLX-VLM / ONNX / GGUF 三选：
  * mlx   → MLX-VLM 权重（默认 OvisOCR3-MLX-4bit-mixed46）
  * onnx  → ONNX 多模态导出（含视觉+文本联合图），用 rag_ocr_model_path
  * gguf  → llama.cpp GGUF 多模态模型，必须配 mmproj 投影文件才能读图
三者共用同一套 PDF/图像 → markdown 管线；缺权重时加载阶段给出明确错误。
"""
from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# 渲染 DPI：与 parsers._OCR_DPI(150) 对齐——归一化流程与引擎基准测试
# 都用它标定；表格裁剪区域用更高 DPI（小区域需要更多像素细节）。
_PAGE_OCR_DPI = 150
_TABLE_RENDER_DPI = 200
_MAX_TOKENS = 2048
# 框线投影检测的渲染参数（与 docling_parser 的分类常量同值，2026-10 实测标定：
# <150 DPI 的降采样会把目录点线融成连续带导致误判）。
_CLASSIFY_DPI = 150
_NONWHITE_LEVEL = 250
_MAX_LINE_THICK = 5

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 默认模型（配置窗口的默认值也用它，改这里即可全链路生效）
DEFAULT_OCR_BACKEND = "mlx"
DEFAULT_OCR_MODEL = "OvisOCR3-MLX-4bit-mixed46"
DEFAULT_OCR_MODEL_PATH = "models/OvisOCR3-MLX-4bit-mixed46"

# VLM 提示词：实测该模型被 OCR 微调「焊死」了输出格式（要求 markdown
# 管道表格仍输出 HTML、要求省略页眉页码照旧转写），格式归一化全部在代码
# 里做（见 _html_tables_to_markdown），prompt 只负责「转写成 markdown」。
_PROMPT = "Transcribe this document to markdown."


# ---------------------------------------------------------------------------
# 引擎单例（惰性加载 / 线程安全 / 可释放）
# ---------------------------------------------------------------------------
_engine = None            # 已加载的 OCR 引擎 impl（含 .kind 与 .generate）
_engine_kind = None
_engine_lock = threading.RLock()
_infer_lock = threading.RLock()   # mlx generate / llama 推理非线程安全：串行化
_engine_failed = False    # 加载失败标记：避免每页重试刷屏
_config = {
    "backend": DEFAULT_OCR_BACKEND,
    "model": DEFAULT_OCR_MODEL,
    "path": DEFAULT_OCR_MODEL_PATH,
    "gguf_path": "",
    "mmproj_path": "",
    "tokenizer_path": "",
    "vlm_handler": "mtmd",
    "chat_format": "",
}


class OvisUnavailable(RuntimeError):
    """引擎不可用（未安装依赖 / 模型缺失 / 后端未实现）。"""


def configure(backend: str = "", model: str = "", path: str = "",
              gguf_path: str = "", mmproj_path: str = "",
              tokenizer_path: str = "",
              vlm_handler: str = "", chat_format: str = "") -> None:
    """RAGEngine.__init__ 调用：记录配置。已加载的引擎不受影响。"""
    with _engine_lock:
        if backend:
            _config["backend"] = backend
        if model:
            _config["model"] = model
        if path:
            _config["path"] = path
        if gguf_path:
            _config["gguf_path"] = gguf_path
        if mmproj_path:
            _config["mmproj_path"] = mmproj_path
        if tokenizer_path:
            _config["tokenizer_path"] = tokenizer_path
        if vlm_handler:
            _config["vlm_handler"] = vlm_handler
        if chat_format is not None:
            _config["chat_format"] = chat_format


def _app_root() -> Path:
    # src/agent/rag/ovis_ocr.py → 上溯四级到仓库根
    return Path(__file__).resolve().parents[3]


def _resolve_model_path() -> str:
    raw = _config["path"] or _config["model"] or DEFAULT_OCR_MODEL_PATH
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not path.is_absolute():
        # 相对路径按应用根目录（main.py 所在目录）解析，与其它本地模型一致
        path = _app_root() / path
    return str(path)


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 各后端的引擎 impl：统一接口 `generate(png_path) -> str`（返回 VLM 原始文本，
# HTML→markdown 归一在 ocr_image_to_markdown 里统一做）。
# ---------------------------------------------------------------------------
class _MLXOCREngine:
    kind = "mlx"

    def __init__(self, model, processor, prompt):
        self._model = model
        self._processor = processor
        self._prompt = prompt

    def generate(self, png_path: str) -> str:
        from mlx_vlm import generate

        with _infer_lock:
            out = generate(self._model, self._processor, self._prompt,
                           image=png_path, max_tokens=_MAX_TOKENS, verbose=False)
        return out.text if hasattr(out, "text") else str(out)


class _ONNXOCREngine:
    kind = "onnx"

    def __init__(self, session, tokenizer, processor, prompt_text):
        self._session = session
        self._tokenizer = tokenizer
        self._processor = processor
        self._prompt_text = prompt_text

    def _build_inputs(self, png_path):
        import numpy as np
        from PIL import Image

        images = [Image.open(png_path).convert("RGB")]
        if self._processor is not None:
            batch = self._processor(text=self._prompt_text, images=images,
                                    return_tensors="np")
        else:
            batch = self._tokenizer(self._prompt_text, return_tensors="np")
        return {k: np.asarray(v) for k, v in dict(batch).items()}

    def _feeds(self, encoded, input_ids):
        import numpy as np

        input_names = {it.name: it for it in self._session.get_inputs()}
        feeds: dict = {}
        for name, spec in input_names.items():
            if name == "input_ids":
                value = input_ids
            elif name == "attention_mask":
                value = np.ones_like(input_ids, dtype=np.int64)
            elif name == "position_ids":
                if name in encoded:
                    base = np.asarray(encoded[name], dtype=np.int64)
                    missing = input_ids.shape[1] - base.shape[-1]
                    if missing > 0:
                        tail = base[..., -1:] + np.arange(1, missing + 1, dtype=np.int64)
                        base = np.concatenate((base, tail), axis=-1)
                    value = base[..., :input_ids.shape[1]]
                else:
                    declared = list(spec.shape or [])
                    if len(declared) == 3:
                        raise OvisUnavailable(
                            "该 ONNX VLM 需要显式三维 M-RoPE position_ids；"
                            "processor 未生成此输入，不能用普通二维位置替代"
                        )
                    value = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
            elif name == "token_type_ids":
                value = np.zeros_like(input_ids, dtype=np.int64)
            elif name in encoded:
                value = encoded[name]
            else:
                raise OvisUnavailable(
                    f"ONNX 模型需要未能自动构造的输入 '{name}' ({spec.shape})；"
                    "请使用与导出图匹配的专用 processor"
                )
            feeds[name] = np.asarray(value)
        return feeds

    def generate(self, png_path: str) -> str:
        import numpy as np

        encoded = self._build_inputs(png_path)
        if "input_ids" not in encoded:
            raise OvisUnavailable("ONNX processor 未返回 input_ids")
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        generated: list = []
        eos = self._tokenizer.eos_token_id
        eos_ids = (set(eos) if isinstance(eos, list)
                   else {eos}) if eos is not None else set()
        for _ in range(_MAX_TOKENS):
            outputs = self._session.run(None, self._feeds(encoded, input_ids))
            logits = np.asarray(outputs[0])
            if logits.ndim < 3:
                raise OvisUnavailable("ONNX 输出不是生成模型 logits")
            nxt = int(np.argmax(logits[0, -1]))
            if nxt in eos_ids:
                break
            generated.append(nxt)
            input_ids = np.concatenate(
                (input_ids, np.asarray([[nxt]], dtype=np.int64)), axis=1
            )
        return self._tokenizer.decode(generated, skip_special_tokens=True)


class _GGUFOCREngine:
    kind = "gguf"

    def __init__(self, backend):
        self._backend = backend

    def generate(self, png_path: str) -> str:
        # 图像以 [image:path] 标记喂给多模态 GGUF（llama.cpp chat_handler 解析）。
        messages = [{"role": "user", "content": f"[image:{png_path}] {_PROMPT}"}]
        parts: list = []
        with _infer_lock:
            for ev in self._backend.chat_stream(messages, temperature=0.2):
                if getattr(ev, "type", None) == "content":
                    parts.append(ev.content or "")
        return "".join(parts)


class _GGUFConfig:
    """把 OCR 的 GGUF/mmproj/handler/format 适配成 LlamaCppBackend 期望的
    config 字段子集（llm_vision_projection_path / llm_vlm_handler /
    llm_chat_format + 推理超参）。"""

    def __init__(self):
        self.llm_vision_projection_path = _config.get("mmproj_path") or ""
        self.llm_vlm_handler = _config.get("vlm_handler") or "mtmd"
        self.llm_chat_format = _config.get("chat_format") or ""
        self.local_gpu_layers = -1
        self.local_llm_n_ctx = 4096
        self.max_context_tokens = 4096
        self.local_model_batch_size = 512
        self.local_model_threads = 4
        self.top_p = 0.95
        self.top_k = 20
        self.min_p = 0.05
        self.repetition_penalty = 1.0
        self.max_tokens = _MAX_TOKENS


def _build_onnx_prompt(tokenizer, processor) -> str:
    """生成 ONNX VLM 的起始文本（带 generation prompt）。无模板则退化为原 prompt。"""
    owner = processor or tokenizer
    try:
        return owner.apply_chat_template(
            [{"role": "user", "content": _PROMPT}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return _PROMPT


def _load_mlx():
    try:
        from mlx_vlm import load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except Exception as e:
        raise OvisUnavailable(
            f"mlx-vlm 未安装，OvisOCR3 不可用: {e}（pip install mlx-vlm）"
        ) from e
    model_path = _resolve_model_path()
    if not Path(model_path).is_dir():
        raise OvisUnavailable(
            f"OvisOCR3 模型目录不存在: {model_path}"
            f"（配置的 rag_ocr_model_path）"
        )
    model, processor = load(model_path)
    prompt = apply_chat_template(
        processor, load_config(model_path), _PROMPT, num_images=1
    )
    # MLX 内存上限：与嵌入/重排两个 MLX 入口一致，防止同步期 phys_footprint 暴涨。
    from .model_backends import _cap_mlx_memory_limits
    _cap_mlx_memory_limits()
    logger.info("OvisOCR3 引擎加载完成: %s（MLX-VLM）", model_path)
    return _MLXOCREngine(model, processor, prompt)


def _load_onnx():
    try:
        import onnxruntime as ort
        from transformers import AutoProcessor, AutoTokenizer
    except ImportError as e:
        raise OvisUnavailable(
            f"ONNX OCR 需要 onnxruntime 与 transformers: {e}"
        ) from e
    model_path = _resolve_model_path()
    if not (Path(model_path).is_file() or Path(model_path).is_dir()):
        raise OvisUnavailable(f"ONNX OCR 模型路径不存在: {model_path}")
    model_dir = Path(model_path) if Path(model_path).is_dir() else Path(model_path).parent
    if Path(model_path).is_file():
        model_file = Path(model_path)
    else:
        cands = sorted(Path(model_path).glob("*.onnx")) + \
                sorted(Path(model_path).glob("onnx/*.onnx"))
        if not cands:
            raise OvisUnavailable(f"目录中未找到 ONNX 文件: {model_path}")
        model_file = cands[0]
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(_config.get("onnx_threads") or 4))
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_file), sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False)
    except Exception:
        processor = None
    # tokenizer 目录：显式配置优先（UI「tokenizer 路径」），缺省回退模型所在
    # 目录——.onnx 单文件分发通常不带 tokenizer，必须由用户指定目录。
    tok_dir = _config.get("tokenizer_path") or str(model_dir)
    if not Path(tok_dir).is_dir():
        logger.warning("ONNX OCR 配置的 tokenizer 目录不存在，回退模型目录: %s", tok_dir)
        tok_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_dir, local_files_only=True, trust_remote_code=False)
    logger.info("ONNX OCR 引擎加载完成: %s（CPU）", model_file)
    return _ONNXOCREngine(session, tokenizer, processor,
                          _build_onnx_prompt(tokenizer, processor))


def _load_gguf():
    try:
        from ..model_backends import LlamaCppBackend
    except ImportError as e:
        raise OvisUnavailable(f"GGUF OCR 需要 agent.model_backends: {e}") from e
    gguf_path = (_config.get("gguf_path") or _config.get("path")
                 or _config.get("model"))
    if not gguf_path:
        raise OvisUnavailable("GGUF OCR 未配置 GGUF 模型路径 (rag_ocr_gguf_path)")
    try:
        backend = LlamaCppBackend(str(gguf_path), _GGUFConfig())
    except Exception as e:
        raise OvisUnavailable(f"GGUF OCR 模型加载失败: {gguf_path}: {e}") from e
    logger.info("GGUF OCR 引擎加载完成: %s", gguf_path)
    return _GGUFOCREngine(backend)


def _get_engine():
    """惰性加载 OCR 引擎（按 backend 分派 mlx/onnx/gguf）；失败抛 OvisUnavailable。"""
    global _engine, _engine_failed, _engine_kind
    with _engine_lock:
        if _engine is not None:
            return _engine
        if _engine_failed:
            raise OvisUnavailable("OCR 引擎此前加载失败，跳过（见上方日志）")
        backend = _config["backend"]
        try:
            if backend == "mlx":
                impl = _load_mlx()
            elif backend == "onnx":
                impl = _load_onnx()
            elif backend == "gguf":
                impl = _load_gguf()
            else:
                raise OvisUnavailable(f"不支持的 OCR 后端: {backend}")
        except OvisUnavailable:
            raise
        except Exception as e:
            _engine_failed = True
            raise OvisUnavailable(f"OCR 引擎加载失败 ({backend}): {e}") from e
        _engine = impl
        _engine_kind = backend
        return _engine


def release_engine() -> None:
    """释放引擎单例，归还显存/统一内存。RAGEngine.close() 调用。"""
    global _engine, _engine_failed, _engine_kind
    # 非阻塞拿推理锁：有 worker 正在 generate 时不能释放——置 None 只是
    # 换掉单例引用，在途推理仍抱着旧模型，紧接着的 _get_engine() 会再造
    # 一份 → 两代内存并存（与 docling/marker 转换器两代并存的竞态同源）。
    # 拿不到锁就跳过本次释放；close() 在下次同步结束会再调，届时回收。
    if not _infer_lock.acquire(blocking=False):
        logger.info("OCR 推理进行中，本次引擎释放跳过（留待下次回收）")
        return
    try:
        with _engine_lock:
            if _engine is not None:
                _engine = None
                _engine_kind = None
                logger.info("OCR 引擎已释放")
            # 释放后允许下次重新尝试加载（用户可能刚修好模型目录）
            _engine_failed = False
            # MLX 缓存冲刷**必须无条件执行**：本函数是 RAGEngine.close() 每轮
            # 同步结束的收尾入口。旧实现把 mx.clear_cache() 放在上面的 if 里，
            # 于是「上一轮已释放 / 从未加载成功」时直接跳过 —— 第二轮以后的同步
            # 结束就再也冲刷不到 MLX 缓存，分配器按形状单调增长，表现为
            # 「同步跑一会就被系统杀掉」（与 vector_store 里 mx.clear_cache 的
            # 长注释是同一类问题）。
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass
    finally:
        _infer_lock.release()


# ---------------------------------------------------------------------------
# HTML 表格 → markdown 管道表格
# ---------------------------------------------------------------------------
_HTML_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_HTML_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_ANY_TAG = re.compile(r"<[^>]+>")


def _clean_cell(html: str) -> str:
    import html as _html_mod

    text = re.sub(r"<br\s*/?>", " ", html, flags=re.I)  # 单元格内换行→空格
    text = _ANY_TAG.sub("", text)
    text = _html_mod.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", r"\|")


def _html_tables_to_markdown(text: str) -> str:
    """把 VLM 输出的 HTML 表格转成 markdown 管道表格。

    为什么必须代码归一：该模型被 OCR 微调把输出格式焊死了——实测三种
    prompt 措辞（含明令 "HTML is forbidden"）都照旧输出 ``<table>``，
    prompt 服从干不过训练分布。转换是确定性的，零 token 开销。
    """

    def _conv(m: re.Match) -> str:
        rows: List[List[str]] = []
        for row_html in _HTML_ROW.findall(m.group(0)):
            cells = [_clean_cell(c) for c in _HTML_CELL.findall(row_html)]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
        lines.extend("| " + " | ".join(r) + " |" for r in rows[1:])
        return "\n".join(lines)

    return re.sub(r"<table[^>]*>.*?</table>", _conv, text, flags=re.S | re.I)


# ---------------------------------------------------------------------------
# 图像 → markdown
# ---------------------------------------------------------------------------
def ocr_image_to_markdown(img) -> str:
    """归一化后的灰度图（uint8 2D）→ markdown 文本。引擎不可用抛 OvisUnavailable。

    落临时 PNG 交给后端引擎（mlx_vlm / onnxruntime / llama.cpp 都吃文件路径），
    各后端的 generate 实现负责实际推理，这里统一做 HTML→markdown 归一。
    """
    import tempfile

    import numpy as np
    from PIL import Image

    from .parsers import _ocr_normalize

    engine = _get_engine()

    arr = np.asarray(img)
    if arr.ndim == 3:  # 容错：调用方传入彩色图
        arr = arr.mean(axis=2).astype(np.uint8)
    arr = _ocr_normalize(arr)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Image.fromarray(arr).convert("RGB").save(tmp_path)
        # 推理非线程安全（各后端都抱一块设备/GPU），4 个 ingest worker 串行化；
        # 模型本身就在一块设备上，串行不损失吞吐。
        text = engine.generate(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return _html_tables_to_markdown(text).strip()


# ---------------------------------------------------------------------------
# PDF / 图像文件 → markdown
# ---------------------------------------------------------------------------
def _page_blocks(page) -> list:
    """页面文本块 [(x0, y0, x1, y1, text), ...]，按阅读序（先 y 后 x）。"""
    try:
        raw = page.get_text("blocks")
    except Exception:
        return []
    blocks = []
    for b in raw:
        if len(b) >= 5 and b[6] == 0 and b[4] and b[4].strip():  # 0 = 文本块
            blocks.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                           b[4].strip()))
    blocks.sort(key=lambda b: (round(b[1], 1), b[0]))
    return blocks


def _find_tables(page) -> list:
    """PyMuPDF 内置表格检测（1.23+），返回表格 bbox 列表。

    对**有竖线的网格表格**可靠；但横线型表格（中文书常见：只有水平框线）
    它只能框出表头行（实测 bbox 高度仅 29pt，数据行全在框外）——那种表格
    由 _ruled_table_bboxes 的框线投影补上。
    """
    try:
        finder = page.find_tables()
    except Exception:
        return []
    tables = []
    for t in finder.tables:
        bbox = getattr(t, "bbox", None)
        if bbox and (t.row_count > 1 or t.col_count > 1):
            tables.append(tuple(float(v) for v in bbox))
    return tables


def _vector_table_lines(page) -> int:
    """页面原生矢量长直线数量（零渲染，用于文本页表格预筛）。

    文本层页面的表格框线一定是 PDF 矢量绘制，数长直线即可判断「这页有
    没有表格」，不需要渲染；与 docling_parser._vector_table_lines 同算法
    （>=页面 30% 长宽的长直线），本地自包含避免跨模块私有引用。
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return 0
    count = 0
    w, h = float(page.rect.width), float(page.rect.height)
    for d in drawings or ():
        for item in d.get("items", ()):
            if item and item[0] == "l":
                p1, p2 = item[1], item[2]
                if (abs(p1.x - p2.x) > w * 0.3
                        or abs(p1.y - p2.y) > h * 0.3):
                    count += 1
    return count


def _longest_run_span(row) -> Tuple[int, int]:
    """一维布尔数组最长连续 True 段的 (起, 止)。"""
    import numpy as np

    if not row.any():
        return (0, 0)
    idx = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
    starts, ends = idx[::2], idx[1::2]
    k = int(np.argmax(ends - starts))
    return (int(starts[k]), int(ends[k]))


def _ruled_table_bboxes(page) -> list:
    """框线投影定位「横线型」表格区域（中文书常见样式，find_tables 的盲区）。

    原理与 docling_parser._page_has_ruled_table 同源：150 DPI 渲染后，
    通长水平实线在行投影上是尖峰；把间距一致（表格行结构）的实线聚类，
    区域 = 首末框线之间 + 框线水平跨度并集。装饰性横线间距杂乱，聚类时
    自然被拆成小簇并被「>=4 条线」下限过滤。
    返回 PDF 点坐标 bbox 列表。
    """
    import numpy as np

    try:
        pix = page.get_pixmap(dpi=_CLASSIFY_DPI, colorspace=_fitz_gray())
    except Exception:
        return []
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width)
    nonwhite = arr < _NONWHITE_LEVEL
    h_frac = nonwhite.mean(axis=1)

    def _runs(mask):
        if not len(mask):
            return []
        idx = np.flatnonzero(np.diff(np.concatenate(
            ([0], mask.astype(np.int8), [0]))))
        return list(zip(idx[::2].tolist(), idx[1::2].tolist()))

    # 候选实线：(y, x0, x1)——细带 + 中线最长连续段 >=25% 页宽
    rules = []
    for s, e in _runs(h_frac > 0.25):
        if e - s > _MAX_LINE_THICK:
            continue
        mid = (s + e) // 2
        x0, x1 = _longest_run_span(nonwhite[mid, :])
        if (x1 - x0) / arr.shape[1] >= 0.25:
            rules.append((mid, x0, x1))
    if len(rules) < 4:
        return []

    # 聚类：相邻线间距与簇内中位间距一致（容差 35%）才并入
    clusters: List[List[Tuple[int, int, int]]] = [[rules[0]]]
    for r in rules[1:]:
        cl = clusters[-1]
        gaps = [b[0] - a[0] for a, b in zip(cl, cl[1:])]
        med = sorted(gaps)[len(gaps) // 2] if gaps else (r[0] - cl[-1][0])
        if abs((r[0] - cl[-1][0]) - med) <= 0.35 * med:
            cl.append(r)
        else:
            clusters.append([r])

    def _header_band_above(first_y: int, gap_px: float,
                           x0: int, x1: int) -> Optional[int]:
        """首条框线上方的「表头填充色带」顶部（px）。

        表格头部常见深色/彩色填充带：厚（>5px）、非白占比高，不是「线」
        所以进不了规则聚类。注意它与首条框线之间**隔着第一行数据**（带底
        → 行距 → 首线），不能按连续性找，而是在首线上方 2 个行距的窗口里
        搜厚实非白带，要求带底与首线距离 ∈ [0.4, 1.8] 个行距（≈表头行
        的高度），且 x 跨度与框线重叠 >50%（防止并入章节标题色带）。
        """
        lo = max(0, int(first_y - gap_px * 2.2))
        hi = int(first_y - gap_px * 0.4)
        if hi <= lo:
            return None
        rows = nonwhite[lo:hi, :].mean(axis=1)

        def _runs(mask):
            if not len(mask):
                return []
            idx = np.flatnonzero(np.diff(np.concatenate(
                ([0], mask.astype(np.int8), [0]))))
            return list(zip(idx[::2].tolist(), idx[1::2].tolist()))

        runs = [r for r in _runs(rows > 0.5)
                if r[1] - r[0] > _MAX_LINE_THICK]
        if not runs:
            return None
        band_top, band_bottom = lo + runs[-1][0], lo + runs[-1][1]
        if not (0.4 * gap_px <= first_y - band_bottom <= 1.8 * gap_px):
            return None
        band = nonwhite[band_top:first_y, :]
        cols = np.flatnonzero(band.mean(axis=0) > 0.3)
        if len(cols) and (cols[-1] - cols[0]) > 0.5 * (x1 - x0):
            return band_top
        return None

    scale = 72.0 / _CLASSIFY_DPI  # px → pt
    bboxes = []
    for cl in clusters:
        if len(cl) < 4:  # 少于 4 条线（3 个行距）视为装饰线，不算表格
            continue
        y1 = cl[-1][0] + 2
        gaps = [b[0] - a[0] for a, b in zip(cl, cl[1:])]
        med_gap = sorted(gaps)[len(gaps) // 2]
        band_top = _header_band_above(cl[0][0], med_gap,
                                      min(r[1] for r in cl),
                                      max(r[2] for r in cl))
        y0 = (band_top if band_top is not None else cl[0][0]) - 2
        x0 = min(r[1] for r in cl)
        x1 = max(r[2] for r in cl)
        bboxes.append((x0 * scale, y0 * scale, x1 * scale, y1 * scale))
    return bboxes


def _table_regions(page) -> list:
    """合并两路表格检测：find_tables（网格型）+ 框线投影（横线型）。

    重叠去重：像素路能框出整表（find_tables 只框表头行），故重叠时保留
    更大的 bbox。
    """
    regions = list(_find_tables(page))
    for bbox in _ruled_table_bboxes(page):
        overlap = any(_bbox_overlap(bbox, r) for r in regions)
        if not overlap:
            regions.append(bbox)
        else:
            # 重叠：用更大的替换小的（通常是像素路整表 vs find_tables 表头行）
            bigger = [r for r in regions if _bbox_overlap(bbox, r)]
            max_r = max(bigger, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > \
                    (max_r[2] - max_r[0]) * (max_r[3] - max_r[1]):
                regions[regions.index(max_r)] = bbox
    return regions


def _bbox_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _intersects(block, table) -> bool:
    """文本块与表格区域是否相交（块中心落在表格内即算）。"""
    cx = (block[0] + block[2]) / 2
    cy = (block[1] + block[3]) / 2
    return (table[0] <= cx <= table[2]) and (table[1] <= cy <= table[3])


def _table_region_to_markdown(page, bbox) -> str:
    """裁剪表格区域 → 高 DPI 渲染 → 归一化 → VLM → markdown。"""
    import numpy as np

    clip = _fitz_rect(bbox)
    pix = page.get_pixmap(dpi=_TABLE_RENDER_DPI, clip=clip,
                          colorspace=_fitz_gray())
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width)
    return ocr_image_to_markdown(arr)


def _text_page_markdown(page) -> str:
    """文本页：直取文本；有表格时把表格区域换成 VLM 转的 markdown。

    填回策略：按阅读序遍历文本块，落在表格区域内的块跳过，在「首个被
    跳过块」的位置插入该表格的 markdown——保持表格出现在原文的上下文
    位置，而不是简单堆到页尾。

    性能（2026-10 审查修复）：旧实现无条件做 150 DPI 整页渲染 + 像素
    投影——纯文本页（书的大多数）这次渲染必然被丢弃，500 页书 = 500 次
    整页渲染（A4@150DPI ≈ 8.7MB 缓冲/页）。现在先用**零渲染**的矢量线
    预筛：文本层页面上的表格框线一定是矢量绘制（get_drawings），
    >=4 条长直线才值得进像素投影定位；没有矢量线的页面直接返回纯文本
    （无线表的纯文字排版页，像素投影也找不到东西）。
    """
    if _vector_table_lines(page) < 4:
        return page.get_text("text").strip()

    tables = _table_regions(page)
    if not tables:
        return page.get_text("text").strip()

    blocks = _page_blocks(page)
    if not blocks:
        return page.get_text("text").strip()

    parts: List[str] = []
    emitted: set = set()
    for b in blocks:
        hit = next((t for t in tables if _intersects(b, t)), None)
        if hit is None:
            parts.append(b[4])
            continue
        if hit in emitted:
            continue  # 同一表格的后续文本块：已被 VLM 结果取代
        emitted.add(hit)
        try:
            table_md = _table_region_to_markdown(page, hit)
        except Exception as e:
            logger.warning("表格区域识别失败，保留原文本: %s", e)
            table_md = ""
        if table_md:
            parts.append(table_md)
        else:
            # VLM 失败：退回该表格区域的原始文本块，不丢内容
            parts.extend(
                blk[4] for blk in blocks if _intersects(blk, hit)
            )
    return "\n\n".join(p for p in parts if p.strip())


def _fitz_gray():
    import fitz
    return fitz.csGRAY


def _fitz_rect(bbox):
    import fitz
    return fitz.Rect(*bbox)


def _extract_pdf(path: Path) -> Optional[str]:
    """PDF 逐页处理：文本页直取（含表格填回），图片页整页送 VLM。"""
    try:
        import fitz
    except ImportError as e:
        raise OvisUnavailable(f"PyMuPDF 未安装: {e}") from e

    doc = fitz.open(str(path))
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError("PDF 已加密且无法用空密码解锁")
        parts: List[str] = []
        for page_num in range(len(doc)):
            if page_num and page_num % 50 == 0:
                try:
                    fitz.TOOLS.store_shrink(100)
                except Exception:
                    pass
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                parts.append(_text_page_markdown(page))
                continue
            # 无文本层 → 整页渲染送 VLM（图片版页面）
            pix = page.get_pixmap(dpi=_PAGE_OCR_DPI, colorspace=fitz.csGRAY)
            try:
                import numpy as np

                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width)
                if float(arr.std()) < 1.0:
                    continue  # 纯色空白页：VLM 必然空转，跳过
                md = ocr_image_to_markdown(arr)
            finally:
                pix = None
            if md:
                parts.append(md)
    finally:
        doc.close()
        # 收尾冲一次 MuPDF 内部对象缓存：不足 50 页的文档（页内循环的
        # 周期性 shrink 一次都不会触发）也会残留渲染/解析对象。
        try:
            fitz.TOOLS.store_shrink(100)
        except Exception:
            pass
    return "\n\n".join(p for p in parts if p.strip()) or None


def extract_markdown(filepath: str) -> Optional[str]:
    """PDF / 图像文件 → markdown。非 PDF/图像格式返回 None（调用方走其它路径）。"""
    path = Path(filepath)
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        if suffix in _IMAGE_EXTS:
            import numpy as np
            from PIL import Image

            with Image.open(str(path)) as im:
                arr = np.asarray(im.convert("L"))
            md = ocr_image_to_markdown(arr)
            return md or None
    except OvisUnavailable:
        raise  # 引擎缺失是配置问题，让路由层决定回退
    except Exception as e:
        logger.warning("OvisOCR3 解析失败 %s: %s", path.name, e)
        return None
    return None


def is_ovis_available() -> bool:
    """引擎是否可能可用（按 backend 快检依赖 + 权重，不加载模型）。

    mlx  → 需要 mlx_vlm 且模型目录存在
    onnx → 需要 onnxruntime 且模型文件/目录存在
    gguf → 需要 llama_cpp 且 GGUF 模型文件存在
    """
    backend = _config["backend"]
    if backend == "mlx":
        if not _importable("mlx_vlm"):
            return False
        return Path(_resolve_model_path()).is_dir()
    if backend == "onnx":
        if not _importable("onnxruntime"):
            return False
        p = _resolve_model_path()
        return Path(p).is_file() or Path(p).is_dir()
    if backend == "gguf":
        if not _importable("llama_cpp"):
            return False
        gp = _config.get("gguf_path") or _config.get("path") or _config.get("model")
        return bool(gp) and Path(gp).is_file()
    return False
