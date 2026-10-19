"""Local and remote language-model backends used by the Agent.

Optional runtimes are imported lazily so the default OpenAI-compatible mode
continues to work without MLX, ONNX Runtime, Transformers, or llama.cpp.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol

from .llm_client import LLMClient, LLMError, StreamEvent

logger = logging.getLogger(__name__)

LLM_BACKENDS = ("openai", "mlx_lm", "mlx_vlm", "onnx", "llama_cpp")


class LLMBackend(Protocol):
    def list_models(self) -> List[str]: ...

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> Iterator[StreamEvent]: ...

    def close(self) -> None: ...


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_model_path(path: str, *, expect_file: bool = False) -> str:
    """Resolve a configured local model path without performing network access."""
    raw = os.path.expandvars(os.path.expanduser((path or "").strip()))
    if not raw:
        raise LLMError("未配置本地模型路径")
    candidate = Path(raw)
    if not candidate.is_absolute():
        executable = Path(sys.executable).resolve()
        candidates = [
            Path.cwd() / candidate,
            project_root() / candidate,
            executable.parent / candidate,
        ]
        # macOS .app: <App>.app/Contents/MacOS/<exe>，允许 models 与 .app 同级。
        if len(executable.parents) >= 3:
            candidates.append(executable.parents[2] / candidate)
        candidate = next((item for item in candidates if item.exists()), candidates[1])
    candidate = candidate.resolve()
    if not candidate.exists():
        raise LLMError(f"本地模型不存在: {candidate}")
    if expect_file and not candidate.is_file():
        raise LLMError(f"需要模型文件，实际为目录: {candidate}")
    if not expect_file and not candidate.is_dir():
        raise LLMError(f"需要模型目录，实际为文件: {candidate}")
    return str(candidate)


def discover_local_models(models_dir: str = "models", backend: str = "") -> List[str]:
    """Return local model paths compatible with the selected backend."""
    try:
        root = Path(resolve_model_path(models_dir))
    except Exception:
        return []
    backend = (backend or "").lower()
    found: List[str] = []
    if backend == "llama_cpp":
        found.extend(str(p.resolve()) for p in root.rglob("*.gguf"))
    else:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            names = {p.name for p in child.iterdir() if p.is_file()}
            if backend in ("mlx_lm", "mlx_vlm") and not (
                "config.json" in names and any(n.endswith(".safetensors") for n in names)
            ):
                continue
            if backend == "onnx" and not any(n.endswith(".onnx") for n in names):
                continue
            found.append(str(child.resolve()))
    return sorted(found)


class _TaggedOutputParser:
    """Turn local-model text tags into reasoning/content/tool StreamEvents."""

    _OPENERS = (("<think>", "reasoning"), ("<tool_call>", "tool"))
    _CLOSERS = {"reasoning": "</think>", "tool": "</tool_call>"}

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "content"
        self._tool_buffer = ""
        self.tool_calls: List[Dict[str, Any]] = []

    @staticmethod
    def _tool_call(payload: str) -> Optional[Dict[str, Any]]:
        raw = payload.strip().strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("本地模型返回了无法解析的 tool_call: %s", raw[:300])
            return None
        if isinstance(obj, list):
            obj = obj[0] if obj else None
        if not isinstance(obj, dict):
            return None
        if isinstance(obj.get("function"), dict):
            function = obj["function"]
            name = function.get("name", "")
            arguments = function.get("arguments", {})
        else:
            name = obj.get("name", obj.get("tool", ""))
            arguments = obj.get("arguments", obj.get("parameters", {}))
        if not name:
            return None
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return {
            "id": str(obj.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
            "type": "function",
            "function": {"name": str(name), "arguments": arguments},
        }

    def feed(self, text: str) -> Iterator[StreamEvent]:
        self._buffer += text or ""
        while self._buffer:
            if self._mode in self._CLOSERS:
                closer = self._CLOSERS[self._mode]
                idx = self._buffer.find(closer)
                if idx < 0:
                    if self._mode == "tool":
                        self._tool_buffer += self._buffer
                        self._buffer = ""
                    else:
                        keep = max(len(closer) - 1, 0)
                        safe = self._buffer[:-keep] if keep and len(self._buffer) > keep else ""
                        if safe:
                            yield StreamEvent("reasoning", content=safe)
                            self._buffer = self._buffer[len(safe):]
                        break
                    continue
                before = self._buffer[:idx]
                self._buffer = self._buffer[idx + len(closer):]
                if self._mode == "reasoning" and before:
                    yield StreamEvent("reasoning", content=before)
                elif self._mode == "tool":
                    self._tool_buffer += before
                    call = self._tool_call(self._tool_buffer)
                    if call is not None:
                        self.tool_calls.append(call)
                    self._tool_buffer = ""
                self._mode = "content"
                continue

            match = None
            for opener, mode in self._OPENERS:
                idx = self._buffer.find(opener)
                if idx >= 0 and (match is None or idx < match[0]):
                    match = (idx, opener, mode)
            if match is not None:
                idx, opener, mode = match
                if idx:
                    yield StreamEvent("content", content=self._buffer[:idx])
                self._buffer = self._buffer[idx + len(opener):]
                self._mode = mode
                continue

            keep = max(len(opener) for opener, _ in self._OPENERS) - 1
            safe_len = max(0, len(self._buffer) - keep)
            if safe_len:
                yield StreamEvent("content", content=self._buffer[:safe_len])
                self._buffer = self._buffer[safe_len:]
            break

    def finish(self) -> Iterator[StreamEvent]:
        if self._mode == "tool":
            self._tool_buffer += self._buffer
            call = self._tool_call(self._tool_buffer)
            if call is not None:
                self.tool_calls.append(call)
        elif self._buffer:
            event_type = "reasoning" if self._mode == "reasoning" else "content"
            yield StreamEvent(event_type, content=self._buffer)
        self._buffer = ""
        self._tool_buffer = ""
        self._mode = "content"


class _LazyLocalBackend:
    def __init__(self, model_path: str, config: Any):
        self.model_path = model_path
        self.config = config
        self.model = os.path.basename(model_path.rstrip(os.sep)) or model_path
        self._load_lock = threading.RLock()
        self._generation_lock = threading.RLock()
        self._closed = False

    def list_models(self) -> List[str]:
        return [self.model]

    def _check_open(self) -> None:
        if self._closed:
            raise LLMError("本地模型后端已关闭")

    def clone(self):
        # Local runtimes keep multi-GB weights. Sharing one synchronized backend
        # is intentional; constructing an extraction-only copy would double RAM.
        return self


class MLXLanguageBackend(_LazyLocalBackend):
    """Direct MLX-LM text model backend."""

    def __init__(self, model_path: str, config: Any):
        super().__init__(resolve_model_path(model_path), config)
        self._model_obj = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        self._check_open()
        if self._model_obj is not None:
            return
        with self._load_lock:
            if self._model_obj is not None:
                return
            try:
                from mlx_lm import load
            except ImportError as e:
                raise LLMError("MLX-LM 未安装，请安装: pip install mlx-lm") from e
            logger.info("正在从本地加载 MLX-LM: %s", self.model_path)
            self._model_obj, self._tokenizer = load(
                self.model_path,
                lazy=False,
                trust_remote_code=False,
            )
            logger.info("MLX-LM 加载完成: %s", self.model_path)

    def _prompt(self, messages: List[Dict[str, Any]], tools) -> Any:
        kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self._ensure_loaded()
        with self._generation_lock:
            try:
                from mlx_lm import stream_generate
                from mlx_lm.sample_utils import make_logits_processors, make_sampler

                prompt = self._prompt(messages, tools)
                sampler = make_sampler(
                    temp=max(0.0, float(temperature)),
                    top_p=float(getattr(self.config, "top_p", 0.95)),
                    min_p=float(getattr(self.config, "min_p", 0.0)),
                    top_k=int(getattr(self.config, "top_k", 0)),
                )
                logits_processors = make_logits_processors(
                    repetition_penalty=float(getattr(self.config, "repetition_penalty", 1.0)),
                )
                parser = _TaggedOutputParser()
                prompt_tokens = 0
                completion_tokens = 0
                for response in stream_generate(
                    self._model_obj,
                    self._tokenizer,
                    prompt,
                    max_tokens=int(getattr(self.config, "max_tokens", 32768)),
                    sampler=sampler,
                    logits_processors=logits_processors,
                ):
                    prompt_tokens = int(getattr(response, "prompt_tokens", prompt_tokens) or prompt_tokens)
                    completion_tokens = int(
                        getattr(response, "generation_tokens", completion_tokens + 1)
                        or completion_tokens + 1
                    )
                    for event in parser.feed(getattr(response, "text", "")):
                        yield event
                yield from parser.finish()
                yield StreamEvent(
                    "done",
                    tool_calls=parser.tool_calls,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    finish_reason="tool_calls" if parser.tool_calls else "stop",
                )
            except LLMError:
                raise
            except Exception as e:
                raise LLMError(f"MLX-LM 推理失败: {type(e).__name__}: {e}") from e

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            self._model_obj = None
            self._tokenizer = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass


class MLXVLMBackend(_LazyLocalBackend):
    """Direct MLX-VLM backend for text/image/audio/video chat models."""

    def __init__(self, model_path: str, config: Any):
        super().__init__(resolve_model_path(model_path), config)
        self._model_obj = None
        self._processor = None
        self._model_config: Dict[str, Any] = {}

    def _ensure_loaded(self) -> None:
        self._check_open()
        if self._model_obj is not None:
            return
        with self._load_lock:
            if self._model_obj is not None:
                return
            try:
                from mlx_vlm import load
            except ImportError as e:
                raise LLMError("MLX-VLM 未安装，请安装: pip install mlx-vlm") from e
            config_path = Path(self.model_path) / "config.json"
            if config_path.exists():
                self._model_config = json.loads(config_path.read_text(encoding="utf-8"))
            logger.info("正在从本地加载 MLX-VLM: %s", self.model_path)
            self._model_obj, self._processor = load(self.model_path, lazy=False, strict=True)
            logger.info("MLX-VLM 加载完成: %s", self.model_path)

    @staticmethod
    def _media(messages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        media = {"image": [], "audio": [], "video": []}
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                for kind, value in re.findall(
                    r"\[(image|audio|video):([^\]]+)\]", content, flags=re.IGNORECASE
                ):
                    value = value.strip()
                    if value:
                        media[kind.lower()].append(value)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", ""))
                if part_type in ("image", "input_image", "image_url"):
                    value = part.get("image") or part.get("input_image") or part.get("image_url")
                    if isinstance(value, dict):
                        value = value.get("url")
                    if value:
                        media["image"].append(str(value))
                elif part_type in ("audio", "input_audio"):
                    value = part.get("audio") or part.get("input_audio")
                    if value:
                        media["audio"].append(str(value))
                elif part_type in ("video", "input_video"):
                    value = part.get("video") or part.get("input_video")
                    if value:
                        media["video"].append(str(value))
        return media

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self._ensure_loaded()
        with self._generation_lock:
            try:
                from mlx_lm.sample_utils import make_sampler
                from mlx_vlm import apply_chat_template, stream_generate

                media = self._media(messages)
                template_kwargs: Dict[str, Any] = {
                    "num_images": len(media["image"]),
                    "num_audios": len(media["audio"]),
                }
                if tools:
                    template_kwargs["tools"] = tools
                prompt = apply_chat_template(
                    self._processor,
                    self._model_config,
                    messages,
                    **template_kwargs,
                )
                sampler = make_sampler(
                    temp=max(0.0, float(temperature)),
                    top_p=float(getattr(self.config, "top_p", 0.95)),
                    min_p=float(getattr(self.config, "min_p", 0.0)),
                    top_k=int(getattr(self.config, "top_k", 0)),
                )
                parser = _TaggedOutputParser()
                prompt_tokens = 0
                completion_tokens = 0
                kwargs = {
                    "max_tokens": int(getattr(self.config, "max_tokens", 32768)),
                    "sampler": sampler,
                    "verbose": False,
                }
                for response in stream_generate(
                    self._model_obj,
                    self._processor,
                    prompt,
                    image=media["image"] or None,
                    audio=media["audio"] or None,
                    video=media["video"] or None,
                    **kwargs,
                ):
                    prompt_tokens = int(getattr(response, "prompt_tokens", prompt_tokens) or prompt_tokens)
                    completion_tokens = int(
                        getattr(response, "generation_tokens", completion_tokens + 1)
                        or completion_tokens + 1
                    )
                    for event in parser.feed(getattr(response, "text", "")):
                        yield event
                yield from parser.finish()
                yield StreamEvent(
                    "done",
                    tool_calls=parser.tool_calls,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    finish_reason="tool_calls" if parser.tool_calls else "stop",
                )
            except Exception as e:
                raise LLMError(f"MLX-VLM 推理失败: {type(e).__name__}: {e}") from e

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            self._model_obj = None
            self._processor = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass


class ONNXLanguageBackend(_LazyLocalBackend):
    """Direct ONNX Runtime + Transformers + NumPy local generation backend.

    It intentionally recomputes the full sequence instead of requiring a
    model-specific KV-cache naming convention. This supports ordinary exported
    decoder graphs and multimodal graphs whose processor inputs match the ONNX
    input names. Unknown mandatory inputs produce an actionable error.
    """

    def __init__(self, model_path: str, config: Any):
        raw = resolve_model_path(model_path, expect_file=Path(model_path).suffix.lower() == ".onnx")
        super().__init__(raw, config)
        self._session = None
        self._tokenizer = None
        self._processor = None
        self._model_file = ""

    def _ensure_loaded(self) -> None:
        self._check_open()
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            try:
                import onnxruntime as ort
                from transformers import AutoProcessor, AutoTokenizer
            except ImportError as e:
                raise LLMError(
                    "ONNX 本地后端需要 onnxruntime 与 transformers"
                ) from e
            path = Path(self.model_path)
            model_dir = path if path.is_dir() else path.parent
            if path.is_file():
                model_file = path
            else:
                candidates = sorted(path.glob("*.onnx")) + sorted(path.glob("onnx/*.onnx"))
                if not candidates:
                    raise LLMError(f"目录中未找到 ONNX 文件: {path}")
                model_file = candidates[0]
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(getattr(self.config, "onnx_threads", 4)))
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(model_file),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._model_file = str(model_file)
            try:
                self._processor = AutoProcessor.from_pretrained(
                    str(model_dir), local_files_only=True, trust_remote_code=False
                )
            except Exception:
                self._processor = None
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), local_files_only=True, trust_remote_code=False
            )
            logger.info("ONNX 本地模型加载完成(CPU): %s", model_file)

    def _prompt(self, messages, tools):
        owner = self._processor or self._tokenizer
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        try:
            return owner.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            return owner.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _sample(logits, generated, temperature, top_p, min_p, top_k, repetition_penalty):
        import numpy as np

        values = np.asarray(logits, dtype=np.float64).copy()
        if repetition_penalty != 1.0:
            for token in set(generated[-64:]):
                values[token] = values[token] / repetition_penalty if values[token] > 0 else values[token] * repetition_penalty
        if temperature <= 0:
            return int(values.argmax())
        values /= max(temperature, 1e-5)
        if top_k > 0 and top_k < values.size:
            cutoff = np.partition(values, -top_k)[-top_k]
            values[values < cutoff] = -np.inf
        values -= np.nanmax(values)
        probs = np.exp(values)
        probs /= probs.sum()
        if min_p > 0:
            probs[probs < probs.max() * min_p] = 0
        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[order])
            remove = cumulative > top_p
            if remove.size:
                remove[0] = False
            probs[order[remove]] = 0
        total = probs.sum()
        if not math.isfinite(float(total)) or total <= 0:
            return int(values.argmax())
        probs /= total
        return int(np.random.default_rng().choice(values.size, p=probs))

    def _initial_inputs(self, prompt: str, messages):
        import numpy as np

        media = MLXVLMBackend._media(messages)
        if self._processor is not None and any(media.values()):
            images = media["image"] or None
            if images:
                try:
                    from PIL import Image
                    images = [Image.open(x).convert("RGB") if os.path.isfile(x) else x for x in images]
                except Exception:
                    pass
            processor_kwargs: Dict[str, Any] = {
                "text": prompt,
                "return_tensors": "np",
            }
            if images:
                processor_kwargs["images"] = images
            if media["audio"]:
                processor_kwargs["audio"] = media["audio"]
            if media["video"]:
                processor_kwargs["videos"] = media["video"]
            batch = self._processor(**processor_kwargs)
        else:
            batch = self._tokenizer(prompt, return_tensors="np")
        return {k: np.asarray(v) for k, v in dict(batch).items()}

    def _feeds(self, encoded: Dict[str, Any], input_ids):
        import numpy as np

        input_names = {item.name: item for item in self._session.get_inputs()}
        feeds: Dict[str, Any] = {}
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
                        raise LLMError(
                            "该 ONNX VLM 需要显式三维 M-RoPE position_ids；"
                            "processor 未生成此输入，不能用普通二维位置替代"
                        )
                    value = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
            elif name == "token_type_ids":
                value = np.zeros_like(input_ids, dtype=np.int64)
            elif name in encoded:
                value = encoded[name]
            else:
                raise LLMError(
                    f"ONNX 模型需要未能自动构造的输入 '{name}' ({spec.shape})；"
                    "请使用与该导出图匹配的专用 processor/后端"
                )
            feeds[name] = np.asarray(value)
        return feeds

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self._ensure_loaded()
        with self._generation_lock:
            try:
                import numpy as np

                prompt = self._prompt(messages, tools)
                encoded = self._initial_inputs(prompt, messages)
                if "input_ids" not in encoded:
                    raise LLMError("ONNX processor 未返回 input_ids")
                input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
                prompt_tokens = int(input_ids.size)
                generated: List[int] = []
                decoded = ""
                parser = _TaggedOutputParser()
                eos = self._tokenizer.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
                for _ in range(int(getattr(self.config, "max_tokens", 32768))):
                    outputs = self._session.run(None, self._feeds(encoded, input_ids))
                    logits = np.asarray(outputs[0])
                    if logits.ndim < 3:
                        raise LLMError(
                            f"ONNX 输出形状 {logits.shape} 不是生成模型 logits；"
                            "分类/Embedding 模型不能作为主 LLM"
                        )
                    token = self._sample(
                        logits[0, -1], generated, float(temperature),
                        float(getattr(self.config, "top_p", 0.95)),
                        float(getattr(self.config, "min_p", 0.0)),
                        int(getattr(self.config, "top_k", 0)),
                        float(getattr(self.config, "repetition_penalty", 1.0)),
                    )
                    if token in eos_ids:
                        break
                    generated.append(token)
                    input_ids = np.concatenate(
                        (input_ids, np.asarray([[token]], dtype=np.int64)), axis=1
                    )
                    new_decoded = self._tokenizer.decode(generated, skip_special_tokens=False)
                    delta = new_decoded[len(decoded):]
                    decoded = new_decoded
                    yield from parser.feed(delta)
                yield from parser.finish()
                yield StreamEvent(
                    "done",
                    tool_calls=parser.tool_calls,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(generated),
                        "total_tokens": prompt_tokens + len(generated),
                    },
                    finish_reason="tool_calls" if parser.tool_calls else "stop",
                )
            except LLMError:
                raise
            except Exception as e:
                raise LLMError(f"ONNX 本地生成失败: {type(e).__name__}: {e}") from e

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            self._session = None
            self._tokenizer = None
            self._processor = None


class LlamaCppBackend(_LazyLocalBackend):
    """Direct llama-cpp-python backend for GGUF LLM, VLM, and tool chat."""

    def __init__(self, model_path: str, config: Any):
        super().__init__(resolve_model_path(model_path, expect_file=True), config)
        self._llama = None

    def _chat_handler(self):
        projection = getattr(self.config, "llm_vision_projection_path", "") or ""
        if not projection:
            return None
        projection = resolve_model_path(projection, expect_file=True)
        try:
            from llama_cpp import llama_chat_format as formats
        except ImportError as e:
            raise LLMError("llama-cpp-python 未安装") from e
        handler_name = str(getattr(self.config, "llm_vlm_handler", "mtmd")).lower()
        handlers = {
            "mtmd": formats.MTMDChatHandler,
            "llava15": formats.Llava15ChatHandler,
            "llava16": formats.Llava16ChatHandler,
            "qwen25vl": formats.Qwen25VLChatHandler,
            "minicpmv26": formats.MiniCPMv26ChatHandler,
            "moondream": formats.MoondreamChatHandler,
            "nanollava": formats.NanoLlavaChatHandler,
        }
        cls = handlers.get(handler_name)
        if cls is None:
            raise LLMError(f"不支持的 llama.cpp VLM handler: {handler_name}")
        kwargs = {"clip_model_path": projection, "verbose": False}
        if handler_name == "mtmd":
            kwargs["use_gpu"] = int(getattr(self.config, "local_gpu_layers", -1)) != 0
        return cls(**kwargs)

    def _ensure_loaded(self) -> None:
        self._check_open()
        if self._llama is not None:
            return
        with self._load_lock:
            if self._llama is not None:
                return
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise LLMError(
                    "GGUF 后端需要 llama-cpp-python: pip install llama-cpp-python"
                ) from e
            self._llama = Llama(
                model_path=self.model_path,
                n_ctx=max(512, int(getattr(self.config, "max_context_tokens", 32768))),
                n_batch=max(1, int(getattr(self.config, "local_model_batch_size", 512))),
                n_threads=max(1, int(getattr(self.config, "local_model_threads", 4))),
                n_gpu_layers=int(getattr(self.config, "local_gpu_layers", -1)),
                chat_format=getattr(self.config, "llm_chat_format", "") or None,
                chat_handler=self._chat_handler(),
                verbose=False,
            )
            logger.info("GGUF 模型加载完成: %s", self.model_path)

    @staticmethod
    def _multimodal_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        pattern = re.compile(r"\[image:([^\]]+)\]", flags=re.IGNORECASE)
        for message in messages:
            copied = dict(message)
            content = copied.get("content")
            if isinstance(content, str):
                paths = [value.strip() for value in pattern.findall(content) if value.strip()]
                if paths:
                    text = pattern.sub("", content).strip()
                    parts: List[Dict[str, Any]] = [
                        {"type": "image_url", "image_url": {"url": path}}
                        for path in paths
                    ]
                    if text:
                        parts.append({"type": "text", "text": text})
                    copied["content"] = parts
            normalized.append(copied)
        return normalized

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self._ensure_loaded()
        with self._generation_lock:
            parser = _TaggedOutputParser()
            tool_acc: Dict[int, Dict[str, Any]] = {}
            usage: Dict[str, int] = {}
            finish_reason = ""
            try:
                stream = self._llama.create_chat_completion(
                    messages=self._multimodal_messages(messages),
                    tools=tools or None,
                    temperature=float(temperature),
                    top_p=float(getattr(self.config, "top_p", 0.95)),
                    top_k=int(getattr(self.config, "top_k", 20)),
                    min_p=float(getattr(self.config, "min_p", 0.05)),
                    repeat_penalty=float(getattr(self.config, "repetition_penalty", 1.0)),
                    max_tokens=int(getattr(self.config, "max_tokens", 32768)),
                    stream=True,
                )
                for obj in stream:
                    if obj.get("usage"):
                        usage = dict(obj["usage"])
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    yield from parser.feed(delta.get("content") or "")
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        yield StreamEvent("reasoning", content=reasoning)
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0))
                        slot = tool_acc.setdefault(
                            idx,
                            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        function = tc.get("function") or {}
                        name = function.get("name") or ""
                        if name and name not in slot["function"]["name"]:
                            slot["function"]["name"] += name
                        slot["function"]["arguments"] += function.get("arguments") or ""
                yield from parser.finish()
                calls = [tool_acc[k] for k in sorted(tool_acc)] or parser.tool_calls
                for call in calls:
                    if not call.get("id"):
                        call["id"] = f"call_{uuid.uuid4().hex[:12]}"
                yield StreamEvent(
                    "done",
                    tool_calls=calls,
                    usage=usage,
                    finish_reason=finish_reason or ("tool_calls" if calls else "stop"),
                )
            except Exception as e:
                raise LLMError(f"llama.cpp GGUF 推理失败: {type(e).__name__}: {e}") from e

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            llama = self._llama
            self._llama = None
            if llama is not None:
                try:
                    llama.close()
                except Exception:
                    pass


def create_llm_backend(config: Any) -> LLMBackend:
    backend = str(getattr(config, "llm_backend", "openai") or "openai").lower()
    if backend == "openai":
        return LLMClient(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout=config.request_timeout,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
        )
    path = getattr(config, "llm_model_path", "")
    if backend == "mlx_lm":
        return MLXLanguageBackend(path, config)
    if backend == "mlx_vlm":
        return MLXVLMBackend(path, config)
    if backend == "onnx":
        return ONNXLanguageBackend(path, config)
    if backend == "llama_cpp":
        return LlamaCppBackend(path, config)
    raise LLMError(f"未知主模型后端: {backend}; 可选值: {', '.join(LLM_BACKENDS)}")
