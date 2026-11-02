"""从 HuggingFace 镜像下载模型权重（默认 https://hf-mirror.com）。

设计要点（对应需求）：
- 调用 HF 的 REST API（``/api/models/{repo}?full=true`` / ``/tree``）获取文件列表；
- 只拉取「需要的文件」（权重/配置/tokenizer 等，排除 .gitattributes / README /
  .md / .lock / .cache 等元数据）；
- 用标准库 ``urllib`` 流式下载，支持**断点续传**（HTTP ``Range`` 头 +
  本地已下字节续写），网络中断/进度可恢复；
- 通过 ``progress_cb`` 回传下载进度（总进度 / 当前文件 / 速度），便于 UI 展示。

等价于「Swift URLSession + 进度 + 断点续传」，但实现为纯 Python 以契合本项目的
PyQt6 / PyInstaller 技术栈。
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .model_registry import HF_MIRROR

# 进度回调签名：f(progress: dict)。dict 字段见 download_models 内构造。
ProgressCB = Optional[Callable[[dict], None]]
CancelCB = Optional[Callable[[], bool]]


class _Cancelled(Exception):
    """内部取消信号（与进度/取消回调配合）。"""


class _Speed:
    """轻量滑动窗口速率估计（bytes/sec）。"""

    def __init__(self) -> None:
        self._t = time.monotonic()
        self._b = 0.0
        self._ema = 0.0

    def feed(self, nbytes: int) -> None:
        now = time.monotonic()
        dt = now - self._t
        if dt <= 0:
            return
        rate = nbytes / dt
        # 指数滑动平均，避免抖动。
        self._ema = rate if self._ema == 0 else self._ema * 0.7 + rate * 0.3
        self._t = now
        self._b += nbytes

    def speed(self) -> float:
        return self._ema


def _api_get(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Base-Aug-Agent/1.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        import json

        return json.loads(resp.read().decode("utf-8"))


# 可重试的瞬时 HTTP 状态码（镜像偶发限流 403 / 429、网关抖动 5xx）。
_RETRYABLE_HTTP = frozenset({403, 429, 500, 502, 503, 504})


def _api_get_with_retry(url: str, timeout: int = 30, retries: int = 5) -> dict:
    """带退避重试的 API 请求：应对镜像偶发的 DNS / 连接 / 限流抖动。

    对确定性的 HTTP 错误（如 404 仓库不存在）直接抛出、不重试；对 403/429/5xx
    等瞬时错误退避重试。
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return _api_get(url, timeout)
        except urllib.error.HTTPError as e:
            if e.code in _RETRYABLE_HTTP:  # 瞬时：退避重试
                last_err = e
                if attempt < retries - 1:
                    time.sleep(min(2 ** (attempt + 1), 10))
                else:
                    raise
            else:  # 404 等确定性错误：不重试
                raise
        except Exception as e:  # DNS 解析失败 / 连接超时等瞬时错误 → 退避重试
            last_err = e
            if attempt < retries - 1:
                time.sleep(min(2 ** (attempt + 1), 10))
    raise last_err if last_err is not None else RuntimeError("API 请求失败")


def list_repo_files(repo_id: str, mirror: str = HF_MIRROR) -> List[Tuple[str, Optional[int]]]:
    """返回 ``[(rfilename, size), ...]``（size 可能为 None）。

    优先用 ``/tree/main?recursive=true``（hf-mirror 返回带 size 的列表），
    ``?full=true`` 作为回退（镜像常省略 size，此时 size 为 None）。两路均带
    退避重试，以应对镜像偶发的 DNS / 连接抖动。
    """
    last_err: Optional[Exception] = None
    # 主路径：tree 接口（带 size 的列表）。
    try:
        data = _api_get_with_retry(f"{mirror}/api/models/{repo_id}/tree/main?recursive=true")
        if isinstance(data, list):
            files = [
                (t.get("path"), t.get("size"))
                for t in data
                if isinstance(t, dict) and t.get("type") == "file" and t.get("path")
            ]
            if files:
                return files
    except Exception as e:  # 镜像 API 差异/网络问题 → 退回 full 接口
        last_err = e
    # 回退：full=true（siblings，镜像常省略 size）。
    try:
        data = _api_get_with_retry(f"{mirror}/api/models/{repo_id}?full=true")
        sibs = data.get("siblings") or []
        files = [(s.get("rfilename"), s.get("size")) for s in sibs if s.get("rfilename")]
        if files:
            return files
    except Exception as e:
        last_err = e
    raise RuntimeError(
        f"无法获取仓库文件列表: {repo_id}（镜像 {mirror}）。"
        f"请检查网络或仓库名是否正确。原始错误: {last_err}"
    )


def needed_files(
    files: List[Tuple[str, Optional[int]]], role: str = ""
) -> List[Tuple[str, Optional[int]]]:
    """过滤出模型运行真正需要的文件，剔除元数据/文档/缓存与冗余权重。

    按角色精细化（避免把整个仓库都拉下来）：

    - ``reranker``：只保留量化 ONNX（``*model_quantized.onnx``）+ tokenizer/config，
      排除其它 ONNX 变体（fp16/int8/uint8）与 PyTorch/MLX 权重；
    - 其它（MLX 的 llm/embedding 等）：保留 ``*.safetensors`` + 配置/tokenizer，
      排除 ``.py`` / PyTorch(``.bin/.pt/.pth``) / ``pytorch_model.bin`` 等无关文件；
    - 通用：排除 ``.gitattributes/.gitignore/.DS_Store/README/*.md/*.lock/.cache``。
    """
    out: List[Tuple[str, Optional[int]]] = []
    for fn, size in files:
        if not fn:
            continue
        low = fn.lower()
        base = low.rsplit("/", 1)[-1]
        # 通用排除
        if base in (".gitattributes", ".gitignore", ".ds_store"):
            continue
        if base.startswith("readme") or base.endswith(".md"):
            continue
        if base.endswith(".lock"):
            continue
        if base.endswith(".py") or base.endswith(".pyc") or base.endswith(".ipynb"):
            continue
        if ".cache/" in low or low.startswith(".cache/"):
            continue
        # PyTorch / 其它框架权重（MLX/ONNX Runtime 用不到）
        if base.endswith((".bin", ".pt", ".pth")) or base == "pytorch_model.bin":
            continue
        if role == "reranker":
            # 仅量化 ONNX + 配置；排除其余 ONNX 变体与 MLX/PyTorch 权重
            if base == "model.safetensors":
                continue
            if low.endswith(".onnx") and "model_quantized.onnx" not in low:
                continue
        out.append((fn, size))
    return out


def _download_file(
    repo_id: str,
    filename: str,
    dst: Path,
    size: Optional[int],
    mirror: str,
    existing: int,
    cancel_cb: CancelCB,
    state: dict,
    progress_cb: ProgressCB,
    speed: _Speed,
) -> int:
    """下载单个文件（支持断点续传）。返回本次**新写入**的字节数。

    更新 ``state['file_done']``（文件内绝对位置）与 ``state['overall_done']``
    （跨文件累计绝对字节，preflight 已计入已存在部分；本函数只在写入时累加增量，
    若服务端忽略 Range 从头重写则先扣回已计入的 existing，避免重复计数）。
    """
    url = f"{mirror}/{repo_id}/resolve/main/{urllib.parse.quote(filename)}"
    headers = {"User-Agent": "Base-Aug-Agent/1.0"}
    resume = existing
    if size is not None and resume >= size:
        # 已完整。
        state["file_done"] = size
        if progress_cb:
            progress_cb(dict(state))
        return 0
    if resume > 0:
        headers["Range"] = f"bytes={resume}-"

    resp = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=120)
            break
        except urllib.error.HTTPError as e:
            if e.code == 416:  # 请求区间不满足 → 视为已完成
                state["file_done"] = size or resume
                if progress_cb:
                    progress_cb(dict(state))
                return 0
            if attempt >= 4:  # 其它 HTTP 错误（如 5xx）确定性较强，不再重试
                raise
            time.sleep(min(2 ** (attempt + 1), 10))
        except Exception:  # URLError（DNS/连接）/ timeout / 连接中断 → 退避重试
            if attempt >= 4:
                raise
            time.sleep(min(2 ** (attempt + 1), 10))

    assert resp is not None
    start = resume if (resp.status == 206 and resume > 0) else 0
    if start == 0 and resume > 0:
        # 服务端忽略了 Range（返回 200 全文）：从头重写，先扣回已计入的 existing。
        state["overall_done"] -= resume
        state["file_done"] = 0
    state["file_done"] = start

    # 文件可能位于子目录（如 onnx/model_quantized.onnx），先确保父目录存在。
    dst.parent.mkdir(parents=True, exist_ok=True)

    mode = "rb+" if start > 0 else "wb"
    written = 0
    last_cb = time.monotonic()
    try:
        with open(dst, mode) as f:
            if start > 0:
                f.seek(start)
            while True:
                if cancel_cb and cancel_cb():
                    raise _Cancelled()
                buf = resp.read(262144)  # 256KB 块
                if not buf:
                    break
                f.write(buf)
                written += len(buf)
                state["file_done"] += len(buf)
                state["overall_done"] += len(buf)
                speed.feed(len(buf))
                state["speed"] = speed.speed()
                now = time.monotonic()
                if progress_cb and (now - last_cb) >= 0.2:
                    progress_cb(dict(state))
                    last_cb = now
            f.flush()
            os.fsync(f.fileno())
    finally:
        resp.close()
    if progress_cb:
        progress_cb(dict(state))
    return written


def download_models(
    specs: List[Tuple[str, str, str]],
    mirror: str = HF_MIRROR,
    progress_cb: ProgressCB = None,
    cancel_cb: CancelCB = None,
) -> int:
    """下载一批模型。

    ``specs`` 为 ``[(role, target_dir, repo_id), ...]``：每个模型下载到其
    ``target_dir``（即用户配置并解析出的目录），确保下载完成后能被同一
    ``resolve_model_path`` 在下次启动正常定位（避免反复判定缺失）。返回总下载字节数。
    """
    # 1) 列文件 + 计算总大小。
    plan: List[Tuple[str, Path, List[Tuple[str, Optional[int]]], str]] = []
    overall_total = 0
    for role, target_dir, repo_id in specs:
        files = needed_files(list_repo_files(repo_id, mirror), role)
        overall_total += sum(s or 0 for _, s in files)
        plan.append((repo_id, Path(target_dir), files, role))

    # 2) 预填「已存在字节」（断点续传起点）。
    overall_done = 0
    for _repo_id, local_dir, files, _role in plan:
        for fn, size in files:
            fp = local_dir / fn
            if fp.exists():
                overall_done += min(fp.stat().st_size, size or fp.stat().st_size)

    state = {
        "phase": "downloading",
        "model_total": len(plan),
        "overall_total": overall_total,
        "overall_done": overall_done,
        "model_index": 0,
        "model_local_name": "",
        "file_index": 0,
        "file_total": 0,
        "filename": "",
        "file_done": 0,
        "speed": 0.0,
    }
    if progress_cb:
        progress_cb(dict(state))

    speed = _Speed()
    for mi, (repo_id, local_dir, files, _role) in enumerate(plan):
        if cancel_cb and cancel_cb():
            raise _Cancelled()
        local_dir.mkdir(parents=True, exist_ok=True)
        state["model_index"] = mi
        state["model_local_name"] = local_dir.name
        state["file_total"] = len(files)
        for fi, (fn, size) in enumerate(files):
            if cancel_cb and cancel_cb():
                raise _Cancelled()
            state["file_index"] = fi
            state["filename"] = fn
            state["file_total"] = size or 0
            state["file_done"] = 0
            if progress_cb:
                progress_cb(dict(state))
            dst = local_dir / fn
            existing = dst.stat().st_size if dst.exists() else 0
            _download_file(
                repo_id, fn, dst, size, mirror, existing,
                cancel_cb, state, progress_cb, speed,
            )
        # 该模型全部文件下载完成 → 写标记（供启动检测判定「是否已就绪」，
        # 避免半下载目录被误判为已存在而跳过续传）。
        (local_dir / ".download_complete").write_text("ok")
    state["phase"] = "done"
    state["overall_done"] = overall_total
    if progress_cb:
        progress_cb(dict(state))
    return overall_done


def human_size(nbytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(nbytes)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.1f} {u}" if u != "B" else f"{int(v)} B"
        v /= 1024
