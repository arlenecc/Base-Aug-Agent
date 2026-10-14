"""进程内共享的原子写工具。

固定 ``path + ".tmp"`` 的写法在多 writer 场景下会互相 truncate 同一个临时
文件，落盘半截或混合内容；写失败还会把 .tmp 残留下来被后续读取当成数据。
统一收敛到这里：唯一临时名 + 权限继承 + finally 清理 + fsync。
"""
from __future__ import annotations

import json
import os
import tempfile

try:
    _PROC_UMASK = os.umask(0)
    os.umask(_PROC_UMASK)
except OSError:  # pragma: no cover - 极端平台
    _PROC_UMASK = 0o022


def atomic_write_text(path: str, content: str, encoding: str = "utf-8") -> None:
    """把 ``content`` 原子写入 ``path``（tmp + ``os.replace``）。

    - 临时文件名唯一（``tempfile.mkstemp``），并发写不会互相覆盖；
    - 权限继承目标文件原权限，新建时用 ``0o666 & ~umask``（mkstemp 默认
      0600，直接 replace 会让文件突然变成「仅属主可读」）；
    - replace 前 ``fsync``，掉电不会得到空文件；
    - finally 清理半成品，失败不会留下 .tmp。

    抛出的 ``OSError`` 由调用方决定记日志还是上抛。
    """
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=dir_name)
    try:
        try:
            try:
                mode = os.stat(path).st_mode & 0o7777
            except OSError:
                mode = 0o666 & ~_PROC_UMASK
            os.fchmod(fd, mode)
            with os.fdopen(fd, "w", encoding=encoding) as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            fd = -1  # fdopen 已接管，finally 不再重复 close
            os.replace(tmp, path)
        finally:
            if fd >= 0:  # fchmod/fdopen 抛异常时避免 fd 泄漏
                try:
                    os.close(fd)
                except OSError:
                    pass
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def atomic_write_json(path: str, payload, *, indent: int | None = 2) -> None:
    """原子写 JSON（``ensure_ascii=False``，中文可读）。"""
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=indent))


def backup_corrupt(path: str) -> None:
    """把解析失败的坏文件改名为 ``.corrupt-<时间戳>``。

    旧行为是记一条 warning 后按空数据继续，下一次 ``_save()`` 就把坏文件
    覆盖掉——用户数据（手工编辑写错、版本不兼容）静默丢失且无从排查。
    改名后既保留现场，又不影响空初始化。
    """
    import time as _time
    try:
        if not os.path.exists(path):
            return
        stamp = _time.strftime("%Y%m%d-%H%M%S")
        os.replace(path, f"{path}.corrupt-{stamp}")
    except OSError:
        pass
