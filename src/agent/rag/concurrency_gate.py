"""按文件大小限流的并发闸门（读写锁语义）。

PDF 转换（docling/marker）的转换器不是线程安全的，且大文件并发会撑爆内存。
本模块提供一个轻量闸门：

  * 小文件（< threshold）：允许多个（small_capacity）并发转换，提高小文件吞吐。
  * 大文件（>= threshold）：独占——运行期间不允许任何其它（大或小）转换并发，
    防止大文件把内存撑爆。

为避免大文件饿死，大文件开始排队时会阻止新的小文件入场（已入场的小文件
照常跑完）。

与 threading.RLock 不同，这里用 threading.Condition 实现，调用方在
``acquire`` / ``release`` 之间自行持有转换器实例，闸门只负责「同一时刻有几路
转换在跑」的计数与互斥。
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class _Permit:
    """一次 ``acquire`` 的入场许可，固化「读者 / 写者」分类。

    见 :class:`SizeAwareConversionGate` 的注释：release 必须凭据归还，不能按
    size 重新推断分类。
    """

    __slots__ = ("is_writer",)

    def __init__(self, is_writer: bool) -> None:
        self.is_writer = is_writer

    def __repr__(self) -> str:  # 便于排查：日志/栈里能直接看出分类
        return f"_Permit({'writer' if self.is_writer else 'reader'})"


class SizeAwareConversionGate:
    """按文件大小限流的并发闸门（读写锁语义）。

    典型用法（docling/marker 各持一个模块级单例）：:

        gate = SizeAwareConversionGate(threshold=20 * 1024 * 1024, small_capacity=2)
        permit = gate.acquire(file_size)   # 阻塞直到拿到入场许可
        try:
            converter = pool.acquire()      # 取一个转换器实例
            ...
        finally:
            pool.release(converter)
            gate.release(permit)            # 归还入场许可（传 acquire 的返回值）
    """

    def __init__(self, threshold: int, small_capacity: int = 2) -> None:
        self.threshold = int(threshold)
        self.small_capacity = max(1, int(small_capacity))
        self._readers = 0
        self._writing = False
        self._writer_waiting = 0
        self._lock = threading.Lock()
        self._read_cv = threading.Condition(self._lock)
        self._write_cv = threading.Condition(self._lock)

    # -- acquire 返回的许可令牌 -----------------------------------------
    #
    # **为什么必须由 acquire 固化「读者/写者」分类，release 不能按 size 重判**：
    # 旧实现 release(size) 用「当时的 threshold」重新算 size >= threshold。若
    # 本次 acquire 与 release 之间阈值被 configure() 改小（RAGEngine.__init__
    # 每次构造都会 configure，同步期间重建引擎/改配置都会命中），原本按读者入场
    # 的请求就会在 release 时走写者分支 —— `_writing = False`（本来就是 False，
    # 空操作），而 `_readers` 永远不会被减回来。_readers 单调增长后，写者的等待
    # 条件 `while self._readers > 0` 永不满足 ⇒ **所有后续大文件转换永久阻塞**，
    # 且没有任何超时/日志（表现为「同步莫名卡住」）。反向错位同理。
    # 故这里在 acquire 时就把分类写进 permit，release 只认 permit。

    def configure(self, threshold: int, small_capacity: int) -> None:
        """热更新参数（引擎初始化时调用）。

        刻意**不更换闸门对象**、只在锁内改写字段：若换对象，已 acquire 的
        在飞持有者还挂在旧闸门上，新旧两个闸门各自计数，限流与互斥直接失效
        （最坏情况两个大文件同时开跑）。阈值热改后，已入场的大文件照常独占
        到结束，新请求按新阈值分类。
        """
        with self._lock:
            self.threshold = int(threshold)
            self.small_capacity = max(1, int(small_capacity))

    def active(self) -> int:
        """当前在飞的转换数（小文件读者 + 大文件写者）。

        供调用方判断「还有没有别的转换在跑」——例如 docling 每次转换后
        归还 MPS 缓存，只有自己是最后一班（active()==0）时才真正需要清。
        """
        with self._lock:
            return self._readers + (1 if self._writing else 0)

    def acquire(self, size: int) -> _Permit:
        """申请入场许可；必要时阻塞。返回许可令牌，**必须**交给 :meth:`release`。

        ``size`` 是文件字节数。小文件走「读者」（最多 small_capacity 个并发），
        大文件走「写者」（独占，且优先于新入场的小文件）。
        """
        with self._lock:
            if size >= self.threshold:
                # 写者：等大文件/写者全部退场，且新小文件不许入场
                self._writer_waiting += 1
                while self._readers > 0 or self._writing:
                    self._write_cv.wait()
                self._writer_waiting -= 1
                self._writing = True
                return _Permit(is_writer=True)
            # 读者：无写者、读者未满、且无大文件在排队
            while (
                self._writing
                or self._readers >= self.small_capacity
                or self._writer_waiting > 0
            ):
                self._read_cv.wait()
            self._readers += 1
            return _Permit(is_writer=False)

    def release(self, permit: _Permit) -> None:
        """归还入场许可，唤醒等待方。

        ``permit`` 是 :meth:`acquire` 返回的令牌。切勿改成按 size 重新判定
        分类（原因见 __init__ 下方的长注释）。重复/多余的 release 会被记日志
        并忽略——否则 _readers 会变负，写者的保护条件形同失效。
        """
        is_writer = permit.is_writer
        with self._lock:
            if is_writer:
                if not self._writing:
                    # 双重 release：再减一次会把状态搅乱，宁可报错也不要错减。
                    logger.error("SizeAwareConversionGate: 重复的写者 release，已忽略")
                    return
                self._writing = False
            else:
                if self._readers <= 0:
                    logger.error("SizeAwareConversionGate: 多余的读者 release，已忽略")
                    return
                self._readers -= 1
            self._write_cv.notify_all()
            self._read_cv.notify_all()
