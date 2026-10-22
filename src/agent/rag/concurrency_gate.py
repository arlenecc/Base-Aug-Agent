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

import threading


class SizeAwareConversionGate:
    """按文件大小限流的并发闸门（读写锁语义）。

    典型用法（docling/marker 各持一个模块级单例）：:

        gate = SizeAwareConversionGate(threshold=20 * 1024 * 1024, small_capacity=2)
        gate.acquire(file_size)          # 阻塞直到拿到入场许可
        try:
            converter = pool.acquire()    # 取一个转换器实例
            ...
        finally:
            pool.release(converter)
            gate.release(file_size)       # 归还入场许可
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

    def acquire(self, size: int) -> None:
        """申请入场许可；必要时阻塞。

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
            else:
                # 读者：无写者、读者未满、且无大文件在排队
                while (
                    self._writing
                    or self._readers >= self.small_capacity
                    or self._writer_waiting > 0
                ):
                    self._read_cv.wait()
                self._readers += 1

    def release(self, size: int) -> None:
        """归还入场许可，唤醒等待方。"""
        with self._lock:
            if size >= self.threshold:
                self._writing = False
            else:
                self._readers -= 1
            self._write_cv.notify_all()
            self._read_cv.notify_all()
