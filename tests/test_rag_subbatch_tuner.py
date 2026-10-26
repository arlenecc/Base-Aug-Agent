"""自适应嵌入子批大小控制器 _SubBatchTuner 的单元测试（#125，#126-#128 修订）。

覆盖状态机与内存安全（2026-10 重标定：MLX 默认后端下范围 [2, 64]、起步 16、
步长 2）。断言一律用常量表达，改档位不需要改测试。
- 初始 16，窗口 = 2 个子批，步长 2
- 爬升：试探 +2，更快（≥3%）则采纳并继续上调
- 回退：试探变慢则退回上一个数字，进入 monitor
- 上限：64 封顶
- 降档：monitor 下连续 2 窗口低于历史最优 75% → 降 1 档（步长 2），下限 2
- 恢复（#129）：尖峰降档后速率回到最优 95% → 直接回到最优尺寸（消除降档
  单向棘轮——旧版「降到 1 再也不回去」的根因）
- 重新爬升：monitor 下速率比历史最优高 10% → 重新爬山（尺寸 ≥ 最优尺寸时）
- RSS 软墙（#129）：force_down 无闩锁降档，超 8GB 避让、回墙下即停
- 预热丢弃：首个完整窗口（含模型加载）不参与判定
- 形状预算：_shape_cap_size 对超长文本钳制子批条数（替代绝对 RSS 阈值，
  后者在真实同步进程基线 6-10GB 下必然误触发并卡死在下限——#126 修订）；
  但最低保 2 条、不夹到 1（#139），并按后端取预算（mlx=4@8192、onnx=2@8192）

度量 ops = n × max_len²，测试直接以确定性 (n, max_len, dt) 驱动，
score = Σops / Σdt，无真实模型依赖。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC),):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.rag.vector_store import (  # noqa: E402
    _SUBBATCH_MAX,
    _SUBBATCH_MIN,
    _SUBBATCH_START,
    _SUBBATCH_STEP,
    _SubBatchTuner,
    VectorStore,
    _normalize,
    _shape_cap_size,
)

# 尺寸档位一律相对常量推导，避免把 16/18/20 之类的数字写死在断言里。
S0 = _SUBBATCH_START                 # 起步
S1 = _SUBBATCH_START + _SUBBATCH_STEP  # 第一次试探
S2 = _SUBBATCH_START + 2 * _SUBBATCH_STEP  # 第二次试探


def _report_window(tuner: _SubBatchTuner, size: int, score: float) -> None:
    """以确定的窗口分数上报一个完整窗口（2 个子批）。

    每个子批 ops = size × 100²，两个子批合计 dt = 2×100²×size/score，
    保证窗口分数恰为 score。首个预热窗口会被调优器丢弃，这里先喂一个
    哑预热窗口（极慢分数、反正被丢弃）确保后续窗口生效。
    """
    sub_ops = float(size) * 100.0 ** 2
    if not tuner._warmed:
        for _ in range(2):
            tuner.report(size, 100, sub_ops / 1e9)
    dt_each = sub_ops / score
    for _ in range(2):
        tuner.report(size, 100, dt_each)


# ---------------------------------------------------------------------------
# 基本状态
# ---------------------------------------------------------------------------

def test_initial_size_is_start():
    tuner = _SubBatchTuner()
    assert tuner.propose() == _SUBBATCH_START == 16


def test_window_needs_two_subbatches():
    tuner = _SubBatchTuner()
    tuner.report(S0, 100, 1.0)  # 单个子批 → 窗口未满，状态不变
    assert tuner.propose() == S0


# ---------------------------------------------------------------------------
# 爬升
# ---------------------------------------------------------------------------

def test_climb_accepts_faster_candidate():
    tuner = _SubBatchTuner()
    _report_window(tuner, S0, 80_000.0)      # 基线 @S0
    assert tuner.propose() == S1             # 进入试探
    _report_window(tuner, S1, 100_000.0)     # 试探 @S1：+25%
    assert tuner.propose() == S2             # 采纳并继续上调


def test_climb_reverts_on_regression():
    tuner = _SubBatchTuner()
    _report_window(tuner, S0, 80_000.0)      # 基线 @S0
    assert tuner.propose() == S1
    _report_window(tuner, S1, 50_000.0)      # 试探 @S1：更慢
    assert tuner.propose() == S0             # 退回上一个数字
    _report_window(tuner, S0, 80_000.0)      # monitor 窗口：不再试探
    assert tuner.propose() == S0


def test_climb_caps_at_max():
    tuner = _SubBatchTuner()
    size = tuner.propose()
    while True:
        # 分数随尺寸线性增长，保证每档试探都被采纳
        _report_window(tuner, size, size * 1_000.0)
        new_size = tuner.propose()
        if new_size == size:
            break
        size = new_size
        assert size <= _SUBBATCH_MAX
    assert size == _SUBBATCH_MAX == 64


# ---------------------------------------------------------------------------
# 降档（monitor）
# ---------------------------------------------------------------------------

def _settle_in_monitor_at_start(tuner: _SubBatchTuner) -> None:
    """基线 @S0 → 试探 @S1 失败 → 停在 S0（monitor）。"""
    _report_window(tuner, S0, 80_000.0)
    _report_window(tuner, S1, 50_000.0)
    assert tuner.propose() == S0


def _settle_in_monitor_at_start_plus(tuner: _SubBatchTuner) -> None:
    """爬到 S1 后试探 @S2 失败 → 停在 S1（monitor）。

    基线 @S0(80k) → 试探 @S1(100k, +25% 采纳 → S2) → 试探 @S2(100k, 持平
    <3%) → 回退到 S1 进 monitor。
    """
    _report_window(tuner, S0, 80_000.0)
    _report_window(tuner, S1, 100_000.0)
    assert tuner.propose() == S2
    _report_window(tuner, S2, 100_000.0)
    assert tuner.propose() == S1


def _drive_slow_windows(tuner: _SubBatchTuner, size: int, n: int) -> None:
    """连续上报 n 个慢窗口（50% × 历史最优）。

    EMA（α=0.5）从最优值起步时需要 3 个慢窗口才让 streak 达到 2 并触发
    降档；EMA 已在低位时 2 个即可。测试统一多喂一个，幂等。
    """
    for _ in range(n):
        _report_window(tuner, size, 40_000.0)


def _drive_until_size(tuner: _SubBatchTuner, target: int, max_windows: int = 12) -> None:
    """持续上报慢窗口（50% × 历史最优）直到尺寸降到 target。

    EMA（α=0.5）从最优值起步首次降档需 3 个慢窗口，之后 EMA 已在低位、
    2 个甚至 1 个即可——不写死次数，逐窗驱动直到到达目标尺寸。
    """
    for _ in range(max_windows):
        if tuner.propose() == target:
            return
        _report_window(tuner, tuner.propose(), 40_000.0)
    assert tuner.propose() == target


def test_monitor_degrades_on_sustained_slowdown():
    tuner = _SubBatchTuner()
    _settle_in_monitor_at_start_plus(tuner)
    # 建立历史最优
    _report_window(tuner, S1, 80_000.0)
    assert tuner.propose() == S1
    # 连续慢窗口（< 75% × 80000 = 60000）→ 降 1 档（步长 2）。EMA 平滑下
    # 首个降档需要 3 个窗口。
    _report_window(tuner, S1, 40_000.0)
    assert tuner.propose() == S1  # 1 个窗口不够
    _drive_slow_windows(tuner, S1, 2)
    assert tuner.propose() == S0


def test_monitor_floors_at_min():
    tuner = _SubBatchTuner()
    _settle_in_monitor_at_start_plus(tuner)
    _report_window(tuner, S1, 80_000.0)
    # 持续劣化：一档一档降到底（步长 2），触底后不再下降。步长 2 后从 S1 到
    # 下限有 8 档，窗口预算要给足（每档约需 2-3 个窗口才触发降档）。
    for expected in range(S0, _SUBBATCH_MIN - 1, -_SUBBATCH_STEP):
        _drive_until_size(tuner, expected, max_windows=40)
        assert tuner.propose() == expected
    _drive_slow_windows(tuner, _SUBBATCH_MIN, 4)
    assert tuner.propose() == _SUBBATCH_MIN == 6


def test_monitor_recovers_without_stepdown():
    tuner = _SubBatchTuner()
    _settle_in_monitor_at_start_plus(tuner)
    _report_window(tuner, S1, 80_000.0)
    # 单个慢窗口后恢复 → 不降档
    _report_window(tuner, S1, 50_000.0)
    _report_window(tuner, S1, 80_000.0)
    _report_window(tuner, S1, 80_000.0)
    assert tuner.propose() == S1


# ---------------------------------------------------------------------------
# 恢复（#129：消除降档单向棘轮）与重新爬升（monitor → climb）
# ---------------------------------------------------------------------------

def test_recovery_after_transient_slowdown():
    """#129：瞬时尖峰（内存压力）降档后，速率恢复必须回到最优尺寸。"""
    tuner = _SubBatchTuner()
    _settle_in_monitor_at_start_plus(tuner)
    _report_window(tuner, S1, 80_000.0)   # best = 80000, best_size = S1
    # 尖峰：连续慢窗口 → 一路降到下限（步长 2，档位多，窗口预算给足）
    _drive_until_size(tuner, _SUBBATCH_MIN, max_windows=40)
    # 尖峰过去：速率回到 97.5%（≥95% 恢复线，远低于 110% 重爬线）
    _report_window(tuner, _SUBBATCH_MIN, 78_000.0)
    assert tuner.propose() == S1          # 直接回到最优尺寸
    # 稳定在 S1：后续正常窗口不再抖动
    _report_window(tuner, S1, 78_000.0)
    assert tuner.propose() == S1


def test_recovery_uses_direct_step_not_climb():
    """恢复不走爬山：plateau 上试探必然持平被回退，直接步进才能真回去。"""
    tuner = _SubBatchTuner()
    _settle_in_monitor_at_start_plus(tuner)
    _report_window(tuner, S1, 80_000.0)
    _drive_until_size(tuner, _SUBBATCH_MIN, max_windows=40)
    # 恢复窗口分数与历史最优完全持平（plateau）→ 仍须回到 S1
    _report_window(tuner, _SUBBATCH_MIN, 80_000.0)
    assert tuner.propose() == S1


def test_reclimb_when_fast_at_best_size():
    """环境变快且已在最优尺寸：EMA ≥ 1.10 × best → 重新爬山试探更大尺寸。"""
    tuner = _SubBatchTuner()
    _settle_in_monitor_at_start_plus(tuner)
    _report_window(tuner, S1, 80_000.0)   # best = 80000
    # +25% → 重新爬升（基线窗口保持当前尺寸）
    _report_window(tuner, S1, 100_000.0)
    assert tuner.propose() == S1
    _report_window(tuner, S1, 100_000.0)  # 基线窗口完成 → 试探 +步长
    assert tuner.propose() == S2


def test_rss_wall_force_down_without_latch():
    """#129：软墙 force_down 逐档下降、无闩锁，回墙下恢复分支照常带回。"""
    tuner = _SubBatchTuner()
    _settle_in_monitor_at_start_plus(tuner)
    _report_window(tuner, S1, 80_000.0)
    tuner.force_down()                   # 超墙：S1 → S0
    assert tuner.propose() == S0
    tuner.force_down()                   # 连续超墙：再降一档
    assert tuner.propose() == S0 - _SUBBATCH_STEP
    # 一路降到下限后不再下降
    while tuner.propose() > _SUBBATCH_MIN:
        tuner.force_down()
    assert tuner.propose() == _SUBBATCH_MIN == 6
    tuner.force_down()                   # 已在下限：不再下降
    assert tuner.propose() == _SUBBATCH_MIN
    # 无闩锁：速率恢复（≥95% 最优）→ 恢复分支把尺寸带回
    _report_window(tuner, _SUBBATCH_MIN, 78_000.0)
    assert tuner.propose() == S1


def test_reclimb_not_triggered_by_small_gain():
    tuner = _SubBatchTuner()
    _settle_in_monitor_at_start_plus(tuner)
    _report_window(tuner, S1, 80_000.0)   # best = 80000
    # +8%（低于 10% 阈值）→ 仍应保持 monitor，不触发爬升
    _report_window(tuner, S1, 86_000.0)
    _report_window(tuner, S1, 86_000.0)
    assert tuner.propose() == S1          # 若已进入爬升，基线窗口后会是 S2
    _report_window(tuner, S1, 86_000.0)
    assert tuner.propose() == S1


# ---------------------------------------------------------------------------
# 预热丢弃（#126 修订）
# ---------------------------------------------------------------------------

def test_first_window_discarded_as_warmup():
    """首个完整窗口（含模型加载，基线虚低）只用于预热，不参与判定。"""
    tuner = _SubBatchTuner()
    sub_ops = float(S0) * 100.0 ** 2
    # 首个窗口：极慢分数（模拟冷启动）
    for _ in range(2):
        tuner.report(S0, 100, sub_ops / 1000.0)
    assert tuner.propose() == S0          # 未进入试探
    assert tuner._warmed
    # 第二个窗口才作为爬升基线
    _report_window(tuner, S0, 80_000.0)
    assert tuner.propose() == S1          # 基线生效 → 试探
    # 预热只发生一次
    _report_window(tuner, S1, 50_000.0)
    assert tuner.propose() == S0


# ---------------------------------------------------------------------------
# 形状预算（#126 修订：替代绝对 RSS 阈值）
# ---------------------------------------------------------------------------

def test_shape_cap_clamps_long_texts():
    """激活 ∝ size × max_len²，预算 6M：超预算时收缩但不低于 _SUBBATCH_MIN(6)。"""
    assert _shape_cap_size(5, 1732) == 5   # 预算允许 2，但下限 6 > 请求 → 原样
    assert _shape_cap_size(5, 1414) == 5   # 同上
    assert _shape_cap_size(2, 2000) == 2   # 请求小于下限：按请求返回，不放大
    assert _shape_cap_size(1, 2000) == 1   # 请求只有 1 条时不放大
    assert _shape_cap_size(16, 8192) == 6  # 默认后端 6M 预算 @8192 收缩到下限 6
    assert _shape_cap_size(9, 8192) == 6   # 收缩不低于 _SUBBATCH_MIN


def test_shape_cap_leaves_short_texts_alone():
    """短文本不受预算影响：≤1095 字符时 5 条全保留。"""
    assert _shape_cap_size(5, 1095) == 5
    assert _shape_cap_size(5, 500) == 5
    assert _shape_cap_size(2, 50) == 2
    assert _shape_cap_size(5, 0) == 5      # 边界：零长文本不钳制


def test_shape_cap_is_backend_aware():
    """预算按后端取：mlx=4@8192、onnx=2@8192，均被下限 6 抬平。"""
    assert _shape_cap_size(16, 8192, "mlx") == 6   # 预算允许 4，被下限 6 抬高
    assert _shape_cap_size(16, 8192, "onnx") == 6  # 预算允许 2，被下限 6 抬高
    assert _shape_cap_size(16, 8192) == 6          # 未知后端：6M 预算 → 0 → 下限 6
    assert _shape_cap_size(16, 2048, "onnx") == 16   # 预算允许 35，但请求只有 16
    # 下限永远不会把请求值放大：请求 1 条仍返回 1
    assert _shape_cap_size(1, 8192, "mlx") == 1


# ---------------------------------------------------------------------------
# 集成：_embed_chunks 使用 tuner 且向量按原顺序回填
# ---------------------------------------------------------------------------

def test_embed_chunks_uses_tuner_and_preserves_order():
    from tests.conftest import FakeEmbeddingFunction

    ef = FakeEmbeddingFunction()
    store = VectorStore(persist_dir="/tmp/unused-subbatch-test", embedding_function=ef)
    store._ef = ef  # 白盒注入，绕过懒初始化（不触 LanceDB）

    chunks = [
        {"text": "字" * length, "source": "s", "chunk_index": i}
        for i, length in enumerate(range(50, 550, 10))
    ]
    records = store._embed_chunks(chunks, start_idx=0)

    assert len(records) == len(chunks)
    # 向量必须按原顺序回填（分桶只影响计算顺序，不影响结果顺序）
    for chunk, rec in zip(chunks, records):
        expected = _normalize(ef([chunk["text"]])[0])
        assert rec["vector"] == pytest.approx(expected, abs=1e-6)
        assert rec["text"] == chunk["text"]
    # tuner 收到了每个子批的上报，且尺寸始终在 [下限, 上限] 内
    assert store._subbatch_tuner._win_n <= 2
    assert _SUBBATCH_MIN <= store._subbatch_tuner.propose() <= _SUBBATCH_MAX
