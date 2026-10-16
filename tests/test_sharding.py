"""Tests for the LPT file sharding used by RAG ingest workers (#120)."""
from __future__ import annotations

import os
import pytest


def _import_shard():
    try:
        from agent.rag.engine import _shard_files_balanced
    except Exception:  # rag 依赖缺失（lancedb 等）的环境
        pytest.skip("rag engine deps unavailable")
    return _shard_files_balanced


def _make_files(tmp_path, sizes):
    paths = []
    for i, sz in enumerate(sizes):
        p = tmp_path / f"file_{i}.dat"
        p.write_bytes(b"x" * sz)
        paths.append(str(p))
    return paths


def test_nine_files_four_workers_no_empty_shard(tmp_path):
    """9 文件 4 worker：旧 ceil 切片会产生空片（3×3=9），新逻辑必须全满。"""
    shard = _import_shard()
    files = _make_files(tmp_path, [100] * 9)
    shards = shard(files, 4)
    assert all(len(s) > 0 for s in shards)
    assert sorted(sum(shards, [])) == sorted(files)  # 无丢失、无重复


def test_largest_files_go_to_distinct_workers(tmp_path):
    """最大的 4 个文件必须分属 4 个不同 worker（LPT 核心保证）。"""
    shard = _import_shard()
    # 4 个大文件 + 8 个小文件
    files = _make_files(tmp_path, [1000, 1000, 1000, 1000] + [10] * 8)
    shards = shard(files, 4)
    big = {f for f in files if os.path.getsize(f) == 1000}
    for s in shards:
        hits = [f for f in s if f in big]
        assert len(hits) <= 1


def test_counts_balanced_when_sizes_equal(tmp_path):
    """等大文件应均分：9/4 → [3,2,2,2]。"""
    shard = _import_shard()
    files = _make_files(tmp_path, [100] * 9)
    shards = shard(files, 4)
    counts = sorted(len(s) for s in shards)
    assert counts == [2, 2, 2, 3]


def test_totals_balanced_better_than_round_robin(tmp_path):
    """极端大小分布：贪心累计应比按序切片更均衡（慢书摊开）。"""
    shard = _import_shard()
    # 两个超大 + 一堆小文件
    files = _make_files(tmp_path, [10_000_000, 9_000_000] + [100] * 14)
    shards = shard(files, 4)
    totals = [sum(os.path.getsize(f) for f in s) for s in shards]
    # 两个大文件必须在不同片，且没有片同时背两个大文件
    for s in shards:
        big = sum(1 for f in s if os.path.getsize(f) > 1_000_000)
        assert big <= 1
    assert max(totals) - min(totals) < 10_000_000


def test_missing_file_treated_as_zero_size(tmp_path):
    """扫描后消失的文件（TOCTOU）：按 0 处理，不抛异常。"""
    shard = _import_shard()
    files = _make_files(tmp_path, [500, 500])
    files.append(str(tmp_path / "gone.dat"))  # 不存在
    shards = shard(files, 2)
    assert sorted(sum(shards, [])) == sorted(files)


def test_single_worker_returns_all(tmp_path):
    shard = _import_shard()
    files = _make_files(tmp_path, [1, 2, 3])
    assert shard(files, 1) == [files]
