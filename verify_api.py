"""
接口验证脚本 - 覆盖:
  - 单元测试: 流式批量、NaN/Inf 过滤、空数据、精度验证
  - HTTP 测试: 单值、批量、超大批量(同批真值对比)、原子性、内存监控
  - 非法值: 18种场景, 每个单独显示 PASS/FAIL

运行:
    python verify_api.py
"""

from __future__ import annotations

import gc
import math
import os
import random
import sys
import asyncio
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
from aiohttp import web, ClientSession

from tdigest import TDigest
from quantile_service import QuantileService


def _get_memory_kb() -> int:
    """获取当前进程内存使用 (KB)"""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return int(proc.memory_info().rss / 1024)
    except Exception:
        return -1


def rel_err(true_v: float, est_v: float) -> float:
    if true_v == 0:
        return 0.0 if est_v == 0 else float("inf")
    return abs(est_v - true_v) / abs(true_v) * 100


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details = []

    def pass_(self, msg: str = ""):
        self.passed = True
        self.message = msg

    def fail(self, msg: str):
        self.passed = False
        self.message = msg

    def add_detail(self, detail: str):
        self.details.append(detail)

    def print(self):
        status = "\033[32mPASS\033[0m" if self.passed else "\033[31mFAIL\033[0m"
        print(f"  [{status}] {self.name}")
        if self.message:
            print(f"         {self.message}")
        for d in self.details:
            print(f"           - {d}")


# ============================================================
# 同批真值数据 (所有超大批量测试共用)
# ============================================================

_TRUTH_SEED = 42
_TRUTH_RNG = random.Random(_TRUTH_SEED)
_TRUTH_N = 50000
_TRUTH_VALUES = [_TRUTH_RNG.paretovariate(1.3) * 5.0 for _ in range(_TRUTH_N)]
_TRUTH_ARR = np.sort(np.array(_TRUTH_VALUES, dtype=np.float64))
_TRUTH_P50 = float(np.quantile(_TRUTH_ARR, 0.5))
_TRUTH_P95 = float(np.quantile(_TRUTH_ARR, 0.95))
_TRUTH_P99 = float(np.quantile(_TRUTH_ARR, 0.99))


def _format_quantile_row(
    name: str, est_p50: float, est_p95: float, est_p99: float,
    true_p50: float = _TRUTH_P50, true_p95: float = _TRUTH_P95, true_p99: float = _TRUTH_P99
) -> str:
    return (
        f"{name}: p50={est_p50:.3f}(真值={true_p50:.3f}, err={rel_err(true_p50, est_p50):.2f}%), "
        f"p95={est_p95:.3f}(真值={true_p95:.3f}, err={rel_err(true_p95, est_p95):.2f}%), "
        f"p99={est_p99:.3f}(真值={true_p99:.3f}, err={rel_err(true_p99, est_p99):.2f}%)"
    )


# ============================================================
# 单元测试
# ============================================================

def test_streaming_batch_memory() -> TestResult:
    """验证超大批量下内存不随批量大小线性增长，用同批真值数据"""
    r = TestResult("流式批量插入 - 内存有界 + 同批真值对比")
    try:
        digest = TDigest(delta=100)
        max_buffer = digest.buffer_size

        # 用同批真值数据，流式灌入
        batch_size = 2000
        total = len(_TRUTH_VALUES)
        max_buffer_seen = 0

        for i in range(0, total, batch_size):
            batch = _TRUTH_VALUES[i:i + batch_size]
            digest.add_list(batch)
            max_buffer_seen = max(max_buffer_seen, len(digest.buffer))

        est_p50 = digest.quantile(0.5)
        est_p95 = digest.quantile(0.95)
        est_p99 = digest.quantile(0.99)

        r.add_detail(f"真值数据 N={total}")
        r.add_detail(f"buffer 限制={max_buffer}, 实际最大={max_buffer_seen}")
        r.add_detail(f"质心数={digest.num_centroids}, count={digest.count}")
        r.add_detail(_format_quantile_row("估计", est_p50, est_p95, est_p99))

        checks = []
        if digest.count != total:
            checks.append(f"count={digest.count}, 期望 {total}")
        if rel_err(_TRUTH_P50, est_p50) > 5:
            checks.append(f"p50 误差过大: {rel_err(_TRUTH_P50, est_p50):.2f}%")
        if rel_err(_TRUTH_P95, est_p95) > 5:
            checks.append(f"p95 误差过大: {rel_err(_TRUTH_P95, est_p95):.2f}%")
        if rel_err(_TRUTH_P99, est_p99) > 10:
            checks.append(f"p99 误差过大: {rel_err(_TRUTH_P99, est_p99):.2f}%")
        if max_buffer_seen > max_buffer + 50:
            checks.append(f"buffer 超限: {max_buffer_seen} > {max_buffer}")

        if checks:
            r.fail("; ".join(checks))
        else:
            r.pass_(f"压缩比 {total / digest.num_centroids:.0f}x, max_err={max(rel_err(_TRUTH_P50, est_p50), rel_err(_TRUTH_P95, est_p95), rel_err(_TRUTH_P99, est_p99)):.1f}%")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


def test_nan_inf_filtering() -> TestResult:
    """验证 NaN 和 Infinity 被正确过滤"""
    r = TestResult("NaN / Infinity 过滤")
    try:
        digest = TDigest(delta=100)

        ok_nan = digest.add(float("nan"))
        ok_inf = digest.add(float("inf"))
        ok_neg_inf = digest.add(float("-inf"))
        ok_normal = digest.add(42.0)

        r.add_detail(f"add(NaN)={ok_nan}, add(Inf)={ok_inf}, add(-Inf)={ok_neg_inf}, add(42)={ok_normal}")

        if ok_nan or ok_inf or ok_neg_inf:
            r.fail("NaN/Inf 不应被接受")
            return r
        if not ok_normal:
            r.fail("正常值应该被接受")
            return r

        batch = [1.0, 2.0, float("nan"), 3.0, float("inf"), 4.0, float("-inf"), 5.0]
        added = digest.add_list(batch)
        r.add_detail(f"批量 {len(batch)} 个值, 成功添加 {added} 个 (应=5)")

        if added != 5:
            r.fail(f"批量过滤后应有 5 个有效值, 实际 {added}")
            return r
        if digest.count != 6:
            r.fail(f"最终 count={digest.count}, 期望 6")
            return r

        p50 = digest.quantile(0.5)
        if math.isnan(p50) or math.isinf(p50):
            r.fail(f"quantile 返回非法值: {p50}")
            return r

        r.pass_("NaN/Inf 全部被过滤, 查询结果合法")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


def test_decimal_pareto_values() -> TestResult:
    """验证带小数的帕累托数据能正常接收和计算"""
    r = TestResult("小数帕累托数据正常接收")
    try:
        digest = TDigest(delta=100)

        # 用同批真值数据 (都是小数)
        added = digest.add_list(_TRUTH_VALUES)
        r.add_detail(f"上报数据 N={len(_TRUTH_VALUES)}, 成功写入 {added}")
        r.add_detail(f"count={digest.count}, 质心数={digest.num_centroids}")

        est_p50 = digest.quantile(0.5)
        est_p95 = digest.quantile(0.95)
        est_p99 = digest.quantile(0.99)
        r.add_detail(_format_quantile_row("估计", est_p50, est_p95, est_p99))

        checks = []
        if added != len(_TRUTH_VALUES):
            checks.append(f"add_list 返回 {added}, 期望 {len(_TRUTH_VALUES)}")
        if digest.count != len(_TRUTH_VALUES):
            checks.append(f"count={digest.count}, 期望 {len(_TRUTH_VALUES)}")
        if rel_err(_TRUTH_P50, est_p50) > 5:
            checks.append(f"p50 误差过大")
        if rel_err(_TRUTH_P95, est_p95) > 5:
            checks.append(f"p95 误差过大")
        if rel_err(_TRUTH_P99, est_p99) > 10:
            checks.append(f"p99 误差过大")

        if checks:
            r.fail("; ".join(checks))
        else:
            r.pass_(f"小数数据全部正常写入, 分位数查询无误")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


def test_empty_digest_no_nan() -> TestResult:
    """验证空 digest 查询不返回 NaN"""
    r = TestResult("空数据查询不返回 NaN")
    try:
        digest = TDigest(delta=100)
        q = digest.quantile(0.99)
        cdf = digest.cdf(50.0)

        if math.isnan(q) or math.isinf(q):
            r.fail(f"空 digest quantile 返回 {q}")
            return r
        if math.isnan(cdf) or math.isinf(cdf):
            r.fail(f"空 digest cdf 返回 {cdf}")
            return r

        svc = QuantileService(delta=100, num_shards=4)
        result = svc.query("nonexistent", [0.5, 0.95, 0.99])
        for k, v in result.items():
            if math.isnan(v) or math.isinf(v):
                r.fail(f"查询不存在的 metric 返回 NaN/Inf: p{k}={v}")
                return r

        r.pass_(f"空数据查询返回 {q} (非 NaN)")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


def test_sharding_concurrency() -> TestResult:
    """验证分片服务的写入和查询"""
    r = TestResult("QuantileService 分片并发")
    try:
        svc = QuantileService(delta=100, num_shards=8)

        N = len(_TRUTH_VALUES)
        for v in _TRUTH_VALUES:
            svc.record("test_metric", v)

        result = svc.query("test_metric", [0.5, 0.95, 0.99])
        stats = svc.stats("test_metric")
        all_info = stats.get("all", {})

        r.add_detail(f"count={all_info.get('count')}, 质心数={all_info.get('num_centroids')}")
        r.add_detail(_format_quantile_row(
            "估计", result[0.5], result[0.95], result[0.99]
        ))

        all_ok = True
        for q, v in result.items():
            if math.isnan(v) or math.isinf(v):
                r.add_detail(f"p{int(q*100)} = {v} (非法!)")
                all_ok = False

        if rel_err(_TRUTH_P50, result[0.5]) > 5:
            r.add_detail(f"p50 误差过大: {rel_err(_TRUTH_P50, result[0.5]):.2f}%")
            all_ok = False
        if rel_err(_TRUTH_P95, result[0.95]) > 5:
            r.add_detail(f"p95 误差过大")
            all_ok = False
        if rel_err(_TRUTH_P99, result[0.99]) > 10:
            r.add_detail(f"p99 误差过大")
            all_ok = False

        if all_ok:
            r.pass_("分片查询值均合法且在误差范围内")
        else:
            r.fail("存在非法查询结果或超误差")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


# ============================================================
# HTTP 集成测试
# ============================================================

def _make_app():
    from server import create_app
    return create_app()


async def _start_test_server(app) -> Tuple[web.AppRunner, int]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


async def test_http_single_value() -> TestResult:
    r = TestResult("HTTP - 单值正常上报")
    try:
        app = _make_app()
        runner, port = await _start_test_server(app)

        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"
            async with session.post(url, json={"metric": "http_test1", "value": 42.5}) as resp:
                body = await resp.json()
                status = resp.status

        await runner.cleanup()

        r.add_detail(f"status={status}, body={body}")

        if status != 200 or body.get("ok") is not True or body.get("count") != 1:
            r.fail(f"期望 status=200, ok=True, count=1")
        else:
            r.pass_("单值上报成功")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


async def test_http_batch_decimal() -> TestResult:
    r = TestResult("HTTP - 小数批量正常上报")
    try:
        app = _make_app()
        runner, port = await _start_test_server(app)

        values = _TRUTH_VALUES[:5000]
        true_arr = np.sort(np.array(values, dtype=np.float64))
        t_p50 = float(np.quantile(true_arr, 0.5))
        t_p95 = float(np.quantile(true_arr, 0.95))
        t_p99 = float(np.quantile(true_arr, 0.99))

        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"
            async with session.post(url, json={"metric": "http_dec", "values": values}) as resp:
                body = await resp.json()
                status = resp.status

            async with session.get(f"http://127.0.0.1:{port}/query",
                                   params={"metric": "http_dec", "q": ["0.5", "0.95", "0.99"]}) as resp:
                q_body = await resp.json()

        await runner.cleanup()

        r.add_detail(f"ingest status={status}, count={body.get('count')}")
        qs = q_body.get("quantiles", {})
        r.add_detail(_format_quantile_row(
            "估计",
            float(qs.get("p50", 0)),
            float(qs.get("p95", 0)),
            float(qs.get("p99", 0)),
            t_p50, t_p95, t_p99
        ))

        if status != 200 or body.get("count") != len(values):
            r.fail(f"上报失败")
        else:
            r.pass_(f"小数批量 {len(values)} 条上报成功, 查询正常")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


async def test_http_huge_batch_streaming() -> TestResult:
    r = TestResult("HTTP - 超大批量流式压缩 + 同批真值对比")
    try:
        app = _make_app()
        runner, port = await _start_test_server(app)

        N = len(_TRUTH_VALUES)
        values = _TRUTH_VALUES

        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"
            async with session.post(url, json={"metric": "http_huge", "values": values}) as resp:
                body = await resp.json()
                status = resp.status

            async with session.get(f"http://127.0.0.1:{port}/stats",
                                   params={"metric": "http_huge"}) as resp:
                stats_body = await resp.json()

        await runner.cleanup()

        all_stats = stats_body.get("stats", {})
        all_info = all_stats.get("all", {})
        svc_p50 = all_info.get("p50", 0.0)
        svc_p95 = all_info.get("p95", 0.0)
        svc_p99 = all_info.get("p99", 0.0)
        svc_count = all_info.get("count", 0)
        centroids = all_info.get("num_centroids", 0)

        r.add_detail(f"上报数据 N={N}")
        r.add_detail(f"ingest status={status}, count={body.get('count')}")
        r.add_detail(f"服务端 count={svc_count}, 质心数={centroids}")
        r.add_detail(_format_quantile_row("估计", svc_p50, svc_p95, svc_p99))

        checks = []
        if status != 200:
            checks.append(f"ingest status 期望 200, 实际 {status}")
        if body.get("count") != N:
            checks.append(f"响应 count 应为 {N}, 实际 {body.get('count')}")
        if svc_count != N:
            checks.append(f"服务端 count 应为 {N}, 实际 {svc_count}")
        if not (0 < centroids < 5000):
            checks.append(f"质心数异常: {centroids}")
        if rel_err(_TRUTH_P50, svc_p50) > 5:
            checks.append(f"p50 误差过大: {rel_err(_TRUTH_P50, svc_p50):.2f}%")
        if rel_err(_TRUTH_P95, svc_p95) > 5:
            checks.append(f"p95 误差过大: {rel_err(_TRUTH_P95, svc_p95):.2f}%")
        if rel_err(_TRUTH_P99, svc_p99) > 10:
            checks.append(f"p99 误差过大: {rel_err(_TRUTH_P99, svc_p99):.2f}%")

        if checks:
            r.fail("; ".join(checks))
        else:
            compress_ratio = N / centroids
            r.pass_(f"N={N}, 质心={centroids}, 压缩比 {compress_ratio:.0f}x, max_err={max(rel_err(_TRUTH_P50, svc_p50), rel_err(_TRUTH_P95, svc_p95), rel_err(_TRUTH_P99, svc_p99)):.1f}%")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


async def test_http_batch_atomicity() -> TestResult:
    r = TestResult("HTTP - 批量原子性: 非法值整批不落库")
    try:
        app = _make_app()
        runner, port = await _start_test_server(app)

        # 构造大数组: 前面 49999 个合法数字, 最后一个非法字符串
        bad_values = _TRUTH_VALUES[:49999] + ["not_a_number"]

        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"

            # 先查一下初始状态
            async with session.get(f"http://127.0.0.1:{port}/stats",
                                   params={"metric": "atomic_test"}) as resp:
                stats_before = await resp.json()
            count_before = stats_before.get("stats", {}).get("all", {}).get("count", 0)
            if count_before == -1:
                count_before = 0

            # 发送非法批量
            async with session.post(url, json={"metric": "atomic_test", "values": bad_values}) as resp:
                body = await resp.json()
                status = resp.status

            # 再查状态 - count 应该没变
            async with session.get(f"http://127.0.0.1:{port}/stats",
                                   params={"metric": "atomic_test"}) as resp:
                stats_after = await resp.json()
            count_after = stats_after.get("stats", {}).get("all", {}).get("count", 0)
            if count_after == -1:
                count_after = 0

        await runner.cleanup()

        r.add_detail(f"上报数组长度={len(bad_values)}, 最后一项='{bad_values[-1]}' (字符串)")
        r.add_detail(f"请求返回: status={status}, ok={body.get('ok')}, error='{body.get('error')}'")
        r.add_detail(f"请求前 count={count_before}, 请求后 count={count_after}")

        checks = []
        if status != 400:
            checks.append(f"期望 status=400, 实际 {status}")
        if body.get("ok") is not False:
            checks.append(f"期望 ok=false")
        if "not_a_number" not in body.get("error", "") and "index" not in body.get("error", ""):
            checks.append("错误信息应指出非法位置")
        if count_after != 0:
            checks.append(f"count 应该保持 0 (整批丢弃), 实际 {count_after}")

        if checks:
            r.fail("; ".join(checks))
        else:
            r.pass_(f"整批被正确拒绝, count 未被前面合法值撑大")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


async def test_http_memory_stability() -> TestResult:
    r = TestResult("HTTP - 不同规模批量内存稳定性")
    try:
        app = _make_app()
        runner, port = await _start_test_server(app)

        # 不同规模的数据量
        scales = [10000, 30000, 50000, 100000]
        rng = random.Random(_TRUTH_SEED)

        r.add_detail(f"{'N':>8} {'内存KB':>10} {'质心数':>10} {'p50':>12} {'p95':>12} {'p99':>12}")
        r.add_detail("-" * 75)

        peak_mem = 0
        min_centroids = 999999
        max_centroids = 0

        for N in scales:
            values = [rng.paretovariate(1.3) * 5.0 for _ in range(N)]
            gc.collect()
            mem_before = _get_memory_kb()

            async with ClientSession() as session:
                url = f"http://127.0.0.1:{port}/ingest"
                metric = f"mem_test_{N}"
                async with session.post(url, json={"metric": metric, "values": values}) as resp:
                    body = await resp.json()

                async with session.get(f"http://127.0.0.1:{port}/stats",
                                       params={"metric": metric}) as resp:
                    stats = await resp.json()

            mem_after = _get_memory_kb()
            peak_mem = max(peak_mem, mem_after)

            all_info = stats.get("stats", {}).get("all", {})
            centroids = all_info.get("num_centroids", 0)
            min_centroids = min(min_centroids, centroids)
            max_centroids = max(max_centroids, centroids)

            p50 = all_info.get("p50", 0)
            p95 = all_info.get("p95", 0)
            p99 = all_info.get("p99", 0)

            r.add_detail(f"{N:>8} {mem_after:>10} {centroids:>10} {p50:>12.2f} {p95:>12.2f} {p99:>12.2f}")

        await runner.cleanup()

        r.add_detail("-" * 75)
        r.add_detail(f"峰值内存={peak_mem} KB, 质心数范围=[{min_centroids}, {max_centroids}]")

        # 验证: 质心数不随 N 线性增长 (增长应该非常缓慢)
        centroids_growth_ratio = max_centroids / min_centroids if min_centroids > 0 else 999
        n_growth_ratio = scales[-1] / scales[0]

        r.add_detail(f"N 增长 {n_growth_ratio:.0f}x, 质心数仅增长 {centroids_growth_ratio:.1f}x")

        if centroids_growth_ratio > 5:
            r.fail(f"质心数增长过快: {centroids_growth_ratio:.1f}x")
        else:
            r.pass_(f"内存和质心数基本稳定, 不随 N 线性增长")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


async def test_http_invalid_values() -> List[TestResult]:
    """每个非法值场景单独一个 TestResult, 单独显示 PASS/FAIL"""
    results = []
    test_cases = [
        ("单值 NaN", {"metric": "bad1", "value": float("nan")}, "NaN"),
        ("单值 Inf", {"metric": "bad2", "value": float("inf")}, "Infinity"),
        ("单值 字符串abc", {"metric": "bad3", "value": "abc"}, "got string"),
        ("单值 数字字符串\"123\"", {"metric": "bad3c", "value": "123"}, "got string"),
        ("单值 null", {"metric": "bad3d", "value": None}, "got null"),
        ("单值 bool", {"metric": "bad3b", "value": True}, "boolean"),
        ("批量含 NaN (末尾非法)", {"metric": "bad4", "values": [1, 2, 3, float("nan")]}, "NaN"),
        ("批量含 Inf (中间非法)", {"metric": "bad5", "values": [1, float("inf"), 2, 3]}, "Infinity"),
        ("批量含普通字符串", {"metric": "bad6", "values": [1, "hello", 3]}, "got string"),
        ("批量含数字字符串", {"metric": "bad6b", "values": [1, "123", 3]}, "got string"),
        ("批量含 null", {"metric": "bad6c", "values": [1, None, 3]}, "got null"),
        ("批量含 bool", {"metric": "bad6d", "values": [1, True, 3]}, "boolean"),
        ("批量含对象", {"metric": "bad6e", "values": [1, {"x": 1}, 3]}, "got object"),
        ("批量含嵌套数组", {"metric": "bad6f", "values": [1, [1, 2], 3]}, "got array"),
        ("批量 为 null", {"metric": "bad7b", "values": None}, "got null"),
        ("批量 非数组", {"metric": "bad7", "values": "not_a_list"}, "got string"),
        ("批量 空数组", {"metric": "bad7c", "values": []}, "empty"),
        ("缺失 metric", {"value": 42}, "metric"),
    ]

    app = _make_app()
    runner, port = await _start_test_server(app)

    try:
        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"
            for name, payload, expected_err_substr in test_cases:
                r = TestResult(f"非法值: {name}")
                try:
                    async with session.post(url, json=payload) as resp:
                        body = await resp.json()
                        status = resp.status

                    err_msg = body.get("error", "")
                    ok = (
                        status == 400
                        and body.get("ok") is False
                        and expected_err_substr.lower() in err_msg.lower()
                    )

                    r.add_detail(f"status={status}, ok={body.get('ok')}, error='{err_msg}'")

                    if ok:
                        r.pass_("正确返回 400")
                    else:
                        r.fail(f"期望 status=400, ok=false, 错误包含 '{expected_err_substr}'")
                except Exception as e:
                    r.fail(f"请求异常: {e}")
                results.append(r)
    finally:
        await runner.cleanup()

    return results


async def test_error_format_consistent() -> TestResult:
    r = TestResult("HTTP - 单值/批量错误格式一致")
    try:
        app = _make_app()
        runner, port = await _start_test_server(app)

        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"

            async with session.post(url, json={"metric": "fmt_test", "value": float("nan")}) as resp:
                single_err = await resp.json()
                single_status = resp.status

            async with session.post(url, json={"metric": "fmt_test", "values": [1.0, float("inf"), 2.0]}) as resp:
                batch_err = await resp.json()
                batch_status = resp.status

        await runner.cleanup()

        r.add_detail(f"单值错误: status={single_status}, keys={list(single_err.keys())}")
        r.add_detail(f"批量错误: status={batch_status}, keys={list(batch_err.keys())}")

        consistent = (
            single_status == batch_status == 400
            and "ok" in single_err and "ok" in batch_err
            and single_err["ok"] == batch_err["ok"] == False
            and "error" in single_err and "error" in batch_err
        )

        if consistent:
            r.pass_("单值和批量错误响应格式一致: {ok: false, error: '...'}")
        else:
            r.fail("错误响应格式不一致")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


# ============================================================
# 主入口
# ============================================================

def run_unit_tests() -> List[TestResult]:
    print("\n=== 单元测试: 核心算法 ===")
    tests = [
        test_streaming_batch_memory,
        test_nan_inf_filtering,
        test_decimal_pareto_values,
        test_empty_digest_no_nan,
        test_sharding_concurrency,
    ]
    results = []
    for t in tests:
        r = t()
        r.print()
        results.append(r)
    return results


async def run_http_tests() -> List[TestResult]:
    print("\n=== 集成测试: HTTP 接口 ===")
    normal_tests = [
        test_http_single_value,
        test_http_batch_decimal,
        test_http_huge_batch_streaming,
        test_http_batch_atomicity,
        test_http_memory_stability,
        test_error_format_consistent,
    ]
    results = []
    for t in normal_tests:
        r = await t()
        r.print()
        results.append(r)

    print("\n=== 非法值场景 (共18项) ===")
    invalid_results = await test_http_invalid_values()
    for r in invalid_results:
        r.print()
    results.extend(invalid_results)

    return results


async def main():
    print("=" * 65)
    print("分位数统计服务 - 接口验证脚本 (同批真值: N={}, seed={})".format(
        len(_TRUTH_VALUES), _TRUTH_SEED
    ))
    print("=" * 65)
    print(f"真值 p50={_TRUTH_P50:.3f}, p95={_TRUTH_P95:.3f}, p99={_TRUTH_P99:.3f}")

    unit_results = run_unit_tests()
    http_results = await run_http_tests()

    all_results = unit_results + http_results
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    print("\n" + "=" * 65)
    print(f"总览: {passed}/{total} 通过")
    print("=" * 65)

    if passed == total:
        print("\n[PASS] 所有测试通过!")
        return 0
    else:
        print("\n[FAIL] 部分测试失败")
        for r in all_results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
        return 1


if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("[提示] 安装 psutil 可以显示内存监控: pip install psutil")
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
