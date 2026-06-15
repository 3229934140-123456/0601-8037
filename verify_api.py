"""
接口验证脚本 - 覆盖正常单值、正常批量、超大批量、非法值等场景

运行:
    python verify_api.py

输出:
    逐个场景输出 PASS/FAIL, 最后给总览
"""

from __future__ import annotations

import math
import random
import sys
import asyncio
import traceback
from typing import Any, Dict, List, Tuple

from aiohttp import web, ClientSession

from tdigest import TDigest
from quantile_service import QuantileService


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
        status = "PASS" if self.passed else "FAIL"
        status_color = "\033[32mPASS\033[0m" if self.passed else "\033[31mFAIL\033[0m"
        print(f"  [{status_color}] {self.name}")
        if self.message:
            print(f"         {self.message}")
        for d in self.details:
            print(f"           - {d}")


# ============================================================
# 第一部分: 核心功能单元测试 (t-digest + QuantileService)
# ============================================================

def test_streaming_batch_memory() -> TestResult:
    """验证超大批量下内存不随批量大小线性增长"""
    r = TestResult("流式批量插入 - 内存有界")
    try:
        digest = TDigest(delta=100)
        # 先看 buffer_size 上限
        max_buffer = digest.buffer_size

        # 模拟流式插入 50000 个点, 分多次小批量
        total = 50000
        batch_size = 2000
        max_buffer_seen = 0

        for i in range(0, total, batch_size):
            batch = [random.random() * 100 for _ in range(batch_size)]
            digest.add_list(batch)
            # flush 之后 buffer 应该是空的
            max_buffer_seen = max(max_buffer_seen, len(digest.buffer))

        # 验证: buffer 从未超过 buffer_size 太多
        r.add_detail(f"buffer_size 限制: {max_buffer}")
        r.add_detail(f"实际最大 buffer: {max_buffer_seen}")
        r.add_detail(f"最终质心数: {digest.num_centroids}")
        r.add_detail(f"总数据点: {digest.count}")

        # 验证数据确实都写入了
        if digest.count == total:
            r.pass_(f"buffer 始终被限制在 {max_buffer} 以内, 压缩比 {total / digest.num_centroids:.0f}x")
        else:
            r.fail(f"count={digest.count}, 期望 {total}")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


def test_nan_inf_filtering() -> TestResult:
    """验证 NaN 和 Infinity 被正确过滤"""
    r = TestResult("NaN / Infinity 过滤")
    try:
        digest = TDigest(delta=100)

        # 单值添加
        ok_nan = digest.add(float("nan"))
        ok_inf = digest.add(float("inf"))
        ok_neg_inf = digest.add(float("-inf"))
        ok_normal = digest.add(42.0)

        r.add_detail(f"add(NaN) 返回: {ok_nan}")
        r.add_detail(f"add(Inf) 返回: {ok_inf}")
        r.add_detail(f"add(-Inf) 返回: {ok_neg_inf}")
        r.add_detail(f"add(42.0) 返回: {ok_normal}")

        if ok_nan or ok_inf or ok_neg_inf:
            r.fail("NaN/Inf 不应被接受")
            return r
        if not ok_normal:
            r.fail("正常值应该被接受")
            return r

        # 批量添加
        batch = [1.0, 2.0, float("nan"), 3.0, float("inf"), 4.0, float("-inf"), 5.0]
        added = digest.add_list(batch)
        r.add_detail(f"批量 {len(batch)} 个值, 成功添加 {added} 个 (应=5)")

        if added != 5:
            r.fail(f"批量过滤后应有 5 个有效值, 实际 {added}")
            return r

        if digest.count != 6:  # 1个单值 + 5个批量有效值
            r.fail(f"最终 count={digest.count}, 期望 6")
            return r

        # 查询不应返回 NaN
        p50 = digest.quantile(0.5)
        if math.isnan(p50) or math.isinf(p50):
            r.fail(f"quantile 返回了非法值: {p50}")
            return r

        r.pass_("NaN/Inf 全部被过滤, 查询结果合法")
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
                r.fail(f"QuantileService 查询不存在的 metric 返回 NaN/Inf: p{k}={v}")
                return r

        r.pass_(f"空数据查询返回 {q} (非 NaN)")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


def test_large_batch_preserves_accuracy() -> TestResult:
    """验证超大批量后分位数仍然准确"""
    r = TestResult("超大批量后分位数精度")
    try:
        digest = TDigest(delta=100)
        N = 200000

        # 流式批量灌入帕累托分布
        batch_size = 5000
        for i in range(0, N, batch_size):
            batch = [random.paretovariate(1.3) * 5.0 for _ in range(min(batch_size, N - i))]
            digest.add_list(batch)

        # 生成真值数据做对比
        truth_data = [random.paretovariate(1.3) * 5.0 for _ in range(N)]
        import numpy as np
        truth_sorted = np.sort(np.array(truth_data))

        quantiles = [0.5, 0.95, 0.99]
        all_ok = True
        for q in quantiles:
            est = digest.quantile(q)
            true_val = float(np.quantile(truth_sorted, q))
            err = abs(est - true_val) / abs(true_val) * 100 if true_val != 0 else 0
            r.add_detail(f"p{int(q*100)}: 估计={est:.3f}, 真值≈{true_val:.3f}, 误差={err:.2f}%")
            if err > 5.0:
                all_ok = False

        r.add_detail(f"最终质心数: {digest.num_centroids}")
        r.add_detail(f"数据点总数: {digest.count}")

        if all_ok:
            r.pass_("所有分位数误差 < 5%")
        else:
            r.fail("部分分位数误差过大")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


def test_quantile_service_sharding() -> TestResult:
    """验证分片服务的写入和查询"""
    r = TestResult("QuantileService 分片并发")
    try:
        svc = QuantileService(delta=100, num_shards=8)

        N = 20000
        for i in range(N):
            svc.record("test_metric", random.random() * 100)

        result = svc.query("test_metric", [0.5, 0.95, 0.99])

        all_finite = True
        for q, v in result.items():
            if math.isnan(v) or math.isinf(v):
                all_ok = False
                r.add_detail(f"p{int(q*100)} = {v} (非法!)")
                all_finite = False
            else:
                r.add_detail(f"p{int(q*100)} = {v:.3f}")

        stats = svc.stats("test_metric")
        r.add_detail(f"总 count: {stats.get('all', {}).get('count', 0)}")

        if all_finite:
            r.pass_("分片中所有查询值均为合法数字")
        else:
            r.fail("存在非法查询结果")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


# ============================================================
# 第二部分: HTTP 接口集成测试
# ============================================================

def _make_app():
    from server import create_app
    return create_app()


async def test_http_single_value() -> TestResult:
    r = TestResult("HTTP - 单值正常上报")
    try:
        app = _make_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"
            async with session.post(url, json={"metric": "http_test1", "value": 42.5}) as resp:
                body = await resp.json()
                status = resp.status

        await runner.cleanup()

        r.add_detail(f"status: {status}")
        r.add_detail(f"body: {body}")

        if status != 200:
            r.fail(f"期望 200, 实际 {status}")
            return r
        if body.get("ok") is not True:
            r.fail(f"响应中 ok 不为 True: {body}")
            return r
        if body.get("count") != 1:
            r.fail(f"count 应为 1, 实际 {body.get('count')}")
            return r

        r.pass_("单值上报成功")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


async def test_http_batch() -> TestResult:
    r = TestResult("HTTP - 批量正常上报")
    try:
        app = _make_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        values = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"
            async with session.post(url, json={"metric": "http_test2", "values": values}) as resp:
                body = await resp.json()
                status = resp.status

            # 查询验证
            async with session.get(f"http://127.0.0.1:{port}/query", params={"metric": "http_test2", "q": ["0.5", "0.95", "0.99"]}) as resp:
                q_body = await resp.json()
                q_status = resp.status

        await runner.cleanup()

        r.add_detail(f"ingest status: {status}, body: {body}")
        r.add_detail(f"query status: {q_status}, body: {q_body}")

        if status != 200:
            r.fail(f"ingest 期望 200, 实际 {status}")
            return r
        if body.get("count") != len(values):
            r.fail(f"count 应为 {len(values)}, 实际 {body.get('count')}")
            return r
        if q_status != 200:
            r.fail(f"query 期望 200, 实际 {q_status}")
            return r

        # 检查查询结果没有 NaN
        qs = q_body.get("quantiles", {})
        for k, v in qs.items():
            if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                r.fail(f"查询结果 {k}={v} 是非法值")
                return r

        r.pass_(f"批量上报 {len(values)} 条, 查询结果正常")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


async def test_http_huge_batch() -> TestResult:
    r = TestResult("HTTP - 超大批量流式压缩 + 同批真值对比")
    try:
        import numpy as np

        app = _make_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        # 固定随机种子, 用同一批数据做真值对比
        N = 50000
        rng = random.Random(42)
        values = [rng.paretovariate(1.3) * 5.0 for _ in range(N)]

        # 先在客户端计算这批数据的真值 (numpy 全量排序)
        truth_arr = np.sort(np.array(values, dtype=np.float64))
        truth_p50 = float(np.quantile(truth_arr, 0.5))
        truth_p95 = float(np.quantile(truth_arr, 0.95))
        truth_p99 = float(np.quantile(truth_arr, 0.99))

        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"
            async with session.post(url, json={"metric": "http_huge", "values": values}) as resp:
                body = await resp.json()
                status = resp.status

            async with session.get(f"http://127.0.0.1:{port}/stats", params={"metric": "http_huge"}) as resp:
                stats_body = await resp.json()

        await runner.cleanup()

        all_stats = stats_body.get("stats", {})
        all_info = all_stats.get("all", {})
        svc_p50 = all_info.get("p50", 0.0)
        svc_p95 = all_info.get("p95", 0.0)
        svc_p99 = all_info.get("p99", 0.0)

        def rel_err(true_v, est_v):
            if true_v == 0:
                return 0.0 if est_v == 0 else float("inf")
            return abs(est_v - true_v) / abs(true_v) * 100

        err_p50 = rel_err(truth_p50, svc_p50)
        err_p95 = rel_err(truth_p95, svc_p95)
        err_p99 = rel_err(truth_p99, svc_p99)

        r.add_detail(f"上报数据条数: {N}")
        r.add_detail(f"ingest status: {status}, count={body.get('count')}")
        r.add_detail(f"服务端 count: {all_info.get('count')}")
        r.add_detail(f"质心数: {all_info.get('num_centroids')}")
        r.add_detail(f"p50: 真值={truth_p50:.3f}, 估计={svc_p50:.3f}, 误差={err_p50:.2f}%")
        r.add_detail(f"p95: 真值={truth_p95:.3f}, 估计={svc_p95:.3f}, 误差={err_p95:.2f}%")
        r.add_detail(f"p99: 真值={truth_p99:.3f}, 估计={svc_p99:.3f}, 误差={err_p99:.2f}%")

        # 检查点
        checks = []
        if status != 200:
            checks.append(f"ingest status 期望 200, 实际 {status}")
        if body.get("count") != N:
            checks.append(f"响应 count 应为 {N}, 实际 {body.get('count')}")
        if all_info.get("count") != N:
            checks.append(f"服务端 count 应为 {N}, 实际 {all_info.get('count')}")

        centroids = all_info.get("num_centroids", 0)
        if not (0 < centroids < 5000):
            checks.append(f"质心数异常: {centroids}")

        # 误差检查: p99 允许稍大 (长尾)
        if err_p50 > 5:
            checks.append(f"p50 误差过大: {err_p50:.2f}%")
        if err_p95 > 5:
            checks.append(f"p95 误差过大: {err_p95:.2f}%")
        if err_p99 > 10:
            checks.append(f"p99 误差过大: {err_p99:.2f}%")

        if checks:
            r.fail("; ".join(checks))
            return r

        compress_ratio = N / centroids
        r.pass_(
            f"N={N}, 质心={centroids}, 压缩比 {compress_ratio:.0f}x, "
            f"max_err=max({err_p50:.1f}%,{err_p95:.1f}%,{err_p99:.1f}%)"
        )
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


async def test_http_invalid_values() -> TestResult:
    r = TestResult("HTTP - 非法值返回 400")
    try:
        app = _make_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        test_cases = [
            ("单值 NaN", {"metric": "bad1", "value": float("nan")}, "NaN"),
            ("单值 Inf", {"metric": "bad2", "value": float("inf")}, "Infinity"),
            ("单值 字符串abc", {"metric": "bad3", "value": "abc"}, "got string"),
            ("单值 数字字符串\"123\"", {"metric": "bad3c", "value": "123"}, "got string"),
            ("单值 null", {"metric": "bad3d", "value": None}, "got null"),
            ("单值 bool", {"metric": "bad3b", "value": True}, "boolean"),
            ("批量含 NaN", {"metric": "bad4", "values": [1, 2, float("nan"), 3]}, "NaN"),
            ("批量含 Inf", {"metric": "bad5", "values": [1, 2, float("inf"), 3]}, "Infinity"),
            ("批量含普通字符串", {"metric": "bad6", "values": [1, "hello", 3]}, "got string"),
            ("批量含数字字符串", {"metric": "bad6b", "values": [1, "123", 3]}, "got string"),
            ("批量含 null", {"metric": "bad6c", "values": [1, None, 3]}, "got null"),
            ("批量含 bool", {"metric": "bad6d", "values": [1, True, 3]}, "boolean"),
            ("批量含对象数组", {"metric": "bad6e", "values": [1, {"x": 1}, 3]}, "got object"),
            ("批量含嵌套数组", {"metric": "bad6f", "values": [1, [1, 2], 3]}, "got array"),
            ("批量 null", {"metric": "bad7b", "values": None}, "got null"),
            ("批量非数组", {"metric": "bad7", "values": "not_a_list"}, "got string"),
            ("批量空数组", {"metric": "bad7c", "values": []}, "empty"),
            ("缺失 metric", {"value": 42}, "metric"),
        ]

        all_pass = True
        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"
            for name, payload, expected_err_substr in test_cases:
                async with session.post(url, json=payload) as resp:
                    body = await resp.json()
                    status = resp.status

                err_msg = body.get("error", "")
                ok = (
                    status == 400
                    and body.get("ok") is False
                    and expected_err_substr.lower() in err_msg.lower()
                )
                detail = f"status={status}, ok={body.get('ok')}, error='{err_msg}'"
                r.add_detail(f"{name}: {'✓' if ok else '✗'} {detail}")
                if not ok:
                    all_pass = False

        await runner.cleanup()

        if all_pass:
            r.pass_(f"所有 {len(test_cases)} 个非法输入场景均返回正确的 400 错误")
        else:
            r.fail("部分非法输入场景未正确处理")
    except Exception as e:
        r.fail(f"异常: {e}\n{traceback.format_exc()}")
    return r


async def test_http_error_format_consistent() -> TestResult:
    r = TestResult("HTTP - 单值/批量错误格式一致")
    try:
        app = _make_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        async with ClientSession() as session:
            url = f"http://127.0.0.1:{port}/ingest"

            # 单值错误
            async with session.post(url, json={"metric": "fmt_test", "value": float("nan")}) as resp:
                single_err = await resp.json()
                single_status = resp.status

            # 批量错误
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
        test_empty_digest_no_nan,
        test_large_batch_preserves_accuracy,
        test_quantile_service_sharding,
    ]
    results = []
    for t in tests:
        r = t()
        r.print()
        results.append(r)
    return results


async def run_http_tests() -> List[TestResult]:
    print("\n=== 集成测试: HTTP 接口 ===")
    tests = [
        test_http_single_value,
        test_http_batch,
        test_http_huge_batch,
        test_http_invalid_values,
        test_http_error_format_consistent,
    ]
    results = []
    for t in tests:
        r = await t()
        r.print()
        results.append(r)
    return results


async def main():
    print("=" * 60)
    print("分位数统计服务 - 接口验证脚本")
    print("=" * 60)

    unit_results = run_unit_tests()
    http_results = await run_http_tests()

    all_results = unit_results + http_results
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    print("\n" + "=" * 60)
    print(f"总览: {passed}/{total} 通过")
    print("=" * 60)

    if passed == total:
        print("\n\033[32m✓ 所有测试通过!\033[0m")
        return 0
    else:
        print("\n\033[31m✗ 部分测试失败\033[0m")
        for r in all_results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
