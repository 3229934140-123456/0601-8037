"""
验证脚本: 对比 t-digest vs 朴素随机采样

运行:
    python benchmark.py

该脚本会展示:
1. 正态分布、长尾(帕累托)分布、多峰分布下的精度对比
2. 朴素随机采样为什么在长尾/多峰分布下会严重失真
3. t-digest在高并发写入下的吞吐表现
"""

from __future__ import annotations

import math
import random
import threading
import time
import sys
from typing import List, Tuple

import numpy as np

from tdigest import TDigest
from quantile_service import QuantileService


def ground_truth_quantiles(data: List[float], quantiles: List[float]) -> List[float]:
    """使用numpy计算全量排序后的真实分位数"""
    arr = np.sort(np.array(data, dtype=np.float64))
    return [float(np.quantile(arr, q)) for q in quantiles]


def random_sampling_quantiles(
    data: List[float], quantiles: List[float], sample_size: int
) -> List[float]:
    """
    朴素随机采样方法: 随机抽N个点, 排序取分位数
    这就是"网上经常被提到的朴素做法"
    """
    if sample_size >= len(data):
        return ground_truth_quantiles(data, quantiles)
    sample = random.sample(data, sample_size)
    return ground_truth_quantiles(sample, quantiles)


def reservoir_sampling_quantiles(
    data_stream, quantiles: List[float], sample_size: int
) -> List[float]:
    """
    水库采样(数据流场景下的等价朴素做法)
    """
    reservoir = []
    n = 0
    for x in data_stream:
        n += 1
        if len(reservoir) < sample_size:
            reservoir.append(x)
        else:
            j = random.randint(0, n - 1)
            if j < sample_size:
                reservoir[j] = x
    return ground_truth_quantiles(reservoir, quantiles)


def generate_pareto(shape: float, scale: float, n: int) -> List[float]:
    """
    帕累托(长尾)分布. shape越小尾巴越重.
    典型的延迟分布、收入分布、文件大小分布都是这个形状
    """
    return [scale * (random.random() ** (-1.0 / shape) - 1.0) for _ in range(n)]


def generate_bimodal(n: int) -> List[float]:
    """
    双峰分布: 两个正态分布混合
    模拟"有两类用户": 大部分快, 小部分特别慢
    """
    result = []
    for _ in range(n):
        if random.random() < 0.9:
            result.append(random.gauss(10.0, 2.0))
        else:
            result.append(random.gauss(200.0, 30.0))
    return result


def generate_normal(n: int) -> List[float]:
    return [random.gauss(50.0, 15.0) for _ in range(n)]


def relative_error(true_val: float, est_val: float) -> float:
    if true_val == 0:
        return 0.0 if est_val == 0 else float("inf")
    return abs(est_val - true_val) / abs(true_val)


def run_precision_comparison() -> None:
    print("=" * 70)
    print("精度对比: t-digest vs 朴素随机采样(1%采样率)")
    print("=" * 70)

    QUANTILES = [0.5, 0.95, 0.99, 0.999]
    N = 1_000_000
    SAMPLE_SIZE = N // 100

    scenarios = [
        ("正态分布(N=1M)", generate_normal(N)),
        ("长尾帕累托(shape=1.2, N=1M)", generate_pareto(1.2, 1.0, N)),
        ("双峰分布(90%快+10%慢, N=1M)", generate_bimodal(N)),
    ]

    for name, data in scenarios:
        print(f"\n--- {name} ---")

        true_qs = ground_truth_quantiles(data, QUANTILES)

        digest = TDigest(delta=100.0)
        digest.add_list(data)
        tdigest_qs = [digest.quantile(q) for q in QUANTILES]

        sampling_qs = random_sampling_quantiles(data, QUANTILES, SAMPLE_SIZE)

        print(f"{'Quantile':<10} {'真值':>12} {'t-digest':>12} {'t-digest误差%':>12} "
              f"{'随机采样':>12} {'采样误差%':>12}")
        for i, q in enumerate(QUANTILES):
            t = true_qs[i]
            td = tdigest_qs[i]
            sp = sampling_qs[i]
            td_err = relative_error(t, td) * 100
            sp_err = relative_error(t, sp) * 100
            marker = " ⚠️ 严重失真" if sp_err > 20 else ""
            print(f"p{int(q * 100) if q >= 0.01 else f'{q*100:.1f}':<7} "
                  f"{t:>12.3f} {td:>12.3f} {td_err:>11.2f}% "
                  f"{sp:>12.3f} {sp_err:>11.2f}%{marker}")

        print(f"  t-digest 质心数: {digest.num_centroids}, "
              f"压缩比: {N / digest.num_centroids:.0f}x")


def run_sampling_trap_demo() -> None:
    print("\n" + "=" * 70)
    print("陷阱演示: 为什么朴素采样在长尾/多峰下必然失败")
    print("=" * 70)

    print("""
【核心原理】
设采样率为r, 总体有N个点:
  - 中位数(p50)对应的"真值排位": N/2
    采样后期望排位: r*N/2, 只要r*N足够大, 就比较稳

  - p99对应的"真值排位": N * 0.99 (即在第99%位置的值)
    尾部1%区域总共只有 N*0.01 个点
    采样后该区域期望样本数: r * N * 0.01

设 N=1,000,000, r=1% (即SAMPLE_SIZE=10,000):
  - p50: 采样后中位数附近有 ~5000 个样本点 ✓ 很稳
  - p99: 尾部1%区域只有 ~100 个样本点, 直接用第9900大的那个值
         这100个点的分布方差极大!
  - p999: 尾部0.1%区域只有 ~10 个样本点 ✗ 基本瞎猜

对多峰分布: 如果少数模式占比 < 采样率, 可能整个模式都没被抽到,
分位数直接跳到另一峰, 产生数量级误差.

【下面做20次重复实验, 看误差分布的方差】
""")

    N = 500_000
    SAMPLE_SIZE = 5000  # 1%采样率
    QUANTILES = [0.95, 0.99, 0.999]

    data = generate_pareto(1.2, 1.0, N)
    true_qs = ground_truth_quantiles(data, QUANTILES)

    trials = 20
    tdigest_errs = {q: [] for q in QUANTILES}
    sample_errs = {q: [] for q in QUANTILES}

    for _ in range(trials):
        digest = TDigest(delta=100.0)
        shuffled = data[:]
        random.shuffle(shuffled)
        digest.add_list(shuffled)

        for q in QUANTILES:
            true_v = true_qs[QUANTILES.index(q)]
            tdigest_errs[q].append(relative_error(true_v, digest.quantile(q)) * 100)
            sample_v = random_sampling_quantiles(data, [q], SAMPLE_SIZE)[0]
            sample_errs[q].append(relative_error(true_v, sample_v) * 100)

    print(f"帕累托长尾分布 N={N}, 20次重复实验:")
    print(f"{'分位数':<8} {'t-digest 平均误差%':>18} {'t-digest 最大误差%':>18} "
          f"{'采样 平均误差%':>18} {'采样 最大误差%':>18}")
    for q in QUANTILES:
        print(f"p{int(q * 100) if q >= 0.01 else f'{q*100:.1f}':<7} "
              f"{np.mean(tdigest_errs[q]):>17.2f}% {np.max(tdigest_errs[q]):>17.2f}% "
              f"{np.mean(sample_errs[q]):>17.2f}% {np.max(sample_errs[q]):>17.2f}%")

    print("""
结论:
  越往尾部(p99, p99.9), 朴素采样的误差方差爆炸式增长,
  可能某次误差1%, 下次直接100%+, 完全不可控.
  而t-digest的误差稳定, 始终在几个百分点以内.
""")


def run_concurrency_benchmark() -> None:
    print("=" * 70)
    print("高并发写入吞吐测试")
    print("=" * 70)

    svc = QuantileService(delta=100.0, num_shards=32)

    NUM_THREADS = 32
    OPS_PER_THREAD = 50_000
    TOTAL = NUM_THREADS * OPS_PER_THREAD

    def worker(tid: int) -> None:
        rng = random.Random(tid)
        for i in range(OPS_PER_THREAD):
            v = rng.paretovariate(1.3) * 5.0
            svc.record("request_latency", v)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]

    print(f"线程数={NUM_THREADS}, 每线程写入={OPS_PER_THREAD}, 总计={TOTAL:,} 条")

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.time()

    elapsed = t1 - t0
    throughput = TOTAL / elapsed

    print(f"耗时: {elapsed:.2f}s")
    print(f"吞吐: {throughput:,.0f} ops/sec")
    print(f"统计: {svc.stats('request_latency')}")

    QUANTILES = [0.5, 0.95, 0.99]
    result = svc.query("request_latency", QUANTILES)
    print(f"\n高并发写入后查询:")
    for q in QUANTILES:
        print(f"  p{int(q * 100):<3} = {result[q]:.3f}")


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    try:
        run_precision_comparison()
        run_sampling_trap_demo()
        run_concurrency_benchmark()
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()
