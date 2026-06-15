"""
HTTP压测客户端 - 用于测试server.py的高并发性能

用法:
    1. 启动服务端: python server.py --port 8080
    2. 另开终端运行: python http_benchmark.py --port 8080 --clients 100 --requests 10000
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from typing import List

import aiohttp


async def worker(
    session: aiohttp.ClientSession,
    url: str,
    metric: str,
    num_requests: int,
    results: List[float],
    errors: List[int],
) -> None:
    for _ in range(num_requests):
        value = random.paretovariate(1.3) * 5.0
        try:
            t0 = time.time()
            async with session.post(
                url, json={"metric": metric, "value": value}
            ) as resp:
                await resp.read()
            results.append(time.time() - t0)
        except Exception:
            errors.append(1)


async def query_worker(
    session: aiohttp.ClientSession,
    url: str,
    metric: str,
    num_requests: int,
) -> None:
    for _ in range(num_requests):
        try:
            async with session.get(
                f"{url.replace('/ingest', '/query')}",
                params={"metric": metric, "q": ["0.5", "0.95", "0.99"]},
            ) as resp:
                await resp.read()
        except Exception:
            pass


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--clients", type=int, default=50)
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--metric", default="bench_latency")
    args = parser.parse_args()

    ingest_url = f"{args.host}:{args.port}/ingest"
    total = args.clients * args.requests

    print(f"[HTTP Benchmark] {args.clients} 并发客户端, 每客户端 {args.requests} 次请求, 总计 {total:,} 次")
    print(f"[HTTP Benchmark] 目标: {ingest_url}")

    results: List[float] = []
    errors: List[int] = []

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [
            worker(session, ingest_url, args.metric, args.requests, results, errors)
            for _ in range(args.clients)
        ]
        await asyncio.gather(*tasks)
    t1 = time.time()

    elapsed = t1 - t0
    throughput = total / elapsed

    print(f"\n--- 写入结果 ---")
    print(f"成功请求: {len(results):,}")
    print(f"错误请求: {len(errors):,}")
    print(f"耗时: {elapsed:.2f}s")
    print(f"吞吐: {throughput:,.0f} req/sec")

    if results:
        results.sort()
        def pct(p):
            idx = min(int(len(results) * p), len(results) - 1)
            return results[idx] * 1000
        print(f"\n请求延迟 (ms):")
        print(f"  p50 = {pct(0.5):.2f}")
        print(f"  p95 = {pct(0.95):.2f}")
        print(f"  p99 = {pct(0.99):.2f}")

    print(f"\n--- 查询服务端统计 ---")
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{args.host}:{args.port}/stats", params={"metric": args.metric}
        ) as resp:
            data = await resp.json()
            print(data)


if __name__ == "__main__":
    asyncio.run(main())
