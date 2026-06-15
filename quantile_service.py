"""
并发安全的分位数统计服务

设计策略:
    使用"分片(sharding)"降低锁竞争:
    - 将写入流量分配到N个独立的t-digest分片, 每个分片有自己的锁
    - 查询时合并所有分片得到全局近似结果
    - 这种设计使写入吞吐量随分片数近似线性扩展

窗口策略:
    支持时间窗口(如最近1分钟, 5分钟, 全部历史), 每个窗口独立维护digest
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tdigest import TDigest


@dataclass
class Shard:
    digest: TDigest
    lock: threading.Lock = field(default_factory=threading.Lock)


class WindowedDigest:
    """
    单个时间窗口内的分片digest集合
    """

    def __init__(self, delta: float = 100.0, num_shards: int = 16) -> None:
        self.num_shards = num_shards
        self.shards: List[Shard] = [
            Shard(digest=TDigest(delta=delta)) for _ in range(num_shards)
        ]

    def add(self, value: float, shard_idx: int) -> None:
        shard = self.shards[shard_idx % self.num_shards]
        with shard.lock:
            shard.digest.add(value)

    def add_batch(self, values: List[float], shard_idx: int) -> None:
        shard = self.shards[shard_idx % self.num_shards]
        with shard.lock:
            shard.digest.add_list(values)

    def merge_digest(self, other: TDigest, shard_idx: int) -> None:
        """
        合并一个已经校验好的 TDigest (原子性批量提交用).
        只合并到指定分片, 保证负载均衡.
        """
        shard = self.shards[shard_idx % self.num_shards]
        with shard.lock:
            shard.digest.merge(other)

    def snapshot(self) -> TDigest:
        merged = TDigest(delta=self.shards[0].digest.delta)
        for shard in self.shards:
            with shard.lock:
                merged.merge(shard.digest)
        return merged

    def reset(self) -> None:
        for shard in self.shards:
            with shard.lock:
                shard.digest = TDigest(delta=shard.digest.delta)


class QuantileService:
    """
    高并发分位数统计服务主入口

    使用方法:
        svc = QuantileService()
        svc.record("latency", 42.5)
        result = svc.query("latency", [0.5, 0.95, 0.99])
        print(result)  # {0.5: 23.1, 0.95: 87.5, 0.99: 142.3}
    """

    def __init__(
        self,
        delta: float = 100.0,
        num_shards: int = 16,
        windows: Optional[Dict[str, float]] = None,
    ) -> None:
        self.delta = delta
        self.num_shards = num_shards
        self._shard_counter = 0
        self._counter_lock = threading.Lock()

        if windows is None:
            windows = {
                "1m": 60.0,
                "5m": 300.0,
                "15m": 900.0,
                "all": float("inf"),
            }
        self.windows: Dict[str, float] = windows

        self._metrics: Dict[str, Dict[str, WindowedDigest]] = {}
        self._metric_lock = threading.Lock()

        self._last_rotation: Dict[str, float] = {}
        self._start_time = time.time()

    def _next_shard(self) -> int:
        with self._counter_lock:
            idx = self._shard_counter
            self._shard_counter = (self._shard_counter + 1) % self.num_shards
        return idx

    def _get_or_create(self, metric: str) -> Dict[str, WindowedDigest]:
        if metric in self._metrics:
            return self._metrics[metric]
        with self._metric_lock:
            if metric not in self._metrics:
                wds = {}
                for wname in self.windows:
                    wds[wname] = WindowedDigest(
                        delta=self.delta, num_shards=self.num_shards
                    )
                self._metrics[metric] = wds
                self._last_rotation[metric] = time.time()
            return self._metrics[metric]

    def record(self, metric: str, value: float) -> None:
        """记录单个数据点"""
        wds = self._get_or_create(metric)
        shard_idx = self._next_shard()
        for wd in wds.values():
            wd.add(value, shard_idx)

    def record_batch(self, metric: str, values: List[float]) -> None:
        """批量记录数据点(更高吞吐)"""
        if not values:
            return
        wds = self._get_or_create(metric)
        shard_idx = self._next_shard()
        for wd in wds.values():
            wd.add_batch(values, shard_idx)

    def record_digest(self, metric: str, digest: TDigest) -> None:
        """
        合并一个已经校验好的 TDigest (流式原子性批量提交用).
        用于流式解析场景: 全部校验通过后, 一次性合并到服务端.
        """
        if digest is None or digest.count == 0:
            return
        wds = self._get_or_create(metric)
        shard_idx = self._next_shard()
        for wd in wds.values():
            wd.merge_digest(digest, shard_idx)

    def query(
        self, metric: str, quantiles: List[float], window: str = "all"
    ) -> Dict[float, float]:
        """
        查询指定metric在指定窗口下的多个分位数

        参数:
            metric: 指标名称
            quantiles: 分位数列表, 如 [0.5, 0.95, 0.99]
            window: 窗口名, 如 "1m", "5m", "15m", "all"

        返回:
            {分位数: 估算值} 的字典. 若指标不存在或无数据, 值为 0.0.
        """
        if metric not in self._metrics:
            return {q: 0.0 for q in quantiles}

        self._maybe_rotate(metric)

        wds = self._metrics[metric]
        if window not in wds:
            return {q: 0.0 for q in quantiles}

        digest = wds[window].snapshot()
        if digest.count == 0:
            return {q: 0.0 for q in quantiles}
        return {q: digest.quantile(q) for q in quantiles}

    def query_all_windows(
        self, metric: str, quantiles: List[float]
    ) -> Dict[str, Dict[float, float]]:
        """查询所有窗口的分位数"""
        result: Dict[str, Dict[float, float]] = {}
        for wname in self.windows:
            result[wname] = self.query(metric, quantiles, window=wname)
        return result

    def stats(self, metric: str) -> Dict:
        """获取指标统计信息"""
        if metric not in self._metrics:
            return {}
        self._maybe_rotate(metric)
        wds = self._metrics[metric]
        info: Dict = {}
        for wname, wd in wds.items():
            snap = wd.snapshot()
            info[wname] = {
                "count": snap.count,
                "num_centroids": snap.num_centroids,
                "p50": snap.quantile(0.5),
                "p95": snap.quantile(0.95),
                "p99": snap.quantile(0.99),
            }
        return info

    def list_metrics(self) -> List[str]:
        return list(self._metrics.keys())

    def _maybe_rotate(self, metric: str) -> None:
        """
        轮转时间窗口.
        对非"all"窗口, 过期后重置. 对于有界窗口, 使用"滑动"近似:
        每次过期后清空重新累积. 这是监控系统常用的策略.
        """
        now = time.time()
        last = self._last_rotation.get(metric, self._start_time)
        if now - last < 1.0:
            return

        self._last_rotation[metric] = now
        wds = self._metrics[metric]
        for wname, duration in self.windows.items():
            if duration == float("inf"):
                continue
            if now - last >= duration:
                wds[wname].reset()

    def reset(self, metric: Optional[str] = None) -> None:
        if metric is None:
            for m in list(self._metrics.keys()):
                self.reset(m)
            return
        if metric not in self._metrics:
            return
        for wd in self._metrics[metric].values():
            wd.reset()
