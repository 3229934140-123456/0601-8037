"""
t-digest: 用于分位数估计的概率数据结构

核心思想:
    将数据点聚合成多个"质心"(centroid), 每个质心包含:
    - mean: 该簇的均值
    - weight: 该簇包含的数据点数量

    在分位数轴(q=0到q=1)上, 质心的大小受"缩放函数"约束:
    - 分布的尾部(q接近0或1)只能有很小的质心(高精度)
    - 分布的中部可以有较大的质心(低精度但省空间)

    这保证了: p99, p99.9等尾部分位数始终有足够的样本支撑精度
"""

from __future__ import annotations

import math
import bisect
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Centroid:
    mean: float
    weight: float

    def merge(self, other: "Centroid") -> None:
        total = self.weight + other.weight
        self.mean = (self.mean * self.weight + other.mean * other.weight) / total
        self.weight = total


class TDigest:
    """
    t-digest实现, 基于Ted Dunning和Otmar Ertl的论文.

    参数:
        delta: 压缩参数, 控制质心数量上限. delta越大, 精度越高但内存越多.
               通常delta=100~1000可获得p99误差<1%
        compression: 别名, 与delta相同
    """

    def __init__(self, delta: float = 100.0) -> None:
        self.delta = delta
        self.centroids: List[Centroid] = []
        self.buffer: List[Tuple[float, float]] = []
        self.buffer_size: int = max(int(delta * 5), 500)
        self.total_weight: float = 0.0

    def _k_scale(self, q: float) -> float:
        """
        k1缩放函数: 将分位数位置q映射到"聚集预算"k.
        这是t-digest的核心: 尾部q≈0和q≈1的区域导数很大,
        意味着分配更多质心(更高精度); 中部导数小, 质心更大(省空间).

        k(q) = (δ/(2π)) * arcsin(2q - 1)
        """
        return self.delta / (2.0 * math.pi) * math.asin(2.0 * q - 1.0)

    def _k_to_q(self, k: float) -> float:
        """k_scale的逆函数"""
        return 0.5 * (math.sin(k * 2.0 * math.pi / self.delta) + 1.0)

    def _max_size(self, q: float) -> float:
        """
        计算在分位数位置q处, 质心允许的最大权重.
        由缩放函数的导数(局部密度)决定.
        """
        return 4.0 * self.total_weight * q * (1.0 - q) / self.delta

    @staticmethod
    def _is_valid_number(x: float) -> bool:
        """检查数值是否合法(非NaN, 非Inf/-Inf)"""
        return not (math.isnan(x) or math.isinf(x))

    def add(self, value: float, weight: float = 1.0) -> bool:
        """
        添加一个数据点(带可选权重).
        返回 True 表示添加成功, False 表示值非法(NaN/Inf)被拒绝.
        流式处理: 缓冲区满了就立即压缩, 内存使用有界.
        """
        if not self._is_valid_number(value) or weight <= 0:
            return False
        self.buffer.append((value, weight))
        self.total_weight += weight
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
        return True

    def add_list(self, values: List[float]) -> int:
        """
        流式批量添加数据点.
        内存不会随values大小线性增长: 每积累 buffer_size 个点就压缩一次.
        返回成功添加的数量(过滤掉 NaN/Inf 后的值).
        """
        added = 0
        for v in values:
            if not self._is_valid_number(v):
                continue
            self.buffer.append((v, 1.0))
            self.total_weight += 1.0
            added += 1
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
        return added

    def add_weighted(self, pairs: List[Tuple[float, float]]) -> int:
        """
        流式批量添加带权重的数据点.
        pairs: [(value, weight), ...]
        返回成功添加的数量.
        """
        added = 0
        for v, w in pairs:
            if not self._is_valid_number(v) or w <= 0:
                continue
            self.buffer.append((v, w))
            self.total_weight += w
            added += 1
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
        return added

    def _flush_buffer(self) -> None:
        """
        将缓冲区中的新数据与现有质心合并.
        这是一个O(n log n)操作, 但在批量执行时摊销成本很低.
        """
        if not self.buffer and not self.centroids:
            return

        pending: List[Tuple[float, float]] = []
        for c in self.centroids:
            pending.append((c.mean, c.weight))
        pending.extend(self.buffer)

        pending.sort(key=lambda x: x[0])

        self.centroids = []
        self.buffer = []

        if not pending:
            return

        cur_mean, cur_w = pending[0]
        weight_so_far = 0.0

        for i in range(1, len(pending)):
            mean, w = pending[i]
            q_lo = weight_so_far / self.total_weight
            q_hi = (weight_so_far + cur_w + w) / self.total_weight
            max_w = self._max_size(0.5 * (q_lo + q_hi))

            if cur_w + w <= max_w:
                total = cur_w + w
                cur_mean = (cur_mean * cur_w + mean * w) / total
                cur_w = total
            else:
                self.centroids.append(Centroid(cur_mean, cur_w))
                weight_so_far += cur_w
                cur_mean, cur_w = mean, w

        self.centroids.append(Centroid(cur_mean, cur_w))

    def quantile(self, q: float) -> float:
        """
        查询q分位数 (q ∈ [0, 1]).
        使用线性插值在质心之间估算.
        若digest为空, 返回 0.0(调用方应通过count判断是否有数据).
        """
        if not self.centroids:
            if self.buffer:
                self._flush_buffer()
            else:
                return 0.0

        if not self.centroids:
            return 0.0

        if q <= 0.0:
            return self.centroids[0].mean
        if q >= 1.0:
            return self.centroids[-1].mean

        target = q * self.total_weight
        cumulative = 0.0

        for i, c in enumerate(self.centroids):
            if cumulative + c.weight >= target:
                if i == 0 or c.weight <= 0:
                    return c.mean
                prev = self.centroids[i - 1]
                offset = target - cumulative
                frac = offset / c.weight
                return prev.mean + frac * (c.mean - prev.mean)
            cumulative += c.weight

        return self.centroids[-1].mean

    def cdf(self, x: float) -> float:
        """
        查询值x对应的累积分布函数值(即P(X <= x)).
        若digest为空, 返回 0.0.
        """
        if not self.centroids:
            if self.buffer:
                self._flush_buffer()
            else:
                return 0.0

        if not self.centroids:
            return 0.0

        if x <= self.centroids[0].mean:
            return 0.0
        if x >= self.centroids[-1].mean:
            return 1.0

        cumulative = 0.0
        for i, c in enumerate(self.centroids):
            if c.mean >= x:
                if i == 0:
                    return 0.0
                prev = self.centroids[i - 1]
                span = c.mean - prev.mean
                if span <= 0:
                    return cumulative / self.total_weight
                frac = (x - prev.mean) / span
                return (cumulative + frac * c.weight) / self.total_weight
            cumulative += c.weight
        return 1.0

    @property
    def count(self) -> int:
        return int(self.total_weight)

    @property
    def num_centroids(self) -> int:
        self._flush_buffer()
        return len(self.centroids)

    def merge(self, other: "TDigest") -> None:
        """合并另一个t-digest"""
        self._flush_buffer()
        other._flush_buffer()
        for c in other.centroids:
            self.buffer.append((c.mean, c.weight))
            self.total_weight += c.weight
        self._flush_buffer()

    def __repr__(self) -> str:
        return (
            f"TDigest(delta={self.delta}, count={self.count}, "
            f"centroids={self.num_centroids})"
        )
