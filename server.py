"""
高并发分位数统计服务 - HTTP API

流式写入设计:
    POST /ingest 的超大 values 数组:
    1. 使用 aiohttp 流式读取 request body, 边读边解析 (不攒完整 bytes)
    2. 使用 ijson 增量解析 JSON, 逐个取出 values.item
    3. 先校验并写入临时 TDigest (全内存校验阶段)
    4. 全部校验通过后, 把临时 TDigest merge 到服务端的分片 digest
    5. 任何一项非法 → 临时 TDigest 直接丢弃, 服务端数据 untouched

内存峰值 = 网络接收缓冲区 + 流式批量缓存(500条) + 临时 TDigest 质心(≈几百个)
          + 服务端分片 digest 质心摘要
不会随 values 条数线性增长.

启动:
    python server.py

API:
    POST /ingest
        Body: {"metric": "latency", "value": 42.5}
        或批量: {"metric": "latency", "values": [1.2, 3.4, 5.6]}

    GET /query?metric=latency&q=0.5&q=0.95&q=0.99&window=all

    GET /stats?metric=latency

    GET /metrics

    GET /health
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from decimal import Decimal
from typing import Any, AsyncGenerator, List, Tuple

import ijson
from aiohttp import web

from tdigest import TDigest
from quantile_service import QuantileService

routes = web.RouteTableDef()

svc: QuantileService = None

_STREAM_BATCH_SIZE = 500
_BIG_PAYLOAD_THRESHOLD = 500_000  # 500KB 以上走流式


def _error_response(message: str, status: int = 400) -> web.Response:
    """统一的错误响应格式"""
    return web.json_response({"ok": False, "error": message}, status=status)


def _type_name(v: Any) -> str:
    """返回值的 JSON 语义类型名"""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float, Decimal)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _check_strict_number(v: Any) -> Tuple[bool, float, str]:
    """
    严格检查 v 是不是 JSON number (Python int/float/Decimal, 排除 bool).
    返回 (ok, float_value, error_msg).
    不做任何隐式转换: "123" 字符串直接拒绝.
    Decimal 来自 ijson 流式解析, 是合法的 JSON number.
    """
    if isinstance(v, bool):
        return False, 0.0, "boolean is not a valid number"
    if isinstance(v, Decimal):
        try:
            f = float(v)
        except Exception:
            return False, 0.0, "cannot convert Decimal to float"
        if math.isnan(f):
            return False, 0.0, "NaN is not allowed"
        if math.isinf(f):
            return False, 0.0, "Infinity is not allowed"
        return True, f, ""
    if not isinstance(v, (int, float)):
        return False, 0.0, f"expected number, got {_type_name(v)}"
    f = float(v)
    if math.isnan(f):
        return False, 0.0, "NaN is not allowed"
    if math.isinf(f):
        return False, 0.0, "Infinity is not allowed"
    return True, f, ""


def _validate_and_convert_strict(raw_list: Any) -> Tuple[List[float], List[str]]:
    """
    严格校验 values 列表: 只接受真正的 JSON number.
    不把 "123" 这种字符串当数字.
    返回 (转换后的 float 列表, 错误信息列表).
    """
    values: List[float] = []
    errors: List[str] = []
    for i, v in enumerate(raw_list):
        ok, fv, msg = _check_strict_number(v)
        if not ok:
            errors.append(f"index {i}: {msg}")
            continue
        values.append(fv)
    return values, errors


class _AsyncByteStream:
    """
    适配器: 把 aiohttp StreamReader 包装成带异步 read 方法的文件类对象.
    ijson.parse_async 需要 await f.read(n) 接口.
    """

    def __init__(self, content):
        self._content = content
        self._buffer = b""
        self._eof = False

    async def read(self, n: int = -1) -> bytes:
        """异步读接口, 供 ijson.parse_async 调用."""
        if self._eof and not self._buffer:
            return b""

        if n == -1:
            while not self._eof:
                chunk = await self._content.readany()
                if not chunk:
                    self._eof = True
                    break
                self._buffer += chunk
            result = self._buffer
            self._buffer = b""
            return result

        while len(self._buffer) < n and not self._eof:
            chunk = await self._content.readany()
            if not chunk:
                self._eof = True
                break
            self._buffer += chunk

        result = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return result


async def _stream_and_validate(
    content,
) -> Tuple[bool, str, TDigest, str]:
    """
    真正的流式处理: 边读 request body 边解析边校验.
    校验通过返回 (True, metric, 临时digest, "")
    校验失败返回 (False, metric, 空digest, 错误信息)

    关键: 全部校验通过后才会把临时 digest merge 到服务端.
          中间任何一项非法 → 整批丢弃, 服务端数据 untouched.
    """
    stream = _AsyncByteStream(content)
    pending_digest = TDigest(delta=svc.delta if svc else 100.0)

    metric = ""
    metric_seen = False
    values_seen = False
    has_value_single = False
    idx = -1

    try:
        async for prefix, event, value in ijson.parse_async(stream):
            if prefix == "metric" and event == "string":
                metric = value
                metric_seen = True
                continue
            if prefix == "metric" and event != "string":
                return False, "", TDigest(), (
                    "missing or invalid 'metric' (string required)"
                )

            if prefix == "value":
                has_value_single = True
                if not metric_seen:
                    return False, "", TDigest(), (
                        "missing or invalid 'metric' (string required)"
                    )
                ok, fv, msg = _check_strict_number(value)
                if not ok:
                    return False, metric, TDigest(), f"invalid 'value': {msg}"
                tmp = TDigest(delta=svc.delta if svc else 100.0)
                tmp.add(fv)
                return True, metric, tmp, ""

            if prefix == "values":
                if event == "start_array":
                    values_seen = True
                    continue
                if event == "null":
                    return False, metric if metric_seen else "", TDigest(), (
                        "'values' must be a list of numbers, got null"
                    )
                if event not in ("start_array", "end_array", "map_key"):
                    t = _type_name(value)
                    return False, metric if metric_seen else "", TDigest(), (
                        f"'values' must be a list of numbers, got {t}"
                    )

            if prefix == "values.item":
                if not metric_seen:
                    return False, "", TDigest(), (
                        "missing or invalid 'metric' (string required)"
                    )
                idx += 1
                ok, fv, msg = _check_strict_number(value)
                if not ok:
                    return False, metric, TDigest(), (
                        f"invalid values in 'values': index {idx}: {msg}"
                    )
                pending_digest.add(fv)

    except ijson.JSONError as e:
        return False, metric if metric_seen else "", TDigest(), (
            f"invalid json body: {e}"
        )

    if not metric_seen:
        return False, "", TDigest(), (
            "missing or invalid 'metric' (string required)"
        )

    if has_value_single:
        return True, metric, pending_digest, ""

    if not values_seen:
        return False, metric, TDigest(), (
            "missing 'value' or 'values' in body"
        )

    if idx == -1:
        return False, metric, TDigest(), "'values' cannot be empty"

    return True, metric, pending_digest, ""


@routes.post("/ingest")
async def ingest(request: web.Request) -> web.Response:
    content_length = request.content_length
    big_payload = (
        content_length is None or content_length > _BIG_PAYLOAD_THRESHOLD
    )

    if big_payload:
        # 真正的流式处理: 边读边校验, 不攒完整 body
        ok, metric, pending_digest, err_msg = await _stream_and_validate(
            request.content
        )
        if not ok:
            return _error_response(err_msg, 400)

        count = pending_digest.count
        if count == 0:
            return _error_response("no valid values received", 400)

        # 全部校验通过, 才把临时 digest 合并到服务端
        svc.record_digest(metric, pending_digest)
        return web.json_response({"ok": True, "count": count})

    # 小 payload: 走原来的快速路径 (完整 JSON 解析)
    try:
        data = await request.json()
    except Exception:
        return _error_response("invalid json body", 400)

    if not isinstance(data, dict):
        return _error_response("body must be a JSON object", 400)

    metric = data.get("metric")
    if not metric or not isinstance(metric, str):
        return _error_response(
            "missing or invalid 'metric' (string required)", 400
        )

    if "values" in data:
        raw_values = data["values"]
        if raw_values is None:
            return _error_response(
                "'values' must be a list of numbers, got null", 400
            )
        if not isinstance(raw_values, list):
            return _error_response(
                f"'values' must be a list of numbers, got {_type_name(raw_values)}",
                400,
            )

        values, errors = _validate_and_convert_strict(raw_values)
        if errors:
            return _error_response(
                f"invalid values in 'values': {'; '.join(errors)}", 400
            )

        if not values:
            return _error_response("'values' cannot be empty", 400)

        svc.record_batch(metric, values)
        return web.json_response({"ok": True, "count": len(values)})

    if "value" in data:
        raw_value = data["value"]
        ok, fv, msg = _check_strict_number(raw_value)
        if not ok:
            return _error_response(f"invalid 'value': {msg}", 400)
        svc.record(metric, fv)
        return web.json_response({"ok": True, "count": 1})

    return _error_response("missing 'value' or 'values' in body", 400)


@routes.get("/query")
async def query(request: web.Request) -> web.Response:
    metric = request.query.get("metric")
    if not metric:
        return web.json_response(
            {"ok": False, "error": "missing 'metric'"}, status=400
        )

    qs_raw: List[str] = request.query.getall("q", [])
    if not qs_raw:
        qs_raw = ["0.5", "0.95", "0.99"]

    quantiles = []
    for qs in qs_raw:
        try:
            q = float(qs)
            if 0.0 <= q <= 1.0:
                quantiles.append(q)
        except ValueError:
            pass

    if not quantiles:
        quantiles = [0.5, 0.95, 0.99]

    window = request.query.get("window", "all")
    result = svc.query(metric, quantiles, window=window)

    resp = {
        "ok": True,
        "metric": metric,
        "window": window,
        "count": svc.stats(metric).get(window, {}).get("count", 0),
        "quantiles": {
            f"p{int(q * 100)}" if q == int(q * 100) / 100 else f"{q}": v
            for q, v in result.items()
        },
    }
    return web.json_response(resp)


@routes.get("/stats")
async def stats(request: web.Request) -> web.Response:
    metric = request.query.get("metric")
    if not metric:
        return web.json_response(
            {"ok": False, "error": "missing 'metric'"}, status=400
        )
    return web.json_response({"ok": True, "metric": metric, "stats": svc.stats(metric)})


@routes.get("/metrics")
async def list_metrics(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "metrics": svc.list_metrics()})


@routes.get("/health")
async def health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "status": "ok", "ts": time.time()})


def create_app() -> web.Application:
    global svc
    svc = QuantileService(delta=100.0, num_shards=32)
    app = web.Application()
    app.add_routes(routes)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="高并发分位数统计服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--delta", type=float, default=100.0, help="t-digest压缩参数")
    parser.add_argument("--shards", type=int, default=32, help="分片数, 降低锁竞争")
    args = parser.parse_args()

    global svc
    svc = QuantileService(delta=args.delta, num_shards=args.shards)

    app = web.Application()
    app.add_routes(routes)

    print(f"[QuantileService] starting on {args.host}:{args.port}")
    print(f"[QuantileService] delta={args.delta}, shards={args.shards}")
    print(f"[QuantileService] 流式阈值: {_BIG_PAYLOAD_THRESHOLD} bytes")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
