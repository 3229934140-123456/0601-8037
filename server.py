"""
高并发分位数统计服务 - HTTP API

流式写入设计:
    POST /ingest 的超大 values 数组不会被完整加载为 Python 对象列表。
    使用 ijson 从 JSON bytes 中流式提取单个数字, 按批处理写入 digest,
    内存峰值 = HTTP body bytes + 小批量缓存(默认500条) + digest 质心摘要,
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
from typing import Any, Iterator, List, Tuple

import ijson
from aiohttp import web

from quantile_service import QuantileService

routes = web.RouteTableDef()

svc: QuantileService = None

_STREAM_BATCH_SIZE = 500


def _error_response(message: str, status: int = 400) -> web.Response:
    """统一的错误响应格式"""
    return web.json_response({"ok": False, "error": message}, status=status)


def _type_name(v: Any) -> str:
    """返回值的 JSON 语义类型名"""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
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
    严格检查 v 是不是 JSON number (Python int/float, 排除 bool).
    返回 (ok, float_value, error_msg).
    不做任何隐式转换: "123" 字符串直接拒绝.
    """
    if isinstance(v, bool):
        return False, 0.0, "boolean is not a valid number"
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


async def _streaming_ingest_body(
    data_bytes: bytes
) -> Tuple[bool, str, int, str]:
    """
    使用 ijson 流式解析 JSON body.
    - 不构建完整的 values Python 列表
    - 逐元素提取, 校验后按批写入 svc
    返回 (成功?, metric, count, 错误信息)
    """
    buf = io.BytesIO(data_bytes)

    metric: str = ""
    try:
        for prefix, event, value in ijson.parse(buf):
            if prefix == "metric" and event == "string":
                metric = value
                break
            if prefix == "metric":
                return False, "", 0, (
                    "missing or invalid 'metric' (string required)"
                )
    except Exception as e:
        return False, "", 0, f"invalid json body: {e}"

    if not metric:
        return False, "", 0, "missing or invalid 'metric' (string required)"

    buf.seek(0)
    count = 0
    batch: List[float] = []
    idx = -1

    try:
        for item in ijson.items(buf, "values.item"):
            idx += 1
            ok, fv, msg = _check_strict_number(item)
            if not ok:
                return False, metric, 0, (
                    f"invalid values in 'values': index {idx}: {msg}"
                )
            batch.append(fv)
            count += 1
            if len(batch) >= _STREAM_BATCH_SIZE:
                svc.record_batch(metric, batch)
                batch = []
    except ijson.JSONError as e:
        return False, metric, 0, f"invalid json body: {e}"

    if idx == -1:
        buf.seek(0)
        try:
            top_keys = list(ijson.keys(buf, ""))
        except Exception:
            top_keys = []
        if "values" not in top_keys and "value" not in top_keys:
            return False, metric, 0, "missing 'value' or 'values' in body"
        if "values" in top_keys:
            buf.seek(0)
            obj = json.loads(data_bytes)
            vals = obj.get("values")
            if vals is None:
                return False, metric, 0, (
                    "invalid values in 'values': expected array, got null"
                )
            if not isinstance(vals, list):
                return False, metric, 0, (
                    f"'values' must be a list of numbers, got {_type_name(vals)}"
                )
            if len(vals) == 0:
                return False, metric, 0, "'values' cannot be empty"
        return False, metric, 0, "missing 'value' or 'values' in body"

    if batch:
        svc.record_batch(metric, batch)
        batch = []

    if count == 0:
        return False, metric, 0, "'values' cannot be empty"

    return True, metric, count, ""


@routes.post("/ingest")
async def ingest(request: web.Request) -> web.Response:
    content_length = request.content_length
    big_payload = content_length is not None and content_length > 1_000_000

    if big_payload:
        data_bytes = await request.read()
        ok, metric, count, err_msg = await _streaming_ingest_body(data_bytes)
        if not ok:
            return _error_response(err_msg, 400)
        return web.json_response({"ok": True, "count": count})

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
        return web.json_response({"error": "missing 'metric'"}, status=400)

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
        "metric": metric,
        "window": window,
        "quantiles": {f"p{int(q * 100)}" if q == int(q * 100) / 100 else f"{q}": v
                      for q, v in result.items()},
    }
    return web.json_response(resp)


@routes.get("/stats")
async def stats(request: web.Request) -> web.Response:
    metric = request.query.get("metric")
    if not metric:
        return web.json_response({"error": "missing 'metric'"}, status=400)
    return web.json_response({"metric": metric, "stats": svc.stats(metric)})


@routes.get("/metrics")
async def list_metrics(request: web.Request) -> web.Response:
    return web.json_response({"metrics": svc.list_metrics()})


@routes.get("/health")
async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "ts": time.time()})


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
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
