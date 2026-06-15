"""
高并发分位数统计服务 - HTTP API

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
import json
import math
import time
from typing import List, Tuple

from aiohttp import web

from quantile_service import QuantileService

routes = web.RouteTableDef()

svc: QuantileService = None


def _error_response(message: str, status: int = 400) -> web.Response:
    """统一的错误响应格式"""
    return web.json_response({"ok": False, "error": message}, status=status)


def _is_valid_number(x) -> bool:
    """检查是否是合法的有限数字"""
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    f = float(x)
    return not (math.isnan(f) or math.isinf(f))


def _validate_float_list(raw_list) -> Tuple[List[float], List[str]]:
    """
    校验并转换 float 列表.
    返回 (合法值列表, 错误信息列表).
    """
    values: List[float] = []
    errors: List[str] = []
    for i, v in enumerate(raw_list):
        if isinstance(v, bool):
            errors.append(f"index {i}: boolean is not a valid number")
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            errors.append(f"index {i}: cannot convert '{v}' to number")
            continue
        if math.isnan(fv):
            errors.append(f"index {i}: NaN is not allowed")
            continue
        if math.isinf(fv):
            errors.append(f"index {i}: Infinity is not allowed")
            continue
        values.append(fv)
    return values, errors


@routes.post("/ingest")
async def ingest(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return _error_response("invalid json body", 400)

    if not isinstance(data, dict):
        return _error_response("body must be a JSON object", 400)

    metric = data.get("metric")
    if not metric or not isinstance(metric, str):
        return _error_response("missing or invalid 'metric' (string required)", 400)

    if "values" in data:
        raw_values = data["values"]
        if not isinstance(raw_values, list):
            return _error_response("'values' must be a list of numbers", 400)

        values, errors = _validate_float_list(raw_values)
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
        if not _is_valid_number(raw_value):
            if isinstance(raw_value, bool):
                return _error_response("invalid 'value': boolean is not a valid number", 400)
            try:
                fv = float(raw_value)
                if math.isnan(fv):
                    return _error_response("invalid 'value': NaN is not allowed", 400)
                if math.isinf(fv):
                    return _error_response("invalid 'value': Infinity is not allowed", 400)
            except (TypeError, ValueError):
                pass
            return _error_response("invalid 'value': must be a finite number", 400)

        svc.record(metric, float(raw_value))
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
