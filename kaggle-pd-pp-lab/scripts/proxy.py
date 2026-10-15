"""Minimal aiohttp proxy used by B (DP2) and D (PD disaggregation).

rr mode : round-robin each /v1/completions request to one of N full replicas.
pd mode : same flow as vLLM's official benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py
          (v0.8.5): send the request to the prefill instance with max_tokens=1 and wait for it
          to finish (the producer's KV is now in the PyNccl lookup buffer), then send the
          ORIGINAL request to the decode instance and stream its SSE output back unchanged.
          Unlike the official proxy, a failed prefill returns HTTP 502 instead of silently
          continuing (which would make the decode instance recompute the prompt).

Per-request timings are appended to --log (JSONL) so TTFT can be decomposed.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


def build_app(mode: str, backends: list[str], prefill: str | None, decode: str | None, log_path: str | None):
    rr = itertools.cycle(range(len(backends))) if backends else None
    state: dict = {}
    log_f = open(log_path, "a", buffering=1, encoding="utf-8") if log_path else None

    def log(rec):
        if log_f:
            log_f.write(json.dumps(rec) + "\n")

    async def on_startup(app):
        state["session"] = ClientSession(connector=TCPConnector(limit=0), timeout=ClientTimeout(total=None))

    async def on_cleanup(app):
        await state["session"].close()
        if log_f:
            log_f.close()

    async def stream_back(request, url, body, headers, rec, t_ref):
        sess: ClientSession = state["session"]
        async with sess.post(url, json=body, headers=headers) as r:
            resp = web.StreamResponse(status=r.status)
            resp.content_type = r.headers.get("Content-Type", "text/event-stream").split(";")[0]
            resp.headers["X-PDLab-Upstream"] = url
            await resp.prepare(request)
            first = True
            async for chunk in r.content.iter_any():
                if first:
                    rec["upstream_first_byte_ms"] = round((time.perf_counter() - t_ref) * 1000, 3)
                    first = False
                await resp.write(chunk)
            await resp.write_eof()
            rec["upstream_status"] = r.status
            return resp

    def fwd_headers(request):
        h = {}
        for k in ("X-Request-Id", "Authorization"):
            if k in request.headers:
                h[k] = request.headers[k]
        return h

    async def completions(request: web.Request):
        t0 = time.perf_counter()
        body = await request.json()
        rid = request.headers.get("X-Request-Id", "")
        rec = {"ts": time.time(), "rid": rid, "mode": mode}
        try:
            if mode == "rr":
                i = next(rr)
                rec["backend"] = i
                resp = await stream_back(request, backends[i] + "/v1/completions", body, fwd_headers(request), rec, t0)
            else:
                pbody = dict(body)
                pbody["max_tokens"] = 1
                pbody["stream"] = False
                pbody.pop("stream_options", None)
                async with state["session"].post(prefill + "/v1/completions", json=pbody,
                                                 headers=fwd_headers(request)) as pr:
                    ptext = await pr.text()
                    rec["prefill_status"] = pr.status
                t1 = time.perf_counter()
                rec["prefill_ms"] = round((t1 - t0) * 1000, 3)
                if pr.status != 200:
                    rec["error"] = "prefill failed: " + ptext[:300]
                    log(rec)
                    return web.json_response({"error": rec["error"]}, status=502)
                try:
                    rec["prefill_prompt_tokens"] = json.loads(ptext).get("usage", {}).get("prompt_tokens")
                except Exception:
                    pass
                resp = await stream_back(request, decode + "/v1/completions", body, fwd_headers(request), rec, t1)
                if "upstream_first_byte_ms" in rec:
                    rec["decode_first_byte_ms"] = rec.pop("upstream_first_byte_ms")
            rec["total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            log(rec)
            return resp
        except (asyncio.CancelledError, ConnectionResetError) as e:
            rec["error"] = f"client disconnected: {type(e).__name__}"
            log(rec)
            raise
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            log(rec)
            return web.json_response({"error": rec["error"]}, status=502)

    async def health(request):
        sess: ClientSession = state["session"]
        targets = backends if mode == "rr" else [prefill, decode]
        for t in targets:
            try:
                async with sess.get(t + "/health", timeout=ClientTimeout(total=5)) as r:
                    if r.status != 200:
                        return web.Response(status=503, text=f"{t} -> {r.status}")
            except Exception as e:
                return web.Response(status=503, text=f"{t} -> {e}")
        return web.Response(text="ok")

    async def models(request):
        t = backends[0] if mode == "rr" else decode
        async with state["session"].get(t + "/v1/models") as r:
            return web.Response(status=r.status, text=await r.text(), content_type="application/json")

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rr", "pd"], required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backends", default="", help="comma-separated base URLs (rr)")
    ap.add_argument("--prefill", default=None)
    ap.add_argument("--decode", default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    backends = [b.rstrip("/") for b in a.backends.split(",") if b]
    if a.mode == "rr" and not backends:
        ap.error("--backends required for rr")
    if a.mode == "pd" and not (a.prefill and a.decode):
        ap.error("--prefill and --decode required for pd")
    app = build_app(a.mode, backends, a.prefill and a.prefill.rstrip("/"), a.decode and a.decode.rstrip("/"), a.log)
    web.run_app(app, host="127.0.0.1", port=a.port, access_log=None, print=lambda *x: print(*x, flush=True))


if __name__ == "__main__":
    main()
