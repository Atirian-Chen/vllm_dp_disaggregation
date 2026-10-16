"""LOCAL TESTING ONLY - a tiny fake of vLLM's OpenAI completions server.

Used to exercise the client, proxy, launcher, runner and report generator on a
machine without a GPU. It never produces numbers that go into a real report:
every run made with --mock is tagged "mock": true and analyze_results.py refuses to
mix mock and real runs.

Roles: mixed | producer | consumer. producer/consumer emulate the PD hand-off with a
file registry; like vLLM 0.8.5's SimpleBuffer, a consumer request whose KV never arrives waits forever.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time

from aiohttp import web

REG = os.path.join(tempfile.gettempdir(), "pdlab_mock_kv")


def key(ids) -> str:
    return hashlib.sha1(json.dumps(ids).encode()).hexdigest()


def probe(rec):
    p = os.environ.get("PDLAB_PROBE_LOG")
    if p:
        rec.update({"ts": time.time(), "pid": os.getpid(), "mock": True})
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")


def build(role: str, prefill_ms_per_tok: float, decode_ms: float):
    os.makedirs(REG, exist_ok=True)
    lock = asyncio.Lock()  # one "GPU": serialize prefill compute

    async def completions(req: web.Request):
        body = await req.json()
        ids = body["prompt"]
        n_in, n_out = len(ids), int(body["max_tokens"])
        k = key(ids)
        skip_prefill = False
        if role == "consumer":
            path = os.path.join(REG, k)
            probe({"ev": "select_begin", "n_tokens": n_in})
            while not os.path.exists(path):  # like vLLM 0.8.5: wait for the producer, never "miss"
                await asyncio.sleep(0.05)
            n = n_in * 28672
            probe({"ev": "select", "hit": True, "n_tokens": n_in, "ms": 2.0, "matched_tokens": n_in,
                   "bytes_k": n // 2, "bytes_v": n // 2, "bytes_hidden": n_in * 3072})
            probe({"ev": "recv_batch", "bypass_model_exec": True, "n_reqs": 1, "n_tokens": n_in, "ms": 3.0})
            skip_prefill = True
            os.remove(path)
        async with lock:
            if not skip_prefill:
                await asyncio.sleep(n_in * prefill_ms_per_tok / 1000)
        if role == "producer":
            open(os.path.join(REG, k), "w").close()
            probe({"ev": "insert", "n_tokens": n_in, "bytes_k": n_in * 14336, "bytes_v": n_in * 14336,
                   "bytes_hidden": n_in * 3072, "ms": 0.2})
        words = [f" w{i}" for i in range(n_out)]
        usage = {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}
        if not body.get("stream"):
            await asyncio.sleep(decode_ms / 1000 * (n_out - 1))
            return web.json_response({"id": "cmpl-mock", "object": "text_completion", "model": body["model"],
                                      "choices": [{"index": 0, "text": "".join(words), "finish_reason": "length"}],
                                      "usage": usage})
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        for i, w in enumerate(words):
            if i:
                await asyncio.sleep(decode_ms / 1000)
            ev = {"id": "cmpl-mock", "object": "text_completion", "model": body["model"],
                  "choices": [{"index": 0, "text": w, "finish_reason": "length" if i == n_out - 1 else None}],
                  "usage": None}
            await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
        if (body.get("stream_options") or {}).get("include_usage"):
            await resp.write(f"data: {json.dumps({'id': 'cmpl-mock', 'choices': [], 'usage': usage})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def health(_):
        return web.Response(text="")

    async def models(_):
        return web.json_response({"data": [{"id": "pdlab"}]})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--role", default="mixed")
    ap.add_argument("--prefill-ms-per-tok", type=float, default=0.02)
    ap.add_argument("--decode-ms", type=float, default=2.0)
    a = ap.parse_args()
    print("Model loading took 2.9 GiB (MOCK)", flush=True)
    web.run_app(build(a.role, a.prefill_ms_per_tok, a.decode_ms), host="127.0.0.1", port=a.port, access_log=None)
