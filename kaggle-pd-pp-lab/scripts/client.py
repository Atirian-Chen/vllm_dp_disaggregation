"""Streaming /v1/completions client, closed-loop load generator and per-run metrics.

Definitions (client side, wall clock = time.perf_counter):
  TTFT  : request sent -> first SSE event carrying a generated token (choices non-empty).
  ITL   : gap between consecutive token events (pooled over all requests).
          Exact per-token only when events == output tokens; that fraction is reported.
  TPOT  : per request (E2E - TTFT) / (output_tokens - 1).
  E2E   : request sent -> stream finished.
A request is successful only if HTTP 200, [DONE] seen, usage present, prompt_tokens and
completion_tokens equal the requested values, and the generated text is non-blank.
"""
from __future__ import annotations

import asyncio
import json
import time

import aiohttp

from common import mean, percentile


async def one_request(session: aiohttp.ClientSession, url: str, model: str, item: dict, timeout_s: float) -> dict:
    payload = {"model": model, "prompt": item["prompt_token_ids"], "max_tokens": item["max_tokens"],
               "temperature": 0.0, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    rec = {"id": item["id"], "ok": False, "expected_prompt_tokens": len(item["prompt_token_ids"]),
           "expected_output_tokens": item["max_tokens"], "t_send_epoch": time.time()}
    t0 = time.perf_counter()
    chunk_times, texts = [], []
    usage, saw_done, status = None, False, None
    try:
        async with session.post(url + "/v1/completions", json=payload, headers={"X-Request-Id": item["id"]},
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            status = r.status
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
            buf = b""
            async for chunk in r.content.iter_any():
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        saw_done = True
                        continue
                    ev = json.loads(data)
                    if ev.get("usage"):
                        usage = ev["usage"]
                    # vLLM streams one event per generated token, even when the token's text
                    # delta is empty (e.g. an incomplete UTF-8 byte); every choice event counts
                    for ch in ev.get("choices") or []:
                        chunk_times.append(time.perf_counter())
                        texts.append(ch.get("text") or "")
        t_end = time.perf_counter()
        text = "".join(texts)
        rec.update({"http_status": status, "n_chunks": len(chunk_times), "text_head": text[:120],
                    "prompt_tokens": (usage or {}).get("prompt_tokens"),
                    "output_tokens": (usage or {}).get("completion_tokens")})
        problems = []
        if not saw_done:
            problems.append("no [DONE]")
        if usage is None:
            problems.append("no usage")
        if rec["prompt_tokens"] != rec["expected_prompt_tokens"]:
            problems.append(f"prompt_tokens {rec['prompt_tokens']} != {rec['expected_prompt_tokens']}")
        if rec["output_tokens"] != rec["expected_output_tokens"]:
            problems.append(f"output_tokens {rec['output_tokens']} != {rec['expected_output_tokens']}")
        if not chunk_times or not text.strip():
            problems.append("empty output")
        if problems:
            raise RuntimeError("; ".join(problems))
        ttft = (chunk_times[0] - t0) * 1000
        e2e = (t_end - t0) * 1000
        n_out = rec["output_tokens"]
        rec.update({"ok": True, "ttft_ms": round(ttft, 3), "e2e_ms": round(e2e, 3),
                    "tpot_ms": round((e2e - ttft) / (n_out - 1), 4) if n_out > 1 else None,
                    "itl_ms": [round((b - a) * 1000, 3) for a, b in zip(chunk_times, chunk_times[1:])],
                    "chunks_eq_tokens": len(chunk_times) == n_out, "text": text})
    except Exception as e:
        rec.update({"ok": False, "http_status": status, "error": f"{type(e).__name__}: {e}"[:500],
                    "n_token_events": len(chunk_times), "e2e_ms": round((time.perf_counter() - t0) * 1000, 3)})
    rec["t_end_epoch"] = time.time()
    return rec


async def run_closed_loop(url: str, model: str, items: list[dict], concurrency: int, timeout_s: float) -> tuple[list[dict], float, float, float]:
    """Keep exactly `concurrency` requests in flight until all items are done."""
    queue = list(items)
    results: list[dict] = []
    conn = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(connector=conn) as session:
        async def worker():
            while queue:
                it = queue.pop(0)
                results.append(await one_request(session, url, model, it, timeout_s))
        t_start_epoch = time.time()
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0
    order = {it["id"]: i for i, it in enumerate(items)}
    results.sort(key=lambda r: order[r["id"]])
    return results, wall, t_start_epoch, time.time()


def summarize(results: list[dict], wall_s: float) -> dict:
    ok = [r for r in results if r["ok"]]
    itl = [x for r in ok for x in r["itl_ms"]]
    out_tok = sum(r["output_tokens"] for r in ok)
    in_tok = sum(r["prompt_tokens"] for r in ok)

    def pct(vals, q):
        v = percentile(vals, q)
        return round(v, 3) if v is not None else None

    ttft = [r["ttft_ms"] for r in ok]
    tpot = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
    e2e = [r["e2e_ms"] for r in ok]
    m = mean(tpot)
    return {
        "n_requests": len(results), "n_ok": len(ok), "success_rate": round(len(ok) / len(results), 4) if results else 0,
        "wall_s": round(wall_s, 3),
        "req_per_s": round(len(ok) / wall_s, 4), "output_tok_per_s": round(out_tok / wall_s, 2),
        "input_tok_per_s": round(in_tok / wall_s, 2),
        "ttft_p50_ms": pct(ttft, 50), "ttft_p95_ms": pct(ttft, 95), "ttft_mean_ms": round(mean(ttft), 3) if ttft else None,
        "tpot_mean_ms": round(m, 3) if m is not None else None, "tpot_p50_ms": pct(tpot, 50), "tpot_p95_ms": pct(tpot, 95),
        "itl_p50_ms": pct(itl, 50), "itl_p95_ms": pct(itl, 95), "itl_p99_ms": pct(itl, 99), "itl_max_ms": round(max(itl), 3) if itl else None,
        "n_itl_samples": len(itl),
        "e2e_p50_ms": pct(e2e, 50), "e2e_p95_ms": pct(e2e, 95),
        "chunks_eq_tokens_frac": round(sum(r["chunks_eq_tokens"] for r in ok) / len(ok), 4) if ok else None,
        "errors": sorted({r.get("error", "")[:160] for r in results if not r["ok"]})[:5],
    }
