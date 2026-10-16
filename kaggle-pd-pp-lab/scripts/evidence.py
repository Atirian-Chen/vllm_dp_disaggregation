"""Evidence extraction from server logs, PD probe logs and proxy logs.

Log strings are those of vLLM 0.8.5 V0:
  "Initializing a V0 LLM engine (v0.8.5.post1) with config: ... tensor_parallel_size=1, pipeline_parallel_size=2, ..."
  "Model loading took %.4f GiB and %.6f seconds"   (once per worker / pipeline stage)
  "Using XFormers backend."
  "Initializing PyNcclConfig under kv_transfer_config ..."
  "[rankN]: Failed to receive all KVs and hidden states, redo model forwarding."   (WARNING)
  "[rankN]: Successfully received all KVs and hidden states, skip model forwarding." (DEBUG)
"""
from __future__ import annotations

import re
from pathlib import Path

from common import mean, percentile, read_jsonl

PATTERNS = {
    "engine_v0": re.compile(r"Initializing a V0 LLM engine \(v([^)]+)\)"),
    "engine_v1": re.compile(r"Initializing a V1 LLM engine"),
    "tp": re.compile(r"tensor_parallel_size=(\d+)"),
    "pp": re.compile(r"pipeline_parallel_size=(\d+)"),
    "model_loading_gib": re.compile(r"Model loading took ([\d.]+) ?GiB"),
    "xformers": re.compile(r"Using XFormers backend"),
    "attn_backend": re.compile(r"Using (\w[\w ]*?) backend\."),
    "pynccl_kv": re.compile(r"Initializing PyNcclConfig under kv_transfer_config"),
    "kv_fail": re.compile(r"Failed to receive all KVs and hidden states"),
    "kv_success": re.compile(r"Successfully received all KVs and hidden states"),
    "cuda_graph": re.compile(r"Graph capturing finished in ([\d.]+) sec"),
    "kv_cache_blocks": re.compile(r"# (?:cuda|CUDA|GPU) blocks: (\d+)"),
    "error": re.compile(r"(Traceback \(most recent call last\)|\bERROR\b|CUDA out of memory)"),
}


def read_from(path: str | Path, offset: int = 0) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            return f.read()
    except FileNotFoundError:
        return ""


def log_offsets(state: dict) -> dict:
    out = {}
    for p in state.get("procs", []):
        try:
            out[p["name"]] = Path(p["log"]).stat().st_size
        except FileNotFoundError:
            out[p["name"]] = 0
    return out


def scan_text(text: str) -> dict:
    r = {}
    m = PATTERNS["engine_v0"].search(text)
    r["engine"] = f"V0 {m.group(1)}" if m else ("V1" if PATTERNS["engine_v1"].search(text) else None)
    for k in ("tp", "pp"):
        m = PATTERNS[k].search(text)
        r[k] = int(m.group(1)) if m else None
    r["model_loading_gib"] = [float(x) for x in PATTERNS["model_loading_gib"].findall(text)]
    r["attention_backends"] = sorted(set(PATTERNS["attn_backend"].findall(text)))
    r["pynccl_kv_init"] = bool(PATTERNS["pynccl_kv"].search(text))
    r["kv_fail_count"] = len(PATTERNS["kv_fail"].findall(text))
    r["kv_success_count"] = len(PATTERNS["kv_success"].findall(text))
    r["cuda_graph_s"] = [float(x) for x in PATTERNS["cuda_graph"].findall(text)]
    r["kv_cache_blocks"] = [int(x) for x in PATTERNS["kv_cache_blocks"].findall(text)]
    r["error_lines"] = [ln[:300] for ln in text.splitlines() if PATTERNS["error"].search(ln)][:10]
    return r


def scan_server_logs(state: dict, offsets: dict | None = None) -> dict:
    """Per process: scan its log from `offsets[name]` (or from this launch)."""
    out = {}
    for p in state.get("procs", []):
        off = (offsets or {}).get(p["name"], p.get("log_offset", 0))
        out[p["name"]] = scan_text(read_from(p["log"], off))
    return out


def _window(recs, t0, t1):
    return [r for r in recs if t0 <= r.get("ts", 0) <= t1]


def pd_probe_stats(producer_log: str | None, consumer_log: str | None, t0: float, t1: float) -> dict:
    cons = _window(read_jsonl(consumer_log), t0, t1) if consumer_log else []
    prod = _window(read_jsonl(producer_log), t0, t1) if producer_log else []
    sel = [r for r in cons if r.get("ev") == "select"]
    sel_begin = [r for r in cons if r.get("ev") == "select_begin"]
    hits = [r for r in sel if r.get("hit")]
    rb = [r for r in cons if r.get("ev") == "recv_batch"]
    ins = [r for r in prod if r.get("ev") == "insert"]
    sb = [r for r in prod if r.get("ev") == "send_batch"]
    kv_bytes = [r.get("bytes_k", 0) + r.get("bytes_v", 0) for r in hits]
    hid_bytes = [r.get("bytes_hidden", 0) for r in hits]
    tokens = [r.get("matched_tokens", 0) for r in hits]
    sel_ms = [r["ms"] for r in hits]

    def r3(x):
        return round(x, 3) if x is not None else None
    total_bytes = sum(kv_bytes) + sum(hid_bytes)
    total_ms = sum(sel_ms)
    return {
        "n_select": len(sel), "n_select_hit": len(hits), "n_select_miss": len(sel) - len(hits),
        "n_select_begin": len(sel_begin), "n_select_pending": max(0, len(sel_begin) - len(sel)),
        "n_recv_batches": len(rb), "n_recv_bypass_true": sum(1 for r in rb if r.get("bypass_model_exec")),
        "n_recv_bypass_false": sum(1 for r in rb if not r.get("bypass_model_exec")),
        "n_producer_inserts": len(ins), "producer_send_batch_ms_mean": r3(mean([r["ms"] for r in sb])),
        "kv_bytes_total": sum(kv_bytes), "hidden_bytes_total": sum(hid_bytes),
        "kv_bytes_per_req_mean": r3(mean(kv_bytes)), "kv_bytes_per_token": r3(sum(kv_bytes) / sum(tokens)) if sum(tokens) else None,
        "transfer_ms_p50": r3(percentile(sel_ms, 50)), "transfer_ms_p95": r3(percentile(sel_ms, 95)),
        "transfer_ms_mean": r3(mean(sel_ms)),
        "recv_batch_ms_p50": r3(percentile([r["ms"] for r in rb], 50)),
        "effective_GBps": r3(total_bytes / (total_ms / 1000) / 1e9) if total_ms > 0 else None,
        "k_shape_example": hits[0].get("k_shape") if hits else None,
        "mock": any(r.get("mock") for r in cons),
    }


def proxy_stats(proxy_log: str | None, t0: float, t1: float) -> dict:
    recs = _window(read_jsonl(proxy_log), t0, t1) if proxy_log else []
    if not recs:
        return {"n": 0}

    def pc(key, q):
        v = percentile([r[key] for r in recs if r.get(key) is not None], q)
        return round(v, 3) if v is not None else None
    out = {"n": len(recs), "n_error": sum(1 for r in recs if r.get("error"))}
    if recs[0].get("mode") == "pd":
        for k in ("prefill_ms", "decode_first_byte_ms", "total_ms"):
            out[f"{k}_p50"], out[f"{k}_p95"] = pc(k, 50), pc(k, 95)
    else:
        b = {}
        for r in recs:
            b[str(r.get("backend"))] = b.get(str(r.get("backend")), 0) + 1
        out["per_backend"] = b
        out["upstream_first_byte_ms_p50"] = pc("upstream_first_byte_ms", 50)
    return out
