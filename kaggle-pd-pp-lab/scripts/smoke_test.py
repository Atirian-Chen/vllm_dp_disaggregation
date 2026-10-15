"""Smoke test + authenticity checks for one architecture (start -> verify -> stop).

  python scripts/smoke_test.py --arch A      (P0)
  python scripts/smoke_test.py --arch C      (P1)  verifies real 2-stage pipeline
  python scripts/smoke_test.py --arch D      (P2)  verifies real KV transfer + negative control
  python scripts/smoke_test.py --arch B      (P3)  verifies two independent replicas

Writes results/smoke/<arch>.json ; the benchmark runner only runs architectures whose
smoke status is PASS (unless --ignore-smoke).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback

import servers
from client import one_request, run_closed_loop
from common import (ARCH_LEVEL, ARCH_NAMES, SMOKE_DIR, WORKLOAD_PATH, Budget, BudgetExceeded, ensure_dirs,
                    gpu_compute_apps, gpu_query, load_config, now_iso, read_json, write_json)
from evidence import log_offsets, pd_probe_stats, proxy_stats, scan_server_logs


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def check_arch(arch: str, ev: dict, gpu: list[dict], apps: list[dict], extra: dict, mock: bool) -> list[str]:
    """Return a list of failed checks (empty == pass)."""
    fails = []
    mem = {g["index"]: g["mem_used_mib"] for g in gpu}
    inst = {k: v for k, v in ev.items() if k != "proxy"}
    if not mock:
        for name, e in inst.items():
            if not (e["engine"] or "").startswith("V0"):
                fails.append(f"{name}: expected V0 engine, log says {e['engine']}")
    if arch == "A":
        e = inst["gpu0"]
        if not mock and e["pp"] != 1:
            fails.append(f"pipeline_parallel_size={e['pp']} (expected 1)")
        if not mock and not (mem.get(0, 0) > 4000 and mem.get(1, 0) < 1000):
            fails.append(f"GPU memory pattern {mem} (expected GPU0 busy, GPU1 idle)")
    elif arch == "B":
        for n in ("gpu0", "gpu1"):
            if not mock and inst[n]["pp"] != 1:
                fails.append(f"{n}: pp={inst[n]['pp']}")
            if len(inst[n]["model_loading_gib"]) != 1:
                fails.append(f"{n}: expected one full weight load, got {inst[n]['model_loading_gib']}")
        if not mock and not (mem.get(0, 0) > 4000 and mem.get(1, 0) > 4000):
            fails.append(f"GPU memory {mem}: both replicas should hold a full model + KV cache")
        pb = extra.get("proxy", {}).get("per_backend", {})
        if len(pb) < 2:
            fails.append(f"round-robin did not use both replicas: {pb}")
    elif arch == "C":
        e = inst["pp2"]
        loads = e["model_loading_gib"]
        if not mock:
            if e["pp"] != 2 or e["tp"] != 1:
                fails.append(f"tp={e['tp']} pp={e['pp']} (expected tp=1 pp=2)")
            if len(loads) != 2:
                fails.append(f"expected 2 per-stage weight loads, got {loads}")
            ref = read_json(SMOKE_DIR / "A.json") or {}
            full = (ref.get("log_evidence", {}).get("gpu0", {}).get("model_loading_gib") or [None])[0]
            extra["single_gpu_weight_gib"] = full
            # each stage holds ~half the layers (+ the tied embedding / lm_head copy), so clearly < full model
            if full and loads and max(loads) > 0.8 * full:
                fails.append(f"stage weights {loads} GiB not clearly smaller than single-GPU {full} GiB")
            if not (mem.get(0, 0) > 3000 and mem.get(1, 0) > 3000):
                fails.append(f"GPU memory {mem}: both stages should hold weights + KV cache")
        extra["pp_stage_weight_gib"] = loads
    elif arch == "D":
        pre, dec = inst["prefill"], inst["decode"]
        if not mock and not (pre["pynccl_kv_init"] and dec["pynccl_kv_init"]):
            fails.append("PyNcclConnector init line missing in a log")
        pd = extra["pd_proxied"]
        n = extra["n_proxied_requests"]
        if pd["n_select_hit"] < n:
            fails.append(f"KV hits {pd['n_select_hit']} < proxied requests {n} (decode recomputed prefill?)")
        if pd["n_recv_bypass_false"] != 0:
            fails.append(f"{pd['n_recv_bypass_false']} decode prefill steps were recomputed instead of using received KV")
        if pd["kv_bytes_total"] <= 0:
            fails.append("no KV bytes received")
        ctrl = extra["pd_control"]
        if ctrl["n_select_miss"] < 1 or ctrl["n_recv_bypass_false"] < 1:
            fails.append(f"negative control did not register a miss/recompute: {ctrl}")
    return fails


async def smoke_requests(endpoint: str, model: str, items: list[dict], timeout: float):
    seq = []
    for it in items:  # sequential: clean per-request latency
        seq.append(await run_one(endpoint, model, it, timeout))
    conc, wall, _, _ = await run_closed_loop(endpoint, model, items, len(items), timeout)
    return seq, conc


async def run_one(endpoint, model, it, timeout):
    import aiohttp
    async with aiohttp.ClientSession() as s:
        return await one_request(s, endpoint, model, it, timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--keep", action="store_true", help="leave servers running afterwards")
    ap.add_argument("--force", action="store_true", help="ignore level time cap")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    level = ARCH_LEVEL[args.arch]
    budget = Budget(cfg)
    try:
        budget.enter_level(level, force=args.force)
    except BudgetExceeded as e:
        print("BUDGET STOP:", e)
        sys.exit(2)
    wl = read_json(WORKLOAD_PATH)
    if not wl:
        sys.exit("workload missing: run prepare_workload.py first")
    items = wl["smoke"]
    model = cfg["model"]["served_name"]
    timeout = cfg["timeouts"]["request_s"]
    out = {"arch": args.arch, "arch_name": ARCH_NAMES[args.arch], "level": level, "time": now_iso(), "mock": args.mock,
           "status": "FAIL"}
    t_start = time.time()
    try:
        st = servers.start(args.arch, cfg, mock=args.mock, consumer_debug=(args.arch == "D"))
        out["startup_s"] = st["startup_s"]
        out["gpu_after_start"] = st["gpu_after_start"]
        out["gpu_apps_after_start"] = st["gpu_apps_after_start"]
        offs = {p["name"]: p["log_offset"] for p in st["procs"]}
        t0 = time.time()
        seq, conc = asyncio.run(smoke_requests(st["endpoint"], model, items, timeout))
        t1 = time.time()
        out["requests_sequential"] = seq
        out["requests_concurrent"] = conc
        extra = {}
        procs = {p["name"]: p for p in st["procs"]}
        if "proxy" in procs:
            extra["proxy"] = proxy_stats(procs["proxy"]["extra"].get("proxy_log"), t0, t1)
        if args.arch == "D":
            prod_log = procs["prefill"]["extra"]["probe_log"]
            cons_log = procs["decode"]["extra"]["probe_log"]
            extra["n_proxied_requests"] = len(seq) + len(conc)
            extra["pd_proxied"] = pd_probe_stats(prod_log, cons_log, t0, t1 + 1)
            # negative control: bypass the proxy -> the producer never saw this prompt
            ctrl_item = dict(items[-1])
            ctrl_item["id"] = "control-no-prefill"
            ctrl_item["prompt_token_ids"] = list(reversed(ctrl_item["prompt_token_ids"]))
            t2 = time.time()
            decode_url = f"http://127.0.0.1:{cfg['ports']['inst1']}"
            out["control_request"] = asyncio.run(run_one(decode_url, model, ctrl_item, timeout))
            time.sleep(0.5)
            extra["pd_control"] = pd_probe_stats(prod_log, cons_log, t2, time.time())
        time.sleep(1.0)
        ev = scan_server_logs(st, offs)
        out["log_evidence"] = ev
        out["gpu_after_requests"] = gpu_query()
        out["gpu_apps"] = gpu_compute_apps()
        out["extra"] = extra
        all_req = seq + conc
        bad = [r for r in all_req if not r["ok"]]
        fails = [f"request {r['id']} failed: {r.get('error')}" for r in bad]
        fails += check_arch(args.arch, ev, out["gpu_after_start"], out["gpu_apps"], extra, args.mock)
        # greedy determinism inside this architecture and agreement with A (informational)
        same = sum(1 for a, b in zip(seq, conc) if a.get("text") == b.get("text"))
        out["seq_vs_conc_identical_outputs"] = f"{same}/{len(seq)}"
        ref = read_json(SMOKE_DIR / "A.json")
        if ref and args.arch != "A" and ref.get("status") == "PASS":
            ref_t = {r["id"]: r.get("text", "") for r in ref["requests_sequential"]}
            agree = []
            for r in seq:
                a, b = ref_t.get(r["id"], ""), r.get("text", "")
                agree.append({"id": r["id"], "identical": a == b,
                              "common_prefix_chars": common_prefix_len(a, b), "len": len(b)})
            out["agreement_with_A"] = agree
        out["checks_failed"] = fails
        out["status"] = "PASS" if not fails else "FAIL"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-4000:]
    finally:
        out["duration_s"] = round(time.time() - t_start, 1)
        if not args.keep or out["status"] != "PASS":
            out["stop"] = servers.stop(cfg, quiet=True)
        write_json(SMOKE_DIR / f"{args.arch}.json", out)
        budget.mark_level(level, "pass" if out["status"] == "PASS" else "fail", out.get("error", "")[:200])

    print(f"=== SMOKE {args.arch} ({ARCH_NAMES[args.arch]}): {out['status']}  [{out['duration_s']} s, startup {out.get('startup_s')} s]")
    for r in out.get("requests_sequential", []):
        print(f"  {r['id']:12s} ok={r['ok']} ttft={r.get('ttft_ms')} ms e2e={r.get('e2e_ms')} ms "
              f"in={r.get('prompt_tokens')} out={r.get('output_tokens')} text={r.get('text_head', '')[:60]!r}")
    for name, e in (out.get("log_evidence") or {}).items():
        print(f"  [{name}] engine={e['engine']} tp={e['tp']} pp={e['pp']} weights_GiB={e['model_loading_gib']} "
              f"attn={e['attention_backends']} kv_init={e['pynccl_kv_init']} kv_fail={e['kv_fail_count']} kv_ok={e['kv_success_count']}")
    for g in out.get("gpu_after_start", []):
        print(f"  GPU{g['index']}: {g['mem_used_mib']:.0f} MiB used after start")
    if args.arch == "D" and "extra" in out:
        p = out["extra"]["pd_proxied"]
        print(f"  PD transfer: hits={p['n_select_hit']}/{out['extra']['n_proxied_requests']} bypass_false={p['n_recv_bypass_false']} "
              f"KV bytes/req={p['kv_bytes_per_req_mean']} bytes/token={p['kv_bytes_per_token']} transfer p50={p['transfer_ms_p50']} ms "
              f"~{p['effective_GBps']} GB/s")
        print(f"  negative control (direct to decode): {out['extra']['pd_control']}")
    if out.get("agreement_with_A"):
        print("  agreement with A:", [(a["id"], a["identical"]) for a in out["agreement_with_A"]])
    for f in out.get("checks_failed", []):
        print("  CHECK FAILED:", f)
    if out.get("error"):
        print("  ERROR:", out["error"])
        print(out.get("traceback", "")[-2500:])
    print(f"  details: {SMOKE_DIR / (args.arch + '.json')}")
    sys.exit(0 if out["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
