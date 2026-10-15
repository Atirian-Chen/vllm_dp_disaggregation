"""Small closed-loop benchmark: archs x {prefill_heavy, decode_heavy} x concurrency {2, 8} x reps.

  python scripts/run_experiments.py --archs A,C,D,B --reps 1        # first full round (16 runs)
  python scripts/run_experiments.py --archs A,C,D,B --reps 2        # adds rep 2 only (rep 1 is skipped)

Resumable: each run is saved atomically to results/runs/<arch>__<workload>__c<conc>__r<rep>.json
and skipped next time if it is complete and valid. Each architecture is started once,
all its runs are executed, then it is stopped and GPU memory release is verified.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import traceback

import servers
from client import run_closed_loop, summarize
from common import (ARCH_NAMES, LOG_DIR, RUNS_DIR, SMOKE_DIR, WORKLOAD_PATH, Budget, BudgetExceeded, GpuSampler,
                    append_jsonl, ensure_dirs, load_config, now_iso, read_json, write_json)
from evidence import log_offsets, pd_probe_stats, proxy_stats, scan_server_logs


def run_path(arch, wname, conc, rep):
    return RUNS_DIR / f"{arch}__{wname}__c{conc}__r{rep}.json"


def is_done(path, mock: bool) -> bool:
    d = read_json(path)
    return bool(d and d.get("complete") and d.get("mock", False) == mock and d["summary"]["success_rate"] >= 0.9)


def one_run(arch, st, cfg, wl, wname, conc, rep, mock) -> dict:
    key = f"{wname}|c{conc}|r{rep}"
    entry = wl["runs"][key]
    model = cfg["model"]["served_name"]
    tmo = cfg["timeouts"]["request_s"]
    endpoint = st["endpoint"]
    # warmup at the same concurrency, discarded
    w_res, w_wall, _, _ = asyncio.run(asyncio.wait_for(
        run_closed_loop(endpoint, model, entry["warmup"], conc, tmo), cfg["timeouts"]["run_s"]))
    offs = log_offsets(st)
    with GpuSampler(cfg["gpu_sample_interval_s"]) as gs:
        res, wall, t0, t1 = asyncio.run(asyncio.wait_for(
            run_closed_loop(endpoint, model, entry["measure"], conc, tmo), cfg["timeouts"]["run_s"]))
    time.sleep(0.5)
    procs = {p["name"]: p for p in st["procs"]}
    out = {
        "arch": arch, "arch_name": ARCH_NAMES[arch], "workload": wname, "concurrency": conc, "rep": rep,
        "mock": mock, "time": now_iso(), "t_start_epoch": t0, "t_end_epoch": t1,
        "workload_sha256": wl["meta"]["runs_sha256"],
        "input_tokens": cfg["workloads"][wname]["input_tokens"], "output_tokens": cfg["workloads"][wname]["output_tokens"],
        "warmup": {"n": len(w_res), "n_ok": sum(r["ok"] for r in w_res), "wall_s": round(w_wall, 2)},
        "summary": summarize(res, wall), "gpu": gs.summary(),
        "server_log_evidence": scan_server_logs(st, offs),
        "requests": [{k: v for k, v in r.items() if k != "text"} for r in res],
    }
    if "proxy" in procs:
        out["proxy"] = proxy_stats(procs["proxy"]["extra"].get("proxy_log"), t0, t1 + 1)
    if arch == "D":
        out["pd"] = pd_probe_stats(procs["prefill"]["extra"]["probe_log"], procs["decode"]["extra"]["probe_log"], t0, t1 + 1)
    out["complete"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="A,C,D,B")
    ap.add_argument("--workloads", default="prefill_heavy,decode_heavy")
    ap.add_argument("--conc", default=None, help="comma list, default from config")
    ap.add_argument("--reps", type=int, default=1, help="run reps 1..N (completed reps are skipped)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--ignore-smoke", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore the total session-time cap")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    wl = read_json(WORKLOAD_PATH)
    if not wl:
        sys.exit("workload missing: run prepare_workload.py first")
    budget = Budget(cfg)
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    wnames = [w.strip() for w in args.workloads.split(",")]
    concs = [int(c) for c in args.conc.split(",")] if args.conc else cfg["concurrency_levels"]
    reps = list(range(1, min(args.reps, cfg["max_reps"]) + 1))
    fail_log = RUNS_DIR / "_failures.jsonl"
    summary_rows = []

    for arch in archs:
        smoke = read_json(SMOKE_DIR / f"{arch}.json")
        if not args.ignore_smoke and not (smoke and smoke.get("status") == "PASS" and smoke.get("mock", False) == args.mock):
            print(f"[{arch}] SKIP: smoke test not PASS (run smoke_test.py --arch {arch} first)")
            summary_rows.append((arch, "skipped: no smoke PASS"))
            continue
        todo = [(w, c, r) for w in wnames for c in concs for r in reps if not is_done(run_path(arch, w, c, r), args.mock)]
        if not todo:
            print(f"[{arch}] all {len(wnames) * len(concs) * len(reps)} runs already complete - skip")
            summary_rows.append((arch, "complete (cached)"))
            continue
        try:
            budget.check_total(args.force)
        except BudgetExceeded as e:
            print("BUDGET STOP:", e)
            break
        print(f"[{arch}] {ARCH_NAMES[arch]}: starting servers for {len(todo)} runs", flush=True)
        try:
            st = servers.start(arch, cfg, mock=args.mock)
        except Exception as e:
            print(f"[{arch}] server start FAILED: {e}")
            append_jsonl(fail_log, {"time": now_iso(), "arch": arch, "stage": "start", "error": str(e)[:3000]})
            summary_rows.append((arch, "server start failed"))
            continue
        st_info = {"startup_s": st["startup_s"], "gpu_after_start": st["gpu_after_start"]}
        write_json(LOG_DIR.parent / "state" / f"startup_{arch}.json", st_info)
        n_ok = 0
        try:
            for (w, c, r) in todo:
                try:
                    budget.check_total(args.force)
                except BudgetExceeded as e:
                    print("BUDGET STOP:", e)
                    break
                if not servers.alive(st):
                    raise RuntimeError("a server process died; see logs in results/logs")
                t = time.time()
                print(f"[{arch}] {w} c={c} rep={r} ...", end=" ", flush=True)
                try:
                    out = one_run(arch, st, cfg, wl, w, c, r, args.mock)
                    out["server_startup_s"] = st["startup_s"]
                    write_json(run_path(arch, w, c, r), out)
                    s = out["summary"]
                    n_ok += 1
                    msg = (f"ok {s['n_ok']}/{s['n_requests']}  TTFT p50/p95 {s['ttft_p50_ms']}/{s['ttft_p95_ms']} ms  "
                           f"ITL p50/p95 {s['itl_p50_ms']}/{s['itl_p95_ms']} ms  TPOT {s['tpot_mean_ms']} ms  "
                           f"{s['output_tok_per_s']} tok/s  ({time.time() - t:.0f}s)")
                    if arch == "D":
                        p = out["pd"]
                        msg += f"  KV hits {p['n_select_hit']} bypass_false {p['n_recv_bypass_false']} xfer p50 {p['transfer_ms_p50']} ms"
                    print(msg, flush=True)
                    if s["success_rate"] < 0.9:
                        print(f"   WARNING: success rate {s['success_rate']}, errors: {s['errors']}")
                except Exception as e:
                    print(f"FAILED: {type(e).__name__}: {e}", flush=True)
                    append_jsonl(fail_log, {"time": now_iso(), "arch": arch, "workload": w, "conc": c, "rep": r,
                                            "error": f"{type(e).__name__}: {e}"[:2000],
                                            "traceback": traceback.format_exc()[-3000:]})
            summary_rows.append((arch, f"{n_ok}/{len(todo)} new runs ok"))
        except Exception as e:
            print(f"[{arch}] ABORTED: {e}")
            append_jsonl(fail_log, {"time": now_iso(), "arch": arch, "stage": "run", "error": str(e)[:2000]})
            summary_rows.append((arch, f"aborted: {e}"[:120]))
        finally:
            res = servers.stop(cfg, quiet=True)
            print(f"[{arch}] servers stopped, GPU memory released: {res['gpu_freed']}", flush=True)

    print("\n=== run summary ===")
    for a, s in summary_rows:
        print(f"  {a}: {s}")
    print(budget.report())


if __name__ == "__main__":
    main()
