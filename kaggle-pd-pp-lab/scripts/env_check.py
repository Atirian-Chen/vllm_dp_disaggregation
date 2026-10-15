"""Step 1: environment check (no CUDA context is created; only nvidia-smi is used).

Writes results/env/env_check.json and starts the session budget clock.
Exit code 0 = usable 2xT4 environment, 1 = blocking problem.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import urllib.request

from common import ENV_DIR, Budget, ensure_dirs, gpu_query, load_config, now_iso, run_cmd, write_json


def disk_free_gb(path: str):
    try:
        return round(shutil.disk_usage(path).free / 1e9, 1)
    except FileNotFoundError:
        return None


def internet_ok(url="https://huggingface.co") -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-budget", action="store_true", help="restart the session clock (new Kaggle session)")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    budget = Budget(cfg)
    if args.reset_budget:
        budget.reset_session()

    info = {"time": now_iso(), "python": sys.version.split()[0], "platform": platform.platform(),
            "cpu_count": os.cpu_count()}
    gpus = gpu_query()
    info["gpus"] = gpus
    rc, cc = run_cmd(["nvidia-smi", "--query-gpu=index,compute_cap,driver_version,pcie.link.gen.max,pcie.link.width.max",
                      "--format=csv,noheader"], timeout=15)
    info["gpu_details_raw"] = cc.strip()
    rc, smi = run_cmd(["nvidia-smi"], timeout=15)
    info["nvidia_smi_header"] = "\n".join(smi.splitlines()[:4])
    rc, topo = run_cmd(["nvidia-smi", "topo", "-m"], timeout=15)
    info["topology"] = topo.strip()
    try:
        with open("/proc/meminfo") as f:
            info["ram_total_gb"] = round(int(f.readline().split()[1]) / 1e6, 1)
    except Exception:
        info["ram_total_gb"] = None
    info["disk_free_gb"] = {p: disk_free_gb(p) for p in ("/tmp", "/kaggle/working", "/dev/shm")}
    info["internet"] = internet_ok()

    problems, warnings = [], []
    if len(gpus) < 2:
        problems.append(f"Need 2 GPUs, found {len(gpus)}. In Kaggle: Settings -> Accelerator -> 'GPU T4 x2'.")
    elif not all("T4" in g["name"] for g in gpus[:2]):
        warnings.append(f"GPUs are {[g['name'] for g in gpus]}, not T4; results not comparable to the plan.")
    busy = [g for g in gpus if g["mem_used_mib"] > 500]
    if busy:
        warnings.append(f"GPU memory already in use: {[(g['index'], g['mem_used_mib']) for g in busy]} "
                        f"(leftover processes? run scripts/servers.py stop)")
    if not info["internet"]:
        problems.append("No internet. In Kaggle: Settings -> Internet -> On (requires phone-verified account).")
    if (info["disk_free_gb"].get("/tmp") or 0) < 25:
        warnings.append("Less than 25 GB free in /tmp (venv ~8 GB + model ~3.5 GB).")
    shm = info["disk_free_gb"].get("/dev/shm")
    if shm is not None and shm < 1:
        warnings.append(f"/dev/shm only {shm} GB; NCCL SHM transport may fail. "
                        f"The server launcher will set NCCL_SHM_DISABLE=1 for PP/PD in that case.")
    info["problems"], info["warnings"] = problems, warnings
    info["ok"] = not problems
    write_json(ENV_DIR / "env_check.json", info)

    print(info["nvidia_smi_header"])
    for g in gpus:
        print(f"GPU{g['index']}: {g['name']}  mem {g['mem_used_mib']:.0f}/{g['mem_total_mib']:.0f} MiB  util {g['util_pct']}%")
    print("compute_cap / driver / PCIe:", info["gpu_details_raw"].replace("\n", " | "))
    print("topology:\n" + info["topology"])
    print(f"python {info['python']}, cpus {info['cpu_count']}, RAM {info['ram_total_gb']} GB, disk free {info['disk_free_gb']}, internet {info['internet']}")
    for w in warnings:
        print("WARNING:", w)
    for p in problems:
        print("BLOCKING:", p)
    print(budget.report())
    print("ENV_CHECK", "PASS" if info["ok"] else "FAIL")
    sys.exit(0 if info["ok"] else 1)


if __name__ == "__main__":
    main()
