"""Shared helpers: paths, config, atomic JSON, nvidia-smi sampling, budget guard.

Only the standard library is used so that every script can run either with the
Kaggle system Python or with the isolated vLLM virtualenv.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
LAB_ROOT = Path(os.environ.get("PDLAB_ROOT", SCRIPTS_DIR.parent)).resolve()
CONFIG_PATH = Path(os.environ.get("PDLAB_CONFIG", LAB_ROOT / "configs" / "experiments.json"))
RESULTS = Path(os.environ.get("PDLAB_RESULTS", LAB_ROOT / "results"))
LOG_DIR = RESULTS / "logs"
STATE_DIR = RESULTS / "state"
ENV_DIR = RESULTS / "env"
SMOKE_DIR = RESULTS / "smoke"
RUNS_DIR = RESULTS / "runs"
WORKLOAD_PATH = RESULTS / "workload" / "workloads.json"
VENV = Path(os.environ.get("PDLAB_VENV", "/tmp/pdlab-venv"))
MODEL_DIR = Path(os.environ.get("PDLAB_MODEL_DIR", "/tmp/pdlab-models"))
HF_HOME = os.environ.get("PDLAB_HF_HOME", "/tmp/pdlab-hf")

ARCH_NAMES = {
    "A": "Single-GPU mixed P+D (GPU0)",
    "B": "DP2: two full replicas + round-robin proxy",
    "C": "PP2: one instance, layers split over GPU0/GPU1",
    "D": "PD disaggregation: prefill GPU0 -> KV (PyNccl) -> decode GPU1",
}
ARCH_LEVEL = {"A": "P0", "C": "P1", "D": "P2", "B": "P3"}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    for d in (RESULTS, LOG_DIR, STATE_DIR, ENV_DIR, SMOKE_DIR, RUNS_DIR, WORKLOAD_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def write_json(path: Path, obj) -> None:
    """Atomic write so a killed session never leaves a half-written result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def append_jsonl(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return out


def run_cmd(cmd: list[str] | str, timeout: float = 60, env: dict | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
                           timeout=timeout, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -9, f"TIMEOUT after {timeout}s: {cmd}"
    except FileNotFoundError as e:
        return -127, str(e)


def venv_python() -> str:
    p = VENV / "bin" / "python"
    return str(p) if p.exists() else sys.executable


def tail(path: Path, n: int = 40) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return "<no log>"


# ---------------------------------------------------------------- GPU sampling
def has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def gpu_query() -> list[dict]:
    if not has_nvidia_smi():
        return []
    rc, out = run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                       "--format=csv,noheader,nounits"], timeout=15)
    rows = []
    if rc != 0:
        return rows
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            rows.append({"index": int(parts[0]), "name": parts[1], "mem_used_mib": float(parts[2]),
                         "mem_total_mib": float(parts[3]), "util_pct": float(parts[4])})
        except ValueError:
            continue
    return rows


def gpu_compute_apps() -> list[dict]:
    """Processes holding GPU memory, mapped to GPU index via UUID."""
    if not has_nvidia_smi():
        return []
    rc1, uu = run_cmd(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], timeout=15)
    rc2, apps = run_cmd(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
                         "--format=csv,noheader,nounits"], timeout=15)
    if rc1 != 0 or rc2 != 0:
        return []
    idx = {}
    for line in uu.strip().splitlines():
        a = [x.strip() for x in line.split(",")]
        if len(a) == 2:
            idx[a[1]] = int(a[0])
    rows = []
    for line in apps.strip().splitlines():
        a = [x.strip() for x in line.split(",")]
        if len(a) >= 3 and a[0].isdigit():
            rows.append({"pid": int(a[0]), "gpu": idx.get(a[1]), "used_mib": float(a[2]) if a[2].replace('.', '', 1).isdigit() else None,
                         "name": a[3] if len(a) > 3 else ""})
    return rows


class GpuSampler:
    """Background nvidia-smi poller: per-GPU peak memory and mean utilization."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[tuple[float, list[dict]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            rows = gpu_query()
            if rows:
                self.samples.append((time.time(), rows))
            self._stop.wait(self.interval_s)

    def __enter__(self):
        if has_nvidia_smi():
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        per = {}
        for _, rows in self.samples:
            for r in rows:
                d = per.setdefault(str(r["index"]), {"mem_peak_mib": 0.0, "util": []})
                d["mem_peak_mib"] = max(d["mem_peak_mib"], r["mem_used_mib"])
                d["util"].append(r["util_pct"])
        out = {}
        for k, d in per.items():
            u = d["util"]
            out[k] = {"mem_peak_mib": d["mem_peak_mib"],
                      "util_mean_pct": round(sum(u) / len(u), 2) if u else None,
                      "util_max_pct": max(u) if u else None,
                      "busy_frac": round(sum(1 for x in u if x > 0) / len(u), 3) if u else None}
        return {"n_samples": len(self.samples), "per_gpu": out}


# ---------------------------------------------------------------- budget guard
class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Wall-clock guard. Kaggle bills the whole GPU session, so we measure from the
    first env_check of this session (results/state/budget.json)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["budget"]
        self.path = STATE_DIR / "budget.json"
        self.state = read_json(self.path, None)
        if self.state is None:
            self.state = {"session_start": time.time(), "session_start_iso": now_iso(), "levels": {}}
            self._save()

    def _save(self):
        write_json(self.path, self.state)

    def reset_session(self):
        self.state["session_start"] = time.time()
        self.state["session_start_iso"] = now_iso()
        self._save()

    def elapsed_h(self) -> float:
        return (time.time() - self.state["session_start"]) / 3600.0

    def check_total(self, force: bool = False):
        cap = self.cfg["total_session_hours_cap"]
        if self.elapsed_h() > cap and not force:
            raise BudgetExceeded(f"Session wall-clock {self.elapsed_h():.2f} h exceeds cap {cap} h; "
                                 f"stop and save results (use --force only if you consciously accept more GPU time).")

    def enter_level(self, level: str, force: bool = False):
        self.check_total(force)
        lv = self.state["levels"].setdefault(level, {"first_start": time.time(), "attempts": 0, "status": "running"})
        lv["attempts"] += 1
        self._save()
        cap_min = self.cfg["level_caps_min"].get(level)
        spent = (time.time() - lv["first_start"]) / 60
        if cap_min is not None and spent > cap_min and lv.get("status") != "pass" and not force:
            raise BudgetExceeded(f"Level {level} debugging time {spent:.1f} min exceeds cap {cap_min} min. "
                                 f"Keep the error evidence in results/ and move on (or --force).")

    def mark_level(self, level: str, status: str, note: str = ""):
        lv = self.state["levels"].setdefault(level, {"first_start": time.time(), "attempts": 0})
        lv["status"] = status
        lv["note"] = note
        lv["updated"] = now_iso()
        lv["spent_min"] = round((time.time() - lv["first_start"]) / 60, 1)
        self._save()

    def report(self) -> str:
        lines = [f"Session started {self.state['session_start_iso']}, elapsed {self.elapsed_h():.2f} h "
                 f"(cap {self.cfg['total_session_hours_cap']} h)"]
        for k, v in sorted(self.state["levels"].items()):
            lines.append(f"  {k}: status={v.get('status')} attempts={v.get('attempts')} "
                         f"spent={(time.time() - v['first_start']) / 60:.1f} min "
                         f"(cap {self.cfg['level_caps_min'].get(k)} min) {v.get('note', '')}")
        return "\n".join(lines)


def percentile(values, q: float):
    """Linear-interpolated percentile (same convention as numpy default); q in [0, 100]."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    pos = (len(vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
