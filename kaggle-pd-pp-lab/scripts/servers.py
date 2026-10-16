"""Start / stop / inspect the serving processes for one architecture.

  A  single instance, CUDA_VISIBLE_DEVICES=0                      client -> :8100
  B  two full instances (GPU0 :8100, GPU1 :8200) + rr proxy       client -> :8000
  C  one instance, CUDA_VISIBLE_DEVICES=0,1, --pipeline-parallel-size 2 (mp backend)
                                                                   client -> :8100
  D  prefill/kv_producer GPU0 :8100 + decode/kv_consumer GPU1 :8200 (PyNcclConnector)
     + pd proxy                                                    client -> :8000

All vLLM instances share the same flags (dtype, max_model_len, memory fraction,
max_num_seqs, max_num_batched_tokens, chunked prefill OFF, prefix caching OFF, seed).

CLI:  python scripts/servers.py start --arch D [--mock]
      python scripts/servers.py stop
      python scripts/servers.py status
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common import (ENV_DIR, LOG_DIR, SCRIPTS_DIR, STATE_DIR, HF_HOME, VENV, ensure_dirs, gpu_compute_apps, gpu_query,
                    load_config, now_iso, read_json, run_cmd, tail, venv_python, write_json)

STATE_FILE = STATE_DIR / "servers.json"
IS_WIN = os.name == "nt"


@dataclass
class ProcSpec:
    name: str
    cmd: list[str]
    env: dict
    log: str
    health: str
    gpus: str = ""
    role: str = ""
    extra: dict = field(default_factory=dict)


def model_path(cfg) -> str:
    m = read_json(ENV_DIR / "model.json")
    return m["path"] if m else cfg["model"]["repo_id"]


def vllm_cmd(cfg, port: int, extra: list[str]) -> list[str]:
    v = cfg["vllm"]
    cmd = [venv_python(), "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_path(cfg), "--served-model-name", cfg["model"]["served_name"],
           "--host", "127.0.0.1", "--port", str(port),
           "--dtype", v["dtype"], "--max-model-len", str(v["max_model_len"]),
           "--gpu-memory-utilization", str(v["gpu_memory_utilization"]),
           "--max-num-seqs", str(v["max_num_seqs"]),
           "--max-num-batched-tokens", str(v["max_num_batched_tokens"]),
           "--no-enable-chunked-prefill", "--no-enable-prefix-caching",
           "--swap-space", str(v.get("swap_space_gib", 4)),
           "--seed", str(v["seed"]), "--disable-log-requests"]
    if v.get("enforce_eager"):
        cmd.append("--enforce-eager")
    return cmd + list(v.get("extra_args", [])) + extra


def base_env(cfg, gpus: str, run_tag: str) -> dict:
    env = dict(os.environ)
    env.update(cfg["vllm"].get("engine_env", {}))
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["HF_HOME"] = HF_HOME
    env["HF_HUB_OFFLINE"] = "1" if read_json(ENV_DIR / "model.json") else "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_LOGGING_LEVEL"] = env.get("PDLAB_VLLM_LOG_LEVEL", "INFO")
    env["PYTHONUNBUFFERED"] = "1"
    env["PDLAB_RUN_TAG"] = run_tag
    env.pop("PYTHONPATH", None)
    shm = read_json(ENV_DIR / "env_check.json", {}).get("disk_free_gb", {}).get("/dev/shm")
    if shm is not None and shm < 1:
        env["NCCL_SHM_DISABLE"] = "1"
    if os.environ.get("PDLAB_NCCL_SAFE") == "1":  # documented fallback for NCCL transport problems
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_SHM_DISABLE"] = "1"
    return env


def build_specs(arch: str, cfg: dict, mock: bool = False, consumer_debug: bool = False) -> tuple[list[ProcSpec], str]:
    p = cfg["ports"]
    tag = f"{arch}-{int(time.time())}"
    u0, u1, up = f"http://127.0.0.1:{p['inst0']}", f"http://127.0.0.1:{p['inst1']}", f"http://127.0.0.1:{p['proxy']}"
    probe_dir = LOG_DIR / "pd_probe"

    def inst(name, port, gpus, extra, role="", probe=None, log_level=None):
        if mock:
            cmd = [sys.executable, str(SCRIPTS_DIR / "mock_vllm_server.py"), "--port", str(port), "--role", role or "mixed"]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(SCRIPTS_DIR)
        else:
            cmd = vllm_cmd(cfg, port, extra)
            env = base_env(cfg, gpus, tag)
        if log_level:
            env["VLLM_LOGGING_LEVEL"] = log_level
        if probe:
            probe_dir.mkdir(parents=True, exist_ok=True)
            env["PDLAB_PROBE_LOG"] = str(probe_dir / f"{probe}.jsonl")
            if not mock:
                env["PYTHONPATH"] = str(SCRIPTS_DIR / "pd_probe")
                # vLLM 0.8.5's PD KV helper only handles the FlashAttention cache layout; T4 uses the
                # XFormers/PagedAttention layout -> see scripts/pd_probe/pdlab_kv_layout.py
                env["PDLAB_PAGED_KV_FIX"] = "1"
        return ProcSpec(name, cmd, env, str(LOG_DIR / f"server_{arch}_{name}.log"), f"http://127.0.0.1:{port}/health",
                        gpus=gpus, role=role, extra={"port": port, "probe_log": env.get("PDLAB_PROBE_LOG")})

    def proxy(args):
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        return ProcSpec("proxy", [venv_python() if not mock else sys.executable, str(SCRIPTS_DIR / "proxy.py"),
                                  "--port", str(p["proxy"])] + args + ["--log", str(LOG_DIR / f"proxy_{arch}.jsonl")],
                        env, str(LOG_DIR / f"server_{arch}_proxy.log"), f"{up}/health", role="proxy",
                        extra={"port": p["proxy"], "proxy_log": str(LOG_DIR / f"proxy_{arch}.jsonl")})

    if arch == "A":
        return [inst("gpu0", p["inst0"], "0", [])], u0
    if arch == "B":
        return [inst("gpu0", p["inst0"], "0", []), inst("gpu1", p["inst1"], "1", []),
                proxy(["--mode", "rr", "--backends", f"{u0},{u1}"])], up
    if arch == "C":
        return [inst("pp2", p["inst0"], "0,1", ["--pipeline-parallel-size", "2", "--tensor-parallel-size", "1",
                                                 "--distributed-executor-backend", "mp"])], u0
    if arch == "D":
        kv = cfg["kv_transfer"]

        def ktc(role, rank):
            return json.dumps({"kv_connector": kv["connector"], "kv_role": role, "kv_rank": rank, "kv_parallel_size": 2,
                               "kv_buffer_size": kv["kv_buffer_size"], "kv_ip": "127.0.0.1", "kv_port": kv["kv_port"]})
        return [inst("prefill", p["inst0"], "0", ["--kv-transfer-config", ktc("kv_producer", 0)], role="producer",
                     probe="producer"),
                inst("decode", p["inst1"], "1", ["--kv-transfer-config", ktc("kv_consumer", 1)], role="consumer",
                     probe="consumer", log_level="DEBUG" if consumer_debug else None),
                proxy(["--mode", "pd", "--prefill", u0, "--decode", u1])], up
    raise ValueError(f"unknown arch {arch}")


def _http_ok(url: str, timeout=3) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def start(arch: str, cfg: dict, mock: bool = False, consumer_debug: bool = False, timeout_s: float | None = None) -> dict:
    ensure_dirs()
    stop(cfg, quiet=True, wait_gpu=not mock)
    specs, endpoint = build_specs(arch, cfg, mock, consumer_debug)
    timeout_s = timeout_s or cfg["timeouts"]["server_start_s"]
    procs, state = [], {"arch": arch, "mock": mock, "endpoint": endpoint, "started": now_iso(), "t_launch": time.time(), "procs": []}
    for s in specs:
        Path(s.log).parent.mkdir(parents=True, exist_ok=True)
        logf = open(s.log, "a", encoding="utf-8")
        logf.write(f"\n===== {now_iso()} START {arch}/{s.name}: {' '.join(s.cmd)}\n")
        logf.flush()
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WIN else {"start_new_session": True}
        pr = subprocess.Popen(s.cmd, env=s.env, stdout=logf, stderr=subprocess.STDOUT, **kw)
        procs.append((s, pr))
        d = asdict(s)
        d.pop("env")
        d["pid"] = pr.pid
        d["log_offset"] = os.path.getsize(s.log)
        state["procs"].append(d)
        write_json(STATE_FILE, state)
        print(f"launched {s.name:8s} pid={pr.pid} gpus={s.gpus or '-'} log={s.log}", flush=True)
    # vLLM instances first, proxy last
    deadline = time.time() + timeout_s
    pending = list(procs)
    last_print = 0
    while pending:
        for s, pr in list(pending):
            if pr.poll() is not None:
                state["status"] = "failed"
                state["error"] = f"{s.name} exited with code {pr.returncode}"
                write_json(STATE_FILE, state)
                msg = f"{s.name} exited early (code {pr.returncode}). Last log lines:\n{tail(Path(s.log), 60)}"
                stop(cfg, quiet=True)
                raise RuntimeError(msg)
            if _http_ok(s.health):
                pending.remove((s, pr))
                print(f"ready   {s.name:8s} after {time.time() - state['t_launch']:.0f}s", flush=True)
        if time.time() > deadline:
            names = [s.name for s, _ in pending]
            logs = "\n".join(f"--- {s.name}\n{tail(Path(s.log), 30)}" for s, _ in pending)
            stop(cfg, quiet=True)
            raise TimeoutError(f"servers {names} not healthy after {timeout_s}s\n{logs}")
        if time.time() - last_print > 30 and pending:
            last_print = time.time()
            print(f"  waiting for {[s.name for s, _ in pending]} ({time.time() - state['t_launch']:.0f}s)", flush=True)
        time.sleep(0.5 if mock else 3)
    state["status"] = "ready"
    state["startup_s"] = round(time.time() - state["t_launch"], 1)
    state["gpu_after_start"] = gpu_query()
    state["gpu_apps_after_start"] = gpu_compute_apps()
    write_json(STATE_FILE, state)
    return state


def alive(state: dict | None = None) -> bool:
    state = state or read_json(STATE_FILE)
    if not state:
        return False
    for p in state.get("procs", []):
        if not _pid_alive(p["pid"]):
            return False
    return True


def _pid_alive(pid: int) -> bool:
    if IS_WIN:
        rc, out = run_cmd(["tasklist", "/FI", f"PID eq {pid}"], timeout=10)
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # zombie check
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except Exception:
        return True


def _kill_tree(pid: int, sig):
    if IS_WIN:
        run_cmd(["taskkill", "/T", "/F", "/PID", str(pid)], timeout=20)
        return
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def stop(cfg: dict | None = None, quiet: bool = False, wait_gpu: bool = True) -> dict:
    cfg = cfg or load_config()
    state = read_json(STATE_FILE) or {}
    pids = [p["pid"] for p in state.get("procs", [])]
    for pid in pids:
        _kill_tree(pid, signal.SIGTERM)
    t0 = time.time()
    while any(_pid_alive(p) for p in pids) and time.time() - t0 < 20:
        time.sleep(0.5)
    for pid in pids:
        if _pid_alive(pid):
            _kill_tree(pid, signal.SIGKILL if not IS_WIN else signal.SIGTERM)
    if not IS_WIN:  # orphans from earlier crashed sessions / mp workers
        # never match the runner itself: only server entrypoints and the venv's mp workers
        for pat in ("vllm.entrypoints.openai.api_server", "scripts/proxy.py", "mock_vllm_server.py",
                    f"{VENV}/bin/python.*spawn_main", f"{VENV}/bin/python.*resource_tracker"):
            run_cmd(["pkill", "-9", "-f", pat], timeout=10)
    freed = None
    if wait_gpu and not state.get("mock") and gpu_query():
        # idle baseline recorded by env_check (other tenants of the GPU, e.g. a desktop) + slack
        base = read_json(ENV_DIR / "env_check.json", {}).get("gpus", [])
        thresh = {g["index"]: max(600.0, g["mem_used_mib"] + 300) for g in base}
        deadline = time.time() + cfg["timeouts"]["stop_s"]
        while time.time() < deadline:
            g = gpu_query()
            if all(x["mem_used_mib"] < thresh.get(x["index"], 600.0) for x in g):
                freed = True
                break
            time.sleep(2)
        else:
            freed = False
    if state:
        state["status"] = "stopped"
        state["stopped"] = now_iso()
        write_json(STATE_FILE, state)
    res = {"killed_pids": pids, "gpu_freed": freed, "gpu": gpu_query()}
    if not quiet:
        print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "status"])
    ap.add_argument("--arch", choices=["A", "B", "C", "D"])
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.action == "start":
        st = start(args.arch, cfg, mock=args.mock)
        print(json.dumps({k: st[k] for k in ("arch", "endpoint", "startup_s")}, indent=2))
    elif args.action == "stop":
        stop(cfg)
    else:
        st = read_json(STATE_FILE) or {}
        print(json.dumps({"arch": st.get("arch"), "status": st.get("status"), "alive": alive(st),
                          "gpu": gpu_query(), "apps": gpu_compute_apps()}, indent=2))


if __name__ == "__main__":
    main()
