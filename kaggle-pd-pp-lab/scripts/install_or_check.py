"""Step 2/3: isolated vLLM virtualenv + pinned model download.

Why an isolated venv: Kaggle's preinstalled torch/transformers move every month;
installing an old vLLM into the system env silently up/downgrades them. `uv` with
`--exclude-newer` resolves every dependency as it existed when vLLM 0.8.5.post1
was released, which keeps the environment reproducible months later.

Usage (system python):
  python3 scripts/install_or_check.py --step env     # venv + vllm (5-10 min, once)
  python3 scripts/install_or_check.py --step model   # model snapshot (~3.1 GB)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import ENV_DIR, LOG_DIR, MODEL_DIR, VENV, Budget, BudgetExceeded, ensure_dirs, load_config, read_json, run_cmd, write_json

VERIFY_SNIPPET = r"""
import json, torch
info = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
try:
    info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
except Exception as e:
    info["nccl"] = repr(e)
if torch.cuda.is_available():
    info["devices"] = [{"name": torch.cuda.get_device_name(i), "cc": list(torch.cuda.get_device_capability(i))}
                       for i in range(torch.cuda.device_count())]
    try:
        info["p2p_0_1"] = torch.cuda.can_device_access_peer(0, 1) if torch.cuda.device_count() > 1 else None
    except Exception as e:
        info["p2p_0_1"] = repr(e)
import vllm, transformers
info["vllm"] = vllm.__version__
info["transformers"] = transformers.__version__
try:
    import xformers; info["xformers"] = xformers.__version__
except Exception as e:
    info["xformers"] = "missing: " + repr(e)
import sys; info["python"] = sys.version.split()[0]
print("VERIFY_JSON=" + json.dumps(info))
"""


def sh(cmd: list[str], log: Path, timeout: int) -> int:
    print("$", " ".join(cmd), flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        try:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            f.write(f"TIMEOUT after {timeout}s\n")
            return -9


def ensure_uv(log: Path) -> list[str]:
    import shutil
    if shutil.which("uv"):
        return [shutil.which("uv")]
    rc, _ = run_cmd([sys.executable, "-m", "uv", "--version"])
    if rc != 0:
        if sh([sys.executable, "-m", "pip", "install", "-q", "uv"], log, 300) != 0:
            raise RuntimeError("pip install uv failed; see " + str(log))
    return [sys.executable, "-m", "uv"]


def verify(py: str) -> dict:
    rc, out = run_cmd([py, "-c", VERIFY_SNIPPET], timeout=300)
    for line in out.splitlines():
        if line.startswith("VERIFY_JSON="):
            return json.loads(line[len("VERIFY_JSON="):])
    raise RuntimeError("verification failed:\n" + out[-3000:])


def step_env(cfg: dict, args) -> dict:
    log = LOG_DIR / "install.log"
    venv = Path(args.venv)
    py = venv / "bin" / "python"
    version = args.vllm_version or cfg["vllm"]["version"]
    exclude = args.exclude_newer or cfg["vllm"]["exclude_newer"]
    prev = read_json(ENV_DIR / "install.json", {})
    if py.exists() and prev.get("ok") and prev.get("verify", {}).get("vllm") == version and not args.force:
        print(f"venv already OK: vllm {version} at {venv} (skip). Use --force to reinstall.")
        return prev
    t0 = time.time()
    uv = ensure_uv(log)
    if not py.exists():
        if sh(uv + ["venv", "--python", cfg["vllm"]["python"], str(venv)], log, 600) != 0:
            print("uv could not provision Python", cfg["vllm"]["python"], "-> falling back to system python")
            if sh(uv + ["venv", "--python", sys.executable, str(venv)], log, 300) != 0:
                raise RuntimeError("venv creation failed; see " + str(log))
    rc = sh(uv + ["pip", "install", "--python", str(py), f"vllm=={version}", "--exclude-newer", exclude],
            log, args.timeout)
    if rc != 0:
        raise RuntimeError(f"vllm install failed (rc={rc}); see {log}")
    info = verify(str(py))
    rc, freeze = run_cmd(uv + ["pip", "freeze", "--python", str(py)], timeout=120)
    (ENV_DIR / "venv_freeze.txt").write_text(freeze)
    res = {"ok": True, "venv": str(venv), "requested_vllm": version, "exclude_newer": exclude,
           "install_seconds": round(time.time() - t0, 1), "verify": info}
    problems = []
    if not info.get("cuda_available") or info.get("device_count", 0) < 2:
        problems.append("torch in venv does not see 2 CUDA devices")
    if any(d["cc"] != [7, 5] for d in info.get("devices", [])):
        problems.append(f"unexpected compute capability {info.get('devices')}")
    if str(info.get("xformers", "")).startswith("missing"):
        problems.append("xformers missing: V0 on Turing needs the XFormers attention backend")
    res["problems"] = problems
    res["ok"] = info.get("vllm") == version and not any("2 CUDA" in p for p in problems)
    write_json(ENV_DIR / "install.json", res)
    return res


def safetensors_ok(path: Path) -> tuple[bool, str]:
    """Header-consistency check (no torch): 8-byte header length + JSON header whose largest
    data offset must end exactly at the file end. Catches truncated / interrupted downloads."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n <= 0 or n > size:
                return False, f"bad header length {n}"
            header = json.loads(f.read(n))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        ok = 8 + n + end == size
        return ok, f"expected {8 + n + end} bytes, file has {size}"
    except Exception as e:
        return False, repr(e)


def step_model(cfg: dict, args) -> dict:
    py = str(Path(args.venv) / "bin" / "python")
    repo = args.model or cfg["model"]["repo_id"]
    rev = args.revision if args.revision is not None else (cfg["model"]["revision"] if repo == cfg["model"]["repo_id"] else "main")
    target = MODEL_DIR / repo.split("/")[-1]
    code = (
        "from huggingface_hub import snapshot_download\n"
        f"p = snapshot_download({repo!r}, revision={rev!r}, local_dir={str(target)!r}, "
        "allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE'])\n"
        "print('MODEL_PATH=' + p)\n")
    t0 = time.time()
    rc, out = run_cmd([py, "-c", code], timeout=1800)
    if rc != 0:
        raise RuntimeError("model download failed:\n" + out[-3000:])
    cfgj = json.loads((target / "config.json").read_text())
    st = sorted(p.name for p in target.glob("*.safetensors"))
    size_gb = round(sum(p.stat().st_size for p in target.glob("*.safetensors")) / 1e9, 2)
    bad = {}
    for name in st:
        ok, msg = safetensors_ok(target / name)
        if not ok:
            bad[name] = msg
    if bad:
        raise RuntimeError(f"corrupted/truncated weights {bad}; delete {target} and rerun this step")
    res = {"ok": bool(st), "repo_id": repo, "revision": rev, "path": str(target), "safetensors": st,
           "weights_gb": size_gb, "download_seconds": round(time.time() - t0, 1),
           "num_hidden_layers": cfgj.get("num_hidden_layers"), "num_key_value_heads": cfgj.get("num_key_value_heads"),
           "hidden_size": cfgj.get("hidden_size"), "num_attention_heads": cfgj.get("num_attention_heads"),
           "torch_dtype_in_checkpoint": cfgj.get("torch_dtype")}
    write_json(ENV_DIR / "model.json", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["env", "model", "verify"], required=True)
    ap.add_argument("--venv", default=str(VENV))
    ap.add_argument("--vllm-version", default=None, help="override (fallback e.g. 0.9.2 with a different --venv)")
    ap.add_argument("--exclude-newer", default=None)
    ap.add_argument("--model", default=None, help="override repo id (fallback Qwen/Qwen2.5-0.5B-Instruct)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    budget = Budget(cfg)
    try:
        budget.enter_level("P0", force=args.force)
        if args.step == "env":
            res = step_env(cfg, args)
        elif args.step == "model":
            res = step_model(cfg, args)
        else:
            res = verify(str(Path(args.venv) / "bin" / "python"))
        print(json.dumps(res, indent=2))
        ok = res.get("ok", True)
        print("INSTALL_STEP", args.step, "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    except BudgetExceeded as e:
        print("BUDGET STOP:", e)
        sys.exit(2)
    except Exception as e:
        budget.mark_level("P0", "error", f"{args.step}: {e}"[:300])
        print("INSTALL_STEP", args.step, "FAIL:", e)
        print(f"Full log: {LOG_DIR / 'install.log'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
