"""Local (no GPU) tests. Run:  python tests/test_local.py

1. pd_probe/sitecustomize.py patches a fake vllm SimpleConnector through the import hook
   and logs select/insert/recv/send events (hit, miss, bytes, bypass flag).
2. percentile / summarize correctness on hand-computed inputs.
The end-to-end mock pipeline (launcher, proxy, client, runner, report) is exercised by
`tests/run_mock_pipeline.sh`.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FAKE_TORCH = '''
class _T:
    def __init__(self, n, es=2, shape=None):
        self.n, self.es, self.shape = n, es, shape or (n,)
    def numel(self): return self.n
    def element_size(self): return self.es
class cuda:
    @staticmethod
    def is_available(): return False
'''

FAKE_CONNECTOR = '''
from torch import _T
class SimpleConnector:
    def __init__(self, hit=True): self.hit = hit
    def select(self, input_tokens, roi):
        if not self.hit: return [None] * 5
        n = input_tokens.numel()
        return [input_tokens, _T(n, 1, (n,)), _T(28*n*2*128, 2, (28, n, 2, 128)), _T(28*n*2*128, 2, (28, n, 2, 128)), _T(n*1536)]
    def insert(self, input_tokens, roi, key, value, hidden): pass
    def recv_kv_caches_and_hidden_states(self, model_executable, model_input, kv_caches):
        for L in model_input.attn_metadata.seq_lens:
            r = self.select(_T(L, 8), None)
        return ("hidden", r[0] is not None, model_input)
    def send_kv_caches_and_hidden_states(self, model_executable, model_input, kv_caches, hidden_or_intermediate_states):
        for L in model_input.attn_metadata.seq_lens:
            self.insert(_T(L, 8), None, _T(28*L*256), _T(28*L*256), _T(L*1536))
'''

DRIVER = '''
import types
from torch import _T
from vllm.distributed.kv_transfer.kv_connector.simple_connector import SimpleConnector
mi = types.SimpleNamespace(attn_metadata=types.SimpleNamespace(seq_lens=[1024, 128]))
c = SimpleConnector(hit=True)
out = c.recv_kv_caches_and_hidden_states(None, mi, [])
assert out[1] is True
SimpleConnector(hit=False).recv_kv_caches_and_hidden_states(None, mi, [])
c.send_kv_caches_and_hidden_states(None, model_input=mi, kv_caches=[], hidden_or_intermediate_states=None)
print("driver ok")
'''


def test_probe():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "torch.py").write_text(FAKE_TORCH)
        pkg = d / "vllm" / "distributed" / "kv_transfer" / "kv_connector"
        pkg.mkdir(parents=True)
        for p in [d / "vllm", d / "vllm" / "distributed", d / "vllm" / "distributed" / "kv_transfer", pkg]:
            (p / "__init__.py").write_text("")
        (pkg / "simple_connector.py").write_text(FAKE_CONNECTOR)
        log = d / "probe.jsonl"
        env = dict(os.environ, PDLAB_PROBE_LOG=str(log),
                   PYTHONPATH=os.pathsep.join([str(ROOT / "scripts" / "pd_probe"), str(d)]))
        r = subprocess.run([sys.executable, "-c", DRIVER], env=env, capture_output=True, text=True, cwd=str(d))
        assert r.returncode == 0, r.stderr
        recs = [json.loads(x) for x in log.read_text().splitlines()]
        evs = [x["ev"] for x in recs]
        assert evs[0] == "patched", evs
        sel = [x for x in recs if x["ev"] == "select"]
        assert [x["hit"] for x in sel] == [True, True, False, False], sel
        assert sel[0]["bytes_k"] == 28 * 1024 * 2 * 128 * 2 and sel[0]["matched_tokens"] == 1024
        rb = [x for x in recs if x["ev"] == "recv_batch"]
        assert [x["bypass_model_exec"] for x in rb] == [True, False] and rb[0]["n_tokens"] == 1152
        assert sum(1 for x in recs if x["ev"] == "insert") == 2
        sb = [x for x in recs if x["ev"] == "send_batch"]
        assert sb and sb[0]["n_reqs"] == 2, sb  # keyword-argument call path
        # evidence aggregation over these records
        from evidence import pd_probe_stats
        st = pd_probe_stats(None, str(log), 0, 1e12)
        assert st["n_select_hit"] == 2 and st["n_select_miss"] == 2 and st["n_recv_bypass_false"] == 1
        assert st["n_select_begin"] == 4 and st["n_select_pending"] == 0, st
        # a select that never returns (0.8.5: KV that was never produced) shows up as pending
        with open(log, "a") as f:
            f.write(json.dumps({"ev": "select_begin", "n_tokens": 128, "ts": 2e12}) + "\n")
        st2 = pd_probe_stats(None, str(log), 1.5e12, 3e12)
        assert st2["n_select_pending"] == 1 and st2["n_select"] == 0 and st2["n_recv_bypass_false"] == 0, st2
        assert st["kv_bytes_per_token"] == 28672.0, st
    print("test_probe OK")


def test_probe_inactive_without_env():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "scripts" / "pd_probe"))
    env.pop("PDLAB_PROBE_LOG", None)
    r = subprocess.run([sys.executable, "-c", "import sys; print(any(type(f).__name__=='_Finder' for f in sys.meta_path))"],
                       env=env, capture_output=True, text=True)
    assert r.stdout.strip() == "False", r.stdout + r.stderr
    print("test_probe_inactive_without_env OK")


def test_metrics():
    from common import percentile
    from client import summarize
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert abs(percentile(list(range(1, 101)), 95) - 95.05) < 1e-9
    reqs = [{"ok": True, "itl_ms": [10, 10, 30], "output_tokens": 4, "prompt_tokens": 8, "ttft_ms": 100.0,
             "tpot_ms": 50 / 3, "e2e_ms": 150.0, "chunks_eq_tokens": True},
            {"ok": False, "error": "HTTP 500"}]
    s = summarize(reqs, 2.0)
    assert s["n_ok"] == 1 and s["success_rate"] == 0.5 and s["output_tok_per_s"] == 2.0
    assert s["itl_p50_ms"] == 10 and s["ttft_p95_ms"] == 100.0 and s["errors"] == ["HTTP 500"]
    print("test_metrics OK")


if __name__ == "__main__":
    test_probe()
    test_probe_inactive_without_env()
    test_metrics()
    print("ALL LOCAL TESTS PASSED")
