"""PD KV-transfer probe, auto-imported by Python's `site` module when this directory is on
PYTHONPATH (the server launcher only adds it for architecture D).

It does NOT change vLLM behaviour: it wraps four methods of vLLM 0.8.5's
`SimpleConnector` (PyNcclConnector) and appends one JSON line per call to
$PDLAB_PROBE_LOG:

  producer:  send_batch (gather KV from paged cache + enqueue), insert (bytes enqueued)
  consumer:  select     (blocking fetch of one request's KV+hidden over PyNccl, incl. signalling)
             recv_batch (all selects + writing KV into the paged cache) with the official
                        `bypass_model_exec` flag: True means the decode instance skipped the
                        prefill forward pass and used the transferred KV/hidden states.

Measurement cost: one torch.cuda.synchronize() per call so that timings include the GPU work.

Separately, PDLAB_PAGED_KV_FIX=1 applies pdlab_kv_layout.patch to vLLM's KV helper so the
reference PD connector works with the PagedAttention (XFormers, Turing) cache layout.
That one DOES change behaviour (it is a functional fix) and is documented in the report.
"""
import os

_LOG = os.environ.get("PDLAB_PROBE_LOG")
_KVFIX = os.environ.get("PDLAB_PAGED_KV_FIX") == "1"

if _LOG or _KVFIX:
    import importlib.abc
    import importlib.util
    import json
    import sys
    import threading
    import time

    _lock = threading.Lock()

    def _emit(rec):
        if not _LOG:
            return
        rec["ts"] = time.time()
        rec["pid"] = os.getpid()
        try:
            with _lock, open(_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def _nb(t):
        try:
            return int(t.numel() * t.element_size())
        except Exception:
            return 0

    def _sync():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def _patch(mod):
        cls = mod.SimpleConnector
        o_select, o_insert = cls.select, cls.insert
        o_recv, o_send = cls.recv_kv_caches_and_hidden_states, cls.send_kv_caches_and_hidden_states

        def _safe(build):
            """Bookkeeping must never raise into vLLM."""
            try:
                _emit(build())
            except Exception as e:
                _emit({"ev": "probe_error", "error": repr(e)[:300]})

        def _seq_lens(args, kwargs):
            mi = kwargs.get("model_input", args[1] if len(args) > 1 else None)
            try:
                return list(mi.attn_metadata.seq_lens or [])
            except Exception:
                return []

        def select(self, input_tokens, roi):
            t0 = time.perf_counter()
            ret = o_select(self, input_tokens, roi)
            _sync()
            ms = round((time.perf_counter() - t0) * 1000, 4)

            def build():
                hit = ret is not None and ret[0] is not None
                rec = {"ev": "select", "hit": bool(hit), "n_tokens": int(input_tokens.numel()), "ms": ms}
                if hit:
                    rec.update({"matched_tokens": int(ret[1].shape[0]), "bytes_k": _nb(ret[2]), "bytes_v": _nb(ret[3]),
                                "bytes_hidden": _nb(ret[4]) if ret[4] is not None else 0,
                                "k_shape": list(ret[2].shape), "dtype": str(getattr(ret[2], "dtype", ""))})
                return rec
            _safe(build)
            return ret

        def insert(self, input_tokens, roi, key, value, hidden):
            t0 = time.perf_counter()
            o_insert(self, input_tokens, roi, key, value, hidden)
            ms = round((time.perf_counter() - t0) * 1000, 4)
            _safe(lambda: {"ev": "insert", "n_tokens": int(input_tokens.numel()), "bytes_k": _nb(key), "bytes_v": _nb(value),
                           "bytes_hidden": _nb(hidden), "k_shape": list(key.shape), "ms": ms})

        def recv(self, *args, **kwargs):
            t0 = time.perf_counter()
            out = o_recv(self, *args, **kwargs)
            _sync()
            ms = round((time.perf_counter() - t0) * 1000, 4)

            def build():
                seq_lens = _seq_lens(args, kwargs)
                return {"ev": "recv_batch", "bypass_model_exec": bool(out[1]), "n_reqs": len(seq_lens),
                        "n_tokens": int(sum(seq_lens)), "ms": ms}
            _safe(build)
            return out

        def send(self, *args, **kwargs):
            t0 = time.perf_counter()
            out = o_send(self, *args, **kwargs)
            _sync()
            ms = round((time.perf_counter() - t0) * 1000, 4)

            def build():
                seq_lens = _seq_lens(args, kwargs)
                return {"ev": "send_batch", "n_reqs": len(seq_lens), "n_tokens": int(sum(seq_lens)), "ms": ms}
            _safe(build)
            return out

        cls.select, cls.insert = select, insert
        cls.recv_kv_caches_and_hidden_states, cls.send_kv_caches_and_hidden_states = recv, send
        _emit({"ev": "patched", "module": mod.__name__})

    def _patch_layout(mod):
        import pdlab_kv_layout
        pdlab_kv_layout.patch(mod)
        _emit({"ev": "kv_layout_patched", "module": mod.__name__})

    _TARGETS = {}
    if _LOG:
        _TARGETS["vllm.distributed.kv_transfer.kv_connector.simple_connector"] = _patch
    if _KVFIX:
        _TARGETS["vllm.distributed.kv_transfer.kv_connector.utils"] = _patch_layout

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            fn = _TARGETS.get(fullname)
            if fn is None:
                return None
            sys.meta_path.remove(self)  # avoid recursing into ourselves while resolving
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                _TARGETS.pop(fullname, None)
                if _TARGETS:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    fn(module)
                except Exception as e:  # never break the server because of the probe
                    _emit({"ev": "patch_error", "module": fullname, "error": repr(e)})

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
