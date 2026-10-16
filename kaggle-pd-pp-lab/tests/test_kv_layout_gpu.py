"""GPU test (needs CUDA + the vLLM 0.8.5 venv):  <venv>/bin/python tests/test_kv_layout_gpu.py

Uses vLLM's real ops on the PagedAttention (XFormers / Turing) cache layout to show that
  1. the stock PD helper reads scrambled KV and its write raises the IndexError seen on Kaggle;
  2. the pdlab_kv_layout adapter reads exactly what the XFormers backend wrote and writes
     received KV back bit-identically.
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pd_probe"))
import pdlab_kv_layout  # noqa: E402
from vllm.attention.ops.paged_attn import PagedAttention  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector import utils  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
NB, BS, H, D, T = 16, 16, 2, 128, 100  # Qwen2.5-1.5B: 2 KV heads, head_dim 128; vLLM block size 16
dtype = torch.float16
one = torch.tensor(1.0, device=dev)

layer = types.SimpleNamespace(self_attn=types.SimpleNamespace(
    attn=types.SimpleNamespace(kv_cache_dtype="auto", _k_scale=one, _v_scale=one)))
model_exec = types.SimpleNamespace(model=types.SimpleNamespace(config=types.SimpleNamespace()))
keys = torch.randn(T, H, D, device=dev, dtype=dtype)
values = torch.randn(T, H, D, device=dev, dtype=dtype)
slots = torch.randperm(NB * BS, device=dev)[:T]

# reference: what the XFormers backend itself writes during a prefill
kv_ref = torch.zeros(PagedAttention.get_kv_cache_shape(NB, BS, H, D), device=dev, dtype=dtype)
kc, vc = PagedAttention.split_kv_cache(kv_ref, H, D)
PagedAttention.write_to_paged_cache(keys, values, kc, vc, slots, "auto", one, one)

helper = utils.model_aware_kv_ops_helper.__new__(utils.model_aware_kv_ops_helper)
helper.is_deepseek_mla, helper.use_mla_opt, helper.tp_size = False, True, 1

# 1. stock behaviour
k_stock = utils.model_aware_kv_ops_helper.get_kv_from_cache(helper, kv_ref, H, D)[0][slots]
stock_read_ok = torch.equal(k_stock, keys)
try:
    utils.model_aware_kv_ops_helper.put_kv_to_cache(helper, model_exec, keys, values, layer, torch.zeros_like(kv_ref), slots, 0, T)
    stock_write = "no error"
except Exception as e:
    stock_write = f"{type(e).__name__}: {e}"
print(f"stock read equals written KV: {stock_read_ok}   stock write: {stock_write}")
assert not stock_read_ok and "Dimension out of range" in stock_write

# 2. adapter
pdlab_kv_layout.patch(utils)
kg, vg = helper.get_kv_from_cache(kv_ref, H, D)
assert torch.equal(kg[slots], keys) and torch.equal(vg[slots], values), "adapter read mismatch"
kv_new = torch.zeros_like(kv_ref)
helper.put_kv_to_cache(model_exec, kg[slots], vg[slots], layer, kv_new, slots, 0, T)
assert torch.equal(kv_new, kv_ref), "adapter write does not reproduce the backend's own cache"
# FlashAttention layout (5-D) still goes through the original code path
fa = torch.zeros(2, NB, BS, H, D, device=dev, dtype=dtype)
assert not isinstance(helper.get_kv_from_cache(fa, H, D)[0], pdlab_kv_layout._PagedSlotGather)
print("adapter: read == written KV, write == backend cache (bit-exact), FA path untouched -> OK")
