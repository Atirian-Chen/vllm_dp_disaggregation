"""Paged-KV-layout adapter for vLLM 0.8.5's PD KV helper (loaded only into architecture D).

Why: `vllm/distributed/kv_transfer/kv_connector/utils.py::model_aware_kv_ops_helper`
hard-codes the FlashAttention cache layout [2, num_blocks, block_size, kv_heads, head_dim]:
  - get_kv_from_cache  reads slots with kv_cache[i].reshape(-1, kv_heads, head_dim)
  - put_kv_to_cache    writes with ops.reshape_and_cache_flash
On Turing (T4) V0 uses the XFormers backend, whose cache is PagedAttention's layout
kv_cache = [2, num_blocks, block_size * kv_heads * head_dim], viewed as
  key   [num_blocks, kv_heads, head_dim // x, block_size, x]   (x = 16 bytes / elem size)
  value [num_blocks, kv_heads, head_dim, block_size]
With that layout the stock read returns scrambled values and the stock write crashes with
"IndexError: Dimension out of range" (observed on Kaggle 2xT4).

What this changes: only when kv_cache is the 3-D paged layout, the read gathers each slot
(block = slot // block_size, offset = slot % block_size) and the write uses
PagedAttention.split_kv_cache + ops.reshape_and_cache -- the same ops the XFormers backend
itself uses to fill its cache. Transferred tensors keep the standard [tokens, kv_heads,
head_dim] format. The PyNccl pipe / lookup buffer / request matching are untouched, and
5-D (FlashAttention) caches still go through the original code.
"""


class _PagedSlotGather:
    """Indexable by a slot tensor like the FlashAttention view: returns [T, kv_heads, head_dim]."""

    def __init__(self, cache, is_key: bool):
        self.cache = cache
        self.is_key = is_key
        self.block_size = cache.shape[3]

    def __getitem__(self, slots):
        import torch
        slots = slots.long()
        blk = torch.div(slots, self.block_size, rounding_mode="floor")
        off = slots % self.block_size
        if self.is_key:  # [nb, H, D/x, bs, x] -> [T, H, D/x, x]
            out = self.cache[blk, :, :, off, :]
        else:            # [nb, H, D, bs] -> [T, H, D]
            out = self.cache[blk, :, :, off]
        return out.reshape(out.shape[0], out.shape[1], -1)


def patch(mod):
    helper = mod.model_aware_kv_ops_helper
    o_get, o_put = helper.get_kv_from_cache, helper.put_kv_to_cache

    def get_kv_from_cache(self, kv_cache, num_heads, head_size):
        if kv_cache.dim() != 3 or getattr(self, "is_deepseek_mla", False):
            return o_get(self, kv_cache, num_heads, head_size)
        from vllm.attention.ops.paged_attn import PagedAttention
        key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, num_heads, head_size)
        return _PagedSlotGather(key_cache, True), _PagedSlotGather(value_cache, False)

    def put_kv_to_cache(self, model_executable, keys, values, layer, kv_cache, slot_mapping, start_pos, end_pos):
        if kv_cache.dim() != 3 or getattr(self, "is_deepseek_mla", False):
            return o_put(self, model_executable, keys, values, layer, kv_cache, slot_mapping, start_pos, end_pos)
        from vllm import _custom_ops as ops
        from vllm.attention.ops.paged_attn import PagedAttention
        num_heads, head_size = keys.shape[1], keys.shape[2]
        key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, num_heads, head_size)
        attn = layer.self_attn.attn
        ops.reshape_and_cache(keys.to(kv_cache.device).contiguous(), values.to(kv_cache.device).contiguous(),
                              key_cache, value_cache, slot_mapping[start_pos:end_pos].flatten(),
                              attn.kv_cache_dtype, attn._k_scale, attn._v_scale)

    helper.get_kv_from_cache = get_kv_from_cache
    helper.put_kv_to_cache = put_kv_to_cache
