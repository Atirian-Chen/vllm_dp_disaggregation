"""Build the fixed token-ID workload shared by all four architectures.

Prompts are sent as token-ID lists, so every architecture sees exactly the same
input length (no re-tokenization drift) and the PD lookup buffer can match the
prefill/decode halves of a request by token sequence. Every prompt starts with a
unique request header, so there is no shared prefix between requests.

Run with the venv python (needs transformers):  python scripts/prepare_workload.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from common import ENV_DIR, WORKLOAD_PATH, ensure_dirs, load_config, now_iso, read_json, write_json

SUBJ = ["The scheduler", "A decode worker", "The KV cache manager", "Each prefill batch", "The router",
        "A pipeline stage", "The block allocator", "The tokenizer service", "An attention kernel",
        "The memory profiler", "A replica", "The load balancer", "The serving engine", "A request queue"]
VERB = ["allocates", "evicts", "transfers", "streams", "batches", "schedules", "copies", "reuses",
        "partitions", "validates", "measures", "throttles", "prefetches", "records"]
OBJ = ["paged blocks", "attention keys", "value tensors", "token budgets", "hidden states", "request metadata",
       "latency samples", "GPU memory", "sequence groups", "the logits buffer", "prompt chunks", "CUDA graphs"]
TAIL = ["before the next iteration", "under heavy load", "when the queue grows", "across two devices",
        "with a strict deadline", "after the warmup phase", "for every new sequence", "at a fixed interval",
        "without extra copies", "while decoding continues", "over the PCIe link", "during peak traffic"]


def make_text(rng: random.Random, n_sentences: int) -> str:
    out = []
    for _ in range(n_sentences):
        out.append(f"{rng.choice(SUBJ)} {rng.choice(VERB)} {rng.randint(2, 512)} {rng.choice(OBJ)} {rng.choice(TAIL)}.")
    return " ".join(out)


class MockTokenizer:
    """Local-test stand-in: one token per whitespace-separated word."""
    name_or_path = "mock-whitespace"

    def encode(self, text, add_special_tokens=False):
        return [abs(hash(w)) % 50000 + 100 for w in text.split()]


def build_prompt(tok, uid: str, n_tokens: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    head = tok.encode(f"Request {uid}. Read the following engineering notes carefully.\n", add_special_tokens=False)
    tail = tok.encode("\nQuestion: Summarize the notes above in plain English.\nAnswer:", add_special_tokens=False)
    body_needed = n_tokens - len(head) - len(tail)
    if body_needed <= 0:
        raise ValueError(f"n_tokens={n_tokens} too small")
    body = []
    n_sent = max(4, body_needed // 12)
    while len(body) < body_needed:
        body = tok.encode(" " + make_text(rng, n_sent), add_special_tokens=False)
        n_sent *= 2
    ids = head + body[:body_needed] + tail
    assert len(ids) == n_tokens
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock-tokenizer", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    if WORKLOAD_PATH.exists() and not args.force:
        print(f"workload exists: {WORKLOAD_PATH} (use --force to rebuild; NOTE rebuilding changes prompts)")
        return
    if args.mock_tokenizer:
        tok = MockTokenizer()
        tok_name = tok.name_or_path
    else:
        from transformers import AutoTokenizer
        model = read_json(ENV_DIR / "model.json")
        if not model:
            raise SystemExit("run install_or_check.py --step model first")
        tok = AutoTokenizer.from_pretrained(model["path"])
        tok_name = model["path"]

    data = {"meta": {"created": now_iso(), "tokenizer": tok_name, "config_workloads": cfg["workloads"],
                     "concurrency_levels": cfg["concurrency_levels"], "max_reps": cfg["max_reps"]},
            "smoke": [], "runs": {}}
    sm = cfg["smoke"]
    seed = 1000
    for i in range(sm["prefill_heavy_requests"]):
        w = cfg["workloads"]["prefill_heavy"]
        data["smoke"].append({"id": f"smoke-ph-{i}", "workload": "prefill_heavy", "max_tokens": w["output_tokens"],
                              "prompt_token_ids": build_prompt(tok, f"smoke-ph-{i}", w["input_tokens"], seed + i)})
    for i in range(sm["decode_heavy_requests"]):
        w = cfg["workloads"]["decode_heavy"]
        data["smoke"].append({"id": f"smoke-dh-{i}", "workload": "decode_heavy", "max_tokens": sm["decode_heavy_output_tokens"],
                              "prompt_token_ids": build_prompt(tok, f"smoke-dh-{i}", w["input_tokens"], seed + 100 + i)})
    seed = 10_000
    for wname, w in cfg["workloads"].items():
        for conc in cfg["concurrency_levels"]:
            for rep in range(1, cfg["max_reps"] + 1):
                key = f"{wname}|c{conc}|r{rep}"
                entry = {"warmup": [], "measure": []}
                # warmup >= concurrency so the first full-size batch (and any first-shape
                # kernel selection / JIT) happens before the timed window
                n_warm = max(cfg["warmup_requests"], conc)
                for phase, n in (("warmup", n_warm), ("measure", w["num_requests"])):
                    for i in range(n):
                        uid = f"{wname[:2]}-c{conc}-r{rep}-{phase[0]}{i}"
                        seed += 1
                        entry[phase].append({"id": uid, "workload": wname, "max_tokens": w["output_tokens"],
                                             "prompt_token_ids": build_prompt(tok, uid, w["input_tokens"], seed)})
                data["runs"][key] = entry
    blob = json.dumps(data["runs"], sort_keys=True).encode()
    data["meta"]["runs_sha256"] = hashlib.sha256(blob).hexdigest()
    lens = {k: sorted({len(x["prompt_token_ids"]) for x in v["measure"]}) for k, v in data["runs"].items()}
    data["meta"]["prompt_lengths"] = lens
    write_json(WORKLOAD_PATH, data)
    n = sum(len(v["warmup"]) + len(v["measure"]) for v in data["runs"].values())
    print(f"wrote {WORKLOAD_PATH}: {n} run prompts + {len(data['smoke'])} smoke prompts, sha256={data['meta']['runs_sha256'][:16]}")
    for k, v in lens.items():
        print(f"  {k}: input lengths {v}")


if __name__ == "__main__":
    main()
