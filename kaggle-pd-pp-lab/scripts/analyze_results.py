"""Aggregate runs -> summary.csv, figures/*.png, EXPERIMENT_REPORT.md, results ZIP.

Runs with the Kaggle system python (needs matplotlib only). Every number in the report
comes from results/runs/*.json and results/smoke/*.json; unmeasured cells print N/A.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
import zipfile
from pathlib import Path

from common import (ARCH_NAMES, ENV_DIR, RESULTS, RUNS_DIR, SMOKE_DIR, STATE_DIR, load_config, mean, percentile,
                    read_json, read_jsonl)

ARCHS = ["A", "B", "C", "D"]
ARCH_SHORT = {"A": "A 单卡混合", "B": "B DP2", "C": "C PP2", "D": "D PD分离"}
ARCH_EN = {"A": "A single", "B": "B DP2", "C": "C PP2", "D": "D PD-disagg"}
# reference categorical palette (dataviz skill), fixed order, slots 1-4
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
WL_CN = {"prefill_heavy": "Prefill-heavy", "decode_heavy": "Decode-heavy"}


def fmt(v, nd=1):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def load_runs(include_mock: bool):
    runs = []
    for p in sorted(glob.glob(str(RUNS_DIR / "*__*__c*__r*.json"))):
        d = read_json(p)
        if d and d.get("complete") and (include_mock or not d.get("mock")):
            runs.append(d)
    return runs


def slo_thresholds(runs, cfg):
    s = cfg["slo"]
    out = {}
    lo = min(cfg["concurrency_levels"])
    for w in cfg["workloads"]:
        base = [r for run in runs if run["arch"] == "A" and run["workload"] == w and run["concurrency"] == lo
                for r in run["requests"] if r["ok"]]
        if base:
            out[w] = {"ttft_ms": round(s["ttft_factor"] * percentile([r["ttft_ms"] for r in base], 50), 1),
                      "tpot_ms": round(s["tpot_factor"] * percentile([r["tpot_ms"] for r in base], 50), 2),
                      "source": f"A c={lo}: {s['ttft_factor']}xTTFT_P50, {s['tpot_factor']}xTPOT_P50"}
        else:
            out[w] = dict(s["fallback_abs"][w], source="absolute fallback (A missing)")
    return out


def add_goodput(run, slo):
    t = slo[run["workload"]]
    reqs = run["requests"]
    good = [r for r in reqs if r["ok"] and r["ttft_ms"] <= t["ttft_ms"] and (r["tpot_ms"] or 0) <= t["tpot_ms"]]
    run["summary"]["slo_attainment"] = round(len(good) / len(reqs), 4) if reqs else None
    run["summary"]["goodput_req_s"] = round(len(good) / run["summary"]["wall_s"], 4)


def pd_breakdown(run):
    """Per-request join of client TTFT with proxy timings (by request id)."""
    proxy_log = RESULTS / "logs" / "proxy_D.jsonl"
    if not proxy_log.exists():
        return None
    recs = {r["rid"]: r for r in read_jsonl(proxy_log)
            if run["t_start_epoch"] - 1 <= r.get("ts", 0) <= run["t_end_epoch"] + 1}
    rows = []
    for r in run["requests"]:
        pr = recs.get(r["id"])
        if r["ok"] and pr and pr.get("prefill_ms") is not None and pr.get("decode_first_byte_ms") is not None:
            rows.append((r["ttft_ms"], pr["prefill_ms"], pr["decode_first_byte_ms"]))
    if not rows:
        return None
    xfer = (run.get("pd") or {}).get("transfer_ms_p50") or 0.0
    ttft = percentile([x[0] for x in rows], 50)
    pre = percentile([x[1] for x in rows], 50)
    dfb = percentile([x[2] for x in rows], 50)
    return {"n": len(rows), "ttft_p50": ttft, "prefill_rt_p50": pre, "decode_first_p50": dfb,
            "kv_transfer_p50": xfer, "decode_first_minus_xfer_p50": max(dfb - xfer, 0.0),
            "proxy_client_overhead_p50": percentile([x[0] - x[1] - x[2] for x in rows], 50),
            "kv_share_of_ttft": xfer / ttft if ttft else None}


def aggregate(runs):
    """(arch, workload, conc) -> mean over reps of each summary metric (+ min/max)."""
    groups = {}
    for r in runs:
        groups.setdefault((r["arch"], r["workload"], r["concurrency"]), []).append(r)
    keys = ["ttft_p50_ms", "ttft_p95_ms", "itl_p50_ms", "itl_p95_ms", "itl_p99_ms", "tpot_mean_ms", "tpot_p95_ms",
            "e2e_p50_ms", "e2e_p95_ms", "output_tok_per_s", "req_per_s", "success_rate", "slo_attainment",
            "goodput_req_s", "chunks_eq_tokens_frac"]
    agg = {}
    for k, rs in groups.items():
        a = {"n_reps": len(rs), "n_requests": sum(r["summary"]["n_requests"] for r in rs)}
        for m in keys:
            vals = [r["summary"].get(m) for r in rs if r["summary"].get(m) is not None]
            a[m] = mean(vals)
            a[m + "_min"], a[m + "_max"] = (min(vals), max(vals)) if vals else (None, None)
        g = {}
        for r in rs:
            for gi, gv in (r.get("gpu", {}).get("per_gpu") or {}).items():
                g.setdefault(gi, {"mem": [], "util": []})
                g[gi]["mem"].append(gv["mem_peak_mib"])
                g[gi]["util"].append(gv["util_mean_pct"])
        a["gpu"] = {gi: {"mem_peak_mib": max(v["mem"]), "util_mean_pct": mean(v["util"])} for gi, v in g.items()}
        if k[0] == "D":
            pds = [r.get("pd") for r in rs if r.get("pd")]
            a["pd"] = {f: mean([p.get(f) for p in pds]) for f in ("transfer_ms_p50", "transfer_ms_p95", "kv_bytes_per_req_mean",
                                                                    "effective_GBps", "kv_bytes_per_token")}
            a["pd"]["n_select_hit"] = sum(p["n_select_hit"] for p in pds)
            a["pd"]["n_recv_bypass_false"] = sum(p["n_recv_bypass_false"] for p in pds)
            bds = [b for b in (pd_breakdown(r) for r in rs) if b]
            if bds:
                a["pd_breakdown"] = {f: mean([b[f] for b in bds if b[f] is not None]) for f in bds[0] if f != "n"}
        agg[k] = a
    return agg


# ---------------------------------------------------------------- figures
def figures(agg, cfg, outdir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable, skipping figures:", e)
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.titlecolor": "#0b0b0b", "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True})
    groups = [(w, c) for w in cfg["workloads"] for c in cfg["concurrency_levels"]]
    made = []

    def grouped(metric, title, ylabel, fname, digits=0):
        fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
        width = 0.19
        for i, a in enumerate(ARCHS):
            xs, ys = [], []
            for gi, (w, c) in enumerate(groups):
                v = agg.get((a, w, c), {}).get(metric)
                if v is not None:
                    xs.append(gi + (i - 1.5) * (width + 0.01))
                    ys.append(v)
            if xs:
                bars = ax.bar(xs, ys, width=width, color=COLORS[a], label=ARCH_EN[a], edgecolor="#fcfcfb", linewidth=1)
                for x, y in zip(xs, ys):
                    ax.text(x, y, f"{y:.{digits}f}", ha="center", va="bottom", fontsize=6.5, color="#52514e")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{WL_CN[w]}\nconc={c}" for w, c in groups])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.18))
        fig.tight_layout()
        p = outdir / fname
        fig.savefig(p)
        plt.close(fig)
        made.append(p)

    grouped("ttft_p95_ms", "TTFT P95 (ms, lower is better)", "ms", "ttft_p95.png")
    grouped("itl_p95_ms", "ITL P95 per streamed token (ms, lower is better)", "ms", "itl_p95.png", 1)
    grouped("output_tok_per_s", "Output throughput (tokens/s, higher is better)", "tok/s", "throughput.png")
    grouped("tpot_mean_ms", "TPOT mean (ms/token, lower is better)", "ms", "tpot_mean.png", 1)
    grouped("goodput_req_s", "Goodput (req/s meeting TTFT+TPOT SLO)", "req/s", "goodput.png", 2)

    # PD breakdown
    rows = [((w, c), agg[("D", w, c)]["pd_breakdown"]) for w, c in groups if ("D", w, c) in agg and agg[("D", w, c)].get("pd_breakdown")]
    if rows:
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=140)
        parts = [("prefill_rt_p50", "prefill request round-trip (queue+prefill on GPU0)", "#2a78d6"),
                 ("kv_transfer_p50", "KV transfer (PyNccl select)", "#eb6834"),
                 ("decode_first_minus_xfer_p50", "decode first token (excl. transfer)", "#1baf7a"),
                 ("proxy_client_overhead_p50", "proxy/client overhead", "#8a8984")]
        for yi, (g, b) in enumerate(rows):
            left = 0
            for k, lab, col in parts:
                v = max(b.get(k) or 0, 0)
                ax.barh(yi, v, left=left, color=col, edgecolor="#fcfcfb", linewidth=1.5, height=0.55,
                        label=lab if yi == 0 else None)
                left += v
            ax.text(left, yi, f"  TTFT p50 {b['ttft_p50']:.0f} ms", va="center", fontsize=7, color="#52514e")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{WL_CN[w]} c={c}" for (w, c), _ in rows])
        ax.set_xlabel("ms (median of each component; components are approximate)")
        ax.set_title("D: PD-disaggregated TTFT decomposition", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.25), ncol=2)
        fig.tight_layout()
        p = outdir / "pd_ttft_breakdown.png"
        fig.savefig(p)
        plt.close(fig)
        made.append(p)
    return made


# ---------------------------------------------------------------- report
def pctf(v):
    return "N/A" if v is None else f"{v:.0f}%"


def ratio(a, b):
    return a / b if (a is not None and b not in (None, 0)) else None


def pct_change(a, b):
    r = ratio(a, b)
    return None if r is None else (r - 1) * 100


def observations(agg, cfg):
    """Purely data-driven sentences; nothing here is asserted without the numbers."""
    obs = []
    lo, hi = min(cfg["concurrency_levels"]), max(cfg["concurrency_levels"])
    for w in cfg["workloads"]:
        for c in cfg["concurrency_levels"]:
            present = [a for a in ARCHS if (a, w, c) in agg]
            if len(present) < 2:
                continue
            for m, better, name in (("ttft_p95_ms", min, "TTFT P95"), ("itl_p95_ms", min, "ITL P95"),
                                    ("output_tok_per_s", max, "输出吞吐"), ("goodput_req_s", max, "Goodput")):
                vals = {a: agg[(a, w, c)][m] for a in present if agg[(a, w, c)].get(m) is not None}
                if vals and not any(vals.values()):
                    obs.append(f"{WL_CN[w]} c={c}：{name} 全部为 0（没有架构满足 SLO）")
                elif vals:
                    best = better(vals, key=vals.get)
                    obs.append(f"{WL_CN[w]} c={c}：{name} 最优为 {ARCH_SHORT[best]}（{fmt(vals[best])}）；"
                               + "，".join(f"{ARCH_SHORT[a]}={fmt(v)}" for a, v in vals.items()))
    return obs


def comparison_lines(agg, cfg):
    """Key ratios used by the mechanism discussion."""
    L = {}
    lo, hi = min(cfg["concurrency_levels"]), max(cfg["concurrency_levels"])
    for w in cfg["workloads"]:
        for c in cfg["concurrency_levels"]:
            A = agg.get(("A", w, c))
            for x in "BCD":
                X = agg.get((x, w, c))
                if A and X:
                    L[(x, w, c)] = {m: pct_change(X.get(m), A.get(m)) for m in
                                    ("ttft_p50_ms", "ttft_p95_ms", "itl_p50_ms", "itl_p95_ms", "tpot_mean_ms",
                                     "output_tok_per_s", "goodput_req_s")}
    return L


def sgn(v, unit="%"):
    return "N/A" if v is None else f"{v:+.0f}{unit}"


def mechanism_text(agg, cfg, cmp):
    lo, hi = min(cfg["concurrency_levels"]), max(cfg["concurrency_levels"])
    T = []

    def get(x, w, c, m):
        return (cmp.get((x, w, c)) or {}).get(m)

    # PP2
    pp = [(w, c, get("C", w, c, "ttft_p50_ms"), get("C", w, c, "tpot_mean_ms")) for w in cfg["workloads"] for c in cfg["concurrency_levels"] if ("C", w, c) in cmp]
    if pp:
        T.append("**PP2（C）相对单卡（A）**：" + "；".join(f"{WL_CN[w]} c={c} TTFT P50 {sgn(a)}、TPOT {sgn(b)}" for w, c, a, b in pp) + "。")
        slower = [x for x in pp if x[3] is not None and x[3] > 5]
        faster = [x for x in pp if x[3] is not None and x[3] < -5]
        T.append(f"TPOT 比单卡高 5% 以上的配置 {len(slower)}/{len(pp)} 个，低 5% 以上的 {len(faster)}/{len(pp)} 个。"
                 "机制上：1.5B 模型单卡即可放下，PP 不减少每个 token 的计算量（仍要串行经过全部 28 层），却在每个 step 增加一次跨卡激活传输（PCIe、无 NVLink）"
                 "以及两个 stage 之间的同步/调度开销；vLLM V0 用 2 个 virtual engine 交错不同批次来填充流水线，只有同时有 ≥2 个批次在飞时才可能摊薄气泡。"
                 + ("本次实测中 PP2 的 TPOT 高于单卡的配置占多数，与上述开销一致。" if len(slower) > len(pp) / 2 else
                    "本次实测并未出现“PP2 普遍更慢”，与上述预期不一致；需要结合两卡利用率与更多重复来判断，样本有限，不做强结论。"))
    # DP2
    dp = [(w, c, get("B", w, c, "output_tok_per_s"), get("B", w, c, "ttft_p50_ms")) for w in cfg["workloads"] for c in cfg["concurrency_levels"] if ("B", w, c) in cmp]
    if dp:
        T.append("**DP2（B）相对单卡**：" + "；".join(f"{WL_CN[w]} c={c} 输出吞吐 {sgn(v)}、TTFT P50 {sgn(t)}" for w, c, v, t in dp) + "。")
        gains = [v for _, _, v, _ in dp if v is not None]
        best = max(gains) if gains else None
        T.append("DP2 是两份完全独立的副本：单个请求不会被两张卡加速，收益只来自把并发请求分到两份计算/KV 资源上（每个副本的有效 batch 减半、prefill 排队被分摊）。"
                 f"理论吞吐上限为单卡的 2 倍（+100%），本次实测最大增益 {sgn(best)}。"
                 "单卡未饱和时（小模型、低并发，decode 以访存为主，batch 增大时步时几乎不变）增益通常明显小于 +100%；"
                 "闭环固定并发下若单卡的排队延迟随 batch 超线性增长，DP2 把每卡并发减半后增益也可能超过 +100%。低并发（c=2）下 DP2 还要承担代理转发开销。"
                 "是否“值得”两张卡，应看相同 SLO 下的 Goodput 而不是峰值吞吐：见表中 Goodput 列。")
    # PD
    pdl = [(w, c, get("D", w, c, "itl_p95_ms"), get("D", w, c, "ttft_p50_ms"), get("D", w, c, "output_tok_per_s"))
           for w in cfg["workloads"] for c in cfg["concurrency_levels"] if ("D", w, c) in cmp]
    if pdl:
        T.append("**PD 分离（D）相对单卡**：" + "；".join(f"{WL_CN[w]} c={c} ITL P95 {sgn(i)}、TTFT P50 {sgn(t)}、吞吐 {sgn(o)}" for w, c, i, t, o in pdl) + "。")
        improved = [x for x in pdl if x[2] is not None and x[2] < -5]
        worse = [x for x in pdl if x[2] is not None and x[2] > 5]
        if improved:
            T.append("在 " + "、".join(f"{WL_CN[w]} c={c}" for w, c, *_ in improved) +
                     " 中 D 的 ITL P95 比单卡低 5% 以上，与 PD 分离的核心机制一致：V0 关闭 chunked prefill 时，新请求的整段 prefill 作为独立 step 插入，"
                     "正在 decode 的请求必须等它完成，单卡混合部署的 ITL 尾部因此被拉长；D 把 prefill 移到 GPU0，GPU1 上对应的 step 只需把收到的 KV 写入分页缓存"
                     "（bypass_model_exec=True），对 decode 的打断短得多。")
        if worse:
            T.append("在 " + "、".join(f"{WL_CN[w]} c={c}" for w, c, *_ in worse) +
                     " 中 D 的 ITL P95 反而比单卡高 5% 以上：consumer 侧接收 KV（同步阻塞的 PyNccl select）和写入分页缓存的 step 同样会插在 decode step 之间，"
                     "当这部分耗时与单卡上一次 prefill 相当（输入短、prefill 本来就快）时，PD 分离无法改善甚至恶化 decode 尾延迟。")
        if not improved:
            T.append("本次数据中没有任何配置的 ITL P95 改善超过 5%，因此不能声称 PD 分离改善了 decode 尾延迟。")
        tt = [x for x in pdl if x[3] is not None]
        if tt:
            n_up = sum(1 for x in tt if x[3] > 0)
            T.append(f"TTFT：D 的 TTFT P50 高于单卡的配置 {n_up}/{len(tt)} 个。D 的首 token 路径是“代理 → prefill 实例完成 prefill（max_tokens=1）→ 返回代理 → 代理请求 decode 实例 → "
                     "decode 端经 PyNccl 拉取 KV 与 hidden states → 写入分页 KV → 采样首 token”，比单卡多一次 HTTP 往返、一次串行 KV 传输和一次 consumer 侧调度；"
                     "只有在高并发、单卡上 prefill 排队严重时，D 才可能因为 prefill 有专属 GPU 而降低 TTFT。")
        tp = [x for x in pdl if x[4] is not None]
        if tp:
            n_up = sum(1 for x in tp if x[4] > 5)
            T.append(f"吞吐：D 的输出吞吐比单卡高 5% 以上的配置 {n_up}/{len(tp)} 个。D 用两张卡，但吞吐受两端中较慢的一端限制："
                     "Decode-heavy 下 GPU0 大部分时间只做很短的 prefill，GPU1 承担全部 decode，等效于“一张卡 decode + 一张卡大部分空闲”，因此 PD 分离不一定提升总吞吐（可对照 GPU 利用率表）。")
    # KV transfer cost
    kv = [(w, c, agg[("D", w, c)]) for w in cfg["workloads"] for c in cfg["concurrency_levels"] if ("D", w, c) in agg]
    if kv:
        parts = []
        for w, c, a in kv:
            p = a.get("pd", {})
            b = a.get("pd_breakdown", {})
            share = b.get("kv_share_of_ttft")
            parts.append(f"{WL_CN[w]} c={c}：每请求 KV {fmt((p.get('kv_bytes_per_req_mean') or 0) / 1e6, 1)} MB，传输 P50 {fmt(p.get('transfer_ms_p50'))} ms"
                         f"（约 {fmt(p.get('effective_GBps'), 2)} GB/s），占 TTFT P50 约 {fmt(share * 100 if share is not None else None, 0)}%")
        T.append("**KV Cache 传输成本**：" + "；".join(parts) + "。理论值：Qwen2.5-1.5B 每 token KV = 28 层 × 2(K,V) × 2 KV 头 × 128 维 × 2 字节 = 28,672 字节，"
                 "1024 token ≈ 29.4 MB（另外 SimpleConnector 还传输全部 token 的 hidden states，约 3 KB/token）。GQA（2 个 KV 头）让这个模型的 KV 很小；"
                 "对 MHA 或更大的模型、更长的输入，传输量成比例增长，而 T4 之间只有 PCIe（Kaggle 上通常无 P2P，经主机内存中转），传输占比会显著上升，"
                 "这就是 PD 收益可能被传输成本抵消的条件：输入很长、KV 头多、链路慢，而 decode 侧被 prefill 打断的代价又不大（低并发、短输入）。")
    return T


def report(cfg, runs, agg, slo, figs, include_mock):
    env = read_json(ENV_DIR / "env_check.json", {})
    inst = read_json(ENV_DIR / "install.json", {})
    model = read_json(ENV_DIR / "model.json", {})
    budget = read_json(STATE_DIR / "budget.json", {})
    smoke = {a: read_json(SMOKE_DIR / f"{a}.json") for a in ARCHS}
    v = inst.get("verify", {})
    mock = any(r.get("mock") for r in runs) or any((s or {}).get("mock") for s in smoke.values())
    L = []
    L.append("# Kaggle 2×T4：单卡 / DP2 / PP2 / PD 分离推理架构小规模对比实验报告\n")
    if mock:
        L.append("> ⚠️ **本报告包含 MOCK（本地假服务器）数据，仅用于测试流水线，不代表任何真实性能。**\n")
    L.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}。所有数值均由 `results/runs/*.json` 与 `results/smoke/*.json` 自动汇总；未测量项为 N/A。\n")

    L.append("## 1. 实验环境\n")
    gpus = env.get("gpus", [])
    L.append("| 项目 | 值 |\n|---|---|")
    gpu_desc = ", ".join("GPU%d %s %.0f MiB" % (g["index"], g["name"], g["mem_total_mib"]) for g in gpus) or "N/A"
    L.append(f"| GPU | {gpu_desc} |")
    L.append(f"| GPU 细节 (compute cap, driver, PCIe gen/width) | {env.get('gpu_details_raw', 'N/A').replace(chr(10), ' / ')} |")
    L.append(f"| GPU P2P (torch.can_device_access_peer 0→1) | {v.get('p2p_0_1', 'N/A')} |")
    L.append(f"| vLLM / 引擎 | {v.get('vllm', 'N/A')}，V0 引擎（VLLM_USE_V1=0；0.8.5 在 CC<8.0 上本就回退 V0），XFormers attention |")
    L.append(f"| PyTorch / CUDA / NCCL | {v.get('torch', 'N/A')} / {v.get('torch_cuda', 'N/A')} / {v.get('nccl', 'N/A')} |")
    L.append(f"| transformers / xformers / Python | {v.get('transformers', 'N/A')} / {v.get('xformers', 'N/A')} / {v.get('python', 'N/A')} |")
    L.append(f"| 依赖解析 | uv `--exclude-newer {inst.get('exclude_newer', 'N/A')}`，完整列表见 env/venv_freeze.txt |")
    L.append(f"| 模型 | {model.get('repo_id', 'N/A')} @ `{model.get('revision', 'N/A')}`，FP16（checkpoint 为 {model.get('torch_dtype_in_checkpoint', '?')}），权重 {model.get('weights_gb', 'N/A')} GB |")
    vc = cfg["vllm"]
    L.append(f"| 统一服务参数 | dtype={vc['dtype']}, max_model_len={vc['max_model_len']}, gpu_mem_util={vc['gpu_memory_utilization']}, max_num_seqs={vc['max_num_seqs']}, "
             f"max_num_batched_tokens={vc['max_num_batched_tokens']}, chunked prefill=OFF, prefix caching=OFF, CUDA graph={'OFF (eager)' if vc['enforce_eager'] else 'ON'}, seed={vc['seed']} |")
    L.append(f"| 采样 | temperature=0, ignore_eos=true, 固定 max_tokens；prompt 以 token ID 发送，所有架构完全相同 |")
    if budget:
        L.append(f"| 会话计时 | 自 {budget.get('session_start_iso')} 起，生成报告时已用 {(time.time() - budget['session_start']) / 3600:.2f} h |")
    L.append("")

    L.append("## 2. 架构对照与真实性验证\n")
    L.append("| 架构 | 启动方式 | 权重分布 | KV Cache 路径 | Smoke | 启动耗时 |\n|---|---|---|---|---|---|")
    desc = {
        "A": ("1 个 vLLM 实例，CUDA_VISIBLE_DEVICES=0", "GPU0 完整模型，GPU1 空闲", "GPU0 本地分页 KV"),
        "B": ("2 个独立 vLLM 实例 + 轮询代理", "GPU0、GPU1 各一份完整模型", "各副本本地，请求级分发"),
        "C": ("1 个 vLLM 实例，--pipeline-parallel-size 2 --tensor-parallel-size 1，mp backend", "GPU0 前 14 层+embedding，GPU1 后 14 层+lm_head", "各 stage 保存自己层的 KV；层间传激活"),
        "D": ("prefill 实例(kv_producer, GPU0) + decode 实例(kv_consumer, GPU1)，PyNcclConnector + PD 代理", "两卡各一份完整模型", "GPU0 prefill → PyNccl(NCCL) 发送 KV+hidden → GPU1 写入分页 KV 后继续 decode"),
    }
    for a in ARCHS:
        s = smoke[a] or {}
        L.append(f"| {ARCH_SHORT[a]} | {desc[a][0]} | {desc[a][1]} | {desc[a][2]} | {s.get('status', '未运行')} | {fmt(s.get('startup_s'))} s |")
    L.append("")
    L.append("**真实性证据（来自 smoke test 的日志与 nvidia-smi）：**\n")
    for a in ARCHS:
        s = smoke[a]
        if not s:
            L.append(f"- {ARCH_SHORT[a]}：未运行 smoke test。")
            continue
        ev = s.get("log_evidence") or {}
        mem = {g["index"]: round(g["mem_used_mib"]) for g in s.get("gpu_after_start", [])}
        items = [f"{n}: engine={e['engine']}, tp={e['tp']}, pp={e['pp']}, 权重加载={e['model_loading_gib']} GiB" for n, e in ev.items() if n != "proxy"]
        line = f"- {ARCH_SHORT[a]}（{s['status']}）：" + "；".join(items) + f"；启动后显存 MiB {mem}"
        if a == "B" and s.get("extra", {}).get("proxy"):
            line += f"；代理分发计数 {s['extra']['proxy'].get('per_backend')}"
        if a == "D" and s.get("extra"):
            p, c = s["extra"].get("pd_proxied", {}), s["extra"].get("pd_control", {})
            line += (f"；经代理的 {s['extra'].get('n_proxied_requests')} 个请求中 KV 命中 {p.get('n_select_hit')} 次、"
                     f"decode 端重算 prefill {p.get('n_recv_bypass_false')} 次，官方 DEBUG 日志 'Successfully received all KVs' {ev.get('decode', {}).get('kv_success_count')} 次；"
                     f"负对照（绕过 prefill 直接发给 decode）：未命中 {c.get('n_select_miss')}、重算 {c.get('n_recv_bypass_false')}，"
                     f"官方 WARNING 'Failed to receive all KVs' {ev.get('decode', {}).get('kv_fail_count')} 次（应只来自负对照）")
        if s.get("agreement_with_A"):
            same = sum(x["identical"] for x in s["agreement_with_A"])
            line += f"；与 A 的贪心输出逐字一致 {same}/{len(s['agreement_with_A'])}（FP16 下不同 kernel/批次可能导致少量差异，仅作合理性参考）"
        if s.get("checks_failed"):
            line += f"；**失败检查：{s['checks_failed']}**"
        if s.get("error"):
            line += f"；**错误：{s['error'][:300]}**"
        L.append(line)
    L.append("")

    L.append("## 3. 性能结果\n")
    L.append(f"负载：闭环固定并发（{cfg['concurrency_levels']}），每个配置 {cfg['warmup_requests']} 个预热请求（不计入）+ 正式请求；"
             f"Prefill-heavy = {cfg['workloads']['prefill_heavy']['input_tokens']} 输入 / {cfg['workloads']['prefill_heavy']['output_tokens']} 输出 token，"
             f"Decode-heavy = {cfg['workloads']['decode_heavy']['input_tokens']} / {cfg['workloads']['decode_heavy']['output_tokens']}。\n")
    L.append("指标定义：TTFT=客户端发出请求到首个携带生成 token 的流式事件；ITL=相邻 token 事件的间隔（全部请求合并后取分位数，`chunk=token` 比例见表）；"
             "TPOT=单请求 (E2E−TTFT)/(输出token−1)；Goodput=同时满足 TTFT 与 TPOT SLO 的请求数/墙钟时间。\n")
    L.append("SLO（实验前在 configs/experiments.json 中预先登记的规则）：" + "；".join(f"{WL_CN[w]} TTFT≤{t['ttft_ms']} ms、TPOT≤{t['tpot_ms']} ms（{t['source']}）" for w, t in slo.items()) + "\n")
    for w in cfg["workloads"]:
        L.append(f"### {WL_CN[w]}\n")
        L.append("| 架构 | 并发 | reps | 请求 | 成功率 | TTFT P50 | TTFT P95 | ITL P50 | ITL P95 | TPOT 均值 | E2E P95 | 输出 tok/s | req/s | SLO 达标 | Goodput req/s | chunk=token |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cfg["concurrency_levels"]:
            for a in ARCHS:
                g = agg.get((a, w, c))
                if not g:
                    L.append(f"| {ARCH_SHORT[a]} | {c} | 0 | N/A |" + " N/A |" * 12)
                    continue
                L.append(f"| {ARCH_SHORT[a]} | {c} | {g['n_reps']} | {g['n_requests']} | {fmt((g['success_rate'] or 0) * 100, 0)}% | {fmt(g['ttft_p50_ms'])} | {fmt(g['ttft_p95_ms'])} | "
                         f"{fmt(g['itl_p50_ms'])} | {fmt(g['itl_p95_ms'])} | {fmt(g['tpot_mean_ms'])} | {fmt(g['e2e_p95_ms'], 0)} | {fmt(g['output_tok_per_s'])} | "
                         f"{fmt(g['req_per_s'], 2)} | {fmt((g['slo_attainment'] or 0) * 100, 0)}% | {fmt(g['goodput_req_s'], 2)} | {fmt((g['chunks_eq_tokens_frac'] or 0) * 100, 0)}% |")
        L.append("")
    if any(g["n_reps"] > 1 for g in agg.values()):
        L.append("多次重复时表中为各 rep 指标的均值；各 rep 的最小/最大值见 summary.csv。\n")

    L.append("### GPU 资源（正式计时窗口内 nvidia-smi 采样）\n")
    L.append("| 架构 | 负载 | 并发 | GPU0 显存峰值 MiB | GPU0 平均利用率 | GPU1 显存峰值 MiB | GPU1 平均利用率 |\n|---|---|---|---|---|---|---|")
    for (a, w, c), g in sorted(agg.items()):
        g0, g1 = g["gpu"].get("0", {}), g["gpu"].get("1", {})
        L.append(f"| {ARCH_SHORT[a]} | {WL_CN[w]} | {c} | {fmt(g0.get('mem_peak_mib'), 0)} | {pctf(g0.get('util_mean_pct'))} | {fmt(g1.get('mem_peak_mib'), 0)} | {pctf(g1.get('util_mean_pct'))} |")
    L.append("\n注：显存峰值主要由 `gpu_memory_utilization` 预分配决定，不代表实际 KV 使用量；利用率为 nvidia-smi 的 kernel 活跃时间比例，粒度粗。\n")

    dkeys = [(w, c) for w in cfg["workloads"] for c in cfg["concurrency_levels"] if ("D", w, c) in agg]
    if dkeys:
        L.append("### PD 分离：KV 传输与 TTFT 分解\n")
        L.append("| 负载 | 并发 | KV 命中 | decode 端重算 | 每请求 KV MB | 字节/token | 传输 P50 ms | 传输 P95 ms | 有效带宽 GB/s | 客户端 TTFT P50 | prefill 往返 P50 | decode 首包 P50 | 代理+客户端开销 P50 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for w, c in dkeys:
            g = agg[("D", w, c)]
            p, b = g.get("pd", {}), g.get("pd_breakdown", {})
            L.append(f"| {WL_CN[w]} | {c} | {p.get('n_select_hit')} | {p.get('n_recv_bypass_false')} | {fmt((p.get('kv_bytes_per_req_mean') or 0) / 1e6, 2)} | "
                     f"{fmt(p.get('kv_bytes_per_token'), 0)} | {fmt(p.get('transfer_ms_p50'), 2)} | {fmt(p.get('transfer_ms_p95'), 2)} | {fmt(p.get('effective_GBps'), 2)} | "
                     f"{fmt(b.get('ttft_p50'))} | {fmt(b.get('prefill_rt_p50'))} | {fmt(b.get('decode_first_p50'))} | {fmt(b.get('proxy_client_overhead_p50'))} |")
        L.append("\n传输时间 = consumer 侧 `SimpleConnector.select` 的阻塞时间（含信令往返、5 个张量的元数据与 NCCL 接收，探针在其后 `cuda.synchronize`），"
                 "字节数 = 收到的 K+V 张量大小（hidden states 另计）。各分量取各自中位数，相加不严格等于 TTFT 中位数。\n")

    L.append("## 4. 图表\n")
    for f in figs:
        L.append(f"![{f.stem}](figures/{f.name})\n")

    L.append("## 5. 结果解读\n")
    L.append("### 5.1 数据直接给出的结论（自动生成）\n")
    for o in observations(agg, cfg):
        L.append(f"- {o}")
    L.append("")
    L.append("### 5.2 机制解释（措辞随实测方向自动选择；样本少，均为描述性结论）\n")
    for t in mechanism_text(agg, cfg, comparison_lines(agg, cfg)):
        L.append(t + "\n")
    L.append("**哪些差异来自硬件、哪些来自架构**：T4 FP16 算力（~65 TFLOPS 峰值）与 320 GB/s 显存带宽决定了单卡 prefill/decode 的绝对速度；"
             "PCIe 互联（无 NVLink，Kaggle 上通常无 GPU P2P）决定了 PP 激活传输与 PD KV 传输的单位成本；1.5B 小模型使单卡远未饱和，削弱了 DP/PP 的意义。"
             "架构本身的影响体现在：PP 让每个 token 串行经过两张卡；DP 只增加并发容量不缩短单请求；PD 用一次 KV 传输换取 prefill 与 decode 的隔离。\n")

    L.append("## 6. 局限性\n")
    for t in ["只有 2 张 T4（Turing，无 bf16、无 FlashAttention-2，使用 V0 + XFormers），与 H100/B200 + NVLink/RDMA 的生产集群差异巨大；",
              "模型只有 1.5B 且是 GQA（2 个 KV 头），KV 体积很小，PD 传输成本被低估，PP/DP 的收益也被低估；",
              "PD 使用的是 vLLM V0 的 PyNcclConnector/SimpleConnector 参考实现：KV 以整请求为单位、prefill 完成后同步拉取，不与计算重叠，"
              "也传输了全部 token 的 hidden states；生产实现（V1 的 NIXL/P2P NCCL/LMCache，layer-wise 异步传输）开销模型不同；",
              "关闭了 chunked prefill（SimpleConnector 的要求），统一应用到所有架构；开启 chunked prefill 的单卡混合部署会显著缓解 decode 被 prefill 打断的问题，结论不能外推；",
              "闭环负载、每配置 20–40 个请求、重复次数少，P95 的统计不确定性很大，只报告描述性统计，不声称统计显著或容量边界；",
              "Kaggle 是共享虚拟化环境，PCIe 拓扑、CPU 争用、nvidia-smi 采样粒度都会引入噪声；",
              "DP2 与 PD 经过一个 Python aiohttp 代理，A/C 由客户端直连，代理开销（localhost，通常 ~1 ms 量级）计入 B/D 的 TTFT。"]:
        L.append(f"- {t}")
    L.append("")

    L.append("## 7. 完成状态\n")
    planned = [(a, w, c) for a in ARCHS for w in cfg["workloads"] for c in cfg["concurrency_levels"]]
    done = [k for k in planned if k in agg]
    L.append(f"主矩阵完成 {len(done)}/{len(planned)} 个配置。")
    missing = [k for k in planned if k not in agg]
    if missing:
        L.append("未完成（表中为 N/A，不得视为已测量）：" + "、".join(f"{ARCH_SHORT[a]}/{WL_CN[w]}/c={c}" for a, w, c in missing))
    fails = read_jsonl(RUNS_DIR / "_failures.jsonl")
    if fails:
        L.append(f"\n运行失败记录 {len(fails)} 条（results/runs/_failures.jsonl），最近一条：`{fails[-1].get('error', '')[:300]}`")
    L.append("")

    L.append("## 8. 简历描述建议（仅基于已完成并验证的内容）\n")
    ok_arch = [a for a in ARCHS if (smoke[a] or {}).get("status") == "PASS" and any(k[0] == a for k in agg)]
    if mock:
        L.append("（MOCK 数据，不生成简历描述。）")
    elif not ok_arch:
        L.append("（尚无通过验证的真实测量，暂不生成。）")
    else:
        names = "、".join({"A": "单卡混合", "B": "DP2", "C": "PP2", "D": "PD 分离"}[a] for a in ok_arch)
        L.append(f"- 在 Kaggle 2×T4 上基于 vLLM {v.get('vllm', '')} 搭建 {names} {len(ok_arch)} 种推理部署架构的可复现对比实验（Qwen2.5-1.5B，FP16），"
                 "编写流式压测客户端统一测量 TTFT/ITL/TPOT/吞吐/Goodput，实现断点续跑、超时与 GPU 资源回收。")
        if "D" in ok_arch:
            d = [agg[k] for k in agg if k[0] == "D" and agg[k].get("pd")]
            if d:
                kvmb = mean([x["pd"].get("kv_bytes_per_req_mean") for x in d])
                xf = mean([x["pd"].get("transfer_ms_p50") for x in d])
                L.append(f"- 基于 vLLM PyNcclConnector 部署真实 Prefill/Decode 分离（GPU0 prefill → NCCL 传输 KV → GPU1 decode），通过无侵入探针统计 KV 传输量与耗时"
                         f"（实测每请求平均 {fmt((kvmb or 0) / 1e6, 1)} MB、传输中位数 {fmt(xf, 1)} ms），并用绕过 prefill 的负对照验证 decode 端未重算 prefill。")
        if "C" in ok_arch:
            L.append("- 通过日志与显存证据验证 PP2 的层切分（两个 stage 各加载约一半权重），结合 TTFT/TPOT 数据分析小模型在 PCIe 互联下流水线并行的通信与气泡开销。")
        L.append("（请把上面的数字与本报告表格核对后再使用；不要写入未在表中出现的提升百分比。）")
    L.append("")
    (RESULTS / "EXPERIMENT_REPORT.md").write_text("\n".join(L), encoding="utf-8")


def write_csv(agg, path):
    keys = sorted({k for g in agg.values() for k, v in g.items() if not isinstance(v, dict)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arch", "workload", "concurrency"] + keys)
        for (a, wl, c), g in sorted(agg.items()):
            w.writerow([a, wl, c] + [g.get(k) for k in keys])


def make_zip(out_dir: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    zp = out_dir / f"pdpp_results_{ts}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(RESULTS):
            for fn in files:
                p = Path(root) / fn
                if p.suffix == ".zip":
                    continue
                if p.stat().st_size > 200 * 1024 * 1024:
                    continue
                z.write(p, Path("results") / p.relative_to(RESULTS))
        for extra in ("configs/experiments.json",):
            p = RESULTS.parent / extra
            if p.exists():
                z.write(p, extra)
    return zp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-mock", action="store_true")
    ap.add_argument("--zip-dir", default=None, help="default: parent of the lab dir (/kaggle/working)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    runs = load_runs(args.include_mock)
    slo = slo_thresholds(runs, cfg)
    for r in runs:
        add_goodput(r, slo)
    agg = aggregate(runs)
    write_csv(agg, RESULTS / "summary.csv")
    figs = figures(agg, cfg, RESULTS / "figures")
    report(cfg, runs, agg, slo, figs, args.include_mock)
    print(f"{len(runs)} runs, {len(agg)} configs -> {RESULTS / 'EXPERIMENT_REPORT.md'}, {len(figs)} figures")
    if not args.no_zip:
        zp = make_zip(Path(args.zip_dir) if args.zip_dir else RESULTS.parent.parent)
        print("ZIP:", zp)


if __name__ == "__main__":
    main()
