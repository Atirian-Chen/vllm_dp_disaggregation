"""Generate kaggle_pd_pp_experiment.ipynb from the files in this directory.

The notebook is self-contained: every script is embedded with %%writefile, so uploading
the single .ipynb to Kaggle is enough.   python build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = "/kaggle/working/kaggle-pd-pp-lab"
FILES = ["configs/experiments.json", "scripts/common.py", "scripts/env_check.py", "scripts/install_or_check.py",
         "scripts/prepare_workload.py", "scripts/client.py", "scripts/proxy.py", "scripts/evidence.py",
         "scripts/servers.py", "scripts/smoke_test.py", "scripts/run_experiments.py", "scripts/analyze_results.py",
         "scripts/pd_probe/sitecustomize.py", "scripts/mock_vllm_server.py"]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text, collapsed=False, raw=False):
    meta = {"jupyter": {"source_hidden": True}, "collapsed": True} if collapsed else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": (text if raw else text.strip("\n")).splitlines(keepends=True)})


md("""
# Kaggle 2×T4：单卡 / DP2 / PP2 / PD 分离 推理架构小规模对比

**按顺序执行，每一步看到 `PASS` 再继续。** 详细说明见 README.md（与本 Notebook 一起交付）。

| 步骤 | 内容 | 预计耗时 | 级别/时间上限 |
|---|---|---|---|
| 0 | 写入项目文件 | <1 min | – |
| 1–2 | 环境检查（双 T4、Internet、/dev/shm）与预算计时 | 1 min | P0 |
| 3 | 安装隔离 venv：vLLM 0.8.5.post1（依赖冻结到 2025-05-20） | 5–10 min | P0 ≤ 60 min |
| 4 | 下载 Qwen2.5-1.5B-Instruct（固定 revision）+ 生成固定 token 负载 | 2 min | P0 |
| 5 | A 单卡 smoke | 3 min | P0 |
| 6 | C PP2 smoke（验证真的是两段流水线） | 4 min | P1 ≤ 30 min |
| 7 | D PD 分离 smoke（验证真实 KV 传输 + 负对照） | 4 min | P2 ≤ 60 min |
| 8 | B DP2 smoke | 4 min | P3 ≤ 30 min |
| 9 | 正式小规模 benchmark（16 个配置） | 30–50 min | 总会话上限 6 h |
| 10 | 报告、图表、ZIP | 1 min | – |
| 11 | **关闭所有进程，然后在 Kaggle 右上角 Stop Session** | – | – |

整套首轮预计 **1–1.5 GPU 小时**。所有结果写入 `/kaggle/working/kaggle-pd-pp-lab/results/`，每个配置完成即落盘，可断点续跑。
""")

code(f"""
import os, glob, json
LAB = "{LAB}"
for d in ["scripts/pd_probe", "configs", "results"]:
    os.makedirs(os.path.join(LAB, d), exist_ok=True)
os.chdir(LAB)
VPY = "/tmp/pdlab-venv/bin/python"   # isolated vLLM venv created in step 3
print("working dir:", os.getcwd())
""")

md("""
## 0. 写入项目文件（直接运行以下所有单元格，无需修改）
这些单元格把全部脚本写到 `kaggle-pd-pp-lab/` 下。代码与 README 中的项目文件完全一致。
""")
for f in FILES:
    content = (ROOT / f).read_text(encoding="utf-8")
    code(f"%%writefile {LAB}/{f}\n" + content, collapsed=True, raw=True)

md("""
### （可选）恢复上一次会话的结果
如果之前的会话中断，可把下载的 `pdpp_results_*.zip` 作为 Kaggle Dataset 添加到本 Notebook（右侧 *Add Input*），
运行下面的单元格恢复已完成的配置，之后的步骤会自动跳过它们。
""")
code("""
import zipfile
zips = sorted(glob.glob("/kaggle/input/**/pdpp_results_*.zip", recursive=True))
if zips and not glob.glob("results/runs/*.json"):
    with zipfile.ZipFile(zips[-1]) as z:
        members = [m for m in z.namelist() if m.startswith("results/") and "/state/" not in m]
        z.extractall(LAB, members=members)
    print("restored from", zips[-1], len(members), "files")
else:
    print("nothing to restore" if not zips else "results/runs already present - not overwriting")
""")

md("""
### （可选）CPU 干跑：不消耗 GPU 额度验证 Notebook 流程
在 **Accelerator = None** 的会话里，把下面的 `RUN_MOCK_DRY_RUN` 改成 `True` 再运行：用假的 vLLM 服务器（mock）跑完
smoke → benchmark → 报告全流程，结果写到单独目录 `/kaggle/working/mock_results`，不会与真实结果混在一起。
默认 `False`，且检测到 GPU 时自动跳过，所以 GPU 会话中 “Save & Run All” 不会执行它。
""")
code("""
import shutil, subprocess
RUN_MOCK_DRY_RUN = False
has_gpu = shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
if RUN_MOCK_DRY_RUN and not has_gpu:
    MOCK_ENV = "PDLAB_RESULTS=/kaggle/working/mock_results"
    !{MOCK_ENV} python3 scripts/prepare_workload.py --mock-tokenizer --force
    !for a in A C D B; do {MOCK_ENV} python3 scripts/smoke_test.py --arch $$a --mock; done
    !{MOCK_ENV} python3 scripts/run_experiments.py --mock --archs A,D --workloads prefill_heavy --conc 2
    !{MOCK_ENV} python3 scripts/analyze_results.py --include-mock --no-zip
else:
    print("mock dry run skipped", "(GPU session)" if has_gpu else "(RUN_MOCK_DRY_RUN=False)")
""")

md("""
## 1. 环境检查（必须看到 `ENV_CHECK PASS`）
应看到两张 `Tesla T4`、compute_cap `7.5`、`internet True`。同时开始会话计时（预算保护的基准）。
若 FAIL：右侧 *Settings → Accelerator* 选 **GPU T4 x2**，*Internet* 打开，然后 *Run → Restart & clear outputs* 重来。
""")
code("!python3 scripts/env_check.py")

md("## 2. GPU 时间预算状态（任何时候都可以重新运行）")
code("!python3 -c \"import sys; sys.path.insert(0, 'scripts'); from common import Budget, load_config; print(Budget(load_config()).report())\"")

md("""
## 3. 安装隔离的 vLLM 环境（P0，约 5–10 分钟，只需一次）
在 `/tmp/pdlab-venv` 用 uv 安装 `vllm==0.8.5.post1`，所有依赖按 2025-05-20 的版本解析（不改动 Kaggle 系统 Python）。
必须看到 `INSTALL_STEP env PASS`，且输出中 `devices` 为两张 cc `[7, 5]`、`xformers` 有版本号、
`xformers_backend_import` 为 `ok`、`triton_target` 不是 `FAILED`（这两项就是 T4 上 vLLM 启动时会走的导入路径）。
如果之前的会话装过旧版 venv，本步骤会自动重新验证并修复。
失败时日志在 `results/logs/install.log`。
""")
code("!python3 scripts/install_or_check.py --step env")

md("## 4. 下载模型（固定 revision）并生成所有架构共用的 token 负载")
code("""
!python3 scripts/install_or_check.py --step model
!{VPY} scripts/prepare_workload.py
""")

md("""
## 5. A 组：单卡 smoke test（P0）
期望：`SMOKE A ... PASS`，日志行 `engine=V0 0.8.5.post1 tp=1 pp=1`，`attn=['XFormers']`，GPU0 显存约 12 GB、GPU1 接近 0。
""")
code("!{VPY} scripts/smoke_test.py --arch A")

md("""
## 6. C 组：PP2 smoke test（P1）
期望：`tp=1 pp=2`，`weights_GiB` 有 **两个** 值，每个都明显小于 A 的单卡权重（约 55–60%：一半的层 + 各自一份 tied embedding/lm_head），证明是层切分而不是 TP2 或两个副本；两张卡都占用显存。
""")
code("!{VPY} scripts/smoke_test.py --arch C")

md("""
## 7. D 组：PD 分离 + 真实 KV 传输 smoke test（P2，最关键）
期望：
- `PD transfer: hits=8/8 bypass_false=0`：每个经代理的请求，decode 实例都从 prefill 实例收到了 KV，且 **没有重算 prefill**（vLLM 官方 `bypass_model_exec=True`）；
- `bytes/token=28672`：与理论值 28 层×2×2 头×128×2B 一致；
- 负对照（绕过 prefill 直接发给 decode）出现 `n_select_miss: 1, n_recv_bypass_false: 1`，说明探针能区分“真的传了”和“重算了”；
- decode 日志里官方 DEBUG 行 `kv_ok` 计数 ≥ 8，`kv_fail` 只来自负对照。
""")
code("!{VPY} scripts/smoke_test.py --arch D")

md("## 8. B 组：DP2 smoke test（P3）\n期望：两个实例各加载一次完整权重，代理把请求分到两个副本（`per_backend` 两个都 > 0）。")
code("!{VPY} scripts/smoke_test.py --arch B")

md("""
## 9. 正式小规模 benchmark
只运行 smoke 为 PASS 的架构；顺序 A → C → D → B；每个架构启动一次、跑完 2 负载 × 2 并发后立即关闭并确认显存释放。
每个配置完成即写入 `results/runs/`，中断后重新运行本单元格会自动跳过已完成配置。
""")
code("!{VPY} scripts/run_experiments.py --archs A,C,D,B --reps 1")

md("""
### 9b.（可选，仅在时间充裕时）第二次重复
只会补跑 rep 2（rep 1 跳过），约再用 30–40 分钟。先看第 2 步的预算状态再决定。
""")
code("# !{VPY} scripts/run_experiments.py --archs A,C,D,B --reps 2")

md("## 10. 汇总：报告、图表、ZIP")
code("""
!python3 scripts/analyze_results.py
from IPython.display import Markdown, Image, display
display(Markdown(open("results/EXPERIMENT_REPORT.md", encoding="utf-8").read()))
for f in sorted(glob.glob("results/figures/*.png")):
    display(Image(f))
print(sorted(glob.glob("/kaggle/working/pdpp_results_*.zip")))
""")

md("""
## 11. 结束：关闭所有服务并释放 GPU（务必执行）
运行后确认两张卡显存都接近 0。**然后在 Kaggle 页面右上角点击 Stop Session（或 Run → Stop session）**，否则 GPU 额度会继续计时。
下载结果：右侧 *Output*（/kaggle/working）里找到 `pdpp_results_*.zip`，点击下载。
""")
code("""
!python3 scripts/servers.py stop
!nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
print("REMINDER: now click 'Stop Session' in the Kaggle UI to stop GPU quota usage.")
""")

md("""
## 附：排错工具
""")
code("""
# 最近的服务器日志（把 A/B/C/D 和实例名换成需要的）
!ls -la results/logs/
!tail -n 60 results/logs/server_D_decode.log
!tail -n 5 results/runs/_failures.jsonl 2>/dev/null
!python3 scripts/servers.py status
""")
code("""
# 随时打包当前 results（不做分析），防止会话意外结束丢数据
import shutil, time
p = shutil.make_archive(f"/kaggle/working/pdpp_partial_{time.strftime('%H%M%S')}", "zip", LAB, "results")
print(p)
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                              "language": "python", "sourceType": "notebook", "dataSources": []}},
      "nbformat": 4, "nbformat_minor": 4}
out = ROOT / "kaggle_pd_pp_experiment.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, len(cells), "cells")
