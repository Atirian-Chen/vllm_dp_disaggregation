# kaggle-pd-pp-lab：Kaggle 2×T4 上的单卡 / DP2 / PP2 / PD 分离推理架构对比

在 **两张 Tesla T4（Kaggle 免费 GPU）** 上，用 **同一模型（Qwen2.5-1.5B-Instruct，FP16）、同一 token 级负载**，对比四种部署架构的 TTFT / ITL / TPOT / 吞吐 / Goodput，并对 PP2 和 PD 分离做“真的发生了”的证据检查。

| 组 | 架构 | 实现 |
|---|---|---|
| A | 单卡 P/D 混合 | 1 个 vLLM 实例，GPU0 |
| B | DP2 | 2 个独立 vLLM 实例（GPU0、GPU1 各一份完整模型）+ 轮询代理 |
| C | PP2 | 1 个 vLLM 实例，`--pipeline-parallel-size 2 --tensor-parallel-size 1`（mp backend） |
| D | PD 分离 | prefill 实例（`kv_producer`，GPU0）+ decode 实例（`kv_consumer`，GPU1），vLLM 官方 `PyNcclConnector` 传输 KV，代理串联 |

> **当前状态（诚实说明）**：代码、Notebook 和本地测试已完成；**还没有在 Kaggle 双 T4 上运行过，因此目前没有任何 T4 性能数据**。真实结果需要你按下文步骤在 Kaggle 运行后，由 `analyze_results.py` 自动生成 `EXPERIMENT_REPORT.md`。
>
> 已在本地（不占 Kaggle 额度）验证过的部分：
> - **安装链路（真实）**：WSL2 Ubuntu 中用 `install_or_check.py` 安装成功（约 100 s），解析结果为 vllm 0.8.5.post1、torch 2.6.0+cu124、xformers 0.0.29.post2、transformers 4.51.3、Python 3.11.17。
> - **单卡 A 组（真实 vLLM，RTX 4070 Laptop + Qwen2.5-0.5B，仅用于验证流程，不是实验数据）**：服务参数全部被 0.8.5 接受；日志证据解析正确（`engine=V0 0.8.5.post1 tp=1 pp=1`）；流式客户端在 4 个配置、32 个请求上 100% 成功，token 事件数 = 输出 token 数；顺序与并发两次贪心输出一致。
> - **PD 探针（真实 vLLM 类）**：在 vLLM 0.8.5 的 `SimpleConnector` 上成功包装全部 4 个方法；命中/未命中/bypass/字节数逻辑由单测覆盖。
> - **完整流程（mock）**：四组 smoke（含 PD 负对照）→ 32 个 benchmark 配置 → 断点续跑跳过 → 报告/图表/ZIP。
> - **未能在本地验证**：PP2（C）与 PD 分离（D）需要两张 GPU、跨卡 NCCL，只能在 Kaggle 上第一次真正运行，这也是 smoke 中设置了自动真实性检查和 P1/P2 时间上限的原因。
>
> 本地验证中发现并已修复的两个问题：① 预热请求数少于并发数时，第一个满批次落在计时窗口内（现在预热数 = max(4, 并发)）；② vLLM 对文本为空的 token 也会发流式事件，只统计非空 chunk 会合并 ITL 间隔（现在每个 token 事件都计时）。
>
> 第一次 Kaggle 运行（2026-10-16）发现并已修复：四组服务启动时都报 `ModuleNotFoundError: No module named 'setuptools'`。T4 上 V0 使用 XFormers 后端，会导入 triton，而 triton 3.2 运行时 `import setuptools`；vLLM 0.8.5 只在 Python ≥ 3.12 时声明这个依赖，`uv venv` 也不会预装它（Ampere 及以上的卡走 FlashAttention，不经过这条路径，所以本地 4070 测试没暴露）。现在安装时显式装 setuptools，安装校验会实际导入 XFormers 后端并初始化 triton 的 CUDA driver；旧 venv 会被自动重新校验和修复。修复已在本地用 `VLLM_ATTENTION_BACKEND=XFORMERS` 强制走 T4 同一路径、真实启动 vLLM 验证通过。同一次运行中，可选的 CPU 干跑单元格在 GPU 会话里也被执行并因 `| head` 触发 BrokenPipe，现在该单元格默认关闭且检测到 GPU 时自动跳过。

---

## 1. 关键技术决策（以及为什么）

**vLLM 0.8.5.post1 + V0 引擎 + XFormers。** 依据 vLLM 0.8.5 源码（`vllm/engine/arg_utils.py`）：
- V1 引擎要求 compute capability ≥ 8.0（“Compute Capability < 8.0” 会回退到 V0），T4 是 7.5；
- 该版本 V1 的 PP 只支持 Ray 后端，mp 后端下 PP 会回退 V0；
- V0 在 Turing 上使用 XFormers attention（随 vLLM 0.8.5 自动安装 `xformers==0.0.29.post2`）；
- V0 有官方 PD 分离示例 `examples/online_serving/disaggregated_prefill.sh`：`PyNcclConnector` + `kv_producer/kv_consumer` + 代理（先向 prefill 发 `max_tokens=1`，完成后把原请求发给 decode）。只用 NCCL + TCP，**不需要 Docker、RDMA、NIXL/UCX**，适合 Kaggle。

因此四组统一用 V0（`VLLM_USE_V1=0` 显式设置），保证可比。没有选更新的版本：V1 的 PD connector（NIXL、P2pNccl）在 T4 + 无 Docker 的环境里不确定性更大，而 0.11 之后 V0 已被删除。

**隔离 venv + 依赖冻结。** Kaggle 系统镜像的 torch/transformers 每月变化，直接往系统环境装旧版 vLLM 很容易冲突。`install_or_check.py` 用 `uv` 在 `/tmp/pdlab-venv` 建 Python 3.11 venv，`uv pip install vllm==0.8.5.post1 --exclude-newer 2025-05-20`，让所有传递依赖都按 vLLM 0.8.5 发布时的版本解析，完整清单写入 `results/env/venv_freeze.txt`。

**统一服务参数**（所有实例完全相同）：`--dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.8 --max-num-seqs 32 --max-num-batched-tokens 8192 --no-enable-chunked-prefill --no-enable-prefix-caching --seed 0`，CUDA graph 开启。关闭 chunked prefill 是 V0 `SimpleConnector` 的要求（否则 decode 端会打印 “You should set --enable_chunked_prefill=False” 并回退重算），为公平起见四组都关闭——这一点会放大单卡混合部署中“prefill 打断 decode”的现象，报告中作为局限说明。

**负载以 token ID 发送。** `prepare_workload.py` 用模型自己的 tokenizer 生成精确长度的 prompt（Prefill-heavy 1024→32，Decode-heavy 128→256），每个请求都有唯一前缀；`temperature=0, ignore_eos=true`。四组架构用完全相同的 token 序列，PD 的 lookup buffer 也按 token 序列匹配 prefill 与 decode 两半。

## 2. 真实性验证（smoke test 自动检查，不通过则该架构不进入 benchmark）

| 架构 | 自动检查 |
|---|---|
| A | 日志 `Initializing a V0 LLM engine (v0.8.5.post1)`、`pipeline_parallel_size=1`；GPU0 占显存、GPU1 空闲 |
| B | 两个实例各有一次完整权重加载（`Model loading took ~2.9 GiB`），两张卡都占显存，代理确实把请求分到两个副本 |
| C | `tensor_parallel_size=1, pipeline_parallel_size=2`；出现 **两条** `Model loading took X GiB`，每条都明显小于单卡（< 0.8×A 的权重量；每个 stage 约一半的层，外加各自一份 tied embedding / lm_head，约为单卡的 55–60%），证明是层切分而不是 TP2 或两个副本；两张卡都有显存 |
| D | 两个实例都打印 `Initializing PyNcclConfig`；**无侵入探针**（`scripts/pd_probe/sitecustomize.py`，只在 D 的进程里通过 PYTHONPATH 加载）记录每次 `SimpleConnector.select` 收到的 K/V 字节数和耗时，以及官方 `bypass_model_exec` 标志：经代理的每个请求都必须 **命中 KV 且 bypass=True（decode 端没有重算 prefill）**；再做一次 **负对照**（绕过 prefill 直接请求 decode），必须出现未命中 + 重算，证明探针能区分两种情况。decode 日志中的官方 DEBUG 行 `Successfully received all KVs and hidden states, skip model forwarding` 也被计数 |

另外每组都会和 A 的贪心输出比对（仅作合理性参考，FP16 下不同 kernel/批次允许少量差异），所有请求都必须 HTTP 200、`[DONE]`、usage 中 prompt/completion token 数等于预期、输出非空，否则记为失败而不是成功。

## 3. 指标定义

- **TTFT**：客户端发出请求 → 收到首个携带生成 token 的 SSE 事件（D 组包含代理、prefill 排队与计算、KV 传输、decode 首 token）。
- **ITL**：相邻 token 事件的间隔（vLLM 对每个生成 token 都发一个事件，文本为空的 token 也计入），全部请求合并后取 P50/P95/P99；同时报告“事件数 = 输出 token 数”的请求比例，以确认 ITL 是逐 token 的。
- **TPOT**：单请求 (E2E − TTFT)/(输出 token − 1)，报告均值与分位数（与 ITL 分开）。
- **吞吐**：输出 token/s、请求/s（墙钟时间 = 首个请求发出到最后一个完成）。
- **Goodput / SLO 达标率**：规则在实验前写入 `configs/experiments.json`：TTFT ≤ 3×A(c=2) TTFT P50，TPOT ≤ 1.25×A(c=2) TPOT P50（分负载）。
- **系统**：nvidia-smi 每 0.5 s 采样两卡显存峰值与利用率；服务启动耗时单独记录；D 组的 KV 字节数、传输时间、TTFT 分解（prefill 往返 / decode 首包 / 代理开销）。

## 4. 文件结构

```text
kaggle-pd-pp-lab/
├── kaggle_pd_pp_experiment.ipynb   # 自包含 Notebook（内嵌全部脚本，上传这一个文件即可）
├── build_notebook.py               # 从 scripts/ 重新生成 Notebook
├── README.md / requirements.txt
├── configs/experiments.json        # 模型、版本、统一参数、负载、预算、SLO 规则
├── scripts/
│   ├── common.py                   # 路径、原子写、nvidia-smi 采样、预算保护
│   ├── env_check.py                # 双 T4 / Internet / 磁盘 / /dev/shm 检查，启动会话计时
│   ├── install_or_check.py         # uv 隔离 venv + 固定 revision 模型下载
│   ├── prepare_workload.py         # 固定 token 负载
│   ├── servers.py                  # A/B/C/D 进程编排、健康检查、超时、进程组清理、显存释放确认
│   ├── proxy.py                    # DP 轮询代理 / PD 代理（aiohttp）
│   ├── client.py                   # SSE 流式客户端、闭环负载、指标
│   ├── evidence.py                 # 日志/探针/代理证据解析
│   ├── smoke_test.py               # 每个架构的 smoke + 真实性检查
│   ├── run_experiments.py          # 可续跑 benchmark
│   ├── analyze_results.py          # summary.csv、图表、EXPERIMENT_REPORT.md、ZIP
│   ├── pd_probe/sitecustomize.py   # PD KV 传输探针（只读包装，不改变行为）
│   └── mock_vllm_server.py         # 仅本地/CPU 干跑用的假服务器
└── tests/test_local.py             # 本地单测（探针 import hook、指标）
```

## 5. Kaggle 操作步骤（第一次使用也能照做）

**准备（不耗 GPU）**
1. 登录 kaggle.com。账号需要完成 **手机验证**，否则无法打开 Internet 和 GPU（*Settings → Phone verification*）。
2. 左侧 *Create → New Notebook*。
3. 菜单 *File → Import Notebook*，选择本目录的 `kaggle_pd_pp_experiment.ipynb` 上传。
4. 右侧边栏 *Session options*（或 *Settings*）：
   - **Accelerator** 选 **GPU T4 x2**（不要选 P100 或单卡 T4）。选择后会话开始计 GPU 时间。
   - **Internet** 设为 **On**（下载 vLLM 和模型需要）。
   - Persistence 不需要。
5. （可选、推荐）先在 Accelerator = None 的会话里跑 Notebook 的 “CPU 干跑” 单元格，确认流程无误，再切换到 GPU T4 x2。

**运行（GPU 会话）** —— 从上往下依次运行，每步看到 PASS 再继续：

| 单元格 | 正常输出 | 不正常时 |
|---|---|---|
| 第一个代码单元 + “0. 写入项目文件”下的所有 `%%writefile` | 每个显示 `Writing ...` | 重新运行 |
| 1 环境检查 | 两行 `Tesla T4`、`compute_cap 7.5`、`internet True`、`ENV_CHECK PASS` | 检查 Accelerator / Internet 设置后 *Restart session* |
| 3 安装 | `INSTALL_STEP env PASS`；JSON 里 `"vllm": "0.8.5.post1"`、两个 `"cc": [7, 5]`、`xformers` 有版本 | 看 `results/logs/install.log`；P0 上限 60 分钟 |
| 4 模型 + 负载 | `INSTALL_STEP model PASS`；`input lengths [1024]` / `[128]` | 网络问题重试一次；仍失败见 §7 |
| 5 A smoke | `SMOKE A ... PASS`，`engine=V0 0.8.5.post1 tp=1 pp=1 attn=['XFormers']` | 见 §7 |
| 6 C smoke | `SMOKE C ... PASS`，`tp=1 pp=2`，`weights_GiB` 两个值、各约为 A 的 55–60% | 见 §7；P1 上限 30 分钟 |
| 7 D smoke | `SMOKE D ... PASS`，`hits=8/8 bypass_false=0 bytes/token=28672`，负对照 `n_select_miss: 1` | 见 §7；P2 上限 60 分钟 |
| 8 B smoke | `SMOKE B ... PASS`，`per_backend` 两个副本都有请求 | 见 §7 |
| 9 benchmark | 每个配置一行 `ok 40/40 TTFT ... ITL ... tok/s`，D 组额外显示 `KV hits ... bypass_false 0` | 中断后直接重新运行本单元格，已完成的会跳过 |
| 10 汇总 | 显示报告与 6 张图，打印 ZIP 路径 | – |
| 11 结束 | 两张卡 `memory.used` 接近 0 | 再运行一次；仍不行就 *Stop Session* |

**怎样确认真实 KV Transfer（第 7 步）**：看 `PD transfer:` 一行——`hits` 等于经代理的请求数、`bypass_false=0`（decode 没有重算 prefill）、`bytes/token=28672`（=28 层×2(K,V)×2 KV 头×128 维×2 字节）、有非零传输时间；负对照出现 miss 与重算。原始证据在 `results/logs/pd_probe/consumer.jsonl`、`results/logs/server_D_decode.log`（搜索 `Successfully received all KVs`）和 `results/smoke/D.json`。

**查看与下载结果**：报告 `results/EXPERIMENT_REPORT.md` 会直接显示在第 10 步输出里；右侧 *Output* 面板（`/kaggle/working`）中下载 `pdpp_results_<时间>.zip`（包含 runs、smoke、日志、图表、报告、环境清单）。会话结束后 `/kaggle/working` 不保留，**一定要先下载 ZIP**（或用 *Save Version* 保存）。

**结束 GPU 占用**：运行第 11 步 → 确认显存为 0 → 页面右上角 **Stop Session**（或 *Run → Stop session*）。Kaggle 按会话时长计费，不是按实际 GPU 计算时间；进程关了但会话不停，额度仍在消耗。

## 6. GPU 时间预算

| 阶段 | 预计 |
|---|---|
| 环境检查 + venv 安装 + 模型下载 | 10–15 min |
| 4 个 smoke（每个含一次 vLLM 启动与 CUDA graph 捕获） | 12–20 min |
| 16 配置 benchmark（每个架构启动一次） | 30–50 min |
| 汇总与清理 | 2 min |
| **首轮合计** | **约 1–1.5 h** |
| 可选第二次重复 | +30–40 min |

内置保护：`results/state/budget.json` 记录会话起点；总上限 6 h（超过后 runner 不再启动新配置）；P0/P1/P2/P3 分别 60/30/60/30 分钟的调试上限（超过后 smoke 拒绝再试，需显式 `--force`）；服务启动超时 15 分钟、单请求 180 s、单配置 15 分钟；每个架构跑完立即关闭并确认显存释放。

## 7. 分级止损与常见问题

- **P0（环境 + 单卡）失败**：先看 `install.log` / `server_A_gpu0.log` 末尾。
  - uv 拉不到 Python 3.11 → 脚本自动退回系统 Python 建 venv。
  - pip 解析失败 → 备选版本：`python3 scripts/install_or_check.py --step env --vllm-version 0.9.2 --exclude-newer 2025-07-15T00:00:00Z --venv /tmp/pdlab-venv-092`（0.9.2 仍保留 V0 与 PyNcclConnector，但未经本项目验证）。改用备选版本时，在 Notebook 里执行 `os.environ["PDLAB_VENV"] = "/tmp/pdlab-venv-092"; VPY = "/tmp/pdlab-venv-092/bin/python"`，并且 **四组都要用这个版本重跑**，不能混用两个版本的结果。
  - CUDA graph 捕获报错 → 在 `configs/experiments.json` 把 `enforce_eager` 改成 `true`（四组统一，报告会注明 CUDA graph=OFF）。
  - 1.5B 显存不足（理论上不会）→ `--model Qwen/Qwen2.5-0.5B-Instruct`，四组统一换模型。
- **P1（PP2）失败**：典型是 NCCL 通信。先在 Notebook 里 `import os; os.environ["PDLAB_NCCL_SAFE"]="1"` 再重跑第 6 步（设置 `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`，PP 与 PD 都会使用）。30 分钟内仍失败：保留 `results/smoke/C.json` 和日志，继续 D。
- **P2（PD）失败**：
  - 两个实例卡在启动（互相等 NCCL rendezvous）→ 检查 14579/14580 端口、同上 `PDLAB_NCCL_SAFE=1`；
  - `hits < n` 或 `bypass_false > 0` → decode 端在重算 prefill，**这不算 PD 分离**，smoke 会判 FAIL，D 不进入 benchmark；
  - 60 分钟内仍失败：保留错误证据，A/B/C 照常完成，报告中 D 为 N/A。不要用重算 prefill 或分开计时相加来代替。
- **P3（DP2）**：两个独立实例，失败概率最低；失败则报告中 B 为 N/A。
- **会话意外中断**：重新打开 Notebook，跑第一个代码单元 + 所有 `%%writefile`，若上次 ZIP 已作为 Dataset 添加，运行“恢复”单元格；第 3 步若 venv 丢失需要重装（/tmp 不跨会话）；benchmark 会跳过已完成配置。
- **残留进程 / 端口占用**：`!python3 scripts/servers.py stop`。

## 8. 本地验证（不占 Kaggle GPU）

```bash
python tests/test_local.py
```

验证 PD 探针的 import hook（命中/未命中/bypass 标志/字节数）与指标计算。mock 全流程（假 vLLM 服务器 → 代理 → 客户端 → smoke 检查 → 续跑 benchmark → 报告/图表/ZIP）：

```bash
PDLAB_RESULTS=/tmp/mockres python scripts/prepare_workload.py --mock-tokenizer
```

```bash
PDLAB_RESULTS=/tmp/mockres python scripts/smoke_test.py --arch D --mock
```

```bash
PDLAB_RESULTS=/tmp/mockres python scripts/run_experiments.py --mock
```

```bash
PDLAB_RESULTS=/tmp/mockres python scripts/analyze_results.py --include-mock
```

mock 数据只用于测试流程，报告顶部会强制标注 MOCK，也不会生成简历描述。

## 9. 局限

2×T4（Turing、PCIe、无 NVLink，Kaggle 上通常无 P2P）、1.5B GQA 小模型、V0 参考版 PD connector（整请求同步拉取 KV，不与计算重叠，还传输 hidden states）、关闭 chunked prefill、闭环低并发、每配置 32–40 个请求、重复次数少。结论只适用于这个小规模环境，不能外推到 H100/B200 生产集群，也不能声称统计显著或容量边界。
