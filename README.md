# vllm_dp_disaggregation

在 Kaggle 免费的 2×Tesla T4 上，用 vLLM 0.8.5 对比四种推理部署架构：单卡 P/D 混合、DP2、PP2、Prefill/Decode 分离（PyNcclConnector 真实 KV 传输）。

完整项目、运行步骤和设计说明见 [kaggle-pd-pp-lab/README.md](kaggle-pd-pp-lab/README.md)；上传 Kaggle 只需要 [kaggle-pd-pp-lab/kaggle_pd_pp_experiment.ipynb](kaggle-pd-pp-lab/kaggle_pd_pp_experiment.ipynb)。

状态：代码与 Notebook 已完成并在本地验证（mock 全流程 + 真实 vLLM 单卡流程）；Kaggle 双 T4 实验尚未运行，仓库中暂无 T4 性能数据。
