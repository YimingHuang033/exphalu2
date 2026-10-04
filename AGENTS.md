# AGENTS.md — exphalu2 工作约定

## 硬性规则

1. **所有命令只能通过 `scripts/` 运行**，不在仓库根目录直接执行实验命令。
2. **所有日志写入 `log/<类别>/`**，脚本负责创建；类别集合：`smoke / generation_eval / perf / interp / setup / vis`。
3. **所有产物写入 `results/<类别>/<run_id>/`**：config_snapshot.json、generation.json、trajectory.json、detection.json、judge.json、eval.csv、interp 块。
4. **一切可配置项（路径/参数/key 环境变量名）写在 `config/`**，代码里不硬编码路径。
5. **git 忽略所有资源产物**（log/results/vis、模型权重、数据）；仅 `results/smoke/` 与 `tests/fixtures` 的小文件入库作回归对照。
6. **失败如实记录**：方法/数据/服务不可用时输出状态码与原因（blocked/invalid），不静默置零、不用占位结果冒充。

## 环境

- conda `tim`（Python 3.11；vLLM 0.15.1 → 0.30.0 升级中，携带 torch 2.13.0 + transformers 5.18.0；升级完成后用 `bash scripts/smoke/run_smoke.sh config/smoke.yaml qwen3_5_4b vllm` 复验 Qwen3.5-4B）。
- GPU：2× RTX 4090。模型与 `/mnt/data` 磁盘有坏道（详见 README 已知问题）；`Qwen3.5-4B` 用 `/home/tim/Proj/resource/` 副本。
- vLLM 实机约束：generate/pooling 两 runner 分时加载；引擎核心进程退出有延迟，连续任务间需等显存释放（backend 已内置重试）。

## 常用命令

```bash
bash scripts/setup/env_check.sh                    # 环境与可读性自检
bash scripts/smoke/unit_tests.sh                   # 固定数组数学测试（27 项）
bash scripts/smoke/run_smoke.sh                    # vLLM 端到端冒烟
bash scripts/smoke/run_smoke_transformers.sh       # legacy 端口冒烟
bash scripts/generation_eval/run_pipeline.sh config/generation_eval.yaml qwen2_5_0_5b vllm 16
bash scripts/perf/run_perf.sh
bash scripts/interp/run_interp.sh
bash scripts/vis/plot_eval.sh generation_eval <run_id>
```

CLI 直调（脚本内用法）：`python -m reppl2.cli <verify-backend|generate|detect|judge|evaluate|perf> --config config/x.yaml --category <类别> --run-id <id> ...`。阶段文件存在且 config hash 一致时自动跳过（断点恢复），`--force` 重跑。

## 代码约定

- 分数方向：所有方法"越大越可能幻觉"；Inner 单独输出。
- 缺失值 `null` + 状态码，不用 NaN 混入 CSV 排名。
- vLLM 两个 runner（generate / pooling）分时加载：切换前必须 `release_for_replay()` 并确认显存释放。
- 新增 baseline：在 `reppl2/baselines/registry.py` 登记 status（planned/implemented/verified/evaluated/blocked）与方向。
- 数学核心改动必须同步 `tests/test_math.py` 固定数组用例。
