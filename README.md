# exphalu2 — RePPL 2.0 幻觉检测实验系统

实现 [DESIGN.md](DESIGN.md) 的 RePPL 2.0 A/B 方法与工程系统：基于推理引擎（vLLM 为主）的跨采样传播不确定性幻觉检测，附带本地 LLM judge 与概率/熵族 baseline，统一评测输出 CSV。

- **推理引擎为主 + 保留旧端口**：默认 vLLM 原生（generation runner 采样/logprob + pooling runner `token_embed` 逐 token 末层状态）；同时保留旧 exphalu 的 Transformers 前向作为 legacy 兼容端口（`backend: transformers`），用于引擎暂不支持的架构。SGLang 接口就位但未安装，状态 blocked。
- **当前验证模型**：`Qwen3.5-4B`（`/home/tim/Proj/resource` 副本）与 `Qwen2.5-0.5B/1.5B-Instruct`。`/mnt/data/Qwen3-4B` 因磁盘坏道 blocked（见"已知问题"）。

## 目录合同

| 目录 | 用途 | git |
|---|---|---|
| `reppl2/` | 唯一源码包（backends / methods / baselines / judges / data / evaluation） | 入库 |
| `config/` | 全部可配置项：模型路径、采样参数、数据路径、key 的环境变量名 | 入库 |
| `scripts/` | **唯一命令入口**，按类别分子目录：`setup/ smoke/ generation_eval/ perf/ interp/ vis/` | 入库 |
| `log/` | 所有日志，子目录与 scripts 一一对应 | 忽略 |
| `results/` | 所有产物（config_snapshot / generation / trajectory / detection / judge / eval.csv），子目录一一对应 | 忽略；`results/smoke/` 小文件入库作回归对照 |
| `vis/` | 可视化 PNG 输出 | 忽略；`vis/smoke/` 例外 |
| `tests/` | 固定数组数学验证（DESIGN M1 验收） | 入库 |

## 快速开始

```bash
# 环境自检（conda tim、GPU、模型/数据可读性）
bash scripts/setup/env_check.sh

# 数学固定数组验证
bash scripts/smoke/unit_tests.sh

# 端到端冒烟（合成数据、小 K；默认 Qwen3.5-4B + vLLM）
bash scripts/smoke/run_smoke.sh [config] [model_key] [backend]
bash scripts/smoke/run_smoke.sh config/smoke.yaml qwen2_5_0_5b vllm

# legacy Transformers 端口冒烟
bash scripts/smoke/run_smoke_transformers.sh [model_key]

# 真实数据完整评测（triviaqa，产出 eval.csv）
bash scripts/generation_eval/run_pipeline.sh [config] [model_key] [backend] [num_samples]

# 性能基准（分阶段耗时/tokens每秒/显存）
bash scripts/perf/run_perf.sh

# 可解释性导出（输入 mu/r/p_hat、B 编辑影响、逐 token NLL）
bash scripts/interp/run_interp.sh

# 可视化（需要先有完成的 run）
bash scripts/vis/plot_eval.sh <category> <run_id>
```

所有脚本自动写入 `log/<类别>/<脚本>-<时间戳>.log`；产物在 `results/<类别>/<run_id>/`。

## 方法与分数方向

- **RepPPL-A**（DESIGN §4）：末层输入-输出关联的跨采样波动 → Inner；乘法重标定 `risk=(Inner+ε)·Outer`。
- **RepPPL-B**（DESIGN §5）：对输入单元做确定性编辑（`[MASKED]`），固定输出重放，末层池化状态差 `q[k,j]` 归一化 → Inner。保存归一化 A 与原始 q 两份。
- **baseline**：outer-perplexity、LNPE、eigenscore-last（末层变体，显式命名）、semantic-entropy（NLI 蕴含，需 deberta-mnli 可读）/ semantic-entropy-lexical（词面分组回退变体）、length（对照协变量）。
- **合同**：所有分数方向统一为"越大越可能幻觉"，不做测试集方向翻转；Inner 单独输出（`reppl-a-inner` / `reppl-b-inner`）以检验 H1 增量。
- **Judge**：本地 LLM judge（self-judge），硬 Yes/No + 固定标签 likelihood（Yes/No 首 token 归一）双分数；`JudgePair` 融合接口就位。远程 provider（API LLM / Jev / StartLux）登记于 config，无 key/服务，blocked。

## 配置

- `config/base.yaml`：模型注册表、数据集路径、采样与聚合超参、judge provider 注册表。
- 覆盖：`config/{smoke,smoke_transformers,generation_eval,perf,interp}.yaml`。
- API key 一律通过环境变量注入（`API_LLM_KEY` 等，名称登记在 config），不入库。
- CLI 通用参数：`--config --category --run-id --model --dataset --num-samples --backend --strict-env`。

## 已知问题与 blocked 清单（如实记录）

1. **`/mnt/data`（/dev/sda）磁盘坏道**：`Qwen3-4B/model-00001-of-00003.safetensors` 出现 I/O error（`dd` 可复现），mmap 触发 SIGBUS。该模型本机不可用；`trivia_qa` 数据与其余已扫描模型读取正常。`Qwen3.5-4B` 使用 `/home/tim/Proj/resource` 副本。
2. **Qwen3.5-4B 引擎原生支持**：vLLM 0.15.1 注册表无 `Qwen3_5*`；Transformers 4.57.6 同样不识别 `qwen3_5` 架构（冒烟实测）。已下载 vLLM 0.30.0 wheel（携带 torch 2.13.0 + transformers 5.18.0）后台安装中；完成后运行 `bash scripts/smoke/run_smoke.sh config/smoke.yaml qwen3_5_4b vllm` 复验，并把结果补记到本节。
3. **SGLang**：环境未安装，backend 显式报 `BackendError`（无静默回退）。
4. **数据集**：SQuAD/CoQA 原始 JSON 在未挂载的外置盘，适配器已实现但 blocked（在 config 填路径即启用）；NQ 无本地文件。
5. **P0/P1 强基线**（SeSE/HAD/D-Score/RAUQ/LAFaCT/LaaB/Semantic Energy）：本轮未实现，注册表状态 `planned`（`reppl2/baselines/registry.py`），详见 DESIGN.md 附录 P3-S5。
6. **多轮工具调用轨迹**：适配器未接，`planned`。
7. **J=1 退化**：无上下文且单句的问题只有一个事实单元，跨单元 softmax 恒为 1，Inner≈0（risk 退化为 ε·Outer）。segmentation 已对单句问题做子句切分兜底，但语义单元质量有限；有 RAG 上下文的任务（squad 类）才能充分体现 A/B 的输入定位价值。
8. **小样本观察**（16 条 triviaqa，Qwen2.5-0.5B，judge 标出 2 正例）：`eigenscore-last` AUROC≈0（方向与原论文预期相反）；样本量不足以定论，正式实验前需在更大样本上复核方向约定。其余方法（reppl-a/b、outer-perplexity、lnpe、judge-continuous）在该小样本上 AUROC=1.0，仅作管线验证、不作性能结论。

## 已完成的验证记录（2026-10-04）

| 项 | 模型 | 后端 | 结果 |
|---|---|---|---|
| 固定数组数学测试（27 项） | - | - | ✅ 通过 |
| verify-backend 端到端（采样/logprob 对齐 max diff 0.0017/token_embed 末层状态） | Qwen2.5-0.5B | vLLM 0.15.1 | ✅ 通过 |
| 冒烟全链（generate→detect(A/B+5 baselines)→judge→evaluate→eval.csv） | Qwen2.5-0.5B | vLLM 0.15.1 | ✅ 通过（results/smoke/smoke-20261004-152451） |
| 冒烟全链（legacy 端口） | Qwen2.5-0.5B | Transformers | ✅ 通过（results/smoke/smoke-20261004-152901） |
| 真实数据评测（triviaqa 16 样本，judge 2 正例，AUROC 全表） | Qwen2.5-0.5B | vLLM 0.15.1 | ✅ 通过（results/generation_eval/geneval-20261004-153006） |
| perf 两阶段基准 | Qwen2.5-0.5B | vLLM 0.15.1 | ✅ 通过（results/perf/） |
| interp 解释导出（mu/r/p_hat、B 编辑、逐 token NLL） | Qwen2.5-0.5B | vLLM 0.15.1 | ✅ 通过（results/interp/） |
| vis ROC/方法对比图 | - | - | ✅ 生成（vis/generation_eval/） |
| Qwen3.5-4B | - | vLLM / Transformers | ⏳ blocked（见已知问题 2，升级后复验） |

## 里程碑对照（DESIGN §11）

- **S0 底座** ✅：types/backends/verify-backend/数学层 + 固定数组测试。
- **S1 A + 对照闭环** ✅：generate → detect（A、B、baselines）→ judge → evaluate（AUROC/AUPRC/bootstrap CI → eval.csv）。
- **S2 本地 judge** ✅；JudgePair 融合接口 ✅；远程 provider blocked。
- **S3 B + 解释** ✅（受控版：J≤4、K≤5，编辑覆盖率入 interp 产物）。
- **S4 perf / vis** ✅ 基础版。
- **S5 强基线与多轮** ⏳ planned（见上）。
