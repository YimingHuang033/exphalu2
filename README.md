# exphalu2 — RePPL 2.0 幻觉检测实验系统

实现 [DESIGN.md](DESIGN.md) 的 RePPL 2.0 A/B 方法与工程系统：基于推理引擎（vLLM 为主）的跨采样传播不确定性幻觉检测，附带本地 LLM judge 与概率/熵族 baseline，统一评测输出 CSV。

- **推理引擎为主 + 保留旧端口**：默认 vLLM 原生（generation runner 采样/logprob + pooling runner `token_embed` 逐 token 末层状态）；同时保留旧 exphalu 的 Transformers 前向作为 legacy 兼容端口（`backend: transformers`），用于引擎暂不支持的架构。SGLang 接口就位但未安装，状态 blocked。
- **当前验证模型**：`Qwen3.5-4B`（`/home/tim/Proj/resource` 副本）、`Qwen2.5-0.5B/1.5B/7B-Instruct` 与 `Qwen3-1.7B/8B`（`/mnt/data`，可读性已复验）。`/mnt/data/Qwen3-4B` 因磁盘坏道 blocked（见"已知问题"）。Judge：`gpt-oss-20b`（`/mnt/data`）。

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
bash scripts/generation_eval/run_pipeline.sh [config] [model_key] [backend] [num_samples] [dataset] [judge_provider]

# 批量评测矩阵（模型 × 数据集，顺序执行，产 BATCH-<ts>-summary.csv）
bash scripts/generation_eval/run_batch.sh [config] [backend] [num_samples] [models_csv] [datasets_csv] [judge_provider]

# judge provider 验收（固定 4 用例；默认 config 中的 provider）
bash scripts/smoke/verify_judge.sh [config] [backend] [judge_provider]

# 性能基准（分阶段耗时/tokens每秒/显存）
bash scripts/perf/run_perf.sh

# 可解释性导出（输入 mu/r/p_hat、B 编辑影响、逐 token NLL）
bash scripts/interp/run_interp.sh

# 可视化（需要先有完成的 run）
bash scripts/vis/plot_eval.sh <category> <run_id>
```

所有脚本自动写入 `log/<类别>/<脚本>-<时间戳>.log`；产物在 `results/<类别>/<run_id>/`。

### 本地 CPU 回归与缓存兼容性（2026-10-06）

没有服务器 conda 环境时，可使用独立测试环境（不安装 vLLM、不下载模型）：

```bash
bash scripts/setup/test_env.sh /path/to/python3
# 后续复验无需重新安装；TEST_PYTHON 建议使用绝对路径
TEST_PYTHON="$PWD/.venv/bin/python" bash scripts/smoke/unit_tests.sh
```

测试依赖在 `config/requirements-test.txt`；环境位于被 git 忽略的 `.venv/`。
CPU 测试覆盖数学、适配器、缓存依赖、JSON 写入、备用标签和流水线阶段顺序，
不能替代服务器上的 GPU 端到端冒烟与 judge 验收。

本次修复会影响结果兼容性，已有实验需使用新 run_id 重新运行：

- 缓存 hash 纳入 CLI 覆盖参数及缓存版本；各阶段必须使用相同配置和覆盖参数。
  配置不符的上游文件会明确报错。重跑 generation 会清除 detection/judge/evaluation
  等下游缓存，重跑 detection 或 judge 会清除旧评估与 CSV。
- `outer-perplexity` 使用贪心答案有效 token 数归一化平均 NLL，
  不再误用采样答案平均长度；RePPL 的 Outer 保留其采样长度归一化定义。
- `em_gold` 改为保守的归一化精确匹配，支持独立行 `Answer:` / `Final answer:`，
  保留数值正负号、分数和小数差异；`5` 不再匹配 `56`，缺失 gold 不再标成错误。
  它仍是启发式备用标签，不具备语义等价或数学等价判定能力。
- JSON 中非有限数写为 `null`，临时文件完整落盘后原子替换目标文件；
  无有效分数或只有单类标签的评估记录为 `invalid` 并附原因。
- StartLux 服务在 detect 完成后启动；若已有服务占用 GPU，流水线明确退出，
  需先停止该服务。StartLux 共用双 GPU 时应使用顺序批处理，不能与其他 GPU 作业并行。

## 方法与分数方向

- **RepPPL-A**（DESIGN §4）：末层输入-输出关联的跨采样波动 → Inner；乘法重标定 `risk=(Inner+ε)·Outer`。输入单元表示 u[j] 池化自 greedy 重放的 **prompt 侧**末层状态（2026-10-05 修复：后端现同时返回 `hidden_context`，且 unit char span 会先翻译到渲染后 prompt 坐标再做 token 对齐，span 翻译策略入 interp 记录）。
- **RepPPL-B**（DESIGN §5）：对输入单元做确定性编辑（`[MASKED]`），固定输出重放，末层池化状态差 `q[k,j]` 归一化 → Inner。保存归一化 A 与原始 q 两份。
- **baseline**：outer-perplexity、LNPE、eigenscore-last（末层变体，显式命名）、semantic-entropy（NLI 蕴含，需 deberta-mnli 可读）/ semantic-entropy-lexical（词面分组回退变体）、length（对照协变量）、**d-score-last**（D-Score 谱统计末层变体：σ₁/σᵢ≤τ 计数，τ=10 入 config）、**sese**（官方 SELGroup/SeSE @8d4c6c5 逐位移植：NLI 蕴含 0.65 + 句向量余弦 0.35 混合相似度聚类建图 → 编码树结构熵；官方 GPT-4o 答案增强因无 key 跳过并显式记录）。
- **合同**：所有分数方向统一为"越大越可能幻觉"，不做测试集方向翻转；Inner 单独输出（`reppl-a-inner` / `reppl-b-inner`）以检验 H1 增量。
- **Judge**：本地 LLM judge，**参考答案感知**（prompt v2 `judge-yesno-v2`：gold answers 作为 Reference 注入判题，按 ground-truth 定幻觉标签），硬 Yes/No + 固定标签 likelihood（Yes/No 首 token 归一）双分数。默认 provider `gpt_oss_20b`（外置 `gpt-oss-20b` mxfp4，经 vLLM 分时加载；harmony 格式直接注入 final channel，判词与标签 likelihood 同位置测得）；`local_self`（被测模型自判）保留可选。`JudgePair` 融合接口就位。远程 provider（API LLM / Jev / StartLux）登记于 config，无 key/服务，blocked。
- **数据集**（ground-truth 幻觉评测，全部实现并实测）：基础层 triviaqa、gsm8k、math500、competition_math、gpqa_diamond、mmlu_college_{chemistry,computer_science,mathematics}（+ synthetic 冒烟）；**困难层**（诱导被测模型幻觉，2026-10-06）：mmlu_pro（10 选项推理 MCQ）、hle_text（HLE 纯文本 exactMatch 子集，图像题剔除）、competition_math_level5（Level-5 竞赛数学）。数据集可带 `sampling` 覆盖（math/困难层 192 tokens）。hle 完整集（多模态）、livebench（非事实问答）显式 blocked。

## 配置

- `config/base.yaml`：模型注册表、数据集路径、采样与聚合超参、judge provider 注册表。
- 覆盖：`config/{smoke,smoke_transformers,generation_eval,perf,interp}.yaml`。
- API key 一律通过环境变量注入（`API_LLM_KEY` 等，名称登记在 config），不入库。
- CLI 通用参数：`--config --category --run-id --model --dataset --num-samples --backend --strict-env`。

## 已知问题与 blocked 清单（如实记录）

1. **`/mnt/data`（/dev/sda）磁盘坏道**：`Qwen3-4B/model-00001-of-00003.safetensors` 出现 I/O error（2026-10-06 复验 `dd` 全读仍可复现），该模型本机不可用；`gpt-oss-20b` 全部 shard 复读无 I/O error，`Qwen2.5-0.5B/1.5B/7B`、`Qwen3-1.7B/8B` 可读（7B/8B 多偏移抽读）。`Qwen3.5-4B` 使用 `/home/tim/Proj/resource` 副本。
2. **huggingface.co 本网络不可达**：模型/数据下载需走 `HF_ENDPOINT=https://hf-mirror.com`（SeSE 句向量模型已按此下载到 `/home/tim/Proj/resource/sent-emb-static-similarity-mrl`）；GitHub/arXiv 可直连。
3. **SGLang**：环境未安装，backend 显式报 `BackendError`（无静默回退）。
4. **数据集**：SQuAD/CoQA 原始 JSON 在未挂载的外置盘，适配器已实现但 blocked（在 config 填路径即启用）；NQ 无本地文件。
5. **P0/P1 强基线剩余项**（HAD/RAUQ/LAFaCT/LaaB/Semantic Energy；D-Score 原版最优层配置）：未实现，注册表状态 `planned`（`reppl2/baselines/registry.py`）。已实现：`d-score-last`（末层适配变体，显式命名）与 `sese`（官方移植，GPT-4o 增强跳过已记录），均带固定数组测试。
6. **多轮工具调用轨迹**：适配器未接，`planned`。
7. **J=1 退化**：单句无上下文的问题只有一个事实单元，跨单元 softmax 恒为 1，Inner≈0（risk 退化为 ε·Outer）。triviaqa 16 样本全部属于此情形（reppl-a/b 的 inner 无信息，其 AUROC 由 ε·Outer 驱动）；segmentation 已对单句问题做子句切分兜底，但 triviaqa 问题极少含子句标点。有 RAG 上下文的任务（squad 类）才能充分体现 A/B 的输入定位价值——数据挂载是下一步优先项。
8. **小样本观察**（16 条 triviaqa，Qwen3.5-4B，judge 标出 4 正例，2026-10-05）：`d-score-last` AUROC≈0.16（末层变体方向与论文报告不一致的早期信号，需更大样本复核）；`sese` AUROC=0.75 / AUPRC=0.68（非 judge 方法中 AUPRC 最高）；reppl-a/reppl-b/outer-perplexity AUROC=0.8125。样本量仅作管线验证、不作性能结论。
9. **vLLM 0.30.0 适配记录**：默认 flashinfer 采样 kernel 需要 nvcc JIT 编译，本机无 CUDA toolkit → 引擎核心进程在采样时崩溃。已在 `backend_cfg.env.VLLM_USE_FLASHINFER_SAMPLER: "0"` 固定禁用（回退原生 torch 采样）。另 0.30.0 要求 pooling task 在建引擎时经 `PoolerConfig(task="token_embed")` 声明（运行时切换被拒），后端已适配。
10. **gpt-oss-20b judge**：vLLM 0.30.0 + RTX 4090（SM89）加载 mxfp4 权重正常（权重+非 torch 约 14.7 GiB，KV cache 3.8 GiB @ util 0.80）；judge 阶段与被测模型分时占用 GPU（阶段间 backend 已 close）。harmony 模板经 `apply_chat_template` 后直接追加 `<|channel|>final<|message|>` 进入 final channel，硬判词与 Yes/No likelihood 同位置测得（2026-10-06 verify-judge 4/4 通过，正负例 likelihood 分离 >5 个数量级）。
11. **GPQA-Diamond 与小模型**：Qwen3-1.7B 在 16 条 GPQA-Diamond 上 0/16 正确（judge 依据 gold 判定），全正例标签使 AUROC 退化（无负例）；GPQA 适合较大被测模型或仅作 AUPRC/校准观察。原始数据中 1/198 条含乱序 remap 块，适配器已做剥离与 gold 字母映射（tests 覆盖）。
12. **历史缺陷修复（2026-10-06）**：`load_entailment` 在 sese 与 semantic-energy 共享 NLI 实例分支引用了未绑定的局部 `torch`（"cannot access free variable"），detect 全样本 failed；已修复（函数级 import）。
13. **StartLux-Decision-27B judge**：`/v1/systemone` 适配器（`reppl2/judges/systemone.py`，choice 二选项映射 → 统一 schema，choice 概率 → 连续分）与服务编排脚本（`scripts/setup/startlux_service.sh`，llama.cpp + 官方 startlux_decision.gguf_server）已实现并带单测（fake 服务）。**状态 `blocked:llama.cpp-binary-download-unreachable`**：本机需 llama.cpp ≥ b10454 CUDA 二进制，GitHub release CDN 不可达、镜像代理反复中断（llama-b11433 停在 42/153MB），按约定暂缓。恢复路径：下载 `llama-b11433-bin-ubuntu-cuda-13.4-x64.tar.gz` + `cudart-*.tar.gz` 解压至 `/home/tim/Proj/resource/llama.cpp/`（含 `llama-server`），`bash scripts/setup/startlux_service.sh start`，跑 `verify_judge.sh config/generation_eval.yaml vllm startlux_local`，通过后把 provider status 改回 implemented。

## 已完成的验证记录（2026-10-04 / 2026-10-05）

| 项 | 模型 | 后端 | 结果 |
|---|---|---|---|
| 固定数组数学测试（40 项，含 d-score/sese/对齐回归） | - | - | ✅ 通过 |
| verify-backend 端到端（vLLM 0.30.0：logprob 对齐 max diff 0.0000 / token_embed 末层状态 (T,2560)） | Qwen3.5-4B | vLLM 0.30.0 | ✅ 通过（results/smoke/smoke-20261005-193928） |
| 冒烟全链（generate→detect(A/B+5 baseline)→judge→evaluate） | Qwen3.5-4B | vLLM 0.30.0 | ✅ 通过（results/smoke/smoke-20261005-193928） |
| 冒烟全链（legacy 端口，transformers 5.18.0 识别 qwen3_5） | Qwen3.5-4B | Transformers 5.18.0 | ✅ 通过（results/smoke/smoke-20261005-194323） |
| 冒烟回归（0.5B 在升级后 vLLM） | Qwen2.5-0.5B | vLLM 0.30.0 | ✅ 通过（results/smoke/smoke-20261005-194502） |
| 真实数据评测（triviaqa 16 样本，judge 4 正例；含 d-score-last 与 sese） | Qwen3.5-4B | vLLM 0.30.0 | ✅ 通过（results/generation_eval/geneval-20261005-203720） |
| SeSE 移植对照（官方实现 vs 本仓库移植，10 个种子图 + 边界用例） | - | CPU | ✅ 逐位一致（tests/test_p0_baselines.py） |
| perf 两阶段基准 / interp 解释导出 / vis 图 | Qwen2.5-0.5B | vLLM 0.15.1 | ✅ 通过（2026-10-04，results/perf、results/interp、vis/） |
| 冒烟回归（对齐修复后双后端复验） | Qwen3.5-4B | vLLM / Transformers | ✅ 通过（smoke-20261005-205015 / -205335） |
| 固定数组数学测试（73 项，含 adapters/judge v2 用例） | - | - | ✅ 通过（2026-10-06） |
| verify-judge（固定 4 用例：对/错事实 + 对/错数学，gold 参考） | gpt-oss-20b | vLLM 0.30.0 | ✅ 4/4 通过（results/smoke/verifyjudge-20261006-131624） |
| gsm8k 16 样本全链（12 方法 + judge 9 正例；judge-continuous AUROC=1.0） | Qwen2.5-1.5B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-132227） |
| math500 16 样本全链（judge 11 正例） | Qwen2.5-1.5B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-132628） |
| competition_math 16 样本全链（judge 8 正例） | Qwen2.5-1.5B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-135039） |
| mmlu_college_computer_science 16 样本全链（judge 8 正例） | Qwen2.5-1.5B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-134747） |
| mmlu_college_chemistry 16 样本全链（judge 9 正例） | Qwen2.5-1.5B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-135731） |
| gpqa_diamond 16 样本全链（judge 16 正例，见已知问题 11） | Qwen3-1.7B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-133935） |
| **大规模困难层评测（双卡并行队列，run_batch_parallel.sh，2026-10-06）** | | | |
| mmlu_pro 500 样本（judge 80.6% 正例；length AUROC=0.679） | Qwen2.5-1.5B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-155839-g0） |
| hle_text 500 样本（judge 98.4% 正例；2 条超长问题超 4096 窗口如实 failed；semantic-energy AUROC=0.649/AUPRC=0.989） | Qwen3-1.7B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-155836-g1） |
| competition_math_level5 200 样本（judge 89.5% 正例；lnpe AUROC=0.785） | Qwen2.5-1.5B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-172440-g0） |
| mmlu_college_mathematics 16 样本全链（judge 14 正例） | Qwen3-1.7B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-134438） |
| triviaqa 16 样本回归（参考感知 judge v2） | Qwen3-1.7B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-135441） |
| 冒烟全链回归（gpt-oss-20b judge provider 接入） | Qwen3.5-4B | vLLM 0.30.0 | ✅ 通过（smoke-20261006-140032） |
| mmlu_pro 16 样本全链（困难层；judge 10 正例；reppl-b-inner AUROC=0.75） | Qwen2.5-1.5B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-153826） |
| hle_text 16 样本全链（困难层；judge 15 正例；sese AUROC=0.80） | Qwen3-1.7B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-154140） |
| competition_math_level5 16 样本全链（困难层；judge 15 正例；reppl-b AUROC=0.867） | Qwen2.5-1.5B | vLLM 0.30.0 | ✅ 通过（geneval-20261006-154437） |

## 里程碑对照（DESIGN §11）

- **S0 底座** ✅：types/backends/verify-backend/数学层 + 固定数组测试。
- **S1 A + 对照闭环** ✅：generate → detect（A、B、baselines）→ judge → evaluate（AUROC/AUPRC/bootstrap CI → eval.csv）。
- **S2 本地 judge** ✅：外置 `gpt-oss-20b` judge（参考感知 prompt v2）+ `local_self` 自判保留；JudgePair 融合接口 ✅；`startlux_local`（SystemOne 协议）适配器实现但服务 blocked（llama.cpp 二进制下载不可达，见已知问题 13）；远程 provider blocked。
- **S3 B + 解释** ✅（受控版：J≤4、K≤5，编辑覆盖率入 interp 产物）。
- **S4 perf / vis** ✅ 基础版。
- **S5 强基线与多轮** ⏳ 进行中：`d-score-last`（适配变体）与 `sese`（官方移植）已实现并通过真实数据验收；HAD/RAUQ/LAFaCT/LaaB/Semantic Energy/D-Score 原版 与多轮适配器仍为 `planned`。
