# TODO：RePPL C–E 的 SuperGPQA / PopQA 实验计划

2026-10-10。基础云端代码：`01ad1ae`。本轮只验证当前两个数据集，不扩大到多轮工具调用。

**目标：证明输入语义传播的不确定性 Inner 本身有用，并在控制 Outer 后仍提供增量。** 不能仅凭融合接近 Outer、筛掉难样本、使用答案 judge，或在测试集调符号/权重声称改进。

算法、状态与工程边界见 [CDE_IMPLEMENTATION.md](docs/CDE_IMPLEMENTATION.md)。

## 已完成的编码

- [x] C：固定答案对照方向，有符号输入贡献的跨采样波动；随机方向对照。
- [x] D：固定答案、语义单元对齐的保义视图，传播响应向量方差；范数方差与输出方差对照。
- [x] E：仅传播波动的非负校准，d/cd 模式，嵌套分组 OOF；保存模型与逐项贡献。
- [x] 原生 vLLM 末层单 token 观察，不用 Transformers 模型前向；generate/pooling 分阶段。
- [x] 冻结原 generation，源文件只读；逐样本缓存、内容 hash、null 状态、成本与覆盖。
- [x] Inner/Outer/原乘法/gentle 分开输出；共有样本配对差值与分组 bootstrap。
- [x] CPU 数学与流程合同测试；这不代表 GPU smoke 或 AUROC 已通过。

## P0：云端 GPU 和数据合同验收

- [ ] 找到截图对应的两个完整 run；记录绝对路径、dataset/generation/config_snapshot/judge 文件及 hash。不得重采 greedy 来替换错误答案。
- [ ] 新建 `config/cde_supergpqa_smoke.yaml`、`config/cde_popqa_smoke.yaml`，复制 `config/cde.yaml` 后设 `limit: 8`。核对 model key/path 与冻结 generation 一致。现有配置通过 base.yaml 合并，不支持任意多级 include。
- [ ] 显式设置标签来源：主任务用原数据 gold 的 `em_gold`；要对齐截图 judge 标签，则单独预注册 `judge` run。不能将两种标签分数直接比较。SuperGPQA 保持 Final answer 格式要求，截断输出排除。
- [ ] 固定模型权重 revision、tokenizer 版本、vLLM 版本与 dtype，写入云端实验记录。代码检查模型 key/path，但路径相同不能保证权重未被替换。
- [ ] 跑 prepare；人工盲看题干/单元/改写（不要看 detector score 或 gold 来挑改写）。检查否定、量词、公式、关系、数值、实体、选项引用没有变化；检查相同 unit_id 的中和含义一致。
- [ ] 跑 detect；确认实际载入 native vLLM，并返回完整后缀最后 token 的非零有限状态；不允许 Transformers fallback。这里需真实 GPU 验收，CPU mock 不能替代。
- [ ] 确认 generate/pooling 分时释放显存、模型没有重复常驻；记录显存峰值、准备/读出耗时和 token 数。
- [ ] 重跑 detect 验证断点跳过；修改准备文件则应拒绝旧结果；`--force` 后重算。源 generation hash 必须不变。
- [ ] 若后端不支持模型或末层提取失败，记录 blocked 并停止扩量。当前 SGLang stub 不能列为成功实现。

从仓库根目录通过脚本运行，填入实际云端路径；环境变量名在 config/cde.yaml 中定义：

```bash
export CDE_SOURCE_RUN=/absolute/path/to/original/cloud/run
export CDE_RUN_ID=cde-popqa-smoke-v1
bash scripts/generation_eval/run_cde.sh prepare config/cde_popqa_smoke.yaml
bash scripts/generation_eval/run_cde.sh detect config/cde_popqa_smoke.yaml
bash scripts/generation_eval/run_cde.sh evaluate config/cde_popqa_smoke.yaml
```

默认脚本激活 conda tim。另一个 Python 环境可设置 `CDE_PYTHON=/absolute/path/to/python`，确保 numpy/scipy/sklearn/YAML 和目标 vLLM 环境已安装。CPU 回归入口：

```bash
TEST_PYTHON=/absolute/path/to/python bash scripts/smoke/unit_tests.sh
```

8 条 smoke **不训练 E**，分组与标签类别通常不足。错误会保留原因，不以填零方式凑够样本。

## P1：各 200 条探索，先判断 Inner

- [ ] 两个数据集各取固定顺序 200 条；新配置、新 run_id，冻结 J≤4、M=4、min_views=2、原 K、tau/信号阈值/模板/中和词。C 的无效采样会排除但保留计数，至少 K_valid≥2。
- [ ] 优先看 D（不依赖 PopQA 竞争答案），然后 C；默认同时计算，共享同题重复读出缓存。
- [ ] 报告 C/D 成功、low_signal、不可解析、无竞争者、改写拒绝的数量与比例；不能只给成功子集 AUROC。
- [ ] 输出每个方法共有样本上的 Outer，检查“新方法更高”是否仅来自覆盖变化。
- [ ] 按 J、有效 M/K、答案唯一数、题长/输出长度、PopQA 流行度、SuperGPQA 学科分层查看。不把这些变量加进 Inner。
- [ ] 看 Inner 单独 AUROC/AUPRC 及分布，尤其看相近 Outer 分箱内的区分能力。AUPRC 必须附实际正例率。
- [ ] 抽查 20 个高 Inner 与 20 个低 Inner：展示输入单元、编辑、C 的 a[k,j] 符号/波动，D 的 B/V/u；分别解释“强但稳定”与“弱信号不可测”。
- [ ] C 与随机方向比较；D 与范数方差、单纯输出表示方差比较。目标方向不胜随机方向、D 不胜输出方差时，不能声称新传播机制有效。
- [ ] 检查 PopQA 字面不同但同义的候选；当前没有语义同义合并，必须报告此局限。若增加合并器，其输入只能是预测文本，不能使用 gold aliases；冻结后另建版本。
- [ ] 检查 D 改写噪声；若主要高分来自语义改变，先修复改写合同，不能在错误改写上追指标。

## P2：补齐关键消融与公平对照

这些是下一轮实验/编码任务，未完成前不能写成“全部对照已验证”。

- [ ] 重复原视图的独立原生 replay 噪声测试（禁用缓存或跨进程），不能把缓存返回完全一样当作噪声低的证据。
- [ ] 同类型无关编辑、保义编辑、长度匹配编辑，排查 placeholder/OOD、语法破坏与长度混淆；与主中和同预算。
- [ ] 原 A/B/AB、outer-perplexity、LNPE、长度共用相同 frozen generation。旧 B 自带 Outer 与新版本定义不同的地方必须披露；必要时只比较旧 Inner 加同一 Outer 的独立消融。
- [ ] 同标签/同折的 Outer+长度 logistic、普通末层 probe，以及 Outer+长度+传播统计诊断模型。它们是有监督 baseline，不能重命名为 Inner；其训练预算与 E 一致。
- [ ] C 多竞争者方向、D 不同合法改写种子、J/M 预算曲线只在探索集/dev 选择；不得在全测试集选最好结果。
- [ ] 复现近期方法时锁定官方 commit/权重/原公式及全部依赖，缺权重写 blocked；末层替代中间层必须命名 variant。现有项目登记的候选包括：
  - [SeSE](https://github.com/SELGroup/SeSE)：检查原配置、NLI 与回答增强依赖，报告省略步骤。
  - [RAUQ](https://github.com/mbzuai-nlp/rauq-hallucination-detection)：原方法需要 attention；允许作为单独参考实验，但不能声称符合本方法纯推理框架部署合同。
  - [Semantic Energy](https://github.com/MaHAAA/SemanticEnergy)：审计现有 port 中 logprob 代替 scalar logit 的差异，不标成完全忠实复现。
  - [HAD](https://github.com/pku0xff/HAD)：运行前重新检查权重/代码可用性，不能用项目登记的旧 blocked 状态代替实时核验。
- [ ] 本清单是仓库已有近期方法的复现任务，不是截至某日期的完整 SOTA 调研。扩展新 baseline 时另行核对论文日期、官方实现与任务适配，避免仅选择早期弱 baseline。

## P3：E 的监督验证

- [ ] 完成 200 条测量后，先使用 `cde.e.mode: d`。运行下面两步；如果每折类别/实体不足，增加样本或预先调整折数，不静默退成逐行随机划分。

```bash
bash scripts/generation_eval/train_inner_e.sh config/cde_popqa_200.yaml
bash scripts/generation_eval/run_cde.sh evaluate config/cde_popqa_200.yaml
```

- [ ] cd 模式单列共有有效样本，绝不以 C 缺失补 0。当前 config hash 包含模式；切换模式须新配置/run_id，并重新通过准备/检测阶段，禁止篡改已有缓存 identity。后续可增加受控的特征缓存导入以避免重复 GPU 成本。
- [ ] 审计每折 train/test 的 PopQA 主体实体不交叉；SuperGPQA 精确问题相同分组，近重复题另外聚类审计。
- [ ] 核对 RMS、正则与非负权重只从训练数据拟合；核对输入是三/六个传播不确定性特征，且 feature_values、contributions、model_hash 可追溯。
- [ ] 报告 all-zero weights 的折数；如果全部为零，承认当前传播统计没有支持，不用 intercept 或最终乘上 Outer 的分数包装。
- [ ] 与同预算监督 baseline 比较，E 与 C/D 分开标注标签成本；现有分组 bootstrap 区间条件于已拟合 OOF 模型，正式监督结论还需重复整个嵌套训练或独立 heldout。

## P4：扩大两个 frozen run，预注册判定规则

- [ ] 冻结 200 条探索选择后，`limit: 0`、新 run_id 跑现有完整 run。已经查看结果的数据只称探索/复现实验；最好另外保留未用于选择方案的独立题目。
- [ ] 不再以截图的 Outer 0.660 / 0.896 作为不同样本/不同标签下的固定目标：必须在相同标签与共有 ID 上重算 Outer。
- [ ] 主表同时列 Inner、Outer、原乘法、gentle、覆盖、成本；配对比较以 `cde_evaluation.json` 的 paired_outer 为准。
- [ ] 晋级要求：Inner 本身有判别力；传播特定对照成立；控制 Outer/长度后有样本外增量；组合在共有样本上超过 Outer。两数据集分别报告，不用一个数据集提升掩盖另一个退化。
- [ ] 若 C/D Inner 仍接近随机或没有条件增量，停止加大融合权重。保留负结果，诊断末层读出是否编码相关语义、占位编辑是否过强、改写噪声是否占主导。
- [ ] 所有阈值、方向和超参预先冻结；不做测试集符号翻转，不从大量尝试中只报最优。两数据集收益都未验证前，不声称 beat SOTA。

## Judge 流水线与标签独立性

- [ ] 复用现有 judge provider/服务脚本；API LLM、JEV、StartLux、开源约 20B LLM 任取两个，固定 provider/model/prompt/revision、盲评原 greedy；分歧单列并人工抽审。
- [ ] 用户指定的 [StartLux-Decision-4B](https://huggingface.co/startlux-models/StartLux-Decision-4B) 是待部署目标；当前 base.yaml 配的是 27B GGUF，本次没有将它静默视为 4B。部署前核对真实模型/API 合同再配置。
- [ ] C/D 特征生成不能看到正确答案或 judge verdict；E 仅训练折可用二元标签。judge-continuous 若是标签来源，不作为“1.000 检测 baseline”排名。
- [ ] `label_source: judge` 仅消费与冻结 generation config hash 一致的源 judge.json；不一致需重新做明确的 generation 绑定，不能人工改 hash 绕过校验。

## 交付与验收状态

- [x] 新代码和运行入口、实施说明、本 TODO 随 Git 交付。
- [ ] 云端 GPU smoke 日志及原生 runner 证明。
- [ ] 两数据集 200 条探索结果、覆盖与解释案例。
- [ ] 必做控制与同监督预算对照。
- [ ] 完整 run 配对报告，保留失败样本和负结果。

本地 CPU 测试通过不代表 C–E 已提升 AUROC；真实推理与效果由上述云端验收决定。
