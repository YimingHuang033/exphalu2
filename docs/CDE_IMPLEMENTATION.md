# RePPL C–E 实现说明

2026-10-10；基于云端提交 `01ad1ae`。研究候选已编码，真实 GPU 跑通与检测收益待验证。具体执行任务见 [TODO.md](../TODO.md)。

## 保留的中心思想

输入语义单元 → 对答案语义表征的贡献 → 贡献的不确定性 Inner → 与生成不确定性 Outer 重标定。

C/D 不使用答案正确性 judge，不将输出一致性作为 Inner。E 只对传播波动统计做有监督、非负权重校准。解释是**干预下的表征贡献及其波动**，不是对基础模型真实因果路径或事实正确性的证明。

## 共用测量合同

- 冻结旧 run 的 dataset.json、generation.json，继续检测同一个 greedy 答案；原文件只读。
- SuperGPQA 字母映射为完整选项，选项块不改动；当前语义单元只取题干。PopQA 使用预测文字，候选同一性仅做保守字面规范化，不使用 gold aliases，也不声称已经解决同义答案聚类。
- 使用同一生成模型的原生 vLLM pooling/token_embed 重放；强制 model_impl=vllm，禁止静默落到 Transformers 模型前向。AutoTokenizer 仅用于分词。
- 一次性分词完整固定模板，取最后一个后缀 token 的末层状态；归一化为 g。绝不拼接分别分词的前缀/后缀，不读中间层、attention、梯度或完整词表 logits。
- 当前 SGLang backend 仍为 blocked，不能据此宣称已在 SGLang 验证。vLLM 路径复用云端已有 replay；新增模板仍需真实 GPU smoke。
- prepare 阶段只运行 generate；detect 阶段释放 generate 后只运行 pooling，避免交替加载。
- 同一题内缓存完全相同 token 序列；保留重复采样的统计频率。缓存不创造独立随机状态。
- J≤4，M≤4，默认保留至少 2 个有效视图。无有效单元/竞争答案/视图均显式 null + 原因，不补零。

固定模板见 [config/cde.yaml](../config/cde.yaml)。同一语义单元在保义视图中通过 unit_id 对齐；精确 span、角色、数字、实体受机械验证，并由只看问题的模型验证语义等价性。模型检查不等于人工金标；需要盲抽样审计。

## C：有方向贡献在输出采样间的波动

记 H 为末层归一化读出 g，候选概念编码为 E(c)，原题为 x，单元中和后为 x^-j。固定方向：

```
e = normalize(E(greedy_candidate) - E(competitor))
d[k,j] = g(x, sampled_candidate[k]) - g(x^-j, sampled_candidate[k])
a[k,j] = dot(d[k,j], e)
u[j] = std_k(a[k,j], ddof=0) / (mean_k(abs(a[k,j])) + tau_c)
Inner_C = mean_valid_j log1p(u[j]^2)
```

竞争者优先取原采样中最常见的非目标候选；MCQ 无竞争采样时固定种子选另一选项；PopQA 没有竞争候选则 C unavailable。不可解析/截断采样排除并记录，至少需要 2 个有效采样。若有效采样完全一致，贡献方差可为 0，这不是正确性保证。

低于 min_signal_c 的单元不进入均值；全部低信号返回 null。保存 a、mu、scale、std、u、有效单元与覆盖率。随机方向对照共用同一响应张量与成本。

## D：固定答案的保义输入视图传播波动

固定 greedy 候选 c0；只变换题干表达，不重新生成答案：

```
d[m,j] = g(x^m,c0) - g(x^(m,-j),c0)
B[j] = mean_m ||d[m,j]||^2
V[j] = mean_m ||d[m,j] - mean_m(d[m,j])||^2
u[j] = V[j] / (B[j] + tau_d)
Inner_D = mean_valid_j u[j]
```

默认归一化状态下 u 在 [0,1]。方向相反、范数相同的作用会被检测；仅幅度方差会遗漏它。J=1 仍有效，没有跨 j softmax。

同时输出 d-norm-only-inner 与 d-output-variance-inner。后者仅看未编辑问题的输出状态波动，不称为 RePPL。保存 B、V、平均作用向量、u、语义单元与完整编辑文本。弱信号不能解释为“可信”。

## E：传播波动统计的受限监督校准

- d 模式：D 的 u 的 mean/max/top2_mean；cd 模式再加入 C 的 log1p(u²) 的相同三项。
- 缺失任一所需测量则排除，不给 C 补零；J=1 的 top2_mean 等于唯一元素。
- 训练集 RMS 缩放、不中心化；使用非负权重的 L2 正则 logistic 模型，训练 intercept 只用于概率。

```
Inner_E = sum_l w[l] * feature[l] / train_rms[l], w[l] >= 0
P_train(error) = sigmoid(intercept + Inner_E)
```

Inner_E 不含 intercept。特征列表是硬性白名单，不包含 Outer、长度、judge score、原始 hidden state、贡献强度或 coverage。所有权重近零会保留 zero_weights 标记。

现有数据默认 5×3 嵌套分组 OOF：外层留出预测；内层选择正则；每次拟合的尺度只用训练行。PopQA 按 subj_id/subj，MCQ 按完整问题文本分组。保存训练/测试 ID、分组、特征、权重、模型 hash 与逐项贡献。**这是监督探索结果，不是独立盲测。**

## 分数、产物和实现边界

Outer 统一采用冻结 greedy 输出的平均负 logprob，剔除尾部 EOS，与旧 outer-perplexity 定义相同。包含 reasoning 的 run 继续使用完整原输出概率；Inner 候选取最后的答案。

分别报告 Inner、`(Inner + epsilon) * Outer`、`Outer * (1 + lambda * Inner)`。最后一种命名 gentle，仅是融合消融。

输出位于 results/generation_eval/<CDE_RUN_ID>/：

- config_snapshot.json：配置、源文件 SHA 与 CDE 版本。
- cde_prepared.json：单元、视图、语义检查、准备成本。
- cde_detection.json：各方法分数/状态、贡献统计、编辑、读出成本。
- cde_e_oof.json：监督训练折模型、OOF 预测、排除样本与标签来源。
- cde_evaluation.json：覆盖、AUROC/AUPRC、共有样本上相对 Outer 的配对差值与实体/问题组 bootstrap。旧单方法区间仍是行 bootstrap，已加注；比较结论优先使用 paired_outer。

缓存校验配置/源 generation/准备文件内容；E 还校验 detection 内容和标签 hash。修改准备文件后旧 detection 会被拒绝，必须显式 --force 重算。失败记录存在时默认跳过，要重试也需 --force。

当前未实现：跨模型观察器、SGLang 原生后端、独立 train/test E 部署、同预算 Outer+长度/末层 probe、无关/保义中和控制、多轮工具调用。不得用本次编码完成冒充这些实验已完成。
