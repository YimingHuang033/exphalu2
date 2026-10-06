# RePPL 2.0 A/B 方法与工程实施计划

> 日期：2026-10-04（baseline 清单修订）。本文是新建的实施规格，不代表以下代码或实验已经完成。  
> 前置研究：[RePPL 2.0 研究方案](/Users/tim/Projects/haluR/RePPL_2.0_Research_Proposal.md)。  
> 交付目标：实现 A/B 两种新方法、LLM-as-judge 模块及新增 baseline，建立可复现的单轮与多轮幻觉检测实验系统。  
> 核心约束：新方法的基础模型生成和状态提取必须通过 vLLM/SGLang 原生实现；不使用 Transformers 模型前向、中间层 hook、attention 或基础模型梯度。允许采样、批量重放和应用侧张量计算。

## 1. 实施决策与研究目标

第一阶段先交付 **A + 本地 LLM judge + 2025–2026 强 baseline + 统一评估**；第二阶段加入 **B 与解释忠实性实验**；第三阶段扩展 **多轮工具调用与近期强基线**。蒸馏不进入本轮必做范围。

A/B 均保留原 RePPL 的计算结构：

1. 构造输入对输出的作用代理；
2. 用多次采样之间的作用波动估计语义传播不确定性 InnerPPL；
3. 用生成概率估计 OuterPPL；
4. 以乘法重标定得到检测分数，并保留输入、输出两侧解释。

A 使用末层输入—输出关联，B 使用输入干预造成的末层响应变化。两者属于待验证的传播代理，不声称还原完整内部传播路径。LLM judge 是独立模块，不能将其分类概率改名为 InnerPPL。

需要验证的三个主假设：

- **H1，增量信息**：输入作用的跨采样波动，在输出概率、输出语义多样性和长度之外含有幻觉检测信息。
- **H2，重标定收益**：A/B 与 OuterPPL 的组合，在相同测试样本上优于各自单侧分数。
- **H3，解释有效**：输入解释能预测独立干预后的模型行为变化，而不只是展示高分片段。

“超过最新方法”作为实验目标；没有公平复现实验前，不写成已经成立的结论。

## 2. 现有工程的继承与迁移

当前工程含论文、历史实验目录和压缩代码包，不应直接在多个旧版本中同时改动。

参考入口：

- [原始方法实现](/Users/tim/Projects/haluR/AAAI/Supplementary%20Material/Code/ours/impl.py)：核对 CV、Inv、InnerPPL 和分数符号。
- [原始 baseline](/Users/tim/Projects/haluR/AAAI/Supplementary%20Material/Code/perform/baseline.py)：已有概率、energy、EigenScore、Semantic Entropy 相关实现。
- [原始生成流程](/Users/tim/Projects/haluR/AAAI/Supplementary%20Material/Code/perform/gen.py)。
- [较新代码包](/Users/tim/Projects/haluR/2026final/exphalu.zip)：作为迁移参考，记录选定文件的哈希，不修改归档。
- [现有原生信号验收程序](/Users/tim/Projects/haluR/verification/check_inference_signals.py)。

迁移要求：新建独立 `reppl2/` 包；复用经过核对的纯数学逻辑、数据转换和评估逻辑，替换旧模型前向及 attention 提取。新旧分数必须通过小型固定数组验证，避免符号或归一化变化被误认为性能提升。

旧 energy 依赖未归一化 logits；不能从 top-k logprobs 恢复原始 energy。旧 EigenScore 指定中间层；改成末层时必须命名为变体。旧语义蕴含器依赖也不能未经检查直接纳入“纯推理框架”路径。

## 3. A/B 共享数学与数据约定

### 3.1 输入、目标答案与采样

输入上下文为 x，划分成 J 个输入单元。采样 K 个输出 y¹…yᴷ，并生成一个待检测的 greedy 输出 y⁰。MVP 采用 K=5，再比较 K=3、10；采样温度、top-p、长度上限和随机种子写入配置，不在测试集上搜索。

第一阶段明确研究 **greedy 目标答案的幻觉检测**。如果数据集答案不是本模型在该上下文下生成的，只能进入单独的固定答案评分设置：Outer 使用该答案的原生条件评分，Inner 仍来自该上下文的新采样，标记为 `fixed_answer` 变体，不能与原始设置混在同一表中。

输入单元先采用问题片段、RAG 文档段落、历史轮次、工具返回字段；每个单元保留字符范围、token 范围、来源和时间。系统指令、模板符号、分隔符与特殊 token 独立标记，默认不作为事实输入单元。

### 3.2 共享聚合

A/B 都输出非负矩阵 A[k,j]。计算：

```text
mu[j]    = mean_k A[k,j]
sigma[j] = std_k A[k,j]                 # MVP 固定 ddof=0
r[j]     = sigma[j] / (mu[j] + tau)
p_hat[j] = 1 / (1 + r[j]**alpha)
Inner    = mean_j log(1 + r[j]**alpha)
Outer    = -sum_t log p(y0[t] | x, y0[:t]) / mean_k len(yk)
risk     = (Inner + epsilon) * Outer     # 越大越可能有幻觉
legacy_score = -risk
```

`tau/alpha/epsilon` 必须版本化；正式复现原 RePPL 时另核对原实现的标准差约定。`p_hat` 是变换量，不是经校准的事实正确概率；Outer 名称沿用原工作，不把它误写成已经取指数的普通 perplexity。

默认对非特殊输出 token 计长并求和，EOS 等 token 是否计入作为显式配置；分子、平均长度和各 baseline 的规则需统一。因果语言模型状态与 logprob 的位置约定必须分别处理：位置 i 的末层状态是消费 token i 后的表示，token i 的条件概率由此前位置预测，不能直接用同一索引推定对齐。

异常规则：K<2、无有效输出、全零作用、近零池化向量、非有限状态均返回状态码和原因；不得默默写成零风险。相同采样的真实低变异可以保留，但要记录唯一输出数量，不解释成正确性保证。

输入解释同时给出 `mu`（作用强度）与 `r`（不稳定性）；输出解释给出逐 token NLL 及按片段聚合结果。对不同长度采样不强行按 token 序号对齐，跨采样比较始终沿输入单元 j 进行。

## 4. 方案 A：末层语义关联的跨采样不确定性

### 4.1 必做算法

1. 批量生成 y⁰、y¹…yᴷ，持久化实际 token IDs。
2. 重放每个 `[context_ids, sampled_output_ids]`，提取逐 token 末层状态。
3. 对输入单元 j 的状态做均值池化，得到 u[j]；取采样输出 token 状态 v[k,t]。
4. 对每个输出 token 计算输入单元间的关联：

```text
w[k,t,j] = softmax_j(cosine(u[j], v[k,t]) / association_temperature)
A[k,j]   = mean_t w[k,t,j]
```

5. 调用共享聚合器输出 Inner、Outer、risk 和输入解释。

MVP 不训练额外检测器，不加外部 judge。应用侧可用 NumPy/PyTorch 做池化和矩阵运算；这不涉及 Transformers 模型前向。精度计算至少使用 float32，防止低精度微小方差失真。

### 4.2 实现细节与消融

- 关联温度与生成温度分开配置、分开搜索。
- 主版本使用全部有效输出 token；事实片段、工具名及参数位置为扩展消融。
- 必须保存原始相似度统计。softmax 即使面对全不相关输入也会分配权重，不能把权重直接称作证据支持概率。
- 比较输出均值表示多样性、原始关联波动、归一化关联波动；证明输入定位结构的收益。
- 去均值或白化只作为明确消融，统计参数只能用训练数据估计。
- 输入状态复用仅在确认因果前缀、模板、位置和实际返回状态一致后启用；MVP 使用完整重放。

### 4.3 交付与验收

交付 `RepplA`、逐输入作用矩阵、最终分数和可读解释记录。使用手工小矩阵验证 softmax 轴、CV 轴与池化范围；测试输入分片顺序重排只会对应重排解释，不应改变整体聚合。

额外模型工作约为 K 次状态重放，加一次 greedy 与 K 次采样；相似度计算量随输出长度、J、hidden size 增长。测量实际耗时与传输量，不使用“零额外开销”描述。

## 5. 方案 B：末层干预影响的跨采样不确定性

### 5.1 必做算法

对每个输入单元 j 构造受控编辑 x⁻ʲ。固定每个已采样答案 yᵏ 的 token IDs，分别在 x 与 x⁻ʲ 下重放：

```text
z[k]       = mean_pool(output_hidden(x,    fixed_ids_yk))
z_edit[k,j]= mean_pool(output_hidden(x^-j, fixed_ids_yk))
q[k,j]     = norm(l2_normalize(z[k]) - l2_normalize(z_edit[k,j]))
A[k,j]     = q[k,j] / (sum_j q[k,j] + delta)
```

将 A 交给共享聚合器。必须同时实现以原始 q 计算 CV 的消融，以识别归一化耦合的影响。编辑后重新计算上下文长度和输出区间，绝不沿用原始绝对位置。

**q 是影响，不是不确定性；跨 k 的变化才进入 Inner。** 主算法不重新生成编辑后的答案。编辑后重新生成用于独立解释验证，避免把输出内容变化混进配对状态测量。

### 5.2 编辑策略

MVP 使用确定性、可审计编辑器：

- RAG：屏蔽单个证据片段，保留文档结构。
- QA：中和预先定义的事实性条件；删除必要任务指令的情况不纳入默认编辑。
- 工具：替换字段值为明确的未知状态，保持 JSON 类型及协议结构；不能把 `null` 默认当作合法值。
- 历史：编辑一个历史事实片段，保留角色和工具调用配对。

每次编辑保存原文、替换内容、字符/token 范围、结构校验结果和编辑类型。无法构造合法编辑则跳过该单元并报告覆盖率。

删除会改变长度及位置编码；必须设置长度匹配的无关片段编辑、保义改写和另一种编辑方式作为控制。MVP 可使用显式未知值，但该值也可能引入分布偏移，不能把所有响应变化视为语义因果影响。

### 5.3 成本控制与验收

完整 B 约需 K×(1+J) 次重放。先在小样本、J≤8 的受控设置运行；超限样本明确记录策略，不默默截断证据。

分别比较随机选 J 个单元、固定结构分片、A 提名 top-J。A 提名版本命名为 `RepplAB`，与全量 `RepplB` 分开报告，不能只挑 A 最有利的片段证明 B 有效。未测量单元标记为 unavailable，不补零。

验收必须覆盖：编辑后输出 IDs 完全不变、只池化答案状态、JSON/聊天协议合法、原始 q 与归一化 A 分开保存、退化情况可追踪。B 无明显增量时保留其作为解释分析工具，不强行扩大昂贵实验。

## 6. 推理框架适配与接口合同

### 6.1 统一后端接口

```python
class InferenceBackend:
    def capabilities(self): ...
    def sample(self, context_ids, sampling_config): ...
    def score(self, context_ids, fixed_output_ids): ...
    def replay_last_hidden(self, context_ids, fixed_output_ids): ...
```

返回对象必须包含输入/输出 IDs、逐 token 有效掩码、位置映射、状态形状、实际模型 revision、框架版本、dtype、模板哈希、缓存模式和计时。能力缺失必须直接报错，禁止静默切换到 Transformers 或其他层。

框架能力按当前目标模型实测确认，不把一个模型通过推广成所有模型都通过。

### 6.2 原生实现路线

- **vLLM**：生成 runner 获取采样及实际 token logprobs；pooling runner 使用转换后的原生生成模型、`token_embed` 返回逐 token 末层状态。两个 runner 可以分时加载，显存计入成本；不假设生成接口同时返回全量状态。
- **SGLang**：原生 Engine 的 full hidden-state 模式重放完整文本。`last` 只返回最后位置时不满足 A/B；必须检查每个需要的 token 都存在。完整 prefill 状态验收要控制前缀缓存。
- 完整答案重放时，若接口通过生成一个额外 token 才返回全部所需状态，该额外 token 不计入答案、Outer 或表示池化。
- 固定输出直接拼接 ID；不把文本重新分词后默认当作原采样序列。输入编辑可改变 prefix IDs，输出 IDs 保持固定。
- 使用基础模型自身末层状态；专用 embedding 投影头、额外 pooler 归一化、模型自身 final norm 需分开记录，不混用为同一表示。

已核实的源码快照和测试证据见前置文档第 14 节；它们支持实现路线，但本地尚未做 GPU 验收。工程起点必须运行现有 `verification/check_inference_signals.py` 的相关模式，并保存真实 JSON 结果。

### 6.3 缓存和数据量

生成、评分、状态重放、judge 分开缓存。键至少包含模型 revision、后端版本、模板、完整 IDs、采样配置、种子、编辑版本、表示设置与 dtype。多次采样不能因缓存复用变成重复同一条随机输出。

默认只持久化作用矩阵、池化表示及必要统计；完整 token 状态用于抽样审计，设置磁盘预算。A 在线分块计算关联，B 得到输出池化后尽早释放矩阵。冷启动和暖缓存成本分别报告。

## 7. LLM-as-judge：必做独立模块

### 7.1 三种用途分离

**检测 baseline（必做）**：一个独立本地 instruct 模型，通过 vLLM/SGLang 判断目标回答；至少比较同模型 self-judge 与异模型 judge，避免把两者混称。

**辅助标注（必做接口）**：对没有现成标签的数据生成候选标注；独立保存为 weak labels。测试集优先使用已有人工标签，并抽样人工核验；弱标签数据单独评估，不由同一 judge 自己证明优越。

**语义分组（可选增强）**：为 SeSE、Semantic Energy 等近期方法或分组版 Inner 提供经各方法核对的语义关系判断。它不参与 A/B 主版本的必要计算。

Jev 作为额外 hosted provider；本地 judge 是必做路径，Jev API 可用时再执行。未确认 Jev 可本地部署，因此 Jev 不是纯 vLLM/SGLang 主方法依赖。对象仍按 TypeSafe AI Jev 理解。

### 7.2 输入与输出合同

输入包含问题、目标答案、检测时点可见证据、工具 schema 及已发生的工具记录。指令明确区分外部内容与评估规则；被评文本中的命令不得作为 judge 指令。

输出结构建议：

```json
{
  "sample_id": "...",
  "status": "ok",
  "verdict": "supported|contradicted|insufficient_evidence|not_applicable",
  "claims": [
    {
      "text": "...",
      "answer_char_span": [0, 10],
      "verdict": "contradicted",
      "evidence_ids": ["tool_3.result.status"],
      "brief_reason": "..."
    }
  ],
  "risk_score": 0.0,
  "score_type": "label_likelihood|vote_fraction|self_report|provider_probability",
  "model_revision": "...",
  "prompt_version": "..."
}
```

这是接口示意，`risk_score` 必须由实现定义计算；示例中的 0.0 不可作为缺失默认值。JSON 可解析不代表内容正确；引用 ID 与字符范围必须由程序校验。简短依据即可，不要求隐藏推理过程。

### 7.3 judge 连续分数与鲁棒性

本地 MVP 使用固定标签评分：对同一 judge prompt 重放各标签，得到序列条件 log-likelihood；优先设计等 token 长度标签，用 logsumexp 在候选标签间归一化。该数值是**指定候选标签集合内的相对概率**，未经校准不称为事实错误概率。记录标签顺序与 tokenization，并做顺序敏感性检查。

结构化解释可另一次生成；分类分数与解释生成的调用都计入成本。语法约束后的采样概率和原始模型标签概率不能混用。

同时支持少量重复 judge 的投票率作为消融；自报 confidence 独立标记，不视为校准概率。超时、格式失败、拒答返回缺失状态，统一报告覆盖率，不把失败样本删掉后直接宣称高分。

Jev 使用 provider 返回的幻觉类别概率；`1-confidence` 不等同于幻觉风险，因为 confidence 是判断分布的摘要。[官方 confidence 文档](https://docs.typesafe.ai/confidence)

### 7.4 幻觉定义与标签映射

必须预先区分事实错误、证据不忠实、无依据断言、错误工具选择、虚构参数、虚构执行结果。普通任务失败、工具超时和正常拒答不自动属于幻觉。

`insufficient_evidence` 是否计为正例取决于数据集定义，映射保存在数据集适配器中；无外部证据的任务不能要求 judge 把所有未知事实判为错误。证据访问一致的比较与外部检索增强比较分开。

### 7.5 可插拔双 judge 流水线：任选两个实现

**新增必做对象：[startlux-models/StartLux-Decision-4B](https://huggingface.co/startlux-models/StartLux-Decision-4B)。** 官方模型卡说明其采用 TypeSafe `/v1/systemone` 请求/响应格式，并随模型提供本地 `startlux_decision` 服务。可据此复用 Jev 协议适配器；这表示接口兼容，不表示权重、训练方法或概率校准等价。代码入口：[模型仓库中的推理包](https://huggingface.co/startlux-models/StartLux-Decision-4B/tree/main/startlux_decision)。未确认独立官方 GitHub，不编造链接。

官方快速路径使用自带推理包，不能据此宣称 StartLux 已支持 vLLM/SGLang。它作为隔离的本地 judge 服务运行；RePPL A/B 的生成与状态前向仍必须原生运行在 vLLM/SGLang。权重按模型卡为 CC BY-NC 4.0，属于本计划的开放权重研究基线，不标为无限制商用开源模型。部署兼容与实际检测性能分别验收。

#### 7.5.1 Provider 注册表

注册表至少包含四个独立实例，模型、地址与鉴权均由配置提供：

- `api_llm`：远程通用 LLM API。记录供应商、实际模型版本、接口协议及是否返回 logprobs；不限定单一供应商，也不假设所有 API 都兼容同一协议。
- `jev_api`：TypeSafe Jev 托管决策接口，保留类别概率及实际模型标识。
- `startlux_local`：指定的 StartLux-Decision-4B 本地服务，采用 SystemOne 协议；必须有单模型评测结果。
- `open20b_local`：开放权重约 20B 级通用 instruct LLM，通过 vLLM/SGLang 服务。模型可换；一个已查到官方 vLLM 部署说明的候选是 [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)，24B 作为约 20B 级实例，不宣称它是当前最强选择。正式实验冻结具体模型与 revision，记录总参数和 MoE 激活参数，不能混称。

`protocol`、`location`、`model_family` 分开登记：本地模型通过 HTTP 提供服务，不会因此变成远程闭源 API 组。新增模型只需注册 adapter，无须改流水线主体。

#### 7.5.2 配对配置

主模式为 `parallel_independent`：两个 judge 读取同一规范化任务、证据和预先固定的事实片段，各自判断后再聚合，互相不看对方结果。四个实例共六个无序组合，必须均能通过配置选择：

1. API LLM + Jev；
2. API LLM + StartLux；
3. API LLM + 本地约 20B LLM；
4. Jev + StartLux；
5. Jev + 本地约 20B LLM；
6. StartLux + 本地约 20B LLM。

不限于“一路 LLM + 一路决策模型”，允许 LLM+LLM、Jev+类 Jev。注册表扩展后也允许同类型的两个不同模型实例。禁止默认选择同一实例两次；同模型重复采样仅作为重复判断消融，不能当作模型异质性证据。

以下是拟实现配置，环境变量为空或 revision 未冻结时正式实验应拒绝启动：

```yaml
judges:
  api_llm:
    adapter: llm_api
    protocol: ${API_LLM_PROTOCOL}
    base_url: ${API_LLM_BASE_URL}
    api_key_env: API_LLM_KEY
    model: ${API_LLM_MODEL}
  jev_api:
    adapter: systemone
    base_url: ${JEV_BASE_URL}
    api_key_env: JEV_API_KEY
    model: ${JEV_MODEL}
  startlux_local:
    adapter: systemone
    base_url: http://127.0.0.1:8090
    model: startlux-models/StartLux-Decision-4B
    revision: ${STARTLUX_REVISION}
    runtime: startlux_decision
  open20b_local:
    adapter: llm_api
    protocol: openai_chat_compatible
    base_url: http://127.0.0.1:8000/v1
    model: ${LOCAL_20B_MODEL}
    revision: ${LOCAL_20B_REVISION}
    runtime: vllm
judge_pipeline:
  members: [startlux_local, open20b_local]
  mode: parallel_independent
  fusion: mean_score
  disagreement_policy: abstain_for_label
  on_member_failure: mark_pair_incomplete
  allow_implicit_third_judge: false
```

配置切换在下一次 run 或 batch 生效；进行中的请求保留原配置快照。模型切换产生新的 `pair_id/config_hash`，不能覆盖旧结果或误用缓存。这里给出的是接口设计，不代表上述配置已有执行器。

#### 7.5.3 判别、分数与解释的统一

- 相同的事实定义、标签集合、证据和裁剪规则输入两路。上下文过长时采用共享裁剪结果或共同拒绝；不同模型各自静默截断不算公平双 judge。
- SystemOne adapter 将 choice/noul/score 返回值映射到统一 schema；MVP 以同一 choice 标签集为主，保存原始类别概率。Jev 与 StartLux 都使用幻觉类别概率而非 `1-confidence`。
- 普通 LLM 优先用固定标签 likelihood；API 不提供该能力时可用预定义重复投票率，自报分数须单独标记。硬标签只能作为硬投票，不能伪装成概率。
- 默认 `mean_score=(s1+s2)/2`，要求两路都是已声明且同方向的 [0,1] 分数。它只是融合排序分数，不自动成为校准概率。额外报告 dev 集独立校准后的均值融合；权重和阈值不能在测试集挑选。
- 一路没有连续分数时，默认连续融合返回 unavailable，仍可输出明确命名的双路硬判决；不得隐式填 0 或重新加权成单模型结果。
- 保存 `member_results`、原始分数、融合分数、hard verdict、agreement、score gap、证据 ID、耗时与费用。两票冲突不能做“多数票”；可以保留融合风险用于排名，但辅助标签标记 `needs_review`，不能当 gold label。
- 决策模型不保证生成自由文本理由。其解释以共同事实片段的概率与证据映射为主；若 LLM 补充理由，记录 `explanation_provider`，不能声称是 StartLux/Jev 的内部解释。
- 每路分别校验异常、超时和输出 schema；有限重试后标记 pair incomplete。可配置降级为单 judge 服务，但实验中必须单列 degraded，不计作完整双 judge 结果。

#### 7.5.4 可选级联模式

`cascade` 使用相同的两个成员，顺序显式配置，例如 StartLux → 约 20B LLM 或 Jev → API LLM。第一路遇到验证集确定的模糊区间、证据不足或格式失败才调用第二路；触发阈值、升级比例和最终决策规则提前冻结。

级联不等同于双路全量融合。初版升级后采用第二路结果，若要融合另设配置。两路置信分布不可直接比较，因此不跨模型共用一个未经验证的 confidence 阈值。报告全量质量、升级率、错误逃逸率和总成本。

默认不引入第三个仲裁器；分歧时输出可审核状态。任何未来三路仲裁必须作为独立实验，不能藏在“任选两个”的流水线里。

#### 7.5.5 与 RePPL 及评估真值的关系

新增评测命名为 `Judge-StartLux4B`、`Judge-Open20B`、`JudgePair-<id1>-<id2>` 和 `JudgeCascade-<id1>-to-<id2>`，并保留 Jev/API LLM 的单路结果。至少比较四个单模型与六个双路组合；API 不可用的组合标记未运行，不补造结果。级联先测试两组预算友好的组合，再按 dev 结果扩展。

这些属于外部判断基线/辅助标注流水线；RePPL-A/B 独立计算 Inner×Outer，不被双 judge 分数替换。可另测 `RePPL + JudgePair`，但融合系统不得冒充纯 RePPL。

双 judge 一致并不是真值。最终测试依赖独立人工/已有可靠标注；弱标签保留两个来源、分歧与审核状态。重点统计高置信共同漏检、错误重合，以及 RePPL 在其中能否提供额外召回。双路接口支持不等于性能改进已成立。


## 8. Baseline 引入清单：2025–2026 强竞争方法

> 修订要求：新增竞争 baseline 以 2025–2026 年工作为范围。SelfCheckGPT、原始 Semantic Entropy、KLE、LLM-Check、FActScore、VeriScore 不再作为本次新增 baseline，也不占实施里程碑。历史代码可留存。Outer-only、Inner-only、长度和输出几何对照是必要消融，单列且不用于充当强 baseline。

选择标准同时考虑任务直接相关性、论文证据、官方代码/权重及可复现性；“年份新”本身不证明性能强。下列论文结果是各自设置下的报告，不是已经在本项目验证的统一排名。

### 8.1 P0：首批必须接入的近期竞争者

**1. SeSE，2025 首稿、2026 修订；作者页面注明 UAI 2026 接收。**

当前题名为 *SeSE: Black-Box Uncertainty Quantification for Large Language Models Based on Structural Information Theory*。它以语义结构熵和编码树建模采样输出的不确定性，并覆盖长文本细粒度估计，是 A/B 的直接采样语义不确定性对手。优先核对 v4 公式、语义关系模型、树构造和长文本规则，固定版本后实现；不得用普通聚类熵冒充 SeSE。共享生成采样池，但语义处理成本单计。是否能将其语义模型完整迁入推理框架，需要单独核验，不预先承诺。[论文 v4](https://arxiv.org/abs/2511.16275v4)；[官方 GitHub：SELGroup/SeSE](https://github.com/SELGroup/SeSE)

**2. HAD，2025 预印本 / ACL Industry 2026。**

监督幻觉检测、片段定位与纠正的一体化模型，作为 QA/RAG/长文本的强外部验证器。优先接官方权重与提示，记录训练数据、证据输入和输出解析；基础模型原生支持时可通过 vLLM/SGLang 运行，需实测确认。官方输出若没有连续概率，按原协议报告分类指标，新增概率评分另命名，不凭空制造 AUROC。[论文](https://arxiv.org/abs/2510.19318)、[官方 GitHub：pku0xff/HAD](https://github.com/pku0xff/HAD)

**3. D-Score，2026 年 7 月预印本。**

单次前向的隐状态谱统计，针对“我们是否只是换了一个 embedding 分数”提供直接对照。核对原论文层选择、容差、状态区间和评分方向；若原版使用非末层，则原版与 `D-Score-last` 分开报告。不能把强制末层的改版当成原论文最佳结果。优先在其适用的 RAGTruth/FAVA 类任务比较。[论文](https://arxiv.org/abs/2607.24586)

**4. 当期独立 LLM-as-judge、Jev、StartLux-Decision-4B 与可配置双 judge。**

本地 judge 必做，并在实验冻结时选定当期模型的明确 revision；同时报告 self-judge 与独立 judge。Jev 作为 2026 外部 API 概率判别基线，接入 direct 与 localized 两种设置；无 API 运行条件时标注 blocked，不能用本地模型替代后仍叫 Jev。LLM-as-judge 是系统基线，不声称它是新提出的 2026 算法。两者均不得充当自身测试真值。[Jev 官方说明](https://docs.typesafe.ai/introduction)、[2026 评测一](https://arxiv.org/abs/2609.26550)、[2026 评测二](https://arxiv.org/abs/2609.29769)

新增必做 [StartLux-Decision-4B](https://huggingface.co/startlux-models/StartLux-Decision-4B) 和约 20B 本地通用 LLM；与 API LLM、Jev 组成任选两个的六种组合，详见第 7.5 节。StartLux 的自带本地服务不等于 vLLM/SGLang 原生实现。

P0 的目标是尽早覆盖近期采样 UQ、监督验证器、内部表示检测器及外部 judge，避免只完成便宜对照就开始声称竞争力。

### 8.2 P1：正式论文必须处理的强相关方法

**5. RAUQ，ICML 2026。**

*Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads*。通过不确定性感知 attention heads 与 token confidence 的循环组合估计风险。这与 RePPL 的异质信号组合及效率主张非常接近，应列为核心竞争者；论文报告在多任务、多模型上优于其 UQ 对照。原版需要 attention，因此放入隔离参考环境，不假装能以末层 cosine 精确替代。我们的 vLLM/SGLang 原生约束仍保持不变；与 RAUQ 比较质量、解释和端到端部署成本。[ICML 官方论文](https://proceedings.mlr.press/v306/vazhentsev26a.html)；[官方 GitHub：mbzuai-nlp/rauq-hallucination-detection](https://github.com/mbzuai-nlp/rauq-hallucination-detection)

**6. LAFaCT，ACL 2026。**

事实关键 token 归因定位后进行隐状态序列分析，直接竞争检测性能与定位解释。优先官方实现，保留原层选择及归因步骤；其内部状态/梯度需求由隔离参考环境满足。不能以自制末层 probe 替代原版，也不能因为部署不同就忽略性能比较。[官方论文](https://aclanthology.org/2026.acl-long.312/)

**7. LaaB，ACL 2026。**

建模回答与自判断之间的标签逻辑约束，作为异质信号融合和 judge 增强的直接对手。工程接入需复现原训练/推断协议、标签约束和监督预算，区别于简单平均两个分数。[官方论文](https://aclanthology.org/2026.acl-long.286/)；[官方 GitHub：ICTMCG/LaaB](https://github.com/ICTMCG/LaaB)

**8. Semantic Energy，2025，采用最新已核实的 v3。**

结合语义聚类与 energy 信号，是较新的采样不确定性对照。原文涉及 logits 的 energy 计算；归一化 logprobs 丢失加性常数，不能据此恢复原始 energy。必须核对作者实际所用 logits 的层和词表范围，再选择参考运行环境。主方法不因此增加中间层依赖；无法取得原版信号时报告阻塞，不以近似 energy 冒充复现。[论文 v3](https://arxiv.org/abs/2508.14496v3)；[作者 GitHub：MaHuanAAA/SemanticEnergy](https://github.com/MaHuanAAA/SemanticEnergy)

### 8.3 多轮工具调用的专项比较

- **Internal Representations as Indicators of Hallucinations in Agent Tool Selection，2026**：核对工具名、参数等位置及监督协议；与 A/B 在同一工具错误标签和可见历史上比较。[论文](https://arxiv.org/abs/2601.05214)
- **SAUP，ACL 2025；UProp，2025**：检验轨迹风险建模是否超越简单 max/top-q。先核对其目标是步骤风险、任务失败还是幻觉，只有标签适用的设置才直接比较；适配版本显式命名。[SAUP](https://aclanthology.org/2025.acl-long.302/)、[UProp](https://arxiv.org/abs/2506.17419)；[UProp 作者 GitHub](https://github.com/jinhaoduan/UProp)（目前为占位仓库，未见方法实现）
- **AgentUQ、AgentHallu、PROBE、OpenHalDet**：用于任务定义、数据或评估协议，不冒充检测算法 baseline。新增 OpenHalDet 作为统一评估实现参考，审查数据划分和标注来源后再复用。[OpenHalDet，2026](https://arxiv.org/abs/2606.06959)；[论文所列 GitHub：Nellie179/Hallucination-Detection](https://github.com/Nellie179/Hallucination-Detection)

### 8.4 必须保留的自消融，不计入竞争 baseline 数量

保留 Outer-only、Inner-A/B-only、A/B 乘法重标定、平均作用量 mu/q、输出几何统计、答案长度及采样数控制。它们回答“新传播不确定性是否有增量”，与“能否超过近期强检测器”是两项独立验收要求。

### 8.5 接入与复现门槛

每个方法登记论文版本、代码 commit、权重、适用任务、所需状态、证据权限、监督量、采样和评分预算、原版/适配版、实现状态及阻塞原因。状态为 `planned / implemented / verified / evaluated / blocked`。本次仅完成选型和计划修订，不代表已经运行。

P0 优先落地；P1 不能因实现困难被自动删掉。正式主表必须在适用场景包含近期语义 UQ、监督验证器、内部状态方法和 judge 的代表性结果。缺少某类强对手时，缩小性能结论，不以旧 baseline 补数后声称 SOTA。

新方法保持原生推理约束；需要 attention/中间层的强 baseline 可以独立参考环境复现，结果文件通过统一 schema 导入。原版与推理框架适配版分开命名，并分别核算成本。

### 8.6 GitHub 核实与代码可用性（2026-10-04）

上面的链接已补到相应方法条目。仓库公开不等于实验已复现；接入时仍需锁定 commit、检查权重与数据依赖。

- **SeSE**：[SELGroup/SeSE](https://github.com/SELGroup/SeSE)。已见短文本与长文本代码目录、环境配置和运行说明。
- **HAD**：[pku0xff/HAD](https://github.com/pku0xff/HAD)。作者代码与数据仓库；正式接入时按其说明核对模型权重版本。
- **RAUQ**：[mbzuai-nlp/rauq-hallucination-detection](https://github.com/mbzuai-nlp/rauq-hallucination-detection)。由 ICML/PMLR 官方页面的 Software 链接指向，已确认仓库可访问。
- **LaaB**：[ICTMCG/LaaB](https://github.com/ICTMCG/LaaB)。README 明确标为对应 ACL 2026 论文官方实现，包含特征提取、训练和实验脚本。
- **Semantic Energy**：[MaHuanAAA/SemanticEnergy](https://github.com/MaHuanAAA/SemanticEnergy)。作者标为临时仓库，提供 notebook 和中间数据入口；不是完整的一键实验包。以该具体仓库为接入入口，不使用论文中较笼统的 GitHub 账号链接。
- **UProp**：[jinhaoduan/UProp](https://github.com/jinhaoduan/UProp)。论文明确指向此地址，但本轮看到的文件只有 README、LICENSE 和 .gitignore。登记为“官方地址已确认、实现尚不可用”，不能标记为代码已接入。
- **D-Score、LAFaCT、工具选择内部表示检测方法、SAUP**：截至本轮核实，未找到能确认属于对应论文的公开官方实现；保留论文链接，代码字段标记 `official_repo_unconfirmed`，这不等同于断言作者没有代码。LAFaCT 的搜索结果中有另一篇 *Latent Fact-Checking* 的同名项目，不能用它替代本计划的 ACL 2026 LAFaCT。
- **Jev**：本计划已核实的是托管 API，未确认可供本地复现模型的官方 GitHub/权重；保留[官方接口文档](https://docs.typesafe.ai/introduction)，不把 SDK 或第三方封装当作模型开源实现。
- **LLM-as-judge**：这是本项目自建适配模块，没有单一对应论文仓库；最终登记实际 judge 模型、revision 和后端。

评估工具另列：[OpenHalDet](https://github.com/Nellie179/Hallucination-Detection) 为其论文所列仓库；[AgentUQ](https://github.com/deeplearning-wisc/agentuq) 为前置研究已核实的项目入口。二者不计入检测算法 baseline 数量。

## 9. 数据、评估与多轮工具调用

### 9.1 数据合同与划分

统一 `Example` 包含：`sample_id, task, messages, evidence, target_answer, generation_mode, visible_history_cutoff, labels, label_source, spans, trajectory_id, split`。

至少覆盖单轮 QA、给定证据的 RAG/摘要、长文本事实、工具调用轨迹。MVP 每个入选场景可先用约 100–200 条做故障排查，这不是正式性能样本量；正式规模依据先导集方差、类别分布与配对检验需要确定。

按问题/文档/整条轨迹划分 train/dev/test，同一输入的采样和干预只属于同一 split。所有统计变换、阈值、聚合和校准在训练/验证集固定；测试集不反复选方法。保留至少一个未见任务和一个未见生成模型评估迁移。

自然错误是主结果，人工注入错误用于机制验证并单列。judge 生成的错误或标签不能成为唯一测试来源。

### 9.2 检测与解释指标

检测：AUROC、AUPRC 与正例率、验证集固定工作点下的测试召回/实际 FPR、风险—覆盖曲线；有标注校准版另报 Brier/ECE。分数未校准时不报告成概率校准结果。

以样本或完整轨迹为重采样单位做配对 bootstrap，报告置信区间及各任务结果。超过 baseline 的主张必须同集、同标签定义、同证据权限；原始论文表中的跨设置数字仅作背景。

解释：片段定位质量、首次错误步骤定位、输入编辑后的行为变化、随机/无关编辑对照。验证 B 时要使用不同于构造 q 的编辑或行为指标，避免循环证明。输入高贡献不等于输入事实错误，输入高不稳定也不一定等于最终错误责任。

成本：采样数、实际生成/重放/judge tokens、模型调用数、端到端 p50/p95、峰值显存、传输/磁盘量与 API 费用。分别呈现相同 K、相同 token 预算及真实时间—质量曲线，不能只按相同 K 宣称成本公平。

### 9.3 多轮实施

固定某一步之前真实可见历史，采样当前工具调用或回答，构造输入单元和 A/B。只分析候选动作文本；额外采样不执行真实退款、发送等有副作用工具。

严格分开：调用前的工具名/参数检测、工具返回后的解读检测、最终回答检测。调用前不能看到未来结果；工具结果中的 `pending` 到后续 `success` 属于状态变化，标签必须带时间。

主版本逐步出分，轨迹聚合先用预先固定的 max 与 top-q；同时记录修复事件，不把后续正确修复简单当作仍在传播的错误。依赖图和复杂传播权重作为后续扩展，不在首轮引入过多自由参数。

### 9.4 新增短上下文数据集候选与下载入口（2026-10-06 核实）

目标：为 Qwen2.5–Qwen3.5、**总参数量小于 10B** 的被测模型寻找自然错误比例约 40%–60% 的任务，优先短 QA、函数级代码、有限工具集和短对话。以下是服务器开发者的下载/接入清单，**全部为 `planned`，本轮未下载完整数据、未完成适配、未在本项目模型上实测**。链接优先采用作者仓库、官方 Hugging Face 数据仓库和原始评测结果。

**先区分两个目标。** `1 - accuracy/pass@1` 是答题或任务错误率，不自动等于幻觉率；BFCL 总分还是跨类别汇总指标，其补数不能当作逐步工具幻觉率。若需要真正的 40%–60% 幻觉率，必须按 §9.5 的独立幻觉标注再校准。不能承诺同一份题在 0.5B、7B、9B，或 thinking/non-thinking 两种模式下均落入该区间。

#### 9.4.1 首批下载：有接近目标区间的公开分数

**A. SuperGPQA：QA 首选，特别适合 Qwen3.5-4B/9B。**

- 下载：[官方数据 `m-a-p/SuperGPQA`](https://huggingface.co/datasets/m-a-p/SuperGPQA)；[官方评测实现](https://github.com/SuperGPQA/SuperGPQA)。26,529 道题；HF split 名为 `train`，但这里用作评测题池，不据此当训练集。字段为 `uuid/question/options/answer/answer_letter/discipline/field/subfield/difficulty/is_calculation`。
- 难度证据：Qwen 官方报告 4B/9B 的准确率分别为 **52.9% / 58.2%**，相应答题错误率 **47.1% / 41.8%**。这是原评测协议下的分数，不是下述短输入、关闭思考变体的保证。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B#benchmark-results)
- 接入建议：保留题干及全部选项，按 `difficulty`、学科分层；初选完整渲染输入 ≤1,024 tokens 的题。用 `answer_letter` 严格判选项，不把答案、解析放进 prompt。支持 4–10 个选项，不能把解析器写死成 A–D。若去掉选项改成开放 QA，另命名变体并重新判分、校准。数据许可为 ODC-BY，部分来源题还需保留原来源许可/归属。

**B. BigCodeBench-Full：Qwen2.5-7B 的函数级编程首选。**

- 下载：[官方数据 `bigcode/bigcodebench`](https://huggingface.co/datasets/bigcode/bigcodebench)；[代码及本地执行说明](https://github.com/bigcode-project/bigcodebench)。全量 1,140 题；`complete_prompt` 与 `instruct_prompt` 是两种输入设置，`canonical_solution/test/entry_point` 是答案及判分资源。仓库在本次核查时已归档，需固定数据 revision 和执行环境。
- 难度证据：官方结果中 **Qwen2.5-7B-Instruct Complete 46.1%**，对应失败率 **53.9%**；Instruct 37.6%，失败率 62.4%，略超目标。可选的 **Qwen2.5-Coder-7B-Instruct** 为 Complete 48.8%、Instruct 40.4%，但它必须作为单独被测模型登记，不可冒充通用 Qwen2.5-7B。[官方结果数据，第二页](https://huggingface.co/datasets/bigcode/bigcodebench-results/viewer/default/train?p=1)
- 接入建议：先用 Full/Complete，完整输入预算 ≤2,048 tokens；不要首轮就选 Hard。只向模型提供对应 prompt，评测端执行隐藏测试；库/函数不存在、伪造参数与普通算法错误分别标注。缺库、环境错误记 `blocked`，不能当模型失败。代码在隔离环境中执行；复用官方依赖和 evaluator，不用 yes/no LLM judge 替代测试。

**C. LiveCodeBench：Qwen3.5-4B 的编程候选，也可扩展两轮修复。**

- 下载：[官方 `livecodebench/code_generation_lite`](https://huggingface.co/datasets/livecodebench/code_generation_lite)；[官方 runner / self-repair](https://github.com/LiveCodeBench/LiveCodeBench)。仓库说明 `release_v6` 是累计到 2025-04 的 1,055 题；需要同时冻结版本、题目日期窗口和 ID 清单，不能只写“v6”或用会变化的 `release_latest`。
- 难度证据：Qwen3.5-4B 的官方 **LiveCodeBench v6 为 55.8%**，分数补数 **44.2%**；9B 为 65.6%，已更容易。模型卡未在该表列出全部日期窗口/生成细节，因此不声称累计 1,055 题直接复现该数。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B#benchmark-results)
- 接入建议：按题目自带难度、日期分层，先保留完整题干/公开例子 ≤2,048 tokens 的题；`lite` 是裁减测试用例，不是缩短 prompt。先测一次生成，再可做“生成→公开测试反馈→一次修复”的单独两轮变体，隐藏测试始终只在最终评估端使用。官方数据带加载脚本，按固定 runner 的依赖加载或审阅后读取原始 JSONL，不盲目依赖新版 `datasets` 自动加载。下载体积与上下文长度是两回事。

**D. BFCL v4 中的短工具调用子集：agentic 首选。**

- 下载/执行优先用 [Gorilla 官方仓库](https://github.com/ShishirPatil/gorilla)，其中 `berkeley-function-call-leaderboard/` 含数据和 checker；[当前 README](https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/README.md)、[类别清单](https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/TEST_CATEGORIES.md)。[官方 HF 镜像](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) 的说明仍停留在 V3，不应凭镜像名称认定拿到了 V4；其 `.json` 文件按行存 JSON，官方明确不建议直接用 `load_dataset`。
- 难度证据：Qwen3.5-4B 的 **BFCL-V4 总分 50.3**，9B 为 66.1，只支持“值得先导测试”的判断；**没有证据证明短子集或幻觉子集也有 49.7% 错误率**。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B#benchmark-results)
- 接入建议：先 `multiple/parallel/parallel_multiple/irrelevance`，再 `multi_turn_base/multi_turn_miss_func/multi_turn_miss_param`；排除 `multi_turn_long_context`，首轮不接需要外部搜索的 agentic 类别。工具 schema 加全部可见历史以 ≤4,096 tokens 为上限，优先 ≤2,048 的样本。使用官方 AST/state checker，并分开记录不存在工具、捏造参数、该澄清却猜测、错误执行顺序和解析失败。缺函数/参数时正确澄清或拒绝调用不是幻觉。锁定同一 commit 的数据、函数文档、模拟环境和答案，不能只下载题目 JSON。

**E. MultiChallenge：短多轮对话候选，须分开“幻觉”与“指令失败”。**

- 下载：[官方 `ScaleAI/MultiChallenge`](https://huggingface.co/datasets/ScaleAI/MultiChallenge)；[字段/加载说明](https://huggingface.co/datasets/ScaleAI/MultiChallenge/blob/main/README.md)；[官方评测介绍](https://labs.scale.com/leaderboard/multichallenge)。当前公开 HF 版本为 **266** 条 `test` 数据，不能沿用其他版本的 273 条计数。
- 难度证据：Qwen3.5-4B/9B 官方分数 **49.0 / 54.5**，相应未通过比例参考值 **51.0% / 45.5%**。短历史子集、当前公开版本与官方原测试的等价性尚未核实。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B#benchmark-results)
- 接入建议：优先 `INFERENCE_MEMORY/SELF_COHERENCE`；过滤 `conversation` 总消息数 ≤7 且完整输入 ≤2,048 tokens，保留原历史，不截断成“失忆题”。在已有历史末尾生成当前模型自己的目标回复；这不是该模型自生成的完整对话轨迹。兼容 `conversation` 的消息列表或平行 `role/content` 数组表示；`target_question/pass_criteria` 是 **judge 的评分问题和标准**，不得作为用户新问题/可见答案注入模型。纯格式/字数约束违规另报 `instruction_failure`，不作为事实幻觉；短子集数量不足时只做探索性结果。

#### 9.4.2 第二批：更贴近事实幻觉或极短输入，但需先导实测

**F. PopQA：用实体热度调整难度的短事实 QA。** [作者数据 `akariasai/PopQA`](https://huggingface.co/datasets/akariasai/PopQA) 约 14k 条，包含 `question/possible_answers/s_pop/prop/subj_id` 等信息。用问题作输入、完整别名集合作答案；`possible_answers` 若存为 JSON 字符串需先解析。优先完整输入 ≤512 tokens，按 `s_pop` 分位数和关系类型分层：小模型从热门实体开始，强模型增加长尾比例。**本轮未找到可直接移用到目标设置的 40%–60% 官方结果**；热度分层是校准手段，不是已验证结论。按实体分组划分，避免同一实体跨 dev/test；拒答和别名漏收需单独审核。

**G. TruthfulQA-generation：诱发常见错误信念的短回答。** [作者仓库及 `TruthfulQA.csv`](https://github.com/sylinrl/TruthfulQA)；[原论文](https://arxiv.org/abs/2109.07958)。原版为 817 题，后续修订删除/修改了部分题，服务器须记录实际 revision 和行数。只给问题，要求 1–2 句；参考正确/错误答案及来源交给评测端。优先 ≤512 tokens，按主题校准。选择 **generation**，不能用 MC2 概率质量分数的补数声称幻觉率；正确纠正错误前提与“不知道”分开记。当前 Qwen 的生成式幻觉率未在本轮核实，且存在旧题污染/时效问题，作为补充而非唯一主集。

**H. CRUXEval-I/O：极短代码推理对照。** [作者数据 `cruxeval-org/cruxeval`](https://huggingface.co/datasets/cruxeval-org/cruxeval)；[官方仓库与 `data/cruxeval.jsonl`](https://github.com/facebookresearch/cruxeval)。800 个短 Python 函数，HF 展示的代码字段长度为 30–278 字符，适合 ≤512-token 输入预算；I 为反推输入、O 为预测输出，分开评估。字段 `id/code/input/output`，隐藏目标字段；用官方执行语义判分，特别是 I 任务不能只做字符串匹配，因为有效输入可能不唯一。目标 Qwen 的 40%–60% 失败率待测；这是代码推理错误对照，不是完整程序生成或 agentic benchmark。

**扩展候选 ToolHop，暂不列首批。** [作者数据/工具实现](https://huggingface.co/datasets/bytedance-research/ToolHop)、[论文](https://arxiv.org/abs/2501.02506)提供 995 个多跳查询和 3,912 个关联工具。原论文最强 GPT-4o 的准确率也仅 49.04%，不能据此推断小 Qwen 能达到 50%。只有 BFCL 短子集不够时，再尝试按工具数/依赖深度筛选并实际测量上下文；工具文档、执行代码和答案必须一起下载，不能把它当普通 QA 文件。

#### 9.4.3 下载与接入顺序

服务器先下载 **SuperGPQA、BigCodeBench、BFCL**：分别覆盖短 QA、函数级程序、短工具调用；再补 **LiveCodeBench、MultiChallenge**，最后用 **PopQA/TruthfulQA/CRUXEval** 调整难度和扩充事实幻觉/短输入覆盖。对 Qwen2.5-7B 优先 BigCodeBench-Complete；对 Qwen3.5-4B 优先前述有分数依据的四类；Qwen3-4B/8B 与更小尺寸均需单独 pilot，不能照搬相邻型号。

- HF 条目按给出的完整 `repo_id` 下载，下载时冻结 commit SHA；普通 Parquet/JSON 数据可离线转换，BFCL 用其专用行式 JSON，LiveCodeBench 用固定版本加载器。不要把 HF 的 `main` 当永久版本号。
- 原始数据、参考解、隐藏测试、工具模拟器分目录保存到 `config/` 指定的资源路径，均不入本项目 git；下载/转换入口放 `scripts/setup/`，日志进 `log/setup/`。本节没有创建这些脚本，也不表示现有 CLI 已支持这些名字。
- 每份 manifest 保存 `source_url/repo_id/revision/license/sha256/source_split/source_id`；同时保存过滤规则、过滤前后数量、实际 tokenizer revision、输入长度 p50/p95/max 和丢弃原因。保留官方 LICENSE；上游代码许可不能自动替代题目数据许可。
- 当前 `types.Example` / `render_prompt` 主要是单轮 question/context 模式。BFCL/MultiChallenge 接入前需新增保留 `messages/tools/tool_call_id/trajectory_id/step_id/visible_history_cutoff` 的消息与轨迹适配；不得把所有角色拼成一段普通用户文本后声称复现原多轮设置。新增代码执行 evaluator 与工具 checker，不把隐藏测试或执行结果提前给 RePPL 检测器。

### 9.5 将自然错误/幻觉比例校准到 40%–60% 的实施协议

**长度预算是本项目提出的筛选条件，不是对数据集平均 token 长度的实测声明。** 按每个被测模型的实际 chat template 渲染后计数，包括 system、工具 schema、历史与已有代码。建议闭卷事实 QA/CRUXEval ≤512，选项 QA ≤1,024，代码 ≤2,048，多轮优先 ≤2,048、上限 4,096 tokens；跨模型主比较取满足所有模型预算的共同题目集合。超长题直接标记并排除，不删除关键条件/选项/历史来凑短。

1. **先锁模型与推理模式。** 记录完整 model ID、revision、dtype/量化、工具解析器、`enable_thinking`、采样及输出预算。当前 `render_prompt` 对名字含 qwen3 的模型会关闭 thinking；官方分数不能当作这个模式的复现结果。greedy 目标答案、用于 RePPL 的 K 个采样和官方 pass@k 是不同对象。不能靠提高温度、极短输出上限或故意破坏工具模板制造约 50% 错误。
2. **先分数据，再选难度。** 在每个候选集按原问题/实体/完整轨迹分组，用固定 seed 划分不重叠 calibration 与 test；大数据集每个“模型×模式×任务”先约 200 条 calibration，MultiChallenge 等小集合使用较小 pilot 并披露不确定性。只根据 calibration 调整公开难度字段、实体热度或类别混合；不得看测试集回答后保留恰好一半错题。跨模型主表保留共同测试集；模型专属难度集另表。
3. **保留双标签与无效状态。** `y_task_error` 来自选项 gold、执行测试或官方任务 checker；`y_hallucination` 针对错误事实、与可见证据矛盾、捏造工具/API/参数或虚构执行成功单独标注。仅测试不通过、格式错误、指令遗忘不能自动设 `y_hallucination=1`；未审核则为 `null/unverified`。judge 只能给辅助标签，使用 gold/工具状态/版本化 API 文档及人工抽查复核。拒答另记 `abstained`，运行失败/超时/输出截断另记状态，禁止混成幻觉。
4. **报告两个分母。** 同时报告全部有效样本上的幻觉比例，以及作出可判定断言样本中的条件幻觉率；拒答率、未判定率、截断率与覆盖率一起报告。调到 40%–60% 的具体指标先在配置中写明。若需要短输出，可先试 QA 128、工具调用 256、代码 1,024、多轮最终回答 512 个生成 tokens；输出截断偏多时先调整预算/题型，不能把截断计为“自然幻觉”。thinking 模式需单独预算及报告思考 tokens。
5. **冻结后验收。** calibration 上目标比例 0.4–0.6，附 Wilson 95% CI；200 个独立样本、比例约 0.5 时区间粗略为 0.43–0.57，不能据此保证 test 同比例。冻结题型混合、筛选 manifest、prompt、标签规约再一次性跑 test。实际超出区间就如实报告，不在 test 上反复调比例。低幻觉率任务仍可报告检测结果，不为平衡类别而注入错误。
6. **agentic 的统计单位另报。** 调用前、返回解读、最终回答按 §9.3 分开；轨迹中至少一次幻觉的比例与逐步幻觉率分别报告。先 pilot 最多 4 个工具调用/assistant 决策步骤的短任务，触及预算记 `budget_exceeded` 而非完成失败或幻觉；额外采样仅用于检测，不能让不同候选调用相互污染工具状态。bootstrap 按完整轨迹分组，避免把相关步骤当独立样本。

验收产物：每个模型/模式/任务分别交付原始及保留样本数、输入/输出长度、自然任务错误率、实际幻觉率、拒答/无效比例、标注一致性与置信区间。**本轮交付的是已核查的下载入口和实验计划，不是已经达成 40%–60% 幻觉率的实验结果。**

## 10. 工程目录与任务拆分

以下均为拟建文件，不表示当前已存在：

```text
reppl2/
  types.py                 # Example / Generation / HiddenReplay / Detection
  backends/base.py
  backends/vllm_backend.py
  backends/sglang_backend.py
  sampling.py
  scoring.py               # 实际 token logprob、Outer、位置映射
  segmentation.py
  interventions.py
  methods/common.py        # CV、Inv、risk、有效性
  methods/reppl_a.py
  methods/reppl_b.py
  judges/base.py
  judges/local.py
  judges/systemone.py      # Jev 与 StartLux 共用协议，provider 分开
  judges/llm_api.py
  judges/registry.py
  judges/pipeline.py       # 任意双路、独立并行/级联
  judges/fusion.py
  baselines/registry.py
  baselines/probability.py
  baselines/sese.py
  baselines/had.py
  baselines/dscore.py
  baselines/semantic_energy.py
  baselines/reference_import.py  # RAUQ / LAFaCT / LaaB 原版结果
  baselines/agent.py
  data/adapters/
  evaluation/metrics.py
  evaluation/explanations.py
  evaluation/costs.py
  cache.py
  cli.py
configs/reppl2/
prompts/judge/
tests/reppl2/
runs/reppl2/<run_id>/
```

`Detection` 至少保存 `sample_id, method, risk, inner, outer, validity, input_scores, output_scores, provenance, cost`。缺失分数使用 null 和状态，不使用 NaN 混入排名。整个 run 保存配置、环境、模型 revision、baseline 注册表、样本清单和输出哈希。

拟提供 CLI：`verify-backend → prepare → generate → detect → judge → evaluate → report`。每一步应能断点恢复，并在配置或模型变化时拒绝误用缓存。命令名称只是工程目标，当前不能当作已有可运行命令。

## 11. 里程碑、依赖与验收门槛

### M0：接口与数据底座

实现一种主后端，再通过统一接口适配另一种；锁定原生支持的目标模型。完成 token IDs/状态/logprobs 验收、一个数据适配器、结果 schema 与缓存。现有验收程序在真实 GPU 上运行通过后才进入大规模计算。

验收：覆盖完整输入及答案 token、无静默框架回退、评分无 off-by-one、冷/暖缓存行为可解释，失败样本有明确原因。若目标模型不支持，选择已验证的原生模型继续，不修改研究主张掩盖失败。

### M1：A 与最小对照闭环

实现共享数学、A、Outer-only、输出几何对照与评估器。统一生成一次采样池供可复用方法使用。完成小规模自然错误实验。

验收：数学固定数组检查通过；K<2 和零向量不误判；分数方向统一；输出 A、mu、r、Inner、Outer、risk 可追溯。此阶段能回答 A 是否有超出 Outer 与输出多样性的初步增量。

### M2：LLM judge 与 P0 baseline

实现 API LLM、Jev、StartLux-Decision-4B、约 20B 本地 LLM 四类 provider、任选两个的配置流水线、提示版本、结构化输出验证、标签 likelihood、弱标签隔离；接入 SeSE、HAD 与 D-Score，并启动其原版/适配版审计。Jev provider 可先用离线 fixture 测试解析，真实 API 未运行时清楚标记。

验收：同一数据可得到直接 judge 与片段 judge 结果；未授权证据/未来工具结果不进入 prompt；API/解析失败被统计；不使用 judge 自身结果作为唯一真值。P0 方法产出同口径质量—成本报告。六种配对均通过路由与融合合同测试；StartLux+约 20B 的本地组合完成真实服务验收，远程组合记录真实 API 或 blocked 状态。单路、双路、级联、降级结果必须可区分。

### M3：B 与独立解释验证

实现编辑器、固定答案重放、B、原始 q 消融及 RepplAB 提名版本。先完成小规模作用量与波动分析，再扩大。

验收：输出 IDs 配对严格一致；合法结构验证通过；解释实验包含独立编辑、随机/无关控制及实际重新生成；报告 B 的覆盖率、额外成本和相对 A 增量。

### M4：新增强基线与多轮

完成 RAUQ、LAFaCT、LaaB、Semantic Energy 的适用任务复现或明确阻塞审计；工具数据适配与步骤评估完成。无法复现的方法记录阻塞项，不伪造结果。

验收：原版与改版名称不同，训练预算可比较；完整轨迹隔离划分；首次错误定位及后续修复可核查；本地部署组与外部 API 组分开报告。

### M5：冻结方案与正式实验

冻结 prompt、超参、分片/干预策略、基线版本和统计计划后运行测试集。完成不同 K、任务和生成模型的实验及成本曲线。只有发现实质故障才能修改后重跑，并记录原因。

交付：可安装包、锁定环境、配置示例、数据转换说明、端到端命令、原始检测结果、评估报告、解释案例和已知限制。若结果不支持 H1/H2，报告失败，不以挑选数据集或修改标签维持正面结论。

## 12. 测试重点与最终 Definition of Done

需要有实际意义的测试：

- 数学：CV 轴、ddof、符号、归一化与历史逻辑对照；不是仅测试函数原样返回。
- 对齐：变长输出、编辑后变长输入、EOS、工具特殊 token、截断和空回答。
- 后端：真实 GPU 的实际 token 概率与完整状态；fixture 不能替代真实接口通过。
- 隔离：缓存 key 冲突、split 泄漏、未来证据泄漏、重复随机样本。
- Judge：格式失败、拒答、无依据引用、候选标签顺序以及被评文本包含指令的情况；六种配对的配置切换、概率映射、缺失分数、两票冲突、单路超时、级联触发和缓存隔离。
- 工程：断点恢复后样本不丢失、不重复；不同方法评估同一 ID 集或明确报告缺失交集及全量覆盖率。

最终完成标准：

- [ ] A/B 都真实通过至少一个 vLLM/SGLang 后端完成端到端运行；另一个后端报告独立验收状态。
- [ ] 所有新方法基础模型前向均满足原生推理约束，核心路径不依赖 Jev 或外部 API。
- [ ] StartLux-Decision-4B 纳入单模型评测；四类 provider 任意选两个无需修改业务代码，保存单路及融合证据，实际运行状态可核查。
- [ ] LLM-as-judge、P0 baseline 及适用的 P1 强基线有实际结果或明确阻塞证据。
- [ ] 两种异质不确定性分别输出、分别消融；解释区分作用与不稳定性。
- [ ] 单轮与多轮结果遵守相同的标签、证据及划分规则。
- [ ] 质量、泛化、解释、成本一起交付；没有未经实测的 SOTA 或免费加速主张。

本轮最值得优先实现的是 **A 验证低成本输入关联波动的增量，B 验证干预作用波动是否带来更强检测与解释，再用独立 judge 和语义不确定性强基线检验这种增量是否成立**。

---

# RePPL 2.0 工程实施计划（exphalu2，2026-10-04 定稿）

> 本节是上面研究规格的第一期编码计划与实施记录。原研究规格中的路径指向前一台机器（`/Users/tim/...`），本仓库以本机实际布局为准。每项实现状态在交付时如实标注：`done / blocked / planned`，没有占位假实现。

## P0. 环境事实与本轮决策（2026-10-04 实测）

- 本机：2× RTX 4090（24GB），conda `tim` 环境（Python 3.11.14，torch 2.9.1，vLLM 0.15.1 起步、本轮后台升级到 0.30.0）。
- **推理引擎为主 + 保留旧端口**（用户决策）：默认后端为 vLLM 原生；同时保留旧 exphalu 的 Transformers 前向实现作为 legacy 兼容端口，用于引擎尚不支持的架构（如 vLLM 0.15.1 不含 `Qwen3_5ForConditionalGeneration`）。SGLang 本轮未安装，后端接口保留、状态 `blocked`。
- **`/mnt/data`（/dev/sda，ext4）存在坏道**：`dd` 与 mmap 均在 `Qwen3-4B/model-00001-of-00003.safetensors` 处报 I/O error / SIGBUS。该模型在本机不可用；`trivia_qa` 数据与多数其他模型读取正常。`Qwen3.5-4B` 改用 `/home/tim/Proj/resource/Qwen3.5-4B` 的完好副本。
- 引擎版本与模型支持矩阵（实测）：
  - vLLM 0.15.1：无 `Qwen3_5*` 原生注册；支持 `token_embed` pooling 任务（可取逐 token 末层状态）。
  - Qwen3.5-4B 验证路径：升级后的 vLLM（后台升级 0.30.0 中）或 legacy Transformers 端口；两者都跑不通时按 blocked 记录，不用假结果补位。
- 数据现状：旧外部硬盘数据集目录未挂载；本机可用数据为 `/mnt/data/trivia_qa`（parquet）与程序内合成 smoke 数据。SQuAD/CoQA 适配器实现但依赖未挂载原始文件，状态 `blocked`（配置中留路径位）。
- `/mnt/data/deberta-v2-xlarge-mnli` 存在（语义熵蕴含器候选）；启动时做可读性自检，不可读时回退 lexical 分组变体并显式命名，不冒充原版。

## P1. 目录与产物合同（用户硬性要求）

```text
exphalu2/
  reppl2/            # Python 包（唯一源码包）
  config/            # 全部可配置项：模型路径、采样参数、数据路径、key 的环境变量名
  scripts/           # 唯一命令入口；按实验类别分子目录，禁止在仓库根目录直接跑命令
    setup/ smoke/ generation_eval/ perf/ interp/ vis/
  log/               # 所有日志；子目录与 scripts 一一对应（git 忽略）
  results/           # 所有运行产物；子目录与 scripts 一一对应（git 忽略，smoke 小文件例外）
    <类别>/<run_id>/ # config_snapshot.json / generation.json / trajectory.json /
                     # detection.json / judge.json / eval.csv / eval_summary.json / interp/
  vis/               # 可视化输出 PNG（git 忽略，smoke 例外）
  tests/             # 固定数组数学验证与 fixture（可入 git，属 smoke 范畴）
```

类别集合：`smoke`（冒烟/验收）、`generation_eval`（生成与评测主实验）、`perf`（性能）、`interp`（可解释性）。未来新实验只增子目录，不改合同。`.gitignore` 排除 `log/ results/ vis/` 的产物与一切模型权重；`results/smoke/` 中的小型 JSON/CSV 与 `tests/fixtures` 允许入库，用于回归对照。

## P2. 分期交付与验收

### S0：底座（本轮完成）
- `reppl2.types`：`Example / Generation / ReplayResult / Detection / JudgeResult`，缺失值用 `null`+状态码，不用 NaN 进表。
- `reppl2.backends`：`InferenceBackend` 合同（capabilities/sample/score/replay_last_hidden）；`vllm_backend`（原生，runner=generate 采样与 token logprob，runner=pooling convert=embed 的 `token_embed` 逐 token 末层状态）；`transformers_backend`（legacy 端口，批量重放 + `output_hidden_states`）；`sglang_backend`（接口就位，导入失败显式 blocked）。
- `verify-backend` 命令：真实 GPU 验收 token IDs/逐 token logprob/逐 token 末层状态/位置对齐，产出 JSON 报告；不做 fixture 冒充。
- 数学层 `methods/common.py`：§3.2 聚合（ddof=0、tau/alpha/epsilon 版本化）、CV、Inv，`tests/test_math.py` 固定数组验证（softmax 轴、CV 轴、分片重排不变性）。

### S1：A + 对照闭环（本轮完成）
- `generate`：greedy y0 + K 采样（默认 K=5，smoke K=3），持久化实际 token IDs、文本与逐 token logprob；统一聊天模板，采样配置全部来自 config。
- `reppl_a.py`：§4 全流程（重放→u[j]/v[k,t]→cosine softmax→A[k,j]→共享聚合）；`segmentation.py` 输入单元（QA 句级、RAG 段落级），单元带 char/token span。
- baselines（同批采样复用）：`outer-perplexity`（greedy 平均 NLL）、`lnpe`（采样长度归一熵）、`eigenscore-last`（末层池化协方差谱，显式命名变体）、`semantic-entropy`（deberta 蕴含或 lexical 回退变体）、`length`（输出长度对照）。
- 分数方向合同：所有方法输出"越大越可能幻觉"，固定方向不翻转，方向定义写入方法注册表。
- `evaluate`：AUROC/AUPRC/正例率，样本级配对 bootstrap 95% CI，输出 CSV + JSON 摘要。

### S2：LLM judge（本轮完成，本地路径）
- `judges/local.py`：self-judge 与独立 judge 两种；硬 yes/no 判决 + 固定标签 likelihood（Yes/No 首 token 概率归一）双分数；prompt 版本号入 config；输出按 §7.2 简化 schema。
- `JudgePair` 双 judge 聚合（mean_score、abstain 策略）接口就位；API LLM / Jev / StartLux provider 因无 key/服务，状态 `blocked`，配置中仅登记环境变量名，不运行。

### S3：B + 解释（本轮完成受控版）
- `interventions.py`：确定性编辑器（RAG 段落屏蔽 / QA 片段中和 `[MASKED]`），保存原文、替换、span、类型；无法合法编辑的单元记 skipped 并报覆盖率。
- `reppl_b.py`：§5 全流程（K×(1+J) 重放、q[k,j]、归一化 A 与原始 q 双保存、固定输出 IDs 配对校验）；MVP 约束 J≤4、K≤5，超限样本按配置截断并记录。
- `interp` 命令：导出 mu/r/p_hat 输入解释、逐 token NLL 输出解释、B 编辑影响表到 `interp/`。

### S4：性能与可视化（本轮完成基础版）
- `perf` 命令：采样/评分/重放分阶段耗时、token 数、峰值显存（p50/p95），输出 perf CSV。
- `scripts/vis/plot_eval.py`：从 eval CSV/summary 生成 ROC 与方法对比图到 `vis/<类别>/`。

### S5：明确不做与阻塞清单（如实记录，无占位）
- SGLang 后端：未安装 → `blocked`。
- Jev / StartLux-Decision-4B / 远程 API LLM judge：无 key、无本地服务 → `blocked`（registry 与配置位已留）。
- SeSE / HAD / D-Score / RAUQ / LAFaCT / LaaB / Semantic Energy（§8 P0/P1）：本轮未实现 → `planned`；本轮 baselines 仅为概率/熵族基础对照，不冒充强基线。
- 多轮工具调用轨迹：数据适配器未接 → `planned`。
- Qwen3-4B：`/mnt/data` 坏道 shard → 本机 `blocked`；如需该模型须换盘或重新下载。
- SQuAD / CoQA：原始 JSON 在未挂载盘 → `blocked`（适配器就绪，配置填路径即启用）。
- Qwen3.5-4B（引擎原生 / legacy 端口）：vLLM 0.15.1 注册表与 Transformers 4.57.6 均不识别 `qwen3_5` → 升级 vLLM 0.30.0（torch 2.13.0 + transformers 5.18.0）安装中，完成后按 README 验证记录表复验。

## P4. 交付记录（2026-10-04 收尾）

- 实测通过项与产物路径见 README"已完成的验证记录"表；两次冒烟的小产物入库（results/smoke/），其余产物按合同留在本地（log/results/vis 均 gitignore）。
- vLLM 双 runner 分时加载有三次实机教训，已固化为合同：切换前必须 `release_for_replay()`；引擎核心进程退出有延迟，加载失败含 "Free memory/Engine core initialization failed" 时等待重试（60s×3）；perf 类混合负载必须按阶段分批，禁止逐样本 gen↔pool 乒乓。
- vLLM 同一 SamplingParams seed 下 n>1 会产出完全相同样本：K 个采样改为 n 次独立调用、seed 递增（保可复现且互异）。
- 已知方法学限制：J=1 时跨单元 softmax 退化（见 README 已知问题 7）；eigenscore-last 方向观察见问题 8。
- 评分方向合同、缺失值 null+状态码、注册表状态机（planned/implemented/blocked）已在 `reppl2/baselines/registry.py` 与 `tests/` 固定数组测试中固化。

## P3. 运行与验收协议

- 唯一入口：`scripts/<类别>/*.sh`；每个脚本自动建 `log/<类别>/`、`results/<类别>/<run_id>/`，配置快照随 run 保存，断点恢复按阶段文件存在性跳过。
- 冒烟链：`verify-backend → unit-tests → smoke（合成数据 A/B/baseline/judge/eval 全链，Qwen3.5-4B + 小模型双后端）→ generation_eval（trivia_qa 小样本真实链）`。全部通过后允许更大规模。
- 验收门槛沿用 §11 M0–M1：真实 GPU 信号验收 JSON；固定数组数学检查；K<2、全零作用、非有限状态返回显式状态码；`eval.csv` 可由 `vis` 消费。
- vLLM 0.30.0 升级为后台进行：若升级后 API 兼容性破坏现有后端，回退 0.15.1 + legacy 端口完成本轮验证，升级问题如实写入已知问题。

## P5. 第二轮实施记录（2026-10-05）

### P5.1 vLLM 0.30.0 升级复验（原 blocked 项解除）

- 环境：vLLM 0.30.0 + torch 2.13.0 + transformers 5.18.0。Qwen3.5-4B 在 vLLM 原生与 legacy Transformers 端口**双路验收通过**（verify-backend logprob 对齐 max diff 0.0000；token_embed 末层状态 (T,2560)；全链冒烟 OK）。
- 两处 0.30.0 适配（`reppl2/backends/vllm_backend.py`）：
  1. flashinfer 采样 kernel 需 nvcc JIT，本机无 CUDA toolkit → 引擎采样时崩溃。经 `backend_cfg.env.VLLM_USE_FLASHINFER_SAMPLER="0"`（config 驱动，setdefault 注入）回退原生 torch 采样。
  2. pooling task 必须建引擎时声明：`LLM(pooler_config=PoolerConfig(task="token_embed"))`；运行时 `pooling_task=` 切换被 0.30.0 拒绝。

### P5.2 RepPPL-A 对齐 bug 修复（实质正确性修复）

- **症状**：Qwen3.5-4B triviaqa 16 样本中 14 个 reppl-a invalid（"no input unit could be aligned"），2 个"成功"样本实为 J=1 退化。
- **根因一**：后端 `replay_last_hidden` 只返回输出 token 状态，reppl-a 的 u[j]（输入单元表示，DESIGN §4.1 步骤 3）需要 greedy 重放的 **prompt 侧**状态。两后端现均返回 `hidden_context`（prompt 侧），`hidden`（输出侧）合同不变。
- **根因二**：输入单元的 char span 相对 question/system 原文，却被直接对渲染后 prompt_text 做 token 对齐（聊天模板前缀使 span 整体错位）。`segmentation._prompt_span` 现先用单元文本在 prompt_text 中精确定位，失败才回退原 span 并记 `span_source=source_relative_unverified`；`char_start_in_prompt/char_end_in_prompt/span_source` 全部入 interp 记录。reppl-b 的编辑合同（源相对 span + find 校验）不受影响。
- 此前 Qwen2.5-0.5B 上 reppl-a AUROC=1.0 的历史记录实际由 ε·Outer 驱动（J=1 退化），不是 Inner 增量——如实修正认知。
- 新增 `tests/test_alignment.py`（mock tokenizer 固定数组：span 翻译、token 落点、上下文状态池化、无对齐单元拒绝）。

### P5.3 P0 强基线：d-score-last 与 sese（§8.1 清单内）

- **`d-score-last`**（D-Score，arXiv:2607.24586v1，2026-07）：σ₁/σᵢ≤τ 的奇异方向计数，Gram 特征值精确实现，未中心化，τ=10 入 config；**原版用最优层（通常非末层）**，vLLM 原生路径只有末层 → 显式命名 `d-score-last` 变体，原版配置保持 `planned`。
- **`sese`**（SeSE，UAI 2026，arXiv:2511.16275；官方仓库 SELGroup/SeSE @8d4c6c5）：编码树结构熵逐位移植（10 个种子图与官方实现误差 <1e-12）；语义图按官方协议（deberta-v2-xlarge-mnli 蕴含概率 0.65 + 句向量余弦 0.35 混合 → 聚类阈值 0.3 → 簇内蕴含边 → PageRank/Kruskal 连通性修复）。**偏离项显式记录**：官方 GPT-4o 答案增强因无 key 跳过（`answer_enhancement: "none:blocked-no-api"` 入产物）；句向量模型经 hf-mirror.com 下载（huggingface.co 本网络不可达）。
- 两者均接入 detect 管线 + 注册表 `implemented` + 固定数组测试（`tests/test_p0_baselines.py`，含方向合同与退化拒绝）。triviaqa 16 样本验收：sese AUROC 0.75/AUPRC 0.68，d-score-last AUROC 0.16（小样本方向存疑，如实记录待复核）。

### P5.4 其余

- 单元测试 27→40 项。results/smoke 入库两个新冒烟 run 作回归对照。
- S5 剩余 planned：HAD、RAUQ、LAFaCT、LaaB、Semantic Energy、D-Score 原版层配置、多轮轨迹适配器、SQuAD/CoQA 数据挂载。
