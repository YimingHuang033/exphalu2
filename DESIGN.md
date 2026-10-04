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

## P3. 运行与验收协议

- 唯一入口：`scripts/<类别>/*.sh`；每个脚本自动建 `log/<类别>/`、`results/<类别>/<run_id>/`，配置快照随 run 保存，断点恢复按阶段文件存在性跳过。
- 冒烟链：`verify-backend → unit-tests → smoke（合成数据 A/B/baseline/judge/eval 全链，Qwen3.5-4B + 小模型双后端）→ generation_eval（trivia_qa 小样本真实链）`。全部通过后允许更大规模。
- 验收门槛沿用 §11 M0–M1：真实 GPU 信号验收 JSON；固定数组数学检查；K<2、全零作用、非有限状态返回显式状态码；`eval.csv` 可由 `vis` 消费。
- vLLM 0.30.0 升级为后台进行：若升级后 API 兼容性破坏现有后端，回退 0.15.1 + legacy 端口完成本轮验证，升级问题如实写入已知问题。
