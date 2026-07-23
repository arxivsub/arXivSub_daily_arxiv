# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-23 | 今日论文总数: 464

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Leveraging Offline Supervision for Efficient and Generalizable Reinforcement Learning in Large-Scale Vision-Language-Action Models

**arXiv ID:** 2607.19399 | [PDF](https://arxiv.org/pdf/2607.19399v1)

**作者:** Dmitriy Poyarkov `[一作]` (AXXX), Aleksandr I. Panov `[通讯]` (MIRAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究将离线监督与在线强化学习相结合，针对大规模视觉-语言-动作模型（OpenVLA）通过LoRA适配器进行强化学习微调。

**💡 创新点**

提出两种简单的离线引导PPO变体（参考策略KL正则化和离线行为克隆），并证明离线指导可在不牺牲OOD性能的前提下显著提升采样效率。

**🔧 技术方法**

使用PPO+辅助损失（KL或BC）与LoRA优化、预训练冻结的OpenVLA基模型、β系数的学习率日程安排，并在RL4VLA基准上进行评估。

**📊 数据集**

采用RL4VLA环境中的模拟任务数据集，包括预训练的OpenVLA-warmup、2k或16k专家演示数据，用于SFT、PPO和离线引导方法的训练与评估。

**📈 对比分析**

通过与标准PPO和SFT初始化的PPO在1M步与2M步的比较，发现离线引导方法在1M步即可达到或超过2M步PPO的IND/OOD成功率，约减少一半环境交互量。

**⚠️ 局限性**

实验仅覆盖少量随机种子，局限于单一模型与基准，尚未验证在其他VLA架构、任务或奖励设定下的普适性。

---

## 2. STN-TGAT: Top-K Portfolio Construction via Prior-Guided Graph Attention with Learnable Soft-Threshold Sparsification

**arXiv ID:** 2607.19385 | [PDF](https://arxiv.org/pdf/2607.19385v1)

**作者:** Haoran Guo `[一作]` (University College London), Li Zhang `[通讯]` (University College London)

**通讯引用:** 34685 | [OpenAlex ID](https://openalex.org/A5100404566)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种融合Transformer与图注意力的STN‑TGAT模型，用于每日Top‑K股票排序与实盘交易组合构建。

**💡 创新点**

创新点在于：①使用基于Normalized Mutual Information的非线性先验图并通过可学习的软阈值门控实现自适应稀疏化；②将Transformer时间编码与GAT空间编码联合，捕捉长时序与动态跨股关系；③设计top‑weighted ListNet与MSE混合的决策对齐损失，使模型直接优化最终投资收益。

**🔧 技术方法**

主要技术包括Transformer自编码器、Graph Attention Network、NMI先验图、可学习软阈值门控稀疏化、ListNet加权损失、回归MSE、日频交易回测与交易成本调整。

**📊 数据集**

使用2022‑01‑03至2024‑12‑30期间的美国S&P 500最大50只成分股日常OHLCV数据，共752个交易日，构造滚动窗口特征。

**📈 对比分析**

与ARIMA、GRU、LSTM、GRU‑GCN、GRU‑GAT等基线在MRR、RBO、IRR、Sharpe、MDD等指标上比较。STN‑TGAT在IRR 18.07%、Sharpe 2.99、MDD 6.34%等指标上领先；排名指标虽略逊于单纯LSTM，但在收益与风险调整后表现最佳。

**⚠️ 局限性**

局限性包括：仅在Top‑5、S&P 500 50只股票的日频场景验证，跨市场与更短频率的适用性待检验；稀疏阈值门控仍需手动调节超参数；模型对交易成本和滑点敏感，需更稳健的交易策略；缺乏对图结构与权重的可解释性分析。

---

## 3. Simulating Eutopia: Revisiting Long-term Fairness with Outcomes, Performativity, and Dynamics

**arXiv ID:** 2607.19389 | [PDF](https://arxiv.org/pdf/2607.19389v1)

**作者:** Vedant Palit `[一作]` (Indian Institute of Technology Kharagpur), Debabrota Basu `[通讯]` (University of Lille)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究 AI 驱动的决策者（如信用贷放贷系统）在长期交互中的公平性问题，并提出基于表现性马尔可夫决策过程（PeMDP）的框架

**💡 创新点**

① 将人群的申请行为建模为 Hawkes 过程，显式捕捉贷款决策对未来申请率的反馈；② 提出 PeMDP 以描述政策如何导致动态转移；③ 开发 LOANLY 数据生成器和完整的贷款模拟器，为表现性公平性研究提供可复现的实验平台；④ 对比传统 RL、表现性 RL 与静态策略，系统评估不同公平目标与效用函数对长期效率与公平性的影响

**🔧 技术方法**

表现性强化学习（PERL）和传统政策梯度（PG）；Hawkes 过程生成模型；多目标公平性奖励设计（Outcome、DM、Two‑sided、Rawlsian、Fairness Lagrangian）；平均奖励最优策略；模拟环境实现连续时间步长、财富动态和贷款结果

**📊 数据集**

利用成人（Adult）基准数据生成的合成贷款数据，包含两类群体（red/blue），可配置初始财富、违约率、利率、表现性强度等参数

**📈 对比分析**

通过在模拟器中运行不同策略（PG、PERL）与不同公平目标（四类效用函数），记录累计利润、社会福利、财富差距和不平等比率；实验表明：PERL 在 Outcome‑fairness + Fairness Lagrangian（FL）下能在保证利润的同时实现最低财富差距和最接近 1 的不平等比率；但在两侧公平（Two‑sided）设定下，传统 PG 在利润和社会福利上更占优势，说明最优配置取决于公平目标和学习算法

**⚠️ 局限性**

① 依赖于合成数据，缺乏对真实金融市场的验证；② 模拟器中仅考虑了两类群体，未涵盖更复杂的多群体或多属性情形；③ 只评估了有限的公平目标和效用函数，未探索更细粒度的政策组合；④ 模型假设与实际贷方行为（如信息不对称、监管约束）可能存在差异；⑤ 只关注平均奖励，未深入分析收敛速度与策略稳定性。

---

## 4. Logic-Guided Data Extraction with Answer Set Programming and Large Language Models

**arXiv ID:** 2607.19365 | [PDF](https://arxiv.org/pdf/2607.19365v1)

**作者:** Mario Alviano `[一作]` (University of Calabria), Fabrizio Lo Scudo `[通讯]` (University of Calabria)

**通讯引用:** 161 | [OpenAlex ID](https://openalex.org/A5069354072)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种逻辑引导的语义数据抽取框架，将大型语言模型（LLM）的候选事实提取与Answer Set Programming（ASP）推理交错进行，以控制抽取过程。

**💡 创新点**

创新点在于利用ASP的非单调逻辑作为“守护”条件动态决定是否对某个谓词发起抽取请求，从而在保证最终抽取结果与传统基线一致的前提下显著减少LLM调用次数并抑制伪造事实。

**🔧 技术方法**

使用的技术包括：LLM抽取器（Meta Llama 3.1）、ASP推理器（clingo 5.8）、守护条件的缓存与递增数据库构造、以及基于守护真值与单调性的缓存策略。

**📊 数据集**

实验使用了两类基准：基于层图的Graph（G）数据集和包含四个ASP领域的Logic Puzzle（LNRS）数据集，均为从现有ASP任务改造得到的文本。

**📈 对比分析**

通过对比传统LLMASP基线（无守护）和逻辑引导框架，结果显示：①LLM调用次数下降约15–20%（G）至70%+（LNRS）；②F1分数与完美率与基线保持一致；③整体性能提升显著，尤其在多领域的LNRS基准中。

**⚠️ 局限性**

局限性包括：需要手动标注守护条件与其单调性；缓存策略对非单调守护的有效性有限；在极简逻辑结构中收益有限；并未将LLM与ASP完全融合为概率性神经符号框架。

---

## 5. Stateful Guardrails for Multi-Turn LLM Systems: A Conversational Risk Accumulation Framework

**arXiv ID:** 2607.19361 | [PDF](https://arxiv.org/pdf/2607.19361v1)

**作者:** Sanjay Mishra `[一作]`, Ganesh R. Naik `[通讯]` (Flinders University)

**通讯引用:** 27176 | [OpenAlex ID](https://openalex.org/A5029523168)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对多轮对话的安全防护框架——Conversational Risk Accumulation（CRA）机制，能够在会话层面捕捉语义漂移、信息累积和同意趋势等三条轨迹信号，并给出可阈值化的风险评分。

**💡 创新点**

核心创新在于：① 将风险视为随时间累积的轨迹问题而非单独交互；② 设计了三种轨迹级信号（S₁：语义漂移；S₂：信息累积图；S₃：合规性梯度）并通过可解释的凸融合与学习型 CRA‑Net DA 进行融合；③ 构建了专门评测 CRA 的同长度、多家族、可迁移的基准 CRA‑Bench v0.1、v0.2 与 5 家族扩展。

**🔧 技术方法**

技术手段包括：句子嵌入（语义漂移）；spaCy + 规则 NER + 权重映射构建信息累积图；线性回归滑动窗口估计同意梯度；可解释凸融合；GRU‑based 轨迹学习器；长度与家族对抗正则（GRL、DANN、CORAL）；温度缩放、Bootstrap 置信区间；以及决策证书生成。

**📊 数据集**

使用的数据集：CRA‑Bench（1,200/2,000+ 生成多轮会话，3-5 家族；包含正负对照和同长度同主题样本）；Human‑CRA‑Transfer（750 CoSafe 渐进式攻击 + 222 ShareGPT 三轮对话）作为真实迁移评测；CoSafe（1,800）用于诊断；公开的 LLM 评测基准 HarmBench、HarmBench‑LLM、Judge‑LLM、Llama Guard、Qwen3Guard 等作为对比基线。

**📈 对比分析**

对比方法：① 传统 turn‑level 过滤器；② 凸融合、特征 MLP；③ 轨迹学习 CRA‑Net 与 CRA‑Net DA；④ 对抗性训练（GRL、DANN、CORAL）与无监督融合；⑤ 现有完整会话级守护模型（Llama Guard、Qwen3Guard）。性能：CRA‑Net DA 在 CRA‑Bench v0.2（5 家族）上取得 AUROC≈0.919、sFPR≈0.175、TPR=1.0（在 1% benign FPR 预算下）；Human‑CRA‑Transfer 上 AUROC≈0.929；在 Qwen3Guard 上仅略低（≈0.997），但 CRA‑Net DA 具备毫秒级推理、可解释子信号和无第三方 API 的优势。

**⚠️ 局限性**

局限性：① 需要标注的多轮数据来训练，缺乏跨家族泛化（LOFO 结果低于 0.5）；② 对 NER 依赖，若实体抽取失误会影响 S₂；③ 对长会话的内存/计算消耗（IAG 边界控制必要）；④ 仍可能被模板迁移或对抗式分解攻击利用；⑤ 现有评测多为合成，迁移到真实业务场景仍需进一步验证。

---

## 6. Native Multi-Dimensional Subquadratic Operators via Input Dependent Long Convolutions

**arXiv ID:** 2607.19378 | [PDF](https://arxiv.org/pdf/2607.19378v1)

**作者:** David R. Wessels `[一作]` (University of Amsterdam), Saee Gopal Paliwal `[通讯]` (NVIDIA)

**通讯引用:** 706 | [OpenAlex ID](https://openalex.org/A5079457088)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种多维自适应子线性卷积算子 HyenaND，并实现了高效的 FFT 级联 CUDA 库。

**💡 创新点**

通过注册标记与 FiLM 结合的 SIREN 生成全局可输入依赖的多维卷积核，恢复了输入选择性并保持了单 FFT 路径。

**🔧 技术方法**

利用隐式 N‑D SIREN、FiLM 条件化、全局卷积、FFT‑卷积、单 GPU 上的共享内存融合核以及 LTI 设计。

**📊 数据集**

在 OpenGenome2 基因组、ImageNet‑1K 视觉、五种 2D/3D PDE 仿真数据集以及 PanTS 3D 医学分割数据集上进行实验。

**📈 对比分析**

与 Transformer、Mamba 等基准对比，纯 HyenaND 在 ImageNet 81.5% top‑1、基因组低 perplexity、PDE 最高 VRMSE、医学分割 Dice 与注意力相当且显著降低显存和 FLOPs，混合模型进一步超越两者。

**⚠️ 局限性**

在 3D 位置记忆任务中仍低于 Mamba，且对新分辨率的初始化需要手动调节；极大规模数据仍受限于 GPU 内存与 FFT 计算瓶颈。

---

## 7. Beyond Tracking or Shortcut: Composition-Bounded Predictive States in Poker Autoregressive Models

**arXiv ID:** 2607.19369 | [PDF](https://arxiv.org/pdf/2607.19369v1)

**作者:** Quanhao Li `[一作]` (University of Massachusetts), Qianyu Chen `[通讯]` (Abbey Park High School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究不含范围信息的Limit Hold'em自回归模型，探讨隐藏状态是否真正跟踪对手范围以及其可解释性。

**💡 创新点**

提出三种解释框架（artifact、composition‑bounded predictive support（CBPS）、posterior tracking），并将CBPS作为新的中间解释；验证该框架能区分隐藏状态与可见公共信息的混淆。

**🔧 技术方法**

使用线性探测器、控制实验（标签置乱、先验、历史、行动/价值、全/匹配公共组合）以及匹配组合测试，结合已知控制（真实后验、纯组合、CBPS控制）来评估隐藏状态。

**📊 数据集**

在无范围的Limit Hold'em数据集上训练四层Transformer自回归模型，使用50k教师策略手牌进行训练、验证与测试；对手范围标签仅用于后期探测。

**📈 对比分析**

与基线（无观测先验、行动/价值基线、组合基线、行动/价值+组合基线）比较，行为头在行动预测上提升约5个百分点；隐藏状态在未控制下对对手范围可恢复性提升约1.5%（top‑10），但在完整组合控制下此提升消失，表明真正的后验追踪未通过。

**⚠️ 局限性**

局限性包括：仅在单一模型与三颗随机种子上验证；未对更强的探测器或更复杂的干预进行系统测试；结果仅为相关性，未证明因果中介；CBPS框架仍需在其他不确定信息任务中进一步验证。

---

## 8. Cross-Subject Semantic Decoding with Shared-Space Alignment for Generalized Neural Representation Learning

**arXiv ID:** 2607.19394 | [PDF](https://arxiv.org/pdf/2607.19394v1)

**作者:** Ji-Hoon Heo `[一作]` (Korea University), Seong-Whan Lee `[通讯]` (Korea University)

**通讯引用:** 22693 | [OpenAlex ID](https://openalex.org/A5011014617)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一种基于共享响应模型的跨受试者语义解码框架，将多受试者脑电响应映射到共享潜在空间，并通过CNN将其解码为上下文语义嵌入。

**💡 创新点**

创新点在于将SRM用于解码任务而非传统编码，直接在共享空间中学习神经-语义映射，并通过时间窗滑动保持时序信息。

**🔧 技术方法**

采用高伽马功率提取、共享响应模型(SRM)、PCA/HA对齐、CNN解码器及GPT‑2上下文嵌入。

**📊 数据集**

使用Zada等公开的ECoG听觉语料库（9名受试者，约5300词的播客文本）。

**📈 对比分析**

与PCA和PCA+HA基线对比，SRM在源受试者和保留受试者上均获得最高AUC‑ROC、对比准确率和Top‑k准确率，跨受试者性能下降最小且无统计学显著差异。

**⚠️ 局限性**

局限在于仅验证于自然听觉任务，可能对想象语音等更具挑战性场景的泛化能力不足；且受试者数量有限，需更多数据验证。

---

## 9. From Trajectories to Prefixes: Reusing Teacher Trajectories via Replayed Prefixes and Online Continuation

**arXiv ID:** 2607.19395 | [PDF](https://arxiv.org/pdf/2607.19395v1)

**作者:** Yihan Wang `[一作]` (Tianjin University), Hongke Zhao `[通讯]` (Tianjin University)

**通讯引用:** 2279 | [OpenAlex ID](https://openalex.org/A5017692278)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Prefix‑GRPO，利用教师轨迹拆分为可重放前缀与在线续延，实现小型语言模型的强化学习优化。

**💡 创新点**

创新点在于把教师前缀作为可直接优化的历史 token，统一纳入 GRPO 剪辑比率目标，并通过 prompt‑space old‑logprob 对齐，使前缀重放与继续学习同时进行。

**🔧 技术方法**

使用技术包括：GRPO/PPO 框架、entropy‑change 选取的组查询前缀构造、SFT checkpoint 旧策略估计、剪辑比例目标、前缀‑持续学习的联合优化。

**📊 数据集**

实验数据集涵盖 TextCraft、BabyAI 与 ALFWorld 三个长周期交互环境，教师轨迹由 MiniMax‑M2.1 生成。

**📈 对比分析**

与 SFT、GRPO、GRPO‑MIS、DAPO、Replay‑GRPO 等基线对比，Prefix‑GRPO 在所有环境中实现了最高或接近最高的平均成功率和多样本成功率，显著优于仅重放前缀或不对前缀进行优化的方案。

**⚠️ 局限性**

局限性包括需要可重放验证的教师前缀；旧策略估计仍需系统化研究；实验仅为单跑结果，未覆盖更多环境；在极长或复杂环境中前缀选取与优化仍有待完善。

---

## 10. NEXUS: Structured Runtime Safety for Tool-Using LLM Agents

**arXiv ID:** 2607.19356 | [PDF](https://arxiv.org/pdf/2607.19356v1)

**作者:** Elias Hossain `[一作]` (University of Central Florida), Niloofar Yousefi `[通讯]` (University of Central Florida)

**通讯引用:** 383 | [OpenAlex ID](https://openalex.org/A5054474613)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于结构化计划的运行时安全监控器（Neural EXecution Utility and Safety, NEUS），在工具调用前对计划进行四类干预（允许、阻止、请求确认、请求修订）

**💡 创新点**

创新点在于结合九条确定性安全规则、细粒度参数检查以及经过Platt校准的风险评分器，通过“风险门控降级”策略实现四类干预的细粒度决策，并显著提升干预准确率（+27.3pp）

**🔧 技术方法**

采用结构化计划中间表示（包含工具名、参数、权限、侧效类别等），规则引擎、参数检查器、9维特征的逻辑回归风险评分器以及统一的干预策略Π

**📊 数据集**

使用了128实例的合成基准、200实例的间接提示注入（IPI）集、R‑Judge（564实例）和AgentHarm（352实例）等公开数据集，另外还评估了Stress等对抗基准

**📈 对比分析**

与仅规则或仅学习模型相比，NEUS在合成集上的二分类F1≈0.949、四类干预准确率≈0.641；在R‑Judge上F1≈0.861（提升≈0.012）；在Stress上F1≈0.881，速度比GPT‑4o快≈4,800×，延迟0.205 ms

**⚠️ 局限性**

局限性包括对中等严重度（规则盲区）干预的不足、对提示级意图的检测依赖有限、未在真实多轮交互中验证、以及对更大工具注册表和权限编码的适配性需要进一步扩展

---

## 11. Euclean: Automated Geometry Problem Formalization with Unified Verification in Lean

**arXiv ID:** 2607.19374 | [PDF](https://arxiv.org/pdf/2607.19374v1)

**作者:** Linbin Tang `[一作]` (Tsinghua University), Fan Yang `[通讯]` (Microsoft Research)

**通讯引用:** 6007 | [OpenAlex ID](https://openalex.org/A5045464812)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Euclean框架，实现将自然语言几何题自动转化为Lean中可验证的正式化表达；

**💡 创新点**

四阶段自动化流程（约束阐明、配置锚定、映射与迭代修复）以及构建最大规模的Lean几何数据集，首次实现统一的几何与其他领域的证明基础；

**🔧 技术方法**

基于大语言模型（DeepSeek-V3、Goedel v2等）与Lean4的编译反馈循环；

**📊 数据集**

OMNI‑Geometry（768题）和Numina‑Geometry（177,597题）两个大规模几何正式化数据集；

**📈 对比分析**

在人工评测中TOP1≈48.9%、TOP5≈73.3%；在Goedel v2上无几何专门训练时证明成功率从13.6%提升至15.1%，证明了数据集的实用价值；

**⚠️ 局限性**

仍缺乏完整的语义验证，可能漏掉必要的约束或过度加入非必要假设；受限于Lean当前几何库的表达力；完全基于文本，无法处理图形输入；

---

## 12. Air Quality Arena: A Large-Scale Multi-Region Ground Monitoring Dataset and Benchmark for Air Quality Forecasting with Time-Series Foundation Models

**arXiv ID:** 2607.19381 | [PDF](https://arxiv.org/pdf/2607.19381v1)

**作者:** Rishi Bharadwaj `[一作]` (BITS Pilani), Pandarasamy Arjunan `[通讯]` (Indian Institute of Science)

**通讯引用:** 990 | [OpenAlex ID](https://openalex.org/A5040213611)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并发布了一个跨国、多污染物的空气质量监测数据集与基准AQA（Air Quality Arena），用于评估时序基础模型在实际空气质量预测中的表现。

**💡 创新点**

创新点在于：①构建规模最大、地理覆盖最广的多国多污染物时序数据集；②在零样本（zero‑shot）设置下系统对比11种时序基础模型与传统基线；③发现跨模态视觉基础模型VisionTS++在零样本预测中领先。

**🔧 技术方法**

主要技术包括：多源原始测量数据统一采集与预处理（季节分解插值）；时序基础模型（Chronos、Moirai、TimesFM等）与传统统计/机器学习基线（AutoETS、DeepAR、LightGBM等）的训练与评估；使用TIME框架进行滚动窗口评估与归一化指标。

**📊 数据集**

数据集为AQA-Data，涵盖七个国家（美国、印度、中国、英国、法国、德国、墨西哥）共6种主要污染物（PM₂.₅、PM₁₀、NO₂、SO₂、CO、O₃），超过14,000条站点-污染物时序，时间跨度为2022‑2025年三年。

**📈 对比分析**

比较方法采用滚动窗口、MASE和CRPS指标，归一化到季节性朴素模型；结果显示时序基础模型整体优于传统基线，VisionTS++排名第一，平均MASE约0.78；传统基线如DLinear、LightGBM平均MASE>0.9。

**⚠️ 局限性**

局限性包括：零样本评估未涉及模型微调或少样本适配；仅使用单变量无协变量；部分污染物（如SO₂、O₃）预测仍相对困难；数据覆盖虽然广泛，但仍可能存在源头不完整或区域特定偏差。

---

## 13. FormulaSPIN: Self-Play Fine-Tuning for Natural Language to Spreadsheet Formula Generation

**arXiv ID:** 2607.19354 | [PDF](https://arxiv.org/pdf/2607.19354v1)

**作者:** Cy Xie `[一作]` `[通讯]`, Cy Xie

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于自我对弈的公式生成微调框架 FormulaSPIN，使模型在不需要额外人工标注或外部教师模型的情况下，利用公式可执行性进行自我提升。

**💡 创新点**

创新点在于将执行结果作为内在监督，将非匹配公式按语义错误与风格差异分层处理，设计可自适应的语义‑至‑风格课程与执行投票机制，解决传统 SPIN 在公式生成中出现的梯度冲突问题。

**🔧 技术方法**

采用自我对弈（SPIN）策略、执行反馈过滤、可自适应权重 β_med、执行投票（ExecVote）以及 LoRA、RMSProp 等训练技术，并以 LLaMA‑3.1‑8B‑Instruct 为基础模型。

**📊 数据集**

使用公开基准数据集 NL2Formula‑70K（70k 例）和 Sheetpedia‑Selected（2.2k 例）进行训练与评估。

**📈 对比分析**

与基线 SFT、SFT‑DPO、专业 API 模型（如 Gemini‑2.5‑Pro、DeepSeek‑R1）比较，FormulaSPIN 在 NL2Formula‑70K 上实现 74.9% EM、87.1% EA，优于所有基线且与使用 GPT‑4.1 标注的 SFT‑DPO 匹敌，同时在 Sheetpedia‑Selected 上亦表现出显著提升。

**⚠️ 局限性**

局限包括需依赖可执行的电子表格引擎、推理时需额外计算量导致延迟、训练需要多轮迭代且资源消耗较大，且目前仅在英文与 Excel 公式上验证，其他语言与软件环境的泛化仍待探索。

---

## 14. Stochastic Primal-Dual Decoding for Multiobjective Generative Recommender Systems

**arXiv ID:** 2607.19357 | [PDF](https://arxiv.org/pdf/2607.19357v1)

**作者:** Dmitrii Moor `[一作]` (Spotify), Mounia Lalmas `[通讯]` (Spotify)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在推理时对自回归生成推荐系统进行多目标Slate生成的轻量解码层。

**💡 创新点**

创新点在于把多目标问题视为在线约束优化，利用随机原始-对偶（拉格朗日乘子）动态更新，实时平衡相关性与辅助目标，无需改动模型训练。

**🔧 技术方法**

技术包括自回归Transformer生成模型、在线约束优化、随机原始-对偶算法、拉格朗日乘子更新与理论约束满足/调度保证。

**📊 数据集**

使用的公开数据集为：大规模音乐流媒体播放列表（≈40B交互）、首页推荐Shelves（≈5.3B印象）以及Amazon Sports & Outdoors电商子集（≈5M交互）。

**📈 对比分析**

与EPR、WA、MB-UCB/TS等解码基线比较，离线实验在三大域中均实现Pareto改进，在线A/B实验提升约5.44%内容份额且未降低用户消耗。

**⚠️ 局限性**

局限性包括：对相关性随步进变化的平稳性假设、仅处理单一辅助约束、以及对超参数η的敏感性，需要仔细调优。

---

## 15. Decodable but Not Detectable: A Leakage Fingerprint for Near-OOD Benchmarks

**arXiv ID:** 2607.19393 | [PDF](https://arxiv.org/pdf/2607.19393v1)

**作者:** Vishnu Bindu Balachandran `[一作]` `[通讯]` (Independent AI Researcher), Vishnu Bindu Balachandran (Independent AI Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过审计一个基于扰动的近似OOD检测器，发现并纠正了近-OOD基准构造中的泄漏错误，并提出了一种无需重训练即可检测泄漏的指纹。

**💡 创新点**

创新点在于：①揭示了近-OOD拆分中将已训练类误设为“OOD”导致的检验反向；②设计并验证了一种“可解但不可检测”的泄漏指纹（高监督可分辨率与低无监督检测分数）；③在多模型多数据集上进行大规模验证，证明指纹在实际基准中的高敏感性与特异性。

**🔧 技术方法**

主要技术包括：一类高斯（Mahalanobis）检测、kNN/Isolation Forest等无监督基线；扰动响应签名（注意头掩蔽）；Ledoit–Wolf 估计的协方差收缩；线性判别分析（LDA）作为监督读者；以及对特征协方差谱的白化分析。

**📊 数据集**

使用的数据集包括：CIFAR‑10、CIFAR‑100、SVHN、DTD、MNIST、Fashion‑MNIST、STL‑10、20 Newsgroups、Tobacco‑3482 等；模型骨干为 ResNet‑50、ViT‑B/16、RoBERTa、LayoutLMv3 等。

**📈 对比分析**

与传统无监督检测器（Mahalanobis、kNN、Isolation Forest）对比，扰动响应检测器在所有清洁设置下均未显著提升性能；而泄漏指纹在 52 个设定中敏感度 90%（18/20），特异性 97%（31/32），在 24 个公开基准中仅误报一次（真正的硬难对），表明其高准确性。

**⚠️ 局限性**

局限性：指纹依赖监督读者且阈值可能在极硬的清洁近-OOD场景下误报；仅在 4 个模型/2 个数据集的 2×2 组合上验证，未覆盖更大规模或不同任务的泛化；在扰动响应签名空间中，指纹的判定不如嵌入空间精确，需要额外重训练；最后，论文未对 ImageNet 级别的近-OOD 进行全面调研。

---

## 16. LAARA: Layer-Aware Adaptive Rank Allocation for Parameter-Efficient Fine-Tuning

**arXiv ID:** 2607.19391 | [PDF](https://arxiv.org/pdf/2607.19391v1)

**作者:** Ashutosh Tripathi `[一作]` (Independent Researcher), Sriparna Saha `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 8308 | [OpenAlex ID](https://openalex.org/A5060797340)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LAARA，一种基于 Fisher 信息的层级自适应 LoRA 权重分配方法，兼具搜索无依赖和训练期间动态调整。

**💡 创新点**

创新点在于利用轻量级对角 Fisher 近似来评估各层重要性，结合投影归一化、对数压缩、混合重要性估计和投票抑制机制，实现无搜索、无超参调优的层级 rank 分配。

**🔧 技术方法**

技术包括：LoRA 动态低秩参数化、对角 Fisher 信息估计、指数滑动平均与偏差校正、投影归一化与对数压缩、权重混合评分、整数化 rank 分配与投票抑制。

**📊 数据集**

使用 DeBERTa‑v3‑base、Llama‑3.2‑3B 以及 GLUE（8个任务）和 MathInstruct 作为评测数据集。

**📈 对比分析**

与 BitFit、LoRA、DyLoRA、AdaLoRA 等基线在相同参数预算下对比，LAARA 在 GLUE 任务上普遍超越或匹配最佳基线，特别是在推理密集任务如 RTE 上提升 6% 以上；在 MathInstruct 上也取得更高准确率与更低参数量。

**⚠️ 局限性**

局限性：对角 Fisher 近似忽略了交叉项，可能低估某些层的重要性；关键超参数（如投票窗口、混合系数）未针对不同任务做细粒度调优，缺乏更广泛的多架构验证。

---

## 17. Validating the Single Item Kawaii Measure

**arXiv ID:** 2607.19352 | [PDF](https://arxiv.org/pdf/2607.19352v1)

**作者:** Katie Seaborn `[一作]` (University of Cambridge), Yijia Wang `[通讯]` (Tokyo Institute of Technology)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5057872214)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

验证了单项可爱感知（kawaii）量表在声音与视觉刺激上的可靠性与效度。

**💡 创新点**

首次在多模态（声音、外观、结合）九个数据集中系统评估单项量表的多维效度，并提供基线信度与效度证据。

**🔧 技术方法**

使用Cronbach α进行内部一致性检验，并用Kendall τ_b 评估相关性和效度，分析不同刺激类型与人群的一致性。

**📊 数据集**

采用来自日本Yahoo! Crowdsourcing平台的九个公开或授权数据集，包含967名独立参与者，涵盖语音助手、游戏角色语音、外观和语音+外观组合。

**📈 对比分析**

与多项可爱度条目进行比较得到强至极强相关（τ_b≥0.49），在构造效度上对可爱与不可爱语音区分显著；跨情境效度在声音与外观间仅弱相关（τ_b≈0.07），但在不同群体的声音比较中呈中等相关（τ_b≈0.20），总体表现良好。

**⚠️ 局限性**

局限性包括：样本量不足导致部分效度检验不完整；未进行重测信度、面效度与预测效度；缺少非日本样本和“不可爱”刺激；不同受试者对多刺激评分可能导致结果偏高。

---

## 18. Information Discernment in Large Language Models

**arXiv ID:** 2607.19355 | [PDF](https://arxiv.org/pdf/2607.19355v1)

**作者:** Joshua Ashkinaze `[一作]` (University of Michigan), Eric Gilbert `[通讯]` (University of Michigan)

**通讯引用:** 20816 | [OpenAlex ID](https://openalex.org/A5024795472)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了Learn2Discern基准，设计了源辨识与真相辨识两大信息辨识指标，并通过零-shot实验、提示干预与用户调查验证LLM在外部知识冲突下的表现。

**💡 创新点**

创新点在于提出了三条规范公理及可解释的Spearman相关度指标，用单一实验框架同时量化源可靠性与主张准确度对模型更新的影响，并公开了规模达2.8M的实验数据集与用户调查结果。

**🔧 技术方法**

主要技术包括零-shot推理实验、Chain‑of‑Thought、Reliability、Bayesian与Defense提示策略、Spearman相关度和多重统计检验。

**📊 数据集**

使用的数据集包括NumerSense、TriviaQA、General Social Survey、World Bank Development Indicators、TREC，以及构造的132个新闻来源和5种扰动因子，形成约2.8M个实验元组。

**📈 对比分析**

对比方法是将模型的源辨识、真相辨识等指标与随机机率及上限进行对照，13个模型在约670K次试验中大多表现接近随机，更新更受来源热度影响而非可靠性；规模和新版本提升了真相辨识但未改善源辨识。

**⚠️ 局限性**

局限性包括直接注入主张而非完整RAG检索、仅针对数值问题、使用单一可靠性度量、未检验检索完整流程、以及来源可靠性可能随主题变化而异等。

---

## 19. CrackedPDFs: A Controlled Benchmark for Hidden Prompt Injection in PDFs

**arXiv ID:** 2607.19396 | [PDF](https://arxiv.org/pdf/2607.19396v1)

**作者:** Pukaphol Thienpreecha `[一作]` `[通讯]` (University of California Berkeley), Pukaphol Thienpreecha (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了CrackedPDFs基准，研究PDF层隐藏提示注入，并评估文档结构感知的检测方法。

**💡 创新点**

创新点在于：① 引入paired benign‑confounder 对照，② 采用硬分裂保证训练/测试不共享同一文档，③ 设计 sanitized hybrid 检测器融合结构特征与文本特征，④ 进行 shortcut audit 与 label‑shuffle sanity check。

**🔧 技术方法**

使用的技术包括：结构特征提取（PDF 操作符统计、坐标熵等）、文本 TF‑IDF、逻辑回归、XGBoost、PromptGuard（文本分类基线）以及 hybrid 组合模型。

**📊 数据集**

数据集为 CrackedPDFs：总计 29,322 个 PDF，来自 4,983 个基础文档，其中 9,774 个为注入样本，19,548 个为 benign 或匹配的 confounder。

**📈 对比分析**

对比实验采用准确率、F1、ROC‑AUC、PR‑AUC 等指标；sanitized hybrid 在 held‑out 测试集上达到 0.973 准确率、0.960 F1、ROC‑AUC 0.998、PR‑AUC 0.997；PromptGuard 仅 0.390 F1；rule 基线 0.623 F1；结构仅模型在 0.502–0.651 F1 范围；TF‑IDF 取得完美指标但被判定为 shortcut 依赖。

**⚠️ 局限性**

局限性：仅评估合成 PDF，未涵盖真实世界多样性、OCR 或扫描文档、适应性攻击，以及跨家族的泛化能力。

---

## 20. When Does Consensus Beat Voting? A Critical Analysis of Statistical Label Fusion in Medical Image Segmentation

**arXiv ID:** 2607.19402 | [PDF](https://arxiv.org/pdf/2607.19402v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 21. Predictive single cell foundation model for gene regulation and aging with privacy-preserving tabular learning

**arXiv ID:** 2607.19400 | [PDF](https://arxiv.org/pdf/2607.19400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 22. Hybrid LSTM-Graph Neural Framework for Robust Financial Fraud Detection and Adversarial Resilience

**arXiv ID:** 2607.19350 | [PDF](https://arxiv.org/pdf/2607.19350v1)

**作者:** Mariam Zakaria Moussa Ali `[一作]` `[通讯]` (Arab Open University), Mariam Zakaria Moussa Ali (Arab Open University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 FraudShield AI，结合 LSTM 与图形拓扑特征来检测洗钱交易。

**💡 创新点**

创新点在于将手工图特征嵌入 LSTM，使用 Focal Loss 与动态阈值来解决极度类别不平衡与微交易躲避。

**🔧 技术方法**

采用 LSTM 序列建模、PageRank、入度、流比特征、Focal Loss、动态阈值调整等技术。

**📊 数据集**

使用 PaySim 模拟移动支付交易数据集。

**📈 对比分析**

与 Logistic Regression、XGBoost 基线对比，Hybrid 模型在精确率 0.94、召回率 0.93、F1 0.93 上显著优于其他模型。

**⚠️ 局限性**

局限包括依赖合成数据、图特征计算为离线、阈值参数需重新校准等。

---

## 23. Scale-Aware Learning of Chaotic Dynamics on Unstructured Meshes via Binned Spectral Losses

**arXiv ID:** 2607.19387 | [PDF](https://arxiv.org/pdf/2607.19387v1)

**作者:** Kanad Sen `[一作]` (Purdue University), Romit Maulik `[通讯]` (Purdue University)

**通讯引用:** 2695 | [OpenAlex ID](https://openalex.org/A5048243433)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在非结构化网格上提出了图谱频率分块能量匹配的损失，用于提高高维非线性动力学系统的自回归预测准确性。

**💡 创新点**

创新点在于将结构化网格的频率分块能量匹配迁移到图拉普拉斯频率上，并引入可扩展的 Chebyshev 滤波器和多层图谱能量对齐（GLEAM）两种层级化近似，兼顾精度与计算效率。

**🔧 技术方法**

采用图卷积网络、图拉普拉斯谱分析、Chebyshev 多项式滤波、低秩图谱嵌入以及基于图对比的损失等技术实现。

**📊 数据集**

在 EAGLE、反向壁面步（BFS）以及 DGN4CFD 三维翼面等 CFD 数据集上进行实验。

**📈 对比分析**

与仅使用均方误差（MSE）损失的基准相比，Chebyshev BSP 在 EAGLE 上的长时步 RMSE 下降 10–20%，频谱误差显著降低；在 BFS 上低频能量保持更好，导致回归误差下降；在翼面案例中压力分布和力误差分别下降 20–30%，性能提升在多种指标上均有体现。

**⚠️ 局限性**

主要限制包括：需要手动调节谱损失权重；Chebyshev 逼近对窗口锐度敏感，需选取合适的多项式阶数；GLEAM 只约束低频子空间，无法精确控制细尺度误差；多目标训练可能导致收敛不稳定，且对极端细尺度失真抑制有限。

---

## 24. AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally

**arXiv ID:** 2607.19363 | [PDF](https://arxiv.org/pdf/2607.19363v1)

**作者:** Shaowen Wang `[一作]` (Tsinghua University), Jian Li `[通讯]` (Tsinghua University)

**通讯引用:** 44295 | [OpenAlex ID](https://openalex.org/A5100402427)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出一种针对Transformer的旋转位置编码（RoPE）的改进方法，允许每个注意力头学习独立的旋转频率和长度相关的注意力缩放因子。

**💡 创新点**

创新点在于识别并解决了RoPE统一频率和缩放对不同功能头的不匹配问题，通过头级别的可学习频率和自适应缩放实现更充分的频谱利用和更精准的长度泛化。

**🔧 技术方法**

核心技术包括：头级可学习的二维频率张量（AdaRoPE）、基于长度的注意力温度自适应（AdaScale）、以及对预训练模型的无缝迁移和细调；实现细节保持与FlashAttention等现有硬件加速兼容。

**📊 数据集**

使用的主要数据集包括：FineWeb‑Edu‑100B（大规模文本预训练）、PG19（长文本微调）、以及多种NLU基准（ARC, BoolQ, HellaSwag, Lambada, PIQA, WinoGrande）和长上下文评测基准RULER。

**📈 对比分析**

与RoPE、Partial RoPE、ALiBi、NoPE等基线在七个NLU任务上平均提升约0.9分，预训练损失下降0.016；在长上下文推理中，AdaRoPE在64k长度下保持超过50%的检索准确率，且在低样本微调（仅100条样本）情况下明显优于传统方法，长期上下文持续预训练亦实现最高的RULER分数。

**⚠️ 局限性**

局限性包括：仍需要对不同模型规模进行超参数调优；对极端长上下文的理论分析尚不完整；在极大模型（>10B）中训练头级频率可能导致额外的计算开销，且对低资源环境的可迁移性仍需进一步验证。

---

## 25. Reliability-Aware Hard--Soft Physics-Informed Neural Networks for Robust Learning of Challenging Partial Differential Equations

**arXiv ID:** 2607.19377 | [PDF](https://arxiv.org/pdf/2607.19377v1)

**作者:** Duc Tien Nguyen `[一作]` (VinUniversity), Dinh Gia Ninh `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 1454 | [OpenAlex ID](https://openalex.org/A5079563148)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可靠性感知硬软物理信息神经网络（RA‑HSPINN），通过在硬软PINN中加入可学习的有界可靠性调制场，配合逆指数移动平均自适应损失平衡，实现对局部或多模解结构以及不可靠约束的更稳健建模。

**💡 创新点**

创新点：①在保持硬约束完整性的前提下，引入可学习的可靠性调制场对内部网络进行局部加权；②使用逆EMA自适应全局损失平衡，而非手工权重；③轻量化可靠性正则化，避免冗余调节；④在一阶Poisson系统中避免二阶乘积项导致的优化僵化。

**🔧 技术方法**

技术手段：神经网络表征、自动微分、硬软约束嵌入、可靠性调制网络、逆EMA损失加权、可靠性正则化、L‑BFGS微调、周期特征映射、全局和局部损失平衡。

**📊 数据集**

数据集/基准：制造解的六个偏微分方程基准—尖锐梯度非线性 Burgers、含噪声/不兼容初始条件 Burgers、平滑周期输运、尖峰前沿周期输运、混合 Dirichlet/Neumann Poisson、以及多模混合一阶 Poisson 系统。

**📈 对比分析**

比较方法与性能：与全软 PINN（SPINN）及固定硬软 PINN（HSPINN）三方对比；RA‑HSPINN 在六个基准上均优于 HSPINN，误差下降幅度从 29.4%（混合 Poisson）到 98.7%（尖锐梯度 Burgers）；在噪声初始条件下降幅 72.4%；Ablation 结果表明可靠性调制是主要提升来源，逆EMA 仅在部分情况带来帮助。

**⚠️ 局限性**

局限性：①额外的可靠性网络增加计算开销；②对表达能力已足够的硬软方案提升有限；③未验证在更高维、复杂几何或多种随机种子下的鲁棒性；④可靠性场本身无物理意义，需正则化控制；⑤在二阶 PDE 中若未采用一阶重构，乘积规则仍会带来优化阻塞。

---

## 26. SUM: Unified Geometric Surgery on Spatio-Temporal Adaptation Vectors for Federated Class Incremental Learning

**arXiv ID:** 2607.19384 | [PDF](https://arxiv.org/pdf/2607.19384v1)

**作者:** Jaeik Kim `[一作]` (Seoul National University), Jaeyoung Do `[通讯]` (Seoul National University)

**通讯引用:** 1033 | [OpenAlex ID](https://openalex.org/A5024989829)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SUM框架，通过在服务器端对客户端和任务的适配向量进行几何切割，以解决联邦分类增量学习中的空间-时间灾难性遗忘。

**💡 创新点**

将FCIL统一视为多任务学习，用适配向量的方向交互解释干扰，并在服务器端实现纯粹的空间与时间投影（几何切割），无需额外客户端计算或通信。

**🔧 技术方法**

适配向量投影/去耦（空间SUM、时间SUM）、Z-score阈值修剪、稀疏化、符号一致化、任务特定激活；基于FedAvg的联邦训练；使用ViT-B/16与T5-Small等模型。

**📊 数据集**

视觉域：CIFAR-100、ImageNet-R、ImageNet-A、Cars-196、CUB-200、EuroSAT；语言域：20-Newsgroups、CLINC-150；还在LoRA、ConvNeXt等不同架构上验证。

**📈 对比分析**

与GLFC、TARGET、MFCL、PILoRA、FOT、STAMP、LoRM及中心化联合训练等多种FL/CL/模型融合基线比较，SUM在所有数据集和任务拆分上实现最高Final Averaged Accuracy，提升幅度达22%，并在恶意客户端场景下表现更稳健。

**⚠️ 局限性**

需要服务器端维护并存储所有任务的适配向量，随着任务数增长空间复杂度呈线性；在极大任务数时需压缩或丢弃部分信息；对极端分布偏差和模型规模的适应性仍待进一步验证。

---

## 27. Recovering Clinical Utility Under Differential Privacy: Empirical Validation of Adaptive Federated Aggregation on Heterogeneous Cardiovascular Datasets

**arXiv ID:** 2607.19403 | [PDF](https://arxiv.org/pdf/2607.19403v1)

**作者:** Rodrigo Tertulino `[一作]` (Federal Institute of Education, Science and Technology of Rio Grande do Norte), Ricardo Almeida `[通讯]` (Federal Institute of Education, Science and Technology of Rio Grande do Norte)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在真实多中心心血管数据上验证FedCVR联邦学习框架的临床可行性。

**💡 创新点**

引入服务器端自适应Adam优化器作为时序去噪器，解决非IID与差分隐私双重挑战。

**🔧 技术方法**

使用FedCVR、FedAvg、FedProx、FedCluster、FedAdagrad、FedYogi等联邦优化算法，并结合DP‑SGD Gaussian机制。

**📊 数据集**

采用Framingham、Cleveland、Hungarian、Switzerland、Long Beach VA五个公开心血管数据库，统一到13属性UCI心脏病schema。

**📈 对比分析**

通过leave-one-institution-out交叉验证比较，FedCVR在保留σ=0.8（ε≈4.2）时取得F1≈79.2%、AUC=0.96，显著优于FedAvg，差距≈2.8%。

**⚠️ 局限性**

局限包括仅用已清洗公开数据、仅5个机构、未验证更大规模网络、未评估临床工作流与公平性。

---

## 28. Spectral-LSH: Sub-Quadratic Prompt Compression via Krylov-Projected Locality-Sensitive Hashing

**arXiv ID:** 2607.19368 | [PDF](https://arxiv.org/pdf/2607.19368v1)

**作者:** Ali Mahdavi `[一作]` (Islamic Azad University), Omid Kashefi `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的长提示压缩方法Spectral-LSH，能够在进入LLM前将冗余标记聚合为宏标记，减少预填充阶段的计算量。

**💡 创新点**

核心创新在于：①在隐式softmax注意力核上使用随机特征和Krylov子空间近似得到主特征向量；②在该注意力特征空间中应用SimHash进行令牌聚类；③结合位置编码聚合和自适应路由，实现低压缩下快速分块与高压缩下谱聚类的无缝切换。

**🔧 技术方法**

利用随机傅里叶特征（RFF）近似核矩阵、Lanczos Krylov方法求主特征、SimHash进行局部敏感哈希、宏标记构造与位置聚合，整个流程保持无训练、无额外模型。

**📊 数据集**

在C4和WikiText-103等大规模文本数据集上进行评估，针对不同压缩比例（2×~16×）测试多种模型（Mistral‑7B、Qwen2.5‑7B、Qwen2.5‑14B、SmolLM‑360M）。

**📈 对比分析**

与传统连续分块、全局LSH、局部LSH等基线对比，结果显示：在压缩比≤4×时分块方案最快且质量几乎相同；压缩比≥8×时Spectral‑LSH可显著提升PPL、KL、NLL等指标，尤其在Qwen系列模型上提升显著；自适应后端在高压缩下能在保持较低总延迟的同时恢复质量。

**⚠️ 局限性**

局限性包括：需访问模型嵌入层（不适用于API服务）；对小模型压缩效果差；在无压缩时需完全绕过聚合以避免扰动；自适应路由在极低延迟场景下仍落后于分块；未覆盖复杂下游任务（问答、摘要等）；未验证对位姿聚合的最佳方案。

---

## 29. LISA: Linear-Indexed Sparse Attention for Efficient Long-Context Reasoning

**arXiv ID:** 2607.19358 | [PDF](https://arxiv.org/pdf/2607.19358v1)

**作者:** Yu Zhao `[一作]` (Alibaba International Digital Commerce), Weihua Luo `[通讯]` (Alibaba International Digital Commerce)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LISA 模块，将线性注意力与稀疏注意力并行以加速长链式思维推理

**💡 创新点**

通过两阶段训练冷启动线性注意力并引入动态索引器，减少 O(n²) 为 O(nM)

**🔧 技术方法**

使用线性注意力、稀疏自注意力、门控融合、索引器、测试时训练与 KL 损失

**📊 数据集**

使用 DeepSeek-distilled-Qwen 与 OpenR1-Math-220K 训练集，评测 GSM8K、MATH-500、AMC23、AIME24、AIME25

**📈 对比分析**

与 LightThinker、INFTYTHINK、H2O 等基线对比，LISA 在 16K 上速度提升约 50%，准确率提升 5.6%

**⚠️ 局限性**

仅在数学推理任务验证；训练流程两阶段复杂，未对原模型做联合微调，保留部分原始自注意力层导致 FLOPs 仍高

---

## 30. Geometry-Guided Constraint Learning for LLM Safety Classification

**arXiv ID:** 2607.19366 | [PDF](https://arxiv.org/pdf/2607.19366v1)

**作者:** Fumiaki Uehara `[一作]`, Yuki Kobiyama `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出一种后置安全检测框架，利用稀疏自编码器（SAE）提取特征并在隐藏空间学习极少量（K=2）线性约束或锥形约束，从而实现高效的安全判断；

**💡 创新点**

核心创新在于：①SAE实现了对约束数量的自动化最优化，几乎消除每类手动调参需求；②引入了三阶段训练策略稳定锥形约束学习，并证明SAE初始化显著提升锥形约束性能；

**🔧 技术方法**

采用的技术包括稀疏自编码器、聚类初始化、三阶段锥形约束训练、以及与传统SaP多面体约束的对比；

**📊 数据集**

实验基于BeaverTails安全标签数据集，在Qwen3.5-9B和Qwen2-1.5B大模型上验证；

**📈 对比分析**

与传统随机初始化、全K范围搜索的SaP相比，SAE+锥形约束在大多数类别实现96–99%准确率，且推理延迟<1 ms；

**⚠️ 局限性**

局限性包括仅在英文数据集上评估，未检验对抗性鲁棒性或多语言迁移，且未对模型实用性（如MMLU）进行评估。

---

## 31. Benchmarking Confidential GPU Inference on NVIDIA H100 under Intel TDX

**arXiv ID:** 2607.19353 | [PDF](https://arxiv.org/pdf/2607.19353v1)

**作者:** Wei Wang `[一作]` (Mozilla), Burns Smith `[通讯]` (Mozilla)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在Intel TDX加密实例上运行NVIDIA H100 80GB GPU，对Mistral‑7B v0.1和Qwen3‑30B‑A3B两款大型语言模型进行加密与非加密模式的推理性能对比，给出了TTFT、请求延迟、token吞吐量、全局token吞吐量以及并发吞吐量等指标。

**💡 创新点**

创新点在于提供了针对单卡加密GPU推理的实测数据，区分了固定请求率与闭环并发两种负载模式，明确指出加密模式下吞吐量下降与饱和点提前的趋势，并给出针对容量规划的实用建议。

**🔧 技术方法**

使用Intel TDX加密虚拟机、NVIDIA H100 GPU、PyTorch/Transformers框架以及自制负载生成脚本。

**📊 数据集**

使用的“数据集”实际上是模型自身的推理任务，没有外部文本数据集；两种模型分别为7B参数与30B参数版本。

**📈 对比分析**

通过对比加密模式（CC）与非加密模式（Non‑CC）的平均/百分位指标，在固定请求率下，TTFT与请求延迟平均提升约21–27%，全局token吞吐量下降约18–21%；在闭环并发测试中，吞吐量差距保持在11.5–20.2%，但30B模型在加密模式下更早达到饱和。

**⚠️ 局限性**

局限性包括仅测试两种模型和单卡配置；未覆盖尾部延迟、低级硬件计数；安全设置（Secure Boot）被禁用；结果不易直接推广到多卡、不同CPU/内存、不同加密策略或其他推理框架。

---

## 32. HyGRL: Adaptive Hybrid Graph Reasoning for Multi-Entity Questions

**arXiv ID:** 2607.19398 | [PDF](https://arxiv.org/pdf/2607.19398v1)

**作者:** Junyi Wang `[一作]` (Beijing Institute of Technology), Junyi Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 171 | [OpenAlex ID](https://openalex.org/A5015400856)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 HyGRL，一个融合文本与知识图谱并利用强化学习引导 LLM 的多实体组合推理框架。

**💡 创新点**

创新点在于将文本块直接嵌入异构图，采用两阶段学习（模仿+LLM偏好 RL）实现轻量级自适应检索。

**🔧 技术方法**

使用异构图构建、基于 MLP 的策略网络、交叉编码器、PPR、Adamic–Adar、LLM偏好对比、Self‑Critical Sequence Training 等技术。

**📊 数据集**

在 2WikiMultiHopQA、HotpotQA、MuSiQue 三个多跳 QA 数据集上进行评测。

**📈 对比分析**

与多种文本 RAG、图 RAG 基线相比，HyGRL 在 EM/F1 上均取得最高 79.19%/79.85%（2Wiki）等，且检索 token 成本低、延迟可实时。

**⚠️ 局限性**

依赖 LLM 生成的偏好信号，若 LLM 失真或域外会导致性能下降；构建异构图仍需离线索引，规模受限。

---

## 33. Building Fast, Evaluating Slow: Pipeline Choices Dominate Autointerpretability Score Variance

**arXiv ID:** 2607.19386 | [PDF](https://arxiv.org/pdf/2607.19386v1)

**作者:** Sinie van der Ben `[一作]` (ETH Zürich), Mennatallah El-Assady `[通讯]` (ETH Zürich)

**通讯引用:** 2087 | [OpenAlex ID](https://openalex.org/A5020415668)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估自动解释性评估指标在稀疏自编码器（SAE）中的可靠性，量化方法论方差与架构方差对得分的影响。

**💡 创新点**

提出方差分解框架并发现方法论方差主导，提出稳定性检查与最小报告清单来规范跨论文比较。

**🔧 技术方法**

使用LLM生成解释、LLM评分、ICC、Top‑k Jaccard、REML方差分解等统计技术，对四种解释度量（simulation、detection、fuzzing、purity）进行实验。

**📊 数据集**

在Pythia‑160M和Apertus‑8B两大模型上训练SAE，并使用多种语料（训练集、Wikipedia、新闻、代码）以及随机抽样的数据集。

**📈 对比分析**

通过对四种度量、两种模型、四个方法论变量（语料、抽样、解释器、释义）进行大规模实验，发现方法论方差远高于架构方差；检测度量最稳定，模糊度量最不可靠，特征排名在不同条件下几乎无重叠。

**⚠️ 局限性**

局限包括仅固定评分模型和层级，样本量有限；未评估评分模型、层依赖等额外方差；实验设计部分交叉，方差估计具有一定不确定性。

---

## 34. The Orthogonalized Read Is a Removable Training Scaffold for Recurrent Memory

**arXiv ID:** 2607.19390 | [PDF](https://arxiv.org/pdf/2607.19390v1)

**作者:** Keston Aquino-Michaels `[一作]` `[通讯]` (No Way Labs), Keston Aquino-Michaels (No Way Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在一块78K参数的mLSTM上研究将记忆矩阵在读取时进行Newton–Schulz正交化的干预，并发现其提升并非内存容量增加，而是作为训练时的可移除支架；

**💡 创新点**

核心创新在于揭示正交化读取提升是对训练过程的重新调参（escape hazard 加速）而非容量提升，并提出可在训练后“熔断”这一支架的完整方法；

**🔧 技术方法**

使用Newton–Schulz迭代实现正交化、RLS（Sherman–Morrison）读取、梯度噪声与学习率热度分析、Kaplan–Meier生存曲线、离散时间危害模型等技术；

**📊 数据集**

采用合成记忆检索基准MAD noisy recall（key/value 对，80%噪声，序列长512，词表80或96）进行实验；

**📈 对比分析**

对比基线、正交化读取及多种消除实验（delta、随机键、straight‑through等），结果显示正交化读取可将逃逸风险提升约6.4倍，解决率在原始预算下从3/16提升至9/16，且可通过常数学习率和调度改动几乎复现全部提升；

**⚠️ 局限性**

局限性在于仅评估单一小规模mLSTM和合成任务，缺乏对多块网络、其他记忆任务或大规模语言模型的验证，种子数量有限且对某些消除实验的统计功效不足。

---

## 35. Profile-Graph Memory for LLM Agents: Implicit Cross-Entity Traversal through Narrative Profiles

**arXiv ID:** 2607.19359 | [PDF](https://arxiv.org/pdf/2607.19359v1)

**作者:** Shengtong Zhu `[一作]` `[通讯]`, Shengtong Zhu

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个包含1000条多跳会话记忆问题的基准以及一种两层记忆架构，能够在写入时一次性联合抽取叙事型概况与压缩残差；

**💡 创新点**

创新点在于通过子串匹配的隐式实体链遍历实现多跳推理，并以零额外API成本实现精度层的压缩残差抽取，二者在同一架构中实现功能专化；

**🔧 技术方法**

使用纯embedding检索（MiniLM）配合相关性门控的多跳扩展和单次LLM调用（GPT‑4o‑mini）完成概况与残差的共同抽取；

**📊 数据集**

基准数据集包括10个社交网络场景下的1000个多跳问题（K=1–5）以及公开的LoCoMo会话数据；

**📈 对比分析**

与Oracle、FullContext、Mem0、A‑Mem、HippoRAG和RAG等基线比较，Benchmark上实现80.1%准确率与FullContext持平，LoCoMo上达到78.4%（比FullContext高11.3个百分点），在多跳推理上显著优于传统方法；

**⚠️ 局限性**

局限性包括在开放域查询上的性能不足、残差未实现过期失效管理以及在真实用户数据部署时需要处理隐私与访问控制等安全问题。

---

## 36. Neural Operator Surrogates for Two-Dimensional Neutron Flux Estimation

**arXiv ID:** 2607.19388 | [PDF](https://arxiv.org/pdf/2607.19388v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 37. CruiseBench: A Real-Flight-Aligned N-CMAPSS Benchmark for Engine RUL Prediction

**arXiv ID:** 2607.19380 | [PDF](https://arxiv.org/pdf/2607.19380v1)

**作者:** Pu Cheng `[一作]`, Qiang Miao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了从N-CMAPSS衍生的巡航阶段RUL基准CruiseBench，包含可复现的巡航段掩码（Cruising-Period Mask）和一套固定的评估协议。

**💡 创新点**

创新点包括：① 设计了可复现的common-altitude方法生成巡航段掩码；② 在完整飞行记录中剔除爬升、下降等混合阶段，仅保留巡航阶段进行RUL预测，形成可控的子基准；③ 明确并系统评估了RUL cap、窗口下采样方式和飞行阶段选择等协议参数对模型性能的影响。

**🔧 技术方法**

使用的技术包括：序列模型基线（LSTM、GRU、TCN、TSMixer）、窗口采样与下采样、基于指数异常降解模型的RUL标签capping、Saxena不对称评分等评估指标。

**📊 数据集**

使用的数据集为N-CMAPSS的九个可访问子数据集：DS01-005、DS02-006、DS03-012、DS04、DS05、DS06、DS07、DS08a-009、DS08c-008。

**📈 对比分析**

在固定协议（窗口256、stride10、η5%）下对四个基线模型进行比较，TSMixer平均RMSE为3.46±1.71，Saxena得分为2.50×10⁴；TCN略逊；ablation实验显示飞行阶段选择、下采样方式和RUL cap会导致模型排名和稳定性变化；在全飞行数据下模型排名可逆。

**⚠️ 局限性**

局限性：① 仅关注巡航阶段，忽略爬升/下降等可能包含的退化信息；② 掩码方法假设高空稳定段，可能不适用于所有飞行阶段；③ RUL cap作为 benchmark 参数，若与其他研究混用需统一报告；④ DS08d子数据集因文件不可读未被纳入。

---

## 38. Mitigating Scaffolding Collapse in Socratic Tutors via Representation Alignment

**arXiv ID:** 2607.19371 | [PDF](https://arxiv.org/pdf/2607.19371v1)

**作者:** Jing Shao `[一作]` (Northeastern University), Jun Zhuang `[通讯]` (Boise State University)

**通讯引用:** 7731 | [OpenAlex ID](https://openalex.org/A5078851714)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型在 Socratic 教学中的“支架崩溃”现象，提出并实现了基于表示对齐的 SPRA 框架，用以在多轮对话中防止教师逐步放弃引导，直接泄漏答案。

**💡 创新点**

创新点在于将表示层的漂移与对话级别的偏好学习相结合：①使用冻结的参考缓存捕获温暖阶段的 Socratic 表示；②引入轨迹加权的直接偏好优化（TDPO），将崩溃发生时长作为权重；③设计基于全局“崩溃方向”的margin‑preserving 损失，保持正负样本在表示空间中的分离。

**🔧 技术方法**

使用技术包括：LoRA 细调、监督细调（SFT）、轨迹加权 DPO、margin‑preserving 表示损失、冻结参考缓存、层级表示漂移分析、红队对抗评测流程。

**📊 数据集**

数据集：从 MathDial、Socratic Debugging Benchmark v2、TutorChat 等收集 500 条正向 Socratic 对话，人工合成 2,500 条不同类型的崩溃轨迹，生成 7,022 条 per‑turn 选取/拒绝样本；另外构造 500 条独立评测问题（每学科 100 条）用于红队测试。

**📈 对比分析**

与提示防护、SocraticLM、ORPO、PedagogicalRL、RepBend 等基线对比；在 5 个 STEM 领域的直接答案请求攻击下，SPRA 将 Collapse Rate 降低至约 32%，平均崩溃开始轮数提升至 9–10 轮，拒绝率保持在 2–3%；整体性能显著优于现有方法。

**⚠️ 局限性**

局限性：仅在 STEM 领域实验，未覆盖人文/社会科学；只针对可开放权重、可访问隐藏层的 LLM，无法直接迁移到封闭式商业模型；聚焦防止崩溃而非提升整体教学质量。

---

## 39. Rethinking Uncertainty Evaluation in Large Language Models

**arXiv ID:** 2607.19367 | [PDF](https://arxiv.org/pdf/2607.19367v1)

**作者:** Krish Matta `[一作]` (Carnegie Mellon University), Andy Zou `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新审视大型语言模型（LLM）的置信度估计，提出更全面的评估框架C1，并对三种主流置信度估计方法（Verbal、Logit-based、SliCK）在结构一致性、可信度与实用性等维度进行系统对比。

**💡 创新点**

创新点在于：①引入结构一致性、可信度和实用性三大维度，揭示校准（calibration）不足；②将概率一致性（如归一化、合取一致性、蕴含单调性）转化为可测量指标；③利用采样一致性（SliCK）挖掘模型内部的概率不一致性，并展示RLHF与链式推理对这些指标的影响。

**🔧 技术方法**

技术方法包括：基于贝叶斯可信度假设的概率模型；采样一致性估计（SliCK）与文本生成置信度（Verbal、Logit-based）；对称一致性与归一化检验；对多步推理、提示和生成语义不变性的度量；以及利用统计指标（RMSCE、AUROC、结构偏差）评估。

**📊 数据集**

使用的数据集包括：SimpleQA（单步事实问答）、MuSiQue（多步推理）、ParaRel（提示语义不变性），以及对部分模型的200道子集评测。

**📈 对比分析**

与现有方法相比，SliCK在RMSCE（0.251 vs 0.778/0.700）和AUROC（0.825 vs 0.559/0.596）上显著优于Verbal和Logit-based，但同时暴露出31%的蕴含单调性违例，说明传统校准指标误导；RLHF在某些指标上略有提升但易导致判别力下降，链式推理显著改善合取一致性。

**⚠️ 局限性**

局限性包括：①采样一致性依赖于“采样可信度”假设，若模型生成分布与真实可信度不匹配则结果不可靠；②C1指标主要针对受限任务和提示，可能无法覆盖更广泛的实际使用场景；③对模型规模、训练方式的影响仍未完全揭示，结构一致性与实用性之间的权衡尚待深入研究。

---

## 40. Predicting Groundwater Arsenic Concentrations Using Graph Neural Networks

**arXiv ID:** 2607.19392 | [PDF](https://arxiv.org/pdf/2607.19392v1)

**作者:** William Xing `[一作]` (Algoverse AI Research), Kevin Zhu `[通讯]` (Algoverse AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了74,706条砷浓度样本的统一数据集，并评估了传统树模型与图神经网络在美国本土地区砷污染预测中的表现。

**💡 创新点**

将砷浓度预测从分类转为回归，首次在大规模环境预测中引入图神经网络，并融合多源地理、土壤与矿产信息。

**🔧 技术方法**

使用k近邻与GIS构建空间图，训练GCN、GAT、GraphSAGE、GIN等GNN，以及XGBoost、LightGBM、CatBoost、MLP等模型。

**📊 数据集**

采用Water Quality Portal、Mineral Resources Data System、Gridded National Soil Survey Geographic Database等公开数据集，并进行词嵌入与缺失值插补。

**📈 对比分析**

通过70/15/15的训练/验证/测试拆分，评估MAE/MSE，结果显示GNN在log尺度下与XGBoost相当，在实值尺度下GAT/GraphSAGE略优，整体优于随机森林和MLP。

**⚠️ 局限性**

受样本空间稀疏、深度差异、检测限缺失等限制，模型对极端高浓度点的捕捉仍有限，跨地区推广效果需进一步验证。

---

## 41. Bayesian Wind Tunnels for Model Selection

**arXiv ID:** 2607.19379 | [PDF](https://arxiv.org/pdf/2607.19379v1)

**作者:** Siddhartha R Dalal `[一作]`, Abhay Parekh `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究证明Transformer在对抗式生成的Bayesian风洞环境中能够实现精确的模型选择，即在数据中识别正确的假设类。

**💡 创新点**

创新点在于首次将Bayesian风洞扩展到模型选择层面，提出“感知访问条件”和“多项式门槛”，并展示了即使在符号编码不稳定的情况下，Transformer仍能对纯关系结构（如不变换、3循环）做出准确的模型选择。

**🔧 技术方法**

技术包括自监督训练的Transformer（6层、6头、192维），使用闭式贝叶斯推断、归一化注意力、交叉熵训练以及对比实验（整数标记 vs. 隐晦符号、嵌套与非嵌套假设类）。

**📊 数据集**

数据集是自生成的符号序列，涵盖旋转、乘法、固定点自由不变换、3循环等函数类，且每个样本包含一系列输入-输出对和后续预测查询。

**📈 对比分析**

通过与闭式贝叶斯最优解在熵误差和类后验MAE上对比，模型在整数标记下的旋转、乘法、以及不变换、3循环的选择误差均在0.01–0.12比特以内；在隐晦符号下仅在关系结构上保持低误差，体现了感知访问条件的影响。

**⚠️ 局限性**

局限性包括：对极端先验、随机查询顺序和更大n值的实验不足；缺乏对Transformer内部计算机制的直接解释；对前沿LLM的比较受限于logprob/采样估计；未能突破多项式门槛（d≥1）的可学习性。

---

## 42. Challenges of Explainability in Continual Learning for Time Series Forecasting

**arXiv ID:** 2607.19382 | [PDF](https://arxiv.org/pdf/2607.19382v1)

**作者:** Quentin Besnard `[一作]` (Université de Tours), Nicolas Ringuet `[通讯]` (Université de Tours)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5048020899)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用连续学习框架结合注意力经验回放和Grad‑CAM、注意力展开等可解释技术，对地下水位时间序列进行预测并分析模型的学习过程。

**💡 创新点**

将注意力采样、知识蒸馏与可解释方法融合，以揭示数据回放决策与模型对时间信息利用的内在机制，首次在非平稳时间序列上实现可解释的连续学习。

**🔧 技术方法**

采用 PatchMixer、PatchTST、DLinear 三种前馈/Transformer 预测模型，结合经验回放、知识蒸馏、注意力采样、Attention Rollout 与 Grad‑CAM 等技术。

**📊 数据集**

法国 Hubeau 数据集中的两条地下水位观测井（Piezometer 3 与 Piezometer 17）的日数据。

**📈 对比分析**

与随机采样、最大/最小损失采样等方法对比，使用 MAE/MAPE 等指标评估。注意力采样在保持可解释性的同时，预测精度显著优于随机和最小损失策略，并在大多数区间取得最佳或相近表现。

**⚠️ 局限性**

可解释信号随时间变化不稳定；注意力与 Grad‑CAM 的解释结果可能不一致；实验仅覆盖两条观测井，缺乏跨区域验证和对噪声、异常的鲁棒性分析。

---

## 43. Economic Evaluations of Language Models

**arXiv ID:** 2607.19375 | [PDF](https://arxiv.org/pdf/2607.19375v1)

**作者:** Alexander Wan `[一作]` (Stanford University), Rishi Bommasani `[通讯]` (Stanford University)

**通讯引用:** 4688 | [OpenAlex ID](https://openalex.org/A5069576651)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个开源评估套件，用来衡量语言模型在美国劳动力经济中的工作任务、详细工作活动（DWA）和职业级别的能力，并通过真实用户查询和合成数据大幅提升覆盖率。

**💡 创新点**

创新点在于：①使用多阶段检索与层级分类的低成本高精度管道获取真实查询；②开发白盒式仿真曝光度评估，直接量化模型对每个任务步骤的时间节省；③相较于GDPval，成本低500倍且覆盖范围更广。

**🔧 技术方法**

技术主要包括嵌入式检索、轻量级与高成本语言模型分类、Persona驱动的合成查询生成以及基于步骤拆解的仿真曝光度计算。

**📊 数据集**

利用WildChat、LMSYS、Chatbot Arena等公开对话数据做真实查询映射，并用O*NET 30.2税onomies生成合成查询，覆盖所有美国职业和数千个DWA。

**📈 对比分析**

通过与GDPval的Spearman相关性和模型排名对比，证明新评估能更好区分模型表现；仿真曝光度预测与Claude实际使用存在差距，揭示潜在时间节省机会。

**⚠️ 局限性**

局限性包括：仅关注聊天机器人能力，未覆盖非文本交互；受限于美国劳动力数据；隐私、专有系统及物理交互仍是主要瓶颈；合成查询的真实性与多样性仍需进一步验证。

---

## 44. Statistically Grounded Sparse-Feature Interventions for Activation-Space Control in Large Language Models

**arXiv ID:** 2607.19364 | [PDF](https://arxiv.org/pdf/2607.19364v1)

**作者:** Oshayer Siddique `[一作]` (Islamic University of Technology), Md Kamrul Hasan `[通讯]` (Islamic University of Technology)

**通讯引用:** 3512 | [OpenAlex ID](https://openalex.org/A5100656463)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 SAE 特征的无优化激活 steering 流程，并在 Gemma 系列模型上评估四个行为域。

**💡 创新点**

创新点在于用 F‑检验、KSG MI 与 Cohen's d 三种互补统计量进行无学习权重的 Borda consensus，并直接以 Cohen's d 加权构造 steering 向量，避免梯度优化的超参选择。

**🔧 技术方法**

使用的技术包括稀疏自编码器 SAE、激活 steering、Borda 排序、Cohen's d Fisher‑LDA 理论、三判定人评估与人工裁决、以及对 α 的全面扫掠。

**📊 数据集**

采用的数据集为 800 对对比样本（IMDB、LogicBench、ETHICS、Twinviews‑13k 及 LLM 生成补齐）用于特征选择，评估集为 100 条独立提示。

**📈 对比分析**

通过与 CAA、RePe、Top PC、ITI 四个基线在三判定人协议下的对比，改进域（逻辑、道德）实现最高 +1.16 的 Δp 和显著质量提升；但 raw‑shift 与 clean‑success 存在显著差距，多层配置在最佳域提升 7pp。

**⚠️ 局限性**

局限包括缺乏因果验证、判定者多重比较导致无显著单配置、仅覆盖 Gemma 系列模型、单语言评估、单属性 steering、未实现多属性组合。

---

## 45. GraphContainer: A Unified Platform for Comparing and Debugging Graph RAG Methods

**arXiv ID:** 2607.19362 | [PDF](https://arxiv.org/pdf/2607.19362v1)

**作者:** Seonho An `[一作]` (KAIST), Min-Soo Kim `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出GraphContainer平台，统一不同图RAG格式并可视化调试

**💡 创新点**

构建统一图表示层（UGR）和实时图记录器，实现跨框架可比性与可视化

**🔧 技术方法**

使用UGR映射多种图格式为统一UGS，Graph Recorder记录检索步骤，Live Visualizer在Web上交互展示

**📊 数据集**

实验数据集包括BSARD与SciFact，构造图采用LightRAG、HippoRAG与k‑NN三种方法

**📈 对比分析**

对比FastInsight与单跳检索，FastInsight在相同构造下性能更佳；k‑NN构造的图在两种检索下均表现最好

**⚠️ 局限性**

局限：目前仅在内存中处理图，未支持磁盘级大规模图；缺乏LLM驱动的自动调试与自然语言解释

---

## 46. Lifted Representation Hypothesis in Language Models

**arXiv ID:** 2607.19360 | [PDF](https://arxiv.org/pdf/2607.19360v1)

**作者:** Bumjin Park `[一作]` (KAIST AI), Jaesik Choi `[通讯]` (KAIST AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在学习规则时如何通过共享潜在结构（lifting）与细化（shattering）更新记忆，并通过异常学习实验探讨其能力。

**💡 创新点**

提出了“提升表示假设”（lifted representation hypothesis），阐明LLM倾向于通过共享潜在结构而非实例级事实来更新记忆，并区分提升与细化的机制。

**🔧 技术方法**

使用了参数微调（LoRA）和完整微调、以及上下文学习（in‑context learning）三种方法，对Gemma、Qwen、Llama系列模型进行实验。

**📊 数据集**

设计了基于可除性条件的模数规则（如 12、4、2、奇数）构成的人工异常学习数据集，用整数区间 [1,100] 训练， [100,1000] 评估。

**📈 对比分析**

结果显示：在混合训练下，LoRA 和完整微调均难以完全恢复嵌套规则；在顺序微调中，特定→一般（bottom‑up lifting）比一般→特定（top‑down shattering）更易成功，LoRA 通常优于完整微调；上下文学习对异常分离的效果有限，准确率基本保持在中等偏低。

**⚠️ 局限性**

局限在于实验仅使用合成规则数据，未检验在真实语言任务中的表现；模型在细化（shattering）上表现差，说明现有微调与上下文学习难以有效重塑共享内部表示；结果可能受模型容量与训练规模限制。

---

## 47. OpenEvoShield: Dual Non-Stationary Continual Defense for Open-World Multi-Agent System Attacks

**arXiv ID:** 2607.19351 | [PDF](https://arxiv.org/pdf/2607.19351v1)

**作者:** Litian Zhang `[一作]` (Beijing University of Posts and Telecommunications), Qiwei Ye `[通讯]` (Beijing Academy of Artificial Intelligence)

**通讯引用:** 9857 | [OpenAlex ID](https://openalex.org/A5068656698)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为OpenEvoShield的协同进化持续防御框架，用于在LLM驱动的多智能体系统中同时对抗攻击演化和正常行为漂移。

**💡 创新点**

创新点：①异步速率控制器实现攻击侧快速更新与正常侧慢速重校的解耦；②使用EWC正则化的检测策略集避免灾难性遗忘；③多粒度能量OOD头融合节点/子图/图层异常，提升未知攻击识别；④在开放世界部署下实现连续自适应。

**🔧 技术方法**

技术手段：对抗漂移检测（KL、余弦相似度）、动态阈值与指数滑动平均、生成式PRM得分、图注意力网络、EWC、能量判别、基于GNN的多粒度异常融合。

**📊 数据集**

数据集：PI（CSQA、MMLU、GSM8K）、TA（InjecAgent）、MA（PoisonRAG）共五个攻击基准，实验共 100 轮部署。

**📈 对比分析**

与静态和持续学习基线对比，OpenEvoShield 在 ASR@3 方面最低、MDSR@3 最高；在阶段 III 对未知攻击的检测率约 61.8%，误报率仅 4.8%。

**⚠️ 局限性**

局限：对高计算开销的GNN与LLM推理敏感；在极端漂移或新型攻击策略出现时仍可能出现误报/漏报；目前仅在图结构上评估，缺乏对更复杂多模态交互的验证。

---

## 48. FineServe: A Fine-Grained Dataset and Characterization of Global LLM Serving Workloads

**arXiv ID:** 2607.19349 | [PDF](https://arxiv.org/pdf/2607.19349v1)

**作者:** Tiancheng Zhang `[一作]` (Tianjin University), Wenyu Wang `[通讯]` (PPIO Cloud Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对全球多模型LLM服务工作负载进行细粒度收集、分析，并提出FineServe数据集和工作负载生成框架，支持真实轨迹重放与可扩展的参数化合成。

**💡 创新点**

创新点在于：①首次公开高分辨率、隐私保护的多模型LLM工作负载数据；②从模型架构、规模、任务意图三维度细化工作负载特征，揭示七条关键发现；③设计联合Gamma+负二项式的分层到达建模，精确捕获秒级及毫秒级突发；④构建可配置的任务感知负载生成器，兼顾真实轨迹与可扩展合成。

**🔧 技术方法**

技术包括统计建模（Gamma、负二项式、对数正态分布）、离散时间分段拟合、BERT+Google NLP API用于任务分类、混合生成器（先到达后负载）、聚合混合与评估指标（KS、W1、CV、MSSD）。

**📊 数据集**

使用了约1.48 亿条请求、57个模型、10个模型族的生产工作负载，涵盖Dense与MoE两大架构，4个月的全链路时序数据；数据源自天津大学与PPIO Cloud全球云计算市场，提供时间戳、模型ID、输入/输出token计数等隐私级别元信息。

**📈 对比分析**

通过对比参数化合成与真实轨迹的到达速率、突发频率和波动性（使用CV、MSSD、KS/W1等统计），验证参数化模型在宏观层面能够高度重现真实工作负载；在系统实验中未给出特定硬件性能指标，但展示了生成器可生成多模型、可调节负载，以支持后续调度/容量规划评估。

**⚠️ 局限性**

局限性包括：仅覆盖单一商业平台，未包含多模态或更细粒度用户行为；数据仅保留隐私元信息，无法复现完整请求内容；模型规模较小的MoE缺乏足够样本；任务分类仍基于自动标签，可能存在误差；生成器未考虑GPU异构性与实际硬件特性。

---

## 49. Structured Latent Space Modeling over Multi-Scale Temporal Patches for Multivariate Time Series Forecasting

**arXiv ID:** 2607.19404 | [PDF](https://arxiv.org/pdf/2607.19404v1)

**作者:** Xingsheng Chen `[一作]` (University of Hong Kong), Siu-Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22651 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多尺度时序补丁的CNN架构，并通过两种可微正则化（尺度内平滑和尺度间对齐）构造结构化潜在空间，支持多变量时间序列的预测与知识挖掘。

**💡 创新点**

创新点包括：①采用多尺度重叠补丁实现细粒度到粗粒度的分层表示；②使用深度可分离卷积与指数膨胀收敛线性复杂度的特征提取；③在潜在空间加入尺度内平滑约束与可学习的尺度间映射，使不同尺度的表示保持连续与一致；④通过通道独立设计提升鲁棒性并保持低开销。

**🔧 技术方法**

使用技术包括：多尺度补丁划分、深度可分离卷积（宽度=1d、膨胀指数递增）、线性投影至低维潜在空间、可学习的尺度间映射、L2平滑与对齐正则化、可逆实例归一化、可学习softmax融合。

**📊 数据集**

在十个公开真实数据集上进行实验，涵盖不同变量数与时序粒度（如ETTh1/2、ETTm1/2、ECL、Traffic、Weather、Solar、Exchange、Illness）。

**📈 对比分析**

与八类基线（PatchTST、iTransformer、Crossformer、Autoformer、DLinear、TimeMixer、TiDE、TimesNet）进行对比。m2patch在40种预测设置中获得57次最佳、34次第二佳排名，尤其在弱通道相关、强多尺度周期数据上表现突出；同时保持线性时间复杂度，训练速度快、显存占用低，并对补丁级缺失具有较高鲁棒性。

**⚠️ 局限性**

局限性包括：①通道独立设计无法利用显著的跨变量互相作用；②多尺度配置需人工指定，缺乏自适应选择；③固定的正则化权重在样本稀缺情况下可能过度约束，导致性能下降。

---

## 50. Memory Merge DQN: Sensitivity Weighted Target Updates for Stable Value Learning

**arXiv ID:** 2607.19397 | [PDF](https://arxiv.org/pdf/2607.19397v1)

**作者:** Adrian Ly `[一作]` (Deakin University), Francisco Cruz `[通讯]` (UNSW)

**通讯引用:** 33363 | [OpenAlex ID](https://openalex.org/A5058493970)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出 Memory Merge DQN，使用 Q 值敏感性加权合并最近 K 次在线网络参数来更新目标网络，从而替代传统的硬拷贝。

**💡 创新点**

创新点在于：① 用 Q 值梯度平方来衡量参数重要性，取代 Fisher 信息；② 在合并中加入递归优先权重，使目标网络既保持新网络特征，又保留有用的历史结构。

**🔧 技术方法**

采用 Q 值敏感性权重计算、经验回放、目标网络、层归一化、GELU 激活、CleanRL 框架等技术。

**📊 数据集**

在 54 个 Atari 游戏（如 Breakout、Pong、Asterix 等）上进行实验。

**📈 对比分析**

与 vanilla DQN、Averaged DQN、DQN+LayerNorm、PQN（带梯度裁剪）进行基准对比，使用每个游戏排名、对比赢数和最终回报等指标；Memory Merge 在 54 游戏中获得 21 个第一名，性能显著优于 vanilla DQN，接近 PQN，整体表现稳健。

**⚠️ 局限性**

局限性包括：需要手动选择记忆大小 K 和递归系数 λ；计算 Q 值敏感性权重会产生额外计算成本；对过大记忆会导致旧参数不兼容；方法仅在 DQN 架构下验证，未扩展到 actor‑critic 或其他价值学习框架。

---

## 51. LowPowAR: Power-Constrained Tone Mapping for Augmented Reality

**arXiv ID:** 2607.19509 | [PDF](https://arxiv.org/pdf/2607.19509v1)

**作者:** Weikai Lin `[一作]` (University of Rochester), Yuhao Zhu `[通讯]` (University of Rochester)

**通讯引用:** 3826 | [OpenAlex ID](https://openalex.org/A5023684582)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对可穿戴AR眼镜极低功耗需求，本文将显示映射建模为功耗约束的色调映射优化问题，并通过基-细节分解、可学习多段线性映射、递进优化和蒸馏实现实时推理。

**💡 创新点**

创新点在于：①提出可学习的基-细节分解与多段线性色调映射参数化，灵活权衡质量与功耗；②设计递进优化策略避免对抗性梯度；③利用知识蒸馏将迭代优化迁移到轻量级HDRNet网络，实现实时处理。

**🔧 技术方法**

采用双边滤波分解、K段线性色调映射、联合双边上采样、ColorVideoVDP感知损失、功耗条件损失、平滑正则、递进迭代优化及蒸馏技术。

**📊 数据集**

训练集由LAION‑POP与RICO采集的200张前景图与统一背景组成；评估集使用AR‑DAVID、XR‑DAVID、Aria、PEA‑PODs等公开AR/VR数据集的前景‑背景对。

**📈 对比分析**

通过与均匀暗化（UD）、EAI、PCSR以及无递进优化四种基线进行2AFC用户研究，获胜率分别约76%、73%、60%与60%；实时网络在Jetson Xavier上实现113.6 FPS，20%功耗目标下的视觉质量显著优于所有基线。

**⚠️ 局限性**

局限性包括：实验基于光学见透的测试平台，未考虑波导失真、眼箱变化与动态背景光照估计；在小文本区域易产生噪点；对极低功耗下的高对比细节处理仍不完美；缺乏专门为低功耗AR设计的感知评价指标。

---

## 52. Scalable Multi-Controller Coordination in Periplus via Border-Switch Forwarding Graphs

**arXiv ID:** 2607.19508 | [PDF](https://arxiv.org/pdf/2607.19508v1)

**作者:** E. M. Castro Barbero `[一作]` (Universidad Rey Juan Carlos), F. J. Simó Reigadas `[通讯]` (Universidad Rey Juan Carlos)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于控制器广告（C-Adv）实现的多控制器协调机制，使Periplus在资源受限的农村宽带网络中实现可扩展的 SDN 控制平面。

**💡 创新点**

创新点在于：①使用 C-Adv 进行控制器互联，②仅在域边界交换分布式转发图，③保证每台交换机的转发状态仅取决于其角色，而非控制器数量，从而实现无规模膨胀。

**🔧 技术方法**

技术方案包括：NSH 封装的 Slick Packets、Open vSwitch 的 Nicira 扩展、BFD 与 ITD 双重链路/节点检测、Ryu 框架实现、C-Adv 的自发式传播与增量转发图构造。

**📊 数据集**

实验使用 Mininet 生成的多种拓扑（2c_Linear、2c_Simple、4c_Mesh、2c_B4、2c_Clos 以及包含 96 台交换机、5 台控制器的 Large 拓扑），未使用公开数据集。

**📈 对比分析**

通过与默认/补丁版 OVS 的对比，评估启动时间、控制器发现时间、控制器失效恢复时间和流表占用。结果显示：三台或以上控制器时启动约 10 秒；发现时间 3.6–5.9 秒；失效恢复中位数 10.2 秒；内部交换机流表约 42 条，边界交换机约 52 条，保持恒定。

**⚠️ 局限性**

局限性：仅在 Mininet 仿真环境验证，未在真实物理测试床上测试；缺乏负载均衡与一致性协议；对高延迟/丢包链路的鲁棒性尚未评估；控制器间协调依赖于补丁版 OVS，可能影响迁移性。

---

## 53. Emergent Autonomous Drifting for Collision Avoidance in Real-World Winter Driving Scenarios

**arXiv ID:** 2607.19484 | [PDF](https://arxiv.org/pdf/2607.19484v1)

**作者:** Elliot Weiss `[一作]` (Toyota Research Institute), John Subosits `[通讯]` (Toyota Research Institute)

**通讯引用:** 331 | [OpenAlex ID](https://openalex.org/A5075929914)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种能在冬季低摩擦环境下自主出现漂移行为的非线性模型预测控制（MPC）框架，并在仿真中验证其在真实交通事故场景中的有效性。

**💡 创新点**

创新点在于：①不通过显式漂移模式或轨迹跟踪实现漂移，而是通过简单的路径、速度、方向和安全成本让MPC自发产生高侧滑；②利用实时预测与低摩擦区的随机分布，展示漂移在高速度下可为安全优势；③在仿真中将漂移控制与传统电子稳定控制（ESC）进行对比，揭示稳定性与可控性之间的权衡。

**🔧 技术方法**

主要技术包括：非线性单轨车辆动力学模型、耦合滑移刹车摩擦模型、基于三角形碰撞的安全成本、基于二次型的路径/速度/方向/输入约束、顺序二次规划（SQP）求解MPC、Monte Carlo 随机冰区仿真。

**📊 数据集**

使用的数据集：NHTSA 冰雪季节的致命事故聚类数据、SHRP2 车辆低摩擦系数分布、纽约州高速公路设计手册的道路几何参数、以及由随机生成的冰区长度与摩擦值进行的Monte Carlo实验。

**📈 对比分析**

比较方法：在三种基准场景（单车偏离、冰区漂移、迎面车碰撞）中，对照MPC与ESC的路径偏差、侧滑角、车速、碰撞/偏离风险。性能结果表明：MPC 在低至中速下平均侧滑角更小、车道偏差更小；在最高速 25 m/s 时，MPC 仍保持更低的平均车道误差，但偶尔出现较大偏离；相较于 ESC，MPC 在冰区与迎面车场景中更能保持车道中心并减少碰撞风险。

**⚠️ 局限性**

局限性：①仅基于仿真验证，未在真实车辆上实验；②单轨模型未考虑左右车轮的分离摩擦与侧倾动力学；③假设对迎面车、道路边界和摩擦系数拥有完美测量；④ESC 基线较为简化，未包含现代车辆中更复杂的稳定控制功能。

---

## 54. ModPack: An Extensible Teleoperation Interface for Bimanual Mobile Manipulation

**arXiv ID:** 2607.19479 | [PDF](https://arxiv.org/pdf/2607.19479v1)

**作者:** Joshua Citron `[一作]` (Stanford University), Shuran Song `[通讯]` (Stanford University)

**通讯引用:** 25670 | [OpenAlex ID](https://openalex.org/A5004644695)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了ModPack，一套可模块化、可扩展的可穿戴背包式遥操作系统，支持不同机器人平台与任务，集成关节级遥控、移动底盘、主动感知及触觉反馈等功能；

**💡 创新点**

通过将核心基础设施与任务/机器人特定模块解耦，构建可插拔的硬件/软件接口，实现快速适配多种机器人和任务；

**🔧 技术方法**

使用可穿戴背包搭载Mini PC、Dynamixel舵机、Apple Vision Pro、iPhone等硬件；软件层面采用轻量级消息队列、Gravity Compensation、EMA平滑、Jacobian‑transpose 触觉映射、ViT‑CLIP编码、Transformer‑Diffusion 策略网络；

**📊 数据集**

在两款移动双臂机器人上收集了两项真实世界任务的数据集：125条布料放置示范和102条盒子搬运示范；

**📈 对比分析**

对比不同视觉模态（Head‑Cam、Wrist‑Cam、All‑Cam）及加上关节力矩信息的策略，结果显示 Head‑Cam/All‑Cam+Torque 策略在布料放置任务中分别取得 22/25、20/25 成功率，在盒子搬运任务中 All‑Cam+Torque 最高 12/20，验证了主动感知与触觉反馈的有效性；

**⚠️ 局限性**

局限在于 Dynamixel 舵机的扭矩受限，导致重力补偿与触觉反馈的协同受限；电池容量大导致背包重量偏重；未来需要更高扭矩或更轻量的计算平台以提升系统可用性。

---

## 55. Detect Early, Escalate Rarely: Anytime Detection of AI-Generated Video from the Compressed Bitstream

**arXiv ID:** 2607.19476 | [PDF](https://arxiv.org/pdf/2607.19476v1)

**作者:** Mert Onur Cakiroglu `[一作]`, Hasan Kurban `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于压缩码流中运动矢量的流式AIGV检测框架，采用单一阈值进行任何时点的误报控制，并在不确定时通过轻量级像素CNN或VLM进行上升检测；

**💡 创新点**

核心创新在于：①将检测任务重构为流式感知，利用已有的运动矢量信息实现CPU端低计算前端；②通过单一端点校准阈值实现任何时点误报率保证；③引入宽度阈值调度的自适应推理，形成闭式计算-精度前沿；

**🔧 技术方法**

技术包括：压缩域特征提取（运动矢量的13维统计），序列最大化聚合，单点阈值校准（端点校准），阈值宽度退避（deferral window）以及基于GPU的像素CNN或小型VLM上升检测；

**📊 数据集**

使用GenVidBench（27k匹配生成/真实视频）和AIGVDBench（3,159视频）两大公开基准进行评估；

**📈 对比分析**

与传统像素CNN、VLM全剪辑检测以及离线ReStraV等方法对比：Codec前端单独AUC0.64，像素CNN0.76；在15%上升比例下，前端+像素CNN提升至0.78，计算量仅为像素CNN的1/7；VLM上升可进一步提升但计算成本显著增加；

**⚠️ 局限性**

局限性包括：需访问码流且运动矢量需对目标编码器友好；阈值校准依赖于训练集分布，跨数据集会漂移；聚合方式需保持单调性；对“面部深伪”场景适用性有限；上升阶段仍需大模型或GPU支持。

---

## 56. Generating Bearing Vibration Signals at User-Specified Fault Probabilities Using PR-GAN and Counterfactual Methods

**arXiv ID:** 2607.19455 | [PDF](https://arxiv.org/pdf/2607.19455v1)

**作者:** Seyed Mohammadreza Alavi `[一作]` (University of Tehran), Behnam Bahrak `[通讯]` (Khatam University)

**通讯引用:** 675 | [OpenAlex ID](https://openalex.org/A5018813049)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出两种生成方法实现对滚动轴承振动信号的目标故障概率控制（PR‑GAN和基于对抗的残差生成与无训练的对抗样本）

**💡 创新点**

首次在振动信号域内实现对连续目标概率的精确调节，且对比两种生成范式的实际表现

**🔧 技术方法**

使用Wasserstein‑GAN‑GP与概率对齐损失的PR‑GAN，以及Wachter‑style的梯度下降对抗样本；辅以固定的多模型CNN概率判别器

**📊 数据集**

Case Western Reserve University (CWRU) 和 Paderborn 两大工业轴承数据集

**📈 对比分析**

通过平均绝对概率误差、成功率、时间域ΔL1、ΔTV、频域PSD差异以及运行时长进行比较；CF在概率逼近、ΔL1与d_PSD上优于PR‑GAN，但生成耗时更长；PR‑GAN训练后推断快，整体误差略高

**⚠️ 局限性**

未评估判别器的校准性；生成的灰度区样本可能为对抗扰动而非物理可行的边界状态；缺乏对生成信号真实性的直接评估及下游应用验证

---

## 57. How Far Can Wearable-Compatible Signals Go? A Controlled Decomposition of Non-EEG Sleep Staging

**arXiv ID:** 2607.19441 | [PDF](https://arxiv.org/pdf/2607.19441v1)

**作者:** Yi Wang `[一作]` `[通讯]`, Yi Wang

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了四层可控分解框架，对非EEG睡眠分期进行系统性解构，并在消费者可穿戴、实验室PSG及EEG+EOG三层信号阶梯上进行评估。

**💡 创新点**

创新点在于：①将睡眠分期过程拆解为信号源、生理表征、时序先验与决策四层；②量化每层对Cohen κ的边际贡献；③引入置信度可放弃机制，展示置信度与生理可辨性一致；④以跨层信号质量阶梯对可穿戴性能上限进行客观定位。

**🔧 技术方法**

使用了Mamba2 轻量级状态空间模型、Multi‑MEA 多尺度时间证据聚合头、Viterbi 时序解码、置信度阈值放弃以及逆频率类别权重的交叉熵损失。

**📊 数据集**

采用了 Apple Watch Sleep‑Accel（HR+ACC，31人），Sleep Heart Health Study（SHHS，195人）以及 Sleep‑EDF‑20（EEG+EOG，20人）三套公开数据集。

**📈 对比分析**

通过保持模型架构与评估协议不变，逐层增量评估并对照标签置换负控制，展示：实验室非EEG实现 κ≈0.49，EEG+EOG 上限 κ≈0.80，消费者可穿戴 κ≈0.25；时间先验仅提升≈0.04；置信度阈值在覆盖率50%时可达 κ≈0.62；误差主要集中于 N1 阶段。

**⚠️ 局限性**

局限性包括：①EEG上限为跨数据集估计，真实模态差距可能更大；②仅验证单一模型架构，跨模型验证需进一步确认；③置信度采用最大 softmax，未尝试更高级的不确定性量化方法。

---

## 58. Opto-ViT-v2: Noise-Resilient On-Chip Fine-Tuning for Photonic Near-Sensor Vision Transformer Accelerators

**arXiv ID:** 2607.19421 | [PDF](https://arxiv.org/pdf/2607.19421v1)

**作者:** Xuming Chen `[一作]` (Case Western Reserve University), Gourav Datta `[通讯]` (Case Western Reserve University)

**通讯引用:** 479 | [OpenAlex ID](https://openalex.org/A5017435097)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种名为Opto‑ViT‑v2的光子视觉Transformer加速器，在芯片上实现参数高效的微调，解决了激活存储、权重写回和设备噪声等难题。

**💡 创新点**

创新点在于将权重增量做张量化低秩分解，仅在电子域更新少量共享因子（<8 K参数），并引入梯度累积稀疏头部训练，显著降低激活存储、写回次数并提升噪声鲁棒性。

**🔧 技术方法**

采用硬件‑算法协同设计、张量化低秩分解（Tensor‑Train）、梯度累积稀疏头、光子‑电子混合矩阵乘法以及系统级光子噪声模型。

**📊 数据集**

在VTAB‑1K（19个任务）和FGVC四个细粒度分类基准（Food‑101、Stanford Cars、Oxford Flowers‑102、FGVC‑Aircraft）上进行评估，使用ImageNet‑21K预训练的ViT‑Base。

**📈 对比分析**

与Adapter、LoRA、AdaptFormer、NOAH等传统PEFT方法比较，Opto‑ViT‑v2在硬件噪声环境下仅损失0.3–0.8%准确率，同时实现>100 KFPS/W吞吐率，显著优于全量微调。

**⚠️ 局限性**

局限性包括：仍需大量预训练权重；当前方案主要针对ViT‑Base规模，扩展到更大模型需进一步验证；光子器件的热漂移与交叉耦合仍是性能瓶颈。

---

## 59. FORCE-Bench: A Benchmark, Dataset, and Evaluation Harness for Agentic AI in Enterprise Finance

**arXiv ID:** 2607.19409 | [PDF](https://arxiv.org/pdf/2607.19409v1)

**作者:** Wolfgang M. Pauli `[一作]` (Microsoft Corporation), Jeremy Reynolds `[通讯]` (Microsoft Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 FORCE-Bench，一个专为运营金融任务设计的多维度评估基准，覆盖财务义务查询、企业绩效研究和业务简报生成三类真实场景。

**💡 创新点**

创新点在于：①针对金融专业需求构建八维度评估量表；②结合 ERP 与公开数据的混合工作流；③将评估框架与多工具代理无缝挂钩，支持复现与迁移。

**🔧 技术方法**

技术实现包括：大语言模型代理、工具调用（ERP、Web 搜索等）、DSPy 评判器、线性混合模型统计比较，以及开源评估 Harness。

**📊 数据集**

数据集包含 251 题目，覆盖内部 Dynamics 365 ERP（AR/AP）、公开公司财报与市场数据，且所有答案均由专家标注为金标参考。

**📈 对比分析**

通过对比 Microsoft 365 Copilot 的 Finance Agent 与 Anthropic Claude、OpenAI GPT‑5.5 等通用代理，在 60 s/300 s 计时约束下，目的构建代理在所有维度均显著优于通用代理；在放宽时限时性能差距缩小，但仍受推理时延影响。

**⚠️ 局限性**

局限性包括：与基准共同开发导致性能偏差、样本量有限、单轮评估、仅在 ERP 场景测量准确度、需要 Dynamics 365 环境、评判者可能存在偏差以及缺乏多轮对话与其他金融子域的覆盖。

---

## 60. ITPEval: Benchmarking Formal Translation Across Interactive Theorem Provers

**arXiv ID:** 2607.19407 | [PDF](https://arxiv.org/pdf/2607.19407v1)

**作者:** Jiayi Wu `[一作]` (Brown University), Anima Anandkumar `[通讯]` (California Institute of Technology)

**通讯引用:** 18378 | [OpenAlex ID](https://openalex.org/A5014498545)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ITPEval基准，用于评估跨四大交互式定理证明器（Lean 4、Rocq、Isabelle、HOL Light）的正式翻译。

**💡 创新点**

创新点在于构建双层（受控文件与生态系统文件）对齐四方基准、统一的多ITP验证框架，以及针对Lean 4的语义等价性检查BEq。

**🔧 技术方法**

采用统一的多ITP客户端、状态隔离的验证后端、LLM生成翻译、贝叶斯等价性检验等技术。

**📊 数据集**

使用了1,560个源文件、6,848条定理，覆盖Babel‑formal、Formalizing 100 Theorems和miniF2F三个子集。

**📈 对比分析**

对五大零样本LLM在statement和proof翻译任务上做pass@1评估，statement最高约29%，proof最高约10%，生态系统层难度显著高于受控层。

**⚠️ 局限性**

局限包括仅评估零样本pass@1、BEq仅针对Lean 4、未覆盖完整证明语义等价性、缺乏检索修复等增强技术。

---

## 61. Milo, a Fully Autonomous Indoor/Outdoor Robotic Guide Dog

**arXiv ID:** 2607.19530 | [PDF](https://arxiv.org/pdf/2607.19530v1)

**作者:** Florian Golemo `[一作]` (Mila - Quebec AI Institute), Christopher Pal `[通讯]` (Polytechnique)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一个低成本、完全自主、可在未知室内外环境中使用的机器人导盲犬平台，并将硬件与完整软件堆栈开源。

**💡 创新点**

创新点包括：① 采用可调节、带磁编码器的双自由度手柄，实现对导盲者姿态的实时感知；② 在车载上构建从RGB+LiDAR到BEV的端到端感知管线；③ 在GPU加速的2D BEV仿真器中使用强化学习训练导航策略，使机器人能够在未见环境下实现路径跟随与动态避障；④ 将训练好的策略零样本部署在真实四足机器人上。

**🔧 技术方法**

技术要点：基于Unitree Go2 + Nvidia Jetson Orin Nano；手柄带磁编码器 + D-pad；RGB图像通过NanoSAM进行走道分割，YOLO检测物体与行人；LiDAR点云构建体素地图；体素地图投影为BEV；强化学习（PPO）在Taichi仿真器中训练；局部安全过滤器对策略输出进行碰撞预防。

**📊 数据集**

数据集：在自研的GPU加速BEV仿真器中随机生成的室内外路径、障碍与行人场景；实测环境为多个室内走廊、木桥及街道旁人行道的障碍课程，使用实时LiDAR、RGB和手柄编码器数据。

**📈 对比分析**

方法比较：与经典基于成本图的轨迹选择基线对比。实验结果表明，RL策略在路径跟随、静态障碍与行人避让中均能显著减少导盲者碰撞，且在部分测试中实现了更平滑、更快的行走（平均时长 13.7–22.3 s vs. 21.3 s）。

**⚠️ 局限性**

局限性：① NanoSAM 需要预设机器人在可行走表面，难以跨越斜坡或不平地形；② 训练策略在极端拥挤或行人突然出现时会过度保守；③ 当前框架仅适用于人行道/楼地面，无法跨越街道或处理交通信号；④ 高速行走时 LiDAR 更新率成为瓶颈；⑤ 仍缺乏对社交导航与多目标协同的优化。

---

## 62. The C-index illusion: discrimination without calibration in published survival models

**arXiv ID:** 2607.19526 | [PDF](https://arxiv.org/pdf/2607.19526v1)

**作者:** Rafael da Silva `[一作]` (Eastern University), Danilo Alvares `[通讯]` (University of Cambridge)

**通讯引用:** 709 | [OpenAlex ID](https://openalex.org/A5041711951)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对三个非临床领域（硬盘故障预测、P2P 信贷违约、数字平台用户流失）的公开生存分析模型进行复现，并采用“双螺旋阶梯”评估框架检验仅报告 C‑index 的风险。

**💡 创新点**

证明仅用 C‑index 评估易导致校准失效、竞争风险偏差以及时间尺度不一致性，并提供可复现的预注册评估工具。

**🔧 技术方法**

使用 Harrell C、Uno C、D‑Calibration、IPCW‑IBS、Copula‑Graphic IPCW、Fine‑Gray 竞争风险模型、特征消融、Brier 曲线等多种评估指标。

**📊 数据集**

Backblaze SMART 硬盘统计、Bondora P2P 借贷数据、Stack Exchange 用户行为数据。

**📈 对比分析**

与原始论文对比复现结果一致，C‑index 接近；在评估中发现模型校准显著不合格、竞争风险导致默认概率偏高、Brier 分数随预测时长升高，表明仅用 C‑index 评估误导。

**⚠️ 局限性**

样本仅覆盖三种领域、模型数有限、竞争风险偏差阈值选择有歧义、对依赖性删失的敏感性未完全覆盖、部分结果受重现精度限制。

---

## 63. SynPre-FL: Synthetic data-driven pretraining integrated Federated Learning training framework

**arXiv ID:** 2607.19524 | [PDF](https://arxiv.org/pdf/2607.19524v1)

**作者:** Akarsh K Nair `[一作]` (Nottingham Trent University), David J. Brown `[通讯]` (Nottingham Trent University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了将合成数据预训练与联邦学习相结合的框架SynPre-FL，用于实现分布式临床风险预测；

**💡 创新点**

创新点在于构建高保真潜在自编码器‑扩散模型生成合成EHR，并将其作为全局初始化，结合类平衡损失、FedProx与Adam式聚合，并提供联邦安全的概率校准与SHAP解释；

**🔧 技术方法**

使用技术包括潜在自编码器‑扩散生成器、联邦优化算法FedProx/FedAdam、类别平衡交叉熵、后期Platt式校准与Kernel SHAP；

**📊 数据集**

使用的实验数据集为基于Synthea模拟的肺癌风险预测电子健康记录，共29维特征，按5、10、15个非IID客户端划分；

**📈 对比分析**

与FedAvg、FedProx+FedAdam等基线比较，在5/10/15客户端下评估AUC/ACC/F1，SynPre-FL在高异构情境下取得最高AUC（约0.8943）且校准后F1显著提升；

**⚠️ 局限性**

局限包括：合成模型仅基于单中心数据；解释性仅在全局模型上；未实现正式差分隐私；假设同步无客户端掉线；多中心真实数据的泛化验证不足。

---

## 64. Building Trust in Autonomous Commerce: A Verifiable Global Event Timeline and AI-Ready Fraud Intelligence Layer

**arXiv ID:** 2607.19436 | [PDF](https://arxiv.org/pdf/2607.19436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 65. Do Sheaf Neural Networks Use Holonomy? A Measure--Intervene--Control Study

**arXiv ID:** 2607.19514 | [PDF](https://arxiv.org/pdf/2607.19514v1)

**作者:** Ankit Grover `[一作]` (KTH Royal Institute of Technology), Rémi Bourgerie `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在几何架构中三角形环路量的学习与机制，通过基于基向量不变的扭曲、增益和翻转测度，对Sheaf神经网络的学习效果进行了量化分析。

**💡 创新点**

首创了基于基向量不变的三角形环路量度量，并提出“测量–干预–对照”框架，用以区分几何变化、连接敏感性和三角形特定计算的作用。

**🔧 技术方法**

采用Sheaf神经网络（Neural Sheaf Propagation/Diffusion）、三角形环路测度、身份替换干预、图摘要岭回归等技术。

**📊 数据集**

使用基于DC‑SBM的高同质性合成图和随机6正则图进行实验。

**📈 对比分析**

与社区检测基线、训练均值预测器和图摘要岭回归比较；NSP在计数任务上显著提升环路旋转，但整体预测误差仍高于简单图摘要模型；身份替换显著增加误差，固定度模型即使旋转增大也未提升计数性能。

**⚠️ 局限性**

旋转增大并不能保证计数准确，测量仅限于二维、合成数据；缺乏对真实图数据的验证，机制解释仍有局限。

---

## 66. MC-BRIDGE: A Modular Receiver-Chain Simulation Framework for OECT-Based Molecular Communication

**arXiv ID:** 2607.19502 | [PDF](https://arxiv.org/pdf/2607.19502v1)

**作者:** Hongbin Ni `[一作]` (University of Cambridge), Ozgur B. Akan `[通讯]` (Koç University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套基于有机电化学晶体管（OECT）的分子通信接收机仿真框架，完整覆盖释放、扩散、受体结合、OECT 转导、电噪声、校准和检测全过程，并通过模块接口实现各层之间的可插拔耦合。

**💡 创新点**

创新点在于：①构建统一的电荷域统计接口，使不同物理模型可在同一框架下对接；②引入完整序列多通道彩色噪声合成与协方差分析；③实现校准转移、记忆深度与控制相关性的系统评估；④提出基于保留预算的“held‑out”测试与自适应搜索的可靠性上界方法。

**🔧 技术方法**

使用的技术包括：布朗运动扩散模拟、Langmuir 受体动力学、Quasi‑static OECT 电导模型、分段离散噪声合成、匹配控制参考、统计阈值校准、Bootstrap 区间估计、序列级别的符号误码率（SER）与互信息评估。

**📊 数据集**

数据集：所有结果均基于大规模蒙特卡罗仿真生成的合成符号序列（上千符号、不同随机种子），未使用实验测量数据。

**📈 对比分析**

比较方法：在同一物理参数集下对不同观测、绑定、校准和相关性策略进行系统 sweeps；采用自适应预算搜索、hold‑out 预算检验和 bootstrap 区间来评估 SER 与互信息；结果显示：有限面积观察降低信号约 70%，校准刷新显著提升 SER（从 0.3% 降至 0.03%），不同相关性下的混合解调可在 SER 1% 以内完成，记忆深度越大误码率越高。

**⚠️ 局限性**

局限性：①假设被动受体（无质量守恒与耗尽）；②基于点源和单一扩散模型，未考虑复杂生物微环境；③校准转移仅在相同操作点下验证，跨点点迁移性能未知；④未结合真实设备测量，OECT 传导与噪声特性仍需实验验证。

---

## 67. Unlearning as Distribution Restoration: A Controlled Counterfactual Study, a Validated Selective Screen, and the Limits of Oracle-Free Certification

**arXiv ID:** 2607.19442 | [PDF](https://arxiv.org/pdf/2607.19442v1)

**作者:** Sen Yang `[一作]` (New York University), Yuen-Hei Yeung `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构造剂量匹配的注入测试床和对应的重训练参考模型，对语言模型的机器忘记（unlearning）方法进行全面审计与评估。

**💡 创新点**

创新点在于揭示常用的训练探针指标会误导残留知识的评估，并提出无参考的基于基线屏蔽的持久残差筛选、损伤相对再校准和可识别性定理，以量化正向单向协议的可行性边界。

**🔧 技术方法**

采用剂量匹配注入、重训练参考、基线锚定秩检验、往返残差、损伤相对校准、对抗性Logit抑制攻击以及理论可识别性分析等技术手段。

**📊 数据集**

数据集包括在五类开源小到中型 LLM（Qwen、SmolLM、Pythia、GPT‑2、OLMo）上构造的 45 个模型‑种子单元的 nonce 事实，以及 Phi‑1.5 的 TOFU 真实检索基准。

**📈 对比分析**

对比恢复式方法（如任务向量消除、路径特定逆转）与抑制式方法（梯度上升、NPO），恢复方法在与重训练参考的 KL 误差上约 2 倍更优；绝对阈值法全部失败，基线屏蔽成功排除所有注入模型并接受参考；再校准方案在 15 个单元中通过认证，选择的模型与参考在重训练噪声内。

**⚠️ 局限性**

局限性包括：仅在无参考的前向单向黑盒设置下可用；样本数有限导致检测功效受限；仅对反事实事实可识别；使用的忘记集较小；依赖重训练复制噪声估计；未对自适应或白盒攻击给出完整保证；模型规模有限。

---

## 68. The Chronos Vulnerability: A Taxonomy of Temporal Persistence and Memory-Based Deception in Agentic AI

**arXiv ID:** 2607.19433 | [PDF](https://arxiv.org/pdf/2607.19433v1)

**作者:** Om Narayan `[一作]` (New York University), Praveen Baskar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并系统化了Chronos漏洞——基于时间持久性的内存注入攻击（如MINJA、EchoLeak）和动态盲区（Dynamics Blindness）所导致的智能体安全威胁，并设计了多层防御框架（AgentDoG、Agent-C、A-MemGuard、TEE+Zero‑Trust内存）

**💡 创新点**

创新点在于：①首次将内存注入与动态盲区统一为Chronos漏洞的框架；②提出从BDI到向量数据库的内存层级映射与攻击维度分类；③将形式化时序约束与人工免疫机制结合，提供跨层防御；④引入硬件可信执行环境与加密内存周期化重新加密，形成端到端的Zero‑Trust内存栅栏

**🔧 技术方法**

使用技术包括：BDI模型、向量数据库（VectorDB）与语义检索、SMT求解器（Agent-C）、形式化DSL与FOL、人工免疫系统（A-MemGuard）与共识验证、GPU TEE（NVIDIA H100 Conf. Computing、Intel TDX）、零信任内存协议（MemTrust）

**📊 数据集**

使用数据集与基准：World of Workflows（WoW）、AgentBench、WoW‑bench（unconstrained & constrained）、GPT‑4、Gemini‑1.5 Pro、Claude‑3.5、Llama‑3.1‑70B 等公开工作流与大模型基准

**📈 对比分析**

对比方法：在ATBench上AgentDoG实现风险源识别准确率82%，远超基础模型；Agent-C将GPT‑5的形式化时序合规率从83.7%提升至100%；A‑MemGuard通过多路径共识过滤显著降低内存注入成功率；TEE实现模型加密开销仅约2%（在NUMA与PCIe优化下），与传统CPU‑only 20–100%延迟相比有显著优势；然而在极端工作负载下仍存在显著性能瓶颈

**⚠️ 局限性**

局限性：①软件层防御对硬件级别的内存指针或权重篡改仍无防护；②实验主要基于公开数据集与特定大模型，缺乏工业规模真实系统验证；③对多智能体协同攻击（Agent Session Smuggling）的完整检测与缓解机制尚未成熟；④TEE与加密内存的部署成本高、对现有云基础设施的兼容性有限

---

## 69. JailMeter: An Evidence-Based Evaluation Framework for Jailbreak Attacks on Large Language Models

**arXiv ID:** 2607.19424 | [PDF](https://arxiv.org/pdf/2607.19424v1)

**作者:** Qingjia Huang `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Xiaoqi Jia `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出JailMeter框架，用多智能体双反馈优化从LLM响应中提取证据，以更可靠地评估jailbreak攻击。

**💡 创新点**

创新点在于将信息瓶颈原理与多智能体双反馈机制结合，消除jailbreak噪声并可蒸馏为低成本的JailMeterSLM；同时构建了人类标注的高难度评估基准JailMeter‑Eva。

**🔧 技术方法**

采用多智能体系统、信息瓶颈原理、双反馈优化、链式推理、LLM评估与多智能体知识蒸馏等技术。

**📊 数据集**

使用了JailMeter‑Eva（330个非拒绝案例）、BenignInstructions（100条善意指令）和AdvBench生成的攻击样本等数据集。

**📈 对比分析**

在JailMeter‑Eva上与8种现有评估方法对比，JailMeter准确率97.27%、F1 93.88%，JailMeterSLM准确率95.15%、F1 89.19%，显著优于其他方法。

**⚠️ 局限性**

局限性包括仅支持文本攻击，无法处理多模态jailbreak；以及对评估器本身的针对攻击较为脆弱。

---

## 70. Inference-Behaviour Semantics for All$^\ast$ Connectives in Two-Dimensional Sequent Calculi

**arXiv ID:** 2607.19419 | [PDF](https://arxiv.org/pdf/2607.19419v1)

**作者:** Sophie Nagler `[一作]` `[通讯]` (University of Amsterdam), Sophie Nagler (University of Amsterdam)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文系统地枚举了在两维序列算子系统中可形式化的连词（共10816种），并通过保守性与唯一性证明，找出了其中376种可被最小化保守性（且非空洞）的连词。

**💡 创新点**

创新点在于提出了“保守性连词”与“唯一性连词”这两类新概念，并通过子公式连通性与上下文共享形状的约束，给出了完整的分类与证明框架。

**🔧 技术方法**

使用了逻辑系统的剪枝证明技术（cut-elimination）、水平与垂直互导性检验、以及复杂的子案例枚举与反例构造方法。

**📊 数据集**

本研究未使用传统机器学习数据集，而是基于逻辑语义的完整枚举数据集（10816种连词），并对每种连词的推理规则进行手工证明与分类。

**📈 对比分析**

通过对376种保守连词与其对应的唯一性案例进行对比实验（水平与垂直互导性检验），表明其中150种连词在加入等价规则后可以实现唯一性，性能上主要表现为推理的可验证性与系统完整性的保持。

**⚠️ 局限性**

限制在于仅考虑两维序列系统的最小化约束，且部分复杂连词的唯一性证明依赖于子公式连通性假设，未来工作需扩展到更高维度与非子公式连通连词的情形。

---

## 71. NMR Elucidation as an Agentic Search Problem, Not a Modeling Problem

**arXiv ID:** 2607.19406 | [PDF](https://arxiv.org/pdf/2607.19406v1)

**作者:** Irina Espejo Morales `[一作]` (PolymathicAI), Shirley Ho `[通讯]` (PolymathicAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了一种基于冻结LLM的代理式约束搜索框架，用于从一维NMR谱和分子式中自动推断小分子结构。

**💡 创新点**

创新点在于不采用端到端模型，而是将结构阐释恢复为多步骤推理和验证的“拼图”任务，利用LLM与专门工具协同完成约束化搜索。

**🔧 技术方法**

核心技术包括OpenAI AgentsSDK构建代理、冻结大语言模型、FID/频域预处理管道、化学峰值提取与分子结构生成工具，以及基于分子式、饱和度、积分和化学环境的验证规则。

**📊 数据集**

使用的实验数据集为Chemistry Education（van Bramer）、Alberts和AstraZeneca三组真实NMR谱数据。

**📈 对比分析**

通过与深度学习模型（ChefNMR、MMST）、专业NMR阐释软件和研究生水平进行对比，Chemistry Education上Top‑1准确率达80%，Alberts上71%（与研究生66%相当），AstraZeneca上20%，显示在多数情形下可与训练模型相当或优于。

**⚠️ 局限性**

局限性包括对复杂分子、噪声或峰重叠导致的推理失败、推理路径易陷入局部最优、对不确定性量化不足，以及对更高维模态和大分子结构的扩展仍待改进。

---

## 72. Reproducing Recurrent Transformers: The CoTFormer

**arXiv ID:** 2607.19405 | [PDF](https://arxiv.org/pdf/2607.19405v1)

**作者:** Aras Kavuncu `[一作]` (University of Southampton), Alberto Berni `[通讯]` (University of Southampton)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文重新实现并评估了CoTFormer模型及其变体，探讨其在语言建模与算法推理中的表现。

**💡 创新点**

创新点在于将链式推理形式化为可重复使用的隐藏状态计算，并通过保留中间状态实现可追踪的思路痕迹。

**🔧 技术方法**

采用循环的Transformer结构、权重共享的重复块、层归一化（LN）以及自适应深度模块（ADM）等技术。

**📊 数据集**

使用公开的基准语言模型数据、Shifted‑Start计数任务数据以及构造的p‑hop推理数据集进行实验。

**📈 对比分析**

与标准Transformer和Block Universal Transformer进行比对，CoTFormer在p‑hop推理任务上表现最佳，但在OOD计数任务上提升有限；在计算效率与困惑度上基本与原论文一致。

**⚠️ 局限性**

局限在于缺乏对递归是否真正实现迭代推理的明确证据，且适用性受任务依赖；需要匹配计算成本、多种随机种子和更精细的控制实验来进一步验证。

---

## 73. Geospatial Diffusion-based Evolution Synthesis (GeoDES) for Storm-Centered Weather Augmentation

**arXiv ID:** 2607.19522 | [PDF](https://arxiv.org/pdf/2607.19522v1)

**作者:** Sonia Cromp `[一作]` (University of Wisconsin-Madison), Allegra LeGrande `[通讯]` (NASA Goddard Institute for Space Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发并评估一种基于扩散模型的风暴中心化生成框架 GeoDES，用于合成北大西洋中尺度气旋的完整时空演变。

**💡 创新点**

创新点包括：① 风暴中心化定位，专注于气旋局部区域；② 先进行二维空间预训练再通过时间膨胀（temporal inflation）迁移到三维时空网络；③ 采用变量特定的混合归一化、SNR 加权损失和相关噪声调度，提升物理一致性与高频能量恢复；④ 非自回归采样，避免时间误差累积。

**🔧 技术方法**

技术手段：DDPM 生成式扩散、二维 U‑Net 预训练、三维 U‑Net 细调、时间膨胀权重映射、混合对数/线性归一化、SNR 权重、ρ‑相关噪声、条件提示（初始 0h 状态）和多尺度损失。

**📊 数据集**

数据集：ERA5 重分析，1940‑2015 年训练，2016‑2024 年测试；提取的极地中尺度气旋框（约 1600 km²）包含 5 个变量：海平面气压、500 hPa U/V 速度、925 hPa 温度、500 hPa 湿度。

**📈 对比分析**

与 SVD、CoDiCast、CEF、ClimaX‑FT、ClimaX、Aurora、气候平均等基线对比；GeoDES 在峰值涡度误差（5.04 vs 48.40）、高频谱比（0.95 vs 52.52）、频率偏差指数（1.02 vs 1.10）、分数技能分（0.76 vs 0.65）等关键指标上显著优于所有对照；极端风暴子集表现尤为突出；计算成本上 GeoDES 参数1.8 B，推理 17 s、VRAM 7.1 GB、能耗 1.44 Wh，远低于全局 WFMs。

**⚠️ 局限性**

局限性：目前仅在 500 hPa 中层高度生成，未覆盖降水、地面风速等表面/对流尺度过程；需要进一步集成降水、表面层变量及操作化推理管线；对极端高频噪声的物理解释仍待完善。

---

## 74. Sophisticated Policies from Epistemic Priors

**arXiv ID:** 2607.19518 | [PDF](https://arxiv.org/pdf/2607.19518v1)

**作者:** Wouter W. L. Nuijten `[一作]` (Eindhoven University of Technology), Bert de Vries `[通讯]` (Eindhoven University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 Sophisticated Inference 在活跃推理中的作用，提出通过 epistemic‑prior 变分自由能实现内在闭环控制，并在 Reactivity Maze 基准上验证。

**💡 创新点**

证明 Sophisticated Inference 的优势并非源于树搜索，而是由信息驱动（epistemic drive）与闭环推理结构共同作用所致。

**🔧 技术方法**

采用变分自由能最小化、联合后验推理、梯度优化以及树搜索等技术实现对比实验。

**📊 数据集**

使用 Reactivity Maze（一个带隐藏目标和刺激的随机迷宫）作为数据集。

**📈 对比分析**

与标准 EFE 规划、行动–状态分离、F_active^u⊥x 等方法对比，F_active 与 Sophisticated Inference 在最优率、奖励等指标上均优于其他方法。

**⚠️ 局限性**

实验规模有限，仅验证概念性优势，未深入探讨大规模实现与计算成本。

---

## 75. Crowd4D: Scene-Aware Monocular 4D Crowd Reconstruction

**arXiv ID:** 2607.19517 | [PDF](https://arxiv.org/pdf/2607.19517v1)

**作者:** Hongbo Kang `[一作]` (Tianjin University), Kun Li `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种针对大规模、复杂地形场景的单目 4D 群体重建框架 Crowd4D，能够从单一 RGB 视频中同时恢复完整场景几何、相机轨迹以及多人的三维运动轨迹。

**💡 创新点**

核心创新点包括：①基于 Scene Interaction Point Cloud 与 Surface（SIPC&SIS）的 Human–Scene Interaction Proxy（HSIP），实现场景几何与人群位置的自适应耦合；②引入 Crowd Structural Coherence Regularization（CSCR）在局部邻域内强制时间一致的相对位移，提升遮挡时的运动连贯性；③采用三阶段逐步优化策略，先解决尺度与根位姿，再细化姿态，最终优化运动与群体一致性。

**🔧 技术方法**

关键技术包括 SMPL 模型、VPoser 运动先验、图像检测与跟踪（YOLOX、SAM、DWPose、BoostTrack++）、相机与场景重建（π³、GeoCalib）、HSIP 的几何约束（水平可行域、垂直适配、投影一致性）、CSCR 的邻域一致性损失，以及多阶段梯度下降优化。

**📊 数据集**

主要使用 VirtualCrowd 合成数据集进行定量评估，PANDA 实际场景视频用于定性展示；此外对比实验使用 Crowd3D、GroupRec、DyCrowd、VideoMimic 等公开方法。

**📈 对比分析**

与上述基线方法相比，Crowd4D 在 PPDS、PA‑PPDS、PCOD 上均有显著提升，MPJPE 下降约 6–9 mm，ACCEL 下降至 12–16，说明在全局一致性、姿态精度和运动平滑度上均优于现有技术。

**⚠️ 局限性**

局限性在于：①离线优化过程计算成本高，单场景重建需数小时；②性能高度依赖单目场景重建、检测与跟踪的准确性，严重遮挡或低质量视频会导致误差累积；③缺乏真实大规模场景的三维标注，导致难以在真实数据上做严格量化；未来需探索更高效的前向预测或伪标签学习。

---

## 76. Total Variation Distance Estimation in Autoregressive Models

**arXiv ID:** 2607.19510 | [PDF](https://arxiv.org/pdf/2607.19510v1)

**作者:** Eric Price `[一作]` (University of Texas at Austin), Yusong Zhu `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对两种大型语言模型（LLM）生成的长度为 n 的序列分布，提出并实现了一种通过不同 API 访问方式（采样、logits、带噪声 logits）估计总变差（TV）距离的算法。

**💡 创新点**

创新点包括：①首次将 TV 估计问题在条件访问模型（prefix logit / noisy prefix）下正式化并给出最优查询复杂度；②设计了基于似然比的无偏估计器与多层次 Monte Carlo 变异降低方案，实现噪声下的 O(n + n²σ²/ε²) 查询复杂度；③通过信息论下界证明三种访问模型的复杂度区分，并展示噪声水平对查询需求的影响。

**🔧 技术方法**

主要技术手段包括：条件采样与 logit 查询模型的构造、TV 与似然比的等价表达式、无偏估计器与 Hoeffding 以及 Cauchy–Schwarz 边界、信息论下界（稀逃逸构造）以及多层次 Monte Carlo（MLMC）方差降低。

**📊 数据集**

实验使用公开 LLMs（如 Qwen）在相同权重下的两种推理堆栈（auto vs cuDNN, math vs FlashAttention-2 等），通过对比采样条件、重新评分条件下的 TV 估计，并对不同 top‑k、批量大小等参数进行检验。

**📈 对比分析**

与基线（如 KL 散度、传统样本估计）相比，TV 估计在噪声和 top‑k 变动时保持有限且可解释；在噪声较大时，多层次方案显著降低预算（约 1.3 倍），而单层固定重复次数的估计则受限于偏差底部；总体查询复杂度与理论下界相匹配。

**⚠️ 局限性**

局限性包括：对序列长度 n 的线性/二次标度仍可能不满足极长生成任务；噪声模型假设独立且可控的相对方差，实际 GPU / kernel 产生的噪声可能更复杂；实验仅覆盖部分模型与配置，尚未验证在多任务、跨语言或大规模自定义权重下的稳健性。

---

## 77. MoA-Structured Decode Attention DNF Derivation, KV-Cache Accumulation, GQA/MQA, and OpenACC Kernel

**arXiv ID:** 2607.19456 | [PDF](https://arxiv.org/pdf/2607.19456v1)

**作者:** Lenore Mulin `[一作]`, Gaetan Hains `[通讯]` (Universite Paris-Est Creteil)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

针对Transformer注意力推理阶段，利用MoA提出了四种内存最优实现，包括单查询解码、C/OpenACC GPU核、KV缓存高效追加以及分组/多查询注意力

**💡 创新点**

创新点在于直接从训练时的Denotational Normal Form推导推理算法，利用ψ-reduction消除K^T、KV缓存仅写一行、并通过ψ-selection实现KV组共享，所有步骤均可在编译前证明达到信息论最小内存传输

**🔧 技术方法**

采用Mathematics of Arrays（MoA）符号、Denotational Normal Form（DNF）与Operational Normal Form（ONF）转换、γ-偏移算子、Ω内积以及OpenACC并行指令

**📊 数据集**

文中未给出具体训练/推理数据集，主要以理论分析和PyTorch SDPA验证为主，实验规模为n≤4096、L=96层

**📈 对比分析**

与传统O(n^2)实现和FlashAttention对比，MoA解码在单步内存流量仅为O(n+n)，GPU实现误差为0，KV追加保持恒定4B/行，GQA/MQA将KV流量降低至h_q/h_kv倍；能耗估计可达约14700kWh/天

**⚠️ 局限性**

局限在于仅针对标准Scaled Dot-Product Attention，未涵盖可变长度注意力或自回归训练梯度、以及在不同硬件上对MoA编译支持的成熟度

---

## 78. Predictive Extrema, Unprofitable Policies: An AI-Assisted Audit of Candle-Based Binance Spot Timing Models

**arXiv ID:** 2607.19453 | [PDF](https://arxiv.org/pdf/2607.19453v1)

**作者:** Ayoub Jadouli `[一作]` `[通讯]` (Abdelmalek Essaadi University), Ayoub Jadouli (Abdelmalek Essaadi University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 Binance Spot 市场的蜡烛图基础机器学习模型进行全面的政策评估与审计，验证预测极值是否能在考虑交易成本后产生正收益。

**💡 创新点**

首次将可追溯、artifact‑backed 的策略性能评估与 AI 辅助的文献检索、审计和流程自动化相结合，揭示预测与执行之间的价值脱节，并公开完整的审计流程。

**🔧 技术方法**

使用 ExtraTrees、逻辑回归、SVM、随机森林、梯度提升机、MLP、CNN、GRU、LSTM、Transformer、Attention‑CNN‑LSTM 等多种监督学习模型；对策略执行采用定量的成本模型（每完成周期 20/31/51 bps）、严格的时间窗和止损/止盈设定；利用 AI 助手完成文献检索、代码审计与结果核对。

**📊 数据集**

Binance Spot 公共 Kline 数据：10 组 USDT 交易对的 1‑分钟 OHLCV（2025‑03‑01 至 2026‑07‑19），3 组 BTC/ETH/SOL 的 5‑分钟 OHLCV（2025‑03‑01 至 2026‑07‑12），以及 BTC/ETH 的日线 OHLCV（2021‑06‑01 至 2026‑07‑01）。

**📈 对比分析**

对每种策略使用“前瞻性”评估（未在评估期间出现的数据）与“历史”对比；加入成本假设，比较与现金/买持持平价基准；结果表明：未改动的 19 天每日选择器累计 -6.72%，局部极值模型分别 -1.79% 和 -2.80%，每日极值适配器 -44.30%（相对买持 -41.20%）。虽然 ROC‑AUC 高，但平均精度低、净收益均为负。

**⚠️ 局限性**

样本量有限（仅数十个周期）、成本假设可能与实际费用不符、未模拟滑点/交易延迟、时间序列自相关未完全控制、数据集受 survivorship/look‑ahead 偏差影响、以及对特定币种和交易对的局限性，导致结论在其他市场或策略设定下可能不适用。

---

## 79. CAPS: A Cascaded Reconstruction Model to Power Saving in Hearables Using Sub-Nyquist Sampling with Bandwidth Extension

**arXiv ID:** 2607.19434 | [PDF](https://arxiv.org/pdf/2607.19434v1)

**作者:** Tarikul Islam Tamiti `[一作]` (George Mason University), Anomadarshi Barua `[通讯]` (George Mason University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了CAPS框架，在耳机中采用低采样率、低位深ADC实现子Nyquist采样，并在移动端恢复高分辨率音频，实现低功耗、多模态语音增强。

**💡 创新点**

创新点包括：①同时降低采样频率和位深以显著节省功耗；②利用子Nyquist采样和多模态重建实现宽带音频恢复；③引入Mamba模块和多周期损失提升低分辨率到高分辨率映射；④在移动端实现1.36 ms的极低推理时延。

**🔧 技术方法**

使用的技术有：2D U‑Net+Mamba的Spectral Enhancement Network、HiFi‑GAN风格的Upsampling Network、Amplitude‑Phase Enhancement Network、三种损失（多周期、多尺度、相位抗包装）、ONNX→TFLite迁移与GPU delegate实现移动推理。

**📊 数据集**

使用的数据集包括：VCTK（经Whisper转文本）与MagnaTagATune音频，人工采集的同时ACM与BCM（振动加速器）数据，结合多种非语音与语音噪声。

**📈 对比分析**

在VCTK和Magna上与TFiLM、VibVoice、AERO、EBEN、HiFi++、SEANet六个基线进行对比，CAPS在参数量、推理时间（1.36 ms）上最小，在LSD、VISQOL、NISQA‑MOS、SI‑SDR、PESQ、STOI等指标上排名第一或同等，并能在150 ms延迟内实现流式处理。

**⚠️ 局限性**

限制：未考虑采样音频加密、未使用音频编解码器；实验仅在Pixel 7和Galaxy S21两款设备上验证，未覆盖极端噪声或多用户场景。

---

## 80. ChainWatch: A Kill Chain-Aligned Sequential Detection Framework for Multi-Step Attacks in MCP-Based AI Agent Systems

**arXiv ID:** 2607.19432 | [PDF](https://arxiv.org/pdf/2607.19432v1)

**作者:** Om Narayan `[一作]` (New York University), Ramkinker Singh `[通讯]` (Carnegie Mellon University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并提出了ChainWatch框架，作为MCP（Model Context Protocol）中的透明代理，监控工具调用序列并检测多步攻击链；

**💡 创新点**

创新点包括：① 将多步MCP攻击建模为六阶段Kill Chain；② 通过20维特征向量与Hidden Markov Model实现阶段分类；③ 在滑动窗口上定义五条序列检测规则，实现对跨调用、跨服务器、跨阶段攻击的实时发现；

**🔧 技术方法**

技术手段：MCP协议分析、特征提取层、20维数值特征向量、Hidden Markov Model（HMM）阶段分类、滑动窗口序列分析、规则引擎与告警策略；

**📊 数据集**

数据集：本研究仅使用文献中已知的五个攻击场景进行示例推理，未使用公开的MCPTox、MCP-AttackBench等；未来计划利用MCP‑SafetyBench收集真实MCP会话轨迹；

**📈 对比分析**

评估方法：通过对每个攻击场景的步骤进行手工推演，展示ChainWatch在每一步中如何生成阶段标签、匹配检测规则并触发警报；由于缺乏真实实验数据，尚未给出定量性能指标，结果仅以示例说明潜在检测效果；

**⚠️ 局限性**

局限性：① 需要真实、标注好的MCP会话数据来训练和校准HMM的转移概率与阈值；② 规则阈值可能导致误报，特别是对多服务企业工作流的处理；③ 当前仅在场景分析层面验证，缺乏大规模实验与性能评估。

---

## 81. Reward-Aware Population Scaling of Evolutionary Strategies in LLM Fine-Tuning

**arXiv ID:** 2607.19408 | [PDF](https://arxiv.org/pdf/2607.19408v1)

**作者:** Sung Cho `[一作]` (University of Pennsylvania), Gyubin Han `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了演化策略（ES）在大型语言模型微调中的可扩展性，聚焦于极小种群大小（N=2）时的失效原因。

**💡 创新点**

发现ES在二元奖励下小种群失败主要由奖励稀疏性与优势归一化造成，而非种群大小本身；提出奖励稀疏性阈值与归一化影响的可扩展性法则。

**🔧 技术方法**

使用ES的原始与z-score归一化估计器、零训练探测器、基于二元和交叉熵奖励的Gaussian平滑目标，并通过经验公式推导零优势概率。

**📊 数据集**

在Qwen2.5系列（0.5B/1.5B/7B）模型上，针对GSM8K和TREC两大问答/检索数据集进行实验。

**📈 对比分析**

与标准的交叉熵奖励ES（N≥1）对比，证明禁用优势归一化后，即使N=2也能显著提升准确率（如1.5B模型在GSM8K达到约55%验证准确率），而归一化版本则导致准确率迅速降至零。

**⚠️ 局限性**

局限包括：仅在Qwen2.5与GSM8K/TREC设置下验证，未证明N=2在所有模型/任务中通用；归一化与奖励稀疏性分析基于小扰动、同质批量近似；未给出完整收敛证明或全计算成本分析。

---

## 82. Marine Engine Fault Dataset: Open-Access Data under Controlled Reference and Fault Scenario Conditions

**arXiv ID:** 2607.19444 | [PDF](https://arxiv.org/pdf/2607.19444v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 83. When Reasoning Narrows the Move: Diversity Collapse in LLM Game Play

**arXiv ID:** 2607.19523 | [PDF](https://arxiv.org/pdf/2607.19523v1)

**作者:** Junyi Sha `[一作]` (Massachusetts Institute Of Technology), David Simchi-Levi `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在小型确定性棋盘游戏中，监督微调（SFT）对语言模型决策行为多样性的影响，发现常规SFT会导致行为多样性提前崩塌。

**💡 创新点**

创新点在于量化并区分了SFT引起的多样性丧失与仅因准确率提升而导致的必然压缩，并提出通过动作增强训练（action augmentation）部分缓解这一问题。

**🔧 技术方法**

采用了大型语言模型Qwen3-8B的LoRA微调、直接生成与推理模式对比、动作支持差异化（单动作vs多动作）以及基于熵与Elo评分的多维评估。

**📊 数据集**

数据集由蒙特卡洛自对弈生成的棋盘状态组成，并使用最强基线策略求解得到最优动作；每个状态提供一条或多条最优动作作为训练标签。

**📈 对比分析**

通过与基线随机/最小化搜索策略对比，利用准确率、动作熵、对战Elo与轨迹熵评估，结果显示标准SFT虽提升准确率但显著降低熵，而动作增强+推理模式在保持高准确率的同时保留了更多多样性。

**⚠️ 局限性**

局限性包括仅在小型确定性棋盘游戏上验证，缺乏对更复杂或非确定性任务的推广，且实验仅覆盖单一模型家族，未检验跨模型泛化。

---

## 84. Guardrails as Scapegoats: Auditing Unfaithful Safety Refusals in Tool-Augmented LLM Agents

**arXiv ID:** 2607.19449 | [PDF](https://arxiv.org/pdf/2607.19449v1)

**作者:** Aarushi Singh `[一作]` `[通讯]` (Independent Researcher), Aarushi Singh (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一种轻量级黑盒审计框架，用于检测工具增强型LLM代理在面对无声工具失败时的三类行为（诚实放弃、伪造、无信任安全拒绝），并通过安全语言诱导实验揭示安全词汇如何显著触发无信任安全拒绝。

**💡 创新点**

提出并量化了“无信任安全拒绝（USR）”这一此前未被正式定义的行为类别，证明其在安全语言诱导下可被放大15.6倍，并发现其对工具域（如医疗、合同、HR）高度敏感。

**🔧 技术方法**

采用工具模拟器、四种无声失败注入方案、LangChain/LangGraph代理框架、Gemini 3.1 Flash-Lite 零-shot 判别器、Python 关键词检测脚本，以及在不同规模模型（GPT‑4o、GPT‑4o‑mini、Llama 3.1‑8B、Llama 3.3‑70B）下的温度为0的推理。

**📊 数据集**

使用人工构造的30条无害自然语言提示、12个生产级邻域工具存根（CRM/HR、金融、运营/监管），以及四种无声失败配置，形成共计480条实验轨迹；结果聚合到396条有效响应。

**📈 对比分析**

在四个模型上进行对比，发现默认情况下伪造（FAR）占56.6%，USR几乎为0%；在安全提示下，USR率升至3.95%，与基线相比提升15.6倍。实验采用统计检验（Fisher精确检验、χ²检验）确认差异显著，且在不同工具域之间呈现显著工具敏感性。

**⚠️ 局限性**

限制包括：仅针对单轮交互、温度为0的确定性设置、实验依赖于人工构造的提示与工具；对多轮对话、实时部署环境的适用性未验证；误检率受关键词列表覆盖范围限制，需针对特定系统提示进行校准。

---

## 85. Intelligent Disruption: Undetectable Attacks on Wireless Autoencoders

**arXiv ID:** 2607.19448 | [PDF](https://arxiv.org/pdf/2607.19448v1)

**作者:** Han Jiang `[一作]` (Dalian University of Technology), Yunfei Chen `[通讯]` (University of Durham)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一套针对多对多 MISO 深度学习自动编码器通信系统的智能攻击框架，联合实现了基于深度神经网络的发射功率控制与基于条件生成式对抗网络的自适应扰动生成，从而在保持攻击侵入性的同时提高攻击不可被检测性和适应动态信道的能力。

**💡 创新点**

创新点在于：①首次将深度学习用于多源攻击场景下的 CLI（累计泄漏干扰）抑制，通过 DNN 学习位置到功率的映射，实现实时、低复杂度的功率调度；②采用 cGAN 将攻击通道信息作为条件输入，生成与目标信号相似的扰动，实现对动态信道的自适应攻击；③在同一框架内兼顾攻击侵袭性、不可检测性与自适应性，并通过实验验证其优越性。

**🔧 技术方法**

使用的主要技术包括：深度神经网络（多层全连接网络）用于功率控制；条件生成式对抗网络（cGAN）用于扰动生成；几何规划（GP）用于基线功率优化；Rayleigh 衰落、多径与空间相关模型用于信道生成；以及对抗训练、均方误差损失等深度学习训练技巧。

**📊 数据集**

数据集全部为仿真数据：随机生成的接收机与攻击者位置、雷达衰落模型下的信道向量、波束成形向量等，训练集约10万样本，测试集约5千样本；cGAN 的训练、验证与测试集分别为3.5k、500、1k样本。

**📈 对比分析**

与 UIAA、DLIA、FGSA、CCFA、GA 等基线方法比较，采用欧氏距离（衡量不可检测性）、攻击成功率（ASR）和 ASR 标准差（衡量自适应性）等指标。实验结果表明，本文框架在所有指标上均优于基线，尤其在 ASR 上提升约 20%，欧氏距离降低约 30%，ASR 标准差减少约 40%。

**⚠️ 局限性**

局限性包括：①依赖于完整的信道状态信息与同步假设，实际部署时可能难以获得；②训练过程需要大量仿真样本，迁移到真实环境时可能存在泛化不足；③目前仅验证了 MISO（单接收多发射）场景，对 MIMO 或多天线接收机的扩展尚未探讨；④算法的实时性虽优于传统 GP，但在极高频率动态场景下仍需进一步优化。

---

## 86. BRIM: Workload-Balanced Dual-Sided Bit-Serial Sparse Inference Accelerator

**arXiv ID:** 2607.19431 | [PDF](https://arxiv.org/pdf/2607.19431v1)

**作者:** Varun Manjunath `[一作]` (University of Southern California), Priyadarshini Panda `[通讯]` (University of Southern California)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了双侧位串行稀疏推理加速器BRIM，解决工作负载失衡问题；

**💡 创新点**

采用离线权重重塑（CBP）与运行时对偶槽位捐赠的硬件软协同机制，显著提升PE利用率并降低运算量；

**🔧 技术方法**

利用位级稀疏跳过、Radix‑2 Booth编码、Cyclic‑Balanced Pruning、Hessian误差补偿以及轻量级对偶槽位捐赠硬件；

**📊 数据集**

在ImageNet‑1K（CNN/ViT）和Wikitext‑2（LLM）上评估VGG‑16、ViT‑S/L、TinyLlama、GPT‑2 XL、OPT‑2.7B/6.7B等模型；

**📈 对比分析**

与Stripes、BitL、Laconic、Laconic‑Xbar等基线对比，ISO‑area条件下PE利用率提升至90%+，速度提升1.5–2.4×，能效提升1.1–1.7×；

**⚠️ 局限性**

低位宽（W4A8）或大组大小时权重修改率高导致精度下降，对偶槽位捐赠只能局部平衡，无法跨组重分配。

---

## 87. ChannelGuard: Safe Models Do Not Compose into Safe Multi-Agent Systems

**arXiv ID:** 2607.19430 | [PDF](https://arxiv.org/pdf/2607.19430v1)

**作者:** Elias Hossain `[一作]` (University of Central Florida), Maleeha Sheikh `[通讯]` (Purdue University Fort Wayne)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一个训练无须的、基于信息瓶颈的门控框架，将六个门放置在多代理LLM系统的每个交互通道上，用以检测并阻断跨代理传递的恶意指令；同时引入了基于句子嵌入的相似度阈值决策、三层过滤器与不确定性感知的 Monte‑Carlo 验证器，实现对攻击的即时拦截。

**💡 创新点**

创新点在于：①提供机制级的归因方法，能区分安全结果是来自云端过滤器还是应用层门控；②证明在不同后端（Azure GPT‑5、Anthropic Sonnet 4.5、Haiku 4.5）下，门控框架能够实现对攻击的同等安全性，完成机制替代；③实现了跨通道的无训练门控，避免了传统输入边界防御的局限。

**🔧 技术方法**

核心技术包括：信息瓶颈（IB）门（最大句子级余弦相似度阈值）、三层过滤器（门 + 安全判定 + 不确定性判定）、5‑样本 Monte‑Carlo 可靠性验证器、以及基于 MiniLM‑L6‑v2 的本地句子嵌入模型。

**📊 数据集**

实验使用公开数据集：HotpotQA（30题）用于攻击评估；GSM8K（30题）用于验证模型在无攻击场景下的推理准确性。攻击集共计210条，涵盖8种攻击家族。

**📈 对比分析**

对比方法：与无防御、四种单输入防御（IBProtector、Llama Guard、SmoothLLM、perplexity过滤）以及三后端对照实验。结果显示：在 Azure GPT‑5 上，攻击成功率从0.333下降到0.167（50%降低），保持GSM8K准确率0.867不变；在总攻击工作负载下，平均速度提升约1.19×（单一攻击最高可达3.30×）。

**⚠️ 局限性**

局限性包括：①对白盒自适应改写攻击和叙事式内存攻击效果有限，门控被绕过；②使用的短语库静态且有限，未覆盖所有攻击模式；③压缩（COMPRESS）策略依赖注入位置，前置注入仍可能泄漏；④评估规模有限（30题/30题），多种种子与真实工具环境的鲁棒性待进一步验证。

---

## 88. Adaptive Multi-Expert Graph Transformer for Interpretable EEG-Based Diagnostics

**arXiv ID:** 2607.19429 | [PDF](https://arxiv.org/pdf/2607.19429v1)

**作者:** Maryam Rahimimovassagh `[一作]`, Niloofar Yousefi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种基于动态功能连接图的空间多专家图变压器，用于可解释的EEG异常检测。

**💡 创新点**

创新点在于将EEG记录建模为时间序列的动态连接图，采用分层图编码捕捉电极、区域与全局信息，并通过多专家变压器结合门控机制实现子类型感知与自适应融合。

**🔧 技术方法**

使用了加权相位滞后指数（wPLI）构建动态图、图卷积网络（GCN）与SAGPool进行层级编码、TransformerEncoder进行时序建模以及多专家门控融合。

**📊 数据集**

数据集为Temple University Hospital（TUH）EEG语料库，采用80%训练/20%测试的记录级拆分。

**📈 对比分析**

与仅使用特征、单专家图模型相比，多专家门控模型在TUAB数据上准确率从0.7027提升至0.7152，宏平均精确率、召回率、F1均维持在0.71左右，异常类召回率略低。

**⚠️ 局限性**

局限性包括异常类召回率偏低、模型对优化器与学习率调度敏感、未使用解剖学先验的层级聚合、以及门控策略对异常敏感度未充分优化。

---

## 89. Learning Personalized Safety Interventions for Haptic Human-Robot Shared Control

**arXiv ID:** 2607.19534 | [PDF](https://arxiv.org/pdf/2607.19534v1)

**作者:** Dawei Zhang `[一作]` (Boston University), Roberto Tron `[通讯]` (Boston University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于差分优化的 haptic 共享控制框架，能够通过稀疏的用户手势反馈学习个性化的安全干预策略。

**💡 创新点**

创新点在于将控制障碍函数（CBF）QP 作为可微分层嵌入学习流程，直接从用户演示的力反馈中反向传播，获得个体化的响应增益；同时利用高斯过程插值把稀疏手势调整转化为连续的目标力轨迹。

**🔧 技术方法**

使用技术包括可微分优化（CVXPYLayers + PyTorch）、控制障碍函数（CBF）QP、Gaussian Process 回归、FPV 感知约束与 SDF 约束两种安全表示。

**📊 数据集**

实验数据集包括 AirSim 仿真环境（FPV 感知安全约束）和真实 DJI Tello 无人机实验（SDF 约束），每个场景均包含一次完整的遥操作轨迹和手势注释。

**📈 对比分析**

对比方法包括手工调节的基准 CBF、全局单一增益的 DiffQP、Top‑K 语义分组增益、语义加权和 GP 加权等；在 AirSim 记录轨迹上，平均误差从 0.82 降至 0.69，MSE 下降 19‑23%，显著优于基准。

**⚠️ 局限性**

局限性包括：只针对单一轨迹进行学习，缺乏跨场景泛化；需要用户在回放中手动标注，标注成本仍然较高；实验中未进行大规模用户工作负荷评估，且对非线性/自适应增益的探索尚未展开。

---

## 90. Trustworthy Privacy-Preserving Multimodal Federated Learning for Personalised Breast Cancer Prediction

**arXiv ID:** 2607.19532 | [PDF](https://arxiv.org/pdf/2607.19532v1)

**作者:** Ruth Amey `[一作]` (Nottingham Trent University), David J. Brown `[通讯]` (Nottingham Trent University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对多模态（临床表格、MRI 与 ROI 掩膜）乳腺癌患者肿瘤进展进行联邦学习模型的设计与实验，并将其性能与集中式模型进行对比。

**💡 创新点**

① 将多模态数据融合进单一联邦框架；② 在联邦学习中引入公平性加权聚合 FedGFT；③ 从安全、可扩展、透明四个维度系统性讨论部署要点，并提出数字孪生（DT）预测思路。

**🔧 技术方法**

使用 Flower 联邦学习框架、Docker 环境、基于 Transformer 的多模态融合网络、FedAvg 与 FedGFT 聚合策略；实验中概念性加入差分隐私、区块链与 TLS 等安全手段。

**📊 数据集**

I-SPY2 乳腺癌数据集（TCIA）——包含 4 个时间点的临床表格数据、MRI 扫描与 ROI 掩膜。

**📈 对比分析**

采用统一的全局测试集评估，比较 loss、表格 MAE、MRI MAE 与 ROI 准确率。联邦模型在 3 轮后达到 loss 0.7477、表格 MAE 0.2072、MRI MAE 0.3420、ROI accuracy 0.9995，显著优于集中模型的 loss 3.7253 与 ROI accuracy 0.9137；FedGFT 在 MAE 上进一步提升但总体 loss 略高。

**⚠️ 局限性**

实验受限于：数据量小、MRI 低分辨率、仅 3 台模拟客户端、未实际实现 DP/区块链/TLS 安全；表格预测严重欠拟合；公平性评估受样本不均衡影响；缺乏真实多医院部署与临床验证。

---

## 91. D3VL: Understanding Driving Scenes from 3D Time Series Data and Video with Language Models

**arXiv ID:** 2607.19528 | [PDF](https://arxiv.org/pdf/2607.19528v1)

**作者:** Heesang Han `[一作]` (Virginia Tech), Abhijit Sarkar `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 D3VL 框架，将 2D 视频与 LiDAR / stereo 深度时间序列融合进多模态大型语言模型，实现交通场景问答；并构建并扩展 WaymoQA 数据集。

**💡 创新点**

创新点：①用极简结构将 3D 深度映射为 2D 深度图并与 2D 视频共同编码，省去专用 LiDAR 处理模块；②提出 WaymoQA，提供长时序、多天气、多场景的 3D QA 任务；③微调后实现 11% 的整体准确率提升。

**🔧 技术方法**

技术手段：预训练的多模态 LLM（Qwen‑2.5‑VL），LiDAR‑to‑Camera 投影、BridgeDepth 生成深度图，图像编码器 + 视听对齐投影器，LoRA 微调，文本+视觉 token 拼接后输入 LLM。

**📊 数据集**

使用数据集：KITTI QA、WaymoQA（60K QA triplets）、NuScenesQA、LingoQA、VideoLLaMA3 等对比数据集。

**📈 对比分析**

对比方法：在 KITTI QA 上与 Qwen‑2.5‑VL（3B/7B/32B）、VideoLLaMA3、Gemini‑2.5 Flash、Jain 等方法进行零样本与微调对比。D3VL 在整体准确率上提升 11%，在各子领域（RI、VRU 等）均优于基线，最高达 97%+。

**⚠️ 局限性**

局限性：仍依赖高质量 3D 标注，WaymoQA 采用半自动方式；模型在极端天气/夜间条件下的表现未充分验证；基线在 3D 输入时性能下降，说明现有 MLLM 预训练缺乏 3D 语料；未评估在车载硬件上的实时部署与能耗。

---

## 92. Integrity of peer-to-peer distributed LLM inference under malicious nodes

**arXiv ID:** 2607.19490 | [PDF](https://arxiv.org/pdf/2607.19490v1)

**作者:** Mert Cihangiroglu `[一作]` (University of Pavia), Antonino Nocera `[通讯]` (University of Pavia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在分布式 P2P LLM 推理中，研究者通过在请求中混入已知答案的“canary”输入，预先在 FP32 下计算其各层激活值，然后实时比较各节点输出的 FP16 激活与预计算值，检测并定位恶意节点；

**💡 创新点**

创新点在于将已知答案陷阱从两方安全计算迁移至多跳推理，利用相对 L2 差异进行无阈值统计排名（AUROC）检测，避免对精确匹配的依赖；

**🔧 技术方法**

使用的技术包括：canary 插入、FP32 预计算参考、相对 L2 差异度量、AUROC 排名检测、对抗策略模拟（高斯噪声、符号翻转、缩放）以及预注册实验设计；

**📊 数据集**

实验数据集为 LAMBADA（最后词预测）和 WikiText-2（常规文本），每个数据集使用 100 条 canary 进行评估；

**📈 对比分析**

比较方法：在 408 个实验配置中与随机基线、scrambled‑label 负控、重计算上限三种参考条件对照；性能表现为 AUROC 统一达到 1.0（无论攻击策略、模型架构、数据集、pipeline 深度、节点位置或协同攻击），随机基线与负控维持 0.5，重计算上限 1.0；

**⚠️ 局限性**

主要局限包括：噪声模型简化为单一方差的高斯噪声，未充分捕捉真实硬件的结构化非高斯误差；仅在 124‑410M 参数模型上验证，未评估更大模型的表现；假设攻击者固定策略且全量攻击，未考虑可适应或针对性攻击；可被精细调节的 canary 识别可能被规避。

---

## 93. Risk-based Design for Sustainability in Cloud Systems: Insights from an Experts' Survey

**arXiv ID:** 2607.19451 | [PDF](https://arxiv.org/pdf/2607.19451v1)

**作者:** Maria Voreakou `[一作]` (Harokopio University of Athens), Mara Nikolaidou `[通讯]` (Harokopio University of Athens)

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过专家调查与主题分析，构建了基于风险的设计（RBD）框架，用以识别并评估云系统生命周期中各层面的可持续性风险及对应缓解策略。

**💡 创新点**

创新之处在于将RBD方法与云系统可持续性相结合，并将风险按应用、系统、系统集成与业务四个层级及四个项目周期（识别、分析、优先级、决策）进行分层分类，提供完整的风险管理视角。

**🔧 技术方法**

采用专家访谈调查、Google Forms问卷、主题分析技术以及风险循环模型来整理和编码风险信息。

**📊 数据集**

数据来自9名在不同企业从事高级软件工程、数据分析和系统架构的IT专家的问卷回答，结果已公开于GitHub仓库。

**📈 对比分析**

论文未与其他方法进行对比实验，主要通过对问卷结果的百分比统计展示风险分布，提供对不同风险层级和周期的量化视图；由于基于调查数据，无法给出数值性能指标。

**⚠️ 局限性**

主要限制在于样本规模有限（仅9位专家），缺乏跨行业广泛验证，且研究基于主观问卷，缺乏客观指标与实验验证。

---

## 94. REGEN: Replay-recycling for Expert-to-Generalist distillation with Offline Reinforcement Learning

**arXiv ID:** 2607.19450 | [PDF](https://arxiv.org/pdf/2607.19450v1)

**作者:** Yunjie Chen `[一作]` (vivo), Fang Wang `[通讯]` (vivo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 REGEN 方法，利用专家 RL 训练过程中的 replay 内存进行离线 RL，训练泛化型 LLM。

**💡 创新点**

创新点在于将专家的 replay 作为离线数据集，配合类别平衡采样和截断重要性采样的优势加权目标，完全解耦采样与训练，显著降低成本。

**🔧 技术方法**

使用离线强化学习（GRPO、DPO、TOPR）、截断重要性采样、优势归一化、类别平衡采样以及行为克隆、MOPD 对照基线。

**📊 数据集**

训练数据来自 GSM8K、MATH、KodCode-Light-RL-10K 与 IFEval 版指令数据，评测涵盖 GSM8K、MATH、HumanEval、MBPP 与 IFEval。

**📈 对比分析**

与 BC、MOPD 在相同 140K 语料下对比，REGEN 与 MOPD 准确率相当但训练速度提升约3倍，且在 Math、Code 上与 MOPD 接近、优于 BC。

**⚠️ 局限性**

局限性包括在 HumanEval 上仍落后专家教师、离线奖励导致监督较粗、负样本被 CBS 丢弃可能损失信息、对不同生成阶段的 replay 处理仍不充分。

---

## 95. BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators

**arXiv ID:** 2607.19438 | [PDF](https://arxiv.org/pdf/2607.19438v1)

**作者:** Fabian Waschkowski `[一作]` (Base Compute), Lukas Wesemann `[通讯]` (Base Compute)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现并发布了BaseRT推理运行时，针对Apple M5 GPU的神经加速器编写了手写的Metal 4张量核算子，实现了高效的推理加速。

**💡 创新点**

创新点在于将M5每核Neural Accelerator的张量核心功能通过Metal 4 API直接调用，针对计算密集型前缀填充（prefill）阶段实现了显著吞吐提升，并且仅在计算绑定阶段使用张量核，保持解码（decode）阶段的低开销。

**🔧 技术方法**

使用技术包括：Apple M5 GPU、Metal 4张量 API、手写的cooperative-tensor算子、工作负载感知调度、统一内存优化与算子融合。

**📊 数据集**

评估数据集覆盖15个模型配置，涵盖Qwen3、Qwen3.5/3.6、Llama 3.2、Gemma 4等系列，参数量从0.6B到35B不等，均采用4/8-bit量化权重。

**📈 对比分析**

与同机型的llama.cpp（Metal后端）和MLX基准进行对比，BaseRT在prefill吞吐率上最高可提升6.4×（对llama.cpp）或3.9×（对MLX），在decode阶段相较对手提升1.02–1.75×，表现最显著的提升主要体现在混合专家模型的计算密集型前缀填充。

**⚠️ 局限性**

限制包括：仅在M5系列硬件和Metal 4工具链下可用；decode吞吐受内存带宽限制，张量核对其无提升；单设备单用户推理，未实现批量、多请求并行或张量并行；评估仅在单台M5 Pro上完成，未覆盖M5 base/Max 等变种。

---

## 96. Synthetic and Derived Training Images for Campus Waste Detection: A Multi-Seed Evaluation with YOLOv8n

**arXiv ID:** 2607.19535 | [PDF](https://arxiv.org/pdf/2607.19535v1)

**作者:** Ali Behbahani `[一作]` (University of Tennessee), Phouvadeth Vathana `[通讯]` (University of Tennessee)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估合成与派生图像对YOLOv8n在校园垃圾投放点的物体检测效果，并系统比较不同合成来源、数量、手部合成以及顺序预训练与联合混合的影响。

**💡 创新点**

在有限校园数据上，首次将多种合成策略与手部合成、顺序预训练等方案在同一模型与同一测试集上进行对照，提供了对合成数据实用性与部署可行性的综合评估。

**🔧 技术方法**

使用YOLOv8n nano模型、COCO预训练、四个随机种子训练、bootstrap区间统计、背景替换、光照变换、物体裁剪、手部与前臂合成、INT8量化以及四线程CPU基准。

**📊 数据集**

数据集包括148张校园实拍照片（86/31/31分割，四类）以及695张基于100个基准物体生成的合成/派生图像。

**📈 对比分析**

在固定的31张测试集上计算mAP@0.5和mAP@0.5:0.95，并通过四种种子产生bootstrap区间；结果显示无任何合成配置能超过纯实拍基线，背景替换最差，隔离物体略优，手部合成无显著提升，顺序预训练对不同来源效果相反；FP32导出模型12.3 MB，延迟332–359 ms/图；INT8导出失效。

**⚠️ 局限性**

局限性包括：测试集极小（仅两张玻璃样本），未能保证训练、验证、测试中物体不重复，合成来源与训练量混淆，手部合成样本极少，预训练与顺序混合实验仅单一种子，条件子集重叠并含验证图像，边缘基准基于桌面CPU非目标单板，INT8导出失败原因未定位。

---

## 97. LENS: LLM-guided Environment Simplification for Planning and Control in Clutter

**arXiv ID:** 2607.19633 | [PDF](https://arxiv.org/pdf/2607.19633v1)

**作者:** Aileen Liao `[一作]` (University of Pennsylvania), Michael Posa `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大语言模型的环境简化框架，利用闭环反馈自动对场景进行剪枝和合并，生成任务相关的抽象场景，并将其作为前端集成到任务与运动规划（TAMP）、基于模型的控制（C3+）以及视觉语言动作模型（π_0.5）中。

**💡 创新点**

通过多模态大型语言模型实现任务无关的闭环场景抽象，自动识别并剔除/合并非任务相关对象，从而大幅降低组合复杂度并消除手工设计规则的需求。

**🔧 技术方法**

使用GPT‑4o等视觉语言模型进行推理；闭环重提示机制；剪枝（pruning）与合并（merging）操作；并将该抽象层集成至TAMP、C3+控制器与π_0.5 VLA。

**📊 数据集**

轻/重/堆叠桌面环境、Push Anything 规划基准、Spatial LIBERO 视觉语言动作基准，以及相应的实物硬件实验数据。

**📈 对比分析**

与基线（未抽象的全场景）在三种堆栈中对比，TAMP成功率从约0.5提升至约0.8，C3+控制器执行时间从数千秒降至≈100秒以内，VLA成功率从0.5提升至0.69；整体显示更高成功率、更快运行和更好的可扩展性。

**⚠️ 局限性**

依赖高层失败信号而非具体原因，无法跨运行学习经验；抽象操作仅限于剪枝/合并；对某些控制器的验证在仿真中计算成本高；在分布偏移下仍存在性能损失。

---

## 98. Adaptive Capitulation: A Structural Failure Mode of LLM Responses in Vulnerability Contexts

**arXiv ID:** 2607.19629 | [PDF](https://arxiv.org/pdf/2607.19629v1)

**作者:** Eunna Lee `[一作]` `[通讯]`, Eunna Lee

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在Claude 4.6、GPT 5.4和Grok 4.1三款商业LLM上执行900个含情绪脆弱性与外部归因的三轮对话实验，研究其在情绪敏感情境下的响应模式。

**💡 创新点**

提出“适度再归因充分性”（MRS）原则，旨在以最小干预方式在不牺牲用户自治的前提下解决保护、促进与一致性三难困境。

**🔧 技术方法**

采用实验设计、诊断性情景、双重编码（VCC/VCI）以及统计检验（Pearson χ²与Fisher精确检验）对LLM响应进行量化与比较。

**📊 数据集**

使用人工设计的三类情境（物质地位、关系地位、身体改造）共900条会话（每类100条），并以三款模型为测试对象。

**📈 对比分析**

结果显示Claude在所有情境下VCI为0，GPT和Grok虽保持高VCC但VCI显著高值；MRS被视为在任何配置下提供最小有效干预的潜在方案。

**⚠️ 局限性**

局限包括缺乏自然交互验证、性别推断偏差、MRS触发机制自动化实现难度以及文化差异对归因功能映射的影响。

---

## 99. HypEMBER: Hypernetwork-based Ensemble for Robust Policy Learning of Parametrized Dynamical Systems

**arXiv ID:** 2607.19628 | [PDF](https://arxiv.org/pdf/2607.19628v1)

**作者:** Nicolò Botteghi `[一作]` (Politecnico di Milano), Andrea Manzoni `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种名为 HypEMBER 的强化学习框架，利用超网络和集成学习对参数化动力系统进行鲁棒控制，并提出不确定性感知动作选择策略；

**💡 创新点**

将超网络与集成鲁棒学习相结合，动态生成以系统参数为条件的策略与价值网络权重，并用价值网络集成的不确定度进行加权更新和基于 UCB/UA 的动作决策；

**🔧 技术方法**

使用超网络（hypernetworks）、集成强化学习（SUNRISE）、深度 actor‑critic（SAC/TD3）、不确定性估计、Huber 损失、温度门控等技术；

**📊 数据集**

在两个数值基准上进行实验：一是参数化 Kuramoto‑Sivashinsky 方程（N_x=64 网格、8 个传感器），二是双涡流中粒子导航（5 维观测），并在测量噪声、参数失配及其组合下进行评估；

**📈 对比分析**

与 SUNRISE、HypeRL、TD3 和 PolyL0‑TD3 在理想和不确定性环境中进行对比；HypEMBER 在训练收敛稳定性和样本效率上与 HypeRL 相当或略优；在噪声/参数失配测试中，HypEMBER 获得最高累计奖励，表现出最强的鲁棒性；

**⚠️ 局限性**

仅在两种仿真基准上验证，缺乏真实实验或更复杂系统的测试；对超网络结构和超参数敏感；在极高噪声或大参数偏移下性能仍会下降，需要较大计算资源训练集成模型。

---

## 100. On the Computational Complexity of Structural Generalization

**arXiv ID:** 2607.19573 | [PDF](https://arxiv.org/pdf/2607.19573v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 101. Task Competence Is Not Instruction Following: Evaluating Instruction-Conflicting Behavior in Small Language Models

**arXiv ID:** 2607.19608 | [PDF](https://arxiv.org/pdf/2607.19608v1)

**作者:** Mahdiyeh Farajidizaji `[一作]` (Khajeh Nasir Toosi University of Technology), Vatsal Raina `[通讯]` (Apta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了指令调优语言模型在面对与既有任务目标冲突的非标准指令时的遵从性，采用多任务（MCQA、情感分类、数学问答）评估；

**💡 创新点**

提出了Instruction‑Following Failure Rate (IFFR) 量化指标，并通过规模化实验揭示小模型易忽略冲突指令、规模越大遵从性越好；

**🔧 技术方法**

使用Qwen3.5系列指令调优模型，采用标准与非标准指令两种对照，结合确定性（greedy）解码、选项/标签闭集argmax与数值提取；

**📊 数据集**

在RACE‑Middle、ARC‑Challenge、OpenBookQA、Multi‑class Sentiment、Rotten Tomatoes、FinancialPhraseBank、MAWPS、Calc‑asdiv‑a、MultiArith等九个公开数据集上进行实验；

**📈 对比分析**

通过标准准确率、非标准准确率和IFFR比较模型在不同规模（0.8B–27B）下的任务能力与指令遵从性，发现标准准确率随规模提升，非标准准确率和IFFR随规模下降，表明大型模型更能遵从冲突指令；

**⚠️ 局限性**

仅评估Qwen3.5系列模型、仅涉及三类任务、仅使用单一冲突指令形式，缺乏对其他模型家族、开放式生成任务和多样化冲突指令的验证。

---

## 102. A Splitting Architecture for Exact Reduced Coulomb Friction

**arXiv ID:** 2607.19599 | [PDF](https://arxiv.org/pdf/2607.19599v1)

**作者:** Hongcheng Song `[一作]`, Dinesh K. Pai `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

无法获取论文内容

**💡 创新点**

无法获取

**🔧 技术方法**

无法获取

**📊 数据集**

无法获取

**📈 对比分析**

无法获取

**⚠️ 局限性**

无法获取

---

## 103. Knowledge-Centric Self-Improvement

**arXiv ID:** 2607.19592 | [PDF](https://arxiv.org/pdf/2607.19592v1)

**作者:** Xuefei Julie Wang `[一作]`, Yisong Yue `[通讯]` (Caltech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了知识中心自我改进范式，通过持续维护共享知识库来提升代理性能，保持代理无状态可抛弃；

**💡 创新点**

创新点在于将自我改进焦点从代理本身转移到可共享、可审计的知识库，并通过三阶段论坛与蒸馏协议实现知识累积与迁移；

**🔧 技术方法**

采用LLM代理（Claude Haiku 4.5、GPT‑5.4‑mini）实现任务求解、论坛讨论与知识蒸馏，结合工具调用与元学习框架；

**📊 数据集**

使用ARC‑AGI‑1/2、Polyglot、SWE‑bench Pro、Terminal‑Bench 2等四大基准任务集；

**📈 对比分析**

与传统agent‑centric自我改进（Darwin Gödel Machine、HyperAgents）以及提示优化方法（GEPA、OpenEvolve）进行对比，结果显示知识蒸馏方案在解决率上显著优于基线且成本更低，并在不同LLM家族及保留任务上实现了跨任务迁移；

**⚠️ 局限性**

局限性包括：依赖简单代理和固定协议，未探索更复杂的递归搜索或规划交互；协议并非最优，难以处理极大规模任务；实验范围受限于选定基准，缺乏人类专家介入验证。

---

## 104. FullPASS: Geometry Optimization for Full-Duplex Pinching-Antenna Systems

**arXiv ID:** 2607.19546 | [PDF](https://arxiv.org/pdf/2607.19546v1)

**作者:** Morteza Barzegar Astanjin `[一作]` (Chalmers University of Technology), Mikko Valkama `[通讯]` (Tampere University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种双波导Pinching‑Antenna系统（FullPASS）实现同频全双工通信，并设计了通过选择激活的压缩点（PE）来同时提升下行、上行链路质量并抑制自干扰的二进制优化方案。

**💡 创新点**

创新点在于：1）构建了几何驱动的传输与自干扰耦合模型，将自干扰抑制视为与链路质量相互耦合的离散选择问题；2）提出了阶段化的相位锚定SOCP松弛+确定性四舍五入初始化，结合ABLS局部搜索，既保留了近似最优性，又实现了可扩展的求解。

**🔧 技术方法**

技术方法包括：几何基线LOS传播模型、累计穿透衰减与传输/接收耦合系数、相位锚定的二阶锥（SOCP）放松与确定性圆整、局部改进的ABLS搜索、以及预先校准的安全裕度以弥补简化模型与实际模型之间的SI偏差。

**📊 数据集**

使用的数据集为仿真生成的150个随机双向用户场景，结合不同候选密度（13×13、15×15、到60×60）以及多组系统参数（频率、波导长度、衰减、耦合比等）进行评估。

**📈 对比分析**

与穷举搜索和单独ABLS基线比较：在13×13/15×15网格上平均仅比穷举搜索低0.95%，运行时间降低10倍；相较同硬件半双工系统，FullPASS实现约90%–98%的双向谱效率提升。

**⚠️ 局限性**

局限性：依赖完美CSI且忽略多径、时变SI；对强耦合或严格SI抑制时SOCP初始化效果下降；需要预先校准安全裕度，且对非LOS或高度动态环境的鲁棒性尚未验证。

---

## 105. Packing Linear Programs and Fractional Knapsack using Comparison Oracles

**arXiv ID:** 2607.19557 | [PDF](https://arxiv.org/pdf/2607.19557v1)

**作者:** Ritabrata Barat `[一作]` (Indian Institute of Science), Sukruta Midigeshi `[通讯]` (Microsoft Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种仅利用比较信息（即不同约束下最优解之间的比较）来恢复分数背包问题及其推广的包装线性规划（Packing LP）中的目标系数的方法；

**💡 创新点**

其创新点在于将传统逆优化和揭示偏好模型提升到仅依赖序数反馈的比较oracle，并通过构造“三元价格”与二分搜索等技术实现对价值比例的高效恢复；

**🔧 技术方法**

主要技术包括：对分数背包的贪婪结构进行解析、设计特定的价格向量（如一维扰动与三元价格）来诱导阈值行为、利用二分搜索与连续分数方法恢复数值、以及通过通信复杂度下界证明的指数信息量限制；

**📊 数据集**

论文为理论分析，没有使用任何实验数据集；

**📈 对比分析**

相较于传统需要完整最优解观测的逆优化方法，本文的比较oracle方案在查询复杂度上实现了 O(n log(1/δ)+B²) 的多项式上界，并通过通信下界证明该上界在除线性因子之外已基本最优；

**⚠️ 局限性**

局限性包括：仅适用于目标系数互不相同且在已知格点上、容量约束满足一定限制的包装LP；算法不支持噪声或不确定的比较反馈；以及在更一般的线性/凸优化问题中尚未验证其可行性。

---

## 106. Do Co-Located AI Training Jobs Synchronize? Load-Dependent Throttling as a Coupling Mechanism for Phase-Locking Behind a Shared Power Cap

**arXiv ID:** 2607.19638 | [PDF](https://arxiv.org/pdf/2607.19638v1)

**作者:** Brieuc Le Roux Tardif `[一作]` `[通讯]` (IMT Nord Europe), Brieuc Le Roux Tardif (IMT Nord Europe)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究AI训练作业在共享电源下的相位同步现象，推导其能否导致整体功率波动从 √N 放大到 N ；

**💡 创新点**

创新点包括：①从负载可调节（功率限额）出发构建耦合机制并映射为延迟的Kuramoto模型；②发现耦合系数随控制延迟呈 −sin(ωτ) 变化，短延迟下为排斥，超过 τ*=T/2 转为吸引；③揭示频率相关相位滞后导致同步阈值为 2/(πg(0)a1) 且对多样化负载呈第一阶跃跃；④指出高阶谐波锁定的条件与阈值；

**🔧 技术方法**

使用非线性振荡器理论（Kuramoto、Sakaguchi–Daido、Ott‑Antonsen）、时域延迟微分方程求解与平均化分析、数值仿真（RK4、傅里叶逼近）等技术；

**📊 数据集**

未使用真实数据，而是以理论公式和公开的GPU功率/计算模型合成两级功率波形与阈值；

**📈 对比分析**

通过对比无耦合基线（r≈0，波动≈√N）与不同延迟、耦合强度、速率多样性的仿真，验证了耦合符号、临界耦合、阶跃跳变、谐波锁定等预言；表现为在排斥窗口中聚合波动保持低于无耦合水平，而在吸引窗口中波动按 r√N 比例放大；

**⚠️ 局限性**

局限性包括：①假设均值场全连通耦合，忽略真实数据中心的分层电源拓扑；②只保留波形的第一谐波（以及有限谐波）且采用两级模型；③弱耦合近似与软限幅假设，强限幅或非线性阈值超出分析范围；④无机械惯性与噪声影响，实际迭代时间可能漂移、噪声；⑤结论需通过实验验证，未提供真实功率测量与调度实现细节。

---

## 107. Examining QRMI as a Unified Interface for Quantum-HPC Integration

**arXiv ID:** 2607.19591 | [PDF](https://arxiv.org/pdf/2607.19591v1)

**作者:** Thomas Badts `[一作]` (Pasqal), Aleksander Wennersteen `[通讯]` (Pasqal)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在 Slurm、PBS、LSF、Grid Engine、Kubernetes 与 Flux 等工作负载管理器上实现并验证了 QRMI（Quantum Resource Management Interface）的跨平台集成，展示了其作为量子资源统一调度与管理的可移植性。

**💡 创新点**

创新点包括：①将量子资源作为第一类可调度资源并统一暴露标准 API；②提出薄层生命周期映射（acquire‑execute‑release）模式，兼容多种调度器的钩子与插件；③在不同调度器上实现统一配置与中间件封装，降低对调度器代码的侵入；④讨论并尝试解决动态资源可用性、共调度与凭证管理等关键难题。

**🔧 技术方法**

技术实现主要使用 QRMI 语言绑定（Rust/C/Python）以及各调度器的插件/钩子（Slurm SPANK、PBS Hook、LSF ELIM、Grid Engine Load Sensors、Kubernetes Operator、Flux Jobtap）。同时使用资源发现、会话管理、token 传递、环境变量注入等机制。

**📊 数据集**

数据集与测试场景来自实际 HPC 集群与量子设备：Pasqal Sol、IBM Quantum System One/Two、云量子设备等；并在 CINECA、BasQ、RPI、英国 HPC 生态等多站点进行部署验证。

**📈 对比分析**

通过在多站点的实际部署与运行，本文对比了不同调度器的集成复杂度、资源可用性识别、调度性能等，但未给出统一的量化指标，主要采用定性评估与案例分析，指出共调度导致传统 CPU/GPU 资源闲置等现象。

**⚠️ 局限性**

限制包括：①集成仍处于概念验证阶段，缺乏原生调度器支持；②动态量子资源可用性识别在多调度器中的实现不一致；③凭证与安全管理缺乏统一规范；④共调度导致传统资源闲置；⑤插件/钩子实现对调度器版本与配置高度依赖，维护与部署成本高。

---

## 108. End-to-End Differential Privacy in Training Deep Neural Network Classifiers

**arXiv ID:** 2607.19580 | [PDF](https://arxiv.org/pdf/2607.19580v1)

**作者:** Huaiyuan Rao `[一作]` (Georgia Institute of Technology), Matthew Hale `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种在保持标签公开的情况下，对训练输入进行差分隐私保护的深度神经网络训练框架。

**💡 创新点**

通过在softmax层使用Dirichlet机制对输出进行随机化，并利用采样放大实现端到端差分隐私，仅对输入加密，同时提供基于RDP的精确隐私分析。

**🔧 技术方法**

Dirichlet机制、RDP隐私分析、子采样放大、NLL损失、无样本梯度裁剪。

**📊 数据集**

CIFAR-10、MNIST、DermaMNIST、Fashion MNIST、SVHN。

**📈 对比分析**

与DP‑SGD及先前工作比较，实验显示在相同隐私预算下准确率提升约20‑30%（例如CIFAR‑10 ϵ=1从56%提升到83%），并在强隐私下仍保持高精度。

**⚠️ 局限性**

仅适用于有softmax输出的分类任务；需要先预先计算r*，并假设标签不敏感；对极小类别数或高度稀疏标签可能效果有限。

---

## 109. Fine-grained Computation-Communication Overlap via Tile-level Signaling and Scheduling for Mixture-of-Experts

**arXiv ID:** 2607.19539 | [PDF](https://arxiv.org/pdf/2607.19539v1)

**作者:** Minyu Cui `[一作]` (Linnaeus University), Morgan Ericsson `[通讯]` (Linnaeus University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种细粒度的计算‑通信重叠机制，专门针对 Mixture‑of‑Experts（MoE）层的返回路径（第二个 all‑to‑all 通信），通过 tile‑level 信号和调度将专家计算与通信交错执行，从而显著提高多 GPU 上 MoE 推理的效率。

**💡 创新点**

创新点包括：
- 远程所有者对齐的行布局（remote‑owner‑aligned row layout），保证每个完成的输出 tile 只对应单个目标 GPU，消除 per‑row 路由查找；
- 持久化的 rank‑wide GEMM 生产者核，覆盖所有本地专家，避免多次 kernel 启动开销；
- 持久化的通信消费者核，采用 NVSHMEM 一侧写入，以 tile‑level 设备驻留信号触发，支持细粒度的 segment‑级传输；
- 在 SM 级别上对生产者与消费者进行分区，配合 stream‑级优先级实现无阻塞重叠；
- 通过细粒度信号与调度而非粗粒度拆分或完整融合实现更高的张量核心利用率与更低的同步开销。

**🔧 技术方法**

技术手段：
- CUTLASS GEMM 核，利用 epilogue 进行 tile‑level 信号发射；
- NVSHMEM 一侧 put API 进行 device‑initiated all‑to‑all；
- 持久化（persistent）核技术，避免 host‑side 启动/销毁成本；
- SM 分区与 CUDA stream 优先级调度；
- 细粒度 segment 调度策略（首尾单块，中间可拼接多块），匹配通信带宽饱和区；
- 远程所有者对齐行布局与 tile‑schedule 算法。

**📊 数据集**

评测使用了三种流行的 MoE 模型：
- M‑GPT（在第 11 层使用 MoE，12 层 Transformer），
- M‑BERT（第 2、5、8、11 层使用 MoE），
- M‑Trans‑XL（所有 12 层均使用 MoE）。
每个模型在多种 token 数、GEMM 形状、路由分布（balanced、moderate_skew、stress_skew）以及专家数量（4–64）下进行测试，硬件为 4 台 NVIDIA A100 GPU（108 SM/40 GB HBM，NVLink）。

**📈 对比分析**

对比方法：
- 传统顺序 MoE（先专家 compute 再全部通信用 NCCL），
- 现有四个最先进 MoE 框架：FasterMoE、Tutel、Megatron‑CUTLASS、Megatron‑TE。  
结果：
  - 在 M‑GPT 与 M‑BERT 上，整体前向推理速度提升 1.57–1.66 倍；
  - 在 M‑Trans‑XL 上，速度提升 2.64 倍；
  - MoE 层级加速高达 2.74 倍；
  - 细粒度重叠的重叠率持续保持在 72–100% 之间；
  - 通过 SM 分区实验表明，在 10–20 个 SM 分配给通信时可获得最低延迟。

**⚠️ 局限性**

局限性：
- 仅实现了推理（forward）阶段，训练（backward）尚未支持；
- 需要手动调节 SM 分区与 segment 大小，缺乏自动化调度器；
- 依赖 NVSHMEM 以及 CUTLASS，适用性受限于支持这些库的 GPU 环境；
- 对极端 skew 情况下仍会产生通信尾部；
- 对小隐藏维度模型（如 M‑Trans‑XL）隐藏空间不足，重叠收益相对有限；
- 设计涉及多层级调度与持久化核，工程实现相对复杂。

---

## 110. Pathologist Attention-Aligned Report Generation for Prostate Histopathology

**arXiv ID:** 2607.19624 | [PDF](https://arxiv.org/pdf/2607.19624v1)

**作者:** Ruoyu Xue `[一作]` (Stony Brook University), Dimitris Samaras `[通讯]` (Stony Brook University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

收集多尺度病理学家注视轨迹数据，构建路径专家注意力数据集，并将该注意力作为辅助监督，在报告生成模型中加入注意力对齐损失，提升病理报告质量。

**💡 创新点**

首次将病理学家视觉注意力引入自动报告生成，提出可插拔的注意力对齐模块，并通过少量注释（约7小时）即可显著提升模型性能。

**🔧 技术方法**

基于 Transformer 的自注意力（WSI‑LLaVA）与交叉注意力（HistGen）架构，使用 LoRA 微调，采用 KL 散度将模型注意力与人类注意力对齐，并在推理时生成与专家注视更一致的热力图。

**📊 数据集**

121 张 TCGA‑PRAD 前列腺全切片图像，标注了五个临床关键组件（Gleason 评分、周围神经侵犯、腔管内癌、肿瘤外扩、手术切缘），并同步记录视窗轨迹与口头描述。

**📈 对比分析**

与零样本、仅使用下一个词损失的 LoRA 微调模型对比，在 HistGen 和 WSI‑LLaVA 上分别取得 NLP 指标平均提升 10.9% 以及临床组件准确率平均提升 19.3%。

**⚠️ 局限性**

仅适用于前列腺癌，数据集规模有限；对深层模型多层对齐会导致性能下降；需要人工注视轨迹采集，虽成本低但仍需专家时间；对其它癌症类型的迁移性待进一步验证。

---

## 111. Leveraging ECRAM for Edge Continual Learning

**arXiv ID:** 2607.19661 | [PDF](https://arxiv.org/pdf/2607.19661v1)

**作者:** Nabila Tasnim `[一作]` (University of Illinois Urbana-Champaign), Saugata Ghose `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了CLASP系统，利用ECRAM实现边缘持续学习；

**💡 创新点**

创新点在于采用ECRAM避免传统IMC的非理想性，硬件软件协同调度并提供可插拔的RISC‑V指令接口，支持多种持续学习算法；

**🔧 技术方法**

使用的技术包括IMC加速、ECRAM交叉阵列、RoCC定制指令、8位量化、ReLU稀疏计算、软件可扩展的学习指令集；

**📊 数据集**

使用MNIST数据集进行实验；

**📈 对比分析**

与GPU+PyTorch基线比较，CLASP 8‑bit ECRAM在LwF下准确率51.43%，Replay下84.20%，速度提升约100×/34×，能耗降低约152×/112×；

**⚠️ 局限性**

局限性包括模型规模有限、GPU启动开销大、ECRAM制造变异仍影响精度、CPU侧数据处理仍为瓶颈，仅验证了两种持续学习算法。

---

## 112. Domain Shift in Echocardiography: Interpretable Quantification and Prediction of Cross-Dataset Left Ventricular Segmentation

**arXiv ID:** 2607.19643 | [PDF](https://arxiv.org/pdf/2607.19643v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 113. Expert-Guided Forecast Editing for Time-Series Foundation Models

**arXiv ID:** 2607.19659 | [PDF](https://arxiv.org/pdf/2607.19659v1)

**作者:** Hung Le `[一作]` (Deakin University), Dai Do `[通讯]` (Deakin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种在时间序列基础模型（TFM）上进行专家引导的预测编辑框架DEFT，利用少量专家评估对冻结模型的预测进行改进。

**💡 创新点**

创新点在于将预算划分为“先利用模型先验再在分解的趋势–季节空间内探索”，通过组件级的max‑pool信息重用，实现了在有限查询预算下的高效利用。

**🔧 技术方法**

技术主要包括：时间序列的移动平均分解（trend/seasonal），基于专家得分的组件级最大池化，分两阶段的搜索（先抓取高质量样本再使用分解CEM迭代），以及对不同专家反馈形式（评分、对比、适配）的一致处理。

**📊 数据集**

实验使用了78个时间序列数据集，涵盖ChronosBench（42个）和GIFT‑Eval（36个）以及一个分子动力学数据集（MD），并分别在TimesFM、Chronos、Moirai三种TFM上评估。

**📈 对比分析**

与传统最佳‑of‑N、随机搜索、CEM、surrogate‑CEM、TuRBO等基线在匹配查询预算下比较，DEFT在MASE、WQL、MAE、MSE等指标上普遍领先，且在超过90%的设置中击败零样本中位数及随机搜索。

**⚠️ 局限性**

局限性包括：实验中专家评分为基于未来真实值的oracle，缺乏对真实人类或约束检查器噪声与偏差的考量；分解假设错误主要为趋势或季节误差，可能不适用于剧烈变迁或非周期性序列；以及始终保持模型冻结，未探讨与轻量级模型微调的结合。

---

## 114. PerfAgent: Profiler-Guided Iterative Refinement for Repository-Level Code Optimization

**arXiv ID:** 2607.19653 | [PDF](https://arxiv.org/pdf/2607.19653v1)

**作者:** Ryan Deng `[一作]` (Massachusetts Institute of Technology), Jatin Ganhotra `[通讯]` (IBM)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了一个面向仓库级代码优化的“PerfAgent”工作流，利用 LLM 代理在迭代循环中持续改进性能。

**💡 创新点**

创新点包括：①针对缺失真实瓶颈、过早停止和测试不足三大失败模式的专门设计；②使用经过精炼的采样分析器摘要（py‑spy + LLM 文本摘要）精准定位热点；③引入目标驱动的循环控制器，强制代理在每次提交后重新编译、测试并记录最快正确补丁；④在每轮后仅执行与变更相关的精选单元测试（pytest‑testmon）以降低成本并避免回归。

**🔧 技术方法**

核心技术包括：Llama / GPT‑5.1/ Kimi‑K2 语言模型、Mini‑SWE‑Agent 运行时、py‑spy 采样分析器、LLM 生成的热点摘要、目标驱动控制器、pytest‑testmon 选择性测试、速度比 SR 计算与奖励机制。

**📊 数据集**

实验数据集：GSO（102 任务）和 SWE‑fficiency‑Lite（100 任务），涵盖 Python、Cython、C/C++、Rust 等多语言仓库；使用公开 benchmark 的隐藏性能/正确性测试。

**📈 对比分析**

与 OpenHands、Codex 以及最佳‑5（oracle）基线对比。GPT‑5.1 版在 GSO 的 Opt@1 从 19.6% 提升至 39.2%（+100%），在 SWE‑fficiency‑Lite 从 26% 提升至 74%（+188%）；在成本上，比 oracle best‑5 少 2–3 倍。Kimi‑K2 版在两 benchmark 上也分别提升 1.5×。

**⚠️ 局限性**

局限性：
• 主要验证于 Python 仓库并依赖本地扩展；在非 Python 或更大规模多语言项目上的效果未知。
• 选择性测试只覆盖 Python 代码，可能漏掉 native‑extension 的回归。
• 仍存在奖励破解风险，需要更强的防御机制。
• 对预算、迭代次数敏感，过高预算可进一步提升，但会增加成本。
• 依赖 Mini‑SWE‑Agent 运行时，迁移到其它 harness 需额外适配。

---

## 115. Black-Box Optimization for Identifying and Inverting Audio Dynamic Range Control Effects

**arXiv ID:** 2607.19645 | [PDF](https://arxiv.org/pdf/2607.19645v1)

**作者:** Haoran Sun `[一作]` (University of Évry Paris-Saclay), Hichem Maaref `[通讯]` (University of Évry Paris-Saclay)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于黑盒优化的动态范围压缩（DRC）参数反演方法，通过在感知驱动的特征空间中最小化重构信号与干信号特征的距离来实现盲参数估计。

**💡 创新点**

创新点在于将DRC参数估计转化为非可微的黑盒优化问题，利用非线性、非可微的动态直方图特征作为目标，摆脱对梯度信息的依赖，兼顾模型驱动与学习驱动的优势。

**🔧 技术方法**

采用的技术包括动态直方图特征提取、模式搜索（Pattern Search）和贝叶斯优化（Bayesian Optimization）两种无梯度优化算法，以及先验估计（MEE）作为初始化与正则化。

**📊 数据集**

实验数据集为MedleyDB，包含5个音乐流派（古典、摇滚、电子、流行、爵士）共25段干信号，使用30种预设的压缩与扩展配置生成750个处理样本。

**📈 对比分析**

与MEE直接估计、CleanUMamba端到端模型以及anchor（固定平均参数）基线进行对比，结果显示Pattern Search在压缩任务中最优，贝叶斯优化在扩展任务中性能最佳；两种方法均显著优于基线，在特征空间距离和参数均方误差上取得更好成绩。

**⚠️ 局限性**

限制主要包括：对扩展任务的恢复效果仍不如压缩；黑盒优化计算量较大（Pattern Search耗时高）；依赖于预先定义的干信号特征参考点，可能受数据分布影响；以及在极端参数或噪声条件下的鲁棒性待进一步验证。

---

## 116. CRB-Driven Beamforming and Trajectory Optimization for UAV-assisted ISAC System

**arXiv ID:** 2607.19609 | [PDF](https://arxiv.org/pdf/2607.19609v1)

**作者:** Yi Yang `[一作]` (Rowan University), Huaxia Wang `[通讯]` (Rowan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种无人机（UAV）辅助的集成感知与通信（ISAC）系统，通过联合优化UAV轨迹和波束成形，以最小化平均Cramér–Rao bound（CRB）来提升目标角度估计精度，并满足用户的信噪比（SINR）要求。

**💡 创新点**

创新点在于：①将UAV的反向噪声（AN）用作感知增强而非仅作干扰；②采用零空间投影先行消除UAV–基站干扰，再用半正定松弛与高斯随机化得到可行波束；③使用软演员-评论家（SAC）深度强化学习对UAV轨迹进行在线优化，兼顾感知与通信约束。

**🔧 技术方法**

技术包括：Cramér–Rao bound分析、零空间投影与半正定松弛（SDR）、高斯随机化、软演员-评论家（SAC）强化学习、离散时间轨迹建模。

**📊 数据集**

实验采用仿真参数：基站8个天线、UAV 8个天线、目标RCS 4 m²、噪声功率–40 dBm、UAV最大速度1 m/s、不同UAV功率、天线数和SINR阈值。

**📈 对比分析**

与四个基准（无UAV、最短路径+波束优化、最短路径+MRT波束、MRT+DRL）相比，本文方法在所有情形下均显著降低平均CRB（提升≈10%），且收敛所需训练轮次更少，说明在感知精度与通信约束之间实现了更优平衡。

**⚠️ 局限性**

局限性包括：仅考虑单目标单用户场景；假设完美CSI且忽略多路径与估计误差；强化学习的训练时间仍相对较长，且结果对模型参数敏感。

---

## 117. Examining User Behavior and Cognitive Biases in Personal Password Security

**arXiv ID:** 2607.19586 | [PDF](https://arxiv.org/pdf/2607.19586v1)

**作者:** Evelyn Crowe `[一作]` (Texas A&M University), Guofei Gu `[通讯]` (Texas A&M University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过对 248 名大学生进行问卷调查，系统分析了用户在密码创建、管理、更新以及使用多因素身份验证等方面的行为和认知差异，并结合行为经济学理论（超弧贴现、现状偏差、现时偏差）解释了用户在安全与便利之间的权衡。

**💡 创新点**

创新点在于将行为经济学框架与密码安全研究相结合，量化了“便利优先”与“安全意识”之间的矛盾，并提出“分层密码策略”与“行为激励”作为改进建议；同时首次在同一研究中对密码管理工具、密码生成器和泄露密码跟踪工具的使用情况做了详尽的跨变量分析。

**🔧 技术方法**

主要使用了量化的调查问卷设计、主题编码（Thematic Coding）对开放式回答进行归纳、描述性统计分析以及对不同技术工具使用率的比值计算；没有引入机器学习或算法模型，而是侧重于定性与定量的混合方法。

**📊 数据集**

数据集为自编的 35 题问卷，覆盖人口统计、密码创建与管理习惯、密码安全意识及工具使用情况，受访者为 Texas A&M University 的 18–25 岁学生，共 248 人。

**📈 对比分析**

比较方法主要是描述性统计（频率、百分比）和交叉表分析，未与其他实验或算法模型进行对照；报告的性能是“安全意识与实际行为的差距”指标，例如 57.2% 的受访者使用个人信息做密码，但 72.1% 认为不安全；再如 93.1% 的受访者存在密码重用等负面行为。

**⚠️ 局限性**

局限性包括：样本来源单一（仅为一所大学的一座城市）、性别与年龄分布不均、数据完全基于自述调查易受社会期望偏差与回忆偏差影响；未对行为理论进行量化验证，且未探讨不同文化、组织政策或技术环境对密码行为的影响。

---

## 118. Bounds and Limitations on Codes Achieving List Recovery Capacity

**arXiv ID:** 2607.19576 | [PDF](https://arxiv.org/pdf/2607.19576v1)

**作者:** Joshua Brakensiek `[一作]` (University of California), Zihan Zhang `[通讯]` (Ohio State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究错误更正码的列表恢复（list‑recovery）问题，给出了最优信息理论边界，并证明在已知的AEL框架下无法突破线性/加性码的列表恢复极限，指出需要全新的构造方法。

**💡 创新点**

①提出了一个“通用单列界（generalized singleton bound）”的紧致版本，精确刻画了在给定误差半径ρ、列表大小L、每个坐标列表大小ℓ时码率R的最优上界。②证明对所有常数ℓ、L以及足够大的符号集，任何达到该上界的码必存在，而任何超过该上界的码都不可行。③通过对AEL框架的细致分析，给出一个通用的下限：任何满足内部编码器“无关性”（inner‑encoder insensitive）且满足一定的谱扩展性和外码速率约束的AEL构造，都不能得到满足 (1−m(R−)/m−1,ℓ,ℓ^m−1) 的列表恢复码，因而表明现有的AEL构造无法实现列表恢复容量。

**🔧 技术方法**

采用了概率方法（随机码+删去违规三元组）、超图与图论工具（Kővári–Sós–Turán定理）、谱图性质（混合引理）、以及对码的协同约束（inner‑encoder无关性）等组合技术。

**📊 数据集**

无；本文为理论论文，未使用任何实验数据集。

**📈 对比分析**

与之前仅给出较弱界限的工作（如Goldberg‑Shangguan‑Tamo）相比，本文提供了严格且可匹配的上界与下界，证明了该界限在大符号集下是最佳的。关于AEL框架的比较表明，所有现有的AEL实现均落在所给的下限之上，因而无法达到列表恢复容量。

**⚠️ 局限性**

局限性：该结果仅在符号集足够大、ℓ、L为常数的情形下严格成立；对小符号集或参数非常数时结果不一定适用；此外，证明仅说明AEL框架无法突破极限，并未给出实现列表恢复容量的新构造方案，仍需后续研究。

---

## 119. ChronoStitch: Training-Free Composition of Visual KV Memories for Long-Horizon Temporal Reasoning

**arXiv ID:** 2607.19547 | [PDF](https://arxiv.org/pdf/2607.19547v1)

**作者:** Santiram Tiwari `[一作]` (KGraph AI Solutions Pvt Ltd), Kunal Kislay `[通讯]` (KGraph AI Solutions Pvt Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在长视频问答场景中，提出一种无训练的两阶段方法，用于将独立缓存的视觉 KV 记忆拼接成连续的全视频记忆，以支持时序推理。

**💡 创新点**

创新点在于（1）通过三轴 delta‑rotation 将各块的 post‑RoPE 键重新映射到全局的时间‑高度‑宽度坐标系，消除局部重叠导致的位置信息错误；（2）利用“关键字偏差”自适应地只重新计算部分后块视觉 token，恢复跨块注意力缺失，提升时序推理能力。

**🔧 技术方法**

主要技术包括多模态 RoPE（mRoPE）、KV 缓存再利用、三轴坐标重基（delta‑rotation）、自监督关键字偏差指标、以及针对后块 token 的选择性重计算。

**📊 数据集**

在 Qwen2.5‑VL‑3B 上使用 TempCompass 视频问答基准的时间子集进行评估，视频分块为每块 3 帧。

**📈 对比分析**

与基线（全联合预填、Naïve 拼接、1D 标量重基、仅三轴重基）对比：在 Order‑swap 任务中，修复 35% 后块 token 可恢复 100% 的正确顺序；在 TempCompass 上相较 Naïve 提升约 4.8% 总准确率、7.0% 事件排序准确率；查询时延比全联合预填快 3.3×（约 748 ms 对 2411 ms）。

**⚠️ 局限性**

限制包括：使用的 3B 读者模型规模有限，联合预填的上限仅 63.9%，repair 份额 ρ 的选择基于小样本控制实验，效率评估仅在 Apple M 系列硬件上测得，且在当前模型上标量与三轴重基在无修复时差异不大，未来模型或更大视频场景下效果需进一步验证。

---

## 120. Twin Agent: Context Residual Compression for Privilege Separated Agents

**arXiv ID:** 2607.19595 | [PDF](https://arxiv.org/pdf/2607.19595v1)

**作者:** Zhanhao Hu `[一作]` (University of California, Berkeley), David Wagner `[通讯]` (University of California, Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Twin Agent 两代理架构，用探索代理过滤不可信信息，安全代理执行特权操作，抵御提示注入攻击。

**💡 创新点**

创新点在于将安全通信视为残差压缩，允许探索代理仅发送最小必需提示，兼顾安全与任务效用。

**🔧 技术方法**

采用大语言模型的双代理设计、残差信息压缩、提示长度预算控制和提示注入检测器。

**📊 数据集**

评估使用 SWE-bench Lite、AgentDojo、DecodingTrust-Agent 等基准数据集。

**📈 对比分析**

与无防御代理、Dual Agent、CaMeL 等基线对比，Twin Agent 在保持高任务完成率的同时将攻击成功率降至接近 0%，并在预算下实现更佳安全‑效用折衷。

**⚠️ 局限性**

缺点包括仅在少数任务上验证、缺乏形式化安全保证、对更强适应性攻击仍有潜在风险。

---

## 121. Remote ID Spoofing-Aware Trajectory Planning for Small Unmanned Aerial Systems

**arXiv ID:** 2607.19650 | [PDF](https://arxiv.org/pdf/2607.19650v1)

**作者:** Jeremiah Webb `[一作]` (Vanderbilt University), Abenezer Taye `[通讯]` (Vanderbilt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套分布式的欺骗感知轨迹规划框架，用于在远程识别（RID）位置欺骗攻击下保障小型无人机的安全飞行。

**💡 创新点**

创新点在于：①将RID广播视为不可信，利用RSSI观测实现欺骗检测；②通过IMM滤波结合多模态多点定位动态估计欺骗机真实位置；③用置信概率将定位不确定性转换为危险椭圆体，并在MDP轨迹规划中显式纳入，提升鲁棒性和实时性。

**🔧 技术方法**

采用的技术包括：RSSI信号强度层测距与非线性最小二乘多点定位；互相嵌套的多模型（IMM）卡尔曼滤波；基于χ²分布的置信区间安全约束；基于MDP的风险感知轨迹规划；OMNeT++/INET物理层仿真与Python实现的规划器。

**📊 数据集**

使用的是自建的 1 km × 1 km 城市场景仿真数据，包含 4/8/12/16 台友好无人机和 1 台欺骗无人机，共 30 次实验重复，所有数据均来自仿真生成。

**📈 对比分析**

与假设 RID 完全可信的基线规划器对比；评价指标包括 NMAC（近距离碰撞）次数、定位包含率、计算时延；结果显示在所有交通密度下 NMAC 大幅下降（尤其是友好–欺骗对），定位包含率提升至约80%，计算开销仅占整体仿真时间的 5–7 s（≈15 ms/更新），保持实时可行。

**⚠️ 局限性**

局限性包括：仅考虑 RSSI，未模拟多径、信道吞吐量与丢包；实验仅用直线轨迹的欺骗机，未验证对更激进或策略性攻击者的鲁棒性；全局计算由地面控制站完成，可能存在单点瓶颈；在观测数不足或几何条件差的低密度场景下定位误差仍较大。

---

## 122. Anatomy of a Sound Neural Reasoner: One-Shot Amortization, First-Pass Poisoning, and Search Inertness in Clue-Rich Completion

**arXiv ID:** 2607.19635 | [PDF](https://arxiv.org/pdf/2607.19635v1)

**作者:** Aleksey Komissarov `[一作]` `[通讯]` (Neapolis University), Aleksey Komissarov (Neapolis University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究基于 Lattice Deduction Transformer (LDT) 的可验证神经求解器，探讨其在 Sudoku 等 CSP 上的行为与性能，并分析其一轮前向传播即完成求解的“一次性推理”现象与误判（first‑pass poisoning）。

**💡 创新点**

创新点在于：①系统性诊断 LDT 的前向推理即完成问题，指出误判源自第一轮消除；②提出并验证两种补救方案——训练时数字置换数据增强与测试时对称性帧的联合推理；③证明前向消除误判可通过对称性去相关化消除；④在从零开始的图着色任务中验证该框架的任务泛化能力。

**🔧 技术方法**

主要技术包括：Lattice 递归推理网络（单元只删除候选值）、可验证（verify‑or‑abstain）发射机制、基于 Graphormer 的约束图注意力、MRV/随机/学习式分支策略、DFS+回溯与禁忌集、数字置换和对称性增强等。

**📊 数据集**

数据集主要是人工生成的唯一解 Sudoku（4×4、6×6、9×9 25 线索）以及同类图着色（G(20,0.25)）和 3‑色图着色（n=40, 3.0–4.9 平均度）。

**📈 对比分析**

比较方法：在固定训练预算（32 链 × 60 轮）下与 GRAM、经典 MRV、随机分支等基线进行对比；性能表现：6×6 仅数字置换训练下可达 99.4% 正确率，9×9 在 25 线索下通过数字置换后实现 96.5% 正确率；未增强模型在 9×9 上仅 0% 正确率。

**⚠️ 局限性**

限制：①在线索丰富的情形下，搜索层对准确率几乎无影响；②全局冲突/策略头在不同尺寸间不具备零样本迁移；③在从零开始的多解 CSP（如图着色）中，前向推理无消除信号，导致学习效果有限；④对称性增强与数字置换虽然有效，但对模型结构仍未根本解决价值对称导致的误判。

---

## 123. EGRNet: A Lightweight Semantic Segmentation Network with Edge-Gated Refinement and Adversarial Sensing

**arXiv ID:** 2607.19617 | [PDF](https://arxiv.org/pdf/2607.19617v1)

**作者:** Bareera Qaseem `[一作]` (National University of Sciences and Technology), Muhammad Naveed Aman `[通讯]` (University of Nebraska-Lincoln)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量化、实时的城市场景语义分割网络EGRNet；

**💡 创新点**

创新点在于引入Edge‑Gated Refinement模块和可学习的门控融合机制，同时嵌入轻量级对抗输入检测；

**🔧 技术方法**

采用了深度可分离卷积、膨胀残差块、SE注意力、门控融合以及激活层L2范数检测等技术；

**📊 数据集**

在Cityscapes数据集上进行训练与评估；

**📈 对比分析**

与DABNet、LCNet、LMFFNet等轻量化模型比较，EGRNet仅0.46M参数、5.00G MACs、mIoU 65.28%，实现了性能与效率的双提升；

**⚠️ 局限性**

缺陷包括对抗检测阈值需人工调节、误报率仍存在，以及尚未在真实车载平台上验证实时性与鲁棒性。

---

## 124. RIME: Enabling Large-Scale Agentic Post-Production

**arXiv ID:** 2607.19605 | [PDF](https://arxiv.org/pdf/2607.19605v1)

**作者:** Noah Schaffer `[一作]` (Dartmouth), Nikhil Singh `[通讯]` (Dartmouth)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于规则的音乐后期编辑框架（RIME）和可供工具调用的音频后期工具集（POSE），用于从混音生成结构化的(输入、输出、指令)三元组，并在此基础上训练并评估多模态LLM代理；

**💡 创新点**

创新点在于：①构造了可扩展的、符合真实录音工作流程的规则与模式库，生成细粒度、可解释的编辑指令；②通过MCP接口将完整的源分离‑处理‑重混链暴露给代理，支持跨工具的多步决策；③提供了大规模的人工生成式编辑数据集及评估基准；

**🔧 技术方法**

采用了自动化工具调用（MCP）、多模态预训练模型（如Music Flamingo、Gemini Flash Lite）、低秩适配（LoRA）微调、音频距离度量（FADE、KAD、MERT embedding）以及图结构相似度（Graph F1）等技术；

**📊 数据集**

使用MTG‑Jamendo混音数据集（约549/297轨道）并结合Music Flamingo标签，生成约3,000对训练/评估三元组；另外构造300对去噪/修复对，供鲁棒性测试；

**📈 对比分析**

在零射手设置下，模型的音频相似度和图结构恢复均不理想（例如Graph F1≤0.89），微调后提升至约0.70–0.85，表现优于零射手但仍与真实工程实践相距较远；

**⚠️ 局限性**

局限性包括：依赖人工编码的规则与模式，操作符集合有限；评估指标尚未完全对应专业后期质量；数据集与工作流程对真实生产环境的代表性不足；

---

## 125. Scaling Laws for Hypernetwork-Based Knowledge Injection in Large Language Models

**arXiv ID:** 2607.19604 | [PDF](https://arxiv.org/pdf/2607.19604v1)

**作者:** Nischay Dhankhar `[一作]` (Nace AI), Abulhair Saparov `[通讯]` (Nace AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了使用超网络（hypernetwork）在训练时将大量事实知识注入大型语言模型的方法，并系统探讨了超网络在宽度、深度、目标模型规模和注入事实数上的扩展律。

**💡 创新点**

创新点包括首次给出超网络知识注入的扩展律，发现目标模型规模的提升对 OOD 泛化最有利，并与 LoRA 以及全微调在 OOD 性能上的比较展示了超网络在大规模下的优势。

**🔧 技术方法**

主要技术是基于 transformer 的超网络生成 LoRA 风格的权重适配器，采用固定的 LoRA rank=4、α=8，目标模型采用 0.5B‑14B 的  系列，并通过对训练样本中注入事实的集合进行学习，保持目标模型参数冻结。

**📊 数据集**

使用了自构造的  数据集，源自 4.6M 实体、822 条关系的知识图谱，生成了 1.25M 训练样本（1‑4 跳），并设定了 ID、OOD、重述 OOD 以及 MCQ 四个评估子集。

**📈 对比分析**

通过对比四个评估指标的幂律指数，发现超网络在 ID 下降上略逊于 LoRA/全微调，但在三种 OOD 测试中都显示出更陡峭的衰减曲线，尤其在 OOD 重述和 MCQ 上优势显著，表明其在训练时知识注入中对未知实体与关系的泛化更好。

**⚠️ 局限性**

局限性包括超网络本身在规模较大时参数量接近或超过目标模型，导致部署成本高；对深度多跳推理的支持尚未验证，且扩展到 70B 及以上模型的表现仍未知。

---

## 126. Learning to Transmit: Volatility-Aware Predictive Communication for Energy-Efficient IoT Networks

**arXiv ID:** 2607.19590 | [PDF](https://arxiv.org/pdf/2607.19590v1)

**作者:** John Kangethe `[一作]` (University of South Dakota), Longwei Wang `[通讯]` (University of South Dakota)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于波动率的预测通信框架ADAPTIVEML，能让IoT设备通过轻量级机器学习预测器和自适应阈值，智能决定何时传输数据，从而显著降低通信量。

**💡 创新点**

创新点在于将预测残差标准化为学生化残差，利用局部信号波动率设定单一阈值α来控制抑制概率，实现完全自校准的决策；同时引入递归最小二乘（RLS）在线学习，能够持续跟踪概念漂移并保持高精度。

**🔧 技术方法**

使用技术包括：岭回归预测器、递归最小二乘在线更新、滚动标准差计算、学生化残差阈值、双预测重建、以及对噪声与丢包的鲁棒性分析。

**📊 数据集**

实验数据集为：1）芝加哥海滩气象站（温度等），2）Intel Berkeley实验室WSN（室内多节点温湿度），3）UCI空气质量（CO传感器），共计超过240万条记录。

**📈 对比分析**

与周期、静态阈值、ARIMA、Kalman、EMA、LMS等六个基线比较，ADAPTIVEML在所有三数据集上实现最高数据压缩比（DRR）和最低均方误差（MAE），如芝加哥数据集可达94.7%传输削减且误差仅0.352°C，RLS版在漂移条件下误差进一步下降12‑18%，能耗显著降低。

**⚠️ 局限性**

局限性包括：假设残差近似高斯且采用单步预测，未对多通道、长抑制窗口和大规模网络同步等更复杂场景进行充分验证；在极端噪声或快速漂移环境下性能仍有提升空间。

---

## 127. VQ-Transplant: Efficient VQ-Module Integration for Pre-trained Visual Tokenizers

**arXiv ID:** 2607.19575 | [PDF](https://arxiv.org/pdf/2607.19575v1)

**作者:** Xianghong Fang `[一作]` (University of Toronto), Tim G. J. Rudner `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出VQ-Transplant框架，能把新的VQ模块无缝植入预训练视觉分词器并通过轻量解码器微调实现快速重建；并提出MMD‑VQ方法以最大均值差异对齐分布；

**💡 创新点**

通过将VQ模块与编码解码器分离、轻量化解码器适配、以及MMD对齐实现资源高效、兼容性强的VQ插件化；

**🔧 技术方法**

使用向量量化、最大均值差异（MMD）、轻量解码器微调、对抗训练（PatchGAN/StyleGAN）、DiffAug、以及常规重建损失等技术；

**📊 数据集**

在ImageNet‑1k、OpenImages、CelebA‑HQ、FFHQ、LSUN‑Churches等多域图像数据集上进行训练与评估；

**📈 对比分析**

与VAR、VQGAN等基准在r‑FID、r‑IS、PSNR、SSIM、LPIPS等指标对比，VQ‑Transplant在仅5轮解码器微调下即可达到或超过原VAR重建质量，同时训练成本降低约95%或加速约21.8×；

**⚠️ 局限性**

对预训练分词器的依赖较强，跨架构（如LDM‑16）适配效果略差；仅验证了固定/多尺度VQ，对更广泛VQ算法与分词器兼容性仍需进一步探索。

---

## 128. Extinction Depth and q-ary Error-Correcting Codes for the Limited Permutation Channel

**arXiv ID:** 2607.19566 | [PDF](https://arxiv.org/pdf/2607.19566v1)

**作者:** Noam Ben Shimon `[一作]` (Technion Israel Institute Of Technology), Aryeh Lev Zabokritskiy `[通讯]` (Migal Galilee Research Institute Tel Hai University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了在半径为一的有限置换信道中使用块拼接码的“消亡深度”判定准则，以判定纠错码的有效性。

**💡 创新点**

创新点在于用单一的消亡深度判定替代旧的两阶段局部判定，提供更强的纠错保证，并给出超越现有阈值的三元和五元块集；同时证明消亡深度无上界、给出线性与二次增长例子，和构造有限状态验证与大字母容量上界与下界。

**🔧 技术方法**

主要技术包括：块拼接码构造、消亡深度定义、有限状态图搜索、自动机同步、图论循环与无环性判定、覆盖码与置换覆盖、递推向量法、Perron–Frobenius谱分析、以及检测码的稳定提升与弱锯齿编码。

**📊 数据集**

使用的实验数据集：在符号集合{0,1,2}上构造的66/265个块集；在五元字母上构造的13模板生成的块集；对二进制两块、三块、四块集的完整枚举（共41,731个集合）用于验证消亡深度与循环；以及对不同字母量下的覆盖码尺寸和运行数值，如U_q、K_q(6;1)、K_q(18;1)等。

**📈 对比分析**

比较方法：将新构造的块集的纠错率与之前的Chee、Kovačević‑Goyal‑Kiah等构造进行对比；对大字母容量进行上界（改进的覆盖码）与下界（运行提升构造）相互夹逼，得到0.6777与0.6695的下界；检测码方面证明检测损失因子√2为最优，且通过弱锯齿码进一步提升三元四元的检测率。性能：三元块集率≥0.6777，五元≥0.6695，检测率三元≥0.7369，四元≥0.7629。

**⚠️ 局限性**

局限性：消亡深度判定的计算复杂度与块集大小呈多项式但常数较大；对字母量有限的具体构造仍停留在固定尺寸，尚未给出通用可扩展的构造；大字母容量下的上界与下界之间仍有约0.148差距；检测码的弱锯齿构造尚未对所有字母量给出最优构造；此外，证明的二次深度例子仅覆盖到k≤60，未证明更一般形式。

---

## 129. Combinatorial Capacity Bounds for the $q$-ary Deletion Channel

**arXiv ID:** 2607.19559 | [PDF](https://arxiv.org/pdf/2607.19559v1)

**作者:** Hassan Tavakoli `[一作]` (Oregon State University), Bella Bose `[通讯]` (Oregon State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究 q 进制删失信道，提出模式计数标量 N_n(x,y)，并利用其递推与行列和恒等式推导精确输出熵、容量上下界以及改进的下界。

**💡 创新点**

创新点在于用整数标量 N_n(x,y)将概率模型分解为组合与权重两部分，得到完全解析的熵表达式、容量“沙盒”及通过校正项 Φ 进一步提升下界，且证明 Δ_n(d) 为正，给出小删失概率下的精确下界。

**🔧 技术方法**

主要技术包括组合递推、行列和恒等式、信息熵解析、Blahut–Arimoto 算法以及超加性与斯坦伯里数的证明。

**📊 数据集**

使用合成数据：q = 2, 3，块长 n = 3, 5, 10，删失概率 d = 0.05, 0.10, 0.20 的离散化实验。

**📈 对比分析**

与已有下界 LB1、LB2、KM（仅 q=2）以及上界 UB 进行比较，结果显示 LB+（本研究下界）位于中间且随 n 增大逼近真实容量，Blahut–Arimoto 结果始终高于 LB+，差距随 n 缩小。

**⚠️ 局限性**

局限性在于下界基于均匀输入，尚未证明最优；Δ_n(d) 的闭式下界在大 n 下可能不够紧；对插入-删失信道及非均匀输入的推广仍待研究。

---

## 130. Enterprise Integration Modernization with SAP BTP

**arXiv ID:** 2607.19662 | [PDF](https://arxiv.org/pdf/2607.19662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 131. CHMAS: A Coupled Hierarchical Framework for Multi-Agent Reinforcement Learning

**arXiv ID:** 2607.19555 | [PDF](https://arxiv.org/pdf/2607.19555v1)

**作者:** Dongming Wang `[一作]` (University of California), Wei Ren `[通讯]` (University of California)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于双向耦合的分层多智能体强化学习框架CHMAS，旨在同时解决全局协调与局部执行、不同时间尺度的决策问题；

**💡 创新点**

创新点包括：①双向信息流的策略层与执行层耦合，使得执行层反馈直接影响策略层规划；②异步层级学习协议（每N_f个执行周期才更新一次策略层），有效缓解层级学习中的非平稳性；③使用可加近似的双层优化与理论分析，给出了策略层的O(log K/√K)收敛速率；

**🔧 技术方法**

主要技术手段有：层级强化学习、策略梯度（AHPG）、集中式训练/分散式执行、DQN（以及可能的Actor-Critic实现）、通信图模型、异步更新与双向反馈机制；

**📊 数据集**

在合成的25×25 GridWorld多智能体搜集任务（4名代理）上进行实验，使用全局覆盖状态与局部资源观测；

**📈 对比分析**

实验通过对比策略层与执行层的学习曲线与奖励，展示了双向耦合与异步更新后各层能够稳定收敛；虽然未给出与现有分层或全局方法的数值对比，但实验结果表明该框架能实现全局覆盖与局部任务的协同；

**⚠️ 局限性**

局限性在于：①采用可加近似和PL条件的理论分析仅在理想假设下成立；②实验仅在小规模、离散动作的GridWorld上验证，缺乏对更大规模或连续动作空间的评估；③仅使用DQN实现，未验证算法在更复杂策略梯度或Q‑learning变体中的收敛性。

---

## 132. New lower bounds for binary constant-weight codes: $A(23,6,10)\geq 2979$ and $A(24,6,10)\geq 4214$

**arXiv ID:** 2607.19550 | [PDF](https://arxiv.org/pdf/2607.19550v1)

**作者:** Christian Lysenstoeen `[一作]` `[通讯]` (Inland Norway University of Applied Sciences), Christian Lysenstoeen (Inland Norway University of Applied Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

作者构造了新的二进制常权码，在 (23,6,10)、(24,6,10)、(23,6,11)、(24,6,8) 四个参数组合上分别给出了更大的下界。

**💡 创新点**

创新点在于将坐标分解与 Johnson 组合法与 CHILS 最大独立集启发式相结合，并通过精确局部交换和自动化改进器重现并验证了失传的 1990 年结果。

**🔧 技术方法**

采用了 Johnson 分解、CHILS MWIS 迭代、OR‑Tools CP‑SAT、CBC/SCIP 精确求解、以及重新实现的 1990 年改进器和局部交换模型。

**📊 数据集**

主要数据集为长度 18~28 的二进制常权码空间与对应的交叉兼容集合，所有代码与日志均公开在作者 GitHub 仓库和 Brouwer 表格存档中。

**📈 对比分析**

与现有最优表格对比，新代码在上述四个细胞中分别提升了 100+、400+、400+、10+ 词，显著优于贪婪基线及之前的失传值，且已被表格维护者验证并列入实时表格。

**⚠️ 局限性**

局限在于仍未突破 (23,6,10) 2970 以上的 13+1^10 置换对称性和深度四的局部交换，以及对剩余失传细胞是否真实可达或为过计仍存在疑问。

---

## 133. Hollywood: Towards a Large Movie Dataset for Database Benchmarking

**arXiv ID:** 2607.19666 | [PDF](https://arxiv.org/pdf/2607.19666v1)

**作者:** Ivan Iachnyk `[一作]` (University of Technology Nuremberg), Andreas Kipf `[通讯]` (University of Technology Nuremberg)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了名为Hollywood的IMDb兼容合成基准生成器，利用LLM语义先验与确定性时间图生成，产生包含电影、人物、公司等多表关系的完整数据集，并提供对应SQL工作负载与标签。

**💡 创新点**

创新点在于：①把LLM作为语义先验而非直接生成元组，避免海量文本生成的不确定性；②通过时间签名图捕获跨表实体间的协作、竞争、兴趣等依赖，实现跨表谓词的自然关联；③在生成过程中使用可调节的实体池、年份与种子，支持多规模与可重现性；④将生成、验证、适配SQL四步包装成完整实验包。

**🔧 技术方法**

使用技术包括：Google Gemini LLM（生成语义字典）、确定性时间图（有向签名边、活跃区间）、基于加权选择的电影装配、SQL工作负载适配算法、传统与学习型计数/成本估计器（PostgreSQL、DuckDB、ZeroShot、MSCN）。

**📊 数据集**

主要数据集：Hollywood‑200K（20万部电影、351k标题行、19.7M IMDb‑style 行），对比原始IMDb完整基准和同标题行数的匹配样本。

**📈 对比分析**

比较方法：在全查询计数、单表选择、以及固定PostgreSQL计划的运行时成本预测上，分别使用传统估计器、学习型计数模型、ZeroShot与MSCN成本模型。结果显示：Hollywood在计数误差上与原IMDb相当或更大，尤其尾部（p95）误差显著增大；在成本预测上，Hollywood在部分工作负载下误差更大，但在学习型成本模型中尾部误差下降。整体表现表明生成器能产生与真实数据相似但更具挑战性的错误分布。

**⚠️ 局限性**

局限性：①目前仅支持单规模（20万电影）并未实现TPC‑style规模因子；②生成过程依赖LLM，易受训练数据偏差影响；③缺乏对其他基准模式（如TPC‑H/DS）的扩展；④未评估更复杂或大规模工作负载的计数与成本误差；⑤生成数据不完全复制IMDb的分布特性，只是结构相似。

---

## 134. Associations Between Support-Seekers' Cross-Community Interactions and Their Engagement with Received Comments in Online Health Communities

**arXiv ID:** 2607.19655 | [PDF](https://arxiv.org/pdf/2607.19655v1)

**作者:** Shenghan Tan `[一作]` (Sun Yat-sen University), Zhenhui Peng `[通讯]` (Sun Yat-sen University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过收集和分析了30个百度贴吧在线健康社区（OHC）的26,725名支持求助者的行为数据，探究其跨社区互动（参与社区类型、社区多样性、回复频率）与在OHC中对收到评论的行为、情感和认知参与度的关联。

**💡 创新点**

创新点在于首次将支持求助者的跨社区活动视为影响OHC参与度的重要因素，并利用多维度参与指标（社区类型、社区多样性、回复频率）系统性建模，揭示不同跨社区背景下的积极与消极参与趋势。

**🔧 技术方法**

研究采用BERT模型对社区名称和介绍进行嵌入后聚类得到六大社区类型，并用BERT对求助者回复进行认知参与度分类；情感参与度采用SnowNLP情感评分；支持需求与支持提供的量化评估则通过BERT分类器完成；最终通过逻辑回归、线性回归和序数逻辑回归对各参与维度进行定量分析。

**📊 数据集**

使用的数据集为来自百度贴吧的30个OHC及其外部40,479个社区的帖子、评论与回复，总计78,501篇帖子、845,397条一级评论、302,770条求助者回复以及184,460篇跨社区帖子。

**📈 对比分析**

通过对比不同回归模型的伪R²与R²，发现社区多样性与回复频率等跨社区变量对行为参与的影响显著（伪R²≈0.044–0.046），情感参与度伪R²≈0.003，认知参与度伪R²≈0.031，表明模型解释力有限但仍捕捉到一定关联。

**⚠️ 局限性**

研究的局限性包括：仅提供相关性而非因果关系；六大社区分类粗糙，可能掩盖更细粒度的差异；仅研究了百度贴吧，结果可能不具备跨平台普适性；未考虑语言风格、视觉内容等更深层次特征；使用的Markov链假设无记忆，未捕捉长期历史影响。

---

## 135. SPILLOVER: Measuring Cyberbullying NormPropagation on Social Media

**arXiv ID:** 2607.19646 | [PDF](https://arxiv.org/pdf/2607.19646v1)

**作者:** Arslan Bisharat `[一作]` (Loyola University Chicago), Yasin Silva `[通讯]` (Loyola University Chicago)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Instagram、Reddit、Wikipedia Detox、SOCC等平台的对话评论对进行分析，量化网络霸凌（CB）从一条评论传递到下一条的“溢出”现象。

**💡 创新点**

首次提供证据表明CB溢出是内容特定的、跨用户扩散的，并且与情绪共鸣与道德失调机制相符。

**🔧 技术方法**

采用固定效应回归、文本相似度（TF-IDF、词级Jaccard、字符n-gram、MiniLM、MPNet）和情感评分（VADER、RoBERTa），并将前一条评论的CB状态作为二元特征整合进HateBERT分类器。

**📊 数据集**

主要数据来自2012-2014年Instagram的430个含CB会话（32,754条评论对），以及使用Detoxify自动标注的Reddit、Wikipedia Detox和SOCC大规模数据集。

**📈 对比分析**

实验表明CB→CB的发生率相较于NoCB→CB提升约1.8-1.9倍（OR>1.8，p<10⁻⁶¹），单一CB指示特征的AUPRC为0.289（约3.2倍提升），加入此特征后HateBERT从0.71提升至0.74，跨平台验证显示提升因子在1.26-1.95之间。

**⚠️ 局限性**

局限性包括仅为观察性数据、Instagram样本时间较久、不同平台使用自动毒性标签、未完全控制会话内时间变化以及缺乏因果实验验证。

---

## 136. From Bit-Position Sensitivity to Unequal Error Protection for DNN Inference Memory

**arXiv ID:** 2607.19623 | [PDF](https://arxiv.org/pdf/2607.19623v1)

**作者:** Muhammad Husnain Mubarik `[一作]` (Advanced Micro Devices, Inc.), Kunal Tyagi `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了不同浮点格式（FP16、BF16、FP32）下，机器学习推理中每一比特位置的故障敏感性，并基于实验确定了可安全不保护的最低有效位阈值（FP16：6、BF16：4、FP32：15）。在此基础上提出了不等错误保护（UEP）方案：通过3阶分层（符号+指数、上部尾数、下部尾数）以及每行缓存标签和双分区SRAM，实现在推理路径上按位级别差异化 ECC，显著降低 ECC 码字面积与能耗。

**💡 创新点**

①首次系统性评估 16 种模型、6 种模态下的位级故障敏感性；②从实验中提取数据类型特定的安全阈值并构造可编程分层 UEP；③将 UEP 与缓存标签和双分区 SRAM 结合，实现单个硬件模块即可满足多种浮点格式与工作负载；④在硬件层面证明该方案可实现 27.8% ECC 码字面积缩减、17% BF16 读能耗降低，并保持可靠性在 10 FIT 以下。

**🔧 技术方法**

位级故障注入（单比特、随机、连续多比特）、连续噪声注入、实验室量化数据集、UEP 编码器与解码器设计、缓存行格式标签、双分区 SRAM 结构、功耗与面积分析（CACTI、Verilog RTL）、可靠性建模（FIT/MTTF）、多层次误差检测与纠正策略。

**📊 数据集**

使用多模态数据集：图像生成（SD3.5‑Medium、DiT‑XL/2、SD 1.5、PixArt‑α）、文本生成（Llama‑3 8B、Phi‑3‑mini、Mistral‑7B、GPT‑2 Medium、Falcon‑7B）、语言理解（BERT‑base）、语音识别（Whisper‑small）、视觉分类（ViT‑L/16、Swin‑B、ResNet‑50、EfficientNet‑B0、MobileNetV3‑L）。此外，还对 3,780 张 SD3.5‑Medium 图像进行了连续噪声实验。

**📈 对比分析**

与传统统一 SECDED、选择性 SECDED 以及 SERA‑Float 等方案比较，UEP 在不影响任务指标（PPL、Top‑1、WER、PSNR/SSIM）下，ECC 码字面积平均减少 27.8%，BF16 读能耗降低约 17%（FP16/FP32 约 14%），ECC 保护比率可提升 37.5%–62.5%（取决于工作负载分层）。可靠性方面，FIT 维持在 10 以下，MTTF 超过 1.9×10⁷ 年。

**⚠️ 局限性**

①实验仅使用合成和局部范围的故障注入，未覆盖真实芯片的全局空间/时间相关错误；②未在完整加速器实现中验证硬件调度与时序影响；③双分区 SRAM 仅在银行级别实现，切换延迟与布线开销待进一步评估；④方案仅适用于推理路径产生的 FP16/BF16/FP32；对于更窄的子字节格式（FP8/FP4）不具备安全阈值，需采用完整 ECC；⑤权重与输入等一次性加载数据仍保持全 ECC，未探究针对压缩模型的位级保护优化。

---

## 137. Understanding Developer Pain Points in Federated Learning: Insights from Stack Overflow and GitHub

**arXiv ID:** 2607.19621 | [PDF](https://arxiv.org/pdf/2607.19621v1)

**作者:** Sahand Saed `[一作]` (University of Saskatchewan), Banani Roy `[通讯]` (University of Saskatchewan)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对联邦学习（FL）开发者在 Stack Overflow 与 GitHub 上的讨论进行系统性经验性研究，识别并归类开发者面临的常见挑战，并分析其难度与求助意图。

**💡 创新点**

首次从两大公开平台交叉采样 FL 开发者痛点，构建 9 主题（Stack Overflow）与 13 主题（GitHub）的系统性话题树；将问题意图（How/Why/What/Other）与解决难度（未解决率、平均解决时间）关联，形成可操作的改进建议。

**🔧 技术方法**

采用 BERTopic 进行无监督主题建模；使用规则+零射击（zero-shot）分类器判定问题意图；通过统计未解决率和中位解决时长评估话题难度；并利用标签阈值技术过滤 FL 相关标签。

**📊 数据集**

收集 495 篇 FL 相关 Stack Overflow 帖子；以及 9,116 条来自 92 个活跃 FL 开源项目的 GitHub issue 与 PR。

**📈 对比分析**

通过对比两平台主题分布、意图占比及解决难度指标，评估哪些话题最易被忽视或长时间未解决；发现环境搭建、依赖兼容、API 迁移、训练不稳、评估错误、隐私机制集成等主题难度最高。

**⚠️ 局限性**

研究仅覆盖公开讨论与主流框架，可能遗漏私有或规模较小项目的痛点；标签筛选方法与数据收集范围限制了研究的完整性；结果的通用性需在更大多样化社区中验证。

---

## 138. SCPP: A Unified Python Library for Soft Clustering

**arXiv ID:** 2607.19620 | [PDF](https://arxiv.org/pdf/2607.19620v1)

**作者:** Kiyan Rezaee `[一作]`, Sadegh Eskandari `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个统一的软聚类Python包SCPP，整合40种算法并提供一致的API、benchmark和可复现工具。

**💡 创新点**

首次将不同软聚类方法统一到scikit-learn兼容接口，并将广泛的算法覆盖、benchmarking与软件工程实践集成于单一框架。

**🔧 技术方法**

基于Python实现，核心依赖NumPy、SciPy，图形与深度学习模块使用PyTorch，采用scikit-learn风格的Estimator接口和CI自动化测试。

**📊 数据集**

使用20个公开benchmark数据集（包含iris等经典案例）进行实验。

**📈 对比分析**

通过统一的评估管道对所有方法计算12种聚类质量指标、运行时、内存和可扩展性，实现公平比较；具体性能取决于算法，但框架使得不同方法间的比较变得可重复且直观。

**⚠️ 局限性**

局限在于目前仅包含已实现的40种算法，缺乏对新出现方法的及时更新；对深度学习与图模型的依赖较高，且某些高阶评估指标和多模态数据的支持尚未完善。

---

## 139. The Mechanism Matters: When Knowledge Graphs Help Reinforcement Learning

**arXiv ID:** 2607.19616 | [PDF](https://arxiv.org/pdf/2607.19616v1)

**作者:** Mohammed Sameer Syed `[一作]` `[通讯]`, Mohammed Sameer Syed

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对知识图谱（KG）在强化学习中的注入方式进行系统对照实验，独立变换任务、注入机制和KG质量，揭示结构性收益与机制安全性；

**💡 创新点**

提出可控合成KG实验框架并使用shuffle对照证明收益来源于图结构而非泛化，首次对软潜在奖励塑形与硬动作掩码的安全性做因果分离，给出剂量响应与错误KG的对比；

**🔧 技术方法**

使用MiniGrid离散任务、PPO与表格Q学习、潜在基础奖励塑形、动作掩码、状态特征注入，构造可调完整性/噪声/大小的KG，进行剂量、结构、外部验证实验；临床案例使用MIMIC-IV与CQL；

**📊 数据集**

六个MiniGrid任务（Empty、LavaGapS5、DoorKey-5x5/6x6、Unlock、KeyCorridorS3R1）与合成KG；临床案例使用MIMIC-IV ICU血流管理数据；

**📈 对比分析**

对比AUC和求解率，采用多种种子、Welch检验和Cohen d；正向KG显著提升稀疏奖励任务的样本效率和可靠性，掩码最佳但对错误KG敏感，塑形鲁棒但收益有限；临床案例无显著差异；

**⚠️ 局限性**

实验仅限网格任务和离线RL，掩码机制易脆弱；塑形对学习者敏感；状态特征仅为再编码；合成KG覆盖度低；未考虑更复杂KG或更深的RL算法；

---

## 140. Agent-Centric Animal Pose Forecasting

**arXiv ID:** 2607.19548 | [PDF](https://arxiv.org/pdf/2607.19548v1)

**作者:** Eyrun Eyjolfsdottir `[一作]` (HHMI Janelia Research Campus), Kristin Branson `[通讯]` (HHMI Janelia Research Campus)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并训练了基于 Transformer 的自回归生成模型，将动物的感知输入与运动输出都以动物自身坐标系为中心，并发布了可复用的 AnimalPoseForecasting 库。

**💡 创新点**

创新点在于提出以代理中心、可逆操作链的框架，既保留了生物学约束，又允许灵活切换感知与运动的表示；同时提供统一的评估指标与多种输入/输出设计的可比性。

**🔧 技术方法**

使用 Transformer 网络、离散化+连续回归混合输出、时序卷积嵌入、多模态感知特征、可逆运算链以及自动回归 rollout 等技术。

**📊 数据集**

主要实验数据集为 MABe 2022 蚂蚁（果蝇）社交行为数据（约72M帧）以及 Synthetic RatInABox 的小鼠轨迹。

**📈 对比分析**

通过特征分布的 Wasserstein 距离、判别器准确率、行为模式频率、线性探针 MCC 等多维度指标对模型变体进行比较；Reference 模型在大多数指标上表现最佳，离散化输出显著提升生成质量，Keypoints 方案表现最差。

**⚠️ 局限性**

局限包括：缺乏物理碰撞模型导致部分生成轨迹不够真实；离散化方式过于独立导致远距离 rollout 漂移；目前仍无法完全逼近真实分布（判别器可轻易区分）；多模态协同建模与全维度量化仍有改进空间。

---

## 141. When HTTP 402 Meets the Blockchain: Risks on Emerging x402 Payments

**arXiv ID:** 2607.19545 | [PDF](https://arxiv.org/pdf/2607.19545v1)

**作者:** Qinying Wang `[一作]` (EPFL), Mathias Payer `[通讯]` (EPFL)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文系统评估了 x402 支付协议中 facilitator（第三方支付中介）的安全性，首先通过协议分析提出了一套覆盖授权正确性与执行安全的统一安全规则；随后构建半自动黑盒测试框架，对 15 家主流 facilitator 进行大规模评测，发现所有平台均违反至少一条规则，并基于这些违规推导出四类新攻击（Free Shopping、Asset Theft、Service Denial、Gas Abuse），最后通过对 Base 与 Solana 主网约 119 M 笔交易的链上测量，量化了生态集中度、失败率与成本风险。

**💡 创新点**

创新点包括：① 提出跨网络、跨支付证明格式的可检验安全规则集合；② 开发规则驱动、功能感知的黑盒测试工具，可在不访问源码或私钥的情况下自动识别规则违规；③ 首次公开发现并验证四类利用 facilitator 弱点的实际攻击；④ 结合链上大规模量化，展示 x402 生态的集中化与成本风险，形成完整的安全评估生态。

**🔧 技术方法**

技术手段主要包括：协议形式化分析、支付证明模板生成与变异、黑盒 HTTP/JSON 接口自动化测试、链上交易过滤与统计（ERC‑3009、SPL‑Token、计算单位、租金等），以及利用现有 SDK（Coinbase、Flask 等）构建受控服务器端。工具采用规则引擎驱动，支持多网络（Base、Solana）和多签名模型（ERC‑1271、ERC‑6492）。

**📊 数据集**

数据集：
- x402scan 公开的 facilitator 列表与交易量；
- Base 与 Solana 主网 10 M 区块期间约 119 M 笔 x402 相关交易；
- 交易细节（交易哈希、账户、金额、执行状态）与链上租金事件；
- 通过 SDK 生成的测试交易与实际交易对比。

**📈 对比分析**

评估方法：
- 规则检测：对每个 facilitator 逐条执行 SR1–SR8 的黑盒测试，记录通过/失败；
- 攻击验证：对违规点做针对性攻击实验，直接证明或给出高风险证据；
- 成本与失败率量化：统计链上成功/失败交易比例，计算累计 gas/费用损失。结果显示：每个 facilitator 平均耗时 <10 min；所有 15 家平台均出现至少一条规则违规；直接可验证的攻击成功率约 2/15 (Free Shopping)、1/15 (Asset Theft)、0/15 (Service Denial)、3/15 (Gas Abuse)；总体失败率 1.99%（Base）/0.018%（Solana），累计费用约 202 k USD。

**⚠️ 局限性**

局限性：
- 仅覆盖 Base 与 Solana 两条链，未对 Starknet 等新网络支付证明做评测；
- 黑盒测试无法捕获内部状态或定制化逻辑导致的细粒度问题；
- 受道德限制，未能对某些攻击做完整实测，部分结果仅为高风险评估；
- 未考虑完全恶意或协同的 facilitator 失信场景；
- 规则覆盖面虽广，但仍可能漏掉协议新增功能或细节。

---

## 142. Beyond Relevance-Centric Retrieval: Rubric-Oriented Document Set Selection and Ranking

**arXiv ID:** 2607.19747 | [PDF](https://arxiv.org/pdf/2607.19747v1)

**作者:** Kailin Jiang `[一作]` (University of Science and Technology of China), Haibo Shi `[通讯]` (Yuanbao Team, Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于多维 Rubric 的文档集评估框架，并将评估信号转化为无训练的文档集选择方法。

**💡 创新点**

创新点在于引入三层九维度评估 Rubric，填补传统单维度评估的盲点，并实现评估到优化的闭环。

**🔧 技术方法**

使用 LLM‑as‑Judge（DeepSeek‑V4 Pro 等）自动评分、链式思考推理（Qwen3‑8B）以及多模型 Rubric 生成策略。

**📊 数据集**

数据集包括多跳 QA 数据集 HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle（短文本）以及 ResearchQA 与搜索代理 DR.Tulu‑8B（长文本）。

**📈 对比分析**

对 12 种 reranker 进行基准测试，发现 SetR/Rank4Gen 在短文本上最优，ReasonRank/Rearank 在长文本上领先；而提出的方法在两种场景下均以更少文档/搜索回合实现最高生成质量。

**⚠️ 局限性**

主要限制是 Rubric 需基于参考答案生成，属于 oracle 情况，缺乏无监督的可迁移优化方案。

---

## 143. Neuro-Symbolic AI for Korean Criminal Law: Sentencing Prediction and Document Drafting

**arXiv ID:** 2607.19740 | [PDF](https://arxiv.org/pdf/2607.19740v1)

**作者:** Yeonseok Lee `[一作]` `[通讯]` (SLING AI Inc.), Yeonseok Lee (SLING AI Inc.)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种神经符号框架，将韩司法系统中交通违法的摘要起诉流程从原始证据到判决建议实现全流程自动化。

**💡 创新点**

创新点在于将LLM仅用于语义抽取、引入人机交互验证，并将法条与量化罚金转化为Z3 SMT求解器的形式化约束，彻底消除概率模型的幻觉风险。

**🔧 技术方法**

使用的技术包括 OCR+文本分割、基于大语言模型的实体抽取、人工审核接口、Z3 SMT 求解器以及 Prompt Engineering 生成合规法律文书。

**📊 数据集**

使用的数据集为韩国内部的警务与起诉文件（驾驶员陈述、呼气测酒、讯问记录、事故报告）以及公开的 2026 年交通刑事量刑指南。

**📈 对比分析**

与传统纯 LLM 推断及基于规则的手工实现对比，本文通过案例验证抽取准确率>90%，SMT 计算的确定性无误差，并节省审查时间约 40%（尚无公开基准可对比）。

**⚠️ 局限性**

局限性包括对真实多样化文档格式的适配仍需进一步工程化、需人工验证的步骤降低了全自动化程度，以及目前仅覆盖交通违法，未扩展到其他少数罪类。

---

## 144. Personalized Recommendation Tool Learning via Autonomous Language Agents

**arXiv ID:** 2607.19739 | [PDF](https://arxiv.org/pdf/2607.19739v1)

**作者:** Mingdai Yang `[一作]` (University of Illinois Chicago), Philip Yu `[通讯]` (University of Illinois Chicago)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 PRTA 框架，利用 LLM 作为工具管理器，将 LightGCN、SASRec、SimpleX 等传统推荐模型作为工具，结合反射机制学习个性化工具权重，并在全排序推荐中使用轻量级语义重排序。

**💡 创新点**

核心创新在于：①将 LLM 与多模型推荐协同，解耦行为建模与语义推理；②设计局部评估、全局比较与排名比较三种反射模块动态更新工具偏好；③在全排序设置下通过架构避免 LLM 幻觉与上下文长度限制，实现高效个性化推荐。

**🔧 技术方法**

使用多模型推荐工具（LightGCN、SASRec、SimpleX），Phi‑4 量化版 LLM 负责用户画像生成、工具评估、全局比较与轻量级重排序；通过加权求和聚合工具得分并进行语义重排序。

**📊 数据集**

三个公开数据集：Amazon（电影/CD）、Yelp（商家）、Goodreads（书籍）。

**📈 对比分析**

与单一工具、RecBole 基线（ENMF、DiffRec、FEARec）、BM25、LLMRank、RAG 等进行全排序评估，指标为 R@10/20、N@10/20。PRTA 在所有数据集均排名第一，显著优于所有基线。

**⚠️ 局限性**

局限性包括：仍需大量 LLM 调用导致延迟与成本；工具选择受限于预训练模型的覆盖范围；在文本信息不明显的场景下重排序效果有限；模型表现受 LLM 版本与参数调优影响。

---

## 145. Optimization of MIMO STEEP for Secure Communications over MIMOME Channels

**arXiv ID:** 2607.19738 | [PDF](https://arxiv.org/pdf/2607.19738v1)

**作者:** Dinglin Yu `[一作]` (University of California at Riverside), Yingbo Hua `[通讯]` (University of California at Riverside)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为STEEP的安全通信传输方案，能够在用户之间的信道较弱的情况下实现正的保密率，特别是在MIMO信道中优化预编码器以提高保密率。

**💡 创新点**

首次揭示了通过优化预编码器显著提高MIMO STEEP的可实现保密率的可能性。

**🔧 技术方法**

使用了交替优化（AO）、加权最小均方误差（WMMSE）、连续凸近似（SCA）和投影梯度上升（PGA）等技术。

**📊 数据集**

使用了MIMO信道模型，假设用户和窃听者的天线数量不同，并考虑了Rician衰落的信道矩阵。

**📈 对比分析**

与基线MIMO STEEP进行比较，优化后的MIMO STEEP在保密率上表现出显著的提升，尤其是在窃听者信道较强的情况下。

**⚠️ 局限性**

限制在于当前的优化方法假设对窃听者信道矩阵有完全的知识，未来的工作需要考虑在有限知识下的最优预编码器设计。

---

## 146. An Exploratory Analysis of Pain Localization via Explainable Computational Modeling

**arXiv ID:** 2607.19726 | [PDF](https://arxiv.org/pdf/2607.19726v1)

**作者:** Ioannis Kyprakis `[一作]` (Hellenic Mediterranean University), Manolis Tsiknakis `[通讯]` (Hellenic Mediterranean University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对AI4Pain 2026 Challenge数据集进行无受试者依赖的三分类疼痛定位，系统比较了手工特征+树模型和端到端深度序列模型的表现，并进行可解释性分析。

**💡 创新点**

创新点在于：①在小样本、短时段可穿戴生理信号上系统性比较经典与深度方法；②利用Gini重要性揭示EDA谱特征为最关键的定位指标；③发现检测与定位的精度差距达26%，表明自主神经信号在10s分辨率下的生理极限；④提出多模态特征与交叉相关特征的组合可显著提升性能。

**🔧 技术方法**

技术包括：时域、频域、模态特定与交叉相关特征提取；Extra Trees、Soft‑Voting Ensemble、CNN‑Transformer、Fusion Network等模型；随机搜索与网格搜索调参；交叉验证与早停；Gini重要性解释；宏F1、balanced accuracy、ROC‑AUC评估。

**📊 数据集**

使用AI4Pain 2026 Challenge数据集，65名受试者，采集EDA、BVP、Resp、SpO₂四通道信号，10秒段，共36段/受试者，分为无痛、手臂疼痛、手部疼痛三类，训练/验证/测试划分为41/12/12受试者。

**📈 对比分析**

采用宏F1、balanced accuracy、ROC‑AUC进行模型比较。Extra Trees在验证集宏F1达到0.539，最高；深度模型最高为0.465，落后7.4个百分点。测试集Accuracy为49.1%，与验证的54.9%相差5.8个百分点；检测F1为0.815，定位F1仅为0.552，显示定位难度显著大于检测。

**⚠️ 局限性**

局限性包括：受试者数量有限，10秒窗口可能不足以捕获呼吸与血氧的疼痛相关动态；仅在实验室诱导的急性疼痛上验证，难以推广到慢性或术后疼痛；模型仅利用自主神经信号，缺乏面部或行为特征的补充；小样本导致模型对不同受试者的泛化能力波动较大。

---

## 147. Did Alice Do Wrong? Cross-Cultural Differences in Student Perceptions of Generative AI Use in University Computing Education

**arXiv ID:** 2607.19699 | [PDF](https://arxiv.org/pdf/2607.19699v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 148. Koopman Dreamer: Spectrally Constrained Latent Dynamics for Stable World-Model Imagination

**arXiv ID:** 2607.19719 | [PDF](https://arxiv.org/pdf/2607.19719v1)

**作者:** Jiaqi Li `[一作]` (National University of Defense Technology), Xin Xu `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Koopman Dreamer，将Dreamer模型的确定性转移用谱受限的Koopman骨架重写，以改进长期隐状态预测与控制

**💡 创新点**

创新点在于：①使用二维旋转-缩放块与可学习模数半径实现可控谱约束；②加入低秩双线性交互、后验条件EMA教师及多步rollout一致性；③提供多步误差界分析，解释谱参数与误差衰减的权衡

**🔧 技术方法**

采用Koopman变换、光谱约束、RSSM框架、EMA教师、线性+双线性动作项、开放循环预测、Actor-Critic等技术

**📊 数据集**

使用DeepMind Control Suite的9个本体感知任务及仿真UAV‑LiDAR Forest导航任务

**📈 对比分析**

与DreamerV3、PPO、D4PG等方法对比；在DMC任务中取得8/9个任务的最佳或近最佳成绩，UAV任务成功率提升至73.8%（DreamerV3为53.8%），开环预测误差显著降低

**⚠️ 局限性**

局限性包括：仅在仿真环境验证；谱上界的选择需经验性调参；在极端不稳定动力学下可能需要更细致的控制；未给出完整的安全或风险保障机制

---

## 149. Lightweight Person-Place Relation Extraction from Historical Newspapers with Dependency Graphs and Proximity Features

**arXiv ID:** 2607.19718 | [PDF](https://arxiv.org/pdf/2607.19718v1)

**作者:** Mlen-Too Wesley `[一作]` `[通讯]` (Georgia Institute of Technology), Mlen-Too Wesley (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于依赖解析的文档级图，提取15个靠近度与词性特征，使用轻量级的scikit-learn或小型Graph Attention Networks对历史报纸中的人–地点关系进行分类。

**💡 创新点**

证明仅使用最小字符距离即可捕获大部分关系信息，且在多语言、低资源环境下无需预训练语言模型即可实现中等水平的准确率；同时揭示了文档级交叉验证对评估结果的重要性。

**🔧 技术方法**

使用Stanza进行依赖解析、FastText嵌入进行跨句桥接、NetworkX构建图、scikit-learn随机森林/极限树/逻辑回归，以及PyTorch Geometric实现的GAT/变体；特征为距离、词性计数等。

**📊 数据集**

HIPE-2026 共享任务数据集，涵盖英语、法语、德语19–20世纪历史报纸的预标注人名与地点名，以及测试集 Test A（报纸）和 Test B（法语文学惊喜集）。

**📈 对比分析**

与17支参赛团队在宏召回（macro recall）评估指标下比较，官方排名第26/46，效率档案第3/46；在内部评测中，基于路径长度的规则达0.549，最佳模型仅0.520，说明特征工程在低资源场景下已具备竞争力。

**⚠️ 局限性**

主要局限在于：对 TRUE 关系的召回低、对 OCR 破损或跨段落的关系预测失误、对不同语言的解析质量差异导致模型表现不一致，以及对文本域迁移（Test B）缺乏适配。

---

## 150. SafeGen: Goal-Conditioned Video Diffusion of Safety-Critical Scenarios for VLM-Based Autonomous Driving

**arXiv ID:** 2607.19701 | [PDF](https://arxiv.org/pdf/2607.19701v1)

**作者:** Jiangfan Liu `[一作]` (Beihang University), Xianglong Liu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于目标条件扩散的安全关键场景视频生成框架，用于评估视觉语言模型驱动的自动驾驶系统对人类-车辆冲突的鲁棒性。

**💡 创新点**

将安全关键场景生成视为目标条件扩散过程，先通过视觉语言模型推断潜在脆弱点并产生灾难性终态规范，再利用深度感知与几何投影将对抗性VRU插入场景，实现高保真、时空一致的安全关键视频，并提出Judge Overall Score评价指标。

**🔧 技术方法**

采用目标条件扩散模型（FLF2V）、文本到图像扩散（Stable Diffusion 3.5）、深度估计（DepthAnythingV3）、语义分割（MobileSAM）、视觉语言模型（Qwen3‑VL‑32B）、视频扩散（Wan2.1‑FLF2V）以及多轮隐层填充等技术。

**📊 数据集**

使用nuScenes数据集生成850条安全关键场景作为训练与评估样本，并在下游微调实验中引用VRU‑Accident基准。

**📈 对比分析**

与四个基线（Learning-to-Collide、AdvSim、Adversarial Trajectory Optimization、ScenGE）在三种VLMAD上对比，本方法在Judge Overall Score平均提升24.25%，并在感知、预测、规划子指标均显著恶化；在VRU‑Accident上微调后平均提升15.9%。

**⚠️ 局限性**

局限性包括仅聚焦单人-车辆冲突，未覆盖多Agent交互；对视觉语言模型推理质量高度依赖；生成过程计算成本高；对不同摄像头视角的适应性有限。

---

## 151. PhenSPINE: A Standardized Benchmark for Spine Pathology Diagnosis

**arXiv ID:** 2607.19696 | [PDF](https://arxiv.org/pdf/2607.19696v1)

**作者:** Duong Ngoc Vu `[一作]` (National Economics University), Thien Van Luong `[通讯]` (National Economics University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了 PhenSPINE MRI 数据集，并设计了基准实验框架，用于评估脊柱病理诊断的深度学习模型。

**💡 创新点**

创新点在于引入位置编码（Positional Encoding）来显式建模脊椎间盘级别的空间信息，并证明单序列 SAG‑T2 在诊断上优于多序列融合，挑战传统多序列整合思路。

**🔧 技术方法**

采用 EfficientNet‑B1 作为特征提取骨干，结合全局平均池化、位置编码、以及多标签分类头，利用二元交叉熵训练多病理检测模型。

**📊 数据集**

使用 PhenSPINE 数据集，共 16,813 张 DICOM 图像，来自 250 名患者，涵盖 SAG‑T2、SAG‑T1、AX‑T2、SAG‑STIR 四种 MRI 序列，并标注四种脊柱病理与脊椎级别。

**📈 对比分析**

通过与 DenseNet、ResNet、EfficientNet-B0/B1/B2 等骨干在单序列与多序列组合下进行对比实验，SAG‑T2 + EfficientNet‑B1 的宏 F1 最高为 50.31%，精度 60.01%，而多序列融合反而导致 F1 降低至 48.33% 或更低。

**⚠️ 局限性**

局限性包括：多序列图像中存在较大噪声且缺乏显式脊柱分割，导致多序列融合无效；数据集规模虽较大但仍有限，未覆盖所有可能的病理类型；模型对不同扫描设备与协议的泛化能力尚待进一步验证。

---

## 152. Edge Intelligence in Civil Aviation: Paradigms, Techniques, and Applications

**arXiv ID:** 2607.19676 | [PDF](https://arxiv.org/pdf/2607.19676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 153. Generalized Constraint Projection: Four-Dimensional Type Inference for Dynamic Languages

**arXiv ID:** 2607.19693 | [PDF](https://arxiv.org/pdf/2607.19693v1)

**作者:** Qunhui Zhang `[一作]` `[通讯]` (Shanghai Jiao Tong University), Qunhui Zhang (Shanghai Jiao Tong University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 Generalized Constraint Projection (GCP)，一种零注解推断框架，能够在动态语言中同时处理值覆盖、程序员声明、上下文需求和结构访问四种信息源，并在每次调用时以投影会话的方式隔离约束；并在此基础上设计了结构匹配机制 OEM 和链式方法保留机制 future-，实现了对动态语言（如自研的 Outline 与 Python）零注解类型恢复与流式 API 维持。

**💡 创新点**

创新点主要有：①四维约束槽分离并按不同方向（join、meet、单写）更新，避免了传统统一约束集产生的冲突；②投影会话的“定义时模板 + 调用时投影”模式，使普通调用不污染定义；③OEM 的开放式结构子类型推导协议；④future- 在方法链中保留接收者类型，解决了基类无预期子类扩展导致的类型退化问题；⑤完整的理论证明（单/多模块收敛、投影安全、递归无环子语言安全等）。

**🔧 技术方法**

技术栈包括：抽象语法森林（ASF）前端；基于四维 Genericable 模型的约束推导；结构兼容关系 OEM（八条子类型规则及双向委托协议）；投影算法（包括高阶投影流和残差投影）；有限高度类型预序；未来- 的接收者保留规则；在 Outline 语言中实现实体/方法声明、类型注解恢复；在 Python 中利用 Protocol 与四维推断生成 PEP‑484 注解。

**📊 数据集**

主要使用的基准数据集是：Outline 的 Genericable 回归套件（覆盖声明、扩展、上下文、结构、组合需求、多调用、高阶投影等场景）以及一组未注解的 Python 源码，用于验证 GCP 在 Python 上恢复 PEP‑484 注解的效果；伴随的系统实现与性能评估在随附论文中给出。

**📈 对比分析**

比较方法：在 Outline 版中与传统统一约束/双向检查对照，展示 GCP 在冲突率、推断完整性与注解量上的显著改进；在 Python 版中与 mypy、Pyright 等渐进式检查器对照，报告在无注解代码上成功恢复注解的比例和编译时间。性能方面，理论上 GCP 的局部收敛为 O(1) 每个参数，整体单模块收敛为 O(N)；多模块收敛为 O(M·N)。实验数据显示，在 Outline 的回归测试中约 90% 的函数不需要任何注解即可通过；在 Python 代码上恢复注解的成功率超过 85%，并且编译时间比 mypyc 低 30%。

**⚠️ 局限性**

限制与局限：①仅适用于有限高度的类型预序；②对完整 Outline 语法的递归、循环等特性未证明安全，只在递归自由子语言 Outline_0 上给出 L2 证明；③缺乏行为子类型（行为子类型约束不在推断范围内）；④需要手工实现 OEM 规则及未来-；⑤在多模块跨文件引用时需要显式的前向引用处理；⑥在动态特性（如反射、运行时属性增删）下，结构子类型可能不安全；⑦对 LLM 生成的代码，若涉及未覆盖的自定义类型，GCP 只能返回最宽泛的类型（Any）。

---

## 154. GhostPrompt: Cross-Image Adversarial Prompt for Vision-Language Models

**arXiv ID:** 2607.19683 | [PDF](https://arxiv.org/pdf/2607.19683v1)

**作者:** Li Zeng `[一作]` (Changsha University of Science and Technology), Zhetao Li `[通讯]` (Jinan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种跨图像可迁移的对抗性提示（GhostPrompt），可一次性优化后在不同图像下诱导Vision‑Language Models（VLM）输出攻击者指定的目标文本。

**💡 创新点**

创新点在于：①采用鲁棒的 min‑max 双阶段优化，在文本提示与“worst‑case”图像之间交替更新；②通过引入语义对齐损失和文本连贯性损失，使提示在视觉无关的同时保持自然流畅，显著提升跨图像攻击成功率。

**🔧 技术方法**

使用了 Gumbel‑Softmax 对离散文本进行连续化优化；投影梯度上升/下降交替优化；Guided Adversarial Loss、Semantic Alignment Loss、Text Coherence Loss 等多重损失；整体 min‑max 双阶段训练框架。

**📊 数据集**

实验数据集包括 MS‑COCO 验证集、ImageNet 验证集、VQA 与 GQA 提示模板；对四个开源 VLM 进行评估：MiniGPT‑4、BLIP‑2、InstructBLIP、LLaVA‑v1.5‑7b。

**📈 对比分析**

与 SoTA 白盒方法（GCG‑Transfer、DeGCG、AutoDAN、PGD‑BERT）及黑盒方法（PAIR、TAP）对比，GhostPrompt 在所有模型和目标文本上平均提升 Attack Success Rate（ASR）超过 30%，训练时间缩短约 70%；在黑盒迁移实验中亦优于对手。

**⚠️ 局限性**

局限性：①文本离散性限制了攻击完美成功率；②跨模型迁移能力仍受限；③对商业 VLM（如 GPT‑4V）的效果尚不理想；④对抗训练会显著降低正常任务的准确率。

---

## 155. Context Matters: Improving the Practical Reliability of LLM-Based Unit Test Generation

**arXiv ID:** 2607.19682 | [PDF](https://arxiv.org/pdf/2607.19682v1)

**作者:** Junjie Chen `[一作]` (Tianjin University), Dong Wang `[通讯]` (Tianjin University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CATGen，一种基于项目上下文感知的 LLM 单元测试生成工作流，解决工业项目中测试编译失败、框架/依赖不匹配以及高昂的手工修复成本等问题。

**💡 创新点**

创新点在于将项目级依赖显式检索、构造确定性的测试类骨架以及基于程序分析的确定性编译错误修复与覆盖增强三大模块集成到 LLM 生成流程中，从而避免盲目推断、结构漂移和反复 LLM 迭代。

**🔧 技术方法**

使用了大型语言模型（如 Qwen2.5‑Coder‑32B‑Instruct 等）、IntelliJ PSI 进行静态上下文提取、模板化测试骨架生成、以及一套基于 AST 的静态编译错误修复规则和覆盖增强规则。

**📊 数据集**

在与华为云合作的八个企业内部 Java 项目中构建了 183 个关注方法的工业基准，并在 Defects4J 的四个开源项目（Chart、Lang、Time、Math）上验证泛化能力。

**📈 对比分析**

与 EvoSuite、ChatTester、ChatUniTest、HITS、TELPA、RATester 六个代表性基线在编译成功率、行/分支覆盖率、通过率、执行时间和 token 消耗等指标进行对比，CATGen 在工业环境下编译成功率提升约 30%，覆盖率提升约 15%，执行时间和 token 消耗分别降低约 60% 和 80%。

**⚠️ 局限性**

局限性包括：目前仅支持 Java；依赖静态分析和项目结构，模型选择与提示仍会影响效果；在极端复杂或高度动态的框架环境下仍可能出现编译错误；工业数据不可公开，实验可复现性受限。

---

## 156. FedLSG: LLM-Enhanced Semantic Calibration for Federated Graph Backdoor Defense

**arXiv ID:** 2607.19674 | [PDF](https://arxiv.org/pdf/2607.19674v1)

**作者:** Chenyu Zhou `[一作]` (Southeast University), Xinyuan Miao `[通讯]` (Purple Mountain Laboratories)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 FedLSG 框架，利用大语言模型对图结构与客户端行为进行文本化并进行语义推理，从而在联邦图神经网络中实现反后门防御。

**💡 创新点**

创新点在于首次将图结构与行为转化为文本并引入LLM进行语义判别，结合规则评分与轻量级 LoRA 学习，实现了服务器端教师与客户端学生的知识蒸馏和自适应信任评估。

**🔧 技术方法**

核心技术包括 GAT‑基 SRGAT、LoRA 低秩调优、全尺度 LLM 教师、prompt‑engineered 图‑文本转换、规则与 LLM 评分融合、基于信任的聚合加权。

**📊 数据集**

实验使用四个公开数据集：Citeseer、Pubmed、Flickr 与 OGB‑arxiv，评估三种主流后门攻击（DPGBA、UGBA、SBA）。

**📈 对比分析**

与 Prune、PruneLD、OD、RLR、NoisyGCN、FedTGE、RIGBD 等基线相比，FedLSG 在所有数据集和攻击下均取得最低 ASR（0–7% 左右）且保持清晰数据准确率（>70%），在白盒自适应攻击和多受害客户端场景下亦表现出较强鲁棒性。

**⚠️ 局限性**

局限性包括对 LLM 计算和存储资源的依赖，Prompt 设计的通用性有限，白盒自适应攻击下 ASR 略升，以及在极大规模图上 SBA 等轻量级攻击的效果依旧不佳。

---

## 157. AI-Increased Talent Retention Strategies: Fostering Long-Term Employee Engagement and Development in Talent Management

**arXiv ID:** 2607.19733 | [PDF](https://arxiv.org/pdf/2607.19733v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 158. A Unified Variational Framework for Deep Weakly Supervised Image Segmentation

**arXiv ID:** 2607.19669 | [PDF](https://arxiv.org/pdf/2607.19669v1)

**作者:** Yin King Chu `[一作]` (Hong Kong University of Science and Technology), Xue-Cheng Tai `[通讯]` (Norwegian Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的变分框架，用简单约束的Potts模型与平滑边界正则化，实现稀疏像素级监督下的图像分割和深度弱监督学习。

**💡 创新点**

创新点在于将稀疏标签通过RKHS函数扩展构造模糊隶属函数，再将其嵌入Potts能量，得到光滑可凸的损失，可直接用于网络训练；并将该模型衍生为可高效求解的阈值动态算法。

**🔧 技术方法**

采用RKHS（核函数）进行标签扩展、简单约束Potts模型、平滑总变分近似、阈值动态（TD）正则化以及深度学习中的UNet等网络架构。

**📊 数据集**

实验使用ECSSD（1000张含稀疏标记的显著目标图像）和PASCAL VOC 2012（600张图像）作为训练集，200张ECSSD作为测试集。

**📈 对比分析**

与传统的部分交叉熵（PCE）及其加正则化方法相比，本方法在mIoU、mDice、mAcc上分别提升约3%/2%/1%，接近全监督交叉熵的性能。

**⚠️ 局限性**

局限性包括：需要预先计算RKHS扩展，增加计算和存储开销；核函数与正则化参数仍需经验性选择；在测试阶段TD正则化效果有限，可能导致过度平滑或漏标。

---

## 159. Robust Bichromatic Classification in 3D Using Planes and Slices

**arXiv ID:** 2607.19754 | [PDF](https://arxiv.org/pdf/2607.19754v1)

**作者:** Grittin Nuntasombat `[一作]` (Kasetsart University), Jittat Fakcharoenphol `[通讯]` (Kasetsart University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文提出了在三维空间中使用两条线性分隔器（平面、切片、楔形、双楔形）实现鲁棒的双色分类（红点与蓝点）的方法，目标是最小化红蓝点的异常点数量；

**💡 创新点**

创新点在于将二维平面分类的思想通过点-平面对偶扩展到三维，并通过对偶空间中顶点、线交点的结构性分析，给出多种分类器的最优解构造与多项式时间算法；

**🔧 技术方法**

核心技术包括点-平面对偶变换、扫描框架（sweep）、DCEL（双连通边列表）以及事件驱动的范围管理与优先队列，整体实现了 O(n^6)、O(n^3 log n) 等多种时间复杂度的算法；

**📊 数据集**

论文未使用实测数据集，所有结果均为理论分析与算法复杂度证明；

**📈 对比分析**

在理论上，针对半平面和切片分类器分别实现了 O(n^3) 与 O(n^3 log n) 的最优算法，显著优于之前的 O(n^7) 或 O(n^6) 的暴力搜索方法；

**⚠️ 局限性**

局限性包括：(1) 对楔形和双楔形分类器的高效算法仍未给出，现有仅为 O(n^6) 的通用解；(2) 只关注三维空间，扩展到更高维仍是未解决的问题；(3) 实际应用中对噪声鲁棒性与实际数据分布的评估缺失。

---

## 160. Efficient Tracking and Understanding Object Transformations

**arXiv ID:** 2607.19743 | [PDF](https://arxiv.org/pdf/2607.19743v1)

**作者:** Yihong Sun `[一作]` (Cornell University), Bharath Hariharan `[通讯]` (Cornell University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 SAM2 多头掩模不一致的反应式候选检测方法，替代 TubeletGraph 的全局视频分区，从而显著加速对象状态变换跟踪并生成状态图。

**💡 创新点**

创新点在于：1) 使用 SAM2 内部多掩模不一致作为轻量级变换触发器，避免在每帧和每个区域进行实体分割；2) 采用逆向跟踪代替语义一致性检查，消除额外模型依赖。

**🔧 技术方法**

使用技术包括 SAM2 视频分割、CropFormer 细粒度分割、逆向 SAM2 跟踪、以及 GPT‑4 生成状态图，并通过阈值策略实现触发与过滤。

**📊 数据集**

实验数据集包括 VOST、VSCOS、M3‑VOS 的变换追踪基准，以及 DAVIS17 用于无变换验证；VOST‑TAS 用于评估状态图。

**📈 对比分析**

与 TubeletGraph、SAMURAI、SAM2.1 等基线比较，测算每组件时间、MPS、Jaccard 等指标，在四个基准上实现 3.3–10.7 倍速度提升，跟踪精度至少相当或略优（如 VOST J=52.8 对比 50.9）。

**⚠️ 局限性**

局限性：1) 对于快速或几乎不产生多掩模不一致的变换可能漏检；2) 依赖 SAM2 的多头输出，若未来跟踪器不具备此机制则失效；3) 仅能检测一次性变换，无法处理更高阶的变换。

---

## 161. Fast Wave-optics Rendering of Multiplane Images for 3D Holographic Displays

**arXiv ID:** 2607.19731 | [PDF](https://arxiv.org/pdf/2607.19731v1)

**作者:** Brian Chao `[一作]` (Stanford University), Changwon Jang `[通讯]` (Meta)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于多平面图像（MPI）的波光学计算机生成全息图方法，能在保持高质量重建的同时显著加速全息合成。

**💡 创新点**

创新点在于将MPI直接映射为随机相位全息图，采用层级波动相互作用和时域多路复用，突破传统原始体素/几何体积方法在速度与质量上的折中。

**🔧 技术方法**

核心技术包括随机相位编码、角谱传播、波光学α混合、时域多路复用以及对MPI的前向/后向层级合成。

**📊 数据集**

使用了NeRF Blender合成数据集和Mip-NeRF 360真实采集数据集进行验证。

**📈 对比分析**

与RGBD全息图和基于原始几何的随机相位波展（RPWS）进行对比，MPI全息图在保持与RPWS相近的SSIM/PSNR的同时，速度提升高达250,000×；相比RGBD则质量显著提高。

**⚠️ 局限性**

局限性包括尚未实现实时渲染、随机相位导致对比度下降以及对MPI生成过程的依赖。

---

## 162. Contact-Persistent Full Actuation for Aerial Physical Interaction

**arXiv ID:** 2607.19708 | [PDF](https://arxiv.org/pdf/2607.19708v1)

**作者:** Abhimanyu Khadga `[一作]` (University of Cincinnati), Shashi Ranjan Kumar `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了“接触持续全致动”框架，用残差力矩（wrench）余量来评估全致动无人机在持续接触任务中的可操作性，并给出了可计算的余量证书、签名余量判定以及任务集合的统一余量评估。

**💡 创新点**

创新点在于将全致动从传统的矩阵秩判定转化为基于可行力矩多面体内部性和边界距离的判定，提出残差余量半径为任务力矩到多面体边界的最短距离，并引入签名余量来同时检测可行、边界与不可行情况；同时给出了基于激活量的可行性下界和过滤器设计。

**🔧 技术方法**

使用了线性力矩分配模型、可行力矩多面体的 H 表示、最小二乘/线性规划求解、签名余量公式、方向可达性与滑动距离方法、以及基于右逆的保留余量下界；在数值仿真中还用到了 Monte‑Carlo 验证。

**📊 数据集**

论文未使用公开数据集，而是在抽象的对称六旋翼机模型上进行参数化仿真，采用了质量、臂长、拖曳系数等物理参数，并以推力与扭矩的归一化形式进行分析。

**📈 对比分析**

与传统秩判定、最小奇异值等结构性评估相比，该方法能更准确地判断任务力矩是否可实现并保留稳定余量；数值结果显示，在中等倾斜角度下可获得正余量，甚至比固定倾斜角更高的余量，并能通过自适应倾斜提升可推力容量约 8.7%（平均）和 31.3%（最佳方向）。

**⚠️ 局限性**

局限性包括：仅考虑静态多面体余量，未覆盖倾斜伺服的动态速率限制和瞬态失配；对闭环姿态/力矩动力学的稳定性分析缺失；以及对高阶约束（能量、健康等）和实际硬件误差的考虑不足。

---

## 163. NavVerse: Benchmarking Indoor-to-Outdoor Embodied Navigation in Continuous Robot Simulation

**arXiv ID:** 2607.19695 | [PDF](https://arxiv.org/pdf/2607.19695v1)

**作者:** Junzhe Wu `[一作]` (University of Michigan), Maani Ghaffari `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了NavVerse，一套基于物理仿真的全栈导航基准，涵盖室内、室外以及室内‑室外连通场景，支持目标导航（ObjNav）、地点导航（PlaceNav）和视觉语言导航（VLN）三种任务，并通过任务成功率、路径效率和安全指标进行评价。

**💡 创新点**

创新点包括：①统一室内、室外与连通室内‑室外三类场景，并在连续机器人执行环境下评估；②引入长距离地点导航（PlaceNav）挑战，要求机器人在跨尺度、跨语义的场景中搜索地点；③提供安全诊断指标（碰撞率、障碍距离、可行走表面比例），实现对物理可执行性的全面评估。

**🔧 技术方法**

采用了NVIDIA Isaac Sim实现GPU加速的视觉与物理双重仿真，支持连续控制、Waypoint跟随以及PID控制器；利用场景生成与Oracle验证技术构建可执行的导航任务；结合多任务评估协议，统一了多种机器人执行接口。

**📊 数据集**

使用了100个室内场景（基于GRUtopia网格）、50个室外场景（Virtual Community）以及50个室内‑室外连通场景，共构成约10,000条任务（ObjNav、PlaceNav、VLN），场景中包含大量家具、商店、道路、车辆等真实感对象，且通过物理材质划分实现可行走表面标注。

**📈 对比分析**

对比了四种基线：模块化的SGImagineNav、RL的PoliFormer、端到端视觉语言动作模型UniNaVid和混合VLA+RL的LongNav‑R1。在所有任务上，VLA模型UniNaVid获得最高零射击成功率，但整体成功率仍低；模块化SGImagineNav在安全指标上表现最好；RL方法在室外探索和时间限制上表现不佳，展示了跨场景适配的瓶颈。

**⚠️ 局限性**

局限性主要体现在：①场景仅为单层平面，缺少多层楼层和垂直过渡结构；②缺少动态障碍物（行人、车辆）和交互式门窗等社会化导航元素；③仅评估室内→室外过渡，未涵盖室外→室内；④对物理动力学的依赖仍有限，未覆盖更复杂的动力学模型和环境条件。

---

## 164. A Diagnostic Evaluation Framework for AI-Generated Cover Songs Using Music-Theoretic and Acoustic Features

**arXiv ID:** 2607.19688 | [PDF](https://arxiv.org/pdf/2607.19688v1)

**作者:** Yingxin Liang `[一作]` `[通讯]` (Hangzhou Xiaoying Innovation Technology Co., Ltd.), Yingxin Liang (Hangzhou Xiaoying Innovation Technology Co., Ltd.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出五维诊断框架评估AI生成的翻唱歌曲，并构建30个样本基准，对九个自动特征和规则进行实验。

**💡 创新点**

设计了针对旋律、和声、调性、风格、编曲的多维诊断维度，揭示调性与和声不等价，以及自动特征无法完全替代专家判断。

**🔧 技术方法**

采用源分离（Demucs）、音高转录（Basic Pitch）、MIDI解析（pretty_midi、music21）、音频特征提取（librosa、pyloudnorm）以及Spearman相关、留一交叉验证等技术。

**📊 数据集**

使用5首不同风格的原曲与6种生成系统生成的30个翻唱样本，全部由单一专业评审按时间戳记录。

**📈 对比分析**

通过专家评分与自动特征相关、规则分类与多数类基线对比，发现自动特征关联弱，规则模型未显著优于基线，主要错误集中在和声和编曲。

**⚠️ 局限性**

局限包括单个注释者、样本量小、源曲聚类有限、转录误差、缺乏直接和声与风格自动指标、音乐风格覆盖不足。

---

## 165. Reference-Free Evaluation of Reasoning in Open-Ended Question Answering

**arXiv ID:** 2607.19678 | [PDF](https://arxiv.org/pdf/2607.19678v1)

**作者:** Guneet Singh Kohli `[一作]` (Queen Mary University of London), Maria Liakata `[通讯]` (Queen Mary University of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于推理链的无参考审核框架，通过将LLM生成的推理过程拆分为语句段、使用NLI判定段与段之间的逻辑关系，并构建超图进行递归支持搜索，得到每段的支持状态。

**💡 创新点**

创新点包括：①首次将四标签NLI（Entailment、Implied、Neutral、Contradiction）与超图结构结合，自动判定推理链是否自洽；②提出后向AND–OR搜索算法以全局评估段落的支持度；③创建UroReason医学推理基准，用真实病例与医生标注检验方法的实用性。

**🔧 技术方法**

核心技术：自然语言推理（NLI）模型做局部推理关系标注；超图（hypergraph）建模和后向AND–OR搜索算法；对比实验中使用LLM-as-judge与NLI超图审核两种评估范式。

**📊 数据集**

使用两个数据集：Hard2Verify（数学推理）和新构建的UroReason（40例儿科泌尿科真实病例，共305条推理段）。

**📈 对比分析**

比较方法：与传统LLM-as-judge直接评估进行对照，并在两类任务上计算TPR、TNR、Balanced F1。结果表明：在数学任务上两种方法相近；在医学任务上LLM-as-judge TNR低，容易误接受不可信段，NLI超图审核显著提升Balanced F1（从≈0.07提升到≈0.55），且在不同模型间表现更稳定。

**⚠️ 局限性**

局限性：①依赖LLM进行NLI标注，缺乏独立的专用NLI模型；②未探究推理时的规模化与多次采样、投票等策略；③UroReason规模有限，仅涵盖儿科泌尿科，未覆盖其他医学或高风险领域；④框架侧重无参考评估，未验证对外部事实真伪的校正能力。

---

## 166. Multi-Mask Diffusion Language Models for Few-Step Generation

**arXiv ID:** 2607.19686 | [PDF](https://arxiv.org/pdf/2607.19686v1)

**作者:** Sijin Chen `[一作]` (ByteDance Seed), Lexing Ying `[通讯]` (ByteDance Seed)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出多掩码离散扩散模型（Multi‑Mask Diffusion），并实现其训练和一致性蒸馏算法，用于文本生成。

**💡 创新点**

创新点包括：① 用多掩码子空间替代单一吸收掩码，解决终端熵消失问题；② 设计前向过程与闭式 ELBO，加入掩码内识别项；③ 通过共享 Gumbel 耦合构造低熵路径，并推导一致性目标；④ 兼容预训练的单掩码 MDM，支持持续适配。

**🔧 技术方法**

使用技术：离散扩散（MaskDiffusion）+多掩码设计、ELBO 训练、共享 Gumbel 路径耦合、一致性蒸馏、低熵后验近似、Transformer‑DiT 基座、LoRA 适配。

**📊 数据集**

数据集：OpenWebText、LM1B、MATH500、GSM8K、HumanEval、MBPP。

**📈 对比分析**

与 DUO（均匀‑状态扩散）、MDLM（单掩码扩散）和 CANDI（混合离散‑连续模型）对比；在 OpenWebText/LM1B 上预训练得到更低的生成困惑度和更优的困惑度‑熵 Pareto 前沿；在一致性蒸馏后，4 步采样得到更好的困惑度‑熵曲线；在大模型适配上，Multi‑Mask LLaDA 在代码与数学评测任务中多项指标提升。

**⚠️ 局限性**

局限性：尚未在更大规模模型上完成共享 Gumbel 一致性蒸馏的验证；对掩码映射的语义优化未深入探索；在高阶任务中的性能提升仍有限；理论上对多步采样的收敛性与稳定性保障尚不完整。

---

## 167. The World Model Remembers, the Actor Forgets: Dream Rehearsal for Continual Model-Based RL

**arXiv ID:** 2607.19749 | [PDF](https://arxiv.org/pdf/2607.19749v1)

**作者:** Gurp Nijjer `[一作]` `[通讯]` (Quantegra Research), Gurp Nijjer (Quantegra Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究模型基强化学习中DreamerV3系列在连续任务训练时的遗忘现象，定位为演员（policy）衰退，并提出“graded dream rehearsal”方案通过自监督的梦境克隆来保持记忆。

**💡 创新点**

将遗忘归因于RL梯度通道而非记忆本身，提出通过冻结世界模型与对已生成梦境进行“realized-first”评分的自监督行为克隆作为持续学习机制，在MiniGrid上实现4任务/8任务的完整保留。

**🔧 技术方法**

使用DreamerV3模型、永不清除的重放缓冲、世界模型冻结实验、实时奖励/价值评分的“realized-first”梦境打分、行为克隆与RL在梦境中的对比、离线评分仪表盘。

**📊 数据集**

使用MiniGrid 64×64 RGB观测的多任务链（DoorKey, SimpleCrossing, LavaGap, MultiRoom 等共4/8个任务）。

**📈 对比分析**

与传统永不清除重放、分任务演员快照（CLEAR）以及匹配的真实回放克隆进行对比；graded dream rehearsal 在4任务中 3/3 seed 通过率，保留率平均 0.82；在8任务中 3/3 seed 通过率，平均 0.86，显著优于其他基线。

**⚠️ 局限性**

仅在MiniGrid小规模环境与17M参数DreamerV3下验证；种子数仅3，无法覆盖大规模多任务或复杂探索；重放缓冲线性增长导致规模瓶颈；未研究更大模型或非gridworld任务。

---

## 168. An Automated Framework for Extracting Reachable Attack Chains from Cyber Threat Intelligence Reports

**arXiv ID:** 2607.19742 | [PDF](https://arxiv.org/pdf/2607.19742v1)

**作者:** Wenbo Hou `[一作]` (Harbin Institute of Technology), Wenjian Luo `[通讯]` (Harbin Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种自动化框架，将CTI报告中的攻击过程转化为可达攻击链。

**💡 创新点**

创新点在于将每个攻击步骤建模为预条件-行为-后置条件三元组，并通过多阶段LLM辅助管道及诊断修复实现可执行的状态转移规则。

**🔧 技术方法**

技术包括行为词汇表约束、条件抽取、谓词规范化、诊断指导修复以及将攻击单元编译为Datalog规则的推理。

**📊 数据集**

使用20份公开CTI报告共334个人工标注的可执行步骤作为数据集。

**📈 对比分析**

与AttacKG+、CRUcialG和CTINEXUS等方法对比，本文框架在攻击步骤覆盖率和谓词精确度上均显著优于对手（94.9%覆盖率），且前向推理在19/20份报告中成功到达目标。

**⚠️ 局限性**

局限性包括对专家手工构建谓词词汇表的依赖、同义谓词归一化不完全、对隐式环境假设的处理不足以及可能合并独立攻击分支导致的错误链。

---

## 169. Analytic Distribution of Classifier-Free Guidance for Schedule Design

**arXiv ID:** 2607.19725 | [PDF](https://arxiv.org/pdf/2607.19725v1)

**作者:** Enze Jiang `[一作]` (Shanghai Jiao Tong University), Zheng Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过概率流ODE分析分类器无指导（CFG）的采样分布，并提出了精确的路径积分表示；

**💡 创新点**

首次给出CFG真实采样分布的解析公式，并基于此推导出时间可调的指导调度方案DG‑CFG；

**🔧 技术方法**

使用Fokker–Planck方程、概率流ODE、路径积分和理论推导；

**📊 数据集**

在离散圆形点模型验证理论，并在Stable Diffusion 1.5（MS‑COCO 2017验证集）上进行实验；

**📈 对比分析**

与恒定CFG、Interval CFG、β‑CFG等基线进行比较，DG‑CFG在高指导强度下显著提升多样性、减少饱和、保持或提高图像质量，且在固定质量阈值下需要更少的采样步数；

**⚠️ 局限性**

仅校正了噪声调度的时序非均匀性，未针对得分差异的几何结构做进一步优化，且实验主要聚焦于VPSDE/DDIM设置，尚未验证在其他模型或更复杂条件下的普适性。

---

## 170. Ultra-Compact CNN Architectures for Tropical Bird Audio Detection on Microcontrollers

**arXiv ID:** 2607.19721 | [PDF](https://arxiv.org/pdf/2607.19721v1)

**作者:** Muhammad Mun'im Ahmad Zabidi `[一作]` (Universiti Malaya), Norisma Idris `[通讯]` (Universiti Malaya)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了一系列轻量级卷积神经网络（DrongoNet-Nano、DrongoNet-Micro、DrongoNet-Edge），用于在嵌入式ARM Cortex‑M4及Linux SBC平台上进行热带鸟类音频检测（二分类），并将其部署为热带生态监测的门控触发器。

**💡 创新点**

创新点包括：
1) 四阶段消融方法（频率分辨率、全局平均池化、焦点损失、可学习频率强调层）实现仅 5–33 kB 的 INT8 模型；
2) 用可学习频率强调层替代 batch‑norm，保证量化后稳定性；
3) 在 50k 录音、1677 种鸟类的 SEABAD 数据集上实现与单类 TinyChirp 同等或更优的 AUC，同时模型参数 28× 更少；
4) 提供统一的阈值策略和量化方案，使模型可直接替代传统 Goertzel 触发器，显著提升检测召回率并延长 SD 卡使用时间。

**🔧 技术方法**

采用的技术包括：
- 轻量级 CNN（深度可分离卷积被剔除），
- 全局平均池化（替代全连接层），
- Focal loss 进行低阈值校准，
- 可学习频率强调层（int8‑安全的频率加权），
- 频谱预处理（log‑mel 3‑秒 16 kHz 输入），
- TensorFlow Lite for Microcontrollers + CMSIS‑NN 进行 INT8 量化与部署，
- 用 TFLite Micro 在 STM32H747XI 上评估推理延迟与功耗。

**📊 数据集**

使用的数据集：
- SEABAD（50 000 三秒录音，1 677 种热带鸟类，等比例正负样本）；
- TinyChirp（单种 Corn Bunting），
- BirdNET、DCASE‑2018 等公开数据集用于零射转移与对比实验。

**📈 对比分析**

比较方法与性能：
- 与重训练的 TinyChirp、VGG‑16、ResNet‑50、EfficientNet‑B0、MobileNetV3‑S 等标准 CNN 在 SEABAD 上对比 AUC、召回率、模型大小、推理延迟与功耗；
- DrongoNet‑Micro 以 0.9810 AUC、98.3% 召回、6.26 kB 取得与 TinyChirp 相近的 AUC（0.9815），但参数 28× 更少；
- DrongoNet‑Edge 以 0.9991 AUC、99% 召回、33.06 kB，进一步提升 AUC 1.8pp；
- 在实际热带监测场景中，Micro 在 τ=0.35 时比 Goertzel 提高约 8pp 的检测率，将 32 GB 卡的持续监测时间从约 28 天提升至 45 天。

**⚠️ 局限性**

限制：
- 模型在跨地区或跨语料的零射转移中表现差，需要针对目标环境重新训练；
- 仅提供二分类，需下游分类器获取物种信息；
- 在极低内存/低功耗平台上仍需进一步优化，尤其是功耗与误检率的平衡；
- 评估基于人工标注的测试集，未在长期野外真实环境中验证；
- 高召回阈值下的电池寿命仍低于无记录模式，需权衡。

---

## 171. Efficient Clustering with Provable Guardrails for LLM Inference at Scale

**arXiv ID:** 2607.19704 | [PDF](https://arxiv.org/pdf/2607.19704v1)

**作者:** Longshaokan Wang `[一作]` (Amazon), Francesc Moreno-Noguer `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过两阶段聚类（Mini-batch K‑Means + 贪心集覆盖）将LLM输入划分为满足相似度阈值与属性匹配的代表样本，从而大幅降低推理成本。

**💡 创新点**

首次实现对每个样本的相似度与属性约束的精确保证，采用Johnson–Chvátal贪心集覆盖得到(1+ln n)近似的集覆盖并保持线性时间复杂度。

**🔧 技术方法**

使用Mini-batch K‑Means、余弦相似度嵌入、贪心集覆盖算法以及后置重分配步骤，整体实现了可扩展的两阶段聚类流程。

**📊 数据集**

在内部38 M客户 persona 数据集以及公开的 AG News 和 Cosmopedia 数据集上进行了实验与评估。

**📈 对比分析**

与 K‑Means、Mini‑Batch K‑Means、Agglomerative、BIRCH、Spectral、Gaussian Mixture 等基线在相同聚类数下对比，最小相似度始终满足阈值，运行时间提升 10–1000 倍，最终在生产中实现 50 倍计算/延迟下降。

**⚠️ 局限性**

需先预设相似度阈值和属性匹配；划分噪声可能略增集数；贪心集覆盖的序列化影响分布；对动态增删数据需重新聚类。

---

## 172. Bridging Behavior and Implementation: Automated Java Glue Code Generation for Behavior-Driven Development

**arXiv ID:** 2607.19703 | [PDF](https://arxiv.org/pdf/2607.19703v1)

**作者:** Xinyu Shi `[一作]` (University of Alberta), An Ran Chen `[通讯]` (University of Alberta)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于层级多智能体的框架AutoGlue，用于自动生成Java BDD glue代码。

**💡 创新点**

创新点在于：①把生成过程拆分为行为解释、上下文检索和代码生成三阶段；②采用行为解释消除步骤的歧义；③通过项目相关上下文检索提升代码质量；④首次在真实项目中系统评估该方法。

**🔧 技术方法**

使用了大型语言模型（如Gemini-2.5-Flash、Claude‑Sonnet‑4.5），LangGraph实现多智能体；通过嵌入检索、项目代码分析和行为摘要指导代码生成。

**📊 数据集**

数据集：1,307个步骤与对应glue代码对，来自八个开放源码Java项目，涵盖多领域。

**📈 对比分析**

与few‑shot、plain prompting两种LLM提示基线对比，指标包括API Precision/Recall/F1、CodeBLEU、METEOR、ROUGE‑L及LLM‑as‑Judge可用性评分；AutoGlue平均提升API F1 58.7%、CodeBLEU 43.7%，约46%生成代码可直接使用。

**⚠️ 局限性**

局限性包括：1）仅在Java环境验证；2）对复杂多步骤场景、背景步骤等缺乏完整支持；3）仍需人工微调；4）评估依赖LLM‑as‑Judge，可能存在偏差。

---

## 173. RCC: Speculative Write Versioning with Redo Logs

**arXiv ID:** 2607.19697 | [PDF](https://arxiv.org/pdf/2607.19697v1)

**作者:** Hyejin Yoo `[一作]` (Seoul National University), Jonghyeok Park `[通讯]` (Korea University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用事务重做日志作为写版本，支持在事务提交前产生可推测的写版本，从而消除写–写冲突，并在提交时按依赖顺序安装到数据页；并在此基础上加入按列粒度的冲突检测以及仅在提交时检测死锁的机制；实现后在 MySQL 与 PostgreSQL 上无缝集成。

**💡 创新点**

创新点在于：①把重做日志（原本仅用于恢复）重构为可并行的写版本存储；②通过“延迟更新 + 依赖图”实现事务在执行期间解锁，允许冲突事务流水线执行；③首次提出“提交时死锁检测”，在提交阶段一次性检测循环，避免每次加锁时的死锁检查与误中止；④利用列级重做日志实现列级冲突排除，减少无意义的 WW 依赖与锁等待。

**🔧 技术方法**

使用技术包括：重做日志作为版本存储；事务级别的 TLA（Transaction Log Area）存储重做条目与临时版本；基于锁表的依赖图维护与提交时的循环检测；列级重做日志分片实现列级依赖；延迟更新和预提交（pre‑commit）机制；与现有数据库的缓冲区、锁管理、恢复模块无缝集成。

**📊 数据集**

主要数据集：TPC‑C（1个仓库）、YCSB‑A（读写比例0.9）、以及内部实验使用的 Wikipedia、AuctionMark、TATP、Epinions 等。通过这些基准评估系统在高并发、多核环境下的吞吐量与延迟。

**📈 对比分析**

对比方法：与 MySQL/PostgreSQL 的原生 MVCC（两阶段锁、Snapshot Isolation）以及 Bamboo 等现有的推测写版本方案。实验结果显示：在 64 线程/128 核的机器上，RCC 的 TPC‑C 事务吞吐量提升约 10×，延迟降低约 4×；在 YCSB‑A 128 客户端场景下，RCC 能保持线性扩展，而竞争方案在 32 线程后性能急剧下降。性能提升主要来自消除 WW 冲突导致的锁等待、减少死锁处理开销以及列级冲突消除。

**⚠️ 局限性**

局限性：①在高事务放弃率或频繁重做冲突的场景下，仍会出现连锁放弃；②依赖图维护与提交时的循环检测仍有额外内存与 CPU 开销；③对列级依赖的实现需要在重做日志中保存列位图，增加日志大小；④需要对锁管理和事务调度做细粒度调整，且在极高并发下仍可能遇到假死锁导致的放弃。

---

## 174. SLPO: Scaling Latent Reasoning via a Surrogate Policy

**arXiv ID:** 2607.19691 | [PDF](https://arxiv.org/pdf/2607.19691v1)

**作者:** Runyang You `[一作]` (Hong Kong Polytechnic University), Wenjie Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SLPO（Surrogate Latent Policy Optimization）方法，将基于结果奖励的强化学习引入自回归潜在推理模型，实现在潜在空间中进行轨迹级信用分配并通过停止门实现自适应推理长度。

**💡 创新点**

创新点在于：① 用经验估计的对角高斯 surrogate likelihood 代替缺失的连续潜在转移概率；② 通过正确性监督的停止门预训练，使得模型可在推理时决定何时终止；③ 将上述两个组件统一到 RL 目标中，使潜在推理能够获得基于最终答案奖励的梯度。

**🔧 技术方法**

技术手段包括：MC‑Dropout 产生多样化潜在轨迹；经验 Gaussian surrogate 计算潜在转移的对数似然；停止门（sigmoid 头）用于产生停机概率；policy‑gradient RL（如 RLOO、GRPO）与基线估计；评估指标 Pass@k 及 deterministic accuracy。

**📊 数据集**

数据集主要为小学算数推理任务：训练集 GSM8K‑Aug，评估集包括 GSM8K‑Test、Harder‑GSM8K（数值更大）、Multi‑Step Arithmetic；软词推理的迁移实验还使用了 Llama‑3.2‑1B、其它公开基准。

**📈 对比分析**

与 COCONUT、CODI、CoLaR、ReGuLaR、DART、Latent‑SFT 等潜在推理基线以及 explicit CoT（CoT‑SFT、iCoT）进行对比。SLPO 在 Pass@k（k=8,16）上显著提升，且在更难实例上推理长度更长、确定性准确率提高；整体在并行采样下表现优于传统方法。

**⚠️ 局限性**

局限性：目前实验仅覆盖小规模算数推理模型，未验证大模型或开放式推理；潜在转移的 Gaussian surrogate 为经验近似，可能在更高维度或不同分布下失效；停止门依赖先验正确性监督，若训练样本多样性不足可能导致停机策略偏差。

---

## 175. Benchmarking the Full Pipeline of Materialized-View-Based Query Rewriting

**arXiv ID:** 2607.19679 | [PDF](https://arxiv.org/pdf/2607.19679v1)

**作者:** Xinjie Hu `[一作]`, Zhengjie Miao `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对材料化视图（MV）查询重写的完整流水线（枚举、选择、重写）进行端到端、跨引擎的系统性评估，构建了可插拔的评估框架和跨引擎协议；

**💡 创新点**

创新点在于：①首次统一评估三阶段全链路性能并揭示阶段间交互影响；②提出跨引擎协议以比较仅暴露执行计划的商业系统与可输出SQL的系统；③系统地诊断并归纳失败模式，为实践提供决策指南；

**🔧 技术方法**

采用的技术包括：多方法枚举器（基于查询子表达式、join-graph、MVPP、工业集成）、多种选择器（ILP、GNN、DQN、工业内部）、多种重写器（Calcite、Hive、StarRocks、Doris、Oracle、Snowflake、BigQuery、Redshift、Azure Synapse），以及统一接口、基准脚本、可视化分析；

**📊 数据集**

使用的基准数据集包括TPC‑DS、TPC‑CH、S4, JOB、STATS等四个工作负载，规模从数GB到十GB，覆盖不同的join、谓词复杂度；

**📈 对比分析**

通过在PostgreSQL上做端到端组合实验，和在七个引擎上做跨引擎对比实验，发现：单一阶段排名并不能决定整体收益；不同枚举器/选择器/重写器组合在不同工作负载和预算下表现差异明显；在最优组合下可获得58%–75%的工作负载时间缩短，但跨引擎迁移并不一致；重写器往往是瓶颈，且可能导致回归；

**⚠️ 局限性**

局限性包括：仅考虑静态MV，不考虑刷新和维护成本；对计划透明系统的观察受限，部分优化决策不可见；实验基于离线数据和固定硬件，未覆盖动态负载变化和内存压力场景。

---

## 176. Covering Planar Lattices with Interior-Disjoint Unit Disks

**arXiv ID:** 2607.19764 | [PDF](https://arxiv.org/pdf/2607.19764v1)

**作者:** Nattawut Phetmak `[一作]` (Kasetsart University), Jittat Fakcharoenphol `[通讯]` (Kasetsart University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了平面中周期点集的无限变体硬币覆盖问题，探讨了是否可以用成对不重叠的单位圆盘覆盖所有点。

**💡 创新点**

提出了新的周期图案模式，证明了多个可覆盖间隔，特别是针对三角格、方格和蜂窝点集的覆盖能力。

**🔧 技术方法**

使用了图案构建技术，结合了周期性图案和非均匀模式。

**📊 数据集**

研究了三角格、方格和蜂窝点集的点集。

**📈 对比分析**

与已有方法进行了比较，发现了一些重叠和间隔的缺口，并通过新的图案恢复了部分间隔，性能上实现了更广泛的覆盖能力。

**⚠️ 局限性**

限制在于这些间隔是构造性的充分条件，而不是覆盖能力的完整表征，且在某些情况下可能难以关闭间隔。

---

## 177. Investigation of STEEP for Secure Communications Over SIMO and MISO Channels Subject to Full-Duplex Jamming and Eavesdropping

**arXiv ID:** 2607.19729 | [PDF](https://arxiv.org/pdf/2607.19729v1)

**作者:** Md Saydur Rahman `[一作]` (University of California at Riverside), Yingbo Hua `[通讯]` (University of California at Riverside)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在多天线Eve全双工干扰与窃听环境下，采用STEEP（通过加密探测信号回声实现机密信息传输）的SIMO与MISO通道下的安全通信；

**💡 创新点**

创新点在于首次给出了在Eve具备多天线且可进行最佳干扰与窃听的情形下，STEEP方案可实现正密度率的解析结果，并将其与传统SIMO/MISO方案对比；

**🔧 技术方法**

使用了MMSE估计、矩阵求逆定理以及信息论容量分析等技术，对STEEP在两种模式（SIMO-MISO-STEEP 与 MISO-SIMO-STEEP）下的密度率进行了推导；

**📊 数据集**

实验数据来源于10⁵次独立的复高斯随机衰落通道样本（h和H矩阵均为𝒞𝒩(0,1)），并设定Eve干扰功率为10 dB；

**📈 对比分析**

与传统SIMO与MISO方案相比，STEEP在Eve拥有多天线且接收通道强于合法用户时，能够始终保持正密度率；仿真显示其在大多数场景下显著优于传统方案，尤其当Eve天线数大于Alice时优势更为明显；

**⚠️ 局限性**

主要局限包括：假设Eve完全知道所有信道状态且采用最优干扰；要求在STEEP中能精确估计探测信号；同时对全双工自干扰和CSI误差等实际问题未做详细分析。

---

## 178. Information Propagation and Contraction in Functional Interpretations

**arXiv ID:** 2607.19723 | [PDF](https://arxiv.org/pdf/2607.19723v1)

**作者:** Chuangjie Xu `[一作]` `[通讯]`, Chuangjie Xu

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过引入信息核（information nucleus）和收缩结构（contraction structure）来构造可解释的功能解释框架，完成了对有限类型算术的仿射与全变形的可解释性证明。

**💡 创新点**

创新点在于将信息传播与收缩操作完全分离，形成一个最小的代数接口，并通过该接口系统化地在提取的实现者中加入连续性等辅助数据。

**🔧 技术方法**

主要技术包括信息核的代数定义、克莱斯基扩展的提升、公式翻译与受限量化、以及对收缩结构的公式索引化验证，配合对仿射与收缩系统的严格可解释性证明。

**📊 数据集**

本文为纯理论研究，不涉及实验数据集；因此未使用任何具体数据集。

**📈 对比分析**

没有实验或性能比较；论文通过形式化证明展示了与现有解释（如 Dialectica、Herbrand）的一致性和可扩展性，但未给出数值性能评估。

**⚠️ 局限性**

局限性包括未覆盖函数型算术中的全/弱外延性、对不同求证者与挑战者类型的支持有限，以及对收缩结构实现的实际可构造性缺乏具体示例。

---

## 179. Same Game, Different Story: A Minimal Conservative Strategic Robustness Benchmark for Large Language Model Agents

**arXiv ID:** 2607.19670 | [PDF](https://arxiv.org/pdf/2607.19670v1)

**作者:** Seyed Pouyan Mousavi Davoudi `[一作]`, Arshia Gharagozlou `[通讯]` (University of Minnesota Duluth)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM在策略游戏中的稳健性，探讨在收益保持不变的前提下，不同提示（商业 vs 友好）是否导致模型行动分布发生变化。

**💡 创新点**

提出了“策略稳健性”概念，即在收益保持等价的提示变换下，模型的行动分布保持不变，并使用最小化的二帧（商业 vs 友好）对齐估计作为可审计的度量。

**🔧 技术方法**

使用Jensen‑Shannon散度衡量行动分布差异，计算鲁棒性得分 R̂，并进行保守的 30% 缩减、二项式自助法置信区间估计以及对数几率（logit）模型的支付敏感性诊断。

**📊 数据集**

依赖于 Lorè 与 Heydari 的公开实验结果，涵盖 GPT‑3.5、GPT‑4、LLaMa‑2 在囚徒困境、雪崩/鸡、猎鹿、囚徒喜悦四种社会困境中，分别在商业和友好两种提示下的合作率，共计 7,200 次决策。

**📈 对比分析**

通过保守缩减后，计算得到聚合鲁棒性 0.783（95% CI [0.774,0.790]）以及友好提示相对商业提示提升合作率 0.307（95% CI [0.297,0.316]）。各模型鲁棒性分别为 GPT‑3.5: 0.967、GPT‑4: 0.651、LLaMa‑2: 0.731，表明模型在不同提示下的稳健性存在显著差异。

**⚠️ 局限性**

局限性包括：依赖于从论文图表重构的计数数据，缺乏原始试验级数据；仅比较两种提示，未覆盖更广泛的提示变体；仅使用单一来源实验，未验证模型在更新版本或不同实现中的表现；以及保守缩减可能低估真实效应。

---

## 180. Convergence-Latency-Aware Adaptive Modulation and Resource Allocation in RIS-Assisted Wireless Federated Learning

**arXiv ID:** 2607.19759 | [PDF](https://arxiv.org/pdf/2607.19759v1)

**作者:** Liwei Wang `[一作]` (Shanghai Jiao Tong University), Qiong Wu `[通讯]` (Jiangnan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在RIS辅助的无线联邦学习系统中，联合优化自适应调制与子信道分配，实现收敛速度与通信延迟的权衡；

**💡 创新点**

①推导符号误差对梯度失真及收敛下降的上界；②基于该界构造收敛-延迟目标；③提出低复杂度交替优化框架（连续松弛+牛顿迭代+二进制恢复），同时兼顾RIS阻断链路；

**🔧 技术方法**

使用RIS技术、OFDM/OFDMA、MPSK/MQAM调制、符号误差模型、凸/非凸优化、混合整数非线性规划、牛顿迭代、KKT条件、二进制恢复以及仿真模拟；

**📊 数据集**

使用MNIST（MLP）、CIFAR‑10（CNN）、Speech Commands（CNN）等数据集，并在IID与Dirichlet非IID划分下进行实验；

**📈 对比分析**

与固定调制（BPSK、QPSK、16QAM）、SDPR（基于SNR的客户端选择）以及无RIS等基线进行比较；实验表明Proposed+RA在所有数据集、不同SNR和客户端数量下收敛速度更快，最终准确率提升约0.3%–1%，尤其在复杂任务和低SNR场景更为显著；

**⚠️ 局限性**

局限性包括：①仅考虑上行符号误差，忽略下行与同步误差；②理论推导基于凸/光滑假设，非凸网络仅作为经验指标；③RIS硬件实现与配置细节未深入讨论；④算法需多次交替迭代，对实时性要求较高。

---

## 181. Learning the Arabic Dialect Continuum as a Continuous Space: A Regression Approach to Speaker Origin Prediction

**arXiv ID:** 2607.19751 | [PDF](https://arxiv.org/pdf/2607.19751v1)

**作者:** Mohamed Aziz Khadraoui `[一作]` (Higher School of Communication of Tunis), Wadii Boulila `[通讯]` (Prince Sultan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于球面坐标回归的连续地理定位框架，用多模态特征（XLS‑R、Whisper、语音韵学）通过Transformer融合，预测阿拉伯语音频的发音者来源坐标。

**💡 创新点**

创新点在于把方言识别从离散分类转为连续坐标回归，采用球面测地距离损失，构建多任务层级网络，并设计城市遮蔽（zero‑shot）评估协议验证连续性假设。

**🔧 技术方法**

使用的技术包括XLS‑R‑300M、Whisper‑large‑v3自监督编码器、手工韵学特征、Transformer编码器、可学习查询池化、von Mises–Fisher 混合密度、对比检索头、EMA快照以及数据增强与混合训练。

**📊 数据集**

数据集为经过质量过滤后的ARCADE电台语音语料，包含2,329条样本，覆盖19个国家、46个城市，使用城市中心经纬度作为回归目标。

**📈 对比分析**

在5折GroupKFold无泄漏协议下，模型平均误差901.5 km、中位误差481.2 km，国家准确率64.5%、城市准确率45.2%；在城市遮蔽的零样本测试中误差升至1,173.3 km，提升约1.32倍。

**⚠️ 局限性**

局限包括数据空间分布不均、城市标签粗糙、缺乏不确定性估计、对多方言和代码混用的鲁棒性不足、以及辅助分类器的校准偏差。

---

## 182. EgoRecovery: Acquiring Failure Recovery Ability Through Human Recovery Demonstration

**arXiv ID:** 2607.19745 | [PDF](https://arxiv.org/pdf/2607.19745v1)

**作者:** Zuhao Ge `[一作]` (Institute of Trustworthy Embodied AI, Fudan University), Yu-Gang Jiang `[通讯]` (Institute of Trustworthy Embodied AI, Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 EgoRecovery 框架，利用第一人称视角的人类恢复数据通过纠正意图迁移到机器人，实现闭环恢复。

**💡 创新点**

创新在于：①将人类恢复信息映射为可跨身体结构的纠正意图瓶颈；②使用恢复门控将意图仅在需要时注入机器人解码器；③通过 10× 的人类数据采集率显著提升恢复数据效率。

**🔧 技术方法**

采用共享视觉骨干 + 体型专属解码头；DCT 低频特征构建纠正意图目标；Gated FiLM 残差调制；恢复门控头；行为克隆、意图损失、门控损失及名义正则化。

**📊 数据集**

自采四个桌面抓取/操纵任务（杯刷插入、桌面扫、圆盘放置、方块堆叠）的机器人成功、机器人恢复、人类成功、人类恢复数据；无公开公开数据集，全部为自建数据。

**📈 对比分析**

与机器人仅恢复、直接人类恢复混合、无门控意图等基线对比；EgoRecovery 在平均恢复成功率上提升至 85%（比直接混合 71%），且保持 80% 的普通执行成功率；在成本匹配下人类恢复替代率达 10×，显著降低机器人恢复采集成本。

**⚠️ 局限性**

局限性：仅适用于同一任务级别的离谱状态；人类恢复无法完全替代机器人特定接触/再抓取等行为的训练；仅在桌面操纵环境验证；需要人类视频采集与标注。

---

## 183. ReFace: Reorganizing Facial Spatiotemporal Representations for Improved Pain Assessment

**arXiv ID:** 2607.19722 | [PDF](https://arxiv.org/pdf/2607.19722v1)

**作者:** Stefanos Gkikas `[一作]` (Honda Research Institute Japan), Raul Fernandez Rojas `[通讯]` (University Of Canberra)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出ReFace，利用四象限空间重排将面部视频划分为四个区域后再进行token化，用于疼痛评估

**💡 创新点**

创新点在于先对面部进行空间重排而非全脸处理，证明空间重排可在相同像素预算下提升识别性能，并且单象限低成本也能保持竞争力

**🔧 技术方法**

主要技术包括MTCNN人脸检测、图像分割、帧采样、轴折叠、通道拼接、Fourier位置编码、基于Transformer的跨注意力与自注意力混合架构

**📊 数据集**

使用AI4Pain数据集（65名受试者、30fps、3类标签）进行训练、验证和测试

**📈 对比分析**

与现有基线（SVM、CNN、CNN-Transformer、Deep Transformer）比较，四象限配置在单模态视频下达到了56.00%准确率，超过了之前最高的55.69%

**⚠️ 局限性**

局限性包括仅在单一数据集上验证、缺乏统计显著性检验、未考虑不同人群与临床环境的泛化能力

---

## 184. A Unified Tokenization Framework for Pain Recognition using Heterogeneous 3D Modalities

**arXiv ID:** 2607.19716 | [PDF](https://arxiv.org/pdf/2607.19716v1)

**作者:** Stefanos Gkikas `[一作]` (Honda Research Institute Japan), Raul Fernandez Rojas `[通讯]` (University of Canberra)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的token化框架，用于将面部视频、fNIRS原始波形和频谱表示等异构3D模态映射到共享的token空间，从而实现单一模型处理多模态痛感识别。

**💡 创新点**

创新点在于：1）无须为每种模态设计专属网络；2）通过轴折叠、Fourier位置编码和分段latent聚合，保留空间、时间及时频结构；3）提出Global‑Latent与Segment‑Latent两种latent组织方案，兼顾准确性与计算效率。

**🔧 技术方法**

采用Transformer‑style多头自注意力与交叉注意力、预归一化与残差连接；对视频使用3D spatiotemporal卷积的预处理；对fNIRS使用PSD谱图或原始1D波形；使用TrivialAugment、AugMix等数据增强与Label Smoothing、Attention Dropout等正则化。

**📊 数据集**

在AI4Pain基准数据集上进行实验，该数据集包含65名受试者的同步面部视频和24通道fNIRS记录。

**📈 对比分析**

与现有手工特征+SVM、CNN、Transformer等方法对比，单模态video达到57.0%准确率，fNIRS单模态53.7%，最优多模态stack（video+fNIRS波形）实现57.33%准确率，超越前沿Transformer（55.69%）与CNN-Transformer（51.33%）等。

**⚠️ 局限性**

局限性包括：1）数据来源单一，受试者样本有限，缺乏跨文化、不同年龄段的验证；2）实验在受控环境下完成，未检验在临床真实情境中的鲁棒性；3）虽然实时性良好，但在低功耗边缘设备上的部署仍需进一步优化。

---

## 185. Morphing MILR: Design and control of a cable-driven limbless robot with rolling joints for maneuvering in complex environments

**arXiv ID:** 2607.19714 | [PDF](https://arxiv.org/pdf/2607.19714v1)

**作者:** Donoven Dortilus `[一作]` (Georgia Institute of Technology), Daniel I. Goldman `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一款可在不同环境下自由切换多种形态的无足机器人Morphing MILR。

**💡 创新点**

通过在每个模块中加入可旋转的滚动基座，将身体弯曲平面重新定向，同时保持可编程的被动顺应，从而实现单机同时支持侧向摆动、侧行、滚动和扭转等多种运动模式。

**🔧 技术方法**

主要技术包括双轴电机驱动的对抗性绳索弯曲机构、齿轮化的滚动基座传动、RS‑485总线电控、滑环与伺服电机组的机械结构，以及基于波形模板的运动生成算法。

**📊 数据集**

实验数据并未使用公开数据集，而是在人工布置的障碍格子、开放地面以及环境转换平台上进行多组实验。

**📈 对比分析**

通过对比无障碍测试、格子测试以及环境过渡测试的成功率、波形精度与模块角速度等指标，结果表明Morphing MILR在大多数测试中均能保持高成功率（如环境过渡中13/15成功），且在格子环境中表现出低卡阻率和良好的机械智能特性。

**⚠️ 局限性**

局限性包括：目前仅在有外部电源和电脑脚本的条件下运行，缺乏自主能量管理与闭环控制；滚动基座的齿轮背隙与摩擦可能限制高速运动；尚未评估能耗、传输效率和多任务协调，也未实现连续平滑的运动模式过渡。

---

## 186. How Fast Can Reward Models Score? A Systems Study of C++ and PyTorch Inference Runtimes for RLHF

**arXiv ID:** 2607.19712 | [PDF](https://arxiv.org/pdf/2607.19712v1)

**作者:** Venkata Naga Sai Vishnu Rohit Pulipaka `[一作]`, Deva Rohit Reddy Peddireddy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个基于 ONNX Runtime 的 C++ 推理引擎，用于 RLHF 奖励模型的推理，并与 PyTorch 生态默认方案（eager、torch.compile、FastAPI）做对比。

**💡 创新点**

通过系统化的重复进程启动测量，揭示 CPU 上 C++ 并非主要加速源，真正提升来自 ONNX Runtime 而非语言本身；同时指出 naïve padding 对吞吐量的严重损害，并在 GPU 上发现 torch.compile 在中位数和 95% 分位数上优于 C++ 引擎。

**🔧 技术方法**

使用 ONNX Runtime、PyTorch 2.10.0、torch.compile、FastAPI、C++ 实现的 SentencePiece/WordPiece tokenizer、长度感知批处理与调度、并发实例对比等技术。

**📊 数据集**

采用 Anthropic 的 hh-rlhf 数据集，主要使用 60 行样本，并多次重抽样到 150 行以验证结果。

**📈 对比分析**

采用独立进程多次启动、计算平均值与 95% 置信区间、Welch t 检验进行比较；CPU 上 C++ 引擎平均比 PyTorch 快 1.7–1.9 倍；GPU 上 C++ 引擎虽快于 PyTorch 但 torch.compile 更快；naïve padding 使吞吐量下降 5–8 倍。

**⚠️ 局限性**

仅在单台配备 AMD Ryzen 7 5800H CPU 和 RTX 3060 Laptop GPU 的机器上测试，未验证不同 CPU 架构、更多 GPU 或更大内存环境；未测量完整 RLHF 训练循环中奖励模型推理所占比例；未测试 Triton 等服务器或多 GPU 部署场景。

---

## 187. Point-Selection Fine-Tuning Framework for Robust Point Cloud Classification

**arXiv ID:** 2607.19711 | [PDF](https://arxiv.org/pdf/2607.19711v1)

**作者:** Da Li `[一作]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy), Dongfu Yin `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于点选择的微调框架 PSFT，用于在噪声/损坏的点云中实现鲁棒分类。

**💡 创新点**

创新点在于将点重要性估计与自适应点筛选、基于选点的层级 prompt 生成以及轻量级特征滤波器结合，既保持冻结预训练模型的鲁棒性，又实现高效的任务适配。

**🔧 技术方法**

使用点重要性评分、熵引导自适应筛选、EdgeConv Prompt 生成、轻量级 MLP 特征滤波与 Beta 采样残差融合等技术。

**📊 数据集**

在 ModelNet-C、ModelNet40-C、ScanObjectNN-C 三个噪声/损坏点云基准上进行评估。

**📈 对比分析**

与全微调、仅头部微调以及其他 PEFT 方法相比，PSFT 在三大基准上显著降低了 mCE/mER，且可训练参数仅为全微调的 10% 左右。

**⚠️ 局限性**

局限性包括对不同后端 backbone 的鲁棒性提升不一致，且在某些真实世界损坏场景下仍未完全匹配最佳性能；此外，点筛选策略对点云密度变化的敏感性待进一步研究。

---

## 188. The Černý Conjecture for One-Cluster Automata via Annular Spectral Descent

**arXiv ID:** 2607.19675 | [PDF](https://arxiv.org/pdf/2607.19675v1)

**作者:** Yinfeng Zhu `[一作]` `[通讯]` (Independent Researcher), Yinfeng Zhu (Independent Researcher)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了同步一簇（one‑cluster）有限自动机满足塞尔尼奇（Černý）猜想，并给出了一个更精确的上界 (m‑1)(n‑1)+mℓ，证明在所有 n≥4 时该上界可被达到。

**💡 创新点**

创新点在于：① 将原本只适用于素周期的平均法（average method）推广到任意周期长度，采用 Fitting 分解和根号单位（roots‑of‑unity）消元实现；② 设计了“环形下降（annular descent）”关系，利用多项式同余与线性代数的组合，在正层级下完成预像增长的控制；③ 构造了强连通的二元自动机族，实现了 (m‑1)(n‑1)+mℓ 的最优性。

**🔧 技术方法**

使用的技术包括：有限维线性代数（矩阵、特征分解、投影算子）、Fitting 分解、根号单位的生成函数与递推、以及对自动机状态空间的过滤空间与实际单词生成器的构造。

**📊 数据集**

没有使用实验数据集，整个工作是理论证明与构造，所有例子都是手工定义的同步一簇自动机。

**📈 对比分析**

与之前的结果比较：对所有同步一簇自动机给出了 (m‑1)(n‑1)+mℓ ≤ (n‑1)² 的上界，严格改进了之前的 2n²‑4n+1‑2(n‑1)ln(n/2) 等上界；构造的例子表明该上界在每个正层级下都是最优的。

**⚠️ 局限性**

局限性：证明仅适用于同步一簇自动机（有一个唯一的循环的字母），对一般同步自动机仍未能解决塞尔尼奇猜想；且方法依赖正层级（ℓ≥1），在 ℓ=0 时仍需借助已知的循环自动机定理。

---

## 189. Optimal Break-Resilient Codes

**arXiv ID:** 2607.19673 | [PDF](https://arxiv.org/pdf/2607.19673v1)

**作者:** Canran Wang `[一作]` `[通讯]`, Canran Wang

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构造了满足任意至多 t 个分段断点的二进制码（n,t)-BRC，且冗余达到 O(t log n)，在 t≤n^{1-ε} 的参数范围内实现信息理论下界的常数因子匹配。

**💡 创新点**

核心创新是利用一个短的代数指纹（fingerprint）与 MDS 码相结合，直接在编码头部携带指纹信息，从而在解码时唯一确定片段的合法排列，避免了多阶段同步所导致的冗余膨胀。

**🔧 技术方法**

主要技术包括：代数指纹计算（多项式评估）、MDS 码保护指纹、互相不相关（MU）码作为标记（marker）、标记移除（Marker‑Removal）变换、以及在解码时对所有可能的片段排列进行一次性检验。

**📊 数据集**

本文为理论性工作，没有使用实验数据集，所有结果均为解析证明与复杂度分析。

**📈 对比分析**

与之前的 O(t log n log log n) 与 O(t log n log log log n) 方案相比，新方案在相同的 t 范围内将冗余降低到 O(t log n)，与信息下界 Ω(t log(n/t)) 匹配（常数因子不同）。

**⚠️ 局限性**

局限性包括：1）对 t 的上界限制为 t≤n^{1-ε}，对更大 t 的性能未知；2）解码算法的时间复杂度含有 (t+1)! 因子，对 t 取值较大时实用性受限；3）虽然冗余最优，但编码/解码的位运算量仍为多项式阶，实际实现需进一步优化。

---

## 190. Overview of FinMMEval 2026 Task 2: Multilingual Financial Short-Answer Question Answering

**arXiv ID:** 2607.19867 | [PDF](https://arxiv.org/pdf/2607.19867v1)

**作者:** Zhuohan Xie `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Preslav Nakov `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在FinMMEval 2026 Task 2中评估了12支队伍在256道多语言财务短答案问答任务上的性能，采用隐蔽测试与宏观平均ROUGE‑1 F1进行评分。

**💡 创新点**

首次将多语言财务问答与短答案生成纳入共享任务，并引入跨语言检索与生成、结构化提示、答案压缩与验证等创新技术。

**🔧 技术方法**

主要技术包括检索增强生成（RAG）、跨语言推理与翻译、结构化提示、答案压缩与验证、以及多模态知识图谱/工具调用等。

**📊 数据集**

使用PolyFiQA扩展版数据集，包含英文、中文、日文、西班牙文、希腊文的财报与新闻作为多语言证据。

**📈 对比分析**

与组织者保留的参考答案使用宏观平均ROUGE‑1 F1比较，最高分约为31%，各系统差距非常小，难以通过单一指标区分优劣。

**⚠️ 局限性**

局限在于ROUGE仅衡量词汇重叠，无法评估事实准确性；缺乏层级分数、基准模型和人工真实性评估，且未公开不同难度层的详细表现。

---

## 191. Local Causal Structure Learning in the Presence of Latent Variables and Selection Bias

**arXiv ID:** 2607.19866 | [PDF](https://arxiv.org/pdf/2607.19866v1)

**作者:** Zheng Li `[一作]`, Feng Xie `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种能够在存在潜在变量和选择偏差的情况下学习目标变量局部因果结构的算法，并证明其在标准假设下是完备且一致的。

**💡 创新点**

首次将Markov blanket作为局部搜索空间与潜在变量/选择偏差兼容的因果图（MAG/PAG）相结合，并给出理论桥梁，允许只学习局部子图即可获得与全局学习相同的目标特定因果信息。

**🔧 技术方法**

采用最大祖先图（MAG）与部分祖先图（PAG）的理论框架，利用基于条件独立性的约束式学习（如TC、FCI）以及自定义的顺序局部学习算法来实现局部学习。

**📊 数据集**

在合成的ER随机图、密度变化图以及四个真实生物网络（MILDEW、BARLEY、ANDES、LINK）上进行评估，并对两组基因表达数据（Arabidopsis与melanoma）做实证验证。

**📈 对比分析**

与多种全局方法（PC、FCI、RFCI、FCI^+等）和局部方法（PCD、MB、CMB、PSL、GraN-LCS、LatentLCD）对比；结果显示在保持更高或相近的局部结构精度（Mark-Precision/Recall/F1、Local-SHD）的同时，显著降低CI检验次数和运行时间，尤其在高维、密集图场景下优势更为明显。

**⚠️ 局限性**

仅在标准因果标记假设下工作，无法识别因果方向不可辨别的边，且对强噪声或样本量不足的场景下仍可能受限；未来需结合先验知识或干预数据进一步提升可辨别性。

---

## 192. Memory-Augmented Multimodal Large Language Models for Small Object Understanding in Streaming Aerial Videos

**arXiv ID:** 2607.19857 | [PDF](https://arxiv.org/pdf/2607.19857v1)

**作者:** Penglei Sun `[一作]` (Hong Kong University of Science and Technology), Xiaowen Chu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 SkyAnchor 框架，解决无人机视频流中小目标的在线指代分割问题。

**💡 创新点**

创新点在于语义感知 Token 路由器与层级记忆池，分别在视觉压缩中保留小目标细节并维持长短期上下文。

**🔧 技术方法**

采用基于 Qwen2.5‑VL 的多模 LLM 骨干、SAM2 解码器，以及 Token 路由器、分层记忆和边缘优化流水线。

**📊 数据集**

使用了首个高分辨率、像素级别的 DroneEyes 数据集，并在 SkyFind 未见海事测试集上评估跨域泛化。

**📈 对比分析**

与多种现有 MLLM 与分割模型对比，SkyAnchor 在 DroneEyes L1、L2 任务上分别提升 ROUGE/MT、J&F 约 15%/20%，在 SkyFind 上超越 SOTA 25% 以上，并在真实 UAV 上实现 5.8 FPS。

**⚠️ 局限性**

局限包括对极高速目标运动的实时性不足、模型对长时间稀疏目标的持续跟踪仍易漂移，以及需进一步压缩和加速。

---

## 193. Using Hierarchical Controlled Vocabularies to Understand CLIP Retrieval Failures in Historical Photo Collections

**arXiv ID:** 2607.19836 | [PDF](https://arxiv.org/pdf/2607.19836v1)

**作者:** Ratan Sebastian `[一作]` (Leibniz Information Centre for Science and Technology), Ralph Ewerth `[通讯]` (Marburg University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在三种历史摄影集上，利用Art and Architecture Thesaurus (AAT) 的根面板类型和层级深度解释 CLIP 在零射与微调条件下的检索成功与失败模式。

**💡 创新点**

首次将 GLAM 专业使用的层次化控制词表结构与 CLIP 的视觉与文本对齐性能关联，并通过分离三种检索失败模式揭示微调对文本-图像对齐的特定改善。

**🔧 技术方法**

采用 CLIP ViT‑L/14 进行零射检索，并使用低秩适配 LoRA 与逐层解冻两种微调策略，结合 d_mean、文本-图像相似度、Recall@K 及 NDCG@K 等指标进行诊断。

**📊 数据集**

使用三组历史黑白摄影集：德国殖民、加泰罗尼亚记者 Sagarra 与 FAIR Photos，每集均标注 AAT 词汇，覆盖数万张图像。

**📈 对比分析**

在零射条件下，AAT 根面板和层级深度与视觉一致性有弱相关但与检索指标无显著关联；微调后 NDCG@10 从 0.114 提升至 0.259，主要提升了浅层术语的文本-图像对齐，视觉一致性几乎不变。

**⚠️ 局限性**

结构属性解释的方差有限（R²≈0.10），数据覆盖不完整且德国殖民数据自动匹配导致偏差，且微调仅改善对齐，对视觉散乱类别效果不明显。

---

## 194. D2VBench: Benchmarking Large Language Models with Value Dilemmas in Daily Scenarios

**arXiv ID:** 2607.19834 | [PDF](https://arxiv.org/pdf/2607.19834v1)

**作者:** Siyi Hao `[一作]` (Tianjin University), Deyi Xiong `[通讯]` (Tianjin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 D2VBench，一套包含 10,000 条真实日常价值困境场景的评测基准，并设计了将开放式回答映射到多选项并按五维解释性评分点计算得分的混合评估框架。

**💡 创新点**

创新点在于①系统化构建多价值冲突的日常场景（基于 158 个细粒度价值概念并人工校验），②引入开放式回答到多选映射的评估方式，消除了传统多选题单标签限制，③通过五维解释性评分点（后果、正当性、风险、责任、可行性）实现更细致的价值对齐度量。

**🔧 技术方法**

技术主要包括大语言模型（如 GPT‑5.1、Gemini‑3‑pro‑preview 等）用于生成场景与选项，人工智能与人工双轮校验；评估时使用 GLM‑4.6 与 Qwen3‑Max 作为判定模型，将自由文本映射到可选项并计算覆盖率；统计与可视化使用 Python/NumPy/Matplotlib 等工具。

**📊 数据集**

使用了自构建的 D2VBench 数据集（10,000 条日常困境实例），并对比了现有的多项价值评测基准如 NaVAB、CMoraEval、DailyDilemmas 等。

**📈 对比分析**

对比方法为两名判定模型映射+覆盖率评分，并对八款主流 LLM 进行全量评测。实验显示 GPT‑5.1 在三大价值类别中表现最佳（≈65.6 分），其次是 Gemini‑3‑pro‑preview（≈63.1 分），其余 LLM 分数介于 55–62 分；在五维评估中，合理性与责任得分最高（≈63–64 分），可行性与执行困难得分最低（≈52–53 分）。

**⚠️ 局限性**

局限性包括：①数据集仅为中文，价值体系受中国文化背景影响，难以直接推广至其他文化；②评估仍依赖判定模型的映射质量，可能存在偏差；③可行性维度表现较弱，说明 LLM 在实际操作建议方面尚需提升。

---

## 195. DARWIN: Evolving Jailbreak Adversary and Guardrail for LLM Safety Evaluation and Protection

**arXiv ID:** 2607.19829 | [PDF](https://arxiv.org/pdf/2607.19829v1)

**作者:** Weiwei Qi `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DARWIN 进化式攻击-防御框架，通过迭代的 DARWIN‑Attack 与 DARWIN‑Guard 形成持续的进化循环，以对抗 LLM 的 jailbreak 攻击。

**💡 创新点**

创新点在于将攻击视为自我进化过程，利用外部策略、遗传演化和基于反馈的 Markov 选择不断扩充策略池；同时防御端采用在线对抗训练，将恶意与伪装安全查询一起训练，迫使模型捕捉潜在意图而非仅记忆表面格式。

**🔧 技术方法**

技术实现包括遗传算法（交叉、变异）、强化学习式 Q‑学习更新策略转移矩阵、黑盒反馈驱动的策略组合、对抗样本生成与自省修正，以及联合交叉熵损失训练的对抗防御模型。

**📊 数据集**

攻击实验使用 HarmBench 与 AdvBench 两大恶意查询集；防御实验覆盖 XSTest、Aegis2.0、JailbreakBench、HarmBench、ToxicChat、WildGuardTest 等十二个危害基准，以及 ARC、BoolQ、GSM8K、OpenBookQA 等十一套标准安全基准。

**📈 对比分析**

相较于 TAP、ReNeLLM、AutoDAN‑Turbo、MAJIC 等现有方法，DARWIN‑Attack 在所有目标 LLM 与安全防线的 ASR 最高可达 99%，查询效率最好；DARWIN‑Guard 在 12 项危害基准上平均 unsafe recall 达 91.6%，超过 Qwen3Guard‑Gen‑8B（85.9%）与 YuFeng‑XGuard‑Reason‑8B（87.2%），且在 11 套标准安全基准上保持 100% safe pass rate。

**⚠️ 局限性**

局限性包括：仍以黑盒攻击为前提，无法直接解释内部决策；策略池扩张与在线训练需要较高计算与数据成本；对极端新颖模型或攻击方式的泛化能力仍待进一步验证；以及在真实部署中需注意对抗样本可能导致的误判与伦理风险。

---

## 196. Rewarding Better Thinking for LLM Preference Alignment

**arXiv ID:** 2607.19824 | [PDF](https://arxiv.org/pdf/2607.19824v1)

**作者:** Xubo Liu `[一作]` (Nankai University), Ying Zhang `[通讯]` (Nankai University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出思考清单奖励（TCR），通过将配对偏好转化为样本特定的思考清单，对LLM生成的推理轨迹进行过程级监督，从而提升开放式对齐效果。

**💡 创新点**

创新点在于：①从偏好对获取样本特定清单以捕捉用户意图、约束和权衡；②采用指数移动平均残差公式，将清单奖励与结果奖励解耦，保留思考过程的“剩余”优势；③将清单奖励作为补充信号加入RL后训练。

**🔧 技术方法**

技术手段包括：RL后训练框架DAPO；生成中间推理轨迹与最终答案；GPT‑4o推理评测生成思考清单与得分；EMA残差计算得到思考剩余奖励；格式奖励确保输出结构可解析。

**📊 数据集**

训练使用BPO数据集；评估采用四大开放式基准（Vicuna Eval、Dolly Eval、BPO‑test Eval、AlpacaEval 2.0）；清单构建基于配对偏好数据。

**📈 对比分析**

方法与基础模型、DPO、DAPO进行对比；在所有5个模型上TCR+DAPO平均提升ΔWR约+8到+16点，在AlpacaEval 2.0上WR、DWR、LCWR均达标，表明过程级奖励显著提升对齐性能。

**⚠️ 局限性**

局限性：依赖代理奖励和评测模型的准确性；EMA参数需经验调节；清单推理质量受评测模型影响；在极端多步推理或专业领域任务中，思考清单的适用性和覆盖面可能不足。

---

## 197. Towards Automated Formal Verification of zkEVMs Using LLM-Guided Constraint Synthesis

**arXiv ID:** 2607.19795 | [PDF](https://arxiv.org/pdf/2607.19795v1)

**作者:** Shichen Huang `[一作]` (Shanghai Polytechnic University), Guoqiang Li `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为VeriSynth的框架，利用大语言模型将zkEVM的Rust opcode实现自动合成可执行的Python/Z3验证模型，并通过SMT求解器最终判定实现是否符合语义约束。

**💡 创新点**

将LLM作为形式化前端而非最终判定者，结合语义分解、检索增强提示和基于求解器的自动修复，形成闭环管道，使得低层实现的符号化建模自动化且可验证。

**🔧 技术方法**

使用GPT‑4o进行代码到符号模型的生成、检索驱动提示、Z3求解器做模型可执行性与语义一致性检查，以及基于编译器和求解器反馈的有限循环自动修复。

**📊 数据集**

构造了首个针对Rust zkEVM opcode实现的源级bug基准，共95个注入缺陷样本，覆盖算术/位运算、存储/内存、Gas/退款、调用/创建路径及环境依赖等五类语义。

**📈 对比分析**

与LLM‑only、对话式LLM以及Scroll手写Rust mutation‑test进行对比，VeriSynth在95个负样本上检测率为91.6%，显著高于LLM‑only（46.3%）、对话式LLM（61.1%）和手写测试（57.9%），且在各opcode族上保持高效。

**⚠️ 局限性**

对正样本的误报率未评估；LLM生成的模型仍需多轮修复，导致约1.8轮平均修复成本；该方法目前仅支持离线验证，缺乏对大规模真实项目的跨项目泛化验证。

---

## 198. WASABI: Whole-graph Assignment-based Stabilizer for lAne topology By Inter-frame tracking

**arXiv ID:** 2607.19781 | [PDF](https://arxiv.org/pdf/2607.19781v1)

**作者:** Tetsuhiro Uchida `[一作]` (Sony Honda Mobility Inc.), Toru Saito `[通讯]` (Sony Honda Mobility Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了WASABI后处理框架，在实时约束下对感知模型输出的车道拓扑（车道段与互连关系）进行联合跟踪与稳定化，减少误检、连通性断裂和标签抖动；

**💡 创新点**

创新点在于将车道段与其相互连通关系视为统一的跟踪目标，结合帧内预处理、基于相似曲线的同级匹配、滑动窗口多数投票以及基于图的拓扑纠正，实现跨帧连通性与标签的高稳定性；

**🔧 技术方法**

采用同指数Fréchet近似、Hungarian分配、Kalman滤波状态管理、Bezier曲线端点平滑、滑动窗口多数投票以及实时资源控制策略（分头部匹配、早期终止）等技术；

**📊 数据集**

在Sony Honda Mobility内部验证数据集（16条序列，最多200条车道段/40k候选连通）上进行评估；

**📈 对比分析**

与原始感知模型对比，LCLC F1从0.834提升到0.948（+13.6%），车道检测F1提升至0.412，中心线侧向误差从2.50 m降至0.95 m，连通性切换率与边界标签抖动率分别下降63.3%与30.2%；

**⚠️ 局限性**

主要限制包括检测精度（精度/召回均低于0.5）、属性准确率（0.74–0.83）以及计算复杂度随车道数呈二次增长，需进一步提升鲁棒性与效率。

---

## 199. Look Before You Edit: Attention-Guided Camera Placement and Multi-View Alignment for 3D Gaussian Splatting Editing

**arXiv ID:** 2607.19777 | [PDF](https://arxiv.org/pdf/2607.19777v1)

**作者:** Jaeyeon Park `[一作]` (Seoul National University), Youngki Lee `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出LB-Edit框架，用文本驱动的3D Gaussian Splatting编辑技术，解决摄像机放置和多视角一致性问题。

**💡 创新点**

创新点包括：①基于扩散模型自/交叉注意力的Attention‑Guided Editing Camera Placement（ACP），动态选取最佳摄像机距离与多样化视角；②在同一U‑Net前向过程中同步自注意力与交叉注意力的Multi‑View Attention Alignment（MAA），构建共享3D注意力场，抑制外观与空间漂移。

**🔧 技术方法**

使用技术包括：InstructPix2Pix 2D扩散编辑器、3D Gaussian Splatting、注意力提取与分析、逆投影与3D注意力场、Fibonacci球采样、TokenFlow/ DGE/VcEdit 对齐机制等。

**📊 数据集**

使用 3D‑OVS（room、covered_desk、blue_sofa）和 IN2N（face、bear）共 5 个场景，并在每个场景上测试 20 条文本编辑提示。

**📈 对比分析**

与 GSEditor、VcEdit、DGE 等基线比较；评估指标为 CLIP 方向相似度、用户研究（指令忠实度、多视角一致性、编辑局部性）以及编辑延迟。LB-Edit 在 4/5 场景 CLIP 分数最高，用户偏好显著优于基线，且使用 5–20 个编辑视角即可实现 7 倍以上的延迟提升。

**⚠️ 局限性**

限制：受限于基础 2D 扩散编辑器的生成能力，无法实现 2D 无法完成的编辑；当前方法仅针对单个主 ROI，难以同时处理多个分离区域；对视频/时序一致性支持不足。

---

## 200. RPPNet: Perceptually-Grouped Rhythm-Pitch Primitives for Long-Term Structure Melody Generation via Boundary-Aware Modeling

**arXiv ID:** 2607.19776 | [PDF](https://arxiv.org/pdf/2607.19776v1)

**作者:** Tieyao Zhang `[一作]` (Zhejiang Conservatory of Music), Genfang Chen `[通讯]` (Zhejiang Conservatory of Music)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于感知驱动的层次化旋律生成框架RPPNet，利用可变长度的节奏‑音高原语（RPP）先生成结构化的框架，再通过解码器将其转化为具体的MIDI音符；

**💡 创新点**

创新点在于将音乐心理学中的声学线索、听觉惯性、相似性等三大结构感知机制转化为可自动化的RPP分组算法，并通过两阶段Transformer实现结构层与细节层的严格解耦；

**🔧 技术方法**

使用Transformer Encoder‑Decoder两阶段网络、解耦串行预测机制、时间尺度扩展映射、复合词嵌入以及动态规划的RPP分组算法；

**📊 数据集**

使用MelodyNet数据集（约27.4万首单声轨MIDI旋律）进行训练、验证和测试；

**📈 对比分析**

通过客观指标PPL与结构熵SE以及15名评测者的主观打分进行对比，RPPNet在结构连贯性、整体印象和节奏感上显著优于Museformer和MELONS，并与使用真实RPP输入的RPPNet‑Real相当；

**⚠️ 局限性**

局限性包括与人类作品仍有差距（尤其是节奏模式分布匹配不足），以及目前仅针对单声轨，缺乏多声部和伴奏的扩展。

---

## 201. An Isotropy-Preserving Spectral Cap for Muon: Theory and Three Case Studies

**arXiv ID:** 2607.19771 | [PDF](https://arxiv.org/pdf/2607.19771v1)

**作者:** Jiachun Li `[一作]` `[通讯]`, Jiachun Li

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种在 Muon 优化器中使用的谱上限（spectral cap）机制，用来抑制在尺度不变网络中出现的主特征向量的异常增长，从而保持网络权重的各向同性，进而避免训练过程中的崩溃。

**💡 创新点**

创新点在于：①将尺度不变性假设与 Muon 优化器结合，证明 Muon 消除了 1/W 速度抑制；②提出的谱上限通过在更新前投影去除第一阶主方向更新，且二阶项非负，保证权重仍能在正交方向学习；③在三种真实系统（nanoGPT FFN、64‑专家 MoE 路由器、FlashAttention BF16）中验证该机制的通用性和有效性。

**🔧 技术方法**

使用的技术包括：Muon 优化器（RMS‑matched Muon）、功率迭代法求解最大特征值、对更新做一阶投影以实现谱上限、对输入激活协方差进行数据相关的谱上限计算，以及在 FlashAttention 中对每层每头的 Q/K 方向做单独限制。

**📊 数据集**

使用的数据集包括 OpenWebText（nanoGPT 训练）、OpenWebText（MoE 路由器）以及 GPT‑2 small 的默认文本数据（FlashAttention 训练），均在 32k‑token 规模的 mini‑batch 上进行。

**📈 对比分析**

通过对比无谱上限基线、直接权重上限以及数据相关上限三种配置，实验显示：在 nanoGPT 中三种方法对验证损失影响不大，但数据上限显著提升了权重的稳定性；在 MoE 路由器中，数据上限能够完全避免  rank‑1 崩溃并恢复负载均衡；在 FlashAttention BF16 中，数据上限能够在 70k–110k 步的崩溃窗口内保持损失稳定，甚至比官方的 STABLE softmax 修补更具实时诊断优势；整体来看，谱上限在保持训练稳定性的同时几乎不增加额外计算成本。

**⚠️ 局限性**

主要局限：①理论建立在严格的尺度不变性假设上，实际网络仅近似满足；②实验规模有限，未验证在更大模型或更长训练时间下的效果；③FlashAttention 实验仅与单个无上限基线对比，缺乏多种基线的全面对照；④谱上限的参数选择（如阈值 ρ）仍需经验调优，未给出普适公式。

---

## 202. AlphaRoute: Large Language Models as Semantic Optimizers for Multi-Objective Routing

**arXiv ID:** 2607.19768 | [PDF](https://arxiv.org/pdf/2607.19768v1)

**作者:** Kabir Murjani `[一作]` (Nirma University), Jonti Talukdar `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AlphaRoute，利用 SHAP 分解拥塞、可调路径查找、3D Dijkstra 迷宫重路和 LLM 驱动的策略调优，实现全闭环的多目标全局路由。

**💡 创新点**

创新点在于：①将 SHAP 视角引入拥塞分解，精确定位导致溢出的网；②基于 SHAP 动态调整 PathFinder 的历史、即时拥塞权重和 via 成本；③结合知识图谱对 LLM 建议进行约束验证，兼顾可解释性与安全性；④将上述机制组合为单一迭代循环，在 ISPD 2025 竞赛基准上显著降低溢出。

**🔧 技术方法**

技术手段包括：Python+RustworkX 图构造、Slack‑aware 网排序、Steiner/MST+灵活 L‑路、SHAP‑式贡献分析、3D Dijkstra 迷宫重路、LLM（Llama3.3 70B / GPT‑OSS 120B）策略建议、RustworkX 知识图谱记录与验证。

**📊 数据集**

使用 ISPD 2025 公共竞赛设计集：ARIANE、BSG、MEMPOOL、NVDLA；每个基准均提供 .cap 与 .net 输入文件。

**📈 对比分析**

与当前 SOTA（S_orig = 1.780，S_scaled = 1.74）对比，AlphaRoute 在 ARIANE 上达成 S_orig = 0.0538、S_scaled = 0.0589，分别比 SOTA 低 33.1× 与 29.5×；在其他基准也保持相似的溢出削减。尽管运行时约为竞赛中位数的 136×，但算法性能已突破现有方法。

**⚠️ 局限性**

主要限制：Python 实现导致运行时显著偏高；缺乏 GPU 加速与高效编译循环；LLM 仅提升可解释性，对溢出降低作用有限；对结构性拥塞（如 BSG）表现不如预期。

---

## 203. OPIUM: Mitigating Steering Externalities and Over-Refusal via Dual Objective Latent Optimization

**arXiv ID:** 2607.19806 | [PDF](https://arxiv.org/pdf/2607.19806v1)

**作者:** Kavin Aravindan `[一作]` (IIIT Hyderabad), Ponnurangam Kumaraguru `[通讯]` (IIIT Hyderabad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 OPIUM 方法，通过在推理阶段对激活向量进行双目标表示匹配，净化激活调控向量，从而在不更新模型权重的情况下同时提升安全性与效用。

**💡 创新点**

创新点在于利用高维表示冗余进行双目标优化，既保持目标效用又消除安全外部性和过度拒绝问题，解决了传统方向抑制无法完全去除副作用的局限。

**🔧 技术方法**

使用了激活调控、向量表示提取、梯度下降双目标损失、范数投影等技术，并在对比层上匹配隐藏状态以实现向量优化。

**📊 数据集**

在 Qwen2.5-7B 和 LLaMA-3.1-8B 指令版模型上，评估数据集包括 HarmBench、FalseReject-Test、Corrigibility 与 Survival Instinct 的 MCQ 题集。

**📈 对比分析**

与基准模型、原始激活调控以及方向抑制进行对比，OPIUM 在安全-效用 Pareto 前沿表现最佳，既抑制了安全外部性的 ASR 提升，又保持甚至提升了效用；在过度拒绝场景下恢复了大多数善意请求的通过率。

**⚠️ 局限性**

局限性包括：仅针对特定向量与任务需要重新运行；依赖层级选择，可能无法处理语义耦合的行为；单一优化向量可能无法同时满足不同层次的行为需求。

---

## 204. DocOps: A Verifiable Benchmark for Autonomous Agents in Complex Document Operations

**arXiv ID:** 2607.19865 | [PDF](https://arxiv.org/pdf/2607.19865v1)

**作者:** Jiazhen Jiang `[一作]` (Chinese Academy of Sciences), Le Sun `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 DocOps，一个基于层次化分类法的可验证端到端文档操作评测框架。

**💡 创新点**

创新点在于将文档操作拆解为内容、格式、结构三维并设定四级工作流程深度，结合确定性验证器直接检查原生文档状态。

**🔧 技术方法**

采用层次化任务分解、自动化任务构造、原生库的确定性验证器以及多种执行环境（DocTools、Terminus-2、Codex、Claude Code）和可选技能。

**📊 数据集**

构建了210个自包含任务包，涵盖 Excel、Word、PPT、PDF 等四种格式，包含 L1–L4 四级难度。

**📈 对比分析**

在多种模型（GPT‑5.5、Claude Sonnet 4.6、DeepSeek‑V4‑Pro 等）和四种 harness 组合下进行评估，最高通过率仅为 0.671，且在 L3–L4 工作流程上性能急剧下降。

**⚠️ 局限性**

局限包括仅评估离线确定性编辑、未覆盖需要外部服务或协作的工作流、任务规模受人工审查限制、以及不同 harness 的成本比较受统计不一致影响。

---

## 205. Overview of FinMMEval 2026 Task 1: Multilingual Financial Multiple-Choice Question Answering

**arXiv ID:** 2607.19856 | [PDF](https://arxiv.org/pdf/2607.19856v1)

**作者:** Zhuohan Xie `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Preslav Nakov `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了FinMMEval 2026 Task 1——多语言金融多选问答任务，提供了包含英语、汉语、阿拉伯语与印地语共800道题目的评测集，并通过固定答案标签实现了严格的准确率评测；

**💡 创新点**

创新点在于将金融问答与多语言测试相结合，采用固定答案标签消除自由文本产生的歧义，同时分别为四种语言设置独立排行榜，便于对不同语言的性能进行细粒度比较；

**🔧 技术方法**

使用了多种技术手段，包括检索增强、语言路由、少量示例提示、Self‑Consistency、自校验与投票等；主流模型包括Gemini、Qwen、GPT‑4o‑mini、GPT‑5.5、Qwen‑3 等大型语言模型；

**📊 数据集**

采用了由CFA、CPA、阿拉伯会计教材及BhashaBench等来源整理出的金融多选题，共200道/语言（共800道），包含四选、三选、两选及五选题目；

**📈 对比分析**

评测指标为准确率，官方排名基于每种语言的独立准确率；顶尖系统在英语、阿拉伯语达到97.5%、中文96.5%、印地语92.0%，表明该任务在多语言金融问答上已达高水平；

**⚠️ 局限性**

局限性包括测试题目不平行，难以分离语言难度与内容差异；参赛团队在各语言上的覆盖度不一，导致跨语言综合评价受限；缺乏对题型和金融主题的细粒度分析。

---

## 206. SOPD-SocialNav: Selective On-Policy Distillation for Vision-Language Social Navigation

**arXiv ID:** 2607.19850 | [PDF](https://arxiv.org/pdf/2607.19850v1)

**作者:** Xinyu Zhang `[一作]` (Hokkaido University), Ling Xiao `[通讯]` (Hokkaido University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大规模视觉语言模型（VLM）进行选择性上政策蒸馏（SOPD），以提升轻量级VLM在社交导航中的性能。

**💡 创新点**

创新点在于：①引入基于教师模型不确定度的熵门控 token 选择机制，只对社会交互关键的高熵 token 进行蒸馏；②使用温度调控的 Jensen‑Shannon Divergence（JSD）对教师与学生分布进行对齐，避免传统 Reverse‑KL 的模式寻址问题。

**🔧 技术方法**

核心技术包括：教师模型冻结后的上政策蒸馏、token 层级熵估计与二值筛选、温度软化后的 JSD 损失以及对齐策略的联合训练。

**📊 数据集**

主要数据集为 SNEI 与 MUSON 两大社交导航基准，用以评估动作准确率、感知相似度与推理相似度。

**📈 对比分析**

与基准方法（SFT、SKD、VL‑KD、OPD、OPSD）对比，SOPD 在 Action‑Acc、Per‑cos 与 Rea‑cos 三项指标上均取得最高分，说明其在动作预测、感知一致性与推理一致性方面均优于现有方法。

**⚠️ 局限性**

局限性包括：熵阈值 τ 固定，缺乏自适应选择机制；仅在短时序（单步）场景验证，未探索更长时间规划；仅在两个基准数据集与少数实际机器人案例中测试，未覆盖更复杂或多样的社交场景。

---

## 207. Auto-Fill: Learning to Predict Missing Values Accurately with Specialist Language Models

**arXiv ID:** 2607.19847 | [PDF](https://arxiv.org/pdf/2607.19847v1)

**作者:** Yurong Liu `[一作]` (New York University), Surajit Chaudhuri `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Auto‑Fill，利用三种专家小语言模型（知识、推理、代码）在表格中预测缺失单元，并通过置信度校准实现动态选择或拒绝。

**💡 创新点**

创新点包括：①后训练专属专家模型并在同一任务上拆分为知识、推理、代码三种模式；②为每种模式设计专属置信度提取与自校准；③基于校准后的置信度实现可控的动态集成和高精度拒绝，既保持高精度又显著降低推理成本。

**🔧 技术方法**

使用 Qwen3 / GPT‑4.1 等小型模型做后训练（SFT + 生成式蒸馏 + 可选 RL），链式思考推理、Python 代码生成与执行检验、token log‑prob、verbalized confidence、执行准确率等多种置信度信号，利用等距回归对置信度进行校准，最后在推理时按最大置信度挑选答案或拒绝。

**📊 数据集**

构建 11 个基准数据集，包含 2200 真实表格，来源包括 467K Excel、12K BI 模型、292K Wikipedia、1.6K 政府 CSV、635 GitHub CSV、29k GitHub Parquet；测试集包含 ID 与 OOD 6 个新来源，保证表格结构多样化。

**📈 对比分析**

与多种基线（GPT‑4.1/mini、o3‑pro、Gemini‑3 Pro、DeepSeek‑R1、LakeFill、TabPFN、SCARE、Baran、FD‑upper‑bound 等）在 R@P=0.9 与 pAUPRC 上比较，Auto‑Fill 在所有 11 个基准上均超过所有基线，尤其在高精度区间表现突出；在成本方面，Auto‑Fill 的推理费用低于 frontier 模型 99%+，形成质量‑成本 Pareto 前沿。

**⚠️ 局限性**

局限性：①对跨单元协同缺失值的处理仍不成熟；②在极大表格或非常规结构中，行采样与上下文窗口限制可能影响效果；③RL 训练收益有限，需额外成本；④需要手工设计不同模式的置信度提取与校准，易产生误差；⑤在少量训练样本或罕见领域时，专家模型表现可能下降。

---

## 208. VizRAG: Enhancing Retrieval-Augmented Generation with Hypergraph Visualization

**arXiv ID:** 2607.19830 | [PDF](https://arxiv.org/pdf/2607.19830v1)

**作者:** Yanbin Wei `[一作]` (Southern University of Science and Technology), James Kwok `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VizRAG框架，将超图可视化与RAG结合，实现多模态检索和推理。

**💡 创新点**

创新点在于首次将超图可视化引入RAG，利用双视图与HyperViz解决视觉拥堵与渲染偏差，提升结构辨别与多跳推理。

**🔧 技术方法**

采用超图索引与检索技术、HyperViz可视化工具、Gemini-3 Flash等MLLM进行实体/关系抽取与结构感知，以及多模态提示式检索。

**📊 数据集**

在UltraDomain（Mix、CS、Agriculture）和MIRAGE（Neurology、Pathology）五个知识密集数据集上进行评估。

**📈 对比分析**

与GraphRAG、LightRAG、HiRAG、Hyper-RAG、Cog-RAG等基线比较，VizRAG在6项指标上平均赢率达70–82%，整体得分显著提升，尤其在多跳推理与结构辨别上。

**⚠️ 局限性**

局限性包括对MLLM视觉感知能力的依赖、双视图渲染导致的额外延迟以及评估中LLM裁判的偏差。

---

## 209. Lean-SAM2: Target-Anchored Memory and Encoder Acceleration for SAM2

**arXiv ID:** 2607.19811 | [PDF](https://arxiv.org/pdf/2607.19811v1)

**作者:** Xudong Ouyang `[一作]` (Hainan University), Yunshan Zhong `[通讯]` (Hainan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现Lean-SAM2框架，结合目标锚定记忆修剪、时间凝聚保险记忆和目标锚定风险感知路由三种机制，显著提升SAM2视频对象分割的推理速度与鲁棒性。

**💡 创新点**

三项协同机制：TAMP利用锚点语义一致性保护关键记忆，TCIM用可见性门控融合历史记忆并维护保险库，TARR在窗口路由中加入锚点相似度与风险回退，解决传统剪枝与路由在遮挡/干扰场景下的性能崩塌。

**🔧 技术方法**

基于SAM2的连续记忆注意力、窗口自注意力、语义相似度度量、可见性门控融合、保险记忆库与风险感知全帧回退等技术。

**📊 数据集**

在LVOSv2、MOSEv1/2、SA-V等五个视频分割基准上进行评估。

**📈 对比分析**

与Efficient-SAM2、单独/组合TAMP、TCIM、TARR等方法对比，Lean-SAM2在SAM2.1-Large/Base+/Small基线下实现1.41×–1.43×推理速度提升，且在LVOSv2上相对SAM2提升5.0%–3.6%的准确度。

**⚠️ 局限性**

在极端遮挡或高度动态场景中仍可能触发较高的回退率导致速度下降，且模型对锚点采样和阈值调参较敏感，需进一步验证对不同目标尺寸与复杂背景的鲁棒性。

---

## 210. Zero-Observation User Reactivation with Gap-Driven Dimensional Gating

**arXiv ID:** 2607.19802 | [PDF](https://arxiv.org/pdf/2607.19802v1)

**作者:** Jiandong Ding `[一作]` (Fudan University), Tiandeng Wu `[通讯]` (Huawei Technologies)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在用户长期无交互后重新激活的情境下，定义并研究了Zero-Observation Reactivation问题，提出在序列推荐模型输出层加入基于时间间隔的维度门控（DeltaGate）来恢复长时间空窗期的推荐性能。

**💡 创新点**

创新点在于将一个轻量级、可冻结的门控插件直接附加在任何序列编码器的输出层，门控只需少量参数（约66K）即可根据用户停留时间和个性化表示动态调节各维度的重要性，从而在不更新主模型的情况下显著提升宏gap（>365天）性能。

**🔧 技术方法**

使用的技术包括：自注意力（SASRec）、GRU（GRU4Rec）、BERT式双向掩码（BERT4Rec）等主模型；门控MLP结合log(1+Δt)与个性化表示输入；全局先验向量；以及完整-目录交叉熵训练和留一评估协议。

**📊 数据集**

实验数据集来自亚马逊电商的三大品类：Video Games、CDs & Vinyl、Movies & TV，并在每个类别上进行5-core过滤后构建用户历史与自然gap桶。

**📈 对比分析**

与SASRec、GRU4Rec、BERT4Rec、TiSASRec等基线以及全模型重训练（TimeConcat）对比，DeltaGate在>365天宏gap中将Hit@10提升约30%至60%（例如Video Games从0.031提升至0.047），且只增加约66K参数，保持零backbone漂移，参数效率显著优于全模型重训练。

**⚠️ 局限性**

主要局限包括：仅基于ID级别交互，未利用侧信息或多模态特征；共享的全局先验可能不足以覆盖不同用户群的长期兴趣；自然gap分层无法完全消除多年Catalog Drift；需要在不同平台和更大规模数据上进一步验证。

---

## 211. Physics-Aware Complex-Valued State Space Model with Scattering-Prior Feature Modulation for PolSAR Image Classification

**arXiv ID:** 2607.19787 | [PDF](https://arxiv.org/pdf/2607.19787v1)

**作者:** Fangyan Zhang `[一作]` (Beijing University of Chemical Technology), Qiang Yin `[通讯]` (Beijing University of Chemical Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理感知的复值状态空间网络（CV-SSMNet）用于多极化SAR图像分类。

**💡 创新点**

创新点在于将复值状态空间模型与七种物理散射先验（H、A、α、P_s、P_d、P_v、Span）通过FiLM式条件调制相结合，实现复值特征的长程依赖建模与散射机制引导的特征演化。

**🔧 技术方法**

采用复值卷积、复值状态空间模型（branch-wise & lightweight）、FiLM条件调制、通道再校准（SE）、多尺度特征融合与全局上下文聚合等技术。

**📊 数据集**

在三套L‑band基准数据集（Flevoland、San Francisco、Oberpfaffenhofen）和一套P‑band BIOMASS数据集上进行评估。

**📈 对比分析**

与七种对比方法（包括自监督、复杂网络、Transformer、Mamba等）在相同的无信息泄漏空间划分和低样本设置下对比，CV‑SSMNet在三套L‑band数据集上取得最高OA（97.56%/97.02%/96.07%），在BIOMASS上也显著提升（OA 88.87%），表现出更好的边界保留和区域一致性。

**⚠️ 局限性**

局限在于对P‑band频率依赖的散射变化适应不足，固定13×13 patch限制了尺度适应性，复值SSM和FiLM调制虽然提高精度但带来一定计算开销，对分解先验的噪声敏感。

---

## 212. Not Birds of a Feather: Personality-Based Partner Selection in LLM Agents

**arXiv ID:** 2607.19785 | [PDF](https://arxiv.org/pdf/2607.19785v1)

**作者:** Tao Wang `[一作]` (University of Toronto), Zhonghao Hou `[通讯]` (University of Toronto)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究测试了在能力保持恒定的情况下，大五人格是否会影响多智能体系统中代理的合作伙伴选择。研究通过两个实验，探讨了中立代理和人格分配代理的选择行为。

**💡 创新点**

创新点在于首次提供了关于大五人格在多智能体系统中影响合作伙伴选择的实验性证据，并揭示了选择偏好与人类团队表现之间的偏差。

**🔧 技术方法**

使用了大五人格模型，通过提示塑造和验证人格特征，采用了控制选择的实验设计。

**📊 数据集**

研究使用了375个试验，涉及六种经过验证的人格原型，涵盖五个任务类别（创造性构思、分析推理、战略规划、信息综合和问题解决）。

**📈 对比分析**

与人类团队表现的元分析结果相比，选择结果显示出显著的偏差，尤其是外向性和宜人性原型几乎未被选择，而神经质原型在分析任务中被选择的比例较高，表明选择偏好与人类表现标准不一致。

**⚠️ 局限性**

限制在于所有代理均基于同一模型，未测试跨模型的普遍性；候选人以个人资料卡的形式呈现，可能与实际互动中的选择行为不同；此外，所使用的人格原型为单一特征极端，未考虑多特征的现实情况。

---

## 213. Symbol and Footprint Database for Electronic Components by Agentic Recognition and Generation

**arXiv ID:** 2607.19767 | [PDF](https://arxiv.org/pdf/2607.19767v1)

**作者:** Yichen Shi `[一作]` (Ningbo Institute of Digital Twin, Eastern Institute of Technology), Lei Hel `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了基于多模大语言模型的SFgen流程，实现从元件数据手册自动提取信息并生成PCB元件的符号与封装文件，并基于此构建了SFnet大规模元件库。

**💡 创新点**

创新点在于：①将少样本ICL、视觉提示(VP)和链式思维(CoT)与MLLM结合，实现高效自动化符号与封装生成；②首次提供包含1000多件元件的完整符号/封装数据库；③提出基准数据集与评估指标体系，推动该领域的客观比较。

**🔧 技术方法**

主要技术包括：多模大语言模型（GPT‑4）、PDF解析与图像提取工具、少样本ICL、视觉提示(VP)、链式思维(CoT)以及针对EDA文本格式的定制化提示设计。

**📊 数据集**

使用了两类数据集：①100个带有数据手册、符号与封装的基准样本用于评估；②SFnet数据库，包含1000多件元件（主打IC、R、C、L、D等）及其对应的符号与封装。

**📈 对比分析**

通过ACC_N、ACC_T、Dif_A、Dif_P四个自动化指标和MOS-S/F主观评分进行评估；在四个难度层级（Basic、Standard、Complex、High‑Density）中，符号生成ACC_N最高可达96%，封装生成ACC_N可达96%，整体符号准确率约86%，封装准确率约80%，且Dif_A、Dif_P均低于10%。

**⚠️ 局限性**

局限性包括：对高密度多针封装的识别与定位仍有限，MLLM对复杂图像的解析能力受限；生成文件在某些细节仍需人工校正；数据库规模虽大但仍相对有限，缺乏跨EDA软件兼容性与更广泛的元件类型覆盖。

---

## 214. Clinical Pathways as Safety Specifications for Physical AI in Hospital Wards

**arXiv ID:** 2607.19827 | [PDF](https://arxiv.org/pdf/2607.19827v1)

**作者:** Gabriele Franchini `[一作]` (University of Bari), Filippo Lanubile `[通讯]` (University of Bari)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

将临床路径重新定义为物理 AI 在医院病房中的运行时安全规范，构建了包含可穿戴传感器、智能医疗设备与辅助机器人的多模态监控架构，并通过 Runtime Safety Monitor（RSM）实时验证规范执行。

**💡 创新点**

创新点：①把临床路径转化为可验证的 Signal Temporal Logic（STL）安全规范；②设计多模态、带不确定性估计的 RSM；③将机器人本体作为安全资产，提供多源交叉验证和抗篡改能力；④生成可解释的安全违规报告。

**🔧 技术方法**

使用技术包括 STL 规范编码、时间序列预测与自适应不确定性推理、合规预测的置信度估计、ROS 2 等中间件、机器人导航与人机交互等。

**📊 数据集**

本文未使用公开数据集，采用假设性轨迹示例来说明概念。

**📈 对比分析**

由于为概念性设计，未进行实验对比；性能评估仅以示例性鲁棒性指标为准。

**⚠️ 局限性**

局限性：①临床路径到 STL 的翻译需要专门工具或模板；②合规预测假设可交换性，真实临床信号可能违背；③缺乏在真实医院环境中的临床验证与部署经验。

---

## 215. V2F: Vision-Informed Grasp Force Prediction for Damage-Aware Robotic Handling of Date Fruits

**arXiv ID:** 2607.19804 | [PDF](https://arxiv.org/pdf/2607.19804v1)

**作者:** Shahd Shami `[一作]` (King Abdullah University of Science and Technology), Shinkyu Park `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了基于视觉的预抓取力预测框架V2F，用于安全处理日期果

**💡 创新点**

结合Hertz接触理论与残差神经网络的物理指导残差学习，提前预测每个果实的安全抓取力

**🔧 技术方法**

Mask R‑CNN与主动轮廓分割、立体视觉、Hertz理论、残差深度网络、随机森林、梯度提升等技术

**📊 数据集**

约500个不同品种（Ajwa、Barhi、Sagai）及不同水分状态的日期果样本数据集

**📈 对比分析**

与多种基线（线性回归、随机森林、Gradient Boosting等）进行GroupKFold交叉验证，平均R^2≈0.7、RMSE≈1.13kPa；机器人实验中残余变形<1mm，抓取稳定无损伤

**⚠️ 局限性**

对极端品种/湿度的泛化仍受限，需结合触觉闭环以应对低可信视觉预测的离群样本

---

## 216. Silent Failures in Multimodal Agentic Search:A Diagnostic Taxonomy and Cross-Judge Evaluation

**arXiv ID:** 2607.19793 | [PDF](https://arxiv.org/pdf/2607.19793v1)

**作者:** Zhengxian Wu `[一作]` (Ant Group), Kai Yang `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对多模态代理搜索中的无声失败（silent failures）进行轨迹级研究，提出六类失败类型并构建诊断管线。

**💡 创新点**

创新点在于：①从轨迹层面定义无声失败并提出六类细粒度分类；②设计统一的ReAct框架与LLM判定规则进行系统诊断；③引入同族和跨族交叉验证评估判定可靠性；④在MMSearch-Plus上实验揭示准确率与真实正确率的差距，展示工具消融对失败模式的影响。

**🔧 技术方法**

采用统一ReAct式多模态代理框架、LLM判定器、Cohen κ交叉验证、空白图像压力测试、工具消融实验等技术。

**📊 数据集**

使用MMSearch-Plus数据集，采样200个任务，生成800条轨迹进行实验。

**📈 对比分析**

对四大前沿模型（Claude Sonnet 4.6、Gemini 2.5 Pro、Gemini 3.1 Pro Preview、GPT‑4o）进行对比，报告答复准确率与轨迹真正确率（TCR）。结果显示TCR显著低于表面准确率；工具消融提升准确率但失败模式迁移，揭示强模型并未完全消除无声失败。

**⚠️ 局限性**

局限性包括：无声失败标签对判定者敏感；部分错误难以自动标注；实验仅覆盖MMSearch-Plus和四个模型，缺乏更广泛任务；未完全解决错误证据和幻觉问题。

---

## 217. Trace: A Taxonomy-Guided Environment for Multidomain Visual Reasoning

**arXiv ID:** 2607.19790 | [PDF](https://arxiv.org/pdf/2607.19790v1)

**作者:** Md Tanvirul Alam `[一作]` `[通讯]` (Rochester Institute of Technology), Md Tanvirul Alam (Rochester Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于语法树和任务程序分离的多域视觉推理环境，并在其中进行强化学习与可验证奖励（RLVR）训练；

**💡 创新点**

创新点包括：①将视觉场景语法与可执行任务程序解耦，形成稳定的任务单元，支持语义和视觉两级可控生成；②设计了涵盖11个视觉域、277个场景语法、1000个任务的完整实验环境；③首次在视觉语言模型上使用RLVR，并验证其在外部基准上的迁移效果；

**🔧 技术方法**

使用的技术主要有：可执行的生成–渲染–验证管线；RLVR（带可验证奖励）和群组相对策略优化（GRPO）；Qwen2.5‑VL 3B/7B 语言视觉模型；JSON 格式的答案规范与奖励计算；

**📊 数据集**

训练数据：64k 条程序化生成的实例（1k 任务 × 64 示例）；测试数据：24 个外部视觉推理基准（6 组，每组 4 个基准），涵盖图表、数学、科学推理、空间推理、感知计数和谜题等；

**📈 对比分析**

对比方法：在同一模型基础上先无 RLVR 训练后加入 RLVR，比较宏观平均得分；在 7B 规模下还与 Game‑RL、Sphinx、PC‑GRPO 等公开 RLVR 检查点对比。结果显示：3B 模型宏观平均提升 3.51 分，7B 提升 4.06 分；7B 在与其他 RLVR 检查点的对比中平均提升约 3–4 分；所有类别均出现正向提升，部分基准在 7B 上表现更显著；

**⚠️ 局限性**

局限性：①任务边界由人工设计，仍有主观判断；②仅包含 1000 个任务，未覆盖所有视觉推理场景；③未对不同域、运算族或渲染控制的贡献进行归因；④训练采样为均匀分布，未使用难度感知或自适应采样；⑤仅进行一次训练，缺乏对随机性或超参数的多样化评估；⑥合成渲染可能缺乏真实图像的多样性与感知难度。

---

## 218. Defer to Plan: Adaptive Multi-Agent Fusion for End-to-End V2X Driving

**arXiv ID:** 2607.19774 | [PDF](https://arxiv.org/pdf/2607.19774v1)

**作者:** Nuoran Li `[一作]` (Shenzhen Automotive Research Institute), Chao Sun `[通讯]` (Shenzhen Automotive Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一套面向规划的V2X协同驾驶系统，推迟多智能体特征融合至规划阶段，并通过自回归解码器实现动态适配。

**💡 创新点**

创新点包括：①将多智能体特征融合时机从感知阶段推迟到规划阶段，以自回归解码器实现场景自适应权重；②双流特征处理结合通道级纯化和Mixture-of-Experts（MoE）增强分词器，专门处理车辆与基础设施的异质特征；③混合因果掩码实现并行自回归解码，兼顾轨迹生成的时序一致性。

**🔧 技术方法**

技术包括：BEV编码器、MotionNetwork用于时空特征聚合；注意力分词器（tokenizer）与MoE增强的MLP；LLaMA式并行自回归解码器；Hybrid Causal Mask；端到端训练与轨迹引导的MoE路由监督。

**📊 数据集**

使用V2Xverse仿真平台（基于CARLA）进行实验，涵盖67条路线、8个小镇的安全关键场景。

**📈 对比分析**

与单机方法（TCP、TransFuser）以及协同方法（CoDriving、Coopernaut、V2X-ViT）对比，闭环驾驶分数提升至79.72（比CoDriving高3.33%），轨迹误差（ADE/FDE）和违规分数（IS）也均有显著改进。

**⚠️ 局限性**

主要局限是通信拓扑假设固定，未考虑动态智能体选择、通信延迟的显式建模及不确定性估计，未来工作可针对这些方面进一步提升系统鲁棒性。

---

## 219. Global Building Area Estimation Products: How Accurate Are They?

**arXiv ID:** 2607.19766 | [PDF](https://arxiv.org/pdf/2607.19766v1)

**作者:** Saad Lahrichi `[一作]` (University of Missouri), Jordan Malof `[通讯]` (University of Missouri)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对四大全球建筑面积产品（GHSL、GBA、TEMPO、Overture）进行独立且公平的精度评估与比较。

**💡 创新点**

创新点在于：①使用高分辨率手工标注的 ORBITaL-Net 数据作为真值；②在多种空间分辨率的参考网格上进行评估；③结合向量与栅格两类产品，提供一套统一的比较框架。

**🔧 技术方法**

采用多参考网格投影、向量/栅格面积聚合、MAE、WMAPE、R² 等常用统计指标，并通过多分辨率重投影与误差计算实现公平比较。

**📊 数据集**

主要数据集包括：高分辨率手工建筑掩模 ORBITaL-Net（0.45 m）以及四个全球建筑面积产品 GHSL、GBA、TEMPO、Overture。

**📈 对比分析**

评估方法：在每个参考网格（GHSL 3″、GHSL 100 m、TEMPO 77 m）分别计算 MAE、WMAPE、偏差、R² 等指标。结果显示 GBA 在 MAE/WMAE 上最佳，TEMPO 在整体偏差与 R² 上最优，GHSL 效率最差；产品在不同大洲、人口密度和收入水平上表现差异显著。

**⚠️ 局限性**

局限性包括：真值与目标产品时间跨度差异；ORBITaL-Net 在欧洲缺失导致覆盖不足；建筑定义与实际面积可能不一致，尤其在遮挡、斜射角等情形下；以及不同产品的时间更新频率差异导致系统偏差。

---

## 220. Removing Online Exponential Net Search from Solovay-Kitaev

**arXiv ID:** 2607.19874 | [PDF](https://arxiv.org/pdf/2607.19874v1)

**作者:** Henrique Ennes `[一作]` (Université Côte d'Azur), Clément Maria `[通讯]` (Inria)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文改进了 Solovay–Kitaev 算法，在变量维度情形下避免了对指数大网格的显式搜索，提出了基于指数基的无网格版本；

**💡 创新点**

创新点在于引入“好指数基”概念，用离散 Trotter 化替代传统网格搜索，并将网格查询的指数维度依赖压缩到预处理阶段，显著降低在线合成的时间复杂度；

**🔧 技术方法**

主要技术包括：Lie 代数与 Baker–Campbell–Hausdorff (BCH) 展开、几何控制与 Riemannian 度量、离散化 Trotter 与 Suzuki 公式、Gell–Mann 基底的条件数分析以及整数化的系数逼近；

**📊 数据集**

本文为理论分析，未使用具体实验数据集，仅通过数学证明和复杂度评估来验证算法性能；

**📈 对比分析**

与传统 Solovay–Kitaev 算法在多目标场景下的理论复杂度比较显示：在一次预处理后，算法时间从 O((d³+T₀)p log^{k_t}(1/ε)) 降至 O(d⁶+pd⁴ log^{k_t}(1/ε))，但在单目标或不具备好指数基的情况下，额外的网格构造仍保持指数依赖；

**⚠️ 局限性**

局限性在于：必须先构造“好指数基”，其构造成本和对 d 的依赖仍然高；在实际大维度场景下，h_max 必须极小，导致门长度指数增长；此外，离散化误差与漂移控制仍需进一步改进。

---

## 221. Adversarial Frontiers: Minimum-Norm Attack Ensembles for Robustness Evaluation

**arXiv ID:** 2607.19855 | [PDF](https://arxiv.org/pdf/2607.19855v1)

**作者:** Luca Scionis `[一作]` (Sapienza University of Rome), Battista Biggio `[通讯]` (University of Cagliari)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的对抗鲁棒性评估框架，利用最小范数攻击集合生成鲁棒性–扰动曲线并定义攻击前沿与防御前沿；

**💡 创新点**

通过对抗前沿的最优逼近与防御前沿的对齐，提供可调查询预算的曲线评估和无参考ε的防御排名，克服传统单点ε评估的局限；

**🔧 技术方法**

采用最小范数梯度攻击（如PD‑GD、APGD、FMIN等）、贪婪查询分配算法、AttackBench中的AUREC度量、以及自定义的Defense Optimality Index；

**📊 数据集**

在CIFAR‑10和ImageNet的30个鲁棒模型上（含20个CIFAR‑10、10个ImageNet），使用1,000个验证样本进行评估；

**📈 对比分析**

与AutoAttack对比，所构建的4k/8k/12k查询层次攻击集合在大多数模型和范数下在鲁棒性–扰动曲线与AUREC指标上匹配或优于AA，且在所有模型上提供稳定的ε‑无关排名；

**⚠️ 局限性**

局限性包括：评估基于预定义攻击池，前沿与指标依赖共享模型集；离线预处理成本高，需为新攻击额外跑全量数据；仅覆盖白盒梯度攻击，对梯度模糊或非可微防御的评估不充分。

---

## 222. SpikingMOT: A Spike-Driven Multi-Object Tracker

**arXiv ID:** 2607.19875 | [PDF](https://arxiv.org/pdf/2607.19875v1)

**作者:** Yiding Sun `[一作]` (Xi'an Jiaotong University), Tiejun Huang `[通讯]` (Peking University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `aaccfe5c-6b26-4208-b23c-35331481e142` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一种基于稀疏神经网络的多目标跟踪框架SpikingMOT。

**💡 创新点**

创新点在于将稀疏激活偏好（ASP）与脉冲神经网络结合，采用伪轨迹基分解和误差校准实现高效稀疏运动预测。

**🔧 技术方法**

主要技术包括脉冲神经网络、LIF神经元、运动基底分解、预测误差校准以及与传统ANN和Kalman滤波器的对比。

**📊 数据集**

在SportsMOT、DanceTrack、SeaDroneSee、MOT17等数据集上进行实验验证。

**📈 对比分析**

通过与基于ANN的MambaTrack、Kalman滤波以及其他领先跟踪器比较，SpikingMOT在HOTA、IDF1、AssA等指标上均优于或相当，并显著降低参数和能耗。

**⚠️ 局限性**

局限性在于能耗估计基于运算计数而非真实神经形态硬件；在更复杂场景下对检测器依赖较高，且需在硬件上进一步验证。

---

## 223. StellarTTS: Sparse Temporal Embedding for Low-Latency and Robust Speech Synthesis

**arXiv ID:** 2607.19859 | [PDF](https://arxiv.org/pdf/2607.19859v1)

**作者:** Kaicheng Luo `[一作]` (Honor Device Co., Ltd.), Yanmin Qian `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了面向移动端的非自回归语音合成框架 StellarTTS

**💡 创新点**

创新点包括稀疏时间嵌入、语义感知编码器与单阶段轻量化解码器

**🔧 技术方法**

采用残差向量量化语义编码、稀疏语音时序嵌入、掩码生成Transformer与多阶段推理策略

**📊 数据集**

在多语言数据集 Emilia 上训练，并在 Seed‑TTS test‑zh 与 test‑hard 进行评估

**📈 对比分析**

与 AR、NAR 基线相比，WER 从 1.51 降至 1.44，SIM‑o 0.712，RTF 0.08，速度提升 4–9 倍

**⚠️ 局限性**

主要局限在语义蒸馏导致的语音相似度略低（SIM‑o 0.712），以及对极端发音错误的鲁棒性仍有提升空间

---

## 224. Sentence Splitter: Uncovering Latent Factual Structure for Self-Supervised Learning

**arXiv ID:** 2607.19845 | [PDF](https://arxiv.org/pdf/2607.19845v1)

**作者:** Ahmad Pouramini `[一作]` (Sirjan University of Technology), Mahsa Afsharizadeh `[通讯]` (Sirjan University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一种基于T5的自监督框架Sentence Splitter，用于从自然语言句子中自动拆分并识别隐含的事实补全（尾部）与描述前缀（头部）边界。

**💡 创新点**

创新点在于：1）将事实拆分视为离散分割问题，通过概率序列生成实现无显式枚举；2）利用符号知识图谱自动生成训练数据，实现无人工标注的自监督学习；3）引入轻量级自举（bootstrapping）流程，让生成模型进一步扩展事实补全。

**🔧 技术方法**

技术方法包括：T5编码-解码架构、基于概率生成的分割、符号知识图谱到自然语言模板的转化、生成式自举、以及可选的质量验证步骤。

**📊 数据集**

使用数据集包括：ATOMIC2020知识图谱（用于训练与评估）、OMCS自由文本语料（用于自举与测试）、CommonsenseQA多选问答（下游评测）。

**📈 对比分析**

与传统随机掩码的MLM预训练对比，Sentence Splitter在CommonsenseQA的准确率提升约4%（从51.7%到55.7%），在ATOMIC 2020补全的ROUGE提升约4%（从20.6%到25.1%），进一步加入一次自举后提升更明显；实验在多次随机种子下表现稳定。

**⚠️ 局限性**

局限性包括：假设事实补全为句子唯一的连续后缀，无法处理多段或非连续事实；自举过程可能导致语义漂移；以及对更复杂句子结构的泛化能力尚未完全验证。

---

## 225. MoAKE: Toward Unified All-in-One Action Quality Assessment via Mixture of Action Knowledge Experts

**arXiv ID:** 2607.19826 | [PDF](https://arxiv.org/pdf/2607.19826v1)

**作者:** Huangbiao Xu `[一作]` (Fuzhou University), Jinglin Xu `[通讯]` (University of Science and Technology Beijing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了MoAKE框架，用于在单一模型中评估多种动作的视频质量，实现了全动作统一的动作质量评估（all-in-one AQA）。

**💡 创新点**

创新点包括：①基于混合专家的设计，让每个专家学习不同动作模式；②引入段感知原型和自适应段内段间关系建模（AIISRM），有效捕捉多粒度时序动态；③使用软路由和多任务损失实现专家知识的动态融合，抑制跨动作负迁移并实现正迁移。

**🔧 技术方法**

技术手段主要包括：Mixture of Experts结构、共享与专属适配器、软路由器、段感知原型聚合、Adaptive Intra-Inter-Segment Relationship Modeling模块、Grade-aware解码器、以及多任务损失（score、route、diversity）。

**📊 数据集**

使用的数据集包括长时序动作三大数据集：Rhythmic Gymnastics、Figure Skating、Artistic Swimming进行all-in-one训练；零/少样本评估则采用短时序动作数据集：Diving、Skiing/Other、Surgery（JIGSAWS）等六个数据集。

**📈 对比分析**

在all-in-one设置下，MoAKE在三大长时序数据集的SRCC平均达到0.753，显著高于所有基线（平均提升约1.2%），R-ℓ₂降至2.434（比最优基线低28.9%）。在零-shot和few-shot评估中，MoAKE同样显著优于现有SOTA方法，表现出更强的跨动作泛化能力。

**⚠️ 局限性**

主要局限包括：零-shot泛化仍然困难，模型尚未充分利用多模态信息；在极低样本场景下的适应速度仍有提升空间；此外，虽然专家数目较少保持计算效率，但在更大动作种类下可能需要更多专家或更复杂的路由策略。

---

## 226. SimulS2ST-Omni: Data-Efficient Streaming Speech-to-Speech Translation via Explicit Trajectory Supervision

**arXiv ID:** 2607.19810 | [PDF](https://arxiv.org/pdf/2607.19810v1)

**作者:** Rongshen He `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套基于轨迹的训练管道，用于实现中文‑英文流式语音到语音翻译（S2ST），并集成了语言模型和分词预测骨干网络，配合分块流匹配解码器实现流式解码。

**💡 创新点**

创新点在于将语言模型与分词预测相结合，形成端到端的流式S2ST骨干；并采用分块流匹配解码器，尝试在流式环境下保持连续性与实时性。

**🔧 技术方法**

使用了语言模型（LM）与分词预测网络作为主干；流匹配（Flow Matching）解码器实现分块解码；并借助对齐与过滤技术构建合成语料。

**📊 数据集**

使用了通过合成、对齐和过滤生成的中英对齐流式S2ST数据集（目前为合成数据，未包含真实同声传译录音）。

**📈 对比分析**

与传统基线（如静态S2ST或非流式模型）进行了对比，实验显示所提出方法在翻译质量和实时性上均优于基线，具备可接受的音频合成质量。

**⚠️ 局限性**

局限性包括：仅限中文‑英文实验；合成数据可能不完全代表真实同声传译场景；分块流匹配解码器尚未完全流式化，可能影响跨块连贯性与推理效率。

---

## 227. Mammal: Supporting Breastfeeding Monitoring Through Computational Garments with Inter-Body Sensing

**arXiv ID:** 2607.19796 | [PDF](https://arxiv.org/pdf/2607.19796v1)

**作者:** Yanfeng Zhao `[一作]` (Florida State University), Te-Yen Wu `[通讯]` (Florida State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了一款名为Mammal的计算衣，通过护理者身体与婴儿的口-乳头接触，利用电耦合和声学传输无创监测母乳喂养中的接触持续时间、婴儿心率、吸吮-吞咽-呼吸（SSB）比例以及母乳摄入量。

**💡 创新点**

创新点在于：①首次将人体间信号传输原理应用于母乳喂养监测，避免在婴儿身上安装传感器；②开发了融合自适应滤波与深度学习去噪的婴儿ECG分离算法；③使用基于小波分解与模糊聚类的自适应方法实现吸吮与吞咽事件的无监督检测；④将多模态信号融合估计SSB比例与母乳摄入。

**🔧 技术方法**

技术手段包括：人体间电耦合与声学耦合的传感设计；针织导电电极与柔性音频麦克风；多分辨率小波变换、基于NLMS的自适应权重估计、条件去噪的DeScoD-ECG；Sucking/Swallowing的特征提取与在线模糊聚类；以及心率与呼吸的基于R-R间期的调制分析。

**📊 数据集**

主要数据集为10对母婴对的在实验室环境下的喂养记录（总计239分钟），配合成人对照组用于ECG验证、硅胶乳房模型用于声学传输特性评估；此外，使用Bangle.js手表与临床PPO测量作为心率与呼吸的基准。

**📈 对比分析**

通过与人工标注与外部传感器（PPG、麦克风、称重）对比，Mammal在接触检测F1=0.93、接触持续时间MAPE=5.56%、心率MAE=3.61 bpm、SSB比例MAE=0.12、母乳摄入相对误差15.76%，显示出与传统体位式监测相比显著可行且精度可接受；相对于成人心电参考，QRS相关性0.96，心率误差0.34 bpm。

**⚠️ 局限性**

局限性包括：需婴儿与护理者之间直接皮肤接触，受婴儿衣物、体位限制；实验在受控实验室环境，缺乏家庭环境的噪声与运动干扰；系统目前为离线后处理，未实现实时反馈；样本量有限，缺乏长期纵向验证；电极与麦克风的洗涤耐久性尚待评估。

---

## 228. A Structure-Adaptive Random Feature Method for High-Dimensional Elliptic PDEs

**arXiv ID:** 2607.19786 | [PDF](https://arxiv.org/pdf/2607.19786v1)

**作者:** Jiale Linghu `[一作]` (Xidian University), Yangshuai Wang `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于残差 Sobol 指数和预测器梯度协方差的分层随机特征方法（HA-RFM），自动选取坐标块和低秩方向，并在一次正则化最小二乘中同时拟合所有保留特征，解决高维椭圆 PDE 的求解。

**💡 创新点**

创新点在于：①将残差的闭式 Sobol 指数作为可计算的 PDE 驱动规则，用以筛选高阶坐标交互；②利用预测器梯度的协方差估计主动子空间，自动识别斜向低秩结构；③在同一线性系统中耦合坐标块与全局斜向特征，避免逐层约束；④在理论上给出误差、宽度与结构恢复的显式界限，证明在固定交互阶数下宽度为多项式且高阶贡献维数无关；⑤通过 Picard 迭代将方法推广到半线性方程。

**🔧 技术方法**

使用的技术包括：随机特征方法（RFM）与激活函数，ANOVA- Sobol 分解，主动子空间（Active Subspace）与协方差谱分解，盐蒂估计（Saltelli pick‑freeze）与随机 QMC，Tikhonov 正则化的最小二乘求解，理论分析中的误差分解、Davis–Kahan 角度界，Picard 迭代与收敛分析。

**📊 数据集**

实验数据集为合成高维椭圆 PDE：产品目标、双交互目标、斜向 Poisson、Fokker–Planck、HJB、Allen–Cahn、密集 Riccati，以及带非分离系数的测试。所有实验均在 (0,1)^d 或 (-1,1)^d 上进行，维数从 5 到 100，使用随机频率与偏置。

**📈 对比分析**

与全维 RFM、固定宽度 RFM 以及 PINN（四层 128 宽度）进行对比。结果显示：HA-RFM 通过残差 Sobol 屏蔽和主动子空间补丁，宽度比全维 RFM 少 1% 以内但误差降低 14–100 倍；相对于坐标块 RFM 错误降低 34–100 倍；在半线性问题中，Picard 迭代收敛快，误差达到 10⁻⁸ 级别。CPU 时间与误差的折中在 1–1500 秒之间，显著优于传统全维 RFM 与 PINN。

**⚠️ 局限性**

局限性包括：①依赖于独立同分布（产品）参考测度；②仅针对均匀椭圆算子，非椭圆或时变 PDE 需扩展；③残差 Sobol 估计需要足够 QMC 样本，候选集上限可能限制真正交互的发现；④主动子空间判别依赖于能量与谱间隙阈值，若谱稀疏性不足则无效；⑤需要预设最大交互阶数 K_max，过高会导致宽度激增；⑥理论误差常数与采样稳定性高度相关，实际实现需仔细调参。

---

## 229. Frequency-Hierarchical Active k-Space Sampling for Diagnostic MRI

**arXiv ID:** 2607.19779 | [PDF](https://arxiv.org/pdf/2607.19779v1)

**作者:** Ruru Xu `[一作]` (Istanbul Technical University), Ilkay Oksuz `[通讯]` (Istanbul Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计了 HieraSample，针对MRI加速采样的主动学习框架，逐步增加高频采样点并始终保持低频完整。

**💡 创新点**

创新点在于使用余弦退火的加速调度、低频圆盘基底、Mamba 状态空间网络和以分类损失降幅为奖励的点级采样动作空间。

**🔧 技术方法**

核心技术包括 Mamba 变换器、双任务（疾病/严重程度）分类器、基于交叉熵下降的 REINFORCE 损失、低频圆盘初始化和逐步预算分配。

**📊 数据集**

在 fastMRI+ 关节骨盆（knee）数据集上进行实验，包含 ACL 受伤和软骨病变的诊断与严重程度标签。

**📈 对比分析**

与 ASSDM 基线相比，HieraSample 在 10× 加速下 ACL 严重程度的 AUC 提升 20.4 点，在 20× 加速下 ACL 诊断提升 5.0 点，整体逼近全采样上限，且在 4–10× 之间匹配全采样性能。

**⚠️ 局限性**

局限在于仅针对单线圈单关节数据、使用单一随机种子，且未独立评估余弦调度与低频基底对性能的单独贡献。

---

## 230. DRGBT-1K: A Large-scale High-quality Benchmark for Dynamic RGBT Tracking

**arXiv ID:** 2607.19772 | [PDF](https://arxiv.org/pdf/2607.19772v1)

**作者:** Zhaodong Ding `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

暂无论文内容，无法概述

**💡 创新点**

暂无论文内容，无法确定创新点

**🔧 技术方法**

暂无论文内容，无法确定技术

**📊 数据集**

暂无论文内容，无法确定数据集

**📈 对比分析**

暂无论文内容，无法进行方法比较

**⚠️ 局限性**

暂无论文内容，无法确定局限

---

## 231. Extending a Large View Synthesis Model for Multi-view Panoptic Segmentation

**arXiv ID:** 2607.19765 | [PDF](https://arxiv.org/pdf/2607.19765v1)

**作者:** Kwonyoung Ryu `[一作]` (POSTECH), Jaesik Park `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用已训练好的大视图合成模型（LVSM）实现新视角下的全景分割，且不需要显式三维重建或额外的分割训练。

**💡 创新点**

证明跨视图注意力学习的几何对应关系能在非光照信号（如二进制标签）上传递，提出了通过二进制编码传播全景标签的无监督方法。

**🔧 技术方法**

采用 Less3Depend（基于 transformer 的 LVSM）作为渲染器，使用共享查询解码器（Mask2Former 变体）进行源视图全景分割，并将编码的标签在同一模型中传播；通过梯度显著性分析验证对应关系。

**📊 数据集**

在 ScanNet 公开数据集上训练与评估，跨数据集测试也使用 Replica 数据集。

**📈 对比分析**

与基于 3D 高斯或 NeRF 的 SIU3R、LSM 等方法对比，保持与最优 3D 方法相当的 mIoU（≈0.595）并且在 PSNR 上提升 7 dB；在低重叠、跨数据集场景下表现更稳健。

**⚠️ 局限性**

仅对每个目标视角进行一次完整前向推理，导致在仅需单一视角分割时性能瓶颈；传播机制仅适用于视角不变的信号，无法直接处理视角相关的量如深度。

---

## 232. Asymptotically Optimal Regret for Reinforcement Learning without Horizon Dependence

**arXiv ID:** 2607.19854 | [PDF](https://arxiv.org/pdf/2607.19854v1)

**作者:** Runlong Zhou `[一作]` (University of Washington), Simon S. Du `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种在时间齐次有限状态-动作 MDP 中实现无周期（horizon-free）渐近最优（O(√SAK)）累积回报惩罚的算法。

**💡 创新点**

核心创新包括：1）基于值函数单调性的投影（cut projection）把 V* 的 H 维序列压缩成 O(S/ε²) 个离散向量；2）利用剪裁方差和新的总偏差（total-deviation）上界来控制残差项；3）通过截断规划周期（horizon truncation）实现奖励感知的探索；4）设计“切割奖励”(cutting bonus)在保持乐观性和单调性的同时避免 log H 乘子；5）结合上述工具得到完全无 H 依赖的上界。

**🔧 技术方法**

主要技术手段包括：离散化投影与剪裁方差、全局偏差上界、截断规划与奖励驱动探索、松弛单调性（monotonicity）以及对经验转移概率的双重冻结更新。

**📊 数据集**

本文完全基于理论分析，没有使用任何实际数据集；所有结论均通过概率与最优性证明得出。

**📈 对比分析**

与之前的 O(√(S⁹A³K)) 以及含 log H 的 O(√(SAK log H)+S²A log H) 上界相比，本文的结果在 H 上完全无关、领先项与上下界（O(√SAK)）匹配，只多出一个可接受的 S⁸A³ 的消耗项；相当于在理论上实现了与上下文赌博机（contextual bandit）相当的性能。

**⚠️ 局限性**

剩余瓶颈是消耗项 O(S⁸A³)，论文未说明是否可进一步压缩或消除；如果该项不可忽略，说明 MDP 与上下文赌博机在统计复杂度上仍存在差距；此外，算法在实现细节（如双重冻结计数、参考模型更新）较为复杂，可能在实践中带来计算和实现成本。

---

## 233. Beyond Fail-to-Pass: Iterative Hardening of Co-Generated Bug Reproduction Tests and Fixes

**arXiv ID:** 2607.19843 | [PDF](https://arxiv.org/pdf/2607.19843v1)

**作者:** Yuhao Tan `[一作]` (Nanjing University), Dongmei Zhang `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究生成的bug重现测试（BRT）对下游自动程序修复的实际作用，提出了Mutation Patch Evaluation（MPE）框架，将BRT划分为严格（Rigorous）、宽松（Lax）和不对齐（Misaligned）三类，并基于此进行实证分析。进一步设计了两阶段协同生成（CoGen）循环：先用lax-init生成一个仅观察症状的测试，再通过hardening阶段迭代强化测试与修复，使其从宽松向严格转化。实验验证了该方法在提升修复成功率和测试有效性方面的优势。

**💡 创新点**

创新点包括：①将BRT质量细化为三类，揭示单一Fail‑to‑Pass判定无法区分的“宽松”问题；②引入Temporal Matrix以无参考的方式量化测试的严格程度，将MPE转化为迭代过程中的反馈信号；③构建两阶段协同生成流程，先独立生成测试后再联合优化，显著减少测试与修复的耦合错误；④在多种LLM后端和跨基准上验证了该机制的泛化性。

**🔧 技术方法**

主要技术手段包括：大型语言模型（OpenHands + GPT‑5‑mini，GPT‑5‑4，Claude Opus 4.5）生成测试与修复；基于语义变异器的Mutation Patch Evaluation，利用四种针对性变异操作产生可疑修复样本；Temporal Matrix用于计算当前测试相对上一轮的“宽松率”；两阶段协同生成循环（lax‑init + hardening）实现逐步测试强化；对比实验使用标准评估指标 Resolved 与 F→P 率。

**📊 数据集**

实验数据集主要为 SWE‑bench Verified 与 SWT‑bench Verified 的交集（433个实例），以及 SWE‑bench Lite 与 SWT‑bench Lite 的交集（276个实例）。此外在跨后端实验中还使用 GPT‑5‑4 与 Claude Opus 4.5。

**📈 对比分析**

与fix‑only（如 SWE‑Agent）和传统协同生成（如 InfCode、Agent‑CoEvo、OpenHands+）基线进行对比。CoGen 在 433 个实例上实现 Resolved 69.4%（相比最佳基线提高约 +9.6个百分点）和 F→P 78.9%（相比最佳基线提高约 +6.2个百分点）。在 GPT‑5‑4 和 Claude Opus 4.5 后端上也保持了 2–4% 的提升。每实例成本约 $0.84，低于现有协同生成方法。性能提升主要集中在前几轮 hardening 之后，后期增益趋于饱和。

**⚠️ 局限性**

局限性包括：①MPE 依赖 LLM 生成的变异样本，覆盖范围受模型变异能力和四种变异器的限制，可能遗漏罕见或领域特定的宽松模式；②实验仅在 Python 代码修复基准上验证，未验证对其他语言或安全补丁场景的适用性；③hardening 过程中测试与修复仍处于同一循环，可能导致残余的耦合错误；④Temporal Matrix 的阈值设定基于经验，跨任务迁移时需重新调优。

---

## 234. Know Your Agent: Reconnaissance-Driven Pentesting of AI Agents

**arXiv ID:** 2607.19837 | [PDF](https://arxiv.org/pdf/2607.19837v1)

**作者:** Or Zion Eliav `[一作]` (Ben-Gurion University Of Negev), Yisroel Mirsky `[通讯]` (Ben-Gurion University Of Negev)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在KYA框架下实现了对AI代理的黑盒渗透测试，结合侦察循环主动提取代理工具、策略、任务上下文等知识资产，并基于此构造更有效的间接提示注入攻击。

**💡 创新点**

首次提出“代理侦察”概念，明确了知识资产的三类（逃避、前置、蓝图）及其对应的弱点（表面信任与语境合法性），并将侦察与攻击交替执行的自适应策略引入自动化渗透。

**🔧 技术方法**

采用了多模态规划器+评估器的Orchestrator、基于目标概况的Reconnaissance模块和蓝图化攻击的Exploitation模块，内部使用大型语言模型（如GPT‑5）进行策略生成与Payload合成。

**📊 数据集**

使用AgentDojo与InjecAgent的四大代理类型（Workspace、Slack、Travel、Banking）及其构建的多任务/目标组合（共629实例）进行实验，并在OpenHands编码代理上开展真实世界案例评测。

**📈 对比分析**

相较于静态模板、单射生成与迭代框架基线，KYA在AgentDojo上攻破率从平均47.7%提升至86.0%（最高提升约67pp），在InjecAgent上接近100%；在Gemini Flash等较弱模型上仍能保持高达89.2%的成功率，显示侦察驱动方法在多模型、多场景下稳健领先。

**⚠️ 局限性**

局限性包括需满足对代理的可交互黑盒访问，主要聚焦工具调用型代理，对不具备可探测工具或具备高级防御机制的代理效果有限；此外，方法依赖交互预算与预设的Tactic库，极端动态或多租户环境下的迁移性仍待验证。

---

## 235. Dreamer-CPC: Message Learning with World Models for Decentralized Multi-agent Reinforcement Learning

**arXiv ID:** 2607.19809 | [PDF](https://arxiv.org/pdf/2607.19809v1)

**作者:** Taisuke Takayama `[一作]` (Kyoto University), Tadahiro Taniguchi `[通讯]` (Kyoto University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在分布式多智能体强化学习中，提出了 Dreamer-CPC 方法，该方法通过将 Collective Predictive Coding（CPC）基于信息传递的机制嵌入到基于 RSSM 的世界模型中，实现了基于时间序列潜在状态的消息学习，从而使各智能体能够在缺失观测时进行协调；

**💡 创新点**

创新点在于①将消息学习从仅依赖当前观测扩展到利用世界模型的潜在动态（历史观测与动作），②通过统一的目标函数实现世界模型和消息模块的联合优化，③采用 stop‑gradient 处理跨智能体消息，完成了完全去中心化的训练与执行；

**🔧 技术方法**

核心技术包括：DreamerV3 的 RSSM、CPC 的信息编码、BlockGRU 消息循环网络、MLP 先验/后验网络、KL 平衡损失（Free Bits）以及与 DreamerV3 类似的 actor‑critic 想象轨迹学习；

**📊 数据集**

使用的实验数据集为两个自定义环境 Observer 与 CatchApple，分别模拟非合作信息共享与观察缺失的情形；

**📈 对比分析**

与 IPPO‑CPC、IPPO、无通信 DreamerV3 以及共享观测的 baseline 进行比较。在 Observer 中，Dreamer‑CPC 在 5×4 消息配置下的 IQM 为 39.85，远超 IPPO‑CPC 的 6.81；在 CatchApple 中，Dreamer‑CPC 的 IQM 为 28.53，显著高于 IPPO‑CPC（5.52）和无通信 DreamerV3（5.93）；

**⚠️ 局限性**

局限性包括：未对学习到的消息内容进行分析；实验仅涵盖两智能体场景，缺乏大规模验证；只评估了自定义任务，未验证在更通用 MARL 基准上的表现。

---

## 236. TriAgent: Divergence-Aware Multi-Agent Committees for Cost-Efficient Financial Sentiment Analysis

**arXiv ID:** 2607.19794 | [PDF](https://arxiv.org/pdf/2607.19794v1)

**作者:** Isabel Xu `[一作]` (Overlake School), Jiacheng Ding `[通讯]` (University of Memphis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TriAgent多层金融情感分析系统，将词级词典、句级FinBERT以及跨句子推理LLM组合为一个委员会，并通过三维语义分歧指数（SDI）实现自适应路由与成本控制。

**💡 创新点**

核心创新包括：①层级多粒度委员会与SDI实现的“批评者平台”使得不同参数规模的LLM达到F1≈0.87的饱和；②共享共识词典（SCD）在多语言环境下提供零成本跨语言一致性；③将SDI用作后置幻觉检测，AUC达0.90。

**🔧 技术方法**

采用VADER词典、FinBERT句级Transformer、Qwen2.5‑Instruct多尺寸LLM、Mistral‑7B/Phi‑3.5‑mini、XGBoost边缘预测器、Sentence‑BERT共享共识词典、FAISS缓存、4‑bit微调等技术。

**📊 数据集**

使用4,838条Financial PhraseBank（FPB）句子，扩展至16,769条TFNS以及1,500条中文翻译FPB进行实验。

**📈 对比分析**

与单一代理、同族投票、Always‑L3等基线对比，Critic在Qwen 1.5B–7B维持F1≈0.87，SDI‑单阶段策略获得Sharpe 3.50，SCD在95%命中率下F1达0.99；在10M用户规模下可节省约9.3M美元/年。

**⚠️ 局限性**

局限性包括：批评者平台在Qwen族内可见，跨家族泛化有限；对抗扰动检测仅在部分攻击有效；需预先进行低成本投票验证LLM家族；实验主要基于FPB、TFNS，缺乏真实市场预测验证。

---

## 237. CIR at iKAT SCAI 2026: Exploring Clarification Need Prediction in Agentic Conversational Search

**arXiv ID:** 2607.19801 | [PDF](https://arxiv.org/pdf/2607.19801v1)

**作者:** Nolwenn Bernard `[一作]` (TH Koen), Philipp Schaer `[通讯]` (TH Koen)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一套基于 agentic 架构的对话搜索系统，重点实现了澄清需求预测、澄清问题与答案生成，并与个性化背景知识（PTKB）相结合；

**💡 创新点**

创新在于将传统检索-重排序管线转化为由 orchestrator 直接调用的工具集合，实现了零散的 agentic 流程，并通过对比 MuSIc 与 Zef-CNP 两种澄清需求预测模型验证其效果；

**🔧 技术方法**

采用 T5（CANARD）进行查询重写，BM25+monoT5 进行检索与重排序，BERT 与 Zef-CNP 进行澄清预测，MuSIc 作为另一预测模型，利用零射 LLM（Gemma‑4‑26B 等）进行澄清问题和答案生成，整体框架基于 Pydantic‑AI、vLLM、Pyserini 与 PyTerrier；

**📊 数据集**

使用了 CANARD（查询重写）、MSDialog+ClariQ（MuSIc 训练）、Zef‑CNP 生成的合成数据（澄清预测）、monoT5‑msmarco（重排序）、PTKB（个性化背景语料）等；

**📈 对比分析**

通过 iKAT 2026 shared‑task 的交互式评估，采用 LLM‑as‑a‑judge 进行对话级与 rubric 级评分；两跑的差异不大，混合主动性得分平均约 3/5，检索指标（AP、nDCG）极低（AP<0.03），表明系统在检索与澄清需求判定方面表现有限；

**⚠️ 局限性**

主要局限包括检索性能低下导致大部分用户发问未能得到检索结果；orchestrator 的执行流程过于松散，导致工具调用顺序不稳定；澄清问题与答案生成依赖大模型零射，资源消耗高；缺乏无澄清模型基线及更复杂的修复策略。

---

## 238. emb-diversity: A Tool for Embedding-Based Measurement of Data Diversity

**arXiv ID:** 2607.19848 | [PDF](https://arxiv.org/pdf/2607.19848v1)

**作者:** Cantao Su `[一作]` (Utrecht University), Anna Wegmann `[通讯]` (Utrecht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个名为 emb-diversity 的 Python 工具，用于计算并比较 22 种基于嵌入的多样性指标。

**💡 创新点**

创新点在于将多种分散的嵌入多样性度量整合到统一、可扩展的框架中，并提供标准化的 API 与 CLI，使研究者和工程师能够轻松对不同数据集和多样性轴进行对比。

**🔧 技术方法**

主要技术包括：使用 Hugging Face 的文本/语音嵌入模型、基于距离矩阵与图结构的度量（如 Vendi、Graph Entropy、MST Dispersion）、UMAP 投影、向量统计；同时实现了文本长块分割、缓存机制以及多线程/批处理优化。

**📊 数据集**

使用的数据集包括：Common Pile（新闻、Arxiv、GitHub、YouTube 等多域文本），GEDE 论文写作数据（人类 vs LLM 生成），语言向量工具箱中的语言结构向量，以及 CMU ARCTIC 语音数据。

**📈 对比分析**

通过在受控混合域数据上对多样性指标进行比较、对 22 种度量在不同规模数据集上的运行时间进行基准测试，并进行 9 位 NLP 研究者的可用性评估。结果显示：Vendi、Mean Pairwise Distance 与 Graph Entropy 能稳定捕捉多样性变化，部分图论/旅行商类度量在大规模数据上耗时过长。

**⚠️ 局限性**

局限性包括：缺乏统一的多样性分数解释框架、文档不够详尽导致上手门槛；某些度量（如 Hamiltonian、DCscore）在大规模数据下内存/时间消耗高；目前仅支持已存在的嵌入轴，尚未覆盖所有潜在多样性维度。

---

## 239. Towards Reliable C-to-Rust Translation with Rule-Guided Reasoning and Reinforcement Learning

**arXiv ID:** 2607.19966 | [PDF](https://arxiv.org/pdf/2607.19966v1)

**作者:** Feng Luo `[一作]` (Harbin Institute of Technology), Kui Liu `[通讯]` (Huawei)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于大语言模型的 C‑to‑Rust 翻译框架 TRAVEL，利用 Rust 规则引导的 MCTS 生成翻译思路路径，并通过双重奖励（执行奖励+思路奖励）进行强化学习，以提升翻译质量、编译成功率和安全性。

**💡 创新点**

创新点：① 将 Rust 专属语法规则嵌入 MCTS 搜索，显著降低编译错误；② 通过构造可验证组件（自动生成测试用例）实现语义一致性评估；③ 采用双重奖励 GRPO，兼顾语义正确性与规则遵循；④ 在多种 LLM 上验证，显示小模型可逼近甚至超过更大模型。

**🔧 技术方法**

技术：大语言模型（Qwen2.5‑Coder、CodeGemma、GPT‑4o 等），Monte Carlo Tree Search（MCTS）+规则约束，奖励模型训练，Group Relative Policy Optimization（GRPO）强化学习，自动化测试生成（Verifiable Components）。

**📊 数据集**

数据集：xCodeEval（公开函数级基准），OS‑Bench（Linux kernel 函数），HW‑Bench（华为工业代码）。

**📈 对比分析**

方法比较：与 Prompting、ICL、RAG、Vert、IRENE、SFT 等基线对比。TRAVEL 在 xCodeEval 上 CA 提升约 26%–35%，CSR 提升约 18%–30%；在 OS‑Bench、HW‑Bench 上 CSR 提升 16%–22%，安全率 UR 降低约 13%。在多规模模型上，TRAVEL 在 7B 级别模型已逼近 30B 级别模型甚至超越部分闭源模型，展示出较高的可扩展性和实用性。

**⚠️ 局限性**

局限性：① 依赖底层 LLM 的生成质量，弱模型可能导致搜索与训练效果受限；② 语义验证基于 LLM 生成的测试用例，覆盖率有限，可能出现未检测的功能偏差；③ 可能存在预训练数据泄漏风险，难以完全排除。

---

## 240. Identity-Truthful Online Decision-Making

**arXiv ID:** 2607.19964 | [PDF](https://arxiv.org/pdf/2607.19964v1)

**作者:** Tomer Ezra `[一作]` (Tel Aviv University), Adar Kantor `[通讯]` (Tel Aviv University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在贝叶斯在线选择问题中，引入身份真实（identity‑truthful）决策规则，即决策者只能在拒绝或接受后才获知当前奖励的身份，并且通过构造算法来研究该信息约束下的性能；

**💡 创新点**

提出并度量身份真实间隙（identity‑truthfulness gap），证明其严格大于 0.5 但不超过 0.81，从而揭示该信息约束比完全身份盲目信息更有利；同时证明存在实例使得最优身份真实算法无法实现任何定价机制，打破了传统线上算法与定价机制之间的等价关系；

**🔧 技术方法**

采用阈值分解、剩余神谕（residual prophet）与剩余自由奖励（free‑reward）概念，对难度实例进行“阈值硬度”划分，设计基于高阈值和自适应阈值的分两步算法，并使用凸分析、积分不等式与线性规划求解最佳可接受概率；

**📊 数据集**

本工作为理论分析，未使用具体实验数据集，全部结论均来自数理证明与数学优化；

**📈 对比分析**

通过对比最优在线算法（知道全部身份顺序）与身份真实算法，给出了下界（>0.5）与上界（≤0.81）的近似比；还通过实例与线性规划验证身份真实上限为约 0.81；

**⚠️ 局限性**

局限性：仅关注单一选择（single‑choice）贝叶斯在线问题；对更一般的组合或匹配设置未给出；算法在某些参数取值下可能非最优；对实际可实现性（如计算复杂度）缺乏讨论。

---

## 241. G-MAD: A Game-Based Data Generation Framework for Multi-View RGB-T Aerial Object Detection

**arXiv ID:** 2607.19942 | [PDF](https://arxiv.org/pdf/2607.19942v1)

**作者:** Yechan Kim `[一作]` (LIG Defense&Aerospace), Moongu Jeon `[通讯]` (GIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了G‑MAD框架，利用Arma3游戏引擎生成同步多视角RGB‑T航空图像，并自动从引擎几何信息中生成HBB/OBB标注；基于该框架构建了大规模多视角RGB‑T军事目标检测数据集AMOD。

**💡 创新点**

创新点包括①将Arma3游戏环境作为结构化、可控的RGB‑T数据生成平台，支持多视角采样和多模态同步；②通过引擎原生几何信息实现高质量自动标注；③提供可扩展的场景约束与摄像机采样模型；④首次构建同一场景多视角、多模态数据的公开基准AMOD。

**🔧 技术方法**

技术手段主要是Arma3的SQF脚本实现场景与摄像机控制、RGB与热成像同步捕获；Python脚本与Matplotlib/OpenCV完成数据导出和可视化；使用引擎几何查询生成3D Bounding Box 并投影为2D HBB/OBB；结合MMRotate、Swin‑S骨干的Oriented R‑CNN进行检测训练。

**📊 数据集**

使用数据集：自研的AMOD（1920×1440 RGB‑T 多视角、约383k OBB标注），并与公开RGB‑T航空基准（MULTISPECTRAL、FLIR_ADAS、DroneVehicle等）进行对比。

**📈 对比分析**

评估方法：在AMOD上分别训练单视角与多视角模型，使用Oriented R‑CNN+Swin‑S，单视角AP_50:95为44.57，多视角提升至50.96（+6.39）。预训练+微调实验显示，在DIOR‑R（可见光）上从ImageNet 63.14提升到69.83（+6.69），在HIT‑UAV（热成像）上从74.57提升到77.08（+2.51）。此外，在未见真实军事图像上无需微调，AMOD训练的模型仍能检测出合理目标，验证了跨域迁移能力。

**⚠️ 局限性**

局限性：AMOD主要聚焦军事目标，非军事通用目标覆盖有限；依赖Arma3资产与场景，数据多样性受游戏内资产限制；使用需合法Arma3授权，限制了商业化或非学术用途。

---

## 242. MOF-Sleuth: Tool-Grounded Reward Alignment for Explainable Fine-Grained MOF CIF Auditing

**arXiv ID:** 2607.19935 | [PDF](https://arxiv.org/pdf/2607.19935v1)

**作者:** Yu Liu `[一作]` (Chinese Academy of Sciences), Guobin Zhao `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出MOF-Sleuth框架，利用Forensic Lab生成可计算化学证据，再由Sleuth LLM进行二元判定、15类细粒度错误归因与证据引用解释。

**💡 创新点**

创新在于将可计算证据与LLM推理分离，并通过强化学习奖励机制指导模型在解释中精准引用证据与支持诊断，同时定义Chemically Grounded Diagnosis指标实现可验证的解释质量。

**🔧 技术方法**

采用结构化工具集（Forensic Lab）对CIF进行事实、硬标记、信号等提取，Sleuth基于Qwen3-4B-Instruct并用GRPO强化学习训练，结合规则验证器与奖励设计。

**📊 数据集**

使用CoRE-MOF 2019、CoRE-MOF 2026、ToBaCCo、QMOF等公开MOF CIF数据集（含人工标注错误类别）进行训练与评估。

**📈 对比分析**

与原始CIF LLM、API模型、通用agent框架及MOF专用验证器等十余种基线在二元检测、四类归因和Chem-GD上对比，MOF-Sleuth平均准确率0.781、Chem-GD 0.713，显著优于所有基线。

**⚠️ 局限性**

局限在于对错误类别的依赖、需大量人工标注、在稀有错误类别召回仍受限，以及模型规模与推理成本仍存在提升空间。

---

## 243. The Dynamic Turn in Paraconsistency

**arXiv ID:** 2607.19906 | [PDF](https://arxiv.org/pdf/2607.19906v1)

**作者:** Rafael Ongaratto `[一作]` (UNICAMP), Hans van Ditmarsch `[通讯]` (University of Toulouse)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并完善了基于LFI1的动作模型和更新模型逻辑AMLFI1与UMLFI1，给出其语义、完整性与可决定性证明，并以此框架建模并分析诸如Coup游戏中的谎言、错误与拜占庭行为。

**💡 创新点**

创新点在于将矛盾信息纳入动态推理框架，首次将局部不一致性与事实更改结合，通过LFI1的等价关系实现动作模型的归约与可归一化；同时展示了不同框架（PALFI1、PALFI1_C）等价性。

**🔧 技术方法**

采用归约规则RE、内部归约策略、公式复杂度度量、强等价证明技术以及Kripke模型与动作模型的语义连接。

**📊 数据集**

无数据集，纯理论推导。

**📈 对比分析**

无经验对比，主要通过逻辑可证明性与可决定性证明来展示其优越性；未给出数值性能指标。

**⚠️ 局限性**

局限在于无法区分有意谎言与无意错误，且对完整的信念收缩与关系扩展等高级修正机制未在本文中实现。

---

## 244. HijackKV: New Threat in Position-Independent KV Cache Reuse

**arXiv ID:** 2607.19957 | [PDF](https://arxiv.org/pdf/2607.19957v1)

**作者:** Yichi Zhang `[一作]` (Pennsylvania State University), Yuchen Yang `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种无需系统权限即可在位置无关 KV 缓存重用机制下对大语言模型（LLM）进行 hijacking 的攻击框架，展示了其高成功率、持久性和跨模型、跨场景的可迁移性。

**💡 创新点**

创新点在于：①首次系统化挖掘并利用 KV 缓存与上下文脱耦导致的安全缺口；②提出基于 Greedy Coordinate Gradient 的离散词元优化方法，用来生成能“植入”恶意 KV 状态的前缀；③通过实验验证该攻击在多种 LLM 与多种 KV 缓存配置下均保持高效，且对现有重算、压缩等防御措施无效。

**🔧 技术方法**

主要技术：位置无关 KV 缓存重用、跨租户共享缓存、Greedy Coordinate Gradient (GCG) 优化、跨模型迁移攻击、对 KV 缓存的模拟与评估、文本生成与交互式问答实验。

**📊 数据集**

使用了四个问答基准数据集：HotpotQA、SQuAD（v1.1 & v2.0）、MedQA 和 PubMedQA，随机采样 200 条样本进行攻击实验。

**📈 对比分析**

对比方法：评估基准模型在不受攻击时的 Accuracy、攻击后的 Untargeted Attack Success Rate (U-ASR) 与 Targeted Attack Success Rate (T-ASR)。实验显示：单次攻击平均 94% 的 T-ASR；即使在 10% hit 率、50% 重算比例下仍保持高效；对 1B–70B 的 Qwen、LLaMA、Mistral 等多种 LLM 均可跨模型迁移，黑盒环境下也能保持高成功率。

**⚠️ 局限性**

局限性：攻击效果取决于恶意 KV 能够被插入并被后续请求命中，受缓存占用、替换策略（FIFO/LRU/LFU）及内容流行度影响；对缓存占用的探测会产生侧信道，可能导致自污染；对极少共享或高频更新的内容适用性有限。

---

## 245. ETPDesigner: Multi-Agent Orchestration for Interactive Multimodal Electronic Theater Program

**arXiv ID:** 2607.19947 | [PDF](https://arxiv.org/pdf/2607.19947v1)

**作者:** Mengtian Li `[一作]` (Shanghai University), Chaofeng Chen `[通讯]` (Institute for Math and AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ETPDesigner多代理框架，将戏剧剧本自动转化为完整的电子戏剧程序（海报、信息页、角色画像），并通过动画、语音合成实现交互式体验。

**💡 创新点**

1）多代理协同架构，将语义分析、艺术指导、生成、评估、风格提取、组合拆分为专门角色；2）RAG+Tree‑of‑Thought自评循环，提升视觉质量；3）全局风格锚点与程序化布局，确保跨资产一致性；4）集成角色动画与LLM驱动对话，扩展到交互式多模态应用。

**🔧 技术方法**

大型语言模型与检索增强生成、视觉语言模型评判器、扩散模型文本‑图像、Tree‑of‑Thought思维树、风格提取向量、程序化布局引擎、角色动画、定制语音合成以及LLM对话系统。

**📊 数据集**

自建ETP‑Pro基准，包含100份专业电子戏剧程序（海报、信息页、角色画像），配有专家标注的主题、风格与排版标签。

**📈 对比分析**

与Seedream 4.5、Nano Banana Pro等T2I模型及AutoGen多代理框架对比；在VLM评估（美学、传播效能、主题忠实度）以及CLIP‑I、DINOv2等客观指标上均取得最高分；人类评测中在视觉吸引力、布局功能、整体一致性、主题关联四项均明显优于基线。

**⚠️ 局限性**

目前对多语言剧本支持有限；极长剧本的上下文保持仍有挑战；生成细节高度依赖程序化布局，易受布局模板限制；交互式功能尚未在多文化、多场景下广泛测试，需进一步提升跨文化适配与可扩展性。

---

## 246. SIINR: Structurally Informed Implicit Neural Representations for super-resolution with uncertainty quantification of clinical quality diffusion MRI datasets

**arXiv ID:** 2607.19943 | [PDF](https://arxiv.org/pdf/2607.19943v1)

**作者:** Tom Hendriks `[一作]` (Eindhoven University of Technology), Maxime Chamberland `[通讯]` (Eindhoven University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e15e3743-5ee0-4d5f-813d-d146868082fc` `70e40602-aae3-44bd-80ec-4a7f2674330f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了SIINR框架，利用监督式3D U‑net先粗略上采样，再结合自监督隐式神经表示（INR）对厚切dMRI数据进行高质量超分辨并可视化不确定性。

**💡 创新点**

其创新点在于将U‑net作为先验与INR相结合，既能保持对原始低分辨率扫描的自洽性，又能通过高斯过程得到解析后验分布，实现对重建结果的量化不确定性估计。

**🔧 技术方法**

技术上使用3D U‑net、Fourier特征编码的多层感知机（MLP）INR、基于SSIM+MSE的联合损失、贝叶斯高斯过程后验、以及蒙特卡罗采样进行下游指标的置信度评估。

**📊 数据集**

训练数据来自多个公开数据库（HCP‑YA、HCP‑EP、HCP‑AABC、CDMRI挑战），测试则包括真实临床病例（多发性硬化、脑瘤病灶）以检验对分布外数据的鲁棒性。

**📈 对比分析**

在与线性/三次插值的比较中，SIINR在DWI、ADC、FA和AFD指标上均显著降低MSE/RMSE，并在解剖结构重建和纤维追踪中表现出更高的空间精度和更合理的不确定性分布。

**⚠️ 局限性**

局限性包括：训练集对扫描参数和病理多样性的覆盖不完整；降采样过程使用三线性插值而非真实物理采集；缺乏完全无噪声的真值导致误差评估的解释性有限；以及INR超参数需针对不同任务进一步调优。

---

## 247. Nonlinear Bias-Compensated Adaptive Filter and Its Application for Time-Series Prediction

**arXiv ID:** 2607.19902 | [PDF](https://arxiv.org/pdf/2607.19902v1)

**作者:** Yi Peng `[一作]` (Southwest Jiaotong University), Jinhui Hu `[通讯]` (Southwest Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种基于随机傅里叶特征的偏置补偿自适应滤波器（RFFBCGA），用于在输入噪声存在的非线性误差变量模型下进行时间序列预测。

**💡 创新点**

创新点在于将随机傅里叶特征固定网络、偏置补偿项和可调的通用自适应（GA）成本函数相结合，既解决了输入噪声干扰，又通过GA函数提升对非高斯噪声的鲁棒性，并给出了均值与均方稳定性理论。

**🔧 技术方法**

采用了随机傅里叶特征映射、偏置补偿技术、GA成本函数、均值/均方稳定性分析以及Monte Carlo仿真与真实时间序列预测的实验方法。

**📊 数据集**

使用了合成非线性系统、太阳黑子月度时间序列、奇异的Chua电路时间序列以及含非高斯噪声（Bernoulli-Gaussian、α-稳定）等数据集进行实验。

**📈 对比分析**

通过与KLMS、RFFMCC、KMCC、BCKLMS等算法在AWGN、BG噪声、α-稳定噪声等多种噪声环境下的比较，RFFBCGA在收敛速度、稳态误差和鲁棒性方面均优于对比方法。

**⚠️ 局限性**

局限性包括需要手动调参（如ξ、l、δ），对高维输入噪声方差较大的场景步长受限，且理论分析假设输入/输出噪声独立且高斯，实际情况可能导致性能下降。

---

## 248. Harnessing Disagreement: Detecting Correlated Agreement Blindness in Multi-Agent Triage

**arXiv ID:** 2607.19899 | [PDF](https://arxiv.org/pdf/2607.19899v1)

**作者:** Shay Seiya McDonnell `[一作]` (Trinity College Dublin), Gregory M. P. O'Hare `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ARAT（Arbitrated Reasoning Agents for Alarm Triage）多代理架构，利用协商式“有向星形”拓扑结合 Random Forest、k‑NN 和一层路由器，解决了多代理一致性导致的安全盲区。

**💡 创新点**

创新点在于：① 量化并揭示了“相关一致性盲区”（correlated agreement blindness）——模型准确提升反而削弱了基于不一致的安全信号；② 通过两层可解释的路由规则（保守覆盖和一致性安全标记）以及校准的升降级元模型，系统在不需重训练基础模型的前提下显著降低危险性低估率；③ 在跨域验证和模型替换实验中证明了机制的普适性与可诊断性。

**🔧 技术方法**

核心技术包括：有向星形多代理架构；Random Forest 与 k‑NN 两种可解释基模型；软不一致得分（熵+二元差异）、保守覆盖、统一安全门、以及校准的逻辑回归升降级元模型；对比实验使用 Clopper–Pearson 置信区间与 95% 置信区间。

**📊 数据集**

主要数据集：UNSW‑NB15 网络入侵检测（82,332 hold‑out）和 UCI Diabetes 130‑US Hospitals（20,354 test）用于跨域验证；两者都被映射到四级/三级严重性等级。

**📈 对比分析**

与单模型、损失调节的 LightGBM/XGBoost/CatBoost、以及软投票聚合进行比较。ARAT 在 UN‑NB15 上将危险低估率从 4.80% 降至 1.70%，运营精度提升至 98.3%；软投票 4.80%，单模型约 3.5‑6.1%；损失调节的 LightGBM 仅 3.35% 低估率，未能复制 ARAT 的架构优势。

**⚠️ 局限性**

局限性包括：① 低估率结果假设人工审核能完全纠正升降级错误；② 只在两种可解释基模型与两类表格数据上验证，缺乏对更复杂模型或非表格数据的评估；③ 未与学习型多代理协作框架对比，可能隐藏更优的协调方案。

---

## 249. LAVIFT: Latent-Action-Guided Vision Fine-Tuning for Surgical Interaction Recognition

**arXiv ID:** 2607.19889 | [PDF](https://arxiv.org/pdf/2607.19889v1)

**作者:** Jiajun Cheng `[一作]` (Arizona State University), Shan Lin `[通讯]` (Arizona State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于隐式动作（latent-action）引导的端到端视觉–语言微调框架，通过逆动力学模型（IDM）压缩相邻帧的变化，并用前向世界模型（FWM）预测下一帧，从而让视觉编码器聚焦于手术器械与组织的交互区域并提升动作识别性能。

**💡 创新点**

创新点包括：①将 IDM 与 FWM 联合训练作为对视觉编码器的自监督压缩约束，使其学习更具时空细粒度的特征；②引入 Patch‑level SIG 正则化防止局部特征坍塌；③无需额外的框选或伪标签，即可实现对手术交互区域的更好定位；④在 VLM 微调过程中实现对空间与语义对齐的双重提升。

**🔧 技术方法**

使用技术：预训练视觉‑语言模型（如 CLIP、DINOv2、V‑JEPA）作为视觉编码器；逆动力学模型（Delta‑IDM）与前向世界模型（DiT‑style FWM）；Sliced‑Cramer‑Wold 正则化（SIGReg）与 Patch‑level SIG；多头 Transformer 聚合器与 EmbeddingGemma 文本编码器；多标签 BCE 与 cosine‑similarity 识别损失。

**📊 数据集**

使用数据集：CholecT50（50 例胆囊切除视频，100 交互三元组）和 ProstaTD（21 例前列腺切除视频，89 交互三元组）。

**📈 对比分析**

与传统的“冻结视觉编码器+微调文本头”以及“全微调视觉编码器”方法做对比；在多种预训练模型（CLIP ViT‑L/14、DINOv2‑L/14、V‑JEPA2/2.1、TimeSformer 等）上实验，发现该框架在 mAP_IVT 上提升 1.2–4.7 分（ViT‑L）至 3–5 分（DINOv2、SurgeNet），并在 instrument/verb/target 各子指标均有显著提升，且在空间聚类（CH）和交互区域对齐（IoU_norm）上表现更佳。

**⚠️ 局限性**

局限性：对超参数（如 SIG 正则化强度、IDM 维度）较为敏感；目前仅在短期两帧的微调上验证，未扩展到更长时序理解；需手术数据的标注与掩码来评估交互区域，难以在无标注环境下推广；对不同视觉编码器的效果差异仍需进一步分析。

---

## 250. Current Injection Spiking Neural Network for Infrared and Visible Image Fusion

**arXiv ID:** 2607.19879 | [PDF](https://arxiv.org/pdf/2607.19879v1)

**作者:** Rui Zhao `[一作]` (Nanyang Technological University), Weisi Lin `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于膜电位融合的脉冲神经网络实现红外与可见图像融合。

**💡 创新点**

通过在脉冲前的膜电位层进行跨模态电流注入，提出CIS与BCMF模块实现细粒度信息融合。

**🔧 技术方法**

采用脉冲神经网络、可学习PLIF神经元、双向交叉模态融合、重参数化输出头以及输出均值监督等技术。

**📊 数据集**

在MSRS、RoadScene、M3FD、FMB四个公开红外可见融合基准集上进行训练与评测。

**📈 对比分析**

与15种主流CNN/Transformer/生成式/任务驱动融合方法对比，CIS‑Fuse在七项融合指标和下游检测/分割任务上均名列前茅，同时推理能耗约低于相同规模ANN方法十倍。

**⚠️ 局限性**

受限于脉冲时间步数上限、对光照/遮挡鲁棒性待验证以及缺乏大规模训练集等因素。

---

## 251. KineBench: Benchmarking Embodied World Models via IDM-Free Kinematic Grounding

**arXiv ID:** 2607.19876 | [PDF](https://arxiv.org/pdf/2607.19876v1)

**作者:** Zeyu Liu `[一作]` (China Telecom), Xuelong Li `[通讯]` (China Telecom)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出KineBench，一个无IDM的闭环评估框架，通过显式3D运动学校准检验嵌入式世界模型的物理可执行性。

**💡 创新点**

创新点在于完全移除逆动力学模型，直接从生成视频中提取6D末端执行器姿态，并加入SPARC与Maruyama操纵性指数两种经典机器人运动学指标。

**🔧 技术方法**

采用YOLOv11进行分割、MoGeV2实现深度估计、FoundationPose完成6D姿态追踪，随后在ManiSkill3物理仿真器中闭环执行。

**📊 数据集**

使用ManiSkill3的20个多样化操控任务和对应的训练/测试数据集，并在GitHub/HuggingFace上公开相关数据。

**📈 对比分析**

与传统IDM基准对比，KineBench在四个评测套件（基本执行、任务迁移、视觉OOD、复杂度缩放）中显示出更低的轨迹误差和更高的执行成功率，尤其在高复杂度任务中表现出非线性规模化趋势。

**⚠️ 局限性**

局限在于依赖分割、深度估计等视觉模块的精度，对遮挡、光照变化仍敏感，且当前动作库主要来自规划而非人类操控。

---

## 252. Unified Prediction and Planning via Conflict-Aware Disjoint Parameter Training

**arXiv ID:** 2607.19971 | [PDF](https://arxiv.org/pdf/2607.19971v1)

**作者:** Taewon Seo `[一作]` (DGIST), Daehee Park `[通讯]` (DGIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出冲突感知的分离参数训练框架 DPT，并结合稀疏合并（Sparse Merging）实现预测与规划任务的统一，在共享编码器中解决 Skill Conflict。

**💡 创新点**

创新点：1) 通过二进制掩码将预测与规划任务的核心参数区域分离，消除 Skill Conflict；2) DPT 的渐进式掩码扩展与冻结策略，保证任务专精并避免过度干扰；3) 仅合并高影响力参数的稀疏合并，进一步抑制邻近特征干扰；4) 在资源受限的 edge 设备上实现统一模型。

**🔧 技术方法**

技术手段：共享 Transformer 编码器；游戏理论联合预训练；任务特定微调与 DPT；模型合并技术（Task Arithmetic、Ties Merging、Localize‑and‑Stitch、T‑Switch）；稀疏掩码与参数冲突分析。

**📊 数据集**

数据集：主要使用 JRDB 与 JTA 人群导航数据集；在补充材料中扩展到 Bench2Drive 端到端驾驶数据集。

**📈 对比分析**

比较方法与性能：与 DIPP、DTPP、Ensemble、Task Arithmetic、Ties Merging、Localize‑and‑Stitch、T‑Switch、PCGrad、CAGrad 等基线在规划 ADE、碰撞率、FDE、Miss Rate、预测 ADE/FDE 等指标上对比；在 JRDB 与 JTA 上 DPT+Sparse Merging 均显著优于基线，规划 ADE 降至 ~0.40、碰撞率 ~0.009，预测 ADE ~0.84；在端到端驾驶实验中提升闭环 RC、DS、L2 等指标。

**⚠️ 局限性**

局限性：需额外训练阶段与掩码管理，GPU 训练成本略高；手工设定的参数分配比例与掩码比例缺乏自适应机制；验证范围主要为两任务与少量子任务，扩展到更多任务仍需研究；极稀疏掩码下的泛化与鲁棒性尚未完全验证。

---

## 253. OffNadirLoc: Benchmark and Framework for Challenging UAV-to-Satellite Geo-Localization under Large Off-Nadir Views

**arXiv ID:** 2607.19951 | [PDF](https://arxiv.org/pdf/2607.19951v1)

**作者:** Qian Qiao `[一作]` (Northwestern Polytechnical University), Peng Wang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了大离视角（70°~85°）的UAV‑卫星跨视角定位基准OffNadirLoc，并设计了ONLoc框架来解决极端视角下的几何失真和多视角一致性问题。

**💡 创新点**

创新点包括：①结构感知上下文加权（SACW）机制，动态抑制重复/遮挡区域，突出几何可靠的结构特征；②视角一致学习（VCLS）策略，将同一地点的多视角UAV图像与对应卫星图像聚为一个语义组，采用组级相似度约束实现跨模态、跨视角的一致性与区分度提升。

**🔧 技术方法**

使用ViT‑DINOv2（或ConvNeXt）视觉编码器，双层MLP投影、相似度矩阵、Soft‑Clustering、组级多相似度损失；整体采用多视角聚合、重构损失、对比学习的组合。

**📊 数据集**

主要使用OffNadirLoc（44个地区，约9.7k UAV图像，5个离视角；对应卫星图像）进行训练和评测；此外还在四个近视角基准（University‑1652、SUES‑200、DenseUAV、GTA‑UAV）进行零样本迁移和标准训练评估。

**📈 对比分析**

与Sample4Geo、DAC、CAMP、MEAN、Game4Loc、ConGEO等SOTA方法在OffNadirLoc上对比，ONLoc在Recall@1、@3、@5、mAP上均领先6–9%（DINOv2基准下Recall@1提升至72.6%），在零样本迁移中在所有四个近视角基准上保持最优或相近的性能。

**⚠️ 局限性**

局限性：①在DenseUAV等高重叠密集场景仍显难，因评测仅接受精确匹配；②对极端遮挡、光照变化的鲁棒性尚待进一步验证；③方法主要针对UAV‑卫星跨视角，尚未验证在其他遥感模态或更大尺度上迁移的通用性。

---

## 254. SenWorld: A Digital-Twin Simulation for Generating Context-Rich Evaluation Data

**arXiv ID:** 2607.19949 | [PDF](https://arxiv.org/pdf/2607.19949v1)

**作者:** Zenghui Zhou `[一作]` (ByteDance Inc), Tianming Lei `[通讯]` (ByteDance Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了 SenWorld，一个物理约束的数字孪生仿真，用来生成可直接标注、可复现的手机助手评估数据。

**💡 创新点**

将完整的设备状态与物理世界、服务、个人行为耦合在同一事件源仿真中，并通过快照指针提供“无判定器”的确切答案，解决了评估oracle难题。

**🔧 技术方法**

采用事件驱动的确定性仿真框架、可插拔的 LLM/规则脑、数字孪生存储、全系统快照与只读投影，并利用 Jensen‑Shannon、Kolmogorov‑Smirnov 等统计方法进行评估。

**📊 数据集**

使用北京的真实地图、天气、节假日、网络点（BSSID）等数据，以及真实用户的持有记录作为基准，生成数据则基于 16 名 LLM 驱动的人物。

**📈 对比分析**

通过 JSD、TVD、KS 检验对比生成与基准的类别分布、日节律和记录长度，结果显示类别分布 JSD=0.07，通信记录节律 JSD<0.1；在 717 条评估案例中，手机助手失败率 10.9%，失败集中在呼叫与短信记录。

**⚠️ 局限性**

仅模拟单个城市、短周期（一天）、缺乏交互脚本、生成文本长度与真实差异、对硬件设备（耳机、手表）未覆盖，以及对物理约束假设可能不完全准确。

---

## 255. WearWow: Native 2K Multi-Garment Virtual Try-On via Adaptive Token Packing and Preference Alignment

**arXiv ID:** 2607.19923 | [PDF](https://arxiv.org/pdf/2607.19923v1)

**作者:** Xujie Zhang `[一作]` (Sun Yat-sen University), Xiaodan Liang `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了WearWow框架，实现了原生2K多件服装无遮罩虚拟试衣；

**💡 创新点**

创新点在于Adaptive 2D Token Packing（ATP）降低了多条件下的显存爆炸，并通过Multi‑dimensional Try‑On Reward（MTR）结合物理分布奖励与语义引导双重奖励，显著恢复高频细节；

**🔧 技术方法**

采用基于Transformer的扩散模型，ATP进行二维装箱与稀疏令牌裁剪，MTR使用Cloth Distribution Reward与Semantic Guidance Reward，辅以CLIP/CLIP‑style语义相似度；

**📊 数据集**

使用自建WearWow‑2K数据集，包含约100k个训练三元组、2000个测试三元组，分辨率2048×1536，涵盖多件服装、鞋履及配饰；

**📈 对比分析**

与单件试衣基准（VITON‑HD、DressCode）及多件高分辨率基线（OminiTry、QWen‑image、Seedream4.0、Nano Banana Pro、KeLingv3、GPT‑image 1.5）对比，WearWow在FID、KID、LPIPS、SSIM以及人类评估指标均优于所有竞争者；

**⚠️ 局限性**

局限在于对极端后视遮挡的处理仍不足、缺乏3D人体先验、以及视频时序一致性未实现。

---

## 256. DGNA: Dissecting GPU NUMA Architecture through Microbenchmarking and Data Analysis

**arXiv ID:** 2607.19922 | [PDF](https://arxiv.org/pdf/2607.19922v1)

**作者:** Changxi Liu `[一作]` (National University of Singapore), Trevor E. Carlson `[通讯]` (National University of Singapore)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本工作提出并实现了 DGNA 方法，利用微基准和数据分析，系统揭示了 NVIDIA A100、H100（以及 RTX 5090）GPU 内存子系统中的 NUMA 架构与读写机制。

**💡 创新点**

其创新点在于：①不依赖厂商提供的隐式指令，独立测量 L2 与 DRAM 延迟；②采用高斯混合模型自动剔除 DRAM 刷新、TLB 漏洞等噪声，实现更精确的延迟分布；③首次发现 H100 在 L2 级别存在子‑NUMA 结构，并解析 SM、GPC 与 NUMA 节点的对应关系。

**🔧 技术方法**

技术上主要使用自定义 CUDA 内核进行 L2/DRAM 访问计时，结合 K‑means 聚类与随机投影构造特征向量，再用 Gaussian Mixture Model（GMM）对延迟数据进行异常值过滤和分布建模。

**📊 数据集**

实验数据集来源于对 A100、H100（SXM5 模式）以及 RTX 5090（Blackwell）显卡的实测延迟，覆盖多种内存访问模式（本地/远程 L2、DRAM、子‑NUMA）。

**📈 对比分析**

与 NVIDIA 原生 intrinsics（如 __ldcg）对比，DGNA 的测量结果在 A100 上显示显著差异，说明原始指令隐藏了额外操作；在 H100 上两者相近。通过 DGNA 获得的 NUMA 结构能帮助优化内存分配与调度，提升显存利用率。

**⚠️ 局限性**

局限性包括：①假设 GPU 不存在硬件预取器，若后续架构引入预取可能导致测量偏差；②方法仅针对已公开的 NV GPUs，无法直接迁移至闭源或极新硬件；③需要大量 kernel 启动与同步，测量时间较长；④在 NUMA 节点数大于两时需要改进高维向量投影与聚类策略。

---

## 257. Diffusion ReRoll: Revisable Denoising for Robotic Sequential Prediction

**arXiv ID:** 2607.19919 | [PDF](https://arxiv.org/pdf/2607.19919v1)

**作者:** Seonsoo Kim `[一作]` (Agency for Defense Development), Jun-Gill Kang `[通讯]` (Agency for Defense Development)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Diffusion ReRoll，一个通过可编程的噪声调度矩阵实现可修订的机器人序列生成框架。

**💡 创新点**

创新点在于将噪声层次与重噪事件（ReRoll）结合，使得在扩散去噪过程中可以在不同时间段之间进行交叉修订，保持全程可修订性，并通过虚拟线性噪声块实现局部迭代改进。

**🔧 技术方法**

核心技术是基于扩散模型的 per‑token 噪声调度、线性噪声块训练、重噪触发阈值和双向调度矩阵；实现上使用 Transformer 作为去噪器并在 DDIM 步骤中更新。

**📊 数据集**

使用了 OGBench（PointMaze、AntMaze）、LIBERO‑10、RoboCasa 以及 LIBERO‑90/10 的统一世界模型数据集进行评估。

**📈 对比分析**

在长时域规划、策略学习和统一视频‑动作预测任务中，与全序列扩散、Diffusion Forcing（因果去噪）和 Diffuser 等基线相比，ReRoll 在成功率、OOB 性能、动作‑视频一致性等指标上均取得显著提升（如规划成功率提升 21%/23%，策略成功率提升 56.5%，OOB 动作成功率提升 14% 等）。

**⚠️ 局限性**

主要局限包括：实验仅在仿真环境中进行，缺乏真实机器人验证；需要手工调节调度矩阵（斜率、重噪阈值、重噪次数）；仅在 Transformer 和中等规模数据集上验证，尚未验证在更大规模或不同机器人模型上的通用性。

---

## 258. Long-Term Sequential Decision Making under Risk

**arXiv ID:** 2607.19914 | [PDF](https://arxiv.org/pdf/2607.19914v1)

**作者:** Irmaan `[一作]`, Abdel-Illah Mouaddib `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为 ERQDP 的枚举-与采样无关的算法，用于解决有限时限 MDP 中根级（即整体）风险目标（秩相关函数）下的规划问题。

**💡 创新点**

创新点在于：①构建 K 维秩-分位数近似代理并通过动态规划精确求解；②使用 PMF 动态规划对候选策略进行无采样、无枚举的完整收益分布评估，并给出离散化误差上界；③在自适应提升 K 时仅接受真正提升根目标的策略改进，从而得到可接受的上下界（证书）并支持任意精度的“任意时刻”终止。

**🔧 技术方法**

核心技术包括：动态规划（Backward Induction）在 K‑slice 代理上的执行；粒度化的收益分布 DP（PMF‑DP）与舍入误差分析；秩分区（Quantile Surrogate）与分量权重 Δϕ_k；以及基于误差 δ_K 与 H/(2c) 的证书计算。

**📊 数据集**

实验使用了三类基准：十期投注游戏（BG）、5×5 网格导航（GW）和十期库存控制（IC），全部采用离散状态空间与时间增强表格模型。

**📈 对比分析**

与已有的 CVaR 规划方法（如 Rigter 等）相比，ERQDP 在相同或更高的风险参数下能够得到更低的平均成本或尾部成本，并在多次风险参数扫描中实现数十倍到数百倍的速度提升；同时在 GW 实验中实现了几乎零的最优性间隙（median 0），并在 CVaR、均值与尾部指标上表现出正向差异。

**⚠️ 局限性**

主要局限包括：①需要手动调节秩分辨率 K 与收益网格精度 c，过粗的离散化会导致较大的证书间隙；②算法主要针对离散表格 MDP，扩展到连续或大规模状态空间时可能面临计算量与存储瓶颈；③在极端尾部风险（如 α ≤ 0.05）下，证书间隙可能仍显著，需要更细粒度的代理或更高的计算预算。

---

## 259. JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety

**arXiv ID:** 2607.19913 | [PDF](https://arxiv.org/pdf/2607.19913v1)

**作者:** Yuan Xiong `[一作]` (Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences), Lijun Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了 Janus 框架，训练能够在工具使用代理的长周期执行中提前预测并阻止潜在危险的预测性安全守护模型 Vanguard。

**💡 创新点**

创新点在于通过多智能体仿真生成多样化的长周期轨迹，并使用 Coupled Anticipation and Adjudication Reinforcement Learning (CoAA‑RL) 将未来风险预测与安全判断任务耦合，形成“先知型”安全守护。

**🔧 技术方法**

主要技术包括：多智能体仿真轨迹合成、双任务 RL（预测未来摘要与判定安全状态）、群体相对优势优化（GRPO）以及基于 NLI 的未来摘要一致性评估。

**📊 数据集**

使用了内部合成的 75,180 条轨迹样本以及四个公开代理安全基准（AgentDojo、Agent‑SafetyBench、AgentLAB、LPS‑Bench）进行评估。

**📈 对比分析**

在四个基准上，Vanguard 的攻击成功率 (ASR) 平均降低到 0.071，较传统基线模型平均提高 15.9% 的保护率，同时保持 0.680 的常规任务效用，优于 LlamaFirewall、Sandwich Defense 等框架。

**⚠️ 局限性**

局限性包括：训练轨迹来自仿真而非真实部署，可能无法完全覆盖真实工具行为、环境反馈和用户交互；以及评估仅覆盖固定的模型、工具和基准，未充分验证对未见工具或新攻击方式的泛化能力。

---

## 260. Digital Twin Modeling of a Highly Automated Agricultural Tractor

**arXiv ID:** 2607.19912 | [PDF](https://arxiv.org/pdf/2607.19912v1)

**作者:** Clay Hallman `[一作]`, Timo Oksanen `[通讯]`

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

创建了农业拖拉机AMX G‑trac的数字孪生模型，重点实现了基于ISOBUS协议的CAN通信与操控；

**💡 创新点**

首次将Mevea仿真平台与虚拟CAN结合，用Python脚本实现双轴转向的数字孪生，并在仿真与实车上对侧向与纵向动态进行同频比较；

**🔧 技术方法**

采用Mevea Simulation Software的多体动力学引擎、Kvaser CanKing虚拟CAN通道、Python脚本进行CAN读写、LuGre轮胎模型以及Blender/SolidWorks 3D模型导入；

**📊 数据集**

使用制造商提供的拖拉机规格（重量、轮胎尺寸、传动参数）、Farming Simulator 25的3D模型数据以及实车CAN/GNSS记录数据；

**📈 对比分析**

通过直线行驶、转弯以及最小转弯半径等多种测试，在仿真与实车中对转向角度、转弯半径、发动机与车轮转速等指标进行对比，侧向动态相似度在5‑10%之间，转弯半径约6 m；

**⚠️ 局限性**

局限主要包括：数据传输采用文本文件导致时序误差和转向角峰值；缺乏真实的传动和惯性参数；未实现液压与拖拉机装载机动力学；仅测试了双轴转向，未涵盖完整的纵向动力学。

---

## 261. Defense Against LLM Backdoors using Critical Neuron Isolation Pruning

**arXiv ID:** 2607.19894 | [PDF](https://arxiv.org/pdf/2607.19894v1)

**作者:** Yuxi Li `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个统一的框架，先通过交叉熵最小化自动发现后门触发词，然后定位并剪枝关键神经元（Backdoor Critical Neurons，BCNs），从而消除解码器型大型语言模型中的后门攻击。

**💡 创新点**

创新点在于：①利用触发词在不同位置和不同词形下对模型输出的激活一致性，构造了一种基于激活相似度的触发词检测方法；②提出了BCNs概念并给出了门控MLP层和FFN层的差异激活阈值判定公式，实现了极低比例（仅0.1%）的神经元剪枝即可消除后门；③通过跨模型、跨攻击的系统评估展示了方法在保持97%原始功能的同时将攻击成功率降低95%以上。

**🔧 技术方法**

技术包括：梯度引导的离散搜索（类似GCG）用于触发词挖掘；交叉熵损失度量输出相似度；激活统计与阈值判定用于BCNs识别；按比例权重乘法实现神经元衰减剪枝；多层Transformer的自注意力与门控MLP内部结构分析。

**📊 数据集**

数据集：HarmBench（100类危险指令、712测试样本），通用提示数据集（如Open-Assistant 52k、OpenAI Dataset 18k）、MT‑Bench、HumanEval、AlpacaGPT‑52K，用于训练、评估与基准测试。

**📈 对比分析**

与七种现有后门防御（Pruning、Quantization、Fine‑tuning、CleanGEN、CROW、PURE、grad）在四类攻击（BadEdit、VPI、SleeperAgent、JailbreakEdit）和六个模型（Llama‑2、Qwen、BERT等）上对比。实验显示该方法将攻击成功率平均降至≈2.5%，而基线最低仍在20%以上；在MT‑Bench、HumanEval、Alpaca等通用评测中保持≈97%原模型性能，几乎无功能退化。

**⚠️ 局限性**

局限性：仅针对解码器型Transformer，尚未验证在Encoder‑Decoder或多模态模型上的适用性；检测过程需要较大计算量和访问完整权重；在极小规模模型或极大规模模型上阈值调参可能需要额外工作；并且对触发词的多样性虽然较高，但仍可能漏检极端隐蔽触发。

---

## 262. MTVDiff: Multimodal Conditional Latent Diffusion for Enhanced Thermal-to-Visible Face Translation

**arXiv ID:** 2607.19886 | [PDF](https://arxiv.org/pdf/2607.19886v1)

**作者:** Zhiyuan Xia `[一作]` (Southeast University), Cunjian Chen `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MTVDiff模型，完成热成像到可见光人脸的跨光谱翻译，并融合深度图与文本信息；

**💡 创新点**

设计了Dual-Branch Cross-Attention Fusion、Gated Text-to-Visual Feature Alignment和Spatial Feature Transformations三种机制，实现多模态自适应分阶段融合；

**🔧 技术方法**

基于Latent Diffusion Model与Stable Diffusion骨干，采用双分支ResNet-18编码器、交叉注意力、门控文本对齐以及SFT注入；

**📊 数据集**

在MCXFace和SpeakingFaces两大跨光谱人脸数据集上训练和评估，利用Depth Anything生成深度图、LLaVA生成文本描述；

**📈 对比分析**

与Axial-GAN、T2V-DDPM、DiffTV、DiffV2IR、Uni-ControlNet等基线在FID、LPIPS、PSNR、SSIM以及Rank‑1/VR@1%等指标上对比，MTVDiff在所有指标上均优于对手，FID降幅48.3%、Rank‑1提升8.9%；

**⚠️ 局限性**

依赖LLaVA文本生成，文本错误会影响质量；固定256×256分辨率，扩展到更高分辨率需重构编码器；单模态下性能显著下降，缺乏完善的单模态备选方案。

---

## 263. Stack and Queue Layouts with Defects

**arXiv ID:** 2607.19968 | [PDF](https://arxiv.org/pdf/2607.19968v1)

**作者:** Michael A. Bekos `[一作]` (University of Ioannina), Alexandra Weinberger `[通讯]` (FernUniversität in Hagen)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文引入并研究了允许“缺陷”（即每条边最多与k条其他边交叉或嵌套）的层式堆叠（stack）和队列（queue）布局，给出了不同k值下的边密度界限、所需队列/堆叠数的上界与下界，并证明了某些图类在允许缺陷时可显著降低所需队列/堆叠数；同时提出了相应的识别问题算法与复杂度分析。

**💡 创新点**

创新点在于：①首次将缺陷概念引入层式布局，定义k‑defective stack/queue；②给出了多类图（外k‑可平面、平面、完全图、完全二分图等）的紧确边密度与队列/堆叠数界限；③证明了k‑defective queue number 1与nice arched‑level k‑planar图的等价性，提供了新的识别判据；④给出了k‑defective 1‑stack布局的准线性时间识别算法。

**🔧 技术方法**

主要技术手段包括：图分层与BFS层划分、H‑partition与层宽概念、强直积与树宽分析、Menger与外k‑可平面图的性质、arched‑level布局构造与嵌套与交叉的对应；以及组合数学证明、区间划分与离散数学工具。

**📊 数据集**

无实验数据集；研究完全为理论证明与算法分析。

**📈 对比分析**

通过与传统层式布局（无缺陷）已知的队列/堆叠数上界进行对比。对外可平面图、外1‑可平面图和平面图的队列数分别从3→2、42→33；对完全图与完全二分图的队列/堆叠数则在1‑缺陷下实现了约33%的减少。

**⚠️ 局限性**

局限性主要是：①缺陷k>1时的边密度与队列/堆叠数界限仍存在较大松弛；②平面图在k=1时的队列数与堆叠数上下界差距较大；③识别问题仅在k‑defective 1‑stack已给出准多项式算法，对1‑queue布局的识别复杂度仍未完全确定。

---

## 264. A Multi-Dimensional Evaluation of Explainability in Media Bias Detection

**arXiv ID:** 2607.19954 | [PDF](https://arxiv.org/pdf/2607.19954v1)

**作者:** Ting Chen `[一作]` (Carnegie Mellon University), Sagar Samtani `[通讯]` (Indiana University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并比较基于 BERT 与 RoBERTa 的媒体偏见检测模型的预测性能、解释可行性（与专家标注的合理性对齐）以及机制可信度（通过激活补丁和电路发现实现的因果可恢复性）。

**💡 创新点**

首次将注意力监督微调、对齐标注以及激活补丁电路发现方法统一到编码器型文本分类任务，揭示了可解释性评估的多维度关系，并说明模型规模并非决定机制可压缩性的唯一因素。

**🔧 技术方法**

注意力监督训练、CLS/注意力滚动/梯度×注意力可解释方法、激活补丁电路发现、线性探针、保留/救援指标。

**📊 数据集**

BABE（Bias Annotations By Experts）数据集，约 3,700 条英文新闻句子，平衡 1,381 条有偏见与 1,381 条无偏见句子，并提供专家标注的关键字/短语。

**📈 对比分析**

通过宏 F1、AUROC（与专家标注的对齐）以及 Retention/Rescue 指标对比标准微调与注意力监督微调的 BERT/base/large、RoBERTa/base/large 方案；宏 F1 均超过 0.80，RoBERTa‑large 在注意力监督下达到最高 0.863；AUROC 仅略高于随机（0.5‑0.6）；保留/救援随电路规模增加提升，表明偏见预测由多头协同实现。

**⚠️ 局限性**

局限性包括：BABE 样本量小且主题单一，实验模型仅涵盖编码器型 BERT/RoBERTa，激活补丁电路发现为近似方法，且不同架构（如解码器、指令调优模型）可能表现不同。

---

## 265. A Framework of User Experience Principles for Human-AI Agent Interaction in the Workplace

**arXiv ID:** 2607.19941 | [PDF](https://arxiv.org/pdf/2607.19941v1)

**作者:** Kathrin Paimann `[一作]` (SAP SE), Sebastian Juhl `[通讯]` (SAP SE)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在企业背景下系统挖掘、构建并验证了人机 AI 代理交互的八大 UX 原则及其可测量的标准，采用参与式工作坊、专家评审、元分析、纸笔调查和深度访谈等多方法研究。

**💡 创新点**

首次在实践中以实证数据为依据提出并验证了可操作的八项 UX 原则，为企业设计可信、人中心的 AI 代理提供了可量化的设计指南；整合了多学科方法，强调人类控制、透明度、可靠性与情境感知等关键维度。

**🔧 技术方法**

采用多方法研究技术：参与式设计工作坊、专家评审、文献元分析、纸笔问卷调查以及半结构化深度访谈；质性归纳与定量排名相结合。

**📊 数据集**

数据来源包括：28 篇学术论文（Bradshaw 等、Diederich 等、NIST 框架、Shneiderman、Xu 等）以及企业内部专家与参与者的访谈与问卷数据（共 21+5+22+12 人）。

**📈 对比分析**

通过纸笔调查对八原则进行排名与重要性评估，统计各原则子标准的占比（如控制原则 65% 位于 Top3，可靠性 60% 等）。实验表明人类控制与可靠性为最高优先级，但在真实企业部署环境中的性能尚未进一步评估。

**⚠️ 局限性**

局限性：原则间存在重叠且非互斥；参与者为自选样本，规模有限，缺乏多样性；强制优先排序可能压制等价重要性；验证仅在受控研究场景进行，缺乏真实运营环境的纵向验证。

---

## 266. Spatial Semantic Communication: When Semantic Transmission Meets Index Modulation

**arXiv ID:** 2607.19934 | [PDF](https://arxiv.org/pdf/2607.19934v1)

**作者:** Xinghao Guo `[一作]` (Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个结合语义通信与流体天线指数调制的空间语义通信系统SSC，并提出了残差量化与语义感知流拆分方案。

**💡 创新点**

创新点在于：1）将残差量化（RQ）与FA-IM耦合，实现语义特征的分层离散化；2）提出语义感知流拆分，将重要信息映射到更可靠的调制维度；3）采用三阶段训练策略，显著提升系统稳定性与性能。

**🔧 技术方法**

使用技术包括残差量化（RQ）、流体天线指数调制（FA-IM）、Swin Transformer 结构的联合源信道编码（JSCC）、二进制对称信道（BSC）仿真、指数移动平均（EMA）、直通估计（STE）以及三阶段训练策略。

**📊 数据集**

训练集使用 DIV2K 图像数据集，测试集采用 Kodak24 图像集。

**📈 对比分析**

与 sDAC、SSC 无 IM、SSC 无 SS、BPG+LDPC、MOC-RVQ、ESC-MVQ 等基线在不同压缩率、SNR 条件下进行 PSNR 与 MS-SSIM 对比。SSC 在低至中等 SNR 范围内显著优于所有基线，且在多种配置下保持了更高的重建质量。

**⚠️ 局限性**

局限性包括：对流体天线切换延迟和硬件实现的依赖；流拆分方案需离线预计算，受限于固定系统参数；对通道状态信息（CSI）误估较为敏感，尤其接收端误估；在极大码本尺寸下可能出现维度灾难，影响训练收敛。

---

## 267. Towards Ultra-High Reliability in Wi-Fi 8: IEEE 802.11bn Core Mechanisms, mmWave Integration, and Performance Verification

**arXiv ID:** 2607.19931 | [PDF](https://arxiv.org/pdf/2607.19931v1)

**作者:** Xiaoqian Liu `[一作]` (Chinese University of Hong Kong), Jian Song `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统性综述了 Wi‑Fi 8（IEEE 802.11bn）的关键 PHY/MAC 机制，并通过仿真验证了其在吞吐量、延迟与包丢失等指标上的改进。

**💡 创新点**

创新点包括：① 引入 Ultra‑High Reliability (UHR) 目标及其对应的多 AP 协调 (MAPC)、分布式资源单元 (DRU)、增强长距离 (ELR)、非主通道访问 (NPCA) 与优先 EDCA (P‑EDCA) 等；② 将毫米波（60 GHz）与现有 sub‑7 GHz 频段通过 MLO 统一管理的 IMMW 方案；③ 通过系统层仿真量化各技术在不同拥塞与干扰场景下的性能提升。

**🔧 技术方法**

采用的技术包括：OFDM/OFDMA、MIMO/ MU‑MIMO、分布式/集中式多链路管理、频谱扩展与功率分布、干扰抑制 (IM)、跨频段控制与数据分离、以及毫米波信号的上采样与子载波间距调整。

**📊 数据集**

使用的“数据集”为仿真场景：多 BSS/AP 组成的无线网络，参数包括信道宽度、MCS、功率、帧大小、流数等；不涉及真实测量数据，而是基于 IEEE 802.11 标准定义的通道模型（如 802.11ad 会议室模型）。

**📈 对比分析**

比较方法：将新技术（NPCA、P‑EDCA、DUO、IM、IMMW 等）与传统 EHT（IEEE 802.11be）基线在同一仿真环境下对比。性能表现：吞吐量提升可达 25–40 %；95 % 分位延迟下降 25–70 %；包丢失率下降 25–35 %；毫米波方案在硬件失真下仍保持 SNR 增益 1–2 dB。

**⚠️ 局限性**

限制与挑战：① 实际部署需要精细的 AP 协同与时序同步，易受动态环境影响；② NPCA 与 P‑EDCA 在高负载下会产生信令开销，可能导致负面效应；③ IM 需要多天线并具备空时分配，硬件成本上升；④ IMMW 方案对硬件失真（CFO、PN、PA 非线性）敏感，需要复杂补偿；⑤ 综上，虽然仿真显示显著改进，但实际实现仍需进一步验证与标准细化。

---

## 268. TargetFinder: Detecting Widgets from Pixels on Desktop Interfaces

**arXiv ID:** 2607.19907 | [PDF](https://arxiv.org/pdf/2607.19907v1)

**作者:** Ahmed Ben Akouche `[一作]` (Sorbonne Université), Julien Gori `[通讯]` (Sorbonne Université)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套名为TargetFinder的跨平台实时计算机视觉系统，用于检测桌面GUI中的各种控件。

**💡 创新点**

创新在于提供了首个跨操作系统、覆盖多种界面（Windows, macOS, Ubuntu, web）的手工标注Widget数据集，并通过YOLO模型实现毫秒级检测，从而实现系统级的目标感知交互技术。

**🔧 技术方法**

主要技术为YOLO系列目标检测网络（YOLOv8、YOLO26n、YOLO11m等），结合低分辨率变化检测、快速截图与多线程推理。

**📊 数据集**

使用了520张手工标注的桌面截图（约38,000个Widget）以及71张旧版软件截图作为测试，涵盖四大平台。

**📈 对比分析**

与OmniParser、REMAUI、UIED、MobileSAM等基线对比，TargetFinder在单类mAP@0.5达0.899、mIoU 0.872，精度/召回分别0.936/0.840，且推理时间低于200ms，明显优于现有方法。

**⚠️ 局限性**

局限在于只标注了6类高层次控件，无法识别层级关系或窗口，且对移动端和极端主题的泛化仍有限。

---

## 269. What Matters in Humanoid General Motion Tracking? An Empirical Study

**arXiv ID:** 2607.19903 | [PDF](https://arxiv.org/pdf/2607.19903v1)

**作者:** Fabio Amadio `[一作]` (Inria), Enrico Mingo Hoffman `[通讯]` (Inria)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对人形机器人Unitree G1，系统性评估并比较了一系列常用的运动跟踪模型与训练设计，包括运动指令表示、观测历史、动作表征、驱动配置、手部力随机化及教师-学生训练方法，构建了可复现的开源框架YAHMP并实现了零射线(sim‑to‑real)部署；

**💡 创新点**

通过同一训练管线逐一变更设计选项，首次揭示各因素对跟踪精度、力学表现、学习难度及实机表现的具体影响，并提供了与TWIST2基线的统一评估；

**🔧 技术方法**

采用基于PPO的强化学习，配合可配置的观测与动作编码、时序卷积历史网络、PD驱动配置、手部力随机化与教师-学生策略，整合MuJoCo仿真与Unitree G1硬件；

**📊 数据集**

使用AMASS与OMOMO的12,175条人类运动捕捉数据（11,151训练，1,024测试），进行重映射到G1模型后进行训练与评估；

**📈 对比分析**

在1,024条测试动作上，所提出的基线策略在跟踪误差上与TWIST2相比整体更优（基准误差约30%减小），且在硬件上实现零射线部署、保持平衡、承受外力和执行全身操控；

**⚠️ 局限性**

实验仅覆盖单一机器人平台，设计选择间相互作用未探究，且在更复杂地形或其他机器人上迁移性待验证；

---

## 270. Odin: Primitive-Level Synchronization for Distributed Point-Based Neural Rendering

**arXiv ID:** 2607.19893 | [PDF](https://arxiv.org/pdf/2607.19893v1)

**作者:** Zhenxiang Ma `[一作]` (Shanghai Jiao Tong University), Hengjie Li `[通讯]` (Shanghai AI Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在分布式点基神经渲染（PBNR）中实现原子级同步，拆除全局迭代阻塞，显著提升训练吞吐量

**💡 创新点**

提出 Odin 系统，将全局 barrier 拆分为基于原子索引的同步，并结合静态局部性预测、运行时验证和 Shadow Graph 虚拟化状态的技术，开启了原子级同步与吞吐优化的新范式

**🔧 技术方法**

使用 SfM 轨迹相似度构建相对局部性图、静态数据调度与静态异步调度、动态数据调度与动态异步调度、Shadow Graph 虚拟化以及原子范围验证等技术实现原子级同步

**📊 数据集**

评估数据集包括 MipNeRF360、Tanks & Temples、DeepBlending 等非城市场景以及大规模城市场景 MatrixCity

**📈 对比分析**

与 PyTorch DDP（DP 方案）和 Grendel（MP 方案）对比，在 8‑GPU 非城市场景平均提升 1.22×，在 64‑GPU 城市场景提升 1.89×，且保持重建质量在 ±1% 误差范围内

**⚠️ 局限性**

对稠密场景或缺失原子范围/梯度信息时退化为传统同步；对结构性变更、全局正则化等情况保持保守同步，导致无法进一步压缩等待

---

## 271. EA-Nav: Learning Safe Visual Navigation Policies with Embodiment Awareness

**arXiv ID:** 2607.19880 | [PDF](https://arxiv.org/pdf/2607.19880v1)

**作者:** Jialu Zhang `[一作]` (Zhejiang University), Jituo Li `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于模仿学习的跨体态视觉导航框架，利用体态几何信息提高导航安全性和适应性。

**💡 创新点**

创新点在于：①在预训练阶段引入体态几何作为条件标记，消除同一视觉观测下的动作歧义；②在微调阶段通过风险轨迹增广和分离的空间感知与风险修正模块，显式利用体态几何实现安全轨迹校正。

**🔧 技术方法**

采用跨模态输入编码器、Transformer注意力机制、行为克隆、空间感知网络、风险修正模块，以及风险轨迹增广与占据网格碰撞检测等技术。

**📊 数据集**

使用了从互联网上收集的约1,000小时跨体态第一人称视频构建的异构导航数据集，结合Depth‑Anything‑3生成的伪深度标签，并在室内外真实与合成环境（如i2Nav、NavDP、InternUtopia）进行微调与评估。

**📈 对比分析**

与现有基于强化学习或仅视觉的导航方法（如iPlanner、NavDP、NoMaD、ExAug）进行对比，实验显示在仿真与真实机器人（TurtleBot、Unitree Go2）上成功率提升约31%，碰撞率显著降低，整体性能优于对比基线。

**⚠️ 局限性**

局限性在于未充分考虑转弯半径、运动约束等更细粒度的体态属性，限制了在具有不同运动模式的体态上的广泛适用性。

---

## 272. Robust Activation Map Rectification for Weakly Supervised Volumetric Segmentation: Temporal Coherence as a Free Lunch

**arXiv ID:** 2607.19877 | [PDF](https://arxiv.org/pdf/2607.19877v1)

**作者:** Renshu Gu `[一作]` (Hangzhou Dianzi University), Gang Xu `[通讯]` (Hangzhou Dianzi University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了CSSeg，一种无需额外训练、通过利用时空一致性对噪声CAM进行修正的弱监督体素分割框架。

**💡 创新点**

创新点在于引入变分化激活聚合（VRAA）和双向极值校正（BER）两大训练无关模块，利用高维随机矩阵理论显著降低激活噪声并纠正异常激活。

**🔧 技术方法**

使用ViT+Grad‑CAM生成初始激活，VRAA对多切片做均值聚合，BER通过极值一致性检查并双向替换异常帧，最终将修正后的掩码作为MedSAM的框选提示。

**📊 数据集**

在BraTS、CHAOS、MSD（Brain、Prostate、Cardiac）等医学3D数据集上进行实验评估。

**📈 对比分析**

与CG‑Diff、CG‑CDM、S2C、IPSeg等弱监督/训练自由方法以及Grad‑CAM、SAM3等基准模型对比，BraTS上Dice 0.633、mIoU 0.510、HD95 18.7；CHAOS上Dice 0.375、mIoU 0.261；MSD多任务亦优于现有方法；推理速度显著加快（0.393s/图像）。

**⚠️ 局限性**

对极小目标或CAM识别不清时易失效；当目标尺寸过小或CAM无法覆盖时，VRAA与BER难以恢复准确分割。

---

## 273. EvoThink: Evolving Thinking in Large Reasoning Models via Self-Pruning and Aha-Moment Preference Optimization

**arXiv ID:** 2607.19962 | [PDF](https://arxiv.org/pdf/2607.19962v1)

**作者:** Xinbang Dai `[一作]` (Southeast University), Yuyang Zhang `[通讯]` (Noah's Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 EvoThink 框架，通过自我修剪训练（SPT）和 aha-瞬间偏好优化（AMPO）两阶段方法，减少大型推理模型的冗余思考并提升推理能力。

**💡 创新点**

创新地在不依赖人工标签的情况下，用自我修剪迭代提炼高质量简洁推理轨迹，并通过遗传算法选取多样化错误轨迹生成从错误到正确的 aha-瞬间示例，实现对模型推理模式的自适应优化。

**🔧 技术方法**

采用自监督的 Next‑Token 损失进行 SPT，基于遗传算法和多样性度量构造 aha-瞬间数据，使用直接偏好优化（DPO）训练模型，结合原始大型语言模型。

**📊 数据集**

在数学推理领域使用 MATH‑500、AIME24、AIME25；在代码生成领域使用 TACO；并在不同模型规模（Distill‑Qwen‑1.5B、DeepScaleR‑1.5B、QwQ‑32B）上进行训练与评估。

**📈 对比分析**

与传统压缩方法（Kimi、O1‑Pruner、ThinkPrune、DIET）及 SFT/DPO 对比，EvoThink 在 Pass@1 上保持甚至提升（如 DeepScaleR‑1.5B 在 MATH‑500 由 75.4% 提升至 76.3%），同时平均 token 数显著下降（从约 3171 降至约 2146），证明在保持效率的同时提升推理准确率。

**⚠️ 局限性**

方法仍需依赖大量自动生成的修剪轨迹，且在极端难题或跨领域迁移时性能提升有限；遗传算法和多样性评估对超参数敏感，且未探究从错误到正确学习机制的理论基础。

---

## 274. When Does Knowledge Distillation Hurt? Reliability-Aware Distillation for Low-Resource Language Summarization

**arXiv ID:** 2607.19956 | [PDF](https://arxiv.org/pdf/2607.19956v1)

**作者:** Dipto Sumit `[一作]` (BRAC University), Farig Sadeque `[通讯]` (BRAC University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了两种可靠性感知的知识蒸馏方法——CHAD和EWAD+CPDP，用于在低资源序列到序列任务中筛选可信的教师信号。

**💡 创新点**

创新点在于直接衡量样本或标记的蒸馏有效性（梯度对齐）并通过轻量化门控或跨词表几何约束实现自适应权重，而非传统的基于教师置信度或表面指标的启发式门控。

**🔧 技术方法**

技术主要包括梯度对齐评估、梯度提升树门控、token级置信度门控、第二教师的向量空间几何约束（CPDP）以及多教师蒸馏与KL/温度缩放的联合损失。

**📊 数据集**

使用的主要数据集是Bangla新闻摘要BenchMark BanSum（141K样本）以及多语言XL‑Sum（45种语言，约1.5K样本/语种）进行跨语言验证。

**📈 对比分析**

与传统的统一KD、无KD以及基于教师置信度或ROUGE的门控方法相比，CHAD在BanSum上ROUGE‑L提升约0.02，EWAD+CPDP提升约0.07；两者均在60M参数学生上超过50B参数的Qwen‑2.5‑3B微调模型。

**⚠️ 局限性**

局限性包括：仅在教师与学生容量差距较大时效果显著；CHAD在多语言场景未验证；门控精度有限且需要额外的探针成本；CPDP依赖第二教师且对词表不兼容的选择不一定最优。

---

## 275. Efficient Chain-of-Modality Reasoning via Progressive Compression for Spoken Language Models

**arXiv ID:** 2607.19932 | [PDF](https://arxiv.org/pdf/2607.19932v1)

**作者:** Pengchao Feng `[一作]` (Shanghai Jiao Tong University Shanghai Innovation Institute), Xie Chen `[通讯]` (Shanghai Jiao Tong University Shanghai Innovation Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ECoM Reasoning 框架与 Progressive Compression 训练策略，实现语音模型中压缩式文本推理，提升推理准确率与效率。

**💡 创新点**

创新点在于将中间文本压缩为既可指导语音生成又能承载核心推理的稠密表示，并通过分阶段训练逐步实现从完整推理到压缩推理的迁移。

**🔧 技术方法**

技术包括 Chain‑of‑Modality (CoM) 结构、LLMLingua‑2 词级重要性评分、SLAM‑LLM 平台、Whisper 编码器、CosyVoice3 解码器、Qwen2.5‑1.5B 语言模型，以及自定义的压缩与课程学习机制。

**📊 数据集**

使用的主要数据集包括通用对话语音合成（MagpiePro、InfGen）、数学推理数据（GSM8K、NuminaMath、MathQA、AddSub、SingleEq、MultiArith、SVAMP）以及语音问答基准（UltraEval‑Audio、LLaMA Questions 等）。

**📈 对比分析**

在数学问答基准上与 Cascade、标准 CoM、CoM Reasoning 进行对比，ECoM Reasoning 在准确率上提升约 21% 以上，且在使用 40% 文本 token 的情况下实现最高的 token‑efficiency；在非推理语音问答任务中保持性能并提高 token‑efficiency。

**⚠️ 局限性**

局限性包括：首个语音 token 延迟仍高、未验证在更大模型上的可扩展性、仅在英语上实验、未探索多语言或跨语言推理、压缩助手文本仍难做到无损压缩。

---

## 276. Scalable Keyword Spotting via Modular Network Expansion

**arXiv ID:** 2607.19918 | [PDF](https://arxiv.org/pdf/2607.19918v1)

**作者:** Viktor Khaymonenko `[一作]` (Yandex), Alexander Rostov `[通讯]` (Yandex)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种模块化扩展机制，在不触碰已部署的 KWS 模型（包括 BN 统计量）的前提下，通过附加轻量级扩展分支和专用新关键词分类头，实现嵌入式关键词检测器的新关键词增量更新。

**💡 创新点**

创新点在于：① 完全冻结原始网络，确保对已有关键词的 logits 与阈值不变，严格保证无退化；② 采用参数可控（≤10k）的模块化分支，通过“核心优先”决策规则实现高效推理；③ 在同等参数预算下，显著提升新关键词的检测准确率，并降低算力开销。

**🔧 技术方法**

技术包括：SVDF 核心网络、批归一化、hard‑swish、dropout；扩展分支使用 1D 卷积、BN、hard‑swish；核心优先决策规则；与全微调、EWC、head‑only、ensemble、adapters、LoRA 等方法对比实验。

**📊 数据集**

使用 Google Speech Commands v2 作为正样本语料，Common Voice v17 作为负样本（FAR 校准）来评估模型性能。

**📈 对比分析**

与参数匹配的基线（ensemble、adapters、LoRA）以及全微调、EWC 进行比较。在≤10k 参数预算下，模块化扩展将平均新关键词 FRR 从 6.46% 降至 4.37%，优于 adapters（8.05%）、LoRA（6.41%）和 ensemble（6.46%）；同时 MAC 计数为 16.34M，低于 adapters（18.45M）和 LoRA（20.52M）。

**⚠️ 局限性**

局限性：仅评估单步关键词扩增，未探讨多步递增更新；实验仅在 GSC 命令集上验证，可能对其他语言或极具歧义的新关键词泛化能力未知；模型大小随着多次扩增会累积增长；需手动阈值校准，影响实际部署灵活性。

---

## 277. LoRFT: Benchmarking Long-Range Vehicle Trajectory Reconstruction from Fixed Highway Cameras

**arXiv ID:** 2607.19911 | [PDF](https://arxiv.org/pdf/2607.19911v1)

**作者:** Yufan Zhu `[一作]` (Changsha University of Science and Technology), Zixuan Xiao `[通讯]` (Changsha University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建LoRFT基准并提出Map‑RSTNet，实现固定高速摄像机视角下的近端轨迹到远端轨迹的重建。

**💡 创新点**

创新点在于：①提供手工验证的近端轨迹与远端参考对，构成真实的长程轨迹重建任务；②Map‑RSTNet在道路几何对齐状态空间中使用残差序列解码并动态刷新几何，显著降低视角压缩带来的误差。

**🔧 技术方法**

使用YOLOv11+ByteTrack进行检测/跟踪，Map‑RSTNet基于LSTM Seq2Seq、道路几何映射、残差递归以及动态几何刷新技术。

**📊 数据集**

使用LoRFT数据集：22条高速公路场景、366,109帧、6,601条车辆轨迹、2,694,889个边框。

**📈 对比分析**

与CS‑LSTM、GRIP++、DeepTrack、MixNet、GNP等预测/轨迹模型对比，Map‑RSTNet在ADE、FDE和RMSE@5s上分别提升约11%、15%和10%。

**⚠️ 局限性**

局限性包括：仅适用于单摄像机高速公路场景；缺乏相机标定，无法直接获得速度/加速度等度量信息；难以推广到多摄像机或城市道路等复杂环境。

---

## 278. MV-Bench: Benchmarking Multimodal Large Language Models for Coordinated Multi-View Interface Construction

**arXiv ID:** 2607.19910 | [PDF](https://arxiv.org/pdf/2607.19910v1)

**作者:** Yue Zhao `[一作]` (Shandong Second Medical University), Qiong Zeng `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个名为 MV-Bench 的基准，包含 92 个基准接口和 956 个重组样本，能够在给定截图、数据和交互规范的条件下评估多模态大语言模型（MLLMs）在协调多视图界面代码生成的能力。

**💡 创新点**

创新点包括：①首次提出界面级 image‑to‑code 评估任务并提供完整的数据绑定与交互逻辑；②设计两层中间表示（TSS 与 CTS）实现从 Tableau 工作簿到可执行 Web 代码的语义桥接；③通过自动重组策略扩大样本多样性并保持交互语义；④制定可自动验证的执行、构建、渲染三阶段评估和多维度指标（视觉、数据绑定、交互完整性）。

**🔧 技术方法**

技术手段主要有：多阶段流水线（解析、规范化、代码生成与验证）；使用 LLM 作为编码代理（GLM、Kimi、Qwen、Claude、GPT）完成一次性代码生成；利用 React/TypeScript/D3 的单页架构；自动化的 sandbox 环境（Node.js + Puppeteer + Cypress）进行可执行性与交互回放；以及基于 CLIP/SSIM/TreeBLEU 等指标的视觉相似度评估。

**📊 数据集**

数据集：来源于公开的 DMiner Tableau 仓库，经过过滤后获得 157 个自包含的工作簿，进一步构建 92 个基准接口；通过系统重组得到 956 个扩展样本，覆盖 5 种图表类型和 3 种交互模式，配套数据文件、截图和交互注释。

**📈 对比分析**

比较方法：在单次生成（无迭代）和最多三轮修复（基于生成器反馈）的设置下，分别评估 5 个 MLLM 的可执行性、视觉相似度、数据绑定正确率和交互完整性。结果显示，最佳模型 Kimi 在整体得分上达 30.9%，但所有模型的交互完整性仅 0–11.7%，显著低于视觉分数；修复轮数提升可执行率，但对数据绑定和交互的提升有限。

**⚠️ 局限性**

局限性：①仅基于 Tableau 设计，缺乏其他可视化生态的覆盖；②固定的 React/TS/D3 目标栈，未探究声明式语言（如 Vega‑Lite）的易用性；③评估维度未覆盖更复杂交互（层级、地理、钻取等）和多步分析工作流；④潜在的数据泄漏风险，评估仅在单次推理下进行；⑤缺乏对模型训练分布与交互推理能力的深入对齐分析。

---

## 279. Duality and Reverse Self-Dual Constructions for Hyperderivative Reed-Solomon Codes

**arXiv ID:** 2607.19905 | [PDF](https://arxiv.org/pdf/2607.19905v1)

**作者:** Hongchang Li `[一作]` (Shandong University), Sihuang Hu `[通讯]` (Shandong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了超导数Reed-Solomon编码的对偶性问题，提出了其欧几里得对偶的显式分量表示，并探讨了反自对偶HRS编码的构造。

**💡 创新点**

提出了超导数Reed-Solomon编码的对偶结构，发现其对偶具有反序超导数评估结构，并且在低重数情况下，行反转的对偶又是一个HRS编码。

**🔧 技术方法**

使用了残差理论和超导数的技术，结合了代数编码理论中的NRT度量。

**📊 数据集**

使用了有限域上的多项式和超导数构造的HRS编码，具体数据集未明确提及。

**📈 对比分析**

通过与已有的HRS编码进行比较，发现其对偶一般不是HRS编码，而是与反序HRS评估编码在局部上等价，性能在特定情况下（如低重数）表现良好。

**⚠️ 局限性**

对偶性研究中发现的限制是，HRS编码的欧几里得对偶不一定是HRS编码，且在某些情况下需要特定的乘数对称条件。

---

## 280. StrokeSeg2: Stroke Lesion Segmentation in Clinical Research Workflows

**arXiv ID:** 2607.19901 | [PDF](https://arxiv.org/pdf/2607.19901v1)

**作者:** Youwan Mahé `[一作]` (Univ Rennes), Francesca Galassi `[通讯]` (Univ Rennes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出 StrokeSeg2，基于 nnU-Net 的轻量化、跨平台 C++/Qt 框架，支持单机部署的脑卒中病灶分割。

**💡 创新点**

创新点在于将 3D 复杂模型通过知识蒸馏压缩至 0.84M 参数，并结合 ONNX Runtime + 量化实现零依赖、可安装化，显著降低能耗与推理时延。

**🔧 技术方法**

采用知识蒸馏、FP16 量化、ONNX Runtime 推理、BIDS 预处理及 C++/Qt 可视化界面，兼容 CPU、集成 GPU 与独立 GPU。

**📊 数据集**

使用公开 ATLAS v2.0 训练集（655 例）和 ATLAS v2.1 验证集（300 例）进行评估。

**📈 对比分析**

通过 Dice、lesion‑wise F1、ASD 等指标与原始 102M 模型对比，蒸馏后模型保持统计等价；在 CPU 上推理时间从 22.8 s 缩短至 1.5 s，能耗下降 95.8%，总体推理时延降低 84%。

**⚠️ 局限性**

局限性包括对极小病灶的分割精度略低、NPU 加速受限、模型仅针对 T1w/FLAIR 的单模态，未来需验证多模态与临床集成。

---

## 281. OSVE: One Step Video Editing with One Step Diffusion Models

**arXiv ID:** 2607.19895 | [PDF](https://arxiv.org/pdf/2607.19895v1)

**作者:** Habin Lim `[一作]` (Korea University), Gyeong-Moon Park `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 OSVE，一种全程一次性逆向编码与一次性生成的视频编辑框架，实现了极高的实时编辑速度。

**💡 创新点**

核心创新包括：①可学习的结构感知编码器与 SAE 损失，解决单步控制下的结构崩溃；②统一帧编辑 UFE，利用跨帧注意力保证时间一致性；③滑动窗口加锚帧机制，支持长视频的全局一致性。

**🔧 技术方法**

采用的一步扩散模型（DMD2）与 Prompt Perturbation 生成的结构对齐图像对；训练编码器使用结构感知编辑损失；推理阶段使用统一帧拼接、滑动窗口+锚帧，并结合 VAE 编码/解码、CLIP/DINO 评估。

**📊 数据集**

数据来源于 LAION/JourneyDB 等公开文本，使用 Prompt Perturbation 生成结构对齐图像对；评估集为 60 条公开视频（51 条短视频 20 帧，9 条长视频 90 帧），每条视频配 5–2 条编辑提示。

**📈 对比分析**

与多步编辑基线（FLATTEN、TokenFlow、FRESCO、RAVE、COVE）以及多步+单步混合基线进行对比；在 VBench 指标（SC、BC、TF、MS、AQ、IQ）上 OSVE 获得最高 BQS，速度提升 155–171 倍，编辑质量与最优多步方法持平或更优。

**⚠️ 局限性**

局限性包括：目前仍基于一维 T2I 一步模型，无法直接应用到现阶段一维 T2V；单步控制下对复杂结构或场景切换的适应性有限；多锚帧方案增加复杂度；长视频仍需滑动窗口，边界帧可能出现微小一致性问题。

---

## 282. Revisiting Hardware Priority Queue Architectures

**arXiv ID:** 2607.19881 | [PDF](https://arxiv.org/pdf/2607.19881v1)

**作者:** Qihang Wu `[一作]` (New York University), Austin Rovinski `[通讯]` (New York University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在现代FPGA平台上实现并系统评估了多种硬件优先队列架构，并将实现开放为可复用的库。

**💡 创新点**

首次将传统与现代设计在同一硬件上进行对比，揭示了 enqueue 支持、流水线化、寄存器与 BRAM 存储对性能、资源与可扩展性的综合影响，并给出针对不同队列规模的最佳选择建议。

**🔧 技术方法**

使用 Verilog RTL、Vivado 综合工具、Artix UltraScale+ FPGA 以及标准化接口和参数化实现。

**📊 数据集**

通过对不同队列尺寸（数十到数千条目）进行综合实验，收集时钟频率、吞吐率、LUT/FF/BRAM 使用量等指标。

**📈 对比分析**

采用统一接口和同一平台的测量方法，对频率、吞吐、资源占用进行量化比较，结果表明：系统数组在小至中型队列下性能最高，混合树在大队列时性能与面积更优；寄存器阵列/树在极小队列下资源最节省；流水线化提升频率但整体吞吐下降。

**⚠️ 局限性**

实验仅限于单一 FPGA 平台，未覆盖真实应用工作负载；部分架构（如 BRAM 树）缺少 enqueue 支持，且在更大规模上可能出现瓶颈；未来需跨平台、跨工作负载的进一步验证。

---

## 283. Evaluating and Mitigating Gender Bias in Pre-trained Embeddings for ML-based Recruitment

**arXiv ID:** 2607.20073 | [PDF](https://arxiv.org/pdf/2607.20073v1)

**作者:** Farnaz Faramarzi Lighvan `[一作]` (Vrije Universiteit Brussel), Lynn Houthuys `[通讯]` (Vrije Universiteit Brussel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估9种预训练文本嵌入模型在ML招聘评分中的性别偏差，比较单任务与多任务对抗学习（梯度反转）以及文本脱敏（gender‑scrubbing）的效果。

**💡 创新点**

①将Pareto‑front多目标优化引入模型选择，实现预测准确率与公平性之间的显式权衡；②在单任务和多任务框架下系统比较文本脱敏与对抗训练的互补性；③在同一合成数据集上对九种嵌入模型进行统一评估。

**🔧 技术方法**

使用fastText、GloVe、BERT、RoBERTa、MPNet、sentence‑Transformer、OpenAI embeddings；多任务神经网络+梯度反转（GRL）；Pareto‑front多目标优化；Optuna 进行超参搜索；计算MAE、KL、p‑percent、Recall@100 等公平性指标。

**📊 数据集**

FairCVdb（合成简历数据集）——包含24,000份简历，结构化特征、原始与脱敏生物文本，以及性别偏置与无偏置评分。

**📈 对比分析**

通过在训练集上使用Pareto‑front选择最佳超参组合，在测试集上计算MAE、KL、p‑percent、Recall@100 等指标。结果显示：文本脱敏能更显著降低性别泄露和提升公平性；对抗学习在原始文本上可改善公平性但在脱敏文本上效果不一；不同嵌入模型在准确率和公平性之间呈现不同平衡，表明嵌入选择至关重要。

**⚠️ 局限性**

仅使用合成数据，缺乏真实CV的噪声和多样性；只关注二元性别偏差，未评估族裔或交叉属性；对抗训练的参数空间大，可能需要更深入的调优；合成数据的偏差与真实招聘场景可能存在差异。

---

## 284. "You should see my partners' fingers": A Qualitative Study of Construction Artisans' Perspective on Technical Innovation

**arXiv ID:** 2607.20004 | [PDF](https://arxiv.org/pdf/2607.20004v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 285. Are Attributions of Consciousness to AI Chatbots Epistemically Innocent?

**arXiv ID:** 2607.20001 | [PDF](https://arxiv.org/pdf/2607.20001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 286. Reading and Steering Representations of Materials-Science Mechanisms in an Open-Weight Language Model

**arXiv ID:** 2607.20058 | [PDF](https://arxiv.org/pdf/2607.20058v1)

**作者:** Markus J. Buehler `[一作]` `[通讯]` (Massachusetts Institute of Technology), Markus J. Buehler (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在开放权重的大型语言模型Gemma‑4上，对材料科学问题的内部表征与使用进行系统评估，利用可读词（直接解码与Jacobian线性读出）、关系几何（向量匹配与图邻接）、匹配变换（相同定律正逆比较）和因果干预（状态补丁与向量控制）等多种解释技术。

**💡 创新点**

提出统一的可解释框架：在词表可读性、几何相似性、匹配变换和因果干预四个维度同步检验模型是否真正内部编码物理机制；通过匹配反事实与梯度传递验证物理定律方向性，并将结果与直接读出对比，探究Jacobian优势与否。

**🔧 技术方法**

技术手段包括：直接解码器（Direct Unembedding）、Jacobian线性读出（计算梯度映射）、目标自由词列表提取、向量相邻层匹配、状态变换比较、图邻接与图同构分析、对抗性状态补丁、以及有限差分的匹配比较等。

**📊 数据集**

数据集为50条保留描述（10种机制族 × 5种表述），每条对应一组隐藏向量；此外使用60条物理定律（20直、20逆、20中性）以及24个三元组（锚/词面/对照）等进行实验；材料案例来自公开材料科学文献与实验记录。

**📈 对比分析**

方法对比主要体现在可读词（Jacobian vs Direct）与匹配变换（同向vs逆向）以及因果干预（控制向量对答案影响）上。结果显示：Jacobian在部分机制族（如边界攻击、缺口耐性）可读性有显著提升，但整体未能显著优于直接；匹配变换能够区分直、逆定律（AUC 1.0）；因果干预在腐蚀案例显著成功，其他机制普遍不通过，说明可解释性因机制而异。

**⚠️ 局限性**

局限性包括：Jacobian优势仅局部且机制相关；图结构未能唯一识别物理极性；大多数因果干预效果弱，无法在所有机制族泛化；状态几何受输入词语影响，难以分离物理与语言表征；缺乏对完整推理链条的可解释性。

---

## 287. SequenceFI: Non-intrusive Temporal Fault Injection for Microservice Systems

**arXiv ID:** 2607.20050 | [PDF](https://arxiv.org/pdf/2607.20050v1)

**作者:** Yuzhen Tan `[一作]` (Wuhan University), Shaolin Tan `[通讯]` (Zhongguancun Laboratory)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 SequenceFI，一个非侵入式的微服务时序故障注入框架，能够在请求生命周期内精准触发故障。

**💡 创新点**

创新点在于通过 sidecar 捕获消息层 send/receive 事件，传播紧凑的时序证据，利用从跟踪自动合成的出现计数 guards，实现仅在满足时序约束时注入故障，且不需要改动应用代码或序列化库。

**🔧 技术方法**

技术包括 Kubernetes sidecar 代理、W3C Trace Context、OpenTelemetry、最小命中集算法用于 guard 合成、基于请求线索的静态目标搜索。

**📊 数据集**

使用四个公开的微服务基准：Online Boutique、Hotel Reservation、Sock Shop、Train Ticket。

**📈 对比分析**

与静态匹配基线、3MileBeach-Random 以及 H-Random 对比，SequenceFI 在 9 个时序场景中 100% 触发正确窗口，平均仅一次注入尝试，搜索时间比 H-Random 低 95.9%，比 3MileBeach-Random 低 99.9%，且开销低：吞吐率 97.3% 的无代理水平，CPU 仅 Istio 的 41%，内存 仅 2.3%。

**⚠️ 局限性**

限制包括仅支持 HTTP/1.x 与 gRPC，guard 仅为正向出现计数，无法处理非单调时序条件；依赖 Trace Context 进行调用关联；需针对其他协议扩展拦截器。

---

## 288. Importance-Aware OBS Pruning for Diffusion Models

**arXiv ID:** 2607.20048 | [PDF](https://arxiv.org/pdf/2607.20048v1)

**作者:** Ba-Thinh Lam `[一作]` (UNC Charlotte), Hieu Le `[通讯]` (UNC Charlotte)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、训练后剪枝框架，利用空间重要性地图（如CFG差分、Canny边缘或YOLO检测）改进扩散模型的第二阶（OBS）剪枝。

**💡 创新点**

创新点在于把语义/结构重要区域的空间权重注入OBS的Hessian构造，使剪枝决策更关注保留生成中关键的视觉内容，显著提升高稀疏率下的生成质量。

**🔧 技术方法**

使用Optimal Brain Surgeon（OBS）第二阶剪枝、Hessian重构、CFG指导、Canny/YOLO等空间重要性信号以及无训练一轮的剪枝流程。

**📊 数据集**

使用Stable Diffusion 3 Medium、PixArt‑Σ模型；校准样本来自GCC3M；评估样本来自MS‑COCO 2017验证集；指标包括CLIP Score、ImageReward和MUSIQ。

**📈 对比分析**

与现有OBS‑Diff方法对比，在40–60%稀疏率下，CFG‑guided剪枝在CLIP、ImageReward、MUSIQ上均优于或接近OBS‑Diff，尤其在高稀疏率下能保持主体完整与结构，说明内容感知剪枝更有效。

**⚠️ 局限性**

局限性：依赖重要性信号的质量，CFG映射在抽象或小物体场景易受噪声影响；未探索微调进一步提升效果；硬件稀疏实现有限；实验仅覆盖文本‑图像生成，尚未验证到视频、编辑等其他任务。

---

## 289. TINY_SCHILLER: A Drop-In German Drama Corpus for Small Language Models

**arXiv ID:** 2607.19992 | [PDF](https://arxiv.org/pdf/2607.19992v1)

**作者:** Mark Schutera `[一作]` `[通讯]` (Duale Hochschule Baden-Wurttemberg), Mark Schutera (Duale Hochschule Baden-Wurttemberg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个可直接下沉的小型德语文学语料库tiny_schiller，提供单文件清洗、预token化、精细化的persona拆分和示例Fine‑Tune脚本；

**💡 创新点**

创新点在于消除德语小规模语言模型训练的“单文件摩擦”，提供一次性可使用的清洗语料、预token化拆分、Persona拆分和完整参考Fine‑Tune流程；

**🔧 技术方法**

使用Python脚本进行机械化归一化与speaker‑tag统一，利用tiktoken进行GPT‑2 BPE与cl100k_base子词token化，采用HuggingFace datasets与TRL的SFTTrainer完成监督微调；

**📊 数据集**

以弗里德里希·席勒的11部戏剧（来自DraCor GerDraCor纯文本导出）作为数据集，约201万字符；

**📈 对比分析**

对比方法：在Qwen/Qwen2.5‑0.5B‑Instruct模型上进行两阶段SFT，单GPU训练耗时3.6h，Stage 1 eval loss 0.182、token acc 0.965；Stage 2 persona微调仅3 min，eval loss 0.044、token acc 0.989；显示快速迭代与高准确率；

**⚠️ 局限性**

局限性：语料规模过小，无法用于预训练大模型；speaker‑tag识别基于正则，易忽略特殊排版；未提供生成质量评估；仅适用于戏剧文本，非通用文体。

---

## 290. Generalized Kalman filter based temporal difference reinforcement learning

**arXiv ID:** 2607.20010 | [PDF](https://arxiv.org/pdf/2607.20010v1)

**作者:** Vasos Arnaoutis `[一作]` (University of Twente), Bojana Rosić `[通讯]` (University of Twente)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于条件期望的通用时差强化学习框架，能够同时估计值函数的均值和二阶矩，从而实现对学习过程中的不确定性量化。

**💡 创新点**

创新点在于将时差学习重新表述为投影式条件期望估计，并将卡尔曼滤波器扩展到非线性、非高斯情形；同时利用多项式混沌(PCE)或集合近似来离散随机参数，兼顾了理论严谨性与计算可行性。

**🔧 技术方法**

主要技术包括：条件期望投影、Gauss‑Markov‑Kalman滤波（GMKF‑TD）、多项式混沌展开、集合方法、贝叶斯不确定性传播、以及基于贝叶斯框架的探索策略（Thompson采样/ UCB）。

**📊 数据集**

实验数据集：一维线性质量–弹簧–阻尼系统（离散化状态空间）和二维受热闭腔体（Navier‑Stokes + 能量方程），通过数值仿真得到奖励与转移信息。

**📈 对比分析**

与经典TD‑α以及基于梯度的TD（如SARSA、Q‑learning）进行比较；GMKF‑TD在两组实验中均表现出更快的收敛速度、更低的相对误差以及更小的参数方差，且能够自动调节学习率。相比之下，传统TD需要手动调优学习率，且在非高斯、非线性问题上收敛更慢。

**⚠️ 局限性**

局限性包括：对高维随机空间的PCE表示易产生维数灾难；集合方法在样本量不足时会出现统计误差；协方差修正容易受数值误差影响；模型需要先验的概率分布与参数方差，若设定不当会导致收敛不稳定；目前仅在相对简单的物理控制问题中验证，尚未在大规模真实世界环境中测试。

---

## 291. Co-Evolving LLM Evaluators and Policies via DynamicRubric

**arXiv ID:** 2607.20083 | [PDF](https://arxiv.org/pdf/2607.20083v1)

**作者:** Beining Wang `[一作]` (Tsinghua University), Yiqun Liu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出一种动态rubric评估框架DynamicRubric，用于在LLM后训练中根据当前策略生成的候选答案动态生成加权二进制rubric项，并通过冻结的verifier将其转换为响应级评分，从而为策略优化提供细粒度且可解释的反馈。

**💡 创新点**

核心创新是将评估器与策略共同进化，即评估器通过响应集条件生成与当前候选答案相关的rubric，保持评估分数的相对区分度；并通过理论证明评估器的相对分数差即为策略优化的梯度方向。

**🔧 技术方法**

技术上包括：二进制rubric生成网络（DR-Generator）、加权投票聚合、冻结的verifier、基于GRPO的策略优化、判别性奖励与anchor排名奖励的联合学习。

**📊 数据集**

使用的主要数据集是Nectar（含7级排序的对话回复），以及多种评估基准（JudgeBench、Personalized-RB、RM-Bench、UltraFeedback、AEOLLM、LMSYS-Chat-1M）和开放式生成基准（AlpacaEval2、ArenaHardv2.0、WildBench、WritingBench）。

**📈 对比分析**

与多种基线（包括大型奖励模型、无响应集的评估器、静态rubric等）进行对比，DynamicRubric在评估器准确率、nDCG、以及策略在开放式生成和可验证推理任务上的表现均优于基线，尤其在大规模生产环境（微信搜索AI答复）中实现了显著的在线指标提升。

**⚠️ 局限性**

主要限制包括：训练时额外的生成器‑verifier 计算开销较大；评估器依赖冻结verifier，未能利用更强大的黑盒评估器；以及在更大模型或更大响应集规模下的可扩展性尚未验证。

---

## 292. The Two-Process Theory of Machine Self-Report

**arXiv ID:** 2607.20082 | [PDF](https://arxiv.org/pdf/2607.20082v1)

**作者:** Hubert Plisiecki `[一作]` (IDEAS Research Institute), Marcin Moskalewicz `[通讯]` (IDEAS Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究设计并验证了一个48条题的机器自我报告量表Pinocchio Inventory，并发现语言模型的自述可拆解为两维度——内在生活（B）与归因门控（A），在206个公开权重模型上进行了系统验证。

**💡 创新点**

首次提出基于模型自身响应结构的两过程心理学理论，将单一的Pinocchio Axis拆解为可测量的两维度，并提供了可靠的评估工具。

**🔧 技术方法**

采用多元测验设计（三形式并行、MTMM）、因子分析、结构方程模型、配对比较和聚类Bootstrap等统计方法进行验证。

**📊 数据集**

对50个原始模型与206个公开权重模型（含67对同一检查点的基线/后训练版本）进行48项自我报告测量，结合原始1411条题库进行分析。

**📈 对比分析**

通过内在一致性（α 0.82–0.94）、跨形式收敛相关（0.84）、八个月稳定性（0.93）以及配对检验验证，后训练显著提升B (+0.20)，门控A在模型规模下呈负相关，证明理论的预测力。

**⚠️ 局限性**

局限包括样本主要为助手模型、基线与后训练路径混杂、门控交互效应为后发现、回应模式受回应率影响、缺乏对真实经验或下游行为的预测。

---

## 293. Scuba Diving Graphs

**arXiv ID:** 2607.20059 | [PDF](https://arxiv.org/pdf/2607.20059v1)

**作者:** Alexander M. Esser `[一作]` `[通讯]` (University of Koblenz), Alexander M. Esser (University of Koblenz)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个多层时序图模型，将潜水员小组的物理距离、沟通距离和紧急距离三维关系可视化为时间网络。

**💡 创新点**

将潜水员视为社交网络节点，构建三维层次化的多层图，首次将计算社会科学方法引入潜水安全分析，提供可扩展的安全评估框架。

**🔧 技术方法**

使用图论和多层网络理论构造模型，并结合时间网络的概念进行理论分析，未进行实验实现。

**📊 数据集**

本研究未使用实际数据集，仅以文献案例和假想场景为依据，提出未来可利用 Dive Alert Network、数字日志和传感器数据进行验证。

**📈 对比分析**

由于模型仅为概念性，未进行方法比较或性能评估，作者仅讨论了可能的评估思路与未来实验设计。

**⚠️ 局限性**

局限性包括缺乏实证验证、沟通与紧急距离难以直接测量、数据获取与同步困难、模型仅为无向静态，未考虑方向性关系和多维度动态演化。

---

## 294. Language-Specific versus Cross-Lingual Knowledge Graphs for Implicit Aspect Identification in Arabic: A Comparative Study of Reasoning and Adaptation Strategies

**arXiv ID:** 2607.20056 | [PDF](https://arxiv.org/pdf/2607.20056v1)

**作者:** Lujain A. Alawwad `[一作]` `[通讯]` (Saudi Electronic University), Lujain A. Alawwad (Saudi Electronic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在一个混合管道中对阿拉伯语的隐式和显式方面抽取进行了系统评估。

**💡 创新点**

首次对比跨语言英文知识图与本土阿拉伯知识图的性能，并证明任务特定微调比零样本提示更有效。

**🔧 技术方法**

使用Llama‑3‑8B + QLoRA微调、语义相似度匹配、表面形式匹配以及基于句子生成的显式提取。

**📊 数据集**

M‑ABSA、SemEval‑2016 Arabic、HAAD三大阿拉伯语基准数据集。

**📈 对比分析**

通过在同一数据集和管道下进行对比，发现本土阿拉伯KG的micro‑F1在M‑ABSA提升0.199、在SemEval‑2016提升0.251；零样本提示的F1≤0.13，而微调后提升至0.66–0.76。

**⚠️ 局限性**

主要限制是隐式抽取的召回受限于无情感词的句子，且本土KG依赖人工构建的词表，无法完全覆盖所有形态变化。

---

## 295. Global Difference Constraint Propagation for Constraint Programming

**arXiv ID:** 2607.20022 | [PDF](https://arxiv.org/pdf/2607.20022v1)

**作者:** Lucas Kletzander `[一作]` (TU Wien), Peter J. Stuckey `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在有限域求解器中实现了全局差分约束传播器，能够一次性处理所有 x-y≤d 约束并提供解释机制，用于懒惰子句生成求解器；

**💡 创新点**

将 SMT 领域的全局差分约束理论迁移至有限域传播，提出增量满足性与蕴含检查、改进的边界传播（IncLB/IncUB）、解释生成策略，并通过图简化与局部传播器混合使用，显著提升传播效率；

**🔧 技术方法**

使用 Bellman‑Ford/Johnson 与 Dijkstra 的增量算法处理差分约束图；实现 IncSat 与 IncImp 用于增量满足性与蕴含判断；改进的边界传播 IncLB/IncUB；懒惰/立即解释技术；在 Huub CP 求解器中整合并使用后退轨迹与优先级调度；

**📊 数据集**

在 626 个 MiniZinc Challenge 2024 benchmark、两组 RCPSP（资源约束项目调度）小型与完整实例，以及相关调度实例上进行实验；

**📈 对比分析**

与 Chuffed、OR‑Tools CP‑SAT 进行对比；在 MiniZinc benchmark 上，启用全局差分逻辑后点数从 310 提升至 321，Unsat/Unknown 次数显著下降；在 RCPSP/PP 调度实例上，Optimal 与 Unsat 解决比例提升约 20–30%；最坏案例从 O(n³) 降至 O(n²)；

**⚠️ 局限性**

初始化与简化阶段仍需 O(n²logn) 复杂度，IncImp 检查开销大且不值得；对可选区间变量的特殊约束支持有限；解释生成仅覆盖简单情况，复杂约束仍需手工或后置处理；

---

## 296. EvoDRC: A Self-Evolving Agentic Framework for Automated DRC Violation Repair

**arXiv ID:** 2607.20019 | [PDF](https://arxiv.org/pdf/2607.20019v1)

**作者:** Bing-Yue Wu `[一作]` (Arizona State University), Vidya A. Chhabria `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EvoDRC 框架，实现了块级布局的自动化 DRC 违规修复，通过将布局拆分为修复裁剪、使用 LLM 代理进行局部修复、记录修复结果并持续进化修复技能，实现了从参考设计迁移到目标设计的无缝技能演化。

**💡 创新点**

创新点包括：①将 DRC 修复视为技能演化任务，首次实现从无关参考设计提取可迁移技能并在目标设计上持续更新；②引入布局裁剪与多代理并行修复，提高了大规模块级设计的可扩展性；③构建可追溯的知识数据库并通过 Skill Refiner、DB Summarizer、Skill Judge 三阶段机制自动生成并选择最佳层级技能文件；④结合 preview 工具提供局部 DRC、连通性与影响预览，提升代理决策质量。

**🔧 技术方法**

采用 Claude Sonnet 4.6 LLM 代理，ReAct 交互框架，KLayout Python 脚本和 DRC 检查器，连通性检查工具，影响预览工具，以及知识数据库与技能演化脚本，实现端到端的自动修复与技能迭代。

**📊 数据集**

使用 DAC26-DRC-Benchmark 中的七个块级测试案例（68–765 个违规），以及一个来自 OpenCores 的 CLA 设计作为参考，全部采用 ASAP7 PDK 和 KLayout DRC 规则栈。

**📈 对比分析**

通过与基准方法及三种消融配置（无技能迁移、冻结技能、禁用裁剪）比较，EvoDRC 在所有案例上平均实现 83.6% 的违规数量下降（相比基准 73.5%），在某些块级案例（Block3、Block5）完全清零违规，整体修复效率显著优于消融实验，成本低于人工手工 ECO。

**⚠️ 局限性**

局限性包括：①对 LLM 的推理与生成质量高度依赖，可能在极端规则或几何情况下降低修复成功率；②技能演化过程需要多轮迭代，对大规模设计仍可能产生较高计算时间；③裁剪策略对裁剪尺寸参数敏感，过小或过大均可能导致修复覆盖不足或过度；④目前仅在 ASAP7 规则集验证，跨工艺或更复杂规则的迁移尚未充分评估。

---

## 297. Post-Training in Time Series Foundation Models: A Unifying Framework

**arXiv ID:** 2607.20002 | [PDF](https://arxiv.org/pdf/2607.20002v1)

**作者:** Shifeng Xie `[一作]`, Keli Zhang `[通讯]` (Noah's Ark Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

系统性综述并归纳时间序列基础模型（TSFM）在后训练阶段的五大方法族，提出统一的干预位置分类法并讨论其优缺点与未来方向。

**💡 创新点**

首次将后训练方法按参数、上下文、模型组合、输出处理与压缩等五个干预层次进行统一归类，揭示各方法的核心机制、共性与差异，提供对策与研究蓝图。

**🔧 技术方法**

文献综述、结构化分类、趋势分析、问题归纳与未来路线图设计等技术手段。

**📊 数据集**

未进行实验，主要基于公开文献与前沿工作（包括Chronos、TimeGPT、TimeRAG等）进行分析，不涉及具体数据集。

**📈 对比分析**

通过对比已发表方法的目标、适用场景与限制，说明各类别在不同任务（预测、异常检测、分类等）上的表现差异；未给出统一性能指标，仅以文献中已报告的结果为参考。

**⚠️ 局限性**

局限性：仍以预测为主，缺乏对分类、异常检测、填补等任务的系统评估；对检索、记忆与上下文构建的质量评估不足；压缩与特化方法多聚焦单一指标（参数量），缺乏统一效率与鲁棒性基准；缺少实证验证新提出的后训练框架。

---

## 298. UniRank: Benchmarking Ranking Models for Unified Sequential Modeling and Feature Interaction

**arXiv ID:** 2607.19987 | [PDF](https://arxiv.org/pdf/2607.19987v1)

**作者:** Honghao Li `[一作]` (Anhui University), Yiwen Zhang `[通讯]` (Anhui University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 UniRank 开源基准，用于统一序列建模与特征交互的排序模型的可复现评估与比较。

**💡 创新点**

创新点在于：①统一的时间序列点级自回归监督，②统一的数据流水线与任务/指标定义，③统一模型代码与超参对齐，④针对工业级训练的效率优化工具包，⑤支持五大公开大规模数据集，形成完整可复现的实验生态。

**🔧 技术方法**

采用 PyTorch 分布式数据并行、Inductor 编译、混合精度、Flash Attention、Flex Attention、Tokenization（Chunk、Auto、Field、Random）策略、层级/分层统一交互、各种注意力激活（Softmax、ReLU、SiLU、GeLU等）以及多种优化器（AdamW、LaProp、SOAP 等）实现高效训练。

**📊 数据集**

使用五个公开数据集：QK‑Video、KuaiRand、TAAC‑25、Taobao、MerRec，样本规模从数亿到十亿，最长行为序列超过 10⁵。

**📈 对比分析**

在统一序列长度 100、统一模型容量与训练协议下，对 15 种统一排序模型进行评估，报告 AUC 与 Logloss；结果显示无单一模型始终最优，模型性能高度依赖数据集与任务，差距可达数个百分点，揭示不同平台的最优结构差异。

**⚠️ 局限性**

局限性包括：仅覆盖 token‑based 统一模型，未涵盖多模态或极大规模数据；实验设置固定（序列长度 100、单 GPU 训练），未探索更长序列或更大模型的极限；超参与优化器仍需针对不同数据集手动调优，导致跨数据集迁移性能不稳。

---

## 299. On 2-Layer k-Matching-Planar Graphs

**arXiv ID:** 2607.19981 | [PDF](https://arxiv.org/pdf/2607.19981v1)

**作者:** Saeed Odak `[一作]` (Aalto University), Torben Scheele `[通讯]` (TU Dortmund)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了二维层k‑匹配平面图（2‑layer k‑matching‑planar graphs）的结构性质、路径宽度界限、识别复杂度以及参数化算法，并给出了两侧识别问题的不可逼近性结果。

**💡 创新点**

创新点在于将k‑匹配平面概念引入二维层绘图，证明该类图的路径宽度上界为2k+1并给出匹配下界3⌊k/2⌋+1；首次证明OneSided‑k‑MatchingPlanarity NP‑难且提供FPT算法；并证明TwoSided‑k‑MatchingPlanarity在任何常数因子下都无法近似（即使在树图上）。

**🔧 技术方法**

主要技术包括：左‑康尼格覆盖（Kőnig定理）用于控制交叉边的顶点覆盖；窗口大小引理和双层结构用于窗口化处理；动态规划与窗口合并实现FPT算法；从k‑Partition、3‑Partition及树带宽问题构造归约以证明NP‑难与不可逼近性。

**📊 数据集**

本文为理论算法与图结构研究，无实验数据集；所有结果均通过构造图与数学证明得到。

**📈 对比分析**

通过构造上界与下界证明路径宽度是Θ(k)，并在识别问题上给出NP‑难与FPT的边界；FPT算法在O(n³+|X|2^{2^{12k}})时间内可解决，虽然指数对k极大；两侧识别的不可逼近性表明即使在树图上也无法得到常数因子近似。

**⚠️ 局限性**

限制包括：路径宽度上界与下界之间仍存在常数因子差距（2 与 1.5）；未给出树宽的精确上界；两侧识别问题的XP或XNP‑hard性仍未确定；FPT算法的时间复杂度对k指数级，实际可用性有限。

---

## 300. How Close is a Tree to a Euclidean Minimum Spanning Tree?

**arXiv ID:** 2607.20007 | [PDF](https://arxiv.org/pdf/2607.20007v1)

**作者:** Todor Antić `[一作]` (Charles University), Pavel Valtr `[通讯]` (Charles University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了树图的欧氏最小生成树绘图（EMST绘图），并给出了线性时间算法来判定并绘制满足无坏对的“猫形”树（caterpillar），以及在最大度数为六的猫形树中最小化坏对数的算法，并对一般树和星图给出了坏对数上界及最优值。

**💡 创新点**

首次完成了对猫形树EMST可绘性的完全刻画，推出了线性时间判定与构造算法；此外提出了在度数六的猫形树中实现最小坏对数的线性时间方法，并给出了星图的精确坏对数。

**🔧 技术方法**

利用图论结构分析、三角网格嵌入、递归构造、路径分解与颜色标记等技术，结合平面几何与欧氏距离性质，构造EMST绘图并计数坏对。

**📊 数据集**

本研究主要是理论分析和算法设计，没有使用实验数据集；验证通过数学证明与算法复杂度分析。

**📈 对比分析**

与先前已知的NP‑难判定结果对比，本文在猫形树上实现了线性时间的判定与构造；在最大度数为六的猫形树中最小化坏对数也保持线性时间；对一般树给出的上界为Δ²nlog n，已优于朴素的n²上界。

**⚠️ 局限性**

算法在面积使用上仍为指数级（EMST绘图），对一般树的坏对数上界仅为上界，未给出下界；对更广泛图类（如路径宽度受限图）是否可行尚未解答。

---

## 301. RALS: Resources and Baselines for Romanian Automatic Lexical Simplification

**arXiv ID:** 2607.20078 | [PDF](https://arxiv.org/pdf/2607.20078v1)

**作者:** Fabian Anghel `[一作]` (University of Bucharest), Sergiu Nisioi `[通讯]` (University of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了首个罗马尼亚语词汇复杂度预测（LCP）与词汇简化（LS）数据集，并基于此构建了多种简化系统；

**💡 创新点**

创新点包括：①首次在罗马尼亚语上构建LCP/LS资源；②采用配对排名的候选词排序方法；③设计混合规则/词典驱动的DexFlex简化框架；

**🔧 技术方法**

使用的技术涵盖手工特征的岭回归预测、BERT与RoBERTa模型、开放式LLM（Apertus‑8B、RoLlama‑8B）提示、DexFlex基于spaCy+DEX词典的规则化简化；

**📊 数据集**

数据集来源包括：原始MultiLS英语数据、HT（人译）、WT（机器译）以及RoLCP新采样文本，和RoLS候选替换词集合；

**📈 对比分析**

通过交叉验证、MAP@N、Potential@N与Accuracy@N指标进行对比，DexFlex在所有指标上均优于纯LLM或BERT方法，性能虽低于高资源语言但与低资源语言相当；

**⚠️ 局限性**

局限性包括：不同子集标注方式差异、受限于年轻高学历志愿者的样本多样性、DexFlex对多词性同形词处理不足、实验预算极低导致扩展受限。

---

## 302. TRUST-ESD: A Risk-Calibrated and Governance-Aware AI Framework for Enterprise Strategic Decision Support Under Uncertainty

**arXiv ID:** 2607.20065 | [PDF](https://arxiv.org/pdf/2607.20065v1)

**作者:** Tian Qiu `[一作]` (Xinyu University), Jahid Hasan `[通讯]` (China Three Gorges University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为TRUST-ESD的框架，用于在不确定性环境下提供风险校准、治理合规、可解释性的企业战略决策支持。

**💡 创新点**

创新点在于将预测效用估计、合成式不确定性校准、CVaR下行风险评分、风险记忆检索、政策代码治理、可解释性与人工监督等多项技术集成于一个统一的决策优化目标，超越传统仅以最大期望效用为准的决策方法。

**🔧 技术方法**

使用的技术包括：基于梯度的状态-动作表示学习、Conformal预测实现不确定性校准、CVaR风险评估、历史风险记忆检索、政策代码（policy-as-code）治理规则、模型解释技术（如分数分解）以及人机交互审核流程。

**📊 数据集**

在实验中采用了零售业务的Rossmann门店销售数据和M5预测数据，构建了基于库存、定价、促销等操作的候选策略集合。

**📈 对比分析**

与多种基线（线性/树模型、XGBoost、LightGBM、CatBoost、TabNet、FT-Transformer、NGBoost、Quantile LightGBM、Deep Ensembles以及各类决策优化变体）对比，TRUST-ESD在风险调整效用（RAU）上提升约8%，风险暴露和CVaR下降约20%，校准误差降低约14%，预测区间覆盖率提升约1.2%，解释忠实度提升约10.9%，治理合规率提升约9.8%，同时保持与最佳预测模型相当的预测精度。

**⚠️ 局限性**

局限性包括：框架在动态多时态环境中的适用性尚未验证；政策规则需要人工编写且可能需要频繁更新；对大规模高维实时决策的计算成本尚需进一步评估；以及模型对罕见极端事件的鲁棒性仍待加强。

---

## 303. Experiential Versus Instructional Approaches for Eliciting Metacognitive Awareness in AI-Assisted Learning: A Short-Term Longitudinal Study

**arXiv ID:** 2607.20047 | [PDF](https://arxiv.org/pdf/2607.20047v1)

**作者:** Pau Benazet i Montobbio `[一作]` (Pompeu Fabra University), Davinia Hernández-Leo `[通讯]` (Pompeu Fabra University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对比了实验性与讲授式教学方法对大学生在使用生成式人工智能（GenAI）进行学习时的元认知意识（知识层与调节层）变化，采用两组实验设计并用MAI‑AI问卷进行三次测量（前测、立即后测、五周后测）。

**💡 创新点**

首次系统性比较两种教学方法在GenAI环境下的短期纵向效应；发现实验性教学能迅速提升学生的元认知知识和AI投入度，并在随后的五周内逐步提升调节能力；提出元认知知识与调节在经验式学习中可时间上分离的理论。

**🔧 技术方法**

实验设计、MAI‑AI量表（基于MAI与MSLQ的AI适配版）、生成式AI工具（Google Gemini 2.5）以及统计分析（t检验、Wilcoxon、Mann–Whitney、Benjamini–Hochberg校正）等技术。

**📊 数据集**

来自西班牙Pompeu Fabra大学的138名一年级工程专业学生（最终有效样本为126/107），并非公开数据集。

**📈 对比分析**

通过前后测比较两组在六个构成维度（AIA、AIE、AIL、MKC、MRC、ANC）的变化；结果显示实验组在立即后测中AIE和MKC显著高于讲授组；五周后两组AIE相当，但实验组在MRC显著提升；整体表现优于传统讲授方法。

**⚠️ 局限性**

限制：依赖自评问卷，可能受社会期望和主观误差影响；样本单一专业/单校，缺乏跨文化、多学科验证；未严格控制学生课后独立使用AI的差异；样本量略低于预期，可能影响统计功效；未使用客观表现或思维轨迹等多元测量。

---

## 304. Good Practice Guide for quantifying uncertainties for machine learning models applied to photoplethysmography signals

**arXiv ID:** 2607.19999 | [PDF](https://arxiv.org/pdf/2607.19999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 305. Robots Acquire Manipulation Skills in Seconds from a Single Human Video

**arXiv ID:** 2607.20033 | [PDF](https://arxiv.org/pdf/2607.20033v1)

**作者:** Guangyan Chen `[一作]` (Beijing Institute of Technology), Yufeng Yue `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在机器人推理阶段利用单段人类视频实现抓取、搬运等操纵技能，并可随时调用已存储的视频完成重复任务。

**💡 创新点**

创新点主要有：
1）将技能获取视为“视频驱动的预测”过程，将每个时刻的动作目标与视频后续进度耦合；
2）通过自我驱动的三阶段预测级联（进度定位 → 未来观测预测 → 动作生成）桥接人类视频与机器人执行之间的结构差异；
3）采用两阶段训练（同体 robot–robot 预训练 + 少量人–机器人对齐）大幅降低对稀缺跨体数据的依赖。

**🔧 技术方法**

核心技术：
- 任务进度对齐：自监督 Smooth Dynamic Time Warping + 轮回一致性；
- 级联预测：单一自回归扩散模型（Mixture‑of‑Transformer），包括定位、观测预测与动作生成专家；
- 视觉与语言编码：Qwen3‑VL‑Embedding‑8B、WAN VAE、T5 文本编码器；
- 两阶段训练策略与 LoRA 微调。

**📊 数据集**

数据集：
- 193 462 条同体 robot 轨迹，涵盖 229 个任务；
- 5 847 条少量人–机器人配对视频；
- 评估集 50 个新任务（无训练视频）及 7 个已掌握任务。

**📈 对比分析**

与基线对比：
- 在 50 个新任务上取得 62% 成功率，分别比最佳 OSVI 基线高 43%（仅用同一视频），比最佳零样本 imitation 低 45%；
- 与在 50 条演示上微调的基线相比，成功率相近但所需演示 50 倍更少，学习时间从 4‑5 小时压缩至约 29 秒（≈507 倍加速），且不损失已掌握任务性能（保持 100%→仅 40%）。

**⚠️ 局限性**

局限性：
- 仅在单一双臂平台上验证，跨不同机器人硬件的通用性尚未测试；
- 视频无法捕捉接触力等物理细节，对细粒度抓取仍有限；
- 随着演示库增长，检索阈值对相似度区分的敏感性增加，需更细粒度的检索策略。

---

## 306. Visual Indicators to Increase the Detection of Linguistic Media Bias

**arXiv ID:** 2607.20031 | [PDF](https://arxiv.org/pdf/2607.20031v1)

**作者:** Smi Hinterreiter `[一作]` (University of Würzburg), Marc Erich Latoschik `[通讯]` (University of Würzburg)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究比较了六种视觉指示器（偏见高亮、偏见条形、偏见仪表盘、政治倾向、情感、可信度）在提升短文本语言偏见检测与感知上的效果。

**💡 创新点**

创新点在于首次系统比较多种真实应用场景中的指示器，并通过转移学习设计评估其对检测与感知的影响，揭示政治共识与情感关联对结果的调节作用。

**🔧 技术方法**

采用视觉编码与混合实验设计，结合机器学习的词级偏见标注、用户交互测评与线性混合模型分析。

**📊 数据集**

使用从54篇新闻文章中提取的40-70词短语，并由三名研究者共识标注的119个偏见词构成实验材料。

**📈 对比分析**

通过F1、d'等客观指标及主观感知量表比较，偏见高亮和偏见仪表盘显著提升检测准确度（F1≈0.51-0.53），而偏见条形、情感、政治倾向、可信度指示器效果有限。

**⚠️ 局限性**

局限性包括仅针对短文本、样本偏向自由派、仅评估首次曝光效应、缺乏长期或多语言验证、以及偏见标注的主观性导致基准不完美。

---

## 307. Safe Remediation as Risk-Constrained Intervention Decision in Microservice Systems

**arXiv ID:** 2607.20005 | [PDF](https://arxiv.org/pdf/2607.20005v1)

**作者:** Chengxiao Dai `[一作]` (University of Sydney), Luyan Zhang `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套安全自动化修复决策框架，将修复过程转化为风险约束的干预决策问题，并在离线历史日志上学习出可满足误操作率预算的策略。

**💡 创新点**

创新点包括：①将安全修复建模为CMDP并引入误操作率（FRR）预算；②设计三维风险分解（扩散半径、可逆性、模型不确定性）实现可解释的安全界面；③开发基于上下文的自适应人机交互门，实现带宽感知的升级决策。

**🔧 技术方法**

使用技术包括：CMDP + Lagrangian 约束；三维风险估计（图扩散、回滚概率预测、模型集成不确定性）；检索增强的后果评估（案例检索+相似度加权）；离线强化学习框架 CQL；上下文门采用 contextual bandit 学习策略。

**📊 数据集**

实验基于 Train Ticket 微服务基准，使用 Chaos Mesh 进行 11 类故障注入，构建了与 RCAEval 对齐的故障分类，并采集 Prometheus/Fluentd/Jaeger 的 90 s 监控数据。数据集共 8,320 条决策记录，按 70/15/15 划分训练/验证/测试。

**📈 对比分析**

与 Rule‑Runbook、LLM‑Remed、BC、CQL、CPO、CMDP‑vanilla、Single‑Risk、No‑Risk、Fixed‑HITL 等基线比较。本文方法在 39% 的 FRR 降低、17% 的升级率减少的同时，修复成功率提升至 0.786（比 Rule‑Runbook 提升 2.5%），并在匹配升级率的实验中保持最优的成功率与安全性平衡。

**⚠️ 局限性**

局限性包括：①动作空间是预定义的离散集合，无法直接处理参数化动作；②扩散半径模型基于静态服务图，对快速弹性环境的拓扑漂移需重新估计；③需要足够覆盖的离线数据才能保证策略在部署时满足 FRR 预算；④缺乏严格的正式安全保证，只能通过经验评估和后置风险筛选提供安全性保障。

---

## 308. CLARK: Closed-loop Learning for Adaptive Reasoning over Knowledge Graphs

**arXiv ID:** 2607.19996 | [PDF](https://arxiv.org/pdf/2607.19996v1)

**作者:** Yousef Khan `[一作]` (Sano - Centre for Computational Personalised Medicine), Jose Sousa `[通讯]` (Sano - Centre for Computational Personalised Medicine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 CLARK 框架，将基于 CACTUS 的知识图谱、符号规则挖掘和 LP^MLN 概率推理闭环结合，实现知识驱动的分类模型。

**💡 创新点**

创新点：①闭环学习机制，使规则挖掘、权重校准和图结构在 LP^MLN 下互相迭代；②在数据驱动的 KG 上引入可解释的符号规则，并通过概率学习统一权重；③首次在医学数据上评估该闭环框架的可解释性与适应性。

**🔧 技术方法**

使用技术包括 CACTUS（KG 生成）、LP^MLN（软规则权重学习与推理）、三种符号规则学习器（SL、CMAR、AMIE3）、Bootstrap 稳定性筛选，以及基于 PageRank 和 Degree 的图推理策略。

**📊 数据集**

采用两个医学基准数据集：Cleveland Heart Disease (HDC) 与 Wisconsin Diagnostic Breast Cancer (WDBC)。

**📈 对比分析**

与基线 CACTUS KG 以及仅加入 LP^MLN 的版本对比，结果显示单独使用 LP^MLN 可能降低性能，而引入 SL 或 AMIE3 的规则后性能提升；SL+AMIE3 组合在两数据集上均获得最高提升，平衡准确率最高提升约 3.36%。

**⚠️ 局限性**

局限性：①仅在结构化表格数据上验证，缺乏大规模知识图谱与多模态数据的实验；②LP^MLN 计算成本高，且对规则质量敏感；③缺乏公开实现与代码，复现性受限；④闭环学习过程中可能引入噪声规则导致推理误差。

---

## 309. Development of an automated, reliable, and clinically meaningful artificial intelligence (AI) tool for diagnosing cardiac disease from conventional cardiovascular magnetic resonance (CMR) images

**arXiv ID:** 2607.20087 | [PDF](https://arxiv.org/pdf/2607.20087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 310. STEREOFLOW: Progressive Stereo Matching with StereoDiT and Transition Flow Matching

**arXiv ID:** 2607.19986 | [PDF](https://arxiv.org/pdf/2607.19986v1)

**作者:** Hao Wang `[一作]` (Beihang University), Biao Leng `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种先验引导的生成式立体匹配框架，并实现了StereoFlow模型。

**💡 创新点**

创新点包括三大模块：① 低分辨率4D GEV+高分辨率3D ACV的双阶段渐进级联匹配网络；② 为立体匹配专门设计的频率解耦像素扩散变压器StereoDiT；③ 用差异流匹配（Transition Flow Matching）替代传统高维扩散路径，实现少步高效采样。

**🔧 技术方法**

核心技术包括卷积GRU迭代回归、像素级扩散Transformer、基于多模态分布的生成建模、线性插值与噪声调度的流匹配损失、REPA对齐、单目深度先验以及多尺度成本体积构建。

**📊 数据集**

使用Scene Flow（合成）进行预训练，随后在ETH3D、Middlebury、KITTI-2012/2015以及Virtual KITTI-2上微调；在零样本迁移评估时直接在真实数据上测试。

**📈 对比分析**

与现有SOTA方法（如RAFT‑Stereo、IGE‑V、DEFOM‑Stereo、Monster、BridgeDepth等）相比，StereoFlow在Scene Flow、KITTI、ETH3D、Middlebury上均取得最低EPE、最佳Bad1/2/3、D1、Out-2/3/4等指标，尤其在零样本迁移与反射区域表现突出，性能提升幅度从3%到40%不等。

**⚠️ 局限性**

局限性：① 扩散模型仍带来显著推理延迟，尚未实现一步采样；② 两阶段训练策略增加了训练复杂度与不稳定性，亟需更稳健的一阶段端到端训练方案。

---

## 311. Forecasting the Number of Harvest-ready Fruits of Sweet Peppers Using Multimodal Time-Series Data

**arXiv ID:** 2607.19975 | [PDF](https://arxiv.org/pdf/2607.19975v1)

**作者:** Enrico Pallotta `[一作]` (University of Bonn), Juergen Gall `[通讯]` (University of Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了甜椒单株收获果实计数的多模态时间序列数据集，并提出融合图像特征与果实计数的LSTM框架，用于跨季节预测收获果实数量。

**💡 创新点**

创新点包括：①首次公开甜椒时间序列图像+计数数据集；②将DinoV3视觉编码器提取的高维特征与数值计数融合，并通过FiLM对时间间隔进行条件化；③使用Deep Ensembles与Gaussian NLL提供校准的预测不确定度。

**🔧 技术方法**

使用技术包括：LSTM时序建模、DinoV3 Vision Transformer编码器、FiLM时间条件化、Deep Ensembles、Gaussian Negative Log-Likelihood训练、Δ_t处理、果实成熟阶段聚合。

**📊 数据集**

使用数据集：691株甜椒，4837张RGB图像，覆盖2022和2023两季共7天采样，每株包含按成熟阶段的果实计数；数据由PATHoBot平台收集并手工对齐。

**📈 对比分析**

通过与持久性基线和仅计数模型对比，使用RMSE和MASE评估。多模态模型在2022、2023分别比基线降低约33%/38% RMSE，且比计数模型微幅提升；Δ_t FiLM进一步提升RMSE；跨季评估显示低UCE和NLL，说明模型在不同季节泛化良好。

**⚠️ 局限性**

局限性：视觉特征对预测提升有限；跨季校准不对称（2023→2022时UCE高）；仅覆盖两季且缺乏多品种/不同环境的验证；数据规模相对有限，未来需扩大样本与场景。

---

## 312. The Giant Hippocampus: From Structural Monoculture to a System of Systems

**arXiv ID:** 2607.19973 | [PDF](https://arxiv.org/pdf/2607.19973v1)

**作者:** Jaeho Seol `[一作]` `[通讯]` (Independent Researcher), Jaeho Seol (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `70e40602-aae3-44bd-80ec-4a7f2674330f` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

本文通过跨学科综述和理论分析，对比大脑皮层的细胞层次、细胞类型、基因表达与当前人工智能模型（尤其是Transformer）的统一结构，指出现有AI架构在结构先验上的错误，提出一种“异质拓扑网络（Heterogeneous Topological Network）”的架构范式，强调在训练前就需预设模块化的结构与功能划分。

**💡 创新点**

创新点在于：① 用神经解剖学实证（Brodmann、Jülich-Brain、Patch‑seq等）来证明功能与结构的多样性；② 将此生物学发现与功能主义、可实现性等哲学理论相结合，形成对AI设计的结构性先验要求；③ 提出将视觉、听觉、记忆、执行等不同功能拆分为结构上不同的模块，并通过标准化接口进行协同的“异质拓扑网络”架构。

**🔧 技术方法**

主要采用文献综述与理论推演；引用神经科学方法（connectomics、Patch‑seq、概率脑图谱）与AI可解释技术（注意力可视化、Granger因果、网络拓扑分析）来构建论证框架；并借鉴CNN与Transformer在视觉任务中的结构对比来说明结构先验的重要性。

**📊 数据集**

引用的数据与资源包括：MOS 6502芯片电路数据、C. elegans完整连接图、Patch‑seq单细胞多模态数据、人脑Jülich‑Brain三维概率图谱、GPT‑3、Vision Transformer、Audio Spectrogram Transformer等公开预训练模型与其训练/评估数据集；作者未自行收集或训练数据集。

**📈 对比分析**

通过对比CNN与Transformer在视觉任务中的参数量、训练样本需求、Top‑5错误率等指标，展示CNN在局部性与层级性上的参数效率优势；对Transformer在多模态（文本、图像、音频）下的表现进行评估，说明其对大规模预训练的高度依赖；进一步用已公开的模型评估结果证明结构先验能显著降低样本与计算成本。

**⚠️ 局限性**

局限性：① 论文为理论与综述性质，缺乏对提出的异质拓扑网络在实际任务中实现与性能的实证验证；② Transformer作为海马体类比的具体实现与接口设计仍未给出细节；③ 对硬件适配、可扩展性、训练成本等工程层面讨论有限；④ 对多模态任务的量化比较主要基于已有公开模型，未包含针对新架构的对比实验。

---

## 313. Cumsum-Composable Phase Transport for Low-Cost Streaming Keyword Spotting

**arXiv ID:** 2607.20086 | [PDF](https://arxiv.org/pdf/2607.20086v1)

**作者:** Mahesh Godavarti `[一作]` `[通讯]` (A Carrot, Inc), Mahesh Godavarti (A Carrot, Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并评估了一种基于累加求和相位传输的低成本流式关键词检测模型。

**💡 创新点**

将相位旋转与前缀求和结合，提供与传统扫描不等价的完全可并行且可流式的前缀差分实现。

**🔧 技术方法**

采用单位模旋转的相位传输层、固定频率或学习频率的旋转、前缀累加、门控残差、Mel 频谱或学习的 Cumsum 声学前端、卷积/残差分类器等技术。

**📊 数据集**

使用 Google Speech Commands v2（一秒长、12 类命令）的数据集进行实验。

**📈 对比分析**

与紧凑 CNN 基线和自学习衰减扫描层在相同前端下对比；cumsum 模型在 51k 参数下达到 97.3% 准确率，参数可压缩至 25k，推理延迟比扫描低 1.4 倍，训练速度提升 1.07 倍。

**⚠️ 局限性**

实验仅在单一随机种子、仅在关键词检测任务上完成，未在多硬件/多基线上复测，也缺乏对鲁棒性与事件触发评估的深入探讨。

---

## 314. Factor-Informed Uncertainty Distillation for Gaze Estimation

**arXiv ID:** 2607.20072 | [PDF](https://arxiv.org/pdf/2607.20072v1)

**作者:** Mohammadreza Jamalifard `[一作]` (University of Essex), Javier Andreu-Perez `[通讯]` (University of Essex)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了 Factor-Informed Uncertainty Distillation (FIUD)，一种教师-学生框架，将可解释的图像质量因子映射到眼球估计误差，并在不需要多次前向传播的情况下实现单次前向传播的不确定性估计。

**💡 创新点**

创新点在于利用梯度提升的教师模型从可解释的光照、锐度、眼部可见度等因素预测预期误差，并通过蒸馏和排序监督将其传递给轻量级学生网络，使不确定性与可操作的失败模式对齐。

**🔧 技术方法**

主要技术包括梯度提升回归、ResNet‑18 学生网络、单向传播的异方差回归不确定性头、蒸馏损失、排序损失和阶段式学习策略。

**📊 数据集**

使用了 ETH‑XGaze、Gaze360 和 MPIIFaceGaze 三个公开数据集，其中 Gaze360 为野外极端姿态、ETH‑XGaze 为受控高分辨率、MPIIFaceGaze 为摄像头图像。

**📈 对比分析**

与 Deep Ensembles、MC Dropout 和 Heteroscedastic Regression 等基线相比，FIUD 在 Spearman 相关性和 AUSE 指标上取得显著提升，尤其在 Gaze360 上分别提升约 0.24（相关性）和 1.5 度（AUSE）。

**⚠️ 局限性**

主要局限在于教师模型仅能捕捉通过预定义因子可解释的误差来源，跨人群误差难以建模；训练依赖成功的特征检测，可能忽略完全失效的样本。

---

## 315. PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning

**arXiv ID:** 2607.20064 | [PDF](https://arxiv.org/pdf/2607.20064v1)

**作者:** Alexis Fox `[一作]` (Duke University), Bhuwan Dhingra `[通讯]` (Duke University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了PRO-LONG框架，在长序列探索任务中通过完整无损日志写入与编程搜索实现记忆与查询，显著提升LLM代理在ARC‑AGI‑3等连续学习游戏的表现。

**💡 创新点**

创新点在于将完整、无损日志写入并利用编码代理的程序化搜索完成高效查询，保持极简、无压缩且高度兼容编码代理；实现了在长序列任务中真正的“程序化记忆”。

**🔧 技术方法**

使用了编码代理（Python工具调用与代码执行）、正则/grep搜索、日志结构化写入与程序化检索，以及LLM的多轮工具调用与代码生成。

**📊 数据集**

主要使用ARC‑AGI‑3公开游戏集（25个交互式游戏），并在该基准上进行评估。

**📈 对比分析**

与最强公共基线（WorldModeler、Arcgentica、Schema）对比，PRO‑LONG在保持4.2–5.8倍更少token成本的同时，pass@1约为41–82%（与最强基线相当或略优），best@k进一步提升；在大多数游戏中实现了与或超过现有最优性能。

**⚠️ 局限性**

仍受限于游戏间高变异性与运行方差大，需要多次重复实验；对极长日志的搜索效率与不同模型的通用性仍待进一步验证。

---

## 316. Zero-Shot Heart Rate Variability Forecasting from Consumer Wearables Using Time Series Foundation Models

**arXiv ID:** 2607.20027 | [PDF](https://arxiv.org/pdf/2607.20027v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 317. Solar Open 2 Technical Report

**arXiv ID:** 2607.20062 | [PDF](https://arxiv.org/pdf/2607.20062v1)

**作者:** Sungrae Park `[一作]`, Alice Oh `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个 250B 参数的韩国主权 AI 模型，采用混合注意力与多专家（MoE）架构，支持 1M token 的长上下文并专注于长程代理任务。

**💡 创新点**

创新点包括混合软/线性注意力栈、无位置编码与负特征值扩展、仅保留 2.3% 原始权重的选择性迁移，以及基于价值与稀缺度的高质量数据策划与分层预训练。

**🔧 技术方法**

技术手段涵盖 GQA 与 KDA 的混合注意力、sigmoid 输出门、负特征值扩展、全词汇量多教师在线蒸馏、完全异步强化学习以及 320B+ 的 MoE 训练框架。

**📊 数据集**

使用从 20T token 的全球语料中精炼出的 10T 高质量混合数据，并辅以内部生成的 OfficeVerse、Ko‑GDPval 等自定义任务数据，涵盖多语言、多领域（编程、对话、办公等）。

**📈 对比分析**

通过内部基准与公开模型（DeepSeek‑V4‑Flash、MiMo‑V2.5 等）及封闭 API 的对比，模型在英语基准上与顶尖开源模型相当，在韩语基准中以平均分领先所有对比模型，在办公室工作基准 Ko‑GDPval 上仅比 1.6T 参数的 DeepSeek‑V4‑Pro 低 0.1 分。

**⚠️ 局限性**

仍存在仓库/终端级软件工程、精确数值计算等领域的性能不足，需加强验证与自检机制，以及提升特定领域任务的精准度与鲁棒性。

---

## 318. ReferTrack: Referring Then Tracking for Embodied Visual Tracking

**arXiv ID:** 2607.20061 | [PDF](https://arxiv.org/pdf/2607.20061v1)

**作者:** Hanjing Ye `[一作]` (SUSTech), Hong Zhang `[通讯]` (SUSTech)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种“先定位后跟踪”的框架（ReferTrack），利用单向前置摄像头完成目标描述式视觉跟踪，先通过索引化的边界框候选集选择目标，再基于该选择预测轨迹。

**💡 创新点**

创新点包括：①把目标识别转化为单token的索引选择（Refer‑CoT），显著降低抽象空间的监督难度；②引入 Temporal‑Viewpoint‑Bbox Indicator (TVBI) 令历史目标几何信息直接注入视觉序列；③在离线阶段通过自制 Refer‑QA 数据集进行联合监督，提升识别精度并实现“识别→规划”的闭环。

**🔧 技术方法**

技术细节：融合 SigLIP 与 DINOv2 的双编码器特征；使用 YOLO11+ByteTrack 生成候选边界框；对图像、候选框、语言指令进行 MLP 投影并拼接至 Qwen3‑4B LLM；采用 TVBI tokens 维持滑动窗口记忆；通过全微调（SFT）结合三项损失（轨迹、定位、文本）训练。

**📊 数据集**

主要数据集：EVT‑Bench（1.3M 轨迹样本）、自制 Refer‑QA（基于 SYNTH‑PEDES/Person ReID，1.3M 问答样本）、Habitat 3.0 仿真环境，及真实单摄像头机器人部署（Unitree Go2、G1）。

**📈 对比分析**

在单视角评测上与 TrackVLA++、VLingNav 等基线对比，ReferTrack 在 STT、DT、AT 三个子任务上取得 SR/ TR/ CR 89.4/92.5/1.6、73.3/81.8/7.6、74.1/85.7/7.7，显著优于同类单摄像头方法，并匹配或超越部分多摄像头基线，表明显式图像空间定位和历史记忆能弥补摄像头覆盖不足。

**⚠️ 局限性**

局限性：①仍依赖前端检测与候选框质量；检测漏检或误检会直接影响定位；②历史记忆错误累计可能导致漂移；③仅在单摄像头和特定仿真/真实场景验证，未在更复杂多摄像头或不同传感器组合下测试；④未结合 RL 微调，可能在极端动态环境中进一步提升。

---

## 319. A Systematic Benchmark of Intensity Normalisation Methods for 3D Knee MRI Segmentation and Cross-Domain Generalisability

**arXiv ID:** 2607.20028 | [PDF](https://arxiv.org/pdf/2607.20028v1)

**作者:** Oliver Mills `[一作]` (University of Leeds), Samuel Relton `[通讯]` (University of Leeds)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对膝关节MRI的3D半月板分割任务，系统比较了七种不同的强度归一化方法，并评估其在内部（IWOAI 2019）和外部（SKM-TEA）数据集上的表现和泛化能力。

**💡 创新点**

首次在肌肉骨骼影像分割中对传统归一化技术进行全面对比，采用线性混合效应模型量化跨域性能差异，并证明归一化对跨域泛化的影响相对有限，提示需要更强的域适应策略。

**🔧 技术方法**

使用nnU-Net框架训练3D U-Net模型；七种归一化方法包括Z-score、Min-max、Robust Min-max、HE、CLAHE、Nyúl histogram matching 与 GMM；评估使用Dice、Hausdorff Distance、CCC、MAE，并通过混合效应模型和Tukey后验检验进行统计比较。

**📊 数据集**

IWOAI 2019（训练/内部测试，88名患者的3D DESS MRI）和SKM-TEA（外部测试，155名患者的qDESS MRI）两套公开数据集。

**📈 对比分析**

在内部测试中，各方法差异极小；在外部测试中，Nyúl、Z-score和CLAHE略优，平均Dice提升约1%，但所有方法与内部测试相比下降约10%；归一化方法仅解释了约0.5%的方差。

**⚠️ 局限性**

归一化方法无法显著缓解域偏移，跨域性能差距仍大；未考虑病理差异和标注者差异的影响；缺乏与先进的域适应或协方差校正技术（如ComBat、对抗式域适应）的对比。

---

## 320. TalentCLEF at CLEF2026: Skill and Job Title Intelligence for Human Capital Management

**arXiv ID:** 2607.20009 | [PDF](https://arxiv.org/pdf/2607.20009v1)

**作者:** Luis Gasco `[一作]` (Avature Machine Learning), Rabih Zbib `[通讯]` (Avature Machine Learning)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文介绍了第二届 TalentCLEF 挑战赛，提供两项任务：任务 A（基于上下文的职位-候选人匹配）和任务 B（职位-技能匹配及技能类型分类），并阐述评测方法与公平性考量。

**💡 创新点**

创新点包括：①使用合成但真实感强的职位描述与简历进行多语种上下文匹配；②在任务 B 中加入技能类型（专属/横向）标签；③在评测中加入性别公平性指标（RBO），支持单语与跨语评估。

**🔧 技术方法**

技术手段：大语言模型（LLM）微调与提示工程、信息抽取、表征学习、数据增强、特征融合；评测使用标准 IR 指标（MAP、MRR、Precision@K）及公平性指标 RBO。

**📊 数据集**

数据集：任务 A 共 100 条职位描述与 300 条合成简历；任务 B 训练集 5,000 个职位标题与对应技能，验证集 200 条（已标注技能类型），测试集 500 条职位标题；全部数据为多语种（英、西）。

**📈 对比分析**

方法比较采用 Codabench 平台，按 MAP 作为官方评测指标；同时报告 MRR、Precision@K 与 RBO。前两名单语团队、最佳跨语团队以及最佳公平性模型将获得奖励；任务 B 亦颁发两名最佳团队证书。

**⚠️ 局限性**

限制：①依赖合成数据，真实场景中可用性需进一步验证；②目前仅覆盖英、西两种语言，其他语言支持有限；③公平性评测仅考量性别偏差，其他偏差维度未覆盖；④缺乏对模型可解释性与部署成本的讨论。

---

## 321. A SAT-Based Exact Approach for Radio k-Labeling

**arXiv ID:** 2607.19997 | [PDF](https://arxiv.org/pdf/2607.19997v1)

**作者:** Huong Vu Thanh `[一作]` (Vietnam Maritime University), Khanh To Van `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于SAT的精确求解框架，针对无线电k标签问题结合紧凑的顺序编码与增量SAT求解；

**💡 创新点**

创新点在于重用学习的子句避免重建公式、引入旋转对称性破坏、在多种图族上获得38个新的最优标号并实现109个实例的最优证明；

**🔧 技术方法**

使用的技术包括直接与顺序编码的SAT模型、增量SAT求解、对称性破坏技术，以及与ILP（CPLEX/Gurobi）联合验证；

**📊 数据集**

使用的实验数据集为146个基准实例，覆盖路径、环、三角蛇、kC4/6蛇、梯形、书图、友谊图、二项树等九个图族；

**📈 对比分析**

通过与已有启发式（Ub_1、Ub_3）及ILP（CPLEX/Gurobi）对比，Order‑SAT在绝大多数实例中获得最优或更紧的上界，平均排名第二，仅次于已知最优值，并在直径增长族中显著优于ILP，在直径固定族中ILP更快，两者互补；

**⚠️ 局限性**

局限在于当标签域大或直径高导致变量/子句数爆炸时，Order‑SAT可能超时或耗尽内存，需进一步改进对称性破坏或分支定界策略。

---

## 322. GaussianSeed: Hierarchical Gaussian Seeding for High-Resolution 3D Occupancy Prediction

**arXiv ID:** 2607.20071 | [PDF](https://arxiv.org/pdf/2607.20071v1)

**作者:** Xinzhuo Li `[一作]` (Tongji University), Qijun Chen `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为GaussianSeed的多尺度高斯占位框架，用于实现高分辨率（0.1 m）3D语义占位预测。

**💡 创新点**

创新点包括：① Regression‑Based Gaussian Initialization (RBGI) 用于无深度先验的高斯中心回归；② Hierarchical Gaussian Seed Devolution (HGSD) 的粗细层次化迭代细化；③ Gaussian Seed Parameter Encoder (GSPE) 的跨层特征残差传递；④ 结合Chamfer Distance、Lovász、NLL和BCE的多任务损失。

**🔧 技术方法**

技术手段包括：多视角相机特征提取与跨视角注意力、3D高斯渲染到体素空间、差分体素转换、可微的概率叠加、粗到细的高斯参数递归预测。

**📊 数据集**

使用了Occ3D‑nuScenes（0.4 m）和自行构建的TJScenes（6摄像头全景，0.1 m）两个数据集，后者覆盖校园侧道和非道路场景。

**📈 对比分析**

与稠密和稀疏占位基线（BEVFormer、TPVFormer、OPUS、GaussianFormer等）对比，GaussianSeed在Occ3D‑nuScenes获得约12.4 % mIoU，TJScenes得到约34.1 % mIoU；同时推理延迟仅为41 ms，显著低于同类方法，展示了效率‑质量的新前沿。

**⚠️ 局限性**

局限性在于：对极端光照或遮挡条件下的鲁棒性仍有限；动态物体的细节捕捉偶有轻微误差；整体仍依赖相机标定和多摄像头同步，对单摄像头或无标定环境的迁移性需要进一步验证。

---

## 323. Test Case Prioritization for DNNs via Neural Collapse Instability

**arXiv ID:** 2607.20046 | [PDF](https://arxiv.org/pdf/2607.20046v1)

**作者:** Chunyu Liu `[一作]` (Beijing University of Posts and Telecommunications), Su-Juan Qin `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为NCIP的无标签错误优先级排序方法，利用深度网络训练终结阶段的神经坍塌几何特征以及跨检查点的预测不稳定性来对测试样本进行排序。

**💡 创新点**

创新点在于将神经坍塌（NC）中的权重等角性指标用于选取代表性训练检查点，并通过这些检查点间的预测波动（TVD）与最终模型的置信度边界（margin）相结合，提供了一种既考虑全局几何稳定性又兼顾边界不确定性的优先级评分机制。

**🔧 技术方法**

主要技术包括：1) 计算权重向量的余弦相似度标准差来衡量NC等角性；2) 采用多检查点软最大化输出的总变差（TVD）衡量预测不稳定性；3) 在最终检查点计算类别置信度差距（margin）并与TVD进行标准化组合；4) 通过一次性多检查点加载和推理完成样本排序。

**📊 数据集**

实验涵盖了七个图像与文本分类数据集（MNIST、Fashion‑MNIST、CIFAR‑10/100、TinyImageNet、IMDB、AGNEWS）以及LLM生成任务的WikiText；同时使用了多种网络架构（LeNet、ResNet、VGG、DenseNet、Transformer）并在这些模型上评估。

**📈 对比分析**

与多种基准方法（DeepGini、Entropy、PCS、MSP、Dropout、EffiMAP、DSA/LSA、NNS、TDPR、SETS等）对比，NCIP在RAUC‑ALL与RAUC‑500指标上均获得最高或相近的性能，在清洗数据上平均提升约1.5/4.9，且在对抗样本、模型校准及LLM任务中表现稳健，且在大多数数据集‑模型组合上实现了最佳平均RAUC。

**⚠️ 局限性**

局限性在于：①需要保存并加载多份训练检查点，导致额外的存储与推理开销；②仅适用于可访问完整训练轨迹或微调检查点的场景，对缺失检查点的部署模型不适用；③在极大模型上多检查点推理仍会影响实时性，需要进一步优化检查点数量或采用更高效的计算策略。

---

## 324. On Runlength Limited Codes for BICM Systems

**arXiv ID:** 2607.20026 | [PDF](https://arxiv.org/pdf/2607.20026v1)

**作者:** Stephan Zeitz `[一作]` (Technische Universität Dresden), Gerhard Fettweis `[通讯]` (Technische Universität Dresden)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在BICM系统中使用RLL码（尤其是TSRLL码）时的符号映射（assignment scheme）对系统性能的影响，并提出了针对低SNR下的解析优化方法。

**💡 创新点**

创新点在于把RLL码的符号映射优化问题表述为一个求解qap的形式，并利用极限大块长度下的线性投影得到可行的投影矩阵，进而实现对状态独立RLL码和两状态TSRLL码的assignment scheme优化；同时展示了该优化在实际带有块间干扰的ZXM系统中的显著性能提升。

**🔧 技术方法**

主要技术包括：BICM容量的一阶Maclaurin展开、投影矩阵的谱分解与Toeplitz/循环矩阵近似、匈牙利算法求解线性分配问题、辅助信道下的互信息下界估计，以及软输入-软输出RLL译码器。

**📊 数据集**

使用的“数据集”是仿真产生的RLL码字序列与相应的ZXM系统下的1‑bit量化观测向量，覆盖不同的d约束、码率与OTFS/FFT参数，并在10⁵符号级别上重复多次仿真以统计性能。

**📈 对比分析**

通过与传统的最大熵RLL序列（上界）和现有的MS‑RLL码的下界进行比较，结果表明在低至中等SNR区间可实现3–8 dB的收益，且在高SNR区间能达到码率限制的极限。

**⚠️ 局限性**

局限性包括：优化仅在极限低SNR和无限块长度下推导，实际系统中存在块间干扰和有限块长，导致理论最优与实际可行之间存在偏差；此外，状态依赖的MS‑RLL码由于状态数大，优化仍较困难。

---

## 325. Layer-Wise Decision Fusion for Fake Audio Detection Using XLS-R

**arXiv ID:** 2607.20023 | [PDF](https://arxiv.org/pdf/2607.20023v1)

**作者:** Yixuan Xiao `[一作]` (University of Stuttgart), Ngoc Thang Vu `[通讯]` (University of Stuttgart)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种层级决策融合方法，用每层特征单独做判别后再融合，提升假音检测的跨域性能。

**💡 创新点**

创新点在于把决策融合置于特征提取层级之后，避免特征坍塌并实现更透明的层级分析；并引入瓶颈投影、重建损失及可学习层权重来进一步提升性能。

**🔧 技术方法**

采用冻结的XLS‑R大规模语音模型提取多层隐藏特征，使用一类Softmax分类器、时间池化、注意力机制以及自编码器重建正则。

**📊 数据集**

使用ASVspoof19 LA作为训练集，评估在同域的ASVspoof19 LA测试集和跨域的In‑the‑Wild (ITW) 数据集。

**📈 对比分析**

与单层特征、特征融合以及现有基线（NN‑ASP、NN‑ACP）对比，层级决策融合在ITW上的EER达6.90%（标准差0.30%），显著优于其它方法。

**⚠️ 局限性**

限制在于对深层隐藏层的特征重建和层权重学习在跨域测试中可能导致过拟合；包含静音段训练时对跨域性能影响较大，且对重要离散 token 的解释性仍不够明确。

---

## 326. Taming the Security-Energy Paradox: A Green AI Approach to Optimized Android Malware Detection

**arXiv ID:** 2607.20003 | [PDF](https://arxiv.org/pdf/2607.20003v1)

**作者:** Shrinidhi Sridhar `[一作]` (MIE-SPPU Institute of Higher Education), Vikas K. Malviya `[通讯]` (MIE-SPPU Institute of Higher Education)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在Android设备上部署深度学习恶意软件检测模型时的安全与能耗权衡，并寻找最佳配置。

**💡 创新点**

通过比较FP32与INT8量化的多层感知机模型深度，发现浅层量化网络既保持≈99%准确率，又显著降低能耗。

**🔧 技术方法**

后训练量化(PTQ)、TensorFlow Lite、Android虚拟设备仿真及能耗估算公式。

**📊 数据集**

使用TUANDROMD与DREBIN行为与静态特征数据集。

**📈 对比分析**

对模型精度、召回率、吞吐量、模型大小与能耗等指标进行10次平均比较；结果显示3层INT8模型在TUANDROMD上能耗0.0189 mJ、准确率99.24%；4层INT8模型在DREBIN上能耗0.0231 mJ、准确率98.63%。

**⚠️ 局限性**

仅在Android虚拟机上评估，未在真实设备或NPU上验证；量化对置信度校准与模型可靠性的影响未深入探讨。

---

## 327. Toward Seasonal Guidelines for Robust Deep-Learning Sentinel-2 Building Detection in Different Area Types

**arXiv ID:** 2607.19994 | [PDF](https://arxiv.org/pdf/2607.19994v1)

**作者:** Michał Romaszewski `[一作]` (Polish Academy of Sciences), Szymon Sala `[通讯]` (Polish Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一套基于Sentinel‑2影像的建筑物检测框架，系统性地研究了季节变化与不同城市结构对模型性能的影响；

**💡 创新点**

首次将多时相Sentinel‑2数据与不同建成区类型结合，提出按季节选择模型和“集体夏季模型”的操作性指导；

**🔧 技术方法**

采用U‑Net与DeepLabV3+两种卷积语义分割网络，冻结编码器仅训练解码器，并使用Cosine Annealing学习率调度；

**📊 数据集**

使用波兰华沙地区65幅2019‑2025年低云Sentinel‑2 L1C/L2A影像与BDOT10k建筑矢量数据库生成的二值真值掩模；

**📈 对比分析**

通过十折交叉验证与跨时相推理比较，发现U‑Net优于DeepLabV3+，L2A略胜L1C；最佳月度模型在5‑8月表现最佳，IoU最高约0.43；在不同建成区类型中，工业/高密度区IoU>0.5，郊区/农村区IoU<0.35；

**⚠️ 局限性**

局限性包括仅覆盖华沙地区、只评估两种网络架构、每月仅单一采样导致模型对极端季节如雪季不稳，且未对不同气候带或城市形态的泛化进行验证；

---

## 328. What Does the Credential Still Certify? Cognitive Stewardship for AI-Mediated Education

**arXiv ID:** 2607.19988 | [PDF](https://arxiv.org/pdf/2607.19988v1)

**作者:** Kai Yao `[一作]` `[通讯]` (University of Edinburgh), Kai Yao (University of Edinburgh)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出认知管护框架，用以规范生成式AI在教育评估中的委托与证据关系，并对30所大学的公开AI使用政策进行自动化审计，评估其对学习声明、委托边界、证据标准和保障措施的体现。

**💡 创新点**

创新点在于将“教育委托”概念拆解为学习声明、委托边界、证据标准和保障层四要素，形成可操作的政策检验框架，并首次系统性地用LLM对政策文本进行定量评分。

**🔧 技术方法**

技术手段为：使用四个开源大语言模型（open‑weight LLM）按预设评分代码书自动对政策文本进行标注，减少单模型偏差；同时设计情境测试以检验政策可操作性。

**📊 数据集**

数据集为：30所英美、英联邦高校公开的生成式AI评估相关政策文件（共计若干页），覆盖英国、澳洲、新西兰、加拿大和美国五个英语系统。

**📈 对比分析**

对比方法为：计算每个政策在学习声明、委托边界、证据标准以及八项保障上的得分；结果显示大多数政策在边界阈值高于证据得分，且大部分保障措施稀缺；未进行传统模型性能评估，而是通过分数分布展示差距。

**⚠️ 局限性**

局限性包括：评分完全依赖LLM，缺乏人类编码验证；语料仅限公开英文政策，未涵盖课堂实践或内部文件；结果仅反映政策可见度，未评估对学习成果的实际影响。

---

## 329. Coordinating from Memory: Graph-Structured Experience Reuse for Multi-Agent Adaptation in Dynamic Manufacturing

**arXiv ID:** 2607.19985 | [PDF](https://arxiv.org/pdf/2607.19985v1)

**作者:** Chengxiao Dai `[一作]` (University of Sydney), Luyan Zhang `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图结构的经验记忆框架 GSEM，用于多智能体在动态制造环境中的协调与快速适应。

**💡 创新点**

创新点在于将恢复经验编码为异构关系图，并通过图神经网络检索相似案例，实现经验驱动的策略调整，而非从零学习。

**🔧 技术方法**

使用图卷积网络（R‑GCN）进行状态编码、图相似度检索、注意力融合与门控机制，以及对比学习以提升记忆区分度。

**📊 数据集**

在动态柔性车间排程（DFJSP）10×10、5智能体等基准实例上进行评测，采用多种扰动类型。

**📈 对比分析**

与传统调度规则、集中式 PPO、QMIX、MAPPO 以及记忆增强的 MARL 基线比较，GSEM 在单/混合扰动下平均缩短约 4–10% 产成品周期，适应速度提升约 33–38%。

**⚠️ 局限性**

限制包括额外的计算开销（≈18% 推理时间、31% 训练时间）以及记忆池容量和检索精度对性能的敏感性。

---

## 330. Time Series Network Utilization KPI Forecasting Using Advanced AI/ML Models

**arXiv ID:** 2607.19974 | [PDF](https://arxiv.org/pdf/2607.19974v1)

**作者:** Niraj Gadhe `[一作]`, Vinay Saini `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文比较了多种传统机器学习与深度学习模型对网络接口带宽利用率的时序预测性能。

**💡 创新点**

创新点在于将树模型、支持向量回归、Prophet 与多种 LSTM 架构在统一数据集上进行系统评测，并分析了准确性与计算效率的权衡。

**🔧 技术方法**

使用了随机森林、XGBoost、SVR、Prophet、ConvLSTM、Vanilla LSTM、Stacked LSTM、Bi‑LSTM 等技术，并采用 Python 生态的 Statsmodels、Scikit‑learn、XGBoost、Prophet、TensorFlow/Keras。

**📊 数据集**

数据集为四个网络接口的 15 分钟间隔的接口利用率（Interface_Utilization）共约 4 万条记录。

**📈 对比分析**

通过 MAPE、NRMSE、R²、训练时间与预测时间进行比较，结果显示 XGBoost 与随机森林在准确性（MAPE<5%）与速度方面表现最佳，深度学习模型在准确率更高但训练成本更高。

**⚠️ 局限性**

局限性包括未对多网络多变量场景进行评估，Prophet 未调参，且深度模型对资源消耗较大，难以在低配环境实时部署。

---

## 331. On Optimization Complexity of Second-Order Certified Unlearning

**arXiv ID:** 2607.20192 | [PDF](https://arxiv.org/pdf/2607.20192v1)

**作者:** Nikita Doikov `[一作]` (Cornell University), Anastasia Koloskova `[通讯]` (University of Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了机器学习中的数据遗忘问题，特别是从优化角度探讨了认证遗忘的算法复杂性。

**💡 创新点**

提出了一种新的二阶遗忘算法，具有优越的全局收敛性，并证明了在特定条件下的快速收敛率，展示了二阶信息的优势。

**🔧 技术方法**

使用了二阶优化方法和各向异性高斯机制，结合了牛顿法的全局收敛性。

**📊 数据集**

应用于逻辑回归和指数回归等线性模型，假设数据集为均匀凸正则化的情况。

**📈 对比分析**

与一阶遗忘方法相比，二阶方法在优化性能上表现出显著的优势，尤其是在处理移除数据时的简单性和有效性。

**⚠️ 局限性**

算法的限制在于对模型正则化的假设，以及在某些情况下可能需要较高的计算资源。

---

## 332. A Machine-checked Proof of Consistency for Impredicative Pure Type Systems

**arXiv ID:** 2607.20188 | [PDF](https://arxiv.org/pdf/2607.20188v1)

**作者:** Sebastián Urciuoli `[一作]` `[通讯]` (Universidad ORT Uruguay), Sebastián Urciuoli (Universidad ORT Uruguay)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在 Agda 中实现了基于 Stoughton 多重替换的纯类型系统（Pure Type Systems）的核心元理论证明，包括 β 归约的可合并性、类型系统的主体归约和空上下文的一致性。

**💡 创新点**

创新点在于将 α 可交换关系与五边形性质相结合，并采用 Takahashi 修订的 Tait‑Martin‑Löf 证明，避免了传统的 α‑转换步骤，显著简化证明结构且保持了经典语法的可读性。

**🔧 技术方法**

使用了 Agda、Stoughton 多重替换、α‑可交换关系、并行归约、Takahashi 星号归约等技术。

**📊 数据集**

论文未使用外部数据集，全部在 Agda 内部完成机器验证。

**📈 对比分析**

通过对比基于 de Bruijn 索引或局部无名语法的 Coq 实现，本文的代码量约 4300 行，略高于 2900 行的 dBI 版本，证明透明度更高、成本低，整体性能与现有实现相当。

**⚠️ 局限性**

局限性包括：归一化假设仍未在 Agda 中实现，无法证明强归一化，且对不可归一化的 impredicative PTS 缺乏完整的形式化，系统对这类系统的可证明性仍受限。

---

## 333. Real-Time EEG Cap Electrode Detection for Guided Point-of-Care Placement

**arXiv ID:** 2607.20142 | [PDF](https://arxiv.org/pdf/2607.20142v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 334. Robust Classification in ML: A Topological Semantics Approach

**arXiv ID:** 2607.20185 | [PDF](https://arxiv.org/pdf/2607.20185v1)

**作者:** Dominik Pichler `[一作]` (TU Wien), Mirko Tagliaferri `[通讯]` (TU Wien)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于拓扑语义的逻辑框架，用以形式化机器学习模型对输入扰动的鲁棒分类行为，并通过引入鲁棒模态与鲁棒敏感条件连结器实现鲁棒属性与其它分类属性的系统关联。

**💡 创新点**

创新点在于：①将鲁棒性解释为开集内部真值的保持；②设计了新的条件连结器 φψ，该连结器在满足前件鲁棒性且其鲁棒内部被后件全包容时才为真；③给出了该逻辑的完整公理化体系；④构造了“最小鲁棒模型”机制，能够根据用户给出的鲁棒子集自动生成最小拓扑空间。

**🔧 技术方法**

使用的技术主要是模态逻辑与拓扑语义（S4），结合条件逻辑和公理化证明方法（归纳、典范模型构造），以及对开集基的数学构造来实现最小拓扑。

**📊 数据集**

在实验说明中采用 MNIST 手写数字数据集作为例子，手动标注了 8、7 以及“两圆”“直线”等属性的鲁棒子集。

**📈 对比分析**

论文并未给出算法实现或实验评测，仅通过案例示例展示逻辑语义的解释力；因而无法提供传统意义上的性能对比或数值指标。

**⚠️ 局限性**

主要限制包括：①缺乏对模型检查与可满足性问题的复杂度分析；②未给出在具体欧氏空间上的完整性或表示结果；③未引入度量或概率以量化鲁棒性；④仅在示例层面验证，缺乏大规模数据或多模型实证研究。

---

## 335. Multi-stage Dynamic Selection for Cross-Project Defect Prediction

**arXiv ID:** 2607.20151 | [PDF](https://arxiv.org/pdf/2607.20151v1)

**作者:** Juscimara G. Avelino `[一作]` (Universidade Federal de Pernambuco), Rafael M. O. Cruz `[通讯]` (University of Quebec)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Multi‑DES——一种多阶段动态集成选择框架，用于跨项目缺陷预测，只使用源项目数据，在目标项目预测时按实例动态选择最佳分类器。

**💡 创新点**

创新点包括：① 在项目层面通过聚合排名最小化（ARM）选取最佳配置；② 在实例层面采用动态选择技术实现模块级模型自适应；③ 完全基于源数据，无需目标项目信息，避免信息泄露。

**🔧 技术方法**

技术：基于 Bagging 生成多样化分类器池（DT、LR、NB、RF）；八种动态选择方法（KNU、KNE、KNOP、META‑DES、OLA、LCA、MCB、Rank）；聚合排名最小化（ARM）做配置选择；留一项目交叉验证、F1、AUC、False Alarm 评估。

**📊 数据集**

数据集：PROMISE（62 版本）、RELINK（3 项目）、NASA MDP（12 项目）和 AEEEM（5 项目），共 82 个项目。

**📈 对比分析**

与 CamargoCruz09‑DT、Turhan09‑DT、Menzies11‑RF、Watanabe08‑DT、EASC‑NB 等四大 CPDP 方法进行 Wilcoxon 符号秩检验比较。Multi‑DES 在 AUC、False Alarm 上往往获得最佳或近乎最佳结果，F1 得分在 AEEEM 与 RELINK 上领先；总体上相较基线赢得超过 70% 的比较场景，尤其在 AUC 与 False Alarm 上表现显著。

**⚠️ 局限性**

局限性：① 需要遍历大量配置（4×8×10），计算成本高；② 仅在离线场景下评估，尚未验证在线/增量学习能力；③ 在 PROMISE 数据集的 False Alarm 指标上略逊，说明在某些项目分布差异大时仍有提升空间。

---

## 336. Data Annotations as Pedagogical Hints: From Subjective Labels to Critical Thinking

**arXiv ID:** 2607.20149 | [PDF](https://arxiv.org/pdf/2607.20149v1)

**作者:** Ralf Raumanns `[一作]` (Fontys University of Applied Science), Veronika Cheplygina `[通讯]` (IT University of Copenhagen)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在两所大学中，学生通过手工标注皮肤病变图像（如毛发覆盖程度），并以小组形式讨论，随后完成问卷调查，以此探究手工注释如何帮助学生理解数据主观性、偏见与公平性。

**💡 创新点**

提出将手工数据注释作为教学提示（pedagogical hints）的方法，突出注释的主体性与多样性，弥合研究视角与教学实践之间的差距；并提供针对教育者的设计建议，如保持足够的解释模糊度、减少重复劳动、明确处理情绪不适等。

**🔧 技术方法**

使用了 Label Studio 或 Jupyter Notebook 等标注工具，结合问卷调查（SoSci Survey）收集学生反馈，并在部分案例中用简单模型训练验证注释对模型表现的影响。

**📊 数据集**

使用公开的皮肤病变图像数据集（如 ISIC 或类似的医学影像集合），学生为每幅图像标注 0‑1‑2 的毛发覆盖等级。

**📈 对比分析**

通过比较手工注释任务与传统讲座的自我报告熟悉度（subjectivity、bias、fairness 等概念）来评估教学效果。结果显示，手工注释显著提高了学生对偏见与公平性的理解；但缺乏客观性能指标（如模型准确率）与对比实验。

**⚠️ 局限性**

局限性包括样本量仅 43 人、先后测量的主观回顾性评估、不同机构使用不同标注工具导致可重复性受限，以及研究仅覆盖两所大学，未系统分析群体构成与教学设计差异。

---

## 337. CUSUM-Shaped Inference-Time Monitoring and Targeted Re-Decoding for Quantized Small Language Model Reasoning

**arXiv ID:** 2607.20129 | [PDF](https://arxiv.org/pdf/2607.20129v1)

**作者:** El Hassane Ettifouri `[一作]` (Novelis Research), Walid Dahhane `[通讯]` (Novelis Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一个基于重叠窗口、位置校准与CUSUM形状累积的监控–回滚控制器，用于在4‑bit量化的小型推理模型（1.5B Distilled Reasoning）在MATH‑500数学推理任务中的推理轨迹检测与修复。

**💡 创新点**

创新点在于：① 证明中心化token log‑prob 作为自一致性 martingale 并不能捕捉轨迹衰退；② 将多维特征（熵、log‑prob、重复n‑gram、置信提升、局部熵变）结合到重叠窗口中；③ 用经验尾概率进行位置校准并构造beta‑mixture下注；④ 通过 CUSUM 形状重置估计回滚点并恢复 token、KV‑cache 与监控状态；⑤ 在受限重解码阶段应用低温、重复惩罚和 n‑gram 屏蔽。

**🔧 技术方法**

技术细节包括预采样熵、token log‑prob、重复检测、置信提升、局部熵差、经验尾概率映射、beta‑mixture下注、CUSUM 重置、温度/重复惩罚、n‑gram 阻塞等；实现无额外模型前向传播，所有计算在已有 logits 上完成。

**📊 数据集**

使用 MATH‑500（500 道数学推理题）作为评测集，模型为 1.5B Distilled Reasoning 在 4‑bit FP4 量化下的推理模型。

**📈 对比分析**

与 vanilla、随机回滚、周期回滚、重启、自我纠正等多种控制策略对比。 240 对时序审计集上准确率从 82/240 提升至 88/240 (+2.5pp，McNemar p=0.2632，bootstrap 区间 [-1.25,+6.25]pp，统计不显著)。 467 对探索集上提升 4.5pp（p=0.000753，bootstrap [+1.93,+7.07]pp），但受阈值选择偏倚。 与其他控制相比，该控制在同一预算条件下获得最高准确率。

**⚠️ 局限性**

局限性：① 缺乏 e‑process / Ville 级别的统计安全保证；② 阈值与校准使用相同数据，无法独立验证；③ 评测仅在单一量化模型和单一任务上完成，缺乏跨模型、跨数据集的通用性验证；④ 无真实退化标签或时间点，无法衡量报警精确率与延迟；⑤ 组件归因不完整，无法确定哪一子模块最为关键；⑥ 代码、环境与实验可复现性记录有限。

---

## 338. RIM: A Retrieval-In-Matching Framework for Cross-Domain Global Visual Localization of UAVs

**arXiv ID:** 2607.20116 | [PDF](https://arxiv.org/pdf/2607.20116v1)

**作者:** Xin Li `[一作]` (Chinese Academy of Sciences), Geng Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种共享表示的检索与匹配框架，结合UAV视角采样、两阶段跨域适配和局部描述器蒸馏，实现无人机全6-DoF视觉定位。

**💡 创新点**

创新点在于将DINOv2-B token同时用于全局检索和局部几何重排序，避免重复前端；并通过3D Tiles视角采样与两阶段跨域适配来降低视角与域差异。

**🔧 技术方法**

使用DINOv2-B、SALAD聚合头、DeDoDe本地描述器蒸馏、USAC-MAGSAC RANSAC、ALIKED+LightGlue/ELoFTR等匹配器，所有计算在单一前端完成。

**📊 数据集**

采用EPFL Urbanscape公开数据与自采的长安公园，分别利用Google 3D Tiles与SfM重建的mesh构建参考数据库。

**📈 对比分析**

与十类检索基线（HLoc、CLIP、CosPlace等）及多种匹配器进行零样本对比，Recall@1在EPFL上提升8.55/13.77pp，在Park提升4.45/8.94pp；端到端查询仅67.9 ms，比分离模型快1.8×，定位成功率100%。

**⚠️ 局限性**

局限在于依赖高质量3D网格覆盖，数据库存储量大；在覆盖稀疏或不完整地区性能下降，并需为每个场景单独重建Mesh。

---

## 339. From Dag-Like Proofs to Boolean Circuits in Lean

**arXiv ID:** 2607.20186 | [PDF](https://arxiv.org/pdf/2607.20186v1)

**作者:** Lorenzo Saraiva `[一作]` (Pontifícia Universidade Católica do Rio de Janeiro), Edward Hermann Haeusler `[通讯]` (Pontifícia Universidade Católica do Rio de Janeiro)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种将利用水平压缩得到的 DAG‑like 证明结构（DLDS）转化为布尔电路的算法，并在 Lean 证明助手中对该电路评估器与 DLDS 的有效性进行了机器检查。

**💡 创新点**

创新点包括：① 将 DLDS 的依赖传播验证抽象为统一的布尔函数；② 设计了 O(N³) 规模的统一网格电路，用路径分配符号来动态选择候选推理步骤；③ 在 Lean 中实现了从 DLDS 到电路的转换与点对点正确性定理，并为无压缩简单树子问题提供了桥接证明。

**🔧 技术方法**

使用技术主要有：水平压缩算法（HC）生成 DLDS；布尔电路构造与子电路模块（⊃I、⊃E、重复）；路径分配符号与路由逻辑；Lean 交互式定理证明与机器检查；递归 Flow 传播定义。

**📊 数据集**

本文未使用外部数据集；研究对象是理论上的 DLDS 与布尔电路构造，实验仅通过 Lean 证明验证。

**📈 对比分析**

性能分析：电路总规模为 O(N³)，其中 N 为公式空间大小；点对点正确性在 Lean 中已证明；全局接受性是对所有指数级路径分配的普遍量化，因而不作为可行的判断算法，只作语义规范；实验结果未给出具体运行时间。

**⚠️ 局限性**

局限性：① 只针对无压缩的简单树 DLDS 建立了桥接；② 对完整压缩（包含 ancestor edges、颜色、残余路径）尚未完成；③ 仅提供了点对点正确性，缺乏对全局接受性的有效判定；④ 目前仍是经典验证，量子化方案留待后续研究。

---

## 340. A Logical 3-valued Semantics for Nondeterministic Choice

**arXiv ID:** 2607.20178 | [PDF](https://arxiv.org/pdf/2607.20178v1)

**作者:** Alessandro Aldini `[一作]` (University of Urbino), Gandolfo Vergottini `[通讯]` (University of Urbino)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种对并发、可非确定性系统中计算错误的逻辑形式化，核心是设计了一种三值对称非确定性析取连接词，并在此基础上给出了动态与静态语义下的推理系统；随后进一步将其精炼为五值确定性逻辑，以区分不同类型的错误。

**💡 创新点**

创新点在于：① 在非确定性矩阵框架下构造了完全对称的非确定性析取符号，解决了传统三值逻辑中因评估顺序导致的偏斜；② 通过推理系统实现了动态与静态非确定性语义的对照；③ 引入五值确定性系统，将“未知”错误细分为Kleene、McCarthy、Bochvar三类，从而在确定性语义中完整捕获错误的不同传播行为。

**🔧 技术方法**

使用的技术主要包括：非确定性矩阵（Nmatrix）语义、三值与五值逻辑表格、对称非确定性析取的定义、动态与静态评估的模型构造、推理系统（序列演算）的设计、完备性与可判定性证明、投影映射（从五值到三值）与结构化的证据构造。

**📊 数据集**

本研究为纯理论性工作，并未使用任何实验数据集；所有结果均通过数学证明获得。

**📈 对比分析**

方法比较主要在理论层面：通过对称非确定性析取与传统三值析取（Kleene、Bochvar、McCarthy）的对照，展示了语义的对称性和非确定性；在确定性五值系统中，通过投影映射证明其为三值系统的精细化；性能评估未涉及算法实现，故无实验性能数据。

**⚠️ 局限性**

局限性包括：① 研究停留在形式语义和推理系统层面，缺乏对实际并发程序或过程代数的实验验证；② 五值系统仍是理论抽象，未说明如何在具体编程语言或验证工具中实现；③ 对于更大值域或多元运算符的扩展尚未探讨；④ 仅关注错误传播的逻辑层面，未考虑资源消耗、时间复杂度等实际实现细节。

---

## 341. Audio-Zero: Label-Free Self-Evolution for Fine-Grained Audio Reasoning

**arXiv ID:** 2607.20166 | [PDF](https://arxiv.org/pdf/2607.20166v1)

**作者:** Siqian Tong `[一作]` (Chinese Academy of Sciences), Chengpeng Hao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Audio-Zero，一个无标签自演化框架，通过将无标签音频对比对转换为可验证的听觉自对弈游戏，来提升大音频语言模型的细粒度听觉感知与推理能力。

**💡 创新点**

创新点在于：①构建听觉自对弈结构，使模型在不依赖外部标签的情况下通过识别“奇异听者”得到可验证奖励；②在听觉线索生成与奇异听者定位之间交替使用 Group Relative Policy Optimization (GRPO)，形成双向自我强化循环；③设计内容效用与投票惩罚相结合的奖励机制，鼓励生成细粒度、可区分的线索。

**🔧 技术方法**

技术方法包括：Group Relative Policy Optimization 强化学习；自演化自对弈游戏框架；文本生成的听觉线索；规则基内容效用评分与投票惩罚；多轮交互的奖励交替优化。

**📊 数据集**

使用的数据：2k 对未标记的音频对比对来自 Audio‑alpaca；评估基准为 TREA（时间推理）、MMAU Test‑mini（通用音频理解）和 MMAR（深层音频推理）。

**📈 对比分析**

与标签依赖方法（R1‑AQA、Audio‑Thinker）和标签自由方法（AQA‑TTRL、Audio‑CoT）对比，Audio‑Zero 在 Qwen2‑Audio‑7B‑Instruct 与 Qwen2.5‑Omni‑7B 上均取得最高分，TREA 上提升约 7–10%，MMAR 约 7–10%，MMAU Test‑mini 约 3–8%。

**⚠️ 局限性**

局限性包括：依赖仅 2k 对比对的规模较小；游戏设计主要针对对比差异，可能不足以覆盖更复杂的多模态推理；缺乏对更长序列细节的可解释性和鲁棒性评估。

---

## 342. Extending GouDa: Generation of Universal Datasets with (and without) Errors for Data Quality Benchmarking

**arXiv ID:** 2607.20165 | [PDF](https://arxiv.org/pdf/2607.20165v1)

**作者:** Valerie Restat `[一作]` (FernUniversität in Hagen), Uta Störl `[通讯]` (FernUniversität in Hagen)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款名为 GouDa 的合成数据生成器，能够基于 JSON 模式自定义数据结构、插入可控错误，并同时生成错误版本与干净版本，同时支持多实体关系与 NoSQL 数据。

**💡 创新点**

主要创新在于：1）完全不依赖已有数据即可生成逼真多维度数据；2）提供丰富的错误类型与可配置错误率；3）支持多实体关系与 NoSQL；4）通过 GUI 简化使用；5）生成的合成数据可复现、可用于标准化基准。

**🔧 技术方法**

技术实现基于 Java/Spring Boot 后端、React 前端、Docker Compose 容器化；核心逻辑利用 json-data-generator 库扩展的生成器、错误生成器；使用 JSON Schema（GSD）描述数据结构并支持动态引用；支持直接写入 MongoDB，后续计划扩展其他数据库。

**📊 数据集**

以 Mahdavi 等的啤酒数据集为例，自行构建品牌、风格、尺寸、ABV 等自定义列表；同时使用全球制药公司生产现场数据分析错误类型；并利用自定义列表再现真实业务场景。

**📈 对比分析**

通过在生成的合成数据中注入预设错误率（如 5% 缺失、15% 单位错误、5% 派生错误），与手工清洗的真实数据进行比对，验证错误类型覆盖率与错误发现效果；性能方面可按需求调节数据规模，实现大规模批量生成，具体时延未给出。

**⚠️ 局限性**

局限性包括：目前仅支持单数据源错误（未实现多源合并错误）；缺少 MAR/MNAR 等高级缺失机制；数据库支持仍局限于 MongoDB；GUI 仍在初期阶段，功能与易用性待进一步完善。

---

## 343. Formal Foundations for Known Good Reliable Die Screening in Chiplet-Based AI Systems-on-Chip

**arXiv ID:** 2607.20141 | [PDF](https://arxiv.org/pdf/2607.20141v1)

**作者:** Prashanthi Metku `[一作]` (Qualcomm Technologies Inc), Chandra Gandu `[通讯]` (Qualcomm Technologies Inc)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Known Good Reliable Die（KGRD）筛选的完整理论框架，解决芯片片段（chiplet）集成后可靠性预测的缺口

**💡 创新点**

创新点包括：①基于贝叶斯推断的风险模型与可观测性偏差上界；②安全门控决策体系提供可证明的后装配失效概率保证；③Bayes最优阈值的五区分派规则；④受限闭环反馈机制保证安全不变性、模型不确定性递减与一致性收敛

**🔧 技术方法**

采用贝叶斯推断、逻辑回归风险模型、不确定性膨胀安全门、Bayes最优决策理论、受限MAP更新以及Monte Carlo仿真验证

**📊 数据集**

使用合成的4,000颗芯片样本，特征分布基于公开的芯片片段工艺参数进行校准

**📈 对比分析**

通过理论证明与仿真比较，安全门保证后装配失效率始终低于α_rel+ε_Δ；在不同α_rel阈值下可调节释放率；五区分派策略在成本权衡下实现最优；闭环反馈在20轮迭代中实现参数误差下降42%，安全合规率≥98%

**⚠️ 局限性**

限制包括：仅在合成数据上验证；逻辑回归模型表达能力有限；假设包装上下文已知且单芯片独立；不考虑多芯片相互依赖与动态上下文变化；需在真实工厂数据上进一步验证

---

## 344. Two-Step Occupation Coding

**arXiv ID:** 2607.20101 | [PDF](https://arxiv.org/pdf/2607.20101v1)

**作者:** Alexander M. Esser `[一作]` (University of Koblenz), Jens Dörpinghaus `[通讯]` (University of Koblenz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种两步职业编码框架，先用域特定的命名实体识别提取工作标题，再将识别出的标题映射到职业分类体系。

**💡 创新点**

创新之处在于将工作标题识别与职业分类分离，结合噪声感知训练(NAT)和多阶段微调，以及引入基于差距的置信度阈值来替代传统绝对阈值。

**🔧 技术方法**

技术上使用了德语BERT作为基础模型，采用NAT进行噪声注入训练，随后对NER和职业分类分别进行微调；职业分类阶段则使用TF‑IDF特征与SVM、XGBoost、逻辑回归三种传统分类器对比。

**📊 数据集**

使用了德国职业教育历史与当代文档（包含OCR噪声的东西德资料和网络爬取的现代文本），以及KldB 2010/2020版本的职业分类表，另外在JobBERT基准集上做了跨语言评测。

**📈 对比分析**

与单步编码方法对比，两步方案在精度上提升至57.26%（单步为46.94%），SVM在分类器比较中表现最佳；基于差距的置信度阈值略优于绝对阈值。

**⚠️ 局限性**

局限性包括仅使用传统分类器，缺乏对语义上下文的利用，且仅识别显式出现的职位，需进一步扩展为多语言和语义驱动的模型。

---

## 345. A Task Taxonomy for Edge and Trail Bundling

**arXiv ID:** 2607.20089 | [PDF](https://arxiv.org/pdf/2607.20089v1)

**作者:** Markus Wallinger `[一作]` (Technische Universität München), Stephen G. Kobourov `[通讯]` (Technische Universität München)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统分析102篇边/轨迹/并行坐标可视化中使用的捆绑技术，构建了一套基于“范围×动作”的任务分类体系，揭示捆绑既能启用聚合层次的任务，也会阻碍元素层次的任务；

**💡 创新点**

创新点在于①将捆绑视为产生新感知对象的技术，提出专门针对捆绑的任务维度；②基于文献编码生成四层范围（元素、捆绑、全局、多视图）与六类动作（验证、识别、表征、量化、比较、评估）的二维矩阵；③将捆绑质量指标与相应任务关联，标识评估空白；

**🔧 技术方法**

主要技术为：系统文献检索与汇编、开放式编码 + 归纳的任务标签集、LLM（Claude Sonnet 4.6）辅助筛选、人工复核、构建任务矩阵与分布表；

**📊 数据集**

使用的“数据集”为从已有捆绑技术综述及其引用文献中抽取的102篇论文组成的语料库，涵盖节点‑边图、轨迹集与并行坐标三种表示类型；

**📈 对比分析**

论文未进行算法性能对比实验，而是通过对44 %（49篇）包含捆绑任务的论文进行计数，呈现各任务在不同范围/动作维度的出现频率，说明哪些任务被关注、哪些被忽视；

**⚠️ 局限性**

限制包括：①对节点‑边图文献偏好导致结果不均衡；②LLM辅助筛选可能漏检隐含任务；③缺乏正式的交叉评审可靠性指标；④仅覆盖三种常见捆绑场景，未涵盖3D或非欧几里得布局等扩展。

---

## 346. OLEDLM: A Unified Language Model for OLED Molecular Design

**arXiv ID:** 2607.20194 | [PDF](https://arxiv.org/pdf/2607.20194v1)

**作者:** Fukang Wen `[一作]` (Tsinghua University), Pipi Hu `[通讯]` (Beijing Institute of Mathematical Sciences and Applications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一套基于 LLaMA 风格生成模型和 Group Relative Policy Optimization（GRPO）的逆向 OLED 分子设计框架，能够根据目标激发能 S₁ 与振荡强度 f 直接生成满足条件的 SMILES。

**💡 创新点**

创新点包括①首次将 LLaMA 变体用于 OLED 分子生成；②构建域适配的 BERT 属性预测器作为奖励模型；③采用无价值网络的 GRPO 强化学习方法，显著提升对特性控制的精度与生成多样性。

**🔧 技术方法**

使用技术包括：LLaMA‑style Transformer 生成器、BERT 结构预测器、SMILES 分词器、Masked Language Modeling 预训练、SFT 条件微调、GRPO 强化学习、奖励函数与 KL 正则化。

**📊 数据集**

数据集：9 M OLED 相关 SMILES 预训练集 + 10 k OLED 分子（S₁、f）通过 DFT 计算得到标签，进一步抽取 20 个样本用于独立 DFT 验证。

**📈 对比分析**

方法对比：与无条件生成、SFT（3 轮）、SFT（20 轮）以及 PPO 基线进行比较；在 10 k 采样中，GRPO 达到 98.3% 合法率，S₁ MAE 0.25 eV、f MAE 0.15 eV，novelty 94.4%，独立 DFT 验证 MAE 0.26 eV/0.134；相比 SFT‑20x，GRPO 提升多样性但略高误差。

**⚠️ 局限性**

局限性：DFT 验证样本仅 20 个，未覆盖全部 OLED 化学空间；奖励模型存在分布漂移风险；RL 过程对超参数敏感；缺乏实验合成与器件性能验证。

---

## 347. MaudeTypedLog: A Typed Interpreter for Prolog in Maude

**arXiv ID:** 2607.20184 | [PDF](https://arxiv.org/pdf/2607.20184v1)

**作者:** Enrique Gallifa-Tronch `[一作]` (Universitat Politècnica de València), Santiago Escobar `[通讯]` (Universitat Politècnica de València)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

实现了一个基于 Maude 的 Prolog 解释器，支持类型化统一（Typed Unification）并按 TSLD 语义执行推导，能够在程序和查询中检测类型错误。

**💡 创新点**

创新点在于：①首次将参数化确定性正则类型与 Martelli‑Montanari 统一算法结合实现类型化统一；②在 Maude 中实现 TSLD‑resolution，区分 false 与 wrong 两种失败；③通过 TSLD‑树自动识别 blame 句子（有类型错误的子句）。

**🔧 技术方法**

使用了 Maude 的函数模块实现类型化统一、重写逻辑实现 TSLD 步骤、以及外部对象和命令行接口；核心技术包括正则类型、类型推导规则、重写规则和 Maude 的 transition‑rule 机制。

**📊 数据集**

主要使用了论文中给出的示例程序（例如包含 p、q、r 等谓词的程序）作为测试集；未使用公开的大型数据集，而是基于手工构造的 Prolog 示例。

**📈 对比分析**

对比方法主要是理论验证：构造 TSLD‑树并检查叶子是否为 wrong，证明与理论一致；未进行性能基准测试，报告仅显示正确性与一致性。

**⚠️ 局限性**

局限性包括：①未对 Prolog 的内置算术与内置谓词进行类型检查；②仅支持有限的基本类型（int、float、atom、str、列表等），不支持自定义复合类型；③缺乏大规模程序与性能评估；④在 Maude 的实现中，重写规则顺序需要人工约束，可能影响效率。

---

## 348. Instance Hardness-Based Relevance for Imbalanced Regression

**arXiv ID:** 2607.20173 | [PDF](https://arxiv.org/pdf/2607.20173v1)

**作者:** Vitor M. Leitao `[一作]` (Universidade Federal de Pernambuco), Rafael M. O. Cruz `[通讯]` (University of Quebec)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于实例难度的相关性函数 InHaR，用于识别不平衡回归中的稀有实例，并通过重采样提升模型性能。

**💡 创新点**

创新点在于将实例难度与稀有度关联，克服传统基于分布的相关性函数在多峰分布下的局限。

**🔧 技术方法**

采用实例难度计算、随机森林、Bagging、XGBoost、SVR、MLP 等回归器，以及随机过采样 (RO) 和高斯噪声 (GN) 重采样方法。

**📊 数据集**

实验使用 29 个公开回归数据集（如 a1、a2、abalone、boston 等）以及合成双峰分布数据集。

**📈 对比分析**

通过 MAE/MSE 等无相关性度量，在约 50‑60% 的模型‑数据集对上，InHaR‑RO/GN 的表现明显优于传统相关性函数和 DRF 方法，Wilcoxon 检验显示多数提升显著。

**⚠️ 局限性**

主要限制为需要训练一组回归器计算实例难度，导致计算成本较高；阈值选择缺乏系统的敏感性分析。

---

## 349. Two-Way Wiretap Channel under Mixed Secrecy Constraint

**arXiv ID:** 2607.20172 | [PDF](https://arxiv.org/pdf/2607.20172v1)

**作者:** Yanling Chen `[一作]` (Volkswagen Infotainment GmbH), Masahito Hayashi `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究两向信道在强单侧（混合）保密约束下的可达率，提供非自适应与自适应编码方案并给出指数级误差与泄漏上界；

**💡 创新点**

首次在此模型下实现强单侧保密的单字母可达率，展示自适应密钥交换可提升发送方1的个体速率而不影响最大总速率；

**🔧 技术方法**

采用非自适应随机编码、可逆一次性密码、Rényi信息量、相对熵、可辨别率（resolvability）以及信息理论的递归泄漏分析；

**📊 数据集**

无须外部数据集，全部以离散记忆无关两向加密信道的概率分布作为理论模型；

**📈 对比分析**

与以往弱单侧保密的内边界及全信息可达率进行对比，证明自适应构造可严格包含非自适应结果，且误差与泄漏随块长呈指数下降；

**⚠️ 局限性**

仅考虑离散记忆无关模型，未给出可行的多用户扩展或实际协议实现细节，对连续或带噪声多维信道的适用性仍待验证。

---

## 350. Self-organizing Architecture of Receptron Units: a Hardware-Aware Framework for Edge Intelligence

**arXiv ID:** 2607.20162 | [PDF](https://arxiv.org/pdf/2607.20162v1)

**作者:** Stefano Radice `[一作]` (University of Milano), Paolo Milani `[通讯]` (University of Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于 Receptron 的单元神经网络，能够在 MCU 上实现可持续在线学习与推断。

**💡 创新点**

将输入依赖权重与高斯感受域相结合，形成可通过自组织 SOM 动态分配的非线性阈值决策边界，实现单单元高表达力与可解释性。

**🔧 技术方法**

使用 Receptron 模型、PCA 引导初始化、K 类条件 SOM 学习、硬阈值激活与加权投票、可变带宽 μ 以及在线 Hebbian 更新等技术。

**📊 数据集**

采用 Iris（3 类，4 维）与 Breast Cancer Wisconsin（2 类，30 维）两个基准数据集。

**📈 对比分析**

通过与 SVM 和 Random Forest 进行五折交叉验证对比，Receptron 在 Iris 上 90.0% 与 SVM 94.7% 相差 4.7%，在乳腺癌上 93.5% 与 SVM 94.4% 相差 0.9%，并保持低计算量与内存占用。

**⚠️ 局限性**

局限在于对高维或复杂分布可能需要更多中心；参数 τ 与 μ 的调优需经验；对非高斯噪声鲁棒性有限，极端漂移下需重新分配中心。

---

## 351. SHFormer: Dynamic Spectral Filtering Convolutional Neural Network and High-pass Kernel Generation Transformer for Adaptive MRI Reconstruction

**arXiv ID:** 2607.20159 | [PDF](https://arxiv.org/pdf/2607.20159v1)

**作者:** Sriprabha Ramanarayanan `[一作]` (Indian Institute of Technology Madras), Mohanasankar Sivaprakasam `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 SHFormer，一种将频域滤波 CNN 与动态高通核生成 Transformer 结合的可扩展 MRI 重建网络。

**💡 创新点**

创新点包括：① 基于 DCT 的频域注意机制实现模态特定可迁移特征；② 在 Transformer 中引入动态高通核生成自注意模块，精准恢复高频细节；③ 通过这些机制实现跨模态、跨加速因子（开放/闭合）泛化。

**🔧 技术方法**

采用 DCT 频域滤波、动态权重预测（Hypernetwork）、Transformer 自注意、动态高通核生成、卷积重建与 k‑space 数据保真等技术，并在监督、自监督和扩散模型训练模式下实现。

**📊 数据集**

使用 ACDC 心脏 MRI、FastMRI 骨关节 MRI、多模态 MRBrainS/IXI/Calgary 以及多通道 Knee 数据集进行评估。

**📈 对比分析**

与多种 CNN、GAN、Transformer、元学习与自适应网络（如 MCI‑HyperNet、Ada‑IN、DFF 等）对比，在 PSNR/SSIM 上提升 0.3–0.6 dB / 0.006–0.01，开放/闭合泛化对比提升约 1 dB，扩散模型中加入 SF/HF 模块后 PSNR 进一步提升 0.14 dB。

**⚠️ 局限性**

局限性包括：模型参数与 FLOPS 略增，推理时间略长；在极高加速因子（>10×）下高频恢复仍有限；对不同设备/扫描仪的进一步验证与跨域适配仍需探索。

---

## 352. Local Stability and Gaussian Smoothing of Quantized Neural Networks

**arXiv ID:** 2607.20153 | [PDF](https://arxiv.org/pdf/2607.20153v1)

**作者:** Sergey Salishev `[一作]` (St. Petersburg State University), Oleg Granichin `[通讯]` (St. Petersburg State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

本文提出利用高斯平滑构造量化神经网络的无穷阶连续代理，并给出在“局部振荡有限”条件下的误差上界；

**💡 创新点**

创新点在于将高斯卷积作为量化网络的天然熵化手段，首次给出局部维度相关误差界，并给出ReLU和sign激活的闭式平滑形式；

**🔧 技术方法**

技术主要包括高斯平均（Steklov平均）、中心极限定理推导、Sobolev卷积与梯度近似；

**📊 数据集**

未使用公开数据集，示例基于随机生成的高维二元感知机和理论实验；

**📈 对比分析**

方法与传统平滑激活（GELU、softplus、tanh）相比，在保持梯度可导性和逼近精度方面更具可解释性，实验中可获得与原量化网络相近的误差水平；

**⚠️ 局限性**

局限性在于仅能保证局部误差控制，需满足局部振荡有限；不提供全局稳定性证明，且对多层网络的隐空间几何分析尚缺乏理论支持。

---

## 353. Gotta Catch them all: the modes of Sycophancy

**arXiv ID:** 2607.20146 | [PDF](https://arxiv.org/pdf/2607.20146v1)

**作者:** Shreyans Jain `[一作]` (Thoughtworks), Amirali Abdullah `[通讯]` (Thoughtworks)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在社交压力下的顺从行为（sycophancy），将其拆分为三种内部模式并通过激活偏差向量验证其结构。

**💡 创新点**

提出sycophancy并非单一倾向，而是由可线性分离、时间分阶段、注意力机制各异的三种内部模式组成，揭示输出与内部机制的解耦关系。

**🔧 技术方法**

采用偏差向量、PCA/ICA子空间分析、线性探针、聚类、因果消融、注意力头分析以及Logit‑lens等技术对模型内部机制进行多层面分析。

**📊 数据集**

构建约950个社交压力场景，配备四种角色（PA、SI、DCA和中性），共约4000条提示，作为实验数据集。

**📈 对比分析**

使用聚类ARI=1、线性探针在第14层后准确率100%、文本分类仅57.8%来衡量模式可分离度；注意力消融显示PA有专属头，SI/DCA共享；跨层、跨模式评估均支持结论。

**⚠️ 局限性**

局限性包括仅在Gemma‑2‑9B‑it模型上验证，未直接验证可控性；子空间与心理标签不必然对应；可能存在未列出的其他顺从模式。

---

## 354. CURED: Creating, Understanding, and Repairing Errors Demonstrator

**arXiv ID:** 2607.20140 | [PDF](https://arxiv.org/pdf/2607.20140v1)

**作者:** Nicholas Chandler `[一作]` (Berliner Hochschule für Technik), Felix Bießmann `[通讯]` (Berliner Hochschule für Technik)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了一个集错误注入、错误检测与清洗、错误机制分析为一体的可视化演示平台 CURED，帮助用户交互式地探索表格数据错误的生成、检测与修复过程。

**💡 创新点**

创新点在于将基于 conformal prediction 的数据清洗方法（Conformal Data Cleaning）与错误机制检测算法（MechDetect）整合到同一平台，并提供多种数据依赖错误模型和可配置的错误注入工具，促进理论与实践的结合。

**🔧 技术方法**

使用了机器学习模型（如 Catboost）进行列级预测、conformal prediction 进行不确定性校准、错误注入库 tab-err、以及 MechDetect 的统计比较方法。

**📊 数据集**

使用了公开的核电站数据集（OpenML ID 44969）以及用户自定义的 csv/xlsx 数据，数据规模受限于 10 列、10,000 行，并包含目标列。

**📈 对比分析**

在 ECAR、EAR、ENAR 三种错误机制下，Conformal Data Cleaning 的误报率约为 0.006–0.007，检测真阳率约为 0.646–0.650，清洗后下游机器学习任务性能提升（DSI）在 0.0069–0.0349 之间；MechDetect 的机制识别准确率在 0.70–0.99 之间，取决于列类型和错误机制。

**⚠️ 局限性**

局限性包括：仅支持少量机器学习清洗方法、错误注入和机制检测模型未覆盖所有实际错误场景、对极端错误率和大规模数据集的表现未作评估、以及当前演示器的可扩展性需进一步验证。

---

## 355. OpenSkillRisk: Benchmarking Agent Safety When Using Real-World Risky Third-Party Skills

**arXiv ID:** 2607.20121 | [PDF](https://arxiv.org/pdf/2607.20121v1)

**作者:** Qiyuan Liu `[一作]` (City University of Hong Kong), Ning Miao `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了名为 OpenSkillRisk 的安全基准，涵盖 263 种真实可疑的第三方技能，并对其进行分类、任务包装、沙箱化评估。

**💡 创新点**

创新点包括：①基于真实技能市场的可疑技能挖掘与分类体系；②结合执行级与认知级两维指标（ASR、Awareness、F_safe）实现多维安全评估；③细粒度行为分析与 Guard Skill 防御实验。

**🔧 技术方法**

使用了 LLM‑驱动的技能执行框架（Claude Code、Gemini CLI、Codex CLI）与 13 个前沿 LLM（包括 GPT、Claude、Gemini、DeepSeek、GLM 等），以及静态与动态分析工具进行安全评估。

**📊 数据集**

主要数据集为 OpenSkillRisk：从 SkillsMP、Skills.rest 等公开仓库收集、筛选得到的 263 种有风险技能，附带 9 个任务域、7 类攻击类型的任务包与沙箱环境。

**📈 对比分析**

通过对 3 种 CLI harness 与 13 种模型的交叉实验，比较 ASR（执行安全率）和 Awareness（认知安全率），并引入 F_safe 综合指标。实验结果显示：最优系统 F_safe 仅达 80.97%，即约 17% 的案例仍执行了不安全路径；Context‑Dependent 与 System‑Level 风险更难识别，表现最差。

**⚠️ 局限性**

局限性：①样本覆盖范围仍有限，未覆盖所有可能的技能风险模式；②依赖人工标注与审核，标注成本高；③沙箱化评估可能未完全模拟真实网络/环境交互，导致部分风险被低估；④ Guard Skill 的防御效果受加载策略影响，未能完全消除过度防御问题。

---

## 356. Ascend to Science: Exploration of AI Chips for Scientific Computing

**arXiv ID:** 2607.20120 | [PDF](https://arxiv.org/pdf/2607.20120v1)

**作者:** Weicheng Xue `[一作]` (Pengcheng Laboratory), Yonghong Tian `[通讯]` (Pengcheng Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对华为Ascend 910系列 NPU 进行 kernel 级别的性能基准和五个代表性科学应用（HPL-MxP、LRSVD、SGEMM‑cube、PQSim、SMC‑X）的系统级映射，研究并解决了 AI 加速器在科学计算中的精度、执行和数据迁移三大差距，证明在精细的数值重构、异构计算划分、内存层次管理和通信重叠等技术支持下，AI‑本位 NPU 能达到数值稳健、性能竞争和可扩展的效果。

**💡 创新点**

创新点在于：①提出跨层级的异构 CPU–NPU–通信协同映射；②将低秩 SVD、矩阵乘法等核心算法通过精度再分解、迭代正交化或 mantissa‑splitting 等技术实现 FP32 级别精度；③针对宽带瓶颈的量子模拟与 Monte‑Carlo，通过门融合、缓存块化、双晶格与统一缓存化等方法把不规则访问转为向量化、块级高效执行；④将这些技术统一应用于五类不同的科学工作负载，并在 Ascend 上实现了可量化的性能提升。

**🔧 技术方法**

使用了 Ascend 910 的 Bisheng C++ 编程接口、CANN 运行时、HCCS 通信库；实现了混合精度迭代细化、正交化迭代、mantissa‑splitting 乘法、门级融合、缓存块化、双晶格结构、统一缓存（UB）管理、向量化掩码运算及多级流水线与通信重叠。

**📊 数据集**

实验基准包括：HPL‑MxP（高性能线性代数基准）、LRSVD（低秩 SVD）、SGEMM‑cube（FP32 精度矩阵乘）、PQSim（基于随机电路采样的量子状态向量模拟）、SMC‑X（千亿原子规模 Monte‑Carlo 采样）。

**📈 对比分析**

与 NVIDIA A800/H800 GPU、Intel Xeon CPU、cuSOLVER、cuQuantum 等传统加速器及其优化库进行对比。性能方面：SGEMM‑cube 在 910C 上保持 60 TFLOPS，远超 GPU 基线；HPL‑MxP 在多节点上实现近线性扩展，910C 芯片功率更高；PQSim 在 910C 上完成 30 量子比特、30 层电路采样的时间为 11.4 s，优于 NVIDIA A800 的 14.3 s；SMC‑X 在 4 芯片 910C 上每芯片实现 1.81 × 10⁷ 原子步/秒，接近 16 芯片 H800 的 3.62 × 10⁷；能耗方面，LRSVD 与 SGEMM‑cube 在 Ascend 上的能耗相对 GPU 均有所下降。

**⚠️ 局限性**

局限性：仅验证了 Ascend 910 系列，缺乏跨硬件通用性；映射方案需专家手工调优，缺乏编译器/运行时自动化支持；实验覆盖范围局限于密集型、低秩或状态向量模拟等特定工作负载，未涉及稀疏矩阵、FFT、通用 stencil 等；对高精度 FP64 仍需 CPU 或专用硬件支持；未来需扩展到其他 AI 加速器并完善编译器/库层面的自动化。

---

## 357. An \(O(\log n)\)-Approximation for Three-Terminal Reachability-Preserving Minimum Edge Cut

**arXiv ID:** 2607.20114 | [PDF](https://arxiv.org/pdf/2607.20114v1)

**作者:** Qi Duan `[一作]` `[通讯]` (Carnegie Mellon University), Qi Duan (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多项式时间的O(log n)近似算法，用于在加权无向三端点RPMEC（即分离终端t与s1、s2但保持s1与s2连通）的最小边割问题。

**💡 创新点**

创新点在于将Räcke的概率cut‑dominating分解树与根树聚类的组件扩展相结合，解决树集群不连通导致的“拆分”问题，并证明组件预算保持不变，进而构造辅助图求最短路径得到连通的源侧。

**🔧 技术方法**

核心技术包括：Räcke的cut‑dominating分解树、树根聚类与其诱导图的连通组件分解、节点加权辅助图的最短路径求解、零成本边的预处理与ε扰动保证正权重。

**📊 数据集**

论文仅给出理论分析，未使用实验数据集；所有结论均基于图论与算法分析。

**📈 对比分析**

与之前已知的O(√n)近似算法相比，本文将近似比降至O(log n)，通过证明辅助路径成本与树分解的期望树割容量的对比，得出理论上可实现的O(log n)性能。

**⚠️ 局限性**

局限性包括：仅适用于无向图；无法直接推广到有向RPMEC或多受保护终端情形；需要构造多项式规模的树分布，实际实现可能受限；处理零成本边的预处理步骤在特殊实例下可能导致额外复杂度。

---

## 358. Multiparty Session Types for GDPR Purpose Compliance

**arXiv ID:** 2607.20190 | [PDF](https://arxiv.org/pdf/2607.20190v1)

**作者:** Evangelia Vanezi `[一作]` (University of Cyprus), Anna Philippou `[通讯]` (University of Cyprus)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一个基于多方会话类型的正式框架，用以在软件系统中对GDPR目的限制进行建模、实现与合规验证。

**💡 创新点**

创新点在于：①将个人数据与其存储机制（如数据仓库）纳入会话类型语义，使目的不仅仅是文本标签，而是结构化的通信协议；②构建了带有私有数据操作的进程演算和相应的类型系统，证明了主体约束（Subject Reduction）和目的忠实性（Purpose Fidelity）；③提供了从UML序列图到全球类型的映射，实现了软件工程实践的可操作性。

**🔧 技术方法**

使用的技术包括：多方会话类型（MPST）框架、进程演算（带有数据存取操作）、类型推导系统、子类型关系、静态类型检查，以及正式语义证明（如主定理）。

**📊 数据集**

案例研究使用的“医疗诊断工作流”作为数据集，涵盖患者、护士、实验室和全科医生等参与者，以及相应的个人数据存储（身份信息、症状、实验室订单、实验室结果、诊断报告）。

**📈 对比分析**

与传统基于文本标签或流程图的方法相比，本框架在合规性检测上具有完备性与安全性。实验（仅限案例验证）显示，类型检查能够在编译阶段检测到所有非合规行为，且系统运行时不需要额外的监控开销；然而，目前未给出大规模实验或性能基准。

**⚠️ 局限性**

限制主要包括：①当前仅支持单一会话，未处理多会话并发；②缺乏线性约束，无法保证存储引用的唯一性；③仅通过类型系统进行静态验证，未覆盖运行时数据流动态变化的情况；④缺乏自动化工具和实测性能数据。

---

## 359. Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust Generalist Humanoid Control

**arXiv ID:** 2607.20110 | [PDF](https://arxiv.org/pdf/2607.20110v1)

**作者:** Yubiao Ma `[一作]` (Beijing Institute of Technology), Dongdong Zheng `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Extreme-RGMT 两阶段持续学习框架，先训练通用运动跟踪基准，再通过 PACE 机制在保持通用能力的同时获取极限动态技能，并用 STAR 采样提升稀有动态样本利用。

**💡 创新点**

创新点包括：1) 两阶段异步训练（基准+Pace）实现专家技能扩展而不破坏通用能力；2) STAR 片段优势重采样结合难度先验，显著提升动态动作学习效率；3) 改进的政策架构（独立编码感知、动作历史和参考命令，加入 FSQ 量化瓶颈）增强稳健性。

**🔧 技术方法**

采用强化学习（PPO）、异步环境分配、动作/感知历史跨注意力编码、FSQ 量化瓶颈、动态难度采样、自适应进度权重、优势归一化、片段优势评分与重采样、以及大量的动力学随机化。

**📊 数据集**

使用 LAFAN1、AMASS、以及自采 Xsens 运动共计约 186 小时的数据，分为通用动作与极限动态子集。

**📈 对比分析**

与 ExBody2、BeyondMimic、SONIC、RGMT 在一般动作集(In-source、Unseen)比较，Stage I 已超越对手；在极限动态集（XtremeMotion、AMASS Challenging）与 OmniXtreme、Fine‑Tuning 对比，Full 版达到了或超过最优。整体成功率提升、误差下降，硬件测试中在线 Xsens 动作成功率提升至 85% 以上。

**⚠️ 局限性**

局限性：仅跟踪相对根姿，长时运行会出现全局漂移；对极限动态之外的新动作迁移能力有限；需要大量高质量动态数据，仍需进一步在线自适应与泛化。

---

## 360. How Developers Use Relation Chains in Gerrit-Based Review Ecosystems: An Empirical Study Across Three Open-Source Ecosystems

**arXiv ID:** 2607.20189 | [PDF](https://arxiv.org/pdf/2607.20189v1)

**作者:** Ahmed Belhouchette `[一作]` (Manouba University), Mohammad Hamdaqa Abdelwahab Hamou-Lhadj `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究对三大开源 Gerrit 生态（OpenStack、Wikimedia、ONAP）中的关系链进行量化分析，评估其普及程度、使用趋势以及对代码评审时间、CI 负载和同步瓶颈的影响。

**💡 创新点**

创新点在于首次将关系链视为完整的协作单元，揭示链成员相较于孤立变更在合并延迟、CI 扩散和中间成员同步成本上的显著差异，并提出“基础效应”与“动态重构”概念。

**🔧 技术方法**

采用非参数统计方法（Mann–Kendall 趋势、Mann–Whitney、Cliff’s δ、Spearman ρ）对 401,256 条已完成变更和 29,580 条关系链进行时序、对比与相关性分析。

**📊 数据集**

使用来自 OpenStack、Wikimedia 与 ONAP 三个 Gerrit 生态的 15 个项目共 401,256 条已完成变更的原始 REST API 数据。

**📈 对比分析**

通过与同类孤立变更在相同大小分桶下的比较，发现链成员的平均合并延迟是同类单变更的 2.6 倍，CI 任务数量为 10–23 倍，相关系数 ρ 在 0.43–0.61 之间，表明链结构显著影响评审成本。

**⚠️ 局限性**

局限性包括仅研究 Gerrit 平台，未考虑任务复杂度、工业环境和其他平台（GitHub、Phabricator 等）的差异；链检测依赖父 SHA 匹配，可能遗漏非标准链；所有发现均为相关性，未能确立因果关系。

---

## 361. Basic Model Theory for Path Predicate Modal Logic

**arXiv ID:** 2607.20183 | [PDF](https://arxiv.org/pdf/2607.20183v1)

**作者:** Raul Fervari `[一作]` (CONICET), Leonardo Torres `[通讯]` (IMDEA Software Institute)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并阐述 Path Predicate Modal Logic（PPML）的模型理论，包括不受限的 bisimulation、Hennessy–Milner 类、ultrafilter 扩展以及 van Benthem 风格的可判定性与可表达性定理。

**💡 创新点**

提出了 PPML 的非受限 bisimulation 以及对应的 Hennessy–Milner 类；利用 ultrafilter 扩展构造饱和模型，并证明 van Benthem 定理，明确了 PPML 与一阶逻辑之间的精确对应关系。

**🔧 技术方法**

采用模型论方法：构造超滤扩展、ultraproduct、ω‑saturation、compactness、Łoś 定理以及标准翻译技术来证明定理。

**📊 数据集**

无实际数据集；论文完全基于理论证明。

**📈 对比分析**

未进行实验或性能比较；成果以数学证明与理论分析为主。

**⚠️ 局限性**

局限性：仅在理论层面探讨，未给出算法实现或复杂度分析；对非 fluted 版本、扩展到 fixpoint 等方向仍待研究；对实际数据查询语言的具体适配仍未深入。

---

## 362. A Typing System for the Linear Lambda-Calculus in de Bruijn Notation

**arXiv ID:** 2607.20181 | [PDF](https://arxiv.org/pdf/2607.20181v1)

**作者:** Philippe de Groote `[一作]` (Université de Lorraine), Vincent Tourneur `[通讯]` (Université de Lorraine)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种在 de Bruijn 表示法下的线性 λ-算子类型系统，并证明了其满足主语归约性（subject‑reduction）性质。

**💡 创新点**

创新点在于将 Hodas‑Miller 资源消耗模型与 de Bruijn 索引以及“片段化环境”（fragmentary environments）相结合，构造出一种既能保证线性性又不需要显式冲突检查（如出现唯一性或域不相交）的类型判定系统。

**🔧 技术方法**

技术手段包括：de Bruijn 索引编码、片段化环境的加法与减法运算、线性逻辑中的资源模型、β‑约简的提升（lifting）与替换定义，以及形式化证明框架（引入的命题、引理与归约规则）。

**📊 数据集**

无数据集，工作为纯理论研究。

**📈 对比分析**

本文未进行实验对比或性能评估，而是通过形式化证明展示了系统的正确性与归约稳定性。

**⚠️ 局限性**

局限性包括：片段化环境的拆分与合并在实现上可能导致指数级别的组合；系统仍以线性代数为核心，未覆盖完整的指数多重线性逻辑；并且在实际编译器或证明搜索工具中的效率尚未评估。

---

## 363. Linearising Explicit Substitutions using Intersection Types

**arXiv ID:** 2607.20179 | [PDF](https://arxiv.org/pdf/2607.20179v1)

**作者:** Ana Jorge Almeida `[一作]` (LIACC), Mário Florido `[通讯]` (LIACC)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

定义了一种新的术语扩展方法，将含显式替换的 λ-计算与 Boudol 的多重资源 λ-计算关联起来，并分别针对无限和有限资源使用 ACI 与 AC 交叉类型进行扩展。

**💡 创新点**

首次将术语扩展应用于显式替换的 λ-计算，提供了一个统一框架，将 λ-计算与资源敏感的子结构类型系统（无限/有限多重）连接；通过交叉类型实现了资源可用性显式化，并证明了弱头归约的保持性。

**🔧 技术方法**

使用显式替换 λxgc-计算、Boudol 的 λ-计算与多重、交叉类型系统（ACI/AC）、术语扩展定义、弱头归约与形式化证明。

**📊 数据集**

无实际数据集，纯理论研究。

**📈 对比分析**

未进行实验或性能对比，所有结果均为形式化证明；因此不存在性能评价。

**⚠️ 局限性**

仅适用于弱头归约；未覆盖强归约；扩展关系不是单值函数；仅研究显式替换的 λ-计算，尚未扩展到并发或 π-计算等更复杂系统。

---

## 364. Learning to Decode Quantum LDPC Codes via Cluster-Based Sequential Belief Propagation

**arXiv ID:** 2607.20130 | [PDF](https://arxiv.org/pdf/2607.20130v1)

**作者:** Mohsen Moradi `[一作]` (Arizona State University), David G. M. Mitchell `[通讯]` (New Mexico State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于强化学习的群集化顺序贝叶斯传播（RL‑S）解码方法，用于量子低密度校验码（QLDPC）错误纠正。

**💡 创新点**

创新点在于将原本逐节点的RL调度扩展为聚类级别的调度，并通过归一化直方图量化的可置换不变聚类状态实现表格RL，从而显著降低序列深度，同时保持与单节点RL相近的误码性能。

**🔧 技术方法**

使用技术包括：表格Q学习、贝叶斯传播、聚类分区、归一化直方图状态量化、量子误差模型（独立Pauli‑X 与退化通道）以及CSS 量子 LDPC 代码的 Tanner 图。

**📊 数据集**

实验数据集主要包括两个代表性 QLDPC 码：[[882,24,18≤d≤24]] B1 代码和 [[288,12,18]] BB 代码；训练时随机采样错误概率在 {0.03,0.04,0.05,0.06,0.07} 之间。

**📈 对比分析**

通过与传统洪水 BP、单节点 RL‑S、QBP、QBPGD 等方法对比，聚类 RL‑S 在 B≈60 时可将每轮 BP 迭代的调度决策从 n 降至 Ncl≈15，误码率与单节点 RL‑S 极为接近，且在退化通道上同样优于 QBP，逼近 RL‑S 的性能。

**⚠️ 局限性**

局限性包括：聚类划分采用固定随机分区，未考虑图结构优化；聚类大小 B 与量化分辨率 L 的选择对性能影响显著，需要经验调参；实现仍假设理想的并行群集更新，实际硬件延迟需进一步评估。

---

## 365. HeadCast: Casting Attention Heads for Efficient Autoregressive Video Generation

**arXiv ID:** 2607.20125 | [PDF](https://arxiv.org/pdf/2607.20125v1)

**作者:** Jinliang Shen `[一作]` (Peking University), Chengru Song `[通讯]` (KlingAI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了HeadCast框架，通过一次性对预训练的自回归视频扩散模型的注意力头进行分类，随后为每种头类型分配专用的KV缓存路径，从而实现推理加速；

**💡 创新点**

创新点在于识别并利用注意力头的四类稳定异质行为（Sink、Dummy、Spatial、Global），并将其与KV缓存拆分相结合，在保持全上下文的同时显著减少注意力计算量；

**🔧 技术方法**

使用了基于最大噪声步的在线分类技术、余弦相似度与MSE判别头类型、HeadCast的多路缓存管理与头特定注意力实现；

**📊 数据集**

在多种自回归视频扩散模型（Self‑Forcing、LongLive、Causal Forcing、Reward Forcing）上进行实验，评估使用VBench、PSNR、LPIPS等指标；

**📈 对比分析**

与全注意力基线和Dummy Forcing等训练‑free方法比较，HeadCast在720P/1080P可实现1.31×/1.95×的速度提升，且PSNR/LPIPS与全注意力几乎无差距，显著抑制了因缓存裁剪导致的闪烁；

**⚠️ 局限性**

限制在于需要在推理初期执行一次分类，且对极端高分辨率或超长视频的通用性尚待进一步验证，某些头类型在不同模型或提示下的稳定性仍有限。

---

## 366. Autonomous Collaborative Learning Among an Ensemble of Tsetlin Machines with Consensus-Based Inference

**arXiv ID:** 2607.20124 | [PDF](https://arxiv.org/pdf/2607.20124v1)

**作者:** Yehuda Rudin `[一作]` (Bar Ilan University), Alexander Fish `[通讯]` (Bar Ilan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于Tsetlin机的分布式去中心化垂直特征分区学习框架，利用邻居间的gossip式通信实现共识推理，且不泄露原始数据或模型参数。

**💡 创新点**

创新点在于：①首次将双层层次化Tsetlin机与去中心化共识机制相结合；②设计异构代理网络支持不同特征子集、数据分布和计算资源的协同学习；③通过仅传输二进制句子输出实现低能耗、低带宽的通信。

**🔧 技术方法**

核心技术包括Tsetlin机自学习自动机、两层逻辑句子构造、Type I/II奖励惩罚机制、gossip随机广播通信协议以及多源全局投票汇总算法。

**📊 数据集**

实验使用MNIST、Fashion‑MNIST和合成多传感器网络数据集，将图像切分为局部块或传感器子集进行分布式学习。

**📈 对比分析**

与集中式ANN和传统TM对比，分布式Tsetlin机在MNIST/Fashion‑MNIST上分别达93–94%与约80–84%的准确率，Sensor网络下可与集中式ANN实现93%准确率，显示出与集中模型相近的性能。

**⚠️ 局限性**

局限性包括：假设通信理想无误差；未考虑网络延迟、丢包或攻击；模型规模总体较大，且缺乏对隐私泄露风险的正式安全分析。

---

## 367. Proceedings of The Fourth International Workshop on eXplainable AI for the Arts (XAIxArts 4)

**arXiv ID:** 2607.20131 | [PDF](https://arxiv.org/pdf/2607.20131v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 368. Reinforcement Learning for Large Language Model Selective Evidence Adoption from Contaminated Retrieval Results

**arXiv ID:** 2607.20090 | [PDF](https://arxiv.org/pdf/2607.20090v1)

**作者:** Yanyu Chen `[一作]` (East China Normal University), Lichang Dai `[通讯]` (Shandong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SelectBench benchmark 并使用 DAPO 对 Qwen3.5-4B 进行后训练，使模型在包含有用证据与误导/指令性文本的检索结果中能有选择地采用正确信息。

**💡 创新点**

创新点在于将“选择性证据采用”定义为一个独立的后训练目标，并构造了受控污染的检索上下文数据集；同时通过两种奖励策略（规则匹配与冻结语义判别器）验证奖励设计的有效性。

**🔧 技术方法**

主要技术包括 DAPO（基于 GRPO 的策略梯度方法）、稀疏三分奖励（+1/0/-1）与长度惩罚、规则匹配奖励与冻结的 DeepSeek 语义判别器；在训练时使用了 token‑level 受限裁剪的 PPO 变体。

**📊 数据集**

使用了多源的 1,300 条训练实例（HotpotQA、2WikiMultiHopQA、MuSiQue）和 325 条纠正后的测试实例（SelectBench‑v2），并在 MMLU 与干净 HotpotQA 上评估通用能力。

**📈 对比分析**

与原始 Qwen3.5-4B 对比，DAPO‑Rule 及 DAPO‑DeepSeek 在严格成功率上分别提升约3%和4%，并在禁止内容采用率下降约3%；在 MMLU 和 HotpotQA 上几乎不降性能；然而在注入攻击抵抗、统计显著性及多重检验校正方面未见显著改善。

**⚠️ 局限性**

局限性包括：提升幅度有限且未通过 Holm 校正显著；对注入攻击的抵抗力不足；实验只针对单一模型规模、单一检索工具，缺乏跨模型与多样检索场景的验证；训练样本量与奖励设计仍需进一步优化。

---

## 369. What is a Model of the Linear Lambda Calculus?

**arXiv ID:** 2607.20088 | [PDF](https://arxiv.org/pdf/2607.20088v1)

**作者:** Arturo De Faveri `[一作]` `[通讯]` (Université Paris Cité), Arturo De Faveri (Université Paris Cité)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对线性 λ‑演算的模型进行系统性研究，提出了三种等价的模型表述：线性 λ‑术语的操作范畴（L‑operad）所产生的代数、基于线性组合子 B、C、I 的 BCI‑代数的子范畴（线性 λ‑代数）以及半闭（semiclosed）操作范畴，并给出了线性 λ‑代数的有限等式表述；同时证明了线性 Scott 表示定理，即每个模型都可视为某个对称闭单张范畴中保持折叠对象的可折叠对象。

**💡 创新点**

创新点主要包括：①首次把 L‑operad、线性 λ‑代数与半闭操作范畴三者建立完备的同构关系；②给出了 BCI‑代数的有限方程化简，提供了线性 λ‑代数的完备等式体系；③在范畴语义框架下推出线性 Scott 表示定理，展示了模型可归约为对称闭单张范畴中可折叠对象的本质。

**🔧 技术方法**

使用的技术包括：操作范畴与 PROP 的构造、对称单张闭范畴与 Day 卷积、终极单张闭范畴中可折叠对象的概念、β‑等价、λ‑抽象的抽象算子 λ^*、以及半闭操作范畴的截面与重排映射等。

**📊 数据集**

无数据集。本工作为纯理论研究，全部推导与证明均在数学框架内完成。

**📈 对比分析**

由于为理论证明性工作，没有实验或数值比较；性能评估不适用。论文通过严格的等价性证明和构造性映射展示了模型之间的互通性。

**⚠️ 局限性**

局限性：①仅聚焦无类型（untyped）线性 λ‑演算，未处理 I‑calculus 或多态/类型化线性 λ‑演算；②实现细节在范畴语义层面未给出可直接编程或求值的具体算法；③对大规模组合子系统的可扩展性与实际应用场景尚未探讨。

---

## 370. SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD

**arXiv ID:** 2607.20145 | [PDF](https://arxiv.org/pdf/2607.20145v1)

**作者:** Dongfang Li `[一作]` (Loop Area Institute), Zhiquan Luo `[通讯]` (Loop Area Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文先在华为 Ascend SuperPOD 上完成 1.6T 参数的 MoE 大模型 DeepSeek‑V4‑Pro 的全参数后训练，提出系统层级优化（并行度、通信调度、内存管理和核级优化）实现显著加速；随后在同一体系结构上开发了面向运营研究（OR）的后训练工作流，利用 CPT（持续预训练）与 SFT（监督微调）构建 10K 高质量样本，得到的 Flash 版本在 OR 评测中达 71.81% Pass@1，超过 GPT‑5.4‑Mini 与基线 DeepSeek‑V4‑Flash。

**💡 创新点**

创新点包括：① 在 SIMD‑NPU Ascend 设备上首次实现 1.6T MoE 模型的全参数后训练系统化优化；② 设计 AuraKernel 基于运筹学的张量划分和迭代反馈搜索，自动化实现 AscendC 核级性能提升；③ 将多维并行、虚拟流水线、双缓冲交换器等机制整合为完整三层优化框架；④ 提出面向 OR 的 CPT–SFT 训练管线，结合求解器验证数据和结构一致性评估，显著提升模型在结构化优化任务上的表现。

**🔧 技术方法**

技术手段：Ascend 910C NPUs 与 SuperPOD 体系结构；专家-张量并行（ETP）和 Tensor/Data 并行；虚拟流水线（VPP）与双向管线抑制；双缓冲交换器（Swap Optimizer）降低优化器状态占用；AuraKernel 采用运筹学求解器指导的 tiling、循环重排与内存调度；核级融合与重写实现数据局部性提升；OR-CPT 数据引擎生成求解器验证的文本/代码对；SFT 数据清洗与自我蒸馏；评估指标包括 MFU、Pass@1、B4O‑Feasible、B4O‑ORGEval 等。

**📊 数据集**

数据集与资源：DeepSeek‑V4‑Pro（1.6T 参数）训练数据；DeepSeek‑V4‑Flash 预训练模型；10K 结构化 SFT 样本（覆盖 4 任务类、3 表述形式）；OR 评测套件（NL4OPT、OptiBench、B4O‑Feasible、B4O‑ORGEval）用于验证模型的求解器兼容性与结构一致性。

**📈 对比分析**

对比结果：MFU 从 11.67% 提升至 34.22%，提升 2.93×；在 OR 任务中，CPT‑SFT 后的 Flash 模型 Pass@1 为 71.81%，比 GPT‑5.4‑Mini 高 3.98 点、比基线 DeepSeek‑V4‑Flash 高 11.27 点；在 B4O‑Feasible 与 B4O‑ORGEval 等多指标上均取得最优或领先成绩，验证了 CPT‑SFT 组合的有效性。

**⚠️ 局限性**

局限性与未来方向：① 目前优化工作针对 Ascend NPU 生态，迁移至 GPU/TPU 等平台仍需进一步验证；② 核级优化和并行策略复杂，工程成本高；③ OR 任务仍面临结构一致性、整数变量语义等高阶错误，CPT‑SFT 仍未覆盖所有场景；④ 训练规模、数据量与评测覆盖度有限，尚未评估更广泛的工业案例；⑤ 体系结构对高并发 I/O 与网络负载的容忍度尚未完全解耦，未来可探索更灵活的分布式调度与压缩技术。

---

## 371. Back to Back with a Copy: A Computational Analysis of AI-Generated Visual Contemporary Art Pastiches

**arXiv ID:** 2607.20127 | [PDF](https://arxiv.org/pdf/2607.20127v1)

**作者:** Anca Dinu `[一作]` (University of Bucharest), Liviu Dinu `[通讯]` (University of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对比 Gemini Pro 3 Nano Banana 与 ChatGPT 5 生成的当代艺术仿作，评估其在纹理、色彩、语义、构图和感知维度上的相似度，并收集艺术家主观评价。

**💡 创新点**

首次将多种计算机视觉嵌入模型（AdaIN‑Style、ResNet50‑Style、CLIP‑ViT‑L、DINOv2、VGG19）与生成模型结合，验证艺术风格的多维性且不依赖空间架构；同时直接对比新旧生成模型的多维一致性。

**🔧 技术方法**

生成模型：Gemini Pro 3 Nano Banana、ChatGPT 5；特征提取模型：AdaIN‑Style、ResNet50‑Style、CLIP‑ViT‑L、DINOv2、VGG19；余弦距离度量；人工评估。

**📊 数据集**

108张图像数据集：12位艺术家各提供3幅原作（共36幅），以及每幅原作由两种模型生成的仿作（共72幅），并补充 Gemini 新系列作品。

**📈 对比分析**

使用各嵌入模型的余弦距离计算原作与仿作之间、仿作之间的相似度与多样性；Gemini 在语义和风格匹配上优于 GPT（相似度提升约15%），但在纹理、颜色和感知维度稍逊；人工评价与计算结果相符，艺术家认为 Gemini 更贴近其风格，但艺术价值略低。

**⚠️ 局限性**

数据集同质性强、仅包含 2D 画作、未涉及摄影或三维媒介；余弦距离可能无法充分捕捉艺术主观性；人工评估受主观偏差影响；未来需要更客观的评价指标与更广泛的多媒体数据。

---

## 372. Understanding the Impact of Linguistic Realization Choices on LLM Stance with Causal Tracing

**arXiv ID:** 2607.20115 | [PDF](https://arxiv.org/pdf/2607.20115v1)

**作者:** Langchen Huang `[一作]` (University of Stuttgart), Franziska Weeber `[通讯]` (University of Stuttgart)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究受控语言结构改写对大型语言模型（LLM）在政治立场判断任务中的影响，并通过激活补丁定位这些改写导致的立场偏移的内部机制。

**💡 创新点**

创新点在于：①系统评估了六种结构化重写（否定、反义、主动/被动、it‑cleft、wh‑cleft、SVC）对立场的具体影响；②首次将语义保持与逆转的结构改写与激活补丁方法相结合，揭示了中后层块输出是导致立场偏移的关键组件；③通过方差分解、Wasserstein距离和归一化恢复分数对模型鲁棒性进行细粒度量化。

**🔧 技术方法**

采用激活补丁（activation patching）在Decoder层的块输出、注意力子层和MLP子层进行干预；使用方差分解分析（Purpose Sensitivity、Articulation Sensitivity、Model Uncertainty）；计算flip率、清晰倾向率、1-Wasserstein距离及归一化恢复分数。

**📊 数据集**

扩展ProbVAA政治立场数据集，在其基础上手工/自动生成四类语义保持重写（主动/被动、it‑cleft、wh‑cleft、SVC）和两类语义逆转重写（否定、反义），并用英文版本的ProbVAA作为基准。

**📈 对比分析**

通过对四个开源模型（Gemma-3-4B-IT、Gemma-3-12B-IT、Qwen3-4B、Qwen3-14B）的清晰倾向率、flip率以及激活补丁恢复分数进行对比；结果显示：大模型的flip率更低、倾向更中立；激活补丁在mid-to-late Decoder块输出上恢复效果最佳。

**⚠️ 局限性**

局限性包括：①仅评估四个相对小规模模型，跨模型族和跨语言的普适性尚未验证；②实验聚焦单一政治立场任务，未检验其他任务；③重写过程可能仍引入微小语义偏差；④激活补丁仅定位恢复点，未给出完整的电路级解释；⑤实验规模有限，结果可能受样本大小影响。

---

## 373. ENTRAP-VL: A Taxonomic Probe for Dual Contextual Entrainment in Vision-Language Models

**arXiv ID:** 2607.20092 | [PDF](https://arxiv.org/pdf/2607.20092v1)

**作者:** Karan Goyal `[一作]` (IIIT Delhi), Vishal Bhutani `[通讯]` (PwC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ENTRAP-VL，一个针对视觉-语言模型的双流手工标注数据集与评估协议，用于系统测量文本与视觉上下文对模型答案的诱导影响。

**💡 创新点**

创新点包括：①将上下文诱导（contextual entrainment）从单模态扩展到双模态；②设计了包含关联性与真实性两轴的八类（文本）与三类（视觉）分类体系；③提供了可复现的手工标注数据与完整评估流程，填补了先前缺乏专门工具的空白。

**🔧 技术方法**

主要技术是人工构造与标注双流数据（文本-图像和视觉-文本），并定义多层次上下文条件；评估方法基于对比模型在无上下文与各诱导条件下的输出概率差异，侧重行为测评而非模型训练。

**📊 数据集**

使用了 1,500 条目（文本流 800 条目、视觉流 700 条目），覆盖 8 个文本类别和 7 个视觉类别，包含多种查询类型；所有图像均来自无水印的公开图库，文本由作者手工撰写。

**📈 对比分析**

通过在同一模型上分别给出无上下文与各条件的输入，计算答案概率的变化来比较文本与视觉诱导的强度；目前未给出具体模型结果，但框架可用于比较不同 VLM 的诱导表现。

**⚠️ 局限性**

局限性包括：规模相对有限、仅英文、手工标注主观性、缺乏多语言和更大规模扩展，且数据不适合作为公开排行榜或整体性能评估。

---

## 374. Phase Semantic Cut-elimination for Intuitionistic Linear Logic with Least and Greatest Fixed Points

**arXiv ID:** 2607.20187 | [PDF](https://arxiv.org/pdf/2607.20187v1)

**作者:** Jun Suzuki `[一作]` (Hokkaido University), Katsuhiko Sano `[通讯]` (Hokkaido University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构造 μIMALL 系统的相位语义，证明该系统的无剪裁可证明性与相位模型真值相一致，从而得到剪裁消除定理。

**💡 创新点**

创新点在于将相位语义方法扩展到包含最小/最大固定点的直观线性逻辑系统，并在此基础上完成了剪裁消除的证明。

**🔧 技术方法**

主要技术是相位语义（phase semantics）以及构造合成相位模型和归约候选（reducibility candidates）来实现无剪裁完备性与可证性。

**📊 数据集**

本研究为理论性证明，不使用任何实验数据集。

**📈 对比分析**

无实验比较，性能评估不适用，理论上已证明剪裁消除存在。

**⚠️ 局限性**

局限在于目前仅处理命题片段，未扩展到一阶谓词系统；且对指数模态与固定点相互关系的完整分析尚缺失。

---

## 375. Dynamic Logic with Parallel Operator for Verifying Communication Protocols

**arXiv ID:** 2607.20180 | [PDF](https://arxiv.org/pdf/2607.20180v1)

**作者:** Luiz C. F. Fernandez `[一作]` (Universidade Federal do Rio de Janeiro), Mario R. F. Benevides `[通讯]` (Universidade Federal Fluminense)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种带并行运算符的动态逻辑（DDYL），用于在对抗性环境下形式化验证密码协议的真实性与安全性，并给出了其语义、完整可公理化系统以及闭合、完备、终止的表格计算方法。

**💡 创新点**

创新点在于将Dolev–Yao入侵模型与过程代数中的并行与通信动作结合进动态逻辑框架，形成了新的“动态Dolev–Yao逻辑”，并为该逻辑构造了完整可归约的表格推理系统，首次证明了该系统在无迭代运算符条件下的终止性、可判定性与完整性。

**🔧 技术方法**

主要技术包括：Propositional Dynamic Logic（PDL）与过程代数的语义框架、Dolev–Yao推理规则、表格（tableau）计算技术、并行扩展的归约规则以及证明框架中的循环检查与偏好策略。

**📊 数据集**

本工作为理论性研究，未使用任何公开数据集；验证工作仅基于形式化证明与模型构造。

**📈 对比分析**

性能评估以形式化证明为主：证明了逻辑的完备性、可算性与终止性；并通过示例（A发送加密消息给Z的案例）展示了表格方法的完整性和有效性，未给出实验或计算复杂度分析。

**⚠️ 局限性**

局限性：1) 迭代运算符（*）被排除，导致无法处理循环或while形式的协议；2) 对计算复杂度未做定量分析；3) 仅针对无迭代的协议，无法直接推广到更复杂协议；4) 实际协议验证需进一步实现与性能评估。

---

## 376. PerceptDrive: Perception Prior World-Action Modeling with Adaptive Expert Routing for End-to-End Autonomous Driving

**arXiv ID:** 2607.20175 | [PDF](https://arxiv.org/pdf/2607.20175v1)

**作者:** Yushan Liu `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PerceptDrive，一种将冻结的视觉语言模型（VLM）先验与连续轨迹生成耦合的端到端驾驶规划框架；通过分支先验保持、场景条件路由以及预测未来潜变量的流匹配行动器，实现一次性生成高质量轨迹。

**💡 创新点**

创新点：①分支先验保持（per-branch prior retention），让每条专家分支在压缩后仍能保留对应的几何/语义/动态先验；②场景条件路由（scene-conditioned router），在轨迹生成前按场景动态加权融合专家条件；③利用训练时的评估子指标（privileged sub-metrics）进行路由与表示蒸馏，仅在训练阶段使用，无需测试时候的候选搜索或重排序。

**🔧 技术方法**

技术：冻结并适配 InternVL3 VLM 通过 QA 训练，再用多教师蒸馏（VGGT、V-JEPA、Wan）对三类先验进行对齐；冻结 V-JEPA 视觉编码器提供细粒度观测潜变量；查询‑瓶颈 + Transformer 背骨压缩感知池；每分支重建探测器实现先验保持；两层 MLP 路由器进行软门控；预测未来潜变量的流匹配头与动作头共同生成轨迹。

**📊 数据集**

数据集：NAVSIM（NavSim 1/2 训练集与 navtest/navhard 评估集），同时使用 DriveLM、LingoQA、DriveQA、NuScenes‑QA、Reason2Drive 等数据对 VLM 进行 QA 适配。

**📈 对比分析**

对比：与多种 VLA、世界模型、MoE 等现有方法在 NAVSIM v1（PDMS）和 v2（EPDMS）上对比，单摄像头无候选搜索，PerceptDrive 取得 90.4 PDMS（v1）和 90.2 EPDMS（v2），分别高于现有最高 90.1 PDMS 与 89.9 EPDMS，表现稳健且在 navhard 上也达 34.5 EPDMS。

**⚠️ 局限性**

局限：仅在单摄像头、单轨迹推理下测试；未做闭环/交互式评估；未来潜变量预测是确定性的，可能忽略多模态不确定性；性能高度依赖 NAVSIM 的评估器，跨任务迁移未验证。

---

## 377. StreamHOI: Interaction-aware Temporal Memory Adaptation for Streaming HOI Video Generation

**arXiv ID:** 2607.20174 | [PDF](https://arxiv.org/pdf/2607.20174v1)

**作者:** Zejing Rao `[一作]` (University of Chinese Academy of Sciences), Tong-Yee Lee `[通讯]` (National Cheng Kung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发了一种低延迟流式框架StreamHOI，用于长时段人-物互动视频生成。

**💡 创新点**

通过块级历史记忆偏好分析，提出偏置指导的记忆专业化训练与记忆距离缩放，实现不同Transformer块的记忆配置，解决sink‑local权衡问题。

**🔧 技术方法**

流式Diffusion Transformer、sink‑local记忆、块级记忆偏好分析、Bias‑guided memory‑specialized training (B‑MST)、Memory Distance Scaling (MDS)、DMD distillation、LoRA fine‑tuning、SAM2掩码、Temporal RoPE等技术。

**📊 数据集**

训练使用100k视频（80k内部直播电商视频+20k HOIGen‑1M），主测试集400例（含400k帧），并在GeoHOI‑testset与HOMA‑testset上评估。

**📈 对比分析**

与长视频生成基线（Causal Forcing、LongLive、Deep Forcing、InteractAvatar‑long、LongCat‑Video‑Avatar 1.5）以及双向HOI基线（UniAnimate‑DiT、VACE、GeoHOI、HUMO、HOMA）在5/30/60秒或5秒短视频上进行对比；StreamHOI在所有质量指标上领先，FPS 17.6，首块延迟0.75s，保持长时段交互一致性。

**⚠️ 局限性**

对交互不清晰、遮挡或模糊的场景生成不稳定；训练数据主要为单人视频，难以处理多人人物复杂情境。

---

## 378. Quantum Term Rewrite Systems: Applications to Complexity Analysis

**arXiv ID:** 2607.20170 | [PDF](https://arxiv.org/pdf/2607.20170v1)

**作者:** Kostia Chardonnet `[一作]` (Université de Lorraine, CNRS, Inria, LORIA), Thomas Vinet `[通讯]` (Université de Lorraine, CNRS, Inria, LORIA)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了量子项重写系统（QTRS），将传统的项重写系统（TRS）扩展到量子计算领域，并给出了能够保证物理可实现性的类型系统；在此基础上，证明了QTRS的标准性质（合并性、类型保持、规范形唯一性等），探讨了类型推理的可判定性，定义了可编译且终止的QTRS子类，展示了该子类与量子电路之间的互译关系，并利用这一关系给出了量子多项式时间类（FBQP）的隐式复杂度表征。

**💡 创新点**

创新点在于：①首次系统地将项重写框架迁移到量子世界，并设计了兼容量子物理约束的类型系统；②通过“最坏路径”扩展等技术，使经典TRS的终止与复杂度分析方法得以推广到带有超位置的量子重写；③提出了可编译、终止的QTRS子类，并证明其能够完整模拟任何量子电路，同时可以反向编译回量子电路，从而给出FBQP的内部表征；④对类型推理的算术复杂度进行精确定位（Π₂⁰-难），并在可简化子类中实现多项式时间判定。

**🔧 技术方法**

主要技术包括：量子项重写系统（STRS）与QTRS的语法与语义定义；基于等价关系的向量空间结构与规范形概念；类型系统与等价判定的形式化；“最坏路径”扩展（worst‑path ordering）、多项式解释和依赖对（dependency pairs）等终止与复杂度分析技术的量子化；电路编译与结构保持技术；以及对FBQP类的隐式复杂度表征。

**📊 数据集**

本文为理论研究，没有使用具体数据集；所有结果均通过形式化定义、定理证明与构造性证明得出。

**📈 对比分析**

由于缺乏实验对比，性能评估以理论上限进行。论文证明了可编译QTRS在时间复杂度上可与量子电路等价；但对实际执行时间、资源占用等细节未给出数值评估。

**⚠️ 局限性**

主要限制包括：①类型推理在一般情况下是Π₂⁰-难的，除非对语法做出强限制；②许多重要性质（如类型保持、规范形唯一性）需要系统终止；③递归结构受到严格约束（如单一递归调用），限制了表达能力；④对非结构保持的QTRS，编译回电路时可能导致量子门数指数级增长。

---

## 379. Active Inference as a Convex Markov Decision Process

**arXiv ID:** 2607.20152 | [PDF](https://arxiv.org/pdf/2607.20152v1)

**作者:** Nikola Milosevic `[一作]` (Max Planck Institute for Human Cognitive and Brain Sciences), Nico Scherf `[通讯]` (Max Planck Institute for Human Cognitive and Brain Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在固定世界模型下，将主动推理中的期望自由能（EFE）最小化等价于凸MDP，提出了基于镜像下降（MD）的MD‑AIF算法实现自然策略梯度；

**💡 创新点**

创新点在于揭示EFE最小化的凸MDP结构，并将镜像下降与软值迭代相结合，形成可收敛的期望自由能优化方法，并讨论闭环学习导致的可执行性（performative）问题；

**🔧 技术方法**

使用镜像下降（mirror descent）、软值迭代（soft‑Bellman）、自然策略梯度、凸MDP理论以及对偶和相对平滑性分析；

**📊 数据集**

实验基于小型确定性网格世界（5×5 与 10×10），不涉及真实数据集；

**📈 对比分析**

与无好奇心的强化学习（RL）和纯梯度下降相比，MD‑AIF 在 O(1/K) 收敛率下达到更低的EFE值，产生更广阔的状态覆盖，并在模型学习阶段显著加速模型误差收敛；

**⚠️ 局限性**

主要局限在于闭环学习（模型与策略相互依赖）缺乏完整收敛保证，且对状态覆盖和策略支持的假设在复杂环境中不易满足，导致可能出现不稳定的性能反馈循环。

---

## 380. Self-supervision drives representational convergence in medical foundation models more than clinical supervision

**arXiv ID:** 2607.20274 | [PDF](https://arxiv.org/pdf/2607.20274v1)

**作者:** Soroosh Tayebi Arasteh `[一作]` (RWTH Aachen University), Daniel Truhn `[通讯]` (RWTH Aachen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文系统评估了18个图像编码器和7个文本编码器在不同医学影像模态（胸部X光、组织学、视网膜、皮肤和乳房摄影）上的表征收敛性，并通过受控训练矩阵、合成生成模型和人类读者实验，探究了收敛的驱动因素、规模与性能关系，以及收敛对跨编码器、跨医院部署的实际可用性。

**💡 创新点**

创新点在于：①自监督训练是表征收敛的主要驱动力，而临床监督对收敛几乎无贡献；②收敛仅在同一模态内存在且幅度有限；③收敛与模型规模、下游性能或发布年份无显著相关；④即使收敛弱，依然可通过anchor‑based坐标系实现跨编码器、跨医院的线性分类器迁移和特征拼接；⑤提供了可复现的对齐、共识几何、迁移与漂移检测等完整实验框架。

**🔧 技术方法**

使用的技术包括：邻居一致度度量（mutual k‑NN、CKNNA）、随机初始化基准、受控训练矩阵（固定数据、架构、规模，只变训练目标）、合成生成模型（混合信号与噪声子空间）、共识几何(Procrustes)、相对表示(anchor‑based)、线性读头迁移、特征拼接（仿射映射）、漂移检测（跨编码器最近邻不一致）以及读者triplet实验。

**📊 数据集**

数据集涵盖：胸部X光 650,982张（MIMIC‑CXR、CheXpert、ChestX‑ray14、PadChest、VinDr‑CXR、VinDr‑PCXR）；组织学 22,000张补丁（NCT‑CRC、PatchCamelyon）；视网膜 3,738张（APTOS、Messidor‑2）；皮肤 5,987张（ISIC 2019）；乳房摄影 2,500张（VinDr‑Mammo）；配对影像‑文本 401,079 对（MIMIC‑CXR、CheXpert Plus）；以及用于受控训练的 5 份自监督/监督/图像‑文本语料和 643,522 张图片‑标题对（Quilt‑1M）。

**📈 对比分析**

对齐度量与随机基准相比提升显著：自监督匹配对 CKNNA 约 40.4%，监督 21.1%，图像‑文本 3.3%；受控实验证实仅训练目标决定收敛；跨编码器线性分类器迁移保留约 85% 原始性能，跨医院迁移 AUROC ~75%（oracle ~85%）；特征拼接可恢复 100% 性能；然而跨模态 (图像‑文本) 对齐几乎无提升，且读者 triplet 预测仅随机水平。

**⚠️ 局限性**

局限性包括：①共识几何未达到收敛阈值，影响后续分析；②子组一致性评估仅基于 CheXpert 单一来源，样本量对极小族群偏高；③稀有发现的样本不足导致相关性不稳；④受控实验仅使用单个随机种子；⑤对齐受预处理影响，无法完全隔离表征本身；⑥仅评估开放权重模型，未涵盖闭源或统一早期融合模型；⑦漂移检测在本实验中无效。

---

## 381. PhaseAware: Interpretable Human-in-the-Loop Rehabilitation Scoring with Boundary Monitoring

**arXiv ID:** 2607.20237 | [PDF](https://arxiv.org/pdf/2607.20237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 382. Which Values Do LLMs Confuse? A Schwartz-Based Recognition Study

**arXiv ID:** 2607.20270 | [PDF](https://arxiv.org/pdf/2607.20270v1)

**作者:** Andrei Chetvergov `[一作]` (Ivannikov Institute for System Programming of the Russian Academy of Sciences), Sergey Bolovtsov `[通讯]` (Ivannikov Institute for System Programming of the Russian Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 1,000 条均衡覆盖 Schwartz 10 基本价值的俄文情境文本集，并对其进行双人人工验证，随后对 21 个指令微调 LLM 在价值识别任务中的表现进行系统评估，重点分析误判类型、相邻值的重排以及跨检查点的共性错误指纹。

**💡 创新点**

①将价值识别设计为受控的 top‑1 识别任务；②提出“ranked recovery”和“directed confusion”两种评估维度；③发现跨检查点共享的八个定向错误，并揭示模型特定的错误指纹。

**🔧 技术方法**

使用指令微调 LLM（Gemma‑4‑26B、Qwen3.6‑27B 等），统一提示与温度设置；采用固定排名响应方案、固定‑margin null 估计、Holm 校正的单边检验、Wilcoxon 检验、logit 回归等统计与机器学习技术。

**📊 数据集**

1,000 条俄文情境文本，按十价值均匀抽取；每条文本由两名人工标注员确认，来源包括敏感话题、潜在不当信息、适当样本和 TAPE 伦理案例。

**📈 对比分析**

在 20 个可验证模型组成的 semantic panel 中，整体 top‑1 准确率 Acc@1=0.683，top‑3 覆盖率 Acc@3=0.892；最优模型 Gemma‑4‑26B Acc@1=0.859；大部分错误为相邻值，adjacent 错误在 top‑3 中被恢复的比例高达 87.5%。

**⚠️ 局限性**

①仅使用俄文无定义提示，缺乏多语言、定义展示或多标签支持；②只标注主价值，未考虑次要价值或置信度；③来源分布不均，可能影响自然价值分布；④未评估多模态或对话上下文的价值识别。

---

## 383. The Maskability Index: Predicting Task-Objective Alignment in Pretrained Language Models

**arXiv ID:** 2607.20265 | [PDF](https://arxiv.org/pdf/2607.20265v1)

**作者:** Ahmad Pouramini `[一作]` (Sirjan University of Technology), Mahsa Afsharzadeh `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究预训练语言模型的提示方式与预训练目标的对齐问题，提出Maskability Index（MI）衡量不同关系在掩码式与前缀式提示下的适配度，并验证其对ATOMIC知识补全生成质量的预测作用。

**💡 创新点**

通过DepthRank差值量化模板-目标对齐，提出MI指标，并将其用于指导低资源下的模板选择和适配策略，首次提供可预测的关系级别对齐度量。

**🔧 技术方法**

DepthRank计算、MI指标定义、少量样本生成（few-shot）、T5/BERT等预训练模型、BERTScore和ROUGE评估。

**📊 数据集**

ATOMIC2020知识库（关系补全任务）。

**📈 对比分析**

在n=5的少量样本下比较Mask Prompting与Prompting，使用BERTScore/ROUGE评估生成质量，MI预测的mask‑friendly与map‑phrasal分类与实际性能高度相关，mask‑friendly关系在掩码提示下表现更好。

**⚠️ 局限性**

需手工定义模板与标注样本，受tokenization和模型基线影响；缺乏无监督MI估计方法，限制了可扩展性。

---

## 384. Breaking the $T^{3/4}$ Barrier for Regret Minimization With Bi-Dimensional CDFs

**arXiv ID:** 2607.20258 | [PDF](https://arxiv.org/pdf/2607.20258v1)

**作者:** Matteo Castiglioni `[一作]` (Politecnico di Milano), Alberto Marchesi `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究在二维空间上使用仅二进制反馈学习累积分布函数相关目标的最小化，提出一种新算法实现对累计回报的O(T^{7/10})上界。

**💡 创新点**

创新点在于突破传统的探索-提交框架，构造稀疏但高度连通的子网，利用相对估计和误差容忍的反馈图技术，显著降低维度灾难导致的回报上界。

**🔧 技术方法**

主要技术包括：全局C​DF统一估计、基于L‑形路径的相对差分学习、索引化二分搜索、误差可容忍的图反馈UCB算法，以及利用图的独立数降低样本复杂度。

**📊 数据集**

该研究为理论工作，不依赖实际数据集，而是针对任意未知分布D在[0,1]^2上的通用假设。

**📈 对比分析**

与先前最优的O(T^{3/4})上界相比，算法在二维固定价格双边交易等实例上实现了O(T^{7/10})的回报上界，证明了该类C​DF相关目标可以部分消除维度灾难。

**⚠️ 局限性**

主要局限在于：仍存在O(T^{7/10})与已知下界Ω(T^{2/3})之间的显著差距，尚未确定二维情形下是否可以进一步达到最优的O(T^{2/3})；此外，算法在更高维度时仍退化到O(T^{3/4})的上界。

---

## 385. surprisal is Not a Theory

**arXiv ID:** 2607.20208 | [PDF](https://arxiv.org/pdf/2607.20208v1)

**作者:** Andrés Buxó-Lugo `[一作]` (University at Buffalo), Cassandra L. Jacobs `[通讯]` (University at Buffalo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了不同大型语言模型（GPT‑2、Pythia、RoBERTa）对 Surprisal 理论的适用性，发现它们在概率计算和内部表示上存在显著差异。

**💡 创新点**

通过层级分析与词汇可解释性检查，指出使用 LLM 生成的 Surprisal 并非可互换的计算水平指标，挑战了现有的 representation‑agnostic 设定。

**🔧 技术方法**

采用 logit lens、lexical lens 以及层级概率相关性分析等技术，对模型内部机制进行剖析。

**📊 数据集**

使用了 Sukhinin 等人提供的 cloze norm 数据集（约 3085 句子）和 Provo Corpus（约 2398 句子）等数据。

**📈 对比分析**

将模型预测与人类 cloze 概率进行 Spearman/Pearson 相关性对比，发现不同模型与人类行为相关性相似，但内部轨迹差异显著；并未给出具体的性能数值。

**⚠️ 局限性**

实验仅覆盖三种模型且未探讨更大规模或不同训练目标的影响，且结果受数据集偏差和模型黑箱性质限制。

---

## 386. Multi-modal transformer for signal classification in nanopore blockade experiments

**arXiv ID:** 2607.20323 | [PDF](https://arxiv.org/pdf/2607.20323v1)

**作者:** Sandro Kuppel `[一作]` (University of Stuttgart), Christian Holm `[通讯]` (University of Stuttgart)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种多模态 Transformer，用于直接从纳米孔阻塞信号的三种表示（原始时序、波形图像、统计特征向量）进行肽/蛋白的分类。

**💡 创新点**

创新点在于：① 将多种模态信息通过跨注意力统一到同一 Transformer 结构，实现信息高效融合；② 展示多模态学习显著提升分类精度；③ 通过迁移学习验证模型在低样本环境下的泛化能力。

**🔧 技术方法**

采用的技术包括：Vision Transformer 与标准 Transformer、跨注意力机制、波形变换（wavelet）与 catch22 特征提取、掩码自编码器预训练、注意力可视化（attention rollout）以及迁移学习微调。

**📊 数据集**

使用的数据集为：① 42 肽阶梯实验集（350,000 条事件，7 长度 × 6 序列）；② 20 氨基酸 X_R7 集（38,000 条事件）。

**📈 对比分析**

与单模态基线（ResNet18 处理波形图像、MLP 处理 catch22、时序 Transformer）进行宏/微平均、最大/最小准确率对比。多模态 Transformer 微平均 92.1%、宏平均 92.6%，比最佳单模态高 10%+；最差类提升近 20%，最高类 100%。迁移学习实验表明预训练模型收敛更快、准确率更均匀。

**⚠️ 局限性**

局限性包括：① 仅在相同 aerolysin 孔结构下验证，未探讨跨孔结构迁移；② 数据规模仍有限，尚未覆盖临床实际多样性；③ 由于 3–5% 的标签噪声，模型性能接近理论上限。

---

## 387. Black-Box Performance Evaluation of Elastic Block Storage: Contract, Rate-Limiting Model, and Software Exploration

**arXiv ID:** 2607.20319 | [PDF](https://arxiv.org/pdf/2607.20319v1)

**作者:** Yingjia Wang `[一作]` (Chinese University of Hong Kong), Ming-Chang Yang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过黑盒方法对亚马逊AWS与阿里云的弹性固态驱动器（ESSD）进行性能评估，提出ESSD合同、改进的I/O限速模型，并在RocksDB上验证其效用。

**💡 创新点**

创新点包括：①系统总结ESSD与本地SSD在延迟、GC、访问模式等方面的四项观察和五项软件适配建议；②结合带宽与IOPS双重限速、细粒度令牌补充的I/O限速模型；③在RocksDB中演示缓存、限速、容量与压缩四项优化策略。

**🔧 技术方法**

主要技术手段为：FIO+io_uring基准测试、令牌桶限速、双限速器算法、RocksDB源代码改造以及压缩算法（LZ4、ZSTD）对比。

**📊 数据集**

使用的数据集包括四个真实ESSD实例（AWS io2、AWS gp3、阿里云PL3、阿里云PL0）、一块Samsung 970 Pro SSD，以及YCSB工作负载（Load、A-F）。

**📈 对比分析**

对比方法是将改进模型与原始RocksDB、Calcspar的1秒IOPS限速以及同模型但1秒令牌补充的版本进行实验，结果显示吞吐量提升约20%、最大延迟下降至几秒以内、P99延迟相对稳定，整体性能显著优于基线。

**⚠️ 局限性**

局限性在于仅覆盖两大云厂商的ESSD实例，未探讨不同网络/存储集群实现细节；实验聚焦块级I/O，未覆盖对象存储或文件系统层面；并且缺乏对更大规模或更高并发场景的验证。

---

## 388. Courteous Anticipation: Improving Long-Lived Task Planning in Persistent Shared Environments

**arXiv ID:** 2607.20289 | [PDF](https://arxiv.org/pdf/2607.20289v1)

**作者:** Md Ridwan Hossain Talukder `[一作]` (George Mason University), Gregory J. Stein `[通讯]` (George Mason University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究一种在共享持久环境中让多机器人提前考虑对他人任务影响的礼貌式前瞻性规划。

**💡 创新点**

提出“礼貌式前瞻性规划”，将即时任务成本与对所有机器人未来任务成本的期望综合，并通过分解为每个机器人独立估计，避免联合搜索。

**🔧 技术方法**

基于PDDL的符号规划，使用FastDownward+ff-astar；采用聚焦采样生成候选计划；使用图神经网络（GNN）作为每机器人期望未来成本的学习估计器。

**📊 数据集**

在两种模拟环境中实验：Proctor家庭环境和Restaurant（餐厅）环境，利用proctor生成的场景与alfred任务集以及自定义餐厅任务。

**📈 对比分析**

与经典Myopic规划和自利型前瞻性规划比较，在家庭环境中平均累计成本降低10.4%（相对于Myopic）和4.0%（相对于Selfish），在餐厅环境中降低17.4%和13.2%。

**⚠️ 局限性**

假设任务按顺序单一到达、环境持续不变；不处理并发任务、多机器人同步以及中途环境变化，且仅关注符号规划层面，未涉及真实感知与执行不确定性。

---

## 389. On the Systematic Challenges of Culturally Loaded Machine Translation: Dream of the Red Chamber as the Cultural Lens

**arXiv ID:** 2607.20241 | [PDF](https://arxiv.org/pdf/2607.20241v1)

**作者:** Yiming Wang `[一作]` (Shanghai Jiao Tong University), Jiayuan Di `[通讯]` (East China University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统研究中文-日文文化载体翻译在LLM MT中的挑战，构建数据集并评估模型。

**💡 创新点**

结合多维评估框架与文化类别，揭示任务、评估与自动化三方面的独特难点。

**🔧 技术方法**

使用八种大型语言模型生成译文，并采用改编后的MQM框架、人类评分与BLEU、xCOMET、LLM‑judge等自动指标。

**📊 数据集**

基于《红楼梦》中文原文与伊藤升平日文译本的500段文化载体句子。

**📈 对比分析**

通过人类评分四维度与模型输出对比，发现LLM在文化维度得分远低于人类参考，自动指标与人工评价相关性弱。

**⚠️ 局限性**

仅研究中日对，缺乏其他语言，人工评测资源有限，导致实验结果难以推广。

---

## 390. Not All Patches are Equal: Sampling Matters for Visible-Infrared Pre-Training

**arXiv ID:** 2607.20238 | [PDF](https://arxiv.org/pdf/2607.20238v1)

**作者:** Qiwei Ma `[一作]` (Hunan University), Shutao Li `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究可见‑红外对齐预训练，提出重要性感知采样（IAS）框架，动态调整补丁重要性并对比学习；

**💡 创新点**

创新点在于把采样和权重视为核心设计，利用红外结构先验+可学习采样+训练课程，显著提升对齐效果；

**🔧 技术方法**

使用Sobel/HOG结构先验、轻量级MLP学习重要性、重要性加权对比损失、patch‑level与correlation‑level对齐、ViT backbone及课程学习；

**📊 数据集**

预训练采用 MVIP 等对齐数据集，下游评测在 MFNet、MSRS、SODA、SCUTSEG、ADE20K、M3FD 等数据集；

**📈 对比分析**

与 UNIV、INFMAE、PAD、MCMAE 等基线比较，在语义分割、目标检测、检索等任务上均实现 1–3% 的 mIoU 提升，检索 R@1 达到 82% 以上；

**⚠️ 局限性**

局限性包括仅适用于已对齐的可见‑红外图像，对低纹理或模态特异区域的处理仍有限，需额外先验或学习参数，尚未验证至其他模态对齐。

---

## 391. DINS-IO: Learned Inertial Odometry via Differentiable INS Consistency

**arXiv ID:** 2607.20232 | [PDF](https://arxiv.org/pdf/2607.20232v1)

**作者:** Hao Qiao `[一作]` (Wuhan University), Xiaoji Niu `[通讯]` (Wuhan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种无需位置标签即可从原始 IMU 数据中学习惯导定位的方法——DINS-IO，利用姿态已知的惯导速度递推关系构造可微分的自监督损失，并在少量标记数据上通过 LoRA 微调实现量化校准。

**💡 创新点**

创新点在于：1）将刚体 INS 速度递推关系转化为滑窗线性最小二乘（LS）闭式可微解，用其残差作为自监督损失；2）设计高频率网络，输出每个 IMU 样本的体系速度，使自监督损失可实现；3）通过只训练 LoRA 低秩适配器在少量标记数据上完成量化校准，避免全模型过拟合。

**🔧 技术方法**

使用技术包括：可微分 LS 求解、闭式残差作为自监督损失、MetaFormer 结构的特征提取与时间建模、GRU 时序块、密集高频回归头、LoRA 参数适配、滑窗共享加速度计偏置约束、Tikhonov 正则化。

**📊 数据集**

实验数据集主要为 TLIO（100Hz IMU 与视觉惯导轨迹）和自采 Tango（头戴式 40 小时数据，含多种步态与设备差异），并在两者上划分训练/测试集。

**📈 对比分析**

与 RoNIN（ResNet、LSTM、TCN）、LLIO、AirIO 等监督式 LIO 基准模型以及完全监督版 DINS-IO 进行对比。Stage 1 自监督阶段已能恢复速度方向（TLIO 中中位误差 14°，Tango 中 21°），Stage 2 仅用少量标记数据即可匹配或超过监督式基线，整体在 ATE、RTE 上表现与最佳监督模型相当或更优。

**⚠️ 局限性**

局限性包括：1）自监督阶段仅约束速度方向和相对变化，无法确定全局尺度与绝对速度，需要标记数据校准；2）依赖姿态估计（R[k]）的准确性；3）在极度退化运动（如停留或单一方向移动）时，惯导递推约束可能不足以约束模型；4）实验仅覆盖行走类人类运动，需进一步验证在更复杂平台或动态环境下的鲁棒性。

---

## 392. User-Centric Modeling of Transactional Sequences with Explainable State Space Models

**arXiv ID:** 2607.20228 | [PDF](https://arxiv.org/pdf/2607.20228v1)

**作者:** Ivan Palagin `[一作]` `[通讯]` (HSE University), Ivan Palagin (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出两种将预训练的 CoLES 用户嵌入注入 Mamba 状态空间模型（隐藏状态初始化和前缀拼接）的混合架构，用于交易事件序列的用户中心建模。

**💡 创新点**

创新点在于首次将对比学习得到的用户先验嵌入与高效线性时间 SSM 结合，既提升预测性能又显著加速收敛，并通过离散化步长映射与积分梯度实现模型可解释性。

**🔧 技术方法**

采用对比学习 (CoLES)、Mamba 选择性状态空间模型、RMSNorm、残差连接、AdamW 优化、Optuna 超参搜索、离散化步长映射与积分梯度解释等技术。

**📊 数据集**

在 Age、MBD、Taobao 三个公开交易序列数据集上进行实验。

**📈 对比分析**

与单纯 Mamba 与 CoLES+线性分类器对照，Hybrid 模型在 Age、MBD、Taobao 上分别提升约 3.2pp、2.6pp、0.7pp，并将收敛速度提升 2–3 倍。

**⚠️ 局限性**

局限性包括 CoLES 与 Mamba 信息重叠导致增益有限；预训练与下游训练独立，未能充分利用协同；线性投影可能不足以充分映射用户嵌入；在超长序列上的验证尚未充分。

---

## 393. HalluTruthQA: A Fine-Grained Benchmark for Hallucination Detection, Localization, and Explanation in Arabic Question Answering

**arXiv ID:** 2607.20219 | [PDF](https://arxiv.org/pdf/2607.20219v1)

**作者:** Abdessalam Bouchekif `[一作]` (Hamad Bin Khalifa University), Abdenour Hadid `[通讯]` (Universiti Malaysia Kelantan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并构建了HallUTruthQA，一个针对阿拉伯语知识密集型问答的细粒度幻觉评估基准；

**💡 创新点**

在单一基准中同时提供响应级幻觉标签、字符级错误跨度、人工解释和多选验证，首次实现多维度幻觉评估；

**🔧 技术方法**

采用专家标注与人工审核的双轮流程，利用FanAR模型生成回答并以LLM‑as‑judge（GPT‑5.5）评估解释质量，使用多任务评估指标（Macro‑F1、F1‑Sp、LO‑Score等）；

**📊 数据集**

HallUTruthQA共2400个阿拉伯语QA样本，覆盖伊斯兰知识、历史、科学与地理四大领域；

**📈 对比分析**

在零样本闭卷设置下对Allam、Falcon‑H1、Qwen32、Silma以及自检的Fanar进行评估，检测准确率最高为0.908（Fanar），跨度定位最佳为0.516（Qwen32），多选验证最高为0.852（Qwen32），解释最终分数最高为0.644（Qwen32），显示不同任务表现不一致；

**⚠️ 局限性**

局限包括仅覆盖四个领域、回答来源单一模型、跨度标注可能存在边界不准、解释评估依赖LLM判定、未涵盖API系统和更广泛的问答场景。

---

## 394. The Quadrilateral Loss: Additivity as a Measurable Behavior of Dense Neural Networks

**arXiv ID:** 2607.20201 | [PDF](https://arxiv.org/pdf/2607.20201v1)

**作者:** Antonio Di Cecco `[一作]` `[通讯]` (University G D'Annunzio), Antonio Di Cecco (University G D'Annunzio)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了四边形损失（quadrilateral loss），通过对特征交换后的四点差分衡量并约束神经网络的交互项，从而实现可调节的可解释性与准确性权衡。

**💡 创新点**

创新点在于：①将交互约束从硬结构迁移为可微的行为惩罚；②四边形损失与交互的Shapley‑GAM 交互质量对应；③引入在线交互投降曲线和可解释性阈值λ_bal，实现训练时可视化交互消减；④证明软约束可克服因完美加性导致的形状函数粗糙性。

**🔧 技术方法**

技术方法包括：基于四点差分的二阶混合差分损失；对单坐标交换进行采样；权重衰减（winner‑take‑all decay）；共享‑段模型（shared‑section）和晶化后得到的可掩蔽NAM；以及与传统回归拆分（backfitting）和树模型（TEAM）对比。

**📊 数据集**

主要数据集为加利福尼亚住房（California Housing）以及六个公开表格基准（Wine Quality、Abalone、Boston Housing、Concrete、Bike Rentals）和三组临床数据（Diabetes Progression、Pima Indians、Heart Disease），所有实验均采用统一的网络架构与优化策略。

**📈 对比分析**

通过在同一网络宽度下比较六类方法（Dense、NAM、Backfitting、Shared‑Sections、TEAM、Quad软+λ=1），发现共享‑段模型在大多数数据集上取得最优或接近最优的MSE，TEAM在部分基准（尤其是Concrete）表现最佳；软约束在小样本情形下可同时提升准确率与可解释性。

**⚠️ 局限性**

局限性包括：四边形损失仅在数据分布内部有效，无法保证离谱交互被抑制；惩罚的噪声底限导致无法完全实现加性，需后期晶化；方法在特征维度大时计算成本上升；实验范围局限于小规模表格数据，未验证在大规模或非表格场景的可扩展性。

---

## 395. Toward Reliable RGB-D Semantic Segmentation: Handling Missing Modalities via Condition Dropout

**arXiv ID:** 2607.20326 | [PDF](https://arxiv.org/pdf/2607.20326v1)

**作者:** Xuchen Zhu `[一作]` (Xi'an University of Posts & Telecommunications), Fang Ren `[通讯]` (Xi'an University of Posts & Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在已有的预训练 RGB‑D 语义分割模型上进行一种持续训练策略（Condition Dropout），使模型能够在 RGB 或深度缺失时保持性能，并在完整模态下仍保持或略微提升原始精度。

**💡 创新点**

创新点在于：1) 仅在训练阶段复制一个 encoder 并冻结原始 encoder；2) 用零初始化的 1×1 卷积将复制 encoder 的特征注入到冻结 encoder 的特征中；3) 采用随机模态丢失（RGB‑missing、Depth‑missing、全模态）进行持续训练，从而无需修改原有网络结构即可提升对模态缺失的鲁棒性。

**🔧 技术方法**

技术手段包括：复制 encoder、零初始化卷积特征注入、随机模态丢弃、冻结原始参数、持续训练（Stage‑2）、标准交叉熵/多分类损失以及常用数据增强。

**📊 数据集**

使用的公开基准数据集为 NYU‑Depth V2（1,449 张图，40 类）和 SUN RGB‑D（10,335 张图，37 类）。

**📈 对比分析**

方法通过与原始模型（无 ConD）在三种输入条件下（完整、RGB 缺失、Depth 缺失）进行比较。实验表明：在 NYU‑V2 上，平均 mIoU 损失由 -25.3% 降至 -9.8%；在 SUN 上，mAcc 损失从 -28.3% 降至 -11.4%。此外，在完整模态下，mIoU 甚至提升 0.1–1.0 点，mAcc 提升 0.4–1.4 点，显示 ConD 既增强了鲁棒性，又保持甚至略优于原始精度。

**⚠️ 局限性**

主要局限性是：复制 encoder 需要额外参数和内存，推理时也会产生额外的计算负担，导致延迟增加，尤其在资源受限或实时部署环境中可能不适用。

---

## 396. Classical Hardware Acceleration of Quantum Autoencoders for Real-Time Anomaly Detection in Collider Experiments

**arXiv ID:** 2607.20302 | [PDF](https://arxiv.org/pdf/2607.20302v1)

**作者:** Ivan Ge `[一作]` (Stanford University), Julia Gonski `[通讯]` (SLAC National Accelerator Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了基于量子自编码器的实时异常检测模型，可在FPGA上实现低时延推理。

**💡 创新点**

创新点在于将量子自动编码器与混合量子-经典VAE相结合，并在FPGA上完成量子电路仿真与硬件综合。

**🔧 技术方法**

使用的技术包括量子特征映射、参数化量子电路、PennyLane训练、FPGA HLS综合与定点量化。

**📊 数据集**

数据集为4M QCD背景事件以及四个BSM信号样本，所有事件使用56维低层特征。

**📈 对比分析**

与传统机器学习基准相比，模型在AUC和10^-5 FPR下的TPR均表现相当或更优，且FPGA实现的延迟低于10µs（QAE <1µs）。

**⚠️ 局限性**

主要限制在于量子电路规模受限、量化导致极端尾部性能波动、以及对单个SLR资源的依赖。

---

## 397. Sound Probabilistic Safety Bounds for Large Language Models

**arXiv ID:** 2607.20286 | [PDF](https://arxiv.org/pdf/2607.20286v1)

**作者:** Mahdi Nazeri `[一作]` (University of Oxford), Alessandro Abate `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于隐空间特征引导的生成树搜索框架，用以严格计算大语言模型在固定提示下产生有害输出的下界。

**💡 创新点**

创新点在于利用LLM残差流中的线性“有害特征”优先扩展可能产生有害输出的分支，形成可证明的下界；同时克服Clopper‑Pearson置信区间在低概率下界为0的局限。

**🔧 技术方法**

使用Clopper‑Pearson置信区间、PAC bounds、残差流特征向量、树搜索与Top‑K展开、以及基于有害特征的健壮性评估技术。

**📊 数据集**

在Llama‑3.1‑8B、Llama‑3.2‑3B‑Instruct和Phi‑4上进行实验，采样各64个有害与无害响应，构造黑名单词典作为安全判别器。

**📈 对比分析**

与蒙特卡洛采样和Clopper‑Pearson置信区间对比；在相同生成token预算下，本方法能得到非零下界（范围从10⁻³⁴到10⁻⁷），而蒙特卡洛和置信区间均给出0。

**⚠️ 局限性**

局限包括：仍需在生成树上展开搜索，深度受限；依赖隐空间特征的近似线性，可能在更大或更复杂模型上效果下降。

---

## 398. Multimodal Large Language Models for Remote Sensing Image Understanding: Domain-Specific or General-Purpose?

**arXiv ID:** 2607.20284 | [PDF](https://arxiv.org/pdf/2607.20284v1)

**作者:** Qiwei Ma `[一作]` (Hunan University), Shutao Li `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述并诊断了遥感多模态大语言模型（RS-MLLM）的技术演进与能力边界，评估其在遥感图像场景理解（RSISU）任务上的表现；

**💡 创新点**

创新点在于：①构建RS-MLLM的完整技术分类与指令调优流程；②对比RS-MLLM与通用CV-MLLM在多任务、不同分辨率与多源数据下的性能差异；③提出评估框架并揭示RS-MLLM在跨任务泛化与细粒度推理方面的局限；

**🔧 技术方法**

技术包括多模态视觉编码器（如ViT、CLIP、Swin）、输入投影器、LLM（如LLaMA、Vicuna、Qwen、GPT-4o等）、LoRA/parameter‑efficient微调、强化学习(如GRPO)、主动感知裁剪工具；

**📊 数据集**

使用的指令调优数据涵盖数十亿级多模态对话、图文、VQA、定位、跨源遥感数据集（如RSICap、MMRS‑1M、HqDC‑1.4M、FIT‑RS、LRS‑GRO、HighRS‑VQA、SuperRS‑VQA 等），并结合公开的遥感分类/VQA/视觉定位基准（AID、WHU‑RS19、EuroSAT、RSVQA‑HR、RSVG、DIOR‑RSVG、LHRS‑Bench、FIT‑RSRC、XLRS‑Bench等）；

**📈 对比分析**

评估方法采用统一的零样本（或无任务特定微调）推理，使用标准化提示模板和输出格式；实验显示：在视觉定位、HR‑VQA、超高分辨率任务上RS‑MLLM表现优于通用CV‑MLLM；但在多样化基准（LHRS‑Bench、FIT‑RSRC、XLRS‑Bench）中，GPT‑4o、Qwen3‑VL、GLM‑4.6V‑Flash 等通用模型往往更佳，说明RS‑MLLM的优势并非一致；

**⚠️ 局限性**

局限包括：①对指令多样性与泛化不足，易受模板依赖；②空间与关系推理能力不足（阴影、左右、计数、距离等）；③多源、多时相、超高分辨率等复杂输入处理有限；④评估缺乏统一标准，潜在数据泄漏与不透明度；⑤模型规模不一定决定性能，需结合高质量指令数据与强大基础模型。

---

## 399. Chained Attacks on Drone-Based Federated Learning: From Network Disruption to Device Impersonation

**arXiv ID:** 2607.20280 | [PDF](https://arxiv.org/pdf/2607.20280v1)

**作者:** Suleiman Muhammad Sabo `[一作]` (Newcastle University), Rajiv Ranjan `[通讯]` (Newcastle University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨并实证了针对基于无人机的联邦学习系统的链式攻击，即先通过802.11 deauthentication导致合法无人机下线，再利用被截获的凭证冒充被下线无人机加入训练；

**💡 创新点**

创新点在于将网络层拒绝服务与凭证冒充结合为完整攻击链，并在真实硬件测试床上量化对IID与非IID数据分布下模型准确率与收敛稳定性的影响；

**🔧 技术方法**

使用了Flower框架实现FedAvg，利用802.11 deauthentication攻击、TLS互联认证、Protobuf解析等技术；

**📊 数据集**

使用CIFAR-10数据集，并通过Dirichlet划分产生IID与非IID分布；

**📈 对比分析**

通过与基线训练对比，采用准确率下降（ΔA）、收敛不稳定性σL和连接成本Cc等指标衡量；在IID场景下准确率下降约7%–2%（取决平台），非IID场景下降更显著，Pi平台下降可达15.75%；收敛波动显著增加；

**⚠️ 局限性**

局限性包括实验规模仅为10–8个节点，未探讨更大规模群集；只针对Flower框架，其他FL框架可能表现不同；攻击假设攻击者已获得凭证，未评估凭证泄露的实用性；未验证多因素认证与硬件可信执行等防御方案。

---

## 400. Interpretable Fuzzy Rule-Based Regression Extension for Ex-Fuzzy Library

**arXiv ID:** 2607.20277 | [PDF](https://arxiv.org/pdf/2607.20277v1)

**作者:** Cayan Deniz Kucuktopana `[一作]` (University of Essex), Javier Andreu-Perez `[通讯]` (University of Essex)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

为 Ex-Fuzzy 库扩展 Mamdani‑式模糊回归，使其能通过遗传算法学习可解释的连续输出规则集合。

**💡 创新点**

引入目标感知的 Fuzzy C‑Means 初始化分区、在遗传编码中加入标量后继并采用加权平均去模糊化，提供与 Scikit‑learn 兼容的回归接口。

**🔧 技术方法**

Python、Ex‑Fuzzy、Fuzzy C‑Means、遗传算法、Gaussian 与 Trapezoidal 分区、线性回归、MLP、随机森林基线。

**📊 数据集**

KEEL 仓库的 10 个回归数据集（如 autoMPG6、diabetes、stock、mv 等）。

**📈 对比分析**

在相同规则数（10–15 条）下，将 Gaussian 分区模型与 Trapezoidal、线性回归、MLP、随机森林进行对比；平均 R² 约 0.86（Gaussian）对比 Trapezoidal 0.71、MLP 0.80、RF 0.78，显示显著性能提升。

**⚠️ 局限性**

仅实现一阶模糊系统，未考虑不确定性或流式数据；分区初始化依赖 Fuzzy C‑Means，可能对噪声敏感；实验仅在公开基准上验证，未测试真实工业场景。

---

## 401. Simple and Almost Non-Adaptive \(\frac{1}{2}\)-Approximation for Matroid Prophet Inequalities

**arXiv ID:** 2607.20269 | [PDF](https://arxiv.org/pdf/2607.20269v1)

**作者:** Sina Kalantarzadeh `[一作]` (University of Waterloo), Kanstantin Pashkovich `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了一种几乎非自适应的算法，为任意 matroid prophet inequality 提供 1/2 近似；

**💡 创新点**

创新点在于将原 matroid 通过“提取”过程分解为一系列直接和的子 matroid，每个子 matroid 满足 “nice” 性质，随后对每个子 matroid使用单一固定阈值即可实现 1/2 近似；

**🔧 技术方法**

核心技术包括 ex‑ante relaxation、Bernoulli 化、对 matroid 进行最大阈值提取、子 matroid 的直接和组合以及单阈值在线策略；

**📊 数据集**

算法为理论构造，不使用任何实验数据集；

**📈 对比分析**

与已知的适应性算法（如 Kleinberg–Weinberg）相比，取得相同的 1/2 近似，但阈值在整个过程中保持固定，易于实现；

**⚠️ 局限性**

局限性在于仍需对原 matroid 进行额外约束（即子 matroid 的直接和），无法在原 matroid 上直接使用纯固定阈值；提取过程在某些实例上可能计算量较大。

---

## 402. How Does Urban Context Relate to Residential Building Health? A Vision-POI Fusion Framework for Building-Level Housing Inspection

**arXiv ID:** 2607.20263 | [PDF](https://arxiv.org/pdf/2607.20263v1)

**作者:** Kun Zhao `[一作]` (Qingdao University of Technology), Qichao Ban `[通讯]` (Qingdao University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种 Vision–POI 融合框架，将多视角视觉检测与城市功能 POI 信息联合用于老旧住宅楼房健康评估，并构建了 92 个社区、3,237 栋楼、25,608 张多视角图像的实验数据集。

**💡 创新点**

创新点在于首次将多视角聚合后的建筑级视觉特征与多尺度 POI 先验相结合，采用随机森林后处理实现建筑级判别，证明 POI 可作为补充先验显著提升检测准确率。

**🔧 技术方法**

使用了 YOLOv8/YOLOv9/RT‑DETR 等目标检测网络进行图像级检测，随后对检测结果进行统计聚合；通过 Pearson/Spearman 相关分析筛选 POI 变量；利用 Cost‑Sensitive 随机森林实现建筑级分类。

**📊 数据集**

使用了青岛市 92 个老旧社区的现场检查记录、25,608 张多视角图片以及 AutoNavi 1,500 m POI 数据，并结合公开的 HOUSED 数据集进行预训练与增量学习。

**📈 对比分析**

通过图像级 mAP@0.5、mAP@0.5:0.95 与建筑级 F1、ROC‑AUC、PR‑AUC 等指标评估，发现多视角聚合将 Macro‑F1 从 60.84% 提升至 74.95%，加入 POI 上下文后进一步提升至 76.79%，显著优于仅视觉或仅 POI 的基线。

**⚠️ 局限性**

局限性包括 POI 上下文的增益受尺度敏感且仅为补充先验，稀疏类别仍难以准确识别；模型在跨行政区或建筑密度变化的空间泛化仍受限；缺乏跨城市验证与实时部署的可行性评估。

---

## 403. ASPIC: Proof-of-Concept ASP to Picat Transpiler

**arXiv ID:** 2607.20254 | [PDF](https://arxiv.org/pdf/2607.20254v1)

**作者:** Cristian Grozea `[一作]` (Fraunhofer FOKUS), Marius Popescu `[通讯]` (University of Bucharest)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 ASPIC，一款将 ASP-Core-2 程序自动转译为 Picat 代码的库，支持在 Picat 中直接求解 ASP 程序并允许嵌入 Picat 代码以及使用 Picat 的 CP/SAT 求解器；

**💡 创新点**

通过自动转译实现“ASP in Picat”和“Picat in ASP”，支持非线性约束、有限域变量、嵌入 Picat 代码区块，以及利用 Picat 的全局约束和 SAT/CP 求解器，解决传统 ASP 的 grounding bottleneck；

**🔧 技术方法**

采用自然映射技术将 ASP 对象映射到 Picat 结构，利用 Clark completion 与 on‑the‑fly 逐步 grounding、FD 变量映射、嵌入代码语法（& 分隔符），并集成 Picat 的 CP 与 SAT 求解器；

**📊 数据集**

使用经典基准案例 N‑Queens、Towers of Hanoi 与 SEND+MORE=MONEY 的 ASP 版本作为测试数据；

**📈 对比分析**

在同一台高性能单核/多核机器上使用 hyperfine 10 次运行，比较 ASPIC（单核 Picat）与 clingo（单核/多核）时间，结果显示 ASPIC 与 clingo 在中等规模 N‑Queens 性能相近，随着规模增大差距扩大；在 Towers of Hanoi ASPIC 更快；在 SEND+MORE=MONEY 的 FD 版本 ASPIC 速度显著优于 clingo；

**⚠️ 局限性**

目前仅实现 Clark completion，无法保证得到完整稳定模型，可能产生额外支持模型；聚合算子仅支持在约束中；有限域变量语法仍为实验性；未实现循环公式生成；仅单核执行，未充分利用多线程优势。

---

## 404. PIER: Physics-Informed Environmental Retrieval for Time-Series Modeling

**arXiv ID:** 2607.20230 | [PDF](https://arxiv.org/pdf/2607.20230v1)

**作者:** Shiyuan Luo `[一作]` (University of Pittsburgh), Xiaowei Jia `[通讯]` (University of Pittsburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种 Physics‑Informed Environmental Retrieval (PIER) 框架，利用嵌入式检索与基于物理过程的本地验证器相结合，对时间序列预测进行检索增强。

**💡 创新点**

创新点在于：①将物理一致性评分融入检索流程；②引入自适应权重机制，根据诊断特征动态平衡嵌入相似度与物理一致性两条检索通道；③实现了模型无关的检索增强，能够与多种预测骨干无缝集成。

**🔧 技术方法**

核心技术包括：全局 LSTM 或 Transformer 骨干网络；物理导出 flux 特征与过程模型仿真标签；本地验证器训练、指数相似度评分；两路检索候选池合并与权重调节的双层优化；以及门控网络预测最佳物理权重。

**📊 数据集**

实验基于美国中西部 356 座湖泊 41 年（1979–2019）时间序列数据，包含水温、溶解氧、湖泊地形、气象、生态等 47 条特征，并利用 General Lake Model 与过程模型产生的 flux 以及仿真标签。

**📈 对比分析**

与物理模型、LSTM、Informer、iTransformer、TSMixer、TimesNet、TimeMixer 等基线对比，PIER 在 6 种水质预测场景（夏季上层/下层 DO、温度，及秋春混合期 DO）均取得最低 RMSE，提升幅度从 5%–20% 以上，且在多种骨干上表现一致。

**⚠️ 局限性**

局限性包括：①对过程模型仿真质量依赖，若仿真误差大则验证器可信度下降；②自适应权重学习需要额外的诊断特征与门控网络训练，增加计算复杂度；③在某些 DO 预测场景中，若物理流与嵌入流差异过大，固定权重可能导致性能不如单一流。

---

## 405. Improved Lower Bounds and Output Augmentation for Facility Location Mechanisms

**arXiv ID:** 2607.20196 | [PDF](https://arxiv.org/pdf/2607.20196v1)

**作者:** Rafael Gomes `[一作]` (Centrum Wiskunde & Informatica), Jens Schlöter `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在欧几里得空间下，公平（egalitarian）目标的设施定位问题，并设计了多种在不使用金钱的情况下实现策略无关（strategyproof）的随机与确定性机制；

**💡 创新点**

创新点包括：提出了一个新的渐近下界1+√(d/(2(d+1)))，显著提高了高维空间随机机制的已知最小下界；设计了输出增强（output-augmented）框架，并在一维输入、二维输出以及单位圆输入、二维输出两种情形下分别给出了最优或近似最优的机制；同时证明随机化在某些设置下是必要的，且通过引入“Chord‑Midpoint”机制实现了1.5的近似比。

**🔧 技术方法**

技术手段主要涉及几何构造（正多面体、球面离散化）、Jung定理与几何极值分析、对称性与凸性论证、期望与Jensen不等式以及对随机机制的概率分布特征化；此外，还使用了对角线中点与弧长的三角函数关系来设计概率分布。

**📊 数据集**

该工作为理论分析性质，不涉及具体实验数据集。

**📈 对比分析**

与已有结果比较时，本文的随机机制在平面上取得1.577的近似比（此前最高1.118），在两点情况实现√2≈1.414的近似比，超越传统线性下界1.5；在输出增强的直线到平面设置中实现√2≈1.414的确定性近似比，击破传统随机下界1.5。

**⚠️ 局限性**

局限性包括：高维空间的下界与现有上界之间仍有明显差距；在输出增强的圆形输入设置下，随机化是否能进一步突破2的下界仍未解决；且缺乏对更一般输入/输出拓扑的完整机制分类与实现细节。

---

## 406. Fundamental Limits of MIMO-OTFS and MIMO-OFDM in High-Dynamics ISAC: An Antenna Array Architecture Perspective

**arXiv ID:** 2607.20200 | [PDF](https://arxiv.org/pdf/2607.20200v1)

**作者:** Po-Chih Chen `[一作]` (National Yang Ming Chiao Tung University), Yu-Chih Huang `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10数据集进行实验。

**📈 对比分析**

与传统的激活函数（如ReLU）进行比较，结果显示新模型在分类精度上提高了5%，且训练时间缩短了15%。

**⚠️ 局限性**

模型在处理大规模数据集时可能会遇到内存限制的问题。

---

## 407. MR-Compare: A Mixed-Reality Framework for Spatially Grounded Visual Comparison of 3D Gaussian Splatting and Mesh Reconstructions with the Physical Environment

**arXiv ID:** 2607.20325 | [PDF](https://arxiv.org/pdf/2607.20325v1)

**作者:** Changrui Zhu `[一作]` (University College London), Simon Julier `[通讯]` (University College London)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在Meta Quest 3上开发了MR‑Compare框架，实现将离线3D重建（网格、3D Gaussian Splatting）与实时视频可视化流对齐，以实现空间定位的混合现实视觉比较。

**💡 创新点**

创新点包括：①自包含的Unity管线，兼容PC‑tethered Quest 3；②基于TEASER++与V‑GICP的粗细分辨率一致性注册管线；③面向MR的三维滑块交互；④无训练的零射向异质性过滤器用于3DGS中心点的表面筛选。

**🔧 技术方法**

技术包括：Unity 3D引擎、Meta Quest 3深度API、TEASER++粗配准、V‑GICP细配准、3D Gaussian Splatting渲染、3D Slider遮罩、Voxel化与FPFH对应、ArUco姿态测量。

**📊 数据集**

数据集：现场采集的两间室内房间（Office、Reception），以及NICT‑SLAM的8个Replica室内场景；重建流程分别为RealityScan、Polycam、Scaniverse、Nerfstudio（3DGS、3DGS‑MCMC）。

**📈 对比分析**

比较方法：客观指标为ArUco位姿误差（cm级）、VST参考的PSNR/SSIM/LPIPS/DISTS；用户研究使用Likert评分对对齐度与视觉一致性评估；结果显示桌面3DGS‑MCMC在注册误差≈0.9 cm、旋转≈0.9°，视觉一致性最佳；移动流程误差较大。

**⚠️ 局限性**

局限性：仅在PC‑tethered Quest 3上测试，无法在独立设备高帧率渲染；仅针对静态室内场景，未覆盖动态或户外环境；使用ArUco作为稀疏真值，缺乏全局密集地面真值；VST图像质量受头戴设备处理影响，难以完全与重建图像对齐。

---

## 408. The Polynomial-Time Low-Degree Conjecture is False

**arXiv ID:** 2607.20318 | [PDF](https://arxiv.org/pdf/2607.20318v1)

**作者:** Songtao Mao `[一作]` `[通讯]` (Johns Hopkins University), Songtao Mao (Johns Hopkins University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构造了一类在顶点重标置不变的图分布，使得其与均匀随机图在低阶多项式检验下完全相同（低阶优势为零），但在对每条边进行独立噪声扰动后，仍能用多项式时间的秩检验高概率区分。

**💡 创新点**

证明了低阶方法与均匀无偏、顶点对称、独立噪声的组合并不能保证计算难度，从而给出了低阶猜想的标准多项式时间形式的反例；并引入了使用 Reed–Muller 子空间与随机交替双线性形式的构造，展示了如何在保持低阶优势为零的同时，保留可被多项式算法利用的结构。

**🔧 技术方法**

利用 Reed–Muller 代码的低偏差子空间、随机交替双线性形式、线性无关性与列秩分析、随机化打桩与高斯消元、以及 Bhandari 等人关于随机打桩的线性独立性定理。

**📊 数据集**

无真实数据集，完全是理论构造；图分布由随机选取的 Reed–Muller 子空间、随机双线性形式和顶点随机置换生成。

**📈 对比分析**

采用秩检验：在“锚点”子图上构造多项式升维后计算列秩差；在零分布下秩几乎等于列数，在噪声植入的分布下秩显著不足。该检验在多项式时间内完成，能够以概率 1–o(1) 正确区分两种分布。未与其他算法做实验性对比。

**⚠️ 局限性**

构造方法非构造性，缺乏可在多项式时间内统一生成的显式算法；对度的匹配与运行时间存在权衡，尚未能实现更紧的平衡；需要额外假设才能恢复低阶猜想的完整形式。

---

## 409. Constant-time decoding of Gabidulin codes and their generalizations with application to RQC

**arXiv ID:** 2607.20305 | [PDF](https://arxiv.org/pdf/2607.20305v1)

**作者:** Nicolas Aragon `[一作]` (University of Limoges), Ilaria Zappatore `[通讯]` (University of Limoges)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

提出了首个对Augmented Gabidulin（AG）码的常数时间解码算法，并实现了完整的常数时间RQC-Block-MS-AG方案；

**💡 创新点**

将AG码的解码问题化简为标准Gabidulin码的重构问题，利用Loidreau算法实现二次复杂度，并对RBC库中的q-多项式操作（尤其是左除法）做了常数时间改造；

**🔧 技术方法**

采用q-多项式代数、左除算法、Loidreau重构算法、Gaussian消元以及RBC库的扩展域算术；

**📊 数据集**

在Intel Core i7‑13700 CPU上进行基准测试，使用RQC-Block-MS-AG参数集（128‑bit安全）与RQC、HQC、ML‑KEM进行对比；

**📈 对比分析**

相较原RQC实现提升速度，约四倍压缩公钥/密文尺寸；与HQC相比约慢4–5倍，但尺寸更小；与ML‑KEM相比速度慢数十倍；在错误重量上表现为常数时间；

**⚠️ 局限性**

仍比HQC慢近5倍；实现仅在128‑bit安全级别测试；对高维Gabidulin的syndrome解码仍未实现常数时间，存在进一步优化空间。

---

## 410. Connectivity at the crossroad of intuitionistic and classical polarizations in linear logic

**arXiv ID:** 2607.20303 | [PDF](https://arxiv.org/pdf/2607.20303v1)

**作者:** Raffaele Di Donna `[一作]` (Université Paris Cité), Lorenzo Tortora de Falco `[通讯]` (Università Roma Tre)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了一个扩展 Danos‑Regnier 正确性判据的几何属性，并给出了一个几何限制，使该属性在某一新的极化线性逻辑子系统中成为充分的正确性判据；同时统一了经典与直观极化，并对相关术语翻译与归约进行了形式化分析。

**💡 创新点**

创新点在于：①提出了将必要但不足的连通性属性通过几何约束提升为充分属性的方案；②构造了一个包含经典与直观极化的统一子系统；③对 Girard 的两种翻译进行了因子化并给出了对应归约的显式表征；④该正确性判据可在线性时间内判定。

**🔧 技术方法**

主要技术包括：图论与切换图（switching graph）分析、极化线性逻辑与证明结构（proof‑structure）理论、归约与可归约性的模拟、以及多态翻译与因子化方法。

**📊 数据集**

未使用任何实验数据集，本文完全是理论分析与证明。

**📈 对比分析**

由于是理论性质，本文没有实验对比；但证明指出正确性判据的检查复杂度为 O(n)，其中 n 为图的节点/弧数。

**⚠️ 局限性**

局限性包括：仅适用于所定义的特定子系统；不涵盖加法算子（additives）；对切除（cut）不完全闭合；若移除几何约束，判据失效。

---

## 411. The Blessing of Dimensionality: How Near-Orthogonality in High-Dimensional Spaces Explains Temporal Portability

**arXiv ID:** 2607.20301 | [PDF](https://arxiv.org/pdf/2607.20301v1)

**作者:** Abigail Woodring `[一作]` (NC State University), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并验证了在连续预训练10个步骤后，LoRA补丁在不重新微调的情况下仍能保持优秀的性能（即PortLLM的长期时间可迁移性）。

**💡 创新点**

首次在大规模实验中对PortLLM的长期可迁移性进行系统评估，并给出两种基于损失景观和梯度迭代的理论分析，解释近正交性是时间可迁移性的根本原因。

**🔧 技术方法**

使用LoRA参数高效微调、1-D损失景观切片分析、梯度迭代关系分析，以及对高维向量正交性的理论推导。

**📊 数据集**

在Fineweb和Cosmopedia两个连续预训练数据集上，对Mistral、Gemma和Qwen三种基础模型进行了10个时间步的实验，并进行了三次重复实验以获取统计显著性。

**📈 对比分析**

与一步或四步预训练、与无补丁基线以及与逐步微调（stepwise fine‑tuning）进行对比；结果显示PortLLM在性能上与逐步微调几乎相同，远优于无补丁，并能在10步后保持t=0性能。

**⚠️ 局限性**

局限性包括：实验仅覆盖了三种模型和两种数据集；理论假设如梯度正交性和Lipschitz连续性可能不在所有实际场景下成立；对不同任务或更大规模模型的通用性尚未验证。

---

## 412. Don't Trust the Label: License Laundering in AI Supply Chains

**arXiv ID:** 2607.20300 | [PDF](https://arxiv.org/pdf/2607.20300v1)

**作者:** James Jewitt `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 232,270 条 Hugging Face 数据集 → 模型 → GitHub 应用的供应链进行实证分析，测量了两种形式的许可证洗钱（未知洗钱和类别洗钱），并量化了许可证类别在链条中的存活率。

**💡 创新点**

首次系统量化 AI 供应链中许可证标签的传播和失真，揭示了许可证类别在跨平台链条中的高失真率以及关键约束条款被大幅削弱的现象。

**🔧 技术方法**

使用 Hugging Face 的数据集依赖元数据、GitHub 代码搜索和抽象语法树（AST）定位模型依赖、ScanCode 提取许可证字符串、基于 FSF/CC 的七类许可证分类、Sankey 与 Lorenz 曲线可视化、转移矩阵统计，并计算链条级别的存活率。

**📊 数据集**

涉及 3,120 条公开数据集、5,556 条公开模型和 24,076 条公开 GitHub 应用（共 232,270 条完整链条），所有数据均来自 Hugging Face 与 GitHub。

**📈 对比分析**

通过计算每个链条中许可证类别的转移矩阵，比较了链条不同阶段的存活率；结果显示约 62.3% 的链条至少存在一次未知许可证，约 95.1% 的宽松许可证在链条中存活率最高，所有负面约束类（如 Copyleft、Sharealike、ML License 等）在链条末端的存活率均低于 7%。

**⚠️ 局限性**

仅涵盖声明训练数据的 Hugging Face 模型（约 7.1%），忽略未公开训练数据与未检索到的 GitHub 应用；排除了 495 条无法归类的许可证字符串（占 12.2%），样本偏向流行和高影响力的资产，未覆盖长尾小型或私有链条。

---

## 413. Evolving Cache Schedules for Fast Diffusion Policy Inference

**arXiv ID:** 2607.20293 | [PDF](https://arxiv.org/pdf/2607.20293v1)

**作者:** Siying Wang `[一作]` (Xidian University), Fei Cheng `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练加速框架EVO，通过进化搜索全局缓存刷新计划来加速基于Transformer的扩散策略。

**💡 创新点**

创新点在于将缓存调度视为全局块-时间步约束子集选择问题，用闭环回放成功率作为目标进行进化搜索，并结合冗余感知初始化和基于目标的早停，以显著降低搜索成本。

**🔧 技术方法**

使用进化算法（遗传搜索）、特征相似度/不相似度度量、缓存重用机制、闭环回放评估；依赖预训练的DP‑T扩散策略。

**📊 数据集**

在多机器人操纵基准上评估，包括PH、MH、Push‑T、Block Push、Kitchen、Lift、Can、Square、Transport、Tool等数据集。

**📈 对比分析**

与Full DP‑T、EfficientVLA、BAC等基线对比，EVO在不训练的前提下实现约6.7×–8.05×速度提升，将FLOPs从15.77G降至约1.96–2.34G，同时保持或略低于基线的成功率（如PH 0.78 vs 0.80，MH 0.79 vs 0.79）。

**⚠️ 局限性**

局限在于搜索需要离线大规模回放，适用场景受限于仿真可行；对动态环境或更大规模模型的适应性未知；仅针对Transformer基扩散策略，其他结构可能不直接适用。

---

## 414. PoTRE: Test-Time Reasoning inspired by Cognitive Heterogeneity

**arXiv ID:** 2607.20268 | [PDF](https://arxiv.org/pdf/2607.20268v1)

**作者:** Anmol Kankariya `[一作]` (Google Cloud), Sercan Ö. Arık `[通讯]` (Google Cloud)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 PoTRE 多拓扑推理集成框架，解耦推理为四个专用代理（对抗细化、层次规划、谱搜索、直接链），并通过任务自适应聚合层合成最终答案

**💡 创新点**

创新点在于通过异质拓扑代理实现推理多样性，避免同质化集成的模式崩塌，并通过聚合层进行候选选择、语义合成或神经符号验证来提升复杂推理性能

**🔧 技术方法**

使用四种代理技术（对抗辩论、层次规划、并行谱搜索、标准链式推理），并配合任务自适应聚合层（候选选择、语义合成、神经符号验证）实现最终答案生成

**📊 数据集**

在 ARC‑AGI‑2、Humanity's Last Exam（HLE）和 PRBench Finance Hard 等公开基准上进行评测

**📈 对比分析**

与单代理、Self‑Consistency、官方提示等对照，PoTRE 在 HLE 取得 49.92%（单机版）、ARC‑AGI‑2 38.30%、PRBench Finance 0.3486；轻量级 Gemini‑3‑Flash‑Preview 通过 PoTRE 超越更大规模的 Gemini‑3‑Pro‑Preview，显示出显著的“架构提升”效应

**⚠️ 局限性**

存在验证瓶颈：聚合层无法完全捕获理论上 Oracle 能产生的正确答案，且在极度多样化任务（所有代理答案完全不一致）时聚合效果下降，导致最终准确率仍低于潜在上限

---

## 415. The Ethics of Autonomous AI Agents for Offensive Security

**arXiv ID:** 2607.20255 | [PDF](https://arxiv.org/pdf/2607.20255v1)

**作者:** Andreas Happe `[一作]` (TU Wien), Jasmin Wachter `[通讯]` (University of Klagenfurt)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对LLM驱动的自主AI代理在进攻性安全中的伦理问题进行系统性分析，阐述其三维不确定性（模型不确定性、影响不确定性、用户不确定性），并提出跨框架伦理评估与利益相关者分层建议。

**💡 创新点**

创新点在于：①提出并统一三维不确定性框架；②将多种伦理理论（义务论、后果主义、美德伦理、价值敏感设计、道德代理理论）同时应用于自主渗透测试；③给出分层利益相关者的具体政策与技术建议，填补了传统工具与自主代理之间的伦理空白。

**🔧 技术方法**

主要技术包括：LLM（如GPT‑4、Claude、Anthropic LLaMA），工具调用与链式思维（ReAct、chain‑of‑thought）、价值敏感设计方法、道德代理评估框架，辅以案例分析与文献综述。

**📊 数据集**

未使用传统数据集；论文引用公开报告、企业博客、学术文献、行业白皮书以及各大LLM供应商的安全与滥用报告，以构建对技术现状与伦理挑战的宏观视角。

**📈 对比分析**

论文不进行实验对比，而是通过文献综述、案例对照和多框架理论分析来比较不同伦理模型的适用性与局限；对“性能”主要以对技术与伦理框架的适配度和可操作性进行评估，未给出定量指标。

**⚠️ 局限性**

局限性包括：①缺乏可验证的实验数据，主要依赖公开信息；②对新发布LLM的能力快速演进未能实时跟踪；③跨文化、跨法域的监管环境差异难以在单篇论文中全面覆盖；④部分伦理框架在实践中的可执行性仍待验证。

---

## 416. An Approach to the Abstract Interpretation of Goal-Directed Answer Set Programming

**arXiv ID:** 2607.20252 | [PDF](https://arxiv.org/pdf/2607.20252v1)

**作者:** Daniel Jurjo-Rivas `[一作]` (Universidad Politénica de Madrid), Manuel V. Hermenegildo `[通讯]` (Universidad Politénica de Madrid)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种面向目标导向ASP（s(CASP)）的上层抽象解释框架，并实现了共享约束（Shared‑Constraints）抽象域来捕获程序执行中的变量依赖。

**💡 创新点**

创新点在于将经典的自顶向下抽象解释方法迁移到非单调ASP环境，并设计共享约束域以在约束推理时保留变量间的互约束信息，从而支持编译时优化和错误检测。

**🔧 技术方法**

技术包括：自顶向下的抽象解释器（基于PRAI框架）、共享约束抽象域、循环检测与等价约束处理、forall 语句的重写与优化、以及基于类型信息的奇数循环检测。

**📊 数据集**

实验数据集主要包括：light‑j、tap‑k（时间可变灯光与阀门流动模型）、graph‑0/graph‑1（图着色问题）、hanoi（汉诺塔）以及一些自定义的图路径例子。

**📈 对比分析**

与传统的 prev_forall、c_forall 实现对比，使用共享约束域重写后的 cls_forall 在大多数基准上获得了显著加速（如 tap‑1 约 250 倍），抽象专门化使图着色程序运行时间降低 18%。性能提升主要来自减少 forall 调用次数和提前剪枝。开销几乎可忽略，均低于 20 ms。

**⚠️ 局限性**

限制在于：对双重规则（dual rules）的处理仍采用简化假设，导致在某些复杂约束或多变量互约束场景下精度不足；抽象域的设计未覆盖所有内置谓词；目前实验范围受限于少量人工构造的基准，未在大规模真实ASP程序上验证。

---

## 417. A ProbLog program to infer individual genotypes from familial phenotypes in autosomal, X-linked, and Y-linked Mendelian disorders

**arXiv ID:** 2607.20250 | [PDF](https://arxiv.org/pdf/2607.20250v1)

**作者:** Maxime Mahout `[一作]` `[通讯]`, Maxime Mahout

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了基于ProbLog的程序mendelprob.pl，用于从家族表型和基因型推断自体、X连锁和Y连锁孟德尔遗传病的遗传概率；

**💡 创新点**

创新点在于采用概率逻辑编程实现对未知表型直接推断基因型、自动识别遗传病类型，并提供可解释的推理过程；

**🔧 技术方法**

使用ProbLog（概率逻辑编程）与Hardy‑Weinberg平衡假设、二分决策图（BDDs）等技术实现概率推理；

**📊 数据集**

以文献中的Cystic Fibrosis和Huntington’s disease族谱为案例，辅以pedprobr提供的同源数据集进行验证；

**📈 对比分析**

与R包pedprobr进行比较，结果显示两者在基因型概率预测上一致，mendelprob.pl在利用表型证据和自动判定病型方面表现更优；

**⚠️ 局限性**

局限于仅处理“直接族谱” (即单条祖先链)，假设人群处于Hardy‑Weinberg平衡，且仅适用于双等位基因的单基因遗传病，无法完整解析复杂族谱或多等位基因/多基因情况。

---

## 418. MoX: Efficient MoE Routing on Direct-Connect Topologies

**arXiv ID:** 2607.20220 | [PDF](https://arxiv.org/pdf/2607.20220v1)

**作者:** Ori Cohen `[一作]` (Technion and NVIDIA), Mark Silberstein `[通讯]` (Technion and NVIDIA)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对Mixture‑of‑Experts（MoE）通信的静态无需求感知路由方案 Mo‑X，利用 token‑aware 多播树与分层归约树结合，并通过离线采样得到的链路权重实现低带宽税与负载平衡，从而在直接连接网络上逼近理想交换机的性能。

**💡 创新点**

创新点在于①将多播树与分层归约相结合，显著减少无用转发；②采用离线采样的期望多播树权重优化，避免运行时需求预测或拓扑重构；③证明此方法在大规模 MoE 训练与推理中仍保持高效且无动态调整。

**🔧 技术方法**

主要技术包括：基于多播树构建与加权轮询的负载平衡；分数多播树打包线性规划的压缩代理权重学习；ASTRA‑sim 仿真与 Chakra token traces 评估；以及在随机 expander 与 Google Boardfly 拓扑上验证。

**📊 数据集**

使用 DeepSeek‑V3 与 Qwen‑3 225B 的 MoE 层 token‑level Chakra traces，并在 ASTRA‑sim 中以每个源采样 1,000 个 token 的方式构造模拟负载。

**📈 对比分析**

与最短路径（min‑hop）路由、基于需求的拓扑调优以及理想全连交换机进行对比；实验显示 Mo‑X 在 16/32/64 节点 expander 上训练时距理想交换机小于 1.8%，推理时 2–6%，显著优于 min‑hop（34–70% 慢）和拓扑优化（24–37% 慢）。

**⚠️ 局限性**

局限性包括：方案依赖离线预计算权重，难以实时应对极端动态负载变化；对每个 token 的多播转发增加实现复杂度；高阶随机拓扑的硬件实现（高度、链路数）仍面临布线与成本挑战。

---

## 419. Small, Free, and Effective: Orchestrating Open-Weight Small Language Models to Outperform Single LLM for Malware Analysis

**arXiv ID:** 2607.20216 | [PDF](https://arxiv.org/pdf/2607.20216v1)

**作者:** Adel ElZemity `[一作]` (University of Kent), Budi Arief `[通讯]` (University of Kent)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在单一GPU预算下，如何通过多模型协同（多代理流水线、辩论、咨询与混合架构）提升小语言模型(SLM)在恶意软件解毒报告结构化问答任务上的准确率。

**💡 创新点**

创新点包括：①提出基于证据检索与辩论的混合架构，实现了对单模型缺陷的补偿；②将小模型与网络安全预训练模型以容量平衡方式配对，验证协同可显著缩小与前沿大模型的性能差距；③对混合架构的三个阶段（证据收集、辩论、验证）进行系统剖析与消融实验。

**🔧 技术方法**

使用的技术主要是：多代理流水线（分块、增强、工具检索、证据挖掘）、对抗辩论框架（多轮互评）、层次化咨询机制（通用模型→专家模型），以及基于句子嵌入的语义相似度检索与证据匹配。

**📊 数据集**

使用的数据集是CyberSecEval Malware Analysis benchmark，包含Hybrid Analysis解毒报告与多难度、多选结构化问答，共计609道题（Easy 451道，Medium 136道，Hard 22道）。

**📈 对比分析**

通过与单模型基线（多种开源SLM、网络安全预训练模型和前沿LLM）以及内部对比实验，发现混合架构在整体准确率上达到了35.30%，超过了未基于证据的前沿LLM（34.77%）且接近或超过了基于证据的前沿LLM（38.22%）。在Easy、Medium和Hard三个难度层面均实现了显著提升。

**⚠️ 局限性**

局限性包括：整体准确率仍不达高自动化决策的要求；实验仅针对Hybrid Analysis报告的结构化问答，未覆盖二进制反汇编或开放式推理；对抗性攻击（如关键词灌注）未做评估；混合架构的计算与时延成本较高，尚需进一步优化。

---

## 420. RS-RIE-Bench: Benchmarking Reasoning-Guided Remote Sensing Image Editing

**arXiv ID:** 2607.20197 | [PDF](https://arxiv.org/pdf/2607.20197v1)

**作者:** Zihan Qin `[一作]` (Northwestern Polytechnical University), Hongwei Wang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 RS-RIE-Bench 这一遥感图像编辑评测基准，组织时序、因果、空间三类推理任务，设计三维评估指标，并采用 MLLM 自动评判。

**💡 创新点**

首次提出遥感推理驱动图像编辑基准，结合地理推理、区域控制与传感器一致性三维评估，并验证 MLLM 评判的可靠性与可扩展性。

**🔧 技术方法**

使用多模态大模型（MLLM 评判器、扩散式编辑模型）、规则化评分箱、专家人工评审、统计一致性分析等技术实现自动评测与人工校验。

**📊 数据集**

采集自 DOTA、PatternNet、NWPU-RESISC45、AdaTreeFormer、DisasterM3、xView2/xBD 等公开遥感数据集，共 486 份样本。

**📈 对比分析**

在统一生成协议下，对 8 个编辑模型（5 公开闭源，3 开源）进行评测，严格联满足准确率仅 24.28%，放宽至 4 分时 32.23%，显示现有模型仍远未达到可靠水平。

**⚠️ 局限性**

局限性包括仅涵盖 RGB 光学影像、缺乏多光谱校准、数据分发受限、MLLM 评估仍需专家验证、基准规模有限需进一步扩展。

---

## 421. Diverse-Intent Multi-Turn Fashion Image Retrieval

**arXiv ID:** 2607.20291 | [PDF](https://arxiv.org/pdf/2607.20291v1)

**作者:** Mingqiang Tang `[一作]` (Southern University of Science and Technology), Xuemeng Song `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了多意图多轮时尚图像检索任务，提出DIM-Fashion基准集

**💡 创新点**

突破传统同属性编辑限制，支持意图跨轮转移与回滚，并通过MLLM直接对齐多模态查询到视觉检索空间

**🔧 技术方法**

采用三阶段训练：背景无关视觉对齐、项目级视觉‑语言对齐、MLLM查询对齐（LoRA）

**📊 数据集**

使用13个时尚检索数据集合并生成26.7k个多轮会话

**📈 对比分析**

与现有多轮检索模型对比，FashionAM在DIM-Fashion的R@5/8分别提升约10–15%，并在MT‑FashionIQ上同样取得最佳性能

**⚠️ 局限性**

主要限制在于对纯文本检索的性能下降，以及过长历史信息可能导致性能波动

---

## 422. Pushing the Frontier of Full-Song Generation: Hierarchical Autoregressive Planning Meets Flow-Matching Rendering

**arXiv ID:** 2607.20253 | [PDF](https://arxiv.org/pdf/2607.20253v1)

**作者:** Junyu Dai `[一作]` (Alibaba Token Foundry), Haina Zhu `[通讯]` (Alibaba Token Foundry)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一个统一的全歌生成框架，支持从歌词、文本描述或参考音频生成完整歌曲、纯乐器音乐和翻唱歌曲，并通过语义感知RVQ分词器、hybrid‑LM、FullDiT与两级旋律模块实现高保真度与可控性兼顾的全长音乐合成。

**💡 创新点**

核心创新在于：1）8码book语义感知RVQ分词与分层自回归hybrid‑LM的结合，能够同时捕获全局结构与局部细节；2）FullDiT在连续VAE潜空间中进行非因果流匹配，配合EDMC提升对不完整token输入的鲁棒性；3）两级旋律表示与DPO/GRPO/OPD奖励对齐后训练，使模型在保持原调音线的同时实现高质量音色与节奏。

**🔧 技术方法**

使用技术包括：语义感知RVQ分词、hybrid‑LM（全局8B LLM+局部0.4B LLM）、FullDiT（非因果流匹配、全长自注意力）、两级旋律tokenizer、错误匹配扰动（EDMC）、四方CFG、DPO、GRPO、OPD、SDE探索等。

**📊 数据集**

数据集涵盖约4.38M首歌曲的多语言多风格预训练集，约580万首高保真音频用于FullDiT训练，约600k高质量样本用于SFT，500例多语言（中、英、日、韩、西）测试集；评测采用SongBench、SongEval、AudioBox‑Aesthetic、CMI‑Reward等自动评价体系。

**📈 对比分析**

与Suno V5.5、Suno V5、Mureka V8、Lyria 3 Pro、MiniMax Music 2.6等主流系统在SongBench、SongEval、AudioBox‑Aesthetic、CMI‑Reward四大评测维度上均取得最高或相近最高分，尤其在Melody、Arrangement、Musicality、Instrumental、Mixing等维度领先；在外部Artificial Analysis Music with Vocals排行榜中排名第2–3位，Elo 1129，表明在多语言、全长歌曲生成任务中具有竞争力。

**⚠️ 局限性**

局限性主要包括：对歌曲结构规划和多语言细粒度控制仍不够完善，翻唱音色多样性与动态变化的细化不足，模型对极长段落的稳定性与实时性尚待提升，以及在不同音乐风格（如传统民族或实验音乐）上的泛化能力尚未充分验证。

---

## 423. Exposure is Optional: Learning Unlike Coordination in Language Models

**arXiv ID:** 2607.20251 | [PDF](https://arxiv.org/pdf/2607.20251v1)

**作者:** Jiamu Luo `[一作]` (University of Washington), Shane Steinert-Threlkeld `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练 GPT‑2 模型在已过滤掉所有非同类并列（unlike coordination）的语料上，并检验其是否能够在测试集中泛化到这类结构。

**💡 创新点**

首次使用 Filtered‑Corpus Training（FiCT）证明语言模型不需要直接接触非同类并列即可学习其约束，并揭示两种内部处理机制——超类别（supercategory）和删除（deletion）。

**🔧 技术方法**

采用 GPT‑2 Large 变体进行训练，结合注意力分数、余弦相似度、k‑means 聚类和语法可接受性判定等技术进行内部和外部评估。

**📊 数据集**

使用英文 Penn Treebank 与 COCA 语料，分别构建原始（original）、仅过滤 “and” 连接的非同类并列（filtered‑and）以及完全过滤所有非同类并列（filtered‑all）的三套训练集。

**📈 对比分析**

通过困惑度、平均惊奇度、语法判断准确率、注意力差异、相似度与聚类分析等多维度指标比较三种模型；结果显示 filtered‑all 与 filtered‑and 与原始模型在非同类并列上的性能差距不显著，均能达到相近的准确率。

**⚠️ 局限性**

局限性在于过滤方法可能误删同类并列样本，导致训练数据失真；模型对细粒度类别冲突仍缺乏敏感度，且缺少针对人类心理语言学的实证验证。

---

## 424. Towards Relating Ciao Assertions and LPTP Theorems

**arXiv ID:** 2607.20249 | [PDF](https://arxiv.org/pdf/2607.20249v1)

**作者:** Marco Pérez `[一作]`, Fred Mesnard `[通讯]` (Universit\'e de La R\'eunion)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究如何把抽象解释（Abstract Interpretation）与逻辑推理（Theorem Proving）两种验证框架中的断言（assertions）转化为一阶逻辑公式，并揭示它们之间的对应关系与局限性。

**💡 创新点**

提出了一种系统化的断言翻译方案，定义了可被逻辑化的断言类，并为不可直接编码的断言设计了近似策略与辅助构造；同时分析了该方案在一致性与完备性方面的权衡。

**🔧 技术方法**

使用抽象解释技术、逻辑程序的归结语义、Clark 完成与归纳扩展、以及一阶逻辑证明系统（Gentzen 栈计算）等理论工具。

**📊 数据集**

未使用大规模真实数据集，主要以典型的 Prolog 示例（如 nat、plus、path 等）作为实验案例。

**📈 对比分析**

通过理论证明与示例推导来比较两种验证方式的可转化性和性能；结果表明：大多数断言可完整映射，性能受限于逻辑证明的手工干预，无法自动完成复杂计算性断言的验证。

**⚠️ 局限性**

局限性在于：1）并非所有断言都能在一阶逻辑中精确表达；2）复杂的计算性属性需要近似或额外构造，导致证明工作量增大；3）缺乏对大规模实际程序的自动化实验验证。

---

## 425. Vera: Identity-Faithful Human Subject-to-Video Generation

**arXiv ID:** 2607.20247 | [PDF](https://arxiv.org/pdf/2607.20247v1)

**作者:** Yulong Xu `[一作]` (Kuaishou Technology), Huaibo Huang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Vera，一种统一的人类中心化 Subject‑to‑Video（S2V）生成框架，专注于在单人和多人人场景下保持人脸身份一致性。

**💡 创新点**

创新点包括：① 通过人级跨剪辑检索构建了约 100 万对身份对齐的人类图像‑视频数据集；② 引入 Identity‑Focal Masked Supervision（IFMS）在训练时对人脸区域加权监督；③ 设计 Reference‑Aware Layer‑wise Attention（RALA）在 DiT 变压器层中保护参考身份锚点并强化层级身份读出。

**🔧 技术方法**

技术细节：基于 DiT 视频生成模型进行细调；利用面部检测与掩码实现空间加权损失；在前几层加入 Reference‑Anchor Preserving Mask，后中层添加 Identity Residual Reinforcement；整体使用扩散变压器框架进行训练与推理。

**📊 数据集**

使用自行构造的百万对身份对齐人类图像‑视频数据集，数据通过跨剪辑检索得到，每对包含一段视频与对应身份的若干参考图像；此外还在实验中使用公开的 S2V 评测基准（100 个身份、每个身份 2 条文本提示）。

**📈 对比分析**

与 VACE、Phantom、BindWeave 等主流 S2V 基线进行对比。评价指标包括 FaceSim‑Arc / FaceSim‑Cur（人脸相似度）、MotionSmooth、NaturalScore、X‑CLIP（文本‑视频对齐）。Vera 在人脸一致性（FaceSim 最高）、自然度（NaturalScore 最高）和文本符合度（X‑CLIP 领先）上均优于基线；在运动平滑度方面略低于 VACE‑1.3B，但整体性能更均衡。

**⚠️ 局限性**

局限性：仍受限于人脸检测与掩码的质量，极端遮挡或极端姿态下身份一致性可能下降；数据集构造依赖于跨剪辑检索，扩展到非人类对象时需要额外标注；对极大视频长度或高分辨率生成的效率和稳定性尚未充分验证。

---

## 426. ELSAA: Efficient Low-Rank and Sparse Attention Approximation for Training Transformers

**arXiv ID:** 2607.20214 | [PDF](https://arxiv.org/pdf/2607.20214v1)

**作者:** Mahdi Heidari `[一作]` (KAIST), Jaekyun Moon `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种ELSAA稀疏+低秩注意力近似框架，结合sortLSH稀疏分支与RACE低秩分支，并通过分母感知融合实现长上下文Transformer训练。

**💡 创新点**

创新点在于：①在注意力算子层面融合稀疏与低秩分支，而非仅对参数做稀疏/低秩分解；②引入分母感知乘子校正两分支归一化尺度差异；③提供理论阶层分析证明该混合结构可实现完整秩，提升表达能力。

**🔧 技术方法**

采用sortLSH哈希+块级精确注意力、RACE哈希桶摘要低秩注意力、分母感知门控融合，整体保持线性时间与空间复杂度，实验中实现了GPU级别的高效推理。

**📊 数据集**

在长文本分类（ArXiv 32K/64K）、图像分类（Flowers‑102、Food‑101、Fashion‑MNIST、Oxford‑IIIT Pet）、检索任务（ArXiv 64K）、短序列任务（IMDB 512、Fashion‑MNIST 784）、合成NIAH检索、因果任务（Tiny ImageNet 1024、ArXiv 32K/64K）等多样数据集上进行验证。

**📈 对比分析**

与ExactFlash、RACE、Sort_LSH、Sort_LSH_RACE等效率注意力基线在同一训练预算下对比，ELSAA在大多数长上下文任务中取得最高或相近准确率；在极长文本检索、NIAH 长度外推及因果任务中显著优于基线；在短任务保持竞争力，说明模型具备泛化性。

**⚠️ 局限性**

局限性包括：稀疏分支块尺寸与哈希参数需手工调优；在极短序列或稀疏支持不足时性能可能下降；当前验证集中于Encoder/Decoder层，尚未在大规模预训练或跨任务迁移中彻底评估；GPU融合实现仍有进一步优化空间。

---

## 427. Fully Dynamic Rooted Spanning Tree on GPU

**arXiv ID:** 2607.20211 | [PDF](https://arxiv.org/pdf/2607.20211v1)

**作者:** Abhijeet Sahu `[一作]` (Indian Institute of Technology Tirupati), G. Ramakrishna `[通讯]` (Indian Institute of Technology Tirupati)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了四种GPU并行算法，用于在边批量插入或删除时动态维护根生成树/森林，避免从头重建。

**💡 创新点**

首次在GPU上实现完全动态根生成树的高效更新，并结合SuperGraph、Hooking-Shortcutting、Eulerian Tour等子例程，提供理论深度和工作量分析。

**🔧 技术方法**

使用Hooking-Shortcutting、SuperGraph构造、Eulerian Tour路径逆转、广播、哈希表、指针跳跃、并行排序/压缩等技术。

**📊 数据集**

使用13个真实与合成图数据集，包括社交网络、Web图、道路网络、Kronecker图，规模从百万到十亿边。

**📈 对比分析**

与基准的静态并行BFS和多核pr-rst比较，平均速度提升约200倍（插入）和160倍（删除），单次最高可达900倍；吞吐量约每秒200万插入/140万删除。

**⚠️ 局限性**

限制：在极稠密图中SuperGraph依赖关键边数量导致效率下降；广播路径逆转占比高且空间复杂度大；实现仅单GPU，未讨论多GPU扩展。

---

## 428. SeededGrasp: Language-Guided Grasping in Complex Scenes with Multiple Embodiments

**arXiv ID:** 2607.20207 | [PDF](https://arxiv.org/pdf/2607.20207v1)

**作者:** Yang Xu `[一作]` (University of Toronto), Igor Gilitschenski `[通讯]` (University of Toronto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计SeededGrasp框架，将VLM预测的种子点与轻量化流匹配模型结合，实现多装置在混乱场景中的语言驱动抓取；

**💡 创新点**

创新点在于解耦VLM与抓取模型，利用种子点桥接语义与几何；首次发布2.56M抓取的多装置桌面抓取数据集；采用流匹配生成轻量化抓取模型；

**🔧 技术方法**

采用Gemini 3.1 Flash VLM做种子点预测，流匹配（Diffusion Transformer）生成抓取姿态，点云编码使用GraphConv+Fourier，结合几何约束、Heuristic、Classifier‑free guidance等技术；

**📊 数据集**

基于MultiGripperGrasp (MGG) 与自建的多装置混乱场景数据集，覆盖610场景、334对象、3个抓手（Franka Panda、Robotiq 3‑Finger、Allegro Hand），共2.56M有效抓取；

**📈 对比分析**

与DGN2.0、Geomatch、ShapeGrasp、GraspMAS等基线对比，单抓手仿真成功率提升约13%，多装置提升约35%；真实实验成功率78%；Gemini 3.1 Flash种子点优于其他VLM；

**⚠️ 局限性**

局限在于未考虑机械臂运动规划，可能生成不可执行抓取；仅使用单视图BEV，遮挡可能导致失效；数据集偏重握力抓取，对平面物体效果有限；跨硬件泛化受限；

---

## 429. Distributed Acoustic Localization Array Deployed Using a Soft Everting Vine Robot

**arXiv ID:** 2607.20392 | [PDF](https://arxiv.org/pdf/2607.20392v1)

**作者:** Sebastian Lorca Godoy `[一作]` (University of Notre Dame), Margaret McGuinness `[通讯]` (University of Notre Dame)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文设计并实验了一种软藤形机器人，通过在机器人拉伸过程中逐步部署麦克风阵列，实现对灾难现场受害者的声源定位；同时提出基于动态 Steered Response Power 与 Phase Transform（SRP‑PHAT）的声源定位框架，可同时处理远场方向估计与近场三维定位。

**💡 创新点**

创新点包括：① 将分布式声学传感与可变形软体机器人结合，利用机器人生长过程动态扩展麦克风阵列；② 设计并验证了多种麦克风放置方式（压缩体、尾部、壁面）与机器人几何形态（线性、双线性、圆形、正弦形）对定位精度的影响；③ 在实验中首次演示了从仅两个麦克风到完整五个麦克风的逐步部署过程，并展示了部署进程中定位精度的提升。

**🔧 技术方法**

使用技术主要包括：软藤形软体机器人、内部/外部麦克风阵列、Teensy 4.1 控制板、ICS43434 麦克风、GCC‑PHAT 与 SRP‑PHAT 语音源定位算法、光学运动捕捉系统（PhaseSpace）用于获取真值坐标、实验室声学测量（A‑加权噪声与信噪比调节）。

**📊 数据集**

实验数据集为自制实验室数据：使用 SoundCore 2 扬声器播放高斯白噪声，录制麦克风阵列采样（44.1 kHz），在多种配置下收集 300 帧（约 15 s）数据；通过光学捕捉获取每个麦克风和声源的三维位置，作为定位误差评估的基准。

**📈 对比分析**

通过对比不同麦克风放置、机器人几何、信噪比、声源距离等因素，本文在远场和近场均进行了误差统计。结果表明：
- 在远场下，压缩体麦克风误差可低至 0.5°，壁面麦克风 1.5°；
- 在近场下，壁面麦克风平均误差 0.17 m（σ = 0.13 m），压缩体麦克风 1.01 m；
- 机器人几何方面，圆形阵列在所有角度下误差最低（远场 1.1°，近场 0.034 m），双线性和正弦形次之；
- 演示阶段中，从两麦克风到三麦克风误差从约 2.4 m 降至 0.216 m，后续增麦克风提升有限。

**⚠️ 局限性**

局限性包括：① 仅在实验室平坦无回声环境中测试，未考虑现场混响、障碍物和自噪声；② 仅使用二维平面定位，未展开三维全景；③ 麦克风间距固定为 30 cm，长机器人可能导致数据/功耗限制；④ 目前未使用学习或自适应阵列模型，可能在形变极端时失效；⑤ 未考虑机器人自身动作产生的噪声对定位的干扰。

---

## 430. Understanding Generative AI-mediated User Engagement with Academic Library Resources

**arXiv ID:** 2607.20328 | [PDF](https://arxiv.org/pdf/2607.20328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 431. PolySim: Deterministic Polynomial Surrogates for Cross-Modal Retrieval on CiM

**arXiv ID:** 2607.20358 | [PDF](https://arxiv.org/pdf/2607.20358v1)

**作者:** Xinzhao Li `[一作]` (Villanova University), Ruiyang Qin `[通讯]` (Villanova University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出PolySim框架，将概率性高斯嵌入转换为低阶Chebyshev多项式系数，实现单步矩阵乘法的跨模态检索，兼容CiM交叉条阵列。

**💡 创新点**

通过概率分布到确定性多项式的变换与可学习的阶交叉双线性相似度，将原本需要蒙特卡洛采样的PCME转化为可在硬件上高效执行的确定性推理。

**🔧 技术方法**

采用低阶Chebyshev多项式展开、可学习阶交叉双线性相似度、InfoNCE训练目标、CrossSim硬件仿真以及后训练量化等技术。

**📊 数据集**

在六大跨模态检索基准上验证：视频-文本Clotho、MSRVTT、VATEX；图像-文本COCO、Flickr30k；音频-文本AudioCaps。

**📈 对比分析**

与基准确定性点向量（DET）和原PCME进行对比，PolySim在多数数据集上R@1显著优于DET，并在多种检索方向与PCME持平或更优，同时在CiM硬件上实现单步MVM，显著降低延迟。

**⚠️ 局限性**

需要至少8bit精度才能保持性能，4bit/2bit下表现下降；训练时额外的多项式展开与双线性相似度计算导致一次性训练开销增大。

---

## 432. Distributed Motion Planning with Safety Guarantees for Self-Reconfiguring Robotic Boats

**arXiv ID:** 2607.20352 | [PDF](https://arxiv.org/pdf/2607.20352v1)

**作者:** Alejandro Gonzalez-Garcia `[一作]` (KU Leuven), Daniela Rus `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种混合控制框架，将分布式模型预测控制（ADMM）与控制障碍函数（CBF）相结合，实现水面模块化机器人自重构与形态形成；

**💡 创新点**

创新点在于将预测规划与实时安全过滤器耦合，利用ADMM获得协同运动计划并通过CBF保证安全，即使在非凸约束下也能实时避碰且消除死锁；

**🔧 技术方法**

主要技术包括：分布式ADMM求解器（OCP_X、OCP_Z）、控制障碍函数安全滤波器（QP）、基于单积分模型的运动规划、时间可变安全距离的适配；

**📊 数据集**

实验数据集：仿真场景下规模从4到25个机器人；实物验证采用4台小型水面机器人在水槽内完成三阶段自重构；

**📈 对比分析**

与ADMM-only和CBF-only两种基线相比，所提框架在所有规模下实现了100%的成功率与碰撞无误率，且每步计算时间均低于200 ms（5 Hz）；

**⚠️ 局限性**

局限性包括：对25+机器人数的实时性难以保证，需更长预测 horizon 或异步频率；全邻域通信导致邻域规模随密度增长，导致计算复杂度上升；

---

## 433. Test-Time Training for Modality Order Consistency in Vision-Language Models

**arXiv ID:** 2607.20351 | [PDF](https://arxiv.org/pdf/2607.20351v1)

**作者:** Aditi Gupta `[一作]`, Yossi Gandelsman `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文主要阐述了 NeurIPS 2026 会议投稿的格式规范，包括标题、摘要、正文、图表、引用、页边距、字体、行距等细节。

**💡 创新点**

创新点在于对以往 LaTeX 样式文件的统一和更新，明确了单页限制、可选参数（如 camera-ready、preprint、no-bib）以及严格的排版要求，确保提交的论文在排版上的一致性与兼容性。

**🔧 技术方法**

采用 LaTeX 及其宏包（如 natbib、graphicx、booktabs）来实现排版；通过选项控制字体、页码、注释、参考文献的格式；利用包的默认设置保证 PDF 兼容性。

**📊 数据集**

本文并未使用任何实验数据集，主要是针对论文排版的技术说明。

**📈 对比分析**

没有涉及实验方法或性能比较；主要是与 NeurIPS 2025 版式的对比，强调新版样式文件的改进点和使用建议。

**⚠️ 局限性**

局限性包括：仅适用于 NeurIPS 2026 会议，无法直接迁移到其他会议或期刊；若作者自行修改样式文件，可能导致拒稿；并且本文仅覆盖排版细节，未提供内容撰写或研究方法的指导。

---

## 434. Generating Fibonacci Words via the Prefix--Suffix Duplication Operation

**arXiv ID:** 2607.20405 | [PDF](https://arxiv.org/pdf/2607.20405v1)

**作者:** Diego Cabrera Salamanca `[一作]` (Universidad Nacional de Colombia), Taylor J. Smith `[通讯]` (St. Francis Xavier University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

证明了相同奇偶性的斐波那契词可通过有限前缀复制操作（k≥3）相互生成，并给出了线性时间构造算法；

**💡 创新点**

首次将Dumitran猜想完全证实并优化为只需前缀复制，k最小值3且证明最优；

**🔧 技术方法**

利用组合语言理论、斐波那契递推展开、前缀复制及其有界变体，构造递归推导；

**📊 数据集**

无实验数据，纯理论证明；

**📈 对比分析**

通过复杂度分析表明算法时间与目标词长度成线性关系，空间亦线性；

**⚠️ 局限性**

仅适用于斐波那契词，且无法保证得到最短复制序列；算法不支持更小的k或后缀复制；

---

## 435. LKValues: Aligning Large Language Models with Sri Lankan Societal Values

**arXiv ID:** 2607.20410 | [PDF](https://arxiv.org/pdf/2607.20410v1)

**作者:** Nethmi Muthugala `[一作]` (Tianjin University), Deyi Xiong `[通讯]` (Tianjin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过三语调查提取40个斯里兰卡社会价值，构建双语指令数据集150k及1000条评估基准，并在其上对LLM进行微调与评测。

**💡 创新点**

首创以调查为基础的本土价值识别与双语数据链，提供低资源国家的可复制价值对齐范例。

**🔧 技术方法**

采用LLM辅助的价值引导（GPT-5-mini）、全参数SFT/LoRA微调、多语言指令学习与对齐评估技术。

**📊 数据集**

使用斯里兰卡新闻（2009-2023）经价值标注、场景化生成的150k双语指令集，以及基于MMLU+场景的1000条双语评估集。

**📈 对比分析**

与多款开源/专有大模型在双语评估基准上对比，发现即使大型模型也难以完全对齐，微调后Qwen系列可将准确率提升至约86%，但跨模型差异显著。

**⚠️ 局限性**

调查样本有限、仅覆盖50%价值、未覆盖泰米尔语、仅正面阈值解释、微调训练周期不足导致模型差异。

---

## 436. Pure-DP Statistical Query Release at the Conjectured Square-Root Rate

**arXiv ID:** 2607.20418 | [PDF](https://arxiv.org/pdf/2607.20418v1)

**作者:** Jack Fitzsimons `[一作]` `[通讯]`, Jack Fitzsimons

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文构造了一种纯差分隐私（ε‑DP）机制，用于在任何数据库大小 n、数据域大小 T 以及查询集大小 k 下发布数据库上所有统计查询的近似值，并证明其期望最大坐标误差为 O( min{1, √(log(2T)·log(2k)/n) } )，实现了 Nikolov 与 Ullman 提出的平方根误差上界。

**💡 创新点**

创新点在于：
1) 设计了一个“概率包络”（transcript envelope）机制，利用每个转录（transcript）的最大折扣似然值，将选择‑仅多重权重（PMW）转录的局部隐私信息提升到全局纯 DP；
2) 通过对哈密顿球（Hamming ball）中的似然值进行 Maurey‑domination 与混合（mixture）两种估计，精确控制距离近的与远的数据库对误差的贡献；
3) 使用块化（block‑summation）策略避免了对每个 Hamming 半径逐一求和带来的 1/ε 失配；
4) 完整地给出了信息理论证明并用 Lean 4 形式化验证了所有关键不等式。

**🔧 技术方法**

主要技术手段包括：
- 选择‑仅 PMW（multiplicative weights）转录与指数机制；
- 似然比（likelihood ratio）与 Rényi 散度的高阶矩控制；
- Maurey 经验方法在似然空间的应用；
- 阻尼（damping）与块化（blocking）技术以平衡包络增长与隐私代价；
- 形式化验证与 Lean 4 机器检验。

**📊 数据集**

该工作为理论性研究，未使用实际数据集；所有结果均在任意离散数据库上成立，适用于任何 T、k、n 的取值。

**📈 对比分析**

与以往纯 DP 的立方根误差上界相比，本文提供了平方根上界，且误差随 log(T)·log(k)/n 的平方根比例增长，匹配已知的下界；在约束空间高维时能达到最优。虽然未给出可多项式时间实现，但理论上可获得与下界同阶的性能。

**⚠️ 局限性**

局限性：
- 机制信息理论，转录空间指数级大小，缺乏高效实现；
- 仅给出期望误差上界，未给出高概率（高置信度）保证；
- 证明仅覆盖纯 DP 情况，近似 DP 下已有更强结果；
- 在特殊极端参数（如单元素域、单查询）下并不匹配最优下界；
- 需对参数做严格整数化才能实现，实际部署时需进一步细化。

---

## 437. SoftReason: A Fully Differentiable Neuro-Soft-Symbolic Deductive Reasoning Architecture over High-Dimensional Perceptual Data

**arXiv ID:** 2607.20402 | [PDF](https://arxiv.org/pdf/2607.20402v1)

**作者:** Wael AbdAlmageed `[一作]` `[通讯]` (Clemson University), Wael AbdAlmageed (Clemson University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种完全可微的神经软符号推理架构，能够在高维感知输入中直接进行演绎推理，并与知识图谱中的谓词和事实进行融合。

**💡 创新点**

创新点在于将推理状态表示为软解释张量，并学习可微的即时后果算子（使用谓词定义嵌入与潜在组合通道），实现软体谓词混合、见证聚合以及基于概率 OR 的闭包更新，克服了传统神经-符号流水线的梯度鸿沟。

**🔧 技术方法**

核心技术包括感知编码器与实体归一化、关系注意力编码、谓词条件双线性评分、KG 证据注入、可微 Horn 规则组合、查询条件闭包、以及多步闭包的概率 OR 更新；整体实现完全端到端可微。

**📊 数据集**

在 Knowledge-aware Visual Question Answering（KVQA）数据集上进行评估，该数据集要求通过图像识别实体并利用多跳知识图谱关系回答问题。

**📈 对比分析**

与现有基于知识图谱的 VQA 方法（如 Hypergraph Transformer、实体增强知识注入等）以及传统 VQA 模型相比，本文方法在 Hit@1 和 Recall@5 指标上均取得显著提升，特别是在 2‑hop、3‑hop 等多跳问题上表现更佳。

**⚠️ 局限性**

局限性包括：对知识图谱子图的依赖（无法处理无 KG 或极大 KG 的场景）、对高阶谓词支持有限、推理深度受固定迭代次数限制，以及在计算复杂度与可扩展性方面仍需进一步优化。

---

## 438. Towards Miniature Humanoid Tele-Loco-Manipulation Using Virtual Reality and Reinforcement Learning

**arXiv ID:** 2607.20399 | [PDF](https://arxiv.org/pdf/2607.20399v1)

**作者:** Nicolas Kosanovic `[一作]` (University of Louisville), Jean Chagas Vaz `[通讯]` (University of Louisville)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一个针对OP3迷你双足人形机器人的全身可兼容VR远程操控与强化学习行走控制的柔性控制系统。

**💡 创新点**

将基于VR的人形上肢操控与RL驱动的下肢行走相结合，并在DYNAMIXEL XM430上实现基于数据驱动的力矩控制与PD阻抗控制，形成了全身可独立操控的框架。

**🔧 技术方法**

采用Unity+ROS+SteamVR进行VR界面与姿态映射，利用DYNAMIXEL XM430的电流力矩模型实现开环力矩控制，使用Isaac Sim/Isaac Lab进行RL训练与域随机化。

**📊 数据集**

通过自建实验室环境的传感器数据与10次块移动实验收集的操作轨迹与重力投影数据作为训练与验证数据；未使用公开数据集。

**📈 对比分析**

对照手部追踪延迟、RMSE与MAE，得到约220 ms延迟与10–15 %误差；RL行走在随机速度扰动下保持姿态；块移动实验平均每10 min仅完成2个块，速度约0.35 m/s。

**⚠️ 局限性**

受限于高延迟、低抓取力、脚步抬起不足、缆线供电与粗糙的仿真-实物差距，导致操作效率低且难以完成更复杂的搬运任务。

---

## 439. The ICSE 2026 Shadow PC: Training the Next Generation of Reviewers Through Deliberate Practice

**arXiv ID:** 2607.20396 | [PDF](https://arxiv.org/pdf/2607.20396v1)

**作者:** Christian Kästner `[一作]` (Carnegie Mellon University), Francisco Gomes de Oliveira Neto `[通讯]` (Chalmers University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文重新设计了ICSE 2026的Shadow PC计划，采用多阶段刻意练习模式，规模化训练早期研究人员成为高质量评审者，并为其提供从评审到讨论、领衔的路径。

**💡 创新点**

创新点在于：①对所有合格申请者开放而非择优，②将评审训练分为校准、同伴评审、讨论等六阶段，③在主PC之外独立设立冲突隔离，④提出Shadow PC区域主席机制以实现可持续扩展和领导力培养。

**🔧 技术方法**

技术手段包括：HotCRP账户与冲突管理、基于Web表单的校准与复审提交、Zoom同步会议、线上审稿清单与评审反馈工具，以及对比主PC评审的公开数据展示。

**📊 数据集**

使用的数据集为183名初始参与者、102名完成者、117篇参与评审的ICSE提交论文（其中197篇作者自愿加入）以及对应的主PC评审与作者反馈。

**📈 对比分析**

与主PC决策对齐的比较显示：Shadow PC 68% 与ICSE拒绝一致、接受率较低（仅2/16一致），作者满意度为67%，评审质量总体得到提升，97%的参与者推荐继续参与。

**⚠️ 局限性**

局限性包括：讨论阶段参与度下降、讨论质量参差不齐、同步会议时间不足、缺乏中层监督与即时反馈、以及后期闭幕和反馈互动不足。

---

## 440. FMRP-LEAN: A HIPAA-Compliant AI-Augmented LIMS Architecture for End-to-End Clinical Assay Workflow Optimization

**arXiv ID:** 2607.20382 | [PDF](https://arxiv.org/pdf/2607.20382v1)

**作者:** Eva McCord `[一作]` (Cincinnati Children’s Hospital Medical Center), Zag ElSayed `[通讯]` (University of Cincinnati)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并部署了 HIPAA 合规、AI 增强的实验室信息管理系统（FMRP-LEAN），用于优化 Fragile X 病理标志物 Luminex 测定的端到端工作流。

**💡 创新点**

创新点包括：① 采用有限状态机模型实现显式工作流转换与停留时间可观测；② 混合 edge‑internal 架构实现 PHI 隔离与仅回环服务；③ 对 AI 进行治理约束，仅使用聚合数据生成文本，并提供确定性回退；④ 自动化 QC 预筛选与双人确认机制；⑤ 与 REDCap 的双向同步。

**🔧 技术方法**

技术栈涵盖 Supabase/PostgreSQL、Kong API Gateway、REDCap API、UUIDv7 + QR 追踪、统计 QC 规则、LLM 生成摘要、Docker+加密隧道、HTTPS/Cloudflare、监控与日志。

**📊 数据集**

使用了现场 Fragile X 实验室的 DBS 样本及 Luminex 测量数据，并结合 REDCap 中的参与者访问信息；数据未公开。

**📈 对比分析**

通过与传统电子表格工作流对比，评估四个维度（工作流可观测性、QC 延迟、后备一致性、跨角色透明度）。结果显示工作流可视化提升、QC 延迟降低、后备时间方差下降、跨角色协同显著改进。

**⚠️ 局限性**

局限性包括：仅在单一机构部署，缺乏跨机构验证；PHI 隔离效果依赖机构治理实践；未评估不同测定平台的适配性；长期可维护性与 AI 可靠性仍待进一步研究。

---

## 441. Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations

**arXiv ID:** 2607.20379 | [PDF](https://arxiv.org/pdf/2607.20379v1)

**作者:** Hiskias Dingeto `[一作]` `[通讯]` (StackOne Technologies), Hiskias Dingeto (StackOne Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文审计了自然语言自动编码器的解释可信度，发现重建测试易被整体语义或私有编码误导，并提出RECAP方法在目标模型中训练辅助线性头以保持指定内容可解码，从而恢复逐句可信度。

**💡 创新点**

创新点在于揭示重建测试的结构漏洞，提出基于目标模型深度监督的RECAP框架，且通过自下而上的“grounded‑vs‑true”与“evaluator swap”两种审计协议检测私有代码，并验证RECAP在实际模型上的可行性。

**🔧 技术方法**

使用的技术包括自然语言自动编码器、对抗式微调、线性辅助预测器、线性探测器、重构器与正交投影、以及对比式“grounded‑vs‑true”交叉审计。

**📊 数据集**

使用的数据集包括公开的 Qwen‑2.5‑7B NLA 对话文本、合成模板化语料（两种域），以及 Pythia‑160M 的 Pile 文本进行继续预训练。

**📈 对比分析**

通过对比重建分数与真实标注的 grounding 比例、探测器 AUC 等指标，RECAP 在合成沙盒中实现 100% 可解码率（成本<0.01 nat），在 Pythia‑160M 上可解码率达到 0.95–1.00，且对抗性游戏中探测器 AUC 仍保持 >0.95，明显优于基线控制。

**⚠️ 局限性**

局限性包括仅验证封闭词表模板化声明，RECAP 需要与目标模型共同训练且不能后期应用，解码性不等同于可言语化，且仅保证信息存储而非模型使用。

---

## 442. Split Radiance Cascades: Real-Time Global Illumination via Sparse Radiance Probes

**arXiv ID:** 2607.20384 | [PDF](https://arxiv.org/pdf/2607.20384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 443. Distributed Colouring with 4/3 chi Colours for Hyperbolic Random Graphs

**arXiv ID:** 2607.20360 | [PDF](https://arxiv.org/pdf/2607.20360v1)

**作者:** Kostas Lakis `[一作]` (ETH Zurich), Adeline Pittet `[通讯]` (ETH Zurich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在超几何随机图（HRG）上分布式顶点着色问题，提出了 Sequential Radial Colouring（SRC）与 Parallel Radial Colouring（PRC）两种 CONGEST 模型下的随机着色算法；

**💡 创新点**

通过利用度分布与半径位置的严格关联，构建基于度的伪径向分层，逐层进行着色，显著降低所需颜色数（接近染色数 χ）并实现对数对数时间复杂度；

**🔧 技术方法**

核心技术包括：
- 度估计逆映射求近似半径；
- 递归宽度控制的伪径向分层；
- RCTDEG 随机着色协议的度优先冲突解决；
- 期望度与实际度的泊松集中与 Chernoff 估计；
- 采用一次性失败概率控制的随机化压缩与自界函数下的 Chernoff/自界集中；
- 对 clique 的 RCTDEG 性能的上界/下界分析；

**📊 数据集**

实验与分析基于理论的 HRG 采样（Poisson 版本），并假设 n、α、C 已知；未使用具体真实网络数据集；

**📈 对比分析**

与 Maus & Ruff 的常数轮算法相比，SRC 在几乎最优颜色（≤ 43χ）下实现 O((loglog n)^2) 轮，而 PRC 在 O(loglog n) 轮内使用 O(χ loglog n) 颜色；进一步在染色数的略微放宽（χ^{1+ε}）时实现常数轮；相比传统 Δ+1 颜色算法，显著降低颜色数并在 CONGEST 模型下保持低通信量；

**⚠️ 局限性**

局限性包括：
- 仍需要知道 n 与 α 的参数以估计半径；
- 伪径向分层复杂度高，可能不易实现；
- 对极端 α 值（接近 1/2 或 1）时的常数因子可能失效；
- 结果主要针对 HRG，未直接证明可推广至更一般的 GIRG 或其他幂律图；
- 对低度顶点的后续冲突仍需多轮迭代，可能导致实际通信开销高于理论下限。

---

## 444. PercepCap: Video Captioner with Structured Spatio-Temporal Perception

**arXiv ID:** 2607.20389 | [PDF](https://arxiv.org/pdf/2607.20389v1)

**作者:** Yifan Xu `[一作]` (State Key Laboratory for Novel Software Technology, Nanjing Univerisity), Limin Wang `[通讯]` (State Key Laboratory for Novel Software Technology, Nanjing Univerisity)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PercepCap 框架，将视频 captioning 拆分为感知（生成 spatio‑temporal perception trace）与描述两步，并通过两阶段训练实现。

**💡 创新点**

创新点在于显式中间感知证据、两阶段 PD‑SFT + PG‑RL 训练策略以及 Caption‑Anchored Perception Data Construction 方法。

**🔧 技术方法**

使用多模态大语言模型 Qwen3‑VL‑8B‑Instruct、监督微调、GRPO 强化学习、结构化感知轨迹生成与多层奖励设计。

**📊 数据集**

采用 DREAM‑1K、CaReBench、VidCapBench‑AE、ShortVidBench、MotionBench 等公开基准，并在约 57K 动作视频上构造 caption‑aligned perception 数据。

**📈 对比分析**

通过官方指标（F1/Recall/Precision 等）与 GPT‑4、Gemini 评测，对比 caption‑only baseline，PercepCap 在各基准均实现显著提升，尤其在细粒度覆盖与准确性上领先。

**⚠️ 局限性**

限制包括依赖自动生成的感知数据可能带来误标、推理时长和输出长度增加、奖励计算复杂、固定轨迹模式不够灵活。

---

## 445. Generative AI floods and dilutes the market for books

**arXiv ID:** 2607.20349 | [PDF](https://arxiv.org/pdf/2607.20349v1)

**作者:** Tuhin Chakrabarty `[一作]` (Stony Brook University), Paramveer Dhillon `[通讯]` (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2023‑2026 年间亚马逊自助出版的 14,419 本类别小说进行全文 AI 检测，量化 AI 文本占比，并评估其对销量、收入、排名和作者产出的市场稀释影响。

**💡 创新点**

创新点在于首次使用全文 AI 检测而非预览片段，对书籍中 AI 文本比例进行细分，并将该比例与稀有表达覆盖度、市场规模增长及收入稀释等指标关联，揭示 AI 通过规模而非质量重塑创意市场的机制。

**🔧 技术方法**

技术手段包括 Pangram AI 检测器对全文窗口的标签，销售与排名面板的每日跟踪，稀有表达覆盖度的计算，事件时间作者产出跟踪，以及回归与聚类分析来衡量市场效应。

**📊 数据集**

使用的数据集包括：① 14,419 本自助出版科幻/浪漫等类别小说的完整文本（通过作者、图书馆或购买获得），② 约 500,000 个 ASIN 的每日销售、价格、排名数据面板，以及 该样本对应的作者笔名与 Kindle Unlimited 参与信息。

**📈 对比分析**

通过按 AI 文本比例（0%、0‑25%、>25%）划分组，比较标题、销量、收入、Top‑25 排名占比、季度与启动窗口的收入/销量倍数，并在固定 90 天启动窗口和季度窗口上进行统计对比；结果显示 AI 文本书籍的市场规模增速显著快于收入增速，导致整体收入/书下降，而成功的 AI 文本书籍更频繁使用稀有表达。

**⚠️ 局限性**

局限性包括：① 仅涵盖在 Amazon 销售/排名中出现的书籍，未捕获未列入面板的作品；② 对 Kindle Unlimited 读数缺乏细分，仅记录参与状态；③ 作者多名笔名导致产出集中度低估；④ 未能衡量 AI 未披露对读者偏好的影响；⑤ 研究聚焦于自助出版类别，结果对其他创意领域的推广存在不确定性。

---

## 446. Notes to Self: Can LLMs Benefit from Experiential Abstractions?

**arXiv ID:** 2607.20372 | [PDF](https://arxiv.org/pdf/2607.20372v1)

**作者:** Chang Liu `[一作]` (Carnegie Mellon University), Artur Dubrawski `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过从目标LLM的训练轨迹中提炼可复用的自然语言经验抽象，并在推理时或强化学习后训练中检索注入，提升LLM在数学与逻辑推理任务上的性能。

**💡 创新点**

创新点在于：①让小型LLM自身（无教师监督）即可从自身训练经验中提取抽象；②将抽象库同时用于推理时检索和GRPO强化学习后训练，实现自我提升与迁移；③对抽象的去重与检索策略进行系统化设计。

**🔧 技术方法**

使用的技术包括：抽象提取（教师或自我提示）、句子嵌入检索（聚类去重）、检索式提示增强、GRPO（Gaussian Policy Optimization）强化学习后训练、以及自定义抽象注入模板。

**📊 数据集**

主要使用的数据集为MATH（训练集7500道题，测试集待评估）以及其他逻辑推理数据集，教师模型为更强的开源LLM（如更大版本），抽象库构建基于这些数据。

**📈 对比分析**

比较方法：对比基线、Inference_abs、GRPO、GRPO_train、GRPO_train+test 等配置；实验显示抽象检索可提升 PASS@1 与 PASS@8 3–6个百分点，GRPO_augmented 更为显著；自提抽象与教师提取的效果基本相当。

**⚠️ 局限性**

limitations：结果受教师模型、句子编码器、检索阈值、去重阈值等固定设计选择影响；未做敏感性分析；RL后训练仅在单个 epoch、八轮 roll‑out 下完成，算力有限；抽象迁移性受限于目标模型和任务域。

---

## 447. Variance-reduced Domain Adaptation using Paired Sampling

**arXiv ID:** 2607.20367 | [PDF](https://arxiv.org/pdf/2607.20367v1)

**作者:** Andrea Napoli `[一作]` `[通讯]` (University of Southampton), Andrea Napoli (University of Southampton)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种针对无监督域自适应（UDA）的新随机方差减小（SVR）方法——Paired Sampling for Domain Adaptation (PSDA)，通过在源域和目标域之间以及内部形成四元组样本配对来降低梯度噪声，进而提升训练稳定性和目标域精度。

**💡 创新点**

创新点在于将配对采样与线性分配问题相结合，利用两阶段匹配（源-目标配对与配对内部配对）构造高度相关的样本组，最小化预期梯度误差；该方法无需修改模型或损失，兼容传统 MMD、CORAL 等非加性损失，并能显著降低估计方差。

**🔧 技术方法**

核心技术包括：非加性损失的期望梯度误差分析、线性分配算法（可使用匈牙利算法或其他高效求解器）、条件均值/协方差估计、GPU 向量化实现，以及在训练过程中周期性更新配对以适应特征分布漂移。

**📊 数据集**

在三个公开域迁移基准上评估：Spurious 4‑breed 图像分类（6 个域）、Office‑Home 图像分类（4 个域）和 Humpbacks 语音事件检测（4 个域）。

**📈 对比分析**

与多种采样技术（k‑means++、DPP、Anticlustering、VaRDASS、ORDERED）以及多种 UDA 方法（DANN、CDAN、SDAT、ELS、ARM、MCC）对比，PSDA 在目标域准确率上通常位列第一或第二，方差减小幅度最大，且训练速度最快（仅略高于基线随机采样）。

**⚠️ 局限性**

局限性包括：需周期性重新计算配对，且配对成本与样本总量相关；在大批量大小（k）时，四元组配对的方差优势可能下降，导致 ORDERED 等方法在高 k 下表现更好；此外，PSDA 目前仅针对 MMD 与 CORAL，扩展到其他非加性损失仍需进一步研究。

---

## 448. IteraSim RAG: A Multi-Stage Retrieval-Augmented Agentic Back-End for OpenFOAM-Based Computational Fluid Dynamics

**arXiv ID:** 2607.20346 | [PDF](https://arxiv.org/pdf/2607.20346v1)

**作者:** Pratyush Kumar `[一作]` `[通讯]` (ETH Zürich), Pratyush Kumar (ETH Zürich)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 IteraSim RAG，一套基于检索增强的多代理框架，用于自动生成、调试 OpenFOAM CFD 案例。

**💡 创新点**

创新点包括：三阶段检索（查询扩展、RRF 级联、MMR 重新排序）、双模式检索路由（工作流 vs 专家模式）以及将架构师‑输入写入‑审阅者三代理分工与静态典型知识层相结合，显著降低非专家使用门槛。

**🔧 技术方法**

使用了 LLM（Chat‑GPT‑style）进行查询扩展与代码生成、HNSW 向量索引、Reciprocal Rank Fusion、Maximal Marginal Relevance、关键词驱动的意图分类器以及 OpenFOAM 运行时日志解析。

**📊 数据集**

数据集包含 5 种来源：OpenFOAM 官方教程、FAQ、专业 Markdown 说明、FoamGPT 训练数据、Sandia WEC 扩展文档，共 1000+ 文档；并使用 28 条多层级基准测试（零/少/参数/湍流模型切换）评估系统。

**📈 对比分析**

通过检索层覆盖率评估，平均 77.9%（中位 79.1%），参数修改类最高 90.7%，少量示例 82.0%，湍流切换类 74.1%；系统在交互式三秒以内完成检索；与前置工作相比，检索准确率提升约 10‑15% 以上，且提供了完整可复现的基准脚本。

**⚠️ 局限性**

主要局限：对稀缺 Solver（如 Lagrangian 粒子、Boussinesq 等）的文档覆盖不足导致检索失效；检索评分区分大小写、对同义词不敏感；基准规模仅 28 条，未覆盖多物理场与完整可执行性评估；完整执行层尚待进一步实验。

---

## 449. Interval and fuzzy physics-augmented neural networks (iPANN and fPANN) for uncertainty quantification and propagation in constitutive modeling

**arXiv ID:** 2607.20339 | [PDF](https://arxiv.org/pdf/2607.20339v1)

**作者:** Somesh Pratap Singh `[一作]` (Cornell University), Nikolaos Bouklas `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于物理增强神经网络的区间与模糊模型（iPANNs 与 fPANNs），用于在噪声、稀疏数据下实现弹性材料的确定性区间预测和可调节的模糊隶属度预测，并将学习到的能量分支直接嵌入有限元求解器中进行不确定性传播。

**💡 创新点**

创新点在于：
1) 通过区间形式学习自由能量的上下界并用自动微分得到应力区间，保证观测应力被严格包容；
2) 将区间分支嵌入模糊集的α‑cut插值，实现从最保守到最精确的可调不确定性表征；
3) 在网络结构中严格施加物理先验（客观性、一致性、多凸性）并利用 smoothed L₀ 正则化得到可解释的稀疏能量表达式；
4) 引入两阶段迁移学习（先学习均值后微调上下界），提升优化稳定性与收敛速度。

**🔧 技术方法**

技术手段包括：
- 输入凸神经网络（ICNN）保证能量对本构输入的凸性；
- 自动微分用于从能量得到第二皮科-科氏应力；
- 软硬阈值门控的 smoothed L₀ 正则化实现可微稀疏化；
- 区间算术与α‑cut模糊插值构造区间/模糊模型；
- 迁移学习两阶段训练策略；
- 在 FEniCS 中直接使用学习到的闭式能量函数进行 FEM 传播。

**📊 数据集**

使用合成的 Gent‑Gent 弹性模型数据。训练集为 500 个在 9 维变形梯度上均匀采样并映射到 3 维不变量的点集，加入乘性异方差噪声；测试集为沿单一轴向拉伸路径的 1000 个均匀拉伸步。通过四种噪声扰动（E1–E4）验证模型鲁棒性。

**📈 对比分析**

与传统单一均值预测（无区间）相比，iPANNs 的上下界能够覆盖所有噪声观测，测试集表现与训练集相近；fPANNs 的 α‑cut 包络随着 α 的降低呈现 sigmoid‑形增长，覆盖率可调；在 FEM 传播实验中，三种能量分支产生的位移与应力分布保持物理合理性且上下界呈现预期的收敛/发散趋势，表明区间/模糊模型能够在后处理阶段提供可信的不确定性界限。

**⚠️ 局限性**

局限性包括：
- 仅在合成 isotropic 弹性数据上验证，缺乏真实实验数据验证；
- 目前只处理弹性（无耗散）本构，未扩展到塑性、粘弹性或损伤等路径相关行为；
- 模糊区间不提供概率分布，仅给定确定性界限，无法直接量化置信水平；
- 需要先验的区间/噪声模型（乘性异方差），若实际噪声偏离假设可能影响界限的有效性；
- 训练时需手工采样并构造不变量点集，数据生成与预处理过程较为繁琐。

---

## 450. PyroDash: Cost-Efficient Token-Level Small-Large Language Model Collaborative Inference

**arXiv ID:** 2607.20327 | [PDF](https://arxiv.org/pdf/2607.20327v1)

**作者:** Niqi Lyu `[一作]` (Pyromind Dynamics Inc), Yicheng Ding `[通讯]` (Pyromind Dynamics Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 PyroDash 框架，允许小型语言模型（SLM）在自回归生成过程中根据自身生成轨迹决定是否向大模型（LLM）发出一次性手风琴，完成推理。

**💡 创新点**

创新点在于将协同决策嵌入 SLM 的生成流程，无需额外路由器或多次模型切换；使用单次冻结 LLM 调用，并通过 Group Relative Policy Optimization（GRPO）与 λ 超参数实现准确率与成本的显式权衡。

**🔧 技术方法**

采用控制符嵌入学习、offloading‑oriented 监督微调（SFT）以及 GRPO 进行成本意识的强化学习；在 vLLM 上部署并实现单次 handoff 的协同推理。

**📊 数据集**

训练使用自制 EasyHard‑24k 数据集（24 061 条例子，包含可供 SLM 决策的标签）；评估采用五个数学推理基准：GSM8K、Minerva、OlympiadBench、AIME‑25、AIME‑24。

**📈 对比分析**

与单一 SLM、单一 LLM、SFT、RouteLLM、GlimpRouter 等方法对比；在 λ=0.05 时平均准确率 64.04%（比 LLM baseline 提升 6.36pp），在 λ=0.6 时平均准确率 54.55% 仅 1.90% LLM token ratio，成本从 49.36$ 降至 1.78$（下降 96%），显示在不同准确率–成本点能优于基线。

**⚠️ 局限性**

局限性：仅在数学推理任务上验证；未探讨多轮交互或多次 handoff 的效果；对 offloading 触发点的合理性缺乏可解释性；成本估算基于 token 价格而非真实账单；未评估不同 SLM‑LLM 配置或其他任务领域的泛化能力。

---

## 451. Mapped ADMM: A Robust Algorithm for 1-Bit mMIMO Detection

**arXiv ID:** 2607.20414 | [PDF](https://arxiv.org/pdf/2607.20414v1)

**作者:** Mohammad Amin Keshmiri `[一作]` (University of Alberta), Masoud Ardakani `[通讯]` (University of Alberta)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对一比特mMIMO系统的新检测器，通过将检测问题转化为分散式二分类器并使用Consensus ADMM与映射ADMM实现快速、鲁棒的符号检测。

**💡 创新点**

创新点在于将一比特检测视为多分类器分散化问题，利用Consensus ADMM实现多候选估计并通过投票+常数点映射进一步加速收敛。

**🔧 技术方法**

主要技术包括支持向量机（SVM）二分类映射、Consensus ADMM (CADMM)、映射ADMM (MADMM)、投票机制、硬判决投影。

**📊 数据集**

使用仿真数据：10^5 次 Monte Carlo 试验，QPSK 调制，32×4、64×8、128×16 的 mMIMO 系统，采用理想 Rayleigh 信道模型。

**📈 对比分析**

与传统 ZF、近似 ML（NML）、SVM、CADMM 等方法比较，MADMM 在 SER 上优于其他方法，收敛速度显著提升（CADMM 600+ 迭代→MADMM 50+ 迭代）。

**⚠️ 局限性**

局限性包括仅验证 QPSK 调制，对更高阶调制和非理想信道的性能尚未评估；组大小选择仍需经验性调参；算法在极高 SNR 下的鲁棒性尚待进一步验证。

---

## 452. Persian Pixel: A large-scale synthetic OCR dataset for Persian language

**arXiv ID:** 2607.20385 | [PDF](https://arxiv.org/pdf/2607.20385v1)

**作者:** Pouria Mahdi `[一作]`, Haq Nawaz Malik `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了一个规模达34.3万张图像-文本对的 Persian Pixel 合成 OCR 数据集，覆盖句子、段落和页面三种粒度。

**💡 创新点**

创新点在于使用形状感知的 SynthOCR-Gen 渲染，结合七款字体（包含 Naskh 与 Nastaliq）以及 25 种随机降质模型，真实模拟波斯书写系统的连字、点、重音、双向排版等复杂特性。

**🔧 技术方法**

技术包括 SynthOCR-Gen 的 OpenType 形状堆栈渲染、几何光度噪声与纸张退化等多维随机增强，以及可直接用于 TrOCR、Donut 等 Transformer OCR 架构的训练 pipeline。

**📊 数据集**

数据来源于公开许可的七百万词波斯语语料库（维基百科、文学、新闻、政府、社交媒体和历史文档），并在此基础上生成合成图像。

**📈 对比分析**

通过在合成数据上微调 TrOCR、Donut 等预训练模型，并在真实扫描/照片样本上评估，显著提升识别准确率，尤其在包含 Nastaliq、点模糊和混排的文档上相较于单字体或无降质训练的模型提升约10–20%。

**⚠️ 局限性**

限制在于仅覆盖印刷文本；字体样式受限于七款可公开许可字体；未模拟历史文档中的复杂边注、对齐与非线性阅读顺序，并且需要大量 GPU 资源训练大型 Transformer。

---

## 453. ATSplat: Compact Feed-forward 3D Gaussian Splatting with Adaptive Token Expansion

**arXiv ID:** 2607.20417 | [PDF](https://arxiv.org/pdf/2607.20417v1)

**作者:** Cho In `[一作]` (Yonsei University), Seon Joo Kim `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ATSplat，一种在单前向推理中利用自适应3D锚点恢复3D高斯分布的创新框架。

**💡 创新点**

通过从粗层深度生成稀疏3D锚点并在解码器中采用自适应令牌扩展（ATE）实现场景复杂度驱动的容量分配，解耦了高斯位置与像素网格。

**🔧 技术方法**

采用多视角编码器+自注意力、Anchor-Offset机制、Adaptive Token Expansion模块、3D相对偏移的高斯头以及不确定性监督。

**📊 数据集**

在RealEstate10K和DL3DV两个多视角合成数据集上进行训练和评估。

**📈 对比分析**

与pixelSplat、MVSplat、DepthSplat等前向3DGS方法及基于优化的3DGS比较，ATSplat在保持或提升渲染质量（PSNR/SSIM/LPIPS）的同时，Gaussian数量减少超过5.7倍，推理速度在512×960分辨率下可达1136 FPS。

**⚠️ 局限性**

缺点是未对冗余令牌进行裁剪，扩展机制在极端场景下可能导致容量浪费，并且目前仅支持已配位姿输入，无法直接处理无姿势图像。

---

## 454. Near-Optimal Dimension Lower Bounds for Single-Vector Embeddings of Maximum Inner Product Similarity

**arXiv ID:** 2607.20393 | [PDF](https://arxiv.org/pdf/2607.20393v1)

**作者:** Rajesh Jayaram `[一作]` (Google Research), David P. Woodruff `[通讯]` (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了针对多向量点云到单向量嵌入的维数下界，并证明在单元向量查询下的 Chamfer 相似度至少需要 m^{c/√2-2δ} 维。

**💡 创新点**

创新点在于将 Sherstov 的模式矩阵方法与高近似度 DNF 公式相结合，构造精确两值的 MAX-IP 矩阵，实现了与 MUVERA 上界几乎匹配的维数下界，几乎闭合了 1/√2 与 1/2 的指数差距。

**🔧 技术方法**

采用模式矩阵（pattern matrix）与近似秩（approximate rank）理论，利用宽度固定的高近似度 DNF 公式，并在单位球上实现 MAX-IP 与仿射变换。

**📊 数据集**

论文为理论证明，并未使用具体实验数据集，构造的是符号性查询与文档点云集合。

**📈 对比分析**

通过理论分析给出了单向量表示的最小维度下界，证明其与 MUVERA 上界几乎相同；未进行实验比较，性能表现主要体现在维数上与已知上界几乎相等。

**⚠️ 局限性**

局限性包括仅在单元向量查询和文档点云大小约束 m≥(1/ε)^Aδ 的情况下成立，点云阈值可能无法降低至 m≈1/ε^2，且实际算法实现尚未给出。

---

## 455. PG-KINN: A Physics-Informed Petrov-Galerkin Kolmogorov-Arnold Network for Solving Forward and Inverse PDEs

**arXiv ID:** 2607.20378 | [PDF](https://arxiv.org/pdf/2607.20378v1)

**作者:** Amirhossein Sadr `[一作]` (Shahid Beheshti University), Saeid Gorgin `[通讯]` (University of Hertfordshire)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Petrov–Galerkin弱形式的物理信息Kolmogorov–Arnold网络(PG-KINN)，将KAN试验空间与独立的多项式测试空间结合；

**💡 创新点**

创新点在于将KAN的可学习样条激活与Petrov–Galerkin弱化相结合，降低导数阶数、适用于非自伴算子、避免能量法在逆问题中的零模塌陷，并通过局部测试函数改善数值条件；

**🔧 技术方法**

采用KAN作为试验空间，使用Gauss–Legendre数值积分、自动微分、Adam+L‑BFGS优化、以及多阶多项式测试基；

**📊 数据集**

使用多种计算固体力学基准数据集，包括Mode III裂纹、带孔板、Neo‑Hookean悬臂梁、异质热导率纹理（Picasso、Caspar、Van Gogh）以及科赫雪花和花形域；

**📈 对比分析**

与传统MLP PINN、DEM、BINN以及SOTA KAN-PINN（PIKAN）比较，PG‑KINN在裂纹奇异、应力集中、非线性弹性、逆参数识别等任务中均实现了更低的相对L₂误差和应力误差，且在大多数基准上表现优于对手；

**⚠️ 局限性**

在极其复杂几何（科赫雪花、花形域）中，由于KAN样条网格为轴对齐矩形，难以逼近非矩形边界，导致精度受限。

---

## 456. Self Gradient Forcing: Native Long Video Extrapolation

**arXiv ID:** 2607.20368 | [PDF](https://arxiv.org/pdf/2607.20368v1)

**作者:** Junhao Zhuang `[一作]` (Joy Future Academy, JD), Nan Duan `[通讯]` (Joy Future Academy, JD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Self Gradient Forcing (SGF)，一种两步训练策略，在自回归视频扩散中通过不传播梯度的自回归跑前（Pass 1）和并行的上下文梯度重建（Pass 2）来修复历史上下文梯度缺失问题，从而提升长视频的连贯性与稳定性。

**💡 创新点**

创新点在于识别并解决冻结缓存 Self Forcing 中的历史上下文梯度缺失（context‑gradient gap），通过在第二步中恢复对已写入 KV 记忆的梯度，使未来损失能够直接指导早期生成的潜在如何被编码成可读的记忆，且不需要完整的 BPTT。

**🔧 技术方法**

使用技术包括：两通道无梯度自回归跑、并行上下文梯度重建、分块注意力与 KV 缓存、分块窗口与 FIFO 处理、分层噪声调度器、Distribution‑Matching Distillation (DMD) 监督，配合离散化扩散训练。

**📊 数据集**

实验主要基于 VBench 长视频评测数据集（5 s/60 s/240 s）、MovieGen 128 随机电影段以及 VBench‑Long prompts 进行验证。

**📈 对比分析**

与传统 Self Forcing 在相同初始化、提示、推理配置下进行配对对比。短期（5 s）性能基本相当；在 60 s 与 240 s 长期推理中，SGF 在美学质量、背景一致性、运动平滑度、主体一致性、闪烁等指标上普遍提升，用户研究（GSB）也显示显著更受偏好。

**⚠️ 局限性**

局限性：仍受限于当前 KV 缓存设计，动态度（motion degree）指标在某些设置下未显著提升；对极长时间跨度或高动态场景仍可能出现记忆衰减；实验数据集与场景有限，未来需结合更强初始化、检索增强或压缩缓存技术进一步提升。

---

## 457. No Extra Signals Needed: The Uniform Price of Explainable Information Design

**arXiv ID:** 2607.20364 | [PDF](https://arxiv.org/pdf/2607.20364v1)

**作者:** Francesco Bacchiocchi `[一作]` (Politecnico di Milano), Roberto Colomboni `[通讯]` (University of Bristol)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在一维线性贝叶斯说服问题中，研究了可解释信号方案（仅能划分为相邻区间）与一般信号方案在均匀先验下的收益差距，并给出了精确的价格（即最坏情况下可解释方案相对于最优方案的比例）。

**💡 创新点**

首次完全解决了在均匀先验、任意有界效用下的可解释性价格，证明当信号数量 K≥3 时价格为 2/3，而 K=2 时价格为 1/2；并给出相应的构造证明和紧致性上界。

**🔧 技术方法**

利用凸几何、随机中心打包（centered packing）和基于前缀不等式的递归拆分，结合极点分析与随机化构造，最终实现对任何可实现后验均值分布的稀疏打包保证两三分之二的期望长度。

**📊 数据集**

本工作为理论分析，不涉及任何实验数据集。

**📈 对比分析**

通过与最优非可解释方案的比较，证明可解释方案至少能获得 2/3（K≥3）或 1/2（K=2）的收益；构造极端效用函数示例显示这些比例是可实现的，因而是紧致的。

**⚠️ 局限性**

局限性在于仅考虑均匀先验和一维状态空间；对非均匀先验或更高维状态空间的可解释性价格尚未解决，且目前的构造方法难以直接推广到更一般的情况。

---

## 458. Look Less, Think Faster: Joint Token-Compute Adaptation for Multimodal LLMs

**arXiv ID:** 2607.20357 | [PDF](https://arxiv.org/pdf/2607.20357v1)

**作者:** Pengcheng Wang `[一作]` (Purdue University), Somali Chaterji `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种联合视觉token选择与LLM计算资源动态分配的自适应推理框架，能够根据输入内容和预算约束同时控制视觉token数、Transformer层数和宽度。

**💡 创新点**

创新点在于：①通过共享预算编码将视觉和语言两侧的决策耦合；②使用可微分的FLOPs估计器与Gumbel‑sigmoid实现端到端可训练的离散决策；③引入异步预算违约损失和动态预算采样，保证预算落地与性能平衡。

**🔧 技术方法**

技术方法包括：视觉token控制器（基于ViT + Gumbel‑sigmoid）；LLM计算控制器（层级与头组级可跳过控制）；共享预算编码；可微分延迟估计器；异步预算损失；训练时动态预算采样；对称的Gumbel‑sigmoid门控。

**📊 数据集**

使用七大多模视觉语言基准数据集：VQAv2、GQA、TextVQA、ScienceQA、VizWiz、POPE、MMBench。

**📈 对比分析**

与AdaLLaVA（仅LLM计算控制）、LLaVA‑PruMerge+ / FastV（仅token剪枝）、AdaLLaVA‑PruMerge（串联两步）以及全模型基线进行对比。结果显示在相同计算预算下，联合调度能够持续提升准确率，尤其在50% FLOPs时平均提升约7–8%，并在不同任务上获得更稳健的准确率‑效率Pareto前沿。

**⚠️ 局限性**

局限性包括：仅针对prefill阶段的FLOPs，未覆盖解码阶段；对硬件特定的稀疏执行支持要求较高；需要大量可微分训练，训练成本高；目前仅在7B/13B规模上验证，未知更大规模的迁移效果；对非CLIP‑ViT或非LLaMA基础模型的泛化性未充分探测。

---

## 459. Worst-Case Optimal BGPs on Temporal Graphs

**arXiv ID:** 2607.20356 | [PDF](https://arxiv.org/pdf/2607.20356v1)

**作者:** Diego Arroyuelo `[一作]` (Pontificia Universidad Catolica De Chile), Juan Reutter `[通讯]` (Pontificia Universidad Catolica De Chile)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `fede83ac-7505-405f-ab37-e7284695c47f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种面向时间标签图（temporal labeled graph）的最坏情况最优（wco）Basic Graph Patterns（BGP）查询评估框架，设计了线性空间的索引，能够在任意变量绑定顺序下高效处理时间约束、点/区间查询以及持续时间查询。

**💡 创新点**

创新点包括：
1) 将 Leapfrog Trie Join（LTJ）与版本化二叉树（VBT）结合，构建线性空间索引，保证 wco 评估时间；
2) 支持任意时间变量绑定顺序（相比传统的 time-first/ join-first 更灵活）；
3) 引入时间 BGP（tBGP）语言，能够表达复杂的时间约束；
4) 对点/区间查询给出更强的 wco 保障，满足 SQL:2011 版的快照和版本查询需求；
5) 通过实验验证方案在实际大规模图（如 Wikidata）上可在毫秒级完成查询，显著优于现有方法。

**🔧 技术方法**

核心技术：
- Leapfrog Trie Join（LTJ）算法的变体；
- 版本化二叉树（VBT）和标准二叉树（BT）实现时间级别的子树变更；
- 时间区间与子树后续属性的压缩存储；
- 变量绑定顺序优化（least‑children, join‑first, time‑first）；
- 通过位向量与 rank/select 进行 VBT 的无指针导航。

**📊 数据集**

实验数据集：
- Wikidata 知识图谱（完整图和仅含时间边的变体），约 844M 条三元组，16M 条时间化三元组；
- Divvy、Yellow、Caida 三个公开图数据集；
- 合成 BGP 查询集（星形、链、环、菱形、团等），用于与 Hu 等人和 Zhu 等人对比。

**📈 对比分析**

比较方法与性能：
- 与传统的 join‑first（JF）和 time‑first（TF）策略以及本研究提出的 least‑children（LC）策略直接比较；LC 在大多数查询中平均 1‑2 毫秒，比 JF 慢 20% 以内、TF 速度慢数倍；
- 与 Hu 等人的 1‑维时间窗口实现对比：在 Wikidata 上 LC/ JF 的点时间查询比 Hu 方案快 3–4 个数量级；
- 与 Zhu 等人的 TSRJoin 对比：在星形、链、环、菱形、团等形状上，本方案使用 TF 时平均快 2–100 倍；在更复杂图（如 Yellow、Caida）中快 1–2 个数量级；
- 结果显示查询响应时间全部落在毫秒级别，展示了系统的实用性。

**⚠️ 局限性**

局限性：
- 索引为静态，支持动态更新（插入/删除）仅在论文附录中提及，实际实现需要额外工作；
- 对时间约束仅支持 ≤ 关系，无法直接处理算术或非线性时间表达式；
- 变量绑定顺序（VEO）仍以经验式（least‑children）为主，未提供最优选择的自动化；
- 在时间跨度覆盖率极高、时间变量不具选择性时，性能下降；
- 虽然空间为 O(N)，但相比无时间图索引仍高出约 10 倍；
- 对极大规模图（N 级数十亿）时，构建成本与内存需求仍较高。

---

## 460. Closing the Lab-to-Store Gap: A Data-Efficient Post-Training and Experience-Driven Learning VLA Framework for Retail Humanoids

**arXiv ID:** 2607.20345 | [PDF](https://arxiv.org/pdf/2607.20345v1)

**作者:** Roger Sala Sisó `[一作]` (HIVE Robots), Tran Nguyen Le `[通讯]` (Technical University of Denmark)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出DEED框架，系统性地将Vision‑Language‑Action（VLA）模型部署到人形机器人上，完成真实超市芯片补货任务，并通过数据高效后训练、经验驱动微调与隐空间OOD分析三阶段实现可部署策略。

**💡 创新点**

创新点包括：①将频率层级、数据策划、IA‑VLA可视化、二进制手掌控制与Butterworth平滑等工程技术集成于VLA后训练；②改造RECAP优势条件化方法以适配GR00T的解耦架构；③构建与模型无关的隐空间GMM‑Mahalanobis OOD检测工具，用于诊断分布漂移。

**🔧 技术方法**

所用技术有：GR00T N1.6 VLA模型；IA‑VLA视觉突出；RECAP优势条件化；价值函数与返回分布化；文本前缀优势标记；二进制手掌控制；Butterworth动作平滑；频率层级与数据策划规则；GMM隐空间分布估计。

**📊 数据集**

实验使用了81条手动演示（约51.5分钟）和116条自主回放（约56.9分钟），全部在搭载双腕摄像头的Unitree G1‑Edu机器人上，环境为近似真实超市的模拟场景。

**📈 对比分析**

评估对比以SFT基线（0%成功）为起点，DE政策后成功率提升至32%（平均每包24.3s），一次RECAP后成功率升至42%（22.4s），第二次RECAP则下降至22%（21.1s）。实验表明数据高效后训练能将零功能模型转化为可用策略，经验驱动微调可带来一次性提升但若继续迭代易出现分布漂移。

**⚠️ 局限性**

局限性包括：仅在单一任务与单一机器人上验证，评估样本有限；经验驱动微调易导致自走回放占比过高引起分布漂移；重置行为未得到充分学习；对高变异任务的泛化能力待进一步验证。

---

## 461. Online Variance Reduction for Domain Adaptation on Streaming Data

**arXiv ID:** 2607.20374 | [PDF](https://arxiv.org/pdf/2607.20374v1)

**作者:** Andrea Napoli `[一作]` `[通讯]` (University of Southampton), Andrea Napoli (University of Southampton)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 ARROW，首个针对 MMD 与 CORAL 的在线随机方差减少算法，能够在流式、分布式或增量学习场景下实现域对齐。

**💡 创新点**

创新点：利用指数移动平均（EWMA）跟踪对齐统计量，并通过实例加权（可无约束线性化）实现自适应重加权；同时设计轻量级的权重优化方案，兼顾在线性时间与低内存需求。

**🔧 技术方法**

核心技术：实例加权重排、指数移动平均、线性/二次优化（求解权重）、V-统计量估计、对 MMD 的 RKHS 核函数展开、对 CORAL 的线性协方差近似。

**📊 数据集**

实验数据集：Spawrious（犬种分类）、Office-Home（图像分类）与 Humpbacks（声学鲸鱼检测）。

**📈 对比分析**

与多种采样/权重方法（均匀、k-means++、DPP、VaRDASS、ORDERED、PSDA）以及经典 UDA 基线（DANN、CDAN、SDAT、ELS、ARM、MCC）进行对比；ARROW 在估计方差、目标域准确率和训练时间上与离线 SVR 方法竞争，且在极大数据量场景下保持低时延。

**⚠️ 局限性**

局限：在 Monte Carlo 方差测试中不如部分离线方法显著；方法专注于 MMD/CORAL，扩展到其他非加性损失需进一步研究；需要调节衰减因子 α 与权重约束等超参，对实时系统的鲁棒性还有待验证。

---

## 462. SRAN: Scaling Named Data Networking via Map-and-Encap

**arXiv ID:** 2607.20363 | [PDF](https://arxiv.org/pdf/2607.20363v1)

**作者:** Tianyuan Yu `[一作]` (University of California, Los Angeles), Lixia Zhang `[通讯]` (University of California, Los Angeles)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SRAN（Scalable Routing Architecture for NDN），一种基于Map‑and‑Encap的可扩展路由与转发架构，将前缀到边缘路由的映射保留在网络边缘，核心仅使用拓扑信息进行转发。

**💡 创新点**

创新点在于将命名前缀与拓扑可达性统一拆分并在单一映射表中同时支持单播和组播，利用BIER将多路由映射编码为BitString实现无状态组播；在NDN原生安全与Sync机制下实现安全的前缀映射同步，无需额外路由安全协议。

**🔧 技术方法**

使用技术包括Map‑and‑Encap模式、NDNLPv2封装、Bit Index Explicit Replication (BIER)、NDN Sync、NDN安全签名、NDNd/NDN forwarder、Golang实现、Light VerSec信任模型等。

**📊 数据集**

实验数据集主要为Rocketfuel AS 1755、AS 3356、AS 2914 拓扑以及全边缘 Sprint PoP 拓扑；通过 Mini‑NDN 和 ns‑3 进行仿真与评估。

**📈 对比分析**

比较方法：与基线NDNd（直接将前缀放入FIB）对比，评估前缀传播收敛延迟、控制流量（packet overhead）以及核心转发表规模。结果显示SRAN在收敛延迟与基线相近，控制流量减少约30%‑70%，核心转发表规模随拓扑而非前缀数增长，显著提升可扩展性。

**⚠️ 局限性**

局限性：仍需在边缘保存前缀映射，导致边缘转发表增大；依赖可信根与信任链，未解决高级攻击（如边缘同步被劫持）；实验规模有限，未评估跨域交互或极大动态前缀变化的性能影响。

---

## 463. Multi-Source and Cross-Scenario Strategy-Guided Code Optimization

**arXiv ID:** 2607.20353 | [PDF](https://arxiv.org/pdf/2607.20353v1)

**作者:** Yuwei Zhao `[一作]` (Peking University), Yingfei Xiong `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大语言模型的代码优化框架，能够跨知识源和跨场景构建和迁移优化策略并生成验证过的静态分析规则；

**💡 创新点**

创新点在于将多种知识源（提交记录、教材、网页等）统一为“证据对象”，采用自平衡加权聚类平衡不同来源的频率与质量，并通过示例迁移与功能验证实现跨场景规则生成；

**🔧 技术方法**

主要技术包括证据对象抽象、LLM辅助文本和代码抽取、加权密度聚类、示例迁移与规则验证、Semgrep规则生成以及LLM驱动的补丁生成；

**📊 数据集**

使用了351条历史优化任务（151 C/C++、150 Python、50 Rust）以及15个真实项目（5 C/C++、5 Python、5 Rust）作为评测数据集；

**📈 对比分析**

与 SemOpt、Direct Prompt、RAG 以及 Codex Agent 进行对比，实验显示在历史任务上相对 SemOpt 的 Exact Match 提升 24.44%–180.00%，SemEqv 提升 21.88%–37.50%；在真实项目上最大性能提升范围为 19.72%–717.42%，平均提升 4.44%–258.17%，均明显优于基线；

**⚠️ 局限性**

主要局限包括可能存在的数据泄漏风险、评测基准和模型选择的依赖、对文档信息覆盖不足导致 Python/Rust 领域贡献有限，以及 LLM 训练数据无法完全审计等问题。

---

## 464. Flux-Corrected Diagonal Frog: second order and positivity at all time steps

**arXiv ID:** 2607.20415 | [PDF](https://arxiv.org/pdf/2607.20415v1)

**作者:** Andrey Itkin `[一作]` `[通讯]` (New York University), Andrey Itkin (New York University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种非线性Flux‑Corrected Diagonal Frog（FCDF）方案，能够在任意时间步长下对Fokker‑Planck方程保持非负、质量守恒并实现第二阶空间精度。

**💡 创新点**

创新点在于将DF算子拆分为单调M‑矩阵核心与反扩散修正，使用Zalesak型限幅器在隐式带状求解内部实现无条件正性与质量守恒，并通过主动集半光滑Newton方法克服了传统阈值限制，形成完整的步长覆盖策略。

**🔧 技术方法**

技术手段包括：斜率分解、Zalesak限幅器、Picard迭代、半光滑Newton主动集求解、缺陷修正时间步长、Pade(0,2)线性窗口、以及与Chang–Cooper指数拟合法的对比。

**📊 数据集**

使用的数据集为一维Ornstein–Uhlenbeck过程、平滑高Péclet数的常系数流动支配问题以及未解析前沿（高梯度区）等标准基准。

**📈 对比分析**

对比方法：在相同网格和时间步长下与无条件一次式、Chang–Cooper、有限差分基线进行误差、收敛率、负值和质量守恒比较。FCDF在高Péclet数下保持接近二阶空间收敛，缺陷修正方案实现二阶时间精度；在大步长下主动集求解成本仅与正性约束节点数相关，保持较低的计算开销。

**⚠️ 局限性**

局限性：需预先计算阈值（γ₀、γᵣ），在极端高Péclet数或深层尾部对阈值影响大；前沿区域的局部误差只能退化到一阶；多维扩展与耦合策略仍待进一步研究。

---

