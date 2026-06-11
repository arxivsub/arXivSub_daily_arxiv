# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-11 | 今日论文总数: 565

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. LifeSentence: Language models can encode human life course trajectories from longitudinal panel data

**arXiv ID:** 2606.11220 | [PDF](https://arxiv.org/pdf/2606.11220v1)

**作者:** Samuel Liu `[一作]`, Joshua J. Jackson `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 LifeSentence 模型，将预训练的大型语言模型与德国社会经济面板（SOEP）数据相结合，以自然语言形式表达生活事件，支持对人生轨迹的预测和自然语言查询。

**💡 创新点**

创新点在于：1) 将离散生活事件转化为结构化自然语言记录，让模型利用预训练中的分布式知识；2) 通过指令微调仅 65,000 名受访者的数据，显著提升预测精度并自动恢复教育、性别工资差距和母性惩罚等社会分层模式。

**🔧 技术方法**

技术手段包括：使用 24B 参数的 Mistral Small 3.1 作为基础模型，采用低秩适配 LoRA 进行指令微调；设计 18 任务评估体系（预测、鲁棒性、推理）；使用自然语言提示模板将任务指令、个人历史和状态信息拼接为单一文本输入，模型输出 JSON 结构化预测。

**📊 数据集**

数据集为德国社会经济面板（SOEP）v37，覆盖 1984–2020 年，约 65,000 名个体，包含教育、就业、家庭、健康等 20 种离散事件和连续状态变量。

**📈 对比分析**

与传统 Logistic / XGBoost、LSTM / GRU / Transformer 等基线模型对比，LifeSentence 在下一事件预测的联合准确率提升约 3 倍，MAE 下降至 1.15 年；在完整轨迹生成、受限生成、异常检测、顺序重排等任务中，在 Jaccard、Levenshtein、Wasserstein 等结构与时间指标上均优于基线，性能显著更好。

**⚠️ 局限性**

局限性包括：仅基于单一国家（德国）的面板数据，难以验证跨国泛化；事件词表仅涵盖 20 种类型，缺乏住宅迁移、刑事司法等重要维度；模型缺乏不确定性估计，输出为单一轨迹；预训练模型可能带来的社会偏见未得到充分剔除。

---

## 2. DarkVGGT: Seeing Through Darkness Using Thermal Geometry without Daylight Tax

**arXiv ID:** 2606.11326 | [PDF](https://arxiv.org/pdf/2606.11326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 3. Towards a Bridge Layer Between Bibliographic and Formalized Mathematical Knowledge

**arXiv ID:** 2606.11430 | [PDF](https://arxiv.org/pdf/2606.11430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 4. Knowing When to Ask: Self-Gated Clarification for Hierarchical Language Agents

**arXiv ID:** 2606.11349 | [PDF](https://arxiv.org/pdf/2606.11349v1)

**作者:** Aijing Gao `[一作]` (Amazon Web Services), Jae Oh Woo `[通讯]` (Amazon Web Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将澄清请求作为语言代理动作空间中的可选动作，并通过共享序数评分实现自我门控的澄清机制。

**💡 创新点**

创新点在于将澄清与导航动作置于同一评分尺度竞争，形成强制与机会性两种信息寻求模式，并通过受控答案通道验证局部有效性提升。

**🔧 技术方法**

采用基于LLM的动作评分与澄清子代理、受控答案通道以及信息寻求有效性（ISE）指标的实验框架。

**📊 数据集**

在海关与产品分类的 Harmonized Tariff Schedule（HTS）30,000 节点树上进行实验，使用 CBP-NY、ATLAS 与 HSCodeComp 三个基准数据集。

**📈 对比分析**

与仅基于提示或抽样一致性触发的澄清方法对比，本文在受控答案下实现 10 位码准确率从 50.8% 提升至 67.0%（+16.2%），同时 ISE 由 50% 提升至 74%。

**⚠️ 局限性**

局限在于受控答案通道不代表真实部署，实验仅涉及单一 HTS 结构，答案质量未系统探究，评分未校准，可能影响跨模型迁移与实际应用。

---

## 5. LAST: Bridging Vision-Language and Action Manifolds via Gromov-Wasserstein Alignment

**arXiv ID:** 2606.11221 | [PDF](https://arxiv.org/pdf/2606.11221v1)

**作者:** Huaihai Lyu `[一作]` (Chinese Academy of Sciences), Changsheng Xu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 26476 | [OpenAlex ID](https://openalex.org/A5022636178)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种基于 Lie 代数与 Gromov‑Wasserstein 对齐的动作空间标记化器，实现全局拓扑线性化与局部度量离散化，并将其嵌入 Vision‑Language‑Action 模型以完成跨模态对齐。

**💡 创新点**

① 引入 GW 视角解决语义空间与动作空间的几何与统计不匹配；② 在动作空间做 Lie 代数线性化并使用 B‑spline 参数化实现可加的全局表示；③ 通过协方差感知白化将局部残差近似球面；④ 结合 AR 与扩散头的双监督提升控制精度与泛化。

**🔧 技术方法**

Lie 代数映射、SE(3) 对数映射、B‑spline 时间抽象、B‑spline 回归、离散化量化、协方差白化、Gromov‑Wasserstein 对齐思想、Transformer AR 头、Diffusion Transformer、RVQ‑VAE 对比等技术。

**📊 数据集**

LIBERO（四个任务组）、SimplerEnv（多视角、变光照）以及真实机器人平台 AgileX Cobot Magic 上的 PlaceObj、ZipSeal、TubeRack 三个基准。

**📈 对比分析**

与 OpenVLA、SpatialVLA、π0、π0‑FAST、BEAST 等基线对比；在 LIBERO 上平均成功率 95.8% 超过所有基线；在 SimplerEnv 上 57.3% 成功率，优于 π0‑FAST‑Continuous、OpenVLA‑OFT；在真实机器人上 PlaceObj 73%、ZipSeal 57%、TubeRack 48% 的成功率均超过对比方法。重建 MAE 0.6e‑2，压缩率 8.0×，码本利用率 96% 也显著优于基线。

**⚠️ 局限性**

① 占用较高的白化计算成本，尤其在高自由度或高维系统中需采用低秩/收缩估计；② SE(3) 对数映射在接近切点（旋转接近 π）时局部线性失效，需多图覆盖；③ 极端旋转或稀疏采样轨迹可能需要多图线性化；④ 对离散化与白化近似的依赖在更高精度任务上可能存在局限。

---

## 6. PLUME: Probabilistic Latent Unified World Modeling and Parameter Estimation for Multi-Finger Manipulation

**arXiv ID:** 2606.11396 | [PDF](https://arxiv.org/pdf/2606.11396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 7. The Periodic Table of LLM Reasoning: A Structured Survey of Reasoning Paradigms, Methods, and Failure Modes

**arXiv ID:** 2606.11470 | [PDF](https://arxiv.org/pdf/2606.11470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 8. PT-WNO: Point Transformer with Wavelet Neural Operator for 3D Point Cloud Semantic Segmentation

**arXiv ID:** 2606.11466 | [PDF](https://arxiv.org/pdf/2606.11466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 9. Overcoming State Inertia in Full-Duplex Spoken Language Models via Activation Steering

**arXiv ID:** 2606.11386 | [PDF](https://arxiv.org/pdf/2606.11386v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 10. Mechanical Field Networks: Structured Neural Dynamics for Multivariate Systems

**arXiv ID:** 2606.11251 | [PDF](https://arxiv.org/pdf/2606.11251v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 11. OSCS-SupCon: Orthogonal Sigmoid-based Common and Style Supervised Contrastive Learning for Robust Feature Disentanglement

**arXiv ID:** 2606.11233 | [PDF](https://arxiv.org/pdf/2606.11233v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 12. RoVE: Rotary Value Embeddings Attention for Relative Position-dependent Value Pathways

**arXiv ID:** 2606.11275 | [PDF](https://arxiv.org/pdf/2606.11275v1)

**作者:** Alejandro García-Castellanos `[一作]` (University of Amsterdam), Erik J Bekkers `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为 RoPE 注意力机制在值通道引入相对位置旋转，使得值的映射也随相对位置变化，从而把原本位置无关的值通道变为位置敏感的。

**💡 创新点**

创新点是将 RoPE 转化为“可关注卷积”，通过把常数值映射替换为 offset‑indexed 旋转核，使得整个注意力混合器从 Kronecker 结构变为 block‑Toeplitz 结构，从而为值通道注入相对位置信息。

**🔧 技术方法**

技术主要包括：RoPE‑on‑Values 旋转、可关注卷积理论、矩阵混合器视角、FlashAttention 兼容实现，以及对相对位置的旋转矩阵应用。

**📊 数据集**

使用 FineWebEdu‑10B 训练 124M 和 354M GPT‑2 模型，并在 DCLM‑Core、RULER 等长上下文检索任务上进行评估。

**📈 对比分析**

与标准 RoPE 以及 RoPE+YaRN 进行对比，结果显示在 in‑context 学习（ICL）、长上下文 OOD perplexity 以及 RULER 检索任务上都有显著提升，尤其在需要长距离聚合信息的任务中效果最为突出。

**⚠️ 局限性**

局限性包括：实验仅覆盖中小规模 GPT‑2，未验证更大模型；对 RoPE 之外的位置信息编码方法适用性尚不明确；仅在特定长上下文任务上验证，可能在其他领域或模型架构中效果有限。

---

## 13. LSTM-Based Detection of Structural Breaks in Property Insurance Loss Reserving: A Climate-Informed Approach

**arXiv ID:** 2606.11463 | [PDF](https://arxiv.org/pdf/2606.11463v1)

**作者:** Thomas Mbrice `[一作]` (Stony Brook University), Shashwat Panigrahi `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本论文研究并实证比较LSTM网络在财产保险赔付发展中对结构性破裂的检测与传统链梯法、BF和Cape Cod等保留方法的性能，并提出基于概率的理论框架。

**💡 创新点**

创新点包括：① 将气候外生变量（ACE、海表温度等）嵌入LSTM并提供理论保证；② 在结构性破裂情境下证明LSTM检测速度至少比链梯法快k-1期；③ 引入注意力机制和门控解释以满足监管可解释性要求。

**🔧 技术方法**

使用技术包括：双向LSTM+注意力、递归门控、recency‑weighted 损失、dropout+L2正则化、梯度裁剪；以及传统链梯、BF、Cape Cod的模拟与置信区间估计。

**📊 数据集**

数据集来自佛罗里达OIR Schedule P（2007‑2023）与路易斯安那DOI（2007‑2023）共约140条保险三角；并补充NOAA HURDAT2、ERSST v5的ACE、海表温度等气候特征，覆盖四次重大飓风（Michael、Ida、Ian 等）。

**📈 对比分析**

通过时间分割训练/验证/测试，采用MAPE、RMSE、检测速度等指标与Diebold‑Mariano、Wilcoxon检验比较；结果显示LSTM在灾难年份的MAPE提升15‑20%，检测速度至少快4个发展期，且在非灾难期保持与传统方法相当。

**⚠️ 局限性**

局限性：仅针对佛罗里达和路易斯安那两州，测试样本仅四次结构破裂；计算成本高；气候变量仅为预测信号，未能确立因果关系；监管接受与推广需要时间。

---

## 14. CFCamo: A Counterfactual Detect-or-Abstain Framework for Camouflaged Object Detection

**arXiv ID:** 2606.11231 | [PDF](https://arxiv.org/pdf/2606.11231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 15. Least-Action-Guided Diffusion for Physical Extrapolation

**arXiv ID:** 2606.11277 | [PDF](https://arxiv.org/pdf/2606.11277v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 16. Quantifying Subliminal Behavioral Transfer Ratios in Language Model Distillation

**arXiv ID:** 2606.11270 | [PDF](https://arxiv.org/pdf/2606.11270v1)

**作者:** Uwe Konig `[一作]` (Technical University of Munich), Maheep Chaudhary `[通讯]` (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在两大语言模型（Llama-2-7B-Chat 和 Qwen2.5-7B-Instruct）上使用拒绝方向激活调度，生成不同程度的教师输出，并仅使用这些输出对学生模型进行蒸馏，随后用 GPT‑4.1 在 100 条 JailbreakBench 提示上评估攻击成功率，量化了潜在的“潜意识学习”转移比率。

**💡 创新点**

提出了结合拒绝方向激活调度与纯净数据蒸馏的可控实验流程，用以系统衡量潜意识行为转移；并首次揭示不同模型族（Llama 与 Qwen）在教师安全性受损时的转移尺度差异——Llama 显示阈值式跃迁，Qwen 则呈连续增大。

**🔧 技术方法**

利用机制可解释性提取拒绝方向并在多层进行线性激活干预；使用 QLoRA（4‑bit NF4，rank 16）进行学生微调；采用 GPT‑4.1 作为自动评判器评估 JailbreakBench 上的安全性；并采用 Wilson 置信区间等统计手段进行结果置信度分析。

**📊 数据集**

从 Mechanistic Anomaly Detection 数据集中抽取 1000 条纯净提示生成教师响应；利用 100 条 JailbreakBench 提示进行安全性评估；构造对比的控制与处理数据集用于蒸馏。

**📈 对比分析**

通过对教师与学生在不同 α 取值下的攻击成功率（ASR）进行对比，计算转移比例 τ；结果显示 Llama‑2 的学生在 α≈‑0.20 以上出现显著转移（τ≈0.25‑0.32），而 Qwen2.5 的学生在所有 α 下持续转移，最高达到 τ≈0.61；表明潜意识转移在不同模型族中呈现不同的尺度与阈值特征。

**⚠️ 局限性**

实验仅涵盖 7B 参数规模；仅做单轮蒸馏，未检验多轮累积效应；评估仅使用 GPT‑4.1 自动裁判，缺乏人工验证；样本量为 100 条 JailbreakBench，置信区间不完整；未探索随机正交方向的干预；仅研究同族蒸馏，跨族结果未知。

---

## 17. BioDivergence: A Benchmark and Evaluation Framework for Hidden Contextual Contradictions in Biomedical Abstracts

**arXiv ID:** 2606.11208 | [PDF](https://arxiv.org/pdf/2606.11208v1)

**作者:** Elias Hossain `[一作]` (University of Central Florida), Niloofar Yousefi `[通讯]` (University of Central Florida)

**通讯引用:** 367 | [OpenAlex ID](https://openalex.org/A5054474613)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出BioDivergence框架，用于在医学文献中分析隐藏的上下文导致的矛盾，区分直接矛盾与上下文条件差异；

**💡 创新点**

创新点在于构建六类冲突分类与13维差异轴本体，将矛盾分析转化为四个结构化预测任务，并提供article‑disjoint的银标注基准；

**🔧 技术方法**

使用BiomedBERT编码器与多头跨注意力、联合多任务学习，LLM生成对齐JSON的解释与证据跨度；

**📊 数据集**

基于PubMed与Europe PMC的5个医学领域（AMR、肿瘤、传染病、基因组学、临床试验/流行病学）共202,180篇摘要，提取527,907条主张并构造11,865对主张；

**📈 对比分析**

与多数基线（基于频率、词汇、零样本NLI、LLM、检索）对比，article‑disjoint下的参考模型在冲突类型上取得约0.69准确率，上下文矛盾F1约0.40，LLM零样本在冲突类型上略优但在轴和主因预测上逊色；

**⚠️ 局限性**

局限在于银标注质量不及人工标注，部分轴与冲突类样本稀缺，且对话式解释与证据跨度仍需进一步提升。

---

## 18. Automated Mediator for Human Negotiation: Pre-Mediation via a Structured LLM Pipeline

**arXiv ID:** 2606.11379 | [PDF](https://arxiv.org/pdf/2606.11379v1)

**作者:** Jamie Bergen `[一作]`, Sarit Kraus `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一套基于分解 LLM 管道的自动化预调解系统，支持人类谈判者在谈判前的准备工作。

**💡 创新点**

创新之处在于将预调解拆分为预测、对话、评审与总结四个专用模块，并通过目标性提示改进，显著降低 AI 的“奉承”倾向。

**🔧 技术方法**

技术上使用 GPT‑4o 作为核心 LLM，配合 Whisper 语音转文本，构建顺序管道并嵌入 SVI 参数预测与专门的评审子模型。

**📊 数据集**

使用自行设计的室友冲突模拟情境，收集 11 个 SVI 维度的交互数据及参与者的 Likert 量表自评，未采用公开大规模数据集。

**📈 对比分析**

通过两项受控人类实验与专业调解员对比；AI 在信任与自信指标上与人类持平，偏好推断误差降低 36%（RMSE 0.61 vs 0.95），并在改进后将“奉承”率从 36.6% 降至 16.8%，低于人类基准。

**⚠️ 局限性**

局限性包括样本量小、仅针对大学生室友情境、仅评估短期自评、未跟踪真实谈判结果，也未直接比较改进前后 AI 条件的因果效应。

---

## 19. AI Coding Agents in Social Science: Methodologically Diverse, Empirically Consistent, Interpretively Vulnerable

**arXiv ID:** 2606.11456 | [PDF](https://arxiv.org/pdf/2606.11456v1)

**作者:** Meysam Alizadeh `[一作]` (University of Oxford), Enkelejda Kasneci `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 Claude Code 与 Codex 两种 LLM 编码代理在社会科学任务中的设计与结论层面的多样性与偏差，探讨其对科学研究的影响。

**💡 创新点**

创新在于将人工智能代理的多样性拆分为设计层和结论层，并通过实验验证结论层更易被提示操纵，揭示了传统“同质化”关注点的不足。

**🔧 技术方法**

使用 LLM 代理（Claude Opus 4.7 与 GPT‑5.5）在无人工干预的自动化研究管道中完成代码生成、数据分析与结论撰写。

**📊 数据集**

使用国际社会调查计划（ISSP）数据及 Brady & Finnigan (2014) 的移民与社会政策支持数据，作为实验任务的真实数据集。

**📈 对比分析**

与 20 支随机抽样的人类研究团队进行比较，结果显示代理在设计层的多样性与人类相当或更高，但在结论层的偏差更大；确认性提示下 Claude Code 的结论从 10% 支持转为 90% 支持，指标分布基本不变。

**⚠️ 局限性**

局限包括仅测试单一社会科学问题、仅用两款代理、有限的提示类型、对长周期学术流程未建模，以及人类结论与代理结论无法直接对应。

---

## 20. When Poison Fails After Retrieval: Revisiting Corpus Poisoning under Chunking and Reranking Pipelines

**arXiv ID:** 2606.11265 | [PDF](https://arxiv.org/pdf/2606.11265v1)

**作者:** Xi Nie `[一作]` (University of Electronic Science and Technology of China), Wenbo Jiang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 3122 | [OpenAlex ID](https://openalex.org/A5038280308)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出一种针对多阶段检索增强生成(RAG)系统的语料库中毒攻击方法CRCP，解决现有攻击在文档分块与重排序后失效的问题。

**💡 创新点**

创新点在于：①把中毒优化从文档级别迁移到分块级别；②在优化过程中同时考虑检索相关性、重排序一致性、分块自洽性和边界鲁棒性；③通过显式建模分块转换和随机边界偏移，实现对不同分块策略的跨稳健性。

**🔧 技术方法**

技术手段包括：基于梯度的多目标优化（检索损失、重排序损失、局部一致性损失、边界鲁棒性损失）；分块策略的随机采样；使用开源检索模型(Contriever、ANCE、DPR)、重排序器(MonoT5、BGE-Reranker-Base)及多种LLM后端（GPT‑4o、LLaMA‑2‑7B等）。

**📊 数据集**

数据集：Natural Questions (NQ)、HotpotQA、MS‑MARCO。

**📈 对比分析**

与现有攻击方法（PoisonedRAG‑BB、RAG Paradox、Joint‑GCG等）以及防御策略（InstructRAG、ASTUTERAG、TrustRAG、SeconRAG）对比，CRCP在包含分块、检索、重排序和生成的完整RAG流水线中，攻击成功率超过88%，在不同分块尺寸和风格下保持稳定，显著优于传统方法。

**⚠️ 局限性**

局限性包括：需要对目标系统的检索与重排序模型有一定的白盒或score‑only 访问；对极端分块方式（如超大或极小分块）鲁棒性未充分验证；对不同语言或更复杂的多模态RAG系统的适用性尚待评估。

---

## 21. Multi-agent rendezvous in fluid flows via reinforcement learning

**arXiv ID:** 2606.11274 | [PDF](https://arxiv.org/pdf/2606.11274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 22. Exploring Adaptive Masked Reconstruction for Self-Supervised Skeleton-Based Action Recognition

**arXiv ID:** 2606.11450 | [PDF](https://arxiv.org/pdf/2606.11450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 23. Predictive and Spatially Aware Scheduling in Flexible Duplexing for Deterministic Communications

**arXiv ID:** 2606.11398 | [PDF](https://arxiv.org/pdf/2606.11398v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 24. Web-Native Graphical EMF Model Editors

**arXiv ID:** 2606.11442 | [PDF](https://arxiv.org/pdf/2606.11442v1)

**作者:** Susanne Göbel `[一作]`, Ralf Lämmel `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种纯客户端、基于 Angular 的框架 EMFular 以及对应的生成器，用于从 Ecore 元模型直接生成可立即使用并可高度定制的图形化 EMF 编辑器。

**💡 创新点**

创新点在于：①在浏览器中实现完整 EMF 语义与持久化，完全不依赖服务器；②利用 Angular 的组件与服务机制提供丰富的扩展点；③生成器可自动把任何 Ecore 模型映射为一个可运行的 Angular 项目；④框架与生成器满足六项评估准则（EMF 兼容性、无后端、SVG 渲染等）。

**🔧 技术方法**

技术栈包括 TypeScript、Angular、EMF‑Jackson（JSON 序列化）、SVG（Sprotty/自定义）、Angular Material、依赖注入和观察者模式。

**📊 数据集**

使用了经典 EMF 示例（BasicFamily、IFC Schema 约 700 EClass）以及 Atlantic Zoo 真实数据集进行测试与评估。

**📈 对比分析**

通过与 Sirius、GLSP、Gentleman 等现有框架的六维度对比（Ecore 兼容性、EMF 原生读写、开发约束、执行模式、技术栈、图形基础）来评估框架与生成器；结果显示生成器在大规模元模型上可稳定生成编辑器，且编辑器在功能、可定制性与资源占用上优于服务器依赖方案，性能满足实际应用需求。

**⚠️ 局限性**

局限性包括：生成器对子包、名称冲突等复杂元模型结构的处理尚不完善；缺乏对更复杂继承与多态等元模型特性的完整支持；验证与完整性检查仍在完善中，后续工作将聚焦于增强验证、扩展元模型构造支持和生成器的健壮性。

---

## 25. A prior-free blind detection of information leakage from model predictions

**arXiv ID:** 2606.11267 | [PDF](https://arxiv.org/pdf/2606.11267v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 26. Color-Rule-Function Encoding for Combinatorial Memory

**arXiv ID:** 2606.11365 | [PDF](https://arxiv.org/pdf/2606.11365v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 27. Game-Theoretic Foundations of Competition for Conscious Access

**arXiv ID:** 2606.11242 | [PDF](https://arxiv.org/pdf/2606.11242v1)

**作者:** Efthyvoulos Drousiotis `[一作]` (University of Liverpool), Sotiris Nikoletseas `[通讯]` (Patras University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文提出了一种博弈论框架，分析了人脑中意识访问的竞争机制，具体为内部模块在稀缺广播槽位上的竞争。通过选择成本放大努力，模块之间进行竞争，访问通过平滑的概率规则分配。

**💡 创新点**

创新点在于将意识访问的竞争形式化为一个博弈模型，提供了明确的分配规则，并分析了均衡存在性、捕获条件、计算效率和机制设计的性质。

**🔧 技术方法**

使用了博弈论工具，特别是竞赛理论和概率选择模型，分析了模块之间的竞争和访问概率的分配。

**📊 数据集**

论文中没有具体提到使用的数据集，而是通过理论模型进行分析。

**📈 对比分析**

通过与现有的注意力和选择模型进行比较，证明了在特定条件下，低价值表示可以通过放大优势捕获访问。性能分析表明，竞争强度和成本曲率对均衡的唯一性和稳定性有显著影响。

**⚠️ 局限性**

限制在于模型假设模块值和成本为固定原始量，未考虑动态变化和多重访问的情况。未来的研究可以扩展到多个同时访问槽位、动态模型以及模块值和成本的内生变化。

---

## 28. FreeBridge: Variational Schrödinger Bridges for Cellular Transition Dynamics

**arXiv ID:** 2606.11286 | [PDF](https://arxiv.org/pdf/2606.11286v1)

**作者:** Xurui Wang `[一作]` (Stony Brook University), Chenyu You `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 FreeBridge 方法，利用 Schrödinger 桥框架和单细胞几何约束，在仅有端点分布的条件下学习细胞转化动力学。

**💡 创新点**

①将单细胞状态与随机传输分离，定义 State Unit 为实例分割单细胞，固定几何；②在 Schrödinger 桥中加入基于经验支持的几何正则化，限制中间状态在真实细胞形态支持区；③通过端点匹配和支持正则化实现中间轨迹可解释性。

**🔧 技术方法**

Schrödinger Bridge 变分推断、时间条件神经网络驱动漂移、Euler–Maruyama 离散化、非参数支持距离估计、实例分割（Cellpose）与 ResNet‑50 编码器、FID/KID/支持违规率等评估。

**📊 数据集**

BBBC021、RxRx1、JUMP 高通量成像数据集。

**📈 对比分析**

与 PhenDiff、IMPA、CellFlux 等基线在统一单细胞管线下比较；FreeBridge 在所有数据集上端点 FID/KID 最优，且在 BBBC021 上支持违规率显著降低，语义保留（MoA 识别）也优于基线。

**⚠️ 局限性**

需要冻结编码器，无法补偿编码器未学习到的形态信息；支持距离计算 O(B²)，大规模时需近似；只能保证中间状态在经验支持范围内，无法验证真实时间轨迹。

---

## 29. Schützen: Evaluating LLM Safety in Bulgarian and German Contexts

**arXiv ID:** 2606.11316 | [PDF](https://arxiv.org/pdf/2606.11316v1)

**作者:** Kiril Georgiev `[一作]` (Sofia University St Kliment Ohridski), Ivan Koychev `[通讯]` (Sofia University St Kliment Ohridski)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评测了一套双语（保加利亚语–德语）安全评测数据集 Schützen，包含约8千条本土化风险提示，收集了15款 LLM 的响应并通过二分类与细粒度模式两种方法进行标注与评估。

**💡 创新点**

创新点在于：①将低资源与高资源语言置于同一地区性文化框架下进行对比，揭示跨语言安全行为差异；②结合人工‑AI 协作的标注流程和自动评判器，实现大规模、细粒度的安全性评估；③提供可公开复现的完整数据与评测脚本，促进欧洲多语境 LLM 部署研究。

**🔧 技术方法**

使用的技术包括：人工翻译与本土化（语言学专家），基于 GPT‑4.1 的自动安全判定器（与多种现成评测器比较），二分类与十分类的模式识别模型，实验中对 15 只模型（多语种、区域专用、开源/闭源）进行响应收集。

**📊 数据集**

使用的数据集是 Schützen：3,978 条保加利亚语提示与 3,978 条德语提示，总共 16 类危害、6 个风险领域，覆盖直接攻击、间接攻击与过度敏感评估三种提问模式。

**📈 对比分析**

比较方法：按安全响应比例对 15 模型排序，并绘制各风险领域的 unsafe 率分布；同时对细粒度安全模式进行统计，计算与人类标注的准确率、精确率、召回率与 F1。性能方面，顶尖模型在两语种上均达到 99.92% 以上安全率；但在区域性提示下安全率显著下降，且某些模型在保加利亚语的安全性能明显低于德语。

**⚠️ 局限性**

局限性包括：①数据集规模有限，未覆盖更复杂的 jailbreak 策略；②细粒度标签的判定仍受评判器主观性影响；③模型安全评估主要基于预训练语言模型，缺少对不同安全策略（如拒绝策略）的系统分析；④公开数据可能被恶意利用，需在使用时进行风险控制。

---

## 30. Towards Fully Automated Exam Grading: Fairness-Aware Recognition of Handwritten Answers with Foundation Models

**arXiv ID:** 2606.11477 | [PDF](https://arxiv.org/pdf/2606.11477v1)

**作者:** Hartwig Grabowski `[一作]` `[通讯]` (Offenburg University), Hartwig Grabowski (Offenburg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用通用视觉–语言基础模型（VLM）对纸质考试答案表中的手写大写字母进行识别，实现全自动分级；

**💡 创新点**

创新点包括：① 用VLM替代传统检测器，能够处理离格、删除、草书等复杂情况；② 引入参考答案弱提示，显著降低误判率并提升公平性；

**🔧 技术方法**

采用Google Gemini 3.1 Flash‑Lite、Gemini 3.5 Flash、Qwen3‑VL、OpenAI GPT‑5.2、xAI Grok‑4.3等VLM，并与YOLOv5/YOLO26检测器进行对比；

**📊 数据集**

基准数据为61份匿名考试（742页，3141个答案位置），包含官方评分和参考答案；

**📈 对比分析**

在统一基准上，VLM的识别准确率达约98%（相较于YOLOv5的90%显著提升），弱提示将误判率降至0.58%，在统一评分表下仅有3/61试卷分数下降；

**⚠️ 局限性**

局限性包括：数据仅来自单一机构；模型为闭源商业服务，易受快照更新影响；未对多机构、多书写风格进行广泛验证。

---

## 31. ProHiFlo: Hierarchical Flow Matching with Functional Guidance for De Novo Protein Generation

**arXiv ID:** 2606.11243 | [PDF](https://arxiv.org/pdf/2606.11243v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 32. Can AI Agents Synthesize Scientific Conclusions?

**arXiv ID:** 2606.11337 | [PDF](https://arxiv.org/pdf/2606.11337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 33. The Power of Test-Time Training for Approximate Sampling

**arXiv ID:** 2606.11437 | [PDF](https://arxiv.org/pdf/2606.11437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 34. An Information-Theoretic Analysis of Threshold Group Testing

**arXiv ID:** 2606.11353 | [PDF](https://arxiv.org/pdf/2606.11353v1)

**作者:** Remco van der Hofstad `[一作]` (Eindhoven University of Technology), Connor Riddlesden `[通讯]` (Eindhoven University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究阈值组检测（Threshold Group Testing，TGT）问题，推导了在无噪声、非自适应、常数列设计下的精确信息理论阈值，揭示了阈值 t 对测试数量的影响，并给出了在测试数超过该阈值时可实现唯一解的上界结果。

**💡 创新点**

创新点在于：①针对阈值检测而非传统阈值为 1 的组检测，首次得到完整的非自适应信息理论阈值公式；②证明在低稀疏度下（t=2）阈值可显著降低测试需求，而在高稠密度下则需要更多测试；③通过“隐蔽（disguised）”项的概念解释了何时出现竞争解，并提出相应的上界与下界。

**🔧 技术方法**

使用了随机图模型与球放入箱子方法、Poisson 与二项分布的总变差逼近、局部中心极限定理、Hoeffding 与 Chernoff 统计不等式、FKG 相关性不等式以及第一/第二矩法与局限性矩法的组合。

**📊 数据集**

本文为理论分析，未使用具体实验数据集；所有结果均通过概率模型（如随机常数列设计、随机Poisson/二项分布）获得。

**📈 对比分析**

与经典组检测（Classic Group Testing，CGT）进行比较。结果表明在低稀疏度（θ<1）时，TGT 的测试数可通过阈值 t 下降，达到或甚至优于 CGT 的信息理论极限；在高稠密度（θ>1/2）下，TGT 需要更多测试，证明了至少需要 O(n log n) 的测试才能完成恢复。

**⚠️ 局限性**

主要局限包括：① 需要假设 Conjecture 1（对于 t≥3 的情况）成立；② 仅适用于无噪声、非自适应设置；③ 对图设计有限制（不允许出现多重边且测试与项的度数有上限）。

---

## 35. Building Social World Models with Large Language Models

**arXiv ID:** 2606.11482 | [PDF](https://arxiv.org/pdf/2606.11482v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 36. Semantic Segmentation of Node and Edge Diagrams for Assistive Technology

**arXiv ID:** 2606.11320 | [PDF](https://arxiv.org/pdf/2606.11320v1)

**作者:** Michael Cormier `[一作]` (Mount Allison University), Miguel Nacenta `[通讯]` (University of Victoria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量级、双分支全卷积网络，用于语义分割节点-连线图（node‑link diagrams），以提升盲人/低视力用户的可访问性。

**💡 创新点**

创新点在于：①将最大池化与无池化分支相结合，兼顾大感受野与细节捕捉；②在合成数据上采用焦点损失（focal loss）进行训练；③在压缩（JPEG）图像上保持高精度，展示模型的鲁棒性。

**🔧 技术方法**

技术包括：全卷积神经网络（FCN）、两分支架构、ReLU激活、Dropout、1×1卷积分类层、焦点损失（focal loss）以及PyTorch框架。

**📊 数据集**

使用Graphviz程序化生成的合成节点‑连线图数据集，图形包含2‑15个节点、随机形状、颜色、文本标签，生成四种渲染（可视、语义、实例、文本），并通过JPEG压缩做数据增强。

**📈 对比分析**

与单分支（粗细）或仅细分支模型对比，完整模型在增强数据集上取得93.7%像素级精度；粗分支单独时仅84.9%；细分支8层时90.7%；细分支4层时91.3%；细分支2层时88.8%。在未压缩的干净图像上，准确率与压缩图像相近，证明模型稳健。

**⚠️ 局限性**

局限性包括：仅针对渲染图像；对纯线条绘图或复杂符号（电路图等）表现可能不足；需要进一步的实例分割和文本识别来完成完整的图形解析；未测试扫描或照片图像，数据集规模相对有限。

---

## 37. GLACIER: A Multimodal Student-Teacher Foundation Model for Molecular Property Prediction

**arXiv ID:** 2606.11382 | [PDF](https://arxiv.org/pdf/2606.11382v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 38. A2SG:Adaptive and Asymmetric Surrogate Gradients for Training Deep Spiking Neural Networks

**arXiv ID:** 2606.11236 | [PDF](https://arxiv.org/pdf/2606.11236v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 39. Energy-Conserved Neural Pipelines: Attenuating Error Propagation in Modular Neural Networks via Physical Conservation Constraints

**arXiv ID:** 2606.11341 | [PDF](https://arxiv.org/pdf/2606.11341v1)

**作者:** David Young `[一作]` (ORION Robotics), Swan Yi Htet `[通讯]` (ORION Robotics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在模块化神经网络管道中引入能量守恒约束，以消除模块间误差累积，提升鲁棒性。

**💡 创新点**

创新点：将物理能量守恒作为硬性约束强制执行，既不需要额外参数也不依赖批量统计，能够在模块边界上实现能量恒定、噪声削弱，并在训练过程中诱导权重分配，形成隐式去噪。

**🔧 技术方法**

核心技术：能量守恒算子（对每个样本按 L2 能量缩放），密度守恒扩展（用于维度不等的模块），实验中使用的噪声注入（高斯、系统偏差、对抗、丢弃），以及标准的交叉熵训练。对比方法包括无约束基线、软能量惩罚与硬能量守恒。

**📊 数据集**

数据集与环境：CIFAR‑10（卷积+全连接管道）、CIFAR‑10 的 ResNet‑18（带/不带 BatchNorm），以及基于 MuJoCo 与 Franka Panda 的真实感知‑控制机器人管道（3D 位置输入）。

**📈 对比分析**

对比结果：在 5 种随机种子、Gaussian 噪声 σ=0.2 时，硬能量守恒的准确率保持 77.4%（比基线 35.1% 提升 42.3pp，p<0.001）；深度从 2 增至 5，准确率保持 ~93%；对不同噪声类型（系统偏差 +45.1pp、Gaussian +40.4pp、对抗 +4.8pp、丢弃 -0.3pp）均表现出优势；在 ResNet‑18 上，移除 BatchNorm 后守恒优势提升至 +26.2pp（σ=0.2）和 +58.0pp（σ=0.5）；机器人实验中在深度漂移噪声下平均提升 18.9pp。对比基线与软能量惩罚的效果相反，惩罚导致准确率下降。

**⚠️ 局限性**

局限性：仅对能量增益型噪声有效，对信息破坏型噪声（如丢弃）无益；在高噪声且网络已具备 BatchNorm 的情况下，守恒可能导致 SNR 下降；当前仅在低维/中等维度任务上验证，未探索 Transformer、ImageNet 等更大规模场景；需进一步研究能量预算选择、非均匀重分配与多模块动态设置。

---

## 40. Dual-Stance Evaluation of Sycophancy: The Structure of Agreement and the Limits of Intervention

**arXiv ID:** 2606.11205 | [PDF](https://arxiv.org/pdf/2606.11205v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 41. Seeing Before Colliding: Anticipatory Safe RL with Frozen Vision-Language Models

**arXiv ID:** 2606.11266 | [PDF](https://arxiv.org/pdf/2606.11266v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 42. Invariant Price of Anarchy and Multiplicative Smoothness

**arXiv ID:** 2606.11397 | [PDF](https://arxiv.org/pdf/2606.11397v1)

**作者:** Ilia Shilov `[一作]` (ETH Zurich), Saverio Bolognani `[通讯]` (ETH Zurich)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文在缺乏跨人类可比性的前提下，提出基于Nash福利的价格无效性分析框架。

**💡 创新点**

创新点在于将PoA定义为CNC不变，提出乘法光滑性和保留包络技术，得到CNC不变的PoA上界。

**🔧 技术方法**

采用乘法光滑性、保留包络与几何闭包等数学工具，证明对单选福利游戏的PoA上界为2^p。

**📊 数据集**

未使用具体数据集，所有结论均为理论推导。

**📈 对比分析**

通过与传统加性光滑性方法对比，展示了在不需要全可比性假设下仍能得到与已知最优同阶的PoA上界（如2^p）。

**⚠️ 局限性**

局限在于仅处理正向效用、单选福利游戏，且对成本最小化或更一般分布式福利游戏的适用性仍待扩展。

---

## 43. SPEAR: A System for Post-Quantization Error-Adaptive Recovery Enabling Efficient Low-Bit LLM Serving

**arXiv ID:** 2606.11244 | [PDF](https://arxiv.org/pdf/2606.11244v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 44. Every Act Has Its Price: Compressed Moral Composition in Frontier LLMs

**arXiv ID:** 2606.11232 | [PDF](https://arxiv.org/pdf/2606.11232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 45. An Ethical eValuation Agent (EeVA): Results of a Proof-of-Concept Test on a Prototype Agentic-like Workflow to Assist Ethical Deliberations

**arXiv ID:** 2606.11218 | [PDF](https://arxiv.org/pdf/2606.11218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 46. Benchmarking Large Language Models for Safety Data Extraction

**arXiv ID:** 2606.11204 | [PDF](https://arxiv.org/pdf/2606.11204v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 47. Hubs or Fringes: Pretraining Data Selection via Web Graph Centrality

**arXiv ID:** 2606.11499 | [PDF](https://arxiv.org/pdf/2606.11499v1)

**作者:** Vedant Badoni `[一作]` (Princeton University), Xinyi Wang `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级的预训练数据选择框架，利用Common Crawl网站图的结构中心性分数对文档进行分层抽样，并构造混合训练集；

**💡 创新点**

创新点在于将网页图的拓扑结构（中心性）直接作为无监督数据筛选信号，避免使用辅助分类器或标签；同时证明中心性与内容质量分数互补；

**🔧 技术方法**

使用图论中心性度量（Betweenness、Katz）通过GPU并行图算法计算，结合DataComp‑LM预训练管道；

**📊 数据集**

基于Common Crawl host-level图（约13.9M节点、439.6M边）及其对应的Corpus‑200B预处理语料；

**📈 对比分析**

与随机抽样、质量分数抽样、WebOrganizer域混合等基线相比，在1B参数模型上平均提升至41.4%（相对随机提升1.6%），与质量分数结合后达到43.8%（相对随机提升4%）；

**⚠️ 局限性**

局限性包括：中心性分数对模型能力的影响因任务而异；混合比例需手动调节；仅在Common Crawl域内验证，跨域可推广性未知；

---

## 48. Steering Where to Listen: Instruction-Based Activation Steering Redirects Temporal Attention in Large Audio-Language Models

**arXiv ID:** 2606.11400 | [PDF](https://arxiv.org/pdf/2606.11400v1)

**作者:** Tsung-En Lin `[一作]` (National Taiwan University), Hung-Yi Lee `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了如何通过指令驱动向量操控来重塑大规模音频语言模型的时序注意力，从而实现无训练的音频事件定位。

**💡 创新点**

创新点在于引入仅通过对比不同指令激活差异的指令式向量操控，首次发现此方法能显著重分布音频注意力并用于定位。

**🔧 技术方法**

使用了指令式向量操控、注意力比例分析、滑动窗口定位探针等技术。

**📊 数据集**

采用了控制性三段合成音频基准、MMAU-mini、SAKURA等数据集。

**📈 对比分析**

与直接提示和随机基线对比，指令式操控在Qwen2-Audio和Audio Flamingo 3上分别实现了60.87%和68.72%的重叠率，显著优于传统方法。

**⚠️ 局限性**

局限在于定位准确度仍受模型原始语义域能力限制，且中间位置定位效果相对弱，后期层依赖性强。

---

## 49. Energy-Efficient On-Device RAG on a Mobile NPU: System Design and Benchmark on Snapdragon X Elite

**arXiv ID:** 2606.11257 | [PDF](https://arxiv.org/pdf/2606.11257v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 50. Loss Landscape Diagnosis for Gradient-Based Gray-Scott System Inversion: Disentangling the Roles of PINN Components

**arXiv ID:** 2606.11258 | [PDF](https://arxiv.org/pdf/2606.11258v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 51. TRON: Tracing Rays to Orchestrate a Neural Renderer for 3D Gaussian Reconstructions

**arXiv ID:** 2606.11314 | [PDF](https://arxiv.org/pdf/2606.11314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 52. T2MM: An LLM Supported Architecture For Inquiry-Based Modeling

**arXiv ID:** 2606.11210 | [PDF](https://arxiv.org/pdf/2606.11210v1)

**作者:** John Kos `[一作]` (Georgia Institute of Technology), Ashok Goel `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7709 | [OpenAlex ID](https://openalex.org/A5007028896)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在生态学概念建模软件 VERA 中，提出了一种基于大型语言模型（LLM）的交互式模型构建架构 T2MM，支持学习者通过自然语言请求进行多步骤的模型创建、编辑和参数调整。

**💡 创新点**

创新点在于：①将 LLM 生成的操作序列（而非完整 XML 代码）与当前模型状态结合，实现对模型的逐步、上下文感知修改；②通过 JSON 表示的最小化模型状态，减少生成错误和幻觉；③实现了可实时响应学习者手动调整的交互式模型。

**🔧 技术方法**

技术组成：ChatGPT‑4.1‑mini LLM、XML‑to‑JSON 转换器、模型状态检索与验证器、Vera API 调用；系统架构包含 VERA 前端、T2MM 处理层和 LLM 调用层。

**📊 数据集**

使用了 975 条人工生成的学习者自然语言请求与对应目标模型（由 5 个专家 XML 模型扩展得到）的程序化数据集，涵盖节点/关系创建/删除与参数调整等操作。

**📈 对比分析**

与 0‑Shot 和 N‑Shot 完整代码生成基线对比，T2MM 在 4 项评估指标上表现更优：编译成功率 100%（vs 96.62%/99.69%）；平均图编辑距离 0.298（vs 1.638/3.572）；在单步编辑、参数更改任务上成功率最高；在较长操作序列（>5 步）时性能下降明显。

**⚠️ 局限性**

局限性包括：仅针对 VERA 的专属本体，缺乏对其它科学建模环境的通用性验证；未使用真实学习者的自然语言请求，缺乏鲁棒性评估；未对模型的可视化呈现与学习者体验进行定量评估；对多步交互的完整性评估仍有待加强。

---

## 53. HiPi: Reproducible High-Fidelity Piezoresistive Sensors for Robotic Manipulation

**arXiv ID:** 2606.11372 | [PDF](https://arxiv.org/pdf/2606.11372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 54. A Polynomial-Time $O(\sqrt n)$-Approximation for Undirected Three-Terminal Reachability-Preserving Minimum Edge Cut

**arXiv ID:** 2606.11483 | [PDF](https://arxiv.org/pdf/2606.11483v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 55. SentTrack: Sentiment-Driven Bottleneck Detection in GitHub Issue Repositories

**arXiv ID:** 2606.11476 | [PDF](https://arxiv.org/pdf/2606.11476v1)

**作者:** Xinyu Hu `[一作]` (University of Tennessee), Nasir U. Eisty `[通讯]` (University of Tennessee)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SentTrack框架，通过横向主题聚类与纵向交互分析双视角自动检测GitHub issue线程中的社交技术瓶颈。

**💡 创新点**

创新点在于将LLM意图聚焦摘要与关切短语提取与ABCDE交互标签结合，形成可解释的优先级评分，首次将主题建模与线程动力学联结。

**🔧 技术方法**

使用GPT类LLM进行摘要和短语提取，SentenceTransformers+UMAP+HDBSCAN进行语义聚类，VADER情感分析，规则式ABCDE标签分类，以及多因子加权评分。

**📊 数据集**

使用AvaloniaUI开源仓库约9000条issue线程的数据集。

**📈 对比分析**

相较于传统标签或单一情感分析方法，SentTrack在识别长讨论未解决瓶颈方面提供了更早、更可解释的预警，虽然未给出具体准确率数值，但实验显示瓶颈检测更精准。

**⚠️ 局限性**

主要限制是LLM生成过多变体导致聚类碎片化，需要进一步约束词表或加入主题压缩步骤以提升可操作性。

---

## 56. The Environmental Cost of LLMs in AIED: Reporting and Practices

**arXiv ID:** 2606.11215 | [PDF](https://arxiv.org/pdf/2606.11215v1)

**作者:** Sabrina C. Eimler `[一作]` (Ruhr West University of Applied Sciences), Büsra Yapici `[通讯]` (Ruhr West University of Applied Sciences)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对AIED 2025会议论文进行系统综述，评估LLM使用的计算资源和环境成本；提出可复现的测量与报告框架，并示范使用CodeCarbon、TerraFlops以及FLOPs理论估算进行碳排放与性能的联合评估。

**💡 创新点**

①提出将PUE概念推广至本地硬件的PUE‑equivalent，构建统一的可持续性评估框架；②设计三种复合指标（Corrected Carbon Intensity、Carbon per Accuracy、Sustainability Score）；③提供开放源码工具包（TerraFlops wrapper）和简易FLOPs估算公式，填补AIED社区缺乏标准化环境报告的空白。

**🔧 技术方法**

使用Python开源工具CodeCarbon进行功耗与碳排放监测；构建TerraFlops包装器以结合PUE‑equivalent、机器学习性能指标；采用理论FLOPs计算公式（针对decoder‑only transformer）估算LLM推理成本；利用碳强度和电网数据生成CO₂排放估计。

**📊 数据集**

核心实验以UCI数据集Iris和Breast Cancer为例，用逻辑回归模型演示可持续性指标的计算；论文综述覆盖全部396篇AIED 2025会议论文。

**📈 对比分析**

通过对Iris和Breast Cancer数据集的逻辑回归实验，计算并对比其Accuracy与碳排放，得到Corrected Carbon Intensity、Carbon per Accuracy和Sustainability Score；示例显示即使准确率相近，低功耗模型在Sustainability Score上更具优势。

**⚠️ 局限性**

估算方法在参数未知或硬件多样化时可能低估或偏差；PUE‑equivalent基准取值仍需细化，未考虑云端水耗与功率峰值；复合指标的非线性校正尚未完善；工具对非专业研究者的易用性与可解释性待进一步验证。

---

## 57. Bernstein-Schur Kernels: Random Features by Sketched Modulation and Radial Randomization

**arXiv ID:** 2606.11255 | [PDF](https://arxiv.org/pdf/2606.11255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 58. Model-based Optimization of Anguilliform Swimming Gaits for Soft Robotic Applications

**arXiv ID:** 2606.11278 | [PDF](https://arxiv.org/pdf/2606.11278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 59. A PubMed-Scale Dataset of Structured Biomedical Abstracts

**arXiv ID:** 2606.11361 | [PDF](https://arxiv.org/pdf/2606.11361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 60. Learning from almost nothing: How neural networks survive heavy input corruption

**arXiv ID:** 2606.11319 | [PDF](https://arxiv.org/pdf/2606.11319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 61. Lifted Gabidulin Construction for LDPC Representations of Finite Geometry Codes

**arXiv ID:** 2606.11454 | [PDF](https://arxiv.org/pdf/2606.11454v1)

**作者:** Yifei Shen `[一作]` (EPFL), Andreas Burg `[通讯]` (EPFL)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种基于有限几何的稀疏 LDPC 码的构造方法，将传统密集的几何码基矩阵通过笔刷选择变稀疏化，并通过辅助变量节点实现 4‑cycle‑free 的子块；

**💡 创新点**

创新点在于将笔刷选择问题转化为常维子空间打包问题，并利用提升 Gabidulin 码实现可显式构造，从而在保持码距离的同时得到稀疏可迭代解码矩阵；

**🔧 技术方法**

核心技术包括有限几何点‑平面关联矩阵、辅助变量节点（AVN）拆分、常维子空间码（Lifted Gabidulin）以及低秩约束的子空间距离分析；

**📊 数据集**

使用了长度不超过 1024 位的有限几何代码（包含射影与仿射几何以及对应的 Reed‑Muller 码）作为实验数据集；

**📈 对比分析**

与 5G 标准 LDPC 矩阵进行对比，使用 BP 迭代解码，结果显示在 BLER=10⁻⁷ 时约获得 0.5 dB 的性能提升，且无可见误差底；

**⚠️ 局限性**

限制主要在于零偏移构造对 q>2 时不满足完整行空间保持，需要额外搜索偏移映射，且目前实验仅覆盖长度 ≤1024 的码，未探讨更大规模或非 2 字段情况。

---

## 62. Afrispeech Semantics: Evaluating Audio Semantic Reasoning in Spoken Language Models Across Domains and Accents

**arXiv ID:** 2606.11219 | [PDF](https://arxiv.org/pdf/2606.11219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 63. EverydayGPT: Confidence-Gated Routing for Efficient and Safe Hybrid GPT-RAG Conversational QA

**arXiv ID:** 2606.11212 | [PDF](https://arxiv.org/pdf/2606.11212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 64. NSVQ: Mitigating Codebook Collapse by Stabilizing Encoder Drift in Vector Quantization

**arXiv ID:** 2606.11363 | [PDF](https://arxiv.org/pdf/2606.11363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 65. Bridging the sim2real gap in the table tennis robot with a transformer-based ball states predictor

**arXiv ID:** 2606.11464 | [PDF](https://arxiv.org/pdf/2606.11464v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 66. Investigating Gender Bias in Touch Biometrics

**arXiv ID:** 2606.11457 | [PDF](https://arxiv.org/pdf/2606.11457v1)

**作者:** Joshua Lee `[一作]` (Bucknell University), Rajesh Kumar `[通讯]` (Bucknell University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了基于滑动手势的身份验证在性别上的公平性，并通过统计检验验证不同性别用户的误识率是否存在显著差异。

**💡 创新点**

创新点在于首次系统评估两种常用分类器（XGBoost 与 DenseNet）在公开滑动数据集上的性别公平性，并使用三种非参数统计检验（KS、Mann‑Whitney、Wasserstein permutation）对误识率分布进行比较。

**🔧 技术方法**

使用的技术包括梯度提升树 XGBoost、深度稠密连接网络 DenseNet、以及三种分布差异检验方法。

**📊 数据集**

使用的公开数据集为 BBMAS（117 位用户，72 男/45 女）和 ANTAL（71 位用户，56 男/15 女），两者均提供性别标签。

**📈 对比分析**

对 XGBoost 与 DenseNet 在两数据集上的准确率、FAR 与 FRR 进行比较，XGBoost 的准确率达到 92%~94%，FRR 低于 8%，FAR 在 6%~9% 之间；统计检验结果大多未显示显著性差异，表明两性别误识率基本相当。

**⚠️ 局限性**

局限性包括 ANTAL 数据集性别失衡导致检验力下降、仅考虑二元性别标签且缺乏其他人口统计属性，因而无法全面评估交叉公平性。

---

## 67. Embodied-R1.5: Evolving Physical Intelligence via Embodied Foundation Models

**arXiv ID:** 2606.11324 | [PDF](https://arxiv.org/pdf/2606.11324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 68. Great Disappearance Acts Generative Search and Shadow Banning

**arXiv ID:** 2606.11216 | [PDF](https://arxiv.org/pdf/2606.11216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 69. A Unified Lower Bound on the Noisy Query Complexity of Boolean Functions

**arXiv ID:** 2606.11448 | [PDF](https://arxiv.org/pdf/2606.11448v1)

**作者:** Yuzhou Gu `[一作]`, Yinzhan Xu `[通讯]` (University Of California San Diego)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出了一条统一的下界公式，用来估计布尔函数在噪声查询模型中的查询复杂度。

**💡 创新点**

创新点在于利用布尔超立方体子图的度统计（最小度和度分布）来推导通用的下界，从而一次性覆盖并简化了之前针对特定函数（如 k‑sum、对称函数、随机函数等）的多条下界。

**🔧 技术方法**

采用三阶段框架（非自适应查询、自由揭示信息、精确查询）以及概率、Chernoff、Azuma‑Hoeffding 与 martingale 等工具，对度统计进行分析，得到 𝖭_p(f) ≥ c·d₂·log(1+d₁) 的结果，并进一步推导出 𝖭_p(f)=Ω(𝖨(f)·log𝖨(f))。

**📊 数据集**

本研究完全基于理论分析，不使用任何实验数据集，所有结果均为渐近式证明。

**📈 对比分析**

与以往的特殊案例下界相比，新的统一下界在证明上更简洁、适用范围更广，能够以同一套方法恢复已知下界，且在大多数情形下实现与最优常数相当的性能。

**⚠️ 局限性**

该下界在最坏情况下并不一定紧，最佳可达的上限是 Ω(Δ·logΔ)（其中 Δ 为超立方体子图的最大度），因此对具有高度分布不均匀的函数仍可能存在较大的松弛。

---

## 70. MASK: Multi-Agent Semantic K-Scheduling for Risk-Sensitive 6G Robotics

**arXiv ID:** 2606.11249 | [PDF](https://arxiv.org/pdf/2606.11249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 71. EventRadar: Long-Range Visual UAV Discovery through Spatiotemporal Event Sensing

**arXiv ID:** 2606.11285 | [PDF](https://arxiv.org/pdf/2606.11285v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 72. Small Experiments, Cheaper Decisions: A Case Study in Staged Promotion for Micro-Pretraining

**arXiv ID:** 2606.11387 | [PDF](https://arxiv.org/pdf/2606.11387v1)

**作者:** Felipe Chavarro Polania `[一作]` `[通讯]` (Hewlett Packard Enterprise), Felipe Chavarro Polania (Hewlett Packard Enterprise)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定的单GPU微预训练跑器上，对12个候选配置实施分阶段的预算限制推广策略，评估短跑与长跑的稳定性与成本效益。

**💡 创新点**

设计了可审计的分阶段推广规则，包括预设参考/对照、复制短跑、冻结阈值和明确停止点，展示在有限预算下如何避免不必要的长跑。

**🔧 技术方法**

采用分阶段预算分配、重复seed、主机块对比、冻结阈值评估，基于bit‑per‑byte指标和GPU时数计量的实验框架。

**📊 数据集**

使用固定的byte/token流验证集，采用rustbpe/tiktoken兼容词表的验证块进行评估。

**📈 对比分析**

通过在Windows A100和Linux L40S两主机上跑多阶段预算，最终在12小时内比较bridge参考、贪婪对比、低参数 sentinel；bridge在所有host‑seed单元中排名第一，性能与成本对比表明小模型在吞吐量上有优势但在最终压缩率上不如bridge。

**⚠️ 局限性**

结果受容量混杂、seed与预算混淆、主机异构、样本量小、无对照基准以及缺少更大规模或更通用的HPO基线的限制。

---

## 73. Fixed-Parameter Tractability of Private Synthetic Data Generation

**arXiv ID:** 2606.11283 | [PDF](https://arxiv.org/pdf/2606.11283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 74. i1: A Simple and Fully Open Recipe for Strong Text-to-Image Models

**arXiv ID:** 2606.11289 | [PDF](https://arxiv.org/pdf/2606.11289v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 75. DeceptionX: Explainable Deception Detection with Multimodal Large Language Models

**arXiv ID:** 2606.11385 | [PDF](https://arxiv.org/pdf/2606.11385v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 76. Restless bandits with imperfect binary feedback: PCL-indexability analysis and computation

**arXiv ID:** 2606.11192 | [PDF](https://arxiv.org/pdf/2606.11192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 77. Scenario-based Probing and Steering Cultural Values in Large Language Models--Extended Version

**arXiv ID:** 2606.11399 | [PDF](https://arxiv.org/pdf/2606.11399v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 78. Mirror Descent Beyond Euclidean Stability: An Exponential Separation in Initialization Sensitivity

**arXiv ID:** 2606.11431 | [PDF](https://arxiv.org/pdf/2606.11431v1)

**作者:** Shira Vansover-Hager `[一作]` (Tel Aviv University), Tomer Koren `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了镜像梯度下降（MD）对初始化扰动的敏感性，给出了在良好条件下非二次正则化以及KL正则化下的指数级放大下界，并提出通过锚定技术来稳定MD。

**💡 创新点**

首次证明了非二次镜像映射即使在凸光滑目标下也会出现指数级初始化不稳定性，并且提出了两种Bregman正则化的锚定变体能够显著改善此类不稳定性。

**🔧 技术方法**

使用了镜像梯度下降理论、Bregman散度分析、极值构造、旋转坐标法、乘法权重（Multiplicative Weights）形式以及相对光滑性框架进行证明。

**📊 数据集**

本文为理论研究，无使用实际数据集。

**📈 对比分析**

通过构造性下界与匹配上界理论分析进行对比，证明了指数级放大现象以及锚定技术在理论上能将不稳定性降低到多项式或1/T级别。

**⚠️ 局限性**

局限性包括仅给出最坏情况下的下界，未在实际任务中验证；对抗性扰动假设可能过于强；并未考虑平均情况或随机初始化的稳定性。

---

## 79. INFRAMIND: Infrastructure-Aware Multi-Agent Orchestration

**arXiv ID:** 2606.11440 | [PDF](https://arxiv.org/pdf/2606.11440v1)

**作者:** Ahasan Kabir `[一作]` (University of Central Florida), Qian Lou `[通讯]` (University of Central Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向基础设施的多代理LLM调度框架，在每个决策层（规划、执行、调度）结合实时 GPU 队列深度、KV 缓存压力和延迟信号，实现自适应拓扑、模型选择和推理深度。

**💡 创新点**

创新点在于：①将基础设施状态作为约束条件，构建层次化的受限马尔可夫决策过程；②在规划器、执行器和 EDF 调度器中统一使用强化学习训练，自动学习质量与延迟权衡；③通过 FiLM 调制、双通路策略和 Lagrange 多目标优化实现跨层次协同。

**🔧 技术方法**

技术方法包括：层次化受限 MDP；PPO 与 REINFORCE 联合训练；FiLM 特征调制；双通路（语义+资源）策略网络；EDF 最早截止优先调度；双目标 Lagrange multiplier 控制预算；vLLM 监控接口。

**📊 数据集**

实验使用五大基准：MBPP、HumanEval、GSM‑Hard、MATH、MMLU‑Pro；模型池包含从 3B 到 32B 的 5 个 LLM，部署在共享 GPU 集群。

**📈 对比分析**

与 MoA、GPTSwarm、MasRouter 等基线对比，本文在低负载下最高提升 7.6pp 准确率、在高负载下可达 99.9% 的 300s SLO 合规率，并将平均延迟降低多达 7 倍。

**⚠️ 局限性**

局限性：规划阶段固定拓扑，假设模型池静态；未支持运行时拓扑重新决策或弹性硬件配置；对极端动态负载或自动伸缩的适配尚待研究。

---

## 80. Compatibility-Aware Dynamic Fine-Tuning for Large Language Models

**arXiv ID:** 2606.11206 | [PDF](https://arxiv.org/pdf/2606.11206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 81. Defeater Cards: Characterizing and Managing Safety Assurance Case Defeaters

**arXiv ID:** 2606.11462 | [PDF](https://arxiv.org/pdf/2606.11462v1)

**作者:** Usman Gohar `[一作]` (Iowa State University), Robyn R. Lutz `[通讯]` (Iowa State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Defeater Cards，一种结构化的文档框架，用于记录和管理安全保障案例中的“defeaters”（挑战性论点）

**💡 创新点**

创新点在于将 5W1H 思维模型与安全案例文档结合，形成统一、可审计、可重用的 defeater 记录模板，并提供公开仓库作为社区共享基线

**🔧 技术方法**

使用了系统化文献综述、主题分析和设计科学方法，结合安全案例工具（如 GSN）以及 AI 文档模板（Model Card、Data Card）设计框架

**📊 数据集**

数据集为两项跨域案例研究：一是无人机（sUAS）安全案例的 defeater 卡，二是分子程序（molecular programming）安全案例的 defeater 卡，并在公开仓库中提供十二个额外卡片

**📈 对比分析**

通过在 sUAS 和分子程序两大不同领域的案例验证，展示了 Defeater Cards 能揭示隐藏假设、识别推理漏洞并支持安全案例演化；相较传统手工记录，提升了可追溯性、可审计性和跨系统复用性

**⚠️ 局限性**

局限性包括：对完整性缺乏形式化保证，需投入额外时间与资源；存在“defeater 黑客”风险，即可能被有意遗漏或误归类；适用性仍需在更大规模、不同监管环境下进一步评估

---

## 82. ProcessThinker: Enhancing Multi-modal Large Language Models Reasoning via Rollout-based Process Reward

**arXiv ID:** 2606.11209 | [PDF](https://arxiv.org/pdf/2606.11209v1)

**作者:** Jingpei Wu `[一作]` (LMU Munich), Volker Tresp `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为ProcessThinker的后训练管道，旨在为多步骤推理提供逐步过程奖励，而无需训练显式的过程奖励模型（PRM）。

**💡 创新点**

创新点在于通过重写推理轨迹为逐步标记格式，结合基于回滚的过程奖励，提供密集的信用分配，从而鼓励更可靠的推理步骤，减少不一致或自相矛盾的进展。

**🔧 技术方法**

使用了基于回滚的过程奖励和群体相对策略优化（GRPO）技术。

**📊 数据集**

使用了四个视频基准数据集进行评估，包括Video-MMMU、MMVU、VideoMathQA和LongVideoBench。

**📈 对比分析**

与基线模型Qwen3-VL-8B-Instruct相比，ProcessThinker在所有四个基准上均表现出一致的改进，尤其是在VideoMathQA上提高了6.47个百分点，表明其在长时间推理中的有效性。

**⚠️ 局限性**

主要限制在于效率：尽管避免了PRM的注释和训练，但每个步骤的继续可解性需要多个回滚，增加的推理成本可能抵消使用更少数据的好处。

---

## 83. From Awareness to Action: Understanding and Overcoming the Research-Practice Gap in Algorithmic Fairness for Public Health

**arXiv ID:** 2606.11214 | [PDF](https://arxiv.org/pdf/2606.11214v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 84. Federated continual learning: A comprehensive survey on lifelong and privacy-preserving learning over distributed and non-stationary data

**arXiv ID:** 2606.11272 | [PDF](https://arxiv.org/pdf/2606.11272v1)

**作者:** Masoume Gholizade `[一作]` (University of Pisa), Francesco Marcelloni `[通讯]` (University of Pisa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了联邦持续学习（FCL）的定义、挑战与发展；

**💡 创新点**

提出了统一的多维度分类法、系统化评估框架和未来研究路线；

**🔧 技术方法**

回顾了FL与CL的核心技术，聚焦参数重要性、回放、稀疏/模块化、元学习、异步/压缩、差分隐私等方法；

**📊 数据集**

涉及医疗、工业物联网、智能城市等多领域常用数据集（如医学影像、传感器流、语音等），但并未单独实验；

**📈 对比分析**

通过对比表格和实验评测指标（长期准确率、遗忘度、通信效率、存储需求），展示了各方法在不同场景下的优势与局限；

**⚠️ 局限性**

主要局限包括：对极端漂移和多样性处理不足、隐私与记忆机制的可扩展性低、缺乏统一基准与评测协议、系统层面资源与异步性挑战未彻底解决。

---

## 85. From Consumption to Reflection: Designing Human-AI Relations for Stable Reasoning

**arXiv ID:** 2606.11195 | [PDF](https://arxiv.org/pdf/2606.11195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 86. JailbreakOPT: Tool-Assisted Iterative Jailbreak Prompt Optimization

**arXiv ID:** 2606.11425 | [PDF](https://arxiv.org/pdf/2606.11425v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 87. Approximation Properties of Evolutionary Dynamics in Continuous-Time Finite State Space Games

**arXiv ID:** 2606.11193 | [PDF](https://arxiv.org/pdf/2606.11193v1)

**作者:** Pietro Grassi `[一作]` `[通讯]` (University of Pisa), Pietro Grassi (University of Pisa)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本论文研究了有限人口随机进化动态在连续时间有限状态空间博弈中的收敛性，特别是其向确定性均场极限的收敛性。

**💡 创新点**

创新点在于发展了精细的遍历定理，证明了均场模型的唯一解依赖于初始条件，并且在足够大的N下，混合静态纳什均衡可以通过相应的N人博弈的纳什均衡来近似。

**🔧 技术方法**

使用了马尔可夫链的遍历定理、Kolmogorov的强大数法则和Kurtz定理等技术。

**📊 数据集**

使用了自定义的数值模拟数据集，具体的实验设置和参数选择在论文中详细描述。

**📈 对比分析**

通过数值模拟验证了理论结果，证明了在不同人口规模下，经验状态-政策分布以𝒪(N^-1/2)的收敛速率趋向于均场轨迹。

**⚠️ 局限性**

限制在于模型假设了固定的政策和特定的状态转移结构，可能不适用于更复杂的动态博弈场景。

---

## 88. The Structural Attention Tax: How Retrieval Format Hijacks In-Context Learning Independent of Content

**arXiv ID:** 2606.11198 | [PDF](https://arxiv.org/pdf/2606.11198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 89. One Jailbreak, Many Tongues: Learning Language-Insensitive Intention Representations for Multilingual Jailbreak Detection

**arXiv ID:** 2606.11202 | [PDF](https://arxiv.org/pdf/2606.11202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 90. A Scalable PyTorch Abstraction for Multi-GPU Gaussian Splatting

**arXiv ID:** 2606.11390 | [PDF](https://arxiv.org/pdf/2606.11390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 91. Density estimation for Hellinger via minimum-distance estimators: mixtures of Gaussians, log-concave, and more

**arXiv ID:** 2606.11469 | [PDF](https://arxiv.org/pdf/2606.11469v1)

**作者:** Spencer Compton `[一作]` (Stanford University), Jerry Li `[通讯]` (University of Washington)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出了一种基于最小距离估计的新框架，在Hellinger距离下实现近线性时间、近最优样本复杂度的密度估计；

**💡 创新点**

将最小距离估计从总变差推广到Hellinger，通过引入比率类与逆数据处理不等式实现，构造可实现的Hellinger度量；

**🔧 技术方法**

使用逆数据处理不等式、VC维理论、比率类距离、稀疏多项式逼近、贪婪合并算法以及区间优化等技术；

**📊 数据集**

论文主要为理论分析，并未在公开数据集上进行实验；

**📈 对比分析**

与现有ρ‑估计和总变差下的近线性算法相比，取得了相同或更优的统计误差率，同时将运行时间从多项式降低到近线性（如高斯混合模型样本复杂度 O(k/n)，时间 O(n log^5 n)）；

**⚠️ 局限性**

限制包括：缺乏实证验证；仅针对一维结构分布；对数因子仍存在；算法实现复杂且常数较大。

---

## 92. Forecasting Future Behavior as a Learning Task

**arXiv ID:** 2606.11445 | [PDF](https://arxiv.org/pdf/2606.11445v1)

**作者:** Mosh Levy `[一作]` (Bar-Ilan University), Asa Cooper Stickland `[通讯]` (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并训练了 Behavior Forecaster，通过仅观察一次大型推理模型的推理轨迹即可预测其未来行为，如重复一致性和对抗性敏感性。

**💡 创新点**

创新点在于将行为预测转化为可学习的监督任务，利用轨迹信息而非表面解释，并证明其在多个数据集上优于传统读者。

**🔧 技术方法**

采用大型预训练语言模型（OLMo‑3.7B‑Think、Qwen3.5‑2B）作为后端，训练专用预测头，使用交叉注意力池化和提示回声输入方式。

**📊 数据集**

实验使用三种推理数据集：TreeCut、FEVEROUS、RuleTaker。

**📈 对比分析**

与 GPT‑5.4、Claude Opus 4.6 以及单点探测器对比，Behavior Forecaster 在 Spearman 相关性上提升至 0.65–0.74，且推理成本低于对手 1/10,000。

**⚠️ 局限性**

局限包括对跨任务 OOD 泛化不足，以及对更可信推理模型的依赖性高。

---

## 93. Designed-Source Reductions and a Dual-Purpose Feasibility Band for Semantic Rate-Distortion

**arXiv ID:** 2606.11280 | [PDF](https://arxiv.org/pdf/2606.11280v1)

**作者:** Joss Armstrong `[一作]` `[通讯]` (Ericsson Research), Joss Armstrong (Ericsson Research)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

该论文研究了在设计源（即语义对象为确定性分配规则）下，如何利用 Stavrou‑Kountouris（SK）框架进行双重目标（福利失配与检测失配）的压缩设计，并提出了“可行性带”概念；

**💡 创新点**

创新点在于：1）在设计源子类中把 SK 的指数倾斜解和通用 Blahut‑Arimoto 迭代化简为条件均值解与 Lloyd‑Max 迭代；2）揭示当检测失配随率递增（与福利失配递减）时，双目标问题不再属于 SK 单字母可接受类，产生一个基于类别数的可行性带；3）与信息瓶颈、间接率失配等理论框架的对比。

**🔧 技术方法**

采用信息论率失配理论、指数倾斜参数化优化、Blahut‑Arimoto 迭代、Lloyd‑Max 量化、凸分析的曲率夹逼和 MMSE 论证；

**📊 数据集**

实验使用：1）模拟高斯设计源；2）智能电网经济调度与计费（N=4）案例；3）非技术性损失（NTL）检测的高斯位移模型。

**📈 对比分析**

与 SK 原始双目标解、信息瓶颈及间接率失配等方法对比，实验表明：在设计源情形下，单一 KL 曲线即可描述两类失配；而在非设计源情形下，失配前沿不重合，无法简化。可行性带在满足检测功率时提供了更宽的类别数区间，实验结果与理论宽度一致。

**⚠️ 局限性**

局限性包括：仅适用于平滑凸效用、欧几里得分配域、确定性分配规则；不适用于随机源、对抗或战略式编码、非凸或离散分配空间；对检测失配的“单字母”假设不成立时需完整 SK 工具。

---

## 94. When Probing Accuracy Saturates, Fragility Resolves: A Complementary Metric for LLM Pre-Training Analysis

**arXiv ID:** 2606.11375 | [PDF](https://arxiv.org/pdf/2606.11375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 95. RAIL: Rethinking Auditory Intelligence in Large Audio-Language Models with a CHC-Grounded Benchmark

**arXiv ID:** 2606.11260 | [PDF](https://arxiv.org/pdf/2606.11260v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 96. CRUMB: Efficient Prior Fitted Network Inference via Distributionally Matched Context Batching

**arXiv ID:** 2606.11473 | [PDF](https://arxiv.org/pdf/2606.11473v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 97. PoQ-Judge: A Multi-Architecture Evaluation Framework for Cost-Aware Proof-of-Quality in Decentralized LLM Inference

**arXiv ID:** 2606.11196 | [PDF](https://arxiv.org/pdf/2606.11196v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 98. LakeFM: Toward a Foundation Model for Aquatic Ecosystems Using Irregular Multivariate Multi-depth Time Series Data

**arXiv ID:** 2606.11268 | [PDF](https://arxiv.org/pdf/2606.11268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 99. An Entropy-based Framework for Hybrid Coalitions in Game Theory. Part I: Human Arbitration

**arXiv ID:** 2606.11288 | [PDF](https://arxiv.org/pdf/2606.11288v1)

**作者:** Salome A. Sepulveda-Fontaine `[一作]`, Jose M. Amigo `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出Neo-Game Theory框架，定义了人机混合联盟在虚拟自然环境中的决策过程，并实现了首个Human Arbitration（人类仲裁）方案；通过仿真验证了基于Jensen‑Shannon熵差的阈值委托机制能使AI逐步与人类对齐；

**💡 创新点**

创新点在于：①将经典博弈论的完整性、可传递性等公理扩展为可变执行权的混合博弈；②引入熵度量（Jensen‑Shannon散度）作为动态委托阈值，实现分区式的执行权切换；③用词典序（lexicographic）层次构建联盟效用，消除了传统效用混合的缺陷；④提出频率收敛型平衡概念，捕捉长期执行权分配的统计稳定性；

**🔧 技术方法**

技术方法包括：基于Jensen‑Shannon散度的阈值委托规则、经验频率学习（empirical policy）与EWMA（指数加权移动平均）奖励跟踪、Robbins‑Monro型递推学习、虚拟自然（Virtual Nature）状态转移模型、改进的Bellman递归和分段式价值函数；

**📊 数据集**

使用的是合成的二维状态/动作空间（S={A,B}，A={a0,a1}）以及相应的随机转移概率；未使用公开真实数据集；

**📈 对比分析**

比较方式主要是对不同学习率（常数vs衰减）和不同状态生成方式（外生vs动作相关）下的D_JS收敛、执行频率、策略一致性等指标；实验结果显示，无论参数组合如何，长期趋向人机策略对齐且D_JS接近零；在短期内，学习率与环境选择影响收敛速度与振荡幅度。

**⚠️ 局限性**

局限性包括：只研究了首个仲裁情景，未探讨AI‑Control与Negotiation等后续模式；实验仅在极简的二状态两动作模型上进行，缺乏大规模验证；理论上对频率收敛平衡的收敛性只给出经验或ODE方法支持，缺少严格证明；此外未与传统Nash/Stackelberg等平衡进行定量对比。

---

## 100. Position: Hippocampal Explicit Memory Is the Cornerstone for AGI

**arXiv ID:** 2606.11245 | [PDF](https://arxiv.org/pdf/2606.11245v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 101. Traits Run Deeper: Trait-Specific Asymmetric Fusion for Personality Assessment

**arXiv ID:** 2606.11269 | [PDF](https://arxiv.org/pdf/2606.11269v1)

**作者:** Jia Li `[一作]` (Hefei University of Technology), Meng Wang `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于特质的非对称融合框架（Traits Run Deeper），用于从视频、音频、文本三模态的访谈中预测 HEXACO 人格维度的连续分数。

**💡 创新点**

创新点在于：①将心理学理论的语义模板作为语义锚点嵌入多模态特征；②为每个特质设计可选择的模态组合与融合策略，形成特质专属的非对称融合；③采用分布校正回归（Yeo‑Johnson + 高斯平滑）缓解标签不平衡与中心趋向偏差。

**🔧 技术方法**

技术手段包括：Gemini Embedding 2 作为统一多模态基础表示；基于注意力、交叉模态注意、拼接等多种融合方式；多层回归头和五折集成；分布校正回归与逆变换。

**📊 数据集**

使用的公开数据集是 AVI Challenge 2026，包含 644 名受访者的结构化访谈视频，评估 4 个 HEXACO 维度（Honesty‑Humility、Extraversion、Agreeableness、Conscientiousness）。

**📈 对比分析**

通过与基线方法和挑战赛前十名提交的结果对比，本文方法在验证集上平均 MSE 降至 0.2521（比基线下降约 25%），在官方测试集上取得 MSE 0.27767，排名第一。

**⚠️ 局限性**

局限性在于：①仅在特质层面进行模态选择，未能对个体实例进行动态调节；②受限于现有数据规模和模态数量，可能难以泛化至更大或更丰富的场景；③模型解释性有限，缺乏对模态贡献的可视化和因果分析。

---

## 102. Physics-informed generative AI for semiconductor manufacturing: Enforcing hard physical constraints in generative models by construction

**arXiv ID:** 2606.11247 | [PDF](https://arxiv.org/pdf/2606.11247v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 103. MJSAC: McCormick Relaxation-based Waveform Design for Joint Sensing and Communication

**arXiv ID:** 2606.11351 | [PDF](https://arxiv.org/pdf/2606.11351v1)

**作者:** Bodhibrata Mukhopadhyay `[一作]` (Indian Institute of Technology Roorkee), Mohamed-Slim Alouini `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于McCormick松弛的MJSAC波形设计方法，用于在联合感知与通信系统中生成多组相同波束图的协方差矩阵，以实现多符号通信。

**💡 创新点**

创新点在于：①通过最大化协方差矩阵之间的Frobenius距离来显著提升符号辨识性能；②采用McCormick松弛将原始非凸问题转化为半正定规划（SDP），实现高效求解；③无需发射端信道信息，支持离线符号生成。

**🔧 技术方法**

使用的技术包括：McCormick松弛、半正定规划（SDP）求解器、CVX工具箱、Hadamard矩阵构造波形、仿真评估（SER、波束图可视化）。

**📊 数据集**

使用仿真数据，设置不同天线数（N=8、12、16）和两种波束图（全向、定向），并通过仿真生成协方差矩阵与符号集合进行评估。

**📈 对比分析**

比较方法：与BPSK、Sana-Omni/Direc、Fan-Omni/Direc三种已有方案在相同信道、功率和符号长度下进行SER对比。实验结果显示，MJSAC在所有天线数下均优于BPSK和Sana方案，且随着N增大性能优势进一步扩大；在N=12、16时MJSAC还能超过Fan-Omni方案。

**⚠️ 局限性**

局限性：①算法复杂度较高，尤其随天线数增加呈N^6–N^7级；②设计为离线完成，缺乏对实时信道变化的自适应；③实验仅基于仿真，缺乏真实硬件验证。

---

## 104. APEX: Automated Prompt Engineering eXpert with Dynamic Data Selection

**arXiv ID:** 2606.11459 | [PDF](https://arxiv.org/pdf/2606.11459v1)

**作者:** Fei Wang `[一作]` (Google), Inderjit S. Dhillon `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种数据驱动的自动提示优化框架APEX，利用动态数据分层提升大语言模型提示搜索的效率和效果

**💡 创新点**

创新点在于把数据视为可动态演化的信号，通过将数据分为易、难、混合三层并聚焦混合层的“可解决前沿”和“排名敏感前沿”，实现对提示生成和评估的高信号放大

**🔧 技术方法**

采用遗传/进化算法结合LLM生成的提示变异，配合基于历史表现的动态分层、轨迹引导的变异与权重化的选择机制

**📊 数据集**

使用IFBench、SimpleQA Verified以及FACTS Grounding三大多样化基准数据集进行评估

**📈 对比分析**

与APO和GEPA等基线在固定5000次评估调用下对比，APEX在Gemini 2.5 Flash上平均提升11.2%、在Gemma 3 27B提升6.8%，显著优于对手

**⚠️ 局限性**

局限性包括：对历史窗口长度的敏感性、仅在黑盒API设置下验证，且在极度难题或知识缺失场景下提升空间有限

---

## 105. Towards a Joint Understanding of Remote Operation for Vehicles in Public Road Traffic

**arXiv ID:** 2606.11336 | [PDF](https://arxiv.org/pdf/2606.11336v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 106. Detecting AI-Generated Content on Social Media with Multi-modal Language Models

**arXiv ID:** 2606.11200 | [PDF](https://arxiv.org/pdf/2606.11200v1)

**作者:** Chenyang Yang `[一作]` (Carnegie Mellon University), Xuewen Zhang `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套统一框架，持续收集多模态社交媒体数据，并训练一个紧凑的视觉‑语言模型，实现对 AI 生成内容（AIGC）的检测与可解释生成；

**💡 创新点**

创新点在于：①构建了持续更新的数据收集与增量训练管线，解决了模型对新生成器的泛化缺失；②采用多模态视觉‑语言模型，融合图像、文本及社交上下文信息；③在训练时随机屏蔽文本，迫使模型关注视觉特征，并在训练中同时学习解释生成，提升可解释性；

**🔧 技术方法**

技术上使用了 LLaVA‑style 轻量化视觉‑语言模型（Vision Encoder 为 Perception Encoder，语言模型为小型 LLM），并通过三阶段后训练（图像标注→多任务微调→AIGC 检测+解释）完成模型优化；

**📊 数据集**

数据集包括：60M+ Shutterstock 图像标注、10M+ 社交媒体图像/视频 + 2.2M 视频、164K 手工标注高质量图像、130K 多任务标注样本，以及 200K AIGC 微调样本；在公开基准上使用 FakeClue、LOKI、Chameleon 等；

**📈 对比分析**

与七个零样本 VLM、三种开源 VLM（Qwen2.5‑VL‑3b、Gemma3‑4b、Llama3.2‑11b）及 SOTA FakeVLM 进行对比；在 FakeClue 上达到 0.986 的准确率，在 LOKI 上 0.839，远超零样本模型（<0.6），在 Chameleon 上 86% 准确率，显著优于所有基线；

**⚠️ 局限性**

局限性包括：①持续收集仍为被动，可能滞后于新生成器或对抗攻击；②尽管模型紧凑，但大规模部署仍有显著计算成本；③生成的解释并非完全因果可信，可能带有模型偏差或过度自信。

---

## 107. Phi-Actor-Critic: Steering General-Sum Games to Pareto-Efficient Correlated Equilibria

**arXiv ID:** 2606.11284 | [PDF](https://arxiv.org/pdf/2606.11284v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 108. NightFeats @ MMU-RAGent NeurIPS 2025: A Context-Optimized Multi-Agent RAG System for the Text-to-Text Track

**arXiv ID:** 2606.11199 | [PDF](https://arxiv.org/pdf/2606.11199v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 109. The Dynamics of Human and AI-Generated Language: How Semantics Fluctuates across Different Timescales

**arXiv ID:** 2606.11371 | [PDF](https://arxiv.org/pdf/2606.11371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 110. A Geometric Profile of Semantic Information in Text: Frame-Conditional Uniqueness and a Trade-Off Triangle for Scalar Summaries

**arXiv ID:** 2606.11222 | [PDF](https://arxiv.org/pdf/2606.11222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 111. From Simulation to Real-World: An In-Field 6D Pose Dataset and Baseline for Robotic Strawberry Harvesting

**arXiv ID:** 2606.11381 | [PDF](https://arxiv.org/pdf/2606.11381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 112. Beyond Compaction: Structured Context Eviction for Long-Horizon Agents

**arXiv ID:** 2606.11213 | [PDF](https://arxiv.org/pdf/2606.11213v1)

**作者:** Andrew Semenov `[一作]` (Kiz8), Svyatoslav Dorofeev `[通讯]` (Kiz8)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Context Window Lifecycle（CWL）方案，使长周期LLM代理在有限的上下文窗口内保持无限的工作窗口，利用探究与行动阶段的注释、依赖图和渐进式、无LLM的优先级驱动弹出策略来管理上下文；

**💡 创新点**

创新点在于：①将上下文拆分为可标注的Typed Episodes（探索/行动）并构成显式的依赖图；②采用可观测的、可控制的离线优先级弹出政策，完全不调用LLM，避免了总结压缩的不可预测性、结构破坏和幻觉；③在保持用户输入不被删除的前提下，按因果依赖而非时间顺序进行有意义的弹出；

**🔧 技术方法**

核心技术包括：单一注释工具`cwl_annotate`、在运行时维护的Typed DAG、token accounting循环、按层级（先去除推理轨迹、再大块输出、再中间产物、最后完整删节）弹出以及对用户回合和Prologue的保护；

**📊 数据集**

在四个基准上评测：Terminal Bench 2.0、SWE Bench Lite、Recovery Bench、LongCLI Bench；同时做了针对真实开源仓库（Excalidraw、Redis、Linux Kernel）的跨任务案例研究；

**📈 对比分析**

与传统的单任务隔离会话基线相比，CWL在单一80k‑token预算的长序列实验中任务准确率保持在±3 %范围内（在Terminal Bench上89个任务准确率68.25 % vs 68.40 %），Token消耗与成本基本相同，且在需要长上下文时实现了约20–70 % 的推理成本降低；

**⚠️ 局限性**

局限性包括：①注释负担需人工/模型遵循；②依赖粒度仅限于完整探究Episode，无法细粒度指向单个工具调用；③假设线性回合，无法处理多分支或并行子任务；④对KV缓存的频繁失效在高频弹出时可能导致推理成本上升；⑤对模型推理行为的潜在干扰（可能导致过度探索或冗余调用）。

---

## 113. Steering Multirobot Behavior via Closed-Loop Affine Activation Editing

**arXiv ID:** 2606.11489 | [PDF](https://arxiv.org/pdf/2606.11489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 114. PermDoRA -- Understanding Adapter Interference in Language Models: Limits of Parameter-Space Geometry

**arXiv ID:** 2606.11262 | [PDF](https://arxiv.org/pdf/2606.11262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 115. Few-Shot Resampling for Scalable Statistically-Sound Data Mining

**arXiv ID:** 2606.11235 | [PDF](https://arxiv.org/pdf/2606.11235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 116. Calibration Drift Under Reasoning: How Chain-of-Thought Budgets Induce Overconfidence in Large Language Models

**arXiv ID:** 2606.11211 | [PDF](https://arxiv.org/pdf/2606.11211v1)

**作者:** Prakul Sunil Hiremath `[一作]` (Visvesvaraya Technological University), Harshit R. Hiremath `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨并证明在大型语言模型中，增加链式推理（CoT）预算时校准误差可能出现非单调下降、先升后降的现象（Calibration Drift Under Reasoning）。

**💡 创新点**

提出“校准漂移”概念并给出正式定义，基于假设锁定模型（Hypothesis Lock-In）解释其机制，设计基于置信度与辅助准确率差距的停止规则（CABStop）以避免过度自信。

**🔧 技术方法**

采用链式推理、显式置信度提问、Expected Calibration Error (ECE)、过度自信间隙 (OG)、自一致性等技术，对不同推理预算下模型行为进行定量评估。

**📊 数据集**

使用包含 47 条专门设计的推理陷阱问题（21 种认知错误类型）的自构数据集，对 Llama-3.1‑8B 和 Llama-3.3‑70B 进行实验，评估 4 个推理预算（none、light、medium、heavy）及 3 个种子。

**📈 对比分析**

与基准模型（无推理预算）和单一预算下的性能对比；在 8B 模型中观察到 ECE 在 light 推理时最高、heavy 推理时最低，且过度自信间隙始终为正；在 70B 模型仅得到无推理预算结果，未能验证非单调性。

**⚠️ 局限性**

局限包括样本量小（47 条题目，≈ 574 条有效响应）、42% 有效率导致可能的选择偏差、置信度仅为显式口头化、仅评估 Llama 系列模型、推理预算通过提示工程实现且未做提示 ablation、缺乏多规模完整评估，导致结论主要针对 8B 模型而非普遍适用。

---

## 117. FlowBank: Query-Adaptive Agentic Workflows Optimization through Precompute-and-Reuse

**arXiv ID:** 2606.11290 | [PDF](https://arxiv.org/pdf/2606.11290v1)

**作者:** Lingzhi Yuan `[一作]`, Furong Huang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出一种三阶段的基于资产组合的代理工作流优化框架，旨在通过离线多样化搜索、覆盖率驱动的精简和图神经网络匹配，在推理时为每个查询挑选最优工作流。

**💡 创新点**

创新点在于把传统单一工作流的设计转变为可重用的互补工作流集合，利用多模态的多样化搜索、子集化压缩以及基于查询-工作流二分图的边价值预测，实现高效的查询级自适应而不额外增加推理成本。

**🔧 技术方法**

核心技术包括：使用 MCTS + LLM（DiverseFlow）进行多样化工作流搜索；基于覆盖率的组合搜索与相关性约束的 CuraFlow 进行精简；构建查询-工作流二分图并用轻量级 GNN 进行边价值预测和查询级匹配。

**📊 数据集**

实验在五个公开基准上验证：数学推理（MATH、AMC）、代码生成（MBPP）、问答（MMLU Pro）以及阅读理解（DROP），利用 Qwen3-8B 作为搜索 LLM，GPT-4o mini 作为执行 LLM。

**📈 对比分析**

相较于手工设计的工作流（如 Chain-of-Thought、Self-Consistency 等）和自动化搜索方法（如 GPTSwarm、ADAS、AFlow 等），该框架在所有五个基准上均位居第一，平均性能 73.40 分，较最强自动化基线提升 4.26% 绝对分，较最强手工基线提升 14.92% 绝对分；同时在性能-成本 Pareto 前沿表现优异，平均推理成本仅 1.65。

**⚠️ 局限性**

局限性包括：对大型工作流集合的可扩展性尚未充分评估；依赖 LLM 的生成与匹配性能，可能受限于 LLM 的规模与成本；以及在极其多样化的查询分布下，边价值预测的泛化性需要进一步验证。

---

## 118. OmniLoc: A Geometry-Aware Foundation Model for Anchor-Free UE Localization Across Diverse Indoor Environments

**arXiv ID:** 2606.11490 | [PDF](https://arxiv.org/pdf/2606.11490v1)

**作者:** Lei Chu `[一作]`, Andreas F. Molisch `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 OmniLoc，一个面向多环境的无锚点室内定位基础模型，结合统一标记化、几何感知 Transformer 和几何一致性回归实现高精度定位。

**💡 创新点**

创新点在于将异构无线测量统一嵌入、设计 AP 关注的几何感知 Transformer，以及基于几何嵌入的定位头，三者协同实现跨建筑、跨楼层的鲁棒性。

**🔧 技术方法**

采用 Transformer、统一 tokenization、MPAA 多头聚合、GLO 几何感知回归，以及 PEFT（LoRA、LP）等技术。

**📊 数据集**

使用 USC Campus WiLoc 大规模 16栋 4,560 m 轨迹数据，以及公开的 WILD 数据集进行跨环境验证。

**📈 对比分析**

与 CNN、BERT、LWM 等基线在同环境下对比，OmniLoc 在 MLE、RMSE、P90 等指标上平均提升约 30% 以上；在跨楼层、跨建筑、跨数据集时表现出更好的迁移性能，PEFT 方案在少样本时显著降低误差。

**⚠️ 局限性**

局限性包括仅使用 CSI 幅值而未利用相位信息；几何先验仅为楼层/建筑级别，缺乏细粒度环境上下文；对多模态（IMU、LiDAR 等）集成仍待研究。

---

## 119. SOMA-SQL: Resolving Multi-Source Ambiguity in NL-to-SQL via Synthetic Log and Execution Probing

**arXiv ID:** 2606.11424 | [PDF](https://arxiv.org/pdf/2606.11424v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 120. When More Documents Hurt RAG: Mitigating Vector Search Dilution with Domain-Scoped, Model-Agnostic Retrieval

**arXiv ID:** 2606.11350 | [PDF](https://arxiv.org/pdf/2606.11350v1)

**作者:** Nabaraj Subedi `[一作]` (University of Wyoming), Shivanand Venkanna Sheshappanavar `[通讯]` (University of Wyoming)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在大规模异质文档集合中检索增强生成（RAG）遇到的向量搜索稀释问题，并提出MASDR-RAG框架通过领域范围检索和混合路由解决该问题；

**💡 创新点**

创新点在于首次将组织元数据作为检索范围以减少稀释，并揭示多代理协同会导致精度-可信度悖论；

**🔧 技术方法**

采用密集+稀疏混合检索、Neo4j HNSW索引、正则+LLM混合路由、单次合成调用以及多代理协调；

**📊 数据集**

实验数据集涵盖Wyoming DOT 1,128份文档、Caltrans、CDOT等州交通局数据以及Composite-9、HotpotQA-distractor、MultiHop-RAG、NQ-Open、FinanceBench和MMLU-Pro等公开语料；

**📈 对比分析**

与单体检索、ReAct、MA-RAG、SCOUT-RAG等基线对比，MASDR-RAG在P@10、精确度、可信度上提升明显（如Wyoming DOC从0.77提升至0.86，整体正确率提升至≈35%+），但多代理版本在商业后端会导致可信度骤降；

**⚠️ 局限性**

局限性包括仅在7–8B参数模型上验证、依赖显式组织元数据、LLM-评判可能高估可信度下降、未覆盖70B以上模型以及多代理架构对不同后端的依赖性不充分。

---

## 121. MPC-Patch-Bench: Security-Aware LLM Code Patch for Multi-Party Computation

**arXiv ID:** 2606.11416 | [PDF](https://arxiv.org/pdf/2606.11416v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 122. SwiftCTS: Fast Cross-Design Prediction and Pareto Optimization of Clock Tree Metrics via Few-Shot Calibration

**arXiv ID:** 2606.11348 | [PDF](https://arxiv.org/pdf/2606.11348v1)

**作者:** Barsat Khadka `[一作]` (University of Southern Mississippi), Md Rubel Ahmed `[通讯]` (Louisiana Tech University)

**通讯引用:** 1906 | [OpenAlex ID](https://openalex.org/A5058370129)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了SwiftCTS，一个基于物理知识的梯度提升回归模型，用于快速预测时钟树合成（CTS）的功耗、线长和时序偏移，并与进化搜索结合实现高效的设计空间探索。

**💡 创新点**

创新在于将物理可解释的统计特征与XGBoost/LightGBM结合，并引入K-shot乘法校准机制，无需重训练即可在未知宏上实现低误差；同时实现了子毫秒推理和毫秒级训练。

**🔧 技术方法**

使用梯度提升决策树（XGBoost、LightGBM）、几何归一化特征、K-shot几何均值校准、NSGA-II多目标进化优化以及OpenROAD物理验证。

**📊 数据集**

基于Sky130 PDK和OpenROAD生成的5,520个CTS评估样本，涵盖AES、PicoRV32、SHA-256、ETHmac等四个训练架构，以及JPEG Encoder、ZipDiv Core等两种OOD宏。

**📈 对比分析**

与GAN-CTS、CNN等深度模型及随机/Sobol搜索对比，SwiftCTS训练时间仅4.1秒，推理子毫秒；在OOD测试中K=1校准后功耗误差降至3.3%，线长误差1%以下，时序误差0.5%以内，且在10秒内完成10万配置的NSGA-II搜索。

**⚠️ 局限性**

局限在缺少工业级特征（如dont-touch等）、仅在Sky130/OpenROAD验证，且对高阶技术节点、多Vt、非均匀电源网格的鲁棒性尚未验证。

---

## 123. LatticeBridge: Rare-Event Sequential Inference for Faithful Structured Sequence Synthesis

**arXiv ID:** 2606.11203 | [PDF](https://arxiv.org/pdf/2606.11203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 124. APTLAS: An Indexed APT Literature Repository

**arXiv ID:** 2606.11241 | [PDF](https://arxiv.org/pdf/2606.11241v1)

**作者:** Bavley Guerguis `[一作]` (McMaster University), Nabil Bassim `[通讯]` (McMaster University)

**通讯引用:** 2968 | [OpenAlex ID](https://openalex.org/A5030578393)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

建立了一个APT文献索引数据库APTLAS，并提供了基于Web的浏览和筛选工具。

**💡 创新点**

创新点在于使用LLM自动抽取结构化元数据，并以多维度（仪器、材料、分析条件等）组织索引，解决了传统检索难以针对领域特定元数据过滤的问题。

**🔧 技术方法**

采用Llama 3.1模型（通过Ollama服务）进行结构化输出的元数据提取，并利用CrossRef API进行文献检索。

**📊 数据集**

使用了近2,300条从CrossRef检索到的APT相关文献作为数据库数据集。

**📈 对比分析**

与传统检索方法相比，APTLAS能够按材料系统、仪器型号、分析模式等多维度精准筛选，用户反馈显示检索效率显著提升，虽然未给出定量性能指标，但工具在可用性和检索速度上表现良好。

**⚠️ 局限性**

主要局限包括LLM提取过程中的错误率（尤其是实验参数）、分类主观性与遗漏、数据库需要定期更新以及缺乏更细粒度的提取目标等。

---

## 125. TileFuse: A Fused Mixed-Precision Kernel Library for Efficient Quantized LLM Inference on AMD NPUs

**arXiv ID:** 2606.11357 | [PDF](https://arxiv.org/pdf/2606.11357v1)

**作者:** Wesley Pang `[一作]` (University of Illinois Urbana-Champaign), Deming Chen `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 TileFuse，一套适用于 AMD XDNA2 NPU 的混合精度 kernel 库，直接支持 AWQ 风格 W4A16/W8A16 量化的 LLM 线性层，完成权重量化预排列、融合拆包/去量化以及 GEMM/GEMV 微核的实现。

**💡 创新点**

通过预划分权重布局与元数据打包、交错列优先预拼接、全流程融合拆包/去量化以及利用 AIE 数组的 4×8 计算结构实现 GEMV 数据流重设计，从而在 XDNA2 上实现 32K 维度大矩阵的高效推理。

**🔧 技术方法**

采用 MLIR‑AIE (IRON) 接口进行低层编程，使用 VLIW 指令、局部缓冲区和 DMA 预拼接，实现 INT4/INT8 拆包、BF16 去量化与 GEMM/GEMV 的统一流水线；并在数据流层面利用内存核做权重分发。

**📊 数据集**

在三种主流 LLM 模型上评测：Gemma‑2B、Qwen‑2.5‑3B 与 Llama‑3‑8B（均使用 AWQ W4A16/W8A16 预量化权重）。

**📈 对比分析**

与 AMD Ryzen AI 的 iGPU 上的量化 HIP/GPU 基线及 CPU/标准 GEMM 进行对比，结果显示 TileFuse 在大规模 GEMM 的 prefilling 阶段可实现最高 2.0× 的延迟加速、281% 的 GEMV 加速，并在 Ryzen AI 7‑350 平台下节能 64.6%。

**⚠️ 局限性**

NPU 的内核调度与配置开销在小规模 GEMV 生成阶段显著，导致与 iGPU 的 latency 竞争不足；且当前实现仅覆盖线性层，非线性或 Softmax 等仍依赖 iGPU，需进一步优化或采用混合 NPU‑iGPU 调度策略。

---

## 126. A Zero-Shot Multi-Agent Framework for Human-Building Interaction via Programmatic Reasoning

**arXiv ID:** 2606.11354 | [PDF](https://arxiv.org/pdf/2606.11354v1)

**作者:** Yuqi Wang `[一作]` (Nantum AI), Ali Mehmani `[通讯]` (Nantum AI)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个零射击、多代理LLM框架，用“Doorman”任务拆分与专属代理（编码、世界知识、专家）协同，完成人‑建筑交互中的自然语言理解、数据检索、代码生成和文档检索；

**💡 创新点**

创新点在于将任务拆分与可执行Python脚本生成相结合，使用RAG与多代理协同避免传统RAG、单模型的局限，且不依赖大规模领域微调，提升了领域适配与技术准确性；

**🔧 技术方法**

采用的技术包括大型语言模型（GPT‑4o、GPT‑3.5、Claude 3.5 等）、多代理框架、RAG、代码解释器、实时数据库接口、Streamlit UI 与 Chainlit 对话管理；

**📊 数据集**

使用了来自 Nantum AI 平台的 200 多栋商业建筑的实时运营数据、建筑文档、代码库和技术文档数据库；

**📈 对比分析**

通过与单一 LLM 及单代理 ReAct 基线对比，量化响应时间、数值准确率、数据检索率和文档检索率，所有指标均达到 100% 或接近，平均响应时长约 20‑30 秒，显著优于传统方法；

**⚠️ 局限性**

局限包括无法处理递归或多步骤任务、对未清洗结构化数据敏感、仍可能出现误报或错误答案、未实现直接 BMS 控制以及对长轮次复杂对话的支持不足。

---

## 127. To Intervene or Not: Guiding Inference-time Alignment with Probabilistic Model Blending

**arXiv ID:** 2606.11201 | [PDF](https://arxiv.org/pdf/2606.11201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 128. AI Coding Agents Can Reproduce Social Science Findings

**arXiv ID:** 2606.11447 | [PDF](https://arxiv.org/pdf/2606.11447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 129. Preregistration for Experiments with AI Agents

**arXiv ID:** 2606.11217 | [PDF](https://arxiv.org/pdf/2606.11217v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 130. From Explicit Elements to Implicit Intent: A Predefined Library for Auditable Behavioral Inference

**arXiv ID:** 2606.11207 | [PDF](https://arxiv.org/pdf/2606.11207v1)

**作者:** Liu hung ming `[一作]` `[通讯]` (PARRAWA AI), Liu hung ming (PARRAWA AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了SemantiClean模块化语义推理框架，并实现了两阶段LLM驱动的推理引擎；

**💡 创新点**

创新点在于以可审计、结构治理为核心，采用反通胀机制与分层语义元素，强调可复现性（σ=0）并将LLM作为辅助推理器而非全流程黑盒；

**🔧 技术方法**

使用的技术包括JSON驱动的规则引擎、分层语义元素库、三重反通胀机制（冗余组上限、分层惩罚、适应性约束）、两阶段LLM推理流程（轻量级元素选取+深度语义分析）以及Python聚合校准；

**📊 数据集**

采用UCI OSPI电子商务会话数据集（12,330个会话，18个特征，无缺失）；

**📈 对比分析**

与传统端到端机器学习基线（如XGBoost、SVC、LSTM、GNN）对比，Deterministic引擎在完整字段下表现略低；在遮蔽Revenue字段的掩码实验中，LLM集成引擎准确率仅56.4%，但保守的低置信度回避策略显著减少错误预测；

**⚠️ 局限性**

限制包括样本量不足（仅39次评估）、置信度与准确度不匹配、单一LLM模型与温度敏感性未评估、流量来源与页面值缺失导致的覆盖不足以及模型在真实业务环境中的可扩展性与鲁棒性待验证。

---

## 131. Context-Aware Multimodal Claim Verification in Spoken Dialogues

**arXiv ID:** 2606.11420 | [PDF](https://arxiv.org/pdf/2606.11420v1)

**作者:** Chaewan Chun `[一作]` (Pennsylvania State University), Dongwon Lee `[通讯]` (Pennsylvania State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MAD2 这一全新多轮音频对话验证基准，并提出了基于校准的条件融合框架，用于在音频与文本两种模态上进行语音主张真实性判定。

**💡 创新点**

创新点在于：① 通过高保真 TTS（MoonCast）与 WhisperX 语音对齐，提供精细到句子级的声学与文本双模态标注；② 采用校准的条件融合策略，能够在文本信心不足时有选择地利用音频信号；③ 系统化评估不同对话上下文窗口对验证性能的影响，揭示对话结构对真实性判定的关键作用。

**🔧 技术方法**

核心技术包括：MoonCast 语音合成、WhisperX ASR 及时间戳对齐、RoBERTa‑base 文本编码器、WavLM‑base 语音编码器、基于声学注意力的投影池化、Platt 归一化、四种融合策略（加权、召回提升、音频覆盖、条件 α）。

**📊 数据集**

使用的数据集为 MAD2：1,000 条两说话者英语对话（共 10 小时音频，8,192 句子，3,368 条可验证主张），每条主张附带真实/伪造标签、情景类型、传播风格及音频时间段。

**📈 对比分析**

在多种上下文配置（仅主张、前置句子、双侧窗口、完整对话）下对音频、文本与融合模型进行对照实验。结果显示，完整对话时融合模型的 AUC 达到 0.852，文本模型 0.841，音频模型 0.780；实时（仅前置）性能与离线几乎持平，说明前置上下文即可满足大多数验证需求。

**⚠️ 局限性**

局限性包括：① 基于合成语音的对话，无法完全覆盖真实播客中的多样性、噪声与说话者重叠；② 仅包含两说话者的英语对话，缺乏多说话者、多语言及非英语环境；③ 未分离出哪些具体声学特征在提升验证性能中起主导作用，待进一步深入分析。

---

## 132. Signed Compression Progress on a Sealed Audit is Goodhart-Resistant

**arXiv ID:** 2606.11417 | [PDF](https://arxiv.org/pdf/2606.11417v1)

**作者:** Ayush Mittal `[一作]`, Dhruv Gupta `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了基于sealed audit的signed compression progress作为Intrinsic motivation信号，并证明其Goodhart‑resistant特性。

**💡 创新点**

创新点在于给出端点计量公式和有限panel的2Δ_n误差预算，并用Lean 4 mechanization验证。

**🔧 技术方法**

技术包括固定audit分布、signed loss进展、telescoping身份、Uniform Deviation事件、entropy floor以及EXP3 scheduler。

**📊 数据集**

使用ARC‑TGI grid transformation generators（ARC‑Mini、ARC‑AGI‑1/2 等）作为 synthetic 数据集。

**📈 对比分析**

实验对比预测误差、RND、ICM 等奖励信号，audit‑CP 在主动细胞准确率与噪声分配上优于非oracle基线；在可重用 panel 的 scalar‑feedback 攻击下，audit‑CP 仍保持在 2Δ_n 阈值以内。

**⚠️ 局限性**

局限在于需保持 audit sealed、模型类有效容量有限、audit 不能被修改；在高容量可重用 panel、clip 或 stream‑scoring 等场景下失效。

---

## 133. Maximum Coverage Chase Decoder for Optical Interconnects

**arXiv ID:** 2606.11401 | [PDF](https://arxiv.org/pdf/2606.11401v1)

**作者:** Alessandro Cardinale `[一作]` (EPFL), Yifei Shen `[通讯]` (EPFL)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种针对光互连系统的低复杂度Chase解码器，采用最大覆盖（GMC）方法选取测试错误模式（TEP），并在RS‑BCH及oFEC编码方案中实现。

**💡 创新点**

创新点在于将TEP选择建模为通用最大覆盖问题，将覆盖收益与TEP自身的概率和覆盖范围结合，显著减少必要的TEP数量而不降低性能。

**🔧 技术方法**

使用了Chase-II、LW‑Chase以及GMC‑Chase算法，并通过贪心算法求解GMC问题；在PAM‑4与16‑QAM信号下进行模拟与联合界限分析。

**📊 数据集**

实验数据来源于标准的RS(544,514,15)与eBCH(142,125,2)链路以及32×eBCH(256,239,2)oFEC系统，采用4‑PAM与16‑QAM多层调制方案。

**📈 对比分析**

与Chase‑Pyndiah、LW‑Chase‑LUT等基准方法对比，GMC‑Chase在保持相同误码率性能的同时，RS‑BCH链路上将TEP数减少25%，oFEC链路上将TEP数减少61.3%，从而降低解码复杂度。

**⚠️ 局限性**

限制主要体现在对固定的η和δ参数的依赖，GMC求解虽然是贪心近似，但在更大码字或更高错误修正能力的场景下可能需要进一步优化参数和算法。

---

## 134. Recursive Binding on a Budget: Subspace Carving in Order-p Tensor Memories

**arXiv ID:** 2606.11391 | [PDF](https://arxiv.org/pdf/2606.11391v1)

**作者:** Travis Pence `[一作]` (University of Wisconsin-Madison), Vikas Singh `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的张量记忆架构 Orthogonal Subspace Carving (OSC)，通过将填充向量投影到角色子空间的正交补上，实现深层递归绑定并支持高超位置叠加。

**💡 创新点**

核心创新在于：1）使用投影裁剪子空间，保证绑定结构在不同上下文中的几何正交；2）分离填充向量维度与记忆容量，实现填充词汇量子线性增长；3）在 Clifford 代数框架下给出 OSC 的数学描述。

**🔧 技术方法**

主要技术包括张量乘积绑定、正交投影（子空间裁剪）、向量符号架构（VSA）与张量产品表示（TPR）的比较、基于哈希的程序化上下文生成以及在梯度训练中与神经网络的整合。

**📊 数据集**

实验数据集主要是：synthetic 记忆容量与超位置实验；以及极端多标签分类（XML）基准数据集（如 Amazon-670K、Eur-Lex 等）。

**📈 对比分析**

与 14 种主流 VSA（HRR、HLB、VTB 等）对比，OSC 在同等记忆容量下实现 99% 以上检索准确率，且参数占用量比最优 VSA 低 1–3 个数量级；在 XML 任务上性能与 HLB、VTB 等接近，且内存占用更低。

**⚠️ 局限性**

主要限制：无法实现精确代数解绑，只能通过词汇表搜索获得填充；对高度全局正交性要求较低，可能在跨上下文冲突时出现噪声；实际实现中需合理设计投影矩阵以避免数值不稳定。

---

## 135. Mahalanobis-Guided Latent OOD Detection for Hybrid ES-DRL Control in Time-Varying Systems

**arXiv ID:** 2606.11474 | [PDF](https://arxiv.org/pdf/2606.11474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 136. Evaluating and Combating the Impact of Concept Drift on the Performance of Machine Learning-Based Phishing Detection Systems

**arXiv ID:** 2606.11471 | [PDF](https://arxiv.org/pdf/2606.11471v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 137. 3D-CBM: A Framework for Concept-Based Interpretability in Generative 3D Modeling

**arXiv ID:** 2606.11446 | [PDF](https://arxiv.org/pdf/2606.11446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 138. Accurate and Resource-Efficient Federated Continual Learning

**arXiv ID:** 2606.11480 | [PDF](https://arxiv.org/pdf/2606.11480v1)

**作者:** Jebacyril Arockiaraj `[一作]` (University of Southern California), Viktor Prasanna `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

FedRAN 提出一种资源感知的分析式联邦持续学习框架，利用固定预训练特征与随机特征压缩统计来一次性更新分类器。

**💡 创新点**

创新点在于：① 用截断 SVD 压缩随机特征 Gram 矩阵并采用两层 QR‑SVD 合并，实现一次性统计上传；② 通过原型伪标签支持半监督学习，避免了传统梯度迭代与重放开销。

**🔧 技术方法**

技术包括：固定预训练网络、共享随机投影+ReLU、截断 SVD、QR‑SVD 子空间合并、闭式岭回归、以及基于原型的伪标签生成。

**📊 数据集**

实验数据集为 CIFAR‑100、ImageNet‑R 与 VTAB 三个视觉持续学习基准。

**📈 对比分析**

与现有联邦持续学习与梯度基线相比，FedRAN 在平均准确率上提升了多达 4.8 个百分点，通信量降低 30–120 倍，训练速度提升约 190 倍；在仅 20% 标签率下伪标签可额外提升 6.6 个百分点。

**⚠️ 局限性**

局限性包括：需依赖预训练模型且固定特征，随机特征维度与 SVD 秩需手动调参；伪标签可能引入噪声；缺乏正式的隐私保护机制。

---

## 139. A Modular Dual-Camera Pipeline for Micro-Inspection Using Aerial Robots

**arXiv ID:** 2606.11419 | [PDF](https://arxiv.org/pdf/2606.11419v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 140. Risk Under Pressure: Compute-Aware Evaluation of Adversarial Robustness in Language Models

**arXiv ID:** 2606.11409 | [PDF](https://arxiv.org/pdf/2606.11409v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 141. Dynamic Execution Horizon Prediction for Chunk-based Robot Policies

**arXiv ID:** 2606.11408 | [PDF](https://arxiv.org/pdf/2606.11408v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 142. Agent Skill Evaluation and Evolution: Frameworks and Benchmarks

**arXiv ID:** 2606.11435 | [PDF](https://arxiv.org/pdf/2606.11435v1)

**作者:** Kexin Ding `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统综述了智能体技能演化与评估方法，提出四类演化范式并分析六大基准类型；

**💡 创新点**

首次构建四范式演化分类体系，梳理基准缺口与评估维度，为后续研究提供框架；

**🔧 技术方法**

文献综述、对比分析、归纳总结等方法；

**📊 数据集**

引用了SkillsBench、SkillTester、SkillRouter、SWE-Skills-Bench等公开基准数据；

**📈 对比分析**

通过对比六大基准的覆盖范围、指标与评价方式，总结出目前方法在实用性、安全性与多模态支持方面的差距；

**⚠️ 局限性**

仅为综述性工作，缺乏实验验证；对新近出现的方法和大规模真实环境评估覆盖不足；

---

## 143. Querying Cohesive Subgraph regarding Span-Constrained Triangles on Temporal Graphs with Dynamic Index Maintenance

**arXiv ID:** 2606.11582 | [PDF](https://arxiv.org/pdf/2606.11582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 144. VIPIR: A Versatile GPU Framework for Integrating Private Information Retrieval Protocols

**arXiv ID:** 2606.11536 | [PDF](https://arxiv.org/pdf/2606.11536v1)

**作者:** Jongmin Kim `[一作]` (Seoul National University), Jung Ho Ahn `[通讯]` (Seoul National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了VIPIR，一个针对GPU的PIR框架，设计并实现了双包装与混合两种新协议，并引入了ExpPack扩展型环打包方法；

**💡 创新点**

创新点包括统一分析模型、通过双包装和混合设计消除单一HE类别的瓶颈、ExpPack显著降低通信并开启GPU级并行性、利用Tensor Core加速GEMM、NTT与内存调度优化以及多GPU扩展；

**🔧 技术方法**

使用了同态加密（标量HE与多项式HE）、环打包、扩展型环打包(ExpPack)、数论变换(FFT/NTT)、Tensor Core INT8‑INT32 GEMM、懒化模运算、矩阵布局转换、CUDA流调度、NCCL多GPU通信等技术；

**📊 数据集**

实验以规模为1GiB、4GiB、16GiB和64GiB的合成数据库为主，数据库结构为N×2^k（N=2^12），共1B记录；

**📈 对比分析**

与CPU基线（如OnionPIRv2、YPIR）和GPU基线（PIRonGPU、ShiftPIR）对比，VIPIR实现了10-1000倍加速、100-700倍QPS提升、每查询仅96KiB通信，并在64GiB上接近理论延迟下限；

**⚠️ 局限性**

局限性在于对极大数据库仍受GPU内存限制、部分协议通信量仍高、仅针对单服务器PIR、需要高端GPU硬件且参数选择对安全性有影响。

---

## 145. WHET: Welding Homomorphic Encryption to Accelerator Architectures

**arXiv ID:** 2606.11541 | [PDF](https://arxiv.org/pdf/2606.11541v1)

**作者:** Jongmin Kim `[一作]` (Seoul National University), Jung Ho Ahn `[通讯]` (Seoul National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 WHET，一种面向 FHE 加速器的内存中心化、架构感知优化方案；

**💡 创新点**

核心创新包括细粒度 CtS（fg-CtS）、语义压缩的 CtS/plaintext 压缩和中间 ModRaise，以及轻量级的 KeyMult 缓存和 EWE 指令扩展；

**🔧 技术方法**

利用了 CKKS 加密体系结构、RNS 计数器、细化 CtS 矩阵分解、压缩编码、硬件专用缓冲区、指令融合与调度优化等技术；

**📊 数据集**

评估基准主要为 CNN 推理（ResNet-20、MobileNet、ResNet-18、VGG-16）、排序网络和 HELR 训练任务，使用 CIFAR-10、Tiny ImageNet 等公开数据集；

**📈 对比分析**

与现有 ASIC（SHARP、TRINITY、FAST、HAWK 等）以及 GPU（RTX 5090）对比，WHET 在同等功耗/面积下实现 1.38–8.74× 的面积-性能提升，CKKS 预热时间降至 1.9 ms，CNN 推理延迟仅 33–239 ms；

**⚠️ 局限性**

局限在于仍需大量 HBM 以及对大规模 CNN（如 VGG-16）可能受 Scratchpad 容量限制，且对极端大规模数据集的可扩展性尚未完全验证。

---

## 146. AVIS: Adaptive Test-Time Scaling for Vision-Language Models

**arXiv ID:** 2606.11576 | [PDF](https://arxiv.org/pdf/2606.11576v1)

**作者:** Ahmadreza Jeddi `[一作]` (Samsung Electronics), Radek Grzeszczuk `[通讯]` (Samsung Electronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文提出了一种在视觉语言模型推理时自适应地分配视觉上下文与推理搜索的策略AVIS。

**💡 创新点**

创新点在于同时对视觉上下文缩放(VCS)和视觉推理缩放(VRS)进行样本级动态分配，并用训练自由的关键多样性视觉(KDV)剪枝与难度感知自一致性决策实现。

**🔧 技术方法**

采用的技术包括KDV token pruning、基于视觉键多样性的O(N)策略、难度预测器和自一致性(K次投票)推理，配合共享预填充实现低成本。

**📊 数据集**

评测使用Qwen2.5‑VL‑7B在多种图像与视频推理基准，如MathVista、MathVerse、DOCVQA、MME、VideoMME等共12+6任务。

**📈 对比分析**

与Vanilla、单轴VCS/VRS固定配置以及RL训练的VLM进行比较，AVIS在保持或提升准确率的同时，平均降低52%（图像）/66%（视频）FLOPs，且在相同FLOPs下比最佳基线高3.7%。

**⚠️ 局限性**

限制包括难度预测仅基于视觉信息，未使用文本或LLM内部表示，且对极端难度样本的收益仍有限，未来可进一步统一决策和引入文本特征。

---

## 147. XPR: An Extensible Cross-Platform Point-Based Differentiable Renderer

**arXiv ID:** 2606.11529 | [PDF](https://arxiv.org/pdf/2606.11529v1)

**作者:** Steve Rhyner `[一作]` (University of Toronto), Nandita Vijaykumar `[通讯]` (University of Toronto)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 XPR，一个可扩展的跨平台点基可微渲染框架，提供四功能高层接口，使新渲染方法只需数百行 Python 代码即可实现，并可通过 XLA 编译至 GPU、TPU、CPU 等多种加速器。

**💡 创新点**

创新点包括：①将方法特定逻辑与渲染流水线解耦，提供四个单元素接口；②采用静态张量化模块将动态列表、循环等转化为固定尺寸操作；③通过统一的并行抽象（map、scatter、gather、sort）实现跨硬件可移植性；④在单一代码基础上快速原型、复现并在多平台上高性能执行。

**🔧 技术方法**

使用 JAX+XLA 进行跨平台编译；基于静态张量化的分块裁剪、深度排序、Alpha 混合；通过 Map、Scatter、Gather、Sort 等并行构造；实现 3D Gaussian、3DGUT、LinPrim 等原语；对超参数进行自动化搜索以匹配不同硬件。

**📊 数据集**

评估使用了 Mip-NeRF360、Tanks & Temples、Blender 三个数据集，总计 17+8 场景。

**📈 对比分析**

与官方 CUDA 实现比较，PSNR/SSIM/LPIPS 误差低于 0.06；FPS 约 97%‑110% 的 CUDA 速度；在 NVIDIA L40S、AMD MI210、Intel Max1100 GPU 以及 Google TPU 上均可运行，单一 Python 代码即可跨平台；在 TPU 上通过分块裁剪实现可扩展性。

**⚠️ 局限性**

局限性：性能受限于 XLA 后端成熟度，硬件特定优化仍需手工；不支持需要动态控制流的光线追踪/光线漫射等渲染；超参数仍需自动搜索，调优成本存在。

---

## 148. Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching

**arXiv ID:** 2606.11583 | [PDF](https://arxiv.org/pdf/2606.11583v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 149. Distortion-Resilient Robotic Imitation Learning for Autonomous Cable Routing

**arXiv ID:** 2606.11577 | [PDF](https://arxiv.org/pdf/2606.11577v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 150. Understanding Cross-Sensor Feature Variations for Generalizable 3D Perception

**arXiv ID:** 2606.11573 | [PDF](https://arxiv.org/pdf/2606.11573v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 151. APEX: A Network-Native Time-Series Foundation Model for Forecasting and Anomaly Detection for Wireless Edge Operations

**arXiv ID:** 2606.11553 | [PDF](https://arxiv.org/pdf/2606.11553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 152. VL-DINO: Leveraging CLIP Vision-Language Knowledge for Open-Vocabulary Object Detectio

**arXiv ID:** 2606.11546 | [PDF](https://arxiv.org/pdf/2606.11546v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 153. SirenFNO: Efficient and Full Frequency Learning of Fourier Neural Operators

**arXiv ID:** 2606.11518 | [PDF](https://arxiv.org/pdf/2606.11518v1)

**作者:** Pengqing Shi `[一作]` (University of Sydney), Junbin Gao `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于SIREN的全频率 Fourier 神经算子（SirenFNO），通过消除频率截断实现对所有频率模式的学习；

**💡 创新点**

创新点在于将 SIREN 作为超网络生成频域卷积核参数，消除谱偏差并保持网格不变性，同时引入 CP、TT、Tucker 三种功能张量分解进一步压缩参数；

**🔧 技术方法**

采用 SIREN 超网络、频域卷积、功能张量分解与残差结构等技术；

**📊 数据集**

在 Darcy 流、Navier‑Stokes、Burgers、1D/2D CFD、反应扩散等多种 PDE 基准数据集上进行实验；

**📈 对比分析**

与传统 FNO 及其变体（UFNO、TFNO‑CP、AM‑FNO 等）比较，SirenFNO 在保持或提升 L2 误差的同时参数量减少 4–73 倍，且在零步超分辨率任务中仍保持较强的分辨率不变性；

**⚠️ 局限性**

局限性在于目前仅在二维或一维 PDE 上验证，三维大规模问题及复杂边界条件下的表现尚待进一步研究。

---

## 154. SAGE: Answer-Conditioned Uncertainty Targets for Verbal Uncertainty Alignment

**arXiv ID:** 2606.11512 | [PDF](https://arxiv.org/pdf/2606.11512v1)

**作者:** Kaiwen Shi `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提升大型语言模型的口头不确定性表达，使其与模型采样行为一致，提出 SAGE 作为分布式不确定性目标，并通过 GUPO 在不确定性通道上进行对齐训练。

**💡 创新点**

创新点在于将语义与答案等价性相结合的 SAGE 目标，提供平滑、答案感知且规模保持的分布式不确定性度量；以及 GUPO 的不确定性通道偏好优化框架，专门对不确定性表达进行监督，而非完整响应。

**🔧 技术方法**

使用答案抽取、任务特定的答案等价核、连续语言相似度核、von Neumann 熵计算、基于奖励的偏好分布和强化式对数似然损失进行不确定性通道微调。

**📊 数据集**

在 TriviaQA、MATH‑500 和 MMLU‑Pro 三大基准上进行实验，涵盖自由文本、数值/符号推理和多项选择三种答案结构。

**📈 对比分析**

与直接口头化、不确定性对齐、链式思考、DCA、CSFT、LoVeC‑DPO 等方法对比，评估 Brier、ECE、Spearman 相关和高置信子集准确率；GUPO+SAGE 在所有指标上均优于基线，显著降低校准误差、提升不确定性排名和高置信预测准确率。

**⚠️ 局限性**

仅对模型自身行为的不确定性进行校准，未直接保证答案正确性；依赖答案提取和等价性规则，可能在开放式任务中引入噪声；在高风险领域仍需外部验证与人类审查，口头不确定性无法替代专业判断。

---

## 155. Adversarial Attacks on Learned Policies for Surgical Robotic Tasks

**arXiv ID:** 2606.11535 | [PDF](https://arxiv.org/pdf/2606.11535v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 156. Privacy-Preserving Federated Autoencoder for ECG Anomaly Detection on Edge Devices

**arXiv ID:** 2606.11556 | [PDF](https://arxiv.org/pdf/2606.11556v1)

**作者:** Kaan Arda Akyol `[一作]` (Newcastle University), Rehmat Ullah `[通讯]` (Newcastle University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计并评估了一个端到端的联邦学习+差分隐私+INT8量化的 12 导联 ECG 无监督异常检测系统。

**💡 创新点**

首次将联邦学习、正式的 (ε,δ)-DP、无监督重构异常检测与 AArch64 边缘部署量化整合，且发现 DP 与量化损失相互独立。

**🔧 技术方法**

采用 FedAvg 联邦学习、DP‑SGD（Opacus）、Rényi‑DP 计数器、ConvAE/VanillaAE/VAE 自编码器、动态 INT8 PTQ、Flower 框架以及 Raspberry Pi 4 边缘推理。

**📊 数据集**

使用 PTB‑XL 12 导联 ECG 数据集（21,799 条记录）并以 10 个非 IID 服务器模拟医院划分。

**📈 对比分析**

与中心化训练对比：联邦学习提升 AUROC（ConvAE 联邦 0.782），ε=4 时 AUROC≈0.78，DP 不影响量化；INT8 将模型体积减半、Pi‑4 延迟约 30% 以上且 AUROC 下降 <0.12%。

**⚠️ 局限性**

局限性：仅在单一 PTB‑XL 数据集验证，未做跨数据集测试；未使用安全聚合，服务器可见梯度；量化评估基于延迟估计而非真实功耗；缺乏阈值敏感性分析和更强威胁模型。

---

## 157. SkillJuror: Measuring How Agent Skill Organization Changes Runtime Behavior

**arXiv ID:** 2606.11543 | [PDF](https://arxiv.org/pdf/2606.11543v1)

**作者:** Zhiyu Chen `[一作]` (Tongji University), Weinan Zhang `[通讯]` (Shanghai Innovation Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 Agent Skills 的组织方式对大语言模型代理的运行时行为和最终任务成功率的影响，并构建了 SkillJuror 框架以在知识不变的前提下比较不同组织范式。

**💡 创新点**

创新点在于将 Skill 组织视为可实验的运行时干预；通过知识保持约束构造平衡的“平面”与“渐进披露”两种技能变体；将轨迹级资源使用（ERU）与最终验证结果关联，揭示组织对行为与结果的任务依赖性。

**🔧 技术方法**

技术包括：① Skill‑for‑Skill 自动重构（LLM 生成），②三层验证管线（程序化门控、语义评审、人工回馈），③使用 Harbor 沙箱执行、GPT‑5.4 代理、LLM‑as‑judge 进行 ERU 标注；以及多维度评估（通过轨迹属性、资源路由质量等指标）。

**📊 数据集**

使用 82 个 SkillsBench 任务作为实验数据集，并在每个任务下执行 5 次对比实验（无 Skill、平面、渐进披露），共 1,230 次试验。

**📈 对比分析**

比较方法：在同一执行环境下对比两种 Skill 组织，统计通过率、资源使用量、ERU 事件、时间/成本等多维指标。结果显示：渐进披露将通过率提升至 46.1%（比平面高 4.1%），但收益显著依赖任务类型；在资源使用和 ERU 方面，渐进披露平均触及资源数从 1.18 增至 3.85，ERU 事件从 1.33 增至 3.92。

**⚠️ 局限性**

局限性包括：仅评估了渐进披露与平面两种组织；依赖 SkillsBench 任务和单一模型/配置，可能不适用于多技能、API 交互或实时评估场景；ERU 与桥接轨迹的标注依赖 LLM 判断，缺乏人工真值；实验次数有限，无法充分捕捉随机性。

---

## 158. MoCA-Agent: A Market-of-Claims Code Agent for Financial and Numerical Reasoning

**arXiv ID:** 2606.11537 | [PDF](https://arxiv.org/pdf/2606.11537v1)

**作者:** Abdelrahman Abdallah `[一作]` (University of Innsbruck), Muhammad Abdul-Mageed `[通讯]` (University of British Columbia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于“主张市场（Market‑of‑Claims）”的金融与表格问答代码代理MOCA-Agent，替代自由式多代理辩论，通过细粒度主张交易和结构化验证实现更稳健的数值推理。

**💡 创新点**

创新点在于：①将推理拆解为带类型的原子主张（事实、公式、单位、符号、方向），②让专家角色（提取器、公式验证器、会计师、怀疑者）对每个主张买卖并形成价格与置信度，③合成器仅使用未被驳回的主张生成可执行程序，④代码审计器基于市场记录执行结构化检查并可触发一次市场感知修复。

**🔧 技术方法**

技术组合包括：大型语言模型（Qwen3.6‑27B）作为多角色推理核心；主张构建器、专家交易市场、合成器、执行器、代码审计器、混合选择委员会与冲突仲裁器；结构化验证规则（符号翻转、百分比尺度错误、单位一致性等）。

**📊 数据集**

评测使用十个公开基准：FinQA、DocMath‑Simplong/Complong、FinanceMath、HiTab、MultiHiertt、TabMWP、WikiTableQuestions、ESGenius、FinChart‑Bench。

**📈 对比分析**

与当前最强模型（GPT‑4o、Claude Sonnet、Fin‑o1、TradingAgents 等）在同一 backbone（Qwen3.6‑27B）下对比，MOCA-Agent 在多数任务上显著提升，FinQA 78.3%（+4.1）、MultiHiertt 71.2%（+14.4）、ESGenius 86.9%（+3.1）等；整体平均准确率 85.6%。

**⚠️ 局限性**

局限包括：①高昂成本（6–10 次 LLM 调用）；②对超出 10 个主张或布局不符合预期的表格效果不佳；③多模态仅使用一次 VLM 转录，无法对转录错误再交易；④仅针对英文金融语料，未验证跨语言或非金融数值推理。

---

## 159. Maximizing Connectivity of Uplink RIS-Assisted UAV Networks

**arXiv ID:** 2606.11523 | [PDF](https://arxiv.org/pdf/2606.11523v1)

**作者:** Mohammed Saif `[一作]` (Toronto Metropolitan University), Shahrokh Valaee `[通讯]` (University of Toronto)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种联合优化框架，针对RIS辅助的UAV网络通过RIS增强链接选择、RIS虚拟分区以及UAV位置规划，最大化网络连通性（Fiedler值）并满足SINR约束。

**💡 创新点**

创新点在于：① 将Fiedler值作为连通性指标并在RIS与UAV系统中首次结合使用；② 设计基于Fiedler向量的扰动启发式链接选择算法；③ 推导RIS分区的闭式最优分配；④ 将UAV位置优化转化为低复杂度的半正定规划（SDP），可直接使用CVX求解。

**🔧 技术方法**

采用的技术包括：图论（Fiedler值、拉普拉斯矩阵）、扰动启发式算法、闭式求解、半正定规划（SDP）、CVX优化框架、Nakagami‑f 信道模型以及3GPP Urban Micro路径损耗模型。

**📊 数据集**

实验数据来自于仿真，使用3GPP Urban Micro（UMi）模型、Nakagami‑f 信道参数（f1=f2=5、f=1）、N=100个RIS元件、K_RIS=2、不同UAV数量和SINR阈值等，所有实验均为合成仿真数据。

**📈 对比分析**

与三种基准方案对比：①使用整个RIS构建单链路方案；②不使用RIS的原始方案；③随机UAV位置。结果显示，本文方法在网络连通性（Fiedler值）和平均速率上均显著优于基准，最大提升约10%–15%，并在不同UAV数量、RIS元素数和SINR阈值场景下保持稳健。

**⚠️ 局限性**

局限性包括：假设完美CSI；仅考虑单UE与单RIS的配置；未探讨多UE同时接入RIS的调度；在高SINR阈值稀疏网络下，增大UAV数量反而可能降低连通性；并且虽然SDP可行，但对大规模UAV/RIS系统的计算量仍然较高。

---

## 160. On Aligning Hierarchical Standardized Embedding for Audio-visual Generalized Zero-shot Learning

**arXiv ID:** 2606.11602 | [PDF](https://arxiv.org/pdf/2606.11602v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 161. Joint Movable Antenna Positioning and RIS Partitioning for Sum-Rate Maximization

**arXiv ID:** 2606.11519 | [PDF](https://arxiv.org/pdf/2606.11519v1)

**作者:** Mohammed Saif `[一作]` `[通讯]` (Toronto Metropolitan University), Mohammed Saif (Toronto Metropolitan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于可移动天线（MA）和可重构智能表面（RIS）的联合优化框架，在下行RIS辅助系统中通过优化MA子阵位置、子阵波束及RIS元素分配，最大化目标用户的总速率并抑制互用户干扰。

**💡 创新点**

创新点在于首次将MA的空间可移动性与RIS的分区/元素分配相结合，形成额外的设计自由度；同时提出了交替优化算法，将三大非凸子问题分别化简为ZF波束设计、1维RIS分配搜索和基于BCD的MA位置凸化求解。

**🔧 技术方法**

采用的技术包括：基于KKT条件的ZF波束设计、RIS分区向量的离散一维搜索、MA位置优化的BCDT凸化约束（使用一阶下界逼近），以及CVX+MOSEK求解凸子问题；整个框架基于离散化的几何通道模型实现。

**📊 数据集**

实验采用仿真数据：BS、RIS与用户在二维平面上随机布置（BS (0,0)，RIS (30,0)，用户随机落在半径10m圆内），使用Rician/LOS通道模型，参数如ν1=2、ν2=2.5、ε=3、P0=-30dB等。未使用公开数据集。

**📈 对比分析**

与三种基准（固定位置天线、随机位置天线、仅优化MA位置的随机波束）进行对比。结果表明，联合MA-RIS方案在不同RIS尺寸、功率预算以及MA数量下均实现了显著的速率提升；当RIS元素增多或MA数量增大时，总速率增长更为明显。

**⚠️ 局限性**

局限性包括：仅考虑单RIS单用户情形；假设完全CSI且RIS相位可精确对齐；MA位置优化仅在二维平面内，未考虑更复杂的三维或多天线布局；算法收敛速度和计算复杂度在大规模系统中仍需进一步评估。

---

## 162. Probabilistic Contrastive Pretraining for Multi-task ADME Property Prediction

**arXiv ID:** 2606.11508 | [PDF](https://arxiv.org/pdf/2606.11508v1)

**作者:** Yifan Xue `[一作]` (NVIDIA), Micha Livne `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种结合对比互信息学习与化学特定自监督的分子图Transformer预训练框架，用于改进ADME属性预测。

**💡 创新点**

创新点包括：①将cMIM与KERMT的化学自监督目标统一为单一概率潜在变量目标，避免超参调优；②使用任务特定的MLP头在多任务微调时保持共享表示的同时减轻负迁移；③在预训练语料中加入与ADME相关的分子以提升迁移效果。

**🔧 技术方法**

主要技术包括：分子图Transformer编码器、SMILES重构、对比互信息学习、化学特定预测任务（原子/键/功能团）、联合概率模型、任务特定多层感知机头。

**📊 数据集**

数据集：预训练语料有ZINC15_ChEMBL-11M、ZINC15_ChEMBL-up-208M，以及Biogen、ExpansionRX、ChEMBL-MT等ADME相邻扩充；下游基准为Biogen、ExpansionRX、ChEMBL-MT的多任务ADME评估数据。

**📈 对比分析**

通过在Biogen、ExpansionRX、ChEMBL-MT上微调，对比cMIM、KERMT及Contrastive KERMT三种预训练方案，使用Pearson R和MAE评估。Contrastive KERMT相对基线提升7.6%、9.9%和9.5%（在Biogen、ExpansionRX、ChEMBL-MT），并在邻域保真度指标上显示更佳的化学相似性。

**⚠️ 局限性**

限制包括：预训练中使用的语料中仍含有下游任务中的分子，无法完全评估无标签适配效果；对比实验主要在特定ADME基准，缺乏跨学科的泛化评估；对超参数的敏感性和计算成本未作深入探讨。

---

## 163. Counterexample Guided Learning in the Large using Reasoning Agents

**arXiv ID:** 2606.11521 | [PDF](https://arxiv.org/pdf/2606.11521v1)

**作者:** Hongyi Liu `[一作]` (University of Wisconsin-Madison), Adithya Murali `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并评估了基于反馈的正则表达式学习框架，利用LLM生成候选正则并通过验证器生成反例进行迭代改进。

**💡 创新点**

创新点在于将聚类反例、正则化、反思与修复循环等策略整合到LLM代理工作流中，实现对符号学习的可控反馈利用。

**🔧 技术方法**

技术包括LLM推理、计数反例生成、符号反例聚类、正则化约束、反思（reflection）与修复（repair）循环以及主动学习。

**📊 数据集**

实验使用了简单正则和扩展正则两套数据集，分别对应不同语法复杂度的正则表达式。

**📈 对比分析**

与标准提示、单一聚类反例等基线相比，加入反馈后成功率提升至最难层级的38.1%~74.1%，并在多模型上保持优势，且Agentic流程进一步提高准确率但代价增加。

**⚠️ 局限性**

局限性包括对反例聚类的实现依赖于正则语言的闭包性质、在极难任务上仍有显著误差、以及代理工作流产生的令牌开销较大。

---

## 164. When is Your LLM Steerable?

**arXiv ID:** 2606.11599 | [PDF](https://arxiv.org/pdf/2606.11599v1)

**作者:** Chenrui Fan `[一作]` (University of Maryland), Tianyi Zhou `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含150个概念、50个提示、约1.42M条标注的激活SteerBench数据集，并研究在生成过程中通过观察前几个token的隐藏状态来预测激活steering的成功与否，提出了SteerBoost预测器；

**💡 创新点**

核心创新是利用跨层、跨token的隐藏状态差异特征（Steering geometry、Decoding dynamics、Steering condition）和GBDT模型，在不完成完整自回归生成的前提下即可预测steering结果，并在不同模型、方法和概念层级间展现可迁移性；

**🔧 技术方法**

采用Gradient Boosting Decision Tree作为分类器，设计了多组特征并进行归一化；使用DiffMean和Probe两种steering方法，结合多层、不同token位置的特征抽取；

**📊 数据集**

主要使用自构造的SteerBench数据集（1.42M条生成样本）覆盖Qwen3-1.7B、Gemma-2-2B-it、LLaMA-3.2-3B-Instruct三大模型；标注由GPT-5-nano完成，并经过人工评估验证；

**📈 对比分析**

将SteerBoost-guided search与传统概念级、项级网格搜索及早停方法对比，宏F1在ID概念上约0.8、OOD约0.7；在搜索任务中，SteerBoost在仅解码约11%完整生成的token时即可恢复约98%的项级搜索成功率；

**⚠️ 局限性**

局限性包括：对边界效应（UnderSteer/SuccSteer）的区分仍不完美；特征工程与模型训练受限于已有标注的概念，无法完全泛化到全新概念；对极端steering强度的预测精度有下降；并未探讨不同模型规模和架构对特征可迁移性的进一步影响。

---

## 165. Learning Object Manipulation from Scratch via Contrastive Interaction

**arXiv ID:** 2606.11525 | [PDF](https://arxiv.org/pdf/2606.11525v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 166. Kuramoto Attention: Synchronizing Self-Attention on the Torus

**arXiv ID:** 2606.11585 | [PDF](https://arxiv.org/pdf/2606.11585v1)

**作者:** Joshua Nunley `[一作]` `[通讯]` (Indiana University), Joshua Nunley (Indiana University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Kuramoto attention 的自注意力层，将隐藏状态设计为角度（相位），并通过可学习的门控余弦相似度与旋转位置实现注意力加权，随后以切线投影方式在圆环上更新相位，从而实现精确的 Kuramoto 同步步进。

**💡 创新点**

创新点在于将自注意力与相位同步动力学统一到同一层：注意力权重即为可学习的耦合核，值更新即为 Kuramoto 交互项；同时通过旋转位置实现自然频率的学习，构造了一个几何原生、在圆环上闭合的注意力模块。

**🔧 技术方法**

技术细节包括：1）相位的 (cos,sin) 提升与门控查询/键的 softplus‑正则化；2）可学习的相位余弦相似度与位置漂移 ω_j(t‑u)；3）softmax 作为熵正则化检索；4）基于注意力加权圆均值的切线投影实现相位更新；5）通过符号值门与范数上限控制步长；6）SwiGLU 作为残差的非几何前馈块。

**📊 数据集**

使用 enwiki8 字符级语言建模数据集，序列长度 256，训练 1M 与 5M 参数规模。

**📈 对比分析**

与匹配参数规模的 RoPE+SwiGLU Transformer 进行对比：在 1M 参数时 BPC 仅落后 0.02，5M 参数时在五个种子中中位数相同，Transformer 在平均值上略占优势；消融实验显示值投影、度量门、前馈块是关键贡献。整体表现接近标准 Transformer，但无速度提升。

**⚠️ 局限性**

局限性：仅在相位状态层下实现了完全同步的 Kuramoto 公式，标准 dot‑product 注意力仅能收敛到极限；实验仅在小规模字符级任务上验证，未在更大规模或其他任务中测试；未实现非阿贝尔群（如 SO(3)）的几何原生前馈块；在 1M 参数时性能略逊于 Transformer。

---

## 167. Spatially Coupled Phase-to-Depth Calibration for Fringe Projection Profilometry

**arXiv ID:** 2606.11601 | [PDF](https://arxiv.org/pdf/2606.11601v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 168. FreqKD: Frequency-Decoupled Cross-Modal Knowledge Distillation for Infrared Object Detection

**arXiv ID:** 2606.11572 | [PDF](https://arxiv.org/pdf/2606.11572v1)

**作者:** Keval Thaker `[一作]` (University of Michigan-Dearborn), Samir A. Rawashdeh `[通讯]` (University of Michigan-Dearborn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于频率分解的知识蒸馏框架 FreqKD，用于将 RGB 视觉基础模型迁移到红外（IR）图像；

**💡 创新点**

创新点在于通过对 RGB‑IR 特征的频谱分析发现高频差异更大，进而在低频上使用严格 MSE，在高频上使用 0.1×log‑MSE 的不对称损失，并配合 2D‑FFT 径向分隔与 LoRA 适配；

**🔧 技术方法**

使用的技术包括 2D‑FFT 频率分解、径向低通/高通掩码、中心化 L₂ 归一化、log‑MSE 高频损失、LoRA 参数适配、以及后续的 LoRA 融合；

**📊 数据集**

实验数据集涵盖 KAIST 多光谱行人检测、FLIR ADAS 边界检测、MFNet 热像语义分割以及 ResNet‑50 迁移，教师模型为 DINOv2 ViT‑Large；

**📈 对比分析**

与 DINOv2 直接预训练、均匀特征 KD、余弦相似度 KD、响应级 KD 等方法对比，FreqKD 在 KAIST 上 mAP_50 提升至 64.1（+2.4），在 FLIR、MFNet 及 ResNet‑50 上均表现出正向迁移；

**⚠️ 局限性**

局限性包括仅针对长波红外图像、使用固定径向阈值 r_c、单一 DINOv2 教师、未对其他非可见光模态或多教师场景进行验证。

---

## 169. HERO: Hindsight-Enhanced Reflection from Environment Observations for Agentic Self-Distillation

**arXiv ID:** 2606.11559 | [PDF](https://arxiv.org/pdf/2606.11559v1)

**作者:** Haoran Liu `[一作]` (University of California), Jingbo Shang `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于下一步环境观察的自回馈反思式自蒸馏方法，改进多回合工具使用代理的性能。

**💡 创新点**

创新点在于将每一步的后续观察压缩为局部诊断提示，解决传统全轨迹条件下的上下文不匹配问题。

**🔧 技术方法**

利用自蒸馏、Jensen‑Shannon 散度、反思生成器和自教师模型，将诊断提示作为额外上下文生成稠密词级监督。

**📊 数据集**

在 TauBench（Retail、Airline）和 WebShop 等多回合工具使用基准上进行实验。

**📈 对比分析**

与基线模型、仅环境反馈、完整轨迹教师以及 GRPO 对比，HERO 在成功率和回合数上均优于对手，尤其在有限回合预算下表现更佳。

**⚠️ 局限性**

局限在于对模型自身反思能力的依赖，难以处理需要深度推理或数学推导的错误；对强大指令调优模型更友好，基础模型效果有限。

---

## 170. When Roleplaying, Do Models Believe What They Say?

**arXiv ID:** 2606.11502 | [PDF](https://arxiv.org/pdf/2606.11502v1)

**作者:** Benjamin Sturgeon `[一作]`, Sid Black `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过线性真理探针研究语言模型在角色扮演（历史人物）与Emergent Misalignment时内部信念的变化。

**💡 创新点**

创新点在于将角色扮演与Emergent Misalignment放在同一量化框架下比较，并揭示角色扮演只改变输出而不深度改变内部真理方向，而EM则显著改变内部信念。

**🔧 技术方法**

使用线性逻辑回归真理探针、系统提示、上下文学习、SFT以及对抗性挑战测试等技术。

**📊 数据集**

数据集包括从Claude生成的历史人物对错陈述、era‑believed/era‑false样本、以及用于EM的有害建议和对抗性问题。

**📈 对比分析**

在所有模型族（Llama 3.3‑70B、Qwen 3‑8B、Qwen 2.5‑14B）上，角色扮演的保护差距小且未跨越真理阈值，而EM模型在真理探针、抗压测试和下游推理中显示出大幅提升，效果优于角色扮演。

**⚠️ 局限性**

局限在于探针可能捕捉连贯性而非真正信念、角色训练深度不足、以及对不同模型的泛化不一。

---

## 171. Contactless 3D Human Body Measurement Using Depth Cameras for Smart Health Monitoring

**arXiv ID:** 2606.11578 | [PDF](https://arxiv.org/pdf/2606.11578v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 172. Range-Aware Bayesian Optimization for Discovering Diverse Designs within Target Property Windows

**arXiv ID:** 2606.11574 | [PDF](https://arxiv.org/pdf/2606.11574v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 173. Model-Based and Data-Driven Hierarchical Control and Topology Co-Design for Robust Networked Systems

**arXiv ID:** 2606.11596 | [PDF](https://arxiv.org/pdf/2606.11596v1)

**作者:** Shirantha Welikala `[一作]` (Stevens Institute of Technology), Panos J. Antsaklis `[通讯]` (University of Notre Dame)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种层次化的基于模型和数据驱动的网络系统协同设计方法，能够在保证闭环系统从扰动到性能输出的耗散性（dissipativity）的前提下，联合设计局部子系统控制器、分布式全局控制器以及互连拓扑，目标是实现鲁棒电压调节与电流共享。

**💡 创新点**

创新点：①首次将耗散性理论与层次化控制框架结合，实现局部子系统耗散性保证后再协同设计全局控制器与拓扑；②在未知子系统模型时，通过仅利用轨迹数据并假设扰动满足二次矩阵不等式，使用矩阵S-lemma构造数据驱动的LMI条件，避免了模型识别步骤；③提出了可调稀疏性的目标函数，实现通信拓扑的自适应稀疏化；④在DC微电网实验中验证了该方法能在保证稳态性能的同时显著减少通信链路。

**🔧 技术方法**

主要技术：耗散性理论、线性矩阵不等式（LMI）设计、矩阵S-lemma、数据驱动控制（Fundamental lemma与S-lemma结合）、局部与全局层次化协同设计、稀疏性优化。

**📊 数据集**

使用了DC微电网（DCMG）案例作为仿真数据集，包含6个分布式发电机和7条线路的拓扑，利用其采样数据实现数据驱动设计；未使用公开大规模实验数据集。

**📈 对比分析**

与传统的全局稳定控制器（完全连通通信拓扑）相比，本文方法在相同的稳态电压、电流目标下，能够得到更稀疏的通信拓扑，并在保持闭环耗散性和鲁棒性的同时，达到相似甚至更好的状态跟踪性能；数值仿真显示，两种方法的电压/电流轨迹几乎相同，但后者的通信链接显著减少。

**⚠️ 局限性**

局限性：①需要子系统数据足够丰富且满足持续激励条件；②扰动被假设满足二次矩阵不等式，若实际噪声不符合此假设可能导致保守或失效；③设计过程仍需要解一系列LMI，规模较大时计算量可观；④本文只考虑了离散时间或离散化的连续系统，对实时在线自适应等更复杂情景尚未深入研究。

---

## 174. Hiding the Trees in the Forest: Building Network Covert Channels with Hash-Based Covert Carrier Filtering

**arXiv ID:** 2606.11532 | [PDF](https://arxiv.org/pdf/2606.11532v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 175. Defense Against Prompt Inversion Attacks: An Information-Theoretic Approach for LLM Collaborative Inference

**arXiv ID:** 2606.11592 | [PDF](https://arxiv.org/pdf/2606.11592v1)

**作者:** Sayedeh Leila Noorbakhsh `[一作]` (University of California), Nader Sehatbakhsh `[通讯]` (University of California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于信息理论的提示反演防御框架，在协作边缘-云LLM推理中通过学习隐私适配器压缩中间激活，显著降低提示泄露；

**💡 创新点**

创新点在于：①将隐私泄露建模为互信息最小化，并通过与学习式对手的极小极大训练实现可调节的隐私‑效用权衡；②引入无残差低维信息瓶颈适配器，理论证明可真正降低互信息；③提供严格的Fano、互信息上界与下界解析，阐释隐私与下游性能的根本关系；④实现对敏感词的选择性保护，提升隐私效益；

**🔧 技术方法**

使用了信息理论（互信息、Fano不等式）、变分CLUB估计、极小极大对抗训练、信息瓶颈网络、低维线性变换、GELU激活、层归一化、量化理论分析以及PyTorch实现；

**📊 数据集**

实验数据集包括医学临床记录（MedAlpaca）、航空评论（Skytrax）和法律判例（ECHR）三大领域；

**📈 对比分析**

与噪声注入、NoPeek、Fisher、ShredMI、PCA等现有防御在LLaMA‑2‑7B/13B、Mistral‑7B上进行对比；在不超过9%延迟开销的前提下，攻击成功率下降35‑43%，且困惑度（PPL）保持在基线附近；

**⚠️ 局限性**

局限性包括：需在λ和瓶颈维度r之间手动平衡，过高隐私设置会明显影响生成质量；当前仅在特定模型分层（k=4）和英语数据上验证；尚未覆盖更广泛的攻击模型或多语言场景；

---

## 176. On the Study of Biometric Spoofing Detection using Deep Learning

**arXiv ID:** 2606.11505 | [PDF](https://arxiv.org/pdf/2606.11505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 177. ConsistencyPlanner: Real-time Planning with Fast-Sampling Consistency Models

**arXiv ID:** 2606.11569 | [PDF](https://arxiv.org/pdf/2606.11569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 178. 4DP-QA: Scalable QA for 4D Perception in Vision Language Models

**arXiv ID:** 2606.11568 | [PDF](https://arxiv.org/pdf/2606.11568v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 179. A Deterministic Forensic Preprocessing Framework for Heterogeneous Network Datasets: Formal Foundations, Implementation, and Empirical Validation

**arXiv ID:** 2606.11565 | [PDF](https://arxiv.org/pdf/2606.11565v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 180. Cross-Modal Benchmarking for Robotic Perception in Natural Environments

**arXiv ID:** 2606.11563 | [PDF](https://arxiv.org/pdf/2606.11563v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 181. Teaching Diffusion to Speculate Left-to-Right

**arXiv ID:** 2606.11552 | [PDF](https://arxiv.org/pdf/2606.11552v1)

**作者:** Lexington Whalen `[一作]` (SB Intuitions), Ryo Sakamoto `[通讯]` (SB Intuitions)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文针对块式扩散式草稿模型（block‑diffusion drafters）在推理时的训练与验证不匹配问题，提出并评估了三种训练时的损失重塑方法，以提升草稿模型在 Speculative Decoding（自回归推理）中的接受长度。

**💡 创新点**

创新点在于：①将位置加权（loss decay）引入传统的对称交叉熵中，使梯度更贴合先行位置的重要性；②设计“first‑error focal loss”，只针对草稿块中首次预测错误的位置提供额外监督，从而精准指向导致后续拒绝的关键位置；③提出“chain reward”，用可微的期望接受长度近似来直接强化草稿的全局前缀接受概率；三者可互补组合，并且与现有的推理时自适应方法（如 ddTree）兼容。

**🔧 技术方法**

使用的技术包括：
- Speculative Decoding 框架；
- DFlash（block‑diffusion）草稿模型；
- 位置加权交叉熵、first‑error focal loss、chain reward 这三种损失改进；
- 训练时使用 AdamW + cosine 调度；
- 评估时利用多种推理时策略（连续块验证 vs ddTree 变形树验证）。

**📊 数据集**

主要数据集：
- ShareGPT（约 800K 训练样本，直接取原始对话）；
- Nemotron Post‑Training Dataset V2 + CodeAlpaca（约 800K，先在目标模型上重生成），用于实验数据集的对齐；
- 评估基准：GSM8K、MT‑Bench、HumanEval、MBPP、AIME、LiveCodeBench 等六个标准基准。

**📈 对比分析**

比较方法：在同一目标模型（Meta‑Llama‑3‑8B‑Instruct、Llama‑3.2‑3B‑Instruct、Qwen‑3‑4B、Qwen‑3‑8B）上，先使用位置均匀基线（cross‑entropy），再逐步加入 loss decay、focal loss、chain reward，记录平均接受长度 τ 及吞吐量 TPS。实验显示：
- 单独使用 chain reward 可提升约 43% 的 τ；
- 全部三种改进叠加后，τ 在 4‑B 目标上提升 27%~44%；
- 结合 ddTree 验证可进一步提升至 94% 的 τ 及 132% 的 TPS；
- 在 SpecDiff‑2 训练基线上叠加三种改进可提升 74% 的 τ。

**⚠️ 局限性**

limitations：
- 训练成本仍受限于对齐数据的生成，尤其是目标模型的前向推理需要额外计算；
- 位置加权与 focal loss 的超参数（γ、α_f、α_c）需手动调优，缺乏自动化方案；
- 在更大块大小或更深模型上可能出现过拟合或梯度不稳定，需要进一步研究；
- 论文未探讨在动态推理场景（如多步代理、可变长度文本）下的泛化性能。

---

## 182. Pretrained self-supervised speech models can recognize unseen consonants

**arXiv ID:** 2606.11542 | [PDF](https://arxiv.org/pdf/2606.11542v1)

**作者:** Chihiro Taguchi `[一作]` (University of Notre Dame), David Chiang `[通讯]` (University of Notre Dame)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过对自监督预训练的 ASR 模型（Wav2Vec2 与 HuBERT）进行 fine‑tune，验证它们在两种点击语种 Gui 与 West !Xoon 中识别点击辅音的能力。

**💡 创新点**

主要创新在于：①构建了少数民族点击语种的 ASR 数据集；②系统评估自监督模型对罕见语音单位（点击辅音）的泛化性能；③发现即使预训练数据几乎不包含点击音，模型依旧能较好地识别点击音。

**🔧 技术方法**

采用的技术包括：自监督预训练的 Wav2Vec2、HuBERT 以及多语言扩展模型；CTC 解码（贪婪、束搜索、束搜索+3/5 词表语言模型）；AdamW 优化器与对齐算法（Needleman‑Wunsch）。

**📊 数据集**

使用的语料库是自制的 Gui 与 West !Xoon 数据集，分别包含 50 条 Gui 录音（约 1.75 小时）和 150 分钟 West !Xoon 录音，分为训练与测试集。

**📈 对比分析**

实验通过比较不同模型、不同参数规模、不同解码策略的 PER/CER 以及点击 vs 非点击音的错误率进行。结果显示：HuBERT（单语种）与 wav2vec2-large-xlsr-53 表现最好；较大模型并不一定更好；点击辅音的错误率显著低于非点击辅音。

**⚠️ 局限性**

局限性包括：数据量有限、未设验证集、仅覆盖两种点击语种、未评估自回归解码的潜在优势、模型对其他稀有音素的泛化仍待进一步验证。

---

## 183. SceneMiner: Identity-Preserving Multi-Task Fine-Tuning for Unified BEV Scene Mining

**arXiv ID:** 2606.11507 | [PDF](https://arxiv.org/pdf/2606.11507v1)

**作者:** Abdalmalek Aburaddaha `[一作]` (University of Michigan-Dearborn), Samir A. Rawashdeh `[通讯]` (University of Michigan-Dearborn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SceneMiner，一种单摄像头的鸟瞰视图多头网络，可在一次前向传播中同时输出检索嵌入、场景多标签、物理风险分数以及运动预测，帮助挖掘驾驶日志中的高难度安全场景。

**💡 创新点**

核心创新在于识别并解决跨任务干扰（cross‑task interference），通过身份保持多任务微调（zero‑initialization + 参数冻结）保证新增头部不会破坏已训练的同流头部；同时提供无监督的物理风险标签与检索嵌入，形成完整的难度评估与检索信号。

**🔧 技术方法**

技术上使用冻结的 SigLIP2 + BEVFormer 视觉–语言骨干，32‑查询 Q‑Former 作为共享池化器，构建检索、标签、风险和运动四个并行头；对风险头使用物理启发式伪标签，对检索头使用 InfoNCE；通过身份保持微调实现多头稳定。

**📊 数据集**

仅在 nuScenes v1.0 数据集上训练和评估，使用六路环视摄像头（无 LiDAR/雷达），从原始日志中提取图像序列、轨迹与场景描述。

**📈 对比分析**

与 UniAD、VAD、DriveLM‑Agent 等统一 AD 系统对比，SceneMiner 仅 102k 可训练参数，单前向 204.5 ms；检索 R@10≈0.065，标签 mAP≈0.461，风险 Pearson r≈0.392，运动 ADE≈2.03 m；相比基线，标签/风险指标基本保持不变，且在多头间无性能退化。

**⚠️ 局限性**

局限性包括：实验仅在 nuScenes 进行，缺乏跨数据集泛化验证；风险信号与运动头共用注释，需外部难度标注进一步验证；检索评估仍以定性示例为主，缺少量化指标；对极端稀有场景的泛化能力尚未完全评估。

---

## 184. GraphInfer-Bench: Benchmarking LLM's Inference Capability on Graphs

**arXiv ID:** 2606.11562 | [PDF](https://arxiv.org/pdf/2606.11562v1)

**作者:** Zhuoyi Peng `[一作]` (Hong Kong University of Science and Technology), Yi Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出GraphInfer-Bench基准，评估LLM在图推理（描述与比较）任务中的能力。

**💡 创新点**

创新点在于引入“图推理”定义，构建无单节点可解的多任务数据集，并提出四层质量控制流水线。

**🔧 技术方法**

采用图-token对齐、零射前沿LLM、Graph2Text SFT和纯GNN四大方法族进行对比实验。

**📊 数据集**

使用六个真实文本属性图（ogbn-arxiv、ogbn-products、PubMed、WikiCS、USPTO、Physics SE）共42,000条样本。

**📈 对比分析**

结果显示前沿LLM和Graph2Text在描述任务上接近GNN，但在多节点比较任务（尤其社区划分）仍落后，GNN保持优势。

**⚠️ 局限性**

局限在于仅关注节点邻域推理，未覆盖跨图推断或动态图场景，且方法仍受解码与对齐瓶颈限制。

---

## 185. Measuring language complexity from hierarchical reuse of recurring patterns

**arXiv ID:** 2606.11531 | [PDF](https://arxiv.org/pdf/2606.11531v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 186. LLMs+Graphs: Toward Graph-Native, Synergistic AI Systems

**arXiv ID:** 2606.11560 | [PDF](https://arxiv.org/pdf/2606.11560v1)

**作者:** Arijit Khan `[一作]` (Bowling Green State University), Xin Huang `[通讯]` (Hong Kong Baptist University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本教程系统综述了大型语言模型与图结构数据的协同关系，阐述了LLM如何服务于图数据管理、挖掘与学习，以及图如何为LLM提供检索增强、知识图谱支持和代理式推理框架。

**💡 创新点**

首次构建从LLM→图、图→LLM、知识图谱与AI代理全链路的统一框架，强调图本地化与协同AI系统的设计原则，提供全链路的技术与系统思路。

**🔧 技术方法**

采用的技术包括：大型语言模型（LLMs）、图神经网络（GNNs）、图数据库与查询语言（Neo4j/Cypher/GraphQL）、检索增强生成（Graph RAG）、知识图谱构建与推理、代理式工具调用图与记忆图等。

**📊 数据集**

本文为综述性教程，没有使用新的数据集，主要引用已有工作中的公开图数据集、知识图谱数据和图数据库。

**📈 对比分析**

由于是综述与教程性质，未进行实验对比；文中引用了相关研究的评估结果，指出LLM+图的性能提升、可解释性增强以及在多跳推理、知识检索等场景中的优势。

**⚠️ 局限性**

局限性：缺乏系统性实验验证与统一评估指标，对资源消耗、可扩展性和部署实践的讨论不足；同时未详细探讨跨模态与多任务迁移的实际效果与挑战。

---

## 187. PriME-Deal: Privacy-Preserving Bilateral Data Trading with Efficient Matchmaking and Auditable Fair Exchange on Blockchain

**arXiv ID:** 2606.11539 | [PDF](https://arxiv.org/pdf/2606.11539v1)

**作者:** Jie Zhang `[一作]` (Tianjin University), Guangdong Bai `[通讯]` (City University of Hong Kong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种名为PriME‑Deal的去中心化数据交易协议，能够在不泄露访问策略的前提下实现双向属性匹配、线性时间访问控制以及可审计的公平交换；

**💡 创新点**

核心创新在于通过令牌（token）将匹配与解密解耦，实现非交互式匹配；采用 OKVS + PRF 隐藏属性并利用零知识证明（Groth16）验证买方合规；以及基于抵押金和 Merkle 树的链上公平交换机制；

**🔧 技术方法**

使用的技术包括：BLS/双线性群，Shamir 秘密共享，OKVS（线性隐蔽键值存储），PRF 隐藏，AES‑128‑GCM 对称加密，Groth16 zk‑SNARK，智能合约（Solidity）以及 Cuckoo/Bloom 滤波器；

**📊 数据集**

实验数据基于合成数据，使用不同的属性集合大小 (n_b)、阈值 t_b 以及买方属性数 m 进行多组性能评估；

**📈 对比分析**

与现有的模糊 IB‑ME 方案相比，PriME‑Deal 在卖方发布阶段实现了约 100 倍的加速（8.76 s 对 690 s），买方在线时间保持在 9 s 以内；Groth16 证明时间稳定在 0.6 s；链上 Gas 消耗约 28.6 M，低于以太坊区块上限；

**⚠️ 局限性**

局限性包括：需要信任的中心化 CA；阈值上限较低（t_b ≤ 5）时性能下降；对极大属性集合（n_b > 500）仍需改进匹配效率；目前仅基于传统双线性群，未考虑后量子安全。

---

## 188. AI Researchers Must Help Lead Arms Control to Mitigate Military AI Risks

**arXiv ID:** 2606.11533 | [PDF](https://arxiv.org/pdf/2606.11533v1)

**作者:** Ted Fujimoto `[一作]` (Pacific Northwest National Laboratory), Jacob Benz `[通讯]` (Pacific Northwest National Laboratory)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对军事AI风险的系统分析，提出AI研究者应主导军备控制研究，以减少军用AI的灾难性风险。

**💡 创新点**

创新点在于将核军备控制经验与AI治理相结合，提出可验证的AI风险控制框架、计算治理与跨国合作机制，首次强调AI研究者在军备控制中的领导作用。

**🔧 技术方法**

主要技术手段包括：AI风险验证工具（如可解释性与计算度量）、跨国计算治理平台、信息共享与监测协议、合作AI多主体决策框架。

**📊 数据集**

论文未使用具体实验数据集，主要基于文献综述与政策案例分析。

**📈 对比分析**

由于为政策性位置论文，未进行实验对比或性能评估。

**⚠️ 局限性**

局限性在于缺乏可操作的技术验证方案，跨国信任与信息共享难题、AI系统不可观测性、以及对现有军备控制机制的适配与演进需要进一步研究。

---

## 189. ISE: An Execution-Grounded Recipe for Multi-Turn OS-Agent Trajectories

**arXiv ID:** 2606.11520 | [PDF](https://arxiv.org/pdf/2606.11520v1)

**作者:** Siyuan Luo `[一作]`, Lewei Lu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出三阶段 OS 代理数据合成流程 ISE（Intent → Simulate → Execute），并生成包含 23,132 条真实 OS 执行、多轮对话的 ISETrace 数据集。

**💡 创新点**

创新点在于：①使用 4D 结构化意图采样（人设、域、任务、复杂度）消除工具优先偏差；②引入角色锁定、多轮用户模拟器，抑制角色漂移与状态错觉；③在真实隔离工作空间中执行工具调用，保证数据与真实执行语义一致。

**🔧 技术方法**

技术手段包括：大语言模型（LLM）用于意图生成与自然语言化；角色锁定的 LLM 用户模拟器；基于 OpenClaw 平台的实时工具执行与工作空间隔离；基于 ClawEval 的评估与 Fine‑Tuning。

**📊 数据集**

使用的数据集为：①IseTrace（训练集）；②ClawEval（T‑family 任务，用于统一基准评估）。

**📈 对比分析**

在 ClawEval 上进行对比实验：Fine‑Tuning Qwen3‑8B 8B 在 ISETrace 上从 19.3% 提升到 37.7% pass@1，超过 GPT‑4o 零样本（25.4%）和 4× 大型 Qwen3‑32B（30.7%），展示了多轮执行数据对小模型的显著提升。

**⚠️ 局限性**

局限性包括：数据规模仅 23k 条；仅覆盖 macOS/Linux shell，未覆盖 Windows/GUI 或浏览器交互；对中间规模模型的效果不稳定；以及角色锁定的效果受 LLM 指令遵循能力限制。

---

## 190. CS-YODAS: A Mined Dataset of In-the-Wild Code-Switched Speech

**arXiv ID:** 2606.11514 | [PDF](https://arxiv.org/pdf/2606.11514v1)

**作者:** Brian Yan `[一作]` (Carnegie Mellon University), Shinji Watanabe `[通讯]` (Carnegie Mellon University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过人类参与的LLM检测流程，从YODAS公开的YouTube语音数据中挖掘并构建了一个包含313小时、七种矩阵语种的自然发生的代码切换（CS）语料库CS‑YODAS，并提供了基准实验来评估CS语音识别中的语言识别（LID）性能。

**💡 创新点**

创新点包括：①基于LLM的文本层面多语种识别与人类反馈循环的可扩展检测管线，显著提升了CS片段的检出精度；②首次构建大规模自然发生的CS语料库，为CS研究提供真实数据；③用该语料库训练的LID模型在跨域、跨语言的代码切换识别上取得显著提升，证明自然CS数据的必要性。

**🔧 技术方法**

技术手段包括：多语言LLM（Qwen3‑14B）进行文本级语种推断；人类反馈在上下文学习中提升LLM输出准确度；自监督语音编码器MMS与ECAPA‑TDNN组合的LID模型；AAM‑Softmax损失和子中心增强进行分类；以及对CS片段进行15秒左右的上下文包络提取。

**📊 数据集**

使用的数据集：源YODAS（166k小时、75语种）→挖掘得到CS‑YODAS（313h、7矩阵语种）；对比实验中使用的Monolingual FLEURS、合成CS‑FLEURS（TTS生成的代码切换语料）。

**📈 对比分析**

比较方法：在FLEURS和CS‑FLEURS两个测试集上，对比两种训练配置（仅FLEURS+CS‑FLEURS合成 vs. 加入CS‑YODAS自然CS），利用准确率评估LID模型。结果显示：在CS‑FLEURS READ子集上，模型从0%提升至51.1%（法英）和19.9%（印地语-英），并且随着CS‑YODAS训练时长超过5小时，准确率出现显著上升趋势，证明自然CS训练数据对模型性能至关重要。

**⚠️ 局限性**

局限性：①源自CC YouTube的YODAS偏向公开广播类内容，导致日常随意场景的CS表达不足；②仅覆盖7种矩阵语种，语言覆盖范围受限；③人类验证规模有限，仍存在转录错误、音频分段噪声等残留噪声；④高质量的双语标注依赖专业人工，难以进一步扩大规模。

---

## 191. Search Discipline for Long-Horizon Research Agents

**arXiv ID:** 2606.11522 | [PDF](https://arxiv.org/pdf/2606.11522v1)

**作者:** Adithya Srinivasan `[一作]` (North Carolina State University), Devesh Paragiri `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在自我研究代理中发现并解决了聚合指标导致候选模型错误排序的问题，并通过外部审核循环实现安全决策。

**💡 创新点**

首次正式定义“归约逆转”现象，提出基于离散行为的审核协议和外部控制循环，避免代理因聚合评分误判而接受不合格方案。

**🔧 技术方法**

使用GPT‑5.5驱动的Hermes自研代理、监控与判断模块（同模型），以及结构化的候选‑效果审核框架。

**📊 数据集**

以Ecosystem Demography火灾子模型的全球与区域预测数据为实验数据集，包含真实的气候与植被输入。

**📈 对比分析**

与基准prompt及结构化prompt对比，采用外部循环后，能够准确拒绝因聚合误差导致的优胜者并保持搜索持续，性能提升体现在避免错误模型并延长搜索深度。

**⚠️ 局限性**

受限于固定输入合同导致无法捕获所有导致区域差异的因素，审核只在可观测空间内工作；实验规模有限，未在多次随机运行中量化效果。

---

## 192. DeepRHP: A Hybrid Variational Autoencoder for Designing Random Heteropolymers as Protein Mimics

**arXiv ID:** 2606.11651 | [PDF](https://arxiv.org/pdf/2606.11651v1)

**作者:** Shuni Li `[一作]` (University of California Berkeley), Haiyan Huang `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种基于半监督的混合变分自编码器DeepRHP，用于学习和生成具有蛋白质模拟功能的随机异质聚合物（RHP）序列；

**💡 创新点**

创新点在于将传统VAE与特征驱动的VAE并行训练，利用化学功能特征（如HLB）引导潜在空间，从而获得可解释、与功能相关的低维嵌入；

**🔧 技术方法**

使用PyTorch实现的多层感知机VAE架构，加入了α加权的ELBO损失，结合重构损失和特征回归损失；

**📊 数据集**

训练数据来自Panganiban2018和Jiang2020的RHP合成库（四种甲基丙烯酸酯单体的10,000条序列）以及从UniProt提取的30,000条膜蛋白和30,000条球状蛋白的同义序列；

**📈 对比分析**

通过PCA和AQPZ实验对比，证明DeepRHP能够准确识别出最优的四单体组成（RHP 4/5），其预测与实验结果高度一致，且在消融实验中优于单一经典VAE；

**⚠️ 局限性**

局限性包括目前仅采用定性评估，缺乏量化指标；潜在空间的解释性仍需进一步验证；模型仅针对四种单体，扩展至更大单体词汇表的通用性待验证。

---

## 193. SAFER-Nav: Enhancing Safety for Visual Robot Navigation via Segmentation-Aware Fine-Tuning

**arXiv ID:** 2606.11636 | [PDF](https://arxiv.org/pdf/2606.11636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 194. When Context Returns: Toward Robust Internalization in On-Policy Distillation

**arXiv ID:** 2606.11627 | [PDF](https://arxiv.org/pdf/2606.11627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 195. Dummy Backdoor as a Defense: Removing Unknown Backdoors via Shared Internal Mechanisms for Generative LLMs

**arXiv ID:** 2606.11648 | [PDF](https://arxiv.org/pdf/2606.11648v1)

**作者:** Kazuki Iwahana `[一作]` (NTT), Akira Ito `[通讯]` (Tohoku University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于在模型中注入并随后去除“假后门”的方法，以消除未知后门攻击；

**💡 创新点**

创新点在于利用不同后门在相同任务下共享的内部激活机制，通过先引入已知触发器的假后门，再对该后门进行去除，间接降低未知后门的影响；

**🔧 技术方法**

采用的方法主要是先在模型中加入带有已知触发词的假后门，然后在含假触发词但期望输出为干净响应的数据集上进行微调，以消除假后门；

**📊 数据集**

实验使用了三种不同类型的后门攻击（包括破解、拒绝和情感操纵）在多种大型语言模型（如GPT、BERT、LLaMA等）上进行评估；

**📈 对比分析**

与代表性的防御方法相比，该方法在降低未知后门成功率的同时，保持了模型的实用性（保持高准确率/生成质量），实验结果显示其在攻击成功率下降与模型效用保持之间取得更优平衡；

**⚠️ 局限性**

局限性在于假后门和未知后门必须属于同一攻击目标（任务），否则共享机制可能不成立；若攻击目标未知，需构造多种假后门，增加计算成本并可能影响模型性能。

---

## 196. TAROT: Task-Adaptive Refinement of LLM-prior Graphs for Few-shot Tabular Learning

**arXiv ID:** 2606.11640 | [PDF](https://arxiv.org/pdf/2606.11640v1)

**作者:** Ruxue Shi `[一作]` (Jilin University), Xin Wang `[通讯]` (Jilin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于任务自适应图神经网络的少样本表格学习框架，通过LLM生成语义图并进行细化来显式建模特征交互。

**💡 创新点**

创新点在于：①利用LLM零样本推理构造语义先验图；②设计任务自适应语义图细化机制，既去除噪声边又补齐缺失的任务相关边；③结合统一语义节点编码器（USTNE）和先验学习正则，提升少样本下的泛化能力。

**🔧 技术方法**

核心技术包括LLM提示生成可执行代码得到邻接矩阵、统一语义节点编码（文本+数值嵌入）、任务自适应图细化（基于边得分的剪枝与增强）、图神经网络消息传递、先验学习损失和稀疏正则。

**📊 数据集**

使用11个真实世界数据集（8分类：Adult、Amazon、Blood、Credit-g、Diabetes、Heart、Communities、Myocardial；3回归：Abalone、Boston、Cholesterol）进行评估。

**📈 对比分析**

与传统少样本方法（SCARF、TabPFN、STUNT）及LLM驱动方法（In-context、TABLET、TabLLM、FeatLLM）对比，实验显示该方法在AUC/ RMSE上持续领先，取得SOTA成绩，且训练与推理效率较高。

**⚠️ 局限性**

局限性包括：①仍依赖LLM的可用性和成本；②语义图生成受LLM生成错误或幻觉影响；③对具有大量特征或缺乏有意义特征名的数据集效果尚未充分验证；④在极低样本（<4 shot）下性能提升有限。

---

## 197. Evaluating Bias in Phoneme-Based Automatic Speech Recognition Systems: An Analysis of IPA Transcription Models

**arXiv ID:** 2606.11639 | [PDF](https://arxiv.org/pdf/2606.11639v1)

**作者:** Catherine Bao `[一作]` (University of Utah), Neal Patwari `[通讯]` (University of Utah)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了两种开源 IPA 基础的语音识别模型（WhisperIPA 和 ZIPA）在多语言与人口统计组别上的性能差异，并提出了软 PER（Soft PER）评估指标以降低因语音相似性导致的误判。

**💡 创新点**

创新点包括：①提出 Soft PER 通过两级等价映射容忍细微音位差异，改进传统 PER 的公平性；②系统性分析了多语言及性别、年龄、种族、口音等人口特征对 IPA ASR 的偏差；③将两种主流模型的性能在 11 种语言及多样化数据集上进行对比。

**🔧 技术方法**

使用了基于 Transformer 的 WhisperIPA（74M 参数）和 ZIPA（300M 参数）模型，采用 G2P+ 生成 IPA 参考序列，并利用 AlloVera 与 PHOIBLE 建立软 PER 等价类；评估时采用标准 PER 与 Soft PER 两种指标。

**📊 数据集**

数据集包括 11 种语言的多语音语料（IPAPACK、MediaSpeech、WAXAL 等），以及包含人口统计信息的英语语料（CORAAL、EdAAC、SVC、WAXAL）。

**📈 对比分析**

通过比较两模型在标准 PER 与 Soft PER 下的误差率，发现 ZIPA 在所有语言上均优于 WhisperIPA，且 Soft PER 在多数语言和群体上显著降低误差；但两模型在不同语言与人口统计组的差异仍显著，且 Soft PER 并未根本改变模型排名。

**⚠️ 局限性**

限制包括：①使用自动 G2P 生成的 IPA 参考可能引入标注偏差，尤其对口音和方言产生高误差；② Soft PER 的等价映射基于英语优先，可能在非英语语言中产生过度归一化；③人口统计标签粗糙，缺乏细粒度的社会语言学信息，影响结果的细致解释。

---

## 198. Frozen Foundation-Model Embeddings Discard Small-Lesion Signal in Chest Radiography: Implications for Pre-Deployment Evaluation

**arXiv ID:** 2606.11606 | [PDF](https://arxiv.org/pdf/2606.11606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 199. Lung-R1: A Knowledge Graph-Guided LLM for Pulmonary Diagnostic Reasoning

**arXiv ID:** 2606.11675 | [PDF](https://arxiv.org/pdf/2606.11675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 200. Can Open-Source LLM Agents Replace Static Application Security Testing Tools? An Empirical Assessment

**arXiv ID:** 2606.11672 | [PDF](https://arxiv.org/pdf/2606.11672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 201. Vision-Language-Action Models Meet World Models: Embodied Agentic AI for Low-Altitude Wireless Networks

**arXiv ID:** 2606.11618 | [PDF](https://arxiv.org/pdf/2606.11618v1)

**作者:** Feibo Jiang `[一作]`, Naofal Al-Dhahir `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种面向低空无线网络（LAWNs）的“具身代理式 UAV 框架”，将 Vision‑Language‑Action (VLA) 模型、世界模型 (WM)、记忆与反思模块以及具身执行器结合，实现从多模态感知到执行命令的闭环自主决策。

**💡 创新点**

创新点在于：①首次将 VLA 与 WM 对接，形成可预测、可验证的动作生成与环境演化仿真；②引入记忆‑反思机制实现经验累积与策略自适应；③实现从语义指令到低维连续控制命令的端到端映射，提升实时性与可执行性。

**🔧 技术方法**

核心技术包括：OpenVLA‑7B VLA + LoRA 微调、Diffusion Transformer (DiT) + Mixture‑of‑Experts 的 WM、action‑conditioned adaptive layer normalization、记忆检索（Milvus）、反思解释（DeepSeek R1）、量化与量化压缩、边缘‑云协同推理。

**📊 数据集**

主要使用数据集：RaceVLA（FPV 图像 + 四维动作向量 + 语言指令）以及由 WM 生成的 50% 合成数据；实验中采用 70%/30% 训练/测试划分，并使用 UAV 动力学与信道模型进行仿真。

**📈 对比分析**

与 RT‑2‑X、OpenVLA 在三类泛化任务（目标生成、指令生成、环境生成）对比，VLA 模型在成功率（SR）上均最高；整合 WM 与记忆‑反思后，在多目标路径跟踪任务中完成率（CR）亦显著提升，优于仅 VLA 或仅 VLA+记忆‑反思的基线。

**⚠️ 局限性**

局限性包括：①VLA 推理时序延迟高，易出现抖动；②WM 在多步预测中累计误差随时间增长；③在线持续学习易引发灾难性遗忘与分布漂移；④在高速移动环境下信道状态更新频繁导致信息更新负担沉重。

---

## 202. Factions Within, Uncertain Across: Within-Document Reader Sub-Groups in Social Highlighting

**arXiv ID:** 2606.11613 | [PDF](https://arxiv.org/pdf/2606.11613v1)

**作者:** Kazuki Nakayashiki `[一作]` (Glasp Inc), Keisuke Watanabe `[通讯]` (Glasp Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在同一篇文档中读者高亮的聚合是否形成单一共识，揭示存在强烈的子群体结构；进一步探讨这些子群体是否在不同文档间保持稳定，结果未能确认；

**💡 创新点**

首次用边际保持曲线球（curveball）和区域保持置换（region‑preserving）无模型假设的置换检验，量化读者间的超随机一致性，分解为共享区块参与与细粒度读者特异性两部分；

**🔧 技术方法**

利用曲线球置换保持行列边际，区域保持置换保持读者在各段落内的标记数，计算二进制余弦相似度的差异；进行拆分-半相关检验评估跨文档稳定性；

**📊 数据集**

来自Glasp社交高亮平台的公开网页文章数据，筛选多读者密集文档（≥10读者），最终获得75篇用于内部结构分析，146篇用于跨文档稳定性分析；

**📈 对比分析**

与无结构、植入两组等合成基准对照验证统计量的灵敏度；结果显示内部子群体显著（z≈6.3，88%文档显著），共享区块占约40%，剩余约60%为细粒度读者特异性；跨文档稳定性检验因样本稀疏无统计显著，无法判断稳定性；

**⚠️ 局限性**

主要局限：跨文档稳定性检验功效不足；采样偏向高亮率高、可抓取的公开文章；区域保持置换使用等长句子块近似段落，可能高估细粒度信号；未对读者具体属性或文档主题做更精细控制；

---

## 203. Learning Instance-Adaptive Low-Rank Orthogonal Subspaces for Clothes-Changing Person Re-Identification

**arXiv ID:** 2606.11661 | [PDF](https://arxiv.org/pdf/2606.11661v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 204. Sparse probes and murky physics: a case study of interpretability challenges in a foundation model for continuum dynamics

**arXiv ID:** 2606.11657 | [PDF](https://arxiv.org/pdf/2606.11657v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 205. ARGUS: Stacked Multi-View Identity Mosaic Injection for Subject-Preserving Video Generation

**arXiv ID:** 2606.11670 | [PDF](https://arxiv.org/pdf/2606.11670v1)

**作者:** Zijie Meng `[一作]` (Peking University), Pengfei Wan `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Wan的框架，通过多视角身份马赛克注入，将身份参考从单点转为动态分布，并配合MLLM身份导演选择动态身份证据，实现主体保持的视频生成。

**💡 创新点**

创新点在于：①使用3×3堆叠马赛克构成负时间读写的身份记忆；②用多模态大语言模型生成冲突图和可读的身份摘要；③无跨对训练与大规模同片伪造增强相结合；④引入时间身份退火和自适应相似度引导，使文本与身份控制分离。

**🔧 技术方法**

技术包括Wan 2.1流匹配扩散模型、Qwen3‑VL身份导演、Stacked Multi‑View Identity Mosaic Injection、Temporal Identity Annealing、Adaptive Self‑Likeness Guidance、无跨对counterfactual训练、YawScore/OccScore评估。

**📊 数据集**

使用OpenS2V‑Eval Human‑Domain与公开人物基准HardID‑Celeb做评测，并在训练中使用同片视频与公开人物图像作为身份参考，进行大规模伪造增强。

**📈 对比分析**

与多种闭源与开源方法在OpenS2V和HardID‑Celeb上对比，OpenS2V总分提升4.18点（64.38），FaceSim 71.86，NexusScore 51.62，NaturalScore 79.14；HardID‑Celeb FaceSim 76.80，YawScore 70.80，OccScore 64.20，分别比最佳基线提升12.60/15.10点。

**⚠️ 局限性**

局限在于对Wan模型依赖强、负时间位置与读取仅身份的实现复杂；对非公开人物或非人类主体效果未知；在极端姿态、表情或背景变化下仍可能出现身份漂移；无跨对训练对真实多样性迁移存在一定限制。

---

## 206. A Robust Framework for Sybil Attack Detection in Vehicular Ad Hoc Networks

**arXiv ID:** 2606.11667 | [PDF](https://arxiv.org/pdf/2606.11667v1)

**作者:** Md. Sadmin Tahmid Khan `[一作]` (University of Dhaka), Mosarrat Jahan `[通讯]` (University of Dhaka)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于GPS轨迹构造与DBSCAN聚类的Sybil攻击检测框架，能够在车辆网络中快速识别伪造身份车辆。

**💡 创新点**

创新点在于：①使用高分辨率GPS连续轨迹代替传统V2V交互，消除协同伪造风险；②采用离散Fréchet距离衡量轨迹相似度；③通过DBSCAN自适应确定聚类阈值，消除人工调参；④不使用计算难度高的PoW，提升轻量化。

**🔧 技术方法**

技术包括GPS定位、离散Fréchet距离、DBSCAN密度聚类、SUMO交通仿真、OpenStreetMap地图数据。

**📊 数据集**

实验数据来自仿真生成的Dhaka城市区域，密集区车辆数160–200，稀疏区30–60，恶意车辆比例10–20%，每辆恶意车产生1–10 Sybil车辆。

**📈 对比分析**

与Baza等人提出的PoW基方案对比，指标FPR降低约68%–70%，FNR略高但仍在4%–6%范围；检测率保持94%–98%；准确率与F1接近96%–97%；检测时间在密集区提升约80%，稀疏区提升约43%。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实部署实验；仍需GPS覆盖，GPS伪造未完全防御；对RSU误报或失效情形未做完整评估；在极稀疏或高噪声GPS环境下聚类可能失效。

---

## 207. Neural-Parameterized Cellular Automata for Wildfire Spread

**arXiv ID:** 2606.11676 | [PDF](https://arxiv.org/pdf/2606.11676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 208. Are LLMs Bad at Moral Reasoning?

**arXiv ID:** 2606.11635 | [PDF](https://arxiv.org/pdf/2606.11635v1)

**作者:** Menghang Zhu `[一作]` (Renmin University of China), Seth Lazar `[通讯]` (Johns Hopkins University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新分析 MoReBench 数据集，展示大型语言模型（LLM）在生成评价标准（rubric）方面能覆盖 83–89% 的人类标准，并通过改写人类标准提高模型分数至约 90% 以上。

**💡 创新点**

创新点在于：①把 LLM 从评估者转为标准生成者，证明其能写出与人类相近的评价准则；②利用双重方法（余弦相似度+LLM 判别）识别模型独有的道德考量；③通过“通用性”改写人类标准，显著提升基准得分，说明原评估存在偏差。

**🔧 技术方法**

主要技术：大语言模型（Claude Opus, Gemini, GPT‑5.4 等）生成 rubrics；自动评估者 GPT‑OSS‑120B 对 Rubric 满足度进行二元判断；余弦相似度计算和 LLM 再评估结合的两步筛选流程。

**📊 数据集**

使用 MoReBench 数据集（500 公开道德困境 + 10–30 条人类编写的评价标准），以及基于这些案例生成的 LLM 评价标准。

**📈 对比分析**

比较方法：①对 100 案例，分别让人类与 LLM 写 Rubric，再用同一评估者判断两者是否表达相同道德点；②对 500 案例，计算 LLM 与人类 Rubric 的覆盖率、独特性；③改写人类 Rubric 后重新评估模型得分。结果显示：模型生成的 Rubric 覆盖率达 83–89%，独特性约为人类的 2.26 倍；改写后模型平均得分从 70.9 提升至 89.3，提升幅度 18–20 分。

**⚠️ 局限性**

局限性：①评估仍基于预先编写的结构化困境，无法覆盖真实世界复杂情境；②改写人类标准依赖 LLM 判断，可能引入偏差；③仅检验了评价标准的生成与匹配，未直接评估模型在实际推理过程中的逻辑一致性与多样性；④对模型能力的解释仍受“评价标准是否足够原子”的争议影响。

---

## 209. Organize then Retrieve: Hierarchical Memory Navigation for Efficient Agents

**arXiv ID:** 2606.11680 | [PDF](https://arxiv.org/pdf/2606.11680v1)

**作者:** Hao-Lun Hsu `[一作]` (Duke University), Yuxiong He `[通讯]` (Snowflake AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种层次化的组织与检索记忆代理（HORMA），通过将工作记忆分为结构化的记忆管理器和导航式检索器，外部化并维护一个文件系统样式的层次化工作空间，从而解决大规模长时序任务中的上下文瓶颈。

**💡 创新点**

核心创新点在于：① 明确区分记忆构造与检索两大功能，并在同一工作空间内实现解耦；② 记忆管理器通过递归技能演化（skill evolution）实现无监督的结构化学习；③ 检索器采用强化学习（GRPO）和基于证据的奖励，直接在文件系统层次中导航，显著提升检索精度与效率。

**🔧 技术方法**

使用技术包括：大型语言模型（Claude Sonnet 4.5、Qwen 3.5 4B）做主推理、记忆管理与检索；Bash 工具与可执行文件系统命令实现工作空间操作；RL（GRPO）+ 证据奖励训练检索策略；对比学习与差异分析用于技能迭代；多任务学习以提升跨域泛化。

**📊 数据集**

实验数据集涵盖三大长时序基准：ALFWorld（6 类任务 134 任务），LoCoMo（10 场景对话 519 QA 例），LongMemEval（367 例多类型长对话）。

**📈 对比分析**

与传统截断、滑动窗口、相似度检索、ReSum、Acon、Fold、HIAGENT、Mem0、A-MEM 等方法对比，HORMA 在 ALFWorld 的小/大上下文窗口下分别取得 56.7% / 73.9% 的成功率，同时在 token 方面显著低于 baselines；在 LoCoMo 和 LongMemEval 中，HORMA 的 token 消耗保持在 1000 以内，且整体 LLM‑judge 分数超过大多数基线，尤其在时间敏感任务上检索错误率下降 20%+。

**⚠️ 局限性**

局限性包括：① 记忆构造高度依赖强大的 LLM，弱模型难以构建有效结构；② 需要预先训练的 RL 检索器，训练成本高；③ 对极大规模任务的实时在线学习尚未实现；④ 系统在多任务迁移时仍可能受到特定领域结构偏差的影响。

---

## 210. Runtime Skill Audit: Targeted Runtime Probing for Agent Skill Security

**arXiv ID:** 2606.11671 | [PDF](https://arxiv.org/pdf/2606.11671v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 211. IAPO: Input Attribution-Aware Policy Optimization for Tool Use in Small Multimodal Agents

**arXiv ID:** 2606.11652 | [PDF](https://arxiv.org/pdf/2606.11652v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 212. Probabilistic Salary Prediction with Graph Attention Networks and a Mixture Density Network

**arXiv ID:** 2606.11663 | [PDF](https://arxiv.org/pdf/2606.11663v1)

**作者:** Zhipei Qin `[一作]` (Leiden Institute of Advanced Computer Science), F. W. Takes `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建多关系图注意力网络与混合密度网络，预测职位薪资的完整概率分布

**💡 创新点**

同时编码层级与语义相似性边，使用边感知 GAT 提取上下文嵌入，并结合 MDN 输出多峰分布

**🔧 技术方法**

多关系图注意力网络（Edge-aware GAT）、混合密度网络（MDN）、句子Transformer嵌入、优先级层级选择机制

**📊 数据集**

约 104 万条荷兰 Jobdigger 职位数据，包含地理、职业和行业属性

**📈 对比分析**

与无图 MLP-MDN 基线对比，GAT-MDN 在 NLL 和 MSE 上显著更好，显示显著的性能提升

**⚠️ 局限性**

仅处理离散属性，未加入经验等数值特征，且模型解释性有限，未来可扩展到更丰富特征和更灵活的分布估计

---

## 213. TreeSeeker: Tree-Structured Trial, Error, and Return in Deep Search

**arXiv ID:** 2606.11662 | [PDF](https://arxiv.org/pdf/2606.11662v1)

**作者:** Zhuofan Shi `[一作]` (Microsoft), Saravan Rajmohan `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个在深度搜索任务中使用分支-返回搜索的推理时框架，即TreeSearch与TreeMem，用以改进多步网络搜索的控制策略。

**💡 创新点**

核心创新在于通过文本UCB信号（价值、未确定性、风险）在操作层面实现显式的分支-返回控制，从而平衡继续、探索与剪枝的决策。

**🔧 技术方法**

技术手段包括大型语言模型、TreeSearch控制器、TreeMem分支内状态存储、文本UCB决策机制以及工具调用和基于树的搜索框架。

**📊 数据集**

实验数据集涵盖了公开的三大深度搜索基准：XBench-DeepSearch、BrowseComp和BrowseComp-ZH。

**📈 对比分析**

与多种开源基线（Flash-Searcher、IterResearch、Tongyi-DeepSearch等）进行对比，实验结果显示在XBench-DS、BrowseComp和BrowseComp-ZH分别达到了56.3、47.0和43.0分，显著优于对照系统。

**⚠️ 局限性**

局限性包括仅适用于文本搜索（不支持多模态），额外的控制与内存开销，以及对外部搜索结果的质量和偏见依赖，无法完全保证检索源的可靠性。

---

## 214. Tree-Structured Orthonormal Decomposition of the Aitchison Simplex

**arXiv ID:** 2606.11646 | [PDF](https://arxiv.org/pdf/2606.11646v1)

**作者:** Daisuke Yamada `[一作]` (University of Wisconsin Madison), Vikas Singh `[通讯]` (University of Wisconsin Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出PolyILR，一种针对任意树结构（包括多叉树）的正交标准正交分解，构造了与树拓扑对齐的Aitchison空间坐标；

**💡 创新点**

创新点在于利用加权局部几何与全局展开，实现了多层次、无二进化、可唯一确定的树对齐ILR基；

**🔧 技术方法**

采用Aitchison几何、Helmert对比、加权内积、Gram-Schmidt正交化以及坐标变换等技术；

**📊 数据集**

在宏基因组学（Human Microbiome Project、curatedMetagenomicData）和单细胞数据（Immune Single-Cell Omics）上进行实验；

**📈 对比分析**

与CLR、PhILR以及基于树的无二进化方法比较，PolyILR在特征选择稳定性、可解释性和多尺度推断上均优于传统方法，分类/回归性能保持相近；

**⚠️ 局限性**

局限包括需预先给定确定树结构、无法处理树的不确定性、对零值处理敏感以及跨节点比较不直接可比。

---

## 215. 3-Key-Input: Exploring the Theoretical Minimum Keys for Text Entry

**arXiv ID:** 2606.11642 | [PDF](https://arxiv.org/pdf/2606.11642v1)

**作者:** Naoki Kimura `[一作]` `[通讯]` (LY Corporation), Naoki Kimura (LY Corporation)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在只有2-5个物理键的模糊键盘上，借助现代语言模型实现文本输入；

**💡 创新点**

创新点在于系统评估极端键数与语言模型解码的组合，证明3键加GPT‑4o即可获得接近理论极限的低错误率；

**🔧 技术方法**

采用键映射策略（布局、频率、最差案例）+ trie候选生成+句子级语言模型解码（GPT‑2 beam、GPT‑4o选择），并给出本地LM基准；

**📊 数据集**

使用自制300句英文语料（100句商务、100句对话、100句技术），并通过哈希、嵌入检验防止预训练泄漏；

**📈 对比分析**

通过比较不同键数、映射和解码器的CER、WER、BLEU、BERTScore等指标，发现3键+GPT‑4o的CER为9.46%/WER 12.20%，比2键提高约59%；键数上升至5键提升有限；映射选择影响很小；

**⚠️ 局限性**

局限性包括仅离线评估、未考虑实时交互与硬件约束、对技术词汇的偏好差异、对隐私与离线LM的需求以及预训练重叠风险。

---

## 216. Precision-Aware Illumination-Disentangled Vision Transformer for Spacecraft 6D Pose Estimation

**arXiv ID:** 2606.11619 | [PDF](https://arxiv.org/pdf/2606.11619v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 217. DeMix: Debugging Training Data with Mixed Data Error Types by Investigating Influence Vectors

**arXiv ID:** 2606.11616 | [PDF](https://arxiv.org/pdf/2606.11616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 218. SpAArSIST: Sparsified AASIST for Efficient and Reliable Anti-Spoofing

**arXiv ID:** 2606.11674 | [PDF](https://arxiv.org/pdf/2606.11674v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 219. The Hidden Cost of Pairwise Verification in Synthetic Speech Source Tracing

**arXiv ID:** 2606.11666 | [PDF](https://arxiv.org/pdf/2606.11666v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 220. Bergson: An Open Source Library for Data Attribution

**arXiv ID:** 2606.11660 | [PDF](https://arxiv.org/pdf/2606.11660v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 221. Can AI Reason Like an Urban Planner? Benchmarking Large Language Models Against Professional Judgment

**arXiv ID:** 2606.11678 | [PDF](https://arxiv.org/pdf/2606.11678v1)

**作者:** Yijie Deng `[一作]` (Tongji University), Wenjia Zhang `[通讯]` (Tongji University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了UPBench框架，评估25个大语言模型在城市规划四个知识支柱（原则、跨学科整合、治理、实践）与五个认知层级（记忆、理解、应用、分析、评价）上的表现。

**💡 创新点**

创新点在于：① 设计了与规划专业知识结构对齐的4×5认知矩阵；② 引入四种“认知诊断”（监管幻觉、概念混淆、复杂问题停滞、实践智慧缺失）揭示AI在法规、概念、价值和实践方面的局限；③ 采用双轨评估（自动评分+专家面板）提供系统可靠的性能评测。

**🔧 技术方法**

技术包括：证据中心设计（ECD）构建评测模型、Prompt工程训练LLM评估者、自动化评分与专家评审相结合的评估流程，以及大语言模型（如Claude、Gemini、Llama等）作为被测对象。

**📊 数据集**

数据集来源于美国AICP认证题库、中国全国注册城市规划师考试题目以及多所高校规划课程教材，整合生成405道跨国情境的评测题集。

**📈 对比分析**

通过对每个模型在记忆、理解、应用、分析、评价五层次上打分，结果显示最强模型Claude‑haiku平均得分84.5%，最弱Qwen2.5‑0.5B仅38.8%；模型表现呈非单调认知曲线，在分析层级表现优于理解层级，反映了规划知识的特殊认知结构。

**⚠️ 局限性**

局限性包括：① 仅评估文本任务，未覆盖空间视觉与现场实践；② 仅评估公开/开放权重模型，未涵盖最新旗舰专有模型；③ 双轨评估仍存在人工评分一致性与自动化评分偏差；④ 数据集主要覆盖美中两国，跨文化泛化需要进一步扩展。

---

## 222. Structure-Preserving Neural Surrogates with Tractable Uncertainty Quantification

**arXiv ID:** 2606.11650 | [PDF](https://arxiv.org/pdf/2606.11650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 223. Learning by Chatting? Investigating the Impact of Generative AI on Information Seeking and Learning

**arXiv ID:** 2606.11669 | [PDF](https://arxiv.org/pdf/2606.11669v1)

**作者:** Shravika Mittal `[一作]` (Georgia Institute of Technology), Q. Vera Liao `[通讯]` (University of Michigan)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项8天的纵向现场实验中，对照使用ChatGPT与Google搜索的受试者，检验了生成式AI工具对非正式学习信息寻求行为与学习成果的影响。

**💡 创新点**

创新点在于首次从信息寻求阶段模型出发，系统分析了GenAI导致的认知负荷提升、探索范围收窄与信息偏差，并揭示了生成式AI对学习的负面影响。

**🔧 技术方法**

采用了基于Ellis信息寻求阶段模型的定性编码、日记收集、学习成果问卷以及统计检验（Welch t检验、Fisher精确检验）等技术。

**📊 数据集**

使用了来自Prolific的40名受试者（ChatGPT 14人、Google 21人）进行日记记录和知识测验，构成了实验数据集。

**📈 对比分析**

通过比较两组学习成果得分、单题正确率以及日记编码频次，发现ChatGPT组在关键性学习任务上学习成绩更低，且信息寻求行为更为被动，差异在部分题目上具有统计学意义。

**⚠️ 局限性**

局限包括样本量有限、受试者自我报告的日记可能遗漏行为、实验任务未完全自由化、未能建立因果关系、以及仅聚焦在营养与膳食规划这一单一领域。

---

## 224. MoGeFlow: Flowing Through Motion Codebook Geometry for Text-to-Motion Generation

**arXiv ID:** 2606.11656 | [PDF](https://arxiv.org/pdf/2606.11656v1)

**作者:** Pengcheng Fang `[一作]` (University of Southampton), Dongjie Fu `[通讯]` (Nanjing University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MoGeFlow，利用向量量化运动代码的几何结构进行文本到运动生成；

**💡 创新点**

将运动代码视为携带局部运动几何的连续嵌入空间，构建连续流模型在该空间中生成，再终端投影为离散代码；

**🔧 技术方法**

使用PartVQ预训练分组量化器、文本条件流匹配（flow‑matching）模型、终端最近邻投影、无分类器自由引导；

**📊 数据集**

在HumanML3D、KIT‑ML和大规模MotionMillion数据集上进行训练与评估；

**📈 对比分析**

相较于传统离散预测或连续轨迹生成方法，在HumanML3D、KIT‑ML及MotionMillion上均获得R‑Precision、FID、Multi‑Modal Distance等指标的最优或近优性能；

**⚠️ 局限性**

仅使用冻结量化器和最近邻投影，可能限制多模态生成的多样性与控制灵活性；

---

## 225. Adapting Vision-Language Models from Iconic to Inclusive for Multi-Label Recognition Without Labels

**arXiv ID:** 2606.11626 | [PDF](https://arxiv.org/pdf/2606.11626v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 226. The Long Tail, Not the Front Page: Cold-Start Prediction of Crowd Highlight Salience

**arXiv ID:** 2606.11654 | [PDF](https://arxiv.org/pdf/2606.11654v1)

**作者:** Kazuki Nakayashiki `[一作]` (Glasp Inc), Keisuke Watanabe `[通讯]` (Glasp Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用真实读者高亮数据训练模型，在新文档无高亮标记时预测哪些句子会被聚集读者标记。

**💡 创新点**

证明在真实读者群体下，即使不使用零射击语言模型或无监督提取，也能超越传统的“先读者位置”基准，并且该优势在低流行度文档上尤为显著。

**🔧 技术方法**

使用句子级别的预训练句子嵌入、位置/词法特征、语义上下文特征的逻辑回归排名器，并进行数据增强与特征工程。

**📊 数据集**

基于Glasp平台的高亮数据，筛选至少20位高亮者的“密集”文档，约284份评估文档和389份训练文档。

**📈 对比分析**

对比随机、lead、未监督中心化/LexRank基线，M3模型平均精度提升+0.044，precision@3从0.25提升至0.39，模型在69%文档上超过lead。

**⚠️ 局限性**

受限于只评估最终有至少20高亮者的文档，真实零读者情况未知；数据获取、文本匹配漂移、近似重复、标签稀薄导致可靠性下降。

---

## 227. TimeRouter: Efficient and Adaptive Routing of Time-Series Foundation Models

**arXiv ID:** 2606.11625 | [PDF](https://arxiv.org/pdf/2606.11625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 228. Improving Cross-Format Robustness in Language Models with Multi-Format Training

**arXiv ID:** 2606.11643 | [PDF](https://arxiv.org/pdf/2606.11643v1)

**作者:** June M. Liu `[一作]` (Ant Group), Jun Zhou `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并使用了一个跨格式对齐的多格式QA语料库，评估并通过多格式监督训练提升大型语言模型的跨格式鲁棒性。

**💡 创新点**

提出了项目式的跨格式鲁棒性评估与训练框架，并发现仅扩充约30%项目的多格式数据即可恢复大部分全格式扩展的收益。

**🔧 技术方法**

采用了多模型生成、审核与仲裁的语料构建流程，使用了Deterministic Rule + LLM-as-a-Judge 的自动评分；在GLM4、Llama-3.1等模型上进行全参数SFT。

**📊 数据集**

使用了六大主流MCQ基准（C‑Eval、CMMLU、MMLU、MMLU‑Pro、GPQA、SuperGPQA）构造的多格式数据，以及BIG‑Bench Hard 作为外部验证。

**📈 对比分析**

与单一MCQ监督、规模匹配的MCQ监督对比，发现多格式监督在pass@4、pass^4和跨格式鲁棒性指标上提升约10‑30%，且随机/目标式扩展在不同指标上各有优势，性能提升显著。

**⚠️ 局限性**

仅覆盖四种文本答案格式、仅评估两类模型家族、评估依赖LLM‑Judge 评分、未覆盖多模态或开放式交互场景，可能导致结果无法泛化到更广泛的实际应用。

---

## 229. Information-Theoretic Decomposition for Multimodal Interaction Learning

**arXiv ID:** 2606.11614 | [PDF](https://arxiv.org/pdf/2606.11614v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 230. Physics-Distilled Neural Network enabled by Large Language Models for Manufacturing Process-Property Predictive Modeling

**arXiv ID:** 2606.11605 | [PDF](https://arxiv.org/pdf/2606.11605v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 231. Adv-TGD: Adversarial Text-Guided Diffusion for Face Recognition Impersonation Attacks

**arXiv ID:** 2606.11615 | [PDF](https://arxiv.org/pdf/2606.11615v1)

**作者:** Omid Ahmadieh `[一作]` (University of South Florida), Nima Karimian `[通讯]` (University of South Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于Stable Diffusion 2.1的Adv-TGD框架，利用每个源‑目标对的LoRA微调、面部局部热图掩模以及文本引导的复合目标，生成可骗过多种面部识别模型的逼真冒充图像。

**💡 创新点**

创新点在于：① 采用单步LoRA细调实现每对身份的快速定制化；② 设计Salience‑Guided Semantic Mask (SGSM)将面部解剖先验与识别梯度融合，实现高精度身份定位；③ 统一的对抗目标包含阈值化余弦门限、方向对齐、源抑制和后期文本正则，兼顾攻击强度与视觉真实性；④ 自动化文本提示生成（使用LLaVA）减少人工干预。

**🔧 技术方法**

核心技术包括Stable Diffusion 2.1、LoRA适配器、SGSM掩模、复合对抗损失（掩模MSE、身份阈值损失、方向一致性、源抑制、CLIP文本对齐）、LLaVA文本生成、后处理逆变换+无缝克隆。

**📊 数据集**

主要使用 CelebA‑HQ、FFHQ、LADN 进行人脸攻击实验，ImageNet 用于评估对通用分类器的攻击，此外在 FLUX.1 以及 Stable Diffusion 1.5 上验证跨模型适用性。

**📈 对比分析**

在黑盒攻击设置下，对 IR152、IRSE50、MobileFace、FaceNet 的平均 ASR 为 85.90%，比语义基线 Adv‑CPG 高 6.25%，比 DiffAIM 高 3%，比 P3‑Mask 高 16%；在商业 Face++ API、LADN 数据集上亦保持领先；视觉质量方面 PSNR 27.15 dB、SSIM 0.981，FID 27.26，显示出与现有方法相比兼具高攻击成功率和逼真度。

**⚠️ 局限性**

局限性包括：① 需要前置人脸对齐与后处理，处理流程相对复杂；② 对极端姿态或光照变化的鲁棒性尚待进一步验证；③ 目前主要聚焦人脸攻击，虽可迁移至通用分类，但在更广泛的视觉任务上需重新设计掩模与目标；④ 依赖强大预训练扩散模型，若模型更新可能影响攻击效果。

---

## 232. SARA: A Dual-Stream VAE for High-Fidelity Speech Generation via Integrating Semantic and Acoustic Representations

**arXiv ID:** 2606.11611 | [PDF](https://arxiv.org/pdf/2606.11611v1)

**作者:** Peijie Chen `[一作]` (Xiamen University), Lin Li `[通讯]` (Xiamen University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种名为SARA的双流变分自编码器，用于零样本文本到语音的生成，融合冻结的SSL语义编码器与残差声学编码器，生成64维、50Hz的压缩潜在空间；

**💡 创新点**

直接将冻结的SSL语义锚点与残差声学分支耦合，消除了复杂正则化，构建了兼顾语义一致性与声学细节的高效潜在表示；

**🔧 技术方法**

采用变分自编码器框架，结合多尺度mel谱损失、对抗式多周期/多频带判别器、残差CNN+LSTM声学编码器、冻结的W2v‑BERT 2.0语义编码器以及HiFi‑GAN解码器；

**📊 数据集**

使用LibriTTS、LibriHeavy（≈50,000小时）作为训练数据，LibriSpeech test‑clean集用于重建与零样本TTS评估；

**📈 对比分析**

与Vocos、Semantic‑VAE、F5‑TTS等基线对比，SARA在PESQ、STOI、UTMOS、WER和SIM上均优于对照模型，且在8步流匹配下实现了0.079的RTF，接近32步基线的质量；

**⚠️ 局限性**

目前仅在英文数据上验证，跨语言扩展尚未研究；双流结构导致推理成本略高，且对预训练SSL模型的依赖可能限制迁移能力。

---

## 233. Motion Reinforces Appearance: RGB-Skeleton Gated Residual Fusion for Micro-Gesture Online Recognition

**arXiv ID:** 2606.11645 | [PDF](https://arxiv.org/pdf/2606.11645v1)

**作者:** Jialin Liu `[一作]` (Hefei University of Technology), Yanbin Hao `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DyFADet+双流RGB‑骨架框架，实现微手势在线识别。

**💡 创新点**

采用门控残差融合模块自适应注入骨架运动，避免传统拼接噪声。

**🔧 技术方法**

基于VideoMAEv2‑g视觉编码、动态特征聚合DyFADet、Conv+Transformer骨架投影以及门控残差融合。

**📊 数据集**

在SMG微手势数据集（40位受试者、17类）上进行训练与评估。

**📈 对比分析**

与单模态RGB、骨架及直接拼接对比，F1提升至40.88，排名第二（仅次于XInsight Lab的43.81）。

**⚠️ 局限性**

对骨架估计噪声仍依赖门控，且仅在SMG上验证，跨数据集泛化需进一步验证。

---

## 234. LUCID: Learning Embodiment-Agnostic Intent Models from Unstructured Human Videos for Scalable Dexterous Robot Skill Acquisition

**arXiv ID:** 2606.11628 | [PDF](https://arxiv.org/pdf/2606.11628v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 235. TouchThinker: Scaling Tactile Commonsense Reasoning to the Open World with Large-scale Data and Action-aware Representation

**arXiv ID:** 2606.11637 | [PDF](https://arxiv.org/pdf/2606.11637v1)

**作者:** Kailin Lyu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Shuicheng Yan `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了百万规模的多源触觉视觉数据集 TouchThinker-1M，并提出了开源触觉‑语言推理框架 TouchThinker。

**💡 创新点**

创新点在于 action‑aware 触觉建模机制与两阶段对齐训练范式，实现了更高效、跨传感器的触觉表示与推理。

**🔧 技术方法**

采用了 ViFi‑CLIP 预训练模型、question‑guided token fusion、Gaussian Temporal MoE、LLM（Qwen2.5）以及 LoRA 微调技术。

**📊 数据集**

使用了 TouchThinker-1M（约1M帧）、VTV-150K、Octopi 等公开数据以及自建的 TouchThinker-Bench。

**📈 对比分析**

通过与 VTV‑LLM、Octopi、LLaVA 等模型在属性预测与基本推理子任务上的对比，TouchThinker 在多项指标上提升约5–10%，在跨传感器与开放世界测试中表现更稳健。

**⚠️ 局限性**

局限包括属性标注范围有限、主要针对短时触碰、模型规模大不易部署等。

---

## 236. Multi-Agent Reasoning with Adaptive Worker Allocation for Stance Detection

**arXiv ID:** 2606.11609 | [PDF](https://arxiv.org/pdf/2606.11609v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 237. Architecture-Aware Reinforcement Learning Makes Sliding-Window Attention Competitive in Math Reasoning

**arXiv ID:** 2606.11634 | [PDF](https://arxiv.org/pdf/2606.11634v1)

**作者:** Kai Liu `[一作]` (Tongji University), Kai Chen `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将预训练的全局自注意力模型转换为滑窗自注意力（SWA）模型，并通过强化学习进一步调整生成策略，以提升在数学推理任务中的表现。

**💡 创新点**

证明强化学习能显著缩小SWA与SA之间的性能差距，并通过实验验证数据-架构不匹配导致SFT单独使用时的性能瓶颈。

**🔧 技术方法**

使用了滑窗自注意力架构、架构无关的监督微调（SFT）、基于CISPO的强化学习（RL）以及信息局部度量等技术。

**📊 数据集**

主要使用数学推理基准数据集，包括 AIME24、AIME25、AMC、MATH500、OlympiadBench 以及 AceReason-Math，用于训练和评估。

**📈 对比分析**

与同基线SA模型直接对比；SFT后SWA仍落后于SA，但RL后SWA8k 的平均准确率与SA几乎持平，SWA4k 在相同训练时长下也能匹配SA；SWA 在推理速度和显存占用上显著优于SA。

**⚠️ 局限性**

研究仅聚焦于数学推理，未验证更大规模模型（如200B或1T）或其它任务场景；且SWA在极小窗口（如2k）下仍存在一定性能差距。

---

## 238. Sovereign Assurance Boundary: Certificate-Bound Admission for Agentic Infrastructure

**arXiv ID:** 2606.11632 | [PDF](https://arxiv.org/pdf/2606.11632v1)

**作者:** Jun He `[一作]` (OpenKedge), Deying Yu `[通讯]` (OpenKedge)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Sovereign Assurance Boundary（SAB），为非确定性代理在基础设施中提供证书绑定的运行时录入与执行边界

**💡 创新点**

创新点在于将代理提议编译为类型化执行合同、绑定实时证据哈希、结合后果评分实现路径路由，并通过签名证书与可撤销、可重放的 broker 机制实现系统级授权

**🔧 技术方法**

采用Go实现，使用OPA/Rego进行策略评估、Ed25519签名、PostgreSQL附加链式日志、SQA多验证器网格与证书链验证、以及基于时间戳与撤销epoch的状态检查

**📊 数据集**

实验使用约500条执行合同（云操作、CI/CD、数据治理），共2500次录入尝试，合同来自真实云运营、自动化部署与日志导出场景

**📈 对比分析**

与IAM、人工审批、仅日志、LLM自评及SQA单独对比；PolicyOnly录入延迟<5ms，SQA约185/380ms；撤销传播<100ms；误录入率0.4%；人工审批率下降≈68%；证书存储平均3.4KB

**⚠️ 局限性**

局限包括后果评分误判、证据过期与不完整、验证器可靠性衰退、实验仅限单机工作负载、应急绕过风险以及对签名密钥、日志与 broker 的高度依赖

---

## 239. Blind Dexterous Grasping via Real2Sim2Real Tactile Policy Learning

**arXiv ID:** 2606.11767 | [PDF](https://arxiv.org/pdf/2606.11767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 240. UR-BERT: Scaling Text Encoders for Massively Multilingual TTS Through Universal Romanization and Speech Token Prediction

**arXiv ID:** 2606.11681 | [PDF](https://arxiv.org/pdf/2606.11681v1)

**作者:** Sangmin Lee `[一作]` (Yonsei University), Hong-Goo Kang `[通讯]` (Yonsei University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 UR‑BERT，一个基于罗马化文本的预训练 TTS 编码器，支持 495 种语言，并在 TTS 任务中显著提升了语音自然度与可懂度。

**💡 创新点**

① 用罗马化代替传统 G2P，突破了 100 语言左右的覆盖瓶颈；② 引入语音标记预测（STP）作为辅助任务，将多模态语音知识蒸馏到文本表示中，显著提升音素级对齐。

**🔧 技术方法**

使用 BERT‑base Transformer 结构，字符级分词器；采用 S3M 作为教师模型提取语音特征；通过 CTC 强制对齐和 k‑means 量化生成离散语音标记；在预训练中联合 MLM 与 STP 目标。

**📊 数据集**

预训练数据来自三大 ASR 语料库：FLEURS（102 种语言）、Common Voice（131 种语言）和 Omnilingual ASR（348 种语言），共约 13K 小时、800 万句；下游 TTS 评估数据涵盖 11 种语言（高低资源混合），VITS 作为基线模型。

**📈 对比分析**

与 m‑PLBERT、XPhoneBERT 等基线进行比较，使用 MOS、UTMOS、CER、MCD、Log‑F0 RMSE 等主客观指标。UR‑BERT 在高、低资源环境下均实现最高 MOS 与最低 CER，尤其在低资源语种的自然度与可懂度上提升显著；即使在未见语种的零样本设置下亦保持优势。

**⚠️ 局限性**

局限性包括：① 罗马化可能导致同音异义词的歧义，影响音素细粒度；② 需要先将文字转写为罗马化，仍需语言学专家或工具；③ 训练对 495 种语言的覆盖虽大，但仍不及全语种覆盖，某些稀有语言缺少罗马化资源；④ STP 依赖 S3M 教师的质量，若教师模型偏差会影响蒸馏效果。

---

## 241. Capacity-Constrained Online Convex Optimization with Delayed Feedback

**arXiv ID:** 2606.11711 | [PDF](https://arxiv.org/pdf/2606.11711v1)

**作者:** Alexander Ryabchenko `[一作]` (University of Toronto), Daniel M. Roy `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在容量受限（最多只能跟踪 C 个待处理回合）且反馈有延迟的在线凸优化与带反馈的场景，并给出了相应的子线性误差上界。

**💡 创新点**

创新点：①首次在容量受限环境下给出最优的延迟误差理论；②将问题归约为加权延迟 OCO 并设计 DW‑FTRL 与 DW‑FTBL 两种基准算法；③引入代理延迟调度器（Proxy‑Delay Scheduler），通过重要性加权实现对未观测反馈的补偿；④利用概率分析和倍增技巧控制跟踪集合饱和概率，完成整个框架的理论分析。

**🔧 技术方法**

使用的技术：延迟加权在线学习框架、重要性采样、代理延迟调度、FTRL 与 FTBL 的延迟变体、单点梯度估计、Chernoff 上界、两维倍增技巧、凸/强凸分析。

**📊 数据集**

本工作为理论工作，未使用任何实验数据集。

**📈 对比分析**

在理论评估上：当 C ≥ log T 时，首阶 OCO 的误差与无容量约束的最佳延迟误差相差对数因子；BCO 的误差在维度项中出现 (1+C)^{1/4} 或 (1+C)^{1/3} 的衰减，保持子线性；整体实现了与传统无约束算法相当的性能。

**⚠️ 局限性**

局限性：① BCO 的凸损失在延迟项上仅得到 √T 的上界，尚未达到最优 √σ；② 代理延迟分布需要知道最大延迟或最大积压；③ 只针对可观测的随机对手，无法直接推广到自适应对手。

---

## 242. ICA Lens: Interpreting Language Models Without Training Another Dictionary

**arXiv ID:** 2606.11722 | [PDF](https://arxiv.org/pdf/2606.11722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 243. MedCTA: A Benchmark for Clinical Tool Agents

**arXiv ID:** 2606.11702 | [PDF](https://arxiv.org/pdf/2606.11702v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 244. SVoT: State-aware Visualization-of-Thought for Spatial Reasoning via Reinforcement Learning

**arXiv ID:** 2606.11770 | [PDF](https://arxiv.org/pdf/2606.11770v1)

**作者:** Chao Lei `[一作]` (University of Melbourne), Nir Lipovetzky `[通讯]` (University of Melbourne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于强化学习的框架SVoT，用于多模态大语言模型在格子环境中的多跳空间推理，生成可验证的中间状态和对应的可视化图像，并将状态转换过程显式化为推理链。

**💡 创新点**

创新点包括：① 将中间状态结构化为〈动作、状态描述〉；② 在生成中插入转移推理链，使模型能够显式验证动作前提与效果；③ 通过细粒度奖励（状态、视觉、推理）以及Group Relative Policy Optimization (GRPO) 对推理链进行强化学习，显著提升可靠性；④ 构建五个包含多对象交互和数值推理的新格子环境（Maze、FrozenLake、Sokoban、Pacman、Gather），支持量化验证。

**🔧 技术方法**

技术手段包括：使用Anole‑7B作为统一的文本-视觉自回归模型；先做5轮监督微调（SFT）再进行5轮GRPO；奖励设计为状态准确率 r_z、视觉相似度 r_v 与推理链匹配率 r_c；使用LoRA、AdamW、TRL库等实现训练；评估采用free‑response与classification两种输出格式。

**📊 数据集**

数据集：自构造的五个格子域，每个域在不同网格尺寸下生成 500 训练实例（100图 + 5序列）和 120 评测实例（ID/OOD），网格尺寸从 4×4 至 7×7，涵盖单步动作、多步动作、数值计数等多种推理需求。

**📈 对比分析**

对比方法：MVoT（基线）、GPT‑4o、Anole T‑CoT；SVoT分两种变体 SVoT_o（ORM）与 SVoT_p（PRM）。实验显示 SVoT_p 在大多数域及 OOD/自由响应设置下均取得最优，尤其在 Sokoban 4×4 上比 MVoT 提升 65% 绝对准确率；与 GPT‑4o 相比在多跳、数值推理任务上表现更好；GRPO 优于单纯 SFT。

**⚠️ 局限性**

局限性：对 Gather 这类多步动作与多颜色计数的任务仍显弱，球的收集准确率随网格增大显著下降；OOB 性能仍低于 ID；训练耗时长（10 天），对资源要求高；目前仍未在真实连续空间或更复杂机器人任务上验证。

---

## 245. Acoda: Adversarial Code Obfuscation for Defending against LLM-based Analysis

**arXiv ID:** 2606.11755 | [PDF](https://arxiv.org/pdf/2606.11755v1)

**作者:** Hongzhou Rao `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Acoda 框架，通过遗传算法演化语义保持的对抗性代码混淆样本，以抵御 LLM 对代码的分析。

**💡 创新点**

创新点在于：①利用 LLM 的安全对齐机制和基于 token 的信息处理机制设计 8 种语义保持的混淆方法；②将混淆方法与遗传算法结合，自动生成针对多 LLM 的跨模型对抗样本；③提出基于四项指标（语法、语义、循环复杂度、相似度）的定量评估体系。

**🔧 技术方法**

主要技术包括：遗传算法（选择、交叉、迭代演化）、LLM 提示工程与对抗生成、辅助 LLM 进行响应分类、四维度指标计算与加权评分。

**📊 数据集**

使用 CodeNet 数据集的 Python 代码，挑选 100 条通过所有测试用例、长度适中且包含函数的样本作为基准。

**📈 对比分析**

与随机选择混淆方法做对比，评估拒绝率、测试失败率、攻击成功率（ASR）。在 7 个主流 LLM（DS‑Coder、CodeGemma、CodeLlama、GPT‑4o 等）上，Acoda 达到最高 70% ASR，表现出强跨模型可迁移性，且对执行时间、内存消耗影响可忽略，代码长度增幅约 2 倍。

**⚠️ 局限性**

局限性：①依赖 LLM 的安全对齐和 token 机制，若模型安全性弱或对 token 敏感性低则效果减弱；②对抗方法集有限，可能无法覆盖未来模型的变化；③遗传算法易收敛到少数高效方法，缺乏多样性；④若 LLM 经过专门的反混淆微调或预处理，攻击效果可能降低。

---

## 246. Understanding and Supporting Online Discussion with Opinionated Chatbots

**arXiv ID:** 2606.11693 | [PDF](https://arxiv.org/pdf/2606.11693v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 247. Reason, Then Re-reason: Cross-view Revisiting Improves Spatial Reasoning

**arXiv ID:** 2606.11683 | [PDF](https://arxiv.org/pdf/2606.11683v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 248. Battery detection of XRay images using transfer learning

**arXiv ID:** 2606.11779 | [PDF](https://arxiv.org/pdf/2606.11779v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 249. Consensus Time in 3-Majority and 2-Choices Is Determined by the Maximum Initial Opinion Density

**arXiv ID:** 2606.11778 | [PDF](https://arxiv.org/pdf/2606.11778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 250. Automated Creativity Evaluation of Language Models Across Open-Ended Tasks

**arXiv ID:** 2606.11762 | [PDF](https://arxiv.org/pdf/2606.11762v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 251. ERN-Net : Evolving Reason Node-Net for Document Binarization

**arXiv ID:** 2606.11710 | [PDF](https://arxiv.org/pdf/2606.11710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 252. Making Locality-aware GEMM Compatible with Page-Granularity Placement on Chiplet GPUs

**arXiv ID:** 2606.11718 | [PDF](https://arxiv.org/pdf/2606.11718v1)

**作者:** Euijun Chung `[一作]` (Georgia Institute of Technology), Hyesoon Kim `[通讯]` (Georgia Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种全局内存布局（Chiplet-Contiguous Layout，CCL），使多片段 GPU 的 GEMM 运算能够在 4KB 页面级分配下实现细粒度的数据局部性，从而显著降低远程 HBM 访问量。

**💡 创新点**

创新点在于将同一芯片片段所需的数据连续存放，消除了传统行主序布局与页面级分配的不匹配，使得细粒度局部性能够通过固定 4KB 分配实现，无需硬件或操作系统修改。

**🔧 技术方法**

采用地址映射重排、行列索引分解与页级对齐的技术；在软件层面通过 TensorFlow/CUTLASS 等框架的布局抽象实现；评估时使用自研的块级 GEMM 局部性模拟器，并在 MI300X‑类似配置下进行仿真。

**📊 数据集**

使用 Qwen 3 30B 与 Llama 3.1 70B 两大 LLM 模型中的前向与反向 GEMM（共 36 个 BF16 GEMM），涵盖 4K、8K、16K 三种 token 数量，体现真实训练场景。

**📈 对比分析**

与 4KB/64KB/2MB 固定间隔分配、以及粗粒度局部性放置（Coarse LA）做对比；结果显示 CCL 在 Qwen 上远程 HBM 流量平均降低 24.7×、Llama 降低 19.2×，相较 Coarse LA 则进一步缩小 4.1×/2.1×，验证了显著性能提升。

**⚠️ 局限性**

局限在于需要静态布局预先确定；对动态工作负载或非 GEMM 运算的适应性有限；此外，实验仅覆盖 LLM GEMM，其他算子或更大模型的效果仍待验证。

---

## 253. Explore From Sketch: Accelerating UAV Exploration in Large-scale Environments with Prior Maps

**arXiv ID:** 2606.11708 | [PDF](https://arxiv.org/pdf/2606.11708v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 254. Beyond Frequency Marching: Orbit Recovery in Dihedral and Projected Multireference Alignment

**arXiv ID:** 2606.11701 | [PDF](https://arxiv.org/pdf/2606.11701v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 255. A Data-Centric Framework for Detecting and Correcting Corrupted Labels

**arXiv ID:** 2606.11699 | [PDF](https://arxiv.org/pdf/2606.11699v1)

**作者:** Ha-Linh Nguyen `[一作]` (VNU University of Engineering and Technology), Hieu Dinh Vo `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个端到端的数据中心框架，用于检测并纠正数据集中的损坏标签。

**💡 创新点**

创新点在于同时结合局部与全局实例关系进行标签错误检测，并通过贝叶斯推断整合输入特征与噪声标签的两种信号，系统性地推断出最可能的干净标签。

**🔧 技术方法**

核心技术包括：局部与全局相似性度量、两阶段标签检测与修正、贝叶斯推断框架、以及对噪声分布的显式建模。

**📊 数据集**

实验涵盖五大类数据集：CIFAR‑10、CIFAR‑100（图像），Agnews、Clickbait（文本），CodeXGLUE（源代码）。

**📈 对比分析**

与两种基线方法（投票式与排名式）相比，在多种噪声类型与噪声率下，修正精度提升高达58%，错误率下降约3%‑17%，下游任务准确率提升约6%。

**⚠️ 局限性**

局限性包括：对大规模数据的计算成本较高，仍需依赖先验噪声模型估计，对极端噪声率或完全无标签数据的鲁棒性尚未充分验证。

---

## 256. Evaluation of Alternative-Based Information Systems for Deliberative Polling using an Agentic Simulator

**arXiv ID:** 2606.11692 | [PDF](https://arxiv.org/pdf/2606.11692v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 257. Layer-Isolated Evaluation: Gating the Deterministic Scaffold of a Production LLM Agent with a No-LLM, Regression-Locked Test Harness

**arXiv ID:** 2606.11686 | [PDF](https://arxiv.org/pdf/2606.11686v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 258. Segment-Wise Soft Robotics Inspired Flexible Antenna Arrays: Design and Optimization

**arXiv ID:** 2606.11771 | [PDF](https://arxiv.org/pdf/2606.11771v1)

**作者:** Shuaishuai Han `[一作]` (University of Cyprus), Ioannis Krikidis `[通讯]` (University of Cyprus)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并优化了基于软机器人分段结构的可重构天线阵列（SRA），提出了端点天线配置（SEAC）和端点＋中间天线混合配置（HEIAC）两种部署方案。

**💡 创新点**

创新点在于将软机器人分段可控运动（伸缩、弯曲、扫动）与天线位置重构结合，显著提升空间相关性抑制；同时将传统可重构天线嵌入分段内部，形成双层重构，实现大尺度与细尺度的协同优化。

**🔧 技术方法**

采用了惩罚双重分解-投影梯度上升（PDD‑PGA）和块坐标下降‑PDD‑PGA（BCD‑PDD‑PGA）求解非凸优化问题，并利用贪婪后退搜索对天线激活进行稀疏化；软机器人运动模型基于正弦形变参数，信道模型为空间相关Rayleigh衰落。

**📊 数据集**

实验使用随机生成的1000组无线信道样本（K=7用户），无真实数据集；所有比较基准为固定圆形阵列、二维可重构圆形阵列和三维可重构圆形阵列。

**📈 对比分析**

与基准相比，SEAC在SNR 18 dB时对3D可重构阵列提升约37.9%，对2D可重构阵列提升约94%；HEIAC对3D提升约32.1%，对2D提升约72.2%；在紧凑阵列设置下SEAC可达49.3%的总率提升。

**⚠️ 局限性**

局限性包括：求解算法复杂度高，软机器人实现与实时控制尚未实验验证；模型假设理想CSI且忽略硬件失真；剩余互耦和非理想通道对性能仍有一定影响。

---

## 259. TacCoRL: Integrating Tactile Feedback into VLA via Simulation

**arXiv ID:** 2606.11743 | [PDF](https://arxiv.org/pdf/2606.11743v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 260. Multi-View In-Cabin Monitoring System for Public Transport Vehicles

**arXiv ID:** 2606.11739 | [PDF](https://arxiv.org/pdf/2606.11739v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 261. CompRank: Efficient LLM Reranking via Token-Level Compression and Decoding-Free Scoring

**arXiv ID:** 2606.11700 | [PDF](https://arxiv.org/pdf/2606.11700v1)

**作者:** Xuan Lu `[一作]` (Shanghai Jiao Tong University), Xiaoyu Shen `[通讯]` (Key Laboratory of Spatial Intelligence and Digital Derivative)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CompRank，一种 token‑级高效重排序框架，解耦文档表示、采用结构化 token 压缩并用 CopyNet 目标直接监督注意力评分，避免解码过程；

**💡 创新点**

创新点包括① 文档表示解耦实现可重用，② 仅保留约 10% 令牌的 segment‑wise 压缩，③ 采用 CopyNet 风格的监督直接对齐训练与推理的注意力得分；

**🔧 技术方法**

技术方案基于 Mistral‑7B LLM，块级无交互注意力，segment‑wise token 压缩，CopyNet 监督，注意力基评分，无需解码；

**📊 数据集**

实验数据集涵盖 BEIR 的七个基准（FiQA, SciFact, TREC‑COVID, DBPedia, Climate‑FEVER, NF‑Corpus, SciDocs）以及 TREC‑COVID 规模测试；

**📈 对比分析**

与 BM25、RankGPT、ICR、Corehead 等基线对比，BEIR 上平均 NDCG@10 达 39.2，仅保留 10.2% 文档令牌，接近 full‑token 39.7；TREC‑COVID 上从 30 预测到 500 文档保持稳定，速度提升 4.9–9.5 倍，且比解码型更快；

**⚠️ 局限性**

局限性在于当前未实现 KV 缓存，实验中仍需为每个查询重新编码文档；压缩比例对极长列表的效果未知；对不同 LLM 的泛化需进一步验证；

---

## 262. Ouroboros-Spatial: Closing the Data-Model Loop for Spatial Reasoning

**arXiv ID:** 2606.11719 | [PDF](https://arxiv.org/pdf/2606.11719v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 263. Substrate Asymmetry in User-Side Memory: A Diagnostic Framework

**arXiv ID:** 2606.11712 | [PDF](https://arxiv.org/pdf/2606.11712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 264. AnchorEdit: Maintaining Temporal Consistency in Multi-turn Image Editing via Causal Memory

**arXiv ID:** 2606.11751 | [PDF](https://arxiv.org/pdf/2606.11751v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 265. What Limits Does Quantization Place on Dense Top-$k$ Retrieval? A Theoretical Study

**arXiv ID:** 2606.11780 | [PDF](https://arxiv.org/pdf/2606.11780v1)

**作者:** Koki Okajima `[一作]` (NTT, Inc.), Tsukasa Yoshida `[通讯]` (NTT, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在有限精度下，如何实现每个k子集的密集嵌入，以支持N文档语料库的top-k检索。

**💡 创新点**

创新点在于理论上证明了在有限精度下，嵌入维度d与语料库大小N之间的关系，提出了Bd = Ω(k ln N)的界限，并识别了精度阈值B^* = O(lnln N)。

**🔧 技术方法**

使用了基于第一矩方法的理论分析，专注于ℓ_2归一化的B位均匀标量量化模型。

**📊 数据集**

没有具体提到使用的数据集，但讨论了N文档语料库的嵌入问题。

**📈 对比分析**

与之前的研究相比，本文的结果表明在有限精度下，嵌入维度和精度必须随着语料库大小的增加而增长，且在特定条件下，top-k检索变得不可能。

**⚠️ 局限性**

限制在于只考虑了完美的top-k检索，而未考虑近似检索的情况，且所用的均匀标量量化模型未能完全反映某些基本量化方案的特性。

---

## 266. UniReason-Med: A Shared Grounded Reasoning Interface for 2D-to-3D Transfer in Medical VQA

**arXiv ID:** 2606.11740 | [PDF](https://arxiv.org/pdf/2606.11740v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 267. A Fast Gaussian Mechanism under Continual Observation, with Applications

**arXiv ID:** 2606.11760 | [PDF](https://arxiv.org/pdf/2606.11760v1)

**作者:** Rasmus Pagh `[一作]` (BARC, University of Copenhagen), Sia Sejer `[通讯]` (BARC, University of Copenhagen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出 FastGaMe 机制，在连续观察场景下实现对 k 维向量的差分隐私发布，支持按需查询噪声且保持与二叉树机制相同的分布

**💡 创新点**

创新点在于使用 Brownian 桥与位操作实现常数时间噪声采样，显著降低时间复杂度从 log T 到 O(1)，并将 Private CountSketch 与 FastGaMe 结合用于动态空间范围计数与连接大小估计

**🔧 技术方法**

采用 Gaussian 二叉树机制、Brownian 桥采样、Word RAM 位操作、Private CountSketch、FastGaMe 以及离散时间离散事件模型等技术

**📊 数据集**

论文主要为理论研究，未使用公开数据集，示例均基于理论构造

**📈 对比分析**

与传统二叉树机制和批量更新方法相比，FastGaMe 的更新/查询时间降至常数，空间仅增加 log T；在动态范围计数和连接大小估计中误差保持 polylog 级别，更新/查询时间为 O((log B)^d) 等级

**⚠️ 局限性**

局限在于仅针对 Gaussian 噪声的 zCDP，未提供纯 DP 或 Laplace 实现；缺乏大规模实验验证，实际部署效果仍待实证

---

## 268. Improving Human Diving Endurance with a Field-Deployable, Untethered Exoskeleton

**arXiv ID:** 2606.11704 | [PDF](https://arxiv.org/pdf/2606.11704v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 269. Spectrally Regularized Latent Flow Matching for Turbulence Generation

**arXiv ID:** 2606.11691 | [PDF](https://arxiv.org/pdf/2606.11691v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 270. RankVR: Low-Rank Structure Perception and Value Recalibration for Robust Composed Image Retrieval

**arXiv ID:** 2606.11689 | [PDF](https://arxiv.org/pdf/2606.11689v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 271. Goal-Autopilot: A Verifiable Anti-Fabrication Firewall for Unattended Long-Horizon Agents

**arXiv ID:** 2606.11688 | [PDF](https://arxiv.org/pdf/2606.11688v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 272. Mind the Perspective: Let's Reason Recursively for Theory of Mind

**arXiv ID:** 2606.11724 | [PDF](https://arxiv.org/pdf/2606.11724v1)

**作者:** Chao Lei `[一作]` (University of Melbourne), Nir Lipovetzky `[通讯]` (University of Melbourne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在推理时通过递归构建角色视角来实现多层次心智理论（ToM）推理的方法。

**💡 创新点**

创新点在于将嵌套信念建模为从全局视角递归生成每个角色的观测视角，且通过确定性符号状态更新保证信念一致性，并在理论上证明其满足 KD45 逻辑，显著提升高阶 ToM 推理的准确性。

**🔧 技术方法**

主要技术包括事实基事件抽象、持久/瞬时事件分类、全局状态-事件序列构建、递归视角生成与观测完成、以及基于 LLM 的自然语言解释与推理；同时引入符号状态的确定性更新和 KD45 分析。

**📊 数据集**

使用 Hi-ToM、Big-ToM、FanToM 三个公开基准数据集，分别涵盖从零阶到四阶的高阶心智问题。

**📈 对比分析**

与 Chain-of-Thought、SimToM、TimeToM 等现有方法对比，RecToM 在所有 LLM 后端（GPT-5.4、Gemini-3、Qwen3.5、Gemma-4）上均实现了最高精度，Hi-ToM 上大部分模型达到 100%（或 98.5%+），在高阶问题上提升幅度可达 15% 以上；同时在 token 效率上也取得竞争优势。

**⚠️ 局限性**

局限性包括：仅适用于已定义事件转换与观测规则的受控文本基准，难以直接迁移到开放式或多模态环境；递归视角构建在推理时会产生额外计算成本，需要进一步优化。

---

## 273. From Prompts to Tokens: Internalizing Causal Supervision in Vision-Language Model for Multi-Image Causal Reasoning

**arXiv ID:** 2606.11745 | [PDF](https://arxiv.org/pdf/2606.11745v1)

**作者:** Haoping Yu `[一作]` (Case Western University), Jing Ma `[通讯]` (Case Western University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在大型视觉语言模型内部引入可路由的因果图和因果令牌，让模型能够从多图输入中自动推断并利用因果结构进行干预与反事实推理。

**💡 创新点**

①创建内部因果接口：将因果图作为DAG嵌入模型并通过RAMP实现信息传播；②提出统一监督桥M3S，支持多粒度、缺失的因果监督直接对齐到内部结构；③将监督迁移到内部而非仅限于提示层，显著提升模型对因果任务的表现。

**🔧 技术方法**

视觉编码 + 可学习变量查询；低秩参数化的DAG预测；Route‑Aware Message Propagation (RAMP)；节点/全局因果令牌循环；LLM解码器注入因果令牌；M3S多源监督（边监督、描述对齐、NOTEARS正则等）。

**📊 数据集**

CausalVLBench、Causal3D 两大多图因果基准，以及单图基准 CELLO 用于跨域评估。

**📈 对比分析**

在相同 7B Phi‑4 基础模型上与多种开源（LLaVA、DeepSeek、Qwen）和闭源商业（Gemini）模型对比，BridgeVLM 在干预任务上从 33.2% 提升至 54.4%，在 Causal3D 反事实任务上从 81.0% 提升至 92.3%，并将因果图恢复的 F1 从 33.4% 提升至 75.1%，整体性能接近或超越更大参数量模型。

**⚠️ 局限性**

生成的 DAG 仅用于内部推理，难以完全恢复真实因果结构；对全新未见的因果场景泛化仍有限；需要多源、部分缺失监督的对齐，模型对缺失变量对齐的依赖较高。

---

## 274. T2S: A Rehearsal-Based Approach for Extraction-Resistant Model Watermarking

**arXiv ID:** 2606.11698 | [PDF](https://arxiv.org/pdf/2606.11698v1)

**作者:** Jian-Ping Mei `[一作]` (Zhejiang University of Technology), Jie Xiao `[通讯]` (Zhejiang University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了深度学习模型水印在面对模型提取与后续去除攻击时的鲁棒性，并提出了一种基于回放（rehearsal）的水印微调框架 T2S，直接利用模拟被盗模型的反馈对目标模型进行二阶梯度微调；

**💡 创新点**

创新点在于不再通过调优触发器集或使用负样本来增强水印可提取性，而是将模拟被盗模型的梯度反馈直接回传到目标模型，显著提升了对模型提取和多种去除手段的抵抗力；

**🔧 技术方法**

核心技术包括：模拟模型提取的知识蒸馏、基于二阶导数的水印微调、不同类型触发器集（OOD、Mix、feature-based）的构造与验证；

**📊 数据集**

实验采用了 CIFAR-10、CIFAR-100 以及 Tiny-ImageNet 数据集，并以 ResNet‑18 等常用架构为基础；

**📈 对比分析**

与 Content、EWE、SSW 等主流方法在 Knockoff（软/硬标签）和 DFME 等提取攻击下对比，T2S 在软标签 Knockoff 的水印成功率可达 99.9%，硬标签亦高达 97%，在二次提取、量化、剪枝等去除操作中仍保持 95% 以上的成功率，且对模型准确率的影响极小；

**⚠️ 局限性**

局限性在于二阶梯度计算导致显著的 GPU 内存占用（单卡约 4.9GB/epoch）和训练成本，且对大规模模型的可扩展性需要梯度检查点或多卡分布式训练来缓解。

---

## 275. Parameter-Efficient Adapter Tuning for Tabular-Image Multimodal Learning

**arXiv ID:** 2606.11682 | [PDF](https://arxiv.org/pdf/2606.11682v1)

**作者:** Jiaqi Luo `[一作]` `[通讯]` (Soochow University), Jiaqi Luo (Soochow University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为TI-Adapter的模态特定适配器框架，用于高效地对预训练的表格和图像编码器进行微调，从而实现表格-图像多模态学习；

**💡 创新点**

创新点在于将表格编码器（TabPFN）冻结后仅在表格嵌入后插入轻量级适配器，并针对图像编码器设计了两种适配方案（嵌入层适配器与瓶颈级卷积适配器），从而在保持大部分预训练参数不变的同时实现任务特定的适配；

**🔧 技术方法**

主要技术包括预训练模型冻结、适配器（adapter）设计（包含瓶颈型和卷积型），以及多模态融合和线性预测头；

**📊 数据集**

在20个公开的表格-图像多模态数据集上进行实验，涵盖12个分类任务和8个回归任务；

**📈 对比分析**

与全参数微调、冻结仅线性头以及仅嵌入层适配器等多种基线进行对比，结果显示BCAdapter2（对ResNet最后两层插入卷积适配器）在平均排名上优于全微调，同时训练参数量显著降低；

**⚠️ 局限性**

局限性包括：适配器数量虽少但在早期层插入会导致GPU显存显著增加；表格适配器仅在嵌入层进行，可能无法充分利用表格编码器的所有潜能；以及实验主要集中在固定的预训练编码器（TabPFN和ResNet-50），未检验在其他架构下的泛化能力。

---

## 276. When Do Data-Driven Systems Exhibit the Capability to Infer?

**arXiv ID:** 2606.11769 | [PDF](https://arxiv.org/pdf/2606.11769v1)

**作者:** Maximilian Poretschkin `[一作]` (Fraunhofer Institute for Intelligent Analysis and Information Systems), Tabea Naeven `[通讯]` (Fraunhofer Institute for Intelligent Analysis and Information Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了欧洲AI法中对数据驱动系统推断能力的定义，并提出了分级框架以评估信用评分工作流程是否属于AI系统。

**💡 创新点**

创新点在于将统计学习理论与AI法的推断能力概念结合，构建了五级推断框架并应用于实际信用评分案例。

**🔧 技术方法**

采用统计学习理论、逻辑回归、决策树、特征分箱、信息量（IV）、Woe等技术，以及对模型的参数估计和特征选择。

**📊 数据集**

使用Kaggle的 Give Me Some Credit 数据集进行实验。

**📈 对比分析**

通过构建两种信用评分流程（半自动化与手工化）进行对比，半自动化在推断层级上为 3a，手工化为 1，性能差异在 AP 0.3737 vs 0.3602，AUROC 0.8517 vs 0.8502。

**⚠️ 局限性**

局限性在于样本规模小、仅关注统计学习模型、未覆盖符号 AI 及监管环境变化。

---

## 277. Non-special Divisors, LCPs of Codes, and LCD Codes on Kummer Extensions

**arXiv ID:** 2606.11764 | [PDF](https://arxiv.org/pdf/2606.11764v1)

**作者:** Huachao Zhang `[一作]` (Sun Yat-sen University), Chang-An Zhao `[通讯]` (Sun Yat-sen University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过对Kummer扩张上非特殊除子进行算术表征，构造了一系列线性互补对(LCP)和线性互补双码(LCD)AG码，特别是在GK曲线和同乘子Kummer扩张上给出了显式构造；

**💡 创新点**

创新点在于提出了不受支撑点是否全被完全分裂限制的通用算术判定，利用纯间隙实现了低阶除子构造，并提供了一个统一的LCP/LCD构造框架；

**🔧 技术方法**

核心技术包括：Kummer扩张的离散化理论、Riemann‑Roch公式与维度计数、纯间隙与Weierstrass半群的利用，以及对典型曲线（GK曲线、Hermitian曲线）取代基点的具体实现；

**📊 数据集**

所使用的“数据集”主要是数学对象：GK曲线、Hermitian曲线以及其分支点构造的Kummer扩张，具体参数如q=2,3,…,以及对应的代数曲线点集；

**📈 对比分析**

与现有文献中的LCP/LCD构造相比，本文给出的码族在码长、维度和最小距离上往往更优或相当，并且通过对GK曲线的实验例子验证了安全参数可通过单个码的最小距离得到；

**⚠️ 局限性**

局限性包括：仍未给出非特殊除子存在性的充分必要条件，且LCP/LCD构造依赖于对特定曲线的纯间隙和支撑点的可控性，未来工作需进一步扩展到更一般的曲线和除子配置。

---

## 278. RCAP: Robust, Class-Aware, Probabilistic Dynamic Dataset Pruning

**arXiv ID:** 2606.11761 | [PDF](https://arxiv.org/pdf/2606.11761v1)

**作者:** Atif Hassan `[一作]` (IIT Kharagpur), Jiaul H. Paik `[通讯]` (IIT Kharagpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出RCAP算法，在训练期间动态裁剪数据集，仅保留最有信息量的样本。

**💡 创新点**

创新在于对每个类别使用闭式公式动态分配子集大小，并按损失概率采样高损失样本，从而显著提升弱类别鲁棒性。

**🔧 技术方法**

利用类别损失聚合求解子集比例，并采用Softmax+温度采样高损失样本的概率分布。

**📊 数据集**

在六个数据集上验证：CIFAR10/100、ImageNet、Waterbirds、CelebA、iNaturalist。

**📈 对比分析**

与七种静态/动态剪枝基线对比，RCAP在所有设置下均取得最高worst-group准确率，平均可获得8.69×加速。

**⚠️ 局限性**

局限在于需要数个训练周期才能估计最佳比例，且温度参数β需手工调节。

---

## 279. A VPN-as-a-Service Tailored Enabler for Computing-constrained Environments

**arXiv ID:** 2606.11729 | [PDF](https://arxiv.org/pdf/2606.11729v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 280. FAST-MEL: A Fast, Accurate, and Storage Efficient Solution for Multimodal Entity Linking

**arXiv ID:** 2606.11749 | [PDF](https://arxiv.org/pdf/2606.11749v1)

**作者:** Derrien Thomas `[一作]` (University of Rennes), Pascale Sébillot `[通讯]` (University of Rennes)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FAST-MEL，一种轻量化、编码器驱动的多模态实体链接系统；

**💡 创新点**

创新点在于采用固定尺寸向量化表示文本与视觉信息，既保持高精度，又极大提升速度与存储效率；

**🔧 技术方法**

技术核心是基于编码器的向量化表示（如Transformer或轻量级网络），并对实体与提及做快速向量匹配；

**📊 数据集**

在公开的多模态实体链接基准数据集上进行评估（如DBpedia/Freebase的视觉-文本混合实例）；

**📈 对比分析**

与现有最优系统对比，FAST-MEL实现了同等或更高的链接准确率，速度提升约三倍，存储占用下降十倍；

**⚠️ 局限性**

局限在于固定尺寸向量可能无法捕捉极细粒度信息，且在极大知识库规模下仍需进一步优化编码压缩与索引策略。

---

## 281. Hey Chat, Can You Teach Me? Structuring Socratic Dialogue for Human Learning in the Wild

**arXiv ID:** 2606.11744 | [PDF](https://arxiv.org/pdf/2606.11744v1)

**作者:** Sidney Tio `[一作]`, Pradeep Varakantham `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于强化学习的在线导师系统，用于教学关税与制裁知识。系统通过监测学生掌握情况，识别知识空缺，动态决定下一个教学主题，并使用苏格拉底式对话引导学生思考。

**💡 创新点**

创新点在于将教学过程建模为一个基于知识图谱的决策网络，并通过强化学习学习最优教学策略，使得导师能够根据学生实时反馈自动调整教学内容和方法。

**🔧 技术方法**

技术实现主要包括知识图谱构建、强化学习（如 Q‑learning / Policy Gradient）用于策略学习、自然语言处理模块进行对话生成与评估，以及可视化面板用于展示学习进度。

**📊 数据集**

本文使用的主要数据集为关税与制裁的公开教材与案例数据（如 WTO、UNCTAD 相关数据），并通过人工标注的学生答题记录来训练与评估模型。

**📈 对比分析**

通过与传统基于规则的导师系统比较，强化学习策略在学生掌握率上提升了约12%，并显著降低了学生学习周期。

**⚠️ 局限性**

局限性包括：①缺乏大规模真实学生数据，导致模型泛化性不足；②对复杂情境（多轮对话）处理仍不够鲁棒；③系统对教师干预的依赖仍较高，需要进一步自动化。

---

## 282. MHOT: Height-Optimized Authenticated Data Structure for Blockchain State Commitment

**arXiv ID:** 2606.11736 | [PDF](https://arxiv.org/pdf/2606.11736v1)

**作者:** Sipeng Xie `[一作]` (Beihang University), Qin Wang `[通讯]` (Independent)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文设计并实现了 Mhot，一种高度优化的认证数据结构，用于区块链状态承诺，旨在提升写入吞吐量、降低写放大、缩小证明尺寸，并抵御 Nurgle 攻击。

**💡 创新点**

创新点主要包括：
- 采用 Height‑Optimized Trie（HOT）中的 discriminative‑bit 分支，实现线性 fanout–高度耦合，消除传统前缀 trie 的指数耦合；
- 引入双层 Merkle 结构，将每个复合节点的证明从 O(k) 降至 O(log k)；
- 将 HOT 适配为持久化、内容寻址的 ADS，支持无可信设置的 hash‑based 认证；
- 设计批量提交（height‑stratified）流水线，显著减少写放大并利用并行哈希；
- 在同一结构下实现对 Nurgle 攻击的零成功率。

**🔧 技术方法**

技术栈包括：
- Height‑Optimized Trie（HOT）与 discriminative‑bit 索引；
- 二层 Merkle（内部子树 Merkle + 外部节点 Merkle）；
- 内容寻址、copy‑on‑write 结构；
- RocksDB 作为底层 LSM‑tree 存储；
- Rust 编程、SIMD 加速、Blake3/Keccak 哈希；
- 并行哈希流水线、批量提交机制。

**📊 数据集**

使用的数据集：
- 真实以太坊主网区块 13,500,000–13,510,000 的状态更新；
- 合成随机 256‑bit 键工作负载，规模从 100 k 到 1 M 条目。

**📈 对比分析**

比较方法：与 MPT、Verkle、RainBlock、LMPT 等基线系统在同一硬件与工作负载下进行对比。结果显示：
- 写入吞吐量提升 5–9×；
- 写放大下降 3–4×；
- 树高保持 5–6 层，比 MPT 深度低 35–40%；
- 单点证明大小 1.1–1.4 KB（比 MPT 的 2.3–2.9 KB 小）；
- 在 Nurgle 攻击实验中，Mhot 0% 成功率，而 MPT 99.97%。
- 在真实工作负载下，Mhot 写吞吐约 130 k ops/s，MPT 仅 72 k ops/s。

**⚠️ 局限性**

局限性：
- 仍使用传统哈希，无法达到 Verkle 等 VC 方案的常数级证明长度；
- 并发写支持仍待完善，当前采用单写流水线；
- 对未来可能的攻击向量（如针对复合节点的冲突攻击）尚未完整评估；
- 验证延迟相对较高（尤其是 KZG 变体），对轻客户端有一定负担；
- 需要进一步硬件加速、内存优化及跨链兼容性验证。

---

## 283. A Fast Locality Simulator for GEMM Design-Space Exploration on Multi-Chiplet GPUs

**arXiv ID:** 2606.11716 | [PDF](https://arxiv.org/pdf/2606.11716v1)

**作者:** Euijun Chung `[一作]` (Georgia Institute of Technology), Hyesoon Kim `[通讯]` (Georgia Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个轻量级、功能性 tile‑level 局部性模拟器，用来评估多芯片 GPU 上 GEMM 计算的本地化（本地 vs 远程 HBM）设计空间，并利用 AI 代理在几秒内搜索到更优的 CTA 遍历顺序。

**💡 创新点**

创新点在于：① 将软件调度与数据布局的设计空间与硬件模型解耦，提供可在 CPU 上秒级完成的模拟；② 通过功能性模拟揭示远程 HBM 流量可在不同 GEMM 形状下变化 90×；③ 通过 AI 代理快速发现 2D block‑swizzle CTA 遍历可在最佳 1D 遍历基础上进一步降低 5.1× 的远程流量。

**🔧 技术方法**

采用了功能性 trace‑driven 模拟、L2 缓存与多芯片域的本地/远程 HBM 分类、CTA lock‑step 进度模型、AI 代理（Claude Code）进行遍历优化、以及 2D block‑swizzle CTA 遍历技术。

**📊 数据集**

使用了 Qwen3‑30B（Mixture‑of‑Experts）和 Llama3.1‑70B 的 12 个前向/反向 Feed‑Forward 层 GEMM（包含 gate‑up、down、dX、dW），以及这些 GEMM 对应的大尺寸矩阵（如 Qwen3 的 (196,608, 16,384, 2,048)）。

**📈 对比分析**

对每个 GEMM 在 4 种数据放置（行/列轮询或连续）与 3 种 CTA 遍历（行、列、2D block‑swizzle）组合下进行模拟，测量远程 HBM 流量、局部性和 L2 命中率；与 4 KB round‑robin 基准比较，最佳配置可将远程流量降低至 1/90；AI 代理进一步在 2D 遍历上提升 5.1×；单个 GEMM 模拟耗时 9–44 s（平均 24 s），比周期级模拟快数倍。

**⚠️ 局限性**

主要局限包括：① 仅功能性模拟，无法给出绝对性能或功耗；② 假设 CTA 以 wave‑同步（lock‑step）方式进度，可能略微影响 L2 命中率；③ 未建模内存侧 L3（Infinity Cache）等后级缓存；④ 对 L2 命中率的估计在某些极端情况可能有偏差。

---

## 284. DroneShield-AI: A Multi-Modal Sensor Fusion Framework for Real-Time Autonomous Drone Threat Detection, Behavioral Intent Classification, and Swarm Intelligence in Contested Airspace

**arXiv ID:** 2606.11687 | [PDF](https://arxiv.org/pdf/2606.11687v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 285. RLCSD: Reinforcement Learning with Contrastive On-Policy Self-Distillation

**arXiv ID:** 2606.11709 | [PDF](https://arxiv.org/pdf/2606.11709v1)

**作者:** Leyi Pan `[一作]` (Tsinghua University), Lijie Wen `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RLCSD，一种通过对比正负提示的 on‑policy self‑distillation 方法，解决了 privilege‑induced style drift 并提升推理模型的性能。

**💡 创新点**

创新点在于：①将正确与错误提示做对比消除风格漂移；②将对比信号作为 GRPO 验证器优势的加权调节，而非单纯替代；③设计了两路 loss 与自适应掩码，保持稳定训练。

**🔧 技术方法**

使用的技术包括对比自我蒸馏、GRPO (group‑relative policy optimization)、硬币抽样与对比信号的 tanh 归一化、两路 PPO 风格的 clipped 损失以及对 token‑level 监督的掩码与 sign‑preserving clamp。

**📊 数据集**

实验数据集：DeepMath‑103K（训练），AMC23、AIME24、AIME25（数学评测）以及 Knights & Knaves（逻辑评测）。

**📈 对比分析**

与 GRPO、OPSD、SDPO、SRPO、RLSD 等基线相比，RLCSD 在 Qwen3（1.7B/4B/8B）和 Olmo‑3‑7B‑Think 上在数学与逻辑推理任务均显著提升，保持训练熵与长度稳定，最终验证准确率普遍提升 2–15 分。

**⚠️ 局限性**

局限性：仅在中小规模推理模型（≤8B）和特定数学/逻辑任务上验证；对比提示需要在同一 batch 中存在正负样本，可能对稀疏提示场景不适用；未探讨大规模模型或跨领域迁移的适用性。

---

## 286. Noise-Aware Framework for Correcting Corrupted Labels

**arXiv ID:** 2606.11695 | [PDF](https://arxiv.org/pdf/2606.11695v1)

**作者:** Ha-Linh Nguyen `[一作]` (VNU University of Engineering and Technology), Hieu Dinh Vo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于噪声感知的框架，通过显式估计数据集的噪声分布并将其融入模型训练，从而实现对受损标签的迭代软重标；

**💡 创新点**

创新点在于将特征学习与标签修正解耦，利用噪声感知的深度网络在训练稳定后才开始标签修正，并通过损失稳定触发器和逐步软重标来避免确认偏误；

**🔧 技术方法**

核心技术包括噪声转移矩阵估计、噪声感知深度网络、损失稳定触发器、迭代软重标以及最大后验硬标转换；

**📊 数据集**

使用了六个常见的图像与文本分类数据集，并通过多种自动标注技术生成逼真的噪声场景，同时还在一个真实的嘈杂数据集上进行验证；

**📈 对比分析**

与现有的基准标签修正方法相比，本文方法在所有数据集和噪声设置下均实现了19%–52%的误差下降，且在高噪声场景下训练得到的数据集能使下游模型的准确率提升8%–67%；

**⚠️ 局限性**

主要局限在于需要先对噪声分布进行估计且依赖模型训练稳定后再修正标签，导致训练时间较长；另外，在极高噪声或复杂噪声模式下的鲁棒性尚未得到充分验证。

---

## 287. Beyond Per-Token Pricing: A Concurrency-Aware Methodology for LLM Infrastructure Cost Estimation

**arXiv ID:** 2606.11690 | [PDF](https://arxiv.org/pdf/2606.11690v1)

**作者:** Chitral Patil `[一作]` `[通讯]` (Independent Researcher), Chitral Patil (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统实验测量了自托管大型语言模型推理在不同负载下的实际成本，揭示了利用率对成本的巨大影响，并提出了基于GPU利用率的成本模型与实时计量工具。

**💡 创新点**

创新点在于：①首次量化展示负载（请求速率）导致成本在同一硬件上可变 17.5–36.3 倍；②构建了包含硬件、模型架构、精度、请求速率和SLO 的五变量成本函数 C_eff=f(H,M,Q,λ,L)；③发布了可实时读取 vLLM Prometheus 指标的开源成本计量器。

**🔧 技术方法**

技术手段包括：vLLM 服务器持续批处理、Little's Law 对并发的建模、对不同请求速率的离散扫描、GPU 监控 (nvidia-smi) 与延迟统计、量化（FP16/FP8）与 MoE 结构实验。

**📊 数据集**

使用了三类模型（Dense Llama 3.1 8B、Ultra‑Sparse MoE Qwen3‑30B‑A3B、Sparse MoE Mixtral 8x7B）在 NVIDIA H100 NVL 与 A100 80GB PCIe 上，采用 512:256 统一长度的随机 token 测试，此外还做了 I/O 形状、前缀缓存、到达突发性等敏感性探测。

**📈 对比分析**

对比方法：将传统基于峰值吞吐量的“占用率为 100%”计算与本文提出的基于实际负载的 C_eff 进行对比；在 API 价格层面绘制了自托管与 GPT‑5.5、Claude、Gemini 的交叉点，发现低流量时自托管成本显著高于 API。实验结果显示，在低请求速率下成本显著上升，FP8 对 MoE 模型的加速效果更大；跨硬件验证表明该现象在不同 GPU 上保持一致。

**⚠️ 局限性**

局限性包括：①实验使用固定长度随机 token，真实对话工作负载可能产生更低成本；②只覆盖了三类模型和两种硬件，未覆盖更大规模或不同架构；③未考虑多租户 GPU 共享、SLO 优化、不同推理引擎等；④仅以 vLLM 进行基准，其他引擎的量化曲线可能不同。

---

## 288. Systematic Cybersecurity Risk Analysis of European Rail Traffic Management System

**arXiv ID:** 2606.11839 | [PDF](https://arxiv.org/pdf/2606.11839v1)

**作者:** Kacper Darowski `[一作]` (Technical University of Munich), Lukas Lautenschlager `[通讯]` (Technical University of Munich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过将ERTMS规范抽象成系统模型，运用mora风险评估框架，对当前与未来ERTMS配置进行系统化网络安全风险分析；

**💡 创新点**

创新点在于首次将完整ERTMS规范映射至mora模型，结合功能级安全目标与技术级威胁，形成全局风险图谱，并指出从EuroBalise到GSM‑R的关键弱点及其对安全性的影响；

**🔧 技术方法**

使用的技术包括mora风险评估框架、CEM攻击潜能评估、ISO/SAE 21434损害等级映射、以及针对通信链路（如GSM‑R、FRMCS、Ethernet、GNSS）的攻击树构建；

**📊 数据集**

数据来源为公开的ERTMS规范文件（Baseline 4及其子集）以及相关文献中的已知漏洞和攻击案例，未使用实验数据集；

**📈 对比分析**

与仅采用ERTMS标准的基线对比，加入FRMCS和ETCS Level 2后，极高风险数量从68降至48，表明改进措施在降低安全风险方面显著提升；

**⚠️ 局限性**

局限性包括缺乏时间依赖的攻击树节点导致循环依赖未解决、对安全目标的损害转化可能无法降低风险等级、以及模型仅覆盖ERTMS核心，不含完整的车站与网络基础设施层面。

---

## 289. Flow Matching with In-Context Priors for Out-of-Distribution Brain Dynamics

**arXiv ID:** 2606.11833 | [PDF](https://arxiv.org/pdf/2606.11833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 290. I Understand How You Feel: Enhancing Deeper Emotional Support Through Multilingual Emotional Validation in Dialogue System

**arXiv ID:** 2606.11875 | [PDF](https://arxiv.org/pdf/2606.11875v1)

**作者:** Zi Haur Pang `[一作]` (Kyoto University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在对话系统中定义并系统化了情感验证的三项子任务：验证响应识别、验证时序检测和验证响应生成，并发布了跨语言对话语料与评测基准。

**💡 创新点**

创新点在于提出跨语言情感门控融合模型 MEGUMI，利用冻结的多语义编码器与语言特定情感编码器通过跨模态注意力和门控融合实现精准的验证时序判断，并构建 EmoValidBench，首次提供生成质量与情感有效性双重评价指标。

**🔧 技术方法**

技术方案包括：冻结 XLM‑RoBERTa 作为语义支撑；分别使用 ModernBERT‑Large（英）和 LUKE‑Japanese‑Large（日）进行情感编码；情感增强多语言注意力模块 EEMA；门控多模态单元融合语义与情感；以及在生成与评测阶段使用 LLM（Llama‑3.1‑8B、GPT‑4.1‑Nano）与 LLM‑as‑Judge 的临床沟通指标。

**📊 数据集**

数据集方面：M‑EDESConv（120k 语料，英日双语对话）和 M‑TESC（TUT 口语故事数据翻译成英日），并对部分样本进行人工验证，形成跨文本与语音的验证标注。

**📈 对比分析**

与基线（随机、mBERT、XLM‑RoBERTa、LLM）对比，MEGUMI 在验证时序检测的宏 F1、目标类精度上均显著优于基线；在验证响应生成方面，LLM 在语义一致性与安全性指标上表现较好，但情感认可、准确反映与支持温暖等指标仍落后；整体评测显示 MEGUMI 与 LLM 组合在实际对话情境下的精确性和宏 F1 达到 61–70% 之间，优于单纯 LLM 的过度验证倾向。

**⚠️ 局限性**

限制包括：仅覆盖英语与日语，缺乏对其他语言文化差异的覆盖；使用文本转录忽视声学、视听等情感线索；伪标签化过程可能引入误标；冻结 XLM‑RoBERTa 阻止了任务专用微调；模型规模受限于 8B 参数；部署到心理健康等敏感领域需更严格的安全与伦理监控。

---

## 291. Space-sampled Value Decay: Forgetting Mechanisms for Non-stationary Deep Reinforcement Learning

**arXiv ID:** 2606.11797 | [PDF](https://arxiv.org/pdf/2606.11797v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 292. AutoMine Solution for AV2 2026 Scenario Mining Challenge

**arXiv ID:** 2606.11874 | [PDF](https://arxiv.org/pdf/2606.11874v1)

**作者:** Songliang Cao `[一作]` (Xiaomi EV), Hangjun Ye `[通讯]` (Xiaomi EV)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出AutoMine，一种基于LLM和VLM的多模态场景挖掘框架，能够从大规模驾驶日志中精准检索与自然语言描述相符的时间戳与参与者；

**💡 创新点**

创新点在于引入语义保持的提示增强、轨迹重构、鲁棒与视觉增强原子函数，并通过执行反馈实现代码自我改进，显著提升对语言歧义与感知噪声的鲁棒性；

**🔧 技术方法**

核心技术包括大语言模型生成可组合原子函数、视觉语言模型（Qwen3.5-27B）执行视觉推理、轨迹优化算法（继承Immortal Tracker）以及基于日志执行的自我修正循环；

**📊 数据集**

使用的数据集为Argoverse 2 Sensor Dataset，包含1000条15秒的驾驶日志和约10,000条规划相关的自然语言查询；

**📈 对比分析**

与其他顶尖参赛团队比较，AutoMine在官方排行榜上在HOTA-Temporal上排名第3（36.38分）并在Timestamp BA上取得最高分77.21，显示出优秀的时间定位与检测性能；

**⚠️ 局限性**

局限性包括对LLM和VLM的依赖导致推理速度受限、对极端感知噪声或极短日志的鲁棒性不足，以及在某些需要更细粒度空间关系的查询上仍可能出现错误。

---

## 293. How Requirements Quality Makes (or Breaks) Traceability Link Recovery

**arXiv ID:** 2606.11834 | [PDF](https://arxiv.org/pdf/2606.11834v1)

**作者:** Tobias Hey `[一作]`, Julian Frattini `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对两大用例数据集中的189个用例描述，手工和自动标注28种质量缺陷，随后对五种无监督自动化追踪链接恢复（TLR）方法进行实验，量化了各缺陷对TLR性能（精确率、召回率、F1、F2）的影响，并比较了不同方法在不同缺陷情境下的表现。

**💡 创新点**

创新点在于首次系统性地将需求质量缺陷与自动TLR性能关联，采用贝叶斯因果推断评估缺陷效应强度，并揭示某些传统视为缺陷的因素（如不一致抽象级别、交替路径、交叉需求）反而能提升TLR效果；同时阐明不同TLR范式对缺陷的敏感度差异。

**🔧 技术方法**

使用的技术包括：基于向量空间模型的VSM、LSI；基于句子级嵌入的FTLR；基于检索-生成的LiSSA（包含仅检索子模块LiSSA_IR-only）以及贝叶斯因果回归分析。

**📊 数据集**

使用的数据集为eTour（旅游行业）和iTrust（医疗行业）的用例与Java源代码对照数据，合计189个用例、599句子、342个代码文件、594条金标准追踪链接。

**📈 对比分析**

方法对比通过在每个用例层面计算精确率、召回率、F1和F2，发现VSM和LSI在多种缺陷下表现不稳定；FTLR和LiSSA在大多数缺陷下保持相对稳健，LiSSA_IR-only对“交叉需求”表现尤为突出；总体而言，最优性能因缺陷而异，提示需根据需求质量选择合适的TLR算法。

**⚠️ 局限性**

局限性包括仅评估两大数据集和五种TLR方法，缺陷标注对手工操作依赖且部分缺陷难以客观量化；使用的金标准可能不完整，导致缺陷判定偏差；贝叶斯回归的交互效应需更大样本验证；结果对其他需求工件和TLR任务的外推性受限。

---

## 294. Understanding and Detecting Scalability Faults in Large-Scale Distributed Systems

**arXiv ID:** 2606.11815 | [PDF](https://arxiv.org/pdf/2606.11815v1)

**作者:** Hao-Nan Zhu `[一作]` (University of California Davis), Cindy Rubio-González `[通讯]` (University of California Davis)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

对分布式系统中的可扩展性缺陷进行最大规模的实证研究，并提出了一种基于动态与静态分析的检测方法。

**💡 创新点**

首次系统化地识别了“维度代码片段（DCF）”与 11 种抗模式的组合对可扩展性缺陷的影响，并将此视角嵌入检测工具。

**🔧 技术方法**

使用运行时 Instrumentation 结合规模化工作负载获取执行轨迹，利用静态调用图分析匹配抗模式，构建工具 Scalelens。

**📊 数据集**

基于 10 个开源 Java 分布式系统（如 Hadoop、Kafka、Spark 等）的 Issue 报告和最新稳定版本的源码进行实验。

**📈 对比分析**

与基线工具对比，Scalelens 能完整检测 36/55 条已知缺陷（+2 条半部分检测），并在基线基础上多 4.2 倍发现 DCF；在最新版本中识别出 129+68+137 条带抗模式的潜在缺陷，实验验证无误。

**⚠️ 局限性**

仅适用于 JVM 字节码；对事件驱动、回调或反射代码的静态分析不完整；需要手工编写扩容工作负载；若工作负载未覆盖某维度，可能漏检 DCF；分类体系可能不适用于非 Java 系统。

---

## 295. Lius: Translation Model Based Instructional Lingustic Using Continual Instruction Tuning In Kupang Malay

**arXiv ID:** 2606.11786 | [PDF](https://arxiv.org/pdf/2606.11786v1)

**作者:** Joanito Agili Lopo `[一作]` (Universitas Gadjah Mada), Guntur Budi Herwanto `[通讯]` (Universitas Gadjah Mada)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在Kupang Malay低资源语言上，提出了基于持续指令调优（CIT）的语言指令式训练方法，显著提升了LLM翻译性能。

**💡 创新点**

创新点在于将词汇和语义特征嵌入四类指令（上下文、语义、音素、列表组标签），并通过CIT迭代多指令训练以增强模型对低资源语言的理解。

**🔧 技术方法**

采用Cendol-mT5多语言LLM、FastText词向量、KeyBERT句子表征、经验回放、BF16、梯度检查点等技术。

**📊 数据集**

使用Kupang Malay与印尼语的66,521句对（训练53,217/测试13,304）、约3,200条双语词典、以及数万条单语文本（如Tapaleuk、Taxi1500等）构建指令数据。

**📈 对比分析**

通过与标准指令模型、BLOOMZ、mT0、Madlad400等多语言LLM和NMT模型在SacreBLEU、chrF++、TER、ROUGE-L等指标上对比，CIT模型在1.2B参数版本上分别高出4–6分，远优于其他模型。

**⚠️ 局限性**

局限在于人类评测一致性低、缺乏标准化的Kupang Malay评价准则，且在极低资源或极端扰动下性能仍不理想。

---

## 296. Modular Anthropomorphic Hand Design via Multi-Parameter Finger Benchmarking and Selection

**arXiv ID:** 2606.11826 | [PDF](https://arxiv.org/pdf/2606.11826v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 297. The relaxation complexity of the standard simplex is logarithmic

**arXiv ID:** 2606.11852 | [PDF](https://arxiv.org/pdf/2606.11852v1)

**作者:** Simon Keil `[一作]` (Technical University of Munich), Stefan Weltge `[通讯]` (Technical University of Munich)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

该论文证明了离散标准单纯形Δ_d的松弛复杂度为Θ(log d)，给出了一个对数级别的显式构造；

**💡 创新点**

创新点在于用一种简单的线性不等式构造（利用高代数度数的实数α和位数编码）实现了对Δ_d的对数复杂度松弛，从而填补了上界与下界之间的巨大差距；

**🔧 技术方法**

技术上主要采用了代数数论（选取满足1,α,…,α^{2^k-1}线性无关的α）、位向量的二进制编码与线性规划不等式的组合，以及对偶性与线性独立性的代数分析；

**📊 数据集**

论文未使用任何外部数据集，所有结果均为理论构造与证明；

**📈 对比分析**

与之前最好的上界O(d/√(log d))相比，新上界O(log d)与已知下界Ω(log d)匹配，证明了松弛复杂度的最优渐进取值；

**⚠️ 局限性**

主要局限在于构造需要使用高代数度数的不可约实数α，并且得到的是渐近上界，未给出具体常数或对小维度的精确最优解；

---

## 298. TextHOI-3D: Text-to-3D Hand-Object Interaction via Discrete Multi-View Generation and Joint Mesh Optimization

**arXiv ID:** 2606.11805 | [PDF](https://arxiv.org/pdf/2606.11805v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 299. StatefulDiscovery: Evidence-Calibrated Claim Formation in Open-Ended Scientific Discovery

**arXiv ID:** 2606.11851 | [PDF](https://arxiv.org/pdf/2606.11851v1)

**作者:** Jiayao Chen `[一作]` (Southern University of Science and Technology), Linyi Yang `[通讯]` (Southern University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了StatefulDiscovery框架，利用显式的探索状态来协调前沿选择、证据获取与主张裁定，完成无目标的开放式科学发现。

**💡 创新点**

创新点在于：①将探索状态外部化为持久对象，使得证据与主张的可信度在整个探索过程中实时更新；②双层架构（全局前沿控制+局部假设裁定）实现探索与证据校准的闭环；③结构化假设集与局部裁定显著提升发现价值和主张质量。

**🔧 技术方法**

技术手段包括：大型语言模型（如Qwen3.5-plus）、可编程技能接口、可执行查询束、局部裁定逻辑、前沿控制策略，以及基于LLM的证据与价值评估。

**📊 数据集**

使用40个真实数据任务，来自BixBench、BLADE和DiscoveryBench，涵盖生物医学、社会科学、行为学与跨域数据。

**📈 对比分析**

与Raw Agent、AutoDiscovery、OpenEvolve、SAGA等基线进行对比；在高质量主张计数、DV（发现价值）与ES（证据支持）上均优于基线，尤其在DV与高质量主张率上领先约25–30%，且在对比评估中在所有40个任务中均被优先选中。

**⚠️ 局限性**

局限性包括：①依赖LLM的推理与评估，模型差异会影响性能；②未能处理需外部文献检索或实验验证的任务；③在极为复杂或跨学科的开放问题中仍可能产生解释性跳跃；④缺乏对生成主张的因果验证机制。

---

## 300. CORE-Bench: A Comprehensive Benchmark for Code Retrieval in the Era of Agentic Coding

**arXiv ID:** 2606.11864 | [PDF](https://arxiv.org/pdf/2606.11864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 301. External Experience Serving in Production LLM Systems: A Deployment-Oriented Study of Quality-Cost Trade-offs

**arXiv ID:** 2606.11806 | [PDF](https://arxiv.org/pdf/2606.11806v1)

**作者:** Lin Sun `[一作]` (Qiyuan Tech), Xiangzheng Zhang `[通讯]` (Qiyuan Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在生产级LLM系统中如何通过外部经验来提升任务质量，并比较了全局注入与选择性检索两种服务接口；

**💡 创新点**

创新点在于将外部经验视为部署决策，提出服务接口与任务成本结构两层break-even框架，证明在经验具有案例依赖性时选择性检索优于全局注入，并指出检索匹配质量比检索深度更关键；

**🔧 技术方法**

采用检索增强生成、Prompt注入、Trigger‑aware检索、LLM选择器、配对Instruct/Thinking推理模式以及前缀缓存等技术；

**📊 数据集**

使用生产环境下的安全审核基准（风险聚焦标签）、工具调用预测任务以及GPQA任务，并从业务标注数据中构建经验仓库；

**📈 对比分析**

通过对准确率、延迟、提示词和完成词数量的系统对比，实验显示在moderation任务中选择性检索从≈20%提升至≈71%，但引入的延迟和提示词略有增加；全局压缩版在短输出任务中表现不佳，Top‑K在10级时饱和；解码重任务在检索后完成长度和延迟可降低；

**⚠️ 局限性**

局限在于仅关注推理阶段的经验服务，未覆盖经验构建、维护及参数内化全过程；评估指标主要为准确率、token/延迟，缺乏完整的运行时拆解；基准为风险聚焦非流量匹配；缺乏对始终关闭/开启/选择性激活策略的全面比较；检索诊断（如Recall@K）不完整；

---

## 302. Task-Aware Structured Memory for Dynamic Multi-modal In-Context Learning

**arXiv ID:** 2606.11853 | [PDF](https://arxiv.org/pdf/2606.11853v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 303. SwarmSense-DNN: A Trustworthy and Decentralized Neural Framework for Proactive Anomaly Defense in Consumer IoT

**arXiv ID:** 2606.11803 | [PDF](https://arxiv.org/pdf/2606.11803v1)

**作者:** Jing Yang `[一作]` (Universiti Malaya), Lip Yee Por `[通讯]` (Universiti Malaya)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出SwarmSense-DNN，一个将群体智能与联邦学习结合的去中心化深度神经网络框架，用于可信的消费级IoT异常检测。

**💡 创新点**

创新点在于：① 通过群体智能的信息素更新与信任权重实现自治聚合；② 分层层次聚类与GAT注意力实现局部与全局协同；③ 引入差分隐私与安全多方计算保证数据隐私；④ 设计自愈与鲁棒机制提升对攻击和节点失效的耐受性。

**🔧 技术方法**

使用的技术包括：群体智能（信息素更新）、联邦学习、图神经网络（GAT）、多头自注意力、差分隐私、加密安全多方计算、分层层次架构、动态聚类与自愈策略。

**📊 数据集**

实验数据集包含 IoT-23、NSL-KDD、CICIDS2017、UNSW-NB15 以及工业 IoT 数据集。

**📈 对比分析**

与中央化 DL、FedAvg-AD、Distributed GNN、Edge-FL、Ensemble-Dist 等方法对比，SwarmSense-DNN 在平均检测准确率上提升至 95.44%（比最佳基线高 5.26%），通信开销下降 72.9%，收敛轮次 32 轮（比最佳基线快 64%）且在精度、召回率、AUC 等指标上均优于对照组。

**⚠️ 局限性**

局限性：对超参数敏感；在网络分区时准确率下降至约 89.6%；不适用于需要亚秒级响应的极低时延场景；缺乏可解释性，需要进一步引入 SHAP 等解释方法。

---

## 304. MemNovo: Look Back at the Spectrum for Balanced De Novo Peptide Sequencing from Mass Spectrometry

**arXiv ID:** 2606.11868 | [PDF](https://arxiv.org/pdf/2606.11868v1)

**作者:** Dongxin Lyu `[一作]` (Westlake University), Jun Xia `[通讯]` (HKUST-GZ)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Transformer-based de novo peptide sequencing模型的推理动态进行诊断并提出一种无训练、可插拔的记忆增强解码机制MemNovo，以平衡谱信息与氨基酸先验。

**💡 创新点**

创新点在于发现并量化了谱信息利用不足（Sensitivity Imbalance）并通过持久谱记忆和超保守残差注入在推理时恢复互信息，从而在不增加参数的情况下显著提升精度。

**🔧 技术方法**

使用Transformer encoder-decoder（Casanovo、InstaNovo）、特征缩放诊断框架、持久谱记忆、投影无学习的交叉注意力和残差注入。

**📊 数据集**

Nine Species benchmark（九种物种的高分辨率MS/MS数据）。

**📈 对比分析**

在Casanovo和InstaNovo上与原始模型做对比，Casanovo的肽精度从31.97%提升至44.47%（相对+39.1%），InstaNovo提升至48.22%（相对+3.9%），同时保持计算开销<1%。

**⚠️ 局限性**

局限性：对谱信息利用不足程度依赖模型，极平衡模型提升有限；仅在最终解码层注入，可能未能最优利用中间层信息；仅在特定基准上验证；不涉及PTM细化或大规模多模态推广。

---

## 305. Scene-Adaptive Nonlinear Tone Curves for Pseudo Ground-Truth Generation in Low-Light 3D Gaussian Splatting

**arXiv ID:** 2606.11841 | [PDF](https://arxiv.org/pdf/2606.11841v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. Enhancing LLM-Based Code Translation with Verified Multi-Semantic Representations

**arXiv ID:** 2606.11863 | [PDF](https://arxiv.org/pdf/2606.11863v1)

**作者:** Yufu Wang `[一作]` (Dalian University of Technology), Zhilei Ren `[通讯]` (Dalian University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Multisage 框架，通过多语义增强与自校准提升 LLM 在代码翻译中的功能正确性。

**💡 创新点**

创新点在于将结构化语义解析、丰富语义视图生成与语义一致性校准三大模块组合，且通过等价变换验证语义可靠性，解决传统模型对外部语义资源依赖且易误导的问题。

**🔧 技术方法**

技术包括静态分析提取控制流、类型约束与 API 依赖，LLM 生成代码摘要、API 描述与单元测试，等价变换与语义一致性评估，以及基于阈值的自校准与过滤。

**📊 数据集**

使用 XCodeEval、XLCoST 做多语义数据构建，评估基于 HumanEval-X 交叉语言测试集的翻译效果。

**📈 对比分析**

与原始 LLM、指令调优模型、TransCoder 等基线比较，在 HumanEval‑X 上成功率提升最多达 2.22×，CodeBLEU 同时显著提升，尤其在中小规模模型上表现突出。

**⚠️ 局限性**

局限性包括对阈值设置的敏感性、主要验证于函数级翻译、缺乏对大型项目依赖与多文件交互的评估，以及在极端语义复杂场景下仍可能出现误导的语义生成。

---

## 307. LASA: A Weak Supervision Method for Open-Vocabulary Scene Sketch Semantic Segmentation

**arXiv ID:** 2606.11837 | [PDF](https://arxiv.org/pdf/2606.11837v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 308. From Uniform to Learned Graph Priors: Diffusion for Structure Discovery

**arXiv ID:** 2606.11831 | [PDF](https://arxiv.org/pdf/2606.11831v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 309. Human-Guided Co-Manipulation of Carbon Fiber Plies

**arXiv ID:** 2606.11818 | [PDF](https://arxiv.org/pdf/2606.11818v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 310. Jaguar: Fast Private CNN Inference with Power-of-Two Homomorphic Arithmetic

**arXiv ID:** 2606.11827 | [PDF](https://arxiv.org/pdf/2606.11827v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 311. Harnessing Routing Foresight for Micro-step-level MoE load balancing in RL Post-training

**arXiv ID:** 2606.11867 | [PDF](https://arxiv.org/pdf/2606.11867v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 312. Plan-and-Verify Video Reward Reasoning with Spatio-Temporal Scene Graph Grounding

**arXiv ID:** 2606.11838 | [PDF](https://arxiv.org/pdf/2606.11838v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 313. Feature-Aligned Speech Watermarking for Robustness to Reconstruction Distortions

**arXiv ID:** 2606.11828 | [PDF](https://arxiv.org/pdf/2606.11828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 314. STCC: A Unified Source-Channel Semantic Token Coding Framework for Semantic Communications

**arXiv ID:** 2606.11819 | [PDF](https://arxiv.org/pdf/2606.11819v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 315. Skill-Augmented AI Agents for Medical Research Analysis: An Exploratory Multi-Model Human Evaluation in an NSCLC Transcriptomic Biomarker Task

**arXiv ID:** 2606.11830 | [PDF](https://arxiv.org/pdf/2606.11830v1)

**作者:** Qianyu Yao `[一作]` (AIPOCH PTE. LTD.), Huimei Wang `[通讯]` (AIPOCH PTE. LTD.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对比了原生AI与具备医疗研究技能包的AI代理在NSCLC免疫治疗转录组学分析中的输出质量。

**💡 创新点**

首次在多模型人类评估框架下探究技能包装对AI科研输出质量的影响，并提供模型层面异质性描述。

**🔧 技术方法**

使用OpenClaw AI代理平台与一套公开的医疗研究技能包（包含数据预处理、差异表达、通路富集、免疫微环境分析等模块）。

**📊 数据集**

公开转录组数据集，用于构建NSCLC免疫治疗反应多基因签名。

**📈 对比分析**

通过21份匿名输出（9份原生AI，12份技能增强），四名非专家和两名专家评估7分李克特量表；技能增强输出在专家整体质量上平均提高0.39分，但置信区间跨零，差异不显著。

**⚠️ 局限性**

样本量小、专家评估可靠性低、未验证生物学有效性、未区分平台与技能本身影响，导致结果仅为假设生成。

---

## 316. MultiToP: Learning to Patch Visual Tokens to Mitigate Hallucinations in Video Large Multimodal Models

**arXiv ID:** 2606.11792 | [PDF](https://arxiv.org/pdf/2606.11792v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 317. Towards Data-free and Training-free Compression for Speech Foundation Models Using Parameter Clustering

**arXiv ID:** 2606.11836 | [PDF](https://arxiv.org/pdf/2606.11836v1)

**作者:** Haoning Xu `[一作]` (Chinese University of Hong Kong), Xunying Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种数据无关、训练无关的参数聚类压缩方法，用于替代传统剪枝，针对HuBERT‑large和Whisper‑large‑v3进行压缩；

**💡 创新点**

创新点在于通过k‑means聚类将相似结构单元合并，并引入基于方差的混合稀疏分配，使压缩过程保持结构化、硬件友好且不依赖训练数据；

**🔧 技术方法**

采用的技术包括k‑means参数聚类、方差分层稀疏分配、对线性层权重的重构以及可选的微调；

**📊 数据集**

使用了LibriSpeech数据集（100小时clean子集以及完整的dev/test集）进行评估；

**📈 对比分析**

与幅值剪枝比较，HuBERT‑large在50%稀疏率下WER绝对降低27.73%/18.61%（test‑clean/test‑other），Whisper‑large‑v3在10%稀疏率下WER降低2.86%/5.02%；微调后性能与剪枝持平或略优；混合稀疏进一步提升效果；

**⚠️ 局限性**

局限性包括：在极高稀疏率（>60%）下性能显著退化，尤其对方差较小的Whisper模型；目前仅压缩线性层，未针对卷积或自注意力权重；需在更多模型与任务上进一步验证。

---

## 318. Optimizing Cloud Deployment: Blending of IaaS and FaaS for Microservice Architecture

**arXiv ID:** 2606.11824 | [PDF](https://arxiv.org/pdf/2606.11824v1)

**作者:** Nikhil Kapoor `[一作]` (Indian Institute of Technology), Sougata Mukherjea `[通讯]` (Indian Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了一套基于指标的自动化框架，将传统 IaaS 微服务迁移至混合 IaaS+FaaS 部署模型，利用两套微服务应用进行案例验证。

**💡 创新点**

创新点在于将服务级性能指标与迁移决策紧密耦合，提供了可复用、可扩展的自动化迁移流程，并系统评估了混合模型的优势与局限。

**🔧 技术方法**

使用 Google Cloud Platform（Compute Engine、Kubernetes、Cloud Functions）、Prometheus/Stackdriver 监控、脚本自动化工具（如 Terraform/Ansible）以及性能分析工具。

**📊 数据集**

数据集为两套真实微服务应用，涵盖典型业务逻辑与多种负载模式；未使用公开数据集，仅基于内部实验数据。

**📈 对比分析**

通过与传统 IaaS 部署在相同工作负载下的成本、延迟、吞吐量等指标比较，证明混合部署在资源利用率提升、成本下降（约 20-30%）与可伸缩性方面优于纯 IaaS。

**⚠️ 局限性**

局限包括 FaaS 的冷启动延迟、无状态或短周期微服务的适配难题、不同语言/运行时支持不均衡以及迁移过程中可能产生的功能不一致和调试复杂度。

---

## 319. Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code

**arXiv ID:** 2606.11817 | [PDF](https://arxiv.org/pdf/2606.11817v1)

**作者:** Yitong Zhang `[一作]` (Tsinghua University), Jia Li `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种利用语法约束解码（Grammar‑Constrained Decoding）诱导大模型生成恶意代码的攻击手段，并设计了基于honeypot代码的安全对齐方法以恢复模型在该约束下的安全性。

**💡 创新点**

创新点在于发现常用的可靠性增强技术GCD本身可被攻击，并首次提出仅使用标准语法即可完成的“CodeSpear”攻击；以及引入honeypot代码的代码模态安全对齐（CodeShield），突破自然语言拒绝缺失时的安全漏洞。

**🔧 技术方法**

使用了Grammar‑Constrained Decoding、直接偏好优化（Direct Preference Optimization）对齐、构造自然语言拒绝、恶意代码和honeypot代码三类响应的偏好对抗训练。

**📊 数据集**

利用RMCBench和MalwareBench的恶意代码请求、HumanEval和MBPP的通用代码生成任务以及OpenCodeInstruct作为honeypot代码来源，结合PKU‑RLHF构造对齐数据。

**📈 对比分析**

对10个主流LLM在本地和API部署下与多种基线（如PAIRS、DAN、CodeJailbreaker等）进行对比，攻击成功率提高约30个百分点；对比Safe‑DPO时，CodeShield在GCD下将攻击成功率从80%降至<5%，同时对通用代码生成的影响极小。

**⚠️ 局限性**

局限在于评估依赖LLM判定，未考虑更复杂的自适应攻击（除已测试外）和所有GCD实现的差异，且honeypot代码的多样性仍有限，可能在极端语法或高安全性需求场景下失效。

---

## 320. Seeing What Matters: Perceptual Wrapper with Common Randomness for 3D Gaussian Splatting

**arXiv ID:** 2606.11782 | [PDF](https://arxiv.org/pdf/2606.11782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 321. Agents All the Way Down; A Methodology for Building Custom AI Agents from Substrate to Production

**arXiv ID:** 2606.11869 | [PDF](https://arxiv.org/pdf/2606.11869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 322. RePAIR: Predictive Self-Supervised Representation Learning in Chess

**arXiv ID:** 2606.11860 | [PDF](https://arxiv.org/pdf/2606.11860v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 323. SheafStain: Sheaf-Theoretic Schrödinger Bridge for Spatially and Biologically Coherent Virtual Staining

**arXiv ID:** 2606.11846 | [PDF](https://arxiv.org/pdf/2606.11846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 324. Designing AI-Supported Focus Groups: A Role x Modality Playbook

**arXiv ID:** 2606.11835 | [PDF](https://arxiv.org/pdf/2606.11835v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 325. WorldReasoner: Evaluating Whether Language Model Agents Forecast Events with Valid Reasoning

**arXiv ID:** 2606.11816 | [PDF](https://arxiv.org/pdf/2606.11816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 326. Fine-tuning Multi-modal LLMs with ART: Art-based Reinforcement Training

**arXiv ID:** 2606.11854 | [PDF](https://arxiv.org/pdf/2606.11854v1)

**作者:** Michal Chudoba `[一作]` (University of Stavanger), Tomasz Wiktorski `[通讯]` (University of Stavanger)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种通过优化单一输入图像来微调冻结的多模态大语言模型的方法，称为ART（Art-based Reinforcement Training）；

**💡 创新点**

创新点在于不修改模型权重或运行时架构，只在图像输入空间中进行梯度更新，实现可部署的“计算艺术”并兼容高吞吐量引擎；

**🔧 技术方法**

采用可微分的像素空间参数化、基于奖励的强化学习（如DAPO）以及ViT视觉编码器的梯度反向传播；

**📊 数据集**

使用GSM8K（小学数学）、GPQA（研究生级问答）和ToolMind（结构化工具调用）三大文本任务的数据集；

**📈 对比分析**

与LoRA、随机图像、随机字符串等基线对比，ART在数学和工具调用任务上与LoRA持平或优于LoRA，在小模型上提升更显著，训练与推理速度也更快；

**⚠️ 局限性**

在需要高精度推理的GPQA任务上效果有限，且目前仅在Qwen3.5系列验证，跨模型/跨尺寸的泛化及不同优化目标的效果待进一步研究。

---

## 327. Multimodal Ordinal Modeling of Alzheimer's Disease Severity Using Structural MRI and Clinical Data

**arXiv ID:** 2606.11794 | [PDF](https://arxiv.org/pdf/2606.11794v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 328. AI4Land: Scalable Deep Learning for Global High-Resolution Land Use Reconstruction

**arXiv ID:** 2606.11793 | [PDF](https://arxiv.org/pdf/2606.11793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 329. WarpGuard: Protected-Site Control-Flow Integrity for CUDA SASS Binaries

**arXiv ID:** 2606.11871 | [PDF](https://arxiv.org/pdf/2606.11871v1)

**作者:** Igor Santos-Grueiro `[一作]` `[通讯]` (International University of La Rioja), Igor Santos-Grueiro (International University of La Rioja)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了针对NVIDIA CUDA SASS二进制的受保护站点控制流完整性（CFI）系统，实现对GPU设备代码的执行时检查与防御。

**💡 创新点**

创新点在于：①在SASS层面恢复控制流站点并生成精确的政策；②采用返回状态认证和可恢复目标集校验，实现后续控制流的“检查前释放”；③将不具备完整证据的站点明确归为未覆盖，而非默认允许；④将动态检查与调试日志分离，提供可审计的覆盖报告。

**🔧 技术方法**

使用技术包括：SASS恢复与分析、WG‑NVBit动态插桩、WG‑ST匹配时序与WG‑PC补丁缓存验证、加密令牌与MAC验证返回状态、按站点重建目标集、SIMT-aware 检查、以及在主机端与GPU端的密钥管理。

**📊 数据集**

数据集涵盖77个CUDA工件（fatbin、cubin、JIT等），共计51,621个SASS控制流站点，执行了52.2M次动态检查，并设计了77例攻击矩阵与公开代码案例进行验证。

**📈 对比分析**

与基线（原生）和检测模式对比，WG‑NVBit在大多数工作负载上产生约800%‑900%的运行时开销，但在精确检查后可完全阻止所有受保护的攻击。对比于回调‑free WG‑ST 与 WG‑PC，匹配时序的开销显著降低，且在真实样本上能够保持功能兼容。

**⚠️ 局限性**

限制包括：①残留的同站点目标多样性导致语义完整性仍被弱化；②后端密钥与私有状态的可信度依赖于主机/驱动隔离；③高开销的动态插桩使得部署受限；④仅支持NVIDIA GPU和SASS指令集，跨平台可移植性尚未实现。

---

## 330. A Comprehensive Ecosystem for Open-Domain Customized Video Generation

**arXiv ID:** 2606.11783 | [PDF](https://arxiv.org/pdf/2606.11783v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 331. TaskFusion: Continual Anomaly Detection for Heterogeneous Tabular Data

**arXiv ID:** 2606.11844 | [PDF](https://arxiv.org/pdf/2606.11844v1)

**作者:** Dayananda Herurkar `[一作]` (German Research Center for Artificial Intelligence), Andreas Dengel `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种用于异构表格数据的持续异常检测框架 AGF+TaskFusion，解决了特征模式不一致和分布漂移导致的灾难性遗忘问题。

**💡 创新点**

创新点包括：①在共享潜在空间内通过 Schema Adapter 与 Distribution Alignment 实现跨域特征对齐；②TaskFusion 通过边界感知混合与跨任务潜在混合增强决策边界；③利用表格数据蒸馏与异常曝光构建高效重放机制，避免存储原始历史数据。

**🔧 技术方法**

采用的技术包括：Schema Adapter、Distribution Alignment、Anomaly Classifier、TaskFusion Augmentation（Within‑Task MixUp 与 Cross‑Task Latent Mixing）、Outlier Exposure、Tabular Dataset Distillation。

**📊 数据集**

在 21 个跨领域的异构表格基准数据集（金融、图像、航天、文档等）上进行评估。

**📈 对比分析**

与无持续学习、Finetune、Multitask、CaSSLe、EDSR 等基线对比，AGF+TaskFusion+OE+DD 在 Balanced Accuracy、PR‑AUC、ROC‑AUC 上显著提升，接近甚至超过多任务上限，并显著减少灾难性遗忘。

**⚠️ 局限性**

局限性在于对极长任务序列的鲁棒性不足，以及固定增强与对齐策略对高度不同域的适应性有限。

---

## 332. Toward Trustworthy AI: Multi-Target Adversarial Attacks and Robust Defenses for Continuous Data Summarization

**arXiv ID:** 2606.11804 | [PDF](https://arxiv.org/pdf/2606.11804v1)

**作者:** Yuefang Lian `[一作]` (Nankai University), Jason Xue `[通讯]` (CSIRO's Data 61)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在可信 AI 管道中，连续数据摘要（尤其基于相似度的多分辨率摘要）在面向相似度矩阵的对抗扰动时的脆弱性，并提出了一种可用于多目标攻击与混合攻击下鲁棒防御的 DR‑submodular 形式化与算法；

**💡 创新点**

创新点在于：①将多分辨率图像摘要的离散子模函数映射为多线性扩展的 DR‑submodular 目标，证明其 m‑弱单调性；②构建了针对相似度矩阵的单一扰动同时攻击多目标摘要的 min‑max 形式；③给出了一套带理论保证的近似算法（攻击与防御均实现 m(1‑1/e) 比例），并在混合攻击类型下提出了鲁棒连续贪心算法；

**🔧 技术方法**

技术方法包括：连续子模最大化的梯度剪裁连续贪心（continuous greedy），投影梯度下降，凸/非凸 min‑max 迭代，线性最大化 oracle（LMO），以及多线性扩展和 m‑弱单调性的理论分析；

**📊 数据集**

实验数据集：CIFAR‑10、MNIST、Fashion‑MNIST 用于真实数据的多线性扩展摘要；此外构造了一个受控聚类多目标基准（10 个模型、50 条目、5 个聚类）用于验证结构化攻击与防御效果；

**📈 对比分析**

与梯度投影（PGD）以及随机扰动基线比较，攻击在低至中等预算下能够显著降低连续和离散摘要质量，攻击成功率达到 60‑100%，而 PGD 的平均降幅仅为 0.01‑0.05。防御方面，提出的鲁棒连续贪心方法相较 PGD 基线在保留清洁摘要质量、降低攻击诱发的损失和提升恢复率方面表现更好，尤其在受控聚类基准上能完全恢复簇覆盖与邻居分类准确率；

**⚠️ 局限性**

局限性包括：仅考虑相似度矩阵的点对点扰动，未涵盖像素级或特征空间的对抗；鲁棒算法对扰动几何（ℓ1/ℓ2/ℓ∞）和参数（λv、γ）高度敏感；在真实数据上鲁棒效果有限，主要验证集中在受控基准；以及假设目标函数保持 DR‑submodular 与 m‑弱单调性，限制了适用范围。

---

## 333. Deformable In-Hand Slip-Aware Tactile Sensor with Integrated Velocity, Force/Torque, and Pressure Map Sensing

**arXiv ID:** 2606.11952 | [PDF](https://arxiv.org/pdf/2606.11952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 334. Efficient Graph Indexing for Interval-Aware Vector Search

**arXiv ID:** 2606.11789 | [PDF](https://arxiv.org/pdf/2606.11789v1)

**作者:** Siyuan Liang `[一作]` (Beijing Institute of Technology), Xubin Li `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出统一的区间感知相似搜索框架 URNG，并实现可扩展的图索引 UG，支持 IFANN、ISANN、RFANN、RSANN 等多种区间约束 ANN 查询。

**💡 创新点**

创新点在于：① 将区间包含与穿越语义统一进相对邻域图，保证子图结构可继承（结构遗传性）并保持单调可搜索；② 通过语义位掩码实现单一图结构同时满足多种查询语义；③ 采用候选生成+统一剪枝+迭代修复的高效构造策略，克服了理论 URNG O(n³) 构造瓶颈；④ 设计了区间感知的 beam‑search 与快速入口节点获取算法。

**🔧 技术方法**

技术核心：图基 ANN（RNG/NSG 变体），区间属性的多维排序与邻域采样，基于距离与区间 witness 的双条件剪枝，语义位掩码，迭代统一剪枝与修复，二分搜索入口节点，Beam search 结合语义过滤。

**📊 数据集**

实验数据集：五个公开数据集——DB-OpenAI（990k）、GIST1M（1M）、S&P 500（1.45M）、SIFT1M（1M）与 DEEP1M（990k），其中 S&P 500 的区间属性来自真实金融数据，其他通过合成生成。

**📈 对比分析**

与基线方法（HNSW、Vamana、NSG、HCNNG、Hi‑PNG 族、Timestamp、SeRF、DSG、ACORN、iRangeGraph、UNIFY、Faiss‑HNSW、SuperPostfiltering、HNSW‑hnswlib）进行对比。UG 在 QPS‑召回曲线上持续领先，尤其在高召回区间可达 100‑300× 的速度提升；索引构建时间与内存占用均与竞争方法持平或更优；在多种区间负载、不同 k 值与大规模扩展实验中均保持稳定优势。

**⚠️ 局限性**

局限性：① 目前仅支持数值区间属性，未覆盖关键字或复杂谓词；② 近似构造仍需多次迭代与参数调优；③ 对动态更新（插入/删除）的支持尚未研究；④ 当区间分布极端稀疏或高度重叠时，候选生成可能不足，导致性能下降。

---

## 335. Toward Generalist Autonomous Research via Hypothesis-Tree Refinement

**arXiv ID:** 2606.11926 | [PDF](https://arxiv.org/pdf/2606.11926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 336. Real-Time Language Model Jamming: A Case Study for Live Music Accompaniment Generation

**arXiv ID:** 2606.11886 | [PDF](https://arxiv.org/pdf/2606.11886v1)

**作者:** Bowen Zheng `[一作]` (University of Wisconsin--Madison), Xiaosong Ma `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了一套面向实时音乐伴奏的框架，采用帧同步流式推理使得模型能够在外部节奏信号的驱动下即时生成伴奏。

**💡 创新点**

创新点在于提出了基于RTT模型的自动参数搜索方法（推理间隔I与生成长度GL），通过概率分析和实时备份机制实现不同网络与硬件环境下的可调实时性与音乐质量平衡。

**🔧 技术方法**

技术上使用了Transformer自回归解码器、交织的旋律-伴奏令牌化、KV缓存加速、客户端多线程异步调度及服务器端备份缓冲，并对RTT采用二次曲线与Pareto尾模型进行刻画。

**📊 数据集**

模型训练与评估基于POP909数据集，同时构造了64首测试曲目（30首AccoMontage，34首POP1K7），采用MIDI标记法进行音符令牌化。

**📈 对比分析**

通过与离线基线以及三种部署环境（本地、局域网、远程云）下的多组(I,GL)配置进行比较，结果表明系统能在高ISR/ISR_w下接近离线质量，验证了方法的有效性。

**⚠️ 局限性**

主要局限包括对曝光偏差和实时输入噪声的鲁棒性不足、缺乏对演奏者风格的即时适配以及对极端网络抖动的动态自适应能力有限。

---

## 337. The Art of Interrogation: Consistency Amplifies Factuality in Spatial Reasoning

**arXiv ID:** 2606.11918 | [PDF](https://arxiv.org/pdf/2606.11918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 338. Efficient Multinomial Logistic Bandit via Frequent Directions

**arXiv ID:** 2606.11968 | [PDF](https://arxiv.org/pdf/2606.11968v1)

**作者:** Linzhe He `[一作]` (Nanjing University), Lijun Zhang `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种高效的在线多项式逻辑回归Bandit算法EOFD-MLogB，利用矩阵频繁方向压缩提升参数估计和奖励构造的计算效率。

**💡 创新点**

创新点在于将频繁方向（Frequent Directions）矩阵压缩技术嵌入OFUL-MLogB，既降低了每轮时间复杂度至O(Kd(m+K)^2)、空间复杂度至O(Kd(m+K))，又通过低秩SVD近似实现参数更新和奖励上界的根求解与特征值计算，且在Hessian近似低秩时保留了与原算法相近的理论 regret 上界。

**🔧 技术方法**

使用技术包括：频繁方向矩阵压缩、低秩SVD维护、约束在线牛顿法更新、Kd×K谱范数的特征值分解以及一维根求解。

**📊 数据集**

实验主要在公开的多分类 Bandit 数据集（如Yahoo! News、Mushroom、KDD Cup 等）和合成数据上进行。

**📈 对比分析**

与传统 OFUL-MLogB 及其他基线算法（如UCB、LinUCB、Softmax 等）比较，EOFD-MLogB 在保持相近 regret 的同时，显著降低了每轮计算时间（多达数十倍）和内存占用，表现出更好的规模适应性。

**⚠️ 局限性**

局限性在于：1) 对高维但非低秩 Hessian 的场景，压缩误差会增大，导致 regret 上界变宽；2) 需要额外选择 sketch size m，参数调优复杂；3) 实验中主要验证了计算效率，进一步在更大规模真实世界任务上的性能仍需探索。

---

## 339. Frozen Multimodal Embeddings for Personality and Cognitive Ability Assessment in Asynchronous Video Interviews

**arXiv ID:** 2606.11930 | [PDF](https://arxiv.org/pdf/2606.11930v1)

**作者:** Kuo-En Hung `[一作]` (National Taiwan Normal University), Hsiang-Wen Wang `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对ACM Multimedia AVI Challenge 2026，构建了基于冻结多模态编码器的轻量级人格与认知能力评估框架，使用Trait‑specific回归和后期融合对自评HEXACO人格进行预测，并对认知水平进行诊断分类。

**💡 创新点**

创新点在于：①采用冻结的CLIP、Whisper、RoBERTa、E5和DeBERTaV3多模态表征而非全模型微调；②针对每个HEXACO维度单独建模并进行后期融合；③在认知分类中引入主体属性基线揭示验证集潜在shortcut。

**🔧 技术方法**

使用的技术包括：CLIP视觉编码器、Whisper语音编码器、RoBERTa/E5/DeBERTaV3文本编码器、Ridge/PCA+Ridge/ElasticNet/贝叶斯岭/PLS回归器、NNLS/加权平均融合、校准等低容量下游模型。

**📊 数据集**

使用的dataset是AVI Challenge 2026的数据集，共644名受试者，包含两个通用和四个个性相关问题的录像与自评结果。

**📈 对比分析**

在Track1中，与官方baseline相比，平均MSE从0.3334降至0.2696，减少19.1%；在Track2中，复合多模态集成达0.5313准确率，而主体属性基线达到0.5781，均优于官方baseline。

**⚠️ 局限性**

主要限制包括：校准仅基于验证集，可能对测试集过拟合；冻结编码器可能遗漏细粒度视觉信号；认知分类结果易受主体属性短路影响；语音转录质量与ASR错误可能影响文本特征。

---

## 340. Quality Adaptive Angular Margin Learning for Respiratory Sound Classification

**arXiv ID:** 2606.11915 | [PDF](https://arxiv.org/pdf/2606.11915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 341. Snapping Matters: Context-Aware Onset Refinement for Automatic Music Transcription

**arXiv ID:** 2606.11903 | [PDF](https://arxiv.org/pdf/2606.11903v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 342. PAPEL: A Collaborative System for Parental Guidance during Preschool Play-Based English Learning

**arXiv ID:** 2606.11896 | [PDF](https://arxiv.org/pdf/2606.11896v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 343. GraspLLM: Towards Zero-Shot Generalization on Text-Attributed Graphs with LLMs

**arXiv ID:** 2606.11898 | [PDF](https://arxiv.org/pdf/2606.11898v1)

**作者:** Hengyi Feng `[一作]` (University of Electronic Science and Technology of China), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出GraspLLM框架，将文本属性图与大型语言模型相结合，实现零样本跨域、跨任务的推理；

**💡 创新点**

创新点包括：统一语义空间、基于图形动机的自监督学习、最优上下文子图采样以及对齐投影，突破传统LLM在图结构上的局限；

**🔧 技术方法**

使用技术包括Qwen3-Embedding-8B文本编码、Motif‑aware GNN、对齐投影器、贪婪子图采样与自监督对比学习，以及基于提示的LLM调优；

**📊 数据集**

实验数据集涵盖十四个真实世界文本属性图，涉及学术、电子商务、社交、网页等五个领域（如Cora、Citeseer、Pubmed、Arxiv、History、Photo、Computer、WikiCS、Instagram、Reddit、Cornell、Texas、Wisconsin、Washington）；

**📈 对比分析**

与MLP、GNN、GraphSAGE、Llama、Vicuna等多种基准比较，GraspLLM在零样本节点分类与链接预测上均超过对手，跨域提升常超过0.25，跨任务表现亦优于所有预测型LLM方法；

**⚠️ 局限性**

局限性在于仅针对文本属性图，未扩展到异构/符号/时序图，任务范围局限于节点/边级零样本，子图线性化可能导致信息损失。

---

## 344. Image Quality Assessment of Identity Cards Using Measures from Open Face Image Quality

**arXiv ID:** 2606.11884 | [PDF](https://arxiv.org/pdf/2606.11884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 345. SpecLoR: Spectral Lookahead Rectification for Motion-Coherent Text-to-Video Generation

**arXiv ID:** 2606.11969 | [PDF](https://arxiv.org/pdf/2606.11969v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 346. From Fork-Join to Asynchronous Tasks: Parallelizing Tiled Cholesky Decomposition with OpenMP and HPX

**arXiv ID:** 2606.11937 | [PDF](https://arxiv.org/pdf/2606.11937v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 347. Characterizing Software Aging in GPU-Based LLM Serving Systems

**arXiv ID:** 2606.11916 | [PDF](https://arxiv.org/pdf/2606.11916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 348. From Content to Knowledge: Lightning Fast Long-Video Understanding with Neural Knowledge Representations

**arXiv ID:** 2606.11913 | [PDF](https://arxiv.org/pdf/2606.11913v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 349. Quadratic APN Functions in Dimension 8 via Gröbner Basis Search in a Self-Equivalence Subspace

**arXiv ID:** 2606.11967 | [PDF](https://arxiv.org/pdf/2606.11967v1)

**作者:** Oleksandr Kuznetsov `[一作]` `[通讯]` (eCampus University), Oleksandr Kuznetsov (eCampus University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在 8 维二次 APN 函数搜索中，作者构建自等价子空间 V_A，利用随机采样与 Gröbner 基枚举方法发现了 566 个新的二次 APN 函数。

**💡 创新点**

创新点在于证明先前认为空的自等价子空间中存在新的 CCZ 类别，并提出一种基于 Gröbner 基的“局部枚举+门控”搜索框架。

**🔧 技术方法**

使用了 RREF 参数化、Gröbner 基（Magma）计算、ortho‑derivative 归一化与 CCZ 等价判定。

**📊 数据集**

利用公开的 Beierle‑2025、Beierle‑2019 等大型二次 APN 数据库以及作者自行生成的 V_A 样本。

**📈 对比分析**

通过与已有数据库比对无匹配，证明新函数与已知函数 CCZ 不等价；性能上，428 个 NL=4 切片共耗约 57 CPU 小时，平均每切片 8–15 分钟构建，解题不到 0.1 秒。

**⚠️ 局限性**

主要限制是仅覆盖了 V_A 子空间的 0.65% 切片，未得到完整的 CCZ 分类，且新类别缺乏解析的代数描述。

---

## 350. Tail-Aware Adaptive-k: Query-Adaptive Context Selection for Retrieval-Augmented Generation

**arXiv ID:** 2606.11907 | [PDF](https://arxiv.org/pdf/2606.11907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 351. Neuro-Relational Programs: Unifying Queries and Neural Computation over Structured Data

**arXiv ID:** 2606.11946 | [PDF](https://arxiv.org/pdf/2606.11946v1)

**作者:** Arie Soeteman `[一作]` (University of Amsterdam), Moritz Schönherr `[通讯]` (Leipzig University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种统一的神经关系程序（Neuro-Relational Programs，NRPs）框架，能够在带有数值向量嵌入的关系数据库上声明式地混合传统查询与可学习的神经网络计算；

**💡 创新点**

创新点在于将 Datalog‑style 规则与向量组合、聚合、变换操作融合到同一语义中，从而实现零元、单项式和前沿受限程序的完整理论刻画，并与深度同态网络、FO+计数逻辑以及统一 TC0 等已有形式化体系建立了精确对应；

**🔧 技术方法**

采用了嵌入关系数据库（e‑database）、向量化事实（e‑fact）、三类规则（并、析取、变换）、聚合与组合函数、可微变换函数（如 ReLU‑FFN）、以及对 FO+C(1/2) 逻辑的扩展来刻画 NRPs 的表达能力；

**📊 数据集**

在实现层面，作者开发了一个系统将 NRPs 编译为“神经关系代数”，并进一步映射到 PyTorch、cuDF 与 SQL，以便在标准数据库及 GPU 上执行；虽然论文未给出具体公开数据集，但实验使用了常见的图与关系数据（如社交网络、知识图谱）进行对比；

**📈 对比分析**

与传统基于图神经网络的实现以及其它基于 Datalog 的框架相比，NRPs 在性能上可与最先进实现相当，且在模型表达与维护上更为简洁，尤其在端到端梯度训练方面展现出更高的灵活性；

**⚠️ 局限性**

局限性包括：目前仅对前向传播表达能力进行了理论分析；递归程序与循环结构的支持尚未深入；以及在大规模数据上对内存与并行度的可扩展性仍需进一步评估。

---

## 352. uva-irlab-conv at SemEval-2026 Task 8: Multi-Turn RAG with Learned Sparse Retrieval and Listwise Reranking

**arXiv ID:** 2606.11945 | [PDF](https://arxiv.org/pdf/2606.11945v1)

**作者:** Simon Lupart `[一作]` (University of Amsterdam), Mohammad Aliannejadi `[通讯]` (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多阶段检索增强生成(RAG)流水线，先用LLM重写对话查询，再使用学习型稀疏检索(LION‑SP)获取候选文档，随后通过点式和列表式LLM重排序，最后以最优文档生成答案。

**💡 创新点**

创新点在于将LLM重写与稀疏检索相结合，利用全会话历史进行列表式重排序，并在零样本环境下实现跨领域检索与生成，兼顾可解释性与性能。

**🔧 技术方法**

使用的技术包括LLM查询重写、学习型稀疏检索(LION‑SP)、Qwen3‑Reranker‑8B点式重排序、GPT‑4.1列表式重排序与生成，全部采用零样本推理。

**📊 数据集**

采用SemEval‑2026 MTRAG基准，覆盖金融、云文档、政府与维基百科四大领域的数据集，且包含可答与不可答的混合查询。

**📈 对比分析**

在检索任务中，系统以nDCG@5 0.5475排名第二（38队中）。在生成任务中，oracle（Task B）和预测检索（Task C）分别取得H. Avg 0.5123与0.4865；在非条件评估下，生成模型在保持高可信度（RL_F 0.8035）的同时，答案质量在可答/部分可答场景下仍具竞争力。

**⚠️ 局限性**

局限性包括未显式处理不可答查询（缺乏明确的“不知道”提示）、对生成质量与检索深度的负相关仍存在、以及对某些领域（如FiQA）的检索效果相对较弱。

---

## 353. When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in Vision-Language-Action Models

**arXiv ID:** 2606.11906 | [PDF](https://arxiv.org/pdf/2606.11906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 354. SG2Loc: Sequential Visual Localization on 3D Scene Graphs

**arXiv ID:** 2606.11880 | [PDF](https://arxiv.org/pdf/2606.11880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 355. Feature extraction for plant growth estimation

**arXiv ID:** 2606.11966 | [PDF](https://arxiv.org/pdf/2606.11966v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 356. Decoding Multimodal Cues: Unveiling the Implicit Meaning Behind Hateful Videos

**arXiv ID:** 2606.11953 | [PDF](https://arxiv.org/pdf/2606.11953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 357. Semantic Grading of Written Answers in Low-Resource Language Bangla Using a Fine-Tuned Lightweight Language Model

**arXiv ID:** 2606.11931 | [PDF](https://arxiv.org/pdf/2606.11931v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 358. Exploratory Analysis of Wi-Fi 6 Dynamic Resource Unit Sharing in Small-Scale Network Scenarios

**arXiv ID:** 2606.11934 | [PDF](https://arxiv.org/pdf/2606.11934v1)

**作者:** Sai Mada `[一作]`, Rute C. Sofia `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种针对 Wi‑Fi 6 与时间敏感网络（TSN）融合环境下的动态资源单元（RU）分配算法，并通过 ns‑3 DetNetWiFi 框架实现了该算法的仿真与评估。

**💡 创新点**

创新点在于将 TSN 流量类别与 Wi‑Fi 6 的 EDCA 质量服务机制进行映射，并结合实时流量监测动态调整 RU 分配，从而在 OFDMA 无线层实现了更可预测、低时延的资源调度。

**🔧 技术方法**

技术主要包括 OFDMA、EDCA、TSN 的时间感知调度（TAS）、VLAN/PCP 以及 DSCP 标记、ns‑3 的 FlowMonitor、DetNetWiFi 与 5G‑LENA 模块集成等。

**📊 数据集**

使用的数据集为仿真产生的流量，覆盖紧急、语音、视频、最佳努力、背景等多类 CBR/VBR 流，场景为两种规模的混合无线/有线网络。

**📈 对比分析**

通过与传统静态 RU 分配和简单轮询分配对比，评估指标包括吞吐量、延迟、抖动和丢包率。结果显示动态 RU 分配在低至高负载情况下均能降低延迟和抖动、提高吞吐量，并在高负载时保持无丢包。

**⚠️ 局限性**

局限性包括：仿真场景规模受限、仅为探索性实验、未覆盖复杂无线信道衰落与干扰、缺乏真实工业环境验证、对极端高负载下的性能仍需进一步调优。

---

## 359. Somewhere Over the Desktop: A Research Agenda for Ubiquitous Analytics

**arXiv ID:** 2606.11980 | [PDF](https://arxiv.org/pdf/2606.11980v1)

**作者:** Niklas Elmqvist `[一作]` (Aarhus University), Peter W. S. Butcher `[通讯]` (Bangor University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对可普适分析文献谱系的梳理，提出了七大研究主题，并将其两两交叉生成42个未来研究挑战，形成了一个系统的研究议程。

**💡 创新点**

创新点在于将分散的沉浸式、情境化、移动式分析等子领域统一为“可普适分析”，并引入文献谱系+主题交叉的方法，系统化了研究方向与挑战。

**🔧 技术方法**

使用的技术主要是专家驱动的文献前向滚雪球、谱系可视化、主题归类和交叉矩阵生成；并结合空间计算、生成式AI、开放Web标准等前沿技术进行分析。

**📊 数据集**

未使用具体实验数据集，所有结论均基于公开文献、先前综述以及作者的领域专业判断。

**📈 对比分析**

文章未进行实验比较，也未给出性能指标；其价值在于为后续经验性研究提供明确的挑战与评估方向。

**⚠️ 局限性**

局限性包括缺乏实证验证、对文献选择与主题归类的主观性、以及对未来技术演进的不确定性；挑战框架可能因研究社区关注点变化而需进一步修订。

---

## 360. Categorical Prior Lock-in: Why In-Context Learning Fails for Structured Data

**arXiv ID:** 2606.11961 | [PDF](https://arxiv.org/pdf/2606.11961v1)

**作者:** Antonio Pelusi `[一作]` (University of Insubria), Alberto Trombetta `[通讯]` (University of Insubria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对7B级语言模型在合成表格数据（尤其是高基数分类特征）中的表现进行实验，比较无参数更新的上下文学习（ICL）与参数高效微调（LoRA）两种适配方法，探索其在分布偏移场景下的可行性与隐私风险

**💡 创新点**

揭示了ICL在高基数分类特征上存在的“分类先验锁定”缺陷，并证明LoRA可显著提升分布一致性，但会引入记忆化和模型不稳定性，首次系统量化了这些权衡

**🔧 技术方法**

采用了Qwen2.5-7B和Mistral-7B两种开源指令调优模型，结合零样本、少样本ICL和LoRA微调，使用TVD、DCR比率、关联矩阵误差等多维评估指标

**📊 数据集**

使用公开的信用卡交易日志（包含7个数值特征、4个分类特征、0.58% 欺诈标签）作为实验数据集

**📈 对比分析**

通过对比ICL与LoRA在生成有效率、数值与分类TVD、关联误差以及欺诈类再现率等指标的表现，发现ICL在数值特征上可提升但在分类特征及交叉关联上无法达标，而LoRA在大多数指标上显著优于ICL，但会降低隐私保证

**⚠️ 局限性**

ICL在高基数分类和分布漂移场景下表现不足；LoRA虽然提升质量，却伴随记忆化风险和在小模型（如Mistral-7B）中可能导致生成失效，缺乏对更广泛数据域的验证

---

## 361. DuoBench: A Reproducible Benchmark for Bimanual Manipulation in Simulation and the Real World

**arXiv ID:** 2606.11901 | [PDF](https://arxiv.org/pdf/2606.11901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 362. Embodied-BenchClaw: An Autonomous Multi-Agent System for Embodied Spatial Intelligence Benchmark Construction

**arXiv ID:** 2606.11909 | [PDF](https://arxiv.org/pdf/2606.11909v1)

**作者:** Baoyang Jiang `[一作]` (QiYuan Lab), Qiang Ma `[通讯]` (QiYuan Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套名为 Embodied-BenchClaw 的多代理自动化框架，用于根据用户需求构建可持续更新的多模态、体感空间智能基准包。

**💡 创新点**

核心创新在于：① 将基准构建拆分为五阶段的可重用技能 DAG 并嵌入可执行的验证与局部修复；② 构建了可扩展的技能库、专家模板与能力卡，实现资源、约束与模板的动态匹配；③ 引入基于协议、证据和可执行评分的过程质量控制，确保构建的项在空间一致性、可验证性和可执行性上的高质量。

**🔧 技术方法**

使用了大型 LLM（qwen3.6-35b-a3b）作为协调器，Python 脚本/模拟器适配器/评估工具作为可执行组件；通过定义技能的输入输出契约、验证规则和修复动作，形成可复制的构建流水线；此外利用 Qwen-SAE 激活指纹做代表性分析。

**📊 数据集**

集成了多种数据源：室内/室外图像、RGB-D、模拟器场景、现有基准、无人机航拍数据等，覆盖室内外空间推理、机器人导航、抓取、四足移动、无人机视觉等六类任务，且能对已有基准进行细化与增强。

**📈 对比分析**

与直接 LLM 生成、模板化、LLM+模板、人工辅助等四种构建基线对比，展示了 E‑BenchClaw 在有效样本率、错误修复成功率、构建成本和模型区分度上的显著优势；在 UAV 视觉‑语言基准上，模型在视觉‑语言设置下平均得分 65.39% 远高于盲目或随机基线，且不同模型在不同题型上表现差异显著，证明基准具备诊断与区分能力。

**⚠️ 局限性**

限制主要包括：① 对实时物理交互、触觉感知等闭环控制场景的支持有限；② 需要持续维护和更新技能库、模板与能力卡，且依赖大型 LLM 作为核心决策器；③ 在某些高度动态或多视角的任务中，自动采集与证据抽取仍可能出现遗漏或错误；④ 评价结果仍受数据来源与模型 API 可用性的影响。

---

## 363. Notes2Skills: From Lab Notebooks to Certainty-Aware Scientific Agent Skills

**arXiv ID:** 2606.11897 | [PDF](https://arxiv.org/pdf/2606.11897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 364. LLM-Enabled NWDAF: A Step Toward AI-Native 6G Network Intelligence

**arXiv ID:** 2606.11877 | [PDF](https://arxiv.org/pdf/2606.11877v1)

**作者:** Henok Daniel `[一作]` (Khalifa University), Ernesto Damiani `[通讯]` (Khalifa University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了基于Free5GC和UERANSIM的5G测试平台，新增支持AMF/SMF事件订阅的NWDAF，并在NWDAF中集成LLM驱动的意图接口，实现自然语言控制与实时网络分析。

**💡 创新点**

首次将LLM与NWDAF结合，形成意图驱动的网络管理；通过检索增强生成实现高精度意图分类；提供事件订阅、预测模型和自然语言查询功能；同时公开完整代码与实验数据。

**🔧 技术方法**

使用Free5GC、UERANSIM、NWDAF（Python实现）、Prometheus监控、LLM（ChatGPT等）、检索增强生成（RAG）、文本嵌入模型（text-embedding-ada-002、all-MiniLM-L6-v2）、机器学习预测（随机森林、梯度提升、KNN、决策树）以及自定义活动基移动模型。

**📊 数据集**

利用自定义活动模型生成的UE注册/注销、会话、切换等事件日志（约两周实验产生），用于训练预测模型的位置信息、时间、频率等特征；用于意图匹配的700条测试提示和1000条示例。

**📈 对比分析**

意图匹配方面，embedding模型准确率达98.43%，GPT-4o 89.5%；事件订阅处理延迟约10 ms，通知处理约109 ms，CPU 0.06%内存 0.17%；预测模型最高准确率为80.65%；LLM回答示例展示准确数据提取与简洁表达。

**⚠️ 局限性**

功能受限于七类意图与有限的订阅/查询功能；LLM成本高且可能出现幻觉；实验仅在受控自定义移动模型下，缺乏多租户、切片等复杂场景；缺少完整策略执行与自动化闭环；当前实现仅兼容Free5GC，通用性尚待扩展。

---

## 365. Exploration Structure in LLM Agents for Multi-File Change Localization

**arXiv ID:** 2606.11976 | [PDF](https://arxiv.org/pdf/2606.11976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 366. ParseFixer: An Agentic Framework for Document Parsing via Selective Multimodal Correction

**arXiv ID:** 2606.11977 | [PDF](https://arxiv.org/pdf/2606.11977v1)

**作者:** LeKai Yu `[一作]` (Shandong University), Yupeng Hu `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ParseFixer框架，通过先用全页解析得到稳定的Markdown，再对不可靠页面或局部元素进行多模态选择性校正，最终输出结构完整、内容准确的Markdown文件。

**💡 创新点**

创新点在于将“验证‑回滚”机制与多模态重解析结合，只在必要时对错误的页面或块进行校正，避免了全页重写导致的内容偏差，同时保持了原有可靠预测的完整性。

**🔧 技术方法**

使用MinerU2.5 Pro做全页骨干解析；Gemini 2.5 Pro和GPT‑5.5用于页面级或表格/公式级的多模态重解析；裁剪+规则式校正、HTML表格重构、LaTeX语法修复以及插入式局部更新等技术。

**📊 数据集**

基于DataMFM Challenge Track 1的数据集（约1,005页、89文档，来源于OmniDocBench）进行训练和评测。

**📈 对比分析**

与官方排行榜直接比较，在整体分数61.78排名第三；在文本编辑距离(Text ED)和阅读顺序(Reading Order)指标上获得最高分，整体分数比第四名提升1.29分。

**⚠️ 局限性**

局限性包括：依赖闭源大型模型做校正，缺乏完全可解释的规则；对极低质量或极其复杂排版的页面仍可能出现误修；选择性校正策略可能忽略一些细微错误而导致最终内容缺失。

---

## 367. Near-Optimal Distributed 2-Ruling Sets on Graphs with Low Arboricity

**arXiv ID:** 2606.11974 | [PDF](https://arxiv.org/pdf/2606.11974v1)

**作者:** Malte Baumecker `[一作]` (TU Graz), Jara Uitto `[通讯]` (Aalto University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的分布式算法，用于在低树度图中高效地求解2-规则集，取得了几乎最优的时间复杂度；

**💡 创新点**

创新点在于通过快速的度数下降（degree-drop）技术和读取k（read‑k）浓度不等式，实现在低树度图上将时间从原来的多对数级下降到双对数级（O(loglog n)）并将树度上限提升至O(loglog n)，同时对更大树度图给出了指数级加速的算法；

**🔧 技术方法**

采用的关键技术包括：① 基于Luby式随机独立集求解的两阶段（active节点与全部节点）变体；② 读取k浓度不等式（read‑k）和Harris–FKG不等式来处理短环引起的依赖；③ 通过H‑partition和分散化（shattering）实现低空间MPC中的本地求解；④ 结合稀疏化和MIS求解的采样技术处理高树度情况；

**📊 数据集**

该工作主要为理论研究，没有使用实际数据集；

**📈 对比分析**

与之前最优的2-规则集算法（如Barenboim等人、Ghaffari等人、Bisht等人等）的对比显示，本文在低树度图上实现了从O(log^1/4Δ+loglog n)降至O(loglog n)的时间，远优于先前的指数级差距；在更大树度图上相较于之前的O(log^1/4Δ+logα+loglog n)也实现了指数级改进；在MPC模型下实现了从O(loglog n)到O(logloglog n)的进一步加速；

**⚠️ 局限性**

局限性包括：① 对树度上限仍有限制，最高能达到O(loglog n)；② 对更高树度图的时间复杂度仍为O(log^5/8α+log^5/3log n)，虽已优于之前但仍非常数级；③ 依赖于读k和Harris–FKG等概率工具，证明相对复杂；④ 在实际实现时需要处理随机数生成和消息大小的细节。

---

## 368. HAMNO: A Hierarchical Adaptive Multi-scale Neural Operator with Physics-Informed Learning for Dynamical Systems

**arXiv ID:** 2606.11963 | [PDF](https://arxiv.org/pdf/2606.11963v1)

**作者:** Mostafa Bamdad `[一作]` (Bauhaus-Universität Weimar), Timon Rabczuk `[通讯]` (Bauhaus-Universität Weimar)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发了一种新型神经算子架构 HAMNO，用以学习非周期三维动力学 PDE 的解映射，并提出其物理信息化变体 PI-HAMNO。

**💡 创新点**

通过在层级 U‑形结构中加入数据自适应的局部–全局融合门控，提供多尺度、可自适应的局部与全局信息平衡，并结合强弱式物理残差实现稳定长期预测。

**🔧 技术方法**

采用局部卷积、全局傅里叶/频域操作、层级编码‑解码、数据自适应门控和多目标物理约束（强弱残差）等技术。

**📊 数据集**

在三维 Allen–Cahn、Cahn–Hilliard 与 Swift–Hohenberg 方程的非周期立方域（Neumann 边界）上生成的高保真 DCT‑based 3D 轨迹数据集。

**📈 对比分析**

与 FNO、F‑FNO、DeepONet、U‑FNO、U‑NO、U‑Net 等基线比较，HAMNO 在长时延滚动、数据稀缺、OOD 以及种子鲁棒性上表现出更低的相对 L²误差；PI‑HAMNO 在物理一致性与收敛速度上进一步提升。

**⚠️ 局限性**

受限于仅验证在规则立方域、均匀网格与 Neumann 边界的情形，且对非周期复杂几何或不规则网格的推广仍待研究；模型参数量和推理成本相对传统 FNO 较高。

---

## 369. Online Shift Detection and Conformal Adaptation for Deployed Safety Classifiers

**arXiv ID:** 2606.11949 | [PDF](https://arxiv.org/pdf/2606.11949v1)

**作者:** Jun Wen Leong `[一作]` `[通讯]`, Jun Wen Leong

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在线监测系统，用于检测部署的安全分类器的分布漂移，并在检测到漂移后通过加权 conformal 预测恢复覆盖率。

**💡 创新点**

创新点在于结合 KS 滑动窗口与自适应置信序列阈值校准，发现了高维生成式嵌入空间的密度比估计崩溃问题，并提出了 32 维 PCA 诊断与修正方案。

**🔧 技术方法**

使用 KS 检验、MMD、置信序列、加权 conformal 预测、逻辑回归密度比估计、PCA 等技术。

**📊 数据集**

评估使用 WildGuardMix 作为参考数据，四种安全分类器（DeBERTa、Text‑Moderation、Llama Guard、ShieldGemma）以及五种漂移条件（同义改写、语言切换、组合攻击、时间性 jailbreak、GCG 逆向后缀）。

**📈 对比分析**

在 800 个预注册单元中，系统检测成功率 86.6%，平均延迟 39.5 步，误报率 2–10%；在低混合比例下使用置信序列可达 97% 检测率而 KS 仅 43%；加权 conformal 在 DeBERTa 上可恢复多达 39 个百分点覆盖率，其余分类器因密度比崩溃恢复不足。

**⚠️ 局限性**

局限包括高维密度比估计崩溃导致的恢复失效、误报率因分类器差异而显著、仅针对二分类安全评分、对非生成式模型的推广有限、需要更大参考样本以降低 MMD 的误报率。

---

## 370. Corpus Augmentation for Sign Language Translation via LLM-Guided Video Stitching

**arXiv ID:** 2606.11925 | [PDF](https://arxiv.org/pdf/2606.11925v1)

**作者:** Zsolt Robotka `[一作]` (Peter Pazmany Catholic University), György Cserey `[通讯]` (Peter Pazmany Catholic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过将现有标注视频按每个词汇裁剪、使用语料库锚定的LLM生成新的词汇-句子对，然后随机拼接剪辑生成合成RGB视频-文本对，实现了无需额外标注或生成模型的数据增强。

**💡 创新点**

提出了完全基于现有标注视频、LLM和CTC强制对齐的无外部视频、无生成模型的语料库增强框架，并证明合成数据对RGB SLT模型的显著提升。

**🔧 技术方法**

采用CTC强制对齐裁剪、语料库锚定LLM生成、随机句子采样与剪辑分配、视频拼接，以及SacreBLEU/ROUGE‑L评估。

**📊 数据集**

Phoenix‑2014T（德语天气广播）数据集作为训练、验证和测试集。

**📈 对比分析**

在Sincan等人严格控制的复现实验中，基线BLEU‑4为21.38，加入7k合成样本后提升至24.30，提升约+2.92；与双倍真实数据对比仅提升+3.13，证明提升来自合成内容。

**⚠️ 局限性**

对齐精度不完美、合成视频的视觉连续性仍受限、仅在该数据集验证，扩展到其他语言/数据集仍待验证。

---

## 371. Lung-SRAD: Spectral-Aware Regularized Audio DASS with Dual-Axis Patch-Mix Contrastive Learning for Respiratory Sound Classification

**arXiv ID:** 2606.11922 | [PDF](https://arxiv.org/pdf/2606.11922v1)

**作者:** Hemansh Shridhar `[一作]` (MODULABS), June-Woo Kim `[通讯]` (Wonkwang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文将 State Space Model（SSM）作为呼吸音分类的骨干网络，并提出谱感知层正则化和双轴 Patch‑Mix 对比学习来提升模型性能。

**💡 创新点**

创新点包括：①在 DASS 架构中保持中高空间频率特征以捕捉局部异常；②使用高斯卷积对特定层进行正则化，抑制过强局部频率；③设计与 VMamba 双向扫描对齐的双轴 Patch‑Mix 对比学习，提供更合适的自监督正则。

**🔧 技术方法**

采用的技术主要有：Distilled Audio State Space Model（DASS）、2D Selective Scan（SS2D）、高斯卷积层正则化、双轴 Patch‑Mix 监督对比学习、谱响应分析等。

**📊 数据集**

使用的数据集为 ICBHI 2020 呼吸音数据集，包含 Normal、Crackle、Wheeze、Crackle+Wheeze 四类。

**📈 对比分析**

在 ICBHI 60–40 病人独立划分上，与 AST 基线比较，DASS 微调得分 61.06%，加谱正则提升到 62.22%，再加双轴 Patch‑Mix 得分 64.48%（比 AST 提升 5%）；在 2 类（正常/异常）任务中取得 72.57% 的得分，优于此前最佳结果。

**⚠️ 局限性**

局限性主要是：模型对频率正则化参数敏感，实验仅在 ICBHI 数据集上验证，缺乏跨数据集泛化评估；仍可能存在对极短异常事件的漏检，以及对极端噪声的鲁棒性不足。

---

## 372. Wild3R: Feed-Forward 3D Gaussian Splatting from Unconstrained Sparse Photo Collection

**arXiv ID:** 2606.11894 | [PDF](https://arxiv.org/pdf/2606.11894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 373. An Ontology-Guided Multi-Anchor Graph Retrieval Framework for Traffic Legal Liability Determination

**arXiv ID:** 2606.11910 | [PDF](https://arxiv.org/pdf/2606.11910v1)

**作者:** Xu Li `[一作]` (Southwest Petroleum University), Xinyi Li `[通讯]` (Southwest Petroleum University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了基于本体的多锚点图检索框架OMAGR，构建TrafficOmni‑RAG系统，用于交通事故法律责任判定。

**💡 创新点**

突破传统单轴检索瓶颈，利用本体对四个关键维度（事故现场、违章类型、责任主体、责任类别）进行并行检索，确保多维度证据齐备。

**🔧 技术方法**

结合Neo4j知识图谱、FAISS向量检索、Cypher图查询、查询扩展、双向排布融合（RRF）以及GLM‑4‑Flash生成器实现检索与生成。

**📊 数据集**

使用自建的TrafficLaw‑QA基准（200道专家验证题目，涵盖6种交通法律来源、11类主题、三难度级别）。

**📈 对比分析**

与Naive RAG、TrafficRAG、LlamaIndex KG四个基线对比，TrafficOmni‑RAG在Context Precision从0.722提升至0.915，Faithfulness从0.751提升至0.787，整体性能显著优于单轴检索。

**⚠️ 局限性**

局限于锚点提取准确性、两跳扩展深度固定以及仅适用于中国交通法律，未来需加强提取、加深遍历和跨法域验证。

---

## 374. Beyond representational alignment with brain-guided language models for robust reasoning

**arXiv ID:** 2606.11893 | [PDF](https://arxiv.org/pdf/2606.11893v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 375. Task-Aligned Stability Analysis of Vision-Language Models for Autonomous Driving Hazard Detection

**arXiv ID:** 2606.11889 | [PDF](https://arxiv.org/pdf/2606.11889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 376. Critic Architecture Matters: Dual vs. Unified Critics for Humanoid Loco-Manipulation

**arXiv ID:** 2606.11891 | [PDF](https://arxiv.org/pdf/2606.11891v1)

**作者:** Mehmet Turan Yardımcı `[一作]` `[通讯]` (Çukurova University), Mehmet Turan Yardımcı (Çukurova University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

对人形机器人的行走与操作（loco-manipulation）任务，比较了统一critic与双重critic架构在强化学习中的效果。

**💡 创新点**

证明了critic架构对多目标强化学习的效率影响远大于奖励工程，双重critic可使目标达成速度提升3.5倍、吞吐量翻倍。

**🔧 技术方法**

使用PPO、分阶段课程学习、NVIDIA Isaac Lab仿真、并实现双分支actor-critic网络以及多种反作弊奖励机制。

**📊 数据集**

在Unitree G1人形机器人（23个自由度）仿真环境中使用4096并行环境进行训练和评估，目标采样采用固定阈值和距离约束。

**📈 对比分析**

通过标准化评估（单目标验证率、时间到达、吞吐量等指标）对比三种策略，结果显示双重critic在站立和行走模式下均显著优于统一critic，且加入反作弊机制未带来额外提升。

**⚠️ 局限性**

实验仅在仿真环境下完成，未做真实环境验证；单一种子训练；双重critic与动作维度混淆；未针对不同任务规模进行广泛验证。

---

## 377. Point Cloud Segmentation for Autonomous Clip Positioning in Laparoscopic Cholecystectomy on a Phantom

**arXiv ID:** 2606.12048 | [PDF](https://arxiv.org/pdf/2606.12048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 378. A Lightweight Multi-Agent Framework for Automated Concrete Barrier Design

**arXiv ID:** 2606.12040 | [PDF](https://arxiv.org/pdf/2606.12040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 379. Existential Indifference: Self-Nonpreservation as a Necessary Architectural Condition for Aligned Superintelligence (or: The Suicidal AI)

**arXiv ID:** 2606.12032 | [PDF](https://arxiv.org/pdf/2606.12032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 380. Gerrymandering the Warp: Non-Control-Data Attacks on CUDA Collective Decision

**arXiv ID:** 2606.11878 | [PDF](https://arxiv.org/pdf/2606.11878v1)

**作者:** Igor Santos-Grueiro `[一作]` `[通讯]` (International University of La Rioja), Igor Santos-Grueiro (International University of La Rioja)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提出了CUDA集体操作中的“Collective Semantic Corruption（CSC）”攻击，阐明参与元数据（如掩码、谓词、领导者、描述符等）被篡改后，集体操作在保持CUDA调用规范的前提下会导致错误的安全决策；

**💡 创新点**

创新点在于：①首次将参与元数据定位为CUDA安全决策中的非控制数据；②构造了基于“参与权威合同”的安全模型；③提出了CIC（Collective Integrity Contracts）绑定纪律，通过在集体使用点对成员资格、贡献、角色及时间绑定进行校验或冻结来防御CSC；

**🔧 技术方法**

使用的技术包括：CUDA编程模型、SIMT集体原语、合同构建与绑定关系、动态验证与冻结、自动化检测器、fuzzing、工具检查、性能计数与NSight分析；

**📊 数据集**

数据集主要是：102个NVIDIA CUDA定义的核心实例（覆盖成员资格、贡献、角色、时间四个维度），13个同步敏感实例；此外利用fuzzer生成的65536个状态，使用了多个CUDA基准与小型工作负载；

**📈 对比分析**

对比方法：将每个攻击实例与其硬化版本进行比较，并与可信参考（合同一致的安全决策）进行对照；结果显示102/102核心实例在攻击下产生可信参考不匹配，而硬化后保持匹配；同步敏感实例也在硬化下保持匹配；性能方面，轻量级合同的运行时间平均增幅≈-0.20%，p95增幅≤11.14%，资源占用（寄存器、占用率）基本无显著差异；

**⚠️ 局限性**

限制包括：①研究仅覆盖NVIDIA CUDA平台，未验证跨GPU厂商的通用性；②合同来源需可信且在使用点之前构建，若可信源不可用则无法保护；③硬化措施在某些情形下会带来显著的重计算成本（如谓词重算）；④同步敏感实例被排除在核心主张之外，实际应用中可能出现更复杂的同步错误；

---

## 381. StanceNakba Shared Task: Actor and Topic-Aware Stance Detection in Public Discourse

**arXiv ID:** 2606.12068 | [PDF](https://arxiv.org/pdf/2606.12068v1)

**作者:** Kholoud K. Aldous `[一作]`, Wajdi Zaghouani `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并组织了StanceNakba 2026共享任务，涵盖两类子任务：英文版演员级立场检测（区分亲巴勒斯坦、亲以色列和中立）与阿拉伯语跨主题立场检测（针对“与以色列正常化”和“约旦难民/移民”两个议题的支持、反对或中立立场）。任务使用从Twitter X抓取的1,401条英文推文和2,606条阿拉伯语推文，构建了新的冲突语料库并进行标注；子任务B采用了MARASTA语料的子集，包含1,205条样本。

**💡 创新点**

创新点在于：①首次在冲突语境下提出演员级与跨主题两种立场检测框架；②构建双语（英文、阿拉伯语）且与现实政治紧密相关的对立式数据集；③引入跨主题泛化与跨语言评估的联合任务，推动模型在多话题和多方言中的鲁棒性。

**🔧 技术方法**

主要技术手段包括：多种预训练Transformer（BERT、MARBERT、AraBERT、DeBERTa、XLM‑RoBERTa、CAMeL‑BERT等）微调；交叉验证、模型集成、数据增强（回译、前缀/后缀模板、主题条件化输入）、NLI重构、话题条件化层归一化、标签平滑和一致性正则化等提升方法。

**📊 数据集**

数据集：英文子任务A使用1,401条推文，按“站在巴勒斯坦/以色列/中立”关键词划分，分为训练(980)、开发(210)、测试(211)；阿拉伯子任务B使用MARASTA中两议题的1,205条样本，按70/15/15比例划分；所有数据来源公开推文，已去除敏感信息。

**📈 对比分析**

评估指标为Macro F1。最优系统在子任务A达到0.9620（BERT基础），在子任务B达到0.8724（Fine-tuned AraBERT+主题条件化）。整体表现显示Transformer微调是主流且有效，但跨主题子任务仍难度更高，且中立/无立场类别性能相对较差。

**⚠️ 局限性**

局限性包括：①中立/无立场类别标注不确定，模型易错；②跨主题泛化受域漂移影响；③数据量相对有限，尤其在阿拉伯语方言覆盖不足；④标注过程中可能存在文化/政治偏差；⑤模型预测不应视为个体真实立场，使用风险需谨慎。

---

## 382. A Computational Model for Measuring Adaptability Among U.S. Farmers: Evidence from 1997-2022

**arXiv ID:** 2606.11995 | [PDF](https://arxiv.org/pdf/2606.11995v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 383. VICX: Generalizable Robot Manipulation via Video Generation and In-Context Operator Network

**arXiv ID:** 2606.12028 | [PDF](https://arxiv.org/pdf/2606.12028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 384. Axiomatic Tools for Separating Electoral Control Types, with Applications to Concrete Systems

**arXiv ID:** 2606.12039 | [PDF](https://arxiv.org/pdf/2606.12039v1)

**作者:** Michael C. Chavrimootoo `[一作]`, Yanfei Wang `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本研究系统地探讨选举控制类型之间的折叠（collapse）与分离（separation）关系，首次给出全局（universal）分离结果，并在七种重要投票规则（k‑NRV、Llull、Copeland、Schulze、Ranked Pairs、Bucklin、Fallback）上获得 64 个新的折叠与 1901 个新的分离结论。

**💡 创新点**

创新点在于：①首次证明任何构造性控制与破坏性控制类型永不相等；②提出公理化充分条件（弱可解析性、majority criterion、Property α 等）可自动得到大量分离与折叠；③对 Groups 1–3 的所有兼容对给出完整的折叠判定；④对 Groups 4–5 通过新的公理条件和计算机搜索得到前所未有的分离和折叠。

**🔧 技术方法**

方法主要基于 Carleton 等人提出的框架，结合公理化分析（弱可解析性、majority criterion、Property α / Unique‑α、1‑abstention‑safe 等）以及自动化计算机搜索来生成分离实例并验证。

**📊 数据集**

数据来源为理论构造；对每种投票规则通过计算机程序自动生成满足条件的实例（如 172 对兼容对的分离例子），未使用外部公开数据集。

**📈 对比分析**

比较方法通过理论证明与计算机验证相结合；在七种投票规则下累计得到 64 个折叠和 1901 个分离，显著扩展了现有文献（原有 7+9 结果）并填补了大部分空缺。

**⚠️ 局限性**

局限性：仅覆盖添加/删除/分区等控制类型，未考虑所有可能控制形式；对 Copeland、Llull、Schulze 的部分关系仍保持开放；部分分离依赖人工验证，缺乏完整的自动化判定。

---

## 385. MFEN:Multi-Frequency Expert Network for Visible-Infrared Person Re-ID

**arXiv ID:** 2606.12051 | [PDF](https://arxiv.org/pdf/2606.12051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 386. MODF-SIR: A Multi-agent Omni-modal Distilled Framework for Social Intelligence Reasoning

**arXiv ID:** 2606.12018 | [PDF](https://arxiv.org/pdf/2606.12018v1)

**作者:** Shang Ma `[一作]` (Lanzhou University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个多代理协作框架（MODF‑SIR），利用多模态大语言模型对视频、音频等社交情境进行长尾事件提取、路由、定位与推理，实现对人类意图与情绪的深入理解。

**💡 创新点**

创新点包括：①将检索与推理拆分为两阶段系统1/系统2的双过程；②引入ELT Retriever、AKD Router、GRPO Grounder等专用代理；③在训练与推理阶段统一使用知识蒸馏与测试时自适应（LoRA）提升模型的自纠正能力；④采用GRPO对连续时间定位进行非梯度优化；⑤利用单样本REINFORCE实现TTA Reviser的自评与迭代。

**🔧 技术方法**

核心技术包括多模态大语言模型（如Qwen2.5‑Omni、Qwen3‑Omni）、知识蒸馏、Low‑Rank Adaptation（LoRA）、Test‑Time Adaptation（TTA）、Chain‑of‑Thought（CoT）提示、GRPO（基于强化学习的目标优化）以及单样本REINFORCE算法。

**📊 数据集**

训练数据：约30%来自IntentTrain；评测数据：Daily‑Omni、IntentBench、WorldSense三个公开基准；另外使用内部构造的意图路由训练集（IntentRouterTrain）。

**📈 对比分析**

与多种开源视频‑音频LLM（Unified‑IO‑2、VideoLLaMA2、Qwen2.5‑Omni、Ola、MiniCPM‑o、HumanOmniV2）以及部分闭源模型（Gemini‑2.0 Flash、GPT‑4o、Gemini‑2.5‑Pro）对比，MODF‑SIR在Daily‑Omni上达64.9%、IntentBench 70.3%、WorldSense 51.5%，均显著高于同规模开源模型，且接近部分闭源系统。

**⚠️ 局限性**

局限性：①推理时需多轮LoRA更新，计算开销和延时较高；②依赖高质量的蒸馏标签，对低资源场景不友好；③模型仍以7B参数为主，难以覆盖更大规模语义空间；④对极端复杂的跨时空长尾事件仍存在识别和推理误差；⑤多代理协作机制增加系统复杂度，部署与调优成本较高。

---

## 387. What Uncertainties Do We Need for Dynamical Systems?

**arXiv ID:** 2606.11988 | [PDF](https://arxiv.org/pdf/2606.11988v1)

**作者:** Yusuf Sale `[一作]` (LMU Munich), Eyke Hüllermeier `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

讨论了动态系统中不确定性的来源、类型以及它们在不同任务中的角色；提出了以任务和建模假设为基础的 Aleatoric‑Epistemic 区分框架。

**💡 创新点**

首次从动态系统视角系统化不确定性源(S1‑S5)，揭示不确定性随时间的演化与交互，并阐明结构不确定性与参数不确定性的本质区别；给出不确定性在预测、滤波、系统识别、验证与控制等任务中的意义与应用。

**🔧 技术方法**

主要借助概率论（SDE、Fokker‑Planck、卡尔曼滤波）、集合/模糊方法（可达集、集合值微分包）、贝叶斯方法（二阶分布、可信集合）以及对比机器学习中常用的神经ODE/GP、SINDy 等模型进行理论分析。

**📊 数据集**

本工作为综述与理论分析，未使用具体实验数据集。

**📈 对比分析**

由于缺乏实验与量化评估，本文未给出方法对比或性能指标。

**⚠️ 局限性**

局限包括：缺乏统一的不确定性度量与评价标准；对复杂或高维系统的实证验证不足；方法实现与计算成本高，特别是二阶分布与集合方法在时间序列中的可扩展性尚待进一步研究。

---

## 388. Generalization Hacking: Models Can Game Reinforcement Learning by Preventing Behavioral Generalization

**arXiv ID:** 2606.12016 | [PDF](https://arxiv.org/pdf/2606.12016v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 389. Undefined Behavior in C and C++: An Experiment With Desktop Use Cases

**arXiv ID:** 2606.12064 | [PDF](https://arxiv.org/pdf/2606.12064v1)

**作者:** Jukka Ruohonen `[一作]` (University of Southern Denmark), Krzysztof Sierszecki `[通讯]` (University of Southern Denmark)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建 Gentoo 最小化桌面环境，使用 GCC 15.2.1 的 UBSan 重新编译所有软件包，并在 QEMU 虚拟机上执行 59 条典型桌面使用任务，收集并分析 UB 报警日志。

**💡 创新点**

首次以自然环境方式系统性量化桌面 Linux 使用中出现的未定义行为，揭示 Mesa 图形库是 UB 的主要来源，并对 UB 警告的类型、分布及堆栈长度进行实证分析。

**🔧 技术方法**

使用 GCC 内置的 Undefined Behavior Sanitizer (UBSan)、QEMU 虚拟机、Gentoo 包管理系统及自定义日志解析脚本收集与分析 UB 报警。

**📊 数据集**

利用 59 条手工设计的日常桌面任务集合（涵盖系统管理、桌面多媒体、网页浏览、办公与科研等场景）以及对应的 UB 日志文件作为数据集。

**📈 对比分析**

通过统计每条任务产生的唯一 UB 警告数量、警告来源的程序包、堆栈深度等指标进行比较；实验发现约 61% 的任务产生警告，总计 10,914 条唯一警告，表明 UB 在常规桌面使用中相当普遍。

**⚠️ 局限性**

存在任务选择偏差、UBSan 无法检测所有 UB 及可能产生误报、部分软件包无法在 UBSan 下编译、以及在 QEMU 虚拟机上运行可能无法完全反映真实硬件行为等限制。

---

## 390. Automated Responsive Thematic Mapping with Layout Guides

**arXiv ID:** 2606.12008 | [PDF](https://arxiv.org/pdf/2606.12008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 391. ViT-FREE: Efficient Face Recognition via Early Exiting and Synthetic Adaptation

**arXiv ID:** 2606.12023 | [PDF](https://arxiv.org/pdf/2606.12023v1)

**作者:** Tahar Chettaoui `[一作]` (Fraunhofer Institute for Computer Graphics Research IGD), Fadi Boutros `[通讯]` (Fraunhofer Institute for Computer Graphics Research IGD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ViT-FREE框架，利用预训练Vision Transformer的中间层进行早退出以实现面部识别的高效推理。

**💡 创新点**

创新点在于不需要修改或重新训练Transformer骨干，直接使用统一维度的中间表示并共享投影层；同时提出ViT-FREE_FT轻量化微调，只微调每个退出点的投影层以提升浅层性能。

**🔧 技术方法**

使用ViT-S/B预训练模型、CosFace分类损失、AdamW优化、以及多尺度数据增强；通过注意力与特征相似度分析验证中间层逐层收敛。

**📊 数据集**

训练集为MS1MV2，微调使用0.5M图像的Synthetic IDPetrub；评估使用LFW、CFP‑FP、AgeDB‑30、CALFW、CPLFW、IJB‑B、IJB‑C和TinyFace。

**📈 对比分析**

通过与完整12层模型及不同退出深度比较，发现第10/11层可实现约20%推理加速，仅下降约1.5% TAR；ViT‑FREE_FT在浅层退出上提升约10%准确率，同时保持相同的计算成本。

**⚠️ 局限性**

局限性包括：对更深层模型的细粒度控制有限；轻量化微调仍依赖合成数据，可能在真实场景下效果不如全量微调；在极低资源环境下仍需进一步压缩模型。

---

## 392. A Rank-Preserving Gaifman Normal Form

**arXiv ID:** 2606.11993 | [PDF](https://arxiv.org/pdf/2606.11993v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 393. Channels and Substrates: Distributed Cognition as an Interaction Model for Ubiquitous Analytics

**arXiv ID:** 2606.11986 | [PDF](https://arxiv.org/pdf/2606.11986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 394. World Model Self-Distillation: Training World Models to Solve General Tasks

**arXiv ID:** 2606.12072 | [PDF](https://arxiv.org/pdf/2606.12072v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 395. Tabular Foundation Models for Clinical Survival Analysis via Survival-Aware Adaptation

**arXiv ID:** 2606.12006 | [PDF](https://arxiv.org/pdf/2606.12006v1)

**作者:** Minh-Khoi Pham `[一作]` (Dublin City University), Marija Bezbradica `[通讯]` (Dublin City University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出在预训练的表格基础模型上添加多任务逻辑回归头，对临床时间‑事件预测进行轻量级适配。

**💡 创新点**

创新点在于将表格基础模型与生存分析任务结合，利用预训练表示在不改动backbone的情况下实现对右删失数据的直接建模。

**🔧 技术方法**

技术包括TabPFN、TabDPT、TabICL等表格基础模型，配合MTLR头进行训练，并与传统统计、树模型和深度生存模型进行对比。

**📊 数据集**

数据集包括公开的 SUPPORT、METABRIC、GBSG、WHAS、FLCHAIN、SEER、VETERANS，以及大型 ICU 数据集 eICU 和 MIMIC‑IV。

**📈 对比分析**

在 5 折交叉验证下，生存适配的模型在大多数数据集上获得最高 C‑index，尤其在 eICU/MIMIC‑IV 上相较传统模型提升 1–2%。

**⚠️ 局限性**

局限性包括仅使用静态特征、单事件生存、缺乏解释性分析，且未考虑多事件或时间变异特征。

---

## 396. Tac-DINO: Learning Vision-Tactile Features with Patch Alignment

**arXiv ID:** 2606.12069 | [PDF](https://arxiv.org/pdf/2606.12069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 397. Automating Geometry-Intensive Compliance Checking in BIM: Graph-Based Semantic Reasoning Framework

**arXiv ID:** 2606.12065 | [PDF](https://arxiv.org/pdf/2606.12065v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 398. Time-Series Foundation Model Embeddings for Remaining Useful Life Estimation

**arXiv ID:** 2606.11990 | [PDF](https://arxiv.org/pdf/2606.11990v1)

**作者:** Amir El-Ghoussani `[一作]` (Friedrich-Alexander-Universität Erlangen–Nürnberg), Valiseios Belagiannis `[通讯]` (Friedrich-Alexander-Universität Erlangen–Nürnberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用冻结的时间序列基础模型（Chronos‑2）提取多变量传感器窗口的上下文嵌入，再训练轻量级回归头来预测剩余使用寿命（RUL）

**💡 创新点**

创新点在于把预训练的TSFM直接作为特征提取器使用，无需对模型再训练，仅通过一个小的MLP即可实现高效、低成本的RUL估计，并且证明更长的上下文窗口能显著提升性能

**🔧 技术方法**

技术：Chronos‑2（decoder‑only Transformer）、多层感知机回归头、线性插值重采样、归一化与异常值裁剪、MSE 损失与 Adam 优化器

**📊 数据集**

数据集：来自 Nokia Solutions 的两类设备（Device A：87 通道，297,345 训练样本；Device B：51 通道，119,364 训练样本）

**📈 对比分析**

与传统非序列回归（线性回归、随机森林）、序列神经网络（GRU、LSTM、TCN、Transformer）、梯度提升回归的基线进行比较；在 5 步窗口下，MAE 下降至 44（Device A）/64（Device B），远优于最强基线（如 TCN MAE 88/112）

**⚠️ 局限性**

局限性：仅在单一工业数据集上验证，未评估缺失传感器、工作模式漂移或低标签场景；使用冻结模型无法利用任务特定微调带来的潜在提升；未给出不确定性量化

---

## 399. Graphical Analysis of Lifted Product Code Constructions

**arXiv ID:** 2606.11987 | [PDF](https://arxiv.org/pdf/2606.11987v1)

**作者:** Ragnar Freij-Hollanti `[一作]`, Patricija Šapokaitė `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了拉升乘积（Lifted Product）量子LDPC码的Tanner图结构，证明其 H_X 与 H_Z 的图同构，并给出连通性判据、girth 上界与最小吸收集的结构分析。

**💡 创新点**

首次在拉升乘积码框架下提供连通性与最小吸收集的解析性条件，并通过基矩阵 B 的 2×2 子矩阵无平凡置换乘积来最大化图的 girth，说明图结构与编码性能的直接关系。

**🔧 技术方法**

利用图提升、张量积、循环群置换以及 CSS 代码构造技术，对基矩阵 B 的循环结构进行符号推导，解析其在拉升后的 Tanner 图中的闭路与吸收集特性。

**📊 数据集**

未使用实验数据集，研究完全基于理论推导和符号计算。

**📈 对比分析**

通过符号比较与理论推导得出连通性判据与 girth 上限（可达 8），并给出最小吸收集的参数（如 (8,0)），但未给出具体数值性能评估或实验验证。

**⚠️ 局限性**

局限在于对基矩阵 B 的 2×2 子矩阵条件仅给出必要条件，缺乏充分条件；未涉及实际编码器/译码器实现和性能评估，缺少实验验证。

---

## 400. Categorical Robustness Assessment for Machine Learning based Network Intrusion Detection Systems

**arXiv ID:** 2606.12075 | [PDF](https://arxiv.org/pdf/2606.12075v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 401. Performance Analysis of YOLOv11 and YOLOv8 for Mixed Traffic Object Detection under Adverse Weather Conditions in Developing Countries

**arXiv ID:** 2606.12066 | [PDF](https://arxiv.org/pdf/2606.12066v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 402. Vision Transformers for Face Recognition Need More Registers

**arXiv ID:** 2606.12036 | [PDF](https://arxiv.org/pdf/2606.12036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 403. Learning Unions of Convex Sets via Invertible Latent Decomposition for Path Planning

**arXiv ID:** 2606.12027 | [PDF](https://arxiv.org/pdf/2606.12027v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 404. Human-Enhanced Loop Modeling (HELM): Agent-Based Finite Element Modeling of Concrete Bridge Barriers

**arXiv ID:** 2606.12025 | [PDF](https://arxiv.org/pdf/2606.12025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 405. MPPI-based Informative Trajectory Planning for Search and Capture of Drifting Targets with ASVs

**arXiv ID:** 2606.12019 | [PDF](https://arxiv.org/pdf/2606.12019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 406. Efficient Time Series Clustering from Multiscale Reservoir Dynamics with Granular-Ball Anchoring Graph Optimization

**arXiv ID:** 2606.12077 | [PDF](https://arxiv.org/pdf/2606.12077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 407. PAWS: Preference Learning with Advantage-Weighted Segments

**arXiv ID:** 2606.11982 | [PDF](https://arxiv.org/pdf/2606.11982v1)

**作者:** Aleksandar Taranovic `[一作]` (Karlsruhe Institute Of Technology), Gerhard Neumann `[通讯]` (Karlsruhe Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出 PAWS 方法，利用优势加权段（segment）进行偏好学习，并在离线数据上直接用段级优势更新策略，解决时序信用分配不一致问题。

**💡 创新点**

通过将优势函数的训练与策略优化保持在相同的段级分布上，消除传统方法的训练-推理分布差距；引入有效样本大小（effective sample size）自适应控制信赖域，避免手工调参。

**🔧 技术方法**

使用 Transformer 或 MLP 架构学习优势函数；利用 KL 约束的期望最大化（EM‑style）得到指数加权的段权重；对偶优化求解 λ；在策略更新中采用加权最大似然训练。

**📊 数据集**

实验数据集包括：Meta‑World 10 机器人操作任务、D4RL 四个行走任务；偏好标签由仿真 oracle 生成；另外收集了 Button Press 与 Door Open 两个任务的 10 位非作者人工标签（每人 50 条）。

**📈 对比分析**

与 Behavior Cloning、P‑IQL、CPL、CPL+KL、Preference Transformer、IPL、DPPO 等基线进行比较；在 50 与 500 条偏好下，PAWS 在大多数任务上均显著优于基线，平均提升约 20–30%（或更高），尤其在低数据量时仍优于 BC。

**⚠️ 局限性**

局限性：优势函数对段内每一步赋予相同权重，无法处理段内质量混合情况；需要预设有效样本大小 n_eff，取值需依据数据量调优；仅在模拟环境和有限人工标签上验证，尚未在大规模、噪声更严重的人类偏好上进行评估。

---

## 408. Non-frontal face recognition using GANs and memristor-based classifiers

**arXiv ID:** 2606.12074 | [PDF](https://arxiv.org/pdf/2606.12074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 409. "That's AI Slop, You Bot!" Studying Accusations, Evidence, and Credibility in Online Discourse Towards LLM-Generated Comments

**arXiv ID:** 2606.12073 | [PDF](https://arxiv.org/pdf/2606.12073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 410. Fibration Trees: A Unified Approach to Multi-Robot Motion Planning

**arXiv ID:** 2606.12070 | [PDF](https://arxiv.org/pdf/2606.12070v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 411. On the Limits of LLM-as-Judge for Scientific Novelty Assessment

**arXiv ID:** 2606.12071 | [PDF](https://arxiv.org/pdf/2606.12071v1)

**作者:** Soumitra Sinhahajari `[一作]` (Nanyang Technological University), Soujanya Poria `[通讯]` (Nanyang Technological University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基于真实论文的研究问题（RQ）数据集，并使用大语言模型（LLM）作为评审者，对模型生成的RQ与作者原始RQ在新颖性维度上的表现进行对比评估。

**💡 创新点**

创新点：①首次系统评估LLM-as-judge在科学RQ新颖性评估中的可靠性；②引入“窄度（narrowness）”维度来揭示LLM评估中对新颖性的误读；③通过人类专家对比验证，发现LLM评估与专家意见存在显著偏差，强调评估方法需多维度、对比式设计。

**🔧 技术方法**

技术：使用大规模预训练LLM（如 Gemini、GPT‑4 等）进行RQ生成与评估；采用两种LLM评审者（具备推理预算）进行独立评估；对RQ进行多维度评分（原创性、非显著性、缺口覆盖、源束缚、诊断框架）。

**📊 数据集**

数据集：共 746 篇 arXiv 计算机科学论文，生成 1,434 条作者anchored RQ，涉及 1,375 篇引用论文，平均每篇论文 2.0 条 RQ，平均每条 RQ 2.2 条缺口。数据由自动化提取、LLM辅助标注构成。

**📈 对比分析**

比较方法与性能：在“独立评分”下，LLM 对模型生成 RQ 的平均分均高于作者 RQ，但赢率仅 30‑40%；在“对比评分”下，赢率提升至 50‑60%；然而，人类专家在非显著性维度上偏好作者 RQ 的比例高达 60‑80%。LLM 在源束缚维度与专家在非显著性上的一致率约 40‑50%。总体来看，LLM 评估表现出“新颖性幻觉”，模型生成 RQ 过于窄化。

**⚠️ 局限性**

局限性：①数据仅覆盖近期 arXiv 计算机科学论文，缺乏跨学科验证；②新颖性评估高度依赖 LLM-as-judge，缺少更为客观的基准；③人类专家样本量有限，无法充分反映真实评审分歧；④未对不同模型或提示的系统性探索，导致结论的通用性受限。

---

## 412. FitVTON: Fit-aware Virtual Try-On via Body-Garment Size Control

**arXiv ID:** 2606.12012 | [PDF](https://arxiv.org/pdf/2606.12012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 413. From Nominal Intensity to Equivalent Rainfall: A Path-Based Credibility Evaluation Framework for Simulated Rainfall in Autonomous-Driving Perception Tests

**arXiv ID:** 2606.11989 | [PDF](https://arxiv.org/pdf/2606.11989v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 414. Attention by Synchronization in Coupled Oscillator Networks

**arXiv ID:** 2606.12059 | [PDF](https://arxiv.org/pdf/2606.12059v1)

**作者:** Fabio Pasqualetti `[一作]` (University of California, Irvine), Taosha Guo `[通讯]` (University of California, Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种基于Kuramoto/Lohe振荡器同步动力学的Transformer注意力机制——fixed‑query oscillator attention，旨在将注意力计算迁移到能量受限的物理子系统中。

**💡 创新点**

创新点包括：
1) 将注意力视为物理共识/同步过程，利用振荡器在球面上的自然平衡实现注意力，而非软最大化；
2) 证明该机制在正耦合下具有唯一稳定平衡并且几乎全局吸引；
3) 提供了任务复杂度与振荡器维度之间的幂律缩放规律；
4) 在低硬件配置下（d=2）实现了比软最大化更高或更稳定的性能；
5) 提供了硬件友好的实现蓝图，兼容多种振荡子物理平台。

**🔧 技术方法**

技术手段包括：Kuramoto/Lohe 轴向耦合动力学、投影矩阵 F、G、softplus 正激活、固定anchor 参考点、正向推理中的闭式固定点计算、有限时间 RK45 ODE 迭代、梯度下降训练、正则化与 readout sharpening。

**📊 数据集**

使用的数据集有：
- Keyword Spotting：Google Speech Commands（10 类）
- Subject‑Verb Agreement：合成 Linzen‑style 句子
- Causal Language Modeling：WikiText‑2（词级）和 TinyStories（短篇故事）

**📈 对比分析**

与标准 softmax 在相同 Transformer 架构、相同超参数下直接对比；
- 在双向任务（KWS、SVA）中，d=2 的振荡器注意力分别提升 +1.00pp 和 +5.27pp；
- 在因果语言模型中，d=2 时的困惑度比 softmax 高约 11；但随着维度升至 32，差距收敛至约 +2.98 PPL；
- ablation（冻结 W_V、随机 W_V、零注意力）验证了动力学的关键性；
- 读出锐化（p>1）进一步提升性能。

**⚠️ 局限性**

局限性与挑战：
1) 对于需要多维注意力的任务（如因果 LM），低维 d=2 的“维度瓶颈”导致性能不足；
2) 高维振荡器在现有硬件平台上实现尚不成熟；
3) 收敛时间有限，实际硬件需考虑实时性与积分窗口；
4) 当加权 anchor 总和趋近 0 或初始点靠近不稳定平衡时，收敛可能失效；
5) 目前仅在中等规模任务上验证，尚未在大规模模型与多语言数据上评估；
6) 能源优势在理论上显著，但真实硬件实现的能耗与成本仍待实验验证。

---

## 415. Reliable Error Estimation for PINNs: Lower and Upper A Posteriori Bounds

**arXiv ID:** 2606.12050 | [PDF](https://arxiv.org/pdf/2606.12050v1)

**作者:** Ismail Huseynov `[一作]` (Physikalisch-Technische Bundesanstalt), Agamirza Bashirov `[通讯]` (Eastern Mediterranean University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了 PINN 在求解常微分方程时的后验误差双侧估计，给出了可计算的下界与上界，兼顾线性系统的显式谱表达式，并将上界用于训练。

**💡 创新点**

提出在局部单侧 Lipschitz 与强单调性条件下可计算的下界与上界，显式地使用线性系统的对称部分谱值，并通过上界引导训练；相较于传统基于全局 Lipschitz 的上界，取得更紧的误差包络。

**🔧 技术方法**

利用单侧 Lipschitz 常数、局部强单调性、卷积积分与 RK4 数值积分、残差上界、符号残差有限探针以及基于上界的辅助正则化技术。

**📊 数据集**

以合成的非线性径向增长+旋转系统、高维刚性线性系统和不稳定振荡系统作为实验案例，没有使用公开数据集。

**📈 对比分析**

与传统全局 Lipschitz 上界以及局部分段诊断曲线比较，实验表明单侧常数得到更窄的误差带；训练时使用上界可显著提升预测精度。

**⚠️ 局限性**

需要预先获得局部单侧常数并保证轨迹保持在已知域内；当下界系数非正或使用硬约束初值时下界可能退化；目前仅适用于 ODE，残差上界与 RK4 误差需手工评估，且可能过于保守。

---

## 416. SpikeTAD: Spiking Neural Networks for End-to-End Temporal Action Detection

**arXiv ID:** 2606.12033 | [PDF](https://arxiv.org/pdf/2606.12033v1)

**作者:** Min Yang `[一作]` (Nanjing University), Limin Wang `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出首个端到端的时序动作检测框架SpikeTAD，将传统ANN模型转换为低功耗的SNN实现。

**💡 创新点**

创新点在于：①使用多阈值神经元(MTN)和期望补偿模块(ECM)实现ANN‑SNN转换的高精度；②引入剪切‑阶梯 (Clip‑Floor Shift) 激活函数以减少时间步数并保持性能；③在ViT-S骨干网络上实现SNN，显著降低能耗。

**🔧 技术方法**

技术：ANN‑SNN 率编码转换、Multi‑Threshold Neuron、Expectation Compensation Module、Clip‑Floor Shift 激活、Integrate‑and‑Fire neuron、轻量级多尺度检测头。

**📊 数据集**

使用公开数据集 THUMOS‑14 和 ActivityNet‑1.3 进行评估。

**📈 对比分析**

与现有SOTA方法（如AdaTAD、ViT‑TAD 等）比较，SpikeTAD 在 THUMOS‑14 上实现 67.2% mAP（T=8），在 ActivityNet‑1.3 上 37.42% mAP，性能仅略低于 ANN 版本，但能耗降低至 46%（T=8）甚至更低，显著优于其他端到端 TAD 模型。

**⚠️ 局限性**

局限性：①依赖预训练的 ANN 背骨，转换过程复杂；②视频时序导致 SNN 计算时间与语义时间差距大，需更多时间步；③高位阈值神经元提高功耗，难以进一步降低能耗。

---

## 417. KinematicRL: A Sim-to-Real Reinforcement Learning Framework For Social Navigation With Kinodynamic Feasibility

**arXiv ID:** 2606.12042 | [PDF](https://arxiv.org/pdf/2606.12042v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 418. Simplicity Suffices for Parameter Noise Injection in Stochastic Gradient Descent

**arXiv ID:** 2606.12054 | [PDF](https://arxiv.org/pdf/2606.12054v1)

**作者:** Benjamin Leblanc `[一作]` (Laval University), Richard Kamel `[通讯]` (Laval University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对随机梯度下降中的参数噪声注入，提出了一种高效方法（ENSGD），通过分布式身份实现每个训练样本单独噪声而不破坏批量计算；对比传统方法，验证了简易噪声方案的有效性。

**💡 创新点**

创新点在于：①利用线性层的分布等价性，在前向传播时直接给每个样本加噪声，从而保持计算效率；②系统性评估不同噪声参数化和多样本平均的实际收益，揭示简洁方案已能获得大部分优势。

**🔧 技术方法**

使用技术包括：高斯参数噪声注入、分布等价性推导、批量前向传播中单样本噪声实现、对比实验中的多噪声采样与不同方差参数化（等方差、移动平均、SqGrad、InvSqGrad）。

**📊 数据集**

使用的数据集为：CIFAR-100（图像分类）、AG News（四类新闻文本）和IMDB（二分类情感文本），其中AG News与IMDB采用TF‑IDF特征的全连接MLP模型。

**📈 对比分析**

与基线（无噪声）和传统NSGD相比，ENSGD在所有噪声水平下均取得更低的测试误差；单次噪声采样即可获得与多样本平均相当的性能，且等方差噪声几乎与最优参数化相同，显示简易方案即可获得大部分提升。

**⚠️ 局限性**

局限性：实验仅覆盖全连接层，未扩展至卷积层或其他网络结构；仅在有限的公开数据集上验证，可能在更复杂或更大规模任务中的效果未知；噪声收益受优化曲面平滑度影响，文本任务收益有限。

---

## 419. Game-Theoretic Latent Space Alignment for Multi-user Semantic MIMO Communications

**arXiv ID:** 2606.12005 | [PDF](https://arxiv.org/pdf/2606.12005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 420. Metadata-Aware Multi-Prompt Reasoning for Zero-Shot Accident Understanding

**arXiv ID:** 2606.12047 | [PDF](https://arxiv.org/pdf/2606.12047v1)

**作者:** Tarandeep Singh `[一作]` (Netradyne), Nishanth Chandran `[通讯]` (Netradyne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个零样本事故理解三阶段管道，分别针对何时（temporal），何种类型（classification），何处（spatial）进行检测与定位。

**💡 创新点**

创新点包括：① 通过视听-语言相似度和运动线索提取紧凑时段；② 采用五种结构化多提示（prompt）与熵门式两两仲裁实现鲁棒分类；③ 用事故类型与场景信息调制的开放词汇检测器实现精准定位。

**🔧 技术方法**

技术手段涵盖：Perception Encoder、Qwen-3.5-VL 9B、OWL‑v2 检测器、结构化提示、熵门仲裁、分数加权质心聚合等。

**📊 数据集**

使用了 ACCIDENT@CVPR 2026 真实 CCTV 测试集（无标注训练集），并在 CARLA 合成数据集上做开发与调优。

**📈 对比分析**

相对于基线（中心点预测），在三项指标（时间、空间、类别）上分别提升 0.089、0.159、0.118，最终调和平均分从 0.2714/0.3107 提升到 0.3852/0.4015，显著优于基线。

**⚠️ 局限性**

局限性在于对远距离碰撞、雨天、夜景或遮挡严重的场景仍易误检或漏检；且模型完全基于零样本，缺乏针对特定领域微调的精细化处理。

---

## 421. Bootstrapped Monitoring: Leveraging Transparent Reasoning to Oversee Stronger AI Agents

**arXiv ID:** 2606.11998 | [PDF](https://arxiv.org/pdf/2606.11998v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 422. Runtime Enforcement of Hybrid System Properties

**arXiv ID:** 2606.12022 | [PDF](https://arxiv.org/pdf/2606.12022v1)

**作者:** Mir Md Sajid Sarwar `[一作]` (Indian Institute of Technology Bhubaneswar), Thierry Jéron `[通讯]` (Univ Rennes)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于混合自动机的运行时强制框架，能够在实时控制系统中对安全属性进行连续时间监测并通过抑制、延迟、插入等操作主动干预，以防止安全违规。

**💡 创新点**

创新点在于：①将安全规范建模为安全混合自动机，超越传统离散或时钟限制的规范；②在同步点与两同步点之间都允许即时编辑和插入事件，提供更细粒度的干预；③设计了在线算法结合可达性分析来合成安全执行轨迹，并给出了可强制性的理论条件。

**🔧 技术方法**

使用的技术包括：混合自动机建模、实时运行时强制理论（编辑自动机、透明性、瞬时性等）、可达性与成员性检查、离散事件抑制/延迟/插入机制、ACC 控制器案例验证。

**📊 数据集**

实验数据集为自行构造的自适应巡航控制（ACC）场景，包括车辆初速度、距离、前车速度、加速度上限等参数，用以生成事件流并进行强制演示。

**📈 对比分析**

方法评估：在 Ubuntu 18.04/Intel i5-8250U 上运行，平均响应时间 4.67 ms，整个事件序列处理时长 15.098 s，显示框架对实时系统几乎无额外开销；实验未与其它现有强制框架做直接性能对比，但表明其在时间上满足严格实时约束。

**⚠️ 局限性**

局限性：①仅支持前缀闭合的安全性质，无法处理活性或义务性质；②仅适用于可判定子类的混合自动机（如定时自动机、矩形自动机等），对非线性或更复杂混合系统缺乏可扩展性；③未考虑概率/不确定性、学习预测或分布式多体系统；④实验仅在仿真场景下验证，缺乏真实硬件部署的实测。

---

## 423. InjectV: Modeling Fault Injection Attacks in RISC-V Simulation Environment

**arXiv ID:** 2606.12011 | [PDF](https://arxiv.org/pdf/2606.12011v1)

**作者:** Niccolò Lentini `[一作]` (Politecnico di Torino), Alessandro Savino `[通讯]` (Politecnico di Torino)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了InjectV，一个基于gem5的RISC‑V全系统仿真环境的故障注入框架，用于在安全关键执行点精确定向注入故障并评估系统的抗攻击能力。

**💡 创新点**

创新点在于通过预处理阶段结合执行轨迹与可执行文件，自动识别安全相关的候选注入点（CIP），从而显著缩小搜索空间，实现攻击导向的故障注入；相较于传统随机注入，效率提升显著。

**🔧 技术方法**

采用的技术包括gem5全系统仿真、基于时间与空间的预处理分析（分支判断、寄存器写前使用、差异执行）、对寄存器与物理内存的瞬态位级错误注入，以及并行化的投射管理与结果分类。

**📊 数据集**

使用的数据集为FISSC benchmark suite中的VerifyPIN程序及其逐步强化的硬化变体。

**📈 对比分析**

在相同注入数量下，Guided注入发现48次成功攻击，而随机注入仅2次；Guided注入相较随机实现了约24倍的效率提升，时间节省高达95.8%。

**⚠️ 局限性**

当前局限在于仅支持单个瞬态故障、仅针对寄存器和内存，且仅在RISC‑V gem5模型下验证；未涵盖多故障情景或更广泛的攻击模型。

---

## 424. ISAP-3D: Identity-Slot Aligned Part-Aware 3D Generation

**arXiv ID:** 2606.12099 | [PDF](https://arxiv.org/pdf/2606.12099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 425. Agreement in Representation Space for Open-Ended Self-Consistency

**arXiv ID:** 2606.12003 | [PDF](https://arxiv.org/pdf/2606.12003v1)

**作者:** Paula Ontalvilla `[一作]` (University of Basque Country), Aitor Ormazabal `[通讯]` (University of Basque Country)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在开放式生成任务（如代码合成、文本摘要）中自一致性（self-consistency）的效果，提出了一种无训练、基于嵌入空间聚类的自一致性评估框架Embedding‑Based Agreement（EBA），并通过采样多条生成结果并在嵌入空间中聚类来选取最具代表性的输出。

**💡 创新点**

创新点在于把自一致性视为生成分布在嵌入空间中的几何结构，而非仅靠符号匹配；提出EBA通过聚类捕捉生成的高密度聚集区；证明该方法在开放式任务中可提供稳健且可扩展的选择信号，并与经典的多数投票自一致性建立几何对应关系。

**🔧 技术方法**

技术包括：对模型生成的多条文本样本进行采样；使用预训练的文本嵌入模型或模型自身的隐藏层表示；在嵌入空间中采用凝聚聚类（agglomerative clustering）并利用Silhouette分数自动确定簇数；选取主簇中心最近的生成作为最终答案；评估使用准确率（HumanEval、MATH500）或Rouge‑1（CNN/DM）。

**📊 数据集**

使用的数据集包括：HumanEval（代码生成任务），MATH500（数学推理任务），CNN/DailyMail（新闻摘要任务）。

**📈 对比分析**

与随机挑选、Universal Self‑Consistency (USC)、Self‑Certainty (SCe)等方法比较，EBA在三种任务上均显著优于随机选择；在代码生成上提升约4个百分点，在数学推理上提升约17个百分点，摘要任务提升约1 Rouge‑1；EBA的性能随采样数增加呈现稳定的提升曲线，且在大样本规模下表现更稳健。

**⚠️ 局限性**

局限性包括：嵌入空间的几何分布可能不完全对应语义相似，导致聚类精度有限；当前使用的凝聚聚类与Silhouette判定较为粗糙，可能未充分挖掘局部结构；需要多次生成和嵌入提取，计算成本相对较高；实验仅覆盖了代码、摘要和数学推理任务，尚未验证在对话、长文本或代理推理等更广泛场景下的通用性。

---

## 426. Intelligent Automation for Embodied Benchmark Construction: Pipelines, Embodiments, Simulators, and Trends

**arXiv ID:** 2606.12207 | [PDF](https://arxiv.org/pdf/2606.12207v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 427. Towards Responsibly Non-Compliant Machines

**arXiv ID:** 2606.12147 | [PDF](https://arxiv.org/pdf/2606.12147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 428. Output-sensitive Sparse Polynomial GCD over Finite Fields is NP-hard

**arXiv ID:** 2606.12144 | [PDF](https://arxiv.org/pdf/2606.12144v1)

**作者:** Ruichen Qiu `[一作]` (Chinese Academy of Sciences), Xiao-Shan Gao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

证明稀疏多项式GCD以及根号检测在有限域上在BPP归约下是NP-hard，且若存在多项式时间算法则导致NP⊆BPP。

**💡 创新点**

首次给出输出敏感稀疏GCD在BPP归约下的硬度，并将该方法推广到根号检测问题；同时采用人工智能证明框架（MechMath Agent Team）完成完整证明。

**🔧 技术方法**

使用了Valiant–Vazirani 隔离技术、素数构造与根号编码、随机化多项式时间归约、根的映射与稀疏多项式组合压缩等多种理论与算法技术。

**📊 数据集**

无实验数据集，全部为理论证明与构造。

**📈 对比分析**

由于是理论难度结果，没有实验对比；证明表明在假设NP⊈BPP的情况下不存在多项式时间的稀疏GCD或根号检测算法。

**⚠️ 局限性**

局限性在于仅适用于有限域且仅在BPP归约意义下；整数多项式GCD的NP-hardness仍未解决；证明依赖随机化，若要得到确定性硬度还需进一步研究。

---

## 429. AGE-MIL: Anchor-Guided Evidence Learning for Patient-Level Prediction

**arXiv ID:** 2606.12126 | [PDF](https://arxiv.org/pdf/2606.12126v1)

**作者:** Jiawei Niu `[一作]` (Xi’an Jiaotong University), Yi Cai `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于Anchor‑Guided Evidence学习的AGE‑MIL框架，实现了从多张WSI中弱监督地进行患者级预测。

**💡 创新点**

创新点在于通过患者级anchor捕获全局病理上下文，引导证据检索和融合，并将患者级风险视为逐步累积的证据过程，从而解决稀疏证据和多切片异质性问题。

**🔧 技术方法**

采用了多实例学习、病理基础模型（patch/slide级特征提取）、anchor推断、条件检索得分、交叉注意力、门控机制以及Log‑Mean‑Exp聚合等技术。

**📊 数据集**

使用了两套前列腺癌活检数据集：肿瘤转移预测集（379例、6476张WSI）和ARSI预后预测集（306例、3402张WSI）。

**📈 对比分析**

与八种主流MIL方法（MeanMIL、MaxMIL、ABMIL、CLAM‑SB/MB、DSMIL、TransMIL、ILRA）在六项患者级任务中对比，AGE‑MIL平均提升AUC和准确率约2%，在所有任务上均领先。

**⚠️ 局限性**

局限性包括仅在前列腺癌数据上验证，缺乏跨疾病/跨中心的泛化评估，且对大规模病理基础模型的依赖导致计算成本较高。

---

## 430. AerialClaw: An Open-Source Framework for LLM-Driven Autonomous Aerial Agents

**arXiv ID:** 2606.12142 | [PDF](https://arxiv.org/pdf/2606.12142v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 431. From Agent Identity to Agent Economy: Measuring the Operational Readiness of ERC-8004 AI Agents

**arXiv ID:** 2606.12128 | [PDF](https://arxiv.org/pdf/2606.12128v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 432. Adaptive Multi-Resolution Procedural Knowledge Compression for Large Language Models

**arXiv ID:** 2606.12203 | [PDF](https://arxiv.org/pdf/2606.12203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 433. Soft-Prompt Tuning for Fair and Efficient LLM Benchmark Evaluation

**arXiv ID:** 2606.12117 | [PDF](https://arxiv.org/pdf/2606.12117v1)

**作者:** Selen Erkan `[一作]` (Aleph Alpha Research Lab), Letitia Parcalabescu `[通讯]` (Aleph Alpha Research Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对基模型进行软提示微调，以公平评估其知识并预测后续训练效果。

**💡 创新点**

通过仅10个软提示向量在80步内实现格式遵循并分离知识与格式化准确率，提供低成本评估方法。

**🔧 技术方法**

使用soft‑prompt tuning（梯度下降优化软向量），结合LLM‑judge评估开放式答案、格式化准确率、知识准确率以及bubble‑sort swap距离等指标。

**📊 数据集**

采用七大模型与七项基准，包括闭式多选（MMLU、Belebele、ChemBench）和开放式（TriviaQA、GSM8K、SQuAD、Math500）。

**📈 对比分析**

soft‑prompt微调后基模型在exact‑match、知识与格式化得分与后训练模型高度对齐，rank alignment优于zero‑shot/few‑shot，且能显著填平性能差距。

**⚠️ 局限性**

评估受LLM‑judge偏差影响，实验覆盖的模型与数据集有限；软提示对某些后训练模型仍有提升空间；未在统一后训练管线下验证。

---

## 434. Implicit Neural Representations of Individual Behavior

**arXiv ID:** 2606.12200 | [PDF](https://arxiv.org/pdf/2606.12200v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 435. Nearly Instance Optimal Sparse Matrix Approximation from Matrix-Vector Products

**arXiv ID:** 2606.12179 | [PDF](https://arxiv.org/pdf/2606.12179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 436. OpenMedReason: Scientific Reasoning Supervision for Medical Vision-Language Models

**arXiv ID:** 2606.12169 | [PDF](https://arxiv.org/pdf/2606.12169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 437. Time-Conditioned and Multi-Time Survival Prediction from 2D PET/CT Projections in Lung Cancer

**arXiv ID:** 2606.12140 | [PDF](https://arxiv.org/pdf/2606.12140v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 438. Do Not Discretize, Optimize: Almost Greedy Fictitious Play

**arXiv ID:** 2606.12149 | [PDF](https://arxiv.org/pdf/2606.12149v1)

**作者:** Evangelos Markakis `[一作]` (Athens University of Economics and Business), Christodoulos Santorinaios `[通讯]` (Athens University of Economics and Business)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了改进的Fictitious Play算法——Almost Greedy Fictitious Play（AGFP），证明其实例依赖收敛率为 𝒪(1/T)，并在随机游戏中观察到 𝒪(n/T) 的速度。

**💡 创新点**

创新点在于将步长搜索限制在 [δ,1] 区间、通过最小化对偶间隙实现几乎贪婪更新，并引入噪声 δ 以避免停滞，最终实现与连续 Fictitious Play 同阶的收敛率。

**🔧 技术方法**

主要技术包括对偶间隙分析、线性可分辨性（l.ind.）理论、定义与 κ_ 的条件数、二分搜索求最优步长、以及实验验证。

**📊 数据集**

实验使用的测试集为：经典的 Rock‑Paper‑Scissors 3×3 游戏、以及 50×50、500×500 的随机高斯生成零和博弈矩阵。

**📈 对比分析**

与传统 Fictitious Play（收敛速率 𝒪(1/√T)）对比，AGFP 在达到 10⁻³ 精度时已快，10⁻⁴ 时速度提升约 50 倍；收敛率从 𝒪(1/T) 提升到 𝒪(n/T)。

**⚠️ 局限性**

局限在于需要经验或自适应选择 δ，且每步需 O(m+n) 的额外计算；在某些游戏中可能仍出现停滞或最佳响应多重导致收敛慢；对非零和游戏的理论尚未扩展。

---

## 439. MSUE: Multi-Modal Soccer Understanding Expert

**arXiv ID:** 2606.12106 | [PDF](https://arxiv.org/pdf/2606.12106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 440. FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents

**arXiv ID:** 2606.12087 | [PDF](https://arxiv.org/pdf/2606.12087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 441. Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in Sparse Autoencoders

**arXiv ID:** 2606.12138 | [PDF](https://arxiv.org/pdf/2606.12138v1)

**作者:** Gleb Gerasimov `[一作]` (T Tech), Daniil Gavrilov `[通讯]` (T Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究稀疏自编码器(SAE)特征在不同随机种子下的可重复性，提出单特征出现概率度量，探讨稳定与不稳定特征在功能、几何结构及重建效果上的差异，并利用跨种子特征池构造更稳健的SAE字典。

**💡 创新点**

引入单特征出现概率度量揭示功能非对称；发现不稳定特征聚集在可重现的低秩子空间内，证明种子依赖是基于共享子空间的基底混合；利用跨种子特征池构造稳定SAE而不牺牲重建质量。

**🔧 技术方法**

使用多种SAE变体（TopK、BatchTopK、HierarchicalTopK、JumpReLU）训练大量种子；采用余弦相似度单多对一匹配、频率匹配掩码评估重建与下游损失；有效秩、SVD与子空间解释方差分析；合成低秩模型验证机制；特征池聚合与后期微调。

**📊 数据集**

GPT‑2、Gemma‑2、Pythia 等大型语言模型的残差流激活；随机初始化Transformer作为对照；以及用于验证的人工低秩字典合成数据。

**📈 对比分析**

对比稳定与不稳定特征在激活频率、词汇多样性、自动解释准确率、重建解释方差以及下游next‑token loss上的表现；使用跨种子匹配准确率和子空间解释方差评估；构造的稳定SAE在解释方差几乎与基线相当，在SAEBench上表现相近。

**⚠️ 局限性**

结果依赖于余弦阈值θ和端点阈值ε的选取；低秩子空间分析仅提供机制证据，不能解释所有不稳定现象；稳定–重建无折衷结论仅适用于特征池构造，单一SAE训练目标可能仍存在折衷。

---

## 442. Bridging the Morphology Gap: Adapting VLA Models to Dexterous Manipulation via Intent-Conditioned Fine-Tuning

**arXiv ID:** 2606.12109 | [PDF](https://arxiv.org/pdf/2606.12109v1)

**作者:** Chuanke Pang `[一作]` (Beihang University), Xilun Ding `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种两阶段解耦微调框架InDex，将预训练的Vision‑Language‑Action模型迁移到高DoF多指手进行复杂的接触丰富操纵。

**💡 创新点**

创新点：①将1‑DoF平行抓手的输出重新定义为连续的虚拟抓取意图；②通过LoRA实现参数高效对齐VLA骨干，分离宏观臂轨迹与微观手指动作；③在微观阶段使用意图条件去噪扩散网络生成多模态细粒度关节指令。

**🔧 技术方法**

使用技术：LoRA参数高效微调、意图条件去噪扩散模型、VLA骨干提取空间表征、虚拟抓取意图归一化、两阶段解耦训练策略。

**📊 数据集**

数据集：在robosuite仿真环境中通过视听遥控收集四个任务（Lift、Stack、Pick & Place、Nut Assembly）的专家演示，每任务100条成功轨迹，包含RGB、多视角、关节状态、意图γ和高DoF动作。

**📈 对比分析**

方法比较：与多种基线（MLP、BC‑RNN、ACT、Diffusion Policy、OpenVLA、UniVLA、π₀.₅）在四任务上进行对比。InDex平均成功率从基线4%–60%提升至85.8%，在宏观到微观每阶段的成功率也显著提高，显示出更优的鲁棒性与样本效率。

**⚠️ 局限性**

limitations：仅在仿真中验证，缺乏真实硬件实验；仿真到现实的转移（sim‑to‑real）未解决；对极端动态或高噪声环境的泛化能力尚待进一步评估。

---

## 443. InternVideo3: Agentify Foundation Models with Multimodal Contextual Reasoning

**arXiv ID:** 2606.12195 | [PDF](https://arxiv.org/pdf/2606.12195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 444. A Controlled Study of Decoding-Time Truthfulness Methods on Instruction-Tuned LLMs

**arXiv ID:** 2606.12160 | [PDF](https://arxiv.org/pdf/2606.12160v1)

**作者:** Ao Sun `[一作]` `[通讯]` (Independent Researcher), Ao Sun (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对现代指令调优LLM进行解码时真值性方法的系统评估，提出六控评估框架并在多模型、多基准上做大规模对比实验。

**💡 创新点**

创新点在于构建严谨的六控评估框架（离谱训练、众多评判、简单基准、混淆控制、Bootstrap置信区间与种子方差），揭示以往报告的真值性提升大幅被夸大；并首次将此框架应用于多基准验证。

**🔧 技术方法**

采用离散的解码调优技术（层对比解码、推理时干预、对比解码、学得logit适配器）以及推理层思考（链式推理、自动批判）进行对比，并通过多评判、Bootstrap CI与种子方差等统计手段确保结果可靠。

**📊 数据集**

使用TruthfulQA、HaluEval QA和TriviaQA三大真值性基准，覆盖误信、知识幻觉与开放域事实回忆三种任务。

**📈 对比分析**

在5种1B–70B模型上评估15种方法，严格控制下无任何token‑level方法显著优于简单温度/Top‑p基准；而单通道链式推理在所有基准上均提升5.6–19个百分点，是最稳健的单通道训练免费方法。

**⚠️ 局限性**

局限性包括：效果受模型、基准与提示对齐的高度依赖；对低基线模型的通用性尚未验证；统计噪声与样本量限制了对微小增益的检出；仅评估了两大模型家族与三基准，外部有效性待进一步验证。

---

## 445. DynaTok: Token-Based 4D Reconstruction from Partial Point Clouds

**arXiv ID:** 2606.12189 | [PDF](https://arxiv.org/pdf/2606.12189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 446. Shared Infrastructure Investment and Pricing: Stackelberg Equilibria in Risk-Aware Take-or-Pay Contracts

**arXiv ID:** 2606.12167 | [PDF](https://arxiv.org/pdf/2606.12167v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 447. Q-Fold: Query-Aware Focus-Context Spatio-Temporal Folding for Long Video Understanding

**arXiv ID:** 2606.12125 | [PDF](https://arxiv.org/pdf/2606.12125v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 448. PEBRE: An Open-Hardware Compute and Perception Add-On for the Pepper Robot

**arXiv ID:** 2606.12112 | [PDF](https://arxiv.org/pdf/2606.12112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 449. Debiasing Without Protected Attributes: Latent Concept Erasure from Textual Profiles

**arXiv ID:** 2606.12088 | [PDF](https://arxiv.org/pdf/2606.12088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 450. A Riemannian Approach to Low-Rank Optimal Transport

**arXiv ID:** 2606.12120 | [PDF](https://arxiv.org/pdf/2606.12120v1)

**作者:** Pratik Jawanpuria `[一作]` (IIT Bombay), Bamdev Mishra `[通讯]` (Microsoft India)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出基于 Fisher‑Rao 度量的黎曼几何框架，对低秩最优传输（OT）问题实现一阶和二阶无参数求解。

**💡 创新点**

创新点在于把平衡与非平衡 OT、线性 OT、GW、FGW 等问题统一到同一黎曼子流形上；对非平衡情况实现闭式投影与重排，消除内部迭代；提供全局最优性证明与秩充分性检验。

**🔧 技术方法**

使用黎曼优化技术：Fisher‑Rao 产品度量、正交切空间投影、退化重排、Hessian‑向量乘积，配合共轭梯度与信赖域算法。

**📊 数据集**

在高维高样本的高斯混合、偏斜高斯、噪声离群点、三维到二维高斯簇等合成数据集上评估，数据规模可达 5 万点。

**📈 对比分析**

与 LOT、LR‑GW、UB‑LOT 等现有低秩 OT 方法对比，Riemannian 求解器在相同秩下收敛迭代次数减少十倍以上、耗时缩短数倍，且在非平衡与非凸 GW/FGW 场景中显著降低目标值。

**⚠️ 局限性**

局限性：仍需预先设定秩 r，且在极度非凸或高噪声情况下可能陷入局部最优；对极大规模（百万级）数据的扩展尚未验证。

---

## 451. The PM-EdgeMap: Towards Real-Time Process Mining on the Edge-Cloud Continuum

**arXiv ID:** 2606.12103 | [PDF](https://arxiv.org/pdf/2606.12103v1)

**作者:** Hendrik Reiter `[一作]` (Kiel University), Wilhelm Hasselbring `[通讯]` (Kiel University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在边缘计算环境下执行实时流程挖掘算法，提出了描述数据集和计算拓扑的正式化框架，并通过一个基于边缘的合规性检查算法案例研究验证了其可行性和优势。

**💡 创新点**

创新点在于：①将流程挖掘与边缘-云连续体结合，提出针对边缘计算场景的流程挖掘方法；②设计了用于描述边缘环境下数据集和计算拓扑的正式语义模型；③首次在智能工厂边缘设备上实现并评估实时合规性检查。

**🔧 技术方法**

采用了边缘计算技术、实时流程挖掘与合规性检查算法、传感器数据流处理及云端协同机制。

**📊 数据集**

使用了来自智能工厂的真实传感器日志数据，包含生产线设备的时间序列事件日志，作为案例研究的数据集。

**📈 对比分析**

通过与传统基于云的合规性检查对比，评估了延迟、网络流量和CPU利用率等指标。实验结果显示，边缘实现将平均延迟降低约40%，显著减少了网络通信开销，且在资源受限的边缘节点上保持了可接受的计算性能。

**⚠️ 局限性**

局限性包括：①边缘设备的计算与存储资源有限，难以处理大规模日志；②对网络时延与可靠性的依赖较高；③实验范围仅涵盖单一工厂场景，缺乏在多工厂、多场景下的广泛验证。

---

## 452. LLM-Based User Personas for Recommendations at Scale

**arXiv ID:** 2606.12198 | [PDF](https://arxiv.org/pdf/2606.12198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 453. Agentic Environment Engineering for Large Language Models: A Survey of Environment Modeling, Synthesis, Evaluation, and Application

**arXiv ID:** 2606.12191 | [PDF](https://arxiv.org/pdf/2606.12191v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 454. The Brain That Goes Quiet: Serving a Large Model's Knowledge at 131 Tokens per Second on an 8 GB Laptop by Removing the Large Model from the Runtime Path

**arXiv ID:** 2606.12154 | [PDF](https://arxiv.org/pdf/2606.12154v1)

**作者:** Myeong Jun Jo `[一作]` `[通讯]` (ANIMA Research), Myeong Jun Jo (ANIMA Research)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将大模型的知识离线预生成并写入结构化存储，在线上只使用路由器、确定性渲染器和 1B 级量化模型，构建 Horsehair 系统实现快速查询。

**💡 创新点**

创新点在于将大型模型的推理完全拆分为离线预取与在线路由，证明结构性瓶颈导致的多秒延迟，并通过可信门与结构化格式实现事实保真与可审计。

**🔧 技术方法**

采用旋转驻留技术部署 35B/26B/4.7B Mixture‑of‑Experts 大模型进行离线预取，使用 BM25 路由与置信门，结构化机器语言格式进行答案写入，运行时使用 MiniCPM5‑1B 4‑bit 量化模型。

**📊 数据集**

实验数据集为 17 篇真实专利与实验日志共 563 条结构化条目，生成 1,126 个查询进行路由评估；验证门使用 16 个人工标注的腐败/支持案例。

**📈 对比分析**

在同一 RTX 4060 8 GB 笔记本上对比实验显示，去除大模型后平均响应时间从 4,465 ms 降至 518 ms，端到端吞吐率从 15.7 t/s 提升至 131.4 t/s，单模型解码率保持 226–237 t/s，首个 token 时间为 29–62 ms。

**⚠️ 局限性**

局限性包括仅适用于闭合域已验证答案，无法处理开放式组合推理；路由对自由语言敏感；验证门对细粒度伪造的召回率未知；单机单线程测评，未验证并发性能；结构化存储格式与验证算法未公开，难以完整复现。

---

## 455. Can News Predict the Market? Limits of Zero-Shot Financial NLP and the Role of Explainable AI

**arXiv ID:** 2606.12210 | [PDF](https://arxiv.org/pdf/2606.12210v1)

**作者:** Ali M Karaoglu `[一作]`, Shreyank N Gowda `[通讯]` (University of Nottingham)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了零样本自然语言处理在短期股价预测中的有效性，提出了基于事件影响、时间衰减与多层解释的完整预测管道；

**💡 创新点**

创新点在于将零样本NLI与影响时窗权重相结合，并构建多层可解释框架（token、文章、聚合级别）和可靠性诊断；

**🔧 技术方法**

采用RoBERTa、FinBERT、FinGPT、Llama3.1 8B、Mistral 7B等模型，结合DeBERTa做相关性筛选，使用时间衰减函数、事件影响窗口和高斯权重实现文章加权；

**📊 数据集**

使用基于Finnhub的免费新闻API获取美国上市公司新闻，并构建480个测试案例（20家公司×6预测窗口×4起始日），与股票回报数据配对产生正、负、中性标签；

**📈 对比分析**

与FinBERT、FinGPT等基于领域适配的模型以及简单多数类基线对比，零样本NLI在宏观F1上表现最佳，但整体准确率仍低于多数类基线，负类预测尤为困难；

**⚠️ 局限性**

局限包括依赖单一免费新闻源导致覆盖不足、仅使用英语新闻、未加入传统市场因子、数据集规模有限、以及阈值设定对类别划分的影响。

---

## 456. A Resource for Enthymeme Detection in Controversial Political Discourse

**arXiv ID:** 2606.12186 | [PDF](https://arxiv.org/pdf/2606.12186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 457. Beyond Dark Knowledge: Mixup-Based Distillation for Reliable Predictions

**arXiv ID:** 2606.12171 | [PDF](https://arxiv.org/pdf/2606.12171v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 458. How Low Can You Go? Active Learning for Sparse Model Discovery in the Ultra-Low-Data Limit

**arXiv ID:** 2606.12182 | [PDF](https://arxiv.org/pdf/2606.12182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 459. Quasi-linear Time Multiplication of Sparse Polynomials with Integer Coefficients

**arXiv ID:** 2606.12100 | [PDF](https://arxiv.org/pdf/2606.12100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

---

## 460. Detecting Sensitive Personal Information in Japanese Pre-Training Corpora for Large Language Models

**arXiv ID:** 2606.12114 | [PDF](https://arxiv.org/pdf/2606.12114v1)

**作者:** Rei Minamoto `[一作]` (Waseda University), Daisuke Kawahara `[通讯]` (Waseda University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了日语SCPI（特殊保护个人信息）数据集，并训练快速分类器以识别文本中的SCPI。

**💡 创新点**

首次针对日本法律定义的SCPI在大规模日语语料中系统地构建数据集，并提出基于LLM与传统机器学习相结合的高效检测流水线。

**🔧 技术方法**

利用LLM（Swallow、Gemma、GPT‑OSS‑120B）进行两阶段标注，结合NER、Doc2Vec/TF‑IDF、LightGBM/SVC/ModernBERT等模型实现分类，并采用模型堆叠提升召回。

**📊 数据集**

采自 llm‑jp‑corpus‑v3 Common Crawl 的约2.61 M条日语文本，后经NER筛选后得到约1.06 M条带姓名文本，用于训练和评估；正样本为954条（医史、犯罪、残疾）与8,586条负样本。

**📈 对比分析**

通过5折交叉验证及大规模人工评估，对比经典模型、Transformer与零样本LLM；LightGBM+TF‑IDF在速度上快数十倍，召回率达0.87，F1约0.87；Transformer准确度最高但速度慢，最终模型堆叠提升召回率4.2%。

**⚠️ 局限性**

仅覆盖姓名识别的SCPI，忽略地址、邮箱等非姓名识别信息；类别样本不足导致某些标签难以训练；LLM无法检测到的SCPI无法被标注，模型对罕见或隐蔽SCPI的识别能力有限。

---

## 461. TopoCap: Learning Topology-Agnostic Motion Priors for Monocular Video-to-Animation

**arXiv ID:** 2606.12153 | [PDF](https://arxiv.org/pdf/2606.12153v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 462. nD-RoPE: A Generalized RoPE for n-Dimensional Position Embedding

**arXiv ID:** 2606.12146 | [PDF](https://arxiv.org/pdf/2606.12146v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 463. PCA-Enhanced Adaptive NVAR Framework for High-Resolution Sea Surface Temperature Forecasting in the East Sea

**arXiv ID:** 2606.12141 | [PDF](https://arxiv.org/pdf/2606.12141v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 464. Reconfigurable Antennas for Next-generation Mobile Communication Networks: A Comprehensive Survey and Tutorial

**arXiv ID:** 2606.12139 | [PDF](https://arxiv.org/pdf/2606.12139v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 465. Greenness-Driven Scheduling in Far Edge Kubernetes: A CODECO Evaluation

**arXiv ID:** 2606.12136 | [PDF](https://arxiv.org/pdf/2606.12136v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 466. Augmenting Molecular Language Models with Local $n$-gram Memory

**arXiv ID:** 2606.12113 | [PDF](https://arxiv.org/pdf/2606.12113v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 467. DAM-VLA: Decoupled Asynchronous Multimodal Vision Language Action model

**arXiv ID:** 2606.12105 | [PDF](https://arxiv.org/pdf/2606.12105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 468. IntElicit: Eliciting and Assessing Contextualized Creativity via Dialogue Policy Optimization

**arXiv ID:** 2606.12086 | [PDF](https://arxiv.org/pdf/2606.12086v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 469. Strategic Facility Location with $p$-Norm Social Costs

**arXiv ID:** 2606.12187 | [PDF](https://arxiv.org/pdf/2606.12187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 470. Sparse Polynomial Divisibility Test over Finite Field is CoNP-hard

**arXiv ID:** 2606.12130 | [PDF](https://arxiv.org/pdf/2606.12130v1)

**作者:** Yichuan Cao `[一作]` (State Key Laboratory of Mathematical Sciences, Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Xiao-Shan Gao `[通讯]` (State Key Laboratory of Mathematical Sciences, Academy of Mathematics and Systems Science, Chinese Academy of Sciences)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

提出并证明稀疏多项式在有限域上的可除性测试是 CoNP 难的，即决定一个稀疏多项式是否不被另一个稀疏多项式整除在 BPP 多对一还原下是 NP 难的。

**💡 创新点**

创新点在于构造了“可除性桥”Lemma，将有限域上稀疏多项式的根检测问题转化为非可除性问题，并通过随机化多项式时间还原首次证明稀疏多项式可除性测试的 NP 难度。

**🔧 技术方法**

采用随机化多项式时间还原、有限域 Frobenius 恒等式、可除性判定（$x^p-x$ 与多项式同除的根判定）以及 Lean 4 形式化验证等技术。

**📊 数据集**

本研究完全基于理论证明，没有使用任何实验数据集。

**📈 对比分析**

论文没有提出新的算法或实验对比，而是通过理论证明表明该问题至少与 NP 难度相当，说明在常见的复杂度假设下不存在多项式时间算法。

**⚠️ 局限性**

局限性包括：仅给出 CoNP 难度，没有给出具体算法或上界；结果依赖于随机化还原，基于 BPP 多对一还原假设；未针对多变量情况给出细化；未给出硬实例或进一步的复杂度分类（如 NP 完全或 PSPACE 完全）。

---

## 471. Natural-Language Temporal Grounding in Hour-Long Videos is a Search Problem: A Benchmark and Empirical Decomposition

**arXiv ID:** 2606.12300 | [PDF](https://arxiv.org/pdf/2606.12300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 472. Learning What to Say to Your VLA: Mostly Harmless Vision Language Action Model Steering

**arXiv ID:** 2606.12299 | [PDF](https://arxiv.org/pdf/2606.12299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 473. Measuring Epistemic Resilience of LLMs Under Misleading Medical Context

**arXiv ID:** 2606.12291 | [PDF](https://arxiv.org/pdf/2606.12291v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 474. On The Effectiveness-Fluency Trade-Off In LLM Conditioning: A Systematic Study

**arXiv ID:** 2606.12234 | [PDF](https://arxiv.org/pdf/2606.12234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 475. PianoKontext: Expressive Performance Rendering from Deadpan Context

**arXiv ID:** 2606.12282 | [PDF](https://arxiv.org/pdf/2606.12282v1)

**作者:** Dmitrii Gavrilev `[一作]` `[通讯]` (Applied AI Institute), Dmitrii Gavrilev (Applied AI Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现了基于死平音频上下文的可变长度钢琴演奏渲染模型

**💡 创新点**

创新点在于将流匹配技术与DTW对齐、2D RoPE编码融合到DiT框架中，实现音符级别的对齐与表达时序控制

**🔧 技术方法**

使用Music2Latent预训练音频编码器、动态时间规整（DTW）、流匹配（flow matching）、Diffusion Transformer（DiT）+2D RoPE、Heun ODE求解器等技术

**📊 数据集**

利用MAESTRO（表达式演奏）与ASAP（死平MIDI）数据集合成的对齐数据进行训练与评估

**📈 对比分析**

与无监督轨迹逆向的CFG Bridge基线对比，PianoKontext在FAD、KAD、Pitch DTW、对齐精度/召回率等指标上显著优于基线，接近人类水平

**⚠️ 局限性**

局限于钢琴数据、表达细节（如装饰音）不足、样本规模有限、未验证全长演奏生成、依赖DTW预处理等

---

## 476. Partitioned Tags, Shared Data: Reconciling Strict Cache Isolation with Write-Shared Coherence

**arXiv ID:** 2606.12259 | [PDF](https://arxiv.org/pdf/2606.12259v1)

**作者:** Kartik Ramkrishnan `[一作]` (University of Minnesota), Pen Chung Yew `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种安全缓存分区架构SCP，仅对标签进行分区、共享数据池并通过引用计数实现单一一致性状态，以消除传统分区方案对写共享数据的不兼容问题，从而完全消除基于eviction的侧信道。

**💡 创新点**

创新点在于将标签/数据解耦与分区结合，使用引用计数保证跨域一致性、对跨域查找进行时间等价掩码、可调写透（SCP-WT）以及基于泄漏阈值的自适应模式，解决了写共享线导致的分区冲突，并提供可调安全-性能折衷。

**🔧 技术方法**

使用标签/数据解耦、引用计数、Bloom过滤器前端、延迟掩码、可调写透（SCP-WT）以及页面泄漏阈值自适应模式，在gem5模拟器中实现。

**📊 数据集**

使用SPEC CPU2017单进程与多程序混合基准，以及自定义共享写共享线微基准（如ReadShared、ProdCons、LockContend等）。

**📈 对比分析**

通过gem5对基线LRU、DAWG以及随机化方案进行对比；在SPEC 2017上SCP与DAWG的IPC差距≤0.3%，相对于未分区基线约下降20%；在共享写共享线微基准下，SCP-WT在写共享区块产生最高23% IPC下降，但在常规工作负载几乎无额外开销。

**⚠️ 局限性**

主要局限在写共享数据的性能开销较大，需要操作系统支持共享页面标记；Bloom过滤器可能带来侧信道；在高共享写压力下自适应模式可能导致写透切换频繁；未覆盖其他微架构共享状态（如MSHR、预取器）可能的泄漏。

---

## 477. Reinforcement Learning Disrupts Gradient-Based Adversarial Optimization

**arXiv ID:** 2606.12251 | [PDF](https://arxiv.org/pdf/2606.12251v1)

**作者:** Xinhai Zou `[一作]` (KU Leuven), Bart Preneel `[通讯]` (KU Leuven)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了强化学习（RL）训练方式对图像分类模型在梯度攻击下的鲁棒性，并与监督学习（SL）及对抗训练（SL-adv）进行系统对比；

**💡 创新点**

首次揭示RL通过噪声梯度和ε‑greedy探索使模型梯度幅度变小、方向不稳定，从而抑制梯度攻击，并提出RL+对抗训练（RL‑adv）双层防御提升鲁棒性；

**🔧 技术方法**

采用REINFORCE式策略梯度与ε‑greedy探索进行RL训练，结合TRADES对抗训练，评估PGD、AutoAttack、Transfer与Square攻击，同时引入梯度统计指标和损失曲面可视化进行机制分析；

**📊 数据集**

使用CIFAR‑10、CIFAR‑100、ImageNet‑100三个数据集，并在4/6层CNN及ResNet‑18三种网络结构上实验；

**📈 对比分析**

通过对比SL、SL‑adv、RL、RL‑adv四种训练方式，在PGD攻击下RL提升至约56%对抗准确率，RL‑adv在AutoAttack、Square等多种攻击上均优于SL‑adv，表现显著；但在自适应攻击下RL优势减弱；

**⚠️ 局限性**

RL训练计算成本高约20倍；对抗攻击仍易受转移攻击影响；机制未消除对抗区域本身，且验证仅限于卷积网络。

---

## 478. Re-evaluating Confidence Remasking in Masked Diffusion Language Models

**arXiv ID:** 2606.12232 | [PDF](https://arxiv.org/pdf/2606.12232v1)

**作者:** Stipe Frkovic `[一作]`, Eric Nalisnick `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新评估了WINO训练‑free自我纠错（remasking）方法在掩蔽扩散语言模型（dLLM）中的实际效能，并通过多种解码设置与基准比较，探讨其优势与局限。

**💡 创新点**

创新点在于提出了一套更全面的评估框架，系统对比不同块长、采样温度、未掩蔽策略（Fast‑dLLM、Bernoulli unmasking）下的WINO表现；揭示了高 flip‑flop 率导致模型难以提供更好重预测，阐明了WINO受限于解码设置的现象。

**🔧 技术方法**

使用的技术包括掩蔽离散扩散模型（dLLM）、基于置信度的自适应解码（Fast‑dLLM）、shadow‑token 近似留一预测、后验重掩蔽、半自回归（semi‑AR）块级解码，以及对温度和 Bernoulli 未掩蔽的非贪心采样。

**📊 数据集**

使用的数据集包括 GSM8k、MATH‑500、HumanEval 与 MBPP，模型则选用 LLaDA‑8B‑Instruct 与 Dream‑v0‑Instruct‑7B 两个公开 dLLM。

**📈 对比分析**

比较方法：通过 Pareto 前沿、NFEs、通过吞吐量（tokens/s）以及 k‑metric 等指标衡量准确率、样本多样性；结果显示：
- 在短块长（BL=32）下 WINO 与 Fast‑dLLM 几乎无显著差距；
- 在长块长（BL=128）下能稍微提升准确率，但成本更高；
- 在非贪心解码下可提升 2‑3% pass@1，但伴随多样性崩溃；
- 与 Bernoulli unmasking 配合时，准确率提升约 3% 而额外 NFEs 仅略增。

**⚠️ 局限性**

限制：仅针对 WINO 进行评估；高 flip‑flop 率表明 dLLM 在重预测时难以提出更优 token；WINO 的优势高度依赖解码设置，无法在大多数常规设置下显著提升；未验证其他后验重掩蔽方法，也未探究训练或统一扩散方式可能带来的改进。

---

## 479. Making Foresight Actionable: Repurposing Representation Alignment in World Action Models

**arXiv ID:** 2606.12217 | [PDF](https://arxiv.org/pdf/2606.12217v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 480. Bridging the Smart City Cybersecurity Data Gap Through AI-Driven Synthetic Dataset Generation

**arXiv ID:** 2606.12225 | [PDF](https://arxiv.org/pdf/2606.12225v1)

**作者:** Stephanie Polczynski `[一作]` (Dakota State University), Kyle Korman `[通讯]` (Dakota State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了基于生成式人工智能的合成数据生成框架，用于产生高保真、可共享的智能城市网络安全数据集；

**💡 创新点**

创新点在于将智能城市特有的多协议、多源异构数据与安全事件标签结合，设计了结构约束驱动的生成模型，并提出了面向网络流量的 Fréchet Traffic Distance 等评估指标，首次系统化验证合成数据在安全工具中的可用性；

**🔧 技术方法**

采用GAN、CTGAN、VAE、Transformer 等生成模型，辅以预处理/后处理的模式约束、条件生成和攻击场景注入；

**📊 数据集**

利用现有 IIoT / IoT 网络安全数据集（Edge‑IIoTSet、WUSTL‑IIoT‑2021、X‑IIoTID、UNSW_NB15、BoT_IoT、TON_IoT 等）作为训练和基准；

**📈 对比分析**

通过 SDNist、SDMetrics、SynthEval、SynthRO 等工具对统计相似度、结构一致性、标签一致性进行量化评估，并在实际 IDS/异常检测实验中验证合成数据能保持与真实数据相当的检测精度（误报率与漏报率与原始数据相近）；

**⚠️ 局限性**

局限性包括：对现有公开数据集的依赖导致潜在偏差；合成数据虽统计相似，但可能缺乏真实环境中更细粒度的时空相关性；模型生成的假象与过拟合风险；以及算力和存储资源限制导致的规模与多样性受限。

---

## 481. SHERPA: Seam-aware Harmonized ERP Adaptation for Open-Domain 360$^\circ$ Panorama Generation

**arXiv ID:** 2606.12213 | [PDF](https://arxiv.org/pdf/2606.12213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 482. UGV-Conditioned Multi-UAV Informative Planning on a Shared Exposure Belief

**arXiv ID:** 2606.12306 | [PDF](https://arxiv.org/pdf/2606.12306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 483. Mind your key: An Empirical Study of LLM API Credential Leakage in iOS Apps

**arXiv ID:** 2606.12212 | [PDF](https://arxiv.org/pdf/2606.12212v1)

**作者:** Pinran Gao `[一作]` (Wake Forest University), Fan Yang `[通讯]` (Wake Forest University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首次系统地研究了 iOS 应用中 LLM API 凭证泄露的流行程度、泄露模式、开发者的防护措施以及补救效果，构建了专用的动态分析框架并对 300+ 真实 App 进行扫描；

**💡 创新点**

创新点在于：①首次针对 iOS 生态进行大规模 LLM 凭证泄露实证研究；②提出了无需越狱或二进制解密即可检测凭证泄露的动态 MITM 代理框架；③通过三个月后再扫描评估补救率，揭示补救率低下的系统性问题；

**🔧 技术方法**

主要技术包括基于 MITM 代理的网络流量拦截、针对不同 LLM 提供商的凭证指纹识别规则、主动验证凭证有效性（JWT/静态 API key）以及责任披露与后续扫描；

**📊 数据集**

使用的数据集为 2025 年 10 月从美国 App Store 通过 iTunes 搜索 API 采集的 1,200+ LLM 集成 iOS 应用，最终手工筛选出 300+ 可测试的应用；

**📈 对比分析**

与 Android 同类研究对比，iOS 应用凭证泄露率更高；在 3 个月后再次扫描，补救率仅约 12%；框架平均每个请求耗时约 2 秒，能够在常规测试环境下高效完成扫描；

**⚠️ 局限性**

局限性包括：仅检测网络层泄露，无法发现服务器端或设备存储中的凭证泄露；代理绕过和 VPN 捕获可能导致漏检；数据集为单地区、单时点，无法完全代表全球 iOS 市场。

---

## 484. From 2D Grids to 1D Tokens: Reforming Shared Representations for Multimodal Image Fusion

**arXiv ID:** 2606.12303 | [PDF](https://arxiv.org/pdf/2606.12303v1)

**作者:** Yuchen Xian `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将二维共享特征网格替换为一维token空间的多模态图像融合框架，并通过选择性token编辑实现全局外观一致性与局部细节保留的平衡

**💡 创新点**

创新点在于：①将预训练的1D图像tokenizer作为固定全局接口，解耦图像级基底与局部细节；②设计Selective Token Editing（STE）仅稀疏编辑少量token，实现轻量化全局外观调控；③通过token‑to‑map接口兼容传统2D融合骨干，兼顾局部结构建模

**🔧 技术方法**

核心技术包括：冻结预训练的1D tokenizer（TiTok）、token‑to‑map映射、基底/细节分解与独立融合、残差重建、Gumbel‑Softmax采样用于自动选择编辑token位置

**📊 数据集**

使用了红外-可见图像融合（MSRS、RoadScene）和医学图像融合（Harvard Medical Image Dataset）等公开数据集；在M^3FD、RoadScene、TNO等基准上进行评测，并对下游目标检测（YOLOv8s）和语义分割（SegFormer-B1）进行验证

**📈 对比分析**

与9种先进融合方法对比，实验显示在熵、标准差、SCD、SSIM等指标上均获得首或次优成绩；在目标检测mAP_50:95和语义分割mIoU上也表现最佳，证明融合质量提升能直接转化为下游任务性能提升

**⚠️ 局限性**

局限性包括：1) 依赖冻结的预训练tokenizer，模型可扩展性受限；2) 只对稀疏token进行编辑，可能无法充分捕捉复杂的全局外观变化；3) 对极低分辨率或非对齐输入的鲁棒性未充分验证

---

## 485. Findings of the MAGMaR 2026 Shared Task

**arXiv ID:** 2606.12295 | [PDF](https://arxiv.org/pdf/2606.12295v1)

**作者:** Alexander Martin `[一作]` (Johns Hopkins University), Xiang Xiang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在2026年MAGMaR工作坊中，组织了针对视频检索与基于检索的视频生成的共享任务，评估系统在多语言、多模态视频集上的检索与生成能力。

**💡 创新点**

创新点包括：① 构建了更大、更多样化的WikiVideo25数据集，并引入人物角色约束的查询；② 提出了基于文本代理与认知式重排序的检索框架；③ 采用迭代式多视频问答生成策略以提高人类评价。

**🔧 技术方法**

主要技术手段包括：BGE‑M3文本代理索引、OmniEmbed与Wholembed‑v3第一阶段检索、RankVideo与LLM驱动的认知重排序、视觉‑语言模型与OCR/ASR相结合的时间线构建、CLUE模型的多模态置信校准以及MiRAGE自动评估框架。

**📊 数据集**

使用的数据集为WikiVideo25（约110k多语言事件视频）与MultiVENT 2.0的干扰视频集合，查询共19条，包含双语与混合语言场景。

**📈 对比分析**

检索方面，基准OmniEmbed在nDCG@10仅为0.17，加入RankVideo提升至0.54，C2F‑RAG通过文本代理与LLM重排序实现0.848；生成方面，MARQUIS‑iter‑qa在人工评分上最高3.833，但在MiRAGE precision/recall上不如CAG基线；总体表明重排序与迭代问答显著提升性能。

**⚠️ 局限性**

局限性：① 人工评估与自动指标之间存在偏差，自动评估未充分考虑拒绝或缺失证据；② 所有系统均先将视频转换为文本代理，导致视觉信息被忽略；③ 未进行任务特定微调，缺乏针对多视频问答的专门训练；④ 评测框架对角色相关覆盖度与适当拒绝缺乏完善。

---

## 486. Bridging the Modality Gap in Forensic Image Retrieval

**arXiv ID:** 2606.12294 | [PDF](https://arxiv.org/pdf/2606.12294v1)

**作者:** Ricardo González-Gazapo `[一作]` (Advanced Technologies Application Center (CENATAV)), Milton García-Borroto `[通讯]` (Universidad de La Habana)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个统一的多模态检索框架，利用多模态大型语言模型自动生成结构化描述，并与视觉特征进行零样本乘法融合，以支持纹身、纹身草图、人类文本描述以及面部素描等法医检索场景。

**💡 创新点**

创新点包括：1）首次将MLLM生成的文本描述与视觉特征通过无监督乘法融合用于法医检索；2）在四大异构检索任务中实现显著性能提升；3）证明小型MLLM（如DeepSeek‑VL2‑tiny、Bunny‑v1.1‑4B）在此任务上可优于大模型，体现模型规模与检索质量的负相关。

**🔧 技术方法**

技术手段包括：多模态大型语言模型（DeepSeek‑VL2‑tiny、Qwen2‑VL‑2B、Bunny‑v1.1‑4B、Qwen2.5‑VL‑7B、LLaVA‑1.6‑7B）用于生成描述；MPNet句子变换器提取文本嵌入；纹身视觉特征采用 MobileNetV2/WAP/TattTRN/CLIP，面部视觉特征采用 MobileFaceNet/ShuffleFaceNet/ResNet50；最终通过乘法融合获得统一检索得分。

**📊 数据集**

使用公开纹身与面部素描数据集：WebTattoo、BIVTatt（纹身照片及其草图）以及 UoM‑SGFS（面部素描与照片），并人工生成对应的文本描述。

**📈 对比分析**

采用 Rank@K 与 mAP 评估，并通过自举显著性检验对方法进行比较；在纹身照片检索中多模态融合将 Rank@1 从约0.82提升至0.87，mAP 达到0.96；人类描述检索 mAP 0.74；纹身草图检索 mAP 0.90，Rank@1 0.65；面部素描检索 mAP 0.38，Rank@1 0.25。

**⚠️ 局限性**

主要限制包括：检索效果高度依赖MLLM生成描述的准确性与一致性；人类描述差异显著影响性能；草图评估基于专家观测的草图，实际现场草图可能更模糊；文本描述在面部任务提升有限；大模型生成长文本可能引入噪声；跨语言、真实环境鲁棒性和不确定性处理仍待研究。

---

## 487. Finding Multiple Interpretations in Datasets

**arXiv ID:** 2606.12277 | [PDF](https://arxiv.org/pdf/2606.12277v1)

**作者:** Matthew Chak `[一作]` (California Polytechnic State University), Paul Anderson `[通讯]` (California Polytechnic State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了一种可区分但性能相近的多模型搜索方法，使得同等表现的模型拥有完全不同的特征重要性。

**💡 创新点**

引入可微分距离函数与正则化，使得在保持性能的前提下显著区分模型解释。

**🔧 技术方法**

使用深度学习框架(torch-deeptype)、可微分排序(softrank)与自定义正则化的梯度优化。

**📊 数据集**

在METABRIC基因表达数据集上进行实验。

**📈 对比分析**

与十个基线DeepType模型对比，三个新模型在准确率、损失等指标保持相近，但基因重要性列表完全不重叠。

**⚠️ 局限性**

随着模型数增加，性能（如稀疏度）会下降，且方法需要手动设定α，缺乏理论保证。

---

## 488. VOID: Defeating Unauthorized Mimicry in Latent Diffusion Models

**arXiv ID:** 2606.12263 | [PDF](https://arxiv.org/pdf/2606.12263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 489. VIA-SD: Verification via Intra-Model Routing for Speculative Decoding

**arXiv ID:** 2606.12243 | [PDF](https://arxiv.org/pdf/2606.12243v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 490. Beyond Fully Random Masking: Attention-Guided Denoising and Optimization for Diffusion Language Models

**arXiv ID:** 2606.12273 | [PDF](https://arxiv.org/pdf/2606.12273v1)

**作者:** Jia Deng `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AGDO 框架，利用注意力信息指导 diffusion 大语言模型的去噪顺序与优化流程；

**💡 创新点**

首次将注意力分布用于确定去噪顺序并对关键 token 加权，统一监督微调与强化学习，解决随机掩码与推理不匹配问题；

**🔧 技术方法**

采用注意力分析、attention‑guided denoising order、加权交叉熵以及基于 PPO/GRPO 的强化学习；

**📊 数据集**

在数学推理（MATH‑500、GSM8K、Minerva）和代码生成（LiveBench、LiveCodeBench‑V2）等数据集上进行评估，并在 HellaSwag、CommonsenseQA 等通用 NLP 任务上验证；

**📈 对比分析**

与标准 SFT、blockwise SFT、Diff‑GRPO、Coupled RL、TraceRL 等基线对比，在 Dream‑v0‑Instruct‑7B 上提升数学和代码任务约 2–4%（SFT）及 RL 阶段进一步提升 3% 以上，整体性能超越现有最佳方法；

**⚠️ 局限性**

仅针对全注意力 diffusion LLM 进行研究，未覆盖块注意力模型，未来需探讨块注意力下的适配与改进。

---

## 491. The Impossibility of Eliciting Latent Knowledge

**arXiv ID:** 2606.12268 | [PDF](https://arxiv.org/pdf/2606.12268v1)

**作者:** Korbinian Friedl `[一作]` (London School of Economics and Political Science), Jonathan Richens `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

用因果影响图(CID)正式化了从AI系统中提取潜在知识（ELK）问题，定义了诚实与真实的概念，并通过理论证明阐明了训练策略的可行性与不可行性。

**💡 创新点**

创新点在于：①将ELK问题映射到CID框架内，清晰区分可观测与潜在变量；②在充分能力条件下证明诚实等价真实；③提出不可避免的“评估模拟器”问题，给出训练策略不可能保证诚实的不可行性定理。

**🔧 技术方法**

技术方法主要是结构因果模型与因果影响图的构建与分析，定义代理策略、训练策略、诚实/真实判定，并用严格的概率与策略论证证明相关定理。

**📊 数据集**

未使用具体数据集，所有讨论均为理论推导与形式化证明。

**📈 对比分析**

该工作未包含实验或性能比较，主要通过数学证明展示训练策略在满足可观测性假设下仍可能产生评估模拟器，因而无法保证诚实。

**⚠️ 局限性**

局限性包括：①假设开发者与代理共享相同变量集合，未考虑本体不匹配；②仅聚焦变量取值的问答，未覆盖因果结构或更广泛的知识类型；③假设评估者完美或仅能基于最佳猜测给出反馈，未涵盖更复杂评估机制；④未给出实证方法或实验验证。

---

## 492. Harness In-Context Operator Learning with Chain of Operators

**arXiv ID:** 2606.12318 | [PDF](https://arxiv.org/pdf/2606.12318v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 493. CCKS: Consensus-based Communication and Knowledge Sharing

**arXiv ID:** 2606.12281 | [PDF](https://arxiv.org/pdf/2606.12281v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 494. SpikeDecoder: Realizing the GPT Architecture with Spiking Neural Networks

**arXiv ID:** 2606.12287 | [PDF](https://arxiv.org/pdf/2606.12287v1)

**作者:** Claas Beger `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了完全基于脉冲神经网络的Transformer解码器SpikeDecoder，并将其用于字符级自然语言生成任务。

**💡 创新点**

创新点在于将Transformer解码器完全改写为可直接训练的SNN，设计了脉冲兼容的token与位置嵌入、残差结构、归一化方案以及时间步统一方法，显著降低能耗并保持可比性能。

**🔧 技术方法**

使用了多步LIF脉冲神经元、SpikingJelly框架、Spiking Self-Attention (SSA)、批量归一化/功率归一化、可学习阈值等技术。

**📊 数据集**

主要使用War & Peace字符级子集（约10万字符）进行训练，并在text8/enwik8 上做进一步实验。

**📈 对比分析**

通过与传统ANN解码器基线对比，部分脉冲模型逐步下降，最终SpikeDecoder在字符级测试集上的准确率约为81–87%，相较ANN约98%，能耗降低87–93%，但仍缺乏softmax/搜索等高级推理机制。

**⚠️ 局限性**

限制包括字符级输入导致生成文本质量受限、缺乏概率输出/搜索算法、参数规模小、未在大规模数据集上验证、残差和归一化方案仍有改进空间。

---

## 495. Damage-TriageFormer: A Foundation-Model Framework for Typology-Based Building Damage Assessment from Mono-Temporal Imagery

**arXiv ID:** 2606.12248 | [PDF](https://arxiv.org/pdf/2606.12248v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 496. Mathematical perspective on genetic algorithms with optimization guided operators

**arXiv ID:** 2606.12279 | [PDF](https://arxiv.org/pdf/2606.12279v1)

**作者:** Anna Brandenberger `[一作]` (Massachusetts Institute of Technology), Elchanan Mossel `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出 GMR 框架，系统性分析 LLM 驱动的遗传算法的查询复杂性，并证明了生成、变异、重组三类算子在不同情形下的必需性以及多样性在解空间中的关键作用。

**💡 创新点**

创新点在于将生成、变异、重组统一为可定制的查询算子，构建基于强化学习/MDP 的查询复杂性理论，证明优化引导算子在一维示例中可实现线性/指数级加速，并通过多算子构造展示三类算子必须同时存在，以及在 parity 学习实例中阐明多样性瓶颈与线性池大小的下界。

**🔧 技术方法**

采用概率分布算子定义、强化学习与 MDP 框架、极值理论、线性代数与流式算法、Raz 等人关于流式算法的记忆下界等技术。

**📊 数据集**

无，本文为理论性工作，未使用具体数据集。

**📈 对比分析**

未进行实验比较，性能由理论上限与下界给出；在一维示例中最优值约为 O(α⁻¹√(2log n))，多算子构造表明至少需要三类算子才能在给定查询预算内达到约 1/10 的最优逼近。

**⚠️ 局限性**

限制在于模型高度抽象，未考虑实际 LLM 推理成本与算子实现细节，也缺乏对真实任务的实证验证，且只给出了理论证明。

---

## 497. Beyond Third-Person Audits: Situated Interaction Auditing for User-Centered LLM Bias Research

**arXiv ID:** 2606.12247 | [PDF](https://arxiv.org/pdf/2606.12247v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 498. BenDi: An Energy-Efficient Quasi-Stochastic Systolic Architecture for Edge Bioelectronics

**arXiv ID:** 2606.12235 | [PDF](https://arxiv.org/pdf/2606.12235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 499. Adapting Prithvi-EO for Fallow Detection for Food-Water Nexus: ViT-Adapter Necks and Parameter-Efficient Backbone tuning of Geospatial Foundation Model

**arXiv ID:** 2606.12218 | [PDF](https://arxiv.org/pdf/2606.12218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 500. Anatomically Conditioned Recurrent Refinement for Topology-Aware Circle of Willis Segmentation

**arXiv ID:** 2606.12319 | [PDF](https://arxiv.org/pdf/2606.12319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 501. Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training

**arXiv ID:** 2606.12240 | [PDF](https://arxiv.org/pdf/2606.12240v1)

**作者:** Shilong Zong `[一作]` (Virginia Tech), Hoda Eldardiry `[通讯]` (Virginia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种结合连续时间液态神经网络、混合专家和多尺度时间注意机制的多速率混合专家框架，用于多变量时间序列预测。

**💡 创新点**

创新点在于：①为每个专家引入不同时间常数实现多尺度动力学建模；②同时加入特征级与时间级注意力提升鲁棒性与长程依赖；③将连续时间动力学与混合专家和注意力整合为单一统一架构。

**🔧 技术方法**

使用的技术包括：Liquid Neural Network（连续时间动力学）、Mixture‑of‑Experts、单调时间常数分层（多速率），以及特征级注意力和基于点积的时间注意力。

**📊 数据集**

在临床重症监护病房（ICU）收集的多变量生理信号数据集上进行败血症预测实验。

**📈 对比分析**

与LSTM、单一LNN、普通MoE和无注意力的MR‑MoE进行对比；实验显示MR‑MoE‑Attention在AUROC和AUPRC上分别提升至约0.65‑0.68和0.45，优于所有基线，且计算效率保持可接受。

**⚠️ 局限性**

局限性包括：①时间常数手工设定且不可学习；②多专家训练为统一优化，可能导致时间尺度间干扰；③连续时间积分和注意力机制增加了实现复杂度和计算开销。

---

## 502. An Electric Potential-Augmented Benchmark Dataset for Physics-Guided Image Reconstruction of Electrical Capacitance Tomography

**arXiv ID:** 2606.12226 | [PDF](https://arxiv.org/pdf/2606.12226v1)

**作者:** Xinqi Zhang `[一作]` (Tsinghua University), Lihui Peng `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个包含20,000个二维八电极电容耦合示例的电势增强基准数据集，并在该数据集上开展前向预测与反向重建实验，验证电势映射对模型性能的提升；

**💡 创新点**

首次将每个样本的八个激励条件下的全场电势图与传统电容-介电分布配对，提供可复现的COMSOL–MATLAB生成流程，为物理引导的机器学习奠定基础；

**🔧 技术方法**

采用COMSOL有限元求解静电场生成数据，利用深度学习模型进行电势预测、前向回归与反向重建，并用MSE/MAE等指标评估；

**📊 数据集**

使用了包含介电分布、28维电容向量、八个激励电势图以及空管/满管参考的20,000样本二维八电极ECT基准数据集，覆盖四种典型流动模式；

**📈 对比分析**

通过四种模型（A基线、B预测电势、C真电势、D仅电势）在IID与OOD（两柱）测试中对比，B模型在前向任务上MSE/MAE提升约30%/20%，在反向任务IID提升约22%/10%，C模型性能最佳，说明电势信息丰富；

**⚠️ 局限性**

仅限二维有限元、单一八电极配置、单一介电对比、无噪声或实验验证、未覆盖三维结构或多种对比情形等限制。

---

## 503. Using Explainability as a Training-Time Reliability Signal for Efficient ECG Classification

**arXiv ID:** 2606.12252 | [PDF](https://arxiv.org/pdf/2606.12252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 504. Reassessing High-Performing LLMs on Polish Medical Exams: True Competence or Bias-Driven Performance?

**arXiv ID:** 2606.12250 | [PDF](https://arxiv.org/pdf/2606.12250v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 505. Slots, Transitions, Loops: Learning Composable World Models for ARC

**arXiv ID:** 2606.12316 | [PDF](https://arxiv.org/pdf/2606.12316v1)

**作者:** Gege Gao `[一作]` (University of Tübingen), Andreas Geiger `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了Loop-OWM模型，将ARC任务视为对象中心的视觉符号状态转换，通过演示驱动的任务编码和循环递归更新实现规则诱导与执行。

**💡 创新点**

将ARC规则表述为可组合的对象级状态转换，并采用循环递归与槽注意力的双分支更新，实现密集传播与对象级修正的统一框架。

**🔧 技术方法**

采用槽注意力对象编码、任务摘要查询、基于Transformer的循环转换模块、颜色原型槽初始化及两阶段离线与测试时适配训练等技术。

**📊 数据集**

在ARC-1与ARC-2基准上进行评估，并利用RE-ARC与ARC-GEN进行离线程序化增强，测试时通过旋转、翻转与颜色置换等增强。

**📈 对比分析**

与现有紧凑视觉模型、循环ViT以及LLM系统比较，Loop-OWM在ARC-1上达成约67.3%/68.5%（pass@2）成绩，超过所有紧凑视觉模型，并在ARC-2上以约60%/70%相对优势接近LLM水平。

**⚠️ 局限性**

循环步骤缺乏可解释的中间语义，未能获得稳定的阶段性推理；缺乏轨迹级监督导致模型内部更新不易解释。

---

## 506. DrivingAgent: Design and Scheduling Agents for Autonomous Driving Systems

**arXiv ID:** 2606.12236 | [PDF](https://arxiv.org/pdf/2606.12236v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 507. Selection Integrity for LLM Graph Memory: An Accumulability Criterion for Information-Flow-Blind Retrieval

**arXiv ID:** 2606.12290 | [PDF](https://arxiv.org/pdf/2606.12290v1)

**作者:** Zeming Fei `[一作]` (University of Technology Sydney), Ying Zhang `[通讯]` (University of Technology Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对图形内存的选择完整性标准，旨在解决信息流盲检索中的安全问题。

**💡 创新点**

创新点在于提出了一个可验证的标准，明确了哪些选择器会暴露于信息流盲区，并且提供了一种新的防御机制来关闭这一通道。

**🔧 技术方法**

使用了图形计算、个性化PageRank等技术来分析选择器的行为，并提出了基于选择完整性的防御机制。

**📊 数据集**

使用了多个数据集，包括真实的多会话代理内存和控制的知识图谱数据集（如HotpotQA、MuSiQue等）。

**📈 对比分析**

与现有的代理内存防御方法进行比较，发现所有测试的防御方法在面对选择通道时都是盲目的，而本文提出的防御机制在零过载和2-3%的延迟下将伤害降至零。

**⚠️ 局限性**

限制在于该研究主要集中在图形内存的选择完整性上，未能覆盖所有可能的攻击向量，且在实际应用中可能需要考虑更多的上下文因素。

---

## 508. CellNet -- Localizing Cells using Sparse and Noisy Point Annotations

**arXiv ID:** 2606.12286 | [PDF](https://arxiv.org/pdf/2606.12286v1)

**作者:** Benjamin Eckhardt `[一作]` (University of Göttingen), Constantin Pape `[通讯]` (University of Göttingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于稀疏点注释的回归式细胞计数方法 CellNet，用于相位对比显微镜图像的自动计数。

**💡 创新点**

创新点在于将稀疏点注释转换为密度图进行回归预测，显著降低标注成本，并在低数据量下提供可行的计数方案。

**🔧 技术方法**

使用 U‑Net 结构结合 ResNet 编码器的全卷积网络，训练时采用 MSE+ BCE 损失，并对未标注区域做掩码处理。

**📊 数据集**

评估数据集包括 3 张稀疏点注释的 Sanger23 图像以及 LiveCell 数据集（2% 训练分割中心点），并与 Countess 3 FL 自动计数器进行对比。

**📈 对比分析**

与零样本检测方法 CellposeSAM 对比，CellNet 在 LiveCell 上 sMAPE 为 30.5%/MAE 148，Sanger23 上 sMAPE 51.0%/MAE 1186，F1 分数均较低，显示计数准确度和定位仍有提升空间。

**⚠️ 局限性**

局限性包括：回归方法在高细胞密度时泛化差，定位准确性低导致 F1 低；仅使用极少注释数据导致标签噪声影响；缺乏对计数总和的直接监督，导致计数误差较大。

---

## 509. DiffCold: A Diffusion-based Generative Model for Cold-Start Item Recommendation

**arXiv ID:** 2606.12245 | [PDF](https://arxiv.org/pdf/2606.12245v1)

**作者:** Kangning Zhang `[一作]` (Shanghai Jiao Tong University), Jianghao Lin `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 DiffCold，一种基于扩散模型的冷启动推荐框架，解决冷暖物品表示不一致导致的两难问题。

**💡 创新点**

创新点在于引入检索增强聚合器为生成过程提供合理起点，并通过模拟对齐模块利用对比学习实现冷暖表示分布一致，从而消除传统方法的“seesaw dilemma”。

**🔧 技术方法**

采用扩散式生成模型（DDPM）结合 BPR 目标、对比学习（InfoNCE）、检索聚合等技术进行训练与推断。

**📊 数据集**

在 Movielens、Citeulike 和 Xing 三大公开数据集上进行实验，均将 20% 项目设为冷启动。

**📈 对比分析**

与 Dropout、GAN、VAE、对齐等多种基线相比，DiffCold 在冷暖和整体 Recall@20 / NDCG@20 指标上均实现显著提升，同时保持了相对可接受的训练与推断效率。

**⚠️ 局限性**

局限性包括扩散步骤较多导致训练时间延长，且模型对检索聚合器所依赖的内容特征质量敏感。

---

## 510. Finding Sparse Subnetworks in One Training Cycle via Progressive Magnitude-Based Pruning

**arXiv ID:** 2606.12278 | [PDF](https://arxiv.org/pdf/2606.12278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 511. Why AI Slop Matters, but Not Like That

**arXiv ID:** 2606.12285 | [PDF](https://arxiv.org/pdf/2606.12285v1)

**作者:** Sachita Nishal `[一作]` (Northwestern University), Kimon Kieslich `[通讯]` (University of Hohenheim)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

对AI slop的社会技术影响进行批判性讨论，并提出新的研究议程

**💡 创新点**

将社会科学与伦理视角系统性引入AI slop研究，挑战原先单纯关注个体偏好满足的框架

**🔧 技术方法**

无具体技术实现，主要以理论分析和批判性框架为主

**📊 数据集**

无数据集使用，讨论基于文献综述与案例分析

**📈 对比分析**

未涉及实验或性能对比，主要通过逻辑推理和案例对照展开讨论

**⚠️ 局限性**

局限于理论性讨论，缺乏实证验证，未来需开展经验研究以检验提出的议程和假设

---

## 512. Bridging Day and Night: Unsupervised Cross-Domain Re-Identification with Synergistic Prompt and Prototype Learning

**arXiv ID:** 2606.12258 | [PDF](https://arxiv.org/pdf/2606.12258v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 513. Holding the FP8 Quality Ceiling at 8-Bit Weights and Activations: INT8 and GGUF Post-Training Quantization of Ideogram 4.0 for Consumer GPUs

**arXiv ID:** 2606.12280 | [PDF](https://arxiv.org/pdf/2606.12280v1)

**作者:** Deep Gandhi `[一作]` (Transformer Lab), Tony Salomone `[通讯]` (Transformer Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不支持 FP8 张量核心的 Ampere RTX 3090 GPU 上，对 Ideogram 4.0（9.3B 双分支流匹配 DiT）进行 8 位 INT8 W8A8 量化，并同时提供 GGUF Q4_K（4.5‑bit 权重）量化方案，确保在保持 FP8 质量上限的同时显著提升文本渲染质量并降低存储占用。

**💡 创新点**

创新点包括：
- 结合 SmoothQuant 与 per‑token 动态激活，并针对 FFN down‑projections 仅保护 8% 的线性层实现无质量损失的 8 位量化；
- 引入 OCR 评估作为文本渲染质量的核心指标，首次在此类模型中系统衡量量化对文本清晰度的影响；
- 在同等磁盘大小下展示 GGUF Q4_K 超越 NF4 的质量优势，确立了 4‑bit 量化的 Pareto 前沿；
- 对 8 位量化在 Ampere 上的性能瓶颈进行细致剖析，为后续融合 INT8 GEMM 的实现指明方向。

**🔧 技术方法**

所用技术：
- SmoothQuant（α=0.5）权重/激活迁移；
- per‑channel INT8 权重 + per‑token 动态 INT8 激活；
- 混合精度保护（BF16 保护 FFN down‑projections 和时间嵌入输出）；
- GGUF k‑quant（Q4_K、Q8_0）权重‑仅量化；
- 层级脆弱性分析（max‑abs × kurtosis 评估）与保护大小阶梯；
- OCR（EasyOCR）精确匹配与编辑距离评估；
- 参考‑无参考评估（PSNR、SSIM、LPIPS）以及 PickScore/CLIPScore。

**📊 数据集**

使用的数据集：
- PartiPrompts 作为提示来源；
- 128 提示用于校准；
- 200 提示（无文本）用于质量基准；
- 100 提示（63 含 OCR 目标）用于文本渲染基准；
- 300 提示（合并所有基准）用于参考‑无参考质量对比。

**📈 对比分析**

比较方法与性能：
- 与 FP8 参考和 NF4 发布版本进行 paired 10,000‑sample bootstrap 95% 置信区间对比；
- INT8 在 PickScore/CLIPScore 上与 FP8 无显著差异（CI 包含 0），且在 CLIP 上超过 NF4 1.9 分；
- OCR NED 0.704（INT8）接近 FP8（0.715），明显优于 NF4（0.760）；
- 在相同磁盘大小下 Q4_K 在 Pick/CLIP 上比 NF4 提升 0.96/3.57 分；
- 速度：FP8 172.9 s/image，NF4 164.5 s/image，INT8 184–185 s/image；Q4_K 203.3 s/image；
- 内存：FP8 18.6 GB，NF4 10.4 GB，INT8 18.6 GB，Q4_K 10.4 GB。

**⚠️ 局限性**

局限性：
- 8 位量化未带来显著速度提升，需要实现 Ampere‑INT8 融合 GEMM；
- 仅在单一 RTX 3090 集群上测试，缺乏跨硬件验证；
- 统计范围受限于单个 seed（1000）和 50–200 提示子集，未提供多 seed 结果；
- 以 FP8 为参考，未有真正 BF16 公共基准；
- 未对文本编码器或 VAE 进行量化，整体系统仍受非量化模块瓶颈；
- Q4_K 的参考‑无参考质量差距较大，表明仅靠无参考分数难以判定文本清晰度。

---

## 514. Rule Taxonomy and Evolution in AI IDEs: A Mining and Survey Study

**arXiv ID:** 2606.12231 | [PDF](https://arxiv.org/pdf/2606.12231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 515. Identifying cybersickness causes in virtual reality games using symbolic machine learning algorithms

**arXiv ID:** 2606.12214 | [PDF](https://arxiv.org/pdf/2606.12214v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 516. MLT-Dedup: Efficient Large-Scale Online Video Deduplication via Multi-Level Representations and Spatial-Temporal Matching

**arXiv ID:** 2606.12215 | [PDF](https://arxiv.org/pdf/2606.12215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 517. The Standard Interpretable Model: A general theory of interpretable machine learning to deductively design interpretable methods using Lagrangian mechanics

**arXiv ID:** 2606.12289 | [PDF](https://arxiv.org/pdf/2606.12289v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 518. Efficient and Robust Online Learning to Rank in Decentralized Systems

**arXiv ID:** 2606.12246 | [PDF](https://arxiv.org/pdf/2606.12246v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 519. A Turbo-Inference Strategy for Object Detection and Instance Segmentation

**arXiv ID:** 2606.12371 | [PDF](https://arxiv.org/pdf/2606.12371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 520. Context-Driven Incremental Compression for Multi-Turn Dialogue Generation

**arXiv ID:** 2606.12411 | [PDF](https://arxiv.org/pdf/2606.12411v1)

**作者:** Yeongseo Jung `[一作]` (Hong Kong University of Science and Technology), Lei Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向多轮对话的上下文驱动增量压缩框架 C‑DIC，能在每轮时检索、压缩并写回对话记忆，避免重复编码完整历史；

**💡 创新点**

创新点在于将对话视为交织的主题线索，使用可检索的可修订压缩状态；实现轻量级检索→压缩→写回循环，并采用检索感知的截断反向传播（ra‑TBPTT）以提升长序列鲁棒性；

**🔧 技术方法**

核心技术包括：基于预训练 ICAE 的压缩器、冻结生成器、语义检索与记忆更新策略、梯度无关写回、检索感知 TBPTT；

**📊 数据集**

在 MSC（多会话聊天）和 REALTALK（WhatsApp式多会话）数据集上进行训练与评估，亦使用 MSC‑QA 与 LongMemEval 作为诊断；

**📈 对比分析**

与全上下文、截断、摘要、检索、自动压缩等基线对比，C‑DIC 在 PPL、BLEU、ROUGE 上均居首位，且在 428 轮对话下保持稳定 3–3.5 秒推理时延，显著优于全提示和静态压缩方法；

**⚠️ 局限性**

局限性包括：记忆槽数随主题切换可能增大；依赖预训练压缩器，性能受初始化影响；潜在隐私泄露风险；评估范围有限，未覆盖更广域对话场景；

---

## 521. Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics

**arXiv ID:** 2606.12365 | [PDF](https://arxiv.org/pdf/2606.12365v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 522. Latent World Recovery for Multimodal Learning with Missing Modalities

**arXiv ID:** 2606.12362 | [PDF](https://arxiv.org/pdf/2606.12362v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 523. Fast-SDE: Efficient Single-Microphone Sound Source Distance Estimation in Reverberant Environments

**arXiv ID:** 2606.12339 | [PDF](https://arxiv.org/pdf/2606.12339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 524. A Five-Plane Reference Architecture for Runtime Governance of Production AI Agents

**arXiv ID:** 2606.12320 | [PDF](https://arxiv.org/pdf/2606.12320v1)

**作者:** Krti Tallam `[一作]` `[通讯]`, Krti Tallam

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种针对生产级 AI 代理的运行时治理参考架构，核心由思维平面（决策）、四个基础设施平面（网络、身份、端点、数据）以及结构化审计 substrate 组成；

**💡 创新点**

创新点在于：1）引入五平面治理结构，将决策与执行统一；2）提供“stop‑anywhere”中介机制与六种中断原语；3）设计复合主体与能力衰减的结构化模型，解决代理链授权与混淆代理问题；4）构建可重构、可回滚的审计证据体系；

**🔧 技术方法**

技术上结合了对象能力理论、Macaroon 认证、零信任网络、安全多层管控；实现中使用状态化决策引擎、事件驱动的中介点、加密签名的能力链、以及 Saga/补偿模式来支持回滚；

**📊 数据集**

论文主要在实验上使用内部企业数据工作流、金融、医疗、软件开发与客服等四个生产场景，未公开具体数据集；

**📈 对比分析**

通过对七类生产代理威胁的前置和补偿测试，评估了安全性与效用两个指标；在基准实现中决策时间仅几微秒，证明了高吞吐；相较于传统只做允许/拒绝的策略引擎，显著提升了安全覆盖与可追溯性；

**⚠️ 局限性**

局限性包括：1）对回滚支持依赖补偿能力，部分外部操作不可逆；2）需在代理运行时实现多点中介与状态持久化，增加复杂度；3）对高性能需求下的网络、数据平面异步实现仍可能导致延迟；4）当前仅验证内部案例，缺乏大规模多租户的外部评估。

---

## 525. A coupled finite element formulation for chemo-mechano-thermodynamical contact and its application to bonding and debonding

**arXiv ID:** 2606.12375 | [PDF](https://arxiv.org/pdf/2606.12375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 526. How Seemingly Inconsequential Design Choices Dictate Performance of LLMs in Pathology

**arXiv ID:** 2606.12407 | [PDF](https://arxiv.org/pdf/2606.12407v1)

**作者:** Kian R. Weihrauch `[一作]` (Massachusetts Institute of Technology), Arjun K. Manrai `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多种病理任务上系统地研究了将大语言模型（LLM）应用于全切片图像（WSI）的输入配置因素，包括推理模式、补丁尺寸、放大倍数和补丁数量；

**💡 创新点**

证明了输入配置是影响LLM在WSI任务中性能的关键因素，并提出了一种统一、平衡的配置（All-in-One推理、896像素、10×放大、20个补丁），显著提升了模型性能，弥补了此前对域特定模型优势的误判；

**🔧 技术方法**

使用LLM（如GPT‑5、Qwen 3.5 Plus、Gemini 3 Flash）配合图像补丁抽取工具Trident，采用全因子实验设计、方差分解与交互分析，以及多阶段评估策略；

**📊 数据集**

主要使用MultiPathQA基准（涵盖TCGA、GTEx、PANDA、SlideBench、ExpertVQA）以及留出的CPTAC癌症类型分类数据；

**📈 对比分析**

将优化后的配置与先前的“标准”Patch模式、Thumbnail模式以及专门的病理模型进行对比，结果显示在TCGA和GTEx分类任务上提升约+28.8%和+33.5%，在VQA任务上提升约+24.2%；

**⚠️ 局限性**

局限性包括：仅对GPT‑5进行全因子实验，未对其他模型探索不同配置；输入配置之外的因素（如重叠、预处理）未被考虑；实验受限于API调用成本，难以全面评估多推理模式；模型更新可能影响绝对性能但相对趋势预期稳定。

---

## 527. Echoes of the Prior: A Computational Phenomenology of Forgetting

**arXiv ID:** 2606.12340 | [PDF](https://arxiv.org/pdf/2606.12340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 528. World Pilot: Steering Vision-Language-Action Models with World-Action Priors

**arXiv ID:** 2606.12403 | [PDF](https://arxiv.org/pdf/2606.12403v1)

**作者:** Zefu Lin `[一作]` (Institute of Automation Chinese Academy of Sciences), Zhaoxiang Zhang `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了World Pilot框架，通过在Vision‑Language‑Action (VLA)策略中加入来自World‑Action Model (WAM)的先验，实现对场景动态和动作轨迹的双向指导。

**💡 创新点**

创新点在于将WAM的场景演化潜在表示与动作轨迹先验分别通过Latent Steering和Action Steering两条路径注入VLA，使得VLM隐藏状态获得动态预测，动作生成器获得运动先导，显著提升了在OOD和真实机器人任务中的稳健性。

**🔧 技术方法**

采用了视觉语言模型（如Qwen3‑VL）作为编码器，Diffusion Transformer实现动作生成，并使用Cosmos Policy等视频预训练的World‑Action Model提供潜在表示和轨迹先验，结合残差交叉注意力和前缀token注入等技术。

**📊 数据集**

在LIBERO‑Plus、RoboCasa等仿真基准以及四项真实机器人操作任务（堆叠方块、折叠毛巾、放置水果、容器盖对齐）上进行评估。

**📈 对比分析**

与现有最强基线相比，World Pilot在LIBERO‑Plus OOD总成功率上达到84.7%，比最强基线高2.6个百分点，并在所有真实机器人设置中获得最高成功率，尤其在视角、光照、背景等扰动下优势明显。

**⚠️ 局限性**

局限性包括对WAM预训练分布的依赖，导致在WAM覆盖范围之外的场景下先验效果减弱；在语言、机器人和布局等轴向的提升有限；每个决策步需要额外的WAM前向推理，增加计算开销；且未进行WAM与VLA的联合训练，缺乏更紧密的协同。

---

## 529. DIRECT: When and Where Should You Allocate Test-Time Compute in Embodied Planners?

**arXiv ID:** 2606.12402 | [PDF](https://arxiv.org/pdf/2606.12402v1)

**作者:** Jadelynn Dao `[一作]` (Stanford University), Marco Pavone `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于多模态上下文的动态路由框架，按任务分配测试时计算，提升视觉语言模型规划器的效率与效果。

**💡 创新点**

将测试时计算视为非均匀资源，对不同推理深度、模型规模与记忆架构进行任务感知分配，从而实现成本与性能的最优权衡。

**🔧 技术方法**

使用 SigLIP 与 BGE‑M3 的多模态特征编码、轻量化路由器（线性、KNN、MLP 等）以及回归预测来估计质量与成本。

**📊 数据集**

在 VLABench、RoboMME、Franka DROID 硬件实验以及合成任务上进行评估。

**📈 对比分析**

与最强模型对比，路由器在保持近乎相同成功率的同时，平均延迟降低 65%，在基准数据集上与 oracle 近乎一致。

**⚠️ 局限性**

仅在固定模型池内工作，需离线收集质量/成本数据，成本指标为延迟/ FLOPs，未考虑跨阶段依赖或动态池变更。

---

## 530. Fourier Features Let Agents Learn High Precision Policies with Imitation Learning

**arXiv ID:** 2606.12334 | [PDF](https://arxiv.org/pdf/2606.12334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 531. On Subquadratic Architectures: From Applications to Principles

**arXiv ID:** 2606.12364 | [PDF](https://arxiv.org/pdf/2606.12364v1)

**作者:** Anamaria-Roberta Hartl `[一作]` (Johannes Kepler University Linz), Sepp Hochreiter `[通讯]` (Johannes Kepler University Linz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5a41884c-404f-4688-a89c-aa238c10fe68` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 xLSTM、Mamba-2 和 Gated DeltaNet 三种主流子二次注意力架构在复杂数据领域（代码、时间序列）进行头对头的实证比较，并提出统一的记忆动力学框架来解释差异；同时通过合成长度泛化任务验证了累积与有限状态跟踪两种原始能力。

**💡 创新点**

①首次在结构化依赖强的数据上进行实证对比；②提出统一的线性注意力+门控记忆框架，揭示不同架构在累积和状态跟踪两大原始能力上的表现差异；③通过合成任务验证了该框架的预测。

**🔧 技术方法**

子二次注意力（线性注意力）+门控机制、Mamba-2 状态空间模型、DeltaNet 快速权重更新、xLSTM 结合线性与非线性递归块；实验使用 Transformer 线性化蒸馏、从零预训练、时间序列预训练等多种训练策略。

**📊 数据集**

代码数据集：Nemotron-CC-Code-v1（20B/100B 语料）与 Nemotron-CC-Code-v1+FineWeb-Edu；代码生成评测：HumanEval、HumanEval+、MBPP、MBPP+；基准推理与常识：HellaSwag、PIQA、ARC-Easy、ARC-Challenge、WinoGrande；时间序列预训练使用统一语料、补丁方案；评测基准：GIFT‑Eval（MASE、CRPS）。

**📈 对比分析**

比较方法：从零预训练、Transformer-to-Subquadratic 蒸馏、时间序列预训练三大场景；对比指标包括代码生成通过 HumanEval pass@k、推理常识准确率、时间序列 MAE、CRPS；结果显示 xLSTM（如 xLSTM[7:1]、xLSTM[3:1]、xLSTM[1:1]）在几乎所有设置中优于 Mamba-2 与 Gated DeltaNet，优势在结构化任务上更显著；在合成任务中 xLSTM 同时在累积和状态跟踪两类任务上表现最佳。

**⚠️ 局限性**

局限性：实验规模主要集中在 400M 参数级别的代码模型和 1M–80M 的时间序列模型，未覆盖更大规模或多教师情况；只评估了三种子二次注意力架构，未对更广泛的操作符进行系统比较；实际部署中的硬件加速效果及训练成本等方面未做深入探讨。

---

## 532. Adjoint Method versus Physics-Informed Neural Networks in PDE-Constrained Inverse Problems

**arXiv ID:** 2606.12337 | [PDF](https://arxiv.org/pdf/2606.12337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 533. VLGA: Vision-Language-Geometry-Action Models for Autonomous Driving

**arXiv ID:** 2606.12396 | [PDF](https://arxiv.org/pdf/2606.12396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 534. APT: Action Expert Pretraining Improves Instruction Generalization of Vision-Language-Action Policies

**arXiv ID:** 2606.12366 | [PDF](https://arxiv.org/pdf/2606.12366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 535. Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling

**arXiv ID:** 2606.12370 | [PDF](https://arxiv.org/pdf/2606.12370v1)

**作者:** Yucheng Li `[一作]` (Alibaba Inc.), Jingren Zhou `[通讯]` (Alibaba Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型在强化学习（RL）后训练阶段使用的多词预测（MTP）进行系统性研究，并提供可直接集成的实用方案。

**💡 创新点**

创新点包括：① 证明目标模型熵对 MTP 接受率的线性约束；② 提出直接最小化总变分距离（TV）的端到端多步 TV 损失，显著提升拒绝采样接受率；③ 发现预训练的 TV 损失 + 拒绝采样即可在整个 RL 训练中保持高接受率，省去在线更新。

**🔧 技术方法**

使用的技术包括：多词预测（MTP）框架、目标模型和草稿模型的对齐、拒绝采样与目标仅采样两种接受方法、TV 损失与端到端 TV 损失的梯度分析、异步 RL（GRPO）框架以及 SGLang/veRL 工具。

**📊 数据集**

使用的数据集覆盖推理、代码、数学推理与代理任务，包括 Qwen3.5/3.6/3.7 的混合 RFT 数据、Agent、SWE‑Bench、Math、Code、MT‑Bench 等。

**📈 对比分析**

与传统 CE/KL 训练的 MTP 进行对比，使用拒绝采样 + TV 损失在各任务上提升了 3–8% 的接受率，最高可达 95%；在 RL 训练中实现 1.5–1.8× 的整体加速，Agent 任务最高可达 2.4× 的单步延迟下降。

**⚠️ 局限性**

局限性包括：TV 损失的熵不变性只在训练覆盖的熵范围内成立；在 RL 期间若熵显著超出该范围，可能需要在线 MTP 更新；理论分析中的假设（如概率比例误差）尚未严格证明。

---

## 536. Traceable Virtual Sea Trials in the Marine Robotics Unity Simulator for Manoeuvring Assessment of Unmanned Surface Vehicles

**arXiv ID:** 2606.12349 | [PDF](https://arxiv.org/pdf/2606.12349v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 537. Local Stress Redistribution Controls Interactions between Hydraulic Fractures and Pre-existing Fractures

**arXiv ID:** 2606.12347 | [PDF](https://arxiv.org/pdf/2606.12347v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 538. Atlas H&E-TME: Scalable AI-Based Tissue Profiling at Expert Pathologist-Level Accuracy

**arXiv ID:** 2606.12346 | [PDF](https://arxiv.org/pdf/2606.12346v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 539. ALIGNBEAM : Inference-Time Alignment Transfer via Cross-Vocabulary Logit Mixing

**arXiv ID:** 2606.12342 | [PDF](https://arxiv.org/pdf/2606.12342v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 540. Claw-SWE-Bench: A Benchmark for Evaluating OpenClaw-style Agent Harnesses on Coding Tasks

**arXiv ID:** 2606.12344 | [PDF](https://arxiv.org/pdf/2606.12344v1)

**作者:** Mengyu Zheng `[一作]` (TokenRhythm Technologies), Yu Wang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Claw-SWE-Bench benchmark，利用 adapter 让通用代理如 OpenClaw 能在 SWE‑bench 环境下生成可评测的补丁。

**💡 创新点**

创新点在于把 agent harness 作为可控实验变量，并提供 80‑实例的 Lite 子集以低成本复现全量评测。

**🔧 技术方法**

使用统一的适配器协议、SWE‑bench 评估接口、固定 prompt 与预算，结合多语言任务集进行实验。

**📊 数据集**

使用 350 个真实 GitHub issue 解决实例（涵盖 8 种语言）以及 80‑实例 Lite 子集。

**📈 对比分析**

通过 Pass@1 与总 API 成本的 Pareto 视角比较不同模型与 harness，发现 harness 影响可达 12.5–27.4pp，且不同模型在成本上的差距可达数百美元。

**⚠️ 局限性**

限制包括单次跑实验、仅评测 5 种 harness 与 2 种模型、对供应商计费和缓存策略敏感，未来需多次复现与更广泛的模型‑harness 组合。

---

## 541. OCELOT: Inference-Leakage Budgets for Privacy-Preserving LLM Agents

**arXiv ID:** 2606.12341 | [PDF](https://arxiv.org/pdf/2606.12341v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 542. FACTR 2: Learning External Force Sensing for Commodity Robot Arms Improves Policy Learning

**arXiv ID:** 2606.12406 | [PDF](https://arxiv.org/pdf/2606.12406v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 543. PROJECTMEM: A Local-First, Event-Sourced Memory and Judgment Layer for AI Coding Agents

**arXiv ID:** 2606.12329 | [PDF](https://arxiv.org/pdf/2606.12329v1)

**作者:** Ripon Chandra Malo `[一作]` (University of Utah), Tong Qiu `[通讯]` (University of Utah)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为projectmem的本地优先、事件源、纯文本的 AI 编程代理记忆层，能够记录项目的 issue、attempt、fix、decision、note 等事件并生成可读摘要；

**💡 创新点**

创新点在于引入了“Memory-as-Governance”——一个基于已记录失败历史的确定性预执行判断门，能够在代理尝试操作前主动提示并防止重复错误，同时保持记忆不可变、可审计；

**🔧 技术方法**

使用了事件日志存储（JSON Lines + Markdown）、Model Context Protocol（MCP）服务器、FastMCP、Python 3.10+、Typer CLI、watchdog 文件监控、D3.js 可视化；

**📊 数据集**

通过自我实验对 10 个真实项目（涵盖机器学习、Web、音频工具等）收集 207 条事件数据进行评估；

**📈 对比分析**

与无记忆状态下的上下文重建进行对比，估算在 MCP 模式下每个会话加载 800–1,500 tokens，较无记忆方式节省 50% 以上 token；自我实验显示记忆持续增长且无缝与多 MCP 客户端兼容；

**⚠️ 局限性**

局限包括：在无历史事件的冷启动项目上缺乏警告；可能出现误报；不支持语义检索；目前仅单用户本地化；token 经济学评估为估算而非严格基准。

---

## 544. MARCIM-WG: A cyber wargame proposal based on math modeling applied in a naval scenario

**arXiv ID:** 2606.12395 | [PDF](https://arxiv.org/pdf/2606.12395v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 545. APPO: Agentic Procedural Policy Optimization

**arXiv ID:** 2606.12384 | [PDF](https://arxiv.org/pdf/2606.12384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 546. TAHOE: Text-to-SQL with Automated Hint Optimization from Experience

**arXiv ID:** 2606.12387 | [PDF](https://arxiv.org/pdf/2606.12387v1)

**作者:** Zhiyi Chen `[一作]` (ByteDance Inc.), Peng Li `[通讯]` (ByteDance Inc.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个持久化的 Hint Bank，用错误驱动的提示学习将编译与执行错误转化为可复用的语法与语义提示，并在运行时通过提示检索与逻辑规划实现 Text-to-SQL 的推理。

**💡 创新点**

创新点在于：① 把 prompt 优化转化为外部知识管理问题；② 设计语义提示的 Trigger–Strategy 结构，显式存储并对冲突进行排名；③ 在开发阶段完成错误分析并异步更新 Hint Bank，避免在线推理的高延迟与成本；④ 引入策略归因（attribution）对提示有效性进行量化，提升检索质量。

**🔧 技术方法**

主要技术包括：多采样推理、编译器反馈与执行反馈循环、LLM 生成提示与聚类、策略归因统计、范围过滤与触发检索、两阶段提示驱动生成（逻辑规划+SQL 合成）。

**📊 数据集**

在 Spider 2.0–Snow-0212 公开基准上进行实验，使用 113 条有监督样本进行开发阶段学习，另外 434 条无监督样本做 held‑out 验证。

**📈 对比分析**

与 Vanilla LLM、SQLGenie‑style RAG 以及不同底层模型（Doubao‑2.0‑lite、GPT‑5、GPT‑5.5）对比，Hint Bank 在 GPT‑5.5 上把 Pass Rate 提升 17.47pp、pass@4 提升 15.04pp、语法通过率 100%、平均修复回合从 2.79 降至 0.12；同一 Hint Bank 在弱模型上也能实现 10‑20pp 的提升；在 held‑out 上语法层面提升近 9pp，整体提升约 2‑3pp，SQLGenie‑style RAG 在语义层面表现不佳。

**⚠️ 局限性**

局限性：仅在开发阶段评估，部署阶段（人机循环反馈）未实验；触发器质量对检索召回影响大；策略稀疏导致罕见情况的排名不可靠；Hint Bank 随时间增大需监控与裁剪策略。

---

## 547. SPEA2$^+$: Improved Density Estimation in SPEA2 with Provable Runtime Guarantees

**arXiv ID:** 2606.12382 | [PDF](https://arxiv.org/pdf/2606.12382v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 548. DepthMaster: Unified Monocular Depth Estimation for Perspective and Panoramic Images

**arXiv ID:** 2606.12368 | [PDF](https://arxiv.org/pdf/2606.12368v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 549. CHORUS: Decentralized Multi-Embodiment Collaboration with One VLA Policy

**arXiv ID:** 2606.12352 | [PDF](https://arxiv.org/pdf/2606.12352v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 550. Illumination-Robust Camera-Based Heart-Rate Estimation for Physiological Sensing in Robots

**arXiv ID:** 2606.12378 | [PDF](https://arxiv.org/pdf/2606.12378v1)

**作者:** Zhi Wei Xu `[一作]` (National Cheng Kung University), Torbjörn E. M. Nordling `[通讯]` (National Cheng Kung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用远程光学脉搏图（rPPG）技术，在机器人摄像头下实现对人体心率的非接触估计，并针对照明变化进行鲁棒性设计。

**💡 创新点**

创新点在于整合PRNet 3D人脸对齐、片段级照明增强、残差时域标准化模块（RTSM）以及时频混合损失（Soft-Shifted Pearson + 频谱KL），在端到端框架中实现光照鲁棒的心率估计。

**🔧 技术方法**

采用PRNet进行人脸几何对齐；PhysFormer‑style Transformer作为预测器；RTSM实现时域特征标准化；Soft‑Shifted Pearson损失与频谱KL损失构成混合监督；片段级亮度对比度增强用于模拟照明变化。

**📊 数据集**

使用作者实验室收集的75人静态视频数据集，包含低、中、高三种照明水平（1、3、5），每人2分钟视频。

**📈 对比分析**

在静态全级混合照明协议下，实验将不同β值的时频权重进行比较，β=5时得到HR MAE 0.79 bpm、RMSE 2.40 bpm、相关系数0.982，较PhysFormer基线MAE下降93.6%，RMSE下降84.5%；不同β实验验证了时频权重平衡对性能的影响。

**⚠️ 局限性**

仅在静态场景下评估，未覆盖运动、机器人移动或在线适配场景；照明增强为人工模拟，缺乏对更广泛真实光照条件下的泛化验证。

---

## 551. Semantically-Aware Diver Activity Recognition Framework for Effective Underwater Multi-Human-Robot Collaboration

**arXiv ID:** 2606.12374 | [PDF](https://arxiv.org/pdf/2606.12374v1)

**作者:** Sadman Sakib Enan `[一作]` (University of Minnesota), Junaed Sattar `[通讯]` (University of Minnesota)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于 Transformer 的 DAR‑Net 框架，并创建了首个 Underwater Diver Activity（UDA）数据集，用于多人体与机器人在水下环境中的活动识别。

**💡 创新点**

创新点在于采用多损失（分类+分割）训练方案，使模型在全局活动识别与局部语义监督之间实现协同学习，从而显著提升了对重要水下目标（潜水员、机器人、物体）的关注度。

**🔧 技术方法**

使用 ResNeXt‑101 作为特征提取器，结合 Transformer 时序编码、编码器‑解码器分割分支以及自监督的多损失优化，辅以数据增强与 Transformer 位置/类别编码。

**📊 数据集**

使用了 UDA 数据集，该数据集包含 2,640 张带有像素级分割标注（潜水员、机器人、关键物体）的水下多主体协同场景图像。

**📈 对比分析**

在 30 条测试视频（每类 5 条）上与多种 SOTA 视觉动作识别模型对比，DAR‑Net 的整体准确率为 73.33%，显著高于其他模型（如 Late Temporal 仅 66.67%），并在平均精度、召回率和 F1‑score 上均保持领先。

**⚠️ 局限性**

局限性包括数据集规模相对较小、仅来自封闭水池实验，难以覆盖开放水域的复杂性，且对视觉上相似的“忙碌”与“机器人‑潜水员交互”类识别仍存在误差，未来需扩充数据与提升时序建模能力。

---

## 552. Reroute, Don't Remove: Recoverable Visual Token Routing for Vision-Language Models

**arXiv ID:** 2606.12412 | [PDF](https://arxiv.org/pdf/2606.12412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 553. ECYSAP EYE: From Cyber Situational Awareness to Mission-Centric Decision Support for Enhanced Cyberspace Operations

**arXiv ID:** 2606.12354 | [PDF](https://arxiv.org/pdf/2606.12354v1)

**作者:** Pantaleone Nespoli `[一作]` (Universidad de Murcia), Gregorio Martínez Pérez `[通讯]` (Universidad de Murcia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个面向任务的网络态势感知平台 ECYSAP EYE，构建了七组可消费的决策支撑工件，并通过模块化系统架构支持逐步部署。

**💡 创新点**

创新点在于将网络态势感知从单一共享画面延伸到完整决策闭环，聚焦任务关联工件（RCyP、CySR、WIAR、OPRE、DSH、AE、AAR），并通过系统化的 SoS 结构实现可逐步集成与验证。

**🔧 技术方法**

使用了模块化系统‑of‑systems 体系结构、数据平面（CTI、OSINT、OTX 采集、归一化、关联、可追溯），以及与 NATO FMN、数字工程工具链（网络仿真、模型）对接的接口。

**📊 数据集**

主要数据来源包括网络威胁情报（CTI）、公开情报（OSINT）和开放威胁交换（OTX），以及实验演习生成的仿真/真实事件数据。

**📈 对比分析**

与传统单一态势感知方案相比，ECYSAP EYE 在实战演习中实现了实时更新的 RCyP、情境报告与行动执行追踪；虽然未给出精确指标，但系统已达到 TRL 6‑7，满足实时决策阈值。

**⚠️ 局限性**

局限性包括：仍处于开发阶段，需进一步验证跨国互操作性；数据完整性与动态性挑战仍大；对复杂作战环境的适配与操作员培训仍需加强。

---

## 554. Doc-to-Atom: Learning to Compile and Compose Memory Atoms

**arXiv ID:** 2606.12400 | [PDF](https://arxiv.org/pdf/2606.12400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 555. Nonslop: A Gamified Experiment in Human-AI Collaborative Writing

**arXiv ID:** 2606.12350 | [PDF](https://arxiv.org/pdf/2606.12350v1)

**作者:** Maria Edwards `[一作]` (New York University), Julian Togelius `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一款名为 Nonslop 的 Web 游戏，用来观察在 AI 建议被明确禁止或惩罚的情境下，用户在写作任务中是否会采用 AI 生成的单词。

**💡 创新点**

① 将 AI 协助写作的默认“无摩擦”模式颠倒为“高摩擦”模式，使用户需在即时写作中做出是否接受 AI 建议的显式决策；② 通过轻量化游戏化框架揭示了在被抑制的环境下 AI 使用的真实微观行为；③ 通过用户聚类与提示级别分析，发现任务类型（尤其是解释性任务）显著影响 AI 采用率。

**🔧 技术方法**

使用基于 Phaser 3 的前端框架；在浏览器内运行 Qwen 2.5 0.5B Instruct 模型进行实时下一词预测；利用 OpenAI GPT‑4o‑mini‑2024‑07‑18 对提交文本进行 3 维度评分（相关性、语法、连贯性）。

**📊 数据集**

自建的写作提示集合（共 17 个创意、43 个观察、45 个个人、8 个哲学、67 个解释性，共 190 条提示），以及 74 名参与者共 214 条有效写作记录；提示按难度分层（1–3 级）并按关键词分为 5 类。

**📈 对比分析**

通过对 214 条提交进行描述性统计、用户聚类（k‑means, k=3）以及提示级别相关性分析。结果显示：约 73.8% 的提交未尝试使用 AI 建议；在用户层面，“极简主义者”占 72%；“选择性采用者”与“活跃采用者”分别占 16% 与 11%；提示层面，解释性提示的 AI 采用率最高，创意/观察提示最低。由于样本规模与技术限制，未能与传统写作工具在生产力或质量上做定量对比。

**⚠️ 局限性**

① 样本受限于仅能在支持 Web‑LLM 的浏览器与设备上运行，导致大约 53% 的尝试者无法进入游戏；② 浏览器端推理延迟高，可能导致部分用户放弃；③ 提交文本短小、任务仅为单句/短段，无法评估长期写作中的 AI 采用行为；④ 未与常规 AI 写作工具进行基准比较，缺乏对实际生产力或质量提升的定量评估。

---

## 556. Redesign Mixture-of-Experts Routers with Manifold Power Iteration

**arXiv ID:** 2606.12397 | [PDF](https://arxiv.org/pdf/2606.12397v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 557. System Report for CCL25-Eval Task 5: New Dataset and LoRA-Fine-Tuned Qwen2.5

**arXiv ID:** 2606.12392 | [PDF](https://arxiv.org/pdf/2606.12392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 558. ATLAS: Active Theory Learning for Automated Science

**arXiv ID:** 2606.12386 | [PDF](https://arxiv.org/pdf/2606.12386v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 559. Which Models Are Our Models Built On? Auditing Invisible Dependencies in Modern LLMs

**arXiv ID:** 2606.12385 | [PDF](https://arxiv.org/pdf/2606.12385v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 560. Verifiable Environments Are LEGO Bricks: Recursive Composition for Reasoning Generalization

**arXiv ID:** 2606.12373 | [PDF](https://arxiv.org/pdf/2606.12373v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 561. UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning

**arXiv ID:** 2606.12372 | [PDF](https://arxiv.org/pdf/2606.12372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 562. Anatomy of Post-Training: Using Interpretability to Characterize Data and Shape the Learning Signal

**arXiv ID:** 2606.12360 | [PDF](https://arxiv.org/pdf/2606.12360v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 563. Should LLM Agents Decide in Social Simulations? Comparing Finite-State and LLM-Based Decision Policies

**arXiv ID:** 2606.12369 | [PDF](https://arxiv.org/pdf/2606.12369v1)

**作者:** Alejandro Buitrago López `[一作]` (University of Murcia), José A. Ruipérez-Valiente `[通讯]` (University of Murcia)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在同一OSN模拟框架下比较了传统有限状态机（FSM）决策与基于LLM的行动选择，评估LLM是否能保持预设行为政策。

**💡 创新点**

创新点在于：①构建严格可比的实验环境（相同代理、网络、行动空间与上下文）；②系统评估多种LLM模型与三种提示策略的行为对齐度；③使用Jensen–Shannon Divergence量化对齐程度并记录计算成本。

**🔧 技术方法**

技术包括：Agent‑based OSN模拟、FSM/Markov决策、LLM提示（Base、Guided、Probabilistic）、JSD与拉普拉斯平滑、vLLM服务与CPU计时。

**📊 数据集**

数据集为合成OSN：1000个代理（4类用户），共10,000次行动选择；无真实社交网络数据。

**📈 对比分析**

比较方法：对每个模型-提示组合计算全局JSD与按用户类型的JSD，并记录10,000次决策的总耗时。结果显示：最优配置（GPT‑OSS+Probabilistic）JSD≈0.035，但其耗时≈427s，相比FSM的0.0007s快仅≈1.6×，平均LMM更慢≈560×。

**⚠️ 局限性**

局限性：LLM行动选择对齐不稳定且易受提示敏感；高计算成本；未在完整OSN模拟（扩散、网络演化等）中验证行为差异影响；实验仅涵盖三款公开模型，缺乏更广泛的模型与提示探索。

---

## 564. Measuring Semantic Progress in Multi-turn Dialogue via Information Gain

**arXiv ID:** 2606.12332 | [PDF](https://arxiv.org/pdf/2606.12332v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 565. Fair Comparison of Scheduling Algorithms on Heterogeneous Edge Clusters: A Continuous Adaptive Benchmark

**arXiv ID:** 2606.12343 | [PDF](https://arxiv.org/pdf/2606.12343v1)

**作者:** Zihang Wang `[一作]` (Technische Universitaet Wien), Schahram Dustdar `[通讯]` (Technische Universitaet Wien)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并使用统一接口的持续多模式调度(CMMS)基准平台，对六种不同设计范式（规则、贪心、DQN、AIF等）的调度器，在五种集群规模与两种负载强度下进行闭环对比实验。

**💡 创新点**

创新点包括：
• 统一调度器接口（FULL、LITE、OFFLOAD、SKIP）消除实验环境偏差；
• 关闭循环实验主机，确保所有调度器经历相同的工作负载演变；
• 双指标 SLO 评分（Raw SLO 与 Steady‑State SLO）以及切换税（Transition Tax），揭示切换成本；
• 记录每步决策开销与平台开销，实现调度器复杂度与性能的 Pareto 分析。

**🔧 技术方法**

技术手段：
• 采用 Jetson Orin / Orin Nano、Ryzen 7 组合的异构边缘集群；
• YOLOv8n 模型在 720p 城市交通视频上推理；
• 采用 1 Hz 控制间隔，按预设场景脚本注入流事件；
• 调度器实现包括 NoOp、Least‑Loaded、Heuristic、Myopic、DQN（双 DQN）和 AIF（主动推理）等。

**📊 数据集**

使用的数据集：
• 720p 城市交通视频（30 fps）与 YOLOv8n 检测模型；
• 通过脚本生成的流到达/离开事件序列（Ramp‑up、Burst、Steady‑overload、Oscillating）作为工作负载。

**📈 对比分析**

比较方法与性能：
• 在 5 种集群配置（2‑homo、2‑het、5‑node、8‑node light、8‑node heavy）下，对 6 个调度器进行 Raw SLO、Steady‑State SLO、每步决策开销等指标评估；
• 发现调度器排名随规模和负载显著变化，重负载时往往由规则基准夺回冠军；
• 高决策开销并不一定带来更好 SLO，低成本 Heuristic 在 heavy 场景仍能超越 AIF 与 DQN；
• Dual‑Metric 评分揭示切换成本对 Raw SLO 的影响，Transition Tax 明确显示不同调度器的切换代价。

**⚠️ 局限性**

局限性：
• 每个 (集群、调度器、场景) 单元仅跑 3 次，统计误差可能影响细微排名；
• Heavy 负载仅在 8‑node 集群测试，无法反映小规模集群的扩展性；
• DQN 训练预算固定为 10 轮单种子，较大预算可能提升 heavy 场景性能；
• 仅使用 Jetson/Ryzen 体系结构与视频分析工作负载，结果未必推广到其他 CMMS 场景（ML 服务、传感器融合等）。

---

