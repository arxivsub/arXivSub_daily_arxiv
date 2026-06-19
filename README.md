# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-19 | 今日论文总数: 590

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Information Lattice Learning as Probabilistic Graphical Model Structure Learning

**arXiv ID:** 2606.19366 | [PDF](https://arxiv.org/pdf/2606.19366v1)

**作者:** Haizi Yu `[一作]` (Kocree, Inc.), Lav R. Varshney `[通讯]` (Kocree, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

论文提出信息格学习（ILL）框架，将其解释为概率图模型（PGM）的结构学习，通过投影到分区格和提升规则实现可解释的概率规则。

**💡 创新点**

创新点在于将分区格中的抽象作为确定性商变量，将规则视为其边缘分布，从而把ILL转化为最大熵因子图学习；同时区分格图和贝叶斯网络的语义，提出抗链选择与可解释性评估。

**🔧 技术方法**

主要技术包括分区格理论、投影-提升（projection‑lifting）算法、最大熵（Shannon/L₂）重构、对抗链搜索、因子图与约束型概率模型、熵与描述长度正则化。

**📊 数据集**

论文没有给出具体实验数据集，而以象征性例子（钢琴音符、骰子点数）和理论示例说明方法；在实际应用中已在视觉分类等任务中达到 state‑of‑the‑art 结果。

**📈 对比分析**

与传统 PGM 的对比主要是概念层面，ILL 在解释性和抽象层次上更强；在实验上与视觉分类基线相比性能相当或更优，且能提供更直观的规则解释。

**⚠️ 局限性**

局限性包括：缺乏因果解释能力、对大规模状态空间时提升计算昂贵、L₂ 重构与 Shannon 最大熵的差异导致因子图形式不一、可辨识性与多重抗链解的歧义以及高阶因子导致推理成本高。

---

## 2. When to Trust, How to Distill: Multi-Foundation Model Guidance for Lightweight, Robust Scientific Time Series Forecasting

**arXiv ID:** 2606.19363 | [PDF](https://arxiv.org/pdf/2606.19363v1)

**作者:** Rupasree Dey `[一作]` (Colorado State University), Sangmi Lee Pallickara `[通讯]` (Colorado State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过 Guard 框架，将多模 TSFM 教师模型的知识蒸馏到轻量级学生模型，实现边缘设备可部署的科学时序预测。

**💡 创新点**

引入实例级上下文路由和不确定性门控机制，动态选择教师并在分布失配时抑制错误指导，解决 TSFM 零样本误差和资源限制问题。

**🔧 技术方法**

采用多教师知识蒸馏、上下文路由网络、温度门控网络、轻量级 Transformer 学生模型以及自适应温度调节等技术。

**📊 数据集**

在气象、生态碳通量、土壤水分、能源电网负荷四大科学领域的数据集上验证，包括 MPI‑BGC Jena 气象、DayCent 碳通量、Quench 土壤水分以及 ETTm1/ETTh1 电网负荷。

**📈 对比分析**

与零样本教师和现有 SOTA 深度学习基准（DLinear、PatchTST、iTransformer 等）对比，Guard 在所有数据集上均超过教师并在大多数任务中取得约 28% 平均 RMSE 降低，单数据集最佳 RMSE 低至 0.095，优于最强基准 40% 以上。

**⚠️ 局限性**

方法依赖教师提供的可靠不确定性估计，若教师误差高度相关或校准不足，路由效果下降；此外，教师推理的单次 GPU 前置成本为 15–28 小时，且在极端分布偏移时性能可能受限。

---

## 3. Vancomycert: A Certified Neuro-Symbolic Drug Delivery System (Case Study)

**arXiv ID:** 2606.19532 | [PDF](https://arxiv.org/pdf/2606.19532v1)

**作者:** Alistair Sirman `[一作]` (University of Southampton), Michael John Williams `[通讯]` (Schlumberger Cambridge Research)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并形式化验证了一套基于神经网络的 Vancomycin 给药控制器，实现了在无界时间内保证药物浓度不超过安全阈值。

**💡 创新点**

首次将网络验证与闭环药代动力学模型、无限时间不变性证明相结合，利用专用 DSL 将单一安全规范同时导出给自动验证器和交互式定理证明器。

**🔧 技术方法**

使用了 Vehicle（针对神经网络的 DSL）、Marabou（线性约束验证器）和 Coq（交互式定理证明器）三大技术，配合 ReLU MLP、PK/PD 一室模型及定理化的不变性推导。

**📊 数据集**

采用合成的临床模拟数据——从 PK/PD 仿真产生的 50 位病人、8 小时间隔的状态‑剂量对，随后按 80/20 划分用于训练和测试。

**📈 对比分析**

论文未给出与现有手工或基于规则的给药方案的性能比较，主要关注验证过程；在验证方面，Marabou 成功验证了线性性质，Coq 完成了无限时间安全证明，整体证明时间数小时级。

**⚠️ 局限性**

主要限制包括：无法提供完整的可重现证书（Marabou 仅给出不透明公理）、依赖线性化的安全性质、对 SMT 求解器的可靠性有限、网络规模受 ReLU 约束，且未对实际临床效果进行评估。

---

## 4. S-JEPA : Soft Clustering Anchors for Self-Supervised Speech Representation Learning

**arXiv ID:** 2606.19398 | [PDF](https://arxiv.org/pdf/2606.19398v1)

**作者:** Georgios Ioannides `[一作]` (Carnegie Mellon University), Ravid Shwartz-Ziv `[通讯]` (New York University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

S-JEPA 通过使用软目标的 GMM 后验分布与 KL 散度进行掩码预测，构建了一种单通道的自监督语音编码器训练流程。

**💡 创新点**

创新点在于将硬聚类标签替换为软 GMM 后验，并通过在线 GMM 更新和自适应层选择，消除了离线重聚类和手工层选择的需求。

**🔧 技术方法**

采用的技术包括 JEPA 结构、KL 损失、两阶段训练（MFCC 与 EMA 编码器特征的 GMM）、自适应有效秩层选择以及周期性切换的 EMA 速率。

**📊 数据集**

训练数据主要为约 83,000 小时的英语 LibriLight 与 Granary 子集，评估使用 SUPERB 基准任务（ASR、情感识别、槽位填充）。

**📈 对比分析**

与现有 SSL 方法相比，S‑JEPA 在 90M 参数以下实现了最低 WER（12.10%）并在情感识别上与 HuBERT‑Base（≈95M 参数）相当，仅占其 55% 参数量；在 SUPERB 任务中在子 90M 范围内均占优。

**⚠️ 局限性**

局限性包括对周期性 EMA 调度缺乏系统评估、仅在英语数据上验证、以及 GMM 与其他软目标方法之间的理论关联尚待深入探讨。

---

## 5. Deontic Policies for Runtime Governance of Agentic AI Systems

**arXiv ID:** 2606.19464 | [PDF](https://arxiv.org/pdf/2606.19464v1)

**作者:** Anupam Joshi `[一作]` (UMBC), Lalana Kagal `[通讯]` (MIT)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于Rei deontic policy语言与OWL/RDF语义网的外部治理引擎，用来在LLM驱动的代理AI系统执行工具调用或A2A消息时，进行权限、义务、豁免、冲突解决等多维度的运行时决策。

**💡 创新点**

创新点：①将义务（obligation）与豁免（dispensation）等非访问控制概念纳入政策表达，②利用元策略实现冲突解决，③通过OWL子类推理实现语义化的资源约束，④与A2AS等行业标准框架无缝对接，⑤在LLM之外实现毫秒级的高性能推理，消除提示注入导致的安全漂移。

**🔧 技术方法**

技术栈：Rei框架（基于OWL/RDF）、RDFox逻辑推理引擎、TripleExtractor中间件、A2AS B Pillar（凭证签名）和Meta-Policy API；部署为三步 Extract–Evaluate–Apply 的插件模式。

**📊 数据集**

数据集：文中主要使用自定义的OWL/Turtle 示例片段（如健康信息、金融交易等）作为演示数据；未引用公开工业数据集。

**📈 对比分析**

性能评估：在单机（RHEL 9 + RDFox 7.5）上，<10 ms 的端到端决策延迟，RDFox查询 < 1 ms，满足同步动作拦截需求；未给出与现有政策引擎（XACML、OPA、Cedar）的量化对比，主要通过案例演示表达能力差异。

**⚠️ 局限性**

局限性：①尚未完整实现加密凭证验证与跨域信任链；②义务执行与生命周期管理仍依赖后端人工或外部审计；③未在大规模多租户或分布式环境下做压力测试；④缺乏自动化的政策冲突检测与静态分析工具；⑤论文重点在功能演示，缺乏与行业标准完整集成与验证。

---

## 6. VERITAS: Verifier-Guided Proof Search for Zero-Shot Formal Theorem Proving

**arXiv ID:** 2606.19399 | [PDF](https://arxiv.org/pdf/2606.19399v1)

**作者:** Manish Acharya `[一作]` (Vanderbilt University), Yifan Zhang `[通讯]` (Vanderbilt University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VERITAS框架，在零样本环境下通过把Lean定理证明器的结构化反馈（语法错误、类型错误、目标进展、证明完成）直接反馈给生成模型，提升正式定理证明的成功率。

**💡 创新点**

创新点在于：①将检证器的中间信号完整地注入生成过程；②设计了两阶段（Best‑of‑N + Critic‑guided MCTS）协议并保证单个实例的单调性；③采用批量Lean验证显著降低验证成本；④发布了55道组合数学基准。

**🔧 技术方法**

主要技术包括大语言模型（Claude Sonnet/Haiku）生成和评价策略；检索器从Mathlib提取前置命题；Critic-guided MCTS改进UCB搜索；批量Lean验证和策略指导的提示工程。

**📊 数据集**

使用的基准数据集为 miniF2F（244道数学定理）和自制 VERITAS‑CombiBench（55道组合数学定理）。

**📈 对比分析**

与传统的最佳采样（Best‑of‑5）、手工策略组合（Portfolio）以及无LLM或无搜索的消融实验相比，VERITAS 在 miniF2F 上取得 40.6% 的成功率（高于 36.9% 的 Best‑of‑5），在 CombiBench 上获得 7.3%（高于 1.8% 的 Best‑of‑5）。

**⚠️ 局限性**

局限性包括：仍无法解决更高难度的 IMO 级创意定理；依赖于确定性检证器的结构化信号；需要大量计算资源和昂贵的 API 调用；检索器和 Critic 的性能仍有提升空间。

---

## 7. Techniques for Peak Memory Reduction for LoRA Fine-tuning of LLMs on Edge Devices

**arXiv ID:** 2606.19528 | [PDF](https://arxiv.org/pdf/2606.19528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 8. The Sheaf Laplacian: A Topological Framework for Data Fusion and Consensus in Distributed Sensing Networks

**arXiv ID:** 2606.19529 | [PDF](https://arxiv.org/pdf/2606.19529v1)

**作者:** Manuel Hernández `[一作]`, Eduardo Sánchez-Soto `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出将 sheaf Laplacian 作为分布式数据融合与共识的新数学工具，替代传统图拉普拉斯。

**💡 创新点**

创新点在于使用 sheaf 结构自然建模高维异构数据与复杂关系，扩展了拉普拉斯算子在多模态网络上的适用性。

**🔧 技术方法**

运用了拓扑学中的 sheaf 理论、线性代数的拉普拉斯与余边算子、图信号处理中的谱分析与能量度量等技术。

**📊 数据集**

未给出具体实验数据集，论文主要通过理论推导与示例说明。

**📈 对比分析**

与经典的 gossip 算法和 Kalman 滤波器进行概念性对比，指出在异构性、非标量共识与复杂关系建模上的优势，但未在实验中量化性能。

**⚠️ 局限性**

主要局限包括理论与实现的高复杂度、需要手工或学习定义 restriction maps、计算量大等。

---

## 9. AgentArmor: A Framework, Evaluation, \& Mitigation of Coding Agent Failures

**arXiv ID:** 2606.19380 | [PDF](https://arxiv.org/pdf/2606.19380v1)

**作者:** Kenneth Ge `[一作]` (Anthropic Fellows Program), Andre Assis `[通讯]` (Constellation)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究AI编程代理的安全失效机制，并提出AgentArmor干预方案

**💡 创新点**

将失效拆分为underspecification、capability和harness三类，并设计三击策略、命令分类器、确定性护栏及自我编辑工具

**🔧 技术方法**

基于OpenCode harness、两阶段命令风险分类器、immutability daemon、工具截断、系统提示扩展等技术

**📊 数据集**

使用20个编码环境、59个合成转录模板、8种真实失效启发场景，评测Claude Opus 4.6、GPT 5.4、Gemini 3.1 Pro等模型

**📈 对比分析**

采用500+样本的对比实验，AgentArmor显著降低违规率，且在统计上显著优于基线

**⚠️ 局限性**

仅评估编辑与部署模式，未覆盖绿色与监控，OpenCode harness已改动，缺乏真实部署数据，样本规模仍有限

---

## 10. PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models

**arXiv ID:** 2606.19534 | [PDF](https://arxiv.org/pdf/2606.19534v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 11. Neural Architectures as Functional Priors in Physics-Informed Control Problems

**arXiv ID:** 2606.19368 | [PDF](https://arxiv.org/pdf/2606.19368v1)

**作者:** Sonia Rubio Herranz `[一作]` (Universidad Complutense de Madrid), Antonio López Montes `[通讯]` (Universidad Complutense de Madrid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了神经网络架构在受控常微分方程（RLC电路和Duffing振子）中的隐式函数先验作用，比较了不同网络架构对学习控制策略的影响。

**💡 创新点**

发现架构相关的功能专化现象：傅里叶基架构倾向产生高频丰富、能量更高的控制信号，而传统多层感知器（MLP）生成更平滑、能量更低的控制，说明网络结构会隐式选择不同的可行控制解。

**🔧 技术方法**

采用Physics‑Informed Neural Networks（PINN），分别用MLP和FourierKAN两种架构构建状态网络与控制网络，利用ODE残差、边界条件、能量与光滑正则化构造损失函数，并通过自动微分进行训练。

**📊 数据集**

使用模拟数据：线性RLC系统（L=1,R=0.4,C=1,T=8）与非线性Duffing系统（δ,α,β等参数已设定），无公开数据集。

**📈 对比分析**

对比方法：在相同物理约束、损失权重、训练参数下，分别计算终点误差、控制能量E(u)、光滑度S(u)以及频谱重心。结果显示所有架构都能实现极低的终点误差，但能量、光滑度和频谱结构显著不同；傅里叶基网络能量较高、频谱重心偏高；MLP能量低、光滑度好。

**⚠️ 局限性**

局限性：仅在二维ODE模型上验证，架构种类有限；缺乏多次随机初始化、不同超参数的统计分析；结果可能不直接推广到高维、PDE或更复杂控制问题。

---

## 12. Emergent Alignment

**arXiv ID:** 2606.19527 | [PDF](https://arxiv.org/pdf/2606.19527v1)

**作者:** Martin Kolář `[一作]` `[通讯]` (Czech Technical University in Prague), Martin Kolář (Czech Technical University in Prague)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在线自监督框架Emergent Alignment，利用模型自身的“良心”步骤在训练循环中持续评估并纠正伦理偏差。

**💡 创新点**

创新点在于将自我审查与双目标损失（SFT+DPO）相结合，实现不依赖外部判别器即可实现自我对齐，并在各种场景中保持对齐。

**🔧 技术方法**

采用双目标损失ℒ_Hybrid，结合监督式微调（SFT）和直接偏好优化（DPO），并在推理时使用冻结的参考模型进行对比。

**📊 数据集**

主要使用公开LLM数据集（如qwen3-4b instruct、代码破解任务数据），并利用Qwen3-30b-a30b作为评判者进行对齐评估。

**📈 对比分析**

与代表性对齐方法（Representation Engineering、Inoculation Prompting、Honest Confessions、Constitutional AI）比较，Emergent Alignment在对齐得分上达到约91，优于其他方法且对模型能力几乎无影响。

**⚠️ 局限性**

局限性包括无法在“睡眠代理”模式下提前检测潜在失误，仅在激活后才触发对齐；对高复杂伦理冲突仍需更大判别模型或人工介入。

---

## 13. Secure Coding Drift in LLM-Assisted Post-Quantum Cryptography Development: A Gamified Fix

**arXiv ID:** 2606.19474 | [PDF](https://arxiv.org/pdf/2606.19474v1)

**作者:** R. D. N. Shakya `[一作]`, Nalin A. G. Arachchilage `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Secure Coding Drift in PQC（SCD-PQC）模型，并设计了基于游戏化的LLM辅助PQC开发框架

**💡 创新点**

将安全编码漂移从静态缺陷转为长期行为现象，并将LLM从被动代码生成器转变为主动安全共航员，首次将行为学与加密软件工程结合

**🔧 技术方法**

采用大语言模型（LLM）、规则式静态分析、LLM-as-a-Judge评估以及游戏化反馈机制

**📊 数据集**

未公开具体数据集，实验使用人工构造的PQc实现和内部测试数据

**📈 对比分析**

尚未完成实证比较，本文仅给出评估计划，预期能提升安全验证率、降低漂移速率

**⚠️ 局限性**

缺乏大规模实证验证，游戏化奖励易被规避，模型假设依赖LLM可信度和攻击场景的准确性

---

## 14. Tracking Representation Dynamics in Large Language Models with Persistent Homology

**arXiv ID:** 2606.19542 | [PDF](https://arxiv.org/pdf/2606.19542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 15. Physics-Informed Discovery of Yield Functions in Plasticity via Convex Neural Representations

**arXiv ID:** 2606.19375 | [PDF](https://arxiv.org/pdf/2606.19375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 16. Human-like autonomy emerges from self-play and a pinch of human data

**arXiv ID:** 2606.19370 | [PDF](https://arxiv.org/pdf/2606.19370v1)

**作者:** Daphne Cornelisse `[一作]` (New York University), Eugene Vinitsky `[通讯]` (New York University)

**通讯引用:** 1842 | [OpenAlex ID](https://openalex.org/A5015277482)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了一种将极少量人类驾驶示例作为正则化锚点，结合大规模自对弈强化学习，训练兼容人类规范的自动驾驶策略。

**💡 创新点**

创新点在于证明仅使用30分钟至3小时人类数据即可显著提升政策的安全性与人类相容性，而不需要繁琐的奖励工程或域随机化；并系统地量化了人类数据量对性能的影响。

**🔧 技术方法**

使用的技术包括PPO自对弈强化学习、行为克隆正则化（KL惩罚）、稀疏安全目标奖励，以及对大量模拟转移（约20 B步）的训练。

**📊 数据集**

数据集主要是Waymo Open Motion Dataset的日志（约30分钟人类轨迹用于行为克隆），并通过PufferDrive 2.0在这些轨迹基础上生成约60 年（20 B步）的模拟自对弈经验。

**📈 对比分析**

与无人类数据的自对弈、SMART‑tiny‑CLSFT等IL基准相比，所提方法在自对弈和人类重放评估中均取得更低的碰撞率（尤其是责任碰撞率降至约0.6–0.7%），并保持高任务完成率；同时在分布式真实性与碰撞严重度上也优于IL基准。

**⚠️ 局限性**

局限性包括：在极端协作场景下碰撞率仍可提升；评估基于仿真重放与IDM代理，真实道路迁移效果未知；对锚点策略的依赖与其熵等特性影响尚未完全理解。

---

## 17. Detecting Hallucinations for Large Language Model-based Knowledge Graph Reasoning

**arXiv ID:** 2606.19351 | [PDF](https://arxiv.org/pdf/2606.19351v1)

**作者:** Xinyan Zhu `[一作]` (Beijing University of Posts and Telecommunications), Chuan Shi `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LUCID方法，用于检测LLM驱动的知识图谱推理框架中的幻觉现象。

**💡 创新点**

首次将LLM注意力、KG语义相似度和KG结构信息三合一，用图神经网络整合进行幻觉检测。

**🔧 技术方法**

利用Transformer注意力矩阵、MiniLM嵌入、语义余弦相似度以及GINE图神经网络。

**📊 数据集**

在CWQ、WebQSP、GrailQA、QALD-10等公开KGQA数据集上进行评估。

**📈 对比分析**

与15种基线（通用与RAG特定）相比，LUCID在ACC、AUC、PCC和AVG指标上均实现SOTA，平均提升约6-8%。

**⚠️ 局限性**

对稀疏子图、跨语言/跨领域场景的鲁棒性仍有限，且依赖LLM内部注意力提取，导致对不同模型的适配性需进一步验证。

---

## 18. Physical Atari: A Robust and Accessible Platform for Real-time Reinforcement Learning on Robots

**arXiv ID:** 2606.19357 | [PDF](https://arxiv.org/pdf/2606.19357v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 19. Ensembles of Large Language Models for Identifying EQ-5D Studies in PubMed Based on Their Abstracts

**arXiv ID:** 2606.19345 | [PDF](https://arxiv.org/pdf/2606.19345v1)

**作者:** Zhyar Rzgar K. Rostam `[一作]` (Obuda University), Gábor Kertész `[通讯]` (Obuda University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了一种基于Google Gemini与Gemma大型语言模型的集成框架，用于从PubMed摘要自动检测是否报告EQ‑5D健康相关生活质量数据。

**💡 创新点**

创新之处在于将few‑shot提示、加权投票和Soft Stacking（逻辑回归元学习器）三步结合，显著提升了模型在该任务上的权重F1和整体准确率。

**🔧 技术方法**

使用的技术包括Google Gemini/Gemma LLM、few‑shot提示、权重加权投票、Soft Stacking（逻辑回归）、以及精度、召回率、权重F1等评估指标。

**📊 数据集**

使用的数据集为200篇手工标注的PubMed文献摘要，其中121篇正例（报告EQ‑5D）和79篇负例，来源于EuroQol搜索结果。

**📈 对比分析**

方法与单一LLM模型比较，集成模型（加权投票）得到权重F1 0.74、准确率 0.74，单模型最高为0.71；Soft Stacking模型得到权重F1 0.72、准确率 0.73，说明集成策略提升了性能。

**⚠️ 局限性**

限制在于数据集规模仅200条，且仅包含PubMed摘要，缺乏对更大语料和其他数据库的验证，可能影响模型的推广性。

---

## 20. Computational Identifiability

**arXiv ID:** 2606.19361 | [PDF](https://arxiv.org/pdf/2606.19361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 21. How LLMs Fail and Generalize in RTL Coding for Hardware Design?

**arXiv ID:** 2606.19347 | [PDF](https://arxiv.org/pdf/2606.19347v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 22. Quantifying Aleatoric Uncertainty of In-Context Learning for Robust Measure of LLM Prediction Confidence

**arXiv ID:** 2606.19353 | [PDF](https://arxiv.org/pdf/2606.19353v1)

**作者:** Jinseok Chung `[一作]` (POSTECH), Namhoon Lee `[通讯]` (POSTECH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出利用大语言模型内部关注头的自函数向量（self‑function vectors）来估计在上下文学习（ICL）中的天生不确定性（AU），并配套设计了首个针对ICL不确定性分解的评估协议。

**💡 创新点**

创新点在于：①把ICL视为隐式贝叶斯推断，利用内部表示直接估计AU；②构造自函数向量作为提示特定的隐含概念；③设计可独立调节AU与EU的合成任务及基于WordNet的多选题集，实现严谨的分解评估；④将分解结果用于幻觉检测等可信性任务。

**🔧 技术方法**

技术包括：贝叶斯视角的概率分解、因果间接效应分析选择显著关注头、对自函数向量的注入干预、熵与互信息计算、Spearman相关性评估。

**📊 数据集**

主要数据集：合成toy实验、WordNetMCQ1/2（单/多答案多选题）、AG News、Emotion、HellaSwag、GSM8K；模型实验基于LLaMA2（7B/13B/70B）、Qwen2.5‑7B、Mistral‑7B。

**📈 对比分析**

与基线方法（总熵、语义熵、UQ_ICL、MaxProb、Lookback Lens）对比，self‑function向量在AU/EU的控制任务上获得更高的Spearman相关性，且在幻觉检测任务中与或优于这些熵基准，显示出更可靠的分解效果。

**⚠️ 局限性**

局限性包括：①自函数向量仅为对后验的间接近似，真实后验结构的捕获程度待进一步理论验证；②需要针对不同模型调优干预层与关注头数量，限制了跨架构的通用性。

---

## 23. Improving Code-Switching ASR with Code-Mixing Guided Synthetic Speech

**arXiv ID:** 2606.19381 | [PDF](https://arxiv.org/pdf/2606.19381v1)

**作者:** Yue Heng Yeo `[一作]` (Nanyang Technological University), Eng Siong Chng `[通讯]` (Nanyang Technological University)

**通讯引用:** 7342 | [OpenAlex ID](https://openalex.org/A5070872826)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出基于代码切换指标CMI的多重评价DPO框架，利用伪帧语言标签和CMI_speech来对TTS生成的代码切换语音进行偏好学习，并将优化后的语音用于ASR数据增强。

**💡 创新点**

创新点在于：①将CMI扩展到音频层面并作为偏好信号；②在DPO中融合MER、UTMOS与Δ_CMI三重评价，形成对比式偏好；③利用TTS解码器的跨注意力生成伪帧语言标签，实现音频级语言对齐。

**🔧 技术方法**

使用技术包括：Direct Preference Optimization（DPO）偏好学习；伪帧语言标签与Language Alignment Loss（LAL）；CMI_speech和Δ_CMI指标；CosyVoice2多语言TTS模型；Whisper ASR与UTMOS MOS评估；MER评估。

**📊 数据集**

使用SEAME Mandarin‑English对话式代码切换语料库（约192小时）进行文本与语音的训练与评估，仅依赖该语料内部的数据。

**📈 对比分析**

在Whisper Large和CTC‑Conformer两种ASR架构下，分别使用真实数据、CosyVoice生成的数据、DPO（UTMOS+MER）以及DPO（UTMOS+MER+Δ_CMI）进行fine‑tune；结果显示加入Δ_CMI后，Whisper的DevMAN/DevSGE MER从12.1%/17.8%降至8.9%/14.2%，CTC‑Conformer从16.8%/23.6%降至15.4%/21.9%，显著提升。

**⚠️ 局限性**

局限性在于：实验仅在单一SEAME语料上验证；伪帧语言标签的准确性依赖于TTS解码器的注意力分布，可能存在误差；对其他语言对或更大规模语料的泛化性能尚未评估。

---

## 24. Where to Place the Query? Unveiling and Mitigating Positional Bias in In-Context Learning for Diffusion LLMs via Decoding Dynamics

**arXiv ID:** 2606.19349 | [PDF](https://arxiv.org/pdf/2606.19349v1)

**作者:** Zhengheng Li `[一作]` (Southeast University), Puzhi Xia `[通讯]` (Southeast University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了扩散式大语言模型（dLLMs）在上下文学习（ICL）中的查询位置敏感性，并提出了一种无训练、无标签的自适应查询路由方法 Auto-ICL。

**💡 创新点**

创新点在于揭示查询位置是 dLLMs ICL 的一阶变量，发现空间“最近效应”和时间“解码轨迹”造成的偏置，并设计基于全时序平均置信度（Average Confidence）的路由策略，实现了任务自适应的查询定位。

**🔧 技术方法**

主要技术包括扩散式语言模型（LLaDA‑8B‑Base、Dream‑7B‑Base）、注意力流（Attention Rollout）分析、解码轨迹可视化、平均置信度指标和 Auto-ICL 的三步路由算法。

**📊 数据集**

实验使用了五类数据集：顺序推理（GSM8K、MATH、MBPP）和全局感知（Sudoku、Countdown）来评估方法的通用性。

**📈 对比分析**

与传统的 Vanilla（尾部）、Prefix（前缀）、Random（随机）和 Oracle（基于标签的最优）基线相比，Auto-ICL 在大多数任务上匹配或超过最优静态位置，接近 Oracle 性能，且额外推理延迟仅为 0.08–0.09 秒。

**⚠️ 局限性**

局限性在于需对每个可能位置执行一次完整前向推理，导致推理成本提升，未来工作需开发轻量级早停预测或 Beam Search 以显著降低计算开销。

---

## 25. The Complexity of Auditing Disclosure-Robust Defeasible Explanations

**arXiv ID:** 2606.19401 | [PDF](https://arxiv.org/pdf/2606.19401v1)

**作者:** Haoyang Li `[一作]` `[通讯]` (University of Sydney), Haoyang Li (University of Sydney)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了“边界图谱”（boundary atlas），将符号化的可约束分类器编译成可在线查询的结构，并定义了“鲁棒核心”（robust core）以衡量在后续披露信息下仍保持预测一致的最小特征集合；

**💡 创新点**

其创新点在于首次将披露鲁棒性（disclosure‑robustness）引入解释性分析，并证明在接受记录下寻找最小鲁棒核心属于第二层多项式层次（Σ₂^p‑complete），同时将可读查询与优化查询在复杂度上完全区分；

**🔧 技术方法**

技术手段包括知识编译将决策树/规则列表转化为下覆盖面（lower‑cover faces）的锚点与否决器对（anchor/defeater）配对，使用多项式扫描完成预测、活跃原因与否决前沿的读取，及通过子集最小化与最小命中集求解鲁棒核心；

**📊 数据集**

实验数据集涵盖德国信用、成人/收入以及乳腺癌诊断等公开表格数据，采用二进制特征的“小型布尔立方”（k≤12）进行完整枚举；

**📈 对比分析**

在这些小立方实验中，鲁棒核心大小普遍在单个位数内，贪心近似几乎总能达到最优；相较于传统正则化解释，鲁棒核心在多数实例上与最小正例解释一致，且在恶意构造实例中表现出线性规模与贪心差距；

**⚠️ 局限性**

主要限制包括：编译生成的图谱可能指数级增大；仅在一位二值化特征下实验；鲁棒性仅相对给定披露空间，无法证明对更广泛披露的鲁棒性；以及最小鲁棒核心并非唯一，结果依赖确定性优先策略。

---

## 26. Supporting Design Decisions in Rule-Based Model Transformations

**arXiv ID:** 2606.19342 | [PDF](https://arxiv.org/pdf/2606.19342v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 27. Measuring Curriculum Alignment across Topical Coverage, Competency, and Cognitive Depth: A Longitudinal Framework Applied to CS2013 and CS2023

**arXiv ID:** 2606.19469 | [PDF](https://arxiv.org/pdf/2606.19469v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 28. REVEAL++: Differentiable Phenotypic Grouping for Vision-Language Retinal Modeling of Alzheimer's Disease Risk

**arXiv ID:** 2606.19522 | [PDF](https://arxiv.org/pdf/2606.19522v1)

**作者:** Ethan Elio Meidinger `[一作]` (University of Virginia), Ruogu Fang `[通讯]` (University of Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了REVEAL++框架，将视网膜图像与结构化临床风险文本通过可微分表型权重进行多模态对比学习，用以预测阿尔茨海默症风险。

**💡 创新点**

创新点在于将表型相似度离散化分组转为连续可微权重，使用软多正对比损失实现表型梯度监督，并实现端到端学习。

**🔧 技术方法**

使用了RETFound视觉编码器、GatorTron文本编码器、对比学习、soft‑weighted multi‑positive contrastive loss、sigmoid 门控以及可微分联合运算。

**📊 数据集**

基于英国生物样本库（UK Biobank）全眼底照片和结构化风险因子生成的临床文本，样本数约41,342（训练/验证/测试）。

**📈 对比分析**

与RETFound+GatorTron、RET‑CLIP、PMC‑CLIP、BiomedCLIP等基线对比，REVEAL++在AD发病预测中AUROC 0.678、Balanced Accuracy 0.613、F1 0.236、MCC 0.168，优于其他方法。

**⚠️ 局限性**

局限包括对可微分门控超参数的敏感性、对表型标签生成的依赖、未在多中心数据验证以及缺乏因果解释与临床可解释性。

---

## 29. Proprioceptive Invariant State Estimation for Humanoid Robots on Non-Inertial Ground

**arXiv ID:** 2606.19512 | [PDF](https://arxiv.org/pdf/2606.19512v1)

**作者:** Falak Mandali `[一作]` (Purdue University), Yan Gu `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种仅利用本体和脚部惯性测量单元的自适应不变扩展卡尔曼滤波器，用于在非惯性地面上实时估计类人机器人基座相对地面的位置和速度。

**💡 创新点**

创新点在于通过将脚部接触约束与不变EKF相结合，显式建模地面运动的非线性效应，并且在不需要外部地面传感器的情况下实现基座姿态与速度的可观测性。

**🔧 技术方法**

采用了Lie群不变扩展卡尔曼滤波、正向运动学、惯性测量单元融合与观测矩阵可观测性分析等技术。

**📊 数据集**

在阿吉利泰克（Agility Robotics）Digit类人机器人上，在模拟的摇摆和俯仰运动的移动跑步机上进行实验，利用Vicon运动捕捉系统进行真值标定。

**📈 对比分析**

与基线SRS（静态地面假设）和DRS（需要外部地面IMU）进行比较，结果显示该滤波器收敛速度提升约96%，位置误差降低约80%，最终稳态误差低于4cm。

**⚠️ 局限性**

局限性包括在单轴角速度缺失时基座位置可观测性受限，需要脚部不打滑且对大幅接触冲击敏感；基座姿态在单一IMU配置下不可观测，需额外的躯干IMU增强。

---

## 30. Protein Representation Learning with Secondary-Structure and Energy-Filtered Hydrogen-Bond Graphs

**arXiv ID:** 2606.19374 | [PDF](https://arxiv.org/pdf/2606.19374v1)

**作者:** Mohamed Mouhajir `[一作]` (UM6P), Dongqi Fu `[通讯]` (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一种基于图神经网络的蛋白质表示学习模型 SSProNet，利用二级结构标签和能量过滤的氢键边构建蛋白质图，并在节点中加入二级结构先验；

**💡 创新点**

创新点在于引入生物物理驱动的图拓扑（氢键+能量过滤）以及二级结构先验，保持 SE(3) 不变性，提升模型对蛋白质全局结构的感知和可解释性；

**🔧 技术方法**

采用 ProNet 的层次化 SE(3) 不变编码、边门控信息传递、DSSP 解析二级结构与氢键、能量阈值过滤、以及多尺度信息融合的 GNN 架构；

**📊 数据集**

使用 Fold 结构分类（约 16k 蛋白、1195 个 fold）、EC 反应分类（37k 蛋白、384 个 EC）、LBA 结合亲和力（3507 复合物 PDBbind）三大公开数据集；

**📈 对比分析**

与多种基线（GCN、IEConv、DWNN、GearNet、HoloProt、MACE、SEGNN、GVP‑GNN、ProNet、SCHull 等）对比，SSProNet 在 Fold、反应分类和 LBA 任务上均实现或超过 SOTA，Fold 准确率提升约 7%，反应分类达 88.3%，LBA Pearson/Spearman 分别为 0.613/0.616，表现显著但训练时长略高；

**⚠️ 局限性**

局限性包括依赖 DSSP 解析及氢键能量阈值，训练时间比传统近似更长，LBA RMSE 仍略低于最佳基线，弱氢键阈值对性能与速度的权衡需要进一步调优，且模型对缺失或低质量结构的适应性尚未充分验证。

---

## 31. Cost-Optimal LLM Routing with Limited User Feedback under User Satisfaction Guarantees

**arXiv ID:** 2606.19376 | [PDF](https://arxiv.org/pdf/2606.19376v1)

**作者:** Herbert Woisetschläger `[一作]` (Technical University of Munich), Shiqiang Wang `[通讯]` (University of Exeter)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了SLARouter，一个在线LLM路由器，利用稀疏单侧观测反馈学习成本最优路由并满足SLA

**💡 创新点**

①在稀疏单向反馈下实现在线学习和成本最优；②通过Lyapunov漂移+惩罚框架扩展，实现SLA合规和成本最优理论保证；③引入LLM judge补充反馈；④自适应V参数无需手工调优

**🔧 技术方法**

Lyapunov漂移+惩罚框架、在线随机梯度下降、预测器多标签分类器、虚拟队列、稀疏反馈补偿、正类重加权等技术

**📊 数据集**

Qwen 3.5模型族（2B、9B、35B、122B）与11个公开基准：ACPBench、ARC Challenge、ARC Easy、BoolQ、GSM8K、LAMBADA、MMLU、SciQ、GPQA、SocialIQa、WinoGrande

**📈 对比分析**

对比CARROT、Causal LLM Routing、MESS+；在所有基准上无需手调参即可满足SLA；平均运营成本比最佳基线低约2.22×，在高成本场景降低1.25×到2.5×；在大多数基准中获得最高满意率，且在稀疏反馈下保持约3%–5%的SLA误差

**⚠️ 局限性**

假设用户反馈率恒定且IID，无法处理反馈突发或用户相关性；对极度负偏反馈的鲁棒性待提升；需要LLM judge补充，仍需评估其误差

---

## 32. On Epimorphisms of Hypergraphic Automata and Input Symbol Semigroups

**arXiv ID:** 2606.19394 | [PDF](https://arxiv.org/pdf/2606.19394v1)

**作者:** Jasem Hamoud `[一作]` `[通讯]` (Moscow Institute of Physics and Technology), Jasem Hamoud (Moscow Institute of Physics and Technology)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文对通用超图自动机及其输入符号半群的满同态进行了完整的表征，提出了超图的弱满同态和强满同态的两个不同概念，并证明了这两种概念在重要的p*-超图子类中必然重合。

**💡 创新点**

创新点在于引入了超图的弱满同态和强满同态的概念，并证明了这两者在p*-超图中重合的必要性和充分性。

**🔧 技术方法**

使用了超图理论和代数结构的相关技术，特别是对超图同态的研究。

**📊 数据集**

使用了有效的p-可定义边的超图数据集，特别是投影平面和仿射平面。

**📈 对比分析**

通过比较不同的超图同态，证明了在特定条件下，三元组(f, , g)是通用超图自动机的满同态，并给出了必要和充分条件。

**⚠️ 局限性**

限制在于未对S(H_X, H_Y)上的同余结构及相应的商自动机进行研究，这一问题预计会比满同态问题更复杂。

---

## 33. cAPM: Continual AI-Assisted Pace-Mapping with Active Learning

**arXiv ID:** 2606.19373 | [PDF](https://arxiv.org/pdf/2606.19373v1)

**作者:** Dylan O'Hara `[一作]` (Rochester Institute of Technology), Linwei Wang `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1839 | [OpenAlex ID](https://openalex.org/A5101710700)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种结合主动学习与连续学习的AI辅助心室扑动（VT）定位框架（cAPM），能够在多任务、跨心脏几何与病理条件下持续积累并迁移知识，显著减少所需的起搏点数量；

**💡 创新点**

核心创新在于：①任务无关的神经网络占位模型将起搏位置映射到12导联心电图；②利用不确定性驱动的主动学习策略高效挑选信息量最大的起搏点；③采用两种连续学习策略——基于记忆缓冲的元学习（-Meta）与基于可增长集成的模型（-Ensemble），实现跨任务知识保留与迁移；

**🔧 技术方法**

技术包括深度多层感知器（MLP）占位模型、基于高通滤波和Savitzky–Golay平滑的噪声估计、期望改进（EI）采集函数、元经验重放（MER）与梯度对齐、以及动态权重化的模型集成与剪枝；

**📊 数据集**

使用EDGAR数据库中的两套人类心室模型，分别在5种组织状态（健康、四种大小与位置不同的梗死瘢痕）下生成高密度起搏点的12导联心电图仿真数据；

**📈 对比分析**

与基线BOATMAP（基于高斯过程的主动学习）对比，在三类任务序列（相同心脏与病理、相同心脏不同病理、不同心脏与病理）下，-Meta与-Ensemble平均分别需要约4–5个起搏点完成定位，显著低于BOATMAP的13–15个；同时定位误差降低至2–4 mm，失败率从BOATMAP的10%下降到<2%，显示更高的临床可行性；

**⚠️ 局限性**

主要局限包括：仅在有限的两套心室模型和5种病理条件上验证；未涉及真实VT重建的挑战；模型假设起搏点可自由选择，未考虑心脏运动、接触误差及临床决策影响；未来需在动物与人类真实VT案例中进一步验证。

---

## 34. Interactive Pareto navigation for deep multi-task learning

**arXiv ID:** 2606.19521 | [PDF](https://arxiv.org/pdf/2606.19521v1)

**作者:** Augustina C. Amakor `[一作]` (TU Dortmund), Sebastian Peitz `[通讯]` (TU Dortmund)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出交互式Pareto导航框架PPE，在深度多任务学习中根据决策者偏好主动探索Pareto前沿；

**💡 创新点**

通过将predictor–corrector连续方法与Krylov子空间相结合，避免显式Hessian计算，实现高维问题下实时偏好驱动的探索；

**🔧 技术方法**

使用predictor–corrector连续方法、MINRES求解线性系统、MGDA作为修正器、Krylov子空间技术以及自动微分；

**📊 数据集**

在toy问题、MultiMNIST和UCI Census Income三组数据集上验证；

**📈 对比分析**

与加权和（WS）和Pareto MTL进行对比，PPE在相同迭代量下获得更优的Pareto点且计算时间更低；

**⚠️ 局限性**

受限于局部最优收敛、对偏好设置的依赖以及高维Hessian近似的数值稳定性问题。

---

## 35. Performance Analysis and Optimization of 3D Generative Diffusion Models across GPU Architectures

**arXiv ID:** 2606.19365 | [PDF](https://arxiv.org/pdf/2606.19365v1)

**作者:** Jeeho Ryoo `[一作]` (Fairleigh Dickinson University), Byeong Kil Lee `[通讯]` (University of Colorado)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

对医学3D扩散模型Med-DDPM在V100、A100、H100三代NVIDIA GPU上进行系统级性能分析，并通过启用TF32 Tensor Core路径和3D channels-last内存布局两种架构感知优化显著提升计算效率。

**💡 创新点**

创新点在于：①首次用Nsight Compute深入剖析3D扩散模型的核级运行时、指令分布、内存访问与调度瓶颈；②提出并验证了基于Tensor Core的TF32加速与通道后置（channels-last）布局的组合优化，二者可显著提升Tensor Core利用率并降低SM周期与指令量；③系统性比较不同GPU架构下的微体系结构行为，揭示了内存布局转换和张量转换对性能的影响。

**🔧 技术方法**

使用的技术包括：PyTorch 2.0+（配合cuDNN 8.9）、Nsight Compute (2025.2.1) 进行核级与微架构级分析；启用TF32（torch.backends.cudnn.allow_tf32 = True、torch.backends.cuda.matmul.allow_tf32 = True）；将U-Net及数据张量改为NHWC（channels-last）布局；对比Baseline、OPT1（TF32）、OPT2（channels-last）和OPT12（两者结合）。

**📊 数据集**

主要使用医学3D MRI 数据集（脑部T1、T2、FLAIR扫描及其对应的分割掩码），但文中未给出具体公开数据集名称，推测为常用公开脑部MRI数据集（如ADNI或OASIS）或自建内部数据集。

**📈 对比分析**

比较方法：在相同超参数下分别在V100、A100、H100上跑Baseline、OPT1、OPT2、OPT12，收集核时长、指令数、IPC、Tensor Core利用率、内存带宽、L1/L2命中率等指标。结果显示：在A100/H100上，TF32可将SM周期降至0.18×/0.39×、指令数降至0.09×/0.18×、Tensor Core利用率提升至约10×，IPC保持1.0±0.1；channels-last在所有GPU上大幅压缩SM周期与指令数，但未能提升吞吐率，表现为高缓存命中率、低DRAM带宽、低算力利用。整体而言，TF32+channels-last组合（OPT12）在A100/H100上可获得≈3-5×的加速（取决于评测指标），而在V100上因缺乏TF32支持加速有限。

**⚠️ 局限性**

局限性包括：①channels-last布局导致大量内存转化与小型内核，未实现充分的Tensor Core利用，需进一步融合和调度优化；②仅在单卡单机上评测，未考虑多卡/分布式训练的扩展性；③实验未覆盖更大规模的数据集或不同模态，难以验证在多任务或异构场景下的稳健性；④V100缺乏TF32，无法在老旧GPU上实现相同级别加速；⑤未探究FP8/INT4等更低精度对生成质量与收敛的影响。

---

## 36. Sign-Language Datasets at Scale: A Comprehensive Survey on Resources, Benchmarks, and Annotation Standards

**arXiv ID:** 2606.19352 | [PDF](https://arxiv.org/pdf/2606.19352v1)

**作者:** Yiming Ni `[一作]` (University of Washington), Wei Cheng `[通讯]` (University of Washington)

**通讯引用:** 482490 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对120个手语数据集进行系统综述，分析数据集挑战并提出统一的24字段数据表模板与公开GitHub仓库，提供多任务（SLR、SLT、SLP）基准结果；

**💡 创新点**

首次统一覆盖35种手语的120个公开数据集，提出统一的数据表模板和开放式资源库，系统评估多任务基准；

**🔧 技术方法**

利用数据整理、可视化、评估指标（WER、BLEU）和多模态特征提取，进行基准实验并提供对比分析；

**📊 数据集**

120个公开手语数据集（Fingerspelling、ISLD、CSLD），涵盖35种手语，主要用于SLR、SLT和SLP任务；

**📈 对比分析**

采用WER和BLEU等标准指标在PHOENIX14T、CSL‑Daily、How2Sign、YouTube‑ASL、OpenASL等数据集上进行对比实验，结果显示高资源数据集性能优于低资源，任务间仍存在显著差距；

**⚠️ 局限性**

数据集不平衡、标注不一致、可访问性差、缺乏社区验证和人类评估、以及对低资源语言覆盖不足等限制

---

## 37. MortarBench: Evaluating Mortgage Loan Origination Agents

**arXiv ID:** 2606.19416 | [PDF](https://arxiv.org/pdf/2606.19416v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 38. FloatDoor: Platform-Triggered Backdoors in LLMs

**arXiv ID:** 2606.19535 | [PDF](https://arxiv.org/pdf/2606.19535v1)

**作者:** Nils Loose `[一作]` (University of Luebeck), Thomas Eisenbarth `[通讯]` (University of Luebeck)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于平台差异触发的 LLM 后门攻击，利用两阶段 LoRA 调优在模型中植入可在指定平台激活的恶意行为；

**💡 创新点**

创新点在于将浮点运算的跨平台差异转化为可学习的残差特征，构造触发器与任务适配器实现无输入模式、无模型改造的后门；

**🔧 技术方法**

技术方法包括：1）构造跨平台残差差异（CPRSD）并用线性探针放大其可分离性；2）利用两段 LoRA（trigger 与 task）分别实现平台指纹提取与平台条件输出；3）在推理时保持原始架构，仅修改权重；4）使用 KL 伪装、正则化保证模型主干性能；

**📊 数据集**

实验使用 Qwen3-4B 与 Qwen3-8B 两个开源模型；训练语料为多样化 Prompt 集合；攻击案例包含单词标记指纹与针对 NVIDIA A100 生成漏洞代码；平台覆盖 NVIDIA GPU、Google TPU、AWS Graviton、Alibaba Yitian-710 等；

**📈 对比分析**

对比基线模型，攻击后在 MMLU、HellaSwag 上性能变化 <1pp；在指纹任务中平台识别准确率>95%；在漏洞代码任务中，攻击成功率从基线 11.8% 提升至 49.0%，对非目标平台的误报仅 +3.9%；对抗干扰（高斯噪声、稀疏剪枝）后路由通道被彻底破坏，说明攻击高度依赖精细权重；

**⚠️ 局限性**

局限性包括：1）需获取目标平台和对照平台的推理环境来收集残差样本；2）对平台浮点差异依赖，若平台采用 FP32/FP16/混合精度或精确运算则效果下降；3）攻击对模型架构固定，仅通过 LoRA；4）在已部署防御（如 fp32 计算、权重正则化）下难以触发；5）未考虑攻击者通过动态调优或多平台训练以逃避现有防御的情况。

---

## 39. Predicting Mergeability of Parameter-Efficient Fine-Tuning Updates

**arXiv ID:** 2606.19549 | [PDF](https://arxiv.org/pdf/2606.19549v1)

**作者:** Lin Tang `[一作]` (Sichuan University), Yuxuan Wang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MergeProbe，一种在LoRA适配器训练初期就能预测合并可行性的轻量级方法；

**💡 创新点**

创新点在于把合并可行性建模为单任务效用与合并后保留度的组合，并通过训练早期的权重、梯度、子空间、Fisher与激活重叠等多维信号来预测，随后自动选择合并、重权、剪枝或路由决策；

**🔧 技术方法**

使用LoRA低秩适配、梯度与激活余弦、子空间重叠、Fisher权重、PCA、XGBoost回归+分类器，以及合并策略的成本敏感决策；

**📊 数据集**

在五个领域的 MERGE-PEFT 基准上测试：数学推理（MATH）、代码生成（HumanEval）、科学问答（Science QA）、通用指令遵循（Instr.）和安全/拒绝（Safety）等；

**📈 对比分析**

与直接平均、TIES、Fisher、LoRA‑LEGO、OSRM、FlyLoRA 等基线对比，MergeProbe 在平均保留度和最差任务保留度上均优于所有基线，且在保持低部署成本的前提下实现最高的整体保留率；

**⚠️ 局限性**

局限性包括仅针对LoRA更新和Transformer语言模型，需使用校准批次和轻量级训练监控；在适配器数量增大时集合级预测可能组合爆炸；合并可行性标签受所选合并算子和评估任务影响。

---

## 40. LEAP: Layer-skipping Efficiency via Adaptive Progression for Vision Transformer Distillation

**arXiv ID:** 2606.19483 | [PDF](https://arxiv.org/pdf/2606.19483v1)

**作者:** Jiaqi Zhang `[一作]` (Brown University), Randall Balestriero `[通讯]` (Brown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 LEAP 框架，通过自适应层跳进度学习实现 Vision Transformer 的特征蒸馏。

**💡 创新点**

创新点在于利用 CKA 相似度动态切换教师的中间特征作为蒸馏目标，消除手工层匹配并显著缩小教师-学生差距。

**🔧 技术方法**

采用 Vision Transformer、特征投影器、CKA 相似度评估、在线自适应课程与早停技术。

**📊 数据集**

实验使用 ImageNet‑100、ImageNet‑1K、ADE20K、Oxford/Paris 检索集及 ImageNet‑C 进行评估。

**📈 对比分析**

与基线、单层目标和密集一对一匹配对比，LEAP 在 ImageNet‑100 线性探针提升至 90.1%，ImageNet‑1K 检索 mAP 提升 3–8%，同时实现约 21% 的训练时间和 25% 的 FLOPs 节省。

**⚠️ 局限性**

局限性包括需要可访问教师的中间特征、CKA 阈值需手工调优，以及尚未验证跨架构或跨模态的蒸馏效果。

---

## 41. JustDiag!: A Diagnostic Justification Engine for Accountable Root Cause Analysis

**arXiv ID:** 2606.19407 | [PDF](https://arxiv.org/pdf/2606.19407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 42. Thermodynamic Signatures of Reasoning: Free-Energy and Spectral-Form-Factor Diagnostics for Hallucination Detection in Large Language Models

**arXiv ID:** 2606.19404 | [PDF](https://arxiv.org/pdf/2606.19404v1)

**作者:** Salim Khazem `[一作]` `[通讯]` (Talan), Salim Khazem (Talan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于LLM注意力拉普拉斯谱的“自由能签名”（Free‑Energy Signatures）特征，用于检测生成文本中的幻觉。

**💡 创新点**

创新点在于将注意力拉普拉斯矩阵视为量子哈密顿量，利用平衡热力学量（分区函数、自由能、谱熵、热容）与随机矩阵理论的谱形状因子（SFF）构建多尺度、连续的谱描述；同时证明其对注意力扰动的Lipschitz稳定性、对先前谱摘要的表达力覆盖以及无监督检测器的PAC‑style AUROC 上界。

**🔧 技术方法**

使用热力学函数、谱熵、热容、SFF等特征，并通过逻辑回归探针或无监督RMT偏差评分实现检测；特征提取涉及对每层注意力做对称化后拉普拉斯矩阵的稠密特征分解，计算分区函数、自由能、谱熵、热容、以及SFF；对长序列提供Lanczos/Chebyshev近似。

**📊 数据集**

在六款开源 LLM（Llama‑3‑8B、Llama‑3.1‑8B、Mistral‑7B、Qwen2.5‑7B、Gemma‑2‑9B、Phi‑3‑medium）与六个基准（TruthfulQA、HaluEval、TriviaQA、NQ‑Open、GSM8K、MATH‑500）上进行评估。

**📈 对比分析**

与现有注意力谱基线（LapEig、GoR‑4）以及无监督概率/熵基线（MSP、PPL⁻¹、Semantic Entropy 等）对比，平均AUROC达0.763，比LapEig提升6.5点、比GoR‑4提升2.4点；无监督RMT偏差评分平均AUROC为0.71；探针仅需数百标签即可逼近全量标签下的性能。

**⚠️ 局限性**

限制包括：需白盒访问每层post‑softmax注意力；对长序列的O(n³)稠密特征分解是计算瓶颈；在短文本/低token数时SFF估计噪声较大；无监督RMT偏差得分对任务有符号依赖，需每个任务校准；攻击者可通过优化生成来伪造Wigner–Dyson统计，降低无监督检测的鲁棒性。

---

## 43. Exposing the Unsaid: Visualizing Hidden LLM Bias through Stochastic Path Aggregation

**arXiv ID:** 2606.19344 | [PDF](https://arxiv.org/pdf/2606.19344v1)

**作者:** Matteo Pelossi `[一作]` (ETH Zurich), Mennatallah El-Assady `[通讯]` (ETH Zurich)

**通讯引用:** 2066 | [OpenAlex ID](https://openalex.org/A5020415668)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于聚合Sankey树的可视化分析框架，用于系统地审计大型语言模型（LLM）的偏见。

**💡 创新点**

创新点在于将随机生成路径聚合为结构化树并进行语义分类，利用对比推理（contrastive inference）展示不同语义上下文下的代币概率差异，从而揭示隐藏的偏见。

**🔧 技术方法**

采用系统化扰动（prompt augmentation + 语义本体注入）、温度采样、语法解析、结构聚类、辅助LLM语义分类、树结构构建和自定义Sankey可视化等技术。

**📊 数据集**

使用的评估数据集包括公开的性别与地理本体（如Female/Male names, Arab/Western locations），以及 GPT‑2 XL、Apertus‑8B‑Base、Apertus‑70B‑Instruct 等模型的生成结果。

**📈 对比分析**

通过对比不同本体生成的聚合树，利用对比推理计算反事实概率并绘制对比Sankey图，实验显示该方法能显著揭示模型在性别、种族、毒性等维度的系统性偏见；在用户研究中可用性得分高达 76.9。

**⚠️ 局限性**

主要限制包括高计算开销、对辅助LLM及本体生成的偏见依赖、温度越高导致聚类稀疏，以及缺乏对提示语微调的“蝴蝶效应”分析。

---

## 44. DevOps and General Developers: Insights from Stack Overflow's 2023 Survey

**arXiv ID:** 2606.19395 | [PDF](https://arxiv.org/pdf/2606.19395v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 45. Execution-bound advisory automation for agentic AI: a reproducible AIBOM-driven CSAF-VEX framework

**arXiv ID:** 2606.19390 | [PDF](https://arxiv.org/pdf/2606.19390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 46. Trustworthy Multi-Agent Systems: Mitigating Semantic Drift with the Argent Signaling Protocol

**arXiv ID:** 2606.19356 | [PDF](https://arxiv.org/pdf/2606.19356v1)

**作者:** Anantha Sharma `[一作]` `[通讯]` (Synechron), Anantha Sharma (Synechron)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多智能体LLM系统中引入 Argent Signaling Protocol（ASP），为每个回答附加置信度、依据、随机性和假设索引的结构化头部，帮助控制器区分可修复与不可修复的失败。

**💡 创新点**

创新点在于将这些质量信号以机器可读的方式嵌入回答中，实现跨代理通信的可审计与可治理，并通过漂移监测实现动态路由。

**🔧 技术方法**

使用了基于 token 重叠、引用得分和创新比例的手工权重公式计算信号，并结合 Jensen–Shannon 距离监测漂移，再依据阈值进行修复或容纳决策。

**📊 数据集**

数据集为 Array BioPharma/Ono 许可协议的 27 道文档驱动 QA 题目，检索采用 TF‑IDF 并划分约 220 词的块。

**📈 对比分析**

在单体控制器模式下，ASP 将通过率从 12/81 提升到 21/81；在两代理管道中，ASP 侧车将 100% 未依据的输出阻止至下游决策，显著抑制语义漂移。

**⚠️ 局限性**

局限包括信号权重和阈值手工调参、仅评测单一法律文件、缺乏跨模型通用性、未验证假设衰减与自适应漂移机制，以及依赖黑盒估计器。

---

## 47. Interpretable and Verifiable Hardware Generation with LLM-Driven Stepwise Refinement

**arXiv ID:** 2606.19387 | [PDF](https://arxiv.org/pdf/2606.19387v1)

**作者:** You Li `[一作]` (University of Texas at Austin), David Z. Pan `[通讯]` (University of Texas at Austin)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个结合大型语言模型与形式化精细化规则的硬件生成框架，能在每一步都提供可验证的RTL实现；

**💡 创新点**

创新点在于把LLM作为“决策代理”与预先定义的精细化规则配合，形成可证明的逐步精化过程，并构建了完整的自动化代理系统；

**🔧 技术方法**

使用Claude 4.6 LLM、LangGraph进行代理编排、Dafny进行形式化验证、以及自定义的硬件描述语言实现；

**📊 数据集**

使用VerilogEval V2基准套件（156个设计问题）作为评测数据集；

**📈 对比分析**

与Claude Opus 4.6和多代理系统VeriMaAS比较，Pass@1达到92.3%，Pass@10为87.2%，在保持高正确率的同时，Token和时间成本略高；

**⚠️ 局限性**

局限性包括对顺序设计的精化难度更大、对非功能性约束的处理依赖LLM、并需要人工干预来细化设计细节。

---

## 48. ProMUSE: Progressive Multi-modal Uncertainty-guided Staged Evidential Alzheimer Disease Classification

**arXiv ID:** 2606.19371 | [PDF](https://arxiv.org/pdf/2606.19371v1)

**作者:** Long Doan `[一作]` (Kennesaw State University), Chen Zhao `[通讯]` (Kennesaw State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种名为ProMUSE的分阶段多模态诊断框架，能在不完整模态下利用不确定性判断是否继续获取昂贵的MRI/PET影像，最终实现对阿尔茨海默病的高精度预测。

**💡 创新点**

创新点包括：①利用Dirichlet基础的主体逻辑模型对单模态产生置信与不确定性；②通过Dempster–Shafer理论融合多模态的信念与不确定性；③设计阈值驱动的分阶段获取策略，实现成本与精度的平衡；④采用混合交叉熵与KL正则化的证据学习损失，提升不确定性校准。

**🔧 技术方法**

技术手段包括：多层感知机处理临床特征；图神经网络（GraphSAGE）提取MRI/PET图结构特征；Softplus激活生成证据；Dirichlet分布与主体逻辑估计不确定性；Dempster–Shafer融合；阈值搜索算法与分阶段决策；混合交叉熵+KL损失进行训练。

**📊 数据集**

使用三个公开阿尔茨海默病数据集：ADNI、AIBL和OASIS3，涵盖CN、MCI和AD三种诊断标签，并分别在三种二分类任务（CN vs AD、CN vs MCI、MCI vs AD）上评估。

**📈 对比分析**

与传统单模态与多模态基线（KNN、SVM、LR、RF、NN、MOGONET、TMC等）比较。ProMUSE在CN vs AD和CN vs MCI任务中几乎与最佳模型同等或更优，平均准确率接近最高值；在MCI vs AD任务表现相对较弱。其最大优势在于平均可节省50–90% MRI/PET使用，患者每人可节省约2300–3500美元。

**⚠️ 局限性**

局限性：①在MCI vs AD任务上准确率仍低于部分基线；②阈值选择依赖数据驱动，可能在不同临床环境下需要重新调优；③实验仅基于公开数据集，尚未在真实临床工作流中验证；④多模态融合假设模态间信息独立，可能忽略某些跨模态交互。

---

## 49. SPINE: A Fault Injection Profiler for Quantized Neural Networks under Accumulated Faults

**arXiv ID:** 2606.19526 | [PDF](https://arxiv.org/pdf/2606.19526v1)

**作者:** Nathan Guimarães `[一作]` (Federal University of Rio Grande do Sul), Jose Rodrigo Azambuja `[通讯]` (Federal University of Rio Grande do Sul)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发并评估了一种基于GDB的累积权重位翻转注入框架（SPINE），用于在边缘微控制器上对量化神经网络进行逐层敏感性分析。

**💡 创新点**

①提出累计位翻转与误判界定的注入模型；②直接在目标二进制上注入，无需重训练或代码改动；③结合实际内存布局（Packed vs Unpacked）评估，揭示低位错误主导、内存打包导致可靠性下降。

**🔧 技术方法**

GNU 调试器（GDB）驱动的位翻转注入器、统计误判率与置信区间公式、层级隔离的注入循环、误判比较器、Survival Probability 曲线分析等技术。

**📊 数据集**

SAT-6 机载图像数据集（28×28四通道，零填充至32×32）。

**📈 对比分析**

通过对三种拓扑、三种量化精度（8-bit、4-bit unpacked、4-bit packed）共九个配置进行注入，直至每层达到100个错误。结果显示：第一层最脆弱，低位错误占主导；Packed 4-bit 内存布局失效率显著提升，Unpacked 更耐受；统计置信区间验证了实验代表性。

**⚠️ 局限性**

仅考虑累积位翻转，未涵盖多种故障模式；实验仅在Cortex‑M3芯片上进行，缺乏可移植性验证；未在真实辐射环境下进行实验，模型基于仿真。

---

## 50. Beyond the GUI Paradigm: Do Mobile Agents Need the Phone Screen?

**arXiv ID:** 2606.19388 | [PDF](https://arxiv.org/pdf/2606.19388v1)

**作者:** Li Gu `[一作]` (Mila), Yang Wang `[通讯]` (Mila)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过让移动代理仅使用命令行界面（CLI）完成任务，评估了三种前沿编码代理在AndroidWorld和MobileWorld基准上的表现，并提出了CLI-Advantage套件验证CLI相对于传统GUI的优势。

**💡 创新点**

创新点在于把CLI视作与GUI同等重要的移动代理交互范式，并系统性证明CLI代理在无移动特定后训练的情况下可与最先进的GUI代理竞争，甚至在专门设计的CLI任务中显著优于GUI。

**🔧 技术方法**

使用了大语言模型编码代理（Claude Code、Terminus‑2、mini‑swe‑agent）、ADB命令行工具、系统提示工程、通用工具包装器以及人工-LLM协作构建的Oracle解答。

**📊 数据集**

实验数据集包括公开的AndroidWorld、MobileWorld基准任务集，以及新构建的45条CLI‑Advantage任务模板。

**📈 对比分析**

比较方法采用统一的规则化验证器，测量任务成功率和平均步骤数；结果显示Claude Code在AndroidWorld和MobileWorld分别达71.8%/51.9%的成功率，均超过所有可复现的GUI基线，并在CLI‑Advantage套件中成功率提升约30%且步骤数减半。

**⚠️ 局限性**

主要局限在于高昂的模型API费用（约8k USD），CLI代理仅能访问终端可见状态，无法处理视觉密集或需要GUI流程的任务，且当前实现仍低于Oracle最高壁垒。

---

## 51. Disentangling Linguistic Relatedness from Task Alignment in Cross-Lingual Transfer

**arXiv ID:** 2606.19346 | [PDF](https://arxiv.org/pdf/2606.19346v1)

**作者:** Ahmed Haj Ahmed `[一作]` (Haverford College), Alvin Grissom `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在七种多语言LLM上使用阿拉伯方言数据进行微调，随后在Semitic与非Semitic语言上进行零样本阅读理解测试，探讨是否存在语系特异的跨语言迁移。

**💡 创新点**

创新点在于将脚本差异、语言族群、模型规模与架构等因素系统对照，并通过Chain‑of‑Thought推理消融验证微调提升主要来自任务格式对齐，而非语言知识迁移。

**🔧 技术方法**

技术方法包括LoRA微调、基于logprob的多选答案评分、Chain‑of‑Thought推理、回归分析控制基线准确率以及配对McNemar置信区间评估。

**📊 数据集**

使用的数据集为：阿拉伯六种方言共4800例的微调集；Belebele多语言多选阅读理解基准，每种语言100例用于零样本评测。

**📈 对比分析**

比较方法为零样本准确率与配对置信区间，结果显示弱基线模型（如GPT‑OSS‑120B、20B）在所有目标语言上均有显著提升（平均+30–35个百分点），而强基线模型提升有限；MoE模型提升最大，非Semitic语言同样受益，说明提升不依赖语言族群。

**⚠️ 局限性**

局限性包括仅评估单一多选阅读理解任务，未验证在其他任务（如生成、形态学）中的迁移效果；微调使用单一阿拉伯源语言且统一超参；数据规模有限；未分析MoE专家路由对迁移的影响。

---

## 52. Spectral DPPs via NEPv: A Scalable Continuous Relaxation of Determinantal MAP for Diversity-Aware Data Selection

**arXiv ID:** 2606.19411 | [PDF](https://arxiv.org/pdf/2606.19411v1)

**作者:** Richard Yi Da Xu `[一作]` `[通讯]` (Hong Kong Baptist University), Richard Yi Da Xu (Hong Kong Baptist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于Stiefel流形的连续DPP-MAP松弛，并推导出非线性特征值问题（NEPv）及其自洽场（SCF）迭代求解器，实现了低秩核的近线性扩展；同时给出了从子空间到离散子集的利用率采样+贪心逼近方案；

**💡 创新点**

首次将DPP-MAP映射到Stiefel流形，得到新的特征向量依赖的NEPv，伴随新的逆Gram矩阵扰动定理，证明了局部收敛并提供了显式的收敛速率；

**🔧 技术方法**

Stiefel流形优化、SCF迭代、NEPv理论、低秩Nyström/随机特征映射、利用率采样、块Lanczos/LOBPCG求特征向量；

**📊 数据集**

仅在本文中使用了四类合成二维点集（独立锚点、冗余集、均匀分布、网格簇多重），未使用真实数据集；

**📈 对比分析**

与软最大化（simplex）松弛和D-optimal设计松弛进行对比；在冗余和秩缺陷簇场景下，本方法获得更高的最小对偶距离和更大的L_S值；在简单或分离良好的场景下表现相当；

**⚠️ 局限性**

缺乏真实数据实验，收敛性仅局部保证；需要良好初始化和eigengap，无法保证全局最优；逼近步骤可能产生显著间隙；

---

## 53. Simulating Robotic Locomotion in Sand: Resistive Force Theory in an Open-Source Physics Engine

**arXiv ID:** 2606.19504 | [PDF](https://arxiv.org/pdf/2606.19504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 54. Characterizing Narrative Content in Web-scale LLM Pretraining Data

**arXiv ID:** 2606.19468 | [PDF](https://arxiv.org/pdf/2606.19468v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 55. OpenRath: Session-Centered Runtime State for Agent Systems

**arXiv ID:** 2606.19409 | [PDF](https://arxiv.org/pdf/2606.19409v1)

**作者:** Fukang Wen `[一作]`, Ruilin Xu `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

OpenRath 提出了将多智能体工作流中的所有运行时状态（对话、工具调用、内存、沙箱、分支等）统一封装为单一可传递、可分支、可合并、可审计的 Session 对象，并提供对应的工具边界、内存平面和持久化机制；

**💡 创新点**

其创新点在于将运行时状态视为第一类对象，使用 PyTorch 风格的统一接口（Session、Layer、Router、Placement、Memory Plane、Compose、Flow）实现跨角色、跨工具、跨沙箱的可插拔、可重放的工作流；

**🔧 技术方法**

主要技术包括：Session 对象作为运行时数据流；Layer 作为可复用的变换；Router 控制动态工作流选择；Placement 管理沙箱与后端执行；Memory Plane 记录可持久化的内存操作；以及 JSONL 线性化的 lineage 导出；

**📊 数据集**

该工作未依赖特定数据集，而是关注系统架构与实现细节，使用多样化的工具和后端（本地、OpenSandbox、MCP 等）进行实验验证；

**📈 对比分析**

本报告未给出定量性能对比，而是通过单元测试、侧写验证和可重放导出展示实现的确定性和可审计性；后续评估将关注并行分支调度、内存检索质量和任务级别基准；

**⚠️ 局限性**

局限包括：内存平面功能尚未完全实现并验证；对外部后端支持有限（如 OpenSandbox 仅可选）；未覆盖完整的模型推理性能评估；并且当前实现仅在单机环境下验证，尚未测试大规模分布式部署。

---

## 56. Lightweight Non-Line-of-Sight Channel Detection for ML-assisted Bluetooth Direction Finding

**arXiv ID:** 2606.19497 | [PDF](https://arxiv.org/pdf/2606.19497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 57. An alternative way of defining finite graphs

**arXiv ID:** 2606.19393 | [PDF](https://arxiv.org/pdf/2606.19393v1)

**作者:** Maxim Nazarov `[一作]` `[通讯]` (Moscow Institute of Electronic Technology), Maxim Nazarov (Moscow Institute of Electronic Technology)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的图线性符号I[G]，利用顶点与边的自动同构类索引构造完整图不变量，简化图同构判定与图的存储与操作；

**💡 创新点**

创新点在于将自动同构类的数值索引转化为符号表示，得到一个唯一且可直接比较的完整不变量，且定义了基于字符串处理的图操作与颜色化方法；

**🔧 技术方法**

采用最大码（maxi‑code）和邻接矩阵映射、递归构造符号的算法，结合自动同构类的线性排序与颜色化技术，实现了从普通图到I[G]的转换及其逆向重构；

**📊 数据集**

文中未给出具体实验数据集，主要通过标准图类（路径、环、完全图、完全二分图）以及小规模（≤7）图的示例演示方法；

**📈 对比分析**

同构判定可直接通过字符串比较实现O(n²)时间，构造I[G]仍与图同构问题等价；重构普通图亦为O(n²)，显著降低了后续同构比较的复杂度；

**⚠️ 局限性**

主要局限在于生成I[G]的算法仍是ISO-hard，面对大规模图时不可行；存储空间为O(n²)，且目前仅针对无向无环单纯图给出完整理论，缺乏对有向或多重图、超图的完整扩展与复杂度分析。

---

## 58. MonaVec: A Training-Free Embedded Vector Search Kernel for Edge and Offline AI Systems

**arXiv ID:** 2606.19458 | [PDF](https://arxiv.org/pdf/2606.19458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 59. Calibrating Generative Models to Feature Distributions with MMD Finetuning

**arXiv ID:** 2606.19496 | [PDF](https://arxiv.org/pdf/2606.19496v1)

**作者:** Nathaniel L. Diamant `[一作]` (Stanford University), Brian L. Trippe `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对预训练生成模型进行微调，使其在特定特征空间（如化学指纹、蛋白二级结构比例、DNA活性峰值）上与目标分布一致。

**💡 创新点**

提出基于最大均值差异（MMD）的无偏得分函数梯度估计与留一基线，实现对黑盒、非可微特征的分布匹配，并加入 KL 正则化控制与预训练模型的偏差。

**🔧 技术方法**

使用 MMD 损失、留一基线的无偏得分函数梯度、KL 正则化；同时在实验中对不同核（Tanimoto、能量距离、Jaccard）做选择。

**📊 数据集**

实验数据集包括 174 个抗生素分子、CATH 蛋白域二级结构分布、DeepMEL2 人类黑色素瘤增强子与 AlphaGenome 预测的细胞类型活性分布。

**📈 对比分析**

与直接微调、CGM-relax（均值匹配）等基线比较；在抗生素实验中实现更低的指纹 MMD 与更高的分子有效率；在蛋白实验中在对称 KL 与 MMD 之间获得更好的平衡；在 DNA 生成任务中提升细胞类型特定活性分布匹配；总体性能在多任务上均优于基线。

**⚠️ 局限性**

MMD 对高维特征的检验功效下降；方法要求生成模型具备可计算采样轨迹对数概率；目前仅针对已知目标特征样本，尚未直接支持仅凭实验测量的约束。

---

## 60. Algebraic Dead Directions in LayerNorm Transformers: A Forward-Pass-Only Diagnostic at LLM Scale

**arXiv ID:** 2606.19491 | [PDF](https://arxiv.org/pdf/2606.19491v1)

**作者:** Tejas Pradeep Shirodkar `[一作]` (International Institute of Information Technology), P. J. Narayanan `[通讯]` (International Institute of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并验证了一种可以直接从LayerNorm参数读取的“死亡方向”，即归一化尺度的逆向量 γ⁻¹/‖γ⁻¹‖。该方向在预训练Transformer的随机初始化时即为后置LayerNorm中心化激活协方差的零空间，并在训练后成为深度“死亡”方向之一。作者通过单前向传递和FP64协方差矩阵，能够在不使用梯度、采样或大量前向后向步骤的情况下验证这一性质。

**💡 创新点**

创新点包括：
• 通过层归一化的均值投影矩阵推导出闭式死亡方向，首次提供参数可读的死向量；
• 设计了“Schur比”诊断，量化训练后死亡方向深度的增大；
• 推导并验证残差流最小奇异值的深度不变性（在大多数模型中保持不下降）；
• 建立LayerNorm与RMSNorm的代数二分（LN存在死亡方向，RMSNorm不存在），可直接通过参数分类。

**🔧 技术方法**

使用的技术包括：
• 奇异学习理论与Fisher信息矩阵的KFAC分解；
• 协方差矩阵的FP64单前向求解与SVD，提取最小奇异值与方向；
• Schur补与Marchenko–Pastur理论，用于计算Schur比；
• 统计对比与余弦相似度测量，评估方向匹配。

**📊 数据集**

使用的数据集与模型：
• 14个预训练Transformer（160 M–35 B参数），覆盖语言、图像、视频、混合模态；
• 语言模型使用WikiText‑103校准；
• 视觉模型使用ImageNet校准；
• 对每个模型在其自身输入分布上采样，构造协方差。

**📈 对比分析**

比较方法与结果：
• 对9个LayerNorm模型，在随机初始化时，方向余弦 |cos(u*, γ⁻¹/‖γ⁻¹‖)| ≥ 0.9999，平均 0.99999；
• 对5个RMSNorm模型，余弦均值 ≈ 0.015，近似随机；
• Schur比 Δγ 在随机初始化时约 0–0.3，训练后趋近 1；
• 残差流最小奇异值在 13/14 模型中保持非下降；
• 诊断开销仅为一次前向传递与一次FP64协方差分解，显著低于传统采样或梯度方法。

**⚠️ 局限性**

局限性：
• 仅在LayerNorm与RMSNorm两种归一化上验证；BatchNorm、GroupNorm 等尚未验证；
• Gemma 4‑31B 在多模态校准下出现真实死亡方向，但其根源未明；
• 需要FP64协方差以避免FP32噪声阈值问题；
• 该方法只给出死方向识别，无法直接解释训练动态或提升性能。

---

## 61. A Topos-Theoretic Interpretation of Blockchain Systems: Sheaves of Consensus and the Logic of Decentralized Truth

**arXiv ID:** 2606.19519 | [PDF](https://arxiv.org/pdf/2606.19519v1)

**作者:** Manuel Hernández `[一作]`, Eduardo Sánchez-Soto `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出以拓扑范畴论为基础的区块链共识建模框架，将共识过程表述为层化（sheafification），并利用Grothendieck拓扑、子对象分类器和Yoneda引理阐释分布式系统的内部逻辑与全局状态的关系。

**💡 创新点**

创新点：①将共识视为Sheaf条件的满足，构造了“计算时空”site (C,J)；②使用子对象分类器Ω揭示区块链内部逻辑为直觉主义逻辑，说明排中律失效与网络不确定性的对应；③通过Yoneda引理将全球状态的本体确定性与局部视图的可观测性对比，解释了概率最终性的理论根源。

**🔧 技术方法**

技术手段：拓扑范畴论、Sheaf理论、Grothendieck拓扑、内部逻辑、Yoneda引理、层化（sheafification）与子对象分类器分析。

**📊 数据集**

未使用具体数据集，论文为纯理论推导与形式化建模。

**📈 对比分析**

论文未给出实验比较，主要通过逻辑和几何对应关系说明模型优势；若在后续工作中与传统FSM模型或已有共识算法对比，期望在形式化验证、逻辑可解释性上优于传统方法。

**⚠️ 局限性**

局限性：①缺乏实现细节与可扩展性验证；②未给出实际网络节点有限视野下的算法实现；③对真实区块链网络中节点的计算能力与信息交互模型未做深入评估；④理论框架在大规模系统中的可操作性和性能评估仍待实验验证。

---

## 62. LooseControlVideo: Directorial Video Control using Spatial Blocking

**arXiv ID:** 2606.19495 | [PDF](https://arxiv.org/pdf/2606.19495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 63. Zero-Inflated Gaussian Distributions Enable Parameter-Space Sparsity in Estimation-of-Distribution Algorithms

**arXiv ID:** 2606.19369 | [PDF](https://arxiv.org/pdf/2606.19369v1)

**作者:** Andreas Faust `[一作]` (University of Freiburg), Juergen Becker `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种利用零膨胀高斯分布（ZIG）作为采样法的估计分布算法（EDA），实现对稀疏黑盒优化问题的无偏搜索；

**💡 创新点**

创新点在于将稀疏性与连续值的联合依赖建模为两层潜在高斯变量，并通过ICIN假设实现模型可识别；同时开发了基于逆映射与归一化的稀疏估计器，避免了传统手工稀疏算子和双层设计；

**🔧 技术方法**

采用零膨胀高斯模型、ICIN假设、积分逆映射、对抗训练的MLP逆映射、协方差矩阵投影以及基于精英重放的EDA框架；

**📊 数据集**

在二维的“月球着陆机”Gymnasium基准任务中，用90维二次控制器参数进行实验；

**📈 对比分析**

与密集Gaussian EDA、手工稀疏进化算法以及自定义稀疏EDAs进行比较；结果显示ZIG‑EDA在100代内平均收敛至约308的回报，比其它方法提前约1/2代完成，并在每个运行中仅使用约12个活跃参数，表现最优；

**⚠️ 局限性**

缺点包括：未给出理论收敛或样本效率分析；对极稀疏维度的相关性恢复性能下降；模型对高维强相关情形的估计噪声较大；需要进一步的结构化或正则化以提升可扩展性。

---

## 64. Spectral Retrieval-Augmented Time-Series Forecasting

**arXiv ID:** 2606.19412 | [PDF](https://arxiv.org/pdf/2606.19412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 65. Granularity-Regulated Adaptive Computational Efficiency for Optimal Verification in Test-Time Scaling

**arXiv ID:** 2606.19354 | [PDF](https://arxiv.org/pdf/2606.19354v1)

**作者:** Ardit Krasniqi `[一作]` (European University of Tirana), Elira Dervishi `[通讯]` (European University of Tirana)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GRACE框架，用于理论分析和实践验证在测试时计算（TTS）中验证器细粒度与计算预算的最优关系，并基于此设计了自适应Granularity策略。

**💡 创新点**

核心创新在于统一的理论模型揭示验证细粒度的相位转移（粗细梯度随预算与难度变化而转变），给出了闭式阈值与计算-性能前沿，并基于此提出了能实现前沿的自适应算法。

**🔧 技术方法**

使用了计算-性能函数建模、闭式阈值推导、Log‑concavity与Monotonicity的理论证明，以及基于难度估计的GRACE‑Adapt自适应算法；实验中对比了多种固定细粒度方法。

**📊 数据集**

在三大数学推理基准上进行实验：MATH‑500、GSM8K 和 AIME。

**📈 对比分析**

与固定细粒度（ORM、PRM、Beam Search 等）以及其他基线（Self‑Consistency、AURORA 等）在相同计算预算下对比，GRACE‑Adapt 在所有基准上均实现了提升，最大提升达 3.4%（AIME）并在难度高的任务中表现尤为突出。

**⚠️ 局限性**

局限性包括：假设验证成本可分离且验证精度随细粒度单调；对多模态、非可分离成本或高度不确定的难度估计的适用性尚未验证；在极端预算或超难任务下的稳健性仍需进一步研究。

---

## 66. A Hybrid GNN-FEM Framework for Phase-Field Fracture Simulation. Physics-Preserving Hybridization for Generalizable Surrogate Modeling

**arXiv ID:** 2606.19378 | [PDF](https://arxiv.org/pdf/2606.19378v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 67. DeXposure-Claw: An Agentic System for DeFi Risk Supervision

**arXiv ID:** 2606.19501 | [PDF](https://arxiv.org/pdf/2606.19501v1)

**作者:** Aijie Shu `[一作]` (University of Edinburgh), Fengxiang He `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种将图时间序列预测与LLM决策相结合的去中心化金融风险监控系统，在此系统中LLM仅基于结构化的预测证据来生成可审计的监管建议。

**💡 创新点**

创新点在于将LLM从直接对原始链上交易推理迁移到“预测‑证据‑决策”管道，结合数据健康与置信度门控，显著提升覆盖率与可审计性，同时通过六轴评估框架提供监管对齐的错误干预测量。

**🔧 技术方法**

使用的核心技术包括图时间序列基础模型（如EvolveGCN）、确定性监测器与压力情景生成器、LLM（Claude Sonnet 4.6/Opus 4.7）进行票据草拟、数据健康与置信度门控以及六轴评估工具。

**📊 数据集**

实验数据来源于 DefiLlama 公开链上度量，构成约 4,300 个协议、24,300 种代币、4,370 万条曝光条目、283 周的快照（2020‑03‑2025‑08）。

**📈 对比分析**

与传统持久性规则、快照 LLM、单纯预测器等基线相比，系统在 F1 由 0.0076 提升至 0.0288（提升 31%），误干预率约 0.37，且在保持较低成本（Sonnet 4.6 约 5 倍更便宜）的同时实现了更高的票据质量。

**⚠️ 局限性**

局限性包括仅关注单一 DeFi 信贷曝光面向、每周更新频率不足以捕捉数小时级危机、以及对已知历史事件的预训练偏倚可能影响解释质量。

---

## 68. How Linear Is a Transformer Feed-Forward Block? Per-Block Linear Recoverability Is Learned, Not Architectural

**arXiv ID:** 2606.19379 | [PDF](https://arxiv.org/pdf/2606.19379v1)

**作者:** Stuart Whipp `[一作]` `[通讯]` (Independent Research), Stuart Whipp (Independent Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Transformer前馈网络（FFN）的非线性特性，通过将每个FFN视为输入激活到输出激活的逐位置映射，分解该映射为最佳线性近似和残差，并测量其线性可恢复性。

**💡 创新点**

提出了一种精确的每块FFN线性可恢复性度量，展示了线性可恢复性是学习到的特性，而非架构特性，并且残差并不是低阶乘法的。

**🔧 技术方法**

使用了闭式最小二乘法来计算线性近似，并通过低秩双线性探针来分析残差。

**📊 数据集**

使用了GPT-2、Pythia-160m和llama-160m模型的12个FFN块，数据集为WikiText-2。

**📈 对比分析**

与传统的线性基线比较，发现训练的线性基线在这些激活上可能严重低估线性可恢复性，且线性可恢复性在不同块之间差异显著，且与激活函数无关。

**⚠️ 局限性**

研究的局限性包括只在小规模模型和有限的语料库上进行，且残差探针仅为单一低秩双线性形式，可能无法捕捉更复杂的残差结构。

---

## 69. DynAMO:Dynamic Asset Management Orchestration via Topological Multi-Agent Scheduling

**arXiv ID:** 2606.19382 | [PDF](https://arxiv.org/pdf/2606.19382v1)

**作者:** Kanishk Kushwaha `[一作]` (Gati Shakti Vishwavidyalaya), Dhaval C. Patel `[通讯]` (IBM Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了 DynAMO，一种基于计划-执行架构的工业资产管理调度引擎，支持分阶段验证、并行执行与安全保证。

**💡 创新点**

在计划阶段强制模式化的 JSON 模式验证并将工作流图转为 DAG，实现工具感知的动态并行调度以及上下文剪枝减少推理延迟。

**🔧 技术方法**

Plan‑then‑Execute 结构、Schema‑constrained Planner、顶点化执行引擎、Token‑Budgeted Context Pruning、LLM（如 GPT‑4）推理以及多线程并行调度。

**📊 数据集**

AssetOpsBench 141 条工业查询，涵盖传感器数据检索、异常分析、故障模式映射及工单生成。

**📈 对比分析**

通过顺序与并行执行对比、工具 I/O 仪表化、并发压力测试、上下文大小与延迟评估、错误注入与可重复性测试；并行执行平均可节省 1.6× 延迟、LLM 推理占 90% 以上、上下文剪枝降低约 30% 推理时间。

**⚠️ 局限性**

未与外部框架进行对标、准确性评估有限、LLM 推理仍是瓶颈、并发时资源争用导致不稳定、仅在模拟环境验证，真实部署可能存在更多变动。

---

## 70. Closing the Social-Semantic Gap: SPSD for Edge-Based Prompt Compression in Cloud LLM Inference

**arXiv ID:** 2606.19364 | [PDF](https://arxiv.org/pdf/2606.19364v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 71. ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?

**arXiv ID:** 2606.19531 | [PDF](https://arxiv.org/pdf/2606.19531v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 72. Weibull Weight-Scale Parameter Evolution under AdamW Training Dynamics

**arXiv ID:** 2606.19367 | [PDF](https://arxiv.org/pdf/2606.19367v1)

**作者:** Tiexin Ding `[一作]` `[通讯]`, Tiexin Ding

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究AdamW训练过程中权重尺度λ的增长、过冲与放松机制，并通过三力分解（对齐力、注入力、衰减力）与Weibull分布框架建立桥梁。

**💡 创新点**

首次将AdamW平方范数动态拆解为三种物理力，并证明对齐力在上升阶段占主导，阐明λ的过冲-放松轨迹；同时提出基于样条位移的稀疏检查点恢复方法，实现无动量状态下的力恢复。

**🔧 技术方法**

采用AdamW分析、Weibull分布拟合、三力分解公式、闭环验证、样条插值恢复技术、跨架构与学习率/种子鲁棒性实验、对齐-衰减平衡分析。

**📊 数据集**

使用wikitext‑103（单域）、Pile（多域）以及公开的Pythia模型检查点（70M/160M/410M/1B）和自训练的Llama‑arch 70M。

**📈 对比分析**

通过自训练模型的真实动量验证三力分解，发现对齐力占比88–94%，跨种子、学习率和架构保持一致；样条方法在自训练验证中恢复对齐力精度92–94%，约为两倍于两点基线；对公开检查点的λ曲线重建高度一致，证明方法可在无动量的实际模型中使用。

**⚠️ 局限性**

局限在于三力分解仅在可获取动量的自训练模型中直接测量，公开模型需依赖插值恢复且误差未知；k‑locked传输类组件才能映射到λ，选择类组件需直接用RMS；数据依赖的峰值调节仅为探索性观察，需更系统的混合数据实验验证。

---

## 73. Bistable by Construction: Wall-Clock-Calibrated State Monitors Have No Moment-Detection Regime at Agent Cadence

**arXiv ID:** 2606.19386 | [PDF](https://arxiv.org/pdf/2606.19386v1)

**作者:** Manvendra Modgil `[一作]` `[通讯]` (Modint Intelligence), Manvendra Modgil (Modint Intelligence)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对自主代理运行时监视器的状态饱和陷阱进行审计与实验，纠正之前误用的衰减机制，并系统评估不同时间标度下的阈值触发行为。

**💡 创新点**

首次揭示监视器校准方式（样本时间 vs 壁钟时间）决定是否出现常数报警或失效，并证明这一现象是整个校准类的属性，而非具体模型。

**🔧 技术方法**

使用持续情感引擎、Leaky Integrator（壁钟校准与样本时间CUSUM）以及边缘检测触发器，并在实验中注入合成与真实间隔时间。

**📊 数据集**

采用SWE-bench-Verified 的 20 条调试轨迹（包括 5 个原始轨迹和 15 个按实例 id 选取的轨迹）以及 5 次真实代理运行的壁钟时序。

**📈 对比分析**

通过对不同 Δt（0~600s）进行预注册网格扫描，比较水平阈值触发器与边缘触发器的报警次数；水平触发器在 Δt≤1s 时始终报警，Δt≥60s 时不报警；边缘触发器始终仅触发 0–3 次；实验表明边缘触发器能逃避陷阱，但仍无法恢复人类干预时点的可靠性。

**⚠️ 局限性**

限制包括：仅使用单一模型与轨迹格式、合成时间间隔的理想化、真实时序样本有限、未覆盖更大部署延迟、未验证跨领域通用性、以及基于平均速率的标定规则对瞬时激增的预测不足。

---

## 74. 3D Scene Graphs: Open Challenges and Future Directions

**arXiv ID:** 2606.19383 | [PDF](https://arxiv.org/pdf/2606.19383v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 75. DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

**arXiv ID:** 2606.19348 | [PDF](https://arxiv.org/pdf/2606.19348v1)

**作者:** DeepSeek-AI `[一作]`, Zongqing Yao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一系列可处理百万级上下文长度的大模型（DeepSeek‑V4 系列），通过混合稀疏/压缩注意力、混合专家网络以及 Muon 优化器，实现了显著的计算效率提升，并在预训练、后训练（含多域专家蒸馏）与推理阶段完成全流程。

**💡 创新点**

核心创新包括：
• 结合 KV 压缩与稀疏索引的 hybrid‑attention（Lightning Indexer + Sparse Attention）以及更高压缩率的 Heavy‑Compression Attention；
• 引入 “Manifold‑Constrained Residual Mapping” 的 Hyper‑Connection 以增强数值稳定性；
• 使用 Muon 优化器替代 AdamW，配合混合 Newton‑Schulz 正交化与动态参数化；
• 通过 Fine‑Grained EP、TileLang 及 Host Codegen 实现通信与计算的深度重叠，提升 MoE 训练吞吐；
• 在后训练中采用全词表 On‑Policy Distillation 与 FP4 量化，兼顾高精度与推理加速；
• 为多任务/多域应用设计了 Interleaved Thinking 与 Quick Instruction 机制。

**🔧 技术方法**

技术栈包括：
• Transformer + Multi‑Token Prediction (MTP) + DeepSeekMoE；
• Hybrid‑Attention：KV 压缩、Lightning Indexer、Sparse/MQA、Grouped Projection；
• Muon 优化器（动量、Nesterov、RMS 重标定、Newton‑Schulz 迭代）；
• 精度与可复现性工具：TileLang DSL、Host Codegen、SMT‑Solver 形式化分析、FP4/QAT、混合 FP8/FP4 推理；
• 通信/并行框架：Fine‑Grained Expert Parallelism、ZeRO‑style Muon 并行、Context Parallelism 与两阶段通信；
• 后训练流水线：On‑Policy Distillation、FP4 QAT、Full‑Vocabulary KL、教师调度；
• 部署与弹性：预抢占式推理、WAL、DSec 沙箱、微 VM、容器层式加载。

**📊 数据集**

训练集：约 32T tokens，覆盖 Web、代码、数学、长文档、跨域领域数据，按 Token‑Splitting 与 FIM 策略打包；后训练专家领域分别为数学、编码、代理、指令跟随等。评测集：
• 知识/推理：AGIEval、MMLU‑Pro、GPQA、Simple‑QA、Chinese‑SimpleQA、OpenAI MRCR、CorpusQA 等；
• 编程/数学：BigCodeBench、HumanEval、GSM8K、MATH、IMObench、Apex、CF 内部竞赛；
• 长上下文：LongBench‑V2、OpenAI MRCR、CorpusQA；
• 代理：Terminal Bench 2.0、SWE‑Verified、SWE Multiling。

**📈 对比分析**

比较方法：统一内部评测框架，使用相同温度、上下文长度、token‑budget；与 DeepSeek‑V3、Gemini‑3.1‑Pro、Claude‑Sonnet‑4.5 等对标。性能结果：
• 在 1M‑token 长上下文任务上，V4‑Base 仅消耗 V3‑Base 27% FLOPs、10% KV；V4‑Pro 10% FLOPs、7% KV；
• 在知识/推理/代码/数学基准上，V4‑Base 超过 V3‑Base 并接近 Gemini‑3.1‑Pro，尤其在 long‑context、数学推理和编码通用任务上领先；
• 由于高效的 KV 与 FLOPs 设计，推理吞吐提升约 2–3×，而且显存占用降低至 1/5。

**⚠️ 局限性**

限制与挑战：
• 训练过程仍受 MoE 路由与激活 outlier 影响，需 Anticipatory Routing 与 SwiGLU clamping；
• 对极大参数规模（> 100B）与更长上下文的进一步扩展仍需更高算力与存储；
• 部署依赖复杂的混合并行与多级 KV 管理，非开源框架（TileLang、DSec）对一般研究者门槛较高；
• 在某些领域（如高风险决策、可解释性）与领先的专有模型仍存在 3–6 个月的性能差距；
• 量化（FP4）在极低精度硬件上仍需验证，部分 GPU 需要支持 FP8‑E4M3；
• 需要持续维护大规模多域专家与 On‑Policy Distillation 的资源与成本。

---

## 76. Pruning via Causal Attribution Preserves Reasoning Performance in Large Language Models

**arXiv ID:** 2606.19350 | [PDF](https://arxiv.org/pdf/2606.19350v1)

**作者:** Amogh Sheth `[一作]` (Edison Academy Magnet School), Yuhao Ge `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的因果归因剪枝方法 CAP，通过在校准集上对各注意力头进行遮蔽实验评估其对推理性能的因果影响，并将头级因果分数转化为权重级重要性，随后在全局稀疏约束下对权重进行细粒度剪枝。

**💡 创新点**

创新点在于：①将注意力头视为可单独进行因果干预的单位，直接测量其对任务损失的期望增量；②用中位数聚合提升因果估计稳健性；③将头级因果分数映射为权重级重要性因子，兼顾因果信息与权重幅度；④在多步推理任务上实现比传统相关性剪枝（如 Wanda）更优的性能。

**🔧 技术方法**

采用的技术包括：前向钩子实现头级遮蔽；在小规模校准集上多次子样本评估并取中位数；将因果分数正则化至 [1,10] 区间并与权重绝对值相乘得到权重重要性；全局阈值二分搜索得到目标稀疏率；无训练、无权重更新，完全为一次性剪枝。

**📊 数据集**

使用的校准与评估数据集为 GSM8K、StrategyQA、ARC‑Challenge（用于推理校准及最终评测），并在 WikiText‑2 上测得困惑度作为语言模型质量参考。

**📈 对比分析**

与基线 Wanda 进行对比（均为训练无关的一次性剪枝），在 10%–20% 稀疏率下，CAP 在 Llama‑3‑8B‑Instruct 的 ARC‑Challenge 上相对提升达 61%，在多模型/多任务组合中普遍优于 Wanda；在 50% 稀疏率下，CAP 的性能下降更为明显，尤其在 Llama‑3 上因 MLP 重要性估计粗糙导致崩溃。

**⚠️ 局限性**

主要限制包括：①高稀疏率（≥40%）时仅按头级因果评估，导致 MLP 权重被误剪造成模型崩溃；②仅评估最终答案准确率，未对中间推理步骤的完整性、逻辑连贯性等做细粒度分析；③只针对 Llama‑3 与 Mistral 进行实验，对 MoE 等动态路由模型适用性不足；④未与更强基线（如 SparseGPT）进行充分对比，限制了对极端稀疏场景的全面评估。

---

## 77. WorkBenchMark: A LEGO-Based Assembly Benchmark with an Assembly-by-Disassembly Baseline for the Smart Manufacturing League

**arXiv ID:** 2606.19358 | [PDF](https://arxiv.org/pdf/2606.19358v1)

**作者:** Wenbo Ma `[一作]` (RWTH Aachen University), Till Hofmann `[通讯]` (RWTH Aachen University)

**通讯引用:** 95 | [OpenAlex ID](https://openalex.org/A5014024166)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了WorkBenchMark——一个基于LEGO Duplo的机器人装配基准，包含400个不同难度层级的任务、对应的仿真环境以及完整的任务规范文件。

**💡 创新点**

创新点在于：①将机器人装配任务统一为可复现的基准，填补了先前各自独立任务设计的空白；②提出开源词汇感知 + Assembly‑by‑Disassembly（ABD）规划的集成流水线，实现了从自然语言任务描述到可执行的动作序列的端到端流程；③通过开放式语义检测（GroundingDINO + SAM）和3D姿态估计（FoundationPose）解决了“open‑world”感知问题。

**🔧 技术方法**

使用的主要技术包括：基于体素状态的ABD规划（递归拆解+抓取可达性与压入稳定性检查），开源语义检测框架（GroundingDINO + SAM）实现目标识别，FoundationPose进行精确6D姿态估计，MoveIt2+OMPL进行碰撞检测与轨迹规划，以及ROS 2实现抓取与放置的状态机。

**📊 数据集**

使用的数据集为WorkBenchMark 400个仿真任务（每个层级100个）以及40个在真实机器人上复现的任务，任务信息以YAML文件形式提供，涵盖从单砖堆叠到复杂互锁结构的不同难度。

**📈 对比分析**

通过与基于VLM/VLA的端到端策略（Gemini 2.5 Flash + 预训练VLA模型）进行对比，发现结构化管线在所有层级上表现更稳健：成功率在最高层级仍可达约67%，而VLM/VLA在第四层级仅约7%；执行准确率、规划时间和稳定性违约率也显著优于基线；此外，结构化管线的操控时间随任务复杂度增长更平滑。

**⚠️ 局限性**

局限性包括：①基准目前仅限于LEGO Duplo，未覆盖更复杂工业部件；②在真实环境中仅评估了基本抓取与放置动作，未涉及多手臂协作、动态障碍物或大尺度结构；③基准的任务配置仍依赖人工生成，可能在某些极端互锁或几何约束上不足；④相比纯学习方法，ABD规划仍需手工设计可达性与稳定性检测规则。

---

## 78. FlexLAM: Resolving the Bottleneck Trade-off in Latent Action Learning

**arXiv ID:** 2606.19408 | [PDF](https://arxiv.org/pdf/2606.19408v1)

**作者:** Takanori Yoshimoto `[一作]` (University of Tsukuba), Tatsuya Matsushima `[通讯]` (University of Tokyo)

**通讯引用:** 396 | [OpenAlex ID](https://openalex.org/A5083001889)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了 Latent Action Models (LAM) 的固定容量瓶颈问题，提出 FlexLAM 通过保留前缀训练实现可变长度潜在动作编码。

**💡 创新点**

创新点是利用嵌套丢弃（nested dropout）训练可变长度前缀有效码，使单一模型在不同令牌预算下均优于独立训练的固定容量模型。

**🔧 技术方法**

技术包括保留前缀训练、嵌套丢弃、潜在动作对齐、固定的因果潜在令牌评估器，以及对照实验。

**📊 数据集**

使用 DeepMind Lab 的仿真环境、Ego4D 实际视频数据，以及公开的视频数据集进行预训练。

**📈 对比分析**

在 DMLab 的多任务评估和 Ego4D 重建任务中，FlexLAM 在每个 token 预算下均超过 Fixed-K 基线，且在稀缺标签和窄源标签场景下更稳健。

**⚠️ 局限性**

局限性包括实验主要集中在 DMLab，真实视频对照为外部基线，且未评估最终可执行策略性能，需进一步验证在真实任务中的迁移效果。

---

## 79. Emyx: Fast and efficient all-atom protein generation

**arXiv ID:** 2606.19377 | [PDF](https://arxiv.org/pdf/2606.19377v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 80. DiffusionVS: A Generative Framework for Robust Visual Servoing Based on Diffusion Policy

**arXiv ID:** 2606.19397 | [PDF](https://arxiv.org/pdf/2606.19397v1)

**作者:** Hongkang Cui `[一作]`, Haoyao Chen `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于扩散策略的视觉伺服框架DiffusionVS，利用归一化像素坐标预测相机速度，实现平滑且鲁棒的视觉伺服；

**💡 创新点**

创新点包括：1）将扩散生成模型应用于视觉伺服，提供时序一致的动作序列；2）在线交互式训练机制，通过实时收集与回放数据提升模型泛化；3）将扩散模块无缝集成进现有回归网络，显著提升性能；

**🔧 技术方法**

技术手段主要为扩散政策（DDPM）网络、时间步嵌入、Mish激活、残差MLP结构、回放缓冲区的在线训练；

**📊 数据集**

使用了基于AprilTag的视觉标定板，在PyBullet仿真环境中随机采样相机姿态；在真实实验中使用AUBO-i5机械臂与眼内手RGB摄像头；

**📈 对比分析**

通过与传统回归视觉伺服（如GraphVS）以及无姿态信息的基线进行对比，DiffusionVS在仿真中成功率100%、平移误差0.54cm、旋转误差0.53°，真实机器人成功率93%，比回归基线显著提升；

**⚠️ 局限性**

局限性在于对大规模交互数据需求高，在线训练仍需较多时间；在极端噪声或大视角变化下，模型仍可能出现振荡；对非AprilTag目标的适应性尚待进一步验证。

---

## 81. Hidden Anchors in Multi-Agent LLM Deliberation

**arXiv ID:** 2606.19494 | [PDF](https://arxiv.org/pdf/2606.19494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 82. Insulin4RL: Real-Time Insulin Management in the Intensive Care Unit for Offline Reinforcement Learning

**arXiv ID:** 2606.19481 | [PDF](https://arxiv.org/pdf/2606.19481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 83. Can In-Context Learning Support Intrinsic Curiosity?

**arXiv ID:** 2606.19476 | [PDF](https://arxiv.org/pdf/2606.19476v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 84. Scaling Generative Foundation Models for Chest Radiography with Rectified Flow Transformers

**arXiv ID:** 2606.19460 | [PDF](https://arxiv.org/pdf/2606.19460v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 85. 3D-DLP: Self-Supervised 3D Object-Centric Scene Representation Learning

**arXiv ID:** 2606.19451 | [PDF](https://arxiv.org/pdf/2606.19451v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 86. Playful Agentic Robot Learning

**arXiv ID:** 2606.19419 | [PDF](https://arxiv.org/pdf/2606.19419v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 87. Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias

**arXiv ID:** 2606.19544 | [PDF](https://arxiv.org/pdf/2606.19544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 88. LLM-Mediated Human-AI Interaction in Search and Rescue: Impact of Expertise on Attentional Allocation

**arXiv ID:** 2606.19514 | [PDF](https://arxiv.org/pdf/2606.19514v1)

**作者:** Elahe Oveisi `[一作]` (Oklahoma State University), Hemanth Manjunatha `[通讯]` (Oklahoma State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在网格式搜索救援游戏中比较两种LLM辅助和无辅助条件，分析任务绩效、眼动与规划行为。

**💡 创新点**

首次将大型语言模型与实时自然语言指导结合，并通过眼动分析揭示其对注意力分配与认知负荷的影响。

**🔧 技术方法**

使用基于BFS的结构化状态表示的LLM（大语言模型）生成指导，配合Tobii眼动追踪和MiniGrid仿真环境。

**📊 数据集**

实验数据来自13名受试者在MiniGrid搜索救援任务中的行为记录与眼动数据。

**📈 对比分析**

对比方法采用线性混合效应模型，结果显示LLM显著提升任务效率（victims/step、总奖励），但未提升被救人数；眼动显示注意力转移至聊天界面并增加瞳孔波动。

**⚠️ 局限性**

局限包括样本量小、仅在模拟环境中评估、LLM可能产生错误输出时未测试，以及缺乏更丰富的生理测量。

---

## 89. LLM Doesn't Know What It Doesn't Know: Detecting Epistemic Blind Spots via Cross-Model Attribution Divergence on Clinical Tabular Data

**arXiv ID:** 2606.19509 | [PDF](https://arxiv.org/pdf/2606.19509v1)

**作者:** Akshat Dasula `[一作]` (Centific AI Research), Jaideep Srivastava `[通讯]` (University of Minnesota-Twin Cities)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在结构化临床表格数据上评估大型语言模型的推理与不确定性，并通过跨模型归因差异检测其认知盲点。

**💡 创新点**

创新点在于提出跨模型归因差异度量与外部校准器，揭示LLM缺乏自我评估并实现基于结构化模型的校准。

**🔧 技术方法**

使用跨模型归因差异（ADS）、SHAP特征注入、少量示例提示、对数回归/XGBoost校准器等技术。

**📊 数据集**

数据集为MIMIC‑IV的急性肾损伤（AKI）预测任务，共10k例，测试集300例。

**📈 对比分析**

与XGBoost对比，LLM在零样本时准确率仅49%，少样本+SHAP提升至75.3%，同时校准误差从0.254降至0.080。

**⚠️ 局限性**

局限包括单模型单任务、样本量有限、提示策略有限、特征归因可信度不高、未验证临床可用性。

---

## 90. Diffusion Language Models: An Experimental Analysis

**arXiv ID:** 2606.19475 | [PDF](https://arxiv.org/pdf/2606.19475v1)

**作者:** Thomas Bertolani `[一作]` (University of Modena and Reggio Emilia), Lorenzo Baraldi `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估并对比了现代扩散式语言模型（DLMs）与自回归模型的性能与计算效率

**💡 创新点**

提出统一的实验协议，系统分析推断时关键参数（denoising步数、上下文长度、块大小、并行掩码比例）对质量与成本的影响，并揭示不同架构在任务上的专长与折衷

**🔧 技术方法**

使用离散扩散、混合块式扩散与自回归技术，结合多步denoising、块级并行重采样和编码器-解码器结构

**📊 数据集**

评估数据集覆盖推理、编码、翻译、知识和结构化推理，包含MMLU、HellaSwag、GSM8K、HumanEval、MBPP、WMT16 En–De、Sudoku等

**📈 对比分析**

采用统一的lm-evaluation-harness框架对比，多模型在同一预算下的准确率与通用性；结果显示全序扩散模型在知识与全局约束任务上表现最好，块式扩散在推理与代码生成上更优；自回归基线仍在大规模语言生成上占优

**⚠️ 局限性**

限制在于扩散模型推断成本高，扩散步数与上下文长度的交互复杂，块式模型的块大小与掩码比例调优难度大，且不同任务对同一模型的适配性差异显著

---

## 91. ITNet: A Learnable Integral Transform That Subsumes Convolution, Attention, and Recurrence

**arXiv ID:** 2606.19538 | [PDF](https://arxiv.org/pdf/2606.19538v1)

**作者:** Ashim Dhor `[一作]` (Indian Institute of Science Education and Research Bhopal), Pin Yu Chen `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一的可学习积分变换网络（ITNet），将卷积、注意力和递归模型归结为同一可学习核的特殊情况。

**💡 创新点**

创新点在于把位置与特征共同编码的核作为可学习的交互函数，实现了通用的连续算子近似器，并通过可学习低秩分解、蒙特卡罗采样和块融合等技术实现可扩展。

**🔧 技术方法**

核心技术包括：可学习的二维MPL核（含随机傅里叶映射）、多头结构、残差连接、预归一化、Triton块融合、重要性加权Monte Carlo积分以及低秩核分解。

**📊 数据集**

在ImageNet-1K、GLUE、ModelNet40、VQA v2、NLVR2等标准数据集上进行实验。

**📈 对比分析**

与专门化的CNN、Transformer、SSM等基线进行比较，ITNet在同等规模下在所有任务上均匹配或超越基线，尤其在多模态和3D点云任务中表现突出。

**⚠️ 局限性**

局限性包括：大规模模型训练的稳定性与计算成本仍高，尚未在自回归生成任务中充分验证；多模态训练时耦合度高导致效率受限。

---

## 92. Mesh Inference: A Formal Model of Collective Intelligence Without a Center

**arXiv ID:** 2606.19537 | [PDF](https://arxiv.org/pdf/2606.19537v1)

**作者:** Hongwei Xu `[一作]` `[通讯]` (SYM.BOT), Hongwei Xu (SYM.BOT)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了一个去中心化、自治的网格推理（Mesh Inference）模型，证明在每个代理仅通过发布失真、已打标签的观测、且不共享权重或隐藏状态的前提下，整个网络可以收敛到唯一答案，并在满足特定政策时实现集体可识别性与观测唯一区别性。

**💡 创新点**

创新点在于：
1) 将能量最小化推理（如深度平衡模型、Hopfield网络）分布到自治节点并给出统一的“入射/发射”政策；
2) 通过一个单一的入射/发射政策，统一说明了收敛、识别完整性和观测唯一区别性三大性质；
3) 证明在线性高斯域下，该网格推理等价于中心化最优解；
4) 设计了内容地址化的谱系（lineage）作为唯一的全局侧信道，既保证了信息传递完整，又不泄露内部状态。

**🔧 技术方法**

主要技术包括：
- 能量基平衡推理（clamp‑and‑relax）
- 多代理自由能最小化与图信号去噪（Laplacian）
- M‑矩阵理论保证收敛与无发散
- 基于图连通性的识别完整性判定
- 内容地址化谱系实现源识别与无偏转发
- 异步 Jacobi/高斯‑赛德尔迭代保证在有延迟或节点离线时仍收敛
- 对比实验使用线性高斯合成数据、统一的随机图结构

**📊 数据集**

实验使用合成的线性高斯数据集：每个节点持有对隐藏变量的失真、秩不足的投影，联合观测可恢复完整状态；并在随机链、网格等图拓扑上验证收敛、精度、延迟与泄露。

**📈 对比分析**

与方法比较：
- 与中心化最优解（一次全局求解）相比，网格推理在精度上几乎无损（误差 < 0.1%）；
- 与孤立节点（无共享）相比，精度提升 27 倍；
- 延迟以 O(diam²) 迭代计数衡量，证明与图热扩散相符；
- 泄露度评估：单次查询泄露一个秩 ≤ p 投影，累计查询可完成模型反演；若查询集合的秩不足，则永远无法完全重构，体现了隐私双向性。

**⚠️ 局限性**

局限性：
1) 只在线性高斯域下严格证明，非线性关联（如自注意力、softmax）仍是开放问题；
2) 需要节点满足入射/发射政策，若有策略偏差或恶意阻断，识别完整性可能失效；
3) 需要足够的网络连通性（carrier‑graph connectivity），链路断开会导致精度退化到先验；
4) 隐私安全取决于查询预算与源新颖性，无法在任意查询下保证绝对安全；
5) 对真实数据集与大规模部署的实验尚未完成，性能与扩展性仍需进一步验证。

---

## 93. A Tool for the Synthesis of Adaptive Probabilistic Processors Based on the Ising Model

**arXiv ID:** 2606.19533 | [PDF](https://arxiv.org/pdf/2606.19533v1)

**作者:** Jonathan Juracy Carneiro da Silva `[一作]` (Federal University of Rio Grande do Sul), Jose Rodrigo Azambuja `[通讯]` (Federal University of Rio Grande do Sul)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一款针对Ising模型的自适应概率处理器合成工具，能够根据问题结构自动生成Hamiltonian、分配p-bit数量并选择最合适的更新算法。

**💡 创新点**

创新点在于整合了自动映射、资源自适应分配与动态算法选择，能够针对不同优化问题动态调整硬件资源和采样策略，而非固定配置。

**🔧 技术方法**

使用技术包括Ising映射、Gibbs采样、模拟退火、模拟量子退火与集群更新，以及基于结构特征的p-bit分配决策算法。

**📊 数据集**

实验使用了TSP、图着色、SAT、匹配、分割与最大割等多种图/约束优化基准实例，并将其映射为Ising形式。

**📈 对比分析**

通过与固定算法（Gibbs、SA、SQA、集群）对比，结果表明自适应选择在多类实例中均能提升收敛性能与解质量，尽管SQA获得最优能量但耗时更长，SA平衡性能最优。

**⚠️ 局限性**

局限性在于评估仅基于软件仿真，未验证在实际p-bit/MTJ硬件上的实现与能耗，且算法选择仍依赖预定义的结构特征阈值。

---

## 94. Does Text Actually Help? Uncovering and Resolving Text Collapse in Multimodal Time Series Forecasting

**arXiv ID:** 2606.19413 | [PDF](https://arxiv.org/pdf/2606.19413v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 95. A Categorial and Sheaf-Theoretic Semantics for Autonomic Component Ensembles

**arXiv ID:** 2606.19525 | [PDF](https://arxiv.org/pdf/2606.19525v1)

**作者:** Manuel Hernández `[一作]`, Eduardo Sánchez-Soto `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了基于范畴论与层析理论的SCEL系统新语义模型，将自主机器人社会建模为拓扑空间上的层析；

**💡 创新点**

创新点在于将信息共享与协作视为层析的粘合操作，系统故障和任务不可解性映射为层析协同障碍并可用层析上同调群定量化；

**🔧 技术方法**

采用范畴论（对称单张范畴、余极限等）、层析理论（层析与同调）、预层析/层析语义、并以实例演示；

**📊 数据集**

通过协同地形映射的案例研究（机器人集合{A,B,C}）演示模型，无使用公开数据集；

**📈 对比分析**

本文未进行量化性能对比，仅通过理论推导与案例展示证明方法可识别不可解任务；

**⚠️ 局限性**

局限包括：需要深厚的范畴/同调背景，模型对大规模系统的可扩展性未实验验证，且对实际机器人软件集成仍有技术壁垒。

---

## 96. Latent Confounded Causal Discovery via Lie Bracket Geometry

**arXiv ID:** 2606.19610 | [PDF](https://arxiv.org/pdf/2606.19610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 97. FAPO: Fully Autonomous Prompt Optimization of Multi-Step LLM Pipelines

**arXiv ID:** 2606.19605 | [PDF](https://arxiv.org/pdf/2606.19605v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 98. Ricci flow for the Bures--Helstrom qubit metric

**arXiv ID:** 2606.19493 | [PDF](https://arxiv.org/pdf/2606.19493v1)

**作者:** Andrew Lesniewski `[一作]` `[通讯]` (Baruch College), Andrew Lesniewski (Baruch College)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

对单比特状态空间内的Bures–Helstrom度量（最小单调Riemann度量）进行自洽的黎曼几何Ricci流研究，给出其在旋转对称坐标下的显式解析解，并讨论其在体积规范化下的稳定性与谱分解；

**💡 创新点**

将Bures–Helstrom度量视为三维圆球半球的度量，利用Hamilton–DeTurck规范化把Ricci流化为线性热方程，从而得到显式的齐次收缩解；在体积规范化下给出完整的线性化谱，证明其相对稳定并给出谱间隙；

**🔧 技术方法**

利用黎曼几何（Ricci流、DeTurck矢量场）、球面微分几何、偏微分方程（热方程、Laplace算子）、对称空间（旋转对称、球对称）等理论工具；

**📊 数据集**

无数据集，纯理论分析；

**📈 对比分析**

无实验比较，主要通过解析计算与理论推导验证；

**⚠️ 局限性**

仅在单比特的Bures–Helstrom度量上得到显式结果，未证明在更一般的单调度量或更高维量子系统上的保持单调性与稳定性，仍需进一步研究。

---

## 99. Concept Flow Models: Anchoring Concept-Based Reasoning with Hierarchical Bottlenecks

**arXiv ID:** 2606.19489 | [PDF](https://arxiv.org/pdf/2606.19489v1)

**作者:** Ya Wang `[一作]` (Fraunhofer Institute for Open Communication Systems), Adrian Paschke `[通讯]` (Fraunhofer Institute for Open Communication Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出概念流模型(CFM)，将概念瓶颈改为层级决策树，实现概念的结构化使用

**💡 创新点**

创新点在于通过层级化、概念驱动的决策路径降低信息泄漏并提升可解释性

**🔧 技术方法**

使用CLIP等视觉语言模型生成概念嵌入、LLM进行节点标注、Lasso回归挑选概念，并在树上学习可微分转移概率

**📊 数据集**

在CIFAR‑10/100、UCF‑101、CUB‑200、TinyImageNet等多种视觉分类基准上进行评估

**📈 对比分析**

与平面CBM、PCBM、Labo等对比，CFM在相同概念预算下保持相近或略优的准确率，且有效概念数大幅减少，SIR显著提升，随机概念下准确率显著下降

**⚠️ 局限性**

局限包括对视觉语言对齐的依赖、固定树结构假设、以及概念生成的领域泛化问题

---

## 100. Fail-RAG : A Retrieval Augmented Generation Informed Framework for Robot Failure Identification

**arXiv ID:** 2606.19598 | [PDF](https://arxiv.org/pdf/2606.19598v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 101. Hard or Just Unreached? Diagnosing the Sampling Blind Spot in Math-Reasoning Difficulty Estimation

**arXiv ID:** 2606.19636 | [PDF](https://arxiv.org/pdf/2606.19636v1)

**作者:** Luca Zhou `[一作]` (Sapienza University of Rome), Roberto Dessì `[通讯]` (Not Diamond)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 pass@k 评估在数学推理基准上的盲点，发现被标记为“最难”的样本中有10–29%在相同算力下可通过确定性推理（激活嫁接）解决。

**💡 创新点**

创新点在于：① 用残差流激活嫁接揭示 pass@k=0 不是固有难度，而是可达性问题；② 提供匹配算力下的确定性恢复策略，并验证其在不同模型与基准上的普适性；③ 证明确定性链数与恢复率呈线性关系。

**🔧 技术方法**

技术方法包括：激活嫁接（残差流干预）、多链确定性推理、采样与贪心推理对比、Jaccard相似度衡量机制多样性、不同温度下采样实验。

**📊 数据集**

数据集与模型：四款开源大模型（3B、8B、12B）在三套推理基准（GSM8K、MATH、MMLU‑Pro）上进行实验。

**📈 对比分析**

比较方式：对六个采样种子（k=6）与六条确定性链（greedy + 5 个嫁接向量）进行对比；结果显示在 pass@6=0 级别，确定性链恢复10–29%样本，单链贪心仅 0–9%；不同模型与基准均表现一致，恢复率随链数线性提升。

**⚠️ 局限性**

局限性：① 仅使用单层单位置的七种嫁接向量，恢复率仍有 66–88% 未覆盖；② 对多选题的提升有限；③ 未评估无标签时的可行识别；④ 在小规模 stratum（<100 示例）中噪声较大；⑤ 未探讨确定性相对采样的反向可达性。

---

## 102. PrefSQA: Pairwise Preference Prediction for Speech Quality Assessment and the Critical Role of High Quality Datasets

**arXiv ID:** 2606.19597 | [PDF](https://arxiv.org/pdf/2606.19597v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 103. Analyzing the Narration Gap in LLM-Solver Loops

**arXiv ID:** 2606.19588 | [PDF](https://arxiv.org/pdf/2606.19588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 104. IHBench: Evaluating Post-Interruption Recovery in Voice Agents with Structured Workflows

**arXiv ID:** 2606.19595 | [PDF](https://arxiv.org/pdf/2606.19595v1)

**作者:** Ahmad Salimi `[一作]` (Boson AI), Alex Smola `[通讯]` (Boson AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文未提供具体内容，因此无法总结做了什么。

**💡 创新点**

论文未提供具体内容，因此无法总结创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法总结使用的技术。

**📊 数据集**

论文未提供具体内容，因此无法总结使用的数据集。

**📈 对比分析**

论文未提供具体内容，因此无法总结比较的方法和性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法总结限制因素。

---

## 105. One Demo is Worth a Thousand Trajectories: Action-View Augmentation for Visuomotor Policies

**arXiv ID:** 2606.19586 | [PDF](https://arxiv.org/pdf/2606.19586v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 106. GDGU: A Gradient Difference-based Graph Unlearning Method for Cyberattack Localization in Electric Vehicle Charging Networks

**arXiv ID:** 2606.19566 | [PDF](https://arxiv.org/pdf/2606.19566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 107. SCAN-Planner: Spatial Collision-Aware Local Planning for Route-Guided Long-Range Quadruped Navigation

**arXiv ID:** 2606.19555 | [PDF](https://arxiv.org/pdf/2606.19555v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 108. SEAGAN: domain-Specific and Edge-Aware Graph Attention Network for Dynamic Plant Processes

**arXiv ID:** 2606.19623 | [PDF](https://arxiv.org/pdf/2606.19623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 109. VCG: A Multimodal Retrieval Framework for E-Commerce Video Feeds under Extreme Cold-Start Conditions

**arXiv ID:** 2606.19627 | [PDF](https://arxiv.org/pdf/2606.19627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 110. A BART-based approach with hierarchical strategy for Vietnamese abstractive multi-document summarization

**arXiv ID:** 2606.19591 | [PDF](https://arxiv.org/pdf/2606.19591v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 111. Which Pairs to Compare for LLM Post-Training?

**arXiv ID:** 2606.19607 | [PDF](https://arxiv.org/pdf/2606.19607v1)

**作者:** Jiangze Han `[一作]` (Columbia University), Will Ma `[通讯]` (Columbia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在有限标注预算下，如何在偏好式后训练（DPO）中进行离线比较样本选择。

**💡 创新点**

提出将比较选择视为采样设计问题，给出理论最优的设计准则，证明该准则同时给出RLHF性能的上界与下界；并基于参考策略给出可实施的插件设计。

**🔧 技术方法**

使用实验设计理论、Bradley–Terry模型、DPO优化、信息矩阵（Fisher信息与设计协方差）等技术。

**📊 数据集**

合成表格/上下文模型、IMDb 语料（情感分类器）以及 Anthropic‑HH 训练/测试拆分（使用 Pythia‑2.8B 与 GPT‑4.1 作为评判）。

**📈 对比分析**

与均匀采样、参考策略采样等基线比较，实验显示插件设计在样本效率上明显优于常见启发式方法，尤其在低标注预算时提升更为显著。

**⚠️ 局限性**

理论依赖可识别、平滑、覆盖等严格假设；仅针对离线随机设计，对在线自适应标注或大模型近似性缺乏分析。

---

## 112. pdSTL: Probabilistic Differentiable Signal Temporal Logic for Stochastic Systems

**arXiv ID:** 2606.19561 | [PDF](https://arxiv.org/pdf/2606.19561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 113. REMOP: REmote-Memory-aware OPerator Optimization

**arXiv ID:** 2606.19576 | [PDF](https://arxiv.org/pdf/2606.19576v1)

**作者:** Shiquan Zhang `[一作]` (University of Toronto), Hans-Arno Jacobsen `[通讯]` (University of Toronto)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种面向远程内存的操作符优化框架，通过在操作符内部划分缓冲区来最小化数据传输量和传输轮次，从而降低大规模查询在内存受限情况下的延迟；

**💡 创新点**

创新点在于引入了考虑远程内存往返延迟的转移轮次成本模型（L=D+τC），并据此推导出针对阻塞式嵌套循环 join、外部归并排序和外部哈希 join 的最优缓冲区分配策略；此外，还补充了双缓冲预取机制，实现了与传统基于磁盘的内存管理策略的互补。

**🔧 技术方法**

技术包括：远程内存成本建模、内存预算下的缓冲区分配求解、预取双缓冲、在 DuckDB 中实现的可配置操作符变体，以及与 TCP/IP 和 RDMA（Infiniswap）两种远程内存后端的集成。

**📊 数据集**

数据集：synthetic（用于微基准）以及标准商业 OLAP 工作负载 TPC‑H（22 句）和 TPC‑DS（99 句）在规模因子 10 上。

**📈 对比分析**

比较方法：与原始 DuckDB、基于磁盘的启发式分配、无预取的最优分配、全功能最优分配（含预取）以及 SPHJ 算法进行对比；通过平均运行时间、传输轮次、页交换量等指标评估；在微基准上最高可降低 97% 的传输轮次、48% 的单操作符运行时间；在 TPC‑H/TPC‑DS 的溢出子集上，整体几何平均运行时间分别下降 22.7% 与 26.4%。

**⚠️ 局限性**

局限性：仅适用于可被划分为输入/输出缓冲区且采用批量传输的操作符；对高并发多操作符场景的影响未充分评估；RDMA 后端的实现受限于内核级页面交换的细粒度，导致与用户空间后端相比性能波动；实验以固定 1 GB 本地内存预算为前提，缺乏对更大内存空间下的动态自适应能力。

---

## 114. LaViSA: A Language and Vision Structural Ambiguity Benchmark

**arXiv ID:** 2606.19552 | [PDF](https://arxiv.org/pdf/2606.19552v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 115. Language-Instructed Vision Embeddings for Controllable and Generalizable Perception

**arXiv ID:** 2606.19584 | [PDF](https://arxiv.org/pdf/2606.19584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 116. Understanding Key Features of Time Series Foundation Models from Epidemic Forecasting

**arXiv ID:** 2606.19560 | [PDF](https://arxiv.org/pdf/2606.19560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 117. Displacement Is Not Direction: Evaluating Fidelity Metrics for Quantized LLM Deployment

**arXiv ID:** 2606.19558 | [PDF](https://arxiv.org/pdf/2606.19558v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 118. A hybrid sharp-diffuse interface approach to accurately model melt pool dynamics with rapid evaporation in laser-based processing of metals

**arXiv ID:** 2606.19556 | [PDF](https://arxiv.org/pdf/2606.19556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 119. Toten: Knowledge-Based Ontological Tokenization Of Physical Quantities And Technical Notation In Brazilian Portuguese

**arXiv ID:** 2606.19626 | [PDF](https://arxiv.org/pdf/2606.19626v1)

**作者:** Antonio de Sousa Leitão Filho; Allan Kardec Duailibe Barros Filho; Fabrício Saul Lima; Selby Mykael Lima dos Santos; Rejani Bandeira Vieira Sousa `[一作]` `[通讯]`, Antonio de Sousa Leitão Filho; Allan Kardec Duailibe Barros Filho; Fabrício Saul Lima; Selby Mykael Lima dos Santos; Rejani Bandeira Vieira Sousa

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 TOTEN，一个基于正式工程实体本体的知识驱动词元化框架，用于在巴西葡萄牙语文本中对物理量、单位、数值和符号表达进行语义完整的标记。

**💡 创新点**

将词元化从统计子词分割转为基于本体的三元组⟨O, classify, {inst_τ}⟩，并通过三个外部专用 oracle（Pint、Unicode、RSLP）实现对结构、维度、排版和形态的判别。

**🔧 技术方法**

采用正式本体建模、离散分类函数、特定类型的实例化器、Pint 单位维度库、Unicode 字符数据库、RSLP 葡萄牙语形态学工具，以及 BNF 规范的输出语言。

**📊 数据集**

在内部自建 EngQuant benchmark（800 条结构工程实例）和四个公开巴西葡萄牙语语料（MMMLU PT_BR、BLUEX、ENEM Maritaca、Alvorada‑Bench，共 1,771 条可重建数值）进行实验。

**📈 对比分析**

与八个代表性系统（BPE、Quantulum3、CQE、GLiNER、Pint、udunits-2、spaCy NER）采用 McNemar 统计比较，TOTEN 在原子性、维度一致性和数值重构方面均显著优于对手；在数值重构上内部 0.780、外部 0.775–0.904，显著高于 Quantulum3 0.627–0.703。

**⚠️ 局限性**

仅保留文字原样，未主动规范排版变体；对非 SI 单位、维度无关表达覆盖有限；缺乏对法律/规范文本的评估；缺少下游模型性能验证。

---

## 120. Unsupervised Causal Abstractions Discovery

**arXiv ID:** 2606.19594 | [PDF](https://arxiv.org/pdf/2606.19594v1)

**作者:** Théo Saulus `[一作]` (Mila - Quebec AI Institute), Dhanya Sridhar `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种无监督学习高层因果抽象的框架（UCAD），能够从低层测量数据直接推断出高层结构因果模型，并给出了可训练的目标函数。

**💡 创新点**

创新点包括：① 利用低秩因果图与anchor假设证明低层SCM可被高层f‑DAG抽象；② 在该框架下给出因子可识别性与机制可重参数化的理论；③ 将理论转化为可梯度优化的目标，加入图正则、层次约束和anchor约束。

**🔧 技术方法**

技术手段包括：低秩因果发现（Boolean矩阵分解 + Gumbel‑Softmax 图表示）、anchor约束、层次化的有向无环图约束、加性高斯噪声机制、MSE/对数似然损失、正则化、以及对因子可重参数化的分析。

**📊 数据集**

数据集：① 采用合成的f‑SCM数据（不同规模、层次与噪声设置）；② 真实神经网络实验使用解决“是否能被6整除”任务的多层感知机激活作为低层观测。

**📈 对比分析**

评估方式：用MCC（Pearson）与MCC‑RDC衡量因子重建精度；在合成实验中平均MCC约为0.83；在除法任务中，正概念的MCC‑Pearson为0.41、MCC‑RDC为0.72，明显优于负控。相较于现有基线，UCAD能在无监督条件下恢复与真实因子高度一致的结构。

**⚠️ 局限性**

局限性：① 需要预先知道或设定因子数，缺少对超参数鲁棒性分析；② 在大规模、高维任务上的优化与可扩展性仍需提升；③ 仅验证了因子重建，没有进一步验证在实际干预中的可控性与因果效应的可靠性。

---

## 121. Configurable Clinical Information Extraction with Agentic RAG: What Works, What Breaks, and Why

**arXiv ID:** 2606.19602 | [PDF](https://arxiv.org/pdf/2606.19602v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 122. Uncertainty Decomposition for Clarification Seeking in LLM Agents

**arXiv ID:** 2606.19559 | [PDF](https://arxiv.org/pdf/2606.19559v1)

**作者:** Gregory Matsnev `[一作]` `[通讯]` (ITMO University), Gregory Matsnev (ITMO University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种提示式不确定性分解方法，分别估计动作置信度与请求不确定度，使大型语言模型代理能主动请求澄清并提升任务成功率

**💡 创新点**

将单一置信度分解为动作置信度与请求不确定度两个语义独立信号，并通过阈值化触发澄清动作，满足最新关于交互式代理的不确定性分解与可沟通的研究需求

**🔧 技术方法**

基于提示工程（prompt-based）在黑盒API下实现，使用结构化输出格式在单次前向推理中同时返回推理、两类不确定度及解释；采用阈值决策和历史语义传播的机制

**📊 数据集**

使用WebShop、ALFWorld、REAL三大交互式基准及其改造版WebShop-Clarification、ALFWorld-Clarification（每个任务50%人为歧义）

**📈 对比分析**

与ReAct+UE和UAM两种基准方法在五大LLM骨干（GPT‑5.1、DeepSeek‑v3.2‑exp、GLM‑4.7、Qwen3.5‑35B、GPT‑OSS‑120B）上进行比较；在澄清基准上，平均F1提升73%（相对ReAct+UE）和36%（相对UAM），在标准基准上与基准方法保持相当的错误检测性能

**⚠️ 局限性**

提示式方法受限于固定推理预算，导致“能力稀释”导致任务成功率略降；所有方法均呈现系统性过度自信；聚合策略为隐藏超参数，产品聚合在某些基准仅是轨迹长度的代理，难以获得可解释且可靠的不确定性估计

---

## 123. AI4SE and SE4AI Exploration: A Decade Looking Back and Forward

**arXiv ID:** 2606.19630 | [PDF](https://arxiv.org/pdf/2606.19630v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 124. Where Does Social Reasoning Come From? Capability Provenance in Language Models

**arXiv ID:** 2606.19625 | [PDF](https://arxiv.org/pdf/2606.19625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 125. Joint-task truthfulness of the DMI mechanism

**arXiv ID:** 2606.19618 | [PDF](https://arxiv.org/pdf/2606.19618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 126. Before the Pull Request: Mining Multi-Agent Coordination

**arXiv ID:** 2606.19616 | [PDF](https://arxiv.org/pdf/2606.19616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 127. Building Drift: Documenting On-Site Construction Adaptations Across Material Lifecycles

**arXiv ID:** 2606.19609 | [PDF](https://arxiv.org/pdf/2606.19609v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 128. Safe, Real-Time Active Model Discrimination and Fault Diagnosis for Nonlinear Systems via Differentiable Reachability

**arXiv ID:** 2606.19590 | [PDF](https://arxiv.org/pdf/2606.19590v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 129. Comparing Linear Probes with Mahalanobis Cosine Similarity

**arXiv ID:** 2606.19603 | [PDF](https://arxiv.org/pdf/2606.19603v1)

**作者:** Zhuofan Josh Ying `[一作]` (Columbia University), Nikolaus Kriegeskorte `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了线性探针的 Mahalanobis 余弦相似度 (MCS)，并证明其与 OOD AUROC 近似线性，随后在多模型、多层以及多任务（真相、性别分类与通用 NLP）上系统验证其优越性。

**💡 创新点**

①推导 MCS 与 Fisher 方向之间的闭式表达；②理论解释 AUROC 与 MCS 的线性关系并给出任务独立斜率；③预测并验证线性失效的四种情形；④展示 MCS 在广泛设定下显著优于欧氏余弦相似度。

**🔧 技术方法**

使用线性探针（Logistic Regression/LDA）训练、Mahalanobis 余弦相似度计算、SNR 与 Fisher 距离分析、正态投影假设下的理论推导、仿真与实测对比。

**📊 数据集**

24 个任务：10 个真相数据集、6 个性别分类数据集、8 个通用 NLP 分类基准；模型包括 LLaMA、GPT‑3 等 LLM；层级别涵盖 20、33、50、65。

**📈 对比分析**

对 OOD AUROC 进行线性回归，比较 MCS 与欧氏余弦相似度（ECS）的 R²。MCS 在所有条件下 R² ≥ 0.93，ECS 最差仅 0.06；MCS 的热图结构几乎与 AUROC 热图一致，说明其能精准预测泛化性能。

**⚠️ 局限性**

需要 OOD 标注数据估计 Σ_tot；理论仅适用于 Fisher‑style 探针（LR/LDA），差值均值 probe 等非 Fisher 探针失效；仅验证二分类、残差流特征，未测试多分类、注意力/MLP 或非 LLM 结构；线性关系近似，极端 SNR、类不平衡或小 Fisher 距离时会失效；假设投影为正态，未涵盖重尾分布。

---

## 130. Exploring Feature Extraction Technique Parameters for Acoustic Gunshot Classification

**arXiv ID:** 2606.19568 | [PDF](https://arxiv.org/pdf/2606.19568v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 131. Before the Labels: How Dataset Construction Shapes Suicidality Detection in Clinical Text

**arXiv ID:** 2606.19637 | [PDF](https://arxiv.org/pdf/2606.19637v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 132. CTS-MoE: Implicit Terrain Adaptation via Mixture-of-Experts for Perceptive Locomotion

**arXiv ID:** 2606.19633 | [PDF](https://arxiv.org/pdf/2606.19633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 133. RIVET: Robust Idempotent Voice Attribute Editing

**arXiv ID:** 2606.19629 | [PDF](https://arxiv.org/pdf/2606.19629v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 134. MassSpecGym in the Wild: Uncovering and Correcting Evaluation Pitfalls in AI-Driven Molecule Discovery

**arXiv ID:** 2606.19624 | [PDF](https://arxiv.org/pdf/2606.19624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 135. G-Lox: Group-Adaptive, Privacy-Preserving Bridge Distribution with Two-Party Computation

**arXiv ID:** 2606.19620 | [PDF](https://arxiv.org/pdf/2606.19620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 136. GB-LSR: A Fast Local Spectral Image Representation with a Single Global Bandwidth for Continuous Reconstruction and Super-Resolution

**arXiv ID:** 2606.19617 | [PDF](https://arxiv.org/pdf/2606.19617v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 137. StaminaBench: Stress-Testing Coding Agents over 100 Interaction Turns

**arXiv ID:** 2606.19613 | [PDF](https://arxiv.org/pdf/2606.19613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 138. FlowFake: Liquid Networks for Audio Deepfake Detection

**arXiv ID:** 2606.19579 | [PDF](https://arxiv.org/pdf/2606.19579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 139. Code as Anchor, Memory and Metaphor as Support: Learner Experiences with Multi-View Visualizations

**arXiv ID:** 2606.19570 | [PDF](https://arxiv.org/pdf/2606.19570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 140. On the QUEST for Uncertainty Quantification via Highest Density Regions

**arXiv ID:** 2606.19569 | [PDF](https://arxiv.org/pdf/2606.19569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 141. Mix-QVLA: Task-Evidence-Aware Mixed-Precision Quantization of Vision-Language-Action Models

**arXiv ID:** 2606.19565 | [PDF](https://arxiv.org/pdf/2606.19565v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 142. Advances in Scientific Machine Learning for Coupled Fluid Flow and Transport

**arXiv ID:** 2606.19562 | [PDF](https://arxiv.org/pdf/2606.19562v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 143. Token Factory: Efficiently Integrating Diverse Signals into Large Recommendation Models

**arXiv ID:** 2606.19635 | [PDF](https://arxiv.org/pdf/2606.19635v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 144. Formal Verification of Learned Multi-Agent Communication Policies via Decision Tree Distillation

**arXiv ID:** 2606.19632 | [PDF](https://arxiv.org/pdf/2606.19632v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 145. Closing the Calibration Gap in Semantic Caching

**arXiv ID:** 2606.19719 | [PDF](https://arxiv.org/pdf/2606.19719v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 146. One-Shot Novel View and Pose Human Image Synthesis via 3D Prior Guided Diffusion Model

**arXiv ID:** 2606.19718 | [PDF](https://arxiv.org/pdf/2606.19718v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 147. A Differentiable Composite Approximation Framework for Autonomous Underwater Vehicle Maneuvering Modeling from Sea-Trial Data

**arXiv ID:** 2606.19711 | [PDF](https://arxiv.org/pdf/2606.19711v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 148. Comparative Study on Agility, Efficiency, and Impact Absorption of Bipedal Robots with Active Toes

**arXiv ID:** 2606.19699 | [PDF](https://arxiv.org/pdf/2606.19699v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 149. When Global Gating Is Enough: Admission-Time Hubness Control in Anisotropic Vector Retrieval Systems

**arXiv ID:** 2606.19692 | [PDF](https://arxiv.org/pdf/2606.19692v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 150. ImProNCDE: Impulse-Corrected Neural Controlled Differential Equations with Prototype Learning for Longitudinal Prognosis Prediction

**arXiv ID:** 2606.19680 | [PDF](https://arxiv.org/pdf/2606.19680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 151. CacheWeaver: Cache-Aware Evidence Ordering for Efficient Grounded RAG Inference

**arXiv ID:** 2606.19667 | [PDF](https://arxiv.org/pdf/2606.19667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 152. DF-ExpEnse: Diffusion Filtered Exploration for Sample Efficient Finetuning

**arXiv ID:** 2606.19656 | [PDF](https://arxiv.org/pdf/2606.19656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 153. Convex training of Lipschitz-regularized shallow neural networks

**arXiv ID:** 2606.19652 | [PDF](https://arxiv.org/pdf/2606.19652v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 154. SAFE-Cascade: Cost-Adaptive Vision-Language Routing for Chart Question Answering

**arXiv ID:** 2606.19646 | [PDF](https://arxiv.org/pdf/2606.19646v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 155. Creating Multilingual Mental Health Dialogue Datasets: Limits of Persona-Based Localization via Nationality and Language

**arXiv ID:** 2606.19640 | [PDF](https://arxiv.org/pdf/2606.19640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 156. NRITYAM: Language Models Meet Art and Heritage of Dance

**arXiv ID:** 2606.19727 | [PDF](https://arxiv.org/pdf/2606.19727v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 157. NEST: Narrative Event Structures in Time for Long Video Understanding

**arXiv ID:** 2606.19706 | [PDF](https://arxiv.org/pdf/2606.19706v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 158. Library-Aware Doubles and Iterative Repair for Large Language Model-Generated Unit Tests in OpenSIL Firmware

**arXiv ID:** 2606.19725 | [PDF](https://arxiv.org/pdf/2606.19725v1)

**作者:** Ma Toan Bach `[一作]` (Seneca Polytechnic), Jitesh Arora `[通讯]` (Advanced Micro Devices)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过多代理LLM和检索增强的迭代管道，自动生成并修复 AMD openSIL 固件的单元测试。

**💡 创新点**

创新点在于结合库感知的测试双、覆盖引导的修复循环，显著提升构建成功率和行覆盖率。

**🔧 技术方法**

采用 GPT‑4.1‑mini、o4‑mini、o3 等 LLM，配合 Chroma 向量数据库检索以及 LCOV 覆盖分析。

**📊 数据集**

使用 AMD openSIL GitHub 仓库中的 76 个被测函数（按大小与依赖划分）作为实验数据集。

**📈 对比分析**

相较于直接 LLM 生成，编译成功率从约 50% 提升至 96%，平均行覆盖率从 55% 提升至 99%，迭代次数平均仅 2–3 次，性能显著提升。

**⚠️ 局限性**

仍受深度依赖、指针密集代码以及检索质量的限制，且评估仅聚焦构建和覆盖，未检验语义正确性。

---

## 159. Effect Systems as Abstract Interpretations

**arXiv ID:** 2606.19686 | [PDF](https://arxiv.org/pdf/2606.19686v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 160. BrainG3N: A Dual-Purpose Tokenizer for Controllable 3D Brain MRI Generation

**arXiv ID:** 2606.19651 | [PDF](https://arxiv.org/pdf/2606.19651v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 161. Prompt Quality and Pull Request Outcomes: A Stage-Based Empirical Study of LLM-Assisted Development

**arXiv ID:** 2606.19644 | [PDF](https://arxiv.org/pdf/2606.19644v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 162. Beyond Static Leaderboards: Predictive Validity for the Evaluation of LLM Agents

**arXiv ID:** 2606.19704 | [PDF](https://arxiv.org/pdf/2606.19704v1)

**作者:** Dhaval C. Patel `[一作]` (IBM), Byeolah Kwon `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文聚合并分析了14项基于MCP的工业代理基准的扩展研究，提出了12维度测量体系和以预测有效性（in‑sample 与 out‑of‑sample 相关性）为核心的排名标准，以期改进现有聚合分数排行榜。

**💡 创新点**

创新点在于：①将多项代理基准的评测维度系统化为12个近正交的层级；②提出“预测有效性”作为排行榜的评估准则；③通过三种 OOD 评价标准（保留子集、跨子集迁移、对抗扰动）验证排行榜的转移性；④对四大类评测维度（资产、编排、知识检索、基础设施、推理模式、评估方法）进行多实验交叉验证。

**🔧 技术方法**

使用的技术包括多模态视觉扩展、MCP（Multi‑Choice Prompt）框架、知识插件、置信门控路由、时序语义缓存、Torch.compile GPU 加速、ReAct / Claude Code 迁移、自动场景生成、LLM‑as‑judge 与规则或 DAG 验证器等。

**📊 数据集**

主要使用的数据集为公开的 AssetOpsBench（含 HVAC、变压器、泵、PHMForge 等资产场景）及其衍生的 14 个扩展数据集，数据来源包括 NASA Li‑ion 循环数据、IEC 标准变压器数据、工业监测数据等。

**📈 对比分析**

比较方法：基于 Spearman 相关系数测量 in‑sample 与各 OOD 方案的排名一致性，使用多维度指标（通过多维度排行榜、成本–收益 Pareto 图、置信区间等）评估不同配置的表现。结果显示，聚合分数排名在 OOD 评测上往往不稳定，预测有效性更能反映部署时的性能。

**⚠️ 局限性**

限制：①缺乏大规模实证验证，预测有效性准则尚未通过完整实验验证；②仅针对工业资产运维领域，通用性未知；③12维度的正交性仍是假设，需进一步实验检验；④未与真实部署数据（如操作员干预率、故障率）关联；⑤对资源有限机构可能增加评测成本。

---

## 163. Efficient Neural Network Model Selection for Few-Class Application Datasets

**arXiv ID:** 2606.19712 | [PDF](https://arxiv.org/pdf/2606.19712v1)

**作者:** Bryan Bo Cao `[一作]` (Stony Brook University), Shubham Jain `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于数据属性的分类难度度量，旨在为少类应用数据集选择高效的神经网络模型。通过分析数据侧属性，研究了少类数据集的特性，并展示了如何利用这些特性进行模型选择。

**💡 创新点**

创新点在于提出了“少类独特性”概念，表明少类应用数据集的分类准确性与多类数据集的表现不同，从而帮助选择更高效的模型。此外，提出了一种新的分类难度度量方法，能够快速比较模型和数据集。

**🔧 技术方法**

使用了基于类相似性和分类难度的度量方法，结合了多种神经网络架构（如MobileNet-V2、ResNet-50和ViT）进行实验。

**📊 数据集**

使用了CIFAR-100和ImageNet-200数据集，进行了多次实验以验证提出的分类难度度量和少类独特性的有效性。

**📈 对比分析**

与传统方法相比，使用新的分类难度度量可以在6到29倍的速度下比较模型和数据集的性能，且在少类数据集上，所选模型的效率更高，准确性相似。例如，在移动机器人任务中，所选模型比YOLOv5-nano小42%。

**⚠️ 局限性**

限制在于实验结果主要基于CIFAR-100和ImageNet-1000数据集，可能不适用于其他特定数据集。此外，未对超过10类的情况进行深入研究，且未涵盖数据侧属性中的尺度-分辨率特性。

---

## 164. TerraMARS: A Domain-Adapted Small-Language-Model Pipeline for Mars Terraforming Literature

**arXiv ID:** 2606.19700 | [PDF](https://arxiv.org/pdf/2606.19700v1)

**作者:** Jyotsna Singh `[一作]` (University of Arizona), Scott R. Saleska `[通讯]` (University of Arizona)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了 TerraMARS，一套针对火星 Terraforming 的端到端信息提取流水线，能够回答问题并将文献中的定量约束自动转换为结构化 JSON；

**💡 创新点**

其创新点在于将低秩适配 QLoRA 与小型 Gemma 3.1B 结合，构建单模型完成多任务（问答、组织识别、阶段推理、干预识别、存活链式推理、结构提取），并首次实现文献到 JSON 的全自动转换；

**🔧 技术方法**

使用的技术包括领域适配的小型 LLM（Gemma 3.1B + 4‑bit NF4 量化 + QLoRA）、教师-学生知识蒸馏（Llama 3.2 3B 生成合成指令数据）、多阶段检索与分块、混合精度（bfloat16）训练与梯度累积；

**📊 数据集**

数据集为从 arXiv、PMC、Semantic Scholar 搜集的 393 篇开放获取火星相关论文摘要，随后生成 1,179 条合成指令样本（约束提取 190、问答 199 等），用于训练与验证；

**📈 对比分析**

在 6 条手工评估案例中，模型的 JSON 输出符合 schema，约 70‑80% 的字段与原文对应，表现优于单一任务模型，但整体性能受样本量与摘要信息不足限制，未给出精确的 BLEU/ROUGE 等指标；

**⚠️ 局限性**

局限性包括：仅使用摘要导致信息缺失、领域标签偏向 general、模型规模小导致推理与事实一致性差、合成数据可能携带教师错误、JSON 结构匹配率低且内容真实性不高。

---

## 165. MiqraBERT: Regression-Based Sentence-BERT Finetuning for Biblical Hebrew Parallel Detection

**arXiv ID:** 2606.19638 | [PDF](https://arxiv.org/pdf/2606.19638v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 166. Learning When to Denoise: Optimizing Asynchronous Schedules for Latent Diffusion

**arXiv ID:** 2606.19662 | [PDF](https://arxiv.org/pdf/2606.19662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 167. Parity Selection Rule for Information and Dissipation in Driven Steady States

**arXiv ID:** 2606.19702 | [PDF](https://arxiv.org/pdf/2606.19702v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 168. What sentiment analysis can't see: Measuring whether customers were helped, and what went wrong, across 70,000 support conversations

**arXiv ID:** 2606.19698 | [PDF](https://arxiv.org/pdf/2606.19698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 169. Multi-Granular Attention-Driven Reinforcement Learning Framework for Web Intelligent Enhancement Systems

**arXiv ID:** 2606.19690 | [PDF](https://arxiv.org/pdf/2606.19690v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 170. ForEnt: A Multi-Modal Dataset for Characterizing Quadruped Robot Entrapments in Forest Environments

**arXiv ID:** 2606.19675 | [PDF](https://arxiv.org/pdf/2606.19675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 171. Scaling Self-Play for End-to-End Driving

**arXiv ID:** 2606.19641 | [PDF](https://arxiv.org/pdf/2606.19641v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 172. SAGE-OPD: Selective Agent-Guided Intervention for Multi-Turn On-Policy Distillation

**arXiv ID:** 2606.19659 | [PDF](https://arxiv.org/pdf/2606.19659v1)

**作者:** Yuhang Zhou `[一作]` (Meta AI), Zhuokai Zhao `[通讯]` (Meta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种面向多轮交互的无验证器选择性干预（Verifier‑free Selective Intervention）框架，针对传统的密集式 on‑policy distillation（OPD）在多轮环境中易出现错误累积与 token‑level 细粒度不稳定的问题，设计了按轮次干预、教师置信度加权以及 loss 归一化等机制；

**💡 创新点**

创新点在于：1）引入基于环境反馈与教师判断的轮次级干预标签（Skip/Weak/Strong），避免在不需要的轮次做过度监督；2）利用教师的 top‑1 置信度为每个轮次提供可靠性权重，缓解了因状态漂移导致的监督噪声；3）通过 loss 归一化保持整体训练信号尺度，保证相对权重的可比性；

**🔧 技术方法**

技术包括：on‑policy distillation、密集 token‑level KL 对齐、教师置信度估计（top‑1 概率）、教师干预标签推断（无验证器）、loss 归一化；

**📊 数据集**

实验数据集涵盖三类多轮任务：ALFWorld（实体交互模拟）、ScienceWorld（科学实验任务）和 SearchQA（搜索式问答），使用 Qwen3 系列模型作为教师/学生；

**📈 对比分析**

与基线（off‑policy SFT、标准 OPD、TCOD‑F2B、Entropy‑Aware OPD）以及一次性双回合 OPD 对比。实验显示在 ALFWorld 上可提升 13.3% 以上的成功率，在 ScienceWorld 和 SearchQA 也获得了显著或竞争性提升，且平均轮次略有下降或相近；

**⚠️ 局限性**

局限性包括：1）需要教师推断轮次标签，可能受教师模型能力限制；2）置信度权重依赖于教师的概率分布，若教师不稳定可能导致误加权；3）未在更大规模或更复杂的多轮环境中验证，且计算成本略高于标准 OPD。

---

## 173. OnDeFog: Online Decision Transformer under Frame Dropping

**arXiv ID:** 2606.19721 | [PDF](https://arxiv.org/pdf/2606.19721v1)

**作者:** Daiki Yotsufuji `[一作]` (Yokohama National University), Shinichi Shirakawa `[通讯]` (Yokohama National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在线强化学习方法 OnDeFog，将 DeFog 的帧丢失学习机制集成到 Online Decision Transformer（ODT）中，以实现对帧丢失环境的鲁棒控制。

**💡 创新点**

创新点在于将 DeFog 的 train‑time frame dropping 与 drop span embedding 机制与 ODT 的在线学习框架结合，既消除了 DeFog 对离线数据的依赖，又显著提升了 ODT 在高帧丢失率环境下的性能。

**🔧 技术方法**

使用了 Transformer 架构的 Decision Transformer、ODT、DeFog 的训练时帧丢失、drop span embedding 等技术；并采用经验回放、目标返回重标记等在线强化学习方法。

**📊 数据集**

使用了 D4RL Gym‑MuJoCo benchmark 中的 Hopper、Walker2d、HalfCheetah 的 medium 与 medium‑replay 数据集，并在对应的 OpenAI Gym MuJoCo 环境中进行在线评估。

**📈 对比分析**

通过在 0~0.9 的帧丢失率下进行 100 次评估，计算 IQM 与 95% 置信区间进行比较。实验显示 OnDeFog 在高帧丢失率环境下显著优于 ODT，并在 medium‑replay 数据集上超过 DeFog；在低丢失率或 medium 数据集上表现相当或略逊。

**⚠️ 局限性**

主要局限在于：1) 对低奖励轨迹或大动作空间（如 HalfCheetah）任务的探索能力不足，导致无法发现更高奖励轨迹；2) 在低帧丢失率环境下训练时 drop‑rate 过高，适应不足；3) 受限于 ODT 的探索噪声，整体探索效率有限。

---

## 174. FineREX: Fine-Tuned NER-RE for Human Smuggling Knowledge Graphs

**arXiv ID:** 2606.19710 | [PDF](https://arxiv.org/pdf/2606.19710v1)

**作者:** Elijah Feldman `[一作]` (Thomas Jefferson High School for Science and Technology), Carlotta Domeniconi `[通讯]` (George Mason University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在法院程序文本中构建人类走私网络知识图谱，提出FineREX管线；

**💡 创新点**

通过对LLaMA 3.1 8B进行领域专门化微调，去除文本重写和冗余抽取步骤，显著提升实体与关系抽取质量与效率；

**🔧 技术方法**

使用QLoRA参数高效微调、LLM核心指代映射、图谱合并机制；

**📊 数据集**

自行标注的512个1-3句文本块数据集，包含1962个实体和2591条关系；

**📈 对比分析**

与基线LLaMA 3.3 70B及未微调8B模型对比，FineREX在实体F1提升15.5%、关系F1提升31.5%，节点重复率下降至11.2%，法律噪声降至近一半，端到端处理时间减半；

**⚠️ 局限性**

标注主观性、少量特定实体类别样本不足、合并关系导致信息丢失等局限。

---

## 175. Efficiently Representing Algorithms With Chain-of-Thought Transformers

**arXiv ID:** 2606.19697 | [PDF](https://arxiv.org/pdf/2606.19697v1)

**作者:** Yanhong Li `[一作]` (Allen Institute for AI), William Merrill `[通讯]` (Allen Institute for AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

论文证明链式思考（CoT）Transformer能够直接、近似高效地模拟Word RAM算法，所需的CoT步数仅比理想算法多多项式对数因子。

**💡 创新点**

创新点在于：①首次将CoT模型与Word RAM模型直接对应；②在三种Transformer变体（多项式宽度、连续CoT、混合RNN）中实现了多项式对数的模拟开销；③提出了固定宽度Transformer需要的最小扩展（连续CoT或线性RNN）以实现高效模拟。

**🔧 技术方法**

技术上利用了：右侧唯一硬注意力、层归一化哈希表示、连续CoT的软token持久化、线性RNN的状态传递、对齐编码与解码、以及对指令集的“平坦”与“无乘法”分层分析。

**📊 数据集**

论文为理论分析，不使用实验数据集，主要通过构造证明与信息论下界验证。

**📈 对比分析**

通过与传统TM模拟的二次开销对比，证明CoT模拟在多项式宽度下的开销为 O(t log²t)，在连续CoT或混合模型下为 O(t log t) 或 O(t log²t)（取决于指令集），显著优于传统TM模拟的 O(t·poly(t)) 级别。

**⚠️ 局限性**

限制包括：①固定宽度Transformer的最优性尚未完全证明；②需要额外的连续CoT或RNN层，可能不符合所有实际Transformer实现；③仅在“平坦”或“无乘法”指令集下实现最优开销，对更复杂指令集仍存在 log²t 的开销。

---

## 176. A Unified Framework for Joint Sensor Placement and Scheduling for Intrusion Detection

**arXiv ID:** 2606.19695 | [PDF](https://arxiv.org/pdf/2606.19695v1)

**作者:** Jayanth Bhargav `[一作]`, Shreyas Sundaram `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在入侵检测任务中，提出了联合优化传感器布置位置与方向调度的框架，利用零和博弈的Nash均衡评估检测性能，并将该价值作为子问题的效用用于主问题的子集选择；

**💡 创新点**

首次将博弈论的均衡价值与弱子模（weak‑submodular）优化结合，实现传感器布置与调度的统一设计；

**🔧 技术方法**

使用零和博弈、乘数权重更新（multiplicative weight update）求解均衡、弱子模贪心算法、LP和分布式最优响应求解；

**📊 数据集**

九宫格（9‑room）网格世界环境与人工生成的入侵者路径集（约50条）以及随机生成的传感器检测概率；

**📈 对比分析**

与传统LP、单纯的WMA以及随机布置+随机调度/均匀调度等基线比较，实验表明该框架在给定预算下实现了约92%~100% 的近似最优检测性能，同时计算时间比LP和WMA低数倍至数十倍；

**⚠️ 局限性**

对传感器数量、路径数量和检测概率分布的依赖性仍需进一步评估；弱子模常数估计保守，导致贪心算法的理论下界偏高；在极大规模场景下仍需更高效的近似或分布式实现。

---

## 177. Route-Constrained Robust Fusion Estimation for MEMS/GNSS Integrated Navigation of Unmanned Ground Vehicles in GNSS Degraded Environments

**arXiv ID:** 2606.19687 | [PDF](https://arxiv.org/pdf/2606.19687v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 178. LOKI: Memory-Free Null-Space Constrained Lifelong Knowledge Editing

**arXiv ID:** 2606.19679 | [PDF](https://arxiv.org/pdf/2606.19679v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 179. Safe Local Navigation for Ackermann-Steered Robots in Unmapped Environments

**arXiv ID:** 2606.19672 | [PDF](https://arxiv.org/pdf/2606.19672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 180. Exit-and-Join Dynamics for Decentralized Coalition Formation

**arXiv ID:** 2606.19683 | [PDF](https://arxiv.org/pdf/2606.19683v1)

**作者:** Quanyan Zhu `[一作]` `[通讯]` (New York University), Quanyan Zhu (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文提出了一种基于Aumann–Drèze价值的分散退出–加入（exit–and–join）动态模型，用于研究合作博弈中联盟结构的自发演化，并给出了相应的平衡概念与收敛分析。

**💡 创新点**

创新点包括：①把Aumann–Drèze价值嵌入单个代理的本地决策中，实现非合作的最优响应与合作收益分配的统一；②在此框架下引入Lyapunov函数与潜在游戏理论，给出充分的收敛与稳定性条件；③通过精确与序数边际对齐（marginal alignment）区分凸性与动态实现的关系；④通过大规模随机聚类对称对偶游戏与凸基准游戏的数值实验验证理论。

**🔧 技术方法**

主要技术手段有：Aumann–Drèze价值的矩阵表示与分块结构；离散时间异步最佳响应动力学；接受规则与切换成本的形式化；序数与精确边际对齐定义；潜在游戏与Lyapunov函数的构造与证明；事件触发的线性系统表述。

**📊 数据集**

数据集为人工生成的合作博弈：180个随机聚类对称对偶游戏（n=24、四个簇）与一个凸基准游戏 v(S)=0.45|S|²；通过不同切换成本与接受成本设置，记录每次实验的接受移动次数、终止分区规模与总协同收益。

**📈 对比分析**

对比方法主要是不同成本参数（切换成本c、接受成本κ）对收敛时间、协同收益与分区碎化的影响。实验结果显示：在满足序数边际对齐的聚类游戏中，Lyapunov函数严格递增，所有运行在有限步内终止；切换成本与接受成本升高会显著减少移动次数、降低总协同收益并导致更碎化的终止结构；在凸基准游戏中，动态过程从单点起始可收敛到唯一的全局最优联盟结构。性能表现符合理论预测。

**⚠️ 局限性**

局限性包括：①只考虑单个代理的单向退出–加入偏好，未涵盖组内协商或合并/分裂等更复杂的集体动作；②接受规则和切换成本的设定相对简化，实际系统中可能涉及多重约束与信息不完全；③凸性虽保证效率，但并不能确保动态过程选择全局最优结构；④数值实验仅在合成游戏上验证，缺乏真实世界实例；⑤在存在多重局部最优时，最终终止结构高度依赖初始分区与激活顺序。

---

## 181. Code-Switching Reveals Language Anchoring in Multilingual LLMs

**arXiv ID:** 2606.19668 | [PDF](https://arxiv.org/pdf/2606.19668v1)

**作者:** Jeonghyun Park `[一作]` (Chung-Ang University), Hwanhee Lee `[通讯]` (Chung-Ang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多语言大型语言模型在代码切换输入下的内部语言锚定机制，并提出一种推理时干预方法CANVAS来纠正锚定偏移。

**💡 创新点**

提出Anchor Bias度量衡量代码切换表示相对于源语言或目标语言锚点的几何偏向，并利用此度量实现自适应的源侧对齐干预。

**🔧 技术方法**

使用几何向量相似度、层级平均池化、随机余弦归一化、软插值与上层激活钩子等技术实现CANVAS；同时构造源/目标/语法强制的代码切换四元组。

**📊 数据集**

利用SimpleQA Verified中的源语言事实问答、CodeMixQA的语法强制代码切换变体，并翻译生成目标语言问答；评测多语言模型包括Aya‑Expanse、Qwen3、Llama‑3、Mixtral、Phi‑3.5等。

**📈 对比分析**

与基线贪婪解码、随机对齐和自适应对齐进行对比，CANVAS在多种模型、语言和语法框架下平均提升QA F1 1–3点，尤其在目标框架下提升显著；在多轮对话、双语/三语对话等更复杂场景也获得正向提升。

**⚠️ 局限性**

仅关注源-目标两方的锚定对齐，未能完全捕捉自然代码切换中的局部句法、命名实体和多语言知识差异；对极端混合或多语种切换的泛化仍有限。

---

## 182. A Layered Security Framework Against Prompt Injection in RAG-Based Chatbots

**arXiv ID:** 2606.19660 | [PDF](https://arxiv.org/pdf/2606.19660v1)

**作者:** Gulshan Saleem `[一作]` (Sparkverse AI Ltd), Ali Hassan `[通讯]` (Sparkverse AI Ltd)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对 RAG 聊天机器人中的直接与间接提示注入攻击，设计并实现了一个三层防御框架：输入筛查、基于来源的上下文组装以及输出审计，并配合持续审计循环进行自适应改进。

**💡 创新点**

创新点在于：①将防御覆盖到整个推理流程，联合使用三层互补机制实现对两种攻击向量的完整覆盖；②在层 2 引入权限层次化的指令层级和检索文档扫描，实现对检索内容注入的预先过滤；③通过持续审计日志反馈实现模型自我学习，适应新型攻击。

**🔧 技术方法**

主要技术包括：正则表达式模式匹配与语义异常分类器（基于 MiniLM）用于输入筛查；检索文档块扫描与 provenance‑tag 注释；输出层使用规则引擎、cosine 相似度检查、Llama Guard2 违规检测和语义漂移检测；以及基于日志的阈值校准与模型重训练。

**📊 数据集**

使用 5,080 条样本的综合数据集，包含 PromptBench、BIPIA 公开对抗样本、手工构造的 188 条攻击样本以及两轮 GPT‑4o 语义扩充，正负样本分别来自 MS‑MARCO 过滤得到的 2,240 条客服相关查询。

**📈 对比分析**

对 GPT‑4o、Llama‑3 8B 和 Mistral‑7B 三种模型进行评估；与无防御、单层防御以及 NeMo Guardrails 进行对比。宏平均结果显示，完整框架将攻击成功率从 71.4% 降低到 11.3%（-60.1pp），FPR 为 4.8%；层 3 单独最优，层 1、层 2 亦各自贡献显著；整体延迟中值 61.2 ms，低于 100 ms 运营阈值。

**⚠️ 局限性**

主要限制包括：仅在单轮对话中评估，无法处理多轮累积注入；对知识库内容的假设相对受控，真实场景下的恶意内容多样性未充分覆盖；阈值和检测器对分布漂移敏感；白盒对抗攻击尚未验证；并且层 3 规则对隐式指令的检测能力有限。

---

## 183. PUFFERDOS: Efficient and Effective Attack String Generation for Regular Expression Denial of Service Vulnerabilities

**arXiv ID:** 2606.19654 | [PDF](https://arxiv.org/pdf/2606.19654v1)

**作者:** Shangzhi Xu `[一作]` (University of New South Wales), Siqi Ma `[通讯]` (University of Wollongong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于形式化分析的混合框架，生成高效可在实际程序中可利用的ReDoS攻击字符串，并使用基于函数摘要的组合符号执行进行验证。

**💡 创新点**

①通过理论证明确定最短可导致最大匹配成本的重复单元，得到三种模式（EOL、LIL、PML）并生成更短攻击字符串；②提出ReDoS特定的组合符号执行，使用精简函数摘要以避免路径爆炸并验证攻击字符串的实际可达性。

**🔧 技术方法**

形式化正则表达式模型、模式匹配与递归AST展开、组合符号执行、函数摘要技术、Python/CrossHair/pytype等工具。

**📊 数据集**

17,962个真实漏洞正则、51个CVE实例（31个已验证）、12个热门Python项目（共95个正则），以及公开的正则语料库Corpus和RENGAR。

**📈 对比分析**

与领先工具RENGAR对比，攻击字符串长度平均缩短97×-3872×，在10s阈值下性能提升≈1000×；在CVE复现率达到96.8%，比RENGAR高约19个百分点；在真实项目中发现59个漏洞，比RENGAR多25个，平均耗时约14.6min。

**⚠️ 局限性**

仅针对支持回溯的正则引擎，未覆盖lookaround、命名组等高级特性；符号执行仍有假负风险；仅在Python生态中评估，跨语言通用性待验证。

---

## 184. From 50K to 8.2 Million in 24 Hours: Vozinha's Algorithmic Consecration and the Multilingual Making of World Cup Visibility

**arXiv ID:** 2606.19647 | [PDF](https://arxiv.org/pdf/2606.19647v1)

**作者:** Vinicius Covas `[一作]` `[通讯]` (Universidad Anahuac Mexico), Vinicius Covas (Universidad Anahuac Mexico)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对2026年世界杯卡塔多尔与西班牙0-0平局后门将Vozinha在四种语言（葡语、西语、英语、法语）中的话语扩散进行了多语言计算话语分析，重点研究了平台指标（粉丝数）如何被话语化为象征性认可。

**💡 创新点**

创新点在于将平台指标视为“叙事框架”本身（F4：metric spectacle），并提出了一套九框架分类学、LLM辅助但人类验证的注释流程，以及跨语言框架扩散的可复现分析方法。

**🔧 技术方法**

采用了自然语言处理技术（词汇/语义提示提取、命名实体识别）、大型语言模型（LLM）用于初始框架标注、人工校验、以及对视觉证据的手工文本转录与注释。

**📊 数据集**

使用的数据集包括：葡语、西语、英语、法语的新闻标题、正文、推文、Instagram故事标题以及48张截图（已哈希）；还构建了保守的粉丝增长时间线（仅有8235652的精确抓取点）。

**📈 对比分析**

该研究主要是方法论演示（v0.1 pilot），通过手工和LLM协同标注检验框架分类的可用性，并用跨语言框架出现频次和时间分布做定性比较；未给出数值性能指标，未来计划完成双重注释和Kappa/α一致性评估。

**⚠️ 局限性**

局限性包括：时间线为保守估计，仅有单一精确抓取点；样本为种子样本，未覆盖完整语料；缺乏完整的API级粉丝序列；截图缺乏时序密度；LLM建议需人工验证；尚未完成一致性检验和大规模验证。

---

## 185. Syndesmoscope: The Power of Invariant Plots\\Linked to Traditional Network Views

**arXiv ID:** 2606.19689 | [PDF](https://arxiv.org/pdf/2606.19689v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 186. Latency-Configurable Streaming Speech Enhancement via Asymmetric Temporal Padding

**arXiv ID:** 2606.19688 | [PDF](https://arxiv.org/pdf/2606.19688v1)

**作者:** Yunsik Kim `[一作]` (Pohang University of Science and Technology), Yoonyoung Chung `[通讯]` (Pohang University of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种可配置延迟的双缓冲流式语音增强网络 LaCo‑SENet，能够在单一架构中实现多种算法延迟与质量的平衡；

**💡 创新点**

创新点在于利用非对称时间填充作为训练时超参数实现延迟调节，并通过双缓冲流式框架与选择性状态更新保证全序列与分块推理的一致性；

**🔧 技术方法**

采用 PrimeK‑Net 结构、非对称卷积、双缓冲与选择性状态更新技术，以及 STFT‑复杂掩模重建等方法；

**📊 数据集**

在 VoiceBank+DEMAND 语音混响数据集上进行训练与评估；

**📈 对比分析**

与 RNNoise、GaGNet、DFNet3、aTENNuate 等因果基线对比，LaCo‑SENet 在 12.5 ms 延迟下达到 PESQ 3.35，优于 46.5 ms 延迟的 3.27；通过调节填充比例可在 12.5–75 ms 延迟范围内实现 PESQ 3.35–3.43，接近非因果 PrimeK‑Net 的 3.61；

**⚠️ 局限性**

受限于单一训练超参数与固定参数量，且需要合适的分块尺寸和选择性状态更新才能保持推理一致性，尚未在更广泛的数据集或更大模型规模上验证。

---

## 187. Exploring Multi-Modal Large Language Models and Two-Stage Fine-Tuning for Fashion Image Retrieval

**arXiv ID:** 2606.19684 | [PDF](https://arxiv.org/pdf/2606.19684v1)

**作者:** Nguyen Cao Hoang `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种利用多模态大语言模型LLaVA生成属性感知的三元组，并通过两阶段微调提升时尚图像检索的框架。

**💡 创新点**

① 用LLaVA生成高质量、属性感知的标题和修正文本以增强视觉‑文本对齐；② 两阶段微调结合粗对齐与硬负采样提升细粒度区分；③ 结合句子级提示与相对字幕增强组合推理。

**🔧 技术方法**

CLIP‑ViT/B32视觉编码、LLaVA文本生成、BLIP‑2句子级提示、双编码对比损失、硬负采样、两阶段训练等技术。

**📊 数据集**

FashionIQ 数据集。

**📈 对比分析**

与 CompoDiff、SPRC、MAPNet 等 SOTA 方法对比，在 Dresses、Shirts、Tops‑Tee 等类别上取得 R@10 / R@50 较粗模型略优，但整体仍低于 MAPNet，显示出改进但未突破现有最优。

**⚠️ 局限性**

生成标题的语言多样性导致语义噪声；缺乏空间定位/注意力机制，难以识别细微局部差异；负样本规模与属性分布不均衡，训练样本有限，限制了模型泛化和精细检索性能。

---

## 188. Vortex: Multi-Modal Fusion System for Intelligent Video Retrieval

**arXiv ID:** 2606.19682 | [PDF](https://arxiv.org/pdf/2606.19682v1)

**作者:** Duc-Tho Nguyen `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Vortex多模态视频检索系统，面向Ho Chi Minh City AI Challenge 2025的四项检索任务，整合关键帧提取、OCR/ASR、多模态嵌入及交互式反馈；

**💡 创新点**

创新点在于将CLIP与SigLIP2双重嵌入通过Reciprocal Rank Fusion融合，推出多阶段时间检索与Rocchio反馈机制，并将LLM作为查询解释助手保持用户控制；

**🔧 技术方法**

使用的技术包括AutoShot+L₂过滤关键帧、Qwen2.5‑VL与Whisper进行多模态文本生成、CLIP与SigLIP2嵌入、Milvus+Elasticsearch索引、RRF融合、Rocchio反馈、Temporal Re‑ranking以及LLM辅助查询解释；

**📊 数据集**

数据集为AIC'25官方数据集，包含越南各大媒体频道的视频、关键帧、对象标注、CLIP嵌入、YouTube元数据以及时间戳映射；

**📈 对比分析**

与Baseline（仅CLIP检索）相比，Hybrid RRF提升至27.8分，加入Temporal Search + Relevance Feedback进一步提升至31.2分，总体成绩79.6/88（90.5%），在Final Round中获得Excellent总体评价，Q&A任务表现Outstanding；

**⚠️ 局限性**

局限性包括对极短时变化的捕捉可能不足、LLM建议可能导致意图漂移、以及多模态融合和时间检索在极大规模视频库中仍面临计算与实时性挑战。

---

## 189. TeleMorpher: Toward Robust Simultaneous Motion-Location Editing

**arXiv ID:** 2606.19676 | [PDF](https://arxiv.org/pdf/2606.19676v1)

**作者:** Haengbok Chung `[一作]` `[通讯]`, Haengbok Chung

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种一Shot同时编辑视频中主体动作与位置的框架TeleMorpher，利用运动先验、前景后景分离和无训练的姿态扭曲技术实现高质量、可控的运动位置编辑。

**💡 创新点**

创新点包括：①采用可控3D虚拟形象生成的运动先验代替真实参考视频；②前景-后景解耦与无训练姿态扭曲相结合，显著提升动作对齐与外观保持；③提出两种无训练姿态扭曲策略与主人公引导注入；④定义LPIPS-B与LPIPS-P两项新指标，细化评估背景保持与骨架对齐。

**🔧 技术方法**

技术手段包括扩散模型（Stable Diffusion + DDIM）、MotionEditor框架、Segment Anything进行主体分割、Inpaint Anything完成后景填充、OpenPose提取骨架、text2motion生成运动先验、无训练的姿态扭曲与轻量级Attention键值注入。

**📊 数据集**

使用20条来自YouTube与合成视频、TaiChi数据集（约400帧）进行评测，帧尺寸512×512，长度5帧。

**📈 对比分析**

与Follow-Your-Pose、ControlVideo、MasaCtrl、MotionDirector、MotionEditor等基线在LPIPS-S、LPIPS-B、LPIPS-P、CLIP以及用户主观评价上均表现更佳；尤其在背景保持、骨架对齐与语义一致性方面取得明显优势。

**⚠️ 局限性**

局限性：偶尔出现主角颜色漂移；实验受限于短视频（5帧），未能验证长序列表现；高质量编辑需较大算力，实际应用中可能需分段处理。

---

## 190. Design Considerations for Phase Modulation in Testable Photonic Systems and Co-packaged Optics

**arXiv ID:** 2606.19674 | [PDF](https://arxiv.org/pdf/2606.19674v1)

**作者:** Pratishtha Agnihotri `[一作]`, Steve Blair `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并仿真了热调制的Mach–Zehnder光学调制器与微环调制器，作为硅光子电路的可测试点与校准模块，并与载流子耗尽电调制进行对比。

**💡 创新点**

提出将热调制器嵌入DFT架构以实现低功耗、低占地、可选择性测试，并给出针对不同调制机制的性能权衡表。

**🔧 技术方法**

采用硅热光效应、载流子耗尽电场调制，利用Lumerical FDE、HEAT、varFDTD等仿真工具，并通过微加热器实现局部温度控制。

**📊 数据集**

未使用公开数据集，而是基于仿真参数（SOI波导、微加热器尺寸、波长1550 nm等）生成内部数据。

**📈 对比分析**

通过比较消光比、占地面积、电压需求、带宽、调制效率等指标，结果显示热调制器在消光比与占地上优于电调制，但带宽与响应速度远低于电调制。

**⚠️ 局限性**

热调制器的响应速度慢、热耦合与功率耗散较大，限制了其在高速数据传输和大规模集成中的直接应用。

---

## 191. Denoising Implicit Feedback for Cold-start Recommendation

**arXiv ID:** 2606.19658 | [PDF](https://arxiv.org/pdf/2606.19658v1)

**作者:** Gaode Chen `[一作]` (Kuaishou Technology), Jun Zhang `[通讯]` (Kuaishou Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

为冷启动推荐系统设计了一种去噪方法，利用内容相似的热项生成伪标签，纠正隐式反馈噪声。

**💡 创新点**

创新点在于：基于内容相似热项的伪标签生成、对伪标签加权置信度、通过相对熵与冷启动状态估计样本不确定性，动态调节标签修正。

**🔧 技术方法**

采用双塔模型、内容多模态嵌入、近邻检索、温度软化加权、相对熵与冷启动加权等技术。

**📊 数据集**

使用公开的Amazon Sports、Amazon Baby、Tiktok三大多模态推荐数据集。

**📈 对比分析**

与DECL、RINCE、MWUF等通用去噪基线以及不同骨干网络（NeuMF、LightGCN、SimGCL）对比，DIF在冷项召回和NDCG上均显著提升，尤其在冷启动场景表现突出。

**⚠️ 局限性**

局限性：在Tiktok冷项上NDCG提升有限；对排名性能关注不足；需依赖高质量内容嵌入和近邻检索，部署成本较高。

---

## 192. Vibe Coding for Visualization Implementation: An Empirical Study of Practices and Challenges

**arXiv ID:** 2606.19703 | [PDF](https://arxiv.org/pdf/2606.19703v1)

**作者:** Zhengyu Sun `[一作]` (Nanyang Technological University), Yong Wang `[通讯]` (Nanyang Technological University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对16名不同经验水平的用户使用vibe coding工具实现D3可视化的过程进行了实验研究。

**💡 创新点**

首次系统探究了可视化实现中与传统软件开发不同的提示与迭代模式，并提出了针对可视化的改进设计思路。

**🔧 技术方法**

使用基于大语言模型的vibe coding工具（如Cursor、Windsurf等）和手工编写的D3代码进行对比。

**📊 数据集**

实验采用了四个目标可视化（两个基础图、两个定制图）以及相应的简化数据集，来自D3 Observable Gallery和先前研究系统。

**📈 对比分析**

由于研究聚焦过程分析而非算法性能，没有对不同工具做量化比较，主要通过访谈、日志分析和代码审查来评估用户体验和迭代效率。

**⚠️ 局限性**

限制包括：仅研究固定的目标图形和已准备好的数据集，未覆盖设计探索与数据预处理等完整可视化工作流；样本规模有限，缺乏对更大规模、多任务场景的验证。

---

## 193. Human-on-the-Loop Orchestration for AI-Assisted Legal Discovery

**arXiv ID:** 2606.19812 | [PDF](https://arxiv.org/pdf/2606.19812v1)

**作者:** Anushree Sinha `[一作]` (Google LLC), Debanshu Das `[通讯]` (Google LLC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在电子发现任务中提出人机协作的AI调度框架，防止因误判导致的特权豁免风险

**💡 创新点**

引入“轨迹崩溃”概念，提出针对规划、推理、执行与不确定性阈值的四层验证体系

**🔧 技术方法**

采用ReAct式GPT‑4o代理、BM25检索、Monte Carlo抽样与自一致性置信度估计等技术

**📊 数据集**

使用自制的5,000条合成文档及律师标注的特权标签语料库

**📈 对比分析**

与全自动、阈值HOTL及全人工审查三种条件对比，阈值HOTL在约61%降低特权豁免风险的同时，将律师审查率降至23.7%

**⚠️ 局限性**

合成数据局限、缺乏真实数据验证、未单独消融各层贡献、GPT‑4o在处理特权文档时的数据治理与隐私挑战

---

## 194. EquiVLA: A General Framework for Rotationally Equivariant Vision-Language-Action Models

**arXiv ID:** 2606.19784 | [PDF](https://arxiv.org/pdf/2606.19784v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 195. Agentic Electronic Design Automation: A Handoff Perspective

**arXiv ID:** 2606.19795 | [PDF](https://arxiv.org/pdf/2606.19795v1)

**作者:** Jiawei Liu `[一作]` (Chinese University of Hong Kong), Bei Yu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对80余篇基于大语言模型的EDA代理系统进行系统综述，提出以“handoff validity”为核心的边界分类（Stage‑Bound、Flow‑Bound、Organization‑Bound），并设计了五层EDA Agent Communication Protocol（EACP）框架。

**💡 创新点**

创新点在于将手工有效性（handoff validity）作为统一评估维度，重新划分跨工具、跨阶段、跨组织的交付边界，并为跨边界协同提供标准化协议层级，填补了现有EDA代理研究中缺失的统一视角与协议规范。

**🔧 技术方法**

使用的大型语言模型（LLM）驱动的代理交互、脚本与工具调用技术；系统性文献检索与归纳方法；以及基于协议设计的抽象层级（Discovery、Message、Invocation、Orchestration、Security）。

**📊 数据集**

数据来源为公开发表的80篇论文及其引用的工业/学术数据集（如ICCAD、DAC、OpenROAD等），并未单独构建新的数据集，而是通过对比已有工作中的实验结果进行归纳。

**📈 对比分析**

比较方法主要采用多维度的手工有效性指标（功能验证、规范符合、性能、物理验证等）以及跨边界的可追溯性与一致性评估；虽然论文未给出统一数值指标，但通过案例分析展示不同系统在各自边界下的优势与不足。

**⚠️ 局限性**

局限性包括：EACP仍处于提案阶段，缺乏跨厂商实现与实证验证；对证据质量与治理机制的细化仍待进一步研究；依赖文献归纳，可能忽略部分非公开系统的细节。

---

## 196. Finishing Oltean's Completeness Proof in Lean 4 for Hybrid Logic $L(\forall)$

**arXiv ID:** 2606.19761 | [PDF](https://arxiv.org/pdf/2606.19761v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 197. Temporal Self-Imitation Learning

**arXiv ID:** 2606.19752 | [PDF](https://arxiv.org/pdf/2606.19752v1)

**作者:** Yinsen Jia `[一作]` (Duke University), Boyuan Chen `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Temporal Self‑Imitation Learning（TSIL），一种利用长周期机器人操作中出现的高效成功轨迹来进行自监督的强化学习框架。

**💡 创新点**

创新点在于：①通过从已成功轨迹中提取完成时间来自适应地设定配置条件的时间目标；②利用这些高效成功轨迹构建效率加权的自我模仿重放；③将时间目标与奖励、观测一起条件化，从而将时间效率直接转化为可学习的监督信号。

**🔧 技术方法**

技术包括：基于PPO的actor‑critic结构，时间目标条件化观测与奖励，配置条件的自适应时间目标更新，效率加权的自我模仿损失，以及小容量重放缓冲区的实现。

**📊 数据集**

在15个Meta‑World长周期操作任务（含装配、插入、运输、工具使用、关节物体交互等）上进行实验，这些任务在Isaac Gym/MTBench环境中通过随机初始化和目标位姿生成。

**📈 对比分析**

与标准密集奖励PPO、带步骤成本的PPO、dense‑to‑sparse调度、固定时间目标学习、适应性时间目标学习、以及带通用自我模仿的版本进行对比；TSIL在成功率、AUC、80%成功所需步骤、平均完成时间和训练期间收集的成功案例数上均显著优于所有基线，并且在加入梯度噪声、奖励丢失、PPO剪切比率/学习率扰动等不稳定训练环境下保持鲁棒性。

**⚠️ 局限性**

局限性包括：仍需要足够的探索才能首次发现成功轨迹；仅关注时间效率，未考虑安全、平滑等其他行为度量；需要额外的存储和重放机制，当前采用固定大小缓冲区，可能在高维任务中不够高效；对初始探索阶段的支持不足。

---

## 198. DeQL: A Decision Query Language for Prescriptive Analytics over Relational Data

**arXiv ID:** 2606.19751 | [PDF](https://arxiv.org/pdf/2606.19751v1)

**作者:** Matteo Brucato `[一作]` (OSM Data), Duc Nguyen `[通讯]` (OSM Data)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并规范了一种名为 DeQL 的决策查询语言，能够在关系数据库中以声明式方式定义候选集、决策变量、约束和目标，从而实现预设性决策问题的建模与求解。

**💡 创新点**

创新点在于将决策建模直接嵌入 SQL 语法，提供了候选集定义、决策列粒度、SELECTION 自动化、随机约束、内联模型评分以及多粒度联接等新语法，同时保持查询的关系闭包和可组合性，并在执行层面实现抽象化、求解器调度与优化器规划。

**🔧 技术方法**

技术实现结合了 SQL 语法扩展、抽象 IR / 具体 IR 转换、决策规划器、LP/MILP、网络流等多种求解器后端、聚类抽象、概率约束采样、PREDICT 预测函数、窗口函数约束、自动化 SELECTION 反析等。

**📊 数据集**

示例数据集包括：GPU 分配工作负载、产品库存与价格、食品营养、工人班次、批次调度、营销投入、设施定位、GPU 调度等自定义表；论文未在公开大规模数据集上进行实验评估。

**📈 对比分析**

本文为规范性文档，未给出实验结果；作者描述了可通过抽象化与逼近算法提升大规模实例求解性能，并可与多种求解器后端集成，但没有提供具体性能对比或基准测试。

**⚠️ 局限性**

限制包括：不支持多阶段随机优化、惰性约束生成、非线性求解器完整接口、多目标优化等；在语言层面对约束结构、变量粒度等有一定约束，需在后续版本中进一步扩展。

---

## 199. VFACamou: View-Fused Adversarial Camouflage for Environment-Adaptive Physical Evasion

**arXiv ID:** 2606.19736 | [PDF](https://arxiv.org/pdf/2606.19736v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 200. Beyond Uniform Forgetting: A Study of Sequential Direct Preference Optimization Across Preference Settings

**arXiv ID:** 2606.19744 | [PDF](https://arxiv.org/pdf/2606.19744v1)

**作者:** Pranav Bhandari `[一作]` (Network Analysis and Social Influence Modelling Lab), Mehwish Nasim `[通讯]` (Network Analysis and Social Influence Modelling Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对顺序DPO训练对先前偏好目标的影响进行了系统实验，分析了聚合、对例和四分位数层面的偏好变化。

**💡 创新点**

发现不同目标关系决定了遗忘、稳定、正迁移或重分布的谱，而非统一的遗忘模式，并提出对例偏好边际分析和梯度冲突诊断。

**🔧 技术方法**

使用Direct Preference Optimisation (DPO)、LoRA微调、长度归一化的政策边际、梯度余弦相似度、适配器移动分析等技术。

**📊 数据集**

四个偏好关系场景：HH-RLHF、HelpSteer2、PKU-SafeRLHF、UltraFeedback，分别对应安全-帮助冲突、多属性交互、强安全信号、兼容目标。

**📈 对比分析**

对每个阶段在所有目标上评估相对奖励边际和相对偏好准确率，并通过对例四分位数比较，显示不同顺序导致的偏好改变差异，整体表现从部分遗忘到正迁移。

**⚠️ 局限性**

仅在Llama-3.1-8B-Instruct+LoRA上实验，未验证更大规模或全参数微调；诊断仅限LoRA梯度和适配器，未覆盖激活层；缺乏动态调度和多目标自适应。

---

## 201. Interpreting Neural Combinatorial Optimization via Evolving Programmatic Bottlenecks

**arXiv ID:** 2606.19741 | [PDF](https://arxiv.org/pdf/2606.19741v1)

**作者:** Haocheng Duan `[一作]` (Carnegie Mellon University), Cathy Wu `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Evolving Programmatic Bottlenecks（EPB）框架，通过LLM自动演化程序化概念并与状态依赖路由器结合，将黑盒NCO策略解释为可读程序组合；

**💡 创新点**

创新点包括：①将概念瓶颈扩展为可执行程序；②双块优化（混合文本‑数值梯度）与动态程序库容量管理（Add/Drop）；③利用OCDR路由器实现状态动态组合；④通过程序化蒸馏提升OOD泛化；

**🔧 技术方法**

使用的技术包括LLM驱动的程序演化（TextGrad）、混合文本‑数值梯度下降、注意力路由器（OCDR）、NCO教师模型蒸馏（POMO、LEHD）以及基于程序的知识蒸馏；

**📊 数据集**

实验数据集为TSP（50/100/500）和CVRP（50/100/500）等路由问题；

**📈 对比分析**

方法与原教师模型（POMO、LEHD）在greedy解码和搜索（多起点/随机重构）下进行对比，学生在训练集上误差≤4%（搜索后≤1.25%），在更大规模（TSP‑500、CVRP‑500）OOD实例上学生甚至优于教师约15%提升；

**⚠️ 局限性**

主要限制在于路由器仍是黑盒神经网络，解释性不完整；并且EPB目前未扩展至更复杂约束的NCO场景。

---

## 202. Bidirectional Tutoring for Developmental Motor Learning in Robots: Co-Developed Interaction Dynamics Support Stable Learning

**arXiv ID:** 2606.19728 | [PDF](https://arxiv.org/pdf/2606.19728v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 203. GLARE: A Natural Language Interface for Querying Global Explanations

**arXiv ID:** 2606.19735 | [PDF](https://arxiv.org/pdf/2606.19735v1)

**作者:** Bhavan Vasu `[一作]` (Oregon State University), Rajesh Mangannavar `[通讯]` (Oregon State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个基于大型语言模型的交互式界面，允许用户用自然语言查询全局解释，并将查询转化为结构化的 SQL 语句在本地解释数据库上执行。

**💡 创新点**

创新点在于：① 将 LLM 作为语义解析器，只生成符合预定义 SQL 模板的查询，避免生成错误语句；② 采用 fence‑mask 损失掩码的合成训练，强化对解释相关 SQL 结构的学习；③ 实现零样本迁移，模型在 ADE20K 训练后即可在 Pascal VOC 等全新数据集上工作。

**🔧 技术方法**

主要技术包括：大型预训练语言模型（Gemma 2、Qwen 2.5）使用 LoRA/QLoRA 微调、SQL 模板化语法、合成数据生成、SQL 语法验证与执行、自然语言结果生成。

**📊 数据集**

使用的主要数据集是 ADE20K 作为训练与评估的全局解释来源，实验中还用 Pascal VOC 进行跨数据集迁移测试。

**📈 对比分析**

与基线（正则表达式规则）和未微调模型相比，GLARE 在内部测试集上达 95% 以上的结果匹配率；在鲁棒性测试中保持 80% 以上的匹配；在跨数据集迁移中约 90% 的准确率；在 OOD 语义/语法变化下的匹配率约 35%–40%，表明模型具备一定的泛化与鲁棒性。

**⚠️ 局限性**

局限性包括：① 只能处理预定义的 24 个 SQL 模板，无法覆盖更复杂或未见过的查询语法；② 对数据表结构高度依赖，若全局解释不符合预设 schema 需重新设计；③ 对完全不同的解释形式（非符号化的）支持有限；④ 在极端 OOD 语句或完全新语法时匹配率显著下降。

---

## 204. VOiLA: Vectorized Online Planning with Learned Diffusion Model for POMDP Agents

**arXiv ID:** 2606.19729 | [PDF](https://arxiv.org/pdf/2606.19729v1)

**作者:** Marcus Hoerger `[一作]` (Australian National University), Hanna Kurniawati `[通讯]` (Australian National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过离线模拟收集无任务依赖的数据，利用条件扩散模型学习POMDP的转移、观测采样器和似然模型，并将这些模型蒸馏为高效前馈网络，在在线阶段与GPU并行的VOPP规划器结合，实现了在不需要预先手工建模的情况下进行贝叶斯规划。

**💡 创新点**

创新点在于：①将扩散生成模型与蒸馏技术相结合，获得可在GPU上并行采样的高质量、低成本转移与观测采样器；②提出无任务依赖、可复用的POMDP模型学习框架，能够跨不同任务共享模型；③将蒸馏后的模型与向量化在线规划器VOPP结合，支持连续观测空间的逐步扩展，并通过Progressive Widening实现高效的并行搜索。

**🔧 技术方法**

核心技术包括：条件扩散模型（用于建模多模态转移/观测分布）；ODE‑based采样与前馈网络蒸馏；对比损失训练观测似然模型；VOPP（Vectorized Online POMDP Planner）与Progressive Widening；粒子滤波（Sequential Importance Resampling）进行贝叶斯更新；以及用于高维观测的编码器‑解码器（autoencoder）。

**📊 数据集**

使用三类基准环境的离线模拟数据集：FloorPositioning、LeggedNavigation和TargetFinding，每个环境收集约5万条转移样本，全部来自随机策略的高保真仿真（Isaac Sim/IsaacLab）。

**📈 对比分析**

与VTS（视觉树搜索）和Recurrent SAC（递归SAC）进行对比。实验显示：在FloorPositioning上同样实现100%成功率，仅需5万条样本；在LeggedNavigation上，5万条样本即可获得与或优于Recurrent SAC的成功率，并在未见环境中保持70–80%成功率；在TargetFinding中，完全基于仿真模型的转移/观测/似然实现了10次真实机器人试验全部成功。相比之下，VTS需要数百万样本，Recurrent SAC需要数十万样本，且在新环境中的泛化表现较差。

**⚠️ 局限性**

局限性包括：①依赖高保真仿真，仿真误差可能导致实地迁移时出现失败；②扩散模型在极高维空间下仍有采样瓶颈，虽通过蒸馏大幅降低成本，但在更复杂任务中仍需进一步优化；③VOPP对GPU并行计算依赖较强，CPU或资源受限环境下效果受限；④目前适用于离散动作空间，连续动作的直接处理仍待扩展。

---

## 205. Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning

**arXiv ID:** 2606.19808 | [PDF](https://arxiv.org/pdf/2606.19808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 206. CoRaCommit: A VS Code Extension for Commit Message Generation with Exemplar Retrieval

**arXiv ID:** 2606.19814 | [PDF](https://arxiv.org/pdf/2606.19814v1)

**作者:** Chaoran Cai `[一作]` (Wuhan University), Peng Liang `[通讯]` (Wuhan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个 VS Code 扩展 CoRaCommit，用检索增强生成（RAG）技术自动生成提交信息。

**💡 创新点**

创新点在于：①通过语义+BM25双模检索历史相似提交示例作为 Prompt，显著提升 LLM 生成质量；②支持多 LLM 并行调用，方便候选对比；③基于用户选择和编辑行为的异步反馈，动态更新 LLM 全球与示例级评分，并给出任务特定的 LLM 推荐。

**🔧 技术方法**

使用了 hybrid retrieval（CodeBERT / jina-embeddings 生成向量 + BM25 重新排序）、FastAPI 后端、Node.js 调度、OpenAI‑compatible LLM API、EMA 反馈学习等技术。

**📊 数据集**

使用了 ApacheCM 提交数据集（共 945 条样本）用于检索相似提交和评估。

**📈 对比分析**

通过统一的自动评测框架，将 CoRaCommit 与 Auto Commit Message、AI Commit、Commit Sage 在同一批 945 条样本上使用 BLEU、CIDEr、METEOR、ROUGE‑L 四个指标进行对比，CoRaCommit 在所有指标上均明显优于三者。

**⚠️ 局限性**

局限性包括：依赖 ApacheCM 的覆盖度和质量；检索效率在数据量或并发量较大时需进一步优化；动态评分和推荐需要足够历史反馈才能稳定可靠。

---

## 207. Uncertainty-Aware Reward Modeling for Stable RLHF

**arXiv ID:** 2606.19818 | [PDF](https://arxiv.org/pdf/2606.19818v1)

**作者:** Licheng Pan `[一作]` (Zhejiang University), Hao Wang `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种不确定性感知的奖励建模框架UARM，解决RLHF中奖励模型不可信预测和奖励优势统一标准化导致奖励破解的问题。

**💡 创新点**

创新点在于：①使用分位数回归与合成预测区间实现对奖励预测的条件覆盖置信区间；②将区间宽度转化为观测噪声，构造异方差优势权重，在GRPO中下调不可靠样本的影响。

**🔧 技术方法**

采用分位数回归、合成预测区间（Conformal Prediction）、异方差优势重加权、GRPO等技术。

**📊 数据集**

在HelpSteer、UltraFeedback、PKU‑SafeRLHF三个偏好数据集上进行评估。

**📈 对比分析**

与模型基础不确定性估计（如MC‑Dropout、Deep Ensemble）和分布无关区间估计（如CQR、Clear）等基线相比，UARM在R²@50、MSE@50、MAE@50等不确定性排序指标上均取得最高分，明显提升奖励预测的可靠性。

**⚠️ 局限性**

局限性包括：仅在离线奖励建模阶段验证，缺乏对在线RLHF性能的深入评估；依赖于与训练分布相同的校准集，可能在策略优化导致的分布漂移下表现不佳；未来工作需进一步扩展到更大模型与其他策略优化算法。

---

## 208. Occ-VLM: Occupancy Grounded Vision Language Model for Indoor Scene Understanding

**arXiv ID:** 2606.19776 | [PDF](https://arxiv.org/pdf/2606.19776v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 209. Clusters are All You Need: Pre-Training the Tsetlin Machine with Semantic Clusters from Language Models for Interpretability

**arXiv ID:** 2606.19815 | [PDF](https://arxiv.org/pdf/2606.19815v1)

**作者:** Jiechao Gao `[一作]` (Stanford University), Michael Lepech `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用BERT或Top2Vec生成的语义聚类进行Tsetlin Machine预训练的方法，既保留可解释性，又显著提升文本分类性能。

**💡 创新点**

创新性地设计非负Tsetlin Machine（NTM），通过增强正反馈、去除否定特征，并将聚类得到的高置信关键词直接迁移到TM，实现无监督语义知识迁移。

**🔧 技术方法**

使用BERT/Top2Vec语义嵌入、K‑means或Top2Vec聚类、非负Tsetlin Machine预训练、正向增强反馈、词/短语置信度提取以及标准TM微调等技术。

**📊 数据集**

在AG-News、NYT‑Topics、DBpedia、R8和R52这五个主题分类数据集上进行实验。

**📈 对比分析**

与BoW、char‑CNN、LSTM、FastText、BERT‑base/large、传统TM、TM+GloVe及Top2Vec等基线进行对比，预训练TM在大部分数据集上与BERT‑large相差1–2%，明显优于传统TM，整体准确率提升显著。

**⚠️ 局限性**

方法依赖聚类质量，预训练阶段需要额外离线计算，且对细粒度语义捕获的能力不及完整神经模型，低资源或噪声数据场景下性能可能受限。

---

## 210. Rethinking Sampling Strategy in Link Prediction

**arXiv ID:** 2606.19775 | [PDF](https://arxiv.org/pdf/2606.19775v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 211. The Hidden Environmental Cost of Poor Coding Practices in TensorFlow and Keras Applications: A Study on Resource Leaks and Carbon Emissions

**arXiv ID:** 2606.19799 | [PDF](https://arxiv.org/pdf/2606.19799v1)

**作者:** Bashar Abdallah `[一作]` (Polytechnique Montréal), Mohammad Hamdaqa `[通讯]` (Polytechnique Montréal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估 TensorFlow/Keras 应用中两种资源泄露代码异味（不当模型重用 IMR 与未释放张量引用 UTR）对能耗和 CO₂ 排放的影响；

**💡 创新点**

首次量化 ML 代码异味对可持续性指标的具体增幅，证明代码质量与环境成本直接相关；

**🔧 技术方法**

使用控制实验（重复测量）与 CodeCarbon 能耗估算、统计检验（配对 t 检验、Wilcoxon、Cohen d）等技术；

**📊 数据集**

CIFAR‑10 数据集上的标准 CNN；

**📈 对比分析**

将异味版本与无异味基线在相同硬件/配置下进行配对对比，结果显示 IMR 约提升 32% 能耗，UTR 约提升 46%，差异高度显著（p<0.001），无预测性能提升；

**⚠️ 局限性**

仅在单一网络结构、单一数据集、单一 GPU 上测试，且能耗采用估算而非硬件测量，样本规模有限，实验环境受限，可能影响外推性。

---

## 212. AgentFinVQA: A Deployable Multi-Agent Pipeline for Auditable Financial Chart QA

**arXiv ID:** 2606.19782 | [PDF](https://arxiv.org/pdf/2606.19782v1)

**作者:** Aravind Narayanan `[一作]` (Vector Institute), Shaina Raza `[通讯]` (Vector Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AgentFinVQA 多代理管道，专为金融图表问答设计，支持可审计、无外部 API、本地部署。

**💡 创新点**

创新点在于将规划、OCR、图例映射、颜色像素计数、视觉检查与验证组合成完整流水线，并记录 Model Evaluation Packet（MEP）实现端到端可审计；同时在开放权重模型上保持高准确率，并利用验证器置信度实现人机协作路由。

**🔧 技术方法**

采用 Prompting‑Only LLM（Gemini、Qwen3.6-27B‑FP8）+ Vision‑Language Model、CrewAI 调度、规则驱动的颜色像素计数工具、独立验证器和基于规则的分数评估。

**📊 数据集**

使用 FinMME 金融 VQA 基准（约11,000 条样本，包含条形、折线、饼图等多种图表）。

**📈 对比分析**

与匹配零拷贝基线进行对比，评估平均答案准确率和 Exact Accuracy；Gemini‑3 Flash 版提升 7.68pp（p≈1.1e‑16），Qwen3.6‑27B‑FP8 版提升 4.84pp（p≈3e‑6），MCQ 问题提升最大 8.1pp；验证器置信度可用于人机路由。

**⚠️ 局限性**

局限性包括：仅在 FinMME 上验证，未评估跨数据集泛化；开放权重版在开放式问题上无显著提升；颜色像素工具仅在 5% 数据激活，效果有限；验证器置信度未校准，未实测人机协作效率。

---

## 213. Manifold Bandits: Bayesian Curriculum Learning over the Latent Geometry of Large Language Models

**arXiv ID:** 2606.19750 | [PDF](https://arxiv.org/pdf/2606.19750v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 214. Federated Bilevel Performative Prediction

**arXiv ID:** 2606.19734 | [PDF](https://arxiv.org/pdf/2606.19734v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 215. SAC: Disaggregated KV Cache System for Sparse Attention LLMs with CXL

**arXiv ID:** 2606.19746 | [PDF](https://arxiv.org/pdf/2606.19746v1)

**作者:** Ruiyang Ma `[一作]` (Peking University), Guojie Luo `[通讯]` (Peking University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于CXL的稀疏注意力 KV 缓存分离系统（SAC），通过按需动态加载 top‑k KV 来支持长上下文推理。

**💡 创新点**

创新点：
1) 利用 CXL 的低延迟、cache‑line 级别 load/store 语义，解决 RDMA 全缓存预取导致的传输瓶颈和本地内存浪费；
2) 统一 CXL 内存资源与元数据管理，消除网络 RPC 负担；
3) 通过设备交错与 GPU 热 KV 缓冲区调优进一步提升吞吐量。

**🔧 技术方法**

技术栈：
- CXL（Compute Express Link）
- RDMA 对比基准
- SGLang + HiSparse 框架
- DeepSeek‑V3.2（AWQ 4‑bit 量化）
- 8‑H20 GPU + 2TB CXL 内存池

**📊 数据集**

使用 ShareGPT 数据集（512 条请求）进行端到端推理评测；实验覆盖 16K–128K 令牌上下文长度与 1K 生成长度。

**📈 对比分析**

比较方法：
- 两轮实验（预填充与解码），分别测量吞吐量、TTFT、TBT；
- 与 RDMA 基线、本地 DRAM 上限以及 GPU‑HBM 单机基线对比；
- 结果显示：SAC 相比 RDMA 在解码阶段提升 2.1×吞吐量、降低 9.7×TTFT、降低 1.8×TBT；与本地 DRAM 仅差 9%吞吐，且在高并发下仍保持可扩展性；
- 设备交错策略提高 9–14%吞吐，增大 GPU 热缓存也带来 10%+提升。

**⚠️ 局限性**

局限性：
- 仅在 DeepSeek‑V3.2 上评测，未覆盖 GLM‑5.1、DeepSeek‑V4 等模型；
- 对 CXL 设备与硬件配置依赖较高，未探索更大规模多机部署；
- 仍可进一步优化内存层次结构与热缓存策略；
- 仅关注 top‑k 按需访问，未考虑其他稀疏注意力访问模式。

---

## 216. Policy-aware Vector Search: A Vision for Fine Grained Access Control in Vector Databases

**arXiv ID:** 2606.19803 | [PDF](https://arxiv.org/pdf/2606.19803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 217. Flow Map Denoisers: Traversing the Distortion-Perception Plane for Inverse Problems

**arXiv ID:** 2606.19802 | [PDF](https://arxiv.org/pdf/2606.19802v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 218. TIDY: Thermal Infrared Image Denoising via Wavelet Domain Entropy and Directional Stripe Index

**arXiv ID:** 2606.19813 | [PDF](https://arxiv.org/pdf/2606.19813v1)

**作者:** Tai Hyoung Rhee `[一作]` (Seoul National University), Ayoung Kim `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为TIDY的轻量级热红外图像去噪网络，能够在室内外环境中实时去除随机噪声和固定模式噪声。

**💡 创新点**

创新点在于：①在小波域实现去噪，显著降低空间复杂度并提升速度；②引入Wavelet Entropy（WE）和Wavelet Directional Stripe Index（WDSI）两种热红外专用指标作为损失，精准抑制随机噪声和条纹噪声；③首创SCaN‑TIR真实清噪配对数据集，用于在真实噪声上训练。

**🔧 技术方法**

技术包括：离散小波变换（DWT/IDWT）、NAFNet轻量级网络、FiLM特征调制、WE与WDSI损失、Haar小波、并行卷积实现高速推理。

**📊 数据集**

使用SCaN‑TIR数据集（32.5k真实清噪对）进行训练，评估在IRE、OdomBeyondVision、Multi‑Spectral、MS²等公开数据集上做零样本推断。

**📈 对比分析**

与PPFN、DEAL、TIR‑Diffusion、DestripeCycleGAN、DeepIR等先进方法对比，TIDY在噪声抑制指标（PSNR、SSIM）上均优于竞争者，速度达到约34 Hz，几乎是现有最优方法的2–3倍，且在机器人任务（热惯性里程计、单目深度估计）中显著提升精度。

**⚠️ 局限性**

局限性包括：依赖有限规模的真实配对数据，未利用序列信息与时序一致性；对极端低对比度室内场景的细节恢复仍有提升空间。

---

## 219. DISARM: Target Electronic Device Informed Mitigation of Software Runtime Side-Channel Vulnerabilities

**arXiv ID:** 2606.19807 | [PDF](https://arxiv.org/pdf/2606.19807v1)

**作者:** Tasneem Suha `[一作]` (University of Maine), Prabuddha Chakraborty `[通讯]` (University of Maine)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种硬件-软件协同的实时侧信道漏洞修补方法，在真实嵌入式设备上测量执行时间并生成针对性的代码修复，以降低时序侧信道泄露风险。

**💡 创新点**

创新点在于将硬件真实时序测量与软件修补相结合，实现对不同设备和威胁模型的自适应补丁，只插入必要的无效指令，避免传统常量时间方法的过度修补。

**🔧 技术方法**

技术上采用控制流图分析、污点追踪、CPU周期提取（使用perf）以及基于硬件测量的BoS/LoS严重度评估，并在源代码层插入可观测的无操作指令实现补丁。

**📊 数据集**

使用22个标准加密/数值计算基准（包含Java、C/C++实现），在五款边缘设备（Jetson Nano、Orin Nano、AGX Xavier、Xavier NX、Raspberry Pi）上进行实验验证。

**📈 对比分析**

通过与PENDULUM、DifFuzzAR及常量时间方案对比，实验显示DISARM在原始程序上平均运行时提升仅2%以内，速度比PENDULUM快约40%，比DifFuzzAR快6–10%，代码行数也更少，且所有补丁均通过功能回归和泄露阈值验证。

**⚠️ 局限性**

局限性包括使用固定的无效指令填充，可能在功耗、电磁或故障侧信道下可被区分；仅支持结构化控制流，无法处理递归、间接跳转或函数指针；未考虑更强攻击者对缓存或分支预测状态的操控。

---

## 220. Data Standards for Humanoid Robotics: The Missing Infrastructure for Physical AI

**arXiv ID:** 2606.19769 | [PDF](https://arxiv.org/pdf/2606.19769v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 221. Exploring Pre-training Benefits on Phoneme Addition through Fine-tuning in Speech Synthesis

**arXiv ID:** 2606.19792 | [PDF](https://arxiv.org/pdf/2606.19792v1)

**作者:** Masato Murata `[一作]` (CyberAgent), Tomoki Toda `[通讯]` (Nagoya University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了在低资源文本到语音（TTS）任务中，预训练模型在微调时加入新音素（音素添加）过程的效果，并与从零训练进行对比。

**💡 创新点**

创新点在于：①利用大型语言模型（LLM）生成的音素受控语料，构建可控的模拟实验，排除语言、说话人等混杂因素；②在真实跨语言英语→日语设置验证实验结论；③发现预训练对音素添加帮助有限，但对自然度提升有效，挑战了传统预训练有利于所有任务的假设。

**🔧 技术方法**

技术手段包括 Conformer‑FastSpeech2 语音合成模型、HiFi‑GAN 语音合成器、wav2vec 2.0 语音识别模型评估目标音素错误率（Target PER）、UTMOS 预训练模型评估合成语音自然度；LLM（Claude Opus）用于生成受控文本，espeak‑ng 转换为 IPA 进行音素过滤；使用基于规则的过滤保证音素受控。

**📊 数据集**

使用的数据集包括：LLM 生成的 “Limited” 与 “Full” 英语文本对应的合成语料（分别用于预训练和微调），VCTK 英语多说话人数据（预训练源），JSUT 日语单说话人数据（跨语言微调目标），以及 JVS 句子用于评估测试。

**📈 对比分析**

比较方法：在不同语料大小（100、300、500、800、1000、2000 句）下，分别对微调模型和从零训练模型进行 Target PER 与 UTMOS 评估。实验结果显示：在模拟实验中，微调与从零训练在 Target PER 上相当或更差，但 UTMOS 明显更好；在跨语言实验中，微调在低资源条件下 UTMOS 高于从零训练，但 Target PER 仍不如从零训练。

**⚠️ 局限性**

局限性：①预训练音素知识在音素添加过程中利用不足，模型在微调时需保持已有知识，导致新音素学习受限；②实验仅覆盖有限的音素类型和两种语言，缺乏对更大音素库存、更多语言以及辅助损失方法的探讨；③LLM 生成的合成语料真实性有限，可能对结果产生偏差。

---

## 222. CombEval: A Framework for Evaluating Combinatorial Counting in Large Language Models

**arXiv ID:** 2606.19788 | [PDF](https://arxiv.org/pdf/2606.19788v1)

**作者:** Yuxu Zhou `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了CombEval，一个可动态生成、可控难度的组合计数基准，并用它评估LLM的组合推理能力。

**💡 创新点**

通过Typed Cofola规范与约束自适应生成，提供精确求解器验证，避免数据污染，并实现对象类型、实体规模、约束数、推理深度可调的多维度难度控制。

**🔧 技术方法**

基于Cofola声明式语言及其WFOMC求解器、模板化自然语言翻译、代码增强推理以及大规模LLM评测。

**📊 数据集**

生成的CombEval数据集，包含1,500+问题，覆盖多种对象类型与约束组合，且每题均有Solver验证答案。

**📈 对比分析**

对11款LLM（含开源与闭源）在零shot与代码增强两种设置下进行评测，结果显示模型规模与高级推理机制提升准确率，但在有序对象、不可区分元素、位置约束及嵌套依赖等场景仍显脆弱。

**⚠️ 局限性**

仅支持英文、求解器受限难度高的实例、部分问题缺乏人工验证，以及在模板生成与自然语言表达上仍有语义误差。

---

## 223. Start Right, Arrive Right: Asynchronous Execution via Initial Noise Selection

**arXiv ID:** 2606.19774 | [PDF](https://arxiv.org/pdf/2606.19774v1)

**作者:** Trong-Bao Ho `[一作]` (VinRobotics), Ngo Anh Vien `[通讯]` (VinRobotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为PAINT的无训练、无梯度的异步动作块推理方法，通过在生成前选择合适的初始噪声来实现前缀一致性。

**💡 创新点**

创新点在于把异步推理的前缀一致性问题重新表述为噪声选择问题，利用OT流匹配的局部性特性通过反向Euler逆推得到合适噪声，从而无需修改模型或进行梯度计算。

**🔧 技术方法**

使用了OT流匹配、逆ODE（反向Euler）、repainting原则，并与RTC、TE、BID等基线方法进行对比。

**📊 数据集**

在12个模拟基准环境（Kinetix）以及6个真实世界抓取/操纵任务（单臂、双臂、类人机）上评估，使用GR00T-N1.5（H=16,N=4）和π_0（H=50,N=10）等VLA架构。

**📈 对比分析**

与Naive async、TE、B-spline、BID、RTC等方法在成功率(SR)、平均执行时间(ATR)、前缀一致性(CON)等指标上比较。PAINT在多种延迟下保持或超过RTC的成功率，显著降低前缀不一致，并且不需要梯度或额外训练，计算量相对较低。

**⚠️ 局限性**

局限性包括：1）依赖噪声与动作位置的局部性，若网络跨位置混合强或多模态可能失效；2）仅使用反向Euler逆推，适用于OT流匹配，对曲率较大的扩散模型可能精度不足；3）实验仅在单一延迟设置（d≈3）下验证，未覆盖更广的延迟范围。

---

## 224. Optimal Scheduling in a Question-Answering Forum of Knowledge Workers

**arXiv ID:** 2606.19759 | [PDF](https://arxiv.org/pdf/2606.19759v1)

**作者:** Rohit Negi `[一作]` (Carnegie Mellon University), Mustafa Yilmaz `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于知识工人问答论坛的排队调度模型，并设计了可实现最大处理容量的在线调度算法。

**💡 创新点**

创新点在于将专家协同与合作结合，给出了协同调度的容量上界，并提出了两阶段贪心算法实现协同与协调模式。

**🔧 技术方法**

技术上使用了排队理论、Lyapunov 稳定性分析、图论最大独立集与超图分配问题，以及贪心调度与哈希分配算法。

**📊 数据集**

使用了StackExchange 37 个技术站点的真实提问/回答数据（约 1.8M 条提问和 270K 条回答）以及模拟的专家回答时延样本。

**📈 对比分析**

通过与无协调随机分配、以及在模拟数据下的平均队列长度比较，显示在线贪心调度在负载低于容量阈值时能保持队列稳定，且协同模式容量可比单独协调提升数倍。

**⚠️ 局限性**

局限性包括协同分配问题的 NP 难度、对专家回答时延分布的几何假设、以及缺乏真实协同数据的验证。

---

## 225. Grounded Inference: Principles for Deterministically Encapsulated Generative Models

**arXiv ID:** 2606.19753 | [PDF](https://arxiv.org/pdf/2606.19753v1)

**作者:** Marty O'Neill `[一作]` `[通讯]`, Marty O'Neill

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于四大原子推理原语的生成式模型安全封装框架，并给出了两大反模式与对应的防护微模式与参考架构

**💡 创新点**

首次将生成式模型与传统系统分离为：概率引擎、模型封装器、状态注册表与确定性编排器四个原子，构建了可验证、可审计的封装边界，系统化阐述了两大反模式并提供了可落地的参考实现

**🔧 技术方法**

结合概率推理引擎（如LLM）、传统编排脚本（Python/Kubernetes）、验证层（Pydantic/语法检查）、状态仓库（数据库/向量存储）以及对接外部文档的查询与校验，采用Prompt+Prompt‑Engineering与重试循环等技术

**📊 数据集**

未公开具体数据集；示例使用企业内部的混合结构化/非结构化日志、内部wiki、API文档与样本数据作为验证输入

**📈 对比分析**

论文未给出实验对比或量化性能指标，主要以理论分析与案例描述为主，指出在大规模GPU集群下即使设置温度为0也无法实现位级可重复性，且上下文窗口限制导致token成本和精度折衷

**⚠️ 局限性**

受限于LLM上下文窗口、隐藏的服务层状态、硬件级浮点不确定性以及模型更新导致的漂移，封装方案对token成本和实时性有较高要求，且在高度动态的生产环境中仍需人工干预或外部验证以保证安全

---

## 226. Challenges to Grassroots Organization Engagement with AI Policy

**arXiv ID:** 2606.19816 | [PDF](https://arxiv.org/pdf/2606.19816v1)

**作者:** Carter Buckner `[一作]` (Queer in AI), B. V. Alaka `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过案例研究的方式，记录并分析了Queer in AI组织在美国AI政策制定中的参与过程，运用参与式设计（PD）原则创建了面向跨性别者的AI政策解释器，并在NIST AISIC等机构中推动对性别多样性与AI公正的讨论。

**💡 创新点**

创新点在于：①首次将参与式设计方法系统化应用于AI治理议题，尤其聚焦于酷儿群体；②构建了可共享的政策解释器工具，将社区经验与研究成果转化为可直接用于政策讨论的文档；③通过联盟与跨组织协作，提升了非主流组织在顶层治理中的可见度与话语权。

**🔧 技术方法**

使用技术包括：协作自传民族志（collaborative autoethnography）记录内部经验；Slack、Zoom、Google Docs等在线协作平台支持异步与同步讨论；Chatham House Rule等会谈规则确保信息共享与匿名性；以及政策文档管理与版本控制系统。

**📊 数据集**

数据来源主要为：①社区成员的定性反馈与调查结果（如人口统计学问卷、政策解释器草稿的评审意见）；②美国相关AI政策文件与立法文本（如Biden EO 14110、NIST AI RMF等）；③在NeurIPS、工作坊等场合收集的现场讨论记录与评论。

**📈 对比分析**

本文并未进行传统意义上的算法性能对比；其“比较”主要是对比组织在参与过程中的影响力、文档采纳率与政策讨论的深度。通过对比内部反馈与外部政策文件的修改，展示了参与式设计在提升政策表达和被采纳可能性方面的作用，但未给出定量指标。

**⚠️ 局限性**

局限性包括：①资源有限，志愿者人力不足导致参与频率与深度受限；②组织成员主要分布在美国，缺乏对全球酷儿社群的充分代表性；③政策机构的透明度与响应机制不充分，使得反馈往往被忽视或延迟；④地理与技术障碍限制了跨时区协作，进一步削弱了基层参与的覆盖面。

---

## 227. ParaScale: Scale-Calibrated Camera-Motion Transfer via a Gauge-Invariant Parallax Number

**arXiv ID:** 2606.19805 | [PDF](https://arxiv.org/pdf/2606.19805v1)

**作者:** Zijie Meng `[一作]` `[通讯]` (Peking University), Zijie Meng (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 ParaScale 方法，通过在每帧对参考视频的平移量进行深度校准，恢复目标视频的视差尺度，以实现跨尺度的相机运动迁移。

**💡 创新点**

创新点包括：① 定义了尺度不变的 Parallax Number Π = ‖ΔT‖/，阐明其是相机运动迁移的正确不变量；② 开发了无训练、生成器无关的 ParaScale 插件；③ 提出了尺度对称的 PCE 指标，用于直接衡量迁移过程的尺度保真度。

**🔧 技术方法**

技术手段包括：单目结构光学运动估计（SfM）提取位姿；单目深度估计获取中位深度；基于深度的每帧平移比例校准；将校准后的位姿输入到现有的姿态条件扩散模型（Plücker 或 RT‑matrix）中进行图像生成。

**📊 数据集**

使用了跨四个尺度范畴的参考视频数据集：桌面级、室内/人体级、建筑级以及宇宙/航拍级；目标生成采用 Wan2.1 背骨的 Plücker‑conditioned 控制器。

**📈 对比分析**

对比方法包括 Raw transfer、Global‑norm、MotionMaster、CameraCtrl II 的 Scale‑Calib；评估指标为 PCE、TransErr、RotErr、FVD、CLIP‑SIM。ParaScale 在所有尺度下将 PCE 降低 3 倍以上，TransErr 与 RotErr 与 baseline 接近或更好，且 FVD 与 CLIP‑SIM 均不下降。

**⚠️ 局限性**

局限性：依赖单目深度估计，深度误差会直接影响平移比例；假设场景为刚体，难以处理动态物体或非均匀尺度变换。

---

## 228. HypOProto: Hyperbolic Ordinal Prototypes for Left Ventricular Filling Pressure Classification

**arXiv ID:** 2606.19804 | [PDF](https://arxiv.org/pdf/2606.19804v1)

**作者:** Victoria Wu `[一作]` (University of British Columbia), Teresa S. M. Tsang `[通讯]` (Vancouver General Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了一种基于超曲线原型的可解释左室充盈压（LVFP）分类框架 HypOProto，利用冻结的 DINOv3 特征将 B‑mode 回声视频映射到超曲线空间，并通过原型对齐实现基于 E/e' 的序数判别。

**💡 创新点**

创新点包括：① 将原型按 E/e' 量表在超曲线半径上排序，显式编码临床序数关系；② 引入 HyperPAS 损失，在超曲线空间实现不同类别原型的角度分离；③ 在冻结的可解释基模型上直接学习原型，避免了全端到端训练的算力瓶颈。

**🔧 技术方法**

技术方法包括：冻结 DINOv3 回声特征提取器、空间-时间注意力选取 ROI、基于预测 E/e' 的半径映射到 Lorentz 超曲线、原型分类与距离计算、交叉熵、MAE、L1 与 HyperPAS 损失的联合优化。

**📊 数据集**

使用了 141,086 条独立 cines（32,818 研究）来自某三级中心的私有回声数据集，LVFP 标签通过临床文本报告导出，正常/升高比例约 85%/15%。

**📈 对比分析**

与四种基线（EchoPrime+MLP、DINO+MLP、Proto+DINO、Akerman 等）比较，HypOProto 在 cine 级别和 study 级别均实现了最高的整体准确率和 F1，尤其在升高类 F1 方面提升显著（如 cine 级别升高 F1 0.63，优于 0.56/0.56/0.56/0.56）。

**⚠️ 局限性**

局限性包括：① 依赖冻结特征，可能限制对新设备或新视角的适应性；② LVFP 标签与 E/e' 门限不一致的样本导致误差，尤其在升高与正常边界附近表现差；③ 数据集偏向正常病例，未包含第三个不可确定类别，导致模型在边缘病例上的解释性受限。

---

## 229. An Information Theoretic Framework for Graph Novelty Generation via Latent Mixture Modeling

**arXiv ID:** 2606.19770 | [PDF](https://arxiv.org/pdf/2606.19770v1)

**作者:** Itsuki Nakagawa `[一作]` (University of Tokyo), Kenji Yamanishi `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于信息论的图结构新颖生成框架，利用MDL准则在潜在空间中实现可控的创新与可靠性。

**💡 创新点**

创新点：①将新颖性与可靠性分别用局部与全局描述长度约束；②在潜在空间中以有限混合模型（vMF）建模，并用MDL引导采样；③提供误分类概率的理论上界并证明指数收敛。

**🔧 技术方法**

技术：图自编码器（GAE）编码，有限混合模型（vMF‑mixture）与EM估计，MDL（NML 与 DNML）代码长度，MDL引导的拒绝采样。

**📊 数据集**

数据集：合成 SBM 数据；公开图数据集 Amazon Computers 与 Coauthor Physics。

**📈 对比分析**

对比方法：MDL、LL（负对数似然）与 KL（KL 散度）版本；实验显示MDL在新颖性/可靠性指标上相关性更高，可靠性控制更稳健，整体性能优于 LL 与 KL。

**⚠️ 局限性**

局限：仅在低维潜在空间（≈8）有效，阈值选择经验性，缺乏对高维扩展与多模态数据的探索。

---

## 230. SafeSpec: Fast and Safe LLM via Dynamic Reflective Sampling

**arXiv ID:** 2606.19755 | [PDF](https://arxiv.org/pdf/2606.19755v1)

**作者:** Haotian Xu `[一作]` (Zhejiang University), Cheng Zhuo `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SafeSpec，一种安全感知的投机式推理框架，能够在保持加速的同时有效检测并纠正生成中的安全风险。

**💡 创新点**

核心创新包括：① 在目标模型上加入轻量化安全头，能够在一次前向传播中同时评估语义质量和安全性；② 采用回滚‑反射机制，利用反射提示引导模型重新采样；③ 引入安全导向多采样策略，在检测到不安全片段时主动搜索安全路径，避免硬拒绝导致的过度拒绝。

**🔧 技术方法**

技术手段：投机式推理（draft‑verify）、轻量安全头（两层MLP）、反射提示（Self‑Reflection Prompt）、安全导向多采样、阈值控制、概率视角分析、对抗训练（使用 Qwen3Guard‑Gen‑8B 标注）。

**📊 数据集**

使用的数据集和模型：多种 jailbreak 攻击集（ABJ、Atten.、Code.、Deep.、Rene.、Mouse.、H‑CoT）；XSTest 评估过度拒绝；标准推理基准（GSM8K、MATH、GPQA）；模型以 Qwen3‑32B（Draft‑Qwen3‑1.7B）和 DeepSeek‑R1‑Distill‑Llama‑70B（Draft‑8B）为主。

**📈 对比分析**

与 SpecDecoding、Specreason、SafeDecoding、SecDecoding、Self‑Reminder、RPO 等基线进行对比，SafeSpec 在 Qwen3‑32B 上平均 ASR 降至 7%（≈15% 降幅），过度拒绝率 10%，并在无攻击工作负载上实现 2.06× 的加速；在 DeepSeek‑70B 上平均 ASR 7%，过度拒绝 12%，加速 1.76×，保持了与 Specreason 相近的准确率。

**⚠️ 局限性**

局限性：安全头存在误报，导致反射/多采样开销；阈值设置需手动调优；对极端或未知攻击的鲁棒性仍有限；仅适用于投机式推理框架，需额外训练安全头，增加部署复杂度。

---

## 231. Designing for Interconnected Islamic Learning: A Qualitative Study of Muslim Women's Experiences with Qur'an, Hadith, and Seerah Apps

**arXiv ID:** 2606.19745 | [PDF](https://arxiv.org/pdf/2606.19745v1)

**作者:** Ishrat Jahan Easha `[一作]` (University of Technology Sydney), Riasat Islam `[通讯]` (Greentech Apps Foundation)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对五名穆斯林女性的半结构化访谈，研究了伊斯兰学习应用如何将古兰经、圣训与圣史等文本进行整合，并提出了层级式语境化的概念框架。

**💡 创新点**

创新点在于提出层级语境化这一HCI问题，将语境扩展与解释责任、祈祷流程以及跨设备连续性相结合，并首次以女性视角聚焦数字伊斯兰学习。

**🔧 技术方法**

采用了半结构化访谈与反思性主题分析的质性方法进行数据收集与分析。

**📊 数据集**

数据集为5名来自英国和孟加拉、受教育程度高、活跃于在线伊斯兰学习社群的穆斯林女性的访谈记录。

**📈 对比分析**

该研究未进行系统实现或对比评估，缺乏量化性能指标，只提供定性洞察。

**⚠️ 局限性**

局限性包括样本规模小、仅涵盖受教育程度高且网络活跃的 Sunni 女性，缺乏多元背景、观察性实验或跨平台评估，结果不具统计推广性。

---

## 232. Training-Free Metrics for Synthetic Object Detection Data: A Proxy for Detector Performance

**arXiv ID:** 2606.19817 | [PDF](https://arxiv.org/pdf/2606.19817v1)

**作者:** Myeongseok Nam `[一作]` (GenGenAI), Seungwook Kim `[通讯]` (GenGenAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无训练的度量方法 Conditional-Composition Domain Match（CCDM），用于评估合成训练集对目标检测性能的相对效用。

**💡 创新点**

创新点在于将合成与真实数据的域匹配转化为基于每图元数据（物体数量、框面积、主类）的分层处理，将分层内的外观匹配与分层间的组成差异（Jensen‑Shannon 散度）相结合，从而显著提升对下游检测性能的预测能力。

**🔧 技术方法**

采用预训练特征（CLIP 或 DINOv2），将传统特征空间距离（如 MMD、FID、SWD 等）包装进分层框架，计算条件距离、全局距离与组成散度，并组合成最终指标。

**📊 数据集**

使用 VisDrone‑DET 数据集，并通过 FLUX.1‑dev + LoRA 生成四个合成数据池（共 6,471 张图），与完整的真实训练集进行对比。

**📈 对比分析**

与传统训练无关指标（FID、KID、MMD 等）相比，CCDM‑MMD_CLIP 在 5 个候选训练集上与 YOLOv8 训练得到的 mAP 排名完全一致，Spearman 相关系数达到 1.0，传统指标相关性仅为 0.2 或更低，甚至出现负相关。

**⚠️ 局限性**

局限性包括：仅在目标检测任务中验证，未对实例/全景分割等任务评估；分层数目需满足最小样本阈值，过细分层会导致估计噪声；目前仅在 VisDrone‑DET 上验证，缺乏对更大或不同域数据集的泛化评估。

---

## 233. The Orchestration Gap: Why Process Automation Stalls in Operationally Complex Industries

**arXiv ID:** 2606.19790 | [PDF](https://arxiv.org/pdf/2606.19790v1)

**作者:** Jiechao Gao `[一作]` (Stanford University), Michael Lepech `[通讯]` (Stanford University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文阐述在物流、医疗、建筑等高度碎片化行业中，agentic 系统难以落地是因为缺乏统一的工作流编排抽象。

**💡 创新点**

创新点在于提出了“编排抽取”框架，定义了可检验的编排边界测试、碎片度量、协调开销和可恢复价值，并将行业摩擦映射到必需的编排保障。

**🔧 技术方法**

采用了多方位方法，包括流程映射、指标分解、案例分析和对比框架的功能评估。

**📊 数据集**

使用了公开行业报告和问卷数据（如 McKinsey、BCG、Gartner）以及行业内的成本与效益案例。

**📈 对比分析**

通过将编排需求与现有多智能体框架（AutoGen、MetaGPT、LangGraph 等）对比，发现它们在域知识、硬约束和灰度场景集成方面缺乏系统化保障；论文并未给出具体性能数值。

**⚠️ 局限性**

限制在于缺乏实证部署数据、对各行业摩擦的定性评估、对编排收益的阶段化测量以及对不同技术方案的直接性能对比。

---

## 234. ORAgentBench: Can LLM Agents Solve Challenging Operations Research Tasks End to End?

**arXiv ID:** 2606.19787 | [PDF](https://arxiv.org/pdf/2606.19787v1)

**作者:** Jiajun Li `[一作]` (Southeast University), Wanyuan Wang `[通讯]` (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ORAgentBench，一个针对可执行端到端运营研究（OR）任务的基准，包含107个多文件、隔离环境的真实场景，并设计了完整的验证与评分流程；

**💡 创新点**

创新点在于：①把 OR 的建模、求解与验证完整包装成可执行任务，突破以往只评估文本到公式或预先正式化实例的局限；②通过六维难度构建准则（建模与求解策略、公式结构、约束耦合、动态结构、数据规模、问题理解）系统化刻画任务难度；③构建可自动验证且可重现的评估体系，强调方案质量与可行性并重；

**🔧 技术方法**

技术包括：多轮交互式 LLM 框架（Claude、GPT-5.x 等）与自适应求解器（SCIP 等）集成的 agent harness；隐藏的任务特定 validator，用于判定输出模式、硬约束可行性和目标质量；实验环境基于 Harbor 的隔离 Docker，支持 45 分钟交互+5 分钟求解时间；

**📊 数据集**

数据集由来自学术论文、工业案例和公开基准（如 MIPLIB、CVRPLIB 等）合成的 107 个任务组成，每个任务包含多文件数据、配置、自然语言说明及隐藏验证文件；

**📈 对比分析**

比较方法：对 14 个前沿模型‑agent 组合在全部任务上进行端到端评测，记录可行率、质量分数与通过率；最佳模型（GPT‑5.4）通过率仅 35.51%，硬任务仅 20.59%；通过率与质量阈值敏感，表明模型虽能生成可行解但往往质量不达标；

**⚠️ 局限性**

局限性在于：①目前 agent 仍难以可靠完成真实 OR 任务，主要瓶颈为建模策略与规则理解不足；②缺乏对动态与多周期约束的有效求解机制；③基准主要聚焦于单一实例执行，尚未覆盖长期持续优化与交互式调优场景；

---

## 235. Beyond Entropy: Learning from Token-Level Distributional Deviations for LLM Reasoning

**arXiv ID:** 2606.19771 | [PDF](https://arxiv.org/pdf/2606.19771v1)

**作者:** Xuanzhi Feng `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ICT框架，在RLVR中只对分布上独特的token进行稀疏梯度更新，以改进大语言模型的推理探索。

**💡 创新点**

创新点在于使用Jensen‑Shannon散度挑选“独特”token作为关键分支点，并通过稀疏更新解决传统全token更新导致的熵崩塌与熵爆炸两端的不稳定。

**🔧 技术方法**

采用Jensen‑Shannon散度、第二阶Renyi熵、GRPO、Sparse‑GRPO等信息论与强化学习技术，并在Qwen2.5系列LLM上实现。

**📊 数据集**

使用GSM8K、MATH（Math500）、MMLU‑Stem、GPQA、AIME23/24/25等七个数学/常识/奥赛类基准数据集。

**📈 对比分析**

与GRPO、20‑Entropy、STAPO等基线对比，ICT在所有模型规模和基准上平均Pass@4提升约4.5%，单模型最大提升14.9%；仅更新10%独特token即可获得相当或更佳的性能。

**⚠️ 局限性**

局限性包括：仅在离散token任务上验证，连续域或更大模型的适用性尚未探究；稀疏更新比例对不同任务的最佳取值仍需进一步调优。

---

## 236. SIGMA: Skill-Incidence Graphs for Compositional Multi-Agent Design

**arXiv ID:** 2606.19758 | [PDF](https://arxiv.org/pdf/2606.19758v1)

**作者:** Kun Zeng `[一作]` (Sun Yat-sen University), Xiaoying Tang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Sigma，一种基于技能发生矩阵的多代理系统框架，通过任务条件化的技能集合构建代理节点并解码通信拓扑；

**💡 创新点**

创新点在于将代理能力与通信拓扑分离，利用可重用技能卡构成可执行的任务特定代理，并通过技能邮箱实现运行时消息路由；

**🔧 技术方法**

使用冻结文本编码器对任务和技能进行嵌入，训练技能-代理发生器与技能感知拓扑解码器，并结合硬稀疏投影与软注意力；

**📊 数据集**

在六个基准上评估：MMLU、GSM8K、SVAMP、MultiArith、AQuA 与 HumanEval；

**📈 对比分析**

与单代理、LLM-Debate、GPTSwarm、CARD、ARG-Designer 和 G-Designer 等方法对比，Sigma 在 18 个模型-任务组合中排名第 1 的 16 次，平均提升 2–3 分，且在未见技能库下仅下降 0.96 分，令其在效率与鲁棒性上均表现优异；

**⚠️ 局限性**

局限包括依赖技能库的完整性与质量、伪标签生成的偏差、固定代理槽数、仅在标准基准上测试以及对实时冲突解决与动态规模预测尚未覆盖。

---

## 237. Learning universal approximations for partial differential equations with Physics-Informed Broad Learning System

**arXiv ID:** 2606.19754 | [PDF](https://arxiv.org/pdf/2606.19754v1)

**作者:** Zhiwen Yu `[一作]` (South China University of Technology), C. L. Philip Chen `[通讯]` (South China University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于广义学习系统的物理信息化求解器（PIBLS），通过随机特征与线性输出权重一次性求解偏微分方程，既保持极快的求解速度，又实现几乎机器精度的解。

**💡 创新点**

创新点包括：① 将BLS（仅训练输出层）应用于PDE求解，消除梯度训练瓶颈；② 将求解过程改写为闭式最小二乘优化，线性和非线性方程分别提供直接求解与增强非线性最小二乘扰动算法；③ 给出PIBLS在Sobolev空间内的通用逼近定理，理论上保证任意光滑解可逼近。

**🔧 技术方法**

核心技术：随机特征映射（特征节点、增强节点）+线性输出权重（Moore–Penrose伪逆求解）+解析导数与物理约束残差构造 + 对非线性PDE的增强NLSQ-扰动优化（TRF+随机扰动）。

**📊 数据集**

使用人工制造解的 11 个基准 PDE（TC‑1~TC‑11），包括一维/二维稳态线性、时间相关线性、以及非线性 Helmholtz、弹簧振子、布格方程等。

**📈 对比分析**

与标准 PINN（浅层与深层）、Physics‑Informed ELM（PIELM）、局域 ELM（locELM）以及高分辨率有限元（FEM）比较。实验显示：PIBLS 在所有测试中误差均低于 10⁻¹²（线性）甚至 10⁻¹⁵（非线性），与 FEM 相比误差降低 4–5 位，计算时间比 PINN 快 1–3 个数量级，整体实现了机器精度与极快速度的统一。

**⚠️ 局限性**

局限性：对非线性问题的性能依赖于随机权重初始化范围，需要适当调参；在极大规模或高维复杂域时，随机特征数目仍需增长，可能导致内存与计算负担；目前未对自适应网格或自适应随机化策略进行探索。

---

## 238. Benchmarking Agentic Review Systems

**arXiv ID:** 2606.19749 | [PDF](https://arxiv.org/pdf/2606.19749v1)

**作者:** Dang Nguyen `[一作]` (University of Chicago), Chenhao Tan `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了三款 AI 评审系统（OpenAIReview、未知开源系统和 Reviewer3）以及零射击基线，使用六种 LLM 进行两项基准测试：①基于 ICLR/NeurIPS 论文的质量代理评估；②对八类 arXiv 论文注入四类错误的扰动基准；并在公开部署中收集用户投票与评论解决率。

**💡 创新点**

①首次系统性对比多种 AI 评审系统与不同 LLM 的性能；②提出扰动基准以量化错误检测召回；③展示不同模型互补、联合召回提升的潜力；④在真实用户环境验证系统正面影响。

**🔧 技术方法**

利用多代理结构、跑动摘要、分段上下文提示的 LLM 守护；对评论进行模糊子串匹配并用 LLM 判断错误一致性；对评论嵌入进行聚类；在部署中统计点赞/点踩与解决率；使用 LLM 生成与验证扰动。

**📊 数据集**

①ICLR/NeurIPS 2021–2022 论文（与 Semantic Scholar 连接，提供引用、接受/拒绝、评分）；②来自八个 arXiv 主题（从 74 篇论文中抽样，用于扰动基准）；③公开部署收集的 1,100 篇论文（覆盖 CS/AI、社会科学、生命科学、物理学等领域）。

**📈 对比分析**

采用 pairwise accuracy（论文质量代理下的评论量差异）和错误检测召回率进行对比。最佳配置 OpenAIReview+GPT‑5.5 在 pairwise accuracy 上达 83%；在扰动基准中单模型召回 71.6%，模型联合召回 83.3%；部署中点赞/点踩比 1.44:1，解决率 24%/41%。

**⚠️ 局限性**

仅评估召回率，未直接量化精确度；扰动生成过程受 LLM 偏见影响，可能不代表专家关注的错误；部署反馈缺乏专家标注的错误基准；未覆盖所有可能的错误类型；模型规模与计算成本高，实际部署受限。

---

## 239. A Comparative Study of Pretrained Transformer Models for Quranic ASR: Speech Representations, Label Formats, and Dataset Composition

**arXiv ID:** 2606.19747 | [PDF](https://arxiv.org/pdf/2606.19747v1)

**作者:** Nabil Mosharraf Hossain `[一作]` (Greentech Apps Foundation), Unaizah Obaidellah `[通讯]` (University of Malaya)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对古兰经语音识别进行预训练Transformer模型的系统性消融与比较，探究特征提取、标签格式、训练策略和剪辑长度对性能的影响；

**💡 创新点**

首次通过消融实验确定最佳配置（Wav2Vec2‑XLSR‑53+无Tashkeel标签+30秒剪辑），显著降低WER并实现全古兰经覆盖的基准，提出可扩展的端到端ASR框架；

**🔧 技术方法**

使用预训练的自监督模型Wav2Vec2、HuBERT、XLS‑R作为特征提取器，结合Transformer解码器与CTC损失，并与传统Citrinet模型进行对比；

**📊 数据集**

融合专业朗读集EveryAyah（约870小时）和用户朗读集Tarteel（约54小时），共计约924小时；

**📈 对比分析**

与Citrinet基线对比，Wav2Vec2在EveryAyah上实现WER 0.08，混合数据实现WER 0.11，较基线提升约5个百分点；训练时间从140小时降至40小时，显示出显著的效率与精度提升；

**⚠️ 局限性**

模型参数大约1亿，模型体积1.2GB，推理延迟高，难以部署到移动设备；缺乏音素级与塔吉威德规则的显式建模，仍存在相似辅音混淆和短句识别错误。

---

## 240. QueryGaussian: Scalable and Training-Free Open-Vocabulary 3D Instance Retrieval

**arXiv ID:** 2606.19733 | [PDF](https://arxiv.org/pdf/2606.19733v1)

**作者:** Xiuyuan Zhu `[一作]` (University of Chinese Academy of Sciences), Dongming Zhang `[通讯]` (State Key Laboratory of Communication Content Cognition)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了QueryGaussian，一个无需训练的框架，能够在大规模3D高斯场景中通过自然语言查询快速检索特定实例。

**💡 创新点**

核心创新在于将语义理解与几何表示分离，采用最大权重关联将2D分割结果直接映射到3D，从而消除了全场景语义嵌入的内存与计算瓶颈。

**🔧 技术方法**

使用了预训练的2D开放词汇分割模型（如Grounded‑SAM）进行语义获取、基于最大权重的2D→3D关联以及时间融合与多阶段密度聚类的自适应后处理。

**📊 数据集**

在三类小规模室内场景（约10⁵个高斯）和两类大规模户外场景（约10⁷个高斯）上进行评估，数据来自LERF、MatrixCity和Rubble等公开数据集。

**📈 对比分析**

与场景级嵌入方法（LangSplat、OpenGaussian、Gaussian Grouping）比较，QueryGaussian在小规模场景中实现最高平均IoU（0.6380），在大规模场景中仍能得到0.7676 IoU，且显著降低内存占用（~7GB）并提高查询速度（~20–60秒）。

**⚠️ 局限性**

局限性包括对底层3D高斯场景重建质量的依赖，以及在极端遮挡或低质量分割情况下可能仍出现误检，需要进一步提升鲁棒性和处理非视觉属性的能力。

---

## 241. Prompt, Plan, Extract: Zero-Shot Agentic LLMs Workflows for Lung Pathology Extraction from Clinical Narratives

**arXiv ID:** 2606.19852 | [PDF](https://arxiv.org/pdf/2606.19852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 242. EVM Workloads in the Wild: Evidence for Multi-Dimensional Gas Metering, State Growth, Delayed Execution, and Parallelism

**arXiv ID:** 2606.19869 | [PDF](https://arxiv.org/pdf/2606.19869v1)

**作者:** Lioba Heimbach `[一作]` (Category Labs), Jason Milionis `[通讯]` (Category Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对2025年以太坊（L1）与Base（L2）区块链进行每日电报级抽样（每天3000个区块），对每笔交易进行opcode层级的 gas 细分、状态增量记录，并在9月对交易进行多状态重执行以评估 gas 估算与执行结果的状态敏感性；同时对合约地址和交易进行多维标签，按 MEV、DeFi、Token 等业务类别进行统计；进一步量化了状态增长、冷/热读取比例及存储槽访问变异性。

**💡 创新点**

首次将跨链（L1 vs L2）和跨时间（一年内）的大规模、细粒度 gas 与状态变化测量相结合；通过重执行不同历史状态评估 gas 估算误差与执行差异；提出多维 fee 市场与显式状态定价的实证依据。

**🔧 技术方法**

使用 EVM 轨迹记录（eth_traceTransaction）、历史状态查询（eth_getBlockByNumber + eth_getTransactionReceipt）以及自研的深度消耗回溯算法；采用链上标签源（Spellbook、DefiLlama、Kleros）并补充手工标注；利用统计工具（Jaccard 相似度、系数变异）分析状态敏感性。

**📊 数据集**

采样数据来自以太坊和 Base 的归档节点，覆盖 2025 年全年，每天 3000 个随机区块（约 42% 的以太坊区块、6.9% 的 Base 区块）；在 9 月对所有交易进行多状态重执行；标签覆盖超过 60% 以太坊、75% Base 的交易与地址。

**📈 对比分析**

通过比较每个交易在不同历史状态下的 gas 估算、总 gas 消耗、opcode 组别消耗及存储槽读写重叠率，使用系数变异（CV）和 Jaccard 相似度衡量差异；结果显示 Base 交易的 gas 估算差异率为 46% 而以太坊为 13.9%，存储读写重叠率在 Base 上显著下降；该方法揭示了状态漂移对 gas 估算与执行可预测性的实质影响。

**⚠️ 局限性**

实验仅覆盖 2025 年 9 月的重执行，样本量受计算成本限制；重执行不涵盖执行失败交易，导致状态敏感性被低估；使用随机抽样可能无法完全覆盖极端工作负载；仅针对以太坊与 Base 两条链，结果未必能推广至所有 L2 或 L1。

---

## 243. PSCT-Net: Geometry-Aware Pediatric Skull CT Reconstruction via Differentiable Back-Projection and Attention-Guided Refinement

**arXiv ID:** 2606.19867 | [PDF](https://arxiv.org/pdf/2606.19867v1)

**作者:** Dong Yeong Kim `[一作]` (Seoul National University), Young-Gon Kim `[通讯]` (Seoul National University Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于几何感知的PSCT-Net框架，从双平面X射线重建3D儿童颅骨CT；

**💡 创新点**

引入可微分背投影构造空间先验、Attention‑Guided Projection学习非线性像素‑体素对应，以及线性复杂度的Bidirectional Mamba实现全局上下文建模；

**🔧 技术方法**

可微背投影、注意力引导投影模块、双向状态空间模型、3D cGAN、LSGAN判别器等；

**📊 数据集**

自建的982例儿童颅骨CT数据库PedSkull‑CT，以及LIDC‑IDRI、CTSpine1K、CTPelvic1K等公开数据集；

**📈 对比分析**

在所有公开基准和私有PedSkull‑CT上均优于现有SOTA（如DiffuX2CT、X2CT‑GAN），在LIDC‑IDRI上PSNR达27.18dB、SSIM0.671、LPIPS0.100，PedSkull‑CT上PSNR31.49dB、SSIM0.882、LPIPS0.100；

**⚠️ 局限性**

仍难以恢复亚毫米细节（如细骨缝）以及受限于固定体素分辨率，后续计划采用补丁细化或隐式神经表示进一步提升精度。

---

## 244. The Almost Intelligent Revolution: Options for Scaling Up Deliberation and Empowering People with AI

**arXiv ID:** 2606.19864 | [PDF](https://arxiv.org/pdf/2606.19864v1)

**作者:** Serge Sharoff `[一作]` `[通讯]` (University of Leeds), Serge Sharoff (University of Leeds)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述了大型语言模型在民主协商中的应用、机遇与挑战，结合功能语法视角探讨了语言多样性与社会分层问题。

**💡 创新点**

创新点在于将功能语法理论与LLM技术结合，提出以LLM为媒介的“Habermas Machine”与“iDem”项目，探讨了语言障碍缓解与多元声音纳入的可行路径。

**🔧 技术方法**

使用的技术包括大型语言模型（如Chinchilla、XLM‑R、Salamandra）、RLHF、RAG、文本简化、故事化对话代理和人机协同的监管机制。

**📊 数据集**

数据集涵盖了GPT‑3训练语料（约5000亿词）、XLM‑R 3亿参数训练集、欧盟iDem项目的多语言政府文件以及欧洲议会和联合国文本。

**📈 对比分析**

通过与人工调解员对比，Habermas Machine生成的群体陈述在质量、清晰度、信息量和公平性上优于人类，且提升了共识率；iDem项目证明大约92–96%的官方文本需要简化，LLM能有效生成可理解版本。

**⚠️ 局限性**

主要局限包括训练数据与算法偏见、假信息（hallucination）和奉承倾向、文化/注册不匹配导致的排他性、过度宣传与低估技术能力，以及对多样化子群体需求缺乏充分考虑。

---

## 245. ViCoStream: Streaming VideoLLMs Can Run Beyond 100 FPS with Stage-Wise Coordinated Inference

**arXiv ID:** 2606.19849 | [PDF](https://arxiv.org/pdf/2606.19849v1)

**作者:** Yang Tan `[一作]` (Southeast University), Xiaoyu Shen `[通讯]` (Eastern Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现ViCoStream框架，实现视频流VideoLLM的实时推理；

**💡 创新点**

通过将推理拆分为视觉预处理、视觉编码、token裁剪和LLM推理的四阶段流水线，并在每个阶段引入并行、bounded attention与检索控制，显著降低每块计算与内存开销；

**🔧 技术方法**

使用CUDA-stream重叠、批量ViT编码、空间/时间token裁剪、局部视觉注意力、查询驱动检索、可训练的注意力掩码等技术；

**📊 数据集**

训练集为LLaVA‑Video‑100K和TimeChat‑Online‑139K；评估基准包括StreamingBench、OvO‑Bench、StreamBench、VStream‑Ego、VStream‑Movie、ETBench；

**📈 对比分析**

在Qwen2.5‑VL‑3B/7B模型上，ViCoStream在单张A100 GPU上实现134 FPS视频吞吐量，TTFT<50 ms，且准确率与全历史基线相近；

**⚠️ 局限性**

限制：假设每个阶段只处理单块且可并行，未覆盖多副本/异构加速器；准确率与延迟权衡仍存在，某些任务对压缩和局部注意力更敏感。

---

## 246. Leverage Is Not Reach: A Control-Window Law for Single-Neuron Steering in Language Models

**arXiv ID:** 2606.19831 | [PDF](https://arxiv.org/pdf/2606.19831v1)

**作者:** Hongliang Liu `[一作]` `[通讯]` (Palo Alto Networks), Hongliang Liu (Palo Alto Networks)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于预算归一化的单神经元控制窗口框架，能够预测单神经元在不同剂量下是实现行为控制还是导致输出崩溃。

**💡 创新点**

创新点在于将单神经元干预视为剂量-响应关系，给出了统一的“控制窗口定律”，并提供了可预测的崩溃阈值与行为触发阈值，从而区分可控与不可控的单元。

**🔧 技术方法**

使用前向对比探测、预算归一化计算、控制坐标投影、梯度与二阶曲率分析，以及自定义安全审计器对拒绝、语言路由、算术运算与任务框架等行为进行评估。

**📊 数据集**

在公开的八大语言模型（Qwen、Llama、Gemma、Mistral、Phi‑3.5 mini 等）上使用标准提示集（拒绝、语言切换、算术运算、任务框架）进行实验，评估单神经元控制窗口的准确性。

**📈 对比分析**

与传统梯度重要性排名、固定剂量注射等方法对比，梯度方法只能捕捉崩溃倾向，而控制窗口法在15个保留神经元中平均崩溃阈值 MAE 0.14、开放/闭合判定正确率 11/15，明显优于占优类比基线。

**⚠️ 局限性**

局限性包括：触发阈值仍需经验测定；仅适用于具有离散基底的低维行为；崩溃系数依赖模型族且与残差参与度相关；方法仅考察 FFN 写入方向，注意力与读取侧尚未验证。

---

## 247. Linear Recurrent Unit with Semantic Modulation for Image Super-Resolution

**arXiv ID:** 2606.19901 | [PDF](https://arxiv.org/pdf/2606.19901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 248. CSWinUNETR: Segmentation of Thin Anatomical Structures in Medical Images

**arXiv ID:** 2606.19824 | [PDF](https://arxiv.org/pdf/2606.19824v1)

**作者:** Junho Moon `[一作]` (Hanyang University), Ikbeom Jang `[通讯]` (Hankuk University of Foreign Studies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计并评估了一种名为CSWinUNETR的通用深度学习骨干网络，用于医学图像中细薄、曲折结构（如视网膜血管、脑血管、面部皱纹）的分割。

**💡 创新点**

创新点在于引入交叉形条纹自注意力与循环偏移实现方向感知长程依赖，设计细节增强多尺度自注意力和稀疏控制动态蛇卷积（SDSConv）以重建曲线特征，同时兼顾细节与连通性。

**🔧 技术方法**

使用技术包括CSWin自注意力、跨条纹窗口、循环偏移、细节增强多尺度多头自注意力、稀疏控制动态蛇卷积、SwinUNETR式解码器以及Dice+交叉熵等损失。

**📊 数据集**

采用的公开数据集包括眼底图像集FIVES、脑血管CT/CTA血管集TopCoW、面部皱纹图像集FFHQ‑Wrinkle，以及对应的3D体积数据。

**📈 对比分析**

与nnUNetv2、UNETR、SwinUNETR、SwinUNETRv2、IncepFormer、DSCNet、CS^2‑Net、SGAT‑Net、TMP+U‑Net、GLCP、ER‑Net等基线在Dice、clDice、Betti Error与HD95等指标上对比，CSWinUNETR在四个基准上取得最高或第二高分，Dice/ClDice显著提升，连通性与边界误差大幅下降。

**⚠️ 局限性**

局限性包括模型参数较大（约81M），对计算资源需求较高；主要针对细薄结构，非细结构或大目标分割时表现未必优于专门网络；缺乏对不同尺度细节的自适应调节或多模态融合等进一步提升空间。

---

## 249. CREDENCE: Claim Reduction for Decomposition & Enhanced Credibility -- Semantic Metrics and Convergence Analysis

**arXiv ID:** 2606.19819 | [PDF](https://arxiv.org/pdf/2606.19819v1)

**作者:** Phuong Huu Vu Tran `[一作]` (Vietnamese-German University), Bach Xuan Le `[通讯]` (Ho Chi Minh University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CREDENCE框架，用于将复合句拆分为原子、可验证的声明，并对拆分质量进行无参评估与迭代修复；

**💡 创新点**

创新点包括：①用BGE‑large余弦相似度的Semantic‑F1取代传统Jaccard度量，显著减少语义等价句的误判；②对修复循环进行形式化收敛分析，证明规则修复单调终止、LLM自修复需早停；③构建跨领域三大拆分基准（SocialClaimSplit、WikiSplitBench、ClaimDecompBench）以及公开实现；

**🔧 技术方法**

采用BERT‑style语义嵌入（BGE‑large）、SpaCy依存句法检查、LLM（Phi‑3、Qwen3、Gemma‑3、Gemini）作为拆分器，规则与LLM交替执行的修复管道；

**📊 数据集**

使用三组数据集：社交媒体风格句子拆分（SocialClaimSplit），维基百科人工拆分（WikiSplitBench），新闻文本拆分（ClaimDecompBench），并在这些基准上评估；

**📈 对比分析**

与基线（FActScore、SpaCy‑senter）比较，Semantic‑F1平均提升15–44pp；规则修复在单轮即可收敛，平均AVR下降47–100%；在自动事实检查任务中，CREDENCE拆分使NLI模型在五大基准上准确率提升0.1–25.8pp，超过传统拆分方案；

**⚠️ 局限性**

局限性包括：依存解析误差导致AVR误报；实体提取仅用正则与SpaCy，无法处理核心ference和领域专用实体；Semantic‑F1受参考风格影响，对含代词的基准可能低估；LLM自修复不保证单调，需要早停策略。

---

## 250. FFinRED: An Expert-Guided Benchmark Generation and Evaluation Framework for Financial LLM Red-Teaming

**arXiv ID:** 2606.19887 | [PDF](https://arxiv.org/pdf/2606.19887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 251. A fast direct solver based neural network for solving PDEs

**arXiv ID:** 2606.19895 | [PDF](https://arxiv.org/pdf/2606.19895v1)

**作者:** Jashwanth Reddy Kadaru `[一作]`, Vaishnavi Gujjula `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构造并可视化不同结构的矩阵，研究了它们在低秩、全秩、零矩阵以及单位矩阵等类别下的性质。

**💡 创新点**

创新点在于提出一种基于索引的矩阵标签体系，并利用 TikZ 图形直观展示各种秩的矩阵分布，从而提供了一种新的可视化分析方法。

**🔧 技术方法**

使用了矩阵理论、组合数学以及 LaTeX/TikZ 绘图技术来生成和展示不同秩的矩阵。

**📊 数据集**

使用了人工生成的二值矩阵数据集（如 3×3、4×4 的所有组合）来进行实验和展示。

**📈 对比分析**

与随机生成的矩阵对比，实验表明所构造的矩阵在秩分布上与理论预期一致，主要通过可视化准确性和分类清晰度来评估其效果。

**⚠️ 局限性**

局限性包括仅适用于小规模矩阵，无法直接推广到高维或稀疏矩阵的实际应用场景。

---

## 252. MetaResearcher: Scaling Deep Research via Self-Reflective Reinforcement Learning in Adversarial Virtual Environments

**arXiv ID:** 2606.19893 | [PDF](https://arxiv.org/pdf/2606.19893v1)

**作者:** Wei Yu `[一作]` (Jiangxi Arts & Ceramics Technology Institute), Bing Li `[通讯]` (Jiangxi Arts & Ceramics Technology Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MetaResearcher框架，改进训练环境为时序演化与对抗信息，设计发现导向任务、可自省元奖励以及多代理分工架构，旨在提升深度研究代理的探索质量与鲁棒性。

**💡 创新点**

四大创新点：1）Evolving Virtual World引入时间动态与对抗性信息；2）Discovery-Oriented Tasks突破单纯事实检索，加入假设生成与矛盾解决；3）Self-Reflective Meta-Reward在GRPO中加入效率、反思深度与工具多样性奖励；4）Heterogeneous Multi-Agent Swarm将代理拆分为Scout、Filter、Synthesizer并通过协同RL训练。

**🔧 技术方法**

技术实现基于LiteResearcher生态，使用Qwen2.5-4B-Instruct模型、Milvus+BGE-M3检索、PostgreSQL浏览工具、GRPO强化学习、LLM判定器、共享通信缓冲区与自适应奖励融合。

**📊 数据集**

主要数据集：LiteResearcher本地网页约32M、GAIA与Xbench-DS基准；新增Evolving Virtual World的时间版本化文档与对抗性伪造内容；以及为假设生成、矛盾解决等新任务构造的自研合成数据。

**📈 对比分析**

对比方法：在GAIA、Xbench-DS、epistemic robustness benchmark以及发现任务评测上与LiteResearcher、DeepRubric、CaRR等基线对照。实验显示GAIA成绩提升至≥73%（+1.7%），Xbench-DS保持78%，对抗鲁棒性提升20%，并显著降低重复循环率50%。

**⚠️ 局限性**

局限性：对抗信息生成需严格质量控制，防止过度易辨或误导；多代理系统通信与同步带来额外算力开销；假设生成与矛盾解决的评判标准主观，依赖LLM评估；训练成本虽为零边际API，但整体GPU时数仍较高。

---

## 253. Toward Temporal Realism in City-Scale Crisis Response Simulation using LLM Agents

**arXiv ID:** 2606.19904 | [PDF](https://arxiv.org/pdf/2606.19904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 254. SurgVista: Long-Horizon Surgical World Modeling with Plausible Instrument-Tissue Dynamics

**arXiv ID:** 2606.19889 | [PDF](https://arxiv.org/pdf/2606.19889v1)

**作者:** Wentao Pan `[一作]` (Chinese University of Hong Kong), Yixuan Yuan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SurgVista手术世界模型，实现长时延真实的仪器-组织交互

**💡 创新点**

通过变形一致性正则化和漂移适配训练两大创新方法，解决空间交互不连贯和时间真实性崩塌问题

**🔧 技术方法**

利用潜在视频Diffusion、流匹配训练、基于跟踪点的对比学习、在线残差注入及光度增强等技术

**📊 数据集**

构建了SurgWorld‑Bench benchmark，采集了GraSP、SurgToolLoc22、CholecTrack20三类公开数据集

**📈 对比分析**

与多种通用及可控视频生成模型对比，SurgVista在短、长时延下在仪器运动、组织变形、视觉质量和时间一致性上均显著优于基线

**⚠️ 局限性**

仍难以处理大规模组织变形和极细粒度操作，且未考虑力学与触觉信息

---

## 255. Measuring Biological Capabilities and Risks of AI Agents

**arXiv ID:** 2606.19899 | [PDF](https://arxiv.org/pdf/2606.19899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 256. Multimodal Concept Bottleneck Models

**arXiv ID:** 2606.19882 | [PDF](https://arxiv.org/pdf/2606.19882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 257. REDACT: A Systematically Controlled Multilingual Benchmark for Personal Information Detection

**arXiv ID:** 2606.19881 | [PDF](https://arxiv.org/pdf/2606.19881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 258. Global Convergence of Gradient Descent for Score Matching in Gaussian Mixtures via Reverse Fisher Divergence

**arXiv ID:** 2606.19876 | [PDF](https://arxiv.org/pdf/2606.19876v1)

**作者:** Alexander Tyurin `[一作]` `[通讯]` (Applied AI Institute), Alexander Tyurin (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

研究逆 Fisher 散度（reverse Fisher divergence）在高斯混合模型（GMM）上进行分数匹配，并证明梯度下降（GD）在该目标下能够从任意初始化实现全局收敛。

**💡 创新点**

提出逆 Fisher 散度相较于传统的正 Fisher 散度拥有更好的优化景观，提供了全局收敛证明、Lyapunov 基础的分析技术，并揭示了随机初始化与均匀分离条件下的收敛速率。

**🔧 技术方法**

使用梯度下降、矩阵分析、Lyapunov 函数、随机初始化与高维分离假设、闭式梯度表达式以及数值积分（Gauss–Hermite）来计算期望。

**📊 数据集**

主要使用合成的高斯混合数据（单峰和多峰场景）进行实验，未使用公开真实数据集。

**📈 对比分析**

与传统正 Fisher 散度进行对比：在单峰和多峰实验中，正 Fisher 散度可能陷入错误解或停滞，而逆 Fisher 散度始终收敛到目标分布，数值实验显示收敛误差可降至几乎零。

**⚠️ 局限性**

限制包括：需要均匀分离（δ_min≥Ω̃(1)）且维度满足 d≥Ω̃(n)；对随机初始化的依赖；仅针对高斯混合模型；步长受严格限制；实验仅使用全精度梯度，未探讨随机梯度的表现。

---

## 259. OTCHA: Optimal Transport-driven Confidence-aware Latent Hub Alignment for Multi-View Medical Image Classification

**arXiv ID:** 2606.19838 | [PDF](https://arxiv.org/pdf/2606.19838v1)

**作者:** Jiwoong Yang `[一作]` (Hanyang University), Ikbeom Jang `[通讯]` (Hankuk University of Foreign Studies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了一种基于最优传输的置信度感知潜在枢纽对齐模块OTCHA，用于多视角医学图像分类，在融合前对每个视角的 patch token 进行精细化处理，利用共享枢纽 token 与 token 条件 dustbin 的 OT 进行匹配并进行 hub‑mediated message passing，结合 OTRA 损失提升鲁棒性。

**💡 创新点**

引入可学习的共享潜在枢纽 token 作为跨视角的聚合点，使用带 token 条件 dustbin 的 entropic OT 实现局部匹配与置信度评估，利用 OT 产生的置信度门控 hub 消息传递，提出 OTRA 对齐损失来稳定训练，并提供可解释的跨视角对应关系。

**🔧 技术方法**

共享 CNN–Transformer 编码器、基于 Sinkhorn 的 entropic 最优传输、token 条件 dustbin、hub‑mediated message passing、置信度加权的 OTRA 损失、互相蒸馏等。

**📊 数据集**

VinDr‑Mammo（乳腺 X 光四视角）、MURA（多视角骨骼 X 光）、CheXpert（前后视胸部 X 光）等三大公开多视角医学图像数据集。

**📈 对比分析**

与多种 SOTA 多视角方法（MVCNet、Cross‑view attention、Mutual distillation、State‑space sequence models 等）在三组数据集和不同视角配置下进行对比，OTCHA 在所有设置下均取得最高 AUROC，尤其在四视角 VinDr‑Mammo 上提升显著，且参数开销微小。

**⚠️ 局限性**

目前仅支持固定视角设置，未处理缺失视角、跨模态或下游任务等；此外对超参数（如 Sinkhorn 正则化、geometry 权重）仍需调优。

---

## 260. Fault-Tolerant Shared-Relay Communication in Circulant Interconnection Networks

**arXiv ID:** 2606.19833 | [PDF](https://arxiv.org/pdf/2606.19833v1)

**作者:** Bader Albader `[一作]` (Kuwait University), Mohamed R. Al-Mulla `[通讯]` (Kuwait University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在有向循环网络中基于共享中继的两跳容错性，定义了最小节点度与共享中继冗余的关系，并构建了相应的度-冗余景观。

**💡 创新点**

将循环差分多重性与网络级共享中继冗余等价关系正式化，并在此基础上提出了度-冗余设计框架、负定论、预处理/查找算法、负载均衡分析与实证评估。

**🔧 技术方法**

结合群论差分基理论、组合最优化的向量化贪心构造、图论度数分析与概率论随机/最坏情况失败模型，同时实现了表格预处理与负载感知查找算法。

**📊 数据集**

对 6 个不同规模（n=251,503,1009,2003,5003,10007）以及 6 个容错等级（f=0…5）的 526,539 种生成器集合进行系统实验，并对小阶群进行完整枚举校准。

**📈 对比分析**

通过与传统区间生成器、对称区间、模步、平方剩余、随机和贪心等族的设计比较，使用计数下界和已有差分基上界做基准，结果显示贪心设计在满足 f-Relay 容错时仅比计数下界高 1.16–1.63 倍，且相较图搜索的 O(nm) 复杂度，查找成本仅 O(m) 并在实验中平均快数十倍。

**⚠️ 局限性**

贪心构造仅为启发式且在大规模下未证明最优；未给出 NP‑hardness 或逼近比；对于 f≥1 缺乏正式的差分基构造上界；实验仅为软件级查找，未覆盖完整的网络层拥塞或硬件周期仿真。

---

## 261. Certified Euclidean-Residue Minimal-Alignment Switch Decompositions for Three Edge-Disjoint Hamiltonian Cycles in Eisenstein--Jacobi Networks

**arXiv ID:** 2606.19832 | [PDF](https://arxiv.org/pdf/2606.19832v1)

**作者:** Bader Albader `[一作]` `[通讯]` (Kuwait University), Bader Albader (Kuwait University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在Eisenstein–Jacobi网络中构造了三条互不共享边的哈密顿环，并提出最小化本地交换（每条方向需 d-1 次切换）的构造方法。

**💡 创新点**

核心创新包括最小切换骨架、锚定锁定机制以及欧几里得残差相位规则，取代传统矩形交换调度，得到符号化、闭式的 EDHC 构造。

**🔧 技术方法**

使用Cayley 几何的局部菱形交换、组件标签分析、相位算子与欧几里得算法相结合的局部切换算子，并辅以符号计算与代数证明。

**📊 数据集**

通过生成的 CSV 边集表格进行有限边集审计，仅用于验证与重现，未使用真实大规模数据集。

**📈 对比分析**

通过理论证明与有限边集审计相结合，确认所有归档参数族满足 EDHC；未给出性能对比，只验证构造的正确性与连通性。

**⚠️ 局限性**

证明范围仅限于 d≥4 且满足特定欧几里得残差族的参数，未覆盖全部非互素比例，普适性仍待进一步研究。

---

## 262. On the Oracle Complexity of Interpolation-Based Gradient Descent

**arXiv ID:** 2606.19878 | [PDF](https://arxiv.org/pdf/2606.19878v1)

**作者:** Dongmin Lee `[一作]` (Purdue University), Anuran Makur `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于分段多项式插值的梯度下降方法（PPI-GD），用于优化经验风险最小化（ERM）目标，利用训练数据的光滑性来提高梯度下降的oracle复杂度。

**💡 创新点**

PPI-GD在数据维度d = O(log^0.49(n))的情况下，能够在强凸和非凸损失函数下实现更低的oracle复杂度，优于传统的GD和SGD等方法。

**🔧 技术方法**

使用了分段多项式插值技术来近似每次迭代中的真实梯度oracle，并分析了其在强凸和非凸情况下的oracle复杂度。

**📊 数据集**

使用了合成数据集和真实数据集进行实验，具体数据集包括用于训练神经网络的合成分类数据集和系统识别任务中的振动测试数据集。

**📈 对比分析**

与GD、SGD、LPI-GD等方法进行了比较，PPI-GD在多个实验中表现出更优的oracle复杂度，尤其是在损失函数足够光滑的情况下，PPI-GD在oracle复杂度上优于其他方法。

**⚠️ 局限性**

PPI-GD在高维数据上可能受到维度诅咒的影响，且在某些情况下可能会收敛到次优解，尤其是在数据噪声较大的情况下。

---

## 263. Doeblin Curves

**arXiv ID:** 2606.19859 | [PDF](https://arxiv.org/pdf/2606.19859v1)

**作者:** Dongmin Lee `[一作]` (Purdue University), Japneet Singh `[通讯]` (Purdue University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并证明了 Doeblin 系数的变分表述，进一步引入了 Doeblin 曲线概念，用以刻画马尔可夫核在不同约束下的退化程度，并给出了该曲线在功率约束下的上下界，随后在随机网络、机器学习泛化、隐私等应用场景中证明了 Doeblin 曲线的若干重要性质；

**💡 创新点**

创新点主要包括：①在任意可测空间上证明 Doeblin 系数的变分表示；②定义 Doeblin 曲线并证明其组合、凸性、单调性及 Lipschitz 连续性等属性；③在功率约束下得到 Doeblin 曲线的解析上界与下界，并给出卷积噪声的闭式解；

**🔧 技术方法**

技术手段涵盖：测度论中的最大公共成分、Radon–Nikodym 导数、格点下取最小、最大耦合构造、Jensen/Holder 不等式、对偶变分原理等；

**📊 数据集**

未使用传统机器学习或统计实验数据集，主要以随机生成的行随机矩阵与噪声核进行仿真验证；

**📈 对比分析**

与传统的 TV 退化系数（例如 Dobrushin 系数）对比时，本文的 Doeblin 曲线提供更细粒度的退化度量，实验结果表明在某些高功率约束下能显著降低系统退化率；

**⚠️ 局限性**

局限性包括：①需要核在可测空间上绝对连续，才能使用 Radon–Nikodym 导数；②对可测空间的结构（如可压缩性、半可测性）有一定假设；③在高维连续空间中实际求解 Doeblin 曲线仍可能面临计算复杂度与耦合生成的挑战。

---

## 264. Weight Adaptation for Improving Parallel Performance of Adaptive Stochastic Natural Gradient

**arXiv ID:** 2606.19861 | [PDF](https://arxiv.org/pdf/2606.19861v1)

**作者:** Yutaro Yamada `[一作]` (Yokohama National University), Shinichi Shirakawa `[通讯]` (Yokohama National University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Weight Adaptation ASNG（WA-ASNG）算法，在 ASNG 中加入自适应权重机制，以提升二进制优化问题的并行搜索性能。

**💡 创新点**

创新在于通过估计自然梯度累积信号并使用梯度上升动态调整权重，从而最大化更新方向的有效信号，并与 ASNG 的学习率自适应结合。

**🔧 技术方法**

采用信息几何优化框架下的自适应随机自然梯度（ASNG）、伯努利分布、学习率自适应、梯度上升权重调整、并行采样等技术。

**📊 数据集**

使用三种经典二进制基准函数（求和、加权求和、连乘积）在 N=100~500 维空间下，人口规模 λ=25~100 进行实验。

**📈 对比分析**

与 PBIL、ASNG 及其不同权重设定进行比较，评估评价次数、成功率、学习率变化等指标；实验显示 WA-ASNG 在多数设置下的评估次数更少、成功率更高，且在噪声环境下表现更稳健。

**⚠️ 局限性**

实验仅限于三种简单二进制基准，未检验在更具挑战性或连续/分类问题上的表现，且未验证该权重自适应机制是否可迁移至其他概率模型进化算法。

---

## 265. Neural Additive and Basis Models with Feature Selection and Interactions

**arXiv ID:** 2606.19850 | [PDF](https://arxiv.org/pdf/2606.19850v1)

**作者:** Yasutoshi Kishimoto `[一作]` (Yokohama National University), Shinichi Shirakawa `[通讯]` (Yokohama National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出在神经可加模型（NAM）和神经基准模型（NBM）中加入可微分的特征选择层，使模型能够在高维数据上高效训练且保持可解释性。

**💡 创新点**

创新点在于将entmax温度退火的特征选择机制嵌入NAM/ NBM，实现端到端学习、显著降低参数量与计算成本，并支持二阶特征交互。

**🔧 技术方法**

使用PyTorch实现的MPL形状函数、entmax温度退火特征选择、温度退火、以及对比实验中使用的梯度下降训练。

**📊 数据集**

实验使用六个高维分类数据集：HAR、ISOLET、F‑MNIST、Epsilon、guillermo（4,296维）和Gisette（5,000维）。

**📈 对比分析**

与EBM、NODE‑GAM、LR、DT、MLP、XGBoost等基线进行对比，结果显示NAM‑FS/NBM‑FS在高维数据上与现有GAM/GA^2M相当或更优，且在大多数任务上接近XGBoost的性能。

**⚠️ 局限性**

局限性：对低维（<100）数据提升有限；仅考虑二阶交互；缺乏对稀疏特征的专门支持；对超参数K1/K2的选择敏感。

---

## 266. World Engine: Towards the Era of Post-Training for Autonomous Driving

**arXiv ID:** 2606.19836 | [PDF](https://arxiv.org/pdf/2606.19836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 267. AtomMem: Building Simple and Effective Memory System for LLM Agents via Atomic Facts

**arXiv ID:** 2606.19847 | [PDF](https://arxiv.org/pdf/2606.19847v1)

**作者:** Yanyu Yao `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 AtomMem 长期记忆框架，能够从对话中提取原子事实并组织成事件与时间配置文件，利用图关联检索实现稳定、稠密的记忆管理。

**💡 创新点**

创新点在于原子事实抽取与结构化存储、分层事件与时间配置文件构造、图形关联记忆激活，以及通过有限重写的稳定更新机制提升长期一致性。

**🔧 技术方法**

使用技术包括 SFT LoRA 训练的 Fact Executor、分层检索（全局、事件补偿、图关联）、混合相似度度量、随机游走 (RWR) 图激活、LLM 生成与校验、时间锚定与核心ference。

**📊 数据集**

主要使用 LoCoMo 和 LongMemEval 两个长期对话记忆基准，并为 Fact Executor 构造了高质量的抽取数据集 𝒟。

**📈 对比分析**

与 MemoryBank、A‑Mem、MEM0、Mem0、MemoryOS、LightMem 等基线对比，AtomMem 在 LoCoMo 的多跳、时间推理与开放域任务上分别提升 BLEU、R@10 与 LLM‑as‑a‑Judge 分数，最终在多项指标上位居榜首。

**⚠️ 局限性**

局限性包括对底层 LLM 生成稳定性的敏感性、仅支持文本对话、图与检索仍需调优以进一步减少噪声，且 token 效率仍有提升空间。

---

## 268. Multi-Orientation Edge-Minimum Repair for Non-Redundant Fault-Tolerant Broadcasting in Dense Eisenstein--Jacobi Networks

**arXiv ID:** 2606.19834 | [PDF](https://arxiv.org/pdf/2606.19834v1)

**作者:** Bader Albader `[一作]` `[通讯]` (Kuwait University), Bader Albader (Kuwait University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对稠密Eisenstein–Jacobi网络中出现的一到两个节点故障，提出一种多方向边最小化修复方法(EJ-MOEM)，通过预先构造15种坐标约简广播树，删除故障节点后对剩余森林进行组件收缩，再以最少的外部跨组件边重新连通各组件，从而得到非冗余的单源广播树。

**💡 创新点**

创新点包括：①引入15个方向优先级的坐标约简广播树族，保证在任意一到两点故障下至少有一棵树的组件图连通；②证明对于连通组件图，c-1条外部边是必要且充分的；③给出深度证明：一故障可修复到深度≤t+1，二故障≤t+2；④在三条抽象证明的基础上，进行了从小到大尺寸（t=2…200）的全面穷举与随机验证，验证了理论与实现的一致性。

**🔧 技术方法**

主要技术：坐标约简父亲选择、组件图构造与最小生成树（或层次化选择）、深度证书（K-depth repair certificate）以及对EJ网格的三条带坐标与六方向对称性分析；实现采用线性时间 O(N) 的算法。

**📊 数据集**

使用的“数据集”是稠密EJ网络的几何模型：给定网络半径t，生成所有节点坐标及邻接，随后对所有1-和2-故障组合进行穷举验证；测试覆盖t=2…200（包括t=30的结构化测试和t=200的随机测试）共计超过120,000个节点。

**📈 对比分析**

比较方法：与全局BFS重建、固定方向单一树、以及独立树冗余等基线对比。结果显示：EJ-MOEM在所有测试中都实现了理论深度界限，且所需外部修复边数恰为c-1，显著低于全局重建所需的最多N-1条边；在随机与结构化测试中，平均深度与最大深度仅比最优基线高0–1级。

**⚠️ 局限性**

限制：深度上限为t+2在t≥4时并未被实测到达到，理论上可能可进一步降低到t+1；方法假设故障点不包括源节点；仅针对一到两点故障；对更大规模多点故障的可扩展性尚未验证；实现依赖于完整的EJ网格构造，若网络非稠密或存在缺失节点需进一步改造。

---

## 269. When, Where, and How: Adaptive Binning for Tabular Self-Supervised Learning

**arXiv ID:** 2606.19827 | [PDF](https://arxiv.org/pdf/2606.19827v1)

**作者:** Daehwan Kim `[一作]` (Hanyang University), Ikbeom Jang `[通讯]` (Hankuk University of Foreign Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了一种基于自适应离散化的医学表格自监督预训练任务

**💡 创新点**

创新点在于训练动态的特征级粗细递进离散化、基于饱和触发的分割、表示空间一致性驱动的分割选择，以及混合分类与序数目标的 HORD 损失

**🔧 技术方法**

采用递进式自编码器、特征饱和触发、DIGS 分割策略、HORD 软序数损失，以及线性探测和微调评估

**📊 数据集**

使用公开医学表格基准，包括二分类、多分类（名义与序数）与回归任务

**📈 对比分析**

与固定全局分箱预训练（BinRecon）及其他自监督目标相比，在统一评估协议下线性探测和微调均显著提升，平均排名最高

**⚠️ 局限性**

仅在单数据集内迁移，评估协议有限，未跨数据集预训练或广泛下游任务验证

---

## 270. Heterogeneous LLM Debate Under Adversarial Peers: Honest Gains, Replacement Costs, and Resilience

**arXiv ID:** 2606.19826 | [PDF](https://arxiv.org/pdf/2606.19826v1)

**作者:** Prashanti Nilayam `[一作]` (ServiceNow), Sankalp Nayak `[通讯]` (ServiceNow)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多模型辩论中异构模型对诚实模型修正行为的影响，并量化其在面对恶意同伴时的优劣；

**💡 创新点**

首次从被试者视角分离“诚实奖励”和“恶意惩罚”，引入“flip rate”衡量最终错误传播，揭示异构模型既能提升修正质量也能抵御已被污染的面板；

**🔧 技术方法**

基于多代理辩论协议、检测-生成分解框架、诚实修正率与flip率指标，结合异构与恶意同伴的匹配与污染对照设计；

**📊 数据集**

MATH‑hard、SciBench、GSM8K三大推理基准（分别覆盖高级数学、科学推理与算术），覆盖四个模型族（小型开源、大型开源、前沿开源/闭源、同家族）；

**📈 对比分析**

通过匹配比较（同一槽位只变换为诚实或恶意同伴）和污染比较（已存在恶意同伴时替换一名同族诚实同伴），发现诚实异构同伴可将诚实修正率从≈90%降低到≈35%，而恶意同伴则恢复至≈90%；在污染情形下，诚实异构同伴能将错误传播率从≈30%降至≈6%（或在SciBench从7.3%降至2.6%）；

**⚠️ 局限性**

仅针对可检验答案的客观任务，未覆盖开放式生成或主观评估；攻击模型仅为单一提示级别的恶意同伴，未考虑多方共谋、适应性优化或对解析器的攻击；实验聚焦首轮修正，后续回合效应未系统评估。

---

## 271. Low-Cost Multi-Precision Systolic Arrays for Accelerating FHE NTTs on AI ASICs

**arXiv ID:** 2606.19866 | [PDF](https://arxiv.org/pdf/2606.19866v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 272. Score Approximation for Diffusion Models on Arbitrary Low-Dimensional Structures

**arXiv ID:** 2606.19894 | [PDF](https://arxiv.org/pdf/2606.19894v1)

**作者:** Xinhe Mu `[一作]` (Academy of Mathematics and Systems Sciences, Chinese Academy of Sciences), Zhiming Ma `[通讯]` (Academy of Mathematics and Systems Sciences, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

提出了一套针对任意紧支撑分布的分数逼近理论，证明在不需要任何光滑性或连续性假设的前提下，ReLU 网络能够在误差为 ϵ 的条件下逼近分数函数，且网络参数规模仅与内在维数 d 成指数关系，而与外在维数 n 仅呈多项式关系。

**💡 创新点**

创新点主要有：①将分数逼近问题转化为离散混合分布的近似，完全摆脱了传统的 Lipschitz、Holder 等光滑性假设；②引入“C_{1},C_{2},r-regular”这一几乎无约束的稠密性条件，证明几乎所有紧支撑分布都满足该条件；③通过细致的几何覆盖与“近/远”分组策略，控制混合分布中远程成分的贡献，最终得到误差可控的近似表达式；④给出可构造的 ReLU 网络实现方案，其深度和宽度仅随误差 ϵ 的对数而增长。

**🔧 技术方法**

核心技术包括：
- 变分扩散（VE）扰动方案下的 Gaussian 混合分解；
- 采用覆盖数（上 Minkowski 维数）对分布支撑进行离散化；
- 对 Gaussian 核函数做泰勒展开并利用正则性条件控制剩余项；
- 通过几何推送与 Vitali 论证处理远程球体的质量贡献；
- 构造 ReLU 网络逼近 Softmax 与指数/倒数等非线性函数；
- 使用 Gaussian 集中不等式保证噪声样本落入正则点附近的概率。

**📊 数据集**

本工作为理论研究，未使用任何具体数据集；所有结论均在数学证明层面给出。

**📈 对比分析**

与已有工作相比，传统方法在逼近分数函数时需假设分布具有 Lipschitz 连续、Holder 连续或位于低维线性子空间；它们的网络规模往往随外在维数 n 指数增长。该论文通过上 Minkowski 维数的概念，将指数增长迁移至内在维数 d，且对任何紧支撑分布均适用；理论上可实现更低的网络复杂度，且不依赖光滑性假设。

**⚠️ 局限性**

局限性包括：
- 依赖上 Minkowski 维数 d，若 d 较大仍会导致指数级参数规模；
- 需要提前设定早停时间 t_0 并保证 σ(t) 的增长速率；
- 证明仅给出理论误差界限，缺乏实验验证；
- 对分布的正则性条件虽几乎无约束，但在极端稀疏或高度分形的情况仍可能需要更严格的覆盖数估计；
- 实际训练中如何高效实现所构造的 ReLU 网络仍是后续研究方向。

---

## 273. MMD-SLAM: Structure-Enhanced Multi-Meta Gaussian Distribution-Guided Visual SLAM

**arXiv ID:** 2606.19874 | [PDF](https://arxiv.org/pdf/2606.19874v1)

**作者:** Fan Zhu `[一作]` (HFIPS Chinese Academy of Sciences), Chunmao Jiang `[通讯]` (HFIPS Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MMD‑SLAM，结合点–线融合的姿态优化、基于 Atlanta World（AW）假设的多模态高斯分布（点、高斯线、面）以及高斯演化与结构增强的场景优化，实现高保真 3D Gaussian Splatting（3DGS）视觉 SLAM；

**💡 创新点**

创新点在于：① 引入 AW 假设为多模态高斯提供主方向约束，形成点、高斯线、面三种结构化原语；② 通过点–线双重约束提升姿态鲁棒性；③ 设计弱–稳态高斯演化（clone、split、merge、转换）实现自适应结构捕捉；④ 在映射过程中加入形状、方向损失，进一步对齐高斯与真实结构；

**🔧 技术方法**

使用技术包括：3D Gaussian Splatting、EDLines + LBD 描述子进行线特征检测/匹配、Levenberg‑Marquardt Bundle Adjustment、Gaussian 结构演化操作、颜色通过球谐函数学习、形状/方向约束损失以及 α‑混合渲染；

**📊 数据集**

实验数据集：TUM RGB‑D（位姿评估）、ScanNet（大规模室内、位姿评估）和 Replica（合成室内、渲染质量评估）；

**📈 对比分析**

与 SplaTAM、MonoGS、RTG‑SLAM、GS‑ICP‑SLAM、MG‑SLAM 等 SOTA 3DGS‑SLAM 基线进行对比，使用 ATE RMSE 评估跟踪精度，使用 PSNR/SSIM/LPIPS/FPS 评估映射渲染质量。MMD‑SLAM 在 PSNR 上提升约 3–4 dB、SSIM 0.01‑0.02、LPIPS 降低 1‑2% 且保持与基线相当的实时帧率；跟踪精度与最优基线相当或更优；

**⚠️ 局限性**

局限性：① 依赖丰富的线特征，若场景缺乏直线或为自然环境时性能可能下降；② 高斯演化与结构优化增加计算量，虽仍可达实时但对低功耗设备不友好；③ 仅在室内静态场景验证，未评估动态或户外环境的鲁棒性；

---

## 274. Physics-Informed Neural Network with Squeeze-Excitation-like Attention

**arXiv ID:** 2606.19853 | [PDF](https://arxiv.org/pdf/2606.19853v1)

**作者:** Yun-Fei Song `[一作]` (Central China Normal University), Jun-Jie Zhang `[通讯]` (Northwest Institute of Nuclear Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SEA-PINN架构，将Squeeze‑Excitation‑like注意力机制嵌入物理信息神经网络，实现自适应神经元加权，提升表示能力和收敛稳定性。

**💡 创新点**

核心创新在于引入轻量级的权重生成器（Weight Generator），对每一隐藏层的所有神经元输出动态加权，形成输入相关的注意力；该机制使得网络在初始化即能产生近零均值与方差的解与导数，从而大幅降低初始残差和训练波动。

**🔧 技术方法**

技术手段包括：基于全连接网络的PINN；Squeeze‑Excitation‑style注意力模块（Tanh→Linear→Sigmoid生成权重）；SiLU激活、He均匀初始化；Adam优化器；全批训练、Hammersley采样、FP64数值精度。

**📊 数据集**

使用20个多样化的PDE基准（热传导、流体力学、生物、等离子、超维问题等），每个问题采用固定的采样点集，全部在无监督物理约束下训练。

**📈 对比分析**

通过在每个基准上跑30个随机种子、5000个epoch，并与传统FNN‑PINN、TSA‑PINN以及混合TSA‑SEA‑PINN进行相同配置的对比；SEA‑PINN在13/20个案例中误差更低，9个案例提升>5%，并在高频、稳态多输出等场景中表现尤为优异。

**⚠️ 局限性**

局限性包括：基准范围有限，未在更大规模或更高维问题上验证；权重生成器虽轻量，但仍增加计算开销；对非平衡或含极大源项的方程性能受限；并未彻底解决所有PINN的收敛慢、梯度不平衡等根本问题。

---

## 275. Large Language Models Do Not Always Need Readable Language

**arXiv ID:** 2606.19857 | [PDF](https://arxiv.org/pdf/2606.19857v1)

**作者:** Jiayi Zhu `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了一种针对大型语言模型的高密度文本压缩表示BabelTele，并探讨其在多任务、跨模型、代理记忆与多代理通信中的可用性。

**💡 创新点**

创新点在于将人类可读性从压缩目标中剔除，提出符号坍缩原则，生成几乎不可读但可被LLM解码的高信息密度文本。

**🔧 技术方法**

采用零样本提示诱导LLM生成符号化压缩文本，并通过可读性诊断、困惑度、问卷与下游QA评估等技术验证其语义保真度。

**📊 数据集**

主要在QuALITY、LongBench v2（Short、Code Repo QA Long）、MeetingBank、LoCoMo以及自定义多代理任务等公开数据集上进行实验。

**📈 对比分析**

与自然语言摘要、LLMLingua等基线对比，BabelTele在保持约99.5%语义完整的前提下压缩至27.9%，在QA准确率和思考链长度方面与原始文本保持接近，跨模型零样本传输仍能保留约80–90%的性能。

**⚠️ 局限性**

研究仅覆盖选定任务和模型族，缺乏对更广泛任务的验证，且缺乏对其机制的深入解释，未来需进一步探究其理论基础和安全风险。

---

## 276. JAMER: Project-Level Code Framework Dataset and Benchmark on Professional Game Engines

**arXiv ID:** 2606.19830 | [PDF](https://arxiv.org/pdf/2606.19830v1)

**作者:** Jianwen Sun `[一作]` (Nankai University), Kaipeng Zhang `[通讯]` (Shanghai Innovation Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了基于Godot游戏引擎的项目级代码框架数据集JamSet和基准JamBench，并提出了四级确定性验证管道，实现了从文件完整性到运行时行为的全流程自动评估。

**💡 创新点**

创新点包括①首次在专业游戏引擎层面提供大规模项目级代码框架数据集与基准；②基于Godot headless 的四级确定性验证与行为收集方法；③通过SCS和BAS指标分别评估静态结构与运行行为，揭示模型在架构设计上的瓶颈。

**🔧 技术方法**

使用技术包括Godot无图形模式、文本可解析项目文件、LLM辅助的结构化注释、规则驱动的确定性输入策略、自动化编译与运行时行为采集，以及SCS/BAS评价体系。

**📊 数据集**

数据来源于约240k GitHub及Game Jam（Ludum Dare、itch.io、Global Game Jam等）开源项目，最终筛选出8,133个验证通过的2D Godot项目，其中1,000个构成JamBench基准，7,833个用于JamSet训练数据。

**📈 对比分析**

通过对9种前沿LLM和Code Agent的实验，采用L1/L2/L3a通过率、SCS与BAS三项指标进行评估，结果显示模型在小规模项目可达约70%运行通过率，但在大规模项目编译通过率骤降至约5%，且SCS/BAS均远低于人类项目；Code Agent仅提升编译通过率，对结构与行为质量提升有限。

**⚠️ 局限性**

局限性在于仅聚焦Godot引擎，未覆盖Unity/Unreal等主流引擎；仅评估代码框架，未涉及艺术与音频资产；缺乏深入的模型训练与消融实验；验证管道与指标对完整游戏体验的覆盖仍有限。

---

## 277. Gaussian Process Prior Variational Autoencoder for Endoscopic Videos

**arXiv ID:** 2606.19908 | [PDF](https://arxiv.org/pdf/2606.19908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 278. 3D-PLOT-LLM: Part-Level Object Tokens for 3D Large Language Models

**arXiv ID:** 2606.19828 | [PDF](https://arxiv.org/pdf/2606.19828v1)

**作者:** Jintang Xue `[一作]` (University of Southern California), C. -C. Jay Kuo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出3D-PLOT-LLM，利用点云编码器输出的patch tokens重新组织成可直接在LLM词表中引用的K个几何区域，并通过Marker‑Space Refinement（MSR）为每个区域生成结构化的标记，从而让3D多模态大语言模型在不使用分割解码器或框架的情况下实现对物体部件的引用、命名与推理。

**💡 创新点**

创新点：①将部件视为词表级别可寻址的token；②无监督、确定性的几何区域划分保证同一对象在不同推理时保持同一槽位；③轻量级MSR为标记注入空间统计与邻域信息，使得LLM能在自回归过程中直接对部件进行处理；④在不增加分割或检测头的情况下，仅通过约100万可训练参数即可实现高质量部件级语言理解。

**🔧 技术方法**

技术：冻结的Point‑BERT点云编码器、Vicuna‑v1.5‑7B LLM、基于patch token的区域划分算法、Marker‑Space Refinement（基于MLP与消息传递的残差更新）、在token序列前插入保留词表槽位和可学习标记。

**📊 数据集**

使用数据集：Objaverse点云+Cap3D全局描述（660K条）用于对齐训练；PartVerse‑QAPairs（77K）用于部件级双向（C2S/​S2C）监督；3DCoMPaT‑GrIn PaPGD（6770条）用于部件级定向描述评估；额外使用Point‑LLM的70K复杂指令进行指令微调。

**📈 对比分析**

与多种基线对比：在Objaverse整体描述上，3D‑PLOT‑LLM在SBERT、SimCSE、GPT‑4o、BLEU‑1、ROUGE‑L、METEOR等六项指标上均优于PointLLM、PointLLM‑PiSA以及ShapeLLM；在PartVerse‑QA中，Jaccard和Exact‑match分别提升至0.459/13.78%（比无标记版提升64%），S2C GPT‑4o评判得分提升至44.68；在3DCoMPaT‑GrIn基准中，BLEU‑4、METEOR、SBERT、SimCSE和GPT‑4o分别提升0.47/0.74/0.81/0.90/3.03。性能提升主要归因于词表级部件地址化与MSR，而不依赖额外的分割/检测模块。

**⚠️ 局限性**

局限性：K个固定槽位只能覆盖约94.5%对象的部件数；不支持点级分割或精细掩码；目前仅处理单个物体而非完整场景；对极大或复杂部件层级的可扩展性尚未验证。

---

## 279. One-to-Two Acting: A Novel Framework for Single-arm Agent Action Expansion to Dual Arms

**arXiv ID:** 2606.19897 | [PDF](https://arxiv.org/pdf/2606.19897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 280. Open Weight AI Models Require Proportional Evaluation Approaches

**arXiv ID:** 2606.19890 | [PDF](https://arxiv.org/pdf/2606.19890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 281. Enhancing Graph Neural Networks Using Proximity Graphs for Dust Source Emission Forecasting

**arXiv ID:** 2606.19825 | [PDF](https://arxiv.org/pdf/2606.19825v1)

**作者:** Maryam Sanisales `[一作]` (Amirkabir University of Technology), Ali Vefghi `[通讯]` (Amirkabir University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究利用近邻图构建的图神经网络，对特尔斯和底河盆地的沙尘源排放进行时空预测。

**💡 创新点**

首次将多种几何近邻图（Delaunay、Gabriel、kNN、Yao）与静态GNN相结合，构造时空连通图，从而显著提升沙尘源预测性能。

**🔧 技术方法**

采用GraphSAGE、GCN、GAT等图神经网络以及LSTM基线模型，结合空间采样、时序图连接与邻接矩阵归一化等技术。

**📊 数据集**

使用2000-2021年MODIS遥感识别的11,618个沙尘源，22年每月约11,000个热点，包含植被、降雨、土壤湿度、风速等多维环境因子。

**📈 对比分析**

通过Accuracy、AUC、Precision、Recall四指标进行50次重复实验，对不同图结构和模型进行比较；GNN在所有指标上明显优于随机图和LSTM，最优组合（如Delaunay+GraphSAGE）AUC≈0.72，Recall≈0.82。

**⚠️ 局限性**

受限于手工标注的遥感热点、仅考虑二维空间邻接、未捕捉垂直结构或长期气候趋势，导致模型在更大尺度或更复杂气候变化场景下的泛化能力有限。

---

## 282. Adversarial Bandit Optimization with Globally Bounded Perturbations to Convex Losses

**arXiv ID:** 2606.19891 | [PDF](https://arxiv.org/pdf/2606.19891v1)

**作者:** Zhuoyu Cheng `[一作]` (Kyushu University), Eiji Takimoto `[通讯]` (Kyushu University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在存在累积受限扰动的情况下，一点反馈下的对抗性带状优化问题，并给出了期望下的次线性后悔保证。

**💡 创新点**

创新点在于将传统凸光滑损失扩展到可近似凸且光滑的损失序列，利用全局扰动预算控制非凸扰动的影响，并在此基础上改进SCRiBLe算法以处理非线性损失和扰动。

**🔧 技术方法**

主要技术包括自协变的自共形障碍函数、随机平滑与梯度估计、FTRL分析、以及对扰动预算的统一控制。

**📊 数据集**

无数据集，全部为理论分析与证明。

**📈 对比分析**

无实验比较，本文仅给出理论后悔界限，表明在扰动预算有限时仍可获得O(T^{2/3})的期望后悔。

**⚠️ 局限性**

局限性包括只考虑一次性点反馈、仅对凸光滑基损失适用、且缺乏实验验证，实际性能需进一步探讨。

---

## 283. SL-S4Wave: Self-Supervised Learning of Physiological Waveforms with Structured State Space Models

**arXiv ID:** 2606.19888 | [PDF](https://arxiv.org/pdf/2606.19888v1)

**作者:** Feng Wu `[一作]` (Massachusetts Institute of Technology), Li-wei H Lehman `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出SL‑S4Wave框架，用无监督对比学习方式学习长序列多通道医学波形（如ECG、EEG）并在心律失常报警检测中实现高性能

**💡 创新点**

将结构化状态空间模型（S4）与多尺度全局卷积、残差门控结构相结合，并设计噪声鲁棒与上下文一致的对比损失，突破CNN与Transformer在长序列与噪声条件下的瓶颈

**🔧 技术方法**

S4Wave编码器（SGConv、残差门控）、对比学习（噪声鲁棒损失+上下文一致损失）

**📊 数据集**

PhysioNet MIMIC II、VTaC、Challenge 2015 ECG数据集以及多种EEG任务数据

**📈 对比分析**

与CNN、Transformer、SimCLR、TS‑2Vec、ECGFounder等基线对比，SL‑S4Wave在三大心律失常数据集上实现最高Challenge Score、AUC，并在仅5–10%标注样本时保持或超过所有对比学习模型，展示出优异的标签效率和跨域泛化

**⚠️ 局限性**

对模型解释性有限，需进一步评估在更长序列、多通道数或不同信号频谱下的泛化与鲁棒性；对异常噪声的鲁棒性尚未在极端临床环境中充分验证

---

## 284. Matching Markets meet Cumulative Prospect Theory: Towards Optimal and Adversarially Robust Learning

**arXiv ID:** 2606.19883 | [PDF](https://arxiv.org/pdf/2606.19883v1)

**作者:** Ananya Kunisetty `[一作]` (Indian Institute of Technology Bombay), Avishek Ghosh `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在竞争性两边匹配市场中，研究基于累计前景理论（CPT）的风险敏感决策，并提出了 CPT‑ETGS、改进版 Improved 以及鲁棒变体，求解玩家最优稳定匹配并给出对数阶的 CPT 惯例下的收敛性分析。

**💡 创新点**

①首次把 CPT 作为多玩家 MAB 的损失函数引入两边匹配；②提出利用 CPT 加权的置信区间实现在线消除子优臂，实现对数阶并在 K≫N 时达到下界；③在存在奖励腐败的环境下设计可知/未知腐败预算下的鲁棒算法，仍保持对数阶 CPT 失配。

**🔧 技术方法**

核心技术包括：累计前景理论的概率加权函数（α‑Holder 连续），经验分布的顺序统计估计、DKW 不等式用于构造 CPT 置信区间；自适应臂消除策略；多层 ETGS 机制用于未知腐败预算；改写的 Gale‑Shapley 匹配流程。

**📊 数据集**

实验数据为合成市场，奖励为 Bernoulli 分布，arm 的 CPT 值与均值随机生成，规模从 N=2,K=4 到 N=K=3 的腐败场景；使用固定的 Lipschitz CPT 加权函数 w(p)=p+1.5p(1-p)(1-2p)。

**📈 对比分析**

与基线忽略 CPT 的算法相比，后者在匹配和 regret 上表现差异显著：忽略 CPT 的算法产生线性 regret；改进版 Improved 通过臂消除显著降低 regret；鲁棒算法在已知或未知腐败预算下均保持对数阶，实验结果与理论上限一致。

**⚠️ 局限性**

主要限制：仅在合成数据上验证；假设奖励分布满足 Hölder 连续的权重函数；模型侧重于 K≥N 的竞争性匹配，未考虑动态或非平稳市场；CPT 参数（α、L）需先验设定。

---

## 285. Neural Events: Discrete Asynchronous Autoencoders for Event-Based Vision

**arXiv ID:** 2606.19835 | [PDF](https://arxiv.org/pdf/2606.19835v1)

**作者:** Roberto Pellerito `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种离散异步编码器（Discrete Asynchronous Encoder），通过对原始事件流进行可重标记（retokenization），将高频低信息事件压缩为稀疏、语义丰富的神经事件；

**💡 创新点**

创新点包括：①将事件流映射到离散码本并仅在码本切换时触发事件，实现在时序上极低的数据率；②使用线性注意力（RWKV-7）实现高效的时序建模；③结合重构、速率对齐和潜在平滑三种无监督预训练损失，稳定码本并进一步压缩事件率；

**🔧 技术方法**

采用的技术包括：线性注意力Transformer（RWKV-7）、Gumbel-Softmax离散化、离散变分自编码器框架、基于码本的tokenization、无监督预训练与多任务学习；

**📊 数据集**

使用的数据集有：DSEC-Detection（汽车场景）、Gen1（大规模汽车检测）、N-Caltech101（事件分类），并在这些数据集上进行评估；

**📈 对比分析**

与同期基线（如Inception+SSD、Events+YOLOv3/v4、Swin-T+YOLOX）、异步基线（如DAGr）以及A2S方法比较。结果表明：在DSEC上，TokDAGr在保持或提升mAP（+9.0）同时大幅降低计算与能耗；在Gen1和N-Caltech101上，TokSwinT在mAP、能耗与MFLOPS/ev方面均优于现有同步、异步和A2S方法；

**⚠️ 局限性**

局限性包括：与大参数帧基方法相比在复杂任务上仍有性能差距；固定非重叠空间分块限制了跨边界动态物体的捕捉；未来工作需探索动态分块、上下文自适应码本及在传感器端直接实现神经事件生成。

---

## 286. Query-aware Routing for Filtered Approximate Nearest Neighbors Search

**arXiv ID:** 2606.19898 | [PDF](https://arxiv.org/pdf/2606.19898v1)

**作者:** Qianqian Xiong `[一作]` (Australian National University), Mengxuan Zhang `[通讯]` (Australian National University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了查询感知的路由框架，利用轻量化的ML回归模型预测每个候选过滤ANN方法在当前查询上的召回率，并从离线构建的性能表中挑选最佳方法和参数设置，进而动态完成方法与参数的选择。

**💡 创新点**

创新点在于将特征缩减、回归预测与离线性能表结合，形成跨数据集、跨谓词类型的 per‑query 路由器；该路由器既能获得近似最优的召回‑QPS 权衡，又能在实际查询中保持极低的推理延迟。

**🔧 技术方法**

技术实现包括：① 仅用三维特征（查询选择率、LID、谓词类型）训练多任务 MLP 回归模型；② 离线对每个方法在不同参数下的召回和 QPS 进行基准化，构建查找表；③ 查询时先用 Roaring 位图快速计算选择率，再进行模型推理并通过查找表选取最快且满足召回阈值的配置。

**📊 数据集**

实验在六个真实世界训练集（arxiv、yfcc、LAION‑1M、tripclick、ytb_audio、ytb_video）和五个未见验证集（三个 synthetic、Yahoo800k、DBpedia560k）上进行，覆盖多种维度、标签规模与谓词类型。

**📈 对比分析**

与十种现有过滤 ANN 方法及 RuleRouter 基线比较，实验结果显示路由器在五个验证集上的平均 recall@10≈0.986，QPS 大幅高于任何单一方法，且路由开销仅约 54 µs/查询；在大多数谓词类型与数据集组合上取得了全局近似最优的 recall‑QPS Pareto 曲线。

**⚠️ 局限性**

限制包括：① 仅针对分类标签过滤，未覆盖数值区间（range）过滤；② 需要预先对每个方法和参数进行离线基准，耗时且受数据集变动影响；③ 对极低召回阈值或极大 LID 场景下仍可能选择较慢的方式导致轻微延迟；④ 在完全未知的极端数据分布下的泛化能力尚未完全验证。

---

## 287. Kolmogorov-Arnold Reservoir Computing

**arXiv ID:** 2606.19984 | [PDF](https://arxiv.org/pdf/2606.19984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 288. A Systematic Evaluation of Black-Box Uncertainty Estimation Methods for Large Language Models

**arXiv ID:** 2606.19868 | [PDF](https://arxiv.org/pdf/2606.19868v1)

**作者:** Jiayi Wang `[一作]` (University of Chinese Academy of Sciences), Xu-Yao Zhang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估黑盒不确定性估计方法在大语言模型上的效果，构建统一实验框架并对24种方法进行比较。

**💡 创新点**

通过系统化的分类（verbalization、sampling、explanation、多代理、hybrid）和统一评测，填补了缺乏统一比较的空白。

**🔧 技术方法**

使用公开的LLM、NLI和嵌入模型、采样与提示扰动等技术生成外部信号并计算不确定性。

**📊 数据集**

在TriviaQA、HotpotQA、CoQA（开放式问答）和TruthfulQA（闭合式问答）四个基准上进行评测。

**📈 对比分析**

结果显示没有单一方法始终占优，基于答案空间比较和混合信号的方案（如VPD、SteerConf、DiNCo）在不同设置下表现最好。

**⚠️ 局限性**

受限于仅使用外部可观测输出、方法实现的多样性与对不同任务的适用性不一致，导致部分方法在高精度或特定数据集上表现不稳定。

---

## 289. Learning Alternating Real-Time Automata

**arXiv ID:** 2606.19822 | [PDF](https://arxiv.org/pdf/2606.19822v1)

**作者:** Kazuki Kinoshita `[一作]` (Kyoto University), Masaki Waga `[通讯]` (Kyoto University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了第一套针对交替实时自动机（ARTAs）的主动学习算法，基于L*风格的观测表，能够通过成员资格和等价查询准确识别目标实时语言。

**💡 创新点**

创新点在于：① 证明交替分支可以显著提升实时自动机的简洁性（与确定性RTA相比可实现双指数压缩）而不增加表达能力；② 设计了将观测表从字母扩展到时序词的“证据闭合”机制，并通过二进制整数规划实现最小化的单调基；③ 通过从证据AFA到ARTA的构造，自动生成时间区间门限。

**🔧 技术方法**

核心技术包括：L*算法的变体（AL*RTA）、观测表的时间延迟扩展、证据闭合与floor‑distinct性约束、BIP求解最小单调基、计数器例化与规范化、基于分区函数的时间区间推导。

**📊 数据集**

实验使用了公开的190个随机生成的NRTAs基准，涵盖17组（不同状态数、字母表大小和时序常数≤20），并在Ubuntu 24.04服务器上评估。

**📈 对比分析**

与NRTALearning（学习NRTAs）的对比显示：ARTAs学习得到的自动机状态更少，但等价查询和成员查询数量显著增加，整体运行时间也更长；在大字母表场景中差距尤为明显。

**⚠️ 局限性**

局限性包括：① 需要更多查询导致学习成本升高；② 复杂的单调基最小化导致时间消耗大；③ 只针对ARTAs，没有扩展到其他时序自动机子类或符号自动机；④ 对等价查询的完整性假设较强，实际应用可能受限。

---

## 290. TelcoAgent: A Scalable 5G Multi-KPM Forecasting With 3GPP-Grounded Explainability

**arXiv ID:** 2606.19821 | [PDF](https://arxiv.org/pdf/2606.19821v1)

**作者:** Geon Kim `[一作]` (Kyung Hee University), Vijay K. Shah `[通讯]` (North Carolina State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出TelcoAgent框架，实现5G多KPM预测与可解释诊断，无需站点特定训练。

**💡 创新点**

将零射击时间序列基础模型与自动构建的3GPP知识图结合，并通过ReAct LLM实现因果推理与可操作建议。

**🔧 技术方法**

使用3GPP知识图自动化构建、Chronos‑2/Moirai等TSFM、ReAct LLM代理、PAX‑TS敏感度分析以及OpenStreetMap空间上下文。

**📊 数据集**

使用美国运营商真实3个月、200基站、1小时粒度的5G KPM数据集。

**📈 对比分析**

与六个监督基准（N‑BEATS、GRU、MLP等）比较，Chronos‑2零射击模型在所有7个KPM上nRMSE最低（0.12‑0.19），解释性指标Faithfulness≈0.62，Answer Relevancy≈0.81，显示优于传统模型。

**⚠️ 局限性**

对知识图阈值敏感、LLM可能产生少量事实偏差、未考虑邻近基站空间上下文以及对不同运营商的泛化性尚待验证。

---

## 291. SketchKeyAnime: Reference-anchored Sparse Key-Sketch Animation Synthesis

**arXiv ID:** 2606.19958 | [PDF](https://arxiv.org/pdf/2606.19958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 292. ROSE: Benchmarking the Perception-to-Action Gap in Multimodal Models

**arXiv ID:** 2606.19965 | [PDF](https://arxiv.org/pdf/2606.19965v1)

**作者:** Yihao Wang `[一作]` (Sun Yat-sen University), Keze Wang `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个受控基准 ROSE，用于检验多模态大型语言模型（MLLMs）在同一视觉场景下如何将视觉信息转化为上下文相关的符号动作，并对模型在计数与坐标动作任务间的表现差异进行诊断。

**💡 创新点**

创新点在于：① 通过保持视觉场景不变、仅变换区域约束和输出形式，设计了多任务耦合的评测；② 将计数、区域计数、局部点击、视觉区域点击和排除点击等五种任务模板统一在同一场景上；③ 通过全球点击桥、匹配计数-点击对比等诊断手段拆解感知到动作的瓶颈，揭示了“计数‑动作”差距的两大组成——坐标定位与上下文绑定。

**🔧 技术方法**

使用的技术包括：多模态大型语言模型推理（如 GPT‑5.5、Gemini‑3.1‑Pro、Qwen‑3.6‑Plus 等）；基于图像-文本协议的统一提示模板；精确的符号输出格式验证与 PASS/VALID/SOFT 评估指标；以及一系列自定义的诊断任务（全球点击桥、匹配计数-点击、错误类型分解等）。

**📊 数据集**

数据集为 ROSE v0.1，包含 1,512 个场景（3,024 张图像），5 种视觉来源（中文字符、Emoji 样式、Emoji 内容、像素艺术编辑、像素艺术内容），并在每个场景上生成 5 个耦合任务，总计 7,560 个任务实例。

**📈 对比分析**

与人类参考（98.8% PASS）相比，九种 MLLM 的表现差距显著：GPT‑5.5 最高 92.2% PASS，Gemini‑3.1‑Pro 79.4%，其余模型 14.3%–50.3%；计数任务的平均 PASS 远高于动作任务，且在计数正确的场景下动作仍出现 17.8%–38.0% 的失败；诊断显示，坐标定位仅占差距的一部分，主要瓶颈在上下文绑定和精准执行。

**⚠️ 局限性**

局限性包括：① 基准仅覆盖网格化的符号场景，未能反映自然图像、自由文本、长时序规划或交互反馈的复杂性；② 评测关注的是“单次执行”而非连续动态行为；③ 目前的视觉来源和任务模板有限，缺乏更广泛的视觉类别和更丰富的引用关系；④ 结果仅反映特定模型版本与推理配置，不能作为模型长期排名的依据。

---

## 293. PhysDrift: Bridging the Embodiment Gap in Humanoid Co-Speech Motion Generation

**arXiv ID:** 2606.19935 | [PDF](https://arxiv.org/pdf/2606.19935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 294. Towards Graph-Based Deep Learning for Map Generalization: Insights from Building Footprints Simplification and Aggregation

**arXiv ID:** 2606.19956 | [PDF](https://arxiv.org/pdf/2606.19956v1)

**作者:** Yanning Wang `[一作]` (Hong Kong University of Science and Technology), Yu Feng `[通讯]` (Mainz University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将建筑图斑简化与聚合任务分别转化为节点位移预测与边预测，构建统一图学习框架实现自动地图泛化。

**💡 创新点**

创新点在于：①把聚合任务视为边预测、简化视为节点回归；②在同一模型中联合学习两类任务并用多任务损失平衡；③通过Hamiltonian环后处理评估聚合的拓扑合法性。

**🔧 技术方法**

采用图卷积网络（GCN、GAT、GraphSAGE）实现特征聚合；使用多任务损失、边状态编码、三角网Delaunay约束、Hamiltonian环修正；并加入相对几何特征和注意力机制。

**📊 数据集**

使用德国斯图加特建筑图斑数据，包含1:10,000–1:25,000和1:10,000–1:15,000两级比例，标注一对一和一对多对应关系并转化为图结构。

**📈 对比分析**

与三种主流GNN架构进行对比，评价指标包括链接预测准确率、F1、闭合率、节点位移均方误差等。实验显示GraphSAGE在准确率、闭合率、MSE等方面略优于GCN和GAT，尤其在较大尺度转换时保持更稳定的性能。

**⚠️ 局限性**

主要局限：①节点位移预测仍受数据不平衡与大幅位移稀缺影响，误差较大；②聚合后仍需后处理以满足拓扑约束；③多任务训练可能存在目标冲突，未必优于单任务；④缺乏对复杂地形和多类别对象的广泛验证。

---

## 295. GEMS: Geometric Constraints Enable Multi-Semantic Superposition in LLMs

**arXiv ID:** 2606.19946 | [PDF](https://arxiv.org/pdf/2606.19946v1)

**作者:** Yu Deng `[一作]` `[通讯]`, Yu Deng

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提出 GEMS 方法，利用几何约束在推理时多方向激活注入，实现多语义并行控制。

**💡 创新点**

发现多方向注入崩溃源于分布偏差和方向干扰，并设计范数保持、o_proj 注入、实时正交化三种几何约束，使得训练‑free 多方向调节稳健可行。

**🔧 技术方法**

采用对比激活差向量提取、Gram‑Schmidt 正交化、范数约束的加权组合、Gaussian envelope 层级强度调制以及 o_proj 钩子插入等技术。

**📊 数据集**

在 Qwen3.5‑4B、Llama‑3.2‑3B、Qwen3.6‑27B、Gemma‑4‑31B 上评估，使用 GSM8K、Wikitext‑2 以及自定义 PR/道德情境提示进行实验。

**📈 对比分析**

与 ActAdd 无约束、多方向/单方向基线对比，GEMS 在 GSM8K 上达到 98% 正确率（基线 92%），在 Wikitext‑2 的 PPL 仅提升 2.2%，而无约束 ActAdd 仅 4% 准确率/极高 PPL，表明 GEMS 具备稳健性能。

**⚠️ 局限性**

仅验证了 3 个并行方向、固定权重比例，未系统评估不同模型层范围、量化模型；且只能在模型已具备的能力空间内重塑输出，无法补全缺失能力。

---

## 296. MobileForge: Annotation-Free Adaptation for Mobile GUI Agents with Hierarchical Feedback-Guided Policy Optimization

**arXiv ID:** 2606.19930 | [PDF](https://arxiv.org/pdf/2606.19930v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 297. Co-policy: Responsive Human-Robot Co-Creation for Musical Performances

**arXiv ID:** 2606.19914 | [PDF](https://arxiv.org/pdf/2606.19914v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 298. PolSeT: Polish Semantics of Timbre Dataset

**arXiv ID:** 2606.19987 | [PDF](https://arxiv.org/pdf/2606.19987v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 299. ENPIRE: Agentic Robot Policy Self-Improvement in the Real World

**arXiv ID:** 2606.19980 | [PDF](https://arxiv.org/pdf/2606.19980v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 300. Structure-Oriented Randomized Neural Networks for Poisson-Nernst-Planck and Poisson-Nernst-Planck-Navier-Stokes Systems

**arXiv ID:** 2606.19912 | [PDF](https://arxiv.org/pdf/2606.19912v1)

**作者:** Yunlong Li `[一作]`, Fei Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计了一种面向结构的随机神经网络(SO‑RaNN)方法，用于求解Poisson–Nernst–Planck(PNP)及其耦合电流体(PNP‑NS)方程，结合原始RaNN子问题求解与正性裁剪、质量校正及SAV辅助变量校正，并在PNP‑NS中使用SP‑RaNN实现速度场无散性；

**💡 创新点**

创新点在于将结构导向的校正步骤（正性裁剪、离散时间质量匹配、SAV变量单调性）与随机神经网络相结合，并通过SP‑RaNN实现点态无散速度，此外给出残差基误差估计和局部Picard收敛分析；

**🔧 技术方法**

主要使用技术包括随机神经网络（RaNN）与流量无散构造SP‑RaNN、残差基最小二乘求解、正性裁剪、离散时间质量校正、SAV辅助变量修正以及空间时间分块采样和测量比例采样策略；

**📊 数据集**

实验使用制造解析解的源驱动及标准边界条件进行精度验证，未采用真实物理数据集；

**📈 对比分析**

通过与经典二维/三维有限差分求解器对比，SO‑RaNN在相同网络宽度下误差更低、CPU时间更短，且在长时间模拟中保持稳定性并能实现能量耗散；

**⚠️ 局限性**

主要局限在于对PNP‑NS耦合迭代的全局收敛性未给出完整证明，且对不兼容的Neumann Poisson子问题仅提供条件估计；网络初始化对误差影响显著，缺乏自适应宽度或更完善的误差控制策略。

---

## 301. Multi-Agent Transactive Memory

**arXiv ID:** 2606.19911 | [PDF](https://arxiv.org/pdf/2606.19911v1)

**作者:** To Eun Kim `[一作]` (Carnegie Mellon University), Fernando Diaz `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在多智能体异构系统中提出并实现了 Multi-Agent Transactive Memory (MATM)，允许代理将交互轨迹写入共享存储，并在需要时检索以提升任务性能和执行效率。

**💡 创新点**

创新点在于将传统检索增强生成思想推广到代理生成的轨迹，构建了一个两侧市场的群体级记忆共享框架，并结合轻量级学习排序实现跨代理、跨任务的经验再利用。

**🔧 技术方法**

技术实现包括基于状态条件的键值轨迹索引、密集检索 + 学习重排序 (LTR) 的检索管道，以及在 ALFWorld 与 WebArena 两大交互式环境中进行的实验评测。

**📊 数据集**

使用的数据集为 ALFWorld（训练3553 episode、测试274 episode）和 WebArena（训练724 episode、测试88 episode），以及公开轨迹与自生成轨迹共同构成 MATM 的初始和增量索引。

**📈 对比分析**

与无检索基线对比，检索提升 ALFWorld 成功率约+8%并减少 0.59 步，WebArena 成功率约+2%并减少 1.7 步；再加上学习排序后，ALFWorld 成功率提升至 64.3%并进一步减少步骤，WebArena 达到 20.5% 并显著降低步骤数，整体表现通过 RPP 指标进一步验证。

**⚠️ 局限性**

局限性包括实验仅覆盖两大环境和 34 个消费者模型，未验证跨 benchmark 的通用性；仅聚焦消费者侧，未评估生产者福利或潜在恶意轨迹的风险；LTR 训练采样范围有限，未覆盖全部排名位置。

---

## 302. Semi-Automatic Correction of 3D Tubular Structure Skeletons via Component-Wise MST and Filtered Delaunay Triangulation

**arXiv ID:** 2606.19949 | [PDF](https://arxiv.org/pdf/2606.19949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 303. Semantic-Anchored Evidential Fusion for Domain-Robust Whole-Slide Survival Analysis

**arXiv ID:** 2606.19966 | [PDF](https://arxiv.org/pdf/2606.19966v1)

**作者:** Yucheng Xing `[一作]` (National University of Singapore), Mengling Feng `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于语义锚点与证据融合的全切片生存预测框架（SAEFS），通过视觉‑语言模型生成域不变的VQA语义锚点，结合双流视觉证据与文本证据，并使用Dirichlet‑基础的主观逻辑和谨慎结合规则进行不确定性建模与融合，从而实现仅用源域数据即可零样本跨中心泛化的生存分析；

**💡 创新点**

(1) 利用模板化VQA自动获取高层病理语义锚点；(2) 设计双流结构（全景视觉+文本引导视觉）提升对预测相关区域的捕获；(3) 将Dirichlet分布与主观逻辑用于不确定性建模，并采用谨慎结合规则处理相关证据，避免过度自信；(4) 通过上述设计实现了在四个未见中心的零样本高鲁棒性；

**🔧 技术方法**

视觉‑语言模型（VLM）编码器；多实例学习（MIL）+注意力聚合；文本引导的相似度加权特征聚合；Dirichlet分布 + Subjective Logic；谨慎结合（cautious conjunction）规则；生存损失（Hazard / Survival）+ KL正则；MMD、Kaplan‑Meier、校准曲线等评估手段；

**📊 数据集**

源域：TCGA（肺腺癌LUAD、子宫内膜癌UCEC、肾透明细胞癌KIRC）；目标域（四个未见中心）：CPTAC-LUAD、CPTAC-UCEC、CPTAC-KIRC、NLST-LUAD；在所有实验中仅使用TCGA训练，目标域数据完全不参与训练；

**📈 对比分析**

与8种单模（ABMIL、TransMIL、DSMIL、ILRA、BayesMIL、UMSA、ACMIL、OTSurv）和4种多模（MCAT、MOTCAT、SurvPath、PS3）基线进行零样本评估；SAEFS平均C‑index 0.671，比最优对手高0.062；IBS和INBLL最低；在高域偏移集如CPTAC‑LUAD提升尤为显著；Ablation实验进一步证明各模块对性能贡献；

**⚠️ 局限性**

仍依赖预训练的VQA模型与固定模板问答，可能无法覆盖所有病理细节；对罕见病理亚型的泛化尚未验证；框架在源域多样性不足时可能受限；未考虑目标域适配的在线调优；未来需探索更细粒度语义生成及VLM不确定性评估。

---

## 304. Stellar: Scalable Multimodal Document Retrieval for Natural Language Queries

**arXiv ID:** 2606.19960 | [PDF](https://arxiv.org/pdf/2606.19960v1)

**作者:** Yuxiang Guo `[一作]` (Zhejiang University), Yunjun Gao `[通讯]` (Zhejiang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Stellar，一个通过词汇稀疏表示与磁盘后端延迟交互相结合的可扩展多模态文档检索框架。

**💡 创新点**

创新点包括：① 使用 MLLM 预训练的 next‑token 预测头做词汇投影并通过稀疏化得到高效的过滤表示；② 通过平衡聚类在磁盘上构造语义局部、大小均衡的块，减少随机 I/O；③ 采用成本感知的加载策略动态选择全块加载或特定向量加载；④ 通过稀疏与稠密分数融合进一步提升检索效果。

**🔧 技术方法**

核心技术包括多模态大语言模型 (MLLM) + LoRA 微调、稀疏词汇编码 + 逆索引、token‑级多向量表示 + 迟延交互、平衡聚类 + 磁盘块布局、成本模型驱动的 I/O 选择、分数融合。

**📊 数据集**

使用了五个数据集：DocVQA、InfoVQA、ArXivQA、PlotQA 以及自建的大规模 400k 文档/94 题 LargeDoc。

**📈 对比分析**

与现有单向量、稠密向量和近似多向量检索方法（如 ColPali、ColPali‑PLAID）对比，Stellar 在 R@1、R@10、M@10 上保持或超过 SOTA，且在内存占用和查询延迟上相较 ColPali 下降 1–2 个数量级，On‑disk 方案实现了 0.96 GB 内存和 <120 ms 的平均查询时延。

**⚠️ 局限性**

局限性：仅支持单文档检索，未处理多文档答案聚合；当前实现仅在 CPU 上完成检索，磁盘 I/O 仍是潜在瓶颈；对极大规模或实时场景的硬件加速（如预取、缓存）仍需进一步探索。

---

## 305. Speeding up the annotation process in semantic segmentation industrial applications

**arXiv ID:** 2606.19934 | [PDF](https://arxiv.org/pdf/2606.19934v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. Deep-Unfolded Coordination

**arXiv ID:** 2606.19920 | [PDF](https://arxiv.org/pdf/2606.19920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 307. AutoTam: Specifying Secure Protocol Implementations with Tamarin Model Generation

**arXiv ID:** 2606.19937 | [PDF](https://arxiv.org/pdf/2606.19937v1)

**作者:** Johannes Wilson `[一作]` (Sectra Communications), Niklas Johansson `[通讯]` (Sectra Communications)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了 AutoTam：一种面向加密协议的领域特定语言、其解释器以及将实现直接翻译成 Tamarin 模型的工具，并结合符号执行对实现进行内存安全验证；在 Signed Diffie‑Hellman 和 WireGuard 两个协议上实现并证明安全。

**💡 创新点**

创新点在于：①语言先行，协议实现与符号模型在同一抽象层面对齐；②通过可执行的状态机描述，实现自动、可证的 Tamarin 规则生成，支持循环和完整状态机；③集成符号执行以检测实现层的内存错误，形成完整的验证链。

**🔧 技术方法**

使用的技术包括：AutoTam DSL 与 C‑实现的解释器；Tamarin 证明器进行符号协议验证；KLEE 符号执行引擎对实现进行动态路径探索；HACL+ 加密库实现实际加密函数。

**📊 数据集**

使用的数据集为两条协议案例：一条 Signed Diffie‑Hellman 交换协议（包含签名验证）和 WireGuard VPN 协议（Noise IKpsk2 基础的握手与传输），并在这些协议上生成模型与实现。

**📈 对比分析**

比较方法：与已有的 WireGuard Tamarin 模型对比，AutoTam 生成的模型在细节保真度更高（完整的 HKDF 细化、AEAD 解密验证等）；验证时间：DH 约 3 s，WireGuard 约 360 s（加上源子句）；实现吞吐量：AutoTam 解释器约 600 Mbit/s，接近官方 Go 实现，略低于 Linux kernel；符号执行覆盖率：无内存错误，路径数与实现规模成正比。

**⚠️ 局限性**

局限性：①对 WireGuard 需要手工编写 source lemma 以完成验证；②实现缺失 DoS 保护、窗口管理、并发读写及完整 TUN 设备；③仅验证握手阶段的安全性，传输阶段未完全验证；④对大型循环协议仍需人工调整状态机划分以保持可验证性。

---

## 308. Prismriver: Formalization of Music Theory and Algorithmic Composition in Lean 4

**arXiv ID:** 2606.19936 | [PDF](https://arxiv.org/pdf/2606.19936v1)

**作者:** Leni Aniva `[一作]` (Stanford University), Claire Wang `[通讯]` (University of Pennsylvania)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本论文提出并实现了 Prismriver，一套基于 Lean 4 的音乐理论形式化与算法合成框架，涵盖音高、音阶、和弦、时值、部件等基础概念，并支持非等音阶（xenharmonic）与自定义调式，能够与 Mathlib、Alda、MIDI、MusicXML 等生态无缝集成；同时提供了 monadic 组合子用于算法创作和声部对位的可验证实现。

**💡 创新点**

创新点在于：① 在 Lean 4 中构建通用且可组合的音乐抽象模型，①1 通过群作用实现转置与反射的统一操作；② 通过证明转置与十二平均律下的二面体群等价，提供可验证的和声推导；③ 设计 monadic 接口支持算法生成、分析与合成；④ 兼容 xenharmonic 调式与多重调性，突破传统仅限 12‑TET 的限制。

**🔧 技术方法**

主要技术包括 Lean 4 定理证明器与 Mathlib 库、Alda 音乐播放后端、LilyPond 风格可扩展语法、MusicXML 输出、Monad 组合子、泛型群作用（转置、反射）、泛型音阶/调式抽象、以及与现有音乐 DSL（Tidal、Strudel、Lulu）对接。

**📊 数据集**

论文未使用外部数据集，主要以形式化定义与示例代码为主；所有演示均基于内部构造的音阶、和弦与乐句。

**📈 对比分析**

论文中未给出量化性能比较；通过 #play 指令演示音频播放，并通过与传统 Tidal/Strudel 等 DSL 的功能对比说明，示例演示了算法合成与对位验证，但未进行系统化的速度或资源占用评测。

**⚠️ 局限性**

主要局限包括：① 目前仅实现了有限的音阶与调式，缺乏对所有 xenharmonic 系统的完整支持；② 缺少大规模实验与基准评测；③ 对实时交互与高效音频渲染的支持尚不完善；④ 仅关注形式化与合成，尚未集成完整的可视化与编辑工具。

---

## 309. Addressing Detail Bottlenecks in Latent Diffusion for RGB-to-SWIR Image Translation

**arXiv ID:** 2606.19961 | [PDF](https://arxiv.org/pdf/2606.19961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 310. eCNNTO: A Highly Generalizable ConvNet for Accelerating Topology Optimization

**arXiv ID:** 2606.19921 | [PDF](https://arxiv.org/pdf/2606.19921v1)

**作者:** Shengbiao Lu `[一作]` (Shanghai Jiao Tong University), Xiaodong Wei `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于元素级卷积神经网络（eCNNTO）的加速拓扑优化方法，利用CNN捕捉邻域密度演化并在早期迭代后直接预测近似最优结构。

**💡 创新点**

创新点包括：①引入残差CNN实现空间相关性建模，显著提升结构连通性；②采用“最终阶段”特征进行训练，显著减少所需训练样本；③在单一模型下实现对不同边界条件、加载、网格和非设计域的强泛化。

**🔧 技术方法**

主要技术：卷积神经网络（ResNet结构）、批归一化、ReLU、全连接分类头、交叉熵损失，训练使用Adam；输入为元素窗口的密度演化序列，输出为离散密度类别。

**📊 数据集**

数据集来源于仅两类二维和两类三维基准问题（长梁、简支梁、圆孔梁、桥梁等），共生成约53,200条二维样本和500,000条三维样本；数据规模相对较小。

**📈 对比分析**

与传统SIMP和DLTOP对比：在二维/三维测试中迭代次数分别降低90%/97%，运行时间提升≈80–95%，结构连通性更好，分类误差更低，且所需训练样本仅为DLTOP的11%。

**⚠️ 局限性**

局限性：目前仅适用于结构化网格；对极端几何或多物理耦合的拓扑优化仍需进一步验证；未来可能引入图神经网络或多物理约束以提升鲁棒性。

---

## 311. Modest, artistic, and radical solutions to the environmental impact of image-generating machine learning

**arXiv ID:** 2606.19957 | [PDF](https://arxiv.org/pdf/2606.19957v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 312. Design and Evaluation of Energy-Efficient Whisper Dot-Product Kernel Offloading on a CGLA Architecture

**arXiv ID:** 2606.19913 | [PDF](https://arxiv.org/pdf/2606.19913v1)

**作者:** Takuto Ando `[一作]` (Nara Institute of Science and Technology), Yasuhiko Nakashima `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

实现并评估了 Whisper 语音识别模型中点乘核的硬件卸载，在 IMAX CGLA 架构上实现了 FP16 与 Q8_0 两种精度的加速。

**💡 创新点**

创新点包括：1) 设计了可在线将 FP16 转为 FP32 的点乘核，利用 2 方向 SIMD FMA 和列向多线程并行；2) 引入混合执行策略，将对齐向量交给 IMAX 处理，剩余残量在 CPU 上完成；3) 通过 LMM（32 KB）与 burst 长度 16 的组合，针对 tiny 模型实现 PDP/EDP 的最优平衡；4) 通过敏感度分析展示 burst 长度与 LMM 规模对能耗与延迟的影响。

**🔧 技术方法**

技术手段包括：C++/whisper.cpp 推理栈、IMAX 线性 64‑PE 数组、双缓冲本地内存模块、DMA 直传、定制指令 OP_SML8 与 OP_AD32、FPGA 原型（Xilinx Versal VPK180）和 28 nm ASIC 投影，TDP‑基准下的跨平台性能对比（Jetson AGX Orin、RTX 4090）。

**📊 数据集**

使用 Whisper‑tiny.en、base.en、small.en 三种开源模型；对 21 条 LibriSpeech test‑clean 语句进行转录一致性和延迟校验。

**📈 对比分析**

对比方法：在同一推理堆栈、相同模型文件的基础上，测量端到端延迟和功耗（GPU 采用 TDP，FPGA 采用测量值，ASIC 采用综合估计），计算 PDP。结果显示：Whisper‑tiny.en Q8_0 下，IMAX 的 PDP 为 11.58，比 Jetson AGX Orin 的 27.16 低 2.35×，比 RTX 4090 的 121.38 低 10.48×；随着模型增大，PDP 差距缩小甚至逆转。

**⚠️ 局限性**

限制：1) GPU 能耗基于 TDP，未直接测量；2) ASIC 估计基于 10 % 交替切换，缺乏后仿真和硅波动；3) 仅卸载点乘核，非线性运算仍在 CPU 上，无法实现实时解码；4) 实验使用单一 10 s 音频片段，未覆盖不同长度、噪声、语言等场景；5) LMM 规模与静态功耗权衡，当前 32 KB 不是所有模型的最优点。

---

## 313. Motor Angular Speed Preintegration for Multirotor UAV State Estimation

**arXiv ID:** 2606.19929 | [PDF](https://arxiv.org/pdf/2606.19929v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 314. DiffMath: Symbol- and Graph-Aware Latent Diffusion Transformer for Handwritten Mathematical Expression Generation

**arXiv ID:** 2606.19939 | [PDF](https://arxiv.org/pdf/2606.19939v1)

**作者:** Wei Pan `[一作]` (South China University Of Technology), Lianwen Jin `[通讯]` (South China University Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了DiffMath框架，利用符号与图结构的潜在扩散模型生成手写数学表达式。

**💡 创新点**

创新点在于提出RelAST结构化先验、Symbol/Relation感知VAE以及全局符号计数的AdaLN调控，突破了对密集位置信息的依赖。

**🔧 技术方法**

使用了结构化抽象语法树、变分自编码器、潜在扩散Transformer、AdaLN、CTC、GMM等技术。

**📊 数据集**

采用MathWriting和CROHME两个公开数据集进行训练与评估。

**📈 对比分析**

与FormulaGAN、DiffInk、One‑DM、SD‑XL等SOTA方法在ExpRate、BLEU、FID等指标上进行了定量和定性对比，DiffMath在70.70% ExpRate、5.43 FID等方面显著优于对比模型。

**⚠️ 局限性**

对稀有符号、极深嵌套结构和高密度文本区域仍存在生成错误，模型对极其复杂布局的鲁棒性尚待提升。

---

## 315. CARE: Competence-Aware Reward Shaping for Adaptive Reasoning Length in Video-MLLMs

**arXiv ID:** 2606.19927 | [PDF](https://arxiv.org/pdf/2606.19927v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 316. Light-weight Pronunciation Assessment via Discrete Speech Token Surprisal

**arXiv ID:** 2606.19910 | [PDF](https://arxiv.org/pdf/2606.19910v1)

**作者:** Syeda Faiza Ahmed Sara `[一作]` (Qatar Computing Research Institute), Shammur Absar Chowdhury `[通讯]` (Qatar Computing Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种只使用母语语料、无需音素库或强制对齐的轻量级发音评估框架，结合离散语音单元惊讶度与文本引导的DTW对非母语语音进行无监督或轻监督评分。

**💡 创新点**

创新点在于：1）完全基于母语数据学习离散单元词典和音韵先验；2）利用自监督Encoder+K‑means离散化生成离散单元；3）用n‑gram Token LM计算惊讶度作为发音偏差指标；4）加入Text2DUnit‑DTW在离散空间进行文本对齐，避免了音素对齐与标注；5）框架兼容零样本、轻监督与多任务评估。

**🔧 技术方法**

技术包括自监督学习（HuBERT、CANINE）、K‑means聚类码簿、n‑gram Token LM、动态时间规整（DTW）、Ridge回归、离散单元特征统计、惊讶度计算与加权标准差。

**📊 数据集**

训练使用LibriSpeech 960 h的英语母语语料；评估在SpeechOcean762（250名非母语汉英说话者）上进行；交叉验证在L2‑ARCTIC（24名非母语英语说话者）上完成。

**📈 对比分析**

与GoP、DeepFeature、GOPT、MultiPA、HMamba等传统与深度学习基线对比。零监督DTW距离单独取得Accuracy PCC 0.633；轻监督结合文本对齐的Ridge模型得到Accuracy 0.661、Fluency 0.763、Prosody 0.753；在L2‑ARCTIC上无重新训练的Ridge模型得到0.506/0.492/0.526，轻监督后提升至0.527/0.519/0.557。

**⚠️ 局限性**

局限性包括：仍以英语母语为基准，针对非英语母语或极低资源环境需进一步验证；模型主要针对朗读式评估，对对话或非文本驱动情境效果未知；需要预训练的离散码簿和Token LM，训练成本不可忽视；对口音多样性和长文本的鲁棒性尚待评估。

---

## 317. Beyond Lower Quota: Avoiding Overrepresentation in Multi-Winner Voting

**arXiv ID:** 2606.19968 | [PDF](https://arxiv.org/pdf/2606.19968v1)

**作者:** Anton Baychkov `[一作]` (University of Warwick), Jannik Peters `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了批准制多赢家选举中的过度代表问题，并提出了“正当上限配额”（JUQ）和“正当近似配额”（JNQ）两种新的比例代表性公理，定义并分析了复合 Thiele 规则，证明了 Adams‑AV 是唯一满足 JUQ 的规则，并给出了多项式时间的 UQER 算法来生成满足 JUQ 的完整委员会。

**💡 创新点**

创新点在于：①首次把过度代表问题正式建模并提出可验证的 JUQ 公理；②引入复合 Thiele 规则并证明其在满足上限配额方面的唯一性；③提出 JNQ 公理以平衡欠缺与过度代表，并证明 SLAV 规则是唯一满足该公理的复合 Thiele 规则；④设计了 UQER（上限配额消除规则）实现多项式时间生成 JUQ 满足的委员会，并探讨了与 EJR+ 等现有比例公理的相互关系。

**🔧 技术方法**

主要技术手段为公理化设计与证明、组合优化（通过组合规则的层级化）、潜在函数和交换动态分析（JUQ 与 JNQ 的交换过程）以及对分配方法（除法方法）的类比和转换。

**📊 数据集**

本文为理论性研究，未使用具体数据集；所有结果均基于抽象的批准投票实例和计数模型。

**📈 对比分析**

由于研究聚焦于理论性质和公理满足性，并未进行实验比较；但作者通过归纳、构造反例与证明展示了所提出规则在满足 JUQ / JNQ 与传统公理（如 EJR+、价格性等）方面的优势与局限。

**⚠️ 局限性**

局限性包括：①对全局最优（满足 JUQ 的完整委员会）是否可多项式时间求解仍未完全解决；②在复合 Thiele 类外的规则中，JUQ 与 JNQ 的满足性与效率难以兼顾；③规则的可解释性与实际投票场景的适用性尚待进一步实验验证。

---

## 318. A Measurement Study of Cryptographic Misuse in Embodied AI Mobile Applications

**arXiv ID:** 2606.19983 | [PDF](https://arxiv.org/pdf/2606.19983v1)

**作者:** Junchao Li `[一作]` (Shandong University), Yue Zhang `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了EAIAppZoo基准集（507个具身人工智能移动应用），并通过语义感知静态分析流水线检测了12,975条加密误用案例。

**💡 创新点**

创新点在于首次系统量化EAI移动端的加密错误，揭示其源于低延迟、离线配网和旧版SDK的工程权衡，并通过案例展示其能直接导致物理攻击链。

**🔧 技术方法**

采用了基于Semgrep的规则匹配、JADX反编译、动态拆包与自定义规则库的静态分析技术，对弱算法、硬编码密钥、明文通信等五类误用进行检测。

**📊 数据集**

使用的数据集为EAIAppZoo，包含六大领域（可穿戴设备、服务机器人、工业/农业机器人、教育/社交机器人、清洁机器人、无人机）共507个真实Android应用。

**📈 对比分析**

通过与传统Android加密误用研究对比，检测精度达80.74%，并在不同领域保持一致，表明方法在规模化测量中的稳健性和可操作性。

**⚠️ 局限性**

局限性包括仅采用静态分析，未覆盖运行时加密细节；数据集未公开，影响可重复性；部分误用由第三方库引发，可能导致误报或漏报。

---

## 319. Advancing DialNav through Automatic Embodied Dialog Augmentation

**arXiv ID:** 2606.19948 | [PDF](https://arxiv.org/pdf/2606.19948v1)

**作者:** Leekyeung Han `[一作]` (Korea University), Paul Hongsuck Seo `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过构建三阶段自动生成管线，将现有VLN数据转换为多轮对话导航数据集 RAINbow，并结合 Dual-Strategy Training 和 Graph‑based Transformer Localization，显著提升 DialNav 训练数据量和模型性能。

**💡 创新点**

创新点包括：1) 端到端自动化生成多轮对话导航数据集的 pipeline；2) 结合 data‑guided 与 on‑policy roll‑outs 的 Dual‑Strategy Training，使导航训练更贴合对话交互；3) 将 VLN 预训练知识迁移到定位任务，改进定位模块。

**🔧 技术方法**

使用的技术包括：视觉–语言模型（LLaVA‑1.5）生成场景描述；LLM（GPT‑4o‑mini）重写成自然对话；Graph‑based Transformer 作为定位模型；DUET 负责导航；LANA 负责问题/答案生成；以及基于 VLN 预训练的参数初始化。

**📊 数据集**

主要使用数据集：原始 VLN 数据集 R2R、RxR、CVDN；合成的 238K episode RAINbow；并与原始 2K episode 的 RAIN 数据进行对比实验。

**📈 对比分析**

实验对比基线（Baseline）、+RAINbow、+DST、+GTL 以及组合模型；在 Val Seen 上 SR 从 30.77 提升至 58.24（+89%），在 Val Unseen 上 SR 从 14.52 提升至 29.05（+100%），同时在其他指标（OSR、SPL、NE、NSC、DTC）也实现显著改善。

**⚠️ 局限性**

局限性：① 依赖现有 VLN 轨迹，可能限制环境与行为多样性；② 生成对话虽然自然但缺乏完整人类交互的丰富性；③ 评估仅在 Matterport3D 室内导航场景，未验证在更广泛的 embodied 场景中的通用性。

---

## 320. Spatial-Aware Reduction Framework: Towards Efficient and Faithful Visual State Space Models

**arXiv ID:** 2606.19932 | [PDF](https://arxiv.org/pdf/2606.19932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 321. Blame is easier than praise: Measuring off-ball defensive performance in football

**arXiv ID:** 2606.19931 | [PDF](https://arxiv.org/pdf/2606.19931v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 322. Timage: A Generative Text-in-Image Paradigm for Fine-Tuning Vision-Language Models

**arXiv ID:** 2606.19944 | [PDF](https://arxiv.org/pdf/2606.19944v1)

**作者:** Yifeng Wu `[一作]` (Fudan University), Ruize Han `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在输入图像上生成语义化、可读的文本覆盖层，将视觉与语言查询在空间上对齐，提升视觉语言模型在细粒度空间推理上的表现。

**💡 创新点**

创新点在于将查询文本直接渲染为受限Schrödinger桥生成的可视化覆盖层，既保持了文本可读性，又遵循前景遮挡约束，彻底解决了传统参数微调与无语义视觉提示的空间对齐不足问题。

**🔧 技术方法**

主要技术包括受限Schrödinger桥（cSB）用于生成文本放置与样式，语义热图与遮挡掩码的空间约束，投影Euler‑Maruyama积分求解，语义引导与任务导向双重损失，及自一致性投票以提升布局多样性。

**📊 数据集**

采用VMCBench作为评测数据集，涵盖20个不同的VQA任务（General、Reasoning、OCR、Doc&Chart等）。

**📈 对比分析**

在VMCBench上与多种基线（纯文本提示、启发式覆盖、VPT、LoRA、全微调、GPT‑4o等）对比，Timage在7B基础模型上平均准确率达到87.7%，比大型专有模型高出约7.4%，比参数微调模型高约2.5%，且在各子任务上均实现显著提升。

**⚠️ 局限性**

局限性包括：在极端纹理或低对比度区域时生成的文本可能不完整或与背景融合，导致可读性下降；渲染透明度偶尔不足，影响提示的可视化效果。

---

## 323. Triangular Consistency as a Universal Constraint for Learning Optical Flow

**arXiv ID:** 2606.19938 | [PDF](https://arxiv.org/pdf/2606.19938v1)

**作者:** Yi Xiao `[一作]` (Louisiana State University), Dong Lao `[通讯]` (Louisiana State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出三角一致性（Triangular Consistency）作为光流学习的通用几何约束，兼容监督、无监督和迁移学习。

**💡 创新点**

创新点在于将光流的位移场可合成性转化为可直接监督的三角关系，形成一种无监督、无标签、无额外网络结构的“plug‑and‑play”损失，并通过解析几何变换实现无插值的数据增强。

**🔧 技术方法**

使用光流场的向量加法与偏移插值实现三角一致性损失；采用鲁棒范数、遮挡掩码以及对称/非对称的仿射变换进行训练；在无监督中与光度重建、双向一致性等常规损失相结合；在自监督适配中利用教师‑学生 EMA 机制；在监督训练中作为控制数据增强。

**📊 数据集**

在合成数据（FlyingChairs、FlyingThings3D）、合成/真实场景（MPI‑Sintel、KITTI）、以及零样本转移数据（HD1K、Middlebury）上进行实验。

**📈 对比分析**

与基线（ARFlow、RAFT）对比，在单轮自监督适配中提升18.1%（Clean）/15.4%（Final）Sintel；在无监督训练中提升6–8% EPE、跨域提升14.2% HD1K、1.8% Middlebury；在监督训练中提升23.1% HD1K、18.8% Middlebury；所有实验均保持轻量化、无显著加速损失。

**⚠️ 局限性**

局限性：在运动模式高度受限的 KITTI 等数据集上提升有限；遮挡与多层运动场景中三角关系受限，需要更精细的可见性建模；对数据统计的依赖可能在极端运动场景下效果不佳。

---

## 324. SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour

**arXiv ID:** 2606.19928 | [PDF](https://arxiv.org/pdf/2606.19928v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 325. ADaPT: Token-Level Decoupling for Efficient Large Reasoning Models

**arXiv ID:** 2606.19919 | [PDF](https://arxiv.org/pdf/2606.19919v1)

**作者:** Tingyun Li `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ADaPT 框架，使大语言模型能够在推理时自适应选择快慢两种推理模式，并通过 token 级奖励实现效率与正确性的解耦。

**💡 创新点**

创新点是引入模式选择 token，将效率奖励与答案正确性分离，允许在推理时通过调整 token 概率实现连续的效率‑准确性 Pareto 控制。

**🔧 技术方法**

采用两阶段训练：SFT 让模型学习快慢模式；GRPO 强化学习结合 token‑level mode reward、CISPO 与双重起始 rollout，以优化模式选择。

**📊 数据集**

使用公开数据集 arm-team（SFT）与 CSQA、GSM8K、MATH 进行强化学习，评测在 CSQA、GSM8K、ARC、MATH500、MMLU‑Pro、Olympiad、AIME24 等多任务。

**📈 对比分析**

与 Base、SFT、SFT+GRPO 以及 TLMRE、ARM、R‑4B 等基线比较，ADaPT 在保持近似准确率的同时将推理长度平均缩短约 30%（如 1540→1031 tokens），并在 Pareto 前沿上实现更优的效率‑准确性平衡。

**⚠️ 局限性**

局限性包括仅采用二元快慢模式、未覆盖更细粒度的推理策略、评测受限于标准数学/常识推理基准，且仅在 3B/7B 规模模型上验证，对更大模型或长文本/交互场景的适用性未知。

---

## 326. SpatialSV: Internalizing Interpretable 3D Spatial Awareness in MLLMs via Task-Oriented Visual Supervision

**arXiv ID:** 2606.19915 | [PDF](https://arxiv.org/pdf/2606.19915v1)

**作者:** Jiayu Tang `[一作]` (Sun Yat-sen University), Chao Gou `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6514db3d-8de6-452c-91b7-acdb31787cc4` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SpatialSV 框架，通过任务导向的视觉监督将 2D 视觉特征提升到 3D 空间，内部化并解释多模态大型语言模型（MLLM）的空间认知。

**💡 创新点**

创新点包括：① 用深度、射线、点云三种显式 3D 表征进行细粒度任务监督，突破传统特征蒸馏的模糊性；② 通过 3D‑lifting 结果提供可视化、可解释的内部空间表示；③ 支持半监督学习，利用无标注视觉数据生成 3D 监督信号。

**🔧 技术方法**

技术细节：视觉编码器-语言解码器结构；多层 2D→3D 投影器；解耦的 DPT 任务模块（深度、射线、点云）；梯度、法线等附加损失；整体目标结合文本自回归、特征蒸馏和 3D 任务损失。

**📊 数据集**

数据集：训练使用 MindCube 10k 样本；评测包含 MindCube‑Tiny、VSI‑Bench；还在 Ego3D‑Bench、Spatial457、ViewSpatial‑Bench、3DSR‑Bench、SP‑Bench、TopViewRS、CVBench、MMBench 上验证。使用 DepthAnything‑v3 生成 3D 监督标签。

**📈 对比分析**

对比实验：在 6 个 MLLM（Qwen2.5‑VL、LLaVA‑OneVision、LLaVA‑NeXT‑Video、InternVL3）上，SpatialSV 在 MindCube‑Tiny 提升 3.4%‑12.7%，在 VSI‑Bench 提升 4.4%‑6.8%；在其他 6 个空间与通用基准上均优于基线与纯文本/蒸馏方案；半监督设置（50% 标注）接近全标注水平，提升 14.2%。

**⚠️ 局限性**

局限性：需依赖 3D VFM 生成监督，可能受其精度限制；对动态或极端遮挡场景的 3D 估计仍易失真；目前的可解释性主要通过可视化 3D 结果，缺乏更深层的模型内部机制解释；训练时仍冻结视觉编码器，限制了进一步的视觉特征优化。

---

## 327. Online Dynamic Batching with Formal Guarantees for LLM Training

**arXiv ID:** 2606.19989 | [PDF](https://arxiv.org/pdf/2606.19989v1)

**作者:** Dian Li `[一作]` (Tencent), Jiahong Yan `[通讯]` (Tencent)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在线动态批处理系统 ODB，能够在数据预处理后实时观测样本长度，并在 DataLoader 端动态组装变长批次，同时保持分布式数据并行（DDP）步骤对齐。

**💡 创新点**

创新点在于：① 将长度可观测点从离线缓存迁移到运行时；② 解决分布式组对齐（DGAP）并提供无死锁、有限终止的 Max‑Based 双向对齐协议；③ 通过 token‑级损失缩放实现不同批次大小的公平梯度。

**🔧 技术方法**

采用了 PyTorch DataLoader 包装、Gloo 同步、基于 token‑预算的批次分组、可选 join‑mode 终止、以及轻量级训练器适配器实现。

**📊 数据集**

在公开数据集 UltraChat、LLaVA、ShareGPT4o（含高 CV 的长尾分布）以及生产级多模态混合数据集 MM‑Mix 上进行评测，并使用六种合成分布验证算法正确性。

**📈 对比分析**

与 Standard、Sorted、Packing、GMT/BMT/HFG 等基线比较，单机 2B/8B 全精度训练时 ODB 取得 1.58–2.51× 的样本吞吐提升，双机 8B 全精度提升 1.71–3.78×，生产 MM‑Mix 达 4.43×；性能提升的同时保持与标准基线相当的验证/基准指标。

**⚠️ 局限性**

局限性包括：需要针对模型、精度、注意力堆栈手动调节 L_max；收益受数据集 CV 和短样本比例影响；对极端异质或大规模世界尺寸的适配尚未充分验证；未涵盖 ZeRO‑3/FSDP 等更高级分布式策略。

---

## 328. The Algorithmic-Human Manager: AI, Apps, and Workers in the Indian Gig Economy

**arXiv ID:** 2606.19975 | [PDF](https://arxiv.org/pdf/2606.19975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 329. Evaluation of Augmented Reality-based Intuitive Interface for Robot-Assisted Transesophageal Echocardiography: A User Study

**arXiv ID:** 2606.19971 | [PDF](https://arxiv.org/pdf/2606.19971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 330. Repository-Level Solidity Code Generation with Large Language Models: From Prompting to Fine-Tuning

**arXiv ID:** 2606.19988 | [PDF](https://arxiv.org/pdf/2606.19988v1)

**作者:** Shi Chen `[一作]`, Rubing Huang `[通讯]` (Macau University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了规模最大、质量最高的 Solidity 代码生成基准数据集 SolidityBench，并提出了专门针对 Solidity 的评估指标 SolidityScore，随后在此基准上对三款 7B 规模的 LLM 进行零-shot、CoT、ICL、RAG、SFT 等多种适配策略的系统实验。

**💡 创新点**

创新点包括：①首次提供5,470条完整仓库级 Solidity 代码与自然语言描述对；②设计了域感知的 SolidityScore，强调安全修饰符、继承关系等关键 Solidity 语义；③对比不同适配范式（尤其是 RAG 与 SFT）的效果，验证 SFT 在该领域的显著优势。

**🔧 技术方法**

技术主要涵盖：大型预训练 LLM（Qwen2.5‑Coder、DeepSeek‑Coder、CodeLlama），链式思考（CoT）与结构化 CoT、示例学习（ICL）、检索增强生成（RAG）、参数高效微调（LoRA SFT）以及自定义评估度量 SolidityScore。

**📊 数据集**

数据集为 SolidityBench，来源于 OpenZeppelin、Synthetix、Etherscan 等权威平台，经过清洗、格式化并为 8:1:1 划分为训练/验证/测试，共 5,470 条完整仓库级 Solidity 代码与自然语言规范。

**📈 对比分析**

通过 BLEU 与 SolidityScore 评估，实验结果表明：零-shot 生成效果差；CoT 与 RAG 可提升 15–30% 的 BLEU/Score；SFT 在所有模型上实现最高提升，Qwen2.5‑Coder 最高得到 BLEU≈36、SolidityScore≈0.65；SFT 还显著提高语法正确率（>60%）但编译通过率仍低于 35%。

**⚠️ 局限性**

局限性：①评估主要基于静态文本/语义匹配，未覆盖运行时安全与 gas 成本；②编译成功率仍受限，无法完全处理跨文件依赖与版本冲突；③仅评估 7B 规模公开模型，缺少对更大或专有模型的验证；④数据集虽规模大但仍可能缺少极端业务逻辑或最新 Solidity 语法变更。

---

## 331. Vision-Reasoning-Guided Occlusion Removal from Light Fields

**arXiv ID:** 2606.19985 | [PDF](https://arxiv.org/pdf/2606.19985v1)

**作者:** Mohamed Youssef `[一作]` (Johannes Kepler University), Oliver Bimber `[通讯]` (Johannes Kepler University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种视角推理驱动的光场遮挡移除框架，将光场积分（LFI）与视觉‑语言模型（VLM）相结合，并通过多样本融合降低假影，适用于结构化和非结构化采样的遮挡严重场景。

**💡 创新点**

创新点在于：① 将物理可见性增强的光场积分与语义推理的视觉‑语言模型融合，形成条件语义先验；② 采用多样本融合策略聚合多次生成结果，显著抑制假影；③ 支持结构化与非结构化采样，并通过上下文感知提示提升VLM鲁棒性。

**🔧 技术方法**

技术手段包括光场积分（基于视角匹配与深度估计的几何聚合）、视觉‑语言模型（Gemini 3.1、Qwen‑Image）用于条件细节重建、部分卷积与光流/深度估计的遮挡预处理、以及多样本平均与上下文提示控制。

**📊 数据集**

使用了合成的 4‑Syn 基准数据集（四个光场场景）以及真实自然植被遮挡场景（RGB 与热成像），对照了多种现有光场遮挡移除方法。

**📈 对比分析**

与 Mask4D、ELFNet、MANet 等方法对比，在 4‑Syn 评测中平均 SSIM 达到 0.883（最高），PSNR 26.90（略低于 Mask4D 29.75），显示在结构保持与视觉质量上均优于现有技术；在真实场景中表现出更好的可视化效果与更少假影。

**⚠️ 局限性**

局限性包括：VLM 生成过程可能产生假影，需通过多样本融合和上下文提示缓解；VLM 推理耗时显著（Gemini 3.1 ~23 s，Qwen‑Image ~9 min）；无法保证完全物理一致的重建；对极度遮挡区的重建仍受观测信息不足限制。

---

## 332. CrossFlow: One-Step Generation Across Latent and Pixel Spaces

**arXiv ID:** 2606.19970 | [PDF](https://arxiv.org/pdf/2606.19970v1)

**作者:** Xiyuan Wang `[一作]` (Peking University), Muhan Zhang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了CrossFlow，一种跨空间流匹配框架，直接将噪声潜变量映射为像素级图像，既实现了单步生成，又可替代Latent Diffusion中的VAE解码器。

**💡 创新点**

创新点在于将潜空间的噪声输入与像素空间的输出分离，构造无速度的单步目标，允许模型在训练时直接使用像素级感知和对抗损失，避免潜变量与解码器之间的分布偏移。

**🔧 技术方法**

核心技术包括跨空间流匹配目标设计、使用Vision Transformer生成器、固定预训练的VAE编码器、感知损失与GAN对抗损失、以及Jacobian-Vector Product实现时间导数计算。

**📊 数据集**

在ImageNet‑1k（256×256）分类条件下进行评估。

**📈 对比分析**

与多步Latent Diffusion和其他单步方法对比，CrossFlow‑XL在单次函数评估下获得1.62 FID，优于iMF‑XL/2（1.72 FID）且与DiT‑XL/2、SiT‑XL/2等多步模型（约2.0–2.3 FID）相当。

**⚠️ 局限性**

局限性包括：需要预训练的潜变量编码器，无法一次性训练编码器与生成器；对多步采样场景适配性有限；在极端噪声或潜空间分布偏移时仍可能出现解码失真。

---

## 333. The Bi-Channel Networking Paradigm for Database Systems in the Cloud

**arXiv ID:** 2606.19969 | [PDF](https://arxiv.org/pdf/2606.19969v1)

**作者:** Georg Kreuzmayr `[一作]` (TigerBeetle), Viktor Leis `[通讯]` (Technische Universität München)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了“Bi-Channel”网络通信范式，用于分离数据库系统的控制路径和数据路径。

**💡 创新点**

创新点在于将高性能、无序的用户空间 UDP 数据通道与可靠的内核 TCP 控制通道结合，通过协同工作提升吞吐量与延迟，而无需在用户空间重实现完整 TCP 语义。

**🔧 技术方法**

技术包括：DPDK 基于用户空间 UDP 的高速数据通道；内核 TCP 控制通道；端口选择、队列定位、信用控制等调优技术；在 AWS 云环境下的多 vNIC 配置与流量调度。

**📊 数据集**

数据集：分布式 Join 采用 256 GiB、64 GiB、1 GB–2 TB 规模的关系表；键值存储使用 64 字节更新包；所有实验均在 AWS c7gn.16xlarge 实例（200 Gbps NIC）上进行。

**📈 对比分析**

比较方法：将 Bi-Channel 与纯内核 TCP（通过 io_uring）对比。Shuffle 端口单线程下 175 M 元组/秒（Bi-Channel）vs <50 M（TCP），四线程可达 240 M；KV‑Store 端到端 p95 延迟在 250 µs 时实现 3.9 M 更新/秒，约 2.6× 以上吞吐。整体显示显著提升吞吐与可预测延迟。

**⚠️ 局限性**

局限性：需在支持多 vNIC 与 DPDK 的云平台实现；对流量限速、队列缓冲等参数的调优依赖经验；UDP 数据通道在极低丢包率环境下表现良好，丢包仍需通过控制通道恢复；实现复杂度高，需在 DBMS 内部完成协同与信用管理。

---

## 334. Low-Energy Reduced RISC-V Instruction Subset Processor for Tsetlin Machine Inference at the Edge

**arXiv ID:** 2606.19964 | [PDF](https://arxiv.org/pdf/2606.19964v1)

**作者:** Chanda Gupta `[一作]` (Indian Institute of Technology Roorkee), Sudip Roy `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对 Tsetlin Machine 推理的低能耗可编程 RISC‑V 微处理器及其设计流程

**💡 创新点**

通过指令剖析实现 ISA 减少、数据路径与控制路径精简，并将 TM 推理算法优化为稀疏字面量索引形式，从而在保持可编程性的同时显著降低能耗和执行时间

**🔧 技术方法**

使用 RISC‑V 开源架构、指令剖析、数据路径与控制路径简化、基于 BNN 的基准对比、Xilinx Vivado 合成与功耗测量

**📊 数据集**

CIFAR‑2、Statlog、Gesture、Gas、EMG、FMNIST 等六个常见分类数据集

**📈 对比分析**

与 BNN 在相同硬件、相同数据集、相同准确率下比较，TM 在 98% 以下的执行时间、平均 29.7× 的能耗降低（最大 99.3%），且准确率相当或更优

**⚠️ 局限性**

仅针对 TM 推理的定制化，缺乏对更通用机器学习任务的适配；未考虑模型更新与在线学习的软硬件协同；对深度网络的支持有限

---

## 335. Confidence Calibration for Multimodal LLMs: An Empirical Study through Medical VQA

**arXiv ID:** 2606.19950 | [PDF](https://arxiv.org/pdf/2606.19950v1)

**作者:** Yuetian Du `[一作]` (Zhejiang University), Qiang Zhu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对医疗视觉问答任务中的多模态大型语言模型（MLLMs）进行置信度校准研究。

**💡 创新点**

创新点是提出基于多策略融合的讯问系统（MS‑FBI）与辅助专家LLM评估相结合的新校准方法。

**🔧 技术方法**

技术上采用惩罚机制、挑战与解释双重验证的多策略讯问，并利用专家LLM（如llama3‑instruct‑8B）重新评估置信度。

**📊 数据集**

使用Med‑VQA、VQA‑RAD和SLAKE三个公开医学VQA数据集进行评估。

**📈 对比分析**

与Vanilla、Punish、Top‑K等基线方法比较，平均ECE降低约40%，AUROC提升约6%，在LLaVA、Molmo等模型上均取得最优性能。

**⚠️ 局限性**

局限性在于对通用LLM的泛化验证不足，方法对计算成本有一定开销，且在不同预训练模型上策略效果仍需进一步微调。

---

## 336. Compositionality Emerges in a Narrow Depth-Connectivity Regime: Architecture Constraints and Solution Manifolds

**arXiv ID:** 2606.19941 | [PDF](https://arxiv.org/pdf/2606.19941v1)

**作者:** Dat H. Do `[一作]` (National University of Singapore), Dianbo Liu `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在梯度下降训练中，网络何时能自然形成可组合的内部结构，并提出一种结构感知剪枝（Similarity-based Pruning, SP）与基于图像复杂度的深度预测方法，能够在仅使用MLP的情况下恢复并强化模型的单义性与可组合性。它还构建了EMC^2-Bench评估套件，用来量化参数对语义不变性的影响。

**💡 创新点**

①发现可组合性只在深度与稀疏度交互的“甜点”区域出现；②提出的SP剪枝规则能显著提升可组合性评分；③用图像PNG压缩率作为快速深度预测器；④将经验结果与基于可组合稀疏性、体积比例与特征干扰界限的理论框架结合，解释为什么只有在特定架构下梯度下降能收敛到可组合解。

**🔧 技术方法**

梯度下降（Adam、Muon等）、迭代剪枝–再训练、余弦相似度剪枝、VLM评估（用于判断语义不变性）、基于PNG压缩率的深度回归、理论分析（可组合稀疏性、体积比例、Welch界限）

**📊 数据集**

Picbreeder的手工合成图像（如skull、butterfly、apple）作为基准；额外的未见图像作为外域测试；EMC^2-Bench提供的内部特征可变性评估；使用VLM（如ChatGPT/Stable Diffusion）对生成图像进行语义判断。

**📈 对比分析**

与Lottery Ticket、Wanda、LLM‑Pruner等传统剪枝方法对比，SP在所有基准上均取得非零可组合性得分（最高0.63–0.86），而其它方法得分为0；在外域目标上，SP+深度预测同样能产生单义特征、语义不变性并在预测深度附近出现可组合性峰值；整体性能表明SP在保持参数预算不变的情况下显著提升可组合性。

**⚠️ 局限性**

仅在单一物体或极简图像上验证，无法直接推广到多物体或复杂场景；只针对MLP，未考察CNN/Transformer等更普遍架构；对VLM评估的依赖可能导致主观性；深度预测器虽然有效但仍存在较大方差；可组合性甜点对目标高度敏感，需进一步探索更稳健的架构与优化策略。

---

## 337. MemGUI-Agent: An End-to-End Long-Horizon Mobile GUI Agent with Proactive Context Management

**arXiv ID:** 2606.19926 | [PDF](https://arxiv.org/pdf/2606.19926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 338. The Tao of Agency: Autotelic AI, Embedded Agency and Dissolution of the Self

**arXiv ID:** 2606.19924 | [PDF](https://arxiv.org/pdf/2606.19924v1)

**作者:** Aritra Sarkar `[一作]` `[通讯]`, Aritra Sarkar

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个统一的自我目标生成（autotelic）AI框架，整合了内在动机、资源优先级、因果干预、稳态自我保守、嵌入式智能、马尔可夫屏障以及量子与哲学视角，并给出了基于大语言模型的实现示例。

**💡 创新点**

创新点在于将自我界定的非唯一性与目标生成的内在驱动相结合，形成“自我-目标相对化”机制，并将传统的agent–environment二元划分提升到可量子化的动态切割；同时提出了“自我是操作性必要的幻象”这一概念。

**🔧 技术方法**

采用了算法信息论（Kolmogorov复杂度、Levin复杂度）、因果干预理论、可变能量自我保守（homeostasis）与自由能原理、Markov屏障、量子MDP、以及大语言模型的推理与决策模块。

**📊 数据集**

未使用传统数据集，主要通过理论推导、模拟实验和对比案例（如AIXI、KSA、QKSA、LLM代理）进行验证。

**📈 对比分析**

方法通过理论对比与概念性实验展示，未给出数值性能指标，但强调在目标多样性、可持续性与自我一致性方面的改进，预期在开放式环境中能更好地生成与实现内部目标。

**⚠️ 局限性**

局限性包括自我划分的非唯一性导致的目标分配不确定、对量子实现的实际可行性缺乏实验验证、对大规模实证评估不足，以及在实际复杂环境中实现自我与资源预算平衡的技术挑战。

---

## 339. A Novel FeFET Differential Bit-Cell With Hybrid Volatile and Non-Volatile Memory Modes

**arXiv ID:** 2606.19918 | [PDF](https://arxiv.org/pdf/2606.19918v1)

**作者:** Jianze Wang `[一作]` (National University of Singapore), Xuanyao Fong `[通讯]` (National University of Singapore)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种4T差分FeFET比特单元，可在易挥发或非易挥发模式下工作；

**💡 创新点**

创新点在于仅使用四个晶体管实现跨耦合FeFET差分存储，省略了传统nvSRAM所需的备份-恢复（B&R）操作，同时通过调整写条件即可在挥发与非挥发模式之间切换，写操作利用相同的写电压对BL/SL和BLB/SLB驱动，显著降低写能量；

**🔧 技术方法**

采用HZO基FeFET（SOI结构）与交叉耦合2T增益单元，结合FeCap+BSIM-IMG模型进行仿真，使用GLOBALFOUNDRIES 22 nm PDK完成布局与寄生提取；

**📊 数据集**

无公开数据集，仿真基于实验测得的FeFET I‑V 曲线进行模型校准，随后进行写/读能量、感应延迟等参数的仿真；

**📈 对比分析**

通过与传统6T SRAM、4T-R、7T2R、8T2R、1T1FeFET等设计在晶体管数、存储电压、功率、写时、感应方案以及是否需要B&R等维度进行对比；该单元存储功率仅0.13 µW，写时2 ns，且不需要B&R，表现优于或相当于现有设计；

**⚠️ 局限性**

局限在于写操作需要负写电压，外围电路复杂；对写电压和脉冲宽度仍需保持在一定范围（≥0.4 V）以保证可靠编程；未对温度漂移、长期可靠性与耐久性进行系统验证；

---

## 340. Sensorimotor World Models: Perception for Action via Inverse Dynamics

**arXiv ID:** 2606.20104 | [PDF](https://arxiv.org/pdf/2606.20104v1)

**作者:** Petr Ivashkov `[一作]` (Max Planck Institute for Intelligent Systems), Bernhard Schölkopf `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在离线无奖励像素-动作数据上训练一个隐式世界模型SMWM，加入单步逆动力学正则化以防止表示崩塌并让编码器聚焦可控特征；

**💡 创新点**

创新点是将单步逆动力学正则化作为唯一的抗崩塌机制，同时通过它引导模型学习与动作相关的可控低维表示；

**🔧 技术方法**

使用JEPA框架，联合前向动态模型和逆动力学头，通过前向损失与逆损失共同优化编码器；

**📊 数据集**

利用多种离线模拟数据集，包括TwoRoom、Reacher、Push‑T、OGBench‑Cube等2D/3D控制环境的像素序列与连续动作；

**📈 对比分析**

与SIGReg、Forward‑only、随机动作等基线比较，SMWM在所有四个任务中与SIGReg相当或更优，尤其在3D OGBench‑Cube中实现显著性能提升；

**⚠️ 局限性**

局限性包括假设动作可由连续观测恢复、单帧无法捕捉速度等动态信息、可能丢弃与动作无关但后续任务重要的干扰信息，以及离线数据覆盖不足导致的长程规划误差。

---

## 341. Quadratic Forms for Measuring Geometric Trees in 3-dimensional Space

**arXiv ID:** 2606.20096 | [PDF](https://arxiv.org/pdf/2606.20096v1)

**作者:** Yossi Bokor Bleile `[一作]` (Institute of Science and Technology Austria), Shota Uka `[通讯]` (Technical University of Vienna)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一种基于二次型的三维树状结构形态量化方法，并提出hexplot可视化模型来展示形态统计；同时设计了用于树分解的最优路径算法。

**💡 创新点**

创新点在于将正定半矩阵的二次型用于测量树状结构的方向性分布，将其映射到六边形空间（hexplot）并引入Fisher度量实现无尺度、无旋转不变的形态描述；同时提供了可视化的箱型图和曲线演化分析。

**🔧 技术方法**

主要技术包括：二次型与椭球相联系、特征值分解、极坐标化、Antonelli映射、Fisher信息度量、Minkowski和、可视化算法（hexagonal box plot、曲线绘制）以及基于二次型的贪婪路径分解算法。

**📊 数据集**

使用了来自SWC文件的三条上位树突（apical dendrite）数据，作为实验样本；此外在讨论中提到可对更多神经元树突群体进行实验。

**📈 对比分析**

通过将树突映射到hexplot后，用Fisher距离比较不同样本的形态分布；在图示中展示了两个树突族群的分布差异，表现出高聚类、圆形、细长等形态特征；虽然未给出定量性能指标，但可视化显示分组效果明显。

**⚠️ 局限性**

局限性包括：仅适用于三维正定半二次型，无法捕捉方向向量信息；对极端退化椭球（如平面或线性结构）处理不完善；对更高维或非正定矩阵的推广尚未实现；并且在大规模数据集上的计算复杂度与可视化清晰度需进一步评估。

---

## 342. Holo-World: Unified Camera, Object and Weather Control for Video World Model

**arXiv ID:** 2606.20083 | [PDF](https://arxiv.org/pdf/2606.20083v1)

**作者:** Xiangchen Yin `[一作]` (University of Science and Technology of China), Xiaoyan Sun `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种单帧源到状态的视频世界模型 Holo-World，能够联合控制摄像机运动、物体动态和天气状态；

**💡 创新点**

通过构建统一的控制接口与数据集（HoloStateData）实现了天气编辑与世界保持的解耦，提出 Unified Scene Adapter 与 Scene-Weather Decomposed CFG 两大技术；

**🔧 技术方法**

使用冻结的 Wan 视频扩散网络，结合 World Adapter 与 State Adapter 进行参数分离，并在推理时采用分解式条件引导；

**📊 数据集**

利用 HoloStateData，包括真实视频、配对的模拟天气视频以及 V2V 生成的天气转化视频，提供摄像机、物体与天气的联合监督；

**📈 对比分析**

在 150 样本的 Real 与 Weather 子集上与 Uni3C、GEN3C、VerseCrafter、NeoVerse、Cosmos-Transfer2.5、Wan2.7-Edit 等基线比较，Holo-World 在世界保持、摄像机/物体控制误差、天气对齐率与 VLM 评估等指标均优于对比方法（例如 Weather Alignment 86%/VLM 68.51，Human Preference 83%/62%）；

**⚠️ 局限性**

局限在于仅关注可控视频生成而非完整物理仿真，且对极端天气变化与复杂动态场景的泛化仍有待提升。

---

## 343. Variable-Length Tokenization via Learnable Global Merging for Diffusion Transformers

**arXiv ID:** 2606.20076 | [PDF](https://arxiv.org/pdf/2606.20076v1)

**作者:** Dong Hoon Lee `[一作]` (KAIST), Seunghoon Hong `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于可学习全局合并的可变长度词典，能够在不依赖图像的前提下实现长度调制并保持表示一致性；同时训练了一个能够在不同长度下工作的扩散变压器；

**💡 创新点**

创新点在于：①通过合并(token merging)而非截断实现长度调制；②设计可学习全局合并策略，使合并模式与数据无关且可在生成时使用；③引入比例注意力和合并位置嵌入，实现对合并后token的完整语义处理；

**🔧 技术方法**

使用了视觉Transformer编码器/解码器、SoftVQ框架、聚合层聚类、Straight‑Through技巧、对齐损失、比例注意力、学习位置嵌入、LoRA微调、速度预测损失的LightningDiT；

**📊 数据集**

在ImageNet‑1K 256×256的类条件生成任务上进行训练和评估；

**📈 对比分析**

与Semanticist、FlexTok、SoftVQ、MAETok等可变长度和高压缩率固定长度词典对比；在gFID‑FLOPs/Throughput曲线上，提出的方法在相同算力下实现更低的gFID，且相对于单独训练的模型差距<0.3；LoRA后接近单独训练模型；

**⚠️ 局限性**

仍存在：与2D词典相比生成质量差距；全局合并虽可学习但不一定最优；对极低token数时重建质量略有下降；

---

## 344. What Makes Effective Supervision in Latent Chain-of-Thought: An Information-Theoretic Analysis

**arXiv ID:** 2606.20075 | [PDF](https://arxiv.org/pdf/2606.20075v1)

**作者:** Xinghao Chen `[一作]` (Eastern Institute of Technology), Xiaoyu Shen `[通讯]` (Eastern Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于信息理论分析了隐式链式思维（Latent CoT）的监督瓶颈，提出将过程监督拆解为轨迹监督与空间监督，并通过统一隐式探测器（ULP）量化互信息，从而实现更有效的隐式推理；

**💡 创新点**

创新点在于：①阐明了“双重崩塌”——梯度衰减与语义漂移；②将监督拆解为轨迹监督与空间监督，证明轨迹监督可缓解梯度衰减；③引入生成式重构而非几何压缩，保持信息容量；④设计ULP以变分方式估计互信息，揭示信息-性能绑定；

**🔧 技术方法**

信息理论方法（互信息估计、变分重构），Transformer（GPT‑2）模型，轨迹监督与空间监督损失（生成式重构、几何压缩），ULP变分探测器；

**📊 数据集**

GSM8k扩增集（GSM8k‑Aug）为主要推理数据集；

**📈 对比分析**

与传统Explicit CoT、Outcome‑Supervised Latent CoT以及各空间监督变体对比，轨迹监督+生成式重构实现了最高的推理准确率（约+3%–5%），并且显著优于仅使用几何压缩或无监督的Baseline；

**⚠️ 局限性**

局限包括：仅在GPT‑2规模下验证，尚未在更大模型或多领域任务中测试；过程监督依赖人工注释，难以扩展；ULP估计的互信息为下界，可能低估真实信息量；

---

## 345. Autonomous Event-Driven Multi-Agent Orchestration for Enterprise AI at Scale

**arXiv ID:** 2606.20058 | [PDF](https://arxiv.org/pdf/2606.20058v1)

**作者:** Harsh Rao Dhanyamraju `[一作]` (SAP SE), Aaron Lee `[通讯]` (SAP SE)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了两种多智能体协调架构（DAG Plan & Execute 与 ReAct）在企业规模的持续事件监控与处理中的表现，并提出了任务管理器（Task Manager）用于事件优先级推断、相关事件合并和抢占。

**💡 创新点**

①将连续事件驱动的企业场景拆分为 Persona、Department、Enterprise 三个规模级别；②设计了统一的任务管理器与优先级调度机制；③用 208 个基于真实企业流程的场景对比实验，揭示规模是导致性能下降的主因而非任务复杂度。

**🔧 技术方法**

使用了两种框架：ReAct（单模型连续推理+行动）与 DAG Plan & Execute（规划-执行-重规划三阶段）；实现了双层智能体发现、A2A 协议、基于 LLM 的任务推断与合并；部署了 PydanticAI、AutoGen 等现有工具栈。

**📊 数据集**

构建了 208 个由人工审核并使用 LLM 生成的企业用例数据集，覆盖 Persona (<10 agents)、Department (20–80 agents) 与 Enterprise (200 agents) 三个规模；包含 95 Persona、63 Department、50 Enterprise 级场景；每个场景都给出用户提示、预期智能体序列、并列/混合执行模式及答案片段。

**📈 对比分析**

通过模拟 200 个企业级 Mock 智能体进行评估，测量任务正确性、智能体调用精确度与召回率、队列等待时间以及任务完成率。结果显示：在 Persona 级别两种架构均能取得 74–98% 的正确率；随着规模增大，DAG 的性能在 200 代理时显著下降（尤其是简单任务），而 ReAct 在企业规模保持更高的正确率（简单任务 32–83%），并在失败场景中更具鲁棒性。Task Manager 在高优先级事件队列等待时间上可节省 14–75% 的时间，在 Enterprise 级别通过合并相关事件将正确率提升 20+个百分点。

**⚠️ 局限性**

① Mock 智能体响应速度快，未能反映真实多步骤工具调用的延迟，导致抢占窗口缩小；② 实验仅使用单一调度器，未覆盖多工器分布式调度与优先级公平性问题；③ 关键参数（如检查点量子、重规划阈值）未做调优，结果可能因工作负载不同而差异较大；④ 未包含持久化记忆与长期学习，可能低估实际部署性能。

---

## 346. MirrorDuo: Reflection-Consistent Visuomotor Learning from Mirrored Demonstration Pairs

**arXiv ID:** 2606.20048 | [PDF](https://arxiv.org/pdf/2606.20048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 347. PU-UNet: Stable Multiplicative Interactions for Medical Image Segmentation

**arXiv ID:** 2606.20035 | [PDF](https://arxiv.org/pdf/2606.20035v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 348. ReA-OVCD: Reliability-Aware Open-Vocabulary Change Detection via Semantic and Spatial Refinement

**arXiv ID:** 2606.20032 | [PDF](https://arxiv.org/pdf/2606.20032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 349. Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning

**arXiv ID:** 2606.20002 | [PDF](https://arxiv.org/pdf/2606.20002v1)

**作者:** Yanxi Chen `[一作]` (Alibaba Group), Jingren Zhou `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于端到端强化学习的框架，用于训练大型语言模型在长期生命周期代理中连续解决任务、主动探索环境并自我更新上下文，从而实现“连接点”meta‑capability。

**💡 创新点**

①提出了专门针对meta‑capability的RL后训练框架，①引入细粒度信用分配的GRPO风格算法；②设计专属的跨域任务/环境以激励上下文更新；③展示了在不同域和Ralph‑loop设置下的跨域泛化能力。

**🔧 技术方法**

使用了LLM‑RL技术（agentscope‑style），GRPO（无 critic）改进，动态规划原则的信用分配和优势计算；通过系统提示+“hint”文本实现上下文更新；实验环境包括自定义grid、合成元素、终端模拟器等。

**📊 数据集**

自定义任务环境：1）grid‑游戏（隐藏动作映射）；2）元素合成游戏（随机配方）；3）终端模拟器（Linux/Mac/Windows 命令行）。训练使用任务序列长度4，评估在更难实例、跨域及Ralph‑loop上进行。

**📈 对比分析**

通过训练奖励曲线和评估曲线对比。结果显示：同域第4个任务的成功率从28%提升至76%；在跨域和Ralph‑loop中也观察到奖励提升，说明meta‑capability在不同环境下具有一定泛化；但跨域性能提升不如同域明显。

**⚠️ 局限性**

算法仍包含启发式加权，训练不稳定；上下文更新仅使用单一“hint”文本，未尝试持久化记忆或技能库；任务环境相对简单，未覆盖更复杂或非平稳环境；跨域泛化在某些设置下表现不佳；未与现有LLM后训练管道完整集成。

---

## 350. Tri-Info: Generalizable, Interpretable Failure Prediction for VLA Models via Information Theory

**arXiv ID:** 2606.19998 | [PDF](https://arxiv.org/pdf/2606.19998v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 351. VIMPO: Value-Implicit Policy Optimization for LLMs

**arXiv ID:** 2606.20008 | [PDF](https://arxiv.org/pdf/2606.20008v1)

**作者:** Zhewei Kang `[一作]` (University of California Berkeley), Xuandong Zhao `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种无批评者的 RLVR 方法 VIMPO，利用 KL 正则化最优性条件从策略-参考模型的对数比中直接推导价值和优势，使用终点边界条件训练隐式价值函数，随后通过 PPO 风格的优势更新优化策略。

**💡 创新点**

创新点在于通过 KL 正则化的确定性 MDP 形式推导出闭式价值递推式，并将其作为无批评者的平方误差损失；同时提供了完全无批评者的 TD 量化优势，可直接用于 PPO 更新，兼具简洁性与细粒度信用分配。

**🔧 技术方法**

使用了 KL 正则化策略优化、确定性转移 MDP 设定、策略-参考模型对数比、终点 0 值条件、PPO 剪切损失、Gumbel 或 softmax 归一化、梯度停止等技术。

**📊 数据集**

实验基于 Guru 版数学数据集，评估在 MATH-500、AIME 2024、AIME 2025 与 OlympiadBench 四个数学推理基准上，并在 Qwen3‑4B‑Base 之上进行训练。

**📈 对比分析**

与传统无批评者 GRPO（包括 naive 和 token‑level 版本）相比，VIMPO 在所有基准上取得更高验证精度，尤其在 AIME 2025 上提升约 3.2%，在噪声奖励场景下也表现出更稳健的性能。

**⚠️ 局限性**

局限性包括：固定 β 与冻结参考模型可能限制后期性能；需要完整分布的 KL 计算成本较高；实验仅在 4B 模型和数学推理任务上进行，未验证在更大模型或其它可验证任务中的迁移效果。

---

## 352. Geometry-Preserving in 3D Gaussian Splatting for LiDAR-Camera Extrinsic Calibration

**arXiv ID:** 2606.20103 | [PDF](https://arxiv.org/pdf/2606.20103v1)

**作者:** Kyoleen Kwak `[一作]` (Kyung Hee University), Hyoseok Hwang `[通讯]` (Kyung Hee University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 GeoP-Calib 框架，解决 3D Gaussian Splatting 在 LiDAR‑Camera 目标无标定中出现的几何衰减问题。

**💡 创新点**

通过 Dense Depth Anchoring（多视角 LiDAR 观测聚合）和 Gradient Decoupling（阻止光度梯度更新几何）双管齐下，保持代理几何的度量精度。

**🔧 技术方法**

使用 3D Gaussian Splatting 作为场景代理，聚合多视角 LiDAR 深度作为 Dense Depth Anchoring，采用体素软掩模（VSM）处理遮挡，采用梯度阻断（Gradient Decoupling）避免光度误差导致几何漂移，以及传统的外参优化与深度监督。

**📊 数据集**

在 KITTI‑360 与 KITTI odometry 两个公开行驶数据集上进行实验。

**📈 对比分析**

与 GST、CLAIM、RobustCalib、HiGS‑Calib 等基线进行比较，平均旋转误差降至 1.67°/0.28 m，翻译误差显著提升至 0.063 m，整体性能均优于基线。

**⚠️ 局限性**

旋转精度提升有限；梯度阻断可能限制光度信息的利用；Dense Depth Anchoring 需要额外 50 s 计算时间，增加了运行开销。

---

## 353. MakeupMirror: Improving Facial Attribute Preservation in Diffusion Models for Makeup Transfer

**arXiv ID:** 2606.20094 | [PDF](https://arxiv.org/pdf/2606.20094v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 354. WeGenBench: A Multidimensional Diagnostic Benchmark towards Text-to-Image Model Optimization

**arXiv ID:** 2606.20100 | [PDF](https://arxiv.org/pdf/2606.20100v1)

**作者:** Qian Liang `[一作]` (University of Electronic Science and Technology of China), Chen Li `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WeGenBench中文-英文双语大规模文本-图像生成评测基准，包含4000条精心设计的测试提示，并为每条提示分配多维度标签；

**💡 创新点**

创新点在于：①多维度标签与场景交叉分类，精准定位模型弱点；②采用Vision‑Language模型（VLM）和OCR的混合评估，提出可解释的多维度指标（QA、COT、Anchor‑Match、文本渲染五个指标）；③设计层级化Anchor‑Match裁定流程与结构上限机制，提升美学评测的可靠性；

**🔧 技术方法**

技术手段包括：VLM问答与链式推理、CLIP‑style对比检索、OCR+VLM文本识别、Hungarian匹配、层级化Anchor图库与位置去偏；

**📊 数据集**

数据集为WeGenBench自构，General类2000条（中英各1000条）与Text类2000条（中英各1000条），覆盖16个宏观场景、10个文本渲染场景，包含多标签和难度等级；

**📈 对比分析**

通过与现有基准（PartiPrompts、TIFA、GenEval等）对比，展示WeGenBench在多维度诊断上的优势；在实验中对22种SOTA模型进行评测，结果显示：VLM‑based QA/COT在语义对齐上与人类偏好高度一致（Spearman ρ≥0.91），Anchor‑Match在美学评分上与人工排名差距仅2.06位，文本渲染指标普遍低于人类准确率，凸显该领域瓶颈；

**⚠️ 局限性**

局限性在于：①评测依赖VLM与OCR模型的准确性，若这些模型存在误识或偏见，评估结果会受影响；②中文与英文对齐仍有差距，表明跨语言编码仍不完善；③基准侧重单模态输出，尚未覆盖交互式或动态场景的生成能力；

---

## 355. Site-Specific MIMO Channel Generation via Diffusion and Flow Matching: Fidelity, Efficiency, and Downstream Utility

**arXiv ID:** 2606.20098 | [PDF](https://arxiv.org/pdf/2606.20098v1)

**作者:** Sina Beyraghi `[一作]` (Telefónica Scientific Research), Giovanni Geraci `[通讯]` (Nokia)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

研究了利用条件扩散模型和条件流匹配模型在有限场站数据下生成高质量、站点特定的多输入多输出（MIMO）通道，并将其用于下游任务如CSI压缩和波束对准

**💡 创新点**

首次将扩散和流匹配两种连续生成框架统一比较，证明流匹配在保持站点空间一致性的同时具有更低的采样延迟；同时提出针对位置条件的高效网络结构和多尺度编码器-解码器

**🔧 技术方法**

条件扩散隐式模型（cDDIM）、条件流匹配模型（cFMM）、Beamspace Fourier编码、位置与时间嵌入的特征调制、CRNet CSI压缩网络以及波束对齐网络

**📊 数据集**

采用Sionna射线追踪在伦敦市区模拟生成的3.5 GHz LoS、LoS+NLoS以及28 GHz LoS三种情景下的MIMO通道数据；训练样本量从200到10 k不等

**📈 对比分析**

通过统计相似度（主波束索引误差、余弦相似度、有效秩Wasserstein距离）、采样效率（采样步数与推理时间）、以及下游任务指标（CSI压缩NMSE、波束对齐SNR）进行比较。结果显示cFMM在约10倍的采样速度下与cDDIM相近；在200样本的低数据场景下，生成通道可将CSI压缩NMSE提升至约-8.8 dB，波束对齐SNR几乎逼近全10 k真实数据的上限

**⚠️ 局限性**

对真实测量场景的适用性有限；模型对超大规模MIMO阵列的可扩展性尚未验证；流匹配虽然速度快但在极端多径环境下有效秩匹配略逊于cDDIM

---

## 356. The Hidden Evolution of Disguised Visual Context inside the VLM

**arXiv ID:** 2606.20077 | [PDF](https://arxiv.org/pdf/2606.20077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 357. PaAno+: Multiscale Encoding and Cross-Variable Attention for Time Series Anomaly Detection

**arXiv ID:** 2606.20055 | [PDF](https://arxiv.org/pdf/2606.20055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 358. PACMS: Submodular Context Selection as a Pluggable Engine for LLM Agents

**arXiv ID:** 2606.20047 | [PDF](https://arxiv.org/pdf/2606.20047v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 359. FUSE: Frequency-domain Unification and Spectral Energy Alignment for Multi-modal Object Re-Identification

**arXiv ID:** 2606.20044 | [PDF](https://arxiv.org/pdf/2606.20044v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 360. When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents

**arXiv ID:** 2606.20023 | [PDF](https://arxiv.org/pdf/2606.20023v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 361. Reward as An Agent for Embodied World Models

**arXiv ID:** 2606.19990 | [PDF](https://arxiv.org/pdf/2606.19990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 362. Segment-Level Mandarin Chinese Speech-Based Cognitive Impairment Detection via an Autoencoder with Contrastive Learning

**arXiv ID:** 2606.19996 | [PDF](https://arxiv.org/pdf/2606.19996v1)

**作者:** Yongqi Shao `[一作]` (Shanghai Jiao Tong University), Tao Fang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于语音分段的认知障碍检测框架。

**💡 创新点**

将GRU自编码器与监督对比学习结合，并使用离线+在线谱图增强，实现低资源环境下的鲁棒特征学习。

**🔧 技术方法**

使用GRU自编码器、监督对比学习、SpecAugment谱图增强、Log-Mel分段特征、MLP分类器，以及PyTorch实现。

**📊 数据集**

四个中文数据集：Ye（AD vs CN）、Chou（MCI vs CN）、TAUKADIAL（MCI vs CN）和NCMMSC2021（AD/MCI/CN）。

**📈 对比分析**

通过10折嵌套交叉验证与现有方法比较，在NCMMSC2021上实现96.8%准确率，整体准确率均超过96%，显著优于基线与先前研究。

**⚠️ 局限性**

仍存在数据不平衡、MCI类别混淆等局限，且未在跨语言或多模态场景下验证。

---

## 363. Artificial Intelligence as Game Changer in Cybersecurity: What We Learned in 2025-2026, and how this is relevant for Africa

**arXiv ID:** 2606.20102 | [PDF](https://arxiv.org/pdf/2606.20102v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 364. Multi-Head Attention-Based Feature Extractor Integration with Soft Actor-Critic for Porosity Prediction and Process Parameter Optimization in Additive Manufacturing

**arXiv ID:** 2606.20087 | [PDF](https://arxiv.org/pdf/2606.20087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 365. Exploring the potential of AlphaEarth and TESSERA embeddings for Fine-scale Local Climate Zone Mapping: A case study across five cities in Switzerland

**arXiv ID:** 2606.20034 | [PDF](https://arxiv.org/pdf/2606.20034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 366. HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization

**arXiv ID:** 2606.20097 | [PDF](https://arxiv.org/pdf/2606.20097v1)

**作者:** Zhentao Tan `[一作]` (Alibaba Group), Jieping Ye `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种在注意力头级别混合全注意力（FA）与线性注意力（LA）的混合架构 HydraHead，旨在兼顾长上下文处理与推理精度。

**💡 创新点**

创新点包括：① 通过可解释性分析（激活补丁/路径补丁）识别关键注意力头并仅保留其 FA 计算；② 引入尺度归一化融合模块解决 FA 与 LA 输出分布差异；③ 采用三阶段迁移学习（参数继承、全局蒸馏、长上下文微调）实现高效训练。

**🔧 技术方法**

技术细节：机制可解释性技术（激活补丁、路径补丁），分组查询注意力（GQA），Gated DeltaNet 线性注意力，RMSNorm 归一化，双分支 FA/LA 并行计算，蒸馏与对齐损失。

**📊 数据集**

使用的数据集包括 FineWeb‑Edu（高质量网页文本）进行微调，RULER Needle‑in‑a‑Haystack（长上下文检索）和一系列通用推理基准（MMLU、BBH、MBPP、GSM8k）用于评估。

**📈 对比分析**

与层级、词级及其他头级混合基线以及现有大模型（如 Qwen3、Qwen3.5、Gemma‑3n‑E2B 等）在相同训练设置下对比；HydraHead 在长上下文检索上相较基线提升 69%+，在 512K 上下文接近 Qwen3.5 的表现，同时保持与基线相当或更好的通用推理能力。

**⚠️ 局限性**

局限性：需要先行的可解释性分析和精细的头级选择，极端稀疏比例（如 9:1）仍会显著损失推理性能；训练流程相对复杂，未在更大模型规模或更丰富注意力变体上进行充分验证。

---

## 367. QG-MIL: A Gated Transformer Aggregator for Domain-Agnostic Multiple Instance Learning in Medical Imaging

**arXiv ID:** 2606.20027 | [PDF](https://arxiv.org/pdf/2606.20027v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 368. Source-Grounded Data Generation for Text-to-JSON Learning

**arXiv ID:** 2606.20072 | [PDF](https://arxiv.org/pdf/2606.20072v1)

**作者:** Sunghee Ahn `[一作]` (Seoul National University), Youngjae Yu `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于电子表格的文本到JSON的源对齐数据生成流水线（STAGE）与对应评测基准。

**💡 创新点**

创新点在于将电子表格作为唯一可信的知识来源，生成报告与JSON时通过代码验证保证每个JSON值可从表格提取，解决了传统LLM生成数据缺乏可验证性与可扩展性的问题。

**🔧 技术方法**

利用LLM（如Qwen3-4B、Qwen2.5-3B、Llama-3.2系列）进行报告与JSON合成，并结合代码验证脚本对JSON值与表格进行一致性校验。

**📊 数据集**

使用从Sheetpedia等公开来源收集的多语言电子表格，生成约18K训练样本和851个测试样本的STAGE数据集。

**📈 对比分析**

与JSONSchemaBench、Glaive function-calling、ScrapeGraphAI-100K等基线对比，STAGE在自建评测集上Qwen3-4B的Exact Match从31.37%提升至74.27%，Value Accuracy从45.46%提升至90.69%，在DeepJSONEval上亦实现显著提升。

**⚠️ 局限性**

局限在于仅使用电子表格作为知识来源，无法覆盖仅存于自由文本、PDF、数据库或知识图谱等形式的领域知识；实验仅在小型开源模型上验证，缺乏对更大模型的扩展评估。

---

## 369. Generative Engine Optimization at Scale: Measuring Brand Visibility Across AI Search Engines

**arXiv ID:** 2606.20065 | [PDF](https://arxiv.org/pdf/2606.20065v1)

**作者:** Pratyush Kumar `[一作]` `[通讯]` (Ranqo), Pratyush Kumar (Ranqo)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过Ranqo平台对五大AI搜索引擎进行100k+查询，量化品牌在未命名情境下的可见度，并提出三阶品牌地位可见度阶梯。

**💡 创新点**

创新点在于首次构建大规模GEO测量基线，发现品牌地位对AI可见度的30个百分点阶梯，并揭示来源偏好与情感噪声特征。

**🔧 技术方法**

采用API自动生成提示、提取品牌提及、情感、引用源，并结合六维页面审核评分与闭环推荐算法。

**📊 数据集**

数据集包括102个品牌、102,025条回答、149,912条引用、3,508次完整跟踪运行，覆盖ChatGPT、Gemini、Perplexity、Claude、Grok等。

**📈 对比分析**

通过跨平台来源重叠、位置衰减、情感与提及稳定性等多维度度量，结果显示Tier1品牌可见度达73%，Tier2 44%，Tier3 11%，情感噪声显著高于提及。

**⚠️ 局限性**

局限在于观察性数据、厂商样本偏倚、单一模型配置、手工品牌分层、情感分类器误差，且未验证推荐干预的因果效应。

---

## 370. A Neuromorphic Reinforcement Learning Framework for Efficient Pathfinding in Robotic Mobile Fulfillment Systems

**arXiv ID:** 2606.20031 | [PDF](https://arxiv.org/pdf/2606.20031v1)

**作者:** Junzhe Xu `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `8d10c613-917e-4880-9716-17789f50e119` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现了一套端到端的SDQN-RMFS框架，将基于强化学习的多AGV路径规划与脉冲神经网络相结合，实现了低功耗、低延迟的实际部署。

**💡 创新点**

核心创新点包括：碰撞允许训练策略提升DQN在高密度环境中的收敛稳定性；硬标签知识蒸馏将连续Q值转化为离散一热分布，解决ANN-至-SNN的输出分布不匹配问题；参数缩放加速信息流动，显著降低SNN所需时间步。

**🔧 技术方法**

所用技术包括：DQN（使用双重DQN和经验回放）、硬标签蒸馏、IF模型的SNN、硬件感知参数缩放、Speck神经形态芯片进行物理部署。

**📊 数据集**

使用自定义RMFS仿真数据集：16×16网格、100个存储单元、1–8个AGV的多任务轨迹，用于训练、转换及评估。

**📈 对比分析**

与GPU上原始ANN、GPU上SNN、以及传统RL基线（Dueling DQN、PPO、Actor-Critic）对比，SNN在Speck芯片上实现能耗降低≈11,281×、推理延迟减半，决策质量与原ANN相当。

**⚠️ 局限性**

局限性包括：依赖离线仿真训练，未在真实仓库环境验证；低时间步下仍存在量化误差；对硬件平台高度依赖，需额外安全预判机制。

---

## 371. Hybrid Diffusion Transformer for Instruction-Guided Audio Editing via Rectified Flow

**arXiv ID:** 2606.20101 | [PDF](https://arxiv.org/pdf/2606.20101v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 372. Stitching and dimensionality effects on large artificially generated volume datasets

**arXiv ID:** 2606.20095 | [PDF](https://arxiv.org/pdf/2606.20095v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 373. Adaptive Distance-Aware Trunk Deep Operator Learning for Long-Span Roadway Bridges

**arXiv ID:** 2606.20015 | [PDF](https://arxiv.org/pdf/2606.20015v1)

**作者:** Bilal Ahmed `[一作]` (New York University Abu Dhabi), Mostafa E. Mobasher `[通讯]` (New York University Abu Dhabi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种针对长跨桥梁的自适应DeepONet框架，用于高效预测局部结构响应并快速生成影响线与影响面。

**💡 创新点**

创新点在于引入基于KNN的自适应学习域与距离感知的主干特征，并结合弹性Schur补充实现对局部高梯度响应的精准捕捉与物理一致性。

**🔧 技术方法**

采用Deep Operator Network（DeepONet）、KNN自适应域、距离感知特征、物理驱动的Schur补全场重构、等效壳层降阶模型以及车轮级加载分解等技术。

**📊 数据集**

在合成桥梁模型和阿布扎比Mussafah桥的等效壳层有限元模型上生成了约1.6万和4万条加载样本，涵盖不同车轮位置与强度。

**📈 对比分析**

与传统全域DeepONet和固定Schur-DeepONet进行对比，结果显示自适应方法在预测误差（多数分量<5%）和推理速度（单次预测0.3-0.4s、影响面生成4-5倍速）上显著优于基线。

**⚠️ 局限性**

仅针对静态线性载荷；自适应域基于几何距离，未考虑结构刚度、边界或加载路径；未处理动态、非线性或损伤演化问题。

---

## 374. Beyond Static Endpoints: Tool Programs as an Interface for Flexible Agentic Web Services

**arXiv ID:** 2606.19992 | [PDF](https://arxiv.org/pdf/2606.19992v1)

**作者:** Mugeng Liu `[一作]` (Peking University), Yun Ma `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将 LLM 生成的工具调用抽象为可执行的“工具程序”，通过服务端编译执行，显著降低多步工作流的网络往返和客户端推理负担。

**💡 创新点**

创新点在于：①将工具调用包装为可执行程序并显式标注 READ/WRITE 影响；②引入约束驱动的程序生成、编译与修复机制；③实现效应感知重放保证一次性写操作；④基于在线性能分析的自适应聚合策略。

**🔧 技术方法**

技术细节包括 LLM 合成、约束引导的程序构造、编译器与运行时反馈的局部修复、效应类型检查与重放、基于 WebAssembly 的安全沙箱以及自适应聚合决策。

**📊 数据集**

使用公开的开源应用 Memos、Directus、MinIO 以及四个复杂工作流 benchmark 进行评测。

**📈 对比分析**

与传统逐点调用基线（MWS）和改进版本（-step、-prog）对比，工具程序模式在 N=10 时可将端到端延迟降低至 53.4% 以内、客户端流量降至 96.1% 以内，且在高网络延迟与大工作流复杂度时收益更显著。

**⚠️ 局限性**

局限性包括对 LLM 代码生成质量的依赖、对非幺等或非确定性端点的重放策略有限、程序构造与修复的运行时开销在短小工作流中可能抵消收益、以及需要手工提供效应注解。

---

## 375. EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies

**arXiv ID:** 2606.20092 | [PDF](https://arxiv.org/pdf/2606.20092v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 376. IHUBERT: Vector-Based Semantic Deduplication and Domain-Balanced Pretraining for Persian Resources

**arXiv ID:** 2606.20089 | [PDF](https://arxiv.org/pdf/2606.20089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 377. See-and-Reach: Precise Vision-Language Navigation for UAVs within the Field of View

**arXiv ID:** 2606.20045 | [PDF](https://arxiv.org/pdf/2606.20045v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 378. Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution

**arXiv ID:** 2606.20014 | [PDF](https://arxiv.org/pdf/2606.20014v1)

**作者:** Jannik Hösch `[一作]` (Electronic Arts), Linus Gisslén `[通讯]` (Electronic Arts)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了将预训练大型语言模型（LLM）作为多智能体团队的高层控制器，LLM基于全局状态挑选预训练的RL技能策略进行实时协作，并在自制的2v2 King of the Hill Unity游戏环境中进行评估。

**💡 创新点**

创新点在于：1) 将LLM与多技能RL分层架构结合，完成高层策略规划与低层执行的分离；2) 通过预训练模型实现无人工规则的决策，显著降低开发复杂度；3) 结合玩家体验研究，系统评估LLM驱动智能体的“人类感知度”。

**🔧 技术方法**

采用Gemma 3 27B LLM进行高层决策，使用PPO在Unity ML‑Agents环境中训练四个专用RL技能策略，搭配行为树（BT）和单体Flat RL作为基线；实验通过统计检验（z‑检验、卡方检验）以及玩家Likert量表进行比较。

**📊 数据集**

使用自制的2v2 King of the Hill Unity场景生成的对战数据；训练阶段收集每个技能的强化学习轨迹；评估阶段运行1,000集对战；用户研究共15名参与者进行对抗测试。

**📈 对比分析**

通过在1,000集对战中比较胜率、K/D、伤害、拾取等指标：LLM+RL与手工BT胜率无显著差异（46.4% vs 51.5%，p=0.103），均显著优于Flat RL（67.0% vs 19.2%，p<0.001）；在玩家主观评估中，60% 认为LLM+RL最具人类感知度，且在技能、协调、娱乐等维度均得分最高。

**⚠️ 局限性**

局限性包括：仅在单一小型游戏环境和单一LLM模型上验证；用户研究样本量有限（n=15）；未做系统的prompt ablation；LLM对技能执行缺乏感知，导致某些场景下的切换不够精确；未测试更大规模或协作任务的泛化能力。

---

## 379. Self-Adaptive Scale Handling for Forecasting Time Series with Scale Heterogeneity

**arXiv ID:** 2606.20010 | [PDF](https://arxiv.org/pdf/2606.20010v1)

**作者:** Xu Zhang `[一作]`, Wei Wang `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种自适应尺度处理模块（AS），用于解决时间序列预测中的尺度异质性问题，提升跨尺度数据的预测性能。

**💡 创新点**

创新点在于：①通过尺度校准（SC）子模块学习自适应尺度因子；②引入尺度选择（SS）子模块利用Gumbel-Softmax实现可区分的尺度校准选择，防止过度校准；③在尺度不变的WMAPE损失上进行训练。

**🔧 技术方法**

技术包括卷积特征提取、全连接网络与Sigmoid校准、Gumbel-Softmax可微分二值采样、以及与Transformer/Informer/Autoformer/Performer等多种时间序列预测骨干的无缝集成。

**📊 数据集**

使用阿里巴巴蚂蚁财富平台收集的基金销售信息数据，分为 Fund_1（66个基金）和 Fund_2（106个基金）两组，包含购买和赎回金额两条时间序列。

**📈 对比分析**

在与普通窗口尺度（VS）及无预处理（nop）、全局标准化/归一化等策略对比的实验中，AS 在所有四个骨干模型和两组数据集上均取得最低的 WMAPE、RSE，显著优于基线。

**⚠️ 局限性**

局限性包括：对同一产品内多变量时尺度校准可能不稳定；当前的批量平均策略只能部分缓解训练过程中的尺度因子不稳定；在更大尺度差异的场景下需要进一步研究跨变量尺度协调方法。

---

## 380. StreamKL: Fast and Memory-Efficient KL Divergence for Boosting Attention Distillation

**arXiv ID:** 2606.20005 | [PDF](https://arxiv.org/pdf/2606.20005v1)

**作者:** Guangda Liu `[一作]` (Shanghai Jiao Tong University), Jieru Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一种一遍计算的融合原语，用于在不存储 O(N_Q N_K) 维度的注意力分布的情况下进行注意力蒸馏（KL 散度）前向和后向计算。

**💡 创新点**

创新点包括：① 对两分布 KL 散度进行在线重写，得到只需维护 5 个标量即可在流水线上完成累加的公式；② 以此为基础设计了前向单核化、Split‑K 并行、以及融合的后向原语；③ 引入 TMA 异步 bulk copy、指数运算优化、自动调优等实现细节。

**🔧 技术方法**

主要技术手段：在线 softmax（在线 log‑sum‑exp）、张量核心友好的向量级重缩放、分块/拆分‑K 并行、融合单核化、重计算后向、TMA 异步复制、指数运算取代自然指数、自动调优（Triton autotuner）。

**📊 数据集**

实验没有给出具体的数据集，而是使用通用的 LLM 训练场景（batch=16/32，context 长度 4K–512K），在 A100、H200 GPU 上进行基准测试。

**📈 对比分析**

与三种基线（PyTorch eager、PyTorch graph 编译 + Inductor、FLA）对比，前向时 18×–43× 加速，后向时 2.6×–14× 加速；同时把额外 HBM 用量从 O(N_Q N_K) 降到 O(1)，在 64K 以上上下文能单卡完成，原始实现往往 OOM。

**⚠️ 局限性**

限制：① 仍需在前向后向中进行重计算，导致在算力受限的 GPU 上后向性能略逊；② 依赖 GPU 的 TMA 与张量核心特性，对旧硬件支持有限；③ 仅针对注意力蒸馏（KL 散度）设计，其他注意力变体需进一步适配。

---

## 381. Self-Preference Is Weak or Absent in Verifiable Instruction-Following Revision: A Four-Model Test Under Genuine Authorship

**arXiv ID:** 2606.20093 | [PDF](https://arxiv.org/pdf/2606.20093v1)

**作者:** William Guey `[一作]` (Tsinghua University), Pierrick Bougault `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在修订自身草稿时是否存在自我偏好，使用可验证的指令跟随任务与官方检查器进行实验。

**💡 创新点**

首次在可验证指令跟随修订情境中，结合真正的作者身份和无作者身份的对照，发现作者对验证合格修订的拒绝率与对照相近，并详细分析拒绝理由为缺陷捕捉而非偏好。

**🔧 技术方法**

采用 IFEval 官方检查器、OpenRouter 提供的 mid-tier LLMs、温度 0.7 的多次回答与配对 Bootstrap 置信区间统计方法。

**📊 数据集**

使用 IFEval 数据集（约541条可验证指令提示），涉及关键词、大小写、标题等原子可修订约束。

**📈 对比分析**

对每条草稿生成一条通过检查器验证的修订，分别由作者和不同家族的 fresh 模型判断，计算拒绝率差异 G；结果显示 G 约为 -5.1，置信区间包含 0，表明无显著自我偏好。

**⚠️ 局限性**

局限：仅测试 mid-tier 模型；约束为低纬度原子型，未涉及高纬度或全局修订；比较者家族严格度差异可能混杂，样本量有限，无法排除 <13 的细小效应，未测试前沿模型。

---

## 382. Residual-Space Evolutionary Optimization via Flow-based Generative Models

**arXiv ID:** 2606.20084 | [PDF](https://arxiv.org/pdf/2606.20084v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 383. Process-Verified Reinforcement Learning for Theorem Proving via Lean

**arXiv ID:** 2606.20068 | [PDF](https://arxiv.org/pdf/2606.20068v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 384. AI Conversational Interviewing: Scaling Up Semi-Structured and In-depth Interviews

**arXiv ID:** 2606.20064 | [PDF](https://arxiv.org/pdf/2606.20064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 385. VFILC: Accurate Frequency Extrapolations in Imitation Learning via Sampling Frequency ILC

**arXiv ID:** 2606.20056 | [PDF](https://arxiv.org/pdf/2606.20056v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 386. Comparative Study of Neural Surrogate Architectures for Autoregressive Prediction of Internal Battery States

**arXiv ID:** 2606.20053 | [PDF](https://arxiv.org/pdf/2606.20053v1)

**作者:** Gihyun Lee `[一作]` (Technische Universität Berlin), Sangyoung Park `[通讯]` (Technische Universität Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对DFN模型的内部电化学状态进行自回归预测，并系统比较四种神经网络架构。

**💡 创新点**

在统一训练框架下揭示空间诱导偏差对预测精度与稳定性的影响，发现U‑Net在多尺度特征上最优，并引入多步无滚动训练与当前条件化提升模型鲁棒性。

**🔧 技术方法**

使用MLP、ResNet、U‑Net、FNO四种网络，结合FiLM/CIN/拼接条件化、多步滚动训练，并用PyTorch实现模型与PyBaMM生成的DFN仿真数据。

**📊 数据集**

利用122条DFN仿真轨迹（-3.0C~3.24C C率，0%–100% SOC，WLTC/UDDS/US06等多种驱动）构建训练、验证、测试集。

**📈 对比分析**

通过nRMSE、终端电压与SOC误差、推理延迟与速度提升进行比较；U‑Net在300步自回归后nRMSE≈3%，速度提升5.38×，其余模型性能递减。

**⚠️ 局限性**

仅基于等温DFN单一电池化学，未实现闭环校正，且需验证在多化学、热耦合与衰变场景下的可迁移性与长期稳定性。

---

## 387. Alzheimer's Disease Diagnosis using a Multimodal Approach with 3D MRI and PET

**arXiv ID:** 2606.20037 | [PDF](https://arxiv.org/pdf/2606.20037v1)

**作者:** Loukas Ilias `[一作]` (National Technical University of Athens), Dimitris Askounis `[通讯]` (National Technical University of Athens)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

将3D MRI与PET数据融合，构建端到端的多模态诊断网络并使用稀疏门控混合专家进行自适应分类

**💡 创新点**

首次在同一网络中结合三种融合策略（拼接、GMU、自注意力）与稀疏门控MoE，实现输入自适应多模态学习

**🔧 技术方法**

3D CNN特征提取、Gated Multimodal Unit、门控自注意力、稀疏门控Mixture of Experts、Grad‑CAM可视化

**📊 数据集**

采用阿尔茨海默病神经影像倡议（ADNI）数据集，包含T1加权MRI和FDG‑PET的379名受试者

**📈 对比分析**

与单模态与多模态基线（RF、3D CNN、autoencoder、multiscale DNN等）对比，NC‑AD任务准确率达95.47%，其余任务均优于或匹配现有最高水平

**⚠️ 局限性**

依赖大量标注数据且仅使用影像信息，未结合基因或临床特征，数据规模有限

---

## 388. Activation- and Influence-Aware Ranks (AIR): Function-Preserving SVD Compression for LLMs

**arXiv ID:** 2606.19993 | [PDF](https://arxiv.org/pdf/2606.19993v1)

**作者:** Nico Harder `[一作]` (Fraunhofer HHI), Wojciech Samek `[通讯]` (Fraunhofer HHI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Activation‑和 Influence‑Aware Ranks（AIR）框架，利用后向信号的影响度在SVD压缩中调整低秩近似，获得更高效的 LLM 压缩；

**💡 创新点**

创新点在于通过闭式 ALS 单次层本地迭代，将前向激活与后向影响融合，既保持了激活感知的优势，又无需完整端到端微调即可提升压缩质量；

**🔧 技术方法**

使用的技术包括 SVD 低秩分解、Profiling 矩阵、LRP/Weight×Gradient/Fisher 影响度、闭式 ALS 优化、LoRA 微调以及 GPTQ 量化；

**📊 数据集**

实验数据集涵盖 WikiText‑2（校准与评估）、C4、OpenBookQA、ARC‑E、WinoGrande、HellaSwag、PIQA、MathQA 等；

**📈 对比分析**

与 SVD‑LLM(W)、ASVD、FWSVD、ACIP 等基线在相同校准样本下比较，60% 参数保留时 WikiText‑2 perplexity 降低 18% 以上，匹配 SVD‑LLM(W) 仅需 90% 校准数据，且与 LoRA、全量微调、量化组合后可实现约 64% 显存、53% 推理延迟收益；

**⚠️ 局限性**

局限性包括仅适用于单向 Transformer（如 LLaMA），未扩展到 70B+、Encoder‑Decoder、MoE 等结构，且采用统一的 per‑layer 率，缺乏动态秩分配和 token‑级预算适配。

---

## 389. Learning Critical Testing Literacy Through Puzzles: an Experience Report

**arXiv ID:** 2606.20129 | [PDF](https://arxiv.org/pdf/2606.20129v1)

**作者:** Niels Doorn `[一作]` (Open Universiteit), Beatriz Marín `[通讯]` (Universitat Politècnica de València)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究通过组织13次基于谜题的工作坊，使用P4TEST教学框架进行软件测试素养培养，并在后续两次工作坊中加入工作簿和思维口述方法，系统收集参与者的自评、开放式反思以及口述记录，以探究谜题学习如何支持测试者的感知、实验、沟通和情感等关键能力。

**💡 创新点**

创新点在于：①首次将谜题学习与P4TEST框架结合，形成完整的“设定-解决-反思-关联”教学序列；②引入工作簿与思维口述双重数据采集，克服单一观察方法的局限；③开发开源数字平台，提供可配置的在线工作坊环境，实现混合式学习与数据自动化分析；④系统性对比学生与专业测试者在感知、实验与情绪表达等维度的差异，为未来教学设计提供经验。

**🔧 技术方法**

技术手段包括：使用基于P4TEST的工作簿模板（含Likert量表与开放式问题）、音视频录制的思维口述技术、混合式数字平台（支持谜题解答、数据收集与分析）以及定性编码工具进行主题分析；统计分析采用描述性统计与频数分析。

**📊 数据集**

数据集主要包括：①来自学生与专业测试者的Likert量表得分；②每个谜题的开放式反思文本；③思维口述的音频/视频记录与转录文本；所有数据已在公开研究数据集中提供（含代码簿）。

**📈 对比分析**

对比方法：对Likert得分进行描述性统计，比较学生与专业测试者在实验、感知、沟通、情绪等维度的差异；对开放式文本和思维口述转录进行主题编码，比较各维度在两组中的编码比例；结果显示学生在实验与知识整理上得分更高，专业测试者在倾向与经验上得分更高；整体而言，谜题工作坊能激发高水平的实验与感知活动，但缺乏对学习成效的因果评估。

**⚠️ 局限性**

局限性包括：样本规模小且非随机，缺乏对照组与前后测，无法断定因果关系；语言不通可能导致回应偏差；纸质工作簿存在开放式问题高缺失率；思维口述可能引发观察者效应；数据分析主要为描述性，未做显著性检验；缺乏对学习迁移至真实测试任务的实证验证。

---

## 390. Dual-Agent Framework for Cross-Model Verified Translation of Natural-Language Protocols into Robotic Laboratory Platform

**arXiv ID:** 2606.20120 | [PDF](https://arxiv.org/pdf/2606.20120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 391. Autoregressive Modelling and Synthetic Generation of High-Fidelity, Statistically Equivalent 3D Microstructures for As-Manufactured Misalignments in Fiber-Reinforced Composites

**arXiv ID:** 2606.20117 | [PDF](https://arxiv.org/pdf/2606.20117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 392. Pixel-Level Residual Diffusion Transformer: Scalable 3D CT Volume Generation

**arXiv ID:** 2606.20112 | [PDF](https://arxiv.org/pdf/2606.20112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 393. NAMESAKES: Probing Identity Memorization in Text-to-Image Models

**arXiv ID:** 2606.20155 | [PDF](https://arxiv.org/pdf/2606.20155v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 394. Apparent Psychological Profiles of Large Language Models are Largely a Measurement Artifact

**arXiv ID:** 2606.20205 | [PDF](https://arxiv.org/pdf/2606.20205v1)

**作者:** Jelena Meyer `[一作]` (Max Planck Institute for Human Development), Dirk U. Wulff `[通讯]` (Max Planck Institute for Human Development)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对56个指令微调的大型语言模型（LLM）进行性格（IPIP-NEO-300）和风险偏好（Frey等）测评，采用正向-反向关键化的测量模型分离特质与方向性偏差。

**💡 创新点**

揭示LLM的心理学档案主要是测量工具的偏差造成，而非真实特质；响应偏差随模型能力下降但未消失；内部一致性随响应正交度下降而消失；档案随正向/反向题目选择可被人为操纵。

**🔧 技术方法**

使用正式的心理测量框架（双参数模型：特质θ与偏差b），正向-反向关键化设计、协方差分解、正交度与内部一致性分析；利用Pearson相关、Cronbach α、Fisher z置信区间等统计方法。

**📊 数据集**

人类参考样本：IPIP-NEO-300（20,993名受访者）和Frey等的风险偏好样本（1,507名受访者）；LLM面板56个模型（46开源1–70B参数，10专有）。

**📈 对比分析**

与人类人群相比，LLM在正向-反向相关上呈正相关（≈+0.68），人类呈负相关（≈-0.82）；内部一致性与正交度呈高度负相关（r≈-0.95）而人类仅为-0.41；表明LLM的可靠性几乎完全由响应偏差驱动，性能显著低于人类。

**⚠️ 局限性**

局限：仅将模型视为单一受访者，未考虑模型内部变异；仅评估已微调模型，未考察基础模型；仅覆盖人格与风险两大领域，无法推广到其他构念；正交度分析受测量工具设计限制，可能无法完全捕捉复杂偏差。

---

## 395. SA-VIS: Sparse frame Annotations for training Video Instance Segmentation

**arXiv ID:** 2606.20140 | [PDF](https://arxiv.org/pdf/2606.20140v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 396. Tight Algorithm and Hardness for Submodular Linear Ordering

**arXiv ID:** 2606.20202 | [PDF](https://arxiv.org/pdf/2606.20202v1)

**作者:** Evan Abboud `[一作]` (Technion Israel Institute of Technology), Roy Schwartz `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了最小线性排序问题，目标是找到一个排序，使得所有前缀的值之和最小化。

**💡 创新点**

提出了一种多项式时间算法，能够实现O(√(n/ln n))的近似，同时证明了信息论上的难度结果，表明没有算法能在多项式次数内评估f以获得o(√(n/ln n))的近似。

**🔧 技术方法**

使用了递归平衡切割的经典框架，并结合了全局切割的策略来处理一般的非负子模函数。

**📊 数据集**

没有具体提到使用的数据集，但研究的对象是一般的非负子模函数。

**📈 对比分析**

与之前已知的算法进行比较，之前的最佳近似为2，而本研究提供了O(√(n/ln n))的近似，且证明了该近似的难度。

**⚠️ 局限性**

限制在于该算法依赖于子模函数的特性，且在处理非对称子模函数时可能面临挑战。

---

## 397. Effective Dimension Governs Generalization in Quantum Kernel Vision Models

**arXiv ID:** 2606.20183 | [PDF](https://arxiv.org/pdf/2606.20183v1)

**作者:** Jian Xu `[一作]` (RIKEN iTHEMS), Qibin Zhao `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究发现量子视觉模型中“更多纠缠有助泛化”和“噪声有益”这两种现象其实是同一量子特征核有效维度的不同表现，并给出了完整的谱理论与实证验证。

**💡 创新点**

创新点在于提出有效维度（参与度）作为唯一可测的量子特征核泛化度量，解释了纠缠与噪声对泛化的共同机制，并将其转化为可控的设计原则；同时验证了在真实硬件上噪声可作为谱正则化器。

**🔧 技术方法**

使用的技术包括：量子特征映射（data‑re‑uploading 量子电路）、Hilbert–Schmidt 核、谱分解、有效维度与岭维度的理论推导、噪声通道（全局退相干、幅度耗散）的分析、核SVM分类、实验中的噪声扫描、实证对比、以及在 IBM Heron 硬件上的测量。

**📊 数据集**

所用数据集：Digits（10 类）、Fashion‑MNIST、以及医疗图像数据集 BloodMNIST（4 类和 8 类版本）。全部数据先进行 PCA 降维到 6–8 维。

**📈 对比分析**

比较方法：对不同纠缠拓扑（product、chain、ring、all‑to‑all）和噪声强度进行网格扫描，用核SVM评估测试准确率；通过计算有效维度与测试准确率的相关性和 R² 评估。结果显示：有效维度与测试准确率在纠缠模型中呈负相关（R²≈0.82–0.92），噪声导致的谱收缩在过拟合区能提升准确率（最高 +13%），并出现“倒 U”最优噪声水平；在真实硬件中噪声同样能提升准确率（+0.037）。

**⚠️ 局限性**

局限性：有效维度不是唯一决定泛化的统计量，仍需考虑标签对齐和谱形状；理论中的单调性和正则化效果仅在常数行和下界条件下严格成立；实验受限于 8–12 qubit 量子电路、仅在小型图像数据上验证；在真实硬件上噪声需足够深度且处于过拟合区才能获益；未给出最优噪声水平的闭式表达式，反转 U 形状仍为经验观察。

---

## 398. From Texts to Scores: Tracing the Emergence of Essay Quality Representations in Large Language Models

**arXiv ID:** 2606.20152 | [PDF](https://arxiv.org/pdf/2606.20152v1)

**作者:** Jiaxu Zuo `[一作]` (University of Macau), Derek F. Wong `[通讯]` (University of Macau)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统分析了八大LLM在英、葡三种数据集上内部隐藏表示的书面质量信息，探究其线性可解性与神经元作用

**💡 创新点**

揭示LLM隐藏层中书面质量信息呈线性可解、可跨提示迁移，并识别出与评分高度相关的单个“评分神经元”，及其随文本长度层分布的规律

**🔧 技术方法**

采用线性/非线性探针、交叉提示泛化、PCA降维与神经元干预等方法对LLM内部表示进行解释性分析

**📊 数据集**

使用ASAP++、CSEE（英语）和ENEM（葡萄牙语）三大评卷数据集

**📈 对比分析**

线性探针在深层表示上即可达到与监督AES相近的QWK性能，跨提示泛化仍优于随机，PCA低维子空间已捕获大部分排序信息，非线性探针提升有限

**⚠️ 局限性**

实验仅覆盖1B–14B规模开源LLM，未检验更大商业模型；仅试验两种语言，缺乏跨文化评估；神经元干预局限于单一神经元，未探究更复杂的机制交互

---

## 399. An MSO Framework for Weak-Memory Verification and Robustness

**arXiv ID:** 2606.20134 | [PDF](https://arxiv.org/pdf/2606.20134v1)

**作者:** Giovanna Kobus Conrado `[一作]` (Aarhus University), Andreas Pavlogiannis `[通讯]` (Aarhus University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了弱内存模型在树宽与MSO框架下的可 axiomatizability，推导了多种模型的树宽特性，并提出了读自一致性（reads‑from robustness）的新稳健性判定方法。

**💡 创新点**

首次将树宽与MSO理论统一用于弱内存模型分析，证明哪些模型可在MSO中 axiomatize，揭示MSO不可表达模型与细粒度复杂度的关系，并给出读自稳健性的算法性结果。

**🔧 技术方法**

使用树宽理论、Courcelle定理、MSO逻辑以及细粒度复杂度分析，构造部分修改顺序以证明一致性，并设计了基于MSO的判定算法。

**📊 数据集**

本研究以理论证明为主，无专门实验数据集；若有实验，使用标准弱内存模型示例（如SC、TSO、RMW、RC20等）进行验证。

**📈 对比分析**

通过理论复杂度证明，针对MSO可 axiomatizable 的模型实现了多项式时间的稳健性判定；相较于仅适用于有限执行或基于上下文切换的技术，本方法支持无限执行并提供更宽松的稳健性判定。

**⚠️ 局限性**

受限于部分模型的MSO可表达性（如 Strong R/A 与某些组合）以及对细粒度复杂度假设（SETH）的依赖；实验验证尚未展开，实际工具实现仍待后续工作。

---

## 400. The Correctness Illusion in LLM-Generated GPU Kernels

**arXiv ID:** 2606.20128 | [PDF](https://arxiv.org/pdf/2606.20128v1)

**作者:** Dipankar Sarkar `[一作]` `[通讯]` (Arizona State University), Dipankar Sarkar (Arizona State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对LLM生成的GPU内核，提出一种基于算子模式（op‑schema）的种子化模糊测试（seeded oracle），通过高精度FP64参考和绝对误差阈值，自动发现传统固定形状、小样本检验（allclose‑style）无法检测的正确性幻觉。

**💡 创新点**

创新点在于：① 用算子模式生成真实输入形状并覆盖边界值，显著提升对形状相关 bug 的检出率；② 采用绝对误差阈值并记录完整误差分布，实现对不同 op 与 dtype 的精确校准；③ 构建可复现的评估管道并在五类 GPU 上交叉验证，证明方法的跨硬件一致性。

**🔧 技术方法**

主要技术包括：算子模式-aware 模糊器（基于符号维度采样），FP64 高精度参考核，按(op, dtype)绝对误差阈值的验证器，跨 GPU 自动化执行框架（vast.ai + Backblaze B2），以及失败重放脚本。

**📊 数据集**

数据集为人工标注的控制与错误内核组合，共 26 个算子（24 个单 GPU 版，另 2 个 flash‑attention），其中 15 个正确控制与 9 个 LLM 风格错误（再加 2 个跨 GPU 扩展）。

**📈 对比分析**

比较方法：将传统的固定形状、固定容差的 allclose‑style oracle（bench）与本方法（gpuemu）在同一数据集上并行评估。结果显示：bench 在所有 9 个错误内核上均通过，gpuemu 检测全部 9 个错误且所有 15 个控制通过；跨 GPU 扩展同样保持一致。性能上，seeded oracle 仅在检测错误时增加极小的精度开销，控制内核保持零额外成本。

**⚠️ 局限性**

局限性：① 仅使用作者手工种子化的错误模式，未覆盖真实 LLM 生成的多样错误；② 当前验证器仅做同 dtype 对比，跨 dtype 的误差分析尚未实现；③ Python 客户端缺乏 native bf16 支持，导致 bf16 测试仅通过 fp16 代理；④ 模糊器未覆盖非连续内存布局对结果的影响。

---

## 401. Pose6DAug: Physically Plausible Multi-view Object Swapping for Robot Data Augmentation

**arXiv ID:** 2606.20118 | [PDF](https://arxiv.org/pdf/2606.20118v1)

**作者:** Jonghoon Lee `[一作]` (KAIST), Jinwoo Shin `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于3D网格重建和6D位姿跟踪的失败驱动数据增强框架，通过在已成功演示中替换物体并保持动作轨迹，生成针对新对象的多视角演示，无需额外采集数据。

**💡 创新点**

核心创新在于：①在3D空间进行物体替换，保证几何一致性与物理可行性；②利用统一的6D位姿轨迹对所有摄像头渲染同一网格，实现多视角一致性；③引入随机旋转、平移、缩放等几何扰动，提升样本多样性。

**🔧 技术方法**

技术细节包括：SAM3D等图像到3D模型方法获取目标网格；视频掩码抠图 + inpainting 完成背景填充；机器人正向运动学与6D位姿序列提供轨迹约束；光栅渲染保证跨摄像头的几何一致性。

**📊 数据集**

在RoboCasa365仿真数据集的Counter-to-Cabinet任务上进行实验，针对8个最难的对象进行增量训练。

**📈 对比分析**

与VACE（2D视频编辑）和MimicGen（仿真轨迹适配）对比，提升成功率至22.8%（VACE16.4%，MimicGen15.8%），覆盖率24.5%（VACE18.2%，MimicGen17.2%）；单独用增量数据训练在8硬件对象上实现21.2%成功率（VACE15%，MimicGen5.7%），覆盖7/8对象。

**⚠️ 局限性**

局限性在于：①依赖外部网格重建和位姿估计，对极端遮挡或未知形状的重建精度有限；②增强数据受原始成功轨迹覆盖范围限制，无法覆盖所有失败情形；③对真实场景中非仿真光照、材质的适应性尚待验证。

---

## 402. Modularity-Free Conflict-Averse Training for Generalized PINNs

**arXiv ID:** 2606.20156 | [PDF](https://arxiv.org/pdf/2606.20156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 403. EFIQA: Explainable Fundus Image Quality Assessment via Anatomical Priors

**arXiv ID:** 2606.20108 | [PDF](https://arxiv.org/pdf/2606.20108v1)

**作者:** Pengwei Wang `[一作]` (Medical University of Vienna), Hrvoje Bogunović `[通讯]` (Medical University of Vienna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出EFIQA框架，利用解剖学先验通过无监督的血管重建与知识蒸馏，实现不依赖质量标签、可解释的空间质量评估；

**💡 创新点**

创新点在于将“缺失解剖结构”作为质量判定依据，构建两阶段无监督血管异常检测网络与基础模型适配器蒸馏，从而既获得局部质量图又保持良好的跨数据集泛化；

**🔧 技术方法**

采用遮蔽解剖重建（masked anatomical inpainting）+残差网络训练VUAD，再将其伪标签蒸馏至冻结的视觉基础模型（Vision FM）适配器，使用MSE、BCE等损失；

**📊 数据集**

使用Messidor‑2（高质量血管图）训练VUAD，内部4064张混合质量CFP训练适配器；评估数据集为MSHF、mBRSET、DRIMDB、EyeQ四个公开数据集；

**📈 对比分析**

与MCF‑Net、FGR‑Net、AutoMorph、FIT、MANIQA、TOPIQ、NIQE、IL‑NIQE、ARNIQA等SOTA方法在外部数据集进行BAcc、F1、MCC、AUROC、AUPRC等指标比较，EFIQA平均提升约3.4pp的MCC，整体性能优于或接近最优；在EyeQ亦保持竞争力；

**⚠️ 局限性**

局限在于仅以血管为解剖先验，导致对无血管区或病变区误警；聚合策略过于简单，未考虑FOV、解剖重要性等临床因素；当前仅针对CFP图像，扩展到其他模态仍待验证。

---

## 404. Distill Once, Adapt Life-Long: Exploring Dataset Distillation for Continual Test-Time Adaptation

**arXiv ID:** 2606.20196 | [PDF](https://arxiv.org/pdf/2606.20196v1)

**作者:** Hyun-Kurl Jang `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 DO-ALL 框架，先在源域上使用数据集蒸馏生成少量合成锚点，部署后在持续的测试流中通过锚点匹配为每个目标样本提供稳定参考，并以此进行源重放、MixUp 以及特征对齐等正则化，实现持续测试时的自适应。

**💡 创新点**

创新点在于将数据集蒸馏作为一次性、隐私友好的源知识压缩方式，引入锚点匹配与多重正则化（重放+MixUp+MMD+自适应融合），使得框架可无缝嵌入现有 CTTA 方法，显著提升长期适应鲁棒性。

**🔧 技术方法**

核心技术包括数据集蒸馏（如 WMDD/SRe2L/DELT）、锚点与目标样本的语义匹配、源重放损失、MixUp 目标混合、MMD 特征对齐、以及基于梯度的有害参数融合。

**📊 数据集**

实验在 CIFAR100-C、ImageNet-C 以及长时序 CCC 基准上进行，涉及 15 种污染类型和不同强度。

**📈 对比分析**

与多种基线 CTTA 方法（EATA、RMT、ROID、CoTTA 等）比较，DO-ALL 在 ImageNet-C 上平均误差从 58% 降至 56%，在 CCC 上平均精度从 33.6% 提升至 34.7%；在所有基准中均表现出稳定的性能提升。

**⚠️ 局限性**

局限性包括：需在部署前预先完成蒸馏，蒸馏质量对结果敏感；在极端漂移或突发攻击性噪声下的鲁棒性尚未充分验证；与某些低资源场景的计算与存储开销仍有提升空间。

---

## 405. Randomized Sketching is Robust to Low-Precision Rounding on GPUs

**arXiv ID:** 2606.20195 | [PDF](https://arxiv.org/pdf/2606.20195v1)

**作者:** Aryaman Jeendgar `[一作]` (Technical University of Munich), Hartwig Anzt `[通讯]` (Technical University of Munich)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在 GPU 上实现了稀疏随机投影（SparseStack）的混合精度版本，并在 FP16 与 FP32 中比较了不同的舍入策略（确定性、随机化、加性抖动）

**💡 创新点**

证明了对 SparseStack 的 FP16 量化几乎不影响子空间嵌入质量，且在 GPU 上通过降低原子更新量和内存带宽来显著提升吞吐量

**🔧 技术方法**

利用 CUDA 原子操作、半精度向量化累加、三种 FP16 舍入实现（round‑to‑nearest、exact stochastic rounding、dithered rounding），以及基准 Sketch‑and‑Solve 最小二乘求解

**📊 数据集**

使用合成数据集：高斯随机矩阵、齐次身份矩阵、可调共振度的正交子空间、密集随机最小二乘实例以及高度共振的对抗性最小二乘实例

**📈 对比分析**

与传统 CountSketch、FlashSketch、FP32 SparseStack 以及 CountSketch‑MxP（混合精度）比较；FP16 SparseStack（确定性）在保持与 FP32 同等精度的同时，吞吐量比 FP32 提升约 1.5–2 倍，且在所有实验中精度相近

**⚠️ 局限性**

仅给出经验结果，缺乏对量化 SparseStack 的理论 OSE 证明；量化误差在某些极端输入下可能仍影响性能；目前仅针对稀疏投影，其他分布或结构化投影的适用性未验证

---

## 406. Hybrid ANN-SNN Pipeline with Local Plasticity

**arXiv ID:** 2606.20151 | [PDF](https://arxiv.org/pdf/2606.20151v1)

**作者:** Denis Larionov `[一作]` (Chuvash State University), Ivan Tugoy `[通讯]` (LLC 1T)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种混合 ANN‑SNN 架构，使用预训练的 EfficientNet‑B3 作为冻结的特征提取器，再将其激活向量通过率编码转换为 spike 并喂入 CoLaNET 形单层 SNN 分类器，仅利用局部可塑性规则进行训练；

**💡 创新点**

创新点在于将强大的预训练 ANN 表征能力与高效、可持续学习的 SNN 处理方式分离，证明仅通过本地学习即可在 ImageNet 子集上达到与全连接 ANN 同等的高精度；

**🔧 技术方法**

采用 EfficientNet‑B3 编码器、rate‑coding spike 转换、CoLaNET 的抗 Hebbian、dopamine‑modulated 本地学习规则，以及基因算法+随机下降的超参数优化；

**📊 数据集**

使用了 ImageNet 的 64 类子集（约 35,179 张图像，80/20 训练/测试划分），每张图像尺寸 256×256×3；

**📈 对比分析**

通过单网络测试与 15 规模 Ensemble 对比，SNN 在单次在线训练后获得 99.09% 准确率，几乎与传统 ANN（99.25%）持平，说明局部学习可实现高性能；

**⚠️ 局限性**

局限性包括：①Encoder 预训练已见测试类别，准确率可能受预训练信息影响；②冻结的 ANN 编码器功耗高，未实现完全事件驱动，仍是能耗瓶颈；③仅在 64 类子集验证，未评估在更大或完全新类别上的泛化能力。

---

## 407. RACL: Reasoning-Agent Control Layers for Continuous Metaheuristic Learning

**arXiv ID:** 2606.20142 | [PDF](https://arxiv.org/pdf/2606.20142v1)

**作者:** Antón Asla Manzárraga `[一作]` `[通讯]` (Independent Researcher), Antón Asla Manzárraga (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一种基于推理代理的控制层 RACL，能在现有 metaheuristic（如 ALNS）之上，通过观察运营记忆、推理、有限实验、风险约束和策略整合来提升车辆路径规划的效果。

**💡 创新点**

创新点在于将推理代理作为可解释、可审计的学习层引入 metaheuristic，利用运营记忆驱动的有界实验与决策链，而非传统的单纯操作器选择或强化学习；同时强调不改变业务约束，保证可解释性。

**🔧 技术方法**

技术组合包括：大语言模型 Codex 用于实时推理与生成控制动作；ALNS 车辆路径规划引擎；运营记忆检索与匹配；bounded intervention（有限的算法干预）与 guardrails（风险约束）；策略整合与可解释性输出；统计检验与性能评估。

**📊 数据集**

使用了 Sevilla-9 和 Sevilla-10 两个车辆路径规划数据集，共计 21 个可行案例（含 8 个可比对固定基线的种子）。

**📈 对比分析**

对比方法：与固定基线、基于记忆的策略 OMP、无推理停滞触发策略 STP 等进行同一数据集、相同种子下的配对比较。结果显示：RACL 在 21 个可行案例中，优于/等于 STP 的 18/21 个案例，平均成本降低 -0.641%；优于 OMP 的 18/21 个案例，平均降低 -4.913%；在 8 个与固定基线可比的案例中，平均降低 -8.337%。运行时间基本保持在固定基线附近，无显著额外开销。

**⚠️ 局限性**

局限性包括：验证仅在单一 ALNS 环境下完成，缺乏对其他 metaheuristic、规模更大或不同领域的数据集的通用性评估；当前实现依赖 Codex 在环的推理，实际生产部署仍需完善无缝调用与连续学习机制；对策略退化、过度保守等情况的系统性评估尚未完成；未进行业务用户对可解释性输出的实测验证。

---

## 408. TriFlow: Generating Artist-Like 3D Mesh Topology via Nearest-Vertex Vector Fields

**arXiv ID:** 2606.20131 | [PDF](https://arxiv.org/pdf/2606.20131v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 409. ReNikud: Audio-Supervised Hebrew Grapheme-to-Phoneme Conversion

**arXiv ID:** 2606.20179 | [PDF](https://arxiv.org/pdf/2606.20179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 410. Learning to Prompt: Improving Student Engagement with Adaptive LLM-based High-School Tutoring

**arXiv ID:** 2606.20138 | [PDF](https://arxiv.org/pdf/2606.20138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 411. Quantile of Means: A Bonus-Free Ensemble Method for Minimax Optimal Reinforcement Learning

**arXiv ID:** 2606.20107 | [PDF](https://arxiv.org/pdf/2606.20107v1)

**作者:** Asaf Cassel `[一作]` (Google Research), Aviv Rosenberg `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于分位数的集成方法，用于有限时间范围的马尔可夫决策过程（MDP），以优化强化学习中的探索策略。

**💡 创新点**

该方法首次提供了理论基础，证明了在MDP中实现实例最优的（依赖方差的）遗憾界限，且不需要复杂的计数基础奖励构造。

**🔧 技术方法**

使用了基于分位数的集成方法，结合了量化均值（QoM）估计器，避免了显式的计数和后验计算。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在高维或连续环境中的应用。

**📈 对比分析**

与传统的计数基础方法相比，该方法在不需要奖励分布知识的情况下，达到了与最佳已知结果相匹配的性能，且在多臂赌博机（MAB）设置中也表现出色。

**⚠️ 局限性**

该方法的局限性在于，尽管在MDP中表现良好，但在处理线性或一般函数逼近设置时仍然是一个开放问题。

---

## 412. Implicit Semantic-Aware Communication Based on Hypergraph Reasoning

**arXiv ID:** 2606.20162 | [PDF](https://arxiv.org/pdf/2606.20162v1)

**作者:** Yiwei Liao `[一作]` (China Electric Power Research Institute Co Ltd), Guangming Shi `[通讯]` (Peng Cheng Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于超图推理的隐式语义通信框架HISR。

**💡 创新点**

创新点是将高阶语义关系映射到关系特定语义子空间，解决过平滑问题。

**🔧 技术方法**

采用超图神经网络、子空间投影、位置感知聚合和自适应子空间配置等技术。

**📊 数据集**

在Cora、Pubmed、FB‑Auto和M‑FB15K等基准数据集上进行实验。

**📈 对比分析**

与多种基线相比，HISR在SNR、误差和鲁棒性上提升约36.6%，在低SNR下保持7–13%的优势。

**⚠️ 局限性**

限制包括子空间数与维度需经验调优，且在极高SNR下性能趋于饱和。

---

## 413. ARTEMIS: Agent-guided Reliability-aware Temporal Mask Evolution for Imperfectly Supervised Video Polyp Segmentation

**arXiv ID:** 2606.20161 | [PDF](https://arxiv.org/pdf/2606.20161v1)

**作者:** Tong Wang `[一作]` (Southeast University), Yutong Xie `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种统一框架 ARTEMIS，用于在弱监督（点/涂鸦）和半监督（稀疏全标记帧）下完成视频息肉分割，利用基于SAM2的初始掩码与视觉语言对话式代理进行可靠锚点选择，随后双向传播并通过可靠性加权学习实现精准分割。

**💡 创新点**

创新点包括：① 使用“讨论-裁决”视觉语言代理评估掩码可靠性并动态选择锚点；② 通过双向时间传播将可靠锚点在视频中扩散，解决伪标签几何退化与时序缺失；③ 引入可靠性引导的参考选择、Reference Prototype Transport Module（RPTM）和可靠性加权损失，进一步抑制噪声伪标签并提升时序一致性。

**🔧 技术方法**

核心技术包括 SAM2 生成稀疏提示的稠密掩码、Qwen2.5‑VL‑7B 视觉语言对话式代理、双向时间记忆的 SAM2 推理、RPTM 参考原型传输、可靠性加权 BCE+Dice 损失，以及多尺度深层监督。

**📊 数据集**

在 SUN‑SEG（Easy/Hard, Seen/Unseen）和 CVC‑ClinicDB‑612 两个医学视频数据集上进行实验，分别在点、涂鸦、1/8 和 1/16 训练数据的弱/半监督设置下评估。

**📈 对比分析**

与现有弱/半监督方法（如 SEE、ProMaC、ST‑SAM 等）对比，ARTEMIS 在所有评估指标（Dice、Fβ、MAE、IoU 等）上均取得领先或接近最优成绩，尤其在 Dice 和 Fβ 上提升约 3‑8%，MAE 降低 30‑60% 以上。

**⚠️ 局限性**

局限性：对极端运动、强高光、低对比度边界仍存在分割误差；对长时序视频的漂移与记忆容量仍有待改进；缺乏对不确定性和人机交互的自适应修正。

---

## 414. Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation

**arXiv ID:** 2606.20135 | [PDF](https://arxiv.org/pdf/2606.20135v1)

**作者:** Jianing Guo `[一作]` (Beihang University), Simin Li `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了基于频域的流匹配方法 FAFM，能够生成连续且时序一致的机器人控制动作。

**💡 创新点**

创新点在于将离散动作序列映射到离散余弦变换（DCT）系数空间，并对一阶时间导数引入 Sobolev 正则化，从而解决了异频训练、动作抖动和时序不一致的问题。

**🔧 技术方法**

使用技术包括离散余弦变换、流匹配（Conditional Flow Matching）、Sobolev 正则化（对一阶导数的约束）以及 ODE 求解器实现连续动作回放。

**📊 数据集**

实验数据集涵盖合成小样本、障碍规避、LapGym（腹腔镜手术仿真）、LIBERO（视觉‑语言‑动作任务）以及 Franka 真实机器人演示。

**📈 对比分析**

与 Diffusion Policy、Flow Matching、SFP、FreqPolicy、MPD 等基线方法对比，FAFM 在成功率、运动平滑度（LDLJ）、多模态表达能力、收敛速度和对机械偏置与混合频率输入的鲁棒性方面均显著优于基线。

**⚠️ 局限性**

对需要高频或冲击性动作的任务效果不佳，因为 DCT 适用于低频平滑轨迹，无法很好捕捉急剧变化的动作。

---

## 415. When Calibration Fails the Vulnerable Hospital: Federated Conformal Risk Control via Risk-Curve Shrinkage

**arXiv ID:** 2606.20115 | [PDF](https://arxiv.org/pdf/2606.20115v1)

**作者:** Nafis Fuad Shahid `[一作]` `[通讯]` (Dhaka), Nafis Fuad Shahid (Dhaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种在联邦学习环境下实现分割模型的置信度校准方法，称为基于风险曲线收缩的联邦CRC协议。

**💡 创新点**

创新点在于引入收缩权重n₀平衡各医院的置信度阈值，实现了在保证覆盖率的同时显著提升预测集效率，并证明了收缩阈值对风险分配的影响。

**🔧 技术方法**

采用了风险曲线收缩（类似James–Stein收缩）、有限样本校正项、留一医院外敏感度分析以及分布式数据聚合等技术。

**📊 数据集**

使用了FeTS‑2022脑肿瘤多机构数据集，共20个医院、1,251个体素图像。

**📈 对比分析**

与传统的全局池化CRC和局部CRC相比，该方法在α=0.10时将违规医院数量从约8个降低到1–3个，同时将预测集伸展从约83×降至约2×，在覆盖率与效率之间取得显著平衡。

**⚠️ 局限性**

主要局限在于仅给出边际覆盖保证，缺乏严格的个体医院条件覆盖证明，且方法对校准集比例与测试集比例的匹配敏感，需进一步验证在不同模型与解剖结构上的适用性。

---

## 416. BARReL: a modern backend for Atelier B in Lean

**arXiv ID:** 2606.20121 | [PDF](https://arxiv.org/pdf/2606.20121v1)

**作者:** Ghilain Bergeron `[一作]` (Université de Lorraine), Vincent Trélat `[通讯]` (Université de Lorraine)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

本文提出了一个 Lean4 后端，将 Atelier B 的 B 方法机器与证明负载转换为 Lean 中的定理，支持完整的证明工作流；

**💡 创新点**

核心创新在于将 B 的部分运算显式为带有 well‑definedness 条件的总函数，并在 Lean 中把这些条件作为可证明的子目标，从而实现完整可验证的证明链，并提供轻量级自动化与可插拔设计；

**🔧 技术方法**

实现依赖 Lean4 元编程、Mathlib 集合与关系库、B 语法嵌入、BPO 生成器与 XML 解析、Hilbert 选择器以及 Lean 的依赖类型和自动化策略；

**📊 数据集**

使用了一个三层 Refinement 示例（最小值搜索）作为评估案例，基于此生成了 190 个证明目标；

**📈 对比分析**

与 Atelier B 的交互式证明器对比，生成的证明目标数相近，WD 条件大部分自动闭合；手工证明目标数为 44，全部可在 Lean 中完成，整体证明效率高于传统方法；

**⚠️ 局限性**

局限性包括：生成的 WD 目标过多且未实现子集化；B 子语言覆盖有限（缺少序列、树等）；依赖外部 Atelier B PO 生成器；未实现完全可验证的 PO 生成器。

---

## 417. Belt-Finger: An Affordable Soft Belt-Driven Gripper for Dexterous In-Hand Manipulation

**arXiv ID:** 2606.20193 | [PDF](https://arxiv.org/pdf/2606.20193v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 418. Evaluation of Image Matching for Art Skills Assessment

**arXiv ID:** 2606.20199 | [PDF](https://arxiv.org/pdf/2606.20199v1)

**作者:** Asaad Alghamdi `[一作]` (University of Dayton), Tam V. Nguyen `[通讯]` (University of Dayton)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用图像匹配技术自动评估手绘作品的艺术水平

**💡 创新点**

提出了将传统 SIFT 特征与基于 VGG16 的孪生网络相结合的评估框架，并给出基于匹配百分比的技能等级划分

**🔧 技术方法**

SIFT 键点匹配、Fast Library for Approximate Nearest Neighbors (FLANN)、VGG16 预训练模型、Cosine 余弦相似度、欧氏距离以及深度学习孪生网络

**📊 数据集**

10 类卡通人物模板图像、120 名参与者绘制的 189 份草图以及 83 张用于划分技能等级的标准图像

**📈 对比分析**

通过与人工评估对比，SIFT 方法在相似度评分上与人类评估高度吻合，准确率可达 80%–98%，而 VGG16+孪生网络虽也可分类但准确率略低，且对颜色和背景噪声更敏感

**⚠️ 局限性**

系统难以自动剔除离群点和噪声，SIFT 计算耗时较长，孪生网络对颜色依赖强，Canny 边缘检测参数不稳定，导致部分绘图边缘识别不佳

---

## 419. Pitch Spelling Jazz Lead Sheets, Solo Transcriptions, Classical Piano and Monophonic Scores

**arXiv ID:** 2606.20198 | [PDF](https://arxiv.org/pdf/2606.20198v1)

**作者:** Augustin Bouquillard `[一作]` (École polytechnique), Florent Jacquemard `[通讯]` (INRIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于MIDI输入和小节边界的音高拼写与全局及局部调性估计算法；

**💡 创新点**

创新点包括：1) 将一小节视为注意力窗口，结合最短路径与拼写规则实现可解释的音高拼写；2) 将Weber距离扩展到165种爵士模式；3) 通过两阶段“modal–tonal”优化同时估计全局Key Signature和局部标度；

**🔧 技术方法**

使用最短路径（Viterbi）动态规划、改进的Weber距离、多维状态机处理同音类同名约束、可选的重写规则、以及可配置的成本函数；

**📊 数据集**

评估数据集包括：Real Book 200首、Charlie Parker Omnibook 50首、FiloBass 48首、The Session 62条传统民谣、ASAP 222首古典钢琴、Schumann Kinderszenen 13首、La‑marque‑Goudard 250条；

**📈 对比分析**

与MuseScore自带音高拼写、PKSpell（数据驱动模型）以及Krumhansl‑Schmuckler关键估计做对比。实验显示：在爵士/民谣数据集上，使用165模式并启用重写规则可达约97–98%拼写准确率；在古典数据集上，PSE在Key Signature估计上显著优于K‑S，且在大部分曲目中与PKSpell相当或更好；

**⚠️ 局限性**

局限性包括：1) 对输入音符时值不做利用，只考虑小节边界；2) 对某些数据集的编辑规范差异（如双重升降记号）敏感；3) 在极度即兴或频繁调性跳变的爵士乐段中，局部标度估计仍可能受限；4) 需要手动调参成本较高。

---

## 420. Stable Transformer-Actor-Critic Model Predictive Control: A Contraction Analysis Approach

**arXiv ID:** 2606.20197 | [PDF](https://arxiv.org/pdf/2606.20197v1)

**作者:** Antonio Marino `[一作]` (University of Cambridge), Marco Cognetti `[通讯]` (Laas Cnrs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于Transformer的Actor-Critic模型预测控制（MPC）架构，并通过理论证明和训练正则化实现闭环系统的增量输入输出稳定性与Riemannian收缩性；

**💡 创新点**

创新点在于首次将Transformer的全局增量输入输出稳定性（δISS）与Riemannian收缩理论相结合，形成一个小增益条件来保证完整系统的闭环收敛，并将该理论约束嵌入训练过程；

**🔧 技术方法**

采用的技术包括Transformer序列模型、可微MPC、隐式函数定理、Riemannian收缩理论、增量输入输出稳定性分析以及小增益定理；

**📊 数据集**

实验数据使用仿真三维无人机动力学，生成带有均匀观测噪声和过程扰动的轨迹；

**📈 对比分析**

与未加正则化的Vanilla Transformer-AC-MPC进行对比，结果显示加正则化的网络在目标跟踪误差、碰撞率（0% vs 30%）以及轨迹一致性方面均优于基线方法；

**⚠️ 局限性**

局限性在于全局严格的收缩约束会限制网络在瞬态转弯等激烈动态下的灵活性，未来需要探索状态相关的局部收缩度量以平衡稳健性与机动性。

---

## 421. Evaluating and Enhancing Negation Comprehension in Remote Sensing MLLMs

**arXiv ID:** 2606.20177 | [PDF](https://arxiv.org/pdf/2606.20177v1)

**作者:** Haochen Han `[一作]` (Peng Cheng Laboratory), Fangming Liu `[通讯]` (Peng Cheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RS‑Neg 评测基准和 NeFo 测试时学习方法，以提升遥感多模态大语言模型（MLLM）对否定表达的理解。

**💡 创新点**

创新点：①RS‑Neg 为遥感任务设计的首个涵盖从 region‑level 到 scene‑level 的否定理解基准；②NeFo 通过自监督的真值反转损失与知识保持机制，在不依赖额外标注或教师模型的前提下实现无监督测试时适配，显著提高 MLLMs 的否定推理能力。

**🔧 技术方法**

采用的技术：LLM 驱动的查询合成与多模态对齐、MCTS 动态视觉聚焦验证、LoRA 参数高效微调、KL 反转损失、熵正则化与知识保持正则化。

**📊 数据集**

使用的数据集：通过 7 个遥感公开数据集（NWPU Caption、RSICD、Sydney Caption、VRSBench、AID、UCMerced 等）自动生成 RS‑Neg（约 22k 样本），并在 FloodNet VQA 等未见数据集进行零样本评估。

**📈 对比分析**

比较方法：对比 TENT、SAR、TLM 等现有测试时学习方案，NeFo 在 VQA、MCQ、Grounding、Classification 四大任务上分别提升 4.6%–11.2% 以上，且在未见任务上也实现 2%–6% 的性能提升。

**⚠️ 局限性**

局限性：对状态级否定（如“未被淹没”）的提升相对有限；需要预先构建否定词典，且在样本量极少的场景下 TTL 可能出现过拟合；对大模型的适配仍需更多实验验证。

---

## 422. Computational Methods and Challenges in Cell-Free DNA Analysis for Multi-Cancer Early Detection

**arXiv ID:** 2606.20174 | [PDF](https://arxiv.org/pdf/2606.20174v1)

**作者:** Nicko Starkey `[一作]` (AGH University of Krakow), Krzysztof Rzecki `[通讯]` (AGH University of Krakow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了2022-2025年基于cfDNA的多癌早期检测计算方法，聚焦片段学与表观遗传特征提取与分析，并对方法的生物学可解释性、临床准备度、数据集与评估指标进行系统评估。

**💡 创新点**

首次将传统统计、机器学习与深度学习模型统一对比，强调多模态集成在低测序深度下的优势，并提出标准化评估与数据共享的必要性。

**🔧 技术方法**

采用机器学习（随机森林、梯度提升、SVM）、深度学习（CNN、GCNN、autoencoder）以及集成学习和非负矩阵分解等技术，结合cfDNA片段长度、端点、甲基化、CNV等多维特征。

**📊 数据集**

利用多中心cfDNA WGS/WGBS/ULP‑WGS/WMS等数据集，覆盖多种癌症（HCC、乳腺、肺、结直肠等）及早期/晚期分层，并包含外部验证队列。

**📈 对比分析**

通过AUC、灵敏度、特异度等指标与现有方法（如FRAGMA、FinaleMe、ARTEMIS、THEMIS等）对比，发现集成方法在0.5‑1×测序深度下可达0.95以上AUC，且早期灵敏度普遍在70‑90%。

**⚠️ 局限性**

局限包括样本量不足、人口学变量匹配不充分、对低ctDNA比例的敏感度仍有限、缺乏统一评估标准、bisulfite处理与低覆盖度测序对性能的影响。

---

## 423. Qiskit Code Migration with LLMs

**arXiv ID:** 2606.20173 | [PDF](https://arxiv.org/pdf/2606.20173v1)

**作者:** Jose Manuel Suarez `[一作]`, Alenandro Fernandez `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于自动生成迁移情景分类学的检索增强生成（RAG）架构，用于自动化迁移 Qiskit 代码版本。

**💡 创新点**

创新点在于将数据驱动的分类学作为结构化知识源与 LLM 结合，显著降低幻觉并提高迁移建议的准确性。

**🔧 技术方法**

采用检索增强生成、语义数据库（Qdrant）、低代码工作流（n8n）以及大模型（Gemini Flash‑2.5 与 GPT‑OSS‑20B）。

**📊 数据集**

使用 Qiskit 发布说明生成的迁移情景分类学以及人工合成的 Python 代码片段。

**📈 对比分析**

通过双盲专家评估与停止灯指标比较，Gemini 在限制检索模式下将误判率降至 14% 左右，正确迁移率提升至约 57%，优于 GPT‑OSS‑20B。

**⚠️ 局限性**

局限性包括仅验证合成代码、仅对两款模型和两种检索策略进行实验，缺乏对真实项目和更多模型的泛化评估。

---

## 424. Multi-Modal Contrastive Learning for Implicit Earth Embeddings via Location Tying

**arXiv ID:** 2606.20167 | [PDF](https://arxiv.org/pdf/2606.20167v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 425. BIM-Edit: Benchmarking Large Language Models for IFC-Based Building Information Modeling

**arXiv ID:** 2606.20146 | [PDF](https://arxiv.org/pdf/2606.20146v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 426. Predicting gestational age at birth in the context of preterm birth from multi-modal fetal MRI

**arXiv ID:** 2606.20172 | [PDF](https://arxiv.org/pdf/2606.20172v1)

**作者:** Diego Fajardo-Rojas `[一作]` (King's College London), Jana Hutter `[通讯]` (King's College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用多模态胎儿MRI（解剖T2、功能T2*）与临床变量构建堆叠式机器学习管线，预测出生时的孕周。

**💡 创新点**

首次将多模态MRI特征与堆叠集成相结合，用于预测出生孕周，而非传统的二分类预产期预测。

**🔧 技术方法**

采用随机森林、支持向量回归、XGBoost三类基模型，堆叠元模型；使用MICE插补、内部学生化残差去卷积、特征选择和数据上采样。

**📊 数据集**

426例单胎孕妇数据，包含临床记录、超声测量和胎儿MRI（T2/T2*）解剖及功能特征。

**📈 对比分析**

与仅使用超声或单一模型的分类研究相比，本模型在10折交叉验证中取得R²≈0.13、MAE≈2.74周、准确率0.77、敏感性0.59、特异性0.82。

**⚠️ 局限性**

样本量有限、极早产/非常早产样本不足、类别不平衡、缺乏纵向特征、未区分胎儿/母体临床表型，且MRI成本高。

---

## 427. Robust Assembly State Reasoning from Action Recognition for Human-Robot Collaboration

**arXiv ID:** 2606.20150 | [PDF](https://arxiv.org/pdf/2606.20150v1)

**作者:** James Fant-Male `[一作]` (Tampere University), Roel Pieters `[通讯]` (Tampere University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文系统比较了基于人体动作识别（HAR）输入的装配状态追踪方法，旨在提高人机协作中的任务同步与效率。

**💡 创新点**

创新点在于对多种追踪方法（逻辑、概率增强、HMM、任务级 LSTM 与动作级 LSTM）进行统一实验评估，并强调动作持续时间建模与多样化噪声处理的重要性。

**🔧 技术方法**

采用的技术包括基于规则的逻辑推理、概率时间阈值、隐藏马尔可夫模型、任务专用长短时记忆网络以及通用动作预测 LSTM，结合 ST‑GCN skeleton‑based HAR 模型。

**📊 数据集**

实验使用公开的 HA4M 与 IKEA ASM 组装数据集，分别涵盖复杂的重复动作和自然变异的组装流程。

**📈 对比分析**

在不同噪声水平和真实 HAR 输入下，逻辑与概率增强方法在 IKEA 任务上表现最好，而任务级 LSTM 在 HA4M 任务上取得最高 F1；HMM 在无噪声时可竞争，但在噪声增大时迅速衰退。

**⚠️ 局限性**

局限性包括仅适用于线性无分支任务、对预定义动作列表和统计参数高度依赖、对高噪声与动作顺序变异的鲁棒性不足，并未充分利用对象状态等多模态信息。

---

## 428. Spatially Robust Near-Field SWIPT Using Pinching Antennas: Rate-Energy Tradeoff Bounds

**arXiv ID:** 2606.20133 | [PDF](https://arxiv.org/pdf/2606.20133v1)

**作者:** Zoran Hadzi-Velkov `[一作]` (Ss. Cyril and Methodius University), Arumugam Nallanathan `[通讯]` (Queen Mary University of London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了基于可选择Pinching Waveguide Antennas（PWA）的近场SWIPT空间鲁棒设计，利用服务区（SA）覆盖用户位置并通过离散天线激活实现信息与能量的同时传输。

**💡 创新点**

创新点包括：①以服务区为优化目标，消除传统点基设计对位置误差的敏感性；②将离散天线选择建模为二元二次规划并利用半正定松弛（SDR）获得理论上界；③使用Nesterov平滑与低秩近似加速对偶求解；④设计低复杂度交换式局部搜索实现可行整数解。

**🔧 技术方法**

主要技术手段有：半正定松弛（SDR）、Jensen不等式、Nesterov平滑求解、低秩近似、交换式局部搜索、服务区协方差矩阵构造。

**📊 数据集**

实验使用仿真参数：28 GHz、6 m波导、Δ=λ、M=560、N=40、6 × 6 m工作区，无使用真实数据集。

**📈 对比分析**

通过与传统点基时间共享基准对比，鲁棒设计在R–E曲线上提升显著，能量获取提高至微瓦级；与SDR上界对比，整数解与上界相差约12%；同时在服务区内实现稳定速率，避免了点基方案的频率波动。

**⚠️ 局限性**

限制：仅考虑单用户单服务区场景，扩展到多用户、多服务区需进一步研究；交换式搜索只能保证局部最优；近场波导模型假设无耦合，实际系统可能受额外互调、损耗影响。

---

## 429. ScaffoldAgent: Utility-Guided Dynamic Outline Optimization for Open-Ended Deep Research

**arXiv ID:** 2606.20122 | [PDF](https://arxiv.org/pdf/2606.20122v1)

**作者:** Zhibang Yang `[一作]` (National Engineering Research Center of Software Engineering, Peking University), Yasha Wang `[通讯]` (Peking University Information Technology Institute (Tianjin Binhai))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ScaffoldAgent 框架，用动态轮廓优化实现开放式深度研究（OEDR），通过多轮检索、证据整理与报告生成协同工作。

**💡 创新点**

创新点包括：① 用 Expansion、Contraction、Revision 三种显式结构操作来控制轮廓演化；② 通过结合检索收益、结构连贯性和试写质量的 utility 指标实现即时反馈；③ 基于 UCB 的节点选择与终止策略，使优化过程具备可控性与鲁棒性。

**🔧 技术方法**

主要技术包括：大型语言模型驱动的搜索、轮廓与报告三代理系统；结构决策过程与 UCB 采样；检索相关性、冗余度、结构平衡、引用支持等多维度 utility 评估；以及试写（trial generation）作为评估工具。

**📊 数据集**

使用公开的 DeepResearch Bench 和 DeepResearch Gym 两大基准数据集进行实验，评估报告质量（RACE、Gym 维度）与事实可靠性（FACT）等指标。

**📈 对比分析**

与 Naïve、Single‑Agent（ReAct、IRCoT、WebShaper）和 Multi‑Agent（STORM、WebWeaver、EDR、StackPlanner）等基线对比，ScaffoldAgent 在 RACE Overall、FACT（Eff.c. 与 C.acc.）以及 Gym 多维度得分上均超越所有对手，尤其在生成质量与事实核查方面提升显著。

**⚠️ 局限性**

局限性包括：① 多轮交互评估仍处于初级阶段，缺乏长时序对话数据；② 只在开源 LLM 与固定检索接口下验证，未探索更强大模型或高级搜索系统的组合效果；③ 目前未学习轮廓演化策略，完全基于手工设计的 utility 指标，缺少从经验中自适应学习的能力。

---

## 430. When Does Streaming Tool Use Help? Characterizing Tool-Intent Stabilization in Streaming Retrieval-Augmented Generation

**arXiv ID:** 2606.20113 | [PDF](https://arxiv.org/pdf/2606.20113v1)

**作者:** Elroy Galbraith `[一作]` `[通讯]` (SMG Labs), Elroy Galbraith (SMG Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在流式检索增强生成（Streaming RAG）中，工具查询何时能够提前触发并减少用户感知延迟，提出并量化了查询内的“工具意图稳定化”概念。

**💡 创新点**

创新点在于：① 给出了模型无关、训练不依赖的工具意图稳定化测度；② 推导了隐藏延迟上限 H，作为部署时可预测的工具延迟隐藏阈值；③ 通过实验证明此上限在实际流水线中是保守且可达成的。

**🔧 技术方法**

使用了 BM25 词检索、词级输入流模型、Trigger‑Reflector 机制，以及工具延迟与输入节奏的参数化分析。

**📊 数据集**

采用 CRAG（Comprehensive RAG Benchmark）Task 1 & 2 的验证集进行实验，包含约 1.5k 个问答样本。

**📈 对比分析**

与传统非流式 RAG 进行对比，测得在 L=600 ms、δ=3 w/s、θ=0.8 的配置下，约 73 % 的查询可隐藏至少 80 % 的工具延迟；整体实测延迟收益与 H 预测上限相符甚至略高，验证了理论上限的保守性。

**⚠️ 局限性**

局限性包括：① 仅在均匀词流假设下评估，未考虑实际 ASR 变速与误差；② 稳定化测度依赖词前缀，可能无法捕捉更细粒度的语义转换；③ 只使用 BM25 的稀疏检索，忽略了密集检索的行为；④ 仅针对可字面匹配的答案进行评估，导致查询样本偏向可检索子集。

---

## 431. SAM3 Self-Distillation for Fine-Grained GOOSE 2D Semantic Segmentation

**arXiv ID:** 2606.20130 | [PDF](https://arxiv.org/pdf/2606.20130v1)

**作者:** Xuesong Wang `[一作]` `[通讯]` (Wayne State University), Xuesong Wang (Wayne State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于SAM3图像编码器与轻量化解码器的地面机器人2D细粒度语义分割模型。

**💡 创新点**

创新点在于将SAM3本身作为oracle-box自蒸馏教师，并引入图像级多尺度测试时增强以恢复单输入尺寸模型的多尺度推理优势。

**🔧 技术方法**

主要技术包括SAM3预训练编码器、FPN式轻量解码器、自蒸馏、激进的颜色扰动（photometric distortion）、水平翻转与图像级多尺度TTA。

**📊 数据集**

使用GOOSE 2D与GOOSE-Ex 2D训练集（11,234张RGB图像）、验证集1,369张以及官方1,815张测试集。

**📈 对比分析**

在官方测试集上实现综合mIoU 69.73%（细粒度56类68.0%，粗粒度11类63.5%），排名第4。

**⚠️ 局限性**

局限性包括对稀有类（如moss）识别仍低，对颜色变化的敏感度未完全消除，以及对某些细粒度类别的性能仍依赖模型集成。

---

## 432. HilDA: Hierarchical Distillation with Diffusion for Advancing Self-Supervised LiDAR Pre-trainin

**arXiv ID:** 2606.20189 | [PDF](https://arxiv.org/pdf/2606.20189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 433. HEad and neCK TumOR (HECKTOR) 2025: Benchmark of Segmentation, Diagnosis, and Prognosis in Multimodal PET/CT

**arXiv ID:** 2606.20143 | [PDF](https://arxiv.org/pdf/2606.20143v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 434. FrozenDrive: Zero-Shot Text-Guided Driving Scene Generation and Data Augmentation with Parameter-Free Frozen Diffusion Model

**arXiv ID:** 2606.20110 | [PDF](https://arxiv.org/pdf/2606.20110v1)

**作者:** Yuhwan Jeong `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种零样本文本驱动的驾驶场景生成框架 FrozenDrive，利用冻结的 Stable Diffusion 并通过知识保持的时空注意力实现多视角和时间一致性

**💡 创新点**

创新点在于在不引入任何可训练参数的情况下，仅通过重新构造注意力输入实现多视角、时间一致性，同时加入稀有类别平衡损失提升少数类生成质量

**🔧 技术方法**

采用 ControlNet 条件嵌入、知识保持的多视角膨胀自注意力、时间参考自注意力以及对象比率损失，全部基于冻结的 Stable Diffusion

**📊 数据集**

使用 nuScenes 数据集进行训练与评估，包含多视角图像、HD 地图、深度、相机姿态等信息

**📈 对比分析**

与 MagicDrive、Panacea、DrivingSphere、DiST-4D、DriveArena、MagicDrive‑V2、X‑Scene 等基线比较，FrozenDrive 在 FVD、BEV 语义分割 mIoU、3D 检测 mAP 等指标上取得最优或接近最优，并在夜间与雨天的 AD 模型数据增强实验中显著提升感知与规划性能

**⚠️ 局限性**

限制在于长程时间一致性仍不及专用视频扩散模型，且仅验证在 Stable Diffusion 之上，未来需探索更强背骨与更全面的数据增强评估

---

## 435. MedRLM: Recursive Multimodal Health Intelligence for Long-Context Clinical Reasoning, Sensor-Guided Screening, Evidence-Grounded Decision Support, and Community-to-Tertiary Referral Optimization

**arXiv ID:** 2606.20164 | [PDF](https://arxiv.org/pdf/2606.20164v1)

**作者:** Aueaphum Aueawatthanaphisut `[一作]` `[通讯]` (Thammasat University), Aueaphum Aueawatthanaphisut (Thammasat University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 MedRLM 递归多模态健康智能框架，解决长上下文临床推理、传感器引导筛查和社区到三级转诊支持问题。

**💡 创新点**

创新点包括递归临床分解、可审计的临床证据图记忆、传感器驱动的递归触发以及不确定性门控的转诊优化。

**🔧 技术方法**

采用递归语言模型、检索增强生成、模态专用代理、图结构记忆、传感器流处理与 TinyML 边缘推理技术。

**📊 数据集**

使用 MIMIC‑IV、eICU‑CRD、MIMIC‑CXR‑JPG、CheXpert、PTB‑XL、PhysioNet/CinC 2012 等真实公开或认证数据集。

**📈 对比分析**

对比方法通过在上述数据集上进行多任务评估（AUROC、AUPRC、宏 F1、决策曲线净收益等），但目前仅给出公开基准锚点，未公布 MedRLM 的最终性能。

**⚠️ 局限性**

局限性包括缺乏社区到三级转诊标签、对多模态数据同步与融合的依赖、潜在的生成幻觉风险以及对医生人工复核的高需求。

---

## 436. N-Version Programming with Coding Agents

**arXiv ID:** 2606.20158 | [PDF](https://arxiv.org/pdf/2606.20158v1)

**作者:** Javier Ron `[一作]` (KTH Royal Institute of Technology), Martin Monperrus `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文重新用现代AI编码代理复现并扩展了1986年的N版本编程实验，生成48个通过验收的实现版本，并对其失败相关性与多版本投票可靠性进行系统评估。

**💡 创新点**

创新点在于：①将自然语言规范与多种AI代理、模型与编程语言组合的多样化作为实验因素；②使用大规模（100万）随机测试与统计检验（z统计、ϕ相关）验证失败相关性；③揭示失败集中的规格弱点，并证明即使存在相关性，多版本投票仍能显著提升可靠性。

**🔧 技术方法**

采用的技术包括：AI编码代理（Cursor、Claude Code、Codex、Gemini、OpenCode）、Python/ Rust/ Pascal实现、差分测试与Python参考实现、1M随机输入生成、z统计与ϕ相关性分析、三版本多数投票模拟与根因分析。

**📊 数据集**

使用的数据集为NASA的Launch Interceptor Program（LIP）规范与对应的15个示例、200个验收测试和100万随机测试样本；所有实现均对照Python参考实现进行差分检测。

**📈 对比分析**

比较方法：将实际的共发失败次数与独立伯努利模型预测的期望进行z检验；对各语言和代理组合进行pairwise ϕ相关分析；构造所有三版本单元，计算其投票失败率与单版本平均失败率的差异。实验结果显示，单版本平均失败率从387.44降至130.99，68%三版本单元无失败，证明多版本投票在存在相关性的情况下仍能显著提升可靠性。

**⚠️ 局限性**

局限性包括：仅测试单一LIP规范，未检验长期状态服务或外部依赖；缺乏对LLM采样温度或复现性变化的评估；样本仅为一次生成结果，未探究同一配置下多次采样的内部多样性；因此结论可能对其他领域或不同规范的推广存在限制。

---

## 437. AgenticDB: Agentic Performance Reconfiguration for Database Workloads

**arXiv ID:** 2606.20318 | [PDF](https://arxiv.org/pdf/2606.20318v1)

**作者:** Xinyue Yang `[一作]` (University of Chinese Academy of Sciences), Yanjun Wu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了AgenticDB，一个基于LLM的代理框架，用于数据库工作负载的闭环重配置。

**💡 创新点**

创新点包括：①将运行时反馈作为瓶颈诊断依据的上下文驱动交互循环；②构建跨层安全动作空间，支持DBMS与OS级重配置；③实现验证器、恢复器和审计器，保障执行安全；④利用会话与经验记忆提升规划效率。

**🔧 技术方法**

使用技术主要有：LLM驱动的规划（如GPT‑5.5等）、验证与执行生命周期、状态监控与诊断、记忆增强的规划、以及针对MySQL/PostgreSQL的工具链。

**📊 数据集**

实验数据集包括：YCSB、Sysbench（Read/Write/ReadWrite）、TPC‑H；以及 MySQL 8.0.45 与 PostgreSQL 16.13 数据库实例。

**📈 对比分析**

在相同初始配置、相同预算下与 GPTuner、DB‑BERT、AgentTune、CDBTune 四个基线比较；AgenticDB 在所有工作负载上取得最优结果，平均提升 118.1%，平均时间‑到‑最佳缩短 22.6%，写密集工作负载提升最高可达 337.7%。

**⚠️ 局限性**

局限性包括：受 LLM 调用成本限制；需手工制定安全规则与参数兼容性检测；未在极大规模/多租户环境下评估；对某些 OS 参数的支持不完整，且实验环境（CPU/SSD/内存）对结果迁移性有限。

---

## 438. Confidence-Aware Automated Assessment of Student-Drawn Scientific Models

**arXiv ID:** 2606.20264 | [PDF](https://arxiv.org/pdf/2606.20264v1)

**作者:** Luyang Fang `[一作]` (AI4STEM Education Center), Xiaoming Zhai `[通讯]` (AI4STEM Education Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于视觉Transformer的自适应模型，对学生绘制的科学模型图像进行自动评分，并加入置信度估计以实现选择性自动评分。

**💡 创新点**

创新点在于：①通过低秩适配（LoRA）实现高效的任务迁移；②利用测试时的语义保持扰动生成预测分布，并从中提取响应级置信度；③在置信度阈值下实现自动化与人工审核的动态平衡；④引入选择性信任机制（selective trust）对扰动预测进行过滤，进一步提升评分鲁棒性。

**🔧 技术方法**

技术包括Vision Transformer (ViT) 预训练模型、LoRA 参数高效微调、测试时扰动（crop/rotate）生成预测分布、置信度计算与阈值筛选、Top‑η 选择性过滤。

**📊 数据集**

使用了美国东北部中学的六个 NGSS 对齐科学建模测评项目的学生绘图数据集，共计约 4.8 万张图片，按三级熟练度（Beginning、Developing、Proficient）标注。

**📈 对比分析**

与四种基线（冻结 ViT、LoRA 微调、CA‑Uniform、CA‑Selective）比较，CA‑Selective 在准确率、Kappa、F1 等指标上平均提升约 0.4–1.5%（Kappa 最高提升至 0.760），并展示出置信度与准确率显著正相关，证明选择性自动评分可在保证可靠性的同时扩大自动覆盖率。

**⚠️ 局限性**

局限性包括：①数据仅来自单一地区中学，代表性有限；②依赖人工专家评分作为黄金标准，可能带来评分偏差；③置信度阈值需人工设定，适配不同任务仍需经验；④测试时扰动与过滤策略对不同绘图风格的鲁棒性尚需进一步验证。

---

## 439. Thermodynamic Measure of Intelligence

**arXiv ID:** 2606.20231 | [PDF](https://arxiv.org/pdf/2606.20231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 440. Through the PRISM: Preference Representation in Intermediate States of Video Diffusion Models

**arXiv ID:** 2606.20310 | [PDF](https://arxiv.org/pdf/2606.20310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 441. Editorial Alignment: A Participatory Approach to Engaging Editorial Expertise in LLM-mediated Knowledge Dissemination

**arXiv ID:** 2606.20258 | [PDF](https://arxiv.org/pdf/2606.20258v1)

**作者:** Simon Aagaard Enni `[一作]` (Aarhus University), Kristoffer Laigaard Nielbo `[通讯]` (Aarhus University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在论文中，作者通过与北欧在线百科编辑团队的两轮协作工作坊，构建并实施了一个以编辑标准为核心的 LLM 辅助接口，旨在将编辑专业知识嵌入 LLM 的输出，从而保持机构的知识可信度与编辑权威；

**💡 创新点**

创新点在于将 AI 对齐重新定义为一种基于实践的设计活动——编辑对齐（Editorial Alignment），并将编辑团队的隐性专业标准通过工作坊方法转化为可操作的对齐目标，形成可持续的编辑标准文档；

**🔧 技术方法**

主要技术手段包括：1) 设计参与式工作坊（Future Workshop 与编辑标准工作坊）；2) 通过上下文工程与受限生成（prompt engineering、content filtering）实现对齐目标；3) 利用 LLM（Claude Sonnet 4.6）生成示例文本以激发编辑边界意识；

**📊 数据集**

研究未使用公开数据集，而是依赖：① 本项目原型的用户交互日志（约20条问答对）；② 编辑团队事先自选的3条示例文本；③ 研究者自生成的4条带有不同语体的 LLM 输出；

**📈 对比分析**

论文未提供定量性能指标或与其他方法的对比；评估主要以定性方式进行，关注编辑标准形成过程、工作坊产出以及对系统输出结构的影响；

**⚠️ 局限性**

局限性包括：① 仍存在“数据采集”式参与模式，编辑虽能共创标准但缺乏对最终系统的决策权；② 标准的可执行性与技术实现仍待进一步验证，缺乏对齐效果的实证评估；③ 仅涉及单一北欧百科机构，外推性受限；④ 论文缺乏对 LLM 生成质量与编辑标准满足度的量化度量。

---

## 442. TrustMix: How to Mix Messages in a Mobile Ad-hoc Network

**arXiv ID:** 2606.20251 | [PDF](https://arxiv.org/pdf/2606.20251v1)

**作者:** Yu Shen `[一作]` (RPTU University Kaiserslautern-Landau), Stefanie Roos `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种面向移动自组织网络（MANET）的匿名通信协议 TrustMix，利用信任机制与混淆网络实现消息匿名传输，并实现了 Android 原型实验。

**💡 创新点**

创新点在于：① 在没有中心可信方的情况下通过匿名可信方发现协议实现本地信任转发；② 结合多成员重加密混合技术，降低对成员可靠性的要求；③ 采用可链接环签名实现速率限制，同时不泄露身份；④ 通过零知识证明保证重加密过程安全。

**🔧 技术方法**

使用技术包括：ElGamal 加密、可链接环签名、零知识证明、可验证混洗（Groth 栈）、安全分布式密钥生成（Gennaro 等）、匿名可信方发现协议（类似 Anix）。

**📊 数据集**

数据集主要是基于 Android 设备（5 台手机）进行实验，以及模拟 100 个群组、每组每秒 8 条消息的混合网络仿真，评估匿名熵和吞吐量。

**📈 对比分析**

方法对比：与 Atom 混合网络比较，TrustMix 在混洗时间上约为 Atom 的 45%，但对恶意节点更鲁棒；吞吐量约 3.6 条/秒；匿名熵随池大小、混合跳数增加而提升，但在高比例敌对节点时线性下降。

**⚠️ 局限性**

局限性：① 计算与通信开销较大，吞吐量低，难以支持高频率消息；② 仅在小规模实验验证，未评估大规模群组与多跳网络的可扩展性；③ 受限于群组成员可靠性假设，若多成员受控会降低安全性；④ 需要前置发现信任关系，增加部署复杂度。

---

## 443. CMDS-AD: Cross-Modal Dual-Stream Decoupling for Few-Shot Anomaly Detection

**arXiv ID:** 2606.20300 | [PDF](https://arxiv.org/pdf/2606.20300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 444. Towards 3D karst underwater scene reconstruction from rotating sonar data

**arXiv ID:** 2606.20322 | [PDF](https://arxiv.org/pdf/2606.20322v1)

**作者:** Georgios Evangelos Margaritis `[一作]` (Institut Polytechnique de Paris), François Goulette `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一套从稀疏、噪声较大的声纳数据到三维可交互网格的完整流程，先利用连续时间 SLAM 校正漂移，再采用两阶段深度学习实现表面重建。

**💡 创新点**

创新点在于把连续时间 SLAM 与针对稀疏点云的两阶段学习模型（神经泊松重建 + 点卷积重建）结合，实现了对地下石灰岩通道的高保真重建。

**🔧 技术方法**

核心技术包括 6DoF 连续时间 SLAM、Neural Poisson Surface Reconstruction (nPSR)、Point Convolution for Surface Reconstruction (POCO) 以及 Blender 的体素重建做后处理。

**📊 数据集**

使用了 2023 年 3 月 NavScoot2 机器人在法国 Nîmes 石灰岩地下通道采集的旋转声纳点云（约 5 万点）作为实验数据集。

**📈 对比分析**

通过将 nPSR 产生的下采样点云和 POCO 的原始点云进行对比，发现 nPSR 在缺口、断层最小化方面效果更佳，整体网格更光滑、连续，性能优于传统重建方法。

**⚠️ 局限性**

主要局限是声纳数据稀疏与噪声导致细节缺失，学习模型对新环境的泛化性有限，且当前仍需人类潜水员操作，未实现完全自主采集。

---

## 445. GEN-Guard: Correcting Generalization Failures for Deployable Federated Surgical AI

**arXiv ID:** 2606.20303 | [PDF](https://arxiv.org/pdf/2606.20303v1)

**作者:** Julia Alekseenko `[一作]` (University of Strasbourg), Nicolas Padoy `[通讯]` (University of Strasbourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出GEN-Guard框架，对联邦学习模型进行后期评估与纠正，以提升跨机构的泛化能力。

**💡 创新点**

创新点在于引入Client-Blocked Evaluation检测模型选择失败，并通过Disagreement-Aware Distillation纠正跨机构偏差，实现零射程适配。

**🔧 技术方法**

采用联邦学习（FedAvg、FedProx、SCAFFOLD）训练后，使用CBE与DAD两种技术进行后处理。

**📊 数据集**

使用多中心外科视频数据集：Multi-Cholec（胆囊切除阶段识别）与PolypGen（结肠息肉分割）。

**📈 对比分析**

与传统联邦模型和个性化方法相比，GEN-Guard平均F1/Dice提升1–3点，最差情况提升3–9点，同时保持实时推理速度。

**⚠️ 局限性**

局限在于CBE对验证集代表性敏感，且在极端分布差异时仍可能漏检或误检模型选择失败。

---

## 446. Cinematic Compositing Using Character-Environment-Harmonized Video Generation Models

**arXiv ID:** 2606.20233 | [PDF](https://arxiv.org/pdf/2606.20233v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 447. Co-VLA: Coordination-Aware Structured Action Modeling for Dual-Arm Vision-Language-Action Systems

**arXiv ID:** 2606.20285 | [PDF](https://arxiv.org/pdf/2606.20285v1)

**作者:** Yandong Wang `[一作]` (Donghua University), Chao Zhang `[通讯]` (Samsung R&D Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了Co‑VLA框架，将视觉‑语言‑动作模型改造为包含共享‑残差分解的结构化动作专家SAE以及在部署时实时调节同步、精度与安全的隐知控制器LAC，实现双臂协作的显式结构化与执行优化。

**💡 创新点**

通过显式的共享‑残差分解和任务自适应协调损失，首次将协作结构从动作输出中抽象出来；并在部署时使用能量比、对抗度量与刚度调节的隐知控制器，实现动态同步与安全约束；此外提出Co‑Motion并行演示方式，显著提升并行协作数据量。

**🔧 技术方法**

基于PaliGemma视觉‑语言骨干的π_0模型，构建SAE结构化动作头，配合ℓ₁残差正则、共享均值一致性、时间同步损失等辅助项；部署时引入能量比、对抗度量、适应性刚度的隐知控制器；Co‑Motion采用共享参考框、前瞻预计算与安全中间目标实现并行调度。

**📊 数据集**

主要使用RoboTwin 2.0仿真双臂任务数据（超过50项任务）及其Co‑Motion并行演示，真实世界在AgileX Cobot Magic双臂机器人上收集50次人类演示并进行ID/OOD随机化测试。

**📈 对比分析**

在RoboTwin仿真上与π_0及π_0.5对比，Co‑VLA平均成功率从76%/73%提升至82%，Handover任务从64%提升至91%；在真实世界ID/OOD评估中，Co‑VLA+LAC在五个任务中取得最高成功率（如ID场景从43%提升至67%，OOB从13%提升至27%），显著优于基线。

**⚠️ 局限性**

Co‑Motion并行演示在提升数据效率的同时导致学习难度上升，Co‑VLA在Co‑Motion数据上的成功率下降；辅助损失需手工选择，缺乏自动化；SAE与LAC尚未在扩散策略等其他连续动作框架中验证；对极端时序耦合的泛化能力仍有限。

---

## 448. U$^2$Mamba: A Two-level Nested U-structure Mamba for Salient Object Detection

**arXiv ID:** 2606.20282 | [PDF](https://arxiv.org/pdf/2606.20282v1)

**作者:** Junhui Li `[一作]` (University of Science and Technology Liaoning), Youshan Zhang `[通讯]` (Chuzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种基于Mamba状态空间模型的嵌套U结构网络U^2Mamba，用于单模态显著目标检测。

**💡 创新点**

创新点包括：将Mamba模型嵌入多尺度U块MMUB以提升局部与长距离上下文建模；构建双层嵌套U结构；采用层级监督（BCE+KL）代替传统深度监督。

**🔧 技术方法**

使用了Mamba状态空间模型、Multi-scale Mamba U-Block (MMUB)、U-Net风格编码解码架构以及层级监督损失（BCE+KL）等技术。

**📊 数据集**

训练使用DUTS-TR数据集，评测集包括ECSSD、PASCAL‑S、DUTOMRON、HKU‑IS和DUTS。

**📈 对比分析**

与10种SOTA方法（如U^2Net、VST等）在MAE、maxFβ指标上对比，U^2Mamba在大多数数据集上取得最高或相近性能，MAE最低且FPS更高。

**⚠️ 局限性**

局限性在于对背景与前景色彩或纹理相近的场景仍易误判，且仅支持单模态RGB，缺乏多模态或边缘引导等进一步提升机制。

---

## 449. Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think

**arXiv ID:** 2606.20246 | [PDF](https://arxiv.org/pdf/2606.20246v1)

**作者:** Gia-Binh Nguyen `[一作]` (VinUniversity), Ngo Anh Vien `[通讯]` (VinUniversity)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在大规模预训练的视觉-语言-动作模型（VLA）中，通过一次前向推理利用Centered Kernel Alignment (CKA) 识别并删除冗余的Transformer层，构建更浅、更高效的模型，并在下游任务上进行微调。

**💡 创新点**

创新点在于：① 采用无训练的CKA指标实现全模型层级结构压缩；② 通过提前剪枝减少微调成本与推理延迟；③ 证明层级冗余在现代连续控制VLA中普遍存在，并可通过结构压缩提升低数据场景的鲁棒性。

**🔧 技术方法**

主要技术包括：CKA层相似度度量、聚类相似区块、基于阈值的层剔除、静态剪枝后微调；实验中使用多种VLA基线（π₀、GR00T‑N1.5、SmolVLA）和多任务控制头。

**📊 数据集**

数据集涵盖三大仿真基准（LIBERO、RoboCasa、SimplerEnv）以及10个真实世界抓取/搬运任务，测试机器人包括UR10、UR5、Aloha单臂与双臂。

**📈 对比分析**

与现有无训练剪枝、动态路由以及训练适应方法对比，CLP在保持甚至提升任务成功率的同时，训练时间缩短1.4–1.5倍，推理速度提升约30%，并在低样本（10%数据）情形下显著优于MoLe‑VLA。

**⚠️ 局限性**

局限性：1）使用全局相似度阈值，未针对不同模态（视觉、语言、动作）细粒度剪枝；2）仅在预训练后进行剪枝，未探索在预训练阶段的自适应层选策略。

---

## 450. Actionable Activation Directions for Detecting and Mitigating Emergent Misalignment Across Language Model Families

**arXiv ID:** 2606.20225 | [PDF](https://arxiv.org/pdf/2606.20225v1)

**作者:** Abdul Rafay Syed `[一作]` `[通讯]` (Universität des Saarlandes), Abdul Rafay Syed (Universität des Saarlandes)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在四种不同架构的大模型上，使用不安全代码微调导致的不可预期行为的内部几何结构，并验证了该结构的可测量性、可因果性以及跨架构可迁移性。

**💡 创新点**

提出跨架构误差几何转移的系统实验，发现两级特异性结构；揭示几何捐献者-接受者非对称拓扑；首次将安全代码负控制加入评估；证明仅线性映射无法实现跨架构特异性控制。

**🔧 技术方法**

差值均值方向提取、残差流激活钩子实现因果 Steering、岭回归映射跨模型、随机/正交对照、代码泄漏评估指标。

**📊 数据集**

约6000条不安全代码样本进行微调，115条非编码提示用于方向提取与评估，12条离线提示用于泛化测试，以及与之对应的安全代码负控制微调集。

**📈 对比分析**

使用线性可分度（99.6%）和效应量评估，对比基线与 Steering、跨模型映射，发现内模型 Steering 可降低21–51点代码泄漏，跨模型映射可降低13–46点但缺乏特异性；安全控制显示几何仅因训练内容产生。

**⚠️ 局限性**

实验仅在1–3B参数模型上，单一不安全代码域，使用仅激活 Steering 的干预方法，离线测试样本有限，未验证更大规模或不同安全域的适用性。

---

## 451. Augmenting Game AI with Deep Reinforcement Learning

**arXiv ID:** 2606.20210 | [PDF](https://arxiv.org/pdf/2606.20210v1)

**作者:** Alessandro Sestini `[一作]` (Electronic Arts), Linus Gisslén `[通讯]` (Electronic Arts)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文探讨了如何通过深度强化学习增强游戏AI，以提高游戏角色的可信度和沉浸感，并提出了一个适用于游戏开发的强化学习模型训练框架。

**💡 创新点**

创新点在于提出了一个针对游戏AI的强化学习训练框架，强调了短训练时间、可控性、模块化、可维护性等要求，并展示了如何在实际游戏中应用这些技术。

**🔧 技术方法**

使用了深度强化学习技术，特别是Soft Actor-Critic (SAC)和Proximal Policy Optimization (PPO)算法。

**📊 数据集**

使用了EA SPORTS FC 25和Battlefield 6这两个流行的AAA游戏作为数据集，展示了在这些游戏中应用强化学习的挑战和效果。

**📈 对比分析**

通过与传统手工编码的AI系统进行比较，强化学习增强的AI在可信度和表现上有显著提升。例如，在EA SPORTS FC 25中，新的守门员AI系统的扑救率提高了10%。

**⚠️ 局限性**

限制在于当前的研究仍然面临许多挑战，如训练时间长、模型的可维护性和可控性不足，以及在复杂游戏环境中应用强化学习的困难。

---

## 452. Leveraging systems' non-linearity to tackle the scarcity of data in the design of Intelligent Fault Diagnosis Systems

**arXiv ID:** 2606.20323 | [PDF](https://arxiv.org/pdf/2606.20323v1)

**作者:** Giancarlo Santamato `[一作]` (Scuola Superiore Sant'Anna), Antonio Frisoli `[通讯]` (Scuola Superiore Sant'Anna)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种利用结构非线性在多激励水平下获取频率响应函数(FRF)并将其可视化为图像的技术，随后通过自定义数据增强方法生成大量训练样本，用预训练的MobileNetV2进行迁移学习，构建铁路 Pantograph 的智能故障诊断系统，验证了该系统在无损、螺栓断裂和阻尼器失效三种情形下的识别效果。

**💡 创新点**

创新点在于：①将激励幅度作为额外维度，将不同激励水平的 FRF 叠加成“色谱图”，利用系统的非线性行为提供额外信息；②通过列交换等简单变换对 FRF 数据进行数据增强，避免需要大规模标注或生成模型；③直接将生成的图像喂入 ImageNet 预训练 CNN，充分利用迁移学习的优势，解决数据稀缺问题。

**🔧 技术方法**

技术包括：频率响应函数估计(FFT、p‑Welch)、激励水平控制与测量、图像可视化、列交换增强、混合增强(Mixup、Cutmix 等)、MobileNetV2 迁移学习、全连接分类器、Adam 优化器、精细调参。

**📊 数据集**

数据集：铁路 Pantograph 在三种状态（无损、螺栓失效、阻尼器失效）下，7 个激励水平，每个水平 6 次重复，使用列交换产生 3^7=2187 张图像，每种状态共 6561 张图像（总计 6561 张）。

**📈 对比分析**

采用 20 轮预训练 + 10 轮微调的训练策略，使用 Adam 优化器；在测试集上取得 97.6% 的分类准确率，混淆矩阵显示主要误分类为无损状态的螺栓失效。与传统单激励或纯频谱方法相比，本文方法在数据稀缺的条件下实现了更高的识别性能。

**⚠️ 局限性**

局限性包括：①对激励非线性假设要求较高，若系统近线性则效果减弱；②对螺栓失效的识别仍有误差，需进一步提升敏感度；③仅在三种故障情形下验证，缺乏对多种复杂损伤的泛化能力；④实验设置依赖可控激励源，实际工况中可能受环境噪声影响。

---

## 453. CUPID: Reconstructing UV Texture Maps for Interpretable Person-of-Interest Deepfake Detection

**arXiv ID:** 2606.20302 | [PDF](https://arxiv.org/pdf/2606.20302v1)

**作者:** Giovanni Affatato `[一作]` (Politecnico di Milano), Stefano Tubaro `[通讯]` (Politecnico di Milano)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一个名为Cupid的面向特定人物（POI）的深度伪造视频检测器，利用UV纹理映射和掩码自编码器（MAE）学习通用人脸身份表征，并在测试阶段通过与POI参考视频的对比来判别视频真伪。

**💡 创新点**

创新点包括：①不在训练阶段使用任何深度伪造视频或POI数据，保持POI无关性；②通过UV纹理映射统一不同身份、不同帧的面部区域，实现结构化输入；③利用MAE的自监督重建目标学习身份一致的稠密表征；④提出基于UV残差图的可解释性方法，能直观显示判别时关注的面部区域。

**🔧 技术方法**

核心技术：3D人脸重建产生UV纹理图；掩码自编码器（MAE）用于学习身份表征；对比度学习或嵌入相似度判定；UV残差可视化实现解释性。

**📊 数据集**

实验数据集：四个公开深度伪造数据集（如FaceForensics++、Celeb-DF、UADFV 等），同时在高质量与低质量（强压缩、缩放）场景下进行评估。

**📈 对比分析**

与多种现有POI特定和通用深度伪造检测器（如ID-Reveal、poi-forensics等）对比，Cupid在大多数数据集上取得最高准确率，表现出最优的鲁棒性（对压缩、缩放耐受性最佳）和最快的推理速度；阈值稳定性也优于基线模型。

**⚠️ 局限性**

局限性：①需要可靠的3D重建和UV映射，对遮挡、极端姿态或低质量输入可能受限；②仅处理视觉域，未考虑音频信息；③仍需一定数量的POI参考视频，若参考视频不足可能影响性能；④对某些新型生成模型的攻击可能仍存在一定风险。

---

## 454. ELVA: Exploring Ranking-Driven Universal Multimodal Retrieval

**arXiv ID:** 2606.20280 | [PDF](https://arxiv.org/pdf/2606.20280v1)

**作者:** Yuhan Liu `[一作]` (Xi'an Jiaotong University), Jingmin Xin `[通讯]` (Xi'an Jiaotong University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对多模态检索中的粒度盲点，提出ELVA框架，通过排名驱动的强化学习和可验证奖励来提升粒度信息的捕获和检索性能。

**💡 创新点**

创新点在于：①使用基于规则的可验证奖励（排名奖励和间隔奖励）实现无标签排名学习；②结合GRPO进行探索式强化学习；③设计平衡负样本采样与排名优化策略；④引入MRBench多粒度检索基准。

**🔧 技术方法**

核心技术包括：多模态大型语言模型（MLLM）、对比学习、强化学习（RLVR+GRPO）、连续排名奖励、间隔奖励、负样本平衡采样、生成式嵌入提取。

**📊 数据集**

使用的数据集有：M-BEIR（8个检索子任务）、MRBench（从M-BEIR筛选的多粒度查询）、Share4V、Urban、VisD、MSR-VTT、MSVD等未见数据集。

**📈 对比分析**

与现有方法（如UniIR、LamRA、MM-Embed、Vision-R1、Qwen2-VL等）进行对比，ELVA在M-BEIR上平均Recall@K提升约4%（7B版+3.9%），在MRBench提升13.1%，在零样本视频检索上分别比InternVideo提升约16.5%和8.4%，并在多种未见任务中保持领先或竞争优势。

**⚠️ 局限性**

主要限制是高推理成本（MLLM规模大），可通过特征预计算、层剪枝或使用轻量版ELVA-2B等方法进一步优化；同时需改进检索-重排序集成以降低管道复杂度。

---

## 455. Efficiently Linking Real Scenes with Synthetic Data Generation for AI-based Cognitive Robotics and Computer Vision Applications

**arXiv ID:** 2606.20272 | [PDF](https://arxiv.org/pdf/2606.20272v1)

**作者:** Paul Koch `[一作]` (Fraunhofer IPK), Jörg Krüger `[通讯]` (Fraunhofer IPK)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个闭环流程，将真实场景扫描、仿真生成、数据标注与AI模型训练相结合，以实现高质量、可扩展的机器人视觉训练数据。

**💡 创新点**

创新点在于将扫描、仿真与AI训练循环化，并利用Nerf/Nvdiffrec生成高质量资产、使用Omniverse/IsaacSim进行物理渲染，最终通过AI助手迭代逼近真实场景。

**🔧 技术方法**

采用了Nerf、Nvdiffrec进行3D扫描，Omniverse与Nvidia Isaac Sim进行仿真渲染，结合6D姿态估计、语义分割、风格迁移等深度学习技术。

**📊 数据集**

主要基于自行生成的合成数据，参考YCB-Video、Linemod、GraspNet等公开数据集做对比。

**📈 对比分析**

目前未给出正式实验对比，论文主要阐述框架和方法，预期通过仿真训练与真实标注迭代可提升姿态估计与抓取性能。

**⚠️ 局限性**

局限在于仍存在域差距、仿真真实性不足、对真实标注的依赖、成本与可扩展性等问题。

---

## 456. Quantization as a Malicious Task: Removing Quantization-Conditioned Backdoors via Task Arithmetic

**arXiv ID:** 2606.20254 | [PDF](https://arxiv.org/pdf/2606.20254v1)

**作者:** Kaihsun Yang `[一作]` (National Yang Ming Chiao Tung University), Chia-Mu Yu `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级的参数空间修正方法QVec，用来抵御量化条件后门(QCB)；

**💡 创新点**

将量化引起的参数偏移视为结构化的任务向量，利用对该方向的反向修正实现防御，避免了对量化过程的修改；

**🔧 技术方法**

使用参数差值任务向量、线性参数空间操作、一次性量化估计与轻量化超参数搜索；

**📊 数据集**

在图像分类上使用CIFAR‑10、Tiny‑ImageNet；在LLM上使用Gemma‑2B（内容注入、过度拒绝）和StarCoder‑1B（易损代码生成）等；

**📈 对比分析**

与EFRAP、LAC以及加噪声等基线对比，QVec在保持清洁准确率（CA）基本不变的情况下，将攻击成功率（ASR）压至接近0%；

**⚠️ 局限性**

仅针对QCB有效，无法防御传统后门；依赖确定性量化且对攻击者改动量化配置时效果不确定；若搜索不到满足准确率阈值的α，需退回原模型。

---

## 457. A Multi-Agent system for Multi-Objective constrained optimization

**arXiv ID:** 2606.20236 | [PDF](https://arxiv.org/pdf/2606.20236v1)

**作者:** Federica Filippini `[一作]` `[通讯]` (University of Milano-Bicocca), Federica Filippini (University of Milano-Bicocca)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了MAMO——一种层次化多智能体框架，用于在边缘FaaS资源扩展问题中通过学习奖励权重来动态平衡成本与QoS约束。

**💡 创新点**

创新点在于将目标设计与任务执行解耦，引入权重自适应智能体（WA）在较慢时间尺度上调节奖励中的权重，而不是手工设定，从而让系统自学习最佳权衡。

**🔧 技术方法**

采用强化学习（Deep Q‑Learning）实现TE和WA智能体，使用RL4CC库（基于Ray RLlib），通过线性加权奖励、离散化权重空间以及经验回放等技术构建训练框架。

**📊 数据集**

实验使用模拟的边缘FaaS环境：单函数、10个可用副本，负载为正弦波（昼夜周期）加均匀噪声；基线为Gurobi离线最优解；未使用公开数据集。

**📈 对比分析**

将固定权重（0.99、0.1）和MAMO结果与离线最优进行对比；MAMO在保持拒绝率低于0.05的同时，执行成本略高于最优解，但显著优于单一固定权重策略；实验表明MAMO能够在动态负载下实现良好平衡。

**⚠️ 局限性**

局限性包括：实验仅针对单函数场景，未验证多函数或更复杂约束的效果；权重空间离散且更新频率低，可能导致学习效率和精细度受限；仅使用线性奖励标量化，无法捕捉更复杂的目标交互。

---

## 458. ScholarQuest: A Taxonomy-Guided Benchmark for Agentic Academic Paper Search in Open Literature Environments

**arXiv ID:** 2606.20235 | [PDF](https://arxiv.org/pdf/2606.20235v1)

**作者:** Tingyue Pan `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ScholarQuest 基准，用于评估 LLM 代理式学术论文检索，包含查询生成、答案构造和共享检索后端 ScholarBase。

**💡 创新点**

创新点在于通过分类体系驱动的四类研究意图、自动化答案构造管线和可复现的检索后端，构建大规模、可扩展、意图多样的代理检索基准。

**🔧 技术方法**

采用多源检索、引用图扩展、LLM 相关性判定、自动生成查询及答案，并构建基于 arXiv 的 ScholarBase 环境。

**📊 数据集**

使用 ACM CCS 1,600+ 主题生成 1,000+ CS 主题的 1,111 个查询，答案集结合 arXiv、Semantic Scholar 等公开数据库。

**📈 对比分析**

与稀疏、稠密、混合检索、Google/Scholar/DeepXiv 以及 PaSa、SPAR、PaperScout 等代理系统比较，最佳代理 PaperScout 在 R@100 上从 0.214 提升至 0.314，提升约 46.7%。

**⚠️ 局限性**

局限性包括仅覆盖 CS 领域、仅基于标题/摘要判断相关性、自动答案构造仍可能遗漏文献、缺乏全文信息及跨学科扩展。

---

## 459. SysML Modeling of Digital Twins for Renewable Energy Communities

**arXiv ID:** 2606.20230 | [PDF](https://arxiv.org/pdf/2606.20230v1)

**作者:** Mohammad Samadi `[一作]` (Polytechnic Institute of Porto), Gabriela Lucas `[通讯]` (Cleanwatts Digital)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

基于Modelio将可再生能源社区域模型转化为SysML结构图（设备分类与社区组织视图），识别并提出解决四个语义缺口的方案，计划将模型映射到Eclipse Ditto实现运行时数字孪生。

**💡 创新点**

首次将SysML与SAREF4ENER本体结合，为RECs提供结构与语义兼容的完整模型，并提出通过引用包方式引入本体而非自定义标注，避免模型与实现耦合。

**🔧 技术方法**

使用的技术包括Modelio（SysML）、SAREF4ENER本体、Eclipse Ditto、JSON/REST API、SHACL/pySHACL等。

**📊 数据集**

采用真实硬件数据（Shelly EM 计量器、光伏逆变器、电动车充电器、热泵控制器）的REST API获取实时数据作为案例数据集。

**📈 对比分析**

本文为概念性框架与模型设计，没有进行实验对比或性能评估，后续工作计划在Ditto上生成Thing定义并验证实时同步功能。

**⚠️ 局限性**

局限性在于仅完成结构建模与语义缺口识别，尚未实现完整的SAREF4ENER映射与运行时链接；缺乏实验验证与性能指标；模型规模有限，未覆盖更复杂的多场景和多用户情况。

---

## 460. QMFOL: Benchmarking Large Language Model Reasoning via Quantifiable Monadic First-Order Logic Test Case Generation

**arXiv ID:** 2606.20227 | [PDF](https://arxiv.org/pdf/2606.20227v1)

**作者:** Xinyi Zheng `[一作]` (Huazhong University of Science and Technology), Kailong Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个可自动生成多维可控的单值一阶逻辑推理任务的框架，并基于该框架构建了2880个实例的QMFOL基准；

**💡 创新点**

创新点在于通过深度、宽度、干扰项数量和主题等维度对逻辑复杂度进行细粒度控制，并通过LLM的双向翻译和外部定理证明器实现逻辑一致性验证；

**🔧 技术方法**

主要技术包括基于并行与非并行逻辑规则的自动构造、Prompt工程的FOL2NL与NL2FOL转换、Vampire等定理证明器的round‑trip校验以及对大规模推理模型的推理模式调节；

**📊 数据集**

使用所构建的QMFOL基准数据集，包含960种逻辑/语义配置、3个随机种子共2880个实例；

**📈 对比分析**

通过与六款大型推理模型（Gemini‑3.1‑Pro、GPT‑5.4‑High、Qwen3.5‑27B、DS‑V3.2‑Think等）及两款通用LLM（GPT‑5.4‑None、DeepSeek‑V3.2）在不同深度/宽度、标签类型及干扰项数量下进行对比，结果显示顶级模型在最高难度下仍能保持>97%准确率，性能随深度、宽度与干扰项增加而下降；

**⚠️ 局限性**

局限性包括仅覆盖单值一阶逻辑、对LLM翻译质量和主题域多样性的依赖、未完全评估全一阶逻辑或高阶谓词的可扩展性以及可能存在的翻译/验证误差。

---

## 461. The Register Gap: A Meaning Intelligence Framework for Nigerian Public Discourse

**arXiv ID:** 2606.20255 | [PDF](https://arxiv.org/pdf/2606.20255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 462. Zero-VC: Zero-Lookahead Streaming Voice Conversion via Speaker Anonymization

**arXiv ID:** 2606.20218 | [PDF](https://arxiv.org/pdf/2606.20218v1)

**作者:** Yudong Li `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种名为Zero-VC的严格因果、零前瞻的实时语音转换系统，能够在极低延迟下实现高质量的零样本说话人转换。

**💡 创新点**

创新点在于将说话人匿名化作为扰动机制来平衡音色泄漏与语言/韵律保留，从而显著降低对未来上下文的依赖，达成20 ms的最小算法延迟。

**🔧 技术方法**

采用了说话人匿名化模块、WavLM音色编码、HiFi-GAN风格的因果解码器、Multi-Scale/Period Discriminator训练、Causal Conv1D等技术。

**📊 数据集**

训练使用LibriTTS数据集，评估使用seed-tts-eval（Common Voice子集）的英文学术语音。

**📈 对比分析**

与流式SOTA模型（StreamVC、RT-VC）和非流式模型（LSCodec、CosyVoice、Seed-VC-Small）对比，Zero-VC在源音色泄漏最低（SS‑S = 0.171）、目标音色相似最高（SS‑R = 0.521）、FPC = 0.688、WER = 3.96%、NMOS = 3.81、SMOS = 3.88、RTF = 0.063，算法延迟仅20 ms。

**⚠️ 局限性**

限制在于训练期间仍依赖离线说话人匿名化模块，带来额外预处理开销；未来工作需实现端到端整合并支持跨语言转换。

---

## 463. Beyond Accuracy: Measuring Logical Compliance of Predictive Models

**arXiv ID:** 2606.20208 | [PDF](https://arxiv.org/pdf/2606.20208v1)

**作者:** Guillaume Olivier Delplanque `[一作]`, Zephirin Faure `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Rule Violation Score（RVS）评估指标，用于衡量机器学习模型在不依赖真实标签的前提下，对给定逻辑规则的遵从度。

**💡 创新点**

创新点在于：①区分硬约束与软规则并按数据集违约率自适应归一化；②可与任何模型、数据集、规则集无关；③通过自动生成 SQL 查询实现可扩展、可执行的评估流程；④提供规则级别诊断，帮助发现数据或规则问题。

**🔧 技术方法**

技术方法包括：定义评估环境 (D, ∑, R)；计算数据集违约率 d_r 与预测违约率 p_r；对硬规则求违约次数，对软规则计算 p_r/d_r；聚合权重求全局 RVS；利用关系数据库执行 SQL 生成与计数；支持 Horn 规则的自动化推理。

**📊 数据集**

使用三大基准：Family（知识图谱，12 条关系，约 19k 条三元组）；FB15k‑237（知识图谱，237 条关系，约 272k 条三元组）；DV3F（开放式房地产数据集，约 4.8M 条元组，涉及价格回归）。

**📈 对比分析**

与传统指标（Hits@1、MRR、MAE、R²）并行比较：在 Family、FB15k‑237 和 DV3F 上对比 AnyBURL、CompGCN、UniKER、ExpressGNN、GraphSAGE、Rel‑LLM 等模型。结果显示：同等准确度的模型往往在 RVS 上差异显著；如 CompGCN 在 Family 上 MRR 更高但硬规则违约数显著多于 UniKER；ExpressGNN 在 FB15k‑237 上 MAE/ R² 与 CompGCN 相近，但 RVS 更好；Rel‑LLM 在 DV3F 上 R² 低于 GraphSAGE，却在软规则上表现更优。说明 RVS 能补充传统指标，揭示模型在约束敏感任务中的逻辑一致性。

**⚠️ 局限性**

局限性包括：①仅适用于能翻译成 SQL 的 Horn 规则；②需要先行定义规则集并划分硬/软，规则质量直接影响评分；③未在训练阶段使用，不能直接提升模型性能；④对开放世界假设需额外的反证规则，增加复杂度；⑤计算开销受数据库查询规模影响，若规则体量大或数据量极大可能导致性能瓶颈。

---

## 464. Reliability-Aware Prototype Calibration for Frozen Pose-Flow Video Anomaly Detection

**arXiv ID:** 2606.20312 | [PDF](https://arxiv.org/pdf/2606.20312v1)

**作者:** Ning Dong `[一作]` (Suqian University), Zhuangzhuang Pan `[通讯]` (Universiti Malaya)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在冻结的 pose‑flow 视频异常检测器上，提出了一种后置可靠性感知原型校准（RPC）方法，通过将标准化的最邻近原型偏差与置信门控融合来提升异常排名。

**💡 创新点**

创新点在于：①无需再训练的后置分数校准；②利用冻结潜在空间的原型统计来弥补单一似然的多模态不足；③采用关键点置信门控，控制几何证据的可靠性。

**🔧 技术方法**

使用 K‑means 原型拟合、对角马氏距离、分数标准化与融合、置信门控，以及基于已冻结的 normalizing flow 潜在特征的分数标准化。

**📊 数据集**

实验数据集包括四个：ShanghaiTech Campus（SHTech）及其人类相关变体 SHTech‑HR，UBnormal（UBN）及其人类相关变体 UBN‑HR。

**📈 对比分析**

与原始冻结似然基线以及传统后置方法（高斯、kNN、OC‑SVM、Isolation Forest）比较，RPC 在所有八个配置中平均提升约 2.0% AUC，单项最大提升 4.5%，在多种方法中保持竞争位置。

**⚠️ 局限性**

限制在于：①仅利用缓存骨架信息，无法恢复缺失关节或加入视觉/对象特征；②置信门控对关节缺失无效；③超参数选取依赖代理验证，可能无法覆盖所有真实异常场景。

---

## 465. Navigating Unreliable Parametric and Contextual Knowledge: Explicit Knowledge Conflict Resolution for LLM Inference

**arXiv ID:** 2606.20245 | [PDF](https://arxiv.org/pdf/2606.20245v1)

**作者:** Huang Peng `[一作]` (National University of Defense Technology), Xiang Zhao `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多代理推理框架的 LLM 知识冲突解决方法，能自适应评估内部与外部知识并通过规则实现冲突检测与解释。

**💡 创新点**

创新点包括：① 无需预设可靠来源的自适应知识评估与检索；② 观察者–分析者–推理者三代理协作的自学习规则机制；③ 输出可解释的冲突解决报告。

**🔧 技术方法**

使用技术包括：语义熵与改进的语义邻域熵用于自适应置信度评估；检索增强生成（RAG）；多代理推理与规则诱导（Observer、Analyzer、Reasoner）；LLM 采样、提示工程和规则验证。

**📊 数据集**

使用的数据集为 ConflictBank、ConFiQA、MQuAKE 以及针对噪声比例的 Variant Context 子集。

**📈 对比分析**

与 Direct Answer、In‑Context Learning、Robust RAG、Dynamic Selection 等基线在 EM 和 ROUGE‑L 上进行对比，实验表明在所有三大基准上均获得第一或第二名，EM 提升显著（如 0.19→0.55），对噪声鲁棒性也表现优异。

**⚠️ 局限性**

主要局限包括：多代理交互导致多次 LLM 调用，计算延迟和成本高；规则诱导过程高度依赖 LLM 的抽象能力，可能出现泛化不佳，需要进一步优化效率与稳定性。

---

## 466. Phoenix: Safe GitHub Issue Resolution via Multi-Agent LLMs

**arXiv ID:** 2606.20243 | [PDF](https://arxiv.org/pdf/2606.20243v1)

**作者:** Kipngeno Koech `[一作]` (Carnegie Mellon University), Joao Barros `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了Phoenix，一个基于多代理LLM的安全GitHub issue解决系统，完成从问题追踪到拉取请求的全流程。

**💡 创新点**

创新点包括六代理分工、标签驱动的状态机、七层安全机制、基线感知的测试评估，并将安全放在首位。

**🔧 技术方法**

采用大型语言模型（Claude Sonnet 4）与LangChain实现多代理推理，配合GitHub API、GitPython、FastAPI、SSE等技术实现自动化、状态管理与安全控制。

**📊 数据集**

使用SWE‑bench Lite（24个实例）以及从14个开源仓库抽取的42条真实issue进行评估。

**📈 对比分析**

在SWE‑bench Lite上Oracle解析率为75%，无回归；在42条真实issue上实现100%正确率，硬难度仓库平均解决时间为122秒。

**⚠️ 局限性**

局限性包括文件定位仅基于关键字匹配导致约50%PR定位错误、测试覆盖不足导致回归检测不完善、未与单代理基线对比且功能充分性评估缺失。

---

## 467. Mobile Target Search with Imperfect Perception: A Partially Observable Stochastic Game Theoretical Approach

**arXiv ID:** 2606.20232 | [PDF](https://arxiv.org/pdf/2606.20232v1)

**作者:** Hanzheng Zhang `[一作]` (Tongji University), Shuyu Liu `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在传感器限制、恶意干扰或通信噪声导致的不完美感知下的移动目标搜索问题。通过采用部分可观察随机博弈（POSG）方法，分析了搜索者与目标之间的动态互动，并提出了一种新的可检测性概念来确保搜索策略的有效性。

**💡 创新点**

创新点在于提出了α-可检测性概念，正式确定在存在误报的情况下搜索策略是否能够保证最终检测。同时，开发了基于聚合潜力博弈结构的分布式算法，以应对高维状态和动作空间的计算复杂性。

**🔧 技术方法**

使用了部分可观察随机博弈（POSG）方法，结合了随机递归分析来处理不完美感知下的搜索策略设计，并设计了一个服务器辅助的分布式算法来优化搜索者的策略。

**📊 数据集**

使用了一个10×10的网格区域进行数值模拟，假设搜索者的移动能力为每步1个网格，检测阈值设置为α=0.4，初始时对目标位置没有先验信息。

**📈 对比分析**

与其他启发式搜索策略（如贪婪策略和最大覆盖策略）进行比较，提出的算法在高误报率条件下表现出较低的中位检测时间和较低的方差，显示出其在噪声感知下的鲁棒性和可靠性。

**⚠️ 局限性**

限制在于算法的收敛性难以直接保证，且在高维多智能体设置中，计算复杂性可能导致实际应用中的挑战。

---

## 468. DeepForestVisionV2: Ecology-Driven Taxonomy Expansion for Camera-Trap Monitoring in African Tropical Forests

**arXiv ID:** 2606.20223 | [PDF](https://arxiv.org/pdf/2606.20223v1)

**作者:** Hugo Magaldi `[一作]` (Muséum national d'Histoire naturelle), Sabrina Krief `[通讯]` (Muséum national d'Histoire naturelle)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发并评估了基于DeepForestVision的64类分类模型DeepForestVisionV2，用于非洲热带森林相机陷阱监测。

**💡 创新点**

生态驱动的标签空间从35类扩展到64类，覆盖垂直分层、开放景观和人类接口三种生态梯度，并在同一离线工作流中提升检测准确性与物种覆盖度。

**🔧 技术方法**

利用MegaDetector v5进行目标检测，DINOv3 ViT‑B/16进行图像分类，PyTorch训练，AddaxAI无代码离线推理。

**📊 数据集**

使用1,535,010张照片和243,354段视频，来自塞比托利、CIRAD‑SWM、Biotope等多国项目，覆盖不同站点、栖息地和摄像设置。

**📈 对比分析**

与原始DeepForestVision在相同离线流程下对比，使用精度、平衡精度和宏F1等指标；跨国验证集达86%精度；乌干达视频基准中森林内0.89、河岸0.72、园边0.86，显著提升物种检测数量。

**⚠️ 局限性**

长尾类别精度偏低，校准不足导致置信度低估；视频基准规模有限，未覆盖所有环境；多物种事件可能被单标签忽略；需要后处理校准与阈值调优以实现自动报警。

---

## 469. GNSS Spoofing Threat for V2X communications

**arXiv ID:** 2606.20215 | [PDF](https://arxiv.org/pdf/2606.20215v1)

**作者:** Adolfo P. Jimenez `[一作]` (Universidad Politécnica de Madrid), Felipe Jimenez Alonso `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用低成本软件定义无线电（HackRF One）对真实车辆OBU和RSU进行GNSS信号伪造攻击，验证V2X通信系统在不同速度场景下被操纵的可行性。

**💡 创新点**

首次在实际V2X硬件上实现物理层GNSS伪造攻击，并提出基于Haversine距离、时间离散化和线性插值的坐标生成流水线，展示了攻击能在200 km/h等高速下无检测成功。

**🔧 技术方法**

软件定义无线电、GPS‑SDR‑SIM仿真器、GNU Radio、Haversine公式、时间离散化、线性插值等技术；以及对HackRF One的发射与接收链路、OBU/RSU的硬件平台。

**📊 数据集**

使用路由引擎生成的离散轨迹坐标（人工合成路线），以及现场OBU/RSU收集的真实GNSS观测数据；未使用公开的标准数据集。

**📈 对比分析**

通过对OBU的速度、加速度和位置误差进行实时监测与比较；实验表明伪造信号能在91‑547 m范围内持续占优，且在200 km/h场景下产生超过20 m/s²的异常加速度，表明攻击成功且未被系统检测。

**⚠️ 局限性**

实验仅限单一OBU/RSU配置，未覆盖多车交互；仅在无复杂环境（多路径、遮挡）的受控测试场景下验证；缺乏对抗检测机制的评估；对硬件平台的依赖使得推广性受限。

---

## 470. FlowMaps: Modeling Long-Term Multimodal Object Dynamics with Flow Matching

**arXiv ID:** 2606.20209 | [PDF](https://arxiv.org/pdf/2606.20209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 471. Accelerating Trust Convergence in IIoT: A ML Approach for Dynamic Network Conditions

**arXiv ID:** 2606.20214 | [PDF](https://arxiv.org/pdf/2606.20214v1)

**作者:** Aymen Bouferroum `[一作]` (Inria Lille-Nord Europe), Abderrahim Benslimane `[通讯]` (University of Avignon)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于机器学习的Trust Convergence Acceleration（TCA）方法，利用随机森林预测IIoT网络中信任收敛时间，并动态调节马尔可夫链转移概率，从而在不同网络质量下显著提升信任收敛速度。

**💡 创新点**

创新点在于将网络质量量化为统一的netC参数并通过预训练随机森林模型预测收敛时间，再根据预测结果自适应加速信任转移，最高可将收敛时间缩短28.6%。

**🔧 技术方法**

使用了随机森林机器学习模型、IEEE 802.11ax Wi‑Fi6仿真环境、马尔可夫链信任模型以及信任评估指标（Cooperation Rate、Direct Honesty、Indirect Honesty）。

**📊 数据集**

数据集来源于MATLAB WLAN/Communications Toolbox仿真，约35,000条样本（后平衡为6,000条），包含网络指标（SNR、Packet Loss、Jitter、Latency、Throughput、SINR）及对应的收敛时间类别。

**📈 对比分析**

通过与原始Tm‑IoT模型在Good/Medium/Poor三种网络条件、不同恶意节点比例以及不同网络规模下的仿真对比，TCA在Worst条件下将收敛时间从14降至10，提升约28.6%；在恶意攻击情形下仍保持更快的收敛并保持最终信任准确性。

**⚠️ 局限性**

局限性包括仅在仿真环境验证，未在真实工业现场测试；模型对极端网络条件或多种无线技术的泛化能力未知；并且使用离线预训练模型，可能对网络条件快速变化的响应略显滞后。

---

## 472. CzechDocs: A Multiway Parallel Dataset of Formatted Documents for Minority Languages in Czechia

**arXiv ID:** 2606.20212 | [PDF](https://arxiv.org/pdf/2606.20212v1)

**作者:** Josef Jon `[一作]` (Charles University), Ondřej Bojar `[通讯]` (Charles University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个多格式（HTML、PDF、DOCX）且段落级对齐的捷克及少数族裔语言标记保持机器翻译评估数据集，并在该数据集上对比了LLM直接翻译与 detag‑and‑project 两种方法。

**💡 创新点**

创新点包括①构建高质量、多语种本地化数据集，②在LLM框架下系统评估标记保持性能，并探讨提示工程对标记保留的影响。

**🔧 技术方法**

技术手段包括 Okapi Tikal 提取 XLIFF、Moses Inline 文本格式、基于词对齐的标记投影；LLM 翻译使用 GPT‑4.1‑nano 与 Aya‑expanse‑8b，并结合不同提示策略。

**📊 数据集**

使用的数据集为 CzechDocs，包含 77 条捷克原文、316 条多语言段落（共 60,153 段、15 种语言），主要来源于政府、教育与移民门户网站。

**📈 对比分析**

比较方法：对 detag‑and‑project、直接标记输入、以及显式标记提示三种方式进行 BLEU（含/不含标记）评估；结果显示 LLM 在含标记 BLEU 上略优，显式提示提升标记保留，但 detag‑and‑project 在无标记 BLEU 略低。

**⚠️ 局限性**

局限性：仅在验证集评估，测试集未公开；标记投影方法对文档结构差异敏感；LLM 性能易受提示变化影响，数据集规模对复杂标签类型有限。

---

## 473. Recurrent neural networks approximate continuous functions

**arXiv ID:** 2606.20325 | [PDF](https://arxiv.org/pdf/2606.20325v1)

**作者:** Valentin Abadie `[一作]`, Helmut Bölcskei `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

本文证明了任意定义在[-1,1]上的连续函数都可以通过单一ReLU递归神经网络（RNN）的时间演化实现均匀逼近；只需让网络运行更长时间即可提高逼近精度；

**💡 创新点**

创新点在于引入“神经单元图灵机”（TMNU）这一中间模型，既保留了实现多项式逼近的算法自由度，又能够被RNN严格模拟，从而实现固定网络、可调运行时长的逼近范式，并给出了对应的收敛速率与下界；

**🔧 技术方法**

主要技术包括构造TMNU并证明其可由ReLU RNN 在显式隐藏维数与权重范围内模拟，利用多项式逼近理论推导收敛速率，并通过最优下界证明运行时长的必要性；

**📊 数据集**

该工作为理论性质的论文，未使用任何实际数据集；

**📈 对比分析**

作者通过理论比较，将固定网络的逼近速率与传统多项式逼近速率进行对照，并通过下界证明运行时长不可省略；性能指标为逼近误差随时间的衰减率，结果表明与多项式逼近保持一致；

**⚠️ 局限性**

局限性包括：需要较长的运行时间才能达到高精度；仅在[-1,1]区间内讨论；缺乏实验验证；对网络规模的实际可行性与其他任务的泛化仍待进一步研究。

---

## 474. A Model-Driven Approach for Developing Families of Reinforcement Learning Environments

**arXiv ID:** 2606.20324 | [PDF](https://arxiv.org/pdf/2606.20324v1)

**作者:** Xiaoran Liu `[一作]` (McMaster University), Istvan David `[通讯]` (McMaster University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于模型驱动工程（MDE）的混合遗传算法，自动生成强化学习（RL）训练环境族并支持课程学习（CL）

**💡 创新点**

将环境变异和约束编码为模型变换，利用MDE的变换引擎实现全自动化搜索，并将多目标度量（多样性与复杂度）融合到GA中

**🔧 技术方法**

混合遗传算法、模型变换（如EMF TransactionalEditingDomain）、Gymnasium框架、模拟退火局部搜索

**📊 数据集**

合成的8×8“燃烧森林”网格环境族（基于Burning Forest案例）

**📈 对比分析**

与直接在最难环境上训练的基线对比，使用累计奖励和成功率评估，结果显示课程学习在所有环境中获得100%成功率（除最短课程外），并且学习曲线更平滑，表明性能显著提升

**⚠️ 局限性**

对规模的可扩展性有限，复杂度度量仍需手工定义，DSL和HOTs尚未成熟，实验仅覆盖单一CL场景，未验证其他RL范式

---

## 475. Token-Operations-Oriented Inference Optimization Techniques for Large Models

**arXiv ID:** 2606.20295 | [PDF](https://arxiv.org/pdf/2606.20295v1)

**作者:** Shiguo Lian `[一作]` (China Unicom), Qinghuai Ma `[通讯]` (Hygon Information Technology Co., Ltd.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出面向 token 的四层技术架构（多模型融合、模型优化、算力模型融合、网络算力模型融合），并系统评估其在大规模 Token 运营中的成本、效率与稳定性平衡。

**💡 创新点**

首次构建统一的四层框架并对每层关键技术进行全面梳理，揭示 Token 运营的多维协同与优化路径。

**🔧 技术方法**

综合利用模型能力边界量化、智能路由、模型级联、模型集成、低复杂度注意力、MoE、扩散路径优化、链式推理优化、内存管理、KV 缓存压缩、推测解码、量化与蒸馏等多项技术。

**📊 数据集**

通过公开基准（如 MMLU、BIG‑bench、HELm、Arena 等）以及企业内部业务样本进行评估与验证。

**📈 对比分析**

与传统单模型推理及现有优化框架（如 vLLM、SGLang、OpenRouter 等）对比，平均推理延迟降低 30%+、Token 成本显著下降、吞吐量提升。

**⚠️ 局限性**

局限在多模型协同与网络资源调度的复杂性，极大规模 Token 负载下仍受 KV 缓存、跨节点通信与成本控制的瓶颈影响。

---

## 476. Shifting-based Optimizable Linear Relaxations for General Activation Functions

**arXiv ID:** 2606.20292 | [PDF](https://arxiv.org/pdf/2606.20292v1)

**作者:** Philipp Kern `[一作]` (Karlsruhe Institute Of Technology), Carsten Sinz `[通讯]` (Karlsruhe University Of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种通用的可优化线性松弛框架 SLiR，用于在神经网络验证中对任意激活函数构造可证明的上下界，并通过梯度下降进一步优化这些界限。

**💡 创新点**

创新点在于把线性松弛参数化为斜率，通过平移（求垂直平移量）即可得到可证明的松弛；该方法不需要针对每个激活函数手工推导松弛，而只需闭式临界点或局部 Lipschitz 常数；并引入凸包限定斜率范围、sigmoid 映射以及改进的 Piyavskii Lipschitz 优化以提高收敛速度与精度。

**🔧 技术方法**

技术包括：线性松弛与边界传播（backsubstitution）框架；piecewise linear (PWL) 逼近与上/下包络；改进的 Piyavskii–Schubert Lipschitz 优化；凸包（monotone chain）计算；梯度下降参数化与 Sigmoid 映射；PyTorch 实现和对现有验证器的集成。

**📊 数据集**

使用 MNIST 与 CIFAR‑10 的 CNN 进行实验，涵盖多种非标准激活函数（atansq、gelu、lisht、loglog、mish、swish）以及自定义激活函数，并验证对抗鲁棒性（PGD L∞ 攻击）。

**📈 对比分析**

与 α‑CROWN、Neurify、E‑Guided、CROWN‑DNN 等基线及 VNN‑COMP 2021‑25 的主流方法进行对比；SLiR 在初始化和 20 步梯度优化后均能验证更多属性；在 MNIST 上速度仅比 α‑CROWN 慢 1.4–1.9 倍，在 CIFAR‑10 上差距更小；总体上在大多数基准上实现了最高的验证成功率。

**⚠️ 局限性**

局限性包括：初始化阶段比特定手工松弛方法慢；需要预先提供 Lipschitz 常数或闭式临界点，某些激活函数（如 ELU）在分解时不易处理；极端预激活范围导致斜率范围窄，可能影响收敛；整体计算量仍高于专门定制的快速松弛实现。

---

## 477. Boundary Embedding Shaping with Adaptive Contrastive Learning for Graph Structural Disentanglement

**arXiv ID:** 2606.20283 | [PDF](https://arxiv.org/pdf/2606.20283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 478. Integrating national forest inventory, airborne lidar, and satellite imagery for wall-to-wall mapping of forest structure with computer vision

**arXiv ID:** 2606.20291 | [PDF](https://arxiv.org/pdf/2606.20291v1)

**作者:** Luke J. Zachmann `[一作]` (Vibrant Planet Public Benefit Corporation), Guy Bayes `[通讯]` (Vibrant Planet Public Benefit Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了 VibrantForests 框架，通过 Sentinel‑2 卫星影像结合 lidar 推导的全方位森林结构属性（冠层覆盖度、冠层高度、上层生物量、基面积、二次平均直径）在美国大陆尺度实现 10 m 分辨率、年度壁到壁的预测。

**💡 创新点**

创新点包括：① 将 lidar 生成的全尺度样本作为训练标签，融合多任务量化回归；② 采用 Feature Pyramid Network 与 Vision Transformer 的 Masked AutoEncoder 提升对高生物量、密集冠层的识别能力；③ 通过多目标分位数回归减少回归到均值的偏差，并显著延伸传统被动传感器模型的饱和阈值。

**🔧 技术方法**

技术方法：卫星图像预处理（云掩模、时间中位数合成）、lidar 数据生成 CHM 与冠层覆盖、基于 scikit‑learn 的多目标回归管线、Vision Transformer‑Masked AutoEncoder 冻结特征提取 + FPN 回归头、PyTorch Lightning 训练、分位数回归损失、早停与学习率衰减。

**📊 数据集**

使用的数据集包括：Sentinel‑2 Level‑2A（2024 年季节化合成）、USGS 3DEP lidar（质量等级 2+）、FIA 子样点（可提取冠层覆盖、高度、BA、AGB、QMD）、太平洋西北地区独立现场样本、TreeMap、LANDFIRE、GEDI、公开的 FVS 计算值。

**📈 对比分析**

评估方法：① 在内部验证集（lidar 生成的训练块）上测量 MAE、RMSE、R²；② 对太平洋西北现场样本进行点尺度比较，展示 AGB、Cover、Height、BA 的 MAE/误差；③ 在 64,000 ha 六边形尺度上与 FIA 观测做散点图和统计，检验区域偏差。性能表现：冠层覆盖 R²≈0.91，冠层高度 R²≈0.89，AGB 在验证集 R²≈0.66，现场样本 R²≈0.17（因时间滞后和扰动），区域尺度 AGB R²≈0.78、R²≈0.84 与 FIA 对齐；QMD 与 TPH 的 R² 较低。

**⚠️ 局限性**

局限性：① 对 QMD 与 TPH 的预测灵敏度不足，导致基面积和树径的估计不够精确；② 训练样本的时间滞后与现场扰动导致场地级评估偏差；③ 依赖上游 allometric 模型，若其偏差会传递至卫星模型；④ 在极高生物量（> 450 Mg/ha）和极高冠层高度（> 40 m）下仍可能出现轻微欠估或噪声；⑤ 目前仅覆盖美国大陆，跨区域泛化需要进一步验证。

---

## 479. PsyScore: A Psychometrically-Aware Framework for Trait-Adaptive Essay Scoring and ZPD-Scaffolded Feedback

**arXiv ID:** 2606.20287 | [PDF](https://arxiv.org/pdf/2606.20287v1)

**作者:** Wei Xia `[一作]` (East China Normal University), Chanjin Zheng `[通讯]` (East China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 PsyScore 框架，将神经 IRT 诊断与基于 ZPD 的多代理反馈生成结合，实现了兼顾准确评分与个性化教学反馈的自动作文评分系统。

**💡 创新点**

创新点包括：① Trait‑Adaptive Neural IRT 通过网格搜索初始化实现多维写作特征的心理测量校准；② ZPD‑Scaffolded 多代理反馈生成，将学生潜能作为控制信号；③ 采用多视角评估（偏好对比、模拟修订与专家评估）验证反馈质量。

**🔧 技术方法**

技术手段涵盖：BERT 编码、Graded Partial Credit Model、神经 IRT 估计、Generate‑then‑Fuse 多代理生成、LLM 辅助对齐、对比偏好判定与模拟修订评估。

**📊 数据集**

实验使用 ASAP++ 数据集，该数据集在原 ASAP 基础上提供多维特征评分（如 Content、Organization 等）。

**📈 对比分析**

在评分任务上与 9 个基线对比，PsyScore‑AES 以 0.747 的 QWK 领跑；在反馈评估中与 GPT‑4o、Llama‑4 等 LLM 进行 pairwise 对比，胜率超过 90%；在模拟修订实验中，低水平学生的标准化提升达 17.38%，显著优于无 IRT 版本。

**⚠️ 局限性**

局限性包括：多代理生成导致推理延迟；需要细粒度特征标签，限制在仅有整体评分的数据集上的迁移；仿真修订评估无法涵盖真实课堂中的动机与疲劳等因素。

---

## 480. Lagrange: An Open-Vocabulary, Energy-Based Sparse Framework for Generalized End-to-End Driving

**arXiv ID:** 2606.20274 | [PDF](https://arxiv.org/pdf/2606.20274v1)

**作者:** Shihao Ji `[一作]`, Mingyu Li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为Lagrange的稀疏开源词汇驾驶框架，利用视觉‑语言模型生成语义标记，经过意图驱动的掩码注意力筛选后映射到连续能量场，再通过Lagrangian动作最小化实现物理约束的轨迹规划。

**💡 创新点**

创新点包括：① 开放词汇稀疏标记器实现无类别前置的感知；② 意图驱动的掩码隐层场（MLF）实现自适应关注；③ 将语义标记映射为连续能量场，将轨迹规划转化为物理能量最小化；④ 结合MPPI实现实时可行性检验。

**🔧 技术方法**

使用的核心技术有：Vision‑Language Models（如CLIP/BLIP）进行连续语义编码；掩码跨注意力实现意图过滤；MLP解码生成二维能量场；Lagrangian动力学约束与MPPI采样优化。

**📊 数据集**

实验数据集包括：nuScenes（闭集基准）、CODA（长尾异常）以及Waymo Open Dataset（零样本跨域验证），并通过人工噪声和摄像头掉落进行感知鲁棒性测试。

**📈 对比分析**

与UniAD、SparseDrive、OpenVLA‑Car等现有方法对比，Lagrange在CODA上CR_OOD降至8.7%（最优），nuScenes碰撞率为0.25%并保持24.3 FPS；零样本迁移到Waymo的碰撞率下降至0.45%；在感知噪声与摄像头掉落情形下，碰撞率显著低于其他模型。

**⚠️ 局限性**

局限性：依赖几何区域提议，对非几何或非局部障碍（如黑冰、雨水泛滥等）检测不佳；需要进一步整合连续自由空间分割标记以提升完整路面覆盖。

---

## 481. Single-Stage Hierarchical Rectification for Weakly Supervised Histopathology Segmentation

**arXiv ID:** 2606.20250 | [PDF](https://arxiv.org/pdf/2606.20250v1)

**作者:** Duc T. Nguyen `[一作]` (VinUniversity), Huy-Hieu Pham `[通讯]` (VinUniversity)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种单阶段层次化校正（SSHR）框架，通过在前向传播过程中主动校正浅层特征，直接生成高质量的 CAM，实现弱监督组织病理图像分割。

**💡 创新点**

创新点在于引入 Hierarchical Feature Rectification Module（HFRM），将全局语义信息与局部空间同质化相结合，在特征层面实时消除纹理误差，避免多阶段误差累积，并显著提升训练效率。

**🔧 技术方法**

技术手段包括全局语义校正分支（GSR）利用深层语义注意力对低层通道进行加权；上下文同质化分支（CH）使用大核深度卷积实现空间平滑；残差融合与 1×1 卷积生成多尺度 CAM，采用多标签软边缘损失进行优化。

**📊 数据集**

实验数据集为肺腺癌组织分割数据集 LUAD‑HistoSeg 和乳腺癌组织分割数据集 BCSS。

**📈 对比分析**

与现有多阶段方法（MLPS、ARML、ESFAN、PBIP 等）在两数据集上的比较显示，SSHR 在 mIoU 与 mDice 上均取得最高分，并将训练时间缩短 2–5 倍、推理时延降低至 9.10 ms，表现最优。

**⚠️ 局限性**

局限性包括对深度 backbone 的语义可靠性依赖，以及缺乏全 slide 长程交互与全局上下文建模，导致对染色差异和形态多样性的鲁棒性仍需进一步提升。

---

## 482. SPOT-E: Test-Time Entropy Shaping with Visual Spotlights for Frozen VLMs

**arXiv ID:** 2606.20244 | [PDF](https://arxiv.org/pdf/2606.20244v1)

**作者:** Bo Yin `[一作]` (National University of Singapore), Shuicheng YAN `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SPOT-E，一种在推理时通过视觉Spotlight对冻结的视觉语言模型进行自适应的轻量级调节，提升证据利用率。

**💡 创新点**

创新点在于使用答案熵作为无标签的反馈信号，并通过低熵锚点实现熵塑造，避免因短路导致的错误自信；同时设计了可针对每个实例进行优化的视觉Spotlight与GRPO策略。

**🔧 技术方法**

核心技术包括答案熵与低熵锚点的熵塑造奖励、CLIP基础的多视图视觉Spotlight模块、基于GRPO的每实例测试时优化，以及软掩码与背景降质的视觉干预。

**📊 数据集**

在多种视觉语言模型（Qwen-VL、LLaVA、InternVL以及GPT‑4o、Gemini‑2.5‑Flash等）上，在包含细粒度证据任务的 TextVQA、DocVQA、ChartQA、MathVista、MMMU、GQA、MMBench、POPE 等公开基准数据集上进行评估。

**📈 对比分析**

与多种推理时视觉干预基线（FGVP、SoM、ViCrop、AttWarp 等）及相同解码配置下的原始冻结模型对比，SPOT‑E 在证据密集任务上提升 1–3% 以上，并在视觉噪声、分辨率下降和遮挡等域外干扰下显著减小性能下降。

**⚠️ 局限性**

局限包括对极小或本身模糊的证据仍难以捕获，对长推理或复杂生成任务的提升有限，以及每实例优化仍需额外计算开销。

---

## 483. BAFIS: Dataset + Framework to assess occupational Bias and Human Preference in modern Text-to-image Models

**arXiv ID:** 2606.20241 | [PDF](https://arxiv.org/pdf/2606.20241v1)

**作者:** Thomas Klassert `[一作]` (RheinMain University of Applied Sciences), Biying Fu `[通讯]` (RheinMain University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了多语言文本到图像模型偏差评测平台BAFIS，并基于MAGBIG扩展生成21,140张职业相关合成图像，系统评估了五大主流模型在性别、种族偏差、图像质量与提示对齐上的表现，结合人工偏好反馈。

**💡 创新点**

创新点在于：① 将人类偏好评估与传统自动指标融合的多维度评测框架；② 在多语言（德英）和职业组提示上扩展MAGBIG，提供更细粒度的偏差分析；③ 提出静态但可交互的BAFIS平台，实现匿名随机对战与 Elo/Bradley‑Terry 排名。

**🔧 技术方法**

使用技术包括：CLIP向量相似度（提示对齐）；FID、IS、MagFace（图像质量与面部质量）；YOLOv8+FairFace（性别/种族识别）；Elo/MLE‑Elo（人类偏好排名）；并通过 Python/JavaScript 搭建前端与后端交互。

**📊 数据集**

数据集来源于 MAGBIG（职业提示）扩展至 151 对德英组提示，生成 1,057 个提示共 21,140 张图像；同时采集德国联邦就业局 2024 年职业统计作为社会参考。

**📈 对比分析**

评价方法为：① 先计算自动指标（FID、IS、MagFace、CLIP Cosine）；② 再通过 BAFIS 收集 459 场对战的人工偏好，转化为 Elo/MLE‑Elo 排名；③ 与 ImgSys、T2I Arena 的公开排名对比。结果显示：DALL‑E 3 在提示对齐最高，但在图像质量上被 FLUX.1‑dev 超越；传统指标与人类偏好不完全一致，提示修订对人类评价影响显著。

**⚠️ 局限性**

局限性包括：仅评估 5 款默认配置模型；数据集规模与语言覆盖有限（仅德英）；BAFIS 为静态评测，无法模拟用户交互式 prompt 微调；人工评测受主观偏好与样本大小影响，可能引入噪声。

---

## 484. Image Encryption Algorithm Based on Convolutional Neural Networks and Dynamic S-Box Generation

**arXiv ID:** 2606.20444 | [PDF](https://arxiv.org/pdf/2606.20444v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 485. Learner-based Concept Drift Detection: Analysis and Evaluation

**arXiv ID:** 2606.20216 | [PDF](https://arxiv.org/pdf/2606.20216v1)

**作者:** Md Moman Ul Haque Khan `[一作]` (University of Regina), Samira Sadaoui `[通讯]` (University of Regina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述了监督学习环境下的概念漂移检测算法，并在多种漂移类型（突变、渐变、循环等）与两种基学习器（朴素贝叶斯 NB 与 Hoeffding Tree HT）上进行实验评估，探讨其在实时流数据中的性能。

**💡 创新点**

创新点在于：①将 learner‑based 检测器按统计过程控制（SPC）、窗口化与集成三大范式统一框架；②提供统一实验平台，对比 15 种代表性算法在合成与真实流数据上的表现；③系统阐述了不同基学习器对漂移类型的适应性与性能差异，揭示集成方法在多场景下的优势。

**🔧 技术方法**

技术方法包括：统计过程控制（DDM、EDDM、RDDM、FHDDM、EWMA、FTDD 等）利用 Hoeffding bound、EWMA 等统计检验；窗口化检测（ADWIN、KSWIN、FPDD、MDDM、WSTD、D3）通过动态窗口比较误差或分布差异；集成检测（ARF、AWE、AUE、DWM）采用加权多数投票、在线增删树/模型；并使用预训练+在线更新的学习器框架。

**📊 数据集**

使用的数据集：人工合成流（RT、Sine、Mixed、Aver）覆盖突变与渐变漂移；真实流数据（ELE、CIC 等）用于验证实用性；两类基学习器（NB 与 HT）用于评估模型表达能力对漂移检测的影响。

**📈 对比分析**

比较方法：以 AUC 评价指标对比不同检测器在各漂移类型和基学习器下的性能。结果显示：在突变漂移中，SPC 的 FTDD 最佳；渐变漂移中 EWMA/EDDM 较优；窗口化方法中 KSWIN/WSTD/D3 在突变漂移上表现最好；集成方法中 ARF+HT 在所有情形下取得最高 AUC，AUE+HT 在真实流上最突出。总体而言，集成方法优于 SPC 与窗口化。

**⚠️ 局限性**

局限性：①仅聚焦 learner‑based 检测，未对分布或混合检测进行深入比较；②实验主要基于单标签、低维流数据，缺乏对多类别/高维流的验证；③多数方法对参数敏感，缺乏自动调参机制；④真实流样本量有限，难以验证在大规模工业流中的可扩展性和实时性能。

---

## 486. AutoPass: Evidence-Guided LLM Agents for Compiler Performance Tuning

**arXiv ID:** 2606.20373 | [PDF](https://arxiv.org/pdf/2606.20373v1)

**作者:** Zepeng Li `[一作]` (Shaanxi Normal University), Zheng Wang `[通讯]` (University of Leeds)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多智能体框架，利用LLM在LLVM编译器内部通过编译器状态与运行时反馈对优化管道进行迭代调整，从而实现无训练、推理仅使用的编译器性能调优。

**💡 创新点**

创新点在于：①将LLM与编译器内部信号（IR、编译器备注）紧密耦合，打破了传统黑盒自动调优的限制；②通过多智能体（Score、Analysis、Reasoning、Evaluation）分工与迭代闭环，既能快速定位热点，又能利用实际运行反馈校正策略；③实现了推理仅、无离线训练的“零样本”调优方法，可直接迁移至新基准与硬件。

**🔧 技术方法**

使用了大型语言模型（DeepSeek‑V3.2、ChatGPT‑4o、Qwen3、Gemini‑3 Flash）作为推理引擎；LLM被设计成多智能体结构；对LLVM IR、编译器备注进行结构化提取；利用运行时性能计数与回归诊断实现反馈；在LLVM 17的新 Pass Manager 上实现编译管道编辑、校验与迭代。

**📊 数据集**

在服务器级 x86‑64（Intel Core i9）和 ARM‑64（Raspberry Pi 5 Cortex‑A76）两套硬件上，使用多套基准集：cBench、PolyBench、CoreMark、MiniFE、LULESH，以及 Qsort、BitCount 等示例，覆盖约 64 个工作负载。

**📈 对比分析**

与基准编译器 -O3、Instrumentation‑PGO、CSSPGO、AutoFDO、以及 OpenTuner（3 次迭代）对比。结果显示：在 x86‑64 上获得几何平均 1.043× 加速，在 ARM‑64 上获得 1.117× 加速；在 cBench 上几何平均 1.040×，在 ARM‑64 平台上 1.109×；整体表现优于所有 PGO 方案和受限预算的 OpenTuner，且无训练成本。

**⚠️ 局限性**

局限性包括：1）LLM 可能产生错误或无效的 Pass 名称，需要额外修复；2）对极大函数或超出上下文窗口的代码仍需 Score Agent 过滤；3）仍需在有限的编译‑运行预算内迭代，若预算不足可能导致收敛不充分；4）目前仅支持 LLVM Pass 管道编辑，对源代码级别的更改支持有限；5）在某些平台上初始推理效果不佳，需要多轮迭代修正。

---

## 487. SoftSkill: Behavioral Compression for Contextual Adaptation

**arXiv ID:** 2606.20333 | [PDF](https://arxiv.org/pdf/2606.20333v1)

**作者:** Xijia Tao `[一作]` (University of Hong Kong), Lingpeng Kong `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了将长自然语言技能压缩为短连续前缀（soft skill）的技术，并在冻结模型上训练该前缀以替代大量Markdown技能，实现推理时的行为压缩。

**💡 创新点**

创新点在于以自然语言技能为初始化，仅微调一个短的可学习前缀，从而在不更新模型权重的情况下获得与原始技能相当或更优的性能，同时显著减少技能上下文长度。

**🔧 技术方法**

采用的技术包括：soft‑prefix tuning（基于next‑token预测）、验证选择的检查点、文本/均值/SkillOpt初始化方法、以及与LoRA等参数高效微调方法的对比。

**📊 数据集**

实验数据集来自SkillOpt套件的六个低样本基准：SearchQA、LiveMath、DocVQA、OfficeQA、SpreadsheetBench、ALFWorld。

**📈 对比分析**

在单轮问答任务中，soft skill相较于无技能提升约8.3点(SearchQA)和42.1点(LiveMath)，并用32个虚拟token替代数百/千个Markdown技能；在agentic任务中效果有限，主要提升OfficeQA的准确率，但对Spreadsheet和ALFWorld几乎无提升。

**⚠️ 局限性**

限制包括：对长周期代理行为的压缩效果不足；需要白盒访问模型嵌入接口；可解释性差，跨模型迁移不稳定；性能受训练数据量和超参数设置影响显著。

---

## 488. Interpretable Sperm Morphology Classification via Attention-Guided Deep Learning

**arXiv ID:** 2606.20438 | [PDF](https://arxiv.org/pdf/2606.20438v1)

**作者:** Zahra Asghari Varzaneh `[一作]` (Malmö University), Lars Johansson `[通讯]` (NewLifeAid-Global AB)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一种基于 EfficientNet-B0 + CBAM 的注意力引导深度学习框架，用于自动化评估精子形态学，并通过 Grad-CAM++ 提供可解释性。

**💡 创新点**

创新点在于将通道与空间注意力模块 CBAM 与预训练 EfficientNet-B0 相结合，配合 freeze‑then‑unfreeze 训练策略和 MixUp 正则，显著提升小样本数据集的分类精度，同时通过 Grad‑CAM++ 生成的热力图实现模型决策可视化。

**🔧 技术方法**

技术包括 EfficientNet-B0 预训练模型、Convolutional Block Attention Module（CBAM）、Freeze‑then‑Unfreeze 微调、MixUp 与标签平滑正则、AdamW 优化器、余弦退火学习率、Grad‑CAM++ 可解释性方法。

**📊 数据集**

使用了公开的 SMIDS（3000 张微距图，3 类）和 HuSHem（216 张经专家验证的精子头图，4 类）数据集。

**📈 对比分析**

与 SimpleCNN 及标准 EfficientNet-B0 进行对比。模型在 SMIDS 上取得 90.21% 准确率、0.913 F1 分数；在 HuSHem 上取得 93.94% 准确率、0.948 F1 分数，明显优于基线并在小样本情境中实现显著提升。

**⚠️ 局限性**

主要局限是 HuSHem 测试集样本量较小（仅 33 张），可能导致评估结果的不确定性，且模型在极少样本情况下仍可能对背景噪声过拟合。

---

## 489. TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation

**arXiv ID:** 2606.20426 | [PDF](https://arxiv.org/pdf/2606.20426v1)

**作者:** Hengfei Zhao `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 TaCauchy，一个可扩展的有限元框架，将物理力学计算集成至 Isaac Sim，用于高保真视差触觉传感器仿真，并提供 Cauchy 应力、压力与切向力等真实物理量；

**💡 创新点**

创新点包括：① 直接从超弹性本构模型计算 Cauchy 应力并投影到接触面得到完整力分解；② 自动网格生成与几何自适应细化；③ 模块化传感器接口，支持 GelSight Mini、DIGIT、9DTact 等多种传感器；④ 将物理仿真与光学渲染耦合；⑤ 在 Isaac Sim 中实现高并行 FPS，满足大规模 RL 需求；

**🔧 技术方法**

使用技术主要有：有限元方法（FEM）+ UIPC 求解器、Stable Neo‑Hookean 超弹性模型、WildMeshing 自适应网格化、Cauchy 应力提取与分解、Isaac Sim/Isaac Lab、GPU 并行计算、光学渲染与光照模型；

**📊 数据集**

使用的数据集为：GelSight Mini、DIGIT、9DTact 的几何模型；通过 UR5 + M3733C 6 轴力/扭矩传感器的实验数据，包含 6 个不同施加力（1.26 N–4.73 N）的触觉图像；

**📈 对比分析**

对比方法：将仿真图像与真实实验图像按 SSIM、NCC、Histogram Correlation、PSNR 四指标进行定量比较；平均 SSIM 超过 0.93，PSNR 平均 22 dB，说明仿真与真实高度一致。性能方面，单环境 33.40 FPS，60 并行环境 555 FPS，力学提取耗时 <1 ms；

**⚠️ 局限性**

局限性：① 需要手工校准弹性材料参数，难以持续跟踪材料随时间的磨损、滞后变化；② 高分辨率 FEM 计算开销大，限制了在极大并行环境中的训练速度与规模。

---

## 490. Neural network surrogates with uncertainty quantification for inverse problems in partial differential equations

**arXiv ID:** 2606.20417 | [PDF](https://arxiv.org/pdf/2606.20417v1)

**作者:** Christian Jimenez-Beltran `[一作]` (University of Edinburgh), Konstantinos C. Zygalakis `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出DeepGaLA——一种基于Deep Galerkin和拉普拉斯近似的神经网络代理，用于求解逆问题中的偏微分方程并提供不确定性估计。

**💡 创新点**

创新点在于将拉普拉斯近似与Deep Galerkin相结合，首次为参数化PDE提供随机神经网络代理，并引入DA‑MCMC作为后验验证工具。

**🔧 技术方法**

使用Deep Galerkin法、拉普拉斯近似、有限差分/有限元前向求解、Gaussian过程以及MCMC（RWMH、DA‑MCMC）技术。

**📊 数据集**

在1D、2D椭圆问题及Navier‑Stokes逆问题上进行实验，使用FEM或伪谱求解作为真值，生成训练样本。

**📈 对比分析**

与基于物理信息Gaussian过程（PIGP）比较，DeepGaLA在高维参数下保持评估速度不变，误差与α_val指标随训练样本增加收敛；marginal近似在低数据时更稳健。

**⚠️ 局限性**

局限在于只对最后一层进行随机化，缺乏全贝叶斯网络的不确定性，且理论上对非线性PDE的收敛性尚无严格保证。

---

## 491. MaRDI Open Interfaces for Interoperable Nonlinear Optimization

**arXiv ID:** 2606.20410 | [PDF](https://arxiv.org/pdf/2606.20410v1)

**作者:** Dmitry I. Kabanov `[一作]` (University of Münster), Mario Ohlberger `[通讯]` (University of Münster)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 MaRDI Open Interfaces 软件包，实现跨语言统一接口和自动数据 marshalling，支持 ODE 初值问题和非线性优化，并通过对 Rosenbrock 函数的优化实验演示其功能。

**💡 创新点**

创新点在于通过动态加载、C 中间数据结构以及桥接、转换器、分派机制实现不同语言实现的解耦与统一接口，显著降低跨语言绑定成本，并在单进程内高效完成数据传输。

**🔧 技术方法**

采用 C 语言中间表示、动态库（Dispatch、Bridge 等）、Python、Julia、C 的桥接实现，配合自动化测试（Google Test、Testing.jl、Pytest）和 CI；在优化接口中实现 BFGS、Nelder–Mead 等算法调用。

**📊 数据集**

使用 Rosenbrock 函数（n=5、a=10）作为基准问题进行实验；未使用外部数据集。

**📈 对比分析**

通过 MaRDI Open Interfaces 调用 SciPy 与 Optim.jl 的 Nelder–Mead 与 BFGS，比较迭代次数、函数/梯度评估次数以及最终目标值。结果显示 BFGS 收敛更快、最终误差更小；同一算法在不同实现间仍存在显著差异。

**⚠️ 局限性**

局限在于需要了解各实现的细节（如停止准则）以保证公平比较；仅支持已实现的语言/接口；对大规模/高维问题的性能评估仍有限；输出参数的内存复制仍是潜在开销。

---

## 492. LLM agent safety, multi-turn red-teaming, jailbreak benchmarks, adversarial robustness, safety-critical systems

**arXiv ID:** 2606.20408 | [PDF](https://arxiv.org/pdf/2606.20408v1)

**作者:** Hanwool Lee `[一作]` (AIM Intelligence), Haon Park `[通讯]` (AIM Intelligence)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了NRT‑Bench，一个针对安全关键控制室中LLM操作员团队的多轮红队攻击基准，模拟核电站控制室并记录CSF损失作为客观伤害信号。

**💡 创新点**

引入多轮适应性攻击、团队角色结构、客观物理安全函数信号、可重放攻击数据与评估协议，并展示不同LLM模型的攻击易感性几乎不重叠，表明多模型多样性可实现防御宽度。

**🔧 技术方法**

结合LLM代理（ChatGPT、Claude、Gemma、Qwen3.5）、自定义核电站文本模拟器、十层可配置安全防御栈、基于HTTP的四通道攻击接口、可追溯日志与重放工具。

**📊 数据集**

使用从内部LLM红队代理生成的149条多轮攻击记录（及完整72条完整网格记录），包含攻击通道、情景、策略族、子目标等元信息。

**📈 对比分析**

采用固定攻击重放与配对重放两种协议，对四个LLM模型进行CSF损失率比较，发现单模型ASR在8.7–12.1%之间，攻击成功率与模型差异大，防御层对不同模型效果相反，展示了模型间不重叠的失败集。

**⚠️ 局限性**

①模拟器抽象性不等同真实核电物理；②攻击数据为固定重放，未覆盖所有自适应攻击；③单一种子与托管端点的随机性导致结果波动；④缺乏多场景覆盖与面板决策实际可行性评估。

---

## 493. CRAX: Fast Safe Reinforcement Learning Benchmarking

**arXiv ID:** 2606.20376 | [PDF](https://arxiv.org/pdf/2606.20376v1)

**作者:** Tristan Tomilin `[一作]` (Eindhoven University of Technology), Thiago D. Simão `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CRAX硬件加速安全强化学习基准，基于MuJoCo XLA（MJX）实现高保真3D物理仿真，并提供六套任务、三类代理及三级难度级别，支持GPU并行化；

**💡 创新点**

在安全RL领域首次结合GPU加速与可调安全阈值、难度阶梯、成本/奖励权衡，显著提升仿真速度（≈100–300倍），并引入多种成本约束类型，打造兼具可扩展性与安全可控性的基准；

**🔧 技术方法**

采用MuJoCo XLA物理引擎、JAX编程框架、GPU向量化仿真、强化学习算法（PPO、PPOCost、PPOLag、PPOPID、PPOSauté、P3O、FOCOPS），以及约束马尔可夫决策过程（CMDP）框架；

**📊 数据集**

自定义CRAX环境集（六套任务：Safe Navigation、Safe Velocity、Safe Pathway、Safe Reach、Safe Height、Safe Spider），每套包含三级难度，共同构成评估数据集；

**📈 对比分析**

以500M步骤、5个随机种子对比7种安全RL算法，评估奖励与成本；结果显示P3O和FOCOPS在大多数任务中获得最高奖励，PPOLag在安全性方面最高但奖励最低；Curriculum和安全迁移在特定环境可提升性能；CRAX在仿真吞吐量上比Safety-Gymnasium高约2–3个数量级；

**⚠️ 局限性**

局限包括MJX缺失某些MuJoCo功能限制场景表达；仅评估on‑policy方法，未覆盖离线或模型基础安全RL；仅考虑期望累计成本约束，未探究其他安全形式；仅基于状态观察、单智能体实验，未涉及像素观测、多智能体安全场景。

---

## 494. On the Variance of Temporal Difference Learning and its Reduction Using Control Variates

**arXiv ID:** 2606.20357 | [PDF](https://arxiv.org/pdf/2606.20357v1)

**作者:** Hsiao-Ru Pan `[一作]` (Max Planck Institute for Intelligent Systems), Bernhard Schölkopf `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

分析了基于阶段式 (phased) 表示的多步 TD 学习与 Monte Carlo (MC) 的方差特性，并阐明 TD 减少方差的机制是通过在更大数量的独立轨迹上进行聚合；同时将直接优势估计 (DAE) 视为回归调整的控制变量方法，证明其在大样本极限下比 TD 具有更紧的方差上界；通过链状环境实验对比了 TD、MC、DAE 等估计器的均方误差与贝尔曼误差，展示了不同步长、转移随机性、奖励遮蔽和动作覆盖率对方差的影响；

**💡 创新点**

提出了将优势函数作为控制变量以降低方差的思路，并将 DAE 等价为回归调整的控制变量；证明了多步 TD 的方差上界不超过 MC，且短步长更新具有更低方差；

**🔧 技术方法**

使用阶段式多步 TD、Monte Carlo 估计、控制变量分析、回归调整控制变量（DAE）、实验评估方法（均方误差、贝尔曼误差）

**📊 数据集**

使用自定义的链状 MDP 环境，参数包括状态数、动作数、奖励遮蔽概率、粘性转移概率等，进行 1000 次随机种子重复实验

**📈 对比分析**

通过均方误差对比实验显示：在确定性或奖励遮蔽环境下 TD 的方差与 MC 相当；在粘性转移环境下 TD 的方差被抑制且学习速率随步长 k 变化符合 bias‑variance 直觉；DAE 在大样本下实现方差更低，甚至在低动作覆盖情况下仍优于 TD，整体表现接近贝尔曼迭代

**⚠️ 局限性**

仅在阶段式 IID 采样下证明，未考虑马尔科夫采样；DAE 增加了存储和求解最小二乘的计算开销；理论结果仅在大样本极限下成立，对小样本偏差的严格分析仍缺失

---

## 495. Autonomous Driving with Priority-Ordered STL Specifications Under Multimodal Uncertainty

**arXiv ID:** 2606.20336 | [PDF](https://arxiv.org/pdf/2606.20336v1)

**作者:** Taha Bouzid `[一作]` (Eindhoven University of Technology), Sofie Haesaert `[通讯]` (Eindhoven University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种在多模态不确定性下满足优先级排序的 STL 规范的自动驾驶轨迹规划框架。

**💡 创新点**

创新点在于将 CVaR 风险度量与严格的层级优先级结合，并通过阶层保持奖励函数保证最高优先级规则在不确定性下的严格支配。

**🔧 技术方法**

核心技术包括基于情景树的离散化、多模态预测采样、CVaR 的经验估计、阶层保持奖励以及 MPPI 的实时采样优化。

**📊 数据集**

实验使用基于行驶模型的仿真数据，包括高速公路接管和行人穿行两种场景，分别构造五个前车切入情景和六个行人位置分布。

**📈 对比分析**

通过与不同 CVaR 水平及安全/舒适/目标等规则组合的比较，结果显示安全优先级下可在毫秒级（0.77 ms）到十毫秒级（7.53 ms）内完成规划，且更高 CVaR 水平能显著提升安全满足率。

**⚠️ 局限性**

局限性包括依赖离散情景采样导致对连续分布的近似，实验仅在仿真中验证，且假设自车动力学确定，未考虑模型误差与感知误差的耦合。

---

## 496. Constrained hybrid modelling to predict microbial dynamics and organic matter turnover in soil systems

**arXiv ID:** 2606.20329 | [PDF](https://arxiv.org/pdf/2606.20329v1)

**作者:** Paul Collart `[一作]` (Forschungszentrum Jülich GmbH), Lars Doorenbos `[通讯]` (University of Bonn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

结合神经网络与过程模型，利用宏基因组功能特征预测土壤微生物的生物动力学参数，并用这些参数驱动土壤碳循环模型。

**💡 创新点**

在混合建模框架中引入约束损失函数，将生态学理论与文献范围嵌入训练，显著降低模型等效性并提升未观测状态变量的可解释性。

**🔧 技术方法**

采用可微分过程模型（ODE求解器）、全连接多层感知器、Sigmoid映射与参数范围投影、加权自适应损失（自适应不确定性权重）以及软最大化的可微分约束。

**📊 数据集**

使用人工生成的合成数据（基于随机网络映射的基因组特征与参数集）和欧盟土壤数据库LUCAS的实测CO₂时间序列与宏基因组特征。

**📈 对比分析**

与无约束混合模型、全状态观测参考模型以及纯ML回归模型对比；在合成数据上，约束混合模型在CO₂拟合与隐藏状态均值误差、参数估计误差、约束违背量方面均优于无约束模型；在LUCAS实测上也保持更好的约束满足。

**⚠️ 局限性**

主要局限在于缺乏真实的深度宏基因组与过程率同步数据，约束设定对模型鲁棒性敏感，且模型对不同生态系统的泛化需要进一步验证。

---

## 497. Sparsity, Superposition, and Forgetting: A Mechanistic Study of Representation Retention in Continual Learning

**arXiv ID:** 2606.20431 | [PDF](https://arxiv.org/pdf/2606.20431v1)

**作者:** Jan Wasilewski `[一作]` (Rochester Institute of Technology), Bartosz Krawczyk `[通讯]` (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

构建了一个可控的玩具世界框架，用以精确测量并分析连续学习中的特征重叠(superposition)与遗忘(retention)的关系

**💡 创新点**

通过引入基于线性表征假设的特征定义、可测量的重叠度、特征保持动力学及任务级有效秩(metric)，首次实现了在无真实数据噪声的条件下对持续学习中遗忘机制的可观测性与可检验性

**🔧 技术方法**

使用合成的生成器–分离器管线、经验回放训练策略、SINDy（稀疏识别非线性动力学系统）来拟合特征保持与重叠之间的动力学关系，并采用有效秩评估任务级容量分配

**📊 数据集**

利用自定义的符号数据生成器，其中特征向量、标签生成规则均已知，能够在不同稀疏度与任务分布下生成任意数量的任务样本

**📈 对比分析**

将实验结果与传统连续学习评估（如任务准确率、遗忘度）进行对比，发现特征稀疏度升高会导致重叠增加，但只在特征表示弱且重叠高时显著加剧遗忘；有效秩随稀疏度上升而增加，说明模型在稀疏情境下更充分利用潜在维度

**⚠️ 局限性**

限制包括：特征稀疏度假设为均匀、任务特征子集互斥、SINDy的多项式库可能不足以捕捉所有非线性关系、并未直接将特征保持指标与最终分类性能建立定量关联

---

## 498. LIT-GS: LiDAR-Inertial-Thermal Gaussian Splatting for Illumination-Robust Mapping

**arXiv ID:** 2606.20424 | [PDF](https://arxiv.org/pdf/2606.20424v1)

**作者:** Shikuan Shi `[一作]` (Shenzhen University), Yukang Cui `[通讯]` (Shenzhen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LIT‑GS框架，将LiDAR平面约束注入热成像Gaussian Splatting的姿态/结构优化与高斯参数优化，实现鲁棒三维重建

**💡 创新点**

在热视觉监督下引入两级LiDAR平面几何约束，利用置信度感知的跨模锚点实现姿态/结构共优化，并在高斯渲染中加入点‑平面正则化，解决低纹理/光照变化下的几何漂移与表面膨胀

**🔧 技术方法**

使用3D Gaussian Splatting、可微分 splatting、LiDAR点‑平面约束、COLMAP‑PCD bundle adjustment、SuperPoint+SuperGlue匹配、CLAHE增强、运动自适应权重以及光度与几何联合损失的端到端优化

**📊 数据集**

自建多时段热‑LiDAR组合序列以及公开M2DGR数据集

**📈 对比分析**

与RGB‑LiDAR基线(LIV‑GaussMap)和纯热基线(Thermal3D‑GS)比较，LIT‑GS在PSNR/SSIM/LPIPS/EMD等指标上均优于两者，尤其在低光、强光和阴影场景下显著降低几何误差

**⚠️ 局限性**

仍需高质量LiDAR约束，热图特征匹配受低温梯度限制，对大规模实时在线映射的计算成本和高速运动时的动态遮挡仍存在挑战

---

## 499. On the Redundancy of Timestep Embeddings in Diffusion Models

**arXiv ID:** 2606.20416 | [PDF](https://arxiv.org/pdf/2606.20416v1)

**作者:** José A. Chávez `[一作]` `[通讯]` (Independent Researcher), José A. Chávez (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了扩散模型中时间步嵌入的必要性，删除嵌入后评估其对生成质量与效率的影响。

**💡 创新点**

证明时间步嵌入在高维条件下是冗余的，模型可从噪声样本中隐式推断噪声尺度，并通过理论与实验验证这一结论。

**🔧 技术方法**

采用扩散模型（U‑Net 与 Diffusion Transformer DiT）进行消融实验，并用分数函数、集中性分析等理论工具分析时间信息冗余。

**📊 数据集**

实验基于 CelebA（128×128）与 CIFAR‑10（32×32）两个公开数据集。

**📈 对比分析**

通过 FID、Precision、Recall 以及推理时间等指标比较，发现无时间步嵌入的模型与传统模型性能相当甚至更优，并且推理速度更快。

**⚠️ 局限性**

局限性在于仅验证了低至中等分辨率的数据集，尚未探讨更高分辨率或更复杂任务下时间步嵌入是否仍可忽略；理论假设需要数据的二阶矩分布满足 C≠1 条件。

---

## 500. Pseudo-Feature Padding: A Lightweight Defense Against False Data Injection in Power Grids

**arXiv ID:** 2606.20415 | [PDF](https://arxiv.org/pdf/2606.20415v1)

**作者:** Farhin Farhad Riya `[一作]` (University of Tennessee), Kevin Tomsovic `[通讯]` (Clemson University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于伪特征填充的轻量级防御框架，用于提升电力系统FDIA检测深度神经网络的鲁棒性。

**💡 创新点**

创新点在于动态生成低重要性特征的随机伪特征填充，改变投影矩阵使攻击向量无法满足零空间条件，显著提升对抗鲁棒性。

**🔧 技术方法**

采用树模型特征重要性评估、统计分布拟合与伪特征采样，结合深度神经网络训练与推理时的随机填充技术。

**📊 数据集**

使用IEEE 14、30、118、300节点电网的DC负荷流模拟数据，构建了正负样本数据集。

**📈 对比分析**

与对抗训练、蒸馏、零填充等传统方法对比，攻击检测准确率提升至约94–96%，仅略微增加训练时间，推理开销保持低。

**⚠️ 局限性**

局限性包括在极端白盒攻击或伪特征采样不足时可能被逆向利用，且对大规模实时系统的部署效果尚未充分验证。

---

## 501. Direct Advantage Estimation for Scalable and Sample-efficient Deep Reinforcement Learning

**arXiv ID:** 2606.20411 | [PDF](https://arxiv.org/pdf/2606.20411v1)

**作者:** Hsiao-Ru Pan `[一作]` (Max Planck Institute for Intelligent Systems), Bernhard Schölkopf `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

将Direct Advantage Estimation（DAE）扩展到部分可观测环境（POMDP），并通过离散潜在动力学模型降低其计算复杂度。

**💡 创新点**

创新点在于：①在POMDP中构造返回分解并证明其可用于离线评估；②用低维潜在空间的WTA+VQ‑VAE实现对环境随机性的逼近，从而避免高维生成模型；③在同一框架下实现多步离线学习与POMDP修正。

**🔧 技术方法**

主要技术包括：深度强化学习中的DAE目标、信息向量（历史）表示、LSTM+卷积编码器、离散潜在动力学模型（WTA + VQ‑VAE）、多步回溯、离线样本优先级重放。

**📊 数据集**

在Arcade Learning Environment（ALE）47个游戏环境上进行实验，使用与Dopamine相同的评估协议（sticky actions等）。

**📈 对比分析**

与Rainbow DQN、规模化Rainbow和DreamerV3比较，DAE在网络宽度m=8时能在仅用10%训练帧数的情况下达到或超过Rainbow的表现，并在多步、离线校正和POMDP修正的推动下显著提升样本效率。

**⚠️ 局限性**

局限性包括：需要学习和维护潜在动力学模型，带来额外超参数和实现复杂度；目前仅针对POMDP修正，未探索对模型规模化的更系统研究；潜在模型逼近误差仍可能影响约束满足程度。

---

## 502. The Significance of Style Diversity in Annotation-Free Synthetic Data Generation

**arXiv ID:** 2606.20400 | [PDF](https://arxiv.org/pdf/2606.20400v1)

**作者:** Zahra Abbasiantaeb `[一作]` (University of Amsterdam), Mohammad Aliannejadi `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个完全无人工标注的合成对话生成框架，仅使用意图定义通过大语言模型生成多轮意图对话，并通过主题与风格属性提升数据多样性。

**💡 创新点**

创新点包括：① 将主题与风格属性分离并系统评估其对数据效用的影响；② 提出两种后置风格化模型（Universal 与 Example-based）实现对 LLM 生成文本的语言风格迁移；③ 采用 LLM-as-judge 过滤低质量样本；④ 证明风格多样性对合成数据效用的决定性作用。

**🔧 技术方法**

主要技术包括：GPT‑4.1‑mini 进行零样本对话生成；T5（及其变体）实现风格化；另用 LLM 作为判别器过滤；属性驱动的 Prompt 设计；G‑Vendi 评估合成数据对下游任务的影响；BLEU、准确率等指标评估风格化质量。

**📊 数据集**

实验使用公开的 Schema‑Guided Dialogue (SGD) 数据集和自有的 Enterprise Intent Corpus (EIC) 两大数据集；合成风格化数据集规模分别为 23k/4.8k/4.6k（训练/测试/验证）和 39.9k/9.3k/8.8k。

**📈 对比分析**

与无属性生成、仅主题生成、StylePTB 风格化、以及真实人工标注数据等基线进行对比。结果显示，合成数据在 SGD 上可达 90.7%（与人工标注一致），在 EIC 上可达 93.3%；风格属性对性能提升最为显著；主题属性对 SGD 有轻微收益，对 EIC 则略逊；后置风格化模型显著提升数据效用，但直接在生成时加入风格属性效果更佳。

**⚠️ 局限性**

局限性：仅在意图分类任务上验证，仅使用 GPT‑4.1‑mini 进行对话生成；未对其他 LLM、其他意图分类模型或更复杂对话任务进行评估；风格多样性重要性的结论仅针对意图分类，需进一步验证。

---

## 503. Linked Fates: How Small of an Ambiguity Increase Can Make the Difference Between Equaling and Separating from P?

**arXiv ID:** 2606.20399 | [PDF](https://arxiv.org/pdf/2606.20399v1)

**作者:** Benjamin Carleton `[一作]`, Melissa Welsh `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了NP的可决算子约束版本f(n)，探讨不同约束下的类是否会共同“落在”P或保持分离，提出新的相连命运（linked‑fates）判定标准，并证明除了常数约束外几乎所有情况下无法得到鲁棒的相连结论。

**💡 创新点**

创新点在于：① 引入了“路径毒化（path‑poisoning）”技术，能够在超常数模糊度（super‑constant ambiguity）下实现语言归约；② 利用填充（padding）技巧将f(n)提升到f(n^k)，从而得到新的相连命运对；③ 通过综合上述两种方法给出了新的正向结果，并证明其在任何可相对化世界中都是最优的。

**🔧 技术方法**

主要技术手段包括：
- 可相对化的oracle构造与层级化阶段（stage）技术；
- 路径毒化（path‑poisoning）来控制接受路径数量；
- 填充（padding）技术来扩大输入长度并保持模糊度不变；
- 复杂度类的等价性与分离证明，利用已知的Watanabe结果与Hartmanis‑Hemachandra引理。

**📊 数据集**

无数据集；全部结果均为理论证明与抽象构造，无实验数据。

**📈 对比分析**

本工作不涉及实验比较与性能评估；其成果以数学定理与反例oracle构造的形式给出，证明了在所有可相对化世界中的鲁棒性或缺失。

**⚠️ 局限性**

局限性：
- 对于超出非递增、可计算且不在2^n^Ω(1)范围的模糊度函数，结果仍未给出；
- 仍未能将结果推广至多项式层次（PH）等更高级结构；
- 现有方法无法处理所有非常数模糊度情况的鲁棒相连结论，仍有开放问题。

---

## 504. CATCH-ME if you RAG: a dataset of Contextually Annotated multi-Turn Counterspeech against Hate and Misinformation Exchanges

**arXiv ID:** 2606.20369 | [PDF](https://arxiv.org/pdf/2606.20369v1)

**作者:** Helena Bonaldi `[一作]` (Fondazione Bruno Kessler), Marco Guerini `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了首个多语言、多轮、专家编写、以事实核查文章与NGO报告为外部知识的对抗仇恨与错误信息交叉的对话数据集CATCH‑ME。

**💡 创新点**

创新点在于同时对抗仇恨与错误信息、跨语言、多轮结构、外部知识跨度注释，并提供高质量的检索与生成基准。

**🔧 技术方法**

采用GPT‑4o mini、Llama 3.1、Qwen3‑Embedding等LLM以及BM25、BGE‑M3检索，并结合预编译、交互、手工与翻译等人机协作策略完成数据收集与评估。

**📊 数据集**

使用由23名专家编写的2,015条多轮对话（共12,298回合）和基于事实核查文章与NGO反歧视报告的外部知识，覆盖英语、意大利语、马耳他语、波兰语与西班牙语。

**📈 对比分析**

在零样本检索与生成基准中，Qwen3‑Embedding与BGE‑M3在多语言检索上显著优于BM25，Qwen3 8B的检索式生成在语义相似度、事实性和相关性上接近专家手写，对话质量明显提升。

**⚠️ 局限性**

局限性包括语言与目标群体分布不均、对话为人为构造缺乏自然多样性、检索精度对生成质量影响大、仅提供自动化评估未覆盖人工评测。

---

## 505. Judging to Improve: A De-biased VLM-as-3D-Judge Protocol for Single-Image 3D Generation

**arXiv ID:** 2606.20364 | [PDF](https://arxiv.org/pdf/2606.20364v1)

**作者:** Ali Asaria `[一作]` (Transformer Lab), Deep Gandhi `[通讯]` (Transformer Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种基于无标签轻量化专化的单图像3D生成方法，并针对优化循环设计了跨模型去偏VLM判定协议；

**💡 创新点**

在判定协议中硬化了跨模型、位置偏差校正以及三种失效模式的修复，并构造了可学习的质量对比偏好信号；

**🔧 技术方法**

使用VLM判定（Qwen2.5‑VL 与 InternVL3）、DPO/ORPO/LoRA/SFT 等参数高效微调技术、流式采样与DINOv2特征修复等方法；

**📊 数据集**

采用公开的3D‑FUTURE家具渲染数据以及合成降质变换（裁剪/遮挡/缩放/模糊）和TRELLIS的公开训练数据；

**📈 对比分析**

通过跨模型判定的 win‑rate 与基线比较，在清洗与降质两种输入、六种适配方法下，最高在严重降质下达到 0.50 的平衡（与基线持平），未能突破 65% 的目标；

**⚠️ 局限性**

限制包括样本量小、仅单一基准与资产类别、仅轻量 PEFT、仅公开数据与 VLM 判定、缺乏人工评估；更大规模或专有数据可能突破当前局限。

---

## 506. Train, Retrieve, or Both? A Four-Arm Head-to-Head for Correct Statutory Citation on the Ontario Residential Tenancies Act

**arXiv ID:** 2606.20359 | [PDF](https://arxiv.org/pdf/2606.20359v1)

**作者:** Ali Asaria `[一作]` (Transformer Lab), Deep Gandhi `[通讯]` (Transformer Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个问答系统，用于将自代表租户、房东和帮助台人员的自然语言提问映射到安大略省《住宅租赁法》（RTA）中对应的条文，并在四种模型配置下评估其引用精确度。

**💡 创新点**

创新点在于：① 将“训练 vs. 检索”问题以实证方式拆分为四种对比实验；② 采用轻量级的LoRA微调与BM25+密集检索的混合检索策略，实现零假引发且提升引用准确率；③ 在短法律文本上展示检索模型的“无效提升”现象，证明在低预算、低延迟环境下使用轻量检索即可获得最佳效果。

**🔧 技术方法**

技术包括：Qwen2.5-7B-Instruct基础模型、LoRA微调、BM25+密集向量检索、可选的交叉编码reranker、检索后存在检查的硬性假设过滤、以及基于部分信用的精确匹配评估。

**📊 数据集**

数据集：1) 约2,148条合成的问答–条文对，来自对RTA条文的自然语言改写；2) 27条真实评测问题（每条均引用合法条文），待人工验证；3) 330条未出现于训练的“未见条文”切片用于检验泛化。

**📈 对比分析**

比较方法：在同一27条评测集上，计算引用的精确匹配（区分节/小节、部分信用）、F1、假引发率以及检索召回@k。结果显示：基线零召回/假引发；LoRA微调仅提升至0.148精确匹配；单独检索得到0.44；混合SFT+RAG最高达0.481精确匹配（0.574 F1）且假引发率为0。该混合方案提升幅度为+0.48，远超基线，但仍未达到0.70的目标。

**⚠️ 局限性**

局限性：评测集极小（27条）且待人工验证，导致统计不稳；模型只在单一司法管辖区和单一法律文本上验证，缺乏跨域可推广性；检索召回上限约0.89，限制了最大可实现精确匹配；合成训练数据可能引入偏差；最终混合方案的优势在当前样本量下仅为噪声级别。

---

## 507. A cubical formalisation of conditional independence, Bayesian conditioning, and Pearl's d-separation soundness

**arXiv ID:** 2606.20351 | [PDF](https://arxiv.org/pdf/2606.20351v1)

**作者:** Karen Sargsyan `[一作]` `[通讯]` (Academia Sinica), Karen Sargsyan (Academia Sinica)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在立方体型 Agda 中构造有限概率分布的更高归纳类型（HIT），将条件独立性定义为分布之间的路径，而非点值的等式，利用该框架在无后置公理的情况下构造性地证明半格罗波德公理、交叉公理、Pearl 的三条 do‑calculus 规则、全支持条件化以及 DAG 的 d‑separation 可靠性；同时把该分布 monad 证明为 Markov 类别。

**💡 创新点**

创新点包括：① 使用 HIT 和立方体型路径实现分布型条件独立性，获得可运行的证明与计算；② 在无正性假设的情况下给出交叉公理的构造性证明，避免传统的除零和正性假设；③ 发现标准的凸代数交叉公理不足以支持完整的贝叶斯条件化，提出了可变内权重的广义交叉公理，并证明其兼容原始形式；④ 把整个概率框架与 Markov 类别对齐，展示了其在合成概率学中的应用。

**🔧 技术方法**

主要技术：立方体型类型理论、HIT（高阶递归与路径构造子）、路径型条件独立性、构造性贝叶斯条件化、半格罗波德公理与交叉公理的类型级证明、Pearl do‑calculus 的核形式验证、Markov 类别公理化、Agda 归一化与类型检查。

**📊 数据集**

本工作不使用外部数据集；所有证明均在抽象的有序域接口上完成，最终实例化于有理数（ℚ）的闭区间，构造了自由凸代数。

**📈 对比分析**

由于研究重点是形式化证明而非算法效率，论文不涉及实验比较或性能评测；主要比较点是：在仅依赖有序域接口的前提下实现了所有核心公理与推理规则，无需额外后置公理；证明在 Cubical Agda 下完全类型检查，且所有计算均可在归一化器上执行。

**⚠️ 局限性**

局限性：仅覆盖有限支持的离散分布；不证明 d‑separation 的完整性；未处理连续分布、逆向提取结构独立性的构造；未给出高阶概率程序、数据驱动因果发现或无测量混杂的完整实现。

---

## 508. Critical Percolation as a Synthetic Data Model for Interpretability

**arXiv ID:** 2606.20347 | [PDF](https://arxiv.org/pdf/2606.20347v1)

**作者:** Aryeh Brill `[一作]` (Principles of Intelligence), Tom Ingebretsen Carlson `[通讯]` (Principles of Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于临界均值场渗流的合成数据模型，用于可解释性研究，并实现了生成该数据的近线性时间算法。

**💡 创新点**

创新点在于将渗流簇与随机树、加法凝聚的映射结合，设计循环凝聚算法实现高效采样，同时通过线性探针验证神经网络能线性解码其层次潜变量。

**🔧 技术方法**

使用了均值场渗流理论、随机树与加法凝聚映射、并查集实现的循环凝聚算法、残差多层感知机以及线性探针等技术。

**📊 数据集**

使用自生成的单簇（2×10^5 点）和多簇（2×10^6 点）渗流数据集，嵌入维度可调，标签为线性组合的层次潜变量。

**📈 对比分析**

通过将 MLP 与 Ridge 回归和 1NN 进行对比，MLP 的 R^2 和 MSE 接近 1NN，表明模型已近似最优；线性探针能在网络激活中线性恢复潜变量，误差随潜变量重要性呈幂律下降。

**⚠️ 局限性**

局限性包括仅支持向量化输入与标量回归，未建模特征的组合关系；缺乏因果干预验证；只研究临界状态，未涵盖超临界或有环簇的情况。

---

## 509. Directors Duties in the Age of Agentic Artificial Intelligence

**arXiv ID:** 2606.20453 | [PDF](https://arxiv.org/pdf/2606.20453v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 510. Spectral Query-Key Product Weight Steering for Training-Free VLM Hallucination Mitigation

**arXiv ID:** 2606.20419 | [PDF](https://arxiv.org/pdf/2606.20419v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 511. Topological Data Analysis for High-Dimensional Dynamic Process Monitoring

**arXiv ID:** 2606.20443 | [PDF](https://arxiv.org/pdf/2606.20443v1)

**作者:** Angan Mukherjee `[一作]` (University of Wisconsin-Madison), Victor M. Zavala `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现一种结合拓扑数据分析与神经常微分方程的实时多变量过程监测框架。

**💡 创新点**

将 Euler 特征曲线的时序演化作为隐状态，利用 NODE 捕捉其方向性，从而同时检测主事件与细微事件。

**🔧 技术方法**

使用拓扑数据分析（Euler 特征曲线）、滑动窗口、神经 ODE、PCA、AE 与 Koopman 自编码器做对比。

**📊 数据集**

采用真实的烷烃裂解工厂 56 维传感器数据，约 63 天的时间序列。

**📈 对比分析**

通过与 PCA、AE、KAE 的对比实验显示，TDA‑NODE 在检测主事件和次事件时具有更高的灵敏度与更低的误报率，并在独立测试集上保持了类似性能。

**⚠️ 局限性**

对极小规模、持续时间极短的异常仍有检测限制；模型对窗口大小与预训练数据依赖较大，且在不同工艺上的迁移性尚未验证。

---

## 512. Evolutionary Two-Stage Hyperparameter Optimization Strategies for Physics-Informed Neural Networks

**arXiv ID:** 2606.20442 | [PDF](https://arxiv.org/pdf/2606.20442v1)

**作者:** Fedor Buzaev `[一作]` (HSE University), Fedor Ratnikov `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于演化算法的两阶段超参数优化框架，用于提升 Physics‑Informed Neural Networks (PINNs) 的求解精度和训练鲁棒性

**💡 创新点**

创新点在于将全局演化搜索与低精度探索/高精度精炼相结合，形成可在固定计算预算下高效筛选并训练最佳 PINN 配置的方法；并给出探索预算与最终误差之间的经验性规律

**🔧 技术方法**

使用差分进化变体（JADE、LSHADE）以及灰狼优化、鲸鱼优化等演化算法进行超参数搜索，并通过梯度优化（AdamW）完成权重训练；同时利用残差损失与 RMSE 作为评价指标

**📊 数据集**

在三类经典 PDE 基准上进行实验：阿德维克斯（Advection）、Klein‑Gordon、Helmholtz 方程；使用人工生成的解析解构造训练/验证数据集

**📈 对比分析**

与传统的网格搜索、随机搜索、贝叶斯优化以及 Nelder–Mead 进行对比，实验表明演化搜索在相同计算预算下能将最终 RMSE 降低 28%–77%，在多数实验中明显优于基线方法

**⚠️ 局限性**

限制包括：缺乏对更大搜索空间（更复杂网络结构）和更广泛 PDE 类型的验证；探索阶段的低精度训练可能误判某些超参数组合；且演化算法本身对种群规模和迭代次数敏感，需要手动调节

---

## 513. ExSpike: A General Full-Event Neuromorphic Architecture for Exploiting Irregular Sparsity with Event Compression

**arXiv ID:** 2606.20414 | [PDF](https://arxiv.org/pdf/2606.20414v1)

**作者:** Yuehai Chen `[一作]` (University of Groningen), Farhad Merchant `[通讯]` (University of Groningen)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 ExSpike 的全事件神经形态加速器，通过数据流与硬件协同设计，充分利用 SNN 的不规则稀疏性实现高效推理。

**💡 创新点**

创新点包括：1) 直接编码、卷积、平均池化/全连接的纯事件驱动数据流优化；2) 引入自注意力核心支持 Spike‑driven Self‑Attention；3) 开发邻近位置事件压缩（APEC）技术，减少冗余累加。

**🔧 技术方法**

采用事件驱动卷积、事件驱动平均池化/全连接、Spike‑driven Self‑Attention 核心、邻近位置事件压缩等技术，并在 Xilinx Virtex‑7 FPGA 上实现，使用 8‑bit 固定点权重、16‑bit 电位器，配合 WPE/MPE/FPE 计算单元和 AER FIFO 等硬件结构。

**📊 数据集**

使用 CIFAR‑10、CIFAR‑100 进行分类实验，MLND‑Capstone SegNet 数据集进行分割实验。

**📈 对比分析**

与多款主流 FPGA SNN 加速器（NEURAL、STISA、DeepFire2、SpikeTA、Cerebron、FireFly‑T、SConvNSys）以及 CPU/GPU 进行对比，ExSpike 在保持相近或更高精度的同时，提供最高的 GOPS/W/PE（最高 0.80）和 GOPS/W/kLUT（最高 7.05），比 FireFly‑T 高约 10×能效，并实现最高 479.15 GOPS、281.85 GOPS/W 的吞吐率。

**⚠️ 局限性**

局限性主要体现在 APEC 需要额外缓冲导致资源与功耗略升高；在某些层权重就绪延迟占比大时压缩收益有限；目前仅在 FPGA 上验证，ASIC 化与更大规模模型的适配仍待进一步研究。

---

## 514. DataMagic: Transforming Tabular Data into Data Insight Video

**arXiv ID:** 2606.20388 | [PDF](https://arxiv.org/pdf/2606.20388v1)

**作者:** Yupeng Xie `[一作]` (Hong Kong University of Science and Technology), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一套端到端交互系统DataMagic，能够将原始表格数据和自然语言查询自动转换为结构化、可交互的叙事数据视频。

**💡 创新点**

核心创新在于设计了面向数据视频的声明式规范(Data Video Specification)，通过数据驱动的语义引用和叙事索引触发机制实现数据完整性与时间同步；以及采用Generate‑then‑Orchestrate多智能体架构，先并行生成候选场景再进行全局叙事优化。

**🔧 技术方法**

技术手段包括：多智能体生成管道（Story Planner、Data Manager、Visual Designer、Narration Director、Animation Coordinator）、声明式规范与语义引用、叙事索引触发、D3.js渲染、Remotion视频合成、文本到语音(TTS)、LLM驱动的叙事与问答、结构化数据血缘追溯。

**📊 数据集**

使用了109个真实世界样本，来源于DAComp‑DA和T2R‑bench数据集，涵盖业务报表、财务数据等多种领域。

**📈 对比分析**

通过与四种主流LLM（DeepSeek‑V3.2、Gemini‑2.5‑Pro、GPT‑5、Claude‑Sonnet‑4）直接生成视频的基准对比，并用Gemini‑2.5‑Pro作为自动评判器，评估执行率和五项质量维度。DataMagic将平均质量从2.0级提升至3.89级，执行率超过95%，在动画效果和叙事质量方面提升显著。

**⚠️ 局限性**

局限性主要包括：仍依赖外部LLM，导致API调用成本与延迟；对仅表格型数据的支持，无法直接处理非结构化或多维数据；生成后仍需人工微调；在极端数据量大或复杂叙事场景下的可扩展性尚未验证。

---

## 515. Towards Modality-imbalanced Federated Graph Learning: A Data Synthesis-based Approach

**arXiv ID:** 2606.20382 | [PDF](https://arxiv.org/pdf/2606.20382v1)

**作者:** Zhengyu Wu `[一作]`, Guoren Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 FedMGS 框架，解决多模态图联邦学习中的客户端和节点级模态不平衡，通过隐式语义合成实现缺失模态的图感知恢复。

**💡 创新点**

创新点在于将缺失模态视为图感知的隐式语义合成，结合可用性感知图编码、跨客户端原型引导的语义合成以及可靠性校准融合，避免传统特征重建并充分利用图结构。

**🔧 技术方法**

使用图神经网络的可用性门控编码、原型聚合与潜在空间合成网络、可靠性校准权重，嵌入联邦训练框架并通过原型统计实现通信。

**📊 数据集**

采用多模态图数据集包括 Movies、Grocery、Toys、Flickr30k、DY、Bili Dance、KU、Bili Food，分别对应节点分类、链接预测、模态匹配与模态检索四类任务。

**📈 对比分析**

与多种 FGL、缺失模态 FL 和原型 FL 基线对比，FedMGS 在四项任务上均取得最优表现，节点分类提升约17%最高，检索/匹配提升3–4%，同时保持最佳的效率-性能折中。

**⚠️ 局限性**

局限性在于依赖已知类别标签或语义锚点的原型聚合，缺乏对极端稀疏或动态模态缺失的理论保障，且在高异构场景下可靠性评估仍需进一步研究。

---

## 516. Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe

**arXiv ID:** 2606.20381 | [PDF](https://arxiv.org/pdf/2606.20381v1)

**作者:** Qian Zhao `[一作]` (Ant Group), Jun Zhou `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种基于统一4-bit E1M2/INT4网格的训练方法，该方法在所有线性层的前向、梯度和权重梯度上都使用随机Hadamard变换，显著降低了大规模MoE预训练中的BF16相对损失。

**💡 创新点**

核心创新在于发现并量化了非均匀E2M1格式的Shrinkage Bias，证明了采用均匀网格并全局应用RHT可以消除该系统性偏差，从而在FP4训练中实现更高的量化质量。

**🔧 技术方法**

技术手段包括块级量化、随机Hadamard变换、针对dY的随机舍入以及E1M2/INT4统一4-bit网格的设计。

**📊 数据集**

使用Dense 1.5B、MoE 7.9B和MoE 124B三种大模型的长跑预训练任务（通用文本数据集）进行实验验证。

**📈 对比分析**

通过与BF16、E2M1参考方案对比，评估BF16相对损失、SQNR、有效桶比、以及长跑训练曲线，结果显示E1M2方案在BF16误差上下降约20–30%，并保持与BF16相近的性能。

**⚠️ 局限性**

局限性包括仍存在与BF16的残差gap，且该方法仅在支持E1M2/INT4网格的硬件上可实现，对非均匀E2M1格式在某些场景下的适用性未完全消除。

---

## 517. ARC: Adaptive Robust Joint State and Covariance Estimation

**arXiv ID:** 2606.20428 | [PDF](https://arxiv.org/pdf/2606.20428v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 518. Computing Twin-Width via Treedepth and Vertex Integrity

**arXiv ID:** 2606.20331 | [PDF](https://arxiv.org/pdf/2606.20331v1)

**作者:** Robert Ganian `[一作]` (TU Wien), Mathis Rocton `[通讯]` (TU Wien)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了两项关于图的twin-width参数的固定参数化算法：①在树深度（treedepth）为参数时，给出一个以2-近似的FPT算法；②在顶点完整性（vertex integrity）为参数时，给出一个可以精确计算twin-width的FPT算法。

**💡 创新点**

创新点主要包括：①首次利用定向twin-width（oriented twin-width）作为中间量，证明其与twin-width函数等价并可更易算法化；②通过“冗余子图”递归删减与合并策略，突破以往仅基于删除距离的可行性限制；③结合Ramsey超图论与Ramsey pruning技术，构造可在顶点完整性参数下将子图重构到原图的算法。

**🔧 技术方法**

使用的技术包括：
- 定向twin-width的定义与转换算法；
- 树深度分解与同义块（twin-block）识别；
- 递归删减规则与重构规则；
- Ramsey超图理论与Ramsey pruning用于构造均匀大组；
- 结构化的合并段（X_v-merge）与块（block）的分析；
- 组合与插值技术将子图的收缩序列映射回原图。

**📊 数据集**

该工作完全是理论性的，没有使用任何实验数据集；所有结论均在形式化的图论模型与算法分析框架内给出。

**📈 对比分析**

由于此前不存在关于twin-width的可计算算法，论文并未与已有算法直接比较；作者通过证明“在树深度或顶点完整性参数下可得到FPT近似/精确算法”来突破长期存在的NP难度壁垒。算法复杂度为可计算函数f(p)乘以多项式，其中f(p)涉及指数塔（2↑↑p）或更高阶的多重指数；近似因子为2-近似，随后转换为精确序列时的宽度约为2^2^2^t。

**⚠️ 局限性**

限制与挑战：
- 算法运行时间包含极高阶指数塔，实际可实现性有限；
- 近似算法仅在树深度参数下提供2-近似，且转换成标准twin-width时宽度仅上界为双重指数；
- 精确算法仅在顶点完整性参数下可行，仍无法直接针对twin-width本身的参数；
- 证明过程高度依赖结构化的定向合并与Ramsey理论，难以进一步简化；
- 由于缺乏实验验证，算法在实际图实例上的性能尚未评估。

---

## 519. Multi-View Decompilation for LLM-Based Malware Classification

**arXiv ID:** 2606.20436 | [PDF](https://arxiv.org/pdf/2606.20436v1)

**作者:** Bercan Turkmen `[一作]` (Independent Researcher), Vyas Raina `[通讯]` (SPARK)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在无源码条件下，利用大型语言模型对Ghidra与RetDec生成的伪C代码进行恶意/良性二分类，并验证多视图（双反编译器）能提升分类效果。

**💡 创新点**

首次提出并验证无训练、低成本的多反编译器视图融合策略，显著提升LLM在恶意软件识别中的召回率与F1。

**🔧 技术方法**

使用指令调优的大型语言模型（Gemini、GPT‑3.5、Claude、Qwen、LLaMA）以及Ghidra与RetDec反编译器，并设计单视图、双视图与争议触发一致性推理的prompt。

**📊 数据集**

构建了一个100个C程序（50良性、50恶意）的基准，分别编译为64‑bit ELF对象，并用Ghidra与RetDec生成匹配的伪C视图。

**📈 对比分析**

在准确率、精确率、召回率及恶意类F1上进行比较，实验显示多视图模式在大多数模型中显著提升F1（最高提升约+13.9%），尤其改善召回率；单视图性能受反编译器差异影响。

**⚠️ 局限性**

仅适用于无加壳、无混淆的对象文件；未涵盖完整可执行文件或其他商业反编译工具；仅使用两种开源反编译器，可能无法代表更广泛工具的多样性。

---

## 520. FlowBender: Feedback-Aware Training for Self-Correcting Conditional Flows

**arXiv ID:** 2606.20404 | [PDF](https://arxiv.org/pdf/2606.20404v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 521. MixProLAP: Mixture-Induced Uncertainty Modeling for Probabilistic Language-Audio Pretraining

**arXiv ID:** 2606.20418 | [PDF](https://arxiv.org/pdf/2606.20418v1)

**作者:** Yu Nakagome `[一作]` (LINE WORKS Corporation), Soo-Whan Chung `[通讯]` (NAVER Cloud Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个概率音频-文本预训练框架，采用音频混合而非掩蔽来引入不确定性，并加入多层次包含损失。

**💡 创新点**

用音频混合产生语义超集，并在概率分布上进行包含关系建模，提出多级混合比例的包含损失，解决传统掩蔽在音频上的局限。

**🔧 技术方法**

概率表示学习（高斯分布）、PPCL、闭式采样距离（CSD）、跨模态包含损失、多级包含损失、VIB、CLAP、HTS-AT、GPT-2、AdamW优化等。

**📊 数据集**

AudioCaps 与 ClothoV2 音频-文字数据集。

**📈 对比分析**

与微调 CLAP 基线在零样本音频-文本检索任务（Recall@1/10、mAP@10）进行对比，方法在 A→T 与 T→A 两方向均取得显著提升，尤其在跨域检索中表现更佳。

**⚠️ 局限性**

混合方式对文本不确定性处理仍不充分，T→A 检索有时略逊于 CLAP，且对长音频或多事件复杂度的建模仍存在局限。

---

## 522. Automating SKILL.md Generation for Computer-Using Agents via Interaction Trajectory Mining

**arXiv ID:** 2606.20363 | [PDF](https://arxiv.org/pdf/2606.20363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 523. Geometry-Aware Superpixel Graph Transformer with Metadata for Skin Lesion Classification

**arXiv ID:** 2606.20390 | [PDF](https://arxiv.org/pdf/2606.20390v1)

**作者:** Muhammad Azeem `[一作]` (Edge Hill University), Ardhendu Behera `[通讯]` (Edge Hill University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于超像素的几何感知图形变换器（GeoMeta-GT），通过将皮肤病变图像分解为超像素节点并将患者元数据作为专门的节点融入图中，实现了区域级的图形学习与诊断。

**💡 创新点**

创新点在于：①将相邻超像素间的几何关系（距离、方向）编码为边属性，提升了空间上下文的表达；②将元数据直接嵌入为图节点并与所有区域相连，实现结构化的多模态融合；③提出了边感知图形变换器，在注意力机制中直接利用边属性进行加权；④使用相似度加权的结构细化模块进一步去噪并强化语义一致性。

**🔧 技术方法**

技术包括：冻结预训练CNN提取特征、SLIC超像素分割、节点特征的平均/最大池化、几何边属性构造、元数据线性投影为节点特征、边感知图形变换器、相似度加权细化、全局均值池化与二分类器。

**📊 数据集**

在四个公开数据集上评估：ISIC2024、HAM10000、PAD‑UFES‑20 与 HIBA。

**📈 对比分析**

与多种基线（CNN、ViT、混合CNN‑LSTM、对比学习、上下文感知GNN等）对比，GeoMeta‑GT 在所有数据集上均取得显著提升，准确率最高可达 98.61%（ISIC2024）、98.23%（HAM10000）、97.17%（PAD‑UFES‑20）、95.41%（HIBA），并在召回率和 F1‑score 上也表现优异。

**⚠️ 局限性**

局限性包括：仅处理二分类任务，未涵盖多类别诊断；模型依赖于预训练 CNN 的特征，可能对不同域或少样本情况适应性有限；元数据的表示仍为简单投影，缺乏对不同元字段重要性的动态权重；解释性方面仍需进一步可视化分析。

---

## 524. Organizing in the Digital Age: Understanding Community, Challenges, and Consequences in Digitally-facilitated Labor Organizing

**arXiv ID:** 2606.20375 | [PDF](https://arxiv.org/pdf/2606.20375v1)

**作者:** Frederick Reiber `[一作]` (Boston University), Allison McDonald `[通讯]` (Boston University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过17份半结构化访谈，分析美国工会成员在Discord、WhatsApp、Slack等文字聊天工具中的组织结构、交互动态及文化影响。

**💡 创新点**

填补了以往研究忽视的正式工会成员与数字工具交互视角，系统揭示工具选择对参与、信任与权力分配的双重影响。

**🔧 技术方法**

使用Discord、WhatsApp、Slack、Signal等常用即时通讯平台。

**📊 数据集**

访谈数据共17名工会成员，覆盖科技、教育、法律、食品娱乐等行业。

**📈 对比分析**

通过对比不同工具的使用频率、访问权限和冲突程度，发现集中式平台赋予工人更多自治权，但也带来管理冲突；分散式平台则降低管理冲突但功能受限；总体缺乏量化性能指标。

**⚠️ 局限性**

仅覆盖美国工会，样本规模有限，难以推广至全球；招聘渠道偏向已网络化工会，可能偏向积极参与者；研究未系统考察种族与数字鸿沟。

---

## 525. ARGUS: Production-Scale Tracing and Performance Diagnosis for over 10,000-GPU Clusters

**arXiv ID:** 2606.20374 | [PDF](https://arxiv.org/pdf/2606.20374v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 526. An Infrastructure-less, Control-Independent Solution to Relative Localisation of a Team of Mobile Robots using Ranging Measurements

**arXiv ID:** 2606.20365 | [PDF](https://arxiv.org/pdf/2606.20365v1)

**作者:** Paolo Golinelli `[一作]` (University of Trento), Daniele Fontanelli `[通讯]` (University of Trento)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种完全去中心化、无需基准点、控制无关的多机器人相对定位算法，并通过真实实验和仿真验证其在低可观测、部分连通环境中的鲁棒性。

**💡 创新点**

核心创新在于：①使用多假设贝叶斯框架（Particle Filter + Gaussian–von Mises Mixture Model）保留所有可行解，避免因不可观测导致单一估计失效；②不依赖运动控制或固定基准点，适用于任意平台；③通过信息共享实现测量信息在网络中的传播，即使某些机器人没有直接测量也能获得全局定位。

**🔧 技术方法**

算法技术包括：离散时间单车动力学模型；基于扩展卡尔曼滤波的线性化预测；稀疏距离测量更新；系统atic采样与正则化的粒子重采样；Gaussian–von Mises混合模型聚类；BIC自适应簇数选择；协作更新与信息共享。

**📊 数据集**

实验使用 LIMO 差速机器人，搭载 DWM1001 UWB 进行相互测距，OptiTrack 运动捕捉系统提供真值；仿真用于5机器人部分连通网络场景。

**📈 对比分析**

与真值比较使用三项指标：真实位姿概率、最优估计误差、簇面积；在三种场景中均展示了高概率定位、误差小于几厘米、簇面积随时间收敛。未与其他现有方法直接对比，但实验结果表明在弱可观测和部分连通条件下仍能保持良好精度。

**⚠️ 局限性**

局限性包括：①对计算和存储资源要求较高，粒子数与簇数决定复杂度；②估计可能滞后，因通信仅在测量时同步；③在极端噪声或长距离无测量时易失真；④需在网络中至少保持一定连通性以实现信息传播。

---

## 527. Quantum-classical physics-informed Kolmogorov-Arnold networks for PDEs

**arXiv ID:** 2606.20326 | [PDF](https://arxiv.org/pdf/2606.20326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 528. CoLI: A Reproducible Platform for Continuum Robot Learning via Monolithic 3D Printing and Isomorphic Teleoperation

**arXiv ID:** 2606.20389 | [PDF](https://arxiv.org/pdf/2606.20389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 529. InfantFace: Detecting infant faces in neonatal clinical environments

**arXiv ID:** 2606.20449 | [PDF](https://arxiv.org/pdf/2606.20449v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 530. Agentic AutoResearch forSpace Autonomy: An Auditable, LLM-Driven Research Agent for Aerospace Control Problems

**arXiv ID:** 2606.20394 | [PDF](https://arxiv.org/pdf/2606.20394v1)

**作者:** Amit Jain `[一作]` (Massachusetts Institute of Technology), Richard Linares `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

AutoResearch通过大语言模型自动化科研循环，提出、执行并分析实验以训练航天器的导航与控制策略，并在循环内嵌入可信度层对结果进行审核。

**💡 创新点**

创新点在于将LLM驱动的搜索与自我审核机制（测量种子噪声、重采样验证、留一剪枝）结合，形成可复用的“家族契约”框架，使自动化实验结果既可信又可解释。

**🔧 技术方法**

核心技术包括大型语言模型（Claude Sonnet 4.5）驱动的超参数搜索、行为克隆学习、随机搜索基线、控制障碍函数实时安全滤波器，以及种子噪声测量、重采样检验和留一剪枝的统计审核流程。

**📊 数据集**

使用的训练数据为专家最优控制（最小能耗）演示轨迹，分别在相对轨道摆渡任务中采集256条演示，在碰撞规避对接任务中采集128/256条演示。

**📈 对比分析**

与等价随机搜索对比，LLM搜索在摆渡任务将平均终端距离从约16 m降至0.094 m（重采样均值0.101 m，超过15σ），在对接任务实现安全评分从基线约3.5降至0.058 m（重采样均值0.058 m，满足严格安全门限），而随机搜索在对接任务根本无法产生可行策略。

**⚠️ 局限性**

局限性包括实验迭代数仅为几十次、单一LLM后端、随机搜索基线与LLM搜索的训练预算不完全匹配、未对搜索方差进行系统评估，以及仅在单个实验跑中验证结果。

---

## 531. SSD: Spatially Speculative Decoding Accelerates Autoregressive Image Generation

**arXiv ID:** 2606.20543 | [PDF](https://arxiv.org/pdf/2606.20543v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 532. The FID Lottery: Quantifying Hidden Randomness in Generative-Model Evaluation

**arXiv ID:** 2606.20536 | [PDF](https://arxiv.org/pdf/2606.20536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 533. The Token Is a Group Element: On Lie-Algebra Attention over Matrix Lie Groups

**arXiv ID:** 2606.20547 | [PDF](https://arxiv.org/pdf/2606.20547v1)

**作者:** Przemyslaw Musialski `[一作]` `[通讯]` (New Jersey Institute of Technology), Przemyslaw Musialski (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出将注意力中的“token”设为矩阵Lie群的元素（即纯粹的变换），并用该token的对称不变量log(g_i^{-1}g_j)直接计算闭式代数范数作为注意力分数，得到一种名为Lie‑Algebra Attention的自注意力机制；

**💡 创新点**

创新点在于：①不再把token视为携带变换的向量，而是把token本身置于Lie群上；②通过token本身的对称不变量直接读出注意力分数，消除对表征理论（如不可约表示、Clebsch‑Gordan积、旋转向量等）的依赖；③实现了对非紧非阿贝尔仿射群（scale、shear等）的完整处理，突破了传统不可约表示与指数映射方法的限制；

**🔧 技术方法**

使用的技术包括：矩阵Lie群的指数与对数映射、Lie代数的块分解、基于Frobenius范数的加权平方范数作为分数、标准Transformer骨架（多头注意力、前馈网络、层归一化）以及在token上直接做局部变换的输出修正；

**📊 数据集**

实验数据集为人工合成的连续步长序列完成任务，分别在SE(2)、SO(3)和Aff(2)三组Lie群上生成长度为8的序列，随机缺失中间一个token并打乱顺序；

**📈 对比分析**

与两种对照模型比较：①用相同不变量但学习的MLP核（参数量多约50–80×）的LieTransformer式模型；②用传统向量token与标准点积注意力的模型。结果显示：闭式代数范数分数在参数量大幅减少的同时，位置恢复误差和对称性误差均低于学习核模型，且明显优于向量token基线；

**⚠️ 局限性**

局限性：依赖于主图的对数单值性，若相对变换靠近或跨越图边界则分数不定义或数值不稳定；块加权范数假设每个块内部同构，可能在更复杂任务中欠缺表达能力；目前仅在序列完成任务上验证，尚未在更大规模或真实世界的3D仿射任务上进行实证。

---

## 534. VisDom: Sparse Novel View Synthesis with Visible Domain Constraint

**arXiv ID:** 2606.20531 | [PDF](https://arxiv.org/pdf/2606.20531v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 535. Predictability as a Fine-Grained Measure for Privacy

**arXiv ID:** 2606.20546 | [PDF](https://arxiv.org/pdf/2606.20546v1)

**作者:** Linda Lu `[一作]` (Cornell University), Karthik Sridharan `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种新的细粒度隐私度量“可预测性”，衡量攻击者在已泄露数据基础上，通过算法输出进一步预测未知个体敏感信息的能力。

**💡 创新点**

创新点在于将攻击者的核心知识建模为随机过程，利用可预测性与传统差分隐私区分开来，并通过广义矩估计（GMM）给出渐近可计算的可预测性上界；还设计了针对随机分片攻击的可预测性校准噪声方案，显著提升了 ERM 的精度。

**🔧 技术方法**

主要技术包括：随机过程建模、可预测性定义、广义矩估计与其渐近效率、协方差校准噪声、以及与差分隐私的组合。

**📊 数据集**

论文未给出具体实验数据集，主要以理论推导和通用的统计假设（如 i.i.d./混合马尔可夫过程）为基础。

**📈 对比分析**

与传统差分隐私的最坏情况保证相比，本文的可预测性校准噪声在相同隐私预算下可获得更低的精度损失；在随机分片攻击情景下，可预测性上界与攻击者泄露比例呈显著倒数关系。

**⚠️ 局限性**

局限性包括：需要已知或可估计攻击者的随机过程；对过程的混合性、可混合性等统计假设要求较高；现有结果主要为渐近分析，缺乏有限样本下的具体误差界；对未知攻击过程的适应性仍需进一步研究。

---

## 536. Execution-State Capsules: Graph-Bound Execution-State Checkpoint and Restore for Low-Latency, Small-Batch, On-Device Physical-AI Serving

**arXiv ID:** 2606.20537 | [PDF](https://arxiv.org/pdf/2606.20537v1)

**作者:** Liang Su `[一作]` `[通讯]`, Liang Su

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 FlashRT，一个低延迟单流 LLM 与机器人控制的执行时钟，将整个前向推理捕获为静态缓冲区的 CUDA 图，并在此基础上实现执行状态胶囊（capsule）以实现状态快照、恢复、分叉和回滚。

**💡 创新点**

创新点在于：①将 KV 缓存与执行状态拆分，采用无块表指针的静态缓冲区捕获，使得整个执行状态成为可冻结的完整对象；②在同一运行时上同时提供低延迟执行和完整状态快照，满足单流、低延迟、交互式 AI 的需求；③引入胶囊机制，让前缀重用、分叉和回滚从计算密集转为带宽密集，兼容多模型（LLM、VLA、机器人策略）共享的执行契约。

**🔧 技术方法**

技术手段包括：CUDA Graph 捕获、无块表指针的静态缓冲区布局、FP8/FP4 量化、MTP 预取、静态图重放、胶囊的字节级快照与恢复、三层执行契约（底层缓冲区、图、计划），以及与 SGLang、vLLM 的对比评测。

**📊 数据集**

数据集与模型：使用 Qwen3.6（混合线性+全注意力 LLM）、Higgs TTS（4B）和 π_0 风格的 VLA 机器人策略，分别在 RTX 5090、Jetson AGX Thor、DGX Spark 等设备上进行实验；并在机器人实验中使用自定义的动作重播数据。

**📈 对比分析**

比较方法：在相同模型、相同 GPU 上对比 FlashRT 与 vLLM（开启 APC）以及 SGLang 的性能，主要指标为首 token 延迟（TTFT）和首次音频延迟（TTFA），不包含 MTP 预取；实验显示 FlashRT 的冷启动 TTFT 为 200 ms，胶囊恢复后仅 50 ms，提升幅度随前缀长度从 2k 到 16k 递增至 27×；与 vLLM APC 进行比较时，胶囊恢复仍低于 APC，且在 8k 以上工作集时 vLLM 自动缓存失效，FlashRT 仍保持 50 ms 的稳定延迟。

**⚠️ 局限性**

限制：仅适用于单流/低并发、固定形状范围的场景；对多并发高吞吐量不具竞争力；胶囊需要与特定权重、量化和图版本绑定，跨版本迁移受限；对多形状、长序列动态变更的支持有限；在分布式集群或多 GPU 环境下未评估。

---

## 537. Sovereign Execution Brokers: Enforcing Certificate-Bound Authority in Agentic Control Planes

**arXiv ID:** 2606.20520 | [PDF](https://arxiv.org/pdf/2606.20520v1)

**作者:** Jun He `[一作]` (OpenKedge), Deying Yu `[通讯]` (OpenKedge)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本研究提出 Sovereign Execution Broker (SEB)，一种在代理控制平面中将证书绑定的授权转化为运行时可执行、可撤销、可审计的短期身份的机制。

**💡 创新点**

创新点在于将提案认证、运行时执行与可撤销性、漂移检测和失效闭合结合为完整的参考监控，消除了代理运行时持有长期凭证的风险。

**🔧 技术方法**

使用技术包括 Ed25519 数字签名验证、STS/TokenRequest 代理、PostgreSQL 事务、AWS STS 与 Kubernetes TokenRequest 的短期凭证下放、数据库中的 nonce 预留、以及对证书与请求的完整匹配与漂移判断。

**📊 数据集**

实验数据集基于 AWS EKS v1.28 集群和 AWS EC2 安全组，使用了 5,000 次安全组更新和 K8s 资源补丁的真实工作负载。

**📈 对比分析**

与直接 IAM、静态策略代理、仅审计等基线比较，SEB 的平均延迟增加约 28 ms（K8s）或 137 ms（AWS），并在 1 ms 网络条件下可实现 820 rps（K8s）或 240 rps（AWS）吞吐。

**⚠️ 局限性**

主要局限包括对云提供商 IAM 的依赖、凭证下发冷启动延迟、无法满足高频低延迟控制、单点故障、部署复杂度高、撤销传播延迟和最终一致性导致的漂移窗口。

---

## 538. FlowEdit: Associative Memory for Lifelong Pronunciation Adaptation in Flow-Matching TTS

**arXiv ID:** 2606.20518 | [PDF](https://arxiv.org/pdf/2606.20518v1)

**作者:** Harshit Singh `[一作]` (University Of Maryland), Nityanand Mathur `[通讯]` (Smallest AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 FlowEdit 的持续适配框架，能在冻结的流匹配 TTS 模型上通过优化文本嵌入层的扰动来纠正专有名词发音，并将纠正信息存入现代 Hopfield 网络做成可检索的记忆。

**💡 创新点**

创新点在于：①将发音纠正转化为可微分的文本嵌入扰动，避免权重更新导致的灾难性遗忘；②使用内容可寻址的 Hopfield 网络做成终生记忆，支持模糊形态学匹配；③通过对齐、对数似然、梯度反向 ODE 等技术实现低成本、快收敛的编辑。

**🔧 技术方法**

核心技术包括：流匹配（Conditional Flow Matching）+ 反向 ODE 梯度求解、现代 Hopfield 网络（Soft Attention + 相似性门控）、Whisper 语音对齐、Whisper 参考音频、Adam + 梯度裁剪、L2 正则化、语音评估器 wav2vec2.0 等。

**📊 数据集**

使用 312 个多语种专有名词（18 种语言族）构成的 Polyglot-Nouns 基准，每个词有 5 句母语句子，共 1,560 条音频；评估基准包括 LibriTTS‑R 的 500 条通用语音样本。

**📈 对比分析**

与零射击、词典覆盖、全微调、LoRA、Prompt Tuning 等基线比较，FlowEdit 在目标词 PER 方面达到 3.1%（92.7% 相对降幅），同时保持 4.1% 的通用 PER；训练时间仅约 15 秒，远快于微调；人类评测与自动评测一致，且没有引入音质退化。

**⚠️ 局限性**

局限性包括：对单音节词和语调（如普通话、越南语）发音纠正效果仍有限；F0 误差导致的音高不准；在大规模记忆（>10k 条）时检索准确性下降，需要分区或层级化存储；当前仅支持文本嵌入编辑，未涉及音素级的细粒度控制。

---

## 539. Probe-and-Refine Tuning of Repository Guidance for Coding Agents

**arXiv ID:** 2606.20512 | [PDF](https://arxiv.org/pdf/2606.20512v1)

**作者:** Asa Shepard `[一作]` (Williams College), Jeannie Albrecht `[通讯]` (Williams College)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种 Probe‑and‑Refine 调优流程，通过生成合成 bug‑fix 任务反复诊断并修正仓库的指导文件，以提升 LLM 编码代理在 SWE‑bench 上的修复成功率。

**💡 创新点**

创新点在于使用单步 LLM 调用的迭代反馈机制，将仓库知识从静态结构转换为针对性的操作指南，无需多步推理或工具调用。

**🔧 技术方法**

技术上采用 Qwen3.5‑35B‑A3B 模型、单步 LLM 生成/评估/编辑、ReAct 风格代理、SWE‑bench Verified 验证，以及混合效应逻辑回归进行统计分析。

**📊 数据集**

使用的数据集为 12 个公开 Python 仓库的 500 条 SWE‑bench Verified 任务（每个仓库多次实例）。

**📈 对比分析**

在四次独立的 200 步实验中，将 Probe‑and‑Refine 与无上下文和静态知识库基线进行对比，平均 resolve rate 为 33.0%（比无上下文高 7.5pp，p<0.001），提升主要来源于覆盖率而非单个补丁精度。

**⚠️ 局限性**

局限性包括：依赖模型特定的诊断输出，指导文本长度对性能影响不明，跨模型迁移可能失效，实验仅在 Qwen 上完成且仅评估 SWE‑bench，难以验证在其他模型或更大、不同仓库分布上的泛化。

---

## 540. Beyond Global Replanning: Hierarchical Recovery for Cross-Device Agent Systems

**arXiv ID:** 2606.20487 | [PDF](https://arxiv.org/pdf/2606.20487v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 541. Efficient and Sound Probabilistic Verification for AI Agents

**arXiv ID:** 2606.20510 | [PDF](https://arxiv.org/pdf/2606.20510v1)

**作者:** Alaia Solko-Breslin `[一作]`, Krishnamurthy Dj Dvijotham `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于分布鲁棒优化的概率执行监控框架，用以评估并阻止 AI 代理在不确定环境下的安全违规。

**💡 创新点**

创新点在于将概率 Datalog 推导图转换为分布鲁棒线性规划，再通过 SDP 松弛实现可实时的安全风险上界估计，避免了传统独立性假设导致的风险低估。

**🔧 技术方法**

采用分布鲁棒优化、概率 Datalog、线性规划与半正定规划（SDP）等技术，辅以期望和协方差约束。

**📊 数据集**

在终端代理基准 Intercode‑NL2Bash、ATBench 以及 Praline 侧信道漏洞分析任务上进行评估。

**📈 对比分析**

与 Praline、Monte Carlo 采样和阈值化的确定性监控相比，SDP 在保持安全性的同时实现更高的实用性（精度与召回均较高），AUC 更优，且求解时间可控，避免了指数级求解时间和求解超时。

**⚠️ 局限性**

局限性包括：长时间轨迹时风险上界趋近 1；对工具语义依赖手工建模，难以覆盖自定义脚本；以及图深度和相关约束增多时求解时间随多项式增长，可能影响实时性。

---

## 542. Your Mouse and Eyes Secretly Leak Your Preference: LLM Alignment using Implicit Feedback from Users

**arXiv ID:** 2606.20482 | [PDF](https://arxiv.org/pdf/2606.20482v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 543. CalTennis: Large Multi-View Tennis Video Dataset and Benchmark of Monocular-to-3D Pose Estimation

**arXiv ID:** 2606.20542 | [PDF](https://arxiv.org/pdf/2606.20542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 544. DeepSWIP: Quotient-WMC Counterfactuals for Neural Probabilistic Logic Programs

**arXiv ID:** 2606.20526 | [PDF](https://arxiv.org/pdf/2606.20526v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 545. Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages

**arXiv ID:** 2606.20517 | [PDF](https://arxiv.org/pdf/2606.20517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 546. Increasing Resilience of Continuum Robots via Motion Planning Algorithms

**arXiv ID:** 2606.20495 | [PDF](https://arxiv.org/pdf/2606.20495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 547. Caching for Dollars, Not Hits: An Exact Offline Reference for Cloud-Egress Caching and the Crossover That Decides When It Pays

**arXiv ID:** 2606.20539 | [PDF](https://arxiv.org/pdf/2606.20539v1)

**作者:** Madhulatha Mandarapu `[一作]` (VaidhyaMegha Private Limited), Sandeep Kunkunuru `[通讯]` (VaidhyaMegha Private Limited)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了基于云对象存储计费的离线美元最优缓存参考，并用其评估主流缓存策略在实际云价格下的美元损失。

**💡 创新点**

提出了可多项式求解的统一大小间隔线性规划求解离线最优成本，并将流模型扩展为“cost-FOO”紧界，为多尺寸问题提供近似上限，揭示了“异质性损失定律”“竞争前沿”等新规律。

**🔧 技术方法**

间隔包装线性规划、整数线性规划、流网络求解、线性规划松弛、成本感知GreedyDual、离线成本下界（cost‑FOO）等。

**📊 数据集**

合成Zipf访问模式的请求流，以及Twitter的实际twemcache生产缓存轨迹（20k请求窗口）。

**📈 对比分析**

将LRU、LFU、GreedyDual‑Size（成本感知）等在美元成本上与离线最优做对比，发现LRU在成本异质性高时的美元损失显著，GreedyDual‑Size将损失降低约10倍；在实际轨迹中，成本阈值预测与实验一致，说明美元感知缓存在高异质性场景下能显著节省成本。

**⚠️ 局限性**

离线最优解的间隔LP仅适用于统一大小，变尺寸下只能得到近似上界；求解规模受限于行数，无法处理百万级请求；实验仅覆盖单一内存缓存工作负载，未验证大对象或CDN场景；价格向量假设为固定列表价，未考虑动态重定价。

---

## 548. How Do Instructions Shape Speech? Cross-Attention Attribution for Style-Captioned Text-to-Speech

**arXiv ID:** 2606.20532 | [PDF](https://arxiv.org/pdf/2606.20532v1)

**作者:** Nityanand Mathur `[一作]` (Smallest.ai), Sudarshan Kamath `[通讯]` (Smallest.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了风格描述词在流匹配TTS模型CapSpeech中的跨注意力机制，提出DAAM方法并分析了词级热图与声学特征的关系。

**💡 创新点**

首次将DAAM迁移至语音扩散模型，对跨注意力进行词级时间热图解释，并揭示风格词的全局调节、与F0/能量的相关性以及层/步动态。

**🔧 技术方法**

采用流匹配DiT、T5+CLAP编码器、HiFi-GAN声码器，并在每层/ODE步注册前向钩子提取多头注意力，聚合为1D热图；使用方差、PMR、熵、Pearson相关、层/步重要性等指标。

**📊 数据集**

构造了120条风格标题（30形容词×20名词×6模板）与30条文本转录，总计3,600组(风格标题,文本)组合，生成3,520条音频。

**📈 对比分析**

与内容词、功能词对比，计算方差、PMR、熵以及与F0/能量的相关系数，发现风格词的方差最低、相关性最高，层17/ODE步0为风格影响峰值；指标显示跨注意力在模型中起全局调节作用。

**⚠️ 局限性**

仅在CapSpeech单一模型上评估；风格词有限且为人工合成；未做因果干预或头部级分析；缺乏对比其他扩散/流匹配TTS模型的结果。

---

## 549. LedgerAgent: Structured State for Policy-Adherent Tool-Calling Agents

**arXiv ID:** 2606.20529 | [PDF](https://arxiv.org/pdf/2606.20529v1)

**作者:** Md Nayem Uddin `[一作]` (Arizona State University), Chitta Baral `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LedgerAgent，使用显式账本记录任务状态，并在写入操作前通过政策门控验证合法性。

**💡 创新点**

创新点在于将状态抽象为结构化账本并在推理时实时检查政策，避免了传统仅靠提示历史的状态漂移与违规写入。

**🔧 技术方法**

基于 schema 的账本维护、可执行的政策谓词、标准工具调用框架与 LLM 生成相结合。

**📊 数据集**

在 τ^2-bench 的航空、零售、电信四个客服域以及 τ-Trait 的医疗域进行评测。

**📈 对比分析**

与基线（FC）及 IRMA 对比，LedgerAgent 在 pass^1 与 pass^4 指标提升 3–15 分，尤其在涉及写操作的任务中表现显著提升。

**⚠️ 局限性**

局限在于仅适用于结构化工具使用场景，账本只能反映已读取信息，需手工定义路径映射与政策谓词，且无法处理非结构化或不可观察的状态。

---

## 550. FreeStyle: Free Control of Style-Content Dual-Reference Generation from Community LoRA Mining

**arXiv ID:** 2606.20506 | [PDF](https://arxiv.org/pdf/2606.20506v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 551. Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems

**arXiv ID:** 2606.20493 | [PDF](https://arxiv.org/pdf/2606.20493v1)

**作者:** Zewen Liu `[一作]` `[通讯]` (Qilu Institute of Technology), Zewen Liu (Qilu Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型在多智能体系统中作为评估者时的偏见传播，提出并验证了Contagion Networks框架，设计链式和全连通实验以测量跨代理传染矩阵 Γ_N。

**💡 创新点**

创新点包括：①定义跨代理传染矩阵 Γ_N 及其谱半径阈值以划分抑制/持久/级联三种传播范式；②证明评估者多样性可显著抑制传播，并给出至少 3 名评估者即可达到 72.4% 降低 γ_eff 的阈值；③将跨模态传播理论与跨代理传播统一为同一 Γ 形式。

**🔧 技术方法**

使用 Test‑Time Reinforcement Learning (TTRL) 进行策略权重更新，构建链式、全连通等网络拓扑；通过计算谱半径 ρ(Γ_N)、传播因子 β_L 以及有效传染系数 γ_eff 评估传播强度；实验使用 DeepSeek‑chat API 进行 API 调用。

**📊 数据集**

实验数据集为 50 个任务（代码生成、数学推理、文本摘要等），使用三种评估者偏好（结构化、平衡、基于证据）进行 840 次 DeepSeek‑chat API 调用；对比了同模型内部传染系数（0.14–0.30）与 MM‑EPC 跨模型系数（0.85–1.30）。

**📈 对比分析**

比较方法：先在全连通网络中计算谱半径以判断潜在级联风险；随后在链式网络中测量单跳和多跳传染系数，得到 β_3=0.0055，证明抑制；最后通过评估委员会大小 k=1→3 的实验，展示 γ_eff 下降 72.4%，验证多评估者策略。性能上显示同模型内部传播远低于跨模型情况，且多评估者可显著抑制偏见扩散。

**⚠️ 局限性**

局限性包括：①仅使用单一模型族（DeepSeek）且实验规模小（n=2）；②TTRL 仅是最小调节机制，其他适应方法可能产生不同传播；③未覆盖多种网络拓扑（星形、环形等）及开源模型；④缺乏统计显著性检验，结果为探索性估计。

---

## 552. Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation

**arXiv ID:** 2606.20491 | [PDF](https://arxiv.org/pdf/2606.20491v1)

**作者:** Fatma Youssef Mohammed `[一作]` (Norwegian University of Science and Technology), Kostas Alexis `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出轻量级扫描路径预测模型GazeLNN，将人类视线扫描路径预测用于机器人主动视觉。

**💡 创新点**

创新点在于使用液晶神经网络（CfC）与MobileNetV3构建极低计算成本的扫描路径预测，并将其与强化学习主动相机控制相结合。

**🔧 技术方法**

技术包括Liquid Neural Network (CfC)、MobileNetV3特征提取、动态时间扭曲+KL损失、强化学习（APPO）与相机控制策略。

**📊 数据集**

使用OSIE和MIT低分辨率图像数据集进行训练与评估。

**📈 对比分析**

与tSPM-Net等基线对比，GazeLNN在ScanMatch、Levenshtein、Hausdorff等指标上均取得最高分，计算量降低99%，推理速度提升6倍。

**⚠️ 局限性**

局限在于模型仍基于二维热图预测，缺乏对动态场景的时空建模，且在极端光照或遮挡条件下性能可能下降。

---

## 553. How Fragile Are Training-Free AI-Generated Image Detectors? A Controlled Audit of Score Direction, Preprocessing, and Compression

**arXiv ID:** 2606.20488 | [PDF](https://arxiv.org/pdf/2606.20488v1)

**作者:** Jingwen Zhou `[一作]` (Xidian University), Mingzhe Wang `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对训练‑free 的 AI 图像生成检测方法进行了系统性审计，重新实现了两种主流得分（AEROBLADE‑style 和 RIGID‑style）以及一个 kNN 对照，统一了实验设置并逐一检验实现细节、超参数、格式偏差对检测性能的影响。

**💡 创新点**

首次揭示实现细节（LPIPS backbone、预处理分辨率）和方向假设（噪声扰动对抗性）对 AUROC 的显著影响，并指出压缩格式偏差会导致检测结果出现“压缩有利”的误导；基于此提出了最小报告标准（统一 backbone、预处理、JPEG 重新编码）。

**🔧 技术方法**

使用了 LPIPS（AlexNet / VGG‑16）感知距离、Stable Diffusion VAE 进行重建误差、DINOv2‑base 的特征相似度（噪声扰动）、kNN 最近邻距离、以及 JPEG 重新编码以校正格式偏差。

**📊 数据集**

基准数据为 GenImage 验证集，共 1,500 张图像：800 张 ImageNet 真实图像和 700 张由 7 个生成器（ADM、BigGAN、GLIDE、Midjourney、SD1.5、VQDM、Wukong）生成的假图像，采用 JPEG 质量 70 与 50 的压缩。

**📈 对比分析**

通过无阈值 AUROC 对每个生成器进行比较，发现：① 换用 LPIPS backbone 可提升 0.085；② 预处理方式可使同一方法在某些生成器上提升或下降 0.38；③ 噪声水平 σ 对 RIGID 方向假设极其敏感（某些生成器在 σ=0.05 时 AUROC <0.5）。整体表现差异大，单一融合方式并未优于最佳单一得分。

**⚠️ 局限性**

局限性包括：仅评估了 GenImage、7 个生成器、2 种 JPEG 级别、3 种得分方式，样本量有限；未覆盖其他扰动（缩放、模糊、社交网络处理、对抗后处理）；仅使用 DINOv2 和 Stable Diffusion VAE，未检验其他预训练模型的普适性；未研究阈值迁移和校准问题。

---

## 554. GroundControl: Anticipating Navigation Failures in Vision-Language Agents via Trajectory-Consistent Uncertainty Estimates

**arXiv ID:** 2606.20479 | [PDF](https://arxiv.org/pdf/2606.20479v1)

**作者:** Nastaran Darabi `[一作]` (University of Illinois at Chicago), Amit Ranjan Trivedi `[通讯]` (University of Illinois at Chicago)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在视觉‑语言导航任务中，提出一种基于轨迹一致性的不确定性估计方法 GroundControl，并设计了 Selective Risk–Coverage Navigation (SRCN) 评价协议。

**💡 创新点**

创新点是将不确定性定义为与目标导向的距离‑到‑目标动态一致性的统计偏差，而非仅靠动作熵等瞬时指标，并将常数速度 Kalman 过滤器与多维轨迹特征融合，形成可解释且鲁棒的轨迹级不确定性信号。

**🔧 技术方法**

采用常数速度 Kalman 过滤器对距离-到-目标序列建模，提取归一化创新统计、后验协方差扩张；结合轨迹特征（进展、单调性、路径效率、振荡）并以固定权重线性融合得到 episode‑级不确定性分数。

**📊 数据集**

使用 EB‑Navigation 数据集（基于 Habitat 的 Room‑to‑Room 等），共 300 条轨迹，分为 5 个拆分，每个拆分 60 条。

**📈 对比分析**

与 conformal、entropy、self‑consistency、invalid‑action、随机等基线对比，GroundControl 在成功率和 SPL 两种损失下的 AURC/E‑AURC 均显著低于基线，几乎达到 oracle 排序；在 5 个拆分上均取得最佳或接近最佳的选择性风险表现。

**⚠️ 局限性**

局限包括：评估主要是离线的，未在在线控制中充分验证；常数速度模型对所有导航场景的适用性有限；对极短或极长 episode 的表现未作深入分析；需要进一步研究如何与实时恢复策略结合。

---

## 555. Scalable Training of Spatially Grounded 2D Vision-Language Models for Radiology

**arXiv ID:** 2606.20477 | [PDF](https://arxiv.org/pdf/2606.20477v1)

**作者:** Yusuf Salcan `[一作]` (University of Freiburg), Thomas Brox `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

训练了一个可视觉定位的 2D 视觉‑语言模型 RadGrounder，能够在单张 CT/MRI 切片上生成报告、回答医学问答，并输出解剖结构的定位框；

**💡 创新点**

创新点包括：①通过 LLM 与 TotalSegmentator 自动生成大规模 1.2M 双语图像‑文本对及空间标签，消除人工标注；②以 token‑based 方式将定位任务化为文本生成，无需额外分割头；③提出 LLMScore 与 G‑IoU 评估，兼顾语义与空间一致性；

**🔧 技术方法**

使用了 PaliGemma‑2 作为基架（SigLIP‑So400m 视觉编码 + Gemma‑2B 解码器）、LLM（Gemma‑3、GPT‑OSS）驱动的数据生成与评估、以及自定义的 Token‑Bounding‑Box 生成机制；

**📊 数据集**

主要数据集为 RefRad2D（1.2 M CT/MRI 图像‑文本对，含 236 k 自动空间标注）、RefRad2D‑VQA（≈9.6 M QA 对）以及外部 Slake、VQA‑RAD 基准；

**📈 对比分析**

在 Slake 与 VQA‑RAD 公开基准上，RadGrounder 与 BiomedGPT‑B、LLaVA‑Med 等专业医学 VLM 竞争，Slake F1 87.7、Closed Accuracy 90.3，VQA‑RAD Open F1 50.7，显示出可比甚至优越的性能；Token‑Detection 的 G‑IoU（≈43.6）优于 Segmentation，且定位监督不影响报告与 VQA 质量；

**⚠️ 局限性**

局限性包括：数据来自单一医院，缺乏多中心验证；空间标注仅覆盖解剖结构而非病灶定位，限制临床实用性；不同定位方式难以直接通过单一指标公平比较。

---

## 556. Farmer Connect: Improving Farmers' Access to Produce Markets

**arXiv ID:** 2606.20465 | [PDF](https://arxiv.org/pdf/2606.20465v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 557. Slow Brain, Fast Planner: Latency-Resilient VLM-Augmented Urban Navigation

**arXiv ID:** 2606.20458 | [PDF](https://arxiv.org/pdf/2606.20458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 558. Analyzing Defensive Misdirection Against Model-Guided Automated Attacks on Agentic AI Systems

**arXiv ID:** 2606.20470 | [PDF](https://arxiv.org/pdf/2606.20470v1)

**作者:** Reza Soosahabi `[一作]` (Keysight Technologies Inc.), Vivek Namsani `[通讯]` (Keysight Technologies Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种名为 CMPE（Contextual Misdirection via Progressive Engagement）的主动防御机制，用于在 agentic AI 系统中对抗基于模型的自动化 jailbreak 与 prompt‑injection 攻击。

**💡 创新点**

创新点在于：①从概率模型出发证明传统 detect‑and‑block 防御在可迭代查询条件下会被攻击者通过反馈循环逼近成功率 1；②提出 detect‑and‑misdirect 策略，引入“误导诱导假阳性（MI‑FP）”以破坏攻击者的自动评估流程，从而为攻击者的成功率设定上界；③设计轻量级对话式误导方法 CMPE，用可预判的积极意图前缀、语境扩展与后续提问相结合，产生语义可行却不具功能性的回应。

**🔧 技术方法**

使用的技术包括：
- 概率模型与 Bayesian 误差分析；
- 大语言模型（LLM）评判器与多模型集成评估；
- CMPE 误导生成算法（基于 LLaMA‑3、Meta LLaMA‑3、NeuralDaredevil‑8B‑abliterated 等轻量级模型）；
- 现有自动化攻击框架 GPTFuzz 与 PAIR 进行端到端评估。

**📊 数据集**

使用的数据集为 AdvBench（500 条高危 jailbreak 提示）以及相关的 500 条 CMPE 生成误导样本，评估时对每对提示-响应进行 10 次评判；此外还使用公开的 LLM 判别器（StrongREJECT、PAIR、HB‑FT‑LLaMA2‑13B、Llama‑Guard‑3‑8B、GPTFuzz‑RoBERTa）来估计误差率。

**📈 对比分析**

比较方法：在模拟实验中计算每个评判器组合下的 ASR 上界，并与 detect‑and‑block 基线对比；在端到端实验中使用 GPTFuzz 与 PAIR 评估框架，在两种受害模型（Vicuna‑13b‑v1.5 与 NeuralDaredevil‑8B‑abliterated）上测量真实成功率、误导成功率（MI‑FP）和平均迭代次数。性能方面：CMPE 将检测框架下的 verified ASR 从 10%–20% 降至 0–2%，并把平均迭代次数缩短约 2–6 倍，表明误导能有效阻断攻击者的搜索循环。

**⚠️ 局限性**

局限性包括：
- 分析假设每轮攻击有固定的验证预算，未考虑跨轮误导累计效应；
- 主要验证于 jailbreak 场景，尚未覆盖工具调用、跨代理协作等更复杂 agentic 环境；
- 对人类用户的副作用未作实验评估，误导在面向人类的聊天机器人中可能导致混淆或误解；
- 误导的有效性依赖于攻击者评判器的弱点，若评判器通过多模型集成或严格阈值提升，MI‑FP 会下降，进而提升 ASR。

---

## 559. Fisher-Geometric Sharpness and the Implicit Bias of SGD toward Flat Minima

**arXiv ID:** 2606.20469 | [PDF](https://arxiv.org/pdf/2606.20469v1)

**作者:** Md Sakir Ahmed `[一作]` (Gauhati University), Hemen Dutta `[通讯]` (Gauhati University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Riemannian 平坦度 (Sharpness) 定义，并证明其对函数保持重参数化不变；证明 mini‑batch SGD 隐式偏向低 Riemannian 平坦度的最小值，并给出基于 PAC‑Bayes 的泛化界；在 MNIST 与 CIFAR‑10 上做实验验证。

**💡 创新点**

将 Fisher 信息矩阵引入 flatness 测量，得到重参数化不变的 sharpness；统一 SDE 随机噪声分析与 PAC‑Bayes 泛化界；给出理论与实证证明。

**🔧 技术方法**

信息几何、Fisher 信息矩阵、Riemannian sharpness、连续时间 SDE、PAC‑Bayes 证明、对角 FIM 估计、Hutchinson 迹估计、梯度噪声协方差分析。

**📊 数据集**

MNIST（3‑层 MLP）和 CIFAR‑10（TinyCNN）

**📈 对比分析**

与欧氏 Sharpness、SGD/Adam、不同批大小/学习率对比；Riemannian Sharpness 与测试精度高度相关，能更可靠预测泛化，优于欧氏度量。

**⚠️ 局限性**

对角 FIM 近似导致不完全重参数化不变；梯度噪声与 FIM 的比例假设仅近似；在大批量或极大学习率下理论失效；未在更大网络/更复杂数据集上验证。

---

## 560. PCFootprint: A Large-Scale Dataset and Benchmark for Vectorized Building Footprint Extraction from Aerial LiDAR Point Clouds

**arXiv ID:** 2606.20455 | [PDF](https://arxiv.org/pdf/2606.20455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 561. StylisticBias: A Few Human Visual Cues Drive Most Social Biases in MLLMs

**arXiv ID:** 2606.20527 | [PDF](https://arxiv.org/pdf/2606.20527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 562. SARLO-80: Worldwide Slant SAR Language Optic Dataset 80cm

**arXiv ID:** 2606.20523 | [PDF](https://arxiv.org/pdf/2606.20523v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 563. Generating Robot Hands from Human Demonstrations

**arXiv ID:** 2606.20549 | [PDF](https://arxiv.org/pdf/2606.20549v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 564. What Do Safety-Aligned LLMs Learn From Mixed Compliance Demonstrations?

**arXiv ID:** 2606.20508 | [PDF](https://arxiv.org/pdf/2606.20508v1)

**作者:** Sihui Dai `[一作]` (CapitalOne), Mann Patel `[通讯]` (CapitalOne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究安全对齐语言模型在混合合规演示下的行为，探究善意与有害示例混合对后续有害请求响应的影响。

**💡 创新点**

创新点在于系统评估善意与有害合规示例的可交换性、顺序偏置以及对抗性与格式采纳的区分，并揭示偏好优化阶段对有害合规影响。

**🔧 技术方法**

采用混合示例的多次实验、假设检验（卡方检验、逻辑回归）以及在多模型（Llama、OLMo、Gemma、GPT‑OSS）上进行的大规模 in‑context 评估。

**📊 数据集**

使用 RedTeam‑2K、UltraChat、OR‑Bench 等示例库构造演示，HarmBench、SORRY‑Bench、WildGuard‑test 等三大有害评估集合，共计 1404 个有害测试问答。

**📈 对比分析**

通过比较不同模型与不同演示组合下的合规率、格式采纳率，发现模型间存在显著差异；如 Gemma 对有害演示更稳健，Llama 对格式采纳更高，且顺序偏置影响可达 30%。

**⚠️ 局限性**

局限在于仅覆盖四个模型且大部分结果基于公开模型；未对更大规模或多种安全训练算法进行验证，缺乏机制解释与实际防御实验。

---

## 565. Multi-Task Bayesian In-Context Learning

**arXiv ID:** 2606.20538 | [PDF](https://arxiv.org/pdf/2606.20538v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 566. HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining

**arXiv ID:** 2606.20521 | [PDF](https://arxiv.org/pdf/2606.20521v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 567. CoarseSolvers for Exascale Solution of Poisson Problems

**arXiv ID:** 2606.20496 | [PDF](https://arxiv.org/pdf/2606.20496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 568. Marginal Advantage Accumulation for Memory-Driven Agent Self-Evolution

**arXiv ID:** 2606.20475 | [PDF](https://arxiv.org/pdf/2606.20475v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 569. Interpretable Meta-Learning for Multi-Objective Chemical Search

**arXiv ID:** 2606.20497 | [PDF](https://arxiv.org/pdf/2606.20497v1)

**作者:** Antonio Varagnolo `[一作]` (Los Alamos National Laboratory), Nicholas E. Lubbers `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套可解释的线性元学习+贝叶斯自举+动态置信调节的多目标分子搜索框架，能在量子级化学计算约束下快速收敛；

**💡 创新点**

首次将线性图元特征与多任务辅助属性结合的线性元学习应用于多目标化学搜索，并引入贝叶斯自举不确定性估计和自适应置信调节，实现更高效的探索‑利用平衡；

**🔧 技术方法**

使用线性图元特征、线性元学习(LAMeL)、贝叶斯自举集成、Efficient Global Optimization (EGO) 的 PoI 探索策略、K‑means 聚类、动态置信更新以及 UMAP 可视化等技术；

**📊 数据集**

在 QM9 基准（约134k分子，四目标）和真实大规模 SCO 金属有机复合物搜索（从 CSD 生成的配体库）上进行实验；

**📈 对比分析**

与随机搜索和无元学习线性模型进行对比；在 QM9 上收敛约两位数快于随机搜索，在 SCO 搜索中元学习版占比达 78% 的 Pareto 支配率，动态置信调节进一步提升至 52%；RMSE 可降低多达 47%，总体计算成本仅为单线性回归的 2 倍；

**⚠️ 局限性**

生成的 Pareto 集体积较大，需人工筛选；高维目标导致计算量大；动态置信调节缺乏理论评估；不同成本目标的评估未做分离，未来仍需改进。

---

## 570. UltraQuant: 4-bit KV Caching for Context-Heavy Agents

**arXiv ID:** 2606.20474 | [PDF](https://arxiv.org/pdf/2606.20474v1)

**作者:** Inesh Chakrabarti `[一作]` (Advanced Micro Devices), Ashish Sirasao `[通讯]` (Advanced Micro Devices)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在长上下文、多轮对话代理工作负载下实现4位KV缓存压缩，并提供针对AMD GPU的Ultra-TQ与UltraQuant两种高效实现；

**💡 创新点**

通过引入Walsh–Hadamard旋转、定制化的校准中心点、块级FP4微张量与UE8M0尺度，兼顾压缩率与推理质量，且首次将FP4直接映射到CDNA4矩阵核，显著提升了缓存驻留与吞吐率；

**🔧 技术方法**

采用旋转+代码书量化、Lloyd–Max校准、块级尺度、FP4微张量（E2M1）与UE8M0尺度、FP8查询、CDNA4 F8F6F4 MFMA、优化的解码注意力核、Ultra-TQ与UltraQuant两种实现；

**📊 数据集**

使用ShareGPT真实对话数据进行多轮代理工作负载测试，并在GPQA、LCB‑128K、AIME25、MATH500等四个标准基准上评估推理质量；

**📈 对比分析**

与vLLM OSS、AITER FlashAttention、硬件KV缓存以及Ultra-TQ做对比；在多轮代理任务中，UltraQuant相较KV缓存可将P50首词延迟降低3.47×（全轮平均2.3×），输出吞吐率提升1.63×；在大型模型的Decode吞吐率上，UltraQuant达到1.38×（相当于硬件KV缓存1.37×），每令牌延迟仅比Ultra-TQ低0.22×，且对长上下文尤为显著；

**⚠️ 局限性**

仅使用单一常数c在所有模型/头上，可能存在更优的层级/头级校准；UltraQuant仅在超长上下文下显著提升，短上下文无明显优势；Ultra-TQ的中心点校准仅应用于10%层，收益有限；部分基准（如AIME25）显示显著准确率下降，说明方法对某些任务仍有局限。

---

## 571. A-COMPASS: Formal Foundations for Anonymity Analysis in Microdata

**arXiv ID:** 2606.20492 | [PDF](https://arxiv.org/pdf/2606.20492v1)

**作者:** Tamara Tagliavia `[一作]` (Mathematical Institute of Serbian Academy of Sciences and Arts), Silvia Ghilezan `[通讯]` (Mathematical Institute of Serbian Academy of Sciences and Arts)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对 COMPASS 语言进行了扩展，提出 A-COMPASS，支持标准微数据表（每行一人）并实现隐私校验与匿名化操作；同时给出了完整的语义定义并证明了其确定性与可组合性。

**💡 创新点**

创新点包括：① 引入 REPLACE 与 COUNT DISTINCT 语法，突破 COMPASS 只能处理一行一组的限制；② 提供 A-COMPASS 的正式语义模型；③ 在语义基础上证明了 k‑匿名性、l‑多样性、抑制、泛化、随机替换等操作的安全性与完整性。

**🔧 技术方法**

技术方法主要是基于 SQL 的基于 bag 的记号语义（denotational semantics）以及随机轨迹（random trace）和规范排序；利用这些工具实现了对需求、断言、动作的语义解释并完成了数学证明。

**📊 数据集**

使用的实验数据集为一份示例微数据表，包含年龄、邮政编码和年度用电量等属性，用以演示 A‑COMPASS 的使用和匿名化效果。

**📈 对比分析**

本文未给出量化性能评估或与其他工具的实验比较；比较方式主要是概念性说明 A‑COMPASS 如何覆盖 COMPASS 的功能并扩展到更通用的微数据场景。

**⚠️ 局限性**

局限性：① 只适用于完整无缺失值的表；② 不支持子查询和多表操作；③ 目前仅实现 k‑匿名与 l‑多样性校验，未涵盖差分隐私等更高级模型；④ 缺乏实验验证和效率分析。

---

## 572. Agentic Symbolic Search: Characterizing PDEs Beyond Hand-crafted Expressions, Meshes, and Neural Networks

**arXiv ID:** 2606.20467 | [PDF](https://arxiv.org/pdf/2606.20467v1)

**作者:** Zongmin Yu `[一作]` (National University of Singapore), Liu Yang `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了Agentic Symbolic Search（ASYS），一种基于大型语言模型的自动结构搜索框架，用以生成可微分符号程序来近似非线性偏微分方程（PDE）的解析解。

**💡 创新点**

其创新点在于将先验物理知识、问题约束与先前搜索经验转化为可执行的符号程序，并通过进化搜索在结构空间中迭代，而非传统的符号回归或神经网络权重搜索，从而自动挖掘可解释的数学结构。

**🔧 技术方法**

技术上结合了演化搜索（EvE）、基于梯度的 L‑BFGS 参数拟合、可微分符号程序构造，以及一个四维评分向量（物理残差、初值、边界、兼容性）来引导结构生成。

**📊 数据集**

实验使用了五个经典 PDE 案例：非线性薛定谔方程（NLS）、二维 Allen–Cahn 方程、径向 Keller–Segel 聚集模型、Graveleau 自相似 PME 聚焦问题以及非局域的 gCLM 方程，所有案例均为公开的理论与数值基准。

**📈 对比分析**

与专门设计的自相似 PINN（SS‑PINN）基线对比，ASYS 在 NLS、Allen–Cahn、Keller–Segel 和 Graveleau 等案例中获得与或优于基线的 L² 误差（例如 Keller–Segel 0.188 对比 0.258，Graveleau 0.00132），证明了其在结构化求解上的竞争力。

**⚠️ 局限性**

主要限制包括：对单一演化轨迹的高计算成本且不具备对多种初始条件的可扩展性；结构搜索的覆盖范围受限，难以在有限迭代内自动发现全局自相似坐标（如 gCLM）；以及内循环优化在接近奇异行为时可能遇到梯度退化或不收敛。

---

## 573. Data Bias Mitigation under Coverage Constraints & The Price of Fairness

**arXiv ID:** 2606.20461 | [PDF](https://arxiv.org/pdf/2606.20461v1)

**作者:** Bruno Scarone `[一作]` (Northeastern University), Renée J. Miller `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于覆盖约束的数据偏差缓解框架，兼顾增删数据的成本与公平度量；

**💡 创新点**

创新点在于：①引入覆盖约束以保证所有子群体的最小样本量；②允许同时增删数据并通过整数线性规划寻找全局最优；③量化“公平价格”，即在不同公平容忍度下的最小数据修改成本；

**🔧 技术方法**

使用的技术包括：Uniform Bias（UB）公平度量、线性系统求解、整数线性规划（ILP）与Serfling不等式的采样复杂度分析；

**📊 数据集**

实验数据集涵盖COMPAS、Adult、Default三大公开基准；

**📈 对比分析**

方法通过覆盖约束和ILP实现后，保持或提升多种机器学习模型（RF、GBDT、Extra Trees、AdaBoost、Logistic Regression）的准确率，且与传统仅关注公平度量的策略相比，显著降低数据增删成本；

**⚠️ 局限性**

局限性：依赖外部可获得的满足分布的补充数据，无法处理分布漂移与数据可用性不足；对敏感属性选择与公平阈值仍需道德与法律评估。

---

## 574. Minimality of Random Moore Automata under Prefix-Dependent Congruences

**arXiv ID:** 2606.20454 | [PDF](https://arxiv.org/pdf/2606.20454v1)

**作者:** Matías Carrasco `[一作]` (Universidad ORT Uruguay), Sergio Yovine `[通讯]` (Universidad ORT Uruguay)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究在随机的有限状态 Moore 自动机中，考虑前缀依赖的同余关系，并证明在标签分布非退化且每个标签至少有三个可接受符号时，该同余在高概率下为平凡的，从而得到最小化的状态数。

**💡 创新点**

创新点在于将传统的右同余推广到前缀依赖情形，并结合剪枝过程与碰撞自由递推，以及首次出现的“稳定块系统”概念，首次给出在随机 Moore 自动机中高概率最小化的证明。

**🔧 技术方法**

核心技术包括：1) 对可被识别的状态对进行递归剪枝（pair‑pruning）；2) 用碰撞自由递归（collision‑free recursion）近似剪枝概率；3) 对稳定块系统使用一阶矩（first‑moment）上界，从而排除非平凡的同余类。

**📊 数据集**

本研究没有使用真实数据集，而是在完全随机的 IID 转移模型下进行理论分析。

**📈 对比分析**

由于本文为理论研究，没有与其它算法进行实验对比；其性能表现是以概率论的形式表述——在状态数趋于无穷大时，随机系统的同余类数几乎必然等于状态数（即最小化）。

**⚠️ 局限性**

主要限制是证明仅适用于完全独立且同分布的转移函数（即 iid uniform 迁移模型）以及 k_* ≥ 3 的情形；对于 k_* = 1 或 2、以及非均匀或可访问的转移模型，结论尚未成立，需进一步研究。

---

## 575. Current World Models Lack a Persistent State Core

**arXiv ID:** 2606.20545 | [PDF](https://arxiv.org/pdf/2606.20545v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 576. S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligence

**arXiv ID:** 2606.20515 | [PDF](https://arxiv.org/pdf/2606.20515v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 577. Toward Calibrated Mixture-of-Experts Under Distribution Shift

**arXiv ID:** 2606.20544 | [PDF](https://arxiv.org/pdf/2606.20544v1)

**作者:** Gina Wong `[一作]` (Johns Hopkins University), Anqi Liu `[通讯]` (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在分布偏移下，软路由的Mixture-of-Experts（MoE）模型如何失去校准，提出了一种对抗性加权训练方法（Robust MoE/Robust Filtered）来提升在路由诱导的配置重权重下的整体校准与准确率平衡。

**💡 创新点**

核心创新是揭示软路由下即使各专家已校准，聚合仍可能失准；提出以模型自身的严格得分损失为驱动的熵平衡加权（entropy-balanced）对抗重权重，专门针对路由重叠配置，从而在分布偏移下实现鲁棒校准。

**🔧 技术方法**

使用了严格得分规则（cross-entropy、Brier）、熵平衡重权重、分布鲁棒优化理论、路由相关子集筛选（Robust Filtered）、多分类校准度量（ECE、温度缩放）以及MoE架构（4个专家+2层MLP路由）。

**📊 数据集**

在CIFAR-10H（含低人类同意子集）、PACS（四域离散测试）、CivilComments（含种族/身份子集）等图像与文本数据集上进行实验。

**📈 对比分析**

与单专家、Vanilla MoE、MoCaE、FGR等基线对比；结果显示Robust MoE/Robust Filtered在校准误差(ECE)上显著下降（例如CIFAR-10H难样本ECE从0.28降至0.07，CivilComments难样本ECE从0.108降至0.037），准确率保持相近或略有提升，特别在温度缩放后仍保持优势。

**⚠️ 局限性**

局限包括仅在小规模4专家MoE上验证，未探究更大稀疏MoE的可扩展性；高损失示例作为路由重叠配置的代理可能引入噪声，Robust Filtered的筛选规则需进一步细化；使用ECE作为校准度量可能掩盖细粒度子群误差。

---

## 578. Calibration Without Comprehension: Diagnosing the Limits of Fine-Tuning LLMs for Vulnerability Detection in Systems Software

**arXiv ID:** 2606.20502 | [PDF](https://arxiv.org/pdf/2606.20502v1)

**作者:** Arastoo Zibaeirad `[一作]` (University of North Carolina at Charlotte), Marco Vieira `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含834个Linux内核漏洞与补丁对的CWE-Trace基准，用于评估LLM在真实多文件环境下的漏洞检测与CWE分类。

**💡 创新点**

提出了严格的时间切分、上下文感知的漏洞–补丁配对以及两个诊断指标——方向性失败指数（DFI）和层级距离方向（HDD），揭示LLM在安全推理上的盲点。

**🔧 技术方法**

采用零样本评估、LoRA微调、以及针对多任务的指标计算（包括DFI、HDD、Top‑k、MRR等）。

**📊 数据集**

使用了手工校准的834条Linux内核样本，覆盖74个CWE，并对比了5个主流漏洞数据集（Devign、LineVul、PrimeVul、MegaVul、VDISC）进行微调。

**📈 对比分析**

与8个原始LLM和15个LoRA微调模型进行对比，结果显示检测准确率仅略高于50%，CWE分类Top‑1<5%，说明微调主要调节阈值而非提升安全推理。

**⚠️ 局限性**

局限在于对CWE-Trace的高质量标注仍依赖人工审核，且实验仅覆盖C语言内核，无法直接推广至其他语言或更复杂的系统环境。

---

## 579. Software package MaRDI Open Interfaces for improved interoperability in numerical optimization

**arXiv ID:** 2606.20490 | [PDF](https://arxiv.org/pdf/2606.20490v1)

**作者:** Dmitry I. Kabanov `[一作]` (University of Münster), Mario Ohlberger `[通讯]` (University of Münster)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

为 MaRDI Open Interfaces 软件包新增了统一的非线性优化接口，并用它训练物理信息神经网络来求解粘性 Burgers 方程，展示了跨语言调用优化器的可行性和性能。

**💡 创新点**

创新点在于：①构建了面向不同语言（C、Julia、Python）的统一优化接口，②实现了自动化数据 Marshalling 与回调机制，③在同一实验框架下对 SciPy 与 Optim.jl 的 BFGS 线搜索策略进行直接对比，揭示了即使实现相同算法，性能仍有显著差异。

**🔧 技术方法**

主要技术包括：MaRDI Open Interfaces 的模块化架构、Python 的 JAX 自动微分、SciPy 与 Optim.jl 的 BFGS 求解器、以及自定义的回调函数和参数传递机制。

**📊 数据集**

使用的“数据集”是为粘性 Burgers 方程在时空域（t∈[0,2], x∈[0,2]）生成的均匀网格（N_t × N_x）上的自定义采样点，作为损失函数的评估点。

**📈 对比分析**

比较方法：在 42 次独立试验（不同随机种子）下，统计迭代次数、函数/梯度评估次数、最终梯度范数、最终损失和运行时间；结果显示 Optim.jl（特别是某些线搜索策略）在迭代次数、评估次数和耗时上均优于 SciPy，表明跨语言调用不会产生显著性能开销。

**⚠️ 局限性**

局限性：目前仅支持无约束优化；实验仅限于单一 PDE（粘性 Burgers 方程）和 BFGS 算法；尚未验证在更复杂约束或大规模问题中的表现。

---

## 580. Context-Aware Hierarchical Bayesian Modeling of IVF Laboratory Environmental Conditions

**arXiv ID:** 2606.20459 | [PDF](https://arxiv.org/pdf/2606.20459v1)

**作者:** Zahra Asghari Varzaneh `[一作]` (Malmö University), Thomas Ebner `[通讯]` (Kepler Universitätsklinikum)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了55个基于传感器的上下文感知时间特征，并使用层次贝叶斯Beta回归模型对亚洲和北欧两家IVF实验室的高分辨率环境数据进行妊娠率预测，实现跨站点迁移。

**💡 创新点**

创新点包括：①利用滚动标准差、同步温湿度符合区间、最长持续应力时长等多维度上下文特征，捕捉孵化器微环境动态；②在层次贝叶斯框架下采用部分池化共享环境效应，同时保留站点基线；③结合SHAP特征选择减少模型维度。

**🔧 技术方法**

技术手段包括：传感器数据特征工程、SHAP重要性排序、XGBoost、层次贝叶斯Beta回归、PyMC + NUTS采样、Beta分布建模、时间序列交叉验证与部分池化。

**📊 数据集**

使用2024-2025年两家实验室的10分钟间隔传感器数据（温度、湿度、CO₂、TVOC）以及按年龄组的妊娠率；亚洲61周（两段）为训练源，北欧14个月为目标。

**📈 对比分析**

与单站点XGBoost基线和朴素平均预测对比；亚洲交叉验证中，上下文特征将CV-MAE从1.57–1.94%降至0.85–1.30%；北欧测试时，层次模型在35–39岁组MAE 4.30%、R²=0.86，较朴素提升64%；XGBoost在北欧表现差（负R²），显示过拟合。

**⚠️ 局限性**

局限性：数据量有限，尤其北欧仅3个月测试；缺乏患者级别、胚胎质量、刺激方案等潜在混杂变量；聚合水平噪声大，尤其年龄<35组受限；未考虑季节、人员变动等非传感器因素。

---

## 581. MemoryWAM: Efficient World Action Modeling with Persistent Memory

**arXiv ID:** 2606.20562 | [PDF](https://arxiv.org/pdf/2606.20562v1)

**作者:** Sizhe Yang `[一作]` (Chinese University of Hong Kong), Huazhe Xu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为MemoryWAM的世界动作模型，利用滑动窗口、事件边界锚帧和压缩的gist令牌构建高效的持久记忆，从而在长时程、需记忆的机器人操作任务中实现更低延迟和更小GPU占用的决策。

**💡 创新点**

创新点在于将人类记忆的三种形式（短期记忆、事件边界记忆、长期抽象记忆）融合到Transformer注意力机制中，既保留关键视觉细节又通过gist令牌压缩长距离历史，显著提升了记忆效率与任务性能。

**🔧 技术方法**

采用预训练的视频扩散Transformer（DiT）与动作扩散Transformer，结合3D因果视频VAE编码、RoPE位置编码、混合Transformer（MoT）架构，并设计专门的混合注意力掩码与gist令牌机制。

**📊 数据集**

在仿真数据集RMBench（9个双臂操纵任务）以及真实世界ARX双臂机器人+RealSense D455摄像头的数据上进行训练与评估，使用50条专家演示进行学习。

**📈 对比分析**

与VLA基线π_0.5、FastWAM、LingBot-VA等方法对比，MemoryWAM在大多数任务中成功率提升约70个百分点，甚至在“Press Button”任务中达到87%与全历史关注法相同的成功率，同时推理延迟和GPU内存显著低于LingBot-VA。

**⚠️ 局限性**

局限性在于依赖视频扩散模型，导致对语义理解与推理的能力有限，且模型对复杂动态场景的泛化仍受限。

---

## 582. Thinking in Boxes: 3D Editing in Real Images Made Easy

**arXiv ID:** 2606.20556 | [PDF](https://arxiv.org/pdf/2606.20556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 583. JanusMesh: Fast and Zero-Shot 3D Visual Illusion Generation via Cross-Space Denoising

**arXiv ID:** 2606.20563 | [PDF](https://arxiv.org/pdf/2606.20563v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 584. TimeProVe: Propose, then Verify for Efficient Long Video Temporal Reasoning in Activities of Daily Living

**arXiv ID:** 2606.20561 | [PDF](https://arxiv.org/pdf/2606.20561v1)

**作者:** Arkaprava Sinha `[一作]` (University of North Carolina), Srijan Das `[通讯]` (University of North Carolina)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种成本高效的长视频问答框架，将长视频分为轻量级动作检测与基于视觉语言模型（VLM）的精确验证两步，从而只在关键短时段进行高成本推理。

**💡 创新点**

创新点在于①使用动作检测产生稀疏的时序动作框架，再通过边缘LLM根据查询生成候选答案与证据窗口；②只在最有可能的窗口上调用VLM做验证；③引入面向日常生活的开放式时序问答基准OTB，推动了长视频时序推理研究。

**🔧 技术方法**

核心技术包括MS‑Temba轻量级动作检测、CLIP/I3D特征提取、Gemma4‑2B或Qwen2‑7B边缘LLM进行候选生成与评分，以及VideoLLaMA3或GPT‑4o等大型VLM用于验证。

**📊 数据集**

实验使用TSU（Toyota Smart Home Untrimmed）数据集构建OTB基准，并在Charades‑STA上评估时序定位能力；动作检测器在TSU或Charades上训练。

**📈 对比分析**

与多种SFT VLM和agentic框架相比，框架在OTB上提升7.3%准确率，VLM调用次数减少75%，推理成本降低93%；在Charades‑STA上与TimeSuite/Time‑R1相比，精度提升1.3–4.8点，展现了良好的通用性。

**⚠️ 局限性**

局限性包括：假设答案相关证据能被少数局部或合并动作窗口捕获，难以处理需要跨长时间段聚合的全局场景；最终验证仍依赖所选VLM的视觉推理能力，若VLM不够强则可能导致误判。

---

## 585. How Transparent is DiffusionGemma?

**arXiv ID:** 2606.20560 | [PDF](https://arxiv.org/pdf/2606.20560v1)

**作者:** Joshua Engels `[一作]` (Google DeepMind), Neel Nanda `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并提高文本扩散模型DiffusionGemma的透明度与可监测性，探究其在连续潜在空间中的推理机制

**💡 创新点**

将扩散模型的瓶颈映射为可解释的自然语言标记，降低其不透明串行深度并揭示非自回归推理现象（如非时间顺序推理、序列模糊、递归上下文推理）

**🔧 技术方法**

采用logit lens、软最大投影、可解释瓶颈映射、监测评估框架、序列长度预测等技术；对比Gemma 4 autoregressive模型

**📊 数据集**

在多项基准（如Natural2Code、LiveCodeBench、GPQA、GSM8k、multi-hop 事实检索等）以及自定义监测数据集上进行评估

**📈 对比分析**

与Gemma 4进行性能对比；发现DiffusionGemma在监测性与可解释性方面与Gemma 4相当；不透明串行深度在假设可解释瓶颈时降至1.1×，若不可解释则为28.6×

**⚠️ 局限性**

1) 监测评估多canvas，单canvas监测性未知；2) 透明度评估依赖特定训练与架构，未来模型可能不再可解释；3) 假设瓶颈可解释风险高，潜在隐藏推理可能被误判为可解释；4) 实际部署监测成本高

---

## 586. UNIEGO: Proxies as Mediators for Unified Egocentric Video Representation Learning

**arXiv ID:** 2606.20559 | [PDF](https://arxiv.org/pdf/2606.20559v1)

**作者:** Wenhao Chi `[一作]` (University of North Carolina at Charlotte), Srijan Das `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种统一的视角前摄视频编码器，利用多教师知识蒸馏构建代理层进行层级蒸馏

**💡 创新点**

创新点在于通过代理模型桥接不同视角、模态及基础模型的知识，再通过样本级代理选择来避免梯度冲突

**🔧 技术方法**

使用了代理学习、代理融合初始化、选择性代理蒸馏等技术，并基于TimeSformer/UniFormer等视觉编码器实现

**📊 数据集**

在EgoExo-Fitness、Assembly101、EgoExo4D三大视角前摄数据集上进行实验

**📈 对比分析**

与多教师蒸馏、其他基线相比，在动作识别、检索、分割任务上均取得显著提升，达到state‑of‑the‑art性能

**⚠️ 局限性**

局限在代理选择仍基于小损失启发式，缺乏自适应权重机制，可能无法充分利用代理池的全部潜能

---

## 587. Structuring and Tokenizing Distributed User Interest Context for Generative Recommendation

**arXiv ID:** 2606.20554 | [PDF](https://arxiv.org/pdf/2606.20554v1)

**作者:** Ruizhong Qiu `[一作]` (University of Illinois Urbana--Champaign), Hanghang Tong `[通讯]` (University of Illinois Urbana--Champaign)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于生成式推荐的图协同兴趣建模方法G2Rec，利用稀疏化的物品共参与图和软聚类来生成兴趣原型，并将兴趣原型与物品嵌入一起作为生成式序列输入；

**💡 创新点**

创新点在于：①构建稀疏化的共参与图以高效捕捉用户行为；②提出可微软图聚类目标（soft modularity），实现可端到端训练；③通过兴趣原型实现连续兴趣标记，并将其嵌入生成式模型；

**🔧 技术方法**

核心技术包括：图稀疏化采样、可微软图聚类（soft modularity）、兴趣原型嵌入生成、LLM（Llama‑2 13B）生成式推荐以及低秩适配器（LoRA）微调；

**📊 数据集**

实验数据集为Amazon Beauty、Sports、Toys四个子集以及Yelp；

**📈 对比分析**

与经典MF、GRU4Rec、SASRec、BERT4Rec、LightGCN、HeLLM等六种基线相比，G2Rec在Recall、NDCG、MRR等所有指标上均显著提升，线上A/B测试亦显示用户参与度提升0.03%~0.19%；

**⚠️ 局限性**

局限性包括：仍需离线聚类更新频率较低、对极端稀疏物品的兴趣原型提取可能不稳定，且对多模态内容的扩展尚未探讨。

---

## 588. Optimal Deterministic Multicalibration and Omniprediction

**arXiv ID:** 2606.20557 | [PDF](https://arxiv.org/pdf/2606.20557v1)

**作者:** Georgy Noarov `[一作]` (University of Pennsylvania), Aaron Roth `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种能在样本上实现多校准（multicalibration）、奥米预测（omniprediction）和潘预测（panprediction）的确定性预测器，且样本复杂度达到最优的 O(α⁻³) 级别。

**💡 创新点**

突破了之前认为随机化预测器是实现最优样本复杂度的必要条件的观念，证明了即使输出是确定性的也能达到同样的统计性能，并给出了通用的“间隔提示（interval-hint）+在线到批处理 + 细胞分割（cell partition）”的框架实现。

**🔧 技术方法**

核心技术包括：
- 通过“interval-hint”系统构造可在每个上下文上支持的预测值集合；
- 设计基于指数权重（exponential-weights）的在线多校准算法，并通过在线‑批处理转换得到随机化预测器；
- 对随机化预测器进行细胞级别的确定性取样（single‑seed per cell）以实现完全确定化；
- 结合置信区间与分割细胞的权重平方和控制，确保在所有群体和测试上误差不超过目标。

**📊 数据集**

论文主要是理论性工作，没有使用公开数据集，而是基于通用概率分布 P((X,Y)) 进行理论分析和样本复杂度证明。

**📈 对比分析**

与之前的随机化方法相比，本文在样本复杂度上保持 O(α⁻³) 的最优率，且在算法实现上实现了多项式时间与多项式查询时间；与其他已知的确定性多校准算法（如桶化方法）相比，误差缩放从 O(α⁻⁶) 改进为 O(α⁻³)。

**⚠️ 局限性**

限制包括：
- 需要预先知道或构造群组集合 𝔾 或审计器类的有限覆盖；
- 需要分配三份样本（置信区间、在线学习、细胞分割），在样本紧张时会增加样本需求；
- 对离散化网格和阈值代表的选择敏感，过粗的网格可能导致性能下降；
- 在非均匀分布或高维情境下，细胞分割的复杂度可能增大。

---

## 589. From Efficiency to Leakage -- Privacy Backdoor in Federated Language Model Fine-Tuning

**arXiv ID:** 2606.20553 | [PDF](https://arxiv.org/pdf/2606.20553v1)

**作者:** Shanghao Shi `[一作]` (Washington University in St. Louis), Wenjing Lou `[通讯]` (Virginia Tech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在联邦学习下的参数高效微调中，构造隐蔽隐私后门，记录每个训练样本的梯度以实现数据重建。

**💡 创新点**

创新点在于使用单神经元记忆、RaLU激活、LayerNorm不变的后门，实现在状态优化器下的闭式重建且对模型无影响。

**🔧 技术方法**

技术包括参数高效微调（PEFT）、RaLU激活函数、LayerNorm抗干扰设计、Adam/SGD逆向分析与安全聚合突破。

**📊 数据集**

数据集涵盖AGNews、SQuAD、EMRQA‑mSQuAD与GSM8K，模型覆盖BERT、GPT‑2、Qwen‑2、Llama‑3.2。

**📈 对比分析**

与基线梯度逆向、黑盒提取等对比，重建率高达59–79%，语义相似度接近1，SGD下可实现精确重建。

**⚠️ 局限性**

局限性包括对大规模样本的记忆层空间开销、对Adam/AdamW的近似重建、对非IID或分布漂移的鲁棒性有限。

---

## 590. Easy Reads: A Python program for making Scientific Papers on arXiv more Reader Friendly and Accessible

**arXiv ID:** 2606.20550 | [PDF](https://arxiv.org/pdf/2606.20550v1)

**作者:** Vishal Verma `[一作]` `[通讯]` (American Museum of Natural History), Vishal Verma (American Museum of Natural History)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一款名为 Easy Reads 的自动化工具，可通过 arXiv URL 获取论文 LaTeX 源文件，修改正文字体大小和列数，并重新编译生成更易读的 PDF。

**💡 创新点**

将可定制的字体大小、单栏布局等设置直接应用于 LaTeX 源，提供命令行和代码编辑两种使用方式，突破传统 PDF 只能放大缩小的局限。

**🔧 技术方法**

使用 Python 编写脚本完成网络抓取、压缩包解压、LaTeX 文件修改、重新编译 PDF 等步骤，并通过 CLI 参数实现灵活配置。

**📊 数据集**

以 arXiv 上公开的论文 LaTeX 源文件作为输入数据集，覆盖物理、天体物理等多个学科。

**📈 对比分析**

通过对比原始 PDF 与 Easy Reads 处理后的 PDF，主观评估阅读体验提升（字体放大、单栏布局更便于扫描）但未给出量化性能指标。

**⚠️ 局限性**

兼容性受限于不同期刊自定义宏包，图表尺寸、边距等可能出现不一致；工具处于 alpha 版本，仍需手动调整和进一步测试。

---

