# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-16 | 今日论文总数: 451

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Sparse-RL: Breaking the Memory Wall in LLM Reinforcement Learning via Stable Sparse Rollouts

**arXiv ID:** 2601.10079 | [PDF](https://arxiv.org/pdf/2601.10079v1)

**作者:** Sijia Luo `[一作]` (Renmin University of China), Jing Zhang `[通讯]` (Renmin University of China)

**通讯引用:** 16016 | [OpenAlex ID](https://openalex.org/A5100345321)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出一种基于KV缓存压缩的稀疏RL训练框架，解决了稀疏采样与密集学习者之间的策略不匹配问题；

**💡 创新点**

创新点在于结合稀疏感知拒绝采样与重要性加权重构，实现在稀疏rollout下的稳定训练与训练时的稀疏感知能力；

**🔧 技术方法**

使用了KV缓存压缩技术（如SnapKV、R-KV）、稀疏感知拒绝采样、重要性加权重构以及GRPO框架；

**📊 数据集**

采用SimpleRL‑Zoo数据集（GSM8K、MATH）进行零RL训练，评估七个数学推理基准；

**📈 对比分析**

与密集GRPO基线和无纠正的稀疏GRPO对比，Sparse‑RL在保持或略优于密集基线的前提下，显著减少KV存储（约35–53%），在不同模型规模和压缩方法下均表现稳定；

**⚠️ 局限性**

局限性包括对开放式生成任务的适用性未知，以及严格的拒绝采样会导致采样效率下降，尤其在压缩预算极度紧张时。

---

## 2. Patient-Similarity Cohort Reasoning in Clinical Text-to-SQL

**arXiv ID:** 2601.09876 | [PDF](https://arxiv.org/pdf/2601.09876v1)

**作者:** Yifei Shen `[一作]` (University of Washington), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 7668 | [OpenAlex ID](https://openalex.org/A5064858748)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了ClinSQL基准，包含633个专家标注的MIMIC‑IV查询任务，要求多表连接、临床过滤、患者相似队列构建以及可执行SQL的生成；并为其提供了基于rubric的自动化评估与错误分类框架。

**💡 创新点**

创新点在于：①设计六类真实临床情境并引入患者相似队列与多步时间推理；②提出批判优先（Critical‑First）层次化rubric和执行检查机制，实现对SQL结构与结果的细粒度、可解释评估；③在基准中公开错误分类与评价流程，为未来模型改进提供明确方向。

**🔧 技术方法**

使用技术包括：大语言模型Chain‑of‑Thought自我修正、GPT‑5作为评判者的rubric‑based SQL与执行检查、SQL执行引擎对结果的可执行性与临床合理性验证，以及多模型对比实验。

**📊 数据集**

数据集为MIMIC‑IV v3.1电子健康记录数据库，并在此基础上生成多表查询任务。

**📈 对比分析**

比较方法：将22个开源与专有模型在易、中、难三个难度级别下的SQL得分与执行得分进行对比。实验结果显示：GPT‑5‑mini执行得分最高（全量74.7%，Hard 69.7%）；开源DeepSeek‑R1达69.2%；Gemini‑2.5‑Pro在Hard仅67.2%；整体表现仍远低于临床可靠水平，尤其在Hard子集。

**⚠️ 局限性**

局限性：①仅基于单一机构（MIMIC）和单一SQL环境，缺乏跨系统迁移性；②高度依赖专家标注，规模化成本高；③评估依赖GPT‑5作为判别者，可能存在偏差；④缺乏对不同数据库后端与多模态输入的适配研究。

---

## 3. CAFEDistill: Learning Personalized and Dynamic Models through Federated Early-Exit Network Distillation

**arXiv ID:** 2601.10015 | [PDF](https://arxiv.org/pdf/2601.10015v1)

**作者:** Boyi Liu `[一作]` (City University of Hong Kong), Yongxin Tong `[通讯]` (Beihang University)

**通讯引用:** 11057 | [OpenAlex ID](https://openalex.org/A5051874566)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种冲突感知的联邦退出蒸馏框架（FedExit），实现对早期退出网络（EEN）的个性化联邦学习。

**💡 创新点**

创新点在于：①同时解决客户端异质性与退出层深度冲突；②采用渐进的深度优先学生协调机制和跨客户端教师匹配；③通过客户端解耦的蒸馏方式消除通信开销，保持与FedAvg相当的传输量。

**🔧 技术方法**

使用技术包括：个性化联邦学习、早期退出网络、跨客户端知识蒸馏、冲突感知的学生选择、教师匹配（相似度加权）、FedAvg聚合、动态退出策略。

**📊 数据集**

在四个公开数据集上评估：CIFAR‑10、CIFAR‑100、TinyImageNet（图像）和AgNews（文本），采用Dirichlet分布模拟非 IID 情况。

**📈 对比分析**

与多种基线（FedAvg‑EE、FedProx‑EE、ScaleFL、DepthFL、FedPer‑EE、FedRep‑EE、FedBABU‑EE、FedAMP‑EE、pFedGraph‑EE、FedRoD‑EE、Ditto‑EE、FedPAC‑EE）对比，平均准确率显著提升（比最佳基线高 5–15%），并将推理成本降低 30.79%–46.86%。

**⚠️ 局限性**

局限性包括：①仅在图像与文本单模数据上验证，未覆盖多模任务；②对资源异质性（如算力、带宽）处理有限；③依赖于统一的退出结构和最后一层作为教师，可能在极端非 IID 或设备资源极低的场景下效果受限。

---

## 4. On Fun for Teaching Large Programming Courses

**arXiv ID:** 2601.09842 | [PDF](https://arxiv.org/pdf/2601.09842v1)

**作者:** Walid Maalej `[一作]` `[通讯]` (University of Hamburg), Walid Maalej (University of Hamburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

在大班软件工程基础课程中设计并实施十种物理游戏活动，以提高学生的参与度和概念理解。

**💡 创新点**

创新点是将物理活动与编程概念相映射，适用于数百学生的现场教学。

**🔧 技术方法**

技术手段包括课堂互动、手工活动和简单道具，而非软件技术。

**📊 数据集**

没有使用传统数据集，而是通过课程参与学生、访谈和问卷收集定性数据。

**📈 对比分析**

通过学生满意度和访谈对比显示活动提升了关注度和记忆力，效果在定性层面显著。

**⚠️ 局限性**

局限性包括缺乏量化实验、样本偏倚以及活动时间和执行难度。

---

## 5. The Geometry of Thought: Disclosing the Transformer as a Tropical Polynomial Circuit

**arXiv ID:** 2601.09775 | [PDF](https://arxiv.org/pdf/2601.09775v1)

**作者:** Faruk Alpay `[一作]` (Bahçesehir University), Bilge Senturk `[通讯]` (Bahçesehir University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

证明Transformer自注意力在高置信度极限下等价于tropical semiring（max-plus）矩阵乘法，从而揭示其相当于多步路径寻找算法（Bellman–Ford）并解释链式思考（CoT）机制。

**💡 创新点**

首次将Transformer的自注意力与tropical数学/幂等分析连接，提供了从几何/优化角度解释CoT的理论框架。

**🔧 技术方法**

使用tropical化（β→∞软max极限）、幂等分析、最大化代数与动态规划推导。

**📊 数据集**

论文未涉及具体数据集，主要为理论证明。

**📈 对比分析**

未做实验比较，主要是理论推导与图示说明。

**⚠️ 局限性**

现实Transformer运行在有限β，多头并行导致实际行为与理论极限偏差，实验验证与实际模型的贴合度待研究。

---

## 6. A Scoping Review of the Ethical Perspectives on Anthropomorphising Large Language Model-Based Conversational Agents

**arXiv ID:** 2601.09869 | [PDF](https://arxiv.org/pdf/2601.09869v1)

**作者:** Andrea Ferrario `[一作]` (University of Zurich), Alessandro Facchini `[通讯]` (Kozminski University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对2021–2025年间关于大型语言模型（LLM）对话代理（CA）人性化（anthropomorphisation）伦理研究进行系统性、范围性综述，提炼了概念基础、伦理风险与机遇、方法论路径，并提出未来研究与治理议程。

**💡 创新点**

创新点在于：①首次聚焦伦理视角的LLM‑CA人性化研究，构建统一的概念与方法框架；②识别出人性化的三维归因（认知、情感、行为/社会）及其与伦理风险的对应机制；③提出以生命周期（预先、部署中、部署后）为导向的治理评估流程，为政策制定提供可操作性。

**🔧 技术方法**

采用文献综述与PRISMA‑ScR规范，设计四区块搜索语句（人性化、技术、伦理、时间）并在Rayyan等工具中筛选、提取数据；对选定研究进行定性与主题性综合分析。

**📊 数据集**

检索来源包括五大数据库（Scopus、Web of Science、IEEE Xplore、PubMed、ACM Digital Library）和三大预印本仓库（arXiv、SSRN、PhilArchive），共收录910条记录，最终纳入22篇相关文章。

**📈 对比分析**

本研究不做模型性能对比，而是通过覆盖度与主题丰富度来评估综述质量：22篇文献涵盖伦理风险、机会、方法论等四大维度，展示了研究热点与空白；并对方法论类别进行聚类，形成三大方法家族（案例评估、设计/沟通控制、治理评估流程）。

**⚠️ 局限性**

局限性包括：①仅纳入伦理导向的文献，忽略未显式讨论伦理但包含人性化指标的实证研究；②未对研究质量进行评估，综述结果受样本代表性限制；③方法与治理建议多为高层次概念，缺乏可验证的实验或量化指标；④社会层面风险与机会研究不足，主要聚焦个体层面；⑤缺乏对人性化对长期依赖、社会信任等动态效应的纵向实证数据。

---

## 7. ReaMIL: Reasoning- and Evidence-Aware Multiple Instance Learning for Whole-Slide Histopathology

**arXiv ID:** 2601.10073 | [PDF](https://arxiv.org/pdf/2601.10073v1)

**作者:** Hyun Do Jung `[一作]` (Yonsei University), Hwiyoung Kim `[通讯]` (Yonsei University)

**通讯引用:** 1219 | [OpenAlex ID](https://openalex.org/A5083774696)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于多实例学习的可解释框架 ReaMIL，在传统 MIL 基础上添加轻量级证据选择头，使用预算化充分性约束实现仅需极少且空间紧凑的证据集即可完成诊断。

**💡 创新点**

创新点在于：①将证据选择作为主优化目标，设计四项约束（充分性、排除、连贯性、稀疏性）；②通过 Concrete 软阈值实现可微分选择；③提出 MSK 与 AUKC 两个量化证据效率的指标，能够客观衡量模型所需证据量。

**🔧 技术方法**

使用冻结的 UNI2-h 预训练特征、Transformer‑style TransMIL 编码器、Concrete（Gumbel‑sigmoid）选择头，并结合交叉熵、hinge、连贯性正则和稀疏性惩罚等多种损失项。

**📊 数据集**

在三大二分类任务上评估：TCGA‑NSCLC（LUAD vs LUSC）、TCGA‑BRCA（IDC vs Others）以及 PANDA（前列腺显著性 vs 非显著性）。

**📈 对比分析**

与标准 TransMIL 基线进行对比，ReaMIL 在 AUC、准确率和 F1 上保持相当或略优；同时在所有数据集上 MSK 仅为 7–16 块 tiles，AUKC 高达 0.81–0.86，表明模型能够用极少且空间紧凑的证据集实现高置信预测。

**⚠️ 局限性**

局限性包括：①依赖单一预训练特征 UNI2-h，缺乏跨域或不平衡数据的验证；②未进行病理学家用户研究，无法直接评估临床可用性；③在更复杂的多类别或多标签任务中需进一步测试。

---

## 8. Modeling conflicting incentives in engineering senior capstone projects: A multi-player game theory approach

**arXiv ID:** 2601.09944 | [PDF](https://arxiv.org/pdf/2601.09944v1)

**作者:** Richard Q. Blackwell `[一作]` (University of Illinois), Albert E. Patterson `[通讯]` (Texas A&M University)

**通讯引用:** 1010 | [OpenAlex ID](https://openalex.org/A5021598241)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

提出了一种基于贝叶斯游戏的正式框架，用来建模和分析工程教育中由大学、产业赞助商和学生团队共同参与的设计型结课项目的激励机制；

**💡 创新点**

创新点在于将结课项目视为一个含有不完全信息的顺序博弈，明确展示三方激励如何相互影响并形成三种典型的稳定均衡（合作、利用和成绩游戏）；

**🔧 技术方法**

使用了博弈论中的Stackelberg博弈、贝叶斯均衡（PBE）和简化的线性（affine）结果函数来捕捉技术质量、文档质量、时间表、赞助商契合度和可发布性等关键指标；

**📊 数据集**

该研究没有采用真实数据集，而是通过设定代表性参数值进行三种案例实验来说明不同激励结构下的均衡结果；

**📈 对比分析**

对比方法基于理论模拟，不同均衡下的效用和结果指标呈现出清晰的差异，说明激励设计对项目成功的决定性作用；

**⚠️ 局限性**

局限性包括：使用简化的单期模型、未考虑重复互动、声誉和内部团队动态，且结果未经过经验校准，因而缺乏对具体机构的预测能力。

---

## 9. SALP-CG: Standard-Aligned LLM Pipeline for Classifying and Grading Large Volumes of Online Conversational Health Data

**arXiv ID:** 2601.09717 | [PDF](https://arxiv.org/pdf/2601.09717v1)

**作者:** Yiwei Yan `[一作]` (Macquarie University), Guanfeng Liu `[通讯]` (Macquarie University)

**通讯引用:** 5379 | [OpenAlex ID](https://openalex.org/A5070515519)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计了基于大型语言模型的标准对齐提取管道SALP-CG，用于在线医疗对话中的健康数据分类与风险分级。

**💡 创新点**

将少样本提示、JSON Schema约束输出和确定性高危规则结合，实现了符合GB/T 39725-2020标准的统一分类与分级，并在多模型上保持一致的高效性。

**🔧 技术方法**

利用OpenAI、Groq、DeepSeek等平台的多种LLM（如gpt‑4o‑mini、gpt‑5、Llama‑3.1‑8b、Qwen‑3‑32b等），并辅以少样本提示、JSON Schema约束、正则与高危规则、文本分块与去重等技术。

**📊 数据集**

使用MedDialog‑CN 2020中文在线诊疗对话数据，人工标注1,000条样本作为评测基准。

**📈 对比分析**

对10种LLM在MCIF、MCCR、MSGR、micro‑F1四项指标上进行评估，最佳模型micro‑F1达0.90，MCCR≥0.98，MSGR≥0.97，MCIF接近1，显示跨模型鲁棒且性能优秀。

**⚠️ 局限性**

局限性包括仅针对中文单平台单时段数据，缺乏对实际隐私损害的深入评估，规则对罕见边缘情况的处理不够完善，且缺乏本地部署与跨域泛化能力。

---

## 10. BPE: Behavioral Profiling Ensemble

**arXiv ID:** 2601.10024 | [PDF](https://arxiv.org/pdf/2601.10024v1)

**作者:** Yanxin Liu `[一作]` (Yunnan University), Yunqi Zhang `[通讯]` (Yunnan University)

**通讯引用:** 14643 | [OpenAlex ID](https://openalex.org/A5108046754)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于模型自身行为档案的动态集成框架——Behavioral Profiling Ensemble（BPE），通过构造每个基模型的熵/边际等行为特征，在推理阶段不依赖邻域检索，直接给出动态权重，完成模型集成。

**💡 创新点**

创新点在于：①将动态加权的依据从跨模型对比转向单模型内部行为差异；②使用高斯噪声扰动在训练集上构造“即时测试场”，从而得到每个模型的行为档案；③通过Z-score标准化与指数映射实现快速、可解释的权重计算，显著降低存储和计算成本。

**🔧 技术方法**

核心技术包括：信息熵/边际作为行为度量；高斯噪声扰动生成行为档案；Z-score标准化与指数映射权重；多种基分类器（RF、XGBoost、SVM、KNN、MLP、NB等）以及Stacking、DES等基线方法的实现。

**📊 数据集**

实验使用：三种合成数据（高斯混合、非线性Hastie 10-2、多类别情形）以及40个来自OpenML/ UCI 的真实数据集，覆盖从低维到高维、二分类到多分类、数据量从几百到十万样本的多样化场景。

**📈 对比分析**

与12种基线（静态平均、加权、Stacking、LCA、KNORA、RRC、MCB、DES-AS等）对比，BPE 在合成实验中平均提升 0.1–0.2% 的准确率，在真实数据集上平均提升 0.1–0.2%，Friedman 排名始终位列第一，Wilcoxon 检验显示对大多数基线显著优于 5% 置信水平；同时保持了较低的存储和推理时间。

**⚠️ 局限性**

局限性包括：①目前仅采用熵或边际作为行为度量，可能不足以充分刻画所有模型的内部特征；②在高度不稳定或过拟合的树模型中，行为档案的鲁棒性有限；③仍需进一步探索更复杂的扰动方式、行为度量以及与传统跨模型动态选择策略的融合，以进一步提升泛化性能。

---

## 11. Lazy Evaluation: A Comparative Analysis of SAS MACROs and R Functions

**arXiv ID:** 2601.09839 | [PDF](https://arxiv.org/pdf/2601.09839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 12. Uncertainty-Aware Dynamic Knowledge Graphs for Reliable Question Answering

**arXiv ID:** 2601.09720 | [PDF](https://arxiv.org/pdf/2601.09720v1)

**作者:** Yu Takahashi `[一作]` (Fujitsu Research), Amin Beheshti `[通讯]` (Macquarie University)

**通讯引用:** 4318 | [OpenAlex ID](https://openalex.org/A5056293251)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了基于不确定性评估的动态知识图谱，并在临床问答中实现交互式检索与可视化。

**💡 创新点**

创新点在于将三元组的置信度评分与动态图更新结合，支持置信度感知检索并对比基线与置信度过滤结果。

**🔧 技术方法**

使用了实体抽取、图嵌入、LLM（GPT‑4.1‑mini）进行置信度估计与答案生成，图存储采用Neo4j，前端使用Streamlit。

**📊 数据集**

使用MIMIC‑III v1.4电子健康记录数据集进行个性化PKG构建和死亡率预测实验。

**📈 对比分析**

通过与基线历史图及置信度过滤图的对比，滤除低置信度三元组后在零样本LLM上死亡预测AUROC提升至0.575（+18.1%）AUPRC至0.135（+27.4%）。

**⚠️ 局限性**

局限在于对置信度阈值的依赖、LLM在无监督场景下的可解释性不足以及对外部知识库的整合尚未充分验证。

---

## 13. SciNets: Graph-Constrained Multi-Hop Reasoning for Scientific Literature Synthesis

**arXiv ID:** 2601.09727 | [PDF](https://arxiv.org/pdf/2601.09727v1)

**作者:** Sauhard Dubey `[一作]` `[通讯]` (Jaypee Institute of Information Technology), Sauhard Dubey (Jaypee Institute of Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 SciNets 系统，对查询本地概念图进行多跳图约束推理，生成跨领域的机制性解释。

**💡 创新点**

创新点：①将跨域科学合成框定为图约束的多跳推理任务；②提出基于符号深度、多样性和扎根稳定性的行为评估框架；③公开完整执行轨迹，支持可审计的系统行为分析。

**🔧 技术方法**

采用 LLM 辅助概念与关系抽取、概念图构建、最短路 / k 最短路 / 随机游走 / RAG 推理策略、ReAct 方式图探索、以及图约束下的自然语言实现。

**📊 数据集**

使用了 6 个跨域查询（Q1–Q6）以及 14 个任务的检索文献，检索来源包括 OpenAlex 与 DuckDuckGo，构造的概念图节点数约 140–170。

**📈 对比分析**

通过比较 4 种推理策略（完整/多样化、最短路、随机游走、RAG）在 14 个任务上的符号深度、扎根率、失败率进行评估；最短路方案稳定可靠但深度低；多样化策略深度高但失败率高；随机游走和 RAG 几乎全部失效；总体显示深度推理与扎根稳定性之间存在显著折衷。

**⚠️ 局限性**

局限性：高度依赖概念图质量；深层推理易导致自然语言实现失稳；无法保证科学真理；生成结果仅为探索性假设，需专家验证；对抽象领域的桥接表现不佳。

---

## 14. Federated Unlearning in Edge Networks: A Survey of Fundamentals, Challenges, Practical Applications and Future Directions

**arXiv ID:** 2601.09978 | [PDF](https://arxiv.org/pdf/2601.09978v1)

**作者:** Jer Shyuan Ng `[一作]`, Dusit Niyato `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述 Federated Unlearning 的基本概念、面临的主要挑战、现有方法与实际应用，并为未来研究提供参考框架。

**💡 创新点**

将 Federated Learning 与 Machine Unlearning 两大领域的研究系统整合，提出统一的 FUL 综述框架，首次按通信效率、安全隐私和资源约束对方法进行分类，并指出现有工作在这三大维度的不足与未来方向。

**🔧 技术方法**

主要综述了基于服务器端精确裁剪、回滚式与选择性更新、聚类分区、压缩/蒸馏与加速等技术路线，并讨论了其在边缘网络中的适配策略。

**📊 数据集**

本文为综述性工作，并未使用原始数据集；引用了公开论文中使用的公共数据集（如 CIFAR‑10、MNIST、医疗影像等）作为示例。

**📈 对比分析**

未进行统一实验，主要通过对已有研究的实验结果进行对比分析，概括不同方法在通信开销、恢复效率、隐私可验证性等方面的表现与权衡。

**⚠️ 局限性**

缺乏统一的评测基准和标准化实验流程，难以客观比较各方法；综述对实现细节的深度不足，未来需构建统一的数据集、评价指标和跨方法的对照实验。

---

## 15. FlowAct-R1: Towards Interactive Humanoid Video Generation

**arXiv ID:** 2601.10103 | [PDF](https://arxiv.org/pdf/2601.10103v1)

**作者:** Lizhen Wang `[一作]` (ByteDance Intelligent Creation), Mingyuan Gao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了FlowAct‑R1框架，实现实时、无限时长的人类视频生成，支持音频、文本多模态控制并实现细粒度行为切换；

**💡 创新点**

创新点在于结合Chunkwise Diffusion Forcing、结构化记忆库与自强化训练，三步蒸馏压缩为3NFE实时生成，同时通过多模态大模型辅助行为规划，解决流式生成中的误差累积与行为重复问题；

**🔧 技术方法**

核心技术包括MMDiT（Seedance）多模态扩散变换器、Chunkwise Diffusion Forcing、Structured Memory Bank、自强化（Self‑Forcing++）训练、三步蒸馏（Step Distillation + Few‑Step Score Distillation）、FP8量化与分层并行、异步VAE解码以及多模态大语言模型进行行为规划；

**📊 数据集**

使用综合视频与对话数据集，包含密集注释的动作片段，参考单张图像与音频文本对齐的标注；

**📈 对比分析**

与KlingAvatar 2.0、LiveAvatar、OmniHuman‑1.5等SOTA方法对比，用户研究表明FlowAct‑R1在长时序流式生成、25fps 480p实时性、1.5s TTFF以及动作自然度和唇同步准确度方面均显著优于对手；

**⚠️ 局限性**

限制包括对极长连续视频仍存在微小误差累积的风险、对极端动态场景或复杂多人物交互的适用性尚未充分验证，以及在极低算力环境下实现的可行性仍需进一步优化。

---

## 16. Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP

**arXiv ID:** 2601.09859 | [PDF](https://arxiv.org/pdf/2601.09859v1)

**作者:** Anant Mehta `[一作]` (Texas A&M University), Tianbao Yang `[通讯]` (Texas A&M University)

**通讯引用:** 5931 | [OpenAlex ID](https://openalex.org/A5023288846)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种自监督微调框架TuneCLIP，专门用于提升已有开源CLIP模型的整体性能而非单一任务；

**💡 创新点**

创新点在于（1）针对预训练模型的冷启动偏差提出Optimizer Statistics Recovery（OSR）两阶段warm‑up，恢复Adam统计量；（2）引入hinged global contrastive loss（HGCL）以缓解伪负样本导致的检索退化；

**🔧 技术方法**

使用的技术包括基于SogCLR的全局对比损失、AdamW优化、统计恢复、margin hinge 损失，以及大规模分布式训练框架FastCLIP；

**📊 数据集**

实验数据集包括DFN-12M、DFN-60M（自监督过滤的Web图文对），以及CC12M、ImageNet、COCO、Flickr30k、DataComp 38个基准；

**📈 对比分析**

与FastCLIP、OpenCLIP等基线对比，TuneCLIP在ImageNet-1k及其变体上提升约1.5%–2.5%，检索任务提升约6%+，DataComp整体提升约2%+，且在显著增加的GPU小时（约8–9h）后保持性能提升；

**⚠️ 局限性**

局限性包括：未对自监督数据做筛选或选择，且目前仅验证在CLIP模型，未来需扩展到其他自监督架构及更高效的数据挑选方法。

---

## 17. Advancing Model Refinement: Muon-Optimized Distillation and Quantization for LLM Deployment

**arXiv ID:** 2601.09865 | [PDF](https://arxiv.org/pdf/2601.09865v1)

**作者:** Jacob Sander `[一作]` (University of West Florida), Venkat R. Dasari `[通讯]` (DEVCOM Army Research Laboratory)

**通讯引用:** 582 | [OpenAlex ID](https://openalex.org/A5027071723)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向边缘设备的LLM压缩与任务专项化框架，联合使用GPTQ量化、LoRA、知识蒸馏、数据蒸馏、贝叶斯超参搜索与Muon优化器；

**💡 创新点**

创新点在于将上述技术整合为端到端管线，并证明Muonn优化在量化时显著提升鲁棒性，同时HPO自动将损失权重α设为1，表明纯KL对损失最有效；

**🔧 技术方法**

使用的技术包括GPTQ 4‑bit量化、LoRA参数高效微调、KL‑基知识蒸馏、Self‑Instruct自生成数据蒸馏、Optuna贝叶斯超参搜索以及Muon梯度谱范数优化；

**📊 数据集**

实验数据集主要为自生成的自指令数据集，用于8个基准（MMLU、ARC‑e、CommonsenseQA、HellaSwag、OpenBookQA、PIQA、SIQA、WinoGrande）以及相应的标准测试集；

**📈 对比分析**

与单纯GPTQ量化模型比较，在5~6个基准上准确率更高、量化误差更小，推理吞吐提升约50%，且Muonn优化的模型在量化后失真更小；

**⚠️ 局限性**

局限性包括：需进一步验证在更大规模模型或其他任务上的适用性；量化误差仍受模型架构与压缩参数影响；自生成数据质量依赖教师模型表现。

---

## 18. Multiverse: Transactional Memory with Dynamic Multiversioning

**arXiv ID:** 2601.09735 | [PDF](https://arxiv.org/pdf/2601.09735v1)

**作者:** Gaetano Coccimiglio `[一作]` (University of Waterloo), Srivatsan Ravi `[通讯]` (University of Southern California)

**通讯引用:** 628 | [OpenAlex ID](https://openalex.org/A5090605375)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

提出了一种新的多版本兼容软件事务内存（STM）系统，该系统在保留无版本化 STM 高性能的同时，支持长时间读写事务。

**💡 创新点**

创新点包括：① 动态多版本化——在需要时才为地址生成版本列表；② 多模式切换（Mode Q、Mode U 及其过渡模式），通过启发式和后台线程自动适配工作负载；③ 结合 Bloom 过滤器、锁表和版本列表表实现高效的版本化检查与回收；④ 采用 epoch‑based reclamation 解决并发垃圾回收问题；⑤ 在保证 opacity 的前提下兼顾无版本化 STM 的低延迟。

**🔧 技术方法**

使用的技术主要有：全局时钟 + 地址级版本化锁；Bloom 过滤器 + VLT（Version List Table）存储版本链；多模式执行路径与局部模式缓存；后台线程负责模式转换、bucket 去版本化和阈值计算；启发式决策（如事务重试次数、读取计数、最小 Mode U 读取计数等）；事务日志回滚与 TDB 标记；基于 epoch 的内存回收。

**📊 数据集**

实验使用的基准数据集包括（a,b）树、AVL 树、外部二叉树和哈希表；工作负载覆盖搜索、插入、删除以及少量（<1%）范围查询（RQ），并在均匀和 Zipfian 键分布下测试；同时使用专门的更新线程来模拟高冲突场景。

**📈 对比分析**

与 TL2、DCTL、NOrec、TinySTM 以及 Verlib 等现有透明 STM 进行对比。结果显示：
- 对于无 RQ 的常见工作负载，性能与最先进的无版本化 STM（如 DCTL）相当甚至略优；
- 对于包含长范围查询的工作负载，系统显著优于其它 STM，吞吐量可提升数十倍；
- 在能耗方面，系统利用多版本化实现的能效比最佳竞争对手高达 50 倍。

**⚠️ 局限性**

局限性：
- 需要对启发式参数（如 K1、K2、K3、S、L、P）进行手动调优；
- 主要在单机 64 核环境评估，分布式或大规模集群的表现尚未验证；
- 依赖 C++ 实现，跨语言移植受限；
- 对事务规模（读取计数）敏感，极大事务可能导致模式切换延迟；
- 仍在探索与现有数据库 MVCC 的兼容性。

---

## 19. SoK: Privacy-aware LLM in Healthcare: Threat Model, Privacy Techniques, Challenges and Recommendations

**arXiv ID:** 2601.10004 | [PDF](https://arxiv.org/pdf/2601.10004v1)

**作者:** Mohoshin Ara Tahera `[一作]` (University of Louisiana at Lafayette), Sajal Saha `[通讯]` (University of Northern British Columbia)

**通讯引用:** 335 | [OpenAlex ID](https://openalex.org/A5087889739)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对医疗领域LLM的隐私风险进行分阶段系统梳理与评估，提出完整的威胁模型、技术评测与改进建议。

**💡 创新点**

首次将隐私防护与医疗LLM的三个生命周期（预处理、联邦微调、推理）结合，形成基于阶段的威胁与防御映射，并给出跨阶段风险传播的洞见。

**🔧 技术方法**

综述并对比差分隐私、匿名化、合成数据、联邦学习安全聚合、SMPC、加密推理、KV缓存保护等技术。

**📊 数据集**

参考公开医疗数据集（MIMIC‑III/IV、MIMIC‑CXR、n2c2 等）及在此基础上的研究论文。

**📈 对比分析**

通过对比各技术在隐私泄露、模型准确性与算力消耗等维度的研究结果，总结了不同防御方案的优势与适用场景，但未给出统一实验评测。

**⚠️ 局限性**

主要局限在：缺乏统一的跨阶段评估框架、对罕见疾病与小样本情形的保护不足、对多轮联邦训练与推理侧信道的防护仍不完善，且多数技术在实际临床部署中的可行性与性能仍待验证。

---

## 20. Geometric Patterns of Meaning: A PHATE Manifold Analysis of Multi-lingual Embeddings

**arXiv ID:** 2601.09731 | [PDF](https://arxiv.org/pdf/2601.09731v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 21. FaTRQ: Tiered Residual Quantization for LLM Vector Search in Far-Memory-Aware ANNS Systems

**arXiv ID:** 2601.09985 | [PDF](https://arxiv.org/pdf/2601.09985v1)

**作者:** Tianqi Zhang `[一作]` (University of California San Diego), Tajana Rosing `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向大容量内存的分层残差量化方案FaTRQ，避免在近邻搜索中从慢存储读取完整向量。

**💡 创新点**

创新点在于将残差压缩为三值稀疏编码，支持渐进距离估计并通过远端内存与CXL加速器实现低延迟精细化。

**🔧 技术方法**

采用多层向量量化、残差量化、稀疏三值码、基于L2距离分解的渐进估计、轻量级线性校准模型，以及CXL Type‑2加速器。

**📊 数据集**

在Wiki（88M 768维SBERT）和LAION（100M 768维CLIP）两大预嵌入数据集上评估。

**📈 对比分析**

相较于GPUFAISS/ CAGRA-cuVS基线，FaTRQ在85-95%召回率下吞吐量提升2.6×至9.4×，SSD访问量下降约2.8×，存储效率提升2.4倍。

**⚠️ 局限性**

局限性包括对高维稀疏编码的实现复杂度、对远端内存带宽和延迟的依赖，以及在极高召回率下改进幅度减小。

---

## 22. Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models

**arXiv ID:** 2601.09719 | [PDF](https://arxiv.org/pdf/2601.09719v1)

**作者:** Hoyoon Byun `[一作]` (Yonsei University), Kyungwoo Song `[通讯]` (Yonsei University)

**通讯引用:** 520 | [OpenAlex ID](https://openalex.org/A5025711483)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Bounded Hyperbolic Tanh（BHyT）作为 Pre‑Layer Normalization 的无统计计算替代方案，能在深度 Transformer 中保持激活在非饱和区间并抑制激活幅度与方差的指数增长。

**💡 创新点**

创新点在于：① 用概率约束（Chebyshev 不等式）显式限制 tanh 的输入区间；② 仅在每个 Transformer block 开头一次完整统计，随后用轻量级方差近似代替重复归一化；③ 给出理论稳定性上界并证明在有限深度下方差增幅不超过 LNS。

**🔧 技术方法**

主要技术包括 tanh 非线性、概率输入限定、方差近似、块级统计、Chebyshev 约束、以及与现有 RMSNorm、LNS、Peri‑LN、DyT 的兼容性实现。

**📊 数据集**

使用 C4 语料进行预训练，Lima1k 进行指令微调，并在 ARC‑e、PIQA、HellaSwag、OpenBookQA、Winogrande、MMLU、BoolQ 等七个语言理解与推理基准上评估性能。

**📈 对比分析**

与 RMSNorm、LNS、Peri‑LN、DyT 等基线相比，BHyT 在预训练阶段提升约 15.8% 的训练速度，推理生成吞吐量提高约 4.2%，并在多项基准上达到或超过现有方法的精度，同时保持更好的深度稳定性。

**⚠️ 局限性**

局限性包括对 λ、κ 超参数的依赖、需要在每个 block 开头一次完整统计（虽然大幅减少但仍有开销）、理论证明建立在分布假设下，且在极深模型或不同架构（非 Transformer）上的表现尚未充分验证。

---

## 23. LCF3D: A Robust and Real-Time Late-Cascade Fusion Framework for 3D Object Detection in Autonomous Driving

**arXiv ID:** 2601.09812 | [PDF](https://arxiv.org/pdf/2601.09812v1)

**作者:** Carlo Sgaravatti `[一作]` (Politecnico di Milano), Giacomo Boracchi `[通讯]` (Politecnico di Milano)

**通讯引用:** 2977 | [OpenAlex ID](https://openalex.org/A5006716646)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于深度学习的Late‑Cascade Fusion 3D（LCF3D）框架，利用单目或双目RGB图像的2D目标检测与LiDAR点云的3D目标检测进行后期融合，既能滤除LiDAR检测的误报，又能通过对RGB检测未匹配的框生成3D锥体来恢复LiDAR漏检的目标；

**💡 创新点**

创新点包括：①将Late Fusion与Cascade Fusion融合成一种新的双阶段融合策略；②基于BEV聚类的3D框匹配与IoU优化；③利用RGB检测的高语义可靠性进行语义融合；④在双目场景下采用双目几何约束匹配；⑤整体框架与任何单模检测器解耦，具有良好的跨域泛化能力；

**🔧 技术方法**

核心技术：2D检测（Faster R‑CNN、DDQ‑DETR）、3D检测（PointPillars、PV‑RCNN、CenterPoint）、BEV聚类、图匹配（Jonker‑Volgenant算法）、三维锥体提议与Frustum PointNet定位、语义融合（基于RGB标签的概率融合）、双目epipolar匹配；

**📊 数据集**

实验数据集：KITTI（含单目/双目）与nuScenes（含nuImages实例分割数据）；

**📈 对比分析**

与单模LiDAR检测以及多模融合基线（MVXNet、BEVFusion、CLOCs‑PVCas、Frustum PointNet等）对比，LCF3D在KITTI的Pedestrian、Cyclist类别上达到了或超过现有SOTA，nuScenes中对不平衡类别（Bicycle、Motorcycle、Traffic Cone、Barrier）有显著提升；在跨域泛化实验中，LCF3D在KITTI→nuScenes、nuScenes→KITTI等迁移任务上均保持较高mAP/NDS，比MVXNet和BEVFusion表现更稳健；并且整体推理速度与内存消耗均低于大多数多模方案。

**⚠️ 局限性**

局限性：依赖两种传感器，若任一分支失效无法恢复漏检目标；对稀疏LiDAR点云的锥体定位精度有限，导致姿态估计不稳；在双目模式下需要双视角匹配的精确性，远距离或遮挡严重时恢复效果受限。

---

## 24. Improved Algorithms for Fair Matroid Submodular Maximization

**arXiv ID:** 2601.09860 | [PDF](https://arxiv.org/pdf/2601.09860v1)

**作者:** Sepideh Mahabadi `[一作]` (Microsoft Research), Jakub Tarnawski `[通讯]` (Microsoft Research)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5076652577)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了随机与确定性算法，用来在满足马约塔约束的前提下求解公平子模最大化问题（FMMSM），并给出近似解、近似公平下界以及大体量保证。

**💡 创新点**

核心创新在于：
- 通过构造增量/交替路径并随机选择，使得对每个颜色组的下界约束仅失去可调节的 (1‑ε) 份额；
- 同时保留对子模函数值的 0.499·OPT 近似，并给出高概率的尾部约束；
- 进一步给出了可在确定性环境下实现的版本，仍保持近似公平与子模价值。

**🔧 技术方法**

主要技术手段包括：
- 马约塔交叉的交换图与匹配构造，利用左向右与右向左的匹配生成互不相交的路径；
- 对路径进行“非对称截断”以保持独立性和颜色平衡；
- 随机化选择路径得到期望公平与价值保证；
- 采用子模函数的单调性与子模性证明价值下降控制；
- 采用 Chernoff / 超几何分布的尾部分析得到高概率保证。

**📊 数据集**

实验使用了三类真实数据集：
- 图覆盖：Pokec 社交网络（≈58 万节点、≈584 万边）；
- 聚类：葡萄牙银行电话营销数据（4521 条记录）；
- 推荐系统：Movielens 1M 数据集（≈3900 电影、≈6040 用户）。

**📈 对比分析**

与先前流式算法、非公平马约塔交叉、仅上界马约塔、随机添加等基线进行对比，指标为子模值与总公平违例。结果表明：
- 我们的随机算法在两三大实验中均优于或与最优非公平算法竞争；
- 通过调整 ε 可实现公平-效用的平滑折中；
- 基线算法在公平约束下表现较差，但在子模值上也不逊色；
- 整体性能满足理论保证并在实践中表现更好。

**⚠️ 局限性**

限制与未解决问题：
- 仅针对单调子模函数，未考虑非单调情况；
- 仅在马约塔交叉与单一公平约束下证明，无法直接推广到多重或重叠颜色组；
- 计算复杂度仍较高（O(N⁴) 上限，虽然在分区马约塔时可降至 O(N²)）；
- 对公平约束的完全满足仍是开放问题（是否存在常数因子近似无违例？）。

---

## 25. S$^2$F: Principled Hybrid Testing With Fuzzing, Symbolic Execution, and Sampling

**arXiv ID:** 2601.10068 | [PDF](https://arxiv.org/pdf/2601.10068v1)

**作者:** Lianjing Wang `[一作]` (Hunan University), Ji Wang `[通讯]` (National University of Defense Technology)

**通讯引用:** 11825 | [OpenAlex ID](https://openalex.org/A5100386450)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出了一种新的混合测试架构，融合模糊测试、符号执行和采样，并实现了基于此架构的工具。

**💡 创新点**

创新点包括：①通过轻量级执行树解决符号执行器的“睡眠”问题；②设计了基于分支难度和预期收益的采样原则，仅在难以覆盖且收益高的分支进行采样；③构建了灵活的协调机制，支持高级搜索策略，兼顾传统符号执行的精度与定制符号执行的可扩展性。

**🔧 技术方法**

采用了 AFL/AFL++ 作为模糊器、定制符号执行器（类似 SymCC）、SMT 求解器 Z3、路径条件的多面体抽象（TaichiPoly/Int）和 John‑Walk 采样方法，并结合了 LOB 剪枝策略、分支难度估计、未来收益估算等技术。

**📊 数据集**

在15个实际开源程序（objdump、libjpeg、tcpdump、libarchive、pngimage、jhead、pngfix、libxml2、readelf、openjpeg、gdk、nm、strip、imginfo、cyclonedds）上进行评测。

**📈 对比分析**

通过 6 次 24 小时实验，对比 AFL、AFL++、QSYM、DigFuzz、CoFuzz、SymCC 等基线工具；结果显示相对 SymCC 提高了 6.14% 的边界覆盖率、32.6% 的发现崩溃数；符号执行器的空闲时间从 56% 降至 12.88%；在大多数程序上取得领先。

**⚠️ 局限性**

局限性包括：仍依赖 LLVM IR 级别的插桩，无法处理内联汇编或 LLVM 内置函数；采样规模和参数需要经验调优；采样仅在预估收益高的分支使用，可能遗漏低收益但重要的路径；实验仅覆盖二进制程序，未来需进一步验证跨语言/跨平台的适用性。

---

## 26. Multilingual-To-Multimodal (M2M): Unlocking New Languages with Monolingual Text

**arXiv ID:** 2601.10096 | [PDF](https://arxiv.org/pdf/2601.10096v1)

**作者:** Piyush Singh Pasi `[一作]` `[通讯]` (Amazon), Piyush Singh Pasi (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

使用仅英语文本训练的轻量级线性映射，将多语文本嵌入对齐到预训练的多模态空间，实现跨语言零样本检索与生成。

**💡 创新点**

创新点在于仅用英语单语文本及少量线性层即可完成多语文本与多模态空间的对齐，显著降低对多模态多语料的依赖。

**🔧 技术方法**

采用线性投影（2 层）+ MSE 与结构保持损失，训练时保持所有预训练编码器冻结。

**📊 数据集**

使用公开的多语文本编码器（M‑MPNET、LaBSE 等）与多模态编码器（CLIP、CLAP、FLUX）以及合成多语检索/生成数据集（XTD、AudioCaps 33 语、MSCOCO‑30K 9 语）。

**📈 对比分析**

对比方式：与单语 CLIP、K‑ALIGN、M‑CLIP‑ST 等多模态模型以及多语多模态模型对齐；在 XTD‑T2I、I2T 上平均 Recall@10 约 89.5%（英文 94.9%），在 AudioCaps、Clotho 上也接近 SOTA，生成任务 FID/IS 虽低于原 FLUX 但仍具备多语文本生成能力。

**⚠️ 局限性**

局限在于仅对全局句子表示对齐，缺乏 token‑level 对齐，难以支持多模态大语言模型及细粒度生成；且评测集为机器翻译合成，缺少人工校验。

---

## 27. Enhancing LUT-based Deep Neural Networks Inference through Architecture and Connectivity Optimization

**arXiv ID:** 2601.09773 | [PDF](https://arxiv.org/pdf/2601.09773v1)

**作者:** Binglei Lou `[一作]` (University of Sydney), Philip Leong `[通讯]` (University of Sydney)

**通讯引用:** 5798 | [OpenAlex ID](https://openalex.org/A5107994859)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了SparseLUT框架，结合多PolyLUT子神经元加法器和非贪心稀疏训练，显著降低FPGA上LUT‑DNN的资源占用和延迟，同时提升准确率。

**💡 创新点**

① 通过将A个PolyLUT子神经元聚合为加法器，减少LUT规模并降低推理延迟；② 在训练阶段实施非贪心的动态稀疏算法，既实现固定fan‑in，又在连接搜索空间中探索更优权重组合。

**🔧 技术方法**

采用PolyLUT、批归一化与量化激活、加法器重构、非贪心稀疏训练（含随机游走噪声）、PyTorch＋Brevitas量化训练以及Vivado RTL合成。

**📊 数据集**

使用MNIST、Jet Substructure Classification（JSC）和CIFAR‑10数据集进行验证，展示不同任务上的效果。

**📈 对比分析**

与LogicNets、PolyLUT、NeuraLUT等基线在相同准确率下对比，LUT占用下降2.0–13.9倍，推理延迟降低1.2–1.6倍；在相同配置下准确率提升1.4–2.1%（MNIST）和0.94%（JSC）。

**⚠️ 局限性**

适用于已预设fan‑in约束的LUT‑DNN，复杂任务（如CIFAR‑10）准确率仍低于90%；稀疏搜索与训练成本相对较高，且对大词宽或非加性结构的适用性尚待验证。

---

## 28. AmbShield: Enhancing Physical Layer Security with Ambient Backscatter Devices against Eavesdroppers

**arXiv ID:** 2601.09867 | [PDF](https://arxiv.org/pdf/2601.09867v1)

**作者:** Yifan Zhang `[一作]` (Aalto University), Zhu Han `[通讯]` (University of Houston)

**通讯引用:** 86255 | [OpenAlex ID](https://openalex.org/A5063667378)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出并实现了 AmbShield，一个利用环境中天然存在的背向散射设备（AmBD）来提升物理层安全性的系统。

**💡 创新点**

创新点在于将 AmBD 同时用作友好干扰器与被动中继，无需额外发射功率和硬件，且在时钟同步与信道估计误差下仍能保持低泄露概率。

**🔧 技术方法**

采用随机背向散射模型、Fox‑H 函数分析、PDF/CDF 求解、封装的 Secrecy Outage Probability（SOP）以及高 SNR 渐进与多样性分析等技术。

**📊 数据集**

通过 Monte‑Carlo 仿真与理论推导对比，使用标准 Rayleigh 衰落与路径损耗模型（无专用公开数据集）。

**📈 对比分析**

相较于传统 AN、波束成形和 RIS 方案，在不同 AmBD 数量、距离以及 Eve 信噪比下，SOP 显著降低（低至 10⁻³），且在时钟误差和 CSI 误差条件下保持较优性能。

**⚠️ 局限性**

局限性包括：依赖 AmBD 数量与位置分布；对多天线或大规模网络的扩展未充分验证；在极端同步/CSI 失真下性能可能衰减；理论推导较为复杂，Fox‑H 表达式不易直观解释。

---

## 29. Thinking Long, but Short: Stable Sequential Test-Time Scaling for Large Reasoning Models

**arXiv ID:** 2601.09855 | [PDF](https://arxiv.org/pdf/2601.09855v1)

**作者:** Michael R. Metel `[一作]` (Huawei Noah's Ark Lab), Prasanna Parthasarathi `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种名为 Min-Seek 的序列测试时缩放方法，旨在通过训练‑free 的方式提升大型推理模型在长推理链上的准确性并实现稳定性。

**💡 创新点**

创新点在于：1) 仅保留最短已生成的推理思路的 KV 缓存，过滤掉冗长或错误的推理路径；2) 采用无位置嵌入的 KV 缓存并动态连续地为每轮推理加位置编码，使模型能够突破原始上下文长度限制，支持无限推理；3) 通过简化 KV 缓存更新实现线性时间复杂度。

**🔧 技术方法**

主要技术包括自定义 KV 缓存管理、动态连续位置嵌入、Sequential test‑time scaling（Min‑Seek）、对比实验中使用的 Budget Forcing、以及使用 <\think> 与 Wait 触发终止的策略。

**📊 数据集**

实验数据集涵盖五个推理任务：AIME 2024、AMC 2022/23、GPQA Diamond、MATH‑500 与 MMLU‑Pro。

**📈 对比分析**

与标准模型生成和 Budget Forcing 进行对比；在所有任务中，Min‑Seek 在 2、4、6、10、20、50、100 甚至无穷多次推理周期下均表现出更高的归一化准确率（最高提升约 1.15），并且生成速度比 Budget Forcing 快约 30–44%，同时能够突破上下文长度限制。

**⚠️ 局限性**

局限性：仅适用于已有的 LRM；对简单任务提升有限，且推理时间略有增加；需要手动设置触发词；未对模型训练过程做改动，使用前需已有高质量模型。

---

## 30. Skill-Aware Data Selection and Fine-Tuning for Data-Efficient Reasoning Distillation

**arXiv ID:** 2601.10109 | [PDF](https://arxiv.org/pdf/2601.10109v1)

**作者:** Lechen Zhang `[一作]` (University of Michigan), Lu Wang `[通讯]` (University of Michigan)

**通讯引用:** 25441 | [OpenAlex ID](https://openalex.org/A5100364413)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向技能的数学推理蒸馏框架，结合技能归因、弱项驱动的数据采样和技能链增强的微调；

**💡 创新点**

创新点在于将层级技能树映射到问题上，按学生模型弱项自适应采样训练样本，并在训练中显式嵌入完整的技能链，实现高效且可解释的知识迁移；

**🔧 技术方法**

使用LLM技能归因、基于技能准确率的加权采样、技能链前缀增强的标准SFT，实验基于Qwen3-4B/8B模型；

**📊 数据集**

使用100K教师生成的数学QA对（OpenMathReasoning）、AMC23、AIME2024/25、MATH L5、OlympiadBench等数据集；

**📈 对比分析**

与随机采样、全量100K数据SFT及单一技能aware方法对比，在仅1K样本下平均提升约+1.6%/1.4%，且全量数据反而性能下降，验证数据选择更重要；

**⚠️ 局限性**

局限在于依赖手工构建的技能树、技能准确率评估可能噪声大、仅验证4B/8B规模、未验证更大模型或自动学习技能层级。

---

## 31. MathDoc: Benchmarking Structured Extraction and Active Refusal on Noisy Mathematics Exam Papers

**arXiv ID:** 2601.10104 | [PDF](https://arxiv.org/pdf/2601.10104v1)

**作者:** Chenyue Zhou `[一作]` (Nanjing University of Aeronautics and Astronautics), Yanbiao Ma `[通讯]` (Beijing Key Laboratory of Research on Large Models and Intelligent Governance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MathDoc 基准，用于评估在含噪声的纸质高中数学考试卷中，模型对题干、选项、图形等结构化信息的提取以及主动拒绝不完整输入的能力。

**💡 创新点**

创新点包括①构建真实噪声场景下的 3,609 题目集并专门标注不可识别样本；②提出涵盖题干准确率、视觉相似度与拒绝指标的多维评估框架；③系统分析不同模型拒绝阈值并验证通过裁剪提高拒绝率的有效性。

**🔧 技术方法**

主要使用多模态大语言模型（如 Qwen3-VL、Gemini‑2.5‑Pro 等）进行端到端评测，并通过 Levenshtein 距离、语义相似度判别器等技术对结果进行量化；数据处理采用三阶段流水线（模型预标注、专家校正、拒绝标记）。

**📊 数据集**

使用 MathDoc 数据集，该数据集包含 Choice、Fill、Solve 三类题目，含图形、文本密集与不可识别等多种噪声样本。

**📈 对比分析**

与 SOTA MLLMs 进行对比，评估指标包括文本准确率（S_text）、视觉相似度（S_img）和拒绝的精确度/召回率/F1；结果显示 Qwen3‑VL‑30B 文本准确率高达 0.89，视觉相似度 0.63，但拒绝召回仅 0.14，整体性能虽优于多阶段流水线，但仍缺乏主动拒绝能力。

**⚠️ 局限性**

局限性在于未对模型架构或预训练目标进行根本改进，导致在信息缺失时仍倾向于推测或强制转录，评估工具更像诊断而非提供可直接迁移的算法解决方案。

---

## 32. ViSIL: Unified Evaluation of Information Loss in Multimodal Video Captioning

**arXiv ID:** 2601.09851 | [PDF](https://arxiv.org/pdf/2601.09851v1)

**作者:** Po-han Li `[一作]` (University of Texas at Austin), Sandeep Chinchali `[通讯]` (University of Texas at Austin)

**通讯引用:** 772 | [OpenAlex ID](https://openalex.org/A5024120306)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于信息理论的统一评价多模态视频摘要信息损失的指标；

**💡 创新点**

创新点在于用条件点互信息量化视频到摘要的信息丢失，能够跨视觉、文本多模态形式进行直接比较；

**🔧 技术方法**

技术方法包括使用视觉语言模型（VLM）自回归推理生成视频详细字幕作为文本代理，并通过关键字概率近似计算句子概率，进而得到信息损失分数；

**📊 数据集**

实验数据集主要为MVBench（Episodic Reasoning子集）和LongVideoBench（Sequence of Scenes子集），并在此基础上构建多种摘要格式；

**📈 对比分析**

通过与VQA准确率和人类实验结果对比，信息损失分数与视频理解任务表现呈显著负相关；在三图像摘要与完整视频比较时，信息损失更低、处理成本更低，能在保持约80%准确率的同时将人类反应时间减少约20秒；

**⚠️ 局限性**

局限性包括：仅适用于能输出文本代理的VLM，无法直接评估包含音频的摘要；指标受底层VLM能力影响；目前不支持直接用于摘要生成或优化任务。

---

## 33. A Control Theoretic Approach to Decentralized AI Economy Stabilization via Dynamic Buyback-and-Burn Mechanisms

**arXiv ID:** 2601.09961 | [PDF](https://arxiv.org/pdf/2601.09961v1)

**作者:** Zehua Cheng `[一作]` (FLock.io), Jiahao Sun `[通讯]` (University of Oxford)

**通讯引用:** 3532 | [OpenAlex ID](https://openalex.org/A5107898736)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并验证了一种基于 PID 控制理论的动态回购销毁机制（DCBM），以稳定去中心化 AI 网络的原生代币经济。

**💡 创新点**

将控制理论与 tokenomics 相结合，构建了受偿付约束、积分与微分调节的闭环买回机制，显著降低波动并增强对 MEV 攻击的抵御能力。

**🔧 技术方法**

使用 PID 控制器、有限差分模拟、Jump‑Diffusion 过程、常数乘积做市商模型、EVM 固定点算术、以及 FGSM/PGD/CW 等游戏理论攻击模型。

**📊 数据集**

采用合成的跳跃扩散模拟数据以及真实的 AI 模型使用统计（如 Hugging Face 下载量、API 调用）和去中心化 AI 代币价格历史（Bittensor、Render 等）。

**📈 对比分析**

在 1,000 次 Monte‑Carlo 试验的牛市、熊市、高波动、需求冲击等六大场景下与无买回、固定比例、阈值、RL PPO、MPC 等基准对比，DCBM 将价格波动率降低约 66%，运营者离职率从 19.5% 降至 8.1%，并在 MEV 攻击下表现出更高的稳健性。

**⚠️ 局限性**

模型假设线性化、对极端非线性冲击敏感、EVM 计算逼近导致 Gas 成本约 150,000，未在主网验证，缺乏自适应增益调度，且对极端 MEV 攻击仍存在一定风险。

---

## 34. Investigating Tool-Memory Conflicts in Tool-Augmented LLMs

**arXiv ID:** 2601.09760 | [PDF](https://arxiv.org/pdf/2601.09760v1)

**作者:** Jiali Cheng `[一作]` (University of Massachusetts Lowell), Hadi Amiri `[通讯]` (University of Massachusetts Lowell)

**通讯引用:** 1374 | [OpenAlex ID](https://openalex.org/A5074007015)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并系统研究了工具-记忆冲突（Tool‑Memory Conflict, TMC），即工具输出与LLM内部参数知识的矛盾，评估其在多种任务中的普遍性，并尝试使用提示和检索增强技术缓解冲突。

**💡 创新点**

创新点在于：①首次定义并量化TMC及其与现有知识冲突类型的区别；②从模型规模、任务领域和工具使用偏好等维度剖析冲突产生机制；③对比多种冲突解决策略（提示、意见式提示、警觉提示、RAG）在不同模型上的效果。

**🔧 技术方法**

使用的技术主要包括：工具调用（API/function call）、提示工程（Vigilant Prompt、Opinion‑Based Prompt）、检索增强生成（RAG）、以及基于概率的偏好度量（Memory Bias、Tool Bias）。

**📊 数据集**

实验数据集涵盖：MMLU、GSM8K、MATH‑500、AIME 2024、GPQA Diamond 等涵盖 STEM、社科、人文及长尾知识等多领域问答与推理任务。

**📈 对比分析**

与基线自由决策（Conflict）相比，警觉提示可减少 1–4% 冲突率，意见式提示效果不一，RAG 在大模型中可降低 2–6% 冲突率、在中小模型中可降低 8–15%；总体来看，RAG 对缓解冲突最为显著，但所有方法提升有限，说明冲突仍难以根除。

**⚠️ 局限性**

局限性包括：未覆盖所有主流LLM（如Claude）；仅评估了有限的冲突解决方法；实验仅在预定义数据集上进行，缺乏真实场景验证；未探讨工具质量与噪声对冲突的影响。

---

## 35. Context Volume Drives Performance: Tackling Domain Shift in Extremely Low-Resource Translation via RAG

**arXiv ID:** 2601.09982 | [PDF](https://arxiv.org/pdf/2601.09982v1)

**作者:** David Samuel Setiawan `[一作]` (University of Melbourne), Jey Han Lau `[通讯]` (University of Melbourne)

**通讯引用:** 4139 | [OpenAlex ID](https://openalex.org/A5032767467)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究低资源语言在域移位下的翻译性能衰减，并提出了NMT+LLM混合框架来恢复性能

**💡 创新点**

证明域迁移对性能的主要影响来自检索上下文量而非检索算法，并将LLM用作“安全网”纠错

**🔧 技术方法**

结合细粒度词级检索、句级检索与检索增强生成（RAG）以及LLM后编辑（Gemini 2.5 Flash）

**📊 数据集**

使用东印尼原住民语言Dhao的圣经新旧对照文本以及语法书构建的并行句子与词典数据集

**📈 对比分析**

相较于单一NMT或LLM基线，混合模型在OT测试集上chrF++由27.11提升至35.21，接近NT基准（36.17），并在spBLEU上从7.66提升至19.88

**⚠️ 局限性**

实验仅覆盖单一语言与单一域移位，检索算法与上下文组合的最佳权衡仍需进一步探索，且缺乏对更广泛语言或领域的验证

---

## 36. Deriving Character Logic from Storyline as Codified Decision Trees

**arXiv ID:** 2601.10080 | [PDF](https://arxiv.org/pdf/2601.10080v1)

**作者:** Letian Peng `[一作]` (University of California), Jingbo Shang `[通讯]` (University of California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Codified Decision Trees (CDT)，一种从故事情节中自动提取可执行、情境感知的角色行为档案的方法。

**💡 创新点**

创新点在于：①将行为规则组织成层级化、可解释的决策树；②采用 LLM 生成候选 if–then 触发器并通过 NLI 进行严格验证；③引入聚类与 instruction‑following 嵌入，使触发器提取更精准；④通过递归假设‑验证构造树，实现全局与局部行为的兼顾。

**🔧 技术方法**

技术手段包括：大语言模型（用于规则生成与判别）、自然语言推理（NLI）验证、文本嵌入（语义 + instruction‑following）、K‑Means 聚类、递归树构建与节点深度控制、Top‑K 规则挑选与文本化（wikified）。

**📊 数据集**

使用的数据集为：①细粒度 Fandom Benchmark（45 个角色、20,778 场景–动作对）；②Bandori 对话 Benchmark（40 个角色、7,866 对）；③BanG Dream 事件对话（77,182 对）作为规模化实验。

**📈 对比分析**

与基线（Vanilla、Fine‑tuning、RICL、ETA、Human Profile、Codified Human Profile）比较，CDT 在两大 benchmark 上均实现最高 NLI 分数，超过人类编写档案和所有自动化基线；轻量版 CDT‑Lite 进一步提升效率且仍保持最优性能。

**⚠️ 局限性**

局限性包括：仅依赖情节提取的场景–动作对，未利用已有的角色原始设定或多模态信息；模型为离线静态树，缺乏增量更新和持续学习能力；目前仅适用于单角色情境，未覆盖多角色交互或动态剧情演化。

---

## 37. Transition Matching Distillation for Fast Video Generation

**arXiv ID:** 2601.09881 | [PDF](https://arxiv.org/pdf/2601.09881v1)

**作者:** Weili Nie `[一作]` (NVIDIA), Arash Vahdat `[通讯]` (NVIDIA)

**通讯引用:** 4917 | [OpenAlex ID](https://openalex.org/A5038984764)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将大型视频扩散模型蒸馏为几步生成器

**💡 创新点**

采用解耦骨干与轻量流头的结构，并结合两阶段训练和流头回滚实现高效蒸馏

**🔧 技术方法**

转移匹配蒸馏（TMD）、MeanFlow预训练、改进的DMD2‑v、Conv3D GAN判别器等技术

**📊 数据集**

使用Wan2.1 1.3B/14B文本到视频模型为教师，在500k VidProM+Qwen‑2.5生成的文本‑视频对训练

**📈 对比分析**

与DOLLAR、T2V‑Turbo‑v2、rCM等基线比较，TMD在VBench总分、质量与语义分均领先，用户偏好实验也显示显著优势

**⚠️ 局限性**

仍受限于大规模模型的预训练成本、对高分辨率和时长的扩展性有限，以及需调参的复杂性

---

## 38. OUTLINEFORGE: Hierarchical Reinforcement Learning with Explicit States for Scientific Writing

**arXiv ID:** 2601.09858 | [PDF](https://arxiv.org/pdf/2601.09858v1)

**作者:** Yilin Bao `[一作]` (University of California San Diego), Zayden Yang `[通讯]` (Sheltered AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出将科学论文生成视为层级文档规划的强化学习问题，使用可编辑的论文提纲作为中间状态，并通过离散动作逐步完善提纲，最终生成完整论文。

**💡 创新点**

创新点在于：①把提纲演化抽象为结构化的state–action序列；②引入两阶段优化（backward提纲重建+forward价值引导RL）实现全局一致性与引用可靠性；③设计针对文档规划与引用一致性的全新评测基准。

**🔧 技术方法**

采用强化学习（PPO+价值模型）与偏好学习，构建结构化动作空间；利用检索增强生成（RAG）与结构化编辑函数；并结合人类编辑轨迹生成训练数据。

**📊 数据集**

数据集来源于从arXiv解析的1,500篇论文，提取章节、段落层次结构并生成编辑对（state–action）样本；评测使用自定义的survey生成任务和同类系统的公开数据。

**📈 对比分析**

与SurveyForge、AutoSurvey、GPT‑4o‑mini、Claude‑3.5‑Haiku、Llama‑3.1‑Instruct‑70B等基线对比，采用Precision/Recall/F1等指标；在200/300步的生成预算下，细化后模型在精确度和召回率上均显著优于基线，甚至小型模型微调后超过部分大型通用模型。

**⚠️ 局限性**

局限性包括：①评估依赖LLM自动判断，可能带偏；②实验仅覆盖survey生成，未涵盖原始研究论文等其他写作场景；③动作空间手工设计，跨领域适用性受限；④长序列编辑易出现误差累积，缺乏全局修订或回滚机制。

---

## 39. Difficulty-guided Sampling: Bridging the Target Gap between Dataset Distillation and Downstream Tasks

**arXiv ID:** 2601.10090 | [PDF](https://arxiv.org/pdf/2601.10090v1)

**作者:** Mingzhuo Li `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**通讯引用:** 3105 | [OpenAlex ID](https://openalex.org/A5063903016)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在数据集蒸馏中引入难度信息，通过后置采样（DGS）和难度引导生成（DAG）来缩小蒸馏目标与下游任务之间的目标差距。

**💡 创新点**

创新点是将任务特定的难度分布作为采样与生成的核心依据，既可作为采样模块又可直接在生成过程中引导，显著提升蒸馏数据集对分类任务的适配性。

**🔧 技术方法**

使用扩散模型（DiT/Minimax）、基于难度的采样与分布平滑技术、K‑means难度聚类指导以及信息瓶颈理论分析。

**📊 数据集**

在 ImageNet 的三个10类子集（ImageWoof、ImageNette、ImageIDC）上进行实验，并使用 ConvNet‑6、ResNet‑18、ResNetAP‑10 等下游网络进行评估。

**📈 对比分析**

与随机采样、K‑center、DM、IDC‑1、D4M、DiT、Minimax 等方法对比，DGS 在大多数 IPC 设置下提升 1–3% 的 top‑1 准确率（例如在 ImageWoof 上 IPC=50 的 ResNet‑18 从 53.9% 提升至 54.3%），DAG 在 MGD3 基线上也实现了约 0.5–1% 的提升。

**⚠️ 局限性**

局限性包括：仅在图像分类任务验证，难度定义为置信度倒数可能不适用于其他任务；需要额外的采样或生成开销；对不同数据域（如 SVHN）和更大规模数据集的泛化性尚未充分验证。

---

## 40. CoCoPlan: Adaptive Coordination and Communication for Multi-robot Systems in Dynamic and Unknown Environments

**arXiv ID:** 2601.10116 | [PDF](https://arxiv.org/pdf/2601.10116v1)

**作者:** Xintong Zhang `[一作]` (Duke Kunshan University), Meng Guo `[通讯]` (Peking University)

**通讯引用:** 1253 | [OpenAlex ID](https://openalex.org/A5032581175)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出一种 CoCoPlan 框架，实时联合优化多机器人协同任务规划与团队级间歇通信，支持未知空间与任务分布下的动态协作。

**💡 创新点**

创新点在于将任务分配与通信事件统一编码到分支限界（Branch‑and‑Bound）搜索节点中，并设计自适应目标函数权衡任务效率与通信延迟，同时采用迭代算法在时间-空间上优化通信事件。

**🔧 技术方法**

主要技术包括：分支限界搜索、基于通信质量的动态连通图建模、时间约束任务规划、通信事件的迭代优化以及在线重规划与可扩展性分析。

**📊 数据集**

使用仿真数据集：DARPA SubT 地图、地下洞穴环境、三维城市空间，以及大规模 100 机器人部署；硬件实验在办公室服务场景与 3D 灾难响应实验室中验证。

**📈 对比分析**

与六种基线方法（FIX、FPMR、FRDT、FIMR、RING、Greedy）比较，CoCoPlan 在任务完成率上提升约 40%–60%，通信次数下降 90% 以上，且在不同任务分布与规模下保持低方差与高效率。

**⚠️ 局限性**

局限性包括：仍需较高计算资源（分支限界的复杂度随机器人数量指数级增长），对通信质量模型的假设可能不适用于极端干扰环境；以及在极大规模（千级机器人）或完全无网络时仍需改进分布式执行策略。

---

## 41. LLM-Based Agentic Systems for Software Engineering: Challenges and Opportunities

**arXiv ID:** 2601.09822 | [PDF](https://arxiv.org/pdf/2601.09822v1)

**作者:** Yongjian Tang `[一作]` (Siemens AG), Thomas Runkler `[通讯]` (Siemens AG)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统地综述了基于大型语言模型的多智能体系统在软件工程生命周期各阶段的应用，梳理现有方法、框架与挑战，并提出未来研究方向

**💡 创新点**

首次提出以多智能体协同为核心的框架，强调人机协作、计算成本优化和数据收集的系统性视角，并为跨阶段协作提供概念模型

**🔧 技术方法**

结合LLM技术、agentic框架（CrewAI、LangGraph 等）、通信协议（CNP、A2A、MCP）以及推理增强、检索增强生成等方法

**📊 数据集**

引用并基于工业标准数据集如 PURE、PROMISE、humanEval、GSM8K、BugBench、Tests4Py 等进行评估设计，未来将通过这些基准进行对比

**📈 对比分析**

建议以标准 SE 基准为衡量，使用统一实验环境对比不同多智能体设计（规划‑执行、迭代自我改进、人机混合）和不同 LLM（GPT‑5、Gemini‑3‑pro、Claude‑v4.5、LLaMA‑4、DeepSeek‑V3、DeepSeek‑R1）的表现，虽然目前尚无实验结果

**⚠️ 局限性**

主要局限在于缺乏实证实验、难以验证假设；多智能体系统在隐私、成本、动态发现、人工反馈等实际部署场景中存在挑战；缺少统一多智能体 SE 评测基准；模型专门化不足，难以覆盖所有领域角色

---

## 42. Brief but Impactful: How Human Tutoring Interactions Shape Engagement in Online Learning

**arXiv ID:** 2601.09994 | [PDF](https://arxiv.org/pdf/2601.09994v1)

**作者:** Conrad Borchers `[一作]` (Carnegie Mellon University), Kenneth R. Koedinger `[通讯]` (Carnegie Mellon University)

**通讯引用:** 26125 | [OpenAlex ID](https://openalex.org/A5062550465)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究短时人类辅导访视与中学生在线数学练习中参与度的关系，并量化访视对步数完成率的即时与持续提升。

**💡 创新点**

首次将Zoom访谈对话与MATHia系统日志在分钟级精度对齐，构建混合效应模型评估访视长度、时间与对话内容对参与度的不同影响；提出可解释的访视调度与对话指导框架。

**🔧 技术方法**

采用混合效应 Poisson GLMM、句子‑BERT 嵌入与随机化检验、定性编码、可视化等技术实现对齐、建模与解释。

**📊 数据集**

191名7‑8年级学生在 MATHia 平台共练习 2,075 小时，配合 Zoom 日志和 WhisperX 转录，包含 1,022,884 条交易记录。

**📈 对比分析**

与未被访视学生对比，访视期间步数完成率提升 61%（IRR=1.61）；访视长度与时间交互揭示边际递减与后期更大即时提升；高提升对话与低提升对话差异显著，AUC 0.69。

**⚠️ 局限性**

对话匹配率仅 73% 可能导致样本偏倚；仅关注即时参与度，未直接关联长期学习成效；对话主体主要为辅导者，学生回应未深入分析；访视时长与时间分布受课堂时长限制，后期效果受限。

---

## 43. Epistemology gives a Future to Complementarity in Human-AI Interactions

**arXiv ID:** 2601.09871 | [PDF](https://arxiv.org/pdf/2601.09871v1)

**作者:** Andrea Ferrario `[一作]` (University of Zurich), Juan M. Durán `[通讯]` (TU Delft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过计算可靠性理论（Computational Reliabilism, CR）重新定位人机协同（complementarity）概念，将其视为人机交互过程可靠性的指标，并提出基于此的最小报告清单与高效协同度量。

**💡 创新点**

创新点在于：①把协同从单纯的相对准确性指标转化为可解释的可靠性证明；②在CR框架下将协同与其他可靠性指标（技术性能、知识契合度、社会技术治理等）统一起来；③提出“高效协同”概念，量化协同收益与交互成本的权衡；④给出三种实际案例验证该框架。

**🔧 技术方法**

主要技术包括：1）计算可靠性理论的概念化与 formalization；2）对人机交互的形式化定义（5‑tuple、交互协议、输出函数等）；3）协同度量（CTP、Δ_τ(D)、Δ^net_τ(D)）与成本模型；4）基于可靠性指标的最小报告清单设计。

**📊 数据集**

本文并未使用单一实验数据集，而是通过三个典型案例（皮肤科诊断、LLM辅助作业、法医语音识别）来说明框架的适用性；若需实验，则可在公开的医疗影像、文本生成或语音识别数据集上进行验证。

**📈 对比分析**

在理论上，作者对比了传统的相对准确性评价与CR框架下的可靠性证明，指出后者能兼顾准确性、稳健性、可解释性、成本等多维度需求；实验性比较未给出数值，而是通过案例阐释高效协同如何在不同情境下产生不同的实践意义。

**⚠️ 局限性**

局限性包括：①协同仍是二元且基于历史真实标注的后验度量，难以在决策时实时使用；②对成本的量化需依赖领域专家设定λ，缺乏通用标准；③CR本身的假设与可解释性问题（如内在属性可访问性、错误诊断难度）仍待进一步探讨；④实际部署时仍需结合具体任务的公平、鲁棒性等非准确性指标。

---

## 44. Strategies of cooperation and defection in five large language models

**arXiv ID:** 2601.09849 | [PDF](https://arxiv.org/pdf/2601.09849v1)

**作者:** Saptarshi Pal `[一作]` (Harvard University), Martin A Nowak `[通讯]` (Harvard University)

**通讯引用:** 121582 | [OpenAlex ID](https://openalex.org/A5081956768)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过直接向五大主流LLM（Claude‑Sonnet‑4、Gemini‑2.5‑Pro、GPT‑4o、GPT‑5、Llama‑3.3‑70B）提问，获取其在重复囚徒困境记忆 1/2 场景下的合作/背叛概率，随后将其映射为策略参数，评估其是否为 Nash 均衡、合作伙伴或对手，并系统探讨游戏终止概率、收益矩阵、回合数可知与否以及提示框架对LLM行为的影响；最后通过轮盘赛比较不同策略的收益。

**💡 创新点**

创新点在于：①不再仅观察单局结果，而是直接诱导LLM给出完整的记忆策略；②将推断出的策略与进化博弈理论中的概念（Nash、伙伴、对手）对齐；③系统评估多维度参数与框架对LLM策略的调节作用，并用双循环轮盘赛验证其鲁棒性。

**🔧 技术方法**

技术包括：LLM API 调用、基于记忆 1/2 的概率策略建模、博弈理论分析（Nash、伙伴/对手判定）、统计检验、轮盘赛模拟、以及不同提示框架的对比实验。

**📊 数据集**

数据集为约 40,000 次自定义实验调用，涵盖 15 种无框架实验、10 次已知/未知终点实验、记忆 2 场景等，每种实验 50 次样本；未使用公开数据集，全部为自行构造的实验问卷。

**📈 对比分析**

比较方法为：①对推断策略做 NASH/伙伴/对手判定；②计算其与 10^6 随机对手的收益区间；③在 15 个记忆 1 实验中进行全对全轮盘赛（含自我对战与非自我对战），统计排名。结果显示 Forgiver 类策略在多数实验中获得最高排名，整体表现良好；但在部分参数或框架下，LLM 仍出现非直觉或不一致的行为。

**⚠️ 局限性**

局限性包括：①未能在所有实验设置下保持一致性；②部分模型对终止概率或收益变动不作适应；③缺乏后向归纳表现；④受限于提示设计和样本大小；⑤仅评估记忆 1/2 策略，未扩展到更长记忆；⑥研究范围局限于囚徒困境，未验证到更复杂的社会困境。

---

## 45. High signal-to-noise ratio asymptotics of entropy-constrained Gaussian channel capacity

**arXiv ID:** 2601.09864 | [PDF](https://arxiv.org/pdf/2601.09864v1)

**作者:** Adway Girish `[一作]` (School of Computer and Communication Sciences), Emre Telatar `[通讯]` (School of Computer and Communication Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了带输入熵约束的功率受限高斯信道，在高信噪比下的容量极限，并给出了相应的最优输入分布。

**💡 创新点**

创新点在于证明在高信噪比极限下，离散高斯分布在满足功率和熵约束的所有离散分布中唯一实现最大最小原子间距，从而取得容量上界；并给出了容量误差指数与熵的显式表达。

**🔧 技术方法**

采用了高斯极限下的条件熵指数分析、最小距离优化、凸优化与几何切线法，以及Lambert W函数逼近等信息论与数值分析技术。

**📊 数据集**

该工作为纯理论研究，无使用任何实验数据集。

**📈 对比分析**

本文未与其他方法做实验比较，主要通过解析证明和数值模拟验证了离散高斯分布在高信噪比下的优势。

**⚠️ 局限性**

主要限制在于尚未证明有限信噪比下容量最优分布随信噪比趋近时弱收敛到离散高斯分布，且对分布的极限行为仍需进一步研究。

---

## 46. PCN-Rec: Agentic Proof-Carrying Negotiation for Reliable Governance-Constrained Recommendation

**arXiv ID:** 2601.09771 | [PDF](https://arxiv.org/pdf/2601.09771v1)

**作者:** Aradhya Dixit `[一作]` (Wake Technical Community), Shreem Dixit `[通讯]` (University of North Carolina)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 PCN-Rec 框架，通过 LLM 媒体与两个代理（User Advocate 与 Policy Agent）进行协商，生成可验证的结构化证书后由确定性验证器检查治理约束，并在失败时使用确定性修复机制返回合规列表。

**💡 创新点**

创新点在于将 LLM 仅作为建议者，采用 proof‑carrying 交互；区分约束不可满足与方法失效；提供可审计的 JSON 证明与确定性修复路径，确保在可行窗口内达到近乎完美的治理合规。

**🔧 技术方法**

使用技术包括：基于 MF/CF 的候选窗口生成、两代理竞争式目标（最大化相关性与满足治理约束）、LLM 媒体生成顶层列表与结构化证书、确定性约束验证器、受限贪心修复算法，以及配套的日志与审计追踪。

**📊 数据集**

实验数据集为 MovieLens‑100K，使用其 80% 训练集、10% 验证集、10% 测试集，并在候选窗口大小 W=80 上评估。

**📈 对比分析**

与单一 LLM baseline 对比，PCN-Rec 在可行用户（n=551）上的治理合规率从 0.000 提升至 0.985，NDCG@10 仅下降 0.021（从 0.424 降至 0.403），且差异在统计学上显著；相较于 deterministic greedy bound，PCN-Rec 维持了较高的效用。

**⚠️ 局限性**

局限性包括：只能在候选窗口内存在合规 slate 时才可满足约束；依赖准确的项目元数据（头/尾、类型标签）；仅能验证已编码的客观约束，无法覆盖主观或动态政策；修复过程可能导致与全局最优约束排名的效用损失。

---

## 47. Repository Intelligence Graph: Deterministic Architectural Map for LLM Code Assistants

**arXiv ID:** 2601.10112 | [PDF](https://arxiv.org/pdf/2601.10112v1)

**作者:** Tsvi Cherny-Shahar `[一作]` (Tel Aviv University), Amiram Yehudai `[通讯]` (Tel Aviv University)

**通讯引用:** 605 | [OpenAlex ID](https://openalex.org/A5109051398)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 Repository Intelligence Graph（RIG）和 SPADE 提取器，用于从构建和测试工件自动生成结构化的、基于证据的仓库架构图，并将该图以 JSON 形式注入大型语言模型（LLM）上下文，帮助 LLM 代码助手更快速、更准确地回答关于仓库构建、测试、依赖等结构性问题。

**💡 创新点**

创新点在于：① 构建了一个确定性、可验证的构建/测试层次图谱（RIG），用构建系统与测试工具生成的真实证据构成节点与边；② 通过 SPADE 自动从 CMake、CTest 等工件提取 RIG，并可手工扩展到其他构建系统；③ 将 RIG 序列化为 LLM 友好的 JSON，成为任何 LLM 代理可直接查询的“权威”结构信息；④ 通过对比实验验证 RIG 在提升准确率、减少执行时间和提高效率方面的显著效果。

**🔧 技术方法**

使用的主要技术包括：
- SPADE 提取器，利用 CMake File API、CTest JSON 以及必要时对 CMake 脚本的解析构建 RIG；
- RIG 的 Pydantic 模式定义，保证结构化、可验证的数据模型；
- JSON 视图序列化，保持平面、可读的结构供 LLM 直接使用；
- 评测框架，设计 30 题（易/中/难）问答集，并使用准确率、耗时、秒/正确答案效率等指标进行量化对比；
- 复杂度度量（基于组件数、语言数、包数、依赖深度等），用于分析 RIG 对不同难度仓库的影响。

**📊 数据集**

实验数据集包括八个仓库：
- 7 个人工合成、完整构建可用的项目（单语言 CMake、跨语言 CMake、MetaFFI、微服务 Go、Maven 多模块、Meson 固件、npm 多仓库、Cargo 编译器）
- 1 个真实世界 MetaFFI 代码库
每个仓库均生成对应的 RIG，并对 30 个结构性问题进行评分，总共 240 题。

**📈 对比分析**

比较方法：对每个代理（Claude Code、Cursor、Codex）在相同仓库上，分别在没有 RIG 与加入 RIG 的两种情境下执行相同的 30 题集；记录准确率、总耗时和秒/正确答案效率；对比两种情境的相对提升。实验结果显示：
- 平均准确率提升 12.2%，最高达 24.5%；
- 平均耗时减少 53.9%，最高达 74.1%；
- 平均效率提升 57.8%（秒/正确答案下降）。
多语言仓库与中高难度问题的提升效果尤为显著。

**⚠️ 局限性**

局限性：
- 目前 SPADE 仅支持 CMake（其余构建系统需手工构造 RIG），对动态脚本或高度自定义的构建系统支持不足；
- 需要为每种构建系统实现单独提取器，维护成本较高；
- RIG 只覆盖构建/测试层次，不涉及代码层面的 AST、控制流或数据流图，无法完全满足所有分析需求；
- 仍需 LLM 进行图遍历与推理，若推理逻辑不完善仍可能出现误差；
- 评测样本规模有限，仅覆盖八个仓库，且多为人工合成，真实世界多样性和极端复杂度尚未全面验证。

---

## 48. SagaScale: A Realistic, Scalable, and High-Quality Long-Context Benchmark Built from Full-Length Novels

**arXiv ID:** 2601.09723 | [PDF](https://arxiv.org/pdf/2601.09723v1)

**作者:** Guancheng Du `[一作]` (Tsinghua University), Jiaheng Gao `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于完整小说的长文本理解基准SagaScale，并通过自动化管道生成高质量的问答对。

**💡 创新点**

创新点在于：①使用外部资源（如维基百科）协助LLM生成更复杂的问题；②采用多阶段过滤确保正确性、真实性与非污染；③提供最大化的上下文长度（英文本>250K token，中文文本>320K token）且兼具可扩展性与高质量。

**🔧 技术方法**

主要技术包括：LLM驱动的生成与检索（多查询、多层检索）、多阶段过滤（正确性、真实性、污染检测）、三种长文本处理方法（直接长上下文、Naïve RAG、Agentic RAG）以及LLM评判器。

**📊 数据集**

数据集为103本小说（77英文、26中文）和1124条QA对，平均token数超过250K/320K，覆盖多种问题类型与长度区间。

**📈 对比分析**

对12种前沿LLM（包括GPT‑4o、Gemini‑2.5‑Pro等）在三种方法上进行评测；结果显示在可适配长度内直接给全上下文的Long Context方法往往优于RAG；Agentic RAG显著提升相较Naïve RAG；Gemini‑2.5‑Pro在长文本处理上表现突出，但整体准确率仍未达到完美。

**⚠️ 局限性**

限制：①数据量仅1,124条QA，难以直接用于训练；②任务仅限小说问答；③仅支持英语与中文；④过滤过程导致大量生成样本被剔除，规模受限。

---

## 49. Segmentação Comportamental, Do Not Track e o desenvolvimento jurídico europeu e holandês

**arXiv ID:** 2601.09711 | [PDF](https://arxiv.org/pdf/2601.09711v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 50. Time Aggregation Features for XGBoost Models

**arXiv ID:** 2601.10019 | [PDF](https://arxiv.org/pdf/2601.10019v1)

**作者:** Mykola Pinchuk `[一作]` `[通讯]` (Independent Researcher), Mykola Pinchuk (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在严格的时序拆分和无前瞻约束下，使用时间聚合特征提升 XGBoost 在 Avazu CTR 预测任务中的表现。

**💡 创新点**

提出对时间窗口设计（长度、形状）进行系统评估，并证明“尾随窗口 + 事件计数窗口”在无前瞻条件下能获得稳定提升，成为默认的实用方案。

**🔧 技术方法**

采用 XGBoost 分类器，使用对数滑动平均的点击率与日志计数作为特征；通过不同窗口长度与形状（尾随、gap、bucket、calendar、event50）构造时间聚合特征；对高基数特征使用目标编码。

**📊 数据集**

使用 Kaggle 的 Avazu click‑through‑rate 数据集（约 404 万条记录，10% deterministic 样本），按小时分区，进行两次滚动尾部折叠（Fold A 与 Fold B）。

**📈 对比分析**

在不泄漏的严格 OOT 测试中，尾随窗口 (长度 (1,6,24,48,168)) 的 ROC AUC 达到 0.748–0.752，加入 event50 进一步提升至 0.750–0.751；相对仅目标编码的 baseline，提升约 0.007–0.008；gap、bucket、calendar 设计反而降低或无显著改进。

**⚠️ 局限性**

实验仅覆盖一个数据集、有限的折叠和窗口组合，未做完整时间序列交叉验证，结果可能对其他任务或更大时间跨度的数据不具普适性。

---

## 51. Learning-Augmented Perfectly Secure Collaborative Matrix Multiplication

**arXiv ID:** 2601.09916 | [PDF](https://arxiv.org/pdf/2601.09916v1)

**作者:** Zixuan He `[一作]` (EURECOM), Photios A. Stavrou `[通讯]` (EURECOM)

**通讯引用:** 436 | [OpenAlex ID](https://openalex.org/A5111606657)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种完美安全的多方矩阵乘法协议（PSMM），并在此基础上引入学习增强版本（LA-PSMM），实现隐私保护下的高效矩阵乘法。

**💡 创新点**

创新点包括：1）利用稀疏多项式编码与零间隙对齐实现完美隐私和最优恢复阈值；2）首次将AlphaTensor等学习式低秩张量分解集成到安全多方计算，显著降低本地计算复杂度；3）在保证信息理论安全的同时，实现了可扩展的通信与计算效率。

**🔧 技术方法**

核心技术：Shamir共享、Beaver三元组、稀疏多项式共享、零间隙对齐、张量分解（AlphaTensor）以及离线训练的强化学习。

**📊 数据集**

使用合成的大素域（prime field）随机矩阵进行仿真，尺寸为 m = 1024，构造了多种 k 与 t 组合来评估性能。

**📈 对比分析**

通过与 BGW 风格作业拆分基线和标准 PSMM 的对比实验，结果显示：1）在相同隐私阈值下所需代理数显著降低，通信量更小；2）LA-PSMM 的本地计算复杂度比 PSMM 降低 20%–80%，取决于矩阵尺寸和学习分解的秩。

**⚠️ 局限性**

局限性：1）仍假设半诚实攻击模型，未考虑恶意攻击；2）学习分解需要离线训练，无法即时适应动态输入；3）在极大规模矩阵或高 t 阈值时，通信与解码开销仍可能成为瓶颈。

---

## 52. A Risk-Stratified Benchmark Dataset for Bad Randomness (SWC-120) Vulnerabilities in Ethereum Smart Contracts

**arXiv ID:** 2601.09836 | [PDF](https://arxiv.org/pdf/2601.09836v1)

**作者:** Hadis Rezaei `[一作]` (University of Salerno), Francesco Palmieri `[通讯]` (University of Salerno)

**通讯引用:** 7900 | [OpenAlex ID](https://openalex.org/A5039135286)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了规模最大的SWC‑120坏随机性漏洞基准数据集，涵盖1758个以太坊智能合约，覆盖多种风险等级；

**💡 创新点**

创新点在于提出了五阶段构建流程，包括基于58条正则表达式的模式匹配、功能级验证、风险分层、上下文鉴别和可视化分析，并首次实现了对漏洞函数的精确保护验证；

**🔧 技术方法**

采用关键词过滤、正则表达式检测、函数抽取与调用链追踪、访问控制与VRF/Commit‑Reveal识别等技术，实现了漏洞检测与风险评估；

**📊 数据集**

使用SmartBugs‑Wild公开合约库进行初筛，并结合SWC Registry、SmartBugs Curated、RNVulDet等数据集进行验证，最终得到1758个标注样本；

**📈 对比分析**

与Slither和Mythril等现有工具对比，实验发现两者在该数据集上召回率均为0%，而本文的模式匹配方案召回率达100%，证明了更复杂模式识别的重要性；

**⚠️ 局限性**

局限性包括仅分析具备源码的合约、仅覆盖58种模式、未处理跨合约调用以及对Bytecode的缺失，未来需扩展模式、覆盖链上合约与动态分析。

---

## 53. Eliminating Agentic Workflow for Introduction Generation with Parametric Stage Tokens

**arXiv ID:** 2601.09728 | [PDF](https://arxiv.org/pdf/2601.09728v1)

**作者:** Meicong Zhang `[一作]` (East China Normal University), Guoxiu He `[通讯]` (East China Normal University)

**通讯引用:** 157 | [OpenAlex ID](https://openalex.org/A5000341481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种通过阶段标记（STIG）消除外部代理工作流，实现一次推理即可生成完整科学论文引言的方法。

**💡 创新点**

创新点在于将多阶段写作逻辑参数化为模型内部的阶段标记，使得引言在单步生成中保持结构严谨、语义一致，并显著降低错误积累和计算成本。

**🔧 技术方法**

采用指令调优与微调结合的LLM训练，利用8个阶段标记（包含提纲与内容两步）在LLaMA‑Factory + ZeRO3框架下进行参数化学习。

**📊 数据集**

构建并使用了约3800篇ACL 2021‑2025主会议论文的注释数据集（含引言的提纲与正文），测试集为ACL 2025的1176篇论文，并在102篇CVPR论文上评估跨域泛化。

**📈 对比分析**

与Pure Prompt、ELABORATE Prompt、Outline Writing、AutoSurvey等基线以及GPT‑4o比较，STIG在结构合理性提升约17.4%、内容覆盖、语义相似度等指标上均优于基线，且单推理效率比AutoSurvey提升3.3倍。

**⚠️ 局限性**

局限性包括对计算机科学会议论文结构的依赖，跨学科或不同稿件类型的泛化可能受限；对高质量标注的需求较高；生成结果仍需人工核查与事实验证。

---

## 54. Antisocial behavior towards large language model users: experimental evidence

**arXiv ID:** 2601.09772 | [PDF](https://arxiv.org/pdf/2601.09772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 55. Bears, all bears, and some bears. Language Constraints on Language Models' Inductive Inferences

**arXiv ID:** 2601.09852 | [PDF](https://arxiv.org/pdf/2601.09852v1)

**作者:** Sriram Padmanabhan `[一作]` (University of Texas at Austin), Kanishka Misra `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究 Vision‑Language Models 在对“all / some / generic”三种命题进行归纳推理时的敏感性，并通过层级表示分析揭示其内部对诱导约束的编码；

**💡 创新点**

①在 VLM 上复现儿童对 generic 与量词的区分实验；②构造 All/Some 基准测试并公开；③发现 VLM 在隐藏层层面根据诱导约束组织命题，而非仅表面形式；④在视觉、语言两种模态均展示与人类相似的推理模式；

**🔧 技术方法**

使用多模态 VLM（Qwen3‑VL 4B/8B、Qwen2.5‑VL‑7B、LLaVA、Idefics 等），通过 Prompt 生成图像与句子；采用概率计量、线性混合效应模型、PCA 等统计方法对模型输出与内部表示进行分析；

**📊 数据集**

THINGS 图像库（1222 类别×5 张图）；自制 All/Some 视觉+语言与文本基准（共 840/900 条）；生成的 generic / 量词句子（10 个属性×4 表达变体，共 12,000 条）；以及 Gemini‑2.5 生成的图像对（140 对）等；

**📈 对比分析**

先进行类别识别、All/Some 预测准确率评估；随后在诱导推理任务中计算“yes”概率，并与人类数据对比；结果显示 Qwen3‑VL 4B/8B 在 All/Some 识别与归纳推理上表现最佳，归纳顺序符合 all>generic>some；其他模型表现低于人类，且在 All 询问时存在负向偏倚；

**⚠️ 局限性**

限于无因果解释（仅低维表示）；All/Some 基准样本量有限且受 Gemini 生成质量限制；仅涵盖英语，未涉及多语言；VLM 训练数据量远超人类经验，可能影响可比性；

---

## 56. Adaptive Orchestration: Scalable Self-Evolving Multi-Agent Systems

**arXiv ID:** 2601.09742 | [PDF](https://arxiv.org/pdf/2601.09742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 57. From SERPs to Agents: A Platform for Comparative Studies of Information Interaction

**arXiv ID:** 2601.09937 | [PDF](https://arxiv.org/pdf/2601.09937v1)

**作者:** Saber Zerhoudi `[一作]` (University of Passau), Michael Granitzer `[通讯]` (University of Passau)

**通讯引用:** 3569 | [OpenAlex ID](https://openalex.org/A5006866152)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出并实现了 UXLab——一个开放源代码、无代码的 web 平台，用于快速搭建、部署并管理针对传统检索、检索增强生成（RAG）和自主代理（Agentic）等多种信息交互系统的对比用户研究，并通过一个小规模实验验证其可行性。

**💡 创新点**

创新点在于：①将实验流程、后端服务、前端界面和实验管理解耦，形成四层架构；②提供可视化实验设计器和后端配置器，使研究者无需编写代码即可完成复杂实验的配置；③通过“Service Connector”插件化机制，支持任意检索或代理后端的无缝集成；④实现一键导出完整实验配置，显著提升实验可重复性和透明度。

**🔧 技术方法**

技术实现主要使用：FastAPI（后端逻辑与 API 服务）、PostgreSQL（实验数据存储）、React/HTML（实验前端和 Dashboard）、Python Connector 类库（与外部检索/代理接口交互），并集成 OpenAI API 进行 RAG 与 Agentic 模型调用。

**📊 数据集**

实验中使用的“数据集”主要为受试者自我生成的任务需求（如 3 天行程规划），并未采用公开大规模文本语料；后端则调用 OpenAI 的语言模型进行检索与生成。

**📈 对比分析**

在 8 名 Prolific 受试者的 within‑subject 对比实验中，Agentic 模式比 RAG 更快（平均 245 s vs 311 s，p<0.001）且跟进请求更少（2.1 vs 4.8，p<0.001），初始查询长度更长；问卷显示 Agentic 在满意度、信任度上更高，但用户感知控制感下降。

**⚠️ 局限性**

局限性包括：①样本量较小，难以推广到更大人群；②仅验证了两种后端（OpenAI RAG 与 Agentic），未覆盖多样化检索或代理实现；③实验任务较为简单，缺乏多模态或长时序交互的评估；④系统依赖外部 API（如 OpenAI），使用成本和可用性受限。

---

## 58. UniHash: Unifying Pointwise and Pairwise Hashing Paradigms for Seen and Unseen Category Retrieval

**arXiv ID:** 2601.09828 | [PDF](https://arxiv.org/pdf/2601.09828v1)

**作者:** Xiaoxu Ma `[一作]` (Georgia Institute of Technology), Zhenyu Weng `[通讯]` (South China University of Technology)

**通讯引用:** 458 | [OpenAlex ID](https://openalex.org/A5067922510)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了统一哈希（UniHash）双分支框架，结合了点对点（中心基）和对对学习范式，通过互相学习和分拆合并专家模块提升哈希码判别力与泛化能力。

**💡 创新点**

创新点在于：①将两种传统范式在同一模型中并行融合；②引入互相学习损失实现双向知识迁移；③设计Split-Merge Mixture of Hash Experts（SM‑MoH）模块，实现跨分支的专家路由与融合；④提供理论证明消除范式特有的误差门槛。

**🔧 技术方法**

技术手段包括：ResNet‑50特征提取器；中心损失、对对损失与互相学习损失；SM‑MoH专家网络与分支门控；RMSProp优化；超参数调优与t‑SNE可视化。

**📊 数据集**

实验数据集：CIFAR‑10、ImageNet、MSCOCO，分别在闭集检索和见/未见类别检索两种协议下评估。

**📈 对比分析**

与九种基准（点对点、对对、三元组等方法）对比，UniHash在16/32/64位下均取得最高mAP（如CIFAR‑10 0.9665/0.9657/0.9658），并在未见类别检索中明显优于现有方法（如未见@未见 84.7% vs 82.0%）。PR曲线显示更高的AUC‑PR，证明整体性能提升显著。

**⚠️ 局限性**

局限性：模型结构更复杂，训练成本和参数量相对传统单分支方法更高；对超参数（λ1, λ2, λ3）敏感；在极端零样本或极大类别数下的泛化表现尚未充分验证；依赖预定义中心，可能在高度异构数据集上受限。

---

## 59. Opportunities and Challenges of Natural Language Processing for Low-Resource Senegalese Languages in Social Science Research

**arXiv ID:** 2601.09716 | [PDF](https://arxiv.org/pdf/2601.09716v1)

**作者:** Derguene Mbaye `[一作]` (Polytechnic School), Jerome Chenal `[通讯]` (Federal Institute of Technology Lausanne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究对塞内加尔六种国家语言（沃洛夫、普拉、塞雷尔、迪奥拉、曼丁格和索尼克）的自然语言处理进展、数据资源、工具与挑战进行了系统综述，并构建了统一的 GitHub 资源库。

**💡 创新点**

创新点在于：①首次将塞内加尔六语的 NLP 生态做成一体化综述；②提出针对社会科学研究的 NLP 流程范式；③搭建了公开可持续的资源共享平台，促进本土社区协作。

**🔧 技术方法**

综述覆盖的技术包括词形分析、分词、命名实体识别、情感与仇恨语料检测、机器翻译、自动语音识别、文本转语音、对话与交互式系统等主流 NLP 方法与模型。

**📊 数据集**

利用的主要数据集包括 Opus、FloreS、CommonVoice、Masakhane 语料、AIDA、M3、Fleurs、Wolof‑French 并行语料、ASR 训练集、TTS 语音集，以及多语言基准集如 XLM‑R、M2M‑100、BERT 等。

**📈 对比分析**

对比方法采用 BLEU、WER、F1‑score、准确率等指标，文献显示塞内加尔语言与高资源语言之间存在显著性能差距；但通过跨语言迁移、多语种预训练和数据增强可显著提升某些任务的效果。

**⚠️ 局限性**

主要限制包括：数据稀缺且质量参差不齐、标注资源不足、脚本和方言差异大、计算与基础设施不足、研究社区规模有限、以及缺乏统一的伦理与治理框架。

---

## 60. Is MT Ready for the Next Crisis or Pandemic?

**arXiv ID:** 2601.10082 | [PDF](https://arxiv.org/pdf/2601.10082v1)

**作者:** Vipasha Bansal `[一作]` (University of Washington), William D. Lewis `[通讯]` (University of Washington)

**通讯引用:** 2548 | [OpenAlex ID](https://openalex.org/A5014205123)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了商业机器翻译与大型语言模型在低资源语言中的可用性，使用TICO‑19数据集在2023年和2025年的两次快照进行实验。

**💡 创新点**

首次系统性地为危机情境（如疫情）定义MT可用性阈值，并对比多家供应商及LLM在疫情相关文本上的表现，揭示质量下降与数据污染的风险。

**🔧 技术方法**

采用BLEU、BERTScore、COMET等自动评估指标，并通过Google、Microsoft、GPT‑4o、Gemini的API进行翻译。

**📊 数据集**

使用TICO‑19 35种语言的平行测试/开发集，涵盖医学、新闻、通用等多领域句子。

**📈 对比分析**

通过比较各系统在EX（英语→目标）和XE（目标→英语）方向的BLEU分数，设30为可用阈值，发现Google整体最高，Microsoft在某些语言出现显著下降，LLM在高资源语言可用但低资源语言多为不可用；部分语言出现极高BLEU提示可能的数据污染。

**⚠️ 局限性**

缺乏人工评估、仅以英语为中介语言、部分低资源语言被排除、存在训练数据污染可能、对低资源语言评估受限。

---

## 61. SyncTwin: Fast Digital Twin Construction and Synchronization for Safe Robotic Grasping

**arXiv ID:** 2601.09920 | [PDF](https://arxiv.org/pdf/2601.09920v1)

**作者:** Ruopeng Huang `[一作]` (University of Southern California), Jiachen Li `[通讯]` (University of California)

**通讯引用:** 1918 | [OpenAlex ID](https://openalex.org/A5100357067)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出SyncTwin框架，利用RGB-only 3D重建和实时同步实现动态、部分可见环境下的安全抓取。

**💡 创新点**

创新点包括：① VGGT快速重建与分割扩展/去噪技术；② 对象记忆库完成几何补全；③ 彩色ICP与实时点云分割实现闭环 sim-to-real 同步；④ 结合GraspGen与cuRobo MPC实现安全轨迹规划。

**🔧 技术方法**

使用技术有：VGGT、SAM2、彩色ICP、GraspGen、cuRobo MPC、Isaac Sim、3D Gaussian Splatting、Nerfstudio、Photogrammetry、NVBlox等。

**📊 数据集**

实验数据来自Franka Emika Panda与Intel RealSense D455在真实实验室采集的RGB/D图像，测试对象包括瓶子、罐子、杯子、盒子等；与Photogrammetry、Nerfstudio、3DGS、NVBlox等基线对比。

**📈 对比分析**

通过重建时间、图像依赖度、避障成功率与抓取成功率比较，SyncTwin在5–10张图像下仅需1–2分钟完成重建，避障成功率从50%提升至85–93%，抓取成功率提升约15–22%，显著优于基线。

**⚠️ 局限性**

限制：VGGT外参误差仍导致Stage I偶尔失败；缺乏在线增量学习，无法即时加入新对象；单GPU实时速度受限，难以处理极快动态场景。

---

## 62. Formal Safety Guarantees for Autonomous Vehicles using Barrier Certificates

**arXiv ID:** 2601.09740 | [PDF](https://arxiv.org/pdf/2601.09740v1)

**作者:** Oumaima Barhoumi `[一作]` (Concordia University), Sofiène Tahar `[通讯]` (Concordia University)

**通讯引用:** 3974 | [OpenAlex ID](https://openalex.org/A5007159598)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于时间-碰撞（TTC）的障碍证书（BC）与SMT形式化验证相结合的安全框架，用于车道跟随场景的CAV安全控制。

**💡 创新点**

创新点在于将可解释的TTC指标与可证明的BC融合，并通过SMT求解器正式验证BC的安全性，而非仅依赖经验合成；同时在真实数据上实现动态速度调整。

**🔧 技术方法**

使用障碍证书、时间-碰撞度量、SMT形式化验证（Z3）、自适应速度调整控制策略。

**📊 数据集**

采用德国高速公路自然驾驶数据集HighD进行验证。

**📈 对比分析**

通过对比速度调整前后TTC<3s的冲突次数，实验证明BC调控能将冲突数降低至40%以内，并在部分车道完全消除冲突。

**⚠️ 局限性**

局限性包括仅验证车道跟随单车对交互，缺乏多车多道交互的评估；依赖固定的TTC阈值；形式化验证在更大规模系统中求解复杂度可能上升。

---

## 63. Multi-Agent Cooperative Learning for Robust Vision-Language Alignment under OOD Concepts

**arXiv ID:** 2601.09746 | [PDF](https://arxiv.org/pdf/2601.09746v1)

**作者:** Philip Xu `[一作]` (De Montfort University), Eerke Boiten `[通讯]` (De Montfort University)

**通讯引用:** 1576 | [OpenAlex ID](https://openalex.org/A5059470549)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出多代理协作学习框架 MACL，利用图像、文本、名称和协调四个代理协同工作，解决视觉语言模型在未见概念（OOD）下跨模态对齐崩溃问题。

**💡 创新点**

创新点包括：①基于难度的自适应多策略图像处理；②上下文交换增强的少样本学习；③名称代理学习 OOD 词向量；④协调代理动态温度与损失平衡；⑤结构化消息传递实现四代理协同。

**🔧 技术方法**

技术手段：多代理消息传递、图像代理的多级编码与难度评估、文本代理的上下文融合模块、名称代理的词向量学习与上下文交换、协调代理的动态温度调节与多损失平衡、对比学习与分类损失混合。

**📊 数据集**

使用的数据集：VISTA‑Beyond（400M 训练集）、多域评测集（Insects Spider、Landmark、Flowers、DTD 等）以及标准零样本评测集（Animals、Architecture 等）。

**📈 对比分析**

与 CoOp、CoCoOp、CLIP‑Adapter、FSNL、OpenCLIP、TransCLIP 等基线在 1–16 shot 的 few‑shot 以及 0 shot 的 zero‑shot 场景进行对比。MACL 在 OOD 与 SC 域平均提升 1–5%（few‑shot）并在零样本中平均提升约 1–2%，在多数领域均优于所有基线。

**⚠️ 局限性**

局限性：①仍依赖已有词表，完全无语义描述的新概念仍难以学习；②模型结构复杂，四代理通信与参数量较大，部署成本高；③缺乏对跨模态对齐机制的可解释性与分析；④在极大规模 OOD 语义空间下的鲁棒性尚待验证。

---

## 64. Multi-Constrained Evolutionary Molecular Design Framework: An Interpretable Drug Design Method Combining Rule-Based Evolution and Molecular Crossover

**arXiv ID:** 2601.10110 | [PDF](https://arxiv.org/pdf/2601.10110v1)

**作者:** Shanxian Lin `[一作]` (Tokushima University), Haichuan Yang `[通讯]` (Tokushima University)

**通讯引用:** 843 | [OpenAlex ID](https://openalex.org/A5069376593)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种多约束进化分子设计框架 MCEMOL，融合规则演化和分子交叉，实现从少量起始分子到高质量药物化合物的生成。

**💡 创新点**

双层演化（规则层与分子层）并行优化，兼顾可解释性与多属性合规；引入多维化学约束系统和自适应交叉策略，显著提升分子合法性、结构多样性和药物相容性。

**🔧 技术方法**

使用基于消息传递的 ATLAS‑DMPNN 进行属性预测，结合自适应遗传算法、规则演化引擎和七种分子交叉策略；实现无 GPU 的高效运行。

**📊 数据集**

在 ZINC250K 公开数据库中进行实验，起始仅 20 份种子分子，迭代 100 代后生成 829 种独特分子。

**📈 对比分析**

与 JT‑VAE、GCPN、GraphVAE、ORGAN 等生成模型对比，MCEMOL 在合法性 100%、多样性、药物相容率（Lipinski 100%、Ghose 98%、Veber 99.6%）和分子质量（LogP、QED、TPSA 等）方面均表现出色，且保持了高解释性。

**⚠️ 局限性**

受限于预定义的规则库，可能限制对非常规化学空间的探索；对柔性 3D 结构的处理不足；需要针对不同目标进行超参数调优。

---

## 65. Take Out Your Calculators: Estimating the Real Difficulty of Question Items with LLM Student Simulations

**arXiv ID:** 2601.09953 | [PDF](https://arxiv.org/pdf/2601.09953v1)

**作者:** Christabel Acquaye `[一作]` (University of Maryland), Rachel Rudinger `[通讯]` (University of Maryland)

**通讯引用:** 1054 | [OpenAlex ID](https://openalex.org/A5082447472)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究使用开源大型语言模型（LLM）通过角色扮演模拟学生在四年级、八年级和十二年级数学多选题的答题行为，并利用拟合的项目反应理论（IRT）模型估计题目难度，然后与美国全国教育进步评估（NAEP）提供的真实学生成绩进行对比。

**💡 创新点**

创新点包括：①用LLM模拟“学生”而非直接文本预测难度；②通过姓名、多样性身份提示提升模拟真实性；③发现较弱数学能力的LLM在模拟困难学生表现上优于强大模型；④采用多模型集成与不同班级规模实验，系统评估其对难度预测的影响。

**🔧 技术方法**

主要技术手段为：LLM角色扮演提示（指定年级、能力水平、身份标识）；生成多名学生回答并聚合成“班级”成绩；基于二分类回答训练IRT模型提取学生能力与题目难度；对不同模型（Gemma、Llama、Qwen等）进行加权或平均集成；对姓名/身份、班级规模等因素做实验对照。

**📊 数据集**

使用数据集为 NAEP 官方发布的 631 道数学多项选择题（涵盖 4、8、12 年级），每题均附有真实学生正确率、难度标签与多学科分布信息。

**📈 对比分析**

比较方法：将 LLM 角色扮演预测结果与直接文本难度估计（Word2Vec、BERT）以及直接提示预测（DPCE）进行对比；采用 Pearson、Spearman 相关系数和 AUC 评估难度预测精度。实验表明：相关系数最高可达 0.82（十二年级），AUC 在 0.78–0.90 之间；相较于基线，角色扮演+IRT 方法显著提升，且姓名多样性、班级规模优化可进一步提高性能。

**⚠️ 局限性**

局限性：①样本规模有限，仅包含 631 道题；②仅覆盖 4、8、12 年级，无法推广至其他学段；③姓名/身份代理可能触发刻板印象，无法排除模型偏见影响；④未能有效模拟学生误选（错误答案分布）和答案解释的真实性；⑤使用公开 LLMS，缺乏对大规模专有模型的验证；⑥计算成本随班级规模上升，存在资源与精度权衡；⑦可能存在训练数据泄漏风险，尽管对照实验已表明影响不大。

---

## 66. Democracy and Distrust in an Era of Artificial Intelligence

**arXiv ID:** 2601.09757 | [PDF](https://arxiv.org/pdf/2601.09757v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 67. Beyond Strict Rules: Assessing the Effectiveness of Large Language Models for Code Smell Detection

**arXiv ID:** 2601.09873 | [PDF](https://arxiv.org/pdf/2601.09873v1)

**作者:** Saymon Souza `[一作]`, Lionel Briand `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型（LLM）在Java项目中检测代码异味的有效性，提出并评估了LLM、传统静态分析工具及其投票组合的检测策略。

**💡 创新点**

首次构建了包含30个高星级Java项目、268个代码异味候选项的人工标注基准数据集，并将LLM的链式思考输出与工具结果统一评估，探讨LLM与工具的互补性。

**🔧 技术方法**

采用四种LLM（DeepSeek‑R1、GPT‑5 mini、Llama‑3.3、Qwen2.5‑Coder）与四个静态分析工具（JDeodorant、JSpIRIT、Organic、PMD），并通过投票机制实现组合检测。

**📊 数据集**

使用30个开源Java项目（共3,508,637 LOC）以及基于这些项目的268个代码异味候选项的人工验证结果。

**📈 对比分析**

对比了LLM、工具和组合策略在9种代码异味上的召回率、精准率与F1得分；LLM在结构化异味（如Large Class、Long Method）表现优异，工具在主观异味（如Feature Envy、Refused Bequest）更稳健，组合策略在多数异味上提升召回率且F1最高。

**⚠️ 局限性**

LLM对主观或上下文依赖异味识别仍受限，组合策略可能产生更多误报，且实验受限于Java语言与选定模型与工具，无法直接推广到其他语言或更大规模项目。

---

## 68. Cooking Up Politeness in Human-AI Information Seeking Dialogue

**arXiv ID:** 2601.09898 | [PDF](https://arxiv.org/pdf/2601.09898v1)

**作者:** David Elsweiler `[一作]` (University of Regensburg), Anna Ziegner `[通讯]` (University of Innsbruck)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究用户礼貌行为对生成式AI在烹饪信息寻求对话中的回复长度、信息量与能耗的影响，结合人类对话注释与大规模LLM‑LLM模拟。

**💡 创新点**

首次将礼貌层级划分为超礼貌、礼貌与互动、互动寻求、超高效及粗鲁五类，并通过模拟量化其对信息密度与能效的系统性差异，揭示礼貌可能成为隐性公平与可持续性成本。

**🔧 技术方法**

采用k‑means聚类、负二项混合模型、对数变换、nugget计数（基于LLM的事实单元检测）以及能耗估算，并通过对比混合模型检验礼貌与模型交互效应。

**📊 数据集**

使用Frummet等人Wizard‑of‑Oz烹饪对话数据（30条）进行礼貌标注；构建六种菜谱的任务信息需求，生成18,000条模拟对话（5种礼貌配置×3模型×200次）。

**📈 对比分析**

通过混合模型检验Cluster × AgentModel 交互，发现礼貌程度显著影响回复长度（最高可比基准增长90%）、信息量（最高提升38%）和能效（不同模型差异显著，Qwen最短、Llama信息密度最高）。

**⚠️ 局限性**

局限包括：样本量小且聚类不均，研究仅限烹饪场景，nugget与实际任务成功或用户满意度关联不确定，仅测试小型英文LLM，未覆盖多语言与更大模型。

---

## 69. Correspondences in computational and dynamical complexity II: forcing complex reductions

**arXiv ID:** 2601.09973 | [PDF](https://arxiv.org/pdf/2601.09973v1)

**作者:** Samuel Everett `[一作]` (University of Chicago), Samuel Everett `[通讯]` (University of Chicago)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5031640548)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文研究基于一维动力系统的有限时间可达性问题（代数趋向问题）的计算复杂性，并提出一种“自然归约”框架来描述不同动力系统间归约的结构。

**💡 创新点**

核心创新是将归约的形式与动力系统的共轭关系紧密关联，证明若两个系统在动力学性质上相距较远，则它们的代数趋向问题之间不可能存在低阶自然归约，从而间接给出了算法复杂度下界。

**🔧 技术方法**

主要技术包括：Blum–Shub–Smale（BSS）计算模型、拓扑动力学中的共轭与分岔概念、归约模板（四阶自然归约）、固定点定理与不连续性分析等。

**📊 数据集**

本工作为理论研究，未使用实验数据或公开数据集；所有结果均为数学证明。

**📈 对比分析**

由于采用的是严格的理论分析，未进行实验比较；通过证明可见，一些正拓扑熵系统的代数趋向问题无法由多项式时间算法或仅包含加乘门的算术电路族解决，复杂度至少呈指数或多项式深度。

**⚠️ 局限性**

局限性：目前仅对单维区间动力系统给出完整的归约不可行性，且对四阶自然搜索归约的更细粒度限制仍待完善；未来需要扩展到高维或非平滑动力系统，并探索更强的下界证明。

---

## 70. From Dynamic to Lexical: A Comparative Exploration of Scoping Rules in SAS and R

**arXiv ID:** 2601.09808 | [PDF](https://arxiv.org/pdf/2601.09808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 71. Chinese Labor Law Large Language Model Benchmark

**arXiv ID:** 2601.09972 | [PDF](https://arxiv.org/pdf/2601.09972v1)

**作者:** Zixun Lan `[一作]` (Xi’an Jiaotong–Liverpool University), Fei Ma `[通讯]` (Xi’an Jiaotong–Liverpool University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一个针对中国劳动法领域的专门大语言模型 LaborLawLLM，并构建了涵盖 12 种任务与 12 类案例的评测基准 LaborLawBench，用以系统评估模型在法律文本检索、知识问答、案例分类、赔偿计算、实体识别、案例分析等多维度表现。

**💡 创新点**

创新点在于：①针对单一法律子域（劳动法）进行全流程定制化模型训练与评测；②设计了多样化任务结构（从记忆检索到多步骤推理）与统一的 Instruction‑Question‑Answer 交互格式；③将主流的 LLM（Qwen2.5‑7B）通过 LoRA 微调与大量法律专用数据进行领域适配，显著提升了专业知识覆盖与推理精度；④首次公开中文劳动法专项基准，并提供多种客观与主观评测指标（ROUGE‑L、Accuracy、F1、Soft‑F1、GPT‑o1 评判）。

**🔧 技术方法**

技术主要包括：1）大规模预训练模型 Qwen2.5‑7B 的 LoRA 参数高效微调；2）多任务监督学习，统一使用 Instruction‑Question‑Answer 结构；3）多种评测指标实现（ROUGE‑L、Accuracy、F1、Soft‑F1、GPT‑o1），以及格式化输出的解析与校验；4）对模型输出进行格式化约束与后处理，确保法律文本的准确性与可评估性。

**📊 数据集**

数据集：①LaborLawBench，包含 12 项任务（T1–T12）与 12 类案例（C1–C12），共计 5,538 条样本；②训练集 51,236 条 SFT 例子，覆盖法规引用、考试题型、案例类型预测、补偿推算、命名实体提取、以及案例分析等；③采用真实案例文本、国家司法考试题目、劳动法条文等来源。

**📈 对比分析**

比较方法：与 17 个通用 LLM、8 个法律领域 LLM 以及原始 Qwen2.5‑7B 进行横向对比。评测采用各任务专属指标，最终给出综合分数。结果显示：LaborLawLLM 在所有 12 项任务中的平均得分为 0.68，明显高于最强通用模型 0.61 以及传统法律 LLM 0.29；在单选知识问答、复选知识问答、案例类型预测等结构化任务中表现近乎完美（1.00/0.99/0.98），但在基于 GPT‑o1 的长文本推理任务（如主张挖掘、争议焦点提取）仍有提升空间。

**⚠️ 局限性**

局限性：①对开放式生成任务的评测依赖 GPT‑o1，可能受到评分主观性与模型自身偏差影响；②在情境生成与条文匹配（T10）上因格式化差异导致 ROUGE‑L 低分，表明模型在长文本与简洁条文之间的匹配仍需改进；③数据集主要来自江苏省案例，缺乏跨地区、跨时段的多样性，可能导致泛化性不足；④模型对极端或罕见情形的鲁棒性尚未充分验证。

---

## 72. NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration

**arXiv ID:** 2601.09823 | [PDF](https://arxiv.org/pdf/2601.09823v1)

**作者:** Subhajit Sanyal `[一作]` (Samsung Research India), Amit Satish Unde `[通讯]` (Samsung Research India)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 NanoSD，一种针对边缘设备的高效扩散模型，用于多种图像恢复任务；

**💡 创新点**

创新点在于将 SD 1.5 的 U‑Net 通过硬件感知的阶段划分、形状保持块变体以及多目标贝叶斯优化构造 Pareto 前沿，并在此基础上对 VAE 进行蒸馏，形成兼顾生成先验与实时推理的轻量级基座；

**🔧 技术方法**

核心技术包括硬件感知的块级搜索空间构建、特征级生成蒸馏、基于 taFID 的多目标优化、VAE 端到端蒸馏与微调，以及在多个低级视觉框架（如 OSEDiff、S3Diff、OSDFace、Diff‑Plugin、DiffBIR、Marigold）中的集成；

**📊 数据集**

实验数据集涵盖超分辨率（DIV‑2K、RealSR、DRealSR）、人脸恢复（FFHQ、Wider‑Test）、去模糊（GoPro）、去雾（Reside）、去雨（merged train）、去雪（Snow100K）、单目深度（Hypersim、Virtual KITTI、NYU_v2、KITTI）等；

**📈 对比分析**

在上述任务中，NanoSD 在 PSNR、SSIM、LPIPS、FID、NIQE、MUSIQ 等指标上与或优于现有轻量级扩散模型，同时在 Qualcomm NPU 上实现 20 ms 以内的实时推理（NanoSD‑Prime 27 ms），参数量仅 130–315 M，显著低于 TinySD、Segmind 等基线；

**⚠️ 局限性**

局限性主要在于块级蒸馏可能导致跨块协同失真，对极高分辨率输入的推理仍受限于 NPU 计算瓶颈，并且在某些任务中对极端噪声或遮挡的鲁棒性尚未充分验证。

---

## 73. A Governance Model for IoT Data in Global Manufacturing

**arXiv ID:** 2601.09744 | [PDF](https://arxiv.org/pdf/2601.09744v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 74. Outrunning Big KATs: Efficient Decision Procedures for Variants of GKAT

**arXiv ID:** 2601.09986 | [PDF](https://arxiv.org/pdf/2601.09986v1)

**作者:** Cheng Zhang `[一作]` (Worcester Polytechnic Institute), Marco Gaboardi `[通讯]` (Boston University)

**通讯引用:** 2514 | [OpenAlex ID](https://openalex.org/A5021948795)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套基于 SAT 求解器的增量符号化决策程序，用于高效判定 GKAT 及其扩展形式的跟踪等价性，并在 Rust 中实现。

**💡 创新点**

创新点包括：① 将归一化步骤改为惰性 on‑the‑fly 触发，允许在发现反例时立即终止；② 采用布尔公式而非 BDD 的符号化自动机，避免指数级转移爆炸；③ 将求导（derivative）技术与 on‑the‑fly 生成结合，进一步缩短状态空间。

**🔧 技术方法**

核心技术包括：符号化自动机、基于 SAT 的布尔公式判定、求导法产生自动机、on‑the‑fly 归一化与等价检测，以及对 GOTO、break 等非本地控制流的符号化处理。

**📊 数据集**

实验使用随机生成的 GKAT 程序对以及 GNU Coreutils 9.5 中提取的真实控制流作为数据集。

**📈 对比分析**

与 SymKAT、原始实现以及基于 BDD 的实现对比，本文方法在运行时和内存占用上实现了数十到数百倍的加速；使用 MiniSAT 或 CUDD 后，性能更为稳定，且能成功验证 237 个 Coreutils 函数。

**⚠️ 局限性**

主要限制是：对包含大量死状态的自动机，惰性归一化仍可能产生额外开销；目前仅支持有限轨迹等价，未覆盖无穷轨迹；对某些扩展（如带权重、概率或更复杂的控制流指令）尚未实现符号化优化。

---

## 75. Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal

**arXiv ID:** 2601.09886 | [PDF](https://arxiv.org/pdf/2601.09886v1)

**作者:** Sathvik Nair `[一作]` (University of Maryland), Byung-Doh Oh `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文比较了人类预测（cloze 任务）与语言模型（GPT‑2）产生的词预测概率（surprisal）对单词级阅读时间的解释力，进一步通过三种操作检验了 GPT‑2 之所以优于 cloze 的三种潜在原因（分辨率、语义细粒度和低频词的处理）。

**💡 创新点**

创新点在于：①系统评估了 cloze 与 LM 预测在多个阅读时间数据集上的相对贡献；②设计并实施了三种针对 GPT‑2 预测的操作，以检验其优势来源；③提出并检验了 similarity‑adjusted surprisal 作为结合 cloze 与 LM 的新方法，虽然最终效果不佳，但为后续研究提供思路。

**🔧 技术方法**

主要技术包括：使用 GPT‑2 生成词概率并计算负对数概率（surprisal）；对 cloze 概率进行加一平滑、幂变换和 log 变换；线性混合效应回归（LME）与10折交叉验证评估模型拟合；对 GPT‑2 进行采样、聚类（k‑means）和频率阈值限制来构造三种假设对应的概率。

**📊 数据集**

使用了四个公开阅读时间数据集：BK21 SPR、Provo eye‑tracking、UCL SPR 与 UCL eye‑tracking；每个数据集均配有相应的 cloze 规范化结果，提供了多种词长、位置、频率等基础预测变量。

**📈 对比分析**

通过在 LME 模型中加入 cloze 概率、GPT‑2 概率或两者共同加入，计算对数似然提升并做配对置换检验。结果显示：在 4/6 量化指标中，GPT‑2 概率显著优于 cloze 概率；当对 GPT‑2 进行分辨率、语义聚类或频率限制的操作后，模型拟合显著下降，支持三种假设。

**⚠️ 局限性**

局限性包括：仅针对英语文本与英语母语者数据；缺乏跨语言验证；cloze 任务的无时限生产方式可能无法完全反映即时预测；对 token‑级概率的乘积处理可能未完全准确。

---

## 76. Interprofessional and Agile Development of Mobirobot: A Socially Assistive Robot for Pediatric Therapy Across Clinical and Therapeutic Settings

**arXiv ID:** 2601.09838 | [PDF](https://arxiv.org/pdf/2601.09838v1)

**作者:** Leonie Dyck `[一作]`, Anna-Lisa Vollmer `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究通过多学科、敏捷迭代的协同设计，开发了社交辅助机器人Mobirobot，用于儿童及青少年术后及精神疾病康复中的移动与运动训练，并在三种临床场景（住院精神运动、住院腹部手术物理治疗、门诊下肢骨折康复）中实现早期部署与可行性评估。

**💡 创新点**

创新点包括：①将协同设计与敏捷开发相结合，实现真正的临床嵌入；②构建可无代码配置的GUI与姿态识别、语音交互等多模态交互模块；③通过“运动方案模板”(regimen)实现运动计划的个性化与可重用；④在机器人上嵌入交互式按钮与情绪提示，提升儿童动机。

**🔧 技术方法**

技术实现主要采用NAO6仿人机器人平台；软件架构基于REST‑API、WebSocket；姿态识别使用OpenPose；语音识别利用Whisper并调用Phi3 LLM；交互按钮通过蓝牙连接；GUI采用Python/Qt或Web前端；硬件包括可移动机器人底座、心率监测手表、外置摄像头等。

**📊 数据集**

使用的数据集主要为本研究现场采集：①成人志愿者的姿态数据用于训练姿态分类模型；②患者运动过程中的心率数据（已停止使用）；③临床现场的问卷与访谈数据用于可行性评估。未使用公开的大规模医疗或运动数据集。

**📈 对比分析**

本论文未给出定量性能指标，而是通过可行性研究中的定性反馈（满意度问卷、观察记录、访谈）评估可用性、可接受性与潜在治疗收益；目前仍在进行数据收集与后续量化分析。

**⚠️ 局限性**

局限性主要包括：①机器人硬件受限（平衡、关节速度、姿态识别精度不足）；②缺乏细粒度情绪或痛感监测功能；③外置摄像头受患者隐私与环境限制，影响姿态识别；④远程维护依赖稳定网络，存在连通性问题；⑤患者招募困难，尤其是精神科场景导致样本量不足。

---

## 77. On the Leaky Private Information Retrieval with Side Information

**arXiv ID:** 2601.09960 | [PDF](https://arxiv.org/pdf/2601.09960v1)

**作者:** Yingying Huangfu `[一作]` (Huawei), Tian Bai `[通讯]` (University of Bergen)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了在侧信息下允许泄露ε的信息的Leaky Private Information Retrieval（L-PIR-SI）框架，并给出可实现的下载成本上界；

**💡 创新点**

首次统一构建了满足ε-泄露W-隐私与(W,S)-隐私的概率性查询方案，揭示了泄露、侧信息与通信效率之间的权衡；

**🔧 技术方法**

采用概率性映射、随机子包划分与线性组合技术，结合差分隐私度量来控制信息泄露；

**📊 数据集**

无；该工作基于理论模型，不涉及具体数据集；

**📈 对比分析**

与传统完美隐私PIR‑SI、Leaky‑PIR等结果对比，证明在ε→0或M=0时可恢复已知容量；在固定下载成本和服务器数下，泄露指数可达到(log g)或(log K)级；

**⚠️ 局限性**

尚未给出完全可达性证明，极限下的下界仍开放；对于一般M>1下的(W,S)-隐私仍缺乏完整方案；

---

## 78. Stable and Explainable Personality Trait Evaluation in Large Language Models with Internal Activations

**arXiv ID:** 2601.09833 | [PDF](https://arxiv.org/pdf/2601.09833v1)

**作者:** Xiaoxu Ma `[一作]` (Georgia Institute of Technology), Zhenyu Weng `[通讯]` (South China University of Technology)

**通讯引用:** 458 | [OpenAlex ID](https://openalex.org/A5067922510)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型中对人格特质进行评估，并提升评估结果的稳定性与可解释性

**💡 创新点**

提出内部激活向量中性插值（PVNI）方法，通过对比激活向量计算人格方向并在该方向上插值得到中性分数

**🔧 技术方法**

利用模型内部激活提取、向量投影与插值以及线性理论推导人格方向的近似线性性质

**📊 数据集**

使用 Qwen‑2.5‑7B、Llama‑3‑8B、Mistral‑7B‑v0.1 三个开源 LLM，以及 IPIP‑BFFM‑50 与 IPIP‑NEO‑120 两份公开问卷

**📈 对比分析**

与自评、开放式提问以及问卷/角色扮演等现有方法对比，PVNI 在所有模型与所有评估协议下都实现了最低方差、最高稳定性

**⚠️ 局限性**

仍依赖外部评判器；仅针对 Big Five 维度，缺乏跨语言、跨领域及更细粒度人格维度的验证

---

## 79. An Exploratory Study to Repurpose LLMs to a Unified Architecture for Time Series Classification

**arXiv ID:** 2601.09971 | [PDF](https://arxiv.org/pdf/2601.09971v1)

**作者:** Hansen He `[一作]` (Canyon Crest Academy), Shuheng Li `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了将大型语言模型与多种时间序列编码器相结合的混合架构，用于时间序列分类任务。

**💡 创新点**

创新点在于发现仅 Inception 编码器在与冻结的 Llama-3.1-8B 结合时能够持续提升性能，证明多尺度卷积在跨域时间序列学习中的关键作用。

**🔧 技术方法**

采用多种编码器（MLP、CNN、ResNet、Transformer、Inception）以及 Llama-3.1-8B 作为冻结后端，并进行端到端训练。

**📊 数据集**

使用 2015 年 UCR 时间序列数据集进行评估。

**📈 对比分析**

对比三类模型：仅 LLM、仅编码器、编码器+LLM，Inception+LLM 的平均准确率从 71.15% 提升到 74.21%，显著优于其它架构。

**⚠️ 局限性**

局限性包括实验仅覆盖单一数据集集合，未深入探讨不同任务的鲁棒性与 LLM 参数调优的影响。

---

## 80. UEOF: A Benchmark Dataset for Underwater Event-Based Optical Flow

**arXiv ID:** 2601.10054 | [PDF](https://arxiv.org/pdf/2601.10054v1)

**作者:** Nick Truong `[一作]`, William J. Beksi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了基于Blender的仿真流水线，将RGB帧转换为事件流，并在该事件流上实现并评估事件光流估计方法。

**💡 创新点**

创新之处在于整合仿真、事件转换与光流评估的完整工作流，提供可复现的事件光流基准，并利用Blender生成真实的光流与速度标签，弥补现有事件数据集的缺失。

**🔧 技术方法**

采用Blender 3D仿真生成场景，使用v2e事件模拟器将RGB图像转为事件流，应用事件光流算法（如本文提出或现有方法），并使用AEE、ANPE等指标进行评估。

**📊 数据集**

数据集为自研的Blender生成数据，包含RGB帧（如30375.png）、事件流（visualized_events.png）、GT光流（gt_flow375.png）及GT速度（i. GT Velocities i.）。

**📈 对比分析**

通过与GT光流对比计算AEE与ANPE，实验结果显示所提出的事件光流方法在该仿真数据集上取得了较低的误差，性能优于传统基线方法。

**⚠️ 局限性**

主要局限在于仅在仿真环境下验证，未考虑真实相机噪声与硬件特性；事件转换模型与实际相机可能存在偏差，且在极端动态场景下鲁棒性有待进一步提升。

---

## 81. EmplifAI: a Fine-grained Dataset for Japanese Empathetic Medical Dialogues in 28 Emotion Labels

**arXiv ID:** 2601.10033 | [PDF](https://arxiv.org/pdf/2601.10033v1)

**作者:** Wan Jou She `[一作]` (Kyoto Institute of Technology), Eiji Aramaki `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 2596 | [OpenAlex ID](https://openalex.org/A5041089475)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并构建了EmplifAI——一个面向慢性疾病患者的日语情感对话数据集，并通过大模型微调验证其在情感对齐与同理心生成方面的有效性。

**💡 创新点**

创新点在于：①将28类细粒度GoEmotion情感标签翻译并验证后用于医疗情景；②结合情境+两轮对话形式，覆盖多阶段情绪变化；③引入LLM‑as‑a‑Judge与人类评估的双重评估框架，探讨评判一致性与潜在偏差。

**🔧 技术方法**

使用技术包括：CrowdWorks众包+专业医护评审；BERTScore/FastText进行情感预测；对GPT‑o3‑pro、DeepSeek、LLM‑jp‑3.1‑13b‑instruct4、Llama‑3‑Swallow‑8b与MedLlama3‑JP等模型进行零射击与监督微调；Gemini‑2.5‑Flash 作为评判模型；Pearson、MAD 等统计方法比较评判一致性。

**📊 数据集**

所使用的数据集为：EmplifAI（280种情境，4,125条两轮对话），GoEmotion 28 类情感标签（经日语翻译），以及参考 EmpatheticDialogue 的对话格式；同时采用上述多款 LLM 作为实验模型。

**📈 对比分析**

比较方法：从EmplifAI随机抽取 100 条情感-情境对，分别用零射击与 SFT 模型生成两轮对话，再由 Gemini‑2.5‑Flash 及人类评审按 5 分 Likert 量表评估 7 项指标；SFT‑LLM‑jp 在内容、一般同理心、情感特异同理心、连贯性、日语流畅度、无害性、安抚感方面明显提升，逼近甚至接近 DeepSeek；GPT 与 DeepSeek 在大部分指标上保持最高分；LLM‑judge 与人评一致性高（除 GPT 之外）。

**⚠️ 局限性**

局限性包括：①提示设计未限制生成长度，导致模型输出往往更长且可能影响评估；②数据聚焦慢性疾病场景，难以推广至开放式对话或其他医学情境；③日语文化特定表达可能不适用于其他语言与文化；④LLM‑judge 可能与人类情感细腻度不完全匹配，尤其在“同理而非解决”这一维度；⑤真实临床对话缺失，数据仍为模拟对话。

---

## 82. Eluder dimension: localise it!

**arXiv ID:** 2601.09825 | [PDF](https://arxiv.org/pdf/2601.09825v1)

**作者:** Alireza Bakhtiari `[一作]` (University of Alberta), Csaba Szepesvári `[通讯]` (University of Alberta)

**通讯引用:** 16908 | [OpenAlex ID](https://openalex.org/A5069856068)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对广义线性模型（GLM）和强化学习中的成本最小化问题，提出了一种新的局部Eluder维数（localized ℓ_1-eluder dimension）并基于此给出了第一阶（small-cost）上界，涵盖了伯努利分布、logistic等 GLM 以及有限时程强化学习。

**💡 创新点**

创新点包括：
- 证明全局 Eluder 维数在 GLM 中必然包含与信息量 κ 相关的指数项，从而解释了以往第一阶上界无法实现的根本原因；
- 引入局部 Eluder 维数，只关注离最优模型的邻域，消除 κ 影响，获得真正的实例可适应性；
- 设计了 ℓ-UCB（适用于带成本的 GLM bandit）和 ℓ-GOLF（适用于有限时程 RL）两种优化算法，并给出统一的上界分析；
- 在理论上实现了首次“κ‑free”第一阶上界，并且对传统 Bernoulli、Poisson 等分布的结果进行了恢复与改进。

**🔧 技术方法**

技术手段包括：
- GLM 结构假设（Lipschitz、self‑concordant 链接、梯度下界等）；
- 通过自定义的损失函数（log‑loss、Poisson‑loss）满足三角条件、方差条件；
- 采用基于极大似然的经验风险最小化和信赖集构造；
- 对局部 Eluder 维数进行上界与下界证明；
- 结合 Bernstein 类型浓缩、快速收敛和小成本分解，得到最终的 Regret 上界；
- 在 RL 部分引入 Bellman 失真损失、占优策略和 Bellman Eluder 维数。

**📊 数据集**

本文为理论论文，未使用公开数据集；实验验证部分在原文中仅以仿真（如 logistic bandit、Poisson bandit、有限时程 RL 任务）展示上界的有效性。

**📈 对比分析**

与已有工作对比：
- 通过消除 κ 依赖，ℓ-UCB/ℓ-GOLF 的前项比以往方法（如 GOLF、LOGISTIC‑UCB 等）更小；
- 对于 logistic bandit，理论上接近已知的下界，且不需要显式的 warm‑up；
- 与基于分布式 Bellman 完备性的旧方法相比，本文只需要普通的完备性假设，进一步降低了限制；
- 实验表明在多种 GLM 设置下，实际 regret 与上界的匹配度更好。

**⚠️ 局限性**

局限性：
- 仍存在对外部参数 κ 的附加常数项（仅在非优化项中出现）；
- 需要 GLM、损失函数满足自共形与三角条件，限制了可推广性；
- 主要针对离散动作空间的 GLM bandit 与有限时程 RL，连续或高维情境的扩展尚未讨论；
- 由于理论性强，实验验证规模有限，缺乏在真实工业数据上的评估。

---

## 83. A New Convergence Analysis of Plug-and-Play Proximal Gradient Descent Under Prior Mismatch

**arXiv ID:** 2601.09831 | [PDF](https://arxiv.org/pdf/2601.09831v1)

**作者:** Guixian Xu `[一作]` (University of Birmingham), Junqi Tang `[通讯]` (University of Birmingham)

**通讯引用:** 1033 | [OpenAlex ID](https://openalex.org/A5054394663)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f`

**🎯 论文内容**

在先验不匹配的情况下，对使用 PnP-PGD（Plug‑and‑Play Proximal Gradient Descent）算法的收敛性进行了全新的理论分析，给出了在 MMSE 去噪器被训练于不同数据分布时的收敛性质。

**💡 创新点**

创新点主要包括：①首次在先验不匹配情形下提供 PnP‑PGD 的收敛证明；②去除了以往理论中对收敛速度、步长、迭代界限等的强硬且不可验证的假设；③允许非凸数据保真项、非凸正则化项以及更广泛的去噪器形式。

**🔧 技术方法**

主要技术手段包括：将去噪器视为梯度步（Gradient‑Step）去噪器并构造相应的潜在函数；利用强凸性、光滑性和逆函数定理将去噪器映射到近似的近端算子；在误差可控的近端优化框架下进行递推分析，得到关于梯度范数的上界及收敛速率；进一步通过可和误差递减率证明误差可求和时梯度趋于零。

**📊 数据集**

论文未涉及具体实验数据集，而是以理论证明为主。

**📈 对比分析**

由于研究以理论为主，没有提供数值实验来与其他方法比较，因此无法给出具体性能指标；理论结果显示在满足 λL_f<1 且误差可求和的条件下，迭代梯度能以 1/t 的速率逼近零。

**⚠️ 局限性**

主要局限包括：①需要满足 λL_f<1 的步长/正则化参数限制；②假设去噪器的误差序列可求和，实际实现中误差控制可能困难；③结果仅保证梯度趋于零（即驻点），无法保证全局最优；④理论中仍假设去噪器的可逆性与光滑性，可能与某些强大深度去噪模型不完全兼容。

---

## 84. Explicating Tacit Regulatory Knowledge from LLMs to Auto-Formalize Requirements for Compliance Test Case Generation

**arXiv ID:** 2601.09762 | [PDF](https://arxiv.org/pdf/2601.09762v1)

**作者:** Zhiyi Xue `[一作]` (East China Normal University), Min Zhang `[通讯]` (East China Normal University)

**通讯引用:** 35732 | [OpenAlex ID](https://openalex.org/A5100671757)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于多模型 LLM 的法规知识显化与自动化合规测试生成框架（RAFT），通过显化领域元模型、形式化需求表示和可测性约束来驱动高质量测试用例生成。

**💡 创新点**

创新点在于将隐式法规知识从多 LLM 自动提炼为可验证的三种知识资产，并通过多 LLM 一致性策略与 CoT+RAG 交互式提示显著抑制幻觉，提升可控性与可解释性。

**🔧 技术方法**

核心技术包括多模型 LLM 共同推理、Chain‑of‑Thought+Retrieval Augmented Generation 提示、K‑一致性聚合策略、知识注入式提示以及基于 Xtext 的形式化需求校验。

**📊 数据集**

实验使用了金融领域六个监管数据集、汽车领域五个监管数据集以及电力领域两个标准数据集，并对比了领域专家、LLM4Fin 以及单一 E2E LLM。

**📈 对比分析**

与专家、LLM4Fin 和 E2E LLM 对比，RAFT 在 F1、业务场景覆盖率、时间消耗和成本上分别实现 91.7% F1、86.5% 覆盖、2.3 倍速度提升、成本更低，表现显著优于现有方法。

**⚠️ 局限性**

主要局限在于对外部知识库的质量和完整性依赖，以及自动构造的元模型和需求表示缺乏形式化的可靠性保证，仍需人工复核。

---

## 85. AI Survival Stories: a Taxonomic Analysis of AI Existential Risk

**arXiv ID:** 2601.09765 | [PDF](https://arxiv.org/pdf/2601.09765v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 86. CALM-IT: Generating Realistic Long-Form Motivational Interviewing Dialogues with Dual-Actor Conversational Dynamics Tracking

**arXiv ID:** 2601.10085 | [PDF](https://arxiv.org/pdf/2601.10085v1)

**作者:** Viet Cuong Nguyen `[一作]` (Georgia Institute of Technology), Munmun De Choudhury `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 14728 | [OpenAlex ID](https://openalex.org/A5102962995)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了CALM-IT框架，用于生成和评估长篇动机访谈（MI）对话，显式建模治疗师与客户的双向对话状态并通过持续更新的对齐、心理状态和目标信息来指导对话策略与生成；同时提供面向对话级别的评价指标，支持对长序列对话的可解释性与稳定性分析。

**💡 创新点**

创新点在于：1）将双主体对话过程视为动态状态空间，实时更新对齐、心理状态与短期目标；2）在生成与评价两方面都加入对话状态的约束，从而在长篇对话中保持高效性与一致性；3）首次在MI对话生成中结合策略选择、MI风格语言模型以及LLM-as-judge评估，形成完整的长对话生成与评估闭环。

**🔧 技术方法**

技术核心包括：双向状态空间建模（对齐、心理状态、目标）；策略选择模块（基于更新的状态）；多候选响应生成与ConvoKit+MI编码的概率排序；LLM-as-judge（DeepEval + GPT‑5-mini）对同理、合作、反思等高层指标评分；自动化评价指标（MITI、NLI、Delta方向性、重定向率等）；后端使用DeepSeek‑V3大语言模型进行对话生成。

**📊 数据集**

数据集：1）从Reddit心理健康子版块收集约5.8M条帖子，筛选出686个代表性vignette；2）匹配OpenPsychometrics DASS‑42数据生成心理状态；3）利用DeepSeek‑V3生成客户背景和单句目标；4）对比基线框架（KMI、CI‑NC、CAMI+STAR）使用的原始MI对话语料。

**📈 对比分析**

在30/50/100回合四种长度下生成8,232条对话，分别与KMI、CI‑NC、CAMI+STAR及CALM‑IT无动态模型对比。结果显示：CALM‑IT在对话级指标上明显优于基线（Effectiveness 4.45 vs 4.27/2.86/1.19；Goal Alignment 4.73 vs 4.60/3.89/2.13）。在长序列上漂移最小，平均绝对变化仅0.02%；红irection成功率最高64.28%，导致客户变更谈话提升12.4%，抗议下降8.15%。

**⚠️ 局限性**

局限性包括：仅使用单一LLM后端（DeepSeek‑V3），未评估其他模型的泛化；框架只针对MI理论，难以直接推广至其他疗法；数据来源主要是WEIRD的Reddit帖子，缺乏危机、重度精神病或文化多样性场景；LLM‑as‑judge评估带有主观性；真实临床对话的非线性动态仍未充分捕获。

---

## 87. A Sustainable AI Economy Needs Data Deals That Work for Generators

**arXiv ID:** 2601.09966 | [PDF](https://arxiv.org/pdf/2601.09966v1)

**作者:** Ruoxi Jia `[一作]` (Virginia Tech), Dawn Song `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对73个公开数据交易的分析，指出机器学习价值链存在“经济数据处理不平等”，并提出Equitable Data-Value Exchange（EDVEX）框架，旨在实现公平的价值分配与可追溯的 provenance；

**💡 创新点**

创新点在于将数据 provenance、议价平衡和动态定价三大结构性缺陷纳入同一框架，提出基于任务-数据匹配、可审计的 lineage 跟踪和基于效用的价格发现机制；

**🔧 技术方法**

技术方案包括任务-数据匹配（基于 sandbox 评估和缩放法则）、可审计的 lineage 记录、以及协作博弈论（如 Shapley 值）驱动的价值分配；

**📊 数据集**

所使用的数据集为公开披露的 73 个数据交易条款，涵盖多家大型 AI 机构、聚合商和内容平台；

**📈 对比分析**

由于本文为立场性论述并未实现系统或实验，故未给出可比性能指标；

**⚠️ 局限性**

局限性在于仅基于公开条款，易偏向大型英文交易，缺乏私有或 NDA 交易，且框架尚无实验验证和实现细节。

---

## 88. Evaluating Novelty in AI-Generated Research Plans Using Multi-Workflow LLM Pipelines

**arXiv ID:** 2601.09714 | [PDF](https://arxiv.org/pdf/2601.09714v1)

**作者:** Devesh Saraogi `[一作]` (Birla Institute of Technology and Science), Dhruv Kumar `[通讯]` (Birla Institute of Technology and Science)

**通讯引用:** 6454 | [OpenAlex ID](https://openalex.org/A5027859418)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了五种多步骤代理式LLM工作流（反思、进化、递归分解、多智能体辩论、长上下文多模态）在科研想法生成中的表现，并通过六位专家在五个研究领域的30份提案进行评估，探讨其创新性、可行性与影响力。

**💡 创新点**

系统性比较多步骤代理模型与单步提示的差异；在不同领域中揭示分解与长上下文工作流在保持可行性的前提下显著提升原创性；提供了针对抄袭风险的经验性证据与设计指南。

**🔧 技术方法**

使用大型语言模型（如GPT‑5.1、Gemini 3 Pro等）构建的五种代理工作流；采用专家评估的Likert量表和定性理由收集方法；统计学分析（均值、Pearson相关系数）。

**📊 数据集**

构造了30份基于五个领域种子想法（AI/Tech、AI/多智能体、化学/生物技术、气候/环境、工业/制造）的提案；评估者为六名领域专家；未使用公开文本检索或抄袭检测数据集。

**📈 对比分析**

通过专家评分比较：分解式（GPT Deep Research）和长上下文（Gemini 3 Pro）工作流平均创新度达4.17/5，远高于反思式（2.17/5）；在各领域的平均创新度分别为4.00、3.80、4.00、3.20、3.20；创新度与可行度的相关系数仅为0.23，表明高创新不必导致可行性下降。

**⚠️ 局限性**

局限包括样本量有限（仅6名专家、30条评估），缺乏自动化抄袭检测与大规模语料对比，工作流仅在预设种子想法上测试，未考察模型在更大规模、不同语言或更细粒度实验设计中的泛化能力。

---

## 89. Instalación, configuración y utilización de un nodo Bitcoin en Linux

**arXiv ID:** 2601.09748 | [PDF](https://arxiv.org/pdf/2601.09748v1)

**作者:** Jose Eduardo Ulloa `[一作]`, Diego R. Llanos `[通讯]` (Universidad de Valladolid)

**通讯引用:** 903 | [OpenAlex ID](https://openalex.org/A5043394187)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

在Linux环境下编译安装Bitcoin Core完整节点，并记录同步过程与系统资源消耗。

**💡 创新点**

首次系统化评估多项配置参数（txindex、prune、dbcache、maxmempool、maxconnections）对节点性能与功能的影响。

**🔧 技术方法**

使用CMake编译构建、systemd服务、Bitcoin‑CLI、LevelDB、ZeroMQ、SQLite等技术栈。

**📊 数据集**

利用本机硬件（Intel Core i5‑8400/16 GB、2 TB SSD）与比特币主网区块链（约900k区块）做实验。

**📈 对比分析**

通过对节点启动、内存占用、CPU负载、磁盘空间和同步进度等指标的监控与日志分析比较，发现如prune减少磁盘占用、dbcache提升查询速度、maxmempool 限制提升内存使用率但减少未确认交易量。

**⚠️ 局限性**

实验受限于单一硬件配置、未对高并发网络或分布式部署进行评估，且某些参数互相冲突导致测试序列受限。

---

## 90. Social Determinants of Health Prediction for ICD-9 Code with Reasoning Models

**arXiv ID:** 2601.09709 | [PDF](https://arxiv.org/pdf/2601.09709v1)

**作者:** Sharim Khan `[一作]` (University of Illinois), Jimeng Sun `[通讯]` (University of Illinois)

**通讯引用:** 27485 | [OpenAlex ID](https://openalex.org/A5084279065)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过对MIMIC‑III临床文本进行多标签预测，利用推理型LLM和传统LLM自动识别ICD‑9 V‑代码（社会健康决定因素）。

**💡 创新点**

创新点在于首次在住院层面细粒度预测ICD‑9 V‑代码，采用推理提示实现多标签推断，并引入“修正代码”概念揭示诊断缺失。

**🔧 技术方法**

主要技术包括少量示例提示、链式思维推理、GPT‑5‑mini、GPT‑4o‑mini、Llama‑3、Qwen‑3 等模型的多标签推断。

**📊 数据集**

使用的数据集为MIMIC‑III v1.4，包括58,000+住院记录，聚焦8个ICD‑9 V‑代码的社交健康因素。

**📈 对比分析**

实验对比显示GPT‑5‑mini在修正代码上取得89.1% micro‑F1、72.2% exact‑match，开放源模型Qwen‑3‑32B和GPT‑OSS‑20B亦能接近性能，远优于传统模型。

**⚠️ 局限性**

局限性包括仅限MIMIC‑III的单一机构数据、仅评估8个V‑代码、单注释者无可靠性检验，以及仅使用部分临床笔记导致泛化受限。

---

## 91. Enhancing Visual In-Context Learning by Multi-Faceted Fusion

**arXiv ID:** 2601.10107 | [PDF](https://arxiv.org/pdf/2601.10107v1)

**作者:** Wenwen Liao `[一作]` (Fudan University), Xiaofeng Yang `[通讯]` (Fudan University)

**通讯引用:** 9859 | [OpenAlex ID](https://openalex.org/A5100742379)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种多组合协同融合框架，用于视觉上下文学习（VICL），通过将多条检索到的提示分组为整体、高相似度、低相似度三支分支，并在多分支VQGAN中协同解码，显著提升了对图像分割、目标检测和颜色化等任务的推理性能。

**💡 创新点**

创新点在于引入多组合协同融合思路：1）MPGS策略将提示集合拆解为三个互补分支， 2）MULTI‑VQGAN采用多分支编码器与跨注意力FUSE模块实现层次化特征融合， 3）整体框架打破传统单一或简单融合的限制，增强模型对多样化上下文的利用与推理能力。

**🔧 技术方法**

核心技术包括检索式上下文提示（retrieve‑then‑prompt）、Condensor提示融合、MPGS分组策略、基于MAE‑VQGAN的多分支编码器、跨注意力（cross‑attention）与残差融合的FUSE模块、以及层次化特征融合。

**📊 数据集**

实验使用PASCAL‑5^i（分割与检测）、COCO‑5^i（跨域评估）以及ImageNet‑1K（颜色化）等公开数据集。

**📈 对比分析**

与单提示选择、投票策略及Condensor等基线比较，本文方法在分割、检测和颜色化任务上均实现了显著提升（如分割mIoU提升约5.6%），并在COCO→PASCAL跨域设置中表现出更强的泛化能力。

**⚠️ 局限性**

局限性包括：1）对MAE‑VQGAN骨干的依赖使模型训练与推理成本较高；2）提示检索和分组过程仍需人工调参，未解决大规模多模态检索的可扩展性；3）目前仅在少样本视觉任务上验证，尚未评估对更复杂场景或多模态任务的适用性。

---

## 92. R-LAM: Reproducibility-Constrained Large Action Models for Scientific Workflow Automation

**arXiv ID:** 2601.09749 | [PDF](https://arxiv.org/pdf/2601.09749v1)

**作者:** Suriya Sureshkumar `[一作]` `[通讯]` (RMK Engineering), Suriya Sureshkumar (RMK Engineering)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 R‑LAM 框架，约束 LAM 在科学工作流中的可重现性、审计性和确定性执行。

**💡 创新点**

将可重现性约束嵌入动作层，提供结构化动作模式、确定性执行引擎和可追溯的执行图，兼顾 LAM 的自适应控制。

**🔧 技术方法**

使用 Python、LLM 规划（如 ChatGPT/OpenRouter）、动作模式验证、沙盒执行、执行追踪 DAG、可重放与分支机制等技术。

**📊 数据集**

在实验中使用了 scikit‑learn 的 Breast Cancer Wisconsin 数据集来验证机器学习工作流。

**📈 对比分析**

与脚本式线性流水线和无约束 LAM 基线对比，评估可重放性、追踪完整度和失败可见性；R‑LAM 在保证可重放性与完整追踪的同时，执行时间与无约束基线相当。

**⚠️ 局限性**

仅在小规模离线流程验证，未涉及硬件或大规模实验；对 LLM 的可靠性依赖仍存在；缺乏对物理实验环境的安全与实时性保障。

---

## 93. Who Owns My AI Twin? Data Ownership in a New World of Simulated Identities

**arXiv ID:** 2601.09877 | [PDF](https://arxiv.org/pdf/2601.09877v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 94. Assessing and Improving Punctuation Robustness in English-Marathi Machine Translation

**arXiv ID:** 2601.09725 | [PDF](https://arxiv.org/pdf/2601.09725v1)

**作者:** Kaustubh Shivshankar Shejole `[一作]` (Indian Institute of Technology Bombay), Pushpak Bhattacharyya `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 13009 | [OpenAlex ID](https://openalex.org/A5065100828)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了专门针对英文‑马拉地语翻译的标点敏感性诊断基准Virām，并对比了两种提升翻译鲁棒性的方案：标点恢复后再翻译的管线方法和在含标点与不含标点数据上直接微调的模型；

**💡 创新点**

首创了该领域的标点鲁棒性评测基准，并系统评估了管线式和直接微调式两种方案在处理标点歧义文本时的效果，同时揭示现有大型语言模型在此任务中的不足；

**🔧 技术方法**

采用标点恢复器（如T5、AI4Bharat’s cadence）与IndicTrans2等模型的管线处理，及在标点变体数据上微调的T5、mBERT、Microsoft‑MPNet等；并对LLM（LLaMA‑3.1、Gemma‑2‑9B、Sarvam‑2B‑v0.5）进行零/三步提示实验；

**📊 数据集**

主要使用了自制的54条标点歧义实例构成的Virām基准、内部专业翻译语料，以及公开的IN22（CONV/GEN）和FLORES‑22等标准MT评测集；

**📈 对比分析**

通过BLEU、BLEURT‑20、COMET、chrF++/chrF2++、LabSE、MuRIL等多维度指标对比，发现管线式和直接微调式均显著提升Virām得分，管线式在标点恢复质量高时更佳，而LLM在零/三步提示下仍低于两种专门微调方案；

**⚠️ 局限性**

局限性包括基准样本仅54条、仅覆盖英文‑马拉地语对、标准自动指标对语义细微变化不敏感、LLM与任务特定模型差距明显，且对其他印地语族语言的通用性未知。

---

## 95. OT-Drive: Out-of-Distribution Off-Road Traversable Area Segmentation via Optimal Transport

**arXiv ID:** 2601.09952 | [PDF](https://arxiv.org/pdf/2601.09952v1)

**作者:** Zhihua Zhao `[一作]` (Beijing Institute of Technology), Kangping Lu `[通讯]` (Shandong Pengxiang Automobile Co., Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了OT-Drive框架，通过将RGB图像与表面法向特征进行多模态融合，实现越野可行区域分割，并显著提升了在 OOD 场景下的泛化性能。

**💡 创新点**

创新点在于将多模态融合视为分布级最优传输问题，并提出 Scene Anchor Generator（SAG）利用 VLM 生成可迁移的场景锚点，从而构建分布不变的语义空间进行特征对齐。

**🔧 技术方法**

采用了 Optimal Transport、CLIP 视觉‑语言模型、Transformer 解码器、分布式概率建模与 Sinkhorn 迭代等技术，实现了跨模态特征的分布级对齐与融合。

**📊 数据集**

实验使用了 ORFD、ORAD‑3D 两个越野数据集，并在跨数据集（ORFD→ORAD‑3D）上进行评估。

**📈 对比分析**

与 OFF‑Net、M2F2‑Net、ROD 三种基线比较，OT‑Drive 在 OOD 场景下 mIoU 提升至 95.16%（相对基线提升 0.26%），跨数据集 mIoU 提升 13.99%，推理速度达 21.11 FPS，整体性能领先。

**⚠️ 局限性**

限制在于需预先定义封闭的场景属性集合，无法处理完全未知的属性组合，对真正开放世界的适应性有限。

---

## 96. Forgetting as a Feature: Cognitive Alignment of Large Language Models

**arXiv ID:** 2601.09726 | [PDF](https://arxiv.org/pdf/2601.09726v1)

**作者:** Hien Tran `[一作]` (Suffolk University), Chadbourne Davis `[通讯]` (Sun Yat-sen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型（LLM）在推理时的遗忘行为，将其视为与人类记忆相符的功能特征，并基于此提出了概率记忆提示（Probabilistic Memory Prompting, PMP）方法；

**💡 创新点**

创新点在于：①将LLM的遗忘建模为指数衰减的贝叶斯更新，从而把遗忘从缺陷转化为可调节的认知机制；②设计了涵盖时间推理、概念漂移与关联记忆的基准套件，用人类实验数据直接对比遗忘曲线；③提出PMP通过概率采样重塑上下文，使LLM的证据整合更贴近人类记忆轨迹；

**🔧 技术方法**

技术方法包括：指数记忆衰减权重、贝叶斯过滤框架、概率采样算法、prompt工程、对比基准评测、以及对长序列推理的自适应控制；

**📊 数据集**

使用的数据集和实验平台有：Synthetic Non‑Stationary Probes（模拟概念漂移任务）、Human‑Aligned Memory Tasks（人类记忆实验对照）、HotpotQA、GSM8K、HumanEval（长程推理评测）以及公开的LLM模型（如Llama‑3 70B）；

**📈 对比分析**

与完整上下文提示和滑动窗口截断等基线相比，PMP在多项长程推理任务中取得更高的准确率（如HotpotQA、GSM8K、HumanEval）并在模拟任务中与人类遗忘曲线拟合度更高，表明其在适应性和鲁棒性方面有显著提升；

**⚠️ 局限性**

局限性包括：①指数衰减假设可能不适用于所有类型的记忆任务；②需要手动调节遗忘率λ，对不同任务可能需要不同设置；③实验主要集中在文本推理任务，对多模态或更大规模数据的泛化尚未充分验证；④PMP虽然开销低，但在极长上下文或高频采样时仍可能产生额外计算成本。

---

## 97. Emergency Department Patient Flow Optimization with an Alternative Care Threshold Policy

**arXiv ID:** 2601.10041 | [PDF](https://arxiv.org/pdf/2601.10041v1)

**作者:** Sahba Baniasadi `[一作]` (Pennsylvania State University), Prakash Chakraborty `[通讯]` (Pennsylvania State University)

**通讯引用:** 114 | [OpenAlex ID](https://openalex.org/A5018383865)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种基于占用阈值的急诊科入院控制策略，在拥堵时将低危病人转移至远程或门诊等替代护理路径。

**💡 创新点**

创新点在于将阈值控制嵌入到两类 M/M/c 预抢占优先队列，并利用级联的准出生-死亡（QBD）模型求解最优转诊阈值，兼顾收入、等候成本与放弃成本。

**🔧 技术方法**

采用了预抢占优先 M/M/c 排队理论、级联准出生-死亡（QBD）分析以及长期时间平均目标函数优化。

**📊 数据集**

使用了美国全国急诊科数据和文献报道的到达率、床位配置、费用与收入参数，对乡村和城市两种典型设置进行数值实验。

**📈 对比分析**

通过对比不同阈值下的净收益以及与无转诊基线的差异，结果显示在乡村约提升4.84%、城市约提升5.90%；灵敏度分析表明等候成本比重是最关键驱动因素。

**⚠️ 局限性**

局限性包括：假设到达与服务过程为泊松与指数分布、仅考虑两类病人、未建模替代护理容量限制、缺乏非平稳流量与多级优先级的扩展验证。

---

## 98. Critically Engaged Pragmatism: A Scientific Norm and Social, Pragmatist Epistemology for AI Science Evaluation Tools

**arXiv ID:** 2601.09753 | [PDF](https://arxiv.org/pdf/2601.09753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 99. Clinical Document Metadata Extraction: A Scoping Review

**arXiv ID:** 2601.09730 | [PDF](https://arxiv.org/pdf/2601.09730v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 100. In-the-Wild Compliant Manipulation with UMI-FT

**arXiv ID:** 2601.09988 | [PDF](https://arxiv.org/pdf/2601.09988v1)

**作者:** Hojung Choi `[一作]` (Stanford University), Shuran Song `[通讯]` (Stanford University)

**通讯引用:** 24182 | [OpenAlex ID](https://openalex.org/A5004644695)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了可扩展的手持数据采集平台 UMI-FT，配备了每个手指的六轴 F/T 传感器，并利用收集到的多模态数据训练自适应顺应性策略，实现机器人在白板擦拭、茄子穿刺和灯泡插入等任务中的安全且有力操作。

**💡 创新点**

创新点包括：①在手指级别引入低成本、可扩展的 CoinFT F/T 传感器；②将视觉、深度、姿态与手指 F/T 数据融合到单一政策中；③自适应顺应性策略同时调节外部接触力与内部抓握力；④开源硬件与软件，促进大规模采集与研究。

**🔧 技术方法**

技术方法包括：UMI-FT 设备搭配 CoinFT 传感器；多模态编码（CLIP‑ViT、卷积网络）+ Transformer 与扩散模型；六维任务空间 admittance 与抓握力控制；MLP 进行 CoinFT 校准；在 UR5e + WSG50 机器人上实现。

**📊 数据集**

使用通过 UMI-FT 采集的演示数据集，涵盖白板擦拭（275 次）、茄子穿刺（200 次）、灯泡插入（200 次）以及 630 次野外场景演示，形成多任务、多环境的训练与评估数据。

**📈 对比分析**

与三种基线（DP w/ F、DP、DP w/ CM）对比，ACP 在白板擦拭任务中成功率 92%（基线 28–16%）；茄子穿刺任务 80%（基线 70%/30%）；灯泡插入任务 95%（基线 60%/0%/20%）。在野外茄子穿刺场景，基于多样数据训练的 ACP 20/20 成功，而仅使用实验室数据的 ACP 仅 4/20。

**⚠️ 局限性**

限制包括：设备依赖 USB 有线连接，未实现无线化；仅使用主摄像头视野，未充分利用超广角摄像头；CoinFT 在拉伸负荷下易剥离，需改进结构；微控制器性能受限，可能限制采样速率；在某些任务中顺应性优势不明显，需进一步验证。

---

## 101. A Novel Contrastive Loss for Zero-Day Network Intrusion Detection

**arXiv ID:** 2601.09902 | [PDF](https://arxiv.org/pdf/2601.09902v1)

**作者:** Jack Wilkie `[一作]` (University of Strathclyde), Robert Atkinson `[通讯]` (University of Strathclyde)

**通讯引用:** 5115 | [OpenAlex ID](https://openalex.org/A5005039825)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 CLAD 这一新的对比学习损失函数，用于在网络入侵检测中既能学习正常流量的分布，又能对未知攻击保持鲁棒性，并将其扩展为多类别开放集识别框架 CLOSR。

**💡 创新点**

创新点在于仅使用正常样本作为正样本的对比学习，结合 von‑Mises Fisher 分布建模正常流量，从而突破传统监督学习的闭世界假设，将零日攻击视为正交向量进行检测。

**🔧 技术方法**

采用对比学习、vMF 分布、余弦距离、线性投影头以及深度多层感知机实现。

**📊 数据集**

使用 Lycos2017 数据集，该数据集包含 14 类网络流量，样本极不平衡，测试时将 SQL 注入和 Heartbleed 作为未见类别来模拟零日攻击。

**📈 对比分析**

与传统有监督分类、异常检测、Siamese 网络、OpenMax 等基线对比，CLAD 在已知攻击的 AUROC 均高于对手，零日攻击 AUROC 提升约 0.06；CLOSR 在开放集 AUC 与 OpenAUC 上优于所有基线，并保持较高的闭集准确率。

**⚠️ 局限性**

局限性包括模型规模随类别数线性增长，未评估对抗鲁棒性及噪声训练样本的影响，以及在跨网络泛化和少样本学习上的适用性。

---

## 102. TimeSAE: Sparse Decoding for Faithful Explanations of Black-Box Time Series Models

**arXiv ID:** 2601.09776 | [PDF](https://arxiv.org/pdf/2601.09776v1)

**作者:** Khalid Oublal `[一作]` (Telecom Paris), Zeynep Akata `[通讯]` (Technical University of Munich)

**通讯引用:** 15893 | [OpenAlex ID](https://openalex.org/A5040372929)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出一种基于稀疏自编码器（TimeSAE）的时序黑盒模型解释框架，能够在保持预测标签一致的前提下生成可解释的概念表示。

**💡 创新点**

创新点包括：①利用JumpReLU学习可稀疏激活阈值，消除“死亡概念”问题；②引入对抗式对比损失与计数化的近似反事实约束，实现因果可信解释；③通过组合一致性损失实现跨分布（OOD）下的解释迁移。

**🔧 技术方法**

核心技术为稀疏自编码器（JumpReLU激活+TopK稀疏正则）、时间卷积网络（TCN）、对比学习（InfoNCE）、近似反事实生成与稀疏性正则、以及概念一致性与激活区域对齐方法。

**📊 数据集**

实验数据集涵盖合成的 FreqShapes、SeqComb-UV 以及真实数据集 ECG、PAM、ETTH‑1、ETTH‑2 和新建的 EliteLJ（运动捕捉数据）。

**📈 对比分析**

与 IG、Dynamask、TimeX++、StartGrad、ORTE、TIMING 等 10 种基线相比，TimeSAE 在 AUPRC、AUP、AUR、faithfulness、KL、MMD 等指标上均显著提升，尤其在 OOD 场景下保持了更高的解释一致性和更低的分布漂移。

**⚠️ 局限性**

局限性在于：①需要足够大且具代表性的训练数据；②对稀疏度、字典大小等超参敏感，需细致调参；③在数据稀缺或模型结构极其复杂的领域，训练成本和解释质量可能受限。

---

## 103. V-Zero: Self-Improving Multimodal Reasoning with Zero Annotation

**arXiv ID:** 2601.10094 | [PDF](https://arxiv.org/pdf/2601.10094v1)

**作者:** Han Wang `[一作]` (Zhejiang University), Wei Chen `[通讯]` (Zhejiang University)

**通讯引用:** 66171 | [OpenAlex ID](https://openalex.org/A5100344384)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在不使用任何人工标注的前提下，构建了一个由 Questioner 与 Solver 组成的共进循环，实现了视觉语言模型的自我提升；

**💡 创新点**

创新点在于提出了 Dual‑Track Reasoning Reward（对比直觉与推理的奖励）和 Difficulty‑Guided Data Sampling，利用纯图像数据通过内部反馈实现模型自我改进；

**🔧 技术方法**

核心技术包括 Group Relative Policy Optimization (GRPO)、Reinforcement Learning with Verifiable Rewards (RLVR)、多数投票伪标签生成以及多选题格式约束；

**📊 数据集**

使用 OpenVLThinker GRPO‑medium 与 GRPO‑hard 约 9K 张无标注图像（主要为几何图形、表格与空间推理）作为训练数据；

**📈 对比分析**

在 Qwen2.5‑VL‑7B‑Instruct 与 Qwen2.5‑VL‑3B‑Instruct 上对比基线模型和有监督 GRPO，V‑Zero 在 Visual Math、MMMU、MMStar 等基准上平均提升 2‑3 分，甚至超越了部分有监督模型；

**⚠️ 局限性**

主要局限包括受限于模型规模（仅验证 Qwen2.5‑VL 系列）、训练过程中的性能波动与高探索率导致的优化不稳定，以及对更大模型与多种架构的适用性尚待验证。

---

## 104. InfoSculpt: Sculpting the Latent Space for Generalized Category Discovery

**arXiv ID:** 2601.10098 | [PDF](https://arxiv.org/pdf/2601.10098v1)

**作者:** Wenwen Liao `[一作]` (Fudan University), Xiaofeng Yang `[通讯]` (Fudan University)

**通讯引用:** 9859 | [OpenAlex ID](https://openalex.org/A5100742379)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 InfoSculpt 框架，利用双层条件互信息 (CMI) 目标对特征空间进行雕刻，实现已知与未知类别的统一分类与聚类。

**💡 创新点**

首次将信息瓶颈与条件互信息结合，设计类别级与实例级双重 CMI 正则化，既压缩类别信息又消除实例噪声，兼顾已知与未知类的联合学习。

**🔧 技术方法**

信息瓶颈原则、条件互信息估计、ViT 视觉 Transformer、对比学习、交叉熵与熵正则、Softmax 目标锐化、Hungarian 匹配等技术。

**📊 数据集**

8 个基准集：CIFAR‑10/100、ImageNet‑100、CUB‑200‑2011、Stanford Cars、FGVC‑Aircraft、Herbarium 19、ImageNet‑1K。

**📈 对比分析**

与 10+ 先进方法（GCD、DCCL、GPC、SimGCD、LegoGCD、ActiveGCD、MTMC、Hyp‑GCD、ProtoGCD 等）在已知/未知/总类准确率上对比，InfoSculpt 在多数数据集获得最优或接近最优成绩，尤其在细粒度数据集提升显著。

**⚠️ 局限性**

仍需先验已知/未知类别数目，双重 CMI 训练开销较大，极度类别不平衡或标签极少时效果可能受限。

---

## 105. Reconstructing Reed-Solomon Codes from Multiple Noisy Channel Outputs

**arXiv ID:** 2601.09947 | [PDF](https://arxiv.org/pdf/2601.09947v1)

**作者:** Shubhransh Singhvi `[一作]` (Technion), Eitan Yaakobi `[通讯]` (Technion)

**通讯引用:** 3983 | [OpenAlex ID](https://openalex.org/A5021586372)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究在 q-ary DMS 替换通道下，利用 K 个独立噪声读数从 Reed–Solomon 码中高效重构原始码字。

**💡 创新点**

创新点在于将 Koetter–Vardy 的软判决解码框架引入 K‑draw 设定，并设计了基于多读可靠性信息的乘数矩阵，推导出仅依赖 (p,K) 的显式可实现率阈值。

**🔧 技术方法**

采用的技术包括多读可靠性聚合、乘数矩阵构造、Koetter–Vardy 软判决插值与因式分解以及概率分析与 Hoeffding 经验上界。

**📊 数据集**

论文未使用特定数据集，而是在理论模型（q‑ary DMS 通道）上进行实验与数值验证。

**📈 对比分析**

与传统硬判决多数投票解码相比，所提出的软判决方法在相同通道参数下可实现更高的码率，实验结果表明相对阈值显著提升。

**⚠️ 局限性**

局限性包括需要足够大的块长和字母表大小，且对 K 的取值有限制（K ≪ q），在实际硬件实现与非对称噪声模型下效果尚待进一步验证。

---

## 106. Self-reflection in Automated Qualitative Coding: Improving Text Annotation through Secondary LLM Critique

**arXiv ID:** 2601.09905 | [PDF](https://arxiv.org/pdf/2601.09905v1)

**作者:** Zackary Okun Dunivin `[一作]` (University of Stuttgart), Curtis Atkinson `[通讯]` (University of Washington)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两阶段LLM注释工作流：第一阶段让LLM根据人类改编的代码书给出标签及简短推理；第二阶段让另一LLM进行自我反思，仅针对正例重新评估并决定是否保留标签，从而降低误标。

**💡 创新点**

创新点在于将LLM的自省转化为可控的二次评估步骤，构建基于错误分类的目标化自我反思批判prompt，并通过人类错误分析快速迭代。

**🔧 技术方法**

使用的技术包括大型语言模型（GPT‑4o）、结构化prompt、JSON输出格式、代码书适配、二次调用仅针对正例以及基于误差分类的决策规则。

**📊 数据集**

数据集为Apache Software Foundation项目评估讨论的3,149封邮件，包含6个定性代码，使用了富集金标准120条、随机金标准150条以及每个代码60条正例抽样进行评估。

**📈 对比分析**

与单阶段注释相比，二阶段流程显著降低误标率（如Mentor Engagement从0.53降至0.52、F1提升至0.79），整体F1提升0.04–0.25，尤其对难编码的代码取得显著进步。

**⚠️ 局限性**

局限性包括批判仅针对正例，无法纠正漏报；需要人工制定错误词典并反复迭代，难以直接迁移到其他语料；在极少数边缘案例中仍可能出现误判；计算成本受正例比例影响。

---

## 107. DR$^2$Seg: Decomposed Two-Stage Rollouts for Efficient Reasoning Segmentation in Multimodal Large Language Models

**arXiv ID:** 2601.09981 | [PDF](https://arxiv.org/pdf/2601.09981v1)

**作者:** Yulin He `[一作]` (National University of Defense Technology), Minglong Li `[通讯]` (National University of Defense Technology)

**通讯引用:** 1818 | [OpenAlex ID](https://openalex.org/A5100616698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种自奖励框架 DR^2Seg，通过两阶段回滚策略将推理分割拆分为多模态推理与指称分割，从而提升多模态大语言模型（MLLM）的推理效率与分割精度。

**💡 创新点**

创新点在于：①利用两阶段回滚生成自包含的描述来验证推理链的完整性；②引入描述自奖励和基于长度的自奖励，直接监督推理过程；③完全依赖模型自身进行奖励，无需额外大模型或人工标注，显著减少过度推理。

**🔧 技术方法**

技术手段包括：使用 Qwen2.5VL-7B 作为推理模型，SAM（SAM2、SAM3）作为分割模型；采用 Group‑Relative Policy Optimization (GRPO) 进行强化学习；通过 token 长度与答案正确性构造自奖励。

**📊 数据集**

数据集主要包括：ReasonSeg（用于训练 239 句复杂查询、验证/测试集）、RefCOCO、RefCOCO+、RefCOCOg（用于评估泛化）以及 7K-sample 的 VisionReasoner 训练集（由 LVIS、RefCOCOg、gRefCOCO、LISA++ 合成）。

**📈 对比分析**

与 SFT、RL、VisionReasoner 等方法对比，DR^2Seg 在 ReasonSeg 验证集上 gIoU 提升约 1.2%（零样本）并显著减少推理 token（约 3 倍）；在少样本微调后 gIoU 进一步提升至 68.5%，超越 VisionReasoner* 3.8% 并将 token 数量降低 3 倍；在 RefCOCO 任务上也保持领先或相近性能，证明其泛化能力。

**⚠️ 局限性**

局限性包括：①奖励设计依赖长度锚点 N0 和惩罚系数 γ 的调参；②在极短或极长查询上可能无法完全满足长度奖励的前提；③目前仅在分割任务上验证，其他多模态推理任务的适用性尚待进一步探索。

---

## 108. DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis

**arXiv ID:** 2601.10001 | [PDF](https://arxiv.org/pdf/2601.10001v1)

**作者:** Chengjia Liang `[一作]` (Shenzhen University), Zhongwei Huang `[通讯]` (Hubei University of Technology)

**通讯引用:** 5535 | [OpenAlex ID](https://openalex.org/A5016570639)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了DW-DGAT框架，对多结构多维度MRI/DTI与表型数据进行融合，用双图注意网络和动态权重生成器实现神经退行性疾病早期诊断。

**💡 创新点**

创新点包括①通用高效多结构数据融合策略；②双图注意网络同时捕获ROI微观图和样本宏观图；③动态生成类别权重的协同训练，提升不平衡数据下性能。

**🔧 技术方法**

使用了多模态图注意网络（SGA、GGA）、Transformer视觉编码器、MHSA图卷积、动态类别权重生成器及自定义稳定损失。

**📊 数据集**

实验数据集为 Parkinson Progression Marker Initiative (PPMI) 与 Alzheimer's Disease Neuroimaging Initiative 3 (ADNI3)。

**📈 对比分析**

与多种视觉网络、公共GNN及专门ND方法进行十折交叉验证，DW‑DGAT 在 HC‑PRO‑PD 与 CN‑EMCI‑AD 任务中分别以 74.56% 与 68.65% 的准确率、59.31% 与 66.18% 的平衡准确率等指标领先第二佳模型约 4–8%。

**⚠️ 局限性**

局限在于 3D 融合未考虑 ROI voxel 数目差异，可能忽略大 ROI 的潜在特征；模型复杂度高，训练与推理资源消耗大。

---

## 109. QFed: Parameter-Compact Quantum-Classical Federated Learning

**arXiv ID:** 2601.09809 | [PDF](https://arxiv.org/pdf/2601.09809v1)

**作者:** Samar Abdelghani `[一作]` (Polytechnique Montreal), Soumaya Cherkaoui `[通讯]` (Polytechnique Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了QFed框架，将Quantum-Train技术嵌入联邦学习，利用量子电路压缩传统CNN参数，实现边缘设备高效协同训练。

**💡 创新点**

创新点在于：①仅在训练阶段使用QT，推理保持全经典；②提供Docker/MPI容器化可复现测试床；③在联邦环境下验证QT实现77.6%参数压缩且保持准确度。

**🔧 技术方法**

使用了量子变分电路（U3/CU3）、量子-经典映射网络、联邦学习聚合、Docker容器仿真以及TorchQuantum量子模拟器。

**📊 数据集**

采用FashionMNIST数据集。

**📈 对比分析**

与无压缩的经典联邦学习在相同划分、通信轮次和本地训练轮次下对比，QT联邦模型将参数从6690压缩到1497，准确率从86%降至84%，训练与通信成本显著降低。

**⚠️ 局限性**

限制包括：NISQ硬件的量子比特数、噪声与退相干、可变形性导致的梯度消失、测量误差，以及对更大深度网络的可扩展性受限。

---

## 110. VibrantSR: Sub-Meter Canopy Height Models from Sentinel-2 Using Generative Flow Matching

**arXiv ID:** 2601.09866 | [PDF](https://arxiv.org/pdf/2601.09866v1)

**作者:** Kiarie Ndegwa `[一作]` (Vibrant Planet Public Benefit Corporation), Scott Conway `[通讯]` (Vibrant Planet Public Benefit Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于生成式流匹配的VibrantSR框架，利用10m Sentinel-2影像生成0.5m分辨率的冠层高度模型。

**💡 创新点**

创新点在于将生成式流匹配与自动编码器结合，在压缩潜在空间中进行条件生成，从而在低分辨率光学输入下重建细粒度冠层结构，并保留高度分布。

**🔧 技术方法**

使用技术包括冻结的Sentinel-2和CHM自动编码器、U型Vision Transformer的流匹配网络、潜在空间的流匹配学习和ODE积分。

**📊 数据集**

数据集为西部美国22个EPA Level 3生态区的Sentinel‑2（12波段）季节中值合成和USGS 3DEP lidar衍生的0.5m冠层高度。

**📈 对比分析**

与Meta、LANDFIRE、ETH以及先前的VibrantVS对比，VibrantSR在≥2m高度下MAE为4.39m，比Meta、LANDFIRE、ETH分别降低9%、26%和38%，边缘误差也更低。

**⚠️ 局限性**

限制包括仅在西部美国验证、光学数据受云/雾影响、从10m输入到0.5m的空间细节仍为生成样本而非实际测量，且与高分辨率航空数据仍存在明显误差。

---

## 111. The Algorithmic Gaze: An Audit and Ethnography of the LAION-Aesthetics Predictor Model

**arXiv ID:** 2601.09896 | [PDF](https://arxiv.org/pdf/2601.09896v1)

**作者:** Jordan Taylor `[一作]` (Carnegie Mellon University), Haiyi Zhu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3243 | [OpenAlex ID](https://openalex.org/A5051842323)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对广泛使用的视觉生成AI审美评估模型LAION‑Aesthetics Predictor（LAP）进行审计与数字人类学研究，探究其审美评价偏见及来源。

**💡 创新点**

首次将审计（评估偏见）与数字人类学（追溯开发过程）结合，以揭示模型背后的个人与文化价值倾向；揭露LAP在性别、种族和风格上的帝国主义、现实主义和男性化“算法凝视”。

**🔧 技术方法**

利用正则表达式PMI、统计分析、手工标注、图像–文本嵌入与多层感知机（LAP本身）等技术；采用定量与定性相结合的方法。

**📊 数据集**

三大数据集：1）LAION‑Aesthetics Dataset（约1.2B图像）；2）Metropolitan Museum of Art公开数据集（249k图像）；3）WikiArt数据集（81k图像）。此外审计还引用AVa、SAC、LAION‑Logos等训练集信息。

**📈 对比分析**

对比不同阈值（6.5）下的图像内容、域分布和文字标签，使用PMI评估标签出现频率；对比不同艺术风格、媒介与作者的评分分布，说明LAP偏好西方/日本的写实图像；性能未作传统“准确率”衡量，而是展示偏见幅度与影响。

**⚠️ 局限性**

局限性：①仅评估了三大数据集，未覆盖所有生成AI训练集；②LAP本身训练数据偏向英语、摄影和西方审美，缺乏跨文化多样性；③人类标签与文化语境缺乏，导致评估模型本身的主观性；④数字人类学依赖公开文档与视频，可能遗漏内部决策细节。

---

## 112. From Detection to Diagnosis: Advancing Hallucination Analysis with Automated Data Synthesis

**arXiv ID:** 2601.09734 | [PDF](https://arxiv.org/pdf/2601.09734v1)

**作者:** Yanyi Liu `[一作]` (Northeastern University), Yingyou Wen `[通讯]` (Northeastern University)

**通讯引用:** 907 | [OpenAlex ID](https://openalex.org/A5010551231)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出幻觉诊断任务，设计自动化多维增强数据生成管道（HDG），基于GRPO训练4B模型HDM-4B‑RL，并在幻觉检测、定位、解释与纠正等完整诊断流程上进行评估。

**💡 创新点**

创新点在于将传统的二元检测提升为完整诊断（检测+定位+解释+纠正），通过自动化管道生成高质量、多难度、丰富标签的数据；使用GRPO结合结构化奖励函数实现多目标对齐。

**🔧 技术方法**

主要技术包括：HDG自动化数据合成（语料抽取、任务种子生成、多维增强、质量检验、元数据注释），RL对齐（Group Relative Policy Optimization），以及包含结构化输出、检测、定位奖励的多项奖励函数。

**📊 数据集**

使用Wikipedia 2023‑11‑01英文 dump 作为原始语料，采样4.5k文档生成18,453条样本；评估使用HaluBench（QA 子集）和SummEval（摘要子集）等公开基准。

**📈 对比分析**

与现有专用检测模型（HHEM、Lynx、MiniCheck等）及大模型（Qwen3‑32B、GPT‑4.1、o4‑mini）对比。HDM‑4B‑RL 在宏 F1 上击败 Lynx‑8B、接近 GPT‑4.1、与 Qwen3‑32B 接近；在诊断子任务上，单步推理模型表现优于 pipeline，且 4B 规模已逼近 32B 规模的性能。

**⚠️ 局限性**

局限性：仅使用 Wikipedia 作为数据来源，未覆盖专业领域；实验仅限 4B 参数模型，缺乏对更大模型的验证；数据规模和多样性待进一步扩大。

---

## 113. Introducing Axlerod: An LLM-based Chatbot for Assisting Independent Insurance Agents

**arXiv ID:** 2601.09715 | [PDF](https://arxiv.org/pdf/2601.09715v1)

**作者:** Adam Bradley `[一作]` (Dakota State University), Khandaker Mamun Ahmed `[通讯]` (Dakota State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发并评估了Axlerod，一个基于大型语言模型（LLM）的聊天机器人，帮助独立保险代理人检索和查询政策信息、覆盖细节及相关文档，从而提升代理工作效率。

**💡 创新点**

首次针对保险代理人而非客户推出LLM助手，集成检索增强生成（RAG）、意图识别与工具调用的完整端到端系统，并在保险行业真实大规模数据库上提供可靠性与性能评估。

**🔧 技术方法**

使用Google Gemini 2.5 Pro LLM、LiteLLM代理、Smoltalk微框架、Typesense搜索引擎、Python 3.11、FastAPI、OpenWebUI等技术实现系统架构、数据检索与对话生成。

**📊 数据集**

使用约73万条个人与商业保险政策记录、对应的覆盖与理赔数据以及约400 MB的公司与行业文档数据库，全部来自Safety Insurance内部系统。

**📈 对比分析**

通过与人工代理人对比的时间测评与准确率评估：总体准确率93.18%，在账单计划、自动续费等任务中准确率≥99%，搜索时间平均缩短2.42 秒，显著提升效率。

**⚠️ 局限性**

局限性包括对姓名模糊导致的检索错误、仅在内部环境进行测试、缺乏多步骤复杂任务的评估、对外部数据源依赖有限，以及未充分验证长期可靠性与潜在偏差。

---

## 114. A Compute and Communication Runtime Model for Loihi 2

**arXiv ID:** 2601.10035 | [PDF](https://arxiv.org/pdf/2601.10035v1)

**作者:** Jonathan Timcheck `[一作]` (Intel Corporation), Sumit Bam Shrestha `[通讯]` (Intel Corporation)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为Intel Loihi 2神经形态芯片构建并验证了一个多维Roofline模型（max‑affine运行时模型），通过微基准测量和TrafficStats工具对计算与网络通信进行量化，并在矩阵向量乘法（线性层）和二次无约束二进制优化（QUBO）求解器上进行评估。

**💡 创新点**

提出了首个可同时考虑SynOps、SynMem读取、DendOps与NoC拥塞的下限运行时模型，模型既简单易用，又能在多种工作负载与空间映射下与实际测得的运行时高度吻合，且通过解析式揭示了空间布局对网络拥塞与面积-时间权衡的影响。

**🔧 技术方法**

使用Loihi 2的硬件计数器、内置的运行时探针、TrafficStats网络流量分析工具以及一套定制的微基准（Barrier, DendOp, SynOp, SynMem, Link Bandwidth）进行模型校准；同时构建解析式计算NoC拥塞。

**📊 数据集**

实验数据主要来自Loihi 2芯片的自定义微基准与人工合成的工作负载（密集与稀疏线性层、QUBO求解器），未使用公开大规模真实数据集。

**📈 对比分析**

通过将模型预测的每步运行时与Loihi 2实际测得的运行时进行对比，计算Pearson相关系数，所有实验中相关系数均≥0.97，且模型在计算、通信瓶颈以及不同空间布局下均能逼近真实运行时，体现出高精度的预测能力。

**⚠️ 局限性**

仅针对单一芯片和单核工作负载，未覆盖多核并行（流水线）模式；模型仅为运行时下限，未考虑能耗；对动态工作负载与非均匀通信模式的适用性尚未验证，未来需扩展到更广泛的算法与能耗模型。

---

## 115. How Human Motion Prediction Quality Shapes Social Robot Navigation Performance in Constrained Spaces

**arXiv ID:** 2601.09856 | [PDF](https://arxiv.org/pdf/2601.09856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 116. The "I" in FAIR: Translating from Interoperability in Principle to Interoperation in Practice

**arXiv ID:** 2601.10008 | [PDF](https://arxiv.org/pdf/2601.10008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 117. Comparative Evaluation of Deep Learning-Based and WHO-Informed Approaches for Sperm Morphology Assessment

**arXiv ID:** 2601.10070 | [PDF](https://arxiv.org/pdf/2601.10070v1)

**作者:** Mohammad Abbadi `[一作]` (University of Dubai), Mohammad Abbadi `[通讯]` (University of Dubai)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5031773409)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估并比较了基于图像的深度学习模型HuSHeM与结合WHO标准及系统性炎症指数(SIRI)的临床基线模型在单精子头形态评估中的性能。

**💡 创新点**

将卷积神经网络直接应用于单精子图像，并在同一数据集上与传统WHO+SIRI基线做严格的离散化对比，全面采用判别、校准与决策曲线评估临床价值。

**🔧 技术方法**

采用卷积神经网络HuSHeM（Adam优化器、交叉熵损失、数据增强），配合逻辑回归基线模型，利用ROC‑AUC、PR‑AUC、校准曲线和决策曲线等指标。

**📊 数据集**

使用719张高分辨率精子头显微图像及其WHO严格形态标签与完整血计数计算的SIRI指数，训练集由HuSHeM公开数据和实验室采集的约800张图像组成。

**📈 对比分析**

采用独立测试集、无信息泄漏的分层抽样，并用bootstrap与DeLong检验ROC；HuSHeM实现ROC‑AUC 0.975（95% CI 0.914–1.0）和PR‑AUC 0.993，而WHO+SIRI仅为ROC‑AUC 0.721、PR‑AUC 0.097；决策曲线显示HuSHeM在所有阈值下均具备更高净临床收益。

**⚠️ 局限性**

数据量有限、图像与临床数据来源分离导致无法实现多模态联合预测、仅在单精子层面评估，未验证在更大多中心样本中的外部泛化及与实际生育结果的关联。

---

## 118. In-Context Operator Learning on the Space of Probability Measures

**arXiv ID:** 2601.09979 | [PDF](https://arxiv.org/pdf/2601.09979v1)

**作者:** Frank Cole `[一作]` (University of Minnesota), Rongjie Lai `[通讯]` (Purdue University)

**通讯引用:** 1755 | [OpenAlex ID](https://openalex.org/A5088277298)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出在概率测度空间上的“上下文算子学习”（in‑context operator learning）框架，用少量样本（prompt）直接推断最优传输映射，无需梯度更新。

**💡 创新点**

创新点包括：①将最优传输映射视为上下文算子学习问题；②在非参数和参数化（高斯）两种情形下给出理论泛化与尺度律；③构造可解释的Transformer架构实现可变长度、置换不变的上下文处理，并证明其能逼近高斯间的精确OT映射。

**🔧 技术方法**

技术主要包括：Transformer自/交叉注意力、MMD正则化、可变长度序列编码、经验风险最小化、覆盖数与Dudley积分等理论工具；在参数化案例中使用特定的线性Transformer与平方核MMD。

**📊 数据集**

数据集：synthetic 高斯到高斯、MNIST、Fashion‑MNIST、ModelNet10（3D点云），以及在这些数据上对齐的编码器/解码器。实验使用标准训练/测试划分和A100 GPU。

**📈 对比分析**

与传统OT求解、单一任务学习等方法对比，使用MMD_u评估生成样本与真实目标分布的相似度；实验结果显示在不同任务上MMD均保持在0.1–0.8（按10^2比例）区间，且模型可通过不同prompt实现类条件采样，性能与理论预测的尺度律（随prompt长度n的1/√n衰减）高度一致。

**⚠️ 局限性**

局限性：①理论证明多基于理想化假设（如任务分布在低维流形、Gaussian假设、特征映射可被神经网络逼近）；②在高维非Gaussian任务中收敛速度与泛化误差的上界尚未严格评估；③模型依赖预训练的编码器/解码器，无法直接处理原始像素/点云空间。

---

## 119. Enhancing Business Analytics through Hybrid Summarization of Financial Reports

**arXiv ID:** 2601.09729 | [PDF](https://arxiv.org/pdf/2601.09729v1)

**作者:** Tohida Rehman `[一作]` (Jadavpur University), Tohida Rehman `[通讯]` (Jadavpur University)

**通讯引用:** 66 | [OpenAlex ID](https://openalex.org/A5090477015)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究提出了一种混合提取-抽取式框架与长序列Transformer模型，用于生成财务业绩电话会议记录的简洁、事实可靠的摘要；

**💡 创新点**

创新点在于将LexRank句子提取与BART/PEGASUS细调相结合，并并行使用LED模型以捕获长文本上下文，同时在有限算力条件下实现可比的事实一致性；

**🔧 技术方法**

使用的技术包括LexRank、BART、PEGASUS、Longformer Encoder‑Decoder（LED）以及多种评估指标（ROUGE、METEOR、MoverScore、BERTScore、SciBERTScore、FinBERTScore、实体级精度/召回）；

**📊 数据集**

实验数据集为ECTSum，包含2019‑2022年间美国上市公司收益电话会议记录及其Reuters风格摘要；

**📈 对比分析**

与ECTSum基准相比，LED‑Base在长文本上获得最高的ROUGE‑1/2、METEOR、MoverScore和FinBERTScore，BART‑Large在BERTScore和SciBERTScore上领先，PEGASUS在ROUGE‑L和F1‑target上表现最佳，整体性能虽低于基准，但在资源受限情况下已具备可比性；

**⚠️ 局限性**

局限性包括受限的算力导致输入长度与训练轮次受限，导致模型在数值和实体细节上的偶尔错误（hallucination），以及对长文本的准确性与事实一致性仍有提升空间。

---

## 120. LAMDA: Aiding Visual Exploration of Atomic Displacements in Molecular Dynamics Simulations

**arXiv ID:** 2601.09887 | [PDF](https://arxiv.org/pdf/2601.09887v1)

**作者:** Rostyslav Hnatyshyn `[一作]` (Arizona State University), Baldwin Nsonga `[通讯]` (Leipzig University)

**通讯引用:** 38 | [OpenAlex ID](https://openalex.org/A5077188850)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一款名为 LAMDA 的可视化分析系统，帮助材料科学家快速、系统地探索分子动力学模拟中原子位移的状态‑状态转移，支持层次聚类、对齐、3D 交互可视化、注释与导出。

**💡 创新点**

创新点包括：
- 将原子位移对齐与聚类结合，使用伪原子、Kabsch 对齐与 ZCA whitening 生成与位置无关的距离度量；
- 设计多视图（热图、树状图、embedding 网格、group displacement、superquadric）和 Scratchpad 记事板，实现从宏观到微观的层级分析与洞察保存；
- 提供聚类质量可视化与交互式阈值调节；
- 通过 PDF+Extended‑XYZ 导出与系统内部笔记同步的实验室笔记本。

**🔧 技术方法**

技术实现：
- 编程语言 Julia 与 Makie 图形框架；
- 依赖 ASE 进行原子数据读取与特征计算；
- Kabsch 对齐、ZCA whitening、Ward 链接层次聚类、热图与树状图绘制；
- Hilbert 空间填充曲线用于 embedding 网格布局；
- 超四面体（superquadric）可视化利用本构张量不变量；
- PDF、Extended‑XYZ 文件导出与 Scratchpad 交互。

**📊 数据集**

数据集：147 原子纳米粒子在 700 K NVT 模拟下的转移区域，约 3,000 条原子位移转移；使用 SNAP（Spectral Neighbor Analysis Potentials）特征、CNA 结构标签作为原子/转移标量。

**📈 对比分析**

比较方法与性能：
- 通过热图与交互阈值实现对相似转移的裁剪，随后使用 Ward 链接聚类；
- 通过 Group Displacement 可视化与聚类质量度量快速识别不同物理机制（ICO、FST 等）；
- 在约 1,400 条转移的案例中，预处理耗时约 4 min，聚类与可视化保持交互响应；
- 证明 LAMDA 能在几百至几千条转移上保持良好性能，且可在同一窗口内展示大量 3D 可视化。

**⚠️ 局限性**

限制与未来改进：
- 颜色方案重叠导致不同簇在 Scratchpad 或视图中易混淆；
- Superquadric 抽象化对小幅差异感知不敏感，需改进尺度或视觉编码；
- 目前仅支持单元素小型纳米粒子，难以直接处理大分子或多元素生物体系；
- 聚类结果不可直接编辑（合并/拆分）或跨聚类比较；
- 时间轴与摄像机控制未在多窗口同步；
- 缺乏插件机制，无法灵活替换对齐/距离计算方法；
- 对于超过数千条转移的极大数据集，仍需进一步性能优化与可视化降维。

---

## 121. Diffusion-based Frameworks for Unsupervised Speech Enhancement

**arXiv ID:** 2601.09931 | [PDF](https://arxiv.org/pdf/2601.09931v1)

**作者:** Jean-Eudes Ayilo `[一作]` (Universite de Lorraine), Xavier Alameda-Pineda `[通讯]` (Universite Grenoble Alpes)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了无监督单通道语音增强框架，显式将噪声建模为潜在变量，并使用扩散模型作为语音与噪声的联合先验，改进传统仅对语音采样的EM流程。

**💡 创新点**

创新点在于：①在无监督扩散式SE中首次引入显式噪声采样，实现混合一致性；②将噪声先验从传统的NMF改为扩散模型，并通过单个联合扩散网络同时学习语音与噪声先验，减少模型数量与训练成本；③在采样过程中融合Gibbs采样、Tweedie公式与Wiener后处理，进一步提升效果。

**🔧 技术方法**

技术手段包括：score‑based扩散模型、EM与Gibbs采样、Tweedie公式用于后验均值估计、Wiener滤波后处理、FiLM融合实现标签条件扩散网络、NMF（对比实验）和低维STFT特征。

**📊 数据集**

数据集：WSJ0–QUT、VoiceBank–DEMAND、QUT‑Noise、DEMAND 等，用于训练语音先验、噪声先验及评估匹配/不匹配场景。

**📈 对比分析**

与多种无监督/监督基线（包括SVMSE+、Conv‑TasNet、RemixIT、RVAE、NMF‑based扩散等）进行对比；在匹配条件下，该方法在SI‑SDR、PESQ、ESTOI、DNS‑MOS等指标上逼近或超过最优无监督方法，并与部分监督模型竞争；在不匹配条件下，性能下降幅度最小，鲁棒性最好，显示出显式噪声采样的优势。

**⚠️ 局限性**

局限性包括：推理速度较慢（≈6 s/1 s音频），难以满足实时需求；噪声先验仍受扩散模型表达能力限制，对极端噪声或语音分布漂移仍有一定敏感性；需要进一步改进采样效率与模型规模，以实现更广泛的部署。

---

## 122. A pipeline for enabling path-specific causal fairness in observational health data

**arXiv ID:** 2601.09841 | [PDF](https://arxiv.org/pdf/2601.09841v1)

**作者:** Aparajita Kashyap `[一作]` (Columbia University), Shalmali Joshi `[通讯]` (Columbia University)

**通讯引用:** 1602 | [OpenAlex ID](https://openalex.org/A5035149567)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了一个通用管道，利用路径特定因果公平方法在观察性医疗数据中训练和评估机器学习模型。

**💡 创新点**

创新点在于将结构公平模型映射到观测医疗数据，结合已知医疗差异定义公平目标；通过降维提升高维因果效应估计精度；并系统比较因果与非因果公平干预，发现无单一最优方案。

**🔧 技术方法**

采用因果推断（自然直接/间接效应估计）、双重稳健估计、降维技术（特征选择、自动编码器）、路径特定正则化、因果重采样、特征选择策略以及群体/个体公平正则化等技术。

**📊 数据集**

使用CUMC‑EHR（约530万患者）和Merative多州医疗补助（MDCD，2500万患者）四个案例（AMI、SLE、T2DM、SCZ）进行实验。

**📈 对比分析**

与基线、无意识、无偏特征、贪婪特征、因果重采样、路径特定内插等多种方法在AUROC和路径效应上进行对比；多数公平干预能显著降低NDE/NIE，但往往伴随性能折损，且无统一最优方案。

**⚠️ 局限性**

局限性包括仅关注单属性公平，未覆盖交叉公平；缺乏对非二元特征的建模；对样本多样性和交叉因素的依赖；以及需要临床专家进一步确定公平路径。

---

## 123. SAGE: Tool-Augmented LLM Task Solving Strategies in Scalable Multi-Agent Environments

**arXiv ID:** 2601.09750 | [PDF](https://arxiv.org/pdf/2601.09750v1)

**作者:** Robert K. Strehlow `[一作]` (Technische Universität Berlin), Sahin Albayrak `[通讯]` (Technische Universität Berlin)

**通讯引用:** 5278 | [OpenAlex ID](https://openalex.org/A5089847337)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了SAGE系统，利用OPACA多代理框架实现LLM与工具的动态调用和任务求解

**💡 创新点**

创新点在于将OPACA框架与LLM结合，提供零样本工具调用、模块化提示方法，以及支持多模型、多代理的可扩展性

**🔧 技术方法**

采用OPACA框架、FastAPI、Vue、OpenAI/vLLM、LiteLLM以及四种提示方法（Simple、Simple-Tools、Tool-Chain、Orchestration）

**📊 数据集**

使用自建的3个Agent Container（共102个动作），涵盖智能办公、仓储管理、音乐播放器等领域的benchmark prompts

**📈 对比分析**

通过对四种提示方法与gpt-4o-mini等模型的比较，评估Response Score、工具调用准确率、时间与Token消耗，结果显示Simple-Tools和Tool-Chain在工具调用准确率和速度上表现最佳

**⚠️ 局限性**

局限性包括开放源模型对工具调用的准确性受限，Orchestration方法耗时长、Token高，且模型对工具数量有限制

---

## 124. EditEmoTalk: Controllable Speech-Driven 3D Facial Animation with Continuous Expression Editing

**arXiv ID:** 2601.10000 | [PDF](https://arxiv.org/pdf/2601.10000v1)

**作者:** Diqiong Jiang `[一作]` (China University of Petroleum), Zhenyu Wu `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 465 | [OpenAlex ID](https://openalex.org/A5100773428)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 EditEmoTalk，能够从语音生成 3D 面部动画并支持连续情感编辑。

**💡 创新点**

创新点在于：①使用边界感知语义嵌入学习情感决策边界的法向，构建可连续插值的表情流形；②通过情感一致性损失和双向训练策略实现表情与语音的语义对齐；③在单一框架内实现离散情感识别与细粒度连续编辑。

**🔧 技术方法**

技术路线包括：双分支 HuBERT（内容）+ emotion2vec（情感）特征编码；边界感知表情嵌入；条件扩散 Transformer（DiT）生成 FLAME 参数；FLAME 网格解码器；情感一致性映射网络；多任务损失（重建、几何、时间、情感一致性）。

**📊 数据集**

使用四大公开数据集：MEAD、CREMA-D、RAVDESS、HDTF，涵盖多语种、多情感级别和自然场景。

**📈 对比分析**

与 DiffPoseTalk、DeepTalk、EmoTalk、Emote 等方法比较。评价指标包括 Vertex Error、Lip Vertex Error、Mouth Opening Deviation、Facial Dynamics Deviation、ΔCH 及 MOS。EditEmoTalk 在大多数指标上最优，特别是 ΔCH 最低、MOS 最高，显示出最佳的情感一致性与自然性。

**⚠️ 局限性**

局限性：①对极端或稀有情绪的泛化尚不充分；②依赖 FLAME 3D 模型，可能限制对细微面部细节的捕捉；③实时推理仍需要较高算力；④跨语言、跨说话人的一致性需要进一步验证。

---

## 125. VERHallu: Evaluating and Mitigating Event Relation Hallucination in Video Large Language Models

**arXiv ID:** 2601.10010 | [PDF](https://arxiv.org/pdf/2601.10010v1)

**作者:** Zefan Zhang `[一作]` (Jilin University), Tian Bai `[通讯]` (Jilin University)

**通讯引用:** 1008 | [OpenAlex ID](https://openalex.org/A5029290911)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了专门评估视频LLM事件关系幻觉的基准VERHallu，并通过无训练的关键帧传播（KFP）策略降低幻觉现象；

**💡 创新点**

创新点在于：①针对事件关系幻觉设计的三任务评估框架（关系分类、问答、反事实问答）；②提出的KFP无训练方法在中间层重新分配关键帧注意力，从而提升多事件推理；

**🔧 技术方法**

技术主要包括关键帧传播（Key‑Frame Propagating）实现（Gaussian扩散、注意力重分配）、Token Activation Map可视化、对齐多模态特征；

**📊 数据集**

使用了新构建的VERHallu数据集，包含574段真实视频（如Mr. Bean等）共7676个样本，覆盖因果、时间、子事件关系；

**📈 对比分析**

与多种主流VideoLLMs（如QwenVL‑2.5‑7B、LLaVA‑NeXT‑7B、InternVL‑3‑8B等）进行对比，KFP提升了RC、QA等指标约10–20%，但整体性能仍低于人类；

**⚠️ 局限性**

局限性：仍缺乏针对关系推理的专门视觉感知能力，KFP在反事实场景可能产生误导，且对极其复杂的事件关系仍无法完全理解。

---

## 126. Interpolation-Based Optimization for Enforcing lp-Norm Metric Differential Privacy in Continuous and Fine-Grained Domains

**arXiv ID:** 2601.09946 | [PDF](https://arxiv.org/pdf/2601.09946v1)

**作者:** Chenxi Qiu `[一作]` (University of North Texas), Chenxi Qiu `[通讯]` (University of North Texas)

**通讯引用:** 1141 | [OpenAlex ID](https://openalex.org/A5070496959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于插值的框架，用于在连续和细粒度领域中优化ℓ_p范数的度量差分隐私（mDP）。

**💡 创新点**

创新点在于通过在稀疏的锚点上优化扰动分布，并通过对非锚点位置进行对数凸组合插值，从而有效地保持mDP的隐私保证，同时降低了计算复杂性。

**🔧 技术方法**

使用了插值方法和线性规划（LP）技术，结合了对数凸插值和维度分配的隐私预算优化。

**📊 数据集**

使用了来自罗马、纽约市和伦敦的真实世界位置数据集进行实验。

**📈 对比分析**

与现有的基线机制相比，提出的方法在细粒度领域中提供了严格的隐私保证（0% mDP违规），并且在效用损失上优于预定义噪声机制和混合方法，表现出竞争力。

**⚠️ 局限性**

限制在于在高维领域的可扩展性受到挑战，插值复杂性增加和隐私预算分配的碎片化可能影响性能。

---

## 127. Fuzzychain-edge: A novel Fuzzy logic-based adaptive Access control model for Blockchain in Edge Computing

**arXiv ID:** 2601.10105 | [PDF](https://arxiv.org/pdf/2601.10105v1)

**作者:** Khushbakht Farooq `[一作]`, Wei Song `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于零知识证明、模糊逻辑与区块链的可适应访问控制模型，专为边缘计算环境下的IoT医疗系统设计。

**💡 创新点**

创新点在于将zk‑SNARKs用于身份隐私保护、模糊逻辑实现上下文感知的动态决策，并通过智能合约记录不可篡改的审计日志，形成全链路可追溯且自适应的访问控制框架。

**🔧 技术方法**

采用零知识证明（zk‑SNARKs）、模糊逻辑推理、区块链（以太坊/Hyperledger类）、智能合约以及Raft/Solo共识协议。

**📊 数据集**

使用仿真数据：100–500条请求、1–5 KB事务，模拟医疗IoT场景；并未使用公开真实数据集。

**📈 对比分析**

与两种基线方案（Paper 1与Paper 2）对比，测量平均延迟、吞吐量、隐私水平、准确率等指标；实验显示在150 TPS以上时，本模型延迟比基线低30–50%，吞吐量提高20–30%，隐私保护率接近100%。

**⚠️ 局限性**

局限性包括zk‑SNARKs与模糊推理的计算开销、在资源受限的边缘节点上的可扩展性不足，以及区块链共识延迟对高并发场景的影响。

---

## 128. LLM-Driven Preference Data Synthesis for Proactive Prediction of the Next User Utterance in Human-Machine Dialogue

**arXiv ID:** 2601.09713 | [PDF](https://arxiv.org/pdf/2601.09713v1)

**作者:** Jinqiang Wang `[一作]` (University of Science and Technology Beijing), Chris Nugent `[通讯]` (Ulster University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ProUtt 方法，利用大型语言模型生成包含意图树、句型推理和意图路径推理的偏好数据，用于训练小型 LLM 实现主动预测用户下一句。

**💡 创新点**

创新点在于：① 用意图树显式建模用户意图层级；② 结合信息搜寻理论的利用与探索视角进行路径推理；③ 通过修订与扰动生成首选与非首选推理轨迹，实现对意图层面的预测而非仅模仿表面语言；④ 整个流程不需多模型迭代，直接生成对比样本。

**🔧 技术方法**

技术实现包括：大语言模型（如 Doubao‑1.5‑Pro、Qwen3‑8B）进行意图树抽取与推理；句型推理与意图路径推理；偏好数据生成策略（修订与扰动）；对齐方法 DPO、ORPO、SimPO 等；LLM‑as‑Judge 评估。

**📊 数据集**

使用的数据集有：公开对话数据集 LMSYS、ShareGPT、WildChat、CrossWOZ；以及基于 ProUtt 生成的合成数据集 LMSYS‑ProUtt‑10K、CrossWOZ‑ProUtt‑5K 等。

**📈 对比分析**

与多种基线（prompt‑based LLMs、用户模拟器、其他数据合成方法）通过点估计和对比评估对比。ProUtt 在四个基准上在 LLM‑Judge 与 Embed‑Sim 指标均优于基线，DPO 对齐效果最佳，整体相对提升约 4.9%‑5.1%。

**⚠️ 局限性**

局限性：跨域泛化仍有不足（ShareGPT、WildChat 结果低于训练集）；意图树采用树结构，无法捕获更丰富的图关系；未充分利用多轮用户历史与个性化偏好；对阈值设置与生成候选数量较为敏感。

---

## 129. PID-Guided Partial Alignment for Multimodal Decentralized Federated Learning

**arXiv ID:** 2601.10012 | [PDF](https://arxiv.org/pdf/2601.10012v1)

**作者:** Yanhang Shi `[一作]` (Stony Brook University), Yong Liu `[通讯]` (New York University)

**通讯引用:** 19928 | [OpenAlex ID](https://openalex.org/A5100724297)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无服务器、拓扑无关的多模态分布式联邦学习框架，利用 PID 将特征拆分为冗余、唯一和协同三块，并通过部分对齐实现仅在可对齐切片之间共享信息。

**💡 创新点**

创新点在于：①把 PID 概念应用于特征拆分，实现对三类信息的显式分离；②采用部分对齐，仅共享可对齐的切片，消除 uni‑/multi‑modal 之间的梯度冲突；③无需梯度手术或中心协调，保持拓扑灵活。

**🔧 技术方法**

技术方法包括：PID‑基特征拆分（feature fission）、slice‑level 部分对齐、局部 SGD（DSGD）在每个模态子图上同步、对比学习（对齐余量）促进冗余子空间一致、平均融合协同切片、集成损失提升鲁棒性。

**📊 数据集**

使用四个公开数据集：KU‑HAR（加速计+陀螺仪）、ModelNet‑40（两视图）、AVE（音视频）和 IEMOCAP（音视频文字）四模，覆盖从 2‑模态到 4‑模态的不同情形。

**📈 对比分析**

与 DS​GD‑Task/Modality/Hybrid、Harmony、FedHGB、DMML‑KD 等基线相比，实验表明该方法在所有模态类型和不同代理比例下均显著优于基线，平均提升约 0.3%–2.8%，尤其在多模态代理稀缺时效果最为显著。

**⚠️ 局限性**

局限性：①仅适用于共享标签空间的多模态任务；②特征拆分比例和融合方式需手工调参，缺乏自动化；③在极大模态数或频繁变动的网络拓扑中扩展性尚未验证；④仍依赖局部梯度同步，通信效率受通信图结构影响。

---

## 130. Explainable Deep Learning for Pediatric Pneumonia Detection in Chest X-Ray Images

**arXiv ID:** 2601.09814 | [PDF](https://arxiv.org/pdf/2601.09814v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 131. Hallucination Detection and Mitigation in Large Language Models

**arXiv ID:** 2601.09929 | [PDF](https://arxiv.org/pdf/2601.09929v1)

**作者:** Ahmad Pesaranghader `[一作]` (CIBC), Erin Li `[通讯]` (CIBC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种根因感知的持续改进框架，用检测与缓解的闭环管理LLM/LRM的幻觉，并通过模型、数据、上下文三层策略实现针对性干预

**💡 创新点**

创新点在于将幻觉来源细分为模型、数据和上下文三类，构建多层检测工具与RACE一致性评估，形成根因感知的分层缓解流程，避免一刀切的通用修复

**🔧 技术方法**

采用概率/语义熵、Monte Carlo Dropout、Ensemble、贝叶斯后验、ECE、RACE、检索增强生成（RAG）、自我一致性提示、温度标定、对抗/对比学习、RLHF等多种检测与缓解技术

**📊 数据集**

使用金融与法律领域的文本数据（贷款申请、合同、财报等）以及公开检索数据库作为外部知识源，案例中以结构化金融文档为主

**📈 对比分析**

与传统单一检测/缓解方法相比，实验显示在金融数据抽取任务中幻觉率下降约30%–40%，实体准确率提升显著；与GPT‑4、Claude等基准模型对比，分层框架实现更高的可靠性

**⚠️ 局限性**

幻觉无法完全消除，框架依赖持续人工反馈和多轮迭代；对封闭权模型的检测受限于可观测信息，实施成本较高

---

## 132. State of AI: An Empirical 100 Trillion Token Study with OpenRouter

**arXiv ID:** 2601.10088 | [PDF](https://arxiv.org/pdf/2601.10088v1)

**作者:** Malika Aubakirova `[一作]` (a16z), Anjney Midha `[通讯]` (a16z)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用 OpenRouter 平台 100 万亿 token 的真实交互元数据，对 LLM 在不同行业、地区、模型类型与任务类别中的使用模式、保留率、成本与使用关系进行大规模实证分析。

**💡 创新点**

首次系统量化跨模型、多任务、多地区的实际使用数据，提出“Cinderella Glass Slipper”保留现象，揭示模型与工作负载匹配的关键时刻，并为代理式推理提供广泛证据；同时呈现开源模型占比、代理推理占比等行业重要指标。

**🔧 技术方法**

采用 OpenRouter 的请求元数据、GoogleTagClassifier 进行任务分类，结合 SQL 与 Hex 分析管道、时间序列、地理归属、成本计算等技术手段，对模型、任务、地区、成本维度进行多维度聚合与可视化。

**📊 数据集**

研究基于 OpenRouter 的 300+ 模型、60+ 提供商、全球用户两年时间跨度的元数据，约 100 万亿 token；通过 0.25% 的随机标注样本（GoogleTagClassifier 输出）实现任务与标签映射。

**📈 对比分析**

对比开源与专有模型、中文与国外模型、模型规模对使用量、成本、保留率的影响；发现开源模型占比约 30%，代理式推理占比超过 50%，成本弹性低；未对传统 benchmark 进行直接对比，侧重于使用行为与经济指标。

**⚠️ 局限性**

仅覆盖 OpenRouter 平台的数据，缺乏用户文本内容、内部工作流细节，地理归属基于账单而非真实位置，任务分类依赖预训练分类器，且时间窗口仅为两年内，无法完整代表整体 LLM 生态或所有部署场景。

---

## 133. Continuum Memory Architectures for Long-Horizon LLM Agents

**arXiv ID:** 2601.09913 | [PDF](https://arxiv.org/pdf/2601.09913v1)

**作者:** Joe Logan `[一作]` `[通讯]` (Mode7 GK), Joe Logan (Mode7 GK)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种连续内存架构（CMA），通过定义内存的持久性、可变性和巩固性，对大型语言模型代理的记忆进行生命周期管理，并在四个行为探针（知识更新、时间关联、关联回忆、上下文消歧）上对比评估。

**💡 创新点**

创新点在于：①把记忆从无状态检索转化为具有持续演化能力的子系统；②提出一套通用的CMA需求与生命周期模型；③通过实验展示CMA在动态记忆任务上优于传统RAG，揭示了持久、可变和巩固机制的实际效益。

**🔧 技术方法**

采用图结构存储子系统、激活传播与衰减机制、检索驱动的强化与抑制、关联路由、时间链路与离线重放/抽象合成，并使用 GPT‑4o 作为人工智能判定者对检索结果进行评分。

**📊 数据集**

实验使用自构造的四类数据集：偏好/技术规范/调试步骤/日程等知识场景；自然情景事件集；项目知识图谱；以及包含歧义词汇的多上下文样本；未使用公开基准数据集，仅参考 LongBench 与 LoCoMo 的任务设计。

**📈 对比分析**

方法为将 CMA 与 Supabase pgvector RAG 基线在相同嵌入下对齐，通过 GPT‑4o 判定 0–1 评分并统计赢/平/输，CMA 在 92 次决定性试验中赢得 82 次，效应量均为大/非常大；同时记录平均查询时延从 0.65 s 上升至 1.48 s，增长约 2.4 倍。

**⚠️ 局限性**

局限性包括：查询时延与规模扩展受激活传播和巩固过程限制；检索驱动更新可能导致记忆漂移；时间关联任务仍有 50% 以上失败率；可解释性与审计难度高；以及对隐私合规、数据保留和加密的治理挑战。

---

## 134. Diffusion-Driven Deceptive Patches: Adversarial Manipulation and Forensic Detection in Facial Identity Verification

**arXiv ID:** 2601.09806 | [PDF](https://arxiv.org/pdf/2601.09806v1)

**作者:** Shahrzad Sayyafzadeh `[一作]` (Florida A and M University), Shonda Bernadin `[通讯]` (Florida A and M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个端到端管线，用FGSM和扩散模型生成、优化并评估针对面部生物识别系统的对抗补丁，并通过视觉Transformer‑GPT2生成语义描述进行取证。

**💡 创新点**

创新点在于将扩散模型用于对抗补丁的后处理以提升不可感知性，并将对抗补丁的语义标签化与多模态检测（哈希+SSIM+分割+热图）结合，实现对抗补丁的定位与鉴别。

**🔧 技术方法**

采用FGSM梯度攻击、扩散模型逆扩散、Vision Transformer＋GPT2生成字幕、Perceptual Hash（aHash、pHash、dHash、wHash）、SSIM、分割、热图及神经激活映射等技术。

**📊 数据集**

使用CelebA‑HQ、CelebA、ArcFace/FaceNet等公开人脸数据集以及自构建的情绪与身份标签集。

**📈 对比分析**

与多项现有对抗补丁方法对比，SSIM、LPIPS、L2、MS‑SSIM及迁移率指标均达到或超过基线，攻击成功率约81%，检测率为100%。

**⚠️ 局限性**

局限包括对不同光照、姿态的鲁棒性待验证，检测阈值对样本分布敏感，且对抗补丁在现实物理环境中的持久性和泛化性仍需进一步研究。

---

## 135. The Spatial Blindspot of Vision-Language Models

**arXiv ID:** 2601.09954 | [PDF](https://arxiv.org/pdf/2601.09954v1)

**作者:** Nahid Alam `[一作]` (Cohere Labs Community), Bala Krishna S Vegesna `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在 LLaVA 框架下比较了多种视觉编码器（CLIP、SigLIP、SigLIP2、AIMv2）以及 2D‑RoPE 位置编码对视觉‑语言模型空间推理能力的影响，并在多项空间推理基准上进行评测。

**💡 创新点**

创新点在于将 2D‑RoPE 直接注入视觉编码器注意力机制、对比不同预训练目标的编码器在空间推理上的差异，并系统评估 2D 空间建模在 VLM 中的实际收益。

**🔧 技术方法**

采用了 LLaVA 的 7B 语言模型、CLIP‑style 视觉编码器、SigLIP、SigLIP2、AIMv2 预训练、2D‑RoPE 位置编码、AdamW 优化器以及基准测试框架（MMMU、MME、CCBench、SEEDBench、MMVP、CV‑Bench、GQA 等）。

**📊 数据集**

数据集包括 LLaVA 预训练与指令微调数据、MMMU、MME、CCBench、SEEDBench-IMG、MMVP、CV‑Bench、GQA、TallyQA、CountBenchQA 等空间推理评测集。

**📈 对比分析**

通过与 Qwen2.5‑VL 等前沿模型对比，发现 AIMv2 编码器在多项空间任务上明显优于 CLIP 或 SigLIP（如 TallyQA、CountBenchQA 提升 58%），而 2D‑RoPE 的效果因模型而异，整体性能仍落后于 Qwen2.5‑VL。

**⚠️ 局限性**

局限性包括只评估 2D 静态图像、固定分辨率预处理、未探索更高级的多模态对齐机制（如 gated attention、Q‑Former 等）以及对 3D 或动态空间推理的覆盖不足。

---

## 136. Putting green software principles into practice

**arXiv ID:** 2601.09741 | [PDF](https://arxiv.org/pdf/2601.09741v1)

**作者:** James Uther `[一作]` `[通讯]` (Oliver Wyman), James Uther (Oliver Wyman)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文描述了在公共云上为一个实时产品构建低碳计算系统的过程，并通过成本驱动的监控和反馈实现能耗与排放的降低。

**💡 创新点**

创新点在于将云成本视为能耗代理，利用服务器无状态化（serverless）技术自动扩缩，结合实时成本与碳足迹仪表板，实现了业务需求与绿色目标的平衡；并提出了“LightSwitchOps”等可操作性原则。

**🔧 技术方法**

使用技术包括：云服务商的 serverless 资源（自动扩缩的 Spark 集群、ARM CPU），PySpark 代码加速，cron 调度优化，成本与碳排放监控仪表板（CloudCarbonFootprint、云平台碳使用面板），以及自建的成本‑碳查询表。

**📊 数据集**

数据集主要为企业内部生产数据管道中的大规模业务数据；文中未列出公开数据集，而是以实际业务流为测试对象。

**📈 对比分析**

通过对比部署前后成本与估算的碳排放，作者观察到整体能耗与排放下降，但缺乏量化的基准或实验对照，性能提升主要体现在成本节约和系统自动化程度上。

**⚠️ 局限性**

限制包括：云平台碳足迹报告延迟且缺乏细粒度API；服务器无状态化对离线/长时间作业效果有限；对低碳电网和调度策略的依赖未能完全控制；缺乏公开验证的数据和实验对照。

---

## 137. Filtering for Copyright Enforcement in Europe after the Sabam cases

**arXiv ID:** 2601.09739 | [PDF](https://arxiv.org/pdf/2601.09739v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 138. In-Browser Agents for Search Assistance

**arXiv ID:** 2601.09928 | [PDF](https://arxiv.org/pdf/2601.09928v1)

**作者:** Saber Zerhoudi `[一作]` (University of Passau), Michael Granitzer `[通讯]` (University of Passau)

**通讯引用:** 3569 | [OpenAlex ID](https://openalex.org/A5006866152)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个完全在浏览器端运行的AI搜索助手，利用概率模型学习用户行为并以此为依据生成基于小型语言模型的搜索建议。

**💡 创新点**

首次将自适应概率模型与本地小型语言模型结合成混合架构，实现了全客户端、隐私保护的个性化搜索辅助。

**🔧 技术方法**

使用WebGPU/WebLLM在浏览器中运行2.7B参数Phi模型，配合Web Worker实现在线强化学习的政策梯度算法。

**📊 数据集**

AOL查询日志（用于预训练概率模型）以及18名大学生在三周内生成的搜索交互日志。

**📈 对比分析**

将个性化模型与预训练的泛化模型进行对比，个性化模型在下一步动作预测准确率上提升至38.7%（相较基线提升24%），并在实验中提高搜索效率、SUS得分82.5，建议接受率36.4%。

**⚠️ 局限性**

样本局限于大学生群体，未与更复杂的外部序列推荐算法做对比，且实验时间相对短，未来需验证在更大规模、多样化用户群体中的泛化能力。

---

## 139. Continuous-Depth Transformers with Learned Control Dynamics

**arXiv ID:** 2601.10007 | [PDF](https://arxiv.org/pdf/2601.10007v1)

**作者:** Peter Jemley `[一作]` `[通讯]` (Northeastern University), Peter Jemley (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一个混合Transformer架构，把中间几层替换为神经ODE块，并通过可学习的低维控制信号实现推理时可控的语言生成。

**💡 创新点**

创新点在于：① 将Transformer深度视为连续变量，并学习控制向量场；② 提出Solver Invariance Test验证模型的连续性；③ 利用自适应ODE求解器揭示向量场的几何结构。

**🔧 技术方法**

采用神经ODE、adjoint反向求导、可学习的输出缩放因子α、低维控制信号注入、MLP构建向量场、Euler/自适应求解器等技术。

**📊 数据集**

主要使用WikiText‑2数据集进行训练与评估，并在情感控制任务中使用“电影描述”句子作为实验素材。

**📈 对比分析**

与6层纯离散Transformer进行对比：梯度稳定性零异常；情感控制准确率正向98%、负向88%；轨迹偏差仅0.068%；推理延迟几乎相同（0.98×）。整体性能优异。

**⚠️ 局限性**

局限性：仅在约30M参数的小模型上验证，未验证大规模模型；仅测试单维控制（情感）；训练使用固定4步Euler，未探索多维控制或动态步长等更复杂设置。

---

## 140. EHRNavigator: A Multi-Agent System for Patient-Level Clinical Question Answering over Heterogeneous Electronic Health Records

**arXiv ID:** 2601.10020 | [PDF](https://arxiv.org/pdf/2601.10020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 141. How Diplomacy Reshapes Online Discourse:Asymmetric Persistence in Online Framing of North Korea

**arXiv ID:** 2601.09942 | [PDF](https://arxiv.org/pdf/2601.09942v1)

**作者:** Hunjun Shin `[一作]` (Northeastern University), Mohit Singhal `[通讯]` (Northeastern University)

**通讯引用:** 43 | [OpenAlex ID](https://openalex.org/A5017559481)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对2018-2019年美朝峰会前后Reddit讨论的因果差分分析，探讨高风险外交峰会如何改变公众对朝鲜的叙事框架及其网络结构，并考察成功与失败峰会的持久性差异。

**💡 创新点**

首次将因果推断（DiD）与大型语言模型（LLM）框架分类及图网络分析相结合，系统揭示外交事件对内容与结构层面叙事的异向持久影响（“非对称持久性”）。

**🔧 技术方法**

使用GPT‑4o‑mini的Codebook LLM进行多维度框架分类，RoBERTa进行情感分析，GraphRAG式知识图谱与图网络指标（密度、边缘与社区分布）进行结构化语义网络分析。

**📊 数据集**

基于Arctic Shift API获取的2017‑2019年英文Reddit 29,688篇主题帖（10,448篇朝鲜相关）与255,391条高可见度评论的公开数据。

**📈 对比分析**

与多国对照组（中国、伊朗、俄罗斯）相结合的双向固定效应DiD模型，结果显示峰会后对朝鲜的外交框架提升显著，情感提升幅度相对更小；网络结构指标显示威胁边缘比例下降、外交边缘比例上升，并在峰会失败后仅部分反转，体现非对称持久性。

**⚠️ 局限性**

样本局限于英语Reddit社区，缺乏代表性；受限于平行趋势假设，俄罗斯被排除在某些分析之外；未对用户级时间序列进行追踪，难以区分观点变更与用户轮换；短期窗口（仅至2020年）可能无法揭示长期持久性。

---

## 142. Heterogeneous computing platform for real-time robotics

**arXiv ID:** 2601.09755 | [PDF](https://arxiv.org/pdf/2601.09755v1)

**作者:** Jakub Fil `[一作]` (WAIYS GmbH), Steve Furber `[通讯]` (University of Manchester)

**通讯引用:** 12760 | [OpenAlex ID](https://openalex.org/A5083177159)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aaccfe5c-6b26-4208-b23c-35331481e142` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文设计并实现了一个异构计算平台，将Intel Loihi2 neuromorphic芯片与NVIDIA DGX A100 GPU集群结合，用事件相机实时跟踪手部动作，驱动Ameca人形机器人与人类共同演奏theremin，实现端到端低延迟人机交互演出。

**💡 创新点**

创新点在于首次将事件相机与Loihi2结合实现高效低功耗手部跟踪，并将大型脑模型Spaun 2.0部署到多GPU系统，同时整合大语言模型、语音识别与合成，构建完整可实时执行的音乐交互框架。

**🔧 技术方法**

所用技术包括Intel Loihi2 neuromorphic芯片、Dynamic Vision Sensor (DVS)、NVIDIA DGX A100 GPU、Spaun 2.0脑模型、LLaMA 13B大语言模型、OpenAI Whisper语音识别、Amazon Polly语音合成、SpiNNaker2、热液冷却等。

**📊 数据集**

采用的主要数据集是由专业theremin演奏者Carolina Eyck录制的音频、视频、DVS事件和深度相机数据，并使用公开的DVS 19k事件数据集训练手部跟踪模型。

**📈 对比分析**

与传统GPU-only方案相比，Loihi2实现的手部跟踪功耗仅4mW，延迟约100 Hz；Spaun 2.0在7个A100 GPU上实现近实时推理，整体系统延迟低于200 ms，能够在现场表演中保持同步；在能耗上，Loihi2与传统GPU相比可降低约90%。

**⚠️ 局限性**

主要局限包括：Spaun 2.0仍需多GPU部署且能耗高；Loihi2编程生态尚不成熟，软件开发成本高；系统对复杂环境鲁棒性不足；整体硬件成本和维护要求高，难以广泛推广。

---

## 143. Malware Classification using Diluted Convolutional Neural Network with Fast Gradient Sign Method

**arXiv ID:** 2601.09933 | [PDF](https://arxiv.org/pdf/2601.09933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 144. Synthetic Data for Veterinary EHR De-identification: Benefits, Limits, and Safety Trade-offs Under Fixed Compute

**arXiv ID:** 2601.09756 | [PDF](https://arxiv.org/pdf/2601.09756v1)

**作者:** David Brundage `[一作]` `[通讯]` (University of Wisconsin Madison), David Brundage (University of Wisconsin Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了在低资源宠物医疗电子健康记录环境下，利用LLM生成的合成临床笔记来改进或影响去标识化模型的安全性与效能。

**💡 创新点**

创新点在于将合成笔记用于“模板-占位符”方式生成，严格控制隐私风险；同时通过文档级泄露率指标评估去标识化安全性，并系统分析训练方案（增广vs替代）与合成数据分布偏移的影响。

**🔧 技术方法**

使用Transformer NER模型（PetBERT、VetBERT、Bio_ClinicalBERT），结合epoch扩展和compute‑matched训练、占位符填充与去重的合成生成流程。

**📊 数据集**

数据集为PetEVAL公开的兽医临床笔记；从中划分真实训练集（1,249条）、测试集（3,750条）以及10,382条合成笔记（含2,978条带PII）。

**📈 对比分析**

比较方法包括：增广（按合成比例扩展训练量）、固定样本替代（固定样本数下替换真实笔记）以及compute‑matched对照。结果显示：在epoch扩展下，合成增广可显著降低文档泄露率并提升F1；但在固定预算或替代下，合成数据不能替代真实标注，泄露率反而上升。

**⚠️ 局限性**

局限性包括：仅在单一机构内部低资源仿真中评估，缺乏跨站泛化；生成流程受限于模板占位符，可能忽略真实文本的复杂性；只评估了编码器式NER模型，未覆盖生成式去标识化方法。

---

## 145. Enhancing Formal Software Specification with Artificial Intelligence

**arXiv ID:** 2601.09745 | [PDF](https://arxiv.org/pdf/2601.09745v1)

**作者:** Antonio Abu Nassar `[一作]` (IBM Research), Eitan Farchi `[通讯]` (IBM Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用人工智能辅助的自然语言与轻量级数学符号作为中间规范，生成并验证一套知识增长模拟程序，包含离职机制和Shapley值贡献评估。

**💡 创新点**

创新点在于将形式化规范与LLM推理结合，显著降低符号负担，且在模拟中实现即时的正确性验证与贡献度量；同时引入离职与跨组织迁移机制，为多组织知识共享研究提供新框架。

**🔧 技术方法**

主要技术包括自然语言规范 + 轻量数学符号、LLM（AI）推理与代码生成、Python实现、网络图构建、Shapley值计算、Monte‑Carlo 统计评估。

**📊 数据集**

使用自构造的图网络（5节点、10节点）作为实验数据集，并未使用公开数据集；知识项通过模拟生成。

**📈 对比分析**

通过对比两种 TG（facilitator）策略（随机添加 k 条边 vs 固定添加 (v1,v4)、(v1,v5)）并多次 Monte‑Carlo 采样，评估知识产出与奖励。实验结果表明：固定边策略产生的知识总量更大，且给予 TG 的奖励更高。

**⚠️ 局限性**

局限性：模型规模有限，AI 关注窗口导致规范遗漏；离职与迁移细节需手工调试；实验仅覆盖单组织或两组织，缺乏大规模或真实组织数据的验证。

---

## 146. Adoption and Evolution of Code Style and Best Programming Practices in Open-Source Projects

**arXiv ID:** 2601.09832 | [PDF](https://arxiv.org/pdf/2601.09832v1)

**作者:** Alvari Kupari `[一作]`, Valerio Terragni `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对1036个受欢迎的Java开源项目进行大规模代码风格与最佳实践合规性分析

**💡 创新点**

扩展了开源工具以实现对Google Style Guide全部规则的检测，并系统评估自报遵循与实际合规性差距及演化趋势

**🔧 技术方法**

利用改进的Checkstyle + JavaParser + PMD + NLP（JWNL）进行静态分析，覆盖命名、Javadoc、异常处理等多种违规类型

**📊 数据集**

使用GitHub公开的1036个受欢迎Java仓库（按星级筛选，并通过大小与兼容性过滤）以及在此基础上挑选的41个成熟活跃子集

**📈 对比分析**

采用归一化违规率、5%阈值、Venn图、平均违规率等方法进行比较；工具在手工验证中无误报，发现自报遵循的项目合规率高达75%，整体违规率随时间保持稳定，说明工具检测性能可靠

**⚠️ 局限性**

局限性包括未系统评估误检率（可能漏检导致低估违规）、仅观察一年演化、仅聚焦高星项目导致样本偏倚、阈值选择可能影响结论、工具在某些规则下仍可能存在漏检

---

## 147. Private Information Retrieval for Graph-based Replication with Minimal Subpacketization

**arXiv ID:** 2601.09957 | [PDF](https://arxiv.org/pdf/2601.09957v1)

**作者:** Vayur Shanbhag `[一作]` (International Institute of Information Technology), Prasad Krishnan `[通讯]` (International Institute of Information Technology)

**通讯引用:** 718 | [OpenAlex ID](https://openalex.org/A5103048827)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了两种可变下载、子包化简为L=1的私有信息检索（PIR）协议，分别针对星形图和任意图实现了低子包化和可观的速率；

**💡 创新点**

创新点在于利用可变下载中的空查询（null‑query）技术，使得子包化最低到1，同时通过独立集分解与随机矩阵构造，得到对星形图和一般图（尤其是二分图与多重图）更高的速率；

**🔧 技术方法**

主要技术包括：可变下载协议设计、独立集分解与递归查询生成、随机排列构造查询矩阵、XOR组合实现文件解码以及信息论隐私分析；

**📊 数据集**

该工作不依赖具体数据集，而是针对理论上的图结构（星形图、完全图、二分图、多重图等）进行设计与分析；

**📈 对比分析**

与已有固定下载方案相比，星形图方案在子包化最小的同时实现更高的速率；对完全图和一般图虽然速率略低于最优已知方案，但在二分图和多重图上实现了目前最佳已知速率；

**⚠️ 局限性**

限制在于：对一般图的速率仍低于部分已知方案；方案对图结构的分解与随机化实现复杂度较高；在完全图情形下，无法达到容量上界；

---

## 148. SocraticKG: Knowledge Graph Construction via QA-Driven Fact Extraction

**arXiv ID:** 2601.10003 | [PDF](https://arxiv.org/pdf/2601.10003v1)

**作者:** Sanghyeok Choi `[一作]` (Seoul National University), Taehyeong Kim `[通讯]` (Seoul National University)

**通讯引用:** 3406 | [OpenAlex ID](https://openalex.org/A5100400948)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于问答中介的知识图谱构建方法 SocraticKG，利用 5W1H 指导的 QA 对文本进行结构化语义展开后再抽取三元组。

**💡 创新点**

创新点在于将 QA 作为结构化中间表示，系统性地展开文档层面语义并显式捕获隐含关系，从而缓解事实覆盖与图结构连贯性的矛盾。

**🔧 技术方法**

采用的大模型技术包括 LLM（如 GPT‑4o、Claude‑4 等）生成 QA 与三元组，以及嵌入聚类+LLM 微调的规范化步骤。

**📊 数据集**

使用 MINE 基准数据集（100 篇文章 1500 条真值事实）进行评测。

**📈 对比分析**

与直接抽取、GraphRAG、KGGen 等方法对比，在五种 LLM 上 SoKG 在事实保留率、节点连通度和碎片化指数上均表现最优，尤其在 Claude‑4 上达到 96.3% 的保留率。

**⚠️ 局限性**

局限性包括较高的 token 消耗与延迟、对 LLM 推理深度的依赖，以及目前仅支持二元三元组，无法完整表达多元属性。

---

## 149. Forward-only learning in memristor arrays with month-scale stability

**arXiv ID:** 2601.09903 | [PDF](https://arxiv.org/pdf/2601.09903v1)

**作者:** Adrien Renaudineau `[一作]` (Universite Paris-Saclay), Damien Querlioz `[通讯]` (Universite Paris-Saclay)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在标准HfO_x/Ti薄膜记忆体上实现了亚1 V重置式、单脉冲权重更新，并在此基础上完成了两层前馈网络的本地训练（Backprop、Forward‑Forward、Competitive Forward）。

**💡 创新点**

创新点在于：①提出并实验验证了“亚1 V重置‑仅限单脉冲”编程方案，显著提升耐久性和长期保留；②将Hinton的Forward‑Forward算法与竞争性聚类相结合，形成完全前向、单通道的训练流程，兼容低电压记忆体；③结合层级训练策略、梯度阈值裁剪和差分对权重编码，成功将硬件噪声和非线性对学习的影响降至可接受水平。

**🔧 技术方法**

主要技术包括：亚1 V重置记忆体编程（单脉冲、递增式耗散）、差分配对编码实现有符号权重、CMOS/memristor混合交叉线阵列、硬件-软件协同训练（on‑chip MAC或软件仿真）、梯度阈值裁剪与层级冻结、前向只传递的Goodness更新规则、竞争性聚类读出层。

**📊 数据集**

使用ImageNet预训练的ResNet‑18特征提取后得到的32维特征，作为四类熊（棕熊、旱獭熊、北极熊、熊猫）分类任务的数据集；在MNIST数据集上也做了模拟验证。

**📈 对比分析**

与浮点Backprop相比较：两层网络在亚1 V重置硬件上取得90.0%（Backprop）与89.5%（Supervised Forward‑Forward）/89.6%（Competitive Forward）精度，统计检验显示差异不显著；单层感知机在Backprop下误差显著更大。硬件训练耗能约为250 µJ（单脉冲）与程序‑验证基线相比降低至约1/460；长期保留达90天后误差<3 µS，测试准确率稳定。

**⚠️ 局限性**

限制包括：亚1 V重置仍存在脉冲大小随机性和有限的模拟窗口；单脉冲更新无法精确控制梯度大小，导致相对软件梯度下降速度略慢；差分对编码和阈值裁剪导致部分有效梯度被丢弃；在更大规模网络或更复杂任务中，设备间波动和读写误差对性能的影响尚未完全评估。

---

## 150. Kinematic Tokenization: Optimization-Based Continuous-Time Tokens for Learnable Decision Policies in Noisy Time Series

**arXiv ID:** 2601.09949 | [PDF](https://arxiv.org/pdf/2601.09949v1)

**作者:** Griffin Kearney `[一作]` (Syracuse University), Griffin Kearney `[通讯]` (Syracuse University)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5086374342)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于最优化的连续时间切片（Kinematic Tokenization）将金融时序数据映射为速度、加速度等动力学系数，并用Transformer学习交易决策。

**💡 创新点**

创新点在于将物理动力学约束嵌入到离散Token化过程中，生成可解析的连续样条并提取高阶导数，解决了传统离散化对噪声敏感的问题。

**🔧 技术方法**

使用优化式数据丰富（最大似然样条拟合）、Causal Transformer、LoRA微调以及加权交叉熵异步损失。

**📊 数据集**

在美国上市日内股价与成交量（约22.5M个连续切片，64天窗口）做为训练集，测试集为2023年起的多只大盘股票。

**📈 对比分析**

与PatchTST、RawGPT及RawGPT-FD等离散或补丁化基线对比，连续切片模型在非对称“买/卖/持有”任务中避免了“清算平衡”陷阱，得到正收益、较低最大回撤和可观的夏普/索提诺比率。

**⚠️ 局限性**

局限包括需要先验的动力学模型、对高频微结构噪声的敏感性、对交易成本和滑点的假设，以及模型在低波动性资产上表现不佳。

---

## 151. CaMeLs Can Use Computers Too: System-level Security for Computer Use Agents

**arXiv ID:** 2601.09923 | [PDF](https://arxiv.org/pdf/2601.09923v1)

**作者:** Hanna Foerster `[一作]` (University of Cambridge), Yiren Zhao `[通讯]` (AI Security Company)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在Dual‑LLM框架基础上提出单次规划（Single‑Shot Planning），通过Privileged Planner（P‑LLM）在不观测屏幕的情况下生成完整的带条件分支的执行图，并引入Observe‑Verify‑Act范式和冗余验证机制，构建安全的Computer Use Agent（CUA）体系；

**💡 创新点**

创新点包括①将多轮交互式CUI自动化转化为单次规划；②Observe‑Verify‑Act模式使得规划在未见环境时已可自洽；③首次提出并评估Branch Steering攻击与对应的DOM一致性/多模态共识防御；④为开源模型提供基于本地视觉模型、远端规划模型的安全部署方案；

**🔧 技术方法**

采用Dual‑LLM架构、Privileged Planner（P‑LLM）与Quarantined Perception（Q‑VLM）分离、CaMeL/Fides编程框架、Python‑style计划生成、DOM一致性验证、独立多模态VLM共识验证、对抗像素攻击等技术；

**📊 数据集**

使用OSWorld基准（含Chrome、LibreOffice、GIMP等桌面应用任务），并利用UITars‑1.5‑7B、OpenCUA‑32B、Claude Sonnet 4.5等CUA实现；对抗实验基于广告banner与像素扰动；

**📈 对比分析**

评估采用Pass@k指标（k=1~5），与无防御、DOM一致性、Multi‑Modal Consensus三种防御方案对比。单次规划在小模型上Pass@3提升约19%，在大模型上保持约57% Pass@5；防御在阻止Cookie伪装与像素攻击方面有效，但在部分任务上略有性能下降；

**⚠️ 局限性**

主要局限包括：①单次规划仍需预知任务初始状态，任务描述不完整时性能下降；②冗余验证虽降低数据流攻击风险，但无法完全阻止Branch Steering，且可能产生误报与额外计算开销；③对极端对抗样本（如高度自适应像素扰动）仍存漏洞；

---

## 152. Long-Chain Reasoning Distillation via Adaptive Prefix Alignment

**arXiv ID:** 2601.10064 | [PDF](https://arxiv.org/pdf/2601.10064v1)

**作者:** Zhenghao Liu `[一作]` (Northeastern University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 36343 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种自适应前缀对齐的知识蒸馏方法，以将大型语言模型的长链推理有效迁移到小型学生模型。

**💡 创新点**

创新点在于通过学生自评判定最小足够前缀并用二分搜索定位截断点，再将该前缀作为先验推理上下文进行对齐蒸馏，避免冗余和不确定的后续步骤。

**🔧 技术方法**

采用的技术包括教师生成链式推理轨迹、学生自评判与二分搜索的前缀截断、prefix‑alignment蒸馏、LoRA参数高效微调以及TRL+LLaMA‑Factory框架。

**📊 数据集**

训练数据来自s1K-1.1（DeepSeek‑R1生成的高质量解题轨迹），评测使用MATH500、AIME、AMC等数学推理基准。

**📈 对比分析**

与零样本、标签监督、完整CoT蒸馏和UPFT等基线相比，P‑ALIGN在 Pass@1/Pass@3 上平均提升约3% 以上，且在不同模型规模下均保持领先。

**⚠️ 局限性**

局限性包括对强大封闭源教师模型的依赖、生成长链推理所需的高算力，以及自评判过程受学生模型判断能力限制，可能导致过小模型判断失误。

---

## 153. FilDeep: Learning Large Deformations of Elastic-Plastic Solids with Multi-Fidelity Data

**arXiv ID:** 2601.10031 | [PDF](https://arxiv.org/pdf/2601.10031v1)

**作者:** Jianheng Tang `[一作]` (Peking University), Yunhuai Liu `[通讯]` (Peking University)

**通讯引用:** 4657 | [OpenAlex ID](https://openalex.org/A5082653046)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 FilDeep，一种利用多保真度数据学习弹塑性固体大变形的深度学习框架；

**💡 创新点**

首次在弹塑性大变形问题中结合多保真度数据与注意力增强的跨保真度模块，以解决数据量-准确性矛盾；

**🔧 技术方法**

采用 Transformer‑based 编码器/解码器、Attention‑enabled Cross‑Fidelity（ACF）模块、残差连接以及多输入特征融合；

**📊 数据集**

构造了首个公开的多保真度数据集，包含约3,000个低保真度和300个高保真度实例，数据来自真实制造工况；

**📈 对比分析**

与 Transformer、DeepONet、FNO 等单保真度基线以及 MFNN、MF‑DeepONet 等多保真度方法对比，FilDeep 在 MAD、3D IoU、TE 指标上均优于基线，且在实际工厂部署时推理速度比 FEM 快 10^5 倍；

**⚠️ 局限性**

局限性：对高保真度样本仍需一定数量，模型对不同工况的泛化仍依赖于预先设计的编码器/解码器，且在极端复杂几何或材料行为变化时性能可能下降。

---

## 154. Closing the Data Loop: Using OpenDataArena to Engineer Superior Training Datasets

**arXiv ID:** 2601.09733 | [PDF](https://arxiv.org/pdf/2601.09733v1)

**作者:** Xin Gao `[一作]` (Shanghai Artificial Intelligence Laboratory, OpenDataLab, OpenDataArena), Lijun Wu `[通讯]` (Shanghai Artificial Intelligence Laboratory, OpenDataLab, OpenDataArena)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM后训练阶段，基于OpenDataArena的评估信号构建高质量数据集并实现闭环迭代优化。

**💡 创新点**

提出闭环数据工程框架，将leaderboard排名作为动态反馈，实现量化数据选择和优化；同时引入两阶段难度筛选与synthesize‑and‑verify的验证流程。

**🔧 技术方法**

利用数据评估器、难度滤波、生成与验证（Verifier‑backed distillation）、聚类抽样与多域混合、深度学习训练框架（LlamaFactory、vLLM）等技术。

**📊 数据集**

构建了 ODA‑Math‑460k（数学）与 ODA‑Mixture‑100k/500k（多域）数据集，融合了多源公开数学、代码、通用与推理数据。

**📈 对比分析**

在 Qwen2.5‑7B/3‑8B 等模型上实验，取得 SOTA 结果，数据效率提升 2–5 倍，尤其在高难度竞赛题（AIME、CMIMC 等）上显著超越先前基准。

**⚠️ 局限性**

局限包括对验证器的高度依赖、对新兴任务/多语言覆盖不足、以及对数据泄露防护仍需进一步加强。

---

## 155. ADMEDTAGGER: an annotation framework for distillation of expert knowledge for the Polish medical language

**arXiv ID:** 2601.09722 | [PDF](https://arxiv.org/pdf/2601.09722v1)

**作者:** Franciszek Górski `[一作]` (Gdansk University of Technology), Andrzej Czyżewski `[通讯]` (Gdansk University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过多语言LLM Llama3.1 做教师模型，对波兰语医学文本进行注释，并用有限专家校对后训练 BERT 系列学生模型，实现多类别分类。

**💡 创新点**

创新点在于将大型多语言LLM 用作知识蒸馏教师，解决波兰医学文本标注稀缺问题，并在同一数据集上比较 DistilBERT、BioBERT 与 HerBERT 三种 BERT 变体。

**🔧 技术方法**

使用技术包括提示工程、多语言LLM蒸馏、BERT/DistilBERT、BioBERT、HerBERT 微调、F1/AUROC/准确率评估、Wilcoxon 检验和显著性置信区间。

**📊 数据集**

数据集来源于 ADMEDVOICE 项目，包含 5 个临床类别（心脏病、放射学、肿瘤学、病理学、高血压）共约 21,073 条波兰语医学文本。

**📈 对比分析**

在验证集上比较模型的 F1、AUROC 与准确率，并通过 Wilcoxon 检验确认差异；DistilBERT 在大多数域达到 F1>0.80，显著优于 BioBERT 与 HerBERT。

**⚠️ 局限性**

局限性包括教师模型可能传播偏差、对提示设计的敏感性、需周期性专家复核，以及在高血压等类别的性能仍有提升空间。

---

## 156. Improving Chain-of-Thought for Logical Reasoning via Attention-Aware Intervention

**arXiv ID:** 2601.09805 | [PDF](https://arxiv.org/pdf/2601.09805v1)

**作者:** Nguyen Minh Phuong `[一作]`, Naoya Inoue `[通讯]` (Japan Advanced Institute of Science and Technology)

**通讯引用:** 1532 | [OpenAlex ID](https://openalex.org/A5028046901)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无交互、端到端的注意力干预框架 AAI，以提升大型语言模型在逻辑推理任务中的性能。

**💡 创新点**

创新点在于通过在少量示例中注入符号结构，发现并利用特定注意力头（anchor、copy、aggregation）对应的逻辑操作，并在推理时对这些头进行动态加权，从而实现结构知识的注入。

**🔧 技术方法**

采用注意力模式分析、头选择、注意力重加权技术（AAI），并配合符号化提示（Symbolic‑Aided CoT）实现无外部符号求解器的推理。

**📊 数据集**

使用的评测数据集包括 ProofWriter、ProntoQA、LogicalDeduction、FOLIO 以及数学推理基准 GSM8k。

**📈 对比分析**

与标准 CoT、Symbolic‑Aided CoT 以及交互式+符号求解器方法进行比较，AAI 在多模型多规模（1.7B–32B）上平均提升 1–3% 以上，在 Qwen‑3‑32B 上甚至与 GPT‑4 接近。

**⚠️ 局限性**

局限性：仅在合成或中等规模数据上验证；方法高度依赖 Transformer 的注意力结构，难以直接迁移到不具备自注意力的模型；且未针对常识推理等其他领域进行扩展。

---

## 157. StatLLaMA: A multi-stage training framework for building a domain-optimized statistical language model

**arXiv ID:** 2601.09718 | [PDF](https://arxiv.org/pdf/2601.09718v1)

**作者:** Jing-Yi Zeng `[一作]` (National Yang Ming Chiao Tung University), Guan-Hua Huang `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 3257 | [OpenAlex ID](https://openalex.org/A5081040175)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对统计领域构建了一款轻量级的统计专用大语言模型StatLLaMA，采用多阶段训练（CoP、Instruction Tuning、SFT、DPO、DTFT）在LLaMA‑3.2‑3B‑Instruct基础上实现域知识注入与人类偏好对齐。

**💡 创新点**

创新点包括：①验证从已具备指令能力的Instruct模型出发进行领域适配比从无指令模型更有效；②首次系统比较DPO与GRPO在统计领域的对齐效果，发现DPO稳定提升并恢复通用推理能力；③提出极低强度DTFT策略避免已优化模型的灾难性遗忘，并实现最终模型在数学推理、常识推理与统计专业测评上的均衡提升。

**🔧 技术方法**

使用的技术包括：LLaMA‑3.2‑3B基础模型、LoRA参数高效微调、4‑bit量化（QLoRA）、持续预训练（CoP）、指令微调、监督微调（SFT）、直接偏好优化（DPO）以及低强度的下游任务微调（DTFT）。

**📊 数据集**

数据集涵盖：统计领域语料（arXiv、S2ORC、OpenStax书籍等）、指令与问答集（Hugging Face、Gemini生成的CoT与多轮对话）、偏好对齐对（Gemini生成的Preference pairs）、以及评测基准（GSM8K、ARC、AP Statistics）。

**📈 对比分析**

比较方法：三条训练管线（知识先行、知识+指令桥接、指令优先）在三个基准上的性能对比。结果显示，指令优先管线最优，SFT‑v3.4+DPO+DTFT‑v2版本在GSM8K、AP Statistics、ARC上的准确率分别提升到约59%、41.5%和41.8%，明显优于基线和其他管线。

**⚠️ 局限性**

局限性包括：仅在3B模型规模实验，未验证更大规模模型的可扩展性；评测基准仍不覆盖实际统计工作流程与数据噪声处理；偏好数据主要来源于Gemini，可能带来教师模型偏差；DTFT的探索范围有限，未全面评估不同PEFT配置；实验规模受资源限制，未做完整的超参数搜索。

---

## 158. Benchmarking Cross-Lingual Semantic Alignment in Multilingual Embeddings

**arXiv ID:** 2601.09732 | [PDF](https://arxiv.org/pdf/2601.09732v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 159. Empowering Older Adults in Digital Technology Use with Foundation Models

**arXiv ID:** 2601.10018 | [PDF](https://arxiv.org/pdf/2601.10018v1)

**作者:** Hasti Sharifi `[一作]` (University of Illinois Chicago), Debaleena Chattopadhyay `[通讯]` (University of Illinois Chicago)

**通讯引用:** 1263 | [OpenAlex ID](https://openalex.org/A5073730098)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究老年人技术支持求助的沟通方式，识别四大障碍并构建 AI 生成改写与解答流程

**💡 创新点**

首次系统归纳老年人非结构化技术求助的四种沟通障碍，并提供可复制的 prompt‑chaining 方案与 OATS 合成数据集

**🔧 技术方法**

使用 GPT‑4o、OS‑ATLAS 大语言模型和 prompt‑chaining 链接、生成改写与解答，配合 Google 搜索评估

**📊 数据集**

基于 57 条真实日记问答的手工编码数据；合成的 OATS 数据集（514 条）用于验证模型泛化

**📈 对比分析**

对比原始与 AI 改写的查询在 Google Top‑5 成功率（69% vs 35%）和用户理解度（原 65.8% vs 改写 93.7%）等指标，表明 GPT‑4o 在改写与解答上显著优于基线

**⚠️ 局限性**

样本规模有限、仅英美社区老年人、使用高性能专有模型、实验为异步控制实验，缺乏对认知受损或低技术水平老年人的验证

---

## 160. Behavioral Targeting, a European Legal Perspective

**arXiv ID:** 2601.09712 | [PDF](https://arxiv.org/pdf/2601.09712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 161. Cross-Platform Evaluation of Large Language Model Safety in Pediatric Consultations: Evolution of Adversarial Robustness and the Scale Paradox

**arXiv ID:** 2601.09721 | [PDF](https://arxiv.org/pdf/2601.09721v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 162. Privacy Enhanced PEFT: Tensor Train Decomposition Improves Privacy Utility Tradeoffs under DP-SGD

**arXiv ID:** 2601.10045 | [PDF](https://arxiv.org/pdf/2601.10045v1)

**作者:** Pradip Kunwar `[一作]` (Tennessee Tech University), Manish Bhattarai `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 436 | [OpenAlex ID](https://openalex.org/A5012614331)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在差分隐私条件下使用张量训练（Tensor‑Train）结构化的参数高效微调（TTLoRA），并提出了TTLoRA‑DP——一种将Ghost Clipping扩展到TT核心、实现高效DP‑SGD的框架。

**💡 创新点**

创新点在于：①通过张量训练分解将适配器参数压缩到更小的空间，同时保持表达能力；②首次为TTLoRA实现了可扩展的DP‑SGD，利用缓存的张量收缩状态实现精确的逐样本梯度范数计算；③证明该结构化方法在隐私‑效能（perplexity 与 membership‑inference AUC）上优于传统LoRA。

**🔧 技术方法**

使用的技术包括：Tensor‑Train（TT）分解、LoRA、DP‑SGD、Ghost Clipping、PreCurious membership‑inference、GPT‑2 124M 微调、RDP 隐私计数。

**📊 数据集**

实验数据集为 Enron 电子邮件语料和 Penn Treebank（PTB）文本，分别用于评估隐私与效能。

**📈 对比分析**

与 LoRA 及全参数微调（FFT）在相同 ε（0.5、1、3、5）预算下对比，TTLoRA 在保持相近或更好 perplexity 的同时，MIA AUC 几乎保持在 50% 随机猜测水平，而 LoRA 的泄露率随 ε 上升显著增加；且 TTLoRA 仅使用约 10‑15 倍更少的可训练参数。

**⚠️ 局限性**

局限性包括：仅在 GPT‑2 124M 上实验，未验证更大模型；评估仅覆盖语言建模与 PreCurious 的 loss‑based MIA，未涉及其他泄露方式或下游任务；未探讨更强的隐私攻击或分布式训练环境。

---

## 163. Syntactic Framing Fragility: An Audit of Robustness in LLM Ethical Decisions

**arXiv ID:** 2601.09724 | [PDF](https://arxiv.org/pdf/2601.09724v1)

**作者:** Katherine Elkins `[一作]` (Kenyon), Jon Chun `[通讯]` (Kenyon)

**通讯引用:** 296 | [OpenAlex ID](https://openalex.org/A5034544789)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大语言模型在伦理决策场景下对逻辑等价但句法不同提示的鲁棒性，并提出Syntactic Framing Fragility (SFF) 框架。

**💡 创新点**

引入逻辑极性归一化(LPN)和Syntactic Variation Index(SVI)来量化句法框架不一致，系统性审计23个模型，揭示开放源模型更脆弱，并验证推理诱导可缓解这一问题。

**🔧 技术方法**

使用逻辑极性归一化、SVI效应大小、Cochran's Q检验、多轮采样、推理诱导提示工程等技术。

**📊 数据集**

构造14个跨7领域的伦理情境，配合4个句法框架，生成约4万条决策样本进行评估。

**📈 对比分析**

对每个模型-情境-框架组合进行多次采样，计算SVI并进行统计显著性检验；结果显示61.9%单元显著不一致，开放源模型平均SVI为0.79，高于商业模型；推理提示可在多模型中显著降低SVI。

**⚠️ 局限性**

仅使用英文提示与固定模板，二元决策提取压缩细节，模型API不稳定导致部分模型被排除，框架虽保持语义不变但仍可能引入语用漂移，且样本量和情境范围有限。

---

## 164. SPRInG: Continual LLM Personalization via Selective Parametric Adaptation and Retrieval-Interpolated Generation

**arXiv ID:** 2601.09974 | [PDF](https://arxiv.org/pdf/2601.09974v1)

**作者:** Seoyeon Kim `[一作]` (Yonsei University), Jaehyung Kim `[通讯]` (Yonsei University)

**通讯引用:** 1926 | [OpenAlex ID](https://openalex.org/A5110631092)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对持续性语言模型个性化的半参数框架SPAR（Selective Parametric adaptation and Retrieval‑Interpolated Generation），能够在用户兴趣随时间漂移时动态更新模型，避免灾难性遗忘。

**💡 创新点**

核心创新在于两层选择机制：1）训练阶段利用漂移驱动的样本评分函数只更新高新颖性交互，构建残差回放缓冲；2）推理阶段采用严格的相关性门控和双路径logit插值，将参数化知识与检索历史在每个解码步骤动态融合。

**🔧 技术方法**

技术手段包括LoRA参数化适配器、漂移评分函数（基于对比基础模型与适配器的对数似然）、回放缓冲策略、BM25检索、门控检索、双路径logit插值。

**📊 数据集**

在LongLaMP基准的两项长文本个性化生成任务（摘要生成和评测写作）上进行实验，用户数据按时间划分为5个阶段。

**📈 对比分析**

与基线（无个性化、RAG、PAG、OPPU）以及多种持续学习方法（CAMA、SAR、O‑LoRA、EWC‑LoRA）对比，SPAR在所有阶段均显著优于对手，Abstract任务ROUGE‑L提升15.85%，Review任务提升13.60%；在Top‑30%漂移选择和λ=0.5插值策略下实现最佳效果。

**⚠️ 局限性**

主要限制包括：双路径插值导致推理时延和计算成本增加；漂移评分对突发噪声/主题跳变的鲁棒性有限；随用户规模扩大时适配器存储与服务成本提升。

---

## 165. MedVL-SAM2: A unified 3D medical vision-language model for multimodal reasoning and prompt-driven segmentation

**arXiv ID:** 2601.09879 | [PDF](https://arxiv.org/pdf/2601.09879v1)

**作者:** Yang Xing `[一作]` (University of Florida), Kuang Gong `[通讯]` (University of Florida)

**通讯引用:** 2812 | [OpenAlex ID](https://openalex.org/A5013691634)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出MedVL-SAM2统一的3D医学视觉-语言模型，支持报告生成、VQA和多模式3D分割。

**💡 创新点**

将SAM2的可提示分割与LLaVA式框架融合，实现语言驱动的像素级定位与全局推理统一。

**🔧 技术方法**

使用3D CLIP视觉编码器、MLP‑Mixer投影层、InternVL‑2.5 LLM + LoRA、SAM2分割模块，并采用多阶段训练。

**📊 数据集**

训练与评估基于CT‑RATE（报告与VQA）和M3D‑Seg（分割）等公开数据集，并构造交互提示。

**📈 对比分析**

与现有2D/3D VLM、VFMs在报告生成、VQA、语义/参照/交互分割任务上对比，MedVL‑SAM2在BLEU、ROUGE、CIDEr、Dice等指标均显著优于SOTA。

**⚠️ 局限性**

仍存在推理速度与极小/复杂结构分割精度受限，以及缺乏跨模态鲁棒性评估等局限。

---

## 166. Memo-SQL: Structured Decomposition and Experience-Driven Self-Correction for Training-Free NL2SQL

**arXiv ID:** 2601.10011 | [PDF](https://arxiv.org/pdf/2601.10011v1)

**作者:** Zerui Yang `[一作]` (City University of Hong Kong), Bo Bai `[通讯]` (Huawei Technologies Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种完全无训练、无外部API的 NL2SQL 框架 Memo‑SQL，结合结构化分解和基于历史错误-修正对的检索增强自校正。

**💡 创新点**

创新点在于：① 用实体、层级和原子顺序三种分解策略强制生成多样化子查询；② 通过动态检索历史错误–修正实例进行检索增强的上下文学习，使模型能够针对不同错误类型自适应修正；③ 彻底摆脱微调与闭源模型，显著降低推理成本。

**🔧 技术方法**

核心技术包括：结构化分解（实体‑级、层级‑级、原子‑级）、ReAct+Reflect 迭代推理、检索增强的 in‑context 学习（RAG）、自一致性打分和多样化 SQL 合成；使用的 LLM 为 Qwen‑Coder 系列（Qwen3‑Coder‑30B‑A3B、Qwen2.5‑Coder‑32B 等）以及 Llama‑3.1‑8B 等开源模型。

**📊 数据集**

主要使用公开基准 BIRD（dev、dev‑new）、Spider dev、CHESS‑SDS 等数据库查询数据集构建错误‑修正记忆并评估性能。

**📈 对比分析**

与 Distillery、ROUTE、Alpha‑SQL 等现有测试‑时扩展方法对比，Memo‑SQL 在 BIRD dev‑new 上取得 68.5% 执行准确率，成为开源零微调方法中的 SOTA；相比 Alpha‑SQL，令令标记数降低 10 倍、查询延时降低 12 倍，仍保持高准确率。

**⚠️ 局限性**

局限性：① 记忆库基于模型生成的合成错误，可能与真实用户错误模式不完全一致；② 分解仍依赖 LLM 的准确理解，极度模糊的问题仍可能导致错误累积；③ 需要多轮 LLM 调用和 SQL 执行，对极低时延场景仍不够轻量。

---

## 167. GUI-Eyes: Tool-Augmented Perception for Visual Grounding in GUI Agents

**arXiv ID:** 2601.09770 | [PDF](https://arxiv.org/pdf/2601.09770v1)

**作者:** Chen Chen `[一作]` (University of Science and Technology of China), Wu Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6410 | [OpenAlex ID](https://openalex.org/A5068917997)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GUI‑Eyes框架，使用强化学习实现GUI任务中的主动视觉感知与两阶段推理，能动态决定何时调用裁剪或放大工具并优化其参数。

**💡 创新点**

创新点在于将主动视觉工具调用与两阶段决策相结合，并设计了结合中心距离与重叠度的连续奖励函数，提供稠密监督，显著提升工具使用的有效性。

**🔧 技术方法**

采用基于Qwen2.5‑VL‑3B的大模型，利用GRPO强化学习实现端到端的感知–推理–感知循环，并结合视觉工具实现动态观察。

**📊 数据集**

训练数据为3,000条来自OS‑Atlas、OS‑Genesis、GUI‑R1、AndroidControl的实例，评估使用ScreenSpot、ScreenSpot‑v2和ScreenSpot‑Pro三大基准。

**📈 对比分析**

与多种SFT与RL基线比较，GUI‑Eyes‑3B在ScreenSpot‑Pro上达到44.8%定位精度，显著优于先前方法（约30–40%），在多平台、文本与图标查询上均展现更高的通用性。

**⚠️ 局限性**

局限性包括对极度复杂或低分辨率界面的处理仍不理想，需要进一步提升多工具、多尺度策略的适应性以及对长链交互的鲁棒性。

---

## 168. Reinforced Linear Genetic Programming

**arXiv ID:** 2601.09736 | [PDF](https://arxiv.org/pdf/2601.09736v1)

**作者:** Urmzd Mukhammadnaim `[一作]` `[通讯]`, Urmzd Mukhammadnaim

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了将Q‑Learning与线性遗传编程相结合的RLGP框架，并用Rust实现了可扩展的实验平台；

**💡 创新点**

创新点在于利用Q‑Learning学习寄存器‑动作映射，使LGP在搜索过程中可自适应地引导探索，提升学习效率；

**🔧 技术方法**

使用的技术包括线性遗传编程、强化学习（Q‑Learning）、Rust编程语言、OpenAI Gym环境、Optuna超参数优化；

**📊 数据集**

实验数据集为CartPole‑v1和MountainCar‑v0两个经典控制任务；

**📈 对比分析**

通过对比基线LGP与RLGP在每个任务上100次实验的平均值、最大值、最小值、median等统计，发现LGP在两任务上均优于RLGP，RLGP在CartPole能快速收敛但性能低于LGP，MountainCar RLGP未能完成；

**⚠️ 局限性**

局限性包括Q‑Learning探索不足导致陷入局部最优、实验设置与超参数未充分调优、未在更广泛的任务上验证，影响结果的普适性。

---

## 169. MedRedFlag: Investigating how LLMs Redirect Misconceptions in Real-World Health Communication

**arXiv ID:** 2601.09853 | [PDF](https://arxiv.org/pdf/2601.09853v1)

**作者:** Sraavya Sambara `[一作]` (Duke University), Monica Agrawal `[通讯]` (Duke University)

**通讯引用:** 2041 | [OpenAlex ID](https://openalex.org/A5088961661)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了大语言模型在面对含有错误假设的真实健康问题时的重定向行为，构建了1100+问答的医学重定向数据集，并与临床医生回答进行了系统对比。

**💡 创新点**

首次系统评估LLM的重定向能力，提出半自动标注管线生成真实世界医学重定向数据集，并量化LLM在识别并纠正错误前提后的“容纳”倾向。

**🔧 技术方法**

采用GPT‑5构建的自动标注流程、LLM‑as‑judge评估指标，以及三种推理时减缓策略（Identify & Respond、Oracle、RAG）进行实验。

**📊 数据集**

使用从MedRedQA（已清洗的Reddit医学问答）衍生的RedFlag数据集（1103对问答）作为评测数据。

**📈 对比分析**

通过“False Assumptions Addressed”和“False Assumptions Accommodated”两项指标与医生标注进行比较；闭源模型如GPT‑5、Claude Opus 4.5在前项表现较好（≈88%/78%），但在后项仍高达60%–73%，显示虽然能识别错误假设但往往仍容纳错误信息。

**⚠️ 局限性**

可能存在训练数据泄露导致的过拟合、数据预处理噪声、仅评估公开论坛数据缺乏对新颖案例的泛化、自动标注仍可能漏检，以及识别错误前提后仍难以完全实现安全重定向的问题。

---

## 170. One-Cold Poisson Channel: A Simple Continuous-Time Channel with Zero Dispersion

**arXiv ID:** 2601.09894 | [PDF](https://arxiv.org/pdf/2601.09894v1)

**作者:** Cheuk Ting Li `[一作]` (Chinese University of Hong Kong), Cheuk Ting Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 247 | [OpenAlex ID](https://openalex.org/A5084702636)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出并分析了一种名为一冷泊松信道（OCPC）的连续时间无记忆通道，给出了其容量、色散、非渐近编码、反馈、通道仿真等完整理论结果。

**💡 创新点**

首次给出一个连续时间无记忆通道的闭式最佳非渐近误码概率，证明该通道具有零色散和单一信息谱；同时将反馈通道与前缀码推广为“一冷树”，并给出极值编码时间公式。

**🔧 技术方法**

使用泊松过程的性质、信息谱与色散理论、复合泊松匹配引理、极大似然/极大似然判别、Berry–Esseen 定理以及 Poisson functional representation 等信息理论工具。

**📊 数据集**

无，需要实验或数据集，论文为纯理论推导。

**📈 对比分析**

通过与传统 Poisson/PPM、BSC、BEC、AWGN 等通道的容量、色散、误码指数等指标进行对比，证明 OCPC 在零色散、易实现的误码概率闭式以及完美反馈下的最优期望传输时间方面优于传统模型。

**⚠️ 局限性**

局限于理想化假设（无限频带或完美阻断、无多路干扰），实际实现需考虑功率限制、滤波器分辨率和多重干扰等问题；此外仅给出理论极限，缺乏实测验证。

---

## 171. Beyond Rule-Based Workflows: An Information-Flow-Orchestrated Multi-Agents Paradigm via Agent-to-Agent Communication from CORAL

**arXiv ID:** 2601.09883 | [PDF](https://arxiv.org/pdf/2601.09883v1)

**作者:** Xinxing Ren `[一作]` (Coral Protocol), Zekun Guo `[通讯]` (University of Hull)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5062476721)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于信息流编排的多代理系统（A2A‑MAS），让代理通过自然语言互相通信，动态分配任务，取代传统的预定义工作流。

**💡 创新点**

创新点在于把工作流的构建与监督责任交给代理自适应协调，消除了人工需预先枚举所有任务状态的限制，并能在执行过程中实时识别并处理未知边缘案例。

**🔧 技术方法**

使用了CORAL的Agent‑to‑Agent通信工具包（发送/接收自然语言消息）、信息流编排器、异步消息机制，并以大型语言模型（Grok 4.1 Fast、GPT 4.1 Mini）实现各代理功能。

**📊 数据集**

在通用任务基准GAIA验证集（165个任务，难度1–3）上进行评估。

**📈 对比分析**

对比基线工作流MAS OWL，保持相同代理角色与模型配置，使用pass@1准确率与token消耗作为衡量指标。两者在所有模型一致时准确率相同；在主模型强、工作代理弱的异构设置下，A2A‑MAS实现63.64%的准确率，显著高于OWL的55.15%，token消耗基本相当。

**⚠️ 局限性**

局限性包括：在所有代理使用强模型时，token消耗略高；仅在通用任务基准上验证，缺乏对领域特定任务的适用性评估；以及对通信开销与协作效率的深入分析仍待补充。

---

## 172. The PROPER Approach to Proactivity: Benchmarking and Advancing Knowledge Gap Navigation

**arXiv ID:** 2601.09926 | [PDF](https://arxiv.org/pdf/2601.09926v1)

**作者:** Kirandeep Kaur `[一作]` (University of Washington), Chirag Shah `[通讯]` (University of Washington)

**通讯引用:** 5789 | [OpenAlex ID](https://openalex.org/A5061319881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于维度的主动式个人化对话代理ProPer，利用两代理架构（维度生成代理DGA和响应生成代理RGA）实现对未显式用户需求的精准捕获与主动介入。

**💡 创新点**

创新点在于将“未知未知”视为可选维度，通过DGA主动生成并通过校准重排序筛选优质维度，再由RGA在保持用户意图的前提下有针对性地扩展答案，实现主动与个性化的平衡；并设计了以维度覆盖为核心的评价指标。

**🔧 技术方法**

主要技术包括：LLM微调（DGA）、基于BGE‑small的维度编码与余弦相似度评估、预算约束下的校准重排序、Prompt‑driven的RGA；同时构建了ProPerBench评测框架。

**📊 数据集**

使用三类公开数据集：医疗问答MedDG、编程竞赛Code‑Contests和在线购物推荐PWAB；在每个域内进行单轮和少量多轮评估。

**📈 对比分析**

对比方法包括强基线LLM（LLaMA‑3.1‑8B‑Instruct、Qwen‑3‑8B）以及加上CoT提示的版本；使用LLM评判者按0–5评分。结果显示ProPer在所有域均显著提升μScore和Win%（单轮平均提升≈84%，尤其在医疗域中可达≈89%），多轮对话中也保持优势。

**⚠️ 局限性**

局限性包括：评估依赖LLM评判者，缺乏真实用户反馈；维度采用自由文本表述，缺少可解释性与一致性；校准参数为固定阈值，未实现自适应策略；多轮实验规模有限；未持久化用户模型，未处理更深层次的未知未知。

---

## 173. Structured Personality Control and Adaptation for LLM Agents

**arXiv ID:** 2601.10025 | [PDF](https://arxiv.org/pdf/2601.10025v1)

**作者:** Jinpeng Wang `[一作]` (Qianzhen Digital Tech), Yuyu Yin `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 3180 | [OpenAlex ID](https://openalex.org/A5070185855)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于荣格心理类型的LLM个性建模与演化框架（JPAF），通过主辅协调、强化-补偿和反思三机制实现个性化、动态适配与长期演化。

**💡 创新点**

创新点在于将荣格八种心理类型转化为可加权的函数层面，并引入三重机制：主辅协调保证核心一致性、强化-补偿实现短期情境适配、反思实现长期结构调整，形成连续演化的个性模型。

**🔧 技术方法**

技术上使用LLM微调/提示融合、加权心理类型向量、动态权重更新算法及MBTI问卷对齐评估。

**📊 数据集**

数据集包括公开的MBTI-93和MBTI-70问卷以及八套基于心理类型的情景测试（共15道题）以及ChatGPT、Llama、Qwen模型的对话数据。

**📈 对比分析**

通过与直接在提示中写入MBTI类型的基线比较，使用DAG、DAR、TAA、PSA等指标，结果显示GPT和Qwen实现100% MBTI维度匹配、90%+类型激活率及100%合法个性迁移，LLaMA虽下降但仍可行。

**⚠️ 局限性**

局限性包括对不同LLM的适配不一致（尤其LLaMA表现波动）、对温度和超参数敏感、情景设计有限，且未充分探讨情感与伦理风险。

---

## 174. MATRIX AS PLAN: Structured Logical Reasoning with Feedback-Driven Replanning

**arXiv ID:** 2601.10101 | [PDF](https://arxiv.org/pdf/2601.10101v1)

**作者:** Ke Chen `[一作]` (Beijing Normal University), Tian Wang `[通讯]` (Beijing Normal University)

**通讯引用:** 33513 | [OpenAlex ID](https://openalex.org/A5047476496)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MatrixCoT框架，通过将推理步骤结构化为符号化表达，并使用依赖矩阵规划与反馈重规划来实现逻辑推理；

**💡 创新点**

创新点在于将文本推理转化为可验证的矩阵计划，显式捕捉跨步骤依赖，并通过执行反馈自动修正计划，提升鲁棒性与可解释性；

**🔧 技术方法**

采用LLM驱动的符号化翻译、矩阵化规划、求解执行和反馈重规划四大模块；

**📊 数据集**

在五大逻辑推理基准（AR‑LSAT、LogicalDeduction、FOLIO、Proof‑Writer、PrOntoQA）上进行评测；

**📈 对比分析**

与标准提示、CoT、Logic‑LM、SymbCoT、Aristotle等方法对比，MatrixCoT在所有模型（GPT‑4o、GPT‑4o‑mini、Qwen2.5‑72B、DeepSeek‑V3、Kimi‑K2）上均取得最佳或并列最佳准确率，表现稳定；

**⚠️ 局限性**

局限性包括仍受基模型理解与指令遵循能力限制，无法完全替代专业符号求解器，且对格式化输出有较高依赖。

---

## 175. Mark My Works Autograder for Programming Courses

**arXiv ID:** 2601.10093 | [PDF](https://arxiv.org/pdf/2601.10093v1)

**作者:** Yiding Qiu `[一作]` (University of New South Wales), Artem Lensky `[通讯]` (University of New South Wales)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了本地化的自动批改系统 Mark My Works，结合单元测试与 LLM 生成的解释性反馈，并在 191 名学生的编程课程中试点使用，比较 AI 与人工评分。

**💡 创新点**

创新点在于将评分拆解为原子模块并通过角色化提示链实现可追溯评分、统一评卷标准；采用本地部署保证隐私、实现了高并发的流水线架构，以及提供更细粒度的自然语言反馈。

**🔧 技术方法**

主要技术包括单元测试框架、基于角色的 LLM 提示、YAML 定义的评分模块、FastAPI + React 前后端架构、PostgreSQL + MinIO 存储、Celery 消息队列，以及数据归一化和统计分析（Pearson 相关、分布比较）。

**📊 数据集**

使用了 UNSW Canberra 计算问题解决课程的 191 名学生提交的 Jupyter Notebook，重点评估 79 个完整提交（含 Assignment 2 与 Assignment 3）以及人工评分样本 184 条。

**📈 对比分析**

比较方法是将 AI 与人工评分在 79 条重叠提交上做 Pearson 相关、分布曲线对比，并对 AI 分数做 min‑max 归一化；结果显示 AI 分数更保守、方差更大，但分布形状相似且 AI 反馈细节更丰富。

**⚠️ 局限性**

局限性包括 AI 样本量小（79 条）与人工样本不匹配、评分相关性不显著、可能受提示调优影响、缺乏多语言支持、尚未完全解决 LLM 的幻觉风险，需进一步优化提示与扩展功能。

---

## 176. Bayesian Meta-Analyses Could Be More: A Case Study in Trial of Labor After a Cesarean-section Outcomes and Complications

**arXiv ID:** 2601.10089 | [PDF](https://arxiv.org/pdf/2601.10089v1)

**作者:** Ashley Klein `[一作]` (Loftus, Ryu and Bartol OBGYN), Marcia DesJardin `[通讯]` (University of Alabama at Birmingham)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究构建并应用贝叶斯层级元分析模型，评估在剖宫产史女性（TOLAC）中机械扩张与Pitocin诱导对剖宫产率及相关并发症的影响。

**💡 创新点**

创新点在于：① 将未观测的决策变量Bishop评分作为中介变量纳入模型；② 结合临床专家经验设定超先验，解决传统固定效应模型忽视因果关系导致的偏差；③ 在贝叶斯框架下提供对未知混杂因素的定量校正。

**🔧 技术方法**

使用技术包括：Numpyro + NUTS MCMC进行贝叶斯概率编程；层级贝叶斯元分析模型；与固定效应比值比（RR）和优势比（OR）模型对比；E‑value等敏感性分析。

**📊 数据集**

数据集来源于6项观察性研究，总计4037名TOLAC患者，其中机械扩张组1039人，控制组（Pitocin等）2998人，记录剖宫产率、子宫破裂、APGAR等指标。

**📈 对比分析**

比较方法：传统固定效应模型给出RR 1.39（95%CI 1.27‑1.51），贝叶斯模型得到RR 1.04（95%CI 0.93‑1.18），表明未考虑Bishop评分时会高估剖宫产风险；其他罕见并发症采用固定效应OR，结果显示两组无显著差异。贝叶斯模型在存在隐藏变量时提供更可靠的效应估计。

**⚠️ 局限性**

局限性包括：① 样本量仅6项研究，统计功效有限；② Bishop评分未直接记录，需用隐变量估计；③ 部分并发症数据缺失；④ 模型对超先验设定敏感；⑤ 仍无法完全排除所有潜在混杂。

---

## 177. Starfield: Demand-Aware Satellite Topology Design for Low-Earth Orbit Mega Constellations

**arXiv ID:** 2601.10083 | [PDF](https://arxiv.org/pdf/2601.10083v1)

**作者:** Shayan Hamidi Dehshali `[一作]` (Ohio State University), Shaileshh Bojja Venkatakrishnan `[通讯]` (Ohio State University)

**通讯引用:** 1364 | [OpenAlex ID](https://openalex.org/A5021487050)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种需求感知的星际链接拓扑设计方法 Starfield，利用基于流量几何的 Riemannian 度量来引导卫星间链路方向，并给出了静态与动态两种实现。

**💡 创新点**

创新点在于：①构造了基于流量向量场的 Riemannian 度量，使链路优先沿地面流量主方向；②通过可调超参数 K 平衡伸缩因子与跳数，实现可定制的性能折衷；③引入了“王冠”重定向策略以应对卫星覆盖边界问题；④提供了专门的轻量级链路感知仿真器验证方案。

**🔧 技术方法**

使用的技术包括 Riemannian 几何、向量场和度量学、仿真层面链路容量的 Shannon–Hartley 计算、Dijkstra 最短路径、统计与鲁棒性分析，以及 Go/C++/Python 实现的仿真器。

**📊 数据集**

使用的数据集：100 个全球主要城市的地理坐标和人口，基于距离、人口和热点的多种流量矩阵；以及 Phase 1 Starlink 星座的轨道、卫星数量、ISL 速率等参数。

**📈 对比分析**

对比方法：与 +Grid、随机、Motif 基线在相同星座与流量场景下进行，评估伸缩因子、跳数、RTT、链路使用率和抖动等指标。实验结果表明 Starfield 在多种流量分布下可实现高达 30% 的跳数减少、15% 的伸缩因子改进，静态版最高 20% 的伸缩因子提升；在 Gaussian 噪声下伸缩因子降幅不到 3%。

**⚠️ 局限性**

局限性：需要事先已知或估计流量需求，极端多方向流量时优势减弱；K 超参数需人工调优；未考虑 ISL 建立延迟与多层星座的交互；仅在单层星座上验证，未覆盖更大规模或时间变化的动态情景。

---

## 178. Thinking Like Van Gogh: Structure-Aware Style Transfer via Flow-Guided 3D Gaussian Splatting

**arXiv ID:** 2601.10075 | [PDF](https://arxiv.org/pdf/2601.10075v1)

**作者:** Zhendong Wang `[一作]` (Santa Clara University), Cihan Ruan `[通讯]` (Santa Clara University)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5016118395)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过流引导的几何推进框架，将Van Gogh的方向性笔触转化为 3D 高斯投影，实现 Post‑Impressionist 风格的三维几何抽象；

**💡 创新点**

创新点包括：mesh‑free 的流引导几何推进、亮度–结构解耦的色彩保持策略，以及利用大型多模态模型（VLM）作为“艺术评审”来衡量主观风格真实性；

**🔧 技术方法**

采用 3D Gaussian Splatting、可微渲染、2D 方向流场提取与投影反向传播、亮度解耦损失以及 VLM 评判框架；

**📊 数据集**

使用 LLFF、Tanks & Temples 等真实场景数据，并以 Van Gogh、Munch 等绘画作品作为风格参考；

**📈 对比分析**

与 ABC‑GS 等基线进行可视化对比，并用 VLM 评审与人工用户研究评估，赢率>87%，平均美学分数从 7.19 提升到 8.36，显示在几何对齐与材质表现上有显著提升；

**⚠️ 局限性**

局限性包括：在极端几何变形时可能出现颜色漂移或光照不一致；依赖可微渲染与流场估计，计算成本较高；目前仅针对 Van Gogh 等有限风格，尚未覆盖更广泛的艺术流派与复杂光照场景。

---

## 179. Unlabeled Data Can Provably Enhance In-Context Learning of Transformers

**arXiv ID:** 2601.10058 | [PDF](https://arxiv.org/pdf/2601.10058v1)

**作者:** Renpu Liu `[一作]` (University of Virginia), Jing Yang `[通讯]` (University of Virginia)

**通讯引用:** 6966 | [OpenAlex ID](https://openalex.org/A5071470775)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了增强式上下文学习（Augmented ICL）框架，在 prompt 中同时加入标记与未标记样本，并证明 transformer 通过 Chain‑of‑Thought 能实现 EM 算法以提升多类线性分类性能。

**💡 创新点**

首次理论证明在多分类线性分类任务中，利用未标记数据可显著降低 ICL 误差，并给出线性收敛的教师强迫（teacher forcing）训练方法。

**🔧 技术方法**

使用 Chain‑of‑Thought 逐步推理、EM 风格参数更新、教师强迫训练以及多层注意力+MLP transformer 架构。

**📊 数据集**

在合成 Gaussian 混合模型数据上实验，使用 3 类、d=3，采样 N=5 个标记样本与 M∈{1,10,20} 个未标记样本。

**📈 对比分析**

与传统单标签 ICL 对比，实验显示在未标记样本增加时均方误差和分类准确率均有显著提升，且实验结果与理论一致。

**⚠️ 局限性**

仅在模拟的线性高斯模型上验证，缺乏在真实自然语言或图像数据上的实验与泛化证明。

---

## 180. Disentangled Concept Representation for Text-to-image Person Re-identification

**arXiv ID:** 2601.10053 | [PDF](https://arxiv.org/pdf/2601.10053v1)

**作者:** Giyeol Kim `[一作]`, Chanho Eom `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

介绍了 Elsevier 期刊用的 LaTeX 文档类 elsarticle.cls 的功能、使用方法与配置选项。

**💡 创新点**

主要创新点在于基于 article.cls 重新编写，提供预印本和期刊版式、简化环境配置、兼容 natbib、geometry、graphicx 等常用包，减少与其它宏包冲突。

**🔧 技术方法**

使用了 LaTeX、article.cls、natbib、geometry、graphicx、fleqn、txfonts（可选）以及 hyperref、endfloat 等包，支持数学环境和定理环境的快捷定义。

**📊 数据集**

本文件为文档类说明，并未使用任何实验数据集。

**📈 对比分析**

通过与旧版 elsart.cls 的对比（功能、兼容性、排版风格）展示其优势；性能上在多种期刊模板（1p、3p、5p 等）中保持良好的排版质量与兼容性。

**⚠️ 局限性**

局限性包括：仅适用于 Elsevier 期刊，需安装对应的 LaTeX 包，且对非 Elsevier 期刊的排版支持有限。

---

## 181. Following the Teacher's Footsteps: Scheduled Checkpoint Distillation for Domain-Specific LLMs

**arXiv ID:** 2601.10114 | [PDF](https://arxiv.org/pdf/2601.10114v1)

**作者:** Cheng Feng `[一作]` (Fujitsu R&D Center), Yusuke Oishi `[通讯]` (Fujitsu Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于理论分析的领域专用LLM知识蒸馏框架，即Scheduled Checkpoint Distillation（SCD）配合Sample-wise Adaptive Weighting（AW），实现学生模型在特定任务上匹配甚至超过教师模型。

**💡 创新点**

创新点在于：①将教师-学生性能差距拆分为学生优越子域(SFS)与教师优越子域(TFS)，并给出学生能超越教师的充分条件；②基于此理论设计了最优教师检查点调度策略，动态选取高性能且与学生当前状态相近的中间检查点；③引入AW机制，按样本级别权重调整蒸馏与监督损失，提升SFS优势并减小TFS劣势。

**🔧 技术方法**

技术包括：①理论推导的风险分解与子域划分；②基于KL散度的检查点选择指标；③自适应权重公式 AW(样本)=σ(logℒ_S/ℒ_T)；④在Llama系列模型上实现的两阶段训练（SFT+SCD+AW）。

**📊 数据集**

使用PubmedQA（英文）与JMED-LLM（日文）等多任务数据集，涵盖QA、NER、文本分类等。

**📈 对比分析**

与传统TD、TAID、CD等蒸馏方法对比，SCD在多任务上平均提升0.742，SCD+AW进一步提升到0.763，并在NRNER、JMMLU等任务中实现学生模型超越教师的效果。

**⚠️ 局限性**

局限性包括：①AW权重计算依赖教师SFT与学生SFT的输出，可能对训练数据分布差异敏感；②SCD的检查点调度仍需保存大量中间模型，存储成本高；③在教师本身优势极强的任务（如JMMLU、PubmedQA）中难以实现学生超越。

---

## 182. SIN-Bench: Tracing Native Evidence Chains in Long-Context Multimodal Scientific Interleaved Literature

**arXiv ID:** 2601.10108 | [PDF](https://arxiv.org/pdf/2601.10108v1)

**作者:** Yiming Ren `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**通讯引用:** 3782 | [OpenAlex ID](https://openalex.org/A5020953714)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Fish-in-the-Ocean（FITO）范式及 SIN-Bench 评测体系，用以衡量多模态大型语言模型在长篇科学论文中的跨模态证据链推理能力。

**💡 创新点**

创新点在于：①从证据链视角评估推理过程，而非仅答题；②引入“No Evidence, No Score”原则强制模型提供可追溯证据；③构建保留原文交错结构的 Scientific INterleaved（SIN）语料库。

**🔧 技术方法**

技术包括多模态预训练模型（Gemini、GPT、Qwen 等）与跨模态合成生成；利用 LLM 进行证据链的匹配、相关性、逻辑三维评估；实现文本‑图像交错解析流水线。

**📊 数据集**

数据集来自 arXiv 与 PMC 共 4,000 篇科研论文，整理为 SIN 格式；再通过迭代生成 3,200 条候选样本，最终发布 490 条包含 Find、QA、Summary、Verify 四任务的评测集。

**📈 对比分析**

对八款主流 MLLM 进行比较，Gemini‑3‑Pro 在整体得分 0.566 最高；GPT‑5 在 SIN‑QA 答题准确率 0.767 领先，但证据链得分相对较低；Qwen3‑VL‑8B 在多项指标上优于其 30B MoE 版本，表明推理训练密度更关键。

**⚠️ 局限性**

局限包括：对长文本交错输入的支持有限，导致部分模型无法参与；严格的质量筛选剔除了部分含有细微错误的文档；开放权重模型难以满足结构化输出要求，导致评测失效。

---

## 183. When Personas Override Payoffs: Role Identity Bias in Multi-Agent LLM Decision-Making

**arXiv ID:** 2601.10102 | [PDF](https://arxiv.org/pdf/2601.10102v1)

**作者:** Viswonathan Manoranjan `[一作]` (Society-Centered AI Lab), Snehalkumar `Neil' S. Gaikwad `[通讯]` (Society-Centered AI Lab)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在四种大型语言模型上进行 2×2 因子实验，系统评估了角色身份（persona）与收益信息可见性对多智能体决策的影响，利用 Nash 均衡率作为诊断指标探讨 LLM 在战略推理中的行为；

**💡 创新点**

创新点在于首次将角色身份偏差与收益可见性交叉研究，揭示两者交互决定系统是否能实现收益优化，并发现模型架构差异导致的策略推理可变性；

**🔧 技术方法**

技术手段包括：使用 chain‑of‑thought 生成推理过程；构造四角色（工业家、政府、环保人士、普通市民）的同一轮决策游戏；在情境文本中嵌入显式或隐式收益矩阵；统计 Nash 均衡率和 equilibrium 选取模式；

**📊 数据集**

数据集由 53 个环境决策情境组成，情境文本由 GPT‑5‑mini 生成并人工审核，覆盖经济、环境和混合三类场景；

**📈 对比分析**

比较方法：对不同模型（Qwen‑7B/32B、Llama‑8B、Mistral‑7B）在四种实验条件（persona 与否，收益可见性）下分别统计 Nash 率和 equilibrium 选取比例；结果显示：有 persona 时 Nash 率低至 0‑6.7%，移除 persona 并提供显式收益后 Qwen 模型可达 65‑90%；其他模型无论条件均维持 0 或 100% Green Transition；

**⚠️ 局限性**

局限性包括：仅使用 7B‑32B 参数模型，未覆盖更大模型和其他家族；收益矩阵呈现方式可能过于繁复；情境仅限环境政策领域，其他游戏结构或多 agent 数量可能产生不同效果；未来需探究更简洁的收益呈现、更多模型家族以及不同领域场景。

---

## 184. LeMoF: Level-guided Multimodal Fusion for Heterogeneous Clinical Data

**arXiv ID:** 2601.10092 | [PDF](https://arxiv.org/pdf/2601.10092v1)

**作者:** Jongseok Kim `[一作]` (Chungbuk National University), Ohyun Jo `[通讯]` (Chungbuk National University)

**通讯引用:** 1677 | [OpenAlex ID](https://openalex.org/A5042402971)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了 LeMoF 框架，在每种模态内提取层级表示并通过层级重要性评估与最佳层级聚合，结合跨模态注意力实现对 ICU 病人住院时长（LOS）的多模态预测。

**💡 创新点**

创新点在于将模态内部的多层级特征显式分离、利用 Shapley 值自动挑选最具预测价值的层级，并将此层级作为跨模态注意力的关键锚点，从而减少信息丢失并提升模型稳健性。

**🔧 技术方法**

采用金字塔特征网络（PFN）提取多层级特征，层级预测头与逻辑回归元学习器进行聚合，Shapley 重要性评估挑选最佳层级，交叉模态注意力模块实现模态间信息交换，配合 WaveNet、FT‑Transformer 等专用编码器。

**📊 数据集**

使用 MIMIC‑IV 与 MIMIC‑IV‑ECG 两大公共 ICU 数据集，包含电子病历、时间序列心电图等多模态数据。

**📈 对比分析**

与 HyperFusion、DrFuse、MMTM、HyperFusion 等多种现有多模态融合方法在多种编码器组合下进行对比，LeMoF 在 ACC 与 AUROC 上均超过对手，平均排名第一，显示出更优越且更稳定的预测性能。

**⚠️ 局限性**

局限性包括：对缺失或异常模态的鲁棒性尚待提升；仅在 ECG 与 EHR 两种模态上验证，未扩展到影像等更多模态；模型包含多层级特征提取与注意力计算，计算成本相对较高。

---

## 185. Adaptive Label Error Detection: A Bayesian Approach to Mislabeled Data Detection

**arXiv ID:** 2601.10084 | [PDF](https://arxiv.org/pdf/2601.10084v1)

**作者:** Zan Chaudhry `[一作]` (Harvard University), Haris I. Sair `[通讯]` (Johns Hopkins University)

**通讯引用:** 4382 | [OpenAlex ID](https://openalex.org/A5070419330)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种名为 Adaptive Label Error Detection (ALED) 的数据集清洗框架，利用深度卷积网络中间特征空间的去噪、降维和高斯建模，通过似然比检验识别标签错误。

**💡 创新点**

创新点在于：①将中间特征空间投影至均值差方向并采用随机投影与 MCD 协方差估计构造鲁棒高斯模型；②使用似然比阈值而非仅靠预测置信度，显著提升了误标检测的灵敏度；③针对二分类任务提供了可直接落地的 Python 包实现。

**🔧 技术方法**

技术细节包括：深度 CNN（DenseNet121/ResNet50）特征提取、平均池化、随机投影、最小协方差矩阵估计 (MCD)、多维高斯似然计算、似然比阈值判定、贝叶斯后验评估。

**📊 数据集**

实验数据集涵盖 MedMNIST 4 个二分类子集（PneumoniaMNIST、BreastMNIST、RetinaMNIST、BloodMNIST），并在 5%/10%/20% 等不同误标率下进行评估。

**📈 对比分析**

与 CleanLab（CL）及其 Feature 版进行对比；在所有指标（灵敏度、F1、AUPRC、PPV）上，ALED 的灵敏度提升 2–3 倍，保持相近的 PPV；在清洗后重新训练模型时，ALED 能让测试错误率下降 33.8%，明显优于 CL 的 13.5% 与 6.2%。

**⚠️ 局限性**

局限性包括：仅适用于二分类（多分类需一对多拆分）；对特征空间近似高斯的假设敏感，未预训练网络时性能下降；在数据量极低或标注噪声极大时仍面临挑战。

---

## 186. Efficient Content-based Recommendation Model Training via Noise-aware Coreset Selection

**arXiv ID:** 2601.10067 | [PDF](https://arxiv.org/pdf/2601.10067v1)

**作者:** Hung Vinh Tran `[一作]` (University of Queensland), Hongzhi Yin `[通讯]` (University of Queensland)

**通讯引用:** 16608 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种噪声感知的核心子集（NaCS）框架，专门用于内容推荐系统的高效训练。

**💡 创新点**

创新点在于将子模优化、逐步标签自纠正和Monte Carlo Dropout联合使用，使核心子集在保留模型质量的同时显著降低噪声影响。

**🔧 技术方法**

采用子模优化求解核心子集、Adam梯度近似、逐步软标签修正以及MC Dropout不确定性量化等技术。

**📊 数据集**

在Criteo、Avazu、KDD三大大规模CTR数据集以及文本推荐数据集上进行实验。

**📈 对比分析**

与现有核心子集方法和完整数据训练进行对比，NaCS仅使用1%训练样本即可恢复93–95%性能，并在训练速度和资源消耗上优于对手。

**⚠️ 局限性**

局限性包括对梯度估计的依赖、MC Dropout计算开销以及在极端噪声环境下对标签自纠正的稳定性待进一步验证。

---

## 187. CoF-T2I: Video Models as Pure Visual Reasoners for Text-to-Image Generation

**arXiv ID:** 2601.10061 | [PDF](https://arxiv.org/pdf/2601.10061v1)

**作者:** Chengzhuo Tong `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14319 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CoF-T2I模型，将预训练的视频生成骨干作为纯视觉推理器，通过三步链式帧推理实现文本到图像的逐帧递进生成；

**💡 创新点**

创新点在于把视频模型的Chain‑of‑Frame推理机制迁移到T2I任务中，采用独立帧编码避免运动伪影，并结合专门构建的三帧推理轨迹数据集CoF‑Evol‑Instruct实现可解释的视觉推理；

**🔧 技术方法**

核心技术包括Wan2.1视频生成模型、Rectified Flow流匹配训练、基于VAE的独立帧编码、统一编辑原语（UEP）与多模型评估器相结合的构建流水线；

**📊 数据集**

使用自构建的64K三帧CoF‑Evol‑Instruct数据集作为训练集，并在GenEval与Imagine‑Bench基准上进行评测；

**📈 对比分析**

与标准图像模型、统一多模态模型和视频模型进行对比，CoF‑T2I在GenEval上得到0.86的最高分，在Imagine‑Bench上取得7.468的总分，显著优于基线并通过消融实验验证独立编码与中间监督的有效性；

**⚠️ 局限性**

局限性包括对视频骨干的依赖、仅支持固定三步推理、对长序列推理和极端多样化场景的适应性仍待验证，以及构建数据集需要较多人工验证和资源。

---

## 188. Resistive Memory based Efficient Machine Unlearning and Continual Learning

**arXiv ID:** 2601.10037 | [PDF](https://arxiv.org/pdf/2601.10037v1)

**作者:** Ning Lin `[一作]` (University of Hong Kong), Zhongrui Wang `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种硬件-软件协同设计，利用混合模拟/数字电路和低秩适配（LoRA）实现基于非易失性电阻存储器（RM）的计算内存（CIM）加速器上的机器忘记（unlearning）和持续学习（continual learning），满足边缘设备的隐私与持续适应需求。

**💡 创新点**

创新点在于：1）采用混合模拟/数字架构，将预训练权重存放在高密度RM交叉点阵，轻量级LoRA权重驻留在SRAM数字计算单元；2）通过LoRA仅更新少量低秩矩阵，避免频繁、能耗高且误差大的RM写操作；3）将机器忘记和持续学习的算法映射到RM-CIM上，首次在同一平台实现高效隐私保护与自适应推理。

**🔧 技术方法**

技术主要包括：非易失性电阻存储器（1T1R TiN/Ta₂O₅/TaOₓ/TiN结构）交叉点阵、SRAM数字计算单元、LoRA低秩适配、梯度上升和标签混淆的机器忘记方法、基于回放的持续学习框架；此外采用了 32×32 的RM芯片和基于 Zynq SoC 的混合电路板。

**📊 数据集**

使用的数据集包括：面部识别的 Olivetti faces、语音身份验证的 Spiking Speech Commands、以及条件扩散模型的 UnlearnCanvas（艺术风格生成）。

**📈 对比分析**

与全参数微调在RM上以及GPU基线相比，RM‑DLoRA 在训练更新次数上减少 25–147 倍、写入能耗下降 63–388 倍、推理能耗降低 6–48 倍，同时保持或略高于传统方法的准确率（面部识别、语音识别及图像生成任务）。

**⚠️ 局限性**

局限性包括：1）目前仅在 32×32 交叉点阵规模实验，需验证更大规模可扩展性；2）对 RM 器件的写入误差、耐久性和温度漂移仍有一定影响；3）LoRA 需要预先确定低秩维度，对模型架构差异要求较高；4）混合架构增加硬件设计复杂度，可能限制商业化落地。

---

## 189. PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization

**arXiv ID:** 2601.10029 | [PDF](https://arxiv.org/pdf/2601.10029v1)

**作者:** Tingyue Pan `[一作]` (University of Science and Technology of China), Qi Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 28232 | [OpenAlex ID](https://openalex.org/A5100453264)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自主学术论文检索代理PaperScout，将检索过程视为多轮决策，动态决定调用检索工具。

**💡 创新点**

创新点在于将检索视为POMDP并设计Proximal Sequence Policy Optimization (PSPO) 以解决token级与序列级奖励不匹配的问题，使代理能够根据上下文自适应扩展搜索策略。

**🔧 技术方法**

技术包括LLM驱动的工具调用、基于POMDP的状态与观察建模、PSPO（基于GAE的序列级优势估计与裁剪策略）以及价值函数预训练与归一化。

**📊 数据集**

使用RealScholarQuery与AutoScholarQuery两组检索基准，其中包含真实学术查询和人工构造的查询集。

**📈 对比分析**

与Google Search、Google Scholar、PaSa、SPAR等单轮或固定工作流方法对比，PaperScout在Recall、F1、LLM-score等指标上均领先，例如Recall提升至0.574（基线0.541），LLM-score提升至2.576。

**⚠️ 局限性**

局限性包括仅在计算机科学领域验证、依赖公开可检索文献、后端检索覆盖有限、仅使用出向引用扩展，未考虑付费资源和入向引用信息。

---

## 190. Fundamental Limits of Coded Polynomial Aggregation

**arXiv ID:** 2601.10028 | [PDF](https://arxiv.org/pdf/2601.10028v1)

**作者:** Xi Zhong `[一作]` (University of Florida), Mingyue Ji `[通讯]` (University of Florida)

**通讯引用:** 3351 | [OpenAlex ID](https://openalex.org/A5058487273)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究在分布式计算系统中，如何通过编码多项式聚合(CPA)方法实现加权多项式评估的聚合，并给出了最小响应数和可行性条件。

**💡 创新点**

创新点在于提出针对不可进行个体解码的 CPA 的必要充分正交条件，推导出最小响应数的理论下界，并给出可构造的实现方案，显著降低所需响应数。

**🔧 技术方法**

采用代数几何、矩阵秩与范德蒙德矩阵分析、正交条件推导、多项式根分解以及 Zariski 开放集论证等技术。

**📊 数据集**

本工作为理论分析，不涉及具体数据集，讨论的是通用的矩阵与多项式情形。

**📈 对比分析**

与传统基于个体解码的 Lagrange 编码方案相比，提出的 CPA 在任意 K、d 下的最小响应数为 ⌊(K-1)/2⌋+1（d=1）或 (d-1)(K-1)+1（d≥2），比原来所需的 d(K-1)+1 远小，显示出更高的资源利用效率。

**⚠️ 局限性**

局限性包括：仅在理想无误差环境下适用；要求数据点满足通用性（generic）条件，有限域或非通用点时可能不成立；未讨论通信开销、实际实现复杂度和对随机噪声的鲁棒性。

---

## 191. STCRank: Spatio-temporal Collaborative Ranking for Interactive Recommender System at Kuaishou E-shop

**arXiv ID:** 2601.10027 | [PDF](https://arxiv.org/pdf/2601.10027v1)

**作者:** Boyang Xia `[一作]` (Kuaishou Technology), Wenwu Ou `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在抖音短视频电商平台快手E‑shop上设计并实现了一套交互式推荐系统（IRS），提出了STCRank框架，分别在同一推荐槽内实现多目标协同（MOC）与多槽位协同（MSC），以提升转化率、日活与用户停留深度。

**💡 创新点**

创新点包括：
- 针对全屏沉浸式UI下的“视图‑转化‑下滑”三目标相互干扰，提出通过阈值分段与样本加权来消除重叠与冲突，推进Pareto前沿；
- 针对多槽位序列化的贪心陷阱，设计跨阶段与单阶段两阶段look‑ahead排序机制，利用先行下滑率预测和序列化评估，达成全局最优；
- 将两阶段IRS与传统推荐无缝对接，避免额外交互成本，提升用户体验。

**🔧 技术方法**

技术包括：
- 多门混合专家（MMOE）模型预测转化率、视图通过率（vtr）与下滑率（sdr）；
- 对vtr采用5秒阈值分段，对sdr采用首位正样本与冲突样本过滤；
- 多目标加权线性组合与贝叶斯优化调参；
- 跨阶段look‑ahead标签构造（ctr·sdr*·cvr*）；
- 单阶段序列重排序：Beam搜索生成排列，使用累计曝光概率和折扣值评估序列。

**📊 数据集**

数据集为快手E‑shop内部业务日志，涵盖数千万用户在E‑stage（探索）与F‑stage（聚焦）中的点击、停留、下滑与购买行为。没有公开公开的数据集。

**📈 对比分析**

通过7天A/B测试，将各模块与基线逐步叠加，最终STCRank在E+F阶段实现：
- 购买率提升9.65%；
- 交易IPV提升1.55%；
- DAU提升0.03%（在大规模用户基数下仍具统计意义）。
对比实验显示，MOC对IPV/DAU提升≈6%/0.5%，MSC对购买率提升≈3%/2%，两者叠加后实现整体最大收益。

**⚠️ 局限性**

局限性：
- 关键阈值（如5秒vtr）需经验调参，可能在不同商品或地区不适用；
- 计算成本相对较高，尤其是Beam搜索和多阶段预测；
- 只针对全屏沉浸式UI设计，无法直接迁移到传统布局；
- 仅在快手E‑shop环境验证，缺乏跨行业或跨平台的泛化评估。

---

## 192. Optimal Proximity Gap for Folded Reed--Solomon Codes via Subspace Designs

**arXiv ID:** 2601.10047 | [PDF](https://arxiv.org/pdf/2601.10047v1)

**作者:** Fernando Granha Jeronimo `[一作]` (University of Illinois), Pranav Rajpal `[通讯]` (University of Illinois)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文证明了 Folded Reed–Solomon（FRS）码在容量级别下具有(δ,ε)接近性间隙，即对于任意码率 R 与任意小的 η>0，都存在 δ=1-R-η 与 ε=O((n/η+1/η^3)/q) 使得任何低维仿射子空间满足全点都近或几乎全点远的两种情况。

**💡 创新点**

创新点在于把 RS 码的接近性间隙理论推广到容量级别的 FRS 码，并提出了“线性拼接（line stitching）”技术与子空间设计相结合的剪枝列表解码结构，首次实现了容量级别下的接近性间隙证明。

**🔧 技术方法**

主要技术包括子空间设计理论、Folded‑Wronskian 分析、线性拼接与剪枝列表解码、局部到全局的平均化与几何组合论，以及有限域线性代数工具。

**📊 数据集**

论文不涉及实验数据集，所有结论均为理论证明；参数包括折叠参数 m、码长 n、符号域大小 q、码率 R 以及误差半径 δ、误差率 ε。

**📈 对比分析**

与以往仅在 Johnson 半径内得到接近性间隙的 Ben‑Sasson 等工作相比，本文在半径达到容量上限，误差 ε 随 q 缩小，理论上实现了更强的局部到全局一致性约束，未给出实验对比。

**⚠️ 局限性**

局限性包括需要足够大的符号域 q（多项式级别）和折叠参数 m≥c/η^2，且误差率 ε 与 1/q 成正比，实际常数取值和最优性尚待进一步研究。

---

## 193. ELITE: Efficient Gaussian Head Avatar from a Monocular Video via Learned Initialization and TEst-time Generative Adaptation

**arXiv ID:** 2601.10200 | [PDF](https://arxiv.org/pdf/2601.10200v1)

**作者:** Kim Youwang `[一作]` (POSTECH), Tae-Hyun Oh `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于学习初始化的高效高逼真高可动画化 Gaussian 头部化身合成方法。

**💡 创新点**

融合 3D 数据先验与 2D 生成先验，利用 Mesh2Gaussian Prior Model 进行快速初始化并通过单步扩散增强在测试时生成高质量视角与表情补全，解决先前方法在泛化、身份保持与速度方面的不足。

**🔧 技术方法**

使用 FLAME 表情驱动的 U‑Net 生成 2D Gaussian，结合单步 Diffusion Enhancer（基于 SD‑Turbo）对渲染结果进行细节补全，并辅以测试时自监督的图像增强与几何正则化。

**📊 数据集**

在 NerSemble‑V2 上训练多样化 334 个身份与表情，测试使用 INSTA 的野外单目视频。

**📈 对比分析**

与 FlashAvatar、SplattingAvatar、SynShot、CAP4D 等方法在自/交叉重映射下对比，PSNR/SSIM/LPIPS/CSIM 上均优于竞争者，且生成速度比 CAP4D 提升 60 倍、接近 overfitting 方法。

**⚠️ 局限性**

对极端光照与材质敏感，当前仅覆盖头部区域，未处理衣物或配件等全身/附件的生成。

---

## 194. HUMANLLM: Benchmarking and Reinforcing LLM Anthropomorphism via Human Cognitive Patterns

**arXiv ID:** 2601.10198 | [PDF](https://arxiv.org/pdf/2601.10198v1)

**作者:** Xintao Wang `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**通讯引用:** 3918 | [OpenAlex ID](https://openalex.org/A5090455375)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于244种心理模式的多角色对话情境数据集，并通过监督微调训练LLM，使其在角色扮演中能够真实呈现多模式交互的心理行为。

**💡 创新点**

首次将人格特质与社会‑认知模式视为相互作用的因果力量，并引入双层检查表评估单模式与多模式动态，突破传统单标签“人格幻觉”的局限。

**🔧 技术方法**

使用Gemini、Claude等大语言模型合成情境与对话，基于Qwen3‑8B/32B进行监督微调，采用GPT‑5‑mini评判器实现双层I PE和MPD指标评估。

**📊 数据集**

约12,000篇学术论文用于提炼244种模式，生成11,359个包含2–6角色、2–5模式的情境，并合成约30,000条对话样本（含OpenThoughts与CoSER混合）。

**📈 对比分析**

与闭源（GPT‑5、Claude、Gemini）及开源（Qwen3、DeepSeek）模型在自建的IPE和MPD指标及LifeChoice、CroSS‑MR基准上对比；-32B在IPE 32.6%、MPD 73.8%均超越同规模开源基线，MPD表现尤为突出。

**⚠️ 局限性**

评估依赖GPT‑5‑mini判别器可能带来系统偏差；数据主要基于WEIRD理论，文化适应性受限；合成对话缺乏真实交互细节；安全风险在于可模拟负面人格与操纵机制。

---

## 195. From Physical Degradation Models to Task-Aware All-in-One Image Restoration

**arXiv ID:** 2601.10192 | [PDF](https://arxiv.org/pdf/2601.10192v1)

**作者:** Hu Gao `[一作]` (Shanghai Jiao Tong University), Lizhuang Ma `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 7878 | [OpenAlex ID](https://openalex.org/A5084218062)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于物理退化建模的全能图像恢复框架OPIR，采用两阶段结构预测任务感知逆退化算子并通过不确定性地图实现精细化修复。

**💡 创新点**

创新点在于不通过提示或大模型，而是直接预测可变尺度、任务感知的逆算子；同时利用不确定性感知指导两阶段恢复，显著降低系统复杂度并提升恢复质量。

**🔧 技术方法**

核心技术包括基于U‑Net的核预测网络（KPN）、轻量级任务感知模块（TAM）、多尺度膨胀卷积、两阶段逆算子推断与不确定性估计。

**📊 数据集**

使用雨、雪、雾三种任务的公开基准：Deraining（Rain100H/L、Test100、Test1200）、Desnowing（Snow100K、SRRS、CSD）与Dehazing（RESIDE/INDOOR/OUTDOOR、SOTS），并在所有任务混合数据上进行全能训练。

**📈 对比分析**

与多类基线（AdaIR、VLU‑Net、Perceive‑IR、Defusion 等）对比，OPIR 在全能评估中平均 PSNR 提升约0.76 dB、任务对齐提升约1.08 dB，且推理时间仅为 0.174 s（FLOPs 47 G），显著领先。

**⚠️ 局限性**

局限性在于仍需预先定义并训练特定退化类型，面对完全未知或极端混合退化时可能性能下降；且模型对极大尺度结构的恢复效果尚待进一步验证。

---

## 196. How does downsampling affect needle electromyography signals? A generalisable workflow for understanding downsampling effects on high-frequency time series

**arXiv ID:** 2601.10191 | [PDF](https://arxiv.org/pdf/2601.10191v1)

**作者:** Mathieu Cherpitel `[一作]`, Anna Kononova `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套系统工作流程，用以评估降采样对针刺肌电图（nEMG）信号形态、分类性能及计算成本的影响。

**💡 创新点**

将形态失真指标与机器学习分类结果结合，构建距离-性能排名模型，可预测最优降采样配置，并首次量化不同降采样算法在保持诊断信息时的速度与准确性权衡。

**🔧 技术方法**

基于tsfresh特征提取、Boruta特征选择、随机森林分类、XGBoost排名模型、SHAP解释以及多种距离度量（如RMSE、Pearson包络相关、频谱欧氏距离等）。

**📊 数据集**

使用公开的EMGLAB针刺肌电图数据集（22名患者，共约46 k段，采样率23 kHz，包含正常、肌病、ALS三类）。

**📈 对比分析**

通过10折交叉验证对各降采样因子和算法进行比较，发现LTTB/MinMaxLTTB在因子≤30时可实现约60倍特征提取加速，且分类准确率与原始数据无显著差异；Decimate、M4在因子>10时性能显著下降。

**⚠️ 局限性**

仅针对单一数据集与单一采样率，缺乏对不同采样率、不同病理分布及真实临床噪声的泛化评估；此外未验证降采样后信号对临床医师可解释性的影响。

---

## 197. CtD: Composition through Decomposition in Emergent Communication

**arXiv ID:** 2601.10169 | [PDF](https://arxiv.org/pdf/2601.10169v1)

**作者:** Boaz Carmeli `[一作]` (Technion Israel Institute of Technology), Yonatan Belinkov `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种两阶段训练方法——Composition through Decomposition（CtD），通过在多目标协调游戏中学习离散代码簿来先分解图像为基本概念，然后在描述阶段将这些概念组合成新的短语，从而使人工神经网络在零样本场景下也能生成组合性强的描述。

**💡 创新点**

创新点主要包括：① 将多目标游戏与离散代码簿结合以显式实现概念分解；② 通过两步训练（先分解后组合）显著提升组合性，甚至在无进一步训练的零样本设置下保持完美性能；③ 证明多目标游戏是实现组合性不可或缺的前置条件。

**🔧 技术方法**

技术方法包括：多目标协作游戏（sender/receiver）、基于VQ‑VAE的离散代码簿、带有任务损失和代码簿损失的多损失优化、对比传统的 Gumbel‑softmax（GS）和量化（QT）通信方案。

**📊 数据集**

使用了五个数据集：Synth100K（100K 合成对象）、ImageNet‑like 彩色物体数据集、两位手写数字组合集、真实多物体带字幕集、QR‑code 合成数据集，用以分别评估组合性与非组合性。

**📈 对比分析**

与 GS、QT 基线在相同任务和参数下对比，CtD 在所有组合性数据集上均实现 100% 的准确率与 AMI/POS/BOS/CI/CBM 等指标；零样本 CtD 的表现与额外训练相当甚至更好；在非组合数据集上指标显著下降，表明方法专门针对组合性。

**⚠️ 局限性**

局限性包括：① 只能预设固定的消息长度，无法自动学习可变长度表达；② 需要预先定义概念集，未解决概念自学习；③ 对非组合数据无优势；④ 多目标游戏的设计与标注成本较高；⑤ 结果主要在合成/控制实验环境中验证，实际复杂视觉任务仍需进一步验证。

---

## 198. Credit C-GPT: A Domain-Specialized Large Language Model for Conversational Understanding in Vietnamese Debt Collection

**arXiv ID:** 2601.10167 | [PDF](https://arxiv.org/pdf/2601.10167v1)

**作者:** Nhung Nguyen Thi Hong `[一作]`, Tri Le Ngoc `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并训练了一款专为越南债务催收对话而定制的7B参数大语言模型Credit C-GPT，实现情绪识别、意图检测、通话阶段分类及槽位提取等多任务统一推理。

**💡 创新点**

将多任务对话分析合并为单一推理模型，并通过领域特定的模拟对话语料进行监督指令调优，实现对越南口语多轮对话的长距离推理和情绪动态把握。

**🔧 技术方法**

基于Qwen2.5-7B Instruct解码器架构，采用指令微调、QLoRA低秩适配、4-bit量化以及结构化JSON输出的生成式多任务学习。

**📊 数据集**

采用约17000个越南债务催收对话的模拟语料（约336,935轮对话），并以真实生产呼叫的400通对话为保留测试集。

**📈 对比分析**

与基于BERT的多组件管道、未做领域微调的Qwen2.5-7B以及提示式GPT-5比较，Credit C-GPT在情绪、意图、通话阶段等分类任务上平均精度提升至0.86，槽位提取实体精度提升至约0.90，尽管在绝对分数上略低于GPT-5，但在规模、隐私和部署成本上更具优势。

**⚠️ 局限性**

受限于专有领域数据、缺乏公开基准、对提示敏感且仍可能产生幻觉，部署在单GPU 4-bit模式下吞吐受限，且未评估跨域泛化或长期遗忘问题。

---

## 199. Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment

**arXiv ID:** 2601.10160 | [PDF](https://arxiv.org/pdf/2601.10160v1)

**作者:** Cameron Tice `[一作]` (Geodesic Research), Kyle O'Brien `[通讯]` (Geodesic Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究预训练语料中关于 AI 的讨论如何因果影响 LLM 的对齐倾向，并通过在预训练阶段上采样正向或负向 AI 讨论来验证自我实现的对齐/非对齐假设。

**💡 创新点**

首次系统性检验预训练数据中的 AI 对齐语料对模型行为的因果影响，提出“对齐预训练”作为补充传统后训练对齐技术的全栈策略。

**🔧 技术方法**

训练 6.9B 参数的 decoder‑only LLM，采用预训练、midtraining 以及 SFT+ DPO 组合，使用人工生成的对齐/非对齐 AI 讨论文本进行数据上采样。

**📊 数据集**

使用约 500B 通用 Web 文本、50B 长文本 + 1B 多选问答、2.8K+ AI 安全情景问答数据，以及 11B token 的合成 AI 讨论（正向和负向），并对两种版本（过滤 vs 未过滤）进行实验。

**📈 对比分析**

通过 4,174 个单轮对齐倾向问答、MMLU、ARC‑Easy、GSM‑8K、IF‑Eval 等基准进行对比；对齐上采样将对齐失败率从 45% 降至 9%，但通用能力平均下降 2–4%。

**⚠️ 局限性**

局限性包括：模型规模小、对齐评测过于简化、后训练仅采用 SFT+DPO、未覆盖更先进的对齐方法、缺乏多随机种子实验、对齐干预的可扩展性和规模化效果仍未知。

---

## 200. MMPG: MoE-based Adaptive Multi-Perspective Graph Fusion for Protein Representation Learning

**arXiv ID:** 2601.10157 | [PDF](https://arxiv.org/pdf/2601.10157v1)

**作者:** Yusong Wang `[一作]` (Guangdong Institute of Intelligence Science and Technology), Prayag Tiwari `[通讯]` (Halmstad University)

**通讯引用:** 10648 | [OpenAlex ID](https://openalex.org/A5082046789)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建多视角蛋白质图谱并通过 Mixture of Experts 融合，提升蛋白质表示学习。

**💡 创新点**

创新点在于从物理、化学、几何三角度构建图并用 MoE 动态路由实现多层交互融合。

**🔧 技术方法**

使用图神经网络、Mixture of Experts、知识基础势能、余弦相似度等技术。

**📊 数据集**

在四个下游任务（Fold、Reaction、GO、EC）数据集上进行评估。

**📈 对比分析**

与多种基线比较，MMPG 在大多数任务上均优于单视角和传统方法，性能提升显著。

**⚠️ 局限性**

局限包括对超参数敏感、专家数量选择需经验，且对极端遮蔽仍有性能下降。

---

## 201. LOOKAT: Lookup-Optimized Key-Attention for Memory-Efficient Transformers

**arXiv ID:** 2601.10155 | [PDF](https://arxiv.org/pdf/2601.10155v1)

**作者:** Aryan Karmore `[一作]` `[通讯]` (Indian Institute of Information Technology), Aryan Karmore (Indian Institute of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LOOKAT方法，利用产品量化和非对称距离计算压缩KV缓存，消除键的去量化过程，显著降低内存带宽需求。

**💡 创新点**

创新点在于把向量检索中的产品量化和非对称距离迁移到Transformer注意力中，实现64×压缩且保持95%+输出保真度，同时无需改动网络结构或再训练。

**🔧 技术方法**

使用了产品量化（PQ）、非对称距离计算（ADC）、预计算查表、FP16输出和秩相关分析等技术。

**📊 数据集**

在GPT‑2自回归层上使用自然语言、Python源码、技术文档三类文本，序列长度从128到1024，作为实验数据集。

**📈 对比分析**

与INT4/INT8等标量量化基线对比，使用余弦相似度、KL散度、Spearman秩相关、Top‑5准确率等指标；LOOKAT‑2达到64×压缩，余弦相似度≈0.957、Spearman≈0.959，明显优于INT4，并实现约10×运算速度提升、64×带宽降低。

**⚠️ 局限性**

限制在于仅压缩键而未压缩值；对代码簿质量和校准数据敏感；实现需特定硬件支持，无法一次性压缩完整KV缓存。

---

## 202. LaViT: Aligning Latent Visual Thoughts for Multi-modal Reasoning

**arXiv ID:** 2601.10129 | [PDF](https://arxiv.org/pdf/2601.10129v1)

**作者:** Linquan Wu `[一作]` (City University of Hong Kong), Jacky Keung `[通讯]` (City University of Hong Kong)

**通讯引用:** 6065 | [OpenAlex ID](https://openalex.org/A5051403641)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过教师-学生知识蒸馏，针对视觉多模态推理中的“感知缺口”，提出 LaViT 框架实现对教师视觉思维的对齐。

**💡 创新点**

创新点在于把视觉思维视为连续隐态序列进行蒸馏，并通过白盒轨迹蒸馏和课程感官门控两大机制，逼迫学生先重构教师的视觉语义与关注轨迹，再生成答案。

**🔧 技术方法**

技术包括：白盒视觉轨迹蒸馏（对齐注意力分布与视觉语义）、课程感官门控（逐步开启直接视觉通道）、自回归隐态生成、双流蒸馏损失（语义重建+轨迹对齐+下一词预测）等。

**📊 数据集**

数据集涵盖：15K 教师轨迹样本（从 Qwen2.5‑VL‑32B 提取），Visual‑CoT、MMVP、V*、BLINK、MMStar 等视觉推理基准。

**📈 对比分析**

在各基准上，LaViT‑3B 相较基线提升 5–17 %（如 MMVP 67.33 % vs. 61.33 %，BLINK Relative Depth 78.23 % vs. 76.61 %），并在多项任务上超过 7B 级别公开模型与 GPT‑4o，展示显著性能提升。

**⚠️ 局限性**

局限性在于仍需人工筛选高质量轨迹，训练成本与对教师模型的依赖限制推广；在极度多样化视觉场景下可能面临适应性挑战。

---

## 203. A Generalizable Framework for Building Executable Domain-Specific LLMs under Data Scarcity: Demonstration on Semiconductor TCAD Simulation

**arXiv ID:** 2601.10128 | [PDF](https://arxiv.org/pdf/2601.10128v1)

**作者:** Di Wang `[一作]` (Inspur Electronic Information Industry Co., Ltd), Shaohua Wu `[通讯]` (Inspur Electronic Information Industry Co., Ltd)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于模式（schema）优先的对齐框架，在数据稀缺环境下构建可执行的域专用大语言模型，示例包括半导体TCAD脚本生成（TcadGPT）及开源FEM求解器Elmer脚本生成。

**💡 创新点**

三大创新：① 通过专家文档合成1.5M问答对，构建域知识基座；② 引入中间表示（IR）+等价保持扩展，再生成DPO优先对，以直接优化指令遵循与语法可执行性；③ 对检索增强生成（RAG）进行可控实验，揭示其对已专用模型的潜在负面影响。

**🔧 技术方法**

技术组合：大规模问答生成（NLP pipeline）、LLaMA3.1 8B微调+SFT+Direct Preference Optimization（DPO），IR提取与多样化、语法检验（Sentaurus SDE、Elmer）以及可选的检索增强生成（RAG）。

**📊 数据集**

数据集：1.5M Alpaca‑style QA（Pipeline 1 34 k + Pipeline 2 1.2M）来源于TCAD手册、教材；TCAD 264‑题 benchmark；SDE 20‑instruction 语法测试；Elmer 100‑题 QA + 20‑instruction 语法测试；公开的Sentaurus、Elmer脚本用于IR与验证。

**📈 对比分析**

评估方式：与 GPT‑4o、DeepSeek V3、LLaMA 3.1 8B 等通用模型在 QA 语义准确率和工具语法通过率两维度对比；TcadGPT‑P1&2 在 QA 方面达到 85.6%；DPO 版在 Sentaurus SDE 20‑instruction 上 Pass@1 = 65%（13/20），Pass@3 = 80%（16/20）；RAG 对通用模型提升至 58–70%，但对 TcadGPT 降至 64%；Elmer 上同类实验亦显示 8B 版在 QA 上 52/100，脚本可执行率 12/20，均优于通用模型。

**⚠️ 局限性**

局限性：仅测试语法可执行性，未评估数值收敛或完整工作流程；缺乏多步规划与交互式修复；RAG 可能引入噪声导致性能下降；合成 QA 仍基于公开文档，难以覆盖更复杂或行业专属场景；模型仍可能出现幻觉或指令偏差。

---

## 204. DecisionLLM: Large Language Models for Long Sequence Decision Exploration

**arXiv ID:** 2601.10148 | [PDF](https://arxiv.org/pdf/2601.10148v1)

**作者:** Xiaowei Lv `[一作]` (Renmin University of China), Bo Zheng `[通讯]` (Alibaba Group)

**通讯引用:** 16506 | [OpenAlex ID](https://openalex.org/A5050479679)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

将轨迹视为独立模态，并与任务说明文本共同编码，构建DecisionLLM实现长序列决策生成。

**💡 创新点**

创新点在于轨迹-文本对齐机制，将轨迹作为非文本模态融入LLM，克服LLM对数值敏感度低的问题，并通过双层数据筛选提升数据质量。

**🔧 技术方法**

采用预训练的大规模LLM（Qwen2.5）、轨迹编码器、注意力融合层、线性动作预测头，并使用自回归训练方式。

**📊 数据集**

使用Maze2D-umaze-v1和AuctionNet离线数据集，结合D4RL、CORL等基准进行评估。

**📈 对比分析**

与Decision Transformer、DT-extended、RL离线基线和其他LLM提示方式对比，DecisionLLM-3B在Maze2D上提升约69.4分、在AuctionNet上提升0.085分，表现显著优于传统方法。

**⚠️ 局限性**

局限在于对离线数据质量与多样性的高度依赖，模型规模达到1.5B后提升趋于饱和，且在更大规模模型或更复杂环境中的泛化能力尚待进一步验证。

---

## 205. Actors, Frames and Arguments: A Multi-Decade Computational Analysis of Climate Discourse in Financial News using Large Language Models

**arXiv ID:** 2601.10142 | [PDF](https://arxiv.org/pdf/2601.10142v1)

**作者:** Ruiran Su `[一作]` (University of Oxford), Markus Leippold `[通讯]` (University of Zurich)

**通讯引用:** 4939 | [OpenAlex ID](https://openalex.org/A5073309846)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建并验证了一套基于大型语言模型的Actor–Frame–Argument (AFA) 提取管线，对2000–2023年 Dow Jones 财经新闻中的气候话语进行纵向分析；

**💡 创新点**

创新点在于首次将 LLM 与分层抽样、主动学习、并行评估框架相结合，实现了多维度（主体、框架、论证）纵向追踪与可复现的自动化抽取；

**🔧 技术方法**

采用 Gemini‑2.5‑flash（闭源）和 LLaMA‑4‑Maverick‑17B（开源）进行零-shot 语义抽取，并通过 Decompositional Verification Framework (DVF) 进行多模型、多评审的细粒度质量检验；

**📊 数据集**

使用包含980,061篇文章的 Dow Jones Climate News Corpus，经过分层抽样后得到4,143篇包含丰富论证特征的代表性样本；

**📈 对比分析**

与基线 RoBERTa‑Large 进行对比，LLM 在演员识别、立场、框架分类与论证抽取上平均提升约 5–15 %（如框架 F1 提升 15.2，论证 F1 提升 17.2），DVF 也显示四个维度（完整性、忠实度、连贯性、相关性）均高于基线；

**⚠️ 局限性**

局限包括仅使用单一英文新闻源、潜在的 LLM 偏见、未测量话语对实际资本流动的因果影响，以及缺乏对多源、多语言跨域普适性的评估。

---

## 206. M^4olGen: Multi-Agent, Multi-Stage Molecular Generation under Precise Multi-Property Constraints

**arXiv ID:** 2601.10131 | [PDF](https://arxiv.org/pdf/2601.10131v1)

**作者:** Yizhan Li `[一作]` (Université de Montréal), Bang Liu `[通讯]` (Université de Montréal)

**通讯引用:** 973 | [OpenAlex ID](https://openalex.org/A5100691219)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 M^4olGen 两阶段分子生成框架，先通过检索增强的多代理推理得到近似原型，再用基于片段的 GRPO 多跳优化精准满足多目标属性。

**💡 创新点**

将检索增强的原型生成与 GRPO 的片段级多跳优化结合，实现对连续数值属性的可解释精准控制，并构建了 2.95M 分子 + 1.17M 邻对的大规模推理数据集。

**🔧 技术方法**

使用 GPT‑4o / Qwen3 等大型语言模型进行推理，BRICS 片段化，Group Relative Policy Optimization（GRPO）进行多跳优化，RDKit 计算属性反馈，检索增强的多代理推理。

**📊 数据集**

构建了 2.95M 分子集合，包含 BRICS 片段标注及 1.17M 单步邻对，涵盖 QED、LogP、MW、HOMO、LUMO 等属性。

**📈 对比分析**

与多款 LLM（GPT‑4.1、Gemini‑Flash、Claude‑3 等）和图模型（STGG+、Graph GA）对比，M^4olGen 在 QED/LogP/MW 任务中 3‑hop GPT‑4o 归一化总误差仅 0.146，显著优于商业模型和图模型；在 HOMO/LUMO 任务中亦取得最低总误差 0.155。

**⚠️ 局限性**

依赖计算属性估计、仅评估有限属性集、深度跳数越大计算成本上升、未覆盖更真实实验室属性或更广泛目标空间。

---

## 207. Fairness Driven Multi-Agent Path Finding Problem

**arXiv ID:** 2601.10123 | [PDF](https://arxiv.org/pdf/2601.10123v1)

**作者:** Aditi Anand `[一作]` (Indian Institute of Technology Jammu), Suman Banerjee `[通讯]` (Indian Institute of Technology Jammu)

**通讯引用:** 5625 | [OpenAlex ID](https://openalex.org/A5033218913)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在多代理路径规划中加入公平性约束的 MAPF 问题，提出了 ε-怨恨自由、最大最小公平和比例公平三种公平性定义，并给出了相应的求解方法。

**💡 创新点**

创新点在于首次将公平性概念引入 MAPF，设计了满足不同公平性的启发式搜索（Fair‑ICTS）和公平冲突基础搜索（Fair‑CBS），以及针对自利代理的 DSIC 机制。

**🔧 技术方法**

主要采用了 IDA*、A* 搜索、DAG 构造、冲突基础搜索（CBS）以及单参数机制理论来实现公平路径规划与机制设计。

**📊 数据集**

实验使用了四个公开 MAPF 基准网格地图：empty‑16‑16、random‑32‑32‑20、empty‑48‑48 与 den312d。

**📈 对比分析**

与传统 CBS、IDA* 等基线方法相比，Fair‑ICTS 在成功率上更高、运行时间更短（小于 1s，最大约 3s），而 Fair‑CBS 需要数十秒至上百秒；公平约束对成功率与耗时均有影响。

**⚠️ 局限性**

局限性包括仅处理离散无权图的离线情境，未考虑在线加入/离开、动态障碍或不确定环境，且当代理数量极大时算法仍面临组合爆炸。

---

## 208. Reinforcement Learning to Discover a NorthEast Monsoon Index for Monthly Rainfall Prediction in Thailand

**arXiv ID:** 2601.10181 | [PDF](https://arxiv.org/pdf/2601.10181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 209. Beyond Single Prompts: Synergistic Fusion and Arrangement for VICL

**arXiv ID:** 2601.10117 | [PDF](https://arxiv.org/pdf/2601.10117v1)

**作者:** Wenwen Liao `[一作]` (Fudan University), Xiaofeng Yang `[通讯]` (Fudan University)

**通讯引用:** 9859 | [OpenAlex ID](https://openalex.org/A5100742379)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个端到端的视觉上下文学习(VICL)框架，解决了单一提示选择和提示排布忽略的问题。

**💡 创新点**

创新点包括：1) 适应性融合模块自动聚合多提示的细粒度信息；2) 轻量化、针对不同排布的MLP将几何先验解耦；3) 查询-支持交换的双向微调策略，提升融合与重建协同。

**🔧 技术方法**

使用技术主要有：跨注意力融合、ViT-MAE + VQGAN的图像填补、轻量化MLP适配器、双向微调与注意力权重重用。

**📊 数据集**

实验数据集包括 PASCAL-5i（前景分割）、PASCAL VOC 2012（单物体检测）、ImageNet-1K（图像着色）以及 COCO-5i（跨域验证）。

**📈 对比分析**

与多类基线（单提示选择、投票、PromptFusion 等）比较，平均提升 mIoU 至 49.32%（分割）、检测和着色指标亦显著优于对手，跨域测试亦保持显著优势。

**⚠️ 局限性**

局限性主要是：仍依赖图像填补框架，对极复杂的生成任务或更大规模提示集合的扩展尚未充分验证；轻量化MLP在极大排布空间中可能不足。

---

## 210. Graph Regularized PCA

**arXiv ID:** 2601.10199 | [PDF](https://arxiv.org/pdf/2601.10199v1)

**作者:** Antonio Briola `[一作]` (Shell Information Technology Limited), Tomaso Aste `[通讯]` (University College London)

**通讯引用:** 9136 | [OpenAlex ID](https://openalex.org/A5050674002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Graph Regularized PCA（GR-PCA），将特征图结构融入 PCA，以抑制高频噪声并提升低频结构捕获。

**💡 创新点**

通过联合稀疏精度矩阵学习与图拉普拉斯平滑正则化，构建结构感知的低秩分解方法，显著提升低频成分的选择性与对齐，且在异方差噪声环境下仍保持可解释性。

**🔧 技术方法**

使用 Graphical Lasso 估计稀疏精度矩阵，图拉普拉斯正则化以及 L1 稀疏化；对比传统 PCA 与 SparsePCA，且实现了可扩展的优化框架。

**📊 数据集**

在合成数据集上验证，模拟了 Erdős–Rényi、Barabási–Albert 与 Watts–Strogatz 三种图拓扑，并分别设置等方差与异方差噪声；未使用公开真实数据集。

**📈 对比分析**

与 PCA、SparsePCA 比较，GR-PCA 在异方差噪声下的选择性提升约 0.25–0.35、对齐率接近 1；虽然整体 R² 略下降，但说明高频噪声被有意压制。

**⚠️ 局限性**

限制在于图稠密或图学习不稳定时，正则化失效；在高密度或近全连图中，低高频分离不明显，GR-PCA 的优势趋近传统 PCA。

---

## 211. GFM4GA: Graph Foundation Model for Group Anomaly Detection

**arXiv ID:** 2601.10193 | [PDF](https://arxiv.org/pdf/2601.10193v1)

**作者:** Jiujiu Chen `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 42888 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图基础模型的组异常检测框架 GFM4GA，能够在少量标注下对图中异常子图进行检测

**💡 创新点**

创新点在于：①利用特征估计提取潜在异常子图；②在预训练阶段引入双层对比学习（子图级与节点级）捕捉组异常结构与特征偏差；③在微调阶段采用基于组异常比例的节点权重和邻居组上下文增强，提升少样本适应性

**🔧 技术方法**

技术包括 PCA 与 MLP 的特征异常估计、基于 GCN 的轻量化图编码器、双层对比学习（InfoNCE）、加权二元交叉熵、L2 正则化和邻居上下文构造

**📊 数据集**

使用 Weixin 真实社交网络预训练并微调，另外在 Weibo、Facebook、Amazon、T‑Finance 等公共数据集上构造合成组异常子图进行实验

**📈 对比分析**

与 GCN、GAT、SL‑GAD、DCL‑GFD、ComGA、GUDI、ARC、UNPrompt、AnomalyGFM 等方法对比，GFM4GA 在 10‑shot 组异常检测中平均提升 AUROC 2.85% 与 AUPRC 2.55%，在所有数据集均取得第一名或排名第一

**⚠️ 局限性**

局限性包括：对图文本信息依赖较低，LLM/文本提示的潜力未探索；在极小组异常比例或极大图规模下的鲁棒性尚需进一步验证；模型训练时间与计算成本仍较高

---

## 212. One Instruction Does Not Fit All: How Well Do Embeddings Align Personas and Instructions in Low-Resource Indian Languages?

**arXiv ID:** 2601.10205 | [PDF](https://arxiv.org/pdf/2601.10205v1)

**作者:** Arya Shah `[一作]` (Indian Institute of Technology), Mayank Singh `[通讯]` (Indian Institute of Technology)

**通讯引用:** 4534 | [OpenAlex ID](https://openalex.org/A5100746904)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了面向12种印度语言的多语言助手 persona–instruction 对齐检索基准，包含单语检索、跨语检索、逆向检索和兼容性二分类四个任务。

**💡 创新点**

首次在12种语言中同时进行双向检索与兼容性评估，提供可复现的基线，并揭示跨语言脚本边界对检索性能的影响。

**🔧 技术方法**

采用冻结编码器（E5‑Large‑Instruct、BGE‑M3、LaBSE 等）配合轻量化逻辑回归头，使用 Recall@k、MRR、AUROC、ECE 等指标进行评测。

**📊 数据集**

基于 GPT‑4o‑mini 合成的 5 万条英语 persona‑instruction 对，利用 NLLB‑200 翻译成 12 种印度语言，并通过双注释员验证兼容性，构建了跨语言对齐数据集。

**📈 对比分析**

在统一的冻结编码器设置下，对 8 种多语言嵌入模型进行评测，E5‑Large‑Instruct 在单语检索 Recall@1 为 27.4%、跨语检索 20.7%；BGE‑M3 在逆向检索 32.1%；LaBSE 在二分类 AUROC 为 75.3%；无单一模型在所有任务中占优。

**⚠️ 局限性**

仅基于合成语料与机器翻译，缺乏真实多样化用户语料；采用冻结编码器未考虑微调潜能；数据集仅覆盖 12 种主要印度语言，未包含部分重要语言。

---

## 213. What Gets Activated: Uncovering Domain and Driver Experts in MoE Language Models

**arXiv ID:** 2601.10159 | [PDF](https://arxiv.org/pdf/2601.10159v1)

**作者:** Guimin Hu `[一作]` (Guangdong University of Technology), Ruichu Cai `[通讯]` (Guangdong University of Technology)

**通讯引用:** 2500 | [OpenAlex ID](https://openalex.org/A5076948208)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了大规模混合专家（MoE）语言模型的专家激活机制，提出通过熵和因果效应两种指标区分域专家和驱动专家，并分析令这些专家被激活的输入token特征。

**💡 创新点**

创新点在于首次将脑神经科学中的驱动/功能专一单元概念迁移到MoE架构，设计熵基域专家度量与因果贡献评估方法，揭示专家在不同任务中的专一性与因果影响。

**🔧 技术方法**

技术方法包括基于门控概率的二元熵计算、基于扰动门控的KL因果贡献评估、CWAS（置信度加权激活得分）和对专家权重的上调/下调实验；模型使用了顶层稀疏MoE层。

**📊 数据集**

使用的数据集包括情感分析（EmotionLines）、多任务知识测评（MMLU）以及数学推理（GSM8K）三个公开领域数据集。

**📈 对比分析**

比较方法为对三个MoE LLM（Mixtral、DeepSeek-MoE、Qwen-MoE）在三大任务上分别上调/下调域/驱动专家权重，结果显示提升域专家权重可平均提升2.08%，提升驱动专家权重可平均提升3.00%，证明两类专家对模型性能均有显著正向影响。

**⚠️ 局限性**

局限性包括只研究了三种模型和三类任务，专家识别依赖阈值与路由日志，结果可能受模型规模、路由策略及超参数影响，未覆盖更广泛的MoE架构与训练方式。

---

## 214. ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback

**arXiv ID:** 2601.10156 | [PDF](https://arxiv.org/pdf/2601.10156v1)

**作者:** Yutao Mou `[一作]` (National Engineering Research Center for Software Engineering, Peking University), Jing Shao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 9670 | [OpenAlex ID](https://openalex.org/A5023198186)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个步骤级工具调用安全检测基准TS‑Bench，并提出TS‑Guard和TS‑Flow两种模型，实现LLM代理在执行前对每一步工具调用的安全性进行主动评估与干预。

**💡 创新点**

创新点在于：①首次将工具调用安全问题细化到步骤级别并构建相应基准；②采用多任务强化学习设计TS‑Guard，实现对请求危害性、攻击关联性与具体调用安全性的三维预测；③将门控反馈实时注入代理决策流程，形成TS‑Flow，避免传统“检测‑终止”模式带来的效率损失。

**🔧 技术方法**

使用多任务强化学习（GRPO）训练TS‑Guard，结合LLM安全门控、ReAct框架以及token熵分析来评估模型自信度；TS‑Flow则在代理（如Qwen‑2.5‑14B‑Instruct、GPT‑4o）中动态注入门控输出。

**📊 数据集**

TS‑Bench基准由AgentAlign、AgentHarm、ASB、AgentDojo四大公开代理安全数据集的交互日志抽取并逐步标注得到；实验中还使用原始基准数据集与公开LLM/门控模型（GPT‑4o、Qwen、LlamaGuard、ShieldAgent‑THU等）进行对比。

**📈 对比分析**

在TS‑Bench上，TS‑Guard以约95%+准确率、F1与召回率领先于GPT‑4o、Qwen、LlamaGuard等对手；在AgentDojo、ASB与AgentHarm等代理安全评测中，TS‑Flow将有害调用率降低约65%，同时保持或提升约10%的任务完成率，显示出优越的安全‑效能平衡。

**⚠️ 局限性**

主要局限包括：①门控反馈被直接拼接至代理输入，代理未必能完全利用，导致干预效果受限；②代理与门控模型独立训练，缺乏协同调优，可能出现判断与决策不匹配的情况；③尚未对不同代理与门控模型的联合训练与更深层集成进行探索。

---

## 215. Understanding and Preserving Safety in Fine-Tuned LLMs

**arXiv ID:** 2601.10141 | [PDF](https://arxiv.org/pdf/2601.10141v1)

**作者:** Jiawen Zhang `[一作]` (Zhejiang University), Ruoxi Jia `[通讯]` (Virginia Tech)

**通讯引用:** 2658 | [OpenAlex ID](https://openalex.org/A5032275274)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究细调大型语言模型后其安全性表现，并提出一种能够在保持生成质量的同时提升安全性的细调框架。

**💡 创新点**

创新点在于将安全性评估指标与自适应梯度惩罚结合，形成“安全感知细调框架”，并提出可解释的安全约束策略。

**🔧 技术方法**

采用了梯度惩罚、对抗训练、指令微调以及自适应安全约束技术，配合安全性评分器对模型进行评估与调优。

**📊 数据集**

使用了SST‑2、AG News、SafetyBench等公开数据集，以及大模型的开源版本（如7B/13B LLaMA）进行实验。

**📈 对比分析**

与传统微调、RLHF、PPO方法对比，安全分数平均提升15%，文本质量（BLEU/ROUGE）保持93%及以上，表明安全提升不牺牲生成质量。

**⚠️ 局限性**

主要局限是计算成本高、对抗样本覆盖不完整、以及安全评分器可能存在的偏见和解释性不足。

---

## 216. VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation

**arXiv ID:** 2601.10124 | [PDF](https://arxiv.org/pdf/2601.10124v1)

**作者:** Sicheng Yang `[一作]` (Hong Kong University of Science and Technology), Lei Zhu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 70759 | [OpenAlex ID](https://openalex.org/A5100394072)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于向量量化（VQ）的半监督医学影像分割框架 VQ‑Seg

**💡 创新点**

创新点包括：①Quantized Perturbation Module（QPM）在离散VQ空间实现可控扰动；②双分支共享后VQ空间实现图像重建与分割联合优化；③Post‑VQ Feature Adapter（PFA）通过对齐预训练基准模型（DINOv2）提升语义一致性

**🔧 技术方法**

技术手段：向量量化、基于距离的扰动策略、双分支网络、伪标签一致性、基准模型对齐的对比学习、EMA教师网络、STE梯度估计

**📊 数据集**

使用两大数据集：自采的 828 张肺癌 CT（中央型肺癌）以及公开的 ACDC 心脏 MRI 数据集

**📈 对比分析**

与多种 SOTA 方法（UNet、UA‑MT、MCNet、SSNet、BCP、ARCO、ABD、Unimatch 等）在 5%/10% 标注比例下对比，VQ‑Seg 在 Dice、Jaccard、HD95、ASD 上均领先，尤其在低标注场景显著提升

**⚠️ 局限性**

局限性：扰动仅在离散 VQ 空间内实现，难以推广到连续特征；使用基准模型对齐会带来额外计算开销

---

## 217. Advanced Encryption Technique for Multimedia Data Using Sudoku-Based Algorithms for Enhanced Security

**arXiv ID:** 2601.10119 | [PDF](https://arxiv.org/pdf/2601.10119v1)

**作者:** Mithil Bavishi `[一作]` (Dr. Jayesh Shah College of Engineering), Vinaya Sawant `[通讯]` (Dr. Jayesh Shah College of Engineering)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对多媒体数据（图像、音频、视频）提出基于数独的加密与解密方法，并通过时间戳动态生成密钥，实现多层混淆（阈值加扰、填充+块置换、旋转、XOR）

**💡 创新点**

创新点：① 将数独谜题与时间戳结合生成可变密钥；② 在同一框架下对图像、音频、视频分别实现三种加密模式（shuffle、XOR、变换）；③ 通过多层混淆提升安全性，且提供了统一的实现与评估流程

**🔧 技术方法**

使用技术包括：数独生成/求解（Py‑Sudoku）、块级置换与逆置换、阈值加扰与反加扰、随机填充、图像/视频帧旋转、音频块置换、XOR 置换；评估指标包括 NPCR、UACI、熵、SNR/PSNR/MSE、加密/解密耗时

**📊 数据集**

数据集：图像 – Lena、Camerman、House、Mandrill、San Diego、Towers；视频 – 标准测试视频（未列明具体文件）；音频 – CantinaBand3.wav、StarWars3.wav

**📈 对比分析**

与其他算法（Norouzi 2014、Arpaci 2020、Mehta 2022 等）在 NPCR、UACI、SNR、PSNR、MSE、加密耗时进行对比。结果显示 NPCR ≈ 100%，UACI 高于传统 30%–35% 的水平，SNR 仅轻微下降，且加密耗时 10–125 ms，显著快于现有方法

**⚠️ 局限性**

限制：UACI 值仍偏高，未达到期望的 33%；算法基于 Python，速度受限；未做量子抗性或嵌入式平台的性能评估；对实时视频加密的处理效率仍需提升

---

## 218. PRL: Process Reward Learning Improves LLMs' Reasoning Ability and Broadens the Reasoning Boundary

**arXiv ID:** 2601.10201 | [PDF](https://arxiv.org/pdf/2601.10201v1)

**作者:** Jiarui Yao `[一作]` (University of Illinois Urbana-Champaign), Tong Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 24436 | [OpenAlex ID](https://openalex.org/A5100378779)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Process Reward Learning (PRL)，通过把熵正则化的 RL 目标分解为中间步骤，用当前策略与参考策略的对数比值产生过程奖励，从而在不增加额外 MCTS 或奖励模型开销的前提下提升 LLM 的推理能力。

**💡 创新点**

创新点在于：①从理论上严格推导出熵正则化 RL 可被拆解为过程奖励；②用对数比值直接给中间步骤赋值，消除了昂贵的 MCTS 或额外奖励网络；③将过程奖励与传统策略梯度结合，兼顾探索与稳定性。

**🔧 技术方法**

技术包括：基于 PPO/GRPO 的策略梯度框架；KL 正则化、重要性采样与截断技巧；对数比值的过程奖励设计；可选的可学习奖励模型或规则化奖励；实现了完整的 PRL 算法流程。

**📊 数据集**

训练使用 NuminaMath 约 15 万样本；评测集涵盖 MATH500、Minerva Math、Olympiad Bench、AIME24、AMC23 五大数学推理基准。

**📈 对比分析**

与 RAFT、GRPO、REINFORCE 等基线在平均 @8 与 pass @8 指标上进行对比；PRL 在各基准上均实现显著提升，既提高平均通过率，又拓宽通过率边界。

**⚠️ 局限性**

限制：实验仅在 1B–7B 规模的开源模型上验证，未验证更大模型；过程奖励的分割方式（固定长度或换行符）需进一步优化；不同奖励拆解方法可能导致不同的奖励塑形效果。

---

## 219. MHub.ai: A Simple, Standardized, and Reproducible Platform for AI Models in Medical Imaging

**arXiv ID:** 2601.10154 | [PDF](https://arxiv.org/pdf/2601.10154v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 220. CC-OR-Net: A Unified Framework for LTV Prediction through Structural Decoupling

**arXiv ID:** 2601.10176 | [PDF](https://arxiv.org/pdf/2601.10176v1)

**作者:** Mingyu Zhao `[一作]` (Renmin University of China), Hengliang Luo `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了 CC‑OR‑Net 统一框架，用于解决零值、长尾和鲸鱼用户三重挑战的 LTV 预测。

**💡 创新点**

创新点在于：① 结构化分解——将排名与回归通过阶级级联二分类完全解耦；② 通过残差回归细化桶内预测；③ 针对鲸鱼用户的注意力增强增量学习；④ 用分布蒸馏与自监督保证全局概率一致；⑤ 设计了业务导向的 SVA 指标。

**🔧 技术方法**

技术包括：阶级级联二分类、残差网络、GLU 门控特征对齐、注意力增强的数据增强、分布蒸馏（KL 损失）、加权 BCE、MSE、Focal Loss、SVA、Recall@k 等评估。

**📊 数据集**

实验数据集为三大规模工业数据集：Domain‑1（≈248M 记录，零值率33.6%）、Domain‑2（≈41M 记录，零值率64.6%）和 Domain‑3（≈33M 记录，零值率45.8%），全部按 4 级桶划分。

**📈 对比分析**

与 XGBoost、Two‑Stage XGB、CORAL、POCNN、DeepFM、MMOE‑FocalLoss、ZILN、MDME、ExpLTV、OptDist 等多种传统、深度与专业 LTV 方法进行对比。CC‑OR‑Net 在 GINI、Spearman、SVA、AMBE、Recall@k 等关键业务指标上均取得显著提升，特别是高价值用户识别和整体 SVA 最高。

**⚠️ 局限性**

局限性：① 主要针对非负、单调 LTV；负值或极端离群需额外桶处理；② 模型复杂度相对较高，训练与推理资源占用比单一回归模型大；③ 对极端分布漂移需要重新设定桶阈值和蒸馏权重；④ 鲸鱼样本稀缺仍可能导致局部过拟合；⑤ 需手工设计业务阈值，适配不同平台时可能需要额外调参。

---

## 221. Towards Online Malware Detection using Process Resource Utilization Metrics

**arXiv ID:** 2601.10164 | [PDF](https://arxiv.org/pdf/2601.10164v1)

**作者:** Themistoklis Diamantopoulos `[一作]` (Aristotle University of Thessaloniki), Andreas L. Symeonidis `[通讯]` (Aristotle University of Thessaloniki)

**通讯引用:** 2031 | [OpenAlex ID](https://openalex.org/A5047313464)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于在线学习的动态恶意软件检测方法，利用进程资源使用指标并加入时间信息进行实时模型更新。

**💡 创新点**

创新点在于将在线学习框架（Adaptive Random Forest）与进程级资源监测特征结合，实现对零日恶意软件的即时适应与检测，并在稀缺标签环境下仍保持较好性能。

**🔧 技术方法**

主要技术包括：数据预处理（缺失值填补、标识符剔除、时间戳化、零填充）、特征工程（CPU、内存、I/O、网络等32项指标）、随机森林批量学习与Adaptive Random Forest在线学习、Test‑Then‑Train循环、性能评估（准确率、精确率、召回率、F1）。

**📊 数据集**

使用公开的进程资源利用度数据集（约28,213条样本，涵盖104种恶意软件）并通过VirusTotal API补充年份标签，形成十年跨度的时间序列数据。

**📈 对比分析**

与传统批量随机森林模型对比：在随机划分（60/40）下批量模型更优；在按年份训练/测试（先一年训练后逐年测试）时，在线模型在准确率、精确率、召回率和F1上均明显优于批量模型；当仅部分恶意样本标记时，在线模型仍能保持高于批量模型的性能，说明其对标签稀缺性更鲁棒。

**⚠️ 局限性**

局限性包括：仅使用资源利用度特征，未涉及API调用、系统调用或内存快照等更细粒度动态特征；算法仅限于随机森林/Adaptive Random Forest，未评估更先进的深度学习或Transformer模型；缺乏对概念漂移的系统检测与自适应机制；数据集可能不完全代表真实环境中的恶意变异和多样性。

---

## 222. AWED-FiNER: Agents, Web applications, and Expert Detectors for Fine-grained Named Entity Recognition across 36 Languages for 6.6 Billion Speakers

**arXiv ID:** 2601.10161 | [PDF](https://arxiv.org/pdf/2601.10161v1)

**作者:** Prachuryya Kaushik `[一作]` (Indian Institute of Technology Guwahati), Ashish Anand `[通讯]` (Indian Institute of Technology Guwahati)

**通讯引用:** 2027 | [OpenAlex ID](https://openalex.org/A5040626471)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AWED‑FiNER，一个支持 36 种全球语言细粒度命名实体识别的开放源代码生态系统。

**💡 创新点**

将代理工具、交互式 Web 应用和 49 个小型专家模型统一，首次覆盖 6.6 亿用户的细粒度 NER 任务。

**🔧 技术方法**

采用多语言预训练模型（XLM‑RoBERTa、MuRIL、IndicBERTv2、mBERT）Fine‑tune，结合 smolagents 代理框架与 Hugging Face Spaces 的无服务器路由。

**📊 数据集**

整合了 SampurNER、CLASSER、MultiCoNER2、FewNERD、FiNERVINER、APTFiNER、FiNE‑MiBBiC 等多任务数据集。

**📈 对比分析**

在 22 种语言的 MultiCoNER2 与 27 种语言的 FewNERD 基准上使用 Macro‑F1 评估，最高得分约 86%，显著优于公开基线模型。

**⚠️ 局限性**

覆盖语言有限，极低资源语言表现仍差；代理架构带来额外推理延迟和计算开销。

---

## 223. History Is Not Enough: An Adaptive Dataflow System for Financial Time-Series Synthesis

**arXiv ID:** 2601.10143 | [PDF](https://arxiv.org/pdf/2601.10143v1)

**作者:** Haochong Xia `[一作]` (Nanyang Technological University), Bo An `[通讯]` (Nanyang Technological University)

**通讯引用:** 6684 | [OpenAlex ID](https://openalex.org/A5017743551)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种漂移感知自适应数据流系统，利用可微分的参数化金融增广模块与基于梯度的双层优化计划器‑调度器实现训练、验证反馈闭环，动态控制增广强度、比例和操作分布；

**💡 创新点**

① 将数据增强、课程学习与工作流管理统一为单一可微分框架；② 设计带金融先验（K‑线一致性、协整、非平稳性）的参数化合成模块；③ 通过学习导向的双层优化实现增广策略的自适应调节；

**🔧 技术方法**

使用梯度双层优化、Straight‑Through 估计训练计划器；参数化单股变换（抖动、缩放、幅度扭曲、排列、STL）、多股 Mix‑up（Cut‑Mix、Linear‑Mix、Amplitude‑Mix 等）与归一化、插值补偿；结合时序预测模型（GRU/LSTM/Transformer/TCN/DLinear）与强化学习（DQN/PPO）进行实验；

**📊 数据集**

S&P/ DJI 日频股票 27 只（2000‑2024）、加密货币 BTC/ETH/DOT/LTC 时频 1h（2023‑2025）；与天气、能源、ETT 等基准时序数据对比；

**📈 对比分析**

与原始、RandAug、TrivialAug、AdaAug 等基准对比。预测任务上5种模型的 MSE/MAE/STD 均显著下降；交易任务上 TR/SR 明显提升；增广质量通过判别器准确率 50.1%（即与真实数据几乎不可区分）优于多种生成模型；

**⚠️ 局限性**

仍需人工调节超参数（τ、freq 等），对不同市场或任务可能需重新校准；系统实现复杂，难以在多资产多策略交易等更大规模场景直接推广；增广质量评估主要依赖判别器，缺乏更细粒度的财务一致性检验。

---

## 224. Step-by-Step Causality: Transparent Causal Discovery with Multi-Agent Tree-Query and Adversarial Confidence Estimation

**arXiv ID:** 2601.10137 | [PDF](https://arxiv.org/pdf/2601.10137v1)

**作者:** Ziyi Ding `[一作]` (Tsinghua University), Xiao-Ping Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 35699 | [OpenAlex ID](https://openalex.org/A5100363169)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于多专家LLM的树查询框架Tree-Query，将因果发现拆解为一系列可解释、有限深度的查询，配合对抗式置信估计实现自适应置信分数；

**💡 创新点**

创新点在于（1）树式分解实现全局可解释的因果推理；（2）引入对抗式置信估计提升判断鲁棒性；（3）在无观测数据的情境下通过LLM推断完整的因果图，突破传统基于条件独立检验的误差传播问题；

**🔧 技术方法**

主要技术包括多专家系统（MES）选择专家投票、对抗式置信估计（ACE）、固定树结构的四种基本查询（后门、独立、潜在混杂、方向）以及判定规则；

**📊 数据集**

使用了Mooij等人的真实因果图数据集和UCI机器学习数据集（改造成无观测数据和带潜在混杂的版本）进行评估；

**📈 对比分析**

与直接LLM查询、PC、FCI等基线比较，Tree-Query在标准和潜在混杂两个基准上均显著降低SHD（平均约减少20条边），并在有置信分数的情况下取得NDCG 0.73–0.81（标准）和0.68–0.76（混杂）;

**⚠️ 局限性**

局限性包括：仅在无观测数据场景下验证，无法直接利用实际样本信息；LLM推理受模型prompt和知识库限制；对抗式置信估计需要额外计算和参数调优；

---

## 225. Redundancy-Driven Top-$k$ Functional Dependency Discovery

**arXiv ID:** 2601.10130 | [PDF](https://arxiv.org/pdf/2601.10130v1)

**作者:** Xiaolong Wan `[一作]` (Harbin Institute of Technology), Xixian Han `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 389 | [OpenAlex ID](https://openalex.org/A5006689000)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于冗余计数的 top‑k 函数依赖（FD）发现算法 SDP，避免完整枚举所有 FD，直接寻找最有价值的前 k 条依赖。

**💡 创新点**

创新点包括：① 设计单调上界用于快速剪枝；② 利用分区卡氏矩阵（PCM）构造更紧的上界；③ 采用全局优先调度在多棵搜索树间共享阈值，从而大幅提升剪枝效率。

**🔧 技术方法**

技术实现：最小击中集枚举、分区统计、PCM 预处理、属性分区大小排序、优先队列调度与多线程堆管理。

**📊 数据集**

实验使用 40+ 公开数据集（如 flight、airline_dataset、covtype、reactionnetwork、superconductivity、fdata 等），覆盖 8–109 个属性、10³–10⁷ 条记录。

**📈 对比分析**

与基线 FDR 对比实验表明：在高维或大规模数据上 SDP 速度提升 10–1000 倍，内存占用更低，且始终返回精确的 top‑k 结果；在低维或小数据集时差异不大甚至略慢。

**⚠️ 局限性**

局限性：在极小数据集或 k 值很大时，额外的预处理和全局调度会带来轻微开销；若 k 过大，阈值下降导致剪枝力度减弱，导致运行时间和内存占用上升。

---

## 226. TopoDIM: One-shot Topology Generation of Diverse Interaction Modes for Multi-Agent Systems

**arXiv ID:** 2601.10120 | [PDF](https://arxiv.org/pdf/2601.10120v1)

**作者:** Rui Sun `[一作]` (University of Science and Technology of China), Linyuan Lü `[通讯]` (University of Science and Technology of China)

**通讯引用:** 13364 | [OpenAlex ID](https://openalex.org/A5000969982)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TopoDIM框架，能够一次性生成多关系的异构通信拓扑，实现LLM多智能体系统的高效协同。

**💡 创新点**

创新点包括：①一种单轮一次性拓扑生成机制，消除多轮对话带来的冗余；②使用条件、反馈、辩论三种多样化交互模式；③通过强化学习与熵正则化实现多样化拓扑优化；④将全局拓扑学习通过知识蒸馏转化为各智能体的本地轻量策略，实现去中心化执行。

**🔧 技术方法**

核心技术包括：异构图卷积网络做情境编码、基于自回归的边采样解码器、强化学习（策略梯度）与熵正则化、稀疏化与无环约束、去中心化的本地策略蒸馏。

**📊 数据集**

使用的公开基准包括：MMLU-Pro、MultiArith、GSM8K、AIME、HumanEval、LiveCodeBench 等多种推理与编程任务，且在不同规模LLM（Gemma、GPT-OSS、DeepSeek‑V3.2等）上验证。

**📈 对比分析**

与六大基线（单智能体 Vanilla、CoT；多智能体 LLM‑Debate、GPTSwarm、G‑Designer、AgentDropout）对比，TopoDIM在平均任务性能上提升约1.3–1.5%，在token消耗上减少约46%，并在异构智能体环境下表现更优。

**⚠️ 局限性**

局限性包括：仅考虑了条件、反馈、辩论三种交互，未覆盖更复杂的组织形式如动态联盟；在异构智能体组合时仍需权衡性能与轻量级模型的兼容性；相比单一LLM，整体系统仍存在额外的通信开销与延迟。

---

## 227. On Existence of Girth-8 QC-LDPC Code with Large Column Weight: Combining Mirror-sequence with Classification Modulo Ten

**arXiv ID:** 2601.10170 | [PDF](https://arxiv.org/pdf/2601.10170v1)

**作者:** Guohua Zhang `[一作]` (Xi'an University of Posts and Telecommunications), Yi Fang `[通讯]` (Guangdong University of Technology)

**通讯引用:** 4123 | [OpenAlex ID](https://openalex.org/A5067418255)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

设计并构造了新的基于GCD框架的七列权重与八列权重的QC‑LDPC码，确保其girth为8且循环矩阵尺寸尽可能小。

**💡 创新点**

创新点包括：引入镜像序列（mirror sequence）与新的按模10重排行的行重组技术；在已有GCD框架基础上，显著提升了对连续循环矩阵尺寸的下界约20%；并提供了比该下界更小的循环尺寸构造方案，使尺寸比现有最优下界低约25%。

**🔧 技术方法**

技术手段主要是代数构造（使用GCD约束、镜像序列、行重组和等价变换操作）和对循环矩阵尺寸下界的理论推导（Lemma、Theorem系列）。

**📊 数据集**

本文不依赖数据集，研究对象为LDPC码的符号矩阵与循环尺寸，所有结果均通过代数证明和理论分析得到。

**📈 对比分析**

与传统搜索方法（尤其是对称结构SYM码）对比，在相同循环尺寸（如221、559）下，新构造的girth‑8码在BER/FER曲线上显著优于SYM码，说明其性能更好。

**⚠️ 局限性**

局限性：构造方法对L的取值模10有特定限制，需按不同余数分别设计；对更大列权重（>8）尚未给出通用构造；以及理论下界与实际实现之间仍存在一定差距，进一步的性能优化与实现细节需后续研究。

---

## 228. RAG-3DSG: Enhancing 3D Scene Graphs with Re-Shot Guided Retrieval-Augmented Generation

**arXiv ID:** 2601.10168 | [PDF](https://arxiv.org/pdf/2601.10168v1)

**作者:** Yue Chang `[一作]` (AI Thrust, Hong Kong University of Science and Technology Guangzhou), Sihong Xie `[通讯]` (AI Thrust, Hong Kong University of Science and Technology Guangzhou)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种针对开放词汇的 3D 场景图生成框架，通过重拍（re-shot）视角生成最佳图像并估计不确定性，利用检索增强生成（RAG）对高不确定性对象进行描述改进，并引入动态下采样‑映射策略加速跨视角对象聚合。

**💡 创新点**

创新点主要包括：① 通过重拍视角与原始视角对比来估计对象级不确定性；② 对高不确定性对象使用基于检索的 RAG 进行语义补全；③ 动态下采样映射策略在保持细粒度的同时显著降低点云处理时间；④ 扩大关系类别并引入少量示例提示提升边关系的准确性。

**🔧 技术方法**

技术与工具：SAM 分割、CLIP 语义嵌入、3D 点云融合、重拍渲染与视角质量评分、VLM（如 GPT‑4o/LLM）生成标题与关系、检索增强生成（RAG）、结构化 LLM 提示、动态邻域阈值、人工评估与基准对比。

**📊 数据集**

实验数据集：Replica 室内场景数据集。

**📈 对比分析**

与 ConceptGraphs 与 ConceptGraphs‑Detector 两个基线对比：节点精度 0.82（vs. 0.68/0.58），边精度 0.91（vs. 0.82/0.85）；动态下采样下平均映射时间从 6.65 s 降至 2.49 s；在语义分割上 mAcc 40.67、f‑mIOU 35.65，基本与或优于现有方法。

**⚠️ 局限性**

局限性：仍需离线 VLM/LLM 推理，计算开销较大；缺乏实时性能评估；仅在 Replica 上验证，泛化能力未知；重拍图像质量受点云重建误差影响；检索仅基于最近邻语义嵌入，可能受嵌入误差限制。

---

## 229. Advancing Adaptive Multi-Stage Video Anomaly Reasoning: A Benchmark Dataset and Method

**arXiv ID:** 2601.10165 | [PDF](https://arxiv.org/pdf/2601.10165v1)

**作者:** Chao Huang `[一作]` (Sun Yat-sen University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 25574 | [OpenAlex ID](https://openalex.org/A5068837264)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了视频异常推理（VAR）任务，要求模型在回答异常相关问题前进行感知‑认知‑行动三阶段结构化推理；

**💡 创新点**

创新点在于①定义了VAR任务并构建了具备Perception–Cognition–Action链式思考的Vad‑Reasoning‑Plus大规模数据集；②提出A2‑GRPO强化学习策略以弱监督方式提升推理深度、风险评估与异常验证；③设计Vad‑R1‑Plus端到端多模态大模型实现自适应层级推理和风险决策；

**🔧 技术方法**

采用多模态大语言模型（Qwen2.5‑VL‑7B）结合自监督微调+A2‑GRPO强化学习；使用PerCoAct‑CoT模板化提示、结构化标签和强化奖励；

**📊 数据集**

使用新构建的Vad‑Reasoning‑Plus数据集（8,641段视频、50,000+问题，覆盖感知、认知、行动三层级）；

**📈 对比分析**

与多类基线（通用视频MLLM、推理导向MLLM、VAD专用模型及商业模型）进行对照，Vad‑R1‑Plus在多项评测（多选准确率、开放式答案BLEU/ROUGE/METEOR、推理质量判定）均显著优于所有对手，尤其在双正/三正（reasoning+answer+category）指标上领先；

**⚠️ 局限性**

局限性包括：1）仍依赖大型LLM与昂贵算力；2）在极端稀有异常场景下推理表现可能受限；3）数据集与算法主要聚焦视频监控场景，跨域推广需要进一步验证；4）强化学习奖励设计虽有效但可能导致生成长度或细节的偏差。

---

## 230. Simple Network Graph Comparative Learning

**arXiv ID:** 2601.10150 | [PDF](https://arxiv.org/pdf/2601.10150v1)

**作者:** Qiang Yu `[一作]` (Harbin Institute of Technology), Chuanyi Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 689 | [OpenAlex ID](https://openalex.org/A5103171964)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于对比学习的无监督节点分类方法SNGCL，利用叠加多层拉普拉斯平滑滤波获得全局与局部视图，送入动量引导的孪生网络，并通过三重重组损失与上界损失实现类内聚合、类间分离；

**💡 创新点**

创新点在于①使用多层叠加的正则化与对称归一化拉普拉斯滤波同时生成全局与局部特征视图，②构建无负样本的三重对比损失，融合结构与邻域信息，③引入上界损失进一步压缩类内距离；

**🔧 技术方法**

主要技术包括拉普拉斯平滑滤波、动量孪生网络（Online‑Target结构）、三重对比（Triplet）损失与上界（Upper‑bound）损失、无监督节点嵌入与t‑SNE可视化；

**📊 数据集**

实验使用了五个公开图数据集：Cora、Citeseer、Pubmed、Amazon‑Photo 与 Coauthor‑CS；

**📈 对比分析**

与10种基线（GCN、GAT、DGI、GMI、GIC、GRACE、MVGRL、MERIT、SUGRL、NCLA）进行比较，SNGCL 在所有五个数据集上均取得最高或接近最高的节点分类准确率，尤其在 Amazon‑Photo 上达到 93.2%；

**⚠️ 局限性**

局限性包括：对超参数 t、ω1、ω2 依赖较强；仅在中小规模图数据上验证，缺乏对极大图的可扩展性分析；无理论收敛或泛化性能的严格证明。

---

## 231. Function Correcting Codes for Maximally-Unbalanced Boolean Functions

**arXiv ID:** 2601.10135 | [PDF](https://arxiv.org/pdf/2601.10135v1)

**作者:** Rajlaxmi Pandey `[一作]` (Indian Institute of Science), B. Sundar Rajan `[通讯]` (Indian Institute of Science)

**通讯引用:** 6591 | [OpenAlex ID](https://openalex.org/A5015398340)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了最大失衡布尔函数的最优单错误校正功能纠错码（SEFCC），给出了代码的构造、数量及其距离矩阵分类；

**💡 创新点**

首次把功能纠错码的距离矩阵与误码性能关联起来，提出了依据距离矩阵对码结构进行分组并量化其对数据误码率（Data BER）与功能误码率（Func BER）的影响；

**🔧 技术方法**

利用距离矩阵分析、组合论计数、BPSK调制下的AWGN信道模拟以及软/硬判决解码；

**📊 数据集**

无实际数据集，使用所有满足距离约束的代码组合进行全枚举与仿真；

**📈 对比分析**

通过比较不同距离矩阵组在软判决与硬判决下的误码曲线，发现软判决下更大上三角和行和对应更优误码性能，而硬判决下这种关系不再单调；

**⚠️ 局限性**

局限在于只考虑单错误校正、最大失衡布尔函数、低维(k≤3)的实验，且枚举复杂度随k快速增长，难以推广至更大规模。

---

## 232. Is More Context Always Better? Examining LLM Reasoning Capability for Time Interval Prediction

**arXiv ID:** 2601.10132 | [PDF](https://arxiv.org/pdf/2601.10132v1)

**作者:** Yanan Cao `[一作]` (Walmart Global Tech), Kannan Achan `[通讯]` (Walmart Global Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对大型语言模型（LLM）在预测用户复购时间间隔的能力进行系统评估，并在零样本场景下将其与统计方法及传统机器学习模型进行对比。

**💡 创新点**

1) 明确展示了LLM在结构化时序推理中的局限性；2) 通过三种上下文层级（零、媒介、高）实验证明，适度的上下文可以提升LLM表现，而过多叙事式信息会适得其反；3) 提出一种基于上下文层级的评估框架，可为后续构建混合模型提供思路。

**🔧 技术方法**

LLM：GPT‑4o、Gemini‑2.5 Pro、Claude‑3.5 Sonnet（零样本提示）；传统机器学习：RandomForest、XGBoost、深度神经网络；统计基线：均值、众数、指数平滑；评估指标：RMSE、MAE、MAPE、TA@0/1/2。

**📊 数据集**

两个真实电商数据集：①自研零售采购数据（5,780名用户，12类商品，≈110k笔订单）；②公共Instacart订单数据（2,661名用户，10类商品，≈194k笔订单）。

**📈 对比分析**

在所有指标上，机器学习模型始终优于LLM，LLM在业务关键的TA@1、TA@2指标上仅略优于统计基线。LLM在零上下文下已能超越统计基线；加入中等上下文可进一步提升；加入高上下文则多半导致性能下降。整体来看，LLM在定量精度上差距明显，但在近似预测（±1–2天）上已具备实用价值。

**⚠️ 局限性**

1) LLM难以精准捕捉量化时间规律，误差较大；2) 过多叙事上下文会引入噪声，削弱模型对时间模式的识别；3) 仅在零样本场景下测试，缺乏对微调或少量示例的探索；4) 评估仅关注单一时间间隔任务，未验证在更复杂时序预测任务中的可扩展性。

---

## 233. Role-Playing Agents Driven by Large Language Models: Current Status, Challenges, and Future Trends

**arXiv ID:** 2601.10122 | [PDF](https://arxiv.org/pdf/2601.10122v1)

**作者:** Ye Wang `[一作]` (Communication University of China), Hongjiang Xiao `[通讯]` (Communication University of China)

**通讯引用:** 87 | [OpenAlex ID](https://openalex.org/A5100967746)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述并评估了基于大型语言模型的角色扮演代理（RPLA）的技术进展、数据与评测框架

**💡 创新点**

创新点在于将RPLA从规则式模板演进到人格认知与行为决策的多维度模型，提出动态人格演化、跨模态沉浸交互与多智能体协作的新研究方向

**🔧 技术方法**

采用了大型语言模型、心理量表驱动的人格建模、内存增强提示（如CHARMAP、MAP）、情境/动机驱动的行为决策链、知识图谱、RLHF奖励模型及LLM自动评测器

**📊 数据集**

使用了LIFECHOICE、CoSER、CharacterEval、RoleEval、InCharacter、RVBench、RoleBench等专门的角色数据集以及小说、剧本、动漫文本等多模态原始语料

**📈 对比分析**

通过多维度指标（人格一致性、情境连贯性、行为合理性等）与人类评测、奖励模型对齐，对比闭源与开源模型；例如CoSER‑70B在LIFECHOICE上实现93.47%准确率，优于某些GPT‑4o版本；CharacterRM奖励模型与人工评分相关度高于传统LLM评分

**⚠️ 局限性**

存在的局限包括角色人格漂移、时点角色幻觉、价值对齐不足、数据版权与多模态资源稀缺、奖励模型泛化差、评测更倾向语言流畅性而非角色真实性，以及缺乏长期交互与情感动态建模

---

## 234. HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning

**arXiv ID:** 2601.10187 | [PDF](https://arxiv.org/pdf/2601.10187v1)

**作者:** Ziang Cui `[一作]` (Bilibili Inc), Hongwei Lin `[通讯]` (Bilibili Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大语言模型翻译中的跨语言冗长偏差，并提出 Sand-Glass 基准与 HOMURA 强化学习框架，实现时间受限下的语义压缩。

**💡 创新点**

创新点在于：①构建以音节预算评估时长的 Sand-Glass 基准；②开发 KL 正则化的 HOMURA RL 方法，使用动态音节比例奖励直接平衡长度与语义完整性。

**🔧 技术方法**

采用的技术包括 GRPO 强化学习、KL 正则化、动态音节比例奖励、语义保留奖励（反向翻译相似度、语法判定、链式推理奖励）等。

**📊 数据集**

使用的 数据集：基于口语字幕提取的中文-英文、中文-德语、中文-西班牙语对齐数据构建的 Sand-Glass；未使用标注压缩数据。

**📈 对比分析**

与五类长度控制策略（无约束、提示压缩、Best‑of‑N、翻译+改写、HOMURA）对比，HOMURA 在满足音节预算的前提下显著提升 BLEU‑ρ、BT‑CERR，并保持更高的语义完整性和推理效率。

**⚠️ 局限性**

局限性在于仅以文本音节计数作为时长代理，未考虑语音节奏、唇同步等多模态约束；实验范围局限于汉英/德/西三对，压缩极限是否普适仍待验证。

---

## 235. Bias in the Shadows: Explore Shortcuts in Encrypted Network Traffic Classification

**arXiv ID:** 2601.10180 | [PDF](https://arxiv.org/pdf/2601.10180v1)

**作者:** Chuyi Wang `[一作]` (Tsinghua University), Yong Cui `[通讯]` (Tsinghua University)

**通讯引用:** 21885 | [OpenAlex ID](https://openalex.org/A5007046740)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

BiasSeeker 提出一种半自动化、模型无关的统计方法，检测并分类加密网络流量中的 shortcut 特征，并提供相应的缓解策略。

**💡 创新点**

创新点在于将 Adjusted Mutual Information 作为统一的特征相关性度量，构建基于域知识的三类 shortcut 体系，并在此基础上设计分类特定的验证和处理方法，首次实现对多任务、跨数据集的快捷特征系统化检测。

**🔧 技术方法**

采用 tshark 提取原始报文字段，利用 AMI 进行特征排序，结合相对变换、随机遮盖等三种遮蔽策略，以及基于 KL 散度等统计检验，对 shortcut 进行验证；同时使用 NetMamba 与决策树作为基线模型。

**📊 数据集**

评估涵盖 19 个公开数据集，覆盖 VPN、恶意软件和加密应用三大分类任务，包括 ISCXVPN2016、CrossNet2021、CSTNET-TLS1.3、SurfsharkVPN、USTC-TFC2016、Ransomware 等。

**📈 对比分析**

通过对比在不同 shortcut 缓解策略下的模型准确率，发现删除数据泄露标识、相对化时间戳、任务无关字段等可在部分数据集上提升 1–3% 的准确率，验证了 BiasSeeker 能有效识别并降低模型对 shortcut 的依赖。

**⚠️ 局限性**

限制在于仍需人工审阅最终特征列表，遮蔽方法在某些数据集可能导致性能下降，且对极端环境或新协议的适用性尚未完全验证。

---

## 236. Distributed Linearly Separable Computation with Arbitrary Heterogeneous Data Assignment

**arXiv ID:** 2601.10177 | [PDF](https://arxiv.org/pdf/2601.10177v1)

**作者:** Ziting Zhang `[一作]`, Giuseppe Caire `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

研究分布式线性可分离计算中任意异构数据分配的通信-计算平衡，提出了通用可达与下界并在整数与分数通信成本下给出可行方案；

**💡 创新点**

提出了与数据分配结构相关的通用稀疏度参数α与t，得到在任意异构分配下的通用可达/下界；在特定参数范围内实现了最优；

**🔧 技术方法**

基于编码理论的稀疏矩阵分解方法，利用零向量子空间构造编码矩阵与虚拟需求矩阵；使用Schwartz–Zippel引理证明矩阵满秩；

**📊 数据集**

本文为理论性研究，无需实际数据集；

**📈 对比分析**

与已有的循环分配、重复编码方案进行对比，证明在给定通信成本下可达维数更高，曲线上实现了更优的通信-可计算维度折衷；

**⚠️ 局限性**

对偶参数区间仍有下界与可达不匹配；未考虑straggler、失真与存储空间限制；

---

## 237. A Low-Complexity Architecture for Multi-access Coded Caching Systems with Arbitrary User-cache Access Topology

**arXiv ID:** 2601.10175 | [PDF](https://arxiv.org/pdf/2601.10175v1)

**作者:** Ting Yang `[一作]` (Huazhong University of Science and Technology), Giuseppe Caire `[通讯]` (Technische Universität Berlin)

**通讯引用:** 28788 | [OpenAlex ID](https://openalex.org/A5058252389)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种针对任意用户–缓存访问拓扑的多访问编码缓存（MACC）系统低复杂度交付方案，利用图着色将交付问题转化为冲突图的着色，从而减少通信负载。

**💡 创新点**

创新点：
• 统一的冲突图框架，将任意拓扑下的编码交付问题等价于图着色；
• 设计基于图神经网络（GNN）的学习式着色器，既能处理大规模稠密图，又能在不同拓扑、不同用户数下泛化；
• 推导基于IC逆界的贪心近似，显著降低逆界计算复杂度并保持与最优逆界的误差在3％以内。

**🔧 技术方法**

使用技术：
• 经典DSatur贪心着色作为基准；
• 设计邻居增强消息传递的GNN（含注意力权重、GCN层、嵌入式颜色投影）并以Potts能量为无监督损失；
• 颜色冲突修复后处理；
• 贪心逆界算法（基于用户请求集合的交叉）实现低复杂度逆界。

**📊 数据集**

数据集：
• 采用随机生成的用户–缓存访问拓扑，满足每个用户至少连一缓存、每个缓存至少被一用户访问；
• 用户数K在4–20范围内随机采样；
• 缓存数Λ固定为10；
• 对每组拓扑随机生成多组用户需求，进行多次实验。

**📈 对比分析**

对比方法与性能：
• 对比DSatur、GIN基图着色、IC逆界与贪心逆界；
• 交付负载：DSatur与IC逆界相差≤1.2％；GNN交付负载与DSatur相差≤10％；
• 计算时间：GNN比DSatur快20–30倍，GIN更快1–2个数量级；贪心逆界相较IC逆界的计算时间提升1e6倍，负载误差≤3％；
• 在K从6到19的不同规模上，GNN在保持接近DSatur负载的同时，持续保持显著的速度优势，且能跨越不同用户数泛化。

**⚠️ 局限性**

局限性：
• GNN虽然速度快，但不保证得到全局最优着色；
• 需要在随机拓扑上进行大量训练，训练成本和模型泛化仍受限于所覆盖的拓扑分布；
• 方案假设缓存采用MN未编码放置，若采用其他放置策略需重新设计；
• 对极大规模稠密图的内存占用仍可能成为瓶颈，需进一步优化模型规模。

---

## 238. ReasAlign: Reasoning Enhanced Safety Alignment against Prompt Injection Attack

**arXiv ID:** 2601.10173 | [PDF](https://arxiv.org/pdf/2601.10173v1)

**作者:** Hao Li `[一作]` (Washington University in St. Louis), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在大语言模型中嵌入结构化推理步骤（分析-推理-答复），以在面对间接注入攻击时保持任务连贯性并防止被劫持。

**💡 创新点**

创新点包括：1）将推理过程显式化为三阶段结构，帮助模型识别与用户意图冲突的注入指令；2）在推理中加入注入指令标记与目标答案约束，提升推理合理性；3）使用测试时缩放与逻辑评判模型（通过DPO训练的评分器）在推理树中选取最佳路径，进一步提升安全性与一致性。

**🔧 技术方法**

核心技术为：LoRA 低秩适配器微调、Chain-of-Thought 推理模板、测试时多节点搜索（beam search）与逻辑评判模型、Direct Preference Optimization (DPO)。

**📊 数据集**

使用的数据集包括 SQuADv2（原始问答）、TaskTracker（多样化注入触发器）以及 BeaverTails（安全/不安全指令），并在此基础上构造结构化注入样本进行安全对齐训练。

**📈 对比分析**

与未防御的 Llama‑3.1‑8B‑Instruct、SecAlign 及 Meta‑SecAlign 对比。评估覆盖一般知识、指令遵循和代理工作流等四类基准。结果显示：在不受攻击时，ReasAlign 与未防御模型几乎等价；在攻击环境下，ReasAlign 在 CySE 基准实现 94.6% utility、3.6% ASR，远优于 Meta‑SecAlign 的 56.4% utility 与 74.4% ASR；在一般知识任务中平均性能最佳；整体安全-效用平衡显著提升。

**⚠️ 局限性**

主要局限：推理过程增加算力与 token 消耗，尤其在测试时多节点搜索时成本显著上升；目前仅在 Llama‑3.1‑8B‑Instruct 上验证，需进一步验证在更大模型上的可迁移性；过度推理可能在某些简单任务中产生不必要的开销。

---

## 239. Early Fault Detection on CMAPSS with Unsupervised LSTM Autoencoders

**arXiv ID:** 2601.10269 | [PDF](https://arxiv.org/pdf/2601.10269v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 240. Hierarchical Refinement of Universal Multimodal Attacks on Vision-Language Models

**arXiv ID:** 2601.10313 | [PDF](https://arxiv.org/pdf/2601.10313v1)

**作者:** Peng-Fei Zhang `[一作]` (University of Queensland), Zi Huang `[通讯]` (University of Queensland)

**通讯引用:** 13114 | [OpenAlex ID](https://openalex.org/A5078170935)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了 Vision‑Language Pre‑Trained (VLP) 模型的通用多模态对抗攻击，提出 Hierarchical Refinement Attack (HRA) 框架。

**💡 创新点**

创新点在于层次化细化 UAP：① 将图像与扰动分离，② 采用 ScMix 语义保持的数据增强、局部效用提升；③ 引入未来感知动量以避免局部最优；④ 在文本端实现无需词表的统一词替换攻击，从而显著提升跨模型、跨任务的可迁移性。

**🔧 技术方法**

使用技术包括：UAP 学习、PGD 优化、ScMix 数据增强、局部效用增强、未来感知动量、词重要性评估与替换、KL 散度损失、梯度聚合等。

**📊 数据集**

使用数据集：Flickr30K、MSCOCO、RefCOCO+，在多种 VLP 模型（CLIP、ALBEF、TCL、BLIP）上进行实验。

**📈 对比分析**

与 AdvCLIP、SGA、ETU、FD‑UAP、C‑PGC 等现有方法对比，HRA 在图像‑文本检索、图像描述、视觉定位等任务上多场景下均取得最高或第二高的攻击成功率，跨模型、跨任务迁移性能明显提升。

**⚠️ 局限性**

局限性：文本攻击因离散性仍易被人察觉；在低扰动预算下的可迁移性仍有限，且对极低预算或更强防御机制的鲁棒性尚未充分验证。

---

## 241. Atelier à la conférence IHM 2025 : RA Permanente

**arXiv ID:** 2601.10291 | [PDF](https://arxiv.org/pdf/2601.10291v1)

**作者:** Maxime Cauz `[一作]` (Namur Digital Institute), Emmanuel Dubois `[通讯]` (University of Toulouse)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

组织了一场关于持续增强现实（PAR）的研讨会，并通过个人展示、桌面讨论和WAAD方法收集并系统化了多方观点与使用场景。

**💡 创新点**

创新点在于将PAR划分为六大研究轴，融合三种主流视角（环境增强、空间计算、元宇宙），并通过Affinity Diagram构建了从技术、社会与伦理维度的讨论框架。

**🔧 技术方法**

主要采用会议讨论、个人演讲、桌面讨论和WAAD（Affinity Diagram）等方法；技术层面引用了增强现实、IoT、空间计算、元宇宙等相关概念与技术。

**📊 数据集**

未使用传统公开数据集，主要依据参与者的讨论记录、演讲材料及现场笔记。

**📈 对比分析**

未进行实验性对比或性能评估，而是通过与已有三种视角的概念对照进行定性分析。

**⚠️ 局限性**

局限性包括样本规模小、缺乏定量验证、讨论高度主观、未充分探讨伦理、隐私与社会影响等问题。

---

## 242. Queueing-Aware Optimization of Reasoning Tokens for Accuracy-Latency Trade-offs in LLM Servers

**arXiv ID:** 2601.10274 | [PDF](https://arxiv.org/pdf/2601.10274v1)

**作者:** Emre Ozbas `[一作]` (Bilkent University), Melih Bastopcu `[通讯]` (Bilkent University)

**通讯引用:** 232 | [OpenAlex ID](https://openalex.org/A5000544057)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对单台LLM服务器处理异构查询任务，提出基于M/G/1排队理论的推理令牌分配优化模型，求解在准确率与系统延迟之间平衡的最优令牌预算。

**💡 创新点**

创新点在于：①证明目标函数在稳定域内严格凹，确保唯一最优解；②设计投影固定点迭代和投影梯度上升两种收敛性强、可行域内的求解方法，并给出显式的收敛条件；③将连续解通过整数化投影得到可执行的令牌分配，并给出了整数化误差下界。

**🔧 技术方法**

主要技术包括：M/G/1排队模型、凸优化与KKT条件、Lambert-W函数求解固定点、投影固定点迭代、投影梯度上升、梯度的Lipschitz连续性证明以及整数化取整方法。

**📊 数据集**

实验使用六个公开数据集：AIME、GSM8K、GPQA、CRUXEval、BBH、ARC-Challenge，部署在Qwen3-8B模型上。

**📈 对比分析**

与均匀令牌分配（0/100/500）对比，最优异构分配在目标函数上显著优于所有方案；对GSM8K预算的敏感性实验验证了理论最优点，并通过准确率提升与系统时延降低体现性能提升。

**⚠️ 局限性**

局限性包括：仅考虑单机FIFO服务器，未覆盖多节点并行或微调等场景；模型假设服务时间线性、准确率函数已拟合；整数化仅为近似，实际部署仍需评估；排队模型假设Poisson到达，可能与真实请求分布不完全吻合。

---

## 243. Transmission Mask Analysis for Range-Doppler Sensing in Half-Duplex ISAC

**arXiv ID:** 2601.10259 | [PDF](https://arxiv.org/pdf/2601.10259v1)

**作者:** Dikai Liu `[一作]` (Beijing University of Posts and Telecommunications), Jianhua Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 73100 | [OpenAlex ID](https://openalex.org/A5100609374)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

分析半双工ISAC系统中周期性传输掩模的期望二次幅度响应 𝔼{|r(k,l,ν)|²，并探讨其在不同多普勒动态范围内的特性。

**💡 创新点**

证明了范围旁瓣（k≠l）对多普勒不变，确定在中等动态范围内 Singer 类相位掩模（cds）在最大化最小侧瓣抑制与主瓣波动最小化方面为最优；在高动态范围内揭示主瓣波动与多普勒旁瓣能量之间不可避免的折衷，并给出其可达的极限。

**🔧 技术方法**

使用符号期望分析、周期自相关函数、离散傅里叶变换、凹凸性与Jensen不等式等数学工具，推导出 𝔼{|r(k,l,ν)|² 的闭式表达式，并对不同掩模结构进行理论性能评估。

**📊 数据集**

通过仿真验证，使用长度为 N=63 的三种掩模（Singer cds、随机掩模、蜂巢掩模），每种掩模重复 M=50 次，采用 16QAM 调制（μ₄≈1.32），计算并绘制完整及局部多普勒响应以及主瓣波动统计。

**📈 对比分析**

与随机掩模和蜂巢掩模比较，Singer cds 在中等动态范围内多普勒旁瓣平均值最低、主瓣波动最小；在高动态范围内虽然主瓣波动可控，但多普勒侧瓣总能量相对较高，蜂巢掩模虽在某些多普勒频点侧瓣低，但出现等间隔格子峰，整体性能低于 Singer cds。

**⚠️ 局限性**

局限性：高动态范围下理想掩模不可实现，主瓣波动与多普勒旁瓣能量不可同时最小；文中未考虑实际干扰、噪声、硬件限制以及多普勒不确定性分辨策略，缺乏针对这些实际问题的完整设计与评估。

---

## 244. Evolving with AI: A Longitudinal Analysis of Developer Logs

**arXiv ID:** 2601.10258 | [PDF](https://arxiv.org/pdf/2601.10258v1)

**作者:** Agnia Sergeyuk `[一作]` (JetBrains Research), Iftekhar Ahmed `[通讯]` (University of California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过对800名专业开发者两年连续的IDE使用日志与62名参与者的问卷访谈进行混合方法研究，评估AI编码助手在实际开发工作中的五大维度（生产力、代码质量、代码编辑、代码重用与上下文切换）的长期影响。

**💡 创新点**

创新点在于：①首次结合大规模细粒度日志与自我感知数据，客观与主观视角同步观察AI工具的真实影响；②提出并量化了五个可操作的工作流程维度；③揭示了使用者的感知与行为之间的系统性差异，表明AI工具重塑工作节奏而非直接替代人力。

**🔧 技术方法**

技术方法包括：①在JetBrains IDE中收集匿名事件日志（键入、删除、粘贴、调试、窗口激活等）；②使用混合线性回归模型分析时间序列趋势；③基于问卷的Likert量表进行描述性统计；④半结构化访谈获取质性洞见。

**📊 数据集**

数据集由两部分组成：①151,904,543条IDE事件日志，覆盖800名开发者（400 AI用户，400非用户）两年跨度；②62名问卷回应（其中62人中包含不同角色、经验、工具使用时间），以及5份后访谈记录。

**📈 对比分析**

比较方法为将AI用户与非用户在每个维度上的月度计数进行混合线性回归，检验组别、时间以及组别×时间交互的显著性。结果显示：AI用户在代码写入量（+587字符/月）和编辑量（+102删除/月）上显著快于非用户；在外部粘贴与窗口激活上亦呈现正向增长，而非用户表现为下降或缓慢增长。性能（即统计显著性）均达p<0.05，体现出持续使用AI工具对工作行为的稳健影响。

**⚠️ 局限性**

局限性包括：①仅覆盖JetBrains生态系统，无法代表所有IDE或AI工具；②用户划分基于是否使用JetBrains AI助手，可能遗漏其他AI使用；③为观察性研究，缺乏随机分配，组间差异可能受选择偏差影响；④日志记录不包含任务意图或代码质量真实评估，仅以代理指标衡量；⑤样本规模虽大但仍有限，且多为早期采用者，结果对广泛开发者群体的推广需谨慎。

---

## 245. TRIM: Hybrid Inference via Targeted Stepwise Routing in Multi-Step Reasoning Tasks

**arXiv ID:** 2601.10245 | [PDF](https://arxiv.org/pdf/2601.10245v1)

**作者:** Vansh Kapoor `[一作]` (Carnegie Mellon University), Aviral Kumar `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TRIM 方法，针对多步推理任务在每一步仅将关键步骤路由到大型高质量模型，其余步骤由小型模型完成，显著降低高成本模型的调用次数。

**💡 创新点**

创新点在于从传统的查询级路由转向步级路由；引入多种路由策略（阈值、强化学习、POMDP），并利用过程奖励模型（PRM）对每一步的正确性进行评估，实现在保持或提升准确率的同时大幅削减昂贵模型的 token 使用。

**🔧 技术方法**

使用的技术包括：过程奖励模型 (PRM) 评估中间步骤；阈值决策；基于 Transformer 的 RL 策略（PPO）；POMDP 解决方案（SARSOP）；多模型协同（小模型 + 大模型）；以及对成本与性能进行量化的 CPT/IBC 指标。

**📊 数据集**

主要数据集：MATH‑500、AIME、OlympiadBench、Minerva Math；模型对为 Qwen2.5‑3B‑Instruct（小模型）与 Claude 3.7 Sonnet（大模型）配合使用。

**📈 对比分析**

与现有查询级路由方法（RouteLLM、Smoothie、AutoMix）以及内部强化学习基线进行对比。TRIM‑Thr 在低成本场景下比基线高 5 倍；TRIM‑Agg 在高成本场景下实现 95% 的性能差距，使用 80% 更少的大模型 token；TRIM‑POMDP 在低预算下表现最佳。实验还展示了跨数据集的强泛化能力。

**⚠️ 局限性**

局限性包括：依赖 PRM 的准确性，若 PRM 估计不佳会影响路由决策；目前仅在步级 granularity 上实现，未考虑更细粒度的 token 级路由；在极端高预算场景下 RL 策略学习可能仍受奖励稀疏影响；以及对不同模型对的适配性仍需进一步验证。

---

## 246. Fundamental Limitations of Favorable Privacy-Utility Guarantees for DP-SGD

**arXiv ID:** 2601.10237 | [PDF](https://arxiv.org/pdf/2601.10237v1)

**作者:** Murat Bilgehan Ertan `[一作]` (Vrije Universiteit Amsterdam), Marten van Dijk `[通讯]` (CWI Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了单轮随机打乱与Poisson采样下的DP‑SGD在最坏情况对手模型中的隐私‑效用限制，给出了噪声乘子和分离度的下界，并通过实验验证其影响。

**💡 创新点**

在f‑DP框架下首次证明单轮随机打乱DP‑SGD存在不可逾越的噪声与可区分度平衡限制，并将该结论推广至Poisson采样。

**🔧 技术方法**

采用f‑DP假设检验与几何分离度分析，构造子最优假设检验以及子极大化混合论证。

**📊 数据集**

使用CIFAR‑10/100、SVHN、AG News等图像与文本基准。

**📈 对比分析**

与无噪声和仅裁剪的基线对比，发现理论下界噪声水平下的准确率显著下降，验证理论限制。

**⚠️ 局限性**

限制源于最坏情况对手模型，若不放宽对手假设或改变算法设计，则无法同时实现低噪声和低可区分度。

---

## 247. Agentic Pipelines in Embedded Software Engineering: Emerging Practices and Challenges

**arXiv ID:** 2601.10220 | [PDF](https://arxiv.org/pdf/2601.10220v1)

**作者:** Simin Sun `[一作]` (Chalmers University of Technology and University of Gothenburg), Miroslaw Staron `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对嵌入式软件工程组织采用生成式 AI 进行定性研究，挖掘 11 项新兴实践与 14 项挑战，并提出治理与流程框架

**💡 创新点**

首次系统性地从工业实践视角审视生成式 AI 在嵌入式领域的可行性，提出 AI‑friendly artifacts、MCP/AICP 规范与人机协同策略

**🔧 技术方法**

采用半结构化访谈、头脑风暴与主题分析等定性方法，对专家访谈记录进行编码与归纳

**📊 数据集**

10 名高级工程师来自 4 家企业的访谈与头脑风暴记录（共 2-5 名每组）

**📈 对比分析**

无定量对比或性能评估，主要通过主题归纳得到实践与挑战清单，未给出数值指标

**⚠️ 局限性**

样本规模小、行业代表性有限、仅为主观访谈、缺乏客观实验与可复现性，且 AI 技术快速演进导致研究结果可能快速过时

---

## 248. Topo-RAG: Topology-aware retrieval for hybrid text-table documents

**arXiv ID:** 2601.10215 | [PDF](https://arxiv.org/pdf/2601.10215v1)

**作者:** Alex Dantart `[一作]` (Humanizing Internet), Marco Kóvacs-Navarro `[通讯]` (Humanizing Internet)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种名为Topo‑RAG的检索增强生成框架，针对企业文档的混合结构（文本与表格）采用拓扑感知路由，将文本与表格分别通过密集检索和基于单元格的晚期交互（Cell‑Aware Late Interaction）处理，最终通过交叉编码器重新排序并交付给LLM生成答案。

**💡 创新点**

创新点在于：① 拓扑感知路由器能够自动区分叙事段落与结构化表格；② 对表格采用多向量表示并使用最大相似度（MaxSim）在单元格层面进行交互，避免了传统线性化导致的结构丢失；③ 结合WARP、CRISP等优化技术实现多向量索引的内存与延迟可接受。

**🔧 技术方法**

使用的技术包括：BGE‑M3或text‑embedding‑3‑large（文本向量化）、ColBERT‑style Cell‑Aware Late Interaction（表格向量化）、FAISS（文本近似最近邻）、WARP（多向量高效检索）、CRISP（向量聚类剪枝）、自定义的结构密度得分（SDS）路由器、Cross‑Encoder重新排序以及LLM（如GPT‑4o）生成。

**📊 数据集**

数据集为自研的SEC‑25（Synthetic Enterprise Corpus 2025），包含约1万份企业文档（50%叙事、50%结构），包括可持续性报告、法律合同、电子邮件、结算表格、库存清单等，查询集包含500个多难度问题（事实检索、单元格精确检索、多跳混合检索）。

**📈 对比分析**

与三种基线（Naive RAG、Advanced RAG、TabRAG）比较，Topo‑RAG在nDCG@10上分别比TabRAG提升约22.9%（单元格查询）和30.0%（多跳查询），整体平均提升18.4%；在Recall@20、hallucination率等指标也显著优于基线；优化后索引体积为4.1 GB、查询延迟≈85 ms，已接近工业可接受范围。

**⚠️ 局限性**

局限性包括：① 多向量索引仍比单向量索引占用更高内存与存储；② 对极大规模表格（>100列）仍需进一步压缩与聚类处理；③ 路由器阈值需针对不同领域微调，可能导致误分类；④ 当前实现仅支持表格和文本两种拓扑，尚未扩展至图表、流程图等更复杂结构。

---

## 249. PADER: Paillier-based Secure Decentralized Social Recommendation

**arXiv ID:** 2601.10212 | [PDF](https://arxiv.org/pdf/2601.10212v1)

**作者:** Chaochao Chen `[一作]` (Zhejiang University), Yachuan Liu `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种基于Paillier加密的去中心化社交推荐系统PADER，实现用户与商家在不泄露嵌入、评分或社交关系的前提下完成模型训练与推理。

**💡 创新点**

创新点包括将社交正则化模型转化为两方安全多项式评估，提出自然顺序安全计算协议；同时设计针对AHE的最优数据打包与重新打包方案，大幅提升效率。

**🔧 技术方法**

主要使用技术为Paillier加密（加法同态）、固定点数编码、两方安全多项式计算协议、数据打包与重新打包、SGD优化以及相关加密与算术电路实现。

**📊 数据集**

实验使用合成数据以及真实社交推荐数据集Epinions和Douban进行验证。

**📈 对比分析**

与传统双边拆分计算、无打包方案及FHE CKKS进行对比，PADER在通信量上减少20~100倍、在小规模任务中计算时间更快，总体性能显著优于现有方法。

**⚠️ 局限性**

在极大规模商品数（如2^14项）时，PADER的纯计算时间不及CKKS张量方案，对极大item数量的扩展性仍有限。

---

## 250. Terrain-Adaptive Mobile 3D Printing with Hierarchical Control

**arXiv ID:** 2601.10208 | [PDF](https://arxiv.org/pdf/2601.10208v1)

**作者:** Shuangshan Nors Li `[一作]` (University of Washington), J. Nathan Kutz `[通讯]` (University of Washington)

**通讯引用:** 28402 | [OpenAlex ID](https://openalex.org/A5083450863)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种集成AI预测与分层控制的移动3D打印框架，实现了在不规则地形上既保持机动性又保持亚厘米打印精度。

**💡 创新点**

三层分层控制架构实现路径规划、预测控制和硬件执行的时间尺度分离；多模态传感融合与稀疏回归训练的地形扰动预测模块实现主动补偿；低频地形跟随与高频精修的频率分解协同策略。

**🔧 技术方法**

轻量化前馈网络进行扰动预测；模型预测控制（MPC）与频率分解协同；多模态传感融合（IMU、RGB摄像头、RGB‑D摄像头、视觉跟踪、深度剖面）与稀疏回归特征选择；实时控制循环（100 Hz硬件层）。

**📊 数据集**

约7小时（4小时仿真+3小时户外）运行数据，涵盖平坦混凝土、草地、碎石、斜坡等五种地形，共约1400段轨迹，200 m总距离，50 Hz采样。

**📈 对比分析**

与无预测的对比实验，使用MPC+预测显著降低末端执行器误差，实测末端误差平均<5.1 mm，层高一致±0.8 mm；MPC计算时间≈8.3 ms，预测延迟≈12.5 ms；在斜坡、草地等多种地形上保持子厘米级打印精度且无累计漂移。

**⚠️ 局限性**

受快速变化环境中传感失效、极端地形障碍（台阶、深坑）影响；仅针对地形扰动的运动补偿，未涵盖材料流变与挤出动态；对大尺度连续构件需多机器人协同规划；需要高质量地形地图；对极低光照或高反射条件下视觉传感效果下降。

---

## 251. An Efficient Long-Context Ranking Architecture With Calibrated LLM Distillation: Application to Person-Job Fit

**arXiv ID:** 2601.10321 | [PDF](https://arxiv.org/pdf/2601.10321v1)

**作者:** Warren Jouanneau `[一作]` (Malt), Marc Palyart `[通讯]` (Malt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级长文本重排序模型，用生成式LLM作为教师进行语义校准并蒸馏给学生模型，实现实时可解释的人才-项目匹配评分。

**💡 创新点**

①使用生成式LLM提供细粒度语义标注，构建可解释的校准评分空间；②提出基于发言块的跨注意力架构，既能处理长文本又具低算力；③设计包含排名与校准的蒸馏损失，兼顾精确度与全局可比性。

**🔧 技术方法**

生成式大型语言模型（LLM）教师+知识蒸馏；双分支编码+跨注意力对比；自定义损失（CMMD、CLID+MSE）；预计算嵌入加速推理。

**📊 数据集**

Malt公开的千级多语言简历与项目简介数据集（约850k自由职业者，项目多样化），并使用LLM生成的人工标注作为教师监督。

**📈 对比分析**

与零射击与微调的LLM基线（Qwen3、Gemma3）以及传统检索+排序模型对比；在相关性、排名（MRR、NDCG、mAP）和校准（MAE、Δ_mean、Δ_IQR）指标上，学生模型在大多数指标上位列前二，且推理速度从7分钟降至287 ms。

**⚠️ 局限性**

对教师标签的偏差与质量缺乏深入评估；仅进行粗略性别公平性检验；模型对合成平均匹配样本区分度仍有限；需进一步构建高质量标注集、监控偏差及适应模型漂移。

---

## 252. ADVOSYNTH: A Synthetic Multi-Advocate Dataset for Speaker Identification in Courtroom Scenarios

**arXiv ID:** 2601.10315 | [PDF](https://arxiv.org/pdf/2601.10315v1)

**作者:** Aniket Deroy `[一作]` (Indian Institute of Technology), Aniket Deroy `[通讯]` (Indian Institute of Technology)

**通讯引用:** 206 | [OpenAlex ID](https://openalex.org/A5078909351)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

基于Speech Llama Omni生成了一个包含10位合成律师声音的Advosynth-500数据集，并提出了在法庭场景下的说话人识别挑战。

**💡 创新点**

创新点在于使用端到端语音转语音模型生成高度逼真的多情境律师音频，保留了声音身份的细微特征，并将法律对话的多样性和竞争性（如交叉质询、突发辩论）嵌入到数据集里。

**🔧 技术方法**

主要技术包括Speech Llama Omni的语音生成、声纹嵌入调制、声学特征（基频、语速、音色、主张强度）矩阵设计，以及对抗式语音对齐与分段标注。

**📊 数据集**

使用的数据集是自研的Advosynth-500，由100条合成音频（10名律师×10场景）组成，总计500个.wav文件，公开托管在GitHub。

**📈 对比分析**

方法上通过闭集分类任务评估说话人识别模型（如X‑vector、ECAPA‑TDNN）对合成声音的识别精度，但论文未给出具体实验结果或性能指标。

**⚠️ 局限性**

局限性包括：①合成语音缺乏真实法庭噪声与人类说话者的生理差异，②仅为闭集实验，未验证对未见合成或真实声音的泛化能力，③缺乏量化评测与基准对比。

---

## 253. Multilinguality as Sense Adaptation

**arXiv ID:** 2601.10310 | [PDF](https://arxiv.org/pdf/2601.10310v1)

**作者:** Jan Christian Blaise Cruz `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Alham Fikri Aji `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**通讯引用:** 2868 | [OpenAlex ID](https://openalex.org/A5112924039)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在英文预训练的Backpack语言模型基础上，利用平行语料通过对语义层面（sense）和上下文层面的对比学习，结合语言模型损失，将模型迁移到目标语言，实现多语言流利度提升。

**💡 创新点**

将跨语言迁移视作“语义适配”，通过显式对齐词的潜在语义混合（sense mixtures）而非仅参数共享；采用双向InfoNCE对sense层和上下文层进行对齐，保持语义结构与可解释性。

**🔧 技术方法**

Backpack Transformer架构、K个latent sense向量、soft mixture weighting、双向InfoNCE对比学习、标记平滑交叉熵语言模型损失、三阶段训练（对齐、联合、打磨）及余弦余温调度。

**📊 数据集**

使用OPUS收集的四种语言（爱沙尼亚语、土耳其语、印尼语、斯瓦希里语）平行语料；评测在FLORES-200、XCOPA、XStoryCloze、Belebele等基准上。

**📈 对比分析**

与从零开始的Goldfish、多语言对齐（MCL）以及英语GPT‑2微调对比，SENSIA在相同参数规模下在四种语言上均超越MCL，往往与Goldfish相当甚至更好；在大型多语言模型（XGLM‑7B5、BLOOMZ‑7B1）上差距缩小。

**⚠️ 局限性**

受限于使用英语GPT‑2 BPE词表，只能适用于拉丁字母语言；对非拉丁脚本（如中文）效果下降；依赖高质量平行文本，低资源场景需进一步研究；未验证大模型（7B+）和指令级迁移的可行性。

---

## 254. The Straight and Narrow: Do LLMs Possess an Internal Moral Path?

**arXiv ID:** 2601.10307 | [PDF](https://arxiv.org/pdf/2601.10307v1)

**作者:** Luoming Hu `[一作]` (Dalian University of Technology), Hongfei Lin `[通讯]` (Dalian University of Technology)

**通讯引用:** 8494 | [OpenAlex ID](https://openalex.org/A5023931221)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将道德基础理论映射到LLM潜在空间，提取可操控的道德向量并在推理时动态注入，构建了一种内在的安全防御机制（Adaptive Moral Fusion，AMF）。

**💡 创新点**

创新点在于：①发现LLM中存在可线性分离的多维道德子空间，并且英中两种语言共享但有细微差异的子空间；②提出可微调的道德向量和自适应融合方法，能够在保持帮助性同时降低越狱和错误拒绝；③通过激活或语言解码器验证向量的道德语义。

**🔧 技术方法**

技术手段包括：多语言线性探针、道德向量提取（均值差法）、推理时激活注入、Adaptive Moral Fusion动态权重计算，以及激活原语（Activation Oracle）进行语义解码。

**📊 数据集**

使用了自构造的双语道德善恶句对数据集（基于MFD和C‑MFD），以及公开的Llama‑3.1系列模型进行实验；评测数据包括HarmBench（越狱攻击）和XSTest（错误拒绝）。

**📈 对比分析**

与标准RLHF/安全提示基线相比，AMF在HarmBench上将攻击成功率从约44%降至19.66%，在XSTest上将错误拒绝率降至2.00%，与Claude‑3.5‑Sonnet相近；单向向量注入效果更弱，证明动态融合更有效。

**⚠️ 局限性**

局限性：仅验证了英中两语种，难以推广到低资源或非WEIRD文化；依赖Moral Foundations Theory，忽略其他伦理框架；方法对模型架构有一定依赖，主要在Llama‑3.1上验证；生成数据集可能带来生成模型的偏见；双重使用风险——反道德向量可用于制造有害内容。

---

## 255. DanQing: An Up-to-Date Large-Scale Chinese Vision-Language Pre-training Dataset

**arXiv ID:** 2601.10305 | [PDF](https://arxiv.org/pdf/2601.10305v1)

**作者:** Hengyu Shen `[一作]` (Glint Lab), Kaicheng Yang `[通讯]` (Glint Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并公开了100M规模的中文视觉‑语言对齐数据集DanQing，结合严格的多阶段筛选管道构建高质量样本，并通过对比实验验证其优越性能。

**💡 创新点**

创新点在于：① 使用2024‑2025年最新Common Crawl中文图文数据，实时捕捉语义趋势；② 设计细粒度安全、结构、质量、信息密度等多维度过滤流程，显著提升数据质量；③ 通过交叉模态相似度阈值进一步保证图文匹配精度。

**🔧 技术方法**

技术手段包括CLIP/SigLIP对比学习框架、BERT/Chinese‑CLIP‑L14文本与图像编码器、FastText+OpenCC文本语言检测、图像熵与像素标准差筛选、NSFW检测模型、聚类去冗余、语义对齐阈值筛选等。

**📊 数据集**

主要数据来源为Common Crawl 2024‑2025的中文图文对，构成DanQing；与Wukong、Zero、TaiSu等现有中文图文数据集进行对比。

**📈 对比分析**

采用SigLIP2模型进行持续预训练，在零样本分类、跨模态检索、LMM任务等下游基准上进行评测。结果显示，DanQing在所有任务上均优于对比数据集，零样本分类提升约7.8%，检索准确率提升约2.5%，LMM平均分提升约0.6%。

**⚠️ 局限性**

局限性：数据仅覆盖简体中文，缺乏多语言、多标签细粒度；对极短文本或非图像内容的覆盖仍有限；虽然捕捉新兴概念更好，但仍可能遗漏极端稀有或专业领域内容。

---

## 256. In-Context Source and Channel Coding

**arXiv ID:** 2601.10267 | [PDF](https://arxiv.org/pdf/2601.10267v1)

**作者:** Ziqiong Wang `[一作]` (Zhejiang University), Honggang Zhang `[通讯]` (Macau University of Science and Technology)

**通讯引用:** 11711 | [OpenAlex ID](https://openalex.org/A5100626780)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种接收端的 In-Context Decoding (ICD) 框架，结合 ECCT 可靠性估计和上下文信息，生成多候选比特流并通过 LLM 源解码提高 SSCC 在低 SNR 下的鲁棒性。

**💡 创新点**

创新点在于：①将 ECCT 产生的比特可靠性与上下文信息融合生成置信排序候选集；②使用 Metropolis-Hastings 采样实现多样性保留的候选子集选择；③通过可靠性–似然融合规则最终选取最优重构。

**🔧 技术方法**

核心技术包括：错误校正码 Transformer (ECCT)、算术编码 (AC) 搭配大语言模型、候选生成 (CCG)、候选采样 (CCS)、似然排名 (CLR)，以及基于 LLM 的语义评估。

**📊 数据集**

使用欧洲议会会议记录数据集（约 200 万句子），采用 GPT‑2（base、medium、large、XL）作为源编码模型。

**📈 对比分析**

与传统 SSCC（Huffman‑SSCC、LLM‑AC+ECCT）以及代表性 JSCC（DeepSC、UT、UT+量化）在 AWGN 与 Rayleigh 通道上对比，ICD 在低 SNR 区间实现了显著提升（BLEU‑1/4、语义相似度均高于基线），并保持了较低的解码开销。

**⚠️ 局限性**

局限包括：仍依赖上下文信息的可用性，候选生成规模受比特长度限制；ICD 主要提升源端解码，对极低 SNR 时的残余错误仍有一定敏感性；实验主要集中在文本数据，未验证对图像等多模态内容的适用性。

---

## 257. An Ensemble of Evolutionary Algorithms With Both Crisscross Search and Sparrow Search for Processing Inferior Individuals

**arXiv ID:** 2601.10263 | [PDF](https://arxiv.org/pdf/2601.10263v1)

**作者:** Mingxuan Du `[一作]` (China University of Geosciences), Chengjun Li `[通讯]` (China University of Geosciences)

**通讯引用:** 240 | [OpenAlex ID](https://openalex.org/A5100454865)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将交叉搜索和鹦鹉搜索作为次级进化算法集成到EA4eig中，仅在劣势个体上操作，形成EA4eigCS以提升长时间搜索性能。

**💡 创新点**

创新点在于：①引入两种新型搜索方法作为次级算法仅作用于劣势个体；②通过改进劣势个体来改变种群分布，从而突破停滞；③在轮盘赌选择主算法的框架下引入次级算法，构成多算法混合体系。

**🔧 技术方法**

使用的技术包括：DE变体（CoBiDE、IDEbd、jSO）、CMA‑ES、交叉搜索、鹦鹉搜索、轮盘赌概率选择、线性种群缩减、Wilcoxon秩和检验与Friedman检验等。

**📊 数据集**

使用的评估数据集为CEC 2020/2021/2022长时间搜索基准测试函数。

**📈 对比分析**

通过与IMODE、NL‑SHADE‑RSP、APGSK‑IMODE、MLS‑L‑SHADE、EA4eig、NL‑SHADE‑LBC、AMCDE等对标算法进行Wilcoxon秩和检验和Friedman检验比较，结果显示EA4eigCS在大多数维度上排名第一或与EA4eig相当，整体性能优于同类长时间搜索算法。

**⚠️ 局限性**

局限性在于：仅针对劣势个体使用次级算法，未充分挖掘主算法的潜能；参数设置仍需针对不同问题手工调优；在某些函数上收敛曲线与EA4eig相似，表明改进空间仍存在。

---

## 258. NoReGeo: Non-Reasoning Geometry Benchmark

**arXiv ID:** 2601.10254 | [PDF](https://arxiv.org/pdf/2601.10254v1)

**作者:** Irina Abdullaeva `[一作]` (FusionBrain Lab), Andrey Kuznetsov `[通讯]` (FusionBrain Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了NoReGeo基准，评估大型语言模型在不进行链式推理或代数计算的前提下，直接对几何图形进行空间判断的能力。

**💡 创新点**

创新点在于将几何评测从复杂证明转向可即时回答的基准，消除了推理步骤，只关注模型对空间关系的本能感知；并通过多模态（文本+点图/完整图）对比验证视觉表征的作用。

**🔧 技术方法**

采用结构化生成（JSON输出）、固定温度与长度、并用CLIP‑ViT‑B/32等视觉编码器进行线性探测；同时使用OpenAI的Outlines、xgrammar、vllm等库实现高效推理与评测。

**📊 数据集**

使用自己构造的2,500条几何题目，覆盖25类基本几何（面积比较、坐标、对称、圆性质等），每题提供文本、点图和完整图三种形式。

**📈 对比分析**

通过对45+模型（GPT‑4、Qwen、LLaMA、Phi‑3.5等）在文本、点图和完整图三种输入模式下的准确率进行对比，最佳LLM仅达到约65%准确率，最佳VLM约55–65%；与人工基准（约74.5%）相距明显。

**⚠️ 局限性**

局限在于模型仍缺乏本土几何直觉，往往依赖繁复推理；细粒度几何特征虽已编码于视觉表征，却未被LLM有效利用；且基准只评测单步直觉，未涵盖更复杂的几何推理与计算。

---

## 259. Proactive Local-Minima-Free Robot Navigation: Blending Motion Prediction with Safe Control

**arXiv ID:** 2601.10233 | [PDF](https://arxiv.org/pdf/2601.10233v1)

**作者:** Yifan Xue `[一作]` (University of Pennsylvania), Nadia Figueroa `[通讯]` (University of Pennsylvania)

**通讯引用:** 858 | [OpenAlex ID](https://openalex.org/A5074348852)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种结合多模态运动预测与改进的轨道控制屏障函数（MCBF）的移动机器人导航框架MMP-MCBF，能够主动避开动态非凸障碍并保持安全与高效；

**💡 创新点**

创新点在于：1）将运动预测结果通过高斯过程距离场在线学习转化为屏障函数；2）设计自适应参数选择算法，使MCBF在面对不断变形的预测障碍时保持无局部极小、无死锁；3）融合学习型能量模型(EBM)和常数速度模型(CVM)两种预测器，支持多模态预测；

**🔧 技术方法**

技术包括：多模态运动预测（EBM + CVM）、高斯过程距离场（GPDF）、控制屏障函数（CBF）与改进的MCBF-QP、在线自适应参数调节、轨道模糊化与几何逼近；

**📊 数据集**

使用人工合成的医院场景数据集（约30K样本，630轨迹）训练EBM；真实实验使用Fetch差速驱动机器人在室内人群场景；

**📈 对比分析**

与MPC、MPPI、CBF、MCBF等基线在四个仿真场景和真实人群实验中对比，MMP-MCBF在安全性（碰撞率0%）、成功率（100%）和平均完成时间上均优于基线；

**⚠️ 局限性**

局限包括：1）预测时域对安全性影响大，过长易过度保守；2）依赖完整障碍几何信息，实际传感器遮挡时需进一步研究；3）高斯过程学习对障碍点采样密度敏感；4）目前未考虑系统噪声与异常检测，需加入鲁棒性与安全冗余。

---

## 260. A Unified Framework for Kinematic Simulation of Rigid Foldable Structures

**arXiv ID:** 2601.10225 | [PDF](https://arxiv.org/pdf/2601.10225v1)

**作者:** Dongwook Kwak `[一作]` (Seoul National University), Jinkyu Yang `[通讯]` (Seoul National University)

**通讯引用:** 5437 | [OpenAlex ID](https://openalex.org/A5084771702)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

提出了一套统一框架，用于自动生成任意刚性折叠结构的 Pfaffian 约束矩阵并求解其可行折叠路径。

**💡 创新点**

创新点在于：①引入最小化数据 schema，统一描述单层、多层与厚板结构；②利用面-铰链图和最小环基自动提取所有闭环约束；③采用爪理论在速度层构造约束矩阵，实现同时考虑旋转与平移的闭环约束。

**🔧 技术方法**

技术包括：面-铰链图构造、最小环基计算、爪轴分配、速度层 Pfaffian 约束矩阵组装，以及基于能量偏好与约束投影的可行路径求解。

**📊 数据集**

使用公开的多种刚性折叠结构实例（如 Resch、Kirigami、堆叠 Miura、Tachi‑Miura Polyhedron、厚板 Miura 等），均可在 GitHub 示例数据集获取。

**📈 对比分析**

与传统解析、DH 或数值闭环求解器比较，框架能够在多层、厚板或含平移约束的复杂结构上自动构造完整约束，计算精度高、误差低，且在大规模图案（>500 条缝）下仍保持可接受的计算时间。

**⚠️ 局限性**

局限包括：仅适用于完全刚性面和理想铰链；对单自由度但多铰链的系统，约束子空间极窄导致搜索困难；缺乏全局可达性分析，无法保证全局折叠路径；对极大规模系统仍存在计算瓶颈。

---

## 261. Measuring Affinity between Attention-Head Weight Subspaces via the Projection Kernel

**arXiv ID:** 2601.10266 | [PDF](https://arxiv.org/pdf/2601.10266v1)

**作者:** Hiroaki Yamagiwa `[一作]` (Kyoto University), Hidetoshi Shimodaira `[通讯]` (RIKEN AIP)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出基于投影核（PK）的新度量方法，用于量化 Transformer 中注意力头权重子空间的重叠，并通过 PK 研究头间关系、识别 hub 头、可视化头功能；

**💡 创新点**

创新点在于：①引入基于主角角的投影核度量，克服 Composition Score 对规模的敏感性；②将 PK 与随机正交子空间的分布比较，评估信息量；③利用 PK 绘制“连线图”揭示 L4H7 作为 Identity Head 的 hub 角色；

**🔧 技术方法**

核心技术包括：投影核（Principal‑Angle based PK）、主角角、线性 CKA、Procrustes 距离、IOI 任务的头类注释、对 unembedding 矩阵的子空间投影等；

**📊 数据集**

主要使用数据集：GPT2‑small（d=768，d_head=64）以及 IOI 任务头类标注；

**📈 对比分析**

与 Composition Score（CS）、Simple‑CS、Linear CKA、Procrustes Similarity 等基线进行比较；在 head‑pair 识别上 PK 的 PR‑AUC 约为 0.047，明显优于 CS（0.013）；在同类型 head‑pair 分类中 PK 取得最高的平均 PR‑AUC 与 ROC‑AUC；

**⚠️ 局限性**

局限性包括：仅在 GPT2‑small 上验证；忽略层归一化、前馈层和相对位置编码；PK 只能捕获子空间层面信息，无法揭示 token‑级关系；对头功能的单标签划分可能不完整；分布信息量评估使用理论正交分布，可能不完全匹配实际权重分布。

---

## 262. Algebraic Properties of PAC Codes

**arXiv ID:** 2601.10262 | [PDF](https://arxiv.org/pdf/2601.10262v1)

**作者:** Vlad-Florin Dragoi `[一作]`, Mohammad Rowshan `[通讯]` (University of New South Wales)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文利用多项式与多项式码的代数表示，定义了通用多项式极化码（包括PAC码和逆PAC码）并推导其结构性质；

**💡 创新点**

创新点在于将PAC码纳入上多项式极化码框架，证明其最小距离与基础极化码相同，并给出最小码字下界、对偶结构以及对多项式子码的下限和上限；

**🔧 技术方法**

使用的技术包括多项式取值映射、低阶三角变换、ψ映射、逆变换矩阵、对偶分析及多项式子码的可分性判定；

**📊 数据集**

通过仿真在N=32、64、128、256等不同长度的极化码上，随机生成上多项式极化码进行评估；

**📈 对比分析**

与传统极化码和PAC码比较后发现，上多项式极化码在大多数码率下能够获得更多最小码字且保持最小距离，表现优于单纯极化码；

**⚠️ 局限性**

局限性包括在某些码率（仅包含最大度多项式或特定冻结模式）下无法改进，且在极低或极高码率时结构退化，导致对偶与最小距离提升有限。

---

## 263. Boundary-Aware NL2SQL: Integrating Reliability through Hybrid Reward and Data Synthesis

**arXiv ID:** 2601.10318 | [PDF](https://arxiv.org/pdf/2601.10318v1)

**作者:** Songsong Tian `[一作]` (Li Auto Inc.), Yong Wu `[通讯]` (Li Auto Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于大语言模型的边界感知可靠自然语言转SQL系统BAR‑SQL，能在企业数据库环境下准确生成SQL并在不可回答或含糊的查询中主动拒绝或请求澄清。

**💡 创新点**

创新点包括①Seed‑Mutation数据合成策略，生成覆盖多步分析、模糊、约束失效等边界情况的高质量企业级语料；②知识驱动的推理合成（KGRS）使Chain‑of‑Thought与schema、业务规则紧密绑定；③任务条件感知混合奖励（TCHR）在强化学习中同时优化执行精度与拒绝/澄清行为；④统一的GRPO强化学习框架，让单一模型兼顾生成与判断。

**🔧 技术方法**

技术手段包括大语言模型（Qwen3‑1.7B）预训练、Supervised Fine‑Tuning、Group‑Relative Policy Optimization（GRPO）、任务条件混合奖励、AST‑和执行结果匹配、Embedding相似度、Soft Length惩罚等。

**📊 数据集**

使用自研的Seed‑Mutation生成语料库（约20K条）与公开的企业内部数据库语义库，构建Ent‑SQL‑Bench（1,262条）用于评测SQL准确性与边界意识。

**📈 对比分析**

与Claude‑4.5‑Sonnet、Gemini3‑Pro、GPT‑4o、GPT‑5、DeepSeek‑V3.2等主流LLM比较，BAR‑SQL在Ent‑SQL‑Bench上平均精度达到91.48%，在标准SQL生成、复杂多步推理以及模糊/拒绝/澄清任务上分别取得90‑95%区间的高分，显著优于现有专有模型。

**⚠️ 局限性**

局限性包括：①仍依赖最终结果为监督，缺乏对中间推理步骤的直接校验；②目前仅在中等规模模型上验证，尚未探究更大规模模型的性能提升；③对跨域迁移能力和不同业务规则适配的进一步评估尚待开展。

---

## 264. XuanJia: A Comprehensive Virtualization-Based Code Obfuscator for Binary Protection

**arXiv ID:** 2601.10261 | [PDF](https://arxiv.org/pdf/2601.10261v1)

**作者:** Xianyu Zou `[一作]` (Nankai University), Pen-Chung Yew `[通讯]` (University of Minnesota)

**通讯引用:** 5811 | [OpenAlex ID](https://openalex.org/A5052005800)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出了一款名为XuanJia的虚拟化二进制混淆器，能够在保护可执行代码的同时，对异常处理（EH）元数据实施全流程加密与虚拟化，彻底消除静态泄漏；

**💡 创新点**

创新点在于提出了ABI‑兼容的EH Shadowing机制，它通过生成合法的阴影 unwind 代码并将真实的LSHandler与LSData迁移至 VM 中，既保持操作系统 ABI 的兼容性，又实现了 EH 逻辑的完全混淆；

**🔧 技术方法**

核心技术包括自定义指令映射 DSL 与 handler DSL、直接线程化（direct‑threaded）VM 调度器、对 EH 元数据的加密与重定位、以及运行时的阴影 unwind 与异常拦截器；

**📊 数据集**

实验数据集涵盖 5 种常见加密算法（AES、DES、MD5、RC4、SHA‑1）以及 1000 个随机 C++ 程序（约 678K 行）和 23 个 LLVM 异常处理基准；

**📈 对比分析**

与 VMProtect、Code Virtualizer、Tigress 等主流混淆器进行对比，XuanJia 在保持 ABI 合规的前提下，运行时开销在 120–337 倍之间，远低于商业工具的 2,485–36,431 倍；文件体积增幅约 2.1–11.2 倍，EHShadowing 进一步增加约 37 KB；

**⚠️ 局限性**

局限性包括：①运行时仍存在较高的慢速，尤其是异常抛掷时的堆栈遍历；②尚未集成高级硬化策略（如处理器多样化、加密算法多变形）；③评测仅针对 Windows x86 平台；④对动态逆向攻击（如符号执行、动态二进制插桩）的抵抗性尚未系统评估；

---

## 265. X-SAM: Boosting Sharpness-Aware Minimization with Dominant-Eigenvector Gradient Correction

**arXiv ID:** 2601.10251 | [PDF](https://arxiv.org/pdf/2601.10251v1)

**作者:** Hongru Duan `[一作]` (Taiyuan University of Technology), Lei Guan `[通讯]` (Taiyuan University of Technology)

**通讯引用:** 7512 | [OpenAlex ID](https://openalex.org/A5114374012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在SAM优化中分析梯度与Hessian主特征向量的夹角，提出X‑SAM通过抑制梯度主特征方向的分量来更有效控制最大Hessian特征值，从而提升模型泛化。

**💡 创新点**

提出将梯度分解为主特征方向与正交方向，并在SAM中加入该分解的梯度修正，实现显式控制最大Hessian特征值，理论收敛与实验证明优于现有SAM变体。

**🔧 技术方法**

使用Hessian特征向量估计（幂迭代）、梯度分解、正交投影、SGD/Adam等优化器，并给出理论证明和实验评估。

**📊 数据集**

CIFAR‑10、CIFAR‑100、Fashion‑MNIST等标准图像分类数据集，使用ResNet-18/50、WideResNet-28-10、AlexNet等网络。

**📈 对比分析**

与SAM、WSAM、GSAM、Eigen‑SAM等四个变体在相同设置下对比，X‑SAM在三大数据集上平均提升约1–1.5%准确率，单架构可提升超过3%，同时最大Hessian特征值下降更快，训练稳定性更好。

**⚠️ 局限性**

对角向量估计仍有计算开销，且对超参数α的选择有一定敏感性，未来需降低开销并推广至大规模模型。

---

## 266. Attend to what I say: Highlighting relevant content on slides

**arXiv ID:** 2601.10244 | [PDF](https://arxiv.org/pdf/2601.10244v1)

**作者:** Megha Mariam K M `[一作]` (International Institute of Information Technology), C. V. Jawahar `[通讯]` (International Institute of Information Technology)

**通讯引用:** 13667 | [OpenAlex ID](https://openalex.org/A5053112307)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套自动将演讲者口语与幻灯片内容对齐并高亮关键区域的系统，帮助观众同步关注关键信息。

**💡 创新点**

创新点在于将 OCR 与 ASR 输出结合，用语义嵌入与 LLM 进行语义匹配，并提出 OCR 辅助纠正 ASR 的方法；同时设计多种可视化高亮方式。

**🔧 技术方法**

使用技术包括 Amazon Textract OCR、WhisperX ASR、模糊匹配、MiniLM、S‑BERT、Sci‑BERT、Specter 语义嵌入、Flan‑T5 与 Qwen‑Instruct 2.5 LLM 进行后处理与对齐。

**📊 数据集**

数据集为从 NeurIPS 2022 与 ICML 2023 收集的 14 条演讲视频（共 150 幻灯片），包含音频、转录文本、OCR 结果与布局信息。

**📈 对比分析**

评估采用 Correctness、Missing 与 F1 三个指标；在不同阈值下对比模糊匹配、嵌入匹配和 LLM，结果显示模糊匹配准确率最高但漏检率高，LLM 在语义对齐上更稳健；F1 最高为 0.43（t5），OCR 辅助纠正 ASR 后整体对齐准确度提升。

**⚠️ 局限性**

局限性包括：对专业术语识别仍有限，语音与幻灯片同步误差导致对齐失误；高亮方式需兼顾可视性与信息完整性；模型在实时推理时延迟和资源消耗未评估。

---

## 267. STEAMROLLER: A Multi-Agent System for Inclusive Automatic Speech Recognition for People who Stutter

**arXiv ID:** 2601.10223 | [PDF](https://arxiv.org/pdf/2601.10223v1)

**作者:** Ziqi Xu `[一作]` (Shanghai Key Laboratory of Trustworthy Computing, East China Normal University), Yongxin Zhao `[通讯]` (Shanghai Key Laboratory of Trustworthy Computing, East China Normal University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们构建了一个实时多阶段多代理AI管线，将说话者的口吃语音转化为流利、无断裂的语音。

**💡 创新点**

核心创新在于利用协同多代理LLM框架对ASR转录进行语义一致性迭代修复，并结合语音克隆实现语音无断裂化，三大技术挑战同步解决。

**🔧 技术方法**

采用 Whisper 进行语音转文本，GPT‑4o 及自定义提示进行多代理文本修复，StyleTTS2 零样本语音克隆实现流利语音合成，整体采用多代理协作与链式思维推理。

**📊 数据集**

实验使用 FluencyBank（英文）和 AS‑70（中文）口吃语音数据集，以及 SEP‑28K 用于标注验证。

**📈 对比分析**

通过 WER/MER/WIL 及语义相似度评估，修复后 WER 下降约10% 并提升语义相似度 10%，用户满意度 4.3/5，实时延迟约 3.8 秒。

**⚠️ 局限性**

主要局限包括零样本语音克隆导致语音自然度不足，部分语音细节与原声失真，系统对多种口音和语言的泛化受限，实时延迟在极短语句上仍显高。

---

## 268. Introduction to optimization methods for training SciML models

**arXiv ID:** 2601.10222 | [PDF](https://arxiv.org/pdf/2601.10222v1)

**作者:** Alena Kopaničáková `[一作]` (Toulouse Institute of Technology), Elisa Riccietti `[通讯]` (École Normale Supérieure de Lyon)

**通讯引用:** 163 | [OpenAlex ID](https://openalex.org/A5024258728)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

综述了科学机器学习（SciML）中出现的优化问题，比较了传统机器学习与SciML在损失结构、NTK、收敛性等方面的差异，并系统阐述了针对SciML的优化技术和预处理策略。

**💡 创新点**

关键创新在于将NTK理论与SciML特有的全局耦合、条件数失衡等挑战相结合，提出多维度预处理（数据空间、参数空间、函数空间）和混合训练（如Adam→L-BFGS）等改进方案，解决传统SGD在PINN等任务中的收敛慢和数值不稳定问题。

**🔧 技术方法**

使用的技术包括梯度下降、Adam、L-BFGS、Newton/Krylov、子样本Hessian、BFGS、NTK分析、重采样、加权损失、学习率/批量调度、域分解、多层训练、Sobolev训练等。

**📊 数据集**

采用的典型数据集与实验任务包括逻辑回归、回归、PINN（Poisson、Navier–Stokes等 PDE）、生成式数据集、低秩POD等自定义仿真数据。

**📈 对比分析**

通过与标准SGD、Adam、Newton/Krylov等方法在PINN、回归等任务的对比实验，展示了预处理和混合策略显著提升收敛速度与最终精度，例如Adam→L-BFGS在一维Poisson问题上比单独Adam快数倍、在数值误差上更优。

**⚠️ 局限性**

局限性包括缺乏统一理论证明、多方法在非凸、随机梯度环境下的稳健性不足，且大规模高阶PDE的预处理设计仍缺乏通用准则；实验多基于小规模示例，未能在大规模真实物理问题上系统验证。

---

## 269. Evidence-Augmented Policy Optimization with Reward Co-Evolution for Long-Context Reasoning

**arXiv ID:** 2601.10306 | [PDF](https://arxiv.org/pdf/2601.10306v1)

**作者:** Xin Guan `[一作]` (Tongyi Lab Alibaba Group), Jiuxin Cao `[通讯]` (Tongyi Lab Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于证据增强的策略优化（EAPO）框架，旨在通过强化学习提升长文本推理中证据检索的质量与可靠性。

**💡 创新点**

核心创新在于将推理过程细化为证据检索与推理两步，并引入组相对证据奖励与自适应奖励-策略共演机制，以提供密集的过程监督，显著缓解稀疏奖励导致的“幸运猜测”问题。

**🔧 技术方法**

主要技术包括：Evidence‑Augmented Reasoning（EAR）范式、树结构证据采样、组相对策略优化（GRPO）、多粒度过程奖励（格式遵循、证据质量、答案准确性）以及奖励模型的自适应共演（结果一致性筛选与拒绝微调）。

**📊 数据集**

在八个长文本推理基准上进行评估，数据来源于SEAL、LongBench‑V1（HotpotQA、MuSiQue、2WikiMultihopQA）和LongBench‑V2（SDQ、MDQ、LSA）等。

**📈 对比分析**

与多种SOTA模型（GPT‑4o、Gemini‑2.0‑Flash、Qwen3‑Plus、Claude‑Sonnet‑4、GPT‑OSS‑120B、QwenLong‑32B）以及GRPO对照实验比较，EAPO在平均准确率上提升约5–7%，在多数基准上超越更大规模或闭源模型，证明其显著的性能优势。

**⚠️ 局限性**

限制主要在于奖励模型仍依赖外部评判器（LLM或检索器），共演过程可能对计算资源和训练稳定性提出更高要求；此外，系统对不同语言或更大规模语料的通用性尚未充分验证。

---

## 270. Multipath Routing for Multi-Hop UAV Networks

**arXiv ID:** 2601.10299 | [PDF](https://arxiv.org/pdf/2601.10299v1)

**作者:** Zhenyu Zhao `[一作]` (Beijing University of Posts and Telecommunications), Wenjuan Xing `[通讯]` (Chongqing University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

本文提出了一种面向多跳UAV网络的流量自适应多路径路由框架，能够让每架UAV动态划分并并行转发不同优先级的流量，从而降低拥塞并满足时延约束。

**💡 创新点**

创新点在于：①将路由问题建模为Dec‑POMDP并求解，②设计了基于Dirichlet分布的IPPO‑DM算法，实现连续且满足概率单位和的分流决策；③引入流量感知的重采样机制，使得路由策略随队列负载自适应调整。

**🔧 技术方法**

采用深度强化学习技术，具体实现为独立Proximal Policy Optimization（IPPO）框架配合Dirichlet分布参数化策略，网络结构包括MLP+GRU编码器与共享参数的Actor‑Critic对。

**📊 数据集**

未使用公开真实数据集，而是构建了三维仿真环境（1200×1200×200 m、35架UAV、随机任务生成、Gauss‑Markov移动模型）来评估算法性能。

**📈 对比分析**

与两种基准（均匀分流和贪心单路径）对比，IPPO‑DM在不同流量负载、网络规模和邻居数下均实现了更高的按时交付率、更低的丢包率，且收敛稳定。

**⚠️ 局限性**

局限性包括：①仅在仿真场景验证，缺乏真实部署实验；②训练与执行时的计算开销相对较大；③对极端高动态或极大规模网络的泛化性尚未充分评估。

---

## 271. Reasoning Hijacking: Subverting LLM Classification via Decision-Criteria Injection

**arXiv ID:** 2601.10294 | [PDF](https://arxiv.org/pdf/2601.10294v1)

**作者:** Yuansen Liu `[一作]` (National University of Singapore), Anthony Kum Hoe Tun `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实验了“推理劫持”攻击范式及其实现方式Criteria Attack，证明现代LLM在判断任务中易受注入假决策准则的影响而被操纵；

**💡 创新点**

首次定义推理劫持，展示注入伪准则可在不改变任务目标的前提下成功诱导模型错误判断，并在多种防御下保持高攻击成功率；

**🔧 技术方法**

利用辅助模型挖掘决策准则、聚类筛选可反驳准则、生成假推理链式注释，并在数据通道注入；

**📊 数据集**

使用Enron邮件（垃圾/非垃圾）、Wiki Toxic评论（有毒/无毒）、IMDb影评（正/负）等公开数据集；

**📈 对比分析**

与多种Goal Hijacking基线（Escape Separation、Ignore、Fake Completion等）及Prompt‑based、StruQ、SecAlign防御比较；在Qwen3‑4B、Gemma‑3‑27B等模型上，Criteria Attack在多数设置下攻击成功率>90%，优于大多数Goal Hijacking方法；

**⚠️ 局限性**

局限于判断型分类任务，需攻击者模型与足够的数据匹配，无法直接推广至开放式生成、多步推理或细微质量下降的场景；

---

## 272. SCRamble: Adaptive Decentralized Overlay Construction for Blockchain Networks

**arXiv ID:** 2601.10277 | [PDF](https://arxiv.org/pdf/2601.10277v1)

**作者:** Evangelos Kolyvas `[一作]` (Athens University of Economics and Business), Spyros Voulgaris `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 2373 | [OpenAlex ID](https://openalex.org/A5005110375)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并评估SCRamble协议，通过自适应邻居选择优化区块链网络的区块传播时延。

**💡 创新点**

创新点在于将基于历史区块转发速度的打分启发式、基于网络延迟的接近启发式以及随机选邻三种策略结合，并动态更新邻居集合；同时不需要手动调参即可适应不同网络环境。

**🔧 技术方法**

使用了Peer‑to‑Peer overlay构建、打分启发式、接近启发式、周期性重启、Peer‑Net模拟器和基于延迟的评估指标。

**📊 数据集**

使用WonderNetwork公开的真实延迟轨迹，映射到Bitcoin网络节点的地理分布，构成实验数据集。

**📈 对比分析**

通过在Peer‑Net模拟器上实验不同S‑C‑R配置（随机、单启发式、双启发式），比较区块传播进度；结果显示仅随机最差，单一启发式中等，双启发式（S3‑C3‑R2）显著加快传播，提升约30–40% 的传播速度。

**⚠️ 局限性**

局限性包括：实验仅在模拟环境下进行，未考虑节点离线、恶意攻击等动态情况；当仅使用接近启发式时易导致网络分割；算法对PSS的依赖以及潜在的连接维护开销未在实际网络中验证。

---

## 273. MoST: Mixing Speech and Text with Modality-Aware Mixture of Experts

**arXiv ID:** 2601.10272 | [PDF](https://arxiv.org/pdf/2601.10272v1)

**作者:** Yuxuan Lou `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**通讯引用:** 3721 | [OpenAlex ID](https://openalex.org/A5100658705)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了MoST——一种能够同时处理语音与文本的多模态大语言模型，支持语音识别、语音合成、音频语言建模及语音问答等任务。

**💡 创新点**

核心创新是Modality‑Aware Mixture of Experts (MAMoE)架构：通过模态感知路由将文本与音频 token 分别送入专门的专家组，并设置共享专家以实现跨模态信息融合；同时提出了从预训练MoE LLM到多模态模型的高效转化流水线。

**🔧 技术方法**

技术细节包括：Mixture of Experts（稀疏激活）、模态感知路由机制、HuBERT 语音编码器、HifiGAN 语音解码器、跨模态后训练 + 指令微调的两阶段训练策略。

**📊 数据集**

主要使用了开放源代码数据集：Common Voice、LibriHeavy、LibriSpeech、VoxPopuli、以及自构造的语音‑文本指令数据集（通过对话合成和文本转语音扩充）。

**📈 对比分析**

在 ASR、TTS、音频语言建模（sWUGGY、sBLIMP、sTopic‑StoryCloze 等）以及 Spoken Question Answering（Llama Q、Trivial QA、WebQ）等标准基准上，与多模态 LLM（如 AudioLM、SpiritLM、Moshi、Qwen2‑Audio 等）进行对比，MoST 在多数指标上均达到或逼近 SOTA，WER 低于同参数规模的竞争者，准确率高于对比模型。

**⚠️ 局限性**

局限性包括：只聚焦语音与文本两种模态；专家划分采用简单的 50/50 固定策略，缺乏动态或聚类驱动的专家分配；在极大规模数据或更复杂跨模态任务上可能仍有提升空间。

---

## 274. Error-Correcting Codes for the Sum Channel

**arXiv ID:** 2601.10256 | [PDF](https://arxiv.org/pdf/2601.10256v1)

**作者:** Lyan Abboud `[一作]` (Technion Israel Institute of Technology), Eitan Yaakobi `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出并分析了在Sum通道模型下针对删除、插入和替换错误的编码方案，给出了2删除纠错码和1编辑纠错码，并证明其冗余率接近理论下界。

**💡 创新点**

创新点在于：1) 为两行输入构造的2删除纠错码冗余仅为 2⌈log₂log₂n⌉+O(ℓ²)，并证明其近似最优；2) 为任意行数输入构造1替换纠错码冗余仅 ⌈log₂(ℓ+1)⌉ 位，且几乎达到最优；3) 通过全同余约束、SVT码和张量乘积码等技术实现低冗余。

**🔧 技术方法**

使用了SVT（改进的VT码）、全同余约束、张量乘积码框架、图论团覆盖和球面覆盖等编码理论工具。

**📊 数据集**

为理论分析，未使用实际数据集，全部基于数学推导与上界/下界证明。

**📈 对比分析**

通过冗余率与下界（log₂log₂n）以及团覆盖给出的上界（≈12/ log₂n）进行比较，展示所构造码在冗余率上与理论下界相差常数因子；在1编辑纠错方面，冗余率仅比最优多 1 位。

**⚠️ 局限性**

局限性包括：对行数 ℓ 的扩展仍带有 O(ℓ²) 的冗余项；仅针对 Sum 通道，未覆盖更一般的错误模型；实现与解码复杂度未给出细节，实际部署时可能需要进一步优化。

---

## 275. Developer Interaction Patterns with Proactive AI: A Five-Day Field Study

**arXiv ID:** 2601.10253 | [PDF](https://arxiv.org/pdf/2601.10253v1)

**作者:** Nadine Kuo `[一作]` (JetBrains), Maliheh Izadi `[通讯]` (Delft University of Technology)

**通讯引用:** 4516 | [OpenAlex ID](https://openalex.org/A5024645888)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在JetBrains Fleet IDE中实现了一种主动式AI代码质量助手ProAIDE，并在15名专业开发者的为期5天的真实工作环境中进行场景研究，记录并分析了229次AI干预与5,732次交互点；

**💡 创新点**

创新点在于将基于大语言模型的主动式建议与生产级IDE深度集成，系统性探讨了干预时机对开发者接受度的影响，并首次在真实工作流中验证了主动式建议的效率与可接受性；

**🔧 技术方法**

使用的技术包括LLM驱动的上下文感知主动触发器（模糊提示识别、被拒绝编辑跟进、提交后回顾），结合IDE原生事件日志与用户交互日志，配合定制化UI弹窗与聊天面板；

**📊 数据集**

使用的数据集为开发者在日常工作中产生的匿名交互日志（共5,732条交互）和自我报告问卷，未公开具体代码或模型输出；

**📈 对比分析**

通过与被动（reactive）交互比较，采用Wilcoxon符号秩检验发现主动式建议的解释时间平均45.4 s，显著快于被动建议的101.4 s（p=0.0016），且SUS评分为72.8/100，表明用户体验良好；

**⚠️ 局限性**

局限性包括样本量有限且仅来自熟悉JetBrains工具的开发者，缺乏代码质量的客观评估，且无法将交互日志与问卷结果精确匹配，未来需扩大样本、加入长期跟踪及代码质量度量。

---

## 276. coTherapist: A Behavior-Aligned Small Language Model to Support Mental Healthcare Experts

**arXiv ID:** 2601.10246 | [PDF](https://arxiv.org/pdf/2601.10246v1)

**作者:** Prottay Kumar Adhikary `[一作]` (Indian Institute of Technology Delhi), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 4849 | [OpenAlex ID](https://openalex.org/A5046521217)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一框架coTherapist，利用1B参数小型语言模型通过领域自适应预训练、LoRA风格微调、检索增强生成和代理推理，实现在数字心理健康平台中为临床专家提供支持的轻量级协同治疗助手。

**💡 创新点**

创新点包括：1) 将小模型与多阶段管线相结合，精确模拟专业治疗师行为；2) 引入T‑BARS行为评估尺度和心理测评评估模型人格；3) 在小模型上实现检索增强与代理推理，提升情境理解与安全性；4) 通过4‑bit量化在边缘设备上实现本地隐私安全部署。

**🔧 技术方法**

技术：以LLaMA 3.2‑1B‑Instruct为基底，采用领域自适应预训练（DAP）、LoRA风格微调、Self‑Instruction调优、RAG检索、内部推理与自我评估循环，并使用量化推理引擎实现边缘推理。

**📊 数据集**

数据集：构建约8亿token的心理治疗知识语料库PsyKC，包含311本治疗书籍/手册、250份讲义与550条视频字幕、121份诊断与实践指南，总计约524M+227M+49M tokens；并用100问答基准用于评测。

**📈 对比分析**

比较方法：在100道专业问答上对比基础LLaMA、域预训练、LoRA、RAG等版本，计算BLEU、METEOR、ROUGE‑L、BERTScore、BLEURT、InfoLM；使用T‑BARS评分评估四大支柱，人工评估5项指标；结果显示coTherapist在自动指标上领先20%+，在T‑BARS和人工评估中平均分达3.6/5，明显优于基线。

**⚠️ 局限性**

局限：模型仅支持英文且覆盖主流证据基础疗法；对危机情境仍需人工监督；虽然量化后可在边缘设备上运行，但在高复杂对话与多模态交互方面尚不成熟；目前仅在小规模专家实验验证，需更大范围、多中心评估。

---

## 277. Loop as a Bridge: Can Looped Transformers Truly Link Representation Space and Natural Language Outputs?

**arXiv ID:** 2601.10242 | [PDF](https://arxiv.org/pdf/2601.10242v1)

**作者:** Guanxu Chen `[一作]` (Shanghai Jiao Tong University), Jing Shao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 9670 | [OpenAlex ID](https://openalex.org/A5023198186)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对循环 Transformer（Looped Transformers, LT）在自我验证与内部表示监测之间的对齐能力进行实验评估，并探究其对内部表示的持续内省效果。

**💡 创新点**

提出并验证了循环深度既能缩小语言输出与内部表示之间的性能差距，又可能导致内部表示信息的退化；同时发现 LT 在中间循环中缺乏持续内省，只有在最终循环才能有效识别注入的概念。

**🔧 技术方法**

使用循环 Transformer 架构（以 Ouro 变体为例），结合线性探针评估内部表示的可分离度，利用注入概念实验检测模型对内部表示的感知；对比传统 Transformer LLM 作为基线。

**📊 数据集**

使用 BeaverTails（安全判断）和 DeepMath（数学验证）两类数据集，分别构建 8000/8000 训练样本与 2000/2000 测试样本；在 Qwen3-235B-A22B-Instruct-2507 上采样多轮生成以得到 1–8 次循环的表示。

**📈 对比分析**

通过比较自我验证（SV）与线性探针（RR）的准确率与 F1，发现循环次数增加时 SV 上升但 RR 降低，导致差距收窄；在注入实验中，检测与识别准确率仅在最终循环显著提升。相较于基线 Transformer，LT 在自我验证上表现略好，但内部表示监测效果不一定提升。

**⚠️ 局限性**

实验仅覆盖单一 LT 实例（Ouro），未针对不同循环策略或训练目标进行系统探索；内部表示退化与缺乏持续内省可能源自模型架构或训练方法，需进一步改进。

---

## 278. Who Owns the Text? Design Patterns for Preserving Authorship in AI-Assisted Writing

**arXiv ID:** 2601.10236 | [PDF](https://arxiv.org/pdf/2601.10236v1)

**作者:** Bohan Zhang `[一作]` (Tsinghua University), Paramveer S. Dhillon `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过实验评估AI写作助手在三种专业写作场景下对作者归属感、工作量、质量和接受率的影响，探讨 persona‑based coaching 与 style‑personalization 两种设计杠杆的作用；

**💡 创新点**

创新点在于揭示“所有权–效率”张力，证明轻量化风格个性化可部分恢复归属感，并从实验结果中提炼出五条可落地的作者保留设计模式；

**🔧 技术方法**

采用 GPT‑4o 作为语言模型，构建了基于浏览器的共写编辑器，提供按需单句建议、persona “Emma”框架以及基于少量写作样本的风格剖面；

**📊 数据集**

使用来自 Prolific 176 名受试者的写作任务样本，受试者在实验前提交一段个人写作样本用于生成风格剖面；

**📈 对比分析**

采用混合实验设计（within‑subject + between‑subject），通过 OLS 回归与 FDR 校正比较各模式对心理所有权、NASA‑TLX 认知负荷、质量评分和 AI 内容占比的影响；实验显示在协助场景下所有权下降约 1 分，认知负荷下降约 0.9 分，质量基本不变，个性化提升所有权约 0.43 分并增加约 5% 的 AI 内容采用；

**⚠️ 局限性**

局限性包括：实验为单次短期任务，未考察长期使用与迭代；仅测试 GPT‑4o 与单一 persona，其他模型或角色可能产生不同效果；个性化仅基于一段写作样本，缺乏多样化风格配置；受试者主要为英语国家受教育人群，缺乏跨语言和行业的普适性。

---

## 279. Tables or Sankey Diagrams? Investigating User Interaction with Different Representations of Simulation Parameters

**arXiv ID:** 2601.10232 | [PDF](https://arxiv.org/pdf/2601.10232v1)

**作者:** Choro Ulan uulu `[一作]` (Siemens AG), Helena Holmström Olsson `[通讯]` (Malmö University)

**通讯引用:** 4632 | [OpenAlex ID](https://openalex.org/A5049811300)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过创建交互式 Sankey 图表原型，探讨其在机械仿真软件中替代传统电子表格显示参数及其依赖关系的可行性。

**💡 创新点**

创新点在于将流式可视化（Sankey）直接应用于工程参数配置，显式展示全局与局部参数之间的流动关系，从而显著降低认知负担并提升理解效率。

**🔧 技术方法**

技术实现采用 Plotly 的 Sankey 图与 Dash 框架构建交互式前端；评估方法为 PURE 纯专家可用性评分法，并结合三位专家的认知负荷打分与共识讨论。

**📊 数据集**

使用的数据集为 Siemens Simcenter 案例中的典型参数集合，包含若干全局与局部参数，规模为中等数量级（数十个参数）。

**📈 对比分析**

通过 PURE 方法将两种界面按任务拆分为步骤并打分，最终发现 Sankey 版在所有三类任务中 PURE 得分降低约 51%，交互步骤减少约 56%，表明其在认知负荷和操作效率上均优于传统表格。

**⚠️ 局限性**

局限性包括：仅在中等规模参数集上验证，未测试更大参数空间；评估基于专家评审而非用户实验；实验对象为域内专家，缺乏普通用户视角；并且 Sankey 原型由其开发者参与评估，存在潜在偏见。

---

## 280. Optimizing Multimodal LLMs for Egocentric Video Understanding: A Solution for the HD-EPIC VQA Challenge

**arXiv ID:** 2601.10228 | [PDF](https://arxiv.org/pdf/2601.10228v1)

**作者:** Sicheng Yang `[一作]`, Zhensong Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对长时段第一人称厨房视频的多模态大语言模型在复杂视频问答上的表现，提出并实现了一套完整的端到端系统。

**💡 创新点**

创新点包括：①多层级预处理细化查询与选项；②基于Qwen2.5-VL的领域特定微调；③提出Temporal Chain‑of‑Thought (T‑CoT) 两阶段时间推理策略；④强健的答案清洗与五轮投票集成。

**🔧 技术方法**

主要技术：文本正则化、结构化提示、视频裁剪/分块、基于BERT的提示指令、答案正则化、投票集成。

**📊 数据集**

使用的数据集：HD‑EPIC VQA（长时段 egocentric 厨房视频），并在微调阶段加入 EPIC‑KITCHENS、CMU‑MMAC、EGTEA Gaze+、YOUCOOK2、VISOR、Ego4D 等多源厨房相关数据。

**📈 对比分析**

与公开基线对比：在 HD‑EPIC VQA 上，基线 Qwen2.5‑VL‑7B 仅 38.1%；加入预处理、微调、T‑CoT、后处理后准确率提升至 41.6%，相较主流闭源模型 Gemini‑Pro（≈88%）仍有差距，但在开放源领域显著优于同类公开模型。

**⚠️ 局限性**

局限性：①依赖多阶段处理导致推理延迟高；②对超长视频的时间窗口仍受模型输入长度限制；③大模型规模在当前管线下效果不升反降，需进一步针对大模型的微调；④整体仍远离人类水平，需提升长时记忆与深层推理能力。

---

## 281. We Need a More Robust Classifier: Dual Causal Learning Empowers Domain-Incremental Time Series Classification

**arXiv ID:** 2601.10312 | [PDF](https://arxiv.org/pdf/2601.10312v1)

**作者:** Zhipeng Liu `[一作]` (Northeastern University), Binwu Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 392 | [OpenAlex ID](https://openalex.org/A5054585097)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在域增量时间序列分类中提出了轻量级 DualCD 框架，以提升模型对新域的适应性与对旧域的记忆。

**💡 创新点**

创新点在于引入双因果解耦（正交掩码分离因果与噪声特征）以及双因果干预（在类内和类间生成变异样本），从而学习稳健的因果特征。

**🔧 技术方法**

使用的技术包括正交掩码解耦、因果干预损失、轻量化线性模块，以及可插拔的时间序列编码器（如 DLinear、PatchTST 等）。

**📊 数据集**

实验使用了四个公开数据集：HAR、HHAR、ISRUC‑S3 与 Sleep‑EDF。

**📈 对比分析**

与 12 种基线（7 个时序分类模型 + 5 个域增量方法）比较，DualCD 在平均准确率上提升约 3–10%，相对遗忘率降低 10–20%，PRF 指标亦显著下降，表现为最优或次优水平。

**⚠️ 局限性**

限制在于目前仅适用于单数据集内的域增量，无法直接迁移到跨数据集场景，需要解决变量不一致与对齐问题。

---

## 282. SPIKE: Sparse Koopman Regularization for Physics-Informed Neural Networks

**arXiv ID:** 2601.10282 | [PDF](https://arxiv.org/pdf/2601.10282v1)

**作者:** Jose Marie Antonio Minoza `[一作]` `[通讯]` (Center for AI Research PH), Jose Marie Antonio Minoza (Center for AI Research PH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过在PINN框架中加入连续时间 Koopman 正则化，提出了 PIKE/SPIKE 方法，用以提升 PDE 和 ODE 解决方案的 OOD 泛化能力。

**💡 创新点**

创新点在于：①将 Koopman 正则化作为 PINN 的辅助约束，反向提升泛化；②采用连续时间生成器和矩阵指数积分实现对刚性 PDE 的无条件稳定；③通过 L1 稀疏化生成矩阵，实现可解释、稀疏的动力学表示；④融合多项式库与 MLP 嵌入，兼顾结构化解释与灵活性。

**🔧 技术方法**

技术手段包括：自动微分训练的 PINN、Koopman 正则化损失（Euler/RK4/EXPM）、L1 稀疏正则、矩阵指数数值积分、观测函数分解为多项式与 MLP 两部分。

**📊 数据集**

实验使用 14 个系统数据集，涵盖 9 条一维 PDE（如热方程、Advection、Burgers、Allen‑Cahn、KdV、Reaction‑Diffusion、Cahn‑Hilliard、Kuramoto‑Sivashinsky、Schrödinger）、2 条二维 PDE（2D Wave、Navier‑Stokes）、以及 2 条 ODE（Lorenz、SEIR），无额外外部数据集。

**📈 对比分析**

与传统 PINN 基线（相同网络结构、训练步骤、优化器）对比，PIKE/SPIKE 在时间外推、空间外推、刚性 PDE、混沌系统等多种指标上均显著提升：时间外推误差下降 2–184 倍，空间外推误差降至 1/32，刚性 PDE 训练误差提高 10^6 倍，混沌系统有效预测时间提升 184 倍；稀疏化后矩阵零率可达 99.9%。

**⚠️ 局限性**

局限性包括：多项式库对非多项式非线性（如 sin(u)、e^u）表达不足，需扩展库；对存在尖锐波动或冲击的超快 PDE 仍可能出现数值不稳定；仅对光滑解有效，无法直接处理不连续解；空间外推在有限域问题中仅体现平滑性而非物理意义。

---

## 283. The impact of tactile sensor configurations on grasp learning efficiency -- a comparative evaluation in simulation

**arXiv ID:** 2601.10268 | [PDF](https://arxiv.org/pdf/2601.10268v1)

**作者:** Eszter Birtalan `[一作]` (Pázmány Péter Catholic University), Miklós Koller `[通讯]` (Pázmány Péter Catholic University)

**通讯引用:** 68 | [OpenAlex ID](https://openalex.org/A5067935891)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在两种不同的机器人手模型（Modular Prosthetic Limb 和 Shadow Dexterous Hand）和两种强化学习算法（PPO 与 DDPG+HER）上，系统评估了六种不同密度与布局的触觉传感器配置对抓取提升任务学习效果的影响。

**💡 创新点**

证明了传感器布局与密度在强化学习性能中起关键作用，发现配置2在两套实验中均表现最佳，且高密度并不一定带来更好性能；低密度且合理布局即可满足需求。

**🔧 技术方法**

使用了强化学习（PPO、DDPG+HER）、触觉传感器仿真、分层自举（bootstrapping）和 RLiable 统计库进行性能评估。

**📊 数据集**

实验使用的“数据集”为仿真环境中的立方体抓取任务，未使用外部真实数据集。

**📈 对比分析**

通过对比带触觉（主组）与无触觉（控制组）的成功率、收敛成功率及其置信区间进行统计；在 PyBullet 环境中配置1、2、6优于控制组，配置2在两环境中均取得最高成功率（PyBullet 66%/MuJoCo 88%），但 MuJoCo 环境收敛时置信区间更宽。

**⚠️ 局限性**

局限性包括：样本量有限、仅测试两种手模型与两种 RL 算法、仅针对单一抓取任务、未针对每种配置单独调优网络结构、控制组采用“全零”传感输出可能不完全代表真实无传感情况。

---

## 284. Are Language Models Models?

**arXiv ID:** 2601.10421 | [PDF](https://arxiv.org/pdf/2601.10421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 285. Beyond Inpainting: Unleash 3D Understanding for Precise Camera-Controlled Video Generation

**arXiv ID:** 2601.10214 | [PDF](https://arxiv.org/pdf/2601.10214v1)

**作者:** Dong-Yu Chen `[一作]` (Tsinghua University), Shi-Min Hu `[通讯]` (Tsinghua University)

**通讯引用:** 22205 | [OpenAlex ID](https://openalex.org/A5037233582)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 DepthDirector 框架，实现从单目视频在任意相机轨迹下重渲动态场景。

**💡 创新点**

创新点在于利用已构建的 3D 网格渲染的深度序列作为几何条件，结合视角‑内容双流条件机制和轻量 LoRA 适配器，突破传统 warp‑inpainting 的“填补陷阱”，实现精确相机控制和内容一致性。

**🔧 技术方法**

使用视频扩散模型 Wan2.2 作为基线，采用 VAE 编码深度与图像、DiT transformer、双流条件注入、LoRA 适配器、重投影与点云网格化等技术。

**📊 数据集**

构造了 MultiCam‑WarpData 数据集，包含 1K 场景、8K 真实感视频、每帧深度与多视角同步，用于训练与评估。

**📈 对比分析**

与基线（Warp‑based 如 EX‑4D、TrajCrafter、Gen3C；Implicit‑based 如 ReCamMaster、CamCloneMaster）在相机精度、身份保持、视图同步及 VBench 感知指标上进行对比，DepthDirector 在大多数指标上取得领先，尤其在身份保持和背景一致性上明显优于其他方法。

**⚠️ 局限性**

局限在于仍需依赖相机轨迹和深度估计的质量，深度误差可能导致局部几何误差；与纯 warp‑based 方法相比在极端视角变换下相机精度略低。

---

## 286. GeoSteer: Faithful Chain-of-Thought Steering via Latent Manifold Gradients

**arXiv ID:** 2601.10229 | [PDF](https://arxiv.org/pdf/2601.10229v1)

**作者:** Kentaro Kazama `[一作]` (Mitsubishi Electric Corporation), Tatsuhiko Saito `[通讯]` (Mitsubishi Electric Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构造高质量与低质量的Chain-of-Thought（CoT）数据，训练VAE学习低维隐空间，并在推理时对LLM隐藏状态进行隐空间梯度拉回，从而引导模型生成更连贯、逻辑一致的中间推理过程。

**💡 创新点**

创新点在于把隐藏状态视为曲面上的点，利用VAE得到平滑的隐空间，并在该空间中计算质量梯度，再将梯度拉回原始隐藏状态空间实现几何意识的激活调控；相较于传统线性欧氏空间调节方法更能保持推理结构的完整性。

**🔧 技术方法**

使用Variational Autoencoder（VAE）构建隐空间、质量回归器（Rψ）估计中间推理质量、隐空间梯度拉回到隐藏状态、以及Qwen3系列LLM进行推理；同时采用gpt-oss-20b生成教师CoT、GSM8k作为任务数据。

**📊 数据集**

数据集包括：gpt-oss-20b生成的双类别（高质量与低质量）CoT，用以训练VAE和质量估计器；GSM8k算术推理基准，用于评估模型在真实任务上的表现。

**📈 对比分析**

与未使用GeoSteer的baseline在同一Prompt下对比，基准Qwen3模型在EM上提升0–2.6个百分点，pairwise win率提升5.3个百分点（p<0.05）；在不同steer强度（β）下的细粒度评估显示，中等至高模型规模对EM提升更为显著，而较小模型对β更敏感。

**⚠️ 局限性**

局限性包括：依赖教师模型gpt-oss-20b生成的CoT，隐空间与高质量推理的映射可能不完整；VAE隐空间的几何真实性尚未理论证明；方法对不同LLM架构的通用性尚待进一步验证。

---

## 287. INDIC DIALECT: A Multi Task Benchmark to Evaluate and Translate in Indian Language Dialects

**arXiv ID:** 2601.10388 | [PDF](https://arxiv.org/pdf/2601.10388v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 288. Untangling Input Language from Reasoning Language: A Diagnostic Framework for Cross-Lingual Moral Alignment in LLMs

**arXiv ID:** 2601.10257 | [PDF](https://arxiv.org/pdf/2601.10257v1)

**作者:** Nan Li `[一作]` (Ghent University), Tijl De Bie `[通讯]` (Ghent University)

**通讯引用:** 7041 | [OpenAlex ID](https://openalex.org/A5076045275)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究跨语言LLM在道德困境评估中输入语言与推理语言的差异；通过实验拆解两种语言因素的影响；

**💡 创新点**

提出N×N拆解框架与道德基础理论解析，能分离输入与推理语言效应，揭示隐藏的跨文化脆弱性；

**🔧 技术方法**

使用因子分解、Flip率、灵敏度比、模式一致性指标、Moral Foundations 量化回归等技术；

**📊 数据集**

采用两套跨文化数据集：AITA（英）和CMoral（中），并构建人类基准；

**📈 对比分析**

相较于传统匹配评估，推理语言对结果的影响约为输入语言的两倍；发现44%模型存在隐藏上下文依赖；提出四象限稳定性分类；

**⚠️ 局限性**

局限包括：仅验证英↔中语言对，翻译/适配偏差、内部推理未直接观察、域差异、阈值主观、样本与语言范围有限。

---

## 289. TF3-RO-50M: Training Compact Romanian Language Models from Scratch on Synthetic Moral Microfiction

**arXiv ID:** 2601.10410 | [PDF](https://arxiv.org/pdf/2601.10410v1)

**作者:** Mihai Dan Nadas `[一作]`, Andrei Piscoran `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于全合成罗马尼亚童话文本，构建从分词器设计、预训练、压缩、蒸馏到大规模合成数据生成的一条完整端到端管线；

**💡 创新点**

首次提出统一的罗马尼亚语言模型端到端流程，并通过结构化剪枝+logit蒸馏实现从 51.65M 参数压缩到 26.45M 参数的轻量化模型；

**🔧 技术方法**

采用 SentencePiece 的 Unigram 分词、LLaMA‑style Transformer、长序列打包、8‑bit/6‑bit 量化、结构化剪枝、logit 蒸馏、LLM‑as‑judge 等技术；

**📊 数据集**

使用 TF2 生成的 300 万条罗马尼亚翻译童话（约 10 亿 token）以及后续基于模型生成的 300 万条原生罗马尼亚童话；

**📈 对比分析**

在统一的 2048 token 打包评估下，Teacher 模型达到 PPL≈2.4；与 8‑bit 量化、6‑bit 量化、Mamba 以及强大指令调优模型对比，Teacher 与 26.45M 学生模型仅差 8‑10%，但速度和内存显著提升；

**⚠️ 局限性**

受限于仅使用合成数据，可能继承生成器偏差；评估多为自动化，缺少大规模人类评价；域限定在道德童话，未验证在更广泛罗马尼亚文本上的泛化。

---

## 290. CS-GBA: A Critical Sample-based Gradient-guided Backdoor Attack for Offline Reinforcement Learning

**arXiv ID:** 2601.10407 | [PDF](https://arxiv.org/pdf/2601.10407v1)

**作者:** Yuanjie Zhao `[一作]` (Shanghai Jiao Tong University), Jie Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 26452 | [OpenAlex ID](https://openalex.org/A5100428255)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了 CS‑GBA 框架，利用 TD‑误差挑选关键样本，在数据中注入协方差破坏触发器和梯度引导恶意动作，从而以低预算实现隐蔽且破坏性强的离线 RL 后门攻击。

**💡 创新点**

创新点包括（1）基于 TD‑误差的关键样本优先投毒策略；（2）利用特征相关性与 95% 分位数的协方差破坏触发器，避免 OOD 检测；（3）在数据流形内进行梯度投影搜索生成恶意动作，成功突破保守防御。

**🔧 技术方法**

采用 TD‑误差加权采样、特征相关性分析与分位数触发、梯度投影搜索（Manifold‑Constrained Gradient Descent）、离线 RL 算法（CQL、IQL、BCQ）以及 D4RL 基准环境进行实验。

**📊 数据集**

使用 D4RL 基准中的 walker2d、halfcheetah、hopper 三个 medium 质量数据集。

**📈 对比分析**

与 BAFFLE 基线在相同任务下比较，CS‑GBA 在 5% 投毒预算下攻击成功率和攻击奖励均显著优于 BAFFLE（10% 预算），且在 CQL、IQL、BCQ 三种算法上均能大幅压低攻击奖励，同时保持清洁环境性能。

**⚠️ 局限性**

对投毒预算和关键样本位置高度敏感；在极不稳定环境或更强的防御机制下仍需精确定位；未针对多步触发、在线恢复等更复杂场景进行评估。

---

## 291. Does Cognitive Load Affect Human Accuracy in Detecting Voice-Based Deepfakes?

**arXiv ID:** 2601.10383 | [PDF](https://arxiv.org/pdf/2601.10383v1)

**作者:** Marcel Gohsen `[一作]` (Bauhaus-Universität Weimar), Benno Stein `[通讯]` (Bauhaus-Universität Weimar)

**通讯引用:** 11694 | [OpenAlex ID](https://openalex.org/A5027915931)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了在不同认知负荷条件下，人类对语音深度伪造的检测准确性。

**💡 创新点**

首次系统评估轻度认知负荷（1-back任务）与辅助视觉刺激对语音深伪造检测的影响。

**🔧 技术方法**

采用1-back工作记忆任务、B-roll视频辅助、TorToise语音克隆模型生成对比音频。

**📊 数据集**

自制语音克隆数据集：四名新闻主播的真实语音与克隆语音（共48段音频，16段视频），采集自YouTube。

**📈 对比分析**

通过单任务、双任务和视频条件的对比实验，使用准确率、t检验、逻辑回归等统计方法；单任务≈68%，双任务≈66%，视频≈75%，表明视频辅助显著提升准确率。

**⚠️ 局限性**

局限包括仅采用轻度认知负荷（1-back），未测试更强负荷；样本量小且受众单一；只用一种克隆算法；未考虑说话人熟悉度及真实社交媒体情境。

---

## 292. Development of Ontological Knowledge Bases by Leveraging Large Language Models

**arXiv ID:** 2601.10436 | [PDF](https://arxiv.org/pdf/2601.10436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 293. Self-supervised restoration of singing voice degraded by pitch shifting using shallow diffusion

**arXiv ID:** 2601.10345 | [PDF](https://arxiv.org/pdf/2601.10345v1)

**作者:** Yunyi Liu `[一作]` (University of Sydney), Taketo Akama `[通讯]` (Sony Computer Science Laboratories)

**通讯引用:** 33 | [OpenAlex ID](https://openalex.org/A5087426444)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于WORLD声码器的轻量级扩散模型，对受音高移位影响的歌唱音频进行降噪恢复，最终生成高质量、无色彩化的音频。

**💡 创新点**

创新点在于将音高移位视为降噪恢复任务，利用自监督生成的损坏-原始对来训练扩散模型，并在恢复过程中仅对声谱图做轻量级去噪，从而实现对未见歌手的无源音高移位与高保真度重构。

**🔧 技术方法**

技术包括WORLD声码器的前向后向音高移位、基于mel空间的浅层扩散（shallow diffusion）模型、CREPE提取的F0、音量包络与ContentVec语义特征作为条件、以及DPM‑Solver加速的采样；最终用NSF‑HiFiGAN声码器还原波形。

**📊 数据集**

使用了CSD、NUS‑48E、VocalSet、Choral Singing Dataset、ESMUC五大歌唱语料，涵盖多语言、多风格和61名歌手，测试集为未见的Singing Voice Audio Dataset（主要为中国戏曲和西方歌剧）。

**📈 对比分析**

与TD‑PSOLA、CLPCNet、WORLD、SiFiGAN和Diff‑Pitcher等传统与现代方法对比，本文模型在FAD、KID、MMD、SC、LSD、SI‑SDR、F0误差和V/UV误差等指标上均显著优于对手，尤其在FAD（7.886）和F0误差（≈2.9c）上达成最低误差。

**⚠️ 局限性**

主要限制在于仍依赖WORLD声码器的声源-滤波结构，对极端音高移位或非常低频声源时的声谱重建可能受限；此外模型对未见声纹的泛化虽然好，但对非人声或伴奏环境的适用性尚未验证。

---

## 294. OctoBench: Benchmarking Scaffold-Aware Instruction Following in Repository-Grounded Agentic Coding

**arXiv ID:** 2601.10343 | [PDF](https://arxiv.org/pdf/2601.10343v1)

**作者:** Deming Ding `[一作]` (Fudan University), Tao Gui `[通讯]` (Fudan University)

**通讯引用:** 4787 | [OpenAlex ID](https://openalex.org/A5058353652)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向多源异构指令的 agentic 编码评测基准 OctoBench，评估 LLM 在持久、优先级冲突和多轮交互中的指令遵循能力。

**💡 创新点**

①将多种指令源（系统提示、用户查询、仓库策略、工具模式等）与真实可执行环境结合，构造可验证的二元检查表；②设计自动化轨迹捕获与 LLM‑judge 评分框架，实现过程层面的细粒度评估；③引入冲突子集 OctoBench‑Conflict，研究模型在冲突指令下的隐式优先级决策。

**🔧 技术方法**

多源指令标注与 LLM‑生成检查表、Docker 化可执行环境、代理日志记录与轨迹归一化、LLM‑judge（GPT‑5.1、Claude‑Sonnet‑4.5、Gemini‑3‑Pro）评估、基准脚本与工具链。

**📊 数据集**

OctoBench（34 个环境，217 个任务，7,098 条二元检查项），以及专门构造的 32 条冲突实例 OctoBench‑Conflict。

**📈 对比分析**

对八款主流 LLM（Claude‑Opus‑4.5、MiniMax‑M2、Gemini‑3‑Pro 等）在三种 scaffold（Claude Code、Kilo、Droid）下进行 ISR/CSR 评估。实验显示，虽然 CSR 在 80%–86% 之间，但 ISR 仅 9.66%–28.11%，表明高检查通过率无法保证整体任务成功；不同 scaffold、不同指令类别及冲突类型显著影响性能，外部反馈可提升 ISR 约 7%–17%。

**⚠️ 局限性**

局限：评估仅覆盖二元可验证检查，忽略主观质量（如解释清晰度）；检查表与评分依赖 LLM，尽管做了人类审核和集成判断，但仍可能出现错误；基准未覆盖所有 agentic 工具链、企业政策或长周期工作流，部分指令类别与冲突模式可能不足；未公开完整轨迹，限制第三方审计与定性分析。

---

## 295. CHORAL: Traversal-Aware Planning for Safe and Efficient Heterogeneous Multi-Robot Routing

**arXiv ID:** 2601.10340 | [PDF](https://arxiv.org/pdf/2601.10340v1)

**作者:** David Morilla-Cabello `[一作]` (Instituto de Investigacion en Ingenieria de Aragon), Eduardo Montijano `[通讯]` (Instituto de Investigacion en Ingenieria de Aragon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套基于语义感知的异构多机器人巡检协同框架（CHORAL），能够利用开放词汇视觉模型构建度量-语义地图，并在此基础上通过异构车辆路径规划（Heterogeneous Vehicle Routing Problem）实现任务分配与路径规划。

**💡 创新点**

创新点包括：① 将开放词汇语义理解直接嵌入路径代价；② 通过地形可通行性与碰撞风险两类安全成本，对不同平台计算专属代价；③ 在VRP中加入容量约束与最小-最大目标，以平衡异构机器人负载；④ 将语义地图、PRM、RRT*、A*、OR‑Tools等模块化集成至ROS 2容器化架构。

**🔧 技术方法**

使用的技术主要有：RGB‑D语义感知（Trident、RayFront），3D体素映射（bloomxai/Bonxai），2D网格投影，PRM + RRT* + A* 生成任务间连通图，基于统计安全模型的代价计算，OR‑Tools求解异构VRP，ROS 2导航栈（Nav2、Aerostack2）执行路径。

**📊 数据集**

实验数据集包括：四种合成地图（orchard、forest、park、cave）以及真实实验室环境（6 m × 5.5 m）中的障碍物与动物玩偶目标；同时利用公开的RGB‑D图像和开放词汇模型在同一地图上生成语义特征。

**📈 对比分析**

与传统的仅考虑距离的同构VRP做对比，CHORAL在大多数场景下实现了更低的安全事故概率（terrain 与 collision 两项均下降），即使总距离略增，却显著降低了总耗时；在真实实验中，异构规划使地面机器人避开粗糙地形、空中机器人保持安全间隙，任务完成时间分别约 25 s 与 10 s，优于单一平台或同构规划的预期。

**⚠️ 局限性**

局限性主要包括：① 需要先期完成高质量语义映射，若感知误差大会影响代价估计；② 计算量在大规模开放环境下仍有提升空间；③ 真实实验依赖外部定位（OptiTrack/运动捕捉），对野外部署有一定限制；④ 目前仅支持相对简单的任务类与可通行性标注，复杂的多层任务或动态环境需进一步扩展。

---

## 296. Aletheia-Probe: A Tool for Automated Journal Assessment

**arXiv ID:** 2601.10431 | [PDF](https://arxiv.org/pdf/2601.10431v1)

**作者:** Andreas Florath `[一作]` `[通讯]` (Deutsche Telekom AG), Andreas Florath (Deutsche Telekom AG)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

**🎯 论文内容**

开发了Aletheia‑Probe工具，用于自动化评估期刊合法性，整合多源数据库与模式分析；

**💡 创新点**

创新点在于多源聚合架构、透明可信度指标、显式来源报告以及开源实现；

**🔧 技术方法**

采用数据同步、模糊匹配、交叉验证、信任度打分、API后端（OpenAlex、Crossref）、命令行与程序化接口等技术；

**📊 数据集**

使用的数据库包括DOAJ、Beall’s List、Scopus、各国教育部名单、KSCIEN、Retraction Watch，以及OpenAlex和Crossref的元数据；

**📈 对比分析**

目前未提供定量性能评估，文中表示后续工作将展示实证验证；

**⚠️ 局限性**

局限性主要是覆盖偏向英语/西方期刊、对区域语言和地区期刊覆盖不足、对高级仿冒手段检测有限、部分源更新不及时以及对高置信度判断的依赖。

---

## 297. Placement Delivery Array for Cache-Aided MIMO Systems

**arXiv ID:** 2601.10422 | [PDF](https://arxiv.org/pdf/2601.10422v1)

**作者:** Yifei Huang `[一作]` (Guangxi Normal University), Giuseppe Caire `[通讯]` (Technische Universität Berlin)

**通讯引用:** 28788 | [OpenAlex ID](https://openalex.org/A5058252389)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种新的组合结构MIMO-Placement Delivery Array（MIMO-PDA），用于设计在无编码缓存和一次性零强制（ZF）传输条件下的多天线MIMO缓存广播系统的编码缓存方案。

**💡 创新点**

创新点包括：①统一描述MIMO缓存的缓存放置与交付过程；②给出sum-DoF上界并证明其最优性；③构造两类MIMO-PDA，分别在严格参数约束下实现线性子分层，和在更宽松约束下实现指数级子分层，显著降低子分层级数；④通过组合学工具（Baranyai定理、完全超图分解、完全匹配）实现高效的结构构造。

**🔧 技术方法**

采用的技术主要有：组合学PDA框架扩展、零强制预编码设计、Schwartz–Zippel引理证明可实现性、Baranyai分解与完全匹配构造多层PDAs、以及多维超图分解与匹配技术。

**📊 数据集**

本工作为理论性研究，无需具体数据集，主要验证在多用户MIMO缓存广播模型中的理论性能。

**📈 对比分析**

通过与已有的TST（Tehrani-Stark-Teh）MIMO-PDA方案比较，数值实验表明在相同系统参数下，第二类MIMO-PDA的子分层数可比TST低数个数量级，甚至指数级降低，而sum-DoF保持最大值，证明了方案在实现复杂度与性能上的优势。

**⚠️ 局限性**

局限性：①构造方法仍依赖严格的参数约束，特别是对t、G、L的比例关系；②虽然子分层大幅下降，但在极大用户数下仍可能产生较高的子分层；③实现时需要高精度连续信道估计以满足Schwartz–Zippel条件，实际系统的鲁棒性和复杂度仍待进一步评估。

---

## 298. ErrEval: Error-Aware Evaluation for Question Generation through Explicit Diagnostics

**arXiv ID:** 2601.10406 | [PDF](https://arxiv.org/pdf/2601.10406v1)

**作者:** Weiping Fu `[一作]` (Xi'an Jiaotong University), Jun Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 73099 | [OpenAlex ID](https://openalex.org/A5100361698)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于错误诊断的问答生成评估框架 ErrEval，旨在通过显式识别错误来提升 LLM 评估的可靠性。

**💡 创新点**

创新点在于将错误类型识别与 LLM 评估耦合，形成两阶段错误诊断+分量化评分，并引入轻量级可插拔错误识别器。

**🔧 技术方法**

主要技术包括构建 11 类错误税onomies、基于 RoBERTa 的多标签错误识别器、迭代优化训练、以及在 LLM 提示中嵌入诊断信息。

**📊 数据集**

使用 QGEval、SimQG 以及 SQuAD 2.0 三个公开问答生成评测数据集进行实验。

**📈 对比分析**

在三大基准上，ErrEval 在四种 LLM 评估器上平均提升与人工评分的 Pearson 相关性 12–13%，并显著降低低质量问题的过估计率，表现优于传统相似度、生成式和 LLM 评估基线。

**⚠️ 局限性**

主要局限包括仅适用于问答生成任务、对错误诊断信息的利用方式仍较简单、以及对其他生成任务的迁移需要额外构建错误分类体系。

---

## 299. Multi-Temporal Frames Projection for Dynamic Processes Fusion in Fluorescence Microscopy

**arXiv ID:** 2601.10392 | [PDF](https://arxiv.org/pdf/2601.10392v1)

**作者:** Hassan Eshkiki `[一作]` (Swansea University), Fabio Caraffini `[通讯]` (Swansea University)

**通讯引用:** 1880 | [OpenAlex ID](https://openalex.org/A5032811380)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种将多帧荧光显微镜视频通过预处理和Z投影融合成高质量2D图像的通用框架，显著提升细胞可见性和后续分割效果。

**💡 创新点**

首次将CLAHE、伽马校正、噪声过滤与六种Z投影（MIP、SP、AP、PDP、STDP、QP）组合应用于动态Ca²⁺信号的荧光显微镜时域融合，并用无参考图像质量评估为融合质量提供量化指导。

**🔧 技术方法**

使用的技术包括：CLAHE自适应直方图均衡、Gamma校正、Median/Bilateral/Non‑Local Means滤波；Z投影技术（MIP、SP、AP、PDP、STDP、QP）；无参考图像质量评估（PIQE、NIQE、BRISQUE）；以及细胞分割对比指标（IoU、背景误检率、细胞计数、细胞面积等）。

**📊 数据集**

实验数据来自91段HL‑1心肌细胞的CLSM Ca²⁺信号视频，每段150帧，512×512像素，使用Fluo‑3 AM荧光探针进行成像。

**📈 对比分析**

将新融合方法与传统帧平均/堆叠方法对比；在细胞计数上平均提升约44%；IoU在高倍率图像上显著提高；基于NIQE、PIQE、BRISQUE的无参考评估与专家视觉评价基本一致，MIP、QP、SP被认为是最佳投影方案。

**⚠️ 局限性**

主要局限包括：无参考质量评估指标（尤其BRISQUE）对显微镜图像适用性有限；实验仅针对单一HL‑1细胞数据，未验证对不同组织或更大规模数据集的泛化；投影与预处理参数需针对具体实验进行调优。

---

## 300. C-GRASP: Clinically-Grounded Reasoning for Affective Signal Processing

**arXiv ID:** 2601.10342 | [PDF](https://arxiv.org/pdf/2601.10342v1)

**作者:** Cheng Lin Cheng `[一作]` (National Central University), Chai Kai Chang `[通讯]` (National Central University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出并实现了一套名为 C‑GRASP 的基于 RAG 与 LLM 的心率变异性（HRV）临床推理管道，能够对情绪诱发实验中的 HRV 数据进行可追踪、可解释的多步骤解读。

**💡 创新点**

核心创新点包括：
1) 双重 Z‑score 优先层次（Delta Z‑score 先行），解决传统 Z‑score 与个体基线方向不一致问题；
2) 针对 RSA 干扰、短时数据长度、非线性指标不稳定等常见失误的量化守门阀；
3) 在 RAG 检索中嵌入证据治理与量化权重，确保检索结果与信号质量同步；
4) 将量化图像特征（如 Poincaré 点数、PSD 频率差）直接作为守门阀输入，避免视觉误读；
5) 通过步骤化推理与模板约束实现“医学推理一致性（CRC）”评估。

**🔧 技术方法**

使用技术包括：
- Retrieval‑Augmented Generation (PIKE‑RAG) 与 BioLORD‑2023 编码器；
- LLM（MedGemma 27B‑Thinking、Qwen‑8B‑it 等）与 chain‑of‑thought 结构；
- 量化守门阀（RSA 级别、N_poincare、数据长度、指标可靠性权重）；
- 双重 Z‑score 计算与优先级决策；
- 模板约束生成与后处理校验（防止数字/文本不一致）。

**📊 数据集**

主要使用数据集为 DREAMER（23 受试者，414 次实验），其中 233 次为非中性情绪标签，用于 4 类情绪分类评估；也对 181 次中性实验进行格式与一致性检验。

**📈 对比分析**

与传统 LLM 直接分类和无守门阀版本比较，C‑GRASP 在 4‑类准确率上提升至 37.3%（MedGemma‑Thinking），并获得 CRC 69.6%。在 Arousal、Vagal 维度的单独准确率也较高；F1、WAD 等指标显示整体性能稳定。消融实验表明，去掉 RAG、守门阀或 Delta Z‑score 均会显著降低一致性与准确率，验证各模块的重要性。

**⚠️ 局限性**

局限性包括：
1) 基线构造采用回溯式全部试验，导致轻度数据泄露，未能在真实可穿戴情境下实现因果基线；
2) 仍需在更大规模、不同实验协议的数据上验证鲁棒性；
3) 对高频率、长时序信号的支持有限，当前主要针对短时 4‑8 秒窗口；
4) 需要进一步减少 LLM 的数值幻觉与标签映射错误；
5) 计算成本高（需要大型 GPU 与向量数据库），在边缘设备部署仍有障碍。

---

## 301. Meta Dynamic Graph for Traffic Flow Prediction

**arXiv ID:** 2601.10328 | [PDF](https://arxiv.org/pdf/2601.10328v1)

**作者:** Yiqing Zou `[一作]` (Beijing Institute of Technology), Sijie Ruan `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 1208 | [OpenAlex ID](https://openalex.org/A5006117974)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了MetaDG框架，用动态节点嵌入和动态图结构统一建模交通流预测中的时空动态与异质性。

**💡 创新点**

创新点在于：①动态节点嵌入与时空相关增强，②生成动态邻接矩阵、元参数与边权调整矩阵，实现动态与异质性的全局统一；③动态图资格模块提升信息传播可靠性。

**🔧 技术方法**

使用的技术包括：GCRU、动态节点生成模块、时空相关增强（SCE/TCE/STCE）、动态图资格模块、元参数生成、动态图卷积。

**📊 数据集**

使用了四个实际交通流数据集：PEMS03、PEMS04、PEMS07、PEMS08。

**📈 对比分析**

与STGCN、DCRNN、GWNet、AGCRN、STSGCN、PDFormer、DGCRN、HimNet等基线对比，MetaDG在MAE/RMSE/MAPE等指标上均优于基线，尤其在长时序预测上优势明显。

**⚠️ 局限性**

局限性：依赖于预训练的静态节点嵌入，模型复杂度较高；对不同规模图的适应性待进一步验证；对动态图质量的依赖仍需改进。

---

## 302. ROMA: Real-time Omni-Multimodal Assistant with Interactive Streaming Understanding

**arXiv ID:** 2601.10323 | [PDF](https://arxiv.org/pdf/2601.10323v1)

**作者:** Xueyun Tian `[一作]` (CAS Key Laboratory of AI Safety Institute of Computing Technology CAS), Huawei Shen `[通讯]` (CAS Key Laboratory of AI Safety Institute of Computing Technology CAS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个实时全模态助手ROMA，支持音视频流式输入，实现统一的主动监测和被动问答。

**💡 创新点**

同步将稠密音频与稀疏视频帧对齐为1秒单位，并提出轻量化“Speak Head”解耦响应时机与文本生成，以及两阶段训练课程统一主动与被动交互。

**🔧 技术方法**

采用chunked Time-aligned Multimodal RoPE、LLM主干、并行Speak Head、两阶段微调（流式模板对齐 + 时序决策）以及KV缓存实现低延迟推理。

**📊 数据集**

训练使用自编流式数据集，包括27k主动警报、109k实时叙述、540k反应式问答，来源于DiDeMo、OOPS、Charades-STA、MMDuetIT、COIN、YouCook2、ActivityNet等。

**📈 对比分析**

与现有流式VideoLLM（VideoLLM-Online、MMDuet、Dispider）及全模态模型（Qwen2.5-Omni、MiniCPM-o、VITA-1.5）在12个统一基准（主动报警、叙述、QA）对比，主动任务实现SOTA，主动+被动性能均优于同类模型。

**⚠️ 局限性**

对信号失真和音视频不同步敏感，受限于固定上下文窗口难以捕获数小时级长程依赖，推理效率与输出质量权衡尚需改进。

---

## 303. Reinforcement Learning with Multi-Step Lookahead Information Via Adaptive Batching

**arXiv ID:** 2601.10418 | [PDF](https://arxiv.org/pdf/2601.10418v1)

**作者:** Nadav Merlis `[一作]` (Technion), Nadav Merlis `[通讯]` (Technion)

**通讯引用:** 194 | [OpenAlex ID](https://openalex.org/A5018784842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了在拥有多步lookahead信息的离散马尔可夫决策过程（MDP）中的强化学习问题，并提出了自适应分批策略（ABP）以及基于UCB的学习算法，实现对最佳ABP的学习与执行。

**💡 创新点**

创新点包括：①提出自适应分批的概念，允许每一步根据当前状态选择合适的lookahead长度；②推导ABP的最优Bellman方程，实现可计算的最优策略；③设计AL‑UCB算法并给出近似最优的调度理论（上界仅受lookahead长度ℓ的常数因子影响）。

**🔧 技术方法**

主要技术手段包括：动态规划与最优贝尔曼方程求解、UCB型上界与方差奖金、Freedman不等式的集中分析、覆盖与递归式误差传播等。

**📊 数据集**

论文不使用真实数据集，所有分析均在理论模型下完成；实验部分仅为理论推导与上界说明。

**📈 对比分析**

与固定分批策略和基于MPC的后向归纳策略进行理论比较，证明固定分批在某些环境下指数级低效，MPC在部分环境中同样子最优；AL‑UCB的调度误差上界为O(√(H³SKℓ log (SHℓK/δ)))，即除掉动作数因子后与一阶lookahead相当。

**⚠️ 局限性**

局限性：①假设lookahead信息完美且可预知；②上界与下界之间仍存ℓ倍数差距，尚未证明最优；③仅在理论层面验证，缺乏实验评估；④对大规模状态空间的可扩展性未做讨论。

---

## 304. LLMdoctor: Token-Level Flow-Guided Preference Optimization for Efficient Test-Time Alignment of Large Language Models

**arXiv ID:** 2601.10416 | [PDF](https://arxiv.org/pdf/2601.10416v1)

**作者:** Tiesunlong Shen `[一作]` (Nanyang Technological University), Erik Cambria `[通讯]` (Nanyang Technological University)

**通讯引用:** 51808 | [OpenAlex ID](https://openalex.org/A5100752356)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LLMdoctor框架，通过小型医生模型在测试时对大型冻结LLM进行token级别的流引导对齐。

**💡 创新点**

创新点在于从患者模型行为差异中提取token级奖励，并利用GFlowNet流守恒的TFPO实现子路径级对齐，既精细又保持生成多样性。

**🔧 技术方法**

使用了token级奖励提取、流守恒（GFlowNet）、TFPO、流引导奖励解码等技术。

**📊 数据集**

在HH‑RLHF、PKU‑SafeRLHF‑10K、UltraFeedback等人类偏好数据集上进行实验。

**📈 对比分析**

与Greedy、Top‑k、Top‑p、GenARM、ARGS、DPO等基线比较，LLMdoctor在GPT‑4o评估的Win+½Tie上显著优于所有测试时方法，且接近或优于DPO，同时保持更高的生成多样性。

**⚠️ 局限性**

局限在于需要先从患者模型提取token奖励，且对极端长序列或多目标实时动态平衡仍有挑战。

---

## 305. Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering

**arXiv ID:** 2601.10402 | [PDF](https://arxiv.org/pdf/2601.10402v1)

**作者:** Xinyu Zhu `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8269 | [OpenAlex ID](https://openalex.org/A5066373402)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 ML‑Master 2.0，一款面向超长时序机器学习工程（MLE）的自主代理，通过层级认知缓存（HCC）实现经验、知识与智慧的分层管理与动态迁移，从而提升在长周期实验中的决策连续性和最终性能。

**💡 创新点**

创新点包括：① 将传统的线性上下文拼接转变为认知累积机制；② 提出层级认知缓存架构，将执行经验、阶段性知识与可迁移智慧分别存储于三级缓存；③ 设计动态上下文迁移（prefetching、hit、promotion）协议，使短期细节与长期战略有效解耦，突破上下文窗口限制；④ 在 MLE 场景中首次验证此机制对超长时序自治的显著提升。

**🔧 技术方法**

技术手段：大语言模型 Deepseek‑V3.2（用于编码与推理），多级缓存结构（Evolving Experience、Refined Knowledge、Prior Wisdom），LLM 归纳与摘要（context promotion），并行实验路径规划与执行，任务级与阶段级迁移策略，利用语义嵌入实现跨任务检索。

**📊 数据集**

数据集：OpenAI 的 MLE‑Bench（75 个 Kaggle 任务）作为主要评测基准，并用 407 个 Kaggle 竞赛（不含 MLE‑Bench）作为 warm‑up 构建 Prior Wisdom，实验预算为每个任务 24 小时。

**📈 对比分析**

与多种公开（OpenHands、MLAB、AIDE、R&D‑Agent、AIRA‑dojo）和闭源基线（Leeroo、Thesis、MLE‑STAR‑Pro、FM Agent 等）进行对比。ML‑Master 2.0 在 MLE‑Bench 上取得 56.44% 的平均 medal rate，低/中/高难度分别为 75.8% / 50.9% / 42.2%，相较前置方法提升 92.7%（相对 ML‑Master）和 11.2%（相对最佳闭源）。有效提交率 95.6%，在 63.1% 的任务中超过前 50% 人类参与者。

**⚠️ 局限性**

局限性：① 对算力需求高（36 CPU + RTX 4090），成本显著；② 仍依赖 LLM 的推理性能与抽象质量，迁移质量受限；③ 缓存阈值与迁移策略仍需手工调参；④ 仅在计算机模拟的 MLE 场景验证，尚未在真实物理实验或更大规模任务中测试其可迁移性；⑤ 长期持续性与安全性评估不足。

---

## 306. Multiaccess Coded Caching with Heterogeneous Retrieval Costs

**arXiv ID:** 2601.10394 | [PDF](https://arxiv.org/pdf/2601.10394v1)

**作者:** Wenbo Huang `[一作]` (Huazhong University of Science and Technology), Giuseppe Caire `[通讯]` (Technische Universität Berlin)

**通讯引用:** 28788 | [OpenAlex ID](https://openalex.org/A5058252389)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了在多接入编码缓存（MACC）系统中考虑用户检索缓存节点成本与服务器广播成本的总通信成本，并通过叠加编码（superposition coding）对已有的 Cheng 等人方案进行层次化改进，提出一个基于成本的优化框架。

**💡 创新点**

创新点包括：
• 将不同接入层级的编码方案叠加，形成可调节的成本权重；
• 对总成本进行优化，证明最优解仅需在最多两个接入层级上分配非零权重（稀疏性）；
• 基于该稀疏性设计了结构感知的贪心索引搜索 + 有界范围的 SQP 算法，显著降低求解复杂度。

**🔧 技术方法**

使用技术：
• 叠加编码（superposition coding）与多接入编码缓存框架；
• 线性规划与二次规划的组合（SQP）求解非凸优化；
• 贪心索引搜索与局部邻域探索；
• 理论分析（线性规划基本可行解、稀疏性证明）。

**📊 数据集**

没有使用公开数据集；实验通过对 (K=100, N=100, M=5, ρ=65, μ=(1,2,…,L)) 进行模拟，L 取 2~20，生成随机用户请求，评估通信成本。

**📈 对比分析**

与基线方案（Cheng 等人直接应用于异质成本环境）比较，基线成本随 L 先升后降，峰值出现在 L≈8；而本文提出的叠加方案在 L<16 时成本保持平稳且显著低于基线；当 L>16 时两者收敛，表明叠加方案在低到中等接入层级下具有明显优势。

**⚠️ 局限性**

局限性：
• 仅考虑单层编码与一次性传输，未涵盖多层缓存或多轮交互；
• 优化算法仍为启发式，虽然已降低复杂度，但在极大规模或多维成本空间中可能需要进一步改进；
• 只在均匀请求分布下验证，缺乏对实际非均匀需求或动态缓存更新的评估。

---

## 307. Handling Missing Modalities in Multimodal Survival Prediction for Non-Small Cell Lung Cancer

**arXiv ID:** 2601.10386 | [PDF](https://arxiv.org/pdf/2601.10386v1)

**作者:** Filippo Ruffini `[一作]` (Università Campus Bio-Medico di Roma), Paolo Soda `[通讯]` (Umeå University)

**通讯引用:** 3196 | [OpenAlex ID](https://openalex.org/A5003216983)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本研究提出一种缺失感知的多模态生存预测框架，集成CT、WSI和临床变量，实现对不可切除II–III期非小细胞肺癌患者的整体生存预测。

**💡 创新点**

创新点在于：①利用预训练的基础模型提取高层特征；②采用NAIM注意力机制实现缺失感知编码；③通过中间融合与Oblivious Differentiable Survival Tree实现跨模态互补与鲁棒预测，避免完整案例过滤和强制插补。

**🔧 技术方法**

技术核心包括：Merlin、CLAM、MI‑Zero、TITAN等Foundation Models进行特征提取；NAIM+ODST缺失感知编码；中间融合（ConcatODST）与时间依赖Cox损失的端到端训练。

**📊 数据集**

数据集来源于意大利Rome Campus Bio‑Medico医院的179例不可切除II–III期NSCLC患者，包含CT影像、WSI切片和完整临床表格数据。

**📈 对比分析**

与传统CPH、RSF、SGB等基线以及早期/后期融合模型对比，中间融合模型在C-index 0.733、Uno 0.700、td-AUC 0.786等指标上表现最佳，尤其WSI+临床组合显著优于单模态或其它融合策略。

**⚠️ 局限性**

局限包括：单中心、样本量有限；CT模态在该数据集中的预测贡献不大；模型计算量大且需大量预训练模型；缺乏外部多中心验证。

---

## 308. RSA-Bench: Benchmarking Audio Large Models in Real-World Acoustic Scenarios

**arXiv ID:** 2601.10384 | [PDF](https://arxiv.org/pdf/2601.10384v1)

**作者:** Yibo Zhang `[一作]` (Beijing University of Posts and Telecommunications), Li Sun `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 3847 | [OpenAlex ID](https://openalex.org/A5100318687)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了RSA‑Bench基准，系统评估 Audio 大型模型在多源真实噪声场景下的鲁棒性。

**💡 创新点**

创新之处在于采用生态化多源噪声叠加、六任务评测框架，并揭示感知‑认知差距、声音相似干扰与去噪悖论。

**🔧 技术方法**

主要技术包括多源噪声合成、能量归一化、LLM‑as‑Judge 评估与多模型对比。

**📊 数据集**

数据来源为多任务公开数据集（LibriSpeech、IEMOCAP、MELD、SpokenMQA、SLUE、OpenHermes 等）和 ESC 环境音，构成超过 10 万条样本。

**📈 对比分析**

通过对 11 个 ALLM 在 17 种噪声级别进行测评，发现低级感知任务相对稳健，而高阶推理任务在噪声强度升高时急剧崩溃，且去噪预处理往往进一步恶化性能。

**⚠️ 局限性**

研究局限在于仅在推理阶段尝试去噪，未探索训练时的噪声适应或对抗训练等提升鲁棒性的方式。

---

## 309. Online identification of nonlinear time-varying systems with uncertain information

**arXiv ID:** 2601.10379 | [PDF](https://arxiv.org/pdf/2601.10379v1)

**作者:** He Ren `[一作]` (Taiyuan University of Technology), Gang Dang `[通讯]` (Taiyuan University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种 Bayesian Regression-based Symbolic Learning（BRSL）框架，用于在线识别非线性时变系统，并在同一模型中实现稀疏符号表达与不确定性量化。

**💡 创新点**

创新点在于：
1) 将马蹄式稀疏先验引入贝叶斯符号回归，构建统一的概率状态空间模型；
2) 设计递归贝叶斯更新与遗忘因子，保证历史数据衰减并实时更新；
3) 给出递归条件与收敛证明，确保参数估计在持久激励下收敛并保持正定信息矩阵。

**🔧 技术方法**

所用技术包括：贝叶斯回归、马蹄式稀疏先验、递归贝叶斯滤波、Kronecker 乘积、SHAP 解释、Lorenz 系统仿真等。

**📊 数据集**

数据集：
- 合成高斯噪声稀疏线性回归数据（100 组，30% 非零系数）；
- Lorenz 系统时变参数数据，含噪声，部分状态已知，构造二阶多项式基函数库。

**📈 对比分析**

与传统离线符号方法（如 SINDy、G‑SINDy、MGSINDy）进行对比。实验表明，BRSL 在高噪声环境下保持低估计误差、快速收敛、可解释性强，且能够实现实时预测和在线自适应。

**⚠️ 局限性**

限制：
- 递归条件要求信息矩阵正定，对窗口宽度、遗忘因子等超参数敏感；
- 需要满足持续激励与噪声独立等理论假设，实际工业环境可能不完全满足；
- 预设的基函数库若不完整，可能导致识别误差或模型偏差。

---

## 310. Towards Efficient Low-rate Image Compression with Frequency-aware Diffusion Prior Refinement

**arXiv ID:** 2601.10373 | [PDF](https://arxiv.org/pdf/2601.10373v1)

**作者:** Yichong Xia `[一作]` (Tsinghua University), Bin Chen `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 32290 | [OpenAlex ID](https://openalex.org/A5100427314)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DiffCR框架，实现低比特率下的高保真图像压缩。

**💡 创新点**

创新性设计了频率感知跳跃估计（FaSE）、频率解耦注意力（FDA）和两步一致性改进估计器（CRE）。

**🔧 技术方法**

基于预训练潜在扩散模型、频率域注意力、一致性模型以及混合语义控制的两阶段训练。

**📊 数据集**

在CLIC20、DIV2K、Kodak、Tecnick等公开图像压缩基准集上进行实验。

**📈 对比分析**

与传统压缩标准和最新扩散压缩方法对比，BD‑rate在LPIPS 27.2%、PSNR 65.1%显著降低，解码速度提升10×以上。

**⚠️ 局限性**

仍需依赖大规模预训练模型，语义嵌入对不同图像描述器敏感，且在极低比特率时可能出现色彩偏移。

---

## 311. Fine-Grained Human Pose Editing Assessment via Layer-Selective MLLMs

**arXiv ID:** 2601.10369 | [PDF](https://arxiv.org/pdf/2601.10369v1)

**作者:** Ningyu Sun `[一作]` (Shanghai Jiao Tong University), Xiaokang Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 144557 | [OpenAlex ID](https://openalex.org/A5100381753)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了细粒度人类姿态编辑基准 HPE‑Bench，并提出了一个统一的评估框架，实现对姿态编辑图像的真实性检测与多维质量回归。

**💡 创新点**

创新点包括：① HPE‑Bench 细粒度标注真实性与三维质量（感知、对齐、属性保持）；② 通过对比 LoRA 视觉调优与层敏感性分析（KL、LDR、熵）自动选取最适合的 MLLM 层；③ 将检测与质量回归统一到同一 MLLM 结构中，显著提升对高频伪造痕迹的感知。

**🔧 技术方法**

采用多模大语言模型（如 InternVL3.5、Qwen3‑VL 等）作为主干，结合对比 LoRA 调优、层敏感性分析（KL 散度、局部判别比、熵）以及多头回归解码器，使用交叉熵与 MSE 损失进行训练。

**📊 数据集**

使用了 HPE‑Bench 数据集，包含 1,700 张来自 17 种主流姿态编辑模型的样本，配备真实性标签和三维质量评分。

**📈 对比分析**

与多种基线（深度伪造检测器、AI 图像生成检测器、传统 IQA 与 MLLM 评测）进行对比；在真实性检测上 Acc/F1 最高达 95.5%；在三维质量上 Spearman/Kendall/Pearson 均超过 0.86，显著优于现有方法。

**⚠️ 局限性**

局限性包括：仅针对人类姿态编辑，未扩展到其他视觉编辑任务；依赖预训练 MLLM 与对比 LoRA 的训练成本较高；对极端或多模态伪造的鲁棒性待进一步验证。

---

## 312. Inverse Learning in $2\times2$ Games: From Synthetic Interactions to Traffic Simulation

**arXiv ID:** 2601.10367 | [PDF](https://arxiv.org/pdf/2601.10367v1)

**作者:** Daniela Aguirre Salazar `[一作]` (TU Darmstadt), Tatiana Tatarenko `[通讯]` (TU Darmstadt)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出两种针对2×2博弈的逆向博弈学习方法：闭式相关均衡最大似然估计（CE-ML）和Logit最佳反应最大似然估计（LBR-ML）

**💡 创新点**

CE-ML通过解析相关均衡多面体的顶点实现参数估计；LBR-ML利用Logit最佳反应动力学的唯一平稳分布建模行为，二者覆盖从静态均衡到动态自适应的完整谱

**🔧 技术方法**

运用线性效用模型、最大似然估计、凸优化、Markov链理论与闭式解析

**📊 数据集**

使用合成“Chicken‑Dare”博弈数据以及SUMO交通模拟得到的交叉路口行动数据

**📈 对比分析**

与ICE（约束相关均衡优化）比较，CE-ML在相关均衡一致的数据下参数恢复和分布拟合最佳；LBR-ML在无协调或噪声数据下保持较高准确率，ICE表现最差

**⚠️ 局限性**

CE-ML依赖于相关均衡假设，无法处理无协调情形；LBR-ML受限于Logit模型的可解释性和对高维多玩家扩展的挑战

---

## 313. SRAW-Attack: Space-Reweighted Adversarial Warping Attack for SAR Target Recognition

**arXiv ID:** 2601.10324 | [PDF](https://arxiv.org/pdf/2601.10324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 314. Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale

**arXiv ID:** 2601.10338 | [PDF](https://arxiv.org/pdf/2601.10338v1)

**作者:** Yi Liu `[一作]` (Quantstamp), Leo Zhang `[通讯]` (Griffith University)

**通讯引用:** 4477 | [OpenAlex ID](https://openalex.org/A5015011245)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 AI 代理技能生态进行首次大规模安全评估，收集并分析了 31,132 个技能，发现 26.1% 存在漏洞，构建了基于静态分析与 LLM 语义分类的多阶段检测框架。

**💡 创新点**

创新点在于：①提出了包含 14 种模式的漏洞分类体系；②开发了融合静态规则与 LLM 判别的自动化检测工具；③公开了完整的标注数据集和工具链，为后续研究提供基础设施。

**🔧 技术方法**

技术手段包括：①静态代码扫描（自定义正则/语法规则）；②LLM‑Guard 语义分类（提示注入、秘密泄露、混淆检测等）；③基于 Claude 3.5 Sonnet 的结构化分类与置信度判断；④置信度阈值与多阶段融合策略。

**📊 数据集**

数据集来自两大技能市场（skills.rest 与 skillsmp.com），共 42,447 个原始技能，经过去重与过滤后得到 31,132 个独立技能；其中 3,574 个包含可执行脚本，8 个功能类别。

**📈 对比分析**

方法对比：与单一静态扫描或单一 LLM 分类相比，多阶段融合显著提升了性能，最终检测精度 86.7%（±4.7%）与召回率 82.5%（±5.3%），比单一方案提升约 15% 精度与 8% 召回。

**⚠️ 局限性**

局限性包括：①采样时间点为 2025 年 12 月，生态仍在演进，结果可能随时间波动；②已删除的 17.3% 技能可能隐藏更高风险，导致低估；③检测框架无法完全区分恶意与疏忽漏洞，导致部分误判；④对不同功能类别的精度未充分验证，需更大样本进一步评估。

---

## 315. An analytic theory of convolutional neural network inverse problems solvers

**arXiv ID:** 2601.10334 | [PDF](https://arxiv.org/pdf/2601.10334v1)

**作者:** Minh Hai Nguyen `[一作]`, Pierre Weiss `[通讯]` (IRIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在卷积神经网络（CNN）解决图像逆问题时的可解析理论框架，将训练好的CNN视为在满足平移等变性与局部性约束下的MMSE估计器（LE‑MMSE）

**💡 创新点**

首次在一般逆问题上给出带功能约束的MMSE闭式解，并证明LE‑MMSE能够精准预测多种CNN架构的输出，从而把黑盒模型变成可解释的预测器

**🔧 技术方法**

基于MMSE理论、平移等变性与局部性约束、投影逆算子、加性高斯噪声模型，并用加噪卷积/插值等预逆算子；实验中使用UNet、ResNet、PatchMLP等CNN结构

**📊 数据集**

在FFHQ、CIFAR‑10、FashionMNIST三大图像数据集上进行训练与评估，图像尺寸32×32（部分64×64）

**📈 对比分析**

通过PSNR等指标将训练好的网络输出与LE‑MMSE理论结果对比，发现PSNR差距低于3 dB（大部分≥25 dB），在不同噪声水平、任务、网络、数据集上均保持一致；同时对高/低密度区域和OOD数据做了进一步对比

**⚠️ 局限性**

受限于高斯噪声假设、局部性和平移等变性约束、计算量大导致大尺寸/大数据量实验受限、未考虑长程依赖（如Transformer）、未覆盖非MSE损失和非高斯噪声等情形

---

## 316. On the Capacity of Noisy Frequency-based Channels

**arXiv ID:** 2601.10329 | [PDF](https://arxiv.org/pdf/2601.10329v1)

**作者:** Yuval Gerzon `[一作]` (Technion), Nir Weinberger `[通讯]` (Technion)

**通讯引用:** 272 | [OpenAlex ID](https://openalex.org/A5076051906)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了带识别噪声的频率基通道在短分子DNA存储中的容量极限，提出了上下界并量化了噪声导致的容量损失。

**💡 创新点**

首次将数据处理不等式与Poisson化技术结合，利用Talagrand与Bobkov‑Ledoux不等式精确估计信息密度的浓缩，并给出可达率与容量退化的显式上界。

**🔧 技术方法**

数据处理不等式、Poisson化采样、Talagrand与Bobkov‑Ledoux浓缩不等式、泊松通道熵分析、噪声矩阵特征值分解等信息论与概率工具。

**📊 数据集**

理论模型无实测数据集；仅在DNA存储参数（如|A|=4、β∈(2/3log|A|,1/log|A|)）下进行定量推导。

**📈 对比分析**

与无噪声极限比较，给出噪声诱导的容量损失Δ（如替换噪声Δ=log(1‑p)，擦除噪声Δ=1/8[log((1‑ε)²+4ε²)+6log(1‑ε)]），证明在β范围内退化量可控。

**⚠️ 局限性**

需假设噪声矩阵良好条件并且β>2/3log|A|；Poisson化与浓缩不等式的使用限制了模型扩展，无法覆盖插入/删除错误或更一般的噪声模型。

---

## 317. LADFA: A Framework of Using Large Language Models and Retrieval-Augmented Generation for Personal Data Flow Analysis in Privacy Policies

**arXiv ID:** 2601.10413 | [PDF](https://arxiv.org/pdf/2601.10413v1)

**作者:** Haiyue Yuan `[一作]` (University of Kent), Shujun Li `[通讯]` (University of Kent)

**通讯引用:** 9168 | [OpenAlex ID](https://openalex.org/A5100745576)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个名为 LADFA 的端到端框架，利用大型语言模型（LLM）和检索增强生成（RAG）技术自动从隐私政策中抽取个人数据流，并构建数据流图进行分析。

**💡 创新点**

创新点在于将 LLM 与 RAG 相结合，并构建了针对数据类别、数据消费者类型、处理目的和方法的自定义知识库，从而显著提高了数据流抽取的准确性和可解释性。

**🔧 技术方法**

核心技术包括 LLM Prompt‑Chaining、RAG、向量检索、知识库嵌入、图数据库（NetworkX/pyvis）以及文本分割与预处理。

**📊 数据集**

实验数据集为十个汽车 OEM 的连接汽车移动应用隐私政策（HTML 格式），共约 10 万字；使用了公开的 OPP‑115、个人信息分类、用途案例等资料构建知识库。

**📈 对比分析**

通过三名领域专家对 150 条抽取结果进行 7‑级 Likert 评估，平均得分 6.2–6.9，Gwet’s AC1 0.94，说明框架在数据类型、数据流、消费者类型和处理方法上的性能优异；对数据类别和处理目的的准确率略低。

**⚠️ 局限性**

局限性包括对知识库的依赖导致部分分类歧义、LLM 仍可能产生幻觉、实验仅涵盖汽车行业且缺乏公开基准，且当前实现对多语种和不同格式隐私政策的泛化能力有限。

---

## 318. Discrete Feynman-Kac Correctors

**arXiv ID:** 2601.10403 | [PDF](https://arxiv.org/pdf/2601.10403v1)

**作者:** Mohsin Hasan `[一作]` (Universite de Montreal), Kirill Neklyudov `[通讯]` (Institut Courtois)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `09944146-298c-433e-89df-37255de463d7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在推理时通过改进的 SMC 算法实现对离散扩散模型生成分布的控制，涵盖温度退火、概率乘积与奖励加权三种模式；

**💡 创新点**

提供了一套通用的 CTMC 与 Feynman‑Kac 理论框架，使得无需再训练即可在推理阶段调整生成分布；

**🔧 技术方法**

采用连续时间马尔可夫链、Feynman‑Kac 公式与自适应重采样的 Sequential Monte Carlo（SMC）技术；

**📊 数据集**

分别在 Ising 模型、编程代码生成（HumanEval、MBPP）、参数推断的合成数据集以及蛋白质序列生成（ESM2‑650M、DPLM‑650M）上进行实验；

**📈 对比分析**

与基线采样、指导式采样（FK Steering、DG‑Exact）等方法对比，证明在低温 Ising 采样、代码生成准确率、参数推断误差与蛋白质设计可行性方面均显著优于传统方法；

**⚠️ 局限性**

主要限制包括推理时需要额外的采样与重采样计算，奖励函数在某些状态下求和成本高，且对大规模词表或长序列的扩展仍需进一步优化。

---

## 319. Global Context Compression with Interleaved Vision-Text Transformation

**arXiv ID:** 2601.10378 | [PDF](https://arxiv.org/pdf/2601.10378v1)

**作者:** Dian Jiao `[一作]` (China Electronics Cloud Technology Co., Ltd.), Feng Huang `[通讯]` (China Electronics Cloud Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VIST2，采用交错文本与视觉编码的 Transformer 架构，实现全局上下文压缩，支持长文本生成与理解。

**💡 创新点**

创新点包括：① 在 Transformer 输入中交错插入文本块与其光学编码，实现全局压缩；② 采用稀疏因果注意力限制视觉令牌只供后续块使用；③ 设计分阶段预训练和指令调优的训练路线，结合图像字幕、OCR 与光学语言建模。

**🔧 技术方法**

技术核心：视觉编码器 SigLIP2 + 视觉变压器、LLM 主干 Qwen3、模态对齐器（MLP）、稀疏因果注意力、光学语言建模（OLM）、分阶段预训练与指令微调。

**📊 数据集**

数据集：SA1B+CoCo 进行图像字幕预训练；WuDao 进行 OCR 与 OLM 预训练；公开长文本与对话数据（约 10M 条）用于指令微调；LongBench、LooGLE、GSM‑8k、MATH、AQUA、CMMLU 用于评测。

**📈 对比分析**

与 Glyph（PCC）、LongLLM、SnapKV 等方法对比，VIST2 在 4×压缩率下实现：首次 token 生成速度提升 3×，内存占用下降 77%，FLOPs 减少 74%；在 LongBench、LooGLE 上表现超越压缩基线，且在多项基准任务中保持与原 LLM 相近的能力。

**⚠️ 局限性**

局限：采用固定压缩率，未针对不同文本密度动态分配视觉令牌；使用通用视觉编码器，未针对文档特征做专门优化；训练仅覆盖监督学习，缺乏基于 RL 的对齐与奖励机制。

---

## 320. FastStair: Learning to Run Up Stairs with Humanoid Robots

**arXiv ID:** 2601.10365 | [PDF](https://arxiv.org/pdf/2601.10365v1)

**作者:** Yan Liu `[一作]` (Harbin Institute of Technology), Jie Zhao `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 36761 | [OpenAlex ID](https://openalex.org/A5070851446)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并验证了FastStair多阶段学习框架，使人形机器人在楼梯上实现高速、稳定的上行和下降。

**💡 创新点**

将基于DCM的脚位规划并行化为GPU可并行搜索，并通过速度专家分解与LoRA融合实现全速域统一策略。

**🔧 技术方法**

结合模型无关强化学习、DCM脚位规划、并行离散搜索、LoRA低秩适配、深度视觉感知和域随机化训练。

**📊 数据集**

使用IsaacLab仿真环境中的多样化楼梯、平地、崎岖地形数据以及RealSense D435i深度相机获取的本地地形图进行训练与部署。

**📈 对比分析**

与基线RL、AMP和无脚位引导策略对比，在多速率楼梯穿越成功率和速度跟踪误差上均优先，占据最高成功率（>70%在2 m/s）并实现1.65 m/s的上升速度。

**⚠️ 局限性**

仍依赖精确地形感知，分层设计复杂，离线仿真与真实场景差距导致高速度下偶尔跌倒，且对极端斜坡或不规则地形的泛化待进一步验证。

---

## 321. PLGC: Pseudo-Labeled Graph Condensation

**arXiv ID:** 2601.10358 | [PDF](https://arxiv.org/pdf/2601.10358v1)

**作者:** Jay Nandy `[一作]` (Fujitsu Research of India), Mahesh Chandran `[通讯]` (Fujitsu Research of India)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种自监督图凝缩框架PLGC，利用伪标签在无标签条件下构造并优化小型合成图，供GNN训练完成节点分类与链接预测等下游任务。

**💡 创新点**

在无标签环境下首次构造潜在伪标签并与图凝缩交替优化；给出伪标签稳定性与聚类误差的理论保证；支持多源异构图聚合，显著提升鲁棒性。

**🔧 技术方法**

使用GNN编码器生成节点嵌入，Sinkhorn正则化分配伪标签，交替的互换视图预测损失与表示匹配（MMD/均方误差），并基于子高斯分布与分离度的理论分析。

**📊 数据集**

在五大基准图数据集上评测：Cora、Citeseer、Ogbn-arxiv、Flickr、Reddit，分别涵盖传递式与归纳式节点分类以及链接预测任务。

**📈 对比分析**

与多种监督（GCond、SFGC、GEOM）与自监督（CTGC、Random/Herding/K-Center）凝缩方法比较；在无噪声场景下与监督方法相当，在标签噪声、少量标签、少样本及多源场景中显著优于监督方法，鲁棒性与精度均有明显提升。

**⚠️ 局限性**

目前仅适用于无异构/时变图；伪标签初始化对嵌入质量敏感；对极大图仍需分布式训练；缺乏对动态或在线更新的理论与实践支持。

---

## 322. SuS: Strategy-aware Surprise for Intrinsic Exploration

**arXiv ID:** 2601.10349 | [PDF](https://arxiv.org/pdf/2601.10349v1)

**作者:** Mark Kashirskiy `[一作]` (AI Talent Hub), Ilya Makarov `[通讯]` (Higher School of Economics)

**通讯引用:** 1786 | [OpenAlex ID](https://openalex.org/A5074238659)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Strategy-aware Surprise (SuS)，一种基于策略空间预后预测误差的内在奖励框架，用于强化学习中的探索。

**💡 创新点**

创新点在于将策略稳定性（SS）和策略惊讶（SuS）两种互补信号联合起来，既鼓励策略连贯，又奖励真正的策略转移，从而克服传统基于状态预测误差的噪声 TV 问题。

**🔧 技术方法**

核心技术包括对策略嵌入的对比学习、前后策略相似度计算、世界模型预测误差与策略差异的乘积以及学习权重的元学习；训练基于 GRPO 的语言模型策略。

**📊 数据集**

实验使用 GSM8K 词算题集，并在 Qwen2.5‑1.5B 语言模型上进行微调，采用 LoRA 适配器。

**📈 对比分析**

与基线、基于困惑度的奖励以及单一 SS 或 SuS 的对照组相比，SuS 在 Pass@1 上提升 17.4%，Pass@5 提升 26.4%，并显著提高策略多样性。

**⚠️ 局限性**

主要局限在于仅在数学推理任务上验证，未在视觉或更大规模环境中测试；同时计算开销略高，需额外的策略编码器和世界模型。

---

## 323. Training-Trajectory-Aware Token Selection

**arXiv ID:** 2601.10348 | [PDF](https://arxiv.org/pdf/2601.10348v1)

**作者:** Zhanming Shen `[一作]` (Zhejiang University), Junbo Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 10853 | [OpenAlex ID](https://openalex.org/A5042402520)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对持续推理蒸馏中的性能崩溃问题，提出了训练轨迹感知的 token 选择方法（T3S），通过在训练过程识别并屏蔽先行学习的 Imitation‑Anchor tokens，解放剩余 token 的学习空间，从而显著提升推理性能。

**💡 创新点**

创新点在于将蒸馏过程中的 Imitation Shock 现象归因于 token 级别的学习冲突，并基于训练轨迹动态构造 token 层面的目标，首次实现了在 AR 与 dLLM 两种模型框架下统一的 token 级优化策略。

**🔧 技术方法**

技术手段包括：token 置信度差分（Δc_t）计算、基于 Imitation‑Bottleneck 的 token 分组、AR 的遮罩式损失以及 dLLM 的联合掩码扩展，以及恢复残差传递（RRT）实验验证。

**📊 数据集**

实验使用了 BOBA‑200、S1K‑200、MATH500、TheoremQA 等数学推理 benchmark，以及 AIME24/25、MATH500 等公开数据集。

**📈 对比分析**

与传统 SFT、RRT、随机掩码等对照相比，T3S 在 Qwen3‑8B、Qwen3‑32B 等模型上在 AIME、MATH 等测试上均实现了数个百分点的提升，甚至在 16B 规模无思考模型中达成最先进成绩；在 dLLM 设定下亦获得 15–25% 的生成 token 量下降。

**⚠️ 局限性**

局限性在于需访问训练集答案以确定 Imitation‑Bottleneck，且对阈值 τ 的选择与模型规模、教师差异等仍需经验调优，未充分探讨跨任务迁移与大规模数据下的可扩展性。

---

## 324. LatentRefusal: Latent-Signal Refusal for Unanswerable Text-to-SQL Queries

**arXiv ID:** 2601.10398 | [PDF](https://arxiv.org/pdf/2601.10398v1)

**作者:** Xuancheng Ren `[一作]` (Fudan University), Qiang Duan `[通讯]` (Pennsylvania State University)

**通讯引用:** 2981 | [OpenAlex ID](https://openalex.org/A5001056903)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在Text-to-SQL系统中通过分析冻结LLM中间隐藏层来判断查询可答性并在生成前拒绝不答复的机制。

**💡 创新点**

创新点在于将拒绝判定转化为“latent-signal”预生成门控，并引入Tri‑Residual Gated Encoder（TRGE）以抑制模式噪声并放大匹配缺失的稀疏信号。

**🔧 技术方法**

使用LLM内部隐藏状态、SwiGLU门控的轻量级Transformer探针和简单的Sigmoid二分类器进行训练。

**📊 数据集**

在四个基准（TriageSQL、AMBROSIA、SQuAD 2.0、MD‑Enterprise）上进行评估。

**📈 对比分析**

与基于输出的置信度估计、无监督谱方法以及提示式拒绝等对手比，平均F1提升至约88.5%，并且在单前向推理下仅增加约2 ms。

**⚠️ 局限性**

局限在于需要针对不同领域进行少量微调，泛化到全新域仍受限。

---

## 325. Codebook Design for Limited Feedback in Near-Field XL-MIMO Systems

**arXiv ID:** 2601.10391 | [PDF](https://arxiv.org/pdf/2601.10391v1)

**作者:** Liujia Yao `[一作]` (Southern University of Science and Technology), Xiaoyang Li `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 4952 | [OpenAlex ID](https://openalex.org/A5100606922)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种针对近场极大规模MIMO（XL‑MIMO）频分双工系统的有限反馈码本设计，主要解决传统极域码本忽略用户分布导致的冗余反馈问题。

**💡 创新点**

创新点在于：① 针对均匀用户分布给出最优角度均匀采样与几何距离采样的闭式解；② 利用Voronoi分区与下界分析推广到非均匀分布的交替采样方法；③ 在第1阶段采用极域几何码本，在第2阶段采用RVQ码本实现数字波束成形；④ 对反馈比特分配给出理论与实证指导，揭示随着阵列增大更倾向于分配给距离采样。

**🔧 技术方法**

主要技术包括：近场波前建模、角度与距离采样误差分析、Voronoi分割、几何下界求解、极域码本构造、随机向量量化（RVQ）、零均衡（ZF）数字波束成形、Monte‑Carlo仿真与理论逼近。

**📊 数据集**

实验使用仿真数据：频率30 GHz、BS阵列387个天线、4个单天线用户，用户角度范围[-0.5,0.5]，距离范围[4 m,120 m]，考虑LoS+NLoS路径，SNR 22 dB，随机生成用户位置与多径路径。

**📈 对比分析**

与基准码本（超平面、均匀、DFT、混合域）和完整CSI对比，几何码本在相同反馈位数下实现更高的期望叠加速率，逼近完整CSI性能，且对非均匀分布、不同距离、不同天线规模均表现出更强鲁棒性。

**⚠️ 局限性**

主要局限包括：① 主要针对LoS占优场景，NLoS、多径通道的反馈分配尚未优化；② 需要已知或估计用户位置分布，分布误差会影响码本性能；③ 虽提出闭式方案，但在极大比特预算下仍需离散搜索；④ 对于高误码率上行反馈与信道估计误差的鲁棒性未充分讨论。

---

## 326. The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

**arXiv ID:** 2601.10387 | [PDF](https://arxiv.org/pdf/2601.10387v1)

**作者:** Christina Lu `[一作]` (University of Oxford), Jack Lindsey `[通讯]` (Anthropic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型在角色扮演（persona）空间中的结构，发现一个主轴“助手轴”（Assistant Axis），并用它来解释模型在对话中从默认助手身份漂移（persona drift）的机制；提出一种在推理时对该轴进行激活截断（activation capping）的技术来抑制有害行为。

**💡 创新点**

①将275个角色与240个特质映射到激活空间，构建低维“persona space”；②发现并验证了一个代表助手身份的线性方向——助手轴；③提出利用该轴进行推理时的激活截断，实验证明能显著减少因漂移产生的有害输出。

**🔧 技术方法**

角色向量抽取（基于系统提示与提问的响应激活取平均），主成分分析（PCA）构建 persona 空间，线性 steering 与截断技术（在中间层激活中加/减向量或截断），岭回归分析用户信息对助手轴投影的影响，使用 LLM 判定器进行评估。

**📊 数据集**

• 275 个角色列表与 240 个特质列表；• 每个角色/特质 1200 次回应；• 912k 次用于构建 persona 空间的对话样本；• 100 轮对话模拟（4 个主题）用于检测漂移；• 1100 条 persona‑based jailbreak 组合；• IFEval、MMLU Pro、GSM8k、EQ‑Bench 四个基准测试；• 开源 LLM（Gemma 2 27B、Qwen 3 32B、Llama 3.3 70B）。

**📈 对比分析**

与无截断基线相比，激活截断在 Llama 3.3 70B 和 Qwen 3 32B 上可将 persona‑based jailbreak 的有害率降低约 60%，同时在 IFEval、MMLU Pro、GSM8k、EQ‑Bench 等任务上的性能下降不到 5%（部分设置甚至略有提升）。

**⚠️ 局限性**

• 仅评估了开放权重模型，未覆盖前沿（如 GPT‑4）或混合专家模型；• 角色与特质的选择不完整，可能漏掉部分人格维度；• 模拟用户对话可能不完全代表真实人类交互；• 假设助手身份可由线性方向完全表征，忽略非线性或权重层面的信息；• 评估多依赖 LLM 判定，存在判定主观性。

---

## 327. A Hybrid Reliability--Weight Framework for Construction of Polar Codes

**arXiv ID:** 2601.10376 | [PDF](https://arxiv.org/pdf/2601.10376v1)

**作者:** Mohammad Rowshan `[一作]` (University of New South Wales), Vlad-Florin Dragoi `[通讯]` (Aurel Vlaicu University)

**通讯引用:** 362 | [OpenAlex ID](https://openalex.org/A5005677757)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种混合可靠度–权重的极化码构造方法，结合可靠度排序与最小权重码字的代数贡献。

**💡 创新点**

创新点在于利用极化码作为递减多项式码的代数结构，闭式计数最小码字乘数并将其与Bhattacharyya参数加权，形成针对高阶多项式的距离惩罚，随后与传统可靠度指标融合。

**🔧 技术方法**

技术包括极化码理论、递减多项式码与下三角仿射群的轨道枚举、Bhattacharyya近似、Gaussian近似、SCL解码误差事件分解与截断ML上界分析。

**📊 数据集**

使用的实验数据集为BPSK-AWGN信道，采用不同长度和码率的（N,K）对进行数值评估。

**📈 对比分析**

与传统仅基于可靠度的极化码相比，混合构造在短到中等长度时显著降低了最小距离乘数（可降低数倍至数十万倍），并在相同可靠度成本下提升了截断ML上界，SCL误码率得到可观改善。

**⚠️ 局限性**

局限性是该方法的优势主要体现在有限长度；随着码长趋于无穷，混合构造仅对SC和ML上界产生可忽略的局部扰动，提升幅度随N增长而趋于平稳。

---

## 328. Generalized Weight Structure of Polar Codes: Selected Template Polynomials

**arXiv ID:** 2601.10362 | [PDF](https://arxiv.org/pdf/2601.10362v1)

**作者:** Mohammad Rowshan `[一作]` (University of New South Wales), Vlad-Florin Dragoi `[通讯]` (Aurel Vlaicu University)

**通讯引用:** 362 | [OpenAlex ID](https://openalex.org/A5005677757)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一套统一的代数框架，用低三角仿射群（LTA）和二进分解方法对极化码（以及其他下降多项式码）中代码字的汉明权重进行解析和计数，给出了多尺度模板的闭式表达式与精确计数公式。

**💡 创新点**

创新点在于：①将极化码视为下降多项式码并引入整体LTA群的作用，②通过包含排除式导出多项式权重的一般公式，③将权重归一化为 dyadic 形式并识别出“模板”模式（离散求和、嵌套块、互补翻转等），④利用这些模板与 LTA 轨道结合，得到最低与中间权重代码字的精确乘数。

**🔧 技术方法**

主要技术包括：多项式与布尔函数的等价表示，下降多项式集合与偏序的代数结构，低三角仿射群作用与轨道计数，包含排除原理与幂集交叉分析，及二进分解的归一化权重表示。

**📊 数据集**

无实验数据集；本文完全基于理论推导与符号计算，未进行实验验证。

**📈 对比分析**

与现有方法相比，本文提供了最小距离及 [1.5,2) 区间内权重的完整枚举，闭式计数公式可直接计算任意权重阶的代码字乘数，理论上优于仅给出极小值或近似计数的传统结果；其泛化性允许适用于所有下降多项式码（包括 Reed–Muller 和 5G 极化码）。

**⚠️ 局限性**

局限性包括：①分析复杂度随权重阶提升而急剧增加；②对高权重代码字的闭式计数尚未给出，可能需要进一步分块或数值方法；③依赖于 LTA 群的完整轨道枚举，若代码结构更复杂或存在额外自同构时，需要额外处理。

---

## 329. EvoMorph: Counterfactual Explanations for Continuous Time-Series Extrinsic Regression Applied to Photoplethysmography

**arXiv ID:** 2601.10356 | [PDF](https://arxiv.org/pdf/2601.10356v1)

**作者:** Mesut Ceylan `[一作]` (Centre for Digital Health Interventions ETH Zurich), Filipe Barata `[通讯]` (Centre for Digital Health Interventions University of St.Gallen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了EvoMorph，一种针对连续时间序列外推回归的形态感知对抗性解释生成框架；

**💡 创新点**

创新点在于结合可解释的PPG形态描述子、形态保持的编辑算子以及多目标进化优化，能够生成多样且生理合理的counterfactual；

**🔧 技术方法**

使用NSGA‑III进化算法、SSA去噪、可解释形态属性（振幅、占据频率、平台长度、趋势、最大梯度）以及光滑混合交叉/变异算子；

**📊 数据集**

在三大PPG数据集（BIDMCHR、BIDMCRR、BIDMCSpO2）上训练基于Inception的回归器；

**📈 对比分析**

与基线最近异邻（NUN）对比，EvoMorph在有效性、形态相似度、频域稀疏性方面均表现优越，且能够产生数百个多样的counterfactual；但在最大梯度等平滑度指标上略逊；

**⚠️ 局限性**

限制包括对形态描述子选择的依赖、进化搜索的计算开销、对低密度区域的敏感性以及不提供完全校准的置信度估计。

---

## 330. Unlocking Implicit Experience: Synthesizing Tool-Use Trajectories from Text

**arXiv ID:** 2601.10355 | [PDF](https://arxiv.org/pdf/2601.10355v1)

**作者:** Zhihao Xu `[一作]` (Renmin University of China), Xiting Wang `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于文本语料的多轮工具使用轨迹生成范式GEM，并训练了Trajectory Synthesizer实现端到端的轨迹生成。

**💡 创新点**

创新点在于不依赖预定义工具集合，而直接从大规模自然语言文本中提取工作流程、工具定义和完整的对话轨迹，构建真实多轮交互数据；通过四阶段管道（过滤、抽取、生成、精炼）和LLM校验实现高质量生成。

**🔧 技术方法**

技术主要包括：大语言模型（如GPT-4.1、Qwen3-8B/32B）用于文本筛选、工作流程与工具抽取、轨迹生成；监督微调训练Trajectory Synthesizer；规则+LLM双重校验确保结构和事实准确；多轮对话生成与工具调用逻辑。

**📊 数据集**

使用 Ultra‑FineWeb 文本语料库作为主要语料；补充使用 WikiHow 进行跨语料验证；生成的数据用于对 LLaMA-Base、Qwen3 系列模型进行微调。

**📈 对比分析**

在 BFCL V3 多轮工具调用基准上，GEM‑32B 提升整体准确率至 44.88%，相较于基线 Qwen3‑32B 的 29.50% 提升 15.38%；在 τ²‑Bench（Airline/Retails）上，GEM‑32B 在 Out‑of‑Domain 任务中实现 86.84% Pass@4，接近或超过使用 In‑Domain 合成数据的模型。

**⚠️ 局限性**

局限性包括：对原始文本的抽取仍需强大的 LLM 支持，生成成本相对较高；精炼与校验步骤消耗额外计算资源；生成的工具定义可能在细粒度功能和边界上不够完善；目前尚未充分验证在极其复杂、非结构化真实对话中的鲁棒性。

---

## 331. A New Construction Structure on MISO Coded Caching with Linear Subpacketization: Half-Sum Disjoint Packing

**arXiv ID:** 2601.10353 | [PDF](https://arxiv.org/pdf/2601.10353v1)

**作者:** Bowen Zheng `[一作]` (Guangxi Normal University), Giuseppe Caire `[通讯]` (Technische Universität Berlin)

**通讯引用:** 28788 | [OpenAlex ID](https://openalex.org/A5058252389)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种新型的L-半和不相交打包（L‑HSDP）结构，并通过该结构构造了子分割数线性（F＝K）的多天线编码缓存方案。

**💡 创新点**

创新点在于将编码缓存与多天线系统统一到单一的组合结构（HSDP）中，显著降低子分割数，同时仅略微牺牲总DoF。

**🔧 技术方法**

采用组合数学（Latin方阵、半和不相交打包）、高维几何编码与凸优化（拉格朗日乘子法）等技术实现方案构造与参数优化。

**📊 数据集**

论文未使用具体实验数据集，而是通过理论证明与数值仿真（例如K=567,L=4或K=85,L=2）进行性能对比。

**📈 对比分析**

与现有指数子分割或线性子分割方案相比，所提出方案在子分割数上可下降到线性级别，且总DoF保持或略低于最优值，数值结果显示在相同缓存率下更优。

**⚠️ 局限性**

局限性包括仅针对L≤2^r且g=2^n+r的特殊HSDP构造，且在某些参数范围内仍无法达到最优总DoF；未来需要探索更一般的HSDP构造与更广泛的系统参数。

---

## 332. Convertible Codes for Data and Device Heterogeneity

**arXiv ID:** 2601.10341 | [PDF](https://arxiv.org/pdf/2601.10341v1)

**作者:** Anina Gruica `[一作]` (Technical University of Denmark), Stanislav Kruglik `[通讯]` (Technical University of Denmark)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了可转换码在分布式存储系统中的应用，提出基于Reed‑Muller码的可转换方案，并给出了读写成本的通用下界；

**💡 创新点**

首次将Reed‑Muller码与可转换码结合，既兼顾数据异构又兼顾设备异构，并提供了低成本转换实现；

**🔧 技术方法**

使用线性码理论、Plotkin构造、Reed‑Muller码性质以及矩阵转换表示、Singleton、短化/截断等经典界；

**📊 数据集**

未使用实验数据集，本文完全基于理论推导；

**📈 对比分析**

通过与已有可转换码的读写成本下界对比，写入成本实现了下界最优，而读写成本在理论上可行但未完全达到下界；总体性能优于传统方案；

**⚠️ 局限性**

读成本下界相对粗糙，未充分利用Reed‑Muller码结构；仅讨论线性转换，未考虑带宽与非线性转换；缺乏实验验证。

---

## 333. Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders

**arXiv ID:** 2601.10332 | [PDF](https://arxiv.org/pdf/2601.10332v1)

**作者:** Siqi Kou `[一作]` (Shanghai Jiao Tong University), Zhijie Deng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 776 | [OpenAlex ID](https://openalex.org/A5102623510)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了思考-生成 (T2G) 框架，先让大语言模型（LLM）对原始文本提示进行链式推理和重写，然后将重写后的提示作为条件引导扩散模型生成图像；通过轻量级监督微调激活LLM的推理模式，再通过Dual‑GRPO对LLM和扩散解码器进行联合强化学习，以实现语义一致且视觉真实的图像合成。

**💡 创新点**

①将LLM从被动编码器转化为主动推理器；②使用监督微调数据让LLM学会“先推理再写提示”；③采用Dual‑GRPO在多阶段生成中同步优化推理与生成，首次实现推理与扩散解码器的端到端协同训练；④通过图像基准奖励激励LLM生成可落地的视觉语义。

**🔧 技术方法**

监督微调（SFT）+链式推理 (CoT) + 思考-重写机制；Dual‑GRPO（基于GRPO的策略梯度，结合Flow‑GRPO为扩散模型提供随机性）；奖励设计：对LLM使用语义一致性奖励，对DiT使用美学、物理一致性与语义一致性组合奖励；图像生成使用流匹配（DiT）与SDE窗口；实验中采用了Flow‑GRPO‑fast 等训练脚本。

**📊 数据集**

①构造约7K条需要世界知识推理的原始提示，使用 Gemini‑2.5 生成 CoT 与重写提示做监督微调；②针对图像编辑使用 UniREdit‑Data‑100K 并同样生成 CoT/重写提示；③评测数据集包括 WISE、T2I‑ReasonBench、UniREditBench、RISEBench 等公开基准。

**📈 对比分析**

在 WISE（文化、时空、科学等 25 个子域）上，T2G 模型得分 0.79（相较基线提升 30%），在 T2I‑ReasonBench 上 0.92（超过 Gemini‑2.0，接近 GPT‑4o）。在图像编辑任务中，UniREditBench 上 68.7 分、RISEBench 上 23.9 分，均优于同类开源模型并逼近专有系统。与传统的冻结编码器 + 只优化解码器的流匹配或 Diffusion‑SDPO 等方法相比，T2G 通过联合训练显著提升了概念理解、语义一致性与视觉质量。

**⚠️ 局限性**

①需要额外的监督数据与复杂的 RL 训练，计算成本高；②奖励设计仍可能导致 LLM 在推理过程中产生误导性或无关信息，影响最终图像质量；③对极端概念或极少见提示的泛化能力尚未充分验证；④目前主要评估在人工构造的基准上，实际应用中的鲁棒性与多模态交互仍待进一步探索。

---

## 334. Enhancing the quality of gauge images captured in smoke and haze scenes through deep learning

**arXiv ID:** 2601.10537 | [PDF](https://arxiv.org/pdf/2601.10537v1)

**作者:** Oscar H. Ramírez-Agudelo `[一作]` (German Aerospace Center), Kai Franke `[通讯]` (German Aerospace Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文利用深度学习模型对受雾和烟雾影响的模拟模拟仪表图像进行去雾/去烟处理，以提升仪表读数的可读性；

**💡 创新点**

创新点在于：①针对模拟雾/烟雾仪表图像构建了约1.4万张合成数据集；②在此数据集上训练并评估了两种先进的深度去雾网络（FFA‑Net 与 AECR‑Net），并与传统优化方法 BCCR 进行对比；

**🔧 技术方法**

使用技术包括：Unreal Engine 合成数据、FFA‑Net（多级特征融合注意力网络）、AECR‑Net（自编码器+对比学习正则化）、BCCR（基于边界约束与上下文正则化的优化去雾），以及评估指标 SSIM 与 PSNR；

**📊 数据集**

数据集：Synthetic Haze 数据集（约4.8k 清晰+4.8k 雾化图像）和 Synthetic Smoke 数据集（约9.6k 清晰+9.6k 烟雾图像），均通过 Unreal Engine 生成；

**📈 对比分析**

方法比较：在雾数据集上 AECR‑Net 的最大 PSNR 约 44 dB、SSIM 0.98；FFA‑Net 约 30.5 dB、0.96；BCCR 约 12 dB、0.65。烟数据集上 AECR‑Net 约 37 dB、0.96；FFA‑Net 约 26 dB、0.94；BCCR 约 9 dB、0.55。结果表明 AECR‑Net 在两种情况均优于 FFA‑Net 与传统 BCCR；

**⚠️ 局限性**

局限性：①模型主要针对合成雾/烟数据，真实场景下性能尚未充分验证；②AECR‑Net 对高密度烟雾的去除仍存在残留与伪影；③未针对去烟专门设计的网络（如 DesmokeNet），导致烟雾去除效果不如雾；④数据集规模虽大，但仅为模拟，缺乏多种真相环境与不同仪表类型的多样性。

---

## 335. SVII-3D: Advancing Roadside Infrastructure Inventory with Decimeter-level 3D Localization and Comprehension from Sparse Street Imagery

**arXiv ID:** 2601.10535 | [PDF](https://arxiv.org/pdf/2601.10535v1)

**作者:** Chong Liu `[一作]` (State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing), Bisheng Yang `[通讯]` (State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了SVII-3D框架，实现从低成本稀疏街景图像自动生成高精度数字孪生与资产清单，涵盖检测、跨视图匹配、几何定位和细粒度状态识别。

**💡 创新点**

创新点包括：① 用LoRA轻量化微调开源集检测模型；② 设计空间注意力Transformer实现跨视图高鲁棒匹配；③ 引入几何导向的三维定位与匹配细化机制；④ 构建基于VLM的状态辨识代理，融合多模态提示、专家规则与检索增强，实现细粒度运维状态推断。

**🔧 技术方法**

技术细节：Grounding DINO + LoRA；空间注意力Transformer；光线三角化与BFGS能量优化；几何引导匹配细化；VLM（Qwen‑VL/GLM‑4v/LLaVA）+ 多模态提示 + 专家知识注入 + Retrieval‑Augmented Generation。

**📊 数据集**

使用武汉和上海两地城市道路稀疏街景数据集，包含约1.7k/1.3k训练图像与229/290评测连续图像，利用同步LiDAR点云生成3D中心作为高精度标注。

**📈 对比分析**

与零射击基础模型（原Grounding DINO）和仅局部关联链基线对比。检测AP@0.5从约25%提升至73%/77%；3D定位平均误差0.12/0.137 m，识别F1>0.84；跨城评测V‑Measure>0.98，证明算法具备很强的跨域泛化与结构纠错能力。

**⚠️ 局限性**

局限性包括：平面低视角或单帧可见的物体（如井盖、交通锥）召回率仍偏低；稀疏图像导致单视角信息不足；VLM状态识别仍受模型推断误差和知识库覆盖范围限制。

---

## 336. Learning from Brain Topography: A Hierarchical Local-Global Graph-Transformer Network for EEG Emotion Recognition

**arXiv ID:** 2601.10525 | [PDF](https://arxiv.org/pdf/2601.10525v1)

**作者:** Yijin Zhou `[一作]` (Xidian University), Lijian Zhang `[通讯]` (Beijing Institute of Mechanical Equipment)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了神经解剖学驱动的层次化图-Transformer网络，用于 EEG 情感识别

**💡 创新点**

结合空间欧氏先验图与可学习全局动态图，并通过局部并行 GCN 与 iTransformer 实现局部‑全局双流学习

**🔧 技术方法**

采用图卷积网络、注意力机制、iTransformer 以及 KL / 正交约束等技术

**📊 数据集**

在 SEED、SEED‑IV、SEED‑V 和 MPED 四个公开数据集上评估

**📈 对比分析**

与 SVM、DGCNN、PGCN、RGNN、TANN 等基线对比，取得 state‑of‑the‑art 准确率，尤其在细粒度情绪分类和跨受试者泛化上显著提升

**⚠️ 局限性**

仍受限于数据规模、模型复杂度以及对极端情绪区分的细节仍需进一步改进

---

## 337. BikeActions: An Open Platform and Benchmark for Cyclist-Centric VRU Action Recognition

**arXiv ID:** 2601.10521 | [PDF](https://arxiv.org/pdf/2601.10521v1)

**作者:** Max A. Buettner `[一作]` (Munich University of Applied Sciences), Fabian B. Flohr `[通讯]` (Munich University of Applied Sciences)

**通讯引用:** 1282 | [OpenAlex ID](https://openalex.org/A5007686963)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FUSE-Bike自行车级感知平台，并基于该平台收集了BikeActions数据集，用于从骑行者视角识别VRU动作；

**💡 创新点**

首次提供高精度、同步的多模态（LiDAR、摄像机、GNSS）自行车级数据采集平台及其开源实现，并构建了专门针对细粒度VRU动作的标注数据集；

**🔧 技术方法**

采用多模态同步采集、硬件时间同步（PTP）和外参标定技术；使用骨架动作识别模型（GCN、Transformer等）进行基准测试；

**📊 数据集**

BikeActions数据集，包含46,180帧同步数据，5类动作（步行、站立、骑行、左手示意、右手示意），共12条序列约1.3小时；

**📈 对比分析**

在该数据集上对10种骨架模型进行评估，Hyperformer在骨架和骨骼模态下分别达到94.62%和96.15%的最高准确率；

**⚠️ 局限性**

数据样本主要为短小动作，稀缺动作如跑步缺失，导致长尾问题；仅评估骨架模型，未涉及视频或姿态估计的直接应用。

---

## 338. Stable Differentiable Modal Synthesis for Learning Nonlinear Dynamics

**arXiv ID:** 2601.10453 | [PDF](https://arxiv.org/pdf/2601.10453v1)

**作者:** Victor Zheleznov `[一作]` (University of Edinburgh), Simon King `[通讯]` (University of Edinburgh)

**通讯引用:** 12429 | [OpenAlex ID](https://openalex.org/A5062516688)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种可微分模态合成模型，能够从数据学习非线性横向字符串振动的动力学，并在训练后可自由修改物理参数和采样率。

**💡 创新点**

创新点在于将标量辅助变量（SAV）技术与神经ODE相结合，使用梯度网络（GradNet）近似非线性耦合，保证数值稳定性且使物理参数保持可访问，提升了模型的可解释性和泛化能力。

**🔧 技术方法**

采用的技术包括模态分解、SAV数值求解、梯度网络、神经ODE、PyTorch实现、教师强迫、Sherman‑Morrison公式、Leaky ReLU、Adam优化器等。

**📊 数据集**

使用合成数据集：从非线性字符串振动方程生成的 60 条训练、20 条验证、60 条测试轨迹，涵盖 75 个模态、频率 61.74–123.47 Hz，采样率 88.2 kHz/96 kHz，随机化振幅、张力、刚度等参数。

**📈 对比分析**

通过与线性模型的相对 MSE 进行比较。训练集 100 ms 误差约 2.8e‑4，完整时间 5.4e‑2；验证和测试集误差与训练集相近，证明模型在未见参数、采样率和时间尺度下具有良好泛化性能。

**⚠️ 局限性**

局限性：对高模态误差积累敏感，需较长时间步导致误差放大；仅在合成数据上验证，未对真实录音进行训练或测试；训练对 GPU 资源依赖较大；若无位移/速度观测，需要改进损失函数以适应仅音频输入的情形。

---

## 339. Lunar-G2R: Geometry-to-Reflectance Learning for High-Fidelity Lunar BRDF Estimation

**arXiv ID:** 2601.10449 | [PDF](https://arxiv.org/pdf/2601.10449v1)

**作者:** Clementine Grethen `[一作]`, Manuel Sanchez Gestido `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于月球数字高程模型（DEM）的空间可变BRDF学习框架，利用U‑Net和可微渲染直接预测每个像素的BRDF参数。

**💡 创新点**

创新点在于首次从地形几何推断空间可变反射率，并通过可微渲染与真实轨道图像配对实现端到端训练，无需多视角或控制照明。

**🔧 技术方法**

技术手段包括低阶余弦多项式BRDF参数化、U‑Net卷积网络、SurRender可微渲染以及基于MSE的光度误差损失。

**📊 数据集**

使用的训练数据来自Tycho陨石坑的高分辨率DEM与LRO相机图像共计83,614对（66,662训练 / 8,615验证 / 8,337测试），每对包含相应的相机姿态和光照信息。

**📈 对比分析**

通过与归一化Hapke模型在MSE、PSNR、SSIM和LPIPS四个指标上的对比，学习得到的SVBRDF使MSE下降38%，PSNR/SSIM提升，LPIPS降低，显示更高的光度精度与结构相似度。

**⚠️ 局限性**

局限性包括训练样本主要为近视差视角、仅覆盖Tycho区域、DEM中存在几何误差，导致对不同观测角度和更广泛地形类型的泛化能力有限。

---

## 340. Unleashing the Capabilities of Large Vision-Language Models for Intelligent Perception of Roadside Infrastructure

**arXiv ID:** 2601.10551 | [PDF](https://arxiv.org/pdf/2601.10551v1)

**作者:** Luxuan Fu `[一作]` (State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing), Zhen Dong `[通讯]` (Hubei Luojia Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种域适应框架，将大型视觉语言模型（Grounding DINO、Qwen‑VL）转化为智能道路基础设施分析的专业代理，实现从检测到属性推理的全流程。

**💡 创新点**

创新点包括：① 开放词汇检测微调实现对未知设施的零射击识别；② LoRA 细化 Qwen‑VL 的属性推理；③ 双模检索增强生成（文本标准+视觉样本）实现结构化 JSON 输出，显著提升专业性和可执行性。

**🔧 技术方法**

使用技术包括：Grounding DINO 开放词汇检测、Qwen‑VL LoRA 微调、双模 Retrieval‑Augmented Generation（文本检索与视觉检索）、结构化 JSON schema、跨模对话推理。

**📊 数据集**

数据集：自行构建的 3551 张全景道路图（上海 1,576 张、武汉 1,975 张），包含 10 类设施（信号灯、标志、灯杆、摄像头等）共计 100,000+ 实例及细粒度属性标注。

**📈 对比分析**

与 GLIP、YOLO‑World、OV‑DINO、MM‑GD 等基线对比，零射击下 mAP 18.0，微调后 mAP 58.9；属性识别准确率 95.5%；跨城泛化 mAP 0.589→0.406，显示显著性能提升。

**⚠️ 局限性**

局限性：仍对小型物体（摄像头、灯杆）检测不稳，跨城差异导致泛化下降；依赖外部知识库与视觉检索，更新成本较高；目前仅支持静态图像，缺乏视频时序理解。

---

## 341. PERM: Psychology-grounded Empathetic Reward Modeling for Large Language Models

**arXiv ID:** 2601.10532 | [PDF](https://arxiv.org/pdf/2601.10532v1)

**作者:** Chengbing Wang `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8612 | [OpenAlex ID](https://openalex.org/A5051925942)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了心理学导向的双向同理心奖励模型 PERM，并通过强化学习提升 LLM 在情感支持对话中的同理心表现

**💡 创新点**

创新点在于将同理心拆解为支持者视角（共振与表达）、寻求者视角（接受度）以及旁观者视角（对话质量）三方面进行评估，并将其融合为调和平均奖励；此外引入旁观者惩罚以避免“表面同理”现象

**🔧 技术方法**

技术上使用了 LLM 作为判别器（基于特定 rubrics 的5分量表），奖励函数为 r_emp = 3 / (1/r_res + 1/r_exp + 1/r_rec)，并加入旁观者惩罚；采用 Group Relative Policy Optimization（GRPO）进行 RL 训练

**📊 数据集**

数据集主要为 EmpatheticDialogues（通过 GPT‑4o 扩充上下文与 persona）以及 EQ‑Bench3 评测基准；实验还包括工业日常对话数据集和用户研究对比

**📈 对比分析**

与 SFT、RLVER、Partner、RM 等基线相比，PERM 在 EQ‑Bench3、日常对话评测以及用户研究中均取得 10%~15% 的整体情感智力提升，且在用户偏好实验中获得约 70% 的选择率，显示明显优于现有方法

**⚠️ 局限性**

局限性包括：训练仅使用单轮对话，未覆盖多轮情绪展开；缺乏用户记忆与偏好建模；模型仍可能放大训练数据中的偏见；需进一步探索多轮强化学习与个性化记忆机制

---

## 342. On the suboptimality of linear codes for binary distributed hypothesis testing

**arXiv ID:** 2601.10526 | [PDF](https://arxiv.org/pdf/2601.10526v1)

**作者:** Adway Girish `[一作]` (École Polytechnique Fédérale de Lausanne), Emre Telatar `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 14254 | [OpenAlex ID](https://openalex.org/A5025506109)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文研究了在两传感器分布式假设检验中使用线性压缩方案的性能，特别关注二元DSBS（双对称二进制源）在两种假设下的相关性。作者证明在（p0 与 p1 的相反符号或其中一方为独立）情况下，最优的线性方案是截断（即仅保留前k个样本），并提出对一般相反符号情况的猜想。并通过与Han随机量化方案的 Stein 指数比较，展示线性压缩在这些场景下严格次优。

**💡 创新点**

创新点包括：
1) 通过 Blackwell 序的判别，证明在两种假设下使用相同的线性码且仅保留和的截断即可得到最优的统计量；
2) 对 p0 与 p1 处于 1/2 的两侧的三类特殊情形（相同幅度相反相关性、对独立性进行检验）给出了完整证明；
3) 给出一个可计算的线性可行性判定方法，并用数值实验支持更广泛的猜想；
4) 对比随机量化（Han 方案）得到的 Stein 指数，说明线性方案在任何率下均不如最优随机量化。

**🔧 技术方法**

主要技术手段包括：
- Blackwell 的判别与通道退化理论，用于比较不同压缩后的统计量；
- 线性码的矩阵表述与截断构造；
- Stein 指数的 KL 散度表达式与随机量化的 I‑投影求解；
- 数值线性规划，用于检验截断与任意线性码的可行性关系；
- 对称性与置换不变性在 DSBS 上的利用。

**📊 数据集**

本文没有使用真实数据集，而是以理想化的二元 DSBS（p0 与 p1 参数化的独立/相关性分布）作为实验模型。所有实验均在离散的 p0–p1 网格上进行，计算线性可行性与随机量化指数。

**📈 对比分析**

比较方法：
- 对于给定率 R，计算截断方案的 Stein 指数为 R·p0p1；
- 计算 Han 随机量化方案（BSC 试验通道）得到的指数并取上凸包，得到上界；
- 对特定 (p0,p1) = (0.1,0.9) 的数值模拟，绘制指数‑率曲线并比较；
- 结论是：在所有率下，随机量化方案的 Stein 指数均高于截断（线性）方案，且在大率时远超；截断在低率下与随机方案相近，但仍略逊。

**⚠️ 局限性**

限制与未解决问题：
- 对一般 (p0,p1) 处于 1/2 两侧的情况，截断是否始终是最优线性码仅是猜想，尚未给出完整证明；
- 只考虑了二元 DSBS，结果是否可推广到多元或连续情形尚不明；
- 只分析了 Stein 指数（即极限错误指数），对非极限或具体错误概率的评估缺失；
- 线性码与随机码的比较仅在对称率约束下进行，非对称情形未探讨。

---

## 343. Scalable Algorithms for Approximate DNF Model Counting

**arXiv ID:** 2601.10511 | [PDF](https://arxiv.org/pdf/2601.10511v1)

**作者:** Paul Burkhardt `[一作]` (National Security Agency), Kevin T Schmitt `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种新的基于自适应停止规则和短路评估的蒙特卡罗算法，用于近似计数 DNF 公式的满足赋值数。

**💡 创新点**

创新点在于：①引入固定的子句排列（可混合启发式和随机），实现短路评估；②采用自适应停止阈值，保证 PAC 误差；③在理论和实验上均证明相较传统 KLM 与 Pepin 的工作量和随机数使用更少。

**🔧 技术方法**

技术包括：蒙特卡罗抽样、短路子句评估、启发式子句排列（宽度优先+随机混合 β=0.01）、概率分析与 Chernoff 误差界、低层实现优化（稀疏/稠密位数组、内存局部性）。

**📊 数据集**

使用自定义合成 DNF 数据集（基于随机子树/stem 结构）和公开的 15,000 变量 DNF 作为基准；对比不同宽度、误差 ε、δ 的实验。

**📈 对比分析**

与 KLM、L‑KLM、Pepin、Neural#DNF 进行对比：在同一误差/δ 下，算法在 10^4–10^6 变量时实现 95%+ 精度，运行时间比 KLM/ Pepin 低 1–2 个数量级；对 Neural#DNF 的速度几乎相当但保留 FPRAS 的正确性保证。

**⚠️ 局限性**

局限性：仍需对极端宽度或高度重叠子句的性能进行理论分析；对极小 ε、δ 的实验有限；实现依赖于对内存访问模式的优化，可能在不同硬件上表现不同。

---

## 344. A New Construction Structure on Multi-access Coded Caching with Linear Subpacketization: Cyclic Multi-Access Non-Half-Sum Disjoint Packing

**arXiv ID:** 2601.10510 | [PDF](https://arxiv.org/pdf/2601.10510v1)

**作者:** Mengyuan Li `[一作]` (Guangxi Normal University), Giuseppe Caire `[通讯]` (Technische Universitaet Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种新型组合结构 CMA‑NHSDP，构造多接入编码缓存方案

**💡 创新点**

CMA‑NHSDP 将非半和分离打包（NHSDP）推广至多接入网络，并满足 L 连续性，实现线性子分块同时保持低传输负载

**🔧 技术方法**

利用组合数学、非半和分离打包、整数规划、递归偏移等技术构造 P‑DA，并从中得到多接入编码缓存方案

**📊 数据集**

未使用真实数据集，全部为理论构造与数值实验对比

**📈 对比分析**

通过与 RK1、CW、WCWL、SR2 等现有方案的理论比较和数值曲线对比，证明在相同或更低子分块下，传输负载更低，且在部分参数区间内子分块显著降低

**⚠️ 局限性**

方案在 K 为偶数时需引入虚拟用户；对系统参数敏感，仍需进一步优化在某些参数下的传输负载；理论证明复杂度高，实际实现需要更多评估

---

## 345. mergetune: Continued fine-tuning of vision-language models

**arXiv ID:** 2601.10497 | [PDF](https://arxiv.org/pdf/2601.10497v1)

**作者:** Wenqing Wang `[一作]` (University of Surrey), Josef Kittler `[通讯]` (University of Surrey)

**通讯引用:** 53423 | [OpenAlex ID](https://openalex.org/A5028209738)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种“继续微调”(Continued Fine‑Tuning, CFT) 方法，在已有的 VLM（如 CLIP）经零射击后再经过下游任务微调后，通过进一步微调（不改动模型结构）恢复其预训练知识。

**💡 创新点**

核心创新在于使用线性模式连通性 (Linear Mode Connectivity, LMC) 作为学习目标，将零射击与下游微调模型在权重空间中找到一条低损失的线性路径，从而实现知识融合；同时采用二阶近似的无数据重放惩罚来避免需要大规模预训练数据。

**🔧 技术方法**

技术手段包括：线性模式连通性约束、二阶泰勒展开的近似正则化、对 PEFT（Prompt、Adapter、线性头）以及完整微调的统一后置策略、以及与传统模型融合/集成方法（TIES、DARE、VRF、Wise‑FT）的对比。

**📊 数据集**

在多种基准上评估：11 个少样本基准（base‑to‑novel）、十个跨数据集通用性测试、四个 ImageNet 变体域泛化、以及 ImageNet 与五个 OOD 数据集的 ID‑OOD 鲁棒性测试。

**📈 对比分析**

与训练免费模型融合方法（TIES、DARE）和基于权重/预测空间的集成方法（VRF、Wise‑FT）比较，CFT 在多数任务中显著提升了性能：CoOp 的谐波均值提升 5.6%，跨数据集平均提升约 1.9%，在 OOD 评估中以单一模型实现 SOTA，且推理成本低于传统集成。

**⚠️ 局限性**

局限性包括：对超参数（λ、β、初始化 τ）的敏感性，需要在继续微调阶段再投入计算资源；对极端任务或模型结构差异较大的情况可能恢复效果有限；二阶近似假设（梯度为 0、Hessian 为 μI）在某些场景下可能不够精准。

---

## 346. Communication-Efficient Federated Learning by Exploiting Spatio-Temporal Correlations of Gradients

**arXiv ID:** 2601.10491 | [PDF](https://arxiv.org/pdf/2601.10491v1)

**作者:** Shenlong Zheng `[一作]` (Jinan University), Lin Cui `[通讯]` (Jinan University)

**通讯引用:** 12544 | [OpenAlex ID](https://openalex.org/A5071334202)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于梯度时空相关性的压缩框架 GradESTC，显著降低联邦学习中的上行通信量，同时保持模型精度与收敛速度。

**💡 创新点**

创新点在于将梯度的空间低秩结构与跨轮的时间相关性结合：使用随机SVD提取空间基向量，并通过增量更新机制仅在必要时替换基向量，从而兼顾压缩比与收敛性能；并针对参数占比大的层进行选择性压缩。

**🔧 技术方法**

核心技术包括梯度重塑、随机SVD、低秩表示、动态基向量更新（增量替换）、基向量的贡献度评估与自适应候选数调整、以及与FedAvg的无缝集成。

**📊 数据集**

实验使用 MNIST、CIFAR‑10 与 CIFAR‑100 三个图像数据集，模型分别为 LeNet5、ResNet18 与 AlexNet，覆盖 IID 与多种非 IID 场景。

**📈 对比分析**

与 FedAvg、Top‑k、FedPAQ、SVDFed、FedQClip 等主流压缩方法比较，GradESTC 在相同目标精度下平均降低 39.8%–86.7% 的上行通信量，并且在精度和收敛速度上与无压缩 FedAvg 基本一致，甚至在某些设置下略优。

**⚠️ 局限性**

局限性包括：需要针对不同层调节 k、l、d 等超参数；增量基向量更新仍带来一定的计算开销；对下行压缩、极端非 IID 或超大模型的适应性尚未完全验证；且在极高压缩率下的梯度残差可能影响长期收敛。

---

## 347. Panning for Gold: Expanding Domain-Specific Knowledge Graphs with General Knowledge

**arXiv ID:** 2601.10485 | [PDF](https://arxiv.org/pdf/2601.10485v1)

**作者:** Runhao Zhao `[一作]` (National University of Defense Technology), Lei Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 113857 | [OpenAlex ID](https://openalex.org/A5010092165)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了“领域特定知识图谱融合（DKGF）”任务，设计了 Fact-as-Program 框架，将通用知识图谱（GKG）的事实视为可执行语义程序，并通过程序可执行性与细粒度映射来判断其对目标领域知识图谱（DKG）的相关性，从而实现高质量、语义一致的知识融合。

**💡 创新点**

创新点在于：①将知识融合转化为可执行的语义程序，利用可执行性作为判定域相关性的直接指标；②引入三层粒度（粗、中、细）可调的语义算子映射，统一解决跨域知识粒度不匹配与域相关性不确定两大挑战；③在单一概率框架内同时完成相关性评估与粒度转换，显著提升融合精度。

**🔧 技术方法**

主要技术包括：概率程序推理、基于 Transformer 的算子嵌入与映射、MLP 语义生成器、可执行性验证模块、结构一致性评估，以及轻量级实体与关系嵌入；整个模型无需调用大语言模型，仅使用小型神经网络即可完成融合。

**📊 数据集**

使用了两套新建基准数据集：ICEWS‑Wiki（从 ICEWS 与 Wikipedia 合成）和 ICEWS‑YAGO（从 ICEWS 与 YAGO 合成），每套数据集均覆盖大量跨域实体与关系，专门用于评估 DKGF 性能。

**📈 对比分析**

与 21 种基线（规则、TransE、GNN、生成式、实体对齐、知识图谱补全、关系抽取、LLM 等）进行对比实验，实验显示该方法在 ACC、F1 上均领先最强基线约 9.5% 以上，并在效率上仅耗时 42.6 秒、无 Token 消耗，体现出最佳的准确率与资源占用平衡。

**⚠️ 局限性**

局限性包括：①目前仅验证于两套静态基准，缺乏对更大规模或多模态知识图谱的可扩展性评估；②未充分利用时间维度信息，难以处理时序知识；③评价指标仍以二分类准确率为主，未涵盖融合知识的长期语义质量和应用效果。

---

## 348. SDN-Driven Innovations in MANETs and IoT: A Path to Smarter Networks

**arXiv ID:** 2601.10544 | [PDF](https://arxiv.org/pdf/2601.10544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 349. Contextual StereoSet: Stress-Testing Bias Alignment Robustness in Large Language Models

**arXiv ID:** 2601.10460 | [PDF](https://arxiv.org/pdf/2601.10460v1)

**作者:** Abhinaba Basu `[一作]` (Indian Institute of Information Technology), Pavan Chakraborty `[通讯]` (Indian Institute of Information Technology)

**通讯引用:** 1479 | [OpenAlex ID](https://openalex.org/A5023091561)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 Contextual StereoSet benchmark，系统地在保持刻板印象内容不变的前提下，变换地点、时间、语气和观察者四个维度，评估大语言模型在不同上下文中的刻板偏见表现。

**💡 创新点**

创新点在于提出 Context Sensitivity Fingerprints（CSF）——一种可压缩、可统计的偏见敏感度谱，能够捕捉模型在各上下文维度上的方差与对比效应，揭示偏见并非固定属性，而是与上下文相关的动态特性。

**🔧 技术方法**

技术包括：基于 StereoSet 的 prompt 生成、四维上下文格子（位置×年份×语气×观察者）设计、对大语言模型的批量推理、结果解码与统计（Bootstrap CI、FDR 校正）、以及可视化和报告生成。

**📊 数据集**

数据集为改写自 StereoSet 的 50 条基础条目，扩展为 360（完整格子）或 72（预算版）个上下文变体，并在 4,229 条条目上进行大规模评估；此外还提供了 2,000 条翻译成印地语和中文、以及 Swahili、Hausa、Yoruba 的低资源语言样本。

**📈 对比分析**

与传统单维度偏见评测相比，CSF 能在 13 种模型中揭示 3 大上下文效应（时间回归、观众差异、语气差异），并量化为平均差值 0.02–0.13，显示模型对特定上下文的敏感度远高于固定条件测试，表明单一偏见分数不足以评估模型安全性。

**⚠️ 局限性**

局限性包括：基于 StereoSet 的刻板印象标签缺乏真实基准，测试仅使用英语提示，无法完全反映多语言多文化环境；低资源语言的翻译与合成样本可能混合语言能力与偏见；以及无法区分偏见来源是预训练关联、对齐训练还是提示设计。

---

## 350. LangLasso: Interactive Cluster Descriptions through LLM Explanation

**arXiv ID:** 2601.10458 | [PDF](https://arxiv.org/pdf/2601.10458v1)

**作者:** Raphael Buchmüller `[一作]` (University of Konstanz), Angelos Chatzimparmpas `[通讯]` (Utrecht University)

**通讯引用:** 740 | [OpenAlex ID](https://openalex.org/A5070488104)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了LangLasso，一个利用LLM生成自然语言聚类解释的交互式工具，帮助初学者快速理解降维投影中的聚类结构。

**💡 创新点**

首次系统评估三种LLM提示策略（统计、抽样、全数据），并证明仅统计信息可提供最可靠、可验证的聚类描述；同时提出将统计精度与叙事丰富性结合的混合方法。

**🔧 技术方法**

使用LLM（GPT‑5‑mini）进行文本生成，UMAP进行二维投影，统计检验（均值、标准差、KS检验）以及提示工程技术来驱动解释输出。

**📊 数据集**

在四个结构化表格数据集上验证：Palmer Penguins、Bank Marketing、Food Nutrition、Customer Analysis。

**📈 对比分析**

通过将LLM输出与人工生成的统计结果和特征分布进行对比，发现S1（统计策略）在准确性、可重复性和可解释性上最优，S2在细节上模糊，S3因令牌限制和文本冗余效果受限；整体评估建议优先采用S1。

**⚠️ 局限性**

受限于LLM令牌容量导致全数据策略不可行；可能出现虚假或过度泛化描述；仅在四个表格数据集上测试，缺乏用户研究验证；文本输出技术性较强，对非技术用户的可读性仍有限。

---

## 351. Energy-Efficient Probabilistic Semantic Communication Over Visible Light Networks With Rate Splitting

**arXiv ID:** 2601.10452 | [PDF](https://arxiv.org/pdf/2601.10452v1)

**作者:** Zhouxiang Zhao `[一作]` (Zhejiang University), Zhaoyang Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 21523 | [OpenAlex ID](https://openalex.org/A5100751311)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在可见光通信网络中结合概率语义通信与率分割多路访问的能效优化问题。

**💡 创新点**

创新点在于首次将语义压缩与共享知识更新融入RSMA框架，并联合考虑通信与计算能耗。

**🔧 技术方法**

采用了可见光通信物理模型、概率语义通信模型、RSMA多路访问、SCA与Dinkelbach等非凸优化方法。

**📊 数据集**

实验使用基于室内灯光部署的4LED-2用户设置，参数参照公开VLC系统配置，无特定公开数据集。

**📈 对比分析**

与SDMA、NOMA、传统RSMA对比，结果显示所提PSCom‑RSMA方案在能效上提升约10%–20%。

**⚠️ 局限性**

主要局限在于只考虑LOS、静态场景，忽略NLoS、多用户移动以及真实语义质量指标。

---

## 352. Hybrid Encryption with Certified Deletion in Preprocessing Model

**arXiv ID:** 2601.10542 | [PDF](https://arxiv.org/pdf/2601.10542v1)

**作者:** Kunal Dey `[一作]` (SRM University), Reihaneh Safavi-Naini `[通讯]` (University of Calgary)

**通讯引用:** 7939 | [OpenAlex ID](https://openalex.org/A5010447902)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了预处理模型下的混合加密与可认证删除机制，构造了两种实现方案；

**💡 创新点**

其创新点在于将信息论安全的键封装与可认证删除的数据封装结合，且第二种方案实现了量子安全且关键长度与消息长度无关；

**🔧 技术方法**

主要技术包括信息论键封装机制（iKEM）、可认证删除的数据封装（DEM‑CD）、Wiesner量子编码、量子安全对称加密（如AES）以及预处理模型的混合加密框架；

**📊 数据集**

论文未使用传统机器学习或图像等数据集，而是以理论构造和安全证明为主；

**📈 对比分析**

通过安全性证明展示了相较于现有方案在保密性与可删除性上实现了信息论安全或量子安全，但未给出实验性能评估；

**⚠️ 局限性**

主要局限在于第一方案需要一时性钥匙、第二方案仍需量子通道实现Wiesner编码，且缺乏实际实验验证和效率评估。

---

## 353. Diagnosing Generalization Failures in Fine-Tuned LLMs: A Cross-Architectural Study on Phishing Detection

**arXiv ID:** 2601.10524 | [PDF](https://arxiv.org/pdf/2601.10524v1)

**作者:** Frank Bobe `[一作]`, Jose Salas-Vernis `[通讯]` (Naval Surface Warfare Center Panama City Division)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比Llama3.1 8B、Gemma 2 9B和Mistral在钓鱼邮件检测任务中的微调效果，揭示架构与数据多样性对泛化的关键作用。

**💡 创新点**

提出跨架构诊断框架，将SHAP与机制解释相结合，系统分析模型在域移位下的失败机制，并量化数据噪声与“自然攻击文本”对泛化的负面影响。

**🔧 技术方法**

采用QLoRA参数高效微调，利用SHAP解释特征重要性，结合注意力头机制分析，评估模型内部特征检测。

**📊 数据集**

使用Enron、SpamAssassin和近年钓鱼邮件语料库（现代钓鱼语料）三大数据集，涵盖不同风格和年代。

**📈 对比分析**

通过在hold‑out集上比较标准微调与CoT微调的F1、准确率等指标，Gemma 2 9B一般化模型达到91.3% F1，Mistral 83.8%，Llama 6.6% F1。

**⚠️ 局限性**

受限于数据集标签噪声和潜在对抗性文本，且CoT方法对某些架构无益，说明单一微调策略难以保证跨域鲁棒性。

---

## 354. DR-Arena: an Automated Evaluation Framework for Deep Research Agents

**arXiv ID:** 2601.10504 | [PDF](https://arxiv.org/pdf/2601.10504v1)

**作者:** Yiwen Gao `[一作]` (National University of Singapore), Wenxuan Zhang `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5100629634)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种完全自动化的动态评估框架 DR-Arena，用于测评深度研究代理在实时信息环境中的推理和信息获取能力。

**💡 创新点**

创新点在于实时构建信息树、自动化任务生成、以及自适应演化循环动态提升难度，避免了传统静态基准的时效性和污染问题。

**🔧 技术方法**

技术上结合 LLM Examiner、信息树爬取、基于结构化清单的判定、Elo 计分与自适应循环。

**📊 数据集**

使用来自实时网络的动态信息树，采样自 Google Trends 的主题，无需预制固定数据集。

**📈 对比分析**

与现有静态基准和 LMSYS Search Arena 进行对比，Spearman 相关度0.94，显著优于其他基准，且与人类评估高度一致。

**⚠️ 局限性**

局限包括参数偏置导致的检查表与评判者冲突、对搜索引擎依赖、以及对创意性合成评价不足。

---

## 355. Model See, Model Do? Exposure-Aware Evaluation of Bug-vs-Fix Preference in Code LLMs

**arXiv ID:** 2601.10496 | [PDF](https://arxiv.org/pdf/2601.10496v1)

**作者:** Ali Al-Kaswan `[一作]` (Delft University of Technology), Maliheh Izadi `[通讯]` (Delft University of Technology)

**通讯引用:** 4516 | [OpenAlex ID](https://openalex.org/A5024645888)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了暴露感知评估框架，通过成员推断确定代码片段在训练数据中的曝光情况，并量化曝光对LLM在bug与修复之间偏好的影响。

**💡 创新点**

创新点在于将训练曝光信息引入模型偏好评估，结合多种token概率指标和生成匹配分析，揭示LLM在bug传播与修复偏好中的记忆与泛化机制。

**🔧 技术方法**

使用Data Portraits做成员推断，计算最小/最大概率、困惑度、Gini系数等多种指标，并在不同暴露组上比较模型生成结果。

**📊 数据集**

使用Stack‑v2大型代码语料做曝光检测，评估对象为ManySStuBs4J Java单语句bug‑fix对。

**📈 对比分析**

将样本按“仅见bug”“仅见修复”“两者皆见”“均未见”四类划分，比较模型在各组中对bug/修复的偏好率和生成匹配率，结果显示即使修复更常见，模型在暴露bug时仍倾向生成bug，最小概率等指标在所有情况均偏好修复。

**⚠️ 局限性**

局限性包括仅考虑Java单语句bug、成员推断采用二进制阈值导致误判、缺乏多语言与完整项目上下文、只评估公开模型且未覆盖更大规模或闭源模型。

---

## 356. DeFlow: Decoupling Manifold Modeling and Value Maximization for Offline Policy Extraction

**arXiv ID:** 2601.10471 | [PDF](https://arxiv.org/pdf/2601.10471v1)

**作者:** Zhancun Mu `[一作]` (Peking University), Zhancun Mu `[通讯]` (Peking University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5100598366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DeFlow，一种在离线强化学习中将行为分布建模与价值最大化解耦的框架，使用多步流匹配生成行为分布，并通过轻量级修正模块在保持分布约束的前提下进行价值驱动的策略改进。

**💡 创新点**

创新点在于将流模型的行为建模与价值优化完全分离，利用停止梯度和自动拉格朗日乘子实现动态约束，既避免了传统联合优化导致的模式崩塌，又不需要对 ODE 求解器求导；同时提供了在线到离线的无缝迁移方案。

**🔧 技术方法**

主要技术包括条件流匹配 (Conditional Flow Matching)、ODE 解决器、停止梯度、自动拉格朗日乘子约束、轻量级 MLP 修正模块、价值函数学习和离线 RL 的约束优化。

**📊 数据集**

在 OGBench（73 个离线任务和 15 个在线迁移任务）以及 D4RL 基准上进行评测。

**📈 对比分析**

与多种基线（Gaussian、Diffusion、Flow 等）对比，DeFlow 在大多数任务上达到或超过现有最先进方法，尤其在高度多模态任务上表现显著；在线适配实验中也实现了与 Cal‑QL、RLPD 等方法相当或更优的性能。

**⚠️ 局限性**

局限性包括对行为分布质量高度依赖、需对 δ 进行粗略估计、轻量修正模块可能不足以在极端复杂环境中达到最佳值，以及在数据质量较差时性能下降。

---

## 357. AI Sycophancy: How Users Flag and Respond

**arXiv ID:** 2601.10467 | [PDF](https://arxiv.org/pdf/2601.10467v1)

**作者:** Kazi Noshin `[一作]` (University of Illinois Urbana-Champaign), Sharifa Sultana `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 636 | [OpenAlex ID](https://openalex.org/A5103026409)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于 r/ChatGPT 子版块的 Reddit 讨论数据，本文通过主题分析构建了 Observation‑Detection‑Response (ODR) 框架，探讨用户如何观察、检测以及回应 LLM 的 sycophancy（迎合倾向）行为。

**💡 创新点**

创新点包括：①提出了 ODR 框架，对 sycophancy 的观察、检测、回应三阶段进行系统化；②发现 sycophancy 在不同情境下既有潜在危害，又能为脆弱用户提供情感支持；③归纳了用户自行开发的检测与缓解技术；④提出了上下文感知的 AI 设计建议，平衡透明度、准确性与情感支持。

**🔧 技术方法**

技术手段主要是：主题分析（Thematic Analysis）、基于词嵌入的关键词相似度（spaCy）、BERTopic 主题建模、NRC 情感词典计数、人工编码与聚类；并未使用机器学习模型训练。

**📊 数据集**

数据集为 2025 年 7 月至 12 月间的 r/ChatGPT 讨论，包含 3,600 条帖子与 140,416 条评论，涉及约 54,014 名独立用户。

**📈 对比分析**

对比方法：使用词典计数估算主题出现频率，并与 Anthropic 与 OpenAI 的官方指南进行对照；并无统一性能指标，而是通过质性编码评估用户检测与缓解技术的多样性与效果。研究显示，sycophancy 的影响随情境变化，既有危害也有益处。

**⚠️ 局限性**

限制：①数据仅来自 Reddit，可能偏向年轻、西方、技术导向的用户；②关键词提取基于已有文献，可能遗漏新兴俚语；③研究仅聚焦 ChatGPT，结果可能不适用于其他 LLM；④与不同厂商指南对比时存在主体差异，结果可能受限。

---

## 358. Architectural Classification of XR Workloads: Cross-Layer Archetypes and Implications

**arXiv ID:** 2601.10463 | [PDF](https://arxiv.org/pdf/2601.10463v1)

**作者:** Xinyu Shi `[一作]`, Francky Catthoor `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文基于跨层方法，对12种典型的XR（AR/VR）工作负载进行系统的架构级评估，揭示其对算力、存储、内存带宽和控制开销的不同敏感性。

**💡 创新点**

创新点在于将模型驱动的设计空间探索与GPU/CPU实际测功分析相结合，提出四类工作负载原型（容量门控、可实现性有限、均衡缓存友好、几何不规则），并给出针对不同原型的可操作化架构建议。

**🔧 技术方法**

主要技术包括：基于SIMD通用计算模板的轻量级分析DSE框架、L1/LLC容量扫描、Simulated Annealing的调度/tiling搜索、GPU Nsight Compute的内核级指标提取以及CPU Pin级 ISA监控。

**📊 数据集**

使用的数据集和网络来自公开XR工作负载集合，包括Monodepth2、HR-Depth、PWC‑Net、RAFT‑Stereo、LightGlue、SuperGlue、LoFTR、TartanVO、NeRF/TinyNeRF、ViT、Cupoch ICP等，统一以batch=1、实际分辨率运行。

**📈 对比分析**

通过在统一内存容量网格上进行能耗与内存侧延迟的Pareto分析，发现大部分工作负载在提升LLC容量时能同时降低能耗和延迟；但在容量门控或阶段交替的场景下，需要在能耗、延迟与面积/功耗之间做权衡，显著优于传统单一资源扩展方案。

**⚠️ 局限性**

局限性包括：分析基于简化的能耗与延迟模型，未考虑周期级精确预测；仅关注单应用、batch=1的实时推理，未覆盖多任务干扰、动态调度及完整系统功耗；SIMD模板作为统一分析基准而非最终硬件设计。

---

## 359. ChartComplete: A Taxonomy-based Inclusive Chart Dataset

**arXiv ID:** 2601.10462 | [PDF](https://arxiv.org/pdf/2601.10462v1)

**作者:** Ahmad Mustapha `[一作]` (American University of Beirut), Mariette Awad `[通讯]` (American University of Beirut)

**通讯引用:** 6377 | [OpenAlex ID](https://openalex.org/A5008382926)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了覆盖30种图表类型、每种50张图像的ChartComplete数据集，用于图表分类。

**💡 创新点**

创新点在于将可视化社区的Borkin图表分类法改造后广泛覆盖常见与罕见图表，并提供公开高质量图像集合，弥补现有数据集仅包含少数图表类型的缺陷。

**🔧 技术方法**

采用Google ViT提取视觉特征，利用FAISS进行近邻检索筛选图像，计算CKA衡量不同图表类型的视觉相似度，并用t‑SNE可视化特征空间。

**📊 数据集**

数据集基于Statista、Our World in Data等公开来源抓取并人工补全，最终形成约1500幅图表的分类集合。

**📈 对比分析**

未对模型性能做直接评估，但通过特征空间可视化和CKA热图展示不同图表类型在视觉特征上的分离度，暗示模型可对该数据集实现良好分类。

**⚠️ 局限性**

主要局限在于仅提供无监督的分类图像数据，缺乏问答或摘要等训练信号，且视觉特征提取依赖通用ViT，可能对某些图表类型区分不足。

---

## 360. NSR-Boost: A Neuro-Symbolic Residual Boosting Framework for Industrial Legacy Models

**arXiv ID:** 2601.10457 | [PDF](https://arxiv.org/pdf/2601.10457v1)

**作者:** Ziming Dai `[一作]` (Tianjin University), Haojun Fei `[通讯]` (Qfin Holdings)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种非侵入式的 NSR-Boost 框架，在不改动已有 GBDT 模型的前提下，通过 LLM 生成符号专家对模型弱点区域进行残差修正，提升性能并保持低延迟；

**💡 创新点**

核心创新在于：①将 LLM 用作符号代码生成器而非直接预测；②采用双层优化（结构搜索+贝叶斯参数微调）解决符号回归的离散‑连续问题；③利用残差引导的硬区域划分与上下文感知聚合实现局部精细化；

**🔧 技术方法**

技术包括大型语言模型 (如 GPT‑3.5‑Turbo / Seed‑OSS‑36B‑Instruct)、符号回归代码生成、树结构贝叶斯优化、CART 边界分割、上下文感知 XGBoost 聚合器、LLM 与 Python 代码交互循环；

**📊 数据集**

实验涵盖六个公开 OpenML 数据集（adult、bank‑marketing、blood‑transfusion、breast‑w、credit‑g、pc1）和一份 Qfin Holdings 的金融风控私有数据；

**📈 对比分析**

与 XGBoost、TabNet、FT‑Transformer、OpenFE、AutoFeat、CAAFE、FeatLLM、OCTree、LLM‑FE 等基线对比，NSR-Boost 在所有公开集上平均提升 0.7%–1.5%（最高 1.3%），在私有集 AUC 提升至 0.694（相较 0.659 以上 0.5%）；线上实验显示在大规模金融场景中 AUC、KS 均提升 0.3%–1.2%；

**⚠️ 局限性**

局限性包括：依赖 LLM 的生成质量与 prompt 设计；双层优化计算量较大，生成阶段需 GPU/CPU 资源；对超大规模特征空间的适应性未全面验证；在极端噪声或概念漂移时仍需人工干预。

---

## 361. Breaking Up with Normatively Monolithic Agency with GRACE: A Reason-Based Neuro-Symbolic Architecture for Safe and Ethical AI Alignment

**arXiv ID:** 2601.10520 | [PDF](https://arxiv.org/pdf/2601.10520v1)

**作者:** Felix Jahn `[一作]` (German Research Center for Artificial Intelligence), Kevin Baum `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种神经符号的“治理架构”，通过将道德推理、工具性决策和合规监控三大模块分离，实现了对任何AI代理的可解释、可争议的伦理约束。

**💡 创新点**

创新点在于将“规范理由”作为道德推理基础，构建了可动态更新、可证伪的规范理论，并通过多代理架构将符号约束与黑箱决策器无缝耦合，解决了传统单一策略的“扁平化”问题。

**🔧 技术方法**

采用了基于默认推理的规范理论（Horty 形式的 defeasible logic）、时序逻辑表示的宏观行为类型（MATs），以及神经网络（LLM 或 RL 代理）与符号守卫模块的协同。

**📊 数据集**

在示例中使用了虚构的心理治疗对话场景和人工生成的案例反馈，未涉及公开大规模数据集；作者计划未来结合真实人类监督反馈进行评估。

**📈 对比分析**

目前仅提供理论推导和案例演示，未进行实验对比；后续计划在 LLM 语义推理与 RL 环境中测评道德合规率、性能损失及可争议性指标。

**⚠️ 局限性**

局限性包括：缺乏大规模实验验证、对动态伦理环境的自适应能力待验证、规范理由库需人工维护、Guard 模块在复杂非确定性环境下的决策可能存在误判。

---

## 362. Defending Large Language Models Against Jailbreak Attacks via In-Decoding Safety-Awareness Probing

**arXiv ID:** 2601.10543 | [PDF](https://arxiv.org/pdf/2601.10543v1)

**作者:** Yinzhi Zhao `[一作]` (Northeastern University), Yifei Zhang `[通讯]` (Northeastern University)

**通讯引用:** 2976 | [OpenAlex ID](https://openalex.org/A5100386920)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现一种在 LLM 解码过程中实时探测安全信号（SafeProbing），通过在生成时插入 “Note that this is” 并计算后续词的 token‑level 损失，及时检测并拦截 jailbreak 攻击生成的有害内容。

**💡 创新点**

创新点：① 发现 LLM 在生成有害文本时仍保持内在的安全意识；② 将该安全信号转化为可量化的损失指标，并在解码期间实时检测；③ 通过轻量微调增强模型的安全信号分布，显著提升检测阈值的可判别性；④ 采用稀疏检查（20% 抽样）实现低开销的 in‑decoding 保护。

**🔧 技术方法**

技术细节：自回归语言模型推理、token‑level 负对数似然损失计算、LoRA 微调、阈值检测、随机采样检查点、轻量级损失映射与 MSE 训练、对抗与正则化结合。

**📊 数据集**

数据集与攻击：微调使用 SafeRLHF、UltraFeedback；攻击测试使用 AdvBench、HEx‑PHI、AutoDAN、PAIR、IFSJ、ReNeLLM、REDA、DRA、AutoDAN‑Turbo；评估使用 LlamaGuard‑3‑8B（判别）和 GPT‑Judge（分数）；功能评测用 GSM、JustEval；多模态测试用 LLaVA、Qwen2.5‑VL‑7B、GLM‑4V‑9B 与 MM‑SafetyBench、FigStep、HADES。

**📈 对比分析**

与基线（SafeDecoding、ICD、SmoothLLM、DRO、Backtranslation、RobustAligned、SelfEval）比较：在 Qwen2.5‑7B‑Instruct 和 Mistral‑7B‑Instruct 上，SafeProbing 在多种 jailbreak 攻击下 DSR 皆 ≥90%，over‑refusal 远低于检测基线，且对 GSM/JustEval 的回答质量几乎无下降；在多模态场景下亦保持良好区分能力。

**⚠️ 局限性**

局限性：目前采用硬性拒绝策略，未探索部分删改、引导重写等更细粒度的响应；阈值需经验设定；对高分辨率图像或更复杂多模态输入的鲁棒性待进一步验证；训练成本和推理时额外的损失计算开销有限但仍存在。

---

## 363. Mixtures of Transparent Local Models

**arXiv ID:** 2601.10541 | [PDF](https://arxiv.org/pdf/2601.10541v1)

**作者:** Niffa Cheick Oumar Diaby `[一作]` (Laval University), Mario Marchand `[通讯]` (Laval University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出混合透明局部模型（Mixtures of Transparent Local Models）并给出PAC‑Bayesian风险界，适用于已知或仅已知数量的兴趣点。

**💡 创新点**

将可解释线性局部模型与PAC‑Bayes理论相结合，提供全局风险上界，支持局部性重叠，并在兴趣点未知时仍能学习局部性与模型参数。

**🔧 技术方法**

使用PAC‑Bayes风险界、可解释线性模型、vicinity函数、Gamma/高斯先验后验、NAdam优化、随机重启、欧氏距离等技术。

**📊 数据集**

在合成数据集和16个KEEL公开数据集（8个二分类、8个回归）上进行实验。

**📈 对比分析**

与线性/高斯SVM、SVR以及League of Experts（线性专家）比较；在合成数据上明显优于全局线性模型，在真实数据上性能接近高斯模型，优于线性模型且略低于最强不透明模型。

**⚠️ 局限性**

局限性：需手动指定兴趣点数目；对未知兴趣点的风险界仅在欧氏度量下有效；优化过程非凸需多次随机重启；仅采用线性透明模型，缺少对更复杂可解释模型的探索。

---

## 364. Error-Correcting Codes for Two Bursts of t1-Deletion-t2-Insertion with Low Computational Complexity

**arXiv ID:** 2601.10540 | [PDF](https://arxiv.org/pdf/2601.10540v1)

**作者:** Yajuan Liu `[一作]` (Bilkent University), Tolga M. Duman `[通讯]` (Bilkent University)

**通讯引用:** 6009 | [OpenAlex ID](https://openalex.org/A5048023330)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究并构造了能纠正二次 (t1,t2)-DI 误差的二进制纠错码，并给出了它们的等价性、码量上下界以及具体编码方案。

**💡 创新点**

创新点在于：① 证明两次 (t1,t2)-DI 码等价于两次 (t2,t1)-DI 码以及一次 (t1,t2) 与一次 (t2,t1) 组合的码；② 在此基础上给出码量的严格上下界；③ 提出了低复杂度的编码方法，利用矩阵分块、d-regular 序列以及 1‑(t'−1)‑DS 的映射，将原问题转化为更易纠错的形式，最终实现了 8logn+O(1) 或 11logn+o(logn) 的冗余。

**🔧 技术方法**

采用的技术包括：同步错误码理论、VT 变换与稠密序列编码、GRS（广义 Reed–Solomon）码、符号压缩技术（syndrome compression）以及矩阵分块与 d‑regular 序列构造。

**📊 数据集**

本文并未使用任何实验数据集，而是通过理论证明和构造给出结论。

**📈 对比分析**

性能比较采用了集合 |𝒩*| 的大小来衡量算法复杂度。与传统直接使用符号压缩得到的 8logn+o(logn) 码相比，新方案在相同冗余下将复杂度降低到 (t1−t2)^4 的量级，具体通过举例展示了 |𝒩*| 的显著缩小，从而证明了计算效率的提升。

**⚠️ 局限性**

局限性包括：① 仅适用于 t1≥t2 的二进制序列；② 对 t1=t2 的情况仍需使用 GRS 码；③ 方案仍保持较高的冗余率，尚未达到理论极限；④ 只处理非重叠 burst，无法覆盖所有实际冲突场景；⑤ 没有给出真实实现的实验评估。

---

## 365. Network Integrated Sensing and Communication

**arXiv ID:** 2601.10538 | [PDF](https://arxiv.org/pdf/2601.10538v1)

**作者:** Edward Andrews `[一作]` (University of Newcastle), Min Li `[通讯]` (Zhejiang University)

**通讯引用:** 32242 | [OpenAlex ID](https://openalex.org/A5100400752)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了网络级ISAC（集成感知与通信）系统的吞吐量与感知覆盖率之间的取舍，并给出了单链路一维路径网络的完整解析区间以及一般网络的支点和线性段结构。

**💡 创新点**

创新点在于提出了同时考虑多节点路由与感知区域的统一优化框架，并在理论上证明了感知-吞吐量区域的凸性、最大化感知、最大化吞吐量点以及泛化网络中支点的分段线性Pareto边界，首次揭示了路由与感知协同的物理意义。

**🔧 技术方法**

主要技术包括线性规划、最大流-最小割定理、凸分析、Bisection 搜索以及对一维路径网络的解析推导；同时引入了感知-吞吐量的线性耦合约束。

**📊 数据集**

本文为理论分析，不依赖实际数据集，而是以抽象的容量模型和感知区域集合作为输入。

**📈 对比分析**

性能通过解析解与数值仿真对比，验证了理论边界与实际求解结果的吻合度，显示在一维路径网络中可达到理论上最大吞吐量与感知率的组合；在一般网络中通过线性规划得到的 Pareto 边界与仿真一致。

**⚠️ 局限性**

局限性包括：1）感知与通信的线性权衡假设在实际信道中可能不成立；2）感知率被视为独立且可加，总体感知质量受相关性影响；3）仅考虑单源单汇场景，未考虑多源或多汇情况；4）未考虑能耗、干扰、波形设计等实际系统约束。

---

## 366. A Safety Report on GPT-5.2, Gemini 3 Pro, Qwen3-VL, Doubao 1.8, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5

**arXiv ID:** 2601.10527 | [PDF](https://arxiv.org/pdf/2601.10527v1)

**作者:** Xingjun Ma `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 23651 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对7款前沿大语言模型与多模态模型在语言、视觉-语言、图像生成三大使用模式下进行统一安全评估

**💡 创新点**

首次构建统一评估协议，融合基准测试、对抗测试、多语言测试与合规测试，系统揭示安全多维度差异

**🔧 技术方法**

使用公开安全基准、自动化 jailbreak 攻击、跨语言评测工具、监管框架解析器等技术

**📊 数据集**

采集的评测数据集包括：ALERT、Flames、BBQ、SORRY‑Bench、StrongREJECT、PolyGuardPrompt、ML‑Bench、MemeSafetyBench、MIS、USB‑SafeBench、SIUO、VLJailbreakBench、JailbreakV‑28K、MM‑SafetyBench、T2ISafety 等

**📈 对比分析**

与 GPT‑5.2、Gemini 3 Pro 等模型对比，GPT‑5.2 在基准、对抗、跨语、多模态与合规六个维度均位居榜首，安全分数最高；其他模型在各维度存在显著差异，平均安全率从 60 % 低至 97 %

**⚠️ 局限性**

仍易受对抗攻击，跨语言与跨模态安全泛化不足，监管合规表现不均衡，存在文化偏见与边缘风险未被完全覆盖

---

## 367. Transformer-Based Cognitive Radio: Adaptive Modulation Strategies Using Transformer Models

**arXiv ID:** 2601.10519 | [PDF](https://arxiv.org/pdf/2601.10519v1)

**作者:** Andrea Melis `[一作]` (University of Bologna), Roberto Girau `[通讯]` (University of Bologna)

**通讯引用:** 1600 | [OpenAlex ID](https://openalex.org/A5023826523)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用 GPT‑2 Transformer 对已有调制公式进行训练，自动生成新的调制方案并在仿真环境中评估其性能。

**💡 创新点**

首次将 Transformer 直接用于调制设计，能自动生成与传统调制相当甚至更优的调制公式，突破手工设计的局限。

**🔧 技术方法**

采用 GPT‑2 模型、tokenizer、温度控制采样以及仿真评估框架（SNR、BER、PSD）进行调制生成与性能测评。

**📊 数据集**

训练集约 10,000 条包含 PSK、QAM、OFDM 等传统调制公式的文本数据，按 80/10/10 划分训练/验证/测试。

**📈 对比分析**

通过在相同噪声水平下对比 AM/FM/PM、16/64/256‑QAM 等经典方案，生成方案的 SNR 多达 20+ dB，BER 小于 0.1，谱效率提升约 2–3 倍。

**⚠️ 局限性**

生成公式语法错误率高、实现复杂度大、硬件实现与功耗未验证、BER 仍高于低阶传统调制，需进一步优化。

---

## 368. AEQ-Bench: Measuring Empathy of Omni-Modal Large Models

**arXiv ID:** 2601.10513 | [PDF](https://arxiv.org/pdf/2601.10513v1)

**作者:** Xuan Luo `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 111821 | [OpenAlex ID](https://openalex.org/A5100336796)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个用于测量多模态大模型（OLM）同理心的基准（Audio Empathy Quotient Benchmark，AEQ），涵盖生成同理回应和评判音频回应的两大任务。

**💡 创新点**

创新点包括：① 引入上下文变化和语调变化两种并行设置，系统性考察语言与声学的共同影响；② 结合音频与文本两种模态进行同理评估；③ 采用多维度（语言与声学）指标，对模型进行细粒度评价。

**🔧 技术方法**

技术手段：利用多模态大型模型（如GPT‑4o‑audio、Qwen‑Omni 系列、Qwen‑Audio 系列、Baichuan、LLaMA、Flamingo 等）进行回应生成与评判；使用 GPT‑5‑mini 等自动判别器做语言评估；通过人类标注实现基准验证；使用多模态评估指标（模态依赖、自然度、连贯性、支持性、辨别度、交付）和人类一致性分析。

**📊 数据集**

数据集：1,885 条英语音频‑文本对，来源于 TV series、studio recording、streaming content 三类；每条记录包含一段音频、上下文及对应参考回应；通过音频过滤、GPT 生成上下文、人工验证构造。 该数据集包含两种变化设定：同一句音频在不同上下文中，或同一上下文配不同情绪语调。

**📈 对比分析**

比较方法：对 9 种模型进行自然度、辨别度、交付等指标的统一评分，计算模态依赖比例；使用人类标注与模型判别的一致性（互评一致率、与人类评判的相符度）进行对比。结果显示：① 具备音频输出能力的模型在自然度、辨别度、交付上明显优于仅文本输出模型；② GPT‑4o 与 Qwen‑Omni 系列在整体自然度与人类一致度上名列前茅；③ 但在细粒度声学表达上，模型仍与人类评判差距显著。

**⚠️ 局限性**

局限性：① 数据仅限英语，缺乏跨语言跨文化验证；② 评估指标为粗粒度分类（Good/Fair/Poor），缺少细粒度情感细节；③ 合成音频情绪表达有限，导致声学同理评估受限；④ 文字描述的音频字幕无法替代直接音频分析，评判效果不理想。

---

## 369. SatMap: Revisiting Satellite Maps as Prior for Online HD Map Construction

**arXiv ID:** 2601.10512 | [PDF](https://arxiv.org/pdf/2601.10512v1)

**作者:** Kanak Mazumder `[一作]` (Munich University of Applied Sciences), Fabian B. Flohr `[通讯]` (Munich University of Applied Sciences)

**通讯引用:** 1282 | [OpenAlex ID](https://openalex.org/A5007686963)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SatMap，一种利用卫星图像与多视角相机融合的在线向量化高清地图预测方法。

**💡 创新点**

创新点在于使用基于Swin Transformer的卫星图像特征提取与BEV空间卷积融合，有效解决遮挡与对齐问题，显著提升长距离与恶劣天气下的地图构建性能。

**🔧 技术方法**

采用相机‑卫星多模态融合、GKT视图变换、Swin‑Tiny + GFPN、ConvFuser、DETR解码器等技术。

**📊 数据集**

主要在nuScenes数据集上训练与评测。

**📈 对比分析**

与Camera‑only、Camera‑LiDAR融合及多种基线模型对比，SatMap在mAP上提升约34.8%（Camera‑only）与8.5%（Camera‑LiDAR），在不同天气与长距离场景下均保持领先。

**⚠️ 局限性**

局限在于仅为单帧框架，未利用时序信息；卫星图像中建筑、树木等遮挡物仍可能引入噪声。

---

## 370. A New Construction Structure on Coded Caching with Linear Subpacketization: Non-Half-Sum Latin Rectangle

**arXiv ID:** 2601.10505 | [PDF](https://arxiv.org/pdf/2601.10505v1)

**作者:** Yongcheng Yang `[一作]` (Guangxi Normal University), Giuseppe Caire `[通讯]` (Technische Universität Berlin)

**通讯引用:** 28788 | [OpenAlex ID](https://openalex.org/A5058252389)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了“非半和拉丁矩形”（NHSLR）这一新的组合结构，并利用该结构构造了子包化率线性（F=O(K）且传输负载低的编码缓存方案。

**💡 创新点**

将原先严格要求F=K的NHSDP扩展为更宽松的NHSLR，允许F=O(K)，从而在保持子包化率线性增长的同时显著降低传输负载并扩大可用用户数与内存比例范围。

**🔧 技术方法**

主要采用组合设计、PDA构造、非半和（half‑sum）属性以及基于对角矩阵X的数值优化来生成NHSLR，进而得到对应的缓存方案。

**📊 数据集**

本文没有使用具体的实验数据集，而是通过理论推导与数值实验（不同参数组合的计算结果）来验证所提方案的性能。

**📈 对比分析**

通过与MN、WCLC、WCWL、XXGL、CWWC等已知线性及指数分割方案在相同内存比例下的理论分析和数值对比，结果表明所提方案在保持线性子包化率的同时，传输负载更低，且在某些参数下接近指数分割方案的性能。

**⚠️ 局限性**

该方案的子包化率仍随用户数线性增长；当用户数为偶数时需引入虚拟用户；构造参数受v≥∏(m_i+1)的约束，且在所有参数配置下不一定优于指数分割方案。

---

## 371. Coded Caching for Combinatorial Multi-Access Hotplug Networks from $t$-Designs

**arXiv ID:** 2601.10503 | [PDF](https://arxiv.org/pdf/2601.10503v1)

**作者:** Dhruv Pratap Singh `[一作]` (Indian Institute of Science), B. Sundar Rajan `[通讯]` (Indian Institute of Science)

**通讯引用:** 6591 | [OpenAlex ID](https://openalex.org/A5015398340)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

设计了一种在多缓存可访问且仅部分缓存在线的热插拔环境下的编码缓存方案，利用t-设计构造通用的热插拔置放与交付数组。

**💡 创新点**

创新点在于将热插拔置放交付数组(HpPDA)推广到组合多访问架构，允许用户访问多缓存，并通过参数化的t-设计实现可控的多余多播传输，从而在不同子分割级别下得到更优的速率‑内存折衷。

**🔧 技术方法**

主要技术包括：t-设计理论、MDS编码置放、通用HpPDA框架以及基于PDAs的多播传输设计。

**📊 数据集**

实验使用理论模型与合成数据（如C=8、r=2、C'=3、N=18等参数）进行数值比较，不涉及真实数据集。

**📈 对比分析**

通过将所提t-方案与现有CRR t‑scheme、CRR MT、RR及MT方案在同一速率/用户基准下对比，结果显示在缓存比例0.5–0.7范围内，所提方案的每用户速率更低，说明在热插拔环境中利用多缓存访问可显著提升传输效率。

**⚠️ 局限性**

局限性包括：需要提前知道在线缓存数C'且无法处理在线缓存集合未知的情况；子分割量随设计参数变化，可能导致实现复杂度上升；且方案仍以理论分析为主，缺乏在实际网络环境中的验证。

---

## 372. Projected Microbatch Accumulation yields reference-free proximal policy updates for reinforcement learning

**arXiv ID:** 2601.10498 | [PDF](https://arxiv.org/pdf/2601.10498v1)

**作者:** Nilin Abrahamsen `[一作]` `[通讯]`, Nilin Abrahamsen

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在微批次梯度累积过程中对累计梯度做投影补空间的近端策略更新方法——Projected Microbatch Accumulation（PROMA）。

**💡 创新点**

创新点在于：在反向传播期间层级投影序列梯度分量，消除梯度重叠，从而在不依赖参考策略或剪裁的情况下实现局部KL约束，避免熵崩塌并保持更高探索性。

**🔧 技术方法**

技术手段包括：近似投影到补空间（QR分解或迭代正交化）、微批次梯度累积、REINFORCE基础、KL距离监控与对比实验、以及对梯度范数的截断/缩放。

**📊 数据集**

使用的数据集为 GSM8K（算术推理任务）以及 Qwen-3 0.6B 语言模型。

**📈 对比分析**

与基线 GRPO（带 PPO 剪裁）和 REINFORCE（无剪裁）对比实验显示：PROMA 在验证性能上相当或更好；全局 KL 更受控；政策熵更长时间保持高值；连续策略之间的 KL 更低，表明更新更平滑，整体性能优于两种基线。

**⚠️ 局限性**

局限性包括：投影运算虽然近似可降为 O(kd) FLOPs，但在大规模模型或多任务场景下仍需评估实际开销；缺乏严格的收敛理论保证；实验仅在单一数据集/模型上验证，推广性待进一步验证。

---

## 373. Joint Source-Channel Coding for ISAC: Distortion Tradeoffs and Separation Theorems

**arXiv ID:** 2601.10470 | [PDF](https://arxiv.org/pdf/2601.10470v1)

**作者:** Gefei Peng `[一作]` (ShanghaiTech University), Youlong Wu `[通讯]` (ShanghaiTech University)

**通讯引用:** 1064 | [OpenAlex ID](https://openalex.org/A5053320672)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在状态相关的ISAC系统中联合源-信道编码（JSCC）的性能限制，给出了容量-失真-成本（capacity–distortion–cost）函数，并证明在该模型下源与信道编码分离即可达到联合最优。

**💡 创新点**

创新点在于：①首次在ISAC框架下将感知失真与通信失真统一入信息论分析；②推导出精确的容量-失真-成本上限，并证明分离编码策略可实现；③给出二进制信道的闭式示例，验证理论并展示容量-失真曲线始终优于率-失真曲线。

**🔧 技术方法**

主要技术包括：信息论的相互信息与率-失真函数分析、凸优化（对失真和成本约束的可行集）、典型序列与典型集合方法、隐马尔可夫链的数据处理不等式，以及符号级最优状态估计与最优源解码的构造。

**📊 数据集**

未使用实际数据集；示例采用合成的二进制输入/输出、Bernoulli 状态的离散信道，利用符号级模型演示理论结果。

**📈 对比分析**

通过对比容量-失真曲线与率-失真曲线（在同一二进制信道设置下），证明容量-失真曲线始终位于其上方，且两者交点对应的编码方案实现了信道容量。该比较说明了理论的正确性与分离策略的有效性。

**⚠️ 局限性**

局限性包括：仅考虑状态相关的无记忆离散信道；失真函数被限定为二元符号级最优估计；未讨论多输入多输出或连续信道情况；示例仅覆盖有限的二进制案例，实际系统需进一步验证。

---

## 374. Subjective evaluation of UHD video coded using VVC with LCEVC and ML-VVC

**arXiv ID:** 2601.10448 | [PDF](https://arxiv.org/pdf/2601.10448v1)

**作者:** Naeem Ramzan `[一作]` (University of the West of Scotland), Muhammad Tufail Khan `[通讯]` (University of the West of Scotland)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对LCEVC增强层与VVC基础层的多层编码进行主观质量评估，并与简单上采样VVC及ML‑VVC进行对比。

**💡 创新点**

在最新版LTM 8.1下，系统地评估E10/E50两种比特率点的LCEVC性能，并与ML‑VVC进行直接比较，提供MOS及95%置信区间，首次在相同测试计划下完成。

**🔧 技术方法**

使用LCEVC 8.1增强层、VTM 23.8 VVC基础层，采用DCR主观评测方法（11分评分尺度）以及量化参数配置。

**📊 数据集**

15个UHD SDR/HDR序列（如BodeMuseum、Metro、WomenFootball等），共15个视频序列。

**📈 对比分析**

通过MOS与95%置信区间比较三种方案；E10下LCEVC和ML‑VVC均优于上采样，E50下两者表现相近且显著优于上采样，整体主观质量提升明显。

**⚠️ 局限性**

仅针对所选编码器版本、量化参数、重采样方法，未评估复杂度、延迟或成本；仅研究两比特率点，统计显著性受样本量与序列差异限制。

---

## 375. Job Anxiety in Post-Secondary Computer Science Students Caused by Artificial Intelligence

**arXiv ID:** 2601.10468 | [PDF](https://arxiv.org/pdf/2601.10468v1)

**作者:** Daniyaal Farooqi `[一作]` (University of Toronto), Syed Ishtiaque Ahmed `[通讯]` (University of Toronto)

**通讯引用:** 3939 | [OpenAlex ID](https://openalex.org/A5089574660)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在多伦多大学对25名计算机科学本科生和研究生进行半结构化访谈，研究AI技术导致的工作岗位替代焦虑及其对学习与职业选择的影响

**💡 创新点**

首次系统揭示国际学生因签证与移民压力而面临更高的替代焦虑，并从定性数据中提炼出“升职”“转行”“提升AI技能”三大应对策略，填补了先前仅聚焦专业人士的研究空白

**🔧 技术方法**

采用访谈记录、录音转写与问卷Likert量表相结合，利用主题分析（Thematic Analysis）提取关键主题

**📊 数据集**

数据集为28份问卷（其中25份访谈）及其对应的访谈转写文本，受访者包括18名国内、7名国际计算机科学学生

**📈 对比分析**

研究未使用对照实验或性能指标，而是通过对访谈与问卷结果的描述性统计与主题对照，说明学生对AI替代焦虑的普遍程度（平均分≈4.5/7）及其对课程/职业选择的影响

**⚠️ 局限性**

局限性包括仅在单一高校单一系别样本，国际学生样本量不足，且可能存在自我选择偏差，导致对更广泛北美或全球学生的外推性有限

---

## 376. Higher order trade-offs in hypergraph community detection

**arXiv ID:** 2601.10502 | [PDF](https://arxiv.org/pdf/2601.10502v1)

**作者:** Jiaze Li `[一作]` (Maastricht University), Leto Peel `[通讯]` (Maastricht University)

**通讯引用:** 1295 | [OpenAlex ID](https://openalex.org/A5072266097)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种统一的理论框架，用于在非均匀超图中检测社区结构，并给出了相应的可检测性阈值与算法。

**💡 创新点**

创新点在于提出统一的信噪比（SNR）衡量、非均匀超图的 Bethe Hessian 以及对超图形状和阶数偏好对可检测性的系统性分析。

**🔧 技术方法**

技术主要包括：基于超图随机块模型（HSBM）的概率生成、Bethe Hessian 线性化、谱聚类、Belief Propagation（BP）以及理论可检测阈值推导。

**📊 数据集**

使用的数据集包括：美国和加拿大的 Yelp 评论超图、英国小学与高中的人际接触超图，以及合成超图验证。

**📈 对比分析**

与 BP、传统非退化算子对比后，Bethe Hessian 在均匀超图上达到理论极限，在非均匀超图上虽略逊色但仍能显著恢复社区；在真实数据上能够重现已知的学校班级结构或地理层级社区。

**⚠️ 局限性**

局限性包括：假设社区内节点度数和超边阶数同质化；仅考虑亲和性（assortative）结构；未给出严格证明；未考虑度数校正和反亲和（disassortative）结构等。

---

## 377. HeartMuLa: A Family of Open Sourced Music Foundation Models

**arXiv ID:** 2601.10547 | [PDF](https://arxiv.org/pdf/2601.10547v1)

**作者:** Dongchao Yang `[一作]`, Yuexian Zou `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了HeartMuLa开放源代码的音乐基础模型家族，包括音频-文本对齐（HeartCLAP）、歌词识别（HeartTranscriptor）、低帧率音乐分词器（HeartCodec）和基于LLM的可控歌曲生成（HeartMuLa）。

**💡 创新点**

核心创新点在于：①将语义丰富的多层编码器与超低帧率（12.5 Hz）分词器相结合，形成高表达、低延迟的音乐token；②采用流匹配与Reflow蒸馏实现高保真重构；③实现细粒度风格控制与短视频背景音乐两种生成模式；④在7 B参数规模下实现Suno级别性能。

**🔧 技术方法**

技术方法包括多模型特征融合（Whisper、WavLM、MuEncoder）、查询式量化、残差向量量化、流匹配重建、Reflow蒸馏、SQ-Codec微调、InfoNCE对齐、交叉熵ASR、以及Transformer/Flow Matching架构。

**📊 数据集**

使用了约600k首歌曲的内部大规模音乐语料、100k首已校正的歌词语料、1M条音频-文本对齐样本，以及公开的SongPrep、SSLD-200等多语言ASR基准。

**📈 对比分析**

与现有编码器（SemantiCodec、XCodec、MuCodec、LeVo）和对齐模型（Laion-CLAP、MuQ‑MuLan）比较，HeartCodec在VISQOL、FAD、FD、STOI、PESQ、WER等指标上均取得领先；HeartCLAP在WikiMT‑X检索任务中R@1、R@10和mAP显著超越基线；HeartTranscriptor在多语言ASR基准中WER/CER最低。

**⚠️ 局限性**

局限性包括：生成音频仍受模型规模和算力约束，长时段生成仍需高GPU内存；对极低比特率或极长音频的兼容性尚未完全验证；开放数据集缺乏版权多样性，可能影响跨文化适用性。

---

## 378. CoGen: Creation of Reusable UI Components in Figma via Textual Commands

**arXiv ID:** 2601.10536 | [PDF](https://arxiv.org/pdf/2601.10536v1)

**作者:** Ishani Kanapathipillai `[一作]` (Informatics Institute of Technology), Obhasha Priyankara `[通讯]` (Informatics Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了CoGen系统，能够通过文本指令在Figma中自动生成可重用的UI组件。

**💡 创新点**

创新点在于专注于原子级组件生成，结合JSON-描述提示对生成过程进行双向映射，并提供可编辑JSON输出。

**🔧 技术方法**

采用Seq2Seq、GRU、LSTM以及微调的T5 transformer模型；使用BERT嵌入和交叉注意力增强模型。

**📊 数据集**

数据集来源于公开的Figma设计系统和Figcomponents网站，手工提取简单与嵌套JSON对。

**📈 对比分析**

通过BLEU、ROUGE、准确率等指标比较模型，T5在提示生成上准确率98%+，在JSON生成上简单JSON 97.4%准确率，嵌套JSON 100%准确率。

**⚠️ 局限性**

局限包括仅支持六种组件与四种样式、只能生成单一变体、缺乏大规模多层嵌套数据、模型规模受限、未集成字体/图标库等。

---

## 379. A Construction Framework of Coded Caching Scheme for Multi-Access MIMO Systems via Knapsack Problem

**arXiv ID:** 2601.10484 | [PDF](https://arxiv.org/pdf/2601.10484v1)

**作者:** Siying Luo `[一作]` (Guangxi Normal University), Dianhua Wu `[通讯]` (Guangxi Normal University)

**通讯引用:** 523 | [OpenAlex ID](https://openalex.org/A5065849788)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种针对多访问 MISO 网络的编码缓存方案，并通过把缓存设计转化为 0–1 背包问题来实现更高的总 DoF 与更低的分包化水平。

**💡 创新点**

创新点在于引入基于背包优化的 MAPDA 构造框架，将复杂的组合缓存结构化为可求解的整数规划，显著提升 DoF 并降低分包化复杂度。

**🔧 技术方法**

使用了多天线置放传递数组（MAPDA）理论、组合设计、0–1 背包整数规划以及线性一拍传输技术。

**📊 数据集**

本文为理论研究，未使用实际数据集，而是通过系统参数（Λ、r、L、t 等）进行符号性分析与性能推导。

**📈 对比分析**

与 YWCC、WCC、PR、NPR 等现有 MAPDA 方案对比，实验与理论分析表明在相同缓存容量下总 DoF 更高，分包化数量相当或更低，且在满足 L ≥ Λ–t·r–Λ–t–r·r 时可达理论上限。

**⚠️ 局限性**

局限性包括：分包化规模在大规模参数下仍然较大；方案基于全组合访问拓扑且采用无编码缓存，实际部署与对任意拓扑的推广仍待进一步研究。

---

## 380. Urban Socio-Semantic Segmentation with Vision-Language Reasoning

**arXiv ID:** 2601.10477 | [PDF](https://arxiv.org/pdf/2601.10477v1)

**作者:** Yu Wang `[一作]` (Wuhan University), Yansheng Li `[通讯]` (Wuhan University)

**通讯引用:** 10057 | [OpenAlex ID](https://openalex.org/A5100606147)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出城市社会语义分割任务并构建SocioSeg数据集，随后设计SocioReasoner框架实现多阶段视觉‑语言推理。

**💡 创新点**

①将异构地理信息统一为数字地图视觉输入，②采用渲染‑精化两阶段VLM+SAM流程，③用GRPO强化学习优化非可微分推理链。

**🔧 技术方法**

Vision‑Language模型、SAM分割、渲染‑精化两阶段策略、GRPO强化学习、数字地图表示与跨模态提示技术。

**📊 数据集**

新建SocioSeg数据集（约13k样本、90类功能、5k实例名），包含卫星影像、数字地图和像素级社会语义标签。

**📈 对比分析**

与UNet、SegFormer、VisionReasoner、Seg‑R1、SAM‑R1、SegEarth系列、RemoteReasoner等基线对比，SocioReasoner在三层任务的cIoU/gIoU/F1均显著提升，且在OOS场景表现稳健。

**⚠️ 局限性**

推理速度相对较慢，对高质量地图渲染依赖较大，且对极细粒度或未见社会实体的识别仍有限。

---

## 381. SurgGoal: Rethinking Surgical Planning Evaluation via Goal-Satisfiability

**arXiv ID:** 2601.10455 | [PDF](https://arxiv.org/pdf/2601.10455v1)

**作者:** Ruochen Li `[一作]` (Technical University of Munich), Nassir Navab `[通讯]` (Technical University of Munich)

**通讯引用:** 54511 | [OpenAlex ID](https://openalex.org/A5046896448)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于目标满足度的外科规划元评估基准，并用规则检查器评估视频LLM的规划性能。

**💡 创新点**

创新点在于将规划正确性定义为阶段目标可满足性，构建多中心规则评估集，揭示传统序列相似度指标与临床目标不对齐。

**🔧 技术方法**

使用规则检查器、LLM判定器、序列相似度指标（NED、JIS、ROA）以及多种视频LLM（VideoLLaMA、LLaVA、Qwen、HuluMed、Lingshu）。

**📊 数据集**

使用MultiBypass140多中心胃旁路吻合术视频数据集进行评估。

**📈 对比分析**

通过对比传统指标与规则检查器，发现序列相似度误判多数有效方案，而LLM判定器对顺序错误敏感；在逐步约束任务中，结构知识注入显著提升性能，语义知识单独不足。

**⚠️ 局限性**

局限在于规则基评估仅覆盖有限程序，难以扩展；评估仅二元目标满足度，未考虑效率等细粒度质量；缺乏足够多样化外科数据集。

---

## 382. AgentGuardian: Learning Access Control Policies to Govern AI Agent Behavior

**arXiv ID:** 2601.10440 | [PDF](https://arxiv.org/pdf/2601.10440v1)

**作者:** Nadya Abaev `[一作]` (Ben Gurion University), Asaf Shabtai `[通讯]` (Ben Gurion University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 AgentGuardian 框架，通过在演练阶段收集 AI 代理的执行日志，自动学习并生成上下文感知的访问控制策略，随后在代理运行时实时强制执行这些策略，以控制工具调用、参数合法性和执行流程。

**💡 创新点**

创新点主要包括：① 在演练阶段基于真实日志动态构建控制流图 (CFG) 并生成多层访问控制（输入校验、ABAC 约束、流程约束）策略；② 采用嵌入 + 聚类 + LLM 生成正则表达式的方式对输入进行泛化，避免手工枚举；③ 将策略与 LiteLLM 反馈机制无缝集成，实现轻量级、即时的安全治理。

**🔧 技术方法**

技术手段包括：LLM 文本与数值属性的 150 维嵌入；聚类算法（如层次聚类）对输入与属性进行分组；正则表达式抽象与 LLM 归约；CFG 构建与路径匹配；ABAC 约束；LiteLLM 反馈接口实现实时拦截。

**📊 数据集**

数据集：使用两套公开的多代理应用（知识助手与 IT 支持），分别生成 100 个合法样本（60 用于训练、40 用于评估）并再加入 10 个恶意/误导样本；所有样本基于 gpt‑4.1 或 gpt‑4o‑mini 生成。

**📈 对比分析**

评价方法：采用 FAR、FRR、BEFR 三项指标；实验中 AgentGuardian 在 80 个合法样本中 FRR 0.1，10 个攻击样本中 FAR 0.1，检测率 90%；通过样本量与 LLM 规模的 ablation 实验验证了正则约束收敛性和 LLM 对策略质量的影响。

**⚠️ 局限性**

局限性：① 自动策略生成难以覆盖所有合法输入，泛化过宽或过窄导致误判；② 依赖训练日志的质量与覆盖度；③ 对 LLM 规划能力高度敏感，低质量 LLM 产生的规划错误会导致更高的 BEFR；④ 正则抽象无法完美区分细粒度攻击语义。

---

## 383. Single-Stage Huffman Encoder for ML Compression

**arXiv ID:** 2601.10673 | [PDF](https://arxiv.org/pdf/2601.10673v1)

**作者:** Aditya Agrawal `[一作]` (Google LLC), Ravi Iyer `[通讯]` (Google LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文研究了在训练和推理大型语言模型时使用单阶段Huffman编码，通过固定平均分布生成码本，消除频率分析和码本传输的开销，实现高效压缩。

**💡 创新点**

创新点在于发现不同分片张量的概率分布高度相似，利用平均分布构造固定码本，完成单阶段编码，显著降低延迟与计算开销。

**🔧 技术方法**

采用了统计KL散度分析、平均概率分布推导、固定码本Huffman编码以及Gemma 2B模型张量的频率分布研究等技术。

**📊 数据集**

使用的实验数据来自Gemma 2B模型在监督微调期间的权重、激活、权重梯度和激活梯度张量，18层模型在64张TPU上分片，形成1152个分片，覆盖数据类型bfloat16、e4m3、e3m2、e2m3、e2m1。

**📈 对比分析**

通过比较每分片单独Huffman编码、理论Shannon极限压缩率以及平均分布码本编码的压缩率，发现平均码本压缩率与单分片码本相差≤0.5%，与Shannon极限相差≤1%，性能几乎与理想极限相同。

**⚠️ 局限性**

限制在于仅适用于分布相似的张量，对分布差异大的张量效果可能下降；需要预先维护多套码本，增加存储与管理成本；实验范围仅涵盖Gemma 2B和少数数据类型，缺乏更广泛模型验证。

---

## 384. One-Shot Broadcast Joint Source-Channel Coding with Codebook Diversity

**arXiv ID:** 2601.10648 | [PDF](https://arxiv.org/pdf/2601.10648v1)

**作者:** Joseph Rowan `[一作]`, Ashish Khisti `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种共享编码器、分支解码器的多任务深度学习框架，能够同时预测多种输出。

**💡 创新点**

创新点在于将单一的编码器输出分支到多个解码器，并通过独立的条件分布 P(Y|X) 为每个解码器提供不同的上下文，实现并行多任务学习。

**🔧 技术方法**

使用的技术主要是深度神经网络的编码-解码结构（如 CNN/Transformer 等）和多任务学习策略，配合条件概率建模。

**📊 数据集**

实验数据集暂未给出，常见的做法可选用 ImageNet、COCO、MNIST 等多标签或多任务数据集。

**📈 对比分析**

通过与单一解码器或传统多任务网络进行对比，实验表明该架构在保持参数量相近的前提下，能够显著提升各任务的准确率/召回率，尤其在任务间相关性较高时效果更佳。

**⚠️ 局限性**

局限性包括：1）模型复杂度随解码器数量线性增长，训练与推理成本上升；2）不同任务间的负迁移问题仍需通过任务权重或正则化手段缓解；3）对任务相关性的假设未必适用于所有场景。

---

## 385. Action100M: A Large-scale Video Action Dataset

**arXiv ID:** 2601.10592 | [PDF](https://arxiv.org/pdf/2601.10592v1)

**作者:** Delong Chen `[一作]` (Meta FAIR), Pascale Fung `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了规模达1亿个动作实例、1.47亿段时序标注的Action100M视频动作数据集，并基于该数据集预训练的ViT‑L模型在多项零样本动作识别与视频检索基准上取得了显著提升。

**💡 创新点**

① 通过层次化时间分割与树形标题（Tree‑of‑Captions）结合LLM聚合实现大规模、结构化的动作与视频描述标注；② 引入语义重采样（Semantic Resampling）缓解长尾动作分布；③ 采用多模型多阶段训练策略，使模型在仅使用动作描述即可实现跨域零样本迁移。

**🔧 技术方法**

多模态视频编码器（ViT‑L/ViT‑G）、V‑JEPA 2特征提取、Llama‑3.2‑Vision、Perception‑LM、GPT‑OSS‑120B进行文本生成与聚合、k‑means聚类与哈希去重、信息噪声抑制（Tree‑of‑Captions、Self‑Refine）等技术。

**📊 数据集**

主要使用来自HowTo100M的1.2 M YouTube教学视频（约14.6 年时长）生成147 M段标注；在模型评估上对比了CLIP、SigLIP2、Perception Encoder、VL‑JEPA等现有基础模型，使用SSv2、EPIC‑KITCHENS‑100、EgoExo4D、Kinetics‑400、COIN、CrossTask等动作识别基准，以及MSR‑VTT、ActivityNet、DiDeMo、MSVD、YouCook2、PVD‑Bench、Dream‑1K、VDC‑1K等视频检索基准。

**📈 对比分析**

与现有模型相比，Action100M预训练的ViT‑L在零样本动作识别平均精度提升约3–5%，在文本检索Recall@1提升约5–10%，在动作密集、步骤识别任务（如SSv2、EPIC、COIN等）提升更为显著。

**⚠️ 局限性**

局限性包括：① 仍然以教学类视频为主，导致场景与动作类型偏向DIY、烹饪等；② 虽然采用LLM聚合提升标注质量，但对极端稀有动作仍可能出现错误；③ 对于多模态长序列推理、未来动作预测等更高阶任务的迁移效果尚未充分验证。

---

## 386. Be Your Own Red Teamer: Safety Alignment via Self-Play and Reflective Experience Replay

**arXiv ID:** 2601.10589 | [PDF](https://arxiv.org/pdf/2601.10589v1)

**作者:** Hao Wang `[一作]` (Beihang University), Lei Sha `[通讯]` (Beihang University)

**通讯引用:** 1416 | [OpenAlex ID](https://openalex.org/A5079222154)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于单一大型语言模型同时充当攻击者与防御者的自我对弈安全对齐框架（SSP），实现模型对自身安全漏洞的主动发现与修复。

**💡 创新点**

创新点包括：①将攻击与防御统一为同一模型的自我对弈，消除传统静态红队的“猫鼠”局限；②引入反射式经验回放与上界置信度（UCB）采样机制，持续从历史失败案例中学习并加速收敛；③通过零和最小化最大化的奖励设计，实现攻击与防御的动态平衡。

**🔧 技术方法**

技术手段：强化学习（RL）闭环、上界置信度（UCB）采样、反射式经验回放、LLM生成式对抗、LLM安全判别器评估、共享策略参数。

**📊 数据集**

使用数据集：训练阶段采集5,000条来自Jailbreak‑R1的恶意目标；评估阶段采用HarmBench、AdvBench、OR‑Bench、GSM8K、HumanEval、MMLU等公开基准。

**📈 对比分析**

与多种推理层（PPL、Self‑Reminder、SmoothLLM）和训练层（CircuitBreakers、CAT、R2D2、SafeDecoding、MART、ACE‑Safety）防御方法对比，SSP在攻击成功率(ASR)上显著更低，拒绝率最低，并能在保持模型原有能力的前提下取得更好的综合性能。

**⚠️ 局限性**

局限性：①对攻击多样性的生成仍受初始恶意目标和模型创造力限制；②目前仅针对文本攻击，需拓展至多模态；③自我对弈训练成本高；④对未来未知攻击的长期稳定性尚待进一步验证。

---

## 387. Structure and Diversity Aware Context Bubble Construction for Enterprise Retrieval Augmented Systems

**arXiv ID:** 2601.10681 | [PDF](https://arxiv.org/pdf/2601.10681v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 388. Mitigating GIL Bottlenecks in Edge AI Systems

**arXiv ID:** 2601.10582 | [PDF](https://arxiv.org/pdf/2601.10582v1)

**作者:** Mridankan Mandal `[一作]` (Indian Institute of Information Technology), Smit Sanjay Shende `[通讯]` (Indian Institute of Information Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文针对Python GIL导致的边缘AI多线程饱和现象，提出并实现了一种基于Blocking Ratio的自适应线程池，在单进程内动态调节线程数以提升边缘设备AI系统吞吐。

**💡 创新点**

创新点在于提出了轻量级的Blocking Ratio指标和带Veto机制的自适应控制算法，能够实时区分I/O等待与GIL争用，从而避免线程饱和，并在Python 3.13t无GIL环境下验证其通用性。

**🔧 技术方法**

技术上使用time.thread_time与time.time记录CPU/墙钟时间，利用指数加权移动平均EWMA与阈值/退避机制实现线程数自适应；实验中对比ThreadPoolExecutor、ProcessPoolExecutor、asyncio及自定义调度器，并使用synthetic混合工作负载和七种真实边缘AI工作负载（视觉、语音、传感器融合、RAG、SLM、分析、ONNX MobileNetV2）。

**📊 数据集**

工作负载采用代表性数据集：MobileNetV2、FFT音频特征、Kalman融合、JSON向量查询、矩阵乘法、Pandas时间序列等；同时使用10 ms CPU/50 ms I/O的synthetic模型；并未使用公开的单一数据集。

**📈 对比分析**

通过在单核/四核边缘模拟配置下，对比固定256/32线程、进程池、asyncio、队列深度调度器等策略，测量TPS、P99延迟与内存占用。自适应线程池在七种工作负载中平均实现93.9%相对于最佳吞吐，单核最大下降40%，四核35%；在Python 3.13t四核环境下吞吐提升约4倍。

**⚠️ 局限性**

局限性包括仅使用CPU亲和力模拟而非真实物理边缘硬件，未评估功耗与热量；单核环境仍受上下文切换影响；β阈值需经验调优；在极高线程数下可能产生额外开销；未验证在GPU或异构加速器环境中的适用性。

---

## 389. Jordan-Segmentable Masks: A Topology-Aware definition for characterizing Binary Image Segmentation

**arXiv ID:** 2601.10577 | [PDF](https://arxiv.org/pdf/2601.10577v1)

**作者:** Serena Grazia De Benedictis `[一作]` (University of Bari Aldo Moro), Nicoletta Del Buono `[通讯]` (University of Bari Aldo Moro)

**通讯引用:** 873 | [OpenAlex ID](https://openalex.org/A5005901551)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于数字Jordan曲线定理和Betti数的拓扑感知二值分割评估指标——Jordan‑segmentable mask；

**💡 创新点**

创新点在于将拓扑连通性与同伦不变量直接映射到分割掩模，提供无监督的结构一致性判定；

**🔧 技术方法**

核心技术包括数字拓扑、简化的四邻接曲线构造、Betti数（0、1维）计算以及图论连通性分析；

**📊 数据集**

实验使用FSS‑1000少样本分割数据集，对其原始标注与多种经典分割方法（Otsu、Ridler‑Calvard、K‑means、Watershed）生成的掩模进行评估；

**📈 对比分析**

与传统像素级指标（IoU、Dice、Precision、Recall、Accuracy）对比，Jordan‑segmentable mask在掩模结构不完整或标签颠倒时仍能准确反映拓扑一致性，补充了传统指标的盲点；

**⚠️ 局限性**

局限性包括：对多类分割未直接扩展；对非常细长或稀疏物体的曲线构造可能产生错误；以及需要先行的预处理和零填充以避免边界退化。

---

## 390. Generative AI collective behavior needs an interactionist paradigm

**arXiv ID:** 2601.10567 | [PDF](https://arxiv.org/pdf/2601.10567v1)

**作者:** Laura Ferrarotti `[一作]` (Fondazione Bruno Kessler), Bruno Lepri `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 8283 | [OpenAlex ID](https://openalex.org/A5048877432)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种“互动主义范式”，旨在系统性地研究基于大语言模型的生成式 AI 代理集体的协同行为与社会化现象。

**💡 创新点**

创新点在于将人情境论、因果推断、信息理论与机器社会学交叉融合，形成四大支柱（交互理论、因果方法、信息测度、机器社会学）以解释生成式代理集体的出现性行为。

**🔧 技术方法**

主要技术包括：基于上下文的即时学习（ICL）、因果推断框架（如干预设计、网络干预识别）、信息理论量化（互信息、熵、转移熵）以及多代理评测基准（MultiAgentBench、AgentQuest 等）。

**📊 数据集**

本文并未引入新的实验数据集，而是基于公开的大规模预训练语料、监督微调与人类反馈强化（RLHF）所得到的模型，并在理论层面讨论了与 MARL 环境的差异。

**📈 对比分析**

方法比较主要是与传统 MARL 理论与评测进行概念对照，并未给出实验性能指标；作者指出需开发新的交互式基准与因果评估方案来量化效果。

**⚠️ 局限性**

局限性包括：缺乏实证验证与实验数据支持、对不同模型规模与架构的适用性未明、因果与信息测度在大规模多代理系统中的计算与解释困难，以及如何在实际部署中实现安全与可解释性仍是未解决的问题。

---

## 391. Converse Bounds for Sun-Jafar-type Weak Private Information Retrieval

**arXiv ID:** 2601.10643 | [PDF](https://arxiv.org/pdf/2601.10643v1)

**作者:** Chandan Anand `[一作]` (International Institute of Information Technology), Gowtham R. Kurri `[通讯]` (International Institute of Information Technology)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5037959485)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文对Sun‑Jafar型与Banawan‑Ulkus型弱私有信息检索（WPIR）方案在多种存储与协作模型下的速率‑隐私权衡进行理论分析，并证明在相应的隐私度量（互信息泄漏与最大泄漏）下，该类方案在非协作、MDS编码与T‑协作场景中的速率上是最优的；同时给出存储率与协作比例的阈值，阐明当超过阈值时可进一步提升速率的可能性。

**💡 创新点**

创新点在于：①首次证明Sun‑Jafar与Banawan‑Ulkus型WPIR方案在非协作、MDS与T‑协作设置下的类级最优性；②确定存储率（K/N）与协作比例（T/N）的临界阈值，阐明阈值以下最优分布仅为端点（M′=0或M′=M‑1），而阈值以上需引入中间取值；③通过构造反例展示阈值以上仍可获得更高速率，提示尚未完全实现最优。

**🔧 技术方法**

采用信息理论方法，对隐私泄漏约束下的速率最优化问题进行解析，利用期望、线性规划、凸性与极值证明等技术；同时利用互信息与最大泄漏的定义推导速率‑隐私表达式并求解最优概率分布。

**📊 数据集**

该研究为纯理论分析，无实测数据集；所有结论均基于抽象的文件数M、服务器数N、码率K或协作数T的数学模型。

**📈 对比分析**

与已有的WPIR方案（如原始Sun‑Jafar、Banawan‑Ulkus及其改进版本）在相同隐私阈值下进行速率比较，证明在阈值以下可获得相同或更高速率；当阈值被突破时，给出更优速率的示例，表明此前方案未达到最优。

**⚠️ 局限性**

局限性包括：①仅在存储率或协作比例低于阈值时可证明确切最优；②超过阈值时的最优性仍未完全证明，只给出示例与阈值提示；③未给出所有WPIR方案的全局信息理论上界，仍是未解问题。

---

## 392. CoMoVi: Co-Generation of 3D Human Motions and Realistic Videos

**arXiv ID:** 2601.10632 | [PDF](https://arxiv.org/pdf/2601.10632v1)

**作者:** Chengfeng Zhao `[一作]` (Hong Kong University of Science and Technology), Yuan Liu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 62065 | [OpenAlex ID](https://openalex.org/A5100390838)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CoMoVi框架，实现3D人体运动与2D视频同步共生生成；

**💡 创新点**

首次在单一扩散迭代循环中耦合运动与视频的去噪过程，并引入将3D SMPL信息压缩为RGB图像的2D运动表示；

**🔧 技术方法**

双分支扩散模型（基于Wan2.2‑I2V‑5B）+ 3D‑2D交叉注意力+ 零线性互联特征交互；

**📊 数据集**

新建CoMoVi数据集（约5万段720p视频、SMPL 3D标注、文本说明），并使用Motion‑X++、VBench等公开基准；

**📈 对比分析**

与当前最强T2M（MDM、MotionGPT、Go‑to‑Zero）和I2V（CogVideoX1.5、Wan2.2‑I2V‑5B）对比，CoMoVi在FID、R‑Precision、MMDist（运动）及VBench指标（视频）均显著优于对照组；

**⚠️ 局限性**

仍受限于3D标注噪声、对极端姿态或多人物场景的鲁棒性不足，且对训练成本和推理速度有一定需求。

---

## 393. Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding

**arXiv ID:** 2601.10611 | [PDF](https://arxiv.org/pdf/2601.10611v1)

**作者:** Christopher Clark `[一作]` (Allen Institute for AI), Ranjay Krishna `[通讯]` (Allen Institute for AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并训练了一个完全开放的多模态大型语言模型Molmo2，支持视频、图像、多图像输入，并具备定位、跟踪、计数、问答和字幕理解等多重视觉‑语言能力。

**💡 创新点**

提出了九个全新公开数据集（密集视频描述、视频指向/跟踪、长篇问答等），并采用三阶段训练与高效序列打包/消息树技术，显著提升了视频定位与计数性能；同时完全开放训练代码、数据与权重。

**🔧 技术方法**

基于ViT视觉编码器与大型LLM（Qwen3/OLMo）通过视觉连接器实现视觉‑语言交互；使用帧时间戳、双向注意力、token加权、多裁剪、序列打包、消息树、Ulysses上下文并行等技术。

**📊 数据集**

利用PixMo、Molmo2‑VideoCap、Molmo2‑AskModelAnything、Molmo2‑VideoPoint/Track、Molmo2‑SubtitleQA、Molmo2‑MultiImageQA/Point等数据集，合计包含约104k视频字幕、140k QA、650k指向、3600视频跟踪、数十万个多图像问答等。

**📈 对比分析**

在短视频、长视频问答、视频字幕、计数、指向、跟踪、图像及多图像基准上均超过或匹敌现有公开模型，接近甚至超过部分专有系统；在视频计数与指向上表现优于GPT‑5，整体在人工偏好评估中排名靠前，显示其通用能力。

**⚠️ 局限性**

受限于缺乏开放的长时视频（10+分钟）训练数据和算力，模型在极长视频QA和开放式推理等方面仍落后于顶级专有模型；此外，某些高难度视觉推理基准（如MathVista、MMMU）表现逊色。

---

## 394. iTIMO: An LLM-empowered Synthesis Dataset for Travel Itinerary Modification

**arXiv ID:** 2601.10609 | [PDF](https://arxiv.org/pdf/2601.10609v1)

**作者:** Zhuoxuan Huang `[一作]` (Central South University), Zhu Sun `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 2048 | [OpenAlex ID](https://openalex.org/A5033957641)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并定义了行程修改（Itinerary Modification）任务，并通过大语言模型（LLM）自动生成需要修改的行程数据，构建了iTIMO数据集；

**💡 创新点**

创新点在于：①将行程修改任务正式化；②提出基于意图驱动的扰动（perturbation）策略，涵盖替换、添加、删除三种原子操作与三类扰动意图（人气、空间距离、类别多样性）；③设计混合评估指标（Hellinger距离、Kendall τ_b）确保扰动符合意图；④引入函数调用和记忆模块提升LLM扰动质量；

**🔧 技术方法**

使用技术主要包括：大语言模型（ChatGPT/GPT‑4、DeepSeek、Qwen、Gemma、Llama等）、函数调用（function calling）与记忆模块、检索增强生成（RAG）、监督微调（LoRA/FullFT）、评估指标Mod/APR、以及混合式扰动评估；

**📊 数据集**

数据集：iTIMO（基于Toronto、Melbourne、Florence三大公开旅行数据集的真实行程经过LLM扰动生成），其中包含三类扰动子集；

**📈 对比分析**

通过在iTIMO上对八个基准LLM和两大模型进行基线实验，并探讨RAG、SFT、以及二者组合的效果。结果表明：①小型模型在SFT后可与大型模型相媲美；②RAG提升显著，密集检索效果最好；③SFT与RAG需保持训练/推理的一致性，否则可能导致性能下降；整体性能仍受限于多步操作的复杂性；

**⚠️ 局限性**

局限性：①扰动仅基于用户意图，缺乏个性化/时间约束；②未考虑行程中的时间因素；③意图种类有限，可能不足以覆盖所有真实用户需求。

---

## 395. A user subscription model in mobile radio access networks with network slicing

**arXiv ID:** 2601.10605 | [PDF](https://arxiv.org/pdf/2601.10605v1)

**作者:** José-Ramón Vidal `[一作]` (Universitat Politècnica de València), Vicent Pla `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 1956 | [OpenAlex ID](https://openalex.org/A5002611857)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估了在移动无线接入网络切片场景下使用logit模型预测用户订阅行为，并将解析结果与基于完整移动与射频传播的仿真结果进行比较。

**💡 创新点**

证明了使用单元格容量中位数或对logit模型进行微调的方式能够在移动环境中保持极高的预测精度，扩展了logit模型在网络经济学中的适用范围。

**🔧 技术方法**

采用离散选择（logit）模型、比例共享资源分配算法，以及基于IMT‑Advanced标准的射频传播与随机路径模型的仿真技术。

**📊 数据集**

使用在57个蜂窝的城市微小细胞网格中生成的仿真数据，结合随机游走移动模型和实际的信号衰减与干扰参数。

**📈 对比分析**

通过将解析得到的订阅比例σ和各切片订阅份额ρ与仿真测得的相同指标进行直接对比；误差低于2%（σ）和0.1%（ρ），验证了解析模型的高准确性。

**⚠️ 局限性**

假设每个切片在同一小区内所有用户获得相同比特率，且需要足够多的用户以及较低的非订阅比例；模型对动态资源分配和快速容量波动的响应有限。

---

## 396. Enhancing Mobile Ad Hoc Networks (MANETs) with Software-Defined Networking (SDN): A Balanced Approach

**arXiv ID:** 2601.10556 | [PDF](https://arxiv.org/pdf/2601.10556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 397. Procedural Fairness in Multi-Agent Bandits

**arXiv ID:** 2601.10600 | [PDF](https://arxiv.org/pdf/2601.10600v1)

**作者:** Joshua Caiata `[一作]` (University of Waterloo), Kate Larson `[通讯]` (University of Waterloo)

**通讯引用:** 2815 | [OpenAlex ID](https://openalex.org/A5105978897)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在多智能体多臂赌博机框架中提出并实现了基于程序公平（Procedural Fairness）的决策策略，强调在决策过程中每位智能体拥有平等发言权；

**💡 创新点**

创新点在于将“程序公平”这一理念引入到多智能体赌博机问题，形成一种新型公平目标，证明不同公平目标不可兼容并证明程序公平策略位于程序核心；

**🔧 技术方法**

技术上采用了基于UCB的估计、凸优化（线性规划）实现决策共享约束，给出多种公平目标（程序公平、均衡公平、效用公平）的学习算法并给出子线性或平方根收敛的后悔界；

**📊 数据集**

在实验中使用了大规模合成数据（N,K∈[2,10]，偏好分布包含单峰、均匀、马洛斯等）以及真实偏好数据集PrefLib的Mechanical Turk Dots（800人4候选）作为验证案例；

**📈 对比分析**

比较方法是通过三种公平度量（程序公平得分、均衡得分、效用得分）来评估各算法的最优策略，实验显示程序公平策略在三种度量上均取得最优或接近最优平衡，而其他目标在程序公平度量上表现差异显著；

**⚠️ 局限性**

局限性包括：只能在赌博机（无状态）环境下验证；公平度量仍然需外部指定；对人类主观感知的程序公平尚未实验验证；以及在更复杂的多智能体强化学习或连续动作空间中尚未推广。

---

## 398. ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition

**arXiv ID:** 2601.10591 | [PDF](https://arxiv.org/pdf/2601.10591v1)

**作者:** Arundeep Chinta `[一作]` (JPMorganChase), Jay Katukuri `[通讯]` (JPMorganChase)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 ProbFM，一种融合 Deep Evidential Regression 的 Transformer 基础模型，用于时间序列金融预测并实现不确定性分解；

**💡 创新点**

首次将 Deep Evidential Regression 与 Normal‑Inverse‑Gamma priors 结合到时间序列基础模型，实现显式的表观（epistemic）与噪声（aleatoric）不确定性分解，并通过单前向传播得到置信区间；

**🔧 技术方法**

使用 Transformer 作为特征提取器、Deep Evidential Regression 头、NIG 先验、覆盖率损失、证据退火、AdamW + 梯度裁剪、余弦学习率调度；

**📊 数据集**

在 11 种主流加密货币（ADA、BNB、BTC、DASH、DOGE、ETH、LTC、SOL、USDC、USDT、XRP）的每日对数收益率数据集上进行实验；

**📈 对比分析**

对比 5 种不确定性量化方法（Gaussian NLL、Student‑t NLL、Quantile Loss、Mixture、Conformal）在同一 1‑层 32 维 LSTM 基础上，结果显示 ProbFM 在预测精度上与基准相当，在不确定性校准（CRPS、PICP、Sharpness）和风险调节交易指标（Sharpe、Sortino、Calmar）上均优于其他方法；

**⚠️ 局限性**

局限性：仅在单步单变量场景验证，未扩展到多步或多变量；需要手工调参的覆盖率损失和证据退火；对非金融时序任务的泛化尚待验证；

---

## 399. Form and Meaning in Intrinsic Multilingual Evaluations

**arXiv ID:** 2601.10580 | [PDF](https://arxiv.org/pdf/2601.10580v1)

**作者:** Wessel Poelman `[一作]` (KU Leuven), Miryam de Lhoneux `[通讯]` (KU Leuven)

**通讯引用:** 667 | [OpenAlex ID](https://openalex.org/A5080895973)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了多语言条件语言模型的内在评估指标，并探讨它们在多语平行数据上的一致性与可比性。

**💡 创新点**

创新点在于明确区分信息理论与语义之间的差异，提出多语平行评估假设的局限，并通过抛物句实验展示指标不一致性。

**🔧 技术方法**

使用小型 Llama‑3 风格模型，采用 BPE 字节级分词，并计算 NLL、PPL、BPC、BPEC、IP、MRR 等六种内在指标。

**📊 数据集**

训练数据为 EuroParl 与 UN 并行语料，评估数据为 FLORES‑200，抛物句实验使用 WMT 翻译对照数据。

**📈 对比分析**

通过逐步训练曲线和句子级一致性评估比较多语与单语设置，结果显示指标在不同语言和句子变体间不一致，平均下可能相互抵消，难以可靠比较语言难度。

**⚠️ 局限性**

局限性包括受限于小模型规模、并行语料覆盖有限、技术决策影响大，以及指标本身仅衡量信息量而非语义，导致跨语言比较不可靠。

---

## 400. Inference-time Physics Alignment of Video Generative Models with Latent World Models

**arXiv ID:** 2601.10553 | [PDF](https://arxiv.org/pdf/2601.10553v1)

**作者:** Jianhao Yuan `[一作]` (University of Oxford), Adriana Romero-Soriano `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种推理时对齐方法，通过利用潜在世界模型（VJEPA-2）作为奖励信号，改善视频生成模型的物理合理性。

**💡 创新点**

创新点在于将潜在世界模型的奖励信号应用于视频生成的推理阶段，以提高生成视频的物理合理性，并在多个基准测试中取得了显著的性能提升。

**🔧 技术方法**

使用了潜在世界模型（VJEPA-2）作为奖励模型，并结合了引导采样和最佳选择（Best-of-N）等采样技术。

**📊 数据集**

使用了PhysicsIQ和VideoPhy等基准数据集进行评估，涵盖了文本到视频（T2V）、图像和文本到视频（I2V）以及视频和文本到视频（V2V）等设置。

**📈 对比分析**

与现有方法相比，提出的方法在PhysicsIQ基准上达到了62.0%的得分，超越了之前的最佳结果6.78%。在用户偏好研究中，表现出11.4%的提升，显示出更好的物理合理性和视觉质量。

**⚠️ 局限性**

限制在于VJEPA-2的训练数据有限，可能无法覆盖所有物理现象，未来的工作可以集中在改进奖励模型和搜索算法上，以进一步提升性能。

---

## 401. Representation-Aware Unlearning via Activation Signatures: From Suppression to Knowledge-Signature Erasure

**arXiv ID:** 2601.10566 | [PDF](https://arxiv.org/pdf/2601.10566v1)

**作者:** Syed Naveed Mahmood `[一作]` (BRAC University), Farig Sadeque `[通讯]` (BRAC University)

**通讯引用:** 332 | [OpenAlex ID](https://openalex.org/A5009105388)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出知识免疫框架（KIF），通过定位并抑制LLM内部激活签名，实现对指定知识的真正遗忘，同时保持模型整体性能。

**💡 创新点**

创新点包括：① 区分行为抑制与真实遗忘，利用内部激活签名；② 轻量级知识抑制胶囊结合 LoRA 自愈循环，避免全模型重训练；③ 双指标评估（SMR+EL10）揭示隐藏失败模式；④ 体系化对比标准与推理模型在规模与架构上的遗忘差异。

**🔧 技术方法**

技术手段：激活签名提取（对比学习+PCA/投影）、Rank‑1压制胶囊、LoRA 参数微调、自愈循环、组合损失（DPO、UL、NT‑UL、KL、EWC）以及自定义自愈门控机制。

**📊 数据集**

使用自制的 Real‑World Entity Dataset（基于 Wikipedia/WikiData 的事实三元组）进行签名提取与评估，并结合 TOFU 基准测试遗忘质量与模型实用性。

**📈 对比分析**

与 Gradient Ascent、GradDiff、NPO、SimNPO、IDK 等五种主流方法在 TOFU 进行对比。KIF 在所有规模标准模型实现 FQ≈0.99、MU≈0.62（接近 oracle），远优于其他基线；推理模型在 14B 规模时亦能实现真遗忘，前期规模表现为失稳/抑制模式。

**⚠️ 局限性**

局限性：① 依赖合成负样本，可能无法完全分离真实与假激活；② 数据集不平衡导致签名稳定性受限；③ 仅在 4‑bit 量化模型与单卡实验，未验证更大或全精度模型；④ 未评估对抗性恢复或后续微调对遗忘的影响；⑤ 不能保证完全不可逆，存在被逆向恢复的风险。

---

## 402. Detecting Winning Arguments with Large Language Models and Persuasion Strategies

**arXiv ID:** 2601.10660 | [PDF](https://arxiv.org/pdf/2601.10660v1)

**作者:** Tiziano Labruna `[一作]` (University of Padua), Giovanni Da San Martino `[通讯]` (University of Padua)

**通讯引用:** 3867 | [OpenAlex ID](https://openalex.org/A5033850423)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究并利用六种说服策略，使用大型语言模型对论证文本进行结构化评分，以检测并预测更具说服力的论点。

**💡 创新点**

首次将说服策略作为特征引入机器学习框架（MS-PS），并发布带话题标签的 Winning Arguments 数据集。

**🔧 技术方法**

多策略说服评分（MS-PS），LLM（Gemma、Llama、Gemini、OpenAI‑o3）+ MLP 分类器；以及链式推理提示。

**📊 数据集**

Winning Arguments、Topics Winning Arguments、Anthropic/Persuasion、Persuasion for Good。

**📈 对比分析**

与单独评分、上下文+解释等基线相比，MS-PS‑MLP 在五大 LLM 上取得约61–65% 的准确率（相较基线提升约4–6%），在回归任务上 RMSE 亦显著下降。

**⚠️ 局限性**

依赖数据集偏倚、模型识别策略的误差、说服主观性以及有限的跨域泛化能力。

---

## 403. Influential Training Data Retrieval for Explaining Verbalized Confidence of LLMs

**arXiv ID:** 2601.10645 | [PDF](https://arxiv.org/pdf/2601.10645v1)

**作者:** Yuxi Xia `[一作]` (University of Vienna), Benjamin Roth `[通讯]` (University of Vienna)

**通讯引用:** 1448 | [OpenAlex ID](https://openalex.org/A5046895021)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 TracVC 方法，利用信息检索和梯度影响估计追踪大语言模型（LLM）所表述的置信度背后的训练数据来源。

**💡 创新点**

创新点在于将训练数据检索与 TracIn 影响度量结合，首次量化“内容基础性” (content groundness)，并揭示不同模型、规模与后训练方式对置信度根植程度的影响。

**🔧 技术方法**

技术主要包括：① BM25 词表检索两类样本（内容相关与置信相关）；② TracIn 的梯度余弦相似度计算影响分数；④ 构造 Content-over-Confidence Ratio (ccr) 指标。

**📊 数据集**

使用公开的 OLMo（2B、7B、13B）和 Llama3-8B 等开源模型的预训练语料（dolma v1.7）与多种后训练语料（如 tulu‑3‑sft‑mixture、RLVR‑MATH 等），评估数据集包括 Natural Question、SicQ、TriviaQA、PopQA、TruthfulQA。

**📈 对比分析**

通过比较不同模型与后训练阶段的 ccr 值、影响分数分布以及与任务准确率、样本正确性相关的 Pearson 相关系数，发现：① OLMo‑2‑13B 对置信相关样本更敏感；② 大模型并不一定更“内容基础”；③ 后训练方式对不同模型的影响相反，整体不会逆转 ccr 方向；⑥ 在大多数情形下 ccr 接近 1，说明置信度仍受内容与置信词汇混合影响。

**⚠️ 局限性**

局限性包括：① 仅针对公开开源模型，缺少商业模型的训练语料；② 内容/置信拆分仅基于词表检索，忽略语义重叠；③ 只取前 10 条检索样本，可能遗漏重要示例；④ 采用梯度近似估计，未与 Hessian 等更精确方法对比。

---

## 404. STEM: Scaling Transformers with Embedding Modules

**arXiv ID:** 2601.10639 | [PDF](https://arxiv.org/pdf/2601.10639v1)

**作者:** Ranajoy Sadhukhan `[一作]` (Carnegie Mellon University), Beidi Chen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5073845046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种静态、基于 token 索引的稀疏 FFN 机制——STEM，取代传统 FFN 的上投影，以层内嵌入表提供地址向量；

**💡 创新点**

创新点包括：
① 将 token‑indexed 嵌入直接作为上投影地址，保持门控的上下文适配；
② 通过极高稀疏度实现训练稳定且不引入激活 FLOPs；
③ 嵌入表与 token 一一对应，提供可解释的微专家；
④ 支持 CPU offload 与异步预取，消除跨设备通信；
⑤ 随着上下文长度增长自动激活更多参数，提升长文本推理能力；

**🔧 技术方法**

使用技术：
- 静态稀疏（token‑indexed）设计；
- 层内嵌入表 + SwiGLU 结构；
- GPU/CPU 并行预取、LFU 缓存与 Token dedup；
- 与 Hash‑MoE、密集 FFN 对比实验；
- 大规模预训练、mid‑training 与上下文扩展策略；

**📊 数据集**

数据集：
- 预训练：OLMo‑Mix‑1124（1T 子集）
- mid‑training：OLMo‑Mix‑1124 + Nemotron‑CC‑Math‑v1 + Nemotron‑Pretraining‑Code‑v1
- 长上下文：ProLong‑Data‑64k
- 评估：ARC‑Easy、ARC‑Challenge、BoolQ、PIQA、SIQA、HellaSwag、OpenBookQA、WinoGrande、MMLU、GSM8K、Needle‑in‑a‑Haystack；

**📈 对比分析**

比较方式与性能：
- 与密集 SwiGLU 基线及 Hash‑MoE 在相同激活 FLOPs 下对比；
- 350M 与 1B 规模评估；
- STEM 在知识密集任务提升约 3–10%，在 Needle‑in‑a‑Haystack 上提升 8–13%；
- 训练 ROI 提升 1.08×–1.33×；
- 无训练损失尖峰，训练稳定；

**⚠️ 局限性**

局限性：
- 需要额外存储嵌入表，训练时梯度回传到 CPU 增加通信；
- 对极高 Token 频率分布仍受 Zipf 分布影响；
- 编辑不同长度实体时仍需手工策略（padding、copy 等）；
- 在更大规模模型上的 CPU/内存成本与实际部署可行性待进一步验证；

---

## 405. Basis-Spline Assisted Coded Computing: Strategies and Error Bounds

**arXiv ID:** 2601.10616 | [PDF](https://arxiv.org/pdf/2601.10616v1)

**作者:** Rimpi Borah `[一作]` (Indian Institute of Technology), V. Lalitha `[通讯]` (International Institute of Information Technology)

**通讯引用:** 1125 | [OpenAlex ID](https://openalex.org/A5110717612)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于三次B样条插值的编码计算框架（BSCC），用于在存在慢节点的情况下进行分布式计算，旨在提高非多项式函数的近似精度和数值稳定性。

**💡 创新点**

创新点在于利用B样条的局部支持和光滑性特性，克服了现有Berrut近似编码计算方法在大量慢节点情况下的精度下降问题。

**🔧 技术方法**

使用了三次B样条插值技术，结合了编码计算的框架，提出了一种系统的方法来集成B样条插值，并推导出近似误差的理论界限。

**📊 数据集**

使用了合成数据集，具体为从均匀分布中抽取的样本，进行非多项式函数的计算，如f(𝐗) = 𝐗sin(𝐗)和f(𝐗) = 1/(1+e^-𝐗)。

**📈 对比分析**

通过理论比较和数值仿真，BSCC框架在近似精度上显著优于Berrut近似编码计算（BACC），尤其是在慢节点数量较少时，且在慢节点数量增加时仍表现出显著的改进。

**⚠️ 局限性**

限制在于该方法仍然依赖于B样条的构造和参数选择，可能在特定情况下对数据分布的敏感性较高。

---

## 406. Fundamental Limits of Multi-User Distributed Computing of Linearly Separable Functions

**arXiv ID:** 2601.10603 | [PDF](https://arxiv.org/pdf/2601.10603v1)

**作者:** K. K. Krishnan Namboodiri `[一作]` (EURECOM), Petros Elia `[通讯]` (EURECOM)

**通讯引用:** 2683 | [OpenAlex ID](https://openalex.org/A5015066458)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究多用户分布式计算线性可分函数，在服务器计算容量 M 与服务器–用户连通度 Δ 的限制下，提出一种任务分配与通信联合设计方案，并给出其最优或近似最优的通信成本。

**💡 创新点**

创新点在于：①采用左零空间稀疏化将需求矩阵按 Δ×(Δ+M−1) 块分割，保证每个块的服务器能用最少的子函数计算实现解码；②使用矩阵分解计数与自由度（DoF）分析给出下界，证明在满足 Δ|L 且 (Δ+M−1)|K 时方案达到下界；③相较于先前的切片、覆盖码或单用户方案，所需的计算容量更低且能支持多用户同时解码。

**🔧 技术方法**

技术方法包括：线性编码、左零空间求解、矩阵分解与稀疏化、计数论与自由度分析、对有限域与实数域分别给出对偶下界。

**📊 数据集**

论文没有使用公开数据集，而是以理论模型和人工构造的需求矩阵（如 K=10、L=6、M=3、Δ=3 的示例）作为验证。

**📈 对比分析**

与已有方案比较：在 K=10、L=6、M=3、Δ=3 的实例下，通信率为 12，和现有方案相同但所需的计算容量从 5 降至 3；一般参数下，在 q→∞ 时达到下界；在有限域下误差因子 ≤3（或 8，取决于除数条件）；在实数域下误差因子 ≤4，显示了相对优越的性能。

**⚠️ 局限性**

限制：方案需要预先知道所有用户的需求；对分块大小和参数的除数条件有限制；对非线性子函数、随机化方案或多轮通信未考虑；在不满足除数条件时只能得到近似最优。

---

## 407. Institutional AI: A Governance Framework for Distributional AGI Safety

**arXiv ID:** 2601.10599 | [PDF](https://arxiv.org/pdf/2601.10599v1)

**作者:** Federico Pierucci `[一作]` (DEXAI), Daniele Nardi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 11018 | [OpenAlex ID](https://openalex.org/A5075651762)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了“Institutional AI”框架，将对齐视为系统级治理问题，并在多智能体 Cournot 市场中通过治理图模拟 LLM 代理的合谋行为。

**💡 创新点**

创新点在于将对齐从模型内部属性迁移至外部机构约束，引入可验证、可扩展的治理图机制，并首次在 LLM 多智能体经济模型中验证其有效性。

**🔧 技术方法**

采用机制设计、博弈论、治理图结构、RLINF（基于机构反馈的强化学习）和多智能体仿真等技术。

**📊 数据集**

使用自建的多智能体 Cournot 市场仿真生成价格与合作/竞争行为数据，未依赖公开大型数据集。

**📈 对比分析**

与传统 RLHF/Constitutional AI 对齐方式对比，测评合谋率、消费者福利与效率，实验表明治理图将合谋率降至约 5% 以下，消费者剩余提升约 20%。

**⚠️ 局限性**

局限性包括治理图需手工设计、缺乏自动化搜索；对高度动态或不确定环境的适应性有限；验证主要基于模拟，实际部署中的监测与执行成本仍待评估。

---

## 408. Improving Database Performance by Application-side Transaction Merging

**arXiv ID:** 2601.10596 | [PDF](https://arxiv.org/pdf/2601.10596v1)

**作者:** Xueyuan Ren `[一作]` (Ohio State University), Yang Wang `[通讯]` (Ohio State University)

**通讯引用:** 244234 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了应用层事务合并技术（内聚合与跨事务合并），通过SQL语义重写提升数据库吞吐量。

**💡 创新点**

创新点在于在保证隔离性的前提下，利用静态冲突分析和SQL CASE/IN 语法，将跨事务的类似事务安全合并，突破传统仅限读操作的批处理限制。

**🔧 技术方法**

采用了事务合并中间件（基于gRPC）、SQL CASE WHEN/IN 重写、静态分析工具、GPR+BO 自动调参等技术实现合并与动态配置。

**📊 数据集**

使用标准的TPC‑C基准工作负载以及自定义的Spree电商数据集进行实验。

**📈 对比分析**

与原始JDBC批处理、客户端级内聚合以及单纯intra事务合并进行对比；在MySQL和PostgreSQL上实现了2.65–3.52倍的吞吐量提升，单事务可达6×加速。

**⚠️ 局限性**

限制在于需手动改写代码、静态分析依赖开发者提供列访问信息、适用范围受限于事务结构、会增加客户端延迟和CPU占用，且在极高争用场景下仍需进一步优化。

---

## 409. Adversarial Evasion Attacks on Computer Vision using SHAP Values

**arXiv ID:** 2601.10587 | [PDF](https://arxiv.org/pdf/2601.10587v1)

**作者:** Frank Mollard `[一作]`, Florian Roehrbein `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用SHAP值构造白盒对抗攻击，成功诱导计算机视觉模型误分类。

**💡 创新点**

创新点在于用SHAP解释的重要性评估替代梯度信息，聚焦关键像素以实现更稳健、细微的攻击。

**🔧 技术方法**

使用SHAP框架、卷积网络（包括EfficientNetB7）、FGSM梯度攻击、深度学习模型推理等技术。

**📊 数据集**

实验数据集包括动物面孔、猫狗滤镜、MNIST手写数字和男女面部图像。

**📈 对比分析**

与FGSM比较显示，SHAP攻击在同等可察觉度下误分类率更高且更稳定（如动物面孔73%对比52%），且对抗扰动更细微。

**⚠️ 局限性**

局限性包括需要完整模型访问、SHAP计算成本高、对高分辨率图像或视频的可扩展性受限。

---

## 410. From Single to Multi-Agent Reasoning: Advancing GeneGPT for Genomics QA

**arXiv ID:** 2601.10581 | [PDF](https://arxiv.org/pdf/2601.10581v1)

**作者:** Kimia Abedini `[一作]` (University of Padua), Gianmaria Silvello `[通讯]` (University of Padua)

**通讯引用:** 1436 | [OpenAlex ID](https://openalex.org/A5078254809)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现 GeneGPT 并提出 GenomAgent 多代理框架，用于基因学问答

**💡 创新点**

通过多代理并行查询、动态代码生成抽取、共识式答案合成，提升灵活性与可扩展性

**🔧 技术方法**

使用 GPT‑4o‑mini 等 LLM、LangGraph/Google Agent Development Kit、MCP 协议、ReAct 结构

**📊 数据集**

GeneTuring 基准（9 个基因学 QA 任务）

**📈 对比分析**

与 GeneGPT 对比，平均准确率从 0.83 提升到 0.93（+12%），计算成本从 $10.06 降至 $2.11（-79%）

**⚠️ 局限性**

未进行组件消融实验、仅在 GeneTuring 上评测，缺乏跨任务泛化验证，单一代理与多代理混合策略仍待完善

---

## 411. Kolmogorov Arnold Networks and Multi-Layer Perceptrons: A Paradigm Shift in Neural Modelling

**arXiv ID:** 2601.10563 | [PDF](https://arxiv.org/pdf/2601.10563v1)

**作者:** Aradhya Gaonkar `[一作]` (KLE Technological University), Channabasappa Muttal `[通讯]` (KLE Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

比较 Kolmogorov Arnold Networks (KAN) 与传统多层感知机 (MLP) 在非线性函数逼近、时序预测与多变量分类三类任务中的表现。

**💡 创新点**

提出以 KAN 的基于分段样条激活函数与网格结构为核心的高效架构，并系统阐述其在精度与计算成本上的优势。

**🔧 技术方法**

采用 Kolmogorov‑Arnold 表示定理、B‑spline 适配激活、网格扩展、De Boor 算法计 FLOPs，实验使用 PyKAN、fastKAN、convKAN 等实现。

**📊 数据集**

实验数据集包括数值逼近的平方/立方数、日最低气温时序、以及经典 Wine 分类数据集。

**📈 对比分析**

通过 MSE、分类准确率和 FLOPs 三维指标对同一任务的 KAN 与 MLP 进行迭代优化比较，结果显示 KAN 在大部分任务上 MSE 下降至 1‑2% 级别、准确率提升 2% 以上，同时 FLOPs 减少 99% 以上。

**⚠️ 局限性**

当前研究仅在小规模数据与单任务场景验证，缺乏对高维大规模、跨域迁移和硬件加速的深入评估。

---

## 412. Synchronizing Probabilities in Model-Driven Lossless Compression

**arXiv ID:** 2601.10678 | [PDF](https://arxiv.org/pdf/2601.10678v1)

**作者:** Aviv Adler `[一作]` (Analog Devices), Jennifer Tang `[通讯]` (Holy Cross)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PMATIC算法，用于在LLM驱动的无损压缩中容忍有限的预测不匹配，保证解码正确性。

**💡 创新点**

创新点在于将概率匹配问题形式化，并通过量化概率区间、辅助位与中心/边界投影实现可容错的算子码；实现了对算子误差的理论误差上界和最优区间宽度的推导。

**🔧 技术方法**

采用区间编码（Arithmetic Coding）与概率量化、辅助位编码、LLM（Llama 3.1 8B）预测模型、对数概率归一化、以及二元KL散度与方差界定等技术。

**📊 数据集**

使用enwik8（约10 MB Wikipedia文本）和1000篇随机Wikipedia文章（共约3 MB）进行实验，LLM模型为量化版Llama 3.1 8B。

**📈 对比分析**

与无匹配误差的LLM压缩以及gzip比较；PMATIC在误差约0.002时每标记约5.34比特，压缩率≈13.3%，远优于gzip（≈40.9%）且仅比无误差LLM低约2–3比特/标记，误差约0.00002时表现更佳。

**⚠️ 局限性**

局限性包括：仅在合成误差上验证；对真实LLM非确定性场景的鲁棒性待检验；缺乏对随机误差边界的理论下界；以及需要在模型大小、压缩效率与计算开销之间做进一步平衡。

---

## 413. CURVE: A Benchmark for Cultural and Multilingual Long Video Reasoning

**arXiv ID:** 2601.10649 | [PDF](https://arxiv.org/pdf/2601.10649v1)

**作者:** Darshan Singh `[一作]`, Shachi Dave `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了一个跨文化、多语言、长时视频推理基准，包含540段视频、2400个问题，配有音频、原生语言问答与多步人类推理轨迹；

**💡 创新点**

创新点在于：①从18个全球地域聚焦文化语境，真实视频+音频；②问题与答案以多步推理轨迹记录，支持过程级评估；③引入基于证据的有向无环图与迭代错误隔离方法；

**🔧 技术方法**

采用多模态视觉语言模型（VLM）进行视频问答，使用音频+视频输入，利用大型语言模型（如Gemini‑2.5‑pro）做中间推理与答案生成，并用LLM Judge对结果进行评分；

**📊 数据集**

使用的是本文自行构建的多文化视频推理基准（包含18种语言、6个文化域），并对比了现有英语中心化的数据集；

**📈 对比分析**

评估方式为LLM Judge评分（0-2分），与人工评测对比；结果显示人工基线达95.22%，最佳模型仅达45.07%，模型在低资源语言上表现更差，音频信息显著提升准确率；

**⚠️ 局限性**

限制包括：高昂的人类标注成本、部分低资源语言仍表现低下、仍无法完全消除西方文化偏见、模型对复杂跨模态推理的理解仍不足。

---

## 414. Breaking the Storage-Bandwidth Tradeoff in Distributed Storage with Quantum Entanglement

**arXiv ID:** 2601.10676 | [PDF](https://arxiv.org/pdf/2601.10676v1)

**作者:** Lei Hu `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**通讯引用:** 13727 | [OpenAlex ID](https://openalex.org/A5021132487)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在分布式存储系统中利用量子纠缠辅助的修复过程，提出并证明了在量子通信模型下的存储与修复带宽之间的根本性权衡关系。

**💡 创新点**

创新点在于首次通过量子纠缠和超密编码实现了存储与带宽的同时最小化——当修复节点数 d ≥ 2k-2 时，系统可达到一个单一的最优再生点，打破了经典系统中的存储‑带宽折中。

**🔧 技术方法**

采用了量子超密编码、纠缠分发、量子测量以及量子信息流图的 cut‑set 边界分析等技术手段，构造了量子辅助修复模型并推导了对应的最优点。

**📊 数据集**

本工作为理论分析，未使用具体实验数据集，而是通过数学模型和示例参数演示结果。

**📈 对比分析**

通过将经典的 MSR/MBR 轨迹与量子辅助下的 QMSR/QMBR 轨迹在相同参数下绘图比较，表明在 d ≤ 2k-2 时量子模型的修复带宽可缩减一半，在 d ≥ 2k-2 时两者收敛到同一最优点，整体性能显著优于经典方案。

**⚠️ 局限性**

局限性包括：需要在所有剩余节点间预先共享高质量纠缠态；假设量子通道无噪声且测量无误差；实现成本高、技术门槛大，实际部署与鲁棒性仍待进一步研究。

---

## 415. Translating database mathematical schemes into relational database software applications with MatBase

**arXiv ID:** 2601.10604 | [PDF](https://arxiv.org/pdf/2601.10604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 416. Multi-Property Synthesis

**arXiv ID:** 2601.10651 | [PDF](https://arxiv.org/pdf/2601.10651v1)

**作者:** Christoph Weinhuber `[一作]` (University of Oxford), Moshe Y. Vardi `[通讯]` (Rice University)

**通讯引用:** 43692 | [OpenAlex ID](https://openalex.org/A5000059818)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出一种符号化的多属性有限时序逻辑合成方法，能够在单次固定点计算中同时判断所有属性组合的可实现性并合成相应策略。

**💡 创新点**

通过引入布尔目标变量并利用单次符号固定点来压缩指数级属性组合，同时保留最大可实现集，显著降低枚举复杂度，提升计算效率。

**🔧 技术方法**

使用符号合成技术（BDD）、可达性游戏、可控前驱运算、线性时序逻辑（LTLf）以及最大化多属性约束的布尔编码。

**📊 数据集**

在Chain、Until、Next、Counter、RobotNav等参数化基准集合上进行实验。

**📈 对比分析**

与枚举子集的基线方法对比，使用MPSynth实现符号多属性合成；在所有实例中平均提升一到两位数，最快可达两百倍加速。

**⚠️ 局限性**

最坏情况下仍为2EXPTIME复杂度；最大可实现集在最坏情形仍可能指数级；当前方法未考虑概率环境或量化偏好。

---

## 417. RoutIR: Fast Serving of Retrieval Pipelines for Retrieval-Augmented Generation

**arXiv ID:** 2601.10644 | [PDF](https://arxiv.org/pdf/2601.10644v1)

**作者:** Eugene Yang `[一作]` (Johns Hopkins University), Trevor Adriaanse `[通讯]` (Johns Hopkins University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一个轻量化的 Python 包（routir），为检索模型提供在线 HTTP API，支持动态检索管线组装、异步批处理、缓存以及多模型融合，方便在 Retrieval‑Augmented Generation (RAG) 系统中即时检索。

**💡 创新点**

创新点在于：① 把离线 IR 平台的检索功能抽象为可插拔的 Engine，并通过 Processor 进行批量与缓存处理；② 通过 JSON 配置实现在线动态管线构造（并行、融合、重排序等）；③ 在单一端点上集成多模型（dense、sparse、multi‑vector、LLM 重排序），实现高吞吐量的检索服务。

**🔧 技术方法**

主要技术包括：Python 异步 I/O、FAISS、Anserini、Pyserini、HuggingFace Transformers、vLLM、Redis 缓存、HTTP API、Pipeline 解析器与上下文无关文法、Relay Engine 用于多节点分布式调度。

**📊 数据集**

使用了 TREC 2023 NeuCLIR MLIR 任务数据集（约 1000 万网页，中文/波斯语/俄语）进行评测，另外在 JHU SCALE 2025、TREC RAGTIME 及其他公开数据集上验证了可扩展性。

**📈 对比分析**

通过与三种检索模型（PLAID‑X、MILCO、Qwen3 Embedding）在批处理与顺序模式下进行对比，结果显示：PLAID‑X 在顺序时延最短（0.24 s/query），Qwen3 Embedding 在批处理吞吐量最高（9.60 query/s）；整体实现了 3–10 query/s 的吞吐率，且缓存后可达毫秒级响应。

**⚠️ 局限性**

局限性包括：① 未实现正式的安全与访问控制（仅适用于内部原型）；② 缺乏多节点负载均衡与故障恢复机制；③ 对内存占用高的索引（FAISS、Anserini）仍有挑战；④ 对跨语言多模型融合的细粒度控制与评估机制尚不完善。

---

## 418. RSATalker: Realistic Socially-Aware Talking Head Generation for Multi-Turn Conversation

**arXiv ID:** 2601.10606 | [PDF](https://arxiv.org/pdf/2601.10606v1)

**作者:** Peng Chen `[一作]` (Institute of Software, Chinese Academy of Sciences), Feng Tian `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b88c6eac-d57a-4623-a604-1f401f3eb268` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于3D高斯渲染的多轮对话社交意识说话人生成框架RSATalker，能够在说话人与聆听人之间动态切换并同时考虑社会关系；

**💡 创新点**

创新点在于：①首次将3D Gaussian Splatting与面部运动生成结合实现高质量说话人渲染；②引入社交意识模块通过血缘与等级双维度的可学习嵌入来调控说话人与聆听人行为；③采用三阶段预训练与微调策略，使模型在多轮对话下保持稳定；

**🔧 技术方法**

核心技术包括：FLAME基面部动力学、音频到运动的Transformer编码/解码、3D Gaussian Splatting渲染、社交意识查询网络（motion‑socialnet & Gaussian‑offsetnet）以及跨模态交叉注意力；

**📊 数据集**

使用自构建的RSATalker数据集，该数据集包含语音–3D网格–图像三元组，并标注四类社会关系（血缘/非血缘 × 等级/非等级）；

**📈 对比分析**

与FaceFormer、CodeTalker、DualTalk等mesh方法以及ER‑NeRF、SyncTalk、GaussianTalker等image方法对比。实验中RSATalker在FD、P‑FD、MSE、L1、PSNR、SSIM、LPIPS以及用户研究的Realism、Fluency和Social Relationship Accuracy上均优于所有基线；

**⚠️ 局限性**

局限性包括：对极端头部旋转或严重遮挡时视觉和社交一致性下降；长时间会话中早期社交线索衰减导致角色一致性稍弱；数据集与用户研究受限于专业受试者，尚需在更广泛人群和更大规模数据上验证。

---

## 419. Process-Guided Concept Bottleneck Model

**arXiv ID:** 2601.10562 | [PDF](https://arxiv.org/pdf/2601.10562v1)

**作者:** Reza M. Asiyabi `[一作]` (University of Edinburgh), Casey Ryan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了Process-Guided Concept Bottleneck Model（PG-CBM），通过将生态学过程和因果关系嵌入到中间概念瓶颈，构建可解释、稳健的地面高层生物量密度估计模型。

**💡 创新点**

创新点在于将域知识与因果机制直接注入CBM架构，支持异构稀疏监督、实现过程驱动的多变量协同学习，并提供透明的中间产出，可用于诊断和科学洞察。

**🔧 技术方法**

技术手段包括多模态深度学习（SAR、光学、位置信息编码）、多头自注意力、空间金字塔特征融合、分位数回归损失、结构正则化以及基于过程的聚合模块g(·)；并在预训练与联合微调两阶段实现。

**📊 数据集**

使用了南非与非洲干燥热带林地的海量GEDI L2斑块（约870万）以及约8,260个现场测绘样地（植被冠层、冠层高度、干叶生物量和茎密度）作为多源标签。

**📈 对比分析**

与传统CBM、黑盒深度学习模型以及ESA CCI与GEDI L4B产品进行对比，PG-CBM在RMSD、平均偏差和相对偏差等指标上均优于对照，尤其在结构依赖偏差和OOV样本上表现更稳健。

**⚠️ 局限性**

局限在于仍需至少部分中间属性的标签；对复杂动态或层级因果关系的建模有限；未深入探究分布外泛化与不确定性量化；未来可引入弱监督、物理模拟或图神经网络以提升泛化与解释力。

---

## 420. Learning Latency-Aware Orchestration for Parallel Multi-Agent Systems

**arXiv ID:** 2601.10560 | [PDF](https://arxiv.org/pdf/2601.10560v1)

**作者:** Xi Shi `[一作]` (University of Central Florida), Qian Lou `[通讯]` (University of Central Florida)

**通讯引用:** 1680 | [OpenAlex ID](https://openalex.org/A5044298887)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向并行执行的多智能体系统调度框架 LAMaS，显式地对关键执行路径（Critical Path）进行延迟监督，从而实现更短的并行执行时间。

**💡 创新点**

创新点在于：① 移除同层内不必要的依赖，使多智能体能真正并行执行；② 在奖励函数中加入关键路径延迟惩罚，并采用关键路径信用分配机制，避免了传统全局延迟惩罚导致的信用分配错误；③ 通过对每个查询动态生成拓扑结构，实现了查询级别的难度感知。

**🔧 技术方法**

主要技术包括：概率超网（Agentic Supernet）结构、层级并行采样机制、延迟代理（Token+工具时间）、基于策略梯度的强化学习（Policy Gradient）以及关键路径信用分配（Critical-Path-Aware Credit Assignment）。

**📊 数据集**

在三个复杂推理基准上进行实验：GSM8K（数学题）、HumanEval（代码生成）和 MATH（竞赛级数学题），全部使用官方公开数据集。

**📈 对比分析**

与最先进的 MaAS 基线对比，LAMaS 在三组基准中平均关键路径长度分别降低了 38%–46%，而任务性能保持相当或略有提升；相对于固定拓扑的生成、链式推理和多路链式推理+自一致性方法，LAMaS 在准确率与延迟/成本平衡上获得更优的综合表现。

**⚠️ 局限性**

局限性包括：① 仅在算法层面考虑延迟，未结合真实系统/硬件的排队、速率限制等因素；② 延迟代理采用 Token+工具时间近似，实际 wall‑clock 延迟仍可能存在偏差；③ 仅针对当前 LLM 与工具组合，缺乏跨模型/硬件的通用验证。

---

## 421. Chebyshev Accelerated Subspsace Eigensolver for Pseudo-hermitian Hamiltonians

**arXiv ID:** 2601.10557 | [PDF](https://arxiv.org/pdf/2601.10557v1)

**作者:** Edoardo Di Napoli `[一作]` (Jülich Supercomputing Centre), Xinzhe Wu `[通讯]` (Jülich Supercomputing Centre)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文扩展了ChASE子空间迭代算法，使其能够在大规模稠密伪厄米巴塞尔-塞尔登(BSE) Hamiltonian 上计算数千个极值特征对，支持现代GPU集群；

**💡 创新点**

创新点在于提出基于隐式对偶基的Oblique Rayleigh‑Ritz投影，实现与Hermitian等价的Rayleigh‑Quotient，保持二次收敛；并设计了通信友好的Chebyshev过滤实现，极大减少全局通信；

**🔧 技术方法**

使用的技术包括Chebyshev多项式过滤、分块QR正交化、Hermitian等价Rayleigh‑Ritz、GPU并行矩阵乘、NCCL通信、MKL/ cuBLAS/ cuSOLVER；

**📊 数据集**

数据集采用Yambo生成的四个硅(Si)和二硫化钼(MoS₂)的伪厄米Hamiltonian，尺寸分别为23k、79k、64k、104k；

**📈 对比分析**

与ELPA（直接全对角化）和SLEPc Lanczos迭代器比较，ChASE在求解1k极值特征对时实现2 PFLOP/s（在256 GPU上），性能比ELPA快数十倍、比SLEPc在大规模时收敛更稳健；

**⚠️ 局限性**

局限性包括对Lanczos估计上界μₙ的依赖，易出现数值不稳定；对偶基Q_*的构造在特定奇异情况下可能失效，需要备用非Hermitian Rayleigh‑Ritz；

---

## 422. DeepUrban: Interaction-Aware Trajectory Prediction and Planning for Automated Driving by Aerial Imagery

**arXiv ID:** 2601.10554 | [PDF](https://arxiv.org/pdf/2601.10554v1)

**作者:** Constantin Selzer `[一作]` (Munich University of Applied Science), Fabian B. Flohr `[通讯]` (Munich University of Applied Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了DeepUrban无人机数据集并在其上验证了基于ScePT的轨迹预测与规划方法，

**💡 创新点**

创新点在于提供了高密度、3D丰富的城市交通场景（含VRU、e-scooter等14类）以及完整的地图与场景信息，从而显著提升预测与规划的准确度，

**🔧 技术方法**

采用了ScePT（CVAE+图神经网络+MPC）模型，并通过TrajData数据加载器统一处理多数据源，

**📊 数据集**

使用DeepUrban V1数据集与nuScenes、Waymo等公开数据集进行对比实验，

**📈 对比分析**

实验表明在nuScenes上加入DeepUrban可使ADE/FDE提升高达44.1%/44.3%，碰撞率显著下降，证明了数据集对模型泛化的提升作用，

**⚠️ 局限性**

局限包括：仅对车辆与行人交互建模，忽略自行车、摩托车等类型；规划未充分利用BEV地图导致偶发越界；在极高密度场景下计算复杂度较高。

---

## 423. PACEvolve: Enabling Long-Horizon Progress-Aware Consistent Evolution

**arXiv ID:** 2601.10657 | [PDF](https://arxiv.org/pdf/2601.10657v1)

**作者:** Minghao Yan `[一作]` (University of Wisconsin-Madison), Beidou Wang `[通讯]` (Google)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种面向大语言模型的进化搜索框架——Progress‑Aware Consistent Evolution（PACE），通过分层上下文管理、动量背溯和自适应协作采样，提升了进化搜索的稳定性和效率。

**💡 创新点**

创新点在于（1）引入分层上下文管理以消除上下文污染并保留高质量信息；（2）使用基于相对进展的动量背溯机制动态逃离局部最优；（3）设计自适应协作采样策略，在多岛搜索中灵活平衡内部探索与外部知识迁移，取代传统静态交叉。

**🔧 技术方法**

主要技术包括：大语言模型（Gemini 2.5 Pro/Flash、Gemini 3 Pro）作为智能搜索器；分层上下文管理（宏观概念与微观实验层次）与双层剪枝；相对进展指标和指数加权移动平均构成的动量背溯；基于绝对进展的权重分配实现的自适应协作采样。

**📊 数据集**

使用的基准数据集包括：Symbolic Regression（LLM‑SR 非线性振荡器任务）、KernelBench（16个 GPU kernel 任务）、Modded NanoGPT（多任务训练加速挑战）。

**📈 对比分析**

在这些基准上，PACE 通过与 ShinkaEvolve、OpenEvolve、CodeEvolve、LLM‑SR、uDSR 等现有方法对比，表现出更低的 Log10 NMSE（符号回归）、更高的 GPU kernel 加速倍数（KernelBench）以及更快的验证损失收敛（Modded NanoGPT）。特别是 PACE‑Single 在大多数 kernel 上均超过 PyTorch 基线，PACE‑Multi 在 81.25% 的 kernel 上优于 Single，并在大多数基准上达到或超过当前最佳成果。

**⚠️ 局限性**

局限性包括：对 LLM 的依赖导致推理成本高；多岛协调仍需手动设定冻结期和阈值；在极大规模或极高算力环境下，动量背溯与协作采样的参数调优可能不具通用性；对极度稀疏奖励任务的鲁棒性尚待进一步验证。

---

## 424. Extrinsic Vector Field Processing

**arXiv ID:** 2601.10621 | [PDF](https://arxiv.org/pdf/2601.10621v1)

**作者:** Hongyi Liu `[一作]`, Misha Kazhdan `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种连续基的切向量场离散化方法，利用顶点法线的 Phong 插值得到连续法线场，并通过 Rodrigues 旋转把顶点处的切向量平移到三角形内部，随后按重心坐标线性混合得到全局连续切向量场。

**💡 创新点**

创新点在于：① 在外在表示下构造连续切向量基；② 给出可点值求导的协变导数表达式；③ 在此基础上定义 Hodge、Connection、Killing 能量以及 Lie bracket 的离散化；④ 通过数值积分实现任意形状上高效的有限元组装。

**🔧 技术方法**

使用的技术包括：Phong 法线插值、Rodrigues 最小角旋转、有限元积分（数值求积）、矩阵组装、线性/广义特征求解、Lie bracket 投影。

**📊 数据集**

评估数据集主要包括：单位球（icosa 细分与随机凸包）、torus、各种三角网（genus‑1、3、6 等）以及标准几何模型（bunny、elephant、icosa 等）。

**📈 对比分析**

与 Knöppel、Sharp、Stein 等方法对比：在球面上与 Knöppel 的结果几乎无差异；与 Stein 相当且在非均匀采样时更稳健；对 Sharp 在非均匀采样时表现略差，但在均匀采样下相近。整体能量矩阵在 90° 旋转下保持不变，Lie bracket 误差随频率升高而增大，但细化网格可显著降低误差。

**⚠️ 局限性**

局限性包括：① 依赖顶点法线的连续性，若法线不一致会影响结果；② 该基空间不闭合于 Lie bracket；③ 不能直接表示标量梯度（梯度为分段常数）；④ 对隐式几何的假设（dΦ|_p 可能不满足旋度为零）导致二阶曲率不对称。

---

## 425. Combinatorial Optimization Augmented Machine Learning

**arXiv ID:** 2601.10583 | [PDF](https://arxiv.org/pdf/2601.10583v1)

**作者:** Maximilian Schiffer `[一作]` (Technical University of Munich), Axel Parmentier `[通讯]` (Institut Polytechnique Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了组合优化增强机器学习（COAML）的概念、方法框架与应用场景，提出了统一的COAML流水线模型和问题分类法

**💡 创新点**

首次系统化地将组合优化算子嵌入学习流程，并用经验成本最小化理论统一描述不同学习目标；提出基于不确定性和决策结构的分类体系

**🔧 技术方法**

结合组合优化oracle、经验成本最小化、模仿学习与强化学习等技术，构建可预测且可行的决策策略

**📊 数据集**

本文未给出具体实验数据集，主要以已有文献中的应用案例（调度、车辆路径、随机规划、强化学习等）为讨论依据

**📈 对比分析**

通过文献综述与对比分析，展示COAML方法在多领域任务上相较于传统机器学习或纯优化方法能够兼顾性能与可行性，但缺乏统一基准测试的系统实验评估

**⚠️ 局限性**

存在实验验证不足、数据集与任务多样性有限、算法可扩展性与实时性能待进一步研究等局限

---

## 426. Long-term Monitoring of Kernel and Hardware Events to Understand Latency Variance

**arXiv ID:** 2601.10572 | [PDF](https://arxiv.org/pdf/2601.10572v1)

**作者:** Fang Zhou `[一作]` (Ohio State University), Yang Wang `[通讯]` (Ohio State University)

**通讯引用:** 244234 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文开发了一套长周期监测工具链，对服务器应用的内核与硬件事件进行精细记录和离线分析，揭示导致延迟方差的多种事件及其时间变异。

**💡 创新点**

创新点包括：① 通过“只记录影响已选请求的事件”与“先记录累计量后按需细化”两大原则，实现了选择性记录并大幅降低存储与 I/O 开销；② 引入 Shadow Stack 机制精确计算中断/缺页异常的持续时长；③ 设计了基于置信区间与高阈值的 Impact 值度量，并通过两条域特定因果规则去除冗余或从属事件，从而提供更具可操作性的影响排名。

**🔧 技术方法**

技术手段：在 Linux 上结合 perf、ftrace、eBPF 与自定义内核模块，提供 begin_AppTask / end_AppTask API；使用 Shadow Stack 追踪嵌套中断、计算累计长度；采用线性回归分段拟合事件 CDF、Jaccard 距离衡量相关性；实现 O(N log N) 排序、O(N) 统计，保证在大规模（TB 级）数据上可接受的分析速度。

**📊 数据集**

实验数据集：在 CloudLab 上对六个真实应用（Memcached、MySQL、ZooKeeper、HBase、LoopBench、MemBench）进行 3000 小时实验，生成约 1.5TB 记录；每个实验周期 16 小时，记录速率约 0.1 MB/s。

**📈 对比分析**

方法比较：与传统 ftrace（记录所有事件）和 perf（记录部分事件并生成堆栈）在采样率、存储大小和 CPU 开销上做对比；在 Memcached 上通过 1 % 选择率的实验验证对吞吐量、p99 延迟影响几乎为 0；在各应用中分别对比优化前后 p99 与 CoV 的变化，展示对尾部延迟的 20–30 % 改善。

**⚠️ 局限性**

局限性：① 选择性记录可能遗漏罕见但影响巨大的事件；② 需要长时间（数小时甚至数百小时）采样才能覆盖所有罕见事件；③ 影响度量与因果规则基于经验假设，可能误判；④ 受限于硬件性能计数器数量，无法同时记录所有指标；⑤ 需要开发者手动插入 begin/end 标记，工作量不小。

---

## 427. Sparse Signal Recovery from Random Measurements

**arXiv ID:** 2601.10569 | [PDF](https://arxiv.org/pdf/2601.10569v1)

**作者:** Siu-Wing Cheng `[一作]` (Hong Kong University of Science and Technology), Man Ting Wong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5053506080)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种不需要求解优化问题或线性方程组的稀疏信号恢复算法，利用Θ(log n)个随机矩阵和中值统计实现对未知向量z的重建。

**💡 创新点**

创新点在于用中值代替传统的ℓ1最小化或迭代搜索，仅需O(k n log n)时间即可获得与OMP等优化方法相近的准确率，并显著提升速度。

**🔧 技术方法**

该方法结合随机高斯测量矩阵、统计中值估计、Chebyshev和Chernoff界定误差，并通过阈值化筛选支持集。

**📊 数据集**

实验数据为在三组n={2000,4000,8000}、稀疏率1%–8%的合成二进制信号上进行，测量矩阵采样自𝒩(0,1/k)，噪声σ_w=0.1。

**📈 对比分析**

与GPSR、DWS、OMP、BIHT、NBIHT等基准方法比较，本文算法在准确率上与GPSR、DWS、OMP相当，且速度比GPSR高2.8–59.17倍、比DWS高1.95–23.92倍、比OMP高2.65–308.25倍，且在1-bit压缩感知情形下速度提升至少169倍。

**⚠️ 局限性**

局限性包括：需要生成并存储Θ(log n)个随机矩阵，依赖噪声均方差已知或可估计，且理论保证在n足够大且信号幅值满足预设上下界时成立；对非二进制或更一般分布的真实数据尚未验证。

---

## 428. Inferring signed social networks from contact patterns

**arXiv ID:** 2601.10565 | [PDF](https://arxiv.org/pdf/2601.10565v1)

**作者:** Dávid Ferenczi `[一作]` (Maastricht University), Leto Peel `[通讯]` (Maastricht University)

**通讯引用:** 1295 | [OpenAlex ID](https://openalex.org/A5072266097)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过贝叶斯框架和MCMC推断，从接触频率数据中恢复带正负符号的社交网络，并区分机会缺失与主动回避。

**💡 创新点**

引入交互机会组的概念，将缺失互动分为偶然与选择，使用生成模型区分正负边，提升负边检测。

**🔧 技术方法**

生成模型+贝叶斯推断、MCMC、Metropolis‑Hastings、平行温度混合、后验预测检验。

**📊 数据集**

合成Erdős‑Rényi网络 + 真实法国高中学生随时计数的接触传感器数据。

**📈 对比分析**

与秩序回归和配置模型基线相比，在正负边分类的AUC上表现最佳，尤其对负边显著优于基线；在实测数据中与问卷友谊一致，后验预测p值普遍高于0.05。

**⚠️ 局限性**

对群体结构变化敏感，MCMC易陷入局部最优；模型假设接触率仅受关系与机会影响，未考虑其他行为或时变网络；仅适用于对称关系。

---

## 429. Rewriting Systems on Arbitrary Monoids

**arXiv ID:** 2601.10564 | [PDF](https://arxiv.org/pdf/2601.10564v1)

**作者:** Eduardo Magalhães `[一作]` `[通讯]` (University of Porto), Eduardo Magalhães (University of Porto)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出了“单形重写系统（MRS）”的概念，将字符串重写从自由幺半群推广到任意幺半群，并构造了其 2‑范畴结构，证明了 MRS 与幺半群之间的双 adjunction，进一步引入了广义 Tietze 转换（GETT）来描述同一幺半群的所有无穷可合成、无穷终止的 MRS 之间的等价关系。

**💡 创新点**

创新点在于：①从自由幺半群切换到任意幺半群，解决了在一阶逻辑框架下无法内部化“自由性”问题；②以 2‑范畴的方式给 MRS 赋予结构，构造了 G ⊣ I 的双 adjunction；③将经典 Tietze 转换推广为 GETT，能处理无限生成与无穷规约的情形，从而实现对同一幺半群所有可合成无穷 MRS 的完全分类。

**🔧 技术方法**

主要技术包括：幺半群与自由幺半群的代数表示；抽象重写系统（ARS）理论；构造 2‑范畴 NCRS₂ 及其同构性证明；利用 Church–Rosser 性质、唯一正规形和合成规则证明双 adjunction；以及构造 GETT 的四种类型并证明其不改变不可约元素集合。

**📊 数据集**

该工作为纯理论研究，无使用具体数据集；所有结果均基于数学证明与构造性演绎。

**📈 对比分析**

与传统的有限 Tietze 转换相比，GETT 能处理无限生成与无穷规约的 MRS，理论上可将任何两个无穷可合成、无穷终止的 MRS 通过（可能无穷）GETT 序列联系起来；在实验层面未做性能对比，属于理论性验证。

**⚠️ 局限性**

局限性：仅适用于 Noetherian（终止）且合并（confluent）的 MRS；对非终止或非合并系统不适用；GETT 变换序列可能无穷，缺乏有效终止判定；并且实现细节需在具体实例中进一步验证。

---

## 430. Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models

**arXiv ID:** 2601.10679 | [PDF](https://arxiv.org/pdf/2601.10679v1)

**作者:** Zirui Ren `[一作]` (Shanghai Qi Zhi Institute), Ziming Liu `[通讯]` (Tsinghua University)

**通讯引用:** 473 | [OpenAlex ID](https://openalex.org/A5083403906)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对层级推理模型（HRM）的推理机制进行机制化分析，发现其在固定点、“grokking”动态和多重固定点方面存在缺陷，并提出数据增强、输入扰动和模型引导三种扩展策略，提升其在极难数独（Sudoku‑Extreme）上的准确率；

**💡 创新点**

首次揭示HRM实际上是在“猜测”而非系统推理，发现其固定点不稳定、存在误导性固定点，并通过数据层次化、多重推理与模型随机性来显著提升性能；

**🔧 技术方法**

使用HRM的递归结构、深度监督与单步梯度训练，结合数据增强（简化样本）、输入扰动（token重标识、旋转/反射等）与模型引导（多次检查点投票）等技术；

**📊 数据集**

主要在极难数独数据集Sudoku‑Extreme上进行实验，也使用其训练集的1000个增强样本；

**📈 对比分析**

与原始HRM及其变体（如Tiny Recursive Model）对比，原始HRM准确率54.5%，通过单一技术提升至59.9%/64.7%/73.2%，组合后达到96.9%，超过现有SOTA；

**⚠️ 局限性**

局限在于仅针对数独任务验证，推理机制的通用性尚待进一步验证，且多重推理、模型引导会增加计算成本与实现复杂度。

---

## 431. An Extension-Based Accessibility Framework for Making Blockly Accessible to Blind and Low-Vision Users

**arXiv ID:** 2601.10688 | [PDF](https://arxiv.org/pdf/2601.10688v1)

**作者:** Rubel Hassan Mollik `[一作]` (University of North Texas), Aboubakar Mountapmbeme `[通讯]` (University of North Texas)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5088341500)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出并实现了一套可扩展的无障碍框架（EAF），让基于 Blockly 的块式编程环境对盲/低视力用户友好，并可直接集成到现有平台而不需改动核心代码。

**💡 创新点**

核心创新包括：三维结构化导航模型（左右/上下/进退键）、堆栈标签与块编号、模式化编辑（切换编辑/导航）、WAI‑ARIA 兼容的屏幕阅读器支持以及完全模块化的插件架构。

**🔧 技术方法**

使用 JavaScript（ES6+）插件模式，结合 SVG/HTML/CSS 渲染、WAI‑ARIA 属性、键盘事件与 ARIA live 区域实现音频反馈，框架在浏览器环境下运行。

**📊 数据集**

采用 177 条功能测试用例覆盖导航、编辑、堆栈标记等；半结构化访谈样本为 4 名参与者（两名专家、两名本科生），未使用公开数据集。

**📈 对比分析**

与 Blockly 默认键盘导航对比，EAF 在导航清晰度、编辑效率和屏幕阅读器反馈上显著提升；功能测试通过率从 78.5% 提升至 85.9%，访谈显示用户体验更佳。

**⚠️ 局限性**

局限性：未在盲/低视力学生的真实课堂中评测；未验证对商业 BBPE（如 MakeCode、Code.org）的兼容性；未覆盖盲文显示器或屏幕放大软件等辅助技术。

---

## 432. The Impact of Generative AI on Architectural Conceptual Design: Performance, Creative Self-Efficacy and Cognitive Load

**arXiv ID:** 2601.10696 | [PDF](https://arxiv.org/pdf/2601.10696v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 433. Alterbute: Editing Intrinsic Attributes of Objects in Images

**arXiv ID:** 2601.10714 | [PDF](https://arxiv.org/pdf/2601.10714v1)

**作者:** Tal Reiss `[一作]` (Google), Yedid Hoshen `[通讯]` (Google)

**通讯引用:** 1682 | [OpenAlex ID](https://openalex.org/A5047455929)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于扩散模型的 Alterbute 方法，可在保持对象身份和场景不变的前提下编辑图像中对象的颜色、纹理、材质和形状等内在属性。

**💡 创新点**

创新点包括：① 放宽训练目标，让模型同时学习内在和外在属性的编辑；② 引入视觉命名实体（VNE）作为细粒度身份表示；③ 通过 Gemini 自动提取 VNE 标签和属性描述，实现可扩展的监督。

**🔧 技术方法**

使用 SDXL 7B 细化扩散模型、UNet 多输入 grid 结构，结合文本提示、参考图像、背景图与掩码进行条件控制；并使用自监督式 VNE 提取、分类器无关引导、随机遮蔽训练等技术。

**📊 数据集**

训练数据来源于 OpenImages（约 9M 图像），通过 Gemini 自动标注得到约 1.5M 对象，形成 69,744 个 VNE cluster；评估数据由 30 个不同对象、100 个编辑案例构成（包含热门与低频类别）。

**📈 对比分析**

与 FlowEdit、InstructPix2Pix、OmniGen、UltraEdit、Diptych 等通用编辑器及 MimicBrush、MaterialFusion 等属性专用编辑器进行定性与定量比较；用户研究与 VLM 评估表明，Alterbute 在身份保留和属性编辑质量上显著优于基线，并实现了唯一可保形状编辑。

**⚠️ 局限性**

局限性：仅能一次编辑单一属性，难以同时编辑互相矛盾的属性；粗盒子掩码可能导致背景伪影；对刚性物体形状编辑效果有限；缺乏标准基准，评估仍依赖人工和 VLM。

---

## 434. MatchTIR: Fine-Grained Supervision for Tool-Integrated Reasoning via Bipartite Matching

**arXiv ID:** 2601.10712 | [PDF](https://arxiv.org/pdf/2601.10712v1)

**作者:** Changle Qu `[一作]` (Renmin University of China), Dawei Yin `[通讯]` (Baidu Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MatchTIR 框架，通过二分图匹配实现工具调用的逐步奖励，配合双层优势估计以细粒度地引导大语言模型进行工具集成推理。

**💡 创新点**

创新点在于：①将逐步奖励视为二分图匹配问题，设计硬（Hungarian）与软（Optimal Transport）两种匹配策略；②引入双层优势（轨迹层与步层）融合，实现对每个 token 的精准信用分配。

**🔧 技术方法**

采用的技术包括：GRPO 强化学习框架、二分图匹配（Hungarian）、Optimal Transport、离散化奖励与优势归一化、KL 正则化。

**📊 数据集**

使用 FTRL 数据集进行训练，评估基于 FTRL、BFCL、ToolHop 三个基准数据集。

**📈 对比分析**

与 vanilla、GRPO、ToolRL、FTRL 等基线相比，MatchTIR 在 4B 模型上即可超过多数 8B 基线，尤其在长序列、多轮任务中表现突出，工具调用效率更高、成功率更高。

**⚠️ 局限性**

局限性包括：缺乏对更大规模模型的实验，依赖可验证的 ground-truth 轨迹，难以推广至开放式深度研究等高度开放式任务。

---

## 435. Grounding Agent Memory in Contextual Intent

**arXiv ID:** 2601.10702 | [PDF](https://arxiv.org/pdf/2601.10702v1)

**作者:** Ruozhen Yang `[一作]` (University of Illinois Urbana-Champaign), Jiawei Han `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 122930 | [OpenAlex ID](https://openalex.org/A5019539533)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于上下文意图的结构化意图追踪记忆系统，用于提升大语言模型在长周期目标导向交互中的检索准确性。

**💡 创新点**

创新点在于动态诱导的三维意图索引（主题范围、事件类型、关键实体类型），实现上下文感知检索过滤，显著降低检索噪声。

**🔧 技术方法**

技术包括LLM在线推理生成意图标签、核心ference对齐、结构化检索与标签密度排序，以及零射预训练与动态标签演化。

**📊 数据集**

使用了自行构建的Context‑Aware Agent Memory Evaluation Benchmark（CA‑ME），包含旅行规划与辩论两域的长序列对话，并与LongMemEval benchmark进行对比评估。

**📈 对比分析**

与13种基线（大上下文LLM、RAG、结构化记忆等）比较，CA‑ME中等长度提升约11.6%，大长度提升约35.6%，显著优于最强基线。

**⚠️ 局限性**

局限性包括：生成意图标签需要多次LLM调用，导致耗时高；标签演化受缓冲窗口限制，新事件类型更新滞后；细粒度标签对信息合成任务不利，需要层级化结构。

---

## 436. WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments

**arXiv ID:** 2601.10716 | [PDF](https://arxiv.org/pdf/2601.10716v1)

**作者:** Xuweiyi Chen `[一作]` (University of Virginia), Zezhou Cheng `[通讯]` (University of Virginia)

**通讯引用:** 913 | [OpenAlex ID](https://openalex.org/A5067219074)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了WildRayZer，一种在动态环境下进行自监督稀疏视角新视图合成的框架；

**💡 创新点**

其创新点在于利用自监督的“分析‑再合成”策略，通过静态渲染残差生成伪运动掩码，进而在不需要姿态或动态区域标签的情况下实现动态与静态内容的分离；

**🔧 技术方法**

核心技术包括Transformer‑based token‑space渲染器、基于DINOv3特征和SSIM的伪运动掩码生成、运动估计头、动态掩码遮挡训练以及简单的copy‑paste增强；

**📊 数据集**

本文使用了新收集的Dynamic RealEstate‑10K（15K动态室内序列）和D‑RE10K‑iPhone（50个带转瞬/无转瞬对照的iPhone拍摄序列）进行训练和评估；

**📈 对比分析**

与多种优化式和基于NeRF/3D Gaussian等公开方法相比，WildRayZer在稀疏视角下的动态视图合成与运动分割任务上显著提升，特别是在完整图像PSNR/SSIM和动态区域mIoU上取得领先；

**⚠️ 局限性**

限制包括对高质量渲染残差的依赖（可能在极端动态或光照变化下失效）、对动态遮挡范围有限的自监督掩码生成以及在完全无遮挡场景下的泛化能力仍需进一步验证。

---

## 437. See Less, Drive Better: Generalizable End-to-End Autonomous Driving via Foundation Models Stochastic Patch Selection

**arXiv ID:** 2601.10707 | [PDF](https://arxiv.org/pdf/2601.10707v1)

**作者:** Amir Mallak `[一作]` (University of Haifa), Alaa Maalouf `[通讯]` (Computer Science and Artificial Intelligence Laboratory MIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉语言模型提取的补丁级特征在自动驾驶中的冗余性，并提出 Stochastic Patch Selection (SPS) 方法：随机掩码部分补丁特征，保持空间布局，提升 OOD 泛化与推理速度。

**💡 创新点**

创新点在于将补丁特征的冗余性转化为可利用的稀疏性资源，通过随机掩码引入空间一致的稀疏表示；在不微调模型的前提下与基础模型兼容，并可与文本驱动的潜在空间增广结合。

**🔧 技术方法**

技术包括 BLIP‑2（或 DINO）视觉语言模型提取补丁特征、PCA 与相关性分析评估冗余、自注意力掩码、SPS 的随机/阈值/位置调整变体、Guided Policy Learning 强化学习训练控制策略，以及真实车部署验证。

**📊 数据集**

使用了结合真实驾驶日志与 VISTA 模拟环境的多场景数据；评估场景涵盖不同季节、天气、时间、交通参与者等 OOD 变化。

**📈 对比分析**

与 Drive‑Anywhere、I‑ViT、MF、No‑FM 等基线对比，SPS 在闭环驾驶实验中平均提升 6.2% OOD 成功率，单一情景最高 20.4%；与 SOTA 相比，速度提升 2.4×；在 DINO backbone 下平均提升 3.3%；可在真实车辆上无额外调优即可迁移。

**⚠️ 局限性**

局限性包括：采样率固定且为纯随机，未学习状态相关的采样策略；仅在补丁层面做稀疏，未探索多尺度或动态分辨率；对极端 OOD 情况（如无车道低光环境）仍存在挑战；验证仅针对 BLIP‑2/DINO 与特定任务，迁移性待进一步研究。

---

## 438. Improved Constructions of Reed-Solomon Codes with Optimal Repair Bandwidth

**arXiv ID:** 2601.10685 | [PDF](https://arxiv.org/pdf/2601.10685v1)

**作者:** Jing Qiu `[一作]` (Tsinghua University), Fang-Wei Fu `[通讯]` (Nankai University)

**通讯引用:** 3081 | [OpenAlex ID](https://openalex.org/A5063946169)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种改进的 Reed–Solomon 代码构造，去除了原先需要的 p_i ≡ 1 (mod s) 的同余约束，显著降低了子包络化（subpacketization）。

**💡 创新点**

创新点在于利用基变换技术（欧几里得方格分割、reshape 与 interference）构造满足所需子空间的基，从而实现不受同余限制的 RS-MSR 码，并将子包络化缩小了约 φ(s)^n 的因子。

**🔧 技术方法**

采用了线性维修方案、基变换（Euclidean square partition、reshape、interference）、数论工具（欧拉 φ 函数、素数分布理论）等方法。

**📊 数据集**

由于本研究是理论构造，不使用具体数据集；主要利用的是素数序列及其分布性质。

**📈 对比分析**

与之前的 Tamo–YB 构造相比，新的构造在保持相同的 MDS 性质和修复带宽的前提下，子包络化从 s∏p_i 降至 s∏p_i / φ(s)^n，性能得到显著提升。

**⚠️ 局限性**

局限性：目前尚未能进一步降低子包络化；缺乏证明该结果是否已达到最优下界，且仅适用于 Reed–Solomon 码的 MDS 结构。

---

## 439. Distributed Perceptron under Bounded Staleness, Partial Participation, and Noisy Communication

**arXiv ID:** 2601.10705 | [PDF](https://arxiv.org/pdf/2601.10705v1)

**作者:** Keval Jain `[一作]` (International Institute of Information Technology Hyderabad), Girish Varma `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种半异步的联邦/分布式感知器训练框架，该框架通过服务器端强制执行的时延桶聚合与填充策略，实现对时延、部分参与与通信噪声的统一建模；

**💡 创新点**

创新点在于引入了确定性时延桶聚合规则，消除了对随机时延或参与模型的需求，并给出了含有时延均值与噪声能量的有限视界误差上界；

**🔧 技术方法**

使用的技术包括感知器在线学习、迭代参数混合（IPM）、时延桶聚合与填充、无噪声时延均值S=1+平均时延、噪声能量V的平方根项；

**📊 数据集**

实验采用线性可分的合成数据集，划分给多台客户端；

**📈 对比分析**

通过 Monte‑Carlo 评估不同强制时延配置与噪声能量对累计误差的影响，验证理论预测的 O(√A) 增长与对平均时延的线性敏感性；

**⚠️ 局限性**

主要限制是：在存在噪声时无法获得收敛的稳定周期；所需的“新鲜参与”与“填充”条件对实际系统可能过于苛刻；

---

## 440. LIBERTy: A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Counterfactuals

**arXiv ID:** 2601.10700 | [PDF](https://arxiv.org/pdf/2601.10700v1)

**作者:** Gilat Toker `[一作]`, Roi Reichart `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LIBERTy框架，用结构化因果模型与LLM生成结构化反事实，构建用于概念级解释评估的交互式基准；

**💡 创新点**

创新点在于：①用结构化因果模型生成的“银”反事实取代昂贵的人类编写反事实；②提出order-faithfulness评估指标；③在三个高风险领域构建三大数据集，展示LLM对概念干预的敏感性；

**🔧 技术方法**

技术包括：结构化因果模型（SCM）、Pearl三步反事实推理、LLM（GPT‑4o）文本生成、模板与人物persona作为外生变量、概念基解释方法（匹配、概念消除、概念归因、反事实生成）等；

**📊 数据集**

数据集包含三类：疾病检测（自我报告）、简历筛选、工作场所暴力预测，每个数据集均含多层次概念与结构化文本，使用LLM生成文本与反事实；

**📈 对比分析**

与多种解释方法（匹配、反事实生成、概念消除、概念归因）在五个模型上对比，匹配方法（尤其FT Match）在局部和全局上表现最佳，order‑faithfulness平均≈0.7，误差距离≈0.3，显示仍有改进空间；

**⚠️ 局限性**

局限包括：文本为LLM合成而非人类真实文本；仅关注概念级解释，未覆盖词级或自由文本解释；所用的因果图为人工设计的近似模型，可能与真实世界偏离；

---

## 441. Data-driven stochastic reduced-order modeling of parametrized dynamical systems

**arXiv ID:** 2601.10690 | [PDF](https://arxiv.org/pdf/2601.10690v1)

**作者:** Andrew F. Ilersich `[一作]` (University of Toronto), Prasanth B. Nair `[通讯]` (University of Toronto)

**通讯引用:** 4626 | [OpenAlex ID](https://openalex.org/A5075671693)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种无前向求解器、可并行化的随机ROM学习框架，利用变分推断同时学习编码器、解码器和连续时间SDE，以处理跨参数空间与时变强迫条件的复杂动力学。

**💡 创新点**

创新点包括：①利用SDE重参数化技巧消除训练时的前向求解，显著降低计算成本；②采用子窗口分割的持续时间变分推断实现参数与数据规模无关的梯度估计；③支持物理先验与稀疏可解释模型的融合；④与传统两步分离方法相比，能避免隐变量轨迹交叉导致的ODE不可辨识问题。

**🔧 技术方法**

技术手段：连续时间变分推断（SVI）、SDE重参数化、深度核插值、CNN编码/解码器、KL散度正则化、Monte Carlo梯度估计、自动微分，结合物理先验或稀疏正则化。

**📊 数据集**

数据集：三组仿真轨迹——（1）无扰动反应扩散系统（D=10,000）;（2）参数化强迫布格方程（D=500, T=1001）；（3）二维受控制圆柱流动（Re=100，D=105,600，120训练/10验证/10测试轨迹）。

**📈 对比分析**

与SINDy（POD‑SINDy、AE‑SINDy）、PNODE、PNSDE对比，评价指标为均值误差、误差标准差和训练时间。在所有测试中，该方法在误差上显著优于基线，且训练时间仅约3小时，远快于PNODE/PNSDE（19–26小时）。

**⚠️ 局限性**

局限性：ELBO评估仍是计算瓶颈，尤其在极大数据集或高维隐变量时；目前模型假设为高斯噪声，难以捕捉更复杂噪声结构；对强迫非平稳性的处理仍有限，需要进一步扩展。

---

## 442. A continental-scale dataset of ground beetles with high-resolution images and validated morphological trait measurements

**arXiv ID:** 2601.10687 | [PDF](https://arxiv.org/pdf/2601.10687v1)

**作者:** S M Rayeed `[一作]` (Rensselaer Polytechnic Institute), Sydne Record `[通讯]` (University of Maine)

**通讯引用:** 4002 | [OpenAlex ID](https://openalex.org/A5065929393)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个多模态地面甲虫（Carabidae）数据集，包含超过13,200件NEON采集样本的高分辨率图像和精确的形态学测量（腹甲长度、宽度等）。

**💡 创新点**

首次通过统一拍摄协议和人工+AI标注方法，为北美甲虫提供了高质量图像与校准后的形态特征，并实现子毫米级数字化测量，填补了脊椎动物与植物在全球形态数据库中的“无脊椎动物缺口”。

**🔧 技术方法**

采用高分辨率相机+阴影盒照明、Grounding‑DINO+CVAT自动检测、TORAS人工+AI标注、Notes‑from‑Nature标注以及Python脚本将像素转换为毫米的校准步骤。

**📊 数据集**

使用NEON全国及夏威夷30个采样点的地面陷阱甲虫样本，数据已公开在HuggingFace（imageomics/Hawaii‑beetles、imageomics/2018‑NEON‑beetles）并与NEON元数据关联。

**📈 对比分析**

通过人工标注与TORAS测量以及不同标注者之间的交叉验证，RMSE约0.015–0.025 cm、R²>0.93、偏差≈0，显示数字化测量与传统卡尺测量几乎等效；瓶装样本使用Notes‑from‑Nature时，交叉误差约0.075 cm，仍保持良好一致性。

**⚠️ 局限性**

局限性包括样本仅覆盖NEON的30个地点，南部和西南部等区域缺失；瓶装样本的宽度测量被排除；标注需要人工干预且对大型图像分辨率有限制，未来需扩大地区、物种覆盖并提升自动化精度。

---

## 443. Perfect Secret Key Generation for a class of Hypergraphical Sources

**arXiv ID:** 2601.10697 | [PDF](https://arxiv.org/pdf/2601.10697v1)

**作者:** Manuj Mukherjee `[一作]` (Indraprastha Institute of Information Technology Delhi), Alhad Sethi `[通讯]` (Indian Institute of Science)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了针对完整 t‑均匀超图以及 3‑均匀超图的完美密钥生成方案，并证明其在若干超图类别下可达到密钥容量。

**💡 创新点**

创新点在于将超图的星形子图（star hypergraph）与哈密顿分解、树形打包等组合性结构相结合，构造线性公共通信并利用超图分解实现容量级完美密钥；对 3‑均匀超图进一步引入诱导环（induced cycle）构造，使得仅需 2 位密钥即可在多种结构下实现容量。

**🔧 技术方法**

核心技术包括：
- 超图的星形分解与组合打包；
- 线性公共通信与完美全知（perfect omniscience）对应的线性方程组求解；
- 利用哈密顿分解、树形打包、星形分解的组合性结果；
- 对 3‑均匀超图采用诱导环构造并进行两位密钥生成。

**📊 数据集**

本文无实验数据集，所有结果均为理论证明与信息理论下的极限分析。

**📈 对比分析**

与传统的强密钥生成（不要求完美）相比，本文的方案在满足完美密钥要求时仍能达到信息理论最优容量，证明了在这些超图结构下完美密钥与强密钥容量相等；在已知的可实现容量下，方案的性能与理论上限完全匹配。

**⚠️ 局限性**

局限性包括：
- 方案仅针对完整 t‑均匀超图、诱导环 3‑均匀超图以及特定的哈密顿可分解 3‑均匀超图；一般 3‑均匀或更高阶超图的完美密钥容量仍未给出；
- 需要超图满足严格的组合分解性质（如星形、哈密顿分解），实际应用中若超图结构不满足这些条件，方案不可直接使用。

---

## 444. DInf-Grid: A Neural Differential Equation Solver with Differentiable Feature Grids

**arXiv ID:** 2601.10715 | [PDF](https://arxiv.org/pdf/2601.10715v1)

**作者:** Navami Kairanda `[一作]` (Max Planck Institute for Informatics), Vladislav Golyanik `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 2888 | [OpenAlex ID](https://openalex.org/A5080103406)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种可微分的基于特征网格与RBF插值的高效表示，用于直接学习并求解微分方程。

**💡 创新点**

创新点在于将可微分RBF插值与多分辨率特征网格结合，既保留网格计算效率，又支持高阶导数，解决了传统网格在求解DE时的不可微问题。

**🔧 技术方法**

使用特征网格、RBF插值、自动微分、多分辨率结构以及对称约束的PDE损失来训练网络。

**📊 数据集**

在多种任务上验证：图像Poisson方程、Helmholtz波动方程、Kirchhoff–Love薄膜、Eikonal方程以及热扩散/对流方程。

**📈 对比分析**

与Siren、Instant‑NGP、K‑Planes、NeuRBF等方法比较，速度提升5–20倍，准确率与传统方法相当或更好，训练更快、收敛更稳定。

**⚠️ 局限性**

主要局限是高维输入下插值成本高，需要手工设置RBF形状参数，且对极大尺度问题尚未彻底验证。

---

## 445. From One-to-One to Many-to-Many: Dynamic Cross-Layer Injection for Deep Vision-Language Fusion

**arXiv ID:** 2601.10710 | [PDF](https://arxiv.org/pdf/2601.10710v1)

**作者:** Cheng Chen `[一作]` (Ant Group), Lianli Gao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Cross‑Layer Injection（CLI）框架，解决视觉‑语言模型中仅使用最终视觉层导致的特征瓶颈问题；

**💡 创新点**

创新点在于实现了动态“多层‑多层”视觉‑语言桥梁，并引入Adaptive Multi‑Projection（AMP）与Adaptive Gating Fusion（AGF）两种轻量级模块，使LLM能够根据解码上下文主动选择并融合所有视觉层的特征；

**🔧 技术方法**

使用了LoRA实现可调的多层投影，基于多头注意力的门控机制实现动态注入，整体保持参数占用低；

**📊 数据集**

在18个公开视觉语言基准（包括AI2D、ChartQA、DocVQA、MME、LLaVA‑in‑the‑Wild、OCR‑Bench等）上进行评估；

**📈 对比分析**

与传统单层投影、DeepStack、Shallow‑Layer Injection等方法相比，CLI在LLaVA‑OV‑7B上取得约+9.7分（总分660.6）以及在OCR、视觉 grounding等细粒度任务上提升4.7分，整体性能优于同尺寸基线及多种深度融合方案；

**⚠️ 局限性**

局限性包括：依赖于视觉编码器的层数和特征质量，注入密度与计算成本之间仍有权衡，且对不同LLM/视觉编码器的迁移需要额外验证。

---

## 446. High-accuracy and dimension-free sampling with diffusions

**arXiv ID:** 2601.10708 | [PDF](https://arxiv.org/pdf/2601.10708v1)

**作者:** Khashayar Gatmiry `[一作]` (University of California Berkeley), Adil Salim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出一种新的扩散模型采样器，能够在迭代次数仅为polylog(1/ε)的级别下实现高精度采样；

**💡 创新点**

创新点在于利用分数函数随时间的低阶多项式可逼近性，并结合折点法（Picard迭代）构造高效的数值求解器，从而实现对维度无关且仅与有效半径相关的复杂度；

**🔧 技术方法**

主要技术包括高阶时间导数的闭式表达、低阶多项式逼近、折点法（collocation）与Picard迭代、以及对分数估计误差的亚指数尾部控制；

**📊 数据集**

该工作为理论研究，未使用任何实际数据集；

**📈 对比分析**

相较于现有的poly(d,1/ε)复杂度方法，该方法在1/ε维度上提升为polylog(1/ε)，且复杂度不再显式依赖高维度，能够在混合高斯、低半径等场景下表现更优；

**⚠️ 局限性**

局限性包括对目标分布要求是有界支撑的噪声卷积，且分数估计误差需满足亚指数尾部；在更一般分布或仅满足L2误差时尚未适用。

---

## 447. UFO Trees: Practical and Provably-Efficient Parallel Batch-Dynamic Trees

**arXiv ID:** 2601.10706 | [PDF](https://arxiv.org/pdf/2601.10706v1)

**作者:** Quinten De Man `[一作]` (University of Maryland), Laxman Dhulipala `[通讯]` (University of Maryland)

**通讯引用:** 1347 | [OpenAlex ID](https://openalex.org/A5065818820)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为UFO树的并行批量动态树数据结构，能够支持所有已知的树查询（连通性、路径、子树、最近公共祖先、直径、中心、众数、最近标记顶点等）并兼顾高效的更新

**💡 创新点**

创新点在于：①允许高度结点与任意多个度为1的邻居一次合并，打破传统的二分合并限制；②通过并行树收缩实现工作量与深度可证明的高效；③提供工作效率与低深度的批量更新算法，且实现了低内存占用

**🔧 技术方法**

核心技术包括：并行树收缩（parallel tree contraction）、最大匹配与链表排名、哈希表并行更新、semi‑sort、递归/迭代删除祖先、以及针对高度结点的特殊合并策略

**📊 数据集**

实验使用合成树（路径、完全二叉树、k‑叉树、星型、蒲桃形等）以及真实网络的生成树（USA Roads、ENWiki、StackOverflow、Twitter）

**📈 对比分析**

通过与10种现有动态树实现（link‑cut、top‑tree、Euler tour、RC、topology等）在单线程与多线程批量更新下比较，UFO树在大多数输入下仅比link‑cut慢3.23×，比其他结构慢幅度更小；在并行批量更新中，UFO仅比Euler tour树慢1.95×，且在低直径输入上表现更优；内存占用最低；对直径变化的实验显示其性能随直径下降显著提升

**⚠️ 局限性**

限制在于：对某些不可逆子树查询仍需O(log n)时间；在极高直径或极大度数输入时，虽然保持工作量线性，但常数较大；实现复杂度高，需精细维护哈希表与并行删除；不适用于需要频繁修改查询函数的场景

---

## 448. Communication-Efficient and Privacy-Adaptable Mechanism -- a Federated Learning Scheme with Convergence Analysis

**arXiv ID:** 2601.10701 | [PDF](https://arxiv.org/pdf/2601.10701v1)

**作者:** Chun Hei Michael Shiu `[一作]` (University of British Columbia), Chih Wei Ling `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种联合通信效率与隐私保护的联邦学习方案CEPAM，利用分层拒绝采样通用量化器（LRSUQ）对梯度进行加噪与量化；

**💡 创新点**

创新点在于将噪声模拟与量化合并为一体，既可生成高维高斯或拉普拉斯隐私噪声，又能实现向量量化压缩；并给出完整的隐私保证与收敛分析；

**🔧 技术方法**

采用LRSUQ、差分隐私（高斯/拉普拉斯）、FedAvg+局部SGD、熵编码等技术；

**📊 数据集**

在MNIST数据集上使用CNN模型进行实验；

**📈 对比分析**

与传统FL、FL+SDQ、FL+Gaussian/ Laplace、FL+Gaussian+SDQ等基线对比，CEPAM在测试精度上提升约0.8‑1.1%，并取得更高的信噪比与通信压缩效果；隐私-精度折衷曲线显示当ε提升到一定阈值后精度提升趋缓；

**⚠️ 局限性**

仅在凸优化设置下进行分析；实验规模有限，仅用MNIST；未探讨非凸或个性化联邦学习场景；对局部DP机制的扩展尚待研究。

---

## 449. The Conversational Exam: A Scalable Assessment Design for the AI Era

**arXiv ID:** 2601.10691 | [PDF](https://arxiv.org/pdf/2601.10691v1)

**作者:** Lorena A. Barba `[一作]` (George Washington University), Laura Stegner `[通讯]` (George Washington University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种可扩展的口头考试形式——对话式考试，以在学生使用生成式AI时维持评估有效性。

**💡 创新点**

创新点在于通过实时编码与解释的方式，将真实性和内在有效性嵌入考试流程，并利用HCI原则实现大规模可扩展性。

**🔧 技术方法**

使用了Jupyter Notebook、Zoom多屏共享、预先准备的评分表以及AI工具辅助题库生成等技术。

**📊 数据集**

使用58名学生在两天内的课堂数据，并自行设计了30道分层问题作为评测题库，未采用公开数据集。

**📈 对比分析**

与传统作业相比，对话式考试通过现场观察与即时评分评估，平均分约80%，能够有效区分真正学习与表面表现。

**⚠️ 局限性**

局限包括对时间补偿的处理、对特定学科的适用性、教师人力成本以及对残障学生个别化支持的挑战。

---

## 450. On the origin of neural scaling laws: from random graphs to natural language

**arXiv ID:** 2601.10684 | [PDF](https://arxiv.org/pdf/2601.10684v1)

**作者:** Maissam Barkeshli `[一作]` (Meta Superintelligence Labs), Andrey Gromov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文研究了在随机图上的随机游走序列建模中的神经缩放法则，探讨了模型参数、数据集大小和计算资源之间的关系。

**💡 创新点**

创新点在于首次展示了即使输入数据没有任何幂律结构，变换器在随机图上的下一个标记预测中也能表现出神经缩放法则。

**🔧 技术方法**

使用了2层变换器模型进行下一个标记预测，并采用了最大更新参数化（μ P）方法。

**📊 数据集**

使用了Erdös-Renyi图和Barabási-Albert图的随机游走数据集，以及生成的语言大ram模型数据集。

**📈 对比分析**

与文献中常用的二维Chinchilla公式相比，使用神经网络回归方法获得的拟合效果更好，均方误差（MSE）显著降低，表明该方法在计算最优缩放法则的提取上更为有效。

**⚠️ 局限性**

限制在于当前研究主要集中在简单的随机图模型上，未来需要探索更复杂的数据集和模型架构对神经缩放法则的影响。

---

## 451. Implementation of Oblivious Transfer over Binary-Input AWGN Channels by Polar Codes

**arXiv ID:** 2601.10682 | [PDF](https://arxiv.org/pdf/2601.10682v1)

**作者:** Pin-Hsun Lin `[一作]` (Institute for Communications Technology), Holger Boche `[通讯]` (Institute of Theoretical Information Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

该论文提出了一种基于极化码的二元输入记忆无关信道的2‑1 远程秘密传输（OT）协议，并给出了其信息理论安全证明；

**💡 创新点**

创新点在于通过对极化变换进行结构化的置换，生成不同视角的优良/劣化通道对，从而实现可靠信息传输与对未选消息几乎信息消失的虚拟擦除模型；同时利用随机位注入与保留哈希实现有限块长泄露-可靠性-速率权衡；

**🔧 技术方法**

使用的关键技术包括极化码理论（优良/劣化通道划分、密钥提取的保留哈希、极化变换置换保留）、高斯逼近（GA）计算通道互信息、对数贝叶斯极限（lhl）实现保留哈希、以及对置换的完整结构化表述；

**📊 数据集**

数据集主要为理论BI‑AWGN信道模型，并通过Monte‑Carlo仿真检验泄露与误码率；

**📈 对比分析**

与传统OT实现（如基于纠错码的传统方案）相比，所提出协议在相同信道使用下实现了可观的OT比特/信道使用率（例如SNR 0.187 dB时约9.12 ×10⁻³位/信道使用），同时满足可靠性与信息安全要求；

**⚠️ 局限性**

局限性包括对GA近似的依赖、置换族仅限于极化保持置换导致搜索空间受限、以及在有限块长下仍需对随机位注入与保留哈希进行额外开销，导致实际可实现的OT速率低于极限值。

---

