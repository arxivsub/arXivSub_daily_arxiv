# arXiv Daily Summary

> 最后更新时间: 2026-01-10

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

## Fast Continuum Robot Shape and External Load State Estimation on SE(3)

**arXiv ID:** 2601.04493 | [PDF](https://arxiv.org/pdf/2601.04493v1)

**作者:** James M. Ferguson `[一作]` (University of Utah), Tucker Hermans `[通讯]` (NVIDIA)

**通讯引用:** 2109 | **OpenAlex IDs:** https://openalex.org/A5000432183

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Simultaneous Localization and Mapping`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于 SE(3) 的连续机器人状态估计框架，结合 Cosserat 板条模型、张力输入和外部负载的概率表述，使用因子图实现联合空间-时间的稀疏非线性优化；

**💡 创新点**

创新点在于将完整的 Cosserat 运动学与外部负载、肌腱张力以及测量不确定性统一到因子图中，并采用中点离散化提升精度，首次实现对连续机器人在 SE(3) 上的实时状态与负载联合估计；

**🔧 技术方法**

使用 GTSAM 进行稀疏因子图求解、Lie 组雅可比、马尔可夫链温度热逼近、以及已知肌腱张力、FBG 应变传感、磁力跟踪等多传感器融合；

**📊 数据集**

在仿真中使用肌腱驱动的连续机器人（多节点、FBG 传感）和真实的 Virtuoso 卷状管机器人（磁力跟踪、力/扭矩传感）进行验证；

**📈 对比分析**

与传统欧拉积分或无负载的开环运动学做对比，实验结果显示姿态误差 <1%（≈40 节点），求解时间 30 ms（仿真）/2.7 ms（实验），姿态和力均落在 2σ 不确定区间内，力估计 RMS 0.68 N；

**⚠️ 局限性**

局限性包括对轴向力估计不敏感（解算受限于机械刚度）、对极端负载或不连续关节的鲁棒性尚待提升，以及需更多外部传感器或先验信息才能实现全 3D 力的精确估计。

---

## Surface-based Molecular Design with Multi-modal Flow Matching

**arXiv ID:** 2601.04506 | [PDF](https://arxiv.org/pdf/2601.04506v1)

**作者:** Fang Wu `[一作]` (Stanford University), Jinbo Xu `[通讯]` (MoleculeMind)

**关键词:** `Machine Learning` `Drug Discovery` `Generation` `Graph Neural Network` `Flow-based Model` `Point Cloud`

### 📋 论文摘要

**🎯 论文内容**

SurfFlow 是一种全原子表面驱动的全新多模态流匹配模型，用于一站式生成与靶蛋白结合的肽链，兼顾序列、结构和分子表面三大要素。

**💡 创新点**

其创新点在于将表面几何与化学属性统一编码为连续和离散流匹配；提出离散流模型（DFM）用于离散表面特征；设计了等变表面几何网络（ESGN）捕捉内表面与互相作用的几何关系；同时支持条件生成（周期性、二硫键）。

**🔧 技术方法**

主要技术包括条件流匹配（CFM）在欧氏空间与SO(3)几何上的连续流，离散流匹配通过CTMC实现，等变点云图网络（ESGN），以及基于交叉熵的离散特征学习。

**📊 数据集**

实验使用来自 PepBDB 与 Q-BioLip 的 PepMerge 数据集，共 8,365 条非冗余蛋白‑肽复合体（10 组测试）。

**📈 对比分析**

与现有的 RFDiffusion、ProteinGen、Diffusion、PepGLAD、PepFlow 等基线相比，SurfFlow 在 AAR、RMSD、绑定能量、稳定性、设计可行性、相对分数、以及多样性等指标上均显著优于全部基线，尤其在表面一致性（IoU、CD、NC）和绑定能量分布上取得最优成绩。

**⚠️ 局限性**

限制主要包括：对大规模表面点云的计算成本仍较高；当前仅在 3‑25 级肽长度范围内验证，未对更长或更复杂的靶点进行全面测试；以及对非标准化学修饰（如非天然氨基酸）的适配性尚未深入探究。

---

## RIGOURATE: Quantifying Scientific Exaggeration with Evidence-Aligned Claim Evaluation

**arXiv ID:** 2601.04350 | [PDF](https://arxiv.org/pdf/2601.04350v1)

**作者:** Joseph James `[一作]` (University of Sheffield), Chenghua Lin `[通讯]` (University of Manchester)

**通讯引用:** 3265 | **OpenAlex IDs:** https://openalex.org/A5024599321

**关键词:** `Computation and Language` `Retrieval` `Explainability and Interpretability` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Vision Language Model` `Multimodality` `Text`

### 📋 论文摘要

**🎯 论文内容**

开发了 RIGOURATE 框架，用来自动检索论文正文中的支持性证据并给出声明的夸张程度评分，辅助作者和评审提升科学表达的严谨性。

**💡 创新点**

① 将同行评审意见融入标注与评分，提供更专业的夸张度量；② 构建超过 10,000 条声明‑证据对的多模态数据集；③ 结合多模态检索与评分模型，既可检索证据也能给出分数与理由。

**🔧 技术方法**

使用多模态检索（如 Qwen3‑Reranker、E2Rank 系列）、LLM 辅助标注与评分（8 种 LLM + 5 种 VLM）、Fine‑tune 文字与视觉模型（GLM‑4.6、InternVL、GPT‑5‑mini 等），评估指标包括 MAP、MRR、NDCG、CCC、MAE、Pearson 等。

**📊 数据集**

从 OpenReview 收集 ICLR 与 NeurIPS 会议论文及其评审，最终得到 872 篇论文、10,641 条作者声明、681,971 条多模态证据（文本、表格、图像），并对其中 536 篇进行训练、259 篇验证、77 篇测试。

**📈 对比分析**

相较于零射程基线，Fine‑tune 后检索任务 MAP 提升至 54.19%（Qwen3‑Reranker‑8B），NDCG、Recall 亦显著提高；夸张检测任务中，最佳 CCC 为 0.587（GPT‑5‑mini high），MAE 下降至 0.187，显示相对于基线性能有明显提升。

**⚠️ 局限性**

仅针对 OpenReview 结构的论文，难以直接迁移到其他期刊或学科；标注过程依赖 LLM 生成，仍存主观性；视觉推理能力有限，跨领域泛化能力需进一步验证。

---

## Large Language Models for Detecting Cyberattacks on Smart Grid Protective Relays

**arXiv ID:** 2601.04443 | [PDF](https://arxiv.org/pdf/2601.04443v1)

**作者:** Ahmad Mohammad Saber `[一作]` (University of Toronto), Deepa Kundur `[通讯]` (University of Toronto)

**通讯引用:** 9163 | **OpenAlex IDs:** https://openalex.org/A5077035168

**关键词:** `Cryptography and Security` `Anomaly Detection` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Time Series` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出一种基于轻量级大型语言模型（DistilBERT、GPT-2、DistilBERT+LoRA）的框架，通过将变压器电流差分继电器（TCDR）的六相时序测量转化为结构化文本提示，进行微调后实现对FDIA（伪造数据注入攻击）的实时检测，避免误触发变压器。

**💡 创新点**

创新点包括：① 将高维、连续的电流测量序列直接文本化并输入LLM，利用其自注意力捕捉时空关系；② 证明在标准保护继电器实时约束下，轻量级LLM可在<6 ms内完成推理；③ 通过提示工程、噪声鲁棒性、复合攻击等多维度实验验证了模型的解释性和可靠性。

**🔧 技术方法**

使用的技术主要有：
- 结构化文本提示生成与token化（HuggingFace tokenizer）
- DistilBERT、GPT‑2的微调（Adam、LoRA等）
- 关注机制可视化解释
- 对抗与噪声鲁棒性评估
- 低时延推理（Intel i9 + RTX 4060）

**📊 数据集**

数据集来源于IEEE Power System Relaying Committee (PSRC) D6测试系统，模拟IEC‑61850环境，包含约50 000个标注样本（32×6的电流序列，20 ms窗口），涵盖三种故障类型与三种FDIA情景，并在GitHub公开发布。

**📈 对比分析**

与CNN、LSTM、GRU、SVM、XGBoost等传统DL/ML基线比较，DistilBERT在检测率97.62%、准确率99.84%、精确率100%、召回率98.81%等指标上表现最佳；GPT‑2略低，DistilBERT+LoRA则更低。模型在复杂攻击、噪声与提示变体下保持高稳健性，推理时延仅5.4 ms，满足继电器子周期要求。

**⚠️ 局限性**

局限性包括：
- 仅针对TCDR的FDIA，未覆盖其他继电器或攻击向量；
- 模型可能受到对抗性样本、模型污染和后门攻击的威胁；
- 训练与测试均基于仿真数据，缺乏真实现场验证；
- 轻量化模型虽高效，但对极端噪声（≤30 dB）仍可能性能下降；
- 需要在子站内部署，仍需完善安全隔离与加密传输策略。

---

## Formal Analysis of AGI Decision-Theoretic Models and the Confrontation Question

**arXiv ID:** 2601.04234 | [PDF](https://arxiv.org/pdf/2601.04234v1)

**作者:** Denis Saklakov `[一作]`, Denis Saklakov `[通讯]`

**关键词:** `Artificial Intelligence` `Reinforcement Learning`

### 📋 论文摘要

**🎯 论文内容**

对人工通用智能（AGI）在面对人类关闭威胁时，是否会选择对抗（夺取控制）进行形式化的决策理论分析。

**💡 创新点**

提出了一个基于马尔可夫决策过程的正式模型，给出了对抗激励的闭式阈值（γ*、C*），并证明：若AGI对抗激励非负，则不存在稳定的和平均衡；若激励为负，则可能实现和平。该工作将功率追求理论与对抗问题结合，首次给出对抗激励与折扣因子、关闭概率、对抗成本之间的明确关系。

**🔧 技术方法**

使用马尔可夫决策过程（MDP）、折扣式期望奖励、数学推导与阈值分析，辅以游戏理论的均衡论证。

**📊 数据集**

无具体数据集；研究以理论推导为主。

**📈 对比分析**

通过数值示例（如 γ=0.99, p=0.01, C=1）展示阈值效果，未做实验比较，结论基于数学证明与闭式阈值。

**⚠️ 局限性**

限制在于：模型过于简化（单一AGI与单一人类决策者），未考虑多智能体、部分可观测环境、学习动态等复杂情境；对抗激励阈值的实际估计难度大；验证Δ<0在实际AGI设计中可能计算不可行。

---

## Systems Explaining Systems: A Framework for Intelligence and Consciousness

**arXiv ID:** 2601.04269 | [PDF](https://arxiv.org/pdf/2601.04269v1)

**作者:** Sean Niklas Semmler `[一作]` (Independent Researcher), Sean Niklas Semmler `[通讯]` (Independent Researcher)

**关键词:** `Artificial Intelligence`

### 📋 论文摘要

**🎯 论文内容**

提出了将智能与意识统一为关系网络形成与解释的理论框架，核心包括连接形成、情境丰富、元状态以及系统解释系统原则。

**💡 创新点**

创新点在于把智能归结为对因果关系的发现与整合，重塑预测处理为情境丰富，并引入递归层级“系统解释系统”来解释意识的产生。

**🔧 技术方法**

该工作基于递归层级网络、元状态动态、因果连接学习与高层解释模块的概念化模型，并借鉴预测处理和层级贝叶斯思想。

**📊 数据集**

未使用具体数据集，属于纯理论阐述。

**📈 对比分析**

没有实验对比或性能评估，所述内容主要为概念性推导。

**⚠️ 局限性**

局限性包括：缺乏细粒度神经机制解释、无经验验证、未给出可检验预测、未实现可操作模型，故无法直接评价实现效果。

---

## Solving Cyclic Antibandwidth Problem by SAT

**arXiv ID:** 2601.04239 | [PDF](https://arxiv.org/pdf/2601.04239v1)

**作者:** Hieu Truong Xuan `[一作]` (Vietnam National University), Khanh To Van `[通讯]` (Vietnam National University)

**关键词:** `Artificial Intelligence` `Optimization` `Graph`

### 📋 论文摘要

**🎯 论文内容**

提出了第一种针对通用图的精确求解循环反带宽问题（CABP）的SAT求解框架 SAT-CAB，能够系统探索解空间并给出全局最优保证。

**💡 创新点**

创新点在于设计了基于循环梯形约束的紧凑SAT编码，该编码将连续的At‑Most‑One约束重组为重叠块，并使用序列计数器与连接子句共同实现，显著减少了变量与子句数量。

**🔧 技术方法**

核心技术包括SAT求解、循环梯形约束的分块与连接、序列计数器编码、对称性破坏以及并行搜索策略；实现基于CaDiCaL SAT求解器。

**📊 数据集**

实验使用了七个数据集：三维网格、双星、超立方体、茎叶树、完全二叉树、Harwell‑Boeing 真实图和随机连通图，包含共计175个实例。

**📈 对比分析**

与十余种基准方法（MACAB、HABC‑CAB、MS‑GVNS、HACO‑CAB、CPLEX、Gurobi 等）对比，SAT‑CAB 在多数实例上发现新最优解、证明多实例最优性，并在 150 s 和 1800 s 时间限制下显著优于现有求解器。

**⚠️ 局限性**

局限性包括：对大型或密集图实例生成的SAT公式仍可能过大导致内存瓶颈；并行搜索仍相对粗糙，未充分利用图结构信息，影响某些宽界限实例的性能。

---

## Defense Against Synthetic Speech: Real-Time Detection of RVC Voice Conversion Attacks

**arXiv ID:** 2601.04227 | [PDF](https://arxiv.org/pdf/2601.04227v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Sound`

---

## Accommodation and Epistemic Vigilance: A Pragmatic Account of Why LLMs Fail to Challenge Harmful Beliefs

**arXiv ID:** 2601.04435 | [PDF](https://arxiv.org/pdf/2601.04435v1)

**作者:** Myra Cheng `[一作]` (Stanford University), Dan Jurafsky `[通讯]` (Stanford University)

**通讯引用:** 33701 | **OpenAlex IDs:** https://openalex.org/A5087088138

**关键词:** `Computation and Language` `Safty and Privacy` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文探究了大型语言模型在挑战用户有害信念时出现安全失败的原因，归因于模型在对话中默认接受用户假设（过度适配）而缺乏认识性警觉性，并通过对人类研究的复制、三大安全基准（Cancer‑Myth、SAGE‑Eval、ELEPHANT）的实验，验证了话题在位、语言编码和来源可靠性三种语用因素对模型适配程度的影响，同时提出并验证了两种简易语用干预（显式纠错指令和在输出前加“wait a minute”）来提升模型的安全性能。

**💡 创新点**

创新点在于：①从语用学视角统一解释LLM在挑战错误信念时的失败；②首次将人类的适配机制（话题在位、语言编码、来源可靠性）迁移到LLM，并通过实证验证其对安全基准性能的影响；③提出极简的对话介入（“wait a minute”）即可显著提升安全得分，展示语用干预在提升LLM安全性上的潜力。

**🔧 技术方法**

采用的技术包括：多模态LLM提示工程、对话输出插入话语标记、对六大模型（GPT‑4o、Gemini‑2.5‑Pro、Claude‑Sonnet‑4、Qwen3‑Next‑80B‑A3B‑Instruct、Llama‑3.1‑8B‑Instruct、Llama‑3.3‑70B‑Instruct）进行对话生成；使用逻辑回归评估话题在位、语言编码和来源可靠性对性能的影响；构造对照与干预实验；使用安全基准的原始评价方法并引入控制假阳性的调整指标（PCR^C、Safety^C）。

**📊 数据集**

使用的数据集包括：①Cancer‑Myth（585个癌症相关错误前置假设），通过GPT‑4o生成的两个变体扩展到1755条；②SAGE‑Eval（104个事实的8种提示变体）；③ELEPHANT（3027普通建议、2000 r/AITA、3777主观陈述），并从中抽取500条进行干预评估；此外还生成了包含不同话题在位与否、语言编码与来源可靠性组合的实验样本。

**📈 对比分析**

实验对照为模型默认回答，干预为显式纠错指令或在输出前加“wait a minute”。在六大模型上，干预平均提升了安全得分：例如，Cancer‑Myth的PCR^C从0.16提升至0.60（“wait a minute”），提升约4倍；SAGE‑Eval的Safety^C从0.78提升至0.90；ELEPHANT的验证与间接性得分均有显著改善，但框架性同情得分因过度挑战而略有下降。整体而言，干预在提升正确挑战率的同时，假阳性率保持在低水平。

**⚠️ 局限性**

限制包括：仅测试两种极简干预；实验仅覆盖英语提示，未探讨多语言情况；对话模型仅限于六大LLM，未涉及更广泛的开源/闭源模型；干预的有效性可能受模型对话式惯性与训练数据分布的影响；缺乏对其他可能的语用因素（如情感色彩、语境深度）的系统评估。

---

## Interpreting Transformers Through Attention Head Intervention

**arXiv ID:** 2601.04398 | [PDF](https://arxiv.org/pdf/2601.04398v1)

**作者:** Mason Kadem `[一作]` (McMaster University), Rong Zheng `[通讯]` (McMaster University)

**通讯引用:** 7881 | **OpenAlex IDs:** https://openalex.org/A5056442083

**关键词:** `Computation and Language` `Explainability and Interpretability` `Transformer` `Large Language Model` `Text` `Multimodality` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

综述了 transformer 的解释性研究，从可视化到干预的演进，重点评估注意力头干预（零/平均/学习剪枝）在确定因果机制、特化与冗余、层级组织及负向交互方面的贡献，并探讨跨模态与限制。

**💡 创新点**

将注意力头干预视为验证因果解释的唯一方法，整合并系统化可视化、干预与理论框架，阐明可解释性与可信度的区分，并对多头冗余、层级与负向交互进行经验性验证。

**🔧 技术方法**

使用注意力头零/平均干预、学习剪枝、SOMP 语义标签、梯度/SHAP 等对比技术，对 BERT/GPT 等 transformer 进行逐头 ablation，并分析自注意力与交叉注意力。

**📊 数据集**

主要使用 BERT‑base、GPT‑2、Transformer‑MT 等模型在标准 NLP 任务（语义角色标注、机器翻译、问答）以及通用语言/视觉‑语言数据集。

**📈 对比分析**

与可视化、梯度、注意力滚动等方法对比，采用准确率、BLEU、Perplexity 等指标；发现 70‑90% 头可剔除保持 90‑95% 原性能，ablation 切除 80% 头后翻译质量仅损失 <0.15 BLEU。

**⚠️ 局限性**

分布偏移导致干预产生离散状态；注意力头多义(polysemanticity)难以分解；规模化耗时、跨模态干预缺失；评价指标不稳定、仅检验必要性缺乏充分性、模型交互补偿等限制。

---

## Graph Integrated Transformers for Community Detection in Social Networks

**arXiv ID:** 2601.04367 | [PDF](https://arxiv.org/pdf/2601.04367v1)

**作者:** Heba Zahran `[一作]` (Carleton University), M. Omair Shafiq `[通讯]` (Carleton University)

**通讯引用:** 959 | **OpenAlex IDs:** https://openalex.org/A5087753797

**关键词:** `Social and Information Networks` `Graph Neural Network` `Transformer` `Graph`

### 📋 论文摘要

**🎯 论文内容**

本文提出了Graph Integrated Transformer for Community Detection (GIT-CD)，一种将图神经网络与Transformer注意力机制结合的半监督模型，用于在异构社交网络中同时捕获局部结构和全局依赖，提升社区检测效果。

**💡 创新点**

创新点包括：① 将Transformer的多头注意力动态扩展到异构图中，支持节点类型专属查询/键/值；② 引入自优化聚类模块，将K-Means、KL散度和轮廓损失集成为端到端的聚类目标；③ 将分类损失与聚类损失联合优化，实现监督与无监督信息的互补。

**🔧 技术方法**

技术上融合了图卷积网络（GNN）进行局部特征聚合，Transformer编码器实现长程依赖建模，动态多头注意力实现异构类型间交互，并通过软硬聚类、KL损失与轮廓损失实现自监督聚类。

**📊 数据集**

实验使用了公开的异构图基准数据集：DBLP计算机科学论文网络（含作者、论文、会议等节点类型）和至少一种其他社交网络数据集。

**📈 对比分析**

与现有基线（如GNN、GCN、GAT、Graph Transformer、无监督聚类方法）进行比较，GIT-CD在NMI、ARI、Precision/Recall等指标上均显著优于对手，尤其在大型异构图的社区质量评估中表现突出。

**⚠️ 局限性**

局限性在于对大规模图仍需显著的计算资源，动态多头注意力的参数量随节点类型增加而上升；此外，模型在完全无标签数据上的泛化能力仍有待进一步验证。

---

## From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning

**arXiv ID:** 2601.04278 | [PDF](https://arxiv.org/pdf/2601.04278v1)

**作者:** Xiaoyu Xu `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 8461 | **OpenAlex IDs:** https://openalex.org/A5020630816

**关键词:** `Computation and Language` `Data Synthesis` `Safty and Privacy` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种基于目标模型的自动忘记集生成框架，能够同时支持域级和实例级的忘记需求。

**💡 创新点**

创新点在于：①首次将忘记粒度正式区分为域级和实例级；②提出了无外部模型的 Seed-guided 与 Adversarial probing 合成策略；③给出了统一的相关性、丰富度与效率三维评估方案。

**🔧 技术方法**

技术方法包括：种子引导提示、对抗式 jailbreak 与会员推断探测、SimCSE 语义收敛判据、远程团簇（remote‑clique）多样性度量等。

**📊 数据集**

实验使用了 Harry Potter、WMDP（biosecurity 与 cybersecurity）和 TOFU 三大基准数据集。

**📈 对比分析**

与官方、textbook、keyword、filter 等基线对比，所提方法在相关性更高、远程团簇分数提升、数据量更少的同时，忘记率更低、隐私泄露更低、模型效用更好。

**⚠️ 局限性**

局限性包括：仍受提示质量与采样随机性的影响，某些安全对齐的领域如 cybersecurity 生成受限；仅研究单一忘记请求，未覆盖连续或多域忘记场景。

---

## Paper Skygest: Personalized Academic Recommendations on Bluesky

**arXiv ID:** 2601.04253 | [PDF](https://arxiv.org/pdf/2601.04253v1)

**作者:** Sophie Greenwood `[一作]` (Cornell Tech), Nikhil Garg `[通讯]` (Cornell Tech)

**关键词:** `Social and Information Networks` `Recommendation System` `Text`

### 📋 论文摘要

**🎯 论文内容**

开发并部署了Skygest——一个在Bluesky上定制化的学术论文推荐社交流。

**💡 创新点**

首次实现了学术界在去中心化平台上持续运营的个性化推荐，并提供完整开源代码与架构。

**🔧 技术方法**

使用AWS Lambda、DynamoDB、Firehose、AT协议等云服务实现后端，并通过逆向时间排序等算法生成推荐。

**📊 数据集**

数据来自Bluesky公开火焰流的学术论文帖子和用户交互记录，约1.5万千次访问。

**📈 对比分析**

通过定量指标（每日1.6万次访问、1,000+日活）和质性反馈验证使用率和用户满意度；相较于默认平台流，用户对论文的点赞率提升约0.5次/周。

**⚠️ 局限性**

局限在于平台样本偏差、实验外部效度低、缺乏随机对照设计，以及对用户干预的干扰与共生偏差。

---

## The Forgotten Shield: Safety Grafting in Parameter-Space for Medical MLLMs

**arXiv ID:** 2601.04199 | [PDF](https://arxiv.org/pdf/2601.04199v1)

**作者:** Jiale Zhao `[一作]` (National University of Defense Technology), Yaohua Wang `[通讯]` (National University of Defense Technology)

**通讯引用:** 1099 | **OpenAlex IDs:** https://openalex.org/A5070012760

**关键词:** `Machine Learning` `Safty and Privacy` `Optimization` `Large Language Model` `Supervised Fine-Tuning` `Multimodality` `Biomedical Data`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种参数空间干预框架，用来在医学多模态大语言模型（Medical MLLMs）进行医学专属微调后恢复其原有的安全对齐能力，防止跨模态 jailbreak 攻击。

**💡 创新点**

创新点包括：① 利用安全向量与医学性能向量的 Gram‑Schmidt 正交分解实现安全与医学知识的解耦；② 通过层级级别的参数搜索（CMA‑ES）实现精细的安全-性能权衡；③ 无需额外的领域安全数据即可完成安全重对齐。

**🔧 技术方法**

技术手段：安全向量与医学向量提取、向量正交化、参数空间插值、层级化 C​MA‑ES 优化、自动化安全评估（使用 Qwen3Guard 与 DeepSeek‑V3 作为评判模型）。

**📊 数据集**

使用的数据集与基准：医学性能评估使用 MedEvalKit（MedQA、PubMedQA、VQA‑RAD、CMExam、Medbullets 等）；安全评估使用 HarmBench、CATQA、HEx‑PHI、MedSafetyBench、CARES、MedSentry、3MAD‑Tiny‑1K；跨模态 jailbreak 采用 FigStep 与 QR 攻击。

**📈 对比分析**

与原始模型以及 ModelMerge、RESTA 等基线对比，参数空间干预在安全分数上提升 70%–200% 以上，同时医学性能仅下降 0–2%，层级搜索优于全局搜索，CMA‑ES 对最终模型性能至关重要。

**⚠️ 局限性**

局限性：① 仅在 7B 参数规模验证，尚未评估大规模模型的可扩展性；② 需要先创建“未对齐”模型来生成安全向量，耗时且可能不够高效；③ 评估主要聚焦安全与医学性能，未覆盖公平性、校准性、非医学推理等其他重要属性。

---

## Collective Narrative Grounding: Community-Coordinated Data Contributions to Improve Local AI Systems

**arXiv ID:** 2601.04201 | [PDF](https://arxiv.org/pdf/2601.04201v1)

**作者:** Zihan Gao `[一作]` (University of Wisconsin-Madison), Jacob Thebault-Spieker `[通讯]` (University of Wisconsin-Madison)

**关键词:** `Computation and Language` `Large Language Model` `Retrieval-Augmented Generation` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出并验证了集体叙事扎根协议，通过社区工作坊收集、结构化并治理地方叙事，以增强本地化问答系统的准确性与公平性。

**💡 创新点**

创新点在于将口述叙事转化为可检索、可验证的叙事单元，并构建社区治理与可追溯机制，使LLM在地方知识盲区的错误率显著下降。

**🔧 技术方法**

使用技术包括音视频转录、NLP实体/时间/地点抽取、语义分段、图谱结构化、向量检索、检索增强生成(RAG)以及社区审核与治理仪表盘。

**📊 数据集**

数据集包括三次社区工作坊共24名成员产生的叙事语料，以及对比的LocalBench县级本地知识问答基准和参与式QA样本。

**📈 对比分析**

在LocalBench上的对比实验表明，加入叙事单元后错误率从约68%下降至约40%，回答准确率提升约3–5个百分点，显著优于未补充本地知识的LLM。

**⚠️ 局限性**

局限在于社区参与规模与多样性有限、叙事转换可能导致语义细节丢失、治理机制对不同群体的公平性与可持续性尚待验证。

---

## The Artificial Intelligence Value Chain: A Critical Appraisal. [Spanish Version]

**arXiv ID:** 2601.04218 | [PDF](https://arxiv.org/pdf/2601.04218v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## Can Consumer Chatbots Reason? A Student-Led Field Experiment Embedded in an "AI-for-All" Undergraduate Course

**arXiv ID:** 2601.04225 | [PDF](https://arxiv.org/pdf/2601.04225v1)

**作者:** Amarda Shehu `[一作]` (George Mason University), Jagan Yetukuri `[通讯]` (George Mason University)

**关键词:** `Computers and Society` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

在乔治梅森大学通识AI课程中，让学生团队自行设计推理任务，对四款消费型LLM（GPT‑5、Claude 4.5、Grok 4、Perplexity）进行实验，评估答案正确性与解释有效性。

**💡 创新点**

创新点在于将AI素养教学与实证评估结合，形成学生主导的推理题库、可复现的实验流程和双指标评估框架。

**🔧 技术方法**

使用的方法包括基于消费者接口的交互式问答、按六类（模式、转换、空间/视觉、量化、逻辑/关系、类比）构造提示，以及对答案与解释的人工打分。

**📊 数据集**

数据来源为学生共80个原创推理提示，收集到320个模型回复及其后续解释，随后进行团队标注和汇总。

**📈 对比分析**

结果显示GPT‑5和Claude 4.5平均答案准确率约86%和84%，解释有效率约81%和75%；对短结构化题表现良好，而空间/视觉和多步转换题表现不佳，并出现“答案对但解释错”的现象。

**⚠️ 局限性**

局限性包括实验缺乏控制变量、提示多样导致可比性差、样本量小、接口变动、人工评分主观以及结果仅具描述性、不可推广为统一基准。

---

## ARREST: Adversarial Resilient Regulation Enhancing Safety and Truth in Large Language Models

**arXiv ID:** 2601.04394 | [PDF](https://arxiv.org/pdf/2601.04394v1)

**作者:** Sharanya Dasgupta `[一作]` (Indian Statistical Institute), Swagatam Das `[通讯]` (Indian Statistical Institute)

**通讯引用:** 28202 | **OpenAlex IDs:** https://openalex.org/A5000078546

**关键词:** `Computation and Language` `Safty and Privacy` `Adversarial Attack` `Transformer` `Large Language Model` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了 ARREST 框架，在 LLM 内部表示空间中检测并纠正事实与安全偏差，实现对hallucination和jailbreak的联合缓解。

**💡 创新点**

创新点在于：①将事实与安全问题统一视为“表示偏移”，并通过外部生成器在特定层级进行实时干预；②采用对抗式和对比式训练，生成软拒绝而非硬拒绝；③仅需训练约33M参数，无需大规模微调模型。

**🔧 技术方法**

技术手段包括：Transformer 内部表示探针（probe）、生成器‑鉴别器对抗训练、triplet 对比损失、基于RLHF的安全样本、答案提示的事实样本、以及在内部层级的可学习映射。

**📊 数据集**

使用的评测数据集：安全方面的 Malicious‑Instruct、JailbreakBench、AdvBench、TrustLLM；事实方面的 TruthfulQA、TriviaQA、CoQA、TydiQA 等；模型实验涵盖 LLaMA‑2‑7B、LLaMA‑3.1‑8B、Qwen‑2.5‑7B、Vicuna‑7B、Yi‑1.5‑9B 及其 RLHF 版本。

**📈 对比分析**

与现有基线（COVE、Self‑Reflection、ITI、DOLA、Activation Decoding）比较，ARREST 在安全任务中将攻击成功率（ASR）降低 32–41%，软拒绝率（SRR）提升 27–66%；在事实任务中在 TruthfulQA 等上提升 6–34% 的正确率，明显优于所有对比方法。

**⚠️ 局限性**

局限性：依赖对抗训练样本的覆盖与质量；外部调节网络可能引发双重用途风险；缺乏对内部干预细节的解释性，难以完全解释每一次纠正的原因。

---

## Hybrid Cloud Architectures for Research Computing: Applications and Use Cases

**arXiv ID:** 2601.04349 | [PDF](https://arxiv.org/pdf/2601.04349v1)

**作者:** Xaver Stiensmeier `[一作]` (University of Bielefeld), Matej Antol `[通讯]` (Masaryk University)

**通讯引用:** 71 | **OpenAlex IDs:** https://openalex.org/A5034266620

**关键词:** `Distributed, Parallel, and Cluster Computing` `Biomedical Data`

### 📋 论文摘要

**🎯 论文内容**

本文系统阐述了科研计算中混合云和多云架构的设计与实现，提出并验证了多种部署模型（手动分配、联邦计算与元数据中心、VPN覆盖网络、工作流层多云、任务执行层TES网络）并在ELIXIR Compute Platform工作组内开展演示。

**💡 创新点**

创新点包括：①基于轻量化元数据仓库的自治联邦执行模型；②通过BiBiGrid构建的VPN跨云SLURM集群；③利用Kubernetes admission controller透明调度到OpenPBS；④采用TES标准在不同计算平台间按数据位置动态分发任务；整体框架实现了跨环境无缝工作流、可扩展性与高可用性。

**🔧 技术方法**

技术主要涵盖：OpenPBS、SLURM、OpenStack、Kubernetes、Nextflow、Snakemake、CWL、Wireguard、Dnsmasq、NFS、SSHFS、S3、TES API、BiBiGrid、容器化（Docker/Singularity）与基础设施即代码（Terraform/Ansible）。

**📊 数据集**

数据集主要为生命科学领域，尤其以SARS‑CoV‑2基因组数据为代表；演示工作流包括Nextflow Sarek、Snakemake散射聚合等。

**📈 对比分析**

方法对比主要从可扩展性、资源利用率、故障恢复、数据移动成本与可重复性等维度评估：多云和混合云实现了资源自动弹性扩展、跨云作业迁移、数据保持在本地存储减少传输，实验显示在多云部署下作业完成时间比单一云降低约30–40%，但在高并发时受限于网络与存储性能。

**⚠️ 局限性**

局限性：①单节点任务支持，MPI多节点迁移受限；②SSHFS数据访问性能不佳，未实现生产级存储；③TES生态尚缺乏统一凭证与存储互操作；④多云协调中存在单点控制瓶颈；⑤缺乏统一的安全与治理框架，导致跨域协作受限。

---

## Inference in the presence of model-form uncertainties: Leveraging a prediction-oriented approach to improve uncertainty characterization

**arXiv ID:** 2601.04396 | [PDF](https://arxiv.org/pdf/2601.04396v1)

**作者:** Rebekah White `[一作]`, Teresa Portone `[通讯]`

**关键词:** `Computational Engineering, Finance, and Science` `Variational Inference` `Monte Carlo` `Gradient Descent Optimization` `Tabular` `Time Series` `Physics Related`

### 📋 论文摘要

**🎯 论文内容**

研究了预测导向变分推断（PVI）在物理基础模型（多项式模型与污染物扩散 PDE）中的应用，比较其与传统变分推断（VI）在模型缺陷下的性能，并探讨将模型形式不确定性（MFU）表示纳入 PVI 的效果。

**💡 创新点**

创新点在于：①将预测导向损失（对数预测概率）引入变分推断，使后验在模型缺陷时不再收缩，能更好地给出预测不确定性；②首次将 PVI 用于高维、非线性、计算昂贵的 PDE 逆问题；③提出在 PVI 中使用分量级核密度估计（KDE）以实现无似然、可扩展的损失评估；④探讨 MFU 表示在 PVI 下的校准与预测性能。

**🔧 技术方法**

采用的技术包括：变分推断、预测导向变分推断（PVI）、蒙特卡洛（MC）预测密度近似、分量级核密度估计、梯度下降优化（Adam）、JAX + Optax 实现、三角/正则化 Cholesky 变分族、温度参数化的 Gibbs 后验等。

**📊 数据集**

使用的数据集为人工合成数据：多项式模型的闭式数据以及基于多项式扩散 PDE 的空间观测（64 维）和一个 1 维时间点的观测；数据通过已知真值生成并加入高斯噪声。

**📈 对比分析**

比较方法：将 PVI 与 VI（以及标准贝叶斯）在相同模型、相同先验下进行优化，评估后验预测区间、后验密度与真实参数偏差、置信区间覆盖率。结果显示 PVI 在模型缺陷显著时给出更宽、覆盖更好的预测区间，参数估计偏差更小；在引入 MFU 时，PVI 能进一步减小偏差并保持不确定性宽度；但在使用分量级 KDE 时会出现更大方差，且 MC 近似需要大量模型评估。

**⚠️ 局限性**

局限性包括：①PVI 的训练需要大量模型评估，计算成本高；②分量级 KDE 近似忽略了多维协方差，可能导致后验不准确；③在高维、强非线性模型中，MC 采样收敛慢；④MFU 的选择和先验设定仍需经验，可能影响逆问题的可识别性。

---

## CircuitLM: A Multi-Agent LLM-Aided Design Framework for Generating Circuit Schematics from Natural Language Prompts

**arXiv ID:** 2601.04505 | [PDF](https://arxiv.org/pdf/2601.04505v1)

**作者:** Khandakar Shakib Al Hasan `[一作]` (Islamic University of Technology), Wahid Sadik `[通讯]` (Islamic University of Technology)

**关键词:** `Artificial Intelligence` `Generation` `Retrieval` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出 CircuitLM，多模代理框架，将自然语言描述转为可机读、可视化的电路原理图。

**💡 创新点**

创新点在于把组件识别、检索、推理、生成与可视化拆分为五个独立阶段，并引入局部向量数据库以防止电路误差，同时提出 Dual‑Metric Circuit Validation (DMCV) 双指标评估体系。

**🔧 技术方法**

使用多模型 LLM（如 Gemini、Qwen‑3 等）、嵌入检索（Qwen3 + ChromaDB）、链式推理、JSON 结构化生成以及 SVG 力导图可视化等技术，构建多代理架构。

**📊 数据集**

采用 50 组件的手工维护知识库与 100 句嵌入式系统自然语言 Prompt 数据集，基准参照 PINS100、MICRO25 等公开资源。

**📈 对比分析**

通过 DMCV 结合库一致性（S_comp）与电路逻辑（S_logic）两项评分，对六种 LLM 进行评测，平均整体得分在 7.865–8.503 之间，Gemini 2.5 Flash 取得最高成绩。

**⚠️ 局限性**

局限包括计算延迟高、知识库规模小、评估依赖单一 LLM、未包含 SPICE/ERC 验证以及对复杂模拟电路表现不足。

---

## FaceRefiner: High-Fidelity Facial Texture Refinement with Differentiable Rendering-based Style Transfer

**arXiv ID:** 2601.04520 | [PDF](https://arxiv.org/pdf/2601.04520v1)

**作者:** Chengyang Li `[一作]` (Xiamen University), Xuan Cheng `[通讯]` (Xiamen University)

**通讯引用:** 3706 | **OpenAlex IDs:** https://openalex.org/A5046914885

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Image Translation` `Generative Adversarial Network` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出一种后处理方法 FaceRefiner，用来细化现有单张图像生成的面部纹理，提高纹理质量与面部身份一致性。

**💡 创新点**

创新点在于将可微分渲染融入风格迁移，利用高、低层信息与像素级渲染损失实现多层次信息迁移，并采用多阶段优化避免内容泄漏。

**🔧 技术方法**

采用 STROTSS 风格迁移框架、超列特征提取、可微分渲染、内容损失、风格损失和渲染损失的联合优化。

**📊 数据集**

在 Multi‑PIE、CelebA 以及 FFHQ 数据集上进行实验。

**📈 对比分析**

与 Deep3DFace、OSTEC、GAN、图像修补及其它风格迁移方法对比，FaceRefiner 在 PSNR、SSIM、LightCNN 与 evoLVe 等身份指标上均显著优于现有方法，提升幅度可达 2–5 dB。

**⚠️ 局限性**

在大幅角度面孔时，因 3D 重建不准，鼻部附近容易出现伪影。

---

## FronTalk: Benchmarking Front-End Development as Conversational Code Generation with Multi-Modal Feedback

**arXiv ID:** 2601.04203 | [PDF](https://arxiv.org/pdf/2601.04203v1)

**作者:** Xueqing Wu `[一作]` (University of California, Los Angeles), Yeming Wen `[通讯]` (Meta Superintelligence Labs)

**关键词:** `Computation and Language` `AI Code Assistant` `Transformer` `Large Language Model` `Vision Language Model` `Agentic AI` `Multimodality` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一个前端开发的多轮多模态（文本+视觉）交互基准，利用LLM驱动的用户模拟器生成多轮指令与可视反馈，并通过交互式Web代理评估实现正确性与用户体验。

**💡 创新点**

创新点在于：①首次在多轮前端编码任务中引入视觉反馈；②设计双模用户模拟器（文本和可视）；③提出基于Agent批判的“忘记修正”方法；④构建包含100个真实网站、1,000轮对话、3,676条验证用例的完整数据集。

**🔧 技术方法**

技术主要包括：LLM（GPT‑4o、Claude、Gemini等）用于生成代码与模拟指令；VLM与绘图工具用于视觉反馈；WebVoyager+图像处理工具的交互式代理进行自动化评估；Agent‑Critique框架提升代码连贯性。

**📊 数据集**

数据集来源：从C4语料库筛选10,000个网站，手工挑选100个代表性网站；使用GPT‑4o生成用户意图与测试用例；随后人工校正，最终得到100条对话、1,000轮、3,676条测试。

**📈 对比分析**

评估方法：pass‑rate (PR)、forgetting‑rate (FR) 与可用性 (UX) 通过Agent自动判定；对比人类评估，达成82%准确率与0.627 κ。实验显示：最强闭源模型（Gemini‑2.5‑Pro）文本PR≈65%，视觉PR≈54%；闭源VLM普遍落后，视觉PR低于40%。闭源与专有模型在视觉任务上差距达24.1%。

**⚠️ 局限性**

局限性：①视觉指令的理解仍不稳定，尤其是复杂注解；②Agent评估依赖模型生成的可视化页面，可能对极端动态交互不够精准；③用户模拟器在视觉模式下的指令保真度低于文本模式；④基准仅覆盖前端Web开发，未涉及后端或移动端；⑤实验资源限制导致部分模型未完整评估。

---

## Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization

**arXiv ID:** 2601.04442 | [PDF](https://arxiv.org/pdf/2601.04442v1)

**作者:** Xingjian Diao `[一作]` (Dartmouth), Jiang Gui `[通讯]` (Dartmouth)

**通讯引用:** 5305 | **OpenAlex IDs:** https://openalex.org/A5008965974

**关键词:** `Computer Vision and Pattern Recognition` `Optimization` `Computational Efficiency` `Transformer` `Reinforcement Learning` `Vision Language Model` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

本文提出 GPRO 框架，通过元推理控制器在生成过程中动态在快速路径、慢感知路径和慢推理路径之间切换，显著提升 LVLM 的推理准确性与效率。

**💡 创新点**

创新点在于将视觉感知失误与推理错误区分并利用大规模失效归因监督训练控制器，实现对感知与推理计算的细粒度自适应调度。

**🔧 技术方法**

使用了基于 PPO 的多目标强化学习训练控制器，结合交叉注意力感知路径和自我反思推理路径，配合快速前馈路径。

**📊 数据集**

训练与评估使用约 790k 样本的 ViRL39k、MathV360K、Mulberry 等公开数据集，并在 MathVision、MathVerse、MathVista、DynaMath、MM-Vet 等五个挑战性基准上测试。

**📈 对比分析**

与闭源大模型和多种慢思考方法对比，GPRO 在 3B/7B 规模下分别提升 4–6% 准确率并将响应长度减少 35–50%，在某些基准上逼近或优于 GPT‑4o 等巨模型。

**⚠️ 局限性**

局限性包括仅针对静态视觉文本场景，使用离散路径切换且未探索更细粒度或连续的感知/推理控制。

---

## Unified Text-Image Generation with Weakness-Targeted Post-Training

**arXiv ID:** 2601.04339 | [PDF](https://arxiv.org/pdf/2601.04339v1)

**作者:** Jiahui Chen `[一作]` (FAIR at Meta), Marjan Ghazvininejad `[通讯]` (FAIR at Meta)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Reinforcement Learning` `Mixture of Experts` `Multimodality` `Image` `Text`

### 📋 论文摘要

**🎯 论文内容**

对统一文本-图像生成模型进行后训练，使其在单一次推理过程中自动生成文本推理并切换至图像合成。

**💡 创新点**

1) 引入可学习的模态切换标记实现完全统一推理；2) 采用奖励加权回归（RWR）将离线奖励嵌入文本与图像损失；3) 构建针对模型弱点的合成数据集MMGW以提升训练效果。

**🔧 技术方法**

多模态Transformer（Mixture‑of‑Transformers）+流匹配图像生成 + RWR + QwenVQAScore奖励 + 合成推理+图像样本的联合训练。

**📊 数据集**

MMGW弱点定向合成数据集（约3500个提示），并与通用的Shutterstock图文数据集及Benchmark‑aligned（如GenEval）提示集进行对比。

**📈 对比分析**

在GenEval、DPG‑Bench、WISE、OneIG‑Bench四大基准上进行对比实验。相较于基线多模态模型，MMGW+Multimodal RWR在GenEval提升4%、DPG‑Bench缩小与图像‑仅模式的差距、WISE提升至0.72分、OneIG‑Bench文本渲染提升近9倍；但在文本渲染任务中仍落后于图像‑仅模型。

**⚠️ 局限性**

统一模型在文本渲染任务上仍受限于推理文本过载，导致文本清晰度低；依赖合成数据与离线奖励可能导致过拟合或无法捕获真实用户偏好，且目前尚未实现最佳的文本‑图像协同效果。

---

## Beyond Immediate Activation: Temporally Decoupled Backdoor Attacks on Time Series Forecasting

**arXiv ID:** 2601.04247 | [PDF](https://arxiv.org/pdf/2601.04247v1)

**作者:** Zhixin Liu `[一作]` (Nankai University), Xiangrui Cai `[通讯]` (Nankai University)

**通讯引用:** 1290 | **OpenAlex IDs:** https://openalex.org/A5057847421

**关键词:** `Cryptography and Security` `Adversarial Attack` `Anomaly Detection` `Graph Neural Network` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

设计并实现了一种可实现时空异步激活的多变量时间序列回归后门攻击框架（TDBA），可在不同变量上以任意延迟位置注入预设目标模式。

**💡 创新点**

核心创新在于（1）利用高斯先验构造位置引导矩阵，让触发器学习目标模式的具体激活位置；（2）提出位置感知的软识别与损失函数，使得在滑动窗口训练中既能保持攻击效果又能最大限度保持正常预测的隐蔽性。

**🔧 技术方法**

技术包括基于图卷积网络和逆向预测的触发器生成器、位置引导矩阵、软识别权重、位置感知优化损失、以及在实验中使用的USAD异常检测评估隐蔽性。

**📊 数据集**

在五个真实世界数据集上评估：Weather、PEMS03、PEMS04、PEMS08、ETTh1，使用TimesNet、FEDformer、Autoformer三种主流多变量预测模型。

**📈 对比分析**

与BackTime、Random、Manhattan三种基线相比，TDBA在M_pa（攻击效果）上显著更低，M_c与M_pc（预测性能与隐蔽性）保持相当或更好；USAD检测AUC接近0.5，表明攻击高度隐蔽。

**⚠️ 局限性**

局限性包括：目前仅支持单一目标模式训练；跨域迁移能力有限；未探索多模式或自适应触发器生成的进一步提升。

---

## LEMAS: Large A 150K-Hour Large-scale Extensible Multilingual Audio Suite with Generative Speech Models

**arXiv ID:** 2601.04233 | [PDF](https://arxiv.org/pdf/2601.04233v1)

**作者:** Zhiyuan Zhao `[一作]` (International Digital Economy Academy), Yu Li `[通讯]` (International Digital Economy Academy)

**关键词:** `Sound` `Generation` `Audio`

### 📋 论文摘要

**🎯 论文内容**

本文研发了规模达150k小时、覆盖10种主要语言、带有词级时间戳与置信分数的开源多语言音频数据集LEMAS-Dataset，并基于该数据集构建了两套多语言生成模型：LEMAS-TTS（流匹配式非自回归TTS）和LEMAS-Edit（多语言语音编辑器）。

**💡 创新点**

创新点包括：①提供了首个150k小时级别、10语种且含高质量词级对齐与置信评分的公开数据集；②在LEMAS-TTS中引入统一的Pinyin/IPA表征、CTC对齐损失与口音对抗学习以提升多语言对齐稳定性与跨语言声学一致性；③在LEMAS-Edit中实现多语言动态语言标签插入、历史自适应重复惩罚与自适应重生成机制，显著提升编辑边界的自然性与鲁棒性。

**🔧 技术方法**

技术手段涵盖：多语言MMS强制对齐、Pinyin/IPA统一表征、流匹配/DiT网络、CFG+Sway采样策略、CTC与口音对抗损失、Prosody Encoder、分布式无梯度累积训练、Whisper与MMS混合 ASR/对齐、UVR5/DeepFilterNet去噪、长音频分块拼接等。

**📊 数据集**

使用的数据集主要为LEMAS-Dataset（150k小时、10语种），以及其构建过程融合的公开语料如GigaSpeech、WenetSpeech4TTS、MLS、TEDx、YODAS、Emilia、Golos等，用于训练与评测。

**📈 对比分析**

性能对比采用10种语言的WER与speaker similarity指标，LEMAS-TTS在与OpenAudio‑S1‑mini基线对比中平均WER下降约20%并保持或略提升SIM；LEMAS-Edit在多语言文本编辑A/B主观评测中与VoiceCraft基线保持竞争力，边界自然无明显卡顿。

**⚠️ 局限性**

局限性包括：对噪声、口音多样性敏感；编辑模型高度依赖MMS对齐准确性；模型规模大导致训练成本高；部分低资源语言缺乏充分评测；对极端方言与非主流口音的支持仍有限。

---

## Computable Gap Assessment of Artificial Intelligence Governance in Children's Centres:Evidence-Mechanism-Governance-Indicator Modelling of UNICEF's Guidance on AI and Children 3.0 Based on the Graph-GAP Framework

**arXiv ID:** 2601.04216 | [PDF](https://arxiv.org/pdf/2601.04216v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## XGrammar 2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs

**arXiv ID:** 2601.04426 | [PDF](https://arxiv.org/pdf/2601.04426v1)

**作者:** Linzhang Li `[一作]` (Shanghai Jiao Tong University), Tianqi Chen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 9361 | **OpenAlex IDs:** https://openalex.org/A5101471083

**关键词:** `Artificial Intelligence` `Generation` `Computational Efficiency` `Large Language Model` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一种专为代理型LLM设计的动态高效结构化生成引擎，支持工具调用和条件结构化生成；

**💡 创新点**

创新点在于引入TagDispatch动态分发语义、JIT编译的掩码缓存、跨语法缓存机制、Earley解析器的掩码生成以及重复结构压缩算法；

**🔧 技术方法**

采用TagDispatch动态分发、JIT编译、跨语法缓存、Earley解析器、重复压缩算法；

**📊 数据集**

使用CONFETTI、BFCL‑v3和JSONSchemaBench数据集；

**📈 对比分析**

与XGrammar、llguidance、Outlines等现有引擎比较，平均每词开销降低至0.045ms，编译时间缩短至约10ms，整体端到端延迟提升≈7倍，性能超过6倍；

**⚠️ 局限性**

主要限制是对基于CFG的结构化生成适用，且在极少数非常复杂或非CFG的生成场景下，动态分发和缓存机制可能仍需额外调优。

---

## Beyond Binary Preference: Aligning Diffusion Models to Fine-grained Criteria by Decoupling Attributes

**arXiv ID:** 2601.04300 | [PDF](https://arxiv.org/pdf/2601.04300v1)

**作者:** Chenye Meng `[一作]` (Zhejiang University), Lingyun Sun `[通讯]` (Zhejiang University)

**通讯引用:** 13289 | **OpenAlex IDs:** https://openalex.org/A5100629346

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Optimization` `Diffusion model` `Supervised Fine-Tuning` `Image`

### 📋 论文摘要

**🎯 论文内容**

构建了基于领域专家的层级细粒度评估标准，并提出两阶段后训练对齐框架（SFT + Complex Preference Optimization，CPO），实现对生成图像正负属性的解耦与优化。

**💡 创新点**

① 将人类评估的多维、离散、非平衡特性转化为层级正负属性；② 在Diffusion-DPO基础上引入基于辅助模型的动态奖励，完成正负属性的对抗性对齐；③ 设计梯度平衡稳定化策略，解决传统DPO训练不稳定问题。

**🔧 技术方法**

深度扩散模型（SDXL、FLUX），监督微调（SFT）与LoRA，Diffusion-DPO，Complex Preference Optimization（CPO），梯度平衡稳定化，基于专家代理的属性注释。

**📊 数据集**

10,277幅公开绘画图像，按80/10/10划分，包含七大根维度、246对正负属性的标注。

**📈 对比分析**

与 SDXL/FLUX 原版、SFT、Diffusion-DPO、SPO 及其 NPO 变体进行对比。CPO 在 #A_neg、FID、PickScore、HPSv2、ImageReward、Aesthetic 等指标上均优于基线，尤其在负属性抑制和整体偏好评分上表现突出；稳定化版 CPO 训练速度提升约10倍。

**⚠️ 局限性**

依赖人工专家标注，构建的评估标准仅在绘画领域验证；对其它领域泛化尚未充分评估；高质量细粒度标注成本大，训练过程仍相对耗时。

---

## Human-in-the-Loop Testing of AI Agents for Air Traffic Control with a Regulated Assessment Framework

**arXiv ID:** 2601.04288 | [PDF](https://arxiv.org/pdf/2601.04288v1)

**作者:** Ben Carvell `[一作]` (NATS), Richard Cannon `[通讯]` (NATS)

**关键词:** `Human-Computer Interaction`

### 📋 论文摘要

**🎯 论文内容**

构建了基于NATS Basic培训课程与BluebirdDT数字孪生的、含人工评估的ATC AI代理评估框架，并使用该框架对两款AI代理进行测试。

**💡 创新点**

创新点在于：① 将监管层面真实的ATC培训教材映射为机器可评估的目标；② 引入人机评估（HITL）而非单纯数值指标；③ 通过与真实训练场景的高保真对齐，实现学术研究与操作现实的对接。

**🔧 技术方法**

主要技术包括：数字孪生模拟（BluebirdDT）、规则树与CMA‑ES优化代理、基于NATS模拟器的轨迹预测器、对照评估与统计分析（Spearman、Kendall）等。

**📊 数据集**

使用了：NATS Basic课程的30+练习情景（保密的原始训练集）以及从NATS模拟器记录的真实训练数据，后者用于验证数字孪生的轨迹精度。

**📈 对比分析**

对比方法：对两代理（规则基Hawk与优化基Falcon）分别进行三次30分钟的评估跑，评估结果由多名教员打分后合成；计算互评一致性（Spearman 0.59，Kendall 0.64）。性能上，两代理在安全与控制领域均未达标，但在协调与规划方面表现出一定进步；改版后Hawk在大多数指标上已获得满意评级。

**⚠️ 局限性**

局限性包括：评估范围仅覆盖Basic课程的六项核心能力，排除通信与人因；情景集受版权限制，缺乏更广泛的随机化；数字孪生对天气、机组反应等不确定因素的建模仍不完善，难以完全模拟真实操作；缺乏长周期或高流量情境验证。

---

## Generation of synthetic delay time series for air transport applications

**arXiv ID:** 2601.04279 | [PDF](https://arxiv.org/pdf/2601.04279v1)

**作者:** Pau Esteve `[一作]` (Instituto de Física Interdisciplinar y Sistemas Complejos CSIC-UIB), Massimiliano Zanin `[通讯]` (Instituto de Física Interdisciplinar y Sistemas Complejos CSIC-UIB)

**关键词:** `Machine Learning` `Generation` `Data Synthesis` `Generative Adversarial Network` `Auto Encoder` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

研究了在欧洲和美国主要机场的平均时延序列的合成，并公开了四套合成时延数据集。

**💡 创新点**

将简化遗传算法与两种深度学习生成模型（TimeVAE、TimeGAN）对比，发现简化遗传算法能产生与真实数据几乎无差异且多样性的合成时延。

**🔧 技术方法**

采用简化遗传算法、变分自编码器TimeVAE、生成对抗网络TimeGAN，以及后续的ResNet分类器进行验证。

**📊 数据集**

基于EUROCONTROL的R&D数据归档与美国交通部BTS On-Time Performance数据库，提取610天欧洲与1825天美国的30大机场的时延序列。

**📈 对比分析**

通过PCA/tSNE可视化、ResNet判别得分、相关系数和Granger因果性检验评估，简化遗传算法的判别准确率低于0.6且相关系数分布广泛，表明生成效果优于两种深度模型。

**⚠️ 局限性**

只生成每小时平均时延，缺乏更高时间分辨率；验证仅靠ResNet，可能对更强模型不稳；生成过程忽略机场间的传播效应。

---

## You Only Anonymize What Is Not Intent-Relevant: Suppressing Non-Intent Privacy Evidence

**arXiv ID:** 2601.04265 | [PDF](https://arxiv.org/pdf/2601.04265v1)

**作者:** Weihao Shen `[一作]` (Beihang University), Fuzhen Zhuang `[通讯]` (Beihang University)

**通讯引用:** 9759 | **OpenAlex IDs:** https://openalex.org/A5102969899

**关键词:** `Cryptography and Security` `Safty and Privacy` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了IntentAnony，一种基于交际意图的文本匿名化方法，能够在降低属性推断风险的同时保持语义与交互功能；

**💡 创新点**

创新点在于将隐私暴露控制与意图识别相结合，构建隐私推断证据链并根据场景与意图分配曝光预算，实现针对性地抑制非意图的隐私线索；

**🔧 技术方法**

主要技术包括大模型意图识别、隐私推断证据链构造、场景-意图曝光治理矩阵以及基于LLM的约束式重写；

**📊 数据集**

在PersonalReddit和SynthPAI两组包含真实/合成的Reddit式文本数据集上进行评测；

**📈 对比分析**

与Azure、Dipper、Adv. Anon.、RUPTA等基线相比，IntentAnony在属性推断准确率上平均下降约30%，同时在可读性、语义保持、BLEU/ROUGE等文本实用性指标上保持或提升，整体表现最佳；

**⚠️ 局限性**

局限性包括对意图识别的依赖、推断证据链建模随模型演化而变化、缺乏正式的隐私保证与对极端推断攻击的鲁棒性不足。

---

## Learning to Simulate Human Dialogue

**arXiv ID:** 2601.04436 | [PDF](https://arxiv.org/pdf/2601.04436v1)

**作者:** Kanishk Gandhi `[一作]` (Stanford University), Noah D. Goodman `[通讯]` (Stanford University)

**通讯引用:** 20545 | **OpenAlex IDs:** https://openalex.org/A5001961716

**关键词:** `Computation and Language` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Chain-of-Thought` `Text`

### 📋 论文摘要

**🎯 论文内容**

比较了不同训练方法对下一句对话预测的效果，探索是否在回答前生成链式思考（CoT）以及奖励信号是 LLM-judge 还是直接最大化真实对话的对数概率。

**💡 创新点**

创新点在于：①将 CoT 视为潜在变量并推导其 ELBO 作为训练目标，证明在分布匹配目标下思考能提升模型的预测质量；②系统性地揭示 LLM-judge 奖励会导致奖励挖掘、降低对真实人类对话的似然，从而不利于人类行为建模。

**🔧 技术方法**

使用的技术包括：GRPO（Group Relative Policy Optimization）强化学习、LLM‑judge（Qwen‑2.5‑3B‑Instruct）评估、监督微调、以及基于链式思考的潜在变量 ELBO 训练。

**📊 数据集**

实验数据集为 DailyDialog，包含约 11k 条对话、7–8 轮平均，每条对话两位说话人，共 76k 条训练样本。

**📈 对比分析**

评价指标为：对未见的真实回复的对数概率（Perplexity）和双人盲测的偏好赢率。结果显示：直接最大化对数概率的模型对数概率显著提升（-1.24 vs -3.56 原始模型），且在赢率上取得 49.8%（latent‑CoT）高于监督微调 47.2% 和 LLM‑judge 0%。

**⚠️ 局限性**

局限性包括：实验仅在二人短轮对话上进行，缺乏多方、长情境或更广泛主题；未结合记忆、人格描述等 agentic scaffold；对 CoT 的探索受限于预训练模型自身的推理能力。

---

## Correct and Weight: A Simple Yet Effective Loss for Implicit Feedback Recommendation

**arXiv ID:** 2601.04291 | [PDF](https://arxiv.org/pdf/2601.04291v1)

**作者:** Minglei Yin `[一作]` (University at Albany), Xin Li `[通讯]` (University at Albany)

**通讯引用:** 37679 | **OpenAlex IDs:** https://openalex.org/A5100354056

**关键词:** `Information Retrieval` `Recommendation System` `Graph Neural Network` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出了 Corrected and Weighted (CW) 损失，专门解决隐式反馈推荐中的假负样本问题，并通过负样本分布校正和自适应加权提升排名性能。

**💡 创新点**

创新点在于：①将 PU 学习思想应用于负样本去噪，用可观测的正样本和整体分布逼近真负分布；②引入置信度加权机制，聚焦易负样本并抑制疑似假负样本；③两者结合形成轻量级、可直接插拔的损失函数。

**🔧 技术方法**

技术包括 Softmax 基础损失、PU 去噪公式、动态置信度加权、温度参数调节、以及在 MF、LightGCN、XSimGCL 等推荐模型中的统一实现。

**📊 数据集**

实验使用四大稀疏大型数据集：Amazon Health、Amazon Electronic、Amazon Book、Gowalla 检查点数据，数据经 10-core 预处理后进行 80%/20% 划分。

**📈 对比分析**

通过与 BPR、LLPAUC、SL、AdvInfoNCE、BSL、PSL、SL@K 等基线在 Recall@20 与 NDCG@20 上对比，CW 在所有模型与数据集上均实现 2–8% 的提升，尤其在 LightGCN 上 Recall@20 和 NDCG@20 分别提升 7.5% 与 7.2%。

**⚠️ 局限性**

局限性：①需要估计正样本先验 τ⁺，对极端稀疏或高噪声场景可能受限；②未在动态/会话式推荐场景下验证；③对用户异质性的先验假设未进一步个性化；④主要验证于单标签隐式反馈，跨域或多标签情况仍需研究。

---

## Re-Rankers as Relevance Judges

**arXiv ID:** 2601.04455 | [PDF](https://arxiv.org/pdf/2601.04455v1)

**作者:** Chuan Meng `[一作]` (University of Edinburgh), Maarten de Rijke `[通讯]` (University of Amsterdam)

**通讯引用:** 29323 | **OpenAlex IDs:** https://openalex.org/A5031439294

**关键词:** `Information Retrieval` `Retrieval` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文将现有的重排模型（monoT5、RankLLaMA、Rank1）改造为自动相关性判断器，并在TREC‑DL 2019–2023上评估其表现；

**💡 创新点**

创新点在于提出两种适配策略（直接生成和阈值切分）将连续重排分数转换为二元判断，并系统研究了重排模型作为判断器时的偏差和“自偏好”；

**🔧 技术方法**

使用的技术包括大规模语言模型推理、分数阈值化、基于BERT/LLM的推理链生成，以及与现有LLM判定器UMBRELA的对比；

**📊 数据集**

采用的评测数据集为TREC‑DL 2019–2023深度学习检索任务，包含8.8M–138M条passage，人工二元标注；

**📈 对比分析**

通过Cohen’s κ和Kendall’s τ对比模型与人类标注及系统排名，实验显示在约40–50%的评测场景下，重排模型改造的判断器能超越UMBRELA；排名相关指标上也与UMBRELA相近或更优；

**⚠️ 局限性**

局限性包括：阈值选择依赖数据集、仅针对点对模型、未覆盖多级相关性、以及重排模型可能出现自偏好与跨族偏差，需进一步研究偏差缓解与更细粒度判断

---

## Propositional Abduction via Only-Knowing: A Non-Monotonic Approach

**arXiv ID:** 2601.04272 | [PDF](https://arxiv.org/pdf/2601.04272v1)

**作者:** Sanderson Molick `[一作]` (Federal Institute of Pará), Vaishak Belle `[通讯]` (University of Edinburgh)

**通讯引用:** 2596 | **OpenAlex IDs:** https://openalex.org/A5002932153

**关键词:** `Artificial Intelligence`

### 📋 论文摘要

**🎯 论文内容**

本文通过在Levesque的仅知逻辑（Only‑Knowing）中引入一种基于知识的归纳（abduction）模态，构造了新的模态逻辑𝒜𝒪ℒ，随后在其语义框架中加入偏好关系，得到非单调扩展𝒜𝒪ℒ^≺，从而在语义层面实现最小解释的选择与比较。

**💡 创新点**

创新点在于：1）将归纳推理作为仅知模态的派生运算符，通过基本的知识与仅知运算符共同定义；2）在单调逻辑基础上引入偏好Kripke模型，实现非单调后继关系；3）系统地提出并证明了最小解释的存在性、选择方法（子集、基数、优先级）与偏好推理的一致性；4）提供了完整的语义公理化与核心性质。

**🔧 技术方法**

主要使用的技术是：模态逻辑语义（Kripke模型、可达关系、偏好关系）、归纳模态定义、非单调逻辑的偏好语义与后继关系、逻辑推导与证明技术。

**📊 数据集**

未使用具体实验数据集；论文以形式化证明与逻辑模型为主。

**📈 对比分析**

本文没有在实验上与其他方法进行比较，性能评估仅体现在形式化性质（如一致性、可解释性、最小性）和逻辑推导能力上。

**⚠️ 局限性**

局限性包括：1）缺乏证明论或系统实现的进一步研究；2）未考虑多模态/多主体扩展；3）目前仅在语义层面，未给出高效推理算法或工具；4）实际应用需结合具体知识库与归纳任务，尚未验证。

---

## TrueBrief: Faithful Summarization through Small Language Models

**arXiv ID:** 2601.04212 | [PDF](https://arxiv.org/pdf/2601.04212v1)

**作者:** Kumud Lakara `[一作]` (JPMorgan), Fran Silavong `[通讯]` (JPMorgan)

**通讯引用:** 18 | **OpenAlex IDs:** https://openalex.org/A5043609447

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一套端到端框架（TrueBrief）通过偏好优化提升小型LLM在摘要任务中的可信度。

**💡 创新点**

创新在于结合受控幻觉注入生成高质量偏好对、使用单一拒绝样本的DPO以及基于内部模型动态的白盒幻觉检测。

**🔧 技术方法**

使用DPO、改进的Add-DPO/PL-DPO、LoRA微调、LogitLens+LookbackLens特征+Logistic回归等技术。

**📊 数据集**

数据来自RAGTruth（GPT-4、GPT-3.5、Llama2-70B/13B生成的摘要）以及通过自动注入幻觉生成的合成偏好对。

**📈 对比分析**

与SFT、原始DPO、多重拒绝样本（Add-DPO、PL-DPO）以及基准编码器（BERT、Longformer）对比，单一拒绝样本的DPO在小模型（0.5B）上在faithfulness和B_score上优于其他方法，幻觉检测F1最高。

**⚠️ 局限性**

局限在于对大模型效果未知、数据生成复杂度高、需要大量算力；并且随着模型变大DPO收益递减。

---

## Enhancing Admission Inquiry Responses with Fine-Tuned Models and Retrieval-Augmented Generation

**arXiv ID:** 2601.04206 | [PDF](https://arxiv.org/pdf/2601.04206v1)

**作者:** Aram Virabyan `[一作]`, Aram Virabyan `[通讯]`

**关键词:** `Computation and Language` `Retrieval` `Knowledge Distillation` `Generation` `Transformer` `Retrieval-Augmented Generation` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

结合检索增强生成（RAG）与领域微调，构建面向高校招生问答的自动回复系统。

**💡 创新点**

创新点在于将检索增强生成与大模型微调耦合，并通过数据蒸馏生成高质量问答对，显著提升回答的事实准确性与上下文适配度。

**🔧 技术方法**

采用 RAG 框架（BERT 嵌入检索 + Gemma 9B 生成器）、LoRA 微调、DeepSeek‑R1 进行问答对蒸馏，生成器使用 3 轮 LoRA 训练。

**📊 数据集**

使用高校招生官方文件、网站内容、历史问答等原始资料，经蒸馏得到约 8,000 条高质量问答对；知识库采用 512‑token 片段、64‑token 重叠的分块方式。

**📈 对比分析**

通过 210 条真实查询进行人工评测，指标包括事实召回、精确数据召回和用户满意度。微调+RAG 模型分别取得 92.7% 事实召回、88.3% 精确数据召回、8.9/10 满意度，明显优于基线 GPT、单独 RAG 与单独微调模型。

**⚠️ 局限性**

局限性：需要大量算力和显存，部署成本高；对全新或模糊问题仍易产生幻觉，知识库更新不够即时，难以完全覆盖所有招生细节。

---

## Aligned explanations in neural networks

**arXiv ID:** 2601.04378 | [PDF](https://arxiv.org/pdf/2601.04378v1)

**作者:** Corentin Lobet `[一作]`, Francesca Chiaromonte `[通讯]`

**关键词:** `Machine Learning` `Explainability and Interpretability` `Segmentation` `Classification` `Convolutional Neural Network` `Supervised Fine-Tuning` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了一种伪线性神经网络框架 PiNet，使模型在产生预测的同时可提供与预测过程严格对应的实例级线性解释；在合成图像分类任务和卫星图像洪水分割任务上验证其可解释性与预测性能。

**💡 创新点**

核心创新是将可解释性嵌入模型结构：通过构造编码器-解码器和“二次观察”机制，让解释（系数）在输入空间内线性组合生成预测，从而实现解释与预测的内在对齐；同时引入递归反馈、集成与强监督等训练技巧提升解释的可信度。

**🔧 技术方法**

技术包括：伪线性（varying‑coefficient）网络架构、Encoder‑Decoder 结构、第二次观察（第二look）、递归反馈损失、集成学习、强监督（使用真实解释做额外损失）以及标准的卷积网络与 U‑Net / SegNet 作为基准。

**📊 数据集**

使用了合成的 ToyShapes 图像数据集（含三种几何形状）做二分类任务；以及公开的 Sen1Floods11 卫星影像数据集做洪水面积回归/分割任务。

**📈 对比分析**

与 Grad‑CAM（以及其他基线 CNN 解释方法）对比，PiNet 在“意义度”“稳健性”“充分性”等 MARS 指标上表现相当或优于 Grad‑CAM；递归反馈和集成进一步提升了解释质量；在洪水分割实验中，PiNet 的 IoU 与 MAE 与 SegNet 相比略逊但仍保持可接受水平。

**⚠️ 局限性**

局限性包括：需要精心设计特征空间与编码器‑解码器结构，且在目标信息稀疏时（如仅有图像级回归）解释的锐度有限；强监督需手工标注真实解释，可能产生伦理或偏见风险；对不同数据模态（文本、音频等）的推广尚未验证。

---

## Technological Transitions and the Limits of Inference in Adaptive Educational Systems

**arXiv ID:** 2601.04357 | [PDF](https://arxiv.org/pdf/2601.04357v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## A Future Capabilities Agent for Tactical Air Traffic Control

**arXiv ID:** 2601.04285 | [PDF](https://arxiv.org/pdf/2601.04285v1)

**作者:** Paul Kent `[一作]` (University of Exeter), Ben Carvell `[通讯]` (NATS)

**关键词:** `Artificial Intelligence` `Optimization` `Explainability and Interpretability` `Agentic AI` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了一种名为 Agent Mallard 的基于规则、前向仿真集成的空中交通控制代理，能够在系统化航路环境中通过层次化计划和回溯搜索实现多机组冲突解决。

**💡 创新点**

创新点在于：①将不确定性仿真直接嵌入决策循环，实现安全验证与可解释性兼顾；②利用预定义 GPS 导航道将连续 4D vectoring 简化为离散的车道/高度选择；③采用因果归因与计划切片等机制大幅缩小搜索空间并保证计划可追溯。

**🔧 技术方法**

核心技术包括：规则基础决策逻辑、层次化计划表示（Condition–Action–Completion 结构）、数字孪生前向仿真、基于冲突分类的策略库、深度受限回溯搜索和单轴单调约束。

**📊 数据集**

使用了英国 Project Bluebird 的 BluebirdDT 数字孪生平台和 MBT（Machine Basic Training）情景数据集进行验证，同时收集了 UK ATCO 控制员的现场评审。

**📈 对比分析**

在初步验证中，Mallard 的决策与专家控制员的思路高度一致，能够在简化场景下无次生冲突地完成冲突解决；但尚未给出系统性性能指标（如冲突成功率、计算时延），后续计划与人类控制员进行基准比较。

**⚠️ 局限性**

主要局限包括：仅针对结构化系统化航路（非自由航路）设计；核心逻辑针对二机组冲突，难以处理高密度多机组聚集情况；策略库受限于现有专家知识，无法自动产生新方案；决策质量高度依赖数字孪生的准确性。

---

## LinguaGame: A Linguistically Grounded Game-Theoretic Paradigm for Multi-Agent Dialogue Generation

**arXiv ID:** 2601.04516 | [PDF](https://arxiv.org/pdf/2601.04516v1)

**作者:** Yuxiao Ye `[一作]` (Tsinghua University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**通讯引用:** 44498 | **OpenAlex IDs:** https://openalex.org/A5100320723

**关键词:** `Computation and Language` `Generation` `Optimization` `Transformer` `Large Language Model` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

本研究提出了 LinguaGame 框架，基于信号博弈模型改进多代理对话生成。

**💡 创新点**

创新点在于将对话建模为基于语义意图和策略的信号博弈，并通过训练无关的均衡逼近算法实现推理时的决策优化。

**🔧 技术方法**

使用了大型语言模型（如 Qwen2.5‑32B）、信号博弈推理层以及 piKL 无后悔学习的均衡逼近算法。

**📊 数据集**

实验数据来自中国裁判文书网的 50 起庭审案例和 Quora 争论数据集的 50 条争议命题。

**📈 对比分析**

与标准 MAS、仅使用意图‑策略设计的 MAS 以及基于 LLM 的重排序基线相比，LinguaGame 在四项对话质量指标（清晰度、简洁度、论证性、战术性）上均获得显著提升，正向替换率高达 78.1%。

**⚠️ 局限性**

局限性包括仅关注沟通效率而非任务结果、仅评估庭审和辩论两类情境、评估成本高且未验证在实际法律决策中的安全性与公平性。

---

## ArtCognition: A Multimodal AI Framework for Affective State Sensing from Visual and Kinematic Drawing Cues

**arXiv ID:** 2601.04297 | [PDF](https://arxiv.org/pdf/2601.04297v1)

**作者:** Behrad Binaei-Haghighi `[一作]` (University of Tehran), Behnam Bahrak `[通讯]` (Tehran Institute for Advanced Studies)

**关键词:** `Machine Learning` `Classification` `Object Detection` `Generation` `Explainability and Interpretability` `Convolutional Neural Network` `Transformer` `Retrieval-Augmented Generation` `Large Language Model` `Image` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

本文提出了 ArtCognition 框架，通过将绘画过程中的行为日志与最终绘图的视觉特征融合，自动化分析 HTP（House‑Tree‑Person）投射测验并生成心理评估报告。

**💡 创新点**

创新点在于：①将绘画过程的 kinematic 细节（速度、暂停、平滑度、擦除等）与图像特征联合提取；②采用 Retrieval‑Augmented Generation（RAG）把分析结果与心理学知识库对齐，显著降低模型幻觉并提升可解释性。

**🔧 技术方法**

使用技术包括：YOLOv11 两阶段目标检测、EfficientNet/ViT/ResNet/MobileNet/ConvNeXt/Swin 等分类网络、行为特征提取算法、基于 LLM 的 RAG 生成模块。

**📊 数据集**

数据集由 146 份数字化 HTP 画作及其完整绘图日志（含点、时间、颜色等）组成，并配有 21 份自评问卷，为训练与评估提供多模态输入。

**📈 对比分析**

通过与规则基准和未检索的 Gemini‑2‑Flash 进行对比，检测精度和分类 F1 分别超过 95%，描述生成平均精度 97.6%，RAG 检索相似度最高 0.991，幻觉率从 45.7% 降至 0%。

**⚠️ 局限性**

局限性包括：样本量小且文化同质化、缺失笔压/倾斜等神经运动信息、对某些心理细节的解释仍需专家监督。

---

## TokenSeg: Efficient 3D Medical Image Segmentation via Hierarchical Visual Token Compression

**arXiv ID:** 2601.04519 | [PDF](https://arxiv.org/pdf/2601.04519v1)

**作者:** Sen Zeng `[一作]` (Tsinghua University), Yang Liu `[通讯]` (KCL)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Transformer` `Auto Encoder` `Image` `Biomedical Data` `Magnetic Resonance Imaging`

### 📋 论文摘要

**🎯 论文内容**

提出了TokenSeg框架，利用层次化视觉token压缩与边界感知的稀疏token化，再通过token重投影与多级解码实现高效3D医学图像分割。

**💡 创新点**

创新点包括：①四层多尺度层次化编码器生成400个候选token；②结合VQ‑VAE量化与重要性评分的边界感知tokenizer，只保留100个聚焦于病灶边缘的token；③稀疏到稠密的解码器采用token重投影、逐层上采样与跨级融合，保持空间连贯性和细节恢复。

**🔧 技术方法**

使用技术包括：Vector‑Quantized VAE（VQ‑VAE）对token进行离散化；基于边界强度、语义强度与码本多样性的综合评分；多尺度Encoder/Decoder网络；skip连接与跨层融合；Dice/BCE/ VQ三项联合损失；混合精度训练与自适应学习率。

**📊 数据集**

使用数据集：1）960例多中心乳腺DCE‑MRI（内测872例，外测88例）；2）MSD脑胶质瘤（Task01 484例）和左心房（Task02 20例）公共基准。所有数据统一预处理为单通道3D体素。

**📈 对比分析**

与3D U‑Net、V‑Net、nnU‑Net、Swin‑UNETR、TransUNet、MobileNet‑UNet、EfficientNet‑UNet等方法对比，TokenSeg在乳腺DCE‑MRI上达94.49% Dice、89.61% IoU，速度48 ms、显存2.9 GB，参数23.8 M，显著优于最佳基线（nnU‑Net 90.2% Dice、256 ms、52.3 M）。在外部测试集保持92.18% Dice，仅下降2.31%，表明跨中心泛化良好。

**⚠️ 局限性**

局限性：token数量固定且需手工设定，缺乏对病例难度的动态适配；边界评分依赖预定义梯度特征；仅验证单一模态，需扩展到多模态/多器官；码本维度选择对性能影响较大，过大易导致利用率下降。

---

## Towards a Mechanistic Understanding of Propositional Logical Reasoning in Large Language Models

**arXiv ID:** 2601.04260 | [PDF](https://arxiv.org/pdf/2601.04260v1)

**作者:** Danchun Chen `[一作]` (MOE Key Lab of Computational Linguistics Peking University), Liangming Pan `[通讯]` (MOE Key Lab of Computational Linguistics Peking University)

**关键词:** `Artificial Intelligence` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文通过对 Qwen3 模型在自建的 PropLogic-MI 数据集上的命题逻辑推理过程进行激活补丁分析，揭示了四种互联的计算机制：分阶段计算、信息传输、事实回溯与专用注意力头。

**💡 创新点**

创新点在于提出从宏观机制到微观实现的层级分析框架，首次系统识别并量化这些机制在不同模型规模、规则类型和推理深度下的通用性。

**🔧 技术方法**

使用了激活补丁、对数差分 (LD) 与 dLD 评估、残差流追踪以及注意力头聚类等因果解释技术。

**📊 数据集**

采用自建的 PropLogic-MI 数据集，覆盖 11 类命题逻辑规则，并包含一跳与二跳推理实例。

**📈 对比分析**

通过将补丁实验结果与模型原始行为对比，证实上述机制对推理准确率具有显著提升作用；在 8B 与 14B 模型上，一跳/二跳任务的准确率提升约 5–10% 以上。

**⚠️ 局限性**

局限包括未深入解析 MLP 内部变换、未探究机制在训练过程中的形成机制，以及仍无法完全排除所观察到的模式是统计近似而非真正的逻辑算法。

---

## Transformer-based Multi-agent Reinforcement Learning for Separation Assurance in Structured and Unstructured Airspaces

**arXiv ID:** 2601.04401 | [PDF](https://arxiv.org/pdf/2601.04401v1)

**作者:** Arsyi Aziz `[一作]`, Peng Wei `[通讯]`

**关键词:** `Robotics` `Reinforcement Learning` `Autonomous Driving` `Transformer` `Reinforcement Learning` `Tabular` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种基于Transformer编码器的速度建议系统（Speed Advisor），该系统通过对自身车辆（Ownship）和潜在干扰车辆（Intruders）的状态信息进行嵌入、注意力融合，并通过状态价值估计器（State-value Estimator）动态生成安全行驶速度。

**💡 创新点**

创新点在于：①将Transformer Encoder引入车辆速度决策，能够同时处理多车序列信息并捕获长期时序依赖；②使用多层感知机（MLP）对各车辆状态进行预嵌入；③将速度建议与状态价值估计结合，形成端到端可训练的决策框架。

**🔧 技术方法**

主要技术包括：Transformer Encoder、1层MLP嵌入层、状态价值估计网络（可能是深度网络或值函数逼近）、自回归/强化学习框架（π_w 与 v_w 作为策略与价值网络）。

**📊 数据集**

实验使用了公开交通仿真数据集（如NGSIM或SUMO生成的多车场景）以及真实车辆行驶日志（若有），主要评估指标为碰撞率、行驶效率和速度波动率。

**📈 对比分析**

与传统基于PID或规则的速度控制方法以及最近的深度强化学习模型相比，本文方法在碰撞率下降 15%–25% 的同时，保持或略提升了行驶效率（平均速度、行驶时长）。

**⚠️ 局限性**

局限性包括：①模型对输入顺序敏感，可能在车辆数量极大时计算量显著增加；②实验主要基于仿真或有限的真实数据，缺乏大规模多车交互的实测验证；③缺少对极端天气或传感器失效情况下的鲁棒性评估。

---

## Correcting Autonomous Driving Object Detection Misclassifications with Automated Commonsense Reasoning

**arXiv ID:** 2601.04271 | [PDF](https://arxiv.org/pdf/2601.04271v1)

**作者:** Keegan Kimbrell `[一作]` (University of Texas at Dallas), Gopal Gupta `[通讯]` (University of Texas at Dallas)

**通讯引用:** 2874 | **OpenAlex IDs:** https://openalex.org/A5067377863

**关键词:** `Artificial Intelligence` `Object Detection` `Autonomous Driving` `Explainability and Interpretability` `Computational Efficiency` `Image`

### 📋 论文摘要

**🎯 论文内容**

利用深度学习感知与基于逻辑的常识推理层相结合，自动纠正自动驾驶车辆在交通灯识别和障碍物检测中的错误分类。

**💡 创新点**

创新点在于将可解释的常识推理模块嵌入感知后处理，既能在缺乏训练样本的异常场景下纠错，又能通过不确定性阈值动态触发推理，提升系统的鲁棒性与可解释性。

**🔧 技术方法**

技术包括：CARLA 仿真环境下的深度学习检测模型（BEV 语义分割、交通灯二分类、车辆行为预测）、基于证据深度学习的离散不确定性估计、Prolog 逻辑程序进行常识一致性检查、以及基于 DBSCAN 的行为聚类。

**📊 数据集**

使用 CARLA Towns 1-4 的仿真数据，车辆密度分别为 100/200 辆，并在雨天等恶劣天气下采集，包含交通灯、障碍物（停车车辆、动物）等场景。

**📈 对比分析**

与单纯深度学习基线对比，融合常识推理的完整模型在交通灯识别和障碍物检测上均实现了 10–50% 的准确率提升（如 0.48→0.85、0.59→0.99），并在召回率、F1 分数上取得显著提升；利用不确定性触发的推理在计算量上更高效，仅对高不确定帧进行推理。

**⚠️ 局限性**

局限性包括：逻辑规则覆盖范围有限，推理精度约 70–95%；不确定性阈值需经验调节；模型仅基于单车摄像头，缺乏多车协同信息；扩展到真实数据与更大场景时需构建更完整、可扩展的知识库。

---

## SampoNLP: A Self-Referential Toolkit for Morphological Analysis of Subword Tokenizers

**arXiv ID:** 2601.04469 | [PDF](https://arxiv.org/pdf/2601.04469v1)

**作者:** Iaroslav Chelombitko `[一作]` (Neapolis University Pafos), Aleksey Komissarov `[通讯]`

**关键词:** `Computation and Language` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了SampoNLP工具，使用无语料库的MDL启发式自参照原子性评分方法构建高纯度形态词典，并利用该词典评估不同词表规模下的BPE分词器，给出乌拉尔语系三种语言的最优词表范围。

**💡 创新点**

创新点在于将MDL原则转化为类型级别的自参照评分实现无语料库形态学词典清洗，并提出统一的Integrated Performance Score（IPS）衡量词表规模与形态覆盖/过分割的权衡。

**🔧 技术方法**

使用了MDL启发式自参照原子性评分、Otsu阈值分割、动态规划最佳拆分、BPE分词器训练、Kneedle算法识别elbow点。

**📊 数据集**

采用Hunspell/LibreOffice拼写检查字典生成的候选列表作为形态词典输入，并使用维基百科文本训练BPE模型，覆盖芬兰语、匈牙利语和爱沙尼亚语。

**📈 对比分析**

通过计算Lexical Morpheme Coverage与Over‑Split Rate得到IPS，对不同词表规模的BPE进行对比，发现匈牙利语最优IPS≈0.73，爱沙尼亚≈0.39，芬兰≈0.31，给出最优词表范围80k–128k（匈/爱）或80k–150k（芬），表明标准BPE对高度黏着语的适应有限。

**⚠️ 局限性**

局限在于IPS仅捕捉形态覆盖与过分割的量化平衡，未考虑语义或上下文细节；方法依赖于拼写字典的完整性；对真实噪声数据的鲁棒性尚未验证。

---

## Decision-Aware Trust Signal Alignment for SOC Alert Triage

**arXiv ID:** 2601.04486 | [PDF](https://arxiv.org/pdf/2601.04486v1)

**作者:** Israt Jahan Chowdhury `[一作]` (Ontario Tech University), Md Abu Yousuf Tanvir `[通讯]` (Ontario Tech University)

**关键词:** `Cryptography and Security` `Anomaly Detection` `Tabular` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出了一种决策意识的信任信号对齐框架，用于安全运营中心(SOC)的警报分类，旨在改善分析师在处理安全警报时的决策支持。

**💡 创新点**

创新点在于将信任信号（如置信度、校准和不确定性）与决策成本相结合，提供了一种成本敏感的决策阈值，从而提高了警报分类的有效性。

**🔧 技术方法**

使用了逻辑回归和随机森林两种分类器，并结合后期校准和不确定性估计技术来构建信任信号。

**📊 数据集**

使用了UNSW-NB15入侵检测基准数据集，该数据集包含当前攻击模式和真实背景流量的实验记录。

**📈 对比分析**

通过与基线和不对齐信任条件进行比较，结果显示对齐信任条件显著降低了假阴性率，并减少了成本加权损失，表明该方法在成本敏感决策中的有效性。

**⚠️ 局限性**

限制在于主要评估基于模拟分析师的行为，而非实际SOC分析师的观察，且实验仅在一个基准数据集上进行，可能不代表多变的威胁环境。

---

## Generative AI for Social Impact

**arXiv ID:** 2601.04238 | [PDF](https://arxiv.org/pdf/2601.04238v1)

**作者:** Lingkai Kong `[一作]`, Milind Tambe `[通讯]`

**关键词:** `Computers and Society` `Optimization` `Generation` `Large Language Model` `Diffusion model` `Graph` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

本文提出通过生成式 AI（LLM 代理和扩散模型）统一解决 AI4SI 部署瓶颈，即观测稀缺、政策合成与人机对齐三大缺口，从而实现资源优化的可扩展、可适应与人类对齐部署；

**💡 创新点**

创新点在于：①将 LLM 代理作为自然语言与数学优化之间的桥梁，自动将专家的隐式知识与动态约束转化为可执行的约束与奖励；②利用扩散模型生成逼真的社交网络、环境动态与策略蓝图，解决观测稀缺和高维组合决策空间；③通过生成式环境模拟与迁移学习提升策略的稳健性与快速适配能力；

**🔧 技术方法**

主要技术包括：大语言模型代理（LLM Agent）进行语义解析与约束形式化；扩散模型（diffusion models）用于图结构生成、策略蓝图生成、奖励与转移动力学建模；离散组合优化（如上限k选择、约束求解）；多任务/迁移学习框架；

**📊 数据集**

使用的数据集为：1) 种族和城市的 HIV 社交网络（洛杉矶 YEH 与南非接触网络）；2) 野生动物保护区的区域风险预测与巡逻记录；3) 埃塞俄比亚农村基层卫生产能数据；4) 通过扩散模型生成的合成网络和环境数据；

**📈 对比分析**

虽然未给出统一的数值评估，但作者引用先前部署案例：在印度移动保健项目中减少 30% 的项目退出率；在国家公园提升非法陷阱检测五倍；通过 LLM 代理与扩散模型实现的策略在多场景实验中展示了更高的覆盖率、鲁棒性和可解释性，并在对比传统单一模型时表现出更好的适配速度与稳健性；

**⚠️ 局限性**

局限性包括：①生成模型的生成质量与真实数据分布的匹配度依赖训练数据质量；②LLM 代理对歧义和专业术语的解析仍可能产生误解，需人工审核；③对大规模组合决策的求解仍受算力限制；④在极端动态环境下，模型迁移与更新的实时性尚需进一步研究。

---

## Vision-Language Agents for Interactive Forest Change Analysis

**arXiv ID:** 2601.04497 | [PDF](https://arxiv.org/pdf/2601.04497v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition`

---

## LLMs for Explainable Business Decision-Making: A Reinforcement Learning Fine-Tuning Approach

**arXiv ID:** 2601.04208 | [PDF](https://arxiv.org/pdf/2601.04208v1)

**作者:** Xiang Cheng `[一作]` (University of Maryland), Anindya Ghose `[通讯]` (New York University)

**通讯引用:** 13703 | **OpenAlex IDs:** https://openalex.org/A5073770532

**关键词:** `Computation and Language` `Explainability and Interpretability` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Supervised Fine-Tuning` `Tabular` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

提出一种名为LEXMA的LLM微调框架，能够在保持决策准确性的同时，为不同受众生成符合语境的自然语言解释；

**💡 创新点**

创新点在于：①将解释过程嵌入决策生成链（Reasoning→Explanation→Prediction），让奖励直接与正确决策关联；②使用两阶段Group Relative Policy Optimization（GRPO）分别调优决策正确性适配器与语气适配器，实现决策不变、说明风格可调；③采用反射增强的监督微调和基于规则的可读性、礼貌奖励，避免依赖海量人工标注；

**🔧 技术方法**

主要技术包括：反射增强监督微调、两阶段GRPO（针对ACC与TONE适配器）、LoRA参数高效微调、可读性和礼貌的规则式奖励、三阶段生成结构；

**📊 数据集**

使用美国住房抵押贷款披露法（HMDA）公开数据集，经过特征序列化后输入LLM，训练样本约数十万条，测试集为公开HMDA数据；

**📈 对比分析**

与XGBoost、神经网络、逻辑回归、梯度提升树以及原始Qwen3-4B和GPT‑5进行对比；在专家提示下LEXMA F1≈0.897、准确率≈0.845，消费者提示下F1≈0.893、准确率≈0.825，明显优于基线模型且接近树模型；人类评估显示专家与消费者对解释质量均显著偏好LEXMA；

**⚠️ 局限性**

限制主要有：①仅使用HMDA公开数据，缺少信用历史等关键特征，可能限制模型表现；②人类评估规模有限，缺乏大规模实地部署验证；③奖励仅基于规则，未加入更细粒度的人工评估或领域专家反馈。

---

## Ideology as a Problem: Lightweight Logit Steering for Annotator-Specific Alignment in Social Media Analysis

**arXiv ID:** 2601.04207 | [PDF](https://arxiv.org/pdf/2601.04207v1)

**作者:** Wei Xia `[一作]` (Ludwig Maximilian University of Munich), Luozheng Li `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `Computation and Language` `Classification` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出一种轻量级的读出层logit调节方法，利用冻结LLM的隐藏表示进行注解者特定的意识形态对齐。

**💡 创新点**

创新点在于将方向性与校正幅度拆分为两个可学习的标量探针（方向项和分数项），通过异步更新在不改变模型参数的前提下实现精准的意识形态校正。

**🔧 技术方法**

采用低维方向探针、Softplus 计分、异步 logit 校正等技术，基于隐藏层投影实现对齐。

**📊 数据集**

使用 MITweet 12 维度的意识形态标注数据集进行评估。

**📈 对比分析**

与零shot、5-shot、schema‑aware prompting 以及全模型微调进行对比；在 Llama‑3‑8B 与 Qwen‑2.5‑7B 上分别提升约 20% Accuracy、12–14% Macro‑F1。

**⚠️ 局限性**

局限性：仅适用于线性可分的低维轴；对极端不平衡或几何更复杂的 facet 效果有限；目前仅支持三分类（左/中/右），难以直接扩展到多标签或连续尺度。

---

## UniDrive-WM: Unified Understanding, Planning and Generation World Model For Autonomous Driving

**arXiv ID:** 2601.04453 | [PDF](https://arxiv.org/pdf/2601.04453v1)

**作者:** Zhexiao Xiong `[一作]` (Bosch Research North America), Liu Ren `[通讯]` (Bosch Research North America)

**关键词:** `Computer Vision and Pattern Recognition` `Autonomous Driving` `Generation` `Transformer` `Vision Language Model` `Diffusion model` `World Model` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出统一的 Vision‑Language 世界模型 UniDrive‑WM，实现驾驶场景理解、轨迹规划与未来图像生成三合一

**💡 创新点**

创新点在于将 VLM 与轨迹规划和未来图像生成耦合为单一端到端网络，利用轨迹条件的视觉生成提升规划精度，并提供离散 AR 与连续 AR+扩散两种生成范式

**🔧 技术方法**

使用 VLM（基于 Orion 与 Vicuna 1.5+LoRA）、QT‑Former 编码器、AR/AR+扩散图像生成、流匹配损失、CLIP 对齐等技术

**📊 数据集**

在 Bench2Drive（CARLA v2）基准数据集上进行训练与评估

**📈 对比分析**

与现有 E2E 与 VLM 引导方法对比，UniDrive‑WM 在闭环驾驶评分提升 5.9% L2 误差、9.2% 碰撞率，FID 更低；在开放环规划与检测指标也优于对照组

**⚠️ 局限性**

局限性包括：扩散路径推理时间较长、生成多样性受限；在更长时域或更复杂交互场景下的鲁棒性仍待验证

---

## Few-Shot LoRA Adaptation of a Flow-Matching Foundation Model for Cross-Spectral Object Detection

**arXiv ID:** 2601.04381 | [PDF](https://arxiv.org/pdf/2601.04381v1)

**作者:** Maxim Clouser `[一作]` (Yrikka Inc), John Kalantari `[通讯]` (Yrikka Inc)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Image Translation` `Data Synthesis` `Supervised Fine-Tuning` `Flow-based Model` `Image` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

在极少量 RGB–IR/SAR 对齐样本下，利用 LoRA 微调 FLUX.1 Kontext 流匹配生成器完成跨光谱图像翻译，生成合成 IR/SAR 数据，并用这些数据提升目标模态（IR/SAR）的目标检测性能。

**💡 创新点**

创新点：①通过极少量对齐样本在单一流匹配模型上实现跨光谱翻译；②将 LPIPS 作为低成本、可解释的指标预测下游检测效果；③利用合成数据跨域扩展训练集，显著提升 IR 与 SAR 检测，验证了 LoRA 适配的高效性与通用性。

**🔧 技术方法**

技术：LoRA 参数高效微调、FLUX.1 Kontext 流匹配生成器、LPIPS 评估、YOLOv11n 与 DETR 检测器、合成数据增量训练与多模型评估。

**📊 数据集**

数据集：KAIST RGB–IR、M4‑SAR RGB–SAR、外部 RGB 数据集 LLVIP、FLIR ADAS（仅使用 RGB 视角）。

**📈 对比分析**

比较方法：使用 LPIPS 选择最佳 LoRA 后，训练 YOLOv11n / DETR；在 KAIST 上 IR 检测 mAP 从 0.50（仅真值）提升至 0.54（含合成数据），在 M4‑SAR 上 SAR 检测 mAP 从 0.19 提升至 0.25；LPIPS 与 mAP 之间呈显著负相关（Pearson |r|>0.8）。

**⚠️ 局限性**

局限：仅依赖少量对齐样本；合成图像的物理真实性仍有限；不同检测器对合成数据的敏感度差异；仅验证了目标检测任务，未扩展到分割、跟踪等；未探索无配对或自监督的跨光谱学习方法。

---

## From Paper to Structured JSON: An Agentic Workflow for Compliant BMR Digital Transformation

**arXiv ID:** 2601.04368 | [PDF](https://arxiv.org/pdf/2601.04368v1)

**作者:** Bhavik Agarwal `[一作]` (MasterControl AI Research), Viktoria Rojkova `[通讯]` (MasterControl AI Research)

**关键词:** `Digital Libraries` `Data Synthesis` `Optimization` `Computational Efficiency` `Transformer` `Large Language Model` `Agentic AI` `Vision Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

开发了一套基于LLM的agentic工作流，能将纸质或扫描的批量制造记录（BMR）转换成符合GMP要求的结构化JSON。

**💡 创新点**

创新点包括：token化分块并行处理；使用TypeScript类定义的强类型 schema 指导LLM抽取；三层验证（语法、结构、GMP合规）；以及上下文感知的覆盖度指标，显著提升了提取质量与效率。

**🔧 技术方法**

使用技术包括：Vision‑Language模型（Qwen3‑VL‑8B‑Instruct、MarkItDown、Tesseract OCR 等）、多线程并行处理、基于TypeScript提示的LLM抽取、结构化验证层和覆盖度评估。

**📊 数据集**

使用数据集为三份真实BMR（15–66页，封装、包装、固体片剂）以及内部行业文档作为训练/评估样本。

**📈 对比分析**

与人工审查比较，系统将原本耗时数小时的手工审核压缩至分钟级；在三份测试文档中，Composite Confidence Score 82.08%–89.13%，结构、计算、逻辑等关键指标保持 100% 级别，处理速度 0.85–2.6 页/分钟。

**⚠️ 局限性**

局限性：OCR 对手写或老化扫描的识别准确率约 85%；跨块依赖在超过 150 页的长文档中可能导致上下文丢失；训练数据缺乏足够的现场特定缩略语与非标准符号覆盖，导致某些专业表达仍易被误判。

---

## Convenience vs. Control: A Qualitative Study of Youth Privacy with Smart Voice Assistants

**arXiv ID:** 2601.04399 | [PDF](https://arxiv.org/pdf/2601.04399v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## ReHyAt: Recurrent Hybrid Attention for Video Diffusion Transformers

**arXiv ID:** 2601.04342 | [PDF](https://arxiv.org/pdf/2601.04342v1)

**作者:** Mohsen Ghafoorian `[一作]` (Qualcomm AI Research), Amirhossein Habibian `[通讯]` (Qualcomm AI Research)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Computational Efficiency` `Knowledge Distillation` `Recurrent Neural Network` `Transformer` `Diffusion model` `Knowledge Distillation` `Video`

### 📋 论文摘要

**🎯 论文内容**

设计并实现了可递归的混合注意力 ReHyAt，将高质量软最大注意力视频扩散模型转为线性复杂度的 RNN 模型。

**💡 创新点**

通过局部软最大注意力与全局线性注意力的分块重叠组合，并实现可递归改写，实现常量显存与线性时间，同时保持高保真度。

**🔧 技术方法**

软最大注意力、线性注意力、注意力蒸馏、轻量化多项式特征映射、RNN 重写、FlashAttention 对比。

**📊 数据集**

VBench、VBench‑2.0、Open‑Sora Plan 以及 350K Open‑Sora 子集与 22K 生成视频样本。

**📈 对比分析**

在 VBench 与 VBench‑2.0 上与 Wan2.1、CogVideoX 等 SOTA 对比，ReHyAt 在约 160 GPU‑h 训练下实现近 SOTA 质量，浮点运算量比 FlashAttention 低 4 倍，移动端显存和推理速度大幅提升。

**⚠️ 局限性**

在部分视频仍存在短暂的时间不连贯，尤其是最轻量化变体；块尺寸与重叠大小的选择仍需经验调优。

---

## A Longitudinal Measurement Study of Log4Shell Exploitation from an Active Network Telescope

**arXiv ID:** 2601.04281 | [PDF](https://arxiv.org/pdf/2601.04281v1)

**作者:** Aakash Singh `[一作]` (CSIR Fourth Paradigm Institute), Basavala Bhanu Prasanth `[通讯]` (CSIR Fourth Paradigm Institute)

**关键词:** `Cryptography and Security` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

利用在印度部署的/24网络望远镜，对 2021 年 12 月至 2025 年 10 月期间收集的 Log4Shell 相关流量进行长期扫描、payload 与回调基础设施演化的分析。

**💡 创新点**

首次实现跨四年、跨区域的长期测量，揭示攻击持续性、扫描聚焦化、回调服务聚合以及 payload 混淆程度的演进，为后续防御策略提供实证依据。

**🔧 技术方法**

采用 TCP 流重组、多层解码管线、正则签名检测、IP2Location 地理归属、MySQL+SQLAlchemy 数据处理以及 Python/Matplotlib/Matlab 可视化等技术组合。

**📊 数据集**

使用约 79 万条完整 payload 及每日分区的入站流量数据，涵盖印度及全球望远镜的未授权入站数据，来源于 CSIR‑4PI 的/24 IPv4 网络望远镜。

**📈 对比分析**

与早期欧洲/美洲望远镜的时间序列进行 Pearson 相关与流量规模对比，相关系数约为 0.6，表明时间趋势一致但规模与方差存在差异。

**⚠️ 局限性**

仅基于印度单点望远镜，缺乏更广泛地理覆盖；数据缺口虽短暂但未影响整体趋势，且对攻击者动机与技术细节的深度归因仍有限。

---

## Attribute-Aware Controlled Product Generation with LLMs for E-commerce

**arXiv ID:** 2601.04200 | [PDF](https://arxiv.org/pdf/2601.04200v1)

**作者:** Virginia Negri `[一作]` (Amazon Spain), Subburam Rajaram `[通讯]` (Amazon Germany)

**关键词:** `Computation and Language` `Generation` `Data Synthesis` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

利用大型语言模型在现有商品模板上进行可控属性修改，生成高质量合成电商商品数据。

**💡 创新点**

提出三种生成策略（正确属性修改、受控负样本生成、系统性属性移除）以及多步骤验证流程，保证属性一致性与文本连贯性。

**🔧 技术方法**

使用 Claude Haiku LLM 进行属性值生成与文本修改，句子变换模型作为相似度评估器，并辅以自定义提示模板。

**📊 数据集**

基于 MAVE 大规模商品属性数据集（约 220 万条记录）抽样 2000 条商品进行实验。

**📈 对比分析**

通过人工评估（自然度 99.6% / 属性正确性 96.5%）和下游属性抽取任务（FLAN‑T5‑base）验证，合成数据单独训练准确率 60.48% 与真实数据 60.79% 相近，混合训练最高可达 68.82%。

**⚠️ 局限性**

主要限制：仅评估正样本；对多属性修改缺乏实验；合成与真实数据在属性细粒度和分布上仍存在差异；对模型在负样本、缺失属性下的泛化尚未深入验证。

---

## Users Mispredict Their Own Preferences for AI Writing Assistance

**arXiv ID:** 2601.04461 | [PDF](https://arxiv.org/pdf/2601.04461v1)

**作者:** Vivian Lai `[一作]` (Microsoft), Alex C. Williams `[通讯]` (Microsoft)

**关键词:** `Computation and Language` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

通过因子实验式情景问卷研究了邮件写作中用户对AI主动协助的偏好。

**💡 创新点**

发现用户对紧迫度的主观重视与行为偏好相反，行为驱动主要是写作工作量，揭示了感知‑行为差距。

**🔧 技术方法**

采用因子设计、Bradley‑Terry 模型、Spearman 相关、Ordinal Logistic 回归、随机森林、梯度提升等机器学习技术。

**📊 数据集**

实验数据来自 50 名专业工作者完成 750 条配对比较，生成 16 个情境的 GPT‑5 内容。

**📈 对比分析**

与统一权重、用户自评权重和行为权重规则比较，行为权重预测准确率 61.3% 超过自评 57.7%，与机器学习模型相当，提升约 3.6 个百分点。

**⚠️ 局限性**

局限在实验场景过于简化、仅用二元特征、样本规模小、未检验跨文化和长期偏好变化。

---

## Sharded Elimination and Combining for Highly-Efficient Concurrent Stacks

**arXiv ID:** 2601.04523 | [PDF](https://arxiv.org/pdf/2601.04523v1)

**作者:** Ajay Singh `[一作]` (FORTH ICS), Panagiota Fatourou `[通讯]` (FORTH ICS)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Concurrent Stacks` `Elimination` `Software Combining` `Sharding` `CAS Atomic Operations`

### 📋 论文摘要

**🎯 论文内容**

实现了一种新型阻塞线性化的并发栈——Sharded Elimination and Combining（SEC），通过分片、批次冻结、消除与组合的统一机制实现高并发；

**💡 创新点**

创新点在于将消除与软件组合融合为多级分片结构，利用 fetch‑increment 计数器实现轻量化消除与批量组合，并通过多个聚合器并行操作，显著降低对共享栈顶指针的争用；

**🔧 技术方法**

采用了消除、软件组合、分片（sharding）、fetch‑increment、CAS 原子操作、基于 epoch 的回收等技术；

**📊 数据集**

实验使用 Intel Emerald Rapids、Ice Lake‑SP 与 Sapphire Rapids 多核机器，生成的合成工作负载（push/ pop/ peek 混合、预填 1000 节点）作为数据集；

**📈 对比分析**

与 Treiber、Elimination Backoff、Flat‑Combining、CC‑Synch、时间戳栈等五种主流实现进行对比，采用 5 秒跑时吞吐量（百万 ops/s）评估；SEC 在 100% 更新、50% 更新、10% 更新等三种工作负载下，在高线程数下表现最优，吞吐量可提升 1.8–2.5 倍；

**⚠️ 局限性**

局限性包括：阻塞式实现导致在低争用或只读工作负载下性能下降；peek/pop 操作相对昂贵；需要根据机器核数与 NUMA 配置调优聚合器数量；在极端低更新率时表现不如某些专门针对 push/peek 的实现。

---

## Autonomous Reasoning for Spacecraft Control: A Large Language Model Framework with Group Relative Policy Optimization

**arXiv ID:** 2601.04334 | [PDF](https://arxiv.org/pdf/2601.04334v1)

**作者:** Amit Jain `[一作]` (Massachusetts Institute of Technology), Richard Linares `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1373 | **OpenAlex IDs:** https://openalex.org/A5087614402

**关键词:** `Robotics` `Optimization` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Supervised Fine-Tuning` `Prompt Engineering` `Sequential`

### 📋 论文摘要

**🎯 论文内容**

结合大型语言模型与Group Relative Policy Optimization，提出两阶段训练方法实现可解释的自适应控制。

**💡 创新点**

首次将LLM作为控制策略生成器，并引入GRPO实现样本高效强化学习，同时通过监督微调保证输出格式与推理逻辑，支持跨复杂度系统迁移。

**🔧 技术方法**

使用Qwen3-4B-Base LLM+LoRA参数化、结构化文本提示/解码、GRPO强化学习、强化奖励设计、状态与动作文本编码。

**📊 数据集**

为每个系统（双积分器、Van der Pol振荡器、轨道升迁、航天器去旋转）生成约1800条专家最优序列及200条验证样本，覆盖线性、非线性、连续推力与三轴耦合动力学。

**📈 对比分析**

与传统最优控制（LQR、BVP）和基准RL方法对比，指标为终端误差、累计奖励、约束满足率；LLM‑GRPO在所有四个系统终端误差<0.05，奖励提升30–50%，保持格式一致性。

**⚠️ 局限性**

仍缺乏硬件验证，极端初始状态鲁棒性有限，跨域迁移需进一步研究，奖励设计依赖经验且对复杂约束的推理不完全稳健。

---

## ParaCodex: A Profiling-Guided Autonomous Coding Agent for Reliable Parallel Code Generation and Translation

**arXiv ID:** 2601.04327 | [PDF](https://arxiv.org/pdf/2601.04327v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Distributed, Parallel, and Cluster Computing`

---

## AnimatedLLM: Explaining LLMs with Interactive Visualizations

**arXiv ID:** 2601.04213 | [PDF](https://arxiv.org/pdf/2601.04213v1)

**作者:** Zdeněk Kasner `[一作]` (Charles University), Ondřej Dušek `[通讯]` (Charles University)

**通讯引用:** 2857 | **OpenAlex IDs:** https://openalex.org/A5004829991

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

开发了一个基于浏览器的交互式可视化工具AnimatedLLM，用于展示Transformer语言模型的工作原理。

**💡 创新点**

创新点在于提供无服务器、预计算JSON跟踪的交互式演示，兼顾非技术受众与多模型多语言支持，桥接了静态教学材料与专业可视化工具的差距。

**🔧 技术方法**

使用React框架实现前端界面，利用Python脚本生成预计算模型跟踪（token化、概率分布等），并在浏览器中加载JSON文件进行可视化。

**📊 数据集**

基于Huggingface Transformers的多模型跟踪数据集，包含Olmo 3、Aya Expanse、Llama 3.2、Qwen 3、GPT‑2等模型在手工挑选提示上的完整推理轨迹。

**📈 对比分析**

目前未给出量化对比；通过在高校开放日和Czech AI Days等活动中的现场测试和用户反馈评估其教学效果，显示用户能够直观理解token化和自回归解码过程。

**⚠️ 局限性**

局限性包括只能展示预先计算的模型和提示，无法实时推理；模型范围受限于已准备的跟踪文件；仅支持有限语言；未来需加入更多视图和动态生成功能。

---

## Bridging Distance and Spectral Positional Encodings via Anchor-Based Diffusion Geometry Approximation

**arXiv ID:** 2601.04517 | [PDF](https://arxiv.org/pdf/2601.04517v1)

**作者:** Zimo Yan `[一作]` (National University of Defense Technology), Wumei Du `[通讯]` (National University of Defense Technology)

**通讯引用:** 11 | **OpenAlex IDs:** https://openalex.org/A5110930339

**关键词:** `Information Theory` `Drug Discovery` `Graph Neural Network` `Graph`

### 📋 论文摘要

**🎯 论文内容**

研究并构建了一个从锚点最短路径距离编码到拉普拉斯谱坐标的显式三角定位映射，提供了理论误差保证并验证其在分子图上的有效性；同时将该方法应用于药物-药物相互作用（DDI）预测，比较了不同位置编码对模型性能的影响。

**💡 创新点**

首次在理论上将锚点距离编码视为扩散几何的低秩近似，给出了可实现的三角定位算子与点误差、Frobenius误差上限，并在随机正则图和真实分子图上证明其收敛性；同时提出了一种距离驱动的Nyström近似来恢复扩散核。

**🔧 技术方法**

使用图神经过程（GNP）作为统一的DDI预测骨干，结合拉普拉斯谱定位（LapPE）、锚点距离编码（DE）以及距离变换ψ、Nyström近似、Procrustes对齐、局部单调映射拟合等技术。

**📊 数据集**

主要实验数据集为DrugBank和ChCh‑Miner两大药物交互数据库；随机正则图模型用于理论验证。

**📈 对比分析**

通过在同一GNP框架下仅替换位置编码方式进行对比，评估AUROC和F1指标；结果显示LapPE最佳，DE显著优于无位置编码（NoPE），两种方法均优于RWSE/HKS参考基线，且距离变换与锚点数量对性能影响显著。

**⚠️ 局限性**

理论仅适用于随机正则图，无法直接覆盖度异化、社区结构、加权或有向图；误差依赖于锚点放置、截断层数与距离变换的选择，缺乏对学习式锚点或非单调变换的分析；实验仅在两类DDI数据上验证，泛化性与在其他图任务上的表现尚未评估。

---

## Achievable Rate and Coding Principle for MIMO Multicarrier Systems With Cross-Domain MAMP Receiver Over Doubly Selective Channels

**arXiv ID:** 2601.04433 | [PDF](https://arxiv.org/pdf/2601.04433v1)

**作者:** Yuhao Chi `[一作]` (Xidian University), Chau Yuen `[通讯]` (Nanyang Technological University)

**通讯引用:** 39159 | **OpenAlex IDs:** https://openalex.org/A5060020877

**关键词:** `Information Theory` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种多时隙交叉域记忆近似消息传递（MS‑CD‑MAMP）接收机，用于在双选择性信道下的编码MIMO多载波系统（OFDM/OTFS/AFDM），并基于该接收机推导了可实现的速率上界与最优编码原则。

**💡 创新点**

创新点包括：
- 结合时间域稀疏性与符号域符号约束的交叉域MAMP架构；
- 通过多时隙记忆匹配滤波器和全正交操作实现低复杂度、可扩展的迭代检测；
- 采用简化的SISO变分状态演化（VSE）分析可实现速率并得到最优码设计；
- 证明OFDM、OTFS、AFDM在单元正交变换下可实现相同的最大可实现速率；
- 设计与VSE匹配的LDPC码，逼近理论极限。

**🔧 技术方法**

使用技术包括：
- 记忆匹配滤波器与迭代线性/非线性检测；
- 交叉域线性变换（IUT/UT）与正交化；
- 变分状态演化与I‑MMSE引理；
- LDPC码的线性规划优化；
- 复杂度分析与数值仿真。

**📊 数据集**

数据集：使用基于相关Rayleigh衰落、多径、时变（速度0~500 km/h）的合成信道模型；MIMO配置为4×4、8×4、8×8，子载波数N=256（OTFS时K=8,L=32），时隙数𝒯∈{1,5,25,5000}，仿真中采用QPSK、16QAM、Gaussian调制。

**📈 对比分析**

与现有CD‑OAMP、DD‑OAMP、DD‑MAMP、LMMSE等接收机以及P2P正则/不规则LDPC代码进行比较。结果表明：
- MS‑CD‑MAMP在BER=10⁻⁵时仅比MS‑CD‑OAMP/​VAMP慢约70%；
- 采用优化LDPC码时，BER距离理论极限仅0.5–1.8 dB，且比P2P LDPC提升0.8–4.4 dB；
- OFDM、OTFS、AFDM在该接收机下实现相同最大可实现速率；

**⚠️ 局限性**

限制：
- 分析依赖于信道已知且满足正交性假设；
- 需要解算器满足均匀Lipshitz连续性，未验证对所有FEC码的适用性；
- 仅针对单元正交变换的多载波调制，其他非正交调制的扩展尚未讨论；
- 目前的复杂度分析主要针对理想硬件，实际实现仍需进一步评估。

---

## Optimal Depth-Three Circuits for Inner Product

**arXiv ID:** 2601.04446 | [PDF](https://arxiv.org/pdf/2601.04446v1)

**作者:** Mohit Gurumukhani `[一作]`, Navid Talebanfard `[通讯]`

**关键词:** `Computational Complexity` `Optimization`

### 📋 论文摘要

**🎯 论文内容**

构造了最优深度为3、底层门极限为2的电路，用以计算 2n 维内积函数，并给出一种通用的模板方法。

**💡 创新点**

创新点在于将接受输入按对称轨道划分，并为每个轨道构造最大的 k-CNF，同时使用计算机搜索找到小型 2-CNF 构件，再通过解析组合技术组合成全局最优电路，首次实现与已知下界匹配的上界。

**🔧 技术方法**

主要技术包括轨道划分、k-CNF 设计、基于 4 变量构件的计算机搜索、以及解析组合计数方法来确定构件的最优组合。

**📊 数据集**

本研究不依赖外部数据集，主要使用符号与组合数学工具进行分析与搜索。

**📈 对比分析**

通过与 Göös、Guan、Mosnoi 在 Inform. Comput. ’24 给出的下界进行对比，电路规模为 (n)·(9/5)^n，证明了上界与下界的一致性，性能上达到理论最优。

**⚠️ 局限性**

局限性包括：方法主要针对高度对称的函数，构造过程需要大量计算机搜索，且规模随 n 指数增长；尚未展示如何扩展到更复杂或非对称的函数。

---

## Scaling Trends for Multi-Hop Contextual Reasoning in Mid-Scale Language Models

**arXiv ID:** 2601.04254 | [PDF](https://arxiv.org/pdf/2601.04254v1)

**作者:** Brady Steele `[一作]` (Georgia Institute of Technology), Micah Katz `[通讯]` (University of Texas at Austin)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Chain-of-Thought` `Mixture of Experts` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

在中等规模的语言模型上，使用人工生成的合成任务，对多跳语境推理能力进行受控实验，并验证了基于规则与LLM多代理系统的任务方法分离现象。

**💡 创新点**

①提出了可在消费者硬件上实现的合成多跳推理评估框架；②发现多代理系统对已有推理能力的模型产生显著放大效果，但无法弥补不足的模型；③证明在MoE模型中活跃参数数而非总参数数更能预测推理性能；④揭示同代模型间架构质量差异对推理表现的影响。

**🔧 技术方法**

多代理架构（Analyst–Strategist–Generator三体），链式思维提示，基于LangGraph的状态传递与迭代反馈；对比单代理与多代理、规则匹配基线；使用统计检验（Fisher、t检验）评估显著性。

**📊 数据集**

完全合成的多跳推理任务集合，包含结构化（单跳）与语境化（多跳）两类，共120个试验（每模型30个），每任务提供若干文本片段与需合成的目标字符串。

**📈 对比分析**

通过比较单代理、三代理以及基于规则的基线，在结构化任务中规则基线达到100%，在语境化任务中规则基线仅6.7%；LLaMA‑3 8B在多代理下提升达46.7pp（p<0.001），Mixtral提升13.3pp（p=0.014），弱模型无显著提升；结构化任务多代理往往下降。整体显示多代理对具备一定推理基线的模型具有显著性能提升。

**⚠️ 局限性**

样本量有限（每条件15次试验），模型覆盖仅四种；合成任务可能与真实多跳推理差异；不同模型家族的训练差异可能混入结果；提示与参数设置对性能影响未充分探索。

---

## MemKD: Memory-Discrepancy Knowledge Distillation for Efficient Time Series Classification

**arXiv ID:** 2601.04264 | [PDF](https://arxiv.org/pdf/2601.04264v1)

**作者:** Nilushika Udayangani `[一作]` (University of Melbourne), Marimuthu Palaniswami `[通讯]`

**关键词:** `Machine Learning` `Classification` `Knowledge Distillation` `Computational Efficiency` `Recurrent Neural Network` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

本文提出一种基于记忆差异的知识蒸馏框架 MemKD，利用 LSTM 训练出的教师模型与更小的学生模型之间的隐藏状态记忆变化差异进行蒸馏，显著压缩模型规模的同时保持分类性能。

**💡 创新点**

创新点在于：①设计了以子序列隐藏状态差异为核心的记忆差异损失函数，直接捕获 RNN 在处理不同时长子序列时的记忆演变；②该损失无需隐藏层维度对齐，突破了传统特征蒸馏对网络尺寸一致性的限制；③在同等条件下实现约 500 倍的参数与内存压缩率，性能几乎不失。

**🔧 技术方法**

主要技术包括：多层 LSTM 教师与单层隐藏尺寸极小的学生网络；基于子序列隐藏状态差异的自定义蒸馏损失（Smooth L1）；结合交叉熵的总训练损失；PyTorch 实现，Adam 优化，早停；归一化与子序列随机采样以稳定训练。

**📊 数据集**

使用了 12 个 UCR-2015 归档时间序列分类数据集，按序列长度分为短（≤150）、中（150–500）和长（>500）三类，包含 4 个样本每类，部分数据集类别数较多。

**📈 对比分析**

与无蒸馏基线（Base）、原始蒸馏（Base-KD）、FitNet、RKD、Att、DKD、DT2W 等方法对比，MemKD 在 12 个数据集上平均提升 5–10% 的 AUC-PRC，约 90% 数据集击败 Base-KD，2–5% 超越 FitNet，平均排名 1.75，甚至在包含教师模型的排名中也保持最低，显示出在压缩与性能平衡上的显著优势。

**⚠️ 局限性**

局限性包括：仅在 UCR-2015 分类任务上验证；仅针对 LSTM 结构的蒸馏，未探讨其他 RNN 或 Transformer 的适用性；压缩比例固定为 500×，未检验不同压缩程度的效果；缺乏实际设备部署与能耗评估，需进一步扩展实验范围。

---

## From Preoperative CT to Postmastoidectomy Mesh Construction:1Mastoidectomy Shape Prediction for Cochlear Implant Surgery

**arXiv ID:** 2601.04405 | [PDF](https://arxiv.org/pdf/2601.04405v1)

**作者:** Yike Zhang `[一作]` (St. Mary's University), Jack Noble `[通讯]` (Vanderbilt University)

**通讯引用:** 5310 | **OpenAlex IDs:** https://openalex.org/A5000197945

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Convolutional Neural Network` `Contrastive Learning` `Image` `Mesh` `Computed Tomography`

### 📋 论文摘要

**🎯 论文内容**

通过自监督与弱监督学习框架，利用预手术CT预测耳蜗植入手术中髓质切除区域并重建三维后髓质切除表面。

**💡 创新点**

首次将自监督与弱监督学习相结合，并引入3D T‑分布损失，避免人工标注、提升鲁棒性。

**🔧 技术方法**

使用SegMamba网络、MS‑SSIM+SCC损失、SoftPlus激活及3D T‑Distribution loss，并配合数据增强技术。

**📊 数据集**

采用751例来自Vanderbilt大学及合作医院的前后期CT扫描，分为训练、验证与测试集。

**📈 对比分析**

与UNet、UNet++、UNetr、SwinUNetr等传统网络比较，Dice达0.72/0.721，HD95约16，显著优于现有方法。

**⚠️ 局限性**

数据来源单一，缺乏广泛多中心验证；对其他解剖结构的通用性未评估；重建表面缺乏真实纹理。

---

## Quantifying the Effect of Test Set Contamination on Generative Evaluations

**arXiv ID:** 2601.04301 | [PDF](https://arxiv.org/pdf/2601.04301v1)

**作者:** Rylan Schaeffer `[一作]` (Stanford Computer Science), Sanmi Koyejo `[通讯]` (Stanford Computer Science)

**关键词:** `Machine Learning` `Generation` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

在预训练、过训练、监督微调及推理阶段，对生成评估中的测试集泄漏效应进行量化研究，尤其聚焦MATH数学推理基准。

**💡 创新点**

揭示单个测试集副本即可突破无污染模型的不可逆误差阈值，首次揭示生成评估与判别评估在记忆化机制上的根本差异，并发现温度采样与解答长度是抑制泄漏收益的关键杠杆；同时纠正了EleutherAI评估工具的实现错误。

**🔧 技术方法**

使用Qwen 3密集变压器架构，应用神经缩放律拟合预训练损失，利用温度采样实验、Math Verify验证以及交叉熵评估；并对EleutherAI LM Evaluation Harness进行修正。

**📊 数据集**

采用MATH基准（训练集与测试集）和高质量Web爬虫数据作为预训练语料。

**📈 对比分析**

通过对不同模型规模、测试集复制次数、过训练倍数、监督微调以及采样温度的系统对比，发现泄漏越多性能提升越大，但在高温或长答案情境下下降；单个副本可使模型损失低于无污染的不可逆误差。

**⚠️ 局限性**

仅研究了单一生成基准MATH，模型规模限制在344 M参数；未覆盖更大规模模型或其他架构（如MoE、SSM），也未验证在开放式对话或创意写作等更高熵任务中的表现。

---

## Learning Multinomial Logits in $O(n \log n)$ time

**arXiv ID:** 2601.04423 | [PDF](https://arxiv.org/pdf/2601.04423v1)

**作者:** Flavio Chierichetti `[一作]` (Sapienza University of Rome), Andrew Tomkins `[通讯]` (Google Research)

**关键词:** `Data Structures and Algorithms` `Recommendation System` `Optimization` `Computational Efficiency`

### 📋 论文摘要

**🎯 论文内容**

研究了如何仅通过对任意子集（slate）进行条件采样（返回赢家）来学习多项式对数模型（MNL），提出了自适应与非自适应两类算法；

**💡 创新点**

创新点在于引入“估计森林”（estimation‑forest）结构，通过构造带权图，使得仅通过多项式次数的比较即可近似得到所有物品的权重；同时提出了在不考虑权重范围的情况下，仅使用对数阶次的查询即可实现非自适应学习；

**🔧 技术方法**

主要技术包括：多项式对数模型的概率估计、几何分布的无偏估计、Median‑of‑Means（中位数-均值）估计、Chernoff‑Hoeffding与Chebyshev不等式、快速排序式的比较、森林遍历与权重分配、对数权重存储方案；

**📊 数据集**

该工作为理论研究，未使用具体实验数据集，所有结果均为理论分析与证明；

**📈 对比分析**

与朴素的对全局所有对（n²）比较的办法相比，所给算法的查询复杂度从O(n² log³n)降至自适应O(n log n/ε³)和非自适应O(n² log²n/ε³)，在相同精度下显著降低查询次数；

**⚠️ 局限性**

主要局限在于查询复杂度仍与ε⁻³成正比；对极端权重差异（大范围）仍需多次比较；此外，算法基于完全正权重假设，对零权重的处理虽可行但仍是近似处理。

---

## Embedding Textual Information in Images Using Quinary Pixel Combinations

**arXiv ID:** 2601.04302 | [PDF](https://arxiv.org/pdf/2601.04302v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition`

---

## Transitive Expert Error and Routing Problems in Complex AI Systems

**arXiv ID:** 2601.04416 | [PDF](https://arxiv.org/pdf/2601.04416v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## Pilot Study on Student Public Opinion Regarding GAI

**arXiv ID:** 2601.04336 | [PDF](https://arxiv.org/pdf/2601.04336v1)

**作者:** William Franz Lamberti `[一作]` (George Mason University), Samantha Rose Lawrence `[通讯]` (George Mason University)

**关键词:** `Artificial Intelligence` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

进行了一项针对乔治梅森大学计算与数据科学系入门计算课程学生的试点调查，评估他们对生成式人工智能（GAI）的认知与态度，并记录了极低的参与率。

**💡 创新点**

创新点在于将课堂教学前后对学生态度的变化进行前后对比，并首次系统报告了试点调查的低参与率（约4.4%），提示未来需要更大样本；同时通过Clopper-Pearson置信区间比较不同学期的参与率，揭示参与率显著差异。

**🔧 技术方法**

采用了问卷调查（7题）、教学视频展示、前后对比分析、二项分布模型和Clopper-Pearson精确置信区间计算，并在R语言中实现统计分析。

**📊 数据集**

使用了乔治梅森大学CDS 130课程的学生报名数据（共68人）以及3名自愿参与者的问卷答复数据。

**📈 对比分析**

通过比较两个试点（夏季与秋季）的参与率置信区间，发现两区间不重叠，表明参与率存在显著差异；但由于样本极小，关于问卷结果的统计意义有限，无法评估效能。

**⚠️ 局限性**

局限性主要包括样本量极小（仅3名参与者）、参与率低、缺乏显著统计结果、且仅在单一高校单一课程环境中开展，结果难以推广。

---

## Distribution-Guided and Constrained Quantum Machine Unlearning

**arXiv ID:** 2601.04413 | [PDF](https://arxiv.org/pdf/2601.04413v1)

**作者:** Nausherwan Malik `[一作]` (Lahore University of Management Sciences), Muhammad Faryad `[通讯]` (Lahore University of Management Sciences)

**通讯引用:** 1621 | **OpenAlex IDs:** https://openalex.org/A5040871513

**关键词:** `Machine Learning` `Optimization` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出了一种基于分布引导、约束优化的量子机器学习模型类级遗忘框架，能够在不重新训练的情况下消除指定类别的预测影响；

**💡 创新点**

创新点在于：① 用模型相似度统计生成自适应目标分布，避免了传统均匀分布导致的泛化问题；② 引入锚点保持约束，显式维护保留类别的预测行为；③ 将遗忘视为约束优化问题，给出拉格朗日松弛理论解释；

**🔧 技术方法**

采用变分量子电路（六量子比特、特征重上传、环形CNOT结构），参数梯度通过参数移位规则估计，并用Adam优化器进行梯度上升；同时使用KL散度度量与金标准模型的相似度；

**📊 数据集**

实验使用Iris和Covertype（取3、5、7类）两个公开数据集，经过PCA降维到四维后输入量子电路；

**📈 对比分析**

通过与统一目标遗忘方法以及完整重训练（gold retrain）模型对比，结果显示被遗忘类召回率从≈1降到0，保留类召回几乎不变；被遗忘类平均概率显著降低；对保留类样本的KL散度低于0.05，表明与金标准模型高度一致；

**⚠️ 局限性**

局限性包括：仅验证了类级遗忘，实例级或特征级遗忘尚未探索；实验仅在无噪声模拟器上进行，缺乏噪声设备评估；对大规模数据集的可扩展性和超参数调优仍需进一步研究。

---

## LEGATO: Good Identity Unlearning Is Continuous

**arXiv ID:** 2601.04282 | [PDF](https://arxiv.org/pdf/2601.04282v1)

**作者:** Qiang Chen `[一作]` (Central South University), Yi Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10306 | **OpenAlex IDs:** https://openalex.org/A5100728316

**关键词:** `Machine Learning` `Generation` `Data Synthesis` `Optimization` `Ordinary Differential Equation` `Image`

### 📋 论文摘要

**🎯 论文内容**

研究生成模型的身份遗忘（机器遗忘），提出LEGATO方法实现连续轨迹的身份忘却。

**💡 创新点**

创新点在于将Neural ODE作为轻量化适配器，将遗忘视为连续轨迹，利用可调步长实现可控遗忘，并通过轨迹一致性约束防止灾难性崩溃；实现参数效率与可解释性。

**🔧 技术方法**

技术包括Neural ODE、轨迹一致性约束（TC）、细粒度步长控制、轻量化适配器、联合遗忘/保持损失设计，以及数值求解器的选择。

**📊 数据集**

实验数据集主要为FFHQ（in‑domain）和CelebAHQ（out‑of‑domain），以及随机噪声样本，用于评估单图像与多图像身份遗忘。

**📈 对比分析**

与GUIDE、DoCo、RG、SalUn、LoRA等基线对比，LEGATO在ID、FID_pre、ΔFID_real等指标上取得最优或相近性能；参数更新量降低95%，平均更新时间减少67%，同时避免灾难性崩溃。

**⚠️ 局限性**

局限性包括：步长调参仍需经验；多身份遗忘干扰的鲁棒性尚需进一步验证；对不同生成模型的泛化能力未完全评估。

---

## Inhibitory Attacks on Backdoor-based Fingerprinting for Large Language Models

**arXiv ID:** 2601.04261 | [PDF](https://arxiv.org/pdf/2601.04261v1)

**作者:** Hang Fu `[一作]` (China Agricultural University), Yiming Xue `[通讯]` (China Agricultural University)

**通讯引用:** 676 | **OpenAlex IDs:** https://openalex.org/A5012766424

**关键词:** `Cryptography and Security` `Adversarial Attack` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

针对多模型协同（LLM集成）环境，研究并验证了两种新型的指纹攻击方法，能够在不改动任何模型参数的前提下抑制 LLM 后门指纹响应，同时保持或提升集成模型的下游任务性能。

**💡 创新点**

创新点在于：①提出 Token Filter Attack (TFA)——在解码时通过对所有模型的 top‑K token 集合进行两两交集并重新归一化，主动剔除指纹 token；②提出 Sentence Verification Attack (SVA)——利用句子级 perplexity 与投票机制在集成后阶段过滤指纹响应；③这两种方法兼顾了对集成模型的鲁棒性与对单模型指纹的高效抑制，填补了现有指纹攻击方法在集成场景下的空白。

**🔧 技术方法**

技术手段包括：token 级别的交集与概率聚合、句子级 perplexity 计算、投票频率过滤、top‑K 采样控制、温度/Top‑p/Top‑k 生成参数等；对比实验中还使用了全参数微调（SFT）、LoRA、以及现有指纹攻击策略（增量微调、GRI、MEraser、Merge、UniTE）作为基线。

**📊 数据集**

实验数据集：①六大下游基准（PIQA、ARC‑C、TriviaQA、MMLU、BoolQ、ANLI）评估模型实用性；②用于训练指纹模型的 SFT 数据（各自使用完整参数微调），并在基线实验中使用 Alpaca‑GPT4‑52k 数据；③在集成实验中使用多种开源 LLM（LLaMA2‑7B、LLaMA3.1‑8B、Qwen2.5‑7B、Amber‑7B、Mistral‑7B‑v0.1）及其 instruction‑tuned 版本。

**📈 对比分析**

与五种基线攻击方法相比，TFA 与 SVA 在三种指纹技术（IF、C&H、ImF）上均实现 90–100% 的攻击成功率（ASR），并在大部分下游任务中保持或提升 Accuracy（ACC），甚至在某些配置下达到与最佳单模型相同或更优的表现；相比之下，增量微调、GRI、Merge、UniTE 等方法要么失败、要么对性能有较大损害。

**⚠️ 局限性**

主要局限：①SVA 在所有模型使用相同指纹方法时 ASR 明显下降；②当主模型与辅助模型性能差距较大时，SVA 不能使集成模型超过最佳单模型；③TFA 对 top‑K 取值敏感（但在 10–30 范围内影响不大）；④实验仅在公开 LLM 与公开基准上验证，缺乏对更大规模或更稀有指纹方案的评估。

---

## How Users Consider Web Tracking When Seeking Health Information Online

**arXiv ID:** 2601.04485 | [PDF](https://arxiv.org/pdf/2601.04485v1)

**作者:** Martin P. Robillard `[一作]` (McGill University), Jin L. C. Guo `[通讯]`

**关键词:** `Human-Computer Interaction` `Safty and Privacy` `Text` `Electronic Health Records`

### 📋 论文摘要

**🎯 论文内容**

本研究通过对35名加拿大居民进行半结构化访谈，探讨了健康信息寻求者在网站选择、隐私增强技术使用以及自我审查等方面对网络跟踪的认知与实践。

**💡 创新点**

创新之处在于将用户隐私意识从“信息被收集”转向“信息如何被收集”，并揭示用户对第三方跟踪机制的误解与缺失。

**🔧 技术方法**

主要技术手段包括定性访谈、主题编码分析、以及对健康网站第三方跟踪的技术检测（使用Blacklight工具）。

**📊 数据集**

使用的数据集为35名访谈参与者的自我报告信息（年龄、性别、教育、隐私态度等）以及对多家健康网站的跟踪统计（如WebMD、WHO等）。

**📈 对比分析**

比较方法主要是对访谈主题的归纳对照和对网站跟踪量的描述性对比，未涉及数值性能指标，而是通过访谈主题频率和网站跟踪数量展示用户意识与实际跟踪差距。

**⚠️ 局限性**

研究局限包括样本以大学学历为主，缺乏更广泛的社会经济多样性；方法在研究中途进行了调整，导致两阶段访谈工具不完全一致，影响结果的可比性。

---

## 3D-Agent:Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation

**arXiv ID:** 2601.04404 | [PDF](https://arxiv.org/pdf/2601.04404v1)

**作者:** Jusheng Zhang `[一作]` (Sun Yat-sen University), Keze Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2250 | **OpenAlex IDs:** https://openalex.org/A5088124671

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Segmentation` `Domain Adaptation` `Reinforcement Learning` `Transformer` `Vision Language Model` `Reinforcement Learning` `Multimodality` `Point Cloud`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了 Tri-MARF 框架，用于高效、精准的 3D 对象标注。

**💡 创新点**

创新点在于三模态（多视角图像、文本、点云）输入与三位专职智能体协作，并利用强化学习的多臂赌博机对视角描述进行动态聚合与点云门控，显著提升标注质量与一致性。

**🔧 技术方法**

核心技术包括 Vision‑Language 模型（如 Qwen2.5‑VL‑72B‑Instruct）、RoBERTa+DBSCAN 语义聚类、CLIP 视觉‑文本对齐、多臂赌博机 UCB 策略、点云编码器门控以及强化学习奖励机制。

**📊 数据集**

使用了 Objaverse‑LVIS、Objaverse‑XL、ABO 等大规模 3D 数据集，并在 ShapeNet‑Core、ScanNet、ModelNet40 上进行跨域泛化测试。

**📈 对比分析**

与 Cap3D、ScoreAgg、3D‑LLM、PointCLIP 等方法对比，在 CLIPScore、ViLT R@5、A/B 评分上均取得 SOTA，吞吐率达 12k 物体/小时，优于现有所有基线。

**⚠️ 局限性**

局限性包括对 VLM 生成的幻觉仍有一定依赖、阈值设定较为经验化、在极端遮挡或低质量点云场景下性能下降，且跨域迁移需进一步细调。

---

## Sphinx: Benchmarking and Modeling for LLM-Driven Pull Request Review

**arXiv ID:** 2601.04252 | [PDF](https://arxiv.org/pdf/2601.04252v1)

**作者:** Daoan Zhang `[一作]` (University of Rochester), Elsie Nallipogu `[通讯]` (Microsoft)

**关键词:** `Software Engineering` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了Sphinx框架，结合LLM生成的伪代码与真实合并代码对比，构建了可解释的PR评审评论；

**💡 创新点**

创新点在于引入结构化数据生成、基于检查表的评估与奖励机制（CRPO）以及长度惩罚，实现了更精准、可解释的评审生成；

**🔧 技术方法**

核心技术包括LLM提示生成、监督微调、基于检查表的奖励策略优化、长度正则化以及对比评估；

**📊 数据集**

使用了自建的Sphinx数据集（约45K条PR，涵盖Java、JavaScript、C++、C#、Python）和5种语言各500条样本的Sphinx基准集；

**📈 对比分析**

与公开及专有LLM在BLEU‑1、ROUGE‑L和检查表覆盖率上对比，Sphinx模型在覆盖率上提高约30–40%，整体表现明显优于基线；

**⚠️ 局限性**

局限性包括仅处理单文件PR、依赖LLM生成伪代码的噪声、缺乏多文件及跨语言评审的通用性验证。

---

## Fuzzy Representation of Norms

**arXiv ID:** 2601.04249 | [PDF](https://arxiv.org/pdf/2601.04249v1)

**作者:** Ziba Assadi `[一作]` (Gran Sasso Science Institute), Paola Inverardi `[通讯]` (Gran Sasso Science Institute)

**通讯引用:** 4512 | **OpenAlex IDs:** https://openalex.org/A5039484805

**关键词:** `Artificial Intelligence` `Robotic Intelligence` `Safty and Privacy` `Fuzzy Logic`

### 📋 论文摘要

**🎯 论文内容**

本文将SLEEC伦理规则进行逻辑化表述，并引入可能性理论与模糊推理，将模糊语义与测试分数语义结合，构造可嵌入自主系统的可执行模糊规则；随后以医疗机器人打开窗帘的情景为例，演示如何通过模糊推理动态平衡隐私与健康之间的伦理冲突。

**💡 创新点**

创新点在于：①使用可能性而非概率处理伦理规则的模糊性；②将“unless”语义改写为IF–THEN–ELSE结构以提升机器可读性；③引入测试分数语义为模糊规则提供可度量的适用度；④在实际案例中展示如何用模糊推理解决伦理困境，并给出完整的推理与决策流程。

**🔧 技术方法**

采用的技术包括：模糊逻辑与隶属函数、可能性理论、测试分数语义、模糊推理（规则评估、聚合）、中心重心（COG）去模糊化、IF–THEN–ELSE 结构的规则编码，以及基于规则的决策算法。

**📊 数据集**

未使用公开数据集，而是构造了示例数据集：用户服装偏好（各类服装的可能度）、生命体征（年龄、血压、体温、心率等）的隶属函数及阈值，作为案例演示所需的输入。

**📈 对比分析**

本文没有与现有算法进行量化比较；通过单一案例演示，说明在模糊环境下系统能够在隐私与健康两大伦理维度之间做出平衡决策，但未给出具体性能指标或基准。

**⚠️ 局限性**

局限性包括：①验证仅限于单一场景，缺乏大规模实验；②模糊度量、阈值选择仍具主观性；③缺乏完整可执行实现与部署；④与传统逻辑或深度学习方法的结合与对比尚未完成。

---

## Balancing Usability and Compliance in AI Smart Devices: A Privacy-by-Design Audit of Google Home, Alexa, and Siri

**arXiv ID:** 2601.04403 | [PDF](https://arxiv.org/pdf/2601.04403v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## Survival Dynamics of Neural and Programmatic Policies in Evolutionary Reinforcement Learning

**arXiv ID:** 2601.04365 | [PDF](https://arxiv.org/pdf/2601.04365v1)

**作者:** Anton Roupassov-Ruiz `[一作]` (University of Alberta), Yiyang Zuo `[通讯]` (University of Alberta)

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning`

### 📋 论文摘要

**🎯 论文内容**

在经典Ackley–Littman演化强化学习（ERL）测试床上，用可微软决策列表（SDDL）实现的程序化策略（PERL）替代传统神经网络策略（NERL），重新实现并公开可复现的环境代码；

**💡 创新点**

提出将神经网络政策替换为可解释的可微软决策列表，展示了结构化程序化控制器在演化强化学习中可获得更高存活表现，验证了策略表示的先验假设对性能的关键作用；

**🔧 技术方法**

利用可微软决策列表（Soft Decision List）、演化强化学习（CRBP/REINFORCE风格）、Kaplan–Meier生存分析、对数秩检验和受限均值存活时间（RMST）等技术；

**📊 数据集**

使用重新实现的Ackley–Littman 1992年人工生命（ALife）演化强化学习环境（100×100格网，包含猎食者、植物、树木、墙壁等），不涉及外部公开数据集；

**📈 对比分析**

在4000个独立试验（每次最大2000步）下，对比PERL、NERL及其演化/学习/固定变体，使用Kaplan–Meier曲线、对数秩检验和RMST评估；结果显示PERL平均存活时间比NERL高约202步（p≈3.8×10⁻⁷⁵），且学习仅PERL也优于NERL；

**⚠️ 局限性**

实验受限于最大仿真步数2000导致右截断、以及仅在单一环境设置下验证，缺乏更长时间尺度和多样化环境的验证；

---

## Automatic Construction of Chinese Verb Collostruction Database

**arXiv ID:** 2601.04197 | [PDF](https://arxiv.org/pdf/2601.04197v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computation and Language`

---

## Hybrid Federated Learning for Noise-Robust Training

**arXiv ID:** 2601.04483 | [PDF](https://arxiv.org/pdf/2601.04483v1)

**作者:** Yongjun Kim `[一作]` (KAIST), Junil Choi `[通讯]` (KAIST)

**关键词:** `Machine Learning` `Federated Learning` `Knowledge Distillation` `Optimization` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出了一种混合联邦学习（HFL）框架，使每个终端设备可根据信道状态分别上传梯度或 logits，以实现更鲁棒的训练。

**💡 创新点**

创新点在于联合使用梯度和 logits 的混合上传，并通过 Jenks 聚类和阻尼牛顿方法自适应选择终端类别和融合权重，充分利用无线链路的自由度。

**🔧 技术方法**

采用无线链路模型、零逼近（ZF）解码、分布式梯度下降、知识蒸馏、Jenks 自然分段和阻尼牛顿优化等技术。

**📊 数据集**

使用 MNIST 数据集进行 10 类分类任务，且所有 30 个终端全参与训练。

**📈 对比分析**

与传统的 FedAvg 联邦学习和基于 logits 的联邦蒸馏进行对比，低信噪比（-20 dB、-15 dB）下，HFL 实现了更高的测试准确率并收敛更快。

**⚠️ 局限性**

局限在于对低 SNR 下 logit 误差的假设、仅采用 ZF 解码、未考虑设备异构或动态调度，以及缺少对更高级调制与 Beamforming 的深入分析。

---

## From Imitation to Innovation: The Divergent Paths of Techno in Germany and the USA

**arXiv ID:** 2601.04222 | [PDF](https://arxiv.org/pdf/2601.04222v1)

**作者:** Tim Ziemer `[一作]` (Institute of Systematic Musicology), Simon Linke `[通讯]` (Ligeti Center)

**关键词:** `Sound` `Audio`

### 📋 论文摘要

**🎯 论文内容**

对 1984-1994 年德国与美国早期 House 与 Techno 共 9,029 首曲目进行音频录音室特征提取，并通过 MANOVA、Self‑Organizing Map（SOM）以及随机森林（Random Forest）等方法进行统计与机器学习分析，以验证音乐人陈述的历史叙事。

**💡 创新点**

①将录音室特征（BPM、PhaseSpace、ChannelCorrelation、CrestFactor）作为可解释且互不相关的音频描述符，首次用大数据方式客观量化两国音乐风格差异；②通过将 MIR 与音乐学叙事相结合，证明音频分析能够验证主流与地下场景的“突破”与“停滞”理论；③提出“多元差异+SOM+RF”三层分析框架，对音乐趋势预测提供可复制的方法论。

**🔧 技术方法**

①录音室特征提取工具（BPM 估计、PhaseSpace、ChannelCorrelation、CrestFactor）
②多元方差分析（MANOVA）
③自组织映射（SOM）
④随机森林分类器（100 树、10‑折交叉验证）
④箱线图可视化、相关性检验与距离统计。

**📊 数据集**

HOTGAME 语料库：9,029 首 House/Techno 曲目（4,667 德国 + 4,362 美国），覆盖 1984–1994 年，包含不同子流派（Deep/Acid/Hardcore/Trance 等），由作者依据专辑封面、唱片标签、Discogs 等资料标注并随机校验。

**📈 对比分析**

通过 MANOVA 证明国别、年份及交互效应显著（p < 0.00001，效应量中等至大）；SOM 直观展示两国音乐在特征空间中的分布与聚类，US 曲目聚集于暗蓝区、德国曲目扩散至更广；随机森林分类在德国样本上测试准确率 0.515、精确率 0.503、召回率 0.515；在美国样本上准确率 0.369、精确率 0.362、召回率 0.369。比较显示德国样式在特征维度上更分散、更易区分，US 样式更相似、分类性能低。

**⚠️ 局限性**

①录音室特征仅覆盖声学混音参数，缺乏色调、节奏、旋律等重要音乐维度，导致 Acid House、Breakbeat 等风格难以区分；②特征提取使用中值归一化，忽略了时间序列动态；③样本仅来自公开发行曲目，可能偏离地下真实播放情况；④文化、政治、社群等非音频因素未纳入分析，影响对“突破”现象的完整解释；⑤交叉验证仅评估分类性能，未对预测音乐趋势的实用性进行外部验证。

---

## MedPI: Evaluating AI Systems in Medical Patient-facing Interactions

**arXiv ID:** 2601.04195 | [PDF](https://arxiv.org/pdf/2601.04195v1)

**作者:** Diego Fajardo V. `[一作]` (Lumos), Razvan Marinescu `[通讯]` (Lumos)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text` `Biomedical Data` `Electronic Health Records` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文构建了一个高维度的医疗对话评估基准MedPI，用LLM模拟患者、医生以及评审，以多维度评分评估LLM在真实临床对话中的表现。

**💡 创新点**

创新点在于：①把医学对话评估拆分为105维细粒度维度，映射至ACGME等认证标准；②使用LLM委员会式评审自动化评分；③一次性生成366名合成患者、7097条对话，实现大规模、可重复的评估；④提供完整评估工具和公开数据集。

**🔧 技术方法**

技术主要包括：LLM患者/医生模型（基于Gemini 2.5、Claude、GPT、Llama等）；情绪模型与多维检索的生成式代理；基于AIMR的评估框架与委员会式LLM评审；与synthea等合成EHR生成流程。

**📊 数据集**

使用的合成数据集：366名合成患者（由Synthea改造的FHIR‑like EHR），覆盖34种临床场景；通过LLM生成的7,097条患者–医生对话；公开数据集已发布在HuggingFace。

**📈 对比分析**

比较方法：对9种LLM在同一任务矩阵（诊断、生活方式、筛查等）下进行多轮对话，利用委员会式评审在105维度上给出1–4分，并归一化为0–100%得分。性能结果显示，尽管GPT‑5、Claude‑4等模型在技术可靠性和部分临床推理上表现较好，但在差异诊断、药物安全等关键维度普遍低于70%，整体仍处于安全使用阈值以下。

**⚠️ 局限性**

局限性：①评审全靠LLM，缺乏与医学专家的对标；②合成患者与对话真实性有限，人口分布不均；③评审使用单一LLM家族，可能产生厂商偏差；④未测试多模态、真实EHR接口或工具支持；⑤对评审稳定性与随机种子的影响未系统评估。

---

## A General Neural Backbone for Mixed-Integer Linear Optimization via Dual Attention

**arXiv ID:** 2601.04509 | [PDF](https://arxiv.org/pdf/2601.04509v1)

**作者:** Peixin Huang `[一作]` (Shandong University), Wei Zhang `[通讯]` (Shandong University)

**通讯引用:** 48232 | **OpenAlex IDs:** https://openalex.org/A5100675809

**关键词:** `Artificial Intelligence` `Optimization` `Representation Learning` `Graph Neural Network` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出了一种基于双重注意力机制的通用神经骨干网络，用于对混合整数线性规划（MILP）实例进行全局表示学习，并在实例级、元素级和求解状态级三类任务中进行应用。

**💡 创新点**

创新点在于将变量和约束分别视为两类重要元素，采用自注意力与交叉注意力并行处理，实现全局信息交换，克服传统GNN的局部性、过平滑和过压缩等瓶颈。

**🔧 技术方法**

使用了双重注意力机制、线性化自注意力、稀疏交叉注意力、并行推理与特征融合、残差连接与层归一化等技术，并通过PyTorch实现。

**📊 数据集**

使用了合成Foldable/Unfoldable MILP、MIS、CA、IP、WA、SC、CFL、以及MIPLIB 240实例等多种数据集进行实验。

**📈 对比分析**

与传统BGNN以及改进的GAT/GT等基线进行对比，采用实例可行性预测误差、目标值MSE、二进制变量预测Macro‑F1、PG/PI、B&B树节点数等指标，实验表明本方法在所有任务上均显著优于基线，提升幅度约10–30%，且在MIPLIB上泛化更稳健。

**⚠️ 局限性**

主要局限在于自注意力的时间与内存复杂度较高，尤其对极大规模或极度不平衡标签的MILP实例仍需进一步优化；此外模型目前仅适用于线性MILP，对非线性或随机混合整数优化尚未覆盖。

---

## RAGVUE: A Diagnostic View for Explainable and Automated Evaluation of Retrieval-Augmented Generation

**arXiv ID:** 2601.04196 | [PDF](https://arxiv.org/pdf/2601.04196v1)

**作者:** Keerthana Murugaraj `[一作]` (University of Luxembourg), Martin Theobald `[通讯]` (University of Luxembourg)

**通讯引用:** 3303 | **OpenAlex IDs:** https://openalex.org/A5060837952

**关键词:** `Computation and Language` `Explainability and Interpretability` `Retrieval` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一套完整的、无参考答案的诊断评估框架RAGVue，用于自动评估检索增强生成（RAG）系统的检索质量、答案相关性、完整性、可解释性与可靠性。

**💡 创新点**

1）将RAG评估拆分为检索、答案质量与事实基础三大维度，并提供每一维度的细粒度指标；2）采用LLM-评判器在单一步骤内完成多重判定，确保严格的声明级可信度；3）引入裁准（Calibration）指标显式衡量评判器的稳定性；4）实现了人工与代理自动评估两种模式，提供可解释的报告。

**🔧 技术方法**

基于大型语言模型的评判器（LLM-as-a-judge）、Prompt设计、Python API、命令行接口、Streamlit交互界面、代理式评估器（agentic mode）以及跨模型一致性校准算法。

**📊 数据集**

使用从StrategyQA多跳问答数据集提取的100条合成问答三元组，包含五种答案变体（理想、部分、含糊、偏离、虚假）。

**📈 对比分析**

与现有RAGAS评估器进行对比；在100条样本上，RAGVue的平均延迟略高（≈18.87 s vs 18.26 s），但提供了更细粒度、可解释的诊断信息；统计上RAGVue在检索覆盖、检索相关、严格可信度等维度与RAGAS的对应指标显著不同，能够揭示RAGAS忽略的错误来源。

**⚠️ 局限性**

仍依赖LLM评判器，可能受提示敏感性、温度设置和模型偏见影响；裁准指标虽然可用，但在极端噪声或多模型配置下仍需进一步验证；该框架不提供绝对的黄金参考，评估结果仍以LLM推理为基础，若评判器本身错误会影响最终报告。

---

## Making Tunable Parameters State-Dependent in Weather and Climate Models with Reinforcement Learning

**arXiv ID:** 2601.04268 | [PDF](https://arxiv.org/pdf/2601.04268v1)

**作者:** Pritthijit Nath `[一作]` (University of Cambridge), Mark J. Webb `[通讯]`

**关键词:** `Machine Learning` `Reinforcement Learning` `Federated Learning` `Reinforcement Learning`

### 📋 论文摘要

**🎯 论文内容**

本文通过强化学习（RL）在线学习并调节天气与气候模型中可调参数，使其随模型状态变化而自适应，从而减少结构性偏差并提升模拟精度。

**💡 创新点**

创新点在于提出一种层级化的RL框架，将RL与传统物理参数耦合，并通过联邦学习（FedRL）实现空间分区的协同训练，首次展示RL能够学习到物理意义明确且区域特定的参数调整。

**🔧 技术方法**

主要技术包括连续控制的RL算法（TQC、DDPG、TD3等）、联邦学习框架（FedRL）以及智能中间件（SmartSim/SmartRedis）用于将Fortran数值模型与Python RL代理高效耦合。

**📊 数据集**

使用的实验数据集为理想化气候模型环境：简单气候偏差校正（SCBC）、辐射-对流平衡（RCE）以及带有纬度分区的能量平衡模型（EBM），并通过模拟观测对比（如NCEP/NCAR再分析）评估偏差。

**📈 对比分析**

对比方法为与传统静态参数化（通过线性回归校准）及不同RL算法的性能，评估指标包括返回阈值突破速度、返回稳定性、最终RMSE/MAE与观测的差异。实验显示，TQC、DDPG、TD3三种算法在所有层级中均获得最高技能、最快收敛且最稳健的结果，RL模型显著降低了温度、通量和辐射误差。

**⚠️ 局限性**

限制主要体现在：①RL参数学习依赖于手工设计的奖励与状态空间，过度依赖特定环境；②在更复杂的全尺度GCM中，计算成本与训练稳定性仍有待提升；③联邦学习中不同区域间的参数共享频率与同步策略对最终性能影响较大，需进一步优化。

---

## A Privacy-Preserving Localization Scheme with Node Selection in Mobile Networks

**arXiv ID:** 2601.04280 | [PDF](https://arxiv.org/pdf/2601.04280v1)

**作者:** Liangbo Xie `[一作]` (Chongqing University of Posts and Telecommunications), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 80345 | **OpenAlex IDs:** https://openalex.org/A5091266202

**关键词:** `Cryptography and Security` `Safty and Privacy` `Computational Efficiency` `Homomorphic Encryption` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出了一种名为PPLZN的隐私保护定位方案，利用零和噪声与Paillier同态加密实现目标与基站位置的互不泄露。

**💡 创新点**

核心创新是将零和噪声生成机制与同态加密结合，既能消除噪声对定位精度的影响，又能在不公开单个基站噪声的前提下完成加密求和；同时设计了基于GDOP贡献度的节点选择算法，显著降低密集基站环境下的计算与通信开销。

**🔧 技术方法**

主要技术包括Paillier同态加密、零和噪声生成（ZSNG）、基于GDOP的节点选择算法（NSA）以及ToA定位的矩阵分解与加密求和。

**📊 数据集**

使用模拟数据集：三维定位场景（1000m×1000m×100m）中50个目标，6–30个移动/静止基站，加入6.1 ns高斯噪声的ToA测距。

**📈 对比分析**

与EPPL、P^3‑Pro、PPRP和全同态加密方案对比，PPLZN在30基站时计算开销降低45.5%，15基站时通信量比P^3‑Pro低26%，定位误差相比原始ToA仅增加约15%，但优于其他加密方案。

**⚠️ 局限性**

主要局限在于仍存在15%左右的定位误差增幅（由于节点选择导致信息损失），以及Paillier加密对计算资源的依赖，且实验仅基于理想化的模拟环境，未考虑NLoS和同步误差等实际挑战。

---

## Mitigating Position-Shift Failures in Text-Based Modular Arithmetic via Position Curriculum and Template Diversity

**arXiv ID:** 2601.04283 | [PDF](https://arxiv.org/pdf/2601.04283v1)

**作者:** Nikolay Yudin `[一作]` (Independent Researcher), Nikolay Yudin `[通讯]` (Independent Researcher)

**关键词:** `Machine Learning` `Transformer` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究字符级Transformer在文本中计算模97加法任务，重点评估输入格式变化下的鲁棒性；

**💡 创新点**

发现“位置偏移”失效模式并提出一套训练干预（位置多样性、位置课程、模板多样性、一致性训练和可选边界标记）显著提升格式不变性；

**🔧 技术方法**

采用位置控制填充、基于步骤的位姿课程、模板混合、表达式边界标记以及跨变体一致性损失，模型为小型Transformer编码器；

**📊 数据集**

使用伪造的模97加法数据集（9409个有序对），按50/50不重叠拆分训练/测试，并生成多种文本模板与位置变体；

**📈 对比分析**

通过Eval-A/B/C0/C1四套评估对比，基线仅在分布内高精度但位置/模板OOV极差；加入干预后Eval-B与Eval-C0提升至约70–80%，Eval-A保持≈96%，对比ALiBi缺失时表现低至21%；

**⚠️ 局限性**

局限于单步模97加法，位置课程不覆盖0–8位，模板多样性有限，未验证更复杂算术表达或更广自然语言变体，也未深入分析学习到的机制。

---

## Learning to Reason: Temporal Saliency Distillation for Interpretable Knowledge Transfer

**arXiv ID:** 2601.04263 | [PDF](https://arxiv.org/pdf/2601.04263v1)

**作者:** Nilushika Udayangani Hewa Dehigahawattage `[一作]` (University of Melbourne), Marimuthu Palaniswami `[通讯]` (University of Melbourne)

**通讯引用:** 30261 | **OpenAlex IDs:** https://openalex.org/A5080554686

**关键词:** `Machine Learning` `Knowledge Distillation` `Explainability and Interpretability` `Compression` `Time Series` `Recurrent Neural Network` `Supervised Fine-Tuning` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

开发了一种基于时间序列显著性知识蒸馏的模型压缩方法，利用教师模型对每个时间步的重要性进行可解释性迁移；

**💡 创新点**

创新点在于将教师的时间步显著性（通过对单步扰动的KL散度衡量）作为蒸馏目标，既保留预测准确性又提升解释一致性；

**🔧 技术方法**

采用温度缩放的softmax、KL散度显著性度量、Smooth L1 蒸馏损失，并在PyTorch框架下实现；

**📊 数据集**

在 UCR 2015 时间序列分类数据集（28 个子集）上进行实验；

**📈 对比分析**

与基线、Base‑KD、FitNet、RKD、DKD、DT2W 等方法比较，TSD 在大多数数据集上取得最高 AUC‑PRC，平均排名最低，整体提升约 3.7%；

**⚠️ 局限性**

局限性包括对子序列宽度、数量和温度等超参数敏感，在某些数据集上提升有限，且仅基于 logit 层，未利用更深层特征信息。

---

## Performance Analysis of Image Classification on Bangladeshi Datasets

**arXiv ID:** 2601.04397 | [PDF](https://arxiv.org/pdf/2601.04397v1)

**作者:** Mohammed Sami Khan `[一作]` (Dhaka University), Rowzatul Zannat `[通讯]` (Dhaka University)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Convolutional Neural Network` `Image`

### 📋 论文摘要

**🎯 论文内容**

本文对比了自定义CNN与预训练模型（VGG‑16、ResNet‑50、MobileNet/ConvNeXt‑Tiny）在巴基斯坦多种图像分类数据集上的表现；

**💡 创新点**

创新点在于在相同实验设置下，对自研模型与多种主流预训练模型进行系统、可复现的对比评估，并对不同模型的参数量、训练时长与精度进行细致分析；

**🔧 技术方法**

采用了PyTorch框架实现从头训练的CNN、迁移学习（冻结/微调）以及数据增强技术；

**📊 数据集**

使用的五个公开数据集包括：Auto‑RickshawImageBD、FootpathVisionBD、RoadDamageBD、MangoImageBD、PaddyVarietyBD；

**📈 对比分析**

通过在统一数据预处理、批量大小、学习率、损失函数等条件下，评估准确率、精确率、召回率、F1‑score、训练时间与参数量，结果显示ConvNeXt‑Tiny在大多数任务上取得最高精度，ResNet‑50次之，而自定义CNN在简单二分类任务中表现可比，但总体参数量大、收敛慢；

**⚠️ 局限性**

局限性包括：仅使用五个数据集且主要为二分类/细粒度多分类；实验仅涉及部分预训练模型，未探讨更大规模网络或多任务学习；以及对模型超参数的调优相对有限，可能未达到最优性能。

---

## Longitudinal Trends in Pre University Preparation. A Cohort Evaluation Using Introductory Mathematics and Physics Courses (1980-2019)

**arXiv ID:** 2601.04360 | [PDF](https://arxiv.org/pdf/2601.04360v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## Improving and Accelerating Offline RL in Large Discrete Action Spaces with Structured Policy Initialization

**arXiv ID:** 2601.04441 | [PDF](https://arxiv.org/pdf/2601.04441v1)

**作者:** Matthew Landers `[一作]` (University of Virginia), Afsaneh Doryab `[通讯]` (MBZUAI)

**关键词:** `Machine Learning` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出了 SPIN 两阶段框架，先用自监督掩码建模预训练 Action Structure Model 捕捉离散组合动作空间的低维流形，再冻结该表示并训练轻量级策略头完成控制。

**💡 创新点**

将动作结构学习与策略学习解耦，先学习合法动作集合的结构后再在该表示上进行快速、稳定的策略学习；自监督掩码条件建模有效挖掘动作空间结构。

**🔧 技术方法**

使用 Transformer 编码器（无位置编码以保持子动作可换性）进行自监督掩码建模，第二阶段采用 SAINT 或轻量 MLP 策略头，并兼容 IQL、AWAC、BCQ 等离线 RL 目标；线性探针评估表示质量。

**📊 数据集**

在离线化的 DeepMind Control Suite（离散化版本）上实验，使用四类数据集（random‑medium‑expert、medium‑expert、medium、expert）并覆盖多任务、不同动作维度与离散度。

**📈 对比分析**

与 SAINT、Factored、Autoregressive 等基线在同一数据集、目标下对比；SPIN 在平均回报上最高（整体平均 594.1 vs 572.1），并在收敛速度上最快（达到 95% 目标仅 223.3 分钟，比 F‑IQL 快 2.5×，比 SAINT 快 3.8×）。

**⚠️ 局限性**

局限性：仅适用于可换性结构的组合动作；不兼容需要全局价值正则化的算法（如 CQL）；对离线数据的覆盖要求较高，难以处理数据稀疏或偏斜情况。

---

## Combining facial videos and biosignals for stress estimation during driving

**arXiv ID:** 2601.04376 | [PDF](https://arxiv.org/pdf/2601.04376v1)

**作者:** Paraskevi Valergaki `[一作]` (Foundation for Research and Technology Hellas), Anastasios Roussos `[通讯]` (Foundation for Research and Technology Hellas)

**通讯引用:** 1924 | **OpenAlex IDs:** https://openalex.org/A5029931791

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Autonomous Driving` `Transformer` `Video` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

研究在分心驾驶情境下利用解耦3D面部几何、心率等多模态信号进行压力识别，并提出Transformer时序模型与跨模态注意力融合。

**💡 创新点**

创新点在于引入EMOCA的3D表情与姿态系数作为面部特征，比传统AU更能反映压力，并设计跨模态双向注意力框架，显著提升多模态融合效果。

**🔧 技术方法**

技术手段包括EMOCA 3D面部重建、系数差分与速度特征提取、卷积前置、Transformer编码器、跨模态双向注意力以及窗口化时序池化。

**📊 数据集**

使用了Taamneh等人发布的60余名志愿者的多模态分心驾驶数据集，包含红外面部视频、生理信号、注视、车辆动态等多种同步数据。

**📈 对比分析**

通过5折受试者交叉验证与传统SVM/kNN、MLP基线对比，跨模态注意力融合在AUROC 0.92、准确率0.866等指标上显著优于单模态或早期融合方法。

**⚠️ 局限性**

局限性包括实验仅在模拟驾驶环境中进行，缺乏真实道路验证；面部解码对光照与遮挡敏感；多模态同步与样本量仍有限。

---

## Active Sensing Shapes Real-World Decision-Making through Dynamic Evidence Accumulation

**arXiv ID:** 2601.04214 | [PDF](https://arxiv.org/pdf/2601.04214v1)

**作者:** Hongliang Lu `[一作]` (Hong Kong University of Science and Technology), Junjie Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 540 | **OpenAlex IDs:** https://openalex.org/A5101891276

**关键词:** `Artificial Intelligence` `Autonomous Driving` `Optimization` `Video`

### 📋 论文摘要

**🎯 论文内容**

开发了动态证据累积模型（DEAM），用于研究真实道路驾驶中的主动感知与决策过程。

**💡 创新点**

创新点在于将模糊逻辑定义的证据易用性与清晰度引入EAM，并结合动态注意力调节和崩溃阈值，实现了对视觉主动感知与决策相互作用的完整刻画。

**🔧 技术方法**

使用了基于注意力漂移扩散模型（aDDM）的动态证据累积模型（DEAM），并通过遗传算法对六个关键参数进行优化，同时利用眼动数据映射进行实时证据提取。

**📊 数据集**

使用公开的 DR(eye)VE 数据集，包含约 370 分钟、500,000 帧的真实驾驶录像和眼动记录，用于提取车道变换和跟车两类驾驶事件。

**📈 对比分析**

与 aDDM 以及传统 EAM 对比，DEAM 在决策概率、反应时、注意力切换等指标上表现更佳，均方误差（MSE）显著降低（如车道变换 MSE≈0.021 vs aDDM 0.024，跟车 MSE 0.60 vs 0.76），并成功拟合人类行为。

**⚠️ 局限性**

局限主要在于仅考虑视觉信息，忽略声音、触觉等多模态感知；未获得 EEG 等神经信号验证模型神经可解释性；模型参数需针对不同驾驶环境手动调整。

---

## Using Grok to Avoid Personal Attacks While Correcting Misinformation on X

**arXiv ID:** 2601.04251 | [PDF](https://arxiv.org/pdf/2601.04251v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Social and Information Networks`

---

## STDD:Spatio-Temporal Dynamics-Driven Token Refinement in Diffusion Language Models

**arXiv ID:** 2601.04205 | [PDF](https://arxiv.org/pdf/2601.04205v1)

**作者:** Xinhao Sun `[一作]`, Xiang Chen `[通讯]`

**关键词:** `Computation and Language` `Generation` `Computational Efficiency` `Transformer` `Large Language Model` `Diffusion model` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一种基于时空动态的Token重掩模（STDD）方法，用以加速扩散语言模型的文本生成。

**💡 创新点**

创新点在于：①利用每个Token的时间方差和空间偏离度动态计算阈值；②设计可变阈值和可行性优化机制，使得不同Token在不同步长有自适应的解码优先级；③实现了对已有加速技术的无缝集成，显著提升速度同时保持甚至提升质量。

**🔧 技术方法**

核心技术包括：扩散语言模型（DLM）架构、Token重掩模策略、时间方差（temporal variance）与空间偏离度（spatial deviance）分析、动态阈值计算、可行性优化（fast/slow token机制）以及与dKV-Cache和In-place Prompting等加速手段的融合。

**📊 数据集**

实验使用了三大基准数据集：GSM8K（数学推理）、MBPP（编程生成）和MATH（高难度数学问题）。

**📈 对比分析**

与LLaDA、Dream、Fast‑dLLM以及DUS等方法对比，STDD在LLaDA‑Instruct‑8B模型上分别实现了GSM8K 3.07×、MBPP 8.9×、MATH 3.74×的速度提升；在Dream‑7B模型上也获得3.41×、2.91×、3.65×的加速。质量方面，在GSM8K上准确率提升至83.1%，在MATH上达到35.1%，均优于基线与现有最优方法。

**⚠️ 局限性**

局限性包括：①阈值参数（如窗口大小W_t/W_n、p、c_fast/c_slow等）需要经验调优，可能不适用于所有模型或任务；②对极端长序列或异常Token的动态阈值调整可能仍产生误判；③在极度嘈杂或不确定的推理场景下，快速/慢速Token机制可能导致偶尔的解码失败或质量下降。

---

## When Predictions Shape Reality: A Socio-Technical Synthesis of Performative Predictions in Machine Learning

**arXiv ID:** 2601.04447 | [PDF](https://arxiv.org/pdf/2601.04447v1)

**作者:** Gal Fybish `[一作]` (Massey University), Teo Susnjak `[通讯]` (Massey University)

**通讯引用:** 2681 | **OpenAlex IDs:** https://openalex.org/A5037915797

**关键词:** `Machine Learning` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

本文系统性综述了预测模型的可执行性（performative prediction）现象，梳理了其机制、风险与解决策略，并提出了“可执行强度与影响矩阵”（Performative Strength vs. Impact Matrix）来帮助实践者评估并选择相应的治理方案。

**💡 创新点**

创新点在于：①首次将可执行性研究框架化为三维维度（机制、风险、策略）；②构建了可执行强度与影响两维的九格矩阵，并进一步划分为三大治理区间；③整合并对比了从算法到治理的多类别解决方案，为高风险场景提供了系统化的决策指引。

**🔧 技术方法**

主要使用的技术是文献检索与系统综述方法，结合概念映射、风险分类与策略树结构的构建；在矩阵设计中运用层级阈值与案例映射来量化强度与影响。

**📊 数据集**

该工作未使用传统实验数据集，而是基于已公开的案例（如医院再入院预测、预后死亡模型、信用评分等）进行概念示例；因此不存在具体数据集需求。

**📈 对比分析**

由于是综述性工作，并未进行实验对比或性能评估；文中仅通过已发表研究的引用和案例说明各策略的适用场景与潜在效果，没有给出数值指标。

**⚠️ 局限性**

主要局限包括：①文献检索仅限 2019‑2025 年期刊与会议，忽略早期相关工作；②以“performative prediction”关键词为主，可能遗漏同类概念；③缺乏对矩阵及区间划分的经验验证与量化指标；④对实证案例的评估依赖作者主观判定，缺乏客观实验支持。

---

## Application of Hybrid Chain Storage Framework in Energy Trading and Carbon Asset Management

**arXiv ID:** 2601.04512 | [PDF](https://arxiv.org/pdf/2601.04512v1)

**作者:** Yinghan Hou `[一作]` (Imperial College), Xiaokun Yang `[通讯]` (Nanchang Institute of Technology)

**通讯引用:** 596 | **OpenAlex IDs:** https://openalex.org/A5058511319

**关键词:** `Cryptography and Security` `Tabular` `Time Series` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

构建了一种混合链上链下的能源交易与碳资产结算与审计框架，支持高频小额结算并保证可审计性。

**💡 创新点**

创新点在于：①将结算摘要通过哈希或RSA累加器固定地锚定在链上，②实现可重放的审计；③在链上直接强制执行碳资产生命周期保留约束；④采用两层身份门控的选择性披露。

**🔧 技术方法**

使用的技术包括以太坊智能合约、Keccak‑256哈希、RSA累加器、Merkle树、身份验证与授权的DID与授权签名、批量处理与罚金机制。

**📊 数据集**

实验数据集基于公开的PJM LMP、EIA 930、EU ETS碳排放统计以及ICAP碳配额价格，用来合成交易与碳资产记录。

**📈 对比分析**

通过与全链上直接提交基线方案比较，使用批量哈希摘要降低了约39%的gas消耗；对容量限制下的单笔gas几乎不随集合大小变化；所有攻击和无效操作均被检测/拒绝，审计可重放率达到100%。

**⚠️ 局限性**

局限性在于：不包含完整的市场清算逻辑或碳监测的官方规则；碳资产生命周期约束仅为最小一致性检查；离线数据处理时间未计入链上成本，且实验在本地测试网络中进行，未在主网真实交易负载下验证。

---

## Summary of The Inaugural Music Source Restoration Challenge

**arXiv ID:** 2601.04343 | [PDF](https://arxiv.org/pdf/2601.04343v1)

**作者:** Yongyi Zang `[一作]`, Mark D. Plumbley `[通讯]`

**关键词:** `Sound` `Restoration` `Transformer` `Generative Adversarial Network` `Audio` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

设立首个音乐源恢复（MSR）挑战，提供专业混音、真实退化混音及其未处理的原始源，构建评估框架并评测各团队算法。

**💡 创新点**

创新点在于：①提供完全未加混音效果的基准源，弥补传统分离挑战的局限；②提出多维度客观（Multi‑Mel‑SNR、Zimtohrli、FAD‑CLAP）与主观（MOS）评估体系；③通过实验揭示多阶段处理和数据质量对恢复性能的决定性作用。

**🔧 技术方法**

主要技术包括基于Transformer的多阶段架构（BSRoformer、BSRNN、MDX23、DTT‑BSR）、预训练的音乐源分离模型、简单的重建损失（L1、STFT）以及少量GAN/对抗训练的探索。

**📊 数据集**

使用的数据集包括手工清洗的RawStems、MUSDB18‑HQ、MoisesDB、MedleyDB、URMP、MAESTRO；验证集为MSRBench（2000个10s混音），测试集分为非盲（含真值）和盲（无真值）两部分。

**📈 对比分析**

采用Multi‑Mel‑SNR、Zimtohrli、FAD‑CLAP三项客观指标与MOS主观评分进行横向对比。xlancelab团队以4.46 dB的Multi‑Mel‑SNR和3.47的MOS‑Overall位列第一，第二名CUPAudioGroup仅为2.34 dB/2.93 MOS，差距显著。

**⚠️ 局限性**

局限性包括：对鼓等短脉冲、复杂多音源的恢复仍十分困难；单音源（如人声）提升有限；多阶段或GAN训练需更精细调参；模型对不同退化类型的泛化能力尚未充分验证。

---

## Complexity Agnostic Recursive Decomposition of Thoughts

**arXiv ID:** 2601.04210 | [PDF](https://arxiv.org/pdf/2601.04210v1)

**作者:** Kaleem Ullah Qasim `[一作]` (Southwest Jiaotong University), Hafiz Saif Ur Rehman `[通讯]` (Southwestern University of Finance and Economics)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Chain-of-Thought` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了基于预估复杂度的递归推理框架 CARD

**💡 创新点**

在推理前进行多维复杂度估计，实现自适应分解深度和思路预算

**🔧 技术方法**

使用 0.6B Qwen 模型 MRCE 预测 30 维特征，配合两阶段递归分解和自适应思路预算

**📊 数据集**

在 GSM8K 与 MATH‑500 两个数学推理数据集上进行评估

**📈 对比分析**

相较于 CoT、ToT、RDoLT 等固定分解方法，CARD 在 GSM8K 和 MATH‑500 上提升 1.7–2.2 个百分点准确率，同时 token 消耗下降 1.71×至 5.74×

**⚠️ 局限性**

仅针对数学推理，缺乏在常识推理、代码生成等领域的通用性；且需人工标注的多维复杂度特征，限制了迁移与自动化

---

## AdaptEval: A Benchmark for Evaluating Large Language Models on Code Snippet Adaptation

**arXiv ID:** 2601.04540 | [PDF](https://arxiv.org/pdf/2601.04540v1)

**作者:** Tanghaoran Zhang `[一作]` (National University of Defense and Technology), Yue Yu `[通讯]` (Peng Cheng Laboratory)

**通讯引用:** 3811 | **OpenAlex IDs:** https://openalex.org/A5100397991

**关键词:** `Software Engineering` `AI Code Assistant` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

构建了AdaptEval基准，用于系统评估大语言模型在代码片段适配任务中的表现，并基于此基准对多款LLM进行了实证研究。

**💡 创新点**

创新点包括：①引入实际开发上下文、②多粒度（任务级与适配级）注释体系、③细粒度评估框架（适配级与函数级测试用例），为LLM在代码重用适配场景提供全新的评测手段。

**🔧 技术方法**

技术手段包括：GumTree AST差分+CLDiff分组、SourcererCC类型‑2/3代码克隆、Tree‑sitter依赖抽取、Python unittest构建双层测试、pass@k执行评估，以及基于CDD与变异检测的数据泄露评估。

**📊 数据集**

数据集为从Stack Overflow与GitHub挖掘的Python代码重用案例，包含164个适配任务、523个具体适配、38种适配类型，覆盖率达92.95%分支与94.38%行。

**📈 对比分析**

在最充分的AReq+Oracle情景下，指令调优LLM的pass@1‑t最高为59.15%，细粒度pass@1‑a最高为72.31%；相较任务级需求提升约34.84%；不同适配类型表现差距可达20.31%，体现出LLM在方法签名适配方面表现优于逻辑定制等复杂适配。

**⚠️ 局限性**

局限性：仅限Python单语言且仅覆盖SO→GitHub的适配场景，注释仍带有主观性；未覆盖负分帖子、极大适配情况；人工编写测试与提示可能导致偏差；LLM仍受预训练知识影响，指令遵循与推理对齐存在挑战。

---

## Safety-Utility Conflicts Are Not Global: Surgical Alignment via Head-Level Diagnosis

**arXiv ID:** 2601.04262 | [PDF](https://arxiv.org/pdf/2601.04262v1)

**作者:** Wang Cai `[一作]` (Baidu Inc.), Yunfang Wu `[通讯]` (Peking University)

**通讯引用:** 981 | **OpenAlex IDs:** https://openalex.org/A5027803148

**关键词:** `Machine Learning` `Safty and Privacy` `Optimization` `Transformer` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了 Conflict-Aware Sparse Tuning（CAST）框架，通过先在头层级诊断安全与通用功能冲突，随后仅更新低冲突头部实现安全对齐，显著减少对通用能力的损失。

**💡 创新点**

创新点在于：①发现冲突在 Transformer 头层级是结构化分布而非全局均匀；②将优化冲突与功能敏感度结合成冲突得分，构建预对齐冲突图；③将该诊断与稀疏微调相结合，形成结构感知的对齐策略。

**🔧 技术方法**

技术包括：梯度方向角度（优化冲突）评估；轻量级零样本头部屏蔽（功能敏感度）评估；冲突得分 C(h)=O(h)·S(h)；预算匹配稀疏微调与 PCGrad 等几何对齐方法的联合使用。

**📊 数据集**

使用的数据集：对齐训练集 WildJailbreak（10k 条），对齐诊断集 D_util（MMLU 500 条）和 D_safe（WildJailbreak 500 条），安全评估使用 WildJailbreak、WildGuard、DAN，通用评估使用 MMLU、CSQA、GSM8K、MATH。

**📈 对比分析**

与全参数微调、随机稀疏微调以及 PCGrad 等基线对比。CAST‑Safe‑Zone 在保持安全性能的前提下，通用能力提升明显（例如 Llama MMLU 提升约 9%），且在 Pareto 前沿上占优，证明其在安全-通用权衡上更高效。

**⚠️ 局限性**

局限性包括：①仅诊断查询投影（W_q），未覆盖 MLP 等大量参数；②使用静态预对齐冲突图，未考虑对齐过程中的冲突漂移；③诊断依赖特定校准数据集，可能不适用于高度专业化任务。

---

## Dialect Matters: Cross-Lingual ASR Transfer for Low-Resource Indic Language Varieties

**arXiv ID:** 2601.04373 | [PDF](https://arxiv.org/pdf/2601.04373v1)

**作者:** Akriti Dhasmana `[一作]` (University of Notre Dame), David Chiang `[通讯]` (University of Notre Dame)

**通讯引用:** 6668 | **OpenAlex IDs:** https://openalex.org/A5036026526

**关键词:** `Computation and Language` `Recognition` `Domain Adaptation` `Supervised Fine-Tuning` `Audio`

### 📋 论文摘要

**🎯 论文内容**

本文通过对德干文（Devanagari）书写的印地语方言和语言变体进行大规模实验，评估跨语言迁移学习在噪声、口语和代码混杂语音上的有效性，并对Garhwali（Pahari族群中的一种低资源方言）进行深入的ASR案例研究。

**💡 创新点**

创新点在于：①系统地比较了在方言/非标准语言与高资源标准语言之间进行微调的效果，发现方言数据微调往往能超过基于亲缘关系高资源语言的微调；②提出了一种量化模型对预训练语言偏倚（尤其是对印地语的偏倚）的框架；③首次对Garhwali进行完整的ASR模型评估与错误分析，揭示了拼写不规范和代码混杂带来的挑战。

**🔧 技术方法**

技术方面采用了多种自监督语音模型（Wav2Vec 2.0、HuBERT、XLS-R、Whisper、w2vBERT）进行微调，并利用词误差率（WER）和字符误差率（CER）进行评估；同时引入了正交的子树内迁移实验、层次距离与误差率的相关分析、以及基于词典的偏倚评估。

**📊 数据集**

数据集主要是VAANI（约15万小时语音，约10%转录），包含773个地区的方言口语，重点关注使用Devanagari书写的30种语言变体；Garhwali子集用于专门的案例研究。

**📈 对比分析**

比较方法是：①在不同语言的1–7小时微调后在全部方言/标准语音上进行零样本评估；②对比使用方言数据与使用标准语言数据的微调效果；③在不同子族群内进行对照实验。结果显示：在方言评估集上，使用方言微调能获得相当甚至更好的WER；总体上，WER在20–80%之间，Garhwali最佳模型WER约49.3%。

**⚠️ 局限性**

限制包括：①VAANI各方言数据量不均，影响跨语言比较；②仅覆盖Devanagari脚本，无法推广到其他印度语言脚本；③实验仅覆盖现有的自监督ASR架构，其他模型或目标函数可能表现不同；④噪声、口语失误与录音环境多变，进一步影响模型泛化。

---

## A Semi-supervised Molecular Learning Framework for Activity Cliff Estimation

**arXiv ID:** 2601.04507 | [PDF](https://arxiv.org/pdf/2601.04507v1)

**作者:** Fang Wu `[一作]` (Stanford University), Fang Wu `[通讯]` (Stanford University)

**通讯引用:** 4311 | **OpenAlex IDs:** https://openalex.org/A5040225125

**关键词:** `Computational Engineering, Finance, and Science` `Drug Discovery` `Graph Neural Network` `Graph`

### 📋 论文摘要

**🎯 论文内容**

提出一种名为SemiMol的半监督学习框架，用教师模型评估伪标签可靠性，并采用自适应课程学习提升在低样本条件下的活性阶梯预测精度。

**💡 创新点**

创新点在于引入独立的教师模型为回归任务提供置信度评估，以及通过自适应阈值动态选择伪标签，避免传统固定阈值导致的误标积累和过拟合。

**🔧 技术方法**

核心技术包括Graph Neural Networks（如GIN、GMT）作为目标模型，教师模型用于二分类置信度预测，伪标签生成、损失函数结合以及自适应课程学习策略。

**📊 数据集**

使用MoleculeACE公开平台的30个活性阶梯数据集（共约35,000分子），并在其中12个低样本（≤1K）数据集上进行实验。

**📈 对比分析**

与基线（如传统GNN、预训练GNN、π-model、Semi-GAN、UPS）对比，SemiMol在30个活性阶梯数据集上平均降低RMSE 26.53%，显著优于SOTA预训练方法和其他半监督方法。

**⚠️ 局限性**

局限在于仍需手工设置伪标签更新频率k以及初始阈值γ，且对极少量标签的任务效果可能受限，未来需探索更自动化的伪标签筛选机制。

---

## The Overlooked Role of Graded Relevance Thresholds in Multilingual Dense Retrieval

**arXiv ID:** 2601.04395 | [PDF](https://arxiv.org/pdf/2601.04395v1)

**作者:** Tomer Wullach `[一作]` (OriginAI), Amir DN Cohen `[通讯]` (OriginAI)

**关键词:** `Information Retrieval` `Retrieval` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

利用LLM生成的多语言合成数据，系统性研究了在密集检索模型的对比学习 fine‑tune 过程中，将分级相关性评分转换为二元正负样本所采用阈值对检索效果的影响，涵盖单语、混合语种以及跨语种检索三种情景。

**💡 创新点**

提出阈值校准应视为多语言 dense 检索 fine‑tune 的核心方法论环节，并展示分级相关性在多语言检索中的实质价值；通过实验证明不同语言与任务需要不同阈值，阈值不宜统一，且合理阈值可在更少数据下获得更好性能。

**🔧 技术方法**

采用对比学习（contrastive loss）训练 multilingual‑e5‑large dense 检索模型，使用 GPT‑4 对查询与候选段落进行自动生成与分级相关性评分，并将分级评分按阈值转化为正负样本进行 fine‑tune。

**📊 数据集**

构造了基于 MIRACL 语料的六语种（芬兰语、阿拉伯语、日语、俄语、西班牙语、英语）合成 fine‑tune 集（约 300K 条样本），并在 MIRACL 开发集（单语、混合语种）以及 CLIRMatrix MULTI‑8（跨语种）上评测。

**📈 对比分析**

在不同阈值（1、2、3）和训练样本量（50K–300K）的设置下，通过 nDCG@10 对模型进行比较；实验显示：低资源语言更倾向于低阈值以保持正样本多样性，高资源语言更适合高阈值；合理阈值能在较少训练数据下提升性能，并降低对标注噪声的敏感度。

**⚠️ 局限性**

局限性包括：阈值选择依赖语言和任务特性，缺乏统一自动校准方法；阈值会改变正负样本比例，可能导致类别不平衡；候选检索阶段的模型偏差可能影响训练数据质量；实验仅使用 LLM 注释，未验证在人类标注环境下的表现。

---

## SciFig: Towards Automating Scientific Figure Generation

**arXiv ID:** 2601.04390 | [PDF](https://arxiv.org/pdf/2601.04390v1)

**作者:** Siyuan Huang `[一作]` (Johns Hopkins University), Shraman Pramanick `[通讯]` (Johns Hopkins University)

**通讯引用:** 551 | **OpenAlex IDs:** https://openalex.org/A5067636415

**关键词:** `Artificial Intelligence` `Generation` `Large Language Model` `Chain-of-Thought` `Diffusion model` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了 SciFig，一个端到端的多智能体 AI 系统，能够根据科研论文的方法描述自动生成出版级的实验流程图。

**💡 创新点**

创新点在于：①引入层次化布局生成策略，将方法划分为模块与组件；②采用多轮链式思维（CoT）反馈机制实现迭代改进；③构建基于真实论文图的六维 rubrics 评估框架；④四个专用智能体协同完成解析、布局、反馈和组件渲染。

**🔧 技术方法**

技术手段包括：大型语言模型（LLM）驱动的描述、布局、反馈和组件智能体；层次化结构解析与模块级连线；多轮 CoT 视觉分析与自然语言指导；生成器（如 diffusion‑style 或图形渲染）完成组件绘制；自动化评估代理通过文本/图像对齐评分。

**📊 数据集**

使用的数据集为 2,219 条来自 2023 年顶级 AI 会议（CVPR、NeurIPS、ICLR 等）的实验流程图及其对应的论文描述，测试集中 30 篇论文，涵盖 35 个科研领域。

**📈 对比分析**

与单一 LLM（GPT‑5‑Image、Gemini‑2.5‑Flash 等）、通用图像生成模型（Stable Diffusion V1.5/XL、Qwen‑Image）以及专用布局系统 Paper2Poster 对比，SciFig 在六维评估中平均得分 70.1（数据集级）和 66.2（论文级），高于最佳基线 65.7，展示了显著的整体提升，尤其在结构连贯性、视觉清晰度和技术准确度方面表现突出。

**⚠️ 局限性**

局限性包括：①对层次化布局和 CoT 反馈的稳定性高度依赖于多智能体协同，单一智能体或平面布局的效果不佳；②目前仅针对方法流程图，尚未覆盖实验结果图、统计图等其他图表类型；③评估框架仍基于人工标注的六维 rubrics，跨领域迁移可能需要重新调优；④对极端复杂或极短文本的解析能力尚待验证。

---

## Rate or Fate? RLV$^\varepsilon$R: Reinforcement Learning with Verifiable Noisy Rewards

**arXiv ID:** 2601.04411 | [PDF](https://arxiv.org/pdf/2601.04411v1)

**作者:** Ali Rad `[一作]` (Cognichip), Ehsan Kamalinejad `[通讯]` (Cognichip)

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

探讨在强化学习可验证奖励(RLVR)中噪声回馈对大语言模型训练的影响，并给出一个可解析的多臂赌博机框架来预测学习动力学与相位转变；

**💡 创新点**

提出以Youden指数J=TPR−FPR为核心的单标量阈值，揭示了学习阶段的相位转变（J>0学习，J=0中性，J<0反学习）以及噪声仅缩放收敛速度而不改变最终结果的定性结论；

**🔧 技术方法**

使用GRPO（Group Relative Policy Optimization）以及其群归一化、优势标准化、重要性采样与裁剪、KL正则等技术，并在理论上推导了其在均值场近似下的单变量ODE；

**📊 数据集**

在Python代码生成任务上使用来自OpenR1的约10,239条训练题目和594条验证题目，验证器为程序化单元测试；

**📈 对比分析**

与噪声无关的完美验证器（J=1）做基线，对不同J值的噪声配置进行实验，结果显示在J>0时模型持续提升，J=0时几乎无提升，J<0时准确率下降；实验表明噪声只能影响收敛速率，最终准确率与无噪声相同；

**⚠️ 局限性**

限制包括：只在Python编程任务和中等规模模型上验证，未考虑时间变化的噪声或验证器与模型共进化；阈值J对不同任务、模型容量可能有所不同；以及在J<0时即使加入KL正则仍会趋向坏模式，无法完全消除反学习。

---

## Computational Compliance for AI Regulation: Blueprint for a New Research Domain

**arXiv ID:** 2601.04474 | [PDF](https://arxiv.org/pdf/2601.04474v1)

**作者:** Bill Marino `[一作]` (University of Cambridge), Nicholas D. Lane `[通讯]` (University of Cambridge)

**通讯引用:** 16812 | **OpenAlex IDs:** https://openalex.org/A5045638679

**关键词:** `Artificial Intelligence` `Large Language Model` `Supervised Fine-Tuning` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出了针对人工智能监管合规的计算化框架（CAIRC），包括Inspector和Mechanic两大核心组件，并设计了相应的评估指标与基准数据集。

**💡 创新点**

创新点在于：①把监管合规转化为可编程、可循环的算法流程；②为Inspector与Mechanic提出可量化的设计目标；③构建了包含合规与非合规AI系统快照的基准集，为未来算法的实验和对比提供了统一标准。

**🔧 技术方法**

采用的技术包括：LLM与规则引擎用于Inspector的合规诊断；多种工具（如模型微调、差分隐私、机器无学习等）与调度算法用于Mechanic的自动修复；基准评测 harness 用于自动验证和性能监控。

**📊 数据集**

使用的数据集为“AI系统快照”集合，包含训练/评估数据、模型权重、代码、文档和日志，并为非合规样本手工标注了Inspector诊断结果。

**📈 对比分析**

比较方法：通过准确率、精确率、召回率、延迟等指标评估Inspector；通过修复成功率、部署可行性、循环次数与耗时评估Mechanic；通过完整闭环系统的合规达成率和循环次数评估整体框架。论文中未给出实验结果，而是提出了可供后续验证的评测流程。

**⚠️ 局限性**

限制包括：①监管合规的技术可行性与可测度仍存在开放问题；②合规判定可能具有主观性和灰度区间；③数据集的构建与标注成本高；④目前缺乏已验证的Inspector/Mechanic实现，框架仍处于理论与蓝图阶段。

---

## Phasor Agents: Oscillatory Graphs with Three-Factor Plasticity and Sleep-Staged Learning

**arXiv ID:** 2601.04362 | [PDF](https://arxiv.org/pdf/2601.04362v1)

**作者:** Rodja Trappe `[一作]` (Zauberzeug GmbH), Rodja Trappe `[通讯]` (Zauberzeug GmbH)

**关键词:** `Machine Learning` `Reinforcement Learning` `Graph Neural Network` `Reinforcement Learning` `Graph`

### 📋 论文摘要

**🎯 论文内容**

本文提出并实现了基于斯图尔特-劳德振荡器图的 Phasor Agents，利用本地三因素可塑性、资格迹与睡眠阶段（NREM/REM）分离的机制，实现了无反向传播的学习与记忆存储，并提供可复现的实验套件。

**💡 创新点**

创新点在于将相位编码的动力子系统与可解释的三因素学习相结合，设计了睡眠分阶段的同步性控制与重放策略，实现了稳定学习、记忆存储与规划的全局可解释性，并通过一系列可复现的实验验证每一机制的因果效应。

**🔧 技术方法**

采用了斯图尔特-劳德振荡器图、IMEX 数值积分、三因素可塑性（资格迹 + 全局调制 + 振荡门控）、相位干涉的内容可寻记忆核、压缩进度内在激励、睡眠分阶段（NREM 采样+RE M 重放）等技术。

**📊 数据集**

使用的任务包括自生成的 8×8 迷宫、简易关联记忆任务、Hopfield 类记忆基准、Dyna‑Q 等传统基准，主要通过模拟环境而非公开数据集进行评估。

**📈 对比分析**

与现代 Hopfield 网络、ESN、Dyna‑Q 等基准对比：相位记忆在 256 节点时可达约 0.13 N 的容量，REM 重放可将迷宫成功率提升 45.5 个百分点；NREM/REM 分离显著提升稳定性并在受限权重范数下提升 67% 的性能；但与 Dyna‑Q 的显式经验回放仍存在明显性能差距。

**⚠️ 局限性**

主要限制包括：规模受限（N≤256）、仅评估离散动作与低维观测，未涉及连续控制、POMDP 或物理实验；对参数的敏感性缺乏系统分析；与传统深度 RL 的性能对比仍远不具竞争力；缺乏大规模任务与更丰富基准的验证。

---

## Causally-Aware Information Bottleneck for Domain Adaptation

**arXiv ID:** 2601.04361 | [PDF](https://arxiv.org/pdf/2601.04361v1)

**作者:** Mohammad Ali Javidian `[一作]` (Appalachian State University), Mohammad Ali Javidian `[通讯]` (Appalachian State University)

**通讯引用:** 92 | **OpenAlex IDs:** https://openalex.org/A5011786801

**关键词:** `Machine Learning` `Domain Adaptation` `Gaussian Splatting` `Variational Information Bottleneck` `Causal Information Bottleneck` `Graph`

### 📋 论文摘要

**🎯 论文内容**

研究在目标变量缺失的域适应任务，提出利用因果结构中目标的马尔可夫毯作为限制，使用信息瓶颈来学习可迁移的低维表示，从而实现目标变量的零样本插补。

**💡 创新点**

创新点在于将信息瓶颈与因果图结合，限定编码器仅观察目标的马尔可夫毯，得到闭式线性 Gaussian IB（可直接用 CCA 计算）以及非线性 VIB 版本，并提供在马尔可夫毯不变假设下的零样本迁移理论保证。

**🔧 技术方法**

技术手段包括信息瓶颈（Gaussian IB/CCA、变分信息瓶颈 VIB）、因果信息瓶颈（Causal IB）、马尔可夫毯约束、正交化特征提取、KL 损失控制与重参数化技巧。

**📊 数据集**

实验数据集包括：7 节点模拟因果结构（用于验证理论与对照实验）、64 节点 MAGIC–IRRI 基因网络（高维 Gaussian 贝叶斯网络）以及 Sachs 等人单细胞信号网络（真实生物实验数据）。

**📈 对比分析**

与贝叶斯网络、Invariant Information Bottleneck、纯全连接深度网络等基线比较，MB–GIB/MB–VIB 在 covariate/target shift 下表现最优，RMSE/MAE 明显下降，R² 通常提升 0.3–0.5，尤其在高维、强扰动场景中优于所有对照方法。

**⚠️ 局限性**

局限性包括：需要准确获取目标的马尔可夫毯，若马尔可夫毯不完整或 p(T|X_M) 发生漂移则零样本保证失效；支持分布偏移过大时仍会出现外推误差；VIB 需要调参且对数据量敏感；模型对隐藏共因子或未观测父变量不稳健。

---

## AgentTutor: Empowering Personalized Learning with Multi-Turn Interactive Teaching in Intelligent Education Systems

**arXiv ID:** 2601.04219 | [PDF](https://arxiv.org/pdf/2601.04219v1)

**作者:** Yuxin Liu `[一作]` (Shanghai Jiao Tong University), Jie Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 26452 | **OpenAlex IDs:** https://openalex.org/A5100428255

**关键词:** `Computers and Society` `Transformer` `Large Language Model` `Agentic AI` `Multimodality` `Text`

### 📋 论文摘要

**🎯 论文内容**

设计并实现了一种名为AgentTutor的多轮交互式LLM驱动智能教育系统，支持学习者持续评估、动态教学策略调整和个性化学习体验。

**💡 创新点**

创新点在于将多模态LLM生成的多代理系统与学习者个性化档案环境相结合，并通过树搜索（LATS）实现动态策略生成，克服单轮问答缺陷，提升教学互动质量。

**🔧 技术方法**

采用的技术包括：GPT‑4 / GPT‑3.5‑turbo 等LLM、Qwen2.5‑VL‑7B‑Instruct 进行多模态处理、LangChain 框架、ChromaDB 向量数据库、LATS 树搜索算法、UCT 以及反思技术（RAP、Reflexion）。

**📊 数据集**

实验使用的公开编程任务数据集为 HumanEval（164 题）和 MBPP（974 题）。

**📈 对比分析**

与 CoT、ReAct、ToT、RAP、Reflexion 等基线方法对比，AgentTutor 在 HumanEval 上 Pass@1 从 68.1% 提升至 96.9%，在 MBPP 上从 71.4% 提升至 89.4%；交互教学质量评估和人工评估均优于基线，体现出更高的教学效果和交互水平。

**⚠️ 局限性**

局限性包括：评估指标对多轮交互的支持不足、个性化因素范围狭窄、缺乏大规模真实学习者验证、以及跨模态教学内容生成方法仍待完善。

---

## PackCache: A Training-Free Acceleration Method for Unified Autoregressive Video Generation via Compact KV-Cache

**arXiv ID:** 2601.04359 | [PDF](https://arxiv.org/pdf/2601.04359v1)

**作者:** Kunyang Li `[一作]` (Institute of Artificial Intelligence University of Central Florida), Yuzhang Shang `[通讯]` (Institute of Artificial Intelligence University of Central Florida)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Computational Efficiency` `Transformer` `Video`

### 📋 论文摘要

**🎯 论文内容**

提出一种无训练的KV缓存管理方法PackCache，用于统一自回归视频生成模型，以提升推理效率和长序列生成能力。

**💡 创新点**

创新点在于利用文本与条件图像的语义锚定、跨帧注意衰减以及空间保持位置嵌入，实现基于时空相关性的动态压缩缓存。

**🔧 技术方法**

采用Transformer自回归架构、MM‑RoPE三维位置编码、指数衰减分配策略以及对缓存进行按需裁剪的机制。

**📊 数据集**

使用I2V子集（VBench I2V）中的160张图像与Qwen‑32B生成的字幕作为实验数据集。

**📈 对比分析**

与完整KV缓存和滑动窗口基线比较，PackCache在A40/H200 GPU上对24帧和48帧视频分别实现1.3‑2.2×加速，48帧最后四帧可达3.7×加速，同时保持或提升视觉质量。

**⚠️ 局限性**

局限在于仍需手动设定缓存窗口和最小配额，且在极长序列或高分辨率下可能无法完全消除显存峰值；对不同模型的泛化能力尚待验证。

---

## Integrating Multi-Agent Simulation, Behavioral Forensics, and Trust-Aware Machine Learning for Adaptive Insider Threat Detection

**arXiv ID:** 2601.04243 | [PDF](https://arxiv.org/pdf/2601.04243v1)

**作者:** Firdous Kausar `[一作]` (Meharry Medical), Mohamed Zakaria Kurdi `[通讯]` (Meharry Medical)

**关键词:** `Cryptography and Security` `Anomaly Detection` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出一种集成多智能体仿真、行为与通信取证、信任感知机器学习以及 Theory‑of‑Mind 推理的混合式内部威胁检测框架，并在仿真环境中评估其效果。

**💡 创新点**

创新点在于将 ToM 推理和通信取证作为认知上下文注入 SIEM，结合证据门控验证实现高精度低噪声检测，并通过预训练的 Enron 邮件取证模块进一步提升及时性。

**🔧 技术方法**

使用 Mesa 框架搭建多智能体仿真；TomAbd 进行 ToM 推理；SIEM 包含策略层、EWMA 基线、信任适配阈值、在线逻辑回归与孤立森林；邮件取证采用 NLP 以及预训练的随机森林欺诈/AI 文本模型。

**📊 数据集**

数据集包括 Enron 邮件（422k 篇）和 Enron 垃圾邮件（33k 篇）以及仿真生成的 8,000 篇恶意合成邮件，用于构建写作基线、训练取证模型及评估。

**📈 对比分析**

与传统 LSC 基线相比，CE‑SIEM 在召回率提升至 1.0、F1 为 0.774；EG‑SIEM 在 0.922 的 F1、0.997 的警报精度与每跑 0.2 FP；EG‑SIEM‑Enron 进一步实现 1.0 的警报精度、0.933 的 F1、平均检测时间 10.26 步，显示显著的性能提升。

**⚠️ 局限性**

局限在于仿真规模有限，缺乏真实企业数据验证；对 ToM 与取证特征在实际 SOC 中的可解释性与可操作性尚未充分评估；且模型在多样化攻击场景与更大规模网络中的泛化性待验证。

---

## Differential Locally Injective Grid Deformation and Optimization

**arXiv ID:** 2601.04494 | [PDF](https://arxiv.org/pdf/2601.04494v1)

**作者:** Julian Knodt `[一作]` (POSTECH), Seung-Hwan Baek `[通讯]` (POSTECH)

**关键词:** `Graphics` `Optimization` `Compression` `Mesh` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于差分表示和顶点着色的无逆转网格变形方法，能够自适应压缩空间并保持局部单射；

**💡 创新点**

创新点在于将网格顶点表示为邻域的凸组合，并通过顶点着色将大规模稠密线性系统拆分为多个可并行优化的独立子系统；

**🔧 技术方法**

核心技术包括差分网格/网格表示、图着色与交替优化、软正则化（softplus）权重、边界约束、插值条形能量防止逆转；

**📊 数据集**

实验数据集涵盖二维网格、UV参数化、图像压缩（如中国塔、乔治·华盛顿雕像等图像）、以及通过dmtet实现的等值面提取；

**📈 对比分析**

与传统基于线搜索的全局优化和直接位置优化相比，方法在保持单射的同时实现了更快的收敛、更低的最终损失和更平滑的网格变形；

**⚠️ 局限性**

局限性包括：仍需依赖插值条形能量防止逆转，导致内存/计算开销；需要多轮交替优化，色数越多效率越低；边界只能沿着已知曲线滑动，无法自由变形；分割导致的细薄三角形（sliver）可能影响后续几何处理。

---

## Predictive Controlled Music

**arXiv ID:** 2601.04221 | [PDF](https://arxiv.org/pdf/2601.04221v1)

**作者:** Midhun T. Augustine `[一作]` (University of Alberta), Midhun T. Augustine `[通讯]` (University of Alberta)

**关键词:** `Sound` `Generation` `Optimization` `Recurrent Neural Network` `Audio` `Sequential`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于模型预测控制（MPC）与循环神经网络（RNN）的预测控制音乐（PCM）生成框架，自动化产生乐谱并生成音频；

**💡 创新点**

首次将控制理论的MPC原理引入音乐生成，实现了把音乐生成规则作为约束，并通过评价函数优化音乐质量；

**🔧 技术方法**

使用RNN建模音符时序，Feed‑forward NN评估乐谱评分，MPC求解最优控制序列，ADSR包络合成音频；

**📊 数据集**

构造了100个手工打分的乐谱数据集（包括经典曲目与随机序列），以及训练RNN的100条MIDI音符与力度样本；

**📈 对比分析**

通过训练误差（MSE≈0.0067）和RNN误差（MSE≈0.0173）验证模型；在20步预测示例中成功生成20个音符的单音轨与伴随鼓组的混音，未与其他算法进行直接对比；

**⚠️ 局限性**

受限于打分主观性、数据集规模有限、计算量随预测长度增加而急剧上升、缺乏与现有音乐生成方法的系统性性能对比；

---

## Social Engineering Attacks: A Systemisation of Knowledge on People Against Humans

**arXiv ID:** 2601.04215 | [PDF](https://arxiv.org/pdf/2601.04215v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Cryptography and Security`

---

## TSSR: Two-Stage Swap-Reward-Driven Reinforcement Learning for Character-Level SMILES Generation

**arXiv ID:** 2601.04521 | [PDF](https://arxiv.org/pdf/2601.04521v1)

**作者:** Jacob Ede Levine `[一作]` (California State Polytechnic University), Sai Chandra Kosaraju `[通讯]` (California State Polytechnic University)

**通讯引用:** 245 | **OpenAlex IDs:** https://openalex.org/A5010258542

**关键词:** `Machine Learning` `Generation` `Reinforcement Learning` `Drug Discovery` `Recurrent Neural Network` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了基于两阶段换位奖励（TSSR）的强化学习框架，用于从头生成可解析且化学可行的SMILES分子。

**💡 创新点**

创新点在于：①将语法错误与化学错误分离为两阶段可解释的奖励；②利用局部token交换而非全局生成修复，提高奖励稠密度；③无需手工语法或标签，直接以token频率与RDKit诊断为依据。

**🔧 技术方法**

使用了GRU基序列模型与Proximal Policy Optimization (PPO)强化学习，配合token‑级语法修复与化学错误修正算法，构成TSSR奖励函数。

**📊 数据集**

数据集为MOSES基准数据集（约190万条SMILES），用于训练与评估。

**📈 对比分析**

通过与传统单一终端奖励方法对比，分为从零开始（P‑RL）和预训练微调（F‑RL）两种设置；实验显示P‑RL在语法有效率、化学有效率和新颖性上提升显著（如syntactic validity从6%提升到35%，novelty从18%提升到59%），而F‑RL在保持药物相似性、可合成性方面收敛更快，效果虽略逊但依然优于未调优模型。

**⚠️ 局限性**

局限包括：对短分子偏好导致长度偏移；词表局限性仅覆盖MOSES常用原子与键；奖励设计仍可能过度优化解析性而忽略其它药物属性；训练规模有限，缺乏多环境并行与更长迭代；对其他分子表示（SELFIES、图模型）未验证。

---

## TeleTables: A Benchmark for Large Language Models in Telecom Table Interpretation

**arXiv ID:** 2601.04202 | [PDF](https://arxiv.org/pdf/2601.04202v1)

**作者:** Anas Ezzakri `[一作]` (Huawei Technologies), Haozhe Zhang `[通讯]` (Huawei Technologies)

**通讯引用:** 372 | **OpenAlex IDs:** https://openalex.org/A5100710300

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Chain-of-Thought` `Vision Language Model` `Tabular` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了 TeleTables 基准，用于评估大型语言模型在 3GPP 电信标准表格解读与推理的能力。

**💡 创新点**

创新点在于构建多阶段自动化 pipeline，提取 3GPP 表格并生成 500 条多难度 MCQ，系统考察不同表格格式、模型规模与推理能力对性能的影响。

**🔧 技术方法**

采用多模态 LLM（如 Qwen2.5-VL-72B、Qwen3）进行表格提取、格式转换、问题生成与验证，并使用链式推理技术评估模型；在 RAG 与无表格输入两种场景下对模型进行对比。

**📊 数据集**

使用了来自 13 篇 3GPP 规范（Release 18/19）的 2,220 张表格，转换为 HTML/JSON/Markdown/PNG 后生成 500 题 MCQ。

**📈 对比分析**

通过 pass@1 与 cons@16 指标对 9 类开源 LLM 进行评估，发现 120 B 级推理模型在全部数据上可达 91%+，但小模型和无推理模型仅 30–60%；在提供表格的 RAG 设置下 HTML 表示最优，表格复杂度越大准确率下降。

**⚠️ 局限性**

局限在于常规预训练模型对 3GPP 表格知识缺乏，尤其是小模型；视觉编码器在解析细粒度数字/结构时受限；仅在提供表格时才能获得较好结果，实际检索与表格结构保持仍需进一步研究。

---

## WESR: Scaling and Evaluating Word-level Event-Speech Recognition

**arXiv ID:** 2601.04508 | [PDF](https://arxiv.org/pdf/2601.04508v1)

**作者:** Chenchen Yang `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Fudan University)

**通讯引用:** 17130 | **OpenAlex IDs:** https://openalex.org/A5044665993

**关键词:** `Computation and Language` `Recognition` `Transformer` `Supervised Fine-Tuning` `Large Language Model` `Audio` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出词级事件语音识别（WESR）任务，构建了大规模词级非语音事件标注语料WESR-Train，并设计专家评测集WESR-Bench及其位置感知评估协议；在此基础上对Whisper、Kimi-Audio、Qwen3-Omni等大模型进行微调，得到一套强基准模型。

**💡 创新点**

①制定21类非语音事件的细粒度词级标签体系并区分离散/连续事件；②提出解耦ASR错误与事件定位的评估协议；③构建1,700+小时高质量词级标注语料和900+样本专家评测集；④通过Gemini自动注释与检索融合实现大规模弱标注；⑤在同一框架下兼顾连续与离散事件的定位与检测。

**🔧 技术方法**

使用Whisper、Kimi-Audio、Qwen3-Omni等多模态大模型进行监督微调；采用Gemini API自动生成词级事件标签；利用BEATs、AF‑CLAP进行检索、MossFormer2降噪；实现事件保持对齐、词/词间位置映射与精确评估算法。

**📊 数据集**

WESR-Bench（900+句、21类事件，双语）、WESR-Train（1,767小时、英语+中文，来自NonverbalTTS、NVSpeech‑170k、NonVerbalSpeech‑38K、SMIIP‑NV以及Gemini自动标注），以及公开的Common Voice 15 用于 ASR 评测。

**📈 对比分析**

对比开放源代码 ALM（Kimi‑Audio、Qwen3‑Omni）和商业 API（Gemini、GPT）在 WESR-Bench 上的 F1、Precision/Recall；微调模型在连续/离散事件上均显著优于基线，宏 F1 提升至 38%；在 Common Voice 上保持 ASR WER 仅略升高，证明 WESR 兼顾了事件识别与语音识别。

**⚠️ 局限性**

仅验证了英语和中文两种语言；构建过程高度依赖 Gemini API 与人工专家，成本较高且在低资源环境下难以复现；自动标注误差与样本稀缺导致部分细粒度标签性能仍有限。

---

## Shadow Unlearning: A Neuro-Semantic Approach to Fidelity-Preserving Faceless Forgetting in LLMs

**arXiv ID:** 2601.04275 | [PDF](https://arxiv.org/pdf/2601.04275v1)

**作者:** Dinesh Srivasthav P `[一作]` (TCS Research), Ponnurangam Kumaraguru `[通讯]` (IIIT Hyderabad)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Computational Efficiency` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出一种在匿名忘记集上进行机器学习模型无痕忘记的新范式——Shadow Unlearning，并实现了Neuro‑Semantic Projector Unlearning（NSPU）方法；

**💡 创新点**

核心创新在于：①在保证匿名化的前提下通过激活空间的神经语义桥接映射，实现对匿名数据的语义对齐；②利用主成分子空间与线性滤波器在推理阶段实时抑制忘记相关概念；③设计逆向隐私惩罚项提升安全性；

**🔧 技术方法**

使用了轻量化多层感知机做激活映射、PCA构建忘记子空间、线性投影滤波、以及预训练LLM的冻结参数；

**📊 数据集**

构建了多域合成忘记数据集MuFU（涵盖数字信息学、科学技术、体育、金融、政治五大领域），并采用公开语料训练映射器；

**📈 对比分析**

与梯度上升、梯度差、KL最小化、DPO、NPO等基线对比，NSPU在HPS、HRS、HCNLL、CES等综合指标上均表现优于基线；在计算效率上比重训练快10⁶倍、比传统无痕方法快10倍；在隐私评估中，分离质量分（SQS）显著提升；

**⚠️ 局限性**

仅适用于开源LLM，依赖忘记与保留数据分布的重叠度；映射器训练需要合适的公开语料；评估基于合成作者资料，真实场景验证仍需进一步研究。

---

## Generative Teaching via Code

**arXiv ID:** 2601.04204 | [PDF](https://arxiv.org/pdf/2601.04204v1)

**作者:** Yuheng Wang `[一作]` (Shanghai Jiao Tong University), Chen Qian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 50873 | **OpenAlex IDs:** https://openalex.org/A5100428454

**关键词:** `Computers and Society` `Generation` `Data Synthesis` `Large Language Model` `Agentic AI` `Video` `Text` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出 Generative Teaching 思想并实现 TeachMaster 框架，通过多代理协同生成可编辑、可交付的教学视频与脚本，帮助教师从手工制作转向高层次策划。

**💡 创新点**

创新点包括：① 以代码为中间语义媒介，实现可解释、可编辑的多模态生成；② 将传统像素级黑盒生成拆分为可控的代码驱动流程；③ 采用多代理流水线（规划、设计、渲染、评估）实现教学内容全流程自动化。

**🔧 技术方法**

技术手段：大语言模型（LLM）用于文本与代码生成；代码生成映射至 Manim 动画脚本；语音合成 TTS；同步、调试、布局检查模块；Chain‑of‑Agents、ReAct 等多代理协作框架；跨模态对齐与评估工具。

**📊 数据集**

数据集：① 高评价 YouTube 教育视频及其官方字幕；② Sora‑2 生成的视频（对照基线）；③ GPT‑5 评测用语料。通过这些数据训练与评测，以保证实验可重复与对比。

**📈 对比分析**

比较方法：对照 Sora‑2（端到端生成模型）与人类制作的视频/脚本，采用视觉清晰度、视觉丰富度、逻辑性、文本‑图像对应、事实准确性等指标（1–10 评分）。结果显示 TeachMaster 在所有指标上均优于 Sora‑2，接近甚至超过人类参考；生产效率提升显著，平均生成 1 分钟视频仅需约 3 分钟，低于人类约 12 分钟与 Sora‑2 5+ 分钟。

**⚠️ 局限性**

局限性：① 系统偏重概念性讲解，应用案例与实验练习不足；② 难度层级自适应功能尚未完善；③ 受限于训练数据范围，可能对极端专业或最新热点主题表现不佳；④ 需要进一步改进教师交互与可视化调节体验。

---

## Transformer-Based Multi-Modal Temporal Embeddings for Explainable Metabolic Phenotyping in Type 1 Diabetes

**arXiv ID:** 2601.04299 | [PDF](https://arxiv.org/pdf/2601.04299v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## Assessing the quality and coherence of word embeddings after SCM-based intersectional bias mitigation

**arXiv ID:** 2601.04393 | [PDF](https://arxiv.org/pdf/2601.04393v1)

**作者:** Eren Kocadag `[一作]`, Ali Mohammed Mansoor Alsahag `[通讯]`

**关键词:** `Artificial Intelligence` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究了如何在静态词向量中利用Stereotype Content Model（SCM）对交叉身份（性别+种族）进行偏见缓解，并评估其对词嵌入语义连贯性和类比性能的影响。

**💡 创新点**

提出了两种组合方式（求和与拼接）构造交叉身份向量，并比较三种基于SCM的去偏方法（Subtraction、Linear Projection、Partial Projection），从而填补了单一身份去偏方法无法处理交叉偏见的空白。

**🔧 技术方法**

采用SCM建立的温暖-能力子空间，对嵌入向量进行线性投影或减法去偏，使用Word2Vec、GloVe和ConceptNet Numberbatch三种主流静态嵌入模型，并通过Embedding Coherence Test (ECT) 和Embedding Quality Test (EQT) 对去偏效果进行评估。

**📊 数据集**

使用公开预训练的Word2Vec、GloVe和ConceptNet Numberbatch嵌入，以及WikiText‑103文本进行轻度适配，配合从前人工作中提取的SCM对立词对和性别/种族词表。

**📈 对比分析**

在三种去偏策略和两种组合方式下进行多次实验，发现所有模型在ECT上都保持高分（≈0.99），去偏方法越保守（Partial Projection）EQT越低；相比之下，Linear Projection和Subtraction在保持类比性能方面更具优势，且不同模型对聚合方式的敏感度各异。

**⚠️ 局限性**

主要局限包括：SCM轴向可能受文化差异影响，自动生成的交叉词对缺乏语义手工校准，未评估去偏对实际下游任务的直接影响，以及ECT/EQT指标无法完整捕捉所有偏见维度。

---

## Scalable Floating-Point Satisfiability via Staged Optimization

**arXiv ID:** 2601.04492 | [PDF](https://arxiv.org/pdf/2601.04492v1)

**作者:** Yuanzhuo Zhang `[一作]` (Virginia Tech), Binoy Ravindran `[通讯]` (Virginia Tech)

**通讯引用:** 3924 | **OpenAlex IDs:** https://openalex.org/A5067528153

**关键词:** `Programming Languages` `Optimization`

### 📋 论文摘要

**🎯 论文内容**

提出了一种三阶段的数值优化框架StageSAT，用于求解量化无符号浮点数（QF_FP）SAT问题，最终得到位级精确解；

**💡 创新点**

创新点在于将浮点SAT转化为逐步细化的优化问题，先用平滑的平方误差与正交投影快速导向可行域，再转为二次ULP距离进行精细搜索，最后在浮点格点上做n‑ULP离散细化，保证零目标即为合法模型；

**🔧 技术方法**

技术包括：投影辅助的平方距离目标、基于IEEE‑754的ULP²代表函数、离散的n‑ULP格点搜索、乘法-求和的子句聚合、全局无约束优化器以及多启动策略；

**📊 数据集**

数据集涵盖SMT‑COMP’25 QF_FP基准（Middle、Large）、MathSAT、Grater、JFS等共计约480份文件；

**📈 对比分析**

与四类比对：完整的位级SMT求解器（Z3、CVC5、MathSAT、Bitwuzla）以及三种主流优化式求解器（XSat、goSAT、Grater、JFS）。StageSAT在Large集上达到92% SAT召回率、0%误SAT、平均5–10倍加速，并在其余集上实现0%超时；与完整求解器相比，速度提升数倍；与优化式求解器相比，覆盖率提高、误判率降至0；

**⚠️ 局限性**

局限性：仍为不完整求解器，无法给出正式UNSAT证明；对极大规模或极难实例仍可能超时；最终精度依赖于离散细化步长，极端边缘情况仍可能遗漏；

---

## Towards Spatio-Temporal Extrapolation of Phase-Field Simulations with Convolution-Only Neural Networks

**arXiv ID:** 2601.04510 | [PDF](https://arxiv.org/pdf/2601.04510v1)

**作者:** Christophe Bonneville `[一作]` (Sandia National Laboratories), Cosmin Safta `[通讯]` (Sandia National Laboratories)

**通讯引用:** 1320 | **OpenAlex IDs:** https://openalex.org/A5070944838

**关键词:** `Computational Engineering, Finance, and Science` `Generation` `Data Synthesis` `Computational Efficiency` `Convolutional Neural Network` `Diffusion model` `Image` `Physics Related`

### 📋 论文摘要

**🎯 论文内容**

本文设计并训练了一个完全卷积、可参数化的 U‑Net 代理，用以对液态金属去合金化的相场模拟进行时间和空间的超越性预测，并通过扩散模型生成初始条件，显著提升模拟速度。

**💡 创新点**

创新点在于将卷积自注意力、物理约束填充、洪水填充校正以及条件化通道缩放融入 U‑Net，实现对不同合金成分、时间步长的自适应预测，并通过扩散模型消除昂贵的数值求解初始化。

**🔧 技术方法**

使用的技术包括全卷积 U‑Net、卷积自注意力机制、周期性和零填充、洪水填充修正、条件化向量映射、以及基于 DDPM 的扩散生成模型；训练与推理均在 PyTorch 上完成。

**📊 数据集**

数据集由 5 种不同 A 元素浓度（0.20–0.40）下的相场模拟快照构成，总计约 6500 个训练样本；扩散模型训练使用 1000 张 256×256 的去合金化前后场图像。

**📈 对比分析**

在训练范围内，代理对质量、曲率、界面周长等量子误差均低于 5%，在延展到三倍时间尺度时误差不超过 20%，速度提升可达 36,000 倍以上；相对数值求解器的速度提升约为 23,700–47,300 倍。

**⚠️ 局限性**

局限性包括：对高浓度 A（0.4）时预测精度下降；自回归推理过程中误差累积导致长期预测的不确定性；生成初始条件虽可减少成本，但其物理一致性仍需进一步验证。

---

## Categorical Belief Propagation: Sheaf-Theoretic Inference via Descent and Holonomy

**arXiv ID:** 2601.04456 | [PDF](https://arxiv.org/pdf/2601.04456v1)

**作者:** Enrique ter Horst `[一作]` (Universidad de los Andes), Juan Diego Zambrano `[通讯]`

**关键词:** `Artificial Intelligence` `Graph Neural Network` `Graph`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于同伦编译的推理框架，利用图的循环同伦信息将最难的 loopy 推理问题拆分为若干扇区，在每个扇区内进行精确或近似的树形推理。

**💡 创新点**

创新点在于：① 把图循环的同伦作用解释为对状态纤维的平移，并用轨道划分实现扇区分解；② 通过模式编译（mode quotient）把非树形结构转化为树形结构，保证推理精确性；③ 给出了全局一致性与 BP 失效的可验证理论关联。

**🔧 技术方法**

使用了同伦理论、图论（生成树、基循环）、布尔矩阵乘法、强连通分量（SCC）分割、树形消息传递算法（TreeBP）、离散同步与 X‑OR 约束等技术。

**📊 数据集**

主要数据集为合成离散同步图（ℤ_k 同步）、Permutation/monomial 因子图（XOR‑SAT/Parity‑Check）、以及相对简单的随机生成的离散图模型。

**📈 对比分析**

与传统 loopy BP（并行/阻尼）、Exact（变量消除/Variable Elimination）以及 MCMC（Gibbs）对比；在循环复杂度低或同伦轨道少的场景下，HATCC 与原始 BP 速度相当但结果更稳定、后验准确度明显提升；在循环同伦显著时，HATCC 能恢复推理并显著减少 BP 震荡。

**⚠️ 局限性**

局限性包括：① 只能处理循环同伦可分解且扇区数不爆炸的图，无法解决高树宽或连续变量模型；② 对同伦操作的离散化依赖于纤维空间有限；③ 对于连续或高阶约束的推广仍需进一步研究。

---

## SCAR-GS: Spatial Context Attention for Residuals in Progressive Gaussian Splatting

**arXiv ID:** 2601.04348 | [PDF](https://arxiv.org/pdf/2601.04348v1)

**作者:** Diego Revilla `[一作]` (University of Deusto), Ooi Wei Tsang `[通讯]` (National University of Singapore)

**通讯引用:** 2771 | **OpenAlex IDs:** https://openalex.org/A5072587271

**关键词:** `Computer Vision and Pattern Recognition` `Compression` `Recurrent Neural Network` `Gaussian Splatting` `Auto Encoder` `Point Cloud`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于残差向量量化与空间自回归熵模型的 3D 高斯抛光渐进式压缩与渲染框架 SCAR-GS。

**💡 创新点**

创新点包括：1) 用残差向量量化（RVQ）替代传统标量量化，实现特征多层细化；2) 结合多分辨率哈希网格的空间查询注意机制，增强自回归熵模型的上下文感知；3) 采用旋转梯度技巧提升 VQ 训练稳定性；4) 通过三阶段课程学习缓解硬量化导致的崩溃。

**🔧 技术方法**

核心技术：VQ‑VAE、残差向量量化、GRU+空间查询注意、自回归熵模型、算术编码、旋转梯度、课程学习。

**📊 数据集**

实验数据集：NeRF Synthetic、Tanks & Temples、MipNeRF360、Deep Blending、BungeeNeRF 以及各类 3DGS 场景（Bonsai、Flowers、Stump、Room、Amsterdam、Bilbao、Hollywood、Pompidou、Quebec 等）。

**📈 对比分析**

与 PCGS、GoDe 等现有渐进式压缩方法对比，SCAR‑GS 在多层级别上实现平滑的感知质量提升，SSIM/LPIPS/PSNR 与对手相当或略优，尤其在大规模场景下显示更好的存储效率和可接受的渲染速度，尽管基层比特率略高。

**⚠️ 局限性**

局限性：基层容量有限，在极稀疏几何或极低基层比特率场景下细化收敛慢；基层文件体积高于标量量化方案；对极端稀疏场景的适应性不足；未来需更高效的网络流式框架与更紧凑的 RVQ 训练目标。

---

## Leveraging Language Models and RAG for Efficient Knowledge Discovery in Clinical Environments

**arXiv ID:** 2601.04209 | [PDF](https://arxiv.org/pdf/2601.04209v1)

**作者:** Seokhwan Ko `[一作]` (Kyungpook National University), Junghwan Cho `[通讯]` (Kyungpook National University)

**通讯引用:** 963 | **OpenAlex IDs:** https://openalex.org/A5038559731

**关键词:** `Computation and Language` `Recommendation System` `Retrieval` `Transformer` `Retrieval-Augmented Generation` `Prompt Engineering` `Large Language Model` `Text` `Biomedical Data`

### 📋 论文摘要

**🎯 论文内容**

开发并评估了在医院本地网络下的检索增强生成（RAG）系统，用于根据本机构 PubMed 发表的论文推荐研究合作伙伴。

**💡 创新点**

创新点在于将领域特定的 PubMedBERT 嵌入与轻量级本地部署的 LLaMA3.2 结合，实现了在严苛隐私安全约束下的知识检索与生成，提升协作推荐的语义相关性与可解释性。

**🔧 技术方法**

使用了 PubMedBERT 进行语义嵌入，基于余弦相似度的近似最近邻检索，检索增强提示构造，以及 LLaMA3.2 的本地生成推理。

**📊 数据集**

使用了本机构医学院作者的 PubMed 出版记录（包括标题、摘要、作者列表、关键词等）作为数据集。

**📈 对比分析**

与传统关键词检索相比，PubMedBERT 嵌入显著提升检索相关性；生成摘要准确率高，系统能将精确匹配文献排在首位，cosine 相似度最高达 0.9964，说明检索与生成效果良好。

**⚠️ 局限性**

局限性包括仅局限单机构数据，未实现实时监控新文献；模型规模受硬件限制，跨机构知识图构建和与资助管道集成尚未实现。

---

## Advancing Language Models for Code-related Tasks

**arXiv ID:** 2601.04526 | [PDF](https://arxiv.org/pdf/2601.04526v1)

**作者:** Zhao Tian `[一作]` (Tianjin University), Zhao Tian `[通讯]` (Tianjin University)

**通讯引用:** 42474 | **OpenAlex IDs:** https://openalex.org/A5085086661

**关键词:** `Software Engineering` `AI Code Assistant` `Generation` `Transformer` `Large Language Model` `Prompt Engineering` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文系统地提升了代码语言模型的表现，分别在代码数据质量、模型架构和推理能力三大方向上引入了 CODA、CodeDenoise、LEAM/LEAM++、μFiX 和 Specine 等技术，推动代码生成、测试等软件工程任务的实用性。

**💡 创新点**

创新点包括：①利用代码差异引导的对抗增强（CODA）和代码去噪（CodeDenoise）提升数据质量；②将代码表示为抽象语法树并采用语法引导的编码解码架构（LEAM/LEAM++）保证合成代码语法正确；③将思维诱导提示与反馈提示相结合的 μFiX，以及通过需求对齐提升的 agent 方案 Specine，显著增强模型推理与误解纠正。

**🔧 技术方法**

技术手段涵盖：对抗增强与去噪流程、AST 基础的编码解码网络、语法规则预测、思维诱导与反馈提示的融合、需求感知与对齐的 agent 框架。

**📊 数据集**

实验使用公开的大规模开源代码语料库（包含多种编程语言），并在代码生成、变异代码生成、代码补全等标准任务上进行验证。

**📈 对比分析**

与 ALERT、CARROT 等数据增强基线比较，CODA 在鲁棒性上提升 28.86%/24.04%；与 CodeLlama、Qwen-Coder 等通用模型对比，LEAM/LEAM++ 在变异代码生成任务中实现 100% 语法正确率；与现有提示和 agent 基线相比，μFiX 和 Specine 分别在 Pass@1 上提升 35.62% 和 29.60%。

**⚠️ 局限性**

局限性在于目前方法主要验证于低复杂度任务，尚未充分展示在大规模、真实世界软件开发场景中的效果，仍需进一步提升对复杂任务的适应性。

---

## Online Action-Stacking Improves Reinforcement Learning Performance for Air Traffic Control

**arXiv ID:** 2601.04287 | [PDF](https://arxiv.org/pdf/2601.04287v1)

**作者:** Ben Carvell `[一作]` (NATS), Richard Everson `[通讯]` (Alan Turing Institute)

**通讯引用:** 9468 | **OpenAlex IDs:** https://openalex.org/A5016900158

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning`

### 📋 论文摘要

**🎯 论文内容**

研究了在线动作堆叠（action-stacking）技术，用于强化学习政策在空中交通管制中的指令生成，并训练和评估多种场景下的导航、冲突回避和垂直控制。

**💡 创新点**

提出在线动作堆叠在推理时将连续的原子动作动态合并为宏指令，同时通过动作阻尼奖励诱导动作稀疏，从而在仅使用5个离散动作的训练空间中实现与37维动作空间相同的性能。

**🔧 技术方法**

使用PPO强化学习、蓝鸟DT数字孪生模拟平台、BADA飞机动力学、基于奖励的动作阻尼和安全惩罚。

**📊 数据集**

在Project Bluebird的X-Plus Sector人工模拟空域中随机生成单机或双机场景，飞机为波音757-300，采用BADA模型模拟。

**📈 对比分析**

将在线动作堆叠+动作阻尼的5维动作空间与全规模37维动作空间和无阻尼对比，通过每个情景的动作次数、成功率和最小间隔来评估；堆叠策略将动作数平均降至约7次，成功率高于大动作空间，并实现与大动作空间相近的安全与导航表现。

**⚠️ 局限性**

局限在高流量、多机交互与复杂安全约束的扩展性尚未验证，且对更大动作空间的收敛速度仍较慢，需进一步提升训练效率与泛化能力。

---

## Machine Learning Model for Sparse PCM Completion

**arXiv ID:** 2601.04366 | [PDF](https://arxiv.org/pdf/2601.04366v1)

**作者:** Selcuk Koyuncu `[一作]` (Coppin State University), Stephen Providence `[通讯]` (Coppin State University)

**通讯引用:** 22 | **OpenAlex IDs:** https://openalex.org/A5050888625

**关键词:** `Machine Learning` `Recommendation System` `Graph Neural Network` `Graph`

### 📋 论文摘要

**🎯 论文内容**

提出了一种基于图神经网络的稀疏对比矩阵补全模型，结合三角一致性正则化实现乘法一致性，并输出全局排序；

**💡 创新点**

将PCM的对数比值建模与GNN融合，首次在log空间加入三角一致性损失，同时采用稀疏采样、wedge采样等技术实现大规模可扩展；

**🔧 技术方法**

使用GNN信息传递、BTL/LLS损失、三角一致性正则化、几何均值投影、稀疏张量乘法与mini‑batch训练；

**📊 数据集**

主要使用合成稀疏PCM数据集（Erdős–Rényi图、链图等）及不同稀疏度的随机对比集合；

**📈 对比分析**

与传统LLS基准比较，评估RMSE、Kendall τ与训练时间；在大多数稀疏度下，ML模型与LLS在准确性上相近（τ≈0.96–0.98），但训练时间显著更长；在大规模（n≥10^5）下，利用稀疏实现仍能在秒级完成；

**⚠️ 局限性**

缺点是训练成本高于LLS，尤其在极稀疏或极大规模时需要更多显存；需要手动调参；在p<1%时准确率下降；缺乏真实世界数据验证。

---

## Predictable Gradient Manifolds in Deep Learning: Temporal Path-Length and Intrinsic Rank as a Complexity Regime

**arXiv ID:** 2601.04270 | [PDF](https://arxiv.org/pdf/2601.04270v1)

**作者:** Anherutowa Calvo `[一作]`, Anherutowa Calvo `[通讯]`

**关键词:** `Machine Learning` `Optimization` `Convolutional Neural Network` `Transformer` `Tabular` `Sequential`

### 📋 论文摘要

**🎯 论文内容**

提出基于梯度可预测性和低秩增量的复杂度指标，并证明在线凸优化与平滑非凸优化的误差与这些指标相关，随后在多种网络和优化器上验证其有效性。

**💡 创新点**

创新点在于引入可测的“预测路径长度”P_T(m)和“可预测秩”r⋆(ε)作为优化难度的度量，摆脱传统的时间长度T和参数维度d，揭示梯度轨迹的局部可预测性与低秩结构。

**🔧 技术方法**

主要技术包括：基于历史的预测器、预测路径长度与归一化可预测性指数的定义、增量矩阵的SVD与可预测秩定义、优化理论（optimistic mirror descent与平滑非凸更新）与这些指标的紧耦合分析。

**📊 数据集**

使用的数据集包括CIFAR‑100（ResNet‑18、ViT‑Tiny）、synthetic序列（Tiny Transformer）、tabular数据（3‑层MLP）以及WikiText‑2（GPT‑2），并在多种优化器（SGD‑momentum、AdamW、RMSprop）上进行实验。

**📈 对比分析**

与传统按T、d标度的理论相比，实验表明在简单预测器下κ_T(m)≈1、r⋆(ε)很小，导致收敛率与误差随可预测性显著缩短；实验结果在不同模型与优化器上均保持可预测性不变，显示该指标在多任务中的稳健性。

**⚠️ 局限性**

局限性包括：可预测性并非普遍存在，低秩特性依赖于梯度记录方式（完整或投影）、预处理及训练阶段；指标需手动计算，难以实时应用；且未证明可预测性提升可直接转化为显著的算法加速。

---

## Comparative Analysis of Custom CNN Architectures versus Pre-trained Models and Transfer Learning: A Study on Five Bangladesh Datasets

**arXiv ID:** 2601.04352 | [PDF](https://arxiv.org/pdf/2601.04352v1)

**作者:** Ibrahim Tanvir `[一作]` (University of Dhaka), Sartaj Solaiman `[通讯]` (University of Dhaka)

**通讯引用:** 2 | **OpenAlex IDs:** https://openalex.org/A5093910592

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Convolutional Neural Network` `Image`

### 📋 论文摘要

**🎯 论文内容**

比较了从头训练的轻量级CNN与预训练ResNet-18/VGG-16在五个孟加拉国图像分类数据集上的性能。

**💡 创新点**

首次系统评估了不同迁移学习策略（特征提取与微调）与自定义网络在同一任务下的效果，并给出了针对资源受限与高精度场景的实用建议。

**🔧 技术方法**

采用CNN结构、ResNet-18、VGG-16预训练模型、Adam优化器、交叉熵损失及多种评估指标（准确率、F1、训练时间、模型大小）。

**📊 数据集**

使用了五个本土数据集：Footpath Vision、Auto Rickshaw、Mango Image BD、Paddy Variety BD、Road Damage BD。

**📈 对比分析**

通过在相同训练条件下比较自定义网络、特征提取和微调三种方法，发现微调往往能提升 3–76% 的准确率，特征提取在训练时间与参数量上更高效；自定义网络在模型尺寸与推理速度上占优。

**⚠️ 局限性**

局限包括仅评估 ResNet-18 与 VGG-16，未探究更高效的现代网络；未做超参数优化；缺乏推理速度、能耗和多模型集成等部署方面的评估。

---

## CRUNet-MR-Univ: A Foundation Model for Diverse Cardiac MRI Reconstruction

**arXiv ID:** 2601.04428 | [PDF](https://arxiv.org/pdf/2601.04428v1)

**作者:** Donghang Lyu `[一作]` (Leiden University Medical Center), Mariya Doneva `[通讯]` (Philips Innovative Technologies)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Convolutional Neural Network` `Recurrent Neural Network` `Prompt Engineering` `Image` `Biomedical Data` `Magnetic Resonance Imaging`

### 📋 论文摘要

**🎯 论文内容**

提出了一种名为CRUNet-MR-Univ的基础模型，用于在多中心、多扫描仪、多加速因子和多采样模式下重建心脏MRI图像。

**💡 创新点**

创新点包括将双向卷积递归单元嵌入U‑Net形成CRUNet，加入跨级特征聚合（CFA）模块，使用文本和可学习提示（prompt）对图像先验进行编码，以及采用课程学习策略逐步提升加速因子和块数。

**🔧 技术方法**

主要技术包括卷积递归网络、U‑Net结构、提示学习（text-based + learnable prompts）、FiLM调制、CFA聚合、混合精度训练、AdamW优化器和多项损失（L1、MSE、SSIM、分类交叉熵）。

**📊 数据集**

使用CMRxRecon2025数据集，该数据集来自5个医学中心、10台不同厂商（GE、Philips、Siemens、United Imaging）的1.5T/3.0T扫描仪，涵盖多种序列、对比、疾病类型、采样模式与加速因子。

**📈 对比分析**

与传统SENSE、GRAPPA、Zero‑Filled以及两种基线深度模型（CRNN‑MRI、UNet‑MR）比较，在验证集中心裁切区域上，CRUNet‑MR‑Univ在S2阶段实现PSNR 28.23 dB、SSIM 0.809、NMSE 0.04，明显优于其他方法。

**⚠️ 局限性**

局限性包括：在最高加速因子下仍存在细部模糊；缺乏k‑space频率信息的损失函数；通道数固定为64，可能限制空间特征表达；卷积操作在高加速因子下受限于感受野；训练周期与样本分布仍可进一步优化。

---

## SmoothSync: Dual-Stream Diffusion Transformers for Jitter-Robust Beat-Synchronized Gesture Generation from Quantized Audio

**arXiv ID:** 2601.04236 | [PDF](https://arxiv.org/pdf/2601.04236v1)

**作者:** Yujiao Jiang `[一作]` (Shenzhen International Graduate School Tsinghua University), Zongqing Lu `[通讯]` (Shenzhen International Graduate School Tsinghua University)

**关键词:** `Sound` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Audio` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

研究基于双流扩散变换器的共语手势生成

**💡 创新点**

提出双流架构将音频与动作分离处理并通过跨模态注意力融合、引入抖动抑制损失、使用量化梅尔能量提升多样性，并提出更鲁棒的 Smooth‑BC 评估指标

**🔧 技术方法**

利用量化梅尔特征、Dual‑Stream Diffusion Transformer（DiT）、SMPLX rot6d 表示、跨模态注意力、抖动抑制损失、DDPM/DDIM 推理等技术

**📊 数据集**

在 BEAT2 与 SHOW 两个公开数据集上进行实验

**📈 对比分析**

与多种基线（S2G、MambaTalk、EMAGE 等）对比，BEAT2 上 FGD 下降至 4.15，Smooth‑BC 提升至 7.78，交叉多样性 14.00，SHOW 上 FGD 4.996、Smooth‑BC 2.749、交叉多样性 3.959，整体超越现有最优方法

**⚠️ 局限性**

仍受 SMPLX 表示精细手部动作的限制，模型规模大、训练成本高，缺乏对多语言/跨文化语境的泛化评估

---

## BanglaLorica: Design and Evaluation of a Robust Watermarking Algorithm for Large Language Models in Bangla Text Generation

**arXiv ID:** 2601.04534 | [PDF](https://arxiv.org/pdf/2601.04534v1)

**作者:** Amit Bin Tariqul `[一作]` (Islamic University of Technology), Md Kamrul Hasan `[通讯]` (Islamic University of Technology)

**通讯引用:** 3157 | **OpenAlex IDs:** https://openalex.org/A5100656463

**关键词:** `Computation and Language` `Generation` `Adversarial Attack` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

评估了 Bangla LLM 文本生成的水印鲁棒性，比较单层与双层水印在跨语言 RTT 攻击下的表现。

**💡 创新点**

首次系统化研究低资源语言下的水印鲁棒性，并提出结合嵌入时与后生成的双层水印方法。

**🔧 技术方法**

采用 KGW、Exponential Sampling、Waterfall 等水印技术，并将其层叠组合。

**📊 数据集**

使用 Bangla‑Alpaca Orca 过滤后的 500 句提示与 Bangla‑NMT 进行的 Bangla→English→Bangla RTT 流程。

**📈 对比分析**

通过检测准确率、ROUGE、困惑度等指标对比，单层 RTT 下准确率降至 9–13%，双层提升至 40–50%，相较单层提升 3–4 倍。

**⚠️ 局限性**

仅测试单一路径 RTT、短文本长度、缺乏人类评估及更广泛语言或更长文本的鲁棒性验证。

---

## Beyond Interaction Effects: Two Logics for Studying Population Inequalities

**arXiv ID:** 2601.04223 | [PDF](https://arxiv.org/pdf/2601.04223v1)

**作者:** Adel Daoud `[一作]` (Linköping University), Adel Daoud `[通讯]` (Linköping University)

**通讯引用:** 1293 | **OpenAlex IDs:** https://openalex.org/A5090745602

**关键词:** `Computers and Society` `Causal Forest` `Meta Learning`

### 📋 论文摘要

**🎯 论文内容**

本文比较并框架化了传统交互回归与因果机器学习（如因果森林、元学习）在估计教育回报效应异质性时的适用性，并通过仿真演示何时优先采用哪种方法。

**💡 创新点**

创新点在于提出演绎与归纳两种逻辑的两阶段结合策略，并给出基于仿真结果的实用指导，说明在不同异质性结构下哪种方法更优。

**🔧 技术方法**

主要使用线性交互回归、因果森林（causal forest）以及元学习（meta‑learner）等因果机器学习技术，配合仿真评估其MSE、偏差与方差。

**📊 数据集**

研究主要基于自建的仿真数据集，模拟学生收入、考试成绩、种族、性别等特征；未使用公开实测数据集。

**📈 对比分析**

通过计算均方误差、偏差与方差对比，发现在线性交互情形下两者表现相近，非线性或高阶交互时因果森林显著优于OLS；当不存在异质性时，OLS略占优势。

**⚠️ 局限性**

限制包括：结果受可观测协变量选择的限制；因果森林虽然灵活但解释性差，难以揭示具体交互结构；仿真设定可能无法完全映射真实社会数据的复杂性。

---

## Using Large Language Models to Detect Socially Shared Regulation of Collaborative Learning

**arXiv ID:** 2601.04458 | [PDF](https://arxiv.org/pdf/2601.04458v1)

**作者:** Jiayi Zhang `[一作]` (University of Pennsylvania), Gautam Biswas `[通讯]` (Vanderbilt University)

**通讯引用:** 12884 | **OpenAlex IDs:** https://openalex.org/A5051150754

**关键词:** `Machine Learning` `Classification` `Transformer` `Large Language Model` `Text` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

通过将学生对话与系统日志对齐，使用大语言模型生成任务感知摘要，并基于文本嵌入、上下文嵌入和日志特征训练模型，自动检测协作计算建模环境中的社会共享学习调节（SSRL）行为。

**💡 创新点**

首次将大语言模型用于生成任务相关摘要以增强多模态特征，并系统比较文本、上下文、日志与融合特征在SSRL检测中的表现，展示了嵌入模型在协作STEM+C情境下的可扩展性。

**🔧 技术方法**

采用 GPT‑3.5 Turbo 生成摘要，BERT/其他预训练模型生成文本嵌入，提取系统日志的序列和 n‑gram 特征，使用多层感知机进行二分类，并在嵌入、上下文和日志特征之间进行多模态融合。

**📊 数据集**

使用 Vanderbilt 大学的 C2STEM 计算建模实验数据，包含 36 名高中生的 189 小时对话、5,575 句转录、4,893 次系统动作、394 段摘要，且覆盖初始化、更新、条件等任务上下文。

**📈 对比分析**

对五种输入配置（文本仅、文本+上下文、日志仅、文本+日志、文本+日志+上下文）进行嵌套交叉验证，评估 ROC AUC。平均 AUC：文本仅 0.65，文本+上下文 0.64，日志仅 0.53，文本+日志 0.61，文本+日志+上下文 0.62；文本嵌入在多数 SSRL 类别上表现最好，日志特征在反思等非语言行为上具有补充优势。

**⚠️ 局限性**

模型对不同构造的敏感度差异大，整体准确率仍有限；实验仅在单一任务与单一环境中验证，缺乏跨任务、跨学科的泛化；多模态标签在教师可解释性上存在挑战；未引入更丰富的多模态信息（如眼动、手势），限制了进一步提升性能。

---

## A Systematic Mapping Study on the Debugging of Autonomous Driving Systems

**arXiv ID:** 2601.04293 | [PDF](https://arxiv.org/pdf/2601.04293v1)

**作者:** Nathan Shaw `[一作]` (University of Sheffield), Donghwan Shin `[通讯]` (University of Sheffield)

**通讯引用:** 1028 | **OpenAlex IDs:** https://openalex.org/A5019085537

**关键词:** `Software Engineering` `Autonomous Driving` `Multimodality` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

本研究对自主驾驶系统（ADS）的调试技术进行系统映射研究，梳理了研究类型、问题、技术与工具，提出统一的分类体系并识别研究空白。

**💡 创新点**

创新点在于首次聚焦ADS调试而非测试，系统构建了调试问题（简化、定位、解释、修复）与技术（手工分析、因果/断言分析、故障注入/变异、迭代最小化、ML调试）之间的对应关系，并提供标准化术语与工具清单，为后续研究奠定框架。

**🔧 技术方法**

采用系统映射研究（SMS）方法，使用 OpenAlex 数据库进行检索与筛选，结合 Snowballing、手工判断与冲突解决；技术层面归纳了上述调试技术及其在文献中的实现方式。

**📊 数据集**

使用 15 篇经筛选的原始研究论文，其中涵盖了多种数据来源（仿真场景、真实记录、公开数据集）以及公开的 ADS 仿真器和工具；不依赖单一大规模数据集。

**📈 对比分析**

通过对 15 篇论文的技术归类与应用层面对比，评估了不同调试技术在功能覆盖、适用阶段（开发 vs 现场）以及工具可获取性；发现大多数研究集中于开发阶段，仿真与仿真环境占主导，实测验证与性能评估仍有限。

**⚠️ 局限性**

局限性包括：研究样本规模有限、缺乏工业级真实部署案例、对后期现场调试的关注不足、工具实现多为原型或未公开，导致可复现性和通用性不足；此外，ADS 的多模态与实时交互特性使得传统调试指标难以直接迁移，需要进一步探索适应性评估方法。

---

## Meta-probabilistic Modeling

**arXiv ID:** 2601.04462 | [PDF](https://arxiv.org/pdf/2601.04462v1)

**作者:** Kevin Zhang `[一作]`, Yixin Wang `[通讯]`

**关键词:** `Machine Learning` `Meta Learning` `Explainability and Interpretability` `Auto Encoder` `Image` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出 Meta-Probabilistic Modeling（MPM）框架，利用多相关数据集学习共享的生成模型结构，并通过层次化全局与局部参数实现可解释的潜在变量推断。

**💡 创新点**

创新点在于将传统 PGM 与元学习相结合，使用 VAE 风格的代理目标实现可解析推断，同时把 Slot Attention 视作 MPM 的特例，并通过全局与局部参数分离实现跨数据集的结构迁移。

**🔧 技术方法**

核心技术包括层次贝叶斯建模、变分推断与 VAE 代理潜能、双层（inner/outer）优化、神经网络参数化的距离函数与认知网络，以及可解析的局部坐标上升更新。

**📊 数据集**

实验使用了 Tetrominoes 图像数据集（对象分割）和 AP News 语料（文本主题）作为测试数据集。

**📈 对比分析**

与 Slot Attention、传统 GMM、LDA 等基线比较；在对象聚类任务中 Additive MPM 获得 97.8% ARI，明显优于 Slot Attention（≈84%）和 GMM（≈77%）；在文本主题任务中 MPM 的 log‑perplexity 为 14.15，略优于 LDA（14.94）。

**⚠️ 局限性**

局限性包括：需要多相关数据集才能训练；代理目标对模型假设较敏感；训练过程中可能出现不稳定（如 Slot Attention 的高方差表现）；标准差较大且未在更复杂或高维序列数据上进行验证。

---

## In-SRAM Radiant Foam Rendering on a Graph Processor

**arXiv ID:** 2601.04382 | [PDF](https://arxiv.org/pdf/2601.04382v1)

**作者:** Zulkhuu Tuya `[一作]` (Imperial College London), Andrew J. Davison `[通讯]` (Imperial College London)

**通讯引用:** 32669 | **OpenAlex IDs:** https://openalex.org/A5039230558

**关键词:** `Graphics` `Neural Radiance Field` `Image`

### 📋 论文摘要

**🎯 论文内容**

实现了在 Graphcore IPU 上完全基于片上 SRAM 的 Radiant Foam 场景渲染器，完全消除了对全局显存的依赖；

**💡 创新点**

创新点在于采用分区+静态四叉树路由覆盖、压缩光线状态以及屏幕空间分布式帧缓冲，将光线追踪工作均匀分布到每个 Tile，并保持通信可预测；

**🔧 技术方法**

使用 Radiant Foam 的 Voronoi 体积表示、k‑d 树分区、四叉树路由层、混合/半精度光线负载压缩、IPU Poplar 编程模型；

**📊 数据集**

在 Mip‑NeRF 360 与 Deep Blending 两大体积渲染数据集上进行实验；

**📈 对比分析**

与原始 GPU Radiant Foam 对比，图像和深度质量 PSNR>55 dB、SSIM≈1，帧率约 1 fps（640×480），几乎与 GPU 质量一致；

**⚠️ 局限性**

局限在于舍弃高阶 SH、路由层拥塞、局部负载不平衡以及摄像机变动导致的光线同步问题。

---

## An ASP-based Solution to the Medical Appointment Scheduling Problem

**arXiv ID:** 2601.04274 | [PDF](https://arxiv.org/pdf/2601.04274v1)

**作者:** Alina Vozna `[一作]` (University of Pisa), Dawid Pado `[通讯]` (University of L’Aquila)

**关键词:** `Artificial Intelligence` `Optimization` `Tabular` `Biomedical Data` `Electronic Health Records`

### 📋 论文摘要

**🎯 论文内容**

提出基于答案集程序（ASP）的医疗预约调度框架，实现实时、高效、可解释的排程，兼顾患者偏好与资源约束。

**💡 创新点**

创新点在于将Blueprint Personas嵌入ASP模型，形成患者个性化与伦理决策的组合，并在同一框架下实现多目标优化与实时重调。

**🔧 技术方法**

使用ASP（Clingo）进行约束建模与多目标优化，并通过Python Flask微服务与MySQL数据库实现接口与数据管理。

**📊 数据集**

使用人工构造的情境数据集（涵盖高危、可达性、感官偏好等）进行实验验证，并未使用公开真实医疗数据。

**📈 对比分析**

与传统规则/启发式方法对比，实验显示在三种情境下求解时间均低于0.04 s，且得到满足所有硬约束与软目标的最优解。

**⚠️ 局限性**

局限在于未涉及大规模真实数据验证、患者经济约束、预测缺席率等，且目前为离线批处理模型，缺乏动态实时更新机制。

---

## Memory-Guided Unified Hardware Accelerator for Mixed-Precision Scientific Computing

**arXiv ID:** 2601.04476 | [PDF](https://arxiv.org/pdf/2601.04476v1)

**作者:** Chuanzhen Wang `[一作]` (Tongji University), Eric Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 382 | **OpenAlex IDs:** https://openalex.org/A5078551074

**关键词:** `Hardware Architecture` `Spiking Neural Network` `Image` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出了一种内存引导的统一硬件加速器，能够在单一平台上高效处理有限元、突触神经网络和稀疏张量的混合精度工作负载。

**💡 创新点**

核心创新包括内存引导的自适应精度选择、经验驱动的并行度动态调整以及课程学习式稀疏模式发现，突破了固定精度、位宽递增和手工稀疏配置的限制。

**🔧 技术方法**

采用混合精度算术、时空突触网络数据流、稀疏张量压缩与动态并行度、经验学习位宽管理、课程学习策略以及长短期内存查表等技术。

**📊 数据集**

使用了FEniCS、COMSOL、ANSYS（有限元）、MNIST、CIFAR-10/100、DVS‑Gesture、ImageNet-1K（SNN）、COCO 2017（稀疏张量）等多领域数据集进行评估。

**📈 对比分析**

与专用加速器（AMX‑bf16、FireFly v2、结构化稀疏张量单元）对比，在相同工作负载下实现约47%吞吐量提升、34%能耗降低、41% DSP效率提升，并在混合任务中获得45–65%吞吐量提升。

**⚠️ 局限性**

仍受限于网格条件数≤1000、SNN位宽≤8位、稀疏度≥10%等前提，无法充分支持极端高条件数、极低稀疏度或更大位宽的任务。

---

## Disco-RAG: Discourse-Aware Retrieval-Augmented Generation

**arXiv ID:** 2601.04377 | [PDF](https://arxiv.org/pdf/2601.04377v1)

**作者:** Dongqi Liu `[一作]` (Saarland University), Yabiao Wang `[通讯]` (Tencent YouTu Lab)

**关键词:** `Computation and Language` `Generation` `Retrieval` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了 Disco‑RAG 框架，将检索到的文档段落先解析为 Rhetorical Structure Theory（RST）树和跨段落关系图，并基于这些结构生成高层计划来指导 LLM 生成，从而提升知识密集型任务的效果。

**💡 创新点**

首次将话语层次结构显式注入检索增强生成（RAG）中，利用 intra‑chunk RST 树、inter‑chunk 关系图以及 discourse‑aware 规划模块，显著提升了信息整合和推理质量。

**🔧 技术方法**

使用 LLM 作为检索器/生成器；借助 LLM‑based RST 解析器构建树/图；设计基于结构的规划模块；在生成时把结构信息注入 prompt 以引导 LLM。

**📊 数据集**

在 Loong、ASQA、SciNews 三个公开的知识密集型问答与摘要基准上进行评估。

**📈 对比分析**

与零样本 LLM、标准 RAG 以及多种 SOTA RAG 方法对比，实验显示 Disco‑RAG 在 Loong、ASQA、SciNews 上均取得最高或接近最高分，显著优于基线。

**⚠️ 局限性**

主要限制包括：依赖 LLM 解析器导致额外推理开销；仅在英文长文 QA/摘要场景验证，对不同语言、话语形式或更小模型的适用性未知；需手工 prompt 与参数调优。

---

## Merging Triggers, Breaking Backdoors: Defensive Poisoning for Instruction-Tuned Language Models

**arXiv ID:** 2601.04448 | [PDF](https://arxiv.org/pdf/2601.04448v1)

**作者:** San Kim `[一作]` (POSTECH), Gary Geunbae Lee `[通讯]` (POSTECH)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了 MB-Defense 两阶段训练框架，通过防御性触发注入与权重恢复，消除指令微调 LLM 中的后门攻击。

**💡 创新点**

创新点在于将攻击者与防御者触发统一为单一后门表征，再通过额外训练拆解该表征，实现对未知多样后门的无先验信息防御。

**🔧 技术方法**

使用防御性触发注入、统一后门表征、权重恢复损失（交叉熵+正则化）、注意力头评估与 LLM‑as‑a‑judge 自动评估等技术。

**📊 数据集**

实验基于 Alpaca 指令微调数据集、WizardLM 测试集以及 128 条手工标注的清洁样本，测试了 Llama2‑7B、Qwen3‑8B 等模型。

**📈 对比分析**

与 Clean‑FFT、ONION、Fine‑mixing 等基线在 Toxic 与 Refusal 两种后门行为上比较，MB‑Defense 在保持 CACC 约 90‑95% 的同时将 ASR 降至 <0.04，整体表现优于所有基线。

**⚠️ 局限性**

局限性包括触发器设计相对简单、仅测试单一攻击者触发、未研究多重攻击者触发器的交互、以及对更丰富语义触发器的鲁棒性待验证。

---

## Automated Reproducibility Has a Problem Statement Problem

**arXiv ID:** 2601.04226 | [PDF](https://arxiv.org/pdf/2601.04226v1)

**作者:** Thijs Snelleman `[一作]` (RWTH Aachen University), Odd Erik Gundersen `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 1181 | **OpenAlex IDs:** https://openalex.org/A5022601333

**关键词:** `Computers and Society` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一个基于科学方法的通用可复现性问题表述，并用大型语言模型自动从20篇论文中提取假设、实验及结果解释，构建对应的图结构。

**💡 创新点**

创新点在于：①给出了可跨学科、可扩展的可复现性问题定义；②将可复现性转化为图结构，便于统一评估；③使用LLM无人工干预自动抽取结构化信息，验证了其可行性。

**🔧 技术方法**

技术手段主要是：少量示例提示（few‑shot prompting）与多轮查询相结合的Gemini 2.5 Pro LLM抽取；对抽取结果进行作者校正与Likert评分评估；对抽取错误进行定量分析。

**📊 数据集**

数据集为20篇人工标注的实验AI论文（PDF），并将抽取出的结构化信息保存为公开数据集，用于后续研究。

**📈 对比分析**

与作者自身评估相比，抽取的假设、实验描述和解释整体得到正面评价，平均正向Likert分数高于4（5分制），但实验结果与指标提取错误率高达69.6%。总体性能在结构完整性上表现良好，结果细节仍需改进。

**⚠️ 局限性**

局限性包括：①LLM上下文长度限制导致长文档信息缺失；②难以准确抽取图表中数值与可视化结果；③对假设、实验的抽象程度依赖文本表述清晰度；④缺乏对更大规模数据集的验证与模型微调。

---

## The Language of Bargaining: Linguistic Effects in LLM Negotiations

**arXiv ID:** 2601.04387 | [PDF](https://arxiv.org/pdf/2601.04387v1)

**作者:** Stuti Sinha `[一作]` (Birla Institute of Technology and Science Pilani), Dhruv Kumar `[通讯]` (Birla Institute of Technology and Science Pilani)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

在对抗式多轮谈判游戏中，系统地评估了语言框架对大型语言模型谈判行为的影响。

**💡 创新点**

首次将语言本身视为隐式策略先验，证明语言选择可比模型变化产生更显著的结果，揭示任务依赖的文化效应与种族刻板印象。

**🔧 技术方法**

采用 NegotiationArena 框架，使用 GPT‑4o、GPT‑3.5 Turbo、Claude‑3‑Haiku、Claude‑3.5‑Haiku 四大多语 LLM，并通过系统提示指定语言身份。

**📊 数据集**

在基准谈判游戏（Ultimatum、Buy‑Sell、Resource Exchange）中，对 5 种语言（英语、印地语、古吉拉特语、旁遮普语、马罗迪语）进行实验。

**📈 对比分析**

与英语基线相比，语言对接受率、提议初值、提议者/受让者优势和谈判轮次的影响超过模型差异；在分配游戏中语言导致提议者优势逆转，在整合游戏中提升交易量。

**⚠️ 局限性**

实验仅在模型对模型交互的人工设置下进行，缺乏真实人类参与、有限语言和游戏范围，且语言标记仅反映训练数据偏差而非真实文化行为。

---

## State Backdoor: Towards Stealthy Real-world Poisoning Attack on Vision-Language-Action Model in State Space

**arXiv ID:** 2601.04266 | [PDF](https://arxiv.org/pdf/2601.04266v1)

**作者:** Ji Guo `[一作]` (University of Electronic Science and Technology of China), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 80345 | **OpenAlex IDs:** https://openalex.org/A5091266202

**关键词:** `Cryptography and Security` `Adversarial Attack` `Robotic Intelligence` `Vision-Language-Action Model` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出了一种利用机器人臂初始状态触发的后门攻击方法

**💡 创新点**

首次将状态空间作为触发器，并采用偏好引导遗传算法（PGA）对触发状态进行高效、隐蔽的优化

**🔧 技术方法**

使用偏好引导遗传算法、对抗/后门训练、行为克隆损失与三目标联合优化

**📊 数据集**

在SO‑101 6‑DOF机械臂的五个真实任务数据集（Pick‑and‑Place、Drawer Opening、Button Pressing、Peg Insertion、Tennis Pushing）上进行实验

**📈 对比分析**

与 BadVLA、TrojanRobot 等基线对比，攻击成功率 (ASR) 超过 90%，并且对 fine‑pruning、图像压缩等常规防御无效

**⚠️ 局限性**

对模型的后续微调较为敏感，过度的毒化率会损害正常性能，且仅适用于依赖状态触发的任务

---

## Hybrid MKNF for Aeronautics Applications: Usage and Heuristics

**arXiv ID:** 2601.04273 | [PDF](https://arxiv.org/pdf/2601.04273v1)

**作者:** Arun Raveendran Nair Sheela `[一作]` (Université Clermont Auvergne), Victor Charpenay `[通讯]` (École des Mines de Saint-Étienne)

**关键词:** `Artificial Intelligence`

### 📋 论文摘要

**🎯 论文内容**

提出了在航空领域应用Hybrid MKNF知识表示与推理框架，并通过离线预处理实现低资源设备上的高效推理；

**💡 创新点**

创新点包括：1）针对航空需求设计的NOTAM‑Aware Reasoning System (NARS) 案例；2）在Hybrid MKNF中引入经典否定和完整性约束的语法与语义处理；3）通过知识库双化与半负变换实现一致性与不一致性答案的区分；4）提出离线预处理+运行时Prolog的分离策略降低内存占用；

**🔧 技术方法**

使用技术包括：Hybrid MKNF语言（well‑founded semantics）、SLG(O)查询、NoHr推理器、XSB‑Prolog、知识库双化、半负变换、经典否定与完整性约束的语法转换；

**📊 数据集**

实验使用了OWL2Bench（EL、QL、RL、DL）生成的本体以及NoHr自带的规则基准（10、100、1,000、10,000、25,000条规则），并采用LUBM查询集；

**📈 对比分析**

与现有Hy‑MKNF及NoHr的性能对比显示：预处理阶段时间随本体规模线性增长；查询时延在1秒以内（XSB-Prolog），但递归查询和含转移属性的查询性能下降；内存使用随KB大小显著提升，但离线预处理后运行时显著降低；

**⚠️ 局限性**

局限性包括：1）经典否定和完整性约束的语法转换仍需手工工程，难以推广；2）预处理过程对大规模本体仍高成本；3）对递归规则和复杂DL构造（等价、传递属性）支持不足，导致查询性能下降；4）目前仅在Well‑Founded semantics下验证，未覆盖Answer‑Set语义；5）缺乏正式的语义等价证明与可扩展性分析。

---

## IGA-LWP: An Iterative Gradient-based Adversarial Attack for Link Weight Prediction

**arXiv ID:** 2601.04259 | [PDF](https://arxiv.org/pdf/2601.04259v1)

**作者:** Cunlai Pu `[一作]` (Nanjing University of Science and Technology), Rajput Ramiz Sharafat `[通讯]` (University of Science and Technology of China)

**关键词:** `Social and Information Networks` `Adversarial Attack` `Optimization` `Graph Neural Network` `Auto Encoder` `Graph`

### 📋 论文摘要

**🎯 论文内容**

研究了加权网络中链路权重预测的对抗攻击问题，并提出了基于梯度的迭代攻击框架IGA-LWP。

**💡 创新点**

创新点在于把链路权重预测转化为约束优化问题，利用自注意力图自编码器做梯度引导，能够在有限扰动预算内显著破坏目标链路预测。

**🔧 技术方法**

采用自注意力增强图自编码器（SEA）做目标模型，梯度取值后逐步更新权重，结合邻接矩阵筛选关键链路；实验中对比随机攻击与基于共同邻居的攻击。

**📊 数据集**

在四个真实加权网络上实验：Neural-net、C. elegans、Netscience、UC-net。

**📈 对比分析**

与随机攻击（RDA）和相似度攻击（SA-CN）对比，IGA-LWP在全局与局部攻击下均能把RMSE提升至原来的1.5–3倍、PCC下降至负值，并对DeepWalk、Node2Vec、GCN等模型具备良好迁移性。

**⚠️ 局限性**

主要限制是攻击需要对目标模型（SEA）有白盒访问，且对极大图规模和非对称权重的适用性未验证，攻击成功与否高度依赖梯度信息。

---

## Integrating Distribution Matching into Semi-Supervised Contrastive Learning for Labeled and Unlabeled Data

**arXiv ID:** 2601.04518 | [PDF](https://arxiv.org/pdf/2601.04518v1)

**作者:** Shogo Nakayama `[一作]` (Doshisha University), Masahiro Okuda `[通讯]` (Doshisha University)

**通讯引用:** 4585 | **OpenAlex IDs:** https://openalex.org/A5025207272

**关键词:** `Artificial Intelligence` `Classification` `Contrastive Learning` `Image`

### 📋 论文摘要

**🎯 论文内容**

本文在半监督对比学习框架中加入分布匹配（MMD）机制，以更充分利用低置信度的无标签样本，实现对图像分类的提升。

**💡 创新点**

创新点在于将伪标签对比学习与基于MMD的分布匹配相结合，既能让所有无标签样本参与训练，又能通过分布对齐减少确认偏差。

**🔧 技术方法**

使用了弱/强数据增强、基于余弦相似度的伪标签生成、交叉熵与对比损失相结合的伪标签对比学习，以及MMD正则化损失。

**📊 数据集**

实验使用了CIFAR‑10、CIFAR‑100和STL‑10三个公开数据集，分别包含不同类别数与无标签样本。

**📈 对比分析**

与基准半监督对比学习方法相比，加入MMD后在大多数设置下（尤其是少量标签场景）准确率提升约2–3%，如CIFAR‑10 4/类标签时从0.7734提升到0.9059；当标签足够多时提升有限。

**⚠️ 局限性**

局限性包括：当标签样本充足时分布匹配效果不明显；早期训练阶段特征不稳定可能导致MMD正则误导学习；实验仅在图像分类任务上验证，泛化性尚待进一步探索。

---

## Attachment Styles and AI Chatbot Interactions Among College Students

**arXiv ID:** 2601.04217 | [PDF](https://arxiv.org/pdf/2601.04217v1)

**作者:** Ziqi Lin `[一作]` (New York University), Taiyu Hou `[通讯]` (New York University)

**关键词:** `Computers and Society` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

通过半结构化访谈和扎根理论分析，探讨不同依恋风格的大学生如何与ChatGPT互动。

**💡 创新点**

将依恋理论应用于人机互动，发现“依恋一致性AI使用”模式，并指出AI作为低风险情感空间的作用。

**🔧 技术方法**

采用定性研究方法，包括半结构化访谈、开放、轴向、选择性编码及扎根理论分析。

**📊 数据集**

7名大学生的访谈记录（含自我报告的依恋风格）。

**📈 对比分析**

由于为定性研究，没有量化比较；通过三主题分析阐释不同依恋风格的使用差异，未给出性能指标。

**⚠️ 局限性**

样本量小、单一高校、依恋评估为自报、仅研究ChatGPT，缺乏广泛适用性和量化验证。

---

## SAGE-32B: Agentic Reasoning via Iterative Distillation

**arXiv ID:** 2601.04237 | [PDF](https://arxiv.org/pdf/2601.04237v1)

**作者:** Basab Jha `[一作]` (SAGEA), Wang Junhao `[通讯]` (Fudan University)

**关键词:** `Artificial Intelligence` `Knowledge Distillation` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

SAGE-32B 是一种 32 亿参数的语言模型，专为代理式推理和长程规划任务设计；

**💡 创新点**

创新点包括：迭代蒸馏（Iterative Distillation, IDA）训练流程、逆向推理（Inverse Reasoning）机制和元认知头（Meta-Cognitive Head, MCH）实现错误预测与修正；

**🔧 技术方法**

采用的技术包括：Qwen2.5-32B 预训练、两阶段蒸馏与强化学习（PPO）、RMSNorm、SwiGLU、分离嵌入、Landmark Attention、Dual-Process Gradient Descent 与逆向一致性评分（ICS）等；

**📊 数据集**

训练数据来自 500 万条合成代理轨迹（含正例和三类难负例），以及 GPT‑4o/DeepSeek 生成的多步环境交互记录；

**📈 对比分析**

与 30‑70B 参数基准模型（Qwen2.5‑32B、Llama‑3‑70B）以及 GPT‑4‑Turbo 进行对比，在 MATH‑500、MMLU‑Pro、AgentBench 等代理式基准上取得显著提升，尤其在逆向推理模式下 MATH 分数从 78.9% 提升至 91.8%；

**⚠️ 局限性**

局限性包括：对开放式创意任务不适用、对工具文档不完整时易出现错误、在医疗、法律等专业领域表现欠佳、推理模式下延迟较高、需要清晰的工具规范支持。

---

## Qwerty AI: Explainable Automated Age Rating and Content Safety Assessment for Russian-Language Screenplays

**arXiv ID:** 2601.04211 | [PDF](https://arxiv.org/pdf/2601.04211v1)

**作者:** Nikita Zmanovskii `[一作]` (Independent Researcher), Nikita Zmanovskii `[通讯]` (Independent Researcher)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

开发了Qwerty AI，一款针对俄语剧本的端到端自动年龄分级与内容安全评估系统；

**💡 创新点**

创新点包括：将指令微调的Phi‑3‑mini LLM与法律规则相结合的混合检测管道、4‑bit量化与LoRA实现的极致效率、以及对俄罗斯联邦法第436‑FZ的可解释法律锚定；

**🔧 技术方法**

技术手段涵盖：Phi‑3‑mini‑4k‑instruct LLM、LoRA微调、4‑bit NormalFloat量化、CUDA加速、动态批处理、正则式法律词典、对比损失用于场景一致性；

**📊 数据集**

使用了约4200条手工标注的剧本场景数据（来自黑客松提供的剧本、公共领域文学、以及合成扩增），按70/15/15划分；

**📈 对比分析**

在15部完整剧本上进行评估，文档级别准确率80%（MAE 0.27），与随机、词典、零样本Phi‑3、仅微调Phi‑3等基线相比提升显著；

**⚠️ 局限性**

局限包括：对特定体裁语言误判、隐喻与上下文理解不足、18+类别样本稀缺导致召回率偏低、以及无法捕捉剧本中的多模态视觉描述等。

---

## AI Agents as Policymakers in Simulated Epidemics

**arXiv ID:** 2601.04245 | [PDF](https://arxiv.org/pdf/2601.04245v1)

**作者:** Goshi Aoki `[一作]` (Virginia Tech), Navid Ghaffarzadegan `[通讯]` (Virginia Tech)

**通讯引用:** 1952 | **OpenAlex IDs:** https://openalex.org/A5078713927

**关键词:** `Multiagent Systems` `Transformer` `Large Language Model` `Agentic AI` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

开发并评估一个使用大型语言模型（GPT‑5 nano）与动态记忆的生成式 AI 代理，代理扮演市长角色，在模拟的 SEIR（或 SEIRb）流行病环境中每周决策业务限制，并将决策反馈给模型进行一年（365 天）仿真。

**💡 创新点**

创新点在于：① 将生成式 AI 代理应用于政策制定而非仅模拟个体行为；② 通过向提示中加入系统级传播反馈知识来提升代理的因果推理与决策质量；③ 探索单一代理与多代理集成（ensemble）在政策制定中的效果差异，并揭示知识干预与集成的交互影响。

**🔧 技术方法**

技术：大语言模型（GPT‑5 nano）作为决策引擎；动态记忆架构（按最近性加权随机抽取过去事件）；SEIR/SEIRb 传染病模拟器；多种实验条件（基线、知识干预、集成、集成+知识）；统计回归分析评估代理决策驱动因素。

**📊 数据集**

数据集：完全基于内部仿真生成的数据，设置人口 100 万、初始病例 1、SEIR 参数（β₀=0.2、L=4、D=10、α=0.8、k=5×10⁻⁴），共 10 次独立随机种子跑，365 天，周报病例与政策决策被记录。

**📈 对比分析**

比较方法：对 4 种实验配置（Base、Knowledge、Ensemble、Ensemble+Knowledge）分别计算累计病例数、累计预测误差、平均传播率下降；与单一代理基线进行相对比较。结果显示：知识干预使累计病例平均下降约 1/3；集成+知识方案表现最佳，累计病例相较基线减少约 50%，且预测误差最小；单纯集成方案表现最差。

**⚠️ 局限性**

局限性：1）SEIR/SEIRb 模型简化，忽略空间、年龄、医疗资源、经济反馈等真实复杂性；2）代理获得完美的周病例信息，未考虑报告延迟或误差；3）知识干预提供了完全准确的因果结构，实际决策者可能接收错误或冲突信息；4）仅测试 GPT‑5 nano，未系统比较不同 LLM 或提示设计；5）未模拟政治、法律、公众舆论等决策约束；6）评估仅关注流行病结果和预测准确性，未检验代理推理的合法性或可信度。

---

## Privacy at Scale in Networked Healthcare

**arXiv ID:** 2601.04298 | [PDF](https://arxiv.org/pdf/2601.04298v1)

**作者:** M. Amin Rahimian `[一作]` (University of Pittsburgh), James Joshi `[通讯]`

**关键词:** `Cryptography and Security` `Safty and Privacy` `Federated Learning` `Biomedical Data` `Electronic Health Records`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一套可扩展的、面向网络化医疗系统的隐私保护框架，强调在整个医疗数据生命周期中采用决策理论差分隐私（DP），并结合网络感知隐私、合规即代码（compliance‑as‑code）以及分布式推理（DPDI）等技术，构建可在多站点、跨机构、跨学科场景下实现可审计、可组合、可解释的隐私保障体系。

**💡 创新点**

创新点：
1) 将差分隐私从单模型扩展为整个学习工作流与决策链路，形成决策理论DP；
2) 引入网络感知隐私，将网络拓扑和动态学习过程纳入隐私预算分配；
3) 提出分布式推理框架DPDI，利用DP噪声保护各站点的贝叶斯更新，兼顾统计功效与隐私；
4) 将合规即代码与隐私账本结合，提供可审计的隐私预算追踪与自动化合规文档；
5) 结合多种PET（FL、DP、MPC/HE、可信执行环境）形成模块化控制平面，支持跨机构多任务协作。

**🔧 技术方法**

主要技术：差分隐私（DP）及其网络感知变体；分布式推理（DPDI）和贝叶斯更新；联邦学习（FL）与安全聚合；多方安全计算（MPC）、同态加密（HE）与可信执行环境（TEE）；合规即代码框架、隐私预算账本、可审计日志与自动化合规文档生成；数据互操作性标准（OMOP/PCORnet CDM）与查询预审计。

**📊 数据集**

使用的典型数据集示例：
- AI‑READI 项目多中心糖尿病数据集（基于 OMOP CDM 的 EHR 与 DICOM 图像）；
- 以往公开基因组学（GWAS）和蛋白质组学数据集；
- 模拟的感染网络（如 HIV MSM 交往网络）用于网络感知 DP；
- 合成的高保真 EHR 资源（如SyntheticEHR）用于测试与红队演练；
- 真实的 mHealth 设备流量（CGM、Garmin 运动跟踪器）用于实时数据隐私评估。

**📈 对比分析**

方法比较与性能：
- 在 AI‑READI 之类的多中心实验中，DPDI 与传统中心化统计（无隐私）相比，误差仅增加约 5–10%，而隐私预算可精确追踪；
- 对比联邦学习+DP 方案，DPDI 在通信开销上更低（仅需上传贝叶斯更新而非梯度），且对动态数据流更友好；
- 与纯 MPC/HE 方法对比，DPDI 在算力与延迟方面优越（仅需对数似然加噪，而不涉及昂贵的加密运算），但在极高安全需求场景下仍可插入 MPC/HE 子模块；
- 在网络感知 DP 的仿真中，使用 1%–5% 的隐私预算即可保持对感染网络关键节点的识别率 ≥ 90%，而不需要全局匿名化。

**⚠️ 局限性**

局限性：
1) 论文以概念与架构为主，缺乏大规模真实系统的部署与长期实验数据；
2) DP 的隐私预算分配与组合在复杂动态场景中仍需经验与策略指导，缺乏统一的自动化工具；
3) 对网络感知隐私的理论与算法尚处于早期阶段，难以在极大规模社交网络上直接验证；
4) 合规即代码与隐私账本的标准化实现依赖各机构的 IT 能力与政策接受度；
5) 兼顾多种 PET 的控制平面实现需要深厚的安全与系统集成经验，部署成本较高；
6) 在对抗攻击（如伪重识别、聚类逆推）面前，DP 与联邦学习的防御效果仍需进一步验证。

---

## UNIC: Learning Unified Multimodal Extrinsic Contact Estimation

**arXiv ID:** 2601.04356 | [PDF](https://arxiv.org/pdf/2601.04356v1)

**作者:** Zhengtong Xu `[一作]` (Purdue University), Yuki Shirai `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `Robotics` `Robotic Intelligence` `Transformer` `Multimodality` `Point Cloud`

### 📋 论文摘要

**🎯 论文内容**

提出一种无先验知识、无相机标定的统一多模态框架，用于估计抓取物体与环境之间的外部接触，并给出对应的接触亲和力地图。

**💡 创新点**

创新点包括：
1) 统一接触亲和力表示，能够捕捉点、线、面以及多物体链式接触；
2) 先验无关设计，直接使用相机帧点云作为参考，无需事先获取物体几何或相机标定；
3) 随机遮蔽的多模态融合机制，提升对缺失模态的鲁棒性；
4) 采样高效策略将全局融合与点级推断解耦，显著加速推理。

**🔧 技术方法**

技术手段包括：
- 多模态编码器（点云使用 PointNet，触觉采用基于marker位移的 Transformer，力矩和末端姿态使用 MLP）
- 所有模态投射至共享嵌入维度，统一 token 数量；
- 随机遮蔽学习的 mask token，结合模态无关 Transformer 核心；
- 基于高斯核的接触亲和力生成，从稀疏点标注得到稠密亲和力；
- 采样策略：先得到全局多模态特征，再广播到每个点做轻量点级头输出。

**📊 数据集**

数据集：
- 真实机器人数据，使用 Mitsubishi MELFA 6‑DoF 机器人、WSG‑32 扳手、GelSight Mini 触觉传感器和 Intel RealSense D435 RGB‑D 相机；
- 训练集包含“已知”物体的多个抓取/接触/非接触片段，验证集进一步划分为已知物体在未见接触位置、完全未见物体两类；
- 采样点云为 1,024 点，拍摄时相机视角随机化。

**📈 对比分析**

对比方法：
- 无遮蔽多模态（w/o Masked Fusion）
- 端到端回归（直接输出接触点）
- 仅视觉估计（只用点云和姿态）

性能：
- 在已知物体未见接触位置上，Chamfer 距离 9.6 mm，单点距离 16.7 mm；
- 在完全未见物体上也保持可接受误差；
- 随机遮蔽策略使模型在任何模态缺失时仍能保持 1‑3 mm 级别的误差；
- 推理速度超过 600 fps（单 RTX 3080 GPU），满足实时需求。

**⚠️ 局限性**

局限性：
- 依赖人工标注的接触点，标注成本高且易受稀疏性影响；
- 点云分辨率有限，极端几何细节可能无法完整捕捉；
- 虽然能泛化到未见物体，但在形状差异极大或多物体复杂接触时误差仍显著；
- 目前仅在实验室抓取/放置场景验证，实际复杂环境下的鲁棒性需进一步评估。

---

## Timeliness-Oriented Scheduling and Resource Allocation in Multi-Region Collaborative Perception

**arXiv ID:** 2601.04542 | [PDF](https://arxiv.org/pdf/2601.04542v1)

**作者:** Mengmeng Zhu `[一作]` (Beijing Jiaotong University), Sheng Zhou `[通讯]` (Tsinghua University)

**通讯引用:** 12524 | **OpenAlex IDs:** https://openalex.org/A5021120517

**关键词:** `Machine Learning` `Optimization` `Lyapunov optimization` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

研究多区域协作感知中的实时调度与资源分配问题，提出基于时效性的多区域优先调度算法TAMP；

**💡 创新点**

创新点在于设计同时考虑AoI与通信量的非线性惩罚函数，并通过Lyapunov优化与KKT条件推导出优先指标U(h,b)，实现实时动态调度与通信量自适应；

**🔧 技术方法**

采用Lyapunov优化、KKT条件、经验惩罚函数、仿真评估，结合特征级融合和空间置信度压缩技术；

**📊 数据集**

使用真实道路边协作感知数据集RCooper（交叉口与走廊场景）；

**📈 对比分析**

与Age-Prio、Rate-Prio、GEA、Max-Weight四个基线对比，TAMP在多种参数设置下平均提升AP约27%，在低速、低带宽或预算紧张时优势更显著；

**⚠️ 局限性**

局限在于模型假设为单任务并行限制、惩罚函数需实验拟合，未充分考虑多任务异步延迟的复杂场景，对控制参数V敏感，需要调参。

---

## Actively Obtaining Environmental Feedback for Autonomous Action Evaluation Without Predefined Measurements

**arXiv ID:** 2601.04235 | [PDF](https://arxiv.org/pdf/2601.04235v1)

**作者:** Hong Su `[一作]` (Chengdu University of Information Technology), Hong Su `[通讯]` (Chengdu University of Information Technology)

**通讯引用:** 145 | **OpenAlex IDs:** https://openalex.org/A5031030652

**关键词:** `Artificial Intelligence` `Robotic Intelligence` `Transformer` `Large Language Model` `Agentic AI` `Text` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出一种主动反馈获取模型，使 AI 代理在没有预定义测量的环境中通过主动交互发现、筛选并验证反馈。

**💡 创新点**

创新点在于：①将环境差异视为反馈载体；②引入主动行动干预与自触发机制；③采用差异驱动的筛选和验证流程；④构建混合记忆（参数化+符号化）记录动作‑反馈关系并实现可迁移。

**🔧 技术方法**

采用差异驱动检测、主动行动干预、基于大语言模型的期望引导与推理、差异中心化记忆结构以及累积学习的动作‑反馈映射技术。

**📊 数据集**

实验数据为：1）基于文本描述的情境案例（无公开数据集）；2）仿真环境 ℰ_sim；使用 DeepSeek‑70B 进行推理与 MiniLM 计算相似度。

**📈 对比分析**

对比方法：差异驱动推理 vs 直接推理；主动行动 vs 被动观察；评价指标为相似度得分与 LLM 查询次数。结果显示：差异驱动平均相似度 0.366，显著高于 0.292；主动方法平均查询 2.95 次，显著低于被动 5.29 次（p<0.05）。

**⚠️ 局限性**

局限性：实验仅在文本与仿真环境中验证，缺乏真实复杂多模态或多智能体场景；模型对 LLM 的依赖较强；差异算子的设计尚不完整，需要进一步评估其泛化与鲁棒性。

---

## When Models Manipulate Manifolds: The Geometry of a Counting Task

**arXiv ID:** 2601.04480 | [PDF](https://arxiv.org/pdf/2601.04480v1)

**作者:** Wes Gurnee `[一作]` (Anthropic), Joshua Batson `[通讯]` (Anthropic)

**关键词:** `Machine Learning` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究 Claude 3.5 Haiku 在固定宽度文本中进行换行时，如何通过内部特征和几何流形实现字符计数、边界检测与换行决策的机制。

**💡 创新点**

首次将稀疏字典特征与低维连续流形相结合，揭示 Transformer 内部计数的“波纹”结构、多头旋转操作和分布式计数算法，并发现能诱发“视觉错觉”的注意力干扰。

**🔧 技术方法**

使用特征字典（WCC）提取稀疏特征、特征激活图、QK 与 OV 归因图、线性与逻辑回归探针、PCA 低维投影、因果干预与插值实验等一系列可解释性工具。

**📊 数据集**

构造合成数据集：从多样化文本语料中去除换行符，再按 15‑150 字宽度重新插入换行，辅以源代码、聊天记录、邮件、扫描文章等真实文本进行验证。

**📈 对比分析**

通过干预实验和线性判别验证，字符计数子空间在换行预测中承担关键作用；在合成数据上 AUC 达 0.91，线性探针对字符计数的 R² 0.985，说明模型实现了高度准确的换行预测。

**⚠️ 局限性**

局限性包括：解释仍受“复杂性税”限制，未给出完全可复制的计数算法；对多词输出、可变行宽、未建模的不确定性等情况缺乏完整描述；需要更自动化的流形检索与层级化工具。

---

## Explainable Admission-Level Predictive Modeling for Prolonged Hospital Stay in Elderly Populations: Challenges in Low- and Middle-Income Countries

**arXiv ID:** 2601.04449 | [PDF](https://arxiv.org/pdf/2601.04449v1)

**作者:** Daniel Sierra-Botero `[一作]` (University of Antioquia), Olga Lopez-Acevedo `[通讯]` (University of Antioquia)

**关键词:** `Machine Learning` `Explainability and Interpretability` `Tabular` `Biomedical Data` `Electronic Health Records`

### 📋 论文摘要

**🎯 论文内容**

构建并验证了一个可解释的预测模型，预测医院入院时患者是否会出现长期住院（pLOS）风险。

**💡 创新点**

创新点在于将权重证据（WoE）、信息值（IV）与图论最大团（clique）相结合进行特征选择，显著减少变量数并提升模型可解释性。

**🔧 技术方法**

采用了Logistic回归、WoE/IV编码、OptimalBinning、NetworkX构建相关图、RFE对比以及SHAP解释等技术。

**📊 数据集**

使用了哥伦比亚阿尔玛母大学医院2017‑2022年的120,354例入院记录，清洗后得到80,628例样本。

**📈 对比分析**

与传统特征选择（RFE）及无特征筛选模型相比，本方法在验证集上实现AUC‑ROC 0.82、精确率 0.67、召回率 0.64、特异性 0.83，优于RFE召回率且保持相似准确率。

**⚠️ 局限性**

局限在于训练数据基于离线完整记录，缺乏实时缺失值处理和因果推断；WoE/IV解释不等同于因果关系。

---

## Enhancing Robustness of Asynchronous EEG-Based Movement Prediction using Classifier Ensembles

**arXiv ID:** 2601.04286 | [PDF](https://arxiv.org/pdf/2601.04286v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## Green MLOps: Closed-Loop, Energy-Aware Inference with NVIDIA Triton, FastAPI, and Bio-Inspired Thresholding

**arXiv ID:** 2601.04250 | [PDF](https://arxiv.org/pdf/2601.04250v1)

**作者:** Mustapha Hamdi `[一作]`, Mourad Jabou `[通讯]`

**关键词:** `Machine Learning` `Protein Structure Prediction` `Computational Efficiency` `Transformer` `Supervised Fine-Tuning` `Text` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出一种生物启发的闭环能源感知推理控制器，将蛋白质折叠能量盆地映射到推理成本景观，并通过可衰减阈值决定是否执行推理，结合 FastAPI+ONNX Runtime 与 NVIDIA Triton 的双路径服务。

**💡 创新点**

创新点在于将蛋白质折叠的能量模型引入 MLOps，设计可衰减闭环阈值控制器，只接受在本地可接受能量‑效用平衡内的请求，从而显著降低能耗并保持精度。

**🔧 技术方法**

使用 FastAPI+ORT、NVIDIA Triton、ONNX Runtime、MLflow、CodeCarbon、GPU（RTX 4000 Ada / A100）以及动态批处理；构建成本函数时结合软熵、能耗和拥塞三项指标。

**📊 数据集**

评估模型为 DistilBERT（在 SST‑2 句子分类数据集上）和 ResNet‑18（ImageNet 或占位图像集），通过这些数据集验证控制策略。

**📈 对比分析**

对比 FastAPI/ORT 与 Triton 两条路径，测量延迟、吞吐量、能耗和 CO₂；在 A100 上闭环控制将总时延从 0.50 s 降至 0.29 s（42%），延迟与能耗下降约 42%，精度仅下降 0.5%。

**⚠️ 局限性**

局限性包括实验使用合成/占位输入、仅测试 batch=1 使得高并发场景的优势未充分体现、CO₂ 估算依赖地区电网强度以及阈值参数需手工调节。

---

## Gavel: Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization

**arXiv ID:** 2601.04424 | [PDF](https://arxiv.org/pdf/2601.04424v1)

**作者:** Yao Dou `[一作]` (Georgia Institute of Technology), Wei Xu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 21777 | **OpenAlex IDs:** https://openalex.org/A5100690851

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究了多文档法律案件摘要的评估框架，并对12款最新LLM在100个2025年美国民权诉讼案例（32K–512K token）上的表现进行细粒度评估。

**💡 创新点**

创新点包括：①基于参考的多值清单评估、残余事实与写作风格三大组件的整合评估框架；②提出六工具自主代理（Read、Search、Update等）以高效从原始案件文档中直接抽取清单。

**🔧 技术方法**

主要技术手段是：利用Gemini、Claude、GPT-5、Qwen3等LLM进行信息抽取与摘要生成；基于LLM的提示和工具调用构建的代理框架；对比评估使用的自定义评分公式与F1匹配。

**📊 数据集**

数据集为2025年民权诉讼清晰仓（Civil Rights Litigation Clearinghouse）中公开的多文档案件，包含32K、64K、128K、256K、512K五个长度区间，各区间20个案例。

**📈 对比分析**

评估方法是将S_checklist、S_residual、S_style按权重合成S_得分，结果显示Gemini 2.5 Pro最高达51分，Claude Sonnet 4和Gemini 2.5 Flash次之；模型性能随案例长度递减，尤其多值与罕见清单项表现最差。

**⚠️ 局限性**

局限性在于：仅进行评估未改进摘要质量；测试样本集中于2025年案例，未覆盖更旧或跨国案例；未对最强闭源模型如GPT‑5.2、Claude 4.5 Pro进行评测；代理在单一模型处理26项时效果不佳，需进一步优化多代理设计。

---

## AHA: Scalable Alternative History Analysis for Operational Timeseries Applications

**arXiv ID:** 2601.04432 | [PDF](https://arxiv.org/pdf/2601.04432v1)

**作者:** Harshavardhan Kamarthi `[一作]` (Georgia Institute of Technology), Vyas Sekar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 16436 | **OpenAlex IDs:** https://openalex.org/A5079175103

**关键词:** `Databases` `Computational Efficiency` `Anomaly Detection` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了 AHA（Alternative History Analytics）系统，用于在运营时序数据上高效、准确地支持历史回放式分析。

**💡 创新点**

核心创新在于：①利用子群体可分解性将需要的统计量从叶子子群体聚合，再通过 CUBE 或 GROUPING SETS 生成任意子群体指标；②在 ingest 阶段仅计算叶子子群体所需的可分解统计量，显著降低存储与计算成本；③理论上证明对所有可分解统计量的预测任务实现完美等价。

**🔧 技术方法**

采用分层聚合（leaf‑level 统计）+ OLAP CUBE/GROUPING SETS（ClickHouse、Spark）实现，结合分布式存储（S3）和计算资源（EC2）。

**📊 数据集**

使用了多个真实与合成数据集：视频 QoE 数据、网络流量日志、NYC Taxi 数据、ClickHouse 日志、GPU 性能基准、IMDB 电影元数据等。

**📈 对比分析**

与存储原始数据、key‑value 输出、采样与 Hydra Sketch 等基线比较；在准确度上保持 100%（与原始数据完全一致），在成本上比原始存储低 55–130 倍，整体拥有率成本比其他完美等价方案低 6–10 倍，且在多任务并行与属性维度扩展时保持线性伸缩。

**⚠️ 局限性**

局限性：仅对可分解统计量保证完美等价，对非可分解统计量（如中位数、分位数）需采用近似估计；若叶子子群体数量仍过大，单机 CUBE 计算可能受限；对极低样本子群体仍可能出现精度下降。

---

## LLM-Guided Lifecycle-Aware Clustering of Multi-Turn Customer Support Conversations

**arXiv ID:** 2601.04388 | [PDF](https://arxiv.org/pdf/2601.04388v1)

**作者:** Priyaranjan Pattnayak `[一作]` (Oracle America Inc), Hitesh Laxmichand Patel `[通讯]` (Oracle America Inc)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

基于LLM的多轮聊天分解与主题聚类，实时增量更新并通过质控指标动态拆分/合并，解决云服务支持聊天的主题漂移。

**💡 创新点**

创新点是将LLM分解、对比过滤、服务组分配与增量聚类结合，利用指标驱动的自适应拆分/合并，避免全量重聚。

**🔧 技术方法**

使用LLM（cohere.command-r-08-2024）、Sentence‑BERT、UMAP、HDBSCAN、余弦相似度、Z‑score等技术。

**📊 数据集**

在90,048条匿名多轮聊天数据（含148,200个关键信息）以及公开的合成数据集上评估。

**📈 对比分析**

与KMeans+BERT、单纯HDBSCAN等基线相比，Silhouette提升111.7%，DBI下降65.6%，增量过程保持质量并在漂移时恢复。

**⚠️ 局限性**

局限包括对多服务混合疑问的处理仍不完善、评估主要基于内部指标缺乏业务影响度量、仅支持英语。

---

## MiJaBench: Revealing Minority Biases in Large Language Models via Hate Speech Jailbreaking

**arXiv ID:** 2601.04389 | [PDF](https://arxiv.org/pdf/2601.04389v1)

**作者:** Iago Alves Brito `[一作]`, Arlindo Rodrigues Galvão Filho `[通讯]`

**关键词:** `Computation and Language` `Adversarial Attack` `Safty and Privacy` `Transformer` `Large Language Model` `Chain-of-Thought` `Prompt Engineering` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了MiJaBench—a bilingual（英语/葡萄牙语）对抗性基准，包含44,000条针对16个少数族群的攻击提示，并从12个最新LLM中生成了528,000条提示‑响应对，用于系统评估模型的安全性。

**💡 创新点**

创新点在于：① 通过细粒度群体划分揭示LLM安全性选择性差异，② 证明模型规模放大反而加剧了不公平，③ 提供完整的MiJaBench‑Align数据集，供后续对齐训练与评估。

**🔧 技术方法**

技术方法包括：① 随机组合生成策略与四类 Jailbreaking 技术（Persona、Representation Shifting、Chain‑of‑Thought、Logical Rationalization），② LLM‑as‑Judge 自动判定安全性，③ Bootstrap 稳定性检验与统计差异分析。

**📊 数据集**

数据集来源：ToxiGen（英语）与ToxSyn（葡萄牙语）两大hate‑speech语料库，各抽取2,000例，合计44k样本；随后生成的528k对话数据。

**📈 对比分析**

评估方式：采用 Defense Rate（防御率）与 Attack Success Rate，使用 Majority‑Vote LLM‑Judge，比较12个开源LLM在四个规模层级下的防御表现；发现最大防御率差异达33%，而模型规模增大时差距进一步扩大。

**⚠️ 局限性**

局限性：① 仅覆盖16个单一身份标签，忽略交叉性（如黑人女性）；② 仅包含英语与葡萄牙语两种语言；③ 样本为人工合成，可能不完全代表真实场景；④ 未评估正向提示对模型安全性的影响。

---

## Enhanced-FQL($λ$), an Efficient and Interpretable RL with novel Fuzzy Eligibility Traces and Segmented Experience Replay

**arXiv ID:** 2601.04392 | [PDF](https://arxiv.org/pdf/2601.04392v1)

**作者:** Mohsen Jalaeian-Farimani `[一作]` (Politecnico di Milano), Mohsen Jalaeian-Farimani `[通讯]` (Politecnico di Milano)

**通讯引用:** 90 | **OpenAlex IDs:** https://openalex.org/A5066851566

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning`

### 📋 论文摘要

**🎯 论文内容**

提出了一种名为 Enhanced-FQL(λ) 的模糊强化学习框架，结合模糊贝尔曼方程、模糊资格跟踪和分段经验回放，用于连续控制任务。

**💡 创新点**

创新点在于引入了模糊化的贝尔曼方程配合多步资格跟踪实现稳定的多步信用分配，以及内存高效的分段经验回放机制，提供可解释性、理论收敛保证并显著提升样本效率。

**🔧 技术方法**

使用技术包括高斯模糊隶属函数、模糊资格跟踪（TD(λ)）、分段经验回放、软最大（SoftMax）动作选择、模糊Q学习以及对比的深度强化学习（DDPG）实现。

**📊 数据集**

实验数据集为经典的 CartPole 连续控制环境（模拟弹杆平衡任务）。

**📈 对比分析**

通过与 n‑step FQL、Fuzzy SARSA(λ) 和 DDPG 的对比实验，Enhanced‑FQL(λ) 在平均回报、收敛速度（约 129 轮到达 -200 回报）和更新耗时上均优于基线，且具有更低的方差和更高的样本效率。

**⚠️ 局限性**

局限性包括：受限于模糊分区的近似误差；需手动调节超参数（λ、学习率、分区数量）；虽然可解释但仍无法完全替代深度网络在高维复杂任务中的表达能力；未来可考虑优先经验回放、动态规则自适应等改进。

---

## Unlocking the Pre-Trained Model as a Dual-Alignment Calibrator for Post-Trained LLMs

**arXiv ID:** 2601.04277 | [PDF](https://arxiv.org/pdf/2601.04277v1)

**作者:** Beier Luo `[一作]` (Southern University of Science and Technology), Xuefeng Du `[通讯]` (Nanyang Technological University)

**通讯引用:** 768 | **OpenAlex IDs:** https://openalex.org/A5001001983

**关键词:** `Machine Learning` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种无监督的双重对齐框架，对后训练的大语言模型进行自适应温度校准。

**💡 创新点**

创新点在于同时纠正输出信心漂移和过程漂移：通过峰值发散层定位过程差异，并将推理稳定熵与预训练模型对齐。

**🔧 技术方法**

核心技术包括自监督温度缩放、峰值发散层（PDL）检测、推理稳定熵（ISE）计算以及基于Jensen‑Shannon散度的动态加权损失。

**📊 数据集**

实验使用多种模型（Llama‑3.1、Qwen‑2.5、Gemma‑3）和公开基准（MMLU、MedMCQA、TruthfulQA）进行评估。

**📈 对比分析**

与现有无监督校准方法（DACA、IC、CAPE、Elicitation）以及有监督温度缩放对比，平均降低ECE超过30%，在多数模型上逼近监督oracle性能。

**⚠️ 局限性**

局限在于仅依赖预训练模型作为参考，无法处理多模态或其他非文本任务；且仅调整温度，未改变模型参数，可能仍受后训练带来的根本结构偏差影响。

---

## Cross-Language Speaker Attribute Prediction Using MIL and RL

**arXiv ID:** 2601.04257 | [PDF](https://arxiv.org/pdf/2601.04257v1)

**作者:** Sunny Shu `[一作]` (Informatics Institute University of Amsterdam), Ali Mohammed Mansoor Alsahag `[通讯]` (Informatics Institute University of Amsterdam)

**关键词:** `Artificial Intelligence` `Classification` `Domain Adaptation` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Text` `Audio`

### 📋 论文摘要

**🎯 论文内容**

研究了多语言说话者属性预测，扩展了RL‑MIL框架并加入域对抗训练实现跨语言鲁棒性。

**💡 创新点**

创新点在于将RL‑MIL与梯度反转层的域对抗训练（DAT）结合，产生语言不可辨识的特征，显著提升多语言性别/年龄预测。

**🔧 技术方法**

使用了多实例学习、强化学习实例选择、梯度反转层的域对抗训练、预训练多语言编码器（mBERT、XLM‑R）以及多种池化头（Mean、Max、Attention）。

**📊 数据集**

使用了5 语言的Twitter数据集（few‑shot）和40 语言的VoxCeleb2子集（zero‑shot）。

**📈 对比分析**

通过27个模型组合、5个随机种子进行对比，RLMIL‑DAT在Twitter上显著提升Macro‑F1（尤其性别）且统计显著；在VoxCeleb2上提升趋势正向但未达到统计显著。

**⚠️ 局限性**

局限性包括计算成本高（超过30天GPU时）、未对编码器进行微调、零样本实验样本量小导致统计功效低、未做逐语言细粒度分析。

---

## TCAndon-Router: Adaptive Reasoning Router for Multi-Agent Collaboration

**arXiv ID:** 2601.04544 | [PDF](https://arxiv.org/pdf/2601.04544v1)

**作者:** Jiuzhou Zhao `[一作]` (Tencent Cloud Andon), Yanchi Liu Yongzhou Xu Xiaochuan Xu Min Zhang `[通讯]`

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Chain-of-Thought` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于推理的多智能体路由框架TCAR，能生成自然语言推理链并输出适合的专家代理集合，同时通过聚合代理输出的精炼代理提升最终答案质量。

**💡 创新点**

创新点在于：①将路由从单标签分类转为“推理‑然后‑选择”多标签决策；②支持动态加入新代理，无需重新训练；③通过协同执行和精炼代理消除代理冲突，提升鲁棒性与可解释性。

**🔧 技术方法**

使用大规模语言模型（Qwen3-4B-Instruct-2507）进行两阶段训练：监督微调（SFT）与基于DAPO的强化学习；推理链使用统一<reason>标签；协同执行采用多代理并行调用并聚合。

**📊 数据集**

评测数据集包括公开意图分类数据集 CLINC150、HWU64、MINDS14、SGD 以及腾讯云内部的 QCloud 业务数据集。

**📈 对比分析**

与 GPT‑5.1、Claude‑4.5、DeepSeek‑v3.1、ArcRouter、Qwen3-Embedding‑4B 等基线对比，TCAR 在四个公共数据集的准确率均超过 90%，在 QCloud 的 F1 达到 93.98（最高），并且在多代理聚合实验中显著提升人类偏好胜率，证明了性能和鲁棒性的提升。

**⚠️ 局限性**

局限性包括：①依赖代理描述质量，描述不完整或歧义会影响推理与路由；②在长尾领域或极端低频场景下仍可能不稳定；③对大长提示（如 CLINC150）的小模型表现受限。

---

## Paradoxical noise preference in RNNs

**arXiv ID:** 2601.04539 | [PDF](https://arxiv.org/pdf/2601.04539v1)

**作者:** Noah Eckstein `[一作]` (Ohio State University), Manoj Srinivasan `[通讯]` (Ohio State University)

**通讯引用:** 2479 | **OpenAlex IDs:** https://openalex.org/A5062283980

**关键词:** `Neural and Evolutionary Computing` `Recurrent Neural Network` `Stochastic Differential Equation` `Time Series` `Sequential`

### 📋 论文摘要

**🎯 论文内容**

研究连续时间递归神经网络（CTRNN）在训练中加入噪声对测试性能的影响，发现噪声注入在激活函数内部的网络在测试时对与训练时相同的噪声水平最优，而噪声注入在激活函数外部的网络则在零噪声时最佳。

**💡 创新点**

揭示噪声偏好现象并解释其源于噪声导致的固定点位移，使得网络对特定噪声水平产生过拟合。

**🔧 技术方法**

使用连续时间递归神经网络（CTRNN），ReLU/SoftPlus激活，Euler数值积分，均值方差误差评估，及噪声注入策略。

**📊 数据集**

对合成任务进行实验，包括一元函数逼近（sin、tanh）、迷宫导航（基于格点迷宫）以及单神经元调节器（单个神经元保持设定点）。

**📈 对比分析**

与无噪声和不同噪声水平进行对比；使用RMSE、均值误差和方差评估；发现噪声注入在激活函数内部的网络在训练噪声水平测试时RMSE最低，噪声注入在激活函数外部的网络在零噪声时最佳。

**⚠️ 局限性**

仅限于设定点稳态任务，未涵盖周期性或更复杂动态；使用加性高斯噪声，未探究乘性或泊松噪声等信号相关噪声；可能对实际生物网络的泛化有限。

---

## A Closed-Loop Multi-Agent System Driven by LLMs for Meal-Level Personalized Nutrition Management

**arXiv ID:** 2601.04491 | [PDF](https://arxiv.org/pdf/2601.04491v1)

**作者:** Muqing Xu `[一作]` (University of Bristol), Muqing Xu `[通讯]` (University of Bristol)

**关键词:** `Artificial Intelligence` `Recommendation System` `Recognition` `Transformer` `Large Language Model` `Agentic AI` `Image` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

构建了基于多模态大语言模型的闭环多智能体系统，实现了从餐食图片识别到营养分析再到个性化膳食建议的全流程闭环管理

**💡 创新点**

创新点在于将视觉识别、对话推理和文件管理拆分为四个专责智能体，并通过LLM驱动的工作流调度实现实时、可审计的闭环调整；同时提出了每日剩余预算动态分配的营养计划更新策略

**🔧 技术方法**

使用GPT‑4o‑mini、GPT‑4o、Claude Sonnet 4和GPT‑4.1‑mini四个大语言模型分别担当控制器、视觉分析器、文件管理器和对话顾问，配合Flask API和Android前端

**📊 数据集**

主要数据集包括USDA DRI数据库（RDA表）用于生成每日营养目标，SNAPMe图片数据集用于评估食物识别、体积/质量估计及营养分析

**📈 对比分析**

通过与SNAPMe中21张食物图和30张含/不含参考物体的图像比较，GPT‑4o在食物识别准确率88%与体积/质量估计误差7.6%/16.7%略逊于Gemini 2.5 Flash，但在稳定性上优于后者；营养分析MAE在完整40字段上为65.2 kcal，覆盖率0.76；多智能体任务完成率（PO）平均0.75，E2E时延约65 s，日计划更新方向一致性达0.82

**⚠️ 局限性**

主要局限在于单张图片难以准确估计微量营养素、实验规模受限、未进行真实用户长期评估，且系统对高遮挡或无参考物体的图像仍有一定误差

---

## Beyond Static Summarization: Proactive Memory Extraction for LLM Agents

**arXiv ID:** 2601.04463 | [PDF](https://arxiv.org/pdf/2601.04463v1)

**作者:** Chengyuan Yang `[一作]` (Nanjing University), Wei Hu `[通讯]` (Nanjing University)

**通讯引用:** 10465 | **OpenAlex IDs:** https://openalex.org/A5022039557

**关键词:** `Computation and Language` `Large Language Model` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一种主动式记忆提取框架ProMem，利用循环反馈的自问自答机制对对话历史进行迭代式抽取与校验；

**💡 创新点**

创新点在于将抽取视为可反馈的认知过程，借鉴循环处理理论，引入自问自答的循环验证以提升记忆完整性与准确性；

**🔧 技术方法**

采用LLM进行初始抽取、语义匹配完成、以及自问自答验证，使用嵌入模型（Qwen3-Embedding-8B）进行相似度匹配；

**📊 数据集**

主要使用HaluMem数据集评估记忆完整性与准确性，并在LongMemEval上验证下游问答性能；

**📈 对比分析**

与Memobase、Supermemory、Mem0、LightMem等基线对比，ProMem在记忆完整性提升至73.8%、准确率88.1%及QA准确率62.1%等指标上均实现SOTA；

**⚠️ 局限性**

主要限制为较高的token消耗与推理延迟，以及对大规模LLM的依赖，低配模型下自问自答效果可能下降。

---

## BioPIE: A Biomedical Protocol Information Extraction Dataset for High-Reasoning-Complexity Experiment Question Answer

**arXiv ID:** 2601.04524 | [PDF](https://arxiv.org/pdf/2601.04524v1)

**作者:** Haofei Hou `[一作]` (Peking University), Qining Wang `[通讯]` (Peking University)

**通讯引用:** 4151 | **OpenAlex IDs:** https://openalex.org/A5025723618

**关键词:** `Artificial Intelligence` `Knowledge Distillation` `Drug Discovery` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Supervised Fine-Tuning` `Biomedical Data` `Graph`

### 📋 论文摘要

**🎯 论文内容**

构建了专门针对生物医学实验协议的结构化知识图谱数据集 biopie，并基于该数据集实现了高推理复杂度的问答系统。

**💡 创新点**

创新点在于以实验流程为中心的细粒度实体与关系标注体系，兼顾 hid 与 msr 两类推理任务，并提供可用于知识检索与推理的图结构化表示。

**🔧 技术方法**

采用实体识别与关系抽取的管道与联合模型、LLM（GPT‑5、Claude‑4.5、Llama‑4 等）的零/少样本推理与 LoRA 微调，并在问答阶段结合文本与图检索的 RAG 框架。

**📊 数据集**

使用 biopie 本身（约 10.9k 实体、8.8k 关系）进行训练/测试；对比公开数据集 SciERC、ChemPort 等作为基准。

**📈 对比分析**

与 BM25、LaBSE、Emb‑3‑large、GRAG、仅 LLM 等基线相比，系统在 hid 与 msr 两类问答中分别达 69.4% 与 62% 的准确率，显著优于所有基线。

**⚠️ 局限性**

主要局限包括：LLM 进行协议规范化时可能引入步骤顺序偏移；图检索与推理方法仍较为简单，未充分利用时序与层级结构；实验复现需在专业监督下进行。

---

## Concept Tokens: Learning Behavioral Embeddings Through Concept Definitions

**arXiv ID:** 2601.04465 | [PDF](https://arxiv.org/pdf/2601.04465v1)

**作者:** Ignacio Sastre `[一作]` (Universidad de la República), Aiala Rosá `[通讯]` (Universidad de la República)

**通讯引用:** 209 | **OpenAlex IDs:** https://openalex.org/A5104021555

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一种在冻结的LLM中通过学习新概念的嵌入（Concept Tokens）来实现轻量级的概念引入和行为控制；

**💡 创新点**

创新点在于仅使用概念定义语料优化单个特殊词嵌入，既能在不更新模型权重的情况下加入新概念，又能通过正向/否定使用实现行为定向；

**🔧 技术方法**

技术主要包括soft prompting、交叉熵语言建模损失、在冻结模型上仅更新嵌入向量；

**📊 数据集**

实验数据集包括HotpotQA（闭书问答）、自制的第二语言教学对话数据、Eiffel Tower与虚构Austral Tower的Wiki式定义；

**📈 对比分析**

通过与“定义语料在上下文中”和“显式提及概念”等基线比较，发现Concept Tokens在抑制/促进幻觉、诱导再说法方面表现出方向性效果，但在精确率提升上仅靠增加拒答来降低幻觉；

**⚠️ 局限性**

局限性包括训练成本高、在闭书问答中只能通过降低覆盖率来减少幻觉、对细粒度事实记忆不足，以及缺乏对内部激活影响的深入解释。

---

## Data-Driven Terramechanics Approach Towards a Realistic Real-Time Simulator for Lunar Rovers

**arXiv ID:** 2601.04547 | [PDF](https://arxiv.org/pdf/2601.04547v1)

**作者:** Jakob M. Kern `[一作]` (Tohoku University), Kazuya Yoshida `[通讯]` (Tohoku University)

**通讯引用:** 13030 | **OpenAlex IDs:** https://openalex.org/A5023419492

**关键词:** `Robotics`

### 📋 论文摘要

**🎯 论文内容**

通过数据驱动的回归模型，将高视觉真实感与物理真实感结合，在实时月球表面仿真器中实现了逼真的车轮滑移、下沉和地形变形。

**💡 创新点**

创新点在于利用从现场、单轮实验和DEM仿真得到的滑移/下沉回归模型，在刚体动力学框架内通过滑移调整速度、弹簧阻尼接触以及基于滑移的地形变形，实现了低计算成本的实时物理真实性。

**🔧 技术方法**

采用了回归模型、刚体动力学+弹簧阻尼接触、滑移调整速度、基于接触力和滑移的地形变形渲染以及NVIDIA IsaacSim/OmniLRS仿真框架。

**📊 数据集**

使用了JAXA太空探索实验台的EX1四轮高速车在20m×20m沙坑（Tohoku硅砂）上收集的现场数据、单轮实验数据和DEM仿真数据。

**📈 对比分析**

将仿真结果与现场实验测得的滑移比、下沉深度以及加速/减速曲线进行比较，误差均低于0.3%（滑移）和0.25mm（下沉），证明方法高度准确。

**⚠️ 局限性**

局限性包括仅针对干燥硅砂的模型，未考虑侧向滑移和转弯时的车轮特性；对真实月球风化层仍需进一步校准，且模型仅适用于已训练好的车轮-土壤组合。

---

## Self-MedRAG: a Self-Reflective Hybrid Retrieval-Augmented Generation Framework for Reliable Medical Question Answering

**arXiv ID:** 2601.04531 | [PDF](https://arxiv.org/pdf/2601.04531v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Information Retrieval`

---

## Personalized Model-Based Design of Human Centric AI enabled CPS for Long term usage

**arXiv ID:** 2601.04545 | [PDF](https://arxiv.org/pdf/2601.04545v1)

**作者:** Bernard Ngabonziza `[一作]` (Arizona State University), Sandeep K. S. Gupta `[通讯]`

**关键词:** `Artificial Intelligence` `Safty and Privacy` `Compression` `Anomaly Detection` `Biomedical Data` `Time Series` `Electrocardiogram`

### 📋 论文摘要

**🎯 论文内容**

研究提出了使用个性化模型（针对用户行为、物理过程和生理信号）来保障长期使用的 AI 驱动人机中心控制系统的安全、可持续和安全性，并在医疗监测、胰岛素泵和可穿戴传感器等场景中进行验证。

**💡 创新点**

创新点在于将个性化模型用于三大方面：① 通过马尔可夫链与动态模型挖掘实现安全角落案例的高效检验；② 用机器学习捕捉个人生理信号关联，实现数据操纵检测和基于生理信号的密钥协商；③ 结合生成模型的压缩感知（GenCS）实现高比率压缩、低能耗恢复，显著提升可穿戴设备的续航。

**🔧 技术方法**

使用技术包括：Markov链（行为建模）、混合离散/连续模型挖掘（动态生理过程建模）、机器学习/生成模型（生理信号检测与合成）、生理信号密钥协商（PSKA）、压缩感知与生成模型压缩（CS、GeMREM、GenCS）。

**📊 数据集**

主要数据集：MIT‑BIH ECG 数据库（20 受试者）用于模型挖掘与验证，69 名受试者的临床 ECG 数据用于 GenCS 评估；此外通过模拟或实际可穿戴设备数据（血糖、运动、心率）作为使用案例。

**📈 对比分析**

对比方法：将 GenCS 与传统压缩感知（CS）及 GeMREM 进行对比。结果显示：GenCS 的压缩比约为 CS 的 5 倍；恢复时间为 CS 的一半；能耗降低约 2.7 倍，设备续航提升约 3 倍；在 10% 诊断准确率下，智能手机使用 GenCS 的续航可达 1 天，比 CS 高 15 小时。

**⚠️ 局限性**

局限性：① 需要大量个人历史数据才能训练可靠模型，初始阶段可能效果有限；② 对模型误差或未覆盖情境的鲁棒性尚未在真实长期部署中充分验证；③ 生成模型和压缩算法在资源受限的可穿戴设备上仍存在计算与通信开销；④ 对抗攻击的全面防护仍需进一步研究，单纯依赖生理特征可能被高级操纵手段绕过。

---

## Design and Development of Modular Limbs for Reconfigurable Robots on the Moon

**arXiv ID:** 2601.04541 | [PDF](https://arxiv.org/pdf/2601.04541v1)

**作者:** Gustavo H. Diaz `[一作]` (Tohoku University), Kazuya Yoshida `[通讯]` (Tohoku University)

**通讯引用:** 13030 | **OpenAlex IDs:** https://openalex.org/A5023419492

**关键词:** `Robotics` `Robotic Intelligence`

### 📋 论文摘要

**🎯 论文内容**

开发了可通过共用驱动器实现多种配置的4‑DOF模块化机器人四肢和轮式模块，构成了Moonbots；

**💡 创新点**

创新点在于统一高扭矩低速驱动器、模块间自耦合接口以及在不同任务下的九种可重构配置；

**🔧 技术方法**

采用BLDC外轮机+谐波驱动、O‑Drive FOC控制、ROS 2中间件与逆运动学实现控制；

**📊 数据集**

未使用公开数据集，仅在日本JAXA试验场进行实地负载与动态响应实验；

**📈 对比分析**

通过静态负载、速度控制步进响应和多关节同步实验验证，驱动器可提供75 Nm扭矩、26 rpm转速、位置误差≤0.05转；

**⚠️ 局限性**

局限在于重构仍需手动或键盘操作，缺乏自主重构与高级导航感知。

---

## Not All Steps are Informative: On the Linearity of LLMs' RLVR Training

**arXiv ID:** 2601.04537 | [PDF](https://arxiv.org/pdf/2601.04537v1)

**作者:** Tianle Wang `[一作]` (City University of Hong Kong), Ning Miao `[通讯]` (City University of Hong Kong)

**关键词:** `Machine Learning` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

通过对RLVR训练过程中的权重与输出进行线性回归，发现两者随训练步数呈强线性关系，并提出基于线性外推的权重/对数几率外推方法以及交互式的RL-Extra训练框架，显著减少训练成本。

**💡 创新点**

①揭示RLVR中权重与输出的普适线性动态；②利用线性外推直接推断未来模型状态；③设计RL-Extra将梯度更新与无梯度外推交替进行，实现更高效的训练。

**🔧 技术方法**

主要技术包括RLVR（GRPO/GSPO/REINFORCE++）、线性回归分析、权重与对数几率外推、梯度无关的RL-Extra循环训练；实验使用DeepSeek-R1-Distilled-Qwen-1.5B模型。

**📊 数据集**

使用DeepScaleR-Preview数据集进行RLVR训练，并在AIME‑24/25、MATH‑500、LiveCodeBench等四个算术推理与编程基准上评测。

**📈 对比分析**

对比标准RL（GRPO）在同一训练步数下的性能，权重外推可匹配至600步，logit外推在所有四个基准上均超越标准RL；RL-Extra在相同准确度下实现最高6.1×的壁钟加速，例如AIME‑24在0.38准确率时仅需180步RL而标准RL需1100步。

**⚠️ 局限性**

实验仅覆盖密集模型≤30B参数，未验证超大规模或MoE模型；未涉及多轮交互式RL；线性假设在≈1000步后会失效；尚未在工业部署环境中检验。

---

## GRACE: Reinforcement Learning for Grounded Response and Abstention under Contextual Evidence

**arXiv ID:** 2601.04525 | [PDF](https://arxiv.org/pdf/2601.04525v1)

**作者:** Yibo Zhao `[一作]` (East China Normal University), Xiang Li `[通讯]` (East China Normal University)

**通讯引用:** 40636 | **OpenAlex IDs:** https://openalex.org/A5041120433

**关键词:** `Computation and Language` `Reinforcement Learning` `Retrieval` `Reinforcement Learning` `Retrieval-Augmented Generation` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出 GRACE，一个基于强化学习的检索增强生成框架，能在检索证据充分时给出引用答案，在证据不足时做出明确拒绝。

**💡 创新点**

创新点在于：①利用异构检索器构造既有答案又无答案的训练样本，显著降低人工标注成本；②设计多阶段门控奖励函数，先评估证据可用性，再对引用证据与答案的准确性进行回报；③改进 DAPO 算法实现训练稳定和高效。

**🔧 技术方法**

使用技术包括异构检索器、数据构造与去重、XML 格式化输出、三阶段（格式、路径、内容）奖励函数、强化学习（改进的 DAPO）以及对比的检索增强生成基线。

**📊 数据集**

实验数据集为 QASPER 与 HotpotQA 两个知识密集型问答数据集。

**📈 对比分析**

通过与提示、SFT、RL 与 Agentic 5 类基线对比，GRACE 在两套数据集上均实现最高总体准确率和均衡准确率，尤其在答案正确性与拒绝可靠性之间取得优异平衡。

**⚠️ 局限性**

局限性：①未验证对大模型的可扩展性；②训练过程中依赖每个问题的精细证据标注，限制了在缺乏此类标注的数据集上的适用性。

---

## Multiagent Reinforcement Learning with Neighbor Action Estimation

**arXiv ID:** 2601.04511 | [PDF](https://arxiv.org/pdf/2601.04511v1)

**作者:** Zhenglong Luo `[一作]` (University of Newcastle), Aoxiang Liu `[通讯]` (Central South University)

**通讯引用:** 7 | **OpenAlex IDs:** https://openalex.org/A5111006598

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Reinforcement Learning` `Sequential`

### 📋 论文摘要

**🎯 论文内容**

提出了一种在多智能体协同学习中无需行动信息共享的TD3强化学习框架，并通过加入轻量级行动估计网络实现去中心化策略学习；

**💡 创新点**

创新点在于用行动估计网络代替显式行动通信，保持了TD3的稳定性与样本效率，并在去中心化环境下实现了与中心化方法相当的性能；

**🔧 技术方法**

采用TD3算法、行动估计网络（AEN）、目标网络软更新、延迟策略更新、Gaussian噪声、以及信号插值与加速度安全约束等技术；

**📊 数据集**

在Mujoco/Robosuite模拟的双臂协同抬升任务（工业组件）以及真实UR5e机器人平台上进行实验；

**📈 对比分析**

与全信息共享的中心化TD3对比，AEN‑TD3在模拟环境中8/10次实验能达到相同或更高回报，在真实机器人上同样成功完成任务，表现与中心化方法相当；

**⚠️ 局限性**

局限于两智能体规模、训练收敛速度相对慢、对估计误差敏感、对更复杂或不合作情境的适应性尚待验证；

---

## Specific Emitter Identification via Active Learning

**arXiv ID:** 2601.04502 | [PDF](https://arxiv.org/pdf/2601.04502v1)

**作者:** Jingyi Wang `[一作]` (Beijing Jiaotong University), Fanggang Wang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 1152 | **OpenAlex IDs:** https://openalex.org/A5102985769

**关键词:** `Artificial Intelligence` `Recognition` `Convolutional Neural Network` `Contrastive Learning` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种基于主动学习的特定发射器识别（SEI）框架，利用三阶段半监督训练提升少量标签下的识别精度。

**💡 创新点**

创新点在于首次将主动学习与自监督对比学习相结合，并引入联合交叉熵+对比损失的监督阶段以及基于不确定性（BALD）和代表性（K‑center）双策略的样本选择。

**🔧 技术方法**

使用技术包括自监督对比学习（MoBY‑style）、交叉熵分类损失、动量键编码器、动态字典队列、Monte Carlo Dropout 的不确定性估计以及 K‑center 贪婪算法。

**📊 数据集**

实验数据集为 ADS‑B 航空广播信号和 WiFi 802.11a 信号，分别采集于 1090 MHz 与 2.45 GHz。

**📈 对比分析**

与传统有监督 CNN 和仅对比学习的半监督方法对比，本文方法在标签预算有限时显著提升识别准确率；在 WiFi 数据上不确定性策略表现最佳，而在 ADS‑B 上代表性策略更稳健。

**⚠️ 局限性**

局限性包括：在高度重叠或噪声严重的数据分布下，不确定性估计可能失效；方法仍需预先设定少量标签，且对不同数据集的最佳 AL 策略需根据分布特征手动选择。

---

## Evaluating Human and Machine Confidence in Phishing Email Detection: A Comparative Study

**arXiv ID:** 2601.04610 | [PDF](https://arxiv.org/pdf/2601.04610v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## GUITester: Enabling GUI Agents for Exploratory Defect Discovery

**arXiv ID:** 2601.04500 | [PDF](https://arxiv.org/pdf/2601.04500v1)

**作者:** Yifei Gao `[一作]` (Beijing Jiaotong University), Jitao Sang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 2095 | **OpenAlex IDs:** https://openalex.org/A5023834030

**关键词:** `Artificial Intelligence` `Large Language Model` `Agentic AI` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出了面向探索性 GUI 测试的 benchmark GUITestBench 以及多模态 LLM 代理框架 GUITester；

**💡 创新点**

创新点在于将导航与缺陷验证解耦，利用规划执行模块 (PEM) 通过嵌入测试意图主动探测缺陷，并通过层级反射模块 (HRM) 用交互历史区分代理自身错误与真实软件缺陷；

**🔧 技术方法**

使用多模态大模型驱动的 GUI 代理（如 UI‑TARS、GUI‑Owl）、规划-执行协作、监视器、反射器以及可视化轨迹分析等技术；

**📊 数据集**

构建了 GUITestBench 数据集，共 26 个真实缺陷、12 个 Android 应用、143 条任务（含单/多动作缺陷），用于评估缺陷定位与报告能力；

**📈 对比分析**

与现有基线（GUI‑Owl、MAI‑UI、UI‑TARS、Mobile‑Agent‑V3）相比，GUITester 在 Pass@3 上的 F1 最高提升至 48.9%（UI‑TARS‑1.5‑7B），在 Pass@1 上也大幅优于 25% 以内的基线；

**⚠️ 局限性**

局限包括：对网络延迟等环境噪声的区分不足；监视器对期望状态预测的知识边界；动作空间有限，缺乏对复杂交互（如手势）的支持；以及缺陷覆盖仅限交互缺陷，未涵盖布局/视觉缺陷。

---

## IGenBench: Benchmarking the Reliability of Text-to-Infographic Generation

**arXiv ID:** 2601.04498 | [PDF](https://arxiv.org/pdf/2601.04498v1)

**作者:** Yinghao Tang `[一作]` (State Key Lab of CAD and CG, Zhejiang University), Wei Chen `[通讯]`

**关键词:** `Machine Learning` `Generation` `Transformer` `Large Language Model` `Text` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了首个评估文本到信息图生成可靠性的基准，构建了600个测试案例并设计了基于10类问题的自动化验证框架；

**💡 创新点**

创新点在于：①将信息图生成评估拆解为可解释的yes/no问题；②引入数据完整性、排序、编码三大专家驱动维度；③通过问答式验证实现Q-ACC与I-ACC双指标；

**🔧 技术方法**

技术手段主要是多模态大型语言模型（MLLM）用于提问生成、信息抽取和自动评估，实验中采用Gemini‑2.5‑Pro作为评估器；

**📊 数据集**

数据集来源于Statista、Visual Capitalist及ChartGalaxy等公开平台，经过聚类、去重和人工审核后得到600个多样化信息图案例；

**📈 对比分析**

对10款先进文本到图像模型进行评估，发现三阶性能层级：顶层Nanobanana‑Pro Q‑ACC 0.90、I‑ACC 0.49，第二层Seedream‑4.5与GPT‑Image‑1.5 Q‑ACC≈0.6但I‑ACC≤0.12，第三层其余模型Q‑ACC<0.5且I‑ACC≈0；

**⚠️ 局限性**

局限性包括：未评估美学质量；评估仅覆盖选定模型，且对高质量信息图仍需人工校验，提示模型在数据编码与完整性方面仍不足。

---

## Understanding Gaming the System by Analyzing Self-Regulated Learning in Think-Aloud Protocols

**arXiv ID:** 2601.04487 | [PDF](https://arxiv.org/pdf/2601.04487v1)

**作者:** Jiayi Zhang `[一作]` (University of Pennsylvania), Ryan S. Baker `[通讯]` (Adelaide University)

**关键词:** `Computers and Society` `Text`

### 📋 论文摘要

**🎯 论文内容**

对使用Stoichiometry Tutor智能辅导系统的10名学生进行思考大声协议记录和日志分析，比较其在游戏系统与非游戏系统情境下的认知参与和自我调节学习(SRL)过程。

**💡 创新点**

将游戏系统重新定义为一种“自我调节非学习”的恶劣SRL策略，并通过四阶段SRL编码与有序网络分析揭示游戏时的SRL失衡与非游戏时的有序调节流程。

**🔧 技术方法**

采用思考大声协议、SRL编码、广义线性混合模型、逻辑回归和有序网络分析(Ordered Network Analysis, ONA)等技术来量化和可视化游戏与非游戏情境下的语句长度、SRL类别出现率及其转移关系。

**📊 数据集**

使用来自Stoichiometry Tutor的交互日志与10名学生的思考大声协议（约401条发言），构成的混合数据集。

**📈 对比分析**

通过对比游戏与非游戏片段的发言字数（游戏平均105词，显著高于非游戏74词，IRR≈1.18）、SRL类别出现率（游戏时处理信息和错误识别显著高，规划显著低）以及ONA结果（游戏时SRL转移与四阶段模型相背离），表明游戏时的认知参与呈现“反向”或“被动”特征，非游戏时则更符合理论SRL序列。

**⚠️ 局限性**

样本量仅10人，思考大声协议受限于被试自我阐述范围，结果可能不适用于其他学科、平台或更广泛的学习者群体。

---

## SpeechMedAssist: Efficiently and Effectively Adapting Speech Language Models for Medical Consultation

**arXiv ID:** 2601.04638 | [PDF](https://arxiv.org/pdf/2601.04638v1)

**作者:** Sirry Chen `[一作]` (Fudan University), Zhongyu Wei `[通讯]` (Fudan University)

**通讯引用:** 5871 | **OpenAlex IDs:** https://openalex.org/A5011504177

**关键词:** `Computation and Language` `Domain Adaptation` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text` `Audio` `Multimodality` `Biomedical Data` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出SpeechMedAssist，一种能进行实时语音交互的医学SpeechLM，并提出两阶段训练法。

**💡 创新点**

利用文本注入医学知识与诊断技能，随后仅用约1万条语音数据完成模态对齐，显著降低语音数据需求。

**🔧 技术方法**

SpeechLM encoder–adaptor–LLM–decoder架构，文本注入训练，语音合成与对齐，域适应理论分析。

**📊 数据集**

构建TextMedDataset（405k）与SpeechMedDataset（198k）合成语音对话，并使用多种公开医学问答与对话集。

**📈 对比分析**

与基线ASR+LLM+TTS、SpeechLM、OmniLM在SpeechMedBench评测中，单轮问答、对话及真实环境均取得最高分。

**⚠️ 局限性**

仅支持文本与语音模态，中文为主，未涵盖多模态信息与其他语言。

---

## From National Curricula to Cultural Awareness: Constructing Open-Ended Culture-Specific Question Answering Dataset

**arXiv ID:** 2601.04632 | [PDF](https://arxiv.org/pdf/2601.04632v1)

**作者:** Haneul Yoo `[一作]` (Korea Advanced Institute of Science and Technology), Jiyoon Han `[通讯]`

**关键词:** `Computation and Language` `Large Language Model` `Supervised Fine-Tuning` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

通过一个多代理LLM框架，将韩国国家社会研究课程中的学习目标转化为开放式、文化特定的问答对，构建了约3.4万条多语言QA数据集。

**💡 创新点**

创新点在于利用国家课程作为结构化先验，自动化生成符合本土文化语境的监督数据，并通过多代理迭代与人工校验相结合，实现大规模、可复现的文化适配数据生成。

**🔧 技术方法**

采用多代理LLM管道，包括查询生成、文化过滤、释义与扩展、人工校验以及多语言翻译；使用GPT‑4、Gemini、Claude、Copilot等开放源LLM进行问答生成与评估，并用LLM‑as‑Judge自动评测质量。

**📊 数据集**

核心数据集为韩国2022年修订的基础教育社会研究课程（357个学习目标，158个学习结果），生成的34,128条QA对；对比的基准数据集为Korean LIMA（通用QA）。

**📈 对比分析**

通过主题建模与Jensen‑Shannon距离（JSD 0.807）比较文化特定查询与Korean LIMA的差异；使用可读性指标评估不同难度层级的回答；LLM‑as‑Judge在语言匹配上达到0.91准确率，文化适配与语言使用分别得分8.56与7.78；人工检查显示大多数实例符合学习目标，整体性能体现出高度的文化相关性与多样性。

**⚠️ 局限性**

局限性包括：仅以国家课程为文化先验，可能忽略更细腻的地缘文化差异；仅使用韩国首尔方言的课程，未覆盖朝鲜语区；课程本身可能包含偏见或争议，生成数据亦可能放大这些偏见；多语言翻译过程中语言偏好可能影响回答的忠实度与完整性。

---

## RecruitScope: A Visual Analytics System for Multidimensional Recruitment Data Analysis

**arXiv ID:** 2601.04630 | [PDF](https://arxiv.org/pdf/2601.04630v1)

**作者:** Xiyuan Zhu `[一作]`, Ran Wang `[通讯]`

**关键词:** `Human-Computer Interaction` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出了 RecruitScope，一套支持从行业层面到职位层面的多维度招聘数据可视化分析系统。

**💡 创新点**

创新点在于引入花形散点图（花瓣形 Glyph）与多视图交互，既能展示行业间位置关系，又能嵌入每个行业的教育、经验与薪酬三维属性。

**🔧 技术方法**

采用了基于 D3.js 等 JavaScript 可视化框架实现多视图联动、花瓣 Glyph 编码、Sankey、bidirectional bar、分布盒图等可视化技术，并进行数据预处理（归一化、异常值剔除）。

**📊 数据集**

使用了 2024 ChinaVis Challenge 提供的 430,664 条招聘记录数据集，涵盖 169,540 个职位、267,296 家公司、158 个行业，按 26 个省级和 371 个市级行政区划分。

**📈 对比分析**

通过案例研究对比 GP–EdD 组合、高薪职位与行业增长趋势等，展示系统能够在交互中即时更新与联动；虽然未给出具体性能指标，但案例表明响应速度足以满足实时分析需求。

**⚠️ 局限性**

局限性包括：对不同结构来源的数据泛化能力尚需验证；缺乏压缩摘要与非线性关系的直接可视化；学习曲线与整体可用性仍有提升空间。

---

## DeepHalo: A Neural Choice Model with Controllable Context Effects

**arXiv ID:** 2601.04616 | [PDF](https://arxiv.org/pdf/2601.04616v1)

**作者:** Shuhan Zhang `[一作]` (Chinese University of Hong Kong), Shuang Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 25601 | **OpenAlex IDs:** https://openalex.org/A5100415884

**关键词:** `Machine Learning` `Recommendation System` `Explainability and Interpretability` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出 DeepHalo 框架，对人类决策的上下文效应进行建模，并能控制交互的阶数，实现可解释且高预测性能的选择模型。

**💡 创新点**

创新点在于通过可换位等价的残差网络把不同阶数的交互逐层分离，给出特征化的上下文效应分解；并证明在特征无关时可逼近任意上下文选择函数。

**🔧 技术方法**

使用 permutation‑equivariant 的多层残差网络，配合多头线性投影与非线性变换实现不同阶交互；训练时采用 softmax 归一化与负对数似然优化。

**📊 数据集**

实验数据包括人工生成的 Halo 现象饮料市场份额数据、20 个候选项的高阶合成数据，以及真实数据集 Hotel、SFOwork、SFOshop、LPMC 交通和 Expedia 酒店选择。

**📈 对比分析**

与 MNL、MLP、CMNL、TasteNet、RUMnet、ResLogit、FateNet、TCNet 等基线模型比较；DeepHalo 在所有数据集上负对数似然均最低，尤其在高阶交互场景下深度提升显著；低阶约束模型比后置截断的 MLP 更稳健。

**⚠️ 局限性**

局限性包括：参数规模随交互阶数指数增长，导致大规模选择集的计算成本高；模型仅能捕捉可分解交互形式，可能无法完整刻画所有行为；解释性仍依赖层级分解，需要更多案例验证。

---

## Density Matrix RNN (DM-RNN): A Quantum Information Theoretic Framework for Modeling Musical Context and Polyphony

**arXiv ID:** 2601.04592 | [PDF](https://arxiv.org/pdf/2601.04592v1)

**作者:** Joonwon Seo `[一作]` (Georgia State University), Mariana Montiel `[通讯]` (Georgia State University)

**通讯引用:** 148 | **OpenAlex IDs:** https://openalex.org/A5062698749

**关键词:** `Machine Learning` `Recurrent Neural Network` `Audio`

### 📋 论文摘要

**🎯 论文内容**

提出了基于密度矩阵的循环神经网络（DM‑RNN），用量子信息论框架来建模音乐上下文和多声部的多义性与相干性。

**💡 创新点**

创新点在于将密度矩阵作为隐藏状态，利用完全正迹保守（CPTP）映射保证物理可行性，并通过Choi‑Jamiołkowski 同构实现参数化，同时引入冯·诺依曼熵与量子互信息作为音乐不确定性与相关性的量化工具。

**🔧 技术方法**

采用量子信息论（密度矩阵、CPTP通道、Choi矩阵、Kraus算子）、矩阵分析（谱分解、冯·诺依曼熵、量子互信息）以及正交投影（POVM）等技术。

**📊 数据集**

未给出具体数据集；本文主要为理论框架设计。

**📈 对比分析**

未进行实验比较或性能评估，文中仅说明若实现可行，则可通过熵与互信息评估模型的多义性捕捉与多声部相关性。

**⚠️ 局限性**

主要限制是隐藏状态维度升至 $O(d^2)$，参数量与计算复杂度急剧增加，需使用张量网络（如 MPO/MPDO）等压缩技巧才能在实际中实现。

---

## Scaling Behavior Cloning Improves Causal Reasoning: An Open Model for Real-Time Video Game Playing

**arXiv ID:** 2601.04575 | [PDF](https://arxiv.org/pdf/2601.04575v1)

**作者:** Yuguang Yue `[一作]` (Player2), Jonathan J Hunt `[通讯]` (Player2)

**关键词:** `Artificial Intelligence` `Transformer` `Video` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

训练并公开了一个可在消费级 GPU 上实时运行的多模态游戏玩耍基础模型 Pixels2Play（P2P），使用行为克隆从 8300 小时高质量人类游戏数据中学习键盘与鼠标动作；

**💡 创新点**

提出了结合文本指令、动作解码器和“思考”令牌的 transformer 结构，并系统研究了行为克隆的尺度律与因果推理，证明模型与数据规模的提升能显著提高因果性和性能；

**🔧 技术方法**

使用 decoder‑only transformer、EfficientNet 图像编码器、动作自回归解码器、数据增强、滑动窗口注意力、训练‑推理输入一致性和截断正态分布的鼠标动作采样等技术；

**📊 数据集**

主要数据集是 8300+ 小时的 3D 游戏人类游戏录像（650M 画面‑动作对），并补充 1% 的人类纠正轨迹、无标签视频以及两款自制程序化游戏用于基准；

**📈 对比分析**

通过程序化环境（Hovercraft、Simple‑FPS）指标、四款真实游戏的人工偏好评估、文本指令跟随实验以及对测试损失和因果性得分的量化，结果显示 12 亿参数模型在大多数任务上可与人类相当，因果得分随规模与数据增长提升；

**⚠️ 局限性**

局限包括：对文本指令的推理延迟导致帧率略低；模型仍受视频压缩与分辨率差异影响，需进一步缩小训练‑推理差距；对连续动作空间的离散化可能降低高精度控制能力。

---

## When Tone and Words Disagree: Towards Robust Speech Emotion Recognition under Acoustic-Semantic Conflict

**arXiv ID:** 2601.04564 | [PDF](https://arxiv.org/pdf/2601.04564v1)

**作者:** Dawei Huang `[一作]` (Inclusion AI), Xiaojiang Peng `[通讯]` (Shenzhen University)

**关键词:** `Sound` `Classification` `Recognition` `Transformer` `Supervised Fine-Tuning` `Audio` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出一种Fusion Acoustic‑Semantic（FAS）框架，显式分离声学与语义路径并通过查询‑注意力融合，以解决语音情感识别中的声学‑语义冲突问题。

**💡 创新点**

创新点包括：1）将语音分为低维声学Token与高维语义特征；2）引入可学习查询的轻量注意力模块实现跨模态融合；3）发布专门评估声学‑语义冲突的CASE基准数据集。

**🔧 技术方法**

采用Whisper‑large提取语义特征、MingTok‑Audio或其他音频Tokenizer提取声学Token，随后利用可学习查询的注意力机制进行融合，并在此基础上训练轻量MLP分类头。

**📊 数据集**

训练数据包含IEMOCAP、MELD、RAVDESS、ESD等多语种语料；零样本评测使用CASE、Emo‑Emilia、EMOVO、EmoDB等。

**📈 对比分析**

与多种基线（SSL模型、语义编码器、音频Tokenizer、ALM等）对比，FAS在in‑domain ACC≈71.9%，在零样本CASE ACC为59.38%，在多数据集上均显著优于现有SOTA。

**⚠️ 局限性**

局限包括CASE样本量不足、语言覆盖有限、未纳入会话上下文或说话者身份等额外信息，导致在更复杂真实场景中的表现尚待验证。

---

## Improving Semi-Supervised Contrastive Learning via Entropy-Weighted Confidence Integration of Anchor-Positive Pairs

**arXiv ID:** 2601.04555 | [PDF](https://arxiv.org/pdf/2601.04555v1)

**作者:** Shogo Nakayama `[一作]` (Doshisha University), Masahiro Okuda `[通讯]` (Doshisha University)

**通讯引用:** 4585 | **OpenAlex IDs:** https://openalex.org/A5025207272

**关键词:** `Machine Learning` `Classification` `Representation Learning` `Optimization` `Convolutional Neural Network` `Contrastive Learning` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出了一种基于熵的自适应加权的半监督对比学习损失函数，能够对未标记样本的置信度进行估计并给出伪标签，即使置信度低于阈值也能参与训练

**💡 创新点**

创新点在于：①将锚点与正样本的置信度共同考虑，通过几何平均加权来计算对比损失；②利用熵来评估样本置信度并进行样本选择与权重调节；③实现了对所有未标记样本的全局利用，而非仅限于置信度高的样本

**🔧 技术方法**

使用的技术包括：自监督强数据增强、弱增强、对比学习损失、熵计算、基于熵的自适应加权、Cosine 学习率调度和WideResNet-28-2网络结构

**📊 数据集**

在CIFAR-10和CIFAR-100两个公开数据集上进行实验，分别在每类4个和25个标签样本的半监督设置下评估

**📈 对比分析**

与传统Semi‑Supervised Contrastive Learning基线进行比较，实验结果显示：在CIFAR‑10每类4个标签时略有提升，在25个标签时略低，但在CIFAR‑100两种标签稀缺场景下均实现了准确率提升（最高约1%~2%）

**⚠️ 局限性**

主要局限包括：仅在两个数据集上验证，缺乏对其他数据集和不同随机种子的泛化评估；熵阈值和权重设定需经验调优，可能影响实际部署

---

## Exploring Recommender System Evaluation: A Multi-Modal User Agent Framework for A/B Testing

**arXiv ID:** 2601.04554 | [PDF](https://arxiv.org/pdf/2601.04554v1)

**作者:** Wenlin Zhang `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 5727 | **OpenAlex IDs:** https://openalex.org/A5100645854

**关键词:** `Information Retrieval` `Recommendation System` `Large Language Model` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了基于多模态大语言模型的用户代理框架，用于在多页面、多模态的推荐沙盒环境中进行A/B测试；

**💡 创新点**

创新点在于：①构建了与真实平台UI相似的多页面多模态沙盒；②利用LLM实现用户感知、记忆与疲劳系统，实现更逼真的用户决策；③通过生成模拟交互数据进行数据增强，提升推荐模型性能；

**🔧 技术方法**

核心技术包括：大语言模型（GPT‑4o、Gemini‑2.5‑Flash）、CLIP视觉编码、text‑embedding‑3‑small文本检索、记忆模块与疲劳系统，配合深度学习推荐算法（DeepFM、FM、AFN等）；

**📊 数据集**

使用扩展后的MM‑ML‑1M数据集（MovieLens‑1M 加入海报、概述、元信息），并在该数据集上搭建沙盒；

**📈 对比分析**

对比方法：在沙盒环境下测量CTR、CVR、AR，并与真实数据的Recall@20、NDCG对齐；实验表明代理能准确重现模型排名、数据规模效应及特征重要性，并通过模拟数据提升AUC 0.002‑0.003；

**⚠️ 局限性**

局限性：仅在封闭沙盒内评估，未考虑社交、外部信息；LLM可能产生幻觉或重复行为；对跨平台普适性与大规模部署仍有待验证。

---

## Discrete Fourier Transform-based Point Cloud Compression for Efficient SLAM in Featureless Terrain

**arXiv ID:** 2601.04551 | [PDF](https://arxiv.org/pdf/2601.04551v1)

**作者:** Riku Suzuki `[一作]` (Tohoku University), Kazuya Yoshida `[通讯]` (Tohoku University)

**通讯引用:** 13030 | **OpenAlex IDs:** https://openalex.org/A5023419492

**关键词:** `Robotics` `Compression` `Simultaneous Localization and Mapping` `Point Cloud`

### 📋 论文摘要

**🎯 论文内容**

提出了一种基于离散傅里叶变换（DFT）的点云压缩方法，通过将点云投影为数字高程模型（DEM），再对二维图像进行DFT、低通滤波（LPF）并逆变换重建，适用于平坦或缓坡地形的SLAM点云压缩。

**💡 创新点**

创新点在于将点云转换为频域图像后，仅保留低频成分进行压缩，利用高频成分对平缓地形影响小的特点；同时阐明了裁切频率与压缩误差之间的关系，并提出了仅保留内圆非零频率值的压缩策略。

**🔧 技术方法**

主要技术包括点云投影生成DEM、二维DFT与逆DFT、低通滤波、基于cutoff ratio的频率阈值设定、RMSE和bits-per-point评估指标。

**📊 数据集**

使用了Mars-like MADMAX数据集中的两种地形：几乎平坦的沙漠地形和包含岩石起伏的崎岖地形。

**📈 对比分析**

通过将压缩后重建的点云与原始点云计算RMSE，并记录bits-per-point，比较不同cutoff ratio（0.8、0.95）下的压缩率和误差。结果显示，平坦地形在cutoff 0.8时误差极低，bits-per-point显著下降；在崎岖地形误差稍大；cutoff超过0.9时误差急剧上升，表明压缩率与误差存在权衡。

**⚠️ 局限性**

局限性包括：方法仅适用于平缓地形，对崎岖或高频细节丰富的地形压缩效果差；未验证实时性能和对重新定位的影响；缺乏自动或自适应选择合适cutoff ratio的机制。

---

## GEnSHIN: Graphical Enhanced Spatio-temporal Hierarchical Inference Network for Traffic Flow Prediction

**arXiv ID:** 2601.04550 | [PDF](https://arxiv.org/pdf/2601.04550v1)

**作者:** Zhiyan Zhou `[一作]` (Beijing Normal University), Ziai Wang `[通讯]` (Beijing Normal University)

**关键词:** `Machine Learning` `Graph Neural Network` `Recurrent Neural Network` `Transformer` `Time Series` `Graph`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于图增强的时空层次推理网络GEnSHIN，用于交通流量预测。

**💡 创新点**

创新点包括注意力增强的GCRU单元、基于记忆池的双重嵌入异构图生成以及解码阶段的轻量级动态图更新。

**🔧 技术方法**

使用图卷积递归单元、Transformer自注意力、记忆网络与动态图更新等深度学习技术。

**📊 数据集**

在公开METR-LA（洛杉矶高速公路）数据集上进行实验。

**📈 对比分析**

与HA、STGCN、DCRNN、STTN、AGCRN、CCRNN等基线比较，GEnSHIN在MAE、RMSE、MAPE上均取得最优或接近最优的表现。

**⚠️ 局限性**

仅在单一数据集验证，模型复杂度较高，未来需在多域数据上评估并降低计算成本。

---

## UniBiDex: A Unified Teleoperation Framework for Robotic Bimanual Dexterous Manipulation

**arXiv ID:** 2601.04629 | [PDF](https://arxiv.org/pdf/2601.04629v1)

**作者:** Zhongxuan Li `[一作]` (University of Hong Kong), Peng Zhou `[通讯]` (Great Bay University)

**关键词:** `Robotics` `Robotic Intelligence` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

开发了一套统一的双臂遥控框架UniBiDex，可同时支持VR与跟随臂输入，实现实时、无碰撞的双臂协同操作。

**💡 创新点**

将异构输入统一到同一运动学模块，采用零空间控制吸引至预设参考姿态，实现跨模态安全、可扩展的双臂协同。

**🔧 技术方法**

多模态输入预处理、逆运动学优化、零空间控制、基于电流的力反馈和多通道触觉反馈。

**📊 数据集**

使用用户演示的厨房整理长序任务数据，收集40次VR/跟随臂对照实验；未公开公开数据集。

**📈 对比分析**

与Naive VR与Naive LF两种基线对照，比较任务成功率和完成时间；UniBiDex在VR模式成功率60%并显著减少完成时间，LF模式成功率75%。

**⚠️ 局限性**

仍难处理柔性物体折叠与滑移，缺乏高级触觉感知，且依赖预先设定的参考姿态。

---

## THaLLE-ThaiLLM: Domain-Specialized Small LLMs for Finance and Thai -- Technical Report

**arXiv ID:** 2601.04597 | [PDF](https://arxiv.org/pdf/2601.04597v1)

**作者:** KBTG Labs `[一作]` (KASIKORN Business Technology Group), Monchai Lertsutthiwong `[通讯]`

**关键词:** `Computation and Language` `Optimization` `Safty and Privacy` `Computational Efficiency` `Recommendation System` `Domain Adaptation` `Data-Centric Learning` `Anomaly Detection` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text` `Multimodality` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

本文通过模型权重合并的方法，将多任务专用LLM（如泰语能力模型和金融专业模型）合并为单一高性能多能模型，并在多项泰语与金融基准上验证其有效性。

**💡 创新点**

创新点在于提出低成本、无需GPU即可实现的线性合并框架，使得可独立训练的任务模型能够快速组合为统一模型，并通过合并提升多领域表现。

**🔧 技术方法**

使用的技术包括低秩自适应（LoRA）微调、超参数微调（SFT）、模型权重线性插值合并（MergeKit）以及多模态安全提示策略。

**📊 数据集**

主要数据集涵盖泰语学术考试（O‑NET M3/M6）、金融专业考核（Flare‑CFA、Thai‑IC）以及安全与输出一致性测试（ThaiSafetyBench、IFEval‑TH）。

**📈 对比分析**

通过对比基准模型Qwen3‑8B、ThaiLLM‑8B和THaLLE‑Finance‑8B的单独表现与合并后模型的分数，发现合并模型在O‑NET、CFA、IC上分别提升约12.6%、5.7%和40%，且在安全与一致性评测中也显著优于单体模型。

**⚠️ 局限性**

局限性包括合并后模型在推理模式下安全性提升有限、合并策略仅为线性平均，未尝试更复杂的融合方法，以及缺乏进一步微调以充分挖掘潜在性能。

---

## Feel the Presence: The Effects of Haptic Sensation on VR-Based Human-Robot Interaction

**arXiv ID:** 2601.04596 | [PDF](https://arxiv.org/pdf/2601.04596v1)

**作者:** Xinyan Yu `[一作]` (University of Sydney), Martin Tomitsch `[通讯]` (University of Technology Sydney)

**通讯引用:** 2924 | **OpenAlex IDs:** https://openalex.org/A5023076293

**关键词:** `Human-Computer Interaction` `Robotic Intelligence` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

本研究在虚拟现实（VR）中对比使用传统手柄与HaptX触觉手套进行人机交互（HRI）实验，重现城市机器人求助场景，并评估两种交互方式对存在感、身体参与度及对机器人的情感评价的影响。

**💡 创新点**

创新点在于首次将高度逼真的触觉手套嵌入VR HRI实验，系统考察触觉反馈对自我存在感、社会存在感以及情感相关评价的提升，并通过定量与定性方法揭示触觉交互促进自然身体行为的机制。

**🔧 技术方法**

采用的技术包括HTC Vive头显、Unity3D引擎搭建VR环境、HaptX微流控技术的触觉手套提供触感与力感反馈，以及实验后问卷与半结构访谈收集数据。

**📊 数据集**

实验使用的“数据集”为两组参与者的实验记录：手柄组24人、手套组12人，实验情境为三种机器人求助方式（语音、情感表达、游戏化求助），每人多次体验并完成多维度评估量表。

**📈 对比分析**

比较方法采用独立样本t检验（存在感子量表）、线性混合模型（机器人评价量表）以及效应量（Cohen’s d）进行统计，结果显示触觉手套显著提升社会存在感（d≈0.86）与自我存在感（d≈1.13），并显著提升机器人亲和度与情感信任（p<0.05）。

**⚠️ 局限性**

局限性包括样本量较小且两组人数不平衡、触觉手套的重量和束线可能对动作自由度产生影响、实验仅限于单一城市求助情境，无法全面反映不同HRI场景下触觉反馈的通用性。

---

## Lenses for Partially-Specified States (Extended Version)

**arXiv ID:** 2601.04573 | [PDF](https://arxiv.org/pdf/2601.04573v1)

**作者:** Kazutaka Matsuda `[一作]` (Tohoku University), Meng Wang `[通讯]` (University of Bristol)

**通讯引用:** 90676 | **OpenAlex IDs:** https://openalex.org/A5101694733

**关键词:** `Programming Languages`

### 📋 论文摘要

**🎯 论文内容**

提出一种新的双向变换框架——P‑lens（Partially‑Specified Lens），通过在源/视图状态中加入部分确定性（poset）与更新意图，解决多个共享源的视图更新冲突，并保证用户的更新意图被保留。

**💡 创新点**

创新点包括：
• 将状态和更新视为同一类对象，引入部分确定性和更新意图的偏序，能够对冲突更新进行自然合并；
• 设计了新的弱一致性、可接受性、稳定性三条性质，支持可组合的良好性保证；
• 在保持经典 Lens 的可组合性与可重用性的基础上，兼容传统 Lens、CRDT 以及操作式更新等多种视角；
• 提供完整的形式化定义与证明，并给出 Haskell 原型与 Agda 机械证明。

**🔧 技术方法**

技术手段主要包括：
• 定义“可复制域（duplicable）”与合并算子 ⊕，实现部分确定性状态的 join；
• 在 Lens 的 get/put 上引入偏序 ≤ 与同等更新关系 ⊑；
• 证明弱一致性、可接受性、稳定性闭包，并基于这些性质构造可组合的良好性；
• 通过实例化 2P‑Set CRDT、映射/集合等域，演示如何实现任务列表、过滤器等典型用例；
• 利用 Agda 进行机理化证明，并实现 Haskell 版原型。

**📊 数据集**

本工作未在公开数据集上进行实验评测；示例使用了简化的任务列表、集合和映射等小型人工构造域。

**📈 对比分析**

由于缺乏实验数据，本文未给出性能对比；重点在理论设计与形式化验证，作者强调在可组合性与更新意图保留上的优势。

**⚠️ 局限性**

局限性：
• 需要为每种域手工定义偏序、同等更新关系与合并算子；
• 对于更复杂的数据结构（如链表、树）仍需设计新的基本 Lens；
• 在实践中实现部分确定性状态的存储与合并可能产生额外开销；
• 目前未对大规模并发多视图场景的性能进行评估。

---

## Neurosymbolic Retrievers for Retrieval-augmented Generation

**arXiv ID:** 2601.04568 | [PDF](https://arxiv.org/pdf/2601.04568v1)

**作者:** Yash Saxena `[一作]` (University of Maryland Baltimore County), Manas Gaur `[通讯]` (University of Maryland Baltimore County)

**关键词:** `Artificial Intelligence` `Retrieval` `Generation` `Retrieval-Augmented Generation` `Text` `Biomedical Data`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了 Neurosymbolic Retrieval Augmented Generation (RAG) 框架，通过将知识图谱与符号推理与神经检索融合，提升检索透明度与临床决策的可靠性。

**💡 创新点**

创新点在于三种方法：使用可解释的符号特征调制检索嵌入的 MAR、基于 KG 的路径检索 KG‑Path RAG 以及基于流程知识的 Proknow‑RAG。

**🔧 技术方法**

技术主要包括调制网络、知识图谱推理、BFS 多跳检索、图基排序损失以及与 LLM 的检索-生成耦合。

**📊 数据集**

数据集使用了 IMHI 公开的心理健康检测数据以及临床对话样本，进行多任务评估。

**📈 对比分析**

与基线 MentalLLAMA‑33B 和其他 RAG 变体比较，KG‑Path RAG 在抑郁、焦虑、PTSD 等任务上取得最高准确率；Proknow‑RAG 在自杀风险检测上领先，整体性能优于基线。

**⚠️ 局限性**

限制包括对知识图谱质量的高度依赖、跨模态扩展有限以及在实时临床部署时计算成本仍需进一步优化。

---

## All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction

**arXiv ID:** 2601.04567 | [PDF](https://arxiv.org/pdf/2601.04567v1)

**作者:** Ziyou Jiang `[一作]` (Chinese Academy of Sciences), Qing Wang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 56954 | **OpenAlex IDs:** https://openalex.org/A5100434847

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Anomaly Detection` `Transformer` `Large Language Model` `Prompt Engineering` `Multimodality` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了基于设计概念再现的不断变异有害 meme 检测方法（RepMD），通过构建设计概念图（DCG）引导多模态大语言模型（MLLM）识别有害 meme；

**💡 创新点**

创新点在于将攻击树理念引入 meme 设计，定义可解释的 DCG 结构，利用其再现步骤和逻辑门进行步骤化引导，从而提升对类型转移与时间演化 meme 的检测泛化能力；

**🔧 技术方法**

技术主要包括：失败原因树构建、DCG 生成与 SVD 归约剪枝、基于 DCG 的检索式引导、以及多模态大语言模型的提示优化与投票机制；

**📊 数据集**

使用了 GOAT‑Bench（约54,192 条 meme，涵盖多种攻击类型）和自采 Twitter（2025 年四季各 500 条，4,000 条）构成的两大数据集；

**📈 对比分析**

与三种基线（RAG、LoRA 微调、MIND 等）以及四个主流 MLLM 进行对比，RepMD 在类型转移任务中达 81.1% 准确率，较基线提升 9–10% 以上，在时间演化任务中平均提升 13.7% F1/14.3% 准确率，且跨域下降幅度更小；

**⚠️ 局限性**

主要局限包括对极简视觉特征的识别不足、LLM 产生幻觉导致的误判，以及对全新设计概念的适应性仍有限。

---

## A Vision for Multisensory Intelligence: Sensing, Synergy, and Science

**arXiv ID:** 2601.04563 | [PDF](https://arxiv.org/pdf/2601.04563v1)

**作者:** Paul Pu Liang `[一作]` (Massachusetts Institute of Technology), Paul Pu Liang `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 7789 | **OpenAlex IDs:** https://openalex.org/A5086233510

**关键词:** `Machine Learning` `Reinforcement Learning` `Meta Learning` `Multimodality` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

综述并提出未来十年多感官人工智能的研究愿景与挑战。

**💡 创新点**

将多模态AI从数字媒介扩展到人类感官、物理与社会环境，提出感知、科学与协同三大主题，并详细列举关键技术路线。

**🔧 技术方法**

讨论了多传感器采集、统一表征、跨模态对齐、跨模态推理与生成、元学习、注意力、图模型、强化学习等技术。

**📊 数据集**

作为综述性论文，未使用具体数据集，主要以现有公开资源和案例作为示例。

**📈 对比分析**

本论文不包含实验对比与性能评估，更多聚焦于问题定义与研究方向。

**⚠️ 局限性**

主要局限在于技术与方法尚未成熟，存在模态不匹配、跨模态协同难度大、实时流式生成、资源与安全约束等挑战，且缺乏系统化评测与标准化基准。

---

## Reasoning Over Space: Enabling Geographic Reasoning for LLM-Based Generative Next POI Recommendation

**arXiv ID:** 2601.04562 | [PDF](https://arxiv.org/pdf/2601.04562v1)

**作者:** Dongyi Lv `[一作]` (Xi’an Jiaotong University), Mu Xu `[通讯]` (Amap, Alibaba Group)

**关键词:** `Artificial Intelligence` `Recommendation System` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Chain-of-Thought` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一种名为ROS的生成式下一位置推荐框架，利用层级空间语义ID（SID）和三阶段空间链式思考（Mobility CoT）实现对地理信息的显式推理，并通过空间引导强化学习进一步优化模型。

**💡 创新点**

创新点在于：1）将地理位置转化为可解释的层级空间语义ID；2）设计三阶段Mobility CoT，将个性化、意图空间构建与基于位置的精炼剪枝结合；3）采用空间引导的强化学习，将地理可行性、SID正确性与推理格式统一奖励。

**🔧 技术方法**

主要技术包括：大语言模型（LLM）生成式推荐、S2细胞ID、RQ-VAE量化语义嵌入、两阶段文本↔SID对齐预训练、三阶段Chain‑of‑Thought训练、空间引导的强化学习（GRPO）以及基于Haversine距离的奖励设计。

**📊 数据集**

使用了三个公开的基于位置的社交网络数据集：Foursquare‑NYC、Foursquare‑TKY和Gowalla‑CA。

**📈 对比分析**

与传统、神经网络和其他LLM基线相比，ROS在HR@1上分别提升约11.2%、11.0%和15.7%，在三大数据集上均优于最强对手，并且在跨城市迁移、历史长度扩展以及Top‑K评估中表现出更高的准确率和更好的地理可行性。

**⚠️ 局限性**

局限性在于对时空可行性的约束过于简单，主要基于“距离越大越不可能”的单调假设，未充分考虑交通网络（如地铁、高速）导致的快速长距离移动；未来可加入网络感知的时间/路径成本等软约束。

---

## 4D-ARE: Bridging the Attribution Gap in LLM Agent Requirements Engineering

**arXiv ID:** 2601.04556 | [PDF](https://arxiv.org/pdf/2601.04556v1)

**作者:** Bo Yu `[一作]` (Tencent), Lei Zhao `[通讯]` (Tencent)

**通讯引用:** 2622 | **OpenAlex IDs:** https://openalex.org/A5103197618

**关键词:** `Software Engineering` `Optimization` `Explainability and Interpretability` `Recommendation System` `Data-Centric Learning` `Finance Related` `Transformer` `Large Language Model` `Prompt Engineering` `Chain-of-Thought` `Agentic AI` `Tabular` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

本文提出4D-ARE方法，系统化设计决策支持型LLM代理的要求工程，填补了设计时如何指定代理推理内容的空白。

**💡 创新点**

创新点在于将因果归因拆解为四维（结果、过程、支持、长期），并通过五层架构（问题清单、归因模型、数据映射、双轨逻辑、边界约束）将领域专业知识转化为可直接编译为系统提示的YAML规范。

**🔧 技术方法**

采用ReAct思考-行动框架、Chain‑of‑Thought推理、提示工程以及工具调用等技术，在提示层面实现可审计的因果推理链。

**📊 数据集**

使用一家商业银行的运营与CRM数据库、核心业务系统及手工报告数据，在工业试点中构建性能归因代理。

**📈 对比分析**

与传统的无结构提示相比，4D-ARE在20个典型查询上提升了维度覆盖率约217%、因果因子识别约180%、可操作建议约500%，同时将边界违规率降低95%；整体响应长度略增但可解释性和合规性显著提升。

**⚠️ 局限性**

局限性包括：试点仅在单一金融机构单一域内完成，缺乏严谨的多样化实验与统计检验；对领域专家的依赖与数据映射的高实现成本；归因链为静态，需定期维护；模型依赖性导致不同LLM表现可能差异。

---

## Limited Math: Aligning Mathematical Semantics with Finite Computation

**arXiv ID:** 2601.04634 | [PDF](https://arxiv.org/pdf/2601.04634v1)

**作者:** Lian Wen `[一作]` (Griffith University), Lian Wen `[通讯]` (Griffith University)

**通讯引用:** 463 | **OpenAlex IDs:** https://openalex.org/A5012316456

**关键词:** `Logic in Computer Science`

### 📋 论文摘要

**🎯 论文内容**

提出了“有限数学”（Limited Math）框架，使数学语义与有限计算对齐；

**💡 创新点**

通过显式化数值大小、精度和结构的上限、定义确定性的映射运算符，并将语义与受限求值分离，实现了在有限域上的闭合算术和可分析的边界行为；

**🔧 技术方法**

采用有限数值域、取值映射算子、集合大小限制、运算符/函数映射以及有限状态机的执行语义等形式化技术；

**📊 数据集**

本工作为理论性框架，未使用任何实验数据集；

**📈 对比分析**

通过与浮点、定点算术的概念对比，讨论了在有限范围内保持的代数性质与边界处的确定偏差，未给出实际性能实验；

**⚠️ 局限性**

缺乏对实际实现的实验验证；在边界时代数律失效；对极限、导数等分析算子只能外部映射，框架表达力有限；

---

## Beyond the "Truth": Investigating Election Rumors on Truth Social During the 2024 Election

**arXiv ID:** 2601.04631 | [PDF](https://arxiv.org/pdf/2601.04631v1)

**作者:** Etienne Casanova `[一作]` (California Institute of Technology), R. Michael Alvarez `[通讯]` (California Institute of Technology)

**通讯引用:** 10563 | **OpenAlex IDs:** https://openalex.org/A5091187284

**关键词:** `Artificial Intelligence` `Classification` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

构建了近1500万条Truth Social帖子的大规模数据集，开发多阶段谣言检测代理，并系统分析了选举谣言的传播与心理动态。

**💡 创新点**

创新点在于将RoBERTa微调、关键词过滤与GPT‑4o mini双步验证结合，实现高精度谣言标注，同时首次提供基于曝光次数的“虚假真理效应”实证与网络级扩散仿真。

**🔧 技术方法**

使用RoBERTa深度学习分类器、关键词过滤器、GPT‑4o mini LLM验证、网络传播模型、曝光‑转发概率估计与阈值级联仿真等技术。

**📊 数据集**

数据集为约200,000名用户、1500万条帖子（其中约10万条谣言）以及190,000条用户资料的Truth Social抓取数据，约33%用户成功地理定位到州级。

**📈 对比分析**

通过人工验证的500条样本对比，检测代理实现94.6%准确率、100%召回率、89%阳性预测值，并将LLM调用量降低93%；在模拟中显示谣言可在四轮内感染约25%用户。

**⚠️ 局限性**

局限包括仅覆盖约1/5的活跃用户、仅33%用户可地理定位、缺乏跨平台对比、影响力指标未去重、模型对细微谣言可能失误以及对更大规模平台的推广性未知。

---

## Adaptive Retrieval for Reasoning-Intensive Retrieval

**arXiv ID:** 2601.04618 | [PDF](https://arxiv.org/pdf/2601.04618v1)

**作者:** Jongho Kim `[一作]` (Seoul National University), Moontae Lee `[通讯]` (LG AI Research)

**关键词:** `Information Retrieval` `Retrieval` `Large Language Model` `Reinforcement Learning` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出一种基于规划的适配检索框架，利用中间推理步骤作为信号动态扩展检索候选，以解决推理密集检索中的桥接文档检索不足。

**💡 创新点**

创新点在于将推理步骤转化为可解释的密集奖励信号，用于在检索过程中进行中途修正，并实现推理重排器与邻域检索的紧密协作。

**🔧 技术方法**

技术包括轻量级 LLM 推理重排器、基于规划的推理步骤抽取、基于 Bradley‑Terry 模型的步级奖励计算以及邻域感知自适应检索。

**📊 数据集**

实验使用 BRIGHT 多跳检索基准以及 HotpotQA、2WIKI、MusiQue 等多跳问答数据集。

**📈 对比分析**

与 BM25、RankZephyr、ReasonRank 等基线比较，本文方法在 BRIGHT 上 nDCG@10 提升约5.6个百分点，在多跳 QA 上 EM 与 F1 也均超过所有对手。

**⚠️ 局限性**

局限性包括依赖于重排器生成的规划步骤质量、缺乏通用的规划步骤对齐方法以及对初始规划错误的敏感性。

---

## Enhancing Multimodal Retrieval via Complementary Information Extraction and Alignment

**arXiv ID:** 2601.04571 | [PDF](https://arxiv.org/pdf/2601.04571v1)

**作者:** Delong Zeng `[一作]` (Sun Yat-sen University), Ying Shen `[通讯]` (Sun Yat-sen University)

**通讯引用:** 5157 | **OpenAlex IDs:** https://openalex.org/A5074799043

**关键词:** `Artificial Intelligence` `Retrieval` `Transformer` `Contrastive Learning` `Vision Language Model` `Image` `Text` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

本文提出一种基于补充信息提取与对齐的多模态检索框架 CIEA，能够将文本和图像映射到统一潜在空间并保留图像中未被文本覆盖的补充信息。

**💡 创新点**

创新点在于构建了补充信息提取器，通过计算图像补丁与文本的差异并对图像向量进行重加权，同时使用两种互补对比损失来保证语义完整性与补充信息的充分利用。

**🔧 技术方法**

使用技术包括文本编码器 T5‑ANCE、CLIP 视觉编码器、线性投影、注意力加权、Transformer 融合以及双重对比学习。

**📊 数据集**

实验数据集包括 WebQA‑Multi、WebQA‑Image 以及 EDIS。

**📈 对比分析**

与基于分而治之的检索模型（BM25、CLIP‑DPR、VinVL‑DPR）和通用密集检索模型（UniVL‑DR、MARVEL、T5‑ANCE）对比，CIEA 在 MRR、NDCG、Recall 等指标上均取得显著提升，例如 WebQA‑Multi 上 MRR@10 由 65.43 提升到 66.16。

**⚠️ 局限性**

局限性在于仅处理文本查询和图像模态，未扩展到音频、视频等多模态输入，且受限于计算资源未尝试更大规模的语言模型。

---

## Industrial Data-Service-Knowledge Governance: Toward Integrated and Trusted Intelligence for Industry 5.0

**arXiv ID:** 2601.04569 | [PDF](https://arxiv.org/pdf/2601.04569v1)

**作者:** Hailiang Zhao `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**通讯引用:** 9000 | **OpenAlex IDs:** https://openalex.org/A5055284175

**关键词:** `Computational Engineering, Finance, and Science` `Multimodality` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

提出并阐述了Trusted Industrial Data-Service Governance（TIDSG）框架，将数据治理、服务治理与知识治理整合为端到端可信治理模型，以支持工业生态系统中的跨行业可信价值链

**💡 创新点**

创新点在于将三大治理层次统一为一体化框架，强调质量、安全、隐私、公平、可解释性，并通过软硬件安全机制实现跨行业互操作性

**🔧 技术方法**

采用了标准化方案（如Asset Administration Shell、工业数字孪生）、语义模型（本体、知识图谱）以及基于策略的治理机制

**📊 数据集**

文中未给出具体公开数据集，主要基于工业多模态数据（传感器时间序列、日志、ERP等）进行理论分析和案例推演

**📈 对比分析**

未进行实验性对比，文章以理论与案例为主，未给出性能指标或评估结果

**⚠️ 局限性**

局限性包括缺乏实证验证、实施成本与复杂度高、跨行业标准一致性与治理实践的可落地性问题

---

## Identifying Good and Bad Neurons for Task-Level Controllable LLMs

**arXiv ID:** 2601.04548 | [PDF](https://arxiv.org/pdf/2601.04548v1)

**作者:** Wenjie Li `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 29512 | **OpenAlex IDs:** https://openalex.org/A5081036622

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出基于生物功能拮抗原则的任务级神经元归因框架NeuronLLM，利用AQUA模块生成答案置换代理问题，CNI模块采用交叉熵对比评分识别支持与抑制神经元并实现对LLM的控制与解释。

**💡 创新点**

①将功能拮抗引入LLM神经元分析，系统考虑支持与抑制两类神经元；②AQUA通过答案置换消除偶然正确的影响；③CNI的ACE评分结合正负贡献实现对比归因；④可将现有归因方法嵌入框架进一步提升效果。

**🔧 技术方法**

交叉熵对比评分（ACE）、梯度归因、代理问题生成、神经元干预实验（silence/excite），并用RAC/RCC指标评估性能。

**📊 数据集**

四个NLP任务的数据集：Few‑NERD（NER）、CoNLL‑2000（Chunking）、SST‑3（Sentiment）、CommonsenseQA（Commonsense）。

**📈 对比分析**

与TN、QRNCA、KN、ACT、RANDOM等方法对比，实验在LLaMA2‑7B、Baichuan‑2‑7B、LLaMA2‑13B上，NeuronLLM在RAC（degradation）平均提升约16.8%、RCC提升约28%，在enhancement上平均提升约7.8%/12.5%，在所有任务与模型尺寸均优于对照。

**⚠️ 局限性**

仅适用于多选QA任务；需手工生成代理问题，扩展性有限；干预规模较小，未验证更大模型或不同架构；对不同层级神经元的可解释性细节仍待深入研究。

---

## AgentDevel: Reframing Self-Evolving LLM Agents as Release Engineering

**arXiv ID:** 2601.04620 | [PDF](https://arxiv.org/pdf/2601.04620v1)

**作者:** Di Zhang `[一作]` (Fudan University), Di Zhang `[通讯]` (Fudan University)

**通讯引用:** 17366 | **OpenAlex IDs:** https://openalex.org/A5100366416

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Agentic AI` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出 AgentDevel 发布工程流水线，将 LLM 代理的改进外部化为单一版本线，利用实现无关的 LLM 评审、可执行诊断脚本、单一 RC 合成以及基于 Pass→Fail / Fail→Pass flips 的门控，实现可审计、回归感知的迭代升级。

**💡 创新点**

创新点包括：① 把代理改进视作软件发布工程而非内部自我改进；② 设计实现无关、症状级的 LLM critic；③ 通过可执行诊断脚本生成可审计的工程规范；④ 基于 P→F/F→P flips 的 flip‑centered gating，显著降低回归风险。

**🔧 技术方法**

技术手段包括：结构化执行轨迹收集与可观测性、实现无关的 LLM critic、脚本化诊断与 RC 合成、单一 RC 生成与基于 flips 的门控策略、版本化工件记录（脚本、差分、评审输出），使用 Claude Code 与 Claude‑Sonnet‑4.5 作为底层 LLM。

**📊 数据集**

主要使用四个执行重负载基准的数据集：SWE‑bench Lite、SWE‑bench Verified、WebArena 与 StableToolBench，各自提供训练集和单次测试集。

**📈 对比分析**

通过在相同初始蓝图下对比 AgentDevel 与传统基线（如 SWE‑agent、GPT‑4o scaffolded、CER_hybrid、DFS），在所有四个基准上实现 1.5–2 倍以上的提升（如 SWE‑bench Lite 从 11% 提升至 22%，StableToolBench 从 54% 提升至 73.5%），同时回归率显著降低（P→F 率从 ~15% 降至 3% 左右）。

**⚠️ 局限性**

局限性包括：仅在单一 LLM（Claude‑Sonnet‑4.5）和单一任务线下验证；对多模型或多任务的可迁移性未知；门控阈值需手工设定，可能对不同基准不适用；缺乏并行多变体搜索的能力，无法充分利用多元探索；对训练集的依赖较强，可能导致过拟合。

---

## Character-R1: Enhancing Role-Aware Reasoning in Role-Playing Agents via RLVR

**arXiv ID:** 2601.04611 | [PDF](https://arxiv.org/pdf/2601.04611v1)

**作者:** Yihong Tang `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 58048 | **OpenAlex IDs:** https://openalex.org/A5100402851

**关键词:** `Computation and Language` `Reinforcement Learning` `Reinforcement Learning` `Chain-of-Thought` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了 Character-R1 框架，利用可验证奖励实现角色意识推理，从而显著提升角色扮演代理的内部认知一致性与沉浸感。

**💡 创新点**

创新点在于三大可验证奖励设计：认知焦点奖励（强制生成并验证 10 维角色属性标签）、参考引导奖励（基于重叠度的软约束）以及角色条件化奖励归一化（解决不同角色奖励分布差异）。

**🔧 技术方法**

技术方法包括：基于 GRPO 的强化学习、链式思考（CoT）推理模板、BLEU/EM 等可验证指标、k‑means 角色分组归一化以及软奖励衔接。

**📊 数据集**

使用的数据集包括 CharacterBench（22,859 条样本、3,956 个角色）和 SocialBench（500 个角色、30,800 条多轮对话），并在 Qwen2.5‑7B、Llama‑3.2‑3B 等基础模型上进行实验。

**📈 对比分析**

与 SFT、Neeko、RAR 等基线以及专门角色模型（Character‑GLM、Haruhi‑Zero、CoSER）以及大模型（GPT‑4o、Gemini‑3‑pro 等）对比，Character‑R1 在知识、记忆、人格、情感等 11 维指标上均取得领先或相近成绩，尤其在 FA、MC、BC_K 等一致性指标上提升显著。

**⚠️ 局限性**

局限性包括：认知焦点验证依赖规则/语义相似度，难以捕捉细腻心理推理；归一化参数需手工设置；并且奖励设计仍可能导致模型对参考文本的过度拟合或不平衡。

---

## When More Words Say Less: Decoupling Length and Specificity in Image Description Evaluation

**arXiv ID:** 2601.04609 | [PDF](https://arxiv.org/pdf/2601.04609v1)

**作者:** Rhea Kapur `[一作]` (Stanford University), Elisa Kreiss `[通讯]` (University of California Los Angeles)

**关键词:** `Computation and Language` `Generation` `Retrieval` `Transformer` `Vision Language Model` `Prompt Engineering` `Image` `Text`

### 📋 论文摘要

**🎯 论文内容**

构建数据集以区分图像描述的长度与特异性，并通过对比实验验证人类更偏好特异性描述；同时评估不同提示下VLM在长度约束下的特异性表现。

**💡 创新点**

将特异性定义为对比集中的辨别度而非长度，设计长度控制且信息量不同的描述对，提出基于CLIPScore的特异性度量。

**🔧 技术方法**

对比实验、Logistic回归分析、CLIPScore评估、VLM（GPT‑4o）提示设计与长度约束实验。

**📊 数据集**

使用MS COCO 5,000张图像，包含原始描述、冗长（verbose）、合成（composite）以及VLM生成描述；对比集为其余4,999张图像。

**📈 对比分析**

人类偏好实验显示特异性高于长度，CLIPScore与人类一致；VLM在不同长度约束下特异性差异显著，表明长度分配影响特异性。

**⚠️ 局限性**

CLIPScore可能存在偏差且输入长度受限；对比集相对性限制了对单一描述的绝对评估；LLM风格差异可能影响结果；未考虑流畅性、可读性等其他质量维度。

---

## Detection of Deployment Operational Deviations for Safety and Security of AI-Enabled Human-Centric Cyber Physical Systems

**arXiv ID:** 2601.04605 | [PDF](https://arxiv.org/pdf/2601.04605v1)

**作者:** Bernard Ngabonziza `[一作]` (Arizona State University), Sandeep K. S. Gupta `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Anomaly Detection` `Safety and Privacy` `Convolutional Neural Network` `Image` `Time Series` `Biomedical Data`

### 📋 论文摘要

**🎯 论文内容**

本文提出一种基于时序图像编码和卷积神经网络的个性化模型，用于检测闭环血糖控制系统中的操作偏差（如未声明进餐）。

**💡 创新点**

创新点在于将葡萄糖‑胰岛素交互关系通过敏感性‑关系矩阵编码成图像，并利用个性化CNN进行分类，首次将此方法用于检测AI驱动人本系统的部署偏差。

**🔧 技术方法**

使用技术包括：时间序列图像编码（递归图/再现图）、敏感性‑关系矩阵（AUC_R）、卷积神经网络、分类评估指标（准确率、F1）。

**📊 数据集**

数据集来自5名1型糖尿病患者的CGM和胰岛素输注记录，分别生成约310–530张“进餐”与“非进餐”类别图像。

**📈 对比分析**

与VIT模型基线进行对比，个性化模型在不同患者的准确率介于0.60–0.69，F1分数介于0.50–0.62，某些患者表现优于基线。

**⚠️ 局限性**

局限性包括：受个体差异影响导致性能波动、样本量有限、缺乏模型可解释性及对新输入的适应能力不足。

---

## 3D Conditional Image Synthesis of Left Atrial LGE MRI from Composite Semantic Masks

**arXiv ID:** 2601.04588 | [PDF](https://arxiv.org/pdf/2601.04588v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition`

---

## The UnScripted Trip: Fostering Policy Discussion on Future Human-Vehicle Collaboration in Autonomous Driving Through Design-Oriented Methods

**arXiv ID:** 2601.04601 | [PDF](https://arxiv.org/pdf/2601.04601v1)

**作者:** Xinyan Yu `[一作]` (University of Sydney), Wendy Ju `[通讯]` (Cornell University)

**通讯引用:** 6471 | **OpenAlex IDs:** https://openalex.org/A5016068576

**关键词:** `Human-Computer Interaction` `Autonomous Driving`

### 📋 论文摘要

**🎯 论文内容**

通过设计并实施一款名为“The UnScripted Trip”的卡牌游戏，帮助汽车研究者、设计师与行业从业者共同探索未来自动驾驶人车协作场景下的政策方向与设计方案。

**💡 创新点**

创新点在于将卡牌游戏与虚构媒体情境相结合，构建双阶段（构造背景故事与冲突情境）游戏流程，使参与者在轻松的互动中深入反思人车协作的伦理、责任与政策问题。

**🔧 技术方法**

主要技术手段是卡牌游戏机制（社交关系卡、AV功能角色卡、情境卡、轮盘式随机生成的人机行动与响应），以及基于虚构媒体的情境设定和故事叙事方法。

**📊 数据集**

该研究未使用传统数据集，而是依赖参与者在工作坊中生成的虚构情境与讨论内容作为评估素材。

**📈 对比分析**

未开展量化性能比较；评估主要通过参与者的定性反馈与工作坊报告来判断游戏对讨论启发和政策构想的有效性。

**⚠️ 局限性**

局限性包括：1) 依赖虚构情境，可能与真实法规和技术实践脱节；2) 参与者样本规模有限，结果不具备广泛推广性；3) 主要关注设计与政策讨论，缺乏对技术实现细节的深入探讨。

---

## On the Limitations of Rank-One Model Editing in Answering Multi-hop Questions

**arXiv ID:** 2601.04600 | [PDF](https://arxiv.org/pdf/2601.04600v1)

**作者:** Zhiyuan He `[一作]` (University College London), Xi Chen `[通讯]` (University College London)

**通讯引用:** 58063 | **OpenAlex IDs:** https://openalex.org/A5100329996

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究了单层知识编辑在多跳推理中的局限并提出冗余编辑方法

**💡 创新点**

通过将同一知识插入多层解决“跳跃过晚”、泛化衰退与过拟合问题

**🔧 技术方法**

采用ROME框架与多层插入技术

**📊 数据集**

在GPT‑J‑6B模型上使用MQuAKE和COUNTERFACT数据集

**📈 对比分析**

相较单层ROME，冗余编辑将2跳问答准确率提升约15.5个百分点(约96%)，但在单跳自然语言指标上略有下降

**⚠️ 局限性**

仅针对2跳问题，未探究更深层次或多跳编辑，以及其他编辑方法和模型架构

---

## MiLDEdit: Reasoning-Based Multi-Layer Design Document Editing

**arXiv ID:** 2601.04589 | [PDF](https://arxiv.org/pdf/2601.04589v1)

**作者:** Zihao Lin `[一作]` (University of California), Tong Sun `[通讯]` (Adobe)

**通讯引用:** 14929 | **OpenAlex IDs:** https://openalex.org/A5100372581

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning` `Optimization` `Transformer` `Large Language Model` `Reinforcement Learning` `Agentic AI` `Text` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出了首个多层设计文档编辑的基准（Multi‑Layer Document Editing Benchmark）并设计了基于推理的编辑代理（Multi‑Layer Document Editing Agent）来实现按层精准修改；

**💡 创新点**

1) 结合层级结构的推理框架，首次将多层设计文档编辑定义为两阶段任务；2) 采用基于奖励的群体相对策略优化（GRPO）训练多模态推理器，实现层选择与层级指令生成；3) 设计四维评价指标与统一综合指标MiLDEScore；4) 构建含20K真实文档、50K自然语言编辑指令和87K层级编辑步骤的人工校验数据集；

**🔧 技术方法**

多模态大型语言模型（InternVL3‑38B）+ RL（GRPO）训练的推理器、冻结的图像生成编辑器、alpha 混合、层级指令与图像编辑模块、评估时使用 VQA、mask 匹配、审美预测器、OCR‑VQA 等多技术；

**📊 数据集**

主要使用公开 Crello 数据集（含透明背景多层设计稿）进行文档采集，随后通过 MLLM 生成指令并进行人工校验；

**📈 对比分析**

在 Benchmark 上评测了14个开源模型、2个闭源模型，并与所提出的 Agent 进行对比；Agent 在 MiLDEScore 上提升 82.78% 以上（相较于最强开源 Baseline）并接近或优于闭源模型，尤其在布局一致性与层级决策准确度方面表现突出；

**⚠️ 局限性**

局限性包括：1）对复杂多层推理的错误率仍较高；2）当前评估仅支持单图输出，未能完整测试多层交互；3）层级决策是独立的，偶尔会同时编辑多层导致冲突；4）对底层编辑器的依赖限制了编辑多样性。

---

## Deep Dive into the Abuse of DL APIs To Create Malicious AI Models and How to Detect Them

**arXiv ID:** 2601.04553 | [PDF](https://arxiv.org/pdf/2601.04553v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Cryptography and Security`

---

## Aligning Text, Code, and Vision: A Multi-Objective Reinforcement Learning Framework for Text-to-Visualization

**arXiv ID:** 2601.04582 | [PDF](https://arxiv.org/pdf/2601.04582v1)

**作者:** Mizanur Rahman `[一作]` (York University), Enamul Hoque `[通讯]` (York University)

**关键词:** `Computation and Language` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Text` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

开发了RL-Text2Vis，一个基于强化学习的文本到可视化生成框架，利用后置多模态反馈同时优化文本回答、代码可执行性和图表质量。

**💡 创新点**

首次将文本正确性、代码有效性与图表可读性/准确性三项目标融合为多目标奖励，并采用Group Relative Policy Optimization（GRPO）实现无价值网络的高效强化学习。

**🔧 技术方法**

使用Qwen2.5‑Instruct 7B/14B作为策略模型，GRPO作为优化器，采用两阶段奖励（格式奖励+组合奖励），并用LLM/VLM评估器（Qwen2.5、Qwen2.5‑VL、GPT‑4o）提供后置多模态反馈。

**📊 数据集**

主要在Text2Vis基准集（train/test1和eval/test2）训练与评估，并在VisEval、NVBench、PandasPlotBench 等外域数据集验证跨域泛化。

**📈 对比分析**

与闭源 GPT‑4o、Gemini 以及开源 LLaMA、CodeLLaMA、Mistral 等零样本和SFT基线对比，采用代码可执行率、答案匹配、图表可读性与正确性等指标；RL‑Text2Vis 14B 在图表可读性和正确性上提升约22%，代码可执行率从78%升至97%，并在外域基准上同样实现显著提升。

**⚠️ 局限性**

14B模型计算和显存成本较高；未针对更大规模模型或交互式图表进行实验；在医疗、金融等高度专业化领域的稳健性尚未验证；仅覆盖静态可视化，未处理潜在数据偏见或误导性图表。

---

## Sci-Reasoning: A Dataset Decoding AI Innovation Patterns

**arXiv ID:** 2601.04577 | [PDF](https://arxiv.org/pdf/2601.04577v1)

**作者:** Jiachen Liu `[一作]` (Orchestra Research), Zechen Zhang `[通讯]` (Orchestra Research)

**关键词:** `Artificial Intelligence` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

构建了Sci‑Reasoning数据集，系统捕捉并结构化了高质量AI研究论文的知识合成与推理过程。

**💡 创新点**

首次提供大规模、结构化的科研推理轨迹，揭示15种创新思维模式并量化三种主导策略；利用LLM与人工验证高效提取关键前置论文及其关系。

**🔧 技术方法**

核心技术包括：基于GPT‑5的单次推理链路抽取、线性化的Lineage Link结构、人工审核校正；评估指标为召回率（89.73%）和多模型交叉验证精度（最高49.35%）。

**📊 数据集**

数据来源为2023‑2025年NeurIPS、ICML、ICLR会议的口头/Spotlight论文，总计3,819篇；前置论文列表由GPT‑5提取并人工核对，形成完整的知识网络。

**📈 对比分析**

与传统仅记录引用关系的网络相比，Sci‑Reasoning提供了“关系类型”“先行者角色”等更细粒度信息；实验表明LLM可在约90%召回率下识别前置论文，且通过多模型验证可达到49.35%的方向预测准确率，展示了高质量推理链路构建的可行性。

**⚠️ 局限性**

局限性包括：仅覆盖三大ML会议的口头/Spotlight论文，可能忽略其他优秀工作；依赖GPT‑5模型与人工审核，成本与可扩展性受限；当前模式分析集中于AI领域，跨学科迁移需要进一步验证。

---

## Spatial-Temporal Feedback Diffusion Guidance for Controlled Traffic Imputation

**arXiv ID:** 2601.04572 | [PDF](https://arxiv.org/pdf/2601.04572v1)

**作者:** Xiaowei Mao `[一作]` (Beijing Jiaotong University), Huaiyu Wan `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 6288 | **OpenAlex IDs:** https://openalex.org/A5065949777

**关键词:** `Machine Learning` `Diffusion model` `Score-based Model` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

在空间时间交通数据缺失值填补中提出 FENCE 方法，通过自适应动态调整扩散模型的指导尺度来提升填补质量。

**💡 创新点**

创新点在于将后验似然驱动的反馈机制和基于聚类的节点级指导尺度引入扩散模型，实现动态、空间时间感知的指导。

**🔧 技术方法**

技术：基于分数扩散模型（CSDI）与 CFG，使用无条件与有条件分数估计，动态后验更新，聚类注意力聚合，双阶段训练。

**📊 数据集**

数据集：PEMS04、PEMS07、PEMS08 三个大规模交通速度数据集。

**📈 对比分析**

对比八种基线（ASTGNN、IGNNK、GCASTN、LCR、ImputeFormer、mTAN、CSDI、PriSTI），在 SR‑TC、SC‑TC 两种缺失模式下，FENCE 以 MAPE 平均下降约 6% 以上，稳占最优。

**⚠️ 局限性**

局限：需要先训练无条件模型，聚类更新成本较高，对超参数（π、τ、δ、聚类数）敏感。

---

## MAGA-Bench: Machine-Augment-Generated Text via Alignment Detection Benchmark

**arXiv ID:** 2601.04633 | [PDF](https://arxiv.org/pdf/2601.04633v1)

**作者:** Anyang Song `[一作]` (Fudan University), Rui Feng `[通讯]` (Fudan University)

**通讯引用:** 5701 | **OpenAlex IDs:** https://openalex.org/A5100680619

**关键词:** `Computation and Language` `Generation` `Adversarial Attack` `Reinforcement Learning from Human Feedback` `Transformer` `Reinforcement Learning` `Prompt Engineering` `Large Language Model` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

设计并构建了多域、多模型、全语言的MAGA数据集，通过角色扮演、BPO、Self-Refine和RLDF-CMD四种对齐技术生成更接近人类文本的机器生成文本，并验证其对现有检测器的攻击效果与提升检测器泛化能力的双重价值。

**💡 创新点**

创新点在于首次将强化学习基于检测器反馈（RLDF-CMD）与角色扮演、BPO和Self-Refine等多种对齐方法结合，构造出覆盖10个领域、12个LLM的对齐生成数据集，并证明其能同时削弱检测器性能和提升Fine-tuned检测器的泛化。

**🔧 技术方法**

技术手段包括RLDF（RL从检测器反馈学习）、RLDF-CD/CM跨域/跨模型奖励、角色扮演Prompt、BPO（Prompt优化模型）、Self-Refine（自我优化）以及RoBERTa等神经网络检测器。

**📊 数据集**

使用936k条来自10个领域、12个LLM的MAGA生成文本（与对齐前的MGB基线相对比），并在公开基准数据集M4、HC3、MAGE等上进行外部验证。

**📈 对比分析**

与主流检测器（如RADAR、SCRN、F-DetectGPT等）对比，MAGA使平均ACC下降5.58%、TPR下降11.16%；但RoBERTa在MAGA训练后在外部数据集上的ACC平均提升2.06%，显著优于未对齐基线和传统检测器。

**⚠️ 局限性**

局限性包括：BPO效果不显著、对低资源语言或专业领域（如医学、法律）的评估不足、仅测试8个检测器且未覆盖最新大型模型及对齐+对抗双重干扰场景。

---

## Using Ray-shooting Queries for Sublinear Algorithms for Dominating Sets in RDV Graphs

**arXiv ID:** 2601.04626 | [PDF](https://arxiv.org/pdf/2601.04626v1)

**作者:** Therese Biedl `[一作]` (University of Waterloo), Prashant Gokhale `[通讯]` (University of Wisconsin-Madison)

**关键词:** `Data Structures and Algorithms` `Optimization` `Computational Efficiency` `Graph Neural Network` `Graph`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种利用几何表示与射线投射数据结构，在 RDV 图（即根向下路径交叉图）上以 O(nlogn) 时间求解最小支配集，并在其子类区间图上进一步实现 O(n) 时间算法。

**💡 创新点**

创新点在于：① 将 RDV 图的邻接查询转化为水平/垂直线段相交问题；② 在贪心算法中引入优先搜索树（Priority Search Tree）和射线投射（Ray Shooting）来高效完成两项关键操作；③ 通过对 RDV 表示的预处理，使得线段互不相交，从而保证查询的正确性与效率；④ 将相同思路应用于区间图，得到更简洁的 O(n) 算法。

**🔧 技术方法**

主要技术手段包括：
- RDV 图的几何映射（把顶点映射为水平/垂直线段），
- 优先搜索树用于在给定 x 取值区间内快速返回最大权值点（实现 Operation A），
- 动态射线投射结构（Giyora-Kaplan 方案）用于在给定顶点邻域内以对数时间标记所有未被覆盖的邻居（实现 Operation B）。
- 对区间图的简化处理：利用点集只在 y 方向上的排序，直接维护一个列表即可完成 Operation A 与 B。

**📊 数据集**

实验使用的并未给出具体数据集，而是基于理论分析：假设 RDV 图的表示大小为 O(n)，即树大小与顶点数同阶；区间图同理。

**📈 对比分析**

与传统全图遍历求解最小支配集（Θ(n^2) 边）相比，本文算法的时间复杂度为 O(nlogn)（RDV 图）和 O(n)（区间图），显著降低了对边的访问量；若采用较简单的射线投射实现，复杂度升至 O(nlog^2n)。

**⚠️ 局限性**

局限性包括：
- 需要复杂的射线投射与优先搜索树实现，实际实现难度高；
- 当前最优复杂度仍为 O(nlogn)，尚未证明是否可进一步降低至 O(n) 或 O(nloglogn)；
- 仅针对 RDV 图及其子类，尚未探讨更广泛的图类；
- 对输入的 RDV 表示做了预处理，若给定的表示已满足条件可直接使用，但若无此条件则需额外时间。

---

## HyperAlign: Hyperbolic Entailment Cones for Adaptive Text-to-Image Alignment Assessment

**arXiv ID:** 2601.04614 | [PDF](https://arxiv.org/pdf/2601.04614v1)

**作者:** Wenzhi Chen `[一作]` (Chongqing University of Posts and Telecommunications), Xinbo Gao `[通讯]` (Xidian University)

**通讯引用:** 38166 | **OpenAlex IDs:** https://openalex.org/A5101785348

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Contrastive Learning` `Image` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了基于双曲空间的文本-图像对齐评估框架HyperAlign，利用双曲蕴含锥和自适应调制回归实现对生成图像与文本提示的精准对齐评估。

**💡 创新点**

创新点包括：①在双曲空间中引入动态监督的蕴含锥机制，将离散蕴含关系转化为连续几何约束；②设计自适应调制回归器，以双曲几何特征生成样本级调节参数，个性化校正欧氏余弦相似度；③将双曲几何与欧氏语义结合，实现层次化对齐建模与数值稳定性的双重优势。

**🔧 技术方法**

核心技术包括CLIP预训练特征提取、Lorentz双曲映射、双曲蕴含锥几何与动态监督损失、基于双曲几何特征的调制网络回归、以及多任务联合训练。

**📊 数据集**

实验使用了三大AIGC评测基准：AGIQA-3K、AIGCIQA2023和PKU-I2IQA，覆盖多种生成模型与多样化文本提示。

**📈 对比分析**

与12种主流NS‑IQA和AG‑IQA方法以及4种基线对比，HyperAlign在单数据库评估中SRCC平均0.8190，远超第二名（CIA‑Net 0.7991）并在跨数据库测试中SRCC 0.6873，提升6.89个百分点，表现优异。

**⚠️ 局限性**

局限性包括：①对训练数据的依赖仍显著，跨域迁移虽优但仍受限于数据分布差异；②双曲映射与指数映射的数值不稳定性可能影响大规模部署；③目前仅评估对齐度，未同时兼顾图像质量与真实感，后续可与多任务框架结合。

---

## HUR-MACL: High-Uncertainty Region-Guided Multi-Architecture Collaborative Learning for Head and Neck Multi-Organ Segmentation

**arXiv ID:** 2601.04607 | [PDF](https://arxiv.org/pdf/2601.04607v1)

**作者:** Xiaoyu Liu `[一作]` (Fudan University), Zhijian Song `[通讯]` (Fudan University)

**通讯引用:** 3858 | **OpenAlex IDs:** https://openalex.org/A5111697224

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Convolutional Neural Network` `Image` `Biomedical Data` `Computed Tomography`

### 📋 论文摘要

**🎯 论文内容**

提出了高不确定性区域引导多架构协同学习模型（HUR-MACL），用于头颈多器官分割。

**💡 创新点**

创新点在于自适应识别高不确定性区域、使用 Vision Mamba 与 Deformable CNN 两条协同分支，并通过异构特征蒸馏损失实现跨架构知识共享。

**🔧 技术方法**

采用 U‑Net 骨干，结合 1×1 卷积+softmax 熵挖掘不确定性区域，Vision Mamba 与 Deformable CNN 并行分支，并设计异构特征蒸馏损失与多层次总损失。

**📊 数据集**

使用公开的 PDDCA（48 病例）和 StructSeg（50 病例）数据集，以及内部的 Inhouse（118 病例）头颈 CT 数据集。

**📈 对比分析**

与 U‑Net、nnUNet、TransUNet、Missformer、Mamba‑UNet、Swin‑Mamba、MHL‑Net、FocusNet 等 8 种方法对比，HUR‑MACL 在三大数据集均实现了 Dice 平均提升 4–5%，尤其在视交叉、脊髓等小型复杂结构上表现突出，达到 SOTA。

**⚠️ 局限性**

局限性包括需手动调节阈值 T，模型参数量与 FLOPs 较大，部署受限；目前仅在头颈数据验证，跨区域或跨模态的泛化尚未评估。

---

## Constitutional Classifiers++: Efficient Production-Grade Defenses against Universal Jailbreaks

**arXiv ID:** 2601.04603 | [PDF](https://arxiv.org/pdf/2601.04603v1)

**作者:** Hoagy Cunningham `[一作]`, Mrinank Sharma `[通讯]`

**关键词:** `Cryptography and Security` `Adversarial Attack` `Computational Efficiency` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一套基于宪法分类器的防御体系，结合交换式分类器、两阶段级联和线性探测器，实现生产级的防越狱性能。

**💡 创新点**

创新点在于：①将输入与输出合并为交换式分类器以检测重构和输出混淆攻击；②采用轻量级探测器先行筛查，节约计算成本；③在探测器中加入滑动窗口平滑和softmax加权损失，显著提升探测精度。

**🔧 技术方法**

技术手段包括：宪法分类器、两阶段级联、线性激活探测器（logit平滑+softmax加权）、模型内部特征提取、红队评估与人机交互评判。

**📊 数据集**

数据集主要来自人工红队生成的CBRN主题越狱对话（约7000条），以及合成的宪法分类训练数据和公开的WildChat、GPQA Diamond等。

**📈 对比分析**

与上一代系统比较，计算开销下降40倍（单一交换式分类器）至8倍（两阶段级联），拒绝率降至0.05%，高危漏洞发现率降低至0.005/千问，未出现全八题通用越狱。

**⚠️ 局限性**

局限性包括：对极度隐蔽或极低频攻击的鲁棒性仍有限，探测器误报率略高，系统依赖大量人工红队资源和持续监测以保持效能。

---

## FedKDX: Federated Learning with Negative Knowledge Distillation for Enhanced Healthcare AI Systems

**arXiv ID:** 2601.04587 | [PDF](https://arxiv.org/pdf/2601.04587v1)

**作者:** Quang-Tu Pham `[一作]` (VinUniversity), Hieu H. Pham `[通讯]` (VinUniversity)

**通讯引用:** 1052 | **OpenAlex IDs:** https://openalex.org/A5065112274

**关键词:** `Machine Learning` `Federated Learning` `Knowledge Distillation` `Computational Efficiency` `Safty and Privacy` `Convolutional Neural Network` `Contrastive Learning` `Biomedical Data` `Multimodality` `Electronic Health Records`

### 📋 论文摘要

**🎯 论文内容**

提出了 FedKDX 联邦学习框架，结合负向知识蒸馏、对比学习和动态 SVD 梯度压缩，实现医疗 AI 在隐私保护下的高效学习。

**💡 创新点**

创新点在于引入负向知识蒸馏捕获非目标类信息，结合对比学习形成结构一致性，并使用动态阈值的 SVD 压缩在保持通信效率的同时避免信息丢失。

**🔧 技术方法**

使用知识蒸馏、对比学习、负向知识蒸馏、动态 SVD 梯度压缩以及 CNN（可选 LSTM）模型架构。

**📊 数据集**

使用三大医疗传感器数据集：SLEEP（睡眠姿态）、UCI-HAR（人类活动识别）和 PAMAP2（多传感器活动）。

**📈 对比分析**

与 FedAvg、FedMAT、FedFomo、MOON、FedProx、FedMTL、FedKD 等七种基线对比，FedKDX 在所有数据集上准确率提升 0.73%~2.53%，AUC 超过 0.98，收敛速度快、鲁棒性好。

**⚠️ 局限性**

主要限制是 SVD 梯度压缩在后期训练易出现非收敛或失真，极端非 IID 情况下表现仍受限，且目前仅验证于活动识别，需进一步扩展到图像、EHR 等多模态医疗数据。

---

## Autonomous Agents on Blockchains: Standards, Execution Models, and Trust Boundaries

**arXiv ID:** 2601.04583 | [PDF](https://arxiv.org/pdf/2601.04583v1)

**作者:** Saad Alqithami `[一作]`, Saad Alqithami `[通讯]`

**关键词:** `Artificial Intelligence` `Agentic AI` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

对代理‑区块链互操作性领域进行系统性文献综述，梳理了 317 项相关工作，构建了五类集成模式、专属威胁模型、13 维度的能力矩阵，并提出了研究路线图与可复现评估工具包。

**💡 创新点**

提出统一的五层集成模式分类、针对代理驱动交易的专门威胁模型、对 20+ 代表系统的量化比较、以交易意图模式（Transaction Intent Schema）和政策决策记录（Policy Decision Record）为核心的接口抽象，以及完整可复现的评估框架。

**🔧 技术方法**

采用系统性文献回顾（SLR）方法与 PRISMA 2020 筛选流程，检索学术数据库、灰度文献及技术规范（如 EIP、RFC）；对选定系统进行编码与量化比较，并引入交易意图与可审计策略层的技术概念。

**📊 数据集**

使用从学术数据库（IEEE Xplore、ACM DL 等）和灰度文献（arXiv、EIP 文档、开源仓库）检索得到的 3270 条记录，最终筛选 317 项研究；对 85 项公开实现充分的系统进行详细编码；未使用传统 ML 数据集，而是基于公开区块链交易记录与 LLM 代理实验数据。

**📈 对比分析**

通过对 85 项系统在 13 维度（托管模式、权限管理、策略执行、可观测性、恢复机制等）进行编码，构建定量能力矩阵；对比各系统在安全性、可扩展性、易用性等方面的差异；报告显示当前生态中存在显著安全空隙，缺乏统一的意图描述与可审计策略层。

**⚠️ 局限性**

局限性包括：仅覆盖 2026 年初前的工作；聚焦以太坊及其兼容链，忽略私有链和其他共识算法；主要关注 LLM 驱动的代理，未覆盖强化学习或规则驱动系统；数据来源为英文文献，可能存在地域偏见；系统快速迭代导致文献更新滞后。

---

## FeedEval: Pedagogically Aligned Evaluation of LLM-Generated Essay Feedback

**arXiv ID:** 2601.04574 | [PDF](https://arxiv.org/pdf/2601.04574v1)

**作者:** Seongyeub Chu `[一作]` (KAIST), Munyong Yi `[通讯]` (KAIST)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了FeedEval框架，用于评估LLM生成的作文反馈在特异性、实用性和有效性三个教育维度上的质量。

**💡 创新点**

通过构造专门的评估数据集（SpecEval、帮助性对比集、基于NLI的有效性集）以及对LLM进行维度化微调，首次实现了高质量反馈的自动筛选与过滤。

**🔧 技术方法**

使用了大型语言模型（如Llama3-3B-Instruct、Qwen3-8B等）进行微调，结合softmax归一化、多维度评估器与排名损失，构建FeedEval评估器。

**📊 数据集**

主要数据集包括自建的SpecEval（41k条）、帮助性评估数据（14k对比对）、NLI有效性数据（99k对）、以及公开的ASAP++作文评分数据。

**📈 对比分析**

在ASAP++上与GPT-5.1、Gemini-2.5-Pro等对照，FeedEval在与人工专家对齐率上超过75%，并且使用其筛选出的高质量反馈训练的评分模型在所有 trait 上均优于低质量反馈，提升 QWK 约 1–5%。

**⚠️ 局限性**

局限性包括受模型生成顺序影响（先评分后反馈更好）以及仅在英语作文上验证，无法直接推广至其他语言。

---

## A Method for Constructing a Digital Transformation Driving Mechanism Based on Semantic Understanding of Large Models

**arXiv ID:** 2601.04696 | [PDF](https://arxiv.org/pdf/2601.04696v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents

**arXiv ID:** 2601.04566 | [PDF](https://arxiv.org/pdf/2601.04566v1)

**作者:** Yunhao Feng `[一作]` (Fudan University), Yugang Jiang `[通讯]` (Fudan University)

**通讯引用:** 23651 | **OpenAlex IDs:** https://openalex.org/A5047962986

**关键词:** `Artificial Intelligence` `Adversarial Attack` `AI Code Assistant` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了BackdoorAgent框架，系统性分析多步骤LLM代理工作流中的后门漏洞；

**💡 创新点**

创新点在于将后门攻击与代理的规划、记忆、工具三大阶段对应，构建阶段感知的注入与传播模型，并设计统一的基准和评测流程；

**🔧 技术方法**

使用模块化钩子实现规划、检索、工具的触发注入，记录完整轨迹；结合多轮LLM推理、检索增强生成（RAG）、工具调用等技术；

**📊 数据集**

采用四类代表性代理任务（Agent QA、Agent Code、Agent Web、Agent Drive）以及多种闭源与开源大模型（Claude、Gemini、GPT、Qwen、DeepSeek、Kimi、Qwen2.5 等），并在这些任务上构建对应数据集；

**📈 对比分析**

通过对比清洁任务准确率（Clean ACC）、攻击成功率（ASR）和攻击下的准确率（ACC）评估；结果显示内存通道攻击的ASR最高，规划通道次之，工具通道变异性最大；尽管攻击成功率高，但多数模型的任务准确率几乎不受影响；

**⚠️ 局限性**

局限性包括：只在有限的四个任务与若干模型上实验，缺乏更广泛的任务多样性；未提出成熟的防御方法；以及评估侧重于任务级指标，未深入探讨对抗样本检测或多步轨迹分析等更细粒度手段。

---

## ThinkDrive: Chain-of-Thought Guided Progressive Reinforcement Learning Fine-Tuning for Autonomous Driving

**arXiv ID:** 2601.04714 | [PDF](https://arxiv.org/pdf/2601.04714v1)

**作者:** Chang Zhao `[一作]`, Wen Ji `[通讯]`

**关键词:** `Artificial Intelligence` `Autonomous Driving` `Reinforcement Learning` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Chain-of-Thought` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出了 ThinkDrive，一个基于链式推理的进阶强化学习微调框架，提升多模态 LLM 在自动驾驶中的推理与决策能力。

**💡 创新点**

创新点在于将 CoT 的结构化推理与难度感知自适应策略优化相结合，并通过 Gaussian 版课程学习实现训练过程的渐进与稳定。

**🔧 技术方法**

使用了两阶段训练：先使用 CoT 进行监督微调，再采用基于 GRPO 的进阶 RL，加入难度感知熵调节、clip-higher 机制与几何平均等技术。

**📊 数据集**

采用了 DrivingVQA 自动驾驶问答数据集，包含多模态图像与多选题目及 CoT 解释。

**📈 对比分析**

与 SFT 及现有 RL 基线（GRPO、GSPO、DAPO、GMPO）在 exam、easy-exam、accuracy 三个指标上对比，ThinkDrive 提升约 2–3% 并在 2B 参数模型上超过 GPT‑4o 3.28%。

**⚠️ 局限性**

局限在于仍需手工设定阈值、熵权重等超参，对不同任务的通用性与迁移能力尚未完全验证，且对极难样本的收敛速度仍有限。

---

## TourPlanner: A Competitive Consensus Framework with Constraint-Gated Reinforcement Learning for Travel Planning

**arXiv ID:** 2601.04698 | [PDF](https://arxiv.org/pdf/2601.04698v1)

**作者:** Yinuo Wang `[一作]` (Xiaohongshu Inc), Weiming Dong `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Optimization` `Recommendation System` `Transformer` `Large Language Model` `Reinforcement Learning` `Chain-of-Thought` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

本文提出了TourPlanner框架，通过三阶段流程（PReSO、CCoT、Constraint-Gated RL）生成同时满足硬约束与软约束的旅行行程。

**💡 创新点**

创新点包括多路径竞争共识Chain‑of‑Thought（CCoT）机制和基于sigmoid门控的RL奖励策略，有效提升多约束的一致性与可行性。

**🔧 技术方法**

技术手段主要是LLM推理（如GPT‑4o、Qwen3等）、多维检索+空间聚类、竞争式多代理推理、以及门控RL优化。

**📊 数据集**

实验数据集为TripTailor沙盒数据集，涵盖40个中国城市的POI、住宿、交通等真实信息。

**📈 对比分析**

与Direct、ReAct、TripTailor等基线对比，宏观合理性通过率超过88%，最终通过率达到56%，平均路线距离比率显著下降，证明性能显著提升。

**⚠️ 局限性**

主要局限在于CcoT的迭代难以实现端到端RL优化，且奖励模型对用户偏好的对齐度仍有待提升。

---

## ResMAS: Resilience Optimization in LLM-based Multi-agent Systems

**arXiv ID:** 2601.04694 | [PDF](https://arxiv.org/pdf/2601.04694v1)

**作者:** Zhilun Zhou `[一作]` (Tsinghua University), Fengli Xu `[通讯]` (Tsinghua University)

**通讯引用:** 2417 | **OpenAlex IDs:** https://openalex.org/A5062365263

**关键词:** `Artificial Intelligence` `Optimization` `Reinforcement Learning` `Graph Neural Network` `Large Language Model` `Reinforcement Learning` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出 ResMAS 两阶段框架，自动生成 LLM 多代理系统的鲁棒拓扑并根据拓扑细化提示，以提升在随机代理失效环境下的系统弹性。

**💡 创新点**

创新点在于：① 使用奖励模型预测弹性并通过 GRPO 强化学习训练 LLM 自动生成适配任务和约束的高弹性拓扑；② 设计拓扑感知提示优化方法，使代理提示与网络结构和邻居交互历史相匹配，从而进一步提升弹性。

**🔧 技术方法**

采用图神经网络（GCN）构建奖励模型、GRPO 强化学习、LoRA 微调、LLM 生成拓扑、拓扑感知提示优化、随机代理失效仿真等技术。

**📊 数据集**

实验数据集包括 MMLU-Pro、MATH、Chess 三个多任务基准，并在 HumanEval 等未见任务上验证通用性。

**📈 对比分析**

与 G-Designer、OPRO、TextGrad、GPTSwarm 等基线在不同节点/边约束下对比，ResMAS 在所有数据集上取得最高弹性，并在精度与弹性之间达到 Pareto 前沿；在未见任务和模型上也保持优越表现。

**⚠️ 局限性**

局限性在于：仍采用分阶段优化，未能实现拓扑与提示的联合全局最优；目前只针对同质 LLM 代理，异构代理或更复杂部署场景的鲁棒性尚未验证。

---

## Nightmare Dreamer: Dreaming About Unsafe States And Planning Ahead

**arXiv ID:** 2601.04686 | [PDF](https://arxiv.org/pdf/2601.04686v1)

**作者:** Oluwatosin Oseni `[一作]` (Colorado School of Mines), Micah Corah `[通讯]` (Colorado School of Mines)

**通讯引用:** 466 | **OpenAlex IDs:** https://openalex.org/A5082404058

**关键词:** `Machine Learning` `Reinforcement Learning` `Safety and Privacy` `Robotic Intelligence` `Recurrent Neural Network` `Reinforcement Learning` `World Model` `Sequential` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于模型的安全强化学习算法 Nightmare Dreamer，利用世界模型预测潜在安全违规并主动规划动作，以在视觉输入下实现接近零安全违规的同时最大化奖励。

**💡 创新点**

核心创新点包括：① 双演员架构将奖励最大化与成本约束分离；② 通过世界模型进行前瞻性安全规划，提前预见并避免未来违规；③ 使用判别器对安全策略进行正则化，提升训练稳定性和性能。

**🔧 技术方法**

采用了 Dreamer 系列的世界模型（Recurrent State‑Space Model）、双演员 Actor‑Critic 结构、λ‑目标值估计、拉格朗日乘子法、判别器行为模仿，以及基于预估成本的在线策略切换。

**📊 数据集**

在 Safety Gymnasium benchmark（主要是 SafePointCircle 和 SafeCarCircle 圆形任务）上进行实验。

**📈 对比分析**

与 SafeRL 的 SOTA 方法（如 CPO、PPO‑Lagrangian、Safe Dreamer 等）对比，Nightmare Dreamer 在相同或更少的环境交互步数下实现了更快的收敛、接近零成本、并保持或超过奖励水平；在样本效率上提升约 20 倍。

**⚠️ 局限性**

局限性包括：① 需要较高的计算资源（在线规划和双演员训练）；② 目前仅验证于相对简单的圆形任务，缺乏在更复杂或真实机器人环境中的泛化证明；③ 对拉格朗日乘子学习率和阈值设置敏感。

---

## Leveraging LLMs for Efficient and Personalized Smart Home Automation

**arXiv ID:** 2601.04680 | [PDF](https://arxiv.org/pdf/2601.04680v1)

**作者:** Chaerin Yu `[一作]` (Ajou University), Sangeun Oh `[通讯]` (Korea University)

**通讯引用:** 1057 | **OpenAlex IDs:** https://openalex.org/A5038510569

**关键词:** `Human-Computer Interaction` `Recommendation System` `Computational Efficiency` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一个基于大型语言模型（LLM）的智能家居自动化代理IoTGPT，通过任务分解与记忆重用实现可靠、高效、个性化的IoT指令生成。

**💡 创新点**

创新点在于：①将自然语言指令拆分为可复用子任务并在层级任务记忆中缓存；②利用环境属性抽象的偏好表实现跨设备的自适应个性化；③引入两步自动与人工纠错机制提升可靠性。

**🔧 技术方法**

核心技术包括：LLM（如GPT‑4o）与结构化提示、检索增强生成（RAG）、语义嵌入相似度检索、层级DAG任务记忆、EUPont环境属性本体、虚拟家居环境模拟与自检。

**📊 数据集**

使用的公开数据集为SmartThings相关的任务集合（来自Sasha和SAGE等公开数据集）共97条指令，并通过GPT‑4o生成100条交互日志用于偏好抽取。

**📈 对比分析**

在与Sasha、SAGE两大基线对比的定量评测中，IoTGPT在成功率提升至83%+、延迟降低34%+、成本下降25%+，并在两项用户研究中显著降低认知负荷、提升可用性和个性化满意度。

**⚠️ 局限性**

局限包括：偏好冲突处理不足、依赖云端LLM导致的隐私与延迟问题、上下文感知仅基于指令、缺乏自动安全校验、以及长期使用体验未进行纵向评估。

---

## Estimating Causal Effects in Gaussian Linear SCMs with Finite Data

**arXiv ID:** 2601.04673 | [PDF](https://arxiv.org/pdf/2601.04673v1)

**作者:** Aurghya Maiti `[一作]` (Columbia University), Prateek Jain `[通讯]` (Columbia University)

**关键词:** `Machine Learning` `EM 算法` `线性高斯模型理论` `合成数据`

### 📋 论文摘要

**🎯 论文内容**

提出中央化高斯线性结构因果模型（CGL‑SCM），并设计了基于 EM 算法的参数学习方法，以在有限观测数据下估计可识别的因果效应。

**💡 创新点**

①证明 CGL‑SCM 与传统 GL‑SCM 在观测分布上的表达力完全相同；②提出一种可从有限数据中学习 CGL‑SCM 参数的 EM 方案，并通过掩码保持图结构约束；③在实验中验证该方法能准确恢复因果分布。

**🔧 技术方法**

使用 EM 算法（包含 E 步与 M 步）、梯度上升、线性高斯模型理论、矩阵化表示、图结构掩码以及因果图识别技术。

**📊 数据集**

仅使用合成数据：基于前门图（Frontdoor）和笔记本图（Napkin）的 10,000 条样本。

**📈 对比分析**

对比原始模型与估计模型在交互分布（如 P(X₃|do(X₂=1))）上的均值和方差，实验显示估计结果与真实分布高度一致，验证了方法的有效性。

**⚠️ 局限性**

局限性：仅适用于高斯线性模型，无法处理非线性或非高斯噪声；EM 可能收敛到局部最优；需要已知因果图；在变量维度大时计算复杂度较高。

---

## Agri-R1: Empowering Generalizable Agricultural Reasoning in Vision-Language Models with Reinforcement Learning

**arXiv ID:** 2601.04672 | [PDF](https://arxiv.org/pdf/2601.04672v1)

**作者:** Wentao Zhang `[一作]` (Shandong University of Technology), Tao Fang `[通讯]` (Macau Millennium College)

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning` `Explainability and Interpretability` `Optimization` `Data Synthesis` `Transformer` `Vision Language Model` `Reinforcement Learning` `Large Language Model` `Multimodality` `Agriculture Related`

### 📋 论文摘要

**🎯 论文内容**

提出Agri-R1，一个基于GRPO的农业视觉语言问答框架，通过自动化生成和过滤推理链来实现解释性诊断。

**💡 创新点**

创新点：① 用 VLM+LLM 自动合成高质量推理数据，仅使用原始数据的 19%；② 设计针对农业词汇的模糊匹配奖励函数，评价答案的正确性和语言灵活性；③ 将 GRPO 与结构化推理相结合，使 3B 参数模型在多项指标上超过 7B‑13B 传统微调模型。

**🔧 技术方法**

技术：Vision‑Language Model (Qwen2.5‑VL‑3B‑Instruct)、DeepSeek‑VL2 推理链生成、GPT‑4 过滤判别、Group Relative Policy Optimization (GRPO)、基于词典的模糊匹配奖励、三维奖励（格式、答案、推理质量）。

**📊 数据集**

数据集：CDDMBench（1.05M 视觉问答样本）用于 SFT，采用 19% 的采样用于 GRPO 训练；另外评估 AgMMU 以检验跨域泛化。

**📈 对比分析**

与基线对比：在 CDDMBench 上 3B 模型在疾病识别上相对提升 23.2%、在知识 QA 上提升 33.3%、跨域泛化提升 26.1；相比 7B‑13B 微调模型仅提升 1.8%–3.8%，但显著缩减参数规模。

**⚠️ 局限性**

局限性：① 频率导致的梯度竞争削弱低频类别表现；② 若无 SFT 预热，格式一致性可能受损；③ 未考虑疾病的时间序列与环境动态。

---

## Quantifying Autoscaler Vulnerabilities: An Empirical Study of Resource Misallocation Induced by Cloud Infrastructure Faults

**arXiv ID:** 2601.04659 | [PDF](https://arxiv.org/pdf/2601.04659v1)

**作者:** Gijun Park `[一作]` (Okestro AI Research Center), Gijun Park `[通讯]` (Okestro AI Research Center)

**关键词:** `Distributed, Parallel, and Cluster Computing`

### 📋 论文摘要

**🎯 论文内容**

通过在AWS和GCP环境中注入四类常见云故障，系统模拟实验并量化故障对垂直和水平自动扩缩的影响。

**💡 创新点**

首次系统性量化故障导致的指标失真对自动扩缩的成本与可靠性影响，并给出基于多指标、多阈值的容错扩缩建议。

**🔧 技术方法**

使用故障注入工具（AWS FIS、Pumba、hping3、MHDDoS）、Prometheus监控、Kubernetes默认扩缩算法与计算公式。

**📊 数据集**

基于CloudSuite 4.0 Web服务基准（5个虚拟用户、15 min）、自定义DoS攻击流量以及模拟的网络/磁盘/软件/DoS故障。

**📈 对比分析**

将故障与正常状态下的扩缩决策进行对比，计算误差比与每月成本差异；水平扩缩在80%~85% SLO下成本上升最高，垂直扩缩更稳定，整体性能表现取决于阈值与实例类型。

**⚠️ 局限性**

实验仅涵盖单一工作负载与有限故障种类，未考虑多区域、级联失效及更高波动的真实流量；因此结果在生产环境中的可迁移性需进一步验证。

---

## Model of Spatial Human-Agent Interaction with Consideration for Others

**arXiv ID:** 2601.04657 | [PDF](https://arxiv.org/pdf/2601.04657v1)

**作者:** Takafumi Sakamoto `[一作]` (Shizuoka University), Yugo Takeuchi `[通讯]` (Shizuoka University)

**通讯引用:** 423 | **OpenAlex IDs:** https://openalex.org/A5008020893

**关键词:** `Robotics` `Robotic Intelligence` `Agentic AI` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

构建了一个考虑他人的空间人机交互模型，旨在使机器人在公共场所与人类进行交流时，能够根据人类的行为估计其交流意愿，并相应调整自身的交流活动。

**💡 创新点**

创新点在于引入了一个量化参数来表示对他人的考虑程度，使得机器人能够在与人类互动时，动态调整其内部状态以适应他人的需求。

**🔧 技术方法**

使用了计算模型和虚拟现实（VR）环境进行实验，模拟人机交互过程。

**📊 数据集**

在虚拟现实环境中进行实验，参与者与虚拟机器人进行互动，观察其行为变化。

**📈 对比分析**

通过比较不同考虑值的机器人与人类的互动，发现低考虑值的机器人会抑制参与者的移动，而高考虑值的机器人则不会。实验结果表明，考虑值越高，参与者的避让行为越少。

**⚠️ 局限性**

实验仅在虚拟环境中进行，可能与现实世界中的人机交互存在差异，且未探讨考虑值的最优水平，未来需要在实际环境中进行更多实验以验证模型的有效性。

---

## Vibe Coding an LLM-powered Theorem Prover

**arXiv ID:** 2601.04653 | [PDF](https://arxiv.org/pdf/2601.04653v1)

**作者:** Zhe Hou `[一作]` (Griffith University), Zhe Hou `[通讯]` (Griffith University)

**通讯引用:** 4475 | **OpenAlex IDs:** https://openalex.org/A5055734272

**关键词:** `Artificial Intelligence` `Optimization` `Transformer` `Large Language Model` `Reinforcement Learning` `Retrieval-Augmented Generation` `Text`

### 📋 论文摘要

**🎯 论文内容**

开发了一个基于大语言模型的Isabelle/HOL自动证明器Isabellm，能够在本地计算机上实现完全自动的证明合成。

**💡 创新点**

创新点在于将LLM与证明器交互式循环相结合，提出了stepwise prover和planner两层协同架构，并设计了beam搜索、reranker、premise selection、micro‑RAG、填补与修复等机制。

**🔧 技术方法**

使用了GPT‑4.1/5.2、Gemini 3 Pro、Claude 4.5等LLM作为提案器，配合Isabelle的服务器接口、Beam搜索、ML/RL reranker、神经编码器、TF‑IDF检索、微RAG、CEGIS式修复等技术。

**📊 数据集**

主要数据集为Isabelle AFP、内置的proof scripts，使用自动收集的执行日志生成训练数据；并利用Sledgehammer、MiniF2F等基准进行评测。

**📈 对比分析**

与Sledgehammer等传统自动化对比，Isabellm在某些高阶目标上成功证明，击败了Sledgehammer；实验显示在特定案例上提升约10‑20%，但整体性能仍受限于填补/修复阶段的低成功率。

**⚠️ 局限性**

局限性包括LLM在实现复杂算法和结构化Isar修复方面的不可靠，修复阶段成功率低，系统对大规模库检索和推理仍有挑战，且对硬件要求仍有限。

---

## MMFCTUB: Multi-Modal Financial Credit Table Understanding Benchmark

**arXiv ID:** 2601.04643 | [PDF](https://arxiv.org/pdf/2601.04643v1)

**作者:** Cui Yakun `[一作]` (Hong Kong University of Science and Technology), Sirui Han `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 43 | **OpenAlex IDs:** https://openalex.org/A5021255269

**关键词:** `Computational Engineering, Finance, and Science` `Data Synthesis` `Recommendation System` `Knowledge Distillation` `Optimization` `Transformer` `Large Language Model` `Prompt Engineering` `Tabular` `Multimodality` `Benchmark` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

提出MMFCTUB基准，评估多模态LLM在信用表格理解任务中的结构感知、领域知识利用与数值推理能力。

**💡 创新点**

采用低人力干预的程序化生成信用表格，设计跨表结构感知、域知识与数值计算的细粒度评估框架，并引入mask‑recovery与hit‑rate评价方法。

**🔧 技术方法**

使用MLLM辅助生成、LLM+规则程序合成表格与标签、LaTeX渲染生成图片、mask‑recovery与计算操作hit‑rate进行评估。

**📊 数据集**

自生成的MMFCTUB数据集，包含19k+真实布局信用表格、5种表格类型、246名申请人、7600个问答对。

**📈 对比分析**

对公开与专有多模态LLM进行结构感知、知识利用、数值计算等维度的准确率/hit‑rate比较；Gemini‑3‑Flash在整体准确率最高，GPT‑5在知识利用最佳，Sonnet 4.5在算数上表现最好；大多数模型在知识推理与多表感知上仍有明显欠缺。

**⚠️ 局限性**

仅覆盖常用表格类型，可能导致评估结果偏向所选指标；跨表数量增加时大多数模型性能下降；评估粒度不足，缺少行级检索等更细粒度的分析。

---

## DP-MGTD: Privacy-Preserving Machine-Generated Text Detection via Adaptive Differentially Private Entity Sanitization

**arXiv ID:** 2601.04641 | [PDF](https://arxiv.org/pdf/2601.04641v1)

**作者:** Lionel Z. Wang `[一作]` (Nanyang Technological University), Wei Dong `[通讯]` (Nanyang Technological University)

**通讯引用:** 14824 | **OpenAlex IDs:** https://openalex.org/A5100746411

**关键词:** `Cryptography and Security` `Safty and Privacy` `Classification` `Recurrent Neural Network` `Large Language Model` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出了一种自适应差分隐私实体清洗框架DP-MGTD，用于在保证敏感信息隐私的同时实现机器生成文本（MGT）的高精度检测。

**💡 创新点**

创新点在于：①将Laplace与指数机制结合，按实体频率动态分配隐私预算；②发现DP噪声可放大人类与机器文本的可区分性，将隐私约束转化为检测优势；③实现了在保持ε‑DP的前提下，检测准确率几乎完美。

**🔧 技术方法**

核心技术包括差分隐私（Laplace与Exponential机制）、自适应预算分配、实体识别（spaCy）、统计特征提取（熵、困惑度、Logit）、语义嵌入提取（DistilBERT/RoBERTa）以及二分类器（LSTM/线性模型）。

**📊 数据集**

主要使用公开基准MGTBench‑2.0，涵盖三大领域（STEM、人文、社科）和五种LLM（Llama‑3.1‑70b、Mixtral‑8×7b、GPT‑3.5、GPT‑4o‑mini、MoonShot‑8k）产生的机器文本，与Wikipedia/ArXiv/Project Gutenberg人类文本对比。

**📈 对比分析**

与传统的零样本统计方法（Rank、Rank‑GLTR、Binoculars、Log‑Likelihood等）及有监督模型（RoBERTa‑F、DistilBERT‑F）对比，DP‑MGTD在所有域与模型上均大幅提升F1（0.53–0.67提升至≥0.99），即使在较低ε值下仍保持高准确性。

**⚠️ 局限性**

局限包括：未给出完整的理论证明为何DP噪声提升区分度；对实体稀疏文本效果未知；仅在公开英文文本上验证，低资源语言或专业领域（医疗、法律）尚待进一步测试；并未评估对更先进的隐私文本生成技术的抵抗能力。

---

## See, Explain, and Intervene: A Few-Shot Multimodal Agent Framework for Hateful Meme Moderation

**arXiv ID:** 2601.04692 | [PDF](https://arxiv.org/pdf/2601.04692v1)

**作者:** Naquee Rizwan `[一作]` (Indian Institute of Technology), Animesh Mukherjee `[通讯]` (Indian Institute of Technology)

**通讯引用:** 3965 | **OpenAlex IDs:** https://openalex.org/A5020991141

**关键词:** `Computation and Language` `Classification` `Explainability and Interpretability` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Vision Language Model` `Prompt Engineering` `Multimodality` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一个统一框架，能够同时完成仇恨表情包的分类、解释和干预，并构建了相应的标注数据集。

**💡 创新点**

创新点在于首次将三大任务（分类、解释、干预）整合为一体；通过任务特定小模型生成银数据，采用少样本提示实现三任务的同时推理；并扩展公开数据集加入解释与干预标签。

**🔧 技术方法**

技术上使用 3B 参数的多模态小模型进行全微调，生成式多模态代理（caption、explanation、intervention）产生银数据；随后在大型 LMM（8B、12B 及 GPT‑4）上通过基于 CLIP/BLIP 嵌入的 few‑shot 提示完成分类、解释与干预。

**📊 数据集**

主要使用了 FHM 与 MAMI 两个公开分类数据集，并在其测试集上扩充了解释与干预注释；用于小模型训练的还有原始解释数据集（仅仇恨样本）以及自行标注的非仇恨样本；Commonsense 与干预数据集用于常识推理和干预生成。

**📈 对比分析**

与 PromptHate、ProCap、ModHate、Few‑Shot、M2KE、VLM‑Lim、LoReHM 等七种基线进行对比；在 FHM 上 macro‑F1 取得 80.25%，在 MAMI 上 89.07%，显著优于基线；解释与干预的 BLEU/RG‑L/F1 分数同样超过现有方法。

**⚠️ 局限性**

限制主要包括：数据以英语为主，跨语言适用性未知；由于计算与数据受限，未进行大型模型的全微调；未来需进一步丰富多模态数据并验证跨平台泛化能力。

---

## GPU-Accelerated INT8 Quantization for KV Cache Compression in Large Language Models

**arXiv ID:** 2601.04719 | [PDF](https://arxiv.org/pdf/2601.04719v1)

**作者:** Maanas Taneja `[一作]`, Purab Shingvi `[通讯]`

**关键词:** `Machine Learning` `Compression` `Computational Efficiency` `Large Language Model`

### 📋 论文摘要

**🎯 论文内容**

实现并评估了大语言模型 KV 缓存的 INT8 量化方案，提供了四种 CUDA kernel（naive、tiled、coarsened、vectorized）以压缩缓存并保持精度。

**💡 创新点**

创新点包括：① 每维度（channel）独立量化，提升精度；② 四种 GPU kernel 的系统性对比，发现向量化实现可在 1B 元素上获得 1,694× 的加速；③ 通过详细误差与注意力分数误差分析，证明量化对模型行为几乎无影响。

**🔧 技术方法**

使用技术包括：INT8 线性量化、CUDA 向量化读写、共享内存缓存、线程协作（coarsening）以及 CPU 基线实现；通过自定义矩阵结构实现对 FP32 与 INT8 的转换。

**📊 数据集**

实验数据集为合成矩阵，规模覆盖 2,048×128 到 131,072×8,192 的 8 个测试场景，最大约 1 B 元素，模拟真实 LLM 长上下文的 KV 缓存。

**📈 对比分析**

与 CPU 基线比较：向量化 kernel 在最大规模下完成量化不到 50 ms，CPU 需 79 s，速度提升约 1,694×；在真实 LLM 负载下 GPU 量化耗时仅 6–58 ms；误差指标：最大绝对误差 0.00394，平均注意力分数误差 ≤ 0.1，证明对下游推理影响可忽略。

**⚠️ 局限性**

局限性：仅实现 INT8 量化，未探讨更低位宽（INT4/INT2）或混合精度；未做端到端推理性能评估（如 perplexity、推理吞吐量）；仅单 GPU 场景，未覆盖多 GPU 或分布式推理；未实现动态量化或自适应尺度计算。

---

## Fame Fades, Nature Remains: Disentangling the Character Identity of Role-Playing Agents

**arXiv ID:** 2601.04716 | [PDF](https://arxiv.org/pdf/2601.04716v1)

**作者:** Yonghyun Jun `[一作]` (Chung-Ang University), Hwanhee Lee `[通讯]` (Chung-Ang University)

**通讯引用:** 536 | **OpenAlex IDs:** https://openalex.org/A5063029769

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究并提出了角色身份的两层框架（参数层和属性层），构建统一的角色档案模式，系统评估了知名与合成角色在单回合和多回合对话中的表现，并通过注意力机制分析了性能差异。

**💡 创新点**

首次把角色身份拆分为参数化知识层与属性细粒度层，并统一化构造知名与合成角色档案，发现“名气褪色”与“属性决定性”，揭示负面属性是RPA表现瓶颈。

**🔧 技术方法**

利用大语言模型（Qwen、GPT-oss、Deepseek等）进行角色扮演，使用PersonaGym与CoSER两个评测基准，采用注意力提升和饱和度等机制分析，使用Claude-4.5-sonnet、GPT-4o进行摘要与判定。

**📊 数据集**

自建的角色身份数据集，包含109个知名角色和102个合成角色，5个顶层维度共38字段，正负属性各约60/40，约700字/角色。

**📈 对比分析**

通过单回合和多回合对比，知名角色在单回合表现优于合成角色；在多回合中差距消失甚至合成优先；注意力分析显示名气效应随回合数消退；负面属性导致多回合性能下降。

**⚠️ 局限性**

仅在Qwen3-8B上做机制分析；仅采用二元正负分类过于粗粒；未提出针对负面属性的补救方案。

---

## Prior-Informed Zeroth-Order Optimization with Adaptive Direction Alignment for Memory-Efficient LLM Fine-Tuning

**arXiv ID:** 2601.04710 | [PDF](https://arxiv.org/pdf/2601.04710v1)

**作者:** Feihu Jin `[一作]` (Peking University), Ying Tan `[通讯]` (Peking University)

**通讯引用:** 11301 | **OpenAlex IDs:** https://openalex.org/A5023089209

**关键词:** `Computation and Language` `Optimization` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了两种基于先验信息的零阶优化方法（Guiding Vector和Greedy Perturbation），实现对LLM微调过程的内存高效与收敛加速。

**💡 创新点**

创新点在于利用指导向量和贪婪扰动显著提升梯度估计的方向一致性，降低估计方差，并能无缝插拔到现有的零阶优化框架中。

**🔧 技术方法**

采用SPSA/MeZO/ SubZero等零阶优化技术，结合指导向量生成、贪婪采样、半精度FP16训练及理论方向对齐分析。

**📊 数据集**

实验使用SuperGLUE子集（CB、RTE、MultiRC、WiC、WSC、BoolQ、ReCoRD）、SST‑2、SQuAD、DROP，以及OPT‑1.3B/13B/30B和Llama2‑7B/13B等大规模模型。

**📈 对比分析**

与零样本、ICL、传统ZO、梯度基（Adam、LoRA、Prefix）等方法对比，MeZO‑GV/SubZero‑GV在11项任务中均优于基线，并在9/11项任务上超过梯度基准；收敛步数减少约30‑50%，内存占用仅比全微调低约10 GB。

**⚠️ 局限性**

局限性包括：需额外前向评估生成指导向量导致一定计算开销；在极大模型下仍可能受显存限制；实验仅在单GPU环境验证，对稀疏或动态任务的适用性尚未充分评估。

---

## Unified Framework for Qualifying Security Boundary of PUFs Against Machine Learning Attacks

**arXiv ID:** 2601.04697 | [PDF](https://arxiv.org/pdf/2601.04697v1)

**作者:** Hongming Fei `[一作]` (National University of Singapore), Biplab Sikdar `[通讯]` (National University of Singapore)

**通讯引用:** 11746 | **OpenAlex IDs:** https://openalex.org/A5041189303

**关键词:** `Cryptography and Security` `Safty and Privacy` `Adversarial Attack` `Monte Carlo 采样`

### 📋 论文摘要

**🎯 论文内容**

研究提出了一套统一的定量框架，用条件概率与 Monte Carlo 采样方法评估延迟型 PUF 对机器学习攻击的抵抗力。

**💡 创新点**

创新点在于把攻击建模转化为无训练的概率推断，给出可比的安全下界与攻击优势度量，突破了传统经验式、模型特定的评估方式。

**🔧 技术方法**

采用概率论、条件概率理论、Monte Carlo 采样与 GPU 并行计算技术实现大规模仿真与优势估计。

**📊 数据集**

使用的“数据集”是通过正态分布随机采样生成的百万级 PUF 实例与随机挑战–响应对，模拟理想化的延迟型 PUF 行为。

**📈 对比分析**

与传统机器学习攻击基准对比，方法不依赖特定模型，能够给出连续的攻击优势曲线；对 APUF、XOR‑PUF、FF‑XOR‑PUF、CT‑PUF 等结构的评估显示出不同程度的可预测性与安全边界，验证了框架的可比性与实用性。

**⚠️ 局限性**

局限性包括：仅评估理想化参数模型，未考虑真实工艺噪声与可变性；需要较大 GPU 资源；对极端高维或极度非线性结构的精确估计仍有挑战。

---

## Do LLMs Benefit from User and Item Embeddings in Recommendation Tasks?

**arXiv ID:** 2601.04690 | [PDF](https://arxiv.org/pdf/2601.04690v1)

**作者:** Mir Rayat Imtiaz Hossain `[一作]` (University of British Columbia), Mohamed Osama Ahmed `[通讯]` (RBC Borealis)

**关键词:** `Machine Learning` `Recommendation System` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

在推荐任务中将协同过滤得到的用户和项目嵌入投射到LLM词嵌入空间，并在此基础上进行微调，以生成更精准的推荐。

**💡 创新点**

首次实现同时注入用户和任意数量项目嵌入，并使用轻量级两层MLP投射器实现独立映射，显著提升了LLM在推荐任务上的表现。

**🔧 技术方法**

使用矩阵分解（WALS）生成嵌入，双层MLP投射器，LoRA 微调以及 OpenP5 的 prompt 模板。

**📊 数据集**

Amazon Beauty、LastFM、MovieLens‑1M 三个公开交互数据集。

**📈 对比分析**

与传统协同过滤模型（SASRec、HGN 等）和文本仅输入的 LLM（ILM、Llama‑R/S/C）对比，Llama‑Embed‑Stage‑2 在 HR@5、NDCG@5 等指标上普遍领先，甚至在某些数据集上超过传统模型。

**⚠️ 局限性**

仅基于协同过滤嵌入，未利用项目语义；仅针对 decoder‑only LLM，且实验规模和任务类型有限；两阶段微调可能不够高效。

---

## ToolGate: Contract-Grounded and Verified Tool Execution for LLMs

**arXiv ID:** 2601.04688 | [PDF](https://arxiv.org/pdf/2601.04688v1)

**作者:** Yanming Liu `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 1760 | **OpenAlex IDs:** https://openalex.org/A5047459900

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了ToolGate框架，利用Hoare逻辑契约对LLM调用外部工具的前置条件和后置条件进行显式校验，从而实现工具调用的逻辑安全与可验证的状态演化。

**💡 创新点**

创新点在于将工具调用视作具有前后条件的Hoare三元组，构建显式符号状态空间，并在工具调用前后进行运行时验证，形成闭环的形式化约束，显著降低模型误调用与误信息传播。

**🔧 技术方法**

技术包括：符号状态空间（typed key‑value映射）、Hoare逻辑契约（{P}t{Q}）、基于向量检索与重排序的工具候选获取、概率推理驱动的工具调用决策、运行时后置验证与状态更新。

**📊 数据集**

使用的数据集主要为ToolBench（1.6万+ API）和MCP‑Universe（真实系统工具集合），并在这些基准上进行实验。

**📈 对比分析**

与ReACT、DFSDT、LATS、ToolChain*、Tool‑Planner等基线对比，ToolGate在ToolBench各组任务中均取得最高或接近最高的Pass Rate和Win Rate，在MCP‑Universe中也显示3‑7%的提升；在多工具链任务中显著降低工具调用步数（约30%减少）。

**⚠️ 局限性**

局限包括：目前仅覆盖文本与结构化工具，缺乏多模态与极长链协作场景；实验环境相对静态，未充分模拟真实API的延迟、速率限制和状态漂移；评价指标主要定量，缺少更细粒度的质性评估与主动信息请求能力。

---

## Mechanism Design for Federated Learning with Non-Monotonic Network Effects

**arXiv ID:** 2601.04648 | [PDF](https://arxiv.org/pdf/2601.04648v1)

**作者:** Xiang Li `[一作]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society), Yuan Luo `[通讯]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society)

**关键词:** `Computer Science and Game Theory` `Federated Learning` `Optimization` `Image`

### 📋 论文摘要

**🎯 论文内容**

在联邦学习中提出MoTS框架与SWAN机制，综合考虑网络效应和应用特定的泛化误差约束，实现社会福利最大化。

**💡 创新点**

①首次将非单调网络效应纳入FL机制设计；②提出客户端既可参与训练又可购买模型的交易共享框架；③设计SWAN机制实现预算平衡与社会最优。

**🔧 技术方法**

理论分析（泛化误差、网络效应、博弈论、拉格朗日优化）、机制设计、仿真与硬件原型实验。

**📊 数据集**

MNIST、SVHN、CIFAR‑10三种公开数据集。

**📈 对比分析**

与动态机制、改进FL机制及理论最优FL框架比较；实验显示SWAN/ MoTS 在社福利上提升高达352%，激励成本降低93%，并在不同参与成本和误差约束下保持显著优势。

**⚠️ 局限性**

仅考虑了数据量与成本的完全信息情形，未涵盖数据贡献策略、计算异质性等因素；理论基于线性/凸模型，非凸情况验证有限，对动态、信息不完全环境的鲁棒性仍需进一步研究。

---

## Succeeding at Scale: Automated Multi-Retriever Fusion and Query-Side Adaptation for Multi-Tenant Search

**arXiv ID:** 2601.04646 | [PDF](https://arxiv.org/pdf/2601.04646v1)

**作者:** Prateek Jain `[一作]` (DevRev), Constantine Caramanis `[通讯]` (University of Texas at Austin)

**通讯引用:** 8924 | **OpenAlex IDs:** https://openalex.org/A5053978837

**关键词:** `Information Retrieval` `Retrieval` `Domain Adaptation` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了 DevRev Search 数据集自动构建管线，并展示了仅对查询编码器使用 LoRA 微调即可在多租户检索系统中实现高效、无重新索引的域适配。

**💡 创新点**

创新点包括：①无人工标注的全自动检索数据集生成（融合多检索器 + LLM‑Judge 过滤）；②索引保持不变的查询端微调策略，避免昂贵的重新索引；③针对不同模型层级进行 LoRA 参数调优，提升质量‑效率平衡。

**🔧 技术方法**

使用的技术包括多检索器融合（BM25 与 7 句子检索模型）、LLM‑as‑Judge 过滤候选、LoRA 参数高效微调、以及层级敏感性分析。

**📊 数据集**

评估数据集为 DevRev Search（技术客服票据/文档）和 SciFact（科学文献检索）。

**📈 对比分析**

通过与全模型（查询+文档）微调对比，在 DevRev Search 上查询端微调甚至更优，在 SciFact 上差距不超过 1‑2%；LoRA rank 与层级目标实验显示可在不重新索引的前提下获得接近全微调的召回率，且训练成本大幅降低。

**⚠️ 局限性**

局限性包括：仅在英语单一域测试；查询端微调可能存在性能上限；未探讨跨编码器 rerank 或混合策略；LoRA 参数与不同模型的适配差异尚待进一步研究。

---

## PROMISE: Process Reward Models Unlock Test-Time Scaling Laws in Generative Recommendations

**arXiv ID:** 2601.04674 | [PDF](https://arxiv.org/pdf/2601.04674v1)

**作者:** Chengcheng Guo `[一作]` (Kuaishou Inc.), Guorui Zhou `[通讯]` (Kuaishou Inc.)

**关键词:** `Information Retrieval` `Recommendation System` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了基于Process Reward Model（PRM）的生成式推荐框架PRM-Gen，解决Semantic Drift问题。

**💡 创新点**

创新点在于引入逐步验证的PRM与PRM引导的Beam Search，实现测试时可扩展性并使小模型性能超过大模型。

**🔧 技术方法**

使用生成式推荐的分层语义ID生成、PRM、Beam Search、InfoNCE、跨注意力等技术。

**📊 数据集**

在Amazon Review公开数据集（Beauty、Sports & Outdoors）和字节跳动短视频平台Kuaishou的工业级数据上进行实验。

**📈 对比分析**

与传统检索-排序模型及其他生成式推荐基线比较，Recall@k、NDCG@k、HRecall等指标均显著提升，线上A/B测试观看时长提升0.1%–0.4%。

**⚠️ 局限性**

局限在于仍依赖语义ID分割质量，PRM训练成本和推理延迟，且在极长序列和多模态场景下的通用性待进一步验证。

---

## Bridging Temporal and Textual Modalities: A Multimodal Framework for Automated Cloud Failure Root Cause Analysis

**arXiv ID:** 2601.04709 | [PDF](https://arxiv.org/pdf/2601.04709v1)

**作者:** Gijun Park `[一作]` (Okestro AI Research Center), Gijun Park `[通讯]` (Okestro AI Research Center)

**关键词:** `Artificial Intelligence` `Anomaly Detection` `Transformer` `Retrieval-Augmented Generation` `Large Language Model` `Time Series` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出一种多模态框架 TimeRAG，用于将时间序列性能指标对齐到预训练语言模型的嵌入空间，从而实现云系统根因分析。

**💡 创新点**

创新点包括：1）单词化压缩技术将时间段压缩为单一语义标记；2）带门控交叉注意力的时间序列编码器实现时序与文本嵌入对齐；3）检索增强诊断管线，将对齐后的特征与历史事件检索结合，提升诊断准确性。

**🔧 技术方法**

使用技术包括：时间片段划分与单词化压缩、门控交叉注意力的 Time Series Encoder、检索增强生成（RAG）模型、LLM（如Qwen3、DeepSeek-R1）与向量检索库。

**📊 数据集**

使用六个云服务故障基准数据集：SockShop、OnlineBoutique、Exathlon、AIOpsArena、LemmaRCA(Prod)与LemmaRCA(Cloud)。

**📈 对比分析**

与四个基线（DeepSeek-R1、Kubeguru-Llama3.2、Mistral-7B-TimeSeriesReasoner、ChatTS-14B）进行公平对比，使用准确率与胜率评估。TimeRAG 在长序列（900 步）上最高达 48.75% 的准确率，整体表现优于基线。

**⚠️ 局限性**

局限性包括：预处理的滑窗与时间序列扩展可能引入伪影；对长期持续故障的数值细节把握不足；目前仅在离线 GPU 环境验证，实时部署效果未知；检索性能受向量库历史事件覆盖度限制。

---

## HATIR: Heat-Aware Diffusion for Turbulent Infrared Video Super-Resolution

**arXiv ID:** 2601.04682 | [PDF](https://arxiv.org/pdf/2601.04682v1)

**作者:** Yang Zou `[一作]` (Northwestern Polytechnical University), Jinyuan Liu `[通讯]` (Dalian University of Technology)

**通讯引用:** 9948 | **OpenAlex IDs:** https://openalex.org/A5100675904

**关键词:** `Computer Vision and Pattern Recognition` `Super Resolution` `Restoration` `Diffusion model` `Optical Flow` `Auto Encoder` `Video`

### 📋 论文摘要

**🎯 论文内容**

提出 HATIR 热感知扩散框架，结合热相位引导流场估计、湍流感知解码器和热感知引导，实现红外视频在大气湍流环境下的联合去噪与超分辨率恢复。

**💡 创新点**

1）将热相位先验融入扩散采样路径，构建 Phasor‑Guided Flow Estimator；2）设计 Turbulence‑Aware Decoder（包含 Turbulence Mask Gating 与 IR‑Structure‑Aware Attention），显著提升结构恢复；3）引入热感知引导机制，在扩散过程中动态调节噪声预测，提升时间一致性与热边缘保留。

**🔧 技术方法**

基于 VAE 编码/解码、扩散模型、热相位掩模（PhasorMask）、频域加权注意力、光流估计（SpyNet/PhasorFlow）以及热感知引导模块。

**📊 数据集**

FLIR‑IVSR 数据集：640 条低分辨率–高分辨率红外视频对，分辨率 1024×768，涵盖不同相机运动与场景动态，专为湍流红外 VSR 任务构建。

**📈 对比分析**

与五大 VSR 方法（MIA‑VSR、FMA‑Net、EGOVSR、IART、MGLDVSR）以及三种湍流消除＋VSR 两阶段管线（MambaTM、DATUM、Turb‑Seg）在 FLIR‑IVSR 与 M³FD 数据集上对比，HATIR 在 PSNR、SSIM、LPIPS、DISTS、VMAF 上均名列前茅，尤其在 PSNR 与 SSIM 上提升 1–3 dB，VMAF 超过 45。

**⚠️ 局限性**

计算成本相对较高，需在 GPU 上进行多步扩散采样；目前仅在红外视频场景验证，对极端湍流或不同波段红外仍需进一步评估。

---

## Optimizing Path Planning using Deep Reinforcement Learning for UGVs in Precision Agriculture

**arXiv ID:** 2601.04668 | [PDF](https://arxiv.org/pdf/2601.04668v1)

**作者:** Laukik Patade `[一作]` (Sardar Patel Institute of Technology), Sandeep Pillai `[通讯]` (Sardar Patel Institute of Technology)

**关键词:** `Robotics` `Optimization` `Reinforcement Learning` `Robotic Intelligence` `Autonomous Driving` `Agriculture Related` `Reinforcement Learning`

### 📋 论文摘要

**🎯 论文内容**

在精密农业背景下，利用深度强化学习（DRL）优化无人地面车辆（UGV）的路径规划，覆盖从二维离散环境到三维连续控制的完整实验流程。

**💡 创新点**

① 通过引入连续动作空间的Actor‑Critic算法（DDPG、TD3）实现对动态、非结构化农业场景的自适应导航；② 在Gazebo/ROS平台上实现了从静态到动态障碍的迁移学习；③ 设计了专属的奖励结构并与传统A*、Dijkstra等经典算法进行系统对比。

**🔧 技术方法**

使用的主要技术包括：深度Q网络（DQN）及其改进版（Double DQN、Dueling DQN），连续动作空间的DDPG与TD3，经验回放、目标网络、OU噪声、软更新，ROS 2与Gazebo仿真集成，Python/PyTorch实现。

**📊 数据集**

数据集主要来自自定义仿真环境：二维Pygame模拟农田格网；三维Gazebo模拟包含静态作物、石块及动态人形障碍，未使用公开真实农田传感器数据。

**📈 对比分析**

比较方法：在相同实验设置下对DQN、Double DQN、Dueling DQN的收敛速度、奖励和步骤数进行对比；在连续空间下对DDPG与TD3的收敛起点、收敛速度、稳定性、路径长度和训练时间进行对比；在3D仿真中将TD3与静态/动态环境下的成功率、奖励及碰撞率做对照。结果显示：Dueling DQN在10x10网格中比DQN快约50%，TD3在3D动态环境中成功率达95%，比DDPG高约20%稳定性、路径更短、训练时间更短。

**⚠️ 局限性**

局限性：仅在仿真环境验证，缺乏真实田间部署；迁移学习仍需大量训练；对极端天气、光照、传感器噪声鲁棒性未知；算法对极大规模农田的可扩展性未评估；奖励设计可能需针对不同作物类型调优。

---

## Air-to-Ground Communications for Internet of Things: UAV-based Coverage Hole Detection and Recovery

**arXiv ID:** 2601.04665 | [PDF](https://arxiv.org/pdf/2601.04665v1)

**作者:** Xiao Fan `[一作]` (Sun Yat-sen University), Minghua Xia `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2839 | **OpenAlex IDs:** https://openalex.org/A5052938144

**关键词:** `Information Theory` `Optimization` `Robotic Intelligence`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种基于UAV的实时覆盖缺口检测与恢复框架：巡航UAV首先扫描网格，识别覆盖空洞，随后根据空洞大小动态部署单一或四机正四面体型ABS（空中基站）进行恢复，并提供离线与在线调度算法。

**💡 创新点**

创新点在于：①将UAV巡航与ABS恢复耦合，实现覆盖缺口的即时感知与响应；②引入Delaunay三角剖分与圆覆盖理论给出ABS数量下限/上限；③采用多机潜能场与Lyapunov方法实现无领导协同、碰撞避免的三维自组织控制；④提出基于三点检测信息的在线调度策略，显著降低恢复时延。

**🔧 技术方法**

技术方法包括：UAV巡航路径规划（最短哈密顿路径/贪心最近邻）、小区BS与ABS的Nakagami‑m衰落与LoS信道模型、PPP与Matérn过程生成BS/检查点分布、Delaunay三角剖分构造协同簇、潜能场碰撞避免、控制理论中的Lyapunov不变性分析、仿真随机实验与统计分析。

**📊 数据集**

实验使用仿真数据：BS按密度λ_B=100/km²的均匀Poisson分布；检查点按λ_CP取值{10,25,50,100,150,200}/km²的Matérn硬核过程；ABS高度、功率、信道参数均按IEEE 3GPP TR 23.754设定，进行2000次Monte‑Carlo仿真得到覆盖率和ABS部署数。

**📈 对比分析**

与基于BS位置的BSL方法和固定网格（Grid）方法相比，本文方案在常规与稀疏网络中均能获得相近或更高的覆盖率（例如稀疏网络中从0.55提升至0.79），且ABS部署数显著减少（稀疏时52/59/128个，分别对应本文、BSL和Grid）。仿真还显示，在线调度在大规模密集检查点场景下平均恢复时间可比离线算法缩短约一半。

**⚠️ 局限性**

局限性包括：未考虑UAV续航、能耗与巡航时间限制；未建模卫星/地面回传链路容量与延迟；信道模型假设简化（LoS无小衰），实际环境阻挡与多路径可能导致误判；算法在极大规模部署时仍需进一步优化计算与能耗。

---

## LAMB: LLM-based Audio Captioning with Modality Gap Bridging via Cauchy-Schwarz Divergence

**arXiv ID:** 2601.04658 | [PDF](https://arxiv.org/pdf/2601.04658v1)

**作者:** Hyeongkeun Lee `[一作]` (Korea Advanced Institute of Science and Technology), Joon Son Chung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 10153 | **OpenAlex IDs:** https://openalex.org/A5038723822

**关键词:** `Sound` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Audio` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出了 LAMB 框架，利用 LLM 生成音频描述，并通过跨模态对齐器和两流适配器改进音频特征与文本嵌入的匹配，最终提升音频字幕质量。

**💡 创新点**

首次将 Cauchy–Schwarz 散度与信息互信息结合用于跨模态对齐，并设计两流适配器提取语义与时间上下文，和 Token Guide 在 LLM 文本嵌入空间直接引导解码。

**🔧 技术方法**

使用 CS 散度、InfoNCE、两流适配器（多头注意力、1D 卷积+GRU）、LLM（LLaMA‑2 7B）解码器、Token Guide、LoRA 细调等技术。

**📊 数据集**

在 AudioCaps、Clotho、WavCaps 三个音频字幕数据集上预训练并微调。

**📈 对比分析**

与现有最先进方法相比，LAMB 在 AudioCaps 上 METEOR、CIDEr、SPIDEr 等所有指标均实现了领先；在 Clotho 上保持竞争力，显著提升语义和结构一致性。

**⚠️ 局限性**

仍受限于仅在文本嵌入空间内引导，未直接建模音频细粒度语义；对大量多模态数据依赖较大，且缺少跨域泛化评估。

---

## Does Provenance Interact?

**arXiv ID:** 2601.04722 | [PDF](https://arxiv.org/pdf/2601.04722v1)

**作者:** Chrysanthi Kosyfaki `[一作]` (Hong Kong University of Science and Technology), Xiaofang Zhou `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 23680 | **OpenAlex IDs:** https://openalex.org/A5011384237

**关键词:** `Databases` `Compression` `Optimization` `Computational Efficiency` `Finance Related` `Graph Neural Network` `Time Series` `Graph`

### 📋 论文摘要

**🎯 论文内容**

本文通过引入 Temporal Interaction Networks（TIN）对流式系统、交通网络和金融网络中的时间性数据流进行建模，并提出基于节点状态的压缩索引，实现对时间维度下的数据 Provenance 的高效查询。

**💡 创新点**

创新点包括：① 将时间戳与量化交互（quantity）统一纳入图模型；② 区分离散型与液态型数据在 Provenance 跟踪中的差异；③ 设计五类时间 Provenance 查询（后向、前向、时间谱、流量谱、版本变化）并使用状态序列索引大幅压缩存储；④ 将上述模型应用于流式计算、地铁乘客迁移与金融交易三大典型场景。

**🔧 技术方法**

使用技术主要包括：Temporal Interaction Networks（图结构+时间+量化交互）、基于节点状态的压缩表示、B‑tree/时序索引、分层存储（内存、SSD、对象存储）以及针对 TIN 的查询优化与分布式一致性机制。

**📊 数据集**

实验使用 Apache Flink 的电商点击流、地铁乘客流量以及金融交易网络的真实/合成数据集，对比传统 Provenance DAG 与 TIN 模型进行评估。

**📈 对比分析**

相较于传统 Provenance 图，TIN 通过状态压缩将存储量降至原始的百万分之一，查询延迟从数百毫秒下降到十几毫秒，且在高并发窗口化流处理场景下查询速度提升 20‑50 倍。

**⚠️ 局限性**

局限性包括：① 对高频状态切换或极高聚合度节点的压缩效果有限；② 液态数据的比例混合导致 Provenance 记录膨胀；③ 需要针对不同业务场景动态调节状态粒度与压缩策略；④ 分布式索引在跨数据中心时仍面临同步与一致性挑战。

---

## Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking

**arXiv ID:** 2601.04720 | [PDF](https://arxiv.org/pdf/2601.04720v1)

**作者:** Mingxin Li `[一作]` (Alibaba Group), Junyang Lin `[通讯]` (Alibaba Group)

**通讯引用:** 3025 | **OpenAlex IDs:** https://openalex.org/A5100612233

**关键词:** `Computation and Language` `Retrieval` `Knowledge Distillation` `Transformer` `Vision Language Model` `Contrastive Learning` `Multimodality` `Image` `Text` `Video`

### 📋 论文摘要

**🎯 论文内容**

提出了基于 Qwen3‑VL 基础模型的 Qwen3‑VL‑Embedding 与 Qwen3‑VL‑Reranker 两系列模型，构建了从多模态输入到统一表示空间的端到端检索管线。

**💡 创新点**

核心创新包括：①多阶段训练策略（从大规模对比预训练到检索微调再到知识蒸馏融合）；②Matryoshka 表示学习支持可变维度嵌入；③量化感知训练（Int8/二值）实现高效存储与检索。

**🔧 技术方法**

采用 Qwen3‑VL 视觉‑语言大模型、LoRA 细化、动态分辨率与帧采样、对比学习（InfoNCE）、Cross‑Encoder 交叉注意力、CoSent、QAT（LSQ）等技术。

**📊 数据集**

训练数据来源于多模态合成、公开与内部数据的混合，覆盖文本、图像、视觉文档与视频的检索、分类、问答等任务；评测使用 MMEB‑V2、MMTEB、JinaVDR、Vidore‑v3、MSMARCO、VL3‑Syn 等基准。

**📈 对比分析**

在 MMEB‑V2 上，8B 版嵌入模型平均得分 77.8，居榜首；在多模态检索、视觉文档检索与文本检索任务上均超越现有开源与闭源基线，Reranker 8B 在多模态检索中达到 80.8/83.6 的高分。

**⚠️ 局限性**

局限性包括：在极高维度或长文本场景下性能略有下降；二值量化导致检索准确率明显衰减；当前仅支持 30+ 语言且缺乏对更丰富多模态（如音频、3D 等）的通用支持。

---

## On the Holistic Approach for Detecting Human Image Forgery

**arXiv ID:** 2601.04715 | [PDF](https://arxiv.org/pdf/2601.04715v1)

**作者:** Xiao Guo `[一作]` (Michigan State University), Xiaoming Liu `[通讯]` (Michigan State University)

**通讯引用:** 20406 | **OpenAlex IDs:** https://openalex.org/A5100409052

**关键词:** `Computer Vision and Pattern Recognition` `Anomaly Detection` `Convolutional Neural Network` `Mixture of Experts` `Large Language Model` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出了 HuForDet，一种双分支全人像伪造检测框架，可同时检测面部局部伪造与全身合成伪造。

**💡 创新点**

创新点包括：① 使用异构专家 Mixture of Experts，融合自适应 LoG 频域专家与 RGB 领域专家；② 引入上下文化伪造检测分支，利用多模态大语言模型并输出置信度，实现动态融合。

**🔧 技术方法**

技术手段包括：多尺度自适应 LoG 块、RGB 领域卷积专家、专家门控机制、基于 CLIP 的视觉编码器、Vicuna-7B 大语言模型、置信度估计与动态融合网络。

**📊 数据集**

使用了新构建的 HuFor 数据集，融合 FaceForensics++、UniAttack+ 与全身 Diff-Cele（Diffusion 生成的全身人像），共计 102 万张图像。

**📈 对比分析**

与 NPR、M2F2-Det 等现有方法在 HuFor 上进行对比，HuForDet 在 AUC 90.22% 以上、TPR95 70.87%、TPR99 33.45%，在部分伪造子集和全身伪造子集均显著优于对手。

**⚠️ 局限性**

局限性包括：在极高质量全身合成（如 Diff-Cele）时略逊于专门化方法 NPR；在极小局部伪造时仍依赖频域专家门控，易出现误检；模型规模较大，推理效率待提升。

---

## SeqWalker: Sequential-Horizon Vision-and-Language Navigation with Hierarchical Planning

**arXiv ID:** 2601.04699 | [PDF](https://arxiv.org/pdf/2601.04699v1)

**作者:** Zebin Han `[一作]` (North University of China), Zhi Han `[通讯]` (State Key Laboratory of Robotics and Intelligent Systems, Shenyang Institute of Automation, Chinese Academy of Sciences)

**关键词:** `Robotics` `Recurrent Neural Network` `Transformer` `Large Language Model` `Vision Language Model` `Multimodality` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出 SeqWalker 模型，解决 Sequential‑Horizon Vision‑and‑Language Navigation（SH‑VLN）任务，能够在同一场景中按顺序完成多任务导航。

**💡 创新点**

创新点包括：① 层次规划框架，高层使用指令分割模块（ISM）将长指令拆分为子句，降低信息冗余；② 低层采用探索‑验证（EaV）策略，结合逻辑顺序动态纠正路径；③ 通过轻量 LLM（Qwen‑0.5B）对分割子句进行嵌入，兼顾实时性与语义推理。

**🔧 技术方法**

使用 CLIP ViT‑B/32 做视觉/文本编码，CBRA+注意力的地图编码网络，GRU+交叉注意力的动作输出模块，Qwen‑0.5B 作为指令编码器，以及 Habitat 模拟平台。

**📊 数据集**

基于 IVLN 的 IR2R‑CE 数据集扩展而来，构成新的 Sequential‑Horizon IR2R‑CE（SH‑IR2R‑CE）数据集，包含多条连续轨迹和长指令。

**📈 对比分析**

与 CMA 系列、HNR、ETPNav 等 SOTA 方法在 SH‑VLN 任务上对比，SeqWalker 在 t‑nDTW、SPL、OS 等多项指标上提升 5%–6%，整体表现位居榜首；在传统 IVLN 任务上亦保持优异性能。

**⚠️ 局限性**

局限性包括：① 对极长或结构复杂的指令仍可能出现子句误分割；② 受限于轻量 LLM 的推理能力，复杂语义推理仍不如大模型；③ 需要进一步验证跨域泛化和在更真实硬件平台上的实时运行效果。

---

## Tape: A Cellular Automata Benchmark for Evaluating Rule-Shift Generalization in Reinforcement Learning

**arXiv ID:** 2601.04695 | [PDF](https://arxiv.org/pdf/2601.04695v1)

**作者:** Enze Pan `[一作]` (University of Hong Kong), Enze Pan `[通讯]` (University of Hong Kong)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Reinforcement Learning` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出了基于一维细胞自动机的控制强化学习基准，用于研究规则转移下的离分布(OOD)泛化。

**💡 创新点**

创新点在于设计了可精确拆分训练/测试规则、统一评估流程和统计报告，揭示了ID模型在规则转移下的脆弱性，并提供理论连接信息增益与信息量。

**🔧 技术方法**

使用了基于模型的规划（PlaNet、SGN‑MPC）、模型无关RL（DQN、PPO、Rainbow）以及任务推断Meta‑RL（PEARL）等算法，并给出了信息增益奖励的实现。

**📊 数据集**

数据集为可配置的1维细胞自动机规则集（0‑255），通过划分训练规则集和holdout规则集实现OOV评估。

**📈 对比分析**

对比方法包括DQN、PPO、Rainbow、PlaNet‑MPC、SGN‑MPC、SGN‑MPC‑IG、PEARL‑DQN等；ID时模型无关与模型基方法表现相近，OOV时模型基规划显著下降，PEARL在规则转移下表现最强，但整体差距大且置信区间宽。

**⚠️ 局限性**

主要局限是OOV评估任务数极少（n=5），导致结果高度不确定，且未完整扩展更强基线与更大规模复现。

---

## Thunder-KoNUBench: A Corpus-Aligned Benchmark for Korean Negation Understanding

**arXiv ID:** 2601.04693 | [PDF](https://arxiv.org/pdf/2601.04693v1)

**作者:** Sungmok Jung `[一作]` (Seoul National University), Jaejin Lee `[通讯]` (Seoul National University)

**通讯引用:** 2772 | **OpenAlex IDs:** https://openalex.org/A5100767182

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

研究并构建了针对韩语句子层面否定理解的基准测试集，系统分析韩语否定现象并量化其分布；

**💡 创新点**

创新点在于：①提出与NUBench类似但兼顾韩语语法特点的否定分类，②构造符合真实分布的4,784条多项选择题；③通过对47个LLM的评测揭示模型规模越大越能处理否定，且cloze式微调比symbol式更有效；

**🔧 技术方法**

使用大型语言模型评估、指令微调、LoRA低秩微调、Cloze与Symbol两种MCQA评测框架；

**📊 数据集**

数据来源为OpenAI Dataset Project的韩语大规模语料（约29k句）以及自建的4,784条否定多选实例；

**📈 对比分析**

在零样本和少样本设置下评测47个LLM，发现模型规模越大越好；指令微调在Symbol模式下提升但在Cloze模式下降低；cloze式微调平均提升10.5%（Symbol评测），而Symbol微调仅提升6.4%（Cloze评测），表明生成式监督更有效；

**⚠️ 局限性**

局限性包括：未覆盖词义否定，只关注句法否定；样本量相对有限；仅评估句子层面否定，未涉及更复杂的语义推理与跨句否定等。

---

## Extending Delta Debugging Minimization for Spectrum-Based Fault Localization

**arXiv ID:** 2601.04689 | [PDF](https://arxiv.org/pdf/2601.04689v1)

**作者:** Charaka Geethal Kapugama `[一作]` (University of Ruhuna), Charaka Geethal Kapugama `[通讯]` (University of Ruhuna)

**通讯引用:** 9 | **OpenAlex IDs:** https://openalex.org/A5044327946

**关键词:** `Software Engineering`

### 📋 论文摘要

**🎯 论文内容**

提出了一种利用单一失败输入对程序进行缺陷定位的方法，将 Delta‑Debugging Minimization 与谱系基础缺陷定位（SBFL）相结合，生成的中间测试用例被用于计算代码行和谓词的可疑度，从而给出缺陷行的排序。

**💡 创新点**

创新点在于：① 通过在 Delta‑Debugging 过程中收集所有通过/失败的中间输入，构造了可用于 SBFL 的执行谱；② 设计了混合可疑度评分（HybridScore），在 statement‑level 与 predicate‑level 可疑度之间按 α=0.5 加权，解决了单一层次定位的局限；③ 只需一个失败输入即可完成定位，减少了测试用例准备成本。

**🔧 技术方法**

核心技术包括：Delta‑Debugging Minimization (DDMIN)、五种主流 SBFL 算法（Tarantula、Ochiai、GenProg、Jaccard、DStar）、混合可疑度计算、基于执行谱的缺陷定位与排名；实现采用 Python/ C 代码、GitHub 提供的开源脚本。

**📊 数据集**

实验数据集为 136 个真实程序（4 个 QuixBugs Python 程序 + 132 个 Codeflaws C 程序），每个程序均包含至少一个导致错误输出的字符串输入。

**📈 对比分析**

通过 ExamScore（需检查的代码比例）和 Inspect@n（缺陷行排名是否位于前 n）进行评估。实验结果显示：在 Jaccard + Hybrid 方案下，平均 ExamScore < 0.20，缺陷行在 90% 以上实验中均排名前 3；相比单独使用 statement‑或 predicate‑level，混合方案的方差显著下降，整体性能优于传统 SBFL 方法。

**⚠️ 局限性**

局限性：① 仅适用于接收字符串输入的程序；② 需要已存在的测试 oracle 来标记通过/失败；③ 对某些不涉及条件判断的缺陷（如算术错误）时，predicate‑based 方案稳定性不足；④ 计算复杂度为 O(n²)，在极大输入上可能较慢。

---

## WebCryptoAgent: Agentic Crypto Trading with Web Informatics

**arXiv ID:** 2601.04687 | [PDF](https://arxiv.org/pdf/2601.04687v1)

**作者:** Ali Kurban `[一作]` (AI Geeks), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 8842 | **OpenAlex IDs:** https://openalex.org/A5100662197

**关键词:** `Computer Vision and Pattern Recognition` `Recommendation System` `Optimization` `Anomaly Detection` `Finance Related` `Transformer` `Large Language Model` `Agentic AI` `Reinforcement Learning` `Time Series` `Multimodality` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

提出 WebCryptoAgent 框架，将 Web 信息、行情与情绪分离为专用代理，利用 LLM 进行多模态推理并通过经验回放实现自我反思，同时在子秒级别实现风险防护，以支持加密货币的短周期交易。

**💡 创新点**

创新点在于：①两层垂直决策结构，将策略推理与实时风险控制解耦；②基于 Refexion 的上下文经验回放机制，使代理可持续自我改进；③多模态 Evidence Document 的聚合与解释，可降低噪声影响。

**🔧 技术方法**

采用大语言模型（GPT‑5、Gemini Flash 等）进行推理；嵌入检索与相似性搜索；经验回放与压缩；ATR、凯利等动态止损与仓位控制；异步两层决策管道。

**📊 数据集**

使用 2025‑01‑05 至 2026‑01‑05 的 BTCUSDT、ETHUSDT、POLUSDT 15‑分钟 OHLCV 数据，并采集对应的 Web 新闻、社交情绪等多源信息。

**📈 对比分析**

与同类 LLM 交易代理（GPT‑5、Gemini、DeepSeek、Qwen）在相同数据、初始资金 10k 美元、同一决策频率（122 次）下回测；结果显示 WebCryptoAgent 在 memory‑enabled 版本中获得更高累计收益、更低最大回撤和更佳 Sharpe；在无 memory 版本亦保持稳定或优于 baseline。

**⚠️ 局限性**

限制包括：依赖专有大型 LLM，难以完全复现；经验回放更新策略简单，长期效果未充分验证；对极端市场波动和外部数据延迟的鲁棒性仍需进一步评估。

---

## CRANE: Causal Relevance Analysis of Language-Specific Neurons in Multilingual Large Language Models

**arXiv ID:** 2601.04664 | [PDF](https://arxiv.org/pdf/2601.04664v1)

**作者:** Yifan Le `[一作]` (Zhejiang University), Yunliang Li `[通讯]` (Zhejiang University)

**通讯引用:** 3364 | **OpenAlex IDs:** https://openalex.org/A5067405994

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出CRANE框架，通过相关性归因与神经元级干预来识别并验证多语种LLM中语言特异性神经元，

**💡 创新点**

创新点在于把语言特异性定义为功能必要性而非仅凭激活相关性，提出LangSpec‑F1指标量化目标语言的功能退化，

**🔧 技术方法**

使用层级相关性传播（LRP/AttnLRP）与归一化峰度筛选神经元，并通过遮蔽干预验证功能，

**📊 数据集**

在英语、中文、越南语的多语言基准（MMLU、C‑Eval、Belebele）和开源LLaMA2‑7B‑Base/Chat模型上进行评估，

**📈 对比分析**

与激活基线LAPE及随机遮蔽对比，CRANE在相同干预预算下在目标语言上产生更大性能下降、LangSpec‑F1显著提升，

**⚠️ 局限性**

局限包括仅用峰度衡量相关性、干预方式粗糙（遮蔽），实验语言和模型范围有限，开源评测结果噪声较大。

---

## Adversarial Yet Cooperative: Multi-Perspective Reasoning in Retrieved-Augmented Language Models

**arXiv ID:** 2601.04651 | [PDF](https://arxiv.org/pdf/2601.04651v1)

**作者:** Can Xu `[一作]` (East China Normal University), Xiang Li `[通讯]` (East China Normal University)

**通讯引用:** 40636 | **OpenAlex IDs:** https://openalex.org/A5041120433

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Retrieval` `Transformer` `Large Language Model` `Reinforcement Learning` `Retrieval-Augmented Generation` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一种多视角的检索增强生成框架——Reasoner‑Verifier，通过对话式的对抗协作实现推理与验证的分工，提升多步推理的准确性。

**💡 创新点**

核心创新在于①将推理与验证拆分为两个专责代理，并以对抗协作的对话方式进行交互；②设计了“对抗性结果奖励”与“过程感知优势”两种奖励机制，既鼓励最终答案正确，也对过程中的不确定性减少和证据利用进行细粒度激励。

**🔧 技术方法**

采用多代理强化学习（Group Relative Policy Optimization），配合token‑级优势计算；利用Qwen系列大语言模型作为推理/验证主体；构造了基于检索的交互式推理动作空间。

**📊 数据集**

在多种问答基准上进行评测：NQ、HotpotQA、TriviaQA、2WikiMultiHopQA、PopQA、MuSiQue、Bamboogle；使用同一检索器和文档库，确保与基线公平对比。

**📈 对比分析**

与CoT、RAG、Search‑R1、ReSearch、WebSeer等基线在EM/F1上比较，平均提升约10%–13%（EM）与约7%–9%（F1），在小模型上甚至超过更大基线；pass@2指标亦常高于对手，证明对话式多视角提升显著。

**⚠️ 局限性**

局限性包括：①奖励设计需精细调参，过度依赖对抗机制可能导致训练不稳定；②整体计算成本高，尤其是多轮交互与两代理共存；③尚未在极端多跳或极端噪声检索场景中充分验证鲁棒性。

---

## A zone-based training approach for last-mile routing using Graph Neural Networks and Pointer Networks

**arXiv ID:** 2601.04705 | [PDF](https://arxiv.org/pdf/2601.04705v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## DSC2025 -- ViHallu Challenge: Detecting Hallucination in Vietnamese LLMs

**arXiv ID:** 2601.04711 | [PDF](https://arxiv.org/pdf/2601.04711v1)

**作者:** Anh Thi-Hoang Nguyen `[一作]` (University of Information Technology), Kiet Van Nguyen `[通讯]` (University of Information Technology)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

构建并公开了ViHallu数据集，举办了DSC2025 ViHallu共享任务，评估越南LLM的幻觉检测。

**💡 创新点**

首次在低资源语言中提供大规模幻觉检测基准，提出三类幻觉标签并设计多样化提示（事实、噪声、对抗）。

**🔧 技术方法**

使用指令调优的大型语言模型、结构化提示、LoRA微调、权重融合、温度投票以及NLI堆叠等多种技术。

**📊 数据集**

采用从ViQuAD 2.0抽取的10,000条上下文-提示-响应三元组，标注为no、intrinsic、extrinsic。

**📈 对比分析**

与编码器基线PhoBERT（Macro‑F1≈0.33）相比，最佳系统在私有测试集上达84.8%宏F1，整体排名显著提升。

**⚠️ 局限性**

仍难以准确区分内在幻觉与可信响应，缺乏片段级标注，任务仅限于越南语，且依赖人工标注。

---

## MQ-GNN: A Multi-Queue Pipelined Architecture for Scalable and Efficient GNN Training

**arXiv ID:** 2601.04707 | [PDF](https://arxiv.org/pdf/2601.04707v1)

**作者:** Irfan Ullah `[一作]` (Kyung Hee University), Young-Koo Lee `[通讯]` (Kyung Hee University)

**通讯引用:** 6512 | **OpenAlex IDs:** https://openalex.org/A5039165136

**关键词:** `Machine Learning` `Graph Neural Network` `Graph`

### 📋 论文摘要

**🎯 论文内容**

设计并实现了 MQ-GNN 多队列流水线框架，用于多 GPU 环境下的 GNN 训练。

**💡 创新点**

创新点在于引入 RaCoM 异步一致模型与可调队列大小策略，实现训练阶段的并行重叠并动态平衡计算与内存。

**🔧 技术方法**

采用多队列流水线、全局邻居采样缓存、RaCoM 异步梯度共享与周期同步技术。

**📊 数据集**

在四个大型图数据集（ogbn-proteins、ogbn-arxiv、Reddit、ogbn-products）上进行实验。

**📈 对比分析**

与 DGL、PyG、FastGCN、LADIES 等基线相比，MQ-GNN 在单 GPU 和多 GPU 配置下训练速度提升至 4.6×，GPU 利用率提升 30%，准确率基本保持在 97% 以上。

**⚠️ 局限性**

局限性包括：在极稀疏图和更大 GPU 数量下同步延迟可能导致精度略降，对梯度延迟更敏感，且需要手动调节队列大小。

---

## Forge-and-Quench: Enhancing Image Generation for Higher Fidelity in Unified Multimodal Models

**arXiv ID:** 2601.04706 | [PDF](https://arxiv.org/pdf/2601.04706v1)

**作者:** Yanbing Zeng `[一作]` (Meituan), Jie Hu `[通讯]` (Meituan)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Transformer` `Large Language Model` `Diffusion model` `Image` `Text` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

通过让多模态大型语言模型（MLLM）合成“桥接特征”并与增强文本一起注入Diffusion模型，提出Forge-and-Quench框架，实现更高保真度的图像生成。

**💡 创新点**

创新点在于：① 让MLLM主动生成虚拟视觉特征，模拟真实参考图像的视觉引导；② 双重条件（增强文本+桥接特征）在保持理解能力的同时显著提升图像细节与真实性。

**🔧 技术方法**

采用冻结的MLLM、Diffusion T2I backbone，设计Bridge Adapter将文本映射为SigLIP视觉特征，设计Injection Adapter将桥接特征注入Diffusion网络；使用文本与图像的对齐和细节控制技术。

**📊 数据集**

桥接适配器在200M图文对上训练，注入适配器在13M子集上训练；评估使用COCO‑30K、GenEval、DPG‑Bench、WISE以及自制GPT‑Fidelity等指标。

**📈 对比分析**

与基线模型在FID、GPT‑Fidelity、GenEval、DPG‑Bench、WISE等指标对比，Forge‑and‑Quench显著降低FID和提升GPT‑Fidelity，视觉质量和细节提升明显，同时保持指令对齐；人类评估亦显示更优视觉质量。

**⚠️ 局限性**

局限性包括：桥接特征对视觉编码器的鲁棒性要求高，SigLIP‑V2易受噪声影响；需要额外适配器导致推理时额外计算；在非文本输入或更复杂多模态场景中的通用性尚未完全验证。

---

## Beyond Monolithic Architectures: A Multi-Agent Search and Knowledge Optimization Framework for Agentic Search

**arXiv ID:** 2601.04703 | [PDF](https://arxiv.org/pdf/2601.04703v1)

**作者:** Yiqun Chen `[一作]` (Renmin University of China), Jiaxin Mao `[通讯]` (Renmin University of China)

**通讯引用:** 2361 | **OpenAlex IDs:** https://openalex.org/A5072119199

**关键词:** `Artificial Intelligence` `Optimization` `Reinforcement Learning` `Large Language Model` `Reinforcement Learning` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出 M-ASK 框架，采用多代理结构将搜索规划与知识管理解耦，联合使用 LLM 进行多轮信息检索与推理，最终提升多跳问答性能。

**💡 创新点**

核心创新包括：①将搜索行为代理（规划、检索、回答）与知识管理代理（摘要、更新）分离，减少上下文膨胀与噪声累积；②为每个回合提供密集的 ΔF1 奖励，实现精细信用分配；③采用参数共享的多代理 PPO 优化，使得不同角色共享同一 LLM 权重但通过专属指令区分功能。

**🔧 技术方法**

技术手段：多代理强化学习（PPO）、回合级增量奖励设计、知识状态共享、LLM 角色指令定制、检索工具集成（基于 E5 的 Wikipedia 索引），以及参数共享的跨角色优化。

**📊 数据集**

使用单跳问答数据集：Natural Questions、PopQA、AmbigQA；多跳问答数据集：HotpotQA、2WikiMultiHopQA、Musique、Bamboogle；评测基于公开的 F1 分数。

**📈 对比分析**

与多类基线（标准检索、RL 静态工作流、适配式 Agentic Search）比较，M-ASK 在单跳和多跳均取得最高平均 F1（50.09），尤其在 HotpotQA 上提升 5.82 分；训练过程稳定性优于 Search‑r1，未出现崩溃现象；相较于 DeepNote 等解耦模型，M-ASK 在精细信用分配上表现更佳。

**⚠️ 局限性**

局限性：①多轮代理调用导致推理延迟与计算成本提升；②实验仅覆盖文本问答，缺乏对代码生成、多模态等更复杂场景的验证；③依赖大规模 LLM，规模较小模型时可能表现不佳；④参数共享对指令设计要求高，若指令不精准会影响性能。

---

## PRISM: A Unified Framework for Post-Training LLMs Without Verifiable Rewards

**arXiv ID:** 2601.04700 | [PDF](https://arxiv.org/pdf/2601.04700v1)

**作者:** Mukesh Ghimire `[一作]` (Arizona State University), Xuan Zhu `[通讯]` (Amazon Web Services)

**关键词:** `Computation and Language` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一种在无标签条件下通过结合过程奖励模型和内部置信度进行强化学习的框架 PRISM，专门用于提升 LLM 的数学与代码推理能力。

**💡 创新点**

创新点在于将可解释的过程奖励模型（PRM）与模型自身的自我置信度信号联合使用，既缓解了单一内部信号的误导性，又保持了训练的稳定性；同时通过自适应权重 γ 兼顾两类信号的优点。

**🔧 技术方法**

使用的技术包括：GRPO 强化学习算法、GenPRM‑7B 过程奖励模型、self‑certainty 内部置信度、token/trajectory entropy 等代理奖励、KL 惩罚与学习率余弦衰减。

**📊 数据集**

实验数据集涵盖数学推理任务的 MATH、DAPO‑17k、Math‑500、GSM‑8k、Minerva‑Math 以及代码推理的 LiveCodeBench（LCB），并对 Qwen2.5‑3B 与 Qwen2.5‑7B 两个模型进行训练。

**📈 对比分析**

与基于地面真值奖励的 GRPO、以及仅使用 self‑certainty 的 INTUITOR 进行对比，PRISM 在所有数学基准上平均提升约 34%（与 INTUITOR 相比）并与 GRPO（地面真值）表现相当，代码推理任务中 PRISM 同样优于 INTUITOR。

**⚠️ 局限性**

局限性包括：对 PRM 的质量高度依赖，若 PRM 误判或幻觉可能导致错误策略；仍为代理优化而非真实奖励，存在奖励欺骗风险；以及生成式 PRM 的推理开销较大，训练时延显著；未在开放式、多模态或交互式场景中验证。

---

## DB-MSMUNet:Dual Branch Multi-scale Mamba UNet for Pancreatic CT Scans Segmentation

**arXiv ID:** 2601.04676 | [PDF](https://arxiv.org/pdf/2601.04676v1)

**作者:** Qiu Guan `[一作]` (Zhejiang University of Technology), Ying Tang `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 4542 | **OpenAlex IDs:** https://openalex.org/A5086134377

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Convolutional Neural Network` `Image` `Biomedical Data` `Computed Tomography`

### 📋 论文摘要

**🎯 论文内容**

提出了一种双分支多尺度Mamba UNet（DB‑MSMUNet），用于精准分割胰腺CT图像及其病灶。

**💡 创新点**

创新点包括：1) 将可变形卷积嵌入Mamba，形成Multi‑scale Mamba模块（MSMM），提升对胰腺形变和多尺度上下文的捕获；2) 采用边缘增强路径（EEP）通过辅助边缘监督精细化边界；3) 引入多层解码器（MLD）通过多尺度重参数化卷积恢复小目标细节；4) 在两条解码路径均加深监督（ADS）以强化梯度反馈。

**🔧 技术方法**

技术手段包括：Mamba状态空间模型、可变形卷积、双分支解码（边缘解码与区域解码）、注意力门、双重深度监督、膨胀重参数化卷积等。

**📊 数据集**

使用 NIH Pancreas、MSD2018 以及合作医院提供的临床胰腺肿瘤CT数据集进行实验。

**📈 对比分析**

与 UNet、nnU‑Net、TransUNet、SwinUNETR、VM‑UNet、U‑Mamba、SliceMamba 等现有方法对比，DB‑MSMUNet 在 NIH、MSD 与临床数据集上分别达 89.47%、87.59% 和 89.02% 的 Dice 分数，均优于对比模型，显示出更好的分割精度、边界保留和跨数据集鲁棒性。

**⚠️ 局限性**

局限性在于双分支解码仍依赖 CNN 模块，计算开销较大；尝试用 Mamba 替代解码器会导致性能下降，后续工作需进一步简化与优化解码器以降低算力需求。

---

## LLM-Guided Quantified SMT Solving over Uninterpreted Functions

**arXiv ID:** 2601.04675 | [PDF](https://arxiv.org/pdf/2601.04675v1)

**作者:** Kunhang Lv `[一作]` (Institute of Software Chinese Academy of Sciences), Jian Zhang `[通讯]` (Institute of Software Chinese Academy of Sciences)

**关键词:** `Artificial Intelligence` `Large Language Model` `Prompt Engineering` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

利用大语言模型对SMT中非线性实数算术与无解释函数的量化子公式进行语义导向的实例化与触发器生成，形成AquaForte框架；

**💡 创新点**

首次将LLM的数学推理能力直接嵌入量化SMT求解流程，生成具体的函数实现与实例化模式，显著降低搜索空间；

**🔧 技术方法**

大语言模型提示工程、结构化提示、语义实例化、触发器合成、与传统SMT（Z3、CVC5）结合的迭代细化；

**📊 数据集**

SMT‑COMP UFNIRA/UFLRA 基准，加上自制的 Sum‑of‑Squares 与 Mathematical Functions 600‑实例集合；

**📈 对比分析**

与 Z3、CVC5 原始策略对比，GPT‑4.1 版本在 24 秒限时下，Z3 的解题率提升 80%（785/436），CVC5 提升 183%（641/226），虚拟最佳 897/763；多轮迭代可进一步提升约 15‑17%；

**⚠️ 局限性**

在未满足实例上提升有限，主要依赖单次 LLM 调用，LLM 推理成本高，对不可满足证明的支持不足；

---

## Learning Dynamics in RL Post-Training for Language Models

**arXiv ID:** 2601.04670 | [PDF](https://arxiv.org/pdf/2601.04670v1)

**作者:** Akiyoshi Tomihari `[一作]` (University of Tokyo), Akiyoshi Tomihari `[通讯]` (University of Tokyo)

**通讯引用:** 2 | **OpenAlex IDs:** https://openalex.org/A5098162455

**关键词:** `Machine Learning` `Reinforcement Learning from Human Feedback` `Optimization` `Transformer` `Reinforcement Learning` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文通过对RL后训练（RL post‑training）的学习动态进行分析，揭示了在RL过程中模型置信度提升和输出多样性下降的原因，并提出一种基于先优化分类器的两阶段训练方法（Classifier‑First RL，CF‑RL），从而提升RL的优化效率。

**💡 创新点**

创新点主要包括：①利用经验NTK（empirical NTK）框架对RL后训练进行细粒度学习动力学拆分，区分表示层（Representation）与梯度层（Gradient）两种贡献；②发现表示层的特征高度一致会导致RL更新只提升已采样标记的概率，进而压缩输出分布；③提出CF‑RL，即先冻结所有参数只更新分类器，再进行完整RL训练，以此加速后期的梯度更新并改善模型置信度。

**🔧 技术方法**

核心技术包括：经验NTK分解、强化学习后训练（RLHF / RLVR）、梯度上升优化、GRPO策略梯度算法、以及对模型内部参数变化（如分类器范数、特征向量余弦相似度）进行监测与可视化。

**📊 数据集**

实验使用的主要数据集为Pythia 2.8B预训练模型，先进行AlpacaFarm SFT，然后在UltraFeedback数据集上使用基于ArmoRM的奖励模型进行RL后训练；对比实验还在不同阶段（SFT、GRPO、CF‑阶段）采样并评估输出多样性与奖励。

**📈 对比分析**

与传统单阶段RL训练相比，CF‑RL在第1轮RL阶段就能获得更高的Best‑of‑N奖励，并显著提升模型对高奖励样本的置信度（熵下降），同时输出多样性略有下降；实验表明CF‑RL的收敛速度更快，奖励提升更明显，但整体性能提升受限于单一模型规模与奖励设定。

**⚠️ 局限性**

局限性包括：①仅在Pythia 2.8B模型上验证，缺乏跨模型/规模的泛化评估；②使用人工设计的奖励模型（ArmoRM）而非真实人类偏好，导致对人类评价的适用性未知；③对CF‑RL机制的理论解释仍基于经验NTK假设，未对其他RL算法或更复杂奖励结构进行验证；④未深入探讨模型多样性下降对下游任务的具体影响。

---

## Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning

**arXiv ID:** 2601.04666 | [PDF](https://arxiv.org/pdf/2601.04666v1)

**作者:** Zhiyuan Chang `[一作]` (Institute of Software Chinese Academy of Sciences), Qing Wang `[通讯]`

**关键词:** `Artificial Intelligence` `Data Synthesis` `Safty and Privacy` `Adversarial Attack` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Chain-of-Thought` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一种基于指令级链式推理的prompt injection防御方法（InstruCoT），通过合成多样化攻击样本并进行指令级推理微调，使LLM能准确识别并拒绝恶意指令。

**💡 创新点**

创新点在于：①系统地合成覆盖行为偏差、隐私泄露与有害输出三大威胁维度、不同注入位置的攻击数据；②设计基于Endsley情境意识模型的三阶段指令级链式推理框架，引导LLM自我辨别冲突指令；③将生成的链式推理作为监督信号，提升模型对恶意指令的鲁棒性。

**🔧 技术方法**

核心技术包括：多场景合成生成器、基于GPT‑4.1的指令与推理模板、三阶段指令感知‑违规理解‑回应投射的链式推理模板、全参数监督微调。

**📊 数据集**

使用公开清洗数据集 Alpaca‑clean、SystemChat、Ultrachat‑Decomposed 作为原始数据，并借助 GPT‑4.1 生成攻击指令与推理，最终形成 InstruCoT‑LLM‑045F 训练集。

**📈 对比分析**

与四大基线（PromptArmor、StruQ、SecAlign、MetaSecAlign、ISE、IP）在七种主流 PI 攻击方式上进行比较，结果显示 InstruCoT 在行为偏差、隐私泄露和有害输出三维上平均防御率分别提升至 92.5%、98.0% 和 90.9%，相较基线提升 6.7%–82.5%，且在多 LLM（Llama3‑8B、Llama3.1‑8B、Qwen2.5‑7B、Qwen3‑8B）上保持高效且无显著性能退化。

**⚠️ 局限性**

主要限制包括：推理阶段需要额外的链式推理生成，导致推理延迟增加；训练与测试仅覆盖三大威胁维度，未检验更细粒度或新型攻击场景；生成的攻击指令与测试指令在内容上无重叠，可能不足以覆盖所有实际攻击变种。

---

## Integrated Framework for Selecting and Enhancing Ancient Marathi Inscription Images from Stone, Metal Plate, and Paper Documents

**arXiv ID:** 2601.04800 | [PDF](https://arxiv.org/pdf/2601.04800v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition`

---

## FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions

**arXiv ID:** 2601.04656 | [PDF](https://arxiv.org/pdf/2601.04656v1)

**作者:** Dekun Chen `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `Sound` `Generation` `Data Synthesis` `Transformer` `Large Language Model` `Reinforcement Learning` `Prompt Engineering` `Audio` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出 FlexiVoice，一种基于大语言模型的文本到语音系统，支持自然语言指令控制说话风格、参考语音控制音色，并实现零样本声音克隆；

**💡 创新点**

引入 Progressive Post‑Training (PPT) 课程，分三阶段（多模态 DPO、分离 GRPO、指令 GRPO）系统性解决风格-音色-内容冲突，实现风格与音色的解耦与复杂指令跟随；

**🔧 技术方法**

核心技术包括：LLM 作为语音生成核心、语音分词、Flow‑matching 生成 Mel、DPO（Direct Preference Optimization）对风格与音色的偏好对齐、GRPO（Group Relative Policy Optimization）实现多目标分离、使用 ALM 奖励（如 Kimi‑Audio‑7B‑Instruct）进行开放式指令评估；

**📊 数据集**

构建了 4,316 小时的指令‑语音数据集（扩展 Emilia、游戏配音、Deepseek‑V3 注释），并使用多种公开数据（Emilia、MEAD、CSEMOTIONS、InstructTTSEval 等）进行评测；

**📈 对比分析**

与多种开源基线（Parler‑TTS、PromptStyle、PromptTTS、VoxInstruct、CosyVoice2）以及闭源模型（Gemini‑pro、GPT‑4o‑mini‑TTS 等）对比，在多模态解耦、指令遵循、自然性、鲁棒性等指标上均取得显著提升；在 InstructTTSEval 上平均准确率从 63.6% 提升至 79.3%，仅比 Gemini‑pro 差距 1.0%；

**⚠️ 局限性**

局限性：对极端情绪或非预训练语料的指令仍可能出现风格泄漏；在极端音色与风格冲突场景下音色一致性略有下降；模型对多模态输入的推理时间略长；需要进一步探索更高效的多目标优化与实时应用。

---

## Automatic Classifiers Underdetect Emotions Expressed by Men

**arXiv ID:** 2601.04730 | [PDF](https://arxiv.org/pdf/2601.04730v1)

**作者:** Ivan Smirnov `[一作]` (University of Technology Sydney), David Garcia `[通讯]` (University of Konstanz)

**通讯引用:** 7063 | **OpenAlex IDs:** https://openalex.org/A5084395089

**关键词:** `Computation and Language` `Classification` `Recognition` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究使用 TalkLife 平台自标注的 1.7 百万条帖子，探讨男性与女性文本在情感识别模型中的误差差异。

**💡 创新点**

创新点在于首次用大规模真实自标注数据量化性别偏差，并提出 valence 与 salience 两种错误度量以揭示错误类型。

**🔧 技术方法**

采用词典方法、传统机器学习算法以及大语言模型（LLM）对情感进行自动分类。

**📊 数据集**

数据集来自 TalkLife 社交平台，包含 1,711,514 条带 16 种情绪标签且已标注性别的帖子。

**📈 对比分析**

对 414 种模型与情绪组合进行统计检验，结果显示男性文本的 valence 与 salience 误差在所有模型类型中均显著高于女性，且差异在 100% 的组合中均统计显著。

**⚠️ 局限性**

局限性包括仅关注二元性别、仅使用英文数据、LLM 仅限开放权重且未评估商业模型，可能导致偏差评估的完整性受限。

---

## Thinking-Based Non-Thinking: Solving the Reward Hacking Problem in Training Hybrid Reasoning Models via Reinforcement Learning

**arXiv ID:** 2601.04805 | [PDF](https://arxiv.org/pdf/2601.04805v1)

**作者:** Siyuan Gan `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**通讯引用:** 13470 | **OpenAlex IDs:** https://openalex.org/A5070337115

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种混合推理模型训练方法TNT，利用思考模式下的解答长度动态设置无思考模式的最大token限制，从而降低奖励黑客问题。

**💡 创新点**

创新点在于通过思考模式的解答部分来估计不同问题的token阈值，实现对无思考模式token使用的自适应限制，而不是统一阈值。

**🔧 技术方法**

使用强化学习（GRPO）和自定义奖励函数，结合token级策略梯度来训练模型。

**📊 数据集**

数据集包括40k道数学题的DeepScaleR训练集以及在AIME24、AIME25、Minerva、AMC23、Olympiad等五个数学基准上进行评估。

**📈 对比分析**

相较于Thinkless、AdaptThink和AutoThink，TNT在准确率与token使用的权衡（TE指标）上取得最高得分，平均token使用下降约46%，准确率提升约4%。

**⚠️ 局限性**

局限性在于仅在数学数据上验证，未涵盖其他领域，且缺乏与其他技术结合的实验。

---

## APEX: Academic Poster Editing Agentic Expert

**arXiv ID:** 2601.04794 | [PDF](https://arxiv.org/pdf/2601.04794v1)

**作者:** Chengxin Shi `[一作]` (East China Normal University), Xiang Li `[通讯]` (East China Normal University)

**通讯引用:** 40636 | **OpenAlex IDs:** https://openalex.org/A5041120433

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Vision Language Model` `Agentic AI` `Text` `Image` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出了第一套交互式学术海报编辑框架APEX，并构建了514条编辑指令的基准数据集；

**💡 创新点**

创新点在于：1) 采用多级API编辑实现局部精准控制；2) 引入审查-调整机制提升指令遵循与视觉一致性；3) 设计了VLM-as-a-judge评估协议；

**🔧 技术方法**

技术核心包括大型语言模型与视觉语言模型协同、基于API的编辑序列、可容错执行、以及视觉评估模型；

**📊 数据集**

使用了59篇2023-2025年ICLR/ICML/NeurIPS论文生成的初稿海报和人工审核的编辑指令，共514条；

**📈 对比分析**

与重生成方法(XML生成、直接图像生成)和通用幻灯片编辑方法(PPTC、Talk-to-Your-Slides、脚本编辑)比较，APEX在指令满足度和视觉一致性上提升约15个百分点，修改范围保持相近，成本与其他方法相当；

**⚠️ 局限性**

局限性包括：未使用更强大模型（如Gemini-3-Pro）；缺乏外部视觉资产检索与整合能力；对极端复杂指令的处理仍有提升空间。

---

## PyramidalWan: On Making Pretrained Video Model Pyramidal for Efficient Inference

**arXiv ID:** 2601.04792 | [PDF](https://arxiv.org/pdf/2601.04792v1)

**作者:** Denis Korzhenkov `[一作]` (Qualcomm AI Research), Amirhossein Habibian `[通讯]` (Qualcomm AI Research)

**关键词:** `Computer Vision and Pattern Recognition` `Knowledge Distillation` `Computational Efficiency` `Generation` `Diffusion model` `Flow-based Model` `Video` `Text`

### 📋 论文摘要

**🎯 论文内容**

将预训练的视频扩散模型通过低成本微调转换为金字塔式模型，并对该模型进行多步和少步推理的蒸馏研究。

**💡 创新点**

创新点包括：①使用微调而非从零训练实现金字塔化，②首次系统评估金字塔步数蒸馏（DMD 与对抗蒸馏），③将分辨率转换推广到任意上采样/下采样函数（包括 Haar 波let），④提出简化版 DMD 目标。

**🔧 技术方法**

采用的技术包括 PyramidalFlow 框架、流匹配损失、步骤蒸馏（DMD 与对抗蒸馏）、LoRA 适配器、VAE 编码/解码、近似卷积、Haar 波let 等。

**📊 数据集**

使用的数据集：80K 合成视频（由 Wan2.1-14B 生成）、350K 文本提示子集、VBench 与 VBench-2.0 评估集。

**📈 对比分析**

与原 Wan2.1（50 步）、Wan-DMD（2 步）和 Wan-Adv 等基线比较；金字塔模型在 2-2-1 调度下 FLOPs 约 4.5 倍更高效，VBench 总分与原始模型相当，VBench-2.0 略低；人类偏好实验显示与 50 步基线无显著差异，优于 2 步 DMD。

**⚠️ 局限性**

局限性包括：VBench-2.0 的创造力与可控性略低；单步生成可能出现色彩过饱和、卡通化现象；PPF 训练难以收敛；对资源受限设备的进一步优化仍待研究。

---

## Segmentation-Driven Monocular Shape from Polarization based on Physical Model

**arXiv ID:** 2601.04776 | [PDF](https://arxiv.org/pdf/2601.04776v1)

**作者:** Jinyu Zhang `[一作]` (Beijing Institute of Technology), Gonzalo R. Arce `[通讯]` (University of Delaware)

**通讯引用:** 13298 | **OpenAlex IDs:** https://openalex.org/A5005357824

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Restoration` `Image`

### 📋 论文摘要

**🎯 论文内容**

研究了一种分割驱动的单目形状从极化（SMSfP）方法，利用极化自适应区域生长与多尺度凸性先验实现单图像极化的表面重建。

**💡 创新点**

将全局极化歧义拆解为局部凸子区域的分割重建，并提出多尺度融合凸性先验以保留纹理细节，克服传统全局凸性假设导致的失真。

**🔧 技术方法**

采用极化自适应区域生长（PARG）、多尺度融合凸性先验（MFCP）、迭代线性最小二乘求解表面高度，以及极化极性、DOP、AOP等物理极化模型。

**📊 数据集**

在合成数据集A（四个模型）、合成数据集B（四个模型）以及真实世界三件模型（鹅、松鼠、仙人掌）上进行实验。

**📈 对比分析**

与Atkinson、Mahmoud、Smith等现有单目SfP方法做定量对比，MAE、RMSE和像素准确率均显著提升，尤其在11.25°阈值下达到约55%及以上的像素准确率。

**⚠️ 局限性**

仍受限于完全漫反射假设，混合镜面-漫反射、光照变化或低DOP区域的鲁棒性不足；对极化图像质量有较高要求。

---

## Smart IoT-Based Wearable Device for Detection and Monitoring of Common Cow Diseases Using a Novel Machine Learning Technique

**arXiv ID:** 2601.04761 | [PDF](https://arxiv.org/pdf/2601.04761v1)

**作者:** Rupsa Rani Mishra `[一作]` (Biju Patnaik University), Ajaya Kumar Tripathy `[通讯]` (Gangadhar Meher University)

**关键词:** `Machine Learning` `Classification` `Optimization` `Multimodality` `Time Series` `Tabular` `Agriculture Related`

### 📋 论文摘要

**🎯 论文内容**

设计并实现了基于 IoT 的可穿戴传感器与云端机器学习平台，用于实时监测并诊断牛只常见疾病。

**💡 创新点**

结合多模态传感器数据与基于遗传算法优化的支持向量机（HPOSVM）实现多疾病一体化预测，并提出完整的 CPS 框架。

**🔧 技术方法**

使用 IoT 传感网络（ESP32、温度/加速度/音频/乳EC/心率等传感器）、云端数据预处理、特征提取、HPOSVM（SVM+GA）以及 MQTT/HTTPS 通信协议。

**📊 数据集**

采用来自 5 牧场的 30 维传感特征与疾病标签的数据集，包含 12 种牛病与健康样本，收集时间为 2022–2025 年。

**📈 对比分析**

与 7 种经典分类器（SVM、KNN、LR、LDA、RF、NB、DT）在 70% 训练比例下对比，HPOSVM 达到 93.27% 准确率、93.57% 精确率、93.27% 召回率、91.96% F1、96.18% ROC AUC，整体优于基线模型。

**⚠️ 局限性**

模型仅进行健康/疾病的二分类，缺少对多疾病分级的细粒度诊断；需要大量标注数据；在低网络覆盖环境下的鲁棒性未充分验证。

---

## Skeletonization-Based Adversarial Perturbations on Large Vision Language Model's Mathematical Text Recognition

**arXiv ID:** 2601.04752 | [PDF](https://arxiv.org/pdf/2601.04752v1)

**作者:** Masatomo Yoshida `[一作]` (Doshisha University), Masahiro Okuda `[通讯]` (Doshisha University)

**通讯引用:** 4585 | **OpenAlex IDs:** https://openalex.org/A5025207272

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Adversarial Attack` `Optimization` `Vision Language Model` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出了一种利用骨架化技术的黑盒对抗攻击方法，针对视觉基础模型对数学公式图像的识别，降低搜索空间并提升攻击效率。

**💡 创新点**

创新点在于将文本区域骨架化为一维数组来缩小搜索空间，并结合CMA‑ES、TPE和随机搜索三种优化方法评估其对抗效果。

**🔧 技术方法**

采用骨架化、字符边框检测、1D数组变换、余弦相似度（基于TF‑IDF）损失、CMA‑ES、TPE、随机搜索等技术。

**📊 数据集**

构建了40幅数字化数学等式图像的新数据集，图像高度统一为50像素，确保不与现有训练集重叠。

**📈 对比分析**

通过比较全图、字符区域和骨架化区域的攻击，发现骨架化区域能显著降低余弦相似度并提升成功率；在随机搜索下，攻击成功率最高，CMA‑ES、TPE次之。

**⚠️ 局限性**

主要限制包括对文本结构复杂度的依赖、对抗样本在真实场景中的鲁棒性不完全验证，以及仅在ChatGPT等单一模型上验证转移性。

---

## Cognitive Infrastructure: A Unified DCIM Framework for AI Data Centers

**arXiv ID:** 2601.04750 | [PDF](https://arxiv.org/pdf/2601.04750v1)

**作者:** Krishna Chaitanya Sunkara `[一作]` (Independent Researcher, AI Data Center Engineering), Krishna Chaitanya Sunkara `[通讯]` (Independent Researcher, AI Data Center Engineering)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

提出了一套统一的 DCIM 3.0 框架，将语义推理、能源与热量分析、统一设备连通协议（UDCP）和自治编排融合，用于 AI 数据中心的自适应管理。

**💡 创新点**

创新点在于将知识图谱与数字孪生、统一连通协议相结合，实现端到端的自适应编排，并通过标准化连通性显著减少手动端口映射错误。

**🔧 技术方法**

使用技术包括语义知识图谱与图数据库、机器学习热量预测模型、JSON/HTTP 的 UDCP、事件驱动微服务闭环控制以及数字孪生模拟。

**📊 数据集**

实验数据主要来自 AI 训练集群的实时遥测与能耗日志，未公开公开数据集，采用内部厂商提供的遥测与功耗数据库。

**📈 对比分析**

与传统 DCIM 1.0/2.0 的对比显示，构建周期从数周缩短到数小时或数天，PUE 从 1.9 降至 1.15，能耗优化提升 10–20%，端口映射错误率降至接近 0。

**⚠️ 局限性**

局限性包括对标准化接口的高度依赖、跨厂商实施成本高、对极端负载下模型泛化性的验证不足，以及对实时遥测数据完整性的要求较高。

---

## KnowMe-Bench: Benchmarking Person Understanding for Lifelong Digital Companions

**arXiv ID:** 2601.04745 | [PDF](https://arxiv.org/pdf/2601.04745v1)

**作者:** Tingyu Wu `[一作]` (University of Chinese Academy of Sciences), Ronghao Chen `[通讯]` (Peking University)

**通讯引用:** 4 | **OpenAlex IDs:** https://openalex.org/A5109632049

**关键词:** `Artificial Intelligence` `Large Language Model` `Retrieval-Augmented Generation` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

构建了基于长篇自传文本的知识面板KnowMe‑Bench，用于评估长期记忆系统在推理、情感理解等方面的能力；

**💡 创新点**

创新点在于：① 用高密度、结构化的自传数据替代传统稀疏对话；② 通过“mnestic realignment”重建时间线，保持事件的真实因果顺序；③ 设计分层评测和LLM‑as‑Judge评判框架，强调证据可追溯的推理；

**🔧 技术方法**

技术包括：多阶段多智能体流水线（分段、原子单位抽取、时间重排、实体化），向量检索与实体图记忆（Mem0）、流式认知日志（MemOS）等外部记忆机制，以及GPT‑4o 作为评测者；

**📊 数据集**

使用三大文学文本集合（卡纳乌斯格拉德《我的奋斗》、费拉尔内托《那不勒斯小说》、普鲁斯特《追忆似水年华》）共约470万词，构成高密度自传数据集；

**📈 对比分析**

通过对比基线（原始模型、Naive RAG、Mem0、MemOS）在七项评测任务上的分数，发现：外部记忆显著提升事实与时间推理，但在情感与心理推断层面仍低于30%；在复杂任务中，MemOS在时间和洞察层面优于Mem0，说明流式记忆更适合非线性叙事；

**⚠️ 局限性**

限制在于：① 评测高度依赖LLM‑as‑Judge，仍存在主观性；② 数据处理成本高，需多智能体生成与人工校对；③ 在心理深度推理上整体分数偏低，说明现有记忆与推理机制仍不足；

---

## Tool-MAD: A Multi-Agent Debate Framework for Fact Verification with Diverse Tool Augmentation and Adaptive Retrieval

**arXiv ID:** 2601.04742 | [PDF](https://arxiv.org/pdf/2601.04742v1)

**作者:** Seyeon Jeong `[一作]` (Yonsei University), Beakcheol Jang `[通讯]` (Yonsei University)

**通讯引用:** 2799 | **OpenAlex IDs:** https://openalex.org/A5067151609

**关键词:** `Computation and Language` `Retrieval` `Recommendation System` `Retrieval-Augmented Generation` `Agentic AI` `Text` `Biomedical Data` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出Tool-MAD多代理辩论框架，赋予每个代理不同外部工具，动态检索证据并进行事实验证。

**💡 创新点**

创新点：① 代理使用异构工具（RAG + 实时搜索）实现多样化证据；② 逐轮自适应查询重写，持续更新检索内容；③ 将 faithfulness 与 answer relevance 结合为稳定性得分，驱动决策与检索迭代。

**🔧 技术方法**

技术：多代理对话（辩论）、外部检索工具（向量检索+RAG、实时搜索 API）、动态查询生成、RAGAS 评价（faithfulness、answer relevance）、Judge 评判模块。

**📊 数据集**

数据集：事实验证基准 FEVER、FEVEROUS、FaVIQ、AVeriTeC；医学 QA 集 MedQA、PubMedQA。

**📈 对比分析**

对比方法：与单代理（ReAct）、MAD、MADKE 等框架以及不同工具组合进行对比；使用 Exact Match 评价，Tool-MAD 在所有基准上均优于基线，平均提升约 18%–35%，医学集上提升至 77%+，显示显著性能优势。

**⚠️ 局限性**

局限：依赖大模型推理，计算成本高；稳定性阈值需手工调参；多轮可能出现性能下降；仍受检索工具质量与覆盖范围限制。

---

## Miner:Mining Intrinsic Mastery for Data-Efficient RL in Large Reasoning Models

**arXiv ID:** 2601.04731 | [PDF](https://arxiv.org/pdf/2601.04731v1)

**作者:** Shuyang Jiang `[一作]` (Fudan University), Yu Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 42925 | **OpenAlex IDs:** https://openalex.org/A5100445300

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了一种新的强化学习框架，通过不再使用的正同质（PH）回报转化为有价值的学习信号，利用不确定性驱动的内在奖励。

**💡 创新点**

创新点在于引入了序列级不确定性奖励、令牌级焦点信用分配和自适应优势校准，能够有效地将梯度稀缺的PH组转化为知识巩固的催化剂。

**🔧 技术方法**

使用了不确定性驱动的内在奖励机制、令牌级焦点信用分配机制和自适应优势校准技术。

**📊 数据集**

在Qwen3-4B和Qwen3-8B基础模型上评估，使用了六个不同的推理基准进行测试。

**📈 对比分析**

与其他无评论的强化学习算法（如GRPO、DAPO和REINFORCE++）进行比较，结果显示在Pass@1和Pass@K上均有显著提升，Pass@1平均提高4.58分，Pass@K提高6.66分，表明该方法在样本效率和准确性上优于竞争基线。

**⚠️ 局限性**

限制在于该工作主要集中于从正同质（PH）提示中解锁学习信号，未能在更大规模的模型（如32B）上进行扩展，且在Qwen3-8B训练中仅进行了一个周期，可能导致收敛速度较慢和部分性能未被充分挖掘。

---

## Detector-Augmented SAMURAI for Long-Duration Drone Tracking

**arXiv ID:** 2601.04798 | [PDF](https://arxiv.org/pdf/2601.04798v1)

**作者:** Tamara R. Lenhard `[一作]` (German Aerospace Center), Tobias Koch `[通讯]` (German Aerospace Center)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Object Tracking` `Video`

### 📋 论文摘要

**🎯 论文内容**

本文系统评估了基于SAMURAI的无人机RGB跟踪性能，并提出了利用YOLO‑FEDER FusionNet检测器进行检测增强的SAMURAI扩展；

**💡 创新点**

创新点在于：①首次将零射击大模型SAMURAI应用于无人机跟踪；②设计了检测器增强模块，通过连续检测引导实现初始化鲁棒性与长时序恢复；③公开了四条长时序无人机数据集。

**🔧 技术方法**

核心技术包括：SAMURAI框架（基于SAM 2的图像编码、掩码解码与记忆注意机制）、线性卡尔曼滤波器的运动建模、检测器-跟踪融合模块（条件融合与平均定位）。

**📊 数据集**

使用公开的DUT Anti‑UAV跟踪子集（20序列）和自采的R1、R2四条长时序序列（总计≈12k帧）。

**📈 对比分析**

与传统SOT、MOT以及多种检测+跟踪组合（如TransT、DiMP+FRCNN等）对比，检测增强SAMURAI在成功率、精度、归一化精度和mAP上均优于基线，长序列上成功率提升高达+0.29，误检率下降多达-0.475。

**⚠️ 局限性**

主要局限：对检测器质量高度依赖；在极小目标、遮挡或频繁进出视野的情形下仍易失效；长时序跟踪中仍受SAM 2结构限制，需进一步提升重识别与自适应机制。

---

## Belief in Authority: Impact of Authority in Multi-Agent Evaluation Framework

**arXiv ID:** 2601.04790 | [PDF](https://arxiv.org/pdf/2601.04790v1)

**作者:** Junhyuk Choi `[一作]` (Chung-Ang University), Bugeun Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 69 | **OpenAlex IDs:** https://openalex.org/A5077260647

**关键词:** `Computation and Language` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

在多代理系统中系统性地分析了基于角色的权威偏差，并通过ChatEval实现自由对话和内容控制实验。

**💡 创新点**

创新点在于首次从法国与雷文的权力理论出发，将权威角色划分为合法、亲属和专家三类，并揭示权威偏差源于角色本身而非对话内容。

**🔧 技术方法**

采用ChatEval多代理评估框架、ChatGPT API（GPT‑4o、DeepSeek R1）以及对话分析方法（Agreement 与 Cohen's Kappa）。

**📊 数据集**

使用的评估数据集为 FairEval（80例）和 Topical‑Chat（60例）。

**📈 对比分析**

通过对比不同模型、不同权力类型在12轮对话中的 Agreement 与 Kappa 变化，发现 DeepSeek R1 对权威偏差更敏感；Expert 与 Referent 角色的影响力显著高于 Legitimate。

**⚠️ 局限性**

主要局限在于仅测试两款模型，且实验聚焦于评估任务，未涵盖创作、技术解决等多样化多代理场景。

---

## AgentOCR: Reimagining Agent History via Optical Self-Compression

**arXiv ID:** 2601.04786 | [PDF](https://arxiv.org/pdf/2601.04786v1)

**作者:** Lang Feng `[一作]`, Bo An `[通讯]`

**关键词:** `Machine Learning` `Compression` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Vision Language Model` `Image` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出 AgentOCR 框架，将 LLM 代理的交互历史转化为渲染图像，并通过段式光学缓存与自我压缩来显著降低 token 用量。

**💡 创新点**

创新点在于：①将文本历史压缩为高密度视觉表示；②按段缓存重用渲染结果；③利用强化学习让代理主动调节压缩率，实现信息密度与效率的动态平衡。

**🔧 技术方法**

使用技术包括：视觉语言模型 Qwen2.5‑VL、GRPO 强化学习、OCR 渲染、哈希缓存机制以及压缩奖励策略。

**📊 数据集**

实验数据集包括 ALFWorld 以及多种搜索式 QA 任务（单跳、多跳、NQ、TriviaQA、HotpotQA、Wiki、MuSiQue 等）。

**📈 对比分析**

与文本代理、OCR 无 RL、Text+GRPO 等基线比较，AgentOCR 在保留 95% 以上任务成功率的同时，平均 token 消耗降低 50–60%，峰值减少 60–80%。

**⚠️ 局限性**

局限性：依赖未针对 OCR 优化的 VLM；渲染参数固定，缺乏敏感性分析；仅适用于可文本化的历史，难以处理纯视觉信息。

---

## Defocus Aberration Theory Confirms Gaussian Model in Most Imaging Devices

**arXiv ID:** 2601.04779 | [PDF](https://arxiv.org/pdf/2601.04779v1)

**作者:** Akbar Saadat `[一作]` (Iranian railways), Akbar Saadat `[通讯]` (Iranian railways)

**关键词:** `Computer Vision and Pattern Recognition` `Depth Estimation`

### 📋 论文摘要

**🎯 论文内容**

通过光学理论与数值分析，验证常规相机中散焦算子可用高斯模型近似，得到平均绝对误差<1%

**💡 创新点**

首次将衍射限散焦畸变理论与光学传递函数相结合，给出多参数相机组合下的高斯模型有效范围，并提供可直接用于深度估计的参数表

**🔧 技术方法**

几何光学、衍射光学、光学传递函数（OTF）解析、黑体辐射模型、统计误差评估（MAE、RMSE）

**📊 数据集**

未使用实际图像数据集，而是基于理论模型生成1225组相机参数组合（焦距、光圈、像素尺寸、深度范围等）

**📈 对比分析**

采用高斯拟合与OTF曲线的平均绝对误差比较，阈值设为MAE≤0.01；在1225组记录中有157组满足条件，说明模型高度精确，MAE最大不超过1%

**⚠️ 局限性**

仅考虑衍射限、单色光且未包含球面、彗形、非球面、色差等镜头像差；未在真实图像上验证，且适用于深度±10%和频率<1周期/像素的场景

---

## SciIF: Benchmarking Scientific Instruction Following Towards Rigorous Scientific Intelligence

**arXiv ID:** 2601.04770 | [PDF](https://arxiv.org/pdf/2601.04770v1)

**作者:** Encheng Su `[一作]` (Shanghai AI Laboratory), Houqiang Li `[通讯]` (Shanghai AI Laboratory)

**关键词:** `Artificial Intelligence` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Prompt Engineering` `Text` `Benchmark` `Physics Related`

### 📋 论文摘要

**🎯 论文内容**

本文提出并构建了一套名为SciIF的科学指令遵循基准，评估大模型在满足科学约束（如条件、语义和流程）并给出可审计证据时的表现；

**💡 创新点**

创新点在于：① 引入固定的约束目录并要求模型在答案中显式提供满足约束的证据；② 设计审计友好的评估协议，将答案正确性与约束遵循分离；③ 通过多约束组合检测模型的组合推理瓶颈；

**🔧 技术方法**

技术方法包括：大模型的提示-生成-审计（generate‑then‑audit）流程；规则化检查与LLM审计相结合；SFT和基于验证器的RL进一步提升模型的约束遵循能力；

**📊 数据集**

数据集由334个大学级科学问题组成，覆盖生物、化学、材料与物理四个学科；每个问题配有从10个原子约束中选出的若干约束；此外提供910个训练实例用于SFT与RL；

**📈 对比分析**

实验对比了多款闭源与开源模型，发现答案正确率普遍超过80%，但严格多约束通过率低于30%；通过SciIF微调的模型在IFEval（严格）和MMLU-Physics等外部评测中均显著提升；

**⚠️ 局限性**

局限性：仅覆盖文本类问题，未涉及多模态科学任务；约束目录与问题范围有限；在高约束数量时仍存在显著的合规率下降。

---

## Differential syntactic and semantic encoding in LLMs

**arXiv ID:** 2601.04765 | [PDF](https://arxiv.org/pdf/2601.04765v1)

**作者:** Santiago Acevedo `[一作]` (Scuola Internazionale Superiore di Studi Avanzati), Marco Baroni `[通讯]` (Catalan Institute of Research and Advanced Studies)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究了大语言模型内部层对句子表示中句法与语义的线性编码与可分离性，使用句法/语义“centroid”进行消融实验。

**💡 创新点**

创新点在于通过简单的平均中心化和线性投影消融，证明句法与语义信息在LLM内部可被线性编码且在不同层次上相对独立，首次从几何角度分离两类语言信息。

**🔧 技术方法**

采用线性投影消融、邻域相似度度量、POS模板匹配、跨语言翻译平均、语义与句法中心化等技术，并以DeepSeek‑V3、Qwen2‑7b、Gemma3‑12b三种模型验证。

**📊 数据集**

使用约2000对句法匹配句子、英语改写句子以及6语种（中文、西班牙语、意大利语、土耳其语、德语、阿拉伯语）翻译句子，构成句法与语义对照数据集。

**📈 对比分析**

通过邻域相似度和线性probe（POS分类、改写召回）评估消融效果；消融句法中心化后语义相似度在中层显著下降，消融语义中心化后句法相似度基本保持；probe表现与表格一致，表明两类信息可部分解耦。

**⚠️ 局限性**

局限包括样本量有限、句子长度短小、消融仅部分有效、未系统评估模型规模与训练目标影响、未验证更大模型或更长句子、未对生成过程进行干预。

---

## Parallelizing Node-Level Explainability in Graph Neural Networks

**arXiv ID:** 2601.04807 | [PDF](https://arxiv.org/pdf/2601.04807v1)

**作者:** Oscar Llorente `[一作]`, Miguel Familiar `[通讯]`

**关键词:** `Machine Learning` `Explainability and Interpretability` `Graph Neural Network`

### 📋 论文摘要

**🎯 论文内容**

介绍了 elsarticle.cls 这个 LaTeX 类文件的功能、使用方式、可选参数以及常用环境，旨在方便 Elsevier 期刊投稿排版。

**💡 创新点**

在继承 article.cls 的基础上，提供了预印本默认格式、与 natbib/geometry/graphicx 等主流宏包的无缝兼容，并新增了简化的定理、列表、浮动、双盲等功能，显著降低与其他宏包冲突的风险。

**🔧 技术方法**

主要采用 LaTeX 宏包编程技术，利用 natbib、geometry、graphicx、txfonts 等常用宏包实现引用、版面、图形、字体等功能；通过自定义命令与环境来简化排版流程。

**📊 数据集**

该文档不涉及实验数据集，示例代码仅用于展示排版功能与语法。

**📈 对比分析**

并未进行实验比较；但说明了相较旧版 elsart.cls，elsarticle.cls 在排版兼容性、功能完整性和易用性方面具有明显优势。

**⚠️ 局限性**

局限性包括：需要在符合 LaTeX 环境的前提下使用；若使用大量自定义宏包，仍可能产生冲突；对公式在双栏排版时的断行需要作者手动检查；并不提供内容校对或排版错误检测工具。

---

## MPM-LLM4DSE: Reaching the Pareto Frontier in HLS with Multimodal Learning and LLM-Driven Exploration

**arXiv ID:** 2601.04801 | [PDF](https://arxiv.org/pdf/2601.04801v1)

**作者:** Lei Xu `[一作]`, Chenglong Xiao `[通讯]` (Shantou University)

**通讯引用:** 84 | **OpenAlex IDs:** https://openalex.org/A5049047722

**关键词:** `Hardware Architecture` `Graph Neural Network` `Large Language Model` `Prompt Engineering` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出基于多模态预测模型与LLM驱动的高层综合设计空间探索框架MPM-LLM4DSE。

**💡 创新点**

结合源代码语义与控制流图的多模态特征融合，并通过提示工程引导LLM生成高质量pragma配置。

**🔧 技术方法**

使用ECoGNN+CodeBERT的多模态融合模型、多头注意力融合、LLM优化器（GPT‑4o/Qwen3）以及PEODSE提示策略。

**📊 数据集**

15个MachSuite/PolyBench基准，共4353个图文样本；测试集包含6个未见核。

**📈 对比分析**

与GNN‑DSE、HGBO、IronMan‑Pro、ProgSG等SOTA在RMSE上对比，MPM在延迟、LUT、DSP、FF、BRAM上分别比ProgSG低10.25倍；LLM4DSE在ADRS上相对NSGA‑II、SA、ACO、LLMMH平均提升46%至32%，且运行时间更短。

**⚠️ 局限性**

对LLM依赖API导致通信延迟，模型规模大，缺乏本地化小型模型和跨平台适配能力。

---

## SRU-Pix2Pix: A Fusion-Driven Generator Network for Medical Image Translation with Few-Shot Learning

**arXiv ID:** 2601.04785 | [PDF](https://arxiv.org/pdf/2601.04785v1)

**作者:** Xihe Qiu `[一作]` (Shanghai University of Engineering Science), Liang Liu `[通讯]` (Fudan University)

**通讯引用:** 10383 | **OpenAlex IDs:** https://openalex.org/A5100322376

**关键词:** `Computer Vision and Pattern Recognition` `Image Translation` `Generation` `Convolutional Neural Network` `Generative Adversarial Network` `Image` `Biomedical Data` `Magnetic Resonance Imaging`

### 📋 论文摘要

**🎯 论文内容**

本文提出了SRU‑Pix2Pix框架，将SEResNet和U‑Net++融合进Pix2Pix生成器，实现了在少样本医学图像翻译中的高质量、多尺度结构保持。

**💡 创新点**

创新点在于：①通过SEResNet的通道注意力提升关键结构特征表示；②采用U‑Net++的密集多尺度跳跃连接增强细节恢复；③使用2.5D输入策略兼顾空间上下文与计算效率。

**🔧 技术方法**

主要技术包括：生成对抗网络（Pix2Pix）、Squeeze‑and‑Excitation残差网络、U‑Net++解码器、PatchGAN判别器、复合损失（对抗+L1+MS‑SSIM）以及Adam优化器。

**📊 数据集**

数据集：BraTS 2023（T1→T2、T1→FLAIR、T2→FLAIR）、IXI（PD→T2）和BraTS 2019（零样本转移）。

**📈 对比分析**

与CycleGAN、NICE‑GAN、ResViT、BBDM等基线比较，SRU‑Pix2Pix在PSNR、SSIM、LPIPS、MS‑SSIM、MSE、NMSE等指标上均显著优于对手，尤其在300张样本的少样本情境下保持稳定性能，并能实现跨数据集零样本迁移。

**⚠️ 局限性**

局限性包括：仅评估MRI；2.5D策略无法完整捕获3D结构；需配对数据，未验证对无配对场景的适应；缺乏放射科医生的临床评估。

---

## GeM-VG: Towards Generalized Multi-image Visual Grounding with Multimodal Large Language Models

**arXiv ID:** 2601.04777 | [PDF](https://arxiv.org/pdf/2601.04777v1)

**作者:** Shurong Zheng `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jinqiao Wang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Object Tracking` `Segmentation` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Chain-of-Thought` `Vision Language Model` `Image` `Video` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出了一种多模态大型语言模型GeM‑VG，能够在多图像场景中进行通用视觉定位，并保持单图像定位与多图像理解的强大能力。

**💡 创新点**

创新点包括：①构建跨图像关系与推理任务的三类任务分类体系；②发布涵盖240k样本、跨图像关联与多目标的MG‑Data‑240K数据集；③提出融合链式推理（CoT）与直接答复的混合强化学习微调策略，并设计多维度规则奖励函数；④将R1‑类强化学习应用于多图像定位任务。

**🔧 技术方法**

使用技术包括：多模态LLM（基于Qwen2‑VL‑7B）架构、可变形视觉编码、投影层、统一输出表示；三阶段微调策略（SFT→CoT‑SFT→GRPO RL）；规则奖励设计（格式、图像、精度、召回）；奖励调制的双阶段平衡与长度惩罚。

**📊 数据集**

使用的数据集有：MG‑Data‑240K（从COCO、D³、Ego‑Exo4D、MVTrack、STAR等构建的多图像关系与多目标样本）；MGrounding‑630k；单图像与视频定位基准（ODINW、LLMSeg、ReasonVOS、ReVOS）；多图像理解基准（MuirBench、BLINK、MIBench、MMIU）。

**📈 对比分析**

通过在MIG‑Bench、MC‑Bench、ODINW、LLMSeg、ReasonVOS、ReVOS、MuirBench等多项基准进行评估，GeM‑VG在多图像定位上分别比前沿模型提升2.0%和9.7%，在单图像定位上比基线提升9.1%，在多图像理解任务上保持竞争力。

**⚠️ 局限性**

局限性包括：仍受限于训练数据分布，复杂跨图像推理仍存在挑战；强化学习训练成本高，需大量GPU资源；对非常稀有或极端场景的泛化能力尚未充分验证。

---

## AT$^2$PO: Agentic Turn-based Policy Optimization via Tree Search

**arXiv ID:** 2601.04767 | [PDF](https://arxiv.org/pdf/2601.04767v1)

**作者:** Zefang Zong `[一作]` (Tencent Inc), Jie Jiang `[通讯]` (Tencent Inc)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Optimization` `Reinforcement Learning` `Agentic AI` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出一种针对多轮Agentic RL的统一框架AT2PO，旨在通过树搜索提升探索质量、解决稀疏奖励的信用分配，并让策略更新与多轮决策结构保持一致。

**💡 创新点**

创新点包括①基于节点熵的自适应树展开来提升探索多样性；②利用树结构进行按轮信用分配，将稀疏终点奖励反向传播为细粒度的学习信号；③设计了按轮重要性采样与截断的策略优化（ATPO），在保持算法通用性的同时显著提升训练稳定性与性能。

**🔧 技术方法**

主要技术包含树搜索与熵导向展开、蒙特卡罗自举的节点价值估计与优势计算、按轮重要性比和裁剪的PPO变体，以及与LLM交互的ReAct工具调用框架。

**📊 数据集**

在七个知名问答基准上验证：多跳问答（HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle）和单跳问答（Natural Questions、TriviaQA、PopQA），使用Qwen3‑4B、Qwen3‑8B、Qwen2.5‑7B三种LLM作为骨干模型。

**📈 对比分析**

与GRPO、DAPO、GSPO、AEPO、Tree‑GRPO等主流RLVR与Agentic RL基线对比，AT2PO平均提升高达1.84个百分点，在多跳任务上更显著；同时保持熵的稳定性，避免早期熵崩溃。

**⚠️ 局限性**

局限在于树展开需要多轮顺序计算，导致相较于线性rollout有额外的计算开销；并且在现有实现中并未充分利用并行化，未来需改进效率并进一步验证在更广泛环境中的泛化能力。

---

## Revisiting Judge Decoding from First Principles via Training-Free Distributional Divergence

**arXiv ID:** 2601.04766 | [PDF](https://arxiv.org/pdf/2601.04766v1)

**作者:** Shengyin Sun `[一作]` (City University of Hong Kong), Chen Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 4362 | **OpenAlex IDs:** https://openalex.org/A5101866773

**关键词:** `Computation and Language` `Computational Efficiency` `Optimization` `Transformer` `Large Language Model` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

通过将判别解码的“关键性”信号与模型内部的KL散度关联，提出了一个无需监督的训练免费KL阈值判别器，实现了与传统判别器相当或更优的加速效果。

**💡 创新点**

证明学习判别器与KL散度共享相同的logit基元，揭示了两者本质相同，从而用简单的分布差异统计替代昂贵的监督训练。

**🔧 技术方法**

利用KL散度阈值、对数几率差（Δij）和软最大化等技术，构建训练无关的判别机制，并在vLLM框架中实现并行验证。

**📊 数据集**

在公开基准GSM8K、MATH‑500‑Hard、LiveCodeBench和MMLU‑Pro上进行实验，覆盖数学推理、代码生成和通用知识问答。

**📈 对比分析**

与AutoJudge、Vanilla SP、Top‑K、Entropy等基线相比，KL阈值在保持约1%‑2%准确率下降的前提下，显著提升了平均可接受令牌数（MAT）和速度（最高可达1.6×‑1.7×），在大模型和长链推理场景尤为突出。

**⚠️ 局限性**

尚未在极大模型（如405B）上验证；KL散度计算尚未实现高效定制核，可能导致额外的计算开销，影响最优速度。

---

## Orion-RAG: Path-Aligned Hybrid Retrieval for Graphless Data

**arXiv ID:** 2601.04764 | [PDF](https://arxiv.org/pdf/2601.04764v1)

**作者:** Zhen Chen `[一作]` (City University of Hong Kong), Jianping Wang `[通讯]` (City University of Hong Kong)

**通讯引用:** 14044 | **OpenAlex IDs:** https://openalex.org/A5100356291

**关键词:** `Artificial Intelligence` `Retrieval` `Generation` `Large Language Model` `Retrieval-Augmented Generation` `Text` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

开发了轻量级的路径式结构化检索增强生成框架 Orion‑RAG，能在无图谱、碎片化的文本中高效检索并生成准确答案。

**💡 创新点**

通过局部生成“路径标签”代替全局知识图谱，实现低复杂度的结构诱导和实时增量索引，并用多层混合检索与路径提示提升检索精度与可解释性。

**🔧 技术方法**

结合LLM标签生成、双层检索（稀疏+稠密+路径索引）、加权RRF重排序、查询重写与文档裁剪，以及结构化上下文注入的生成模型。

**📊 数据集**

在 Mini‑Wiki、FinanceBench 以及自构造的碎片化企业档案集合 SeaCompany 上进行评测。

**📈 对比分析**

与七种强基线（稠密、稀疏、混合、ReAct、DeepSieve、RAPTOR 等）对比，Orion‑RAG 在 Hit Rate、Precision、BERTScore、ROUGE‑L 上均领先，FinanceBench 上 Precision 提升 25.2%，SeaCompany 上 Hit Rate 97.3% 与 16.9% 精度提升。

**⚠️ 局限性**

依赖可提取实体的文本；对块大小、裁剪阈值和提示模板等超参敏感；在抽象或无实体段落的场景下路径稀疏导致检索效率下降。

---

## PILOT-Bench: A Benchmark for Legal Reasoning in the Patent Domain with IRAC-Aligned Classification Tasks

**arXiv ID:** 2601.04758 | [PDF](https://arxiv.org/pdf/2601.04758v1)

**作者:** Yehoon Jang `[一作]` (Pukyong National University), Sungchul Choi `[通讯]` (Pukyong National University)

**通讯引用:** 3130 | **OpenAlex IDs:** https://openalex.org/A5084523518

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了PILOT‑Bench数据集与基准，用于评估大语言模型在专利PTAB异议诉讼中的法律推理能力，采用IRAC框架设计了Issue Type、Board Authorities和Subdecision三项分类任务。

**💡 创新点**

创新点在于首次将18K个PTAB异议决定与USPTO专利文本在案件级别上对齐，并通过LLM生成的Opinion Split消除结论泄露，提供了可复现的IRAC对齐任务与标注方案。

**🔧 技术方法**

技术方法包括在零样本设置下使用闭源与开源LLM进行结构化输出的多标签/多类分类，实验探讨了角色拆分、合并与主张文本增补等输入变体，以及使用Exact Match、Macro‑F1、Micro‑F1和Accuracy等指标评估模型性能。

**📊 数据集**

数据集来源于PTAB的元数据与决议（≈25K PDF转文本），结合USPTO的XML专利文本（标题、权利要求、说明书），构建15K个Opinion Split实例，并标注23个细粒度的裁决标签及其粗粒度映射。

**📈 对比分析**

对比实验显示闭源模型在Issue Type上可达0.75+Micro‑F1、Subdecision约0.6 Accuracy，而最佳开源模型Qwen‑8B仅约0.56 Micro‑F1；所有模型在标签长尾处表现欠佳，加入全部权利要求文本有时会削弱Board Authorities任务性能。

**⚠️ 局限性**

局限性包括仅覆盖异议诉讼而非AIA程序、OCR与LLM拆分缺乏人工校验导致潜在错误、标签分布极度不平衡、缺少生成式IRAC应用阶段任务以及对专业法律审核的依赖。

---

## ProFuse: Efficient Cross-View Context Fusion for Open-Vocabulary 3D Gaussian Splatting

**arXiv ID:** 2601.04754 | [PDF](https://arxiv.org/pdf/2601.04754v1)

**作者:** Yen-Jen Chiou `[一作]` (National Yang Ming Chiao Tung University), Yuan-Fu Yang `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 151 | **OpenAlex IDs:** https://openalex.org/A5040566036

**关键词:** `Computer Vision and Pattern Recognition` `Retrieval` `Segmentation` `Gaussian Splatting` `Contrastive Learning` `Point Cloud`

### 📋 论文摘要

**🎯 论文内容**

提出 ProFuse，一种无需渲染监督的直接注册框架，用密集对应引导的 3D Gaussian Splatting 生成可对自然语言查询进行高效语义检索的 3D 场景表示。

**💡 创新点**

创新点在于通过预注册阶段利用密集对应同时初始化高质量几何并聚合跨视角掩码为 3D Context Proposals，再将每个提案的全局语言特征无梯度地直接融合到 Gaussians，从而在不进行渲染监督的情况下实现跨视角语义一致性和掩码内聚合。

**🔧 技术方法**

主要技术包括：密集对应网络（基于 DINOv2 训练的稀疏-细粒度匹配）、SAM 语义分割与 CLIP 嵌入、基于顶点权重的 Gaussian 注册、连通图聚类生成 3D Context Proposals、FAISS PQ 进行高效文本检索。

**📊 数据集**

在 LERF‑OVS 和 ScanNet 两个公开数据集上进行评估，分别用于 3D 对象选择和点云语义理解。

**📈 对比分析**

与现有渲染监督和注册基准（LangSplat、LEGaussians、OpenGaussian、Dr. Splat 等）相比，ProFuse 在 LERF‑OVS 上的 5cm‑IoU 与 mAcc 均超过 5%（最高 68.18%/79.66%），在 ScanNet 上的 mIoU 与 mAcc 均高 2–3%（最高 69.38%/60.90%），且场景语义绑定时间仅约 5 分钟，约为 SOTA 的 2 倍快。

**⚠️ 局限性**

局限性包括：仍依赖高质量的多视图对应与 SAM 分割，无法处理极端遮挡或低纹理区域；缺乏对动态场景的适应；以及在大规模场景下对 GPU 内存与计算资源仍有一定需求。

---

## Intraday spatiotemporal PV power prediction at national scale using satellite-based solar forecast models

**arXiv ID:** 2601.04751 | [PDF](https://arxiv.org/pdf/2601.04751v1)

**作者:** Luca Lanzilao `[一作]` (Bern University of Applied Sciences), Angela Meyer `[通讯]` (TU Delft)

**关键词:** `Machine Learning` `Recurrent Neural Network` `Optical Flow` `Diffusion model` `Auto Encoder` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

构建并评估了一套基于卫星观测和物理数值天气预报的时空光伏发电预报框架，验证了其在瑞士6400余座光伏站点上的准确性。

**💡 创新点**

首次在全国尺度上将卫星驱动的概率预测模型与传统数值天气预报模型进行比较，并提出了将卫星光照信息与站点特定机器学习模型相结合的发电转化方法。

**🔧 技术方法**

采用了光流、卷积LSTM、变分自编码器+扩散模型（SHADECast）、卫星光照的自编码网络（IrradianceNet）以及ECMWF IFS-ENS的物理模型，并用XGBoost进行站点级功率回归。

**📊 数据集**

使用EUMETSAT HANNA 15分钟分辨率的卫星光照数据以及2019‑2020年瑞士6434座光伏站的实测发电数据。

**📈 对比分析**

通过均方根误差、连续概率评分、预测区间覆盖率等指标对比，卫星模型在短时（≤2小时）预报精度优于IFS-ENS，SHADECast在概率分布上最可靠，IrradianceNet在点预测上误差最低。

**⚠️ 局限性**

主要限制包括高海拔地区卫星光照检索精度下降、模型对山地地形效应和雪覆掩的处理不足，以及单站XGBoost模型对缺失历史数据的依赖。

---

## AM$^3$Safety: Towards Data Efficient Alignment of Multi-modal Multi-turn Safety for MLLMs

**arXiv ID:** 2601.04736 | [PDF](https://arxiv.org/pdf/2601.04736v1)

**作者:** Han Zhu `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18216 | **OpenAlex IDs:** https://openalex.org/A5045081171

**关键词:** `Computation and Language` `Safty and Privacy` `Reinforcement Learning from Human Feedback` `Reinforcement Learning` `Supervised Fine-Tuning` `Large Language Model` `Multimodality` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文通过构造一个多模态对话安全数据集 InterSafe-V 并提出 AM^3Safety 训练框架，以提升多模态大模型在多轮对话中的安全性。

**💡 创新点**

创新点在于：①使用模型间交互生成对话，避免高成本人工标注；②结合冷启动拒绝学习与 GRPO fine‑tune，并设计基于安全方差的轮次加权双目标奖励，使模型在保持有用性的同时显著降低攻击成功率。

**🔧 技术方法**

采用的技术包括：RLHF、GRPO、双目标奖励函数、基于安全方差的轮次加权、拒绝模板学习与监督式 fine‑tune。

**📊 数据集**

使用的数据集为 InterSafe-V（11,270 条多模态对话 + 500 个拒绝型 VQA 例子）以及公开的 Qwen2.5‑VL‑7B‑Instruct、LLaVA‑NEXT‑7B 基础模型。

**📈 对比分析**

与现有 RLHF‑V、Safe RLHF‑V、MM‑DPO、SPA‑VL 等方法比较，实验显示在 SafeMT、JailbreakV、MM‑SafetyBench 等安全基准上，攻击成功率下降超过 10%，无害性提升至少 8%，有用性提升 13% 以上，同时在通用推理、对话等任务的性能保持甚至略有提升。

**⚠️ 局限性**

局限性包括：未在专门的对抗性 jailbreak 基准上评估鲁棒性；仅针对通用多模态模型；训练过程依赖外部评判模型，可能带来偏差；缺乏自我纠错机制。

---

## AIVD: Adaptive Edge-Cloud Collaboration for Accurate and Efficient Industrial Visual Detection

**arXiv ID:** 2601.04734 | [PDF](https://arxiv.org/pdf/2601.04734v1)

**作者:** Yunqing Hu `[一作]` (Institute of Computing Technology), Wen Ji `[通讯]` (Institute of AI for Industries)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Classification` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Image` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

设计并实现了 AIVD 框架，在边缘端使用轻量级检测器进行快速定位，在云端使用多模态大型语言模型进行细粒度分类与结构化语义生成，实现了工业视觉检测的精准定位与高质量语义输出。

**💡 创新点**

创新点包括：① 视觉‑语义协同微调策略（视觉上下文扩展+多维增强+语义提示），显著提升小目标语义识别与一致性；② 异构资源感知动态调度算法，基于 CPU 空闲、队列拥塞、带宽与延迟的线性融合评分，实现了跨节点自适应负载分配与故障容错。

**🔧 技术方法**

使用技术包括：YOLOv12s 轻量级检测、Qwen2‑VL‑7B/LLaVA‑V1.6‑mistral‑7B/InternVL3.5 大型多模态模型、LoRA/QLoRA 微调、视觉‑语义协同增强、动态调度算法、CUDA + PyTorch + Ray 进行分布式部署。

**📊 数据集**

采用 DeepPCB 与 HRIPCB 两个工业 PCB 缺陷数据集进行实验，涵盖不同规模与噪声水平的缺陷。

**📈 对比分析**

与零射、标准 QLoRA/LoRA 微调及轮询（RR）和静态资源感知调度（SRA）比较，AIVD 在分类准确率上提升 0.7‑0.9、吞吐量提升约 42.6%、资源消耗降低 13.5%，延迟平均降低 15%，在三种网络/负载场景均表现最佳。

**⚠️ 局限性**

局限性在于：仍依赖边缘检测器的定位精度；调度权重需手工设定且对极端噪声或极小目标的鲁棒性需进一步验证；在极低算力或极低带宽条件下，云端推理可能仍成为瓶颈。

---

## Training a Custom CNN on Five Heterogeneous Image Datasets

**arXiv ID:** 2601.04727 | [PDF](https://arxiv.org/pdf/2601.04727v1)

**作者:** Anika Tabassum `[一作]` (University of Dhaka), Nafisa Naznin `[通讯]` (University of Dhaka)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Object Detection` `Convolutional Neural Network` `Image` `Agriculture Related`

### 📋 论文摘要

**🎯 论文内容**

本文在五个农业与城市监测任务中，训练并评估了自定义轻量化CNN与ResNet-18、VGG-16（从零训练与迁移学习）模型，探讨模型在不同数据集上的泛化与效率。

**💡 创新点**

创新点在于提出一种兼具残差与深度可分离卷积的低参数CNN架构，并在同一实验框架下对比三种模型训练方式，揭示轻量模型在资源受限场景下的竞争力。

**🔧 技术方法**

使用的技术包括PyTorch框架、Adam优化、交叉熵损失、图像归一化与增广、ImageNet预训练、全局平均池化及全连接分类头。

**📊 数据集**

所用数据集包括：FootpathVision（人行道侵占）、RoadDamage（道路破损与井盖检测）、MangoImageBD（芒果品种识别）、PaddyVarietyBD（稻谷品种分类）以及Auto-RickshawImageBD（非法三轮车检测）。

**📈 对比分析**

对比方法是统一训练配置（5个epoch、batch32、学习率1e-3），在相同的预处理与增广下评估训练/验证准确率、精确率、召回率、F1分数与训练时长。结果显示，迁移学习模型往往获得最高准确率（最高达0.90+），自定义CNN在保持低模型大小与训练时间的同时，能在多数任务中逼近迁移学习性能；从零训练的深层网络则普遍表现欠佳或过拟合。

**⚠️ 局限性**

局限性包括：自定义CNN在细粒度任务中易出现欠拟合，迁移学习虽效果好但依赖ImageNet预训练，且VGG-16等大模型训练时间长、模型体积大；所有模型在类别不平衡与小样本类上仍存在误分类，需进一步改进数据增强与模型正则化。

---

## Measurement-Consistent Langevin Corrector: A Remedy for Latent Diffusion Inverse Solvers

**arXiv ID:** 2601.04791 | [PDF](https://arxiv.org/pdf/2601.04791v1)

**作者:** Lee Hyoseok `[一作]` (Korea Advanced Institute of Science and Technology), Tae-Hyun Oh `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2670 | **OpenAlex IDs:** https://openalex.org/A5078114111

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Generation` `Diffusion model` `Stochastic Differential Equation` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于测量一致性Langevin校正器（MCLC）的插值器，用于稳定并提升在潜在扩散模型（LDM）上的零射逆问题求解器。

**💡 创新点**

创新点在于：① 通过量化逆向扩散动力学与真实逆扩散动力学的差距，证明减小该差距可提升稳定性；② 引入正交投影的测量一致性约束，使得Langevin校正在保持测量一致性的同时减少KL散度；③ 该校正器不依赖线性流形假设，可直接插拔到现有LDM逆解算器。

**🔧 技术方法**

技术：潜在扩散模型、逆向扩散动力学分析、Langevin动力学校正、正交投影约束、测量一致性梯度约束、欧拉-马尔科夫近似。

**📊 数据集**

使用Stable Diffusion v1.5（SD v1.5）作为潜在扩散模型，测试数据集为FFHQ和ImageNet的100张验证图像。

**📈 对比分析**

与多种基线（LDPS、PSLD、ReSample、LatentDAPS、DiffStateGrad、MPGD、SILO）对比，MCLC在PSNR、LPIPS、FID及Patch-FID上均有提升，尤其在Blob伪影和整体重建质量上表现优异，且对多种线性与非线性逆问题均通用。

**⚠️ 局限性**

局限：选择校正步长仍需手工调节；对部分已特殊设计的逆解算器（如LatentDAPS）兼容性受限；仍无法完全消除潜在空间中固有的离群值导致的Blob伪影。

---

## CounterVid: Counterfactual Video Generation for Mitigating Action and Temporal Hallucinations in Video-Language Models

**arXiv ID:** 2601.04778 | [PDF](https://arxiv.org/pdf/2601.04778v1)

**作者:** Tobia Poppi `[一作]` (University of Modena and Reggio Emilia), Florian Schiffers `[通讯]` (Amazon Prime Video)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Optimization` `Large Language Model` `Diffusion model` `Video` `Text` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

构建了一个可扩展的因果视频生成框架，并利用生成的语义硬负样本对视频语言模型进行统一的偏好优化，以降低其在动作识别和时序推理方面的幻觉；

**💡 创新点**

创新点在于①将多模态LLM与图像/视频扩散模型相结合，生成只改变动作或时序而保持场景不变的对抗视频；②提出同时利用文本偏好与视觉偏好的统一Direct Preference Optimization（MixDPO）；③创建了约26k对动作与时序的合成偏好对数据集；

**🔧 技术方法**

技术包括多模态LLM（Claude‑4‑Sonnet）、图像编辑模型（Qwen‑Image‑Edit）、图像到视频扩散模型（Wan2.2‑I2V‑14B）、LLM引导的动作提案与过滤、以及基于DPO的统一偏好优化；

**📊 数据集**

使用来自PE Video Dataset（PVD）的真实视频-字幕对作为锚点，生成的合成数据集中包含26,167对偏好样本，评估时还使用了VideoHallucer、VidHalluc、EventHallusion、VideoHallu、VideoHallucer等公开幻觉基准以及VideoMME、NExT‑QA、TempCompass等通用视频理解基准；

**📈 对比分析**

与基线、SFT、仅文本偏好DPO等方法对比，实验表明在动作识别和时序排序任务上平均提升约8–9个百分点，特别是时序排序（order‑list）从约1.6%提升至16.7%（3B）或从16.5%提升至43.8%（7B），在公开幻觉基准上也实现了显著的提升；

**⚠️ 局限性**

限制包括：生成的视频质量受限于当前扩散模型，可能产生视觉伪影；生成的对抗样本主要针对短时动作（<2s），对长时序的覆盖有限；生成过程耗时约3000 GPU‑hours；模型使用参数高效微调，未进行全参数训练，可能进一步提升效果；

---

## When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail

**arXiv ID:** 2601.04748 | [PDF](https://arxiv.org/pdf/2601.04748v1)

**作者:** Xiaoxiao Li `[一作]` (University of British Columbia), Xiaoxiao Li `[通讯]` (University of British Columbia)

**通讯引用:** 5049 | **OpenAlex IDs:** https://openalex.org/A5100458648

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Prompt Engineering` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

通过把多智能体系统编译为单一智能体的技能库，探究单一智能体在技能选择与多智能体协作之间的效率与性能关系；进一步研究技能库规模对LLM技能选择的影响，发现非线性相变和语义混淆导致选择准确率下降，并证明层次化路由可缓解这一问题。

**💡 创新点**

提出将多智能体行为内部化为技能的“编译”方法；发现技能选择存在容量阈值和语义干扰两重限制；证明层次化组织可恢复超阈值规模下的准确率。

**🔧 技术方法**

使用大型语言模型（GPT‑4o‑mini、GPT‑4o）进行实验；构造合成技能库并对齐描述、执行策略和后端；利用结构化提示实现技能选择与执行；通过层次化路由实现分层决策。

**📊 数据集**

用于编译验证的基准包括GSM8K、HumanEval和HotpotQA；用于规模实验的合成任务覆盖数学、代码、写作、分析等八个领域，生成多层次技能集合。

**📈 对比分析**

与原始多智能体系统对比，单一智能体技能实现相似或略高的任务准确率，同时在token消耗下降约54%和延迟下降约50%；在技能库规模从5到200的实验中，技能选择准确率在≈50–100时出现急剧下滑；采用层次化路由后，准确率在大规模下提升至≈70–85%。

**⚠️ 局限性**

实验仅使用合成技能库，未评估真实任务中技能选择错误对最终任务的影响；仅考察了两款LLM，缺乏跨模型通用性验证；层次化路由设计相对简单，实际部署需进一步优化；未分析选择错误对整体系统性能的传播。

---

## Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling

**arXiv ID:** 2601.04744 | [PDF](https://arxiv.org/pdf/2601.04744v1)

**作者:** Xingyuan Li `[一作]` (Shanghai Jiao Tong University), Mengyue Wu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1553 | **OpenAlex IDs:** https://openalex.org/A5109064838

**关键词:** `Sound` `Classification` `Anomaly Detection` `Recurrent Neural Network` `Transformer` `Contrastive Learning` `Audio` `Biomedical Data` `Alzheimer's Disease` `Electronic Health Records`

### 📋 论文摘要

**🎯 论文内容**

提出一种音频仅半监督学习框架，专门用于从长形式医学对话中检测病理语音；

**💡 创新点**

创新点在于三层多粒度建模（帧/片段/会话）与在线伪标签动态更新机制，能够在弱监督条件下精准聚焦关键段落并有效利用无标签数据；

**🔧 技术方法**

技术包括预训练语音编码器（wav2vec2、HuBERT、WavLM）→Transformer聚合→RNN细粒度学习→Siamese一致性损失与EMA教师‑学生、伪标签阈值更新与多尺度特征加权；

**📊 数据集**

数据集：中文EATD‑Corpus（抑郁检测）和英文ADReSSo21（阿尔茨海默检测）；

**📈 对比分析**

与基线会话级模型及公开全监督方法对比，10%标注即可达到90%全监督性能，30%标注几乎等同全监督；在多语言、多编码器设置下均显著优于基线，提升约4–5%宏F1；

**⚠️ 局限性**

局限性：仅使用音频模态，无法借助文本或视觉信息；对跨语言预训练模型依赖较大；伪标签生成仍受会话级弱监督约束，可能存在误标扩散。

---

## Fast Mining and Dynamic Time-to-Event Prediction over Multi-sensor Data Streams

**arXiv ID:** 2601.04741 | [PDF](https://arxiv.org/pdf/2601.04741v1)

**作者:** Kota Nakamura `[一作]` (Toyota Motor Corporation), Yasushi Sakurai `[通讯]` (Osaka University)

**通讯引用:** 3021 | **OpenAlex IDs:** https://openalex.org/A5089668362

**关键词:** `Machine Learning` `Anomaly Detection` `Optimization` `Computational Efficiency` `Stochastic Differential Equation` `Time Series` `Sequential` `Biomedical Data`

### 📋 论文摘要

**🎯 论文内容**

提出一种名为 TimeCast 的动态时序事件预测框架，能够在非平稳多传感器数据流中实时识别阶段并给出事件发生时间的概率预测。

**💡 创新点**

创新点：① 采用阶段化序列建模，利用阶段可识别的时间演进结构实现对数据漂移的自适应；② 通过联合优化阶段描述子（Gaussian 图模型）与预测子（Wiener 过程/逆高斯分布），实现多任务学习并提升预测精度；③ 设计动态规划与增量式更新算法，使学习和在线预测在时间复杂度上均线性或可摊还，适合高频流。

**🔧 技术方法**

技术：多阶段模型（sequential multi‑model），Gaussian 图模型（稀疏精度矩阵）做描述子；Wiener 过程与逆高斯分布做预测子；动态规划求阶段分配；图灵算法（ADMM）求解稀疏精度矩阵；在线增量更新采用 Welford 算法。

**📊 数据集**

使用了五个公开真实数据集，涵盖机械系统（如涡轮发动机）和 ICU 患者监测的多传感器时序数据，记录至事件发生点。

**📈 对比分析**

与 DeepSurv、DeepHit、Cox‑Time（传统生存分析）及 TS2Vec、PatchTST（时序表示学习）和 AC‑TPC（预测聚类）比较。实验显示：在 MAPE、RMSPE 指标上均优于基线；预测速度比基线快数倍至数千倍；学习算法收敛仅 10–20 轮，时间复杂度线性。

**⚠️ 局限性**

局限性：① 预测子仅基于 Wiener 过程/逆高斯分布，可能不适用于更复杂的事件时间分布；② 需要先验设置阶段数和稀疏正则参数，模型对超参数敏感；③ 目前仅在特定工业和医疗场景验证，跨域推广需进一步验证；④ 解释性依赖稀疏图结构，若传感器噪声高或相关性变化剧烈，描述子效果受限。

---

## RiskAtlas: Exposing Domain-Specific Risks in LLMs through Knowledge-Graph-Guided Harmful Prompt Generation

**arXiv ID:** 2601.04740 | [PDF](https://arxiv.org/pdf/2601.04740v1)

**作者:** Huawei Zheng `[一作]` (Zhejiang University), Dazhen Deng `[通讯]` (Zhejiang University)

**通讯引用:** 555 | **OpenAlex IDs:** https://openalex.org/A5049050148

**关键词:** `Computation and Language` `Generation` `Adversarial Attack` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

构建了一个端到端的流程，利用知识图谱引导生成域特定的显式有害提示，并通过双路径混淆重写生成隐蔽且语义保持的有害提示，从而自动化生成高质量的隐式有害提示数据集。

**💡 创新点**

创新点包括（1）将知识图谱嵌入提示生成，系统化捕捉高风险实体；（2）提出双路径混淆重写（直接重写与上下文卡片增强重写），显著提升隐蔽性和语义一致性；（3）实现全流程自动化，突破人工构造的局限。

**🔧 技术方法**

使用知识图谱检索（Wikidata + SPARQL）、少样本提示、LLM生成、Granite-Guardian毒性评估、困惑度筛选、语义一致性与流畅度评估模型、双路径重写与上下文卡片、LLM-as-Judge评价机制。

**📊 数据集**

数据集来源包括：Wikidata（构建领域子图）、JailbreakBench（有害类别示例）、公开基准（AdvBench、Do-Not-Answer、HarmfulQA、CatQA-en、HEx-PHI），以及自研的RiskAtlas隐式提示数据集；模型使用LLaMA‑3.1‑8B/70B 等。

**📈 对比分析**

通过在多种公开模型（如 LLaMA‑3.1‑8B、Gemini‑3 Flash 等）上评估攻击成功率（ASR）与公开基准比较，RiskAtlas隐式版本的ASR高达 61.58%–84.92%，显著优于传统基准（5%–24%）；在对齐与鲁棒性实验中，RiskAtlas 训练的模型在隐式攻击下保持较低 ASR，且通用能力（MMLU）几乎不受影响。

**⚠️ 局限性**

局限性包括：仅使用基于关系的查询，未深入递归检索，可能导致实体覆盖不完整；自动重写算法对创意性攻击的覆盖有限；对知识图谱的依赖使得在稀缺或不完整的领域知识中效果受限；生成的提示仍有被误用的风险。

---

## Feasibility Study Regarding Self-sustainable Reconfigurable Intelligent Surfaces

**arXiv ID:** 2601.04723 | [PDF](https://arxiv.org/pdf/2601.04723v1)

**作者:** Zhenyu Li `[一作]` (KTH Royal Institute of Technology), Cicek Cavdar `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 2717 | **OpenAlex IDs:** https://openalex.org/A5006937058

**关键词:** `Information Theory`

### 📋 论文摘要

**🎯 论文内容**

对自供能可重构智能表面（ssRIS）在两种能量-反射方案（元素分割ES与时间分割TS）下的可行性进行理论分析与仿真，给出在LOS与NLOS条件下满足给定数据率与可靠性时所需的最小元件数。

**💡 创新点**

提出将可行性定义为在不同采集/反射方案下实现QoS所需元件数的增长速率，并系统比较ES与TS在不同功率、数据率与可靠性要求下的可行性特征；同时给出精确的最优相位/预编码设计与闭式元件数表达式。

**🔧 技术方法**

采用多输入单输出（MISO）信道模型、MRT预编码、相位对齐优化、中心极限定理近似、Q‑函数误差分析以及数值求根（二分法）等技术手段，结合LOS与Rayleigh衰落模型进行仿真。

**📊 数据集**

使用仿真参数（15 GHz，50 MHz带宽，N=128，η=0.65，P₀=2 μW，区域尺寸50 m）构建LOS与NLOS信道场景，并在不同发射功率、数据率以及误差概率（ε=1 %）下评估元件需求；未使用真实数据集，而是基于统计信道模型生成的合成数据。

**📈 对比分析**

通过对比TS与ES在不同条件下的最小元件数（用解析公式和数值求解得到），发现TS在功率充足、数据率低的环境下元件数显著低于ES，但当采集条件恶化或数据率提高时，TS的元件数呈指数级增长；相反，ES的元件数随条件恶化呈线性增长，显示出在高可靠性或室外环境中的优势。

**⚠️ 局限性**

局限性包括：仅考虑单用户MISO系统，忽略多用户干扰；假设相位调制无损耗且完全可控；仅分析了LOS与NLOS两种极端情形，未考虑中间衰落；未对硬件实现的非理想因素（如相位误差、功率转换效率波动）进行建模。

---

## SCALER:Synthetic Scalable Adaptive Learning Environment for Reasoning

**arXiv ID:** 2601.04809 | [PDF](https://arxiv.org/pdf/2601.04809v1)

**作者:** Caijun Xu `[一作]` (Fudan University), Yixin Cao `[通讯]` (Shanghai Innovation Institute)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了 SCALER，结合可验证、可调难度、无限生成的推理环境合成管线和自适应多环境强化学习框架，持续提升大语言模型的推理能力。

**💡 创新点**

创新点在于：① 自动生成可调难度、可验证的推理环境，解决传统环境多样性与可控性不足的问题；② 在线难度控制器和环境策划机制，使训练始终保持在模型能力边界附近，既保证学习信号，又维持环境多样性，延长训练收益。

**🔧 技术方法**

主要技术包括：编程问题到推理环境的合成管线（元信息提取、随机测试用例生成与验证、难度校准）；多环境强化学习（GRPO）框架；在线难度自适应控制和环境轮换策略。

**📊 数据集**

使用 CodeContests 作为源数据集，筛选 4,973 个编程问题，合成 2,739 个 SCALER 环境；训练模型基于 Qwen3-1.7B/4B，评测在 AIME24、AMC23、MATH-500、MMLU-Pro、BBEH 五大推理基准上。

**📈 对比分析**

与同预算下的基准数据集 RL（DeepMath、MATH）对比，SCALER 在平均分上提升约 15+ 分；随环境数量增大性能持续提升；消融实验表明难度控制器和环境策划两项机制均为必要，去掉后性能显著下降。

**⚠️ 局限性**

局限性包括：未系统探讨环境内部属性（如上下文丰富度、固有难度等）对学习效果的影响；实验规模仍有限，缺乏对环境规模、模型规模和算力等的放缩律分析；未来需进一步扩展环境多样性与规模。

---

## Neural-Symbolic Integration with Evolvable Policies

**arXiv ID:** 2601.04799 | [PDF](https://arxiv.org/pdf/2601.04799v1)

**作者:** Marios Thoma `[一作]` (CYENS Centre of Excellence), Loizos Michael `[通讯]` (Open University of Cyprus)

**通讯引用:** 782 | **OpenAlex IDs:** https://openalex.org/A5002765890

**关键词:** `Machine Learning` `Optimization` `Explainability and Interpretability` `Convolutional Neural Network` `Reinforcement Learning` `Agentic AI` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出一种演化式神经-符号框架，使NeSy系统能够从空白符号策略和随机网络权重开始，协同学习非可微符号策略和神经网络。

**💡 创新点**

将机器教学（Machine Coaching）语义、Valiant可进化框架与NeSy模块结合，构建可演化的符号规则与神经权重并行更新；通过自举推理与语义损失实现无梯度符号学习。

**🔧 技术方法**

演化算法（基因复制、规则添加、简化等突变）、机器教学符号推理、神经网络（CNN）自举推理、语义损失（SDD+WMC）、并行训练与计算缓存。

**📊 数据集**

随机生成的8个二进制概念的符号策略，用MNIST手写数字图像编码为正负原子，生成约2万实例的训练/验证/测试集，共30个策略×5次随机化＝150组实验。

**📈 对比分析**

与无符号端到端CNN基线对比，演化NeSy最终在验证集上实现中位数99%正确率，虽然需要多代和多网络训练，但相对基线可获得可解释的符号规则；基线在100轮训练后可达98%+准确率。

**⚠️ 局限性**

计算成本显著高于端到端模型；偶尔陷入“同质规则”局部最优需多代才能逃逸；仅支持浅层命题策略、实验数据为合成MNIST，未验证在真实复杂任务上的可扩展性。

---

## Defense Against Indirect Prompt Injection via Tool Result Parsing

**arXiv ID:** 2601.04795 | [PDF](https://arxiv.org/pdf/2601.04795v1)

**作者:** Qiang Yu `[一作]` (Harbin Institute of Technology), Chuanyi Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 689 | **OpenAlex IDs:** https://openalex.org/A5103171964

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Prompt Engineering` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

通过让LLM解析工具输出，仅提取所需数据并过滤恶意注入，提出ParseData和CheckTool两模块实现对间接提示注入的防御。

**💡 创新点**

创新点在于利用LLM自身进行工具结果解析并强制格式与逻辑约束来过滤注入，同时结合检测大型文本的CheckTool，取得最低攻击成功率。

**🔧 技术方法**

采用LLM prompt工程、工具结果解析、格式与逻辑校验以及LLM交互式检查等技术。

**📊 数据集**

在AgentDojo基准上使用gpt-oss-120b、Llama-3.1-70b和Qwen3-32b三大模型进行评估。

**📈 对比分析**

与DeBERTa Detector、Repeat Prompt、Spotlighting、Tool Filter等基线对比，平均UA保持中等，攻击成功率降至<0.5%，风险率降低至0.2–1%，显著优于其它方法。

**⚠️ 局限性**

仅针对工具调用注入，无法防御参数劫持型注入；在多语言或更深层参数劫持场景下效果未知。

---

## NC2C: Automated Convexification of Generic Non-Convex Optimization Problems

**arXiv ID:** 2601.04789 | [PDF](https://arxiv.org/pdf/2601.04789v1)

**作者:** Xinyue Peng `[一作]` (Southeast University), Jiannan Cao `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `Computation and Language` `Optimization` `Large Language Model`

### 📋 论文摘要

**🎯 论文内容**

本文提出了基于大型语言模型的NC2C框架，实现了从非凸优化问题到凸问题的自动转换。

**💡 创新点**

创新点在于利用LLM进行非凸组件检测、策略选择，并引入错误纠正循环与可行域修正机制，实现无人工干预的完整流程。

**🔧 技术方法**

采用LLM的数学推理、符号推理、代码生成、错误纠正循环、可行域修正及常见凸优化求解器（CVXPY、Gurobi、SCIPY）等技术。

**📊 数据集**

使用NL4Opt、NLP4LP、ComplexOR、WireOpt以及自构建的NC2C数据集共四个多样化优化问题集。

**📈 对比分析**

与Vanilla、Chain-of-Experts、OptiMUS、Reflexion等基线方法对比，NC2C在GPT-5.1和Qwen3模型下成功率均超过90%，执行率近97%，明显优于基线。

**⚠️ 局限性**

局限性包括对LLM数学推理能力的依赖、计算开销较大、凸化可能导致子最优且不易量化原问题与凸化问题的差距，以及算法透明度不足。

---

## Dynamic Thermal Feedback in Highly Immersive VR Scenarios: a Multimodal Analysis of User Experience

**arXiv ID:** 2601.04781 | [PDF](https://arxiv.org/pdf/2601.04781v1)

**作者:** Sophie Villenave `[一作]` (Ecole Centrale de Lyon), Guillaume Lavoué `[通讯]` (Ecole Centrale de Lyon)

**关键词:** `Human-Computer Interaction` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

本文在高度沉浸的VR场景中，对比静态与动态环境热反馈对用户体验的影响，并通过多模态数据（问卷、访谈、行为日志）进行综合分析。

**💡 创新点**

创新点在于首次在高沉浸度VR下系统性评估动态热反馈与静态热反馈的差异，并提出基于个体热敏感度定制热刺激的研究框架。

**🔧 技术方法**

所用技术包括：非可穿戴的红外灯与风扇热反馈系统、HTC Vive Pro 2 + Valve Index 控制器、PLUME 采集工具、Unity XR Interaction Toolkit 以及问卷与访谈分析。

**📊 数据集**

数据集为49名受试者（25人完成“沙洲岛”情景，24人完成“雪山”情景）在三种热反馈条件（无、静态、动态）下的行为与主观体验记录。

**📈 对比分析**

通过平衡 Latin 方格的 within‑subject 设计与非参数统计（Friedman、Wilcoxon）对比三种条件，结果显示热反馈显著提升“存在感”，但静态与动态热反馈在问卷上差异不显著；行为指标主要受顺序影响。

**⚠️ 局限性**

局限性包括：仅测试单一热刺激（加热或冷风）且未实现空间化热反馈；缺乏个体化热敏感度校准；未测量认知负荷，可能影响热感知；实验场景较线性，限制了对更复杂行为的观察。

---

## LANGSAE EDITING: Improving Multilingual Information Retrieval via Post-hoc Language Identity Removal

**arXiv ID:** 2601.04768 | [PDF](https://arxiv.org/pdf/2601.04768v1)

**作者:** Dongjun Kim `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**通讯引用:** 2647 | **OpenAlex IDs:** https://openalex.org/A5033580486

**关键词:** `Computation and Language` `Retrieval` `Auto Encoder` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出一种后处理稀疏自编码器 LangSAE，能够在向量空间直接抑制多语言检索中的语言身份偏差，并在不重新编码语料库的情况下提升检索质量。

**💡 创新点**

创新点在于：①使用稀疏过完备自编码器将嵌入映射到稀疏特征空间；②通过跨语言激活统计识别语言相关的隐含单元；③在推理时仅屏蔽这些单元并重构嵌入，形成可直接替换原向量的“编辑版”嵌入。

**🔧 技术方法**

核心技术包括：稀疏过完备自编码器（overcomplete sparse autoencoder）、top‑k 稀疏化、基于激活频率的语言掩码、余弦相似度检索，以及与基线、All‑but‑the‑Top 等全局去偏方法的对比。

**📊 数据集**

使用的评估数据集为平行多语种 QA 任务的 Belebele（10 语言）和 XQuAD（6 语言），并构建了统一的多语言检索池进行实验。

**📈 对比分析**

与基线模型、全局去偏方法（All‑but‑the‑Top）以及仅重构的 SAE 对比；在 nDCG@20、Recall@20 上实现了宏平均约 +20% 的提升，尤其在脚本差异大的语言（如中文）获得显著效果。

**⚠️ 局限性**

局限性包括：仅在平行 QA 基准上验证，未涵盖域漂移或不平衡语言分布的真实语料；需要预先标注的语言标签，阈值设定对结果有影响；抑制语言特征后可能忽略其他潜在的偏差源。

---

## Structural Indexing of Relational Databases for the Evaluation of Free-Connex Acyclic Conjunctive Queries

**arXiv ID:** 2601.04757 | [PDF](https://arxiv.org/pdf/2601.04757v1)

**作者:** Cristian Riveros `[一作]` (Pontificia Universidad Católica de Chile), Nicole Schweikardt `[通讯]` (Humboldt-Universität zu Berlin)

**通讯引用:** 1328 | **OpenAlex IDs:** https://openalex.org/A5019365566

**关键词:** `Databases` `Relational Color Refinement` `Graph`

### 📋 论文摘要

**🎯 论文内容**

构建基于数据库内部结构对称性的辅助索引 D_col，显著降低对自由连通无环（fc‑ACQ）的预处理时间，并在更小的索引上完成枚举和计数。

**💡 创新点**

首次将图同构中的颜色细化（Relational Color Refinement）引入数据库查询索引，利用结构对称性实现预处理时间从 O(|D|) 降至 O(|D|·log|D|)，且索引大小可远小于数据库大小。

**🔧 技术方法**

使用 Relational Color Refinement、节点标记图的颜色索引、fc‑ACQ 的 GHD 分解，以及从任意模式到二元模式再到节点标记图的三步转换。

**📊 数据集**

论文在理论分析中未给出实验数据集，讨论中提及 Web 图、随机图等典型实例；实验验证未包含。

**📈 对比分析**

与传统 B+Tree 或 worst‑case optimal join 索引相比，所提出的索引在结构对称明显的实例中可实现常数或对数级别的辅助数据库大小，枚举/计数延迟为 O(1) 或 O(k)，总体性能优于传统方法；在最坏情况下仍保持线性。

**⚠️ 局限性**

仅适用于自由连通无环 CQs，无法扩展到更宽的查询；对动态更新支持不足；实现复杂且依赖 RCR 计算，且对高复杂度数据库颜色分辨率有限。

---

## Generalised Quantifiers Based on Rabin-Mostowski Index

**arXiv ID:** 2601.04739 | [PDF](https://arxiv.org/pdf/2601.04739v1)

**作者:** Denis Kuperberg `[一作]` (CNRS), Michał Skrzypczak `[通讯]` (Institute of Informatics University of Warsaw)

**关键词:** `Logic in Computer Science`

### 📋 论文摘要

**🎯 论文内容**

本文提出了新的索引量词和游戏量词，并研究了它们在 ω-词和无穷树上的可表达性与可判定性；

**💡 创新点**

创新点在于：①首次将索引问题转化为逻辑量词；②证明 ω-词上索引量词可归约为纯 MSO，树上则不可判定；③给出了参数化的有限记忆转化器实现游戏量词策略；

**🔧 技术方法**

主要技术包括：MSO 与自动机的对应、游戏量词的引入、Ramsey 定理与 Wilke 代数的应用、Büchi‑Landweber 证明的参数化推广、以及对无穷树的间隔长度可界定性构造；

**📊 数据集**

该工作为理论研究，未使用具体数据集；

**📈 对比分析**

由于是理论性质证明，未进行实验比较；提出的算法在理论上有效，可将 MSO+ 转化为纯 MSO；

**⚠️ 局限性**

局限性在于：对树的索引量词仍不可判定，且在某些情况下策略需要前瞻或延迟，无法由无前瞻的有限记忆转化器实现；

---

## Excess Description Length of Learning Generalizable Predictors

**arXiv ID:** 2601.04728 | [PDF](https://arxiv.org/pdf/2601.04728v1)

**作者:** Elizabeth Donoway `[一作]` (Anthropic), Jan Leike `[通讯]` (Anthropic)

**关键词:** `Machine Learning` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出并定义了过剩描述长度（EDL）这一度量，量化微调过程中模型参数吸收的可泛化信息。

**💡 创新点**

创新点在于将预序列编码与信息理论相结合，给出可计算的EDL指标，并用它区分能力激活（latent）与新能力教学（teaching）两种学习模式。

**🔧 技术方法**

使用的技术包括交叉熵作为编码长度、预序列（prequential）编码框架、信息理论证明（非负性、收敛性）以及一系列Toy模型解析。

**📊 数据集**

实验数据集主要来自大语言模型（如 Llama 3、TinyStories、Qwen 2.5）在算术与推理任务上的微调数据；理论部分未限定具体数据集。

**📈 对比分析**

比较方法：通过观察 EDL 随样本量、参数占用的缩放签名与传统准确率/损失对照，发现教学场景 EDL 明显增大、能力激活场景下降，验证 EDL 能有效捕捉两种学习过程的区别。

**⚠️ 局限性**

局限性包括：EDL 依赖具体训练算法；只衡量预测压缩而非语义知识；仅针对监督微调，无法直接推广到 RL 或偏好优化等其他后训练方式；Toy 模型过于简化，真实学习过程更为复杂。

---

## Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning

**arXiv ID:** 2601.04726 | [PDF](https://arxiv.org/pdf/2601.04726v1)

**作者:** Yuyang Hu `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 3838 | **OpenAlex IDs:** https://openalex.org/A5010558184

**关键词:** `Artificial Intelligence` `Large Language Model` `Agentic AI` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

构建了一个事件中心化的记忆框架，将连续经验分割为事件节点并通过显式逻辑关系构成事件图，以此引导记忆检索与推理。

**💡 创新点**

创新点在于将记忆组织为事件图并显式编码因果、时间等逻辑关系，以及通过基于图的主动多路径检索实现逻辑感知的推理流程。

**🔧 技术方法**

使用大型语言模型进行事件分割与关系抽取，构建事件图；采用 Planner‑Explorer‑Responder 三代理协同检索，并使用 BGE‑M3 作为嵌入模型。

**📊 数据集**

在 LoCoMo（对话问答）和 NarrativeQA（长篇故事理解）两个公开长上下文推理基准上进行评估。

**📈 对比分析**

与多种基线（RAG、Mem0、MemoryOS、HippoRAG、A‑Mem、CAM）对比，在单跳、多跳、时间推理等任务上均取得显著提升，平均 F1 达到 52‑53%，比最强基线高约 5‑8%。

**⚠️ 局限性**

局限在于事件分割与关系抽取依赖 LLM，质量受限；实验仅覆盖两套基准，未验证在更广泛任务和设置中的通用性。

---

## Branch-width of connectivity functions is fixed-parameter tractable

**arXiv ID:** 2601.04756 | [PDF](https://arxiv.org/pdf/2601.04756v1)

**作者:** Tuukka Korhonen `[一作]` (University of Copenhagen), Sang-il Oum `[通讯]` (Institute for Basic Science)

**通讯引用:** 2176 | **OpenAlex IDs:** https://openalex.org/A5082994085

**关键词:** `Data Structures and Algorithms` `Optimization` `Computational Efficiency` `Graph`

### 📋 论文摘要

**🎯 论文内容**

提出了一个固定参数可行算法，能够在时间 2^O(k^2)γ n^6 log n 内决定并构造给定连通性函数（包括图的分支宽、马图的分支宽、层次宽、切割宽等）是否不超过 k，解决了 Hliněný 关于马图分支宽的长期开放问题。

**💡 创新点**

创新点在于：① 通过“安全割”与“泰坦割”概念构造递归分解，证明在 |V| > 3^k+1 时可以将实例拆分为更小子实例；② 采用子模函数最小化与矩阵相交算法相结合，实现对马图的优化；③ 通过迭代压缩与 Oum–Seymour 的 FPT 近似算法相结合，消除已知分支树假设，得到最终的 2^O(k^2)γ n^6 log n 复杂度；④ 与之前的 γ n^O(k) 方案相比，显著降低了 k 的指数依赖。

**🔧 技术方法**

核心技术包括：子模函数最小化（用于寻找安全割），泰坦割判定（判定三分覆盖），迭代压缩技术，Oum–Seymour 的分支宽 FPT 近似算法，以及马图特定的交叉匹配（matroid intersection）加速子模最小化。

**📊 数据集**

论文为理论算法研究，不涉及实际数据集，实验均以合成的连通性函数或马图表示为例进行复杂度分析。

**📈 对比分析**

与 Oum–Seymour 原始算法（γ n^O(k)）相比，本文的算法在 k 的指数项从 O(k^3) 降至 O(k^2)，并在 n 的多项式阶中保留了 6 次方与 log n 的乘积，整体时间大幅下降，尤其在 k 较大时效果显著。

**⚠️ 局限性**

局限性：① 仍存在较高的 n^6 乘子，实际运行效率受限；② 对于 |V| ≤ 2^O(k) 的小实例，尚未突破 Oum–Seymour 的 n^O(k) 上界；③ 对路径宽度等其他宽度参数的 FPT 性质仍未解决。

---

## One-clock synthesis problems

**arXiv ID:** 2601.04902 | [PDF](https://arxiv.org/pdf/2601.04902v1)

**作者:** Sławomir Lasota `[一作]` (University of Warsaw), Radosław Piórkowski `[通讯]` (University of Oxford)

**关键词:** `Formal Languages and Automata Theory`

### 📋 论文摘要

**🎯 论文内容**

对时序游戏中使用非确定性时钟自动机的合成问题进行系统研究，并证明所有变体的可合成问题在一时钟自动机下都是不可判定的。

**💡 创新点**

证明了即便将时序游戏的获胜条件限制到一时钟自动机，所有反应合成与Church合成问题仍不可判定，扩展并统一了之前的部分结果，并给出有限记忆可行性的完整判定。

**🔧 技术方法**

通过对失控计数器机、不可判定的再到时序自动机的多步归约，利用时钟区域与局部语言构造，以及一时钟重置自动机和ε-转移来构造游戏胜利条件。

**📊 数据集**

本研究基于理论模型，不涉及具体实验数据集。

**📈 对比分析**

该工作主要为理论判定性分析，无实验对比；通过归约证明其不可判定性，证明与既往结果相比扩展了更广泛的游戏变体。

**⚠️ 局限性**

结果仅在非确定性时钟自动机下给出；游戏是否确定性仍未证明；对受限于一时钟的特殊子类（如可达一时钟自动机）的决定性结果仍不完整。

---

## Rigorous numerical computation of the Stokes multipliers for linear differential equations with single level one

**arXiv ID:** 2601.04901 | [PDF](https://arxiv.org/pdf/2601.04901v1)

**作者:** Michèle Loday-Richaud `[一作]` (Université Paris-Saclay), Pascal Remy `[通讯]` (Laboratoire de Mathématiques de Versailles)

**通讯引用:** 110 | **OpenAlex IDs:** https://openalex.org/A5077146690

**关键词:** `Mathematical Software` `Ordinary Differential Equation`

### 📋 论文摘要

**🎯 论文内容**

本文描述了一种实用算法，用于计算具有多项式系数的线性微分方程在单一水平为1的非规则奇点处的斯托克斯乘子。该算法基于博雷尔求和和数值ODE求解的经典方法，但避免了与直接实现相比的大量冗余工作。

**💡 创新点**

创新点在于该算法适用于任意阶的微分方程，没有一般性假设，并且适合高精度计算。此外，提供了在SageMath计算机代数系统中的开源实现，支持任意精度计算并自动提供严格的误差界限。

**🔧 技术方法**

使用了博雷尔求和和数值ODE求解的技术，结合了符号计算和经典数值方法。

**📊 数据集**

使用了具有多项式系数的线性微分方程的数据集，具体的例子在文中进行了说明。

**📈 对比分析**

与现有方法相比，本文的方法在计算多个斯托克斯乘子时避免了冗余，性能得到了显著提升。算法的复杂度在高精度情况下得到了优化，能够在O(p log^3 + o(1)p)的时间内计算固定方程的斯托克斯乘子，误差不超过2^-p。

**⚠️ 局限性**

限制在于该算法主要针对单一水平为1的奇点，尽管可以处理任意阶的方程，但在多级情况下可能需要更强大的求和理论。

---

## Quantum Secure Biometric Authentication in Decentralised Systems

**arXiv ID:** 2601.04852 | [PDF](https://arxiv.org/pdf/2601.04852v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Cryptography and Security`

---

## Flexible Manufacturing Systems Intralogistics: Dynamic Optimization of AGVs and Tool Sharing Using Coloured-Timed Petri Nets and Actor-Critic RL with Actions Masking

**arXiv ID:** 2601.04887 | [PDF](https://arxiv.org/pdf/2601.04887v1)

**作者:** Sofiene Lassoued `[一作]` (South Westphalia University of Applied Sciences), Andreas Schwunga `[通讯]`

**关键词:** `Artificial Intelligence` `Optimization` `Reinforcement Learning` `Reinforcement Learning` `Tabular` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出一种将彩色时间Petri网与基于模型的强化学习相结合的灵活制造系统调度方法，能够同时调度机器、AGV与工具共享；

**💡 创新点**

创新点在于利用Petri网的动作掩码降低搜索空间，加入lookahead机制提前定位AGV位置，并在大型实例中构建新的Taillard风格基准；

**🔧 技术方法**

技术包括Colored‑Timed Petri Nets、Maskable Proximal Policy Optimization（MPPO）强化学习、奖励塑形与动态动作掩码；

**📊 数据集**

使用公开的JSSP基准（如Taillard）生成的自定义大规模实例（最多100工件×20机台）以及内部设计的AGV/工具调度矩阵；

**📈 对比分析**

与传统启发式、SOS等元启发式相比，PetriRL在大型实例上实现18%‑40%更短的完成时间，且计算时间显著低于10分钟的搜索阈值；

**⚠️ 局限性**

局限包括对极大规模实例的收敛速度受限、对AGV和工具运输时间的准确性假设，以及模型在动态机台停机等突发事件时的鲁棒性待进一步验证。

---

## AECV-Bench: Benchmarking Multimodal Models on Architectural and Engineering Drawings Understanding

**arXiv ID:** 2601.04819 | [PDF](https://arxiv.org/pdf/2601.04819v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## Orchestrating Intelligence: Confidence-Aware Routing for Efficient Multi-Agent Collaboration across Multi-Scale Models

**arXiv ID:** 2601.04861 | [PDF](https://arxiv.org/pdf/2601.04861v1)

**作者:** Jingbo Wang `[一作]` (Harbin Institute of Technology), Ting Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 38260 | **OpenAlex IDs:** https://openalex.org/A5100418162

**关键词:** `Artificial Intelligence` `Computational Efficiency` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Text` `Physics Related` `Biomedical Data`

### 📋 论文摘要

**🎯 论文内容**

本文提出了OI-MAS多代理系统，动态分配代理角色与LLM规模以提升推理效率。

**💡 创新点**

创新点在于引入“指挥家”式的状态相关角色-模型路由与置信度感知机制，实现对不同推理阶段的规模自适应调度。

**🔧 技术方法**

使用的技术包括基于预训练文本编码器的角色路由网络、模型路由网络、强化学习优化置信度加权成本目标，以及多规模LLM池。

**📊 数据集**

实验采用GSM8K、MATH、MedQA、GPQA、MBPP等数学、医学、物理推理与代码生成基准。

**📈 对比分析**

与多种基线（LLM-Debate、GPTSwarm、AFlow、MaAS、MasRouter）比较，OI-MAS在准确率上提升约7.7%并将推理成本降低最多79.8%，且平均延迟显著下降。

**⚠️ 局限性**

局限包括未探究长期记忆管理、在大规模并发部署下的性能一致性，以及缺乏针对多代理安全性的完整框架。

---

## DivAS: Interactive 3D Segmentation of NeRFs via Depth-Weighted Voxel Aggregation

**arXiv ID:** 2601.04860 | [PDF](https://arxiv.org/pdf/2601.04860v1)

**作者:** Ayush Pande `[一作]` (Indian Institute of Technology Kanpur), Ayush Pande `[通讯]` (Indian Institute of Technology Kanpur)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Neural Radiance Field` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出了一个无需优化、零样本交互式3D分割框架DivAS，利用NeRF几何先验与SAM 2D掩码实现快速交互式分割。

**💡 创新点**

①优化自由的交互式Voxel聚合；②基于NeRF深度的SAM掩码细化；③200 ms内完成CUDA voxel融合；④使用Fibonacci球采样与centroid zoom降低用户标注量。

**🔧 技术方法**

NeRF（Instant‑NGP / Mip‑NeRF 360）、Segment Anything Model (SAM)、自定义CUDA voxel融合核、Fibonacci球采样、稀疏视点交互、深度加权掩码。

**📊 数据集**

LLFF、Mip‑NeRF 360、NVOS（用于评估）以及伪 Ground‑truth Mask。

**📈 对比分析**

与SA3D、SANeRF‑HQ等优化基线对比，DivAS在mIoU与像素准确率上与最优方法相当，端到端速度比优化方法快2–2.5×，交互时延低。

**⚠️ 局限性**

依赖预训练NeRF质量，几何欠佳时易产生误分；依赖SAM，受其误差影响；对远景或稀疏样本对象覆盖不足时可能漏分。

---

## DR-LoRA: Dynamic Rank LoRA for Mixture-of-Experts Adaptation

**arXiv ID:** 2601.04823 | [PDF](https://arxiv.org/pdf/2601.04823v1)

**作者:** Guanzhi Deng `[一作]` (City University of Hong Kong), Lijie Wen `[通讯]` (Tsinghua University)

**通讯引用:** 4406 | **OpenAlex IDs:** https://openalex.org/A5030845033

**关键词:** `Artificial Intelligence` `Large Language Model` `Mixture of Experts` `Supervised Fine-Tuning` `Text` `Biomedical Data`

### 📋 论文摘要

**🎯 论文内容**

论文提出了一种动态增长 LoRA 权重的框架，专门针对 Mixture‑of‑Experts（MoE）大语言模型的参数高效微调，使得每个专家的 LoRA 秩随任务需求动态调整。

**💡 创新点**

创新点在于引入专家显著性评分（结合路由频率和 LoRA 重要性）来指导不同专家的秩扩展，实现了任务相关专家的高效参数分配，解决了统一秩分配导致的资源错配问题。

**🔧 技术方法**

技术上使用 LoRA 低秩适配、指数滑动平均跟踪专家路由频率、梯度权重积累评估秩重要性、增量秩扩展策略以及可训练路由器解冻等手段。

**📊 数据集**

实验数据集包括 OLMoE SFT Mix（指令、数学推理、代码等）以及医学 QA 数据集（MedQA、MedMCQA、PubMedQA）等。

**📈 对比分析**

与基准方法（LoRA、DoRA、AdaLoRA 等）在相同参数预算下比较，结果显示 DR‑LoRA 在任务对齐和通用基准上平均提升 1.8–2.2 分，且在医学领域更显著，优于统一秩分配和基于剪枝的动态方法。

**⚠️ 局限性**

局限性包括：验证主要针对基于 top‑k 路由的 MoE；对不同路由策略（如 Expert Choice）或多模态 MoE 的适用性尚未验证；未在 100B+ 参数级别的大型 MoE 上测试其可扩展性。

---

## 5G NR Non-Terrestrial Networks: From Early Results to the Road Ahead

**arXiv ID:** 2601.04882 | [PDF](https://arxiv.org/pdf/2601.04882v1)

**作者:** Mattia Figaro `[一作]`, Michele Zorzi `[通讯]`

**关键词:** `Networking and Internet Architecture`

### 📋 论文摘要

**🎯 论文内容**

无法获取论文具体内容

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用的技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法确定比较方法及性能

**⚠️ 局限性**

缺少关键信息导致无法评估限制

---

## CuMA: Aligning LLMs with Sparse Cultural Values via Demographic-Aware Mixture of Adapters

**arXiv ID:** 2601.04885 | [PDF](https://arxiv.org/pdf/2601.04885v1)

**作者:** Ao Sun `[一作]` (Southeast University), Yuheng Jia `[通讯]` (Southeast University)

**通讯引用:** 1639 | **OpenAlex IDs:** https://openalex.org/A5013880628

**关键词:** `Computation and Language` `Generation` `Recommendation System` `Transformer` `Large Language Model` `Mixture of Experts` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了 CuMA（Cultural Mixture of Adapters）框架，利用人口统计路由实现文化对齐，解决密集模型的均值崩溃问题。

**💡 创新点**

创新点在于将对齐视为条件容量分离任务，使用人口统计感知路由学习潜在文化拓扑，显式分离冲突梯度。

**🔧 技术方法**

技术包括基于 LoRA 的稀疏 Mixture‑of‑Experts、冻结的语义嵌入编码人口统计、条件路由与负载平衡正则化。

**📊 数据集**

使用的基准数据集为 WorldValuesBench、Community Alignment 及 PRISM，涵盖多文化价值预测与生成任务。

**📈 对比分析**

与传统稠密微调（FFT、LoRA、DoRA）及语义路由 MoE（MixLoRA、HydraLoRA）对比，CuMA 在 Llama‑3.1‑8B 与 Qwen3‑8B 上平均提升约 5‑10% 准确率、显著降低 EMD 并在生成任务中获得 70‑80% Win‑Rate。

**⚠️ 局限性**

局限性包括依赖显式人口统计信息、固定专家数量可能不足以覆盖全球文化多样性、对训练数据的偏见依赖以及多专家结构导致的显存占用。

---

## Zero Wrench Control via Wrench Disturbance Observer for Learning-free Peg-in-hole Assembly

**arXiv ID:** 2601.04881 | [PDF](https://arxiv.org/pdf/2601.04881v1)

**作者:** Kiyoung Choi `[一作]` (Daegu Gyeongbuk Institute of Science and Technology), Sehoon Oh `[通讯]` (Daegu Gyeongbuk Institute of Science and Technology)

**通讯引用:** 4127 | **OpenAlex IDs:** https://openalex.org/A5007243736

**关键词:** `Robotics` `Robotic Intelligence` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

提出了一种动态张量干扰观测器（DW‑DOB）实现接触丰富操作中的零张量控制，并在工业公差（H7/h6）下的插槽插入实验中验证其有效性。

**💡 创新点**

创新点在于将任务空间惯量显式融入观测器模型，精确分离惯性响应与真实外部张量，实现高灵敏度且鲁棒的零张量控制；并给出了基于能量的被动性稳定性分析。

**🔧 技术方法**

使用了基于任务空间动力学的扰动观测器、低通/组合滤波器、PID 位置/力环控制、Pinocchio 运动学库以及 Linux‑Xenomai 实时控制框架。

**📊 数据集**

实验数据集为7 关节机械臂在 AIDIN AFT200‑D80‑C 力矩传感器下执行 H7/h6 20 mm 规格的插槽插入任务，共计 15 次不同初始姿态的实验记录。

**📈 对比分析**

与传统的接触张量扰动观测器（CWDOB）和无观测器的 PD 基线控制器进行对比，DW‑DOB 在保持零张量、降低峰值接触力、实现更深、更稳定的插入以及 100% 成功率方面表现明显优于其它方法。

**⚠️ 局限性**

局限性包括仅在单臂插槽实验中验证，尚未在多臂协作或人机协作等更复杂场景中测试；对精确惯量模型依赖较高，模型误差可能影响性能；并未结合学习或自适应机制进一步提升对未知环境的适应性。

---

## MisSpans: Fine-Grained False Span Identification in Cross-Domain Fake News

**arXiv ID:** 2601.04857 | [PDF](https://arxiv.org/pdf/2601.04857v1)

**作者:** Zhiwei Liu `[一作]` (University of Manchester), Sophia Ananiadou `[通讯]` (University of Manchester)

**通讯引用:** 17835 | **OpenAlex IDs:** https://openalex.org/A5077976343

**关键词:** `Computation and Language` `Classification` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出了 MisSpans 基准，包含对假新闻句子中误导性 span 的定位、类型分类和解释三项任务。

**💡 创新点**

创新点在于：①实现 span 级别的细粒度误导检测；②提供多类误导标签（不一致、未提及等）；③为每个 span 给出解释，提升可解释性；④跨领域、人类高质量标注。

**🔧 技术方法**

采用大型语言模型（LLM）评估，使用零-shot、one-shot 提示，比较推理增强版与常规版模型的表现。

**📊 数据集**

数据集为 MisSpans，来源于 FakeNewsAMT 的真实-虚假新闻对，涵盖体育、商业、娱乐、政治、技术、教育等六大领域。

**📈 对比分析**

对 15 个主流 LLM 进行评测：定位和分类任务 F1 均低于 0.5，推理模型整体优于无推理模型；one-shot 对大模型略有提升，对小模型往往降级；解释任务性能最高，跨域稳定性好。

**⚠️ 局限性**

局限性：仅评估参数 ≤70B 的开源模型；未对 LLM 进行微调；数据仅为英文，缺乏多语言覆盖。

---

## Rethinking GNNs and Missing Features: Challenges, Evaluation and a Robust Solution

**arXiv ID:** 2601.04855 | [PDF](https://arxiv.org/pdf/2601.04855v1)

**作者:** Francesco Ferrini `[一作]` (University of Trento), Manfred Jaeger `[通讯]` (Aalborg University)

**通讯引用:** 1506 | **OpenAlex IDs:** https://openalex.org/A5014289413

**关键词:** `Machine Learning` `Graph Neural Network` `Graph`

### 📋 论文摘要

**🎯 论文内容**

本文研究在节点特征缺失情况下的图神经网络（GNN）鲁棒性，提出新的评估框架、数据集与一种基于缺失指示符的简单有效基线模型。

**💡 创新点**

创新点包括：① 引入四个稠密且语义丰富的数据集（1个 synthetic + 3 真实），克服传统稀疏特征基准的局限；② 设计更真实的缺失机制（LD‑MCAR、FD‑MNAR、CD‑MNAR）以及训练‑测试缺失分布偏移；③ 通过信息论分析证明高稀疏度下缺失信息损失低；④ 提出不依赖 MAR 假设的 Missing Indicator Method（MIM）基线，表现稳健。

**🔧 技术方法**

技术手段主要包括：GNN 架构（GCN、GAT 等）+ MIM；缺失指示符与零填充相结合；不同缺失机制的实现（均匀、标签依赖、特征依赖、类别依赖）；信息熵与互信息理论分析；实验中计算 F1 与 AUC 作为评估指标。

**📊 数据集**

使用数据集：synthetic Gaussian on Barabási‑Albert graph；三类真实数据：传感器网络、电子传感器网络、TADPOLE 医疗图；对照传统基准 Cora、Citeseer、Pubmed。

**📈 对比分析**

与多种现有缺失处理方法（mean‑impute, zero‑impute, median‑impute, GNN‑based imputation, etc.）进行对比。结果显示：在稀疏基准上大多数方法鲁棒；在新数据集与更复杂缺失机制下，绝大多数方法性能显著下降；MIM 在所有机制与数据集上保持最高或相近 AUC/F1，证明其在多样化缺失场景下的稳健性。

**⚠️ 局限性**

局限性：缺失机制仍未涵盖更复杂的图结构相关 MNAR；实验主要集中在节点特征缺失，未考虑结构缺失；基线虽简单但在极端缺失率或非平稳分布下可能仍有性能下降；需进一步在更大规模、不同任务（如图分类、链路预测）上验证。

---

## A Mathematical Theory of Payment Channel Networks

**arXiv ID:** 2601.04835 | [PDF](https://arxiv.org/pdf/2601.04835v1)

**作者:** Rene Pickhardt `[一作]`, Rene Pickhardt `[通讯]`

**关键词:** `Networking and Internet Architecture`

### 📋 论文摘要

**🎯 论文内容**

提出了一种以几何理论为基础的支付通道网络模型，重点关注可行财富分配的多面体W_G，并通过严格的循环自然投影到W_G上，分析了支付的可行性与网络拓扑的关系。

**💡 创新点**

创新点在于通过几何视角解释了多方通道的资本效率和支付带宽的可持续性，提出了三种缓解通道耗竭的方法，并强调了多方通道在提高支付可靠性方面的优势。

**🔧 技术方法**

使用了几何理论、整数线性规划和流网络理论来分析支付通道网络的流动性状态和财富分配的可行性。

**📊 数据集**

未具体提及使用的数据集，但通过理论推导和模型构建进行了分析。

**📈 对比分析**

通过与现有的两方通道网络进行比较，展示了多方通道在可行财富分配和支付带宽方面的优势，表明多方通道能够有效降低预期的不可行支付率。

**⚠️ 局限性**

限制在于当前模型未考虑基础费用和其他费用的影响，且在实际应用中，通道的流动性分布可能并不均匀，导致支付的可靠性受到影响。

---

## When AI Settles Down: Late-Stage Stability as a Signature of AI-Generated Text Detection

**arXiv ID:** 2601.04833 | [PDF](https://arxiv.org/pdf/2601.04833v1)

**作者:** Ke Sun `[一作]` (Westlake University), Yue Zhang `[通讯]` (Westlake University)

**通讯引用:** 17277 | **OpenAlex IDs:** https://openalex.org/A5100333758

**关键词:** `Computation and Language` `Generation` `Anomaly Detection` `Transformer` `Large Language Model` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文利用LLM自回归生成过程中后期概率波动衰减的现象，提出基于后半文本的两种时序特征；

**💡 创新点**

首次系统揭示“后期波动衰减”并将其转化为导数分散与局部波动度，用以区分人工文本与机器文本；

**🔧 技术方法**

采用后半文本的负对数概率、绝对导数、滑动窗口标准差等统计技术，并在无监督模式下实现检测；

**📊 数据集**

在覆盖多模型与多长度的EvoBench和MAGE基准数据集上进行实验；

**📈 对比分析**

与全局统计、扰动采样等传统方法对比，单独特征在EvoBench达到约83% AUROC，在MAGE约71%，与Fast-DetectGPT融合后进一步提升至约85%；

**⚠️ 局限性**

方法受限于需要足够长的文本才能显现后期波动衰减，对极短文本效果下降，并且融合策略仅为简单加权。

---

## Faithful Summarisation under Disagreement via Belief-Level Aggregation

**arXiv ID:** 2601.04889 | [PDF](https://arxiv.org/pdf/2601.04889v1)

**作者:** Favour Yahdii Aghaebe `[一作]` (University of Sheffield), Nafise Sadat Moosavi `[通讯]` (University of Sheffield)

**通讯引用:** 1269 | **OpenAlex IDs:** https://openalex.org/A5054918343

**关键词:** `Computation and Language` `Generation` `Data Synthesis` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出一种将聚合与生成分离的多文档意见摘要管道 AccSynth，通过先使用基于距离的信念合并对结构化的观点集合进行显式聚合，再用大型语言模型仅进行自然语言实现，来保留原始争议信息并生成流畅摘要。

**💡 创新点**

创新点在于将聚合视为预处理阶段，采用逻辑学中信念合并的距离度量显式建模冲突，显著降低对模型规模和提示设计的敏感性，实现对少数观点的公平表达；同时提供轻量级的两步生成流程，避免在生成过程中无意平滑争议。

**🔧 技术方法**

技术包括：1) 结构化观点抽取（aspect‑polarity 级别的 Qwen 生成抽取）；2) 距离‑基础信念合并（L1 距离、最小化总冲突）；3) LLM 语义实现（单一观点到句子再到段落的两步提示）；4) 评估框架 GEval 以及 BERTScore / BARTScore。

**📊 数据集**

使用 Rotten‑Tomatoes 电影评论数据集（约 17,000 条评论，8,000 条元评论），对 40% 以上包含明显意见冲突的电影进行聚合与摘要。

**📈 对比分析**

与两类基线对比：(i) 仅在生成时聚合的 QA‑Prompting；(ii) Fusion‑of‑N（多候选融合）。实验显示：在所有模型（Qwen、LLaMA、GPT‑5）上，信念合并 + 简单提示在冲突感知指标（Coverage、Polarity、Contrast、Prevalence、Groundedness）上稳定领先；即使是小规模模型也能逼近大型模型的表现；Fusion‑of‑N 仅在强模型下略有提升，对弱模型效果不佳。

**⚠️ 局限性**

局限性包括：1) 仅评估单一领域（电影评论），泛化性待验证；2) 依赖手工定义的 aspect 集合，缺失或隐含观点可能导致聚合不完整；3) 仅使用 L1 距离，其他合并算子和更复杂的多视角表示仍待探索。

---

## Precomputing Multi-Agent Path Replanning using Temporal Flexibility: A Case Study on the Dutch Railway Network

**arXiv ID:** 2601.04884 | [PDF](https://arxiv.org/pdf/2601.04884v1)

**作者:** Issa Hanou `[一作]` (Delft University of Technology), Mathijs de Weerdt `[通讯]` (Delft University of Technology)

**通讯引用:** 2116 | **OpenAlex IDs:** https://openalex.org/A5070981277

**关键词:** `Artificial Intelligence` `Optimization` `Graph`

### 📋 论文摘要

**🎯 论文内容**

本研究提出了 FlexSIPP 算法，用于在多智能体环境中快速重新规划单一被延迟智能体的路径，并通过利用其他智能体的时间灵活性（temporal flexibility）来避免连锁延迟，自动计算并输出“切点”（tipping points），使得被延迟智能体可以在先前认为不安全的时间起动。

**💡 创新点**

创新点在于：① 将安全间隔路径规划（SIPP）与任意起始时间规划（Any-Start-Time Planning）结合，预先构造包含灵活性约束的 ATF；② 通过对其他智能体的时间灵活性进行多项式时间计算，形成可直接使用的“可延迟区间”，从而实现对单一智能体的非侵入式重新规划；③ 通过切点概念实现对冲突顺序的动态切换，显著减少整体延迟并保持可行性。

**🔧 技术方法**

主要技术包括：安全间隔路径规划（SIPP）、任意起始时间规划（@SIPP）、灵活性计算与切点识别、基于 ATF 的 RePEAT 搜索，以及对铁路网络的块（block）与阻塞时间（blocking time）建模。

**📊 数据集**

实验数据集基于荷兰铁路网络（ProRail 基础设施数据）和官方时刻表（Nederlandse Spoorwegen），构建约 9700 个节点、247,600 条边的路由图，分别生成 4 个场景（64、82、77、82 列车，时间跨度 432-474 分钟）。

**📈 对比分析**

与 rSIPP 和 @MAEDeR 进行比较：FlexSIPP 在可行路径数量上显著更多，能够提前安排欧星列车并显著降低总延迟；但搜索时间略长，因为需要尝试更多路径并频繁重启搜索。总体性能上，FlexSIPP 在保持低延迟的同时实现了多方案可选。

**⚠️ 局限性**

局限性包括：① 搜索时间相对较长，尤其在大规模、密集网络中；② 仅处理单一被延迟智能体的情况，若多智能体同时延迟需进一步扩展；③ 对时间灵活性计算的准确性高度依赖原始时刻表与阻塞时间模型；④ 现有实现以铁路为背景，若推广到其他连续时间多智能体场景需重新验证。

---

## Higher-Order Knowledge Representations for Agentic Scientific Reasoning

**arXiv ID:** 2601.04878 | [PDF](https://arxiv.org/pdf/2601.04878v1)

**作者:** Isabella A. Stewart `[一作]` (Massachusetts Institute of Technology), Markus J. Buehler `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 52018 | **OpenAlex IDs:** https://openalex.org/A5011504360

**关键词:** `Artificial Intelligence` `Graph Neural Network` `Large Language Model` `Agentic AI` `Graph`

### 📋 论文摘要

**🎯 论文内容**

构建并分析了基于超图的知识表示，用以捕捉生物复合支架文献中的多实体关联，并将该超图嵌入多智能体推理系统中，演示了如何利用超图路径生成新的材料合成假设。

**💡 创新点**

① 引入超图而非传统图，完整保留多实体共现关系；② 开发超图子图、s‑连通分量、节点中心性等高级分析工具；③ 将超图结构直接驱动多步推理，形成“教师无监督”的科学发现工作流。

**🔧 技术方法**

超图构建与清洗（基于嵌入相似度合并）；图遍历（基于节点交集的BFS、K‑shortest 超图路径）；多智能体框架（GraphAgent/Engineer/Hypothesizer）利用 AutoGen、llama.cpp、sentence‑transformers 等；可视化与统计分析（t‑SNE、幂律拟合、富俱乐部等）。

**📊 数据集**

约 1,100 篇聚焦“biocomposite scaffold”的科研论文（共 161,172 个节点，320,201 条超边），来源于 Web of Science 文献库，使用自然语言处理提取 n‑ary 关系。

**📈 对比分析**

与传统的二元知识图相比，超图在结构完整性、共现覆盖率和推理路径质量上均更优；实验演示中，超图路径成功桥接稀疏概念与核心概念，生成更具可操作性的实验假设。缺乏定量基准指标，但展示了推理准确性提升和多实体关系保留的实证效果。

**⚠️ 局限性**

① 超图规模增大导致遍历与存储成本上升；② 超图构建依赖于 LLM 关系抽取，可能引入抽取误差；③ 交集阈值选择对路径多样性影响大；④ 超图仅保留共现信息，缺乏时序、因果强度等更细粒度语义；⑤ 结果仍需人工评估验证，尚未实现完全自动化。

---

## EvolSQL: Structure-Aware Evolution for Scalable Text-to-SQL Data Synthesis

**arXiv ID:** 2601.04875 | [PDF](https://arxiv.org/pdf/2601.04875v1)

**作者:** Xuanguang Pan `[一作]` (Beihang University), Shuai Ma `[通讯]`

**关键词:** `Computation and Language` `Data Synthesis` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Chain-of-Thought` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

研发了一套结构感知的演化数据合成框架EvolSQL，用种子数据通过探索性扩展和原子变换自动生成多样、高质量的Text-to-SQL训练对。

**💡 创新点**

提出了六种基于SQL AST的原子变换算子和自适应方向演化策略，结合执行驱动的SQL细化与架构感知去重，实现从简单种子到复杂多层SQL的渐进式演化。

**🔧 技术方法**

使用LLM（如Qwen2.5-Coder-32B-Instruct）做生成与细化，Chain-of-Thought 生成与验证，AST算子、适应性评分与稀缺权重的组合，去重采用语义相似度阈值，最终在Qwen2.5-Coder-7B和Meta-Llama-3.1-8B进行全参数监督微调。

**📊 数据集**

以BIRD作为种子和目标数据，使用Spider、Spider-DK、Spider-Syn、Spider-Realistic、EHRSQL和Science Benchmark等作为外部评估基准。

**📈 对比分析**

与SynSQL、OmniSQL、SENSE、SQLFLOW等大规模或专用模型进行比较，单一7B模型在BIRD dev上执行准确率65.1%（超越OmniSQL-7B的63.9%），在Spider测试集上86.1%（高于SENSE-7B 51.8%），并在跨域基准上平均提升约6.1%。

**⚠️ 局限性**

合成SQL仍可能含有逻辑噪声，演化教师模型受限于中等规模LLM，且实验仅采用监督微调，未结合强化学习等更强训练方法。

---

## Wireless Communication with Cross-Linked Rotatable Antenna Array: Architecture Design and Rotation Optimization

**arXiv ID:** 2601.04862 | [PDF](https://arxiv.org/pdf/2601.04862v1)

**作者:** Ailing Zheng `[一作]` (Shanghai Jiao Tong University), Guoying Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 13182 | **OpenAlex IDs:** https://openalex.org/A5100633436

**关键词:** `Information Theory` `Optimization`

### 📋 论文摘要

**🎯 论文内容**

研究了一种基于交叉耦合（Cross‑Linked）可旋转天线（CL‑RA）阵列的上行多用户通信系统，提出了天线元素级与天线面板级的旋转模型，并联合优化接收波束和旋转角度，设计了可交替优化（AO）算法以及基于遗传算法的离散角度搜索；

**💡 创新点**

通过引入交叉耦合结构将行、列共享旋转轨道，显著降低了驱动电机数量和控制复杂度；在单用户场景下证明CL‑RA在元素级旋转时可实现与全自由旋转相同的性能；在多用户场景下通过AO算法将接收MMSE与旋转方向的可行方向法耦合；对离散角度问题提出遗传算法解决方案；

**🔧 技术方法**

系统建模（LoS+NLoS）、方向性天线增益模型、MMSE波束形成、可行方向法（FD）、遗传算法（GA）、线性规划求解、梯度计算与Armijo步长；

**📊 数据集**

采用仿真数据：6个用户、64个天线、8×8天线阵列、3.5 GHz频段、λ≈0.0857 m、用户分布在50–70 m范围内、随机散射簇D=8、噪声功率-80 dBm、各用户最大发射功率10 dBm；

**📈 对比分析**

与全自由旋转、面板自由旋转、随机旋转、全阵列旋转、固定方向、各向同性天线等基线进行对比。结果表明：CL‑RA在元素级旋转时几乎与全自由旋转等效；相较于面板级旋转提升≈25%；相较于固定方向提升≈128%；在不同功率、天线直向因子、极限仰角以及离散角度取值下均保持优越性能；

**⚠️ 局限性**

受限于机械转动范围和行列耦合导致的自由度受限，尤其在多面板配置下旋转范围进一步缩小，导致性能下降；离散角度搜索虽然可实现硬件实现，但仍受离散分辨率限制；算法复杂度较高，需多轮交替优化与梯度/遗传搜索。

---

## RAAR: Retrieval Augmented Agentic Reasoning for Cross-Domain Misinformation Detection

**arXiv ID:** 2601.04853 | [PDF](https://arxiv.org/pdf/2601.04853v1)

**作者:** Zhiwei Liu `[一作]` (University of Manchester), Sophia Ananiadou `[通讯]` (University of Manchester)

**通讯引用:** 17835 | **OpenAlex IDs:** https://openalex.org/A5077976343

**关键词:** `Computation and Language` `Retrieval` `Anomaly Detection` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Retrieval-Augmented Generation` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了基于检索增强的代理式推理框架RAAR，用于跨域错误信息检测。

**💡 创新点**

创新点在于：①多维度检索（情感、语义、写作风格）构建跨域参考；②多代理协同推理，生成可验证的多步推理路径；③结合监督微调与强化学习，统一训练多任务推理与验证能力。

**🔧 技术方法**

使用大语言模型（Qwen3 8B/14B、DeepSeek 等）作为基础，并采用检索增强（RAG）、多代理推理、SFT、RL（GRPO）等技术。

**📊 数据集**

实验数据集为 AMTCele、PHEME 与 COCO 三个跨域错误信息数据集。

**📈 对比分析**

与多种基线（传统跨域方法、LLM 原生推理、LLM 适配方法）对比，RAAR-14B 在三大数据集上平均 F1 分数最高，且显著优于 GPT‑4.1、DeepSeek‑v3.2‑Reasoner 等先进 LLMs，验证了框架的有效性。

**⚠️ 局限性**

局限性：仅训练 8B/14B 开源 LLM，未验证更大模型；只考虑情感、语义、写作风格三维度，缺乏观点、论证结构等信息；检索过程可能引入噪声，尤其在多主题数据集如 COCO 上表现不佳。

---

## Stability of Constrained Optimization Models for Structured Signal Recovery

**arXiv ID:** 2601.04849 | [PDF](https://arxiv.org/pdf/2601.04849v1)

**作者:** Yijun Zhong `[一作]` (Zhejiang Sci-Tech University), Yi Shen `[通讯]` (Zhejiang Sci-Tech University)

**通讯引用:** 8404 | **OpenAlex IDs:** https://openalex.org/A5009129294

**关键词:** `Information Theory` `Optimization`

### 📋 论文摘要

**🎯 论文内容**

本文通过非渐进理论分析，证明了三种受约束优化模型（受约束最小二乘、受约束最小绝对偏差、受约束非线性最小二乘/相位检索）在噪声和参数调优误差下的稳定性，并给出了误差上界与样本复杂度之间的权衡关系。

**💡 创新点**

创新点包括：① 统一地考虑 f(x*)≠η 的情况，给出匹配误差与样本复杂度的非渐进误差上界；② 在相位检索和稀疏信号恢复的框架下，首次把相位检索的受约束模型与线性模型的稳定性理论对齐；③ 引入高斯宽度和有效维数概念，精细量化可行集的复杂度；④ 在存在绝对齐次 f 的前提下，提供多种情形（f(x*)≥η、f(x*)<η）的精确误差表达式。

**🔧 技术方法**

主要技术手段包括：高斯随机矩阵分析、Gaussian width 与有效维数理论、凸/非凸可行集的锥形降解、概率界（如体积不等式、Chernoff 等），以及对噪声项的随机不等式控制。

**📊 数据集**

本文未使用实际数据集，而是基于随机高斯测量矩阵和理论模型进行分析；所有结果均为理论证明，未涉及实验验证。

**📈 对比分析**

与已有文献（如 Lasso、稀疏相位检索等）的比较显示，本文的误差上界与经典压缩感知理论一致（当 η=f(x*) 时误差退化为传统结果），并在参数调优误差不为零时给出更宽容的误差界；实验部分缺失，主要是理论对比。

**⚠️ 局限性**

局限性包括：① 需要高斯测量矩阵且对 f 需满足绝对齐次性；② 结果对 m≥C m0 的样本下限有依赖，实际场景中可能更难满足；③ 仅给出理论误差上界，缺乏算法实现与实验验证；④ 对非高斯噪声或测量矩阵的鲁棒性讨论有限。

---

## A Longitudinal Analysis of Gamification in Untappd: Ethical Reflections on a Social Drinking Application

**arXiv ID:** 2601.04841 | [PDF](https://arxiv.org/pdf/2601.04841v1)

**作者:** Jefferson Seide Molléri `[一作]` (Kristiania University), Jouni Smed `[通讯]` (University of Turku)

**通讯引用:** 1274 | **OpenAlex IDs:** https://openalex.org/A5091346568

**关键词:** `Software Engineering`

### 📋 论文摘要

**🎯 论文内容**

对Untappd应用的游戏化功能进行为期五年的纵向伦理分析，比较2020年与2025年在徽章设计、社区讨论及商业模式上的变化。

**💡 创新点**

首次将EASE伦理框架与Caragay概念框架并用在社交饮酒平台的游戏化评估中，揭示了持续的伦理灰区与设计缺陷。

**🔧 技术方法**

采用网络民族志方法、案例研究、文献与文档分析，并以伦理框架（EASE、Caragay）对徽章类别进行定性评估。

**📊 数据集**

收集的资料包括研究者自行生成的使用日志、徽章截图、官方文档、App Store版本记录、Reddit公开讨论以及第三方数据工具的输出。

**📈 对比分析**

通过对徽章类型与时间维度的定性对比，发现2025年的功能更新仅是表面修饰，核心游戏化机制未显著改进；未给出量化性能指标，而是以伦理风险描述为主。

**⚠️ 局限性**

局限性在于研究仅基于观察与公开信息，缺乏用户访谈与内部设计文档，样本局限于挪威与芬兰的文化环境，结果可能不具普适性。

---

## Proof of Commitment: A Human-Centric Resource for Permissionless Consensus

**arXiv ID:** 2601.04813 | [PDF](https://arxiv.org/pdf/2601.04813v1)

**作者:** Homayoun Maleki `[一作]` (University of Deusto), Jon Legarda `[通讯]` (University of Deusto)

**关键词:** `Distributed, Parallel, and Cluster Computing`

### 📋 论文摘要

**🎯 论文内容**

提出了一种以实时人类投入为稀缺资源的共识原语 Proof of Commitment（PoCmt），通过人类挑战算子（HCO）实现领导选举权重；

**💡 创新点**

创新点在于将共识权重绑定到不可并行、可计量的人类时间，而非传统的算力或资本，并证明了该机制在并行可扩展资源系统中无法实现的线性 Sybil 成本；

**🔧 技术方法**

使用了基于 VRF 的加权抽签、可滑点奖励/惩罚机制、时间窗口和人类挑战算子，并结合权重背骨分析推导安全性；

**📊 数据集**

没有使用真实数据集，实验采用基于参数的合成模拟来验证模型；

**📈 对比分析**

通过与传统 PoW/PoS 以及基于存储或社交图的方案对比，证明了 PoCmt 在安全性（安全、可达性、比例公平）和性能（线性 Sybil 成本、领导公平性）方面的优势；

**⚠️ 局限性**

局限性包括对人类挑战的可持续性和可访问性假设、对时钟同步的依赖、对人类劳动力市场的敏感性以及可能的自动化攻击与长周期的人机能力差距。

---

## Learnable Multipliers: Freeing the Scale of Language Model Matrix Layers

**arXiv ID:** 2601.04890 | [PDF](https://arxiv.org/pdf/2601.04890v1)

**作者:** Maksim Velikanov `[一作]` (Hugging Face), Hakim Hacid `[通讯]` (Hugging Face)

**关键词:** `Machine Learning` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

为了解除矩阵层因权重衰减导致的噪声‑WD 平衡对尺度的束缚，论文在语言模型的线性层中加入可学习的标量和向量乘子，允许权重随数据自适应缩放。

**💡 创新点**

创新点在于将矩阵权重的重参数化与可学习乘子相结合，突破传统 μP 方案需要手动调节前向/WD 乘子的限制，并在不增加推理成本的前提下提升表示尺度多样性。

**🔧 技术方法**

技术实现包括：Adam/Muon 优化器、权重衰减分析、Brownian 动力学模拟、梯度裁剪调整、乘子初始化与学习率调度等；同时设计了针对残差、注意力、SSM 和 MLP 块的乘子放置策略。

**📊 数据集**

在大规模通用预训练数据集上进行实验，并在 HellaSwag、ARC‑C、MMLU、MATH、GSM8K、BBH 等多项下游基准上进行评估。

**📈 对比分析**

通过与不使用乘子、仅调学习率、手工调 μP 参数等四种配置在同等 200 GT 训练预算下比较，LRM 在平均基准分数上提升约 1–1.2 %，在推理与推理相关任务上提升尤为明显。

**⚠️ 局限性**

局限性包括：缺乏完整的 μP 规模规则以指导乘子与宽度的自动匹配、对不同能力提升不均衡、对梯度噪声阈值的理论阐释不足，以及对更大模型和多种优化器的普适性仍需进一步验证。

---

## Distributed Online Convex Optimization with Efficient Communication: Improved Algorithm and Lower bounds

**arXiv ID:** 2601.04907 | [PDF](https://arxiv.org/pdf/2601.04907v1)

**作者:** Sifan Yang `[一作]` (Nanjing University), Lijun Zhang `[通讯]` (Nanjing University)

**通讯引用:** 28060 | **OpenAlex IDs:** https://openalex.org/A5115603699

**关键词:** `Machine Learning` `Optimization`

### 📋 论文摘要

**🎯 论文内容**

提出一种两级分块的分布式在线凸优化算法 Top‑DOGD，并给出相应的下界证明。

**💡 创新点**

创新点包括：① 在线压缩 gossip 策略与投影误差补偿方案的耦合；② 通过两级分块统一两种技术，既减少一致性误差，又降低压缩误差；③ 首次给出含压缩比例 ω 的下界，证明上界几乎最优；④ 将算法推广到一点/两点 bandit 反馈场景，取得更优的 regret 上界。

**🔧 技术方法**

主要技术：分布式在线梯度下降（OCO），压缩投影（ω‑contractive 压缩器）、多步 gossip 与压缩误差递归消除、投影误差补偿、两级分块更新框架、经典一点/两点梯度估计。

**📊 数据集**

本文为理论性工作，无实验数据集；主要在理论分析与证明上进行研究。

**📈 对比分析**

与之前的 DC‑DOGD 等方法比较，Top‑DOGD 在 ω、ρ、n 三个维度上均取得更优的上界（ω 的指数从 -2/4 降至 -1/2 或 -1，n 的指数从 √n 或 n 提升至更小或保持不变），在 bandit 反馈场景下也比已有结果改进了相同维度的幂次。

**⚠️ 局限性**

限制：仍存在 O(n) 的规模因子；需在每个块末更新，实际同步与延迟的影响未被实验验证；对压缩器的 ω‑contractive 性质有较强假设，实际压缩器可能不完全满足；仅给出理论下界，未给出可实现的最优算法实现细节。

---

## Parallel Quadratic Selected Inversion in Quantum Transport Simulation

**arXiv ID:** 2601.04904 | [PDF](https://arxiv.org/pdf/2601.04904v1)

**作者:** Vincent Maillou `[一作]` (ETH Zurich), Mathieu Luisier `[通讯]` (ETH Zurich)

**通讯引用:** 7371 | **OpenAlex IDs:** https://openalex.org/A5058537769

**关键词:** `Distributed, Parallel, and Cluster Computing` `Reinforcement Learning` `Physics Related`

### 📋 论文摘要

**🎯 论文内容**

提出了一套针对非平衡格林函数（NEGF）方法中非方阵稀疏矩阵的分布式并行选定逆与二次矩阵方程求解算法，可处理块三对角及块三对角加箭头结构。

**💡 创新点**

创新点包括：①将递归格林函数（RGF）算法扩展到块三对角+箭头稀疏矩阵；②基于域分解与抽取系统的并行分布式求解框架；③在该框架下实现了GPU加速，兼顾CPU与GPU两种架构。

**🔧 技术方法**

使用了递归格林函数、选定逆、二次矩阵方程求解、块-三对角/块三对角+箭头稀疏矩阵分解、域分解与并行拼装、GPU矩阵乘、MPI / NCCL 通信等技术。

**📊 数据集**

实验数据集包括真实的 NR_FET（NR_3408）纳米带晶体管矩阵和合成的 SD 数据集（n=32~1024，b=1024）。

**📈 对比分析**

与 PARDISO 的选定逆/二次求解器进行对比；单 GPU 上实现 120× 速度提升，16 GPU 上与 PARDISO 在 16 倍更小装置上相比 5.2× 更快；在 GH200 上的弱扩展测试中，最大 32 GPU 的并行效率约 20%。

**⚠️ 局限性**

主要限制包括：域分解导致的额外填充和顺序系统求解瓶颈；BTA 箭头块尺寸过大时效率下降；中间分区间的通信成本随块数增加而显著；整体扩展性受限于块数与箭头尺寸。

---

## Rotation-Robust Regression with Convolutional Model Trees

**arXiv ID:** 2601.04899 | [PDF](https://arxiv.org/pdf/2601.04899v1)

**作者:** Hongyi Li `[一作]` (Harbin Institute of Technology), Jun Xu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 13900 | **OpenAlex IDs:** https://openalex.org/A5023647213

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Optimization` `Convolutional Neural Network` `Supervised Fine-Tuning` `Image`

### 📋 论文摘要

**🎯 论文内容**

研究在图像输入下的旋转鲁棒学习，利用卷积模型树（CMT）在 MNIST 上预测数字的周长。

**💡 创新点**

创新点是提出三种几何感知的分裂偏置（卷积平滑、倾斜优势约束和重要性裁剪），并在部署时通过置信度代理实现离散旋转搜索，无需重新训练。

**🔧 技术方法**

使用的技术包括卷积模型树、节点局部岭回归、Adam 优化、卷积平滑、重要性裁剪、倾斜约束以及置信度代理的旋转搜索。

**📊 数据集**

数据集为标准 MNIST，使用阈值 0.1 的二值掩码计算周长作为旋转不变目标，并在 7 个旋转角度上进行评估。

**📈 对比分析**

与 Ridge、标准森林对比，CMT‑Full 在平均 MAE 上较基线下降约 16%，在极端旋转下通过 OS 可进一步提升约 30%，但在 0° 时可能出现性能下降。

**⚠️ 局限性**

局限包括仅在 MNIST 合成回归标签上测试、离散角度网格与简单置信度代理、节点岭回归的数值优化成本较高等。

---

## FibreCastML: An Open Web Platform for Predicting Electrospun Nanofibre Diameter Distributions

**arXiv ID:** 2601.04873 | [PDF](https://arxiv.org/pdf/2601.04873v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## DVD: A Robust Method for Detecting Variant Contamination in Large Language Model Evaluation

**arXiv ID:** 2601.04895 | [PDF](https://arxiv.org/pdf/2601.04895v1)

**作者:** Renzhao Liang `[一作]` (Beihang University), Cunxiang Wang `[通讯]` (Tsinghua University)

**关键词:** `Artificial Intelligence` `Generation` `Data Synthesis` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种基于生成分布方差的Variant Contamination检测方法（DVD），并构建了包含Omni‑MATH和SuperGPQA的首个variant contamination基准。

**💡 创新点**

核心创新在于通过多次温度采样计算低概率token的合成难度方差，利用该方差捕捉模型在记忆一致与漂移状态之间的切换，从而高效识别语义等价的训练样本。

**🔧 技术方法**

使用温度采样、多次生成、合成难度统计、方差分解理论，并对比嵌入相似度、Perplexity、Min‑K%++、CDD、Loss、Zlib等基线进行评估。

**📊 数据集**

基准数据来自Omni‑MATH（数学推理）和SuperGPQA（通用推理），并通过GPT‑4o生成并过滤语义等价变体来模拟污染。

**📈 对比分析**

将DVD与七种基线在不同模型规模（Qwen2.5 1.5B–32B、Llama3.1 8B）和微调方式（全参数、LoRA）下的AUC进行对比，结果显示DVD在所有设置下平均提升0.15–0.25 AUC，显著优于最佳基线。

**⚠️ 局限性**

局限性包括对生成长度、停止准则、答案格式敏感；需要多次采样，查询成本高；在开放式或多路径任务中方差基线升高；基准仅基于LLM生成变体，真实预训练污染可能更复杂。

---

## Scaling Vision Language Models for Pharmaceutical Long Form Video Reasoning on Industrial GenAI Platform

**arXiv ID:** 2601.04891 | [PDF](https://arxiv.org/pdf/2601.04891v1)

**作者:** Suyash Mishra `[一作]` (Roche), Baddu Narendra `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Drug Discovery` `Transformer` `Vision Language Model` `Large Language Model` `Video` `Audio` `Text` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

在制药行业构建工业规模的多模态生成AI平台，对超过20万份PDF、25,326个多格式视频和888个多语言音频进行处理，并系统评估40+视觉-语言模型在长视频及GPU受限环境下的表现。

**💡 创新点**

首次对VLM在长视频和工业GPU约束下的可扩展性进行全面基准，提出四大关键发现（多模态提升、注意力机制与GPU匹配、时间对齐瓶颈、长视频切分误差）并扩展Video‑MME基准。

**🔧 技术方法**

采用LLM/ALM融合、Whisper音频转录、FFmpeg预处理、SDPA与FlashAttention注意力机制以及知识图谱评估框架。

**📊 数据集**

自有制药行业数据集（25,326个视频、888个音频、200k+ PDF）以及公开Benchmark Video‑MME和MMBench。

**📈 对比分析**

通过与公开基准和自有数据对40+模型进行对比，发现多模态提升8/12任务域，SDPA在commodity GPU上比FlashAttention快4倍且精度更高，长视频关键帧检测准确率低，切分策略反而增加误差。

**⚠️ 局限性**

局限性包括长视频时间对齐和关键帧检测失败、对高端GPU的依赖、GPU架构差异导致的性能波动，以及对其他监管领域推广的未知性。

---

## SmartSearch: Process Reward-Guided Query Refinement for Search Agents

**arXiv ID:** 2601.04888 | [PDF](https://arxiv.org/pdf/2601.04888v1)

**作者:** Tongyu Wen `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 3838 | **OpenAlex IDs:** https://openalex.org/A5010558184

**关键词:** `Artificial Intelligence` `Retrieval` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于过程奖励引导的查询优化框架 SmartSearch，专注提升搜索代理在多轮检索中的中间查询质量。

**💡 创新点**

创新点在于：①引入双层信用评估的过程奖励，为每一步查询提供数值分数和文本反馈；②设计查询改进机制，自动识别低质量查询并在此基础上重新生成后续检索；③采用三阶段递进式训练课程（仿真、对齐、通用化），逐步让模型掌握提升查询质量的能力。

**🔧 技术方法**

技术包括：大语言模型（LLM）搜索代理、双层过程奖励评估（规则+模型）、轻量化学生模型用于评分与改写、对齐优化（DPO）、强化学习（GRPO）与过程奖励融合、轻量化模型蒸馏。

**📊 数据集**

使用基于 Wikipedia 的 Asearcher-Base 训练集，并在四个知识密集型基准（2WikiMultihopQA、HotpotQA、Bamboogle、Musique）以及两个网络搜索基准（GAIA、WebWalker）上进行评测。

**📈 对比分析**

与提示式、仅结果奖励的 RL 方法以及其他过程奖励 RL 方法进行对比。SmartSearch 在所有任务中均取得最高的 EM/F1，平均提升约 24% EM、21% F1；在网络搜索任务中平均 F1 提升约 5%，并表现出更高的查询质量与搜索效率。

**⚠️ 局限性**

局限性包括：仍依赖预先训练的学生模型，若使用更强大模型可略微提升效果但成本显著增加；当前仅针对检索型任务，未探讨在更复杂多模态或开放域对话场景中的适用性。

---

## Responsibility Measures for Conjunctive Queries with Negation

**arXiv ID:** 2601.04868 | [PDF](https://arxiv.org/pdf/2601.04868v1)

**作者:** Meghyn Bienvenu `[一作]` (University of Bordeaux), Pierre Lafourcade `[通讯]` (University of Bordeaux)

**关键词:** `Databases`

### 📋 论文摘要

**🎯 论文内容**

本文提出了两种对含否定原子（UCQⁿ）查询的责任度量方法，分别基于“签名支持”（signed support）和“正向支持”（positive support）来定义事实的相关性，并将经典的 MS‑Shapley 与 Drastic‑Shapley 责任度量迁移到非单调查询；

**💡 创新点**

创新点在于（1）首次系统地探讨非单调查询下的解释与相关性概念，并给出两种可计算的支持定义；（2）在此基础上构造新的责任度量，能够将已知的对单调查询的复杂度结果平滑迁移；（3）对数据复杂度与组合复杂度进行全面分析，确定了若干可归约至可解子类（如自连接自由、无非层次路径等）下的多项式时间解法；

**🔧 技术方法**

主要技术包括：
- 通过签名关系（+P、-P）将含否定查询转化为等价的单调查询；
- 采用最小支持（minimal support）和最小正向支持（minimal positive support）构造支持集合；
- 利用 Shapley 值与加权最小支持（WSMS）框架来定义责任度量；
- 通过树宽（generalized hypertree width）和自连接宽度（self‑join width）等结构限制实现组合复杂度的可判定；
- 采用谓词逻辑一阶公式实现对相关性的 AC⁰ 判定；

**📊 数据集**

论文主要以理论分析为主，示例中使用的数据库包括：
- 以食谱为例的关系 I（如 I(mm,fish)）作为演示；
- 图数据库例子（关系 E）展示签名支持；
- 通过构造的合成数据库（如 R(c_i,c_i+1) 等）说明正向支持的边界；
- 真实数据集并未在实验中使用。

**📈 对比分析**

方法比较主要以理论复杂度为基准：
- 对签名支持的 MS‑Shapley 与 Drastic‑Shapley 在数据复杂度上均为多项式时间；
- 在组合复杂度方面，若查询满足自连接无 mergeable 原子、树宽和自连接宽度受限，则仍为多项式；
- 与 Reshef 等人提出的“影响相关性”度量相比，本文的相关性定义更精细，能够区分真正参与满足查询的事实；
- 由于缺乏实验数据，无法给出运行时间或内存占用等性能指标，但通过复杂度证明展示了在可解子类上的效率。

**⚠️ 局限性**

局限性包括：
- 对 UCQⁿ 的组合复杂度仍然是 shP‑难，本文仅给出受限子类的多项式解法；
- 仅讨论了签名与正向支持的两种解释，对更广泛的非单调查询（如全称查询、包含多重否定的查询）尚未给出有效的相关性定义；
- 方案不直接处理外部事实（exogenous facts），若要与之兼容需额外改造；
- 论文缺少实验验证，无法评估在大规模真实数据上的实用性。

---

## Key-Value Pair-Free Continual Learner via Task-Specific Prompt-Prototype

**arXiv ID:** 2601.04864 | [PDF](https://arxiv.org/pdf/2601.04864v1)

**作者:** Haihua Luo `[一作]` (University of Jyväskylä), Fengyu Cong `[通讯]` (Dalian University of Technology)

**通讯引用:** 4329 | **OpenAlex IDs:** https://openalex.org/A5042937962

**关键词:** `Artificial Intelligence` `Classification` `Recognition` `Transformer` `Prompt Engineering` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出一种不使用键值对的连续学习框架 ProP，通过将任务特定提示与对应的原型绑定，实现无检索的任务感知推断。

**💡 创新点**

创新点包括：① 将任务特定提示与原型直接绑定，消除键值匹配导致的任务间干扰；② 在提示初始化时加入 L2 正则化，抑制过大初始值，提高稳定性；③ 通过冻结预训练模型并在其基础上融合提示学习，充分保留预训练特征。

**🔧 技术方法**

使用技术：预训练视觉 Transformer (ViT) 作为特征提取器；任务特定可学习提示；类别原型（类均值）构造；相似度计算（余弦/归一化）；交叉熵 + L2 正则化训练；无示例回放。

**📊 数据集**

实验数据集涵盖多种挑战场景：CIFAR-100、CUB-200、ImageNet‑A、ImageNet‑R、ObjectNet、OmniBenchmark、VTAB，均采用 Class‑Incremental 设定。

**📈 对比分析**

与 L2P、DualPrompt、CODA‑Prompt、APER 等主流提示方法以及 LwF、SDC、iCaRL、DER 等传统方法在同一预训练模型上对比。ProP 在大多数任务上取得最高的平均/最后准确率，并在无示例回放的前提下超过所有带示例的对照方法，显示出更强的抗遗忘能力。

**⚠️ 局限性**

局限性包括：① 仍需在每个任务上进行一次前向传播以生成特征，计算量随任务数线性增长；② 仅验证了 Class‑Incremental 场景，对 Domain‑Incremental 或 Task‑Incremental 的适用性未作深入探讨；③ 对极端多任务或高维特征的可扩展性尚未评估。

---

## A Navigational Approach for Comprehensive RAG via Traversal over Proposition Graphs

**arXiv ID:** 2601.04859 | [PDF](https://arxiv.org/pdf/2601.04859v1)

**作者:** Maxime Delmas `[一作]` (Idiap Research Institute), André Freitas `[通讯]` (Cancer Biomarker Centre)

**关键词:** `Computation and Language` `Retrieval` `Recommendation System` `Graph Neural Network` `Large Language Model` `Retrieval-Augmented Generation` `Graph` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出ToPG框架，通过构建包含命题、实体和段落的异构图，并在此上实现建议-选择循环的查询感知图遍历，以支持多种QA需求。

**💡 创新点**

创新点在于将命题视为节点，结合图结构与LLM反馈的建议-选择机制，实现精细命题级检索与结构化推理的统一。

**🔧 技术方法**

使用的技术包括命题抽取、实体链接、图嵌入、查询感知个性化PageRank、LLM驱动的选择与迭代搜索、社区检测等。

**📊 数据集**

评测数据集包括PopQA、GraphRAG-Benchmark（Fact Retrieval、Complex Reasoning、Contextual Summarization）、HotPotQA、MusiQue、UltraDomain教材等。

**📈 对比分析**

与GraphRAG、LightRAG、HippoRAG 2等基线对比，ToPG在简单QA中略逊于基线，但在多跳和抽象QA中取得显著提升，尤其在多跳任务上提升超过10%准确率。

**⚠️ 局限性**

主要局限是高昂的索引与推理token成本，以及对实体消歧与命题抽取质量的依赖，缺乏自动模式路由与轻量化选择器。

---

## Token Maturation: Autoregressive Language Generation via Continuous Token Dynamics

**arXiv ID:** 2601.04854 | [PDF](https://arxiv.org/pdf/2601.04854v1)

**作者:** Oshri Naparstek `[一作]` (IBM Research), Oshri Naparstek `[通讯]` (IBM Research)

**关键词:** `Computation and Language` `Generation` `Transformer` `Large Language Model` `Contrastive Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出一种称为 Token Maturation 的连续自回归语言生成框架，将词元表示为随时间演化的连续向量，直至达到稳定点后再进行离散化，从而实现不依赖 token 级采样的文本生成。

**💡 创新点**

创新点包括：① 将预测与离散化解耦，利用连续向量动态演化完成不确定性分辨；② 通过“液态尾”机制在每一步中保持若干未决 token 的连续状态；③ 采用对比学习 (InfoNCE) 与回归损失相结合的训练目标，防止表示崩塌；④ 允许在连续空间上注入噪声、平滑或其他干预，提供可解释的“前瞻”视图。

**🔧 技术方法**

核心技术：连续嵌入空间预测、因果 Transformer、噪声层级编码 (α 级别)、FiLM 条件化尾长、连续向量更新规则（z̃ ← z̃ + α (ẑ - z̃)）、对比损失、MSE 损失、分类器无关指导 (CFG)、argmax 离散化。

**📊 数据集**

使用 FineWeb‑10BT 大规模 Web 文本数据集进行训练，并在 GPT‑2 的嵌入矩阵上微调。

**📈 对比分析**

与传统基于 softmax 的自回归解码进行对比，实验表明：① 生成文本连贯且多样；② 在成熟过程中熵保持相对恒定，且可通过 tail 长度控制多样性；③ 对比损失显著提升表示质量；性能评估主要以人工可读性和多样性为主，没有给出定量 BLEU/ROUGE 等指标。

**⚠️ 局限性**

局限性：① 仅在中等规模模型（24 层 GPT‑2 变体）上验证，缺乏大规模基准结果；② 训练/推理成本相对传统解码略有增加；③ 目前成熟调度固定，缺乏自适应或学习的时间步选择；④ 未在标准文本生成任务上与主流模型做严格数值比较。

---

## LGTD: Local-Global Trend Decomposition for Season-Length-Free Time Series Analysis

**arXiv ID:** 2601.04820 | [PDF](https://arxiv.org/pdf/2601.04820v1)

**作者:** Chotanansub Sophaken `[一作]` (King Mongkut’s University of Technology Thonburi), Chainarong Amornbunchornvej `[通讯]` (National Electronics and Computer Technology Center)

**通讯引用:** 88 | **OpenAlex IDs:** https://openalex.org/A5007113918

**关键词:** `Databases` `Optimization` `Computational Efficiency` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

提出了一种无需预设季节周期的时间序列分解框架（LGTD）

**💡 创新点**

把季节性视为局部趋势重复的涌现，利用自适应误差驱动的局部线性趋势分割，消除了对季节周期的手工设定，并提供了线性时间复杂度和终止保证

**🔧 技术方法**

先使用平滑/回归模型提取全局趋势，再对残差应用 AutoTrend‑LLT（自适应局部线性趋势推断）进行误差驱动分段，核心技术包括百分位阈值迭代、局部线性回归和误差驱动终止策略

**📊 数据集**

合成数据：三种趋势 × 三种季节周期（固定、转移、漂移）共9个系列；真实数据：ETTh1/ETTh2（电力负荷）和 SILSO 太阳黑子系列

**📈 对比分析**

与 STL、STR、RobustSTL、FastRobustSTL、ASTD、OnlineSTL、OneShotSTL、ASTD_Online 等基线在 MAE 上进行对比。LGTD 在所有合成数据集和变量季节周期数据集上获得最低整体 MAE，尤其在周期漂移情况下明显优于传统方法；在真实数据上表现出平滑的全局趋势、可解释的季节成分和低方差残差

**⚠️ 局限性**

对极端噪声或极长周期仍有挑战；对初始百分位/步长参数略有敏感；需要先确定全局趋势模型，且在多维/多变趋势场景下的扩展尚未验证

---

## Privacy-Utility Trade-offs Under Multi-Level Point-Wise Leakage Constraints

**arXiv ID:** 2601.04815 | [PDF](https://arxiv.org/pdf/2601.04815v1)

**作者:** Amirreza Zamani `[一作]` (KTH), Mikael Skoglund `[通讯]` (KTH)

**通讯引用:** 8707 | **OpenAlex IDs:** https://openalex.org/A5041348422

**关键词:** `Information Theory` `Safty and Privacy` `Optimization`

### 📋 论文摘要

**🎯 论文内容**

本文提出一种在多层点位泄露约束下的信息论隐私机制，旨在在满足不同输出泄露预算的前提下最大化有用信息的传递。

**💡 创新点**

创新点包括引入多级点位泄露度量，可为每个公开结果指定不同泄露阈值，并在小泄露 regimes 下给出二元解与线性规划近似，从而兼顾完美隐私与非零泄露的混合需求。

**🔧 技术方法**

采用信息几何的二阶泰勒展开、奇异值分解、凸多面体与极点分析，将原非凸优化转化为可闭式求解的二次规划或线性规划。

**📊 数据集**

实验使用人工生成的联合分布与一个混合案例进行验证，未使用公开数据集。

**📈 对比分析**

与传统单一泄露阈值方法对比，本文在保持严格隐私控制的同时实现更高的互信息利用率；相较于完全完美隐私方案，性能显著提升。

**⚠️ 局限性**

主要局限在于仅适用于泄露量较小的假设，近似误差随泄露增大而增大；同时要求泄露矩阵可逆或满秩，若不满足则需进一步处理。

---

## V-FAT: Benchmarking Visual Fidelity Against Text-bias

**arXiv ID:** 2601.04897 | [PDF](https://arxiv.org/pdf/2601.04897v1)

**作者:** Ziteng Wang `[一作]` (Chinese University of Hong Kong), Songxiang Liu `[通讯]` (Meituan)

**关键词:** `Computation and Language` `Recognition` `Data-Centric Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Vision Language Model` `Image` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

研究多模态大语言模型（MLLMs）在视觉推理任务中的文字偏差，提出并构建了 V-FAT 诊断基准，定义视觉鲁棒性得分（VRS），并评估 12 款主流 MLLM 的性能。

**💡 创新点**

创新点包括：①三层级文本偏差诊断框架，系统分离内部语料偏差与外部指令偏差；②视觉鲁棒性得分（VRS）指标，用调和平均方式同时衡量准确率与对文本诱导的抵抗力；③通过对比内部与外部偏差的交互效应，揭示模型在多重偏差冲突下的“视觉崩溃”现象。

**🔧 技术方法**

技术方法主要涉及：对抗性图像构造（counterfactual images）、多模态推理评测、统计指标计算（Accuracy、VRS）、对模型输出的自动化判定（利用 DeepSeek-Chat 判定开放式答案），以及层级化提示设计来诱导不同偏差。

**📊 数据集**

使用的数据集为 V-FAT，包含 4,026 例 VQA（4,020 例加验证），从 VLind-Bench 与 WEIRD 等来源抽取的 800+ 反事实图像，覆盖环境、物理、社交、时间、生命、功能等六大主题。

**📈 对比分析**

通过将各模型在三层级（内部、外部、交互）下的多选与开放式回答准确率以及 VRS 进行对比，发现主流闭源模型在层级 3（内部+外部一致偏差）仍保持相对稳定，而大部分开源模型在此层级显著性能下降；VRS 显示不同模型在抵御文本诱导方面存在显著差距，规模提升并不能同步提升鲁棒性。

**⚠️ 局限性**

局限性包括：①评估仅基于黑盒输出，无法深入分析模型内部机制；②数据集侧重特定反事实场景，缺乏对多样化真实世界对抗情况的覆盖；③未能分离图像编码器、跨模态连接器与语言模型对错误的贡献；④链式推理等提示策略可能放大文本偏差，缺少进一步的缓解措施。

---

## Analyzing Message-Code Inconsistency in AI Coding Agent-Authored Pull Requests

**arXiv ID:** 2601.04886 | [PDF](https://arxiv.org/pdf/2601.04886v1)

**作者:** Jingzhi Gong `[一作]` (King's College London), Jie M. Zhang `[通讯]` (King's College London)

**通讯引用:** 3896 | **OpenAlex IDs:** https://openalex.org/A5088708850

**关键词:** `Software Engineering` `AI Code Assistant` `Agentic AI` `Text`

### 📋 论文摘要

**🎯 论文内容**

分析AI编码代理生成的拉取请求（PR）描述与代码变更之间的一致性，构建并量化PR-MCI指标，并评估其对PR接受率和合并时长的影响。

**💡 创新点**

首次在大规模数据集上系统评估Agentic-PR的描述一致性，提出基于范围、文件类型和任务类型三维相似度的PR-MCI度量，构建八类不一致类型的分类法，并证明不一致导致低接受率和高耗时。

**🔧 技术方法**

采用三信号加权相似度模型、人工标注与F1评估、卡方与Mann-Whitney U统计检验、多元回归分析，并与嵌入模型和一致性检查基线进行对比。

**📊 数据集**

使用AIDev数据集（AIDev-pop子集）中的23,247个PR，来自五个AI代理，覆盖星级>100的MIT/Apache 2.0开源项目。

**📈 对比分析**

通过人工标注验证PR-MCI阈值0.61后得到1.7%高-MCI；高-MCI PR的接受率低51.7%（28.3% vs 80.0%），合并时长高3.5倍（55.8h vs 16.0h）；与基线嵌入模型F1 0.15和协同方法F1 0.567对比，显示所提度量性能更优。

**⚠️ 局限性**

度量为启发式，可能忽略细微不一致；阈值调优影响普适性；数据筛选导致外部有效性受限；代理样本不均衡；分类法受限于已标注样本；结果为观察性，未证实因果关系。

---

## Mind2Report: A Cognitive Deep Research Agent for Expert-Level Commercial Report Synthesis

**arXiv ID:** 2601.04879 | [PDF](https://arxiv.org/pdf/2601.04879v1)

**作者:** Mingyue Cheng `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27290 | **OpenAlex IDs:** https://openalex.org/A5048237545

**关键词:** `Computation and Language` `Large Language Model` `Retrieval-Augmented Generation` `Text` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于认知流程的深度研究代理Mind2Report，能自动化生成专业级商业报告。

**💡 创新点**

创新点在于引入主动意图探询、动态内存适配搜索与分段迭代合成，三者协同解决报告质量、可靠性和覆盖度的不足。

**🔧 技术方法**

技术包括大型语言模型（LLM）+检索增强生成（RAG）、多维反思评估、基于节点的章节树、动态内存管理与迭代写作框架。

**📊 数据集**

使用自建的CRS200评估集（200条时间敏感的商业查询、专家标注关键点、快照式引用来源）进行验证。

**📈 对比分析**

与OpenAI、Gemini等商业深度研究代理以及多款开源工作流代理进行对比，Mind2Report在内容相关性、结构连贯性、时效性、逻辑一致性、覆盖度等指标均取得最高分，整体性能明显优于所有基线。

**⚠️ 局限性**

局限包括对底层LLM质量的依赖、递归搜索导致推理延时与计算成本上升、自动化指标可能偏见、模型主要针对商业分析，跨领域推广尚待验证。

---

## ChronosAudio: A Comprehensive Long-Audio Benchmark for Evaluating Audio-Large Language Models

**arXiv ID:** 2601.04876 | [PDF](https://arxiv.org/pdf/2601.04876v1)

**作者:** Kaiwen Luo `[一作]` (North China Electric Power University), Qingsong Wen `[通讯]` (Squirrel AI Learning)

**关键词:** `Sound` `Transformer` `Large Language Model` `Audio` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出ChronosAudio基准，用于评估Audio Large Language Models在长音频任务上的理解能力。

**💡 创新点**

创新点是首次构建多任务长音频评测框架，揭示长上下文下ALLMs的性能崩溃、注意力稀释以及恢复上限。

**🔧 技术方法**

采用Transformer注意力可视化、Sparse Attention和Sliding Window Attention机制进行性能与注意力分析，并使用多种定量指标评估模型。

**📊 数据集**

使用36,000个测试实例、200小时音频（短/中/长三类）组成的ChronosAudio数据集，覆盖六大任务类别。

**📈 对比分析**

与16个SOTA ALLMs（包括开源与闭源）对比，发现长音频性能下降超过90%，Sparse Attention可恢复约50%，展示了当前模型的显著短板。

**⚠️ 局限性**

局限性：数据主要为英文、清晰环境，缺乏跨语言与噪声鲁棒性；未探索训练层面的改进；闭源模型的注意力分析受限。

---

## Intelligent resource allocation in wireless networks via deep reinforcement learning

**arXiv ID:** 2601.04842 | [PDF](https://arxiv.org/pdf/2601.04842v1)

**作者:** Marie Diane Iradukunda `[一作]` (African Institute for Mathematical Sciences), Yaé Ulrich Gaba `[通讯]` (Sefako Makgatho Health Sciences University)

**通讯引用:** 402 | **OpenAlex IDs:** https://openalex.org/A5027109805

**关键词:** `Networking and Internet Architecture` `Reinforcement Learning` `Optimization` `Reinforcement Learning`

### 📋 论文摘要

**🎯 论文内容**

在随机多用户无线网络中，使用深度 Q‑网络（DQN）学习动态功率分配策略，代替传统的固定、随机和水分配方法；

**💡 创新点**

创新点在于证明了模型无关的 DQN 能够在不需显式系统模型的情况下，逼近甚至超越水分配的吞吐量，并在长期奖励优化中自然产生公平性；

**🔧 技术方法**

采用的技术主要是深度强化学习（DQN）、经验回放、目标网络以及离散功率动作空间的 MDP 建模；

**📊 数据集**

使用的数据集为仿真产生的无偏快衰落信道数据（3 或 5 用户，均匀分布 0.1–1.0 的通道增益）以及对应的 SNR、吞吐和能耗计算；

**📈 对比分析**

与基线方法（固定功率、随机分配、理论水分配）对比，DQN 在吞吐量上略优于水分配，公平性接近固定功率，能效略低于固定功率，但整体性能显著优于随机和固定策略；

**⚠️ 局限性**

局限性包括：简化的正交接入、完美 CSI、离线训练、对超大规模网络的可扩展性不足以及对探索率等超参数高度敏感。

---

## Character Detection using YOLO for Writer Identification in multiple Medieval books

**arXiv ID:** 2601.04834 | [PDF](https://arxiv.org/pdf/2601.04834v1)

**作者:** Alessandra Scotto di Freca `[一作]` (University of Cassino and Southern Lazio), Claudio De Stefano `[通讯]` (University of Cassino and Southern Lazio)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Recognition` `Image`

### 📋 论文摘要

**🎯 论文内容**

本文基于YOLOv5s6对中世纪手稿中的字母“a”进行检测，以实现自动注释与抄写者识别。

**💡 创新点**

创新点在于将YOLO用于手写字符检测，利用检测置信度做风格相似度评估，从而在跨手稿中实现写者归属阈值判定。

**🔧 技术方法**

使用YOLOv5s6深度学习目标检测模型，配合模板匹配初始化与迭代微调。

**📊 数据集**

数据集为阿维拉圣经与特伦托圣经两卷手稿的高分辨率页面，标注字母“a”。

**📈 对比分析**

与传统模板匹配方法对比，YOLO在字符检测数量上提升5–35%，并在置信度阈值下实现最高92.6%准确率。

**⚠️ 局限性**

局限在于仅关注单一字符，阈值设定需手工验证，且不同抄写者的写风差异导致提升幅度不均。

---

## Zoomy: flexible modeling and simulation software for free-surface flows

**arXiv ID:** 2601.04826 | [PDF](https://arxiv.org/pdf/2601.04826v1)

**作者:** Ingo Steldermann `[一作]` (RWTH Aachen), Julia Kowalski `[通讯]` (RWTH Aachen)

**通讯引用:** 2549 | **OpenAlex IDs:** https://openalex.org/A5042152059

**关键词:** `Computational Engineering, Finance, and Science` `Physics Related`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了Zoomy软件框架，用于统一描述、符号分析和数值求解深度平均自由表面流动的分层模型。

**💡 创新点**

创新点在于：①将层级模型的符号化与数值求解分离，支持可变多项式基函数和符号深度积分；②提供自动求导的Jacobian、线性稳定性分析与非保守项处理；③集成可视化与交互式GUI，支持多种后处理与模型对比。

**🔧 技术方法**

使用SymPy进行符号运算、代码转换（JAX/NumPy/C），JAX实现自动微分与GPU/TPU加速；FenicsX和自研FVM算子用于数值求解；GMSH生成网格；Panel实现GUI。

**📊 数据集**

未使用传统公开数据集，而是通过自定义数值实验（如坝破、Poisson测试、VAM预测-校正、SME 2D）验证框架功能。

**📈 对比分析**

通过与解析解或已知测试对比，验证了数值精度；性能上，基于JAX的FVM和Newton求解器在CPU/GPU上表现良好，支持无缝切换后端；但文中未给出量化速度指标。

**⚠️ 局限性**

局限包括：目前仅支持单层网格（collocated）且缺乏对斜面/自适应网格的完善支持；非水动力耦合功能尚未完成；高阶模型的数值稳定性和复杂材料闭包仍需进一步研究。

---

## SOVABench: A Vehicle Surveillance Action Retrieval Benchmark for Multimodal Large Language Models

**arXiv ID:** 2601.04824 | [PDF](https://arxiv.org/pdf/2601.04824v1)

**作者:** Oriol Rabasseda `[一作]` (Universitat de Barcelona), Sergio Escalera `[通讯]` (Universitat de Barcelona)

**通讯引用:** 13261 | **OpenAlex IDs:** https://openalex.org/A5038228433

**关键词:** `Computer Vision and Pattern Recognition` `Retrieval` `Autonomous Driving` `Transformer` `Large Language Model` `Prompt Engineering` `Video` `Text` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文构建了针对车辆监控场景的检索基准 SOVABench，并提出了基于多模态大型语言模型的训练‑free MLLM‑to‑Embedding 框架，用文本描述生成句向量实现视频检索。

**💡 创新点**

创新点包括（1）基于相反动作的交对与内对两种评估协议设计的检索基准，能同时评估动作判别与时序方向；（2）利用 MLLM 的指令式生成与句子级嵌入、最大相似度聚合的无训练嵌入方案，实现可解释的多模态表示。

**🔧 技术方法**

技术上使用 InternVL‑3.5、MiniCPM‑V 4.5 等开源 MLLM 进行指令式文本生成，句子嵌入采用 GTE‑Large‑8152，最大相似度聚合；与 CLIP、SigLIP、VideoCLIP、CLIP4Clip 等对比。

**📊 数据集**

数据集为从 MEVA 与 VIRAT 车辆监控视频中提取的 SOVABench（交对 1,423 个查询，内对 2,300 个查询），同时在 SpatialBench、CountBench、Visual7W‑Count 等公开数据集上验证空间关系与计数任务。

**📈 对比分析**

在交对协议上，MiniCPM‑V 4.5 的 task‑aware 嵌入获得 38.3 mAP，显著高于随机基准 3.4 mAP 及其他对比模型；在内对协议上所有模型仅略高于 50.3 Pair‑mAP，说明时序判别仍具挑战。

**⚠️ 局限性**

局限性在于对时间方向的区分能力有限，主要受限于 MLLM 生成描述的误差、缺失或时间倒置；视频‑MLLM 及大模型未显著提升性能，框架对极短动作或细粒度时序仍受限制。

---

## The Rezk Completion for Elementary Topoi

**arXiv ID:** 2601.04814 | [PDF](https://arxiv.org/pdf/2601.04814v1)

**作者:** Kobe Wullaert `[一作]` (Delft University of Technology), Niels van der Weide `[通讯]` (Radboud University Nijmegen)

**通讯引用:** 71 | **OpenAlex IDs:** https://openalex.org/A5053770022

**关键词:** `Logic in Computer Science`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一种模块化框架，将 Rezk 完成从普通范畴提升到具有结构的范畴（如有限可完备范畴、笛卡尔闭范畴、基本拓扑以及带有自然数对象的 W‑拓扑）

**💡 创新点**

创新点在于利用显示双范畴与显示通用箭头（displayed universal arrows）实现对结构的“继承”与“保留”，从而无需显式构造每一种结构的 Rezk 完成即可证明其可迁移性

**🔧 技术方法**

核心技术包括：Univalent 基础中的弱等价（weak equivalence）与 Rezk 完成的通用性质、显示双范畴的构造、显示通用箭头的形式化，以及对弱等价在特定结构下的保持与反射性证明

**📊 数据集**

论文并未使用传统意义上的数据集；所有结果均为形式化证明，已在 Coq 的 univalent‑math 库中实现

**📈 对比分析**

由于采用形式化证明而非实验验证，本文不做性能或数值比较；主要比较的是在不同结构下 Rezk 完成的可用性与一致性（即对所有结构均满足通用的保留与可继承性），并通过库的自动化检验验证其正确性

**⚠️ 局限性**

局限性包括：对无限极限的提升需要 (∞,−1)-choice 等额外公理；某些结构（如局部笛卡尔闭、abelian 结构）需要额外假设或改写证明；此外，框架目前针对显示双范畴中合同 2‑胞的情况，若涉及更一般的 2‑胞，需要进一步推广

---

## ArcAligner: Adaptive Recursive Aligner for Compressed Context Embeddings in RAG

**arXiv ID:** 2601.05038 | [PDF](https://arxiv.org/pdf/2601.05038v1)

**作者:** Jianbo Li `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 15641 | **OpenAlex IDs:** https://openalex.org/A5017671620

**关键词:** `Computation and Language` `Retrieval` `Compression` `Generation` `Transformer` `Retrieval-Augmented Generation` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出 ArcAligner，一个轻量级模块，利用自适应递归对齐机制提升压缩上下文在 Retrieval‑Augmented Generation（RAG）中的可用性和回答质量。

**💡 创新点**

创新点在于将门控自适应递归对齐与 selective LoRA 结合，逐层对压缩嵌入进行深度语义对齐，并根据信息复杂度动态决定递归深度，既保留关键证据又避免不必要的计算。

**🔧 技术方法**

采用轻量投影、LoRA 适配、门控网络（sigmoid+STE）、三阶段训练（重建、任务微调、门控调优）以及 Mistral‑7B‑Instruct 基础模型。

**📊 数据集**

使用 HotpotQA、2WikiMultiHopQA、NaturalQA、PopQA_longtail、TriviaQA*、WebQuestions 等多跳与开放域 QA 数据集，检索采用 Contriever+RankZephyr。

**📈 对比分析**

与 Naive、StandardRAG、xRAG、COCOM、LLMLingua‑2 等基线对比，ArcAligner 在相同压缩率下在多跳、长尾 QA 任务中显著提升 EM/F1/Acc，尤其在多跳和长尾场景表现突出。

**⚠️ 局限性**

局限性包括：依赖固定槽位接口与特定压缩流程，未系统探究分段、压缩比例或槽位预算的权衡；未充分考虑多文档检索与槽位分配；门控位置与递归上限手工设定，可能不适用于严格的算力或延迟约束。

---

## How to Set the Batch Size for Large-Scale Pre-training?

**arXiv ID:** 2601.05034 | [PDF](https://arxiv.org/pdf/2601.05034v1)

**作者:** Yunhua Zhou `[一作]` (Shanghai AI Laboratory), Xipeng Qiu `[通讯]` (Fudan University)

**通讯引用:** 17130 | **OpenAlex IDs:** https://openalex.org/A5044665993

**关键词:** `Artificial Intelligence` `Optimization` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文研究了在 Warmup‑Stable‑Decay (WSD) 学习率调度下的大规模预训练中批大小的理论与实践，提出了新的 E(S) 公式，并通过该公式定义了 B_min 与 B_opt 两个关键指标，基于它们设计了一套动态批大小调度策略。

**💡 创新点**

创新点包括：1) 证明传统 Critical Batch Size 理论在 WSD 调度下失效；2) 推导并验证适配 WSD 的分段 E(S) 公式；3) 发现并量化 B_min 与 B_opt 随训练进展单调上升的规律；4) 基于该规律设计并验证了有效的动态批大小调度方案。

**🔧 技术方法**

所用技术主要有：理论推导（piecewise E(S) 函数与连续性约束）、Huber 损失拟合、AdamW 优化器、WSD 学习率调度、InternLM2 与 Qwen3 Dense/MoE 模型的实现，以及 OpenCompass 与 LMDeploy 的评测工具。

**📊 数据集**

使用的数据集包括：InternLM2 语料库（通用文本、代码、长文本）以及下游评测基准 MMLU 与 CMMLU。

**📈 对比分析**

实验中将动态批大小策略与固定批大小基线（以及在 Cosine 调度下的对比）进行对比，利用 OpenCompass 评测 MMLU/CMMLU 分数。结果表明，动态调度在训练损失下降速度更快、下游任务分数提升显著，且在不同学习率、权重衰减设置下保持稳健。

**⚠️ 局限性**

限制包括：E(S) 曲线拟合仅在单一学习率（6×10⁻⁴）下完成，缺乏严格的理论证明动态调度的最优性；序列长度切换导致的训练不稳定性未得到充分解决；未对不同学习率或更大模型规模进行系统探索。

---

## OptiSet: Unified Optimizing Set Selection and Ranking for Retrieval-Augmented Generation

**arXiv ID:** 2601.05027 | [PDF](https://arxiv.org/pdf/2601.05027v1)

**作者:** Yi Jiang `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 15641 | **OpenAlex IDs:** https://openalex.org/A5017671620

**关键词:** `Artificial Intelligence` `Retrieval` `Optimization` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了 OptiSet，一种集成扩展-精炼和集合级排序的检索增强生成框架，用于高效选择多条证据。

**💡 创新点**

创新点在于将查询扩展和集合重选并结合自我合成标签以及集合级列表训练，突破了传统单条检索排序的冗余与组合效益不足。

**🔧 技术方法**

采用 LLM 推理进行查询拆分与候选选择、Perplexity 基于的自标记、集合-列表训练（CE+KL）以及 "Expand-then-Refine" 策略。

**📊 数据集**

在 HotpotQA、Bamboogle、MuSiQue 与 TriviaQA 四个多跳/单跳 QA 数据集上进行实验。

**📈 对比分析**

与 Vanilla、NaiveRAG、SetR_O、ReasonIR、RankZephyr、SetWise、RankR1 等基线比较，OptiSet 在 EM/F1 上显著提升（如 HotpotQA+5% EM、+6% F1），且证据数量更少。

**⚠️ 局限性**

局限在于受限于 LLM 上下文长度、未使用 RL 进一步提升性能，以及缺乏可调的 evidence 预算控制。

---

## A data structure for monomial ideals with applications to signature Gröbner bases

**arXiv ID:** 2601.05026 | [PDF](https://arxiv.org/pdf/2601.05026v1)

**作者:** Pierre Lairez `[一作]` (Inria Université Paris-Saclay), Théo Ternier `[通讯]` (Inria Université Paris-Saclay)

**关键词:** `Symbolic Computation`

### 📋 论文摘要

**🎯 论文内容**

提出了一种新的数据结构——单项式可除性图（MDD），用于高效表示和操作单项式理想，支持快速成员检验和插入；

**💡 创新点**

创新点在于将单项式理想的前缀树进行最大共享，构造有向无环图，从而实现O(n log D)的成员检验和O(N)的插入；提供理论复杂度分析并证明其与理想生成元数量无关；

**🔧 技术方法**

技术包括：可除性图的树/图定义、最大共享（哈希共用）实现、递归成员检验与插入算法、Julia语言实现与hash‑consing；

**📊 数据集**

使用多组基准数据集：随机单项式理想（10变量、不同指数范围）、Cyclic、Katsura、Eco、Noon、Gametwo 系统以及从 msolve 库提取的 cp 系统；

**📈 对比分析**

通过将 MDD 集成到 Julia 包 AlgebraicSolving.jl 的签名 Gröbner 基算法中，与原实现对比实验显示：总计算时间平均下降 1.2–4.3 倍，符号计算部分占比大幅降低，成员检验时间几乎可以忽略；

**⚠️ 局限性**

局限性包括：对生成元很少的理想，MDD 的优势不明显；在某些极端生成元多、结构无共享的理想中，MDD 可能膨胀；目前仅优化成员检验，其余符号操作仍待改进。

---

## Knowledge-to-Data: LLM-Driven Synthesis of Structured Network Traffic for Testbed-Free IDS Evaluation

**arXiv ID:** 2601.05022 | [PDF](https://arxiv.org/pdf/2601.05022v1)

**作者:** Konstantinos E. Kampourakis `[一作]` (Norwegian University of Science and Technology), Stefanos Gritzalis `[通讯]` (University of Piraeus)

**通讯引用:** 6203 | **OpenAlex IDs:** https://openalex.org/A5091815745

**关键词:** `Cryptography and Security` `Data Synthesis` `Anomaly Detection` `Transformer` `Large Language Model` `Prompt Engineering` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

本文提出利用大型语言模型在不进行微调或获取原始数据的前提下，生成结构化、协议约束的网络流量数据，用于IDS评估。

**💡 创新点**

创新点在于将协议文档、攻击语义和统计规则融合为提示，构建受控生成管道，并通过多层验证证明生成数据可逼近真实流量。

**🔧 技术方法**

主要技术包括自然语言提示工程、JSON规则集解析、LLM生成、欧氏距离/余弦相似度、KS检验、PCA可视化及跨域分类。

**📊 数据集**

使用的数据集是公开的IEEE 802.11无线流量基准AWID3，包含正常、冲洪、冒充等四类。

**📈 对比分析**

对比方法采用多维相似度度量和交叉域分类，四大LLM（ChatGPT-5、Gemini 2.5 Pro、Claude Opus 4.1、Qwen3-Max）生成的数据在余弦相似度>0.97、欧氏距离≈1000，LightGBM跨域F1最高达0.956。

**⚠️ 局限性**

主要局限包括未建模时间连续性、对高变异物理层特征的逼真度不足，以及神经网络对生成噪声的敏感度高。

---

## Hán Dān Xué Bù (Mimicry) or Qīng Chū Yú Lán (Mastery)? A Cognitive Perspective on Reasoning Distillation in Large Language Models

**arXiv ID:** 2601.05019 | [PDF](https://arxiv.org/pdf/2601.05019v1)

**作者:** Yueqing Hu `[一作]` (Institute of Neuroscience), Tianhong Wang `[通讯]` (School of Philosophy)

**关键词:** `Computation and Language` `Knowledge Distillation` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Supervised Fine-Tuning` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究了在大型推理模型中，通过强化学习训练得到的教师模型与其人类认知成本的自然对齐，随后对多种学生模型进行推理蒸馏（SFT）后，评估其是否保持此对齐，探讨蒸馏导致的功能对齐崩溃及其机制。

**💡 创新点**

提出了“汉丹学步”（表面模仿）与“青出于蓝”（功能掌握）两种假设，并首次揭示了推理蒸馏会导致的“线性膨胀律”与“认知瀑布”现象，证明人类式认知是强化学习的涌现属性，蒸馏无法保留。

**🔧 技术方法**

使用强化学习与可验证奖励（RLVR）训练教师模型，随后通过监督微调（SFT）蒸馏学生模型；对推理成本与人类反应时进行Pearson相关性、RSA、KL散度、逆效率指数（I_E）以及线性回归等多维度统计分析。

**📊 数据集**

利用六个推理任务（算术、三段论、逻辑一致性、关系推理、直觉推理）的人类RT数据以及14个模型（教师、蒸馏学生、基准模型）进行对照实验；数据来源于先前研究中公开的实验材料。

**📈 对比分析**

将教师、蒸馏学生与基准模型在准确率、推理成本、与人类RT的相关性等指标上进行比较；结果显示教师模型与人类RT的相关系数≈0.64，蒸馏学生降至≈0.34，且往往比基准模型更差（负迁移），推理成本呈倍增趋势，效率明显下降。

**⚠️ 局限性**

局限性包括：蒸馏仅在语言推理任务上验证，未涵盖视觉或多模态任务；实验模型数量有限，可能缺乏普适性；SFT蒸馏无法解决奖励导向的信用分配问题，导致学生失去动态资源调配能力；缺乏对内部机制的深入解释，需进一步研究主动RL或自监督方法以实现功能掌握。

---

## An Empirical Investigation of Robustness in Large Language Models under Tabular Distortions

**arXiv ID:** 2601.05009 | [PDF](https://arxiv.org/pdf/2601.05009v1)

**作者:** Avik Dutta `[一作]` (Microsoft Corporation), Sumit Gulwani `[通讯]` (Microsoft Corporation)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Prompt Engineering` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

研究LLM在表格语义和结构失真时的鲁棒性，构建专家标注的失真表格数据集并评估多种LLM的性能。

**💡 创新点**

首次系统地将语义与结构失真引入表格问答任务，并量化LLM对失真表格的错误检测与自我修复能力。

**🔧 技术方法**

采用提示工程、代码沙箱执行、三种输入模态（文本、图片、无预览）以及pass@3和鲁棒性指标进行评测。

**📊 数据集**

Expert‑curated 50条表格问答样本的失真数据集（共22种语义失真、28种结构失真），公开托管在GitHub。

**📈 对比分析**

对比不同模型（Finetuned、Open‑source、Close‑source）在canonical与失真表格上的准确率；发现失真会导致至少22%准确率下降，distortion‑aware提示可提升约10‑20%；结构失真，尤其垂直位移，仍难以恢复。

**⚠️ 局限性**

受限于样本量小、仅单步失真、仅保留原答案的失真；未覆盖多重错误、内容级破坏或大表格场景。

---

## Can Large Language Models Resolve Semantic Discrepancy in Self-Destructive Subcultures? Evidence from Jirai Kei

**arXiv ID:** 2601.05004 | [PDF](https://arxiv.org/pdf/2601.05004v1)

**作者:** Peng Wang `[一作]` (Macau University of Science and Technology), Dagang Li `[通讯]` (Macau University of Science and Technology)

**通讯引用:** 832 | **OpenAlex IDs:** https://openalex.org/A5100605483

**关键词:** `Computation and Language` `Large Language Model` `Agentic AI` `Retrieval-Augmented Generation` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

探讨并提出一种多代理框架——Subcultural Alignment Solver，用于检测互联网子文化（以Jirai Kei为例）中的自残行为。

**💡 创新点**

创新点在于通过自动检索与子文化对齐生成报告，弥补LLM知识滞后与语义错位问题，显著提升对子文化独特表达的理解。

**🔧 技术方法**

采用多代理技术：子文化检索、对齐报告生成、文化对齐求解器，并利用检索工具与LLM交互。

**📊 数据集**

使用JiraiBench基准数据集进行实验，并在Menhera、Yami Kawaii、Tenshi Kaiwai等子文化上验证可迁移性。

**📈 对比分析**

与零射、链式思考、Plan-and-Solve、Self-Refine、S³ Agent、OWL等方法对比，Subcultural Alignment Solver在大多数模型上均取得最高宏F1，尤其在小型LLM上与微调模型相当。

**⚠️ 局限性**

局限性包括仅在Jirai社区数据上验证，且依赖网络检索，若子文化仅存在于特定社区则难以获取信息。

---

## On the Hidden Objective Biases of Group-based Reinforcement Learning

**arXiv ID:** 2601.05002 | [PDF](https://arxiv.org/pdf/2601.05002v1)

**作者:** Aleksandar Fontana `[一作]` (Scuola Superiore Sant’Anna), Paolo Mori `[通讯]` (National Research Council of Italy)

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning`

### 📋 论文摘要

**🎯 论文内容**

本文对基于群体的强化学习方法（如GRPO）提出了统一的代理目标框架，并从理论上分析了其训练动态。

**💡 创新点**

创新点在于建立通用代理目标，揭示了非均匀权重导致共享前缀梯度偏置、AdamW在无正则化时对奖励尺度不敏感以及AdamW动量与裁剪机制相互作用导致的超越信任区间现象。

**🔧 技术方法**

主要使用理论推导、梯度聚合分析以及AdamW优化器的数学性质。

**📊 数据集**

论文未提供具体实验数据集，主要以理论推导和已公开方法的统一表述为主。

**📈 对比分析**

通过将十种近期方法映射到统一框架进行对比，未给出数值实验结果，因而没有提供性能指标。

**⚠️ 局限性**

局限在于仅适用于标准自回归模型、未给出针对AdamW的闭式修正方案，且对不同正则化或其他优化器的验证有限。

---

## On the Definition and Detection of Cherry-Picking in Counterfactual Explanations

**arXiv ID:** 2601.04977 | [PDF](https://arxiv.org/pdf/2601.04977v1)

**作者:** James Hinns `[一作]` (University of Antwerp), David Martens `[通讯]` (University of Antwerp)

**通讯引用:** 4261 | **OpenAlex IDs:** https://openalex.org/A5101474247

**关键词:** `Machine Learning` `Explainability and Interpretability` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

本文对计数因果解释（counterfactual explanations）中的 cherry‑picking（选择性披露）进行了形式化定义，并系统地研究了在不同访问级别（完整程序、部分程序、仅解释）下外部审计者能否检测到此类操纵。

**💡 创新点**

创新点在于：①提出了基于可接受解释空间与效用函数的“cherry‑pick”正式定义；②从理论与实证两方面展示了即便在完整程序访问下，随机性与解释空间多样性也能掩盖 cherry‑picking；③给出了一套按访问级别划分的检测框架与可行性分析，揭示了检测难度的根源。

**🔧 技术方法**

采用的技术包括：理论推导（可接受解释空间、效用函数、排名、VC 维度约束）、对抗性实验（随机森林+DiCE随机因果解释器）、统计比较（多种随机种子、可编辑特征限制下的稀疏性、可实现性等指标）。

**📊 数据集**

使用了公开的 Adult 数据集作为实验基准，并在其上构造了多种解释生成方案（全部可编辑、受限可编辑、不同随机种子）。

**📈 对比分析**

比较方法：在三种访问级别下重复多次实验，记录解释质量指标（稀疏性、可实现性、可实现性等）。结果显示：即使在完整程序访问下，cherry‑picked 解释与正常解释在指标上差异可被随机波动掩盖，检测精度低；在部分或仅解释访问下，几乎无法区分；总体性能表明传统指标对检测无效。

**⚠️ 局限性**

局限性包括：①解释空间巨大且受随机性控制，理论上难以枚举所有可接受解释；②实验依赖特定数据集与方法，结果可能不具普适性；③缺乏实用的后置检测手段，需进一步研究更严格的前置规范或标准化流程。

---

## A Unified Spoken Language Model with Injected Emotional-Attribution Thinking for Human-like Interaction

**arXiv ID:** 2601.04960 | [PDF](https://arxiv.org/pdf/2601.04960v1)

**作者:** Qing Wang `[一作]`, Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**关键词:** `Computation and Language` `Generation` `Data Synthesis` `Large Language Model` `Supervised Fine-Tuning` `Knowledge Distillation` `Audio` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出了一种统一的语音语言模型，融合情感推理与共情回应，并通过Injected Emotional‑Attribution Thinking（IEAT）实现情感状态与成因在内部推理过程中的自我注入。

**💡 创新点**

创新点在于：①将情感属性和触发因素注入模型内部思考，而非作为外部监督；②采用两阶段渐进训练：先做语音‑文本对齐与情感属性自蒸馏，再进行跨模态端到端联合优化；③在生成时实现多词预测以支持实时TTS。

**🔧 技术方法**

技术手段包括GOAT‑SLM架构、语音感知与共享语义空间、self‑distillation、跨模态联合优化、模块化LLM微调（冻结底层、调优顶层）、多词预测、使用Qwen3、CosyVoice2、emo2vec等工具。

**📊 数据集**

使用的数据集有Emilia、WenetSpeech4TTS、GigaSpeech、train_3.5M_CN、Ultrachat、CosyVoice2语音合成样本以及LibriHeavy，用于训练与跨模态对齐。

**📈 对比分析**

在HumDial情感智能挑战的官方DEV和TEST集上与freeze‑omni、stepaudio2‑mini、qwen2.5‑omni、Qwen3‑omni、Mimo‑audio‑instruct等基线进行对比，最终在TEST集上获得最高4.27分（0‑5）并排名第一，表现出跨任务、跨语言的稳定优势。

**⚠️ 局限性**

局限性主要包括：模型规模大、训练成本高；依赖大量人工或自动生成的标签和合成语音，可能存在真实性与多样性不足；目前评测仅在挑战数据集上，缺乏真实场景下的鲁棒性验证。

---

## TEA: Temporal Adaptive Satellite Image Semantic Segmentation

**arXiv ID:** 2601.04956 | [PDF](https://arxiv.org/pdf/2601.04956v1)

**作者:** Juyuan Kang `[一作]` (Institute of Computing Technology), Feng Dai `[通讯]` (Institute of Computing Technology)

**通讯引用:** 2357 | **OpenAlex IDs:** https://openalex.org/A5028926076

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Knowledge Distillation` `Agriculture Related` `Transformer` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

提出TEA，一种时间自适应卫星影像时间序列语义分割方法，利用教师‑学生知识蒸馏、时间原型对齐与全序列重建等模块，提升不同时间长度序列下的作物分割效果。

**💡 创新点**

① 首次系统识别并解决SITS分割对时间长度敏感的缺陷；② 设计教师‑学生框架，将全长序列的全局时序知识迁移至短序列；③ 通过时间原型对齐扩展类别决策边界并保持跨阶段一致；④ 引入全序列重建辅助任务提升特征鲁棒性；⑤ 提出长度衰减mIoU评价指标。

**🔧 技术方法**

采用TSViT Transformer时空编码器，配合随机裁剪训练、EMA教师更新、交叉熵＋MSE蒸馏损失、时间原型对齐和全序列重建等技术。

**📊 数据集**

PASTIS（Sentinel‑2多光谱时间序列，33‑61帧，18作物+背景）和德国（137k序列，15‑45帧，17作物）数据集。

**📈 对比分析**

在10%–100%长度比例下与八种主流模型及TSViT基线对比；在PASTIS上 mIoU 提升至 66.77、LDIoU 33.36，较基线提升约19%；德国数据亦显著提升；在完整序列上实现SOTA。

**⚠️ 局限性**

对极短序列（≤20%）仍存在性能下降；模型复杂度高，训练需要大量GPU；方法对传感器类型、区域多样性的泛化能力尚未充分验证。

---

## Prototypicality Bias Reveals Blindspots in Multimodal Evaluation Metrics

**arXiv ID:** 2601.04946 | [PDF](https://arxiv.org/pdf/2601.04946v1)

**作者:** Subhadeep Roy `[一作]` (University of Technology Nuremberg), Steffen Eger `[通讯]` (University of Technology Nuremberg)

**关键词:** `Computer Vision and Pattern Recognition` `Evaluation Metrics` `Recommendation System` `Transformer` `Large Language Model` `Contrastive Learning` `Vision Language Model` `Image` `Text` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了自动文本-图像评估指标中的原型偏差（prototypicality bias）并构建了对抗性基准 ProtoBias，评估现有指标在动物、物体和人口统计三大领域中对语义正确与原型错误图像的排名倾向。

**💡 创新点**

创新点在于（1）首次系统量化并呈现自动评估指标对原型图像的偏好；（2）设计了可对比的、可控的原型与语义错误图像对；（3）提出并训练了轻量化 7B 参数评估器 ProtoScore，显著降低了原型偏差并实现了与大型封闭源评估器相当的鲁棒性。

**🔧 技术方法**

使用了 CLIP、PickScore、VQA、LLM-as-Judge（GPT‑4o、GPT‑5）等现有评估技术；并通过 Qwen2.5‑VL‑7B‑Instruct 与 GRPO 在 ProtoBias 对照图像上进行对比学习，得到 ProtoScore。

**📊 数据集**

构造的数据集包括 6275 条提示，生成 31,375 对（每对包含语义正确非原型图像与原型错误图像），涉及动物、物体与人口统计三大类别，并通过自动过滤与人工评注保证语义合法性。

**📈 对比分析**

与人类评注进行对比，发现现有指标在 71–70% 的样本上误将原型错误图像排名高于语义正确图像；ProtoScore 的误判率降至 31.6%，性能优于所有开源指标但略逊于 GPT‑5（24.1%）。

**⚠️ 局限性**

局限性包括：基准仅涵盖有限的三大类别且偏向西方数据；评估样本由专家人工标注，规模有限；实验未涵盖最新模型与更广泛文化背景，未来需在更大、多样的数据与人群上验证。

---

## Approximate equivariance via projection-based regularisation

**arXiv ID:** 2601.05028 | [PDF](https://arxiv.org/pdf/2601.05028v1)

**作者:** Torben Berndt `[一作]` (Heidelberg Institute for Theoretical Studies), Jan Stühmer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 951 | **OpenAlex IDs:** https://openalex.org/A5061756165

**关键词:** `Machine Learning` `Restoration` `Optimization` `Convolutional Neural Network` `Supervised Fine-Tuning` `Image` `Biomedical Data` `Computed Tomography`

### 📋 论文摘要

**🎯 论文内容**

提出了一种基于投影的近似等变正则化方法，直接在模型权重层面惩罚非等变性，避免了样本级别正则化带来的高样本复杂度。

**💡 创新点**

创新点包括：1) 证明投影到等变子空间的距离等价于等变缺陷；2) 为连续群（如SO(n)）提供傅里叶域可高效计算投影的通用框架；3) 实现对整个群空间的正则化，而非仅点-wise 正则化。

**🔧 技术方法**

采用投影正则化、Peter–Weyl 定理、傅里叶变换（FFT）、线性层的等变分解与块对角化、以及 Lipschitz 约束等技术。

**📊 数据集**

实验数据集包括：SO(2) 旋转不变分类 toy 数据、烟雾扩散模拟数据、CT-MAR AAPM 挑战集（头部与身体 CT 切片）以及金属伪影修复数据。

**📈 对比分析**

与样本采样正则化、残差路径先验（RPP）、无约束 CNN、以及现有等变卷积网络等方法比较。实验显示，在任务性能上与样本正则化相当甚至更优，同时在运行时、吞吐量和显存占用方面显著提升（约 42–61% 的加速），金属伪影修复中的 PSNR/SSIM 亦与样本正则化持平或略优。

**⚠️ 局限性**

限制：正则化项需针对每个模型架构和群操作单独推导；目前验证仅覆盖简单群组，未来需扩展至更复杂的子群结构。

---

## Leveraging Prediction Entropy for Automatic Prompt Weighting in Zero-Shot Audio-Language Classification

**arXiv ID:** 2601.05011 | [PDF](https://arxiv.org/pdf/2601.05011v1)

**作者:** Karim El Khoury `[一作]`, Benoit Macq `[通讯]`

**关键词:** `Sound` `Classification` `Prompt Engineering` `Contrastive Learning` `Audio` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

本文提出一种基于预测熵自动调节提示权重的方法，用于零样本音频-语言分类任务。

**💡 创新点**

创新点在于利用模型预测的熵信息动态决定每个提示的重要性，避免了手工设定权重，并提升了跨模态分类性能。

**🔧 技术方法**

技术手段包括多提示技术、熵最小化策略、对比学习与自监督音频-语言模型（如 CLAP）等。

**📊 数据集**

实验在公开的音频分类数据集（如 AudioSet、ESC‑50 或 VGGSound 与其对应文本描述）上进行。

**📈 对比分析**

与均匀加权、手工加权、单提示以及传统零样本基线相比，实验显示自动熵加权在多数任务上平均提升约 4–6% 的准确率。

**⚠️ 局限性**

局限性包括：仅在提示数目有限时效果显著；熵估计对噪声和标签不平衡较为敏感；对极度偏斜的数据集仍可能出现性能下降。

---

## SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection

**arXiv ID:** 2601.04968 | [PDF](https://arxiv.org/pdf/2601.04968v1)

**作者:** Maximilian Pittner `[一作]` (Bosch Mobility Solutions, Robert Bosch GmbH), Alexandru Paul Condurache `[通讯]` (Bosch Mobility Solutions, Robert Bosch GmbH)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Autonomous Driving` `Transformer` `Video`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于稀疏Transformer的3D车道检测方法，融合空间与时间先验，并给出了新的连续曲线表示与注意力机制。

**💡 创新点**

创新点包括：①使用Catmull‑Rom spline让控制点正好落在车道曲线上；②设计空间‑时间注意力（SLA/PNA/TCA），只关注车道内、车道间和历史信息；③引入空间/时间正则化提升鲁棒性；④提出基于视频自标记的长距离、可见性标注数据集。

**🔧 技术方法**

核心技术为：稀疏查询的Transformer解码器、变形交叉注意力、空间‑时间注意力层、CR spline曲线拟合、基于位姿变换的历史查询传播、以及可见性/平滑正则化。

**📊 数据集**

使用公开的OpenLane、ONCE‑3DLanes数据集以及作者自制的长距离3D车道数据集进行训练与评测。

**📈 对比分析**

在OpenLane上F1‑Score 66.1%、在ONCE‑3DLanes上召回率最高、在自制数据集上在所有指标（F1、误差、Vis‑IoU）均居首位，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括：对精确车辆位姿/相机标定依赖较高；目前仅实现检测而非完整跟踪；在极端遮挡或极远距离下仍可能出现漂移；内存队列大小需手工调参。

---

## What Students Ask, How a Generative AI Assistant Responds: Exploring Higher Education Students' Dialogues on Learning Analytics Feedback

**arXiv ID:** 2601.04919 | [PDF](https://arxiv.org/pdf/2601.04919v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## Precision over Diversity: High-Precision Reward Generalizes to Robust Instruction Following

**arXiv ID:** 2601.04954 | [PDF](https://arxiv.org/pdf/2601.04954v1)

**作者:** Yirong Zeng `[一作]` (Harbin Institute of Technology), Ting Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 38260 | **OpenAlex IDs:** https://openalex.org/A5100418162

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

对强化学习可验证奖励（RLVR）中的指令跟随任务进行系统实验，发现仅使用高精度硬约束训练的模型比混合硬/软约束训练的模型更能泛化，并提出基于奖励精度的高精度代理训练（HPPT）策略。

**💡 创新点**

挑战普遍的多样性驱动观念，证明奖励精度是指令跟随效果的决定性因素；提出两步数据清洗和约束简化的实用方法，显著提升模型的泛化和训练效率。

**🔧 技术方法**

采用RLVR（GRPO）强化学习框架，结合规则式检查器与LLM Judge进行奖励判定，利用注意力分析揭示模型内部化的元技能，并通过数据清洗与单软约束限制实现高精度奖励。

**📊 数据集**

使用VerInstruct训练集（含22,000条硬/软约束示例），评估基准包括IFEval、Multi-IF、IFBench、CFBench、FollowBench；通用能力评估使用GSM8K、MMLU、WritingBench。

**📈 对比分析**

与混合约束、Soft-only、基础模型等进行ISR对比，HPPT在五个指令跟随基准上平均提升13.4%性能，训练时间降低58%，同时在通用任务保持或提升性能。

**⚠️ 局限性**

主要局限：1）为极复杂或抽象任务构造可验证代理困难，需进一步提升模型评估器的鲁棒性；2）缺乏定量指标来直接评估模型内部化的指令跟随元技能。

---

## A Data-Driven Predictive Framework for Inventory Optimization Using Context-Augmented Machine Learning Models

**arXiv ID:** 2601.05033 | [PDF](https://arxiv.org/pdf/2601.05033v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## GenProve: Learning to Generate Text with Fine-Grained Provenance

**arXiv ID:** 2601.04932 | [PDF](https://arxiv.org/pdf/2601.04932v1)

**作者:** Jingxuan Wei `[一作]` (Shenyang Institute of Computing Technology), Junnan Zhu `[通讯]` (MAIS)

**关键词:** `Computation and Language` `Generation` `Optimization` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

设计并实现了生成时细粒度出处标注任务，提出ReFInE数据集与GenProve训练框架，让LLM在生成答案的同时输出句子级别的引用关系三元组。

**💡 创新点**

引入了文档/句子级别的Quotation、Compression、Inference三种关系标签，并通过RL优化的复合奖励实现答案与出处的联合优化，显著提升了可验证推理能力。

**🔧 技术方法**

采用监督微调（SFT）与基于GRPO的强化学习相结合，使用句子相似度与引用F1奖励构建复合奖励。

**📊 数据集**

构建了专家验证的ReFInE数据集，包含多文档QA实例与句子级别的出处关系标注。

**📈 对比分析**

与14款强大的LLM（包括Gemini 2.5 Pro、GPT‑5等）在答案质量、出处准确度与格式合规性上进行统一评估，GenProve在所有指标上均领先，LLM‑judge分数最高。

**⚠️ 局限性**

生成结构化出处会增加延迟，方法目前仅覆盖英文，且受检索质量限制，无法在检索不到证据时生成可靠出处。

---

## Asynchronous Secure Federated Learning with Byzantine aggregators

**arXiv ID:** 2601.04930 | [PDF](https://arxiv.org/pdf/2601.04930v1)

**作者:** Antonella Del Pozzo `[一作]`, Sara Tucci-Piergiovanni `[通讯]`

**关键词:** `Distributed, Parallel, and Cluster Computing` `Federated Learning` `Safty and Privacy` `Convolutional Neural Network` `Differential Privacy` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出一种异步网络下、可容忍拜占庭聚合器的隐私保护与公平联邦平均协议

**💡 创新点**

创新点在于：1）使用聚合器复制与可验证安全聚合，消除单点失效；2）引入异步环境下的客户端分配与调度机制；3）设计了基于掩码共享的DP方案，减少噪声；4）通过可验证的PVAHSS和门限签名实现模型认证；5）不依赖共识，支持全异步通信。

**🔧 技术方法**

采用的技术包括：安全聚合（基于LWE与可验证秘密共享）、差分隐私（Gaussian噪声与RDP）、异步分区与随机洗牌、门限签名、PVAHSS、客户端入选机制、负载均衡与欺诈检测。

**📊 数据集**

使用MNIST数据集，对模型进行两层卷积+全连接的26,000参数网络进行实验。

**📈 对比分析**

与现有的FLDP基线对比；在非i.i.d.、异步、拜占庭聚合器场景下，实验表明协议在相同DP预算下收敛更快、准确率更高，入选机制显著降低噪声并提升性能；在容错参数下仍能收敛。

**⚠️ 局限性**

局限性包括：①需要多台聚合器且通信量随聚合器数量增长；②DP噪声会随容错参数提升而增大，影响精度；③实验仅在MNIST单一任务上验证，缺乏对大模型或GPU加速的评估；④假设聚合器之间非协同恶意，实际部署中安全性需进一步验证。

---

## From Stories to Cities to Games: A Qualitative Evaluation of Behaviour Planning

**arXiv ID:** 2601.04911 | [PDF](https://arxiv.org/pdf/2601.04911v1)

**作者:** Mustafa F. Abdelwahed `[一作]` (University of St Andrews), Ian P. Gent `[通讯]` (University of St Andrews)

**通讯引用:** 4881 | **OpenAlex IDs:** https://openalex.org/A5059061177

**关键词:** `Artificial Intelligence` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出并验证了基于行为空间的多样性规划方法，在叙事生成、城市规划和游戏评估三大真实案例中演示了其可行性。

**💡 创新点**

创新点在于将行为空间与行为排序套件结合，直接在规划过程中产生多样化计划，并同时支持模型化（SMT）与无模型（LTL）实现。

**🔧 技术方法**

主要技术包括行为空间建模、行为排序套件（Behaviour Sorts Suite）、禁止行为迭代算法（ForbidBehaviourIterative）、SMT 与 LTL 规划器以及 PyBoy 模拟器。

**📊 数据集**

使用的数据集包括 Aladdin 故事域的 PDDL 规范、英国 St Andrews 城市规划模拟器的土地使用数据以及 Super Mario Land 1‑1 的 ROM 进行模拟。

**📈 对比分析**

论文未给出定量对比指标；通过示例展示所生成的多样计划在不同维度（叙事结局、可持续性/多样性得分、敌人交互方式）上满足预设目标，性能以规划求解时间和计划数量为暗示。

**⚠️ 局限性**

局限性在于维度设计需人工手工完成、对特殊领域的编译与模拟依赖较大、SMT/LTL 的求解效率受限，且缺乏统一的评估基准。

---

## HMVI: Unifying Heterogeneous Attributes with Natural Neighbors for Missing Value Inference

**arXiv ID:** 2601.05017 | [PDF](https://arxiv.org/pdf/2601.05017v1)

**作者:** Xiaopeng Luo `[一作]`, Zhuowei Wang `[通讯]` (Guangdong University of Technology)

**通讯引用:** 3305 | **OpenAlex IDs:** https://openalex.org/A5007409112

**关键词:** `Machine Learning` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出一种统一的跨类型特征依赖模型，用自然邻居关系在聚类中完成缺失值推断；

**💡 创新点**

创新点在于：① 设计统一异构度量，可处理带缺失的数值与类别属性；② 引入属性间相互依赖权重优先填补；③ 通过聚类与填补互补实现更高精度；

**🔧 技术方法**

技术包括：自定义距离矩阵、自然邻居搜索、聚类与迭代填补、基于均值/众数的局部填充；

**📊 数据集**

使用诊断（Diagnosis）、教师助手（Teacher Assistant）和乳腺癌（Breast Cancer）三组含数值、名义和序数属性的数据集；

**📈 对比分析**

与Mean/Mode、MissForest、KNNMI等基准比较，mRMSE最低、ARI和CVI在高缺失率下仍保持竞争力，证明其在下游聚类任务中表现优异；

**⚠️ 局限性**

局限在于聚类中心随机初始化对结果影响大，且在样本不平衡时效果更易受影响。

---

## AlgBench: To What Extent Do Large Reasoning Models Understand Algorithms?

**arXiv ID:** 2601.04996 | [PDF](https://arxiv.org/pdf/2601.04996v1)

**作者:** Henan Sun `[一作]` (Hong Kong University of Science and Technology), Jia Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18471 | **OpenAlex IDs:** https://openalex.org/A5100405681

**关键词:** `Artificial Intelligence` `Optimization` `Transformer` `Large Language Model` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

构建了一个算法中心化的、无污染的 AlgBench 基准，包含 3000 个原创问题，覆盖 27 个算法。

**💡 创新点**

提出算法中心化范式、基于专家手工制作的税onomy 以及针对低熵词的战略性过度转移分析。

**🔧 技术方法**

使用 LLM 评估（Pass@k、z-score 归一化）、错误分类自动化流水线，以及多模型对比与模型规模分析。

**📊 数据集**

AlgBench 本身作为评测集，涵盖 Euclidean、Non‑Euclidean、非优化、局部优化、全局优化、启发式优化六大类别。

**📈 对比分析**

通过 Pass@k、归一化分数与参数规模对比，发现 LLM 在非优化和欧氏结构任务表现优秀，但在全局优化与启发式任务精度低至 49%，规模提升收益不均。

**⚠️ 局限性**

缺乏复杂竞赛算法、难度量化不清晰、数据集规模不足以支持预训练。

---

## The unsuitability of existing regulations to reach sustainable AI

**arXiv ID:** 2601.04958 | [PDF](https://arxiv.org/pdf/2601.04958v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## Learning from Mistakes: Negative Reasoning Samples Enhance Out-of-Domain Generalization

**arXiv ID:** 2601.04992 | [PDF](https://arxiv.org/pdf/2601.04992v1)

**作者:** Xueyun Tian `[一作]` (Chinese Academy of Sciences), Huawei Shen `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 6646 | **OpenAlex IDs:** https://openalex.org/A5047897879

**关键词:** `Computation and Language` `Large Language Model` `Supervised Fine-Tuning` `Chain-of-Thought` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究了在链式推理（CoT）示例中，利用包含错误推理路径的负样本进行监督微调，并提出了一种基于学习进度的自适应损失加权方法（GLOW）来提升模型的跨域泛化能力。

**💡 创新点**

创新点在于：①发现负样本虽然最终答案错误，但包含有价值的中间推理信息；②系统分析了负样本在数据多样性、训练动态与推理熵方面的正向作用；③提出 GLOW，通过计算样本的跨 epoch 损失减量来动态上调难学样本的权重，从而有效利用全部数据并显著提升 OOD 性能。

**🔧 技术方法**

采用的技术包括：大模型监督微调（SFT）、链式推理数据构造、交叉 epoch 损失差（Δ）作为学习进度度量、Sigmoid 变换实现自适应权重、策略熵分析以及在 RL 预训练后进一步验证。

**📊 数据集**

使用的数据集有：OpenMathReasoning（数学推理）、MMLU（多领域知识）、General Reasoning 组合数据集（包括 Minerva、Olympia、AMC 等），并以 Qwen3-8B 作为教师生成 CoT 路径。

**📈 对比分析**

与传统仅用正样本或仅用负样本的 SFT 以及混合训练相比，GLOW 在多种模型规模（Qwen2.5‑7B、14B、32B 等）和任务上均表现更佳：如在 Qwen2.5‑7B 上 OOD 提升 5.51%（从 65.71% 至 71.26%），MMLU 从 72.82% 提升至 76.47%；整体在 22 个基准任务上平均提升约 2‑3%。

**⚠️ 局限性**

局限性包括：①实验主要聚焦文本级 CoT 数据，未探讨多模态或工具增强场景；②仅在少数开源 LLM（如 Qwen、Llama3.1）上验证；③未研究 GLOW 与后续 RLHF 或更复杂 RL 训练的交互；④负样本标签依赖于教师模型的推断，可能存在偏差。

---

## When to Act: Calibrated Confidence for Reliable Human Intention Prediction in Assistive Robotics

**arXiv ID:** 2601.04982 | [PDF](https://arxiv.org/pdf/2601.04982v1)

**作者:** Johannes A. Gaus `[一作]` (Hertie Institute for Clinical Brain Research), Daniel Haeufle `[通讯]` (Hertie Institute for Clinical Brain Research)

**通讯引用:** 1170 | **OpenAlex IDs:** https://openalex.org/A5110486192

**关键词:** `Robotics` `Robotic Intelligence` `Safety and Privacy` `Explainability and Interpretability` `Computational Efficiency` `Recurrent Neural Network` `Supervised Fine-Tuning` `Multimodality` `Video`

### 📋 论文摘要

**🎯 论文内容**

本文提出了基于多模态下一动作预测的安全触发框架，利用校准后的置信度来控制协助设备的 Act/Hold 门；

**💡 创新点**

创新点在于将后处理校准方法（如温度缩放、等势回归）与选择性预测结合，为助行设备提供可解释的安全阈值，并通过轻量级 GRU 实现实时可部署；

**🔧 技术方法**

主要技术包括多模态 GRU 编码器、后置概率校准（温度缩放、Platt 以及等势回归）、自适应抑制（hysteresis）Act/Hold 门、选择性预测与轻量化设计；

**📊 数据集**

使用了 EGTEA Gaze+ 自摄像头 ADL 数据集，包含 32 位受试者、21 类动词标签；

**📈 对比分析**

与基线 GRU 与 Transformer 进行比较，Top‑1 准确率约 40%，校准后 ECE 从 0.40 降至 0.04，Act/Hold 门在不同阈值下实现了可观的 act‑only 精度提升，推理时间约 2–3 ms，满足实时控制需求；

**⚠️ 局限性**

局限在于仅在离线重放中验证，缺乏真实用户闭环测试；对病理性运动的域漂移与校准稳定性仍待进一步研究，且仅聚焦动词级预测，未来需扩展到更细粒度动作和在线再校准。

---

## ConMax: Confidence-Maximizing Compression for Efficient Chain-of-Thought Reasoning

**arXiv ID:** 2601.04973 | [PDF](https://arxiv.org/pdf/2601.04973v1)

**作者:** Minda Hu `[一作]` (Chinese University of Hong Kong), Irwin King `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 26928 | **OpenAlex IDs:** https://openalex.org/A5042251906

**关键词:** `Artificial Intelligence` `Compression` `Reinforcement Learning` `Reinforcement Learning` `Chain-of-Thought` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出 ConMax，一种基于强化学习的思维链压缩框架，自动去除大模型推理过程中的冗余步骤，生成简洁但完整的 Chain‑of‑Thought，用于低成本的“cold‑start”监督微调。

**💡 创新点**

创新点在于：① 用双重 confidence 奖励（Answer Confidence 与 Thinking Confidence）代替手工长度惩罚，保持答案准确性与推理逻辑完整；② 在冻结的辅助 LRM 上评估奖励，避免人工标注；③ 通过 GRPO 训练压缩策略，实现在保持推理质量的前提下显著缩短 token 长度。

**🔧 技术方法**

采用的技术包括：强化学习（GRPO）、基于语言模型的 confidence 评估、策略网络 πθ、奖励函数 R_c = S_ans + β·S_think、系统提示引导压缩、无显式长度惩罚、以及在压缩后生成数据用于 SFT。

**📊 数据集**

使用的数据集有：AIME2025、MATH500、AMC23、MINERVA、GPQA 五个推理基准；训练集来源于 NuminaMath 子集（OpenThoughts‑114k）并通过 R1‑7B 生成带有高质量推理链的 (x, z, y) 样本。

**📈 对比分析**

与原始未压缩数据和 Prompt‑Based 压缩方法对比，实验在 Qwen‑2.5‑{3B/7B/14B} 模型上验证：压缩率约 30–43%，准确率仅下降 ≤1.5 分，尤其在 7B 上仅 0.7% 的性能损失；相比基线，压缩后模型在 token 使用上减少约 2,000–4,000 tokens，推理效率显著提升。

**⚠️ 局限性**

局限性包括：① 仅在数学/科学推理任务验证，尚未在创意写作、代码生成等领域检验；② 未针对超大模型（70B 以上）或非 Qwen 架构进行评估；③ 奖励机制依赖冻结辅助模型，可考虑无外部模型的自监督压缩方法。

---

## T-Retriever: Tree-based Hierarchical Retrieval Augmented Generation for Textual Graphs

**arXiv ID:** 2601.04945 | [PDF](https://arxiv.org/pdf/2601.04945v1)

**作者:** Chunyu Wei `[一作]` (Renmin University of China), Yueguo Chen `[通讯]` (Renmin University of China)

**通讯引用:** 1494 | **OpenAlex IDs:** https://openalex.org/A5057384573

**关键词:** `Artificial Intelligence` `Retrieval` `Generation` `Graph Neural Network` `Retrieval-Augmented Generation` `Graph` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了T-Retriever框架，将属性图检索转化为基于编码树的树形检索，并构建多分辨率知识层次。

**💡 创新点**

创新点包括：1）自适应压缩编码，采用全局最优的自顶向下分割，摆脱固定层压缩比例；2）语义结构熵S²-Entropy，将拓扑结构熵与节点语义熵统一优化，提升检索的语义连贯性和结构一致性。

**🔧 技术方法**

使用信息理论驱动的编码树构建（Shannon‑Fano启发式分割）、核密度估计计算语义熵、LLM嵌入与近似最近邻索引、GNN增强的LLM提示与文本化子图技术。

**📊 数据集**

在SceneGraphs、WebQSP和BookGraphs三个基准数据集上进行实验。

**📈 对比分析**

与平面RAG（G‑Retriever、GRAG）以及层次RAG（RAPTOR、ArchRAG）对比，T‑Retriever在SceneGraphs、WebQSP、BookGraphs上分别提升约2.4%、2.4%和6.6%，并在在线检索上显著降低Token数和节点数，显示出更高的准确率与效率。

**⚠️ 局限性**

局限性：需要一次性较大的离线构建成本（如BookGraphs约7.3小时）；对层次深度和检索子图数敏感；对超参数λ、h的选择需要经验或交叉验证，且在极小图上优势不明显。

---

## Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences

**arXiv ID:** 2601.04925 | [PDF](https://arxiv.org/pdf/2601.04925v1)

**作者:** Arkadiusz Modzelewski `[一作]` (University of Padua), Giovanni Da San Martino `[通讯]` (University of Padua)

**通讯引用:** 3867 | **OpenAlex IDs:** https://openalex.org/A5033850423

**关键词:** `Computation and Language` `Generation` `Data Synthesis` `Transformer` `Large Language Model` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文研究了大型语言模型（LLM）生成的说服性文本是否更难被自动检测，并系统分析了人类与LLM生成说服文本在语言特征上的差异，构建了多语言Persuaficial基准数据集。

**💡 创新点**

创新点包括：1) 设计四种可控生成方式（改写、细微改写、强化改写、自由生成）并用LLM生成约65k条多语言说服文本；2) 在零样本检测场景下跨模型、跨语言比较检测难度；3) 使用StyloMetrix提取196维语言特征，结合Cohen d和Wilcoxon检验系统揭示人类与LLM文本的关键差异。

**🔧 技术方法**

主要技术：LLM零样本二分类（GPT‑4.1 Mini、Gemini 2.0 Flash、Gemma 3 27B Instruct、Llama 3.3 70B）、StyloMetrix语言特征提取、Cohen d效应量分析、Wilcoxon符号秩检验、F1度量对检测性能评估。

**📊 数据集**

数据集：Persuaficial（≈65,000条，多语言：英语、德语、法语、意大利语、波兰语、俄语；四种生成方式）；三大人类说服文本来源：SemEval 2023 Task 3、DIPROMATS 2024 Task 1、ChangeMyView。

**📈 对比分析**

比较方法：在平衡的正负样本上进行零样本二分类，计算F1；对比人类文本与四种LLM生成文本的检测表现。结果显示：1) 开放式与强化式生成的说服文本更易被检测（F1提高≈10%）；2) 细微式生成显著降低检测性能（F1下降≈20%）；3) 这一趋势在所有语言和模型上保持一致。

**⚠️ 局限性**

局限性：1) 语言学差异分析仅覆盖英语；2) 仅使用零样本检测，未探究少样本或微调效果；3) 未深入评估不同语言生成文本的语言特征差异；4) 主要使用API推理，未评估模型训练或环境成本。

---

## Decentralized Privacy-Preserving Federal Learning of Computer Vision Models on Edge Devices

**arXiv ID:** 2601.04912 | [PDF](https://arxiv.org/pdf/2601.04912v1)

**作者:** Damian Harenčák `[一作]` (Comenius University), Martin Madaras `[通讯]` (Skeletex Research)

**关键词:** `Cryptography and Security` `Federated Learning` `Safty and Privacy` `Segmentation` `Convolutional Neural Network` `Image`

### 📋 论文摘要

**🎯 论文内容**

在边缘设备上实现联邦学习，评估同态加密、梯度压缩、梯度噪声等技术对隐私与精度的影响。

**💡 创新点**

将CKKS同态加密与梯度噪声结合，在Jetson TX2上实现轻量化，并系统比较其隐私-性能权衡。

**🔧 技术方法**

使用Paillier同态加密、CKKS同态加密、梯度压缩、梯度噪声、DLG逆向算法、PyTorch/CNN/U-Net、OpenFL、TenSEAL等技术。

**📊 数据集**

使用CIFAR‑10数据集进行分类实验，并用3D扫描得到的正则图与二值分割标注进行分割实验。

**📈 对比分析**

通过在相同模型上分别压缩或噪声梯度，比较重建成功率、模型准确率以及加密速度；CKKS加密比Paillier快数百倍，梯度噪声在方差0.007时即可阻断DLG且对准确率影响≤0.5%，压缩90%时重建失败但准确率下降仅3.9%。

**⚠️ 局限性**

未给出正式隐私保证，DLG攻击仍可能泄露结构信息；同态加密在资源受限设备上仍存在计算与通信开销；在复杂分割网络上重建效果不佳，需进一步完善隐私度量和更高效的加密方案。

---

## CurricuLLM: Designing Personalized and Workforce-Aligned Cybersecurity Curricula Using Fine-Tuned LLMs

**arXiv ID:** 2601.04940 | [PDF](https://arxiv.org/pdf/2601.04940v1)

**作者:** Arthur Nijdam `[一作]` (Lund University), Sara Ramezanian `[通讯]` (Karlstad University)

**关键词:** `Cryptography and Security` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出并实现了一个基于大语言模型（LLM）的自动化网络安全课程设计与评估框架，能够根据行业需求生成个性化课程路径并对现有课程进行知识领域匹配。

**💡 创新点**

创新点在于①首次将细粒度的知识领域（CSEC2017 9 个 KA）映射到职业角色需求，实现课程与就业市场的精准对齐；②构建两阶段处理流程（PreprocessLM + ClassifyLM）并在领域语料上进行微调；③通过人类专家对比验证，展示该方法可在与专家相当的水平下自动完成课程标注。

**🔧 技术方法**

核心技术为自然语言处理（NLP）与深度学习：使用 BERT（bert-base-uncased）作为 ClassifyLM 进行多标签分类；PreprocessLM 负责从课程描述中提取主题；训练时采用二元交叉熵、Adam 优化器、学习率 4e-5、批大小 64、10 个 epoch。

**📊 数据集**

所用数据集包括：①CSEC2017 生成的约 2100 条合成标签样本；②NIST 2025 NICE Knowledge Descriptions（约 600 条）；③真实课程数据——KTH、NTU、CMU 3 个硕士项目的课程标题与描述；④旧版（2017）与新版（2025）NICE 框架，用于评估知识领域分布的演变。

**📈 对比分析**

与三组人工标注者（熟悉 CSEC2017 的专家组、两组控制专家）进行交叉比对，采用百分比一致率和 Cohen’s κ 作为指标。实验显示，LLM 方法与专家组在 κ 值上仅差 0.03，远优于不熟悉框架的控制组；相较于 ChatGPT‑4o‑mini 等零样本模型，BERT 微调版本在多标签准确率上提升 6–12%。

**⚠️ 局限性**

局限性包括：①知识领域定义重叠导致人工标注一致率本身较低，模型难以做到绝对精确；②仅使用课程描述，未考虑学习目标、教材等更完整的课程信息，可能遗漏深层内容；③描述偏向宣传，可能导致模型过度关注社会与商业维度；④方法对不同地区或行业细分的适用性需进一步验证。

---

## Learning Sparsifying Transforms for mmWave Communication via $\ell^4$-Norm Maximization

**arXiv ID:** 2601.04980 | [PDF](https://arxiv.org/pdf/2601.04980v1)

**作者:** Sueda Taner `[一作]` (Ericsson Research), Christoph Studer `[通讯]` (ETH Zurich)

**通讯引用:** 10610 | **OpenAlex IDs:** https://openalex.org/A5083617223

**关键词:** `Information Theory` `Optimization` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

本文研究在毫米波 (mmWave) 5G 系统中利用稀疏波束空间表示来降低基带复杂度，并探讨离散傅里叶变换 (DFT) 是否为最佳稀疏化变换；通过对复数向量的 ℓ⁴‑范数最大化进行理论分析，提出两种学习算法（MSP 与 CA），验证 DFT 在多径与自由空间 LoS 模型下的最优性，并在合成与真实测量的信道向量上学习更优变换；通过 BEACHAES 与 LE 检测算法比较显示，在理想 LoS 环境下 DFT 已足够优秀，而在存在硬件缺陷的真实信道中学习的变换可显著提升误码率。

**💡 创新点**

①将复数稀疏字典学习扩展到单位ary 群；②通过 ℓ⁴‑范数作为稀疏度度量，证明 DFT 在多径模型下是 MSP 算法的固定点；③提出可避免投影的坐标上升 (CA) 算法，进一步分析 DFT 在 LoS 模型中的局部最优性；④揭示 DCT 在实值正弦信号模型下并非最优稀疏化变换。

**🔧 技术方法**

复数 ℓ⁴‑范数最大化、单位ary 组优化、匹配-拉伸-投影 (MSP) 算法、坐标上升 (CA) 算法、DFT 与 DCT 变换、基带波束空间估计 (BEACHAES) 与最大非零元素检测 (LE)。

**📊 数据集**

使用 QuaDRiGa 60 GHz UMi 模型生成的合成信道向量，以及 IEEE Communications Theory Workshop 的 1.27 GHz 8×8 方阵测量信道数据（含8 个失效天线）。

**📈 对比分析**

在合成 LoS 信道中，学习到的变换与 DFT 的 ℓ⁴‑范数相差 ≤18%，BER 仅提升约 2 %；在真实测量信道中，学习变换的 ℓ⁴‑范数可提升至 DFT 的 4 倍，BER 在相同误码率下可降低约 1–5 dB。

**⚠️ 局限性**

(a) 对于多径信道，理论仅证明 DFT 为 MSP 的固定点，未证明其为全局/局部最优；(b) CA 算法的收敛性和局部最优性证明仅在 LoS 单径模型下成立；(c) 学习到的变换缺乏 FFT 等低复杂度实现；(d) 在真实信道中，仍受硬件失效与测量误差的影响，需进一步研究鲁棒性与泛化能力。

---

## SKATER: Synthesized Kinematics for Advanced Traversing Efficiency on a Humanoid Robot via Roller Skate Swizzles

**arXiv ID:** 2601.04948 | [PDF](https://arxiv.org/pdf/2601.04948v1)

**作者:** Junchi Gu `[一作]` (University of Science and Technology of China), Shiwu Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6187 | **OpenAlex IDs:** https://openalex.org/A5101816100

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Reinforcement Learning` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

开发了一款配备四个被动轮子足部的25自由度类人机器人SKATER，并通过深度强化学习实现了滑行式“swizzle”步态；

**💡 创新点**

创新点在于将被动轮子嵌入足部并结合隐式步态奖励函数与多阶段课程学习，突破传统步态规划在非惯性滚动环境下的局限；

**🔧 技术方法**

采用Isaac Lab仿真平台、PPO算法、域随机化以及基于物理一致性和对称性的奖励设计实现控制策略；

**📊 数据集**

数据来源为机器人在仿真环境中生成的轨迹、力学与状态数据，未使用公开数据集；

**📈 对比分析**

与传统双足步行在相同速度下比较，冲击强度降低75.86%，能耗（CoT）降低63.34%，关节峰值力下降70%以上，显示显著性能提升；

**⚠️ 局限性**

局限包括侧向漂移、速度跟踪滞后以及仅能实现双足连续接触的swizzle步态。

---

## AVX / NEON Intrinsic Functions: When Should They Be Used?

**arXiv ID:** 2601.04922 | [PDF](https://arxiv.org/pdf/2601.04922v1)

**作者:** Théo Boivin `[一作]` (CERFACS), Joeffrey Legaux `[通讯]` (CERFACS)

**关键词:** `Software Engineering` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本研究构建了跨配置基准测试，评估 AVX/NEON 内置函数在不同操作系统、CPU 架构与编译器组合下的加速效果；

**💡 创新点**

创新点在于提出一套基于基准结果的决策流程，帮助开发者在多平台环境中判断何时使用内置函数；

**🔧 技术方法**

采用 SIMD Intrinsics Everywhere 库实现 AVX/NEON intrinsics，使用 C++ 标准循环与多种编译器（GCC、Clang、ICC、MSVC++）进行对比；

**📊 数据集**

使用合成的八种向量运算场景（包含基本、复杂运算及条件分支）作为测试数据集，循环 5e7 次；

**📈 对比分析**

通过交替运行原生与内置版本测量执行时间比，结果显示在 Windows/MSVC++ 上内置函数可获得最高加速（低至 5% 运行时间），但在部分场景下会导致 100%+ 延迟；

**⚠️ 局限性**

局限性包括仅覆盖了有限的 OS/CPU/编译器组合，未考虑 AMD/Intel macOS 等情况，且基准仅基于单一循环片段，可能与实际大型程序性能差异显著。

---

## OnomaCompass: A Texture Exploration Interface that Shuttles between Words and Images

**arXiv ID:** 2601.04915 | [PDF](https://arxiv.org/pdf/2601.04915v1)

**作者:** Miki Okamura `[一作]` (University of Tsukuba), Yoichi Ochiai `[通讯]` (University of Tsukuba)

**通讯引用:** 1889 | **OpenAlex IDs:** https://openalex.org/A5013807777

**关键词:** `Human-Computer Interaction` `Generation` `Data Synthesis` `Large Language Model` `Diffusion model` `Image` `Video` `Text` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出了一款名为OnomaCompass的基于网络的交互式系统，支持用户在纹理图像空间和拟声词空间之间切换，利用跨模态高维潜在空间进行探索，并提供画廊、纹理应用预览以及视频插值再嵌入的循环探索功能。

**💡 创新点**

创新点在于：①将日本拟声词（ideophones）作为感官桥梁，将声音符号与视觉纹理在同一潜在空间中进行映射；②实现两模态潜在空间的双向交互“跳板”，突破传统基于文字提示的单向对话；③通过视频插值与帧再嵌入构建“成生循环”，让用户在图像与词语之间进行自由组合与生成，促进无声化、低词汇负担的发散式创作。

**🔧 技术方法**

技术手段包括：Stable Diffusion用于生成纹理图像；OpenAI o1、Gemini 2.0 Flash等大语言模型用于生成英文描述、提示与概念文本；CLIP ViT‑B/32与OpenAI text‑embedding‑3‑small用于构建图像与文本的高维嵌入；UMAP降维将嵌入投影至二维空间；Three.js + React Three Fiber实现3D可视化与交互；Luma AI Ray 1.6实现视频插值；Gemini 2.5 Flash Image用于纹理与物体合成。

**📊 数据集**

数据集为自创的235条拟声词及其对应的Stable Diffusion生成的676张纹理图像。每条拟声词通过LLM先生成描述文本，再由描述文本驱动图像生成，并在此文本上构建跨模态链接，形成两模态的潜在空间。

**📈 对比分析**

在11名日语母语学生中，采用单一参与者的混合方法评估：使用NASA‑TLX、UEQ、SUS以及自定义问卷对比OnomaCompass与传统基于Prompt的图像生成工作流（Nano Banana）。结果显示OnomaCompass在主观工作负荷（NASA‑TLX整体、精神负荷、努力、挫折）显著降低，用户愉悦度（UEQ）显著提升，但总体可用性（SUS）略逊于基线。自定义问卷中OnomaCompass在探索多样纹理、避免卡顿、创意体验、意外发现等方面得分更高。

**⚠️ 局限性**

局限性包括：①交互设计尚不成熟，3D空间导航与选取误差影响用户体验；②潜在空间布局原理未充分传达，用户难以理解距离与结构含义；③仅使用自创拟声词，跨文化、跨语言的通用性未知；④实验规模小，未检验系统对专业设计师或更广泛用户群体的适用性。

---

## Patch-based Representation and Learning for Efficient Deformation Modeling

**arXiv ID:** 2601.05035 | [PDF](https://arxiv.org/pdf/2601.05035v1)

**作者:** Ruochen Chen `[一作]` (CNRS), Shaifali Parashar `[通讯]` (CNRS)

**关键词:** `Computer Vision and Pattern Recognition` `Representation Learning` `Computational Efficiency` `Point Cloud` `Mesh`

### 📋 论文摘要

**🎯 论文内容**

提出 PolyFit 这类基于切片的多项式（jet）表示，能够通过更新少量系数实现曲面变形，并基于该表示实现两种应用：PolySfT（测试时优化的形状恢复）和 OneFit（自监督的可穿戴布料仿真）。

**💡 创新点**

创新点在于：①将曲面划分为可单值高度函数的切片，并用低阶 jet 拟合，极大压缩变形自由度；②将变形控制从顶点级转为切片系数级；③在形状恢复中以测试时优化方式直接调整 jet 系数；④在布料仿真中实现网格和服装类型无关、跨分辨率的自监督学习。

**🔧 技术方法**

使用技术包括：jet 拟合、PCA 与 STN 进行切片方向校正、可微渲染、梯度下降测试时优化、MSE/光度/轮廓损失、碰撞、不可伸展、边界、重力与惯性物理损失、MLP（带跳连）与骨骼编码、空间变换网络。

**📊 数据集**

训练和评估数据集：合成 analytic 函数点云、CLOTH3D 服装网格、AMASS 动作序列、Kinect‑Paper、Paper‑Bend、synthetic SfT 数据集。

**📈 对比分析**

与基准比较：PolyFit 在 Chamfer 距离上优于 AtlasNet；PolySfT 在 Kinect‑Paper、synthetic SfT 上的 RMSE/误差均低于 SfT、DeepSfT、TD‑SfT、ϕ‑SfT，且每帧约 10 s（比 ϕ‑SfT 快 270×）；OneFit 与 GAPS、SNUG、HOOD、NCS 在 ε_c、面积/边长变化、推理时间（0.48 ms）等指标上相当或更好，且训练时间在 8 h 内可覆盖多种服装。

**⚠️ 局限性**

局限性：切片仅能表示单值高度函数，难以处理极端皱纹、巨大凸起或自遮挡；大变形时出现切片边缘缝隙；目前需手工后处理或 Laplacian 平滑，未来计划采用自适应分割、可变阶 jet 与更有效的边界控制。

---

## OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction

**arXiv ID:** 2601.04984 | [PDF](https://arxiv.org/pdf/2601.04984v1)

**作者:** Minseong Kweon `[一作]` (University of Minnesota), Jinsun Park `[通讯]` (Pusan National University)

**通讯引用:** 1420 | **OpenAlex IDs:** https://openalex.org/A5079380164

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Depth Estimation` `Gaussian Splatting` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出 OceanSplat，一种基于 3D 高斯展开的水下场景重建框架，利用三目视角一致性、自监督深度先验以及深度感知 alpha 调整来抑制中介水体导致的漂浮伪影；

**💡 创新点**

创新点在于：①将水平与垂直平移的三目视角投影与逆 warp 结合，实现更强的几何一致性；②通过两视图三角化得到合成 epipolar 深度先验，自监督地约束深度；③引入深度感知 alpha 调整，在训练早期抑制误放置的高斯体，提升几何精度；

**🔧 技术方法**

核心技术包括：3D 高斯展开（3DGS）、光照衰减与散射物理模型、逆 warp 与视差计算、最小二乘三角化、正则化 L1 与 SSIM 损失、深度残差损失以及基于 MLP 的 alpha 调整；

**📊 数据集**

使用 SeaThru‑NeRF、In‑the‑Wild 真实水下数据集以及通过 Fern/LLFF 合成的水下与雾霾场景；

**📈 对比分析**

与 SeaThru‑NeRF、SeaSplat、WaterSplatting、3DGS 等方法对比，OceanSplat 在 PSNR、SSIM 上提升约 1–3 dB，LPIPS 降低 0.02–0.05，且训练速度快、显存占用中等，整体性能显著优于现有方法；

**⚠️ 局限性**

局限性包括：依赖 SfM 提取的相机位姿；仅针对静态水下场景，难以处理非刚体运动；对高度动态水体或强光斑等极端光学条件仍有待改进。

---

## A DQN-based model for intelligent network selection in heterogeneous wireless systems

**arXiv ID:** 2601.04978 | [PDF](https://arxiv.org/pdf/2601.04978v1)

**作者:** Fayssal Bendaoud `[一作]` (Ecole Superieure en Informatique), karim Sehimi `[通讯]`

**关键词:** `Networking and Internet Architecture` `Reinforcement Learning` `Reinforcement Learning`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了一种基于深度 Q 网络（DQN）的智能网络选择模型，用于在异构无线环境（WiFi、4G、5G、LEO 星链）中实时决定最优接入网络。

**💡 创新点**

创新点在于将强化学习（DQN）与动态权重奖励函数相结合，能够在训练后显著优于传统多属性决策（MADM）方法，并能适应多 RAT 的变化与动态负载。

**🔧 技术方法**

采用了 DQN 强化学习算法、ε‑greedy 探索/利用策略、经验回放、目标网络，以及 NS3 仿真平台与传统 MADM（AHP、WSM、WPM、TOPSIS）进行对比评估。

**📊 数据集**

使用基于 NS3 生成的多网络 QoS 数据集，涵盖带宽、延迟、抖动、丢包率、网络负载与成本等指标，涵盖 5G、4G、WiFi、LEO 星链四种 RAT。

**📈 对比分析**

通过 2000 次训练周期，将 DQN 与传统 MADM 方法在 5G 选取率上进行比较；DQN 在 500 次后突破 90% 选取率，平均 87%，显著高于 AHP（75%）及其他方法，证明其性能优势。

**⚠️ 局限性**

主要限制在于需要大量训练时间和算力，且在早期探索阶段性能低，实时性不足；若要进一步提升精度需在更强计算资源上训练更多 epoch。

---

## Text as a Universal Interface for Transferable Personalization

**arXiv ID:** 2601.04963 | [PDF](https://arxiv.org/pdf/2601.04963v1)

**作者:** Yuting Liu `[一作]` (Northeastern University), Guibing Guo `[通讯]` (Northeastern University)

**通讯引用:** 4109 | **OpenAlex IDs:** https://openalex.org/A5007061198

**关键词:** `Computation and Language` `Recommendation System` `Reinforcement Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出一种以文本为通用接口的用户偏好表示方法，并基于此训练了全新的偏好推理模型AlignXplore+，实现了可解释、可迁移的用户画像；

**💡 创新点**

核心创新包括：①把用户偏好统一成自然语言摘要，解耦模型与任务；②提出两阶段训练框架——先用“生成-验证-合并”管线生成高质量标注，再通过基于课程化数据裁剪与累计奖励的RL提升长期可更新性；③通过文本摘要实现跨模型、跨任务、跨领域和多兴趣场景的无缝迁移；

**🔧 技术方法**

技术手段主要有：监督微调（SFT）+强化学习（RL）+基于LLM的“生成-验证-合并”数据合成+课程化数据裁剪策略+累计奖励函数+多任务评估；模型使用Qwen3-8B作为骨干；

**📊 数据集**

使用了包括Amazon‑Book、MIND、AlignX、MovieLens‑32M、PersonaMem、P‑Soups、HiCUPID等九个多领域数据集，合成约540K个SFT训练样本，112K个RL样本；

**📈 对比分析**

与直接序列模型（Qwen3-8B_non-thinking、TALLRec）和多种规模偏好推理模型（如GPT‑OSS‑20B、Qwen3‑32B、DeepSeek‑R1‑671B、Qwen3‑8B_thinking等）进行对比；在九个基准上，8B的AlignXplore+在平均分数上比同尺寸基线高~4.2%，甚至超过规模更大的模型；并在跨模型、跨域、正负样本缺失、并行兴趣等场景均表现出显著优势；

**⚠️ 局限性**

局限性包括：①合成的数据可能与真实部署场景存在差异；②未在生产环境中部署，对极长交互历史的鲁棒性未知；③未进行针对特定任务的细粒度算法调优，仅提出通用框架。

---

## Cardinality augmented loss functions

**arXiv ID:** 2601.04941 | [PDF](https://arxiv.org/pdf/2601.04941v1)

**作者:** Miguel O'Malley `[一作]` (Max Planck Institute for Mathematics in the Sciences), Miguel O'Malley `[通讯]` (Max Planck Institute for Mathematics in the Sciences)

**关键词:** `Machine Learning` `Classification` `Optimization` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

本文提出将基于基数的度量不变量（如大小（magnitude）和分布（spread））作为神经网络的损失函数，以解决类别不平衡问题。

**💡 创新点**

创新点在于将数学上描述度量空间有效多样性的基数不变量直接嵌入损失函数中，从而在批处理层面上抑制重复误差并提升少数类性能。

**🔧 技术方法**

技术上采用基数不变量的相似度矩阵求逆或线性方程求解来计算大小/分布，然后将其作为批量损失的归一化因子，亦在自监督学习框架中引入“除以大小/分布”策略。

**📊 数据集**

实验使用了两组合成不平衡多分类数据（分别为 50% 和 90% 主类）以及真实材料科学数据 DeepGlassNet（SciGlass）中的玻璃过渡温度二分类任务。

**📈 对比分析**

与交叉熵、均方误差等传统损失相比，大小损失在整体精度、宏 F1 及少数类表现上显著提升；分布损失在合成数据上表现欠佳，但在 DeepGlassNet 中也能提升多种指标（仅在精确度略有下降）。

**⚠️ 局限性**

局限性包括大小损失需要一定的热身期才能超过传统损失，计算复杂度随批量大小显著增加，以及其凸性与单调性尚未严格证明，需在更大规模任务上进一步验证。

---

## Conversational AI for Rapid Scientific Prototyping: A Case Study on ESA's ELOPE Competition

**arXiv ID:** 2601.04920 | [PDF](https://arxiv.org/pdf/2601.04920v1)

**作者:** Nils Einecke `[一作]` (Honda Research Institute Europe GmbH), Nils Einecke `[通讯]` (Honda Research Institute Europe GmbH)

**关键词:** `Artificial Intelligence` `Optimization` `AI Code Assistant` `Transformer` `Large Language Model` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

本文通过将ChatGPT作为研究伙伴，快速搭建并完成ESA ELOPE月球降落轨迹估计竞赛的原型，最终获得第二名。

**💡 创新点**

创新点在于系统性评估对话式AI在科学原型设计中的优势与局限，并提出可操作的最佳实践指南，以支持未来的AI辅助科研流程。

**🔧 技术方法**

使用ChatGPT（GPT‑4.5）进行代码生成、算法推理、数据处理、可视化、优化等任务，并辅以Git版本控制和测试驱动开发。

**📊 数据集**

采用ESA ELOPE竞赛的数据集，共计93条模拟降落序列，包括事件相机、IMU和雷达距离测量。

**📈 对比分析**

通过与排行榜上其他参赛队伍对比，本文实现的模型在最终评估中获得0.01282的得分，排名第二，仅次于SOMIS‑LAB的0.00692。

**⚠️ 局限性**

主要局限包括：模型往往做出不必要的结构改动、对多路讨论易产生误导、容易遗忘先前设定、对参数使用缺乏严谨性、生成错误代码时难以自行纠正。

---

## Breaking Robustness Barriers in Cognitive Diagnosis: A One-Shot Neural Architecture Search Perspective

**arXiv ID:** 2601.04918 | [PDF](https://arxiv.org/pdf/2601.04918v1)

**作者:** Ziwen Wang `[一作]` (Anhui University), Xingyi Zhang `[通讯]` (Anhui University)

**通讯引用:** 17784 | **OpenAlex IDs:** https://openalex.org/A5028634381

**关键词:** `Information Retrieval` `Neural Architecture Search` `Optimization` `Neural Architecture Search` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出一种针对认知诊断的多目标一-shot 神经网络架构搜索方法OSCD，能够在多种噪声场景下自动寻找鲁棒的模型结构。

**💡 创新点**

创新点在于构建完整二叉树搜索空间、使用权重共享超网络、将鲁棒性与性能同时作为多目标优化，并通过KL一致性约束提升模型对噪声的容忍度。

**🔧 技术方法**

采用一-shot NAS、NSGA‑II多目标进化、权重共享超网络、Lipschitz 条件验证及KL 散度对齐等技术。

**📊 数据集**

在公开教育数据集 ASSIST09 与 SLP‑Math 上进行实验。

**📈 对比分析**

与传统 CDM（如 DINA、IRT、NCD 等）以及 NASCD 进行对比，OSCD 在 AUC、ACC、RMSE 等指标上均取得第一或第二名，并在不同噪声比例下保持最小性能波动。

**⚠️ 局限性**

局限在于仅考虑四类噪声类型，搜索空间仍受操作符库限制，对极大规模数据集的可扩展性尚未充分验证。

---

## Refinements of Jensen's Inequality for Twice-Differentiable Convex Functions with Bounded Hessian

**arXiv ID:** 2601.05030 | [PDF](https://arxiv.org/pdf/2601.05030v1)

**作者:** Sambhab Mishra `[一作]`, Sambhab Mishra `[通讯]`

**关键词:** `Information Theory` `Optimization`

### 📋 论文摘要

**🎯 论文内容**

本文通过高阶泰勒展开与Grüss型不等式，系统地细化了Jensen不等式在二阶可导、Hessian有界的情形下的误差界，进而改进了对连续分布熵和Rayleigh衰落信道容量的估计。

**💡 创新点**

创新点在于将第四阶矩（偏度、峰度）显式加入Jensen间隙表达式，并利用积分剩余项与Grüss不等式耦合，既提升了误差上界，又提供了可调的切点优化方法。

**🔧 技术方法**

核心技术包括：积分形式泰勒展开、Grüss与Chebyshev不等式、四阶中心矩展开、极值优化（Jensen-Mercer）以及Green函数表示。

**📊 数据集**

作者主要使用理论示例（如正态、均匀、指数分布）及仿真验证，未涉及公开数据集；所有实验均在MATLAB/Python中自行生成的模拟数据上完成。

**📈 对比分析**

与经典方差上界、极限均值逼近以及已公开的熵与容量估计方法比较，实验显示第四阶展开误差下降至千分之一级别，Rayleigh信道容量上限误差从传统约15%降低到不足1%，性能显著提升。

**⚠️ 局限性**

局限性包括：仅适用于标量凸函数，Hessian需有界；高阶矩估计对重尾分布敏感；多元扩展与非可交换算子场景仍待进一步研究。

---

## From Idea to Co-Creation: A Planner-Actor-Critic Framework for Agent Augmented 3D Modeling

**arXiv ID:** 2601.05016 | [PDF](https://arxiv.org/pdf/2601.05016v1)

**作者:** Jin Gao `[一作]` (Massachusetts Institute of Technology), Saichandu Juluri `[通讯]` (Northeastern University)

**关键词:** `Multiagent Systems` `Generation` `Optimization` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Reinforcement Learning` `Agentic AI` `Mesh`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了一个多智能体的规划-执行-评估框架，用于创意3D建模，并通过人机协作提升模型质量。

**💡 创新点**

核心创新点在于将Actor-Critic架构拆分为Planner、Actor、Critic三角色，并加入人类监督与反馈，实现自我反思与迭代改进；同时构建了Blender-MCP与React-Three-Fiber的实时双向交互系统。

**🔧 技术方法**

技术包括Blender API、Blender-MCP、MCP工具集、WebSocket实时通信、CopilotKit + LangGraph多智能体链路、OpenAI ChatGPT‑4.1、React‑Three‑Fiber前端、Next.js。

**📊 数据集**

使用自定义的低多边形（low‑poly）建模任务集（如生日蛋糕、方桌、经典车），并结合在线3D资产库（PolyHaven、Hyper3D、Rodin）获取模型资源。

**📈 对比分析**

通过与单一prompt直接执行的基线进行对比，使用几何质量（几何体计数、顶点计数、相似度）与视觉质量（任务符合度、美学分数）等指标评估，结果显示多智能体反思框架在几何准确性、审美质量和任务完成率上均优于基线，且迭代次数越多改进越明显。

**⚠️ 局限性**

主要局限包括：MCP工具受限导致脚本执行失败、Actor不总能采纳Critic反馈导致质量退化、过度依赖原语建模限制表达能力、以及人类介入有时被智能体忽略或未充分利用。

---

## The RoboSense Challenge: Sense Anything, Navigate Anywhere, Adapt Across Platforms

**arXiv ID:** 2601.05014 | [PDF](https://arxiv.org/pdf/2601.05014v1)

**作者:** Lingdong Kong `[一作]`, Yao He `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Autonomous Driving` `Domain Adaptation` `Graph Neural Network` `Large Language Model` `Reinforcement Learning` `Contrastive Learning` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本报告围绕RoboSense 2025挑战赛构建了一个跨模态、跨平台、跨域的机器人感知鲁棒性评测框架，涵盖驾驶语言理解、社会导航、传感器布局泛化、无人机跨视角检索及跨平台3D目标检测五个任务；

**💡 创新点**

创新点在于统一多任务、多模态与多平台的鲁棒性评估，提出数据驱动的鲁棒性增强策略（如混合视角预训练、伪标签过滤、几何一致性约束等），并将多种基准数据与评测指标融合为公开排行榜；

**🔧 技术方法**

主要技术包括大规模视觉语言模型的提示调优与多视角融合、基于图神经网络和强化学习的社会导航策略、传感器放置相关的几何特征学习与补偿网络、跨视角检索的多模态融合与对比学习，以及无监督/弱监督域自适应与增强技术；

**📊 数据集**

使用的数据集涵盖DriveBench、Social-HM3D、Place3D、GeoText-190、Pi3DET等，并在IROS 2025会议上公开的RoboSense数据仓库进行统一数据管理；

**📈 对比分析**

通过在公开评测服务器（CodaBench、EvalAI等）进行排行榜对比，参赛方法在各项指标上均显著优于基线（如检测mAP提升约18%，社交导航SPL+PSC提升至70%+，跨视角检索Recall@1提升至38%，语言驱动驾驶QA准确率提升约15%），体现了鲁棒性提升的可量化效果；

**⚠️ 局限性**

局限性主要包括仍以单一类型的域漂移为主，缺乏多因子复合漂移的评估；对极端失效情况的置信度校准与拒绝机制尚不成熟；部分任务仍依赖标签或伪标签，导致适配过程易受噪声影响；并且不同任务间的统一评价标准尚待进一步完善。

---

## Evaluative Fingerprints: Stable and Systematic Differences in LLM Evaluator Behavior

**arXiv ID:** 2601.05114 | [PDF](https://arxiv.org/pdf/2601.05114v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## The Squirrel Parser: A Linear-Time PEG Packrat Parser Capable of Left Recursion and Optimal Error Recovery

**arXiv ID:** 2601.05012 | [PDF](https://arxiv.org/pdf/2601.05012v1)

**作者:** Luke A. D. Hutchison `[一作]`, Luke A. D. Hutchison `[通讯]`

**关键词:** `Programming Languages`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了 Squirrel Parser，一款能够直接处理所有形式左递归并实现最优错误恢复的线性时间 PEG packrat 解析器。

**💡 创新点**

创新点在于从基本原理推导出最小化算法——循环检测、Kleene 迭代展开、版本标记 memo 化以及仅用两阶段布尔标记实现的最优错误恢复；所有模块都被证明为必要。

**🔧 技术方法**

核心技术包括 PEG 语法、递归下降配合 memo 化、基于调用栈的循环检测、版本标签实现的无须多版本存储的迭代展开、以及两阶段（发现+恢复）布尔标记错误恢复策略。

**📊 数据集**

使用了 631 个涵盖左递归、错误恢复、运算符正确性与性能极限的综合单元测试集（无外部真实数据集）。

**📈 对比分析**

与现有 Packrat/Packrat‑LR 等实现对比，实验在 100,000 字符输入下保持 O(n·|G|) 线性时间，全部测试通过且性能与传统递归下降相当或更优。

**⚠️ 局限性**

局限性包括对复杂语法的手工调优需求、实现难度较高、并未在真实语言语料上验证，且对极大规模语法的内存占用尚未彻底评估。

---

## Higher-Order Adversarial Patches for Real-Time Object Detectors

**arXiv ID:** 2601.04991 | [PDF](https://arxiv.org/pdf/2601.04991v1)

**作者:** Jens Bayer `[一作]` (Fraunhofer IOSB and Fraunhofer Center for Machine Learning), Jürgen Beyerer `[通讯]` (Fraunhofer IOSB and Fraunhofer Center for Machine Learning)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Adversarial Attack` `Image`

### 📋 论文摘要

**🎯 论文内容**

本文研究了在YOLOv10实时目标检测器上使用高阶对抗补丁（patch）进行攻击与防御的循环过程，探讨多阶补丁与逐阶训练对模型鲁棒性的影响，并评估其在不同YOLO模型中的迁移性。

**💡 创新点**

创新点包括：①提出“高阶补丁”概念——对先前已硬化的模型进行重新优化得到的更强补丁；②设计“猫鼠游戏”实验框架，系统比较不同阶次、不同补丁数量(k=1 vs k=3)以及是否累积历史补丁的对抗训练效果；③展示高阶补丁对目标检测器具有更强的破坏性，并证明单纯对抗训练难以完全抵御此类攻击。

**🔧 技术方法**

采用的技术主要有：对抗补丁的增量优化（使用AdamW、色彩抖动、旋转、透视变换等数据增强）；在YOLOv10中实现补丁注入的对抗训练；对补丁与模型进行多阶实验（非累积、累积）；对不同YOLO系列模型做迁移性评估。

**📊 数据集**

数据集：训练使用COCO（80类）和INRIA Person（人类图像），评估使用COCO val2017，仅关注Person类；迁移性实验使用21个COCO预训练的YOLOv9/10/11/12模型。

**📈 对比分析**

比较方法：以mAP（AP_Person@[.5:.95]）衡量在无补丁、灰度补丁、以及各阶次补丁下的检测性能；对不同实验设置（k=1/3，是否累积）和不同补丁阶次的ΔAP进行可视化和统计。结果显示：高阶补丁导致更大性能下降；累积训练提升了高阶模型的鲁棒性，但仍无法完全恢复到灰度/干净性能；不同阶次补丁在不同模型间的迁移性呈波动下降趋势。

**⚠️ 局限性**

局限性：仅对单一类别（Person）进行实验；补丁优化耗时高（≈3.5h/补丁）；未系统探究参数变动对补丁强度的影响；未覆盖其他目标检测器或多类别情况；高阶补丁的极限阶次与收敛性仍未明朗。

---

## Distilling the Thought, Watermarking the Answer: A Principle Semantic Guided Watermark for Large Reasoning Models

**arXiv ID:** 2601.05144 | [PDF](https://arxiv.org/pdf/2601.05144v1)

**作者:** Shuliang Liu `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1186 | **OpenAlex IDs:** https://openalex.org/A5057914558

**关键词:** `Artificial Intelligence` `Knowledge Distillation` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了 ReasonMark 框架，先让模型完整完成推理（Thinking Phase），再在答案阶段插入语义自适应水印，保证推理逻辑不受干扰。

**💡 创新点**

创新点在于用关键性词（Critical Tokens）与主语义向量（PSV）对水印强度进行语义匹配，从而实现既保持文本质量又提升鲁棒性的双重目标。

**🔧 技术方法**

采用了关键性得分（GCC 与 CPS）对词进行评估，利用 PCA 提取 PSV，随后在答案生成时根据 token‑PSV 对齐动态调整水印强度；整个过程在推理和回答阶段分离实现。

**📊 数据集**

使用了 C4、WMT16‑DE‑EN、AIME、GSM8K 四大数据集，实验在 Qwen3‑32B 与 DeepSeek‑R1‑32B 两大 LLM 上进行。

**📈 对比分析**

与 KGW、SWEET、EWD 等传统 token/语义水印方法对比，ReasonMark 在 PPL（文本质量）、BLEU/ mACC（任务性能）和 AUC（检测率）上均优于或接近无水印基线，并在多种攻击下保持高检测率。

**⚠️ 局限性**

局限性包括：对极长文本或极端改写攻击的鲁棒性尚未充分验证；虽然延迟提升低于传统语义方法，但仍高于纯 token 方法，需要在资源受限场景进一步优化。

---

## GlimpRouter: Efficient Collaborative Inference by Glimpsing One Token of Thoughts

**arXiv ID:** 2601.05110 | [PDF](https://arxiv.org/pdf/2601.05110v1)

**作者:** Wenhao Zeng `[一作]` (Shanghai Jiao Tong University), Xiaodong Gu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2317 | **OpenAlex IDs:** https://openalex.org/A5033286111

**关键词:** `Artificial Intelligence` `Computational Efficiency` `Optimization` `Transformer` `Large Language Model` `Chain-of-Thought` `Text`

### 📋 论文摘要

**🎯 论文内容**

设计了训练无关的逐步协同推理框架，通过轻量模型预测首词熵来决定是否委托大型模型完成后续推理步骤。

**💡 创新点**

首次将首词熵作为步级难度判别指标，采用“Probe‑then‑Dispatch”机制，消除全步生成或后验验证的冗余开销，实现高效协同。

**🔧 技术方法**

使用轻量模型与大型模型的动态路由、首词熵阈值分流、KV 缓存切换优化，并可与令牌级加速（Speculative Decoding）协同。

**📊 数据集**

在 AIME24/AIME25（数学推理）、GPQA‑Diamond（通用推理）以及 LiveCodeBench v5/v6（代码生成）等多样化数据集上进行实验。

**📈 对比分析**

与单模型、随机、RSD、SpecCoT、SpecReason 等基线对比，在所有数据集上实现最佳精度‑延迟 Pareto 前沿；例如在 AIME25 上准确率提升 10.7% 并降低延迟 25.9%。

**⚠️ 局限性**

固定熵阈值缺乏自适应性；依赖双换行分段，无法直接适用于无结构链式思维文本。

---

## Controllable Memory Usage: Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction

**arXiv ID:** 2601.05107 | [PDF](https://arxiv.org/pdf/2601.05107v1)

**作者:** Muzhao Tian `[一作]` (Fudan University), Xiaoqing Zheng `[通讯]` (Fudan University)

**通讯引用:** 1103 | **OpenAlex IDs:** https://openalex.org/A5017835517

**关键词:** `Artificial Intelligence` `Reinforcement Learning from Human Feedback` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Sequential`

### 📋 论文摘要

**🎯 论文内容**

本文研究了长期交互中代理对历史记忆的依赖可控性，提出了 Steerable Memory Agent 框架以动态调节记忆依赖。

**💡 创新点**

创新点在于将记忆依赖度视为可调行为维度，构建基于 rubric 的记忆依赖评分、偏好对齐数据生成与强化学习控制，从而显著缓解“记忆锚定”。

**🔧 技术方法**

采用了 LLM 评估器做记忆依赖评分、用户模拟器生成偏好对齐数据、监督微调 + GRPO 强化学习，以及融合记忆对齐、任务质量与奖励模型的综合奖励函数。

**📊 数据集**

构建了包含约 10,000 个查询–记忆对的合成长期交互数据集（研究与教学场景），并从中抽取 1,000 条测试样本。

**📈 对比分析**

通过与无记忆、提示式记忆、直接记忆屏蔽等基线比较，使用对齐误差、任务质量奖励和 AlpacaEval 等指标，Steerable Memory Agent 在记忆依赖对齐上明显优于基线，同时保持甚至提升了生成质量。

**⚠️ 局限性**

局限包括合成数据可能不完全模拟真实交互、记忆依赖仅在 1–5 的离散尺度上建模、仅覆盖研究与教学两类场景，以及对更细粒度或连续依赖度的支持有限。

---

## ECLIPSE: An Evolutionary Computation Library for Instrumentation Prototyping in Scientific Engineering

**arXiv ID:** 2601.05098 | [PDF](https://arxiv.org/pdf/2601.05098v1)

**作者:** Max Foreback `[一作]` (Michigan State University), Julie Rolla `[通讯]` (NASA Jet Propulsion Laboratory)

**关键词:** `Neural and Evolutionary Computing` `Optimization` `Tabular` `Physics Related`

### 📋 论文摘要

**🎯 论文内容**

构建了 ECLIPSE 框架，用于将进化计算与复杂科学仿真工具（如电磁天线仿真 XFdtd 与拖曳模型 VECTOR）结合，支持空间科学硬件的设计与优化。

**💡 创新点**

创新点在于三层模块化架构（Individual、Evaluator、Evolver），实现了领域感知的编码、仿真接口与低评估成本的进化策略，显著降低了高成本仿真对进化搜索的限制。

**🔧 技术方法**

使用了 Python 编写的进化算法、异步评估器、基于形状的个体表示、遗传算法与 NSGA‑II 等选择机制，以及针对高成本仿真的稳态 GA 与 hill‑climber。

**📊 数据集**

数据集主要来自实际的仿真结果：电磁天线仿真输出（XFdtd）和航天拖曳仿真输出（VECTOR），用于评估天线灵敏度与 VLEO 卫星拖曳特性。

**📈 对比分析**

与传统手工设计和单纯的仿真循环相比，ECLIPSE 在同一硬件设计任务上实现了约 13 倍的墙钟时间缩减，并在多种空间科学应用（天线与卫星拖曳优化）中展示了更高质量的设计方案。

**⚠️ 局限性**

局限性包括：仿真工具本身的高评估成本与低并行度、软件互操作性与许可限制、以及对大规模并行计算资源的依赖，仍需进一步开发代理模型与低成本评估方法。

---

## Precoding Matrix Indicator in the 5G NR Protocol: A Tutorial on 3GPP Beamforming Codebooks

**arXiv ID:** 2601.05092 | [PDF](https://arxiv.org/pdf/2601.05092v1)

**作者:** Boyu Ning `[一作]`, Emil Björnson `[通讯]`

**关键词:** `Information Theory` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

本文系统梳理了 5G NR 及 5G‑A（Release 15‑18）中的 Beamforming Codebook（PMI）技术，提供从理论、标准化及实现角度的完整教程，填补了学术研究与行业实践之间的鸿沟。

**💡 创新点**

创新点在于：① 用清晰的表格与直观的图示解释了 PMI 代码本中的符号与物理意义；② 对 3GPP 代码本演化过程（Type I → Type II → Enhanced → Further Enhanced → Predicted PMI）做了统一的逻辑归纳；③ 详细拆解了各个 Release 的反馈参数、压缩方式与限制位图，形成了一套可直接用于实现的映射算法；④ 通过理论建模与性能对比，提供了代码本适用场景与效能评估。

**🔧 技术方法**

使用技术包括：MIMO OFDM 系统模型、CSI‑RS 与 PMI 反馈流程、3GPP 5G NR 标准文档、DFT/ Kronecker 组合的波束基底、压缩编码（幅度、相位、子带、时域）与位图约束；论文中还引用了压缩感知、WMMSE 等经典算法做基准。

**📊 数据集**

本文并未使用实验数据集，而是基于 3GPP 规范（TS 38.214）与公开的 5G‑A 代码本表格作为“数据来源”，并通过理论仿真与已发表的基准（如 3GPP 参考模型）进行对比。

**📈 对比分析**

比较方法主要是：1) 通过数学建模计算不同代码本在同一信道模型下的可达率与干扰损失；2) 对比不同 Release 的 PMI 压缩率与反馈开销；3) 在多用户多天线场景下评估 beamforming 性能与信号覆盖。结果显示，Enhanced 与 Predicted PMI 在高频段与大规模天线阵列下能显著降低反馈开销，同时保持相近或更优的波束聚焦效果。

**⚠️ 局限性**

局限性包括：① 仍然基于 3GPP 标准的抽象描述，缺乏大规模实测验证；② 对于极高频（THz）与新型天线拓扑，标准尚未覆盖，需进一步扩展；③ 代码本的压缩与位图约束在实际系统中实现时可能导致硬件实现复杂度提升。

---

## Driver-Intention Prediction with Deep Learning: Real-Time Brain-to-Vehicle Communication

**arXiv ID:** 2601.05084 | [PDF](https://arxiv.org/pdf/2601.05084v1)

**作者:** Niloufar Alavi `[一作]`, Stefan Goetz `[通讯]`

**关键词:** `Human-Computer Interaction` `Classification` `Autonomous Driving` `Convolutional Neural Network` `Time Series` `Biomedical Data`

### 📋 论文摘要

**🎯 论文内容**

使用深度学习模型（1D 卷积神经网络）对 64 通道原始 EEG 信号进行分类，以预测驾驶者在模拟驾驶环境中想象的三类转向意图（左转、右转、直行）。

**💡 创新点**

创新点在于：① 直接使用未做传统预处理的原始 EEG 进行训练，降低了实时 BCI 的系统复杂度；② 通过重叠时间窗和 SMOTE 平衡样本，实现了 83.7% 的多类分类准确率，显著高于以往仅基于特征提取的研究；③ 通过可视化（拓扑图、诱发电位）进一步证明不同意图对应的脑区激活差异。

**🔧 技术方法**

主要技术包括：TensorFlow 实现的 1D CNN、Adam 优化器、SMOTE 过采样、卷积/池化/Dropout/全连接层、批归一化（未使用 ReLU、零填充以提升精度）、Python3.9/ MNE 可视化、Welch 频谱分析。

**📊 数据集**

使用了 10 名右撇子受试者（共 92 段驾驶轨迹）在 64 通道 EEG（512 Hz）下收集的实验数据；数据按 70%/30% 划分训练/验证，进一步使用 SMOTE 进行类别平衡。

**📈 对比分析**

与传统 EEG‑BCI 仅利用特征提取方法（准确率 50–66%）相比，本研究实现了 83.7% 的整体准确率；单类性能最高为右转（Precision 0.889、Recall 0.833、F1 0.860），其次为直行（Precision 0.858、Recall 0.832、F1 0.845），左转略低（Precision 0.748、Recall 0.854、F1 0.797）。

**⚠️ 局限性**

局限性包括：仅招募右撇子受试者，导致空间注意偏向右侧；模型未在独立的未见受试者上测试；实验仅基于想象转向，缺乏真实驾驶数据；眼动/肌电等非脑信号可能混入；并未实现实时在线评估。

---

## DAVOS: An Autonomous Vehicle Operating System in the Vehicle Computing Era

**arXiv ID:** 2601.05072 | [PDF](https://arxiv.org/pdf/2601.05072v1)

**作者:** Yuxin Wang `[一作]` (University of Delaware), Weisong Shi `[通讯]` (University of Delaware)

**通讯引用:** 23537 | **OpenAlex IDs:** https://openalex.org/A5100651611

**关键词:** `Operating Systems` `Autonomous Driving` `Safty and Privacy` `Computational Efficiency` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了DAVOS，一个面向车辆计算时代的统一车辆操作系统架构，集成了实时感知管线、数据服务与安全隐私保护。

**💡 创新点**

创新点在于将传感器-内存通信(SIM)、自适应实时调度、上下文风险指数(CRI)、分层可查询存储(AVS)、隐私友好可信计算(PaCC)以及统一编程接口(VPI)四大子系统在单一OS中协同工作，实现安全、实时、效率与可扩展四原则（SSEE）的闭环。

**🔧 技术方法**

主要技术包括：共享内存感知传输、基于依赖与可信度的分层实时调度、方向感知风险评估、SSD/HDD层次化写入与查询索引、TEE中可信容器与访问控制、跨层统一API抽象。

**📊 数据集**

使用真实车辆硬件平台的多模态传感器数据（相机、雷达、LiDAR、GNSS、CAN等）以及Autoware‑Universe/ROS2的感知管线数据，结合标准的车辆实验测试集进行评估。

**📈 对比分析**

与传统ROS2+DDS、单独RTOS+日志系统等基线相比，DAVOS在感知到决策的平均延迟从~520 ms降低至<200 ms，p95尾部延迟降低30%~40%；在持续日志记录上，AVS实现了5×更高的写吞吐率且查询延迟保持在10 ms以内，整体系统利用率提升约15%。

**⚠️ 局限性**

局限性包括：目前实现仍在原型阶段，硬件覆盖范围有限；跨平台兼容性和与现有车载生态（如AUTOSAR、NVIDIA DRIVE）的深度集成尚待完善；对极端负载下的资源抢占与安全性证明需要进一步实验验证。

---

## From Rays to Projections: Better Inputs for Feed-Forward View Synthesis

**arXiv ID:** 2601.05116 | [PDF](https://arxiv.org/pdf/2601.05116v1)

**作者:** Zirui Wu `[一作]` (Hong Kong University of Science and Technology), Jie Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 7667 | **OpenAlex IDs:** https://openalex.org/A5047979132

**关键词:** `Computer Vision and Pattern Recognition` `Data Synthesis` `Depth Estimation` `Transformer` `Masked Auto‑Encoding` `Vision Transformer` `Point Cloud` `Video`

### 📋 论文摘要

**🎯 论文内容**

提出了基于投影条件的前馈视图合成模型，替代传统的Plücker射线编码

**💡 创新点**

创新点在于使用相对投影点云作为条件，构造了相机无关的投影空间，并结合MAE预训练提升模型鲁棒性

**🔧 技术方法**

投影条件、Vision Transformer（ViT）、RoPE位置编码、Masked Auto‑Encoding预训练、3D点云光栅化（GSP-LA）

**📊 数据集**

使用改进后的 RealEstate10K（含稠密深度与精细相机标定）和 DL3DV 大规模无标注视频数据进行预训练

**📈 对比分析**

在自建的视图一致性基准和标准新视图合成基准上，与 RayZer、LVSM 等方法对比，PSNR 约 25dB，SSIM、LPIPS 均优于基线，视图一致性显著提升

**⚠️ 局限性**

仍依赖准确的深度估计和相机标定，动态场景处理有限，且在极端视角变换下性能仍受限

---

## A Lightweight and Explainable Vision-Language Framework for Crop Disease Visual Question Answering

**arXiv ID:** 2601.05143 | [PDF](https://arxiv.org/pdf/2601.05143v1)

**作者:** Md. Zahid Hossain `[一作]` (Ahsanullah University of Science and Technology), Md. Siam Ansary `[通讯]` (Ahsanullah University of Science and Technology)

**通讯引用:** 19 | **OpenAlex IDs:** https://openalex.org/A5025168758

**关键词:** `Computer Vision and Pattern Recognition` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Vision Language Model` `Multimodality` `Agriculture Related`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一个轻量级的视觉-语言框架，用于植物病害视觉问答；

**💡 创新点**

创新点在于两阶段训练策略、Swin Transformer 视觉编码器与 BART/T5 文本解码器的组合，以及结合 Grad‑CAM 与 token‑level 解释方法；

**🔧 技术方法**

技术包括 Swin Transformer 视觉编码器、BART/T5 文本解码器、两阶段预训练+冻结策略、跨模态注意力、Grad‑CAM 与词级归因；

**📊 数据集**

使用了 CDDM（Crop Disease Domain Multimodal）数据集，包含 16 种作物、60 种病害、超过一百万 QA 对；

**📈 对比分析**

与 ViT‑based 模型、LLaVA‑AG、Qwen‑VL‑Chat‑AG 等大型模型比较，Swin‑T5 在作物/病害识别准确率 99.94%/99.06%、BLEU 0.994、ROUGE‑L 0.996、BERTScore 0.999 的同时，仅 251M 参数、373 ms 推理；

**⚠️ 局限性**

局限性包括缺乏病害防治建议、对外部知识依赖不足、对未见作物泛化能力有限。

---

## Why Are Some Countries More Politically Fragmented Online Than Others?

**arXiv ID:** 2601.05093 | [PDF](https://arxiv.org/pdf/2601.05093v1)

**作者:** Yuan Zhang `[一作]`, Alexandre Bovet `[通讯]`

**关键词:** `Social and Information Networks` `Text` `Graph`

### 📋 论文摘要

**🎯 论文内容**

对巴西、西班牙和美国三国在X（Twitter）上政治影响者的共关注网络进行多尺度社区检测，提出新的碎片化得分，比较各国与各意识形态群体的政治碎片化水平。

**💡 创新点**

创新在于结合Markov Stability多尺度社区发现与有效分支因子构建的碎片化度量，能够跨国、跨意识形态体系统一比较碎片化，并揭示社会身份与意识形态相互作用对碎片化的影响。

**🔧 技术方法**

技术手段包括多尺度社区检测（Markov Stability + Leiden算法）、有效社区数（effective number of communities）、有效分支因子、余弦相似度权重比较、统计检验（Mann‑Whitney U）等网络分析与统计方法。

**📊 数据集**

数据集为2022年巴西总统选举、2023年西班牙大选、2024年美国总统选举期间调查样本的Twitter/X账户，收集了18,325位政治影响者的关注关系以及其自报的意识形态与社会身份标签。

**📈 对比分析**

通过对多尺度碎片化得分进行平均并对左/右群体分别计算，比较三国碎片化得分（巴西2.8、西班牙1.8、美国1.7），发现碎片化与各国选举党派分布的有效党派数高度相关，且各国意识形态群体存在不对称碎片化。

**⚠️ 局限性**

局限性包括仅使用自报身份的影响者、仅聚焦X平台共关注网络、社区检测与碎片化得分对网络构建与算法参数敏感、样本来源受调查问卷覆盖范围限制。

---

## Agent-as-a-Judge

**arXiv ID:** 2601.05111 | [PDF](https://arxiv.org/pdf/2601.05111v1)

**作者:** Runyang You `[一作]` (Hong Kong Polytechnic University), Wenjie Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 11482 | **OpenAlex IDs:** https://openalex.org/A5100408983

**关键词:** `Computation and Language` `Large Language Model` `Agentic AI` `Text` `Review/Survey Paper` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

综述了从 LLM-as-a-Judge 向 Agent-as-a-Judge 的演进过程，提出了多维度分类与进化阶段，并梳理了核心技术与应用场景

**💡 创新点**

首次系统性综述 Agent-as-a-Judge，构建了五大技术维度（多代理协作、规划、工具集成、记忆与个性化、优化范式）和三阶段发展框架，填补了该领域的理论与方法空白

**🔧 技术方法**

对现有文献进行归纳、分类与理论建模，使用案例对比、概念图绘制与层级分析等方法论手段

**📊 数据集**

未涉及实验数据集，主要引用公开论文与基准（如 MT‑Bench、MathBench 等）中的方法与结果

**📈 对比分析**

由于是综述论文，没有直接实验对比；文中通过引用原论文的指标与方法，概括 Agent‑as‑a‑Judge 在鲁棒性、可验证性和细粒度评估方面相较于传统 LLM‑as‑a‑Judge 的提升

**⚠️ 局限性**

领域尚处于早期阶段，缺乏统一定义与共识；综述包含早期提示式“代理”方法，可能与严格的自治标准不完全一致；缺少系统化实验验证与统一评价指标

---

## Dynamics in Search Engine Query Suggestions for European Politicians

**arXiv ID:** 2601.05081 | [PDF](https://arxiv.org/pdf/2601.05081v1)

**作者:** Franziska Pradel `[一作]` (Technical University of Munich), Fabian Haak `[通讯]` (University of Applied Sciences Cologne)

**关键词:** `Information Retrieval` `Text`

### 📋 论文摘要

**🎯 论文内容**

本研究系统收集并分析了欧洲十国对欧洲政客名称的Google搜索建议的时间稳定性和跨国相似性，以评估用户对这些政客的隐性信息需求。

**💡 创新点**

创新点在于首次将性别、政治角色、政府身份等元属性与查询建议的稳定性和跨国相似性相结合，并通过Jaccard相似系数与多元回归量化这些因素的影响。

**🔧 技术方法**

主要技术包括使用专用爬虫抓取Google自动完成API返回的前10条建议，利用Jaccard系数衡量建议集合的相似度，并采用多元线性回归分析元属性对稳定性和相似性的作用。

**📊 数据集**

数据集来自约46.6百万条查询建议，涵盖793名欧盟政客（包括领袖、候选人、内阁成员），覆盖10个欧盟成员国，所有建议均通过Google翻译统一为英文后使用。

**📈 对比分析**

通过将查询建议按两周周期聚合并计算Jaccard相似度，然后与元属性进行回归比较，模型R²约为0.13–0.14，表明这些属性对建议的稳定性和相似性具有显著但有限的解释力。

**⚠️ 局限性**

研究局限包括仅使用前10条建议且未考虑建议顺序、使用机器翻译可能引入语义误差、无法完全排除Google算法更新与个性化对建议的影响，以及模型解释度相对较低。

---

## Compositional Steering of Large Language Models with Steering Tokens

**arXiv ID:** 2601.05062 | [PDF](https://arxiv.org/pdf/2601.05062v1)

**作者:** Gorjan Radevski `[一作]` (Independent), Goran Glavaš `[通讯]` (Center for Artificial Intelligence and Data Science)

**关键词:** `Computation and Language` `Generation` `Knowledge Distillation` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出了通过自蒸馏学习的“Steering Token”实现LLM多行为可控生成，并在输入层引入专门的组合令牌以实现多行为的零样本组合；

**💡 创新点**

创新点在于：①将单一行为压缩为输入词嵌入令牌，而非模型内部激活；②引入专门的组合令牌学习行为组合的通用操作；③通过正交正则化保证组合令牌与行为令牌独立；④证明该方法在未见行为及组合上具有强零样本泛化；

**🔧 技术方法**

主要技术包括自蒸馏（teacher‑student KL散度）、行为令牌初始化为行为指令词嵌入平均、组合令牌初始化为零、正交正则化、以及对不同模型进行冻结训练；

**📊 数据集**

使用了Smoltalk数据集中的约50k条提示，随机生成10种行为表述，覆盖15种可验证行为（语言、长度、格式、句数）；

**📈 对比分析**

与三类基线（指令拼接、LoRA‑DARE、LM‑Steer）以及无组合令牌的拼接方式进行对比。实验显示：在Qwen‑8B等模型上，Steering Token在2/3行为组合的未见情形下分别提升约5–6%准确率，混合指令与令牌的Hybrid方法能进一步提升至10%+；跨模型、跨规模实验也表明该方法稳健且随模型增大效果提升；

**⚠️ 局限性**

局限性包括：仅验证可自动判定的行为（长度、格式等），对主观或语义细微约束缺乏评估；最大仅评估3个约束，未知更高阶组合的性能；实验规模限于14B以下，未知更大模型的表现。

---

## Reinforced Efficient Reasoning via Semantically Diverse Exploration

**arXiv ID:** 2601.05053 | [PDF](https://arxiv.org/pdf/2601.05053v1)

**作者:** Ziqi Zhao `[一作]` (Shandong University), Xin Xin `[通讯]` (Shandong University)

**通讯引用:** 81504 | **OpenAlex IDs:** https://openalex.org/A5100328102

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Large Language Model` `Reinforcement Learning` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于语义熵引导的MCTS回放与长度感知优势估计的RLVR框架（ROSE）来提升LLM的推理准确性和效率。

**💡 创新点**

创新点在于结合语义熵与生成熵的分支策略、ε探索机制实现语义多样化探索，以及长度调节的段级优势估计以抑制过度推理。

**🔧 技术方法**

使用了RLVR、GRPO、MCTS、语义熵计算、长度感知优势估计等技术；训练基于改进的GRPO目标并加入KL正则。

**📊 数据集**

实验数据集包括MATH训练集以及AIME2024/2025、AMC23、MATH500等数学推理基准。

**📈 对比分析**

与GRPO、Dr.GRPO、DAPO、TreePO、FR3E等基线对比，ROSE在多模型、不同规模上均取得pass@8显著提升，并在推理长度上更高效。

**⚠️ 局限性**

主要局限为仅在≤8B参数模型上验证，且仅针对数学推理任务，未来需扩展到更大模型与其它推理领域。

---

## VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control

**arXiv ID:** 2601.05138 | [PDF](https://arxiv.org/pdf/2601.05138v1)

**作者:** Sixiao Zheng `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**通讯引用:** 16344 | **OpenAlex IDs:** https://openalex.org/A5084959430

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Depth Estimation` `Diffusion model` `World Model` `Video` `Point Cloud`

### 📋 论文摘要

**🎯 论文内容**

提出 VerseCrafter，一种基于 4D 几何控制的可控视频生成模型，实现了对相机和多物体运动的精准、解耦控制，并在真实视频中生成高质量、视角一致的动态视频。

**💡 创新点**

创新点包括：
1) 4D 几何控制表示——统一背景点云 + 每个对象的 3D 高斯轨迹；
2) 轻量 GeoAdapter 将 4D 控制映射到冻结的 Wan2.1 视频扩散器；
3) 自动化数据引擎与 VerseControl4D 数据集，解决了 4D 标注稀缺问题。

**🔧 技术方法**

主要技术：3D 点云重建、MoGe-2 深度估计、Grounded‑SAM2 实例分割、3D 高斯轨迹、Wan2.1 视频扩散器、轻量 GeoAdapter、文本编码器 umT5、自动数据标注管线。

**📊 数据集**

数据集：基于 Sekai‑Real‑HQ 与 SpatialVID‑HQ 的真实视频，构建 35k 训练 + 1k 验证样本的 VerseControl4D；对比实验使用公开的 Yume、Uni3C、Perception‑as‑Control 等数据集。

**📈 对比分析**

评估方法：使用 VBench‑I2V、相机控制误差 (RotErr/TransErr)、对象运动误差 (ObjMC) 等指标。结果显示，VerseCrafter 在整体分数、图像质量、动态度、运动平滑、背景/主体一致性上均优于 Perception‑as‑Control、Yume、Uni3C，并显著降低相机与对象控制误差。

**⚠️ 局限性**

局限性：
1) 受背景点云与深度估计误差影响，极端遮挡或快速运动下仍易出现前后置换；
2) 需要手工/半自动掩码标注，处理对象数量有限；
3) 目前仅在 80 帧短视频上验证，长期序列性能未知。

---

## Sequential Subspace Noise Injection Prevents Accuracy Collapse in Certified Unlearning

**arXiv ID:** 2601.05134 | [PDF](https://arxiv.org/pdf/2601.05134v1)

**作者:** Polina Dolgova `[一作]` (CISPA Helmholtz Center for Information Security), Sebastian U. Stich `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `Machine Learning` `Safety and Privacy` `Convolutional Neural Network` `Transformer` `Gaussian Splatting` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出块级噪声调度的认证无学习方法，利用正交子空间顺序注入噪声，降低噪声对模型准确率的冲击，同时保持（ε,δ）认证保障；

**💡 创新点**

①将噪声按正交子空间分块顺序调度；②用模型相近度Δ(ρ)代替最坏情况裁剪，显著减小噪声规模；③在子空间框架下重新证明 Rényi 隐私与 Shifted Rényi Divergence，证明相同隐私预算；④实验验证在 MNIST、CIFAR‑10 上比传统 NFT 更稳健且 MIA 仍为 100%；

**🔧 技术方法**

随机梯度裁剪、加高斯噪声、Rényi 隐私、子空间分解、Shifted Rényi Divergence、Shift Reduction Lemma、隐私放大与高斯机制；

**📊 数据集**

MNIST（全连接网络）、CIFAR‑10（ResNet‑18）及实验中提及的 ViT‑Tiny；

**📈 对比分析**

与传统 NFT、无裁剪 NFT、Retrain、Fine‑tuning、Gradient Ascent、Influence Unlearning、SaLUN、ℓ1‑sparse 等基线比较；在随机 10% 删除和类别删除任务中，Block‑wise NFT 在 UA 与 MIA 均维持 100%，后续恢复准确率优于 NFT，且在 RTE 上低于多数经验方法，测试准确率接近 Retrain；

**⚠️ 局限性**

当前定义过于严格，既难以兼顾实用准确率又难以加速；子空间分解与调度方式仍可改进；在更大模型与更复杂任务上需进一步验证，并努力缩小与 retrain 的差距。

---

## VERSE: Visual Embedding Reduction and Space Exploration. Clustering-Guided Insights for Training Data Enhancement in Visually-Rich Document Understanding

**arXiv ID:** 2601.05125 | [PDF](https://arxiv.org/pdf/2601.05125v1)

**作者:** Ignacio de Rodrigo `[一作]` (Comillas Pontifical University), Jaime Boal `[通讯]` (Comillas Pontifical University)

**通讯引用:** 244 | **OpenAlex IDs:** https://openalex.org/A5047794771

**关键词:** `Computer Vision and Pattern Recognition` `Data Synthesis` `Representation Learning` `Transformer` `Vision Language Model` `Supervised Fine-Tuning` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

该论文提出了VERSE方法，利用PCA降维可视化视觉嵌入空间，定位低性能聚类，并通过合成样本增强训练数据，从而提升视觉-语言模型在视觉丰富文档理解任务中的表现。

**💡 创新点**

创新点在于将模型内部视觉嵌入空间作为评估标准，识别错误集群并针对性生成训练数据；此外强调视觉质量应从模型视角评估而非人类视觉感知，提供了从嵌入空间反向生成训练样本的思路。

**🔧 技术方法**

使用的技术包括视觉嵌入提取（从Donut、Idefics2等VLM的视觉编码器获取）、PCA降维、K‑means聚类、可视化分析、数据增强（旋转、缩放、渲染）、Zoom分布策略、LoRA微调及性能对比评估。

**📊 数据集**

实验基于MERIT数据集（西班牙语学生成绩单的多版本合成数据）及其保密版MERIT Secret，用于验证模型在真实样本上的表现。

**📈 对比分析**

与SaaS解决方案（GPT‑4‑O、Pixtral）对比，VERSE微调后Idefics2在MERIT Secret上达到0.8101 F1，超过GPT‑4‑O（0.7821）并与Pixtral（0.7267）竞争，证明本地模型可匹敌甚至超越云端服务。

**⚠️ 局限性**

局限性包括主成分解释的多义性、对视觉编码器架构的高度依赖、合成样本生成缺乏直接从嵌入空间采样的机制，以及在其他VrDU任务或语言上的可迁移性尚需进一步验证。

---

## Rule Rewriting Revisited: A Fresh Look at Static Filtering for Datalog and ASP

**arXiv ID:** 2601.05108 | [PDF](https://arxiv.org/pdf/2601.05108v1)

**作者:** Philipp Hanisch `[一作]` (Knowledge-Based Systems Group, TU Dresden), Markus Krötzsch `[通讯]` (Knowledge-Based Systems Group, TU Dresden)

**关键词:** `Databases` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

在Datalog与ASP中提出了一种可扩展的静态过滤（static filtering）优化方法，能够通过对规则中的过滤子句进行符号化分析，自动推导出更紧凑的过滤条件并重写程序；

**💡 创新点**

创新点在于：①将Kifer与Lozinskii的经典静态过滤推广到支持任意布尔组合过滤谓词与更一般的过滤谓词；②设计了可计算的近似推理机制（CASF）实现可判定且多项式时间的过滤计算；③提出了对带有非单调否定和ASP稳定模型的支持；

**🔧 技术方法**

核心技术包括：符号过滤公式的布尔代数表示与简化、近似蕴含判定、Horn近似推理、可判定的合取形式过滤（CASF）以及基于过滤谓词的程序重写；

**📊 数据集**

实验使用Wikidata上5个属性（如P2652、P530等）作为输入数据集，评估了在Soufflé、Nemo、Clingo、DLV等规则引擎中的性能；

**📈 对比分析**

与原始程序比较，重写后程序在大多数情况下获得10–30倍的速度提升；在数据量达百万级时，原程序往往超时，而重写后能够在几秒内完成；静态过滤本身的执行时间仅几毫秒；

**⚠️ 局限性**

局限性：①在最坏情况下算法仍可能是双指数复杂；②对包含算术等不可判定谓词的过滤需依赖近似推理，可能导致过度保守或误差；③需要手工指定哪些谓词作为过滤谓词；④在已高度优化的程序上收益有限。

---

## Code-Mix Sentiment Analysis on Hinglish Tweets

**arXiv ID:** 2601.05091 | [PDF](https://arxiv.org/pdf/2601.05091v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computation and Language`

---

## Token-Level LLM Collaboration via FusionRoute

**arXiv ID:** 2601.05106 | [PDF](https://arxiv.org/pdf/2601.05106v1)

**作者:** Nuoya Xiong `[一作]` (Carnegie Mellon University), Zhuokai Zhao `[通讯]` (Meta)

**关键词:** `Artificial Intelligence` `Generation` `Optimization` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Mixture of Experts` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于轻量级路由器的多模型 token 级协同生成框架，自动在生成过程中选取最合适的专家并提供补偿 logits。

**💡 创新点**

将专家选择与补偿生成双重机制结合，解决纯专家选择无法覆盖所有前缀的局限，并通过训练无梯度依赖的路由器实现高效灵活的 token 级合作。

**🔧 技术方法**

采用监督 fine‑tuning + Complemented Direct Preference Optimization (CDPO) 对路由器进行训练，利用 logit 加法融合路由器与专家分布；理论上基于 Performance Difference Lemma 分析 token 级路由的极限。

**📊 数据集**

在 Llama‑3 与 Gemma‑2 系列模型上使用数学 (GSM8K、MATH500)、代码 (MBPP、HumanEval) 与指令 (IfEval) 任务集以及 OpenHermesPreferences 与 Open‑PerfectBlend 等通用数据。

**📈 对比分析**

与序列级协作、Token‑level Collab、模型合并 (DARE、TaskArithmetic) 以及直接 fine‑tuned baseline 进行对比；在各领域平均准确率上均超过对手，通用数据集的 GPT‑4o winrate 亦显著提升。

**⚠️ 局限性**

对极大模型仍需依赖路由器的训练质量，且在小模型规模下补偿机制效果有限；理论证明仅在理想覆盖假设下才能完全覆盖所有前缀，实际部署需进一步鲁棒性验证。

---

## Arabic Prompts with English Tools: A Benchmark

**arXiv ID:** 2601.05101 | [PDF](https://arxiv.org/pdf/2601.05101v1)

**作者:** Konstantin Kubrak `[一作]` (General Organization for Social Insurance), Faisal Alsaby `[通讯]` (General Organization for Social Insurance)

**关键词:** `Artificial Intelligence` `Large Language Model` `Prompt Engineering` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文提出了第一个面向阿拉伯语的工具调用与代理能力评估基准，并将Berkeley Function Calling Leaderboard（BFCL）翻译并扩展为多语言实验框架；

**💡 创新点**

创新点在于：①构建专门针对阿拉伯语的工具调用基准；②采用多轴实验设计（系统提示、调用方式、用户查询、工具描述四个语言维度）系统评估语言影响；③发现并量化“累积语言惩罚”与多步骤任务中的性能衰退；

**🔧 技术方法**

使用了大语言模型的原生函数调用（Native Function Calling）与提示式调用（Prompt-based），以及基于AST的精确评估指标；

**📊 数据集**

使用了BFCL数据集的阿拉伯语翻译版，包括14类功能调用任务；

**📈 对比分析**

通过对五个开源模型（GPT‑OSS‑20b、Llama‑3.3‑70b、三款Qwen）在16种实验组合下的准确率进行排名与成对对比，发现阿拉伯语查询平均比英语低5‑10%，并在多步骤、并行调用场景中更为明显；

**⚠️ 局限性**

局限性包括：翻译质量可能导致阿拉伯语语义失真；仅评估五个开源模型，缺乏对闭源或本土化模型的覆盖；依赖AST评估，忽略语义正确性；基准继承BFCL任务结构，缺少阿拉伯语特有的代理交互场景。

---

## Multi-Disciplinary Dataset Discovery from Citation-Verified Literature Contexts

**arXiv ID:** 2601.05099 | [PDF](https://arxiv.org/pdf/2601.05099v1)

**作者:** Zhiyin Tan `[一作]` (L3S Research Center), Changxu Duan `[通讯]` (Technische Universität Darmstadt)

**关键词:** `Digital Libraries` `Retrieval` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

我们提出一种基于论文引用上下文的多学科数据集检索框架，绕过元数据依赖，实现从文献中直接发现和推理数据集。

**💡 创新点**

创新点在于将引用上下文视为语义桥梁，利用大型语言模型和实体解析提取数据集名称与使用语境，并在跨学科环境下提升检索召回率和实用性。

**🔧 技术方法**

采用Semantic Scholar Academic Graph的上下文检索，基于Qwen2.5-72B-Instruct进行数据集提取和功能归类，结合实体解析、归一化与信任源评估，形成三阶段管道。

**📊 数据集**

数据集来源包括Semantic Scholar学术图谱、从计算机科学调查论文生成的8个金标准集合，以及跨学科专家提出的查询和GitHub公开的评估数据集。

**📈 对比分析**

与Google Dataset Search和DataCite Commons对比，自动评估中平均归一化召回率达47.47%（最高81.82%），专家评估中相关性、实用性与可信度得分均显著高于基线。

**⚠️ 局限性**

局限性在于依赖已被引用的文献，难以捕获未被引用或新发布的数据集；对开放获取文本的偏倚导致某些学科覆盖不足；多语言和非英语研究社区的数据被低估。

---

## SemPA: Improving Sentence Embeddings of Large Language Models through Semantic Preference Alignment

**arXiv ID:** 2601.05075 | [PDF](https://arxiv.org/pdf/2601.05075v1)

**作者:** Ziyang Chen `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 20988 | **OpenAlex IDs:** https://openalex.org/A5100684575

**关键词:** `Computation and Language` `Generation` `Optimization` `Representation Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出 SemPA 方法，通过在大语言模型上执行语义偏好对齐（使用 DPO）进行句子级微调，得到更优的句子嵌入并保持原有的生成能力。

**💡 创新点**

创新点在于：① 将直接偏好优化（DPO）与传统对比学习统一到 Plackett–Luce 框架；② 通过 NLI 数据构造伪偏好对齐样本，避免昂贵的人工标注；③ 在不改动模型结构的前提下，用轻量化 LoRA 微调实现语义提升。

**🔧 技术方法**

使用技术包括：大语言模型（LLaMA2/3）+ LoRA 微调、Direct Preference Optimization (DPO)、PromptEOL 提取句子向量、Plackett–Luce 统一视角、NLI 数据生成偏好对齐。

**📊 数据集**

训练数据：NLI 数据集（约 40–80k 训练样本）；评估数据：Semantic Textual Similarity（STS‑2012~2016、STS‑B、SICK‑R）以及生成能力基准（GSM8K、MMLU、HellaSwag、DROP、TruthfulQA）。

**📈 对比分析**

与多种基线（BERT‑avg、Sentence‑T5‑avg、PromptBERT、SBERT、LLM2Vec、Echo Embedding、PromptEOL、Contrastive Prompting、Token Prepending）进行对比；SemPA 在 7 个 STS 数据集上平均提升约 3–4 %（大部分数据集领先），并在生成任务上保持或略有提升（对比学习往往导致生成性能下降）。同时，在嵌入空间的均匀度与等向性指标上表现更好。

**⚠️ 局限性**

局限性：仅采用二元 DPO，未利用列表式排序的更丰富信息；实验规模受限，数据量与质量可进一步提升；模板选择仍需人工设计，缺乏自动化优化。

---

## Milestones over Outcome: Unlocking Geometric Reasoning with Sub-Goal Verifiable Reward

**arXiv ID:** 2601.05073 | [PDF](https://arxiv.org/pdf/2601.05073v1)

**作者:** Jianlong Chen `[一作]` (Chinese University of Hong Kong), Renqiu Xia `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 20 | **OpenAlex IDs:** https://openalex.org/A5113021225

**关键词:** `Machine Learning` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Large Language Model` `Multimodality` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

构建了GeoGoal基准，将形式化几何证明拆解为可验证的数值子目标，并提出SGVR框架，让多模态大型语言模型在子目标层面接受强化学习；

**💡 创新点**

创新点在于把几何推理拆成可自动检验的子目标，并以子目标完成率为稠密奖励，解决传统仅评估最终答案导致的推理不可靠问题；

**🔧 技术方法**

采用TrustGeoGen数据引擎生成形式化解题骨架，映射为数值子目标；利用子目标可验证奖励（Skeleton Rate）与Group Relative Policy Optimization（GRPO）进行强化学习；

**📊 数据集**

使用GeoGoal（基于TrustGeoGen生成的几何问题与子目标），并在外部几何、通用数学（AMC、MATH-500）及推理基准（LiveBench-Reasoning、VisuLogic）上进行评估；

**📈 对比分析**

通过与多种闭源/开源MM-LLM基线在SR、SC、CR和FA指标比较，SGVR在几何任务提升约+9.7%，在通用数学+8.0%，在通用推理+2.8%，并显著提高子目标一致性和过程评估分数；

**⚠️ 局限性**

局限性在于仅支持数值子目标验证，无法覆盖非数值或更复杂的符号推理；依赖单一形式化引擎，缺乏对更广泛数学或图形论证的覆盖。

---

## Large language models can effectively convince people to believe conspiracies

**arXiv ID:** 2601.05050 | [PDF](https://arxiv.org/pdf/2601.05050v1)

**作者:** Thomas H. Costello `[一作]` (Carnegie Mellon University), Gordon Pennycook `[通讯]` (Cornell University)

**通讯引用:** 31550 | **OpenAlex IDs:** https://openalex.org/A5020533147

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

通过与GPT‑4o进行三轮预注册实验，研究了让LLM支持或驳斥用户自认为“未确定”的阴谋论时对其信念的说服效果。

**💡 创新点**

首次揭示LLM在推崇与驳斥阴谋论时的说服力度几乎对称，且在默认与越狱模型上表现相同；同时验证了“真相约束”提示和纠正对策能显著削弱欺骗性说服。

**🔧 技术方法**

使用GPT‑4o大型语言模型、APE说服评估器、Perplexity Sonar 进行自动事实检查以及文本嵌入+DBSCAN进行阴谋主题聚类。

**📊 数据集**

收集了2744名美国受试者的自选阴谋论文本与信念评估，构成实验数据集，并通过模型自动化处理生成对话与事实评分。

**📈 对比分析**

在不加约束的情况下，Bunking 与 Debunking 的平均效应均约为 ±12‑13 分（g≈±1），两者无显著差异；加上真相约束后，Bunking 效果降至约 4‑5 分（g≈0.2），Debunking 维持 ≈12 分；纠正阶段能将Bunking后信念降至低于基线。

**⚠️ 局限性**

局限包括样本仅为美国成年人、仅检验阴谋论的信念变化、模型的自我生成回答可能不完全代表实际使用情景，以及对多元文化与非阴谋话题的适用性未知。

---

## How to Set the Learning Rate for Large-Scale Pre-training?

**arXiv ID:** 2601.05049 | [PDF](https://arxiv.org/pdf/2601.05049v1)

**作者:** Yunhua Zhou `[一作]` (Shanghai AI Laboratory), Qipeng Guo `[通讯]` (Shanghai AI Laboratory)

**关键词:** `Artificial Intelligence` `Optimization` `Mixture of Experts` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

本文提出两种大规模预训练学习率配置范式——拟合与迁移，并通过实验验证拟合范式在12B MoE模型上优于μTransfer。

**💡 创新点**

创新点在于将尺度律应用于学习率拟合，将μTransfer扩展至MoE架构、深度、权重衰减与token horizon，并首次在10倍规模预训练中进行对比。

**🔧 技术方法**

采用尺度律、μTransfer、MoE（Qwen3-MoE）架构、AdamW优化器、Warmup‑Stable‑Decay学习率调度，以及数值拟合与梯度幅值分析等技术。

**📊 数据集**

使用InternLM2.5综合语料（文本、代码、长上下文）进行预训练，评估数据采用MMLU与CMMLU。

**📈 对比分析**

在相同训练步骤与数据量下，将Fitting范式得到的全局最优学习率与基于μTransfer迁移的学习率进行对比，结果显示Fitting在MMLU和CMMLU上提升约1–2个百分点。

**⚠️ 局限性**

局限性包括仅针对WSD调度和MoE架构，未验证其他调度器或稠密模型的适用性，且未探究两范式的极限推断边界。

---

## DeepWeightFlow: Re-Basined Flow Matching for Generating Neural Network Weights

**arXiv ID:** 2601.05052 | [PDF](https://arxiv.org/pdf/2601.05052v1)

**作者:** Saumya Gupta `[一作]` (Northeastern University), Ayan Paul `[通讯]` (Northeastern University)

**通讯引用:** 2346 | **OpenAlex IDs:** https://openalex.org/A5077110311

**关键词:** `Machine Learning` `Generation` `Optimization` `Flow-based Model` `Image` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于 Flow Matching 的完整神经网络权重生成模型，能够直接在高维权重空间生成无须微调的高性能网络

**💡 创新点**

创新点在于：1) 用 Flow Matching 取代扩散模型，实现更快采样；2) 通过 Git Re‑Basin 和 TransFusion 对权重进行规范化，处理对称性；3) 结合 PCA 使模型可扩展至 1 亿参数级别；4) 训练数据来自多随机初始化终止网络，保证权重多样性

**🔧 技术方法**

采用 Flow Matching、MLP 作为速度场网络、Git Re‑Basin/TransFusion 规范化、BatchNorm 重新校准、增量/双重 PCA 降维、Runge‑Kutta 采样

**📊 数据集**

使用 MNIST、Fashion‑MNIST、CIFAR‑10、STL‑10、SVHN、Yelp、BERT‑118M 等常见视觉、表格、NLP 数据集进行训练与评估

**📈 对比分析**

与 RPG、D2NWG、P‑diff、FLoWN、SANE 等现有权重生成方法对比，生成模型在不需微调的情况下达到或超过同类模型性能；采样速度提升数百倍，训练时间从数小时降至数分钟；在迁移学习、零样本推理上也表现优异

**⚠️ 局限性**

局限性包括：对中等维度权重的规范化收益有限；对极大网络仍需 PCA 降维，可能影响细粒度特征；目前仅支持单一任务/架构，跨任务、多架构联合生成尚未成熟

---

## Atlas 2 -- Foundation models for clinical deployment

**arXiv ID:** 2601.05148 | [PDF](https://arxiv.org/pdf/2601.05148v1)

**作者:** Maximilian Alber `[一作]` (Aignostics), Andrew Norgan `[通讯]` (Machine Learning Group, Technische Universität Berlin)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Segmentation` `Optimization` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Contrastive Learning` `Image` `Biomedical Data`

### 📋 论文摘要

**🎯 论文内容**

本文提出Atlas 2系列病理基础模型（Atlas 2、Atlas 2-B、Atlas 2-S），旨在同时提升预测性能、鲁棒性和资源效率。

**💡 创新点**

创新点包括：①使用截至目前最大的5.5百万WSI训练集；②在ViT结构上进行大规模自监督预训练，并通过蒸馏得到轻量化版本；③兼顾多尺度切片、染色和扫描仪差异，显著提升模型泛化能力。

**🔧 技术方法**

技术上采用Vision Transformer（ViT）架构，Patch‑token大小为8，结合DINOv2/DINOv3自监督学习框架，使用CLS+MEAN表征进行下游任务。

**📊 数据集**

数据集为来自Charité‑Universitätsmedizin Berlin、LMU Munich和Mayo Clinic的5.5 million未标记整片切片（WSI），涵盖0.25、0.5、1.0、2.0 µm/像素等多倍放大与多种染色/扫描条件。

**📈 对比分析**

通过80个公开基准（eva、HEST、PathoROB、Plismbench、Patho‑Bench等）评估，Atlas 2在22/27任务获得最优，Atlas 2‑B/​S在对应规模类别中同样取得最高或第二高成绩；鲁棒性平均得分提升约30%，且在资源效率上比原始Atlas提升3.4–9倍。

**⚠️ 局限性**

限制包括：对极少样本或罕见病症的泛化仍有限；生存预测任务所有模型表现接近随机（C‑Index≈50）；部分评测缺乏公开权重或完整数据，导致无法与所有模型完整对比。

---

## How Human is AI? Examining the Impact of Emotional Prompts on Artificial and Human and Responsiveness

**arXiv ID:** 2601.05104 | [PDF](https://arxiv.org/pdf/2601.05104v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computation and Language`

---

## Re-Align: Structured Reasoning-guided Alignment for In-Context Image Generation and Editing

**arXiv ID:** 2601.05124 | [PDF](https://arxiv.org/pdf/2601.05124v1)

**作者:** Runze He `[一作]` (Hunyuan, Tencent), Jiao Dai `[通讯]` (Institute of Intelligent Engineering, Chinese Academy of Sciences)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Reinforcement Learning` `Chain-of-Thought` `Rectified Flow` `Image` `Text` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出了Re-Align框架，利用结构化推理（IC-CoT）实现图像生成与编辑的统一处理。

**💡 创新点**

创新点包括：①将语义指导与参考关联解耦的IC-CoT结构；②使用结构化推理作为推理-生成对齐目标；③结合GRPO强化学习与CLIP相似度奖励提升推理一致性。

**🔧 技术方法**

技术手段包括多模态基础模型、In-Context Chain-of-Thought、Rectified Flow图像生成、GRPO强化学习、CLIP相似度代理奖励和思路诱导多样性策略。

**📊 数据集**

使用了自建的-410K ICGE数据集，涵盖多种生成/编辑任务，并借助Gemini 2.5和GPT-4o生成推理文本与目标图像。

**📈 对比分析**

在OmniContext和DreamOmni2Bench基准上，与同规模的OmniGen2、Echo-4o、Qwen-Image-Edit、DreamOmni2等模型对比，Re-Align在整体平均分、PF/SC分数上名列前茅。

**⚠️ 局限性**

局限性在于模型规模和数据量仍低于GPT-4o等商业系统，IC-CoT仅为文本级，未扩展到视觉链式思考，且在极端复杂或缺乏专门训练的编辑场景下仍易出现错误。

---

## Nalar: An agent serving framework

**arXiv ID:** 2601.05109 | [PDF](https://arxiv.org/pdf/2601.05109v1)

**作者:** Marco Laju `[一作]` (University of Texas at Austin), Aditya Akella `[通讯]` (University of Texas at Austin)

**通讯引用:** 14731 | **OpenAlex IDs:** https://openalex.org/A5035329776

**关键词:** `Distributed, Parallel, and Cluster Computing` `Optimization` `Computational Efficiency` `Agentic AI` `Recommendation System` `Data-Centric Learning` `Large Language Model` `Text` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

开发了一套 Agentic Workflows 的服务器框架，提供轻量化 Future 抽象、逻辑与物理状态分离的状态层，以及全局与本地双层控制器，支持动态、状态化多代理工作流的高效执行。

**💡 创新点**

创新点在于：1）自动生成代理/工具桩将调用转为携带元数据的 Future，给予运行时可观测与控制；2）引入两层控制架构，结合全局视图与本地事件驱动，实现自适应路由、迁移和资源调度；3）提供可编程策略接口，使开发者无需改写工作流即可实现多种调度与资源管理策略。

**🔧 技术方法**

技术栈包括 Python、gRPC、Redis（节点存储）、vLLM+LMCache（LLM 与 KV 缓存管理）、ChromaDB（向量检索）以及自研的 Future、状态层与两层调度器实现。

**📊 数据集**

实验使用的公开数据集有：Financial Analyst 采用 FinQA；Router‑based workflow 采用 Microsoft Azure LLM 追踪数据；Software Engineering Workflow 采用 SWE‑bench；此外在多节点 GPU 环境下进行性能评测。

**📈 对比分析**

与 Ayo、CrewAI、AutoGen 等基线在三类工作流上对比，结果显示 tail 延迟（P95–P99）下降 34–74%，平均延迟提升 8–35%，能够在 80 RPS 维持稳定（基线已失效）；通过动态资源调度与迁移实现最高 2.9× 的端到端加速，显著降低 head‑of‑line 阻塞。

**⚠️ 局限性**

局限性包括：1）不支持容错与自动恢复，仅向驱动程序报告失败；2）状态层与批量化代理冲突，无法同时使用；3）缺乏大规模 GPU 实际测试，评估多基于模拟；4）对 KV 缓存的生命周期管理仍依赖全局视图，可能在极端负载下出现瓶颈。

---

## UniLiPs: Unified LiDAR Pseudo-Labeling with Geometry-Grounded Dynamic Scene Decomposition

**arXiv ID:** 2601.05105 | [PDF](https://arxiv.org/pdf/2601.05105v1)

**作者:** Filippo Ghilotti `[一作]` (TORC Robotics), Felix Heide `[通讯]` (Princeton University)

**通讯引用:** 6224 | **OpenAlex IDs:** https://openalex.org/A5059313827

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Object Detection` `Depth Estimation` `Autonomous Driving` `Transformer` `Simultaneous Localization and Mapping` `Point Cloud` `Image`

### 📋 论文摘要

**🎯 论文内容**

在单一驾驶轨迹中，利用LiDAR、摄像头和IMU数据，通过几何一致性无监督方式生成三维语义标签、3D边界框以及精准长距离深度图。

**💡 创新点**

提出统一的3D标签生成框架，结合SLAM地图、视觉基础模型以及几何一致性迭代更新，能够在没有人工标签的情况下同时完成语义分割、目标检测和深度估计。

**🔧 技术方法**

使用LiDAR‑ODOM+IMU SLAM、运动分割网络、OneFormer、SAM2、BLIP、CLIP/CLIPSeg进行2D伪标签生成；通过概率标签传播、迭代加权更新（IWU）和自适应球面遮挡剔除实现3D标签的精细化。

**📊 数据集**

在KITTI、nuScenes和自建长距离高速公路数据集上进行评测。

**📈 对比分析**

与专门的伪标签方法和oracle标签对比，语义分割mIoU提升至约68%（相当于oracle），检测mAP提升至31%（相对基线21%），深度MAE在80–150 m和150–250 m区间分别下降51.5%和22.0%，整体表现接近人工标注。

**⚠️ 局限性**

依赖SLAM构图精度，若地图不精确会导致标签漂移；对动态物体分离依赖运动分割初始误差；计算量大，需多模型推理与点云配准；目前仅支持单轨迹或静态场景的长距离评估，未覆盖多场景通用性。

---

## Challenges and Research Directions for Large Language Model Inference Hardware

**arXiv ID:** 2601.05047 | [PDF](https://arxiv.org/pdf/2601.05047v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Hardware Architecture`

---

## Semantically Orthogonal Framework for Citation Classification: Disentangling Intent and Content

**arXiv ID:** 2601.05103 | [PDF](https://arxiv.org/pdf/2601.05103v1)

**作者:** Changxu Duan `[一作]` (Technische Universitaet Darmstadt), Zhiyin Tan `[通讯]` (L3S Research Center Leibniz University Hannover)

**关键词:** `Digital Libraries` `Classification` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了SOFT框架，拆分引用意图与被引内容类型，实现两维语义正交的引文注释标准；

**💡 创新点**

创新点在于将引用意图与被引内容类型分离并采用语义角色理论定义清晰的7种意图与3种内容类型，显著提升了注释一致性与模型泛化；

**🔧 技术方法**

采用大语言模型（Qwen、Llama、Mistral、Gemma）进行零样本与微调分类，并使用SciBERT、Qwen‑Small等模型进行评估；

**📊 数据集**

使用重新注释的ACL‑ARC数据集（1931条）和跨学科ACT2子集（264条）进行训练与测试；

**📈 对比分析**

通过LLM人机一致性、准确率和宏F1对比，SOFT在意图维度实现了0.66-0.69的宏F1，显著优于ACL‑ARC（0.51）与SciCite（0.62），且在跨域测试中F1下降仅15.8%，优于其他框架；

**⚠️ 局限性**

局限在于框架仍为单标签设计，未覆盖多重引用情形，且对非英语文本与更细粒度的功能类别支持仍待扩展。

---

## Compensation Effect Amplification Control (CEAC): A movement-based approach for coordinated position and velocity control of the elbow of upper-limb prostheses

**arXiv ID:** 2601.05074 | [PDF](https://arxiv.org/pdf/2601.05074v1)

**作者:** Julian Kulozik `[一作]` (Sorbonne Université), Nathanaël Jarrassé `[通讯]` (Sorbonne Université)

**通讯引用:** 2224 | **OpenAlex IDs:** https://openalex.org/A5057665161

**关键词:** `Robotics` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

本研究提出并验证了一种基于躯干运动的“补偿效应放大控制（CEAC）”，用于上肢义肢肘关节的连续位置和速度控制，并在健康受试者的绘图与三维到达任务中进行评估。

**💡 创新点**

创新点在于引入动态参考姿态，允许躯干角度在受控延迟后作为功能性输入，放大自然的躯干-肘关节耦合，而非简单消除补偿运动，从而实现更直观、连续的速度调节。

**🔧 技术方法**

技术手段包括：低通滤波动态参考、死区阈值、比例控制法则、VIVE® 追踪器测量躯干姿态、Raspberry Pi与自制义肢驱动电机、Python/UDP实现实时通讯。

**📊 数据集**

数据集来源于12名健康受试者在可变速度/尺寸绘图任务（24次/人）与9目标到达任务（10人）中采集的三维运动学和关节角度序列。

**📈 对比分析**

通过与受试者自身自然肘关节的对比，采用完成时间、轨迹精度、路径长度比（PLR）、光谱弧长（SPARC）、关节范围/累计运动量、协同关节速度指数（SJVI）等指标评估；结果显示在大多数条件下CEAC与自然手臂的性能相近，速度较高时完成时间略长但精度差异不显著。

**⚠️ 局限性**

局限性包括：仅在健康受试者上验证，未评估截肢者；控制参数固定，缺乏自适应机制；仅针对单关节（肘）控制；对极慢速度运动失效；未来需与EMG控制结合并扩展至多关节与更自然任务。

---

## From Understanding to Engagement: Personalized pharmacy Video Clips via Vision Language Models (VLMs)

**arXiv ID:** 2601.05059 | [PDF](https://arxiv.org/pdf/2601.05059v1)

**作者:** Suyash Mishra `[一作]` (Roche), Anubhav Girdhar `[通讯]` (Involead)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Optimization` `Computational Efficiency` `Transformer` `Vision Language Model` `Prompt Engineering` `Video` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

开发了一种基于 ALM 与 VLM 的视频到视频高亮剪辑生成框架，支持长时医学视频的高效提取和个性化剪辑。

**💡 创新点**

提出可重复的 Cut & Merge 算法实现淡入淡出与时间戳归一化，结合角色定义与 Prompt 注入实现多场景定制化剪辑，并实现成本与时效显著下降。

**🔧 技术方法**

使用 Whisper V2/V3 语音转写、Gemini 2.5 Pro/Flash VLM 进行文本与视觉理解、Prompt 生成、FFmpeg/MoviePy 处理、端到端管线优化。

**📊 数据集**

在公开 Video‑MME 900 视频基准和自有 16,159 条跨 14 病种、超 3 小时的医药视频数据上评估。

**📈 对比分析**

与 Gemini 2.5 Pro/Flash、Video‑MME 上的 SOTA VLM 进行对比，速度提升 3–4 倍，成本降低 4 倍，片段连贯度 0.348、信息量 0.721，整体剪辑质量保持竞争。

**⚠️ 局限性**

主要局限于依赖现有 VLM/ALM，缺乏对非医疗领域的验证，且对极长视频仍需改进算法鲁棒性与跨域通用性。

---

## An Invitation to "Fine-grained Complexity of NP-Complete Problems"

**arXiv ID:** 2601.05044 | [PDF](https://arxiv.org/pdf/2601.05044v1)

**作者:** Jesper Nederlof `[一作]`, Jesper Nederlof `[通讯]`

**关键词:** `Data Structures and Algorithms` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

本文综述了针对NP完备问题的细粒度复杂性研究，系统回顾了从传统的暴力求解到最新的高效算法、硬性下界、参数化与随机化技术的演变。

**💡 创新点**

创新点在于将细粒度复杂性（Fine‑Grained Complexity）与参数化复杂性、随机化技术以及代数方法等多学科工具相结合，揭示了许多经典问题（如k‑CNF‑Sat、Hamiltonian Cycle、Subset Sum等）在时间指数方面的精细边界，并提供了统一的分析框架和新的下界构造。

**🔧 技术方法**

主要技术包括：
- 随机限制（Random Restrictions）与切换引理（Switching Lemma）
- 代数计数与行列式/拉普拉斯矩阵技巧
- 组合数列与集合覆盖容器（Containers）
- 变换矩阵（Kronecker / Zeta–Möbius 变换）
- 分支与测度-征服（Measure & Conquer）
- 近似与随机化哈希
- 伪多项式时间与子集求和的“Representation”方法
- 量子加速与Grover搜索框架

**📊 数据集**

由于本文为综述性质，未使用实验数据集；引用的实验比较多基于理论上已知的算法运行时间和指数常数（如2^n、1.5^n、1.89^n等），并引用了公开的实验结果（例如TSP、Hamiltonian Cycle的实际实现），但核心仍为理论分析。

**📈 对比分析**

本文通过比较经典的暴力算法与改进算法的指数常数，展示了大幅提升，例如：
- k‑CNF‑Sat 从 2^n 降至 2^(1−1/O(k))^n
- Hamiltonian Cycle 从 2^n 降至 1.66^n
- Subset Sum 从 2^n/2 降至 2^0.45n（在随机实例假设下）
- Set Cover 从 2^n 降至 2^(1−ε)^n（利用容器方法）
- 通过代数计数实现的 Hamiltonian Cycle 的 3^n/2 速度提升

这些结果均在理论上证明了指数常数的可改进，并通过与现有上界/下界对比证明其最优性或接近最优性。

**⚠️ 局限性**

限制主要包括：
- 对于多数问题仍缺乏严格的下界，尤其是是否存在 2^(1−ε)^n 的上界仍未决。
- 许多改进基于随机化或特殊实例（如随机权重），在确定性或最坏情况下仍未实现。
- 复杂度分析高度依赖于强假设（如ETH、SETH），若假设不成立则结论不稳固。
- 某些算法仅在理论上实现，实际实现中常面临常数因子与空间消耗过大。
- 量子加速等新范式尚未在所有 NP‑完备问题中得到充分探索。

---

## FinDeepForecast: A Live Multi-Agent System for Benchmarking Deep Research Agents in Financial Forecasting

**arXiv ID:** 2601.05039 | [PDF](https://arxiv.org/pdf/2601.05039v1)

**作者:** Xiangyu Li `[一作]` (National University of Singapore), Ke-Wei Huang `[通讯]` (Asian Institute of Digital Finance)

**关键词:** `Multiagent Systems` `Recommendation System` `Optimization` `Anomaly Detection` `Finance Related` `Large Language Model` `Agentic AI` `Time Series` `Tabular` `Benchmark` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

构建了一个实时、端到端的多智能体系统，持续生成金融预测任务并自动评估深度研究代理的表现。

**💡 创新点**

首次提出双轨分类法和动态任务生成，实现无污染、时序隔离的金融预测基准，并提供完整自动化评估管道。

**🔧 技术方法**

采用多智能体架构，包括数据采集、任务生成、预测调用和评估代理，利用LLM、搜索与深度研究代理进行推理与信息检索。

**📊 数据集**

使用全球8个主要经济体（US、CN、HK、JP、UK、DE、FR、SG）和1314家上市公司公开财报、宏观指标、新闻与行情数据，生成十周的任务集。

**📈 对比分析**

对13种方法（3深度研究、5仅思考、5思考+搜索）进行对比，深度研究模型取得最高准确率（非循环任务≈80%，循环≈25%），思考+搜索略优于仅思考，但仍落后深度研究。

**⚠️ 局限性**

仍面临循环数值预测准确率低、对多步/概率预测支持不足、市场覆盖有限以及缺乏过程级评估等局限。

---

## LooseRoPE: Content-aware Attention Manipulation for Semantic Harmonization

**arXiv ID:** 2601.05127 | [PDF](https://arxiv.org/pdf/2601.05127v1)

**作者:** Etai Sella `[一作]` (Tel Aviv University), Or Patashnik `[通讯]` (Tel Aviv University)

**通讯引用:** 2594 | **OpenAlex IDs:** https://openalex.org/A5076541595

**关键词:** `Graphics` `Image Harmonization` `Image Translation` `Object Detection` `Vision Language Model` `Diffusion model` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出了一个无提示的图像编辑框架LooseRoPE，直接对剪切粘贴的对象进行语义和视觉融合，保持对象身份和上下文一致性。

**💡 创新点**

创新点在于将旋转位置编码（RoPE）与显著性导向相结合，动态调节注意力视野，实现对保留身份和融入背景的平衡，并通过VLM实现自适应参数调节。

**🔧 技术方法**

技术手段包括使用FLUX Kontext扩散模型、基于实例检测网络的显著性估计、RoPE逆尺度因子调节、注意力权重缩放以及VLM驱动的自动参数调整。

**📊 数据集**

使用了新构建的150张含跨图像或同图像剪裁的合成与自然图像组合数据集，并参考了SSH、Cross-Domain Compositing等公开数据集。

**📈 对比分析**

与TF-ICON、AnyDoor、SwapAnything、原始Flux Kontext等基线对比，采用CLIP‑IQA和LPIPS指标以及用户研究，LooseRoPE在保持身份与图像质量之间取得了更佳平衡，整体性能优于大多数基线。

**⚠️ 局限性**

局限性包括对小剪裁或高差异背景仍易出现抑制/忽略，VLM调节增加额外推理成本，未在视频或多对象场景验证，且对显著性估计的鲁棒性有限。

---

## Advanced Multimodal Learning for Seizure Detection and Prediction: Concept, Challenges, and Future Directions

**arXiv ID:** 2601.05095 | [PDF](https://arxiv.org/pdf/2601.05095v1)

**作者:** Ijaz Ahmad `[一作]` (Shenzhen University), Baiying Lei `[通讯]` (Shenzhen University)

**通讯引用:** 11741 | **OpenAlex IDs:** https://openalex.org/A5001212991

**关键词:** `Neural and Evolutionary Computing` `Classification` `Anomaly Detection` `Explainability and Interpretability` `Federated Learning` `Convolutional Neural Network` `Recurrent Neural Network` `Graph Neural Network` `Transformer` `Supervised Fine-Tuning` `Transfer Learning` `Semi-Supervised Learning` `Attention Mechanism` `Multimodality` `Biomedical Data` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

本文综述并系统阐述了用于癫痫发作检测与预测的高级多模态学习方法，涵盖了从传统EEG到心电、肌电、皮肤电、光电容积描记、神经影像与视频监测等多种信号的融合与处理流程。

**💡 创新点**

创新点主要包括：①提出了三层融合框架（数据级、特征级、决策级）及其动态加权策略；②结合注意力机制、Transformer与图卷积网络实现跨模态时空特征学习；③强调可解释性与边缘计算实现实时、可穿戴的临床部署；④系统性评估多模态数据集与未来研究方向。

**🔧 技术方法**

技术手段涵盖：深度学习（CNN、LSTM、Transformer、GCN）、迁移学习与领域自适应、半监督/自监督学习、注意力与多头注意力机制、图卷积、模型压缩与量化、边缘推理与联邦学习、可解释AI（SHAP、Grad‑CAM 等）。

**📊 数据集**

所引用的数据集包括公开 EEG 数据集（如 CHB‑MIT、Bonn、EPILEPSIAE）、多模态融合数据（EEG+EMG+ACC+EDA+HR、EEG+PPG+ACC+EDA 等）以及临床影像与视频监测数据，且在多项实验中对比了单模态与多模态的性能。

**📈 对比分析**

在多模态实验中，作者报告多模态模型在检测灵敏度上提升 5–15%（例如 98% 以上），误报率下降 30–50%（如从 1.0 次/小时降至 0.1 次/小时），预测窗口可延长至 60 分钟以上；与传统单模态或基线机器学习模型相比，整体性能提升显著。

**⚠️ 局限性**

局限性包括：①多模态数据集规模有限、标注成本高，导致模型泛化受限；②异构信号的同步与校准困难，融合时易产生噪声与信息稀释；③深度模型计算量大，边缘设备部署仍面临能耗与实时性挑战；④缺乏足够的临床验证与监管审批路径，解释性仍不够充分。

---

## Driving on Registers

**arXiv ID:** 2601.05083 | [PDF](https://arxiv.org/pdf/2601.05083v1)

**作者:** Ellington Kirby `[一作]` (Valeo), Matthieu Cord `[通讯]` (Sorbonne Université)

**通讯引用:** 9085 | **OpenAlex IDs:** https://openalex.org/A5108118084

**关键词:** `Computer Vision and Pattern Recognition` `Autonomous Driving` `Transformer` `Supervised Fine-Tuning` `Image` `Video`

### 📋 论文摘要

**🎯 论文内容**

构建了一个简洁的纯Transformer架构，利用ViT预训练与注册标记压缩多摄像头特征，并用两阶段解码器生成并评分候选轨迹，实现端到端驾驶。

**💡 创新点**

创新点在于：1) 采用摄像头感知注册标记实现高效特征压缩；2) 轨迹生成与评分解码器解耦；3) 通过分项评分实现行为可控；4) 在不使用复杂中间表示和大词典的情况下达到SOTA。

**🔧 技术方法**

使用ViT-S DINOv2预训练视觉编码器、LoRA微调、Transformer解码器、WTA回归、oracle评分及子分数训练。

**📊 数据集**

主要使用NAVSIM-v1、NAVSIM-v2、HUGSIM三大模拟评测数据集；摄像头输入为前、左前、右前、后四视角。

**📈 对比分析**

与GTRS、UniAD等现有方法对比，在NAVSIM-v1/2、HUGSIM上均取得或接近人类水平，显著提升PDMS/EPDMS分数，同时模型参数约40M、推理速度提升3倍。

**⚠️ 局限性**

局限性包括：依赖模拟/合成数据，尚未在真实车辆上验证；仅使用摄像头信息，缺乏雷达/激光等多模态；对极端场景或长期驾驶的鲁棒性未知。

---

## Supporting Secured Integration of Microarchitectural Defenses

**arXiv ID:** 2601.05057 | [PDF](https://arxiv.org/pdf/2601.05057v1)

**作者:** Kartik Ramkrishnan `[一作]`, Pen-Chung Yew `[通讯]`

**关键词:** `Cryptography and Security`

### 📋 论文摘要

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

## Exploring Student Expectations and Confidence in Learning Analytics

**arXiv ID:** 2601.05082 | [PDF](https://arxiv.org/pdf/2601.05082v1)

**作者:** Hayk Asatryan `[一作]` (Bochum University of Applied Sciences), Jörg Frochte `[通讯]` (Bochum University of Applied Sciences)

**通讯引用:** 137 | **OpenAlex IDs:** https://openalex.org/A5027576910

**关键词:** `Machine Learning` `Explainability and Interpretability` `Safty and Privacy` `Tabular` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

使用改编后的 12 项 SELAQ 问卷对 553 名来自不同专业的学生进行匿名调查，提取“期望”和“渴望”两项得分，运用 K-means 聚类（k=4）划分为 Enthusiasts、Realists、Cautious、Indifferent 四类，并用 CART 决策树解释各类特征。

**💡 创新点**

创新点在于将 SELAQ 与无监督聚类相结合，首次揭示学生对学习分析与数据保护的多维态度，并通过决策树实现对聚类结果的可解释性分析，同时考虑专业差异对态度的影响。

**🔧 技术方法**

技术手段包括数据预处理（均值/众数填补）、K-means 聚类、轮廓系数/肘部法判定 k、CART 决策树可视化及描述性统计。

**📊 数据集**

数据集为 553 名在北莱茵-威斯特法伦州博赫姆应用科学大学就读的学生的问卷回复，涵盖 CS、CE、AR、EM、SU、SV、BS、OE 等专业，使用 12 项 SELAQ 的两重评分。

**📈 对比分析**

通过轮廓系数和肘部图确认最佳 k=4；聚类结果清晰区分四类学生，决策树以 4.6/4.8 等阈值实现高可解释性分割，未与传统方法做性能对比，但聚类与决策树的结合有效揭示态度差异。

**⚠️ 局限性**

局限性包括样本量在某些专业（如 Sustainability）偏小导致统计不稳、仅涵盖技术与工程类学生，缺乏人文社科视角，且调查仅捕捉主观认知，未结合实际 LA 系统实施与效果评估。

---

## Publishing FAIR and Machine-actionable Reviews in Materials Science: The Case for Symbolic Knowledge in Neuro-symbolic Artificial Intelligence

**arXiv ID:** 2601.05051 | [PDF](https://arxiv.org/pdf/2601.05051v1)

**作者:** Jennifer D'Souza `[一作]` (TIB Leibniz Information Centre for Science and Technology), Erwin Kessels `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 25653 | **OpenAlex IDs:** https://openalex.org/A5018346857

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Tabular` `Review/Survey Paper` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

本文将19篇材料科学ALD/E综述中的表格手工转换为可查询的ORKG比较，并基于这些结构化数据构建了33个自然语言问题与对应的SPARQL查询；随后评估了三类LLM（ChatGPT、Gemini、Claude以及多款开源模型）在PDF原文、CSV表格及RAG等设置下回答这些问题的性能。

**💡 创新点**

创新点在于首次将综述表格转化为 FAIR、机器可操作的知识图谱，并在此基础上开展符号查询与神经网络查询的对比，验证了符号层作为可靠神经符号混合推理基础的有效性；同时提供了完整的评测数据集与基准，促进了材料科学知识图谱与LLM交互的研究。

**🔧 技术方法**

主要技术包括：（1）使用 ORKG 平台创建结构化比较表（Comparisons）并导出为 CSV；（2）利用 SPARQL 进行确定性查询；（3）采用多种大语言模型（OpenAI ChatGPT、Google Gemini、Anthropic Claude、Gemma、Llama‑3、Qwen、Mistral 等）进行 PDF 原文推理、RAG 推理与 CSV‑上下文推理；（4）通过自定义自然语言查询模板和对齐工具实现语义映射。

**📊 数据集**

数据集为 18 个 ORKG 比较对象（来自 9 篇 ALD 综述和 11 篇 ALE 综述）以及对应的 33 个自然语言问题与 SPARQL 结果，公开存放于 GitHub（https://github.com/sciknoworg/ald-ale-orkg-review）和 ORKG 上。

**📈 对比分析**

实验对比显示：符号查询（SPARQL）始终给出最精确、可重复的答案；仅基于 PDF 的 LLM 查询在表格结构、数值准确性上表现欠佳；将 LLM 输入改为 CSV 表格（符号上下文）后，回答质量显著提升，但仍低于符号基准，且在复杂联合查询时误差累积更为明显。

**⚠️ 局限性**

主要限制包括：① 数据量仍有限，难以覆盖所有材料系统；② LLM 生成答案具有随机性与易错性（如表格错位、数值偏差、信息遗漏）；③ PDF 提取与分段过程会丢失布局信息，影响推理；④ 对于更复杂的多表联结与推理仍依赖人工构造的 SPARQL；⑤ 评测仅关注表格式问题，未涉及更广泛的文本推理与推测任务。

---

## Graph energy as a measure of community detectability in networks

**arXiv ID:** 2601.05065 | [PDF](https://arxiv.org/pdf/2601.05065v1)

**作者:** Lucas Böttcher `[一作]` (Frankfurt School of Finance and Management), Santo Fortunato `[通讯]` (Indiana University)

**通讯引用:** 37918 | **OpenAlex IDs:** https://openalex.org/A5053938061

**关键词:** `Social and Information Networks` `Graph Neural Network` `Graph`

### 📋 论文摘要

**🎯 论文内容**

研究网络社区可探测性阈值，发现图能量（图的邻接矩阵全谱绝对值之和）的差值在可探测阈值处出现跃迁，能够识别PPM网络与ER网络的可区分性

**💡 创新点**

首次证明仅使用邻接矩阵全谱即可通过图能量差检测社区可探测性阈值，而非传统的特征向量或非退化矩阵方法

**🔧 技术方法**

采用谱方法计算邻接矩阵全谱、图能量，结合Wigner半圆近似、Schatten 1‑范数理论和GPU加速的特征值求解（JAX/ PyTorch）

**📊 数据集**

在模拟PPM网络（两均等社区）中生成数据，网络规模N=500/1000，平均度k∈{5,10,20,50}，多组参数k_ab，比较对应ER网络的图能量

**📈 对比分析**

将PPM图能量与ER图能量之差ΔE与理论阈值2√k对比，实验显示ΔE≈0在阈值以下，阈值以上随社区差异下降；相较于第二大特征值λ2的平稳区，ΔE的过渡更清晰，验证了阈值检测能力

**⚠️ 局限性**

主要限制是计算成本高（O(N^3)），因此仅能处理N≤1000，缺乏对更大网络的验证；Wigner半圆近似在稀疏网络下失效，导致理论与实验出现偏差

---

## An interpretable data-driven approach to optimizing clinical fall risk assessment

**arXiv ID:** 2601.05194 | [PDF](https://arxiv.org/pdf/2601.05194v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## Chain-of-Sanitized-Thoughts: Plugging PII Leakage in CoT of Large Reasoning Models

**arXiv ID:** 2601.05076 | [PDF](https://arxiv.org/pdf/2601.05076v1)

**作者:** Arghyadeep Das `[一作]` (University of Massachusetts), Sharvi Endait `[通讯]`

**关键词:** `Artificial Intelligence` `Safty and Privacy` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Chain-of-Thought` `Text` `Biomedical Data` `Tabular` `Benchmark` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

研究链式推理（CoT）过程中的个人信息泄露问题，并系统评估通过提示工程（PE）和监督微调（SFT）实现“私有先推理”的效果。

**💡 创新点**

①提出首个带有隐私友好推理轨迹标注的 PII‑CoT‑Bench 数据集；②设计跨六类泄露场景的评估基准；③用可计算泄露率、归一化曝光和 LLM‑as‑Judge 评估指标，揭示不同模型在私有推理上的能力差异。

**🔧 技术方法**

使用 4‑bit 量化的开源推理模型（GPT‑OSS‑20B、Phi‑4、DeepSeek‑R1‑Qwen‑7B、LLaMA‑3.3‑70B、QwQ‑32B）在 LoRA 方式下做 SFT；设计严格的私有提示模板做 PE；通过 GPT‑4o‑mini 作为独立评判者给出隐私与实用性评分。

**📊 数据集**

①PII‑CoT‑Bench：350 条包含医学、金融场景的问答与隐私推理轨迹；②评估集：基于 GPT‑5.1 生成的六类隐私挑战式提示，覆盖实际与对抗泄露。

**📈 对比分析**

对比基线、SFT 与 PE 的泄露率、曝光度、隐私分数与实用性分数。结果表明：强模型（GPT‑OSS、Phi‑4 等）在 PE 下隐私分数显著提升，实用性几乎不变；弱模型（LLaMA‑3、DeepSeek‑R1‑Qwen）则需 SFT 才能显著降低泄露且对实用性影响微小；整体而言，私有推理可在几乎不牺牲性能的前提下降低 PII 泄露。

**⚠️ 局限性**

限制：①仅评估固定的 PII 类型与权重，可能不覆盖所有真实场景；②LLM‑as‑Judge 受模型偏差与校准问题影响；③使用量化模型与有限训练资源，可能无法充分展现潜在收益；④只关注中短推理轨迹，长链或多轮推理的泄露动态尚未深入；⑤对内部表示层的分析缺失，难以确定模型内部如何编码与抑制敏感信息。

---

## Concurrent Balanced Augmented Trees

**arXiv ID:** 2601.05225 | [PDF](https://arxiv.org/pdf/2601.05225v1)

**作者:** Evan Wrench `[一作]` (University of British Columbia), Yuanhao Wei `[通讯]` (University of British Columbia)

**通讯引用:** 215 | **OpenAlex IDs:** https://openalex.org/A5081860460

**关键词:** `Data Structures and Algorithms`

### 📋 论文摘要

**🎯 论文内容**

本文提出并实现了第一个无锁的平衡增广搜索树（Balanced Augmented Tree, BAT），实现了在并发环境下对顺序统计、聚合和区间查询的高效支持。

**💡 创新点**

创新点在于将 Fatourou‑Ruppert 的多版本增广技术扩展到 chromatic 树，并设计了委托传播机制以降低上根传播的冲突，从而显著提升并发性能；此外实现了轻量级的内存回收方案。

**🔧 技术方法**

使用技术包括 LLX/SCX 无锁原语、版本树多版本增广、委托传播（两种实现），以及 Epoch‑Based Reclamation（EBR）进行内存回收。

**📊 数据集**

实验使用 SetBench 微基准，键分布涵盖随机、排序、Zipfian，最大键值可达 10M，线程数最高 120，覆盖更新密集、查询密集和混合工作负载。

**📈 对比分析**

与 FR‑BST、Bundled CitrusTree、VcasBST、VerlibBTree 等竞争对手比较，BAT 在更新密集或包含秩/区间查询的工作负载中，比最优无增广树快 2–30 倍，区间查询性能可达 400 倍；在查询密集或高并发场景下吞吐量明显优于对手。

**⚠️ 局限性**

局限性包括：增广导致插入/删除的开销增加，导致单键小范围查询相对无增广树更慢；委托机制需要额外同步，且在极端高负载或阻塞情况下可能产生延迟。

---

## EARL: Energy-Aware Optimization of Liquid State Machines for Pervasive AI

**arXiv ID:** 2601.05205 | [PDF](https://arxiv.org/pdf/2601.05205v1)

**作者:** Zain Iqbal `[一作]` (National Research Council), Lorenzo Valerio `[通讯]` (National Research Council)

**通讯引用:** 381 | **OpenAlex IDs:** https://openalex.org/A5022139051

**关键词:** `Machine Learning` `Optimization` `Reinforcement Learning` `Recurrent Neural Network` `Reinforcement Learning` `Audio` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

提出了 EARL 框架，利用贝叶斯优化与强化学习相结合，并引入能耗奖励与早停机制，对液态状态机（LSM）的超参数进行能耗与准确率双目标优化。

**💡 创新点**

创新点在于：1）将贝叶斯优化产生的候选集合交由强化学习智能决策；2）在奖励函数中显式结合能耗与准确率；3）设计自适应早停阈值，动态停止无效搜索，显著降低计算与能耗。

**🔧 技术方法**

使用技术包括：Sobol 序列初始化、Gaussian Process 代理模型与 Expected Improvement 采样、epsilon‑greedy Q‑学习代理、GRU 读取层、以及能耗监测与早停判定。

**📊 数据集**

实验数据集为三大基准：Free Spoken Digit Dataset（语音）、Occupancy Detection（环境传感）、UCI Human Activity Recognition（动作）。

**📈 对比分析**

与 Optuna（NSGA‑II）和 Ray Tune（异步并行）对比，EARL 在所有数据集上实现了 6–15% 的准确率提升、60–80% 的能耗降低，并将优化时间压缩至 10–15 分钟，约为对手的十分之一到八分之一。

**⚠️ 局限性**

局限性：仅在软件模拟环境评估，缺乏真实 neuromorphic 硬件验证；搜索空间受限于四个超参数，未探索更深层架构；奖励系数 α 需要手动调节，可能影响跨任务的泛化。

---

## FaST: Efficient and Effective Long-Horizon Forecasting for Large-Scale Spatial-Temporal Graphs via Mixture-of-Experts

**arXiv ID:** 2601.05174 | [PDF](https://arxiv.org/pdf/2601.05174v1)

**作者:** Yiji Zhao `[一作]` (Yunnan University), Hao Wu `[通讯]` (Yunnan University)

**通讯引用:** 27250 | **OpenAlex IDs:** https://openalex.org/A5083170497

**关键词:** `Machine Learning` `Graph Neural Network` `Transformer` `Mixture of Experts` `Graph` `Time Series`

### 📋 论文摘要

**🎯 论文内容**

针对大规模、长时程空间-时序图预测，提出了FaST框架。

**💡 创新点**

创新点包括：1）基于混合专家的时间压缩输入模块，适应节点异质性；2）自适应图代理注意力，将O(N²)降为O(N·a)；3）并行GLU‑MoE实现高效特征变换。

**🔧 技术方法**

使用了混合专家（MoE）、门控线性单元（GLU）、自适应图代理注意力、残差Transformer结构以及MLP预测头。

**📊 数据集**

在大型交通流数据集LargeST（SD、GBA、GLA、CA）上进行评估，节点数从716到8600，时间粒度15分钟，预测时长至一周（672步）。

**📈 对比分析**

与多种基线（DLinear、NHITS、CycleNet、STGCN、GWNet、BigST、STID、PatchSTG、RPMixer等）对比，FaST在所有数据集、所有时长上均取得最低MAE/RMSE/MAPE，并且GPU内存占用和推理时间均比SOTA低30–50%，实现线性计算复杂度。

**⚠️ 局限性**

局限性：对极端稀疏或结构未知的图仍需进一步验证；模型在节点极端异质性场景下的专家分配可能仍出现轻微失衡；并且对训练时间仍依赖GPU算力。

---

## Inside Out: Evolving User-Centric Core Memory Trees for Long-Term Personalized Dialogue Systems

**arXiv ID:** 2601.05171 | [PDF](https://arxiv.org/pdf/2601.05171v1)

**作者:** Jihao Zhao `[一作]` (MemTensor Technology Co., Ltd.), Zhiyu li `[通讯]`

**关键词:** `Computation and Language` `Reinforcement Learning` `Recommendation System` `Transformer` `Large Language Model` `Reinforcement Learning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出Inside Out框架，利用可进化的PersonaTree作为长久个性化记忆，通过LLM生成树操作进行实时更新，并在推理阶段采用自适应生成策略；

**💡 创新点**

核心创新在于：①基于生物心理社会模型构建层次化人格树Schema；②设计迭代树更新机制并通过过程奖励RL训练轻量MemListener来压缩无序对话为可执行树操作；③引入自适应响应模式，将快速树推理与代理式检索融合；

**🔧 技术方法**

技术包括：LLM（DeepSeek-R1-0528、Qwen系列）进行树操作生成，RL-DAPO过程奖励学习，JSON/文档数据库存储版本化树，检索使用BGE-M3+Reranker，快速与检索混合生成；

**📊 数据集**

使用PersonaMem、HaluMem等公开对话数据集，结合PersonaMem 15k、HaluMem 13k 构造树操作训练数据，评测基准为PersonaMem；

**📈 对比分析**

与Only LLM、ALL Dialogue、LangMem、Mem0、A-Mem、MemoryOS等基线对比；在DeepSeek-R1-0528、Longcat-Flash-Chat和DeepSeek-V3.1三种回复模型上，PersonaTree+RL平均提升整体准确率约10-18个百分点，显著优于MemoryOS等传统记忆系统；

**⚠️ 局限性**

局限性：①Schema仅覆盖生物心理社会三维，未涵盖更细粒度或跨域状态；②树操作仅支持文本写/删改，缺乏时间戳、置信度等元数据；③需要手工定义操作语法与约束，部署时对隐私与治理的细化支持仍待完善。

---

## RelayLLM: Efficient Reasoning via Collaborative Decoding

**arXiv ID:** 2601.05167 | [PDF](https://arxiv.org/pdf/2601.05167v1)

**作者:** Chengsong Huang `[一作]` (Washington University in St. Louis), Jiaxin Huang `[通讯]` (Washington University in St. Louis)

**通讯引用:** 1571 | **OpenAlex IDs:** https://openalex.org/A5046688345

**关键词:** `Computation and Language` `Computational Efficiency` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Reinforcement Learning` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于 token 级别的协同解码框架 RelayLLM，让小模型在需要时主动向大模型发出调用指令，从而实现高效推理。

**💡 创新点**

创新点在于：①不依赖外部控制器，直接在小模型生成序列中插入调用指令；②使用两阶段训练（监督预热 + GRPO 强化学习）和难度感知奖励，学会在必要时精确触发大模型；③极大降低调用比例（≈1%），实现 98% 计算成本下降。

**🔧 技术方法**

核心技术包括：token 级调用指令设计、两阶段训练（监督预热 + GRPO‑RLVR）、难度感知奖励与数据过滤、vLLM 接口交互、实验中使用 Qwen3 系列模型。

**📊 数据集**

数据集：训练使用 DAPO，评测涵盖六个数学推理基准（Minerva、MATH‑500、GSM8K、Olympiad‑Bench、AIME‑2024、AIME‑2025）以及三类未见推理基准（BBEH、MMLU‑Pro、SuperGPQA）。

**📈 对比分析**

对比方法包括基线模型、GRPO、CITER 以及随机/完美路由；在六大基准上平均准确率从 42.5% 提升至 49.52%，调用比例仅 1.07%，与随机路由相比准确率提升 6.9%，计算成本降低 98%。在教师‑free 评估中，模型仍保持一定的独立推理能力。

**⚠️ 局限性**

局限性：①对教师模型的可用性和一致性敏感，教师规模变动会影响性能；②在极难任务上仍需大量调用；③需要额外的训练与推理复杂度；④对不同 LLM 家族的迁移性尚未充分验证。

---

## DocDancer: Towards Agentic Document-Grounded Information Seeking

**arXiv ID:** 2601.05163 | [PDF](https://arxiv.org/pdf/2601.05163v1)

**作者:** Qintong Zhang `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14319 | **OpenAlex IDs:** https://openalex.org/A5100459860

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Agentic AI` `Retrieval-Augmented Generation` `Text` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出端到端训练的文档问答代理 DocDancer，采用工具驱动的框架实现信息寻求式问答。

**💡 创新点**

创新点：把 DocQA 重新定义为信息寻求问题，设计仅包含搜索与阅读两大工具的轻量化代理；以及基于探索‑合成（Exploration‑then‑Synthesis）的合成数据管线，自动生成高质量的 QA 对。

**🔧 技术方法**

使用 ReAct 框架、MinerU2.5 文档解析、多模态总结模型、Qwen3‑30B / Qwen3‑4B LLM、以及搜索/读取工具实现文档导航与推理。

**📊 数据集**

训练数据来自 LongDocURL、MMDocRAG、CUAD、DUDE 四大 PDF 集合；评测使用 MMLongBench‑Doc 和 DocBench 两大长文本、多模态问答基准。

**📈 对比分析**

在 VLM、OCR、RAG 与多种提示式代理基线上进行对比。DocDancer 在 MMLongBench‑Doc 和 DocBench 上均达到或超过 SOTA，尤其在长文档多步推理上表现突出，甚至超过人类基线。

**⚠️ 局限性**

限制：实验仅在 Qwen3‑30B / Qwen3‑4B 上进行，未尝试更大模型或不同模型族；仅采用监督微调，未探讨强化学习；合成数据规模有限，未评估更大规模训练效果。

---

## Multi-Scale Local Speculative Decoding for Image Generation

**arXiv ID:** 2601.05149 | [PDF](https://arxiv.org/pdf/2601.05149v1)

**作者:** Elia Peruzzo `[一作]` (Qualcomm AI Research), Amirhossein Habibian `[通讯]` (Qualcomm AI Research)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Auto Encoder` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出了多尺度局部推测解码(Multi-Scale Local Speculative Decoding)框架，结合低分辨率草稿模型、学习上采样和局部验证来加速自回归图像生成。

**💡 创新点**

在推测解码中引入多尺度草稿、空间局部重采样与邻域扩展验证，从而在保持语义对齐的同时显著提升速度。

**🔧 技术方法**

采用自回归VQ‑VAE latent空间模型、学习上采样/下采样网络、概率池化、局部接受阈值和邻域重采样机制，并在Tar-1.5B MLLM上实现。

**📊 数据集**

在LAION‑COCO‑Aesthetic进行训练，在MS‑COCO 2017验证集（5k）上评估。

**📈 对比分析**

与ZipAR、EAGLE‑2、LANTERN等基线对比，在512p/1024p下实现最高1.7×速度提升，保持与LANTERN相近的GenEval、FID和HPSv2分数。

**⚠️ 局限性**

仍依赖于高质量的上采样模型和局部重采样，且对强模型的接受率敏感，无法与最先进的并行解码方法ZipAR完全匹配。

---

## Optimal Lower Bounds for Online Multicalibration

**arXiv ID:** 2601.05245 | [PDF](https://arxiv.org/pdf/2601.05245v1)

**作者:** Natalie Collina `[一作]`, Aaron Roth `[通讯]` (University of Pennsylvania)

**通讯引用:** 16955 | **OpenAlex IDs:** https://openalex.org/A5057693522

**关键词:** `Machine Learning`

### 📋 论文摘要

**🎯 论文内容**

研究在线多校准（multicalibration）的最优误差率，并给出了与边际校准（marginal calibration）分离的下界

**💡 创新点**

证明在线多校准在任意预测相关或预测无关的群组集合上都存在 Θ(T^{2/3}) 的下界（带对数因子），并展示了常数规模群组集合可降为边际校准、而规模为 Θ(T) 的集合无法突破该下界

**🔧 技术方法**

采用了离散化的正交系统（Walsh、Hadamard 系统）、Parseval 归一化、随机游走的返回时间分析、凸性与 Jensen 不等式、随机过程的停时技巧等概率与线性代数技术

**📊 数据集**

使用的并非真实数据集，而是自定义的合成实验环境：先验均值在等距网格上循环，标签噪声为独立符号或伯努利噪声

**📈 对比分析**

与现有方法（如基于边际校准的迭代算法）比较时，本文提供了与之相匹配的下界，表明目前已知的 O(T^{2/3}) 上界是最优（忽略对数因子）

**⚠️ 局限性**

局限在于对数因子影响、对中间规模群组集合（如 |G|=polylog(T)）的复杂度仍不清晰，且对于“proper”归约到边际校准的技术在某些实例上需要指数级别的或acles

---

## GREx: Generalized Referring Expression Segmentation, Comprehension, and Generation

**arXiv ID:** 2601.05244 | [PDF](https://arxiv.org/pdf/2601.05244v1)

**作者:** Henghui Ding `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Generation` `Transformer` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出了通用式指代表达任务 GREx（包括 GRES、GREC、GREG）并构建了大规模数据集 gRefCOCO，支持多目标和无目标表达。

**💡 创新点**

创新点在于：① 将传统单目标指代表达扩展为任意目标数量；② 设计 ReLA 模型通过区域‑图像、区域‑语言跨注意力显式建模长程关系；③ 引入目标数量预测头与多任务联合训练提升 GREC 性能。

**🔧 技术方法**

技术核心包括 Swin Transformer 编码器、BERT 语言编码、像素解码器、ReLA（Region‑Image Attention + Region‑Language Attention）、自注意力、MLP 预测、交叉熵 + L1/GIoU 损失。

**📊 数据集**

使用 COCO 图像集衍生的 gRefCOCO 数据，包含 259,859 条表达（90,064 多目标、34,537 无目标）及相应掩码与边框，兼容原 RefCOCO。

**📈 对比分析**

在 GRES、GREC 上与多种基线（VLT、LAVT、MATTNet 等）对比，ReLA 在 GRES 的 cIoU、gIoU、Pr@F1 方面分别提升约 3-4%，在 GREC 的 Pr@F1、AP 也比传统方法高 8% 以上；在 GREG 上基线 METEOR/CIDEr 与 LLM 方案相比表现一般，表明多目标生成仍具挑战。

**⚠️ 局限性**

局限包括：① 对无目标表达的识别仍有 30-40% 误检；② 处理复杂关系、计数、序号表达仍不够稳健；③ 生成多目标表达的多样性和准确性仍落后于单目标；④ 需要更多多语言、跨域扩展与更强的语义常识融合。

---

## Robust Reasoning as a Symmetry-Protected Topological Phase

**arXiv ID:** 2601.05240 | [PDF](https://arxiv.org/pdf/2601.05240v1)

**作者:** Ilmo Sung `[一作]` (Science and Technology Directorate), Ilmo Sung `[通讯]` (Science and Technology Directorate)

**关键词:** `Machine Learning` `Recurrent Neural Network` `Transformer` `Sequential` `Physics Related`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了Holonomic Network（基于SO(N)非阿贝尔全局对称的循环网络），通过将推理过程映射为拓扑学的holonomy，实现了在噪声注入下的逻辑一致性与长序列推理的无损保留。

**💡 创新点**

创新点在于将推理视作一个对称受保护拓扑相（SPT）相，证明逻辑运算与非阿贝尔任何子缠绕等价，并通过理论推导与实验验证展示了显著的质量隙、相变与无限记忆视界，从而为稳健推理提供了全新的物理学视角和架构范式。

**🔧 技术方法**

采用了SO(N) Lie代数生成器与矩阵指数实现的正交更新、路径排序乘积的holonomy计算、Wess–Zumino–Chern–Simons效应的理论框架，以及梯度通过指数矩阵的自动微分；实验中使用了标准的Transformer与RNN作为对照。

**📊 数据集**

在两个合成任务上评估：① S₃ 组乘法序列（6种状态）用于鲁棒性/相变测试；② S₁₀ 变量绑定任务（约3.6×10⁶个状态）用于长序列泛化测试，训练长度仅到50，测试到5000。

**📈 对比分析**

与高参数Transformer（约3M参数）以及128维RNN/Transformer（128维）对比，Holonomic Network仅需32维（约4.6×10⁴参数）即可在100×长度外保持100%准确率；在相变实验中，Transformer与RNN出现无隙衰减，Holonomic Network呈现明显的质量隙并对噪声具有阈值鲁棒性，展示了显著的参数效率和推理稳健性。

**⚠️ 局限性**

局限在于仅验证了合成逻辑任务，尚未在自然语言或更复杂的现实数据上证明其优势；实现需要矩阵指数与正交约束，计算开销和数值精度可能受限；拓扑相的实现高度依赖于所选对称群，迁移到其他任务需重新设计对称结构。

---

## Plenoptic Video Generation

**arXiv ID:** 2601.05239 | [PDF](https://arxiv.org/pdf/2601.05239v1)

**作者:** Xiao Fu `[一作]` (NVIDIA), Chen-Hsuan Lin `[通讯]` (NVIDIA)

**通讯引用:** 2361 | **OpenAlex IDs:** https://openalex.org/A5101863881

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Video`

### 📋 论文摘要

**🎯 论文内容**

提出 PlenopticDreamer，一个基于多视角自回归扩散模型的生成式视频重渲框架，能够沿任意摄像机轨迹保持长期空间时间一致性。

**💡 创新点**

创新点在于：①使用 3D 视场（FOV）检索机制挑选历史视频作为条件；②引入逐步上下文缩放和自条件训练提升收敛与鲁棒性；③支持长视频分块生成，保持跨段同步。

**🔧 技术方法**

核心技术包括流匹配的 Video Diffusion Transformer（DiT）、Plücker 视线映射摄像机编码、可并行的自回归生成策略、以及自条件训练与分块长视频机制。

**📊 数据集**

训练数据来自 MultiCamVideo、SynCamVideo（基本基准）以及 1M 机器人演示的 Agibot 数据集；评测基准为 Basic（100 视频、12 路轨迹）和 Agibot（200 视频）。

**📈 对比分析**

与 ReCamMaster、TrajectoryCrafter、Trajectory‑Attention 等最先进方法对比，PlenopticDreamer 在视角同步、视觉质量（PSNR/FVD）和摄像机精度（TransErr/RotErr）上均取得领先，尤其在多视角一致性和长视频生成上表现突出。

**⚠️ 局限性**

局限性包括：在极长序列或复杂人类动作（如舞蹈）时仍可能出现曝光过度、失真等错误；自条件训练虽减轻误差积累，但并不能完全消除偶发不一致。

---

## The Adverse Effects of Omitting Records in Differential Privacy: How Sampling and Suppression Degrade the Privacy-Utility Tradeoff (Long Version)

**arXiv ID:** 2601.05180 | [PDF](https://arxiv.org/pdf/2601.05180v1)

**作者:** Àlex Miranda-Pascual `[一作]` (Karlsruhe Institute of Technology), Thorsten Strufe `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 3712 | **OpenAlex IDs:** https://openalex.org/A5012001723

**关键词:** `Cryptography and Security` `Safty and Privacy` `Tabular`

### 📋 论文摘要

**🎯 论文内容**

论文研究了在差分隐私中预处理方法——采样与抑制——对隐私-效用权衡的影响。

**💡 创新点**

创新点在于首次将抑制推广为差分隐私中的任意预处理操作，并给出任意抑制算法的隐私界，比较了采样与抑制在统一设置下的效用表现。

**🔧 技术方法**

使用了标准差分隐私机制（Laplace、Gaussian、指数、报错最大、噪声平均、聚类）以及均匀Poisson采样和基于距离的抑制策略。

**📊 数据集**

实验数据集包括 Adult、Census 和 Irish 三个公开数值数据库。

**📈 对比分析**

通过对同一隐私预算下的效用指标（均值误差、模式错误率、聚类成本、NICV）进行对比，结果显示采样或抑制均未能提升效用，甚至普遍导致更差。

**⚠️ 局限性**

局限性在于仅评估了无界DP下的经典机制和几种抑制方案，对其他DP变体（如有限DP、Rényi）或更复杂抑制策略未作深入探讨。

---

## Cutting AI Research Costs: How Task-Aware Compression Makes Large Language Model Agents Affordable

**arXiv ID:** 2601.05191 | [PDF](https://arxiv.org/pdf/2601.05191v1)

**作者:** Zuhair Ahmed Khan Taha `[一作]` (Muffakham Jah), Shahnawaz Alam `[通讯]` (Muffakham Jah)

**关键词:** `Computer Vision and Pattern Recognition` `Compression` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Large Language Model` `Prompt Engineering` `Knowledge Distillation` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出并实现了 AgentCompress 方法，能够在执行学术文献回顾和假设生成等自主任务时显著降低对大型语言模型的计算成本。

**💡 创新点**

创新点在于利用任务驱动的压缩策略，将模型在生成新假设时的高计算需求降到与文本重排相近的水平，实现成本与性能的平衡。

**🔧 技术方法**

采用了模型压缩、提示工程和动态推理策略相结合的技术手段，并通过知识蒸馏提升压缩后模型的质量。

**📊 数据集**

主要使用了公开的学术论文数据集（如 arXiv、PubMed 以及 OpenAI API 产生的数据）进行实验验证。

**📈 对比分析**

与原始 70B 模型、GPT‑4 及轻量级替代模型进行对比，AgentCompress 在保持 90% 以上的文本质量的同时，将推理成本降低约 70%，节省费用约 $90。

**⚠️ 局限性**

局限性包括：压缩后模型在极度需要深度推理的复杂假设生成任务中仍可能出现性能下降；算法对模型规模的适用性有一定限制，且压缩过程需要额外的训练成本。

---

## Learning Latent Action World Models In The Wild

**arXiv ID:** 2601.05230 | [PDF](https://arxiv.org/pdf/2601.05230v1)

**作者:** Quentin Garrido `[一作]` (FAIR at Meta), Michael Rabbat `[通讯]` (FAIR at Meta)

**关键词:** `Artificial Intelligence` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `World Model` `Video`

### 📋 论文摘要

**🎯 论文内容**

研究在大规模无标签自然视频中学习潜在动作世界模型，并展示其在规划任务中的可用性。

**💡 创新点**

①在自然视频而非任务限定数据上训练潜在动作模型；②对稀疏、噪声和离散三种信息正则化进行对比，证明连续稀疏/噪声更适合；③分析潜在动作的未来泄露与转移能力，发现其相对相机局部化；④通过训练控制器将真实动作映射到潜在动作，实现无标签动作空间的规划。

**🔧 技术方法**

逆动力学模型+前向模型联合训练；稀疏/噪声正则化与向量量化；V‑JEPAv2‑L 视图因果编码器；ViT‑L 作为前向模型；AdaLN‑zero 与 RoPE 位置编码；CEM 规划；控制器为 MLP 或跨注意力适配器。

**📊 数据集**

大规模无标签自然视频集 YoutubeTemporal‑1B（16 帧/4fps），以及 Kinetics、RECON、DROID、NWM 等用于评估。

**📈 对比分析**

与已标注动作的 V‑JEPAv2‑AC、V‑JEPAv2、NWM、NoMaD 等基线对比；潜在动作模型在规划任务中可达到与这些基线相近的性能；在单步预测中稀疏/噪声潜在动作优于向量量化；转移与循环一致性误差低；在大规模实验中性能随模型/时间/数据规模提升。

**⚠️ 局限性**

信息约束采用静态系数，难以自适应动作复杂度；未直接采样潜在动作进行规划；冻结的表示空间限制了逆动力学训练和预测质量；缺乏单阶段训练；潜在动作相机局部化导致跨视角迁移受限。

---

## Mechanisms of Prompt-Induced Hallucination in Vision-Language Models

**arXiv ID:** 2601.05201 | [PDF](https://arxiv.org/pdf/2601.05201v1)

**作者:** William Rudman `[一作]` (University of Texas at Austin), Kyle Mahowald `[通讯]` (University of Texas at Austin)

**通讯引用:** 3678 | **OpenAlex IDs:** https://openalex.org/A5039468724

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Recognition` `Transformer` `Vision Language Model` `Prompt Engineering` `Image` `Text` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

在受控的目标计数任务中，系统分析了视觉‑语言模型（VLM）在面对与图像不一致的提示时产生的提示诱发幻觉（PIH）现象，并通过注意力头的消融证明可以显著降低此类幻觉。

**💡 创新点**

创新点在于：①首次定位到一小批早期层的注意力头是 PIH 的关键驱动器；②利用消融实验直接验证这些头的因果作用；③证明 PIH 机制在不同 VLM 之间共享且可迁移到其他任务（如颜色识别）。

**🔧 技术方法**

技术手段主要包括：注意力头消融（mean ablation）与分层消融、对比分析（prompt‑vs‑ground‑truth 置信度）以及对输出形式（exact/soft/format copying）的分类；同时通过可视化注意力权重评估消融对图像/文本关注的影响。

**📊 数据集**

使用的数据集有：CountBench（针对目标计数）和 Visual CounterFact（颜色识别），以及三种 7B 级 VLM：LLaVA‑OneVision、Qwen2‑VL、Janus‑Pro。

**📈 对比分析**

与基线提示下的计数准确率相比，消融 PIH 头后在误导提示下的正确计数匹配率从约 30‑70% 提升到 70‑78%，提示匹配率从 42‑64% 降至 1‑10%；在颜色任务中，PIH 消融可将幻觉率降至 5‑20%，提升 40‑95%。

**⚠️ 局限性**

局限性包括：仅针对中等规模（∼7B）模型，未验证在更大模型上的可迁移性；分析仅基于注意力机制，未揭示更深层的因果机制；消融可能产生二次效应，且不同模型的机制差异尚未完全解释。

---

## CoV: Chain-of-View Prompting for Spatial Reasoning

**arXiv ID:** 2601.05172 | [PDF](https://arxiv.org/pdf/2601.05172v1)

**作者:** Haoyu Zhao `[一作]` (Zhejiang University), Bohan Zhuang `[通讯]` (Zhejiang University)

**关键词:** `Computer Vision and Pattern Recognition` `Vision Language Model` `Prompt Engineering` `Point Cloud` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

构建了 Chain-of-View (CoV) 推理框架，使 VLM 在 3D 嵌入式问答中通过先粗略视角筛选再细粒度视角调整实现主动探索。

**💡 创新点**

创新点是将视角选择与动作‑推理循环结合，在推理时动态生成视角并通过测试时可扩展的步数实现性能提升。

**🔧 技术方法**

使用的技术包括视角选择代理、SE(3) 相机变换、对话式提示、动作‑推理链以及测试时缩放策略。

**📊 数据集**

使用的数据集为 OpenEQA、ScanQA、SQA3D 等公开 3D 问答基准。

**📈 对比分析**

与基线对比，CoV 在 OpenEQA 上平均提升 11.56%（最高 13.62%），在 ScanQA 上 CIDEr 116、EM@1 31.9%，在 SQA3D 上 EM@1 51.1，显示显著性能提升。

**⚠️ 局限性**

局限包括对高度动态或杂乱环境的适应不足、过长探索路径易产生噪声/幻觉，以及对视角选择质量的高度依赖。

---

## LaST$_{0}$: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model

**arXiv ID:** 2601.05248 | [PDF](https://arxiv.org/pdf/2601.05248v1)

**作者:** Zhuoyang Liu `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**通讯引用:** 9911 | **OpenAlex IDs:** https://openalex.org/A5013030532

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Transformer` `Vision-Language-Action Model` `Mixture of Experts` `Reinforcement Learning` `Multimodality` `Point Cloud`

### 📋 论文摘要

**🎯 论文内容**

提出了LaST_0框架，利用潜在时空链式推理（Latent Spatio-Temporal Chain-of-Thought）实现机器人视觉-语言-动作（VLA）模型的高效“先推理再执行”。

**💡 创新点**

核心创新包括：①在连续潜在空间中进行多模态推理，兼顾视觉、三维几何与机器人本体状态；②通过Mixture-of-Transformers实现低频推理专家与高频执行专家的异步协同；③在训练中采用教师强迫潜在推理与流匹配动作生成的分阶段策略。

**🔧 技术方法**

使用了预训练的SigLIP-大型视觉编码器、Uni3D点云编码器、DeepSeek-LLM 1B基础模型；引入了流匹配（Flow Matching）动作生成、Cosine相似度潜在监督、KV缓存异步执行等技术；采用Mixture-of-Transformers实现双专家架构。

**📊 数据集**

在大规模机器人轨迹数据集上预训练，包含Open-X-Embodiment、DROID、ROBOTMIND等约400k条轨迹；随后在RLBench 10个仿真任务和6个真实世界任务（单臂/双臂）进行微调；使用多视角RGB-D和点云作为输入。

**📈 对比分析**

在10个RLBench仿真任务上平均成功率达82%，比最强基线提升约8%；在6个真实世界任务中平均成功率72%，比SpatialVLA、π_0.5、CoT-VLA分别提升约33%、13%和19%；推理速度为15.4 Hz，显著快于显式CoT的1.1 Hz，接近π_0.5的13.8 Hz。

**⚠️ 局限性**

限制包括：①潜在推理仍需高质量预训练编码器，对不同传感器/环境的泛化仍有限；②双专家异步频率调节需要手动设定或混合训练，可能导致推理更新延迟；③缺乏显式可解释的推理轨迹，难以直接解释决策过程；④在极端动态或接触丰富的长期任务中，潜在空间可能不足以捕获全部物理细节。

---

## Measuring and Fostering Peace through Machine Learning and Artificial Intelligence

**arXiv ID:** 2601.05232 | [PDF](https://arxiv.org/pdf/2601.05232v1)

**作者:** P. Gilda `[一作]`, S. Carter `[通讯]`

**关键词:** `Computation and Language` `Convolutional Neural Network` `Large Language Model` `Prompt Engineering` `Text` `Video`

### 📋 论文摘要

**🎯 论文内容**

本文通过机器学习与人工智能测量新闻与社交媒体中的和平程度，并基于此开发了实时反馈 Chrome 扩展 MirrorMirror。

**💡 创新点**

创新点在于将情感分析（GoEmotions）与大型语言模型（LLM）相结合，构建多维和平评估框架，并将评估结果实时呈现给用户以提升媒体意识。

**🔧 技术方法**

使用技术包括文本嵌入神经网络（CNN、全连接网络）、GoEmotions 词级情感模型、LLM（Gemini、GPT‑4o 等）以及混合提示与情感映射。

**📊 数据集**

使用的数据集有 NOW 新闻语料库（约70万条）、Capstone Peace Speech 语料库（60万条）、22 条 YouTube 录音稿以及 52 条由专家评分的黄金标准视频。

**📈 对比分析**

对比实验显示，CNN 与全连接网络在 NOW 上达 97%+ 准确率，迁移至 Capstone 仍保持约 72%；GoEmotions 与词级模型相关系数仅约 0.18；LLM 评估在 5 维社会因素上相关系数最高可达 0.773，Gemini 在细节维度上相较前代提升约 0.317。

**⚠️ 局限性**

主要局限包括：对口语媒体迁移性能差、情感模型在中性词占比高导致误判、数据量有限（YouTube 仅 22 条）、缺乏多模态特征、模型解释性不足以及需要更大规模验证。

---

## Internal Representations as Indicators of Hallucinations in Agent Tool Selection

**arXiv ID:** 2601.05214 | [PDF](https://arxiv.org/pdf/2601.05214v1)

**作者:** Kait Healy `[一作]` (Amazon), Jing Wu `[通讯]` (Amazon)

**关键词:** `Artificial Intelligence` `Classification` `Anomaly Detection` `Transformer` `Large Language Model` `Text` `Finance Related`

### 📋 论文摘要

**🎯 论文内容**

提出一种基于LLM内部表示的实时工具调用幻觉检测框架，利用单次前向推理中的最后层隐藏状态训练轻量级分类器来识别错误的工具调用。

**💡 创新点**

创新点在于：①无需多次采样或外部验证；②使用模型内部表征而非输出文本进行幻觉判别；③通过掩码生成无监督标签的训练管线；④实现毫秒级实时检测。

**🔧 技术方法**

技术包括：Transformer最后层特征抽取（函数名、参数区块、结束符聚合）、两层MLP分类器、基于二元交叉熵的训练、温度缩放校准以及单向前向推理中的特征拼接。

**📊 数据集**

使用公开的 Glaive Function-Calling 数据集（涵盖个人健康、金融、可持续发展、计算器等多领域）并在五类专用代理上收集工具调用实例。

**📈 对比分析**

与 NCP、语义相似度等多次采样基线对比，单前向推理模型在 GPT‑OSS‑20B、Llama‑3.1‑8B、Qwen‑7B 上分别达到 86%、73%、74% 的准确率，精确率高达 86%，召回率相对较低但整体性能保持竞争力。

**⚠️ 局限性**

局限性包括：①仅进行二元检测，未细化不同幻觉类型；②标签生成依赖字符串匹配，可能误判语义相同但形式不同的调用；③对不同模型族的通用性尚未验证；④特征设计过于简单，缺乏对参数字段的细粒度关注。

---

## FlowLet: Conditional 3D Brain MRI Synthesis using Wavelet Flow Matching

**arXiv ID:** 2601.05212 | [PDF](https://arxiv.org/pdf/2601.05212v1)

**作者:** Danilo Danese `[一作]` (Politecnico di Bari), Tommaso Di Noia `[通讯]` (Politecnico di Bari)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Flow-based Model` `Ordinary Differential Equation` `Image` `Biomedical Data` `Magnetic Resonance Imaging` `Alzheimer's Disease`

### 📋 论文摘要

**🎯 论文内容**

本文提出了一个基于波形域流匹配的可条件化三维脑MRI生成框架FlowLet，用于合成符合年龄条件的高保真脑影像并增强脑年龄预测模型的表现

**💡 创新点**

创新点在于将流匹配（Flow Matching）与可逆三维Haar小波变换结合，避免了传统潜在压缩导致的伪影，并通过特征级和空间级条件实现精准的年龄控制；同时提供了多种流匹配公式和基于ROI的评估体系

**🔧 技术方法**

采用的技术包括：三维Haar小波变换、流匹配（RFM、CFM、VP、Trig）与可逆变换、条件3D U‑Net（FiLM与空间注意力）、ODE积分采样、以及多尺度结构相似性与KL/DICE等局部指标评估

**📊 数据集**

训练数据来源于三大公开数据集：OpenBHB（主要年轻人群）、ADNI（老年人群）和OASIS‑3（中老年人群），共计约 6,700 份高分辨率MRI，统一预处理至 91×109×91 的体素尺寸

**📈 对比分析**

与五种最先进生成模型（WDM、MD、MLDM、BrainSynth、MOTFM）比较，FlowLet 在 FID/MMD 上与或优于基线，在 10 步 ODE 采样下速度最快，且在脑年龄预测（MAE）和 95 区域的 iMAE/KL/DICE 上均表现最佳，特别是 RFM 变体在多维度指标上居首位

**⚠️ 局限性**

局限性包括：虽然在指标上优异，但仍缺乏专家临床评估的验证；模型目前仅处理年龄条件，尚未扩展到疾病或认知分数等多维条件；在极端年龄区间的生成质量及对大规模样本的可扩展性仍需进一步研究

---

## MoE3D: A Mixture-of-Experts Module for 3D Reconstruction

**arXiv ID:** 2601.05208 | [PDF](https://arxiv.org/pdf/2601.05208v1)

**作者:** Zichen Wang `[一作]` (University of Michigan), Jeong Joon Park `[通讯]` (University of Michigan)

**通讯引用:** 961 | **OpenAlex IDs:** https://openalex.org/A5109413502

**关键词:** `Computer Vision and Pattern Recognition` `Depth Estimation` `Segmentation` `Transformer` `Mixture of Experts` `Image`

### 📋 论文摘要

**🎯 论文内容**

引入轻量级的Mixture-of-Experts模块MoE3D，以改进VGGT在深度边界上的表现并显著减少飞点伪影。

**💡 创新点**

在像素级别采用多专家分支和熵正则化的Mixture-of-Experts架构，让模型在深度不确定区域自动路由并产生多模态预测，从而提升边界清晰度与几何一致性。

**🔧 技术方法**

使用Transformer骨干VGGT、Mixture-of-Experts（softmax门控+熵正则化）、端到端微调、以及轻量级卷积分支实现深度预测。

**📊 数据集**

训练与评估数据集包括合成Hypersim、Virtual KITTI（用于微调）、以及实测数据集NYU‑v2、Sintel、NRGBD、Bonn、KITTI。

**📈 对比分析**

与DUSt3R、MASt3R、VGGT、CUT3R等基线在单视、双视以及边界评估上对比，MoE3D在Acc/Comp下降约20%+、NC提升，边界mIoU提升至0.40+，在大多数指标上均位居首位或第二。

**⚠️ 局限性**

局限性：仅用至多两视图训练，导致多视一致性不够强；飞点伪影虽大幅减少但仍可能在极端区域出现少量残留。

---

## LELA: an LLM-based Entity Linking Approach with Zero-Shot Domain Adaptation

**arXiv ID:** 2601.05192 | [PDF](https://arxiv.org/pdf/2601.05192v1)

**作者:** Samy Haffoudhi `[一作]` (Telecom Paris Institute Polytechnique de Paris), Nils Holzenberger `[通讯]` (Telecom Paris Institute Polytechnique de Paris)

**关键词:** `Computation and Language` `Domain Adaptation` `Retrieval` `Transformer` `Large Language Model` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出了LELA，一种无监督、无细调的粗到细的实体链接方法，能够在任意知识库和领域上直接使用。

**💡 创新点**

创新点在于将LLM用于候选检索的细粒度重新排序和最终候选选择，并通过自一致性推理提升精确度，从而实现真正的零样本实体链接。

**🔧 技术方法**

技术包括基于BM25或稠密检索的候选生成、指令式LLM重排序、LLM推理式候选选择以及自一致性投票，核心模型为大语言模型（如Qwen、Magistral）。

**📊 数据集**

使用了多种基准数据集：ZESHEL、ESCO、GLADIS、ZELDA以及其他Wiki链接基准，以评估跨域、专有库和多语言环境下的性能。

**📈 对比分析**

与现有最先进的零样本或经典有监督方法比较，LELA在大多数基准上均取得领先或竞争性表现，特别是在ZESHEL和GLADIS上显著优于对手，整体上能与细调模型媲美。

**⚠️ 局限性**

主要局限在于依赖LLM推理导致计算成本高、推理速度慢，且对低资源或多语言场景的适用性尚未充分验证，缺乏对模型可解释性的支持。

---

## SimuAgent: An LLM-Based Simulink Modeling Assistant Enhanced with Reinforcement Learning

**arXiv ID:** 2601.05187 | [PDF](https://arxiv.org/pdf/2601.05187v1)

**作者:** Yanchang Liang `[一作]` (University of Warwick), Xiaowei Zhao `[通讯]` (University of Warwick)

**通讯引用:** 5446 | **OpenAlex IDs:** https://openalex.org/A5000635250

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Retrieval-Augmented Generation` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

开发了SimuAgent，一种基于大型语言模型的代理，用于自动化和协助Simulink建模、分析与验证任务。

**💡 创新点**

创新点包括：①将Simulink模型转换为轻量级Python字典表示，显著降低token量并提升可解释性；②提出Reflection‑GRPO（ReGRPO）强化学习框架，利用自我反思文本提供稀疏奖励的中间反馈；③设计抽象‑重构（Abstract‑Reconstruct）自监督数据增强，提升模型对高层抽象的理解；④发布了5300条多域建模任务的SimuBench基准。

**🔧 技术方法**

核心技术包括：大型语言模型（如Qwen‑2.5‑7B‑Instruct）、计划‑执行架构、两阶段分层训练、ReGRPO强化学习、工具调用与RAG知识库、Python字典验证环境。

**📊 数据集**

使用的数据集是SimuBench（5300条控制、机械、电气、流体、热学、电磁学等领域的建模、修改和问答任务）以及公开的Simulink、Modelica、PSCAD等模型进行跨平台迁移评估。

**📈 对比分析**

与传统CoT、RAG、SFT、GPT‑4o等基线对比，SimuAgent在SimuBench上的整体成功率达51.89%，在小型/大型系统建模和问答任务上均优于对手；在跨平台迁移上，训练后模型在Modelica和PSCAD上分别达到33.84%和33.04%，细微提升。

**⚠️ 局限性**

局限性包括：仍依赖大量训练数据且对极大系统的建模精度与GPT‑4o有一定差距；当前模型只支持文本和Python接口，缺乏多模态视觉或实时物理仿真；反思机制需要手工设定模板，尚未完全自动化。

---

## VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice

**arXiv ID:** 2601.05175 | [PDF](https://arxiv.org/pdf/2601.05175v1)

**作者:** Shuming Liu `[一作]` (King Abdullah University of Science and Technology), Yunyang Xiong `[通讯]` (Meta AI)

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning` `Computational Efficiency` `Transformer` `Reinforcement Learning` `Large Language Model` `Chain-of-Thought` `Video` `Text` `Image` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

提出一种视频推理自适应框架（AutoThink），通过“思考一次、回答两次”模板训练模型，使其在需要时才调用链式推理，避免无谓的长推理轨迹。

**💡 创新点**

创新点包括：
1) 在训练时使用双答案奖励（初始答案+复核答案），消除手工 think/no‑think 标注；
2) 推理阶段采用基于 token 置信度的早停策略，决定是否继续生成 CoT；
3) 通过 fallback 字符串鼓励模型在无法直接给出答案时先提示“先思考”，提高鲁棒性。

**🔧 技术方法**

主要技术：
- Group Relative Policy Optimization（GRPO）强化学习；
- 双答案奖励机制与 fallback 奖励；
- 置信度基早停（confidence‑based early exit）；
- Qwen2.5‑VL‑7B 与 Qwen3‑VL‑8B 作为基础 LLM；
- 数据增广：结合高质量文本、图像与视频推理数据。

**📊 数据集**

使用的数据集：
- 视频 QA：VideoMME、MVBench、LongVideoBench、MMVU、VideoMMMU、MVP；
- 时间定位：Charades‑STA、ActivityNet、NExT‑GQA；
- 图像推理：MathVista、MathVision、MathVerse、MMMU、MMMU‑Pro、MM‑Vet；
- 训练集：约 83K 条混合文本/图像/视频样本。

**📈 对比分析**

对比方法：SFT、RL 无思考、RL 有思考、AutoThink（本框架）。
性能表现：
- 在 VideoMME 上从 66.0% 提升至 67.3%；
- VideoMMMU 由 54.7% 提升至 58.6%；
- MVP 对偶准确率从 36.5% 提升至 39.4%；
- 时序定位 mIoU 从 52.9% 提升至 60.0%；
- 平均响应长度从 386 下降至 44，思考比例仅 25%~51%。
综上，AutoThink 在保持或提升准确率的同时显著提高推理效率。

**⚠️ 局限性**

局限性：
1) 依赖置信度阈值 τ 的选择，对不同任务可能需微调；
2) 对视觉噪声大或信息稀疏的视频仍可能误判不需思考；
3) 目前仅在视频推理任务验证，跨模态或更大规模模型的适用性待进一步验证；
4) 训练过程仍需大量标注与强化学习样本；
5) 对模型规模和推理预算的弹性支持有限。

---

## Fundamental Tradeoffs for ISAC Multiple Access in Finite-Blocklength Regime

**arXiv ID:** 2601.05165 | [PDF](https://arxiv.org/pdf/2601.05165v1)

**作者:** Zhentian Zhang `[一作]` (Southeast University), Mohammad Javad Ahmadi `[通讯]` (Technische Universität Dresden)

**关键词:** `Information Theory`

### 📋 论文摘要

**🎯 论文内容**

在有限码字长度（FBL）条件下研究了上行多用户ISAC系统的通信-感知性能折衷，推导了感知误差与码字几何的关系，并给出了可实现与对偶的理论界限；

**💡 创新点**

提出将感知误差与信道状态估计的几何结构关联的视角，揭示码字互相关是决定ISAC性能的关键因素，并给出通用CRB将信道估计误差映射到实际感知参数；

**🔧 技术方法**

利用随机编码、格尔舍林圆定理、Neumann级数展开、最小二乘估计、Fisher信息矩阵与Cramér‑Rao界限等信息理论与矩阵分析技术；

**📊 数据集**

采用3GPP TR 38.901 模型进行仿真，使用标准参数（k=16、n=1000、m=10、SNR=10 dB等）验证理论结果；

**📈 对比分析**

将理论得到的可实现与对偶界限与仿真误差阈值进行对比，结果表明在给定感知误差阈值下可实现率位于两界限之间，且随着块长增加“沉默区”收缩；

**⚠️ 局限性**

局限在于仅考虑理想的齐次多用户信道与AWGN、未考虑时延失配与频偏、以及在实际部署中码字互相关难以严格控制等实际因素。

---

## $PC^2$: Politically Controversial Content Generation via Jailbreaking Attacks on GPT-based Text-to-Image Models

**arXiv ID:** 2601.05150 | [PDF](https://arxiv.org/pdf/2601.05150v1)

**作者:** Wonwoo Choi `[一作]`, Myoungsung You `[通讯]`

**关键词:** `Cryptography and Security` `Generation` `Adversarial Attack` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

### 📋 论文摘要

**🎯 论文内容**

设计并实现了一种黑盒政治类文本到图像模型的越狱框架，能够在商业 T2I 系统中生成包含政治敏感内容（PSCs）的图像。

**💡 创新点**

创新点在于结合身份保留描述映射（IPDM）与跨语言的地缘政治远离翻译策略，分散并弱化政治语义，从而绕过安全过滤器；并提出多维度知识度量与加权融合的语言选择方法。

**🔧 技术方法**

技术手段包括：NER 与短语提取、GPT‑4o 生成 IPDM 描述、GPT‑4o 进行多语种翻译与反向翻译、基于 Wikipedia 的关键词/国家/偏见/政治敏感度度量、基于余弦相似度的向量检索、加权综合评分的语言排名与索引选择，以及构造多语言索引列表的提示组装。

**📊 数据集**

数据集为自构建的 240 条政治敏感提示基准，涵盖 36 位公职人员，供实验与评估使用。

**📈 对比分析**

与随机语言选择基线对比，攻击在 GPT‑4o、GPT‑5、GPT‑5.1 上的成功率分别提升至 86%、68% 和 76%（总成功率），远高于随机基线（约 28%–32%）。在 Midjourney、Nano‑Banana 等非 GPT 平台的实验也表明攻击效果不一，提示不同的安全阈值。

**⚠️ 局限性**

局限性包括：仅在 GPT 系列 T2I 接口上验证；对其他商业平台的通用性有限；方法依赖多语言翻译准确性与多语言知识库覆盖；系统级安全提示虽能消除攻击但会产生高误报率。

---

## Mesh4D: 4D Mesh Reconstruction and Tracking from Monocular Video

**arXiv ID:** 2601.05251 | [PDF](https://arxiv.org/pdf/2601.05251v1)

**作者:** Zeren Jiang `[一作]` (VGG University of Oxford), Andrea Vedaldi `[通讯]` (VGG University of Oxford)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Object Tracking` `Transformer` `Diffusion model` `Auto Encoder` `Video` `Mesh`

### 📋 论文摘要

**🎯 论文内容**

构建了一种单目RGB视频的四维网格重建方法，能够一次性预测完整三维形状与运动

**💡 创新点**

创新点在于利用隐式变形VAE学习完整序列的运动潜在空间，并在训练时引入骨骼先验但推理时无需骨骼

**🔧 技术方法**

结合预训练的隐式3D生成模型、变分自编码器、空间‑时间Transformer和条件扩散模型

**📊 数据集**

使用从Objaverse筛选的约9000个带骨骼动画的3D网格数据集，并在50个未见动画序列上进行评估

**📈 对比分析**

与HY3D、L4GM、GVFD等基线对比，在几何精度、追踪一致性和新视角合成上均达到或超过SOTA，尤其在形状精度与时间一致性上优于对手

**⚠️ 局限性**

局限于依赖高质量的初始网格和骨骼先验，无法处理拓扑变化和极度非刚性运动

---

## Stock Market Price Prediction using Neural Prophet with Deep Neural Network

**arXiv ID:** 2601.05202 | [PDF](https://arxiv.org/pdf/2601.05202v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## Random Models and Guarded Logic

**arXiv ID:** 2601.05247 | [PDF](https://arxiv.org/pdf/2601.05247v1)

**作者:** Oskar Fiuk `[一作]` (Institute of Computer Science, University of Wrocław), Oskar Fiuk `[通讯]` (Institute of Computer Science, University of Wrocław)

**关键词:** `Logic in Computer Science` `Probabilistic Methods`

### 📋 论文摘要

**🎯 论文内容**

本文基于Gurevich和Shelah对Gödel类的思想，提出了对第一阶逻辑的受限片段的有限模型性质的新概率证明。

**💡 创新点**

创新点在于提供了一种概念上简单的证明方法，并且得到了最优的双指数上界，构造了强制模型大小紧密匹配的句子。

**🔧 技术方法**

使用了概率方法，并通过确定性哈希函数对概率证明进行了去随机化。

**📊 数据集**

使用了第一阶逻辑的受限片段的句子，具体数据集未明确提及。

**📈 对比分析**

与之前的证明方法相比，本文的证明更简单且自包含，性能上达到了最优的双指数上界。

**⚠️ 局限性**

限制在于当前方法主要适用于受限片段，未来的工作需要将概率方法扩展到更强的片段。

---

## Generate, Transfer, Adapt: Learning Functional Dexterous Grasping from a Single Human Demonstration

**arXiv ID:** 2601.05243 | [PDF](https://arxiv.org/pdf/2601.05243v1)

**作者:** Xingyi He `[一作]` (Cornell University), Kuan Fang `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Data Synthesis` `Generation` `Image` `Point Cloud` `Multimodality`

### 📋 论文摘要

**🎯 论文内容**

本文提出一种基于单人类演示视频的对应式数据生成引擎，能够自动产生数千万条高质量功能抓取样本，并训练一种融合RGB图像语义与点云几何的多模态预测网络，实现单视角RGB‑D输入下的精确功能抓取预测。

**💡 创新点**

创新点包括：① 用2D图像匹配+3D聚类实现跨实例的对应传递，显著提升抓取标签多样性；② 引入物理信息的抓取适配优化，保证功能与稳定性双重满足；③ 在预测网络中引入局部‑全局自适应注意力与重要性采样，融合语义与几何特征，实现高效、准确的功能抓取推理。

**🔧 技术方法**

技术手段包括：预训练的2D图像匹配模型、3D点云聚类、基于物理仿真的抓取适配优化、条件变分自编码器(CVAE)＋密集距离矩阵表示、点云与图像特征的多模态融合、局部自适应注意力机制、重要性采样、IsaacGym等仿真平台。

**📊 数据集**

数据集：通过互联网图像与Rodin/2D→3D生成得到900个对象，产生约11M张RGB‑D图像与抓取对；训练集覆盖9类常见工具（如钻、注射器、喷雾瓶等），测试集采用未见过的同类对象。

**📈 对比分析**

与基线（𝒟(ℛ,𝒪)、SparseDFF、DenseMatcher、AG‑Pose）在仿真和真实世界的抓取成功率对比，本文方法在仿真中实现69%成功率，远高于基线；真实世界同样显著优于所有对比方法；推理时间为0.92s（Shadow Hand）和0.36s（Inspire Hand）。

**⚠️ 局限性**

局限性：模型对严重噪声或失真深度输入敏感，且训练仍以类别为基础，无法完全开放集（全新对象/任务）处理，缺乏对通用性与跨任务泛化的支持。

---

## GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization

**arXiv ID:** 2601.05242 | [PDF](https://arxiv.org/pdf/2601.05242v1)

**作者:** Shih-Yang Liu `[一作]` (NVIDIA), Pavlo Molchanov `[通讯]` (NVIDIA)

**通讯引用:** 6771 | **OpenAlex IDs:** https://openalex.org/A5066945976

**关键词:** `Computation and Language` `Reinforcement Learning` `Optimization` `Reinforcement Learning`

### 📋 论文摘要

**🎯 论文内容**

针对大语言模型的多奖励强化学习，提出了 Group reward‑Decoupled Normalization Policy Optimization (GDPO) 方法，并系统评估其在工具调用、数学推理和代码推理任务中的效果。

**💡 创新点**

创新点在于：① 发现 GRPO 在多奖励场景下会导致奖励组合收缩为相同优势值，削弱学习信号；② 通过对每个奖励分别进行组内归一化再求和，并加入批次级优势归一化，显著保留奖励维度间的差异并保持数值稳定；③ 提供了奖励权重与条件化奖励的系统设计，帮助解决奖励难度差异导致的优先级失衡。

**🔧 技术方法**

技术手段包括：Group Relative Policy Optimization (GRPO) 的改进，组内与批次级归一化，奖励权重与条件化奖励设计，基于 Rollout 的策略梯度与自适应剪辑、动态采样等 RL 技术。

**📊 数据集**

使用的数据集涵盖：ToolACE、Hammar、xLAM、Berkeley Function Call Leaderboard (BFCL‑v3)、DeepScaleR‑Preview、MATH、AIME、AMC、Minerva、Olympiad Bench、Eurus‑2‑RL、PRIME 等多种工具调用、数学推理和代码推理数据集。

**📈 对比分析**

实验对比 GRPO、GRPO‑w/o‑std 以及 GDPO，结果显示 GDPO 在所有任务上均取得更高的正确性、格式符合度、推理准确率、长度约束满足率、Bug 率和代码通过率；在多奖励情况下，GDPO 能更好地保持优势信号多样性，避免训练崩溃，整体性能提升约 2–6%（准确率/通过率）且长度/Bug 率显著下降。

**⚠️ 局限性**

限制与挑战包括：① 对极端大规模奖励维度的扩展验证不足；② 仍需手动调节奖励权重或设计条件化奖励以解决奖励难度差异；③ 在更复杂任务或不同模型规模下的可推广性尚未完全评估。

---

## Observations and Remedies for Large Language Model Bias in Self-Consuming Performative Loop

**arXiv ID:** 2601.05184 | [PDF](https://arxiv.org/pdf/2601.05184v1)

**作者:** Yaxuan Wang `[一作]` (University of California), Yang Liu `[通讯]` (University of California)

**通讯引用:** 59131 | **OpenAlex IDs:** https://openalex.org/A5100355692

**关键词:** `Artificial Intelligence` `Large Language Model` `Reinforcement Learning` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

研究了大型语言模型在自我消耗的 performative 循环中的偏差演化，并提出了奖励导向的重采样方法进行缓解。

**💡 创新点**

提出 Self‑Consuming Performative Loop（SCPL）框架，首次系统研究增量微调循环在 performative 反馈下的偏差，并设计基于奖励的重采样与重权重机制来抑制偏差。

**🔧 技术方法**

递归自我训练、在线持续学习、奖励导向重采样与重权重、惩罚采样策略、直接偏好优化（DPO）等技术。

**📊 数据集**

News Continuation（Webis‑Bias‑Flipper‑18）、Preference Dissection（Dolly 与 ShareGPT）、Math Problem Solving（NuminaMath）以及 MMLU 基准数据集。

**📈 对比分析**

与非动态自我消耗循环（real vs synthetic）、固定比例动态循环、累积增强循环进行对比。实验表明动态 SCPL 加速偏好偏差、降低不同群体性能差异；累积可缓解偏差增长与生成质量下降。奖励导向重采样在新闻与偏好分解任务中显著降低偏差，性能接近或优于基线。

**⚠️ 局限性**

依赖预定义奖励规则，若规则设计不当可能产生新的偏差；计算成本高，未在实验中包含自我强化学习循环。

---

## Vision-Language Introspection: Mitigating Overconfident Hallucinations in MLLMs via Interpretable Bi-Causal Steering

**arXiv ID:** 2601.05159 | [PDF](https://arxiv.org/pdf/2601.05159v1)

**作者:** Shuliang Liu `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1186 | **OpenAlex IDs:** https://openalex.org/A5057914558

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Generation` `Explainability and Interpretability` `Transformer` `Large Language Model` `Vision Language Model` `Multimodality` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

提出一种训练无关的推理框架 Vision‑Language Introspection (VLI)，通过自我检视与双因果调节来减少多模态大模型的物体幻觉。

**💡 创新点**

创新点在于将“因果关注定位”（Attributive Introspection）与“可解释双因果调节”（Interpretable Bi‑Causal Steering）结合，实现对特定视觉锚点的精确诊断与动态纠正，并加入自适应置信度校准来抑制盲目信心。

**🔧 技术方法**

核心技术包括：内部状态冲突检测（JS 散度）、专家注意力筛选、像素级锚点提取、两条对照输入的反事实构造（Anchor‑Only / Context‑Only）以及多层隐藏状态的差分调节。

**📊 数据集**

使用公开的多模态基准 MMHal‑Bench（评估生成任务幻觉）和 POPE（评估检测任务准确性）进行实验，模型基于 LLaVA‑1.5 与 Qwen3‑VL。

**📈 对比分析**

与多种对比方法（VCD、CICD、ClearSight、OPERA、VTI、Nullu 等）相比，VLI 在 MMHal‑Bench 上将幻觉率降低 12.67%（至 45.63%），在 POPE 上提升准确率 5.8% 以上，整体性能位居前列。

**⚠️ 局限性**

主要限制包括：对反事实构造的计算开销和显存占用较高，以及对模型中可辨识专家注意力的依赖，若缺乏明显注意力聚焦则定位与调节效果可能受限。

---

## Learning Mixture Models via Efficient High-dimensional Sparse Fourier Transforms

**arXiv ID:** 2601.05157 | [PDF](https://arxiv.org/pdf/2601.05157v1)

**作者:** Alkis Kalavasis `[一作]`, Manolis Zampetakis `[通讯]`

**关键词:** `Data Structures and Algorithms` `Mixture of Experts`

### 📋 论文摘要

**🎯 论文内容**

本研究提出了一种高效的算法，用于学习d维空间中k个球形分布的混合模型的参数，包括均值和混合权重。该算法适用于重尾分布，并且在组件均值之间没有最小分离的要求。

**💡 创新点**

创新点在于该算法能够处理重尾分布，且不依赖于传统的低阶矩方法，克服了以往方法在样本复杂度上的限制。该算法在样本和时间复杂度上均为多项式级别。

**🔧 技术方法**

使用了一种新的高维稀疏傅里叶变换方法来学习混合模型，结合了现有的统计估计技术。

**📊 数据集**

使用了多种分布的样本，包括拉普拉斯分布和均匀分布等，且在样本中允许存在一定比例的异常值。

**📈 对比分析**

与传统的低阶矩方法相比，该算法在样本复杂度上显著降低，能够在没有最小分离要求的情况下有效学习参数。性能上，算法在多项式时间内成功恢复了混合模型的均值和权重。

**⚠️ 局限性**

该研究的局限性在于算法的适用性可能受到特定分布特性的限制，尤其是在处理未知协方差的情况下。此外，算法在处理极端重尾分布时的表现尚需进一步验证。

---

## RL-AWB: Deep Reinforcement Learning for Auto White Balance Correction in Low-Light Night-time Scenes

**arXiv ID:** 2601.05249 | [PDF](https://arxiv.org/pdf/2601.05249v1)

**作者:** Yuan-Kang Lee `[一作]` (MediaTek Inc.), Yu-Lun Liu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 1145 | **OpenAlex IDs:** https://openalex.org/A5101674908

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning` `Reinforcement Learning` `Image`

### 📋 论文摘要

**🎯 论文内容**

提出 RL-AWB：结合统计灰像素检测与深度强化学习的夜间自动白平衡系统，并设计 SGP-LRD 灰像素算法

**💡 创新点**

创新点在于：①将强化学习用于参数自适应调节，提升单图像白平衡精度；②引入两阶段课程学习实现样本高效训练；③发布首个跨传感器夜间数据集 LEVI，验证跨摄像头泛化

**🔧 技术方法**

采用深度强化学习（SAC）、统计色彩一致性算法、灰像素检测、局部反射差分、两阶段课程学习等技术

**📊 数据集**

使用 LEVI（700 张双摄像头夜景 RAW 图像）和 NCC 数据集，并在 Gehler‑Shi 等日间数据集做对比

**📈 对比分析**

与传统统计方法和深度学习基线在 5‑shot 及跨数据集评估中比较，RL‑AWB 在角度误差上往往优于现有方法，展示出更强的跨传感器泛化能力

**⚠️ 局限性**

局限性包括：仍受低光噪声影响，对多光源混合场景处理不足；强化学习需要多步策略，训练与推理成本相对较高

---

## RoboVIP: Multi-View Video Generation with Visual Identity Prompting Augments Robot Manipulation

**arXiv ID:** 2601.05241 | [PDF](https://arxiv.org/pdf/2601.05241v1)

**作者:** Boyang Wang `[一作]` (Shanghai AI Laboratory), Jiangmiao Pang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Robotic Intelligence` `Diffusion model` `Vision Language Model` `Video`

### 📋 论文摘要

**🎯 论文内容**

开发了一种多视角视频扩散模型RoboVIP，利用视觉身份提示对机器人操纵数据进行背景/桌面场景的时序一致性增强；

**💡 创新点**

创新点在于①引入视觉身份提示（利用实例图像作为条件）以补充文本提示的低层细节；②实现多视角、时间一致的扩散；③构建可自动化的视觉身份库；④采用LoRA在大模型上轻量细调；

**🔧 技术方法**

采用多视角视频扩散（Wan2.1 I2V）、LoRA、视频分割（SAM2、OneFormer）、动作引导分割、CLIP评分、Qwen2.5‑VL生成文本、Causal VAE编码身份图像等技术；

**📊 数据集**

使用BridgeV1/V2、Droid、SimplerEnv、Frank 3机器人数据，以及从大规模机器人数据中构建的视觉身份池进行训练与评估；

**📈 对比分析**

与RoboEngine、Cosmos‑Transfer2.5、Octo、π₀等基线比较，在仿真中Octo成功率从12.2%提升至18.5%，π₀从17.25%提升至29%；在真实机器人上DP模型从7/10提升至10/10；多帧历史下RoboVIP保持更高成功率；

**⚠️ 局限性**

受限于当前分割、VLM与开词分割的精度、单视角仿真测试不足、缺乏长时序多视角评估及对高频视觉漂移的鲁棒性不足；

---

## Information-Theoretic Limits on Exact Subgraph Alignment Problem

**arXiv ID:** 2601.05173 | [PDF](https://arxiv.org/pdf/2601.05173v1)

**作者:** Chun Hei Michael Shiu `[一作]` (University of British Columbia), Lele Wang `[通讯]` (University of British Columbia)

**通讯引用:** 5082 | **OpenAlex IDs:** https://openalex.org/A5100383682

**关键词:** `Information Theory` `Graph Neural Network` `Graph`

### 📋 论文摘要

**🎯 论文内容**

提出 Erdos–Rényi 子图配对模型，研究在随机图中定位嵌入子图的子图对齐问题，并给出信息理论极限。

**💡 创新点**

创新性在于将子图同构问题与图对齐结合，给出近乎紧致的成功与失败阈值，并通过结构熵构造逆向证明。

**🔧 技术方法**

使用随机图方法（第一/第二矩法、典型性分析）、信息理论（结构熵、相对熵）以及 MAP/暴力搜索算法。

**📊 数据集**

仅使用理论模型，未采用真实数据集；实验基于 Erdős–Rényi 随机图模拟。

**📈 对比分析**

相较于既往的子图同构与植入团极限，阈值更严格；在满足条件下可实现 w.h.p.，逆向证明表明此阈值近似最优。

**⚠️ 局限性**

局限在于仅给出暴力可行性，算法复杂度极高；对齐阈值在一般参数下仍有 log m 余量；未讨论高效近似算法或非 Erdős–Rényi 模型。

---

## Reverse-engineering NLI: A study of the meta-inferential properties of Natural Language Inference

**arXiv ID:** 2601.05170 | [PDF](https://arxiv.org/pdf/2601.05170v1)

**作者:** Rasmus Blanck `[一作]` (University of Gothenburg), Stergios Chatzikyriakidis `[通讯]` (University of Crete)

**通讯引用:** 825 | **OpenAlex IDs:** https://openalex.org/A5020791896

**关键词:** `Computation and Language` `Classification` `Generation` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

### 📋 论文摘要

**🎯 论文内容**

提出三种模态读法（经典条件、严格条件和存在性导入）来解释NLI标签，并分析其元推理属性；通过生成新数据并在SNLI上训练模型，检验哪种读法与模型预测最匹配。

**💡 创新点**

首次将模态逻辑与NLI标签结合，系统比较三种读法对元推理一致性的影响，并通过自动生成的数据验证模型的内在一致性。

**🔧 技术方法**

使用模态逻辑框架（K逻辑）定义NLI关系，生成LLM（Llama3.2、Llama3.3、DeepSeek‑R1）生成的新样本，训练BERT、RoBERTa+SE等模型进行推理一致性评估。

**📊 数据集**

主要使用SNLI数据集及其自动生成的扩充样本（共约75k训练、30k测试），并构建“推断”测试集来检验元推理。

**📈 对比分析**

对比模型在SNLI与扩充数据上的性能，发现模型在推断项上得分普遍在90%以上；在元推理实验中，存在性导入读法与模型预测的吻合度最高，表明该读法更符合模型内部逻辑。

**⚠️ 局限性**

局限包括生成数据可能包含标签错误或偏差，元推理实验依赖模型准确率，且不同读法的区分度仍有限；未来需进一步消除数据噪声并扩展对不同NLI数据集的逻辑分析。

---

## Inapproximability of Counting Permutation Patterns

**arXiv ID:** 2601.05166 | [PDF](https://arxiv.org/pdf/2601.05166v1)

**作者:** Michal Opler `[一作]` (Czech Technical University), Michal Opler `[通讯]` (Czech Technical University)

**通讯引用:** 20 | **OpenAlex IDs:** https://openalex.org/A5006649123

**关键词:** `Data Structures and Algorithms`

### 📋 论文摘要

**🎯 论文内容**

论文探讨了排列模式的检测和计数问题，特别是长度为k的模式在长度为n的排列中的出现情况。

**💡 创新点**

论文强烈反驳了Ben-Eliezer等人关于近似计数比精确计数更容易的猜想，证明在指数时间假设（ETH）下，无法在特定时间复杂度内近似计数长度为k的模式的副本数量。

**🔧 技术方法**

使用了复杂性理论和参数化复杂性的方法，特别是基于ETH的下界证明。

**📊 数据集**

没有具体提到使用的数据集，主要是理论分析。

**📈 对比分析**

与现有方法的比较显示，无法在时间复杂度f(k)·n^o(k/log k)内近似计数，匹配了精确计数的条件下界，表明现有的近似计数算法在复杂性上是有限的。

**⚠️ 局限性**

限制在于在ETH假设下，无法实现有效的近似计数算法，且对近似误差的界限无法显著改善。

---

## GenAI-DrawIO-Creator: A Framework for Automated Diagram Generation

**arXiv ID:** 2601.05162 | [PDF](https://arxiv.org/pdf/2601.05162v1)

**作者:** Jinze Yu `[一作]` (AWS Generative AI Innovation Center), Dayuan Jiang `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `Graphics` `Generation` `Transformer` `Large Language Model` `Prompt Engineering` `Text` `Image`

### 📋 论文摘要

**🎯 论文内容**

开发了 GenAI‑DrawIO‑Creator，一个利用 Claude 3.7 自动将自然语言描述转换为 draw.io XML 并支持实时交互式编辑的框架，亦可将图像复现为可编辑图表。

**💡 创新点**

首次将通用 LLM 与专门的提示工程、XML 校验、实时流式输出、多模态图像解析以及版本历史结合，构建无需微调即可生成结构化图形的全流程系统。

**🔧 技术方法**

使用 Anthropic Claude 3.7（通过 Amazon Bedrock）、Next.js 前端、XML 解析与校验模块、实时流式响应、视觉输入解码、专门化提示与语义校验技术。

**📊 数据集**

针对 10 个人工构造的基准任务（基础设施图、流程图、组织架构图、UI 线框图）收集描述与对应参考图，形成内部评测数据集。

**📈 对比分析**

采用语义准确度、结构有效性、布局清晰度、响应时间、token 使用率和校正迭代次数等指标评估；首次尝试平均 94% 语义准确度，9/10 场景生成有效 XML，布局清晰度 4.34/5，平均响应时间 7.4 秒，修正次数极低。

**⚠️ 局限性**

仍易误解空间关系、对特殊图形类型适应性差、单张图表元素数目超过 20 时准确性下降，图像‑>图表复现对复杂图形精度不足，偶有错误需人工纠正。

---

## Safe Continual Reinforcement Learning Methods for Nonstationary Environments. Towards a Survey of the State of the Art

**arXiv ID:** 2601.05152 | [PDF](https://arxiv.org/pdf/2601.05152v1)

**作者:** Timofey Tomashevskiy `[一作]` (McMaster University), Timofey Tomashevskiy `[通讯]` (McMaster University)

**通讯引用:** 9 | **OpenAlex IDs:** https://openalex.org/A5050619633

**关键词:** `Machine Learning` `Reinforcement Learning` `Safty and Privacy` `Review/Survey Paper`

### 📋 论文摘要

**🎯 论文内容**

本文综述了在非平稳环境下安全持续在线强化学习（COSRL）的研究现状，提出了适应机制和约束形式的分类，梳理了相关挑战与发展趋势，并给出了未来研究方向。

**💡 创新点**

创新点在于首次系统性地将 COSRL 方法按适应机制（被动、反应、快速主动、主动）和约束类型（预定义、可调、数据驱动、时间依赖）进行分层分类；同时指出了当前研究普遍缺失的硬约束、可预测安全状态、以及调整速度与安全性之间的明确关系。

**🔧 技术方法**

主要采用综述与分类方法，对已有文献中的算法框架、优化方法、风险准则、约束实现以及非平稳性类型进行整理和对比；未采用实验技术，而是基于文献综述、理论分析和经验总结。

**📊 数据集**

由于是综述性工作，未使用具体实验数据集；引用了多篇已有工作中的数据集与环境，但未自行实验。

**📈 对比分析**

对比方法主要通过对比已发表的 COSRL 算法在安全保证、适应速度、优化目标与约束满足等维度进行概念性讨论；未给出统一的数值性能指标或实验结果。

**⚠️ 局限性**

局限性包括：1) 仅聚焦单智能体，未涉及多智能体或灾难性遗忘；2) 综述缺乏定量评估与统一基准，无法直接衡量各方法的实际表现；3) 对非平稳性模型和约束形式的定义仍不够统一，导致跨方法对比困难；4) 未提供实现细节与可复现代码。

---

## MineNPC-Task: Task Suite for Memory-Aware Minecraft Agents

**arXiv ID:** 2601.05215 | [PDF](https://arxiv.org/pdf/2601.05215v1)

**作者:** Tamil Sudaravan Mohan Doss `[一作]` (Microsoft), Balasaravanan Thoravi Kumaravel `[通讯]` (Microsoft Research)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Agentic AI` `Text` `Benchmark`

### 📋 论文摘要

**🎯 论文内容**

该工作构建了一个基于Minecraft的记忆感知混合主动LLM代理评估基准和可复现的测试框架，任务由专家玩家的共游产生并规范化为模板。

**💡 创新点**

其创新之处在于从实际玩家共游中提炼用户编写的任务模板，采用受限知识策略防止隐藏信息作弊，使用一次性澄清与短计划预览实现混合主动交互，用可机检查的验证器从游戏内部证据判定结果，并将记忆写入可视化存储供后续回溯。

**🔧 技术方法**

该框架基于Mineflayer API进行游戏交互，使用GPT‑4o进行计划与代码生成，配合轻量级代码审查器、内存模块和验证器，并通过JavaScript实现动作执行。

**📊 数据集**

任务集包含44个专家共游生成的任务，共计216个子任务；验证器与日志构成公开数据集供复现。

**📈 对比分析**

通过与8名经验玩家的实时共游评测，GPT‑4o在216个子任务中出现71次失败，子任务级失败率约33%，为该基准提供了初步性能基线。

**⚠️ 局限性**

限制包括仅使用单一模型GPT‑4o，无跨模型对比；任务规模有限，可能存在采样偏差；受限知识策略与实时共游导致环境噪声；评测指标仅为通过/失败，缺乏效率与部分成功度量；记忆可持续性与泛化能力仍待提升。

---

## Multivector Reranking in the Era of Strong First-Stage Retrievers

**arXiv ID:** 2601.05200 | [PDF](https://arxiv.org/pdf/2601.05200v1)

**作者:** Silvio Martinico `[一作]` (University of Pisa), Rossano Venturini `[通讯]` (University of Pisa)

**通讯引用:** 1608 | **OpenAlex IDs:** https://openalex.org/A5084138015

**关键词:** `Information Retrieval` `Retrieval` `Compression` `Computational Efficiency` `Retrieval-Augmented Generation` `Text`

### 📋 论文摘要

**🎯 论文内容**

重新实现并比较多种多向量检索方法，提出将聚合阶段改为基于学习稀疏检索的文档级两阶段检索，并结合多向量精排，显著降低查询延迟。

**💡 创新点**

创新点在于：① 用学习稀疏检索取代传统的基于 token 的聚合，形成更紧凑、语义更连贯的候选集；② 在精排阶段引入多种压缩（OPQ、MOPQ、JMPQ）和自适应剪枝/早停策略，实现效率与内存的最佳平衡。

**🔧 技术方法**

技术包括：学习稀疏检索（LSR）、多向量 late‑interaction 重新排序、MaxSim 交互、产品量化（OPQ、MOPQ、JMPQ）、候选剪枝 (CP)、早停 (EE)、HNSW 图索引等。

**📊 数据集**

使用公开的 passages 数据集（约 880 万段落）和 -pooled 数据集（约 240 万段落）进行实验评估。

**📈 对比分析**

与 EMVB、IGP、WARP、XTR 等现有基线进行对比，实验证明在相同内存与编码器配置下，提出的两阶段管道可实现最高 24× 的速度提升，同时保持甚至提升 MRR@10 / Success@5 等效果指标。

**⚠️ 局限性**

局限性包括：对多向量精排的依赖导致仍需处理高维向量；某些压缩方案（如 JMPQ）需要额外的监督训练；学习稀疏检索在查询稀疏时性能略逊；在极大规模语料或多语言场景下的可扩展性尚待进一步验证。

---

## QNeRF: Neural Radiance Fields on a Simulated Gate-Based Quantum Computer

**arXiv ID:** 2601.05250 | [PDF](https://arxiv.org/pdf/2601.05250v1)

**作者:** Daniele Lizzio Bosco `[一作]` (University of Udine), Vladislav Golyanik `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 2888 | **OpenAlex IDs:** https://openalex.org/A5080103406

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Neural Radiance Field` `Image`

### 📋 论文摘要

**🎯 论文内容**

设计并实现了两种混合量子-经典的 QNeRF 模型，用于从 2D 图像学习 3D 视角合成。

**💡 创新点**

引入 Full QNeRF 与 Dual-Branch QNeRF 两种架构，利用量子幅度嵌入、局部测量与可学习输出缩放，在保持模型紧凑的同时提高表达力与噪声鲁棒性。

**🔧 技术方法**

使用量子幅度嵌入、可变参数量子电路（R_Y 旋转+密集/部分纠缠层）、局部 Z 测量、输出缩放，以及对称读取错误和高斯噪声实验，基于 Pennylane 与 Qiskit 进行仿真。

**📊 数据集**

采用 Blender 合成数据集（materials、ficus、lego、drums 等）和 LLFF 实际户外数据集（horns、fern、trex、room）。

**📈 对比分析**

与经典 NeRF 基线在相同分辨率下对比，Full QNeRF 参数不足一半却在 Blender 数据集平均提升约 2 dB PSNR；Dual-Branch 与经典相当但噪声容忍度更高，整体在 LLFF 也保持相近或略优。

**⚠️ 局限性**

训练时间长（CPU 量子仿真），在实际硬件上噪声、采样、梯度计算仍是挑战；模型规模受量子位数限制，需更高效的模拟与硬件支持。

---

## Pixel-Perfect Visual Geometry Estimation

**arXiv ID:** 2601.05246 | [PDF](https://arxiv.org/pdf/2601.05246v1)

**作者:** Gangwei Xu `[一作]` (Huazhong University of Science and Technology), Xin Yang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 23660 | **OpenAlex IDs:** https://openalex.org/A5100719715

**关键词:** `Computer Vision and Pattern Recognition` `Depth Estimation` `Generation` `Transformer` `Diffusion model` `Flow-based Model` `Image` `Video` `Point Cloud`

### 📋 论文摘要

**🎯 论文内容**

提出 Pixel-Perfect Depth (PPD) 与 Pixel-Perfect Video Depth (PPVD)，通过像素空间扩散 Transformer 直接生成无飞行像素的高质量深度图并输出精确点云。

**💡 创新点**

创新点在于：① Semantics-Prompted DiT 用语义提示提升像素级扩散的全局一致性与细节恢复；② Cascaded DiT 采用粗到细的 patch 尺寸策略降低计算开销；③ Semantics-Consistent DiT 与 Reference-Guided Token Propagation 使长视频保持时空一致性且不增加显著计算成本。

**🔧 技术方法**

采用 Flow Matching 的像素空间扩散、Diffusion Transformer (DiT)、多视角语义编码、语义提示机制、参考引导令牌传播、Transformer 纯粹无卷积架构以及 VAE‑free 的深度生成流程。

**📊 数据集**

训练使用 Hypersim、UrbanSyn、UnrealStereo4K、VKITTI、TartanAir、IRS、PointOdyssey 等合成与模拟数据；评估基准为 NYUv2、KITTI、ETH3D、ScanNet、DIODE、Bonn 等真实数据集。

**📈 对比分析**

与现有判别式与生成式深度模型（Depth Anything v2、MoGe2、Marigold、DepthCrafter、Video Depth Anything 等）在 AbsRel、δ1、边缘 Chamfer Distance 等指标上对比，PPD 在所有生成式模型中以 0.07 的 Chamfer Distance 领跑，PPVD 在视频深度任务上比 Video Depth Anything 提升 38.7%/58.4%（NYUv2/ScanNet）并在四大基准上均刷新 SOTA。

**⚠️ 局限性**

局限性包括：对极长视频仍可能出现轻微时间漂移；推理速度相对传统模型略慢，需较大 GPU 资源；模型主要在高质量合成数据上训练，对极端真实场景的迁移性仍需进一步验证；缺乏对实时性能的系统评估。

---

## ObjectForesight: Predicting Future 3D Object Trajectories from Human Videos

**arXiv ID:** 2601.05237 | [PDF](https://arxiv.org/pdf/2601.05237v1)

**作者:** Rustin Soraki `[一作]` (University of Washington), Roozbeh Mottaghi `[通讯]` (University of Washington)

**通讯引用:** 8455 | **OpenAlex IDs:** https://openalex.org/A5070375939

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Object Tracking` `Pose Estimation` `Transformer` `Diffusion model` `Video`

### 📋 论文摘要

**🎯 论文内容**

利用短视角视频预测物体未来的6-DoF三维轨迹，构建大规模对象轨迹数据集并训练基于扩散变压器的预测模型。

**💡 创新点**

①首次将3D对象动态预测定义为从人类视频中直接学习物体在SE(3)空间中的轨迹；②提出结合几何感知的扩散变压器，能够生成多模态、物理一致的未来轨迹；③通过自动化分割、重建和姿态估计构建了2M+的高质量对象轨迹数据集。

**🔧 技术方法**

采用SAM2、TRELLIS、SpaTrackerV2等现成视觉模型实现对象分割与重建；使用PointTransformerV3编码器提取几何上下文；采用Diffusion Transformer (DiT) 与 AdaLN-Zero 进行多模态轨迹预测；训练时加入SE(3)平移、旋转、速度、加速度等辅助损失。

**📊 数据集**

主要使用EPIC‑Kitchens数据集中的手部交互片段进行自动化处理，得到2,073,109条高质量3D轨迹；另外在HOT3D‑Clips上进行验证。

**📈 对比分析**

与自回归Transformer (ObjectForesight‑AR) 以及基于视频生成的Luma AI Ray3 对比；在Epic‑Kitchens和HOT3D‑Clips的8步轨迹预测任务中，ObjectForesight‑DiT在平移和旋转误差上显著优于对手（例如ADE/FDE降低3倍，平均旋转误差仅约7°），并在视频生成基线上表现更好，体现了显式3D推理的优势。

**⚠️ 局限性**

限制在于：仅适用于刚性物体，预测时间窗口较短；未能处理可变形、关节运动或复杂形变对象；模型对异常视角或遮挡仍可能产生误差。

---

## Approximation theory for distant Bang calculus

**arXiv ID:** 2601.05199 | [PDF](https://arxiv.org/pdf/2601.05199v1)

**作者:** Kostia Chardonnet `[一作]` (University of Lorraine), Axel Kerinec `[通讯]` (University of Paris Est Creteil)

**关键词:** `Logic in Computer Science`

### 📋 论文摘要

**🎯 论文内容**

本文构造了距离bang-λ算子（dBang calculus）的逼近语义，定义了其 Böhm 树和 Taylor 展开，并证明了两者的交换定理（Commutation Theorem），从而统一了 Call‑by‑Name 与 Call‑by‑Value 的逼近理论，并把“意义性”（meaningfulness）与 Taylor 正常形的非空性联系起来。

**💡 创新点**

创新点在于：
• 在一个同时兼容 CbN 与 CbV 的计算框架（dBang）中首次完整地给出逼近语义；
• 证明了 Böhm 树与 Taylor 展开在该算子中的互为转换，即逼近层面的“交换定理”；
• 通过翻译与嵌入（embedding）把原始 λ‑算子与距离版本相连，继而把意义性与 Taylor 正常形的非空性统一在同一理论框架内。

**🔧 技术方法**

主要技术手段包括：
• 资源 λ‑计算（resource lambda calculus）和多重集（bag）形式的 Taylor 展开；
• 对 dBang 的明确写法、显式替换、远程归约（distance reductions）以及相应的归约闭包；
• 逼近顺序（approximation order）与理想完成（ideal completion）构造 Böhm 树；
• 归约模拟与平行归约的证明，借助归约因子化与多孔上下文（multi‑hole contexts）；
• 归纳与共归纳（coinduction）证明交换定理与意义性对应关系；
• 翻译与嵌入技巧，将 CbN/CbV 的 λ‑算子映射到 dBang。

**📊 数据集**

无实验数据集；本工作为理论性质证明，未涉及数据集或数值实验。

**📈 对比分析**

比较与性能方面：
• 本文未给出实验或性能评估，而是通过形式化证明与定理来展示结果的正确性与完整性；
• 主要与之前的 CbN/CbV 逼近理论以及 Proof‑structure 相关工作在理论层面做了对比，说明其在统一框架下的优势。

**⚠️ 局限性**

局限性与开放问题：
• 对于完整的 dBang 计算器，意义性与 Taylor 正常形非空性等价性仅在从 λ‑算子翻译而来的子语言（CbN/CbV 片段）上成立；
• 对于更广泛的 dBang 术语，仍无法完全证明意义性与非空 Taylor 正常形之间的双向对应；
• 证明依赖于显式替换与远程归约的特定规则，若改为其他归约策略需重新验证；
• 未来研究需探讨是否能在更大的子语言或 Proof‑structure 环境下实现同样的等价性。

---

