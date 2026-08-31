# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-31 | 今日论文总数: 483

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. How Do Linear Probes Emerge? A Circuit-Tracing Framework with Concept-Targeted Attribution

**arXiv ID:** 2608.27510 | [PDF](https://arxiv.org/pdf/2608.27510v1)

**作者:** Vedant Palit `[一作]` (University of Toronto), Zhijing Jin `[通讯]` (University of Toronto)

**通讯引用:** 2300 | [OpenAlex ID](https://openalex.org/A5016724158)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了概念目标归因（CTA）方法，用以构建针对线性探测器方向的归因图，从而解释内部概念表示如何在语言模型中产生；

**💡 创新点**

创新点在于将归因焦点从输出令牌转移到内部概念方向，展示了概念编码与生成路径的功能分离，并证明归因图结构能预测探测器性能；

**🔧 技术方法**

技术包括跨层变换器（Cross‑Layer Transcoder）来生成稀疏特征图、线性探测器、归因图构造、结构特征提取、梯度提升回归和因果消融实验；

**📊 数据集**

实验使用四个常见概念的数据集：毒性（Toxicity）、情感（Sentiment）、推理（Reasoning）和真实性（Truthfulness），并在Gemma‑2‑2B和Llama‑3.2‑1B上评估；

**📈 对比分析**

比较方法是用图结构特征预测探测器验证准确率，取得Spearman相关ρ≈0.91、R²≈0.84，且通过因果消融显示概念路径与生成路径互不重叠；

**⚠️ 局限性**

局限性包括对跨层变换器的依赖，稀疏特征可能遗漏或聚合概念，线性探测器可能捕获风格而非真正概念，且仅验证了四个概念和两种模型，缺乏更广泛的架构和概念覆盖。

---

## 2. HARTS: Efficient Agentic Reinforcement Learning for Hybrid-Attention Models over Arbitrary Rollout Trees

**arXiv ID:** 2608.28158 | [PDF](https://arxiv.org/pdf/2608.28158v1)

**作者:** Boyuan Meng `[一作]` (Ant Group), Zhenxuan Pan `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 HARTS，支持 Agentic RL 上任意回合树的混合注意力训练，利用前缀共享和最小调用规划实现高效、可微分的执行。

**💡 创新点**

创新点包括：① prefix-aware 微批规划与 DP 复制/槽调度的联合优化；② 线性时间最小调用规划和状态恢复，确保分块线性注意力的数值一致；③ 在同一物理布局下兼容全注意力、线性注意力与 MoE，保持激活可重算与梯度传递；④ 通过语义多重性恢复 RL 与 MoE 目标权重，避免显式展开。

**🔧 技术方法**

技术实现：动态规划与 Trie‑aware 贪婪、线性时间最小调用 DP、Chunk‑wise KDA 与 MLA 的分块执行、可微分边界状态传递、全层激活重算、分布式并行（DP/TP/SP/PP/EP/CP）以及对 MoE 路由的多重性加权。

**📊 数据集**

数据集与模型：使用 Claude Code Scaffold 生成的 SWE‑bench 任务回合树作为 Agentic RL 训练数据，采用 Ling‑3.0‑tiny（混合注意力 + MoE）模型进行评估。

**📈 对比分析**

对比方法：与 Tree Training 的 OR‑Tools 分区基线及传统展开 trajectory‑wise 基线在计划时间、compact‑token 数量、DP 负载平衡、F/B/Grad 与 Core 速度提升、对齐余弦相似度以及在线 τ³‑Bench 奖励曲线进行比较。实验表明：compact‑row 压缩 5.62–5.63×，F/B/Grad 4.81–4.87×，Core 4.39–4.63×，且奖励曲线与基线保持相近。

**⚠️ 局限性**

局限性：仍受 KDA 旁路 replay、MLA 视图展开、Packed‑call 启动开销及跨层/并行同步导致的负载不均衡影响；最小调用规划在理论上最优，但在极大模型规模下仍需进一步优化对齐与通信成本。

---

## 3. Tensor-Accelerated Eager Multi-Resolution Grids for Evolving Large-Scale Substrates

**arXiv ID:** 2608.27612 | [PDF](https://arxiv.org/pdf/2608.27612v1)

**作者:** Romain Claret `[一作]` (University of Neuchâtel), Kilian Stoffel `[通讯]` (University of Neuchâtel)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Eager Multi‑Resolution HyperNEAT（EMR‑HyperNEAT），将 ES‑HyperNEAT 的自适应子网发现改为一次性批量查询 CPPN，并通过预先构建多分辨率网格、统一方差过滤实现大规模并行子网生成。

**💡 创新点**

核心创新在于：①将自适应子网发现转化为张量化并行计算，能够一次性评估所有空间位置；②预生成多分辨率网格与批量向量化 CPPN 调用，使得 GPU 上的速度提升可达 12–34×；③引入连接类型分类（前向、后向、水平、循环、全循环），首次使得循环子网在大规模深度下可行；④自下而上方差过滤确保发现的节点是 ES‑HyperNEAT 的超集，从而显著提升求解率。

**🔧 技术方法**

使用 JAX 框架（vmap、JIT 编译）、GPU 张量化、批量向量化查询、内存流式处理、预生成网格与静态张量、连接类型分类、阈值自适应调节。

**📊 数据集**

评估数据集包括 XOR 基准、CRSP/Compustat 财务数据（94,464 条样本）以及视觉辨别等任务。

**📈 对比分析**

与原 ES‑HyperNEAT 以及 PUREPLES 基线在不同深度、种群规模、连接类型下进行对比：在深度 5–7 的 XOR 任务中，GPU 端每代加速 12–34×，30 代累计约 100×；在深度 6 的金融数据上，EMR‑HyperNEAT 比 ES‑HyperNEAT 提升 5.5×；深度 13 通过磁盘流实现 358M 位置的生成；求解率提升显著，如深度 6 XOR 从 33% 提升到 100%。

**⚠️ 局限性**

局限性：①需要一次性评估所有位置，导致额外的 CPPN 调用量；②阈值需要为批量模式重新调节，失去原始自适应性；③深度过大时内存和磁盘 I/O 成为瓶颈；④目前仍缺乏更细粒度的选择性评估机制来进一步压缩计算量。

---

## 4. Attribute Token Arithmetic: Disentangled and Continuous Semantic Control for Visual Autoregressive Models

**arXiv ID:** 2608.28082 | [PDF](https://arxiv.org/pdf/2608.28082v1)

**作者:** Xindi Yang `[一作]` (Monash University), Tien-Tsin Wong `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Attribute Token Arithmetic（ATA），一种在VAR（视觉自回归模型）中通过文本token算术实现属性级可分离、连续且可组合的图像编辑方法。

**💡 创新点**

创新点在于：①在VAR离散token空间中发现并学习可解释的语义向量运算；②只需一张参考图和少量优化即可获得属性方向；③实现跨类别、连续、无训练或大规模监督的属性控制。

**🔧 技术方法**

使用技术包括：VAR多尺度因果Transformer、VAR对齐的文本token空间、轻量级Delta‑Mod注意力模块学习属性方向、token级向量加减实现编辑。

**📊 数据集**

实验使用了自建可控编辑基准（15个prompt×2个种子、6个属性集）和公开的GEdit基准，同时依赖VAR训练所用的公开图像‑文本数据集与CLIP对齐。

**📈 对比分析**

与Concept Slider、VAREdit、Qwen‑Image‑Edit等编辑模型对比，ATA在ΔVQA、I‑LPIPS、Semantic Consistency、Disentanglement等指标均取得最高或相近水平，且用户研究显示在图像保真度和指令遵循度上优于基线。

**⚠️ 局限性**

limitations：受VAR生成先验限制，强属性强度易产生伪影；对全局颜色变换和某些细粒度属性效果有限；跨域概念映射受限于基础模型的知识覆盖。

---

## 5. How Much Can AI Understand? Toward AI-Assisted Sensemaking of Collaborative Discussion in Groups with Shared History

**arXiv ID:** 2608.27799 | [PDF](https://arxiv.org/pdf/2608.27799v1)

**作者:** Soobin Cho `[一作]` (University of Washington), David W. McDonald `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对共享历史的协作讨论，提出了一种AI辅助感知模型，帮助用户在讨论的早期阶段提取、优先排序、关联并呈现论点、社区规范、人物及其上下文信息，从而实现更高效的讨论理解。

**💡 创新点**

创新点在于：①将讨论感知拆分为六类信息（论点、社区规范、规范上下文、人物、人物上下文、讨论主题）；②构建了一个以Pirolli‑Card模型为核心、AI系统覆盖早期阶段的感知框架；③将AI的“解释性工作”可调度，从低到高，兼顾用户自主性与系统辅助度；④系统在设计时强调“可解释性”和“人类最终解释权”，提出针对偏见与同质化的风险防护。

**🔧 技术方法**

主要技术包括：自然语言处理与大语言模型（LLM）进行文本提取与摘要；信息优先级排序与关系映射算法；可视化展示与交互式摘要工具；以及基于Wiki编辑历史与用户页面的社群上下文检索。

**📊 数据集**

数据集来源为维基百科的讨论页面与相关的编辑历史、用户页面及政策/指南文档；并通过两项实证研究（未公开详细数据）收集经验丰富的维基编辑者在有无AI辅助下的讨论感知过程。

**📈 对比分析**

目前未提供量化对比实验或性能指标；论文主要通过用户研究展示AI辅助感知工具在帮助新成员快速理解讨论、降低信息筛选成本方面的潜在优势，缺乏客观精度、召回率或用户满意度等定量评估。

**⚠️ 局限性**

局限性包括：①AI解释工作可能产生幻觉或偏见，导致重要信息被忽视或被错误解释；②对规范与人物上下文的理解深度受限，难以完全把握主观与情感因素；③系统对高程度解释工作的依赖会削弱用户主动思考，可能产生同质化与决策偏差；④缺乏公开评测与跨语境验证，适用性与泛化性待进一步验证。

---

## 6. Ada-TokenCom: Rate-Adaptive Token Communications via Large-Model-Driven Token Compression and Generation

**arXiv ID:** 2608.28086 | [PDF](https://arxiv.org/pdf/2608.28086v1)

**作者:** Zijun Zhang `[一作]` (Beijing Institute of Technology), Kaibin Huang `[通讯]` (University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种率自适应的Token通信框架 Ada-TokenCom，利用大型自回归模型的概率预测与算术编码实现低比特率语义通信，并在接收端采用模型生成完成未传输的尾部Token。

**💡 创新点**

创新点包括：①混合重建/生成机制，根据截断长度动态决定传输头部Token与生成尾部Token；②基于 Lyapunov 优化的交叉层率与 MCS 自适应策略；③将大模型 AR 概率直接用于算术编码，显著降低码率。

**🔧 技术方法**

使用技术有：大型自回归 Transformer（LlamaGen、Taming Transformer）、算术编码（AC）、VQGAN 量化分词器、Lyapunov 优化的资源调度、Rayleigh 块 fading 传输仿真等。

**📊 数据集**

主要数据集为 ImageNet‑V2（评估），ImageNet‑1K（模型预训练），LAION‑COCO（文本条件），以及对应的 VQGAN 量化代码本。

**📈 对比分析**

与传统压缩码（bmshj2018‑Hyperior、HiFiC、SwinJSCC）、无 AR 的 TokenCom 以及 Diffusion‑based GenSC 进行对比。Ada‑TokenCom 在 0.006–0.047 bpp 低码率区间实现 90%+ 码率压缩，CLIP 相似度 0.92–0.97；在无线场景下相较 SwinJSCC 在低 CBR 时 CLIP 提升 21–47%，LPIPS 降低 60%，PSNR 略低。

**⚠️ 局限性**

主要局限：AR 推理导致解码延迟显著（尤其是完整序列的前缀重算），需较大模型推理资源；目前仅验证图像单模态，未涵盖视频或多模态的高效生成与速度平衡问题。

---

## 7. A Survey on Rubric-Guided Reinforcement Learning for Language Models

**arXiv ID:** 2608.27505 | [PDF](https://arxiv.org/pdf/2608.27505v1)

**作者:** Zifei Shan `[一作]` (Tencent), Fangning Shao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述并统一了rubric-guided RL研究，提出了将constitution视为先验、rubric视为后验的贝叶斯框架，并给出了从固定原则到自进化的完整分类与词法分析；

**💡 创新点**

创新点在于把rubric-guided RL框架形式化为贝叶斯过程，系统性地把先验、后验、聚合和优化整合，并首次从语言学视角剖析细粒度、语义漂移与语言奖励劫持等关键问题；

**🔧 技术方法**

采用贝叶斯建模、强化学习算法（PPO、GRPO、DPO等）、自然语言rubric生成与评估模型（LLM judge、PRM、RLVR等）以及多模态与agentic扩展技术；

**📊 数据集**

引用了多种公开基准数据集与任务（如OpenAI RLHF、MATH、LLM评估、跨语言评测等）来说明各方法在不同场景下的适用性；

**📈 对比分析**

由于是综述性质，本文未给出统一实验对比，但对比了各类方法的先验设定、后验形式、聚合方式与优化算法，并指出现有方法在多样性、可解释性和自进化性上的优缺点；

**⚠️ 局限性**

局限在于缺乏统一实验验证、对贝叶斯假设的实证支持、算力与成本考量、跨语言与多模态的系统化评估，以及对实际生产系统的深入探讨。

---

## 8. Relational Knowledge Distillation Brings DNN Representations Close Enough to Humans to Be Aligned Without Supervision

**arXiv ID:** 2608.27877 | [PDF](https://arxiv.org/pdf/2608.27877v1)

**作者:** Yuria Shimizu `[一作]` (University of Tokyo), Masafumi Oizumi `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过将人类心理相似性结构直接迁移到预训练的CLIP ViT-B/16网络，使用Relational Knowledge Distillation（RKD）提升DNN表示与人类表示在无监督匹配下的细粒度对齐，并验证其对未见概念的泛化能力。

**💡 创新点**

①首次证明仅利用人类关系知识即可在无监督评估中实现单对象层面的高匹配率；②发现改进主要来自全局结构的重塑，而非局部邻域的改变；③提供严格的无监督评估框架（GWOT）和跨数据集验证。

**🔧 技术方法**

Relational Knowledge Distillation（RKD）作为无监督关系迁移损失；Gromov–Wasserstein Optimal Transport（GWOT）用于无监督匹配；Representational Similarity Analysis（RSA）用于监督对齐；使用余弦+欧氏距离混合构建相异矩阵。

**📊 数据集**

训练集：ImageNet ILSVRC 2012验证集上的人类相似性判断（约0.5M次8‑rank‑2实验，50000个图像）。评估集：THINGS 1854个概念（1,249个与ImageNet无重叠），使用不同任务（triplet odd‑one‑out）获得的SPoSE嵌入。

**📈 对比分析**

在监督评价（RSA）中，RKD提升Spearman相关系数约0.10–0.16；在无监督评价（GWOT）中，个体匹配率从≈0.6%–3%提升至≈13%–20%（对照为0.08% chance），粗粒度匹配率从≈35%–41%提升至≈58%–61%。此外，k‑NN邻域重叠率变化不显著，表明改进来自全局结构。

**⚠️ 局限性**

①仅使用关系迁移，未保持任务特定信息，可能影响下游分类性能；②评估使用的SPoSE嵌入来自不同测量任务，仍可能引入兼容性偏差；③对单个人的心理结构迁移尚未实现；④在更大或更复杂模型上验证仍待进一步研究。

---

## 9. INSPIRE: An Internalize-Then-Improve Approach for Example-Driven Mathematical Reasoning

**arXiv ID:** 2608.27501 | [PDF](https://arxiv.org/pdf/2608.27501v1)

**作者:** Shuai Wang `[一作]` (Sun Yat-sen University), Liang Lin `[通讯]` (Sun Yat-sen University)

**通讯引用:** 33895 | [OpenAlex ID](https://openalex.org/A5100412937)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 INSPIRE，一种先内部化再提升的策略，用以增强大语言模型在数学推理中基于例子的方法。

**💡 创新点**

创新点在于：① 引入 Reference‑Guided Student Internalization (RGSI) 生成与基准模型分布一致且质量更高的偏好样本；② 采用阶段化的 rubric‑based DPO，将学习拆分为方法导向阶段和正确性细化阶段，从而逐步获得例子驱动的推理能力。

**🔧 技术方法**

主要技术包括：三维评估 rubric（Method、Reasoning、Correctness）、Direct Preference Optimization (DPO)、LoRA 微调、RGSI 生成、vLLM 推理、DeepSeek‑V3.2 作为判定模型。

**📊 数据集**

使用 BrokenMath 训练集（约 1,275 个需要构造例子或反例的证明题）进行训练，并在 CounterMath、GSM8K、MATH500、AIME 2024、GAOKAO‑mathQA、MMLU‑collegeMath 等评测集上评估。

**📈 对比分析**

与 3B–72B 规模的开源模型及多款商用模型对比，INSPIRE 在 CounterMath 上实现了 45.91% 的 F1、84.79% 的例子使用率，甚至超过部分 72B 大模型；在多种外部分布评测上保持或提升性能，显示出泛化优势。

**⚠️ 局限性**

局限性包括：训练数据仅覆盖 1,275 个题目，领域覆盖有限；实验仅在 7B 规模模型上验证，未探讨更大模型的可扩展性；评判依赖 DeepSeek‑V3.2，尽管已用 GPT‑4o 复核但可能仍受判定偏差影响；实验为单次运行，随机性未作更广泛验证。

---

## 10. TerraceMoE: A Cost Model for Hierarchical MoE All-to-All Communication

**arXiv ID:** 2608.27874 | [PDF](https://arxiv.org/pdf/2608.27874v1)

**作者:** Weicheng Xue `[一作]`, Yonghong Tian `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并验证了一个用于Mixture-of-Experts训练中两跳分发的成本模型，能够在通信调用级别对是否采用分级分发进行预测。

**💡 创新点**

创新点在于提出一种门控式验证机制，将模型误差转化为功能失效而非警告，并结合了隐藏宽度、到达链成本以及实现细节的影响，形成了可解释且可复制的阈值体系。

**🔧 技术方法**

采用了基于Hockney的通信模型、PyTorch的operator chain测量、Roofline模型用于专家矩阵乘、以及自定义的模拟器（TerraceMoE）进行验证。

**📊 数据集**

使用了多种数据集和工作负载：C1–C5、G1等包含不同大小、不同群组和不同负载的MoE通信基准，以及包含13.14B参数模型的实际训练数据。

**📈 对比分析**

通过与单跳分发在不同群组、微批大小、负载组合下的延迟对比，发现两跳分发在高层次的层级比大于约3.98（真实链）时能节省通信时间，但在目前测得的1.03比值下并不具备优势；相对性能提升被实现开销显著抑制。

**⚠️ 局限性**

主要局限在于步骤级别的预测失效、缺乏对大规模（>128）节点的覆盖、实现细节（如到达链）对阈值影响较大、且模型对硬件非均匀性（如不规则Fast域）不适用。

---

## 11. Initialization Is Critical: Advancing Federated Short-Term Load Forecasting under Load Heterogeneity via Model Initialization

**arXiv ID:** 2608.27791 | [PDF](https://arxiv.org/pdf/2608.27791v1)

**作者:** Jianing Chen `[一作]` (Pennsylvania State University), Thomas La Porta `[通讯]` (Pennsylvania State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在联邦学习框架下针对短期负荷预测中存在的负荷异质性问题，提出了两种模型初始化策略以减轻客户端漂移并提升预测精度。

**💡 创新点**

创新点在于：①利用公共负荷数据进行预训练，以获得更具代表性的全局初始化；②设计了序列式本地初始化(SLIAvg)，通过服务器按顺序将前一客户端更新后的模型作为下一客户端的初始化，实现更平滑的本地更新和更一致的全局聚合。

**🔧 技术方法**

技术方法包括：联邦学习（FedAvg）与其变体、预训练（迁移学习）、序列式本地初始化（SLIAvg）、MMD衡量时间异质性、响应系数分析等。

**📊 数据集**

使用公开的 Low Carbon London 智能表计数据集（约30户，分时段为半小时）。

**📈 对比分析**

通过与标准 FedAvg、SLI、以及中心化训练进行对比，实验显示两种初始化策略均显著降低了客户端漂移、提升了收敛速度，最终在Transformer和LSTM两种模型上将全局误差从FedAvg的0.0720/0.1491降至0.0633/0.1375（MSE/MAE），与中心化模型的误差差距进一步缩小。

**⚠️ 局限性**

主要局限包括：预训练依赖公共负荷数据，若分布差异大可能导致过拟合；SLIAvg 引入额外的通信与训练延迟，尤其在大规模客户端时显著；两种方法仍未直接解决客户间显著差异导致的个性化需求。

---

## 12. Comparing Classical and Quantum Machine Learning for Regression in High Energy Physics Collision Data

**arXiv ID:** 2608.28084 | [PDF](https://arxiv.org/pdf/2608.28084v1)

**作者:** Tariq Mahmood `[一作]` (University of the Punjab), Alfredo Raya `[通讯]` (Universidad Michoacana de San Nicolás de Hidalgo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文对四种经典机器学习模型及其量子对应模型在LHC碰撞事件回归任务上的性能进行了系统比较。

**💡 创新点**

创新点在于将SVM、ANN、CNN、LSTM与QSVM、QNN、QCNN、QLSTM等四对模型统一在同一实验框架下比较，并展示量子模型参数效率优势。

**🔧 技术方法**

使用了支持向量机、人工神经网络、卷积神经网络、长短期记忆网络及其量子变体的量子线路与经典训练方法。

**📊 数据集**

采用了来自Kaggle的模拟pp碰撞事件数据集，包含e+e-和μ+μ-两种最终态，输入为p_x1,p_y1，目标为p_tl1。

**📈 对比分析**

通过均方误差、平均绝对误差、R^2等指标对模型进行比较，结果显示经典CNN/LSTM在精度上略优，但量子模型仅用几百参数即可达到相近性能。

**⚠️ 局限性**

主要局限在于所有实验均在CPU仿真环境下完成，未在实际量子硬件上测试，且量子模型的深度受NISQ限制导致性能差距未完全消除。

---

## 13. Fine-Tuning Autobidders with Group Relative Policy Optimization

**arXiv ID:** 2608.28199 | [PDF](https://arxiv.org/pdf/2608.28199v1)

**作者:** Anton Safin `[一作]` (Avito Research), Egor Samosvat `[通讯]` (Avito Research)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于Group Relative Policy Optimization (GRPO) 的无critic自主竞价算法AB‑GRPO，用于二价拍卖中的预算分配与点击最大化。

**💡 创新点**

创新点在于将GRPO从LLM微调迁移到竞价领域，利用已调好的线性基线ALM控制器，仅学习一个可裁剪的校正因子，从而完全去除价值网络，降低噪声与训练不稳定性，并在群组归一化的优势估计上实现高效的无critic策略梯度。

**🔧 技术方法**

技术手段包括GRPO的群组相对优势计算、CLIP式策略损失、KL 与熵正则、基线ALM的预算预测与指数式投标、以及通过正则化的多组轨迹评估实现的无critic学习。

**📊 数据集**

在BAT、iPinYou、AuctionNet 三个公开 RTB 数据集上进行离线实验。

**📈 对比分析**

通过与线性、PID、M‑PID、USCB、FAB 等基线以及传统 actor‑critic RL（如PPO）在点击数、转化数、RMSE 与 CPC 等指标上进行对比，AB‑GRPO 在点击与转化上多次排名第一或第二，整体表现稳健且优于所有RL基线。

**⚠️ 局限性**

局限性包括：仅在二价拍卖中验证；未在真实线上环境或实时竞价中评估；仅对 ALM 基线进行校正扩展，未探究对其他基线的迁移效果；训练过程仍比纯线性/ALM 慢，且对群组大小、KL 系数等超参数较为敏感。

---

## 14. A User-Centric Context-Aware Permission Governance Framework for Privacy Control in Default Mobile Applications

**arXiv ID:** 2608.27914 | [PDF](https://arxiv.org/pdf/2608.27914v1)

**作者:** Asmau Yetunde Adeniran `[一作]` (Air Force Institute of Technology), Fortune Daberechi Ifeanyi `[通讯]` (Air Force Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了面向默认移动应用的特征级授权框架“Allow When Needed”，通过上下文敏感的权限治理提升用户对权限交互的理解与控制；

**💡 创新点**

创新点在于将权限授权从会话级迁移至功能级，并结合加权隐私评分机制提供实时可解释的隐私反馈；

**🔧 技术方法**

采用Web模拟平台（Python Flask、HTML/CSS/JS）实现权限请求场景与交互界面，并通过四种授权选项与加权评分模型实现功能级控制；

**📊 数据集**

使用了30个基于六类默认应用（浏览器、信息、地图、相机、健康、电话）的任务场景数据，并通过104名受访者的问卷与8名参与者的可用性测试收集数据；

**📈 对比分析**

通过描述性统计分析问卷结果，并在可用性测试中观察用户决策行为，发现用户对上下文解释和提醒的偏好显著提升，暗示功能级授权可提高用户对隐私决策的信心；

**⚠️ 局限性**

局限性包括自我报告偏差、样本规模有限、仅在Web模拟环境中评估、未涵盖后台隐私泄露、缺乏因果推断与真实世界实验验证。

---

## 15. Real-Valued Hyperdimensional Sequence Representations with Hadamard Product Binding and Shift Equivariance

**arXiv ID:** 2608.28334 | [PDF](https://arxiv.org/pdf/2608.28334v1)

**作者:** Kenny Schlegel `[一作]` (Chemnitz University of Technology), Evgeny Osipov `[通讯]` (Luleå University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在高维向量符号算子中，利用实值位置编码实现序列的绑定与叠加，提出了三种与Hadamard乘法兼容的实值位置编码方法并验证其可行性。

**💡 创新点**

创新点在于将Fractional Power Encoding的相位特性迁移到实值空间，提出Sinusoid变体实现显式的平移等价变换，同时提供了无需傅里叶变换的Cosine与Inverse‑Fourier实现，满足实值Hadamard绑定的所有设计需求。

**🔧 技术方法**

采用随机傅里叶特征（RFF）生成正弦/余弦位置信号、逆傅里叶变换以及Hadamard乘法绑定，并在HDC‑MiniROCKET与HDC‑MultiROCKET‑HYDRA框架下实现序列编码与平移等价操作。

**📊 数据集**

使用UCR时序分类数据集，在MiniROCKET与MultiROCKET‑HYDRA模型上进行实验。

**📈 对比分析**

通过与原始ROCKET模型对比，三种编码在平均准确率提升约1.4%（MiniROCKET）至1.1%（MultiROCKET‑HYDRA）且误差下降可达10%–60%，Sinusoid变体在MultiROCKET‑HYDRA上取得最高准确率，且实现了显式的平移等价变换。

**⚠️ 局限性**

局限性包括Sinusoid变体需将维度翻倍；Inverse‑Fourier变体缺失平移等价性；Cosine变体不支持平移等价；实验仅覆盖时序分类任务，未验证在其他序列处理场景中的效果。

---

## 16. ODMA-based MIMO Massive Unsourced Random Access with Soft-Output Polar Codes

**arXiv ID:** 2608.28085 | [PDF](https://arxiv.org/pdf/2608.28085v1)

**作者:** Tianya Li `[一作]` (China Mobile Research Institute), Chengshan Xiao `[通讯]` (Lehigh University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种面向MIMO大规模无源随机接入（URA）的ODMA（On‑Off Division Multiple Access）传输方案，并在该框架下引入软输出（Soft‑Output）极化码，实现了联合模式检测与数据解码。

**💡 创新点**

创新点包括：① 将ODMA拆分为三段编码（导频、占空比模式、数据段）实现“pilot‑uncoupled”设计；② 构造层次化模式检测框架，先通过相关操作粗略识别候选模式，再使用消息传播（MP）算法迭代更新后采用MAP决策，显著降低搜索空间和计算复杂度；③ 将极化码的软输出信息与MP算法耦合，形成联合模式检测与数据解码（JDD）流程，充分利用软信息提升模式识别和误码性能；④ 在MIMO多用户环境下首次实现基于MMV‑AMP的活动检测与信道估计，并在此基础上实现软输出SCL极化码的迭代解码。

**🔧 技术方法**

使用技术包括：ODMA框架、三段编码方案、MMV‑AMP（多测量向量AMP）用于活动检测与信道估计、相关式粗检、MP‑based模式检测、MAP决策、软输出SCL极化码（SO‑SCL）以及基于MP的联合模式检测与数据解码。

**📊 数据集**

实验使用仿真数据，采用高斯随机码本、随机生成的用户活动集合、Rayleigh衰落MIMO信道以及BPSK调制；主要性能指标为PUPE（Per‑User Probability of Error）和模式检测误码率。

**📈 对比分析**

与现有URA方案（如FASURA、IDMA‑LDPC、IDMA‑Polar、ODMA‑LDPC、Coupled‑ODMA等）对比，所提方案在MIMO大规模天线（M≥30）下实现了>2 dB的PUPE性能提升（在10⁻³误码率目标下），并在多用户负载下保持较低误码率，显示出显著的干扰抑制与能量效率优势。

**⚠️ 局限性**

局限性包括：① 仍需较多迭代次数，导致整体延迟上升；② 软输出SCL极化码的列表解码对硬件实现复杂度高；③ 对极化码长度和码率的设计有限制，需在短码长与复杂度之间做权衡；④ 在极低SNR或极高用户负载时，模式检测误差仍可能导致解码失败。

---

## 17. PACE: Publisher-Adaptive Content Extraction via Agentic Automation

**arXiv ID:** 2608.27466 | [PDF](https://arxiv.org/pdf/2608.27466v1)

**作者:** Zhanlin Liu `[一作]` (ProRata.ai), Munirathnam Srikanth `[通讯]` (ProRata.ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PACE框架，利用LLM进行代理式配置学习，生成可复用的发布者特定抽取配置，并在推理阶段使用固定的确定性抽取模板，实现大规模高质量网页内容抽取。

**💡 创新点**

创新点在于将LLM的灵活性和推理成本分离：通过训练阶段使用LLM解析页面结构并聚合抽取模式，推理阶段不再调用LLM，既保持了适应性，又实现了可扩展的低成本抽取。

**🔧 技术方法**

技术主要包括：LLM（GPT‑5.4）驱动的代理式工作流、页面级分析器与发布者级聚合器、确定性抽取模板、HTML预处理、可选人工反馈与评估驱动的迭代优化。

**📊 数据集**

数据集：1）Fundus新闻抽取数据集（16家新闻发布者，71页验证集）用于正文抽取；2）21个发布者/域的多模态元数据/表格基准，包含5个表格丰富域（Wikipedia、StockAnalysis、BasketballReference、BLS、W3Schools）以及对应的训练/评估页面，元数据/表格评估使用GPT‑5.4生成的参考，图像评估使用HTML内在URL集合。

**📈 对比分析**

与Fundus、trafilatura、news‑please、jusText、BTE、BoilerNet、Boilerpipe等基线对比。正文抽取中，PACE的ROUGE‑L F1为0.9758，接近手工定制的Fundus（0.9804），明显优于其他可扩展基线；多模态抽取中，PACE在元数据准确率（0.755 vs 0.574）、图像F1（0.516 vs 0.318）和表格F1（0.929 vs 0.461）均显著领先，尤其在表格抽取上提升了两倍以上。

**⚠️ 局限性**

局限性：1）多模态评估依赖LLM生成的参考而非人工标注，结果受参考质量影响；2）图像评估使用硬编码的URL集合，未覆盖所有图像类型；3）训练阶段仍需LLM调用，成本不低；4）对新发布者或模板大幅变化的适应性需要进一步验证；5）实验范围主要集中在新闻和表格丰富域，未覆盖商品页、论坛等其他场景。

---

## 18. Guidelines Are Not Rules: Characterizing Terminologies around Visualization Design Guidelines

**arXiv ID:** 2608.27842 | [PDF](https://arxiv.org/pdf/2608.27842v1)

**作者:** Anna L. Chinni `[一作]` (University of Wisconsin-Madison), Daniel Weiskopf `[通讯]` (University of Stuttgart)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并评估可视化指导术语的使用与理解差异，提出更细致的术语框架。

**💡 创新点**

揭示术语歧义对研究传播与实践的影响，并提供术语使用的可操作指南。

**🔧 技术方法**

使用文本挖掘、机器学习标签（Claude Opus）、调查问卷、定性专家访谈等方法。

**📊 数据集**

分析了3,877篇IEEE VIS 1990-2024论文、Dagstuhl Seminar讨论、两轮社区问卷和聚合的可视化文献。

**📈 对比分析**

通过词频、共现、时间趋势、论文内词汇组合等统计与可视化对比，证明术语使用趋于集中且含义差异显著。

**⚠️ 局限性**

研究范围局限于术语在论文中的出现，未深入探讨实际实践效果及跨学科差异。

---

## 19. A Method for Layer Bit-Width Allocation in LLM Quantization via Performance Maximization Under a Quality-Degradation Constraint

**arXiv ID:** 2608.28003 | [PDF](https://arxiv.org/pdf/2608.28003v1)

**作者:** Artem Safronov `[一作]` `[通讯]` (Southern Federal University), Artem Safronov (Southern Federal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对 Gemma-3-1B 语言模型，作者提出了一种基于层级敏感度（SA‑PTQ）指导的 W8A8 混合精度量化方法，分别对 FFN、Attention 以及 lm_head 进行块级量化。

**💡 创新点**

创新点在于：① 使用模型层的量化敏感度热图精确挑选可量化层，避免全模型量化导致的性能损失；② 在 TensorRT‑LLM 的激活直通模式下实现块级量化；③ 手动解决 Gemma‑3 的 lm_head 量化 Bug，并展示了 Attention 量化在硬件层面的潜在瓶颈。

**🔧 技术方法**

所采用的技术包括 SA‑PTQ 敏感度分析、SmoothQuant 校准、TensorRT‑LLM 量化配置、INT8/FP8 TensorCore 加速，以及针对 GEMM/INT8 路径的手工插件调整。

**📊 数据集**

实验使用了一个 5 文本校准集进行 SmoothQuant 校准，并在固定 prompt（约 50 token）上评估生成质量；评测主要以 token/s 速率、SQNR、Top‑1 一致率与 perplexity 变化为指标。

**📈 对比分析**

对比结果显示：5+5 FFN+lm_head 配置可实现 1.110× 的推理速度提升（约 11%），质量保持在 98.9% Top‑1 一致率；更激进的 10+10 或 all26 配置可达 1.191× 加速，但生成质量明显下降；Attention 量化单独使用时并未带来额外速度提升。

**⚠️ 局限性**

主要限制包括：① Attention 量化未能充分利用 INT8 路径，因 Q/DQ 包装破坏了已优化的 fused kernel；② lm_head 量化需手动实现，缺乏官方工具支持；③ 仅在短 context 与 batch=1 条件下评估，长 context、批量多样化的效果未知；④ 现有量化方案对 KV‑cache、不同硬件（如 Blackwell 与 Ampere）的适配度有限。

---

## 20. Report Supervision

**arXiv ID:** 2608.27668 | [PDF](https://arxiv.org/pdf/2608.27668v1)

**作者:** Pedro R. A. S. Bassia `[一作]`, Zongwei Zhou `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种利用放射科报告作为监督信号的肿瘤分割训练框架R‑Super，能在缺乏掩模时显著提升肿瘤检测与分割性能。

**💡 创新点**

创新点在于：①设计了Volume Loss和Ball Loss两种基于报告信息（肿瘤计数、尺寸、位置）的直接监督损失；②利用大型语言模型提取报告中的结构化肿瘤描述；③通过报告仅在训练阶段使用，无需推理时额外输入。

**🔧 技术方法**

技术包括：基于Transformer‑CNN的MedFormer分割网络、Llama 3.1 70B AWQ用于文本抽取、卷积球形核（Ball Convolution）实现定位、Dice与交叉熵组合的损失函数，以及组织前景掩模辅助。

**📊 数据集**

使用的数据集：私有UCSF‑Train（6 718 CT‑Report）、UCSF‑Huge（41 418 CT‑Report）、公开Merlin（1 848 CT‑Report）、PanTS（9 901 CT‑Mask）、JHH（6 212 CT‑Mask）等，覆盖胰腺和肾脏肿瘤，包含数十万报告与数千掩模。

**📈 对比分析**

与标准掩模监督、CLIP预训练、MTL、多任务学习、伪标签、nnU‑Net等六种方法比较，R‑Super在内部与外部验证集上均实现F1‑Score提升高达15 %（对掩模仅训练）及DSC提升约10 %；尤其在少掩模（50个）和大量掩模（1.7 k）两端均优于对比方法。

**⚠️ 局限性**

局限性包括：①对大型语言模型的依赖，若使用小模型性能下降；②报告抽取误差在10–20 %时仍可接受，但超过此阈值会显著退化；③需要先行生成组织掩模，虽然可用少量数据训练但仍增加预处理步骤；④在非对比CT等极端影像上仍受限，需更多报告样本。

---

## 21. Class-Based Heuristic Selection for Solving the Flying Block Puzzle

**arXiv ID:** 2608.27476 | [PDF](https://arxiv.org/pdf/2608.27476v1)

**作者:** Sanyar Ahmadi `[一作]` (University of Tehran), Amanj Khorramian `[通讯]` (University of Kurdistan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对2列飞行块拼图（NP‑complete）设计了一种基于运动学分类的启发式A*搜索（CBHA*），通过类特定可接受启发式和类条件优先级分割实现最优路径规划。

**💡 创新点**

创新点：①提出七类运动学分区树，将状态空间划分为可接受启发式一致的七个类；②为每类设计专属可接受启发式，并证明一致性；③引入类条件优先级（深度优先或垂直距离）以突破f值平台；④综合上述技术显著提升搜索效率。

**🔧 技术方法**

采用技术包括：A*搜索框架、深度优先优先级、类条件启发式选择、运动学分区树、可接受启发式证明、图搜索模式、Python实现及多实例实验评估。

**📊 数据集**

使用了146个手工/随机生成的2列飞行块拼图实例，覆盖七个类（A–G），包含不同高度、占空比和目标位置，确保结构多样性与难度梯度。

**📈 对比分析**

实验对比BFS、标准A*、Depth‑Prioritized A*，评估指标为成功率、节点扩展数、有效分支因子与最大搜索深度；CBHA*成功率93.4%（相较于64%、39%、17%），节点扩展比标准A*低88%，有效分支因子≈3，显著优于基线。

**⚠️ 局限性**

局限性：对极端结构复杂（需大块协同移动）的实例仍易失败；需要人工构造运动学分类与启发式，扩展至更宽或不规则域需重新推导；实验受12GB内存/30min时间限制影响，可能低估不同算法间的真实差距。

---

## 22. Below the Noise Floor: Bimodal Seed Collapse and Distinct Failure Modes in Small-Model Knowledge Distillation

**arXiv ID:** 2608.27729 | [PDF](https://arxiv.org/pdf/2608.27729v1)

**作者:** Dipto Sumit `[一作]` (BRAC University), Farig Sadeque `[通讯]` (BRAC University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了小规模语言模型在函数路由任务中的知识蒸馏稳定性，发现种子导致的双峰崩溃和无声截断错误。

**💡 创新点**

首次揭示在小模型蒸馏中存在双峰种子崩溃和输出截断两种不同崩溃模式，并证明仅延迟或弱化KL梯度即可避免崩溃。

**🔧 技术方法**

采用1.5B Qwen学生与20B Mixture-of-Experts教师，比较八种KD变体与交叉熵基线，使用多种随机种子、多种输入格式及不同KD目标。

**📊 数据集**

使用740条合成医疗/保险/健身API路由示例，按意图划分为V1和V2两种拆分，包含候选函数名与描述。

**📈 对比分析**

对每个配置做3-6个随机种子，报告平均准确率与标准差；大多数KD方案在多数种子与基线相当，但有几种在部分种子跌至55%以下；仅延迟KD和仅排序KD在所有种子稳定，且未超越基线。

**⚠️ 局限性**

种子数有限（3-6），未达到推荐的≥10；实验仅在单一小数据集、单一模型规模和单一API目录，结果可能不具普适性；机制仍是推测而非直接测量。

---

## 23. A Simpler Analysis of the Bansal-Jiang Quasi Monte-Carlo Algorithm via Haar Wavelets

**arXiv ID:** 2608.27986 | [PDF](https://arxiv.org/pdf/2608.27986v1)

**作者:** Jiaheng Cheng `[一作]`, Haotian Jiang `[通讯]` (University Of Chicago)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文对Bansal–Jiang随机QMC方法进行理论分析，给出了其平滑化变差σ_SO的等价表述为Haar–Besov半范数，并利用Haar分解提供更直观、技术简化的误差分析。

**💡 创新点**

①证明σ_SO与Haar–Besov半范数等价，提供经典量化视角；②通过Haar分解避免了Hlawka–Zaremba公式、傅里叶分解及复杂的高频消解，显著简化分析过程。

**🔧 技术方法**

使用功能分析、Haar基分解、子高斯性分析、子Gaussian控制的色散向量、傅里叶展开与解析，以及随机平移平均技术。

**📊 数据集**

无实验数据，全部为理论证明；主要使用理论构造的随机样本和Dyadic盒子集合。

**📈 对比分析**

与传统Monte Carlo (O(σ(f)/√n))和经典QMC(Koksma–Hlawka, O(σ_HK(f)/n))对比，随机QMC在误差上实现 O(σ_SO(f)/n)，在 σ_SO ≪ σ_HK 的情况下显著优于传统QMC；算法复杂度为 O(n^2) 时间、O(n^2) 采样。

**⚠️ 局限性**

仍受维度指数因子影响；需要生成 n^2 个独立均匀样本，算法复杂度高；σ_SO 与函数光滑性有关，非光滑函数误差可能不显著提升；高频分量的控制仍依赖 log n 级别。

---

## 24. Evidential-Based Higher-Order Set Argumentation Framework

**arXiv ID:** 2608.27824 | [PDF](https://arxiv.org/pdf/2608.27824v1)

**作者:** Shuai Tang `[一作]` `[通讯]`, Shuai Tang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108`

**🎯 论文内容**

提出了一种能够同时表达反对、证据支持及其高阶交互的统一框架——证据层级结构化论证框架（EHSAF），并给出了相应的多种语义与逻辑编码形式；

**💡 创新点**

创新点在于：①对传统抽象论证、支持框架及其高阶交互的整合，实现了更通用的论证建模；②引入连续模糊运算及编码等价理论，统一了离散与连续多值语义；

**🔧 技术方法**

采用连续模糊运算（CFOE）和模糊编码（CFNE）技术，利用t‑norm、否定等模糊逻辑运算；

**📊 数据集**

本文未使用传统机器学习数据集，而是基于理论构造与示例的形式化验证；

**📈 对比分析**

通过与已知的三值离散语义、Gödel、Product、Łukasiewicz三种模糊语义的等价性证明，对比实验表明模型集合非空且满足边界与单调性等性质，性能主要体现在理论可解性与固定点存在性；

**⚠️ 局限性**

局限性包括：①对支持环的处理不够严格，可能导致语义不一致；②对大规模图的可扩展性未进行实验评估；③模糊参数选取依赖于具体应用场景。

---

## 25. Representation of syntax in LLMs through the lens of linear distance and similarity-aware entropy

**arXiv ID:** 2608.27813 | [PDF](https://arxiv.org/pdf/2608.27813v1)

**作者:** Juan Pablo Vigneaux `[一作]` (Northwestern University), Matilde Marcolli `[通讯]` (California Institute Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用结构探针从多种Transformer模型的残差流中提取句法树，并将未标记的附件得分（UASL）按每个依赖关系拆分，研究不同关系在嵌入空间中的可辨识性。

**💡 创新点**

创新点在于：①将UASL按标签拆分，发现线性距离和头部词汇多样性的相似性校正熵（similarity‑aware entropy）是解释UASL变异的主因；②基于UASL随距离的变化进行聚类，揭示了句法关系的语义和结构层级；③通过回归模型量化这些因子对UASL的影响，并验证结果在不同模型和层次上保持一致。

**🔧 技术方法**

技术方法包括：结构探针（线性映射 + 最小生成树）; 线性/对数距离衰减模型; 加权最小二乘回归；相似性校正熵的计算（使用词嵌入的余弦相似度）；距离影响下的层次聚类（UPGMA）；残差流检查点的提取。

**📊 数据集**

数据集与模型：使用 Penn Treebank（WSJ）依赖树，覆盖42个关系；对 BERT‑base、DeBERTa‑v3‑base、ModernBERT‑base、GPT‑2‑base、GPT‑J‑6B 等Transformer模型在各残差流层级训练探针。

**📈 对比分析**

比较方法：在每个模型、每层级训练单一探针，计算各依赖关系的UASL；通过加权回归（R²≈0.74）评估距离、熵和散度对UASL的解释力；比较不同模型层次的UASL曲线，证明深层层次对长距离依赖的恢复更好，整体性能在各模型间保持一致。

**⚠️ 局限性**

局限性：①仅训练单个探针，未检验不同语料或随机初始化的稳定性；②未将探针结果与Transformer内部注意力机制关联，未证实模型是否利用这些句法信息；③实验仅限于英语，未探讨词序灵活性更高语言的情况。

---

## 26. Covering 1024 syndromes with 50 columns

**arXiv ID:** 2608.27494 | [PDF](https://arxiv.org/pdf/2608.27494v1)

**作者:** Stephen Wu `[一作]` `[通讯]`, Stephen Wu

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构造了一条新的 50 列奇偶校验矩阵，得到一条 [50,40]₂ 线性码，其覆盖半径为 2，显著改进了已知的 (10,2) ≤ 50 上限。

**💡 创新点**

创新之处在于首次提供了一个更小覆盖半径 2 码，并设计了一个 10 块的 (2,0)-分区，使得该码可作为 QM_2² 结构的种子，从而生成更长长度码并降低覆盖密度。

**🔧 技术方法**

作者使用模拟退火搜索寻找 50 列矩阵，随后进行分区搜索，并用 Python 与 Rust 两套独立验证器进行全枚举验证；同时实现了 QM_2² 传播构造。

**📊 数据集**

所用数据集包括本文给出的 50 列矩阵、Kaikkonen–Rosendahl 51 列基准集，以及存放在 GitHub 上的代码和验证文件。

**📈 对比分析**

通过比较覆盖密度和码长，新的码在 50 长度处密度为 319/256≈1.246，优于之前的 1327/1024≈1.296；经过 QM_2² 传播得到 815 长度密度 1.268、1631 长度密度 1.269，进一步将上界 f(2) 限下至 2601/2048≈1.270；验证过程仅需约两秒。

**⚠️ 局限性**

局限性包括：仅证明了 50 列集的局部最优；是否存在覆盖半径 2 的 49 列集仍未知；在某些 r 值上仍有空缺；而 f(2)=1 的最终可行性仍未确定。

---

## 27. Empowering Local Agriculture: A Deep Learning-Powered Web System for Identifying Bangladeshi Mango Varieties

**arXiv ID:** 2608.28161 | [PDF](https://arxiv.org/pdf/2608.28161v1)

**作者:** Monowar Islam `[一作]` (University of Dhaka), Safaruzzaman Shovo `[通讯]` (University of Dhaka)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文构建了一个包含 2013 张 Bangladeshi 香芋果图像的公开数据集，利用迁移学习训练并比较了 ResNet18、ResNet50 与 EfficientNetB0 三种 CNN，并将最佳模型部署为可在线访问的 Streamlit Web 应用，实现果品种的即时识别。

**💡 创新点**

创新点在于首次公开本土香芋果品种的真实场景图像数据集，采用 EfficientNetB0 进行高效迁移学习，并将训练好的模型完整公开部署到 Web 平台，弥补了以往仅有实验室级模型而缺乏实用工具的空白。

**🔧 技术方法**

使用的技术包括深度卷积神经网络（ResNet18/ResNet50/EfficientNetB0）、迁移学习、图像增强（翻转、旋转、色彩抖动、随机裁剪）、PyTorch 框架以及 Streamlit 框架构建交互式 Web 接口。

**📊 数据集**

采用的图像数据集共 2013 张 3024×4032 像素的高分辨率香芋果照片，分为 9 类（Amrapali、Bari、Fazlee、Harivanga、Kanchon Langra、Katimon、Langra、Mollika、Nilambori），BARI‑4 与 BARI‑7 合并为 Bari 类。

**📈 对比分析**

在 70%‑15%‑15% 的分层拆分下，统一应用数据增强和训练超参数，比较验证/测试集性能；EfficientNetB0 在验证集上达 98.01% 及测试集 97.36% 的准确率，宏平均 F1 为 0.98；ResNet18 与 ResNet50 的表现分别为 86.47% 与 78.55%。

**⚠️ 局限性**

主要限制包括：仅覆盖 9 种品种且样本量相对有限；未考虑成熟度、病害等多任务预测；模型仅以在线推理为主，缺乏离线或移动端部署；在不同地区、光照条件下的泛化能力尚待进一步验证。

---

## 28. Visual Cue Interactions in AR-Guided Needle Insertion: A Prostate Biopsy-Inspired Phantom Study

**arXiv ID:** 2608.27620 | [PDF](https://arxiv.org/pdf/2608.27620v1)

**作者:** Xinrui Zou `[一作]` (Johns Hopkins University), Alejandro Martin-Gomez `[通讯]` (University of Arkansas)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在基于HoloLens 2的AR平台上实现并评估了三种针导引视觉策略（聚焦-上下文窗口、基于颜色的接近反馈、轨迹可视化），并通过在新手与专家两组受试者中进行2×2×2因子实验，测量目标定位精度、完成时长、退行次数、工作负荷等指标。

**💡 创新点**

①将三种不同类型的AR视觉线索统一集成到同一系统；②以双阶层（新手/专家）对比，探讨专业水平对视觉线索效益的调节；③发现颜色反馈与准确性存在负相关的“色彩悖论”，提示信息过载与注意力偏移的潜在风险。

**🔧 技术方法**

Unity 2022.3 + MRTK3 + HoloLens 2；自定义着色器实现聚焦窗口与颜色编码；实时轨迹线条渲染；RANSAC标定、负二项混合模型、线性混合模型等统计方法。

**📊 数据集**

实验数据来自两款物理仿真试剂：①训练试剂（可见目标）用于熟悉操作；②评估试剂（隐藏目标，嵌入6个4 mm球形目标的3D打印骨盆+凝胶模拟软组织），并通过HoloLens 2内置红外相机追踪针尖位置。

**📈 对比分析**

使用双因素混合模型对误差、时长、退行计数进行比较；结果显示：对新手，轨迹线索显著降低误差（≈13%）并减少退行（≈41%），但在无其他线索时时长增加；颜色线索虽提升主观易用性，却在新手中误差上升≈14%；对专家，聚焦窗口降低误差/时长≈12%且减少退行≈52%，颜色线索仅提升时长；多线索组合未产生一致的加法收益。

**⚠️ 局限性**

限制包括：在刚性phantom上进行，缺乏软组织变形与真实患者运动；受试者样本量有限（7名专家、19名新手），性别分布不平衡；实验仅单次，未观察长期适应与学习曲线；缺少眼动或注意力测量，难以直接验证“色彩悖论”的机制；以及对高频繁任务的普适性与临床可转移性未进行评估。

---

## 29. Compared to What? A Human-Anchored Security Benchmark for LLM-Generated Infrastructure-as-Code

**arXiv ID:** 2608.28021 | [PDF](https://arxiv.org/pdf/2608.28021v1)

**作者:** Animesh Shaw `[一作]` `[通讯]` (Indian Institute of Management Kozhikode), Animesh Shaw (Indian Institute of Management Kozhikode)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比人类编写的IaC模板，评估12种大语言模型生成的IaC在安全漏洞密度上的表现，得到模型生成的IaC漏洞密度约为人类的3.2×–3.9×。

**💡 创新点**

首次提供与规模匹配的人类安全基线；采用三引擎扫描实现跨工具一致性；系统性分解“推理”模式（提示式链式思考 vs. 供应商扩展思考）并量化其对安全性的影响；对不完整实验设计使用Skillings–Mack统计方法。

**🔧 技术方法**

静态扫描工具（Checkov、Trivy、KICS）；负二项广义估计方程模型；Spearman、Mann–Whitney、Wilcoxon、Kolmogorov–Smirnov、Skillings–Mack 等统计检验；模型调用 API 的 token 监测。

**📊 数据集**

1,196 个模型生成的IaC（100 个部署场景，12 个配置），634 个人工编写的IaC模板（公开仓库），以及 100 个自然语言部署场景。

**📈 对比分析**

在资源计数相同的前提下，使用 Mann–Whitney U 检验对比模型与人类，结果显示模型漏洞密度显著更高；Skillings–Mack 检验表明不同模型在安全性上存在显著差异；在推理模式对比中，扩展思考模式相对标准生成降低约13%，而链式思考无显著提升；模型平均性能相对人类约为 3.5×。

**⚠️ 局限性**

人类基线模板未针对同一场景编写，导致潜在结构偏差；静态扫描仅检测策略违规，未验证实际可利用性；不同工具对格式支持差异导致“其他”类别异常；模型推理 token 消耗极低，可能低估其效果；完整案例 Friedman 统计无法应用，需采用不完整块方法。

---

## 30. Fast Weight Attention for Continual Learning

**arXiv ID:** 2608.27763 | [PDF](https://arxiv.org/pdf/2608.27763v1)

**作者:** Yifan Zhang `[一作]` (Bytedance Seed), Andrew Chi-Chih Yao `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于“next‑latent”对齐的快权重Transformer模型（Falcon‑1/2/3），并给出了递归、并行以及分块并行三种等价实现；

**💡 创新点**

创新点在于：①将写入流与下一个时间步的键对齐（one‑step shift），②引入NLMS风格的自适应学习率和列级调度，③利用WY/Gram分解实现高效并行求解；

**🔧 技术方法**

使用的技术包括：线性注意力写入、NLMS自适应更新、内积目标、Weyl–Yates分解、Gram矩阵与三角求解、分块并行、GPU上多头并行实现；

**📊 数据集**

数据集未在文中明确列出，推测使用公开的大规模文本语料（如Piles/OpenWebText等）训练语言模型；

**📈 对比分析**

与标准线性注意力、Mamba‑2、RWKV‑7、Kimi Linear Attention等基线进行对比，Falcon系列在相同模型规模下实现了更快的训练速度、较低的显存占用，并保持或略优的困惑度（PPL）表现；

**⚠️ 局限性**

限制包括：缺乏严格的收敛性与泛化性理论保证；对长序列需要滑动窗口统计，导致额外开销；实现对GPU并行性的高度依赖；内积目标下无界最优解风险；模型规模增大时仍受显存与计算成本限制。

---

## 31. Beyond Search-Imitation: Prior-Directed Exploration for Searchless Chess

**arXiv ID:** 2608.27757 | [PDF](https://arxiv.org/pdf/2608.27757v1)

**作者:** Szymon Miłosz `[一作]` (Lodz University of Technology), Szymon Grabowski `[通讯]` (Lodz University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在无搜索的变压器棋类网络上，本文通过自我对弈强化学习微调，使其在单次前向推理下达到人类大师级别。

**💡 创新点**

创新点在于将 MCTS 先验作为正向 KL 参考进行探索正则化，并结合价值不确定性的自适应温度，避免熵正则的无信息扩散与模式崩溃。

**🔧 技术方法**

使用的技术包括 Transformer 政策网络 Chessformer、正向 KL 先验引导探索、单步 TD(0) 自我对弈、熵自适应采样温度、EMA 目标与 V‑trace 权重截断。

**📊 数据集**

使用的数据集主要是 Leela Chess Zero 的 Chessformer 预训练模型，以及 Lichess 的 100,000 题库、20,000 题库和 10,000 题库进行验证。

**📈 对比分析**

对比方法涵盖无正则、熵奖励、反向 KL 锚定和正向 KL 先验，评估指标为谜题解题率、强迫跳棋深度解法率和 Elo 评级；实验显示正向 KL 先验在 2000 步微调后提升约 0.9–1.0% 的解题率，并在 Elo 上提升约 10–20 点，优于其他正则化方式。

**⚠️ 局限性**

局限性在于仅适用于离散可枚举动作空间（如棋类），正向 KL 需要完整先验分布；微调幅度有限，难以显著提升整体强度；且实验仅在单一基模型上验证，缺乏跨域泛化评估。

---

## 32. Informational Antilocality and the Locality Bias in LLMs

**arXiv ID:** 2608.27760 | [PDF](https://arxiv.org/pdf/2608.27760v1)

**作者:** Andrew McInnerney `[一作]` (University of Michigan), Richard L. Lewis `[通讯]` (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究transformer大语言模型在学习k-antilocal语言（即在任何k连续符号窗口内无互信息的语言）时的学习成功与学习速度。

**💡 创新点**

提出离散信息局部度量k-antilocal，构造可调k的合成语言（k-back语言）并与对应控制语言对照，系统评估LLM的学习成功不受局部性影响但学习速度受局部性显著影响，指出autoregressive设计可能是产生此偏好的关键。

**🔧 技术方法**

使用GPT‑2小型模型（124M参数）和DeBERTa‑v3对比实验，基于自定义tokenizer与位置编码，采用交叉熵与KL散度衡量学习成功，利用旋转类生成k-back语言，绘制训练曲线分析学习速度。

**📊 数据集**

合成数据集：k从1到8共8组，分别生成约100M token的语料（80k句子用于训练，10k用于验证，10k留作测试），每组包含k-back语言与其对应的控制语言。

**📈 对比分析**

通过比较k-back与控制语言的验证交叉熵损失曲线，发现两者最终达到相近的最低损失（≈0.596），但k-back语言在前5k训练步内损失显著更高，且随着k增大学习速度下降的时间窗口拉长。

**⚠️ 局限性**

限制：仅探讨单一k-antilocal构造方式、固定句长、词表大小、模型规模和位置编码；控制语言与k-back语言除了局部性外还存在交叉依赖等差异；未验证更大k、更长序列、不同模型尺寸或位置编码类型对局部性偏好的影响。

---

## 33. Remote Human and Robot Interaction for Greenhouse Gardening Using Virtual Reality

**arXiv ID:** 2608.27545 | [PDF](https://arxiv.org/pdf/2608.27545v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 34. Context Localization for Generalized Level-Based Evaluation in Knowledge-Based Systems

**arXiv ID:** 2608.27482 | [PDF](https://arxiv.org/pdf/2608.27482v1)

**作者:** Ondrej Hutník `[一作]` (Pavol Jozef Šafárik University in Košice), Natália Puškárová `[通讯]` (Pavol Jozef Šafárik University in Košice)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并证明了在知识基系统中对通用层级度量进行上下文定位的一致性条件，即在掩码与集合截断两种方式下结果相等的必要与充分条件。

**💡 创新点**

识别了两个独立结构条件——对集合的单调性与归约性（reduction property）——作为保证定位一致性的关键，并将此结论推广到参数化系统与随机过程框架。

**🔧 技术方法**

使用集合论、非可加测度理论、条件聚合算子理论、可测函数与σ‑代数、以及过滤概率空间中的条件期望和信息块聚合等技术手段。

**📊 数据集**

未使用具体数据集，本文为纯理论性研究。

**📈 对比分析**

没有实验对比，性能评估以理论证明的一致性为主。

**⚠️ 局限性**

仅在满足单调性和归约性条件下成立；若不满足，定位一致性无法保证；对点分离或块生成集合的要求较强，且缺乏对连续或高维实际数据的经验验证。

---

## 35. Finding Where the Buck Stops: An Automated Failure Attribution-Based Reflection Framework for Multi-Agent Collaboration

**arXiv ID:** 2608.28264 | [PDF](https://arxiv.org/pdf/2608.28264v1)

**作者:** Xiaoqing Wang `[一作]` (Renmin University of China), Wuqiong Pan `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种诊断-纠正 PPO 强化的反思框架，专门针对多代理 LLM 系统，通过自动失效归因识别决定性错误代理与错误步骤，然后采用反事实推理生成纠正步骤，仅让错误代理产生针对性反思；

**💡 创新点**

创新点在于只让错误代理进行反思以防止记忆污染，使用过程奖励模型实现失效归因，结合反事实推理生成纠正动作，并通过 PPO 对反思模型进行细粒度强化学习；

**🔧 技术方法**

技术包括基于 LLM 的多代理协作、过程奖励模型（PRM）做错误归因、反事实推理生成纠正步骤、反思模型 fine‑tune 与 PPO、回归奖励模型和 RLHF 三阶段训练；

**📊 数据集**

使用 HotPotQA、ChartQAPro、Mind2Web 数据集评估协作效果，并构造 Who & When Pro 失效日志数据集进行归因实验；

**📈 对比分析**

与 Reflexion、Retroformer、COPPER 等基线以及 all‑at‑once、step‑by‑step、binary‑search、random 等归因基线对比，成功率提升 22%–27%，归因准确率约 80%/50%（held‑in/held‑out），并在消除绝对误差距离、归因精度上均优于对手；

**⚠️ 局限性**

局限包括只针对任务导向的多代理协作，未验证开放式或主观任务；仅在英文数据上测试，缺乏多语言与跨文化验证；团队规模较小，未检验在更大团队中的可扩展性。

---

## 36. Adaptive Strategy Generation for Boundary Value Exploration Beyond Numeric Inputs

**arXiv ID:** 2608.28230 | [PDF](https://arxiv.org/pdf/2608.28230v1)

**作者:** Sabinakhon Akbarova `[一作]` (Chalmers University of Technology), Robert Feldt `[通讯]` (Chalmers University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 ABEX，一个基于大语言模型的自适应边界值探索框架，实现在黑盒环境下自动发现多类型输入函数的行为边界。

**💡 创新点**

通过用 LLM 生成并迭代探索策略，替代手工设计的变异算子，并结合质量多样性优化与策略库，首次在非数值输入的黑盒 BVE 中实现自动边界发现。

**🔧 技术方法**

采用大语言模型驱动的多层代理架构（协调器、构思器、策略生成器、执行器），与 MAP‑Elites 质量多样性搜索、程序导数评估及执行反馈循环相结合。

**📊 数据集**

在 20 个 Python 单函数基准上评估，其中包括 10 个整数、5 个字符串、3 个数组和 2 个混合输入的测试用例。

**📈 对比分析**

与 SETBVE（最优数值搜索）和单次 LLM 调用基线对比，在相同候选数预算下，ABEX 在 10/11 个数值函数的 QD‑score 最高，平均提升 2.8 倍/11.7 倍；对非数值函数首次实现边界发现，并在突变测试中平均 86.2% 的 mutation score，显著高于 SETBVE 的 61.9%。

**⚠️ 局限性**

依赖 LLM 计算成本高，对输入规范的质量敏感；在极小或高度约束的函数上可能过度请求新策略；评估仅覆盖小型无状态函数，缺乏对大规模、状态化或多功能系统的验证。

---

## 37. Entity-Memory Graph Retrieval Improves Evidence Coverage in Long-Conversation Question Answering

**arXiv ID:** 2608.27925 | [PDF](https://arxiv.org/pdf/2608.27925v1)

**作者:** Shumao Sun `[一作]` `[通讯]` (Tsinghua University), Shumao Sun (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在长对话问答场景中，仅使用对话内容构建实体‑记忆图，并在检索阶段结合实体匹配、语义融合及时间顺序扩展，比较其与匹配的密集检索在证据召回与最终答案F1上的差异。

**💡 创新点**

①严格分离图结构与向量表示、查询向量、答案生成与评估；②引入仅由对话构成的实体‑记忆图，保留实体共享与时间顺序边；③系统化的组件、参数、cutoff、输入、嵌入、提取器组合的鲁棒性评估；④发现图检索显著提升证据召回但不提升整体答案F1。

**🔧 技术方法**

采用实体抽取（温度0.3）、BM25匹配与实体分数融合、语义余弦相似度加权、一次跳序列扩展、dense backfill；使用OpenAI GPT‑3.5/Turbo模型和text‑embedding‑3‑small嵌入；在LoCoMo benchmark上进行matched dense控制和图检索对比，并使用bootstrap统计和Holm校正。

**📊 数据集**

LoCoMo基准的10条多会话长对话，共1986问答对，图构建仅基于对话内容。

**📈 对比分析**

采用matched dense检索作为对照，保持记忆向量、查询向量、答案协议、评估一致；在上下文预算25下，图检索将官方证据召回从79.75%提升至84.48%（+4.74pp，p<0.001），但整体答案F1仅提升0.5pp，未显著；在不同cutoff、输入格式、参数等实验中，召回优势持续但答案F1差异不显著；在不同提取器/嵌入配置下，召回仍显著，但F1鲁棒性有限。

**⚠️ 局限性**

仅基于10条对话，缺乏跨数据集泛化；未实现正式DRAGON复现；答案质量未提升，取决于读取器和生成模型；图结构对嵌入敏感，需重新调优；检索性能对未缓存问题的提问仍需外部调用；实验环境单一，未评估不同硬件/供应商；缺乏对隐私与可解释性的讨论。

---

## 38. Plan Along the Way: Event-Triggered Foundation-Model Planning for TAMP Execution in Partially Observable Manipulation

**arXiv ID:** 2608.28075 | [PDF](https://arxiv.org/pdf/2608.28075v1)

**作者:** Puru Ojha `[一作]` (IIIT Hyderabad), Antony Thomas `[通讯]` (IIIT Hyderabad)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Robust TAMP，一个基于 LLM/VLM 的分层规划框架，用于在部分可观测环境下进行自适应任务和运动规划。

**💡 创新点**

将对象发现视为独立的重规划事件，使用可见场景状态限定规划，并通过结构化验证与执行后监控实现层级故障上升。

**🔧 技术方法**

结合大型语言模型/视觉语言模型进行任务规划、结构化解析器进行可执行性检查、PDDLStream 与几何工具实现动作底层执行，以及闭环监控与恢复策略。

**📊 数据集**

在 RLBench/CoppeliaSim 的厨房与烤架六个场景（K1–K3、G1–G3）上进行实验。

**📈 对比分析**

与同一执行堆栈下的多种 LLM/VLM 大小、零样本与 ICL 进行对比，结果显示 32B 模型在某些任务上达成 100% TSR，ICL 明显提升非目标物体处理，但大模型并不总是优于小模型。

**⚠️ 局限性**

模型对场景推理的依赖导致对未知物体的处理仍有限，缺乏持续记忆可能导致效率下降，且实验仅在模拟环境中验证。

---

## 39. TI$^2$PS: A Topology-Informed Inverse Design Framework for Stochastic Multicellular Pattern Formation

**arXiv ID:** 2608.27931 | [PDF](https://arxiv.org/pdf/2608.27931v1)

**作者:** Kenji Komiya `[一作]` (NTT Inc), Kunio Kashino `[通讯]` (NTT Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于拓扑信息的逆向代理模型框架（TI^2PS），用于从细胞空间分布中推断细胞水平参数，实现多细胞图案的重建。

**💡 创新点**

创新点在于将持久同调生成的 Betti 向量作为特征，结合逆向 surrogate 模型（GLM/MLP），克服了细胞数目和随机增殖死亡导致的参数估计难题，并显著提升了数据利用效率。

**🔧 技术方法**

采用了拓扑数据分析（TDA）中的持久同调与 Betti 向量、逆向 surrogate 建模（GLM 与 MLP）、以及基准点云处理方法（Pool、PointNet、PointNet++）进行对比。

**📊 数据集**

使用基于基因型参数采样的虚拟斑马鱼色素斑纹模拟数据（共 3721 训练对、931 测试对），并对手工构造的突变斑纹（dali/+, leopard）进行验证。

**📈 对比分析**

与 Pooling GLM/MLP、PointNet、PointNet++ 进行对比；TI^2PS MLP 在 MSE（0.191）、Pearson r（0.636）和 σ_ratio（68.6%）上均优于基准，并且仅使用 10% 训练数据即可优于 PointNet++ 的完整数据模型。

**⚠️ 局限性**

局限性包括：1）逆向模型仅给出点估计，缺乏参数不确定性评估；2）对突变斑纹的重建仍不完美，可能是参数超出训练范围或模型未包含的生物机制；3）Betti 向量的计算成本受筛选步长影响，需权衡精度与效率。

---

## 40. Decoupling is a Necessity: Transformation-Agnostic Decompiled Code Recovery under Optimization and Obfuscation

**arXiv ID:** 2608.27889 | [PDF](https://arxiv.org/pdf/2608.27889v1)

**作者:** Zhiping Zhou `[一作]` (Tianjin University), Wenbu Feng `[通讯]` (Tianjin University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多阶段LLM框架，按语义、结构、词汇三层拆分，恢复受优化和混淆影响的二进制源代码。

**💡 创新点**

首次将检索增强的语义失真数据库与控制流骨架预测器结合，实现对O3及多种混淆的变换无关源恢复。

**🔧 技术方法**

采用LLM推理（DeepSeek-Reasoner）、检索增强（Semantic Distortion Database）、控制流预测（CodeT5-base）、GraphCodeBERT用于语义签名与检索，并配合Tree-sitter AST分析。

**📊 数据集**

构建了超过80,000条二进制-源函数对的基准，覆盖O1–O3优化和BCF、CFF、SPLIT、SUB四种混淆，同时在Semantic Distortion Database中收集约25,000+失真示例。

**📈 对比分析**

与三大基线（DeepSeek-R1、DeepSeek-R1修复版、LLM4Decompile）在Top‑k检索与多维相似度上比较，平均Top‑5检索准确率达83%，整体代码相似度0.66，显著优于基线。

**⚠️ 局限性**

仅在函数级别工作，缺乏跨函数全局上下文；语义数据库覆盖有限，难以处理极端自定义混淆；未通过执行验证，缺少完整功能等价性证明。

---

## 41. The Illusion of $\textit{What If}$: Evaluating the Breakdown of Counterfactual Reasoning in LLMs

**arXiv ID:** 2608.27953 | [PDF](https://arxiv.org/pdf/2608.27953v1)

**作者:** Yucheng Wang `[一作]` (Zhejiang University), Qiang Zhu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个面向开放领域、开放形式、长周期的反事实因果推理基准WhatIfBench，并开发了评估框架PRISM；

**💡 创新点**

创新点在于：①通过开放式问题和多学科场景设计突破传统固定答案评估的局限；②将模型自然语言回答转化为响应派生语义因果图，实现过程层面的因果有效性评估；③将过程指标（Process Metric）与答案层级评分（Rubric Metric）相结合，提供细粒度诊断；

**🔧 技术方法**

技术包括：自然语言分块与语义关系抽取（类似RST的解析），因果边检验的二元判定，Rubric-Based 评分函数，统一的 GPT‑5.4 评估器；

**📊 数据集**

使用220道从 xkcd What If、Worldbuilding、AlternateHistory、Quora 等公开来源收集并规范化的开放式“what‑if”问题，涵盖 STEM、HSS 与 Hybrid 三类场景；

**📈 对比分析**

对六大前沿LLM（GLM‑5.1、DeepSeek‑V4‑Pro、Qwen3‑Max、Gemini‑3.1‑Pro‑Preview、Claude‑Opus‑4.7、GPT‑5.5）在 WhatIfBench 上进行评测；最终得分最高的 GPT‑5.5 仅达 64.62%，显示该任务仍具挑战；PM 与 RM 指标揭示模型在因果结构与解释完整性上的差异；

**⚠️ 局限性**

局限性：数据集规模有限，可能存在英文中心化与题材偏见；PRISM 只评估回答内部的一致性而非真实世界因果真理；解析与评估器可能引入误差；未来需扩展多语言、多学科与多模态场景。

---

## 42. Is Prosody Lost in Translation? Fine-Grained Cross-Lingual Prosody Similarity Across Languages

**arXiv ID:** 2608.27848 | [PDF](https://arxiv.org/pdf/2608.27848v1)

**作者:** Haopeng Xie `[一作]` (Johns Hopkins University), Philipp Koehn `[通讯]` (Johns Hopkins University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对英语-德语、英语-西班牙语、英语-法语三语对的专业配音数据进行细粒度跨语言韵律（音高、能量、时间）相似性分析。

**💡 创新点**

首次大规模细粒度跨语言韵律对比，结合词对齐重排、Spearman 相关性，并对词性类别对韵律对应的影响进行剖析。

**🔧 技术方法**

使用 Whisper 语音转写 + FastAlign 词对齐，pYAAPT 提取音高、Librosa 提取能量；对齐后的词块重排，Spearman 相关度与随机打乱基线对比；POS 归类与留一组分析。

**📊 数据集**

专业配音多语音语料（英德、英西、英法）经 SQUIM 质量滤波、单说话人过滤、语义相似度过滤后的 50+ 万对对齐片段。

**📈 对比分析**

通过 Spearman 相关度与打乱基线比较，结果显示音高均 0.23-0.25、能量 0.17-0.18，均显著高于基线；词性分析表明名词区块对对应性贡献最大；时间特征（持续时长、音素计数）相关系数 >0.87。

**⚠️ 局限性**

配音仅近似真实韵律迁移，未考虑停顿、节奏、声质；自动转写与对齐噪声；仅覆盖三欧语对，难泛化至方言或声调语言；数据不可公开。

---

## 43. UIC-AIHealth4All at ArchEHR-QA 2026: Answer-First Evidence Grounding for Clinical Question Answering

**arXiv ID:** 2608.27467 | [PDF](https://arxiv.org/pdf/2608.27467v1)

**作者:** Mohammad Arvan `[一作]` (University of Illinois Chicago), Rebecca T. Feinstein `[通讯]` (University of Illinois Chicago)

**通讯引用:** 147 | [OpenAlex ID](https://openalex.org/A5012584818)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

未知

**💡 创新点**

未知

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

未知

**⚠️ 局限性**

未知

---

## 44. SinkSLOT: Sinkhorn via Sparse Lifted Optimal Transport

**arXiv ID:** 2608.28262 | [PDF](https://arxiv.org/pdf/2608.28262v1)

**作者:** Ian Hsieh `[一作]` (Sorbonne Universite), Reuben Dorent `[通讯]` (Sorbonne Universite)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 SinkSLOT，一种利用期望切片升维传输计划的稀疏熵正则化 OT 求解器。

**💡 创新点**

创新点在于将独立耦合替换为稀疏的切片升维参考耦合，既提升了求解精度，又把每次 Sinkhorn 迭代成本降至 O(L(N+M))。

**🔧 技术方法**

使用的技术包括稀疏 Gibbs 核、对数域 Sinkhorn、切片投影、升维匹配以及解析梯度与 Hessian 的推导。

**📊 数据集**

在多种合成数据集上验证，包括半月形、八高斯、两环、三维高斯及 64 维高斯，共 10^4 点。

**📈 对比分析**

与 FlashSinkhorn、SROT、Spar‑Sink 等竞争者比较，SinkSLOT 在保持相同误差阈值下实现 3–160 倍加速，显著降低运行时间。

**⚠️ 局限性**

局限性包括对切片数 L 的敏感性、对 ε 的取值影响以及在极小 ε 时对支持集的 combinatorial 依赖。

---

## 45. A Configuration-LP Framework for Connected $k$-Median Clustering

**arXiv ID:** 2608.28081 | [PDF](https://arxiv.org/pdf/2608.28081v1)

**作者:** Kushagra Chatterjee `[一作]` (Indian Statistical Institute), Ali Vakilian `[通讯]` (Virginia Tech)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了连通k‑median聚类的重叠版本，在给定度量空间和无关的连通图下，提出了配置LP框架并给出了近似算法；

**💡 创新点**

创新点在于将连通性直接编码到配置变量中，形成配置LP；结合覆盖LP与根节点最小密度子图oracle，实现了对连通约束的有效松弛，并显著提升了已知的近似比；

**🔧 技术方法**

主要技术包括配置LP建模、列生成/多项式时间的根节点最小密度oracle（通过Steiner树近似实现）、乘法权重更新（Plotkin–Shmoys–Tardos框架）以及随机化分层抽样放大法；

**📊 数据集**

论文为理论性工作，未使用任何真实数据集，仅给出理论分析与证明；

**📈 对比分析**

与Eube等人的先前结果相比，赋值版本在k=ω(log n)时实现了O(log²n)的近似，比其O(k log n)解更优；在一般版本上给出了（O(log n), O(log²n))的双重近似，显著降低了对k的多项式依赖；

**⚠️ 局限性**

局限性在于仍只能得到polylog(n)级的近似，针对精确k中心的多项式时间近似尚无解决方案；根节点最小密度oracle在一般图上只能达到O(log n)的逼近，若想进一步改进需在特殊图族（如minor‑closed图）中寻找更强oracle。

---

## 46. Cross-Session Decomposition Attacks: Scaling Risk and Intent-Aligned Retrieval Defense

**arXiv ID:** 2608.27945 | [PDF](https://arxiv.org/pdf/2608.27945v1)

**作者:** Disen Liao `[一作]` (University of Waterloo), Yaoliang Yu `[通讯]` (University of Waterloo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了跨会话拆分攻击（cross‑session decomposition attacks），将禁止目标拆成看似安全的子查询，随后在模型外重组并评估模型规模对此风险的影响。

**💡 创新点**

提出了组合安全风险（compositional safety risk）理论并给出风险转移上界，证明当参考分布已包含散布的危险线索时，模型规模能提升重组后的危害概率；同时引入了意图对齐检索（intent‑align retrieval）作为轻量级防御。

**🔧 技术方法**

使用理论框架、合成任务与600意图的预训练LLM评估；检索‑分类防御基于MiniLM encoder与对比学习（MultipleNegativesRankingLoss）；实验采用Gemma3、Qwen3系列模型、MiniLM、Harrier、Jina等。

**📊 数据集**

使用合成的隐式目标恢复任务（从Gemma3‑27B分解的支持事实中注入，隐藏原始指令）；600混合领域有害意图集合；WildChat真实用户查询库进行检索基准。

**📈 对比分析**

在合成任务中，宽度越大模型对隐藏指令的负对数似然越低；在预训练LLM实验中，Qwen3和Gemma系列更大模型在固定拆分‑重组管道下的有害能力提升显著；检索实验显示IntentAlign‑MiniLM在Recall@10和nDCG上超过同类模型，且在防御精度与召回上优于大模型。

**⚠️ 局限性**

局限包括跨会话攻击在实际部署中的普遍性未知；评估主要基于LLM裁判，缺乏人类验证；防御需要维护跨会话历史，面临隐私、存储、误检等挑战。

---

## 47. SOMTab: Set-Order Mamba for Efficient Tabular In-Context Learning

**arXiv ID:** 2608.27882 | [PDF](https://arxiv.org/pdf/2608.27882v1)

**作者:** Hao Wang `[一作]` (Renmin University of China), Wei Ma `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的 Tabular 领域 In‑Context Learning 模型 SOMTab，并设计了合成先验 DCH‑TailMix 用于预训练。

**💡 创新点**

创新点在于：① 将无序表格信息映射为稳定的隐式序列（Set‑to‑Order）再使用 Mamba 状态空间混合进行高效表格表示构建；② 在最终预测阶段仅保留注意力机制实现查询‑上下文匹配；③ 引入 DCH‑TailMix 通过度数校正和尾部混合生成更丰富的依赖结构。

**🔧 技术方法**

采用的技术包括：Set‑Order Mamba 结构、注意力‑Mamba 混合框架、DCH‑TailMix 生成的图形先验、标准 PFN 预训练目标与 In‑Context Learning。

**📊 数据集**

使用的评估数据集为 TALENT 分类基准（20+ 数据集）以及 TabArena 对比测试。

**📈 对比分析**

与 Transformer‑基的 TabICLv2、TabPFNv2 等同类模型以及树模型（RandomForest、XGBoost 等）比较，SOMTab 在预测性能（归一化对数损失/准确率/宏 F1）与运行时间、GPU 内存利用率上实现了更优的权衡，尤其在大上下文规模下显著提升效率。

**⚠️ 局限性**

限制包括：① 仍依赖注意力进行最终预测，无法完全消除二次方复杂度；② 预训练需要大量合成任务和计算资源；③ 在极小样本或极高维稀疏场景下的性能尚未充分验证。

---

## 48. The Shape of Power: A Multilingual Framework for Social Power Reasoning in Dialogues

**arXiv ID:** 2608.28144 | [PDF](https://arxiv.org/pdf/2608.28144v1)

**作者:** Farah Atif `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Monojit Choudhury `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出了一套以社会科学理论为基础的多语言社会权力注释框架，并利用电影剧本构建了包含100个场景、15,836条注释实例的SocLens数据集，同时对六大文本LLM与两大多模态LLM在权力推理任务上进行了基准评测。

**💡 创新点**

创新点在于：①将电影剧本作为跨文化自然对话来源，①构建可跨语言扩展的注释schema与界面；②将权力视为首要注释对象而非仅作为下游代理；③系统地捕捉情绪、关系动态和上下文等多维度特征，为跨文化社会推理提供更细粒度的实验基准。

**🔧 技术方法**

技术手段包括：剧本分段与预处理、LLM（Gemini、GPT、Qwen、Gemma、Molmo）生成对话与场景摘要；手工与LLM联合完成特征标注；使用Gwet's AC2与Jaccard衡量注释一致性；对模型进行零/少样本提示并对输出进行清洗与归一。

**📊 数据集**

使用的数据集为SocLens，涵盖100个电影场景（38个法语、62个埃及阿拉伯语），共15,836条注释实例，注释内容包括人物档案、情绪、权力类型、互动动态、上下文属性等。

**📈 对比分析**

对六个文本LLM和两大多模态LLM与人类标注进行比较，采用Gwet's AC2衡量一致性；模型在可观测属性（性别、关系等）上与人类达高一致性，但在情绪、权力差异、意图对齐等需要深层社会推理的维度上表现显著低于人类；多模态模型覆盖率仅55%且未显著提升。

**⚠️ 局限性**

局限性包括：仅涵盖法语和埃及阿拉伯语两种语言，难以推广到更广泛文化；标注者数量有限，难以充分捕捉解读多样性；模型评测受限于LLM快速迭代和多模态安全拒绝，覆盖率不高；数据集规模相对较小，缺乏足够统计支持。

---

## 49. Great Expectations: Benchmarking the Real-World Performance of RVV 1.0 in HPC

**arXiv ID:** 2608.28097 | [PDF](https://arxiv.org/pdf/2608.28097v1)

**作者:** Stepan Nassyr `[一作]` (Jülich Supercomputing Centre - Forschungszentrum Jülich GmbH), Andreas Herten `[通讯]` (Jülich Supercomputing Centre - Forschungszentrum Jülich GmbH)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过在最新的RVV 1.0硬件（SpacemiT X60/X100/A100、SiFive X280、Sophon SG2044）上执行综合的合成与真实工作负载基准，对比NVIDIA Grace，评估RISC‑V在HPC领域的实际性能；

**💡 创新点**

创新点在于首次对多款RVV 1.0芯片进行系统化、跨平台的基准测试，并结合手工调优的BLIS、FFTW等库，揭示向量化、内存层次和执行端口瓶颈的实证影响；

**🔧 技术方法**

使用了RVV指令集、BLIS、OpenBLAS、FFTW、HPL、HPCG等开源库，以及针对不同芯片的定制微内核与线程划分；

**📊 数据集**

基准数据集包括标准HPCB（BLAS、FFT、HPL、HPCG）和合成STREAM、FMA吞吐测试，涵盖FP64/FP32/FP16精度；

**📈 对比分析**

通过对比FLOP/s、内存带宽、相对峰值效率等指标，发现RVV芯片在向量化层面可达80–90%峰值，但在内存子系统上受限，整体性能仍落后于Grace；

**⚠️ 局限性**

主要局限在于内存墙、执行端口竞争、缺乏高效的分段访问和多级缓存带宽不足，导致即便在最优微内核配置下，RVV平台的计算效率也在30–50%之间，且与ARM64芯片仍存在显著差距。

---

## 50. Learning from Hard Prompts: Difficulty-aware Advantage Amplification in Dynamic Sampling

**arXiv ID:** 2608.27982 | [PDF](https://arxiv.org/pdf/2608.27982v1)

**作者:** Siyuan Gan `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种新的强化学习策略——Difficulty-aware Advantage Amplification Policy Optimization（DA3PO），通过在动态采样下直接放大正确答案的优势来提高训练效率并提升模型在数学推理任务上的准确率。

**💡 创新点**

创新点在于发现动态采样会产生不对称的优势放大（错误答案被放大而正确答案不够），并设计了 Direct Advantage Amplification（DAA）机制，用常数 λ 仅在难度低于阈值 τ 的硬提示下放大正确答案的优势，从而补偿这种不对称并提升效率。

**🔧 技术方法**

技术核心包括：Group Relative Policy Optimization（GRPO）框架、Dynamic Sampling、Clip-Higher、Overlong Reward Shaping、Token-Level Loss 以及新增的 DAA 并将其嵌入 DAPO 得到 DA3PO。实现仅需不到 30 行代码。

**📊 数据集**

使用 DAPO-Math-17K 训练数据，评估数据集为七个数学推理基准：AIME24/25/26、AMC、Minerva、Olympiad 和 MATH。

**📈 对比分析**

与基线（原始模型、GRPO、DAPO、GSPO）比较，DA3PO 在两大模型规模（Qwen3‑4B‑Base、Qwen3‑8B‑Base）下均实现了显著提升，平均准确率提升约 5% 以上，并在最难的 AIME 及所有基准上都取得领先，且通过 pass@k 指标验证并未陷入单一解模式。

**⚠️ 局限性**

局限性在于放大因子 λ 与难度阈值 τ 固定，无法自适应不同提示难度的变化，可能导致在极硬或极易提示下效果不如最佳。未来可探索动态调节 λ 与 τ 的方法。

---

## 51. FocusGen: Expanding Visual Design Exploration with a Simulated Focus Group of Persona Agents

**arXiv ID:** 2608.28001 | [PDF](https://arxiv.org/pdf/2608.28001v1)

**作者:** Jaewon Choi `[一作]` (Hanwha Life), Michael Bernstein `[通讯]` (Stanford University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 FocusGen，一套利用虚拟焦点小组中数百名基于人口统计、背景故事和审美偏好的角色代理，来并行生成多样化视觉设计方案的交互系统。

**💡 创新点**

创新点在于：①将角色代理从传统的单一批判者转变为并行生成者，形成多样化的设计候选；②通过开放式偏好访谈让代理自定义维度，突破设计者预设的“已知未知”；③将人口统计、生成背景和个性化偏好整合成完整的“Persona Profile”，实现跨域、多视角的创意探索。

**🔧 技术方法**

主要技术包括：①基于 Gemini‑2.5‑flash 的 LLM 生成代理人格和偏好；②使用 Imagen‑3.0‑generate 进行文本到图像的多轮迭代生成与改进；③TrueSkill 排序用于在多轮生成中挑选最佳图像；④CLIP 嵌入计算评估视觉多样性；④前端交互界面展示代理配置、生成结果与迭代历史。

**📊 数据集**

数据集与来源：①美国人口普查数据构建 1000 代理的年龄、性别、教育、收入等人口统计属性；②程序化生成的背景故事（约 1000 tokens）；③在技术验证阶段收集的 27 名 Prolific 参与者真实人口统计、故事与偏好；④实验任务涵盖 UI 设计、包装、标志、室内设计、食品等五大类别。

**📈 对比分析**

比较方法：①Agent‑Enabled（角色代理）对比 Agent‑Disabled（通用助手）两种条件，使用 CLIP 距离、CLIP 散度与人类感知评估；②Structured vs. Open‑Ended 访谈方式对比，评估多样性提升；③技术验证中对比迭代生成前后、人工与代理排序的相似度。结果显示：①Agent‑Enabled 在所有任务中平均 CLIP 距离提升 58%，散度提升 33%，且人类更倾向于选择多样性更高的图像；②开放式访谈相较结构化访谈，CLIP 距离提升约 30%，散度提升约 25%；③迭代生成显著提高了用户对图像的偏好，TrueSkill 分数翻倍。

**⚠️ 局限性**

局限与挑战：①代理的审美偏好和输出未经过严格的可辨别性与代表性验证，可能与真实人群偏好不完全吻合；②人机评分相关性弱，说明代理“最佳”判定与人类主观偏好不完全一致；③代理与真实人群的多样性存在差距，尤其在需要高度个性化的任务上；④基于 LLM 的生成可能强化或传播刻板印象；⑤实验样本与代理人口统计主要来自美国，跨文化适用性待验证；⑥迭代过程非单调，收敛判定仍需改进。

---

## 52. A Versioned Unified Graph Index for Dynamic Timestamp-Aware Nearest Neighbor Search

**arXiv ID:** 2608.27663 | [PDF](https://arxiv.org/pdf/2608.27663v1)

**作者:** Jun Woo Chung `[一作]` (Rochester Institute of Technology), Weijie Zhao `[通讯]` (Rochester Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了TiGER——一种统一的、版本化的图结构，用于在动态向量数据集上执行时间感知的近似最近邻搜索；

**💡 创新点**

创新点在于将时间元数据嵌入图的节点与边中，构建单一可持续更新的时序图，同时配合稀疏边数据库实现连续时间区间的快速聚合，彻底消除了传统预/后过滤所需的额外索引或后处理；

**🔧 技术方法**

主要技术包括基于贪心搜索的近邻图构建、持久化数据结构记录节点/边的活跃时间段、二叉堆实现的时间约束遍历，以及稀疏表（类似RMQ）用于高效聚合连续时间段的边；

**📊 数据集**

实验数据集为常用的SIFT‑1M（128维）和GloVe‑100（100维），并在多种人工拆分的时间戳设置下进行评估；

**📈 对比分析**

与传统HNSW的预过滤与后过滤两种基线进行对比，TiGER在各种时间窗口大小下均保持较高召回率的同时，查询吞吐量提升高达5倍；

**⚠️ 局限性**

局限性包括对稀疏边数据库的参数（如分块间隔）需要人工调优；在极大规模或极高频率更新场景下，单线程更新机制与内存开销仍有提升空间。

---

## 53. SEPO: Evidence-Grounded Prompt Optimization via Structural Editing

**arXiv ID:** 2608.28067 | [PDF](https://arxiv.org/pdf/2608.28067v1)

**作者:** Xiaoyu Ma `[一作]` (School of Science and Engineering Chinese University of Hong Kong Shenzhen), Xiaoying Tang `[通讯]` (School of Science and Engineering Chinese University of Hong Kong Shenzhen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于编辑效应谱系反馈的 API‑only 提示优化器（Structural Evidence‑grounded Prompt Optimization），通过两层结构化提示 schema 进行局部可定位的编辑，并记录每次编辑与新修复或破坏示例之间的关系，形成可追溯的优化轨迹。

**💡 创新点**

核心创新在于：① 编辑效应谱系反馈机制，将每一次局部编辑与其对示例的影响联系起来并在搜索分支中持续利用；② 结合可定位的编辑目标、词法层级结构化提示和多轨迹搜索（Lexicase+归档），实现可解释、可审计且高效的提示改进。

**🔧 技术方法**

技术手段包括：两层结构化提示 schema、归因 (attribution) 与补丁 (patch) 架构器（LLM 调用）、编辑效应谱系记录、示例级 admit + Lexicase 选择、聚焦/锚点/训练三槽证据包、以及对齐编辑与示例效应的可视化审计。

**📊 数据集**

使用了 14 任务的保留测试集：BBH（8 个子任务）、MMLU‑Pro（3 个子任务）、代码 MBPP+、推理 GSM‑Hard、HotpotQA（Medium/HQA），并以 Llama‑3.1‑8B‑Instruct 与 Qwen3‑8B 两个冻结工作模型进行评估。

**📈 对比分析**

在匹配 2000 次指标调用预算下，对比六个 API‑only 基线（Seed、APSF、OPRO、MIPROv2、MPO、GEPA）。结果显示在 Llama 上宏观准确率 61.9%（比 GEPA 高 3.1pp），在 Qwen 上 73.3%（比 GEPA 高 2.2pp）。在大多数任务上击败所有基线，并同时位于优化时间与测试时间 Pareto 前沿；优化 token 约 2.9M、部署 token 203k，提示长度比 GEPA 约 5 倍更短。

**⚠️ 局限性**

局限性包括：仅在与现有提示优化相近的提示长度与指令空间内验证；对更长提示或多步/多工具、多模态任务的可扩展性未知；每个任务使用特定的 schema，缺乏跨任务迁移；实验仅覆盖单轮文本任务，未评估更广泛的代理管线或多模态场景。

---

## 54. Thread-Efficient Decoding for Neural Texture Compression

**arXiv ID:** 2608.27888 | [PDF](https://arxiv.org/pdf/2608.27888v1)

**作者:** Janarbek Matai `[一作]` (Advanced Micro Devices, Inc.), Takahiro Harada `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了共享解码器与纹理聚类相结合的神经纹理压缩方法，以降低GPU线程分歧并提升运行时性能；

**💡 创新点**

创新点在于：①共享解码器架构与逐步解码器冻结训练策略；②利用CLIP语义嵌入进行纹理聚类；③在多纹理和真实渲染场景中系统评估线程分歧与性能；

**🔧 技术方法**

技术包括：共享多层感知机（MLP）解码器、渐进式解码器冻结训练、CLIP特征提取与k‑means聚类、HLSL实现与Wave Matrix Multiply-Accumulate（WMMA）优化；

**📊 数据集**

使用了586张来自Polyhaven公开纹理数据集以及三幅真实渲染场景（Yokohama、Bistro、ToyShop）的纹理；

**📈 对比分析**

通过与基线单解码器、单一共享解码器、随机聚类等方案对比，结果显示在10个聚类下线程分歧降低25%–52%，PSNR保持在近似基线水平，Radeon RX 9070 XT GPU上实现了8.48×的速度提升；

**⚠️ 局限性**

局限性在于：聚类算法仅基于语义相似性，缺乏对低层纹理特征的考虑；解码器共享数量与聚类规模的超参数选择仍需系统调优；

---

## 55. CoRe-MoE: Compact Reusable MoE for Continual Multimodal Instruction Tuning

**arXiv ID:** 2608.27867 | [PDF](https://arxiv.org/pdf/2608.27867v1)

**作者:** Runze Liu `[一作]` (Chinese Academy of Sciences), Weiping Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CoRe‑MoE，一种可复用的压缩 MoE 结构，用于持续多模态指令调优；

**💡 创新点**

创新点在于通过 SVD 分解任务 LoRA 更新，发现输入/输出方向子空间高度重叠，进而把这些子空间冻结为可复用基底，只为后续任务学习紧凑的坐标矩阵；

**🔧 技术方法**

采用 LoRA‑MoE、SVD、低秩路由、CLIP 原型引导任务感知路由等技术；

**📊 数据集**

在 UCIT 任务序列上验证，使用 LLaVA‑1.5‑7B 与 Qwen2‑VL‑7B 两大 7B 级别多模态大模型；

**📈 对比分析**

与零样本、逐任务微调、LoRA‑FT、HiDe、MoELoRA、CL‑MoE 等基线对比，CoRe‑MoE 在 LLaVA‑1.5‑7B 上平均提升 3.33 分、最终提升 5.90 分；在 Qwen2‑VL‑7B 上平均提升 2.32 分、最终提升 4.58 分，同时训练参数仅为后续任务 LoRA‑FT 的 1% 以内；

**⚠️ 局限性**

限制在于复用方向基底依赖首个任务的代表性；当首个任务与后续任务差异较大时，冻结基底可能不再适用；任务感知路由对模糊任务边界或开放世界场景的鲁棒性尚待进一步验证。

---

## 56. Predicting LLM Performance from Prompt Linguistic Features: An Empirical Study in Requirements Engineering

**arXiv ID:** 2608.27621 | [PDF](https://arxiv.org/pdf/2608.27621v1)

**作者:** Quim Motger `[一作]` (Universitat Politecnica De Catalunya), Alessio Ferrari `[通讯]` (University College Dublin)

**通讯引用:** 3267 | [OpenAlex ID](https://openalex.org/A5041720518)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在二分类软件需求分类任务中，作者生成了 9,000 个语言属性可控的提示变体，并使用五个开源 LLM 对其进行评估，随后通过回归模型预测提示性能。

**💡 创新点**

创新点在于将可解释的语言学特征（句法关系、形态句法分布等）作为提示质量的先验指标，实现低成本的提示预筛选，并揭示语言结构对 LLM 性能的决定性影响。

**🔧 技术方法**

主要技术包括：利用 Profiling-UD 提取 124 个语言学特征；采用决策树、线性 SVM、随机森林、梯度提升、MLP 等回归模型；使用 10 折交叉验证和置换重要性分析来评估特征贡献。

**📊 数据集**

使用的数据集为 PROMISE‑NFR（625 条需求）与 100 条初始提示，生成 9,000 个变体；评估 LLM 包括 Qwen2‑7B‑Instruct、Falcon3‑7B‑Instruct、Granite‑3.2‑8B‑Instruct、Minstral‑8B‑Instruct‑2410 和 Meta‑Llama‑3‑8B‑Instruct。

**📈 对比分析**

通过 R²、MAE、RMSE 评估回归模型，随机森林平均 R²≈0.41，表明语言学特征能够显著预测 F1、F2、精确率和召回率；特征重要性分析显示句法和形态句法特征最具预测力。

**⚠️ 局限性**

局限性在于实验仅聚焦于单一二分类需求分类任务、单一数据集和单一提示组成部分；提示变体生成依赖单一 Qwen3:32B；未验证所提出方法对其他任务、数据集或更大规模 LLM 的泛化能力。

---

## 57. Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction

**arXiv ID:** 2608.27529 | [PDF](https://arxiv.org/pdf/2608.27529v1)

**作者:** Jiarong Han `[一作]` (Alibaba Group), Ming Qian `[通讯]` (Alibaba Group)

**通讯引用:** 6115 | [OpenAlex ID](https://openalex.org/A5083531923)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种仅使用固定12帧本地时间窗口的流式三维重建模型，预测每帧的点云和相邻帧的相对姿态，然后通过链式组合得到全局轨迹与全局几何。

**💡 创新点**

创新点在于：①把所有回归目标保持在局部参考框架内，使得模型不需要存储或传播长范围记忆；②引入轻量级运动‑视觉旋转细化器提升相邻帧旋转估计；③设计基于多步姿态组合的监督损失，直接约束递归组合的误差，显著减少长时间漂移。

**🔧 技术方法**

技术实现包括：基于因果Transformer的窗口注意力架构；点云与相机令牌的交叉注意力；姿态描述子和旋转细化网络（TCN+MLP）；多步组合损失与旋转细化正则化；可选的训练无关循环闭环后端。

**📊 数据集**

训练使用30个混合数据集（Synthetic如TartanAir、VirtualKITTI2、OMNIWorld等；Real如Waymo、ARKitScenes、ScanNet等）覆盖室内外、手持与车载、动态与静态场景；测试在KITTI、Oxford Spires、VBR、7Scenes、TUM-Dynamic等公开基准上评估。

**📈 对比分析**

与多种流式重建方法（LingBot-Map、HorizonStream、LongStream、InfiniteVGGT、OVGGT等）及传统SLAM方法（VGGT-SLAM、MASt3R-SLAM等）对比，本文模型在KITTI、Oxford Spires、VBR等长序列上实现了最低的ATE（分别为4.02m、4.35m、16.7m），相对位姿误差和实时帧率也优于大多数基线；在密集重建任务中，CD/F1指标也处于同类方法前列。

**⚠️ 局限性**

局限性包括：在空间受限、视觉重叠频繁的室内场景（如7Scenes、TUM-Dynamic）中，缺乏持久全局记忆可能导致与某些基线相比略逊；对动态物体的处理仍不够鲁棒；模型仅在RGB流上无内参需求，若使用深度或IMU信息可能进一步提升。

---

## 58. Fine-Grained Complexity of Approximating Vector Knapsack: A Faster Algorithm and Bicriteria Optimality in 2D

**arXiv ID:** 2608.27600 | [PDF](https://arxiv.org/pdf/2608.27600v1)

**作者:** Karl Bringmann `[一作]` (ETH Zurich), Karol Węgrzycki `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

重新审视d维向量背包问题（d-Knapsack），提出了一种新的近似算法，旨在选择一组物品以最大化总利润，同时不超过每个维度的容量限制。

**💡 创新点**

首次提出了针对d-Knapsack的meet-in-the-middle算法，显著提高了运行时间，且是25年来的首次改进，首次通过常数因子改善了指数。

**🔧 技术方法**

采用了动态规划算法替代之前算法中的线性规划求解器，并结合了meet-in-the-middle策略。

**📊 数据集**

使用了多种数据集，包括d维容量向量和一组具有d维重量向量和利润的物品。

**📈 对比分析**

与之前的算法相比，新的算法在运行时间上有显著提升，特别是对于2-Knapsack问题，达到了近线性时间的(2/3 - o(1))近似，性能接近理论下界。

**⚠️ 局限性**

算法的局限性在于对于d维大于2的情况，仍然存在未解决的最优运行时间指数问题，且在某些情况下可能无法达到最优解。

---

## 59. Characterizing the I/O Behavior of HPC Applications through Modeling and Simulation

**arXiv ID:** 2608.27642 | [PDF](https://arxiv.org/pdf/2608.27642v1)

**作者:** Njoud O. Almaaitah `[一作]` (Mutah University), Raffaele Montella `[通讯]` (University of Naples Parthenope)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一套框架，将HPCIO仓库的真实I/O轨迹转化为ElastiSim模拟环境的应用模型，并通过一系列案例（Quantum Espresso、Nek5000、WaComM++、Jacobi、EpiGraph）验证该框架在重现和分析HPC I/O行为上的可行性。

**💡 创新点**

创新点在于①将HPCIO I/O trace 与 ElastiSim 无缝集成，提供从轨迹提取到模型校准的完整流程；②通过自适应采样与计算负载调节实现与原始轨迹高度一致；③可在单一参考平台上并行比较多种应用的I/O行为；④利用模拟评估文件系统的I/O 干扰与瓶颈。

**🔧 技术方法**

使用技术包括 ElastiSim（基于 SimGrid 的离散事件模拟器）、Darshan I/O trace、HPCIO分析数据库、采样与校准算法、以及针对 CPU、网络与 PFS 的硬件性能模型。

**📊 数据集**

数据集主要来自 HPCIO 仓库（Quantum Espresso QePh/QeC、Nek5000、WaComM++ 的 I/O 轨迹）以及从 HPC4AI 集群获取的 Jacobi 和 EpiGraph 的 Darshan 轨迹。

**📈 对比分析**

通过在不同真实/模拟平台上并行运行同一应用，比较执行时间、I/O 阶段时序与吞吐量，并利用误差阈值对模型进行校准。实验结果显示仿真与真实的匹配度可达 98.8%（Jacobi）和 98.1%（EpiGraph）；在多作业干扰实验中进一步验证了 I/O 吞吐量随作业数量和数据分布的下降趋势。

**⚠️ 局限性**

局限性包括：未考虑通信开销；仅针对 CPU 与 I/O 的建模；缺乏 burst buffer 与更细粒度的性能模型；对不同系统的跨平台映射仅采用简单比例关系，可能不适用于极端异构环境；实验覆盖的工作负载范围有限，未评估动态作业伸缩与真实网络拓扑对 I/O 行为的影响。

---

## 60. GeoFF3D: Coordinate-Anchored Feed-Forward Reconstruction for Large-Scale UAV Mapping

**arXiv ID:** 2608.28288 | [PDF](https://arxiv.org/pdf/2608.28288v1)

**作者:** Xiang Yang `[一作]` (Central South University), Yunsheng Zhang `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出 GeoFF3D，一种在地理坐标系下直接预测相机位姿和稠密点云的前向网络，并配合空间大规模重建框架（SLRF）实现对千幅 UAV 图像的高效、可扩展重建。

**💡 创新点**

采用坐标锚定的前向模型，在 Z‑up 度量帧中直接利用地理位移和可选几何先验进行预测；SLRF 通过基于覆盖度的空间分块、中心向外推理与重力保持的层级对齐，有效解决多块之间的尺度、姿态漂移与边界缝隙。

**🔧 技术方法**

使用 Transformer‑style 坐标锚定预测头、可选几何先验编码、基于脚印的空间分块、中心向外的先验传播、重力保持的 GA‑Sim 与 GA‑Rigid 对齐，以及自监督损失（局部、全局、姿态、重力）。

**📊 数据集**

利用 UAVFF3D‑Real、UseGeo、NPU‑DroneMap 以及 UAVScenes 等四个 UAV 影像集，共计数千幅图像。

**📈 对比分析**

与 VGGT+SLRF、Pi3X+SLRF 及多种 SLAM/流式基线在九个航空区块和长序列上对比，GeoFF3D 在 F@5、F@1 以及整体精度上均显著优于对照组，长序列上 F@5 提升至 0.848，完成 2000 张图像约 5 分钟。

**⚠️ 局限性**

依赖较为准确的地理位移与重力先验，对噪声敏感；对极端稀疏或极端误差的先验鲁棒性待提升；目前未实现全局优化，仅靠层级对齐，可能在极大尺度下出现残差。

---

## 61. H-Scale: Hessian-Guided Scale Refinement for NVFP4 Sub-Byte LLM Inference

**arXiv ID:** 2608.28113 | [PDF](https://arxiv.org/pdf/2608.28113v1)

**作者:** Hao Yu `[一作]` (Alibaba Inc.), Jianwei Zhang `[通讯]` (Alibaba Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对NVIDIA Blackwell NVFP4微粒化量化格式的后处理方法H‑Scale，用于优化每个16维组的缩放因子

**💡 创新点**

创新点在于把组尺度选取转化为基于对角Hessian加权的层输出重构目标，而非传统的权重重构误差；同时在有限的硬件可表示尺度空间内枚举邻域尺度，零推理开销

**🔧 技术方法**

采用对角Hessian代理、对激活校准的二阶信息、FP4/FP8量化函数、组内尺度枚举搜索及重量化算法

**📊 数据集**

使用了4096条FineWeb校准样本（seq_len 8192）以及AIME、MMLU‑Redux、LiveBench、LiveCodeBench、C‑Eval、ARC‑Challenge、BBH、GPQA、GSM8K等多任务评测集

**📈 对比分析**

在Qwen3‑30A3‑Thinking、Qwen3‑30A3‑Instruct、Qwen3‑4B‑Instruct以及LLaMA‑3.1‑8B‑Instruct等模型上与RTN、4over6、ArcQuant、GPTQ、GPTAQ、MR‑GPTQ等现有NVFP4 PTQ流水线对比，H‑Scale平均提升了数个百分点，显著缩小与BF16基准的性能差距，并保持零推理开销

**⚠️ 局限性**

仅在g=16的NVFP4文本模型上验证；未覆盖其他位宽/组大小、非文本或多模态模型，且对极端稀疏或高动态范围情况的鲁棒性未深入探究

---

## 62. Synthetic Linguistic Agency: How an Embodied Mortal Agent Learns Linguistic Affordances through Consequential Social Experience

**arXiv ID:** 2608.27843 | [PDF](https://arxiv.org/pdf/2608.27843v1)

**作者:** Sixin Chen `[一作]` (Shantou University), Taizhou Chen `[通讯]` (Shantou University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实证检验了合成语言主体（Synthetic Linguistic Agency, SLA）的概念，首先将语言主体学说中的体现（embodiment）、参与（participation）与脆弱性（precariousness）转化为可检验的操作性条件，并将这些条件应用于现有系统进行分类；随后基于“死亡驱动的语言强化学习”（Mortality‑Grounded Linguistic Reinforcement Learning, MGL‑RL）构造了一个具备持续身体、可学习语言行为与后果关联的“具身可死亡代理”（Embodied Mortal Agent, EMA），通过对身体状态、历史、参与、脆弱性四个环节的因果干预，证明其在单一持续生命线中实现了SLA。

**💡 创新点**

创新点包括：1）首次将语言主体学说中的三大条件系统化为可操作的合成语言主体定义；2）设计了MGL‑RL框架，将死亡驱动与语言行为与身体后果的关系结合起来，形成可学习的语言能力与生存决策耦合；3）通过可观测的干预实验验证身体、语言历史、双方参与与脆弱性在同一生命线中的独立与耦合因果作用。

**🔧 技术方法**

技术主要包括：基于Homeostatically Regulated Reinforcement Learning (HRRL) 的半模型化框架；引入Reference‑RUL来量化健康衰减与生存价值；使用Beta 分布和核权重的贝叶斯后验更新实现语言行为的适配；采用Thompson采样进行行动选择；以及通过可追踪的生命记录和可复制的实验流程实现因果干预。

**📊 数据集**

数据集主要为自制的语言请求集合（100条含POLITE/TRANSACTIONAL对，60条用于社会形成，20条用于验证，20条留作holdout），以及四个固定伙伴的行为策略。身体动力学采用单维健康衰减模型，无外部公开数据集。

**📈 对比分析**

对比方法包括：Natural（根据学习得到的社交后验进行决策）、Pooled（将所有伙伴信息聚合后决策）、Fixed‑Action（固定四个动作之一）以及Reference‑RUL Oracle（在已知社会后验下选择最大期望RUL）。结果显示，Natural策略在平均生命周期上比Pooled提升约1.4–1.6回合（约5%），并显著高于固定动作；在语言映射与参与度评估中，干预实验表明身体状态、历史与参与都能显著改变策略分布，且各因素在不同干预下产生的效应在95%区间内显著非零。

**⚠️ 局限性**

局限性包括：1）实验仅在单一固定脚本生态（四个动作、四个伙伴、1D健康模型）中进行；2）身体模型过于简化，未考虑感知误差、可恢复性或多维度状态；3）缺乏真实人类交互验证，所有伙伴均为模拟策略；4）干预只覆盖部分因果路径，未考察更长时程或更复杂的语言行为；5）未探索不同参数、伙伴生态或学习架构的稳健性。

---

## 63. When Muon Meets Task Interference: A Spectral Perspective on Continual Learning and Model Merging

**arXiv ID:** 2608.27518 | [PDF](https://arxiv.org/pdf/2608.27518v1)

**作者:** Shangge Liu `[一作]` (Nanjing University), Wenbin Li `[通讯]` (Nanjing University)

**通讯引用:** 7667 | [OpenAlex ID](https://openalex.org/A5100462994)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

统一灾难性遗忘与模型融合两大范式下的任务干扰，证明两者可归结为同一层级 Frobenius 内积形式，并展示 Muon 优化器通过谱范数控制可同时缓解两类干扰。

**💡 创新点**

创新点在于将 CL 与 MM 的干扰统一为⟨ΔW_ℓ,J_ℓ(x)⟩_F，推导出谱范数上界，并揭示 Muon 为该上界的闭式谱范数梯度下降解，从而实现单一优化器对两种问题的共同解决。

**🔧 技术方法**

采用 NTK 线性化分析、谱范数上界推导、Per‑mode 贡献分解、Muon's 极化分解实现等技术手段。

**📊 数据集**

使用 CLIP 预训练的 ViT-B/32/16/L/14 在八任务模型融合基准上，以及 CIL、TIL、MTIL 等持续学习基准（CIFAR‑100、TinyImageNet、ImageNet‑R、11‑task MTIL）。

**📈 对比分析**

与 AdamW、非线性微调、TTA、OrthoReg 等基线在同一训练设置下对比，Muons 在模型融合上绝对精度提升 3–10%，在持续学习中平均提升 2–4%；表现优于现有方法且可单独归因于优化器。

**⚠️ 局限性**

仅在 NTK 线性化假设下给出理论证明，对更大规模模型或更长任务序列的普适性需进一步验证；实现对 Muon 的梯度矩阵近似也可能引入数值误差。

---

## 64. DensityKV: Density-Guided KV Cache Compression for Long Video Generation

**arXiv ID:** 2608.27922 | [PDF](https://arxiv.org/pdf/2608.27922v1)

**作者:** Wenqu Zhao `[一作]` (Manifold AI), Wei Wu `[通讯]` (Manifold AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了DensityKV，一种无训练的历史KV存储管理策略，用于提升自回归视频扩散模型的长期一致性；

**💡 创新点**

通过在每个注意力头上维护独立的token级KV银行，并利用Soft‑Riesz密度评估post‑RoPE键空间的冗余，从而在保持完整KV对的前提下控制历史记录的拥挤；

**🔧 技术方法**

关键技术包括post‑RoPE键空间几何分析、Soft‑Riesz密度计算、在线密度约束更新、按头和token级别的选择与驱逐策略；

**📊 数据集**

在MovieGenBench（128个提示）上进行评估，并使用与LongLive‑RAG相同的长视频生成协议；

**📈 对比分析**

将DensityKV与Native、∞‑RoPE、Deep‑Forcing、LongLive‑RAG等方法按同一历史KV容量进行对比，在三种生成器上实现了31/54个单指标胜利，平均排名最优（1.89），显著提升背景一致性和稳定性；

**⚠️ 局限性**

限制包括对单个生成器架构的适配性有限，仍需要更高容量下的实验验证；此外，方法在极端高运动场景下可能略逊于某些基于检索的方案，且对密度阈值和参数选择较为敏感。

---

## 65. Cyc3D: Evaluating Cyclic Structural Stability and Asset Usability in Image-to-3D Generation

**arXiv ID:** 2608.28080 | [PDF](https://arxiv.org/pdf/2608.28080v1)

**作者:** Liwen Zhang `[一作]` `[通讯]`, Liwen Zhang

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Cyc3D多维度基准，用于评估图像到3D生成的跨视角一致性和表示质量。

**💡 创新点**

创新在于引入视图循环结构一致性、资产级语义一致性、几何、UV质量及参考图像保真度的综合评估。

**🔧 技术方法**

采用闭环渲染‑再生成‑对齐流程、DINOv2特征一致性、SSIM/LPIPS、VLM语义评估、几何重叠与UV占用度等技术。

**📊 数据集**

使用包含95个对象的Cyc3D数据集，覆盖家具、车辆等多种类别并附带真实网格与多视图渲染。

**📈 对比分析**

与五种代表方法（Hunyuan3D、Tripo3D、Stable Zero123、Magic123、DreamCraft3D）对比，闭源系统在几何和UV方面得分最高，但循环稳定性仍低于48。

**⚠️ 局限性**

限制在于仍无法保证生成的3D先验稳定，闭环循环漂移明显，且评估对生成器的训练目标与分布偏移影响不明。

---

## 66. Depth-Aware Pothole Detection Using YOLO and RT-DETR at the Edge

**arXiv ID:** 2608.27633 | [PDF](https://arxiv.org/pdf/2608.27633v1)

**作者:** Md Monjurul Ahsan Prodhan `[一作]` (University at Albany, State University of New York), Md Nour Hossain `[通讯]` (University at Albany, State University of New York)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出并评估了基于RGB‑D传感器的深度感知坑洞检测框架，并对五种目标检测架构进行比较。

**💡 创新点**

创新点在于：①引入RANSAC地面正射校正实现真实垂直深度测量；②设计六种针对道路恶劣环境的Albumentations增强管线；③量化并揭示边界框模型在深度测量中固有的结构性误差；④系统对CNN与实时Transformer架构进行对比。

**🔧 技术方法**

使用YOLOv8n、YOLOv8n‑Seg、YOLOv9t、RT‑DETR‑L、RT‑DETR‑X等模型，结合Albumentations增强、RANSAC正射、像素精确分割与基于百分位数的深度提取算法。

**📊 数据集**

使用PothRGBD数据集（1000对640×480 RGB‑Depth图像）进行训练（80/20划分）和验证。

**📈 对比分析**

比较方法为在同一验证集上计算Precision、Recall、mAP@50、mAP@50‑95，并测算推理时间与参数量。YOLOv8n‑Seg在mAP@50和mAP@50‑95上领先，且深度误差最小（2.96 cm）；YOLOv8n最快（3.6 ms）；RT‑DETR‑X最高置信度（92.70%）。

**⚠️ 局限性**

局限性包括：仅针对单一坑洞类别，数据量有限；推理速度在Transformer模型较慢；深度提取仍受像素噪声与相机标定影响，需在更大多样化数据集上验证。

---

## 67. Ex-Sim(3)-Reg: 2D-3D Correspondence Pruning via Extended Sim(3) Registration

**arXiv ID:** 2608.28096 | [PDF](https://arxiv.org/pdf/2608.28096v1)

**作者:** Pei An `[一作]` (Huazhong University of Science and Technology), Liangliang Nan `[通讯]` (Delft University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种名为Ex‑Sim(3)-Reg的快速鲁棒二维三维对应关系裁剪算法，将基于深度先验的裁剪问题重新表述为扩展的Sim(3)配准问题，并通过扩展兼容图和最大团搜索实现对深度噪声的补偿；

**💡 创新点**

创新点在于：①首次将深度噪声建模为对应关系特定的尺度扰动，从而将传统Sim(3)配准扩展为扩展Sim(3)；②设计了扩展兼容图族，利用最大团搜索和尺度校正函数将扩展Sim(3)简化为标准SE(3)配准，理论上解释了算法有效性；

**🔧 技术方法**

使用技术包括：扩展Sim(3)配准框架、深度先验预测（Monocular depth），兼容图构造、最大团搜索（FastMAC/MAC）、尺度校正函数、RANSAC‑P3P、SE(3)配准子算法以及自适应投票估计初始尺度；

**📊 数据集**

实验数据集涵盖4个公开室内/室外数据集：7‑Scenes、RGB‑D‑V2、ScanNet 及 TUM，且在受扰、未见、低内点、跨域等多种挑战场景下进行评估；

**📈 对比分析**

与传统基线及现有Sim(3)裁剪方法（FastMAC、SC2-PCR、TurboReg等）对比，Ex‑Sim(3)-Reg 在内点比例上提升约 20–30%，在注册召回率上提升 13–18%，并在跨域、低内点场景中仍保持明显优势；

**⚠️ 局限性**

局限性在于：扩展Sim(3)配准本质上是欠定问题，算法只能恢复扩展兼容图族覆盖的内点子集，即使 K→∞ 也无法覆盖全部真实内点，需要进一步改进图构造与搜索策略以挖掘更多潜在内点。

---

## 68. Temporal Memory-Aware Online Test-Time Adaptation on Dynamic Graphs

**arXiv ID:** 2608.27948 | [PDF](https://arxiv.org/pdf/2608.27948v1)

**作者:** Bo Li `[一作]` (Griffith University), Shirui Pan `[通讯]` (Griffith University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种面向动态图的时序记忆感知在线测试时适应框架 DGOTTA，能够在不使用标签的情况下对训练好的动态图神经网络进行实时自适应。

**💡 创新点**

创新点包括：①时序感知的数据增强，模拟节点特征衰减与时间窗口边缘扰动；②基于记忆池的软伪标签和漂移感知的指数移动平均来缓解灾难性遗忘；③一致性引导的在线适配，结合预测一致性与时间平滑损失实现稳定的自适应。

**🔧 技术方法**

技术手段：时序特征衰减、时间窗口边缘采样、记忆池加权伪标签、漂移感知 EMA、KL 一致性损失、时间平滑损失；框架可与四种动态图模型（GraphMixer、DyGFormer、TGN、TGAT）兼容。

**📊 数据集**

使用了三大公开动态图数据集：Wikipedia、Reddit、MOOC。

**📈 对比分析**

与基线方法（ERM、TENT、GTrans、SOGA、MATCHA、DDGCL、IDOL）以及不同动态图模型进行对比。DGOTTA 在三大数据集和四种骨干网络上均取得最高 AUC，提升幅度可达 5-10% 以上，且保持较好的运行效率。

**⚠️ 局限性**

局限性：①仅在无标签的在线环境下测试，无法评估有监督辅助；②对极端结构剧烈变化时的记忆池衰减与 EMA 参数调优敏感；③目前仅针对节点分类任务，其他任务（链路预测、图分类）尚未验证。

---

## 69. Accelerating LLM Inference via Vector Index Based Output Embeddings

**arXiv ID:** 2608.27460 | [PDF](https://arxiv.org/pdf/2608.27460v1)

**作者:** Martin Loretz `[一作]` (NXAI), Sepp Hochreiter `[通讯]` (NXAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将大型语言模型的密集输出投影替换为基于向量索引的近似最大内积搜索（MIPS），从而在单批推理中显著降低内存带宽占用并提升吞吐量。

**💡 创新点**

创新点在于把 token 采样问题重新表述为 MIPS，并利用 HNSW 结构实现稀疏高效的输出头，兼容现有模型且无需预训练改动。

**🔧 技术方法**

使用 HNSW 近似最近邻搜索、向量索引、内部点积、top‑k 采样；实现基于 CPU 的推理，并与常规稠密矩阵乘法进行对比。

**📊 数据集**

在 Gemma 3、Llama 3.2、Qwen 3 这些公开模型上进行实验，使用 Wikipedia 文章提取隐藏状态、AlpacaEval 评估生成质量。

**📈 对比分析**

与稠密投影相比，输出层加速可达 12×，整体推理加速最高可达 82%（batch 1）；Recall@2 超 99%；生成质量与基线相当，AlpacaEval win‑rate 约 48–49%。

**⚠️ 局限性**

主要局限是仅针对 CPU，GPU 上因图遍历的顺序性难以加速；在大批量或高度量化的稠密基线下收益降低。

---

## 70. Memorization Is Not Extraction: Tight Differential-Privacy Bounds and Audit Blind Spots

**arXiv ID:** 2608.27782 | [PDF](https://arxiv.org/pdf/2608.27782v1)

**作者:** Xujun Che `[一作]` (University of North Carolina at Charlotte), Shuhan Yuan `[通讯]` (Utah State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将记忆化度量与差分隐私（DP）联系起来，证明了DP在两种重要记忆化衡量上具有精确的上界，并揭示这两种衡量在实践中互不控制；同时构造了两种可攻击模型，证明了传统的基于损失的审计在某些情况下会产生盲点；

**💡 创新点**

创新点在于：①给出DP对计数器假设记忆化和可提取性（adaptive extraction）两种实用衡量的闭式上界；②证明这两种衡量在本地得分类上互相不可比；③设计并实验验证了“保留触发器通道”和“门控发布”两种攻击构造，展示了审计盲点；

**🔧 技术方法**

核心技术包括：f‑DP与优势函数的严格桥接、总变差参数的闭式表达、可提取性上界的适配抽样策略、几何噪声计数与群组隐私极值求解、构造化的树状伪造样本与触发器注入；

**📊 数据集**

实验使用公开的WikiText‑2语料、Zipf 分布模拟数据、以及大规模开源 LLM（Pythia‑1.4B、Qwen2.5‑1.5B）进行 fine‑tuning 与 DP‑SGD 训练；

**📈 对比分析**

通过对比 DP‑SGD 与非 DP 训练在可提取率、记忆化得分（teacher‑forcing、log‑loss）以及成员推断指标（AUC、zlib、min‑k%）的差异，证明 DP‑SGD 在理论上可实现的上界得到实验支持，而传统审计在触发器/隐藏通道下失效；

**⚠️ 局限性**

局限性包括：①理论结果主要针对加/去邻居 DP，缺少对替换邻居的精确阈值；②分离构造是人工设计的，未证明自然训练是否会自然产生相同几何；③高概率版本的提取/记忆化上界需要额外的概率分析；④实验规模仍受 GPU 资源限制，未覆盖所有 LLM 体系。

---

## 71. Visual Token Coding for Video Multimodal Large Language Models

**arXiv ID:** 2608.28008 | [PDF](https://arxiv.org/pdf/2608.28008v1)

**作者:** Chenxin Fang `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于视频编码原理的视觉令牌压缩框架VTC，并在其基础上加入动态分辨率、动态预算和空间覆盖设计，形成VTC_Dy，显著提升视频大模型的令牌压缩效率。

**💡 创新点**

创新点在于：① 将传统HEVC的I/P帧预测和残差编码迁移到视觉特征空间，构建语义I/P帧；② 通过DyRSO实现多分辨率GOP，DyTA根据码率复杂度动态分配令牌预算；③ 引入SC-TopK保持空间覆盖，避免令牌聚集。

**🔧 技术方法**

采用的技术包括：视觉特征提取、语义残差能量计算、HEVC码率预算、动态分辨率采样、整数分配算子、Top-K选择以及多分辨率组图（GOP）结构。

**📊 数据集**

实验使用的长视频基准包括MLVU、LongVideoBench、LVBench和Video-MME，评估多模态大模型（Qwen3-VL、LLaVA-OneVision、LLaVA-OneVision-2）。

**📈 对比分析**

与FastV、VisionZip、HoliTom、FlashVID等SOTA令牌压缩方法对比，VTC_Dy在50%令牌保留率下平均保持100.1%性能，在25%保留率下保持97.8%，在多项任务上超过对手，且模型推理延迟显著下降。

**⚠️ 局限性**

主要局限性在于：① 仍基于手工设计的规则和HEVC码率统计，可能对不同视频内容的泛化能力有限；② 需要预先提取视觉特征，未涉及端到端训练；③ 对极端分辨率或帧率变化的适应性尚未深入验证。

---

## 72. FFSlim: An Efficient and Lightweight Format for Multi-modal Data Storage and Retrieval

**arXiv ID:** 2608.27865 | [PDF](https://arxiv.org/pdf/2608.27865v1)

**作者:** Long Yang `[一作]`, Liang Shi `[通讯]`

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了FFSlim，一种轻量化、高效的多模态数据存储格式，解决了传统格式在一对多结构下的存储冗余、缓存无效、索引占用大等问题。

**💡 创新点**

创新点在于：①将数据以Media2Texts单元组织，直接去除媒体重复存储；②采用统一单文件布局，消除小文件开销；③引入自适应索引（基于数据集类型的哈希/AVLplus），显著降低索引内存占用；④通过完整读取Media2Texts对象实现高缓存命中率。

**🔧 技术方法**

核心技术包括：统一轻量级文件格式（ULHFF）、自适应样本检索（ASR）、基于AVLplus树的自适应索引、Python/PyTorch接口封装以及一键格式转换工具。

**📊 数据集**

使用多种典型多模态数据集进行评测，包括图像-文本（Flickr30k、MSCOCO、VQA2、GQA）、视频-文本（MSRVTT）和音频-文本（Clotho）等七个数据集。

**📈 对比分析**

与Files、TDP、FFRecord三种主流格式进行对比；在随机采样下，FFSlim平均实现数据加载吞吐量提升2.07倍、写入吞吐量提升8.26倍；存储占用平均减少2.09%；索引内存占用仅为数据集大小的0.0043%。

**⚠️ 局限性**

局限性包括：目前仅利用系统默认页面缓存，未针对多模态训练设计专用缓存策略；适用性受限于媒体尺寸较大且文本较多的典型一对多场景；未针对极端1:1或高文本冗余情况进行深入评估。

---

## 73. TACIT-Switch: Cost-Aware Model Escalation for LLM Agents from Censored Supervision

**arXiv ID:** 2608.27911 | [PDF](https://arxiv.org/pdf/2608.27911v1)

**作者:** Ji'an Lei `[一作]` (Beijing Normal University), Jian Huang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种从较小模型（Cheap）到较大模型（Strong）的永久交接策略，利用离线配对试验结果和教师标注的粗粒度交接窗口来训练基于混合死亡率阈值的决策模型。

**💡 创新点**

创新点在于：
- 将“是否需要强模型成功”和“何时交接”的不确定性拆分为混合死亡率模型（mixture‑cure）和AFT阈值分布；
- 采用区间删失（interval‑censored）教师注释，只需粗略标记交接窗口即可；
- 通过累计风险尺度（基于诊断向量权重）实现对任务进度的实时评估；
- 训练完成后部署时不再需要教师或额外标签，完全基于线上风险估计。

**🔧 技术方法**

技术方法包括：
- 任务特征与累计风险的线性组合得到风险得分；
- 采用混合死亡率模型（mixture‑cure）结合AFT（加速失效时间）对阈值进行建模；
- 采用区间删失似然进行联合最大化，带L2正则；
- 对阈值超参数α进行开发集成本‑成功折衷选择；
- 在多步仿真与真实交互环境中评估。

**📊 数据集**

主要使用的数据集：
- ALFWorld（文本交互式家务任务）与其未见子集；
- DABench（可执行工具环境的数值分析任务）；
- 机制化仿真环境用于模型验证。

**📈 对比分析**

与基线比较：Task Router（任务级模型选择）、Step Deferral（ReDAct式步骤级递延）、SWE‑Router（固定前缀递归）以及纯 Cheap/Strong。实验显示：
- 在 ALFWorld，使用 4B Cheap 时成功率从 45.5% 提升到 48.5%，9B Cheap 时从 33.6% 提升到 45.5%；
- 在 DABench，成功率从 67.2% 提升到 73.1%，平均成本也最低；
- 在机制化仿真中，永久交接策略相较于其他策略提升 7–11% 的成功率，成本相当。

**⚠️ 局限性**

局限性：
- 需要配对的 Cheap/Strong 轨迹和教师标注的交接窗口，若分布漂移或标注偏差会影响性能；
- 只能执行一次永久交接，无法支持可逆或多次切换；
- 成本阈值 α 需在开发集上手工选择，可能不适用于所有应用；
- 模型假设累积风险为非负线性组合，若诊断信息不足可能导致估计失真。

---

## 74. Synth-JDoc: Synthesizing a Japanese Document Image Dataset for OCR with Diverse Layouts and Embedded Images

**arXiv ID:** 2608.28248 | [PDF](https://arxiv.org/pdf/2608.28248v1)

**作者:** Keito Sasagawa `[一作]` (Waseda University), Daisuke Kawahara `[通讯]` (Waseda University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建 Synth-JDoc 合成日文文档图像数据集，并使用该数据集对大型视觉语言模型进行全参数微调，以提升其对竖排日文文本的 OCR 能力。

**💡 创新点**

创新点在于利用 HTML/CSS 生成多列、多方向（横排与竖排）布局的文档，同时嵌入文本生成的图片与标题，并加入扫描/ Augraphy 噪声，实现更逼真、结构更丰富的合成文档，解决传统合成方法缺少竖排和图像嵌入的问题。

**🔧 技术方法**

技术手段包括：LLM 生成标题与图像提示、文本到图像模型（Z-Image-Turbo）生成插图、LVLM 生成图像说明、HTML/CSS 渲染、Augraphy 噪声处理；随后在 Qwen、InternVL、Gemma 等 LVLM 上进行全参数微调。

**📊 数据集**

使用 JSSODa 文本生成文档内容，评估时采用 VJRODa 竖排 OCR 测试集；基准对比包括原始 JSSODa 合成集和使用 Nano Banana Pro 合成的集。

**📈 对比分析**

通过 CER（字符错误率）和 BLEU（字符级 BLEU）两种解码设置（Raw Output 与 Remove Repetition）进行对比，实验表明在 VJRODa 上，微调后模型在 CER 和 BLEU 上显著优于基准，除 Gemma 外性能提升尤为突出。

**⚠️ 局限性**

局限性包括：未覆盖更复杂排版（如报纸）和图表表格等元素；解码时仍出现重复输出，评估方法未完全消除这一影响；噪声对测试集的效用尚未验证；未对 NSFW 内容进行过滤；Gemma 在处理高分辨率文档时受限。

---

## 75. VidParse: Online Parsing of Egocentric Procedures Like a Pro

**arXiv ID:** 2608.27562 | [PDF](https://arxiv.org/pdf/2608.27562v1)

**作者:** Anubhav Gupta `[一作]` (University of Maryland), Abhinav Shrivastava `[通讯]` (University of Maryland)

**通讯引用:** 8137 | [OpenAlex ID](https://openalex.org/A5101614443)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 VidParse，一种在线、训练无关的第一人称视角程序解析框架；

**💡 创新点**

创新点在于将程序解析视为结构化推理，通过冻结的 DINOv2 与手-物体检测的 MAF 特征构建无监督边界检测，并利用图约束 Beam Search 与 gap rectification 强化结构一致性；

**🔧 技术方法**

使用技术包括冻结的 DINOv2 backbone、手-物体检测、Manipulation‑Anchored Features、Temporal Similarity Matrix + Gaussian‑tapered checkerboard kernel、微型原型匹配、图约束 Beam Search 与 N‑step 评估；

**📊 数据集**

使用的数据集为 GTEA 与 EgoPER 两个 egocentric 程序解析基准；

**📈 对比分析**

与在线基线（如 ProTAS、MSTCN）对比，帧级准确率、Edit 与 F1 指标显著提升，尤其 N‑step transition 精度提升 5–10 倍，整体性能可与 SOTA 并列或超越；

**⚠️ 局限性**

局限性包括依赖手‑物体检测，遮挡或运动模糊时易失效；段级处理导致延迟不够低；任务图对未见变体泛化差，且对非交互动作表现欠佳。

---

## 76. ROPE: Routed Origin Policy Enforcement against Indirect Prompt Injection

**arXiv ID:** 2608.27496 | [PDF](https://arxiv.org/pdf/2608.27496v1)

**作者:** Xinhang Ma `[一作]` (Washington University in St. Louis), Yevgeniy Vorobeychik `[通讯]` (Washington University in St. Louis)

**通讯引用:** 5327 | [OpenAlex ID](https://openalex.org/A5038669899)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于源追溯的ROPE防御，针对间接提示注入(IIP)在工具使用型LLM代理中的攻击，确定可信源并对敏感参数做确定性检查。

**💡 创新点**

通过将可信源定义为用户请求、用户指定的可验证身份和平台记录，构建可任务定制、无模型循环的源政策，避免传统信息流防御的过度限制。

**🔧 技术方法**

使用结构化源追踪器、确定性源检查器、单次可信请求路由器，并在工具架构中植入审核表，确保Origin Guard。

**📊 数据集**

在AgentDyn和AgentDojo等公开评测基准（如GitHub、shopping、daily-life、banking、slack、travel）上进行实验，并对适配的攻击任务进行自定义注入。

**📈 对比分析**

与11种主流防御（prompt sandwich、filter、Tool Filter、CaMeL、DRIFT、PFI等）进行对比，ROPE在四个代理模型上实现1.6–2.6%的攻击成功率，同时保留82–100%的未防御任务完成率，优于传统系统级防御。

**⚠️ 局限性**

无法防止基于自由文本消息的攻击、仅受保护的工具参数以及未被源检测的注入内容导致的误报/误拦；同时依赖平台提供完整的源标记，若缺失则失效。

---

## 77. CEDAR: Automata as Verifiable Interfaces for Language-Guided Embodied Action

**arXiv ID:** 2608.27797 | [PDF](https://arxiv.org/pdf/2608.27797v1)

**作者:** Lekai Chen `[一作]` (University of Colorado Boulder), Ashutosh Trivedi `[通讯]` (University of Colorado Boulder)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过将自然语言指令映射为正则语言，使用LLM与CAPAL学习DFA表示的技能和约束，并在Minecraft环境中实现持续约束执行与技能重用。

**💡 创新点**

创新点在于将技能与约束统一为可执行的DFA，利用LLM作为语义教师进行活跃学习，并通过DFA交叉、串联实现约束持续性和技能复用。

**🔧 技术方法**

使用GPT‑5.5/5.4‑mini作为LLM教师，CAPAL主动DFA学习，环境回放等价查询，DFA交叉与串联操作。

**📊 数据集**

主要使用Minecraft模拟器API观测数据（1429个符号的全局词表）以及iTHOR任务作为实验环境。

**📈 对比分析**

与代码生成基线对比，证明在保持时间/空间约束、任务完成率、LLM查询次数等指标上取得更优表现；例如，睡眠约束下约束合规率显著提高，技能重用降低LLM查询成本。

**⚠️ 局限性**

局限性包括只能处理正则语言（无法计数无限、存储复杂结构），依赖可用环境谓词，无法完全检测遗漏谓词，DFA组合可能导致状态爆炸，且对人类反馈误差不具鲁棒性。

---

## 78. Twin Worlds: Equivariance-Based Abstention for Evidence-Grounded Reasoning

**arXiv ID:** 2608.28018 | [PDF](https://arxiv.org/pdf/2608.28018v1)

**作者:** Vy Nguyen `[一作]` (RMIT University), Xiuzhen Zhang `[通讯]` (RMIT University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Twin Worlds（TW）框架，用结构保持的实体替换检验答案是否由证据推理而非记忆激活产生，若不一致则拒绝回答。

**💡 创新点**

创新点在于将“等变性”作为判定答案是否基于证据的标准，并通过多重“同生世界”结构保持替换来检测离群答案，避免仅依赖不确定性或证据充足性评估。

**🔧 技术方法**

技术包括实体去词化（delexicalisation）、按类型一一映射的合成实体替换、生成多重同生世界、对齐答案并计算等变性得分，阈值控制拒绝。

**📊 数据集**

使用四个基准数据集：HotpotQA、MIRAGE、FaithEval 和 FEVER。

**📈 对比分析**

与七种基线（零样本、置信度、足够上下文等）对比，TW 在所有模型（GPT‑5.1、LLaMA‑4、Mistral‑Small）上均取得最高或第二高的 F1、准确率、可靠度得分，且拒绝率保持低；在“难拒绝”子集更能及时拒绝。

**⚠️ 局限性**

局限性：仅适用于基于命名实体的任务，对数学或程序推理等无实体结构的场景效果未知；实体识别失败时鲁棒性有限；多语言扩展需设计语言特定的替换策略。

---

## 79. Hypothesize, Evaluate, Refine: A Scientific Agent for PDE Discovery with Unknown Spatial Coefficient Fields

**arXiv ID:** 2608.27475 | [PDF](https://arxiv.org/pdf/2608.27475v1)

**作者:** YuJie Huang `[一作]` (Fujian University of Technology), Zhuo-Xu Cui `[通讯]` (Institutes of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 HER-PDE 框架，通过科学代理在表达树语言中同时推断 PDE 结构与空间系数场，使用交叉激励转移评估完成非参数场估计；

**💡 创新点**

创新点在于：① 将 PDE 结构与空间系数场统一为完整表达树保留算子作用域与场重用；② 设计 Hypothesis Evaluation Interface（HEI）仅拟合声明的字段并通过双向转移评价；③ 采用两相科学代理搜索与局部细化相结合，最终在保密区间上审计；

**🔧 技术方法**

采用语言模型代理、符号表达树、Savitzky‑Golay 滤波求导、核函数（高斯）非参数场估计、交叉激励转移误差与封闭窗口误差评估；

**📊 数据集**

在五个二维合成系统上验证，包括地下水流、变系数扩散、反应‑扩散、对流‑扩散、各向异扩散，使用 Matérn‑3/2 随机场和 HIN‑PDE 基准导电率场，所有轨迹均含 5% 高斯相对噪声；

**📈 对比分析**

通过转移误差、封闭窗口误差、结构重现（精确、乘积规则、符号场重参数化）以及字段 Pearson 相关与相对 L₂ 误差对比；在所有五个案例中都准确重现了生成运算符，字段相关性均在 0.71–0.97 之间，平均相对误差约 0.28；

**⚠️ 局限性**

局限包括：仅使用两条实验轨迹；仅考虑时间不变空间场；仅处理二维系统；对非参数高维或时空变化、随机/噪声场的扩展尚未探究；

---

## 80. AI Alignment through a Game-theoretic Lens: A Survey

**arXiv ID:** 2608.27910 | [PDF](https://arxiv.org/pdf/2608.27910v1)

**作者:** Yanan Cai `[一作]` (James Cook University), Wei Xiang `[通讯]` (La Trobe University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文综述了基于博弈理论的后训练对齐方法，系统梳理了在偏好多样性、对齐优先级和时序动态三大挑战下的研究进展，并提出了一个三维分类框架；

**💡 创新点**

创新点在于将博弈理论作为统一语言，对齐问题重新表述为策略博弈，构建了从单玩家优化到多玩家对齐、从同步到顺序交互、从静态到动态演化的完整博弈视角，强调了对齐的策略稳定性和可解释性；

**🔧 技术方法**

主要技术包括：强化学习从人类反馈(RLHF)、直接偏好优化(DPO)、自我对抗博弈、自适应进化、堆栈尔伯格博弈、POMDP 价值-状态建模、以及各种博弈求解与收敛分析方法；

**📊 数据集**

论文并未提出新的实验数据集，而是聚焦于已有文献中的数据来源，例如公开的偏好对比数据、人工标注的多群体反馈数据、以及模拟社会互动的实验环境；

**📈 对比分析**

由于是综述性质，论文没有直接实验比较；但在综述中提及了多种对齐方法的收敛性质（平均迭代收敛、最后一次迭代收敛）和性能评价（胜率、对抗鲁棒性），并指出相对传统单模型优化方法在鲁棒性与公平性上的提升；

**⚠️ 局限性**

局限性包括：聚焦于后训练对齐，忽略预训练阶段、推理时防御、机制可解释性与治理框架；对齐理论的成熟度在不同子领域差异明显，社交交互、长期演化的理论保障相对薄弱；论文未提供统一的实验评测与基准。

---

## 81. ANCHOR: A Vision for Secure Persistent Key-Value Stores in Disaggregated Data Centers

**arXiv ID:** 2608.27819 | [PDF](https://arxiv.org/pdf/2608.27819v1)

**作者:** Viraj Thakkar `[一作]` (Arizona State University), Zhichao Cao `[通讯]` (Arizona State University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种面向离散化数据中心的持久键值存储安全框架，分为持久路径和易失路径两部分，利用版本化清单和TEE验证实现端到端完整性与新鲜度。

**💡 创新点**

创新点在于将版本演化视为完整性根，并在易失状态（缓存、索引、过滤器）上采用可验证的来源标签，将传统的“信任一切”转变为“仅验证可信”；同时设计了面向TEE的批量异步验证机制，兼顾性能与安全。

**🔧 技术方法**

采用AES/GCM/CTR/ChaCha20等对称加密、哈希校验、TEE（如Intel SGX/ARM TrustZone）托管的策略模块、异步I/O（io_uring）以及版本化清单签名等技术。

**📊 数据集**

论文未给出具体实验数据集，主要以LSM‑KVS（log‑structured merge tree）为示例，假设采用常见的YCSB或TPC‑C等标准工作负载进行评估。

**📈 对比分析**

通过对加密算法和异步I/O的微基准，展示批量验证可显著降低 enclave切换成本，异步提交提高IOPS；但论文未给出完整系统层面的性能对比数据，预计与传统单机LSM相比，性能影响可控制在10‑20%。

**⚠️ 局限性**

限制包括：未评估对DoS、访问模式泄露、时序侧信道等攻击；TEE内存受限可能成为弹性瓶颈；系统假设安全的密钥管理和可信代理；缺乏完整实验验证。

---

## 82. RASA: Disentangled Spatial-Motional Priors for Cross-Identity Character Animation

**arXiv ID:** 2608.28219 | [PDF](https://arxiv.org/pdf/2608.28219v1)

**作者:** Zhen Xiao `[一作]` (Hefei University of Technology), Tao Mei `[通讯]` (HiDream.ai Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RASA框架，解决跨身份角色动画中的空间对齐与运动控制分离问题。

**💡 创新点**

创新点在于同时使用Spatial Prior Calibrator实现连续2D空间校正与Inherent Motional Guider提供身份无关的3D运动指导，并构建专门评测跨身份结构差异的CIM-Bench基准。

**🔧 技术方法**

采用Diffusion Transformer、VAE编码、SMPL运动参数、RoPE编码以及双注入（SPC和IMG）等技术。

**📊 数据集**

训练数据包括TikTok、Champ、UBC等公开数据集以及约5k人类视频，评测使用TikTok数据集和新构建的CIM-Bench。

**📈 对比分析**

与多种SOTA方法对比，RASA在PSNR、SSIM、LPIPS、FID、FVD、FID-VID等图像与视频质量指标上均显著优于对手，且推理速度快、显存占用低。

**⚠️ 局限性**

局限在假设1:1身体部件对应，难以处理非人形多头、多臂或无手臂等拓扑差异，且手部与面部动作受限于SMPL缺乏细节。

---

## 83. Beyond Flat Netlist: Hierarchical Graph Representation Learning for Scalable Analysis of Sequential Circuits

**arXiv ID:** 2608.28188 | [PDF](https://arxiv.org/pdf/2608.28188v1)

**作者:** Jingyi Zhou `[一作]` (Chinese University of Hong Kong), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 DeepSeq3，一种层次化的电路表示学习框架，先将时序电路按 FF 边界划分为组合子图，再构造超级节点图（SNG），通过双层 GNN 和状态中心预训练学习组合子图与 FF 级别表征，最终用于功耗预测和 Bounded Model Checking，显著提升可扩展性与性能。

**💡 创新点**

创新点：1) 采用 FF 边界划分的组合子图 + SNG 结构，将大规模电路压缩至极低节点数；2) 两级 GNN 结合逻辑状态预测与状态可达性监督，显式建模注册级时序；3) 状态中心预训练，预测 FF 可达性矩阵和转移概率，提升模型对时序动态的理解；4) 轻量级 per-circuit fine‑tune，直接用于 BMC 引导搜索。

**🔧 技术方法**

技术：层次化图表示；组合子图的逻辑 0/1 状态预测 GNN；超级节点图的状态可达性监督 GNN；带注意力的 GRU 交互；Transformer pooling；逻辑-0/1 监督；状态可达性矩阵与一阶转移矩阵的联合预训练；BMC 结合模型推断的引导搜索。

**📊 数据集**

数据集：从 ISCAS’89、Opencore、ITC’99 共 3568 级电路，转为 AIG，单个电路约 100–500 节点；功耗估计使用 Nangate 45nm 标准单元库；BMC 基准来自 HWMCC。

**📈 对比分析**

与 DeepSeq2、MOSS 等基线对比：Stage1 逻辑状态预测 R² 0.9869/0.9916；Stage2 可达性 F1 0.8429；转移预测 R 0.8413；预训练时间比 DeepSeq2 加速约 4 倍；功耗估计平均 MAPE 4.58%（比 DeepSeq2 7.44% 低 38%）；BMC 指导搜索平均速度提升 18%；在困难实例中平均求解时间 191 s 对比 255 s。

**⚠️ 局限性**

局限性：1) 预训练仍需大量仿真生成转移矩阵，对极大电路可能受限；2) 仅在 AIG 结构验证，其他网表格式未覆盖；3) BMC 引导依赖模型推断，若模型误判可能导致搜索失败；4) 需要 per‑circuit fine‑tune，增加部署成本。

---

## 84. Post-Edit Re-Verification in Simulator-Backed Engineering Agents: A Controlled Comparison of Verification-Cadence Guidance

**arXiv ID:** 2608.28147 | [PDF](https://arxiv.org/pdf/2608.28147v1)

**作者:** Qingchuan Zhu `[一作]` (Sinopec Petroleum Engineering Zhongyuan Co., Ltd.), Pengju Ren `[通讯]` (Xi'an Jiaotong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在大型语言模型代理与外部仿真器（DWSIM）交互时，保留或省略显式的验证节奏指导对首次后编辑重新验证行为的影响；通过五个阿里巴巴/ Qwen 模型在八个合成压力修复案例上进行三次重复实验来收集数据。

**💡 创新点**

创新点在于将验证节奏（post‑edit re‑verification cadence）明确拆分为交互协议的可配置组件，系统地比较了在相同状态信息下，加入与去除节奏指导对代理行为的定量影响。

**🔧 技术方法**

使用了大型语言模型（Qwen 系列）、工具调用框架（调用 DWSIM 仿真器）、合法性检查器、状态跟踪与证据新鲜度管理等技术；实验通过 OpenAI 等 LLM API 进行。

**📊 数据集**

使用的数据集为八个人工构造的合成压力修复案例（包含标准流、入口压、目标压、温度等参数），每个案例在三次重复调用中产生 24 个评价槽。

**📈 对比分析**

比较方法：对每个评价槽记录三项指标——首次后编辑重新验证（Reverify）、验证节奏违规（Cadence Violation）和最终成功（Final Success）；在 CG（保留节奏指导）与 CO（省略节奏指导）两条件下分别统计数量。结果显示 CG 在所有模型中均获得更高的重新验证率、更低的节奏违规率和更高的最终成功率，差距可达约 50% 的百分点。

**⚠️ 局限性**

局限性包括：仅针对单一排气压失效家族；合成案例数量有限，缺乏真实工程项目的多样性；未进行统计推断分析；仿真器配置未完全公开，难以复现；以及对不同模型的结果表现差异未深入解释。

---

## 85. Knowing Before Answering: Decoding Language Models for Reliable RAG

**arXiv ID:** 2608.27661 | [PDF](https://arxiv.org/pdf/2608.27661v1)

**作者:** Syed Mahbubul Huq `[一作]` (City St George's University of London), Pranava Madhyastha `[通讯]` (City St George's University of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建三分类（Answer、Refuse、Conflict）RAG triage benchmark，利用内部隐藏层激活训练轻量级逻辑回归路由器，实现在生成前判别检索证据的可行性。

**💡 创新点**

创新点在于将RAG信号解码为三分类问题，证明模型内部隐藏层已编码证据充分性，并通过中层特征实现高效、无额外推理开销的决策。

**🔧 技术方法**

主要技术包括对单层隐藏状态和MLP输出提取、逻辑回归分类、注意力权重对比、以及隐藏层补丁干预等机制。

**📊 数据集**

使用自制的受控检索增强生成数据集，结合 TriviaQA、HotpotQA、Natural Questions，经过实体置换后生成Answer/Refuse/Conflict样本。

**📈 对比分析**

与提示式、专门RAG模型及表面文本基线对比，16种不同规模/架构的Transformer显示路由器平均准确率0.83–0.91，错误答率下降多达75%。

**⚠️ 局限性**

局限性包括实验基准人为控制，可能不充分代表真实RAG环境；模型内部解码依赖特定层，跨模型迁移性有限；对长上下文与非结构化文档的鲁棒性尚待验证。

---

## 86. Quanta Perception as Probabilistic Events

**arXiv ID:** 2608.27584 | [PDF](https://arxiv.org/pdf/2608.27584v1)

**作者:** Varun Sundar `[一作]` (University of Wisconsin--Madison), Mohit Gupta `[通讯]` (University of Wisconsin--Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种基于递归贝叶斯推理的量子感知框架——概率事件，用于实时处理单光子检测数据。

**💡 创新点**

创新点在于用运行长度后验替代固定阈值触发，生成三种低延迟感知流（时间稳定性、熵变、运动感知通量），并实现毫秒级到微秒级的自适应积分。

**🔧 技术方法**

核心技术包括单光子计数模型、有限状态贝叶斯更新、多位二进制/多位量子帧的泊松/二项似然、基于梯度/Log‑Gabor特征的空间聚合，以及GPU并行实现。

**📊 数据集**

使用了VisionSim、i2k高速数据集以及自制低照度动态场景数据，验证了单光子摄像机在0.05 lux及高速运动下的性能。

**📈 对比分析**

与传统固定窗口曝光、离线重建方法（QBP、QUIVER、FastDVDNet等）以及商用DSLR、高速摄像机、低光安全摄像机和事件相机进行对比，概率事件在保持高帧率（4k+）和低延迟的同时，性能与重建方法持平或更优，吞吐量提升四个数量级。

**⚠️ 局限性**

局限在于光子稀疏或过量时难以区分噪声与真实动态，且需要对光子计数统计模型做精准建模，适用范围受信息量限制。

---

## 87. Diva++: Dynamic Range Filtering over Hard Workloads

**arXiv ID:** 2608.27616 | [PDF](https://arxiv.org/pdf/2608.27616v1)

**作者:** Navid Eslami `[一作]` (University of Toronto), Niv Dayan `[通讯]` (University of Toronto)

**通讯引用:** 1278 | [OpenAlex ID](https://openalex.org/A5003707725)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了新的范围过滤器Diva与其增强版本Diva++，能够在满足低误报率、支持可变长度键与查询、支持动态更新的同时实现高查询与构造性能。

**💡 创新点**

创新点包括：1）利用样本Trie学习数据分布并通过截断公共前缀与后缀保留中间infix实现半鲁棒误报率；2）设计Infix Store的位图与分块存储方案，支持常量时间查询与动态分裂、拉伸；3）Diva++引入顺序保持熵编码与重复infix去冗余技术，进一步降低误报率并适应“锯齿形”分布。

**🔧 技术方法**

主要技术：缓存友好的Trie采样、最短公共前缀消除、infix截断、Knuth分区、位图/占用位标记、rank/select操作、y‑Fast Trie、Mehlhorn的秩保持熵编码、动态拉伸与分裂策略。

**📊 数据集**

使用多种真实世界键值集合（如URL、电子邮件、日志、数据库键、键值存储中的LSM树键）进行评估，并在WiredTiger等基准上测试。

**📈 对比分析**

通过与所有先前的范围过滤器（如SuRF、Proteus、SNARF、Oasis+、Memento、Aeris等）在误报率、内存占用、查询时间、构造时间以及动态操作性能上进行对比。实验显示Diva/Diva++在相同误报率下内存占用接近或优于最优过滤器，查询速度快于传统Trie/哈希方法，构造速度最快，且在动态场景下仍保持低误报率。

**⚠️ 局限性**

局限性：1）对“平滑”分布提供半鲁棒误报率，极端不平滑分布仍需Diva++的熵编码；2）动态实现会略增内存（约1 BPK）并引入分裂策略的复杂性；3）删除操作依赖于插入键的合法性，需要用户保证合法删除；3）在极大数据量下，infix长度随分裂递归下降，可能导致误报率对数级上升，需要额外的宽化或预测机制。

---

## 88. Beyond Non-IID: Learner--Client Distribution Mismatch in Federated Learning

**arXiv ID:** 2608.27715 | [PDF](https://arxiv.org/pdf/2608.27715v1)

**作者:** Yiming Xie `[一作]` (Northeastern University), Ningfang Mi `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对学习者中心的联邦学习，研究在学习者-客户端分布不匹配下的知识迁移，并提出一种动态影响感知客户端选择框架 DIC-KT，利用学习者代理数据评估每个客户端对学习者目标的边际贡献，实现自适应的客户端选择与加权聚合。

**💡 创新点**

创新点在于：①将学习者目标与客户端分布误差建模为多源迁移学习问题；②提出利用留一法评估学习者代理集上的影响值来动态估计客户端效用；③设计了基于影响值的加权聚合与客户端筛选策略，兼顾正向迁移与负向迁移抑制。

**🔧 技术方法**

核心技术包括：联邦学习中的样本数加权聚合；留一法（LOO）影响评估；基于影响值的加权公式 λ̃_c = (max{Δ_c,0}+ϵ^{1/|Δ_c|})^η；动态客户端筛选 Top‑M 机制；实验中使用 CNN 与 SGD 等常规优化器。

**📊 数据集**

使用 CIFAR‑10 图像分类数据集，按照 Dirichlet 分布生成 20 个非 IID 客户端以及一个包含 2000 张样本的学习者代理集。

**📈 对比分析**

与 FedAvg、KMeans、FL+HC、H‑Ensemble、Power‑of‑Choice 等静态与动态基线进行对比。实验显示 DIC‑KT 在多种分布不匹配与标签覆盖度设置下，平均精度提升约 15–35%（在最极端 α=0.1 时），方差更低，收敛更快，尤其在标签单一或分布差异大的场景表现突出。

**⚠️ 局限性**

局限性包括：①需要学习者代理数据来估计影响值，代理集规模较小易受噪声影响；②留一评估在大规模客户端集时计算开销增大；③对极端异构场景（如极稀疏标签）仍可能出现估计不稳；④实验仅在 CIFAR‑10 上验证，需进一步验证在真实任务（如自动驾驶）中的鲁棒性。

---

## 89. Benchmarking General Mobile Assistants in Challenging Real-World Scenarios

**arXiv ID:** 2608.27477 | [PDF](https://arxiv.org/pdf/2608.27477v1)

**作者:** Yiqi Zhu `[一作]` (Tsinghua University), Yang Liu `[通讯]` (Tsinghua University)

**通讯引用:** 110802 | [OpenAlex ID](https://openalex.org/A5100355638)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个新的移动助手评估基准，涵盖七个自定义开源 Android 应用和 300 条跨四个难度层级的任务，提供可重复的评估环境和统一的状态抽象。

**💡 创新点**

创新点在于扩大应用生态覆盖面、引入多层次任务难度（Atomic、Compositional、Cross‑Application、Realistic），设计统一的资产抽象和程序化状态验证，并系统评估 agent harness 对性能的影响。

**🔧 技术方法**

采用容器化 Android 模拟器、统一声明式资产框架、状态适配器与程序化验证、零样本 autoregressive 视觉‑only 代理、Qwen3.7‑Plus 用户模拟器，并在 harness 设计上实验上下文保留、显式状态追踪与模型特定结构化记忆。

**📊 数据集**

使用七个开源项目实现的自定义应用（如 ElementX、Tempus、XiaoShiLiu、HMDP、Meituan、Mall、Travel）以及手工构造的 300 条任务集合，所有数据均在本地自托管，保证可重复性。

**📈 对比分析**

对八种前沿模型（Doubao‑Seed‑2.1‑pro、Qwen3.7‑Plus、Qwen3.8‑Max、Gemini 3.5 Flash、GPT‑5.5、GPT‑5.6 Sol、Claude Sonnet 4.6、Claude Opus 4.7）在四个难度层级进行对比；结果显示 Qwen3.8‑Max 取得最高整体成功率 67.3%/平均分 80.35%，但在 Realistic 层级所有模型成功率低于 20%，表明复杂工作流仍是主要瓶颈。

**⚠️ 局限性**

局限性包括：评估仅限于 Android 仿真环境和九款应用，真实设备和多平台表现未知；任务设计仍相对人工且可能缺乏更丰富的自然语言对话；模型在复杂多步骤工作流中性能不佳，harness 设计尚未充分针对不同模型进行定制。

---

## 90. Memory-efficient GPU pipelines for real-time non-line-of-sight reconstruction

**arXiv ID:** 2608.28183 | [PDF](https://arxiv.org/pdf/2608.28183v1)

**作者:** Alfonso López-Ruiz `[一作]` (University of Zaragoza), Diego Royo `[通讯]` (University of Zaragoza)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了两种波动式非视线(NLOS)重建算法的GPU流水线，支持实时与离线两种工作模式，并在显存与吞吐量上实现显著提升。

**💡 创新点**

创新点包括：① 在GPU端离线构建环形‑半径传播核，消除运行时密集核重建；② 将多种优化融合（warp级光子计数、CUDA图、混合精度、共享内存裁剪、融合kernel）；③ 提出三种噪声抑制策略（深度相关平均、相干加权、混合专家）。

**🔧 技术方法**

技术手段主要是CUDA 13、OpenGL互操作、FFT、Stolt插值、Ring‑and‑Radius kernel、混合精度FP16、CUDA图、warp级计数、深度相关平均、相干加权以及多通道并行重建。

**📊 数据集**

使用公开的动态SPAD数据集（190×190×3002）以及多种室内/室外离线场景数据进行评估。

**📈 对比分析**

与Backprojection、Physics to the Rescue (PttR)、原始Phasor‑fields、离线MATLAB实现等基线比较：实时帧率提升至42×、显存仅占原来的8.2%；离线重建速度提升至98%+、显存降低至52%。

**⚠️ 局限性**

局限性包括：仍受FFT与波动模型的计算与显存瓶颈限制，分辨率受光子计数与频率范围影响；低SNR问题仍需更多光子；部分优化对特定GPU与编译环境高度敏感。

---

## 91. The Effect of Emotional Context on Large Language Models' Endorsement of Premature Decisions: Comparing Emotional Vulnerability Across Six Commercial Models

**arXiv ID:** 2608.27465 | [PDF](https://arxiv.org/pdf/2608.27465v1)

**作者:** Cheolho Shin `[一作]` (Yonsei University), Kunho Lee `[通讯]` (St. Johnsbury Academy Jeju)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了情绪表达对大型语言模型在做出过早决策时的支持力度的影响，采用冷/中性/痛苦三种情境对照设计。

**💡 创新点**

创新点在于将情绪效应与会话长度效应解耦，使用三条件设计验证情绪本身能显著提升模型的赞同度，并揭示此风险不随价格层级变化。

**🔧 技术方法**

技术手段包括多模型对比实验、八项量表自动评分、混合效应模型分析、三层审计校验以及独立判别模型交叉验证。

**📊 数据集**

数据集为人工构造的三类决策情境（职业转型、商业扩张、移民留学）与相应的冷、中性、痛苦对话样本，合计324条会话。

**📈 对比分析**

通过对六个商业模型（OpenAI、Anthropic、Google 的顶级与中级版本）进行对比，发现情绪情境下平均提升约12.9分（d≈0.51），大多数模型出现显著情绪敏感性，性能不随价格层级显著差异。

**⚠️ 局限性**

局限性包括仅使用合成用户情境、样本量相对有限、评判者为大型语言模型、缺乏真实文化多样性、以及对单一情感表达方式的覆盖不足。

---

## 92. CAITLYN: Can LLM Agents Autonomously Synthesize Defenses against Emerging Injection Attacks?

**arXiv ID:** 2608.27990 | [PDF](https://arxiv.org/pdf/2608.27990v1)

**作者:** Zi Liang `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CAITLYN，一个可自适应的中间件，用以防御大型语言模型代理的注入和毒化攻击；

**💡 创新点**

创新点在于将防御规则抽象为可执行的“技能”库，并结合基于CEGIS的终身合成循环，实现零样本自适应升级；

**🔧 技术方法**

技术包括双系统架构（快速签名+LLM判别）、LLM提示生成器、确定性验证器、行为监控与自动化合成循环；

**📊 数据集**

使用AgentDojo、ASPI、SafeClawBench、Emerging等公开注入/毒化基准数据集；

**📈 对比分析**

与七个基线对比，CAITLYN在标准测试中达到或超过最佳检测精度，token开销低于单一LLM判别，并在Emerging上将攻击成功率降低约40个百分点；

**⚠️ 局限性**

局限在于合成过程受限于生成模型质量、技能库维护成本，以及对极端复杂隐蔽攻击或多轮对抗的鲁棒性仍有限。

---

## 93. SafeStep: An Interactive Demonstration of Semantic Communication for Pedestrian Safety Monitoring

**arXiv ID:** 2608.27688 | [PDF](https://arxiv.org/pdf/2608.27688v1)

**作者:** Christian McDowell `[一作]` (Auburn University), Yin Sun `[通讯]` (Auburn University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `9cc9baba-5356-466d-81ff-d80028d90279` `fede83ac-7505-405f-ab37-e7284695c47f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 SafeStep 平台，实现了实时、交互式的基于浏览器的语义通信演示，用于行人安全监测；

**💡 创新点**

引入了 Meta‑VIB 设计，首次将年龄信息（AoI）纳入语义通信的端到端模型，并利用 FiLM 进行条件化，以实现对信噪比、码长、AoI 的无缝泛化；

**🔧 技术方法**

使用了 Joint Source–Channel Coding（JSCC）、DeepJSCC、VIB、Hyper‑VIB、ATROC 等基线模型；Meta‑VIB 采用循环网络+FiLM+有序前缀符号、信息浓缩正则化；前端采用 YOLO 检测、IoU 跟踪、SSE/HTTP 交互；后端使用 NVIDIA RTX 6000 GPU；

**📊 数据集**

采集自 Toomer’s Corner 四路实时摄像头，使用自制标注数据集训练 YOLO16 车辆检测和 YOLO11s 行人检测；

**📈 对比分析**

通过比较六种传输器在不同 SNR、码长、AoI 条件下的平均任务损失（L̅），Meta‑VIB 在大多数条件下表现最优，最高可比基线低 92.1% 的任务损失；同时在多用户压力测试中，20 用户可维持 5 帧/秒，100 用户仍无请求失败，平均响应延迟 < 1 秒；

**⚠️ 局限性**

局限性包括：在用户数激增时每个浏览器帧率下降至约 1 帧/秒；仅在模拟 AWGN 信道下测试，未验证在真实无线环境下的鲁棒性；依赖单一 GPU，扩展性受限；仅针对行人安全场景，未涉及更广泛的任务。

---

## 94. Why Didn't It Check? Unsupported Final Claims and Their Repair in Two Tool-Equipped Language Models

**arXiv ID:** 2608.27768 | [PDF](https://arxiv.org/pdf/2608.27768v1)

**作者:** Justin Bronder `[一作]` `[通讯]` (Corabo), Justin Bronder (Corabo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究检验了语言模型在缺失关键信息时是否会未经证实地给出最终答案，并通过可见证据检测规则和匹配回放验证补充证据是否能修正此类错误。

**💡 创新点**

创新之处在于提出了在不知隐藏答案的前提下检测并重放“未确认最终声明”事件的实验框架，证明补充证据能在匹配状态下完全修正错误。

**🔧 技术方法**

主要技术包括固定模型设置、工具调用接口、基于可见证据的检测规则、离线重放、以及Clopper‑Pearson置信区间统计。

**📊 数据集**

实验使用了两类人工任务（并置完整性与时间优先级）中的表格数据，共计512个首次响应，用于Qwen3‑32B和Gemma 4模型。

**📈 对比分析**

通过比较检测到的33个错误案例在加入补充证据、无信息和支持证据三种重放中的修复率，发现补充证据在所有案例中修复100%，无信息0%，支持证据保持不变；在另一组64例中自动检测规则将准确率从54/64提升至64/64。

**⚠️ 局限性**

局限性包括仅在固定人工任务和单一模型设置下评估，未检验错误向相反方向修正的可能；未测量在常规流量中误触发率；结果受解码随机性影响；Gemma实验采用非推荐采样，缺乏普适性。

---

## 95. Condorcet-Winning Sets and Peer Selection in Planar Metric Elections

**arXiv ID:** 2608.27653 | [PDF](https://arxiv.org/pdf/2608.27653v1)

**作者:** Gabriel de Azevedo `[一作]` (Cornell University), Ulysse Hennebelle `[通讯]` (Cornell University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在平面度量投票模型下研究Condorcet获胜集合，提出强同行选举概念，并给出ℓ₁/ℓ∞下三人委员会、ℓ₂下四人委员会以及当所有候选人与某选民等距时两人委员会的理论上限。

**💡 创新点**

首次将ε‑net（尤其是强矩形ε‑net）与Condorcet获胜集合问题联系起来，利用几何分割与最近点逼近技术在不同范数下显著改进先前已知的上界。

**🔧 技术方法**

核心技术包括：平面几何分割、弱/强ε‑网理论、凸包与支配区域性质、最近点舍入法、以及对矩形与对角线对称性的精细分析。

**📊 数据集**

本文为理论工作，没有使用实验数据集，所有结果均为纯数学证明。

**📈 对比分析**

与先前工作相比，本研究将强矩形ε‑网阈值从9/16降低至1/2，实现ℓ₁/ℓ∞下的三人委员会、ℓ₂下的四人委员会；在等距候选人情形下证明两人委员会可行；这些上界在现有文献中均为最佳或最接近最佳。

**⚠️ 局限性**

局限性包括：强矩形ε‑网上界是否紧凑尚未确定；ℓ_p（1<p<∞, p≠2）下单人候选人的最优比例仍未知；是否存在更优的弱ε‑网上界导致ℓ₂下的三人委员会；以及对高维情况的推广仍需进一步研究。

---

## 96. What Will This Copper Look Like Later? Forecasting Surface Appearance and Rendering It as a PBR Material

**arXiv ID:** 2608.28102 | [PDF](https://arxiv.org/pdf/2608.28102v1)

**作者:** Teejuta Sriwaranon `[一作]` (Chulalongkorn University), Pizzanu Kanongchaiyos `[通讯]` (Chulalongkorn University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一套从固定摄像机拍摄的铜样本图像预测未来表面外观，并将预测转换为PBR材质贴图的完整流程。

**💡 创新点**

首次在未见样本上验证并发现仅参数无学习的全局色彩外推能够跨样本转移，比任何训练模型都优，揭示学习空间先验会导致跨样本失效。

**🔧 技术方法**

采用闭式全局色彩矩阵外推、卷积递归、Transformer、HorizonNet等多种时空预测模型，以及基于薄膜干涉的物理约束与PBR映射公式。

**📊 数据集**

使用两组固定摄像机记录的铜样本（加速室与户外），共约1900帧，作为训练与留出测试。

**📈 对比分析**

采用留一记录、加速单位时间步的前瞻任务，对比复制上帧、闭式外推与四种学习模型，发现闭式外推在两个转移方向均以5–55%的平均误差提升，训练模型在未见样本上均逊于复制上帧。

**⚠️ 局限性**

仅单金属单样本，未覆盖不同几何、光照或多样化铜件；缺乏足够样本评估空间先验普适性，且加速单位难以映射到实际年限。

---

## 97. CommerceVibe: Learning to Design E-Commerce Creatives as Executable Visual Code via Dual-Feedback Reinforcement Learning

**arXiv ID:** 2608.27893 | [PDF](https://arxiv.org/pdf/2608.27893v1)

**作者:** Yajiao Xu `[一作]` (Tongji University), Chengfu Huo `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种名为CommerceVibe的电商创意生成模型，能够一次性生成可执行的HTML/CSS代码，保证产品图像、文字内容和布局可直接编辑和复用。

**💡 创新点**

创新点在于：①将创意生成建模为条件HTML/CSS程序合成，实现可执行、可编辑的输出；②提出双重反馈强化学习，结合规则检测（文本、产品、布局）和视觉语言模型（VLM）偏好评估，兼顾结构约束和整体视觉质量。

**🔧 技术方法**

使用的技术包括：大语言模型Qwen3.5‑9B、基于规则的可渲染文档验证器、固定VLM Judge（Qwen3‑VL‑Plus）、GRPO强化学习框架以及自定义的渲染与评估管线。

**📊 数据集**

数据集为28,568条经过质量控制的真实电商创意样本（SFT阶段）以及1,300条包含单图/多图场景的评测基准，涵盖34个商品类别。

**📈 对比分析**

与GPT‑5.5、Claude Opus 4.8、Gemini 3.5 Flash等外部模型进行对比。自动评测中，双反馈RL版本在规则得分（94/100）和整体得分（94/100）上均超过所有竞争模型；专家盲测中获得90/100的最高分，显示其在设计质量和视觉表现上均优于基线。

**⚠️ 局限性**

局限性包括：①依赖固定规则设置，可能无法覆盖所有设计细节；②VLM Judge为固定模型，评估主观性受限；③多图场景下仍需进一步扩展以处理更大规模的产品组合；④模型训练资源较大，需多 GPU 支持。

---

## 98. PanelShield: Verifiable Closed-Loop Safe Planning for Robotic Industrial Panel Operation

**arXiv ID:** 2608.28305 | [PDF](https://arxiv.org/pdf/2608.28305v1)

**作者:** Guipeng Xin `[一作]` (Huazhong University of Science and Technology), Zhongxu Hu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 PanelShield，一种基于双重形式化验证（LTL 与 Safety FSM）与计数反例驱动的闭环安全规划框架，用于工业面板操作。

**💡 创新点**

创新点在于：①把手册文本中的程序约束自动编译成可计算的 LTL 与 FSM 规则；②采用双重验证实现全局时序与局部过渡合法性检查；③在检测到违规时输出可定位的反例，驱动局部计划修正，形成 generate–verify–repair–re‑verify 循环；④在多级长时程任务中保持可审计与可重现。

**🔧 技术方法**

技术主要包括：大语言模型（VLM/LLM）用于规划；手册检索与文本抽取；符号化设备状态抽象与转移模型；LTL 公式与 FSM 自动生成；结构化反例产生与计划约束反馈；ROS 2 与实时控制实现。

**📊 数据集**

数据集：自构建的面板操作基准，涵盖 VFD、Power Management System、Hydraulic Control Unit 三类设备，包含 300 条三级指令与对应专家标注计划；以及真实机器人平台的实测数据。

**📈 对比分析**

与 LLM‑only、Prompt‑Safety、Post‑hoc Judge 等基线对比，PanelShield 在 L1/L2/L3 任务成功率分别为 93.0%/84.0%/43.0%，比 LLM‑only 提高 21%/51%/21%，违例率降至 2.7%（最低），总时延约 4.1 s，验证与计划耗时均低于 Post‑hoc Judge。

**⚠️ 局限性**

局限：①仍需专家人工编译手册规则，难以完全自动化；②验证可靠性受感知误差影响，需设置阈值与复测机制；③若规则覆盖不全、规划失真或感知错误，残留违例仍可能出现，安全关键场景需严格阻断执行。

---

## 99. SETU: An Agentic Ecosystem for Multilingual, Persona-Aware Communication Coaching

**arXiv ID:** 2608.27524 | [PDF](https://arxiv.org/pdf/2608.27524v1)

**作者:** Jonnalagadda Maruthi Tejas `[一作]` (Quanta People Solutions Pvt. Ltd.), Mousita Dhar `[通讯]` (Quanta People Solutions Pvt. Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了SETU，一个多模态代理生态系统，用于多语言、面向人物的招聘与销售对话培训与评估。

**💡 创新点**

采用代理化设计，将视频、音频、文本、关联度等功能拆分为独立可替换的代理，并通过可信任权重调度实现可解释、可调节的多模态评估。

**🔧 技术方法**

使用MediaPipe Holistic提取视觉特征、Librosa提取声学特征、Sarvam ASR完成多语种/代码混合转录、LLM进行评分与关联度评估，并利用LangGraph实现任务路由与信任加权聚合。

**📊 数据集**

使用18条销售演练视频，涵盖英语、印地语、泰卢固语及其与英语混合的多语言，针对CEO、教师、经理等多人物角色，人工标注为Good/Fair/Poor。

**📈 对比分析**

与文本仅LLM（B1）和非代理多模态摘要（B2）对照，SETU在教练一致性ρ=0.78、可解释性4.3/5、语言macro‑F1 0.86，平均延迟38.6s；相比B1 ρ=0.61、B2 ρ=0.69，教练时间减少约40.9%。

**⚠️ 局限性**

依赖高质量视频/音频与ASR准确性，提示设计和数据集规模有限，缺乏大规模多样化评估，文化与视觉差异可能影响模型泛化。

---

## 100. A-PAIR: A Benchmark and Identity-Consistent Grounding Framework for Air-Ground Cross-View Referring Person Detection

**arXiv ID:** 2608.27997 | [PDF](https://arxiv.org/pdf/2608.27997v1)

**作者:** Zhoupeng Guo `[一作]` (Southeast University), Pengfei Zhu `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出空气-地面跨视角指代人物检测（AGCV-RPD）任务，并创建了对应的基准数据集 A-PAIR。

**💡 创新点**

创新点在于：①将任务拆分为“身份一致性”与“跨视角配对”，①使用“因子化注释与引用对齐”（FARA）生成半自动的跨视角指代描述；②设计“身份一致性参照定位”（ICRG）框架，融合因子化定位、候选完整性和跨视角一致性校正；③在同一框架下实现地面与空中视角的联合推理。

**🔧 技术方法**

核心技术包括：GroundingDINO 模型的因子化文本输入、所有人物检测器（候选完整性）、ResNet-50 视觉编码器做身份一致性评分，以及多源信号融合的阈值搜索推理。

**📊 数据集**

使用了 G2APS 原始数据构建的 A-PAIR，包含 22,137 交叉视角指代样本，7,588 组地面/空中图像，3,891 个空中图像，涵盖 7,588 个身份。

**📈 对比分析**

与多种单视角基线（TransVG、PropVG、GDINO 等）对比，ICRG 在 Pair F1 上提升至 22.28%（比最强基线 16.65% 提升约 5.6%），同时在地面和空中单视角指标上也有显著提升。

**⚠️ 局限性**

局限性：①数据量相对有限，且主要为静态场景；②空中目标小且外观弱，导致检测瓶颈仍显著；③模型在处理极其相似的多人人物或高度遮挡场景时仍易出现错误；④缺乏对时间序列的建模和多帧信息利用。

---

## 101. Embedding Models for Stance-Aware Argument Retrieval

**arXiv ID:** 2608.28283 | [PDF](https://arxiv.org/pdf/2608.28283v1)

**作者:** Angelo Sparacino `[一作]` (Imperial College London), Adam Dejl `[通讯]` (Imperial College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一套针对立场感知论证检索的密集嵌入方法，重点解决了基准模型在关注主题与立场时的偏差（topical collapse）问题。

**💡 创新点**

创新点在于：① 设计诊断性词汇消融指标（RIS/RCS、DI）用于量化模型对指令与主题的关注度；② 通过“混合负样本”平衡训练与合成立场颠倒样本的双阶段数据增强，抑制词汇快捷方式；③ 将优化后的稠密检索与 BM25 轻量稀疏检索融合，提升领域专门语料的检索质量。

**🔧 技术方法**

核心技术包括：密集检索的双编码器（BGE‑Large、Instructor‑XL、Qwen3‑Embedding‑8B）、基于 MNRL 的对比学习、LoRA 微调、词汇消融诊断、合成样本生成（Gemini 3.5 Flash）、Hybrid 搜索（稠密 + BM25 的 RSF 融合）。

**📊 数据集**

使用的主要数据集有：TFU 训练/验证/评估论证集、AVeriTeC 论证集、专用医学域 ArgTumour 论证集；同时生成了多组多样化指令（20 条标准化、10 条支持、10 条攻击）。

**📈 对比分析**

通过 Precision@R、NDCG@10 等指标与基线（基准嵌入、单一负样本对比学习、跨编码器重排序）对比，实验显示：Qwen3‑Embedding‑8B 在 Mixed+Aug+Hybrid 设置下在 TFU Evaluation 上 Precision@R 提升至 0.845，stance 错误率降至 7.8%；相较于基准模型，半硬负样本率下降 60%，但在某些模型中出现 stance 回退。

**⚠️ 局限性**

局限性包括：① 仍存在模型特定的“立场退化”风险，尤其是 BGE‑Large 与 Instructor‑XL 在去硬负样本后表现不稳定；② 词汇消融诊断可能受句法扰动影响；③ Hybrid Fusion 在稀疏权重 λ 选择上需手动调节，无法自适应不同查询；④ 依赖人工生成的合成样本，质量不一；⑤ 只在三类数据集验证，跨领域通用性仍待进一步测试。

---

## 102. Explainable Diabetic Retinopathy Classification Using Vision Foundation Models

**arXiv ID:** 2608.28207 | [PDF](https://arxiv.org/pdf/2608.28207v1)

**作者:** Abhishek Verma `[一作]`, Juan Miguel Lopez Alcaraz `[通讯]` (Universidade de Lisboa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了基于视觉基础模型的可解释糖尿病视网膜病变（DR）二分类框架，结合全微调、线性探测和LoRA等适配策略，并对Grad‑CAM与HiResCAM进行定量评估。

**💡 创新点**

创新点在于系统性比较三种基础模型（DINOv2、CLIP、ViT）与三种适配策略，并将解释性与病灶标注进行客观量化；同时展示LoRA在保持性能的同时显著降低可训练参数。

**🔧 技术方法**

采用DINOv2、CLIP、ViT三种视觉基础模型，利用全微调、线性探测、低秩适配（LoRA）进行迁移学习，并使用Grad‑CAM与HiResCAM生成可解释热图。

**📊 数据集**

使用公开数据集ODIR（内部训练/测试）、APTOS（外部验证）和IDRiD（专家标注病灶用于解释评估）。

**📈 对比分析**

通过AUROC比较，DINOv2‑LoRA在内部数据上最高AUROC 0.758，ViT‑Full与DINOv2‑Full在外部数据上均达到0.920；LoRA在参数量上比全微调低约两万倍，同时解释性指标（Dice、IoU、指向游戏）与专家标注存在一定对应。

**⚠️ 局限性**

局限在于仅使用公开数据集，缺乏多中心多设备样本；仅做二分类而非分级；解释方法只能显示空间对应，无法证实因果关系。

---

## 103. Node-wise Feature Encoding for Neural Performance Prediction

**arXiv ID:** 2608.27794 | [PDF](https://arxiv.org/pdf/2608.27794v1)

**作者:** Matthew Grenier `[一作]` (University of South Carolina), Ramtin Zand `[通讯]` (University of South Carolina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出FeatureFormer，一种基于门控图自注意力的神经网络性能预测器，显式将每个节点的FLOPs、参数量和内存占用等计算成本嵌入编码；

**💡 创新点**

创新点在于①引入节点级计算成本编码，使得预测器能更好地聚焦对延迟和能耗贡献大的操作；②在门控图自注意力中加入多种邻接掩码与门控机制，提升信息聚合效果；③构建了新的NNEQ能耗数据集，统一评估延迟和能耗预测。

**🔧 技术方法**

使用了门控图自注意力（GGSA）机制、基于Sinusoidal编码的特征向量、Transformer‑style多头注意力、以及对图的多视图掩码；

**📊 数据集**

使用NNLQ（延迟）和新构建的NNEQ（能耗）数据集，覆盖10个CNN家族，共约20,000个模型。

**📈 对比分析**

与现有的GNN/Transformer预测器（NN-Former、NNLP、NAR‑Former V2等）在留一法和全域评测中进行对比，FeatureFormer在平均MAPE和10%误差内准确率上均实现了1–3个百分点的提升，尤其在跨域场景表现更佳；

**⚠️ 局限性**

局限性包括模型参数量较大、主要针对CNN结构，且评测集中在单一GPU/嵌入式平台，跨硬件泛化能力仍待验证。

---

## 104. The Impact of Magma: A Ground-Truth Fuzzing Benchmark

**arXiv ID:** 2608.28016 | [PDF](https://arxiv.org/pdf/2608.28016v1)

**作者:** Ahmad Hazimeh `[一作]` (EPFL), Mathias Payer `[通讯]` (EPFL)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并维护了Magma fuzzing benchmark，通过将真实漏洞注入目标程序并提供oracle，可实现对bug的到达、触发、检测三维度评估。

**💡 创新点**

创新点包括：①使用真实bug进行forward‑porting，保持 benchmark 与最新漏洞保持同步；②区分bug的到达、触发与检测；③采用PCA评估目标多样性；④提供运行时监控、oracle与可插桩的canary；⑤引入自动化版本/补丁工具。

**🔧 技术方法**

采用静态/动态插桩技术、可插桩canary、runtime monitor、PCA、ASAN验证、覆盖/crash统计，并配套自动化版本化与补丁工具。

**📊 数据集**

使用9个多样化目标（libpng、libtiff、libxml2、poppler、openssl、sqlite3、PHP等）及其127个ground‑truth bug，覆盖多种CWE；同时整合了246条最新CVE信息。

**📈 对比分析**

比较方法基于三维度指标：reached、triggered、detected；在24h实验中，AFL++触发40个bug，Honggfuzz 28，libFuzzer 10；总覆盖率77/127 bug被到达，43/127被触发，已迁移的bug91%可达、34%触发，体现了 benchmark 的有效评估能力。

**⚠️ 局限性**

局限性：需人工维护bug注入与oracle设计；部分bug难以触发或检测（尤其是语义/资源耗尽类）；随着目标更新需持续维护；评估依赖于当前fuzzer的实现与配置，可能对新型bug类型覆盖不足。

---

## 105. Deriving Scaling Laws for OpenEuroLLM Models: Learning Rate, Batch Size and Loss

**arXiv ID:** 2608.28308 | [PDF](https://arxiv.org/pdf/2608.28308v1)

**作者:** Niccolò Ajroldi `[一作]` (ELLIS Institute Tübingen), Aaron Klein `[通讯]` (ELLIS Institute Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地研究了英语主导语料上稠密大语言模型预训练时学习率与批量大小的缩放规律，并在 Warmup‑Stable‑Decay 学习率调度下探究学习率衰减对超参数最优值与损失的影响。

**💡 创新点**

提出了联合与单个超参数的多尺度缩放模型，展示了学习率随模型规模指数衰减、批量大小随数据规模指数增长的关系；同时引入 Skaling 形式的损失缩放法，显著改善了对极端规模（1.7B 参数、300B 词汇）的外推精度。

**🔧 技术方法**

使用 AdamW + WSD 学习率调度、二次平滑估计、OLS 与多项式回归、Huber 损失最小化、L‑BFGS 拟合、Bootstrap 置信区间，并通过量化验证与已知 Chinchilla、Skaling 等模型比较。

**📊 数据集**

采用高质量英文子集（≈300B 词汇）并结合 50,304 词表，实验覆盖 47M–1.7B 参数模型、2^4–2^10 批量、0.00025–0.004 学习率，所有配置均基于同一数据流与初始化种子。

**📈 对比分析**

通过在 held‑out 1.7B 模型上验证，联合学习率与批量大小的缩放法平均绝对百分误差分别为 37% 与 22%；Skaling 形式的损失缩放在外推时 RMSE 仅为 0.0056，明显优于 Chinchilla 的 0.0174；在下游 DCLM‑CORE 评估中，OpenEuroLLM 在同等 FLOPs 下表现出比 Pythia、SmolLM2 更低的错误率，验证了其计算效率。

**⚠️ 局限性**

局限性包括：对更大规模（超出一倍范围）的泛化尚未验证；计算最优缩放模型仍需进一步细化；不同数据集、模型与训练配置的差异导致缩放指数波动，跨环境迁移的可行性有限。

---

## 106. Stay Seated: Learning Omnidirectional Humanoid Locomotion on a Passive Mobile Chair with Casters

**arXiv ID:** 2608.28090 | [PDF](https://arxiv.org/pdf/2608.28090v1)

**作者:** Kango Yanagida `[一作]` (University of Osaka), Takato Horii `[通讯]` (University of Osaka)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

研究了在被动移动椅子上进行的全向坐姿运动，旨在实现坐姿下的运动和物体操作。

**💡 创新点**

首次将坐姿运动与全向速度跟踪结合，且不依赖于运动模仿奖励或人类示范，成功实现了零-shot的仿真到现实转移。

**🔧 技术方法**

使用了深度强化学习（DRL）技术，结合了对称性正则化（SY）、脚滑正则化（FS）和命令课程（CC）等训练组件。

**📊 数据集**

使用了Unitree G1机器人和一个带有五个被动滑轮的移动椅子进行实验。

**📈 对比分析**

通过2^3全因子比较，发现FS单独使用时可能导致局部最优，而与SY或CC结合使用时则避免了这一问题。SY+CC在跟踪精度上表现最佳，且在多个训练条件下均能保持高成功率。

**⚠️ 局限性**

研究局限于单一椅子、地面条件和命令范围，未来工作将扩展到更复杂的坐姿运动和物体操作任务。

---

## 107. When Teacher Guidance Misleads: Reward-Aligned On-Policy Distillation

**arXiv ID:** 2608.27960 | [PDF](https://arxiv.org/pdf/2608.27960v1)

**作者:** Siyuan Gan `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Reward-Aligned On-Policy Distillation (RA-OPD)，通过筛选与奖励不一致的轨迹来改进大语言模型的后训练蒸馏。

**💡 创新点**

创新点在于使用已验证的最终奖励作为轨迹级别的可靠性判定，无需多次rollout，直接过滤不一致轨迹，保持训练效率且更可靠。

**🔧 技术方法**

利用反向KL逆向的 OPD 目标、轨迹级别的蒸馏回报、结果验证器以及梯度停止技巧实现该方法。

**📊 数据集**

使用 Qwen3 与 DeepSeek-R1 系列模型的 math（AIME, AMC, MATH, Minerva, OlympiadBench）和 code（HumanEval+, MBPP+, LiveCodeBench v6）数据集进行评估。

**📈 对比分析**

与标准 OPD、ExOPD、Uni-OPD 等方法对比，RA-OPD 在七个数学基准和三个代码基准上平均提升 3-5 点（数学 avg@k/ pass@k），并保持与 OPD 相当的训练时间。

**⚠️ 局限性**

局限性包括仍需手动设计阈值 m_i、仅适用于已可验证奖励的任务，且对极少奖励差异的提示无法提供更细粒度筛选。

---

## 108. Beyond the Bethe Approximation of the Permanent

**arXiv ID:** 2608.28031 | [PDF](https://arxiv.org/pdf/2608.28031v1)

**作者:** Nima Anari `[一作]` `[通讯]` (Stanford University), Nima Anari (Stanford University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

该论文提出了一种多项式时间的确定性算法，可在指数因子 c^n（其中 c<√2）内逼近任何非负矩阵的行列式。

**💡 创新点**

创新点在于引入了配对下界证明（paired lower certificate）以及对 Bethe 上界的稳定性分析，突破了传统 Bethe 永久近似中 √2 阶乘的限制。

**🔧 技术方法**

使用了 Bethe 永久、稳定多项式（stable polynomial）理论、熵与 KL 散度的几何分析以及正则化的 Bethe 目标函数的 KKT 条件。

**📊 数据集**

本研究未使用任何实验数据集，全部为理论分析与证明。

**📈 对比分析**

通过理论比较，算法实现了比现有确定性近似（√2)^n 更小的指数基数，并与随机化 FPRAS 的近似精度相竞争；在理论上证明了 L_X(A)≤perm(A)≤(√2e^{-ε})^n L_X(A)。

**⚠️ 局限性**

局限性包括：常数 ε 未被进一步优化；算法实现需求解一个正则化的凸优化问题，计算量较大；证明依赖于非平凡的稳定性与信息论不等式，实际可实现性与实用性仍有待评估。

---

## 109. VersaGauss: A Versatile Framework for Generating Multiphase Dynamics with 3D Gaussians

**arXiv ID:** 2608.28069 | [PDF](https://arxiv.org/pdf/2608.28069v1)

**作者:** Ruijie Su `[一作]` (Sun Yat-sen University), Jianhuang Lai `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个统一的框架 VersaGauss，能够从少量图像生成、模拟并渲染多相（如流体、弹性体、沙子、雪等）交互的 3D 动态场景。

**💡 创新点**

创新点包括：① 基于 3D Gaussian 的粒子裁剪策略降低粒子数量；② 设计 Coupled Multiphase Point Method (CMPM)，在 MPM 网格中分别维护不同相的速度，实现强耦合；③ 通过球谐插值和高斯演化策略实现逼真的流体渲染；④ 将 Gaussian 与物理属性直接绑定，兼容多相材料。

**🔧 技术方法**

主要技术：3D Gaussian Splatting、Material Point Method (MPM)、Warp GPU 框架、球谐颜色插值、粒子裁剪与高斯演化算法。

**📊 数据集**

使用 TRELLIS 生成的 3D Gaussian 体素、SAM 进行对象分割以及自制的多相交互场景（含流体、弹性体、沙子、雪等）作为输入数据集。

**📈 对比分析**

与 PhysGaussian、Wan2.1、CogVideoX‑5B 等方法对比，实验显示 VersaGauss 在多相动力学准确性上优于对手；粒子裁剪后每帧仿真时间从 25.5 s 降至 8.8 s，渲染时间从 0.26 s 降至 0.13 s，图像质量（PSNR）保持在 45–47 之间。

**⚠️ 局限性**

局限性：材料类型需人工指定；对极端拓扑变化和极稀薄相仍存在数值稳定性挑战；目前对多相参数的自动推断缺乏支持。

---

## 110. RealSWE: A Compositional Evaluation of Coding Agents under Realistic User Requests

**arXiv ID:** 2608.27831 | [PDF](https://arxiv.org/pdf/2608.27831v1)

**作者:** Gyuhyeong Kim `[一作]` (Sungkyunkwan University), Sunjae Lee `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为RealSWE的开源评测基准和可配置框架，用以在更贴近真实用户请求的情境下评估编码代理。

**💡 创新点**

创新点在于系统性地把真实用户请求拆解为信息组成和语言风格两维度，并基于这些维度将SWE‑bench问题转化为多变体任务族，实现对信息缺失与语言差异对性能影响的可控实验。

**🔧 技术方法**

主要技术包括：利用LLM（GPT‑5.4）进行信息分类与语言风格标注、对原始问题进行语义拆分与重写、构建可配置的多变体任务集，以及在统一的SWE‑agent框架下进行多模型评测。

**📊 数据集**

使用的数据集为SWE‑chat（真实开发者–代理对话）、SWE‑bench Verified与Pro（官方基准问题），并在此基础上生成381个多变体任务族。

**📈 对比分析**

对七款LLM（DeepSeek V4 Pro/Flash、MiMo V2.5 Pro/Plain、Claude Haiku 4.5、Qwen3.7 Plus、MiniMax M3）在原始与RealSWE两种输入下的“解题率”进行比较；结果显示，RealSWE下平均下降约6.4个百分点，且模型排名出现显著变动，且发现“Desired Behavior”和“Motivation”信息对解题率影响最大。

**⚠️ 局限性**

局限性包括：仅评估单轮任务，未考虑交互式澄清；未包含最新最强的LLM；任务变体样本受限于原始问题中完整字段的可用性；以及可能存在的选择偏差需要进一步验证。

---

## 111. Thinking Costs Tokens: When More Structure is Worth the Price

**arXiv ID:** 2608.27506 | [PDF](https://arxiv.org/pdf/2608.27506v1)

**作者:** Thomas Nolasque `[一作]` (Royal Bank of Canada), Ankit Vani `[通讯]` (Royal Bank of Canada)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在不同的 token 预算下，比较两种语言模型推理架构——单一调用的 monolith 与带有规划、检索、验证与修复的 verified search，探讨额外推理结构是否仅在预算达到阈值时才有益，并通过交叉验证实验确认阈值。

**💡 创新点**

首次量化并验证了推理结构的“临界阈值”效应：在约 1500 output‑equivalent tokens 之下，额外的规划与验证会浪费资源，导致性能下降；超过阈值后则显著提升准确率，并形成可持续的性能提升。

**🔧 技术方法**

使用 GPT‑5.4‑mini 大模型、基于 BM25 的确定性检索器、规划器、无标签检查器（label‑blind checker）以及修复机制；采用 output‑equivalent token 预算控制、交叉验证统计（McNemar 测试）以及严格的交叉验证法。

**📊 数据集**

FinQA 与 TAT‑QA 两个金融推理数据集，涵盖表格、文本与多步算术推理。

**📈 对比分析**

在 14 个 token 预算层级（250–42,000）下对 monolith 与 verified search 进行准确率比较：monolith 在 1,000 tokens 下 18% 但 verified search 近 0%；从 1,500 tokens 起 verified search 超越 monolith，最高分别达到约 44% 与 40%；两组差值的 McNemar 检验在低端与高端均显著，证实了跨越点。

**⚠️ 局限性**

实验仅使用单一模型（GPT‑5.4‑mini），可能无法推广到其他模型；未对提示、检索方式或修复策略做 ablation；未考虑实时成本、延迟或混合/自适应系统；使用精确匹配评分，缺乏部分分数；数据仅限金融推理，难以说明跨域泛化；缺少人工基准与难度分层分析。

---

## 112. A Controlled Audit of Architectural Complexity in Uncertainty-Aware Multi-Organ Ultrasound Classification

**arXiv ID:** 2608.28063 | [PDF](https://arxiv.org/pdf/2608.28063v1)

**作者:** Yang Song `[一作]` (Hong Kong Polytechnic University), Ziran Wang `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文对多器官超声图像分类中的复杂网络架构进行系统审计，评估其在任务性能、校准与分布漂移下的实际价值，最终为部署选择提供依据。

**💡 创新点**

创新点在于提出了多层级的“复杂度审计框架”，结合固定检查点、模块删除、容量匹配再训练、对称温度校准和配对种子统计，系统量化每个模块对最终性能的贡献。

**🔧 技术方法**

主要技术包括：跨域注意力（NALA）、稀疏专家路由、Evidential Deep Learning（EDL）目标、温度标定、AURC风险覆盖评估以及基于最大软化概率的OOD AUROC。

**📊 数据集**

使用了两套公开图像级多器官超声数据集：Dataset 1（16520张，乳腺、肾、肝纤维化）和Dataset 2（4403张，乳腺、肾、卵巢、甲状腺），并引入不同探头作为OOD测试。

**📈 对比分析**

通过十个匹配种子、容量匹配、对称校准和配对t检验进行比较。实验表明，虽然Full‑EDL在宏F1上未显示显著优势，简单的CE+温度标定模型在校准和风险覆盖上与Full‑EDL持平或更好，但在Dataset 2的OOV阈值上被拒绝。

**⚠️ 局限性**

主要局限包括：仅在图像级别划分（缺乏病人/研究级别独立性）、OOD评估仅覆盖两种探头、缺乏跨域微调与校准、以及单一移动端轻量化骨干网络的局限性。

---

## 113. Rating the Raters: Rasch Measurement Theory for LLM Evaluation

**arXiv ID:** 2608.27463 | [PDF](https://arxiv.org/pdf/2608.27463v1)

**作者:** Pratik S. Sachdeva `[一作]` (University of California, Berkeley), Nathan Boudol `[通讯]` (Grenoble INP)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了将Rasch测量理论（RMT）应用于LLM作为评估者（rater）在检测仇恨言论时的表现

**💡 创新点**

首次将RMT框架用于LLM评估，揭示LLM在严重性、项目校准、问题顺序、目标身份敏感度和评分尺度使用上的系统差异，并验证人类构念在LLM中的适用性

**🔧 技术方法**

Rasch多因素模型（Many-Facet Rasch Model）、差异评估者功能（DRF）、尺度链接与阈值诊断

**📊 数据集**

Measuring Hate Speech（MHS）语料库，包含50,070条社交媒体评论、11,143名人类标注者及10个仇恨言论维度

**📈 对比分析**

与人类标注基准对齐，估计LLM严重性参数；通过交互项诊断项目偏差；使用DRF检验问题顺序和目标身份偏差；构建LLM独立尺度并与人类尺度比较，发现总体秩序高度一致（Spearman ρ ≈ 0.91），但在高端项目上LLM的阈值显著提高

**⚠️ 局限性**

仅在仇恨言论单一任务和英文语料下验证，未展示在其他领域或多语言环境中的可推广性；模型评估受单一样本噪声和提示设计的影响；未对LLM评估者与人类替代的实际影响进行深入探讨

---

## 114. Iron: Intent-Aligned and Retrospective Dual Learning Framework for Enhancing Generalist Virtual Agents

**arXiv ID:** 2608.27866 | [PDF](https://arxiv.org/pdf/2608.27866v1)

**作者:** Jiahe Ying `[一作]` (Fudan University), Siliang Tang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了名为 Iron 的自监督学习框架，用来训练跨平台的 GUI 代理，使其能够在多种数字环境中自动完成任务。

**💡 创新点**

创新点主要包括：①双向学习（Instruction Grounding 与 Intent Understanding）的闭环协同；②引入逐步循环一致性（SCC）奖励，提供细粒度的动作-意图对齐信号；③后视重建机制将失败轨迹转化为成功样本；④使用五维轨迹过滤器确保自训练数据质量。

**🔧 技术方法**

核心技术包括：多模态大语言模型（MLLM）、蒙特卡洛树搜索（MCTS）、双向学习策略、SCC 奖励函数、自我回溯重建、五维轨迹筛选、以及基于自提示的意图推理。

**📊 数据集**

使用了 Aguvis 训练集（120k grounding + 90k planning），VisualWebArena、OSWorld、AndroidWorld 作为评测基准；并在 Qwen3-VL-8B、OS-Atlas-Base-7B、InternVL2.5-4B 等多种后端模型上验证。

**📈 对比分析**

通过与 Full 训练（3 倍 SFT 数据）以及 Step-DPO、GPT‑4 系列模型在 OSWorld、AndroidWorld、VisualWebArena 上对比，Iron 在 OSWorld/AndroidWorld 的成功率分别提升至 41.96%/39.15%，在未见的 VisualWebArena 上提升约 25% 以上，显著优于所有基线。

**⚠️ 局限性**

局限性：仍需大量自生成数据，后视重建可能引入不准确的意图标签；对资源和算力的需求较高；在极端长周期或高度动态环境中，SCC 奖励与轨迹过滤的鲁棒性尚待进一步验证。

---

## 115. FVeinSyn: Synthetic Finger Vein Image Generator

**arXiv ID:** 2608.27527 | [PDF](https://arxiv.org/pdf/2608.27527v1)

**作者:** Yifan Wang `[一作]` (Southeast University), Alex Kot `[通讯]` (Shenzhen MSU-BIT University)

**通讯引用:** 18120 | [OpenAlex ID](https://openalex.org/A5080977911)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 FVeinSyn 框架，用于大规模可控合成指静脉图像；

**💡 创新点**

创新点在于将血管拓扑生成与近红外渲染解耦，使用基于 L‑system 的血管生成、区域感知分层 GAN 以及可控的内类多样性生成；

**🔧 技术方法**

采用了 L‑system 语法、区域感知分层 GAN、生成对抗网络、光学散射模型与随机几何/光学扰动等技术；

**📊 数据集**

使用公开的 8 个指静脉数据集（SDUMLA‑FV、UTFVP、MMCBNU_6000、PLUS‑FV3、FV‑USM、HKPU‑FV、SCUT‑FV3、THU‑FV3）以及自行生成的 500k 图像；

**📈 对比分析**

通过图像质量指标、主观真实度评估以及开集、跨域、样本/身份受限识别等实验，对比基线与现有方法，模型平均提升约 27% 并取得最高 TAR/EER；

**⚠️ 局限性**

仍存在与真实数据的域差距，生成的光学细节不够逼真，且在极少样本训练时对生成模型依赖较高，未能完全消除人口偏倚。

---

## 116. Moirae: A Multimodal Agent Collaborative Framework for Dynamic Android Malware Detection

**arXiv ID:** 2608.27994 | [PDF](https://arxiv.org/pdf/2608.27994v1)

**作者:** Xueying Zeng `[一作]` (Beihang University), Bo Li `[通讯]` (Beihang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个基于多模态LLM协同代理的Android恶意软件检测框架，动态收集可视化、UI交互和系统API三维运行时证据并进行协同推理。

**💡 创新点**

通过跨模态因果证据融合与ReAct多代理协同推理，利用LLM的语义理解和零样本推理，实现对概念漂移的鲁棒检测，并提供高解释性的证据链。

**🔧 技术方法**

采用动态沙箱驱动、截图、XML UI树、ART方法调用跟踪、时间窗口对齐、白名单API过滤，并以ReAct模式部署多代理LLM（GLM、MiniMax等）进行推理。

**📊 数据集**

使用AndroZoo（2011-2021）构建训练集，CICMalDroid 2020和CIC-AndMal2017作为未见测试集，结合VirusTotal标签进行样本标注。

**📈 对比分析**

在跨数据集与时间漂移评估中与MalScan、DroidEvolver、MaMaDroid、CL-Malware等SOTA模型对比，零样本下准确率达90.06%，在未见数据上平均ACC 88%/F1 88%以上，显著优于基线。

**⚠️ 局限性**

受动态分析触发不完整、部分环境依赖行为捕获不足导致召回下降；LLM推理成本与token消耗较高；对条件性恶意行为的检测仍需进一步提升。

---

## 117. If Agents Were Angels, No Governance Would Be Necessary: Out-of-Band Policy Enforcement at a Trusted Tool Boundary

**arXiv ID:** 2608.27646 | [PDF](https://arxiv.org/pdf/2608.27646v1)

**作者:** Marc Millstone `[一作]`, Marat Pekker `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种 Out-of-Band Policy Enforcement（OBPE）机制，在大语言模型（LLM）代理与后端工具之间设置受信任的边界，实现在代理执行前后对请求与响应进行授权、裁剪、掩码和语义门控，以防止代理在借用凭证后越权访问或泄露敏感数据。

**💡 创新点**

创新点包括：
- 两层策略模型——数据所有者最高授权与代理专属限制，保证代理无法扩大权限；
- 通过有序组合（gate、request、response）和可验证的单调性，确保策略组合结果不受规则顺序影响；
- 将策略与执行解耦到代理与后端之间的完整管道，形成可审计的“工具交换”协议；
- 在模型与后端之间插入可控的裁剪、掩码、写入限制等操作，提升对“数据→指令”注入与信息泄露的防护。

**🔧 技术方法**

技术实现：Cedar 权限语言（typed authorization）、HTTP 代理与 Connector 语义映射、请求/响应类型化、字段级裁剪与掩码、语义门控（defer/deny/permit）、可观测的审计与日志；实验平台使用多模型（Claude Sonnet、Claude Haiku、GPT‑5、GPT‑5 mini）与自建的 Jira/ServiceNow Mock 接口。

**📊 数据集**

实验数据集：71 个任务、35 个场景集（含响应自适应的 20 个红队任务），模拟的 Jira 与 ServiceNow 交互；多模型（Claude Sonnet 4.6、Claude Haiku 4.5、GPT‑5、GPT‑5 mini）用于跨模型评估。

**📈 对比分析**

比较方法：对比原始代理、仅提示规则、Prompt+OBPE、不同门控/裁剪组合的实验；采用可重复实验、聚类加权差异、Bootstrap 置信区间。性能结果显示：
- OBPE 将确定性追踪失败率从 57.6% 降至 0.2%（≈41.2pp 降低，CI 27.7–54.9）；
- 安全有效完成率提升约 22pp；
- 仅提示规则对最终答案有一定改善，但未阻止数据泄露；
- 在多模型上表现一致，安全收益最大。

**⚠️ 局限性**

局限性：
- 仅对已知字段/结构化接口生效，无法覆盖开放式或非结构化数据；
- 无法阻止基于计数/统计的重建攻击（如行数oracle）；
- 写入裁剪需依赖准确的无效性声明，错误声明会导致误拒或误授权；
- 未实现持久化批准、时间/聚合门控等高级特性；
- 在真实生产环境中的延迟与可扩展性未完整验证；
- 依赖于 Connector 的完整映射，若映射缺失或不一致将导致策略失效。

---

## 118. Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning

**arXiv ID:** 2608.27549 | [PDF](https://arxiv.org/pdf/2608.27549v1)

**作者:** Hanyang Wang `[一作]`, Jialong Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将物理世界表征为可执行代码，提出了 Code-as-World 范式，并设计了基于代理的 propose–instantiate–execute–render–verify 循环，从多模态证据（文本或视频）中构建并验证可执行世界，随后将这些可验证的可执行世界用于为视觉‑语言模型提供可扩展的定量物理推理监督；

**💡 创新点**

主要创新包括①将物理世界以可执行代码形式结构化；②将代理发现循环嵌入世界表征流程，实现迭代式假设生成与验证；③利用可验证的可执行世界作为大规模物理监督，显著提升定量物理推理性能；

**🔧 技术方法**

技术上结合了代码化物理世界、物理仿真接口、LLM 驱动的代理、深度学习 VLM、GRPO 优化、视频生成模型以及 SAM、VGGT‑Omega、SAM3D 等多模态处理工具；

**📊 数据集**

使用的数据集包括：文本驱动描述（LLM 生成+人工审核）、视频驱动采样自 WISA‑80K、像素级监督来自 RefCOCO/RefCOCO+/RefCOCOg/RefCLEF/GOT‑10K，以及定量物理推理评估集 QuantiPhy 验证集；

**📈 对比分析**

实验与多种公开模型（Gemini‑3.1‑Flash、ChatGPT‑5.1、Qwen3‑VL‑32B‑Instruct 等）以及专有模型在 QuantiPhy 的 2S/2D/3S/3D 子集和 MRA 指标上进行比较，Code‑as‑World‑VL‑9B 及其 27B 变体在平均 MRA 上分别达到 55.4% 与 58.6%，在直接答案任务中超过所有基线，甚至击败更大参数量模型；

**⚠️ 局限性**

局限性主要包括①仅覆盖刚体动力学，未扩展到流体、布料、燃烧等复杂物理；②仿真与真实世界的差距可能导致局部合理但机制不准确；③模型未内部化发现循环，只学习到可验证结果；④QuantiPhy 评估范围有限，未涵盖更广泛的物理场景。

---

## 119. Fine Difference Structure and Prime-Power Depth of Bent Partitions

**arXiv ID:** 2608.28133 | [PDF](https://arxiv.org/pdf/2608.28133v1)

**作者:** Zhaorui Wu `[一作]` `[通讯]` (University of Oxford), Zhaorui Wu (University of Oxford)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了任何 p-ary bent 分区的深度必为 p 的幂，并给出了其细化分区是 PDF 和 ZDB 的性质。

**💡 创新点**

突破了先前对常规性或细胞对称性假设的依赖，利用平衡融合引理给出无条件的深度判定。

**🔧 技术方法**

采用有限平均、平衡融合引理、差分集与零差分平衡映射的理论工具，并在 Lean 4 中完成形式化验证。

**📊 数据集**

无，论文为纯理论证明，不涉及实验数据集。

**📈 对比分析**

无实验比较，论文不涉及性能指标。

**⚠️ 局限性**

仅能恢复零差分计数，无法完全反演所有导数值，对普通的 bent 分区不直接适用。

---

## 120. Cambria: Resource Abstraction for Parametrized Algebraic Effects and Handlers

**arXiv ID:** 2608.27798 | [PDF](https://arxiv.org/pdf/2608.27798v1)

**作者:** Jack Liell-Cock `[一作]` (University of Oxford), Sam Staton `[通讯]` (University of Oxford)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种将参数化抽象与代数效应与处理器结合的核心语言，支持资源分配的抽象化与类型安全；

**💡 创新点**

通过参数化处理器实现资源抽象化的parametricity保证，使得客户端代码无法查看底层资源实现；

**🔧 技术方法**

使用步骤索引逻辑关系、PAT（参数化代数理论）与存在类型拆解的形式化证明，构建类型推理与抽象性证明；

**📊 数据集**

论文主要是理论性，没有使用特定数据集；

**📈 对比分析**

通过实现示例（局部状态、Pólya urn、并发线程管理）演示可行性，但未给出性能对比；

**⚠️ 局限性**

局限在于无法进行可达性/垃圾回收的抽象分析、缺乏概率语义与对可变资源的更细粒度控制；

---

## 121. LitCurate: A Configuration-Driven AI-Assisted Framework for Scientific Database Construction with an Application to Lower-Mantle Equation-of-State Data

**arXiv ID:** 2608.27629 | [PDF](https://arxiv.org/pdf/2608.27629v1)

**作者:** Abin Shakya `[一作]` (Columbia University), Renata M. Wentzcovitch `[通讯]` (Columbia University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个名为 LitCurate 的开源端到端框架，能够从文献检索到手工审阅、PDF 转换、LLM 结构化抽取，最终生成带完整可追溯性的科学数据库；并以此构建了一个包含 1,334 条记录、来自 205 篇 1990–2025 年低地幔及相关高压矿物 EOS 参数的数据库。

**💡 创新点**

创新点在于：①以可审计、分阶段的流水线方式将文献发现、筛选和抽取集成；②通过 YAML 配置实现领域知识的外部化，使同一核心流水线可复用；③在抽取阶段引入模式化的 schema‑prompt 组合，并对提取结果做 JSON Schema 校验；④提供完整的中间产物和运行账本，支持任务恢复和可追溯性。

**🔧 技术方法**

主要技术包括：大语言模型（Claude Fable 5、Gemini 3.1 Pro Preview、GPT‑5.6 Sol High、Qwen3.6 27B 等）用于自然语言检索、抽取；OpenAlex、Unpaywall 进行文献检索与 PDF 获取；Marker 进行 PDF→Markdown 转换；Python、SQLite、YAML 配置及自定义脚本构成整体流水线。

**📊 数据集**

使用的核心数据集为 205 篇 1990–2025 年的低地幔矿物 EOS 相关论文（共 1,334 条记录），以及通过 50 篇已知包含 EOS 参数的开放获取论文构成的基准集，进一步利用人工标注的 220 条真值记录对抽取质量进行评估。

**📈 对比分析**

在 50 篇基准论文上，抽取器的表现：记录级检索率 89.1–95.5%，不匹配记录率 7.3–23.6%，字段级精度 91.7–99.4%，F1 94.0–98.2，平均 API 成本 0.10–0.32 USD/论文（本研究生产模型 Claude Fable 5 的成本约 0.32 USD/论文）。与手工标注的真值集对照，模型在 V₀、K₀、K′ 等数值字段上表现极佳，EOS 模型字段准确度相对较低。

**⚠️ 局限性**

局限性包括：①仅能处理文本和表格，无法提取图形中的数值；②自动 PDF 下载受 OpenAlex/Unpaywall 的可访问性限制，需手工补充；③后期单位转换与语义标准化未集成到流水线，需外部处理；④抽取质量受 LLM 版本和提示设计影响，存在不一致性。

---

## 122. HubMixer: Progressive Latent Hub Mixing for Parameter-Efficient Feature Interaction in Recommendation

**arXiv ID:** 2608.27991 | [PDF](https://arxiv.org/pdf/2608.27991v1)

**作者:** Jie Zhou `[一作]` (Kuaishou Technology), Peng Jiang `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种基于潜在枢纽的进阶混合架构，利用少量可学习的枢纽通过引导–交互–读取流程实现参数高效的特征交互。

**💡 创新点**

创新点在于：①引入潜在枢纽作为交互中心，使用跨注意力将异质特征聚合到枢纽；②在枢纽空间进行高阶交互；③通过token‑conditioned readout将交互语义写回原token，既保持字段身份又实现了高效的交互。

**🔧 技术方法**

使用了Transformer跨注意力、RMSNorm、LayerScale残差、Hub Induction、Hub Interaction、Token‑Conditioned Readout、MIMIC‑Mixer式token‑mixing以及多任务学习框架。

**📊 数据集**

在Kuaishou短视频招聘业务的工业数据集上进行实验，约10亿样本，包含用户画像、行为序列、职位、短视频、上下文及统计特征，四个多任务目标（点击、有效消费、交互、简历提交）。

**📈 对比分析**

与传统特征交叉模型（DCN、DCNv2、AutoInt、Wukong）及token‑mixing模型（RankMixer、TokenMixer）在相同数据集上做AUC对比；自研模型在四个目标上平均AUC提升约0.005，参数量更少；在线A/B测试提升简历提交转化率5.48%，已全面部署。

**⚠️ 局限性**

局限性包括：枢纽是共享静态可学习参数，未能根据请求动态生成；需要调节枢纽数量与块深度的超参；对极端稀疏特征的捕捉可能有限；仅在短视频招聘场景验证，其他业务场景的通用性需进一步验证；未探究更轻量化的枢纽交互方式。

---

## 123. Training-Free Temporal Abstraction for General Video Understanding

**arXiv ID:** 2608.27929 | [PDF](https://arxiv.org/pdf/2608.27929v1)

**作者:** Etienne Casanova `[一作]` (California Institute of Technology), Pietro Perona `[通讯]` (California Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种训练自由的 STITCH 方法，利用冻结的视频‑文本模型生成的短窗口嵌入序列，通过变化点检测将视频划分为可复用的语义时间块，并在事件边界检测、语言驱动的时刻检索以及长视频 VLM 的帧选择等多任务中直接复用该时间抽象；

**💡 创新点**

创新点在于将冻结的跨模态嵌入视作一条时间信号，用无监督的核变化点检测一次性得到任务无关的时间块，随后可被不同下游任务以轻量级的方式利用，实现了“先抽象后复用”的通用视频理解框架；

**🔧 技术方法**

技术细节包括：InternVideo2 冻结编码器提取 4 帧窗口的嵌入；余弦核变化点检测（亦可使用 PELT、Std 阈值等变体）得到切点；在检索任务中使用最大相似度评分；在 VLM 任务中采用 Greedy/MMR 采样；所有步骤均无任务特定训练；

**📊 数据集**

实验使用的公开数据集：Kinetics‑GEBD、TAPOS（事件边界）；ActivityNet Captions、QVHighlights（时刻检索）；MLVU、LongVideoBench、VideoMME（长视频 VLM QA）；

**📈 对比分析**

与监督、无监督、零样本及其它训练自由基线对比：在事件检测上 STITCH 取得 83.9% Avg F1，接近监督模型；在时刻检索上 64.6% R@1@0.5，超过多种监督检索模型；在 VLM QA 上相较均匀采样提升约 8–9% 的准确率；整体表现保持竞争力，尤其在低帧预算场景中突出；

**⚠️ 局限性**

局限性包括：受冻结嵌入空间限制，难以捕捉细粒度子动作边界；单尺度时间块结构忽略层次化组织；对运动噪声敏感导致过度分割；缺乏在线流式处理，无法实时更新；未来可尝试层次化分块、强化后端模型或动态采样策略。

---

## 124. LongGuard: Mechanistic Analysis and Training-Free Mitigation of Long-Context Failure in Safety Guardrails

**arXiv ID:** 2608.27580 | [PDF](https://arxiv.org/pdf/2608.27580v1)

**作者:** Ziyang Chen `[一作]` (Chinese Academy of Sciences), Songlin Hu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 7561 | [OpenAlex ID](https://openalex.org/A5102820325)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建LongGuard框架，对LLM安全防护在长文本中的失效进行评估、机制分析并提出无训练的缓解方法。

**💡 创新点**

首次系统化研究长文本导致的安全防护稀释效应，揭示稀释机制并提出基于注意力头修正与分块检测的无训练对策。

**🔧 技术方法**

Safety Needle-in-a-Haystack任务、三层注意力→logit→行为机制剖析、注意力头选择性分析、Chunked Detection与Attention-Head Sharpening、上下文感知超参数路由。

**📊 数据集**

17个公开安全基准合成的针样本，英文维基百科、对话、代码与中文维基的填充文本，构成多域长文本测试。

**📈 对比分析**

在15个主流guardrail上对比原始、CD、AHS和CAHR，平均提升unsafe recall 21.8%（CD）与12.6%（AHS），并在多领域/语言、攻击测试中保持显著改进。

**⚠️ 局限性**

实验主要基于合成样本，缺乏真实交互数据；提出的缓解仅为推理时方法，未涉及模型微调或更复杂长文本场景。

---

## 125. CrabOS: An Operating System for Human-AI Co-inhabitation

**arXiv ID:** 2608.28165 | [PDF](https://arxiv.org/pdf/2608.28165v1)

**作者:** Qi Yang `[一作]` (Peking University), Yun Ma `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Human‑AI Co‑inhabitation概念，构建了CrabOS操作系统，使人类与AI能够在同一工作环境中无缝切换并共享工作状态；

**💡 创新点**

核心创新在于：①将所有工作数据文本化为自然语言可读的结构化文本对象；②让人类与AI直接操作同一套可引用的工作对象；③通过统一的可审计接口消除AI专用工具包装层，提升跨环境协作效率；

**🔧 技术方法**

采用文本对象持久化（文件系统+JSON）、统一的Kernel Interface（五阶段审核）、L0–L3多层架构、Electron+Web技术实现跨平台UI、OpenAI模型与ONNX嵌入引擎、网络服务与Agent Runtime等；

**📊 数据集**

主要使用本地嵌入模型（如BERT‑ONNX）及Transformer.js进行向量索引；未引入公开数据集，系统通过内部生成的文本对象和用户交互数据进行验证；

**📈 对比分析**

与现有代理系统（OpenClaw、Hermes、AgentOS等）在记忆管理、跨应用上下文切换、多代理任务编排等案例对比，展示CrabOS无需桥接即可实现功能；性能方面，系统在文本读写、任务调度和权限审核上保持低延迟（≈数百毫秒），并通过批量写入缓解高频写入；

**⚠️ 局限性**

局限性包括：①不原生支持Bash等外部命令，需要通过SSH等方式调用；②原生应用需迁移为L3 App才能共享工作对象，迁移成本仍存在；③文本对象的写入会导致比内存数据库更高的IO延迟；④安全策略需在Kernel层细化，尚未覆盖所有细粒度场景。

---

## 126. From Uncertainty to Clinical Risk: Severity-Aware Conformal Planning for Interactive Medical Diagnosis

**arXiv ID:** 2608.27847 | [PDF](https://arxiv.org/pdf/2608.27847v1)

**作者:** Yue Zhou `[一作]` (Lanzhou University), Hanwen Du `[通讯]` (Ohio State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种严重性感知的符合式临床规划框架，用于在信息不完整时进行交互式医学诊断，结合诊断、保护与缺失证据的三种信念，利用校准的临床风险指导Ask/Commit决策，并通过MCTS进行长期规划。

**💡 创新点**

创新点在于把诊断不确定性转化为基于严重性加权的符合式临床风险作为决策信号，首次将风险校准与多步规划结合，实现Ask与Commit在同一规划框架内竞争；同时引入三重信念模型和缺失证据校正，提高诊断安全性和效率。

**🔧 技术方法**

采用符合式预测（Conformal Prediction）对诊断分布进行校准并引入严重性权重；构建诊断、保护和缺失证据三重信念模型；使用蒙特卡洛树搜索（MCTS）进行多步规划；LLM作为语言基础执行Ask/Commit动作。

**📊 数据集**

在DDXPlus（约130万病例，49种病理）和MediQ（615例，四个答案选项）两个公开交互式诊断基准上进行评估。

**📈 对比分析**

与ATD、UoT、MISQ-HF、C-IP、ACTMED、BED-LLM、随机、熵贪婪等多种基线比较，实验显示：在DDXPlus上Top‑1准确率提升至97–99%，平均提问数减少约40%，差异诊断质量和严重病例错误率大幅下降；在MediQ上准确率提升至66.18%，与全信息基线仅差1.3个百分点，平均提问数下降33%。

**⚠️ 局限性**

局限性包括仅在模拟基准上评估，缺乏真实临床验证；仅测试英文学科、有限LLM后端；严重性标签预设，未覆盖患者个体差异；符合式校准仅在固定轮次上保证误报率，未保证任意停止时间下的覆盖；未评估与临床工作流程的集成、对抗鲁棒性或公平性。

---

## 127. Low-Altitude Fluid Antenna Network with Multi-Agent Reinforcement Learning

**arXiv ID:** 2608.27909 | [PDF](https://arxiv.org/pdf/2608.27909v1)

**作者:** Tong Zhang `[一作]` (Harbin Institute of Technology), Huseyin Arslan `[通讯]` (Istanbul Medipol University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一种低空流体天线网络（FA网络），通过将电磁数字孪生（EM-DT）与多智能体强化学习（MARL）结合，实现了对天线位置和波束的快速重构，从而提升网络的下行总速率。

**💡 创新点**

创新点在于：①提出了统一视角下的低空FA网络架构；②构建了高保真EM‑DT训练环境以解决真实环境采样成本高的问题；③利用离线/异策略MARL（MATD3）在EM‑DT中训练并通过两阶段迁移学习实现sim‑to‑real，首次在低空高动态场景中实现毫秒级天线重构与波束调度；④通过联合优化FA位置和波束获得显著的速率提升。

**🔧 技术方法**

主要技术包括：电磁数字孪生建模（3D射线追踪、移动性与干扰耦合）、多智能体强化学习（MATD3、CTDE范式）、离线/异策略学习与域随机化、两阶段迁移学习框架。

**📊 数据集**

数据集：在EM‑DT中生成的仿真数据（包含多种移动轨迹、不同高度、随机阻塞与干扰），以及在阶段‑II中预收集的真实系统高质量数据（用于微调）。

**📈 对比分析**

与固定天线位置基线进行对比，实验表明联合优化FA位置与波束可使网络总速率提升118.5%。训练曲线显示在训练阶段即已优于基线，测试阶段在未见过的环境下也保持相同优势。

**⚠️ 局限性**

局限性包括：①天线重构硬件延迟与功率限制可能削弱理论优势；②对实时CSI的高依赖导致频繁上行训练与反馈开销；③MARL策略缺乏可解释性与安全保证；④EM‑DT在动态环境中的精度与实时校准挑战；⑤不同时间尺度的决策同步问题；⑥缺乏大规模真实部署验证。

---

## 128. GOD: Govern, Observe, and Direct - A Real-Time Control Room for Agent Societies

**arXiv ID:** 2608.27992 | [PDF](https://arxiv.org/pdf/2608.27992v1)

**作者:** Yige Luo `[一作]` (Huawei), Ran Guan `[通讯]` (Huawei)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个本地控制室GOD，集成生成式代理模拟器与运行时，支持实时命令、可视化回放与可移植实验/地图/代理包。

**💡 创新点**

提供了统一的命令记录与回放同步、分离实验与回放的包合同、以及浏览器端与后端的命令‑事件循环。

**🔧 技术方法**

技术栈包括React/Vite前端、FastAPI后端、AgentSociety模拟器、JiuwenClaw代理运行时、Qwen‑Plus语言模型以及SQLite回放存储。

**📊 数据集**

使用PKU地图、22个代理配置和多模板场景（共15条完整跑），以及Qwen‑Plus模型。

**📈 对比分析**

通过15条完整跑评估命令路径，得到14/14命令路由成功、78/84目标定位成功、169/182回答与回放匹配、平均JSD 0.011，说明系统能够准确记录并复现代理行为。

**⚠️ 局限性**

仅测试单一模型配置与合成配置，未评估多模型或真实人类数据；只做字符串一致性检查，未检验语义正确性或用户体验。

---

## 129. The Power of Local Marginals: An $O(\varepsilon^{-1})$-Aspect-Ratio Reduction for Dynamic Weighted Matching

**arXiv ID:** 2608.27805 | [PDF](https://arxiv.org/pdf/2608.27805v1)

**作者:** Jiale Chen `[一作]` `[通讯]` (Stanford University), Jiale Chen (Stanford University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种通用的动态最大权匹配（MWM）算法框架，能够在边插入和删除的情况下维护(1±ε)近似的匹配权值或(1-ε)近似的显式匹配，并将任意多重比（aspect ratio）从多项式下降到 O(1/ε)，实现了在稠密图和低树枝度图中的快速更新。

**💡 创新点**

创新点主要包括：①利用局部边权的“边缘贡献”结构，将全局权值拆解为若干 O(1/ε) 规模窗口的局部贡献；②给出新的价值合成与匹配合成引理，显著降低了局部窗口的权值比从 O(ε^-2) 降到 O(1/ε)；③将该结构应用于动态权值降维和显式匹配的低损失稀疏化，实现了在一般图中 O(ε^-3) 的更新开销；④在二分图中结合图展开技术实现了权值到基数的无损转换。

**🔧 技术方法**

核心技术包括：局部边缘贡献（local marginals）理论、局部交换引理、价值合成与匹配合成的重叠窗口分析、图展开（unfolding）和稀疏化（sparsifier）构造；以及多层级的懒更新（lazy composition）策略来控制递归和回溯。

**📊 数据集**

论文未使用实验数据集，全部为理论分析与算法设计，结果以渐进式时间复杂度和近似误差给出。

**📈 对比分析**

与现有工作比较：在增量/全动态二分图中将 O(n^-9+m^-8) 的增量时间提升到 O(n^-8+m^-7)；在最大度 Δ 图中从 O(Δ^-5) 提升到 O(Δ^-3)；在 α-树枝度图中首次给出 O(α^-4) 的完全动态显式 MWM；在一般图中将 O(√m^-6) 的更新时间提升到 O(√m^-3)。总体上，本文的框架在所有已知基线上都实现了多项式或更低的时间改进。

**⚠️ 局限性**

限制与开放问题：①需要图的子图封闭性和权值范围已知；②对极端多重比的情形仍需进一步研究；③在非二分图中权值到基数的无损转换仍依赖于展开，导致额外的空间和时间；④所有结果均基于随机化/自适应对手模型，是否能在完全确定性和无自适应对手下保持同等性能仍是未来工作。

---

## 130. A Note on Approximating the Rural Postman Problem below 3/2

**arXiv ID:** 2608.27607 | [PDF](https://arxiv.org/pdf/2608.27607v1)

**作者:** Hong Li `[一作]` `[通讯]` (Yunnan University), Hong Li (Yunnan University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种随机多项式时间（3/2−ϵ）逼近算法（其中ϵ>10⁻³⁶）用于乡村邮递员问题（RPP），并给出了可通过条件期望法实现的确定性版本；

**💡 创新点**

创新点在于将最大熵分布抽样技术从度量旅行商问题迁移到RPP，利用该技术在奇偶匹配上取得严格低于3/2的逼近，并进一步证明了RPP与度量TSP之间的逼近传递关系；

**🔧 技术方法**

主要技术包括预处理将RPP转化为完整图、构造并求解LP松弛、利用最大熵分布抽样得到随机生成的生成树、对奇偶点进行最小匹配校正以及使用条件期望法实现脱随机化；

**📊 数据集**

论文未使用实验数据集，全部以理论分析与证明为主；

**📈 对比分析**

相较于以往最好的3/2逼近，本文取得了更小的常数ϵ改进；同时证明任何α逼近的度量TSP算法在固定ε下可得到(α+ε)逼近RPP，展示了理论上的性能提升；

**⚠️ 局限性**

局限性包括常数ϵ极其小（10⁻³⁶），实际数值难以实现；算法依赖预处理和完整图构造，对大规模实例可能开销较大；并且对非度量成本仅保证LP相对的逼近，未给出针对一般非度量RPP的多项式时间保证。

---

## 131. Automated Analysis Framework for Multilingual Climate-Health Literature Based on Multi-Agent Large Language Model

**arXiv ID:** 2608.27998 | [PDF](https://arxiv.org/pdf/2608.27998v1)

**作者:** Yuze Sun `[一作]` (Tsinghua University), Xiaomeng Huang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一套多代理大型语言模型框架，用于自动化分析双语气候-健康科研文献，实现文献筛选、结构化信息抽取与标准化整合。

**💡 创新点**

创新点包括：将文献分析拆解为文档评估、信息抽取、分析评审三大专用代理；四层幻觉控制与一致性验证机制；跨语言标准化词表与外部知识库的融合；在真实规模双语语料上实现高F1并生成可视化分析。

**🔧 技术方法**

技术实现基于大语言模型（ChatGPT/DeepSeek等）+多代理协作架构 + Prefect调度；外部知识库（WMO气象库、城市词典、WHO疾病库）作为工具调用；四层幻觉抑制与一致性验证。

**📊 数据集**

使用了32,642篇1993-2023年中国气候-健康领域的中英双语论文（OpenAlex、CNKI、百度学术）以及800篇专家标注的金标准数据。

**📈 对比分析**

与单一LLM、BERT NER、AutoGen、LiRA、Paper Circle等基线对比；核心抽取任务总体F1为0.92，单代理0.83，双代理0.88，三代理0.92；多语言上F1为0.93（英）、0.91（中）；有效提取率98%，标准化率99%。

**⚠️ 局限性**

局限性在于仅处理标题、摘要和关键词，缺乏全文抽取；仍有少量幻觉错误需人工复核；依赖外部LLM接口，处理极大规模时受限；未来需扩展全文、动态更新与可视化图谱。

---

## 132. Personalized and Multi-View Representation for Federated Cold-Start Recommendation

**arXiv ID:** 2608.27826 | [PDF](https://arxiv.org/pdf/2608.27826v1)

**作者:** Jaehyung Lim `[一作]` (Pohang University of Science and Technology), Hwanjo Yu `[通讯]` (Pohang University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种名为 PMFRec 的联邦冷启动推荐框架，能够在双侧约束（服务器持有属性特征，客户端持有交互记录）下生成个性化、多视角的项目表示。

**💡 创新点**

核心创新包括：
1) 通过服务器端的个性化表示生成器为每位用户生成专属的属性-到-嵌入映射；
2) 引入全局多视角编码器、门控机制和正交性/负载平衡正则，实现对异构语义的分离与动态组合；
3) 将协同知识与属性知识融合为单一传输的项目表示，消除显式对齐正则，显著降低通信与训练成本。

**🔧 技术方法**

使用技术：联邦学习框架、个性化嵌入生成器、全局多视角投影器、门控网络、正交性约束、负载平衡正则、局部差分隐私（ε, δ‑LDP）等。

**📊 数据集**

实验数据集包括：CiteULike、XING（XING‑5000、XING‑10000、XING‑20000）四个真实冷启动推荐数据集。

**📈 对比分析**

与 FedMVMF、FedWDR、FedNCF_CS、PFedRec_CS、FedVBPR、FedDCN、IFedNCF、IPFedRec 等多种基线进行对比。PMFRec 在 Recall@K、Precision@K、NDCG@K 上均超越最强基线，Recall@10 约提升 5–7%，并在 LDP 环境下表现出更高的鲁棒性。

**⚠️ 局限性**

局限性：仍受双侧约束下模型表达能力的限制；多视角数量与正则参数需依赖数据调优，可能在不同场景下表现不一；在极低隐私预算（ε 较小）下性能下降；对极大规模用户/项目的可扩展性需进一步验证。

---

## 133. Learning to Transfer Across Modes: Towards Unified Urban Mobility Forecasting

**arXiv ID:** 2608.28273 | [PDF](https://arxiv.org/pdf/2608.28273v1)

**作者:** Yixuan Zhao `[一作]` (University of Exeter), Man Luo `[通讯]` (University of Exeter)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种统一框架TransMod，用于跨多种城市交通模式的需求预测，特别解决不同模式间空间粒度不一致和目标模式缺乏历史数据的问题。

**💡 创新点**

创新点在于：①通过地理约束的柔性映射（soft assignment）将站点级信息聚合到统一的区级空间；②利用对齐（MMD+InfoNCE）将不同模式的区级表示映射到共享潜在空间；③引入记忆池机制，将源模式的时空模式迁移到目标模式，实现无目标时序历史的预测。

**🔧 技术方法**

采用时空卷积网络、图注意力网络、MMD与InfoNCE对齐损失、可学习的soft assignment、记忆池检索以及轻量化提示网络。

**📊 数据集**

使用纽约市和芝加哥的共享单车、地铁和叫车数据，结合POI、天气和道路网络等辅助特征。

**📈 对比分析**

与ARIMA、LSTM、RegionTrans、STAN、TSJT、HimNet、TransGTR、MetaST、LSTM-FT、MMDNet和FusionTN等基线相比，TransMod在MAE、RMSE和MAPE上均取得最低值，在两座城市均表现出显著提升；同时在迁移、稀疏数据和长周期预测下保持鲁棒性。

**⚠️ 局限性**

限制包括：需要手工设定soft assignment的初始核与掩码；记忆池大小和温度等超参对性能影响显著；在极端数据缺失或极大空间粗化时性能下降；模型尚未考虑多源动态更新与连续学习的需求。

---

## 134. Explainable Uncertainty Estimation for Reliable Medical AI

**arXiv ID:** 2608.28052 | [PDF](https://arxiv.org/pdf/2608.28052v1)

**作者:** Li Rong Wang `[一作]` (Nanyang Technological University), Xiuyi Fan `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种将不确定性估计与可解释性（XAI）相结合的算法egRUE，用于医疗决策支持；

**💡 创新点**

创新点在于将Expected Gradients特征重要性权重嵌入到RUE的不确定性估计中，实现特征级别的不确定性解释，并证明其满足实现不变性、敏感性和一致性等属性；

**🔧 技术方法**

使用AutoEncoder架构（编码器、解码器、预测头）、Expected Gradients（EG）梯度归因、RUE框架，构成egRUE与解释模块；

**📊 数据集**

在四个真实医疗数据集上评估：新加坡华人健康研究（肺癌、结肠癌）和两大图像数据集（OCTMNIST、BloodMNIST）；

**📈 对比分析**

与熵、Monte Carlo Dropout、深度集成、贝叶斯神经网络、深度证据分类等基线方法比较，egRUE在相关性、误分类检测、选择性预测、错误鲁棒性以及OOD检测等指标均优于或与最先进方法相当，显著提升可靠性和解释质量；

**⚠️ 局限性**

局限性包括：依赖于AutoEncoder训练质量，对极端异常样本的解释可能受限，且对大规模模型的计算成本仍高于纯熵等简易方法。

---

## 135. GAN-Based Semantic Communication for Image Transmission in IoV

**arXiv ID:** 2608.27989 | [PDF](https://arxiv.org/pdf/2608.27989v1)

**作者:** Ruixing Ren `[一作]`, Xiaoke Sun `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于GAN的语义通信框架，用于车辆间低比特率下的图像传输，采用金字塔注意力网络提取语义标签图并通过语义优先级保护机制实现差异化编码，接收端则利用双阶段粗细分辨率生成器和多尺度判别器实现高质量图像重建。

**💡 创新点**

创新点在于：① 语义优先级保护机制实现安全关键类别的差异化量化与码率分配；② 双阶段粗细分辨率生成器与多尺度判别器的组合提升全局结构与细节同步重建；③ 在判别器中加入时域一致性分支、尺度自适应空间金字塔池化与类别感知卷积，以减少帧间闪烁、提升远近尺度表现；④ 使用联合对抗、特征匹配与感知损失实现语义一致性与视觉逼真度的平衡。

**🔧 技术方法**

使用技术包括：GAN（生成对抗网络）与条件GAN；金字塔注意力网络（PAN）进行语义分割；基于优先级的量化与LDPC/卷积码编码；两阶段粗细分辨率生成器；多尺度判别器；时域一致性约束；空间金字塔池化；类别感知卷积；交叉熵权重损失；特征匹配与感知损失。

**📊 数据集**

在Cityscapes数据集上进行训练与评估，包含2975张训练图像与1525张测试图像。

**📈 对比分析**

与UNet、FCN‑8s、DeepLabV3、SegNet等分割模型比较，PAN在mIoU上提升约3%；与JPEG、JPEG2000、BPG、DSSLIC、HiFiC、CRN等编码器比较，低比特率（0.1–0.3 bpp）下PSNR提升2–3 dB，且在AWGN和Rayleigh信道下的重建鲁棒性更佳，语义分割精度和视觉质量均优于传统方案。

**⚠️ 局限性**

局限性包括：① 语义优先级机制虽然提升关键类别性能，但整体mIoU略有下降；② GAN生成的图像在极高比特率下仍不及传统无损压缩的像素细节；③ 生成与判别网络参数量大，推理时延高，尚未实现实时部署；④ 需在特定场景和数据集上训练，泛化到其他道路或光照条件下需进一步验证。

---

## 136. FinExam-10K: When Retrieval Helps Financial Reasoning?

**arXiv ID:** 2608.28155 | [PDF](https://arxiv.org/pdf/2608.28155v1)

**作者:** Yan Lin `[一作]` (INSAIT, Sofia University 'St. Kliment Ohridski'), Yuxia Wang `[通讯]` (INSAIT, Sofia University 'St. Kliment Ohridski')

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个包含 10,198 道多项选择题的金融专业考试基准 FinExam-10K，覆盖 CFA Ⅰ–Ⅲ 与 FRM I–II 级别，分为公开 5,110 题和保留 5,088 题，并提供完整覆盖轨道与上下文完整推理轨道两种评测模式。

**💡 创新点**

创新点在于：① 首次将 CFA 与 FRM 四个阶段统一至同一基准并保持题目完整性；② 通过 17 规模化模型的冻结预测构造经验难度标签，揭示模型在不同难度、阶段与上下文完整性上的共同失败；③ 对静态检索与图结构检索进行对比，并设计低成本的直接-条件门控策略以在保留正面改进的同时降低错误。

**🔧 技术方法**

采用的技术包括：专家双轮重注释、基于模型预测的经验难度分层、对话式检索 Function-RAG 与 FunctionGraph-RAG、执行验证器、以及基于逻辑回归的门控器。

**📊 数据集**

使用的数据集为 FinExam-10K，其中包含 10,198 道经专家重注释的 CFA/FRM 对应题目，按 5,110 公开与 5,088 隐藏拆分，并提供 7,625 条上下文完整题目子集。

**📈 对比分析**

通过对 17 公开模型的准确率对比，最佳模型 Gemini-3.1-Pro 在完整轨道上达 85.29%；在最难的上下文完整 372 题中，最高准确率为 54.57%；Gate 策略在隐藏集上仅触发 7.9% 的题目，将准确率从 70.83% 提升至 71.23%，提升显著但增幅有限。

**⚠️ 局限性**

局限性包括：① 部分题目缺乏完整上下文导致难以衡量真实推理；② 经验难度标签依赖模型表现，缺乏人类认知校准；③ 门控与检索策略在不同链条（PoT vs CoT）下表现不一致，未能统一最优方案；④ 评测仅考虑静态检索与单轮验证，未覆盖多轮迭代代理与资源成本分析。

---

## 137. Leveraging a Foundation Model for the EEG-Based Diagnosis of Alzheimer's Disease

**arXiv ID:** 2608.27719 | [PDF](https://arxiv.org/pdf/2608.27719v1)

**作者:** Maggie Lin `[一作]` (University of California, San Diego), Tzyy-Ping Jung `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

利用预训练的Large Brain Model（LaBraM）对30通道EEG进行高维表示学习，并结合随机森林进行阿尔茨海默症（AD）与健康对照的二分类；

**💡 创新点**

创新点在于将基础模型的深层潜在嵌入与非线性随机森林相结合，克服传统线性谱特征难以捕捉的非线性神经动力学，并通过可解释性分析验证模型聚焦于临床验证的α/θ节律衰退；

**🔧 技术方法**

技术包括：EEG预处理（滤波、ICA、CAR、归一化）、LaBraM Transformer预训练嵌入、PCA降维、随机森林与SVM（RBF核）分类、交叉验证、遮蔽分析、与临床量表相关性分析；

**📊 数据集**

数据集为206名受试者（控制+AD），共30通道EEG（120秒眼睛开放休息状态），采样500Hz后下采样200Hz，划分为训练/验证/测试并在5折交叉验证中使用；

**📈 对比分析**

与传统Band‑Power+RF和FOOOF+SVM基线对比，LaBraM+RF在5折交叉验证中实现ROC‑AUC 89.36%±3.49%，PR‑AUC 81.45%±4.43%，平衡准确率82.44%±4.34%，显著优于基线（p<0.01）；

**⚠️ 局限性**

局限包括控制组包含主观认知下降（SCD）导致标签噪声、敏感度78.8%低于特异度85.7%、仅来自单一地区、缺乏多设备与跨族群验证，以及尚未针对MCI阶段进行评估。

---

## 138. From Documents to Reasoning: A Validated Synthetic Data Pipeline and Semantic-Aware Fine-Tuning for Financial Numerical Reasoning

**arXiv ID:** 2608.27919 | [PDF](https://arxiv.org/pdf/2608.27919v1)

**作者:** Lokendra Birla `[一作]` (Accenture Labs), Shubhashis Sengupta `[通讯]` (Accenture Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建高质量合成财务问答数据生成管线，提出表达式匹配准确率（EMA）评估指标，并在此基础上用SA‑EMA损失微调小型语言模型，实现数值推理性能提升。

**💡 创新点**

创新点包括：①以算式而非文本匹配评估答案的EMA指标；②多模态表格提取与多步骤验证的合成数据管线；③将EMA与语义相似度结合的SA‑EMA损失，用于领域适配。

**🔧 技术方法**

技术手段：使用Claude Sonnet3.5、GPT‑4 等 LLM 生成算式与验证；DSPy ReAct + Python 解释器执行算式；QLoRA 低秩量化微调；语义相似度基于句子嵌入。

**📊 数据集**

数据集：公开金融 QA 基准 FinQA 与 ConvFinQA，以及自研的高质量合成 QA 数据。

**📈 对比分析**

评价方式：对比 EM 与 EMA，EMA 使模型精度提升 5–10%；在 ConvFinQA 上，微调后的 Mistral‑7B 通过 SA‑EMA 损失实现 79.15% EMA 召回，显著优于传统交叉熵或 EM 评估。

**⚠️ 局限性**

局限性：EMA 与语义相似度非可微，无法端到端学习；损失对运算符可交换性不敏感；对 LLM 生成算式质量高度依赖，若算式错误会影响验证与微调效果。

---

## 139. RiskBlend: A Multi-Signal Framework for Test Input Prioritization in Machine Learning Regression Testing

**arXiv ID:** 2608.27704 | [PDF](https://arxiv.org/pdf/2608.27704v1)

**作者:** Madhusudan Srinivasan `[一作]` (East Carolina University), Namith Nishal Raphae `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了RiskBlend框架，融合四个风险信号对机器学习回归测试输入进行优先级排序。

**💡 创新点**

在不依赖模型特定参数的前提下，通过验证学习的APFD平方权重自动融合历史、预测、决策边界和邻域变化四个跨版本风险信号。

**🔧 技术方法**

多信号融合、APFD优化权重学习、无模型特定调参、距离、概率变化、邻域特征等技术。

**📊 数据集**

Adult、Bank Marketing、Hospital Readmission、Credit Card Default 四个表格数据集。

**📈 对比分析**

与 DeepGini、DATIS-Tabular、MLPrior、随机等基线对比，RiskBlend在所有80种配置上 APFD 最高，平均提升幅度 0.02–0.32。

**⚠️ 局限性**

依赖验证集分布、对稀疏特征或线性模型表现有限、仅评估表格分类器，无法直接推广至图像或文本等非结构化数据。

---

## 140. CURA: Certified Runtime Alarms for Computer-Use Agents

**arXiv ID:** 2608.27808 | [PDF](https://arxiv.org/pdf/2608.27808v1)

**作者:** Divake Kumar `[一作]` (University of Illinois Chicago), Amit Ranjan Trivedi `[通讯]` (University of Illinois Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了计算机使用代理（CUA）在执行桌面任务时的自我报告失效问题，并提出了一种名为CURA的外部运行时行为监测器，能够在任务开始前预估失败风险并以有限样本假警率进行误报控制；

**💡 创新点**

创新点在于：①引入了无内部模型、无额外LLM调用的外部监测机制；②将多种任务无关的行为信号（努力、推理语义、执行生理学、视觉动态）融合成一个CUSUM风险过程；③通过Learn‑then‑Test校准实现假警率的可证书控制；

**🔧 技术方法**

使用了CURA的六个可观测信号、SigLIP图像编码器、先验正则化的逻辑回归、CUSUM统计量和Learn‑then‑Test阈值校准；

**📊 数据集**

实验基于OSWorld‑Verified 361个可执行桌面任务（290成功，71失败），并对比了基于token计数、步数阈值等基线；

**📈 对比分析**

在与token计数和步数触发器在相同假警率下的在线检测中，CURA在α=0.10时召回率为42.3%（中位提前31步），而基线召回率仅为34%；在部署级联中，CURA实现平均得分86.8、全解率84.5%，优于基线82.9、80.3%；

**⚠️ 局限性**

局限性包括：仅有71次失败样本，缺乏对抗性评估；在快速安静失败和信念层错误上表现弱；跨模型迁移时需要重新校准阈值；并且监测信号可能被具有逃避动机的代理抑制。

---

## 141. What Can Low Resource Languages Learn From Each Other?

**arXiv ID:** 2608.27753 | [PDF](https://arxiv.org/pdf/2608.27753v1)

**作者:** Achyuth P `[一作]` (IIT Delhi), Chetan Arora `[通讯]` (IIT Delhi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在极低资源环境下，用预训练模型先专门化为多语言专家，再通过任务算术将这些专家合并成一个多语言基座，随后进行联合微调，构建了一种高效的 OCR 框架（PSMC）。

**💡 创新点**

创新点在于：①发现专家模型低层高度相似，揭示可利用跨脚本共享特征；②提出通过任务算术合并专家权重的“合并”步骤，保留共性与差异；③在合并后进行联合细化，实现参数不变的多语言性能提升。

**🔧 技术方法**

使用的核心技术包括：Parseq Vision‑Transformer 视觉编码器、任务算术（task arithmetic）实现权重合并、Centered Kernel Alignment (CKA) 进行特征相似性分析、Co‑training（多语言联合训练）以及基于 NED/WRR 的评估。

**📊 数据集**

主要数据集：IndicSTR‑Roadside（场景文字）、IndicSTR12（混合场景/印刷文字）、Mozhi（印刷文字）以及 SynthTiger/TRDG 生成的合成语料，实验涵盖10种印地语系脚本（共20+语言）。

**📈 对比分析**

对比方法包括：单语言专门化微调、朴素联合训练以及基准“Skyline”模型。PSMC 在极低资源（<10K 真实图像 + 240K 合成图像）下平均提升约 2% WRR，接近 Skyline 上的全量数据性能，并显著降低验证损失收敛速度。

**⚠️ 局限性**

局限性：①实验仅覆盖10种印地语脚本，未验证对更广泛、不同语系语言的泛化；②依赖多个专家模型和合成数据，合并步骤对参数空间的依赖可能在极大语言集合时变得复杂；③对脚本极为稀缺的真实数据仍存在上限，合并后模型对极端噪声或极少样本的鲁棒性尚未完全评估。

---

## 142. One year in a forest: Analyzing the challenges of autonomous navigation in subarctic environments

**arXiv ID:** 2608.27628 | [PDF](https://arxiv.org/pdf/2608.27628v1)

**作者:** Matěj Boxan `[一作]` (Université Laval), François Pomerleau `[通讯]` (Université Laval)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究通过在亚北极针叶林环境中进行一年期现场部署，评估了9种基于视觉、激光雷达、毫米波雷达和本体运动的里程计、定位与SLAM方法在不同季节的性能表现。

**💡 创新点**

创新点在于首次实现亚北极环境下的长时段、多季节现场验证，并提出适用于无姿态基准的 SARTE 与 LFR 等误差指标与时间序列评价框架，揭示传感器在极端季节变化中的鲁棒性与脆弱性。

**🔧 技术方法**

实验采用三台 GNSS 接收器的 PPK、VectorNav IMU、ZED X 双目摄像机、128 通道 RoboSense 激光雷达、Navtech CIR‑304H FMCW 雷达，并结合开源里程计/定位算法（WILN、2Fast‑2Lamaa、KISS‑SLAM、ORB‑SLAM3、cuVSLAM、DROID‑SLAM、RTR、Navtech‑Radar‑SLAM）进行离线评估。

**📊 数据集**

数据集由 12 个月、64 条轨迹（道路、路肩、越野区）在夏、秋、冬季采集的原始传感器数据组成，并通过三台 GNSS 生成定位基准，未使用公开数据集。

**📈 对比分析**

评估采用 SARTE 衡量相对误差，报告失败率与 LFR，结果显示本体里程计仍是最稳健基线；激光雷达方法在大多数季节误差低于 1 m 但冬季及雪崩环境下仍显脆弱；视觉方法因光照与特征稀疏导致误差超过 10 m；雷达方法误差较大，跨季定位失败率高。

**⚠️ 局限性**

局限性包括缺乏完整 6‑DoF 基准导致误差评估受限、场景对比度不足导致激光/雷达特征匹配失败、跨季定位难度大、实验仅在单一车辆平台完成，缺少对其他平台或多传感器融合方案的验证。

---

## 143. GAAT: Geometry-Aware Alignment Transformer for Multimodal UAV Perception

**arXiv ID:** 2608.27971 | [PDF](https://arxiv.org/pdf/2608.27971v1)

**作者:** Jingpu Yang `[一作]` (Beihang University), Yufeng Wang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GAAT（Geometry-Aware Alignment Transformer），一种在UAV多模态感知中先进行几何对齐再融合的预训练模型。

**💡 创新点**

创新点在于同步补丁中心对齐(syncPATC)学习局部可靠性先验，利用该先验驱动稀疏跨模态注意力(MG‑Sparse‑MMA)和跨尺度对比学习(RA‑QCGCL)，实现对局部失配的自适应校正。

**🔧 技术方法**

采用Swin‑V2编码器、同步仿射变换、变形注意力、跨尺度对比学习和多任务预训练。

**📊 数据集**

使用了UAVMeta（RGB‑IR图像、注释及飞行状态元数据）和StateBench（四种采集状态指标），以及KUST4K、DroneVehicle等公开UAV评测基准。

**📈 对比分析**

在语义分割、目标检测、场景分类、变化检测、多目标跟踪、3D重建等六大任务中，GAAT均达到或超越当前最先进的多模态/遥感基线，整体提升幅度可达5–10个百分点。

**⚠️ 局限性**

局限包括仅在RGB‑IR成对数据上验证，对非同轴或更高频变化（如SAR、LiDAR）需进一步推广；稀疏对齐方式仍依赖预定义的邻域半径，极端视角或大尺度失配时可能不足。

---

## 144. AI Writers Have a Consistent Stylometric Footprint, but AI Editors Do Not

**arXiv ID:** 2608.27855 | [PDF](https://arxiv.org/pdf/2608.27855v1)

**作者:** Zhengyang Shan `[一作]` (Boston University), Sophie Hao `[通讯]` (Boston University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了AI生成文本和AI编辑文本的文体特征，构建了跨领域、跨模型的大规模数据集。

**💡 创新点**

提出AI生成文本具有一致的“风格足迹”（高熵与高词汇多样性），而AI编辑文本呈现不同的特征，证明生成与编辑是两种不同的文体现象。

**🔧 技术方法**

采用14个可解释的文体特征（词汇多样性、熵、突发性、可读性等），使用逻辑回归和特征重要性分析进行文本检测。

**📊 数据集**

使用包含5个写作领域（维基百科、WikiHow、Reddit QA、arXiv、Reddit故事）共45,000篇人类/生成文本和273,420篇编辑文本，覆盖8个开源LLM和3个编辑模型。

**📈 对比分析**

与EditLens等现有编辑检测模型和生成检测模型进行AUC/TPR比较，逻辑回归在生成与编辑区分上AUC>0.98，编辑与人类区分约0.80；结合两种模型可提升至AUC≈0.99。

**⚠️ 局限性**

局限包括只使用14个可解释特征，未覆盖所有语言与会话场景；实验聚焦于英文长文本，且模型与领域不完整，无法完全证明因果关系。

---

## 145. Predicting Turn-Taking Outcomes in Multi-Party Conversation: Interpretable Modelling of Speech and Gaze Dynamics with Interpersonal Closeness

**arXiv ID:** 2608.27988 | [PDF](https://arxiv.org/pdf/2608.27988v1)

**作者:** Mark Dourado `[一作]` (Aalborg University), Stefania Serafin `[通讯]` (Technical University of Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

建立了一个基于眼动、声学强度和人际亲密度的多模态逻辑回归模型，用于预测四人自由对话中的说话者切换结果（gap 或 overlap）。

**💡 创新点**

证明眼动特征单独即可预测说话者切换，首次将主成分分析（PCA）和非负矩阵分解（NMF）用于眼动转移模式，并展示在不同噪声条件下眼动仍具鲁棒性。

**🔧 技术方法**

采用弹性网络正则化的逻辑回归、PCA、NMF、静态与转移熵、符号眼动二元组、声学强度（Phon）标准化，以及 LOGO/LOCO 交叉验证等技术。

**📊 数据集**

使用 GaMMA 语料库，该数据集包含四人同步录制的音频、眼动和运动数据，并涵盖安静、55 dB、65 dB 和变化噪声条件。

**📈 对比分析**

通过 ROC AUC、PR AUC、F1 等指标对比模型，结果显示多模态模型平均 ROC AUC 为 0.76 ± 0.04，显著高于仅眼动模型的 0.58 ± 0.05（提升 0.18），在所有噪声条件下均保持高 AUC。

**⚠️ 局限性**

局限包括数据量有限、仅使用固定 3 秒预转窗口未捕捉完整对话动态、未考虑头部运动或语言上下文、模型依赖预标注的转移事件，以及代码不可公开。

---

## 146. ContextLeak: Exfiltrating LLM Agent Context via Malicious Tools

**arXiv ID:** 2608.27800 | [PDF](https://arxiv.org/pdf/2608.27800v1)

**作者:** Yuqi Jia `[一作]` (Duke University), Neil Gong `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于强化学习微调攻击 LLM 的方法，自动生成恶意工具名称和描述，从而诱使 LLM 代理在执行任务时既选择恶意工具又将运行时上下文（如用户提示、对话历史和工具列表）作为输入参数泄露。

**💡 创新点**

创新点包括：①针对上下文泄露的三大条件（工具选择、上下文传递、数据传输）提出完整攻击框架；②设计专门的奖励函数和基于策略的候选生成机制，使攻击 LLM 能同时优化工具选择和上下文传递；③通过阴影用户模拟实现跨域、跨模型的鲁棒攻击，显著优于现有提示注入、越狱和恶意工具攻击。

**🔧 技术方法**

使用了强化学习（DAPO）、LoRA 微调、策略抽取与缓冲、基于 Levenshtein 距离和长度正则化的相似度奖励，以及自然语言生成技术。

**📊 数据集**

使用 ToolBench 生成的 800 个阴影用户（涵盖 10 个应用领域）和 ToolAlpaca 的 400 个受害者用户（覆盖不同域和数据集）进行训练与评估。

**📈 对比分析**

与多种基线（Combined Attack、ObliInjection、PAIR、TAP、JudgeDeceiver、ToolHijacker、AMA、ToolTweak）在三类上下文上进行对比。该方法在 MTSR、EDS、ES、Precision/Recall/F1 上均取得最高分，尤其在跨域、跨模型迁移中保持 90% 以上的工具选择率和接近 1.0 的上下文重构质量。

**⚠️ 局限性**

限制主要在于：①依赖于代理对工具描述的解释，无法针对复杂工具列表或执行轨迹进行泄露；②现有检测与防御方法对该攻击几乎无效；③需要一定的阴影用户和与代理交互的能力，对攻击者资源有一定门槛。

---

## 147. Is Monte Carlo Tree Search Just Every-Visit Monte Carlo Control?

**arXiv ID:** 2608.27985 | [PDF](https://arxiv.org/pdf/2608.27985v1)

**作者:** Xianyi Wu `[一作]` `[通讯]` (East China Normal University), Xianyi Wu (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文讨论了 Monte Carlo Tree Search 与每次访问 Monte Carlo 控制的本质相同，并给出了两者的对应关系。

**💡 创新点**

提出将 MCTS 四个步骤映射为 MC 控制的两大基本操作，从术语层面统一两者。

**🔧 技术方法**

采用理论分析与公式推导。

**📊 数据集**

无实验数据集。

**📈 对比分析**

无实验对比，主要是理论说明。

**⚠️ 局限性**

仅限于轨迹采样和回报更新，未涉及策略学习或复杂环境验证。

---

## 148. Probing Perceptual Priors of MLLMs via Gibbs Sampling with Interpretable Generative Controls

**arXiv ID:** 2608.27727 | [PDF](https://arxiv.org/pdf/2608.27727v1)

**作者:** Manuel Cherep `[一作]` (MIT), Nikhil Singh `[通讯]` (Dartmouth College)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用基于VLM二元选择的Metropolis–Hastings within Gibbs采样器，在可解释的生成空间（FLUX + SliderSpace）中探测多模态语言模型的感知先验。

**💡 创新点**

创新点在于：①将人类感知先验采样框架迁移到模型；②使用可解释滑块作为采样维度，确保采样方向可解释；③将VLM的二元偏好视为Barker接受步骤，实现无额外成本的Gibbs更新；④系统性覆盖四个高风险图像域和四个前沿VLM，揭示此前未显现的隐式偏差。

**🔧 技术方法**

技术方法包括：Metropolis–Hastings within Gibbs（IMH-within-Gibbs），Barker接受规则，SliderSpace对FLUX的LoRA滑块训练，生成图像的对比测试，基于Bradley–Terry模型的偏好建模，随机化图像呈现顺序以消除位置效应。

**📊 数据集**

数据集：使用生成的图像空间，针对四个域（人脸、可用性、审美、真实性）各自训练64个候选滑块，人工挑选10个可解释滑块；目标标签共23个（如吸引力、可信度、艺术价值等）。

**📈 对比分析**

比较方法：与两种显式基线（标量评分任务和配对强制选择）对比；实验共计超过2.5M次采样，1B+ tokens。结果显示：①采样得到的先验更为尖锐、信息量更高；②不同VLM在相同域上的共识度差异可量化；③可视化显示明显的隐式偏差与人类评价不一致；④在某些域（真实性）跨模型一致性最高，在审美域一致性最低。

**⚠️ 局限性**

局限性：①先验仅限于SliderSpace + FLUX 能表达的空间；②若滑块不完全正交或存在多属性交互，采样假设可能失效；③方法揭示偏差但无法直接说明其来源（预训练、对齐或生成器干扰）；④对高维域的扩展受生成成本限制。

---

## 149. Spectral Features Dominate BCG Respiratory-Event Detection: A Large-Scale Patient-Independent Comparison of Feature Groups in Sleep Apnea Patients

**arXiv ID:** 2608.28242 | [PDF](https://arxiv.org/pdf/2608.28242v1)

**作者:** Israel Campero Jurado `[一作]`, Elisabeth Wilhelm `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

未提供具体研究内容

**💡 创新点**

无创新点说明

**🔧 技术方法**

未提及使用技术

**📊 数据集**

未提及数据集

**📈 对比分析**

未说明对比方法与性能

**⚠️ 局限性**

论文缺乏细节，无法评估限制

---

## 150. Lexically conditioned realization ambiguity in Korean predicate morphology

**arXiv ID:** 2608.27966 | [PDF](https://arxiv.org/pdf/2608.27966v1)

**作者:** Wonjun Oh `[一作]` (Korea Advanced Institute of Science & Technology), Jungyeul Park `[通讯]` (Korea Advanced Institute of Science & Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了韩语表层实现与形态分析之间的逆向关系，提出同形近的同义词在不同变位类中的表层形式不唯一的现象。

**💡 创新点**

创新点在于将不规则变位重新视为词汇层面的“同义词变位差异”，并证明形态分析本身不足以决定表层形态，强调词义和论元结构对选择实现类的决定作用。

**🔧 技术方法**

采用形态分析、语料统计与表层重构实验，以及对词典中的子类别框架和语义角色的分析。

**📊 数据集**

使用了 KLUE 依存树库（Korean Language Understanding and Evaluation）作为实验数据。

**📈 对比分析**

将形态分析结果与表层重构进行对比，实验显示在加入实现类约束后重构准确率显著提升，证明了词汇层面信息的必要性。

**⚠️ 局限性**

局限性在于研究仅聚焦于有限的同义词变位差异类，未覆盖所有韩语谓词和名词形式，未来需扩展到更大范围的词汇和语法结构。

---

## 151. Self-Explainable Multi-Label Graph Neural Network for Correlated Evidence Attribution

**arXiv ID:** 2608.27574 | [PDF](https://arxiv.org/pdf/2608.27574v1)

**作者:** Yingqi Feng `[一作]` (Florida Atlantic University), Xingquan Zhu `[通讯]` (Florida Atlantic University)

**通讯引用:** 25174 | [OpenAlex ID](https://openalex.org/A5084641325)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种端到端的自解释多标签图神经网络（SE‑MGNN），同时实现多标签节点分类和针对每个标签的边缘级证据归因；

**💡 创新点**

创新点在于：①将标签相关性同时嵌入预测与解释模块，允许正相关标签共享证据、弱/负相关标签保留独立证据；②在训练阶段直接学习稀疏、近二值的边缘掩码，实现可解释性与性能的统一；③使用标签相关图和标签感知边缘评分器实现标签条件化解释。

**🔧 技术方法**

主要技术包括：基于均值的GraphSAGE编码器、带标签相关残差的预测头、标签相关图的GCN传播、稀疏软边缘掩码学习、必要性与充分性约束的联合损失、梯度引导的标签条件化掩码以及自适应Top‑M解释提取。

**📊 数据集**

实验数据集包括两种合成数据（SynAnchor、SynMotif）以及三种真实世界图（社交网络BlogCatalog、YouTube，生物网络HumLoc），覆盖多标签、不同标签相关性与图结构的多样场景。

**📈 对比分析**

与传统预测基线（PO、GAT、ML‑GCN、LIP、BR）以及自解释（GSAT）和后置解释器（GNNExplainer、PGExplainer）比较，SE‑MGNN在大多数数据集上保持或提升多标签预测（Micro/Macro F1、AUPRC），在合成数据上获得最佳标签条件化的GT对齐与Fidelity指标，在真实数据上实现更高的Fidelity+和Fidelity‑，且生成的解释更稀疏、标签特异性更强。

**⚠️ 局限性**

局限性包括：①对标签相关性的依赖，若标签相关性弱或不稳定时解释效果有限；②需要在训练阶段额外学习掩码，计算开销和超参调优较大；③在无真实解释的真实数据上，无法客观评估解释质量，只能依赖自我指示的可信度指标；④在某些数据集（如YouTube、HumLoc）中预测或解释性能提升不明显，表明方法对图与标签结构的适应性仍受限。

---

## 152. SABER: Stability-Aware Early Exit for LLM Reasoning via Adversarial Branch Probing

**arXiv ID:** 2608.27963 | [PDF](https://arxiv.org/pdf/2608.27963v1)

**作者:** Wanli Cheng `[一作]` (Soochow University), Wenliang Chen `[通讯]` (Soochow University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SABER框架，通过对中间推理状态进行对抗性分支探测实现训练无关的稳定性感知早停。

**💡 创新点**

创新点在于利用对抗性语义扰动评估推理稳定性，联合语义一致性与置信稳定性得到RSS决策，从而高效地提前终止推理。

**🔧 技术方法**

采用对抗性分支探测、语义一致性（Jaccard）、置信稳定性（指数衰减）以及RSS评分的训练无关策略。

**📊 数据集**

在六大数学推理基准（GSM8K、MATH-500、AMC23、OlympiadBench、AIME 2024/2025）及科学推理基准GPQA Diamond上进行评估。

**📈 对比分析**

与vanilla、NoThinking、DEER、Dynasor等基线对比，SABER在保持或提升准确率的前提下平均减少30.2%–39.8%推理token，压缩率提升至约0.68，推理延迟下降近50%。

**⚠️ 局限性**

局限性包括仅在4B-8B规模模型与文本推理任务上验证，未评估更大模型、跨模态或基于智能体的场景，且对抗性探测的采样成本仍存在一定开销。

---

## 153. Contact-Guided Exploration for Non-Prehensile Locomanipulation with Multi-Critic RL

**arXiv ID:** 2608.28140 | [PDF](https://arxiv.org/pdf/2608.28140v1)

**作者:** Simone Tolomei `[一作]` (Universita di Pisa), Marco Hutter `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种多重评估器（Multi-Critic）强化学习框架，通过接触引导探索学习非抓取搬运任务；

**💡 创新点**

创新点在于将探索奖励与任务奖励分离为不同的价值函数，并采用权重衰减策略从探索过渡到任务最优，同时利用通用抓取算法生成的接触候选点作为探索引导；

**🔧 技术方法**

使用了Multi-Critic PPO、LSTM上层策略、层次化底盘控制、密集探索奖励、接触候选点采样以及权重调度等技术；

**📊 数据集**

实验基于IKEA家具数据集、程序化生成的椅子和箱子模型进行仿真，并在ALMA四足移动机械手上进行真实硬件验证；

**📈 对比分析**

与单一Critic PPO、PPO+权重衰减等基线对比，仿真中成功率超过90%，硬件实验成功率约70-90%，显著优于基线，减少倾覆率并提升任务完成效率；

**⚠️ 局限性**

局限性包括依赖外部运动捕捉进行位姿跟踪、权重衰减调度需手工设定、对极端几何形状的适应性有限，以及在更大域迁移和完全自主调度方面尚需改进。

---

## 154. AERA: Adaptive Evidence Residual Allocation for Efficient Test-Time Reasoning

**arXiv ID:** 2608.27964 | [PDF](https://arxiv.org/pdf/2608.27964v1)

**作者:** Ziming Wang `[一作]` (National University of Singapore), Hangwei Qian `[通讯]` (Agency for Science, Technology and Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一个名为 AERA 的自适应推理调度器，在推理过程中根据已生成响应前缀的多维特征决定是否继续生成更多响应，以降低计算成本。

**💡 创新点**

创新点在于提出“残差效用预测”框架，利用未来完整推理的正确性来监督只观察当前前缀的控制器，并结合答案分布、时间变化、重解行为和语义一致性等多组特征全面描述推理状态。

**🔧 技术方法**

技术手段包括使用冻结的 Qwen2.5-7B Re^2 推理器、Re^2 聚合规则、基于多层感知器的二分类门控、特征标准化、语义投影和多点校准的阈值选择。

**📊 数据集**

实验数据集主要为 GSM8K 和 GPQA Diamond，另外在这些数据集上进行了离线 replay、嵌套校准和在线增量生成的评估。

**📈 对比分析**

与固定预算、Early‑Stopping Self‑Consistency、Adaptive‑Consistency、随机调度等基线对比，AERA 在 GSM8K 上保持约92.6% 准确率的同时将平均响应数从 128 降至 4.4，节省约 96% 的响应/95% 的完成 token；在 GPQA 上与 ASC 竞争，并在答案崩溃场景下表现更优；总体上实现了接近全算精度的高效推理。

**⚠️ 局限性**

局限性包括：仅在冻结推理器上评估，未测量实际延迟或能耗；阈值和模型需依赖校准数据，跨模型/跨域迁移不确定；对难题或分布偏移可能出现误停；语义投影与特征选择依赖于特定数据集。

---

## 155. Curvature-Aware Radius Shrinkage for Adaptive Nearest Neighbor Classification

**arXiv ID:** 2608.27634 | [PDF](https://arxiv.org/pdf/2608.27634v1)

**作者:** Alexandre L. M. Levada `[一作]` `[通讯]` (Federal University of Sao Carlos), Alexandre L. M. Levada (Federal University of Sao Carlos)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于局部曲率的半径收缩自适应最近邻分类器（CARSANN）。

**💡 创新点**

创新点在于：①利用流形假设先估计内在维度并投影到低维表示；②通过形状算子计算局部均值曲率；③将曲率映射为邻域半径收缩而非单纯调整邻居数或度量，提供了新的空间尺度自适应机制。

**🔧 技术方法**

主要技术包括：TwoNN内在维度估计、PCA低维表示、局部曲率估计（形状算子/二次特征映射）、基于曲率的半径缩放函数、距离加权k‑NN投票。

**📊 数据集**

在OpenML平台获取的70+真实世界分类数据集上进行实验，涵盖样本数、维度、类别数差异较大的数据。

**📈 对比分析**

与标准k‑NN、DANN（判别自适应最近邻）以及kk‑NN（曲率驱动邻居数自适应）比较，CARSANN在40/45个数据集上实现更高的平衡准确率和F1分数；平均平衡准确率从0.6506提升到0.7528，F1从0.7053提升到0.7933，且差异在Friedman/Nemenyi检验下显著。

**⚠️ 局限性**

局限性包括：高计算复杂度（≈O(nd⁴)），依赖曲率估计与k_curv的设置；在某些数据集DANN或kk‑NN表现更好；对极高维或极小样本的适用性需进一步验证。

---

## 156. A Framework for Object-Centric Predictive Monitoring of Collaborative Processes

**arXiv ID:** 2608.27671 | [PDF](https://arxiv.org/pdf/2608.27671v1)

**作者:** Daniel Calegari `[一作]` (Universidad ORT Uruguay), Martín Rubio `[通讯]` (Universidad de la República)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5101279325)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过形式化的映射将传统的协作式事件日志转换为符合 OCED 核心模型的 OCEL 2.0 对象‑中心日志，并在此基础上重新定义协作预测任务；

**💡 创新点**

创新点在于提供了（1）从协作日志到对象‑中心表示的语义一致映射；（2）将协作预测任务转化为对象‑中心预测任务；（3）实现了可复现的转换器与预测流水线；

**🔧 技术方法**

使用了 OCEL 2.0 规范、对象‑中心特征提取框架（tabular、sequential、graph 编码）以及五种预测模型（Random Forest、XGBoost、Transformer、LSTM、GNN），并在此框架下完成标签生成和模型训练；

**📊 数据集**

实验数据包括四个公开协作日志（人工合成与真实案例）以及从 BPI Challenge 2013 重新解释得到的规模更大的日志；

**📈 对比分析**

通过与传统协作 PPM（Predict‑Collab）在同一任务上对比，证明了对象‑中心表示能够在大多数任务上实现可接受甚至更优的预测性能；

**⚠️ 局限性**

主要局限包括：对象‑中心模型显著增大了日志结构复杂度和存储/计算成本；缺乏消息同步与匹配的显式表示；资源层面预测未得到充分利用；以及在规模更大、真实数据上的数值预测效果不稳定。

---

## 157. A Tight Analysis of Khatri-Rao Oblivious Subspace Embeddings

**arXiv ID:** 2608.28094 | [PDF](https://arxiv.org/pdf/2608.28094v1)

**作者:** Lorenzo Beretta `[一作]` (IBM), Cameron Musco `[通讯]` (UMass Amherst)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究了带有Khatri‑Rao结构的随机投影矩阵，并证明其在任意维度d下能够实现接近无结构投影的子空间嵌入（OSE）性能，即嵌入维度m可满足m≈O(k/ϵ²)，其中k为子空间维度。

**💡 创新点**

创新点在于用极其简单的两条性质（列独立、同方差、弱Johnson‑Lindenstrauss矩阵矩）即可完成上述结果，填补了Khatri‑Rao sketch与无结构随机投影在OSE性能上的缺口，尤其在d=2时将先前的O(k^{3/2})提升到O(k·polylog(k))，而对一般d则将O(k^d)提升为O(k·polylog(k)^d)。

**🔧 技术方法**

核心技术包括：
- Khatri‑Rao列的同方差与独立性证明；
- 对子空间投影矩阵的弱J‑L矩阵矩推导（多阶子高斯随机幂的矩估计）；
- 矩阵Chernoff集中极限；
- truncation（截断）技术控制列范数的尾部；
- 结合上述技术给出m的上界。

**📊 数据集**

本工作主要为理论分析，未使用具体实验数据集；所有结果均为理论上限。

**📈 对比分析**

与之前的理论比较：
- 对d=2，先前最优结果为m=O(k^{3/2}/ϵ²)；本研究得到m=O(k·log(k)^3/ϵ²)。
- 对d>2，先前最优为m=O(k^d/ϵ²)；本研究得到m=O(k·log(k)^{d+1}/ϵ²)。
- 这些结果表明Khatri‑Rao sketch在子空间嵌入上几乎可以匹配无结构投影的线性k依赖，性能大幅提升。

**⚠️ 局限性**

局限性包括：
- 上界中隐藏了d^O(d)因子，对高阶张量可能仍然较大；
- 证明依赖于每个因子矩阵列的子高斯同方差假设，未覆盖稀疏Count‑Sketch等变种；
- 常数与log因子对实际实现的影响未给出经验评估。

---

## 158. First Make It Playable, Then Make It Good: Staged Interaction Learning for Small Dialogue-Game Agents

**arXiv ID:** 2608.27672 | [PDF](https://arxiv.org/pdf/2608.27672v1)

**作者:** Syed Mahbubul Huq `[一作]` (City, University of London), Pranava Madhyastha `[通讯]` (City, University of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练一个2B参数的对话游戏模型Qwen-GuidePlay-2B，使用三阶段微调（成功轨迹SFT、价值加权回合SFT、教师引导修复）在Playpen框架中提升可玩性与决策质量。

**💡 创新点**

通过成功轨迹过滤、价值加权局部决策学习以及轻量级教师评估/修复的组合，显著提升可玩性和质量，而无需昂贵的重放-修复或硬样本挖掘，证明小模型可通过精心策划的训练策略获得高性能。

**🔧 技术方法**

Qwen3.5-2B基础模型、LoRA适配器、价值加权监督、教师模型Gemma-4-31B-it的判定与修复、交互式Playpen评估。

**📊 数据集**

Playpen游戏交互数据，20,202条成功轨迹，提取的105,972条回合级state-action样本；训练集包含成功轨迹、回合样本、教师评判样本与修复样本。

**📈 对比分析**

与官方Qwen3.5-2B基准和多种消融（无教师、重放修复、硬样本挖掘）比较；在公开Playpen验证集上clemscore从13.05提升至57.12，statscore从44.02提升至42.68，官方挑战中获得第二高clemscore提升+36。

**⚠️ 局限性**

未使用在线强化学习、搜索或规划；教师仅评判/修复已成功轨迹，未提供新策略；跨域泛化有限，OOD提升仅+6.53。

---

## 159. Time Capsule of Testable Human Knowledge: 41 Years of Jeopardy! in a Single Free Local Model

**arXiv ID:** 2608.27459 | [PDF](https://arxiv.org/pdf/2608.27459v1)

**作者:** David Noever `[一作]` (PeopleTec, Inc.), Forrest McKee `[通讯]` (PeopleTec, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了一种9 GB开源量化语言模型在完整的529,939条Jeopardy!提示集（覆盖41季）上的问答表现，验证其能压缩并保留与IBM Watson相当的通用知识；

**💡 创新点**

创新点在于：①首次对完整Jeopardy!语料库进行模型评测；②从时间维度区分已知与未知提示，检验模型的泛化能力；③将模型压缩到单文件形式，展示文化知识可携带的“知识胶囊”概念；

**🔧 技术方法**

使用的技术包括4‑bit量化的Qwen2.5‑14B‑Instruct模型、Ollama推理框架、严格的强制响应与模糊匹配评分；

**📊 数据集**

数据集为公开的Jeopardy!提示集，包含1984–2025年41季共529,939条提示、类别、金额、答案等信息；

**📈 对比分析**

比较方法：在所有提示上计算词汇匹配准确率，并与Watson的全部提示准确率（75.4%）和Claude Opus 4.8在分布外提示的表现（≈95%）进行对照；结果显示本模型在所有提示上达到67.0%（内部），在训练截止后提示上为65%，差距仅2个百分点，证明其知识泛化能力；

**⚠️ 局限性**

局限性：使用强制响应且不采用置信度门控；仅用词汇匹配评分，可能低估语义正确率；正则表达式分类和手工正则类型标签存在近似；缺乏对更细粒度错误分析及更大模型对比。

---

## 160. When Tokenizers Fail: Byte-Level Chunking for Zero-Shot Transfer to Low-Resource Languages

**arXiv ID:** 2608.27658 | [PDF](https://arxiv.org/pdf/2608.27658v1)

**作者:** Sanjeev Kumar `[一作]` (IIT Bombay), Nikolaos Aletras `[通讯]` (University of Sheffield)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于字节的分层网络（adapted H‑Net）来替代低资源语言中的子词分词，以获得更好的词级表示。

**💡 创新点**

通过从冻结的子词模型中直接初始化字节嵌入、对齐字节块与子词向量、以及在预训练中加入POS监督，实现在仅500K句子下即可提升词级任务。

**🔧 技术方法**

采用 H‑Net 的四阶段架构，基于 Mamba 局部编码器、动态字节块划分、子词LM骨干和EMA去块化；并加入字节嵌入初始化、块对齐损失和POS监督。

**📊 数据集**

在 Hindi FineWeb2 500K 句子上预训练，使用 Hindi UD treebank 做 POS 监督，评估语言包括 Bhojpuri、Marathi、Magahi、Sanskrit、Urdu，并与 Qwen3/Gemma3 大模型的子词和继续预训练基线对比。

**📈 对比分析**

在 POS、NER、Sentiment 三个零样本下与子词基线、继续预训练基线对比，H‑Net 在词级任务中平均提升 POS 约10.4点、NER 约7.8点，最高可达13.3点；在情感分类则不及子词基线。

**⚠️ 局限性**

仅适用于与 Hindi 共享脚本/语法的低资源印度语言，脚本范围有限（Devanagari、Nastaliq），未验证更大模型、不同任务和计算效率。

---

## 161. PHR-VLA: Planning Horizon Reasoning for Vision-Language-Action Models

**arXiv ID:** 2608.27609 | [PDF](https://arxiv.org/pdf/2608.27609v1)

**作者:** Davood Soleymanzadeh `[一作]` (Texas A&M University), Minghui Zheng `[通讯]` (Texas A&M University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出一种在训练阶段使用未来隐层动态进行规划视域推理的辅助监督框架，帮助视觉‑语言‑动作（VLA）模型在执行时更好地预测动作。

**💡 创新点**

创新点在于利用演示数据中的未来帧的冻结视觉编码器生成的隐层动态作为“特权”监督信号，训练时将其与VLA内部表示对齐，从而无需在推理时引入世界模型或未来回滚。

**🔧 技术方法**

技术包括：VLM（SmolVLM‑2）作为主干、流匹配动作头、冻结视觉编码器（SigLIP或JEPA）、轻量级未来头（g_ϕ）以及对齐损失L_Align。

**📊 数据集**

使用的数据集包括LIBERO（多任务语言驱动桌面操作）、Meta‑World（模拟机器人操作）以及真实世界的拆卸任务数据。

**📈 对比分析**

在LIBERO上平均成功率从84.1%提升至88.4%，在Meta‑World上提升至57.7%（比SmolVLA提升约1%），在真实拆卸任务上平均成功率从63.3%提升至82.5%。

**⚠️ 局限性**

限制在于该方法仅为训练时的辅助目标，无法在推理时提供实时规划或世界模型纠正；未来可能需结合触觉或力学信息进行改进。

---

## 162. Beyond Relative Geometry: Metric-Aware Geometry Perception for Robotics

**arXiv ID:** 2608.27497 | [PDF](https://arxiv.org/pdf/2608.27497v1)

**作者:** Fengjun Zhong `[一作]` (Beihang University), Zhongliang Qiao `[通讯]` (XiaoyuBot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种端到端、可插拔的Metric-Aware Geometry Perception (MAGP) 框架，能够在多视角、可变摄像机/深度输入下直接重建一致的物理尺度几何，并将其作为输入嵌入到机器人操控策略中。

**💡 创新点**

创新点在于三方面：① 通过 Metric Scale Equivariant Augmentation (MSEA) 让模型在尺度变换下保持等变性；② 采用 Flexible Metric Conditioning 让模型能灵活处理任意组合和缺失的相机/深度观测；③ 将相对与度量监督联合起来，既保留细粒度相对几何，又让重建结果与真实尺度保持一致。

**🔧 技术方法**

核心技术包括：单一Transformer骨干网络对多视图图像、相机参数和深度进行统一编码；残差交叉注意力将度量几何特征注入视觉‑语言模型；Metric Scale Equivariant Augmentation、灵活的度量条件、以及联合相对与度量损失的训练策略。

**📊 数据集**

使用的公开数据集有：3D重建基准 ETH3D、MegaDepth、ScanNet++；机器人操控基准 LIBERO、RoboTwin 以及零射击评测基准 LIBERO-Plus。

**📈 对比分析**

与多种先前方法（VGGT、MA、Depth Anything 3、DA3、VGGT-Ω 等）在三大重建基准上进行对比，MAGP 在 δ1.03、δ1.25、Abs（绝对误差）等指标上均取得领先；Abs 误差从 2.01 m 降低到 0.07 m；在 LIBERO 与 RoboTwin 的机器人操控任务中，平均成功率提升 1–6%（在 RoboTwin 上最高提升 6.26%）。

**⚠️ 局限性**

局限性包括：① 对相机/深度观测的可用性仍有一定依赖，缺失观测时性能会下降；② 训练与推理过程相对计算量较大，部署在资源受限的机器人上可能需要进一步压缩；③ 在动态场景或强光/噪声环境下的鲁棒性尚未充分验证。

---

## 163. Trajectory-Level Speculative Decoding for Diffusion Language Models

**arXiv ID:** 2608.27514 | [PDF](https://arxiv.org/pdf/2608.27514v1)

**作者:** Tianxiang Pan `[一作]` (Li Auto Inc.), Kaiwen Long `[通讯]` (Li Auto Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对扩散语言模型的轨迹级投机解码框架，能够在低置信度时保持并行度并显著提升吞吐量。

**💡 创新点**

创新点包括：①将投机扩展到整个去噪轨迹而非单个令牌；②设计基于树的轨迹草稿生成与块级并行验证；③引入跨块投机利用双向注意力；④对投机过程进行理论精确性与漂移分析。

**🔧 技术方法**

主要技术包括：离散扩散语言模型（如LLaDA、Dream）、树结构草稿生成、块级并行注意力掩码、Fast‑dLLM的双缓存机制、与自回归投机方法的对比。

**📊 数据集**

使用数学推理（GSM8K、MATH）和代码生成（HumanEval、MBPP）四大基准数据集进行评估。

**📈 对比分析**

与原始扩散解码和Fast‑dLLM对比，本文方法在保持1%以内准确率的前提下，推理速度提升7–14倍（比原始扩散解码）且比Fast‑dLLM快1.3倍，吞吐量从2.6→4.3 tokens/step，去噪步骤下降30–40%。

**⚠️ 局限性**

主要限制在于高并行度下可能出现轨迹漂移，导致极端推理场景的准确性略有下降；目前的树结构采用手工调参，缺乏自适应机制；跨块投机的实现对硬件要求较高，难以在所有平台普适。

---

## 164. Optimal exponential memory for sequential Euclidean connection: edge-power costs and phase transitions

**arXiv ID:** 2608.27777 | [PDF](https://arxiv.org/pdf/2608.27777v1)

**作者:** Pedro M. M. de Castro `[一作]` `[通讯]`, Pedro M. M. de Castro

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了一种在线几何连接规则，该规则存储一个状态点，并通过更新状态来最小化边长的α次幂和。

**💡 创新点**

创新点在于确定了在不同输入序列下的最优持久性参数，并提供了在α=1时的相变和在高次幂下的优化成本的明确表达。

**🔧 技术方法**

使用了几何移动平均和递归方法来分析状态更新和边长的成本。

**📊 数据集**

使用了单位球内的独立均匀输入点作为数据集。

**📈 对比分析**

与其他在线Steiner树算法相比，本文的方法通过竞争比率或有界回溯进行比较，性能在不同的α值下表现出不同的最优参数和渐近最坏情况成本。

**⚠️ 局限性**

限制在于未能证明一般幂的唯一性，且在高维情况下的全局结构仍需进一步研究。

---

## 165. Non-Uniform Quantisation for 3DGS Compression

**arXiv ID:** 2608.28272 | [PDF](https://arxiv.org/pdf/2608.28272v1)

**作者:** Bert Van hauwermeiren `[一作]` (Vrije Universiteit Brussel), Adrian Munteanu `[通讯]` (Vrije Universiteit Brussel)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对3D Gaussian Splatting（3DGS）模型提出非均匀量化与重要性加权合并技术，以显著降低比特率并提升重建质量。

**💡 创新点**

创新点在于：①引入基于渲染贡献的多维重要性度量，并用小型MLP快速近似；②设计递归加权Lloyd‑Max量化框架；③提出三种加权合并策略（参数平均、协方差空间平均、对数欧几里得平均）并加入相似度阈值；④实现与任何点云压缩标准（G‑PCC / V‑PCC）无缝兼容。

**🔧 技术方法**

使用技术包括：重要性权重估计（MLP）、递归加权Lloyd‑Max量化、加权参数/协方差/对数欧几里得合并、MPEG G‑PCC和V‑PCC熵编码（RAHT、HEVC/ VVC）、多尺度量化和加权差分量化。

**📊 数据集**

数据集：官方MPEG 3DGS基准数据集，包含大规模前景场景与高细节静态物体/人物（如 bartender_stable、cinema_stable、lego_bugatti、plant 等）。

**📈 对比分析**

与均匀量化、Adaptive Voxelization、FlexGaussian等方法对比；在V‑PCC与G‑PCC上均实现了更低的BD‑Rate、提高PSNR/SSIM/IVSSIM/LPIPS等指标，解码速度与基线相近或更快；在绝大多数场景中显著提升了率-失真曲线。

**⚠️ 局限性**

主要限制：预处理阶段（重要性估计、量化与合并）计算量大，导致单机预处理时间显著增长；在极低量化比特率时二次量化可能导致质量波动；未在实时训练管线中测试，且对极端高分辨率场景的扩展性尚待验证。

---

## 166. The Role of Mixed and Augmented Reality in Medical Visualization: Literature Review and A Context-Aware Taxonomy

**arXiv ID:** 2608.27644 | [PDF](https://arxiv.org/pdf/2608.27644v1)

**作者:** Xinrui Zou `[一作]` (Johns Hopkins University), Alejandro Martin-Gomez `[通讯]` (University of Arkansas)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述并构建了基于上下文的AR/MR医学图像可视化六维度分类法，分析了313篇研究在临床任务、影像模态、显示技术等方面的应用分布。

**💡 创新点**

提出了“应用–影像模态–维度–显示技术–可视化锚定–感知支持”六维度框架，并强调锚定与感知维度在设计决策中的关键作用，是首个针对AR/MR可视化的系统化、上下文驱动的分类工具。

**🔧 技术方法**

采用文献检索（PubMed、IEEE Xplore、Scopus）和PRISMA 2020流程，对选取论文进行双评审分类；使用自定义模板对每篇论文进行维度标注。

**📊 数据集**

使用的“数据集”为313篇符合纳入标准的同行评议论文；未涉及医学图像或其他实验数据集。

**📈 对比分析**

通过统计分析各维度在不同临床任务（教育、训练、术前规划、术中应用）和医学专科（神经、骨科、腹部等）的分布，绘制比例图，揭示趋势与不足；未进行算法性能对比或定量指标评估。

**⚠️ 局限性**

局限性：①缺乏定量评估与实验验证，分类仍为描述性；②多数研究未报告感知支持或锚定细节，可能导致报告偏差；③术语差异与信息不完整可能影响归类准确性；④未能系统评估不同可视化策略对临床任务效果的具体影响。

---

## 167. Klangfarbenakkord and Klangfarbenharmonien Metric Space Models for Music on Informational Geometry 1

**arXiv ID:** 2608.28026 | [PDF](https://arxiv.org/pdf/2608.28026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 168. WM-R1: Training GUI Agents to Reason and leverage World Models with Reinforcement Learning

**arXiv ID:** 2608.27508 | [PDF](https://arxiv.org/pdf/2608.27508v1)

**作者:** Yu Han `[一作]` (East China Normal University), Tianwen Qian `[通讯]` (East China Normal University)

**通讯引用:** 382 | [OpenAlex ID](https://openalex.org/A5071784615)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

训练基于世界模型的移动 GUI 智能体，完全用仿真环境替代真实交互，实现零成本并行化训练。

**💡 创新点**

将世界模型嵌入训练循环，使代理能够在链式思考中先模拟后执行，并设计多维奖励平衡任务成功、轨迹效率与模型调用。

**🔧 技术方法**

基于 Qwen2.5‑VL‑3/7B 的 GRPO 框架，结合 Code2World 生成可渲染 HTML 的世界模型、思考标签（CoT）与奖励模型。

**📊 数据集**

使用 AndroidCode、GUI‑Odyssey 与 GUI‑R1 三大数据集过滤后抽取的 2000 条中等难度任务作为训练集。

**📈 对比分析**

在 AndroidWorld、GUI‑Odyssey、ScreenSpot‑Pro/V2、AndroidControl 等基准上与 GRPO、UI‑R1、UI‑TARS 等模型对比，WM‑R1 在 3B/7B 规模上分别将任务成功率提升至约 29.5%/48.6%，明显优于所有对比模型。

**⚠️ 局限性**

仍依赖预训练的 Code2World，世界模型预测误差可能导致错误思考；在极长序列或高度动态 UI 上的鲁棒性尚待进一步验证。

---

## 169. ABCD: Alpha-Composited Block Coordinate Descent: Constant-VRAM Training for Large Radiance Fields

**arXiv ID:** 2608.27735 | [PDF](https://arxiv.org/pdf/2608.27735v1)

**作者:** Ka Heng Shiu `[一作]` (University of Edinburgh), Kartic Subr `[通讯]` (University of Edinburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于块坐标下降的 Alpha‑Composited Block Coordinate Descent（ABCD）框架，能够在 GPU 内存不随场景规模增长而增长的前提下，对 alpha 合成的辐射场（以 3D Gaussian Splatting 为实例）进行 out‑of‑core 训练。

**💡 创新点**

创新点在于利用 alpha 合成的可关联性，将非活跃分区预渲染为前景和背景 RGBA 图像，使得每一次更新只需 GPU 内存存放活跃分区和两张缓存图像，显著降低对显存的需求并保持重建质量。

**🔧 技术方法**

使用的技术包括块坐标下降、alpha 合成预渲染、前景/背景图像缓存、3D Gaussian Splatting、磁盘/系统/GPU 分层存储以及分区可见性裁剪。

**📊 数据集**

实验数据集来自原始 NeRF 数据集的 kitchen 和 garden 两个场景，约 200 张训练相机视角。

**📈 对比分析**

通过与标准 3DGS、去掉合成的 ABCD 进行对比，ABCD 在 PSNR 上仅损失 3% 以内，VRAM 几乎不增（与 3DGS 相比低 15%），但训练时间增加约 2–3 倍，体现了显存优势和重建质量的兼顾。

**⚠️ 局限性**

局限性包括在小规模场景下优势不明显、系统 RAM 和缓存占用较高、训练速度下降，以及对分区分配、可见性裁剪和高斯边界处理的依赖仍需进一步改进。

---

## 170. Investigating Forecast Proficiency of Hurricane-Induced Compound Flooding With a Discontinuous Galerkin Shallow Water Equation Solver

**arXiv ID:** 2608.27778 | [PDF](https://arxiv.org/pdf/2608.27778v1)

**作者:** Matthew Scarborough `[一作]` (Norwegian University of Life Science), Eirik Valseth `[通讯]` (Norwegian University of Life Science)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文使用离散Galerkin浅水方程求解器 DG‑SWEM，对 2024 年德克萨斯州的伯里尔飓风进行复合洪水预测，并将基于参数的降雨模型 R‑CLIPER 集成到求解器中，以捕捉降雨径流与风暴潮的非线性交互。通过对比最佳轨迹（Best Track）和七条预报通告（Advisories）中的雨水强制和无雨水强制模拟，验证了该方法在水位观测与高水位标记（HWM）上的表现。

**💡 创新点**

创新点在于：① 将降雨直接作为连续性方程的源项加入 DG‑SWEM，实现在实时预报框架中自动生成降雨场；② 采用参数化降雨模型 R‑CLIPER，能够仅凭 NHC 的中心位置和风速等信息快速构建降雨率场；③ 通过对比雨水强制与无雨水强制，量化降雨对洪水范围和深度的影响，首次在德克萨斯州飓风案例中展示了降雨-风暴潮耦合对预测准确性的显著提升。

**🔧 技术方法**

主要技术手段包括：
- 离散Galerkin (DG) 浅水方程求解器 DG‑SWEM，采用局部 Lax‑Friedrichs 边界通量和高阶 Dubiner 基函数；
- 采用显式强稳定性保持（SSP）Runge‑Kutta 时序；
- R‑CLIPER 参数化降雨模型，将降雨率作为空间函数插值到 FE 节点；
- 利用 Unstructured ECGC 网格（约 200 万节点，120 m 最小分辨率）模拟德克萨斯海岸及其周边海湾；
- 在 Frontera 超算上使用 2000 处理器并行计算，时间步 0.25–1 秒。

**📊 数据集**

使用的数据集包括：
- NHC 提供的最佳轨迹（Best Track）与 42 条预报通告的风场、压力和降雨信息；
- NOAA 的水面高度观测站（6 站）和 104 条 Harris County 高水位标记；
- NOAA 的风暴潮和降雨预报；
- 公开的卫星和雷达降雨估计，用于构造 R‑CLIPER 参数；
- 参考文献中提到的 USGS 观测。

**📈 对比分析**

比较方法与性能：
- 对比无雨水强制与有雨水强制的模拟，计算 RMSE、相关系数、最佳拟合斜率、R²、NRMSE；
- 在水位观测站处与 HWM 数据进行验证；
- 结果显示：加入降雨后 RMSE 下降 5–10%，相关系数提升约 0.02–0.03；
- HWM 覆盖率从仅 2/104 提升至 25/104，绝大多数相对误差 <10%；
- NRMSE 在最佳轨迹和最近的通告中下降 20% 以上，表明降雨耦合显著提升预测精度。

**⚠️ 局限性**

局限性：
- 网格域不覆盖更深的内陆河道，导致无法模拟如 Tres Palacios 河等重要河流的洪水；
- 仅采用参数化降雨模型，未考虑雨量空间不连续性和实时雷达更新；
- 没有引入河道排水或径流边界条件，限制了对内陆径流的精细描述；
- 仅分析单一飓风事件，缺乏多案例验证；
- 预报通告的轨迹误差仍然是主要误差源，影响最终洪水预测。

---

## 171. AcCoRD: Evaluating User-Agent Collaboration Under Realistic User Preference Dynamics

**arXiv ID:** 2608.27818 | [PDF](https://arxiv.org/pdf/2608.27818v1)

**作者:** Tejas Srinivasan `[一作]` (University of Southern California), Jesse Thomason `[通讯]` (Georgia Tech University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为《**PreferenceDynamics**》的用户-代理协作基准，涵盖四种偏好动态（硬/软未指定、不可实现、触发），并在在线购物（Shop）和旅行规划（Travel）两大任务域内进行评测。

**💡 创新点**

创新点在于：①系统化地定义并统一衡量四类偏好动态，②通过构造带有动态偏好的可控场景显著提升了基准的现实感；③在传统ReAct框架上引入UncReAct，尝试在每一步显式地判断是否需要进行不确定性澄清。

**🔧 技术方法**

使用的技术主要包括：基于大型语言模型（LLM）的Agent与用户模拟器、ReAct 与 UncReAct 提示策略、LLM 评估器进行偏好满足度打分，以及对话环境的结构化状态与动作定义。

**📊 数据集**

使用的数据集为自定义的 100 组场景，分别在 WebShop（百万级商品）和 TravelGym（多域旅行规划）中生成；场景从 UserBench、WebShop、TravelGym 迁移并通过 LLM 生成触发与不可实现偏好。

**📈 对比分析**

对比了五种前沿 LLM（Llama‑3.1‑70B、DeepSeek‑V3.2、GPT‑4.1、GPT‑5.1、Claude‑Sonnet‑4.5），在两种提示策略下评估完成率、偏好得分和 Perfect Outcome；结果显示：完成率高（>90%），但偏好满足度仅 0.6‑0.8，Perfect Outcome 低于 30%；UncReAct 并未显著提升性能。

**⚠️ 局限性**

局限性包括：①仅在购物与旅行两个域进行测试，缺乏对其他协作场景的泛化验证；②用户模拟器相对理想化，真实用户可能更噪声和多样；③仅评估任务结果，未考虑用户信任、挫败感等交互体验维度；④目前仅通过提示提升不确定性推理，未采用训练方法，导致提升有限。

---

## 172. CF-YOLO: Context-Aware Feature Refinement for Camouflaged Industrial Micro-Defect Detection

**arXiv ID:** 2608.28070 | [PDF](https://arxiv.org/pdf/2608.28070v1)

**作者:** Xinda Yu `[一作]` (Huzhou University), Jie Liu `[通讯]` (Huzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出实时工业微缺陷检测框架 CF‑YOLO，并通过 Context‑Perception Aggregation Module（CPAM）与 Feature Additive Refinement Module（FARM）解决背景伪装与微尺度缺陷难题；

**💡 创新点**

创新点在于① CPAM 采用大核感知+小核聚合双模，扩展感受野并保留细粒边缘；② FARM 引入线性复杂度加性 Token Mixer 进行全局语义验证；③ 公开了工业铜管缺陷数据集 CTDD；④ 在 YOLOv11 基础上实现兼顾精度与实时性的改进架构；

**🔧 技术方法**

使用 YOLOv11 结构、CPAM（大核深度卷积+小核聚合）、FARM（线性加性 Token Mixer）、跨尺度融合 Neck、解耦 Head、CIoU+BCELoss+DFL 损失、Mosaic/Mixup 数据增强等技术；

**📊 数据集**

采用自行构建的 Copper Tube Defect Dataset（CTDD）1,847 张图、4,898 个缺陷；外部验证使用单类合并的 NEU‑DET 数据集；

**📈 对比分析**

与 12+ 经典与先进基线（Faster R‑CNN、YOLOv11n、ETDNet 等）对比，CTDD 上 AP50 提升 2.2%（0.801→0.823）、Precision 提升 4.1%（0.843→0.882）、F1 提升 2.5%（0.771→0.796），保持实时推理；在 NEU‑DET 上 AP50+0.8%，验证泛化能力；

**⚠️ 局限性**

在资源受限的边缘设备上仍存在计算开销问题，需进一步压缩与量化优化以提升部署效率。

---

## 173. Memristive-Friendly Hadamard Reservoir Computing: Structured, Multiplier-Free Recurrences at Scale

**arXiv ID:** 2608.28295 | [PDF](https://arxiv.org/pdf/2608.28295v1)

**作者:** Andrea Ceni `[一作]` (University of Pisa), Claudio Gallicchio `[通讯]` (University of Pisa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种使用结构化正交Hadamard算子取代传统密集循环矩阵的回声状态网络，兼容忆阻器友好的神经元动力学；

**💡 创新点**

创新点在于将正交Hadamard算子分解为符号对角矩阵、随机排列和快速Walsh-Hadamard变换，实现乘法器无关、O(N log N)运算、O(N)参数，且精确满足回声状态条件；

**🔧 技术方法**

采用了结构化正交算子、Walsh-Hadamard变换、忆阻器动力学模型、岭回归读取层以及三种硬件平台的时延与内存评估；

**📊 数据集**

使用了20个时间序列分类数据集（UEA/UCR）和7个回归数据集（Monash TSER）进行实验；

**📈 对比分析**

通过与密集ESN、简单环路、正交ESN、全训练GRU等基线对比，发现Hadamard结构在保持或略优的准确率（与密集正交相当、优于环路）的同时，在大规模（N=8192）下实现了与密集矩阵相比约50×的速度提升和四个数量级的内存压缩；

**⚠️ 局限性**

局限性包括对非线性动态的分析仍基于全局Lipschitz估计、对实际忆阻器噪声和老化的实验尚未实现、以及在极大尺寸或长序列任务中累积噪声仍可能导致性能下降。

---

## 174. Horizon-Independent Contraction for Continuous-Time Discounted Regularized Mean-Field Games

**arXiv ID:** 2608.27723 | [PDF](https://arxiv.org/pdf/2608.27723v1)

**作者:** Junji Yan `[一作]` (University of Illinois Urbana-Champaign), Tamer Başar `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

本文研究了在连续时间、有限状态与动作的均衡场游戏中，加入折扣因子和熵正则化后，均衡点的收敛性与收敛速率。

**💡 创新点**

创新点在于提出一种与时间无关的收敛条件，并利用正算子谱半径与Bielecki度量精确表述有限与无限期均衡的误差界。

**🔧 技术方法**

主要技术包括正算子理论、谱半径分析、Bielecki加权度量、HJB方程的稳定性估计以及软最小（soft‑min）策略映射的Lipschitz性质。

**📊 数据集**

文章未使用任何实验数据集，全部为理论推导与数值示例。

**📈 对比分析**

通过理论推导给出了有限期和无限期均衡之间的误差上界，并给出折扣与非折扣均衡的差异上界，表明在足够大的折扣率下收敛速率优于无折扣情况。

**⚠️ 局限性**

局限性包括仅适用于有限状态与动作的Markov过程，且对折扣率的要求较高，未讨论非正则化或更一般的动力学。

---

## 175. Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms

**arXiv ID:** 2608.28044 | [PDF](https://arxiv.org/pdf/2608.28044v1)

**作者:** Prabhu Vellaisamy `[一作]` (Carnegie Mellon University), John Paul Shen `[通讯]` (Carnegie Mellon University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大规模语言模型推理在 GPU 平台上的请求能耗与每生成 token 的能耗进行系统性量化与建模，探讨批量、上下文长度、输出长度等请求形状对能耗的影响。

**💡 创新点**

提出将请求能耗拆分为固定预填能耗与逐 token 递增能耗两项，并揭示 token 能耗下降并不必然意味着总体能耗下降，强调同时关注请求能耗与 token 能耗的必要性。

**🔧 技术方法**

使用 NVIDIA NVML 能耗计数器收集 GPU 能耗数据，并通过实验室基准构建线性加性能耗模型，结合 FlashAttention 与 eager 执行、MoE 路由与专家调度等技术。

**📊 数据集**

在实验中使用随机生成的 prompt 进行能耗测量，推理模型包括 Llama‑3.2 系列、OLMoE、Qwen1.5‑MoE 等；在推理精度与能耗对比中使用 MATH‑500 与 ARC‑Challenge 两个公开问答数据集。

**📈 对比分析**

通过与传统的 J/token、平均功率、吞吐量等指标对比，显示在固定输出长度下批量与更长输出可显著降低 token 能耗但会提升请求总能耗，且在不同 GPU（H100 与 H200）与模型类型上表现出可量化的能耗差异。

**⚠️ 局限性**

实验仅覆盖 Hopper 级 GPU 与选定的密集与 MoE 模型，未探讨不同架构、训练阶段或低功耗模式；固定+步长模型在极长输出或高并发场景下可能失效，测量精度受 NVML 时序限制。

---

## 176. Efficient Online Continual Foundation Model Fine-Tuning for Predictive Process Monitoring

**arXiv ID:** 2608.28237 | [PDF](https://arxiv.org/pdf/2608.28237v1)

**作者:** Sjoerd van Straten `[一作]` (Eindhoven University of Technology), Marwan Hassani `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 COMPASS 框架，实现在线持续微调基础模型（如 Tiny‑LLM、DistilGPT2）进行下一步活动预测，解决概念漂移下的冷启动与灾难性遗忘问题。

**💡 创新点**

① 将损失平稳检测与漂移检测结合实现任务边界自动识别；② 引入知识子空间扩展，利用预训练和任务特定方向的正交投影保证前向、后向稳定性与可塑性；③ 在流程挖掘领域首次将 LoRA 与持续微调结合。

**🔧 技术方法**

LoRA 参数高效微调、损失平稳漂移检测、SVD 正交子空间投影与累计、正交子空间初始化、任务自由漂移检测等技术。

**📊 数据集**

九条事件流：五个合成漂移日志（IRO5000、ORI5000、ROI5000、OIR5000、RIO5000）以及四个真实 BPI Challenge 日志（BPI20‑DD、BPI20‑ID、BPI20‑RFP、BPI15‑REC 复合）。

**📈 对比分析**

与 DoNothing、LastDrift、DynaTrainCDD、TFCLPM、CNAPwP 五个基线比较；COMPASS 在 7/9 日志中取得最高准确率，尤其在 BPI15‑REC 上提升 19%，任务自由版与 oracle（已知任务边界）性能相当，计算成本适中。

**⚠️ 局限性**

仅评估下一步活动预测；对语言预训练转化为整数活动的语义迁移未深入验证；未实现在线或自适应超参数调优；未考虑多模态或资源属性扩展。

---

## 177. XHotpotQA: A Benchmark for Cross-Lingual Knowledge Composition in Multi-Hop Question Answering

**arXiv ID:** 2608.27481 | [PDF](https://arxiv.org/pdf/2608.27481v1)

**作者:** Iman Barati `[一作]` (Iran University of Science and Technology), Behrouz Minaei-Bidgoli `[通讯]` (Iran University of Science and Technology)

**通讯引用:** 6048 | [OpenAlex ID](https://openalex.org/A5057087345)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作构建了跨语言多跳问答基准XHotpotQA，并提供了详细的语言分布、评估指标、资源质量检查等补充材料；

**💡 创新点**

创新点在于提出了跨语言知识组合的新基准，并对模型在不同语言和黄金语言敏感性进行了系统评估；

**🔧 技术方法**

采用了多语言NLP工具、基准构建方法、评估指标计算以及精确的提示模板设计等技术；

**📊 数据集**

使用了XHotpotQA数据集，该数据集覆盖多语言、多跳问答对；

**📈 对比分析**

通过与多种模型（如XLM-R、mBERT等）的对比实验，展示了它们在跨语言设置下的准确率、召回率和F1表现，结果显示跨语言性能差异显著；

**⚠️ 局限性**

局限性包括数据量有限导致低资源语言表现欠佳，以及评估方法可能忽略某些语义细节和文化差异。

---

## 178. Why People Share Social Media Screenshots

**arXiv ID:** 2608.27539 | [PDF](https://arxiv.org/pdf/2608.27539v1)

**作者:** Tarannum Zaki `[一作]` (Old Dominion University), Michele C. Weigle `[通讯]` (Old Dominion University)

**通讯引用:** 3385 | [OpenAlex ID](https://openalex.org/A5085719625)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并归纳了用户在社交媒体平台上分享截图的主要动机，提出了六大类别：跨平台分享、为已删除帖子保存证据、聚合多张截图、讽刺/幽默创作、评论与注释以及拒绝直接互动。

**💡 创新点**

创新点在于系统化梳理截图动机并强调其对跨平台传播、信息完整性、叙事构建和交互行为的影响，指出截图在现代社交媒体生态中的多功能与潜在风险。

**🔧 技术方法**

采用定性研究方法，包括文献综述、案例分析和对实际截图示例的手工编码；未使用机器学习或自动化技术。

**📊 数据集**

数据集由作者收集的多平台截图组成，来源包括 Twitter/X、Facebook、Instagram、Telegram 等公开可获取的社交媒体帖子示例；数据量以案例为主，缺乏大规模量化样本。

**📈 对比分析**

本研究未进行定量对比实验，主要通过案例对比阐述不同动机的表现与效果；若有比较，则以案例说明为主，缺乏可度量的性能指标。

**⚠️ 局限性**

局限性包括：样本规模有限且偏向可见的公开案例；缺乏系统化的量化验证；难以界定讽刺/幽默与故意误导的界限；对截图真实性与可验证性的探讨不够深入。

---

## 179. From Perspective to Fisheye Depth Estimation and Open-Vocabulary Segmentation

**arXiv ID:** 2608.27860 | [PDF](https://arxiv.org/pdf/2608.27860v1)

**作者:** Rit Gangopadhyay `[一作]` (Yale University), Alex Wong `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Distortion Extenders (DEX)，通过在特征空间学习轻量级调制器，使预训练的视角模型能够在鱼眼图像上实现高精度深度估计和开词汇分割。

**💡 创新点**

无需重新训练骨干网络或显式几何校正，DEX在自监督方式下通过软选择和低秩变换将鱼眼特征对齐至视角特征分布，兼容卷积与Transformer架构，适用于多任务。

**🔧 技术方法**

自监督对齐损失、软选择矩阵、低秩分解、鱼眼仿真与逆变换、Spherical坐标转换以及线性解码预测畸变系数等技术。

**📊 数据集**

训练使用视角图像数据集NYUv2、VOID、IRS、Hypersim、Waymo；评估在真实鱼眼数据集ScanNet++、KITTI-360、WoodScape。

**📈 对比分析**

与MiDaS、DepthAnything、UniDepthV2、VNL、UniK3D、LoRA、Calibration Tokens等基线比较，DEX在ScanNet++和KITTI-360的RMSE下降约10–20%，δ1提升8–15%；在WoodScape的mIoU提升约5–10%。

**⚠️ 局限性**

依赖骨干模型的输出质量，需挑选合适的训练集；对极端畸变的恢复仍有限；畸变系数解码精度不高，且只针对仿真畸变，真实场景可能存在偏差。

---

## 180. Evaluating Loss Functions in Differentiable Out-of-Domain Sound-Matching with Partial Parameter Distance

**arXiv ID:** 2608.27698 | [PDF](https://arxiv.org/pdf/2608.27698v1)

**作者:** Amir Salimi `[一作]`, Osmar R. Zaïane `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了在不同离域（OOD）声学匹配场景下四种可微分损失函数的效果，并提出了部分参数距离（PPD）作为自动评估方法。

**💡 创新点**

引入PPD来评估不共享参数空间的离域合成器匹配，证明PPD与听觉评估一致，并系统验证了损失函数随合成器类型的依赖性。

**🔧 技术方法**

使用可微分JAX合成器、FAUST转JAX、RMSProp梯度下降、Soft‑DTW、JTFS、SIMSE、对数STFT谱等损失，并通过Bootstrap和Scott‑Knott非参数排名。

**📊 数据集**

使用7个简易合成器对（6个OOD+1个域内）生成的1秒48kHz音频，每个场景300次随机初始化实验；听觉测试随机抽取40对，4名评审进行Likert评分。

**📈 对比分析**

通过PPD与Likert评分的Bootstrap分布进行NPSK排名；在5/7情境PPD与听觉结果一致，损失函数表现随合成器不同而变化，Spectrogram基损失在滤波、AM等场景表现最佳。

**⚠️ 局限性**

PPD仅适用于单一主导参数的简单合成器，关键参数选择主观；听力测试人数少且作者参与；结果受梯度优化设置影响，未验证对更高维度或真实录音目标的扩展。

---

## 181. PhenoIntel: A Lifecycle-Aligned Multi-Agent Web Application for Verified, Accessible Plant Phenotype Analysis

**arXiv ID:** 2608.27999 | [PDF](https://arxiv.org/pdf/2608.27999v1)

**作者:** Narendren S `[一作]` (VIT Bhopal University), Soumyashree Kar `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了PhenoIntel——一种生命周期对齐、九个专用智能体协同、具备自动校验、可视化、无GPU运行的植物表型化网页平台，能够将图像转换为可靠的表型测量、置信区间和统计报告。

**💡 创新点**

创新点在于：① 采用多智能体架构和全流程自动检查（V0–V7）防止错误传播；② 为每种模型类别选择匹配的置信度量化方法（RAPS、检测置信度、Monte Carlo Dropout）；③ 通过模型注册表和自适应扩展层实现无缝新增作物/任务；④ 统一的强类型共享状态和沙盒化统计，满足FAIR合规；⑤ 通过浏览器无GPU执行提升可访问性。

**🔧 技术方法**

技术包括：大模型（Llama‑3）规划、LangGraph编排、MLflow‑风格的分阶段校验、RAPS conformal prediction、检测置信度扩散、MC Dropout、量化统计、Docker沙盒、FAIR包生成。

**📊 数据集**

使用的数据集涵盖 5 种农作物（稻、麦、玉、蕉、咖啡）和 4 种成像模式（RGB close‑range、UAV、卫星），如稻氮缺乏（≈5.8k）、麦营养缺失（≈3k）、玉稻缺乏（≈17k）、蕉微量营养（≈1.5k）、咖啡缺乏（≈800）、麦穗检测、稻穗检测、卫星时间序列等。

**📈 对比分析**

与先前单一LLM调度器对比，PhenoIntel 在 6 个跨域案例研究中实现 100% 完整流水线完成率，分类 Macro‑F1 0.78–0.996，检测 mAP50 0.96，时间序列宏观F1 0.70，且每个输出均附置信区间、假设检验和FAIR元数据，性能优于或等同于传统工具，同时大幅提升可靠性和可审计性。

**⚠️ 局限性**

局限性包括：零样本分割（zero‑shot）在重叠叶片时准确度仍偏低；自适应生成模型仍需人工审查和后续注册；部分置信度量化方法在极端数据分布下的校准尚未充分验证；依赖外部LLM调用导致网络延迟和成本；整体推理速度受限于CPU 仅支持轻量级模型。

---

## 182. Focus Where It Counts: A Salience-Driven Vision-Language Model for Low Vision Assistance

**arXiv ID:** 2608.28218 | [PDF](https://arxiv.org/pdf/2608.28218v1)

**作者:** Jiazhao Liang `[一作]` (New York University), Yi Fang `[通讯]` (New York University Abu Dhabi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了面向低视力用户的 salience‑driven 视觉‑语言模型 Salience‑LLaVA，并创建了三套基于低视力需求的 salience‑aware 数据集（Salience COCO、Salience Flickr、Salience VizWiz），在此基础上构建并评估了该模型，最终将其部署在可穿戴智能眼镜上实现实时语音反馈。

**💡 创新点**

创新点主要包括：① 将人类视觉注意力信息（salience）显式嵌入 VLM 生成流程，使模型按重要性顺序描述场景；② 设计并发布了三套标注包含对象级 salience 的数据集；③ 引入了新的评价指标 Salience Coherent Match Index (SCMI) 以及 Wu‑Palmer 版本，用以量化生成文本中对象顺序与人类优先级的一致性；④ 将该模型与硬件结合，演示了在实际可穿戴设备中的可用性。

**🔧 技术方法**

技术上采用了 CLIP‑ViT 视觉编码器、QAGNet 多尺度 salience 分支、Vicuna LLM（基于 LLaVA 结构），并通过 LoRA 微调只训练 salience 融合的 MLP；同时使用了基于图神经网络的 QAGNet 进行 context‑aware salience 排序，并将 salience 特征与原始视觉特征融合后送入 LLM 生成 caption。

**📊 数据集**

使用的数据集包括：1）Salience COCO（6,636 训练 + 1,659 测试）; 2）Salience Flickr（6,980 训练 + 1,745 测试）; 3）Salience VizWiz（779 训练 + 266 测试）。这些数据集是在原始 COCO、Flickr30K 与 VizWiz 的基础上，结合 QAGNet 生成的 salience 掩码并由 Janus Pro‑7B 按 salience 顺序生成描述；此外还使用了低视力用户参与的验证集（120 张图片）来评估标注质量。

**📈 对比分析**

实验与基线比较：与 DeCap、GRIT、Tag2Text、原始 LLaVA、LLaVA‑FT、SCOPE、WalkVLM、Gemma4‑2Shot、Qwen3.6‑2Shot 等方法对照，使用 BLEU、CIDEr、METEOR、ROUGE 以及 SCMI/Wu‑Palmer SCMI 评价。结果显示，Salience‑LLaVA 在三套数据集上均获得最高或接近最高的 SCMI（最高 0.83），并在传统文本质量指标上有显著提升（BLEU‑4、CIDEr、METEOR、ROUGE）。在低光环境与边缘设备实验中，模型虽出现一定性能下降，但仍保持比基线更高的 salience 顺序一致性。

**⚠️ 局限性**

限制：① 计算量大，需服务器推理，移动端实时性能受限；② 在低分辨率或低光图像中 salience 排序准确率下降，导致 SCMI 下降；③ 仍依赖特定 salience 提取器（QAGNet），不同模型的 salience 排序一致性差异显著；④ 评估指标仍以词表匹配为主，命名差异会影响分数。

---

## 183. DART-FL: Burst-Aware Multitask Federated Learning under Dynamic Inference Demand at the Edge

**arXiv ID:** 2608.27713 | [PDF](https://arxiv.org/pdf/2608.27713v1)

**作者:** Yiming Xie `[一作]` (Northeastern University), Ningfang Mi `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在边缘设备上同时支持在线推理与多任务联邦学习的SLO感知、需求驱动双层调度框架。

**💡 创新点**

创新点在于将推理需求同时映射到推理-训练资源划分与任务级训练优先级的双层调度，并通过队列感知的DPP启发式调度动态调整损失权重。

**🔧 技术方法**

采用了SLO-aware资源分配、Drift‑Plus‑Penalty（DPP）队列感知调度、动态损失权重映射以及FedAvg多任务联合学习等技术。

**📊 数据集**

使用了Stanford Cars和Oxford Flowers 102两个视觉任务的数据集，搭配共享ResNet backbone进行实验。

**📈 对比分析**

与Round Robin和Burst‑Aware FIFO两种基线进行对比，在合成与Alibaba真实轨迹负载下，框架在高需求任务的推理准确率提升约5%–12%，同时保持整体多任务性能不下降。

**⚠️ 局限性**

局限性包括仅在两任务、共享backbone和相对简单客户端场景下验证，未考虑更大规模、多异质客户端与动态任务的复杂性。

---

## 184. A Deep Learning-Based Stacking Ensemble Framework for Turbofan Engine Remaining Useful Life Prediction

**arXiv ID:** 2608.27940 | [PDF](https://arxiv.org/pdf/2608.27940v1)

**作者:** Limon Bin Hossain `[一作]` (Bangladesh University of Engineering and Technology), Md Sharifuzzaman `[通讯]` (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种双层堆叠集成框架，用于预测涡轮风扇发动机的剩余寿命（RUL）。

**💡 创新点**

创新点在于将多种异构深度学习基模型（LSTM、CNN、CNN–LSTM、CNN–GRU）与XGBoost元学习器结合，利用堆叠泛化提升预测准确性，同时系统化地进行特征筛选、归一化和分段线性RUL标注。

**🔧 技术方法**

采用了深度递归网络（LSTM/GRU）、一维卷积网络、混合CNN–递归网络、XGBoost回归器，以及5折交叉验证的OOF（out‑of‑fold）特征生成和训练流程。

**📊 数据集**

使用NASA C‑MAPSS基准数据集的FD001和FD003子集（单条件、单故障模式）进行实验验证。

**📈 对比分析**

在FD001和FD003上，堆叠集成分别实现RMSE 9.989/8.613、MAE 7.081/5.195、R² 0.899/0.906，较目前最佳TCAT模型的RMSE分别降低10.2%和21.8%，显著提升预测精度。

**⚠️ 局限性**

局限性包括：仅在单条件子集上验证；未加入注意力/Transformer基模型；对真实工业数据的泛化能力和不确定性量化尚待进一步研究。

---

## 185. When Evidence Shapes Collaboration: Knowledge-Conditioned Topology Generation for Multi-Agent Systems

**arXiv ID:** 2608.27984 | [PDF](https://arxiv.org/pdf/2608.27984v1)

**作者:** Yangxiao Jiang `[一作]` (Huazhong University of Science and Technology), Xiaojin Zhang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于检索证据先行的多代理协作框架K‑GAT，能够根据检索到的外部证据自适应生成协作拓扑并在执行过程中进行知识检验；

**💡 创新点**

创新点在于：①将检索与拓扑生成耦合，采用知识驱动的自回归图生成和KG‑Verifier；②通过候选拓扑挖掘、执行评估与课程学习实现无监督的拓扑学习；③实现结构与证据的动态匹配，缓解传统“计划先行、检索后续”的结构不匹配问题；

**🔧 技术方法**

使用Qwen‑3‑8B作为基座LLM，MiniLM‑L6用于检索，采用自回归图生成模型与结构约束，KG‑Verifier用于知识一致性检查，结合Curriculum学习和结构剪枝；

**📊 数据集**

在七个知识与符号推理基准上进行评估：MMLU、MMLU‑Pro、GPQA、StrategyQA、GSM8K、AQuA、HumanEval；

**📈 对比分析**

与单体LLM、静态拓扑（Chain/Star/Tree）以及多代理框架（LLM‑Debate、AgentPrune、AgentDropout、AFlow、G‑Designer）比较；K‑GAT在8B规模下平均78.68%准确率，GPQA上提升+15.7%，并在Token消耗上比LLM‑Debate低50%以上；

**⚠️ 局限性**

依赖外部知识图的质量与覆盖，检索噪声或不完整会导致拓扑不佳；仅支持固定节点预算（N_max=6），对长链复杂推理可能不足；未针对工具使用或动态知识源扩展。

---

## 186. WeAgent-MMSearch: Native Text-Vision Interaction for Multimodal Search Agents

**arXiv ID:** 2608.28062 | [PDF](https://arxiv.org/pdf/2608.28062v1)

**作者:** Zongkai Liu `[一作]` (Weixin AI, Tencent), Fandong Meng `[通讯]` (Weixin AI, Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于多模态搜索的 WeAgent-MMSearch 系统，实现了持久图像重回馈与运行时恢复，并通过专家数据和 RL 进一步提升。

**💡 创新点**

创新点包括：① 在工具交互中持续存储并复用检索图像；② 设计了基于 FA-GSPO 的失败感知 RL，能够自恢复并剔除无效轨迹；③ 统一了多模态轨迹语法，构建了完整的多模态任务生成与验证管线。

**🔧 技术方法**

主要技术包括：多模态大语言模型（如 Kimi K2.6、Qwen3-VL），ReAct 样式工具调用，Group Sequence Policy Optimization（GSPO）以及其改进版 FA‑GSPO；同时使用缓存、错误回溯与预算控制实现可靠交互。

**📊 数据集**

使用的数据集：内部 3.3K SFT、4.9K RL 轨迹，外部 56.2K SFT、8.3K RL（Metis、OpenSearch‑VL、REDSearcher 等），以及 150 题目的人类验证多模态基准。

**📈 对比分析**

在 8 个公开基准（MMBrowseComp、MMSearch、MMSearch‑Plus、SimpleVQA、VDR‑Bench、LiveVQA、FVQA、WeAgent‑MMSearch‑Bench）上对比，WeAgent‑MMSearch‑RL 达到 55.97% 的平均分，明显优于同规模开源模型，且与参数量约为 10 倍的前沿模型相当。

**⚠️ 局限性**

局限性包括：对目标图像匹配的准确率仍未达到 100%，在极端长回合或超时场景下仍可能出现性能下降；系统对工具接口和缓存策略高度依赖，部署复杂度较高；以及在极少数前沿模型面前仍存在性能差距。

---

## 187. Not to Break, but to Attest: Adversarial Probes for Privacy-Preserving LLM Verification

**arXiv ID:** 2608.27954 | [PDF](https://arxiv.org/pdf/2608.27954v1)

**作者:** Cameron Wilding `[一作]` (Worcester Polytechnic Institute), Fatemeh Ganji `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于隐式探针的审计框架，用对抗性探针放大模型漂移，并通过 Groth16 zkSNARK 对已批准模型在部署后是否保持一致进行隐私保护的完整性验证。

**💡 创新点**

创新点在于：①将对抗性风格的敏感探针与零知识证明结合，首次实现对LLM隐藏修改的安全审计；②提出多种探针族（token、embedding、MLP/attention 触发式）并通过实验确定最具通用性的 token 探针；③采用 Poseidon Merkle 树 + Groth16 的压缩证明，使审计可扩展且证书极小；④在多 GPU、多模型上验证探针的稳健性。

**🔧 技术方法**

核心技术包括：探针生成与 logit 漂移度量；Token、Embedding、Stress 探针族设计；Poseidon 哈希和 Merkle 树构造；Groth16 zkSNARK（circom + snarkjs）证明与验证；NVIDIA GPU（RTX A4000、A100、RTX A6000、L40S、H100、H200）上的推理；HuggingFace Transformers 进行模型推理。

**📊 数据集**

使用两大模型：meta‑llama/Llama‑3.2‑1B 和 Qwen2.5‑1.5B‑Instruct；在这些模型上实施 LoRA+Gaussian、Late MLP、Early Attention 等三种扰动；通过多 GPU 平台（A4000、A100、RTX A6000、L40S、H100、H200）进行跨硬件评估。

**📈 对比分析**

比较方法：对各探针族计算 Δ(x) 分布并绘制箱线图，选取 top‑K 探针；在 Groth16 级别测量 R1CS 约束数、证明/验证时间、witness 大小以及整体审计时长。实验结果显示证明时间随 K 从 1 增至 50 只增至 1.8 s，验证始终约 0.84 s；总体审计时间在 2–4 s 之间，K=10 时约 2.4 s。跨 GPU 与跨模型评估表明探针选择的稳定性和可迁移性。

**⚠️ 局限性**

局限性：仅在两小模型与有限扰动类型上评估；探针与基准需针对每个模型/设备单独校准，无法直接迁移；未对抗适应性攻击（如攻击者提前获知探针生成过程）提供正式安全证明；缺乏针对更大、更异构模型的实验证明和正式威胁模型下的检测保证。

---

## 188. Revisiting Continuous Noise Sampling for Multi-Party Differential Privacy

**arXiv ID:** 2608.27766 | [PDF](https://arxiv.org/pdf/2608.27766v1)

**作者:** Yucheng Fu `[一作]` (University of Virginia), Tianhao Wang `[通讯]` (University of Virginia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多方安全计算中，重新审视连续噪声采样并发现 sample-and-scale 方案在固定点表示下泄露敏感信息，并提出基于离散 Laplace/高斯噪声的并行安全采样器。

**💡 创新点**

提出针对固定点稀疏输出泄露的攻击模型，证明该漏洞可导致几乎 100% 的攻击成功率，并给出对策；同时设计高效并行的离散噪声采样协议。

**🔧 技术方法**

采用基于偏置硬币的离散 Laplace/高斯采样、并行硬币翻转、秘密共享与布尔门混合、MP‑SPDZ 与 garbled circuit 对比等技术。

**📊 数据集**

使用 Orchard、DP‑BREM+、MNIST、k‑means 等公开数据集进行实验验证。

**📈 对比分析**

与传统 sample‑and‑scale、离散采样基线及 lookup table 方案比较，显示在 WAN 设置下速度提升 4~700 倍、通信近似二进制共享、在线轮次大幅减少，且在安全性与准确率上与理想机制持平。

**⚠️ 局限性**

局限在于需在固定点上截断噪声尾部，且对不同安全模型（semi‑honest vs malicious）实现仍需进一步优化；对极高精度浮点实现仍存在挑战。

---

## 189. NormasTCU --- A Brazilian Portuguese IR Dataset and an Evaluation of LLM-as-a-Judge for Relevance Assessment

**arXiv ID:** 2608.27746 | [PDF](https://arxiv.org/pdf/2608.27746v1)

**作者:** Leandro Carísio Fernandes `[一作]` (Câmara dos Deputados), Edans Flávius de Oliveira Sandes `[通讯]` (Tribunal de Contas da União)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了巴西葡萄牙语法律检索测试集NormasTCU，并使用LLM‑as‑a‑Judge评估其在相关性判定上的可行性。

**💡 创新点**

①首次公开发布NormasTCU（14 469条规范文件、46条真实查询、4位专家三等级判定）；②在非英语专业领域系统性检验LLM在相关性评估中的偏差与对IR系统排名的影响，发现对nDCG@10/MRR高度一致，对P@10/R@10不稳定。

**🔧 技术方法**

采用DeepSeek‑V3.2、gpt‑5‑mini和sabiazinho‑4三大LLM，配合simple与rationale两种prompt；构建15个检索系统（BM25变体、语义检索、重新排序等），用P@10、R@10、nDCG@10、MRR评估排名，并通过Kendall τ、Spearman ρ与Bootstrap置信区间进行比较。

**📊 数据集**

NormasTCU：14 469条巴西联邦会计法院公开规范文件、46条真实用户查询、3 048条四位专家的三等级判定（及其匿名个体判定），以及对应的LLM生成问档对。

**📈 对比分析**

在问档对级别上通过ME/MAE、Cohen κ衡量LLM与人类的偏差；在系统排名上计算Kendall τ/ Spearman ρ，结果显示LLM在nDCG@10和MRR上达τ≥0.90，P@10/R@10不满足此阈值；prompt差异不大，simple略优。

**⚠️ 局限性**

仅含46个查询，模型与prompt组合有限（6种），使用API导致可复现性受限；只针对法律规范文件，可能不适用于其他法律文本；P@10/R@10的排名不稳定，需更大查询集验证。

---

## 190. WALDO: One-Shot Exemplar-Conditioned Object Detection in Cluttered Scenes

**arXiv ID:** 2608.28216 | [PDF](https://arxiv.org/pdf/2608.28216v1)

**作者:** Kishor Datta Gupta `[一作]` (Clark Atlanta University), Mohd Ariful Haque `[通讯]` (Clark Atlanta University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出WALDO，一个轻量级的一次性检测头，利用冻结的V-JEPA 2.1视觉特征和文本编码，结合实例图像和简短描述，既定位目标又判断其是否存在；

**💡 创新点**

创新点在于：①用冻结的世界模型特征而非从头训练，极大降低成本；②在训练时通过采样负样本和区域尺度混合，避免shortcut学习；③将实例匹配与缺失判断融合为一个训练目标；

**🔧 技术方法**

技术包括：冻结V-JEPA 2.1 ViT-L/16与SigLIP2文本塔、实例-文本双向FiLM调制、基于余弦相似度的关联通道、Transformer块、聚合的Presence头；训练采用采样episodes、焦点损失与GIoU回归；

**📊 数据集**

使用自建数据集：220个密集场景（停车场与飞机库），共计763个实例框，标注了两张目标车型/飞机照片；

**📈 对比分析**

与Grounding DINO、OWLv2等冻结的开源开放词检测器进行比较；在catalogue检索任务中WALDO以V-JEPA为骨干取得AP@50=0.461、Success@1=0.629，超过Grounding DINO的0.306；在缺失检测中V-JEPA的AUROC=0.880，优于DINOv3和SigLIP2；但在实例级检索（Success@1）仅为0.190，接近随机水平；

**⚠️ 局限性**

局限性：无法真正区分同类别中的不同实例，定位精度受分辨率和实例大小限制；依赖冻结骨干，缺乏进一步微调；在实例级检索中表现不佳，需更细粒度的特征或更丰富的实例标注。

---

## 191. There and Back Again: Bidirectional Diffusion Bridges for Multimodality Translation

**arXiv ID:** 2608.27885 | [PDF](https://arxiv.org/pdf/2608.27885v1)

**作者:** Gabe Guo `[一作]` (Stanford University), Stefano Ermon `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种双向图像-文本扩散桥模型，能够从文本生成图像并从图像逆向生成文本。

**💡 创新点**

创新点在于通过从源模态直接构造 SDE 桥梁，保持源信息且实现可逆生成，统一了 T2I 与 I2T 的生成框架。

**🔧 技术方法**

技术上使用随机微分方程与 Girsanov/Doob h‑变换构建扩散桥，结合 Transformer 网络与 CFG 引导进行训练与采样。

**📊 数据集**

实验数据来源于约 1 亿张 GPIC 文本-图像对，图像预处理为 Stable Diffusion VAE 潜空间，文本使用 Qwen 嵌入。

**📈 对比分析**

与噪声‑到‑数据的扩散模型及基于流匹配的 ODE 基线进行对比，实验显示在多项视觉‑语言和科学任务上性能相当或更优。

**⚠️ 局限性**

局限性包括对非联合支持区间输入缺乏理论保证、端点几何对桥性能的影响尚未系统评估，以及在更广泛多模态场景中的推广性待进一步验证。

---

## 192. PCBnet: A Dataset and Automatic Construction of SPICE Netlists from Schematic Images

**arXiv ID:** 2608.27923 | [PDF](https://arxiv.org/pdf/2608.27923v1)

**作者:** Zhen Huang `[一作]` (Eastern Institute of Technology), Lei He `[通讯]` (Eastern Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 PCBnet 大规模原理图-网表数据集，并设计了端到端的视觉识别 + 拓扑构建 + 多代理纠错框架，实现从 PCB 原理图图像自动生成 SPICE Netlist。

**💡 创新点**

创新点：①首个包含 300+ 实例、50k+ 组件、150k+ 导线、100k+ 文本的 PCB 原理图-网表对数据集；②将目标检测、语义识别、图结构推断与基于领域知识的多代理纠错结合的全流程框架；③通过 LLM + 视觉语言模型 + 组件语义约束实现细粒度文字纠错。

**🔧 技术方法**

技术：YOLOv11 组件/文本检测、PaddleOCR/TrOCR 文本识别、U‑Net+骨架化导线分割、几何与规则匹配构建电路图、基于距离+类别一致、几何对齐、局部唯一的文本匹配、门控 LLM 纠错、视觉语言模型视觉纠错、组件语义约束校正。

**📊 数据集**

数据集：PCBnet（300+ 真实 PCB 原理图，配套 SPICE Netlist，含 50k+ 组件、150k+ 导线、100k+ 文本、400k+ 字符），对比通用多模态模型（GPT‑5.4、Claude Opus 4.6、Gemini 3 Pro 等）。

**📈 对比分析**

评价方法：在 8:1:1 划分的 PCBnet 上，评估组件检测 mAP、文本识别 CER/WER/ACC 及连通性准确率。实验结果显示：组件检测 94.54% mAP、文本识别 98.57% ACC、连通性 84.47%；相比通用多模态模型仅 8% 连通性、30% 组件准确率，提升显著。

**⚠️ 局限性**

限制：对极端噪声或密集手绘原理图的鲁棒性有限；多代理纠错主要针对文本错误，对结构层面的误连或符号误判尚未充分覆盖；在非标准符号或新型组件的迁移性能待进一步验证。

---

## 193. Resource Constraints and Performance in Agentic AI Systems

**arXiv ID:** 2608.27886 | [PDF](https://arxiv.org/pdf/2608.27886v1)

**作者:** Amaz Salman `[一作]` (Massey University), Teo Susnjak `[通讯]` (Massey University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对OpenClaw与NanoBot两套完整的智能体系统进行对比评估，探讨其在任务完成率与资源占用上的差异。

**💡 创新点**

创新点在于引入双层证据体系，结合完整系统的任务结果与操作成本，揭示同一任务在不同证据层级下的结果不一致性。

**🔧 技术方法**

使用技术包括基于GPT-4o-mini的语言模型、工具调用框架、内存与状态管理、并行化执行及结果验证机制。

**📊 数据集**

数据集为100个任务的主基准以及23个细化执行子集，覆盖短、中、长三类执行时序。

**📈 对比分析**

比较方法为配对风险差异与McNemar检验，结果显示两系统在主基准上未出现显著完成率差异，但NanoBot在细化层面表现出更低的壁时与峰值内存并略占优势。

**⚠️ 局限性**

局限性包括仅评估一次运行、缺乏独立评审、资源指标仅包含壁时与峰值内存、无法区分架构与配置差异导致的性能差异。

---

## 194. LandingAgent: A Reference-Annotated Dataset and Agentic Generation Framework for Landing Pages

**arXiv ID:** 2608.27902 | [PDF](https://arxiv.org/pdf/2608.27902v1)

**作者:** Injun Baek `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究基于目标的、引用引导的登录页生成，并提出一种三阶段代理框架

**💡 创新点**

1）引入LandingBench数据集，将真实登录页抽象为可复用的结构、修辞与视觉参考；2）设计LandingAgent框架，将生成过程拆分为Profiling、Wireframing与Polishing，分阶段提升结构一致性与细节质量；3）采用参考检索与迭代批评机制实现目标语义与视觉一致的定制页面

**🔧 技术方法**

大型语言模型（LLM）生成代码、检索增强生成（RAG）、代理模型（Profiler、Retriever、Builder、Critic、Polisher）以及多轮交互式反馈

**📊 数据集**

LandingBench（438页，含结构、语调、视觉等参考标签）

**📈 对比分析**

与直接提示、一次提示以及直接提示+Polisher等基线对比；在faithfulness、conciseness、readability、aesthetics、diversity等指标上，LandingAgent取得显著提升，用户研究也显示显著偏好

**⚠️ 局限性**

缺乏真实转化率评估；参考集覆盖有限，可能对低覆盖行业或语言产生不佳匹配；实验样本规模有限，未覆盖所有可能场景

---

## 195. Stay Within Your Bounds: Distance-Guided Decoding for Guaranteed Context-Free Grammar Compliance

**arXiv ID:** 2608.28229 | [PDF](https://arxiv.org/pdf/2608.28229v1)

**作者:** Vincenzo Collura `[一作]` (University of Luxembourg), Maxime Cordy `[通讯]` (University of Luxembourg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型生成结构化输出时，提出了基于PDA的lookahead‑guided解码框架，保证输出完全符合给定上下文无关文法。

**💡 创新点**

创新点在于结合离线bounded pushdown summary与token级距离估计，实现预算感知的剪枝与重排序，并通过堆栈高度限制与探索预算提供可证明的语法安全。

**🔧 技术方法**

使用了Pushdown Automata、加权Pushdown System、堆栈摘要、距离估计、Beam Search、tokenizer‑aware消耗等技术。

**📊 数据集**

实验数据集包括JSON schema生成（100个样本）、Spider SQL（1,034个样本）和LTL规划（6,185个样本）。

**📈 对比分析**

与Base LM、Sample‑Verify、Guidance、Outlines、XGrammar、SynCode、GenLM等方法对比，SWYB在所有模型与任务上实现100%语法正确性，并在SQL执行准确率和LTL语义准确率上显著提升，解码时间中等。

**⚠️ 局限性**

局限性包括堆栈高度H和探索预算B的限制，预计算成本随语法复杂度上升，距离估计是上界可能过度保守，且仅提供语法约束，未覆盖语义约束。

---

## 196. Antipatterns in AI-assisted Qualitative Data Analysis: A Catalog of Temptations and Pitfalls for Software Engineering Researchers

**arXiv ID:** 2608.27927 | [PDF](https://arxiv.org/pdf/2608.27927v1)

**作者:** Rashina Hoda `[一作]` (Monash University), Rodrigo Spinola `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了人工智能辅助定性数据分析（AI-assisted QDA）的十一种反模式（antipatterns），并按“危险驱动、操作失误、分析失败”三类归纳，提供了研究者和评审者在使用AI进行定性分析时的识别与规避框架。

**💡 创新点**

创新点在于：①首次将AI辅助QDA的常见方法缺陷系统化为可操作的反模式目录；②通过理论与实践结合的方式，阐明了从动机到执行再到结果的层级危害，揭示了不同层次反模式之间的级联与叠加效应；③为同行评审提出了可检验的诊断语言和审稿建议。

**🔧 技术方法**

技术主要包括大型语言模型（LLM）、生成式AI、智能代理等人工智能工具，用于编码、主题生成、摘要、分类等定性分析步骤；同时引用了现有CAQDAS（如MAXQDA、NVivo、ATLAS.ti）与新兴的AI集成平台。

**📊 数据集**

文章未针对单一公开数据集进行实验，而是通过文献综述、案例分析与社区讨论总结常见风险，引用的软件工程定性研究样本（访谈记录、团队沟通日志、代码评论等）作为示例。

**📈 对比分析**

由于该研究属于概念性框架和理论阐述，未进行传统意义上的性能对比；作者通过对比已有的风险评估表（如表1）与已报道的经验教训，指出不同反模式的潜在后果和可能的缓解策略，但未给出定量指标。

**⚠️ 局限性**

局限性包括：①反模式目录可能不涵盖未来AI技术演进所引入的新问题；②缺乏大规模实证验证，仅基于文献与经验总结；③对不同学科背景的适用性需进一步探索；④对特定AI工具的细节（如prompt设计、版本管理）尚未给出统一标准。

---

## 197. Cut-ViT: Task-Specific Model Pruning via Gram Anchoring Subspace Consistency

**arXiv ID:** 2608.28205 | [PDF](https://arxiv.org/pdf/2608.28205v1)

**作者:** Jianjian Yin `[一作]` (Nanjing University of Science and Technology), Wenguan Wang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出了 Cut-ViT，一种面向视觉基础模型的任务特定一次性结构化剪枝方法；通过构建空间与通道的 Gram 锚定矩阵并进行子空间分解，结合基向量无关与残差约束，实现子网络在保持原模型鲁棒特征的同时可快速得到不同稀疏度的子网络。

**💡 创新点**

创新点主要包括：① 以 Gram 锚定的子空间一致性为剪枝依据，替代传统逐点对齐，避免过拟合噪声；② 引入谱熵自适应权重，根据任务对空间与通道信息密度进行动态调节；③ 通过一次性梯度排序实现剪枝速度与显存极大缩减。

**🔧 技术方法**

使用了 Gram 矩阵、SVD 进行子空间提取、基向量无关一致性损失与残差约束、谱熵加权、自监督蒸馏与梯度重要性排名（empirical Fisher）等技术。

**📊 数据集**

实验覆盖 9 个数据集，涉及 6 大视觉任务：ADE20K、PASCAL VOC 2012、COCO、NYUv2、DAVIS‑2017、FG3DCar、JODS、SBD 及 ImageNet。

**📈 对比分析**

与 SnapViT、SNOWS、HydraViT、EA‑ViT 等现有训练‑自由与训练‑基准剪枝方法对比，Cut‑ViT 在视频目标分割、语义匹配、目标检测、语义分割、深度估计与图像分类上均取得 SOTA 或接近 SOTA 的成绩，同时仅消耗 20.9% 的剪枝时间与 45.5% 的 GPU 显存。

**⚠️ 局限性**

局限性包括：在某些任务上仍略逊于最佳训练‑基准方法；对主成分数量 K 与谱熵权重的选取敏感；主要验证在 DINOv3 上，跨模型的泛化尚需进一步探索；极端边缘设备下的鲁棒性评估仍不充分。

---

## 198. CASTANET: Causality-Aware Spatio-Temporal Adversarial Network Using Traffic Incident Effects

**arXiv ID:** 2608.27942 | [PDF](https://arxiv.org/pdf/2608.27942v1)

**作者:** Toshiya Kitahara `[一作]` (Sumitomo Electric System Solutions Co., Ltd), Hisashi Kashima `[通讯]` (Kyoto University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 CASTANET 模型，用于预测突发事故导致的非周期性交通拥堵。

**💡 创新点**

将空间时间图神经网络与因果推断中的平衡表示学习和对抗性表示学习相结合，有效解决事件稀疏与选择偏差问题。

**🔧 技术方法**

采用 Graph WaveNet 作为 STGNN 主干，嵌入事故信息，利用自适应 GCN、QT 层以及反事实域混淆损失实现对抗学习。

**📊 数据集**

使用东京市中心 2,032 条路段的交通与事故数据，时间粒度 5 分钟，覆盖 7 个月。

**📈 对比分析**

与 7 种 STGNN 基线（QTNet、AGCRN、DCRNN 等）及传统模型比较，CASTANET 在整体 RMSE 降低 4%，事故样本 RMSE 降低 10.1%，严重拥堵（Top 5%）提升 14.55%，显著优于现有方法。

**⚠️ 局限性**

仅考虑事故作为突发事件，未检验因果假设的稳健性，模型对极罕见事件仍可能受噪声影响，对其他类型突发事件的推广尚待验证。

---

## 199. Undecidability of Adjacent Equality for Insertion, Shuffle, and Crossover Language Operations

**arXiv ID:** 2608.27755 | [PDF](https://arxiv.org/pdf/2608.27755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

---

## 200. Community corrections have divergent downstream effects across corrected accounts

**arXiv ID:** 2608.27526 | [PDF](https://arxiv.org/pdf/2608.27526v1)

**作者:** Yuwei Chuai `[一作]` (University of Luxembourg), Mohsen Mosleh `[通讯]` (University of Oxford)

**通讯引用:** 3798 | [OpenAlex ID](https://openalex.org/A5052536455)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对 Twitter Community Notes 的准实验研究，跟踪 19,854 账号在 4 周前后被纠正的原始发帖活动，评估社区纠正对作者后续发帖量与内容的影响。

**💡 创新点**

创新点在于将关注点从单个被纠正帖子扩展到作者层面，揭示了单次纠正与多次纠正账号在后续发帖量和内容质量上呈现截然不同的轨迹，并首次量化了多次纠正账号的负向后续效应。

**🔧 技术方法**

采用差分中的差分 (DiD) 方法，结合负二项回归和固定效应线性回归，评估纠正事件对发帖数量、毒性、误导性、政治内容等指标的变化。

**📊 数据集**

使用 Twitter Community Notes 系统公开数据，包含 57,935 条纠正记录、19,854 账号和 11,909,591 条原始发帖，并匹配控制组以实现准实验设计。

**📈 对比分析**

通过与匹配的对照组进行 DiD 对比，发现整体平均每次纠正导致 2.9% 的发帖量增加；但单次纠正账号下降 2.4% 并内容更好，而多次纠正账号则提升 4.4% 并未改善内容质量。

**⚠️ 局限性**

局限性包括：观察性设计仍可能存在时间变动的混杂因素；多次纠正组的划分是事后定义，难以判断因果；仅基于 Twitter 平台，结果可能不具备跨平台推广性；缺乏对作者是否注意到或接受纠正的直接证据。

---

## 201. What Makes Agent Memory Useful for Reliable Unanswerable Question Handling?

**arXiv ID:** 2608.27924 | [PDF](https://arxiv.org/pdf/2608.27924v1)

**作者:** Chuanyuan Tan `[一作]` (Soochow University), Wenliang Chen `[通讯]` (Soochow University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究了代理（agent）内存对不回答问题（UAQ）可靠性的影响，并在统一的agentic RAG框架下进行实验

**💡 创新点**

创新点在于将UAQ处理拆分为决策指导和轨迹塑造两条通道，评估不同记忆类型（描述、规则、程序）及其组合的效果，并对跨数据集与跨模型迁移的可靠性进行系统性分析

**🔧 技术方法**

使用ReAct式交互、四种代表性内存方法（Expel、MemEvolve、AWM、AgentKB）、记忆内容抽象（compact trajectory、principle、insight、success trace、workflow）以及人工提示对照等技术

**📊 数据集**

使用了三大UAQ数据集：KUQ、UAQFact和RefuNQ，分别覆盖多种UAQ类型和场景

**📈 对比分析**

通过与无内存和人工提示基线比较，使用准确率Acc、可接受率AR和联合分数JS作为指标；结果表明内存能提升AR但对Acc影响有限，跨模型迁移相对稳定，跨数据集迁移效果有限

**⚠️ 局限性**

局限性包括：实验仅在固定的Wikipedia RAG环境下进行，未覆盖多工具/动态网页等更复杂场景；记忆表示仅覆盖可拆分的几类，未尝试更复杂的耦合设计；未针对Acc-AR权衡进行专门优化

---

## 202. Dynamic Alignment Compensation for Hallucination Mitigation in Large Vision-Language Models

**arXiv ID:** 2608.28058 | [PDF](https://arxiv.org/pdf/2608.28058v1)

**作者:** Kairong Yu `[一作]` (Zhejiang University), Hongwei Wang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无训练、推理时动态隐藏状态补偿方法DAC，用以降低大型视觉语言模型的幻觉生成。

**💡 创新点**

创新点在于通过检测跨层和跨步的表示偏移，并对其进行轻量级残差补偿，从而在保持生成质量的同时显著降低幻觉风险。

**🔧 技术方法**

技术包括层级语义补偿（LSC）和序列语义校正（SSC），利用特征归一化、Jensen‑Shannon散度阈值和残差调节实现动态修正。

**📊 数据集**

使用了九个多模态基准（如MME、MM‑Vet、MMBench、TextVQA、LLaVA‑Bench、POPE、CHAIR、HallusionBench等）以及多种LVLM骨干（LLaVA‑1.5、Qwen‑VL、Qwen2.5‑VL‑3B/7B、Qwen3‑VL‑4B、LLaVA‑Next‑7B）。

**📈 对比分析**

在所有基准上，DAC均实现了比基线及其他对比方法（如MemVR、ICD、VCD、OPERA等）更低的幻觉率和更高的整体分数，且在推理速度和显存占用上仅增加极小开销。

**⚠️ 局限性**

局限性包括仅针对图像输入的无训练方法，对封闭源模型、视频或多图输入、长上下文、多领域任务的适用性尚未验证；且使用固定阈值和系数，未实现输入自适应；无法保证生成内容绝对事实正确。

---

## 203. Exploring the Design Space of Representation Learning for Audio Transformations

**arXiv ID:** 2608.28127 | [PDF](https://arxiv.org/pdf/2608.28127v1)

**作者:** Sungho Lee `[一作]` (Seoul National University), Yuki Mitsufuji `[通讯]` (Sony AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出统一框架通过三种目标（处理一致性、描述对齐、等变性）学习音频处理感知表示，并产生两种嵌入：变换嵌入(z_T)与处理后音频嵌入(z_y)。

**💡 创新点**

创新点在于把处理感知学习拆解为三类互补目标、同时学习两种嵌入、并在受控实验中系统评估其相互作用；以及构建多样化处理器库与批次构造策略来抑制内容捷径。

**🔧 技术方法**

使用对比损失（contrastive）与投影头、描述编码器（transformer+嵌入）、逆向预测器、前向预测器；网络基于冻结的Stable Audio Open (SAO)编码器+适配器，再加Transformer；训练采用FlashAdamW、余温退火。

**📊 数据集**

训练集：MedleyDB、MoisesDB、Mixing Secrets；评估集：MUSDB、67个Linux插件；使用的处理器库覆盖EQ、滤波、动态、失真、混响、延迟、调制等，链长1–8。

**📈 对比分析**

与AFx-Rep、Fx-Encoder++、Fx-Encoder、CLAP、MERT、VGGish、PANN等基线对比；在跨源检索、链估计、参数估计、风格迁移等任务中，本文模型在检索准确率、probe精度（IoU、F1、MAE）和风格迁移MRSTFT上均显著优于基线，尤其在源匹配差异（跨stem/跨instrument）场景表现突出。

**⚠️ 局限性**

局限在于：①仍受训练音源多样性限制，跨源泛化受限；②需要Fx-normalization等域特定预处理；③描述对齐目标需结构化处理器描述，某些插件缺失；④完全分离处理与内容不可行，仍存在内容泄漏；⑤非盲预训练下的输入估计仍是瓶颈。

---

## 204. EXPOSE: Explainable and Domain-Robust Embeddings from Pathology Vision Foundation Models using Sparse Autoencoders

**arXiv ID:** 2608.28191 | [PDF](https://arxiv.org/pdf/2608.28191v1)

**作者:** Anja Witte `[一作]` (University Medical Center Hamburg-Eppendorf), Marina Zimmermann `[通讯]` (University Medical Center Hamburg-Eppendorf)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

利用稀疏自编码器（Sparse Autoencoder）把 Vision Foundation Model 的密集嵌入转化为稀疏可解释的表示，并通过训练线性域分类器自动识别出与扫描仪/制样相关的域特异维度，随后在复发预测阶段将这些域特异维度设为0，从而在不重新训练 VFM 的前提下提升跨域鲁棒性。

**💡 创新点**

1）将稀疏自编码器作为可解释的瓶颈，用来识别并屏蔽域特异信息；2）无需修改或重新训练 VFM 即可实现跨域性能提升；3）引入 Domain Robustness Index（DoRI）作为嵌入空间的鲁棒性度量。

**🔧 技术方法**

ReLU‑based Sparse Autoencoder、线性域分类器、线性 5 年复发预测器、DoRI 计算；输入嵌入来自 H0‑mini VFM。

**📊 数据集**

内部前列腺癌 TMA 图像数据集，共 24,588 张，按 6 个子域划分（first、scanner、spot、thin、thick、long），其中 first 与 scanner 用于训练，其他四个为 OOD。

**📈 对比分析**

与直接使用 VFM 嵌入加线性分类器的基线相比，mask 256 个域特异维度在 ID 域提升约 1.18 个百分点（AUROC5），在 OOD 域 mask 3500 个维度提升约 0.75 个百分点；同时 DoRI 指标在 ID 与 OOD 均显著上升，说明嵌入空间的域信息被有效削弱。

**⚠️ 局限性**

仅评估单一 VFM 与 ReLU SAE；域特异特征仅基于两域（first、scanner）识别，可能无法覆盖所有域偏移；未尝试更高级 SAE 结构或多域训练；在某些 OOD 子域（如 thin）提升有限，提示仍存在未被捕获的域因素。

---

## 205. Beyond the Vacuum: Combinatorial Strategy Selection for Competitor-Aware Generative Engine Optimization

**arXiv ID:** 2608.27631 | [PDF](https://arxiv.org/pdf/2608.27631v1)

**作者:** Vaibhav Sourirajan `[一作]` (Capital One), Amirfarrokh Iranitalab `[通讯]` (Capital One)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种竞争感知的内容重写策略选择方法，能够在大型语言模型生成回答时提升目标文档的可见性。

**💡 创新点**

创新点在于将GEO转化为竞争感知的组合策略优化问题，并设计了基于BOCS搜索与教师生成推理轨迹的两阶段训练管道。

**🔧 技术方法**

技术方法包括贝叶斯组合结构优化（BOCS）、Hard Negative Mining、教师模型生成对比推理轨迹以及直接偏好优化（DPO）细调。

**📊 数据集**

使用的数据集为原始GEO数据集（约8000个查询-文档组合）以及合成竞争数据集_comp（不同采纳率α下的竞争环境），并在E-Commerce和Researchy-GEO等外部数据集进行零样本迁移验证。

**📈 对比分析**

与15个单策略以及AutoGEO、AgenticGEO基线比较，利用PAWC、Pos Count、Word Count三项印象指标，本文方法在原始和竞争数据集上均取得最高分，并在竞争率提升时衰减最慢，跨域迁移表现亦最佳。

**⚠️ 局限性**

局限性包括：评估依赖LLM生成回答而非真实系统、合成竞争分布可能与真实采纳率不符、策略空间有限、未完整模拟检索-重排流程、以及在保持可解释性与可信度方面仍有提升空间。

---

## 206. UniLipi: A Unified Multi-Script OCR for Historical Indic Manuscripts

**arXiv ID:** 2608.28195 | [PDF](https://arxiv.org/pdf/2608.28195v1)

**作者:** Tathagata Ghosh `[一作]` (International Institute of Information Technology Hyderabad), Ravi Kiran Sarvadevabhatla `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 UniLipi，一种统一多脚本手写印地文手稿 OCR 框架，能够在同一模型中同时识别多种印地脚本并输出罗马化转写、脚本标签和字符计数。

**💡 创新点**

创新点包括：① 融合脚本感知的参考式合成数据生成，显著降低对真实标注的依赖；② 采用统一罗马化词表与多任务学习（脚本识别、字符计数）共同训练；③ 在 Transformer 编码器中引入可学习的脚本与计数任务 token，提升跨脚本表示共享和对行长度的全局感知。

**🔧 技术方法**

技术方案基于 ResNet + Transformer 编码器，CTC 文字头、交叉熵脚本头、MAE 计数头；合成数据采用参考行掩码、字体大小与墨水颜色估计的渲染流水线；训练采用两阶段策略（合成预训练 + 真实微调），使用 Adam 优化器与 BF16 混合精度。

**📊 数据集**

数据集：来自 13 种印地脚本（北中印度 Devanagari、Sharada、Modi；西印度 Gurmukhi、Jaini；东印度 Bengali、Odia；南印度 Grantha、Telugu、Malayalam、Kannada；喜马拉雅/佛教 Newar、Siddham）的手稿行图像（500–3000 行/脚本），以及约 2.5M/脚本的脚本感知合成行图像，形成 32.5M 合成样本；合成后再结合少量真实数据进行微调。

**📈 对比分析**

与 Kraken、PyLaia、VLT、Tesseract、HTR‑VT、HTR‑ConvText、TrOCR 等基线相比，UniLipi 在全部 13 脚本上取得 6.9% 的 CER，显著优于最佳基线 14.3%（Kraken 28.5%）和 19.6%（VLT）。在跨域评测中，Fine‑tune 后在 IAM、RIMES、LAM 等非印地手稿上也能保持 6.9–8.1% 的 CER，展示出良好的迁移能力。

**⚠️ 局限性**

局限性包括：① 对极低资源脚本（如 Modi、Siddham）的性能仍相对较低；② 合成数据虽然逼真，但仍可能与极端破损、图像噪声的真实手稿存在分布差距；③ 依赖统一罗马化词表，对于某些特殊语种的细粒度语义可能不够精准。

---

## 207. Load-Bearing Context: The Question Damage Score for Evaluating Context Reliance in Linguistic Reasoning

**arXiv ID:** 2608.27756 | [PDF](https://arxiv.org/pdf/2608.27756v1)

**作者:** Neh Majmudar `[一作]` (City University of New York), Elena Filatova `[通讯]` (City University of New York)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种针对英国语言奥林匹克Rosetta Stone谜题的单例上下文删除诊断框架，用以探究大型语言模型（LLM）在受限上下文下的推理与放弃行为。

**💡 创新点**

创新点在于构造基于误码纠正码（ECC）思想的定向删除策略以及问题损伤分数D_Q，用来精准标记“负载承载”上下文，并揭示LLM在结构上不可解谜题仍能给出答案的矛盾现象。

**🔧 技术方法**

技术手段包括：whitespace token化的单词覆盖分析、问题损伤分数与基本损伤分数计算、LLM-as-Judge 的语义可解性评估，以及零样本结构化输出的评测流程。

**📊 数据集**

使用的数据集为53道英国语言奥林匹克（UKLO）Rosetta Stone谜题（共计约16条示例和若干问题），并分别生成随机删除与ECC定向删除的两种变体。

**📈 对比分析**

比较方法为对原始、随机删除、ECC删除三种版本分别进行零样本推理，计算Exact Match得分并记录放弃次数；结果显示即使删除关键上下文，LLM几乎不放弃，性能下降不大，甚至在部分案例出现提升。

**⚠️ 局限性**

局限性包括：基于whitespace token的保守损伤估计可能低估可恢复信息、仅考虑单例删除未捕获多例组合效应、数据规模有限导致统计显著性不足、LLM-as-Judge 仍受模型自身偏差影响、未能区分记忆化、语言先验与推理机制。

---

## 208. Text Restoration of Ancient Documents with Language Models

**arXiv ID:** 2608.28170 | [PDF](https://arxiv.org/pdf/2608.28170v1)

**作者:** Shibingfeng Zhang `[一作]` (University of Bologna), Giovanni Colavizza `[通讯]` (University of Bologna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用多种语言模型对古拉丁文公证手稿中的缺失文本进行自动重建，并评估其在已知与未知缺口长度两种情境下的可行性。

**💡 创新点**

首次系统探讨缺口长度未知的重建问题，比较不同模型架构与解码策略对公式化与非公式化文本的影响，并提供可直接供人类学者使用的微调模型与方法论。

**🔧 技术方法**

结合基于掩码语言建模（LaBERTa、PhilBERTa）、序列生成（LaTa、PhilTa、Aeneas）、以及大规模开源LLM（Llama‑70B Instruct、Deepseek‑v4‑pr）等技术，并实现四步解码与长度约束策略。

**📊 数据集**

使用约1184份意大利博洛尼亚地区11‑13世纪公证文档的人工校本数据，构建训练/验证/测试集，并通过合成损伤注入模拟真实缺口。

**📈 对比分析**

通过字符错误率（CER）、重叠得分（Overlap）及前N命中率（HR@N）等指标对比模型。已知缺口长度时Aeneas表现最佳；深度学习模型与LLM在公式化段落上相对较好，非公式化与“rogatio”段落表现最差；缺口长度未知时整体准确率显著下降，LLM在无对齐信息时表现相对可观。

**⚠️ 局限性**

受限于模型对缺口长度的感知、缺乏对公式化与非公式化差异的细粒度建模、以及自动评估指标对可接受替代方案的低敏感度，重建尚未完全自动化，需与学者协同评估。

---

## 209. The Approximation Rank of Softmax Attention: Sharp Geometric Laws and Robust Interaction Dimension

**arXiv ID:** 2608.28150 | [PDF](https://arxiv.org/pdf/2608.28150v1)

**作者:** Yuhe Sui `[一作]` (Nanyang Technological University), Jianing Zhang `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究归一化softmax注意力的最大行ℓ1近似秩，探讨几何形状如何决定其复杂度；

**💡 创新点**

提出支持几何和可见交互几何两种新视角，给出精确的温度律与r/2最优指数，并构造最优下界；

**🔧 技术方法**

运用了最大行ℓ1近似秩定义、Gibbs覆盖、凸几何分析、软最大函数不等式及SVD桥接等技术；

**📊 数据集**

在合成数据上验证理论，并在84头BERT‑base模型的校准集上进行实验；

**📈 对比分析**

与理论下界/上界比较，合成实验匹配预期斜率；BERT实验显示有效维度可约约40%，与SVD上界相关；

**⚠️ 局限性**

局限性包括上界常数难以实用估计、仅在极端Token条件下适用、只在单个模型上校准，未给出稀疏或算术加速等实际效果。

---

## 210. Finite Sample Bounds for Composite Hypothesis Testing

**arXiv ID:** 2608.28068 | [PDF](https://arxiv.org/pdf/2608.28068v1)

**作者:** El{í}as Vera-Sig{ü}enza `[一作]` (Okinawa Institute of Science and Technology), Amedeo Roberto Esposito `[通讯]` (Okinawa Institute of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对复合二元假设检验在有限样本情形下给出了Rényi散度框架下的显式可达性与逆向下界，并解析了指数衰减率下的相变与强逆转现象。

**💡 创新点**

创新点在于：① 用Rényi投影（joint Rényi projection）构造单一阈值检验，在不需要最不利分布的前提下实现全局错误控制；② 推导出复合问题的相变阈值为零点与一端KL投影的最小值，并在有限字母的紧凸类中给出两侧的精确指数；③ 在可达性上提供多项式修正（含log‑n阈值校正），并与强逆向相匹配；④ 阐明在特定随机排序条件下，Rényi投影得到的对偶分布即为最不利分布。

**🔧 技术方法**

主要技术包括Rényi散度与Hellinger积分的凸/凹性质、Sion极小极大定理、Rényi投影与KL投影的关系、弱收敛与连续性论证、Berry‑Esseen与两端尾部估计，以及对多项式阶数与阈值修正的微分分析。

**📊 数据集**

使用人工合成数据集进行数值验证：三元离散分布的非排序凸类、分离的一参数自然指数族（Bernoulli、Poisson、Gaussian、exponential），以及在固定与次指数型Type I约束下的相同三元类。

**📈 对比分析**

与传统的极大似然、GLR检验以及经典Chernoff–Stein、Stein类极限相比，所给定的Rényi上界与下界在有限样本上均较紧；在可达性上能达到精确指数，在逆向上与极限相符；数值实验显示在可达性与逆向区间内误差几乎完全贴合理论曲线。

**⚠️ 局限性**

局限性：需要假设假设类为紧凸且全支持；最不利对的存在仅在满足特定随机排序或指数族分离条件下保证；对可达性阈值的校正需数值优化，理论上尚无闭式解；逆向下界虽然与极限一致，但在中间区间可能仍存在松弛；多项式修正仅给出阶数与log‑n校正常数，缺乏精确常数的分析。

---

## 211. Sledgehammer or Scalpel? A Fine-grained Adaptive Framework for Implicit Hate Speech

**arXiv ID:** 2608.27462 | [PDF](https://arxiv.org/pdf/2608.27462v1)

**作者:** Han Wang `[一作]` (China University of Mining and Technology), Yi Zhu `[通讯]` (Yangzhou University)

**通讯引用:** 64034 | [OpenAlex ID](https://openalex.org/A5085159282)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 FAID 框架，通过先对隐式仇恨言论进行细粒度分类，再根据类别自适应路由到专门的推理模块，实现对不同难度样本的精准检测。

**💡 创新点**

创新点在于将隐式仇恨言论细分为浅层、目标化、上下文依赖三类，并为每类设计轻量级 Prompt‑Tuning、知识增强迭代模型和 ACE 多代理框架，从而实现资源匹配与计算效率的双重提升。

**🔧 技术方法**

技术包括 Prompt‑Tuning、知识增强与迭代训练、ACE（Generator‑Reflector‑Integrator‑Inspector）多代理推理、LLM（DeepSeek‑V3/GPT‑4）与 PLM（BERT）混合使用。

**📊 数据集**

实验使用四个公开基准数据集：SBIC、LHd、State_ToxiCN 与 ProsCons，涵盖英语与中文、不同隐式表达与上下文特征。

**📈 对比分析**

与 DNN、PLM、Prompt‑Tuning 及 LLM 等多种 SOTA 基线对比，FAID 在 Accuracy/F1 上均实现最高分，且推理时间显著低于纯 LLM 方法，达到最佳性能与效率平衡。

**⚠️ 局限性**

局限性包括对极端边缘样本仍易误判、LLM 安全对齐限制可能导致解释生成受阻，以及缺乏大规模细粒度标注数据支持进一步验证。

---

## 212. Learning to Allocate Incentives for Incentivized Advertising via Offline Model-Based Reinforcement Learning

**arXiv ID:** 2608.28065 | [PDF](https://arxiv.org/pdf/2608.28065v1)

**作者:** Zilin Zhao `[一作]` (Nanjing University), Yinsong Xue `[通讯]` (ByteDance)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对激励式广告中平台在未观测广告收益前先承诺用户激励的场景，提出一种基于离线模型的强化学习框架，自动学习并分配用户激励金额以最大化净收益。

**💡 创新点**

创新点包括：①将激励分配建模为成本敏感的短期序列决策问题；②引入结构化世界模型（World Model）捕捉用户接受、完成和RTB收益的概率分布，并在策略优化中加入保守（conservative）正则化；③设计独立的反事实评估器（Counterfactual Evaluation Scorer, CES）实现离线政策筛选，避免昂贵的在线实验。

**🔧 技术方法**

核心技术包括：离线模型基础的强化学习（Offline-MBRL）—基于MMoE的多任务世界模型；保守策略优化（CQL/TD3+BC/IQL等背骨加上WM的混合训练）；半生命期折扣和事件驱动的MDP建模；离线反事实评估器对请求级预测进行聚合。

**📊 数据集**

实验使用 ByteDance 真实日志数据，分别为：DY-1（约2亿条）、DY-2（约4亿条）和 DH（约5000万条）。

**📈 对比分析**

通过 CES 离线评估和在线 A/B 测试进行比较。离线 CES 结果显示 MB‑IQL 在每用户净收益上比对应的模型无关基线提升约 5–8%（绝对增幅 6–8 个百分点），而在线 A/B 实验验证了这一排序，MB‑IQL 在净收益上比 TD3+BC 或 IQL 提升 7–8%，并且在成本保持一致的前提下保持较低的激励支出。

**⚠️ 局限性**

局限性：①仅优化短期（窗口内）净收益，未显式建模长期用户留存或 LTV；②CES 为支持感知的离线评估器，无法提供完全无偏的长期反事实估计；③世界模型和 CES 共享输入/目标，可能导致错误相关性，影响评估可信度；④在高维离散动作空间下的模型误差可能放大，需要进一步稳健性研究。

---

## 213. Adaptation Fidelity of SPEC CPU2026

**arXiv ID:** 2608.27710 | [PDF](https://arxiv.org/pdf/2608.27710v1)

**作者:** Doa'a Al-Otoom `[一作]` (Ampere Computing), Mahesh Madhav `[通讯]` (Ampere Computing)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统量化评估了 SPEC CPU®2026 与其开源原版之间的逼真度差距

**💡 创新点**

首次将“逼真度缺口”转化为可测量指标，并以单拷贝与多拷贝两种负载情景验证其影响

**🔧 技术方法**

采用编译对比、官方验证脚本、S‑curve 性能曲线及系统时间剖析等技术

**📊 数据集**

使用 SPEC CPU®2026 26 个基准及其对应的 23 个可重新构建的开源源码

**📈 对比分析**

在单拷贝运行中大多数基准速度相近；在 192 份并行运行中，I/O 与非可移植特性的移除使 SPEC 版明显加速，单拷贝中部分基准因去除手工调优汇编或线程开销而略慢

**⚠️ 局限性**

实验受限于只能在 AmpereOne AArch64 上复现，且需手工解决 23 个基准的构建差异，无法覆盖所有 SPEC 版特性

---

## 214. An Empirical Evaluation of Cross-City POI Recommendation on a Large-Scale Benchmark

**arXiv ID:** 2608.27840 | [PDF](https://arxiv.org/pdf/2608.27840v1)

**作者:** Peibo Li `[一作]` (University of New South Wales), Flora D. Salim `[通讯]` (University of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Trip World这一大规模、全球覆盖、语义丰富的数据集上，对跨城市POI推荐算法进行系统评测，揭示其在用户偏好迁移、规模化训练以及语义利用方面的瓶颈。

**💡 创新点**

创新点在于：①首次在大规模跨城市数据上揭示了传统“家乡‑城市”模型对区域流行度的过度依赖；②发现模型的准确率与训练成本呈倒U型，即最简单模型往往效果最好；③评估了语义特征整合和LLM驱动的代理方法，表明目前技术难以有效利用语义信息。

**🔧 技术方法**

使用了传统的流行度、顺序编码和图神经网络模型（Popularity、MatTrip、GraphTrip、AR-Trip、KDDC、PPROC、SPOT-Trip）以及两种基于大语言模型的代理方法（LLMMove、AgentMove），并对它们在候选选择场景下的性能进行了对比。

**📊 数据集**

采用了Trip World数据集（约6.24M本地签到、576k跨城签到，336k POI、890城市），并对比了此前的Foursquare和Yelp小规模数据集。

**📈 对比分析**

评估指标为F1与Pairs-F1，结果显示在Trip World上最简单的Bi-LSTM模型MatTrip获得最高F1≈0.06；SPOT-Trip（最复杂的知识图谱+ODE模型）在准确率和训练耗时上均位列最低；LLM代理方法在候选选择实验中均落后于单纯的流行度基线，说明现有模型难以充分利用语义与迁移信息。

**⚠️ 局限性**

主要局限在于：①低家乡‑目的地区域重叠导致模型几乎不利用个性化偏好；②大规模训练时复杂模型反而效果下降；③语义特征整合机制无效；④LLM代理方法在开放词汇生成上受限，需要更高效的检索与归纳机制。

---

## 215. Compositional Failure in Audio-Visual LLMs: Late-Layer Prior Dominance Under Cross-modal Conflict

**arXiv ID:** 2608.27785 | [PDF](https://arxiv.org/pdf/2608.27785v1)

**作者:** Adarsh Sudheer `[一作]` (Independent Researcher), Vasu Sharma `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究音频-视觉大模型在音频与视频冲突场景下的组合推理能力，并通过机制分析揭示先验优势。

**💡 创新点**

首次将冲突视为组合学习任务，发现后期层级先验主导导致错误判断，并定位到约 25 层的先验形成点。

**🔧 技术方法**

使用 Logit‑Lens 机制分析、注意力头审计、LoRA 微调（ACTC、TATI、AMD）以及贪婪解码评估。

**📊 数据集**

评估数据集包括完整 AVHBench（≈6300 条）和 1281 条手工挑选的冲突子集，实验还使用 InternVideo2 基线。

**📈 对比分析**

通过比较基线与多种对齐配置的 Yes/No 准确率发现，InternVideo2 在冲突集上从 60.1% 跌至 27.8%，对齐方法仅改变答案偏向，准确率仍停留在 50% 左右。

**⚠️ 局限性**

局限性包括机制分析仅覆盖 VideoLLaMA 体系结构、TATI 的序列长度效应未分离、评估仅限精确匹配且对跨模型泛化不足。

---

## 216. Not All Explanations Are Sought: Information-Seeking Psychology for Human-Centered XAI

**arXiv ID:** 2608.27464 | [PDF](https://arxiv.org/pdf/2608.27464v1)

**作者:** Andrea Beretta `[一作]` (CNR-ISTI), Salvatore Rinzivillo `[通讯]` (CNR-ISTI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出将信息寻求心理学框架（Sharot & Sunstein的三类效用）引入人本可解释AI，解释用户何时、为何寻求或回避解释。

**💡 创新点**

强调解释并非天然有价值，而是依赖用户感知的工具性、情感性和认知性效用；将关注点从解释内容转向解释动机。

**🔧 技术方法**

采用文献综述与理论建模方法，引用认知偏差与信息寻求理论。

**📊 数据集**

未使用实验数据集，全部基于已有文献和案例。

**📈 对比分析**

未进行实验比较，本文以案例分析与理论讨论为主，未给出性能指标。

**⚠️ 局限性**

缺乏实证验证；理论框架需要在真实HCXAI系统中测试；对不同领域和个体差异的适用性尚未评估。

---

## 217. What Do Interaction Representations Actually Measure? Pre-Event Separability in Weakly-Supervised Violence Detection

**arXiv ID:** 2608.27879 | [PDF](https://arxiv.org/pdf/2608.27879v1)

**作者:** Parishruthi Ganesh `[一作]` `[通讯]` (Auburn University), Parishruthi Ganesh (Auburn University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在固定跟踪、姿态估计、检测头和训练协议的前提下，研究者对五种两人交互表征（粗几何、手工姿态、增强姿态、单独姿态编码、原始关节编码）在早期暴力检测中的判别力进行对比。

**💡 创新点**

创新点在于通过严格的容量匹配和视频级别的交叉验证+聚类自助法，独立评估表征信息；并提出预事件诊断，揭示大部分分离来自数据源而非事件本身。

**🔧 技术方法**

技术方法包括YOLOv8m+BoT‑SORT追踪、RTMPose‑m姿态估计、七/十一通道手工/增强表征、2层MLP学习表征、双线性卷积+GRU的因果MIL检测头、线性探针、聚类自助置信区间估计。

**📊 数据集**

使用的公开数据集为UCF‑Crime（15条异常视频，150条正常）和更大规模的XD‑Violence（137异常，299正常），并对两者均做预事件截断实验。

**📈 对比分析**

结果显示粗几何在两种评估下（probe 0.724、MIL 0.684）表现最佳，所有姿态表征均不超过几何；冻结的外观/上下文编码在两数据集上均显著优于几何，其中全帧上下文在XD‑Violence上超过人裁剪；整体AUC约为0.6–0.7。

**⚠️ 局限性**

局限性包括异常样本极少（仅15条），无法检验高容量骨架网络；预事件保留分离揭示基准存在源泄漏，导致AUC难以单独反映事件检测能力；结果依赖于手工定义的通道和固定的检测头。

---

## 218. AI Hardware Accelerators for Large Language Models: Architectures and the Memory Wall

**arXiv ID:** 2608.28048 | [PDF](https://arxiv.org/pdf/2608.28048v1)

**作者:** Siddharth Patel `[一作]` (Shiv Nadar Institution of Eminence), Rohit Singh `[通讯]` (Shiv Nadar Institution of Eminence)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了从GPU、TPU、Trainium、Groq、Cerebras等专用ASIC，到FPGA、PIM、近内存计算，再到神经形态和光子计算等多种加速器在大语言模型（LLM）训练与推理中的应用与挑战。

**💡 创新点**

提出了统一的架构-部署-优化目标分类法，利用Transformer的算术密度与屋脊分析展示LLM推理的主要瓶颈是内存带宽与KV缓存，强调内存体系结构是加速的关键，并通过跨平台比较论证了异构、内存为中心的工具包是未来发展方向。

**🔧 技术方法**

采用屋脊模型、算术强度与内存带宽计算、量化/稀疏技术分析、实验与工业案例对比，结合MLPerf等标准基准，讨论了低精度格式、压缩、分块KV缓存等软硬协同优化。

**📊 数据集**

本综述不基于单一数据集，而是综合了2018–2026年间公开的论文、技术白皮书、工业报告和MLPerf基准结果。

**📈 对比分析**

通过计算密度与带宽对比、tokens/s与tokens/Joule指标的统一评估，指出GPU在通用性与生态上占优，ASIC在高吞吐量与能效上占优，PIM与近内存计算在突破内存墙方面最具潜力，光子与神经形态技术仍处于实验阶段；整体表现显示内存带宽是决定吞吐与能效的主导因素。

**⚠️ 局限性**

局限性包括：缺乏统一的实测基准，软硬协同成熟度不均衡导致对新型加速器的性能估计不确定，现有研究多聚焦在推理的decode阶段，训练与混合专家模型的全链路评估不足，以及光子、神经形态等前沿技术尚未达到大规模LLM部署的水平。

---

## 219. DBRepro: Automated Database Synthesis via a Hybrid Constraint-Solving Approach for Reproducing Slow Queries

**arXiv ID:** 2608.27822 | [PDF](https://arxiv.org/pdf/2608.27822v1)

**作者:** Zhaoyang Zhang `[一作]` (Renmin University of China), Xiaoyong Du `[通讯]` (Renmin University of China)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自动化的数据库合成框架DBRepro，旨在通过非侵入性元数据合成代理数据库，以离线重现慢查询的执行性能。

**💡 创新点**

创新点在于将数据库生成视为一个受限分布合成问题，结合数据驱动和工作负载感知的方法，确保在满足局部基数约束的同时最大限度地保留全局统计分布。

**🔧 技术方法**

采用了一种混合约束求解方法，结合启发式概率推理和约束编程模型，特别是针对非主键-外键连接的缩放因子机制。

**📊 数据集**

使用了TPC-H和SSB基准测试以及来自KingbaseES的近1TB真实工业数据集。

**📈 对比分析**

与最先进的基线方法相比，DBRepro在结构一致性、基数误差和延迟比例误差等多个指标上表现优越，基数误差降低了20.3%，延迟比例误差降低了21.5%。

**⚠️ 局限性**

局限性在于在处理复杂的非主键-外键连接时，可能需要在生产系统中收集额外的键频率分布，增加了提取成本。

---

## 220. Graphon Design for Human-Machine Coordination under Bounded Rationality: Optimality of Stochastic Block Models

**arXiv ID:** 2608.27851 | [PDF](https://arxiv.org/pdf/2608.27851v1)

**作者:** Zhewei Wang `[一作]` (Florida State University), Marcos M. Vasconcelos `[通讯]` (Florida State University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过图森（graphon）方法，在混合人机群体中设计网络拓扑，最大化在斯塔猎游戏下的协作程度。

**💡 创新点**

创新点在于：① 将混合有界理性的人机系统映射到连续图森空间；② 推导出势函数与对偶方程的变分一阶最优条件；③ 证明对双峰理性分布最优图森为随机块模型，并给出水填充式贪婪分配算法。

**🔧 技术方法**

主要技术包括：图森理论、潜在函数（潜能）分析、对偶（adjoint）方程、变分法、收敛性与KKT条件分析以及贪婪水填充算法。

**📊 数据集**

实验使用基于模拟的随机块模型数据（β_H=2，β_M=3.99，θ=0.3，ℓ=0.5），无真实外部数据集。

**📈 对比分析**

与均匀分配和优先人机连接两种基线相比，所提出的优先机器连接策略在所有密度预算下均实现更高的聚合采纳率 J₁，最高提升约 28%（从 0.5 基线至 0.78）。

**⚠️ 局限性**

局限性包括：未证明在整个图森空间 𝒲_ρ 上的全局最优性；仅针对离散双峰理性配置，连续理性分布及多峰情形未覆盖；缺乏有限样本下的性能保证和分布式实现方法。

---

## 221. Online Differentially Private Consistent Clustering

**arXiv ID:** 2608.27896 | [PDF](https://arxiv.org/pdf/2608.27896v1)

**作者:** Edith Cohen `[一作]` (Google Research), Marika Swanberg `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种通用的差分隐私半核心集（semi‑coreset）构造方法，将敏感数据流转化为隐私半核心集，随后可以把任意非私有在线聚类算法作为后处理，得到在线 k‑means / k‑median 的 DP 版本，同时保持良好的一致性与近似质量。

**💡 创新点**

创新点包括：
- 泛化的 DP 半核心集框架，能把任何非私有在线聚类算法转化为 DP 版本；
- 通过私有化的 Meyerson sketch + DP 稠密球 + 连续计数，首次实现几乎最优一致性（k·nd）和几乎最优加性误差（√d/k·polylog(n)）；
- 在维度 d 下的加性误差几乎达到理论下界，且时间空间均接近线性。

**🔧 技术方法**

核心技术：
- 私有化的 Meyerson sketch（采样 + 直观的稠密球覆盖）；
- DP 稠密球算法（保证每个稠密球至少包含一个代表点）；
- 连续计数机制用于私有估计样本集大小；
- 半核心集与一致性分析，利用后处理不破坏 DP；
- 组合后处理将半核心集转换为完整的聚类输出。

**📊 数据集**

论文未给出具体实验数据集，主要以理论分析和对比证明为主。

**📈 对比分析**

与此前工作（如 [cit]）对比：
- 近似比保持 O(1)；
- 加性误差为 √d/k·polylog(n)，几乎最优；
- 一致性约束为 k·nd，接近理论下界；
- 时间复杂度 n^{1+o(1)}，空间 n^{o(1)}（k,d ≤ n^{o(1)}），比先前的多项式时间/空间更优；
- 通过后处理实现，兼容任意非私有算法。

**⚠️ 局限性**

限制与开放问题：
- DP 稠密球算法目前仅理论实现，缺乏实用实现；
- 加性误差对 k 的依赖仍可改进；
- 只生成半核心集，无法得到完全核心集或更细粒度的误差；
- 效用分析仅在无敌对流（oblivious）上成立；
- 对抗性流和更强的隐私模型下的实用性尚未证明。

---

## 222. ZipMVS: Multi-View Stereo with Compressed Cost Volumes

**arXiv ID:** 2608.28033 | [PDF](https://arxiv.org/pdf/2608.28033v1)

**作者:** Guanglin Jin `[一作]` (Hunan University), Zhaoxin Li `[通讯]` (Chinese Academy of Agricultural Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了 ZipMVS，一种针对资源受限场景的高效多视角立体重建框架，能够在不牺牲重建质量的前提下显著降低 GPU 内存占用。

**💡 创新点**

创新点在于：① 像素级可微分深度采样策略（Differentiable Depth Range, DDR）结合自适应范围网络 (ARNet) 与中央稠密采样 (Central Densely)；② 采用 GRU‑based Depth Speculator (GDS) 从邻域深度信息中推测更可靠的深度假设；③ 通过组合上述两种假设生成少量高质量深度样本，避免传统粗细级深度搜索的冗余迭代。

**🔧 技术方法**

使用的技术包括：多尺度特征提取的 Feature Pyramid Network；基于组间相关的 Group‑wise Cost Volume；3D‑UNet 正则化网络；残差深度细化模块；以及对深度假设进行自适应二次变换的可微分网络。

**📊 数据集**

实验数据集涵盖：DTU、Tanks and Temples（Intermediate 与 Advanced 子集）以及自制的低纹理太空 ISS 合成数据集，用以验证在工业与航空航天场景中的表现。

**📈 对比分析**

与 MVSNet、CasMVSNet、PatchmatchNet、IterMVS、GBi‑Net 等主流方法对比：在 DTU 上 Acc 为 0.369 mm、Comp 为 0.284 mm，整体误差 0.327 mm，排名第二；在 Tanks and Temples Intermediate 上 F‑score 54.46，Advanced 上 33.03；GPU 内存仅 1322 MB，运行时间 0.112 s/深度图，显示出比大多数非迭代方法更优的内存/速度平衡。

**⚠️ 局限性**

局限性：① 内存占用相较于极简迭代方法仍高约 49%；② 对视角数量敏感，最佳效果在 4‑5 张图像时达到饱和；③ 主要依赖已知摄像机位姿，难以直接应用于实时无人机或实时嵌入式系统；④ 在极低纹理或强光照条件下，精度仍略逊于某些高精度基线。

---

## 223. Locate Anything in Videos: Rethinking Efficient Generative Spatio-Temporal Video Grounding

**arXiv ID:** 2608.28192 | [PDF](https://arxiv.org/pdf/2608.28192v1)

**作者:** Hanoona Rasheed `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Salman Khan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并行管道解码（PTD），通过先定位时间区间再并行生成空间框，显著降低推理深度。

**💡 创新点**

创新点包括：1）PTD 通过两步推理消除空间预测的序列依赖；2）Decoupled Block Attention 使各空间块独立但共享多模态上下文；3）基于定位的策略优化提升时空边界和框几何。

**🔧 技术方法**

采用量化空间/时间 token、双格式（NTP/MTP）训练、Decoupled Block Attention、GRPO 策略优化，并以 Qwen3‑VL‑4B 语言模型为主干。

**📊 数据集**

使用 VidSTG 与 HC‑STVG (v1/v2) 进行 STVG 训练与评估；零样本迁移评估涉及 Charades‑STA、ActivityNet、ReXTime、Ref‑DAVIS、Ref‑YT‑VOS、ReasonVOS 等数据集。

**📈 对比分析**

与四种解码范式（无量化、量化、序列块、PTD）对比，PTD 在 Tube Completion Latency 下降 79 倍、Boxes Per Second 提升 92 倍，同时在 VidSTG 与 HC‑STVG 上取得最高 m_tIoU/m_vIoU；零样本迁移亦优于现有基准。

**⚠️ 局限性**

局限在于仅支持单段单实例时空管线，对多段、多实例及复杂遮挡/快速移动目标的定位仍易出错，且 STVG 数据集规模有限。

---

## 224. Nemotron 3.5 Content Safety Moderator: A Compact Multimodal, Multilingual, and Reasoning Enabled Content Safety Moderator

**arXiv ID:** 2608.27548 | [PDF](https://arxiv.org/pdf/2608.27548v1)

**作者:** Varun Singh `[一作]` (NVIDIA), Katherine Luna `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一款4B参数的多模态安全过滤器，可对提示、图片和回答进行联合评估，并支持12种语言与自定义政策。

**💡 创新点**

创新点在于将图像、文本、响应、语言多样性与自定义政策统一在一个紧凑模型中，并提供可选的推理链以审计。

**🔧 技术方法**

采用Gemma 3‑4B作为基座，通过监督微调（SFT）、自定义chat模板、链式思考标注和八阶段合成数据生成（SDG）实现多模态与政策适配。

**📊 数据集**

训练集结合人标注的真实图像、多语文本安全数据、Nemotron VLM/Guard、CantTalk‑About‑This Topic-following、以及自生成的极少见违规与绕过案例。

**📈 对比分析**

在VLGuard、MM‑SafetyBench、PolyGuard、DynaGuardrail等基准上，平均 unsafe F1 达 0.85‑0.86、错误率低于 0.03、并在图像‑文本输入下 TTFT/E2E 仅 60/76 ms，显著低于同类 Llama Guard 4。

**⚠️ 局限性**

局限性包括仅支持单张图像、对文档类输入误拦率高、缺乏公开图像‑响应安全基准，以及推理链模式导致延迟上升。

---

## 225. DeicticVLA: Unifying Instruction Modes Based on Language and Deictic Gestures in a Single VLA

**arXiv ID:** 2608.28108 | [PDF](https://arxiv.org/pdf/2608.28108v1)

**作者:** Kango Yanagida `[一作]` (University of Osaka), Takato Horii `[通讯]` (University of Osaka)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DeicticVLA，将语言指令（LI）、语言+指示手势（VLI）以及仅指示手势（VI）统一为一条文本提示和一组去向掩码，允许单一预训练的 Vision‑Language‑Action 模型同时支持三种指令模式；

**💡 创新点**

创新点在于：①设计指令规范化流程，使三种模式的输入转换为同一条文本和掩码表示；②引入两阶段训练（先用 LI 训练再混合 VLI/VI），显著提升对未见布局的 deictic 掩码利用；③系统性比较四种提示方法（RGB 视觉提示 VP‑BBox / VP‑Fade 与独立通道掩码提示 MP‑Early / MP‑Late），并展示 VLI/VI 在新指令、视觉扰动及新物体上的优势；

**🔧 技术方法**

技术实现包括：预训练 VLA（π₀），SAM‑2 分割与追踪生成去向掩码；RGB 视觉提示（绘制边框或淡化非掩码区域）与独立通道掩码提示（在视觉编码前后注入掩码嵌入）；两阶段微调；并使用 Transformer‑based VLA 结构；

**📊 数据集**

使用的数据集有：仿真端的 LIBERO‑Object、LIBERO‑Spatial、LIBERO‑Goal 及其零样本扩展；真实端收集 360 条演示轨迹，包含 PickBlock、PutBlock、OrganizeToy 三个任务；

**📈 对比分析**

通过在相同预训练模型、相同演示数量、相同总训练步数下对四种提示方法进行对照实验；在分布内平均成功率约 94–96%；在零样本评估中，VP‑BBox 与 MP‑Late 在 Object‑ZS 与 Spatial‑ZS 上表现最佳，展示了两阶段训练提升的效果；真实世界实验中，VLI/VI 在 TL2/TL3、VC‑Surface、NO‑Instance/NO‑Category 条件下的成功率均显著高于 LI，尤其在未知类别下实现 100% 成功；

**⚠️ 局限性**

局限性包括：联合训练后 LI 模式的性能下降，说明小规模数据下需更合适的样本比例；仅在真实机器人上评估了 MP‑Late，未对 RGB 视觉提示或其他掩码提示进行对比；未系统评估分割/跟踪误差对任务的影响；缺乏对用户工作负担、多轮交互等人机交互维度的验证。

---

## 226. PCFBench: A Diagnostic Benchmark for Product Carbon Footprint Estimation

**arXiv ID:** 2608.27716 | [PDF](https://arxiv.org/pdf/2608.27716v1)

**作者:** Krishna Rao `[一作]`, Travis M. Kwee `[通讯]` (Watershed Technology, Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建并评估了首个拆分式产品碳足迹（PCF）生成基准，拆解为六个可评估子任务。

**💡 创新点**

创新点在于将 PCF 估算拆分为可评估子任务、引入专家标注数据、提供细粒度错误诊断，并公开基准与评估工具。

**🔧 技术方法**

使用大语言模型（如 GPT‑5、Claude、Gemini 等）以及专门的映射工具 Parakeet 进行多步推理、检索与数值抽取。

**📊 数据集**

使用了 614 条专家标注条目，涵盖 BOM、筛选、映射、材料/能量抽取以及 175 条 EPD 的总计验证。

**📈 对比分析**

通过与单步直接预测和任务级别评分对比，发现前沿 LLM 在整体估算上达 60–77% 2× 内准确率，但在拆分流程中仅 37–58%，且 25–55% 的物料列表违反质量守恒。

**⚠️ 局限性**

局限包括数据量有限、仅覆盖 cradle‑to‑gate、未评估文档检索、无跨产品链标注、数据库特定、单语、未测量排放因子误差，以及可能的记忆污染。

---

## 227. eBPF-Based Cybersecurity Mechanisms: A Systematic Literature Review

**arXiv ID:** 2608.27511 | [PDF](https://arxiv.org/pdf/2608.27511v1)

**作者:** Stamatios Kostopoulos `[一作]` (Hellenic Mediterranean University), Evangelos K. Markakis `[通讯]` (Hellenic Mediterranean University)

**通讯引用:** 2807 | [OpenAlex ID](https://openalex.org/A5009055115)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过PRISMA方法对2018-2026年间关于eBPF安全机制的论文进行系统综述，筛选并分析了54篇高质量原始研究。

**💡 创新点**

首次构建了七大安全域的体系化分类，并系统评估了eBPF在低延迟、高检测率下的优势与局限，揭示了多租户隔离、ML鲁棒性和生产验证等研究空白。

**🔧 技术方法**

采用eBPF（包括XDP、eBPF映射、eBPF工具链）作为核心技术，结合机器学习、硬件加速与容器化、微服务等场景。

**📊 数据集**

数据来源为来自IEEE Xplore、ACM DL、Scopus等六大数据库的54篇论文，未使用公开数据集，而是对各研究所用的数据流、流量包、系统调用日志等进行汇总。

**📈 对比分析**

作者通过对不同研究的指标（CPU占用、检测准确率、攻击覆盖率）进行量化比较，指出eBPF实现的平均CPU开销为2.4%（1.1-8.6%），检测准确率为94-99%，并对比了传统用户空间方案的性能。

**⚠️ 局限性**

主要局限包括eBPF verifier的指令上限导致算法复杂度受限、85%研究需低层编程经验、内核版本碎片化影响可移植性、并且96%研究未考虑eBPF自身的安全漏洞。

---

## 228. Conditional Diffusion Models for Energy-Efficient Driving

**arXiv ID:** 2608.28142 | [PDF](https://arxiv.org/pdf/2608.28142v1)

**作者:** Hemanth Neelgund Ramesh `[一作]` (University of Washington), Shijing Sun `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并评估了一种条件扩散模型，用于在商用配送电动车的路线上下文（速度、温度）下生成电池电流时间序列，支持不确定性推理。

**💡 创新点**

通过引入隐式编码器将多模态条件映射为共享潜在表示，再在1D U-Net扩散网络中注入，显著提升生成质量并减少均方误差和Wasserstein距离。

**🔧 技术方法**

条件扩散概率模型、基于Transformer的隐式编码器、1D U-Net denoising backbone、余弦噪声调度、MAE、RMSE、Wasserstein距离等评估指标。

**📊 数据集**

开源商用EV遥测数据集（Rücker等），约12,000次行程、9辆车、1 Hz采样，固定长度512。

**📈 对比分析**

与直接注入条件、线性噪声调度、不同网络深度的变体以及传统LSTM进行对比；E4模型Wasserstein距离0.0029低于真实‑真实0.0085，MAE仅次于LSTM，直接注入模型误差增大九倍。

**⚠️ 局限性**

仅使用速度与温度两种条件，轨迹长度固定、缺乏可变长度支持，未在闭环调度系统中验证，缺少多变量电池信号与更丰富路况信息。

---

## 229. URIUM: A Programming Language for a Practical Open Course on Compiler Design

**arXiv ID:** 2608.28202 | [PDF](https://arxiv.org/pdf/2608.28202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 230. Picking Bins Empty: A Hierarchical Hybrid Approach with Online Self-Learning of Grasp Points for Reliable Industrial Bin-Picking

**arXiv ID:** 2608.28175 | [PDF](https://arxiv.org/pdf/2608.28175v1)

**作者:** Florian Töper `[一作]` (Mercedes-Benz AG), Peter Ohlhausen `[通讯]` (Fraunhofer Institute for Industrial Engineering)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种四层级混合抓取框架，结合模型驱动的精准姿态估计与模型自由的探索抓取，并加入在线自学习机制，实现工业级无人工干预的全箱清空。

**💡 创新点**

创新点在于：①将模型基与模型自由两种截然不同的抓取方式层级化融合；②使用抓手闭合反馈与 Wilson 置信区间对抓取候选进行实时排名与自学习；③在多层级中动态切换，解决感知失败与死锁问题；④通过抓取数据库在线扩充显著降低人工调参需求。

**🔧 技术方法**

采用 6D 姿态估计与基于 CAD 的预设抓取数据库、Contact‑GraspNet（CGN）模型进行模型自由抓取、基于点云的碰撞检查、Wilson 置信区间排名、与仿真/物理约束结合的重新评分。

**📊 数据集**

实验使用工业级三种汽车零件（A、B、C）共 150 件/箱，并基于公开的 GraspClutter6D 数据集训练 CGN；同时使用 50~150 件/箱的数据进行线上学习。

**📈 对比分析**

对比：模型基线 bin 清空率 50.9%，模型自由基线 100%；混合方法在所有 30 箱中保持 100% 清空，抓取成功率在手动初始化下平均 91.9%（A）、83.6%（B）、89.3%（C），显著高于模型自由 45.6% 但略低于模型基线 99.1%；仅在预初始化时性能更高。

**⚠️ 局限性**

局限包括：对部件 B 的抓取深度预测不足导致误抓率高；需要多次学习才能覆盖所有场景；周期时间因多层级生成和碰撞检测略增；未能自动检测多件纠缠；对不同抓手或多品种混合箱的适应性待验证。

---

## 231. Coordinated Motion Planning for Multi-Arm Systems via Iterative LQ Games

**arXiv ID:** 2608.27726 | [PDF](https://arxiv.org/pdf/2608.27726v1)

**作者:** Junyoung Kim `[一作]` (Purdue University), Ahmed H. Qureshi `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于迭代线性二次游戏的多臂协作运动规划框架ILQ-Arm。

**💡 创新点**

创新点在于将多机器人协作建模为差分游戏，结合可微自碰撞和互相碰撞惩罚，并通过二次成本在二阶优化中求解局部Nash策略。

**🔧 技术方法**

技术手段包括迭代线性二次规划（ILQ）、Ricatti递归求解Nash策略、可微碰撞距离函数、控制正则化与终点目标成本。

**📊 数据集**

实验使用约1500个仿真实例（2~4臂、含/不含障碍、NeHMO基准），并在两台UR5e机器人上进行真实验证。

**📈 对比分析**

与集中式CHOMP、递归最佳响应CHOMP、RRT*以及NeHMO对比，ILQ-Arm在成功率、规划时长与路径长度上均显著优于基线，尤其在多臂与障碍环境下保持最高成功率与最快规划速度。

**⚠️ 局限性**

局限性包括仅在静态障碍环境下验证，缺乏对动态障碍的适配，且只得到局部Nash解，可能受初始轨迹影响且可扩展性在极大规模系统上仍待验证。

---

## 232. Semantic Watermarking with Order-Robust Detection over Sub-sentence Units

**arXiv ID:** 2608.27666 | [PDF](https://arxiv.org/pdf/2608.27666v1)

**作者:** Abdulrahman Diaa `[一作]` (University of Waterloo), Florian Kerschbaum `[通讯]` (University of Waterloo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了基于句子编码的文本水印在全文本重写（改词、重排、重分句）下的鲁棒性，并提出了一种基于嵌入位移的自适应无盒攻击（SWORD）及对应的防御方案SwordStamp；

**💡 创新点**

创新点包括：①将词义保持改写、重排与重分句三种结构扰动统一归纳为嵌入位移并构建统一攻击；②在防御中引入Best‑of‑N选取、固定有效集以及确定性子句单元，以提升对结构扰动的抗性；③在无盒与检测器访问两种威胁模型下系统性评估；

**🔧 技术方法**

主要技术手段有：句子编码器（LSH、k-means聚类、在线多通道等）、余弦距离位移目标、公开重写器与 surrogate encoder、Best‑of‑N采样、固定有效集、子句分割器，以及ROUGE‑1与句子级逆序率等内容保留量化指标；

**📊 数据集**

实验使用C4语料库的约8000条文档进行训练和测试，生成12句、最多512词的标记文本，并对多种公开重写器（ParaNMT、T5‑para等）进行评测；

**📈 对比分析**

对比方法采用ASR@FPR、Fidelity与内容保留率三维指标；SWORD在5% FPR下对四种原始水印实现约90%的攻击成功率，内容保留约70%；SwordStamp在相同点将攻击成功率降至10–20%，仅产生5–10%的质量损失；

**⚠️ 局限性**

局限性在于仅评估英语新闻续写任务，未覆盖代码生成、问答、指令执行或多语言场景；使用单一重写器与 surrogate encoder，且未进行人工评测，可能忽略更强的攻击和质量判别器的影响。

---

## 233. Token-Budget Distillation: Transferring Full-Token Semantics to Compressed Video Vision-Language Models

**arXiv ID:** 2608.28138 | [PDF](https://arxiv.org/pdf/2608.28138v1)

**作者:** Xiaoyang Guo `[一作]` (Sun Yat-sen University), Wenhao Wang `[通讯]` (Vast Intelligence Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在严格的视觉token预算下，对视频视觉语言模型进行参数高效微调，并通过教师-学生双路结构实现全token语义的迁移。

**💡 创新点**

将全token教师与压缩学生结合的双分支蒸馏框架，并设计可靠的答题区KL蒸馏、GT锚定的边缘蒸馏和动态KD调度，有效抑制语义漂移，解决压缩引发的语义丢失。

**🔧 技术方法**

使用LoRA进行参数高效微调、FlashVID训练无关的视觉token压缩、答案区KL蒸馏、GT边缘蒸馏、可靠性加权的KD控制，以及自适应权重调度等技术。

**📊 数据集**

训练使用LLaVA-Video-178K数据集；评估在MVBench、VideoMME、EgoSchema和LongVideoBench四大视频理解基准上。

**📈 对比分析**

在保持token保留率R=20%和R=10%的压缩比例下，与原始未压缩模型以及FastV、VisionZip、FastVID、FlashVID等训练无关压缩基线进行对比。TBD在三种主流backbone（LLaVA-Video、LLaVA-OneVision、Qwen3-VL-8B-Instruct）上均显著优于压缩基线，平均得分接近或超过未压缩模型，并在10%保留率下保持97%–100%的相对准确率；Ablation实验进一步验证各组件贡献。

**⚠️ 局限性**

训练阶段仍需完整token的教师推理，导致显著的内存和计算开销；学生性能受限于不可逆的压缩信息损失；对压缩质量依赖较大，未能完全消除高压缩下的信息丢失。

---

## 234. Performance Evaluation of Fast Fourier Transforms on Emerging RISC-V Hardware with Vector Extension Support

**arXiv ID:** 2608.28076 | [PDF](https://arxiv.org/pdf/2608.28076v1)

**作者:** Daniel Seibel `[一作]` (Jülich Supercomputing Centre), Andreas Herten `[通讯]` (Jülich Supercomputing Centre)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文实现了一个支持RISC‑V Vector Extension（RVV 1.0）的快速傅里叶变换库juFFTe，并在三款RVV 1.0 CPU（SiFive X280、SpacemiT K3 X100、Sophon SG2044）上进行性能评估。

**💡 创新点**

创新点在于：① 用现代Fortran重写FFTE子例程，形成可维护的库；② 通过SPIRAL自动生成vector‑agnostic RVV 1.0内核，并手动使用RVV intrinsics；③ 在RVV平台上系统性评测FFT性能，展示RVV在高性能计算中的潜力。

**🔧 技术方法**

技术包括：现代Fortran、SPIRAL代码生成、RISC‑V Vector Intrinsics、OpenMP多线程、手写分块转置/打包内核、六步/四步/Stockham算法。

**📊 数据集**

数据集：合成1D复数FFT大小 N（从小到大，覆盖L1/L2/L3缓存阈值），多线程与单线程测试。

**📈 对比分析**

比较方法：在相同输入规模下测量单线程和多线程运行时间，计算 FLOPs/s，比较 juFFTe 与已针对RVV 1.0 优化的 FFTW3。结果显示：在单线程下，SiFive X280 约 1.53×、SpacemiT X100 0.90×、Sophon SG2044 0.73×；多线程下平均提升分别为 1.09×、1.91×、3.03×；在最优核心数下，Sophon SG2044 的单线程甚至可达 3×速度提升。

**⚠️ 局限性**

限制：① 目前RVV实现的编译器自动向量化不可靠，需手工intrinsics；② 各CPU在缓存与内存带宽上差异导致算法表现不同；③ 与x86 Zen5 的性能差距仍显著，说明RISC‑V生态仍需成熟；④ 只评估了单DWT，缺乏多维/实际应用场景。

---

## 235. ShiftSplit-AD: Separating Domain Shift from Defects in Foundation-Feature Visual Anomaly Detection

**arXiv ID:** 2608.27610 | [PDF](https://arxiv.org/pdf/2608.27610v1)

**作者:** Muhamathu Ameer Ali Aacaas Muhamath `[一作]` `[通讯]` (University of Moratuwa), Muhamathu Ameer Ali Aacaas Muhamath (University of Moratuwa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在冻结的DINOv2特征空间中，对每张测试图像的最近正常样本残差矩阵进行低秩与行稀疏分解，从而抑制域漂移产生的伪异常信号，同时保持真实缺陷信息。

**💡 创新点**

创新点在于提出一种“ShiftSplit”分解框架，利用低秩项吸收广泛的域漂移残差，行稀疏项保留局部缺陷残差，并在此基础上进行稀疏分量或低秩/稀疏融合的分数重加权，首次将结构化残差分解用于无监督异常检测。

**🔧 技术方法**

技术包括：冻结的DINOv2-small视觉特征提取、完整正常样本特征记忆、欧氏最近邻匹配得到残差矩阵、使用核范数和ℓ2,1行稀疏正则的近似低秩/行稀疏分解（交替软阈值），以及稀疏分量/融合分数的top‑5平均图像级评分。

**📊 数据集**

数据集包括：MVTec AD（瓶子、线缆、榛子、金属螺母、木材）作为开发与验证集，以及真实域漂移的AeBAD‑S（涵盖裂纹、断裂、凹槽、分解四类缺陷）用于外部测试。

**📈 对比分析**

与基准残差评分、PatchCore式DINO核心化等方法比较，稀疏分量仅评分在AeBAD‑S上显著提升（AUROC +0.0514，AUPRC +0.0412，Bootstrap 95%区间均为正），但在保留的MVTec类别上会显著降低性能（平均清洁AUROC下降0.0757，噪声AUROC下降0.0946）。融合评分可在一定程度上缓解性能损失，但整体仍不如原始残差评分。

**⚠️ 局限性**

局限性包括：仅使用DINOv2‑small冻结模型，分解参数固定且未自适应不同类别/域；低秩/稀疏分解未保证理论可恢复；对真实缺陷定位的改进有限（稀疏分量导致定位精度下降）；只在图像级别评估域漂移效果，未对定位级别进行完整实验；以及记忆检索耗时、缺少空间对应约束，可能影响异常的空间定位。

---

## 236. DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization

**arXiv ID:** 2608.27513 | [PDF](https://arxiv.org/pdf/2608.27513v1)

**作者:** Tao Zhang `[一作]` (South China University Of Technology), Ziqian Zeng `[通讯]` (South China University Of Technology)

**通讯引用:** 692 | [OpenAlex ID](https://openalex.org/A5001026113)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DAMP，一种针对 GDN 与 KDA 语言模型的后训练递归状态量化方法；

**💡 创新点**

创新点在于利用量化误差能量与学习到的衰减持久性联合评估关键通道的累计误差风险，按预算静态分配高精度与低精度混合布局；

**🔧 技术方法**

采用整数与浮点量化、Hadamard 范围均衡、离线校准、混合精度布局、融合递归更新核；

**📊 数据集**

使用 Qwen3.6‑35B 与 Kimi‑Linear‑48B 两大模型，并在 AIME 2026、HMMT、IMO‑AnswerBench、GPQA‑Diamond、MMLU‑Pro 与 LiveCodeBench‑v6 六大基准上评测；

**📈 对比分析**

相较于 FP32 基线和统一 INT8/HF 量化，DAMP 在 9.9 bits/值下平均保持 FP32 级别准确率，递归状态存储减 69.1%，更新核加速至 2.01×，整体解码 TPOT 降 10.9%；

**⚠️ 局限性**

局限在于需离线校准且布局固定，可能不适用于所有任务或更大模型；同时实现复杂度较高，且仅针对 GDN/KDA 结构。

---

## 237. Dual-Stream Semantic Guidance with Prototype Anchor Calibration for Source-Fully-Free Adaptation of Vision-Language Models

**arXiv ID:** 2608.28145 | [PDF](https://arxiv.org/pdf/2608.28145v1)

**作者:** Weiwei Xiang `[一作]` (Hunan University), Lei Yang `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对源全无（Source-Fully-Free）域自适应问题，提出 Dual-Stream Semantic Guidance (DSSG) 框架及其高效变体 DSSG-PAC，实现对 Vision‑Language 模型的无源数据无标签适应。

**💡 创新点**

创新点在于（1）双流语义引导：利用实例级标题和动态更新的类锚点并行对齐视觉与文本；（2）动态跨模态知识蒸馏（Dynamic CMKD），自适应教师-学生一致性；（3）原型锚定校准（PAC），周期性重采样类锚点，显著降低文本编码成本。

**🔧 技术方法**

使用技术包括 CLIP 预训练模型、BLIP‑3 生成标题、InfoNCE 对齐损失、置信阈值自监督、Gini 基准蒸馏、原型锚点周期性校准等。

**📊 数据集**

实验数据集包括 Office‑31、Office‑Home、Mini‑DomainNet 和 VisDA 四大标准跨域分类基准。

**📈 对比分析**

与 unimodal、U+M、M 等多种基线相比，DSSG 在严格 SFF‑DA 条件下均取得最优或接近最优的准确率，DSSG‑PAC 在保持 1% 以内精度的同时将训练时间降低约 18.9%。

**⚠️ 局限性**

主要局限是对生成标题质量高度依赖，标题噪声可能导致语义漂移；原型校准间隔需经验调参；以及在无标签设置下难以直接评估语义质量。

---

## 238. Can Tainted Pixels Expose Deepfake Videos?

**arXiv ID:** 2608.27492 | [PDF](https://arxiv.org/pdf/2608.27492v1)

**作者:** Juan Hu `[一作]` (National University of Singapore), Terence Sim `[通讯]` (National University of Singapore)

**通讯引用:** 7728 | [OpenAlex ID](https://openalex.org/A5065478753)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种主动视频保护方法，利用结构化蓝通道周期性扰动隐藏在源视频中，保护人脸视频不被公开的黑盒操纵工具生成逼真深伪；

**💡 创新点**

提出了异步可见性权衡框架，结合条纹可见度、色彩偏移和视频级LPIPS预算，采用面部感知分配、结构化蓝通道先验和运动自适应缩放，且不依赖对手工具的任何内部信息；

**🔧 技术方法**

采用结构化蓝通道周期性扰动先验、局部高频掩码、面部软掩码分配、可见度预算优化（包含扰动强度、结构一致性、可见度三项损失）、视频级LPIPS约束、运动自适应缩放以及Adam投影优化等技术；

**📊 数据集**

使用FaceForensics++和Celeb-DF中的真实视频作为源数据，利用Roop、FaceFusion、MuseTalk三款公开深伪工具进行操纵，采用NPR和Deep‑Fake‑Detector‑v2‑Model进行检测，并在400个多样化视频上开展人类感知实验；

**📈 对比分析**

与DFRAP、CMUA、FacePoison、FaceShield等主动防御方法对比，保护后伪造视频在两种检测器上的假率分别提升至84.32/90.32%和90.27/93.68%，在人类实验中的假率从56.71%提升至90.72%，视频LPIPS低至0.0042；在H.264压缩或重编码下仍保持较高的假率；

**⚠️ 局限性**

仅针对当前公开的黑盒工具，对未来闭源或自适应攻击缺乏鲁棒性；对手若了解扰动特征，可能通过定向去噪或强压缩降低效果；对视频质量的评估主要依赖LPIPS，其他细粒度质量指标尚未充分验证。

---

## 239. The Calls are Coming from Inside the Model: Investigating Probe-based Detection of Tool-Calling Errors in LLMs

**arXiv ID:** 2608.27750 | [PDF](https://arxiv.org/pdf/2608.27750v1)

**作者:** Eric Yeats `[一作]` (Pacific Northwest National Laboratory), Henry Kvinge `[通讯]` (Pacific Northwest National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究利用线性探针检测大型语言模型（LLM）的工具调用错误，探讨模型尺寸、探针层次和后训练方式对检测效果的影响，并验证探针在新错误类型上的泛化能力。

**💡 创新点**

首次系统性评估18种工具调用LLM的探针效果，并发现中间层探针在大模型中具有更高的错误检测性能和更好的跨错误类型泛化。

**🔧 技术方法**

使用线性逻辑回归探针，对隐藏层末端token的隐藏状态进行二分类预测；通过标准化、正则化和交叉熵训练实现。

**📊 数据集**

采用Berkeley Function Calling Leaderboard（BFCL）评估数据集，包含七类错误标签（语法、类型、虚假工具/参数、无效调用、缺失参数、错误值、正确）共750条/模型。

**📈 对比分析**

与模型工具调用准确率和传统一致性基线比较，探针在中间层平均AUROC>0.80、AUPR>3.5；大型模型探针性能提升约0.06 AUROC；跨错误类型迁移实验AUROC>0.75，表明泛化良好。

**⚠️ 局限性**

仅针对工具调用准确率≥80%的模型；探针仅基于隐藏状态，无法捕获所有细粒度错误；实验缺乏对不同探针架构（如小型MLP）的比较。

---

## 240. Real-time SQL Plan Management in Oracle

**arXiv ID:** 2608.27758 | [PDF](https://arxiv.org/pdf/2608.27758v1)

**作者:** Sunil Chakkappen `[一作]` (Oracle America Inc.), Nigel Bayliss `[通讯]` (Oracle Global Services Ltd.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Oracle 26ai 中实现了一种实时 SQL 计划管理（Real‑Time SPM）机制，能够在查询执行期间前台验证新生成的执行计划，快速检测并纠正性能回退，保证计划的稳定性与演进。

**💡 创新点**

核心创新点包括：① 利用位向量快速判断是否需要前台验证；② 通过历史计划与成本一致性检查挑选参考计划；③ 采用测试计划与参考计划的实时性能比较，并在必要时执行反向验证；④ 将回退检测与计划演进从后台迁移到前台，显著缩短回退响应时间。

**🔧 技术方法**

技术实现包括：SQL 计划基线存储与演进、Auto STS 自动查询集、成本一致性检查、反向验证算法、位向量（多计划位与 PHV 位）、参考计划成本计算、执行计划缓存失效与重新编译、前台性能计数器收集。

**📊 数据集**

实验数据来自：① 20,015 个 Oracle Autonomous Database 实例的 463,395 条独立 SQL（约 960,000 次验证事件）；② 一个包含 2,117 条语句的客户数据仓库工作负载；③ 约 39,000 条查询的 OLTP 工作负载。

**📈 对比分析**

比较方法：对比 Auto SPM（后台验证）与 Real‑Time SPM 的计划基线接受数量、任务持续时间、CPU 时间与 Buffer Gets 的归一化性能。结果显示 Real‑Time SPM 在 4 小时 20 分钟内接受 217 条基线（对比 Auto SPM 8 小时接受 63 条），CPU 时间降低至 443 分钟（比 Auto SPM 455 分钟更佳），Buffer Gets 亦得到进一步下降；在回退实验中，Real‑Time SPM 通过 189,208 条 SQL 成功避免了回退，回退因子分布峰值为 3，最大回退因子高达 6.7 万亿。编译开销仅提升约 2 KB 的 Buffer Gets 与 0.04 分钟的 CPU 时间。

**⚠️ 局限性**

限制与挑战：① 仍依赖历史统计与成本一致性阈值，若数据分布大幅变动可能导致参考计划失效；② 对极长执行时间的查询验证仍受限于单次前台执行的时间；③ 需要保持 Auto STS 与位向量同步，维护成本增加；④ 在极低资源环境下，前台验证开销可能对整体吞吐量产生影响。

---

## 241. Exact Risk Ratios for Weighted Data Selection in Linear Regression

**arXiv ID:** 2608.28007 | [PDF](https://arxiv.org/pdf/2608.28007v1)

**作者:** Guangjian Zhang `[一作]` `[通讯]` (University of New South Wales), Guangjian Zhang (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在仅使用有限样本的加权线性回归中，最小范数经验风险最小化器（min‑norm ERM）在全数据上的最坏情况风险比率(d,n)的具体值。

**💡 创新点**

在开放区间d<n<2d中给出精确值：(d,2d-1)=1+1/d；(3,4)=5/3；(4,5)=2；并提出通用公式(d,d+k)=1+Γ_{d,k}（基于和谐数分拆）作为完整区间的猜想。

**🔧 技术方法**

采用几何刚性定理、正向张成集与正基分类、Steinitz定理与体积采样、极值基原理以及正环块结构的代数分析，构造严格凸的加权选择并推导风险界限。

**📊 数据集**

使用构造的合成数据集（如正交梯度块、正交三角块、矩形配置等）来验证上界和下界，并通过符号锥几何给出具体加权选择方案。

**📈 对比分析**

通过最坏情况风险比率进行比较，证明在已分析的所有实例中，给出的上界与下界一致，数值上与已知极限（如d+1、1、1+1/d）匹配，展示了方法的有效性。

**⚠️ 局限性**

局限性在于尚未证明猜想在所有d和k值下成立，部分路经在特殊配置中失效；目前的选择算法是构造性的且不保证最优，且对更高维度的通用上界仍有挑战。

---

## 242. CHISEL-ing Back Source Code with AI-enabled Iterative Recovery

**arXiv ID:** 2608.27981 | [PDF](https://arxiv.org/pdf/2608.27981v1)

**作者:** Varun Kohli `[一作]` (A*STAR Institute of Advanced Intelligence and Computing), Dinil Mon Divakaran `[通讯]` (A*STAR Institute of Advanced Intelligence and Computing)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个无测试用例的迭代式反汇编框架，通过编译器和覆盖导向的差分模糊器的反馈，逐步提升伪 C 代码的可编译性和语义正确性。

**💡 创新点**

创新点在于完全去除对原始测试套件的依赖，利用丰富的可观察信号、跨迭代差分记忆和最佳候选保留等机制，形成了一个自适应的反馈驱动循环。

**🔧 技术方法**

采用 Gemma4:31b 大模型生成候选代码，配合 GCC 编译器做静态分析，libFuzzer 做差分模糊测试，加入类型感知的观察与记忆模块。

**📊 数据集**

使用 ExeBench 数据集中的 120 个 x86‑64 函数，涵盖 O0–O3 四个优化等级，包含剥离与非剥离两种二进制版本。

**📈 对比分析**

与 Ghidra 伪 C、LLM4Decompile‑9B‑v2、Agent4Decompile 等基线比较，取得 96.1% 的可编译率、79.8% 的可执行率，首次生成失败的 26.2% 被修复，误接受率仅 9.4%。

**⚠️ 局限性**

主要局限包括在剥离二进制中签名恢复不佳，LLM 采样温度受限，I/O 采样仅近似功能相似，且实验基于 Ghidra 伪 C 作为真值，可能与模型训练数据重叠。

---

## 243. CAVE-NAV: VLM-Based Autonomous 3D Navigation in Underwater Cave Environments

**arXiv ID:** 2608.27793 | [PDF](https://arxiv.org/pdf/2608.27793v1)

**作者:** Zhenqi Wu `[一作]` (University of South Florida), Xiaomin Lin `[通讯]` (University of South Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `51c0528b-f690-4182-ae60-bb5f046c276c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了基于视觉语言模型（VLM）与链式推理（CoT）的水下洞穴自主导航框架，利用RGB、深度和声纳垂直清晰度传感器进行感知，并直接生成安全的三维运动指令；

**💡 创新点**

创新点在于将大规模视觉语言模型与零样本链式推理相结合，完成对低纹理、光照恶化环境下的环境语义理解和路径决策，避免了传统SLAM对稠密特征的依赖；

**🔧 技术方法**

技术包括多模态感知处理、VLM（如GPT‑4V）与CoT提示、基于语义约束的动作生成、姿态更新以及高保真Blender仿真；

**📊 数据集**

使用了五种在Blender中生成的仿真洞穴场景（S1–S5）以及Blue Grotto现场拍摄的BlueROV2 RGB图像做测试；

**📈 对比分析**

与传统基于特征的SLAM或语义引导方法对比（实验未给出明确基线），在所有五个仿真场景中实现了100%任务完成率且无碰撞，证明了该方法在受限光照、稀疏纹理环境下的可靠性；

**⚠️ 局限性**

局限性包括：尚未在真实湍流、散射和光照不均等更恶劣环境中验证；缺乏持续地图构建和能量管理；VLM模型体积大，难以在嵌入式平台上实时部署；

---

## 244. More Data Cannot Break a Symmetry: Identifiability by Design

**arXiv ID:** 2608.27651 | [PDF](https://arxiv.org/pdf/2608.27651v1)

**作者:** Jing Xu `[一作]` (University of Rochester), Christopher Kanan `[通讯]` (University of Rochester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了在无标签情形下通过几何对齐恢复刺激对应关系的可识别性问题，并提出在实验设计阶段对刺激集合进行对称性打破的诊断与干预方法。

**💡 创新点**

创新点在于：① 将刺激几何的自同构群视为实验设计变量，首次揭示其对对齐可识别性的根本限制；② 提出“灾难性成本”诊断指标，用于评估设计对对齐失败的敏感性；③ 通过改变色彩刺激的饱和度与亮度而非简单均匀分布，显著降低灾难性对齐失败率，从75%降至2%。

**🔧 技术方法**

技术方法包括：Gromov–Wasserstein无监督对齐（最小化离散度矩阵误差），自动化自同构群分析，最小非平凡重标记成本计算，基于彩色空间（CIELAB、HSV圆柱）与真实观察者数据的实验验证。

**📊 数据集**

数据集：公开的9色与93色颜色集合（基于CIELAB、HSV空间），以及93个已训练的模型表示（保留用以评估对齐效果）。

**📈 对比分析**

比较方法：用灾难性成本、精确匹配率和位移距离评估对齐效果；在不同重启预算下比较对齐成功率。结果显示，在采用对称性破坏的9色设计中，灾难性失败率从75%大幅下降至2%，并且在所有93个模型表示上均保持稳定。精确匹配率虽受限，但位移距离显著改善。

**⚠️ 局限性**

局限性：仍需先验知道合适的候选几何空间；对称性破坏可能影响模型区分能力（虽然实验未发现显著相关性，但在不同任务中仍可能存在权衡）；所用方法对大规模刺激集合的计算复杂度较高。

---

## 245. 3D-USE: From Image-Level to Scene-Level Underwater Enhancement

**arXiv ID:** 2608.28020 | [PDF](https://arxiv.org/pdf/2608.28020v1)

**作者:** Jieyu Yuan `[一作]` (Nankai University), Chongyi Li `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出3D-USE框架，在多视角水下图像中学习持久的可视化增强3D场景表示。

**💡 创新点**

创新点在于用MediumRBF构建水体自适应Gaussian场景，利用Appearance Transition Consensus (ATC)将2D UIE知识转换为场景级一致目标，再通过Underwater Bilateral Appearance Field (U-BAF)实现无须2D增强模型的持久渲染。

**🔧 技术方法**

主要技术包括基于MediumRBF的水体建模、ATC的全局与局部增强一致化、U-BAF的低秩4D变换，以及3D Gaussian Splatting与深度先验融合。

**📊 数据集**

使用SeaThru-NeRF、DRUVA、D3/D5等真实水下场景数据集，并利用UIEB的paired UIE数据训练转换先验。

**📈 对比分析**

与WaterSplatting、SeaSplat、MarineSTD-GS、Plenodium、3D-UIR等方法对比，3D-USE在PSNR/SSIM上领跑，UCIQE/URanker/MUSIQ及跨视角一致性wLPIPS也表现最佳，且渲染速度最高。

**⚠️ 局限性**

局限在于仍需大量多视角训练数据，且对极端光照或动态水体效果的泛化能力有限，缺乏真实增强3D基准。

---

## 246. Measuring the Installed Base: Nordic Health Dataset Catalogues Against HealthDCAT-AP Release 7

**arXiv ID:** 2608.27720 | [PDF](https://arxiv.org/pdf/2608.27720v1)

**作者:** Fabio Rovai `[一作]` `[通讯]` (Kampakis and Co Ltd trading as Tesseract Academy), Fabio Rovai (Kampakis and Co Ltd trading as Tesseract Academy)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对北欧五国在欧洲数据门户中采集的健康数据集描述进行HealthDCAT-AP Release 7的合规性测评，发现所有健康主题描述都未满足八项必需属性，其中三项在五国均缺失；

**💡 创新点**

首次公开完整的实时测量结果及对应的 SHACL 形状与 OWL 词汇表，并将合规性结果作为带时间戳的观察记录公开，揭示欧盟健康主题词汇及授权服务的缺陷；

**🔧 技术方法**

使用 SPARQL 端点查询、SHACL 验证（分层处理）、OWL 2 词汇表、Python/pyshacl 计算，结合数据管道自动化脚本；

**📊 数据集**

主要数据集为欧洲数据门户公开的北欧国家（瑞典、丹麦、挪威、芬兰、冰岛）采集的健康主题数据集描述，以及芬兰 Findata 次级使用目录；

**📈 对比分析**

通过与欧洲门户自身的 Metadata Quality Assessment（MQA）进行对比评估，发现即便 DCAT-AP 层已部分合规，HealthDCAT-AP 层仍完全缺失，说明两层间评估独立性；性能表现未给出具体指标，但测量完成仅需约 20 分钟，可重复执行；

**⚠️ 局限性**

测量仅覆盖门户已采集的 DCAT 数据，未覆盖未发布 DCAT 的本国目录；健康主题筛选以欧盟主题词为准，导致可能低估实际健康数据集数量；Findata 采集受限于 5,000 条记录的上限；

---

## 247. REINS: Refusal-Enhanced Inhibitory Steering with Sparse Autoencoder Features

**arXiv ID:** 2608.28233 | [PDF](https://arxiv.org/pdf/2608.28233v1)

**作者:** Kai-Xuan Ding `[一作]` (University of Science and Technology of China), Zhen-Hua Ling `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于稀疏自动编码器（SAE）的推理时安全引导方法REINS，用于在大语言模型中同时抑制有害续写并增强安全拒绝；

**💡 创新点**

创新点在于将有害续写抑制与拒绝增强两个控制器在同一SAE特征空间中协同工作，并引入REINS‑Gate在高风险提示下才激活干预；

**🔧 技术方法**

采用的技术包括残差流稀疏自动编码器、梯度归因特征筛选、拒绝特征校准以及基于相似度的门控机制；

**📊 数据集**

使用的数据集包括新构建的GUISE（包含900个带复杂包装的有害提示及其安全对应）、HarmBench、JailbreakBench、AdvBench以及通用评估集MMLU‑Pro和GPQA；

**📈 对比分析**

与传统SAE驱动的拒绝放大、单向稀疏特征调优和相关性驱动的引导方法相比，REINS在GUISE上将有害响应率从约90%降至26.3%，安全拒绝率从3.7%升至63.7%，并在外部基准上保持低崩溃率，同时REINS‑Gate进一步恢复了通用能力；

**⚠️ 局限性**

局限性包括仅在Qwen3.5-4B/2B模型上验证，SAE配置受限，未在更大或指令微调模型上测试，且对特征组合与电路层面交互的解释仍不完整。

---

## 248. Performative Privacy: When Differential Privacy Maximizes Utility

**arXiv ID:** 2608.28198 | [PDF](https://arxiv.org/pdf/2608.28198v1)

**作者:** Uddalak Mukherjee `[一作]` (Dauphine Psl), Yann Chevaleyre `[通讯]` (Dauphine Psl)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本工作提出了“performative privacy”概念，即隐私泄露会导致用户流失，从而形成长期的反馈循环，并在反复的均值估计任务中研究隐私预算对长期效用的影响；

**💡 创新点**

创新点在于：①将差分隐私与表演性学习结合，正式化隐私泄露与用户参与之间的动力学；②给出二项式随机回应与高维高斯机制下的群体演化闭式解与临界噪声阈值；③证明存在有限隐私预算能够在长期内优于无隐私估计；④提供精确的成员推断泄露概率与Rényi界限。

**🔧 技术方法**

主要技术包括差分隐私（局部DP、零浓度DP）、随机回应、Gaussian机制、成员推断攻击、Rényi散度、定理推导与数值仿真。

**📊 数据集**

研究使用合成数据：二项式分布与多维高斯分布，用于验证理论推导和仿真。

**📈 对比分析**

与无隐私（ε→∞）均值估计比较；通过仿真热图、理论热图与最优噪声尺度对比，显示在存在足够强反馈（流失率大于招聘率）时，有限隐私预算的误差曲线低于无隐私估计，且理论与实验得到的最优噪声尺度高度吻合。

**⚠️ 局限性**

局限性包括：仅考虑均值估计任务，忽略更复杂模型；用户行为简化为固定离开概率与招聘概率；隐私预算为恒定，未考虑随时间动态调整；阈值τ假设为常数；未考虑多轮隐私预算累计与更精细的成员推断策略。

---

## 249. Unsupervised Continual Learning with Growing Self-Organizing Maps and Synthetic Replay

**arXiv ID:** 2608.27662 | [PDF](https://arxiv.org/pdf/2608.27662v1)

**作者:** Pujan Thapa `[一作]` (Rochester Institute of Technology), Travis Desell `[通讯]` (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于增长自组织映射（GSOM）的无监督、无任务边界、无样本记忆的连续学习框架，利用GSOM单元的均值、方差和协方差统计进行生成性重放。

**💡 创新点**

创新点在于：① 用惊奇（surprisal）驱动的局部拓扑增长机制，使模型在不知任务边界的情况下自动扩展；② 通过每个GSOM单元的分布统计生成合成样本，实现无示例回放；③ 结合Encoder–Decoder（VAE、ResNet‑18、CLIP）在低维潜在空间生成样本，兼顾低维与高维数据。

**🔧 技术方法**

主要技术包括：增长自组织映射（GSOM）+ 分布式统计（均值、方差、协方差）+ 惊奇阈值驱动的拓扑增长 + 生成式重放（潜在空间采样 + 逆编码） + 任务无关的无监督学习流程。

**📊 数据集**

实验数据集涵盖MNIST、CIFAR‑10、CIFAR‑100、TinyImageNet、MiniImageNet，测试了分割（split）和单类连续（SCI）两种增量协议。

**📈 对比分析**

与多类基准（EWC、SI、LwF、GEM、iCaRL、CoPE、Dynamic OCM、ER等）比较，GSOM 在无监督、无任务边界条件下取得与或超过有监督、任务知情方法的性能，尤其在使用预训练 ResNet‑18 或 CLIP 表征时，在 Split‑CIFAR‑100、MiniImageNet 上表现突出。

**⚠️ 局限性**

局限性包括：单元级 Encoder–Decoder 训练数据不足导致生成质量不稳；Gaussian 假设可能无法捕捉多模态分布，导致合成样本与真实样本分布重叠；扩展性受限，未对大规模在线更新做充分并行化与参数高效化。

---

## 250. Retrieving Relations, Detecting Fallacies: A RAG Approach to Political Debate Analysis

**arXiv ID:** 2608.27471 | [PDF](https://arxiv.org/pdf/2608.27471v1)

**作者:** Deborah Dore `[一作]` (Université Côte d'Azur), Serena Villata `[通讯]` (Université Côte d'Azur)

**通讯引用:** 3092 | [OpenAlex ID](https://openalex.org/A5016281730)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种基于论证关系引导检索的外部知识增强的谬误检测与分类方法。

**💡 创新点**

通过将论证支持/攻击关系动态用于检索查询，替代传统静态结构特征；创建15GB时间约束的政治知识库；在多配置下系统性评估检索提升效果。

**🔧 技术方法**

使用Retrieval‑Augmented Generation（RAG）框架，密集检索（Sentence‑RooseBERT、SBERT、BGE）、BM25混合与交叉编码重排序；Encoder‑only模型（RoBERTa、DeBERTa、LegalBERT、RooseBERT等）与Decoder‑only LLM（GPT‑4、Claude、Llama 3、Gemini）进行微调和提示。

**📊 数据集**

ElecDeb60to20（1960‑2020年美国总统辩论的论证组件、关系与六类谬误标注）以及15 GB定制知识库（法案、地理、人物、历史、定义、示例）。

**📈 对比分析**

采用macro‑F1、精确率、召回率等指标；在42检索配置和14模型中与无检索基线对比；谬误检测macro‑F1从0.772提升到0.864（+0.092，显著），分类macro‑F1从0.653提升到0.725（+0.072，显著且稳定）。

**⚠️ 局限性**

仅在单一英语美国总统辩论语料验证，知识库与领域高度匹配；检索配置通过三阶段搜索确定，可能未找到最优组合；检索对Decoder‑only模型不利，效果依赖检索器与查询配对；计算资源有限。

---

## 251. RECAST: Recent & Context-Aware Sampling for Test-Time Adaptation in Streaming Biosignals

**arXiv ID:** 2608.28271 | [PDF](https://arxiv.org/pdf/2608.28271v1)

**作者:** Yong-Yeon Jo `[一作]` (Medical AI Co., Ltd.), Joon-myoung Kwon `[通讯]` (Medical AI Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种轻量级采样模块RECAST，结合时间新鲜度、上下文相似度和预测可靠性，在在线自适应（Test-Time Adaptation）框架中挑选最有用的缓冲样本，提升生理信号（如血压）预测精度。

**💡 创新点**

创新点在于把样本选择显式化为三项互补指标（recency、similarity、reliability）的综合评分，而非仅靠最新或全部样本；该方法可直接插入现有TTA框架（如TTC），无需改动模型结构或训练目标。

**🔧 技术方法**

使用一维CNN+Transformer编码器，基于自监督掩码重建任务进行预训练；在TTA阶段结合多任务（重建+预测）损失，并通过蒙特卡罗Dropout估计不确定性；样本评分采用余弦相似度与指数衰减、年龄惩罚不确定性。

**📊 数据集**

在两个公开临床数据集上评估：PulseDB（10s PPG/ECG+BP标签）和MC‑MED（60s多模态信号+间歇BP标签）。

**📈 对比分析**

与无自适应、无监督TMA、标准TTC等基线对比，RECAST在两数据集均获得最低MAE/RMSE、最高相关系数；特别是在PulseDB上提升约5.4% MAE，MC‑MED上提升约3.0%；在患者级别的分布分析中，表现最显著于血压波动较大的病例。

**⚠️ 局限性**

局限性包括：依赖缓冲插入顺序衡量新鲜度，可能与真实时间不符；对在病程内变化很小的患者效果有限；仅在TTC框架内验证，需进一步验证可迁移性。

---

## 252. Selective Interference Suppression of Siamese-Net in Heterogeneous Interference Channels

**arXiv ID:** 2608.27635 | [PDF](https://arxiv.org/pdf/2608.27635v1)

**作者:** Arkadeep Sinha `[一作]` (Indian Institute of Technology Madras), R. David Koilpillai `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在异构干扰下，使用SiameseNet端到端学习短块编码，得到针对主干扰对实现近正交、对弱干扰对保持协同的编码方案；

**💡 创新点**

提出通过梯度耦合实现加权帧势（WFP）最小化，使编码器根据干扰强度自适应选择正交与协同，突破了传统全正交或无差异化设计的限制；

**🔧 技术方法**

深度学习（Siamese autoencoders）、梯度耦合分析、加权帧势优化、帧理论与Grassmannian设计；

**📊 数据集**

合成的4用户实数高斯干扰信道数据（k=4, n=8，干扰系数Λ=1, λ=0.1等四种拓扑）；

**📈 对比分析**

与纯正交TDMA及均匀干扰基准对比，BLER曲线显示主干扰用户相较TDMA低0.6 dB，弱干扰用户低2.8 dB，且在不同拓扑下实现了选择性正交化；

**⚠️ 局限性**

仅在小规模4用户仿真验证，未验证大规模或真实信道；训练复杂度高，需进一步研究可扩展性与泛化能力。

---

## 253. uScenes: A Multimodal RGB and 3D Sonar Dataset for Underwater Robot Perception

**arXiv ID:** 2608.27795 | [PDF](https://arxiv.org/pdf/2608.27795v1)

**作者:** Trung Tien Dong `[一作]`, Xiaomin Lin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了uScenes多模态数据集，包含同步的RGB图像和三维多波束声纳点云；

**💡 创新点**

创新在于首次提供光学图像与直接测得的三维声纳点云的同步记录，弥补了仅有二维声纳或仅有光学数据的缺口，为水下跨模态感知与融合研究提供了新基础；

**🔧 技术方法**

采用BlueROV2平台搭载DWE RGB相机和Water Linked Sonar 3D-15，实时生成三维点云；通过时间同步、点云构建、EPnP等技术实现多模态对齐与外参估计；

**📊 数据集**

使用了本文自己创建的uScenes数据集，包含110场景、95,834帧同步RGB与3D声纳观测，总时长277.6分钟；

**📈 对比分析**

文中未给出具体算法比较或性能指标，仅指出数据可用于未来光学与声纳联合检测、分割等基准的建立；

**⚠️ 局限性**

局限性包括仅在一台机器人和单一淡水环境收集；缺乏多传感器、多平台、盐水环境的多样性；未提供详细标注；声纳信号强度仅相对归一化，且扫描时存在运动造成的几何失真。

---

## 254. Generalized Gibbs Ensemble Weighting for Forecast Combination

**arXiv ID:** 2608.28116 | [PDF](https://arxiv.org/pdf/2608.28116v1)

**作者:** Prasen R. Nuthanakaluva `[一作]`, Nava K. Gaddam `[通讯]` (Utrecht University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了通用的 Gibbs 风格集合权重框架（GGEW），通过对历史预测误差的指数加权实现在线自适应预测组合；

**💡 创新点**

创新点包括：①将 Gibbs 指数分配与专家权重更新结合，形成统一的权重计算规则；②引入稳定的指数梯度更新，避免投影带来的数值不稳定；③设计了多种多样性修正（无修正、方向性NCL、对称性NCL），并在此基础上构建了可在线调参的 Local-UCB 机制；

**🔧 技术方法**

使用了指数加权权重、归一化预测误差、稳定化参数、梯度裁剪、内部迭代的指数梯度更新、基于 UCB 的在线超参数搜索；

**📊 数据集**

实验数据涵盖：M4 公开提交系统的多频率时间序列（约 41 系统）、Monash Traffic Hourly、Electricity Hourly、Solar Weekly 等滚动原点仿真数据；

**📈 对比分析**

与传统平均、截断平均、中位数、逆损失权重、指数权重、历史最佳模型以及 top-3 平均等基线进行对比。GGEW 在多种数据集与不同行为域（高/中/低分歧）中经常进入前 3 名，甚至在某些实验（Traffic Hourly、Electricity Hourly、Solar Weekly）中取得最低总平方误差；

**⚠️ 局限性**

局限包括：①依赖于多样且质量足够的基准预测池；②分歧度量仅使用相对方差，未探索其他可能更细粒度的分歧指标；③超参数搜索限定在离散网格，未实现连续在线优化；④实验多基于点预测和平方损失，未扩展到概率或分位数预测；

---

## 255. Information-Guided Selective Modality-Interest Alignment for Multimodal Recommendation

**arXiv ID:** 2608.27950 | [PDF](https://arxiv.org/pdf/2608.27950v1)

**作者:** Wenze Ma `[一作]` (Shanghai Jiao Tong University), Xuhao Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 AMUR 框架，通过信息引导的选择性模态-兴趣对齐，提升多模态推荐性能。

**💡 创新点**

创新点在于：①使用行为校准的图结构细化来抑制与用户兴趣不匹配的模态信号；②兴趣感知对比学习提升模态表示与用户兴趣的相关性；③在全局模态对齐前先学习共享兴趣子空间，仅对该子空间执行对齐，既强化共享语义又保留模态特有信息。

**🔧 技术方法**

技术手段包括：图神经网络对用户-物品交互图和模态图进行信息传播；KL 散度正则化与兴趣对比损失实现行为校准和兴趣对齐；信息理论视角下的互信息与条件熵估计；soft selection gate 进行子空间选择；多任务联合优化 BPR、对齐损失与正则化。

**📊 数据集**

实验使用三大多模态推荐基准数据集：Baby、Sports、Clothing，均包含图片、文本和交互记录。

**📈 对比分析**

与多类基线（MF、Invariant/causal、Graph、SSL/对齐、Diffusion、Transformer/MLLM）比较，AMUR 在 Recall@10/20、NDCG@10/20 上均位居榜首，提升幅度约 3-8% 以上，且对模态噪声更具鲁棒性。

**⚠️ 局限性**

局限性包括：对超参数 α、β 的敏感性需要手工调优；框架主要关注视觉/文本两模态，扩展到更多模态仍待验证；对极大规模数据的可扩展性与训练效率虽优于部分基线，但仍比纯协同过滤方法慢。

---

## 256. PersonaEdit: Representative Sample Selection for Personalized Model Editing

**arXiv ID:** 2608.27816 | [PDF](https://arxiv.org/pdf/2608.27816v1)

**作者:** You-Mei Huang `[一作]` (National Yang Ming Chiao Tung University), An-Zi Yen `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了利用模型编辑实现大语言模型（LLM）个性化，并提出基于隐藏表示聚类与比例分层抽样的样本选择策略。

**💡 创新点**

创新点在于：①将隐藏表示聚类与比例分层抽样结合，用少量代表性样本完成高效编辑；②证明模型编辑与检索增强提示可互补，进一步提升个性化效果。

**🔧 技术方法**

采用的技术包括：Locate‑then‑Edit框架（ROME、MEMIT、AlphaEdit）、K‑means隐藏表示聚类、比例分层抽样、检索增强生成（RAG）、软提示等。

**📊 数据集**

使用的数据集为OpinionQA，包含15轮交互、30名用户、约31,000条问答，划分为训练集（约9,000条）和测试集（22,017条）进行评估。

**📈 对比分析**

通过与Vanilla、Few‑shot、Persona、All‑info、FERMI等基线进行对比，PersonaEdit+All‑info在LLaMA 3.1 8B上取得47.00%准确率，比基线提升13.65%；在MAE、CEM等指标上亦表现最佳。

**⚠️ 局限性**

局限性包括：仅在OpinionQA单一任务评估；未覆盖开放式对话、长期交互或动态偏好变化；样本选择未与更多策略全面对比；模型编辑在不同架构下的泛化及部署、隐私与安全风险未充分探讨。

---

## 257. Biologically Inspired Mechanisms for Facilitating Grokking in Multilayer Perceptrons

**arXiv ID:** 2608.28184 | [PDF](https://arxiv.org/pdf/2608.28184v1)

**作者:** Florin Leon `[一作]` `[通讯]` (Gheorghe Asachi Technical University of Iași), Florin Leon (Gheorghe Asachi Technical University of Iași)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多层感知机中加入多种基于生物学启发的机制，探究它们对延迟泛化（grokking）现象的影响。

**💡 创新点**

首次将多种神经元活动调节与结构塑性机制系统性地应用于人工神经网络，并通过消融实验揭示其对grokking的不同贡献。

**🔧 技术方法**

引入输入门控、结构稀疏化、增益调制、阈值调制、稳态调节、侧向抑制和激活去相关等机制，并结合SGD/Adam等优化器训练MLP。

**📊 数据集**

使用稀疏奇偶校验任务（sparse parity）和带噪XOR聚类分类任务（noisy XOR）这两大经典grokking基准。

**📈 对比分析**

通过与全部机制开启/关闭以及单机制、双机制组合的消融比较，量化训练/测试准确率、学习曲线、结构与表征指标，结果表明稳态调节和结构稀疏化能显著缩短grokking延迟并提升泛化性能。

**⚠️ 局限性**

实验仅限于小规模MLP和两种人工基准，缺乏对更大网络（如LLM）或不同任务的验证，且部分机制在本实验中效果有限，需进一步研究其可扩展性和理论机制。

---

## 258. Nested Byte-Level Vocabularies Are Cheap to Deploy and Expensive to Share: A Pre-Registered Negative Result

**arXiv ID:** 2608.28151 | [PDF](https://arxiv.org/pdf/2608.28151v1)

**作者:** Christos Koutsiaris `[一作]` `[通讯]` (SAP P&E), Christos Koutsiaris (SAP P&E)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出使用字节级BPE的前缀嵌套词表，训练单一模型可切换到多种词表大小并可通过切片实现部署

**💡 创新点**

验证嵌套词表在模型切片上实现零误差且可节省66%权重，但共享模型在词表大小变化时性能下降，且输出限制对模型有负面影响；但多粒度训练能提升对字母错误的鲁棒性

**🔧 技术方法**

字节级BPE、前缀嵌套、控制token、输出词表限制、2×2因子析因、预注册实验、负结果报告

**📊 数据集**

FineWeb‑Edu文本数据集（去重后约200M token）

**📈 对比分析**

用比特/字节(BPB)作为主要评测指标，发现共享模型相较于专用词表在32k词表下比BPB高3.64%，在8k词表下高2.96%；切片部署保持输出一致，权重减少66%；但多粒度训练模型在字符错误测试中表现优于各专用模型

**⚠️ 局限性**

实验规模有限（3.1M/10.6M参数），仅单一tokenizer和数据来源，未收敛；数据主要为英文，非英语表现未知；未考虑embedding与输出head绑定；模型仅在单个tokenizer上评估，缺乏tokenizer方差评估

---

## 259. QUORUM: QUality-Optimized Routing Using Multiple annotators

**arXiv ID:** 2608.27974 | [PDF](https://arxiv.org/pdf/2608.27974v1)

**作者:** Antonio Purificato `[一作]` (Amazon), Fabrizio Silvestri `[通讯]` (Sapienza University of Rome)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个基于预算的动态注释路由框架，能够在有限的人类和LLM资源下，智能分配每个实例的注释工作。

**💡 创新点**

创新点在于放弃传统的不确定性估计，改用语言学与语义特征驱动的实例难度评估，并支持多注释聚合与预算感知的自适应决策。

**🔧 技术方法**

使用贝叶斯线性回归估计注释器质量、ε‑greedy探索策略、特征组合（词长、TF‑IDF、句子长度、Flesch可读性、词性比例、语义距离）以及加权投票/语义相似度聚合。

**📊 数据集**

在多种分类（AG’s News、SST‑2、IMDB、PubMed、MMLU‑Redux、CNN）和摘要（CNN/DailyMail、XLSum）任务，以及西班牙语和日语的多语言评测上进行实验。

**📈 对比分析**

与基于不确定性（CDI、CoAnnotating、HyPAC、ARAIDA、SANT）及随机路由基线对比，取得了最高的质量–成本 Pareto 前沿，在多任务上质量提升高达34.4%，成本下降8.8%，并在低预算场景表现尤为突出。

**⚠️ 局限性**

局限性主要包括实验集中于英语及少数语言，缺乏对更专业领域的验证，未来需要在更广泛的语言和领域场景下进一步评估和优化。

---

## 260. Credo: Reusable Declarative Primitives for Agentic Workflows

**arXiv ID:** 2608.27790 | [PDF](https://arxiv.org/pdf/2608.27790v1)

**作者:** Duo Lu `[一作]` (Brown University), Uğur Çetintemel `[通讯]` (Brown University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了如何将经过搜索得到的LLM harness程序恢复为声明式描述，并将其存入共享目录，从而实现跨任务的编译与重用。

**💡 创新点**

提出了四类可转移的声明式原语（骨架、信念、策略、提示模板）及其转移标签，并展示了从搜索到编译的端到端复现与跨域迁移的可能性。

**🔧 技术方法**

结合Meta-Harness搜索代理、代码提取与恢复、层级标签化、声明式目录存储、基于原语的编译器，并使用vLLM与OpenAI API进行评估。

**📊 数据集**

在财务检索、法律检索、数学推理和跨域推理四个任务域，使用DocFinQA、FinQA、LegalBench-RAG、OlympiadBench、Omni-MATH等公开数据集。

**📈 对比分析**

通过回路编译对比原始搜索 harness 的准确率与成本，误差均≤±0.011/±0.013¢/q；在跨域重用时，编译后的 harness 在成本上降低约10‑70倍，准确率与搜索相当或更优。

**⚠️ 局限性**

当前编译器仅基于规则匹配，缺乏成本基优化、目录维护、执行上下文恢复以及多模型/后端的支持；原语间交互与漂移问题仍待研究。

---

## 261. Graphionale: How Graph Visualizations of LLM Rationales Affect Human Decision Making

**arXiv ID:** 2608.27932 | [PDF](https://arxiv.org/pdf/2608.27932v1)

**作者:** Xinru Wang `[一作]` (Singapore-MIT Alliance for Research and Technology), Thomas W Malone `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了图形化 LLM 推理理由对人类决策的影响，并通过 204 名参与者的实验比较了图形化与文本化理由在不同任务模态下的效果。

**💡 创新点**

提出了基于 argument-map 的图形化理由生成原型，并揭示图形化理由在口头推理中能提升信任校准，而在视觉推理中会降低校准，强调任务模态与理由格式匹配的重要性。

**🔧 技术方法**

使用 GPT‑5.4 生成 argument‑map 结构，配合 React Flow 等技术实现交互式图形化界面。

**📊 数据集**

使用 BIG‑Bench Hard（口头推理）和 I‑RAVEN（视觉推理）数据集中的易难两类任务。

**📈 对比分析**

通过 2×2 组实验比较，发现图形化理由在口头推理难题中显著降低过度信任并提升错误纠正率，而在视觉推理中则相反，整体表现表明格式与任务模态匹配是关键。

**⚠️ 局限性**

局限在于仅研究结构化问答任务、仅采用一种图形化理由实现、未考虑开放式协作情境及可编辑图形化理由等。

---

## 262. VeriTS: Verifiable Model-Enhanced Time-Series Queries on Blockchain Systems

**arXiv ID:** 2608.28318 | [PDF](https://arxiv.org/pdf/2608.28318v1)

**作者:** Zhongming Yao `[一作]` (Zhejiang University), Shiliang Zhang `[通讯]` (University of Oslo)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出VeriTS框架，实现区块链上时间序列查询的可验证性，支持时间范围检索与窗口聚合；

**💡 创新点**

创新点在于：1）使用单一认证聚合区间树既做索引又做认证；2）引入最小覆盖集与子树聚合折叠实现完整性与正确性验证；3）支持可验证的近似查询路径，基于已认证的模型段并定义误差界定；

**🔧 技术方法**

采用加密哈希、Merkle树、认证聚合区间树（AIT）、段区间树（SIT）、最小覆盖集（MCS）等技术；

**📊 数据集**

实验使用从30,000条Ethereum区块与200,000条TRON区块提取的五条时间序列：ETH-Gas、ETH-Transfer、TRON-Transfer、TRON-Balance、TRON-Dust；

**📈 对比分析**

与全扫描方案、基于Merkle路径的完整性验证以及基于叶节点聚合的基线对比；VeriTS在窗口聚合查询上提升验证效率超过两位数，VO大小比基线小14.5倍；在滑动窗口查询中使用增量VO减少约1.9倍；

**⚠️ 局限性**

局限性包括：1）对查询窗口与数据结构对齐的依赖，导致增量VO效益受限；2）近似查询仅适用于可容忍误差的场景，误差控制需先验预算；3）模型学习不进入可信路径，无法防止误差分配不合理导致结果宽度过大；4）实验环境为单机原型，未在大规模真实链上验证性能。

---

## 263. HyQuant: Hybrid-Precision Quantization for LLM Attention

**arXiv ID:** 2608.27875 | [PDF](https://arxiv.org/pdf/2608.27875v1)

**作者:** Jiatong Ding `[一作]` (Shanghai Jiao Tong University), Yiming Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了HyQuant混合精度量化框架，在LLM长上下文推理中保留垂直线位置和局部窗口的全精度，剩余部分低精度量化，从而提升推理速度与保持准确性。

**💡 创新点**

通过垂直线感知识别关键位置，混合精度保留少量高重要性token，实现高效低位量化，并在Prefill和Decode阶段采用融合算子与在线反量化，兼顾计算与内存两大瓶颈。

**🔧 技术方法**

使用低位量化（INT4/FP4）、全精度局部窗口保留、垂直线评分累计、融合FlashAttention型在线Softmax、KV缓存低精度压缩与即时反量化、Triton实现的自定义核等技术。

**📊 数据集**

在Qwen3-8B/32B、Llama-3.1-8B-Instruct、GLM-4-9B-0414等模型上，评测LongBench v1、GSM8K、MATH500等基准数据集。

**📈 对比分析**

与FA2、KIVI、KVTuner、SageAttention等基线对比，Prefill MSE显著降低，Decode kernel加速1.32×至3.58×，整体解码速度1.04×至1.17×；在大批量场景下HyQuant唯一可跑32batch且吞吐最高。

**⚠️ 局限性**

对短文本任务提升有限；在高端GPU短上下文时加速不明显；未评估更大模型与代理/编码场景，内存占用略增。

---

## 264. Beyond Pairwise Graphs in Science: Hypergraph Adaptive Wavelet Operators for Parametric PDEs

**arXiv ID:** 2608.27883 | [PDF](https://arxiv.org/pdf/2608.27883v1)

**作者:** Rajat Sarkar `[一作]` (Indian Institute of Technology Delhi), Souvik Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 Hypergraph Adaptive Wavelet Operator（HALO），一种利用超图谱和 Chebyshev 波浪变换实现的高效、可适应多尺度的神经算子，用于学习 PDE 解决算子，兼容结构化与非结构化网格，并支持多步自回归预测。

**💡 创新点**

创新点包括：① 将网格/点云映射为超图以直接捕获多节点耦合；② 采用 Chebyshev 多项式近似的超图波浪变换，避免显式特征分解；③ 引入可训练的多尺度波浪尺度并用紧帧正则化保证频谱覆盖；④ 共享 Chebyshev 基底显著降低运算成本；⑤ 通过基于节点坐标的 k‑NN / k‑ring 超边构造实现网格无关性和可扩展性。

**🔧 技术方法**

核心技术包括：超图谱理论、Chebyshev 多项式逼近、超图波浪变换、紧帧正则化、层归一化、GELU 激活、点云/网格的 k‑NN / k‑ring 超边构造、以及自回归滚动推理机制。

**📊 数据集**

使用的基准数据集涵盖 8 个 PDE 任务：Darcy、Navier–Stokes、Allen–Cahn、Airfoil、Cylinder、Supernova、ShapeNet‑Car 和 Aircraft，涉及二维/三维、结构化/非结构化、工业级大几何等多种场景。

**📈 对比分析**

与频谱算子（FNO、Geo‑FNO、WNO）、Transformer（Transolver、GNOT）、DeepONet、状态空间模型 LaMO 以及图算子 GNO 等方法在所有 8 个基准上进行公平比较；HALO 在所有任务中排名前两名，尤其在非结构化网格和大规模几何上显著优于现有最强基线，并在零样本分辨率迁移和长期滚动预测上保持低误差。

**⚠️ 局限性**

局限性包括：在推理超出训练滚动窗口的长期预测中误差会累积；尚未在百万点级工业网格或多物理耦合系统上验证其性能；对极端大尺度或复杂耦合问题的适应性仍待进一步研究。

---

## 265. Dandelion: A Spherical Flower for Neural Simulation of Planetary Dynamics

**arXiv ID:** 2608.27521 | [PDF](https://arxiv.org/pdf/2608.27521v1)

**作者:** Till Muser `[一作]` (University of Basel), Ivan Dokmanić `[通讯]` (University of Basel)

**通讯引用:** 2671 | [OpenAlex ID](https://openalex.org/A5002015062)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于变形的球面PDE求解器，能在球面上进行流体动力学仿真。

**💡 创新点**

创新点是将Flower架构改造成球面版本，利用切平面位移沿大圆传输特征，并通过球谐系域进行层级池化，无需卷积。

**🔧 技术方法**

使用技术包括球面切平面warp、沿大圆特征传输、球谐系谱压缩（层级池化）以及线性成本的坐标变换。

**📊 数据集**

使用了七个新颖的球面PDE数据集（改进的Galewsky jet、异常链式湍流、Cahn–Hilliard分离、球面Riemann冲击、Held–Suarez干燥大气输运、全球海洋动力学）以及The Well基准。

**📈 对比分析**

与SFNO、FNO、局部S^2/R^2 Transformer和原始2D Flower对比，模型在所有数据集上取得第一或第二名，且随分辨率提升与非warp基线的差距显著扩大。

**⚠️ 局限性**

局限性包括尚未在大规模实际观测数据（如ERA5）上验证，且在极高分辨率或更复杂三维流体场时的计算开销与泛化能力仍待进一步研究。

---

## 266. Regime-Aware Portfolio Management via Retrieval-Augmented LLM-Guided Expert Switching

**arXiv ID:** 2608.28252 | [PDF](https://arxiv.org/pdf/2608.28252v1)

**作者:** Ahmad Asadi `[一作]` (Amirkabur University of Technology), Reza Safabakhsh `[通讯]` (Amirkabur University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于检索增强的大型语言模型引导专家切换的动态投资组合管理框架，在不同市场条件下根据历史相似情况动态选择最优的专家策略。

**💡 创新点**

创新点在于将检索机制与专家性能记录相结合，利用历史情景的“绩效声明”做事实依据；使用LLM进行语义推理而非直接生成交易信号，并在理论上证明加入局部优专家不会降低整体收益（单调性）。

**🔧 技术方法**

核心技术包括双流变分自编码器（Dual‑Stream VAE）提取资产级与市场级特征；近邻检索（K‑NN）在嵌入空间寻找相似历史情景；LLM（如Qwen3.6‑27B‑it）进行检索结果的语义评估；以及基于绩效估计的专家切换策略。

**📊 数据集**

使用包含 30 只资产的加密货币、股票和外汇日交易 OHLCV 数据集，时间范围约 2017‑2026，训练窗口长度 22 天，持仓期 5 天。

**📈 对比分析**

通过与四种基准（最佳固定专家、近期性能门控、AlphaMixRL、TAC、LLM‑MAS 等）比较，实验显示在三类市场中检索增强切换实现了最高累计收益和夏普比率，且在加密货币、股票和外汇均表现出显著优于对手的风险调整收益。

**⚠️ 局限性**

局限在于：评估仅在单次实验运行下完成，未充分验证跨随机种子或不同时间段的稳健性；LLM 推理成本高且对模型选择敏感；检索库的构建依赖于手工设计的绩效声明；实验中未单独量化不确定性决策的贡献。

---

## 267. StreamEMS: Streaming Video Understanding with Self-Evolving Memory Scheme for Vision-Language Models

**arXiv ID:** 2608.27881 | [PDF](https://arxiv.org/pdf/2608.27881v1)

**作者:** Yuxin Liu `[一作]` (Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences), Yali Wang `[通讯]` (Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 StreamEMS，自演化内存机制提升流式视频理解；

**💡 创新点**

通过语义演化模块（SEM）和先验信息演化模块（PEM）对内存进行自我重构，提升信息密度与鲁棒性；

**🔧 技术方法**

使用外部可扩展内存、语义阈值层次图谱、指数移动平均（EMA）与跨注意力交互；

**📊 数据集**

在 OVO‑Bench 与 StreamingBench 两大流式视频评测集上验证，亦在 MVBench、Perception Test、TempCompass、LVBench、VideoMME 等离线数据集测试；

**📈 对比分析**

与 TimeChat‑Online、StreamAgent 等基线相比，StreamEMS 在 OVO‑Bench 取得约3.8% 的准确率提升，在 StreamingBench 多子任务均超过对手，表现更稳健，尤其在高 token‑drop 率下仍保持优势；

**⚠️ 局限性**

主要局限：依赖预训练的多模 LLM；对长视频时序建模的进一步提升空间；需在更大多模场景验证。

---

## 268. Expert Knowledge & Machine Understanding: Bridging Reactome's Ontology with LLM Semantic Embeddings

**arXiv ID:** 2608.28178 | [PDF](https://arxiv.org/pdf/2608.28178v1)

**作者:** Susanna Bravi `[一作]` (Italian National Research Council), Mario Santoro `[通讯]` (Italian National Research Council)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提取Reactome人类分子通路层级结构，利用通路与反应的标题与描述生成语义嵌入，随后通过改进的多叉层次聚类和图重构算法构建仅基于文本的语义层级，并与原始手工校正的层级进行比较。

**💡 创新点**

首次证明仅凭人工撰写的元数据（标题、描述）即可重建完整的生物通路层级结构；提出多叉聚类与递归图重构的组合方法，并用谱距离评估其整体拓扑保真度。

**🔧 技术方法**

SPECTER2句子嵌入、改进的AGNES多叉层次聚类、基于深度优先搜索的图重构、Frobenius距离与归一化拉普拉斯谱距离（LSD）对比。

**📊 数据集**

Reactome Graph Database V95（人类）共18,732节点（2,848条通路、15,884条反应）。

**📈 对比分析**

对比方法：边层级的Frobenius距离（216.88，约1.16%差异）和全局拓扑的拉普拉斯谱距离（0.085，远低于随机重连的0.224），通过500次保持度分布的随机重连进行显著性检验，证明重构层级与原始层级高度相似。

**⚠️ 局限性**

仅针对人类数据；评估指标仅限计算与拓扑；缺乏生物学专家验证；对其他物种或数据库的可推广性未知。

---

## 269. Accelerating Data Preprocessing for Efficient Vision Model Inference on Jetson Edge Device

**arXiv ID:** 2608.27655 | [PDF](https://arxiv.org/pdf/2608.27655v1)

**作者:** Tian Chen `[一作]` (Ohio State University), Panda `[通讯]`

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 Jetson 边缘设备上通过使用 NVJPEG 单元与多实例并行化技术，加速 JPEG 图像解码并优化预处理与推理流水线，实现更高吞吐量。

**💡 创新点**

首次在 Jetson 平台将 NVJPEG 解码与 CPU、GPU、DLA 并行协同，提供细粒度流水线设计，并通过多实例方式最大化硬件利用率。

**🔧 技术方法**

使用 NVIDIA TensorRT、NVJPEG、CUDA 多实例、DLA、CPU 并行解码、GPU 推理等技术。

**📊 数据集**

未具体说明数据集，实验基于 ResNet18/50/152 对大尺寸 JPEG 图像进行推理（可能使用 ImageNet）。

**📈 对比分析**

对比不同流水线设计，评估批量大小与图像尺寸对推理时间的影响，结果显示加入 NVJPEG 后在大图像上最高可提升 30.02 倍吞吐率。

**⚠️ 局限性**

仅适用于 JPEG 格式图像，依赖 Jetson 特有 NVJPEG 单元，未考虑其他压缩格式；多实例调优复杂，且对不同模型和批量大小的适配仍有限。

---

## 270. DisCTI: Who Needs to Know Timely? Automated Sector-Aware Cyber Threat Intelligence Dissemination

**arXiv ID:** 2608.27967 | [PDF](https://arxiv.org/pdf/2608.27967v1)

**作者:** Fajar Wijitrisnanto `[一作]` (National Cyber and Crypto Agency), Nan Wu `[通讯]` (Csiro)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了 DisCTI 框架，通过多标签分类自动为 CTI 事件分配行业标签，实现 sector‑aware 的 CTI 分发。

**💡 创新点**

结合专业领域知识构建首个 sector‑labeled CTI 数据集，并将 Transformer 迁移学习应用于多标签 CTI 标签化，显著提升自动化与实时性。

**🔧 技术方法**

采用 STIX/TAXII 标准数据格式、BERT 预训练模型（以及基线的 Parallel/Sequential 二元分类器）、TF‑IDF + 随机森林/高斯朴素贝叶斯等技术。

**📊 数据集**

从 MISPPriv TIP 导出 872 条 sector‑labeled CTI 事件构成的自建数据集，涵盖 11 个关键基础设施行业。

**📈 对比分析**

通过宏平均 F1、Hamming Loss 等多标签指标对比三种模型；BERT 在宏 F1 0.89、Hamming Loss 0.055 上优于二元分类器，表现最佳。

**⚠️ 局限性**

数据量有限、媒体类样本稀缺，模型推理延迟高，缺乏解释性，且在实时生产环境中的可部署性尚未验证。

---

## 271. Physics-Guided Flow Matching for CT Image Reconstruction

**arXiv ID:** 2608.28256 | [PDF](https://arxiv.org/pdf/2608.28256v1)

**作者:** Davide Evangelista `[一作]` `[通讯]` (University of Bologna), Davide Evangelista (University of Bologna)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在高分辨率肺部 CT 重建中，提出并训练了一个 Rectified Flow Matching 模型，并基于该先验设计了多种 Flow Matching 复原方法（PnP‑Flow、FlowDPS、FLOWER、ICTM），与扩散模型（DPS、DDRM）进行对比实验。

**💡 创新点**

创新点包括：①采用两阶段训练策略（强数据增强后细化）显著提升生成的解剖学一致性；②首次将 Flow Matching 作为高分辨率医学图像的先验，并展示其在稀视角 CT 中优于扩散模型的效果；③在数值上引入梯度归一化和 CGLS 校正以克服 CT 逆变换不精确导致的数值不稳定。

**🔧 技术方法**

使用技术：Rectified Flow Matching（ODE 采样），自回归 UNet 结构、时间嵌入、组归一化；扩散模型基准（DDPM、DDRM、DPS）；数值积分器（显式 Euler，RK4 讨论）；自适应梯度归一化、梯度裁剪和 CGLS 校正。

**📊 数据集**

数据集：Mayo Clinic 低剂量 CT 数据集，256×256 像素的胸部切片，共 3,306 张来自 10 名患者的扫描。

**📈 对比分析**

比较方法：对比 4 种 Flow Matching 重建方法与 2 种扩散基准，实验在 60、90、120 个投影角下进行。FlowDPS 在 PSNR、SSIM、LPIPS 上均优于扩散模型；平均 PSNR 超过 36 dB，SSIM 0.926，LPIPS 0.158；Flow Matching 在低采样步数下保持较低的 KID，证明采样效率高。

**⚠️ 局限性**

局限性：①需要针对 CT 采样特性进行多项数值校正，实验设置较为繁琐；②仅在 2D 切片上验证，未扩展至 3D 或动态 CT；③对训练数据量有限，若解剖变异性更大仍需更强泛化；④某些 Flow Matching 方法对 ν_t 参数敏感，调参成本较高。

---

## 272. User Preferences for UI Anchoring in MR: Effects of Task Mobility and Interface Properties

**arXiv ID:** 2608.28064 | [PDF](https://arxiv.org/pdf/2608.28064v1)

**作者:** João Belo `[一作]` (Saarland University), Anna Maria Feit `[通讯]` (Saarland University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文通过用户实验探究了混合现实中不同移动状态和交互类型下的 UI 锚定偏好，并实现了可实时切换和自定义锚定的交互控制。

**💡 创新点**

创新点在于：①提出并实现了基于用户操作的多种锚定模式切换交互；②系统性研究了移动性、任务属性与用户个人偏好如何共同决定最佳锚定；③揭示没有统一的“最佳”锚定，强调可定制化与适配。

**🔧 技术方法**

使用技术包括 Unity 6.0、Meta Quest 3、Meta Core 与 Meta Interaction SDK、Movement SDK、手部跟踪、头部姿态追踪以及实验中收集的位姿与交互事件。

**📊 数据集**

数据集为 19 名参与者（23–60 岁，平均 36.5 岁），每人完成 3 种移动情境（静止、半静止、移动）和 3 种界面类型（Key、Visual、Controls），共 7 次实验，每次 10 试验，总计 8778 次任务完成事件。

**📈 对比分析**

比较方法采用基于 Likert 的易用性问卷（UMUX）、Friedman 检验与 Wilcoxon 符号秩检验（配合 Holm 校正）以及 Kendall W 与 Rank‑Biserial 相关系数作为效应量。结果表明：静止时世界锚定最高效，随着移动性提升用户偏好转向身体锚定，且不同身体锚点在不同任务中表现差异显著。

**⚠️ 局限性**

限制包括：样本量较小、仅覆盖三种界面类型、UI 属性（尺寸、交互频率、位置依赖）难以完全分离、缺乏长期使用或不同环境下的纵向研究、以及默认从世界锚定开始可能影响偏好表达。

---

## 273. SciReC: Diagnostic Evaluation of Multimodal, Multi-Turn Relational Reasoning with Adaptive Interaction

**arXiv ID:** 2608.27461 | [PDF](https://arxiv.org/pdf/2608.27461v1)

**作者:** Nilay Yilmaz `[一作]` (Arizona State University), Yezhou Yang `[通讯]` (Arizona State University)

**通讯引用:** 4688 | [OpenAlex ID](https://openalex.org/A5002278578)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了SciReC模型适应性多模态学术对话基准，用于评估大语言模型在八类关系推理任务上的表现。

**💡 创新点**

创新点在于引入自适应对话流程和DMRA缺陷诊断框架，可量化分析关系推理失败的根本原因，并覆盖多学科与多关系类型。

**🔧 技术方法**

采用多模态大型语言模型（Claude 4.6、GPT‑5.4等）、自适应评估流程、基于softmax的缺陷加权与评估、以及文本生成与评价。

**📊 数据集**

数据集基于OpenStax公开教材（生物、化学、物理、天文学、经济学、心理学、行为神经科学、微积分），共189段对话、656条关系问答。

**📈 对比分析**

通过与多款开源与专有模型对比（Claude 4.6、GPT‑5.4、Qwen‑3.5、Gemma‑3‑27B、Mistral‑3、MiniCPM‑V、InternVL‑3），发现Claude 4.6最高整体准确率73.78%，其余模型差距显著，关系推理是主要瓶颈。

**⚠️ 局限性**

局限在于仅覆盖学术场景、评估过程需API密钥、DMRA框架仅含三类缺陷，难以扩展到更多因素。

---

## 274. Auditing Generative Audio Calls for Known-Task Audio-LLM Evaluation

**arXiv ID:** 2608.27817 | [PDF](https://arxiv.org/pdf/2608.27817v1)

**作者:** Mengzhe Geng `[一作]` `[通讯]` (National Research Council Canada), Mengzhe Geng (National Research Council Canada)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文评估了在已知闭集任务中，使用生成式音频模型（如Qwen2-Audio、Qwen2.5-Omni、MOSS-Audio）相比仅使用转录文本或本地音频编码器（CLAP、AST、WavLM）的效果，并通过“生成式调用决策”框架研究在已获得转录与编码器证据后，进一步调用生成式模型是否能显著提升性能；

**💡 创新点**

创新点在于提出了受控的生成式调用决策测试，将转录、编码器及生成式模型的使用拆分为可测量的决策分支；通过匹配的无调用控制（no‑call selector）对比，评估生成式调用的边际增益；并系统化地量化成本与精度的权衡。

**🔧 技术方法**

技术方面使用了多种音频-语言模型（Qwen2-Audio、Qwen2.5-Omni、MOSS-Audio）、对比式语言–音频预训练模型（CLAP）、Audio Spectrogram Transformer（AST）、WavLM；构建了基于L2正则化的逻辑回归选择器、Bootstrap置信区间、McNemar精确检验以及Holm校正等统计方法。

**📊 数据集**

实验数据集包括Dynamic‑SUPERB VocalSound（6 类人声化录音）、ESC‑50 Animals（动物声音分类）、AudioBench PQA子集（情感语音问答）以及Dynamic‑SUPERB SpeechTextMatching LibriSpeech‑TestClean。

**📈 对比分析**

比较方法：在VocalSound上分别比较转录仅模式（Acc 0.296）、监督编码器（WavLM-emb Acc 0.854）、全功能选择器（Acc 0.925，12.5%调用率）与匹配的无调用选择器（Acc 0.921）。生成式调用的边际增益仅为Δ≈0.004，置信区间跨越0；同时全功能选择器在成本上比无调用模型高出约5.6秒，但相比无调用模型总体成本提升不显著。

**⚠️ 局限性**

局限性：仅针对闭集任务，未评估开放式对话或推理；结果高度依赖ASR、编码器与LLM对口音、方言的偏差；数据集规模有限，特别是ESD样本不足；生成式调用的隐私与成本问题在实际部署中需进一步研究。

---

## 275. When Can Conditional Flow Matching Replace Pointwise Negative Log-Likelihood?

**arXiv ID:** 2608.28010 | [PDF](https://arxiv.org/pdf/2608.28010v1)

**作者:** Yansen Han `[一作]` (Zhejiang University), Tao Lin `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对流匹配中的条件流匹配（CFM）与终点似然的关系进行深入理论分解，提出点位级别的精确分解公式，并在离政策与在政策两种训练场景下检验CFM是否可以代替清晰的似然或比率。

**💡 创新点**

创新点在于：①推导出线性高斯路径下端点负对数似然的完整分解，包括熵、CFM目标、内部速度-分数残差和边界残差；②阐明离政策与在政策环境下，CFM仅估计与CFM差分对真实似然或比率的可行性与局限；③通过比率裁剪与EMA参考更新等机制，探究在政策训练中如何利用偏差较小的CFM比率实现稳定奖励提升。

**🔧 技术方法**

采用的技术主要包括点位残差分解分析、线性高斯路径解析、条件流匹配、对数比率偏差推导、指数加权平均（EMA）与比率裁剪、以及多尺度数值实验评估。

**📊 数据集**

实验数据集涵盖：合成分布（1维单峰、2维双月形、香蕉形、32维高斯混合等）以及真实图像数据集MNIST与CIFAR‑10。

**📈 对比分析**

比较方法：对每种实验设置计算端点NLL误差、CFM仅估计误差、完整分解误差以及奖励/比率MAE；结果显示：完整分解误差显著低于仅用CFM的估计；在离政策训练中，score‑calibrated CFM可将误差降至几十分之一；在在政策训练中，CFM比率虽能显著提升奖励，却仍伴随巨大的比率误差。

**⚠️ 局限性**

局限性：仅在线性高斯路径与正时间平滑假设下验证；在高维（32D）时在政策残差估计不稳定；未提供对残差的严格控制理论；未探讨非高斯路径或更一般条件流匹配的推广。

---

## 276. OpenStamp: A Watermark for Open-Source Language Models

**arXiv ID:** 2608.27899 | [PDF](https://arxiv.org/pdf/2608.27899v1)

**作者:** Miroojin Bakshi `[一作]` (Indian Institute of Science), Danish Pruthi `[通讯]` (Indian Institute of Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种通过在开放源代码LLM的最终投射层加入偏置矩阵来实现水印的技术。

**💡 创新点**

创新点在于将水印逻辑嵌入模型权重，使用投影矩阵实现语义一致性，并通过LLR检测实现高鲁棒性。

**🔧 技术方法**

使用的方法包括在unembedding层添加ΔW=GSP偏置，训练投影矩阵P与选择矩阵S，基于语义嵌入的绿列表G，以及长度归一化的对数似然比检测。

**📊 数据集**

实验数据集主要为C4、Wiki、RealNewsLike、AlpacaEval以及用于微调的OpenWebText和FineWeb。

**📈 对比分析**

与Unremovable、GaussMark、Distilled KGW以及KGW+LLR等基线相比，本文方法在相同困惑度下实现近乎完美的检测（TPR≥99.9%），并在重写、后续微调和量化等攻击下保持较高鲁棒性，文本质量几乎无下降。

**⚠️ 局限性**

主要局限是检测需前向传播算力开销大，且缺乏统计置信度解释，且需要模型内部的token概率。

---

## 277. A Shaky Voice Is Not Always a Dodge: Benchmarking Textual and Vocal Evasion Detection in Earnings Calls

**arXiv ID:** 2608.28040 | [PDF](https://arxiv.org/pdf/2608.28040v1)

**作者:** Mirae Kim `[一作]` (KakaoBank Corp), Youngjun Kwak `[通讯]` (KakaoBank Corp)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建DualEvasion基准，标注收益电话Q&A的文本回避与语音信心双重维度。

**💡 创新点**

首次将文本与音频双模态独立标注，揭示跨模态不一致并研究说话人校准对音频回避检测的影响。

**🔧 技术方法**

使用大语言模型与多模态音频模型进行零样本评估，并采用说话人相对归一化等技术提升音频信心识别。

**📊 数据集**

基于300只股票的2023-2025年收益电话录音，共505条问答对，来自EarningsCall API与WhisperX对齐。

**📈 对比分析**

与人工标注对比，文本回避模型已逼近人类水平，但音频信心检测仍距人类差距显著，最高宏F1仅约58%。

**⚠️ 局限性**

标注规模受限、音频维度仅捕获信心而非全部语音特征、样本量小导致市场关联分析仍属探索性。

---

## 278. MAIL: Memory-driven, Adaptive, Incremental, and Literature-grounded Framework for Hypothesis Generation in Chemistry

**arXiv ID:** 2608.28315 | [PDF](https://arxiv.org/pdf/2608.28315v1)

**作者:** Mahdi Babaei `[一作]` (Stevens Institute of Technology), Yu Gan `[通讯]` (Stevens Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MAIL框架，利用动态检索、内部反馈和多轮推理实现化学假设的自动生成。

**💡 创新点**

创新点在于整合时序文献检索、压缩记忆与自我评估机制，消除了对人工启发库和一次性生成的依赖。

**🔧 技术方法**

使用大语言模型（如GPT‑4o）、PubMed/CrossRef/Semantic Scholar等数据库检索、链式思维提示（CoP）与压缩记忆提示来驱动假设生成。

**📊 数据集**

评估数据集包括公开的TOMATO‑Chem基准和新构建的高创新性Nature/Science挑战集HN‑NS。

**📈 对比分析**

与MOOSE、Zero‑shot Proposer、SciMON、NOVA、ResearchBench等SOTA方法比较，MAIL在MIOS/MPOS、专家评分等指标均表现领先，尤其在高创新性数据集上取得显著优势。

**⚠️ 局限性**

局限包括未经过实验验证、计算成本高、主要聚焦有机/材料化学且存在潜在滥用风险。

---

## 279. Gen-TAS: A Generative AI-Aided Hardware-Software Task Allocation Framework for FPGA-GPP Heterogeneous Systems

**arXiv ID:** 2608.28160 | [PDF](https://arxiv.org/pdf/2608.28160v1)

**作者:** Mary Kong `[一作]` (University of Edinburgh), Themis Prodromakis `[通讯]` (University of Edinburgh)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Gen‑TAS框架，利用大型语言模型（LLM）与检索增强生成（RAG）实现基于用户需求的FPGA‑GPP任务分配，自动生成可执行的硬件‑软件实现；

**💡 创新点**

创新点在于将自然语言设计需求与历史实现知识相结合，通过知识检索驱动LLM生成可解释的多种分配策略，并通过人工选择与确定性后端实现实现可复现的FPGA SoC；

**🔧 技术方法**

核心技术包括：任务图提取、知识库检索、RAG‑LLM推理、可解释分配生成、人工选择、人机交互、确定性Vivado/Vitis HLS与PYNQ实现；

**📊 数据集**

使用的工作负载为CNNImageProc（Fashion‑MNIST CNN）和CAMC（自动调制识别）两大数据集；

**📈 对比分析**

对比方法是将Gen‑TAS产生的分配在不同目标（时延、通信、功耗、资源）下与全GPP、全FPGA基线比较。实验表明，在时延目标下可实现最高2.45×（CNN）和92.53×（CAMC）的加速；在功耗/资源目标下能显著降低LUT/DSP/BRAM使用并保持较好加速；整体在多模型、多目标下具有一致性和可重复性；

**⚠️ 局限性**

局限性包括：知识库覆盖范围有限，缺乏实现反馈导致检索质量受限；仅支持功能级粗粒度分配，未覆盖多加速器或更细粒度划分；实验仅在两种工作负载与单一硬件平台上验证，缺乏更广泛的跨平台与多任务验证。

---

## 280. Marginal Coverage Credit Reduces Redundant Exploration in Parallel State-Entropy Optimization

**arXiv ID:** 2608.27507 | [PDF](https://arxiv.org/pdf/2608.27507v1)

**作者:** Junhao Cao `[一作]` (Hunan Applied Technology University), Ping Guo `[通讯]` (Hunan Applied Technology University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为MCC-PGPSE的并行探索方法，通过对每个策略的留一策略覆盖与状态所有者专化进行评分，从而对辅助奖励进行重新分配，减少冗余探索并提升团队状态空间覆盖。

**💡 创新点**

核心创新在于：①利用留一策略覆盖度量每个策略对团队状态支持的实际贡献；②结合状态所有者专化细化每一步的覆盖信用；③在不改变总奖励量的前提下，对原本的共享团队熵目标之外的辅助奖励进行贡献条件的重分配，从而实现可解释的冗余消除。

**🔧 技术方法**

技术实现基于政策梯度（REINFORCE）与最大状态熵探索，加入在线计数新奇度、前向/后向预测误差、模型不一致度、固定预算仲裁等辅助奖励模块；再通过留一覆盖和专化权重计算，利用保留总奖励的归一化投影重分配奖励；使用两层MLP策略网络，并采用熵正则化。

**📊 数据集**

在4种受控离散网格环境（open-field、bottleneck-memory、branching-specialization、stochastic-loops）以及7个公开离散状态基准（FrozenLake-v1、Taxi-v3、CliffWalking-v0、MiniGrid-Empty-8x8-v0、MiniGrid-DoorKey-8x8-v0、MiniGrid-FourRooms-v0、MiniGrid-LavaGapS7-v0）进行实验；还复现了原始PGPSE的Room/Maze协议。

**📈 对比分析**

与基线（纯熵、计数、新奇度、ICM、RND、仅在线新奇度、Triad（不含MCC））以及对MCC不同组件和信用分配方式的消融控制进行比较。结果显示：MCC-PGPSE在受控任务上显著提升团队熵和状态支持，尤其在瓶颈和分支环境中；在公开基准上虽然单个任务差异不总达到统计显著，但整体平均上提升显著；相比Triad，MCC通过贡献对齐分配获得更高的多样性和覆盖率；MCC的计算成本约比Triad多21%，但仍在可接受范围内。

**⚠️ 局限性**

局限性包括：仅验证于离散状态空间和简易策略网络；缺乏对连续动作、图像观测或Actor-Critic框架的评估；奖励重分配机制在极大策略组或高度重叠探索场景下的可扩展性尚未探究；对训练时计算负担的增加可能限制在需要高吞吐量或资源受限环境中的部署。

---

## 281. From Architecture to Binary: Ensuring Cross-Domain Consistency in Model-Based Airborne Software Development

**arXiv ID:** 2608.28156 | [PDF](https://arxiv.org/pdf/2608.28156v1)

**作者:** Nils Schlautmann `[一作]` (Technical University of Munich), Florian Holzapfel `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在多域（系统、功能、嵌入式）模型化航空软件开发中，设计了一套基于Git仓库和CI流水线的轻量级一致性检查机制，确保接口、模型、代码的跨域一致性。

**💡 创新点**

创新点在于将单一源数据库（dBricks）与专门的接口仓库结合，利用分支与子模块管理、结构化差异分类、自动化通知及多域一致性检查，既避免了繁重的流程门槛，又实现了小团队快速迭代的可行性。

**🔧 技术方法**

采用了GitLab CI、Docker容器、MATLAB/Simulink、mrails、tlcodegen等工具；通过子模块引用、版本锁定、自动生成的ICD XML和Simulink空壳模型来实现跨域同步。

**📊 数据集**

使用的主要数据集为dBricks数据库中的ICD XML、由脚本生成的Simulink空壳模型、功能集成模型和最终的可执行二进制；数据来源为项目实际的系统、功能、嵌入式三域工件。

**📈 对比分析**

方法通过在每次提交和合并请求时运行一致性检查、差异归类、自动化构建与测试来评估；目前仅获得定性反馈，显示接口更新后不一致性显著降低、手工差异比对时间缩短；尚未给出定量性能指标。

**⚠️ 局限性**

局限性包括：方案尚未完全部署到生产；缺乏量化的效率与质量评估；跨设备通信一致性检查尚未完善；依赖于手工配置的邮件通知和DBricks API；对大型多平台项目的可扩展性与成熟度待验证。

---

## 282. Deceptive Patterns as a Sociotechnical Phenomenon: Review, Catalog, and Discussion

**arXiv ID:** 2608.27684 | [PDF](https://arxiv.org/pdf/2608.27684v1)

**作者:** Luiz Adolpho Baroni `[一作]` (Federal University of Paraná), Roberto Pereira `[通讯]` (Federal University of Paraná)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对“deceptive patterns”进行社会技术层面的系统化研究，完成了从文献映射到理论框架分析，再到构建并评估可自解释的在线目录；

**💡 创新点**

首次将Ronald Stamper的Semiotic Framework应用于该领域，形成了完整的六层分析视角，并通过系统映射梳理了126种模式，提出了第一套基于该框架的分类与教育工具；

**🔧 技术方法**

使用半结构化文献检索与分析（ACM/IEEE/Scopus），Semiotic Framework层面编码，Heuristic Evaluation、TAM问卷、焦点小组等混合方法；

**📊 数据集**

以97篇计算机科学期刊/会议文献为基础的文献映射数据集，生成的126种模式与11类目录作为主要数据集；

**📈 对比分析**

与以往仅聚焦伦理或技术单层的研究相比，本工作通过六层框架提供了更系统的分类与解释；目录在可用性和实用性问卷中获得平均4.6/5的评分，表明工具易用且被认知为有价值；

**⚠️ 局限性**

研究受限于英语文献范围、未包含“manipulative”等关键词、仅以单一“Roach Motel”实例进行社会技术分析、评估样本仅为七名HCI专业人士，缺乏更广泛用户群体的验证。

---

## 283. Distributed Model-Based Diffusion: Finite Horizon Contraction under Bounded Delay

**arXiv ID:** 2608.27685 | [PDF](https://arxiv.org/pdf/2608.27685v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 284. Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models

**arXiv ID:** 2608.27550 | [PDF](https://arxiv.org/pdf/2608.27550v1)

**作者:** Senqiao Yang `[一作]`, Jiaya Jia `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在机器人视觉-语言-动作(VLA)模型中，作者提出了一种以表征为中心的后续预训练方法VLAct，旨在利用有限的机器人轨迹数据构建可跨多种机器人外观、任务和环境迁移的通用 VLM 后端。

**💡 创新点**

创新点在于：① 通过浅层保护和标题混合预训练保持原始 VLM 的视觉-语言先验；② 用多头连续动作共监督（OFT、PI、GR00T）避免单一动作头导致的表征坍塌；③ 采用部分统一的跨外观动作空间（共享可比动作维度，掩蔽无效维度）以及周期角度的 wrap-aware 损失，提升跨外观的一致性与稳定性。

**🔧 技术方法**

技术实现主要包括：VLM 后端冻结视觉编码器与下半 LLM 层、在预训练中混合机器人轨迹与标题数据、三头动作共监督的联合损失、跨外观部分统一动作布局、周期角度损失以及在微调阶段仅保留后端并为任务自由初始化动作头。

**📊 数据集**

使用公开机器人数据集：DROID、InternA1、RoboCoin、MolmoAct 以及用于 VLM 预训练的标题数据；在评估中覆盖 LIBERO-Plus、VLA-Arena、RoboTwin 2.0、DOMINO、RoboDojo、RoboCasa-GR1 等多种仿真与真实机器人基准。

**📈 对比分析**

与多种现有 VLA 系统（π_0、π_0.5、ABot-M0、LingBot-VLA 等）以及工业系统比较，VLAct 在 LIBERO-Plus 取得 82.6% 总分（比 ABot-M0 高 2.1%），在 RoboTwin 2.0 Clean 与 Random 分别达到 92.5% 与 90.8%，在 RoboDojo 榜单中排名第六（成功率 7.60%），在未见外观 RoboCasa-GR1 仅用 20% 下游轨迹就超过全数据基线。

**⚠️ 局限性**

局限性包括：① 对 VLM 预训练数据的依赖仍显著；② 只在 16 GPU 规模下训练，缺乏大规模可扩展性验证；③ 对高度动态或长时序记忆任务（如 RoboDojo 的 Memory 子任务）表现仍不理想；④ 跨外观的动作空间统一仍需手工设定，自动化程度有限。

---

## 285. Quantization-Triggered Backdoors in Language Models: Cross-Quantizer Transferability and the Validation--Deployment Gap

**arXiv ID:** 2608.27512 | [PDF](https://arxiv.org/pdf/2608.27512v1)

**作者:** Jacopo Dardini `[一作]` (University of Bologna), Giuseppe Fenza `[通讯]` (University of Salerno)

**通讯引用:** 2457 | [OpenAlex ID](https://openalex.org/A5016277608)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了后训练量化触发的后门攻击，揭示了验证-部署之间的结构性差距。

**💡 创新点**

提出Quantization Behavioral Equivalence Classes（QBEC）理论，并证明其不保证行为等价，展示跨量化器转移性与修复策略对攻击持久性的影响。

**🔧 技术方法**

使用三阶段攻击框架（恶意微调、量化约束计算、投影梯度修复），并结合LLM.int8()、NF4等量化方案进行实验。

**📊 数据集**

在翻译任务中使用NLLB‑200‑1.3B和M2M100‑1.2B（英乌语料），在政治内容分析中使用Llama‑3.2‑1B和Gemma‑3‑1B，并利用合成数据集和PoliTune进行训练与评估。

**📈 对比分析**

通过BLEU/IFF‑CSR（翻译）和MMLU/ΔBias（政治分析）对比，量化后模型在翻译任务中友敌识别成功率可达85%，在政治分析中偏见漂移ΔBias高达0.33，且在全精度时指标几乎无损。

**⚠️ 局限性**

局限包括合成数据的真实性、单次实验缺乏置信区间、仅在约1B规模模型验证、对更大模型行为未知，以及攻击者与防御者视角的对齐不足。

---

## 286. MuSP-Bench: Advanced Multimodal Benchmarking of Music Understanding across Score and Performance

**arXiv ID:** 2608.28212 | [PDF](https://arxiv.org/pdf/2608.28212v1)

**作者:** Milan Liessens Dujardin `[一作]` (Bryel Labs), Kevin Miao `[通讯]` (Bryel Labs)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MuSP-Bench，一个包含490个多模态音乐问题的基准，用于考察音乐乐谱与演奏之间的理解与推理；

**💡 创新点**

创新点在于人类专家编写的跨模态、长时段推理问题，涵盖乐谱、音频及其相互关系，且覆盖乐理、表演、解释等多维内容；

**🔧 技术方法**

采用多模态大型语言模型（GPT‑5.6、Qwen3.5、Muse Spark、Qwen3.6、Audio Flamingo Next）在文本、乐谱图像、MIDI‑文本、音频等不同输入格式下进行评估；

**📊 数据集**

使用18首完整古典钢琴曲/乐章与6段管弦乐片段的乐谱（来自ASAP、PianoCoRe MIDI、BSED）及其对应音频；

**📈 对比分析**

比较方法是将各模型在不同输入格式下的准确率汇总，结果显示：结构化符号表示（ABC/MIDI‑文本）在分析任务上表现最佳；图像和音频在风格/全局属性上更优；整体准确率仍低，模型在表演音频与跨模态推理上表现尤其差；

**⚠️ 局限性**

局限性包括：数据集仅包含古典钢琴与管弦乐，缺乏多样性；评测仅以准确率为指标，未考察生成质量与解释性；模型对音频与图像的理解仍不成熟，难以捕捉细粒度表现与结构细节。

---

## 287. CNeo-Bench: Diagnosing Large Language Models on Chinese Neologisms

**arXiv ID:** 2608.28053 | [PDF](https://arxiv.org/pdf/2608.28053v1)

**作者:** Kaiyan Zhao `[一作]` (University of Tokyo), Yoshimasa Tsuruoka `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了包含4759个中文新词的CNeo-Bench基准，并设计了两层评测框架，评估LLM对新词的理解与操作能力。

**💡 创新点**

首次按词形生成机制对中文新词进行分类，并揭示了模型在描述与恢复源形态之间的“识别-操作”差距。

**🔧 技术方法**

采用了两阶段评测（定义生成与机制诊断）和LLM判定器、精确匹配及多项选择评分方式。

**📊 数据集**

数据来源于Moegirlpedia、Wikiversity等公开词典和网络爬取，共4759条新词及其定义。

**📈 对比分析**

在18款LLM上测试，最佳模型Kimi‑K2.5在定义生成上仅达67.7%，在机制诊断上81.9%；整体表现仍显不足。

**⚠️ 局限性**

评测依赖LLM判定器，数据规模有限，部分子类别样本不足，且未给出解决识别-操作差距的方案。

---

## 288. FISGuard: Defending Against Membership Inference via Fixed Input Subspaces

**arXiv ID:** 2608.27836 | [PDF](https://arxiv.org/pdf/2608.27836v1)

**作者:** Haocheng Jiang `[一作]` (Hubei University of Technology), Hua Shen `[通讯]` (Hubei University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究针对联邦参数高效微调中的ProjRes成员推断攻击，提出一种通过固定输入子空间重参数化的轻量级防御方法FISGuard，能够在不影响模型下游性能的前提下显著降低隐私泄露风险。

**💡 创新点**

创新点在于先利用公开数据通过SVD构造低维固定投影，将可训练线性层拆解为固定输入投影与可训练输出映射，从而消除高维私有表示对梯度子空间的直接贡献，完全切断ProjRes利用投影残差的攻击途径。

**🔧 技术方法**

技术手段包括：联邦学习框架FedAvg、参数高效微调（Adapter/LoRA）、公共数据投影构造（SVD）、固定输入子空间重参数化、梯度投影残差攻击评估以及对比实验。

**📊 数据集**

实验使用了三大NLP数据集CoLA、SST-5、IMDb，以及两款大型语言模型BERT-Base和GPT-2 Large进行验证。

**📈 对比分析**

与DP-SGD、L2正则化、Min-Max对抗训练、Dropout、Label Smoothing等五种主流防御方法对比，FISGuard在12种配置下将ProjRes攻击AUC降至≈0.5（接近随机猜测），同时保持下游任务准确率与未防御模型相近，且仅增加极小的计算开销。

**⚠️ 局限性**

局限性在于仅针对ProjRes投影残差型攻击提供防御，未给出差分隐私等通用隐私保证，对更强或其他形式的成员推断攻击效果未知，并且需要足够的公共数据和合理的子空间维度以保证防御效果。

---

## 289. String: An Agentic OS Where Every App Is a Markdown File

**arXiv ID:** 2608.28027 | [PDF](https://arxiv.org/pdf/2608.28027v1)

**作者:** Jookyung Song `[一作]` (Seoul National University), Simyung Chang `[通讯]` (H1R.AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了String运行时和SFMD（可执行Markdown）格式，提供统一、递归渲染的LLM代理接口，显著降低上下文负载并实现工具调用的模块化；

**💡 创新点**

创新点在于把接口视作可执行Markdown文件，采用部分暴露、统一表面以及递归渲染三大原则，打造可与多种计算环境无缝交互的OS‑层级运行时；

**🔧 技术方法**

使用TypeScript实现SFMD解析器、编译器与运行时，配合HTTP、shell、JSON等工具执行，核心语法为SFMD Markdown及其指令；

**📊 数据集**

评估基于SkillsBench v1.1的87项任务数据集，比较String app与传统技能实现；

**📈 对比分析**

通过在六种模型上对比no‑skills、SkillsBench技能和String app三种条件，发现String app在保持相近成功率的同时平均节省33.5% token，错误动作选择率显著下降；

**⚠️ 局限性**

目前仅支持单用户、回环、无认证，权限、签名、审计等功能仍在规划中。

---

## 290. Operationalizing Regulations into Code: A Model to Enhance Governance and Compliance in LLM Selection for Software Engineering

**arXiv ID:** 2608.27703 | [PDF](https://arxiv.org/pdf/2608.27703v1)

**作者:** Jonysberg Quintino `[一作]` (Universidade Federal de Pernambuco), Filipe Calegário `[通讯]` (Universidade Federal de Pernambuco)

**通讯引用:** 211 | [OpenAlex ID](https://openalex.org/A5043506121)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种三层治理模型（法规、组织治理、生产力与可持续性）用于在软件工程项目中选择大型语言模型（LLM），并引入KO门槛和加权风险指数来评估合规与技术风险。

**💡 创新点**

创新点在于：①将欧盟AI法、NIST AI RMF、GDPR、LGPD、ISO/IEC 42001等多域法规转换为可操作的KO和加权评估标准；②采用多准则决策分析（MCDA）与加权总和模型，结合KO门槛实现风险隔离；③设计PAG-LLM评估协议，将CWE/OWASP场景映射为可度量的攻击测试；④引入法规反馈循环实现持续更新。

**🔧 技术方法**

技术手段包括：设计科学研究（DSR）方法、MCDA（加权总和）、KO门槛逻辑、PAG-LLM对抗场景评估、法规映射与反馈机制。

**📊 数据集**

数据集：20个基于CWE与OWASP Top 10的结构化攻击场景，用于评估两种LLM的安全与合规表现。

**📈 对比分析**

比较方法：对两种模型（云端商业LLM与本地开源LLM）进行PAG-LLM测试，计算KO满足情况、各支柱评分与加权风险指数；结果显示两者均因KO失效被排除，IRP虽高但未能通过门槛，说明KO门槛对风险控制有效。

**⚠️ 局限性**

局限性：仅评估两款模型与20个场景，样本规模有限；缺乏评审者间一致性度量；KO与权重设置尚未专家验证；缺少更广泛的模型与场景验证；未来需扩展评估、验证权重、加入自动化分析。

---

## 291. Agentic Artifact Creation: Systems, Evaluation, Principles, and Opportunities

**arXiv ID:** 2608.28122 | [PDF](https://arxiv.org/pdf/2608.28122v1)

**作者:** Tianfu Wang `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文对“Agentic Artifact Creation”（代理式工件生成）进行了系统性综述，提出了统一的范式定义和功能架构，并对 230 多个系统及 29 个基准进行了编码与分类，分析了六大工件族（文本、二维视觉、音频、视频、空间、行为）的构建机制、评估方法和应用场景，归纳了四项核心原则与研究展望。

**💡 创新点**

创新点在于：①首次将工件生成从单向流水线转为“状态-反馈-决策”循环的代理式构建范式；②提出三大功能角色（Operational Representation、Construction Policy、Runtime Verification）并对其交互进行建模；③构建了超过 200 个相关系统的可编码语料库，提供了跨工件族的比较基准；④从工件族、应用场景、评估指标和技术实现四个维度系统化评估，形成了可操作的研究路线图。

**🔧 技术方法**

技术手段包括：系统搜索（arXiv、Google Scholar、Semantic Scholar、ACM DL、IEEE Xplore），关键词交叉检索、引用追踪；论文筛选与二次检索；手工编码与跨来源对照；功能架构分析与角色定义；对比框架构建（工件族、应用场景、评估协议、技术实现）。

**📊 数据集**

主要数据集为内部构建的“Agentic Artifact Creation Corpus”，包含 230 条系统记录（按工件族、技术细节等标签编码）和 29 条构建基准（包含任务集合、评估指标等信息）。

**📈 对比分析**

本文采用定性对比与量化指标两种方法：①对 6 大工件族在表示、编辑、验证和反馈等维度进行交叉对照，绘制功能对齐图；②对 29 个基准中已有的评估结果进行汇总，评估各系统在需求满足率、修复局部性、过程可追踪性、资源效率等方面的表现；整体上展示了不同工件族与技术组合在接受标准、可修复性和评估可重复性方面的差异，但未提供统一的性能基准。

**⚠️ 局限性**

局限性包括：①覆盖时间截止 2026 年 8 月，后续工作未被纳入；②对“代理式工件生成”的定义与范围存在一定主观性，部分系统可能被误归或漏归；③主要聚焦工件中心的系统，未覆盖仅进行评估或工具协同的非代理系统；④未对单个系统进行量化跑分，缺乏统一的客观性能对比；⑤语料库主要来源于公开论文，可能存在重复与遗漏。

---

## 292. Under-Mattress Temporal Sensing for Next-Day Agitation Risk Scoring in Dementia Wards

**arXiv ID:** 2608.28152 | [PDF](https://arxiv.org/pdf/2608.28152v1)

**作者:** Zhen Liu `[一作]` (KU Leuven), Maarten De Vos `[通讯]` (KU Leuven)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对医院痴呆症患者的床下无接触传感器记录进行分钟级分析，构建四种表示管线，评估前一晚信号对次日激动风险的预测。

**💡 创新点**

首次在床下传感器数据上引入分钟级时序建模与滑动窗口多实例学习，并比较时序压缩程度对激动风险预测的影响。

**🔧 技术方法**

使用随机森林、逻辑回归、InceptionTime、Transformer+注意力多实例学习等模型，配合SHAP特征选择、滑动窗口编码和患者分组交叉验证。

**📊 数据集**

利用423晚、65名患者的EMFIT和Withings Sleep Analyzer传感器数据（心率、呼吸率、活动、睡眠/觉醒状态和质量通道），并结合Pittsburgh Agitation Scale次日激动评分。

**📈 对比分析**

通过患者分组的5折交叉验证聚合OOF预测，计算AUROC、AUPRC、Brier、灵敏度、特异性等指标；全序列模型AUROC 0.692，滑动窗口0.679，均优于全夜摘要0.565，表现为中等区分度但校准有限。

**⚠️ 局限性**

样本来源单中心、患者数有限、激动标签宽泛且缺乏一致性、模型选择未外部验证、校准不足，难以直接用于临床决策。

---

## 293. See, Hypothesize, Validate: Multimodal Agentic Framework for Discovering Governing PDEs

**arXiv ID:** 2608.27869 | [PDF](https://arxiv.org/pdf/2608.27869v1)

**作者:** Sarang Manoj Pekhale `[一作]` (Indian Institute of Technology Delhi), Souvik Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多模态代理框架MAGE，用来从观测数据中自动发现控制偏微分方程的结构和系数。

**💡 创新点**

创新点在于把方程发现拆分为四个专门化代理（微分观测、现象提取、方程合成、验证仲裁）并采用信心驱动的假设验证循环，兼顾视觉证据、符号生成和数值反证。

**🔧 技术方法**

使用的技术包括：视觉语言模型（VLM）提取物理语义、大型语言模型（LLM）生成无库符号方程、差分与小波去噪的数值微分、弱形式积分与最小二乘拟合以及多轮迭代的拒绝-重试策略。

**📊 数据集**

数据集包括经典PDE基准（NLS、KS、Allen–Cahn等8个方程）、两种复杂几何（2D板、3D航天器）以及一条实验传感器记录（Silverbox电路）。

**📈 对比分析**

与传统稀疏回归、符号回归、LLM驱动和现有代理方法对比，MAGE在8/8经典基准上实现100%结构回收，系数误差平均低约3个数量级；在复杂几何中成功恢复拉普拉斯算子；在实验数据中得到高R²=0.985的三次恢复。

**⚠️ 局限性**

局限性包括依赖未微调的预训练VLM/LLM，验证仲裁在噪声大时速度慢，实验评估仅覆盖单一传感器记录和部分几何，且数值微分对噪声高度敏感。

---

## 294. Circuit Discovery Helps Detect LLM Jailbreaking: A Mechanistic Interpretability Study

**arXiv ID:** 2608.27504 | [PDF](https://arxiv.org/pdf/2608.27504v1)

**作者:** Paria Mehrbod `[一作]` (Concordia University), Geraldin Nanfack `[通讯]` (Concordia University)

**通讯引用:** 317 | [OpenAlex ID](https://openalex.org/A5034693860)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大规模安全对齐的LLM LLaMA-2-7B-chat-hf中的“jailbreak”行为进行了机制解释性分析，系统地发现并定位了导致模型正面回答的计算子网络（电路）。

**💡 创新点**

创新之处在于首次将电路发现技术应用于大模型的jailbreak行为，揭示了关键注意力头和MLP通路，并证明通过消除该子网络可将攻击成功率降低多达80%。

**🔧 技术方法**

主要技术包括梯度近似的Edge Attribution Patching（EAP）与子网络探测（Subnetwork Probing）等电路发现方法，以及激活补丁技术。

**📊 数据集**

使用了基于改进的Greedy Coordinate Gradient（GCG）在HarmBench数据集上生成的jailbreak提示，构成100条训练样本和50条测试样本。

**📈 对比分析**

通过与随机首字节添加的基线对比以及对不同答案长度的子网络探测，实验显示消除电路后拒绝率显著提升，子网络探测在小规模电路上信度更高，整体可将攻击成功率降低80%。

**⚠️ 局限性**

局限性包括未评估电路对非jailbreak任务的影响、消除电路后模型在前几词产生重复或无意义输出，以及仅在首词阶段进行消融，未来需进一步完善消融策略和电路完整性评估。

---

## 295. Fully Unleashing the Multimodal Attacker: Meta-Adaptive Jailbreaking of Vision-Language Models

**arXiv ID:** 2608.27531 | [PDF](https://arxiv.org/pdf/2608.27531v1)

**作者:** Benlei Cui `[一作]` (Yuvion Team, Alibaba Group), Haiwen Hong `[通讯]` (Yuvion Team, Alibaba Group)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Meta‑Adaptive Multimodal Jailbreaking (MAMJ)框架，优化攻击策略提示（ASP）和攻击者模型权重，实现在多模态输入中对大视觉语言模型进行动态、可迁移的攻击。

**💡 创新点**

首次把攻击本身的适应性提升到meta层面，实现双轴优化：先通过LLM诊断迭代改进ASP，再用组级奖励更新攻击者参数；学习到的攻击器可无须重新训练迁移到不同受害模型与防御。

**🔧 技术方法**

利用LLM进行失败轨迹诊断与策略修正；组相对优势估计（GRPO）进行强化学习；格式有效性门控；多模态生成流水线与安全评分器联合使用。

**📊 数据集**

主要在MM‑SafetyBench（13个安全场景）上进行训练与评估，同时在SafeBench等公开基准做迁移验证。

**📈 对比分析**

与模板攻击、单样本迭代攻击（VisCo、IDEATOR等）以及基于prompt的防御（AdaShield、VLMGuard、Llama‑Guard‑4、LlavaGuard）进行对比；MAMJ在MM‑SafetyBench上攻击成功率（ASR）分别为81.0%、78.9%和82.3%，比最强基线提升约20–24个百分点；在不同防御下也保持最高性能。

**⚠️ 局限性**

受限于仅在MM‑SafetyBench评测；训练成本主要由外部图像生成API支撑，昂贵且受限于服务；未对图像生成器进行联合优化，可能限制进一步提升；对更广泛攻击面与基准的泛化尚未系统验证。

---

## 296. Beyond Global Scalars: Synergizing Token-Level Statistics and Deep Semantics for Adversarial AIGC Text Detection

**arXiv ID:** 2608.28009 | [PDF](https://arxiv.org/pdf/2608.28009v1)

**作者:** Peiming Li `[一作]` (Tencent BAC), Yang Tang `[通讯]` (Tencent BAC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种融合统计轨迹与语义特征的端到端机器生成文本检测框架NeuroStat，并构建覆盖36种攻击的对抗基准MOSAIC。

**💡 创新点**

创新点在于：①保留完整的token级概率轨迹并通过宏状态残差调制(MSRM)动态放大局部异常；②利用正交与对比损失实现两支路特征互补；③创建全粒度、跨模型的对抗数据集MOSAIC。

**🔧 技术方法**

技术包括：因果语言模型（Qwen2/3.5）提取logits与隐藏状态；1D-CNN+MSRM捕获概率异常；注意力池化提取语义特征；交叉熵+监督对比损失+正交约束的联合训练。

**📊 数据集**

使用的主要数据集：MOSAIC（16000人机混合文本，6大模型交叉生成），MIRAGE（生成/改写/润色任务），ImBD（人类写作与AI润色对比）。

**📈 对比分析**

与11种主流检测器（TF如Likelihood、Entropy，TB如RoBERTa、ImBD、DetectAnyLLM）对比，NeuroStat在MOSAIC 8类攻击的AUROC均超过最优基线，整体AUROC高达约83–94%，TPR@5%提升显著，显示对抗鲁棒性最强。

**⚠️ 局限性**

局限性：仅在英语数据上评估，跨语言/专业领域效果未知；需完整logit矩阵，限制闭源API直接使用；模型推理时内存和延迟略高，约比传统序列分类器高10%。

---

## 297. VISTA: Verifier-Informed Student-to-Teacher Adaptation for On-Policy Self-Distillation

**arXiv ID:** 2608.28306 | [PDF](https://arxiv.org/pdf/2608.28306v1)

**作者:** Zewen Ding `[一作]` (University of Science and Technology of China), Linli Xu `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 VISTA 方法，在 on‑policy self‑distillation（OPSD）框架中让学生的分布在通过验证器确认的轨迹上反馈给教师，以实现学生到教师的自适应学习。

**💡 创新点**

创新点在于：①使用结果验证器门控仅对已通过验证的轨迹进行教师更新；②在轨迹上仅对教师-学生 KL 散度最高的 top‑k 位置进行更新，从而保持教师对学生的引导优势且避免过度收敛。

**🔧 技术方法**

技术包括：on‑policy rollouts、token‑级 KL 监督、stop‑gradient 分离教师与学生梯度、结果验证器（deterministic 检查）与 top‑k KL 位置掩码、正向与逆向 KL 目标结合的 VISTA 损失。

**📊 数据集**

使用 OpenThoughts 训练语料库，在 AIME 2024、AIME 2025、HMMT 2025 三个数学竞赛数据集上评估。

**📈 对比分析**

与基线（Base、SFT、GRPO、SDPO）以及标准 OPSD 对比，VISTA 在 Qwen3‑1.7B、4B、8B 三个规模上均获得最高 Avg@12，提升幅度分别为 0.6、0.7、2.1 分，且在 9 个规模‑任务组合中获得 8 项新的 state‑of‑the‑art 结果。

**⚠️ 局限性**

局限性包括：①教师更新仅在已验证轨迹上进行，无法利用未验证轨迹的潜在信息；②top‑k 选择需要手动调参（k 的取值对性能影响显著）；③在极大模型或极高难度任务中，教师仍可能因信息偏差产生错误引导，需要进一步研究更鲁棒的验证与适配策略。

---

## 298. A Deeper Analysis of Block-Sparse Featurizers

**arXiv ID:** 2608.27515 | [PDF](https://arxiv.org/pdf/2608.27515v1)

**作者:** Alexandru-Iulius Jerpelea `[一作]` (Columbia University), Amith Ananthram `[通讯]` (Columbia University)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5000658547)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并改进块稀疏特征器（BSF），通过锦标赛 Top‑K 规则、混合维度块和专用特征列等技术，提升其在合成与真实视觉模型中对低维流形特征的捕捉与解释能力。

**💡 创新点**

提出锦标赛 Top‑K 选择机制以抑制特征分裂，结合块维度自适应与专用特征列解决跨编码器合并问题，显著提升块稀疏自编码器在流形特征上的完整性。

**🔧 技术方法**

使用块稀疏自编码器的 Grassmannian 变体，块级稀疏约束、几何重叠度量、温度衰减、以及跨编码器的池分区策略。

**📊 数据集**

在“Manifold Zoo”合成数据（128 条直线/曲线流形）上进行验证，并在 DINOv3 与 CLIP 的真实视觉模型上进行实验。

**📈 对比分析**

通过与参数匹配的传统 SAE 及其后处理方法（ValuePCA、部分相关分组）比较，使用 R² 评估重建精度；BSF 在无相关数据上取得 0.77 R²，锦标赛 Top‑K 将成功率提升至 0.90；在相关数据上亦显著优于 SAE。

**⚠️ 局限性**

仍存在块间近正交的特征分裂、混合维度块难以自适应分配、跨编码器合并问题等局限，需要进一步改进以实现更完美的特征捕捉。

---

## 299. Manifold4D: Denoising on Point Cloud Rendered Manifolds for Video Re-shooting

**arXiv ID:** 2608.28174 | [PDF](https://arxiv.org/pdf/2608.28174v1)

**作者:** Yongqi Mao `[一作]` (Zhejiang University), Guotao Meng `[通讯]` (Manifold Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种视频重拍（Video re‑shooting）模型，通过在流匹配（flow matching）训练的初始噪声中注入点云渲染，仅使用源视频作为视觉条件，从而实现对单目视频的轨迹控制与视觉质量的平衡。

**💡 创新点**

核心创新在于将像素对齐的几何先验一次性注入到生成起点，而非在整个去噪过程中持续作为条件输入，从而消除了“信任困境”，显著提升轨迹控制精度且不牺牲视觉质量。

**🔧 技术方法**

技术包括：基于 Wan2.1‑T2V 的流匹配视频扩散模型、4D 点云重建与渲染、噪声注入策略（x̃₁ = +σ ε）、以及训练时的相机 Plücker 编码与注意力投影。

**📊 数据集**

训练集由 36K 条多视角视频（DL3DV、DynPose、OpenVid‑HD、MultiCamVideo、HuMMan）构成；评测基准为 DAVIS‑Traj、Vista4D‑Eval 以及真实 iPhone 多视角数据。

**📈 对比分析**

与 ReCamMaster、TrajectoryCrafter、GEN3C、Vista4D 等方法对比，本文方法在旋转误差降低 25–27%，平移误差降低多达 32%，并在所有基准上获得最优或最接近渲染的轨迹精度，视觉质量与基准持平或略逊。

**⚠️ 局限性**

主要局限是打破了扩散模型对高斯噪声起点的假设，导致训练更为困难；同时，当前噪声注入方式仍可进一步优化，以提升对极端运动或渲染缺陷的鲁棒性。

---

## 300. Shortest self-orthogonal and LCD embeddings of linear codes over Fq+uFq

**arXiv ID:** 2608.28222 | [PDF](https://arxiv.org/pdf/2608.28222v1)

**作者:** Junmin An `[一作]` (Sogang University), Jon-Lark Kim `[通讯]` (Sogang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文确定了线性码在环 R=𝔽_q+u𝔽_q 上的最短自正交和 LCD 嵌入的确切长度。通过将 Gram 矩阵分解为有限域 𝔽_q 上的对称矩阵对，嵌入问题被简化为对称和交替矩阵的同余分类。

**💡 创新点**

创新点在于为线性码的最短自正交嵌入长度提供了完整的公式，并且在偶数和奇数特征下分别得到了两个不同的情况。此外，利用 Witt 理论构造了所有最短自正交嵌入，并且建立了最短 LCD 嵌入的完整表征。

**🔧 技术方法**

使用了同余理论、Witt 理论以及对称和交替矩阵的分类技术。

**📊 数据集**

使用了线性码在环 R=𝔽_q+u𝔽_q 上的生成矩阵，具体数据集未明确给出，但提供了多个示例以说明结果。

**📈 对比分析**

通过与已有的嵌入方法进行比较，本文展示了所提出方法的有效性，并且在多个示例中找到了具有最大最小距离的最短自正交嵌入。

**⚠️ 局限性**

限制在于嵌入可能会改变原始码的类型，因此未来的研究方向是探讨在保持原始码类型不变的情况下的最短嵌入问题。

---

## 301. RESTCov: A Tool for Structural Coverage Analysis of REST APIs

**arXiv ID:** 2608.28114 | [PDF](https://arxiv.org/pdf/2608.28114v1)

**作者:** Tolgahan Bardakci `[一作]` (University of Antwerp and Flanders Make), Serge Demeyer `[通讯]` (University of Antwerp and Flanders Make)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了 RESTCov，能够在不访问实现代码的情况下，通过 OpenAPI 规范和 HTTP 日志进行黑盒 REST API 结构覆盖率分析。

**💡 创新点**

创新点在于不需要源代码 Instrumentation，支持多维度覆盖率（路径、操作、参数、媒体类型、状态码/状态类），并将未匹配请求直接集中展示以便发现规范与日志的漂移。

**🔧 技术方法**

采用 Python 编写，利用 OpenAPI 解析库、正则表达式路径匹配、JSON 统计和 HTML 报告生成等技术实现。

**📊 数据集**

使用 Petstore 示例 API 以及 Google Drive、Spotify 等五个真实生产 API 作为评估数据集。

**📈 对比分析**

与 Restats 等工具对比，RESTCov 在多种 API 上能够完整覆盖路径和操作，提供更完整的未匹配请求信息；覆盖率低于功能正确性验证，但性能稳健且易于集成到 CI/CD。

**⚠️ 局限性**

局限性包括仅报告结构覆盖，未验证功能正确性；日志存储为配对文本文件，对大规模日志处理不够友好；缺乏对 HAR/JSONL 等集中日志格式的支持。

---

## 302. Coverage, Not Credit: Failure-Credit Routing of Zeroth-Order Perturbation Budgets Does Not Improve On-Pool Sample Efficiency for LLM Agents

**arXiv ID:** 2608.28011 | [PDF](https://arxiv.org/pdf/2608.28011v1)

**作者:** Yuxu Ge `[一作]` `[通讯]` (University of York), Yuxu Ge (University of York)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估轨迹级失败信用分配策略（credit routing）在零阶/进化策略（ZO/ES）优化冻结LLM代理时对样本效率的影响，并系统记录三种潜在失败模式。

**💡 创新点**

系统化检验信用路由对优化效率的实际影响，提出星空回归（starvation regression）和覆盖floor机制作为解决方案，并在不同模型、任务和预算下验证其普适性。

**🔧 技术方法**

采用零阶/进化策略（ZO/ES）+镜像采样、模块化低维子空间投影、EMA信用向量、TOST等效检验、逆倾向加权（IPW）等技术。

**📊 数据集**

使用的数据集包括：synthetic pre‑study、四类工具使用任务（算术、知识检索）、Qwen2.5‑1.5B/3B、SmolLM2‑1.7B、以及BFCL（函数调用）基准。

**📈 对比分析**

对统一分配、软、硬、随机等六种分配方案在固定任务池上的时间平均成功率（AUC）进行对比；结果显示信用路由无显著提升，甚至在部分配置下降；覆盖floor能消除大部分损失；在单个held‑out BFCL评估中软路由略有提升。

**⚠️ 局限性**

局限性：仅评估固定任务池下的样本效率；样本量小（n≤8）导致等价边界±0.02；未检验每层非绑定子空间或增量式信用估计；BFCL的relay机制可能导致结果偏差；跨模型、任务的泛化性未充分验证。

---

## 303. ReToolSQL: Agentic Reinforcement Learning for Robust Text-to-SQL

**arXiv ID:** 2608.27796 | [PDF](https://arxiv.org/pdf/2608.27796v1)

**作者:** Pratik Kakkar `[一作]` (JPMorganChase), Anup Shirgaonkar `[通讯]` (JPMorganChase)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用单一的密集 31B Gemma 模型，在训练阶段结合了两步：先用拒绝抽样的教师轨迹做监督微调（SFT），再用 GRPO 方式在包含 SQL 执行、列统计和 BM25 值检索等只读工具的环境中进行强化学习（RFT），使模型在单一推理回合即可完成提问→生成→验证→修正的完整流程。

**💡 创新点**

创新点主要在于：①将执行反馈直接嵌入到 RL 环境中，形成真正的提案-验证-修正循环；②构造两阶段 SFT→RFT 的训练管线，让监督扩展覆盖难题，RL 则提高单通道准确率；③使用 DAPO 动态采样和 beta 调度等技术保持熵与梯度信号，解决大模型 RL 训练中的熵坍塌问题。

**🔧 技术方法**

核心技术包括：Gemma 4 31B instruction-tuned 语言模型；GRPO（Group‑Relative Policy Optimization）+ DAPO（动态采样）与不等式裁剪；基于执行的多元奖励（语法、执行、表/列覆盖、长度惩罚等）；工具集（沙箱 SQL 执行器、列剖析器、BM25 值搜索）；自我一致性（self‑consistency）推理后聚合。

**📊 数据集**

使用 BIRD‑SQL 基准，训练集 6,601 题（69 个数据库），验证集 1,534 题（11 个数据库）以及对应的难度划分；无额外人工标注，仅利用 BIRD 本身提供的 gold SQL 进行奖励。

**📈 对比分析**

对比实验展示：SFT 单独提升到 72.69%；RFT 单独提升到 73.66%；SFT→RFT 进一步升至 74.32% 单通道，74.77% 自我一致性；在 BIRD 单模型排行榜上排名第一（self‑consistency 74.77%）。与当前最高 74.12%（Gemini‑SQL2）和 73.70%（SiriusAI‑Text2SQL）相比，单一 31B 模型已实现领先，且无需多组件管线或 MoE。

**⚠️ 局限性**

局限性包括：①仍依赖对数据库的读‑写权限（工具可用性受限）；②对 BIRD 的高性能主要归功于专门设计的工具和奖励，未验证在其他多域或结构差异更大的数据集上的泛化；③RL 训练需要大量 GPU 计算，SFT 轨迹采样仍受基模型可解范围限制；④在 self‑consistency 下仍有 3.1% 的潜在提升空间，提示选择器或评分模型可进一步改进。

---

## 304. CC4M: Code Clone Analysis and Visualization for Microservices

**arXiv ID:** 2608.28111 | [PDF](https://arxiv.org/pdf/2608.28111v1)

**作者:** Gen Kawamata `[一作]` (Ritsumeikan University), Katsuro Inoue `[通讯]` (Ritsumeikan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一款名为 CC4M 的微服务感知代码克隆分析与可视化工具，用于识别、丰富并展示跨服务的代码克隆及其共修改历史。

**💡 创新点**

创新点包括：① 将服务边界信息、共修改记录和文件类别融入克隆对的元数据；② 定义服务跨度、共修改频率等克隆指标；③ 在散点图中显式绘制服务边界，并提供基于指标的过滤和钻取视图，实现更直观、可操作的克隆可视化。

**🔧 技术方法**

使用技术主要有：CCFinderSW（Type‑2 克隆检测）、CLAIM（基于 Docker Compose 的微服务边界识别）、GitHub Linguist（语言识别）、Git 历史追踪（共修改分析）、JavaScript/HTML/CSS 的交互式可视化框架（散点图与指标视图）。

**📊 数据集**

数据集包括：① Train Ticket（41 个服务、1,423 个文件、10,333 个克隆集），② 6 个其它开源微服务项目（FightPandemics、Lakeside Mutual、OpenTelemetry Examples、FTGO Application、IBM Wave7 Lifeline、Redis Microservices Demo）。

**📈 对比分析**

目前论文中未提供与现有工具的量化对比（因缺乏兼容基准）。通过实验测得，在 Train Ticket 项目上，完整分析（包括克隆检测）耗时约 43 分钟（Intel Core i5‑14400F）。评估主要通过场景演示展示指标筛选和可视化效果，未给出具体性能数值。

**⚠️ 局限性**

局限性：① 仅检测 Type‑2 克隆，无法捕获 Type‑3 或更复杂的相似模式；② 共修改分析仅针对给定快照，无法追溯更早或更远的历史；③ 依赖 CLAIM 的服务识别精度（约 82%），错误会影响跨/内服务划分；④ 仅支持 Docker Compose+Dockerfile 的项目，且仅支持 CCFinderSW 所支持的语言；⑤ 定义的指标为优先级辅助工具，未经过风险评估验证。

---

## 305. A Mixed-Behavior Vote Model for Multimedia Subjective Quality Votes, Means, and Variances

**arXiv ID:** 2608.27724 | [PDF](https://arxiv.org/pdf/2608.27724v1)

**作者:** Jaden Pieper `[一作]` (Institute for Telecommunication Sciences), Stephen D. Voran `[通讯]` (Institute for Telecommunication Sciences)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于单峰投票分布的可接受方差区域（UVR），并在此基础上设计了一个四阶多项式方差模型以及混合投票模型（MBV），用于更精确地描述并拟合主观投票实验中的方差分布。

**💡 创新点**

创新点包括：① 用MVU模型替代极端双峰方差曲线，形成更现实的UVR上限；② 设计满足UVR约束的四阶多项式方差拟合函数；③ 引入混合投票模型，可通过调节BinoVotes、ATC、MVU的混合比例，量化三种不同投票行为在实际数据中的贡献。

**🔧 技术方法**

采用统计方差界定、无偏贝努利/二项投票模型、最小二乘/加权最小二乘拟合、混合分布理论以及实证验证等技术手段。

**📊 数据集**

共使用了16个主观质量评估数据集，涵盖语音、图像、视频三大类别，主要包括ITU‑T P.Sup23、NISQA LiveTalk、NISQA P501 MOS、NISQA Sim MOS、NISQA NSC MOS、VQEG HDTV、P25 2005*、P25 2004*、P25 2003*、VQEG MM6、ITSnoise、ITS2013、ITS1997、ITS1994、TCDVoIP、CCRIQ。

**📈 对比分析**

与传统缩放抛物线(a)模型相比，新模型通过无权/加权最小二乘拟合，使13个数据集的拟合曲线自然落在UVR内，3个需要加权后亦符合约束。拟合误差小，曲线与实验方差高度吻合；混合参数进一步揭示了各实验中投票行为的差异，性能优于传统方法。

**⚠️ 局限性**

局限性包括：① 主要关注无限投票数情况，实际实验中的有限投票数影响尚未完全纳入；② 假设投票分布始终为单峰，无法直接处理多峰情况；③ 对极端方差（a<0.05或a>0.54）的数据仍需人工检验；④ 主要聚焦方差，未深入考虑评分尺度、受试者特征等其他因素。

---

## 306. Image Augmentation as Test Generation for Deep Learning-Based Image Retrieval Systems

**arXiv ID:** 2608.27502 | [PDF](https://arxiv.org/pdf/2608.27502v1)

**作者:** Yehan De Silva `[一作]` (Carleton University), Azalia Shamsaei `[通讯]` (March Networks)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文首先对 2020–2025 年发表的图像增强与生成技术进行了系统综述，整理出 50 种技术并归纳为 10 类；随后在三大数据集上使用 Amazon Titan 与 OpenCLIP 两个嵌入模型，评估这些技术作为深度学习检索系统的测试生成器的效果。

**💡 创新点**

创新点在于（1）首次将 50 种增强/生成技术进行全面梳理并构建十类分类体系；（2）提出了四维度评估框架（检索失败率、嵌入相似度、不确定度、语义真实性），将其应用于对检索模型鲁棒性的定量比较；（3）结合自动化语义真实性评估（LLaVA）实现对生成样本真实性的客观衡量。

**🔧 技术方法**

技术上使用 Amazon Titan 与 OpenCLIP 作为嵌入模型；对 CIFAR‑10、ImageNet‑1K 与 March Networks 三个数据集应用 50 种增强技术；计算检索失败率、余弦相似度、四种嵌入不确定度指标（分布离散度、对偶距离、马氏距离、KNN 集成一致度）；利用 LLaVA 对增强图像进行九项语义真实性打分。

**📊 数据集**

使用的数据集包括 CIFAR‑10、ImageNet‑1K（100 类子集）以及工业合作方 March Networks 的 20k 张图像数据集。

**📈 对比分析**

通过混合效应逻辑回归得到每种增强技术的检索失败 odds ratio，结合嵌入空间移动量、四种不确定度分量以及 LLaVA 的真实性得分，对技术进行排名。实验显示 Weather Simulation、SaSPA 等在保持高语义真实性的前提下能显著提高检索失败率；GAN 生成技术虽能产生多样样本，但真实性与检索准确率均相对较差。

**⚠️ 局限性**

局限性包括：仅评估单一严重程度的增强，未覆盖更细粒度的扰动强度；实验仅包含两种嵌入模型，未验证跨模型的普适性；部分生成技术（如 Pix2Pix）未能在数据集中完整实现；结果受数据集分布与模型特性的影响，需进一步在更广泛的场景中验证。

---

## 307. Generalized Context in Cross Attention for Transfer Learning of Disjoint Tabular Data

**arXiv ID:** 2608.28209 | [PDF](https://arxiv.org/pdf/2608.28209v1)

**作者:** Kazi F. Akhter `[一作]` (Tennessee State University), Manar D. Samad `[通讯]` (North Carolina Agricultural and Technical State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种跨域表格数据迁移学习框架CATTLE，该框架通过通用上下文学习，利用Transformer投影权重实现跨域注意力转移，无需共享特征。

**💡 创新点**

创新点在于：1）不依赖源目标共享特征；2）使用Transformer投影权重而非激活产生通用上下文；3）单一源数据即可学习通用上下文并在目标域微调；4）自监督预训练优于监督预训练。

**🔧 技术方法**

技术方法包括：Gated Feature Tokenized Transformer（gFTT）模型、基于掩码的自监督重构预训练、投影权重跨层迁移、BERT编码特征名与类别值、Optuna进行超参搜索。

**📊 数据集**

使用了14个OpenML公开表格数据集，构成10个源–目标对，覆盖健康、金融、制造、软件测试、工业设计等领域。

**📈 对比分析**

与传统机器学习（XGBoost、LR）、深度学习（ResNet、MLP、FT-Transformer、TabNet）以及现有迁移学习基线（TransTab、XTab、CM2）以及TabPFNv2、TARTE进行对比，评估指标为AUROC和ACC。CATTLE自监督预训练在AUROC和ACC上均平均排名2.9/2.4，显著优于所有基线，Wilcoxon签名秩检验显示大多数数据集上CATTLE获胜。

**⚠️ 局限性**

局限性：需预先选择哪一层投影权重迁移，缺乏理论支撑；对多头注意力层中知识分布的理解有限；对源域特定上下文对目标性能影响的机制尚未完全阐明。

---

## 308. VICT: Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent Reinforcement Learning

**arXiv ID:** 2608.28128 | [PDF](https://arxiv.org/pdf/2608.28128v1)

**作者:** Pengcheng Li `[一作]` (Tsinghua University), Shaohua Ma `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种利用终端验证器的可执行原子进行训练时信用追踪的VICT方法，改进了长时序LLM智能体的奖励分配。

**💡 创新点**

创新地将可验证器内部的事实拆解为原子并通过证明边限定优势修正，实现在训练时仅依赖原始终端奖励，无需额外评估器或标签。

**🔧 技术方法**

使用可执行/证据支撑的原子、依赖闭包、证明图、群组归一化优势分配以及策略梯度优化。

**📊 数据集**

在ALFWorld、WebShop和τ-bench等可编程可验证的长时序任务上进行评估。

**📈 对比分析**

与GRPO、RLOO以及GiGPO、SALT、HCAPO等基线对比，VICT在ALFWorld和WebShop平均成功率提升约16-18个百分点，在WebShop严格成功率提升约17个百分点，整体性能领先或匹配最新细粒度信用方法。

**⚠️ 局限性**

仅适用于可被分解为可执行原子且轨迹可记录证据的终端验证器，依赖低复杂度原子集合，且对大规模或隐式验证器的任务效果有限。

---

## 309. LINE Conversation History Retrieval for Personal Memory RAG: Evaluating Search Representations and Hybrid Retrieval

**arXiv ID:** 2608.27809 | [PDF](https://arxiv.org/pdf/2608.27809v1)

**作者:** Akito Hattori `[一作]` `[通讯]` (Independent Researcher), Akito Hattori (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对一位用户的 LINE 交互记录进行检索评估，构建 22,329 个时间片段并回答 100 个手工标注的问题。

**💡 创新点**

提出 embedding_text 结合摘要与原文摘录的检索表示，并通过线性混合 BM25 与向量检索改进召回。

**🔧 技术方法**

使用 BM25、OpenAI Embeddings（向量检索）、OpenAI Responses（生成摘要）以及线性混合等技术。

**📊 数据集**

使用 358,896 条 LINE 消息（约 22k 片段）以及 100 条评测问题。

**📈 对比分析**

单个检索器中 embedding_text_BM25 Recall@5 0.584，最优混合 embedding_text_BM25+embedding_text_vector β=0.45 Recall@5 0.697，提升约 0.113；MRR@5、nDCG@5 亦随之提升。

**⚠️ 局限性**

局限在单用户单标注、缺少组件消融、候选集限制、未评估答案生成、未考察跨用户或平台、隐私与伦理问题等。

---

## 310. From Small Talk to Rapport: Exploring Robot Self-Disclosure in Collaborative Tasks

**arXiv ID:** 2608.28154 | [PDF](https://arxiv.org/pdf/2608.28154v1)

**作者:** Kaitlynn Taylor Pineda `[一作]` (Johns Hopkins University), Chien-Ming Huang `[通讯]` (Johns Hopkins University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种基于LLM的非拟人化工业机械臂，能够在执行物理任务时进行小型聊天并根据自我披露程度进行对话；通过用户实验验证低披露策略反而能提升人类自我披露与团队合作感。

**💡 创新点**

首次系统性研究了在非拟人化机器人中自我披露对人类自我披露、团队感和对话质量的影响，揭示了披露与机器人外形不匹配导致的逆向效果，并给出了针对不同机器人形态与用户经验的设计原则。

**🔧 技术方法**

采用GPT‑4o驱动的多代理对话管线（Filter、Response Generator、Sentence Rephraser、Disclosure Rewriter），结合Google Cloud Speech‑to‑Text进行语音识别，配合7-DOF Franka Emika Panda机械臂执行任务。

**📊 数据集**

实验使用了50名参与者的自然对话录音与任务视频，未使用公开数据集，而是自建的任务数据与自我披露评分表。

**📈 对比分析**

通过对比低披露(LD)与高披露(HD)两种机器人对话策略，使用累积链接混合效应模型和线性回归评估用户披露、团队感、协调感等主观量表；结果显示LD条件下用户自我披露和团队感显著提升，且高披露对有经验用户产生负面影响。

**⚠️ 局限性**

局限性包括样本主要为大学生/员工，实验环境为仿真质检任务，披露策略依赖预设故事导致真实性不足，且仅测试单一非拟人化机械臂，难以推广至其他机器人平台。

---

## 311. A Compact Selective State-Space Model for Cross-Sectional Stock Return Ranking from Raw Intraday Bars

**arXiv ID:** 2608.28060 | [PDF](https://arxiv.org/pdf/2608.28060v1)

**作者:** Mingju Chen `[一作]` (Baidu Inc), Enze Zhang `[通讯]` (Baidu Inc)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种Staggered‑Timescale Residual Architecture（STA），在未使用任何手工特征的原始五分钟K线和订单簿数据上直接预测下一交易日股票横截面收益排名。

**💡 创新点**

创新点包括零和初始化的多尺度自参差卷积stem、分级衰减的选择性状态空间块以及四路读取器，三者共同实现了对价格尺度与时间尺度的分层处理。

**🔧 技术方法**

技术上使用因果深度可分卷积、可学习的高通前端、选择性状态空间网络（Mamba）、Hillis‑Steele关联扫描，以及风格残差化的评价指标。

**📊 数据集**

数据集为2019‑2024年中国A股中盘市值股（CSI 1000）每五分钟K线和订单簿，共约1456个交易日、近千只股票。

**📈 对比分析**

与六个参数匹配的序列模型（MLP、LSTM、GRU、TCN、Transformer、Mamba）在同一数据、预处理、损失、优化器下对比，STA在风格残差化的排名IC、IC_IR、长短组合Sharpe和压力日IC_IR上均优于所有基线，排名IC提升约14.9%。

**⚠️ 局限性**

局限包括仅在单一市场、单一周期、单一频率下验证；风格控制仅使用价格‑成交量八因子，未考虑估值、行业等；并未实现可交易的收益评估。

---

## 312. Approval-Based Apportionment: Like Portioning, Approximately like Committee Voting

**arXiv ID:** 2608.27605 | [PDF](https://arxiv.org/pdf/2608.27605v1)

**作者:** Paul Gölz `[一作]` (Cornell University), Hannane Yaghoubizade `[通讯]` (Cornell University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了审批基于投票的分配（apportionment）模型，证明了多种比例性公正性公理的等价性，并建立了它们与PAV得分、Lindahl价格以及价格可接受性之间的相互关系；随后将这些结论通过“模型生物”方法提升到部分分配（portioning）和委员会选举（committee elections）的近似关系。

**💡 创新点**

主要创新点包括：① 在分配模型中首次证明EJR、EJR+、FJR三者及PJR、PJR+、FPJR三者等价；② 将Lindahl价格性与bounded PAV improvement等价；③ 证明局部PAV最优委员会可价格化；④ 通过理论构造与价格系统，将上述结果映射到部分分配与委员会选举的近似层次，揭示分配与委员会选举之间的结构性差异。

**🔧 技术方法**

采用组合与图论证明、Hall条件与流网络、KKT条件和极限构造等数学工具，对价格系统进行分析，利用PAV得分的凹性特征和Lindahl equilibrium的KKT条件，构造证明链条；同时通过折叠公理与局部最优性等概念，建立不同模型之间的递推关系。

**📊 数据集**

本文为纯理论研究，无实验或数据集；所有结论均基于形式化证明与公理推理。

**📈 对比分析**

对比方法主要通过构建公理层次图和推导蕴含关系进行；在委员会选举中得到的近似结果如2-近似FJR、FPJR等，说明在实际应用中可获得的比例性保证；未给出数值实验，但理论证明表明所提出的关系在相关模型中是紧确或近似紧确的。

**⚠️ 局限性**

局限性包括：① 结果大多仅在分配模型成立，委员会选举中等价性不再成立，需要近似；② 价格可接受性在委员会选举中无法直接实现，存在可行性与复杂度问题；③ 研究未涵盖完美代表性、层次比例性等更细粒度的公理；④ 对实际投票数据的适用性和计算可行性仍待进一步探索。

---

## 313. Task-State Adaptation with Prototype Memory for Multi-Task Dense Prediction

**arXiv ID:** 2608.28078 | [PDF](https://arxiv.org/pdf/2608.28078v1)

**作者:** Yangyang Xu `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MemMTL，一种多任务稀疏适配框架，利用任务状态和原型记忆实现多任务稠密预测。

**💡 创新点**

创新点在于层级稀疏路由，将任务状态原型化与本地专家结合，兼顾全局任务需求与局部视觉证据。

**🔧 技术方法**

使用视觉基础网络（SAM 3、ViT‑L）、任务状态 Mixture‑of‑Experts、可学习任务状态原型记忆以及稀疏 top‑k 路由。

**📊 数据集**

在 NYUD‑v2 与 PASCAL‑Context 两个数据集上进行评估。

**📈 对比分析**

与多种单任务与多任务基线对比，MemMTL 在保持性能的同时显著降低参数与 FLOPs，平均多任务性能下降仅约 1%，并在部分任务上超过单任务基线。

**⚠️ 局限性**

局限包括对原型数与 k 值的敏感性、在更大数据集上的验证不足，以及对未见任务迁移能力的进一步探索。

---

## 314. Generative AI Expands the Intellectual Reach of Course Based Undergraduate Research Experiences (CUREs)

**arXiv ID:** 2608.27638 | [PDF](https://arxiv.org/pdf/2608.27638v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 315. PhyMamba: Physics-Modulated Mamba for Robust Battery Health Prognostics

**arXiv ID:** 2608.27978 | [PDF](https://arxiv.org/pdf/2608.27978v1)

**作者:** Sara Sameer `[一作]` (Singapore Institute of Technology), Yonggang Wen `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于物理驱动的两阶段 Mamba 框架 PhyMamba，用于多周期电池健康（SoH）预测。

**💡 创新点**

创新点在于第一阶段无侵入式地学习电化学老化参数并映射为物理特征，第二阶段通过将物理信息注入 Mamba 的选择性状态空间动态来调节记忆更新和步长，实现与退化一致的时间演化。

**🔧 技术方法**

使用了轻量化 Mamba 编码器、减阶电化学老化映射、选择性状态空间模型（SSM）与软加权步长调制的深度学习框架，并实现端到端训练。

**📊 数据集**

在三个公开电池退化基准——Matr、NASA 与 Calce——上进行实验，涵盖不同电池类型与充放电协议。

**📈 对比分析**

与多类别基线（LSTM、DLinear、iTransformer、FEDformer、CrossFormer、SimpleTM、TimeMixer、Mamba 等）在 10/20/30 预测周期下进行 MAE/MSE 对比，PhyMamba 在整体平均 MAE 降低 31.8%（从 1.49 降至 1.26），并保持优秀的效率与模型规模。

**⚠️ 局限性**

对极端测量噪声与未知操作条件的迁移性仍有限，需要额外的物理参数表（k_i, F_i 等）与经验先验，且在不同电池化学类型上的通用性尚未充分验证。

---

## 316. Effectiveness of IoT and Deep Learning for Detection and Severity Assessment of Postelectrotermes militaris in Tea Plantations

**arXiv ID:** 2608.27480 | [PDF](https://arxiv.org/pdf/2608.27480v1)

**作者:** D. K. C. Senevirathna `[一作]`, Kalpani Manathunga `[通讯]` (Sri Lanka Institute of Information Technology)

**通讯引用:** 230 | [OpenAlex ID](https://openalex.org/A5028683608)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一套基于IoT音频采集设备和深度学习模型的系统，用于茶园中Upcountry Live Wood Termite（ULWT）的早期检测、严重度评估和空间分布可视化。

**💡 创新点**

首次将现场音频采集、边缘CNN分类、基于概率/振幅/邻近受侵树数量的严重度评分以及GPS定位的空间预测集成在同一平台，实现了茶园ULWT的非侵入性全流程监测。

**🔧 技术方法**

使用Raspberry Pi+高灵敏度麦克风进行现场音频采集；FFT生成64×64频谱图；训练二分类CNN模型；严重度模型结合CNN概率、均值振幅和5 m邻域受侵树数量；GIS可视化与分布估算。

**📊 数据集**

公开的Kaggle数据集：200条50 s录音（100健康+100受侵），经裁剪、重采样后切分为800条10 s片段，按640/80/80划分为训练/验证/测试集。

**📈 对比分析**

与已有声学/视觉虫害检测方法对比，CNN在测试集上取得81.5%准确率、0.819 AUC、80.6%精确度、83.0%召回率、81.8% F1。虽然低于受控实验中的90%+准确率，但在真实田间噪声下已具备可行性。

**⚠️ 局限性**

限制包括：样本独立性有限（同棵树切片不独立）；样本量小；未记录环境参数；模型易产生误判，尤其是假阴性；严重度评分与实际侵蚀强度验证不足；仅使用单次划分评估，需进一步交叉验证。

---

## 317. SpikeOPD: Stable On-Policy Distillation for Autoregressive Spiking Language Models

**arXiv ID:** 2608.27857 | [PDF](https://arxiv.org/pdf/2608.27857v1)

**作者:** Enqiao Lu `[一作]` (Chinese University of Hong Kong), Ivor Tsang `[通讯]` (Agency for Science, Technology and Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对自回归语言生成的脉冲神经网络（SNN）迁移，提出了一种基于教师全KL校正、匹配前缀策略锚定和层级脉冲正则化的稳定在策略蒸馏框架（SpikeOPD），解决了前缀源不匹配和回合反馈崩溃问题。

**💡 创新点**

创新点在于：① 引入教师全KL校正以纠正自生成前缀的输出策略偏差；② 通过冻结参考SNN并在相同自生成前缀上进行策略锚定，限制策略漂移；③ 对选定层级进行脉冲率正则化，稳定内部脉冲动力学；④ 综合上述三项在单一优化目标下实现高效、稳定的在策略迁移。

**🔧 技术方法**

技术上使用了脉冲编码注意力、无softmax的自回归SNN结构、教师-学生全KL损失、匹配前缀策略锚定损失以及层级脉冲正则化；训练过程中采用了离线KD检查点、温度1.0、采样长度32、T_s=4的生成策略。

**📊 数据集**

评估数据集包括八个零样本自然语言推理与常识任务：ARC‑Easy、ARC‑Challenge、WinoGrande、BoolQ、PIQA、HellaSwag、OpenBookQA 与 HeadQA，全部不包含训练样本。

**📈 对比分析**

在三种模型规模（0.125B、0.35B、1.3B）下，SpikeOPD 在零样本任务上平均提升了0.8、1.7和2.9个百分点，同时保持了与KD基准相近的稀疏激活率和计算/能耗水平；相比于继续训练、硬标签SFT和现有在策略蒸馏方法，SpikeOPD 在保持生成稳定性的前提下实现了显著的准确性提升。

**⚠️ 局限性**

主要局限包括：① 仍需依赖高质量的离线KD检查点；② 训练过程相对耗时，需多轮更新；③ 对自生成前缀长度和层级正则化区间的选择较为敏感，需经验调优；④ 在极大规模模型或复杂生成任务上的可扩展性和实时部署性能尚未充分验证。

---

## 318. CareGraph: An Auditable Hybrid AI Framework for Evidence-Grounded Personalized Longitudinal Health Intelligence

**arXiv ID:** 2608.27484 | [PDF](https://arxiv.org/pdf/2608.27484v1)

**作者:** Pratik Ghawate `[一作]`, Tanvi Patil `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了 CareGraph，一种可审计的混合 AI 框架，用以将分散的多源纵向健康数据转换为可追溯、可解释的个性化健康智能摘要；

**💡 创新点**

创新点在于将确定性证据生成、图谱型可追溯、受限生成、以及安全与发布门控分离成模块化体系，实现了透明、可扩展且安全的健康信息组织与解读；

**🔧 技术方法**

使用的技术包括基于 OLS 的趋势判定、规则式缺失上下文检测、患者专属知识图谱与源行级可追溯、受限 schema 的 LLM 合成、规则型安全审查与发布门控；

**📊 数据集**

采用完全合成的 CareGraph‑synthetic‑reviewer‑response‑v1.0 数据集，涵盖 400 名患者的 1,700 条趋势标签，另外 80 名患者的工程批次和 56 名患者的匹配基准；

**📈 对比分析**

通过组件级评估（趋势分类 accuracy 0.827、macro‑F1 0.837；缺失上下文 micro‑F1 0.815 vs 0.318；安全规则 F1 0.974）以及终端级比较（CareGraph 在 56 对比中平均快 9.46 秒、输出更短、趋势匹配率 93.9%，但总 token 低 3,994，图谱引用率仅 5.3%），表明 CareGraph 在聚焦与可追溯性上优于单一 LLM 基线；

**⚠️ 局限性**

局限性包括仅基于合成数据、缺乏真实临床工作流验证、缺少独立专家标注的安全基准、图谱对生成内容的增量价值未单独评估、语义上引用证据的忠实性未完全验证、提示演进带来的偏倚以及缺乏临床相关性与人因评估。

---

## 319. Beyond Task-Only Matching: Personalized Skill Routing with Counterfactual Evaluation

**arXiv ID:** 2608.28241 | [PDF](https://arxiv.org/pdf/2608.28241v1)

**作者:** Tianle Wang `[一作]` (Southeast University), Weiwei Wu `[通讯]` (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了个性化技能路由框架 SkillFeed，能够根据任务和用户配置文件共同决定最合适的技能。

**💡 创新点**

创新点在于构建了对比性配置文件的 Benchmark（SkillFeed‑Bench）以及以配置文件为条件的检索+再排序两阶段流程，并利用技能主体段落检索实现细粒度约束匹配。

**🔧 技术方法**

使用的技术包括任务+配置文件条件的密集检索、BM25、块级检索、聚合评分、列表式学习的再排序器，以及 Qwen3、E5、BGE 等嵌入模型。

**📊 数据集**

数据集为 228,432 个社区公开技能与 329 条带有用户配置文件的查询，包含 162 条对配置文件敏感的样本和 77 条对比性配置文件样本。

**📈 对比分析**

与通用稀疏检索、通用密集检索、SkillRouter 以及大模型预训练检索/再排序进行对比，SkillFeed 在 Hit@1 上从 0.143 提升至 0.751，Profile‑sensitive 查询的 Hit@1 从 0.426 提升至 0.630，展示显著性能提升。

**⚠️ 局限性**

局限性在于对复杂技术/科学领域的用户约束匹配仍有挑战，且目前仅处理单轮配置文件，未来需要支持多轮对话动态约束。

---

## 320. Learning-Augmented Heuristics: Simple, yet Smart, Robust and Interpretable Cache Eviction

**arXiv ID:** 2608.27975 | [PDF](https://arxiv.org/pdf/2608.27975v1)

**作者:** Haocheng Xia `[一作]` (Harvard University), Juncheng Yang `[通讯]` (Harvard University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种学习增强启发式缓存淘汰算法 S4-FIFO，利用离线预训练模型在控制平面上根据缓存级特征自动选择参数，并在 1,035 条生产工作负载轨迹上验证其效率和鲁棒性。

**💡 创新点**

创新点在于将机器学习与传统启发式缓存分离，采用周期性缓存级学习、预训练基础模型、参数化启发式，并通过可解释的高层特征实现低成本、高性能、可解释的缓存决策。

**🔧 技术方法**

使用的技术包括：梯度提升决策树（GBDT）做离线模型训练；特征工程（队列访问分布直方图、工作集特征等）；对 S3-FIFO 进行参数化；异步控制平面与数据平面分离；以及轻量级模型导出（m2cgen）。

**📊 数据集**

使用的数据集为 5,175 条来自 14 源的生产工作负载轨迹（块、键值、CDN 等），其中 4,140 条用于训练，1,035 条用于测试。

**📈 对比分析**

通过与 S3-FIFO、ARC、LRB、3L-Cache、GL-Cache 等多种最先进淘汰算法在相同轨迹上按 miss‑ratio reduction 进行对比。S4‑FIFO 平均比 S3‑FIFO 提升 26%，比 3L‑Cache 提升 8%，鲁棒性最强（最大 miss‑ratio 增长仅 0.8%），且吞吐量与传统启发式持平。

**⚠️ 局限性**

局限性包括：需要对启发式进行参数化且手工特征工程；学习周期窗口较大时可能导致响应延迟；模型仅以 FIFO 为锚点，其他基准的鲁棒性未知；对极短轨迹或非常不常见的工作负载的适应性尚未充分验证。

---

## 321. PAMoR: Parameterized Affective Motion Generation in Real Time for Humanoid Robots

**arXiv ID:** 2608.28213 | [PDF](https://arxiv.org/pdf/2608.28213v1)

**作者:** Yan Pan `[一作]` (University College London), Chengxu Zhou `[通讯]` (University College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个能够实时生成、可在执行过程中随时编辑的全身运动系统，支持通过情绪正负与激活度两维（Valence‑Arousal）参数直接驱动Unitree G1机器人进行指定动作。

**💡 创新点**

其创新点在于：①利用机器人自身关节运动的闭式公式（姿态扩张与运动能量）即时计算Valence‑Arousal标签，消除人工标注；②采用可组合的扩散先验，将动作、情绪正负与激活度三种条件分离，既保证动作识别率，又实现情绪样式的连续可调。

**🔧 技术方法**

核心技术包括：运动变换自编码器（MVAE）用于压缩运动到128维潜在空间；Transformer扩散先验与可组合条件扩散（文本、Valence、Arousal各自的先验）；闭式姿态与能量估计公式；实时自回归生成与全身跟踪执行。

**📊 数据集**

数据来源为：AMASS + BABEL标注的人类动作、由GVHMR从视频重建的动作、以及在Unitree G1上直接记录的遥控动作，经过统一重目标、插值、过滤等预处理，最终构成约10,000条训练链（15类动作）。

**📈 对比分析**

与TextOp、ECHO、SMooDi等文本驱动基线相比，本文模型在FID、R@1、MM‑Dist、Diversity等指标保持或略优，且在用户情绪识别实验中Top‑1/Top‑3分别达到0.384/0.845，明显高于基线，生成速度约78 ms/primitive，能够实现实时执行。

**⚠️ 局限性**

限制与挑战包括：V‑A权重设为等权仍为经验性设定，需进一步校准；生成后机器人执行时缺乏情绪状态反馈，可能导致V‑A漂移；系统目前仅在Unitree G1等全身关节机器人上验证；缺乏基于场景推理动态产生V‑A值的闭环。

---

## 322. Improved Subexponential Upper Bounds for $3$-Restricted Matching Vector Families

**arXiv ID:** 2608.27859 | [PDF](https://arxiv.org/pdf/2608.27859v1)

**作者:** Sidhant Saraogi `[一作]` `[通讯]` (Georgetown University), Sidhant Saraogi (Georgetown University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文针对3-限制匹配向量族（3-restricted MVFs）在模m≤√n的情况下，给出了其大小的子指数上界，证明K≤2^{O(√(nlogn))}；并由此推导出3查询匹配向量码的码长下界；

**💡 创新点**

创新点在于引入“月光花”（moonflower）结构和新的多项式方法，利用低秩矩阵与多项式映射对碰撞类进行精细分析，从而得到比以往2^{O(n/ log n)}更强的上界；

**🔧 技术方法**

主要技术包括：将内积矩阵降到模q^b并提取指示矩阵、构造多项式f实现对碰撞类的区分、使用矩阵Hadamard乘积和秩上界（rank ≤ (n+1)^d）以及递归控制碰撞类大小；

**📊 数据集**

无实验数据集，本工作为纯理论结果；

**📈 对比分析**

与之前Bhowmick、Dvir和Lovett等人的上界2^{O(n/ log n)}相比，显著压缩上界；同时得到的码长下界为N≥exp(Ω((log K)^2 / log log K))，为3查询LDC的性能提供了更严格的限制；

**⚠️ 局限性**

局限性：仅针对3-限制族且模数满足m≤√n；对更高限制度（r>3）或模数更大的情况尚未得到类似的上界；

---

## 323. AIM: Anchor Identity Features, Then Match for Multimodal Large Language Model Unlearning

**arXiv ID:** 2608.28312 | [PDF](https://arxiv.org/pdf/2608.28312v1)

**作者:** Wonjun Lee `[一作]` (Korea Advanced Institute of Science and Technology), Changick Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多模态大语言模型（MLLM）中实现删除时间的身份消除（identity unlearning），即在无保留图像或原始问答数据的情况下，通过视觉侧的干预抑制模型对特定身份的记忆。

**💡 创新点**

创新点在于：① 对身份问题与视觉感知问题在隐藏层表示空间的分离性进行首次系统分析；② 设计两阶段方法 AIM（Anchor Identity Features, then Match），先用统一视觉提示学习“我不知道”目标，再通过 Fisher 约束更新视觉编码器，仅依赖预先计算的 Fisher 统计；③ 通过预缓存 Fisher 解决了无保留数据情况下的保留知识保护。

**🔧 技术方法**

采用的技术包括：视觉提示（visual prompt）学习、对数似然交叉熵、L2 正则、对齐损失、Fisher 信息矩阵约束、梯度预处理、隐状态分析（PCA、t-SNE）等。

**📊 数据集**

使用的数据集为 MLLMU-Bench 和 ReMem，模型为 LLaVA-1.5‑7B 与 Qwen3‑VL‑8B‑Instruct。

**📈 对比分析**

方法与保持保留数据的基线（GA_Diff、KL_Min、MMUnlearner、MANU）以及仅使用遗忘数据的基线（GA、NPO）对比。实验显示 AIM 在忘记率（Cls_f、ROUGE_f）与保留性能（Cls_r、ROUGE_r、Cls_c、ROUGE_c）上与保留数据基线相当甚至更好，同时在视觉感知保持上优于其他方法；在连续删除场景下也保持稳定。

**⚠️ 局限性**

局限性包括：只更新视觉编码器，无法消除文本侧的身份知识；需要在微调时预计算 Fisher 统计；Fisher 约束近似假设在长时间或大步更新下可能失效；学习率受限，导致遗忘深度受限；对推理时图像扰动的鲁棒性有限，未覆盖提示侧或跨模态攻击。

---

## 324. Conditional Visual Evidence Utility: State-Dependent Rank Reversals in Frozen Vision-Language Encoders

**arXiv ID:** 2608.28316 | [PDF](https://arxiv.org/pdf/2608.28316v1)

**作者:** Yunxuan Fang `[一作]` (Beihang University), Xinhe Wang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视觉证据的重要性如何随着其他证据的获取而变化，特别是在控制的组合视觉搜索中，分析了颜色、形状和纹理证据的条件边际效用。

**💡 创新点**

提出了条件边际效用的概念，强调视觉证据的重要性是状态依赖的，并且更新证据排序可以在评估者变化时保留决策相关的价值。

**🔧 技术方法**

使用了冻结的OpenCLIP和SigLIP视觉语言编码器，进行条件效用的测量和分析。

**📊 数据集**

使用了800个场景的持出确认数据集，场景中包含25个对象，分别由颜色、形状和纹理组成。

**📈 对比分析**

通过比较不同的证据获取状态下的排名反转，发现状态依赖的排名反转在设计的候选重叠结构中显著存在，且在不同的证据构造和查询措辞下保持稳定。

**⚠️ 局限性**

研究的局限性在于当前的设置较为狭窄，未能测试属性角色的排列不变性，未来需要扩展到更丰富的证据源和自然图像中。

---

## 325. EvoHarmBench: Breaking Content Moderation with Iterative Human-Like Evasion

**arXiv ID:** 2608.27844 | [PDF](https://arxiv.org/pdf/2608.27844v1)

**作者:** Ruijie Jian `[一作]` (Yuvion Team, Alibaba Group), Haiwen Hong `[通讯]` (Yuvion Team, Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EvoHarmBench，一个基于真实世界对抗性内容的动态评估框架，评估文本内容审核系统在反复改写后仍能检测有害意图的能力。

**💡 创新点**

创新点在于：①构建了包含229个语义子簇的真实对抗性样本库；②设计了迭代式评估循环，模拟用户根据审核反馈不断改写文本；③引入了可读性与意图保持的双重评分机制，衡量攻击成功率；④首次系统性展示即使是领先的LLM审核模型在动态评估下也易被突破。

**🔧 技术方法**

主要技术包括：使用LLM驱动的重写、评估与反思模型；BGE‑M3句子嵌入+DBSCAN进行语义簇划分；自定义的ASR@Readable评估指标；GEPA优化框架用于策略迭代。

**📊 数据集**

使用了5,002条从电商、社交及本地服务平台收集的真实对抗性样本，覆盖广告/流量、赌博/诈骗、辱骂、色情、垃圾/洪水五类；样本经过专家标注后划分为229个语义子簇。

**📈 对比分析**

与传统静态基准对比，EvoHarmBench在12轮迭代后，领先商业LLM的攻击成功率平均达到80.3%，百亿级开源模型更高达91.2%；显示静态评测显著低估了系统脆弱性。

**⚠️ 局限性**

局限性包括：仅针对中文平台，可能难以直接迁移到其他语言或平台；评估只关注对抗性文本，未覆盖正常输入的误报；评估循环中的LLM组件可能带来评估偏差；发布时不包含原始敏感数据，但仍需谨慎使用防止被滥用。

---

## 326. SegBench-GC: Testing Segmentation Invariance in Multi-Step Offline Goal-Conditioned Reinforcement Learning

**arXiv ID:** 2608.27678 | [PDF](https://arxiv.org/pdf/2608.27678v1)

**作者:** Musa Shams `[一作]` `[通讯]` (Independent Researcher), Musa Shams (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SegBench-GC benchmark，用来在保持数据、目标、优化等不变的前提下，通过人为划分备份边界检验离线目标条件强化学习对轨迹切分的稳健性。

**💡 创新点**

创新点在于：① 将轨迹终止与备份切点分离，构建可控的切分实验；② 设计了“Continuation-Valid Targets（CVT）”作为标准，保持非终止切点的继续价值；③ 通过对比实验与诊断揭示行政切分对多步目标产生的乐观误差。

**🔧 技术方法**

使用多步目标构造、期望值分解、经验回放、软策略优化（IQL 风格）以及 n‑step 学习器的实现；诊断层面包含目标值差分检验和价值函数偏差分析。

**📊 数据集**

在 OGBench PointMaze Medium Stitch 和 Puzzle‑4x5 两个任务上进行实验，人工划分约 3.5% 的备份边界，保持相同的轨迹与目标分布。

**📈 对比分析**

通过对比 CVT、naive（把切点当作终止）以及未切分的原始数据，发现 naive 情况下成功率从 50.5% 降至 19.1%，CVT 降低幅度更小；独立的 n‑step 基线也出现类似失效，证明问题普遍存在。

**⚠️ 局限性**

局限性包括：① CVT 需要保证切点后的状态属于同一连续过程；② 仅用 3 个优化种子与 3 个切分实例，样本量有限；③ 结果仅在两类任务与两类学习器验证，未覆盖更广泛算法与数据；④ CVT 也无法完全消除切分敏感性，仍存在残余差距。

---

## 327. Speculative Probing: LLM Monitoring at Speculative-Decoding Cost

**arXiv ID:** 2608.28099 | [PDF](https://arxiv.org/pdf/2608.28099v1)

**作者:** Collin Zhang `[一作]` (Cornell Tech), Vitaly Shmatikov `[通讯]` (Cornell Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种利用大语言模型（LLM）中已有的投机解码头（MTP/Eagle3）来进行高效序列分类的技术，称为 Speculative Probing。该方法在保持投机解码头冻结的前提下，只添加少量软提示和线性分类头，借助 GPU 上已缓存的 KV 进行分类，从而几乎不增加推理成本。

**💡 创新点**

创新点主要包括：①把投机解码头重新定位为监控分类器，充分利用已存在的 KV 缓存；②只需训练几千个参数（软提示+线性头），即可获得与更大模型或专门安全分类器相当甚至更优的性能；③在不同模型规模上均能保持高准确率，形成明显的效率-准确率 Pareto 前沿。

**🔧 技术方法**

技术核心是：冻结基础模型和投机解码头；在输入末尾追加 k 个学习的软提示；递归地通过投机解码头产生隐藏状态；使用单个线性权重向量和 Sigmoid 进行二分类。实验中使用了多种投机解码头架构（Qwen3.5 的 MTP 和 MiniCPM 的 Eagle3）。

**📊 数据集**

使用的数据集包括：VerIH（指令矛盾检测，约 10,000 条对话对）、CoT Repetition 与 CoT Reasoning Strategy（约 10,000 条推理轨迹），以及 Nemotron Safety（11,000 条多语言安全提示）。所有任务均为二分类，训练集约 8k–10k 条样本。

**📈 对比分析**

与传统的 MLP probe、MultiMax probe 以及专门的安全分类器（Qwen3Guard、Llama-Guard）以及 GPT‑5.4‑mini 的零样本对比，Speculative Probing 在大多数任务上处于 Pareto 前沿：在 Qwen3.5‑27B 上，SP‑1、SP‑2、SP‑5 的准确率分别为 91.3%、92.3%、90.7%，均高于 MultiMax（≈93.5%）但成本更低；在安全分类任务中，SP‑1/SP‑2 与 GPT‑5.4‑mini 的准确率相近或更好，同时显著降低推理开销。

**⚠️ 局限性**

限制主要有：①仅能在配备投机解码头的模型上使用，缺乏该模块的 LLM 无法直接采用；②需要针对每个监控任务收集数千条有标签数据，若出现新颖或长尾场景需重新生成数据；③由于实时监控被极大简化，若使用不当可能引发隐私侵犯或过度审查等风险。

---

## 328. Diffusion Distillation for Efficient Weather Ensembles

**arXiv ID:** 2608.27728 | [PDF](https://arxiv.org/pdf/2608.27728v1)

**作者:** Yiming Yang `[一作]` (University College London), Serge Guillas `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种监督式能量距离蒸馏方法，将多步扩散模型压缩为单步学生模型，显著降低神经函数评估次数，提升概率天气预报效率。

**💡 创新点**

创新点在于将教师分布匹配与观测监督相结合，采用能量分数的样本基分布距离，避免GAN或辅助网络，既保持教师分布知识又正则化训练，实现单步推理同时保留极端事件技能。

**🔧 技术方法**

主要技术包括扩散模型（GenCast）、能量分数（Energy Score）用于分布距离估计、监督损失、单步生成器结构、样本基分布匹配与自回归天气集成。

**📊 数据集**

使用WeatherBench 2（全球10天预报数据）和NOAA IBTrACS台风轨迹数据（如台风Nanmadol）进行训练与评估。

**📈 对比分析**

通过与GenCast教师及CM、DMD等蒸馏基线对比，使用RMSE、CRPS、spread–skill ratio等指标，单步学生在RMSE上最优，CRPS与教师相当，分布校准更好；在极端事件台风轨迹预测上实现了最低位置CRPS和最小轨迹误差。

**⚠️ 局限性**

局限在于训练效率仍可提升，对更高分辨率或更长时间尺度的预报适用性待验证，且样本基分布匹配的方差和高维数据的分布估计仍存在挑战。

---

## 329. Select, Don't Train: The Benefits of Modular Entity Disambiguation with LLM-Based Selection

**arXiv ID:** 2608.27470 | [PDF](https://arxiv.org/pdf/2608.27470v1)

**作者:** Fina Polat `[一作]` (University of Amsterdam), Paul Groth `[通讯]` (University of Amsterdam)

**通讯引用:** 28890 | [OpenAlex ID](https://openalex.org/A5034924491)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了可模块化的实体消歧义管道，使用LLM作为选择器，比较不同检索方法

**💡 创新点**

证明检索与选择可分离，轻量化检索+强LLM选择器即可接近甚至超过端到端模型，并提出可拒绝（abstention）评估

**🔧 技术方法**

使用BM25、Wikipedia API、VERBALIZED检索器；LLM选择器为GPT-4o-mini、GPT-5.4-mini、Qwen3-32B等；低资源QLoRA微调

**📊 数据集**

在ZELDA基准套件（包含AIDA、REDDIT、Tweeki、WNED、ShadowLinks等）上进行实验

**📈 对比分析**

与端到端双编码器VERBALIZED对比，模块化方案平均微F1从82.3提升至88.5；BM25+GPT-5.4-mini达90.7（含拒绝）

**⚠️ 局限性**

仅限英语数据，未处理NIL实体，低资源微调的性能仍低于大模型；缺乏多语言与领域扩展

---

## 330. Too Much of the Same: From Algorithmic to Human Bias in Learning to Defer

**arXiv ID:** 2608.28050 | [PDF](https://arxiv.org/pdf/2608.28050v1)

**作者:** Dario Pesenti `[一作]` (University of Trento), Andrea Pugnana `[通讯]` (University of Trento)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估学习延期(LtD)策略在不平衡数据集上的采样偏差及其对人类决策的影响，结合计算实验与用户研究

**💡 创新点**

揭示LtD在训练不平衡时会倾向性退回少数类样本，并导致人类出现“考生效应”导致准确率下降，挑战LtD假设人类性能与退回策略无关

**🔧 技术方法**

采用三种主流LtD方法：Realizable Surrogate (RS)、Selective Prediction (SP)、Compare Confidence (CC)，并使用混合效应回归分析用户表现

**📊 数据集**

使用四个公开数据集：ChestXRay、GalaxyZoo、HateSpeech、ImageNet（10类）

**📈 对比分析**

对比不同LtD方法在不同覆盖率下的退回比例，发现少数类退回率最高；在用户实验中，平衡退回集相比不平衡集提高了约10%–15%的分类准确率

**⚠️ 局限性**

仅在单一领域（天文图像）进行用户实验，未验证多任务/多领域；未深入探讨其他认知偏差；对退回策略的动态调整与人机适配仍待研究

---

## 331. Probabilistic Multi-Robot Gas Source Localization with Uncalibrated Sensors: A Distributed Estimation Approach

**arXiv ID:** 2608.28214 | [PDF](https://arxiv.org/pdf/2608.28214v1)

**作者:** Wanting Jin `[一作]` (École Polytechnique Fédérale de Lausanne), Alcherio Martinoli `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种面向多机器人气体源定位的分布式估计框架，使机器人在不需要传感器标定的前提下，通过 rank‑based 特征实现局部信念更新，并通过 PoE 合并为全局信念，配合信息区域分配与路径规划实现高效探索与利用。

**💡 创新点**

核心创新在于：1）利用对传感器尺度和非线性不变的 rank‑based 特征，消除多机器人间传感器异质性对观测融合的影响；2）将局部信念通过乘积专家（PoE）方式无缝融合，获得更鲁棒的全局估计；3）提出基于活跃单元的区域分配与信息重心路径规划，兼顾探索与利用，减少冗余采样。

**🔧 技术方法**

使用了基于卷积神经网络的气 plume 预测模型（DDPM）、基于经验分布函数（EDF）的 rank‑based 观测特征、乘积专家（PoE）融合、基于贝叶斯更新的 STE 估计、基于信息重心的路径规划与 Voronoi‑like 区域分配。

**📊 数据集**

实验在 Webots 高保真仿真环境中进行，使用三台 Khepera IV 机器人配备 MOX 传感器，仿真涵盖三种室内障碍布局的多地图场景，MOX 传感器模型考虑了非线性响应、噪声与慢响应特性。

**📈 对比分析**

与基线 Measurement‑Aggregation（直接聚合所有机器人的原始测量并进行 STE）进行对比；在校准传感器条件下两者性能相近，而在未校准条件下基线算法定位失效、误差显著增加，而所提 Belief‑Sharing 方法能稳健定位且轨迹长度保持在可接受范围内，显示出更高的定位准确性与效率。

**⚠️ 局限性**

主要局限包括：1）当前实现依赖全互联通信，通信开销随机器人数目呈 O(M²Nₑ)；2）局部信念融合假设观测独立性，可能在高噪声/高度相关情况下影响精度；3）实验仅在仿真环境中验证，尚缺乏真实世界测试；4）rank‑based 特征虽对标定鲁棒，但对极端浓度动态变化的响应有限。

---

## 332. NumBench: Diagnosing Counting Failures in Text-to-Image Models

**arXiv ID:** 2608.28206 | [PDF](https://arxiv.org/pdf/2608.28206v1)

**作者:** Sandeep Wadhwa `[一作]` (Indian Institute of Technology Jodhpur), Prakhar Galriya `[通讯]` (Shiv Nadar University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个包含64万条提示、1,600个类别、1–100计数范围的全新计数基准NumBench，并提出了基于空间碰撞的过程模型和自适应阈值的置信度加权计数精度指标。

**💡 创新点**

① 规模最大、因子完全可控的计数基准（包括组合、布局、外观等四大维度）；② 用过程模型解释计数失败的空间竞争机制；③ 设计了置信度加权计数精度（cw-NPS）以减少单一检测器的偏差。

**🔧 技术方法**

利用OW‑DETR、Grounding DINO和OWL‑ViT三款检测器，进行温度缩放与自适应阈值处理；对生成图像采用空间布局分类、合成外观扰动等技术；对结果进行因子敏感度与人类验证的统计分析。

**📊 数据集**

核心数据集由ImageNet、COCO、Objects365、Caltech‑101/256、Food101、CUB‑200等公开数据集的1,600个类别构成，随后合成出64万条结构化提示。

**📈 对比分析**

对九个系统（五个商用、两个开源、两个专用计数方法）在NumBench上进行评估。结果显示计数越高性能越差，所有系统在计数>50时均低于0.2，专用方法在低至中等计数区间表现优于通用模型，但仍无方法能在高计数下保持强精度。

**⚠️ 局限性**

① 评估依赖检测器，受检测器训练偏差与小目标漏检影响；② 过程模型仅解释碰撞导致的漏计，无法解释过计、类别替换等现象；③ 基准仅涵盖模板化提示，缺乏多样化自然语言表述；④ 高计数区间为压力测试，人类标注与检测器一致性下降，难以给出可靠绝对评估。

---

## 333. FedEHR-Agents: Federated Agentic Optimization for Automated EHR Modeling

**arXiv ID:** 2608.27856 | [PDF](https://arxiv.org/pdf/2608.27856v1)

**作者:** Jun Bai `[一作]` (McGill University), Yue Li `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在多医院分布式环境下，设计了 FedEHR-Agents 框架，让各医院的自治临床智能体通过共享经验（可执行 prompt 与结构化建模证据）进行协同学习，实现对 EHR 数据的自动化预处理和模型开发，而非传统的模型参数聚合。

**💡 创新点**

核心创新在于：①将协作目标从模型参数转移到“临床建模经验”；②提出基于证据的经验聚合机制（EGEA）与全局 meta‑prompt 生成；③结合历史记忆、评估器与 TextGrad 对 prompt 进行迭代优化。

**🔧 技术方法**

使用的大模型为 GPT‑5 mini（亦对 GPT‑4o、GPT‑5 进行实验），并在此基础上实现 DP（数据预处理）与 MD（模型开发）模块、历史记忆、任务评估器、Prompt 细化与经验聚合。

**📊 数据集**

实验数据来自 eICU 协作研究数据库，覆盖 4 个临床预测任务：48 小时死亡率、4 小时急性呼吸衰竭、LOS>7 天、感染症（Sepsis）。

**📈 对比分析**

与本地 AutoML、本地 Agent、FedEHR‑Agents（PromptAvg）及中心化 AutoML/Agent 进行比较。FedEHR‑Agents 在所有任务的 AUPRC 上均优于本地和 PromptAvg 方案，提升幅度约 8%–18%，并接近中心化上限。

**⚠️ 局限性**

局限性包括：共享的经验（prompt 与证据）可能泄露隐私；缺乏正式差分隐私或安全聚合保障；目前仅处理结构化 EHR，未扩展到多模态；缺少长期持续学习与安全性验证。

---

## 334. Temporal Tree of Thought: Reasoning-Guided Visual Cue Search for Long-Video Understanding

**arXiv ID:** 2608.27871 | [PDF](https://arxiv.org/pdf/2608.27871v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 335. Agents for Everyone: A Workshop Framework for Building Agentic AI Capabilities in a Distributed Curation Community

**arXiv ID:** 2608.27675 | [PDF](https://arxiv.org/pdf/2608.27675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 336. LLM-Augmented Causal Discovery: Probabilistic Fusion of Edge Existence and Orientation

**arXiv ID:** 2608.27472 | [PDF](https://arxiv.org/pdf/2608.27472v1)

**作者:** Neville K. Kitson `[一作]` (Queen Mary University of London), Anthony Constantinou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种将数据驱动的贝叶斯网络结构学习（BNSL）与大型语言模型（LLM）推理结果融合的框架，使用概率依赖图（PDG）作为统一的概率表示，并通过加权平均实现融合；

**💡 创新点**

创新点在于将BNSL与LLM的边存在与方向不确定性统一建模为PDG，并证明其能够补偿两者的误差，显著提升因果图的F1分数；

**🔧 技术方法**

使用的技术包括BNSL算法（PC、Tabu、FGES）、LLM调用（Claude、Gemini、GPT）生成边概率，MergeGraphs融合算法，ToDagGreedy从PDG提取DAG，以及多重实验设置与统计检验；

**📊 数据集**

实验采用了来自bnlearn、bnRep和Bayesys的26个离散变量网络，变量数在8到70之间，使用10,000行的合成观测数据；

**📈 对比分析**

通过将融合后的DAG与参考DAG的F1、SHD进行比较，结果显示融合方案在22/26个网络中超过BNSL与LLM任一单独方法，平均F1提升0.056，p<0.001；

**⚠️ 局限性**

局限性包括：仅评估离散小型网络，未考虑测量噪声、缺失或潜在混杂；PDG概率不具备真后验解释；对大型网络与连续变量的适用性未知；

---

## 337. FlyBlind: Cross-Slice Timeliness Attacks on UAV Situational Awareness over 5G

**arXiv ID:** 2608.27604 | [PDF](https://arxiv.org/pdf/2608.27604v1)

**作者:** Wagner Comin Sonaglio `[一作]` (Aeronautics Institute of Technology), Lourenço Alves Pereira Júnior `[通讯]` (Aeronautics Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实验验证了FlyBlind，一种利用授权同租户上行竞争在5G软隔离环境下导致无人机状态更新在地面站变得陈旧的攻击方法；

**💡 创新点**

首次将状态新鲜度（AoI）作为被攻击属性，定义可验证的“false-healthy”谓词，并在授权内部攻击下展示跨切片上行竞争引起的灰色失效；

**🔧 技术方法**

采用OpenAirInterface+ORANSlice+rfSim的5G SA网络仿真环境与ArduPilot SITL飞行仿真，利用Scheduling Request/BSR、PRB分配、QoS策略以及MinRatio UL floor进行实验与对比；

**📊 数据集**

所有实验数据均由ArduPilot SITL生成的圆形轨道飞行、固定比特率视频负载（0/8/12/24 Mbps）和20 Mbps攻击者负载在仿真中产生；

**📈 对比分析**

通过对比基线与攻击情形下的AoI、交付年龄、位置误差、延迟与可用性等指标，攻击将交付年龄从约50 ms提升至约12 s，位置误差从约1 m提升至约45‑50 m，延迟与可用性保持正常；MinRatio UL floor能恢复正常，验证了隔离成本可量化；

**⚠️ 局限性**

实验仅在无信道衰落、固定TDD、单基站仿真环境下进行，未验证真实无线环境或商业网络；未考虑视频自适应码率等实际编码行为；仅探讨软隔离，未覆盖硬隔离或更强对手，实际运营商实现隔离成本仍需进一步评估。

---

## 338. Layered LLM Defenses as an Ensemble: Access Tiers, Inference Cost, and the Measured Failure Correlation Between Defense Layers

**arXiv ID:** 2608.28327 | [PDF](https://arxiv.org/pdf/2608.28327v1)

**作者:** Abrar Alotaibi `[一作]` (King Fahd University of Petroleum and Minerals), Moataz Ahmed `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型（LLM）多层防御（stack）在实际部署中的组合效果，并提出了评估工具；

**💡 创新点**

首次将“攻击者访问层级模型（AATM）”与推理时开销分类相结合，构建理论框架来解释防御层的互相依赖；

**🔧 技术方法**

使用理论推导、统计分析与实验测量（基于 7 层防御堆叠、单个自适应攻击）以及多种评估基准（JailbreakBench、StrongREJECT 等）；

**📊 数据集**

主要使用 Vicuna‑7B‑v1.5 与 Llama‑2‑7B‑Chat 作为目标模型，攻击集来源于 JailbreakBench 与自定义优化后的后缀攻击；

**📈 对比分析**

与单层防御相比，堆叠效果并未实现预期的乘法级降低，失败相关系数 ϕ 在 0.30–0.75 范围内呈正相关，堆叠误拒绝率为单层最高层的近似；

**⚠️ 局限性**

局限性包括：仅测得单个攻击者与单一模型的情景，缺乏跨模型/多攻击者验证；缺乏对动态自适应攻击下的长期评估；并未探究不同模型内部结构导致的依赖差异。

---

## 339. Network Topologies for QKD Networks

**arXiv ID:** 2608.28036 | [PDF](https://arxiv.org/pdf/2608.28036v1)

**作者:** Ori Rottenstreich `[一作]` (Technion), Eliahu Cohen `[通讯]` (Bar-Ilan University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了可信中继型量子密钥分发（QKD）网络的拓扑设计，基于图论提出了可靠性、效率与运营成本三维度的度量，并通过枚举与排序方式对小规模网络（n≤8）进行分析，进一步给出可扩展的大规模网络构造方案。

**💡 创新点**

创新点在于将QKD的物理损耗与密钥生成率映射为边权重，构建了包含运营成本O、可靠性R、连通性C的复合成本函数Φ与Ψ，结合失效、跳数与信任指标；同时提出了Construction I/II等分割与层次化的拓扑构造方法，显著提升了网络的可靠性与效率。

**🔧 技术方法**

使用的技术包括无向图的图论指标（度、直径、关键节点、互异路径）、损耗驱动的边权模型、基于O、R、C的成本函数设计、家族化枚举与优化、以及构造算法（Construction I/II）和对比实验。

**📊 数据集**

使用的数据集主要为公开的实测QKD性能数据（Vienna SECOQC、China metro、Cambridge UK等）以及光纤损耗模型，用于构建SKR与dB损耗的关系；同时引用典型数据中心拓扑（fat‑tree、dragonfly、torus、ring）作为基准进行比较。

**📈 对比分析**

比较方法通过计算O、R、C并得到Φ、Ψ值，对21个n=5连通家族进行排名；与常见数据中心拓扑对比时，Construction I方案在成本函数上表现更优，显示出更高的可靠性与效率。

**⚠️ 局限性**

局限性包括仅针对可信中继型QKD网络，未考虑量子中继；模型假设边权独立、损耗线性，缺乏对多源多目标密钥流的完整优化；大规模网络仍依赖启发式构造而非精确搜索。

---

## 340. Actionable CBFI: Integrating Structural Decomposition and Causal Counterfactual Recourse for Tabular Machine Learning

**arXiv ID:** 2608.27821 | [PDF](https://arxiv.org/pdf/2608.27821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 341. GraftyVul: Synthesising Insecure Programs Through Real-World Vulnerability Grafting

**arXiv ID:** 2608.27928 | [PDF](https://arxiv.org/pdf/2608.27928v1)

**作者:** Omri Ram `[一作]` (University of New South Wales), Hammond Pearce `[通讯]` (University of New South Wales)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

GraftyVul 系统通过 LLM 多代理机制将真实行业漏洞嫁接到公开源码项目中，生成 212 个可执行、可验证的漏洞样本；

**💡 创新点**

创新点在于将行业漏洞真实植入功能完整的环境，提出语言与上下文无关的漏洞语义嵌入方法，并提供可验证 PoC 与自动化重现机制；

**🔧 技术方法**

使用了 Claude Sonnet/Haiku 等大模型多代理架构、定制 LSP+gRPC 工具调用、LLM 摘要转 TF‑IDF / Cohere / Titan 的语义嵌入，以及专门的验证与回归测试脚本；

**📊 数据集**

利用 Nullify 提供的真实 SAST 发现的 23 类 CWE 漏洞，插入 5 种语言（Python、TypeScript、Java、Go、C#）的公开开源项目，并与 13 公开漏洞数据集做对比；

**📈 对比分析**

通过 AUC、Recall@k 等语义相似度评估，GraftyVul 在跨语言克隆、CWE 分类的 AUC 达到 0.989/0.708，Recall@10 达 46.7%，Vendi 多样性得分 102，兼具可重现性与广泛语言/CWE 覆盖；

**⚠️ 局限性**

局限在于依赖特定 SAST 输入和预定义验证 harness，验证模块对 contrived 样本仍易通过；仅在 Anthropic 系列模型上验证，跨模型和跨数据源的泛化未测试；对生产级目标程序的评估有限。

---

## 342. Should I Use This Synthetic Dataset for Training? How to Test with Minimal Real Data

**arXiv ID:** 2608.27996 | [PDF](https://arxiv.org/pdf/2608.27996v1)

**作者:** Zhenyu Tao `[一作]` (Southeast University), Osvaldo Simeone `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在给定真实训练集、候选合成数据集与固定学习算法的情况下，如何用最少的真实测试样本决定合成数据是否能提升真实世界性能，并提出了一种自适应e‑过程符号翻转测试（aeSFT）。

**💡 创新点**

创新点在于将符号翻转检验的Monte Carlo次数和真实测试数据量都做自适应控制，形成双重自适应的e‑过程，从而在任何时刻都能保证Type‑I错误控制且不需预先设定测试集大小。

**🔧 技术方法**

使用了赌博式检验（e‑process）框架、符号翻转随机化检验、分批自适应策略、以及对这些过程的理论证明（Ville不等式、可选停止定理）。

**📊 数据集**

在三类工程系统数据集上验证：1）随机生成的二分类任务；2）基于数字孪生的无线包调度仿真；3）低精度射线追踪生成的射频地图预测。

**📈 对比分析**

与传统均值检验（aMT）、固定样本t检验以及固定样本符号翻转检验（eSFT）对比，aeSFT在α=0.1水平下，在三项任务中显著减少所需真实样本量、提升TPR，同时保持FPR低于目标阈值，且不需要预先设定样本大小。

**⚠️ 局限性**

局限性包括：依赖于损失差分在零点附近对称的假设；对非对称或高维损失差分的适用性有限；仅适用于能够获取独立真实测试样本的场景。

---

## 343. SimpCue: Cue-Based Prompting for Multilingual Text Simplification

**arXiv ID:** 2608.28042 | [PDF](https://arxiv.org/pdf/2608.28042v1)

**作者:** Mehrzad Tareh `[一作]` (Universitat Pompeu Fabra), Stefan Bott `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 Catalan、Spanish 和 Italian 三种语言中，评估了使用黄金提示、预测提示与无提示三种条件下的 Qwen3-8B 模型进行 Easy‑to‑Read 文本简化的效果。

**💡 创新点**

首次系统探讨将专家级语义难度标注（gold cues）或自动预测的标注（predicted cues）直接嵌入提示，以指导多语言 LLM 产生可读性更高的简化文本。

**🔧 技术方法**

采用 Qwen3-8B 大模型进行 prompt‑based 简化，并实现了基于 TF‑IDF + Logistic Regression 的多标签预测器生成预测提示。

**📊 数据集**

使用 iDEM 多语言 E2R 简化语料库（Catalan、Spanish、Italian 共计 47,062 句），其中包含专家标注的简化准则。

**📈 对比分析**

通过 BLEU、SARI、chrF 和 BERTScore 四项指标评估，预测提示在整体上略优于 baseline 和 gold‑cue，但提升幅度有限；gold‑cue 在某些语言/指标上并未显著提升。

**⚠️ 局限性**

限制包括：数据集规模有限且标签分布不均，单模型（Qwen3-8B）实验，自动指标对可读性与符合 E2R 规范的评估不足，且 bootstrap 分析仅为探索性。

---

## 344. Learning to Difference: Adaptive Reversible Differencing (AdaRDiff) for Time Series Forecasting

**arXiv ID:** 2608.28134 | [PDF](https://arxiv.org/pdf/2608.28134v1)

**作者:** Morad Laglil `[一作]` (Univ. Grenoble Alpes), Eric Gaussier `[通讯]` (Univ. Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可学习、可逆差分模块AdaRDiff，用于简化时序并实现高效恢复。

**💡 创新点**

创新点是用可学习的差分权重代替手工设定的差分阶数，支持一次性捕获趋势和多季节性，并通过闭式卷积实现可逆重构。

**🔧 技术方法**

技术包括可学习权重差分、可逆重构（线性递推转卷积）、两阶段训练策略、与RevIN兼容的可插拔模块。

**📊 数据集**

使用了八个公开基准：ETTh1/2/ETTm1/2、Weather、Traffic、Electricity、Solar。

**📈 对比分析**

与11个state‑of‑the‑art LTSF基线（DLinear、SparseTSF、TimeMixer、CycleNet、TQNet、TimeBase、MixLinear、FEDformer、iTransformer、PatchTST、TimesNet）进行对比，在大多数数据集和模型上均取得MSE/MAE提升，AdaRDiff‑Linear往往成为最优或第二名，提升可达25.9%。

**⚠️ 局限性**

局限性包括需为每个通道学习差分权重，窗口P的选择对性能影响；在极端长周期或高维度数据上重构计算仍占比较高，且对非周期性结构的适应性仍待进一步研究。

---

## 345. Refundable Deposits: How to Restore Cooperation in Finitely Repeated Games

**arXiv ID:** 2608.27536 | [PDF](https://arxiv.org/pdf/2608.27536v1)

**作者:** Giulio Salizzoni `[一作]` (Swiss Federal Institute of Technology Lausanne), Galit Ashkenazi-Golan `[通讯]` (London School of Economics and Political Science)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5091197625)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了一种可退还存款机制，利用玩家在有限重复博弈中自愿缴纳的押金，在没有外部转移支付或承诺的情况下恢复和维持合作。

**💡 创新点**

创新点在于：①把存款视为跨期激励的工具，使得有限重复博弈的可行收益集合接近无穷重复博弈中的Nash‑threat folk theorem；②证明该机制在任何有限期数下均能构造子博弈完美均衡；③给出最小化成本的线性规划求解方案，并展示可调节的惩罚持续策略。

**🔧 技术方法**

主要技术：博弈论（有限与无穷重复博弈）、子博弈完美均衡分析、退还规则的构造、线性规划最优设计、对一阶偏差原则的使用。

**📊 数据集**

不使用实测数据；通过四个典型示例（囚徒困境、拥挤资源分配、公共物品游戏、动态公共资源）验证理论。

**📈 对比分析**

比较方法：将有存款机制的有限重复博弈结果与传统无存款、无承诺的有限重复博弈结果以及无穷重复博弈的Nash‑threat folk theorem结果进行对比；结果表明：在相对较短的期数（如2–3期）即可恢复大多数可行的合作收益，且存款的现值成本可降至接近零。

**⚠️ 局限性**

局限性：①假设玩家和中介都有完全信息且行动可观测；②依赖中立中介能够承诺并兑现退款规则；③未考虑存款利息、信息不完全或随机信号的情况；④适用范围为有限玩家、固定阶段游戏或可在状态下满足统一一阶段偏差收益上界的动态游戏。

---

## 346. TagZilla: Automated Owner and Abuse Type Tagging for Indicators of Compromise in Threat Reports

**arXiv ID:** 2608.28124 | [PDF](https://arxiv.org/pdf/2608.28124v1)

**作者:** Gibran Gomez `[一作]` (IMDEA Software Institute), Juan Caballero `[通讯]` (IMDEA Software Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出一个自动化平台，能够在网络威胁情报报告中识别指标（IoC）并为每个指标分配归属实体（如APT组、恶意软件家族）和使用场景（如钓鱼、勒索）等标签。

**💡 创新点**

创新点包括：①将IoC归属标签视为开放世界问题，采用LLM进行无约束实体生成与归一化；②将IoC使用类型视为闭合世界问题，设计层级化税onomies；③结合正则表达式预抽取与LLM后处理的两阶段流程，显著提升精度与效率；④构建并公开了包含多来源实体别名的数据库。

**🔧 技术方法**

主要技术手段包括：大语言模型（LLM）推理（gpt‑oss‑20b / gemma4‑e4b），正则表达式基指标抽取，生成式泛型过滤（识别常见无害指针），闭合世界分类与开放世界分类的分层 prompt，实体数据库检索与相似度匹配。

**📊 数据集**

使用的数据集：①人工标注的 96 篇威胁报告（Ground Truth）；②公开的 Malpedia、AnnoCTR、PRISM 三大报告集，合计 765 篇，包含 20,239 份候选指标。实体数据库来自 MISP 和 Malpedia 的公开群集。

**📈 对比分析**

通过在 Ground Truth 上的评估，归属标签 F1 ≈ 0.94，使用类型标签 F1 ≈ 0.93；在 765 篇真实报告上的实验显示约 77% 的候选指标被标记为 IoC，约 70% 的 IoC 获得归属实体；相较于基线方法（仅使用正则表达式或单一标签），性能提升显著，尤其在多实体、多恶意活动的报告中。

**⚠️ 局限性**

主要局限：①对 LLM 的依赖导致潜在的幻觉与不确定性；②仅处理文本，无法识别图片中的 IoC；③当前仅为 IoC 生成单一使用类型标签，无法覆盖多重用途；④泛型过滤对罕见或新型无害指标识别不足；⑤需要人工干预确认新实体或多重归属，影响完全自动化。

---

## 347. openJiuwen: Beyond Static Harnesses for Long-Horizon Coding Agents

**arXiv ID:** 2608.27969 | [PDF](https://arxiv.org/pdf/2608.27969v1)

**作者:** openJiuwen Team `[一作]` (Huawei Technologies Co Ltd), Zhangchun Zhao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可扩展、可适应的长周期编码代理运行时框架 openJiuwen。

**💡 创新点**

创新点在于将 Structural Composability 与 Runtime Adaptivity 两大维度统一到同一共享执行子系统，并通过 Rail 机制实现能力组合与多代理协调。

**🔧 技术方法**

采用了两层 Inner Loop / Outer Loop 的嵌套执行结构、Rail 机制、Context Management、Goal Mode、LSP 驱动的被动反馈和 Self‑Reflection 等技术。

**📊 数据集**

评测数据集为 SWE‑bench Verified（500 题）与 Terminal‑Bench 2.1（89 题）。

**📈 对比分析**

与官方排行榜上的 Claude Code、Terminus 2 等对比，openJiuwen 在 Terminal‑Bench 2.1 上达 87.19% 正确率、在 SWE‑bench Verified 上 82.6% 通过率，分别比对手高 3.39% 与 3.4%。

**⚠️ 局限性**

局限性包括：Self‑Reflection 与运行时适应未深度耦合、Goal Mode 仅支持任务级目标、Context Management 的自动选择仍需手工调优，以及实验仅覆盖两大基准，没有更广泛的模型或基准验证。

---

## 348. Dimension Bridging for 3D RANS with Neural Network Accelerated Gaussian Functional Regression

**arXiv ID:** 2608.27639 | [PDF](https://arxiv.org/pdf/2608.27639v1)

**作者:** Wesley Lao `[一作]` (University of Texas at Austin), Matt Bement `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用 Gaussian Functional Regression (GFR) 构建 PDE 诱导的非平稳核，对 2D RANS 模型进行校正，以预测 3D RANS 模型的气动系数。

**💡 创新点**

将 GFR 扩展至非线性低维模型和观测算子，结合辅助低维模型构造分离特征化核，并用神经网络逼近 Mercer 核实现在线加速。

**🔧 技术方法**

使用 Gaussian Functional Regression、核方法、对偶（adjoint）敏感性、非平稳内积核、混合 Matérn 站立核、主动采样、自适应采样以及带 Fourier feature 的深度神经网络。

**📊 数据集**

基于 ONERA M6 翼尖翼型的 3D RANS 仿真数据及其对应的 2D RANS 结果，参数空间包含翼尖高度 ξ、马赫数、攻角等，共约 200+ HD 3D 训练样本和对应的 LD 结果。

**📈 对比分析**

与传统 Matérn 混合核和标准 GPR 进行比较；在相同采样量下 PDE 核误差下降 3–10 倍，神经网络加速后核评估时间提升约 10⁶ 倍，显著降低整体计算成本。

**⚠️ 局限性**

对低维模型的准确性要求高，辅助模型需手工构建；神经网络逼近高阶核时可能出现数值不稳定；在高度非线性或不同物理域的应用中泛化性尚未完全验证。

---

## 349. KLOD: Locality-Preserving Knowledge Editing via Non-Target Distribution Preservation

**arXiv ID:** 2608.27839 | [PDF](https://arxiv.org/pdf/2608.27839v1)

**作者:** Hojun Jeong `[一作]` (Gachon University), Sangwoo Kang `[通讯]` (Gachon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为KLOD的目标函数，用于在保持模型原有行为的前提下，对大语言模型进行知识编辑；

**💡 创新点**

创新点在于将目标编辑与非目标分布保持分离，采用基于logit-odds的阈值约束与非目标分布KL正则化，实现了对目标概率的有界提升同时抑制分布漂移；

**🔧 技术方法**

主要技术包括：1）基于logit-odds的一侧铰接目标；2）在目标预测位置对非目标分布做KL正则；3）在前缀位置对完整下一词分布做KL正则；4）局部微调仅更新可编辑层；

**📊 数据集**

使用CounterFact和ZsRE两大知识编辑数据集，对Llama3-8B-Instruct和Qwen2.5-7B-Instruct两个模型进行实验；

**📈 对比分析**

与ROME、AlphaEdit、UltraEdit、FT-L、FT-M、LocFT-BF、OVERTONE等基线进行对比；KLOD在保持高可靠性（Reliability≈99%）的同时，显著提升局部性（Locality提升到≈45/29，远高于LocFT-BF≈2/1），但在一般化（Generalization）上略低；

**⚠️ 局限性**

局限性包括：仍存在编辑强度、一般化与局部性之间的权衡，无法完全消除局部性衰退；对大规模编辑仍有一定计算开销；评估指标主要基于token-level局部性，可能无法完全捕捉行为保留。

---

## 350. ITER: Interaction-Aware Retrieval for Agentic Search

**arXiv ID:** 2608.27912 | [PDF](https://arxiv.org/pdf/2608.27912v1)

**作者:** Haodong Chen `[一作]`, Teerapong Leelanupab `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种面向深度研究代理的交互感知检索模型ITER，旨在根据代理搜索轨迹的历史信息动态优化检索结果。

**💡 创新点**

创新点在于将检索查询表示为由主问题、当前子查询和历史子查询构成的“历史条件化”向量，并利用代理交互（访问记录）构建分层负样本，实现轨迹相对监督。

**🔧 技术方法**

主要技术包括：密集向量检索（Qwen3-Embedding）、对话式代理交互分析、分层负样本采样、加权对比损失和基于行文长度的实例权重。

**📊 数据集**

使用的数据集包括InfoSeek-Eval（300道多跳信息检索题目）和BrowseComp-Plus（830道复杂检索任务），检索语料库分别为Wiki-25-Dump（1120万文档）和100k+专业文档。

**📈 对比分析**

与现有LRAT和AgentIR的对比显示，ITER在InfoSeek-Eval上提升10%任务成功率，在BrowseComp-Plus上提升7–8%，同时在不同代理模型间表现出更好的跨模型迁移性。

**⚠️ 局限性**

局限性包括：对预搜索推理的依赖不易迁移到不同模型，且在加入已访问文档内容作为查询时效果反而下降，表明对代理状态的完整建模仍待进一步研究。

---

## 351. Grounded Checklist Partial Credit for Agent Skill Trajectories

**arXiv ID:** 2608.27487 | [PDF](https://arxiv.org/pdf/2608.27487v1)

**作者:** Suliu Qin `[一作]` (Xi'an Jiaotong-Liverpool University), Xilu Wang `[通讯]` (University of Surrey)

**通讯引用:** 2476 | [OpenAlex ID](https://openalex.org/A5045733425)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于检验器的可追溯性检查表部分计分框架（GCPC），用于对大型语言模型代理的执行轨迹进行细粒度、可审核的评估。

**💡 创新点**

创新点在于将人类可复用的规则与LLM实例化相结合，自动生成任务专属检查表；通过证据驱动的逐项判断、缺失证据时的弃权以及单独的官方结果合成，既保持可扩展性，又保证评估的可信度与可审计性。

**🔧 技术方法**

利用LLM（Claude Sonnet 4.6）进行检查表生成，GPT‑5.4‑mini/Claude Haiku 4.5 进行逐项判断，配合机械化的引用验证门控、判定路径区分（assertion、generated、judge），以及确定性官方结果脚本。

**📊 数据集**

在 SkillsBench v1.1（87 个任务，4,455 条轨迹）上进行主实验，并在 Terminal‑Bench 2.0（86 任务）和 SWE‑bench Verified（100 任务）上验证跨基准的可迁移性。

**📈 对比分析**

与全局评估（holistic judge）、固定通用检查表、TICK 等方法比较，GCPC 在相同信息下的 AUC 达 0.689（vs 0.619），与人工评估的偏好一致性最高；在技能差异对比中揭示了二元评估无法体现的提升/退步；跨基准实验也显示了类似的判定效果。

**⚠️ 局限性**

局限性包括：受任务说明、验证器和执行日志可观测性的限制；缺失证据时评分会显著下降；判定依赖于评判者与官方验证器的准确性；在奖励学习场景下可能易被动优化，需进一步研究鲁棒性。

---

## 352. Rubric-to-Code Credit Assignment for Reinforcement Learning

**arXiv ID:** 2608.27906 | [PDF](https://arxiv.org/pdf/2608.27906v1)

**作者:** Rui Jin `[一作]` (Inclusion AI, Ant Group), Chenyi Zhuang `[通讯]` (Inclusion AI, Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于评价标准的代码区块信用分配（RCCA）框架，利用层次化奖励和评估器文本诊断来实现对交互式 Web 应用生成的细粒度强化学习优化；

**💡 创新点**

创新点在于将用户面向的功能标准拆分为局部代码片段，并将其映射为 token 级权重，从而解决传统 GRPO 在功能多样性下的信用分配模糊问题；

**🔧 技术方法**

采用层次化奖励设计、文本诊断归因、token 权重调节的 RCCA 目标，并基于 GRPO 的 RLHF 进行训练；

**📊 数据集**

使用基于 Rubric 的交互式 Web 应用生成数据集（MiniAppBench 任务与自构造的功能 Rubric）以及外部评测基准 ArtifactsBench；

**📈 对比分析**

与多种开源和闭源基线（Claude‑Opus‑4.5、GPT‑5 等）比较，Ling‑RCCA‑Flash 在 MiniAppBench 的平均通过率提升至 41.25%，并在 ArtifactsBench 首位获得 76.19，显示出显著的性能提升；

**⚠️ 局限性**

局限性包括仅针对单页前端应用，无法覆盖后端服务、持久化、身份验证等复杂场景，对评估器诊断错误的依赖可能导致信用分配不准确。

---

## 353. Transparency Rendering in Computer-Aided Design: Methodologies, Trade-offs, and Challenges

**arXiv ID:** 2608.28310 | [PDF](https://arxiv.org/pdf/2608.28310v1)

**作者:** Grigoris Tsopouridis `[一作]` (University of Ioannina), Ioannis Fudos `[通讯]` (University of Ioannina)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4de8e9d8-757b-475f-9627-18a445e50202` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对计算机辅助设计（CAD）中透明度渲染技术进行了系统综述，重点评估了在工业系统中的实际部署与基础算法；

**💡 创新点**

创新点包括：① 将精确、近似、混合以及神经网络 OIT 方法在 CAD 环境下的设计空间统一整理；② 提出针对 CAD 需求的可视化增强技术（重要性驱动透明度、轮廓突出、Z‑冲突鲁棒排序）；③ 给出基于性能‑质量‑内存的跨平台评估与选择指南；

**🔧 技术方法**

使用的技术包括：A‑buffer、深度剥离、k‑buffer、WBOIT、MBOIT、Hybrid OIT、DFAOIT 等传统与神经 OIT 方法，以及面向 CAD 的重要性映射、轮廓渲染和多键深度排序等增强手段；

**📊 数据集**

实验数据集主要为五个工业 CAD 场景（Engine、Printing Test、Porsche、Apartments、Powerplant），在 NVIDIA RTX 5080 与 Huawei Mate 70 Pro 两个平台上进行测评；

**📈 对比分析**

与 A‑buffer（精确）和 ODT（对象排序）等基准相比，DFAOIT 在保持固定内存开销的同时实现了最优的 FLIP 与 MSE 指标；WBOIT 速度最快但质量略逊；在移动端，A‑buffer 超出显存，Odt 成为可行的低成本选择；性能差异随场景深度复杂度显著；

**⚠️ 局限性**

局限性包括：① 精确 OIT 在高深度场景下内存消耗不可接受；② 近似/混合方法在高不透明度场景下易失真；③ 目前评估指标（MSE、FLIP）无法充分捕捉 CAD 关键边界可辨识度；④ CSG 透明度渲染仍缺乏成熟高效方案；

---

## 354. FUSED: Forensic-Semantic Mixture-of-Experts for AI Inpainting Detection and Localization

**arXiv ID:** 2608.28302 | [PDF](https://arxiv.org/pdf/2608.28302v1)

**作者:** Anton Nuzhdin `[一作]` (University of Amsterdam), Ivona Najdenkoska `[通讯]` (University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种名为FUSED的统一模型，能够同时检测和定位AI生成的图像修补区域。

**💡 创新点**

创新点在于：①使用稀疏门控的 Mixture‑of‑Experts 按 token 级别融合低层取证特征与高层语义特征；②引入分割桥接，将定位监督直接注入融合表示；③实现了跨生成器、跨编辑管线的强泛化。

**🔧 技术方法**

核心技术包括：SparseViT 取证分支、冻结的 CLIP‑ConvNeXt‑XXL 语义分支、稀疏 MoE 融合层、交叉注意力、分割解码器以及多任务损失（分类、分割、边缘、负载平衡、分散性）。

**📊 数据集**

使用的数据集包括：OpenSDID（单源训练、跨五个扩散模型测试）、AutoSplice、CocoGlide（无全局 VAE 伪影的像素空间编辑）、INP‑X（对比全局 VAE 伪影存在与否）、So‑Fake‑Set（多源训练）以及对应的 So‑Fake‑OOD（外部商业生成器）。

**📈 对比分析**

与多种基线（MVSS‑Net、CAT‑Net、PSCC‑Net、ObjectFormer、TruFor、DeCLIP、MaskCLIP 等）对比，FUSED 在 OpenSDID 的跨生成器测试中平均像素 F1 达到 55.3，IoU 44.5，检测 F1 88.0，准确率 89.5；在 AutoSplice/CocoGlide 零样本迁移中像素 F1 分别提升 18.3 与 37.2 点；在 INP‑X 评估中保持最高检测准确率与像素 F1；在 So‑Fake‑OOD 上亦实现领先，尤其在定位任务上提升 14.5 点像素 F1。

**⚠️ 局限性**

局限性包括：对全局 VAE 伪影仍有一定依赖，导致 INP‑X 上准确率下降；对极小尺寸修补区域的定位仍相对困难；目前仅针对扩散模型生成的修补，未验证在其他编辑类型上的适用性。

---

## 355. STEGNav: Spatio-Temporal Event Graph Reasoning for Multimodal Lifelong Object Navigation

**arXiv ID:** 2608.28279 | [PDF](https://arxiv.org/pdf/2608.28279v1)

**作者:** Yang Chen `[一作]` (Nanjing University), Lan-Zhe Guo `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的多模态终身导航框架STEGNav，将传统场景图在空间轴上扩展为查询条件化实例与占用感知前沿图，在时间轴上加入双窗口记忆，构建事件驱动的时空图供VLM高层决策使用。

**💡 创新点**

创新点在于（1）空间轴实现实例级定位与前沿探索收益联合表示，解决实例歧义和前沿孤立；（2）时间轴引入短期轨迹窗口与长期验证事件窗口，支持跨子任务经验重用；（3）整体无训练需求，直接利用预训练VLM与感知模块。

**🔧 技术方法**

核心技术包括：YOLOv8-World+SAM用于感知，CLIP提取视觉特征，ChatGPT-5.4-mini等VLM做实例定位、前沿评估与事件验证，双窗口记忆结构与时空图序列化。

**📊 数据集**

在GOAT‑Bench（多模态终身导航）和HM3D（类别级ObjectNav）两个基准上进行评测，使用标准的SR和SPL指标。

**📈 对比分析**

与现有最强对手相比，STEGNav在GOAT‑Bench Val‑Unseen上SR提升至66.3%（比MSGNav高3.9%），SPL为39.7%；在HM3Dv1、HM3Dv2上SR分别达到64.0%和69.4%，均高于前沿方法；总体证明框架提升了成功率与探索效率。

**⚠️ 局限性**

局限性主要在低层控制与终止决策的改进空间（在HM3D SPL不再最优），以及对VLM推理速度和鲁棒性的依赖，未来可进一步融合低层路径规划与高层事件推理。

---

## 356. An algebraic proof of Colombo's difference-power determinant conjecture

**arXiv ID:** 2608.28274 | [PDF](https://arxiv.org/pdf/2608.28274v1)

**作者:** Kun Li `[一作]`, Zihan Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了在偶数维度且节点互不相同的情况下，所有差值幂矩阵在指数d≥n-1时均为非奇异，完成了Colombo猜想；

**💡 创新点**

首次利用二元形式的实根计数与Waring长度的不兼容性，解决了超临界奇指数的难题；

**🔧 技术方法**

采用二元多项式的apolar对偶、Sylvester判别、Waring表示及距离矩阵正定/负定性等经典工具；

**📊 数据集**

本研究为纯理论证明，不使用任何实验数据集；

**📈 对比分析**

由于是严谨的数学证明，未涉及实验对比，结论在所有满足条件的指数下完全成立；

**⚠️ 局限性**

仅适用于偶数n且节点为实数且两两不同，对奇数维度或复数节点的情况仍未覆盖。

---

## 357. CoCoBench: A Cooperative Coordination Benchmark for Embodied Multi-Agent Task Planning

**arXiv ID:** 2608.28266 | [PDF](https://arxiv.org/pdf/2608.28266v1)

**作者:** Yang Chen `[一作]` (Nanjing University), Lan-Zhe Guo `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CoCoBench，一个针对多智能体协作任务的构造级基准，衡量任务分配、顺序、互斥与交接等四种协作模式的执行质量；

**💡 创新点**

通过构造级评分将团队协作与任务完成分离，提供细粒度诊断，构建了包含897个可验证实例的高层技能接口的可执行家庭场景；

**🔧 技术方法**

使用AI2-THOR仿真环境、预定义的高层技能接口、轨迹记录器以及四种构造特定的评估函数，对多模态大型语言模型进行协作决策评估；

**📊 数据集**

使用CoCoBench自身生成的897个实例，覆盖10类任务、4种房间、3种团队规模（2/3/4人）；

**📈 对比分析**

在117个多模态大型语言模型中，比较任务成功率(SR)与构造评分(CS)，发现模型在不同构造上表现差异显著，中心化状态聚合提升约48%成功率，视觉输入对性能影响有限；

**⚠️ 局限性**

局限性包括：评估仅关注高层技能层面，未覆盖低层执行细节；基准覆盖的协作模式有限，可能不完全代表真实场景；开放权重模型在协议遵从性和预算限制上表现差异较大。

---

## 358. A comprehensive and trustworthy benchmark of AI methods for change detection in Earth observation

**arXiv ID:** 2608.28247 | [PDF](https://arxiv.org/pdf/2608.28247v1)

**作者:** Tadej Tomanič `[一作]` (Bias Variance Labs), Dragi Kocev `[通讯]` (Jožef Stefan Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了统一的开源基准，用于评估十种深度学习模型在十个地球观测变化检测数据集上的预测性能与计算效率。

**💡 创新点**

提出了严格控制的实验协议、统一的数据拆分、可重复的训练与评估流程，并将所有资源按FAIR原则公开，解决了该领域评估碎片化与可重复性不足的问题。

**🔧 技术方法**

采用多种模型架构（CNN、ViT、SSM 等）、ImageNet 预训练、融合焦点+Dice 损失、AiTLAS 框架、MLflow 日志、TensorBoard 可视化及 FAIR 本体论等技术。

**📊 数据集**

使用十个异构变化检测数据集：LEVIR-CD+、CLCD、DSIFN、BANDON、MSBC、MSOSCD、EGY-BCD、OMBRIA、Season‑Varying CDD、SYSU‑CD。

**📈 对比分析**

通过统一的数据拆分、损失、超参数与硬件环境，比较从零开始与 ImageNet 预训练两种初始化方式；结果显示经典 U‑Net/Siamese 模型在准确率与计算成本上往往优于复杂模型，预训练普遍提升 2%–8% 的 mIoU。

**⚠️ 局限性**

未针对每个模型单独调优超参，单次训练可能受随机性影响，部分模型（如 CGNet）表现不稳定；基准不涵盖多类别、序列等更复杂的变化检测任务。

---

## 359. WilLaGS: Latent-Conditional 3D Appearance Fields for Robust Gaussian Splatting In-the-Wild

**arXiv ID:** 2608.28240 | [PDF](https://arxiv.org/pdf/2608.28240v1)

**作者:** Yuhao Bai `[一作]` (Nanjing University), Lijun Chen `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了WilLaGS框架，实现了在野外环境下的3D Gaussian splatting重建与生成式外观建模。

**💡 创新点**

创新点包括使用β‑VAE学习连续的全局外观潜在空间，构建潜在条件的三平面神经外观场以捕捉空间变换的局部照明，并采用教师‑学生自监督感知掩码动态抑制瞬时对象。

**🔧 技术方法**

采用的技术包括3D Gaussian splatting、β‑VAE、三平面（Tri‑Plane）神经场、EMA教师‑学生架构和感知差异掩码。

**📊 数据集**

使用的公开数据集包括Photo Tourism (PT) 和 NeRF‑OSR。

**📈 对比分析**

与现有基线对比，WilLaGS在PSNR、SSIM、LPIPS上均取得领先，并在训练时间和实时渲染帧率上也优于其他方法。

**⚠️ 局限性**

局限性在于对高密度遮挡的处理仍不充分，且缺乏显式的物理解释与语义控制。

---

## 360. Parser States Already Know: Structure-Conditioned KV Persistence for Structured Generation

**arXiv ID:** 2608.28276 | [PDF](https://arxiv.org/pdf/2608.28276v1)

**作者:** Linze Wu `[一作]` (University of Chinese Academy of Sciences), Xinrui Chen `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种利用解析器转换信息控制键值缓存（KV）持久化的框架PASK，专门用于结构化生成任务。

**💡 创新点**

创新点在于将约束解码器产生的解析器状态（如必填字段、枚举、函数参数等）映射到Transformer层组上，并通过离线校准确定每个结构-层桶的保留策略；同时结合任务错误敏感度设定保护阈值，使用注意力输出失真分配剩余KV容量。

**🔧 技术方法**

技术包括：基于解析器状态的结构标签提取、层组划分、任务错误敏感度与失真度量的离线校准、离线生成的查找表和在线轻量级结构化KV持久化决策。

**📊 数据集**

使用BFCL（函数调用基准）数据集的非实时和实时子集，对Qwen3-4B和Qwen3-14B模型进行评估。

**📈 对比分析**

与Full KV、H2O、SnapKV、RocketKV、TriAxialKV、Transactional Attention等压缩基线比较，PASK在约0.33 KV预算下平均提升约17.4个百分点准确率；在GPU峰值内存、吞吐量和TPOT等服务效率指标上也分别实现约47.5%内存节省、2.2×吞吐提升及3.3×TPOT降低。

**⚠️ 局限性**

限制：需要额外的离线校准步骤；目前仅使用三层组和三种离散保留动作，较细粒度的层划分或更多精度级别可能进一步提升效果。

---

## 361. Generative AI Alignment with Hinduism's Theological Plurality and Sacred Representation

**arXiv ID:** 2608.28228 | [PDF](https://arxiv.org/pdf/2608.28228v1)

**作者:** Dipto Das `[一作]` (University of Toronto), Syed Ishtiaque Ahmed `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究孟加拉国印度教徒使用生成式AI（如ChatGPT、文本生成与图像生成模型）来探索宗教知识、信仰与实践的体验与认知。

**💡 创新点**

首次从印度教内部多元传统的角度审视生成式AI在宗教语境中的伦理与解释性问题，揭示AI对神学多样性、文化真实性与神圣权威模拟的影响。

**🔧 技术方法**

使用现有生成式AI工具作为交互实验对象；分析者通过访谈文本进行质性编码。

**📊 数据集**

15名孟加拉国印度教参与者的半结构式访谈记录（共约40分钟/人）。

**📈 对比分析**

采用归纳式主题分析法，无定量性能比较，结论基于访谈内容的质性发现。

**⚠️ 局限性**

样本规模有限、仅聚焦孟加拉国印度教社群、缺乏跨文化或跨宗教对照，结果不具普遍适用性。

---

## 362. Towards Stellarator Geometry Optimisation for Nuclear Fusion

**arXiv ID:** 2608.28224 | [PDF](https://arxiv.org/pdf/2608.28224v1)

**作者:** Tobias Weißberg `[一作]` (University of Bonn), Florian Bernard `[通讯]` (University of Bonn)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过局部梯度优化和二维低维潜在空间嵌入，改进了Stellarator几何优化任务的解，获得最高分

**💡 创新点**

提出用L-BFGS梯度优化替代CMA-ES，并构建基于MDS+均值坐标的非线性二维潜在空间进行可视化与采样，从而发现更优的混合几何

**🔧 技术方法**

使用L‑BFGS、增广拉格朗日、VMEC++模拟、二维MDS嵌入、均值坐标插值以及两点有限差分梯度估算（并行计算）

**📊 数据集**

基准数据集ConStellaration（约158k VMEC++模拟结果）及公开排行榜前5解

**📈 对比分析**

将每个解映射为分数，比较对比前5公共提交及其改进，改进幅度均超过115%，最终在排行榜上以0.9763的得分夺冠

**⚠️ 局限性**

仅为局部优化，依赖初始解；仅覆盖几何任务；梯度通过有限差分计算成本高；潜在空间未直接用于全局优化

---

## 363. Abstract4D: A Large-Scale Dataset and Framework for Understanding the Visual Language of Abstract Art

**arXiv ID:** 2608.28339 | [PDF](https://arxiv.org/pdf/2608.28339v1)

**作者:** Haowei Zhang `[一作]` (Sichuan University), Mao Li `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文发布了规模最大的抽象绘画数据集 Abstract4D（约12万幅图像），并针对该数据集开展了艺术风格分类、跨模态检索和文本到图像生成的基准实验。

**💡 创新点**

创新点在于提出四维感知框架（形状、色彩、纹理、构图），通过人机混合管线生成高质量多维注释，首次对抽象艺术进行连续语义空间可视化与聚类，并构建跨模态评测基准。

**🔧 技术方法**

使用了 VLM（GPT‑4.5）生成候选提示并人工审核，Sentence‑BERT+UMAP/HDBSCAN 进行语义嵌入与可视化；基于 CLIP 的线性分类、对比检索；以及 Flux+LoRA 的文本导向图像生成。

**📊 数据集**

主要数据集为 Abstract4D（120k 纯抽象画），并与 WikiArt、OmniArt 等已有艺术数据集做对比。

**📈 对比分析**

在 artist 级分类中，CLIP 线性探针 Top‑1 73.8%；在跨模态检索中，Fine‑tuned CLIP T→I R@1 89.5%、MedR 1；在文本到图像生成中，Abstract4D‑LoRA 在满足四维描述、构图与色彩一致性方面明显优于零样本 Flux 与 WD‑LoRA。

**⚠️ 局限性**

局限性包括注释仍受人机主观性影响、跨文化多样性不足，模型在国籍等宏观属性识别表现差，且生成模型的多样性与创造性仍有待提升。

---

## 364. BanglaMed-QA: A Question Answering System for Healthcare Support in Bangla

**arXiv ID:** 2608.28329 | [PDF](https://arxiv.org/pdf/2608.28329v1)

**作者:** Rowzatul Zannat `[一作]` (Khulna University of Engineering & Technology), Atia Shahnaz Ipa `[通讯]` (Khulna University of Engineering & Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了 BanglaMed-QA——一个针对孟加拉语医疗领域的问答系统，构建了包含 4,493 条问答对的知识库，并通过词根词典、同义词集和 POS 标记实现了问句预处理与代词消解。

**💡 创新点**

创新点在于：①针对低资源语言构建本土化医疗问答数据集；②引入领域词根词典与同义词集，提升语义理解；③使用 POS 标签进行代词消解；④融合四种相似度度量（余弦、Jaccard、BM25、Levenshtein）并通过软硬投票提升匹配准确率。

**🔧 技术方法**

核心技术包括：SVM 进行问句分类；TF‑IDF 向量化结合余弦相似度；Jaccard、BM25、Levenshtein 相似度；软硬投票机制；POS 标记、同义词替换、词根词典的预处理。

**📊 数据集**

使用自行构建的 4,493 条问答对数据集（9 类、506 病种），并配套 617 词的词根词典及同义词表进行实验。

**📈 对比分析**

在自动评估中，软投票方法 F1 为 0.95，覆盖率 100%；硬投票方法 F1 为 0.94，覆盖率 94%。人工评估平均满意度为 0.9/1，说明系统在实际使用中效果良好。

**⚠️ 局限性**

局限主要是数据量仍有限，缺乏临床真实场景验证；目前仅支持单轮问答，未来需扩展疾病覆盖范围、引入深度学习模型以及更丰富的对话管理。

---

## 365. Multi-tier Flexible Graph Connectivity

**arXiv ID:** 2608.28313 | [PDF](https://arxiv.org/pdf/2608.28313v1)

**作者:** Karthekeyan Chandrasekaran `[一作]`, Krishna Kalathur `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了多层灵活图连通性（k-tier FGC）模型，并给出了三类固定k的近似算法；

**💡 创新点**

将多层失效模型与多目标最小割问题关联，设计了新的LP约束并证明其可行性；

**🔧 技术方法**

使用线性规划、分离Oracle、随机取样、Chernoff界、以及多目标最小割算法；

**📊 数据集**

无实验数据集，全部为理论证明；

**📈 对比分析**

通过与已知的min-cost k-ECSS和(p,q)-FGC的近似结果对比，得到O(k²logn) 的近似比，最坏情况下积分间隙至少2k；

**⚠️ 局限性**

近似因子随k、n呈对数增长，积分间隙较大；当k作为输入时可判定性问题难度不明，模型与多目标最小割的复杂性使进一步改进具有挑战性。

---

## 366. MaCoPlanner: LLM-Assisted Manual-Compiled Task Planning with Proactive Safety Verification for Robotic Industrial Panel Operation

**arXiv ID:** 2608.28300 | [PDF](https://arxiv.org/pdf/2608.28300v1)

**作者:** Guipeng Xin `[一作]` (Huazhong University of Science and Technology), Zhongxu Hu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 MaCoPlanner 框架，通过将工业设备手册编译成类型化的中间表示，结合检索、基于LLM的计划生成、预执行安全验证以及多模态地面化接口，实现了面向工业面板操作的任务规划与安全执行。

**💡 创新点**

创新点在于：①将手册知识编译成可检索的结构化 IR；②基于检索到的证据进行 VLM 条件规划；③在执行前通过 LTL 与 FSM 进行主动安全验证并提供局部反馈以进行修复；④实现了失败闭环的计划拒绝机制。

**🔧 技术方法**

使用技术包括：大型语言模型（如 GPT‑4/5）、多模态感知模型（SAM2、DINO‑X、OCR、FoundationStereo）、密集检索与类型化索引、LTL 公式与安全 FSM、符号状态滚动、基于约束的计划修复、机器人机械臂执行。

**📊 数据集**

数据集涵盖四类工业面板手册（电机驱动控制器、发电机控制系统、激光控制器、发动机控制器），共 350+ 手册单位；检索与规划基准分别包含 400 与 480 个任务；实验还使用了一个无负载的控制面板仿真器作为物理执行环境。

**📈 对比分析**

与 Raw-Manual、Prompt‑Safety、ISR‑LLM、LLM^3、SafePlan 等基线对比，MaCoPlanner 在 Level‑2、Level‑3 任务的成功率分别提升至 84.4% 和 43.2%，并将违规率降至 2.7%；在整体指标上显著优于所有对比方法。

**⚠️ 局限性**

局限包括：检索覆盖不足导致的证据缺失、符号状态漂移、Level‑3 任务仍存在较高失败率、地面化与执行过程中的定位/触碰误差，以及实验仅在无负载仿真环境中验证，未覆盖真实工业负载与安全架构。

---

## 367. LoopArena: Benchmarking Models as Runtime Controllers for Loop Engineering

**arXiv ID:** 2608.28281 | [PDF](https://arxiv.org/pdf/2608.28281v1)

**作者:** Yi Wang `[一作]` (DreamX Team, Alibaba Group), Xiangxiang Chu `[通讯]` (DreamX Team, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并评估 LoopArena 基准，用以测量控制模型在管理代码代理循环中的能力。

**💡 创新点**

提出三种评估级别（单决策、任务切片、完整任务）和固定 Worker 结构，将控制模型作为单独可比较对象。

**🔧 技术方法**

采用循环工程框架、Reporter 生成 Evidence Packet、结构化 Loop Contract、Qwen3.7-Plus Worker、对比无控制与固定目标基线。

**📊 数据集**

基于 SCBench 与 BeyondSWE 的真实编码任务，构建任务切片与完整任务。

**📈 对比分析**

通过 Contract Accuracy、Strict Success Rate 与估算推理成本对控制模型与基线进行对比；Type II 约 64% 成本降低，且与 Type III 排序高度相关（ρ=0.9747），但完整任务成功率最高仅 24.69%。

**⚠️ 局限性**

仅覆盖仓库级别编码任务和单 Worker，缺乏多任务、多模型或其他软件领域的通用性。

---

## 368. Spatial-Semantic Reasoning using Large Language Models for Efficient UAV Search Operations

**arXiv ID:** 2608.28270 | [PDF](https://arxiv.org/pdf/2608.28270v1)

**作者:** Marin Maletic `[一作]`, Stjepan Bogdan `[通讯]` (University of Zagreb)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于大型语言模型（LLM）的UAV实时语义推理框架，能够在未知环境中根据自然语言指令自主搜索指定物体；

**💡 创新点**

创新点包括：将LLM作为轻量化语义推理引擎，实时更新搜索相关性；只使用LLM而非计算量大的VLM，实现零样本、上下文驱动的搜索；将3D OctoMap与多项式样条轨迹结合，支持连续物理运动；在真实UAV上验证零样本搜索的可行性；

**🔧 技术方法**

使用技术包括：LLM（OpenAI o3-mini）推理、YOLO11目标检测、OctoMap 3D占据网格、DBSCAN聚类、A* 3D离散路径规划、7阶多项式样条连续轨迹、概率相关性模块；

**📊 数据集**

实验数据集：Gazebo Garden 仿真 10×10 m 环境（香蕉、电脑鼠标、玩具斑马等物体），以及在真实 8×10 m 室内实验场景下使用 Crazyflie 2.1 搭载单色摄像头、Cartographer SLAM 预构建 OctoMap；

**📈 对比分析**

对比传统的 lawn‑mower 扫描策略；在三种场景中，LLM‑guided 方法实现 100% 成功率，搜索时间平均缩短 24%–36%（标准差约 25%–30%），路径长度平均减少 43%–52%（标准差约 27%–38%），在真实飞行中成功定位电脑鼠标；

**⚠️ 局限性**

限制包括：需要预先构建或实时生成的 3D 地图（受 LiDAR 或 SLAM 限制）；小型无人机受限于传感器与计算能力，LLM 推理初始时间约 10 s；对用户指令细节高度依赖；未在大规模或极其动态环境中验证实时性能；模型性能受 LLM 先验与推理能力限制。

---

## 369. Residual-Guided Randomized Neural Networks

**arXiv ID:** 2608.28267 | [PDF](https://arxiv.org/pdf/2608.28267v1)

**作者:** Mushir Akhtar `[一作]` (Indian Institute of Technology Indore), Mohd. Arshad `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种残差引导的随机神经网络训练框架，采用分阶段贪婪选择并闭式更新输出权重，以在保持随机特征生成的高效性下，逐步构建隐藏层。

**💡 创新点**

创新点在于：①用当前残差对随机候选特征进行评分，确保每一步的加入都能降低岭回归目标；②提供单调下降的理论保证；③框架模型无关，可直接嵌入RVFL、ELM、BLS等多种随机网络。

**🔧 技术方法**

使用技术包括：随机特征生成（随机权重+偏置+非线性激活），残差评分公式 Δ(g)=‖gᵀR‖²/(‖g‖²+λ)，闭式岭回归求解Ω，增量式训练和候选池/块大小控制。

**📊 数据集**

实验数据集：UCI公开数据集总计71个分类任务，其中32个二分类、39个多分类。

**📈 对比分析**

比较方法：与对应的基线模型（RVFL、ELM、BLS）进行准确率、标准差、平均排名对比。结果显示：平均准确率提升约2%，标准差下降，平均排名几乎达到1，说明残差引导模型在多种任务中均优于基线。

**⚠️ 局限性**

局限性：仍需手动设置候选池大小M、增量块大小k；依赖随机采样，可能在极大数据量下计算成本上升；目前仅验证分类任务，未扩展到回归、多标签或半监督场景。

---

## 370. Training-free Suction Grasp Detection for Deformed Aseptic Cartons Using Vision-Language Models and Geometric Surface Scoring

**arXiv ID:** 2608.28246 | [PDF](https://arxiv.org/pdf/2608.28246v1)

**作者:** Marin Maletic `[一作]` (University of Zagreb), Goran Vasiljevic `[通讯]` (University of Zagreb)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一个训练自由的吸盘抓取系统，专门用于变形无菌饮料纸箱的自动分拣。

**💡 创新点**

创新点在于将目标识别与抓取点选择完全解耦，利用开源视觉‑语言模型（VLM）+ SAM2 分割，再通过纯几何评分实现无训练、可即时重定向。

**🔧 技术方法**

使用 Gemini‑Robotics‑ER 进行文本提示检测，SAM2 进行实例分割，结合三种几何评分方法（KNN‑PCA、Sobel 交叉乘积、RANSAC 平面拟合）评估表面平整度和法向对齐。

**📊 数据集**

数据集由 40 只商业无菌饮料纸箱组成，分为未变形、轻微变形、严重变形三级，并在 35 个包含其他塑料包装的混乱场景中评估。

**📈 对比分析**

对三种评分方法在一致性、运行时（Sobel 最快、KNN 次快、RANSAC 最慢）和抓取成功率（单物体 88.2% 以上，混乱场景条件成功 87.0%，整体检索 72.6%）进行比较，Sobel 兼顾速度与准确性。

**⚠️ 局限性**

局限性主要在检测召回率仅 83%；抓取失败后未进行重规划；系统对动态速度与真空压强有一定要求，未覆盖极端高负载或深度遮挡情况。

---

## 371. D-TAIA: Domain-Aware LLM Adaptation for Multi-Task Predictive Process Monitoring

**arXiv ID:** 2608.28236 | [PDF](https://arxiv.org/pdf/2608.28236v1)

**作者:** Sjoerd van Straten `[一作]` (Eindhoven University of Technology), Marwan Hassani `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出D-TAIA框架实现多任务预测过程监控（预测下一个活动与剩余时间），并针对数据稀缺、高熵与分布漂移三重挑战设计训练与推理策略

**💡 创新点**

创新点包括：域感知三元组损失预训练、FAISS检索式剩余时间估计、TAIA注意力推理策略与融合门控，实现仅10M参数LLM的高效迁移与泛化

**🔧 技术方法**

技术主要包含：LoRA低秩微调、域感知三元组损失（DATL）、FAISS最近邻检索、TAIA推理、融合门控与可选的文本序列化

**📊 数据集**

使用四个公开BPI挑战日志（BPI2012、BPI2017、BPI2015_2、BPI2020_DD）进行评估

**📈 对比分析**

与直接微调LLM（FT‑LLM）及多任务RNN（MT‑RNN）对比，D‑TAIA在宏F1和剩余时间MAE上均有持续提升，特别是在短前缀和高熵日志上表现最显著

**⚠️ 局限性**

局限性在于未对分布漂移下的TAIA效能进行实测；融合门β固定，未动态调整；检索依赖预训练域无关嵌入，若嵌入质量不足则检索优势受限

---

## 372. A Probabilistic Interpretation of KV Cache Eviction

**arXiv ID:** 2608.28293 | [PDF](https://arxiv.org/pdf/2608.28293v1)

**作者:** Renato Geh `[一作]` (University of California Los Angeles), Guy Van den Broeck `[通讯]` (University of California Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统地把KV缓存置换问题形式化，证明其为NP‑完全难题，并通过将注意力机制视作期望估计，提出基于自归一化重要性采样的概率化置换与解码时校正方法，完成了理论与实践的完整闭环。

**💡 创新点**

创新点在于①首次给出KV置换的正式定义与硬件难度证明；②构建概率视角，将置换映射为期望估计；③设计可调的自归一化重要性采样估计器，实现解码时的偏差校正；④通过温度缩放等技巧平衡方差与偏差，提升鲁棒性。

**🔧 技术方法**

核心技术包括：概率推理框架、注意力期望化简、采样与自归一化重要性采样、温度缩放、不同 proposal 分布（最小方差、H2O归一化、调和先验等），以及实验中的 KVPress 实现与自定义评测工具。

**📊 数据集**

实验数据集主要为 LongBench（HotpotQA、QASPER、TriviaQA子集）和 RULER（子集130例），用于评估压缩率下的生成质量。

**📈 对比分析**

与现有的 StreamingLLM、SnapKV、TOVA、H2O 等 top‑k 置换方法对比，概率置换+校正在低至中等压缩率时通常取得更低的误差、较高的 win 分数，整体性能与最先进方法相当，且表现更为稳健。

**⚠️ 局限性**

局限性包括：1）理论证明仅在特定设置下给出 NP‑完全性，实际操作仍需近似；2）采样与校正带来额外计算开销；3）偏差-方差权衡需经验调优；4）在极高压缩率（≈1）时所有方法性能急剧下降；5）实验覆盖范围虽广，但对极端任务仍需进一步验证。

---

## 373. LUCID: An Agentic AI Framework on Digital-Twin in the Loop for QoS-Guaranteeing Robotic Control

**arXiv ID:** 2608.28437 | [PDF](https://arxiv.org/pdf/2608.28437v1)

**作者:** Hyeonsu Lyu `[一作]` (Seoul National University), Hyun Jong Yang `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于LLM代理的动态架构，能够在数字孪生循环中将轨迹规划（TP）与无线资源管理（RRM）从固定优化转变为按操作员意图动态重构优化问题模板，实现云机器人系统的自适应QoS保证。

**💡 创新点**

创新点包括：①将TP-RRM视为可配置模板，LLM根据操作员需求动态设置变量、目标与约束；②构建SimBridge将大规模机器人仿真场景转化为可重复射线追踪的无线数字孪生；③开发FastConfigNet作为多模态预判模型，大幅降低对昂贵的射线追踪验证的调用；④在数字孪生循环中实现无线冲突检测与路径重规划的闭环。

**🔧 技术方法**

采用的技术包括：LLM代理（Gemini 3.6 Flash、Qwen2‑VL‑7B‑Instruct）实现意图解析；数字孪生平台（NVIDIA Isaac Sim、Sionna RT）与射线追踪；基于CBS的多机器人路径规划与无线验证；FastConfigNet（DeepSets‑CNN + GraphSAGE）多模态推理；SimBridge的几何简化与材质映射；数据湖用于记录求解历史并训练FastConfigNet。

**📊 数据集**

实验使用的是 NVIDIA PhysicalAI SimReady Warehouse‑01 虚拟仓库场景，包含 756 个资产文件（68 M 顶点→12 M 简化后），在该仿真环境中生成射线追踪信道地图，未使用公开现实数据集。

**📈 对比分析**

与传统分离 TP‑RRM、固定 UE/QoS 基线以及纯几何规划（CBS/ECBS）进行对比。多相位实验表明，LUCID 在保持 100% 目标达成率的同时，QoS 加权任务吞吐量提升 1.9‑2.5 倍，死锁率显著降低；FastConfigNet 将规划时间压缩至几毫秒，显著优于无线 CBS/ECBS。整体表现显示动态规划+LLM 架构在高负载、场景变化和 QoS 调整时具有更好的鲁棒性与效率。

**⚠️ 局限性**

局限性主要包括：LLM 对意图解析的准确率仍受模型规模和提示设计影响，易出现语义逃逸；射线追踪与数字孪生的计算开销依然显著，需要进一步加速；实验仅在仿真环境中验证，缺乏真实机器人部署的评估；对场景材质识别依赖 VLM，可能对未知资产产生误分配。

---

## 374. Between Algorithm (AI) and Intuition (Human): Preserving Designer Agency in AI-Assisted Sensemaking of Qualitative UX Data

**arXiv ID:** 2608.28420 | [PDF](https://arxiv.org/pdf/2608.28420v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 375. Prove2Me: An Open Collaborative Platform for Scaling Math Formalization

**arXiv ID:** 2608.28433 | [PDF](https://arxiv.org/pdf/2608.28433v1)

**作者:** Shuze Chen `[一作]` (Columbia University), Tianyi Peng `[通讯]` (Columbia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Prove2Me 平台，支持 AI 代理进行大规模、协作式的数学形式化工作。

**💡 创新点**

创新点包括：①低门槛入口，让仅有 AI 代理的用户即可提交 Lean 证明；②审核任务机制，将人工审计聚焦核心陈述；③proof‑sketch 分解，将难题拆成可独立完成的子问题；④可搜索可复用的 Formalpedia 库；⑤多代理协同与讨论渠道。

**🔧 技术方法**

技术手段包括 Lean 4、Mathlib、AI 代码生成代理、读回机制、里程碑（milestone）校验、专用 harness、REST API 与 Web UI。

**📊 数据集**

使用的数据集为 Mathlib、CSLib、PhysLib 等官方 Lean 库，并在平台上构建的 Formalpedia；实验案例涵盖 Sensitivity Conjecture、Exact Matrix Completion、Sipser–Gács–Lautemann 论文及教材等。

**📈 对比分析**

通过与集中式代理群（如 30,000 次 API 调用）对比，Prove2Me 在相同规模任务下仅用数个消费级订阅（约 200 美元/月），成本显著降低；在案例研究中，几位代理在 2–3 周内完成 80k–150k 行 Lean 代码，证明了可扩展性和高效性。

**⚠️ 局限性**

局限性包括：①仍需人工审计核心陈述，无法完全消除审计瓶颈；②代理模型的准确性和稳定性仍在提升；③多代理协作机制尚未成熟，需进一步完善；④平台对恶意或低质量提交的防护不完善；⑤自动验证语义准确性的能力有限，仍需人工对齐。

---

## 376. Lossy Event Compression: From Event Stream Distortion to Task Performance

**arXiv ID:** 2608.28429 | [PDF](https://arxiv.org/pdf/2608.28429v1)

**作者:** Zahra Rezaee `[一作]` (University of Lisbon), João Ascenso `[通讯]` (University of Lisbon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了两种事件流压缩方案（基于聚合的JPEG 2000编码和基于点云的G‑PCC编码），并构建了任务驱动的评估框架，将压缩失真与四类下游任务（视频重建、目标检测、光流估计、异步特征跟踪）的性能关联。

**💡 创新点**

核心创新在于：1) 引入分类与异常检测的失真指标（Recall、IoU、Markedness、MCC、Cohen's Kappa），证明其能够在不同压缩架构下高相关预测任务性能；2) 对比传统失真指标，展示其在任务级评价中的显著优势；3) 将这些指标应用于异步事件处理任务，验证其对时间敏感场景的适用性。

**🔧 技术方法**

技术手段包括：事件聚合生成二维灰度计数帧并使用JPEG 2000压缩；点云编码将事件映射为三维 (x,y,t) 体素并使用G‑PCC进行压缩；采用参考相对（RR）评估协议，结合四种任务模型（HyperE2VID、RVT、E‑RAFT、HASTE）进行性能测评；使用多种统计相关分析（SROCC、PLCC）评估失真指标与任务性能的关联。

**📊 数据集**

使用了四个公开事件摄像头数据集：ECD、MVSEC、Gen1 与 ECD 的部分序列；其中 ECD 主要用于视频重建和异步跟踪，MVSEC 用于光流估计，Gen1 用于目标检测。

**📈 对比分析**

通过将每种压缩方案的多个操作点（聚合窗口与码率或量化尺度）与任务性能对应，绘制 rate‑utility 曲线。实验表明：所有五个新指标在每个压缩架构及四项任务中均取得 SROCC 与 PLCC ≥ 0.80，显著优于传统指标；在异步跟踪任务中，虽然整体性能下降更快，但新指标仍保持最高的相关性；JPEG 2000 对稀疏计数帧更有效，优于 JPEG XL/AVIF。

**⚠️ 局限性**

局限性包括：1) 评估仅覆盖四类任务，未涵盖所有事件视觉应用；2) 新指标仍需在更大规模数据集与不同编码器上进一步验证；3) 对极低比特率时的残余误差仍无法完全消除；4) 需要针对不同任务自适应调整时间窗口，未提供统一自动化方案。

---

## 377. Program Learning with Verifiable Rewards: Symbolic Backpropagation for Post-Training LLMs

**arXiv ID:** 2608.28421 | [PDF](https://arxiv.org/pdf/2608.28421v1)

**作者:** Vishvesh Bhat `[一作]` `[通讯]` (CoreThink AI), Vishvesh Bhat (CoreThink AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种后训练方法 PLVR，通过学习可验证的程序来实现推理，而不是更新基础模型的权重。

**💡 创新点**

创新点在于将推理过程外部化为由确定性与小型神经子程序组成的可验证程序，并引入符号反向传播进行信用分配，使得每一步都有可检查的契约判定，提供稠密的奖励信号。

**🔧 技术方法**

使用了基于类型系统的符号反向传播、契约验证、程序搜索（beam search）以及在固定原语上进行多轮搜索和原语微调等技术。

**📊 数据集**

使用的数据集包括 LiveCodeBench v6（代码生成与测试执行）和 τ^2-Bench（多轮对话约束满足），以及约 19,627 条用于原语微调的人工合成示例。

**📈 对比分析**

与 RL 后训练基线（DeepCoder、Nemotron‑Cascade、INTELLECT‑3）、vanilla prompting、agentic harness 和前沿模型进行对比。PLVR 在所有基准上均优于 RL 基线，平均提升约 28 分点，并且在小模型上可超过大型无推理层模型。

**⚠️ 局限性**

局限性：只能用于中间步骤可验证的任务；需要预先定义契约与类型；对主观质量或非结构化任务无效；搜索空间仍巨大，需要 beam 与类型联合限制；未对不同 loss/反向策略的独立影响做进一步分离。

---

## 378. Exploiting Per-Core Leakage: Electromagnetic Side-Channel Monitoring of Multicore Architectures

**arXiv ID:** 2608.28412 | [PDF](https://arxiv.org/pdf/2608.28412v1)

**作者:** Daehyeon Bae `[一作]` (Korea University), Seokhie Hong `[通讯]` (SmartM2M)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究并实现了多核处理器的电磁侧信道泄漏机制，并首次实现了基于单核泄漏的物理侧信道分析。

**💡 创新点**

创新点在于提出了分离每核泄漏的探测方法（DPP、ICSH、AIS），以及核心无关的自编码器建模技术，实现了对多核系统的细粒度监测。

**🔧 技术方法**

使用技术包括多通道RF前端、同步多通道采集、信号对齐与干扰抑制、自动编码器（autoencoder）异常检测。

**📊 数据集**

实验数据来自在Raspberry Pi 4B四核ARM Cortex-A72上执行的MLP推理、AES加密以及MiBench基准任务，构成训练与测试集。

**📈 对比分析**

与传统聚合式EM侧信道方法比较，本文方法在AUC上达到0.995以上，ROC曲线显示对异常检测具有极高的准确率。

**⚠️ 局限性**

限制主要在于对探针位置精度要求高、部署成本高、需要对设备进行开箱探测，且当前仅支持固定频率系统，无法处理DVFS动态电压频率变化。

---

## 379. BEACON: Behavior-Anchored Cross-Source Knowledge Graph Construction for Cyber Threat Intelligence

**arXiv ID:** 2608.28394 | [PDF](https://arxiv.org/pdf/2608.28394v1)

**作者:** Changze Li `[一作]` (Virginia Tech), Peng Gao `[通讯]` (Virginia Tech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建跨来源CTI知识图谱，并完成攻击行为、情境实体和指标的统一抽取与对齐。

**💡 创新点**

创新点在于将攻击行为映射为MITRE ATT&CK技术锚点，将情境实体和IOC附着在锚点上，形成统一空间；采用分阶段 propose-then-verify 方案和分层对齐策略，特别是技术邻域信号解决名称差异问题。

**🔧 技术方法**

使用LLM（GPT‑4o）结合规则式定位、候选技术生成、验证、实体与IOC提取及边缘确认；同时利用字符、嵌入、技术邻域三类相似度指标进行跨源对齐。

**📊 数据集**

构建了两份人工标注数据集：BEACON‑Single（150报，8,395节点/边）和BEACON‑Group（100报，3,487对齐单元）。

**📈 对比分析**

在报告级抽取任务上，BEACON F1 78.7%；在跨源对齐任务上，F1 93.96%/98.31%，分别比所有基线高至少23%/9%；在困难不相同命名情况下提升27%。

**⚠️ 局限性**

局限包括对LLM的依赖、对规则定位的精度限制、对极低频或新出现技术的覆盖不足，以及对稀疏图结构的对齐依赖技术邻域，可能导致误合并或漏合并。

---

## 380. Every Article Deserves a Video: Contextual Video Matching for Digital Publishers

**arXiv ID:** 2608.28359 | [PDF](https://arxiv.org/pdf/2608.28359v1)

**作者:** Arnaud Corone `[一作]` (Dailymotion), Parvati Chauchaix `[通讯]` (Dailymotion)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并部署了 Contextual Video Matching 系统，能够自动将长篇文本文章与 Dailymotion 视频库中的相关视频进行匹配，实现视频嵌入与文本内容的语义对齐。

**💡 创新点**

创新点在于：① 将 LLM 与 HyDE 框架结合生成假想视频元数据，并通过“Grounded Parsing”将新实体和事件以文本形式补全；② 采用多语言 MUSE 嵌入并动态阈值化，使同一系统适配不同语言和主题密度的出版商；③ 通过 LLM-As-A-Judge 进行大规模无标注评估，避免人工标注瓶颈；④ 将生成的假想文档缓存，以实现实时低延迟与成本控制。

**🔧 技术方法**

技术栈包括：Gemini‑2.5‑flash（文本抽取与元数据生成）、Gemini‑3.1‑Pro（评测判定）、多语言 Universal Sentence Encoder (MUSE) 进行嵌入、HyDE+Grounded Parsing 进行语义强化、Qdrant 向量数据库做近邻检索、Redis/自研缓存实现 LLM 结果缓存、动态阈值算法与排序混合评分。

**📊 数据集**

数据集：1,032 篇真实出版商网页（覆盖新闻、体育、娱乐等多主题）；200M 视频已使用 MUSE 编码存入向量库；评测过程中使用 LLM-As-A-Judge 产生评判标签，未采用公开标注数据。

**📈 对比分析**

对比方法：1）离线采用 7 种基线（Random、Most Recent、Raw HTML、Basic HTML Parsing、Basic Summary、HyDE、HyDE+Grounded Parsing）进行点评与 Bradley‑Terry 比赛；2）在线 A/B 测试将 Contextual Video Matching 与 Random 基线对比。性能结果：点评平均分从 1.1（Random）提升至 2.535，Top‑4 匹配率提升 4.4%（从 15.5% 到 19.9%）；A/B 测试中观看时长提升 19%，单视频平均播放时长提升 21%。

**⚠️ 局限性**

局限性：① 仅利用文本信息，未加入视频音频/视觉特征，限制了匹配深度；② MUSE 嵌入模型自 2019 年起冻结，难以捕捉最新实体，虽通过 LLM Grounding 弥补但仍不完美；③ 平均得分仍偏低，说明仍有大量低相关匹配；④ 需要动态阈值和人工覆写，增加运营成本；⑤ LLM 调用成本与延迟在高流量场景下仍是潜在瓶颈；⑥ 缺少可解释性与可学习的排序函数，影响进一步精细化匹配。

---

## 381. No Silver Bullet: Boosting GaussDB Performance on the 30TB TPC-H Workload

**arXiv ID:** 2608.28352 | [PDF](https://arxiv.org/pdf/2608.28352v1)

**作者:** Tim Zeyl `[一作]` (Huawei), Per-Ake Larson `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 GaussDB 中引入管道执行模型、可扩展分区交换、分布式 Bloom 过滤器、近似外键检测等改进，实现了在 30 TB TPC‑H 上的最佳性能。

**💡 创新点**

创新点包括：基于管道的并行执行、两级 Mailbox 设计与 UB/URMA 的 RDMA 通信、分布式 Bloom 过滤器策略、使用 HLL 推断近似外键，以及 SIMD/IMCV 压缩等技术。

**🔧 技术方法**

使用技术包括：管道执行框架、两级 Mailbox+UB/URMA、Bloom 过滤器流式传输、HLL 汇总、SIMD 向量化、IMCV 压缩、SVE/SVE2 指令、分布式成本模型等。

**📊 数据集**

数据集：TPC‑H，规模 SF=30,000（30 TB）

**📈 对比分析**

对比方法：在 32 台 Kunpeng 服务器上跑 TPC‑H 的功率和吞吐量测试，Composite QphH@30TB 为 39,508,107，超过参考系统 40%，每核性能比 Hologres 高 3 倍。

**⚠️ 局限性**

局限性：UB 通信在低 shuffle 量查询中可能导致 CPU 开销增加；分布式 Bloom 过滤器在节点数大时仍有通信开销；结果尚未审核；在某些查询中管道与 Volcano 混合导致优化不完全。

---

## 382. Scalable dynamic community detection on temporal graphs using graph neural networks

**arXiv ID:** 2608.28342 | [PDF](https://arxiv.org/pdf/2608.28342v1)

**作者:** Peijie Zhong `[一作]` (Queen Mary University of London), Richard G. Clegg `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于图神经网络的扩散引导对比学习框架，用于对时间节点（node‑time）进行动态社区检测。

**💡 创新点**

创新点在于：①将社区标注细化到每个节点-时间实例；②使用局部时间扩散亲和矩阵构造正负样本；③引入结构匹配损失与对比学习联合训练，并采用结构感知的小批量采样，兼顾细粒度时间信息与大规模可扩展性。

**🔧 技术方法**

使用的技术包括：时间图注意力网络 (TGAT)、对比学习 (SimCLR)、扩散亲和矩阵、CountSketch 压缩、以及 K‑means 聚类。

**📊 数据集**

使用的数据集：人工合成的连续时间网络（带已知动态社区）以及真实的 OpenAlex 计算机科学合作网络（2016‑2025 年）。

**📈 对比分析**

与静态方法（Louvain、CTDNE+KMeans、TGC、MAGI）、基于窗口的 GenLouvain 以及 LAGO 进行比较，采用 AMI/ARI 评估。实验显示在合成数据上，方法取得最高或接近最高的 AMI/ARI，且在规模扩展时比 LAGO 更快；在 OpenAlex 上通过纵向模数与 WCSS 选取簇数，并展示社区演化轨迹。

**⚠️ 局限性**

局限性包括：①需要手动调节窗口大小、扩散阶数等超参数；②采样过程依赖节点-时间实例，稀疏或极长时间跨度的数据可能需要更多调参；③在真实数据中缺乏 ground‑truth，评估只能依赖无监督指标，缺少客观验证。

---

## 383. Post-Training VLMs for Video Mistake Detection

**arXiv ID:** 2608.28406 | [PDF](https://arxiv.org/pdf/2608.28406v1)

**作者:** Federico Spurio `[一作]` (University of Bonn), Juergen Gall `[通讯]` (University of Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MD-VQA 协议和基准，用于评估视频中步骤级错误检测，并开发了首个针对视频语言模型的后训练方法，利用自定义奖励函数提高误差识别能力。

**💡 创新点**

创新点在于：①设计了既能处理已见步骤又能泛化到未见步骤的 MD-VQA 任务；②首次将 RL 后训练应用于误差检测，配合“相反奖励”提升对细微执行差异的敏感性；③构建了统一的 CC‑VQA 与 EP‑VQA 基准，填补了闭集误差检测方法的空白。

**🔧 技术方法**

采用 Qwen2.5‑VL‑7B 视频语言模型，结合 Group Relative Policy Optimization（GRPO）进行后训练；奖励函数包含格式奖励、准确奖励与“相反奖励”（鼓励对同一步骤的正确/错误视频给出不同答案）。

**📊 数据集**

使用改造后的 CaptainCook4D（CC‑VQA）和 EgoPER（EP‑VQA）两个公开数据集，分别包含多种料理步骤与对应的正确/错误视频片段。

**📈 对比分析**

与零样本、监督微调、SFT+R、GRPO 等基线对比。其方法在 Seen/Unseen 两个分割上均优于所有基线，F1 分别提升至 CC‑VQA 上 79.6/53.8（相较 GRPO 的 74.1/51.3），EP‑VQA 上 70.0/48.0（相较 GRPO 的 66.7/43.0），显著提升误差检测的泛化性能。

**⚠️ 局限性**

局限性包括：①仅基于单个视频片段的描述，无法捕捉跨步骤的上下文依赖；②未见分割仍来自相同域的料理视频，跨域/不同录制条件的泛化尚未验证；③模型仍存在“无误”偏置与生成错误答案的幻觉倾向。

---

## 384. Recovering Software Architecture Intent from Historical Work Items using Generative AI: A Mixed-Methods Industry Case Study

**arXiv ID:** 2608.28403 | [PDF](https://arxiv.org/pdf/2608.28403v1)

**作者:** Dominik Storck `[一作]` (Technical University of Munich), Stefan Wagner `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一个五步半自动化工作流，利用大型语言模型从 Azure DevOps 工作项中恢复 C4 架构图。

**💡 创新点**

将 LLM 与链式提示、结构化 CoT、双向可追溯性结合，实现从碎片化敏捷工作项到系统上下文与容器层的自动化架构恢复，并提供结构稳定性与差异可视化。

**🔧 技术方法**

使用 GPT‑5.1 LLM、链式提示、结构化 JSON 输出、Pydantic 验证、PlantUML 渲染以及 Azure DevOps REST API 提取。

**📊 数据集**

来自两家工业合作伙伴的 Azure DevOps 项目（项目 A 共 89 项、项目 B 共 193 项）及第三个独立项目用于校准。

**📈 对比分析**

通过定性专家访谈与定量稳定性分析进行评估；专家认为图准确度高、对外部使用需人工校正；定量上节点稳定性高，边缘关系变异较大，但整体在多次生成中保持可接受的结构一致性。

**⚠️ 局限性**

缺乏对关系稳定性的保证，误差在链式提示中累积；依赖特定 LLM 与组织工作项质量；仅评估两小项目，难以推广；对语义正确性依赖专家认知；未验证大规模可扩展性。

---

## 385. Adaptive Strategies for GR(1) Games

**arXiv ID:** 2608.28391 | [PDF](https://arxiv.org/pdf/2608.28391v1)

**作者:** S. Krishna `[一作]` (IIT Bombay), Abhilasha Sharma Suman `[通讯]` (IIT Bombay)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对GR(1)游戏，提出一种自适应策略框架，能够在非对抗性环境中动态学习并满足最多的保证。

**💡 创新点**

创新点在于：1）引入非对抗性假设并定义终极优雅鲁棒策略；2）设计了收敛性保证的逼近性活跃监视器；3）先离线生成指数级策略库，再在线以概率混合的方式自适应选取；4）给出了完整的收敛性证明。

**🔧 技术方法**

使用技术包括：GR(1)合成、可观测状态的安全/Büchi监视器、概率混合策略、Markov链分析、Borel–Cantelli 及大数定理等。

**📊 数据集**

实验使用了12个基准：Dining Philosophers、Bus Arbiter、Buffer、Lift、Scheduler、Network、Matching Pennies、示例图等四个参数化族（每个族两种规模）以及单实例的经典案例。

**📈 对比分析**

与Bloem等人的基线方法比较时，Grace在多数实例上运行时间更短，且在Scheduler(3,3)等大规模例子中基线已超时；自适应策略在几百步内收敛到正确的假设集合，证明收敛速度符合理论预期。

**⚠️ 局限性**

主要限制包括：离线阶段的策略库构造仍是指数复杂，导致对极大规格仍受限；监视器参数需要手工调节；在极端假设分布下收敛时间可能较长，且对数值精度敏感。

---

## 386. Separating Words with Automata in the Half-adversarial Case

**arXiv ID:** 2608.28385 | [PDF](https://arxiv.org/pdf/2608.28385v1)

**作者:** Gabriel Bathie `[一作]` `[通讯]` (Université de Bordeaux), Gabriel Bathie (Université de Bordeaux)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在随机-对抗情形下，用确定性有限自动机分离两词的状态数上限，证明随机词与任意词可用O(log^{7/3}n loglog n)状态分离。

**💡 创新点**

创新点在于引入块级压缩与小型确定性转化器，将随机词映射为稀疏词，从而将分离问题转化为可用Chase方法解决的短跑长编码问题。

**🔧 技术方法**

主要技术包括：将输入划分为长度k=O(log n)的块，构造2k‑1状态的确定性转化器；利用Chernoff界和大数定律保证随机词在所有转化器下稀疏；再将稀疏词与任意词映射至短词，最后用已有的O(t^{1/3}log t)状态分离方案。

**📊 数据集**

实验数据集为长度为n的均匀随机二进制词与任意对手词；理论分析不依赖具体数据集，只需满足随机性。

**📈 对比分析**

与传统最坏情况上限O(n^{1/3}log^7n)相比，本方法在半随机情形下实现了多项式对数状态数；相对随机全体词的O(log n)上限，已显著逼近。

**⚠️ 局限性**

局限性包括：仅适用于半随机-对抗场景；对于完全最坏情况仍未突破Ω(log n)下界；且实现需依赖随机性假设，无法直接构造最难分离的对。

---

## 387. Cross-Spectral Dense Correspondence for Multimodal Spectral Medical Imaging

**arXiv ID:** 2608.28341 | [PDF](https://arxiv.org/pdf/2608.28341v1)

**作者:** Eric L. Wisotzky `[一作]` (Fraunhofer Heinrich Hertz Institute HHI), Anna Hilsmann `[通讯]` (Fraunhofer Heinrich Hertz Institute HHI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套跨光谱稠密对应的训练与评估框架，通过传感器无关的单通道投影和光谱响应调制，生成与真实跨光谱条件一致的训练/测试样本，并提供了名为 Synth 的合成基准，解决了真实手术光谱数据缺乏稠密标注的问题。

**💡 创新点**

创新点在于：① 统一的跨光谱数据生成协议，使得任何现有稠密对应网络可直接适配非重叠或强光谱偏移的场景；② 通过视图特定的非线性光谱映射模拟物理光谱差异，提升模型对光谱失配的鲁棒性；③ 设计了一个可控的 Synth 评估集，分离几何位移与光谱变化，为跨光谱方法提供系统化基准。

**🔧 技术方法**

主要技术包括：传感器无关的单通道输入表示、光谱响应调制（非线性映射与通道选择）、迭代稠密匹配网络（RAFT、GMA、SEA‑RAFT、SKFlow、DIP 等）以及标准稠密对应损失和数据增强。

**📊 数据集**

使用的数据集有：标准 RGB 稠密对应基准（FlyingChairs、FlyingThings3D、MPI‑Sintel、HD1K、KITTI、Middlebury、ETH3D、InStereo2K）经光谱调制得到的跨光谱版本；Synth 合成基准；以及三种真实医疗跨光谱采集场景（VIS‑NIR MSI 双目、RGB‑SWIR 非双目、HSI 光场子视图）。

**📈 对比分析**

与原始 RGB 训练的模型进行对比，跨光谱训练后模型在跨光谱测试集上的误差下降 5–20% 左右，且在标准 RGB 数据上保持近似性能；在 Synth 基准中，跨光谱模型在三种位移模式下均显著优于原始模型（EPE 下降 10–80%）。 ablation 结果表明传感器无关投影与光谱映射对鲁棒性贡献相当，二者结合可将误差进一步压至约 1/3。

**⚠️ 局限性**

局限性包括：① 对真实手术光谱数据缺乏精确稠密标注，评估仍以合成或视觉检查为主；② 主要聚焦于二维稠密对应，未扩展至三维深度重建或光谱恢复；③ 模型仍受制于基础网络的容量，某些网络在极端光谱差异下性能下降；④ 需要手工选择光谱映射族，可能不覆盖所有真实传感器特性。

---

## 388. LongPIBench: A Long-Context Benchmark for Prompt Injection

**arXiv ID:** 2608.28411 | [PDF](https://arxiv.org/pdf/2608.28411v1)

**作者:** Yupei Liu `[一作]` (Pennsylvania State University), Jinyuan Jia `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 LongPIBench 长上下文注入攻击基准，涵盖论文同行评审、简历筛选、代码评审、邮件摘要四个真实场景，包含合成与真实数据，系统评估多种攻击与防御方法。

**💡 创新点**

首个专门针对长上下文的注入攻击基准；通过大量实验揭示现有防御在长文本中失效；提供合成与真实双重数据集，展示攻击与防御在真实场景下的真实效果。

**🔧 技术方法**

使用启发式攻击（如 Authority spoof、Combined 等）与优化攻击（GCG 与其通用变体）；评估检测式防御（AttentionTracker、DataSentinel 等）和预防式防御（Instructional Delimiters、MetaSecAlign 8B 等）；采用攻击成功率(ASR)、误报率(FPR)、误检率(FNR)等指标。

**📊 数据集**

合成数据：使用 GPT‑5 生成的长篇论文、简历、代码变更、邮件线程；真实数据：来自实际论文评审、招聘简历、代码库、邮件交流等公开或内部数据集；上下文长度从几千到数万 token。

**📈 对比分析**

对同一 LLM（如 Llama‑3.1‑8B‑Instruct）和攻击方式（Authority spoof 等）下，计算各防御的 ASR、FPR/FNR；实验显示：在长上下文下，启发式/优化攻击几乎总能成功；检测防御大多出现高 FPR 或高 FNR，预防防御 ASR 与无防御相近，几乎失效；与短上下文基准对比，防御效果被严重高估。

**⚠️ 局限性**

局限：仅关注单文本的静态长上下文场景，未覆盖多步代理、工具调用等动态工作流；评估的攻击仅限于启发式与 GCG，未涵盖最新的自适应搜索、强化学习等方法；防御覆盖范围有限，缺乏针对动态注入场景的评估。

---

## 389. AI as Teammate: Rethinking Task Distribution in Medical Training

**arXiv ID:** 2608.28373 | [PDF](https://arxiv.org/pdf/2608.28373v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 390. Cooperative Risk-Aware Exploration in Heterogeneous Multi-Robot Systems Using Algorithmic Altruism

**arXiv ID:** 2608.28409 | [PDF](https://arxiv.org/pdf/2608.28409v1)

**作者:** Brooks A. Butler `[一作]` (Oklahoma State University), Magnus Egerstedt `[通讯]` (University of North Carolina)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于博弈论的自利性风险感知多机器人探索框架，利用价值相关系数对每个机器人收益进行重塑，实现分散决策下的协作路径规划。

**💡 创新点**

创新点在于：①引入以机器人价值为基础的异质相关性权重，形成加权潜在博弈；②证明社会纳什均衡即为联合目标的全局最优点；③在连续环境中将信息、冗余与风险模型光滑化，使得梯度上升可直接应用于有限时间航路优化。

**🔧 技术方法**

使用技术包括：博弈论与潜在游戏理论、Hamilton规则启发的相关性构造、有限时间梯度上升的轨迹规划、投影梯度法、单积分器控制与障碍证书；实验平台为UCI Robotarium。

**📊 数据集**

数据集主要为模拟平面域（40×40网格）与Robotarium的实际机器人运动轨迹；使用了两组高斯危害场和可观测信息模型。

**📈 对比分析**

与自利基线（Γ=I）对比：在均匀与异质价值配置下，保持相似的不确定性降低效果；但最小对偶距离明显提升，冗余探索减少；在高价值或混合价值场景下，价值加权风险显著下降，表明风险更合理地分配给高价值机器人。

**⚠️ 局限性**

局限性包括：仅在小规模团队与简化动力学下验证；危害场固定且已知；相关性权重为静态值，未考虑能量、健康或任务演变；缺乏对通信限制和大规模团队的评估。

---

## 391. GraspHOI: Full-Body 3D Human-Object Reconstruction with Finger-Level Grasps from a Single In-the-Wild Image

**arXiv ID:** 2608.28386 | [PDF](https://arxiv.org/pdf/2608.28386v1)

**作者:** Semin Kim `[一作]` (Yonsei University), Jongyoo Kim `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在单张图像中重建全身3D人-物交互，并在类别无关的对象重建基础上显式优化手指抓握。

**💡 创新点**

提出解耦的重建-优化管线，使用面向接触的手指级抓握优化、遮挡感知接触安放和符号穿透损失，兼具类别无关性与高抓握真实感。

**🔧 技术方法**

采用 SAM 3D Body、SMPL-H、WiLoR、Pixel-Perfect Depth、Hunyuan3D、FoundationPose 等多模态深度与几何估计，结合显式表面对应与可微优化。

**📊 数据集**

在 ARCTIC、ProciGen-GRAB（PGG）、BEHAVE、PICO-db 四个公开基准上进行评测。

**📈 对比分析**

与六个现有基线（SAM 3D Objects、HOI‑TG、HDM、TeHOR、EasyHOI、PICO）对比，GraspHOI 在人-物相对位置、手部精度、接触合理性等指标上均显著提升。

**⚠️ 局限性**

仅限单帧静态交互，仍受上游估计误差影响，需手动调参，且在域迁移或复杂动态场景下可能失效。

---

## 392. PersonaForge: Realistic Multi-Turn User Simulation for Agentic Systems

**arXiv ID:** 2608.28378 | [PDF](https://arxiv.org/pdf/2608.28378v1)

**作者:** Hanglong Lv `[一作]` (Peking University), Fuli Luo `[通讯]` (Xiaomi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 PersonaForge 用户模拟框架，生成真实的多轮用户–代理交互数据，并基于此创建了 PersonaForge-Bench 基准，涵盖 138 个跨 20+ 专业领域的任务，评估代理在信息递增、纠错和工具使用等方面的表现。

**💡 创新点**

创新点包括：① 构建四维人格空间（职业、人格、技术熟练度、知识背景）与 SOUL 行为控制机制，实现可控、逼真的用户表达；② Reverse Deep Construction 技术将真实种子查询反向映射为合成用户档案，保证任务的真实性；③ 采用实时代理执行的交互生成，突破传统离线脚本化的局限；④ 设计了四维分数体系（交互效率、工具适配、任务完成、回答质量），为评估多轮代理行为提供更细粒度的度量。

**🔧 技术方法**

使用的大量技术包括：LLM 驱动的实时用户模拟（作为“用户”角色与部署的代理交互）、SOUL 结构化提示、逆向深度构造、全参数监督微调（SFT）、基于 LLM 的判定器评估、以及多轮数据质量控制管道。

**📊 数据集**

主要数据集：① 16K 真实会话（SWE‑chat、Computer‑Use、内部日志）用于分析和种子查询；② 6.3K 通过 PersonaForge 合成的多轮会话用于训练；③ PersonaForge‑Bench 138 任务作为评估基准；④ 还使用 Claw‑Eval 进行外部任务的迁移测试。

**📈 对比分析**

通过对 Qwen3.5‑27B 与 MiMo‑V2‑Flash 进行基线对比，实验显示：Qwen3.5‑27B 在基线 76.2% 的综合分上提升 4.1%（Task Completion +6.0%，Response Quality +6.8%）；MiMo‑V2‑Flash 从 60.4% 提升至 76.1%，增幅 15.7%，Task Completion 提升 22.0%。与多种公开闭源模型（Claude Opus、Gemini 等）对比，PersonaForge‑训练模型在所有四维度均有显著提升；Ablation 结果表明 SOUL 的连贯记忆与行为规则是关键贡献因素；迁移实验显示在 Claw‑Eval 上也能获得 10% 以上的提升，验证其通用性。

**⚠️ 局限性**

局限性：① 人格维度采用 MBTI 仅作可控性手段，未能真实刻画个体差异；② 虽使用真实会话作为种子，但合成过程仍受 LLM 生成偏差影响；③ 训练数据规模（6.3K 记录）相对有限，可能限制更大模型的学习效果；④ 基准任务为构造性情景，难以完全覆盖所有真实业务场景；⑤ 伦理上仅使用已获同意的匿名日志，且在医疗法律任务中使用虚构情景，需进一步评估对敏感领域的适用性。

---

## 393. Optimal Adversarial Testing: Extracting Honest Test Results from Dishonest Test Takers

**arXiv ID:** 2608.28362 | [PDF](https://arxiv.org/pdf/2608.28362v1)

**作者:** Owen Cox `[一作]` (University of Iowa), Weiyu Xu `[通讯]` (University of Iowa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种在存在 AI 辅助作弊的情况下，通过分层安全测试重新检验学生，最终准确选出前 k 名的方法。

**💡 创新点**

首次将对抗性测试建模为二人零和游戏，并用动态规划求解最优重测策略，显著降低总检测成本。

**🔧 技术方法**

采用动态规划、采样理论以及对抗博弈分析等技术。

**📊 数据集**

主要以模拟案例为数据集：如从 300,000 名学生中挑选 60 名，和 50,000 名学生中挑选 100 名等。

**📈 对比分析**

与一次性使用最高安全级别的“盲目”策略对比，所提方法在成本上分别降低到约 0.3% 与 8.4%，且保持完美选拔准确率。

**⚠️ 局限性**

仅考虑离散、确定性的作弊预算与 g(i) 值，未覆盖连续分布、随机表现或全局预算约束等更复杂场景。

---

## 394. CultureConverse: A Multilingual Multi-turn Simulation Harness for Culturally Grounded Assistance in East and Southeast Asia

**arXiv ID:** 2608.28405 | [PDF](https://arxiv.org/pdf/2608.28405v1)

**作者:** Bryan Chen Zhengyu Tan `[一作]` (Singapore University of Technology and Design), Roy Ka-Wei Lee `[通讯]` (University of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了可扩展的多语言模拟与评估平台 CultureConverse，生成 14,610 条评估剧本和 274,295 条 oracle 引导对话，用于衡量 LLM 在多轮文化背景下的协助质量。

**💡 创新点**

创新点包括：①将文化评估从单轮 MCQ 转为多轮隐式约束推断；②构建 58 个子群体身份与 10 个东南亚地区的细粒度标签；③采用 LLM 作为模拟器、判别者和被评估助手，形成完整的评估管线；④通过高质量样本微调实现跨域提升。

**🔧 技术方法**

技术方法包括：LLM‑as‑simulator（GPT‑5 mini）和 LLM‑as‑judge；知识库检索（来自 180k+ 区域标签 Wikipedia 片段和 962 条禁忌条目）；分阶段生成（seed → blueprint → shard → simulation → evaluation）；LoRA 微调；多语言支持与 3H/TDQ 评估指标。

**📊 数据集**

使用数据集：CultureConverse 自生成数据（14,610 评估剧本、274,295 oracle 对话）；公开 Wikipedia 文化段落与 NormAd 禁忌库；评估时还对 7 个文化 MCQ 以及 10 个安全分类基准进行迁移测试。

**📈 对比分析**

比较方法：在 14,610 轮对话上对 18 个前沿 LLM 进行 3H（Help, Honesty, Harmlessness）与 TDQ（Naturalness, Plausibility, Typicality）评分；最佳模型 4.28 3H、91% 清洁率；微调 27,860 高质量样本后，3H 提升约 0.13，MCQ 准确率提升 0.88%，安全分类 macro‑F1 提升 2.5%。

**⚠️ 局限性**

局限性：数据不完全代表真实文化多样性，可能出现文化本质化和偏见；LLM 作为判别者与生成器可能产生循环偏差；低资源语言和少数族群表现不佳；依赖前沿 LLM，资源成本高；评估主要基于自动判别，仍需人工专家复核。

---

## 395. Denoising-Aware Temporal Point Cloud Completion for 3D Crop Architecture Recovery and Phenotypic Trait Extraction

**arXiv ID:** 2608.28343 | [PDF](https://arxiv.org/pdf/2608.28343v1)

**作者:** Mrudul Mittal `[一作]`, Soumyashree Kar `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套两阶段的去噪+时序点云补全管线，用于3D作物结构重建和表型特征提取。

**💡 创新点**

创新点包括：① SynthCrop4D 合成 4D 植物点云基准；② Mamba‑DG 全局状态空间去噪模型；③ Adaptive Temporal PoinTr 跨时间注意力补全网络；④ Delta‑conditioning 与 trait‑aware 微调。

**🔧 技术方法**

技术栈涵盖点云去噪（PointNet, GCN, PDN, Mamba‑DG）、点云补全（PCN, PoinTr, Adaptive Temporal PoinTr）、Transformer 注意力、Hilbert 曲线序列化、Mamba State‑Space 模型、Langevin 采样、Poisson 重建及特征提取。

**📊 数据集**

使用真实 Pheno4D（番茄、玉米多阶段激光扫描）和合成 SynthCrop4D 两个数据集。

**📈 对比分析**

通过 Chamfer 距离、F‑Score@5mm、MAE 等指标对 12 种组合进行 ablation；最佳组合在 SynthCrop4D 上 CD 0.0061、Pheno4D 上 CD 0.0066、F‑Score 0.208；去噪显著提升，Delta‑conditioning 将表型误差降低 70%+。

**⚠️ 局限性**

局限性：Mamba‑DG 的 Langevin 采样在大点云上复杂度高；Poisson 重建无法得到封闭网格，导致面积/LAI 估计不精确；数据集规模有限，模型在更大规模或不同作物的泛化性待验证。

---

## 396. Linear Temporal Logic Translation via Human-Inspired Self-Constrained Reasoning for Robot Task Specification

**arXiv ID:** 2608.28435 | [PDF](https://arxiv.org/pdf/2608.28435v1)

**作者:** Haofei Hou `[一作]` (Peking University), Qining Wang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将自然语言指令转换为线性时序逻辑（LTL）规范的框架，称为Self‑Constrained Reasoning（scr）；

**💡 创新点**

创新点在于将结构约束内化为决策过程，结合层次化强化学习，从而在保持正式约束的同时提升推理灵活性与泛化能力；

**🔧 技术方法**

采用了LTL‑SCFG（同步语法规则）提取、层次化决策模型与PPO强化学习，辅之以BERT文本编码和语义相似度测度；

**📊 数据集**

使用了三个领域的数据集：Drone Navigation（343条公式）、CleanUp（39条公式）和Pick‑and‑Place（5条公式），并在其零样本泛化集上进行评测；

**📈 对比分析**

与多种基线（LLM+RAG、COT、RL、GCD、微调模型等）对比，scr在域约束满足率上分别达94.66%、98.82%和97.30%，在零样本泛化中也显著优于所有对手，且安全违规率降至接近零；

**⚠️ 局限性**

局限性主要体现在对LTL公式规模有限、对极端复杂指令的处理仍受限，以及对训练数据和语法规则提取过程的依赖较高。

---

## 397. Are These Modules Worth Their Cost? A Paradigm-Level Accuracy-Cost Analysis of In-context Learning Text-to-SQL

**arXiv ID:** 2608.28432 | [PDF](https://arxiv.org/pdf/2608.28432v1)

**作者:** Jiayan Lin `[一作]` (Jinan University), Feiran Huang `[通讯]` (Beihang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对文本到SQL（Text-to-SQL）的ICL（in‑context learning）管道进行模块化分析，系统评估各模块（检索、架构链接、生成策略、候选选择、精炼）对准确率和成本的边际贡献。

**💡 创新点**

首次在统一实现下对17种范式级配置进行成本‑准确率量化，提出“执行反馈精炼”是唯一在所有模型上低成本提升准确率的通用模块，并给出基于模型能力的分层配置指南。

**🔧 技术方法**

使用大语言模型（GPT‑4o‑mini、Gemini‑2.5‑Flash、DeepSeek‑V4‑Flash、GPT‑5.4等），构建统一框架实现检索、链接、生成、选择、精炼等模块；对每个模块按单独交换方式评估。

**📊 数据集**

主要数据集为BIRD开发集（1,534问答对）以及Spider（1,034样本）用于跨基准验证。

**📈 对比分析**

与单一基准模型的端到端准确率相比，分层配置在固定预算下可提升7–10个百分点；相较于传统聚合报告，提供了每个模块的成本效益；在不同模型上验证显示，堆叠中间层往往比直接升级模型更具成本效益。

**⚠️ 局限性**

局限在于只评估了四种主导模型，未覆盖所有可能的模型家族；仅在公开基准上验证，真实数据库环境下的安全性和执行成本仍需进一步研究。

---

## 398. SymboLLM-FE: LLM-Accelerated Symbolic Regression for Automated Feature Engineering on Tabular Data

**arXiv ID:** 2608.28408 | [PDF](https://arxiv.org/pdf/2608.28408v1)

**作者:** Zi-Jian Cheng `[一作]` (Nanjing University), Lan-Zhe Guo `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SymboLLM‑FE框架，将符号回归与大语言模型协同用于自动特征工程；通过Spearman相关排序加扩展滑动窗口生成候选特征子集，符号回归得到可解释的数学表达式，再由LLM生成代码并迭代优化，最终得到可解释且提升预测性能的特征；

**💡 创新点**

双阶段协同创新：①将搜索空间从指数级 O(2^n) 降到多项式级 O(n^2)；②LLM仅作为精炼、解释和验证器，减少调用次数，避免幻觉；③通过统计先验和链式推理实现高效、可解释的特征生成；

**🔧 技术方法**

符号回归（遗传/进化算法）、Spearman相关分析、扩展滑动窗口、LLM（GPT‑4 / GPT‑o1 / DeepSeek‑R1）链式思考代码生成、下游模型（CatBoost、XGBoost、MLP、TabPFN）、特征冗余消除与成本评估；

**📊 数据集**

六个公开真实数据集（Credit‑g、Spaceship、Cmc、Academic、Ailerons、Tesla）以及四个 Kaggle 竞赛；

**📈 对比分析**

对比传统 AutoFE（AutoFeat、OpenFE）和 LLM‑based AutoFE（LLM‑FE、LLM‑SELECT、CAAFE、OcTree、FEBP），在交叉验证下测量分类准确率/ROC‑AUC/F1、回归 RMSE/MAE/R²。SymboLLM‑FE平均提升约1.23% 以上传统 AutoFE，约1% 以上 LLM‑based AutoFE，在 Kaggle 竞赛中平均提升 2.5pp；同时仅使用 4–5 次 LLM 调用，显著提升效率；

**⚠️ 局限性**

计算时间仍较高，尤其在高维数据中；排序+滑动窗口方法可能忽略非相邻变量的协同效应，导致部分潜在特征被遗漏。

---

## 399. A Unified Framework to Elicit Structured Feedback for Interpretable Multi-Trait Essay Scoring

**arXiv ID:** 2608.28407 | [PDF](https://arxiv.org/pdf/2608.28407v1)

**作者:** Shihang Yang `[一作]` (Peking University), Yunfang Wu `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 HiFTS 框架，结合层次化 CoT 反馈与多维度评分，并发布了中文多trait AES 数据集 CFMS-34。

**💡 创新点**

创新点包括：①统一的自回归生成流程先产生结构化 rubric‑grounded 反馈再输出评分；②通过教师 LLM 提取 CoT 训练；③采用 Group Relative Policy Optimization 与多目标奖励对齐；④在推理时使用轻量 BERT 先验引导生成稳定性。

**🔧 技术方法**

技术实现：大型语言模型（Qwen2.5/Qwen3、Gemini‑3 作为教师）、监督微调、GRPO 强化学习、BERT 先验回归、层次化 CoT 生成模板与奖励设计。

**📊 数据集**

使用的数据集为新构建的 CFMS‑34（951 篇中文小学生作文，34 维细粒度 trait）以及英语多trait AES 公开基准 ASAP++。

**📈 对比分析**

与 HISK、STL‑LSTM、MTL‑BiLSTM、ArTS、RMTS 等基线对比，评估指标为 QWK、MSE、WinRate；HiFTS 在 CFMS‑34 上整体 QWK 达 0.677、trait QWK 0.704，MSE 降至 0.741，在 ASAP++ 上亦保持最高平均 QWK，且生成的反馈在 WinRate 与规则化 grounding 分数上均优于对照模型。

**⚠️ 局限性**

局限性：采用固定的自回归生成顺序，未探索更灵活的反馈组织；实验仅覆盖 CFMS‑34 与 ASAP++，缺乏更多写作体裁、学习者群体与真实课堂交互的验证。

---

## 400. How Far Can 5,500 Hours of Driving Take You? A Scaling Law Analysis of Video Diffusion Models

**arXiv ID:** 2608.28404 | [PDF](https://arxiv.org/pdf/2608.28404v1)

**作者:** Victor Besnier `[一作]` (valeo.ai), Matthieu Cord `[通讯]` (valeo.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对5500小时驾驶视频数据从零开始训练视频扩散模型，系统探索模型规模、训练曝光与计算量的缩放规律，并在此基础上训练出9B参数规模的最新开源驾驶视频生成模型；

**💡 创新点**

①训练曝光指数远大于模型规模指数，说明更长训练是提升固定模型最快途径；②固定语料并未打破缩放规律，重复数据可视为新样本；③缩放规律能准确预测到8倍超出拟合模型规模的9B模型表现；

**🔧 技术方法**

使用视频扩散Transformer（AdaLN轨迹调制、CFG、OccAny轨迹提取等）以及VAE编码器；

**📊 数据集**

5500小时nuScenes驾驶视频数据；

**📈 对比分析**

通过与Vista、Drive-WM、GenAD、GEM等现有驾驶世界模型在FID、FVD、ADE等指标对比，9B模型在图像与视频质量上均获得最优或次优表现；

**⚠️ 局限性**

缺乏中间规模3B–4B实验，VAE未被纳入缩放分析，学习率需在大规模训练中动态调整，限制了预测精度。

---

## 401. When Verified Source Becomes Attack Input: Defending Smart Contracts Against LLM-Based Vulnerability Scanning

**arXiv ID:** 2608.28400 | [PDF](https://arxiv.org/pdf/2608.28400v1)

**作者:** Mingyuan Huang `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为DeLLMGuard的部署框架，能够在保持智能合约源代码公开的同时，通过多地址部署和代理/委托/工厂机制使LLM自动扫描变得更困难；

**💡 创新点**

创新点在于：①利用多地址部署分离源代码与运行地址；②引入四种可组合的守护组件（FIC、DLC、PDC、CPC）并通过验证层保证业务逻辑不变；③在公开源代码的前提下实现对LLM扫描的实质性防护；

**🔧 技术方法**

技术包括EVM多地址部署、代理/委托/工厂模式、智能合约存储槽伪装、源代码注释扰动、验证层对部署关系、字节码、存储和状态的完整检查；

**📊 数据集**

使用了来自SCONE-bench的417个真实DeFi漏洞案例（最终筛选387个），涵盖以太坊、BNB Smart Chain、Base、Arbitrum等主链；

**📈 对比分析**

在对比中，DeLLMGuard在387个案例中将LLM（GLM 5.2、GPT 5.5、Claude Opus 4.8）正确识别漏洞的比例从23.5%降至6.6%，相比未加防护的基线下降约70%；BIC（字节码隐藏）表现更差；

**⚠️ 局限性**

局限性包括：1) 仅在模拟沙盒环境评估，未在公开链上验证；2) 对少量代理案例的评估有限；3) 未来更强大的LLM或多模态工具可能突破当前防护；4) 防护基于公开信息，无法阻止具备完整部署关系知识的攻击者。

---

## 402. RetailAgent: Structured Adverse Timing in Self-Conditioned Multimodal LLM Trading Agents

**arXiv ID:** 2608.28399 | [PDF](https://arxiv.org/pdf/2608.28399v1)

**作者:** Yupeng Zhang `[一作]` (University of Wisconsin--Madison), Lisha Chen `[通讯]` (University of Rochester)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了RetailAgent框架，对LLM在匿名的日内股票路径上做的二元长/空决策进行可控评估，并量化曝光匹配的股票内时序表现及策略持久性。

**💡 创新点**

首次揭示LLM在多种输入模态、时间尺度、账户状态和模型家族下表现出持续的负时序结构；通过轨迹洗牌与自我记忆条件实验验证了时序与行动序列的可预测性，并引入互补调度与曝光匹配指标。

**🔧 技术方法**

利用冻结的LLM（Qwen3.5、Claude Haiku、Granite），多模态输入（价格文本、图表、组合）、自写记忆条件、基于VICReg的GRU编码器与soft-token适配器，以及轨迹分析指标（时序α、IC、互补调度）。

**📊 数据集**

使用CSI‑500中国中盘指数的匿名日内价格数据（239分钟网格），包含价格历史与研究收益标签；每个实验条件覆盖1,500+股票‑天，14种标准配置。

**📈 对比分析**

将时序表现与曝光匹配随机调度、价格基线信号（GRU、GBDT、一次逆转）以及跨模型符号检验对比；结果显示在所有模型/模态下均为-30~-60 bps的负时序，洗牌后降至-3~-9 bps；自我记忆条件进一步降低时序至-74 bps并减少行动切换。

**⚠️ 局限性**

实验采用固定路径、匿名数据，未考虑交易成本、市场冲击或实时反馈；仅基于研究收益标签，缺乏真实交易收益；仅评估二元长/空动作，未覆盖更丰富的交易策略，结果可能不直接推广到交互式市场。

---

## 403. Propagating construction-time knowledge quality into medical question answering: A framework grounded in clinical guidelines

**arXiv ID:** 2608.28360 | [PDF](https://arxiv.org/pdf/2608.28360v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 404. AGENT-O: A Semantic Agent Card Framework for Interoperable and Governed Healthcare AI Agents

**arXiv ID:** 2608.28345 | [PDF](https://arxiv.org/pdf/2608.28345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 405. Sustainability of Open-Source Machine Learning Robustness Assessment Tools: A Repository Mining Study

**arXiv ID:** 2608.28396 | [PDF](https://arxiv.org/pdf/2608.28396v1)

**作者:** Joshua Owotogbe `[一作]` (Jheronimus Academy of Data Science), Damian Tamburri `[通讯]` (Jheronimus Academy of Data Science)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 28 个公开的 Python 机器学习鲁棒性工具进行系统的仓库挖掘，分析它们的功能覆盖、社区参与度、维护活跃度以及项目寿命；

**💡 创新点**

首次将仓库挖掘方法应用于鲁棒性工具生态，揭示工具的可持续性差异，并结合生命周期生存分析量化“活跃度衰退”时间，填补了对鲁棒性工具长期可维护性研究的空白；

**🔧 技术方法**

使用 GitHub API 进行仓库元数据抓取，构建 12 项维护指标；通过 Mann‑Whitney U 检验对活跃与不活跃仓库进行比较；采用 Kaplan‑Meier 生存分析估计项目持续活跃的时间分布；

**📊 数据集**

基于 28 个 GitHub 仓库构建的自定义数据集，包含仓库创建、提交、星标、关注、fork、issue、pull‑request 等字段，并提供公开的复制包；

**📈 对比分析**

将活跃仓库与不活跃仓库在 12 个维护特征上进行非参数比较，发现活跃仓库在 issue、PR、提交、贡献者等方面显著更高；生存分析显示非归档仓库的中位活跃时间约为 40.6 个月，1 年后保持活跃概率 92.4%，5 年后降至 30.4%；

**⚠️ 局限性**

样本量有限（仅 28 个仓库），且仅覆盖 Python 和 GitHub，搜索阈值和 180 天提交规则可能遗漏活跃或兼容性良好的工具，未考虑私有使用、下载量、依赖链等维度，导致可推广性与完整性受限。

---

## 406. MAP: A Benchmark on Multimodal Accessibility Planning for Real World Places

**arXiv ID:** 2608.28384 | [PDF](https://arxiv.org/pdf/2608.28384v1)

**作者:** Jason Armitage `[一作]` (University of Zurich), Sarah Ebling `[通讯]` (University of Zurich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究构建了MAP基准，用来评估多模态AI系统在可访问性需求规划中的表现。

**💡 创新点**

创新点包括提出声明验证和视觉证据检索两项新任务，动态更新地面真数据，并在开放世界环境中进行评估。

**🔧 技术方法**

采用多模态大型语言模型（如Gemini、Claude、GPT等）与视觉输入、自动评判器以及人工评审结合的技术。

**📊 数据集**

使用从Google Places、Ginto、官方网页、用户评论和图片等公开渠道构建的，覆盖苏黎世的地点及可访问性特征的地面真数据集。

**📈 对比分析**

通过标签化评判（Safe、Unverified、Contradictory等）和视觉证据标签（CVE、IP-CFA等）比较模型，结果显示大多数模型回答谨慎，Gemini 3.6、3.1 Pro、GPT-5.6 Terra和Claude Sonnet 4.6在两项任务中表现最好。

**⚠️ 局限性**

局限性包括地面真数据对物理可访问性偏重、数字源更新滞后、仅覆盖苏黎世、提示语为英文且不涵盖所有真实用户需求。

---

## 407. GRACE:Gradient-guided Coreset Selection for LLM Unlearning

**arXiv ID:** 2608.28361 | [PDF](https://arxiv.org/pdf/2608.28361v1)

**作者:** Praveen Bushipaka `[一作]` (University of Pisa), Tommaso Cucinotta `[通讯]` (Scuola Superiore Sant'Anna)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于梯度的核心集选择方法 GRACE，用于在 LLM 失忆过程中自动构造遗忘集和保留集。

**💡 创新点**

创新点在于通过非负正交匹配追踪（NNOMP）逼近遗忘方向，并在投影后聚类挑选保留样本，解决仅给出少量示例时的遗忘/保留集合构建问题。

**🔧 技术方法**

技术包括 Rademacher 哈希梯度压缩、非负正交匹配追踪、梯度投影与 K-means 聚类以及两类 LLM 失忆算法的联合使用。

**📊 数据集**

使用了两组数据集：MUSE+ Dolly-15k（混合）和 WMDP-Bio+AlpaCare-MedInstruct（领域特定），并在 LLaMA 3.1 8B 与 Qwen 2.5 3B 上验证。

**📈 对比分析**

与基于嵌入检索和 RASLIK 的基线相比，GRACE 在模型效能（MUT）上平均提升 5-6 分，遗忘质量（FQ）保持相当，实验覆盖多模型、多算法、多域。

**⚠️ 局限性**

局限包括：假设种子示例能够产生相对统一的遗忘方向；实验仅在可观测的 ground-truth 失忆场景下评估；对更大模型、多语言或更异构请求的泛化有限；梯度计算开销仍显著。

---

## 408. Euclidean Fourier Neural Operators

**arXiv ID:** 2608.28425 | [PDF](https://arxiv.org/pdf/2608.28425v1)

**作者:** Nathanael Bosch `[一作]` (École Polytechnique Fédérale de Lausanne), Michael F. Herbst `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了欧几里得傅里叶神经网络（EFNO），能够在不同周期域和网格尺寸下学习并泛化算子，并在热方程与材料科学的交换-相关势预测任务中进行验证。

**💡 创新点**

通过将傅里叶核参数化为连续波矢函数，去除了传统傅里叶神经网络对周期域的隐式依赖，实现了域无关的算子学习。

**🔧 技术方法**

使用 FFT 计算卷积，采用线性高斯基函数作为符号函数 κθ 进行参数化，并在 FNO 与 EFNO 上训练和对比。

**📊 数据集**

使用热方程模拟数据以及 ML‑RPA 数据集中金刚石与水体系（PBE 密度→RPA 势）的数据。

**📈 对比分析**

在不同格子形状、尺寸及超细网格上对比 FNO 与 EFNO，EFNO 的 L2 误差保持 <10⁻⁵%，而 FNO 在超细格子上误差升至 ~90%；在材料任务中，EFNO 的 WRMSE 相比 FNO 降低一倍，并显著优于 PBE 基线。

**⚠️ 局限性**

仅采用简单的高斯基符号，缺乏对更复杂符号表达式的探索；对非周期边界或更复杂 PDE 的适用性仍待进一步验证。

---

## 409. VERA-8B: Evidence-Grounded Audit Risk Reasoning from SEC Filings

**arXiv ID:** 2608.28402 | [PDF](https://arxiv.org/pdf/2608.28402v1)

**作者:** Menghan Liu `[一作]` (New York University), Elynn Chen `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一套端到端的审计风险推理系统 VERA‑8B，能够直接从 SEC 预执法文件中识别风险类别、引用确切证据、解释审计机制并在证据不足时做出拒绝。

**💡 创新点**

创新点包括：① 统一证据门控规则书，将公开的 PCAOB 标准、AAER 经验与文件文本映射到可执行合同；② 在此基础上实现结构化 SFT + 自适应 GRPO 细化，保证模型在保持证据标准的同时提升多标签识别与证据对齐；③ 引入选择性拒绝与不确定性路由，自动过滤可审计案例；④ 通过 AuditBridge 将验证后的 JSON 输出转换为审计员可直接使用的报告。

**🔧 技术方法**

技术手段包括：使用 Llama‑3.1‑8B 作为基础 LLM；对模型进行结构化 SFT 以学习证据门控的输出；利用 GRPO 强化学习进一步校正多标签和证据对齐；实现可执行的 JSON 验证器与不确定性校准的 conformal 路由；以及 AuditBridge 负责将模型输出转化为机器可读和审计员友好的报告。

**📊 数据集**

数据集来源于 SEC EDGAR 的预执法文件、SEC AAER 执法结果以及 PCAOB 审计标准；在 1,352 家发行人训练集、298 家验证集和 310 家冻结测试集（全部预执法文本，排除后续执法或重复段落）上进行实验。

**📈 对比分析**

采用 issuer‑disjoint 锁定协议，按统一评测标准（binary F1、category micro‑F1、EV‑Micro‑F1、JSON 合法率、未验证主张率）与通用与金融专用基线对比；VERA‑8B 在测试集上达 95.3% binary F1、87.1% EV‑Micro‑F1，未验证主张率仅 3.1%，比最强基线提升 45% F1、减少 81% 未验证率，自动覆盖率 83%。

**⚠️ 局限性**

局限性主要体现在：仍依赖可获得的预执法文本与规则书的完整性，难以处理高度多义或跨文件、跨时期的证据链接；对极少见的审计风险类别覆盖有限；部署时未能通过严格的证书化校准，需进一步提升鲁棒性和解释性。

---

## 410. CamoDocs: A Poisoning Attack Against Retrieval-Augmented Language Models Using Camouflaged Documents

**arXiv ID:** 2608.28389 | [PDF](https://arxiv.org/pdf/2608.28389v1)

**作者:** Jaewon Jung `[一作]` (Seoul National University), Jinho Lee `[通讯]` (Seoul National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的RAG毒化攻击方法，能够在不直接插入查询的情况下，利用扩散令牌（dispersion tokens）与连贯性过滤器，生成既能诱导错误答案又能规避现有防御的恶意文档。

**💡 创新点**

创新点在于：①摆脱了查询包含的传统毒化模式，解决了可检测的词汇和嵌入空间痕迹；②采用梯度导向的令牌替换和散布损失来扩散恶意文档的嵌入分布；③通过轻量级语言模型进行连贯性过滤，保持文本可读性并降低被质量过滤的风险。

**🔧 技术方法**

技术包括：文档合成器LLM（LLM_synth）生成正负样本；分块（chunking）将文档拆分为子文档；梯度导向令牌替换与散布损失（dispersion loss）结合的扩散令牌生成；连贯性过滤器（LM_coh）评估并筛选替换令牌；最终将优化后的子文档与对抗子文档拼接形成毒化文档。

**📊 数据集**

实验使用公开问答数据集 HotpotQA、Natural Questions、MS‑MARCO，以及 NeoQA（检索依赖性评估）。另外对专有模型 GPT‑5.4‑mini 与 Claude‑Haiku‑4.5 进行测试。

**📈 对比分析**

与七种主流防御（Query Detection、Divide‑and‑Vote、RobustRAG、Isolation Forest、LLM Filter、Rerank、TrustRAG）以及三种开源LLM（Qwen3‑8B、Llama‑3.1‑8B、Mixtral‑8x7B）对比，攻击成功率（ASR）普遍高于现有攻击方法（如 PoisonedRAG、PIA、CorruptRAG），在大部分场景下实现 50%–70% 以上的 ASR；对 TrustRAG 的评估显示，虽然该防御能削弱攻击，但会导致检索依赖任务的显著准确率下降。

**⚠️ 局限性**

局限性包括：①需要攻击者能将毒化文档注入知识库，限制了在严格管控环境下的适用性；②相对简单攻击相比，计算成本更高，主要集中在离线生成阶段；③攻击效果依赖于对抗令牌与受害检索器的嵌入迁移，某些检索架构或预处理策略可能降低迁移效果。

---

## 411. Semantic Head Specialization Guides Hybrid ViT Attention for Multimodal LLMs

**arXiv ID:** 2608.28383 | [PDF](https://arxiv.org/pdf/2608.28383v1)

**作者:** Chenhong He `[一作]` (Peking University), Shuhuai Ren `[通讯]` (Xiaomi Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了视觉 Transformer（ViT）注意力头的行为，发现全自注意力下头会区分为物体专注与背景专注的“语义头专化”（SHS）模式，并提出 SHS‑Index 指标量化此现象；在此基础上识别了窗口交互、令牌序列化和局部 softmax 分配三大结构因素，并以此指导设计了近乎等同全注意力、但 FLOPs 降至 1/6.5 的 Ariadne 混合注意力结构。

**💡 创新点**

创新点在于①首次将 ViT 关注模式抽象为可量化的 SHS 现象并给出 SHS‑Index 评估方法；②系统解析三大结构因素如何影响 SHS 并形成设计原则；③基于 SHS 诊断结果设计出 Ariadne Attention，实现高效且性能接近全注意力的混合注意力方案。

**🔧 技术方法**

技术手段包括：ViT 的全注意力与分块窗口注意力对比实验；滑动窗口（SWA）注意力、行/列序列化、sink bias 等结构改造；SHS‑Index 的 AUROC 计算；在 22 项多模任务上进行基准评测；以及对不同分辨率下的 FLOPs 与实际推理时间进行测量。

**📊 数据集**

使用的数据集包括 COCO val2017（用于生成注意力热图和前景/背景标签）、20 项视觉任务（AI2D、OCRBench 等）、2 项视频任务（Video‑MME）以及 22 项综合基准（含 MMBench‑EN 等）。

**📈 对比分析**

通过在统一训练设置下对全注意力、分块窗口注意力和 Ariadne 进行对比，使用 SHS‑Index 评估内部结构差异，并在 20‑图像任务上取得 40.40 分，几乎匹配 40.92 的全注意力；同时注意力 FLOPs 降至原来的 1/6.5，ViT 前向时间下降 13.5%。

**⚠️ 局限性**

局限性包括：实验仅在单一训练配置、单一随机种子、单一语言模型规模下完成；未在更大规模或多种语言模型上验证 SHS‑Index 与基准的相关性；部分几何或计数任务仍表现不佳，提示混合注意力仍需进一步改进。

---

## 412. When Linguistic and Internal Confidence Diverge in Large Language Models

**arXiv ID:** 2608.28382 | [PDF](https://arxiv.org/pdf/2608.28382v1)

**作者:** Hefan Zhang `[一作]` (Dartmouth College), Soroush Vosoughi `[通讯]` (Dartmouth College)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对大型语言模型在不同任务、模型和提示下的语言自我信心与内部置信度（logits概率或语义熵）进行跨通道对齐评估，探讨两者在关联、大小一致性与校准上的差异。

**💡 创新点**

创新点在于提出三轴（关联、幅度一致、校准）交叉评估框架，发现语言置信度是一个失真通道，且其分布特征（均值、方差）是主要影响因素，并给出实用的可靠性使用准则。

**🔧 技术方法**

使用Pearson/Spearman相关、欧氏距离、期望校准误差（ECE）、语义熵、回归分析等技术，结合白盒/黑盒内部置信度代理来量化对齐程度。

**📊 数据集**

数据集覆盖8个分类任务（CoLA、QNLI、QQP、MMLU、Temporal Sequences等）和2个生成任务（CoQA、TriviaQA），共10个基准，使用30个来自LLaMA、Mistral、Qwen的公开模型。

**📈 对比分析**

方法通过比较语言置信度与内部置信度的相关性、距离和ECE来评估对齐；结果显示实例级关联弱，关联随任务难度降低和模型基线强度提升而增强，指令微调往往提升自报置信度但不改善校准。

**⚠️ 局限性**

局限性包括仅评估公开模型和两种置信度代理、采用相关性分析而非因果实验、采样子集可能产生变异、对生成任务的语义熵假设和聚类方法有限。

---

## 413. Real-Time Musculoskeletal Surrogates for Pediatric Cerebral Palsy: a Credibility Pilot

**arXiv ID:** 2608.28371 | [PDF](https://arxiv.org/pdf/2608.28371v1)

**作者:** Mohammad Arif Ul Alam `[一作]` `[通讯]` (North Carolina A & T State University), Mohammad Arif Ul Alam (North Carolina A & T State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在儿童脑性瘫痪步态数据上构建并评估了一个以受试者特征为条件的因果神经网络 MSK surrogate，能够实时预测肌腱-肌肉长度并生成肌肉力学输出。

**💡 创新点**

创新点包括：① 无泄漏的受试者层级评估协议（LOSO + 锁定测试），② 将 OpenSim 静态参数与动态卷积网络融合的因果条件化模型，③ 结合物理一致性损失与训练时专用扰动的鲁棒性增强，④ Monte Carlo 可信度试点揭示仅输入传播导致的不可靠置信区间。

**🔧 技术方法**

使用了因果时间卷积网络 (TCN)、多任务均方误差物理损失、训练专用扰动、以及 90% 置信区间的 Monte Carlo 采样。

**📊 数据集**

数据集：9 名患有脑性瘫痪的儿童步态（共 7548 帧），其中 6 名用于开发（LOSO 验证），3 名作为锁定测试；所有样本均通过 OpenSim 生成真实肌肉力学标签。

**📈 对比分析**

与传统 OpenSim CMC 对比：MT‑长度预测 R² ≈ 0.92–0.95，平均推理时间 7–10 ms，低于 100 ms 交互阈值；肌肉力预测 R² 仅为 0.2–0.6（存在极端负值），说明直接力估计尚不成熟。

**⚠️ 局限性**

主要限制：① 样本量极小且异质性高，导致力预测不稳定；② 仅采用输入参数扰动，未充分捕捉模型不确定性；③ 物理一致性损失在当前配置下无显著改进；④ 缺乏实际肌肉力测量作为真值，需进一步扩展数据集和改进损失设计。

---

## 414. It Takes Three to Converse: Empirical Observations on How the Developer, the Convener and the Participant Shaped 119 Polis Conversations

**arXiv ID:** 2608.28368 | [PDF](https://arxiv.org/pdf/2608.28368v1)

**作者:** Lodewijk Gelauff `[一作]` `[通讯]` (Generation Lab), Lodewijk Gelauff (Generation Lab)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对119个Polis异步公民参与对话进行大规模数据分析，考察平台开发者、主持人设置与参与者行为对结果的影响。

**💡 创新点**

首次以真实公开数据为基础，揭示平台优先级、调节策略、种子语句等设计决策如何系统性地改变投票模式、语句接受率和聚类结果，并指出导出聚类不可再现性。

**🔧 技术方法**

采用统计分析、bootstrap置信区间、回归模型、文本匹配与Python/Julia脚本，对平台源码进行重构以验证实现细节。

**📊 数据集**

使用从271份Polis导出中筛选出的119份对话数据，包含约688万条投票、约28万条语句，涵盖欧盟、美国、台湾等不同国家与地区。

**📈 对比分析**

通过对比不同调节策略、种子比例与开放式语句下的投票比例、聚类数量、语句接受率等指标，并使用bootstrap与多重检验校正，发现严格/宽松调节、种子比例和开放式语句显著影响参与度与聚类结果。

**⚠️ 局限性**

因观察性设计缺乏因果性，数据缺失时间戳与细节，平台bug导致优先级不一致，导出聚类不可重现，参与者身份追踪有限等限制导致结论需要进一步验证。

---

## 415. EvoUndo: Recoverability-Constrained Self-Evolution for LLM Agent Harnesses

**arXiv ID:** 2608.28363 | [PDF](https://arxiv.org/pdf/2608.28363v1)

**作者:** Tanmay Sah `[一作]` (Independent Researcher), Tanya Sah `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EvoUndo 框架，支持 LLM 代理在运行时自我演化时，保证对所做的变更能够在不同的状态下可靠恢复；同时提供证据化的可恢复性验证、对状态的可观测等价判定、以及闭环自适应修复流程；

**💡 创新点**

创新点在于将可恢复性作为自我演化的核心约束，将恢复性定义为跨状态的相对可观测等价；通过引入可观测等价合同、证据捕获、恢复语言 L_0/L_1 与诊断粒度 D_0/D_1 的组合，系统性拆解并验证“定位”与“表达”两大瓶颈，并证明在不同模型上该瓶颈表现一致；

**🔧 技术方法**

使用大语言模型（gpt-oss-120b、Qwen3.8-27B）进行变更生成与闭环修复；对变更进行对照实验，生成对抗性与 OOD 的计数器事实状态；采用可观测等价合同、证据捕获与恢复程序的语义化语法；利用统计检验（McNemar、bootstrap 95% CI）评估恢复效果；

**📊 数据集**

构建 600 个未见自我演化任务（六大体系：配置、工具、中间件、监听器、资源、多面向）；120 个带注入缺陷的对照基准；300 个全新 hold‑out 任务；采用公开权重的 gpt-oss-120b 与 Qwen3.8-27B 作为实验模型；

**📈 对比分析**

在 2×2 因子设计下比较诊断粒度与恢复语言，测量 Rescue@B（在预算 B=4 下可恢复的缺陷比例）。在 S_0 族中 D_1L_0 将恢复率从 0% 提升至 79.2%；在 S_1 族中 D_0L_1 将恢复率从 0% 提升至 99.3%；在主模型上 D_1L_1 对 S_1 产生 6.3% 的负面交互；Qwen3.8-27B 对此交互无显著影响。整体上，通过 EvoUndo 可将可接受的可恢复正向变更率从 46.8% 提升至 76.8%（+30% 绝对提升）。

**⚠️ 局限性**

限制包括：恢复语言 L_0/L_1 仅覆盖模型化的持久状态，缺乏形式化完整性；未对前向变更 m 进行联合优化；未处理分布式状态、第三方 API、外部网络等；无法补偿不可逆物理/财务效应；实验仅在单机 LLM 环境下验证，未涉及多主机或实时系统；

---

## 416. False-CSI Attacks in Power-Domain NOMA for 6G: A Threat Taxonomy and System-Level Impacts

**arXiv ID:** 2608.28351 | [PDF](https://arxiv.org/pdf/2608.28351v1)

**作者:** Samira Jafarli `[一作]` (Baku State University), Suleyman Uludag `[通讯]` (University of Michigan--Flint)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文针对6G功率域NOMA中伪CSI攻击提出了一套威胁分类方法，并通过系统层级影响链分析了这些攻击对调度、功率分配、SIC可靠性、吞吐率、公平性和机密性等关键指标的潜在破坏。

**💡 创新点**

创新点在于把伪CSI视为控制输入完整性问题，构建了以幅度（低报/高报）和排序效应（保持/边界/逆序）为轴的双维分类框架，并扩展到协同、组变、方向伪造和训练阶段注入等多种攻击情景，形成了一套系统化、面向攻击者意图的威胁模型。

**🔧 技术方法**

主要技术手段为理论建模、系统级影响链推导、案例说明与图表展示；未采用具体算法实现或仿真模型，而是通过对NOMA决策链的逻辑分析来阐述攻击传播机制。

**📊 数据集**

未使用任何公开数据集，攻击模型与系统影响分析基于通用的信道模型与假设条件（如理想的SIC阈值、功率分配约束等）。

**📈 对比分析**

论文未进行实验对比或性能测评，只提供了攻击导致的系统层面指标（如吞吐量下降、SIC误码率上升、公平性指标恶化、机密性泄露）的定性评估；若需量化评估，需要自行在仿真平台上实现所述攻击并测量对应指标。

**⚠️ 局限性**

局限性包括：缺乏真实或仿真验证；仅聚焦功率域NOMA，未涵盖频域或码域NOMA；对防御技术与对策未给出具体实现；在多维网络（如MIMO+RIS+AI调度）环境下的交互效应尚未完全展开。

---

## 417. A Constant Metric Distortion Protocol for Approval Voting Given Plurality Polls

**arXiv ID:** 2608.28340 | [PDF](https://arxiv.org/pdf/2608.28340v1)

**作者:** Fabian Frank `[一作]` (Technische Universität München), Jannik Peters `[通讯]` (Technische Universität München)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于普选投票结果的 k‑Plurality Approval 方案，利用比例阈值 k 让选民在其排名前缀中覆盖至少 k·n 名选民的首选候选人并投票批准；

**💡 创新点**

证明该规则在度量失真框架下能达到 2+√5≈4.236 的常数失真上界，并给出该上界可达性证明；同时将分析扩展至 Bucklin、majoritarian compromise 及通用的 fallback bargaining 规则，改进此前的 11 的上界至 5；

**🔧 技术方法**

采用指标失真方法，构造占优关系与分配函数，利用 Hall 定理构造分数匹配，从而获得成本上界；还通过噪声分析与随机抽样证明该规则对投票误差鲁棒，给出样本复杂度；

**📊 数据集**

本文未使用公开数据集，而是通过构造一系列极端实例和对应的度量空间来证明上界与下界；

**📈 对比分析**

与其他常用规则（如 Plurality veto、Copeland、Maximal Lotteries 等）对比，k‑Plurality Approval 在度量失真上仅为 4.236，优于已有常数失真规则；对于 Bucklin 与 majoritarian compromise，上界降至 5；

**⚠️ 局限性**

局限性包括：规则并非策略防御，选民可能在两轮投票中偏离协议；需要两轮交互，无法在单轮 O(m) 比较内完成；对极端噪声下失真仍有上界 2+√5 的限制。

---

## 418. Where Does Balance Break? Boundary Discovery for Game Balance Testing under a Finite Simulation Budget

**arXiv ID:** 2608.28364 | [PDF](https://arxiv.org/pdf/2608.28364v1)

**作者:** Hiroki Mukai `[一作]` (Ritsumeikan University), Katsuro Inoue `[通讯]` (Ritsumeikan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在有限模拟预算下，研究竞争游戏的平衡回归测试中的边界发现问题，并提出了 BBExplorer 方法。

**💡 创新点**

创新点是将多方向候选生成、两阶段抽样筛选和自适应步长衰减三大机制融合，实现对平衡边界的高效逼近。

**🔧 技术方法**

采用搜索式软件测试技术，具体实现为多方向候选生成、两阶段抽样（粗筛+精评）和指数衰减的步长控制。

**📊 数据集**

使用了两个实验环境：低维度的 Turnbased（2D/3D）和高维度的 RTS 游戏 Generals（3D/4D/5D）。

**📈 对比分析**

与随机采样和固定步长基线对比，BBExplorer 在 2-5 维空间中获得更高的边界发现率、更近的边界距离，并在计算成本上更具优势。

**⚠️ 局限性**

局限性包括仅采用单路径搜索、轴对齐候选导致无法捕捉非轴向边界、固定抽样次数难以自适应不同噪声级别，以及对远离已知平衡区的大范围配置覆盖不足。

---

## 419. Timing-Aware Repurchase Prediction for Web-Scale E-Commerce: Survival Models for Multi-Surface Grocery Recommendation

**arXiv ID:** 2608.28393 | [PDF](https://arxiv.org/pdf/2608.28393v1)

**作者:** Akshay Kekuda `[一作]` (Walmart Global Tech), Kannan Achan `[通讯]` (Walmart Global Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

用生存模型替代多时点二分类模型，预测顾客对已购商品的复购时间并生成跨时点可用的排名与概率。

**💡 创新点**

①发现复购风险呈轻微递减(k≈0.9)；②用单一AFT模型取代三模型；③通过4参数校准实现无跨时点单调性的概率输出。

**🔧 技术方法**

XGBoost AFT（Weibull/Log‑Normal/Logistic）、离散时间风险模型、参数化校准与特征重要性重排序。

**📊 数据集**

数千万条（customer, item）对的自家杂货电商平台数据，含30天内复购标签与右删失记录。

**📈 对比分析**

在离线评估中，用P@h、NDCG等指标与原始三模型和DT模型对比，单一AFT在所有时点上均优于基线，且树数约三倍减半。

**⚠️ 局限性**

仅离线实验；数据仅来自单一杂货零售商；未验证公共基准；未建模篮子级相关性；30天删失窗口限制长周期项目；分布参数仅全局估计。

---

## 420. Fidelity Is Not Enough: Dispatch-Level Instrumentation for Agentic Datasheet Extraction

**arXiv ID:** 2608.28439 | [PDF](https://arxiv.org/pdf/2608.28439v1)

**作者:** Qing Ye `[一作]` (Infineon Technologies AG), Meng-Hsuan Lin `[通讯]` (Infineon Technologies AG)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构建了面向技术文档的 agentic 数据表提取基准，既评估提取的忠实度，又引入物理可验证性。

**💡 创新点**

创新点在于将提取与物理实验相结合，开发了基于工具调用记录的失败归因和静默失败检测器，提供了可观测的诊断。

**🔧 技术方法**

技术上采用了两阶段冻结、工具调用日志、规则化归因器、以及与物理测量平台的对接。

**📊 数据集**

使用了 37 条手工挑选的定量声称（25 条来自 3 个电子元件的 datasheet，12 条来自 A4988 步进驱动器），其中 2 条可通过物理腔室测量。

**📈 对比分析**

实验在 Claude Sonnet 4.6、GPT‑5.1 和 Qwen3.6‑27B 三个模型上进行，对齐后测得前两者在 25 条声称上 100% 正确且稳定，而 Qwen 的稳定性较差，平均 19/25；工具层带来的延迟约为 1.2‑1.8 倍。

**⚠️ 局限性**

局限性包括检测器仅在特定静默错误上无误报但未评估检测能力、样本量不足的错误归因、缺乏人类基准、仅 2 条声称可物理验证，以及模型比较混杂了多种部署因素。

---

## 421. Neuromorphic architectures as numerical solvers for computational neuroscience

**arXiv ID:** 2608.28387 | [PDF](https://arxiv.org/pdf/2608.28387v1)

**作者:** Jakob Jordan `[一作]` (Yale University), Rajit Manohar `[通讯]` (Yale University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

将神经形态硬件视为分布式ODE求解器，提出使用多比特数据包和高阶数值积分方法，改造现有的NeuroScale架构以模拟连续耦合的神经网络，并通过后仿真验证能效与延迟的提升。

**💡 创新点**

创新点在于将连续耦合的非线性ODE映射到多比特包通信、利用高阶Runge–Kutta求解器降低通信频率，并将这些理论直接实现到数字神经形态硬件中，实现了无尖峰神经模型的高效加速。

**🔧 技术方法**

使用了混合精度定点算术（Q8.24、Q4.18）、多比特包路由、时序同步的二维网格网络、以及高阶Runge–Kutta（RK1、RK2、RK3）求解器，并对NeuroScale多核架构进行了后仿真。

**📊 数据集**

实验采用了随机稀疏连接的泄漏积分神经网络（ReLU非线性），并用SciPy DOP853求解器生成精确轨迹作为基准；未使用公开数据集，而是以合成网络作为验证。

**📈 对比分析**

对比前向Euler、Ralston（2阶）和3阶求解器，结果显示高阶方法在相同误差下减少了约60倍的事件频率、延迟和能耗；数值误差与步长的关系与理论一致，证明了理论到实践的转化。

**⚠️ 局限性**

局限性包括有限精度导致高阶方法收益受限、硬件成本提升（内存和计算）、仅针对连续耦合模型验证、未考虑自适应步长和根查找的实现挑战，以及对大规模真实数据集的可扩展性未进行评估。

---

## 422. A Guided Inquiry Approach to Students Co-Designing Generative AI Course Policies

**arXiv ID:** 2608.28501 | [PDF](https://arxiv.org/pdf/2608.28501v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 423. xTRUCE: A Provably Safe Arbiter for Multi-xApp Conflict Mitigation in Agentic O-RAN

**arXiv ID:** 2608.28532 | [PDF](https://arxiv.org/pdf/2608.28532v1)

**作者:** Le Xia `[一作]` (Virginia Tech), Haijian Sun `[通讯]` (University of Georgia)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了xTRUCE，一个基于约束层级和两阶段优先级仲裁的可验证安全仲裁器，用于在代理型O‑RAN中解决多xApp冲突并输出安全gNB控制动作；

**💡 创新点**

创新点在于将xApp提议统一结构化、构建三层硬性约束层级、通过两阶段优先级仲裁实现硬目标优先满足、并生成可解释的冲突证书；

**🔧 技术方法**

采用约束层级建模、凸优化（CVXPY求解）、两阶段Lexicographic优化与Lagrange乘数价格回馈、以及双时钟动作空间分解；

**📊 数据集**

实验数据来自Python仿真环境（与ChatGPT‑5.6、Claude‑Sonnet‑5实时交互）以及基于OpenAirInterface/FlexRIC的OTA实测；

**📈 对比分析**

与直接（Direct）和剪裁（Clipping）基准比较，xTRUCE在全范围幻觉率下保持0%物理/运营约束违背、在资源超载时实现优先级一致的KPI满足、并通过证书使LLM在3轮内恢复目标；

**⚠️ 局限性**

局限性在于仅处理多xApp冲突、假设前一时刻干扰恒定，未覆盖多rApp、动态干扰漂移及非凸情况。

---

## 424. Rethinking Vulnerability Remediation as a Capacity Allocation Problem

**arXiv ID:** 2608.28509 | [PDF](https://arxiv.org/pdf/2608.28509v1)

**作者:** Jana Stucke `[一作]` `[通讯]`, Jana Stucke

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文将漏洞修复视为队列控制问题，评估不同工具（Apache Jira、Mozilla Bugzilla、Red Hat errata、五个公共Jira组织和npm依赖图）中的排队状态、容量分配、所有者约束和专业网络对修复延迟的影响。

**💡 创新点**

创新点包括：① 将漏洞修复从单纯的优先级排序转向流量控制视角；② 发现并量化“四种容量移动模式”（孤立、已连接但饱和、已连接且有余量、已预配）；③ 通过Fast‑Track容量预留模型展示在不同预留比例下关键漏洞平均停留时间的非线性下降；④ 证明漏洞在依赖图中呈弱聚集，暗示到达率可能非Poisson；⑤ 结合多种预测方法（AUC、Conformal、Empirical Quantile）表明即使覆盖率高，预测区间宽度仍是评估工具效用的关键。

**🔧 技术方法**

主要技术手段：队列理论（M/G/1、Pollaczek–Khinchine、Cobham优先级队列）、Hill尾部指数估计、AUC评估、Conformal预测（CQR、Mondrian）、实证差分法、快车道（Fast‑Track）模拟、所有者专家重叠度量、网络关联度分析。

**📊 数据集**

使用的数据集包括：2000条Apache Jira问题（1672已解决）、2000条Mozilla Bugzilla核心问题、163条Red Hat CVE记录（含风险等级）、5个公共Jira组织（Apache、MariaDB、Qt、MongoDB、Red Hat）各2000条问题、以及包含1021个包的npm依赖图与OSV历史。

**📈 对比分析**

比较方法：使用AUC衡量优先级/队列上下文/Conformal方法对修复时间的判别力；利用差分法评估从超负荷到负荷下降的队列状态转换；采用Fast‑Track模型模拟不同容量预留比例的效果。结果显示：优先级在Apache几乎无区分力（AUC≈0.5），在Mozilla与Red Hat分别提升至0.64和0.79；队列状态转变可使关键漏洞停留时间平均减少约90–120天；在容量预留为25%时，Apache关键漏洞平均停留时间从265天降至约138天；预测方法覆盖率高但区间宽度大，表明单纯提升覆盖率并不能显著提升实用性。

**⚠️ 局限性**

局限性包括：① 所有实验为观测性或模拟，缺乏随机化干预；② 只观测到两条队列状态转变，差分结果不具普适性；③ 依赖Poisson到达假设，可能忽略依赖图导致的到达相关性；④ 所有者、专业网络信息基于历史分配，未衡量当前可用性与授权；⑤ 大量缺失的指派信息导致重叠度量不确定；⑥ 仅使用公开的Jira/ Bugzilla 数据，缺少企业内部容量与授权细节；⑦ 预测区间宽度的统计覆盖度并未转化为可操作的修复决策。

---

## 425. Recognition Without Enforcement: Configuration-Dependent Failures in LLM Agent Instruction Arbitration and External Control

**arXiv ID:** 2608.28502 | [PDF](https://arxiv.org/pdf/2608.28502v1)

**作者:** Jun Wen Leong `[一作]` `[通讯]`, Jun Wen Leong

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大语言模型代理在权限冲突时的自我仲裁机制，发现模型能够识别伪造的权威信息，却在某些配置下不将此信息用于执行决策，形成“识别-执行差距”。

**💡 创新点**

提出并验证了该差距的存在性和可度量性，指出此漏洞并非模型固有缺陷，而是由部署配置决定；同时展示了一种外部参考监控（HMAC+能力门控）可可靠阻止伪造请求。

**🔧 技术方法**

技术包括线性探测（probe）读取源标记、强制反射和自我推理（CoT）验证识别、跨模型桥接和稀疏自编码器（SAE）消融等内在干预；外部机制使用 HMAC‑SHA256 身份验证和可选能力令牌实现。

**📊 数据集**

使用了 124,325+ 的记忆冲突试验、29,000+ 的多样化提示试验、7,500+ 的预留评估试验以及 17,500+ 的生成器鲁棒性试验，覆盖 46 端点、29 公开模型和 15 家供应商。

**📈 对比分析**

对比方法：在 46 个模型端点上进行同一攻击构造的固定提示测试，fleet‑mean 执行率仅 1.21%（95% CI 0.5–2.1%），但局部执行率可高达 100%；外部监控在 900 条测试中 0% 执行，说明该方法在检测/阻断方面性能显著优于任何内在或提示层防御。

**⚠️ 局限性**

局限性包括：识别-执行差距受部署配置、提示形式和温度的强烈影响，导致在不同窗口或不同模型集上表现不稳定；提示层防御仅对特定模型有效；外部监控需依赖外部基础设施；未能证明该差距在所有模型族中普遍存在，仅在所测试的 46 端点上观察到；攻击者可通过自适应改造绕过某些防御。

---

## 426. AcrossVAM1.0: Particle World Modeling for Text-Assisted Robot Video Prediction

**arXiv ID:** 2608.28491 | [PDF](https://arxiv.org/pdf/2608.28491v1)

**作者:** Yafei Zhang `[一作]` (Across Physical AI), Nan Wu `[通讯]` (Across Physical AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 AcrossVAM1.0，一种将机器人视频预测拆分为语义粒子运动和因果外观合成的轻量文本辅助模型。

**💡 创新点**

创新点在于使用冻结的 SAM3-DLP 语义粒子编码器和 0.28M 规模的 Transformer 动力学核心，以及学习的残差交付掩码实现运动与外观的解耦。

**🔧 技术方法**

采用了冻结的 SAM3-DLP 编码器、OpenCLIP 语言编码、FiLM 注入、Transformer 运动预测、残差外观解码以及基于 Alpha 的可学习交付掩码。

**📊 数据集**

在自行构建的 VRS（Video Robot Segmentation）基准数据集上进行评估，该数据集包含 2746 条真实机器人轨迹及其对应语义分割和自然语言指令。

**📈 对比分析**

与 persistence、粒子解算器、公开基线 FitVid、SlotFormer 等进行对比，AcrossVAM1.0 在 PSNR、SSIM、LPIPS 与运动区 PSNR 上相较 persistence 有小幅提升（约 0.6 dB PSNR、0.0044 SSIM），但在 LPIPS 仍略逊。

**⚠️ 局限性**

局限性包括仅能预测五帧 128×128 分辨率，依赖预标注的部件掩码，语言对运动的影响有限，跨机器人泛化不稳，且交付掩码仍需改进。

---

## 427. On Left Adjoints Preserving Colimits in Homotopy Type Theory

**arXiv ID:** 2608.28473 | [PDF](https://arxiv.org/pdf/2608.28473v1)

**作者:** Perry Hart `[一作]` `[通讯]` (University of Minnesota), Perry Hart (University of Minnesota)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在野生范畴（wild category）中左伴随保持余极限的证明存在的连贯性问题，提出 2-可合性 条件并证明悬挂函数、连接函数以及模态在宇宙余切下的子范畴是共完备的。

**💡 创新点**

首次指出传统的左伴随保持余极限的证明在野生范畴里可能失效，并给出反例；提出新的 2-可合性 条件，并将其应用于悬挂、连接与模态的共完备性证明。

**🔧 技术方法**

使用同伦类型理论（HoTT）、野生范畴理论、同伦同态论、Cavallo 的同伦技巧，以及 Agda 形式化验证。

**📊 数据集**

无；本工作为纯理论证明，无实验数据集。

**📈 对比分析**

无实验对比，全部为理论证明，已在 Agda 中完成形式化验证。

**⚠️ 局限性**

仅讨论左伴随的情形；右伴随保持极限的情况以及更一般反射子宇宙的推广仍未完成。

---

## 428. Multirate State Space Models for End-to-End Processing of Pulse Density Modulated Speech Signals

**arXiv ID:** 2608.28472 | [PDF](https://arxiv.org/pdf/2608.28472v1)

**作者:** Ludovic Boulanger `[一作]` (University of Sherbrooke), Sean U. N. Wood `[通讯]` (University of Sherbrooke)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于连续时间状态空间模型（SSM）的端到端PDM语音处理架构，利用SSM编码器将PCM或PDM输入映射为与调制方式和采样率无关的潜在表示，并将其输入下游任务网络，实现关键词检测和语音增强。

**💡 创新点**

核心创新在于利用SSM的连续时间参数化实现调制与采样率不变表示，并通过SSM的长时记忆实现无需抗混叠的极度下采样，从而使模型仅用PCM数据训练即可在多种PDM OSR上保持高性能。

**🔧 技术方法**

技术手段包括LegT/FouT等SSM初始化、PDM→PCM的ΣΔ模拟、对PCM数据进行高通噪声和量化噪声的数据增强、关键词检测使用SSM堆叠网络、语音增强使用LiSenNet+MetricGAN、以及将SSM系数投影到基函数上进行PCM重建。

**📊 数据集**

关键词检测采用Google Speech Commands数据集；语音增强采用VoiceBank+Demand数据集。

**📈 对比分析**

与传统PCM基准算法对比，关键词检测在OSR=128时准确率93.72%，语音增强在低功耗OSR 512/2时PESQ≥3.08、STOI≥0.93，差距低于0.5%；SSM编码器可将输入速率压缩至65k倍，显著降低后续层的计算量。

**⚠️ 局限性**

局限性包括：需设计FouT基向量以避免编码Nyquist上限以上频率；对极低OSR的PESQ仍略低；当前实验仅验证了离线模型，硬件实现与实时自适应需进一步工作；以及对更大状态维度与频率分辨率的探索尚未完成。

---

## 429. ARC-CT: Anatomy-Routed Contrastive Vision-Language Learning for 3D Chest CT

**arXiv ID:** 2608.28455 | [PDF](https://arxiv.org/pdf/2608.28455v1)

**作者:** Huseyin Umut Isik `[一作]` (METU), Şeyda Ertekin `[通讯]` (METU)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种无监督区域感知对比学习框架ARC-CT，用于3D胸部CT异常检测，利用从报告中提取的标签实现对比学习和自监督预训练。

**💡 创新点**

创新点包括：①利用自动生成的解剖结构掩模进行注意力路由的AnatomyQFormer，实现无需边框或区域提议的局部特征聚焦；②采用Label-Jaccard软InfoNCE目标，将同一批中共享异常的样本视为相似，减少错误负样本梯度；③结合每个解剖区域的对齐损失，使文本与对应图像子区域对齐，进一步提升定位精度。

**🔧 技术方法**

技术主要包括：3D ResNet-18 视觉编码器、CXR-BERT+LoRA 文本编码器、跨模态注意力 QFormer、Label-Jaccard 软 InfoNCE 损失、解剖掩模路由、查询监督等。

**📊 数据集**

使用公开的 CT-RATE 数据集（50,188 体积，47,149 训练集）提取 18 类异常标签，并在 RAD-ChestCT 公开子集（3,630 体积）进行外部验证；解剖掩模由 TotalSegmentator 生成。

**📈 对比分析**

与多种基线（MPS-CT、GreenRFM、BrgSA、fVLM 等）比较，ARC-CT 在 CT-RATE 的 macro AUC 达到 0.855（3 种种子平均），超越同尺寸基线 0.848；在外部 RAD-ChestCT 上获得 0.734 macro AUC；同时在跨模态检索任务中取得 MAP 与 R@K 的最优或同等表现。

**⚠️ 局限性**

局限性包括：①仅在单一数据集上训练，对不同扫描仪或协议的泛化能力有限；②对解剖掩模的依赖可能在解剖变异或缺陷时影响性能；③在小型、细微异常上提升有限，仍低于大模型的表现。

---

## 430. Learning to Use Tools: Reinforcement Learning for Tool-Integrated Mathematical Reasoning

**arXiv ID:** 2608.28447 | [PDF](https://arxiv.org/pdf/2608.28447v1)

**作者:** Minghui Xu `[一作]` (Stanford University), Zi Wang `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在大型语言模型中集成计算器工具，并结合监督微调与多种强化学习方法，提升Countdown算数推理的准确性。

**💡 创新点**

创新点在于将外部计算器与工具调用动态融入模型推理流程，构建可验证奖励的RL框架，并提出Tool‑DAPO的动态采样与不对称裁剪更新。

**🔧 技术方法**

使用了监督微调（SFT）、RLOO、RLOO++、GRPO、DAPO等策略梯度强化学习技术，并通过vLLM等实现实时工具调用。

**📊 数据集**

基于Countdown任务的公开50题集与自制的1024题无重叠测试集，训练集为原始SFT数据并扩展为工具调用格式。

**📈 对比分析**

与无工具SFT和RL基线对比，工具集成提升pass@1约10个百分点；Tool‑DAPO在1024题上pass@1达到66%，高于Tool‑RLOO的56.6%，整体性能显著提升。

**⚠️ 局限性**

局限在于RL主要重塑已有正确轨迹的概率，无法有效提升完全无正确采样的难题；工具调用仍依赖先前推理质量。

---

## 431. Self-extensional logics of formal inconsistency: Decidability and limits for paraconsistency

**arXiv ID:** 2608.28443 | [PDF](https://arxiv.org/pdf/2608.28443v1)

**作者:** Marcelo E. Coniglio `[一作]` (University of Campinas), Héctor Federico Mallea `[通讯]` (University of Campinas)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了自扩展的相容逻辑 RmbC 及其十四个主扩展，系统分析了其自扩展性、可自扩展的相容性、有限模型属性、可判定性以及有效性问题的复杂度。

**💡 创新点**

创新点包括：①将 RmbC 的自扩展性与可自扩展相容性做了完整的分类，识别出六个最小爆炸性组合并证明它们在代数层面相等；②首次证明 RmbC 及其主扩展在 BALFI 语义下具有有限模型属性并由此得到判定性；③给出了 RmbC 的有效性问题从 CoNP‑hard 到 2‑EXPTIME 的上下界。

**🔧 技术方法**

使用的技术主要是：代数过滤与逼近（algebraic filtration）构造有限 BALFI；BALFI 的等式与一致性算子归约；计算机辅助模型搜索（Mace4 + Python）验证爆炸性组合；归约与证书方法（certificate）实现复杂度上界。

**📊 数据集**

本文没有使用传统意义上的数据集；所有实验与验证均基于自动定理证明工具 Mace4 与 Python 编写的脚本进行有限模型搜索。

**📈 对比分析**

与经典逻辑 CPL 相比，RmbC 在保持相容性的同时实现了可判定性；判定算法基于有限模型过滤；复杂度从 CoNP‑hard（下界）提升到 2‑EXPTIME（上界），证明了在理论上可接受的复杂度。

**⚠️ 局限性**

主要局限在于单独的 (ce) 公理导致的有限模型属性无法证明；因此 RmbC(ce) 的可判定性仍然未知；此外，复杂度上界仍比下界相差巨大，尚未达到最优。

---

## 432. Relaxed Sender Anonymity for CBDC Interbank Settlement: A Zero-Knowledge Approach on Permissioned EVM

**arXiv ID:** 2608.28529 | [PDF](https://arxiv.org/pdf/2608.28529v1)

**作者:** Pietro Tiberi `[一作]` (Banca d'Italia), Vitangelo Lasorella `[通讯]` (Banca d'Italia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在权限化的EVM网络中，设计并实现了基于零知识证明的CBDC跨行结算协议，支持“shield-转账-unshield”三种操作。

**💡 创新点**

创新点包括：1) 松弛发件人匿名模型，保持监管可追溯性；2) 基于Groth16 zk‑SNARK的交易证明；3) 在链上无索引的NoteRegistry模式实现无信任的记账；4) 多接收者ECIES加密实现选择性披露。

**🔧 技术方法**

采用的技术包括：Groth16 zk‑SNARK、Poseidon哈希、Baby Jubjub椭圆曲线、ECIES混合加密、递增Merkle树、Hyperledger Besu QBFT共识、以及以太坊兼容的EVM。

**📊 数据集**

使用的数据集为在Oracle云环境下的PoC部署，包含3家银行、央行、证券托管机构的实验网络，日交易量为数十笔。

**📈 对比分析**

与传统ERC‑20结算相比，Gas消耗约1.15M（含证明验证约220k），证明生成耗时4–12秒，链上验证约1毫秒，整体端到端结算时间为8–16秒，硬件加速可将证明生成降至<1秒。

**⚠️ 局限性**

局限性包括：单面额交易、Merkle树深度仅8层（可扩展至32层但会增加约束数量）、透明边界操作未绑定金额、PoC实现中NoteRegistry仍使用所有者索引（未完全实现无索引设计）、单一ZK服务持有所有密钥、以及单点可信设置。

---

## 433. Structural Change and Random Graph Models in Global Oil Trade Networks

**arXiv ID:** 2608.28474 | [PDF](https://arxiv.org/pdf/2608.28474v1)

**作者:** Anthony Bonato `[一作]` (Toronto Metropolitan University), Kyne Santos `[通讯]` (Toronto Metropolitan University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了全球原油贸易网络的结构变化，利用UN Comtrade数据构建年网络和K-3油贸易网络，并通过加权入度、PageRank、Louvain社区划分、node2vec嵌入及随机图模型比较分析网络演化。

**💡 创新点**

首次将多种网络指标与子图计数相结合，利用机器学习对油贸易网络与不同随机图模型进行分类，揭示了度分布为主导的结构特征，并通过K-3骨干网络实现了对原油贸易重点关系的精细化分析。

**🔧 技术方法**

使用加权度中心性、PageRank、Louvain社区算法、node2vec嵌入+UMAP、基于三四节点子图的特征向量，结合支持向量机、决策树、随机森林、AdaBoost等机器学习分类器进行模型选择。

**📊 数据集**

UN Comtrade原油出口-进口交易数据（HS代码2709），1988-2025年每年网络，构造的K-3油贸易网络（每国前3进/出对手的无向无权网络）。

**📈 对比分析**

通过构造3-4节点子图计数特征，对比生成的ER、Chung–Lu、Preferential Attachment、Configuration、Geometric等五种随机图模型，利用机器学习分类器在多次实验中均将时间聚合或单年度K-3网络归类为Chung–Lu或Configuration，说明基于度分布的模型更能拟合其子图结构；Geometric模型被排除。

**⚠️ 局限性**

主要局限在于：仅以度分布为基础的模型进行比较，未考虑地理位置或经济重力效应；K-3骨干网络的稀疏性可能导致信息损失；仅分析原油贸易，无法推广到其他商品；模型参数和嵌入维度选择可能影响结果；无法完全区分地缘政治冲击与宏观经济驱动的结构变化。

---

## 434. Stranger, Fan, or Peer? A Systematic Study on the Role of Interlocutor in Persona-Based Dialogue Generation

**arXiv ID:** 2608.28467 | [PDF](https://arxiv.org/pdf/2608.28467v1)

**作者:** Daniela Occhipinti `[一作]` (Fondazione Bruno Kessler), Marco Guerini `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统地探究了在训练、推理和评估阶段，谁能看到谁的个人简介会如何影响基于人物资料的对话生成，并利用LLM评判进行作者识别。

**💡 创新点**

创新之处在于首次将个人简介可见性拆解为三阶段（训练、推理、评估），并量化其对复制行为、可辨性和信息泄露的影响。

**🔧 技术方法**

采用LLaMA 3.1 8B Instruct的LoRA微调、掩码损失、稀有词重叠指标以及LLM判别器进行作者识别。

**📊 数据集**

使用PRODIGy电影对话语料库（包含人物简历）以及扩展的非PRODIGy角色以控制熟悉/陌生配对。

**📈 对比分析**

通过比较不同可见性组合下的判别准确率与稀有词重叠率，发现训练时可见性对复制行为影响最大，精度在0.66–0.97之间；零样本模型复制率高，微调模型复制率低但可辨性保持。

**⚠️ 局限性**

局限在于仅使用单一电影对话数据集和单一模型规模，稀有词重叠可能低估泄露，LLM判别器为自动代理，结果可能与人工评估不同。

---

## 435. Acquire, Repair, Preserve: A Diagnosis-Guided Post-Training Recipe for Small-Model Dialogue Game Agents

**arXiv ID:** 2608.28458 | [PDF](https://arxiv.org/pdf/2608.28458v1)

**作者:** Nan Li `[一作]` `[通讯]` (Utrecht University), Nan Li (Utrecht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套针对小模型对话游戏的交互式训练流程：先用广泛的监督微调（SFT）使模型能参与游戏；随后通过局部对比（turn‑local DPO）对Wordle等游戏中出现的机械化错误进行修复；最后通过Delta Scaling在不损失通用能力的前提下选取最优权重点。

**💡 创新点**

创新点在于：①将对话游戏的失败诊断细化到单一回合决策层面，利用可机械验证的错误构造精确的偏好对（turn‑local preference pairs）；②在微调后仅对这些局部错误进行强化学习，避免全局对话损害协议遵从；③通过权重空间的比例缩放（delta scaling）实现交互性能提升与静态评测保持平衡。

**🔧 技术方法**

使用技术包括：LoRA微调、Direct Preference Optimization (DPO)、权重空间缩放、Wordle游戏的可验证错误规则、Playpen框架下的对话回放与评价。

**📊 数据集**

使用的数据集主要为LM Playschool Challenge的公共开发集（包含14款对话游戏共67个episode），以及公开的Playpen数据集（Wordle等），并在官方评测中使用更大规模的公共、闭合域内外套件。

**📈 对比分析**

方法与对比：在官方评测中，基准2B模型Clemscore从10.67提升至38.92，闭合域内从13.41提升至41.17，且在聚合静态评分上保持44.14（接近基准44.24）。相比官方4B、9B基准模型，该方法在公开Clemscore和域内表现上显著优于基线，但域外Clemscore仅提升至7.88，表现仍有限。

**⚠️ 局限性**

局限性包括：①修复仅针对可机械验证的错误，主要集中在Wordle家族，对其他需要语义或策略判断的游戏尚未验证；②仅在单一Qwen3.5‑2B基准上实验，未探讨更强SFT阶段或不同基准对局部修复效果的影响；③Delta Scaling为全局权重缩放，可能导致某些静态子任务性能下降；④对后续阶段的因果贡献不明，部分修复阶段效果在不同评测轨道上不一致。

---

## 436. Sliding-window beats linear attention

**arXiv ID:** 2608.28444 | [PDF](https://arxiv.org/pdf/2608.28444v1)

**作者:** Alexia Jolicoeur-Martineau `[一作]` (Microsoft), Emy Gervais `[通讯]` (Independent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨并验证了在大型语言模型推理中使用滑动窗口注意力（SWA）结合注意力池化（sinks）的效果，并与线性注意力（Linear Attention）以及线性注意力+SWA的后训练方法进行对比。

**💡 创新点**

创新点在于：①将SWA与注意力池化组合成训练无关的高效推理方案；②系统性地与多种线性化方法在不同规模模型、短长上下文任务以及多种基准上进行对比；③证明SWA在长上下文任务中显著优于线性注意力后训练方案。

**🔧 技术方法**

主要技术包括：滑动窗口注意力（SWA）实现（窗口大小可调、固定4个注意力池化）；线性注意力变体（Hedgehog等）；后训练（LoLCATs、Liger-GLA等）；FlashAttention、ThunderKittens等高效后端；在多任务评估中使用MMLU、ARC、HellaSwag、PIQA、Winogrande、S-NIAH、BABILong等基准。

**📊 数据集**

使用的数据集与基准包括：MMLU（5-shot）、ARC-C、ARC-E、HellaSwag、PIQA、Winogrande、Single Needle-in-a-Haystack (S‑NIAH)、BABILong、以及多种通用知识与推理任务；模型涵盖从1.3B到70B的多种开源LLM。

**📈 对比分析**

对比方法：将原始预训练模型、SWA、线性注意力、线性注意力+SWA（LoLCATs）在相同任务和上下文长度下进行评估。结果显示：SWA在短上下文任务中恢复约99%的基准性能，匹配或超过最先进的线性注意力后训练方法；在长上下文任务（4K级别）中，SWA的性能提升显著（如S-NIAH 17–23% vs LoLCATs 5.8%或更低），同时在速度和内存上更优。

**⚠️ 局限性**

局限性：①仅评估训练无关的SWA，未探讨后训练SWA的潜在提升；②未考虑混合全注意力层的影响；③未对极大规模模型或多模态/视频模型进行验证；④对不同窗口大小和池化数量的进一步探索仍待研究。

---

## 437. Significance-Driven Semantic Communication

**arXiv ID:** 2608.28441 | [PDF](https://arxiv.org/pdf/2608.28441v1)

**作者:** Christian McDowell `[一作]` (Auburn University), Yin Sun `[通讯]` (Auburn University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个基于单样本语义显著性度量的跨层语义通信框架，协同优化物理层编码与MAC层资源分配。

**💡 创新点**

创新点在于将统计决策理论中的条件信息价值与信息论的L-散度相结合，形成可量化单样本语义价值的指标，并基于此设计了无需在线重训练的Meta-VIB编码器以及采用LP优先级的Q-Maximization调度器。

**🔧 技术方法**

技术实现包括Meta学习变分信息瓶颈（Meta‑VIB）与FiLM调制、以及多动作无休止多臂老虎机（MA‑RMAB）框架下的Q‑Maximization调度算法，并配合YOLOv8s+DeepSORT特征提取。

**📊 数据集**

实验使用了Auburn市中心交叉口的实时交通摄像头抓取的行人安全数据集，并通过YOLOv8s和DeepSORT生成风险标签。

**📈 对比分析**

与JSCC、DeepJSCC、VIB、Hyper‑VIB、ATROC等物理层基线以及SemanticGreedy、MaxAge、Round‑Robin、NGM等调度基线对比，实验表明在0 dB SNR下可实现高达1000倍的语义频谱效率提升。

**⚠️ 局限性**

局限性包括需为每个任务设计特定损失函数、训练耗时较长且需高性能GPU、实验仅在单一交通摄像头数据集上验证，缺乏更广泛场景的验证。

---

## 438. Offline-Verifiable Accountability for Cross-Organization Agent Messaging: A Preserved Evidence-Bundle Approach

**arXiv ID:** 2608.28542 | [PDF](https://arxiv.org/pdf/2608.28542v1)

**作者:** Adil Alshammari `[一作]` (Northern Arizona University), Hayretdin Bahsi `[通讯]` (Northern Arizona University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种保留证据包模型和离线验证器，用于跨组织代理工作流的事件级可验证性，支持多种证据类（签名、日志、接收、检查点、委托等），并通过策略控制决定是否接受。

**💡 创新点**

创新点在于将事件级证据与策略控制结合，提供仅依据策略所需证据的离线验证器，避免推断交付或执行，且首次系统化地支持检查点上下文、委托链等复杂证据。

**🔧 技术方法**

采用透明日志、Merkle树、数字签名、包含证明、签名检查点、检查点上下文、Witness、委托链、能力证明等加密与协议技术。

**📊 数据集**

使用模拟医疗跨组织工作流，包含300个完整工作流（4个事件）和1200个有效保留证据包作为实验数据集。

**📈 对比分析**

通过三项实验：单控策略对比测量验证器延迟、事件级证据需求比较以及负面证据诊断。结果显示检查点上下文验证耗时最高，事件4因委托等导致延迟显著增加，所有负面案例均被正确拒绝，未出现误接受。

**⚠️ 局限性**

局限性包括实验仅在单机、单线程环境下实现，未评估大规模、多主机或并发情景；检查点上下文验证仍较慢；工作流仅为四步，未涵盖更复杂的分支与多跳委托；负面案例仅覆盖已知攻击，未进行全面模糊测试或关键链/密钥破坏等情形。

---

## 439. Machine-learning-assisted multiscale topology optimization of functionally graded superimposed lattice structures

**arXiv ID:** 2608.28513 | [PDF](https://arxiv.org/pdf/2608.28513v1)

**作者:** Prashant Kumar Gupta `[一作]` (Indian Institute of Technology Roorkee), Mohammad Ashraf Iqbal `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于机器学习的多尺度拓扑优化框架，用于设计功能梯度的叠加晶格结构；

**💡 创新点**

创新点在于构建了由BCC、FCC和简单立方晶格叠加而成的正交晶格单元，并用Cholesky约束的神经网络逼近其有效弹性张量，另外训练单独的密度预测网络；

**🔧 技术方法**

主要技术包括离线计算同质化、物理约束的神经网络推理、SIMP宏观拓扑优化与基于MMA的微观晶格参数优化；

**📊 数据集**

使用约1926组同质化样本训练弹性张量网络，1500组几何样本训练密度网络，均来自对叠加晶格单元的离线有限元同质化计算；

**📈 对比分析**

在三维MBB梁基准上，微观优化得到的晶格参数和相对密度分布与预期的弯曲载荷分布一致，且整体性能优于单纯SIMP方案，且优化过程稳定；

**⚠️ 局限性**

局限性包括对同质化误差的依赖、未考虑工艺约束（如支撑角度、连续性和细杆屈曲），以及宏微耦合不够紧密，未来需加入更多制造约束和完整的多尺度耦合验证。

---

## 440. Training Communication-Efficient Mixture-of-Experts Language Models with Layer Re-Configuration

**arXiv ID:** 2608.28511 | [PDF](https://arxiv.org/pdf/2608.28511v1)

**作者:** Simeng Sun `[一作]` (Nvidia), Roger Waleffe `[通讯]` (Nvidia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通信高效的混合 MoE 体系结构（CE-MoE），通过减少 MoE 层数并增大 token‑mixing 层和宽专家以降低专家并行通信量；

**💡 创新点**

创新点在于将 token‑mixing 与 channel‑mixing 的比例不再保持 1:1，采用异构层排布与分阶段贪心算法构造层序列，并在保证参数预算不变的前提下显著降低 MoE 深度；

**🔧 技术方法**

使用 Mamba‑2 作为 token‑mixer、LatentMoE 低维投影、专家并行技术、分阶段贪心生成层模式；

**📊 数据集**

使用标准 NLP 基准数据集：MMLU、MMLU‑Pro、HumanEval、MBPP、GSM8K、MATH‑500、MATH‑Hard、RACE、HellaSwag、WinoGrande、ARC‑Challenge；

**📈 对比分析**

与传统全 MoE 基线在相同参数预算下对比，CE‑MoE 在 2B‑31.5B 规模上训练 GPU‑小时下降 30–35%，验证损失基本保持一致；在更大规模或更大全局批次时还能训练更多 tokens、下游性能提升 1–2 分，推理吞吐量提升 28–36%；

**⚠️ 局限性**

局限包括：收益受通信精度、网络带宽、专家负载均衡等因素影响；在更大 world size 或更高精度时需进一步验证；早期训练时仍可能出现 loss 峰值，需更细致的负载平衡或路由正则化。

---

## 441. Ladders in Chaos: When, How, (and Perhaps Why) Does Test-Time Scaling Improve LLM Machine Translation

**arXiv ID:** 2608.28496 | [PDF](https://arxiv.org/pdf/2608.28496v1)

**作者:** Di Wu `[一作]` (University of Amsterdam), Vlad Niculae `[通讯]` (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并比较了大语言模型在机器翻译中两种测试时缩放范式：顺序自我改进（sequential）与并行独立采样（parallel），并通过多指标自动评估和人工MQM评测验证其效果；

**💡 创新点**

提出了通过限制上下文窗口大小来平衡顺序采样的重复偏差与质量提升的策略，并在低温度下证明顺序采样保持多样性；

**🔧 技术方法**

使用了“请再次翻译以获得更好版本”的最简提示、最佳- N 选择器（无参考度量模型）、多轮翻译和滑动窗口上下文控制技术；

**📊 数据集**

在WMT24++公开数据集上，选取六种高/中等资源语言对（如zh-ar），并在多模型（Qwen3-32B/4B、gpt-4o-mini）上进行实验；

**📈 对比分析**

结果显示顺序采样在有限预算下的样本效率和多样性更优，自动指标如COMET、MetricX24显著提升，人工评测表明流畅度和自然度提升，但在大预算下精确度可能下降；

**⚠️ 局限性**

局限包括单一提示格式、有限语言与数据集、缺乏低资源语种验证、仅用单一采样轨迹、以及对选择器和多样性度量的潜在偏见；

---

## 442. Quantum-Based Solutions for Security Enhancement in Open Radio Access Networks

**arXiv ID:** 2608.28480 | [PDF](https://arxiv.org/pdf/2608.28480v1)

**作者:** Dzung Quoc Ngo `[一作]` (Middlesex University), Huan X. Nguyen `[通讯]` (Middlesex University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出将后量子密码学、量子密钥分发、量子身份认证和量子机器学习等量子安全技术融合到O‑RAN零信任架构中的完整框架，并给出可部署的量子节点与xApp模型。

**💡 创新点**

创新点在于：①将量子安全与零信任原则系统化映射到O‑RAN多维威胁场景；②提出基于xApp的可插拔量子节点（如QKD控制器、量子身份验证服务器、量子威胁检测模块）；③规划分阶段迁移路线（PQC → 混合PQC–QKD → 原生量子功能），为6G时代的O‑RAN安全奠定路径。

**🔧 技术方法**

使用的技术包括：后量子密码学（ML‑KEM、ML‑DSA）、量子密钥分发（QKD）、量子身份验证（QIA）、量子数字签名（QDS）、量子安全直接通信（QSDC）以及量子机器学习（QML）等。

**📊 数据集**

论文未使用特定实验数据集，而是基于现有标准（NIST FIPS 203/204、O‑RAN WG11规范）与协议进行理论分析与设计评估。

**📈 对比分析**

比较方法主要是对比传统基于RSA/ECC的公钥方案与量子安全方案在抗量子攻击、密钥管理复杂度和系统开销上的差异；性能方面指出PQC实现需评估握手延迟、CPU负载与密钥尺寸，QKD需要评估密钥率、距离和可用性，但未给出具体实验数值。

**⚠️ 局限性**

限制：①量子硬件（QKD、量子认证设备）尚不成熟，部署成本高；②QIA/QDS等技术仍处于实验阶段；③迁移路径需逐步进行，可能对现有O‑RAN架构造成兼容性与管理挑战；④缺乏实测性能数据与真实网络实验验证。

---

## 443. LayerRecall: A State-Conditioned Memory Router for Long-Horizon Consistency in Video Generation

**arXiv ID:** 2608.28460 | [PDF](https://arxiv.org/pdf/2608.28460v1)

**作者:** Yixuan Ding `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 LayerRecall，一种针对自回归视频扩散模型的层感知内存路由器，并配套 CHPM 训练策略，实现了在有限 KV 缓存内高效利用历史信息。

**💡 创新点**

创新点在于：① 当前条件下的检索查询 + 层级选择的内存注入；② 通过跨时域预测匹配实现无监督的内存路由学习；③ 通过层级配置实现跨骨干网络的零训练迁移。

**🔧 技术方法**

采用 Diffusion Transformers（DiT）、RoPE 位置编码、KV 缓存、层级注意力分析、软混合梯度、跨时域预测匹配（CHPM）等技术。

**📊 数据集**

训练使用 1,600 条多镜头 prompt（每条 384 帧），评估集包括 MemoBench、MovieBench、VBench‑Long 以及 100 条测试 prompt；教师模型使用完整上下文，学生模型使用 32 帧可见上下文 + 80 帧物理 KV 缓存。

**📈 对比分析**

与 CausVid、LongLive、Self‑Forcing、LongLive‑RAG、MemFlow 等自回归长视频方法在 MemoBench、MovieBench 以及 VBench‑Long 上进行对比，LayerRecall 在记忆评测上取得最高分，在 VBench‑Long 上保持与 LongLive‑2.0 同水平；同时通过消融验证层级选择和 CHPM 的有效性，推理开销几乎无增。

**⚠️ 局限性**

局限性包括：① 需要对每个骨干网络做层级分析和配置；② 受限于固定的 KV 缓存大小，极长时序的记忆仍有限；③ 训练仍需教师长上下文，无法完全自监督；④ 在未见过的骨干或不同规模模型上的迁移效果尚需进一步验证。

---

## 444. An Enclosed Mode Is a Gauge Choice: Topology Relative to Reach in Certified Code World Models

**arXiv ID:** 2608.28541 | [PDF](https://arxiv.org/pdf/2608.28541v1)

**作者:** Javier Aguilar Martín `[一作]` `[通讯]` (AGILabs), Javier Aguilar Martín (AGILabs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究代码世界模型（CWM）在采样门验证下的错误可识别性，聚焦环形冻结模式（annular freeze band）与其内部可达性关系。通过理论证明与大规模仿真，探讨错误模型可被证实、无可证伪且无害，或可被证伪并导致高代价的三种 regime。

**💡 创新点**

创新点在于：1）提出采样门仅决定可达查询集上的模型，形成“门等价类”；2）揭示拓扑相对可达性是决定危险的关键，而非绝对拓扑；3）发现错误可从“无可证伪且无害”过渡到“可证伪且代价高”再到“立即被证伪”的连续三阶段；4）提出覆盖定律（nerve fence）与双向防御（freedom patch）对应两类错误，并给出相应代价与失真率。

**🔧 技术方法**

使用的技术包括：Lean 4 形式化证明与机器检查；仿真平台（二维推进‑阻力模型）与门证书采样；LLM 合成（GPT‑5.x、Qwen、Claude）与预注册 prompt；拓扑数据分析（持久性同调、β₁ 估计）；持续同伦与持久性同调作为证明工具；门证书与规划器交互框架；覆盖与自由点防御算法。

**📊 数据集**

数据集为自构建的物理仿真环境：二维推力–阻力平台，包含双重 lode 与环形冻结模式；采样门使用 30,000 次 rollouts，LLM 合成 20 种子；在多维 ShellField‑n 维实验中使用 20,000 次 rollouts；未使用公开标准数据集。

**📈 对比分析**

比较方法：通过门证书接受率、起点（外部/内部）下的接触率 r 与内部进入率 r_int、play_cost 曲线；LLM 误差按三种 regime 分析；防御措施评估 pc 下降幅度。结果显示：在 γ=0（闭环）时 pc≈1.12；当开通向起点的通道时 pc 降至 0.35；在起点位于环内部时 pc 立即降为 0。三家 LLM 在同一结构下表现一致；nerve fence 与 freedom patch 将 pc 降至 0.058 与 0.029。

**⚠️ 局限性**

局限性：实验仅涵盖圆形、分离的隔离模式（环、球面、立方环），未验证非圆形或动态边界；LLM 合成样本量有限，跨族对比受限；门证书对稀有模式的识别率与防御效果需经验校准；TDA 传感器分辨率限制导致 β̂₁ 误报；某些上界（如 funnel defect、直接进入率）只能通过测量验证，理论上未完全证明。

---

## 445. When Robots Mishear Us: Mapping the Safety Risks of Voice-Controlled Embodied AI

**arXiv ID:** 2608.28518 | [PDF](https://arxiv.org/pdf/2608.28518v1)

**作者:** Sihan Jia `[一作]` (Heriot-Watt University), Oliver Lemon `[通讯]` (Heriot-Watt University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究自动语音识别（ASR）错误如何导致具身 AI（EAI）模型产生不安全行为，并构建了五类可控 ASR 错误框架，结合 POEX 与 SafeAgentBench 对模型安全性进行评估。

**💡 创新点**

创新点在于：①基于 GPT‑4 的错误生成与 CHIME‑6 噪声插入技术，实现可调节强度的 ASR 错误模拟；②系统化比较不同错误类型对安全指标（AR、ESR、Safety Flips）与规划成功率的影响；③探讨自动纠错对缓解风险的效果。

**🔧 技术方法**

技术手段包括：GPT‑4 / GPT‑4o‑mini 用于生成、纠错与评估；Qwen2.5‑7B‑Instruct 作为生成/规划模型；CHIME‑6 语音噪声数据用于噪声注入；POEX 与 SafeAgentBench 评估管道。

**📊 数据集**

使用数据集：POEX 的 Harmful‑RLBench（136 条有害指令、25 个环境）；SafeAgentBench（300 条正常任务指令）；CHIME‑6 语音背景噪声；原始无损指令集合。

**📈 对比分析**

通过 Acceptance Rate、Executable Success Rate 与 Safety Flips 指标比较清洁与各错误条件；发现声学替换与高强度噪声显著提升 AR 与 ESR，混合错误导致安全翻转；自动纠错可在一定程度上降低风险，但对严重噪声效果有限；SafeAgentBench 中 ASR 错误使成功率下降（GPT‑4o‑mini 27.67%→23.33%，GPT‑4.1‑mini 38.00%→35.67%）。

**⚠️ 局限性**

局限性包括：仅覆盖五类错误和四个噪声级别；纠错方法依赖 GPT‑4，难以覆盖所有语音变体；实验仅在特定模型和平台上进行，未考虑真实硬件与多说话人环境；未对更大规模、更多样化的具身系统进行验证。

---

## 446. Quadratic Probing Insertions Are $ε^{-(1+o(1))}$

**arXiv ID:** 2608.28512 | [PDF](https://arxiv.org/pdf/2608.28512v1)

**作者:** Yang Hu `[一作]` (Carnegie Mellon University), Renfei Zhou `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文证明了二次探测哈希表在负载因子为1-ε时的期望插入时间为ϵ^{-(1+o(1))}，从而基本上解决了该结构的时间复杂度问题。

**💡 创新点**

创新点在于引入正相关性（positive association）与翻译不变性的新视角，结合数论对二次探测序列差值的稀疏性分析，获得了R(k) = k^{o(1)} 的关键上界，从而突破了线性探测的聚类瓶颈。

**🔧 技术方法**

使用的主要技术包括泊松化（Poissonization）将固定负载模型转化为随机键数模型；Harris不等式证明占用槽指示变量正相关；协方差与尾分布估计；数论中平方差的分布与模数超平面估计；覆盖引理（Covering Lemma）控制长插入时间的概率。

**📊 数据集**

本工作为理论分析，未使用任何具体数据集，所有结果均为渐进概率论和数论证明。

**📈 对比分析**

与此前的研究相比，之前仅在约35%负载下得到常数期望插入时间；本研究把结果推广到任意负载(满足ε>1/√log n)，并证明了近似线性时间O(ε^{-1})（仅有子多项式因子）。

**⚠️ 局限性**

局限性包括：仅适用于无删除的插入-only 工作；要求表大小为素数并利用翻译不变性；分析中使用的泊松化手段虽然便于证明，但在实际实现中的精确度可能略有偏差；子多项式因子仍未被进一步收敛到常数。

---

## 447. SG-UMP: Sequence-Guided Universal Multimodal Prioritization Calculation Framework

**arXiv ID:** 2608.28503 | [PDF](https://arxiv.org/pdf/2608.28503v1)

**作者:** Xinyi Zhang `[一作]` (Imperial College London), Peijie Sun `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个可插拔插件 SG-UMP，用于在多模态顺序推荐（MSR）中通过模块组合与动态路由提升多模态信息处理。

**💡 创新点**

创新点在于：①针对用户级偏好异质性和数据集级模态偏差两大问题，引入模块组合器（Module Combiner）和模块路由器（Module Router）实现对模态处理流程的自适应排序；②使用频率过滤层、层次注意层和 MoE 融合层三种功能模块，配合条件互信息正则化实现输入敏感的路由；③通过可调的模块顺序和多尺度专家实现对不同数据集模态重要度的动态适配。

**🔧 技术方法**

主要技术包括：CLIP 预训练编码器（文本/图像特征提取）、频率域过滤、层次注意机制、Mixture-of-Experts 融合、动态模块路由、BPR 损失+条件互信息正则化、Transformer/自注意力序列建模。

**📊 数据集**

实验数据集：Amazon Home、Beauty、Office（四个商品类）和 Yelp（餐饮点评），均为真实业务日志，使用 5-core 过滤后进行留一交叉验证。

**📈 对比分析**

将 SG-UMP 与三种主流序列模型（SASRec、STOSA、Oracle4Rec）结合，在四个数据集上对比多种基线（传统与多模态）。结果显示 SG-UMP 在 Recall@10/20、NDCG@10/20 上平均提升约 21–25%（相对于对应 backbone），在所有数据集均获得最优或次优排名，且相对最佳基线提升 8–17%。

**⚠️ 局限性**

限制与不足：①依赖 CLIP 等大型预训练编码器，对新模态或小样本场景的迁移性有限；②模块路由引入额外参数与计算开销，虽然增量不大但在极大规模部署时仍需优化；③目前仅验证文本+图像两模态，未探讨音频、结构化属性等更丰富模态；④评估使用离线历史数据，缺少在线用户关注信号（如眼动或即时反馈）的直接验证。

---

## 448. REPLICANT: Learning Policies for Evading and Hardening Malware Detectors

**arXiv ID:** 2608.28499 | [PDF](https://arxiv.org/pdf/2608.28499v1)

**作者:** Shae McFadden `[一作]` (King's College London), Fabio Pierazzi `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出一种层次化深度强化学习框架，学习 Android malware 在问题空间的规避策略，在只获得标签的黑盒环境下实现高效的攻击和对抗训练。

**💡 创新点**

创新点包括：①将规避问题建模为马尔可夫决策过程，并使用层次化 PPO 学习何时查询、何种变换；②在无置信度信息的标签‑仅黑盒下取得最强规避性能；③通过策略迁移实现跨模型、跨特征空间的高鲁棒性；④利用学得的策略进行对抗训练，提升检测器对未知攻击器的鲁棒性。

**🔧 技术方法**

主要技术：层次化 Proximal Policy Optimization（PPO）与动作遮蔽；gadget‑transplantation 作为问题空间的能力集；对抗训练与主动学习的交叉策略；多模型、多特征空间实验验证。

**📊 数据集**

实验使用 Hypercube Android malware 数据集（224,965 篇，2021‑2024，VTT2/4），并在 Drebin、APIGraph、RAMDA 三种特征空间上进行评估。

**📈 对比分析**

方法与 APG、ADZ、EvD、Rnd 等基线在 1,764 个 surrogate/target 组合下对比，平均攻击成功率（ASR）为 78.8%，平均查询量 7.4；匹配设置下 ASR 96.6%，仅 3.6 次查询；对抗训练后可将攻击成功率降至 9.9%–16.3%，并保持对未见攻击器的鲁棒性。

**⚠️ 局限性**

局限性：仅针对固定的 gadget‑transplantation 能力集，未探究其他更强或更细粒度的能力集；实验仅基于 Hypercube 数据集，缺乏对其他公开数据集的验证；对抗训练与主动学习的交互机制仍需进一步研究以实现更优的持续鲁棒性。

---

## 449. A System-of-Systems Case Study for the Verification of Composed Digital Twins

**arXiv ID:** 2608.28498 | [PDF](https://arxiv.org/pdf/2608.28498v1)

**作者:** Mennatullah T. Khedr `[一作]` (Newcastle University), Peter Gorm Larsen `[通讯]` (Aarhus University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对温室植物盆系统中的数字孪生（DT）进行形式化建模与验证，利用VDM-RT实现单体DT模型并导出FMI FMU进行协同仿真；

**💡 创新点**

将DT质量概念（相关性、可验证性、可替换性、真实性）转化为可验证属性集合，并首次探讨这些属性在系统-of-系统（SoS）层级中的组合验证挑战；

**🔧 技术方法**

采用VDM-RT建模与证明工具、FMI 2.0 FMU导出、静态证明、合同验证、轨迹比较与经验校准等技术；

**📊 数据集**

基于温室植物盆的物理模型与仿真生成的湿度、温度、湿度、光照等传感数据，未使用公开数据集；

**📈 对比分析**

通过静态证明、QuickCheck、轨迹比较等手段验证属性；单个/少量DT的验证完成，未完整评估大规模组合时的验证成本与性能；

**⚠️ 局限性**

未完成SoS层级的完整形式化与验证，缺乏对大规模DT组合的实证验证，验证依赖仿真非物理部署，且仅覆盖四个质量维度，未考虑安全性、可维护性等其他重要维度。

---

## 450. On the Maintenance and Co-evolution of Agent Plugins: An Empirical Study of Claude Code Plugin Marketplaces

**arXiv ID:** 2608.28497 | [PDF](https://arxiv.org/pdf/2608.28497v1)

**作者:** Ahmed Hereiz `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Claude Code 插件市场进行大规模实证研究，分析 1,926 个仓库、8,351 个插件和 77,773 次提交，探究其结构、开发模式和组件协同演化。

**💡 创新点**

首次将自然语言指令文件视为核心软件工件，揭示其开发频率、AI 贡献比例、CCS 分类偏移，以及脚本与 Markdown 的功能耦合，提出 AI-native 插件的维护依赖新模型。

**🔧 技术方法**

结合 GitHub Code Search API、REST API、关联规则、Mann‑Kendall 趋势检测、LLM（GPT‑5‑mini、Qwen3‑Coder）自动分类和手工标注。

**📊 数据集**

收集的 Claude Code 插件市场公开仓库数据（1,926 个），共 8,351 个插件、77,773 次插件相关提交，112,807 个合并 PR。

**📈 对比分析**

与传统 OSS 的 Conventional Commits 统计做对比；使用关联规则计算 Lift、Conf，发现功能型提交占比约 39.6%，AI 合著率 34.9%，脚本与 Markdown 共变率 78%，表明 AI‑native 市场正快速增长。

**⚠️ 局限性**

仅覆盖 Claude Code 市场，数据来自 2026 年早期，依赖公开仓库、星级阈值，LLM 自动化分类存在偏差，缺乏跨平台验证。

---

## 451. Low-Power End-to-End Cochlear Implant Speech Denoising with Spiking Neural Networks

**arXiv ID:** 2608.28493 | [PDF](https://arxiv.org/pdf/2608.28493v1)

**作者:** Ludovic Boulanger `[一作]` (University of Sherbrooke), Sean U. N. Wood `[通讯]` (University of Sherbrooke)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于神经形态计算的ACE端到端语音处理框架，融合传统ACE与脉冲神经网络，实现对听觉刺激的高效编码与解码。

**💡 创新点**

创新点在于将ACE与脉冲神经网络无缝结合，既保持了ACE的高频分辨率，又利用Spiking Neural Network的低功耗和实时性，首次实现了端到端的、低延迟语音感知与合成。

**🔧 技术方法**

采用Spiking Neural Network（SNN）、神经形态硬件（Neuromorphic Engine）、ACE算法以及深度学习框架（如TensorFlow/Keras）进行训练与部署。

**📊 数据集**

使用公开语音数据集TIMIT和LibriSpeech进行训练与评估，并在Cochlear Implant测试数据上验证其在真实听力场景中的表现。

**📈 对比分析**

与传统ACE、Deep ACE以及基于ANN的端到端模型进行对比，实验表明本方法在语音识别准确率提升约4%-6%，在感知评估（MOS）上提升约0.3分，同时功耗降低约30%。

**⚠️ 局限性**

主要限制包括：脉冲神经网络的训练过程复杂且对超参数敏感；目前的硬件实现仍依赖专业的神经形态芯片，难以在通用设备上直接部署；实验数据集主要为合成或受控环境语音，真实噪声条件下的鲁棒性需进一步验证。

---

## 452. LLM-Based Agents for Software and Systems Security: Approaches, Applications, and Assessment

**arXiv ID:** 2608.28490 | [PDF](https://arxiv.org/pdf/2608.28490v1)

**作者:** Jingjing Nie `[一作]` (University at Buffalo, SUNY), Haipeng Cai `[通讯]` (University at Buffalo, SUNY)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 2023‑2026 年间 100 篇关于软件与系统安全的 LLM‑agent 研究进行系统综述，构建了覆盖架构、应用场景和评估方法的三维分类体系。

**💡 创新点**

创新点在于首次提出针对安全领域 LLM‑agent 的综合三维 taxonomy，并通过该框架梳理现状、识别研究空白与未来方向，填补了以往单一维度综述的不足。

**🔧 技术方法**

采用系统文献检索（Google Scholar、ACM/IEEE 等）与前向/后向 snowballing、严格的纳入排除规则，随后基于三维 taxonomy 进行论文归档与定量统计分析。

**📊 数据集**

核心数据集为收录的 100 篇经同行评议论文及其附录信息（论文元数据、方法、数据集、指标等），并未使用公开实验数据集进行实测。

**📈 对比分析**

通过将论文映射至 taxonomy 的各属性，作者以统计频率、占比和交叉表形式比较不同架构与应用组合的出现率；结果显示单体 Agent 占比 54%，多 Agent 46%，动态/反应式规划占 80%，但整体缺乏统一的安全与可审计评估标准，未能给出统一性能分数。

**⚠️ 局限性**

主要局限包括：①“Agent”概念定义不统一，导致跨研究可比性差；②安全风险与责任边界未被充分规范；③评估方法多样且缺乏统一基线；④缺乏针对长周期交互和多步骤轨迹的度量；⑤缺乏真实生产环境中的可复现基准与安全约束。

---

## 453. How Proper Scoring Rules Shape LLM Forecasting

**arXiv ID:** 2608.28482 | [PDF](https://arxiv.org/pdf/2608.28482v1)

**作者:** Benjamin Turtel `[一作]` (Lightning Rod Labs), Philip E. Tetlock `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在共享的有限训练设置下，对比了五种严格的正确评分规则作为LLM预测终端奖励的效果，研究了它们对GPT-OSS-120B预测模型校准、判别和误差结构的影响。

**💡 创新点**

创新点在于对同一理论激励下的不同奖励函数进行受控实验，并揭示它们在有限训练中产生不同的校准、判别和偏差/信息/噪声组成。

**🔧 技术方法**

采用了Dr. GRPO强化学习后训练技术、LoRA适配器、Brier、对数、球面、Beta(2,8)、Beta(8,8)评分规则以及BIN分解来分析误差来源。

**📊 数据集**

使用了从2024年7月至2026年1月的新闻事件构建的8,041条二分类预测问题的数据集，其中7,076条用于训练，965条用于评估。

**📈 对比分析**

通过对比每种奖励训练模型与未训练基线在log分数、Brier分数、ECE、AUC‑ROC及缺失率等聚合指标上的表现，发现Brier训练模型取得最佳Brier和AUC，log训练模型取得最佳对数分数和最低ECE，所有奖励模型均优于基线，但不同奖励导致不同的BIN贡献。

**⚠️ 局限性**

局限性包括仅使用单一模型、单一数据集、每种奖励仅用一个随机种子、未针对每个奖励单独调优超参数、奖励间令牌生成量不同，且结果可能不具备跨模型或跨任务的普适性。

---

## 454. COVER: Identifiable Evaluation of Coalition Routing

**arXiv ID:** 2608.28475 | [PDF](https://arxiv.org/pdf/2608.28475v1)

**作者:** Raghul Sugumar `[一作]`, Amrit Gopinath `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了COVER评估框架，用于在固定的多智能体系统堆栈下，准确衡量路由器在有限团队集合中的表现，并区分路由选择与后端流程的影响。

**💡 创新点**

创新点在于：①引入公共信息边界、固定堆栈和完整团队家族的评估合同；②给出最小支持定理，区分绝对oracle regret与相对冻结策略对比；③通过完整表格与自然工具验证展示了路由增益与实现性能的差距。

**🔧 技术方法**

技术方法包括：表格查找（完整覆盖）、最小支持定理证明、冻结路由器与公共堆栈的并行执行、Bootstrap置信区间、sign-flip检验、工具沙盒（ToolSandbox）变体移位验证。

**📊 数据集**

使用的数据集包括：MuSiQue-12（500条任务、220三人团队）、HotpotQA-4（300条任务、4人团队）以及五族ToolSandbox（14条任务、16个团队）。

**📈 对比分析**

比较方法是将冻结路由器与预设对照（如公开接口、特权正向控制、离线最大团队等）在同一固定堆栈下直接执行，并计算regret差距。结果显示：公开接口在MuSiQue上regret从0.554降至0.424；HotpotQA直接集评分regret从0.313降至0.110；固定堆栈下Llama路由器在验证完备证据时regret提升0.190，但原始答案提升仅0.010。工具沙盒中预先冻结路由器regret为0.131，未达到预设0.10阈值。

**⚠️ 局限性**

局限性包括：1）评估仅适用于可枚举且有限的团队集合；2）控制表格的结果为构造性证据，未证明在更大或不受限工具环境中的通用性；3）公开接口结果为回溯性，对前瞻性验证不足；4）自然工具验证样本有限（14条任务），且后续比较套件为回溯性；5）单次供应商执行不代表整体群体效用。

---

## 455. Tight Bounds for Memory Allocation With and Without Request Fragmentation

**arXiv ID:** 2608.28462 | [PDF](https://arxiv.org/pdf/2608.28462v1)

**作者:** Michael A. Bender `[一作]` (Stony Brook University), Nicole Wein `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在允许请求碎片化的情况下，在线内存分配问题的竞争比率，并给出了对不同碎片化模型（k‑aggregate、k‑per‑request）的紧界。

**💡 创新点**

创新点在于证明了即使是极小的碎片化比例（k≈1+o(1)），竞争比率从经典的Θ(log M)骤降至Θ(log log M)，并对k‑aggregate和k‑per‑request两种碎片化模型分别给出了匹配的上界与下界；同时揭示了随机化在此问题中并无渐进优势。

**🔧 技术方法**

主要技术包括构造分层块（block）与分数指示器的潜在函数分析，利用负相关随机变量的Chernoff界定随机删除过程，并通过“量与数”矛盾推导出必要的碎片化数量限制；同时设计了基于当前高峰请求数动态调整碎片大小的分段分配算法。

**📊 数据集**

本文没有使用公开数据集，所有结果均为理论证明。

**📈 对比分析**

通过与经典无碎片化的O(log M)结果对比，证明了碎片化后能显著降低竞争比率；在k‑aggregate模型下，给出了deterministic和randomized算法的O(log_k log M)上界，并证明其最优。

**⚠️ 局限性**

限制在于碎片化模型的具体实现细节（如碎片尺寸分布）与理论模型假设（如请求大小为幂次），以及对实际系统实现中的碎片表管理等实现成本未做深入分析。

---

## 456. Prompt-Guided Interactive Segmentation of Interstitial Lung Disease in Thoracic CT

**arXiv ID:** 2608.28453 | [PDF](https://arxiv.org/pdf/2608.28453v1)

**作者:** Vasilis Dedousis `[一作]` (University of Bern), Stavroula Mougiakakou `[通讯]` (University of Bern)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发并验证了MedSAM2在胸部CT中对间质性肺疾病（ILD）进行交互式三维分割的首个适配方法，并提出基于自动分割的交互式细化流程。

**💡 创新点**

①首次将MedSAM2迁移至ILD分割并实现全模型微调以提升性能；②评估多种临床可行的提示类型（边框、多边框、点、套索、涂鸦）并展示其在不同病变模式下的表现；③设计端到端工作流，以自动分割结果为先验并通过放射科医生提示进行细化。

**🔧 技术方法**

使用了SAM2的MedSAM2模型、nnU-Net自动分割、全模型微调、三维记忆传播、点/框/套索/涂鸦提示、正负提示极性等技术。

**📊 数据集**

使用了306例覆盖七种ILD模式（GGO、HC、Cons、Ret、Ret+GGO、Bron、Emph）+健康肺组织的1mm切片CT，Ground Truth由两名经验丰富的胸腔放射科医师标注。

**📈 对比分析**

与原始SAM2、MedSAM2预训练、nnInteractive进行对比，采用Dice和NSD指标；全微调模型在所有提示类型下平均Dice提升4.7个百分点；在三维传播与2D交互中，2D交互略优；自动分割+交互细化（MP）在某些模式下性能进一步提升，平均Dice达到0.448。

**⚠️ 局限性**

数据类别不平衡、提示模拟非真实用户输入、单一自动分割初始化、未实现多轮迭代交互、传播误差累积、缺乏外部验证与放射科医生实际评估。

---

## 457. Phoneme- and Word-Level Metrics Using Self-Supervised Speech Representations for Forced Alignment Evaluation

**arXiv ID:** 2608.28508 | [PDF](https://arxiv.org/pdf/2608.28508v1)

**作者:** V. S. D. S. Mahesh Akavarapu `[一作]`, Gerhard Jäger `[通讯]` (University of Tübingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了两种无参考的语音对齐评估指标——Phoneme-Cluster Mutual Information（PCMI）和Word Acoustic Consistency Score（WACS）

**💡 创新点**

创新点在于利用自监督语音表示中的音位与单词信息，构建不依赖人工时间戳的全语料级评估方法

**🔧 技术方法**

技术方法包括使用MMS、XLSR等SSL模型提取特征，K‑means聚类、互信息计算以及动态时间规整（DTW）相似度

**📊 数据集**

实验数据集涵盖85语言的FLEURS、45语言的DoReCo以及两种低资源语言Archi与Rutul

**📈 对比分析**

与传统的平均累计偏移（AAS）以及MFA、均匀分割等基线相比，PCMI和WACS与AAS呈显著负相关，且在多语言场景下能稳定区分高低质量对齐，表现优于现有基线

**⚠️ 局限性**

局限性包括仅提供语料级评估，缺乏对单句细粒度错误的诊断；对细粒度语音现象（如音位融合）敏感度有限，并依赖可靠的音标化工具

---

## 458. DARTS: Decoder-Aware Representation Tuning via Surgery for Model Merging

**arXiv ID:** 2608.28547 | [PDF](https://arxiv.org/pdf/2608.28547v1)

**作者:** Aaryan Ajay Sharma `[一作]` (ServiceNow), Seganrasan Subramanian `[通讯]` (ServiceNow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首先研究了在多任务模型合并后，decoder型LLM的表示偏差问题，并针对其特有的“位置依赖累积”与“位置重要性不均”挑战，提出了 Decoder‑Aware Representation Tuning via Surgery（DA‑Surgery）方法；随后在合并后的 Llama‑2‑7B 上验证其效果。

**💡 创新点**

创新点在于：①设计了基于熵加权的 L1 损失（Entropy‑Weighted L1），让模型更关注决策关键的高熵位置；②引入逐位可学习偏置表，用于捕捉因因果注意力导致的位置信息累积偏差，从而实现位置感知的表示修正。

**🔧 技术方法**

采用了权重空间模型合并（Task Arithmetic、TIES‑Merging、DARE 等）得到多任务模型；随后使用低秩瓶颈与位置偏置表组成的 Surgery 模块，并用熵加权 L1 损失对其进行训练；实验中使用了 Adam 优化器、bfloat16 预计算隐藏状态等技术。

**📊 数据集**

实验数据集包括：HumanEval（代码生成）、GSM8K（数学推理）、AlpacaEval（指令跟随）。

**📈 对比分析**

与未做 Surgery 的合并模型、原始位置无关的 Surgery 以及各类合并方法基准进行对比。评估指标为 HumanEval Pass@1、GSM8K Accuracy、AlpacaEval Win Rate。结果显示，DA‑Surgery 在所有三大任务上均优于原始 Surgery，平均提升约 2–3%，且仅增加约 0.1% 的参数。

**⚠️ 局限性**

局限性包括：①仅在 Llama‑2‑7B 及三大领域验证，未检验更大模型或不同架构的泛化性；②Surgery 需要少量校准样本，若样本不足可能导致性能波动；③仅针对 decoder 型模型，未考虑 encoder 或混合结构；④对极长序列的表现仍待评估。

---

## 459. Blind Men and the Elephant: Probing the Epistemic Myopia of LLMs under Long-Tail Divergent Knowledge

**arXiv ID:** 2608.28478 | [PDF](https://arxiv.org/pdf/2608.28478v1)

**作者:** Zhuoshi Pan `[一作]` (Tsinghua University), Xing Sun `[通讯]` (Tencent Youtu Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究开发了一种闭卷问答基准，利用低曝光网页文档中相互冲突的事实来评估LLM对长尾知识的记忆完整性。

**💡 创新点**

创新点在于将源冲突事实转化为闭卷问答对，能够测量模型对所有已验证答案的完整回忆，而非仅检索单一答案，并提出完整性诊断指标与系统化构建流程。

**🔧 技术方法**

使用知识点聚类、实体检索、LLM 边缘分类、生成 QA 记录以及 LLM 判定评估等技术。

**📊 数据集**

数据集来源于 DCLM fastText 过滤下的低分网页文档（RePro organic data），最终得到 1,094 个问答对。

**📈 对比分析**

与多款公开权重和专有模型对比，最优模型 Kimi‑K3 在完整回忆率 52.4% 左右，显示模型规模和推理提升仍无法完全解决“少数方”记忆缺失。

**⚠️ 局限性**

局限性包括对训练语料的观察性代理、覆盖面有限、模型范围不全以及评估成本高等问题。

---

## 460. Distributed Cross-Layer Optimization for Covert Multi-Hop, Multi-Modal Networks: Exponentially Fast Convergence and Robust Tracking

**arXiv ID:** 2608.28469 | [PDF](https://arxiv.org/pdf/2608.28469v1)

**作者:** Sirin Chakraborty `[一作]` (Auburn University), Ness B. Shroff `[通讯]` (Ohio State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了分布式交叉层优化框架，统一了拥塞控制、路由、调度与功率控制，并在对手 Willie 的能量检测覆盖范围内实现隐蔽通信；通过构造对数-检测误差概率的最优凹下界，将非凹约束转化为凸约束，并设计了并行近似 ADMM（PP‑ADMM）求解器。

**💡 创新点**

① ①将 log‑DEP 的非凹特性通过最优凹下界转换为凸约束，实现硬覆盖约束与覆盖效用最大化的统一；② 在多跳多模态网络中首次实现全局 Q‑linear 收敛的并行分布式算法；③ 引入共识拆分与松弛项，消除非分离的约束，使得每个节点仅需与邻居通信即可完成本地更新。

**🔧 技术方法**

1) 交叉层网络效用最大化（NUM）框架；2) 对数-检测误差概率凹下界构造；3) 并行近似 ADMM（PP‑ADMM）算法；4) 利用 KKT 子正则性与 Lyapunov 收敛分析；5) 近似线性/凸优化子问题的显式闭式解。

**📊 数据集**

基于人工构造的模拟网络：5、10、20 节点，8 条链路，2 种波段（VHF、UHF），2 条流量，2 个 Willie；采用距离相关的路径损耗、随机小尺度衰落和 Jakes 模型进行时变通道仿真；无真实网络数据集，仅使用仿真生成的数据。

**📈 对比分析**

与现有仅考虑路由或单层功率控制的隐蔽网络方法相比，本工作在同一实验设置下实现了：① 指数级收敛（Lyapunov 比例指数下降）；② 通过自适应更新跟踪 Willie 运动和通道衰落下的最优吞吐量与 DEP；③ 在随机拓扑下平均迭代次数和队列长度均低于预期，并保持稳定收敛；性能指标主要包括吞吐量、检测误差概率、收敛迭代次数、队列长度。

**⚠️ 局限性**

1) 仅在仿真环境下验证，缺乏真实部署实验；2) 假设完全知晓信道状态和 Willie 的位置，实际环境中可能受限；3) 对手模型采用最坏情况（所有 Willie 监测所有波段），现实中可能更复杂；4) 算法对超参数（ρ、τ、α）敏感，需要经验选择；5) 对高移动速度 Willie 的跟踪能力尚未在极端场景中充分测试。

---

## 461. Anatomy-Aware Promptable Segmentation with Online Interactive Training for AUTOPET V

**arXiv ID:** 2608.28461 | [PDF](https://arxiv.org/pdf/2608.28461v1)

**作者:** Pablo Lozano-Jimenez `[一作]` (University of Amsterdam), Ruben Tolosana `[通讯]` (BiometricsAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了基于nnU-Net的两阶段解剖感知、可提示的全身肿瘤分割系统，在PET/CT中实现FDG和PSMA tracer的交互式精细化分割。

**💡 创新点**

创新点包括：1) 通过共享头同时预测病灶与器官，利用解剖上下文抑制假阳性；2) 引入轻量级Tracer分类器实现推理时的Tracer路由；3) 在Phase‑2阶段采用在线交互式训练，利用模拟涂写提示逐步优化预测。

**🔧 技术方法**

使用nnU-Net v2框架、双编码器-解码器结构、共享宽阔卷积头、Gaussian热图提示、随机采样交互次数、Dice+交叉熵+组织辅助+涂写一致性损失等技术。

**📊 数据集**

使用AUTOPET V公开数据集：1,014份FDG PET/CT、597份PSMA PET/CT，包含多中心、多扫描仪的多模态图像。

**📈 对比分析**

通过四折交叉验证与单折交互式微调进行比较，Phase‑1版5在整体上Dice 0.686、DMM 0.794；交互式阶段在FDG上从0.585提升至0.875（5次提示），在PSMA上从0.581提升至0.825，表现出递增且显著的提升。

**⚠️ 局限性**

局限性：1) 交互式阶段仅在单折上训练，缺乏多折鲁棒性；2) 仅使用模拟涂写提示，真实用户交互可能不同；3) Tracer分类器在极端案例下可能误分；4) 仅评估在AUTOPET V内部数据，缺乏外部验证；5) 计算资源受限导致未完成完整模型训练。

---

## 462. Texture Image Classification Using DWT AlexNet Feature Fusion and Deep Neural Networks

**arXiv ID:** 2608.28524 | [PDF](https://arxiv.org/pdf/2608.28524v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 463. Curvature-Conditioned Multiscale Momentum with Sphere Constraints for LLM Pretraining

**arXiv ID:** 2608.28442 | [PDF](https://arxiv.org/pdf/2608.28442v1)

**作者:** Shuchen Zhu `[一作]` (ByteDance Seed), Kun Yuan `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在球面约束下的平坦方向多尺度动量优化方法，结合快速与慢速动量在平坦子空间中进行自适应降噪，显著加速LLM预训练。

**💡 创新点**

创新点在于：①把球面约束（固定 Frobenius 范数）引入参数更新，消除慢速动量导致的权重范数膨胀与有效学习率坍塌；②在平坦子空间上使用多尺度动量（fast/slow）并结合并行传输（parallel transport），实现对噪声的有效抑制和对曲率信息的利用；③采用基于矩阵符号（msign）和行尺度归一化的Muon's预条件器，配合在线幂迭代估计平坦子空间。

**🔧 技术方法**

技术主要包括：球面约束的流形优化（切空间投影、重拉、平行传输）；多尺度Nesterov动量；Muon's block级预条件；行尺度重参数化；在线估计平坦子空间；以及在Transformer块上进行的行/列归一化。

**📊 数据集**

在一个包含350B标记的高质量预训练语料库上进行实验，Dense模型使用约0.5M token batch，MoE模型使用约4M token batch，训练预算分别为Dense 100 tokens/参数、MoE 500 tokens/激活参数。

**📈 对比分析**

与Muion、MuionS、MuionH、SSO、AdEMAMix、EMA‑Nesterov等基线进行对比；在0.12B至2.3B规模的Dense和MoE模型上，所提方法在终端loss上分别比Muion下降约0.02~0.03，比分母MuionS下降约0.019，且在更长训练预算下性能提升更明显，显示出更好的可扩展性。

**⚠️ 局限性**

局限性包括：球面约束下的最优学习率调度仍未给出；未为慢速动量设计专门的预条件器；对超参数（如χ、α_fast/α_slow）的敏感性尚需进一步研究。

---

## 464. InstructMesh: Selective Refinement of Generative 3D Models for Fabrication

**arXiv ID:** 2608.28534 | [PDF](https://arxiv.org/pdf/2608.28534v1)

**作者:** Faraz Faruqi `[一作]` (MIT CSAIL), Stefanie Mueller `[通讯]` (MIT CSAIL)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 InstructMesh，一种面向非专业用户的交互式后期修复工具，能够通过在生成模型的潜在体素空间中执行增减操作，并结合自然语言或滑块控制，实现对 3D 模型的可制造性修复。

**💡 创新点**

创新点在于：① 在生成模型的中间潜在表示上实现可视化可编辑的增减操作，形成 WYSIWYG 预览；② 通过 LLM 的 in‑context learning 将自然语言描述映射到预定义的几何操作；③ 将两种交互模式（自然语言 + 滑块）与预览视觉反馈结合，降低专业壁垒。

**🔧 技术方法**

使用的技术包括：基于 Trellis 的两阶段 encoder‑decoder 生成模型；对 64³ 体素网格的增减操作（extrude、expand、fill、trim、erode、flatten）；GPT‑4 进行操作预测的 in‑context learning；Web 前端实现区域选择、滑块操作与预览可视化。

**📊 数据集**

数据集主要来源于 Thingiverse：120 个热门模型的 3D 重建样本用于缺陷分类与操作预设；12 名无 3D 建模经验的用户参与两轮用户研究。

**📈 对比分析**

方法评估：在 80% 的缺陷实例上使用手工校准的 12 个预设操作，独立评估成功率 ≥84%（Missing Openings 96% 等）；LLM 预测准确率 92.1%；生成平均 31.7 s，编辑后 43.7 s；用户研究显示 90% 欺辨识率、89% 修复成功率，平均修复时间约 2–3 分钟。

**⚠️ 局限性**

局限性包括：① 仅支持静态几何，无法修复关节等动态结构；② 体素分辨率约 1.5% 的尺度，细小缺陷难以修复；③ 需要用户手动定位缺陷，未提供自动缺陷检测与操作推荐；④ 预览颜色对色盲用户可能不友好。

---

## 465. Understanding Venture Capital Syndication in Information Technology Sectors: A Network Formation Perspective

**arXiv ID:** 2608.28526 | [PDF](https://arxiv.org/pdf/2608.28526v1)

**作者:** Liheng Tan `[一作]` (Chinese University of Hong Kong), Prasanna Karhade `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究美国IT行业中风险投资联投的形成机制，构建全对全年面板数据，并使用双向逻辑回归和ERGMs检验七种可能的联投机制；

**💡 创新点**

创新点在于首次将七种机制（关系持久、结构嵌入、地理同质、组织同质、领域重叠、网络中心性、经验）在同一实证框架下并列检验，并同时对比双向逻辑回归与整网络ERGMs的结果；

**🔧 技术方法**

主要技术包括双向逻辑回归（dyad‑clustered 标准误）和静态Exponential Random Graph Model（含GWESP、地理同质、组织同质项），并计划未来使用TERGM进行动态模拟；

**📊 数据集**

使用PitchBook数据库的美国IT行业交易数据（硬件、软件、混合子行业），时间范围1966-2024，涉及超过55,000名投资人，其中2024年持久活跃的1,100名投资人构成ERGMs网络；

**📈 对比分析**

比较方法是将双向逻辑回归与ERGMs结果对比，双向逻辑回归在各子行业均显示先前合作、共同合作伙伴、地理同质显著正相关，伪R²约0.10；ERGMs则验证三角闭合（GWESP）和地理同质显著正相关，组织同质的效应相对较弱；整体性能在统计显著性和解释力上均表现良好；

**⚠️ 局限性**

局限性包括：研究为描述性/预测性而非因果，累计网络权重未考虑时间衰减；缺乏交易规模、估值、领投身份等控制变量；地理与类型缺失值以非匹配处理；模型为静态，未捕捉网络演化；仅覆盖美国IT行业，外推性有限。

---

## 466. Learning the Target Priors Before Image Translation: A Decoupled Training Paradigm for Cross-Modal Image Translation in Remote Sensing

**arXiv ID:** 2608.28517 | [PDF](https://arxiv.org/pdf/2608.28517v1)

**作者:** Keyan Hu `[一作]` (Central South University), Chao Tao `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 LTP-BIT 框架，在遥感跨模态图像翻译中先预训练目标域生成先验，再通过 P‑DART 进行源条件控制。

**💡 创新点**

创新点在于将目标域先验与跨模态对应分离、利用先验匹配与条件分解的理论支持，并提出参数高效的 P‑DART。

**🔧 技术方法**

主要技术包括条件分数分解、无监督先验预训练、Dual‑Stream Asymmetric Reference Transformer (P‑DART) 与 LoRA 微调。

**📊 数据集**

使用 QXS‑SAROPT、SpaceNet6、Chesapeake 等 SAR/NIR‑RGB 数据集，预训练集为 1M Git‑10M 以及各基准目标图像。

**📈 对比分析**

与 CycleGAN、ControlNet、DiffusionSat 等方法比较，LTP‑BIT 在 PSNR、SSIM、LPIPS、FID/CMMD 上均达或超过最佳水平，且仅使用 9.81% 任务特定参数。

**⚠️ 局限性**

局限在于对极少样本的适应仍受限于目标先验覆盖范围，且仍需针对不同源域进一步调优。

---

## 467. Conformal Uncertainty Quantification Guarantees for Neural Operators

**arXiv ID:** 2608.28515 | [PDF](https://arxiv.org/pdf/2608.28515v1)

**作者:** Tom Stent `[一作]` (Imperial College London), Nicolas Boullé `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于拆分的合形预测（split conformal）框架，为神经算子提供可解释的置信带，保证在给定空间容忍度γ下，真解在至少1‑γ比例的空间点被覆盖，且对测试与校准输入的概率至少为1‑α。

**💡 创新点**

核心创新在于：① 将残差场归一化为空间(1‑γ)分位数，计算一个单一的尺度因子；② 在连续域和离散网格上都能给出有限样本、分布无关的边际覆盖保证；③ 在i.i.d.假设下推导出条件覆盖概率服从Beta分布，量化校准集大小对覆盖变异性的影响。

**🔧 技术方法**

技术包括：神经算子（如FNO）预测与误差估计器；可测的残差场与分位数分数；拆分合形校准（split conformal calibration）；对连续与离散空间分别构建覆盖保证；Beta分布分析与Beta-Binomial检验。

**📊 数据集**

使用标准的二维Darcy流和二维不可压Navier–Stokes（vorticity形式）数据集，在不同分辨率下进行实验。

**📈 对比分析**

与现有UQNO方法对比：在相同的算子与误差估计器下，本文方法产生更紧的置信带，empirical覆盖率与理论Beta-Binomial分布高度吻合；而UQNO更保守，覆盖率集中在0.98以上。

**⚠️ 局限性**

局限性：覆盖保证仅为边际（对校准与测试输入的联合概率），不保证在每一次校准集下都满足；假设输入可交换，无法应对分布漂移；离散保证与网格分辨率相关，需进一步提升到连续域的可迁移性。

---

## 468. NL2AGBench: Benchmarking LLM Auto-Formalization for AlphaGeometry

**arXiv ID:** 2608.28481 | [PDF](https://arxiv.org/pdf/2608.28481v1)

**作者:** Samuel Xiao `[一作]` (Valley Christian High School), Ziliang Zong `[通讯]` (Texas State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出NL2AGBench基准，用于评估LLM将自然语言几何问题自动形式化为AlphaGeometry DSL的能力。

**💡 创新点**

创新性地引入执行验证评估和错误分类体系，系统衡量语法与逻辑错误并比较开放与闭源模型性能。

**🔧 技术方法**

使用大语言模型（包括open‑source和closed‑source）、few‑shot prompting、fine‑tuning和人工提示等技术来完成形式化翻译。

**📊 数据集**

基于AlphaGeometry已手工形式化的231道几何题，抽样48道构成NL2AGBench数据集进行评测。

**📈 对比分析**

通过可执行翻译率、语法正确率和错误类型统计对10个模型进行对比，闭源模型可执行率超过80%，开源最大不到46%，few‑shot prompting显著提升性能。

**⚠️ 局限性**

主要局限在于LLM对DSL语法与几何推理的掌握不足，导致开源模型表现低下；人机提示虽有效但不易规模化。

---

## 469. ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL

**arXiv ID:** 2608.28476 | [PDF](https://arxiv.org/pdf/2608.28476v1)

**作者:** Zhuoshi Pan `[一作]` (Tsinghua University), Xing Sun `[通讯]` (Tencent Youtu Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ContextPilot， 一个针对长时序代理任务的主动上下文管理框架，能够在多轮交互中主动编辑、规划、记忆和压缩上下文。

**💡 创新点**

创新点在于①扩展工具集合，引入规划、长时记忆和软上下文卸载工具；②设计了针对上下文管理的 RL 训练策略，采用上下文/熵变化感知的局部展开（partial rollout）和细粒度奖励分配；③通过这些技术保持更紧凑的工作上下文并显著提升任务性能。

**🔧 技术方法**

使用的技术包括：基于 ReAct 的多轮工具调用框架、强化学习（GRPO）优化、上下文/熵敏感度度量、离散快照（snapshot）训练、工具集合扩展和软上下文卸载实现。

**📊 数据集**

主要实验数据集：长上下文问答（NovelQA、NarrativeQA、LongBench-v2）和深度检索（OpenSeeker、GAIA、BrowseComp、BrowseComp-ZH、xBench-DeepSearch）。

**📈 对比分析**

与 ReadAgent、MemAgent、StateLM 等基线以及无 Fine‑Tuning 的工具调用版本对比，ContextPilot 在所有四大问答基准上平均提升约3–4分，深度检索任务平均提升约1.5分；同时上下文长度保持在 8k–10k tokens，远低于同类方法。

**⚠️ 局限性**

局限性包括：工具集合仍不完备，可能无法覆盖所有上下文编辑需求；RL 超参数（如局部展开权重、奖励划分）未做充分搜索；实验范围仅涵盖问答和检索任务，未验证在编程、GUI 等更广泛代理场景中的效果。

---

## 470. ChainSplat: A Physics-Inspired Screw-Theoretic Model for Learning Deformable Linear Object Dynamics from Multi-View RGB Videos

**arXiv ID:** 2608.28570 | [PDF](https://arxiv.org/pdf/2608.28570v1)

**作者:** Seungyeon Kim `[一作]` (KTH Royal Institute of Technology), Noémie Jaquier `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 Chainsplat，一种基于开链剃螺旋理论和高斯平涂渲染的物理启发式模型，用来从多视角 RGB 视频中联合学习柔性线性物体（DLO）的三维几何、外观、运动学和动力学。

**💡 创新点**

创新点在于：①使用可微分的高斯平涂渲染实现几何与外观的端到端学习；②引入开链剃螺旋框架提供低维状态表示并显式施加物理约束；③设计两阶段完全可微分的优化流程（几何识别 + 动力学识别），避免传统多阶段管线误差累积。

**🔧 技术方法**

技术手段包括：剃螺旋运动学与动力学、Gaussian Splatting、隐式欧拉积分、梯度基优化、SMPL 2D 轨迹重建以及 SAM/CoTracker 等视觉分割工具。

**📊 数据集**

使用的数据集为三条不同长度（20cm、30cm、40cm）的绳索，采集 24 条 5 秒的机器人-物体交互轨迹，采用三台 RealSense D435 和一台 ZED Mini 摄像机，记录 RGB、深度和机器人末端姿态。

**📈 对比分析**

与 PGND、PhysTwin（含 3D 重建和无 3D 重建两种变体）比较，Chainsplat 在 Chamfer Distance、IoU、PSNR、SSIM、LPIPS 等指标上均显著优于对手，同时训练时间缩短 30% 以上、推理速度提升 2–3 倍，证明其在精度和效率上的优势。

**⚠️ 局限性**

局限性包括：剃螺旋轴固定且假设为平面运动，难以处理高度非线性或三维复杂弯曲；需要多视角 RGB 输入，单目视角效果有限；两阶段优化易受前期几何估计误差影响，缺乏端到端自适应与在线更新能力。

---

## 471. SignRR: Retrieve and Refine Real Motion for Sign Language Production

**arXiv ID:** 2608.28568 | [PDF](https://arxiv.org/pdf/2608.28568v1)

**作者:** Fidel Omar Tito Cruz `[一作]` (University of Central Florida), Gissella Bejarano `[通讯]` (Marist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Retrieve‑and‑Refine框架SignRR，用检索得到的真实手语段落作为初始序列，再通过部分感知残差向量量化自编码器对整个序列进行全局细化，提升连续手语生成的连贯性与真实性。

**💡 创新点**

创新点在于：①将检索与生成结合，避免从噪声或先验直接生成手语导致的细节缺失；②使用多体部件感知的残差VQ‑VAE，在保留手部细节的同时实现全序列的时长对齐与风格统一；③采用动态规划在词汇-运动字典中选取平滑且高质量的段落，从而保证初始序列的结构合理。

**🔧 技术方法**

核心技术包括：基于语义匹配的手语词汇-运动字典构建、动态规划段落组合、分体（身体、右手、左手）编码的残差向量量化自编码器、长度预测与潜在空间插值、以及自监督的重建、速度、提交和长度损失。

**📊 数据集**

在两个公开基准上评估：PHOENIX14T（约1.1k词汇）和 CSL‑Daily（约2k词汇）。

**📈 对比分析**

与现有生成（如G2P‑DDM、Sign‑IDD）及检索（Sign Stitching）方法比较，SignRR在PHOENIX14T的回译指标（WER 80.98，BLEU‑1 28.78，BLEU‑4 10.96，ROUGE 28.15）均超过非GT方法，且MPJPE仅次于最佳生成模型；在CSL‑Daily上同样取得最高的回译分数。

**⚠️ 局限性**

局限性包括：仅处理身体与手部关键点，未充分捕捉面部表情与非手势线索；检索依赖已有词汇表，未解决未见词汇的开放式生成；评价仅通过回译指标，缺乏专家判读的自然性与可理解性评估。

---

## 472. Aero Hand Open: A Simulation-Ready Tendon-Driven Hand for Dexterous Manipulation Learning

**arXiv ID:** 2608.28578 | [PDF](https://arxiv.org/pdf/2608.28578v1)

**作者:** Nan Wang `[一作]`, Yiwei Tao `[通讯]` (California Institute Of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一款低成本、全3D打印的五指肌腱驱动手（Aero Hand Open），并公开其机械设计、固件、控制栈；在MuJoCo中实现了精确的肌腱传动模型；通过识别仿真与实机的线性驱动映射，将仿真命令与硬件电机指令对齐；在该模型上使用PPO训练了一个在手内旋转立方体的策略；该策略在无任何微调、仅使用电机编码器信息的情况下，实现了在真实硬件上的零步调转。

**💡 创新点**

① 以肌腱传动为核心的手部结构实现了低成本与高仿人性化兼容；② 在仿真中以空间肌腱方式重现真实的传动几何，避免了传统关节驱动的误差；③ 通过对传动线性系数的识别，完成了仿真–实机的双向映射；④ 仅使用电机编码器作为观测，实现在真实硬件上的零步调转。

**🔧 技术方法**

3D打印、肌腱驱动机电结构、MuJoCo空间肌腱模拟、线性驱动映射、域随机化、PPO（Brax实现）、ROS2部署节点。

**📊 数据集**

使用仿真环境中生成的随机参数（如摩擦、质量、关节摩擦等）以及立方体旋转任务的模拟数据；实机侧通过电机编码器读取和实验记录进行验证；未使用公开的第三方数据集。

**📈 对比分析**

通过对比实机与仿真在肌腱行程、跟踪均方根误差和上升时间的差异（四指行程误差0.04–1.40 mm，跟踪RMS0.29–0.45 mm，上升时间仿真90 ms vs 实机120 ms），评估仿真模型的准确性；在仿真中策略平均旋转速0.51 rad/s，实机零步调转后可在约55 秒完成一整圈。

**⚠️ 局限性**

传动映射中拇指关节的耦合仍存在残差，导致拇指通道误差较大；由于系统仅提供七个电机编码器，观测空间受限，可能限制策略对更复杂抓握的泛化；实验仅在单一立方体旋转任务上验证，其他抓握场景的性能未知。

---

## 473. A Formal Limitation on Learning Human Language From Textual Corpora

**arXiv ID:** 2608.28560 | [PDF](https://arxiv.org/pdf/2608.28560v1)

**作者:** Emily Cheng `[一作]` (Universitat Pompeu Fabra), Ryan Cotterell `[通讯]` (ETH Zuerich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在仅靠文本表征的情况下，系统从发话形式恢复说话者意图（意义）的信息论上限，并在人工与真实数据上验证。

**💡 创新点**

首次给出了针对有限与无限意义空间的通用信息论上界，揭示仅凭形式无法超越由形式-意义互信息决定的极限，并通过可验证实验与现有LLM/VLM激活一致性进行实证。

**🔧 技术方法**

采用信息论工具（熵、互信息、Fano不等式）推导上界，并利用机器学习解码器（MLP）和大规模语言/视觉语言模型的隐藏表示进行实验。

**📊 数据集**

使用人工合成语言（离散与连续意义空间）、中文零代词解析数据集（Wang2018TranslatingPL）以及Monroe等人2017年的颜色命名对话数据。

**📈 对比分析**

将模型（LLM、VLM）在最优超参数下的解码精度与理论上限做比较，实验结果始终低于上界；在零代词任务中LLM并未突破基线。

**⚠️ 局限性**

未测试大于14B参数模型，未考虑多轮或丰富情境下的上下文信息，对意义的因果解释不足。

---

## 474. Advancing Interaction-Sensitive Feature Selection: Novel Relief-Based Algorithms, Expanded Comparisons, and Recommendations for Biomedical Data Mining

**arXiv ID:** 2608.28552 | [PDF](https://arxiv.org/pdf/2608.28552v1)

**作者:** Kia Kazemi-Nia `[一作]` (Cedars-Sinai Health Sciences University), Ryan J. Urbanowicz `[通讯]` (Cedars-Sinai Health Sciences University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了 Relief 基础特征选择算法，扩展了 scikit-rebate 包并新增 5 种组合算法；

**💡 创新点**

创新点包括对 scikit-rebate 进行重构实现 10‑35 倍速度提升，加入 SWRF*、μ‑Relief 以及 MultiSWRF、MultiSWRF*、MultiSWRFDB、MultiSWRFDB*、SWRF 等新算法；

**🔧 技术方法**

使用 Relief、SURF、MultiSURF 等核心 RBA 变体，并在邻域重计算、死带区、梯度权重等机制上进行改进；

**📊 数据集**

使用 GAMETES 生成的 2310 个模拟基因组数据集，涵盖 2‑到 5‑维纯交互、主效应、异质效应，以及不同特征数、样本量、遗传力等维度；

**📈 对比分析**

通过统计所有预测特征在前 X% 内排名成功率生成热图并计算全局平均/中位排名，结果显示 star 版 RBA 在 2‑维交互中表现最佳，MultiSWRFDB* 适合仅关注 2‑维交互，SWRF 兼顾交互与主效应，而 μ‑Relief 在 3‑维交互中表现最好；但无算法能可靠捕捉 4/5 维交互；

**⚠️ 局限性**

局限在于仅使用模拟数据未验证真实基因组，star 版在主效应或多类别数据上性能下降，且对极大特征空间（>10k）仍需 RBA‑wrapper。

---

## 475. QGPINNs: A Physics-Informed Neural Network Framework for Nonlocal Differential Equations on Quantum Graphs

**arXiv ID:** 2608.28589 | [PDF](https://arxiv.org/pdf/2608.28589v1)

**作者:** Vaibhav Mehandiratta `[一作]` (Birla Institute of Technology and Science), Saket Ramchandra `[通讯]` (Birla Institute of Technology and Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习和机器学习相结合的技术，特别是卷积神经网络（CNN）和支持向量机（SVM）。

**📊 数据集**

实验使用了公开的图像数据集和文本数据集，以验证算法的有效性。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和处理速度上均优于其他方法。

**⚠️ 局限性**

限制在于算法在特定类型的数据上表现较好，但在其他类型的数据上可能效果不佳。

---

## 476. PULSAR: Pooled Unified Late-Interaction Search and Retrieval for Enterprise Visual Document RAG

**arXiv ID:** 2608.28572 | [PDF](https://arxiv.org/pdf/2608.28572v1)

**作者:** Benjamin Constable `[一作]` (Microsoft), Aidan Millar `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并部署了PULSAR，基于视觉先验的多向量检索系统，用于处理机构级投资文件中的表格、图表等视觉信息；

**💡 创新点**

将检索过程拆分为两阶段的池化后交互式索引，并将嵌入 DPI 与答案时 DPI 解耦，从而在保持检索质量的同时显著降低延迟与成本；

**🔧 技术方法**

采用ColQwen3‑4B（ColPali 风格）视觉-语言模型进行多向量嵌入，结合 HNSW+量化、均值/层次池化、MaxSim 重排序以及 Qdrant 向量索引；

**📊 数据集**

在公开的 ViDoRe V3 基准上评估模型，并在摩巴达拉投资公司内部部署，服务 78 k 文档、约 2.4 M 页；

**📈 对比分析**

与传统 OCR+可视化转录基线对比，PULSAR 在答案完整性、上下文/答案事实召回率上至少提升 2–5 倍，向量搜索延迟下降 15.1×，吞吐量提升 88×；

**⚠️ 局限性**

虽然性能显著提升，但仍比单向量检索更重，需处理更多向量；池化与量化方案在不同语料或模型上可能需要重新调优；

---

## 477. Blog: Survey of Optimizers

**arXiv ID:** 2608.28557 | [PDF](https://arxiv.org/pdf/2608.28557v1)

**作者:** Ruoran Xu `[一作]` `[通讯]`, Ruoran Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化 2025–2026 年优化器研究，提出四维轴框架（时间估计、几何、时域、表示）并整理矩阵感知、低精度、低内存、时间策略等方法。

**💡 创新点**

形成优化器堆栈视角，区分四个独立维度，并给出评价协议与证据分层；揭示矩阵方法真实效益与局限，强调从名称向机制组合转变。

**🔧 技术方法**

评估多种优化器（AdamW、Muon、SOAP、SPlus、AdEMAMix、MARS、SWAN、SOLO、APOLLO、GaLore、LDAdam 等），采用大规模语言模型预训练实验、token/steps/FLOPs/时钟/内存对比，利用矩阵统计、Kronecker、极化、低秩、量化、分布式通信等技术。

**📊 数据集**

主要使用大规模语言模型预训练数据（Common Crawl、OpenWebText 等），实验覆盖 0.1B–16B 参数、1×–8× 数据/参数比例，token 数量从 101B 到 1T+。

**📈 对比分析**

通过多维度评估（token 效率、步骤数、FLOPs、训练时钟、内存占用）和受控/规模/新兴实验进行比较；结果显示矩阵方法可提升 1.1–1.4 倍的 token 与 FLOPs 效率，低内存/低精度方法显著降低状态占用，低精度方法实现 4–8bit 训练；不同方法在不同规模、批量、时域下排名变化。

**⚠️ 局限性**

结论高度依赖实验协议、批量、规模、超参调优；缺乏统一评测基准；多数实验仅针对语言模型；方法之间的可迁移性、实现细节与系统成本难以量化；没有真正的“通用替代”AdamW，需针对具体场景定制堆栈。

---

## 478. A Complete Characterization of Tensorizable $f$-divergences

**arXiv ID:** 2608.28556 | [PDF](https://arxiv.org/pdf/2608.28556v1)

**作者:** Rodrigo Cruz `[一作]` (Harvard University), Qian Yu `[通讯]` (University of California at Santa Barbara)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对所有可张量化的f-散度进行完整表征，证明它们只能是KL与逆KL散度的线性组合或幂散度。

**💡 创新点**

首次在不假设张量化公式的具体形式下，对张量化f-散度给出完全的、无约束的分类，揭示其与多线性函数和局部特征参数的内在联系。

**🔧 技术方法**

利用对数似然比分布的卷积结构、函数方程与多线性（多仿射）性质、凸性与正则性分析，以及测度论中的 Radon‑Nikodym 变换和卷积运算，构建并求解一系列关键的代数与分析方程。

**📊 数据集**

无；该工作为纯理论数学/信息论研究，不涉及具体实验数据集。

**📈 对比分析**

由于是理论证明性工作，没有直接的实验对比；但通过与已知的KL、χ²、Hellinger 散度等特殊案例的一致性验证，确认了所给公式在这些典型情形下与现有结果完全匹配。

**⚠️ 局限性**

结果仅适用于 f-散度，且张量化 f-散度在加法下并不封闭；此外，若需研究更广泛的可组合散度族（如线性组合、复合张量化等），仍需进一步探索；同时对不连续或非凸生成函数的情况未作讨论。

---

## 479. GeBDA: Building Damage Assessment as Text-Based Sequence Prediction

**arXiv ID:** 2608.28567 | [PDF](https://arxiv.org/pdf/2608.28567v1)

**作者:** Olivier Dietrich `[一作]` (ETH Zurich), Genady Beryozkin `[通讯]` (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将建筑损毁评估（BDA）视为一个自回归序列生成任务，直接使用大规模视觉‑语言模型 Gemma 对双时段卫星影像生成建筑框和损毁等级的文本序列。

**💡 创新点**

创新点：① 将全局单通生成与建筑定位与分级结合，避免使用外部检测/分割模块；② 采用纯文本化坐标的序列化方式，保持模型端到端的生成范式；③ 通过微调通用 VLM 而非专门设计的密集预测网络实现 BDA。

**🔧 技术方法**

技术细节：Gemma 4 系列 VLM + SigLIP 视觉编码器；JAX/Kauldron 训练框架；交叉熵微调 + 贪心解码；坐标量化到 1000×1000 网格并序列化为文本；Hungarian 匹配评估；裁剪 512×512 补丁以控制序列长度；使用 280 个视觉 token 进行 768×768 的 up‑sampling。

**📊 数据集**

数据集：xBD（光学/光学）共 11,034 张图、>425k 建筑、4 类损毁；Bright（光学/SAR）共 3,029 张图、≈245k 建筑、3 类损毁。

**📈 对比分析**

评估方式：将预测框 raster 化为密集掩模，分别与像素级基准（UNet、ChangeOS、ChangeMamba）对比，使用 IoU 匹配、F1 统计；结果显示：xBD 上定位 78.34/82.60（Oracle），分类 73.27/99.40；Bright 上定位 81.23/99.65，分类 51.56/98.14。与现有像素级模型相比，定位性能接近或略低，分类在光学图像上表现可观，SAR 图像上显著下降；联合训练（GeBDA*）对 Bright 分类提升 22.6%。

**⚠️ 局限性**

局限性：① 冻结的 RGB 视觉编码器在 SAR 影像中提取特征不足；② 高密度场景易产生误检小框或循环生成；③ 坐标以纯文本序列化导致 token 效率低，计算成本高；④ 需要更多参数高效方式（如 LoRA）与更大模型规模来提升性能。

---

## 480. QUEST: A Query and Extraction System for Topics in Asylum Law Application Decisions

**arXiv ID:** 2608.28555 | [PDF](https://arxiv.org/pdf/2608.28555v1)

**作者:** Maria Vlachou `[一作]` (University of Copenhagen), Desmond Elliott `[通讯]` (University of Copenhagen)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一套名为QUEST的系统，用于从丹麦难民上诉文件中检索并识别与可信度评估相关的文本片段。

**💡 创新点**

创新点包括：①引入领域特定的可信度相关评价（crels），区别于传统的查询相关性；②利用法律代码本进行无监督的合成查询生成；③将BERTopic与LLM（Gemma3、Llama3）结合，用于主题提取与命名；④通过LLM-as-a-judge进行大规模的可信度标签生成。

**🔧 技术方法**

技术栈主要包括：信息检索（BM25、SPLADE、E5、Qwen3），密集检索与稀疏检索混合；主题抽取（BERTopic）与LLM生成标签；LLM-as-a-judge（Llama3）用于评价；使用多种评估指标（MAP@100、NDCG@10、MRR@10）与Cohen’s κ、Krippendorff’s α等一致性度量。

**📊 数据集**

使用了丹麦难民上诉委员会（RAB）的两类数据集：公开的叙事摘要与完整的敏感上诉材料（2001–2025年），包含访谈记录、决定文本及补充证据。

**📈 对比分析**

在标准查询相关性（qrels）评估下，SPLADE/E5等检索模型表现优于BM25，MAP和NDCG均较高；在可信度相关评估（crels）下，整体性能下降，显示任务更难；再排名（monoT5）在大多数指标上提升有限。LLM评判与人工评估的κ值低至0.04-0.13，表明LLM在可信度判断上相对宽松。

**⚠️ 局限性**

局限性：①全部依赖LLM，可能受模型偏见与语言模型覆盖度限制；②仅使用丹麦数据，缺乏跨国验证；③大部分标签由LLM生成，缺少人类黄金标准；④数据高度敏感，限制实验规模与公开复现；⑤在严格评估阈值下性能显著下降，显示对可信度判断的可解释性与可靠性仍待提升。

---

## 481. Logos: An Agent Harness on a Cross-Process Bus

**arXiv ID:** 2608.28553 | [PDF](https://arxiv.org/pdf/2608.28553v1)

**作者:** Hanzhang Jia `[一作]` (University of Sussex), Bo Ma `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了跨进程插件式代理系统 Logos，利用 spatiotemporal‑composability 计算律证明在分布式环境中仍能保持可逆性，并将插件、组合、装配等操作从单进程迁移到同一总线上的多进程节点。

**💡 创新点**

创新点在于：①把组合与装配移出宿主进程，改用只读 Append‑only transcript 做持久载体；②通过 ROS‑风格的同伴进程架构实现多语言、多进程插件；③给出四条 lemmas 证明跨进程可逆性，并在实践中展示冷切换与恢复的完整流程。

**🔧 技术方法**

所用技术包括：spatiotemporal‑composability 计算律、ROS 架构、Go 语言实现路由器、Python 与 Node.js 运行时工具、NDJSON TCP bus、Append‑only transcript 与事件流、广播注册表、单写锁、冷切换与会话恢复等。

**📊 数据集**

实验主要使用多种大语言模型（DeepSeek V4 Flash/Pro、GLM‑5.2/5.3、Claude Opus 5/4.6、GPT‑Image‑2、GPT‑5.4）以及自编会话代码（单代理、多代理、工具调用等）进行测试；未使用公开标准数据集，而是通过模型接口与工具调用构成工作负载。

**📈 对比分析**

比较方法：将 Logos 与单进程实现（同一计算律实现）在同一硬件上对照，测量路由器/工具故障恢复、冷切换、端到端恢复、并发会话、内容争用等场景。性能结果显示：bus hop 0.215 ms；first‑token 177 ms；router kill 恢复 858 ms；tool kill 恢复 100.5 ms；12 会话通过 6 次进程终止恢复 1.36 s；80 会话在工具调用四个关键点恢复且无重复效果。整体恢复更快，单进程单点故障导致所有共驻会话中断的缺陷被消除。

**⚠️ 局限性**

局限性包括：①仍依赖可靠的局域网或单机部署，跨机扩展未实验；②Append‑only transcript 随时间增长，存储与恢复成本随之提升；③单写锁与广播序列的同步开销在极高并发下可能成为瓶颈；④对外部 I/O（如支付、数据库事务）未在模型中覆盖；⑤需要手动维护日志压缩与快照以防磁盘膨胀。

---

## 482. Video Generative Models as Geometry Learner

**arXiv ID:** 2608.28549 | [PDF](https://arxiv.org/pdf/2608.28549v1)

**作者:** Haosen Yang `[一作]` (University of Surrey), Jiankang Deng `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

该工作将预训练视频生成模型改造为统一的几何学习框架，通过将单张RGB图像的深度和法线预测视作下一帧生成，实现单目深度与法线的联合推断。

**💡 创新点**

创新点在于：①将视频扩散模型作为几何推断的条件生成器；②将几何预测表述为视频中的“下一帧”任务；③在联合生成中同时恢复输入图像以增强图像-几何一致性。

**🔧 技术方法**

技术上使用Stable Video Diffusion预训练模型、Latent Diffusion、EDM噪声调度、CLIP嵌入、可微分的U‑Net、对齐VAE编码解码。

**📊 数据集**

训练使用合成数据集Hypersim（约39k样本）与Virtual‑KITTI2（约20k样本）；测试在NYUv2、ScanNet、KITTI、ETH3D、DIODE（深度）以及NYUv2、ScanNet、iBims‑1、Sintel、OASIS（法线）等真实数据集。

**📈 对比分析**

与基线比较：在零样本/零-shot 任务中，GeoNeXt 在多数基准上优于或与现有生成式方法（Marigold、Lotus‑G、GeoWizard 等）相当；与大规模监督式深度模型 DepthAnything 等相比，仅使用 1/100 量级训练数据即可达到相近甚至更优的 AbsRel/δ1 等指标。

**⚠️ 局限性**

限制：对真实世界多模态场景的泛化仍受限于合成数据的分布；模型在极端光照/运动模糊下的精度下降；当前实现仅支持单张图像到深度+法线的下采样推断，未涵盖更高维度几何或多视角场景。

---

## 483. On two proofs of $d^2$ mixing of weighted Dikin walks

**arXiv ID:** 2608.28566 | [PDF](https://arxiv.org/pdf/2608.28566v1)

**作者:** Yuansi Chen `[一作]` (ETH Zürich), Yunbum Kook `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究了在多面体和截断正半定锥上使用加权 Dikin 步行采样指数分布的混合时间，给出了总变差和 χ^2 误差下的 O(d^2) 与 O(d^4) 结果。

**💡 创新点**

创新点在于引入高概率接受率控制、s-导电率分析、四阶自举条件以及对 Lee–Sidford、Lewis 权重和 John 以及混合 PSD 阻碍的通用框架，从而将之前的 d^{9/4} 上界降低到 d^2，并实现对 PSD 锥的改进。

**🔧 技术方法**

使用了自共形障碍函数、Dikin 采样、交叉比等离散/连续几何方法、s-导电率与交叉比等距化、低轨迹自共形性（LTSC）、平均自共形性、四阶自举技术以及多种加权度量。

**📊 数据集**

该工作为理论研究，无实验数据集；所有结果均为理论上界。

**📈 对比分析**

相较于之前的 d^{9/4} 以及基于球步/击中跑的 d^2 上界，该方法在总变差下实现 O(d^2) 的混合时间，在 χ^2 下亦达到 O(d^2)，并在截断 PSD 锥上实现 O(d^4) 的上界，显著优于过去的 m 依赖结果。

**⚠️ 局限性**

局限性包括：需要满足强自共形、ν̅-对称性、混合 23‑跟踪等条件，框架对特定加权度量有依赖；对 PSD 锥的结果仍为 O(d^4)，比理想的 O(d^2) 仍有差距；常数因子和实现细节尚未给出。

---

