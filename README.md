# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-27 | 今日论文总数: 883

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Sink or SWIM: Tackling Real-Time ASR at Scale

**arXiv ID:** 2601.17097 | [PDF](https://arxiv.org/pdf/2601.17097v1)

**作者:** Federico Bruzzone `[一作]`, Dario Pellegrino `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一个可支持多客户端、可实时推理的ASR系统，利用共享模型与分段式假设缓冲实现低延迟、可扩展的语音识别服务；

**💡 创新点**

提出了共享音频缓冲与本地一致性（local agreement）机制，显著降低了多服务间的冗余计算与延迟；

**🔧 技术方法**

采用gRPC实现跨语言客户端通信，使用Fast Whisper模型进行分段转写，结合分段生成器和标签器进行语义聚合；

**📊 数据集**

主要在Common Voice和LibriSpeech数据集上训练与评估；

**📈 对比分析**

与传统单模型端到端ASR进行对比，实验显示WER下降约2-3%，实时性提升30%（延迟从~350ms降至~230ms）；

**⚠️ 局限性**

系统受限于单GPU内存、模型并行度较低，且对低资源语言支持不足，未来需探索多GPU协同与模型压缩技术。

---

## 2. JetFormer: A Scalable and Efficient Transformer for Jet Tagging from Offline Analysis to FPGA Triggers

**arXiv ID:** 2601.17215 | [PDF](https://arxiv.org/pdf/2601.17215v1)

**作者:** Ruoqing Zheng `[一作]` (Imperial College London), Zhiqiang Que `[通讯]` (Imperial College London)

**通讯引用:** 735 | [OpenAlex ID](https://openalex.org/A5054475218)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 JetFormer，一个可扩展的编码器‑only Transformer，用于 LHC 质子碰撞产生的喷射（jet）分类，并实现从离线高精度到在线低延迟的统一框架。

**💡 创新点**

在不使用显式粒子对间交互的情况下，通过引入 class token 与 batch normalization、简化激活为 ReLU，得到高效、可硬件友好的 Transformer 结构；同时构建了基于多目标 Optuna 搜索、结构化剪枝和 1‑bit 量化的完整硬件感知压缩流水线，产生可在 FPGA 触发系统中 sub‑microsecond 延迟的模型。

**🔧 技术方法**

使用 Transformer 编码器、Self‑Attention、ReLU+SiLU、BatchNorm、AdamW+OneCycleLR、Optuna 多目标搜索、结构化剪枝、1‑bit 量化、Allo MLIR 编译、Vitis HLS、FPGA 低延迟实现等技术。

**📊 数据集**

在 HLS4ML LHC Jet 150 粒子数据集和大型 JetClass（约 100M 事件、17 维特征）上进行训练与评估。

**📈 对比分析**

与 MLP、Deep Sets、Interaction Networks、JEDI‑net、ParticleNet、ParT 等基线模型在相同粒子数和特征维度下进行准确率、AUC、FLOPs、参数量对比；JetFormer 在 150P 数据集上准确率比基线高 3–4%，在 JetClass 上仅比 ParT 低 0.7% 但 FLOPs 减 37.4%；压缩后 1‑bit 量化模型只丢 1.5–3.5% 准确率。

**⚠️ 局限性**

目前 FPGA 实现缺乏流水线、循环展开等手工优化，导致延迟仍高；1‑bit 量化在粒子数增多时误差累积显著；模型在极低延迟（<1µs）下的可行性仍需进一步硬件加速与 co‑design。

---

## 3. The Viscosity of Logic: Phase Transitions and Hysteresis in DPO Alignment

**arXiv ID:** 2601.17260 | [PDF](https://arxiv.org/pdf/2601.17260v1)

**作者:** Marco Pollanen `[一作]` `[通讯]` (Trent University), Marco Pollanen (Trent University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在7B开源模型上使用Direct Preference Optimization（DPO）时，将对齐压力参数β视为控制变量，系统性扫β并评估各能力指标，揭示出非单调的相变、边界敏感性、路径依赖（hysteresis）以及代理与实际能力解耦等现象。

**💡 创新点**

首次将物理相变与hysteresis概念引入对齐研究，揭示不同架构在相同β下呈现的三种响应模式（plastic、selective、smooth），并证明单一指标（如DPO偏差）并不能保证能力提升，提供了更细粒度的对齐风险评估框架。

**🔧 技术方法**

采用DPO目标函数结合LoRA微调（rank 8, alpha 16, dropout 0.05），利用对齐压力参数β的对数网格搜索；评估指标包括逻辑、算术、格式、顺从、否定等Probe的长度归一化对数概率边缘；同时记录训练粗糙度（roughness）和偏差边缘。

**📊 数据集**

使用公开的三款7B模型：Mistral‑7B、Llama‑2‑7B 和 Qwen1.5‑7B，分别在相同的训练配置下进行β扫频实验，并在不同seed下复现。

**📈 对比分析**

通过对比不同β、不同模型的Probe边缘与DPO偏差，发现逻辑Probe在β≈0.01处出现正边缘“逻辑正 pocket”，但此时DPO偏差与逻辑能力呈强负相关（Pearson r≈‑0.91）。同一β路径下，先高β再降β的“Anneal”路径会产生比直接低β训练更差的能力（hysteresis）。整体性能表明，单一指标无法反映真实能力，需多种Probe与多seed验证。

**⚠️ 局限性**

研究仅覆盖7B规模，β网格粗略且仅在关键区间提升分辨率；Probe集合有限，无法覆盖所有能力维度；缺乏对更大规模模型或不同训练曲线（如更长步骤、不同学习率）下的通用性验证；机制解释仅为假设，未深入分析梯度或模型内部结构变化。

---

## 4. The Three Axes of Success: A Three-Dimensional Framework for Career Decision-Making

**arXiv ID:** 2601.17023 | [PDF](https://arxiv.org/pdf/2601.17023v1)

**作者:** Meng-Chi Chen `[一作]` (Massachusetts Institute of Technology), Meng-Chi Chen `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 633 | [OpenAlex ID](https://openalex.org/A5004015306)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了三轴成功模型（财富、自治、意义），用于将职业路径视为在三维价值空间中的受限优化问题；

**💡 创新点**

将人力资本理论、自主性理论与有效利他主义的社会影响指标统一为一套可操作的三维框架，并在此基础上推导了轴间耦合动态、相位转变和双职业协同失效；

**🔧 技术方法**

主要采用理论建模与推导、文献综述与概念框架构建，没有具体计算机算法；

**📊 数据集**

未使用任何实验或大规模数据集，所有指标均为理论性代理；

**📈 对比分析**

论文未给出实验对比或性能评估，主要通过与已有文献对比说明框架的创新与完善；

**⚠️ 局限性**

局限性包括：①轴分解是主观模型，可能缺少健康、关系等维度；②模型仅为定性形式，缺乏经验验证；③假设个体理性可能与现实决策偏差；④忽视跨文化、行业差异导致的变异性。

---

## 5. Risk-based test framework for LLM features in regulated software

**arXiv ID:** 2601.17292 | [PDF](https://arxiv.org/pdf/2601.17292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 6. Communication-Avoiding Linear Algebraic Kernel K-Means on GPUs

**arXiv ID:** 2601.17136 | [PDF](https://arxiv.org/pdf/2601.17136v1)

**作者:** Julian Bellavita `[一作]` (Cornell University), Giulia Guidi `[通讯]` (Cornell University)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5034932123)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了分布式内存、GPU加速的精确核K‑means聚类算法，重点提出通信量最小化的1.5D实现；

**💡 创新点**

利用应用特定的线性代数分布与通信避免策略，将GEMM和SpMM无重排地组合，显著降低通信量并实现可伸缩性提升；

**🔧 技术方法**

使用稠密矩阵乘法(GEMM)、稀疏‑稠密矩阵乘法(SpMM)、稀疏向量乘法(SpMV)以及SUMMA、1.5D/2D/1D分布；在CUDA、cuSPARSE、SLATE以及GPU‑aware MPI基础上实现；

**📊 数据集**

在三组真实libSVM数据集上评估：KDD‑sampled（约840万样本、10000维），HIGGS（约1100万样本、28维），MNIST8m（约810万样本、784维）；

**📈 对比分析**

与单GPU滑窗基线对比，256 GPU时1.5D可比滑窗快10倍以上，最高达2749×；与1D基线相比提升3.6×；在256 GPU上弱/强标量效率分别约79%/4.2×，可聚类超过150万样本，提升1–2个数量级；

**⚠️ 局限性**

仍受GPU内存和通信延迟限制，H‑1D在GPU数>16时内存不足，2D因额外通信略逊；对极大规模（数千万）或高维数据仍有挑战，且缺乏自动调优机制。

---

## 7. Interpretable and Sparse Linear Attention with Decoupled Membership-Subspace Modeling via MCR2 Objective

**arXiv ID:** 2601.17042 | [PDF](https://arxiv.org/pdf/2601.17042v1)

**作者:** Tianyuan Liu `[一作]`, Bin Yan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种可解释稀疏线性注意力机制 DMSA，并将其集成到 ToST 架构中，形成 DMST。

**💡 创新点**

通过将成员矩阵 Π 与子空间 U 进行解耦，并在全空间 S 上稀疏化子空间，实现了更高效、可解释的注意力，并将 MCR^2 目标的梯度展开转化为 DMSA。

**🔧 技术方法**

使用 MCR^2 最优化、梯度展开、软阈值稀疏化、RoPE 位置编码以及多头注意力框架。

**📊 数据集**

在 ImageNet‑1K、CIFAR‑10/100、Oxford‑Pets、Food‑101、SVHN 等视觉分类数据集上进行评估。

**📈 对比分析**

与 ToST、MHSA 及线性注意力 TSSA 进行对比，DMST 在保持线性时间复杂度的同时，ImageNet‑1K top‑1 提升 1.08%–1.45%，内存占用降低 21%，在小规模数据集上亦保持或提升准确率。

**⚠️ 局限性**

仍受 MCR^2 对编码率估计的假设限制；在极大规模序列或非视觉任务上的适用性和可扩展性待进一步验证。

---

## 8. Retell, Reward, Repeat: Reinforcement Learning for Narrative Theory-Informed Story Generation

**arXiv ID:** 2601.17226 | [PDF](https://arxiv.org/pdf/2601.17226v1)

**作者:** David Y. Liu `[一作]` (University of New South Wales), Sebastian Sequoiah-Grayson `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将 Todorov 的叙事平衡理论与 d‑RLAIF 强化学习相结合，完成了对 LLM 的后训练，使其生成的故事在逻辑性、合理性和叙事完整度方面显著提升。

**💡 创新点**

创新点在于首次将叙事理论作为奖励信号源，并使用 LLM‑as‑judge 自动生成多维评估，突破传统监督微调在创造性与多样性上的局限。

**🔧 技术方法**

主要技术包括 d‑RLAIF、GRPO、LoRA、LLM‑as‑judge（Selene‑1‑mini、M‑Prometheus、Gemini‑3‑Flash）以及多尺度奖励信号（RO、RO5、RN）来优化故事生成模型。

**📊 数据集**

使用 TimeTravel 数据集的反事实重述任务（n≈200 进行评注，n≈1871 进行测试）作为实验数据源。

**📈 对比分析**

在与 SFT、指令微调模型对比后，d‑RLAIF 在 minLRC、Narrativity 等指标上接近人类表现，整体性能优于 SFT；但在 BLEU‑4、ROUGE‑L 等文本相似度指标上仍略逊于 SFT。

**⚠️ 局限性**

局限性包括：评估主要依赖 LLM‑as‑judge，缺乏与人类判断的直接对照；实验仅覆盖 7/8B 级模型，未验证更大或更小模型的效果；数据与理论均以西方叙事为主，可能缺乏跨文化通用性。

---

## 9. Measuring Political Stance and Consistency in Large Language Models

**arXiv ID:** 2601.17016 | [PDF](https://arxiv.org/pdf/2601.17016v1)

**作者:** Salah Feras Alali `[一作]` (Qatar University), Saban Kardas `[通讯]` (Qatar University)

**通讯引用:** 246 | [OpenAlex ID](https://openalex.org/A5037884669)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了九种大型语言模型在24个政治议题上的立场，并研究了五种提示策略对立场变化的影响。

**💡 创新点**

创新点在于跨模型、跨议题的系统性比较，并揭示语言提示是影响模型立场的最有效方式，进一步说明模型持久性与开发国背景相关。

**🔧 技术方法**

使用了直接提问、提供相反论点、双方论点、问题表述变化以及不同语言翻译等五种提示技术，并利用二级LLM自动判定模型输出的政治立场。

**📊 数据集**

数据集由24个政治议题的多种提示（共3117条模型响应）构成，其中包括三种语言版本；并未使用公开的标准数据集，而是自构造的提示与响应集合。

**📈 对比分析**

比较方法为统计模型在不同提示下立场变化次数与基线一致性；结果显示GroK‑3‑mini最稳定，Mistral最易变，语言翻译提示可导致约30%议题的立场翻转。

**⚠️ 局限性**

局限在于提示覆盖面有限、论证深度不足、机器翻译误差、自动立场判定的误差以及未考虑更丰富的多轮对话或角色扮演等提示形式。

---

## 10. Interpreting Agentic Systems: Beyond Model Explanations to System-Level Accountability

**arXiv ID:** 2601.17168 | [PDF](https://arxiv.org/pdf/2601.17168v1)

**作者:** Judy Zhu `[一作]` (Vector Institute for Artificial Intelligence), Dhanesh Ramachandran `[通讯]` (Vector Institute for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估并综述现有解释技术在Agentic系统中的适用性，提出系统级解释性框架与方法改进方向。

**💡 创新点**

指出传统解释方法无法捕捉Agentic系统的动态、交互和长期因果关系，提出将解释视为人机交互过程、构建系统级解释基础设施的创新思路。

**🔧 技术方法**

综述LIME、SHAP、saliency maps、TCAV、integrated gradients等传统解释技术，并讨论其与Agentic系统模块（感知、规划、内存、协调等）的关联与不足。

**📊 数据集**

无专门数据集；研究基于文献回顾、案例分析和理论讨论。

**📈 对比分析**

无实验对比或性能评估；通过案例与经验阐释方法适用性与缺陷。

**⚠️ 局限性**

局限性包括：缺乏统一的系统级解释工具；解释与底层机制缺乏可证实性；并发执行与多代理交互导致因果链难以追踪；解释的可重复性与可验证性不足。

---

## 11. Trademark Search, Artificial Intelligence and the Role of the Private Sector

**arXiv ID:** 2601.17072 | [PDF](https://arxiv.org/pdf/2601.17072v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 12. Scientific Image Synthesis: Benchmarking, Methodologies, and Downstream Utility

**arXiv ID:** 2601.17027 | [PDF](https://arxiv.org/pdf/2601.17027v1)

**作者:** Honglin Lin `[一作]` (Shanghai Jiao Tong University), Lijun Wu `[通讯]` (OpenDataLab Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于逻辑驱动的科学图像生成框架ImgCoder和专门的评测基准SciGenBench，系统评估了像素级与代码级生成方法在科学图像上的表现；

**💡 创新点**

核心创新在于将“理解→规划→编码”工作流与程序化渲染相结合，拆分推理与渲染；构建了结合LLM评审与逆向验证的多维度评测体系；并验证了高保真合成图像可持续提升多模态推理性能；

**🔧 技术方法**

采用链式思考规划、程序化代码生成（Python/Matplotlib等）、Diffusion T2I模型、LLM-as-Judge、逆向检验、传统图像指标等多种技术；

**📊 数据集**

使用经可视化过滤的科学说明语料（如SciDocs、公开科研文本）以及从SciGenBench-SeePhys抽取的真实图像，构成1.4K题目、5学科、25图像类型的综合数据集；

**📈 对比分析**

与闭源与开源像素级T2I模型对比，ImgCoder在逆向验证率上可达约77.9%（高于73.4%），LLM评审在结构相关维度得分更高；视觉指标（FID/PSNR）虽不佳，但对下游多模态推理的Fine‑Tune提升约3.7分，且显示出对数据规模的对数线性增长；

**⚠️ 局限性**

局限性包括对高度纹理化或富视觉细节任务（如生物图、化学反应图）表现不如像素级模型；程序化方法受限于代码正确性与执行环境，生成图像多为简化示意；高频噪声与域差异仍需进一步缓解。

---

## 13. pyBiblioNet: a Python library for a comprehensive network-based bibliometric analysis

**arXiv ID:** 2601.16990 | [PDF](https://arxiv.org/pdf/2601.16990v1)

**作者:** Mirko Lai `[一作]` (Università del Piemonte Orientale), Giancarlo Ruffo `[通讯]` (Università del Piemonte Orientale)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发并发布了名为PyBibliometrics的Python库，用于基于网络的综合文献计量分析，支持从OpenAlex自动下载数据、构建引文网络、合作网络和关键词共现网络，并提供可视化与多种网络算法工具。

**💡 创新点**

创新点在于将开放API（OpenAlex）与网络分析深度集成，提供端到端的工作流——从主题选择、数据预处理、根集/基集构建到网络构建、可视化和高级网络指标计算；同时加入NLP技术对关键词与概念进行自动提取，弥补传统文献计量工具对语义层面的不足。

**🔧 技术方法**

使用的主要技术包括Python编程、OpenAlex RESTful API、NetworkX/igraph/Graph-tool等网络分析库、Matplotlib/Plotly/Gephi用于可视化、spaCy/word2vec等NLP工具；库还实现了中心性度量、聚类系数、社区检测（Louvain、Infomap等）以及语义网络构建。

**📊 数据集**

主要数据集来源于OpenAlex公开的全球科研元数据，示例分析聚焦于跨学科的“15分钟城市范式”主题，涵盖了相关领域的期刊文章、会议论文和专利等。

**📈 对比分析**

在示例实验中，作者通过PyBibliometrics生成引文与合作网络，并用网络指标对关键作者、期刊及研究热点进行排序；虽然未提供与其他商业工具（如VOSviewer、ResearchRabbit）直接的量化对比，但展示了库在数据获取、网络构建及可视化方面的完整流程，并指出可进一步通过实验评估算法效率与准确性。

**⚠️ 局限性**

局限性主要包括：1) 依赖OpenAlex数据，受限于其覆盖范围与更新频率；2) 未对大规模数据的计算性能做系统评估；3) NLP分析在多语言环境下的鲁棒性尚未充分验证；4) 目前可视化功能相对基础，需进一步与专业工具对齐。

---

## 14. Ensuring Computer Science Learning in the AI Era: Open Generative AI Policies and Assignment-Driven Written Quizzes

**arXiv ID:** 2601.17024 | [PDF](https://arxiv.org/pdf/2601.17024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 15. FP-THD: Full page transcription of historical documents

**arXiv ID:** 2601.17040 | [PDF](https://arxiv.org/pdf/2601.17040v1)

**作者:** H Neji `[一作]`, FJ García-Marco `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了FP‑THD流水线，用于将古典拉丁文本的图像转录为完整的可编辑文本，保留所有特殊字符、连写字形和缩写符号。

**💡 创新点**

创新点在于将已验证的ParseNet布局分析模型与扩展的Masked Autoencoder‑Vision Transformer（MAE‑ViT）OCR相结合，采用span‑masking和SAM优化，完全不依赖后处理或语言模型即可高精度识别手写和印刷文本，并在输出时直接生成PAGE XML、Markdown和TXT文件。

**🔧 技术方法**

技术栈包括：ParseNet（CNN+encoder–decoder）进行文本行检测与布局分析；MAE‑ViT（ResNet‑18特征提取 + Vision Transformer + span masking）用于行级文字识别；Sharpness‑Aware Minimization（SAM）优化器；无语言模型或后处理；输出采用PAGE XML、Markdown、TXT。

**📊 数据集**

使用了三大数据集：手写文本的Rodrigo（853页）和Bentham（9,198行）数据集；自制的印刷拉丁文本Molino（143页，含缩写和现代拉丁）数据集；以及公开的评测基准（Pero‑OCR、ABBY/BVPB）。

**📈 对比分析**

与Pero‑OCR和ABBY/BVPB基线在Molino测试集上比较，CER从0.0242/0.3379降至0.0178，WER从0.2106/0.6835降至0.0450，表现最优；在手写数据上与Deep CRNN、HTR‑JAND比较，CER分别为1.30%/4.46%和2.02%/4.46%，显示出更高的识别精度且不使用后处理；整体性能显著提升且保持端到端学习。

**⚠️ 局限性**

局限性：对页边注释、单词行或极短行的识别仍不稳定；模型训练以长文本行为主，导致对单词级图像识别效果不足；需要扩展词级别数据、提升对多列复杂布局和噪声强烈图像的鲁棒性。

---

## 16. Toward Risk Thresholds for AI-Enabled Cyber Threats: Enhancing Decision-Making Under Uncertainty with Bayesian Networks

**arXiv ID:** 2601.17225 | [PDF](https://arxiv.org/pdf/2601.17225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 17. Real-Time, Energy-Efficient, Sampling-Based Optimal Control via FPGA Acceleration

**arXiv ID:** 2601.17231 | [PDF](https://arxiv.org/pdf/2601.17231v1)

**作者:** Tanmay Desai `[一作]` (Colorado School of Mines), R. Iris Bahar `[通讯]` (Colorado School of Mines)

**通讯引用:** 3668 | [OpenAlex ID](https://openalex.org/A5047635410)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种面向FPGA的MPPI控制器，利用深度流水线和并行数据流显著降低延迟与能耗。

**💡 创新点**

将MPPI算法从CPU/ GPU迁移到FPGA的硬件软件协同设计，揭示并行可行性并构建全流程流水线架构。

**🔧 技术方法**

使用Xilinx Vitis HLS、XOR‑Shift PRNG + Box‑Muller、CORDIC、深度流水线、片上BRAM划分与DSP块等技术。

**📊 数据集**

在赛道路径跟踪任务上，采用五条随机生成的赛车轨道与11个起始点共55个实验。

**📈 对比分析**

与NVIDIA Jetson Orin Nano GPU（1020 MHz，1024C）和6核Arm CPU对比，FPGA实现平均延迟2.33 ms、能耗14.9 mJ，速度提升3.1–7.5×，能耗降低2.5–5.4×，EDP提升40×。

**⚠️ 局限性**

受限于伪随机数生成的精度、缺乏固定点优化以及未在真实机器人上验证。

---

## 18. Fluxamba: Topology-Aware Anisotropic State Space Models for Geological Lineament Segmentation in Multi-Source Remote Sensing

**arXiv ID:** 2601.17288 | [PDF](https://arxiv.org/pdf/2601.17288v1)

**作者:** Jin Bai `[一作]` (University of Chinese Academy of Sciences), Atta ur Rahman `[通讯]` (University of Peshawar)

**通讯引用:** 23098 | [OpenAlex ID](https://openalex.org/A5017993242)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量级的Topology-Aware State Space Model（Fluxamba）用于遥感图像中地质线状特征的精确分割。

**💡 创新点**

创新点在于：① 将传统SSM的固定扫描路径替换为可微分的拓扑自适应信息流（Prior‑Modulated Flow）与Anisotropic Structural Gate，解决了线状目标与序列化路径之间的拓扑失配；② 引入Hierarchical Spatial Regulator与High‑Fidelity Focus Unit，实现多尺度语义对齐与噪声抑制；③ 通过四向选择性扫描（FS2D）实现实时性与高精度的平衡。

**🔧 技术方法**

技术手段包括：State Space Models（Mamba架构）、可微分几何编码（ASG）、动态权重调度（PMF）、层次特征对齐（HSR）、纯滤波关注单元（HFFU）、边界调制融合（BMF）以及混合损失（WBCE+Dice+Boundary）。

**📊 数据集**

使用三大遥感地质数据集：LROC‑Lineament（月球线状特征）、LineaMapper（欧罗巴冰壳线状特征）和GeoCrack（土壤岩石裂缝），并进行零样本迁移评估。

**📈 对比分析**

与八种SOTA方法（CNN、ViT、Mamba、SCSegamba等）对比，Fluxamba在三组数据集上实现最高mIoU（89.87%）、F1（89.22%）且仅有3.39M参数、6.25G FLOPs、24.12 FPS，显著提升了效率-精度 Pareto 前沿。

**⚠️ 局限性**

局限性包括：对极高频曲率线条的细节捕捉可能受限于四向基向量的离散；目前仅处理二维投影，无法直接扩展到三维体积数据；模型对极端纹理变化的泛化仍有提升空间。

---

## 19. Arabic Sign Language Recognition using Multimodal Approach

**arXiv ID:** 2601.17041 | [PDF](https://arxiv.org/pdf/2601.17041v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 20. Beyond Instrumental and Substitutive Paradigms: Introducing Machine Culture as an Emergent Phenomenon in Large Language Models

**arXiv ID:** 2601.17096 | [PDF](https://arxiv.org/pdf/2601.17096v1)

**作者:** Yueqing Hu `[一作]` (Institute of Neuroscience), Kaiping Peng `[通讯]` (Tsinghua University)

**通讯引用:** 17999 | [OpenAlex ID](https://openalex.org/A5103132724)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对比美国和中国开发的LLM在多模态任务中的文化表现，系统评估了模型来源和提示语言对文化模式的影响。

**💡 创新点**

提出“机器文化”框架，发现文化逆转和服务角色伪装两种新现象，挑战传统的工具性和替代性范式。

**🔧 技术方法**

使用多模态LLM（GPT‑4o、ERNIE 4.5、DALL‑E 3、iRAG‑1.0）与自动评估模型（Gemini‑2.5‑Pro、Qwen‑VL‑Max、DeepSeek‑V3、Claude 4.5 Sonnet）进行生成、解析与统计分析。

**📊 数据集**

通过官方API自动收集的LLM生成内容和文本输出，未使用公开标注集，仅基于模型自我生成的数据。

**📈 对比分析**

采用 2×2 factorial 设计，利用 t 检验、ANOVA、卡方检验等统计方法比较模型来源与提示语言效应；结果显示两者均未按人类文化模式一致，表现不稳定，表明两种范式失效。

**⚠️ 局限性**

仅对单一模型版本、单一提示、缺乏人类标注可靠性；未探究隐含结构和模型演化；对提示工程敏感性未充分评估。

---

## 21. DF-RAG: Query-Aware Diversity for Retrieval-Augmented Generation

**arXiv ID:** 2601.17212 | [PDF](https://arxiv.org/pdf/2601.17212v1)

**作者:** Saadat Hasan Khan `[一作]` (George Mason University), Daben Liu `[通讯]` (Capital One)

**通讯引用:** 577 | [OpenAlex ID](https://openalex.org/A5069329694)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在检索增强生成（RAG）中动态引入多样性的框架DF‑RAG，用来提升推理密集型问答任务的效果。

**💡 创新点**

创新点包括：①将最大边际相关性（MMR）改造成gMMR检索函数，显式平衡相关性与多样性；②通过Planner和Evaluator两阶段LLM推理，动态预测每个查询最优的多样性参数λ，无需任何微调；③设计Oracle上限评估，证明DF‑RAG可逼近理论最佳性能。

**🔧 技术方法**

技术手段：gMMR检索、LLM基的Planner（问题拆解）、Evaluator（评估检索集支持度）与Generator；使用few‑shot/one‑shot提示式LLM调用；采用Token数常数的RAG架构；在Qwen 2.5 72B和Llama 3.3 70B上实现。

**📊 数据集**

实验数据集：LongBench的多跳QA（HotpotQA、MuSiQue、2WikiMultihopQA）、非多跳LongBench（MultifieldQA）以及∞Bench的En.QA。

**📈 对比分析**

评价方式：使用F1得分与官方指标对比。相较于Vanilla RAG、固定λ gMMR、LongRAG、RAPTOR、HippoRAG等基线，DF‑RAG在多跳QA上提升约8–10% F1，在非多跳QA提升约4%；在某些基准上覆盖了高达91.3% Oracle与Vanilla RAG差距，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性：实验仅在两种LLM上进行；样本规模与数据集种类有限；未与训练依赖方法（如Self‑RAG）做深入对比；Planner和Evaluator高度依赖LLM和提示质量；多语言、偏见检测等方面尚未深入探索。

---

## 22. The Global Majority in International AI Governance

**arXiv ID:** 2601.17191 | [PDF](https://arxiv.org/pdf/2601.17191v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 23. Rethinking Benchmarks for Differentially Private Image Classification

**arXiv ID:** 2601.17189 | [PDF](https://arxiv.org/pdf/2601.17189v1)

**作者:** Sabrina Mokhtari `[一作]`, Gautam Kamath `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对差分隐私图像分类的标准化基准和公开leaderboard，聚焦医学影像，系统评估多种技术在不同隐私预算和预训练数据条件下的性能。

**💡 创新点**

创新点包括：①从传统的MNIST/CIFAR-10等公开数据转向隐私关键的医学影像基准；②构建可追踪、可验证的公开leaderboard；③系统对比多种DP训练技巧在医学影像上的泛化效果，揭示标准技巧在医学领域并非普适。

**🔧 技术方法**

使用的技术包括：DPSGD、梯度裁剪、数据归一化、组归一化、权重标准化、数据增强及多倍增强、指数移动平均、CLIP ViT（B/16、G/14）、Wide-ResNet、ScatterNet等模型架构。

**📊 数据集**

使用的数据集为：CheXpert（胸部X光）、EyePACS（糖尿病视网膜病变）、CIFAR-10、ImageNet（ImageNet-1K）及公开预训练数据集（ImageNet-1K、LAION-2B、WIT等）。

**📈 对比分析**

通过在不同ε（1、3、5、8）和不同预训练数据（无、ImageNet-1K、LAION-2B、WIT、‘anything goes’）条件下对比模型性能，报告AUC/ACC。结果显示：在高隐私下ScatterNet+线性优于预训练模型；低隐私下CLIP ViT‑G/14显著优于ScatterNet；CIFAR‑10上成功的技巧在医学影像上不易迁移。

**⚠️ 局限性**

局限性：仅针对两类医学影像（CheXpert、EyePACS）和有限的模型架构；未覆盖更大规模模型或其他任务（如NLP、生成模型）；对预训练数据的依赖性未深入解析；对多标签、少样本情况的细节探索不足。

---

## 24. Odd but Error-Free FastTwoSum: More General Conditions for FastTwoSum as an Error-Free Transformation for Faithful Rounding Modes

**arXiv ID:** 2601.17198 | [PDF](https://arxiv.org/pdf/2601.17198v1)

**作者:** Sehyeok Park `[一作]` (Rutgers University), Santosh Nagarakatte `[通讯]` (Rutgers University)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

**🎯 论文内容**

提出了在所有可信赖四舍五入模式下，FastTwoSum 的更一般的误差自由变换（EFT）充分条件，并给出针对 round-to-odd 的特定条件。

**💡 创新点**

设计了新的、更宽松的条件，使得 operands 的指数差可达近 2p-1，显著扩展了 EFT 的适用范围，并在 round-to-odd 模式下实现无符号错误。

**🔧 技术方法**

通过对浮点数的基数、精度、ulp、前导/后继数等概念的严谨分析，证明了一系列关于 δ∈𝔽 的充分条件，并在此基础上构造了符合 round-to-odd 的错误自由拆分算法 ExtractScalar。

**📊 数据集**

未使用实验数据，主要是理论证明。

**📈 对比分析**

与以往基于 e_a-e_b≤p 的条件相比，新的条件在指数差上几乎翻倍，理论上可处理更大范围输入；实验未给出具体性能，但理论覆盖面更广。

**⚠️ 局限性**

仍需在硬件实现上支持 round-to-odd；对于非对称指令集及 FMA 仍需进一步扩展，且理论证明基于有限浮点模型，实际机器可能出现额外边界误差。

---

## 25. Single-Pixel Vision-Language Model for Intrinsic Privacy-Preserving Behavioral Intelligence

**arXiv ID:** 2601.17050 | [PDF](https://arxiv.org/pdf/2601.17050v1)

**作者:** Hongjun An `[一作]` (Institute of Artificial Intelligence), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于单像素成像与视觉‑语言模型（SP‑VLM）的隐私友好型监控框架，能够在不获取高分辨率图像的前提下识别隐私敏感空间中的异常行为、人数统计和活动识别。

**💡 创新点**

创新点在于将低维单像素感知与视觉‑语言推理相结合，既能在低采样率下消除身份信息，又能在 2000–3000 次采样率区间内恢复行为语义，实现隐私保护与行为智能的兼顾。

**🔧 技术方法**

采用单像素成像（SPI）采集桶检测信号，配合量化的光学测量模型、白化处理、稀疏正则恢复以及以 8B 视觉‑语言模型为骨干的端到端推理网络。

**📊 数据集**

使用基于 AI 生成的合成隐私敏感行为数据集（2000 张图像，涵盖公共洗手间、更衣室和共享淋浴设施），以及 LFW 人脸数据库验证身份不可识别性。

**📈 对比分析**

与参数相同的 Qwen3‑VL‑8B 进行对比，在 4000 次采样下，异常检测准确率达 76.7%，人群计数 MAE 0.69，场景与活动识别 ROUGE‑L 超过 0.62，且在低于 3000 次采样时人脸识别准确率几乎为零，证明了方法在保持隐私的同时具备较好性能。

**⚠️ 局限性**

局限性在于实验主要基于合成数据，缺乏真实世界环境的多样性和复杂性；采样率区间受硬件和模型的特定参数限制，未在多种设备上验证普适性。

---

## 26. Failing on Bias Mitigation: Investigating Why Predictive Models Struggle with Government Data

**arXiv ID:** 2601.17054 | [PDF](https://arxiv.org/pdf/2601.17054v1)

**作者:** Hongbo Bo `[一作]` (University of Bristol), Weiru Liu `[通讯]` (University of Bristol)

**通讯引用:** 5231 | [OpenAlex ID](https://openalex.org/A5002349071)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在政府数据（英国布里斯托市犯罪预测数据）上应用机器学习模型时的偏差与公平性问题，并系统评估了常用的偏差缓解技术的有效性。

**💡 创新点**

揭示了政府数据本身的分布漂移、历史累积偏差和数据发布时间滞后等根本属性导致偏差缓解方法普遍失效；提出交叉特征（intersectional）公平性评估的重要性。

**🔧 技术方法**

使用了传统回归模型（MLP、决策树、随机森林、梯度提升、线性回归）和四种偏差缓解策略（过采样、MixUp、扰动、重加权），以及 MAE 与 R² 作为性能指标、ΔMAE 作为公平性度量。

**📊 数据集**

基于布里斯托市议会公开的犯罪率与人口统计数据（包括种族、宗教、免费学校餐补等特征），采用时间拆分（2016‑2021训练，2022测试）和随机拆分两种划分。

**📈 对比分析**

在两种数据拆分下，模型整体表现良好（R²≈0.9，MAE≈3–6），但大多数偏差缓解方法对公平性提升有限，只有少数组合（如过采样、MixUp 在随机拆分时）在部分特征上取得 25% 以上的改进，整体效果低于预期。

**⚠️ 局限性**

局限包括：只研究单一城市的数据，无法完全推广；评估仅限于单一敏感特征或简单交叉特征，未考虑更复杂多维交互；偏差缓解方法多为数据层面，缺乏对政策、事件驱动分布漂移的动态建模；缺乏对模型在实际政府决策中的可操作性和成本评估。

---

## 27. A Contrastive Pre-trained Foundation Model for Deciphering Imaging Noisomics across Modalities

**arXiv ID:** 2601.17047 | [PDF](https://arxiv.org/pdf/2601.17047v1)

**作者:** Yuanjie Gu `[一作]` (Tsinghua University), Biqin Dong `[通讯]` (Fudan University)

**通讯引用:** 12708 | [OpenAlex ID](https://openalex.org/A5101658700)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种名为Noisomics的通用框架，利用Contrastive Pre-trained (CoP) 模型对多模态成像系统中的噪声进行系统解码与量化

**💡 创新点**

创新地将噪声视为信息来源并构建噪声基因组，采用对比学习分离内容与噪声，实现零样本跨域泛化

**🔧 技术方法**

使用对比预训练、可分离编码器、可参数化量化头、SHAP解释、MMD/KDE评估等深度学习与统计技术

**📊 数据集**

在12个不同的公开数据集（胸部X光、组织病理、视网膜、阿尔茨海默MRI、显微镜、宇宙、卫星、可描述纹理、草图场景、漫画、水下、自然场景）以及手机图像和三光子显微镜数据上进行验证

**📈 对比分析**

与从零训练的基线相比，CoP将RMSE从0.058降至0.021（提升约63.8%），R²从0.456升至0.844（提升约85.1%），并在复杂噪声类型上显著提高分类精度

**⚠️ 局限性**

局限在于噪声基因组仅覆盖五种经典噪声类型，未涵盖更复杂或混合噪声，且对隐含噪声因子的解释仍不完整，训练成本较高

---

## 28. E2PL: Effective and Efficient Prompt Learning for Incomplete Multi-view Multi-Label Class Incremental Learning

**arXiv ID:** 2601.17076 | [PDF](https://arxiv.org/pdf/2601.17076v1)

**作者:** Jiajun Chen `[一作]` (Zhejiang University), Guanjie Cheng `[通讯]` (Zhejiang University)

**通讯引用:** 223 | [OpenAlex ID](https://openalex.org/A5076835003)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在缺失视图和不断增大的类别空间下，提出了IMvMLCIL任务，并提出一种结合任务特定提示和缺失感知提示的prompt学习框架。

**💡 创新点**

创新点在于将缺失视图提示的参数从指数级压缩到线性（通过高效原型张量化），并加入动态对比学习以捕捉不同缺失模式间的语义关系。

**🔧 技术方法**

使用的核心技术包括Transformer+prompt学习、原型张量化（EPT）和动态对比学习（DCL），并借助预训练模型进行特征提取。

**📊 数据集**

实验使用了三大多视图多标签数据集：ESPGame、IAPRTC12和MIRFLICKR，均人工引入缺失率以模拟真实场景。

**📈 对比分析**

与多种IMvMLC和MLCIL基线比较，E2PL在无记忆、200样本以及2/类回放设置下均实现了最高mAP、CF1和OF1，尤其在零记忆场景下显著优于所有对比方法。

**⚠️ 局限性**

局限性包括对极高缺失率下的性能衰减相对有限，以及对预训练Transformer和固定提示长度的依赖。

---

## 29. Multi-Agent Deep Reinforcement Learning Under Constrained Communications

**arXiv ID:** 2601.17069 | [PDF](https://arxiv.org/pdf/2601.17069v1)

**作者:** Shahil Shaik `[一作]` (Clemson University), Yue Wang `[通讯]` (Clemson University)

**通讯引用:** 31648 | [OpenAlex ID](https://openalex.org/A5100371961)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了完全分布式的多智能体强化学习框架 DG‑MAPPO，利用分布式图注意力网络（D‑GAT）实现多跳消息传递，无需集中式训练或全局观测。

**💡 创新点**

首次在完全分布式设置下消除 CTDE 对全球信息的依赖，结合 D‑GAT 的自适应注意力与分布式 SGD 一致化正则，实现在局部通信下可近似全局状态并训练全局价值函数。

**🔧 技术方法**

使用分布式图注意力网络（D‑GAT）、PPO 的分布式 Actor‑Critic、D‑SGD 参数同步、注意力正则化、全局奖励平均化以及多跳消息传递等技术。

**📊 数据集**

在 StarCraftII 多智能体挑战（SMAC）、Google Research Football、Multi‑Agent MuJoCo 等基准环境上进行评估。

**📈 对比分析**

与 MAPPO、HAPPO、MAT‑Dec 等强 CTDE 基线对比，DG‑MAPPO 在多种任务中实现或超越 CTDE 性能，尤其在稀疏通信或大规模队伍下仍保持高胜率。

**⚠️ 局限性**

受限于通信拓扑与跳数的设置，过度稀疏或极低跳会导致全局状态推断不足；同时 D‑GAT 参数同步需要额外通信开销，且在异构动态环境下鲁棒性尚待进一步验证。

---

## 30. Dynamic Role Assignment for Multi-Agent Debate

**arXiv ID:** 2601.17152 | [PDF](https://arxiv.org/pdf/2601.17152v1)

**作者:** Miao Zhang `[一作]` (New York University), Cheng Cao `[通讯]` (Amazon)

**通讯引用:** 2260 | [OpenAlex ID](https://openalex.org/A5101509118)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Meta‑Debate 框架，在多代理辩论前进行元辩论，动态为每个角色分配最合适的 LLM/ VLM；

**💡 创新点**

通过两阶段元辩论（提议生成+同伴评审）并自动生成角色评估标准，实现问题级别的能力感知动态角色分配，超越传统静态或随机分配；

**🔧 技术方法**

角色专属提议生成、同伴评审、基于 LLM 的自动评估标准生成、平均分数选取，以及与现有 MAD/DMAD 等辩论框架的集成；

**📊 数据集**

GPQA、MathVision 与 RealWorldQA 三个多模态/推理基准；

**📈 对比分析**

与统一模型、随机分配及现有 MAD/DMAD 框架对比，动态分配在三大基准上平均提升约 20‑30%（最高提升 74.8%），并显著降低结果方差；

**⚠️ 局限性**

受限于底层模型的根本能力；并且需要额外的提议生成与评审步骤，导致算力与 token 成本上升。

---

## 31. Data-Efficient Meningioma Segmentation via Implicit Spatiotemporal Mixing and Sim2Real Semantic Injection

**arXiv ID:** 2601.17031 | [PDF](https://arxiv.org/pdf/2601.17031v1)

**作者:** Yunhao Xu `[一作]` (Shenzhen Institute of Advanced Technology), Juan Yu `[通讯]` (Department of Radiology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种双重数据增强框架，将 Implicit Neural Representations (INR) 的空间混合与语义注入相结合，用于提升脑膜瘤分割的鲁棒性。

**💡 创新点**

创新点在于将连续可微分形变的 INR 与基于 Sim2Real 的真实病灶纹理注入解耦融合，实现解剖多样性与病灶语义丰富性的同时生成高保真合成样本。

**🔧 技术方法**

采用 INR 生成连续 SVF 并通过 ODE 积分实现可微形变、距离场融合的语义注入、Sim2Real 纹理映射，以及在 nnU-Net、U-Mamba、Swin-UMamba、LKM-UNet 上的多模型训练。

**📊 数据集**

使用 73 例深圳二院脑膜瘤训练集、20 例独立测试集和 362 例 IXI 健康 T1 数据作为背景进行合成。

**📈 对比分析**

在 20 例测试集上与仅用真实数据的基线对比，所有 SOTA 模型的 Dice 均提升至 70.6–80.1%（提升 5.5–14.9%），HD95 降至 8.55–9.06 mm（下降 50–60%）。

**⚠️ 局限性**

局限性包括语义注入未能充分模拟大肿瘤的质量效应，INR 解算过程计算开销较大，以及完全合成数据在性能上仍略逊于真实数据。

---

## 32. Spatiotemporal Semantic V2X Framework for Cooperative Collision Prediction

**arXiv ID:** 2601.17216 | [PDF](https://arxiv.org/pdf/2601.17216v1)

**作者:** Murat Arda Onsu `[一作]` (University of Ottawa), Sean Kennedy `[通讯]` (Nokia Bell Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在数字孪生交通环境下，构建 RSU 通过 V-JEPA 生成未来帧的语义嵌入，并通过 V2X 传输给车辆进行碰撞预测。

**💡 创新点**

创新点在于使用 V-JEPA 进行未来帧嵌入预测、结合数字孪生数据生成以及仅传输语义嵌入实现四阶压缩的实时碰撞预警。

**🔧 技术方法**

技术包括视频联合嵌入预测架构 V-JEPA、轻量注意力探针、YOLOv11 视觉后处理、数字孪生仿真 QLabs、V2X 语义通信。

**📊 数据集**

数据集为使用 Quanser Interactive Labs 生成的 500 条交通视频，其中 115 条包含碰撞场景。

**📈 对比分析**

通过对比不同后处理方法及嵌入格式，模型在无后处理下 F1=76%，热图+二值掩码提升至 84%，并将通信负载压缩到原始视频的四个数量级，传输时延显著下降。

**⚠️ 局限性**

局限性包括仅在仿真环境验证，缺乏真实道路测试；模型仅预测二分类，无法给出具体碰撞时间点；依赖高质量数字孪生数据，部署成本高。

---

## 33. The Relativity of AGI: Distributional Axioms, Fragility, and Undecidability

**arXiv ID:** 2601.17335 | [PDF](https://arxiv.org/pdf/2601.17335v1)

**作者:** Angshul Majumdar `[一作]` `[通讯]`, Angshul Majumdar

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

在理论上给出了一套基于分布式、资源受限的公理化定义，阐明了“通用人工智能”(AGI)必须被视为相对于任务分布、性能指标和资源预算的关系性谓词，并证明了该定义的不可确定性与不可自我证明性。

**💡 创新点**

创新点包括：
① 将 AGI 作为分布式公理化属性而非抽象概念；
② 证明 AGI 在分布变动下不具不变性、转移受限且无法提供普适鲁棒性；
③ 运用 Rice 定理和 Gödel–Tarski 结果，证明任何资源受限的自我认证程序都不可决定 AGI 属性；
④ 明确阐述 AGI 评估必须给出任务分布和预算，消除“AGI 时间点”的语义错误。

**🔧 技术方法**

使用的技术主要是：
- 形式化公理化框架（定义公理 G1–G5、A1–A4）；
- 概率与信息论工具（期望、尾量、总变距离、相互信息泛化界）；
- 计算理论（Rice 定理、可判定性分析、Tarski 语义不定理）。

**📊 数据集**

论文不涉及实验数据集，而是基于抽象的任务族 𝒯 和任务分布 μ 的理论构造，利用一般的概率测度和信息论不等式进行证明。

**📈 对比分析**

没有对齐传统机器学习方法进行实验比较；评价方式是通过理论证明给出的不变量、鲁棒性与自证性的上界，说明现有大模型的“通用性”仅在特定分布下有效，无法保证跨分布或自证。

**⚠️ 局限性**

局限性：
- 结果以理论形式给出，缺少经验验证；
- 依赖于对任务族和预算的有效编码假设；
- 未给出具体的可实现指标或实验验证；
- 讨论侧重于不可判定性和分布依赖性，对实际工程系统的直接指导有限。

---

## 34. A Computer Vision Pipeline for Iterative Bullet Hole Tracking in Rifle Zeroing

**arXiv ID:** 2601.17062 | [PDF](https://arxiv.org/pdf/2601.17062v1)

**作者:** Robert M. Belcher `[一作]` (United States Military Academy), Christopher J. Lowrance `[通讯]` (United States Military Academy)

**通讯引用:** 180 | [OpenAlex ID](https://openalex.org/A5014517592)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一套基于计算机视觉的完整管线，利用YOLOv8检测步枪靶面上的子弹孔，并通过IoU匹配实现不同射击轮次的子弹孔跟踪，从而辅助零点调准；

**💡 创新点**

创新点包括：①将IoU用作跨帧匹配函数以实现子弹孔迭代识别；②提出“去除”式数据增强，通过从真实图像中删减子弹孔生成逼真的连续射击序列；③使用ORB特征与单应性完成透视校正，使不同视角的靶面保持统一姿态；

**🔧 技术方法**

采用的技术包括YOLOv8目标检测、ORB特征匹配与单应性透视校正、基于IoU的对象匹配、两步分割策略（ORB+YOLOfallback）、数据增强（对象去除）以及评估指标mAP50/95、Jaccard指数、迭代分类准确率；

**📊 数据集**

使用的数据集包含：1）280张实测射击图像（Fort Benning）无序列注释；2）1243张公开标注子弹孔数据；3）33对人工与实弹生成的两轮序列（22支纸靶、11支实弹靶）；4）通过合成与增强生成的连续射击序列；

**📈 对比分析**

与公开文献（如YOLOv8在子弹孔检测中mAP50 96.7%）对比，本工作在归一化图像上达到了97.0% mAP50，IoU匹配实现了88.8%的迭代分类准确率，Jaccard指数为0.88，整体管线准确率为40.3%；检测精度与最先进方法相当或略优，且首次实现了系统化的子弹孔迭代跟踪；

**⚠️ 局限性**

局限性包括：①透视校正与分割成功率仅为73%，对重叠或被遮挡靶面表现不佳；②基于IoU的匹配对密集射击或重叠子弹孔可能产生误判；③缺乏多视角连续图像，系统对摄像机运动较为敏感；④完整管线准确率低，主要受分割、检测误差累积影响；⑤需要人工标注来训练和验证，增加了工作量。

---

## 35. SpecBridge: Bridging Mass Spectrometry and Molecular Representations via Cross-Modal Alignment

**arXiv ID:** 2601.17204 | [PDF](https://arxiv.org/pdf/2601.17204v1)

**作者:** Yinkai Wang `[一作]` (Tufts University), Soha Hassoun `[通讯]` (Tufts University)

**通讯引用:** 2285 | [OpenAlex ID](https://openalex.org/A5048817284)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SpecBridge，通过将 MS/MS 光谱嵌入映射到冻结的 ChemBERTa 分子表示空间，实现光谱到分子结构的检索。

**💡 创新点**

创新点在于采用“锁定-对齐”策略：只训练轻量化投影映射和少量光谱编码器参数，将光谱直接对齐到预训练分子空间，避免了端到端对齐与生成的高成本与不稳定性。

**🔧 技术方法**

使用 DreaMS 预训练光谱编码器、ChemBERTa 分子基础模型、残差投影映射和基于 MSE 的对齐损失，配合正交初始化和正交正则化。

**📊 数据集**

评估数据集包括 MassSpecGym、Spectraverse（PubChem 公式检索）和 MSnLib，覆盖从小型候选池到百万级候选池的检索场景。

**📈 对比分析**

与 DeepSets、JESTR、MVP、GLMR 等基线对比，SpecBridge 在 MassSpecGym 的 Recall@1 提升约 16%（84.7% vs 68.5%），在 Spectraverse 与 MSnLib 也实现显著提升（Recall@1 分别为 36.6% 与 53.4%），并将 MCES@1 降至 2.37，显示结构预测更为准确。

**⚠️ 局限性**

局限在于只能检索已存在的分子库，无法生成新结构；仅使用 2D 分子图，未考虑 3D 结构信息，且对极大候选集仍存在分辨率限制。

---

## 36. Thermodynamically Optimal Regularization under Information-Geometric Constraints

**arXiv ID:** 2601.17330 | [PDF](https://arxiv.org/pdf/2601.17330v1)

**作者:** Laurent Caraffa `[一作]` `[通讯]` (Universite Gustave Eiffel), Laurent Caraffa (Universite Gustave Eiffel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个将热力学、信息几何与机器学习正则化统一的理论框架，并证明在三条明确假设下 Fisher–Rao 正则化是热力学最优的。

**💡 创新点**

将 Landauer 原理与 Fisher–Rao 度量相结合，给出了正则化的根本能量下界，并揭示传统欧氏正则化的结构性低效性。

**🔧 技术方法**

采用信息几何（Cencov 定理）、最大熵原理、热力学能量下界（Landauer）以及对高斯和 von Mises 分布的几何推导等技术。

**📊 数据集**

本研究为理论性工作，未使用具体数据集。

**📈 对比分析**

通过理论预测与可实验验证的方式提出 Fisher–Rao 正则化在热力学效率上优于欧氏正则化，实验细节尚待进一步验证。

**⚠️ 局限性**

限制：结果仅在假设 A1–A3 成立时成立；实际学习过程往往非静态；模型假设为最大熵分布，难以直接应用于复杂深度网络。

---

## 37. ConceptACT: Episode-Level Concepts for Sample-Efficient Robotic Imitation Learning

**arXiv ID:** 2601.17135 | [PDF](https://arxiv.org/pdf/2601.17135v1)

**作者:** Jakob Karalus `[一作]` (Ulm University), Friedhelm Schwenker `[通讯]` (Ulm University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ConceptACT，利用演示阶段的 episode‑level 语义概念注释，通过在 ACT 模型的 Transformer 编码器中加入概念注意力机制，改进机器人模仿学习的样本与训练效率。

**💡 创新点**

核心创新在于：① 在 VAE‑Decoder 的最终 Transformer 层引入按概念类别划分的概念 Transformer，实现概念的自适应注意；② 通过监督注意权重而非单纯的预测头，形成更强的结构化先验；③ 仅在训练阶段使用概念注释，无需部署时额外输入。

**🔧 技术方法**

技术手段包括：Transformer 编码-解码架构、VAE 变分自编码、跨模态特征融合、概念注意力（Query‑Key‑Value 仅对 Key/Value 使用概念嵌入）、平均池化概念预测、交叉熵概念损失、KL 正则化以及整体多任务损失。

**📊 数据集**

实验数据集为两套自制机械操作任务：1）Sorting——根据形状与颜色进行逻辑分区；2）Ordering——按多重约束顺序放置物体。每个任务均提供了对象属性、空间关系等 episode‑level 概念标签。

**📈 对比分析**

与标准 ACT、ConceptACT‑Heads、LAV‑ACT（具体/通用）等对比，ConceptACT 在 50%–100% 训练数据下样本效率提升明显，学习曲线收敛速度加快约 40%（即 6,000 步提前完成），优化目标的 optimality gap 下降 42%。统计检验表明改进显著，平均提升 0.2–0.3 分（任务 1）和 0.4–0.5 分（任务 2）以上。

**⚠️ 局限性**

局限性包括：① 需要手工 episode‑level 概念标注，假设概念在整个演示过程中保持不变；② 目前仅验证于两种基于离散概念的抓取任务，未覆盖时间变化或连续概念；③ 对概念分布的依赖可能导致在未见过概念组合的任务上泛化受限；④ 仍需手动设计概念类别与维度，降低自动化程度。

---

## 38. Frequency-aware Adaptive Contrastive Learning for Sequential Recommendation

**arXiv ID:** 2601.17057 | [PDF](https://arxiv.org/pdf/2601.17057v1)

**作者:** Zhikai Wang `[一作]` (Fudan University), Weihua Zhang `[通讯]` (Fudan University)

**通讯引用:** 21926 | [OpenAlex ID](https://openalex.org/A5100370310)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 FACL（Frequency-aware Adaptive Contrastive Learning）框架，利用频率感知的微观自适应扰动和宏观加权重策略，对序列推荐中的对比学习进行改进，保护低频物品不被过度扰动并提升稀疏序列的学习权重。

**💡 创新点**

创新点包括：① 微观层根据物品/用户频率自适应调整扰动概率，降低低频项被误删/误置的风险；② 宏观层对序列进行频率与长度加权，提升稀疏/低频序列的损失贡献；③ 将两者结合于传统数据增强的对比学习框架，实现了多样性与保真度的平衡，显著提升了对长尾场景的鲁棒性。

**🔧 技术方法**

使用技术主要包括：Transformer‑based 序列编码（SASRec 作为骨干），对比学习（InfoNCE）损失，频率感知的自适应数据增强（drop、substitute、insert、crop、reorder），宏观加权重 λ(S_u) 的计算与应用，整体损失为 λ(S_u)(rec+τ·cl)。

**📊 数据集**

实验数据集为五个公开基准：Amazon Beauty、Amazon Sports、MovieLens‑1M、MovieLens‑20M、Yelp。

**📈 对比分析**

与 7 类基线（SASRec、CL4SRec、CoSeRec、CT4Rec、DuoRec、BSARec、RCL）以及不同模型骨干（FMLP）进行对比，评估指标为 HR@5/10 与 NDCG@5/10。FACL 在所有数据集上均优于所有对比方法，提升幅度约 3%–4%（最高 3.8%），并在低频物品/用户的准确率上显著改善。

**⚠️ 局限性**

局限性包括：① 仍依赖全局频率统计，未充分利用物品/用户的语义或内容特征；② 对长序列的自适应调整缺乏进一步细化；③ 需要额外存储频率表和关联项，规模极大时可能导致存储/计算开销；④ 目前仅验证于隐式反馈数据，未测试显式评分场景。

---

## 39. AMVICC: A Novel Benchmark for Cross-Modal Failure Mode Profiling for VLMs and IGMs

**arXiv ID:** 2601.17037 | [PDF](https://arxiv.org/pdf/2601.17037v1)

**作者:** Aahana Basappa `[一作]` (Algoverse AI Research), Kevin Zhu `[通讯]` (Algoverse AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过创建AMVICC基准，系统评估了多模态大语言模型（MLLMs）与图像生成模型（IGMs）在视觉推理与生成任务中的局限性，并对比了它们在同一视觉任务下的失败模式。

**💡 创新点**

创新点在于提出跨模态的失败模式评估框架——AMVICC，并通过将MMVP问答转化为显式与隐式提示，统一评估生成与解释两种任务，从而揭示两种模型共享及特有的错误类型。

**🔧 技术方法**

技术手段包括：利用MMVP基准改造生成提示、使用GPT‑4o自动化评估文本答案、人工双重检查生成图像、构造隐式/显式提示对照实验、对模型的随机性与词汇敏感性进行消融分析。

**📊 数据集**

数据集主要基于MMVP原始的300道视觉推理题，进一步扩充至600道手工设计的隐式与显式提示，用于评测11个MLLM和3个IGM在9类视觉推理子任务上的表现。

**📈 对比分析**

比较方法是计算单一与配对任务的准确率，阈值分别设为80%和70%；结果显示MLLM在文本与颜色感知上表现相对较好，而IGM在空间关系与数量计数方面表现较差，两个模态在数量计数和视角视点等类别存在共同的失败模式。

**⚠️ 局限性**

局限性包括：仅评测3个IGM导致代表性不足、提示设计偏差与人工评估主观性、模型闭源导致内部机制不透明、以及缺乏人类性能基准等。

---

## 40. NewPINNs: Physics-Informing Neural Networks Using Conventional Solvers for Partial Differential Equations

**arXiv ID:** 2601.17207 | [PDF](https://arxiv.org/pdf/2601.17207v1)

**作者:** Maedeh Makki `[一作]` (University of California Riverside), Behzad Mohebbi `[通讯]` (Procter and Gamble Service GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种将神经网络与传统数值求解器耦合的物理信息学习框架（NewPINNs），通过求解器一致性损失训练网络，代替残差加权的 PINN 方法。

**💡 创新点**

核心创新在于把求解器作为黑盒操作员，直接在训练循环中将网络输出推进到下一时间步或迭代，并以此差异作为损失；从而消除了残差加权、频谱偏差和求解器稳定性等传统 PINN 的缺点。

**🔧 技术方法**

技术包括：U‑Net/CNN 作为网络架构；有限体积、有限元和谱（Chebfun）求解器；闭环 “pull‑push” 训练策略；逆问题循环；以及针对稳态与瞬态的专门损失设计。

**📊 数据集**

实验使用三类求解器的标准基准数据：二维 Fokker–Planck 方程（参数 α∈[1,2]），二维舷面驱动腔体（Re∈[2000,3000]），以及 1D Allen–Cahn 与 Kuramoto–Sivashinsky 方程（α 取值区间）。数据来自高精度数值模拟（FVM、FEM、Chebfun）。

**📈 对比分析**

与传统 PINN、纯监督代理模型等方法对比，NewPINNs 在稳态流场、非线性扩散、以及有限时间内的混沌动力学中均能保持误差在 10⁻³–10⁻⁵ 级别，训练更稳定且无需手工调权重。

**⚠️ 局限性**

局限性包括：对长期混沌或高维系统的预测精度随时间增长而下降；受限于求解器精度；网络容量有限导致细节捕捉不足；初始化不当可能导致求解器不收敛或出现数值失败。

---

## 41. Evaluating the Evolution of Critical Thinking, Creativity, Communication and Collaboration in Higher Education Courses

**arXiv ID:** 2601.17018 | [PDF](https://arxiv.org/pdf/2601.17018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 42. Reasoning Beyond Literal: Cross-style Multimodal Reasoning for Figurative Language Understanding

**arXiv ID:** 2601.17197 | [PDF](https://arxiv.org/pdf/2601.17197v1)

**作者:** Seyyed Saeid Cheshmi `[一作]` (University of Minnesota), Dongyeop Kang `[通讯]` (University of Minnesota)

**通讯引用:** 1510 | [OpenAlex ID](https://openalex.org/A5040821714)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套三阶段框架，先用大型教师模型蒸馏链式推理，后在轻量级视觉语言模型上进行监督微调和强化学习，以提升多模态讽刺、幽默、攻击与隐喻的理解。

**💡 创新点**

将可验证奖励的强化学习与链式推理蒸馏相结合，首次在多模态隐喻任务中实现跨风格可解释推理，并证明轻量级模型可超越更大模型。

**🔧 技术方法**

使用大型VLM教师蒸馏、监督微调（SFT）、基于GRPO的强化学习（RLVR）、可验证奖励函数以及跨风格联合训练。

**📊 数据集**

采用MMSD2.0（讽刺）、Memotion（幽默、攻击）、MultiMET（隐喻）等多模态隐喻数据集，并结合V-FLUTE等公开资源。

**📈 对比分析**

与Gemini 2.5 Flash、Qwen2.5‑VL‑32B、LLaMA‑90B等大型模型在同一零样本推理下对比，联合训练的3B学生模型在讽刺、幽默、攻击上均超过大模型，并在多风格通用度上表现优异。

**⚠️ 局限性**

受教师推理质量限制，跨风格迁移受语义距离影响；奖励设计相对简单；缺乏OOD评估和少样本实验；对更多风格与数据集的扩展有限。

---

## 43. Sparsity-Aware Low-Rank Representation for Efficient Fine-Tuning of Large Language Models

**arXiv ID:** 2601.16991 | [PDF](https://arxiv.org/pdf/2601.16991v1)

**作者:** Longteng Zhang `[一作]` (Hong Kong University of Science and Technology), Xiaowen Chu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10124 | [OpenAlex ID](https://openalex.org/A5100730785)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SALR 方法，将低秩适配与幅度剪枝结合，在资源受限环境下实现大模型的高效微调。

**💡 创新点**

创新点包括：① 通过 MSE 分析证明仅对冻结基权进行静态剪枝最优；② 用截断 SVD 恢复残差以显著降低误差；③ 将多低秩适配器合并为单一 GEMM 并采用位图编码+两阶段流水线，实现真正压缩与加速。

**🔧 技术方法**

采用技术包括：LoRA 低秩适配、幅度剪枝、截断 SVD 残差低秩恢复、位图稀疏编码、GEMM 合并、两阶段解码+GEMM 流水线、NF4 量化等。

**📊 数据集**

使用数据集：MetaMath（数学推理）、GSM8K（数学问题）、MMLU（多领域问答）以及 ARC、MC‑TEST、OBQA、RACE 等多任务数据集。

**📈 对比分析**

与 LoRA、LoSA、SparseLoRA、DeepSparse 等方法对比，SALR 在 50% 稀疏下保持与 LoRA 相近的 GSM8K/MMLU 准确率；模型大小减半；推理速度提升 1.7×，训练时内存下降约 30%，吞吐量提升约 20%。

**⚠️ 局限性**

局限性：对极端稀疏（>70%）或更大模型仍可能出现性能下降；稀疏编码与解码仍带来额外开销；目前主要验证 N:M 半结构化稀疏模式，其他稀疏模式与多任务复杂度的适用性待进一步评估。

---

## 44. FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices

**arXiv ID:** 2601.17063 | [PDF](https://arxiv.org/pdf/2601.17063v1)

**作者:** Byeongju Kim `[一作]`, Sangyeob Kim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 FlashMoE 系统，实现了在设备内存受限的边缘环境下对大型 MoE 语言模型的高效推理，利用 SSD 对未激活的专家进行分层存储与按需加载，并结合 ML 预测缓存置换策略提升专家缓存命中率。

**💡 创新点**

创新点包括：①将专家层与非专家层按文件拆分，首次实现仅加载必要非专家权重以显著缩短模型初始化时间；②基于 LRU 与 LFU 递归与频次特征的轻量化前馈网络，逼近 Belady 最优置换，显著提升缓存命中率；③将专家按层级与单元拆分，以极细粒度从 SSD 预取，降低 I/O 负载；④通过实验验证该方案在 16GB RAM 环境下可获得 2.6× 的推理速度提升。

**🔧 技术方法**

采用了 SSD 存储、GPU VRAM 缓存、基于 PyTorch 的分块加载机制、轻量化前馈网络（ML 缓存预测器）、并行预取与异步置换等技术；实验环境包括 AMD Ryzen 9 CPU、RTX 5070 Ti GPU、PCIe 5.0 NVMe SSD。

**📊 数据集**

使用 TriviaQA 数据集进行推理路由记录收集和模型训练；验证集来自 TriviaQA 以评估推理吞吐量和缓存命中率；此外还使用 Qwen3‑30B‑A3B 与 OLMoE‑1B‑7B 两个 MoE 模型进行实验。

**📈 对比分析**

与 Fiddler、DAOP 等基于 DRAM 的 offloading 框架对比，FlashMoE 在相同硬件条件下初始加载时间提升 4–6.8 倍，推理吞吐量提升 7–22%（依赖模型与缓存大小），缓存命中率较 LRU 提升 21–51%（对应 22–35% I/O 降低）。

**⚠️ 局限性**

局限性包括：①依赖 SSD 的读写性能，若 SSD 性能不足会成为瓶颈；②当前实现仅支持单机 GPU，未考虑多 GPU 或分布式场景；③ML 缓存预测模型训练需要额外的路由日志收集与特征构造；④对极大规模模型（数百亿参数）在内存受限设备上的进一步压缩和部署仍需探索。

---

## 45. Equilibrium Refinements Improve Subgame Solving in Imperfect-Information Games

**arXiv ID:** 2601.17131 | [PDF](https://arxiv.org/pdf/2601.17131v1)

**作者:** Ondrej Kubicek `[一作]`, Tuomas Sandholm `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20123 | [OpenAlex ID](https://openalex.org/A5023571961)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

无可用信息

**💡 创新点**

无可用信息

**🔧 技术方法**

无可用信息

**📊 数据集**

无可用信息

**📈 对比分析**

无可用信息

**⚠️ 局限性**

无可用信息

---

## 46. A Constrained Optimization Perspective of Unrolled Transformers

**arXiv ID:** 2601.17257 | [PDF](https://arxiv.org/pdf/2601.17257v1)

**作者:** Javier Porras-Valenzuela `[一作]` (University of Pennsylvania), Alejandro Ribeiro `[通讯]` (University of Pennsylvania)

**通讯引用:** 15991 | [OpenAlex ID](https://openalex.org/A5078862959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种约束优化框架，用于训练表现得像优化下降算法的变换器，强制每层输出减少预期损失，并采用原始-对偶训练方案。

**💡 创新点**

创新点在于通过施加下降约束来训练变换器，使其在层间输出上保持单调下降，从而提高模型的鲁棒性和泛化能力。

**🔧 技术方法**

使用了原始-对偶训练算法，结合了约束学习理论，确保模型在层数增加时收敛到接近最优的统计损失值。

**📊 数据集**

在视频去噪和文本分类任务中应用了不同的数据集，包括CUHK Avenue、UCSD Anomaly Detection、ShanghaiTech Campus等。

**📈 对比分析**

与传统的无约束模型相比，约束变换器在处理输入扰动时表现出更强的鲁棒性，并在分布外泛化能力上有显著提升，同时保持了在分布内的性能。

**⚠️ 局限性**

限制在于约束学习问题的严格可行性假设不易保证，可能导致在某些情况下无法满足所有约束条件。

---

## 47. Cross360: 360° Monocular Depth Estimation via Cross Projections Across Scales

**arXiv ID:** 2601.17271 | [PDF](https://arxiv.org/pdf/2601.17271v1)

**作者:** Kun Huang `[一作]` (Victoria University of Wellington), Neil Dodgson `[通讯]` (Victoria University of Wellington)

**通讯引用:** 5356 | [OpenAlex ID](https://openalex.org/A5049195176)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于交叉注意力的 360° 单目深度估计方法 Cross360，能够在全视角下同时利用局部无畸变的切平面（TP）特征和全局等距投影（ERP）特征。

**💡 创新点**

创新点在于引入 CPFA（Cross Projection Feature Alignment）模块，通过跨投影注意力将 TP 与 ERP 对齐，使局部特征获得全局上下文；以及 PFAA（Progressive Feature Aggregation with Attention）模块，利用跨尺度注意力逐层聚合特征，显著降低了切平面拼接时的边界失真和全局不连续性。

**🔧 技术方法**

技术手段包括：多尺度卷积特征提取（基于 ResNet34）、自注意力和跨投影注意力机制、切平面与 ERP 的几何变换、Switchable 归一化、以及多尺度监督（MSE + 梯度损失）。

**📊 数据集**

实验使用四个公开基准数据集：Matterport3D、Stanford2D3D（真实世界不完整 ERP）、Structured3D、3D60（合成完整 ERP），并对比了视角图像方法和 360° 专用方法。

**📈 对比分析**

在所有数据集上，Cross360 均显著优于现有最优方法，例如在 Matterport3D 上相较 Elite360D 提升了 14.35% 的 Abs Rel、22.76% 的 Sq Rel，且在 Synthetic 数据上在 Abs Rel、Sq Rel、RMSE 上均领先 30% 以上，验证了方法的高效性与准确性。

**⚠️ 局限性**

主要局限包括：对缺失的 ERP 极区更为敏感，需要针对不同视角范围手动调整 TP 采样；以及整体模型仍属于 Transformer+卷积混合架构，参数量和推理速度相对较高，仍有进一步压缩和加速的空间。

---

## 48. Performance uncertainty in medical image analysis: a large-scale investigation of confidence intervals

**arXiv ID:** 2601.17103 | [PDF](https://arxiv.org/pdf/2601.17103v1)

**作者:** Pascaline André `[一作]` (Sorbonne Université), Olivier Colliot `[通讯]` (Sorbonne Université)

**通讯引用:** 12406 | [OpenAlex ID](https://openalex.org/A5005778444)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对医学影像 AI 性能的不确定性进行大规模实证研究，评估了多种置信区间（CI）方法在分割与分类任务中的可靠性（覆盖率）和精确度（宽度）。

**💡 创新点**

首次在医学影像领域系统比较 5 种常用 CI 方法（parametric t、z、bootstrap 的 percentile、basic、BCa，以及 Hoeffding 与 Empirical Bernstein 的分布无关 CI），揭示了不同任务、指标、聚合策略及样本量对 CI 行为的影响，并提出了多项关键经验（如分割比分类需要更小样本、BCa 在某些情况下失效、micro 聚合比 macro 更易收敛）。

**🔧 技术方法**

技术主要包括：核密度估计（KDE）拟合真实性能分布；对分割指标做连续/离散处理；多元 KDE 采样；多种 CI 计算（parametric、bootstrap、concentration inequalities）；统计分析（覆盖率、宽度、覆盖收敛速度 CCP、线性混合模型）。

**📊 数据集**

使用 16 任务的 Medical Segmentation Decathlon（MSD）和 12 任务的公开分类基准，涵盖 19 种模型，形成 228 个分割与 228 个分类基准实例，共 456 个模型-任务组合；指标包括 8 种分割指标（DSC、IoU、ASSD 等）和 6 种分类指标（Accuracy、AUC、F1 等）。

**📈 对比分析**

通过 10,000 次模拟样本量（10–250）评估 CI，结果显示：percentile bootstrap 在大多数设置下表现最稳健；BCa 在极大样本下失效；parametric t 在极小样本下略优；Hoeffding 与 Empirical Bernstein CI 宽度过大；分类任务需要 3–10 倍更大样本才能达到相同覆盖率或宽度；距离型分割指标（ASSD、MASD）收敛慢；宏平均比微平均更难收敛。整体而言，CI 方法在不同情境下表现差异显著，无法一概而论。

**⚠️ 局限性**

局限包括：仅使用 KDE 拟合分布，未做敏感性分析；仅评估了 5 种 CI 方法，未覆盖所有可能的 parametric/bootstrap 变体；任务范围限定为分割与分类，未扩展至检测、注册、生成等；模拟规模巨大但仅在 10–250 样本内探索，超大样本场景不足；对极度不平衡或多类别复杂场景的失效情况仍需进一步研究。

---

## 49. Federated Proximal Optimization for Privacy-Preserving Heart Disease Prediction: A Controlled Simulation Study on Non-IID Clinical Data

**arXiv ID:** 2601.17183 | [PDF](https://arxiv.org/pdf/2601.17183v1)

**作者:** Farzam Asad `[一作]` (Lahore Garrison University), Muhammad Adnan Khan `[通讯]` (Gachon University)

**通讯引用:** 11987 | [OpenAlex ID](https://openalex.org/A5009282836)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在UCI Heart Disease Cleveland子集上，通过人口统计学划分生成四个非IID的模拟医院客户端，使用Federated Proximal Optimization (FedProx) 对心脏病预测模型进行联邦训练并与中心化、局部模型以及FedAvg进行对比。

**💡 创新点**

创新点在于：①提出了基于年龄、疾病患病率等特征的可复现非IID划分方法；②系统性Ablation研究验证proximal参数μ对性能和收敛的影响；③证明FedProx在非IID环境下可超越中心化学习，提升模型准确性与公平性。

**🔧 技术方法**

主要技术包括：Federated Proximal Optimization、逻辑回归（Logistic Regression）与随机梯度下降、proximal正则化、加权平均聚合、批量化与学习率衰减。

**📊 数据集**

使用UCI Heart Disease数据集的Cleveland Clinic子集（293条完整记录）。

**📈 对比分析**

与中心化学习（83.33%）、局部模型（78.45%平均）和FedAvg（84.58%）比较，FedProx在μ=0.05时达到85.00%准确率，表现更好；同时收敛速度比FedAvg快18%，标准差更低，通信成本相近。

**⚠️ 局限性**

限制包括：①仅使用单一来源的模拟非IID划分，未验证真实多机构数据；②仅研究心脏病单一任务；③采用线性逻辑回归模型，未探讨深度学习；④样本量小（293条）；⑤客户端数有限（4个）；⑥未实现差分隐私或其他安全增强；⑦未考虑医疗设备差异、诊断标准差异等真实世界复杂性。

---

## 50. RAM-SD: Retrieval-Augmented Multi-agent framework for Sarcasm Detection

**arXiv ID:** 2601.17002 | [PDF](https://arxiv.org/pdf/2601.17002v1)

**作者:** Ziyang Zhou `[一作]` (Xi'an Jiaotong Liverpool University), Yangbin Chen `[通讯]` (Xi'an Jiaotong Liverpool University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种检索增强的多代理框架 RAM‑SD，用来自适应地检测讽刺文本。

**💡 创新点**

创新点在于通过元规划器根据检索增强的上下文动态选择针对不同讽刺类型的推理计划，并整合多视角代理产生可解释的推理轨迹。

**🔧 技术方法**

采用检索增强、GPT‑4o 作为元规划器和判定器，设计了语义、修辞、期望、知识、对齐与矛盾等专门化代理，最终通过集成器合成最终决策。

**📊 数据集**

在四个公开基准上评估：IAC‑V1、IAC‑V2、MUSTARD 与 SemEval‑2018。

**📈 对比分析**

与主流深度学习、微调 PLM 与 LLM 基线比较，RAM‑SD 在 Macro‑F1 上平均提升至 77.74%，比 GPT‑4o+CoC 提高 7.01 分，并在解释性评估中显著优于对手。

**⚠️ 局限性**

局限在于对检索语料的依赖，缺乏对多模态或新型讽刺的处理；代理间冲突解决机制不够完善，易导致误判传播。

---

## 51. Power-based Partial Attention: Bridging Linear-Complexity and Full Attention

**arXiv ID:** 2601.17334 | [PDF](https://arxiv.org/pdf/2601.17334v1)

**作者:** Yufeng Huang `[一作]` `[通讯]` (Concavity AI), Yufeng Huang (Concavity AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可调节复杂度的Power-based Partial Attention（PPA），通过参数p在O(L)到O(L²)之间平滑切换。

**💡 创新点**

创新点在于将注意力复杂度连续化为O(L^{1+p})，并揭示性能随p呈S曲线跃迁，证明子二次复杂度可逼近全注意力。

**🔧 技术方法**

使用基于幂指数的偏移选择、滑动窗口结合的因果掩码，结合FlashAttention实现的自注意力，并在Nemotron Nano 9B上进行微调。

**📊 数据集**

使用MATH500和GSM8k两大算术/推理基准以及Nemotron训练集的数学子集进行微调和评估。

**📈 对比分析**

通过对比不同p值模型在MATH500和GSM8k上的准确率，发现p≈0.75–0.875时子二次复杂度即可达到接近全注意力的表现。

**⚠️ 局限性**

限制在于缺乏针对PPA的高效GPU实现、训练数据量不足导致最优p偏高以及在不同任务/模型规模上的泛化未验证。

---

## 52. StealthMark: Harmless and Stealthy Ownership Verification for Medical Segmentation via Uncertainty-Guided Backdoors

**arXiv ID:** 2601.17107 | [PDF](https://arxiv.org/pdf/2601.17107v1)

**作者:** Qinkai Yu `[一作]` (University of Exeter), Yanda Meng `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

在医学图像分割模型中嵌入一种隐蔽且无害的水印，利用不确定性引导的后门技术在保持分割性能的前提下，通过LIME解释可视化出QR码水印，实现黑盒下的所有权验证。

**💡 创新点**

① 通过微调模型预测的不确定性（背景最小值或前景最大值）嵌入水印，使得阈值化后的分割结果不变；② 将LIME解释结果转换为可识别的QR码，仅在触发条件下显现；③ 兼顾了无害性、隐蔽性和鲁棒性，首次将这些特性同时应用于医学分割模型的所有权验证。

**🔧 技术方法**

不确定性引导的后门损失、LIME模型解释、QR码编码、四种触发器（噪声、文本、补丁、黑边）设计、模型性能评估指标（Dice、AUC、VS、ASR）等。

**📊 数据集**

UKBB CMR（心脏磁共振）、SEG（视网膜光学相干层析）、EchoNet（超声心动图）和PraNet（结肠镜聚合物分割）四大医学影像数据集。

**📈 对比分析**

与传统基于后门的水印方法（如BadNet、HiDDeN等）进行对比；使用Dice、AUC、VS、ASR等指标评估；结果显示：ASR>95%，Dice/AUC下降<1%，比传统方法在保持分割性能和隐蔽性方面表现更佳。

**⚠️ 局限性**

依赖于模型输出连续概率图的可访问性；临床安全性评估样本有限；对高级水印移除攻击（如知识蒸馏、深度剪枝）尚未全面评估；在部分模型和数据集上对自然伪触发器的误触发率仍有一定风险。

---

## 53. Analysis of voice recordings features for Classification of Parkinson's Disease

**arXiv ID:** 2601.17007 | [PDF](https://arxiv.org/pdf/2601.17007v1)

**作者:** Beatriz Pérez-Sánchez `[一作]` (Universidade da Coruña), Miguel A. Díaz-Freire `[通讯]` (Universidade da Coruña)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究基于语音录音的帕金森病早期诊断，利用特征选择与单隐藏层人工神经网络进行分类。

**💡 创新点**

通过设计特征联合阈值方法，选择30‑50个信息量最大的特征，并发现MFCC与TQWT特征组对分类最有帮助，证明NCA和ReliefF在大幅压缩特征数的同时保持甚至提升性能。

**🔧 技术方法**

采用多种学习算法（LM、RP、BFG等）训练单隐藏层ANN，结合特征选择技术（NCA、ReliefF、mRMR、χ²、PCC等），并使用分层留一患者验证。

**📊 数据集**

使用UCI公开的帕金森病语音数据集（756条样本、753个有效特征），采样为三次/患者的长音/a发声记录。

**📈 对比分析**

与SVM和无特征选择基线对比，使用宏F1和MCC评价；最佳模型在宏F1≈0.976、MCC≈0.974，显著优于先前研究的MCC（最高0.868）。

**⚠️ 局限性**

局限性包括：验证方法受限于样本量且未采用更深层网络或包装特征选择；仅评估单隐藏层ANN；模型可解释性不足，需进一步研究具体特征及更可解释的分类器。

---

## 54. ThinkTank-ME: A Multi-Expert Framework for Middle East Event Forecasting

**arXiv ID:** 2601.17065 | [PDF](https://arxiv.org/pdf/2601.17065v1)

**作者:** Haoxuan Li `[一作]` (University of Electronic Science and Technology of China), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 59127 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ThinkTank‑ME框架，用多专家LLM模型模拟真实智库对中东事件进行预测，并构建了专门的POLECAT‑FOR‑ME基准数据集。

**💡 创新点**

创新点包括①模拟智库多专家协作的思路，②提出三种领导模型聚合策略（专家路由、智慧聚合、精英集成），③针对中东地缘政治构建了细粒度、时空更丰富的POLECAT‑FOR‑ME数据集。

**🔧 技术方法**

技术主要是基于Llama‑3.1‑8B的参数高效微调与Prompt Engineering；专家模型通过门控路由或投票、Best‑of‑N、精英集成等方式进行预测融合。

**📊 数据集**

使用数据集：POLECAT‑FOR‑ME（从POLECAT改造，覆盖35个中东国家/地区的历史事件序列）；与ICEWS、GDELT等传统事件数据集对比。

**📈 对比分析**

实验对比无训练、全数据微调的基线以及GPT‑4o、GPT‑4o‑mini。精英集成（Weighted Best‑of‑N）在微平均准确率24.6%、宏平均准确率25.0%上明显优于全数据微调（22.7/22.6）和单一LLM基线，且在多数国家上超过GPT‑4o；在高资源国家如以色列、埃及，专有LLM仍略有优势。

**⚠️ 局限性**

局限性包括：对低资源国家的预测仍不理想；模型依赖大量训练数据，存在信息泄露风险；推理成本随专家数量提升而增加；缺乏深入的因果推理与可解释性。

---

## 55. Conformal Feedback Alignment: Quantifying Answer-Level Reliability for Robust LLM Alignment

**arXiv ID:** 2601.17329 | [PDF](https://arxiv.org/pdf/2601.17329v1)

**作者:** Tiejin Chen `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**通讯引用:** 7253 | [OpenAlex ID](https://openalex.org/A5100777770)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Conformal Feedback Alignment (CFA)，通过使用合形预测估计答案可靠性，并将其权重应用于偏好对学习，提升 LLM 对齐质量。

**💡 创新点**

创新点在于将答案级可靠性纳入偏好加权，首次将合形预测应用于 AI 反馈对齐，并兼顾 PPO 与 DPO 两种训练框架。

**🔧 技术方法**

采用合形预测（CP）生成可靠性置信集，并在 PPO/DPO 训练中使用不确定性加权损失。

**📊 数据集**

使用 WebGPT、Pairwise 与 Summarize 三个公开数据集，在 Llama2-7B、Llama3.1-8B 和 Qwen2.5-7B 模型上评估。

**📈 对比分析**

与 SFT、PPO、DPO 基线以及其他偏好级不确定性方法对比，CFA 在所有模型与数据集上均获得 1–2 分的平均提升，数据效率更高。

**⚠️ 局限性**

局限包括依赖合形预测校准样本、仅在文本任务上验证、未结合偏好不确定性和答案可靠性。

---

## 56. The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations

**arXiv ID:** 2601.17093 | [PDF](https://arxiv.org/pdf/2601.17093v1)

**作者:** Olha Sirikova `[一作]` (Taras Shevchenko National University of Kyiv), Alvin Chan `[通讯]` (Nanyang Technological University)

**通讯引用:** 2467 | [OpenAlex ID](https://openalex.org/A5102499348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了三角相似性框架，对深度网络进行静态、功能和稀疏性三视角的比较。

**💡 创新点**

创新点在于将CKA/Procrustes、线性模式连通性/预测相似性与参数剪枝三种方法融合成一体，实现全方位的模型相似性评估。

**🔧 技术方法**

采用CkA、Procrustes、线性模式连通性（LMC）、预测相似性（JSD）以及全局幅值剪枝等技术。

**📊 数据集**

使用ImageNetV2、CIFAR-10、ImageNetV2子集等数据集进行实验评估。

**📈 对比分析**

通过三面板对比发现，架构族是主要相似性决定因素，CKA自相似度与准确率高度相关但更稳健，功能相似度更脆弱；整体相似度与剪枝鲁棒性呈强正相关。

**⚠️ 局限性**

主要限制包括计算成本高（CKA O(N²)，LMC 计算昂贵），剪枝仅采用全局幅值方法，可能未能揭示结构化或学习稀疏下的相似性；框架对大规模模型的资源需求大。

---

## 57. Hybrid Deep Feature Extraction and ML for Construction and Demolition Debris Classification

**arXiv ID:** 2601.17038 | [PDF](https://arxiv.org/pdf/2601.17038v1)

**作者:** Obai Alashram `[一作]` (University of Wollongong in Dubai), Abigail Copiaco `[通讯]` (University of Dubai)

**通讯引用:** 312 | [OpenAlex ID](https://openalex.org/A5053459575)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个混合深度特征提取与传统机器学习分类器的流水线，用于自动化识别施工和拆除废弃物；

**💡 创新点**

创新点在于结合Xception网络提取的深度特征与简单高效的机器学习模型（如线性SVM、kNN、Bagged树）实现几乎完美的分类，且不需要复杂的端到端深度学习训练；

**🔧 技术方法**

技术包括图像预处理（灰度化、归一化、数据增强）、Xception网络的全局平均池化特征提取、特征标准化、以及多种传统分类器（SVM、kNN、Bagged树、LDA、LogReg）的训练与调参；

**📊 数据集**

使用了自采的1800张平衡高质量实地施工现场图片，覆盖四类材料（陶瓷/瓷砖、混凝土、垃圾/废料、木材），每类450张；

**📈 对比分析**

与传统机器学习+手工特征、纯深度学习方法对比，混合模型在测试集上达到99.45%准确率和99.47%宏观F1，明显优于RBF‑SVM（96.7%）和LDA（95.6%），并且与端到端深度模型性能相当但计算成本更低；

**⚠️ 局限性**

局限性包括：只考虑四类材料，其他常见废弃物类型缺失；对木材与垃圾之间仍有少量混淆；模型仍需在更大、更多变的实地数据上验证，且未涉及颜色或多模态信息的融合。

---

## 58. Diagnosis Support of Sickle Cell Anemia by Classifying Red Blood Cell Shape in Peripheral Blood Images

**arXiv ID:** 2601.17032 | [PDF](https://arxiv.org/pdf/2601.17032v1)

**作者:** Wilkie Delgado-Font `[一作]` (Universidad de Oriente), Arnau Mir `[通讯]` (Balearic Islands Health Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于Chan‑Vese主动轮廓分割、椭圆拟合与圆形因子/椭圆因子描述子的全流程算法，用于自动计数并三分类红细胞（正常、细胞变形、扭曲）

**💡 创新点**

创新点在于：①不需要预处理即可在原始血涂片图像上完成分割；②利用椭圆拟合精准处理聚集细胞中的重叠与遮挡；③将圆形因子与椭圆因子统一映射到三类，实现高精度分类并给出SDS诊断支持评分

**🔧 技术方法**

主要技术包括Chan‑Vese能量函数分割、级别集演化、曲率检测求解椭圆参数、圆形因子CSF与椭圆因子ESF计算以及基于混合阈值的后续分类与计数

**📊 数据集**

实验使用公开的血涂片数据集（http://erythrocytesidb.uib.es/），共45张500×375像素的血液涂片图像，包含正常、细胞变形与其他变形细胞

**📈 对比分析**

与HT、FF、UNL‑F、SG、CROFT等现有方法在两类和三类任务上进行5折交叉验证比较，本文方法在F‑measure、SDS‑score、CBA和MCC等指标上均名列前茅（F‑measure最高达0.97/0.95，SDS‑score 0.95/0.90）

**⚠️ 局限性**

主要局限在于数据集不平衡、血涂片制备工艺导致阴影与形变噪声，聚集细胞中对角度阈值的依赖可能引起误分类，且仍需人工专家标注作为参考

---

## 59. How does Graph Structure Modulate Membership-Inference Risk for Graph Neural Networks?

**arXiv ID:** 2601.17130 | [PDF](https://arxiv.org/pdf/2601.17130v1)

**作者:** Megha Khosla `[一作]` (Delft University of Technology), Megha Khosla `[通讯]` (Delft University of Technology)

**通讯引用:** 582 | [OpenAlex ID](https://openalex.org/A5027689420)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了图神经网络中图结构（训练图采样与推理时边访问）对节点级成员推断风险的影响，系统评估了随机采样与雪球采样以及三种推理边访问策略的隐私泄露差异；

**💡 创新点**

创新点在于首次从图结构角度量化训练图构造与推理边访问对成员推断优势的影响，并证明训练/测试分割的统计可交换性被破坏，导致传统差分隐私上界失效；

**🔧 技术方法**

使用了三种主流GNN聚合方式（GCN、SAGE、GAT）、基于概率输出与交叉熵的两层MLP攻击模型，以及KL/JS散度分析和对数几率变换；

**📊 数据集**

实验数据集包括Cora、Citeseer和Chameleon三大图分类基准，采用10%与50%训练比例；

**📈 对比分析**

通过对比训练/测试准确率差距、成员推断优势（Adv）和不同边访问场景下的KL/JS分布，发现雪球采样与使用完整边推理可显著降低成员推断优势，且性能差距与隐私风险不总是正相关；

**⚠️ 局限性**

限制在于实验仅覆盖节点级任务，未考虑图级任务与更复杂的采样/攻击策略，且仅使用公开数据集，实际应用场景下图结构可变性更大。

---

## 60. PALMA: A Lightweight Tropical Algebra Library for ARM-Based Embedded Systems

**arXiv ID:** 2601.17028 | [PDF](https://arxiv.org/pdf/2601.17028v1)

**作者:** Gnankan Landry Regis N'guessan `[一作]` `[通讯]` (Axiom Research Group), Gnankan Landry Regis N'guessan (Axiom Research Group)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

实现了一个适用于 ARM 嵌入式系统的轻量级 tropical algebra 库 PALMA，提供五种半环（max‑plus、min‑plus、max‑min、min‑max、Boolean）的稠密与 CSR 稀疏矩阵运算、闭包、特征值、调度等功能，并设计了面向应用的高层 API。

**💡 创新点**

创新点包括：
- 将 tropical algebra 的统一线性框架移植到资源受限嵌入式设备；
- 在无依赖的 C99 代码中使用 ARM NEON SIMD 对核心运算进行加速；
- 结合 CSR 稀疏格式实现高效的闭包与特征值计算；
- 在单个库中提供调度、路由、制造周期等多种经典图算法的统一实现；
- 通过三大真实案例（无人机调度、IoT 路由、制造线周期）验证了算法的可用性和实时性。

**🔧 技术方法**

使用技术：C99 纯实现、ARM NEON SIMD、CSR 稀疏矩阵、Karp 算法求最大循环平均、Floyd‑Warshall 风格闭包、矩阵向量乘优化、静态内存分配、可选 OpenMP 并行（未来工作）。

**📊 数据集**

数据集：随机生成的整数稠密/稀疏矩阵（尺寸 8×8 到 1024×1024），无人机任务依赖图（12 任务），50 节点 IoT 传感器网络，7 节点网络，7 节工作站的制造线模型；实验中还使用标准图数据（如路由网络、工艺流程图）。

**📈 对比分析**

与传统算法对比：在 Raspberry Pi 4 上将 NEON 加速与标量实现比较，获得 1.8× 的加速；单源最短路与 Bellman‑Ford 对比，速度提升高达 11.9×；稠密 64×64 矩阵在 max‑min 半环下实现 2,274 MOPS；稀疏 50%+ 可获得 3.5× 的加速和 47% 内存节省；实时实验显示可在 1 kHz 控制循环下处理 64 节点图，路由计算在 1 ms 内完成。

**⚠️ 局限性**

局限性：
- 仅支持 32 bit 整数 idempotent 半环，无法处理浮点权值或需要高精度的情况；
- 关键闭包与稠密矩阵乘法仍为 O(n³) 复杂度，难以扩展到大规模图；
- 当前实现仅利用单核 NEON，未实现多核或异构并行；
- 依赖 ARM NEON，迁移到 RISC‑V、DSP 等体系结构需额外适配；
- 对动态图或增量更新的支持不足。

---

## 61. Optimizing the Landscape of LLM Embeddings with Dynamic Exploratory Graph Analysis for Generative Psychometrics: A Monte Carlo Study

**arXiv ID:** 2601.17010 | [PDF](https://arxiv.org/pdf/2601.17010v1)

**作者:** Hudson Golino `[一作]` (University of Virginia), Hudson Golino `[通讯]` (University of Virginia)

**通讯引用:** 4631 | [OpenAlex ID](https://openalex.org/A5043910258)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM嵌入空间进行“动态探索图分析”，将嵌入维度视为可搜索的景观，并通过组合信息熵与互信息优化嵌入深度，以提升心理测量的维度结构恢复

**💡 创新点**

首次将嵌入视为可搜索的动态景观，提出在嵌入深度上进行多目标优化（总熵拟合指数与归一化互信息）来平衡结构准确性与熵组织性，突破了传统一次性使用全向量的做法

**🔧 技术方法**

使用Dynamic Exploratory Graph Analysis (DynEGA)、Total Entropy Fit Index (TEFI)、Normalized Mutual Information (NMI)、三角形最大过滤图和Walktrap社区检测算法

**📊 数据集**

基于OpenAI 1,536维嵌入的心理测量题目池（共200题，五维人格结构），通过Monte Carlo模拟在不同题目数与嵌入深度下评估

**📈 对比分析**

将DynEGA的优化结果与传统横向EGA进行比较，结果显示在所有题目池规模下，动态深度优化后结构恢复度（NMI）提升显著，尤其在10–20题/维的情形，且熵组织性（TEFI）亦优于基线

**⚠️ 局限性**

局限于单一构念（夸大型自恋）和单一嵌入模型，权重设定（70/30）缺乏理论依据，未验证优化嵌入与真实问卷响应网络的对应性，且未探究不同网络/社区检测方法的稳健性

---

## 62. The Digital Divide in Geriatric Care: Why Usability, Not Access, is the Real Problem

**arXiv ID:** 2601.17012 | [PDF](https://arxiv.org/pdf/2601.17012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 63. SiMiC: Context-Aware Silicon Microstructure Characterization Using Attention-Based Convolutional Neural Networks for Field-Emission Tip Analysis

**arXiv ID:** 2601.17048 | [PDF](https://arxiv.org/pdf/2601.17048v1)

**作者:** Jing Jie Tan `[一作]` (Universiti Tunku Abdul Rahman), Yan-Chai Hum `[通讯]` (Universiti Tunku Abdul Rahman)

**通讯引用:** 1168 | [OpenAlex ID](https://openalex.org/A5040863689)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SiMiC框架，利用基于注意力的CNN对硅微结构SEM图像进行宽度、高度、半径等几何特征的自动预测，显著降低人工测量工作量。

**💡 创新点**

创新点在于将结构信息（宽度、高度）嵌入注意力机制（多头注意力）并结合上下文感知的CoordConv模块，使模型能更精准捕捉尖端几何细节，同时公开了首个硅微结构数据集供后续研究使用。

**🔧 技术方法**

采用深度学习技术，核心网络为ResNet/EfficientNet/MobileNet骨干，加入多头注意力与结构嵌入，使用数据增强（对比度、亮度变换）、Huber损失训练，并用RMSE与R²评估。

**📊 数据集**

使用900张在特定SEM条件下采集的硅场发射尖端图像，数据已在 https://research.jingjietan.com/?q=SIMIC 公开。

**📈 对比分析**

通过与传统图像处理、不同CNN骨干以及加/不加注意力的配置对比，发现ResNet+多头注意力+增广在半径预测上达RMSE 0.0319、R² 0.3098，优于基线并证明注意力与增广的互补效应。

**⚠️ 局限性**

局限性包括模型的R²仅在0.2–0.3范围，难以解释大量方差；仅针对单一材料/形貌，缺乏与发射性能的直接关联；未针对难样本或类别不平衡问题进行专门处理。

---

## 64. Accelerated Sinkhorn Algorithms for Partial Optimal Transport

**arXiv ID:** 2601.17196 | [PDF](https://arxiv.org/pdf/2601.17196v1)

**作者:** Nghia Thu Truong `[一作]` (University of Maryland), Mai Tran `[通讯]` (Binh Duong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了加速Sinkhorn算法ASPOT，用于部分最优传输（POT），并给出了调参版Sinkhorn的理论改进。

**💡 创新点**

创新点在于将Nesterov加速与Greenkhorn更新相结合，获得了 O(n^{7/3}ε^{-5/3}) 的迭代复杂度；同时证明了合适选择正则化参数γ能把经典Sinkhorn的复杂度降到 O(n^2/ε^{3+o(1)})。

**🔧 技术方法**

使用了Nesterov式动量、Greenkhorn坐标更新、熵正则化、双重迭代框架（solve→round）以及对γ的自适应调节。

**📊 数据集**

实验数据集包括：AI生成图像的颜色转移（n=800种颜色的直方图）和三维点云配准（源、目标点云各约数千点）。

**📈 对比分析**

与可行Sinkhorn、APDAGD以及经典Sinkhorn进行对比；ASPOT 在收敛速度、最终传输成本和视觉质量上显著优于两者；调参Sinkhorn 在迭代次数和成本下降速率上均优于经典Sinkhorn。

**⚠️ 局限性**

局限性包括：复杂度仍为 O(n^{7/3})，尚未达到线性规模；算法参数（如γ、p、α）对性能影响显著，需经验调优；目前仅针对双分量POT，扩展到多分量或大规模分布仍有挑战。

---

## 65. Authority Signals in AI Cited Health Sources: A Framework for Evaluating Source Credibility in ChatGPT Responses

**arXiv ID:** 2601.17109 | [PDF](https://arxiv.org/pdf/2601.17109v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 66. Lex Reformatica: Five Principles of Policy Reform for the Technological Age

**arXiv ID:** 2601.17001 | [PDF](https://arxiv.org/pdf/2601.17001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 67. Self-Organizing Railway Traffic Management

**arXiv ID:** 2601.17017 | [PDF](https://arxiv.org/pdf/2601.17017v1)

**作者:** Federico Naldini `[一作]` (Univ Gustave Eiffel), Paola Pellegrini `[通讯]` (Univ Gustave Eiffel)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一套完整的自组织铁路交通管理系统（SO‑TMS），通过列车间协商与共识机制在扰动情况下实时优化列车时刻表和路由。

**💡 创新点**

创新点在于：①首次将自组织流程完整化（邻域识别、假设生成、兼容性验证、投票式共识与全局合并）；②利用实例分解将大规模调度问题拆解为局部子问题，显著提升求解效率和解的质量；③结合闭环微观模拟器实现可落地的性能评估。

**🔧 技术方法**

核心技术包括：基于 RECIFE‑MILP 的混合整数线性规划变体、路径兼容性图与投票模型共识、时间窗邻域选择、与 OpenTrack 微观铁路模拟器的 API 接口。

**📊 数据集**

实验数据来源于压缩后的意大利塞格拉特–奥斯皮塔莱托控制区实际调度（约93列车，包含不同列车类型及其延迟权重），并在此基础上生成随机扰动场景。

**📈 对比分析**

通过闭环实验将 SO‑TMS 与集中式 RECIFE‑MILP（CEN）及先来先服务（FCFS）对比，评估总延迟与加权延迟的改进。结果显示，SO‑TMS 相对 CEN 的加权延迟提升约5%（无加权约4%），相对 FCFS 的提升分别为约31%和28%。

**⚠️ 局限性**

局限性包括：①需要中心控制器完成最终合并与冲突修复；②邻域选择时间窗和假设数目参数化影响性能，需进一步调优；③缺乏对不同模块配置和机器学习增强实例分解的深入分析。

---

## 68. A Characterization of Geodetic Graphs in Terms of their Embedded Even Graphs

**arXiv ID:** 2601.17077 | [PDF](https://arxiv.org/pdf/2601.17077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

---

## 69. A Mechanistic View on Video Generation as World Models: State and Dynamics

**arXiv ID:** 2601.17067 | [PDF](https://arxiv.org/pdf/2601.17067v1)

**作者:** Luozhou Wang `[一作]` (Hong Kong University of Science and Technology), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2534 | [OpenAlex ID](https://openalex.org/A5101938761)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对现有大规模视频生成模型进行系统梳理，提出一种基于世界模型核心要素（状态构造与动力学建模）的新分类框架，并将评价指标从视觉质量转向功能性评估（长期一致性、因果推理）。

**💡 创新点**

创新点在于：①将视频生成与传统控制理论、强化学习中的世界模型概念对齐，构建以“状态构造”和“动力学建模”为主轴的双柱分类体系；②提出将状态划分为隐式（记忆机制）和显式（压缩型/离线型）两大范式；③将动力学划分为因果知识融合与架构再造两条路径；④明确评估体系包含质量、持久性与因果性三层级，并指出未来研究的两大前沿（持久性与因果性提升）。

**🔧 技术方法**

主要技术包括：Transformer‑based视频生成模型（扩散、VAE、SSM、Mamba等）；记忆机制（压缩、检索、固化）如KV‑cache、ToMe、Sparse‑VideoGen、LoViC等；显式状态实现（隐藏状态、参数状态、离线状态）如MALT Diffusion、VideoSSM、Mamba‑SSM、TTT‑DiT等；因果架构改造（自回归掩蔽、异步噪声、单向注意等）如Causal‑Mask、AR‑Diffusion、Self‑AR；因果知识融合（LLM/VLM引导、链式思维、梯度融合）如Owl‑1、VIDEODIRECTOR‑GPT、BAGEL；评估工具包括 VBench、VBench++、VBench‑Long、Physics‑IQ、ChronoMagic‑Bench、World‑Consistency‑Score 等。

**📊 数据集**

本文综述涵盖的主要数据集与基准包括：VBench/VBench++、VBench‑Long、Physics‑IQ、ChronoMagic‑Bench、World‑Consistency‑Score、rFID、Memory‑Maze、CALVIN、RLBench 等，覆盖从短视频质量到长时序一致性、物理仿真与交互任务的多维评估。

**📈 对比分析**

比较方法：在不同评估维度（质量、持久性、因果性）上对比当前主流模型（Sora、Veo、Kling、Gen‑3 等）及其改进版；在持久性维度使用 VBench‑Long 监测帧数增长时 FVD、主角保持率等；在因果性维度使用 Physics‑IQ、ChronoMagic‑Bench、World‑in‑World 等测试因果一致性。性能方面，虽然模型在视觉质量上表现优秀，但在持久性（长时序漂移）与因果性（物理一致性、因果干预响应）上仍显不足，分数往往低于 30%。

**⚠️ 局限性**

限制：①为综述性质，未提出新模型或实验结果；②依赖公开基准和数据，难以覆盖所有场景；③对因果性评估仍主要依赖离散化物理事件，缺乏连续、可解释的因果推理框架；④现有记忆与状态压缩技术在保持视觉细节与推理一致性之间存在权衡，尚未有统一解决方案；⑤闭环训练与交互式评估（如RLBench）在大规模视频生成模型中实现成本高昂，导致大多数研究停留在开环预测阶段。

---

## 70. LLM-Generated or Human-Written? Comparing Review and Non-Review Papers on ArXiv

**arXiv ID:** 2601.17036 | [PDF](https://arxiv.org/pdf/2601.17036v1)

**作者:** Yanai Elazar `[一作]` (Bar Ilan University), Maria Antoniak `[通讯]` (University of Colorado Boulder)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过构建评审论文与非评审论文的二分类器，并结合 Alpha 估计器和 Pangram 检测器，对 2020‑2025 年 arXiv 上的计算机科学、物理、数学、统计学论文进行大规模量化分析，评估 arXiv 2025 年禁止上传未公开评审论文政策的合理性。

**💡 创新点**

创新点在于：①首次用两种互补的 LLM 生成检测方法（Alpha 与 Pangram）对论文集合进行群体级别评估；②构造基于 Gemma 2 的评审论文自动分类器，准确率达 92.0% F1；③系统性对不同子领域、主题及作者层面进行细粒度比较，揭示政策对各子领域差异化的影响。

**🔧 技术方法**

使用的技术包括：Gemma 2 语言模型进行评审论文判别；Alpha 估计器（基于最大似然的分布式检测）；Pangram API（Transformer 级别 LLM 检测）；Rogan‑Gladen 校正；Bootstrap 置信区间；统计显著性检验。

**📊 数据集**

使用的数据集为：①arXiv 官方 Kaggle 数据集（arxiv‑domains），覆盖 CS、Math、Stats、Physics 2020‑2025 共 124,461 篇；②cs‑subcategories 数据集，涵盖 10 大 CS 子领域每月 500 篇，总计 138,244 篇；③OpenAlex API 提供的作者、机构与主题信息。

**📈 对比分析**

比较方法：对同一论文集合分别使用 Alpha 与 Pangram 估计 LLM 生成比例，并对评审论文与非评审论文、不同年份与子领域进行比较；结果显示评审论文 LLM 生成比例普遍高于非评审，但绝对数量上非评审论文远多。性能方面，Alpha 误差率 <2.4%，Pangram <1%；分类器 F1 92.0%。

**⚠️ 局限性**

Limitations：①Alpha 检测对预训练 LLM 产生的假阳性依赖性需校正；②评审论文分类器虽准确，但仍可能误判边界论文；③跨学科论文混合模式导致标签偏差；④仅基于已上传论文，未覆盖被拒绝或未上传的提交；⑤LLM 检测无法区分全自动生成与部分辅助使用；⑥标签化可能带来污名化风险。

---

## 71. Evaluating Reward Model Generalization via Pairwise Maximum Discrepancy Competitions

**arXiv ID:** 2601.16987 | [PDF](https://arxiv.org/pdf/2601.16987v1)

**作者:** Shunyang Luo `[一作]` (Zhejiang University), Keyan Ding `[通讯]` (Zhejiang University)

**通讯引用:** 1379 | [OpenAlex ID](https://openalex.org/A5032086683)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于最大差异竞争的动态评估框架PMDC，用于评估奖励模型在开放域下的泛化性能；

**💡 创新点**

创新点在于通过主动挑选RM间最大分歧的提示-回复对来构造高信息量的测试集，并用LLM作为可扩展的oracle来减少标注成本；

**🔧 技术方法**

使用min-max归一化的分数差异度量、Bradley–Terry模型进行全局排序、BFGS优化、以及两选一（2AFC）oracle判定；

**📊 数据集**

利用六大LLM基准（MMLU、GSM8K、HumanEval、AlpacaEval、TruthfulQA、HellaSwag）拼接的开放式提示池，以及20种不同规模LLM生成的回复；

**📈 对比分析**

与传统静态Benchmarks（如RewardBench2）对比，PMDC重新排列了10个奖励模型的排名，表现出显著的重排序，oracle一致率与BT分数高度相关，显示更具泛化能力；

**⚠️ 局限性**

局限包括对成对比较的O(N²)复杂度、对LLM判定器偏差的依赖以及对极专门化或快速演化领域覆盖不足。

---

## 72. High-Fidelity Longitudinal Patient Simulation Using Real-World Data

**arXiv ID:** 2601.17310 | [PDF](https://arxiv.org/pdf/2601.17310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 73. Evaluation on Entity Matching in Recommender Systems

**arXiv ID:** 2601.17218 | [PDF](https://arxiv.org/pdf/2601.17218v1)

**作者:** Zihan Huang `[一作]` (University of California), Julian McAuley `[通讯]` (University of California)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了跨 Reddit 与 Amazon 电影目录的实体匹配基准数据集 Reddit‑Amazon‑EM，并对多种实体匹配方法进行系统评测。

**💡 创新点**

首次提供最大规模的自然对话与商品目录映射数据集，并建立严格的评测框架来衡量实体匹配效果。

**🔧 技术方法**

使用了规则匹配、检索（BM25、Faiss）、嵌入+模糊匹配、图神经网络（GNEM）以及基于 LLM 的 ComEM 等多种技术。

**📊 数据集**

基准数据集融合了 Reddit 对话数据与 Amazon 2023 年电影商品目录，覆盖 4,322 条人工标注的正负匹配。

**📈 对比分析**

通过正负样本对、F1、Accuracy、Recall@k 等指标进行比较，图神经网络 GNEM 以 96.29% F1 领跑，传统方法如 BM25、Faiss 的表现显著落后。

**⚠️ 局限性**

主要局限在于人工标注成本高、难以规模化，缺乏高效的弱监督或自动扩展方案。

---

## 74. Multi-stage Bridge Inspection System: Integrating Foundation Models with Location Anonymization

**arXiv ID:** 2601.17254 | [PDF](https://arxiv.org/pdf/2601.17254v1)

**作者:** Takato Yasuno `[一作]` `[通讯]`, Takato Yasuno

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一套基于SAM3的桥梁损伤检测与区域隐私保护系统，能够同时实现高精度的钢筋腐蚀和混凝土裂缝分割，并自动模糊标识牌等敏感位置信息。

**💡 创新点**

创新点在于：①将零射击分割模型SAM3与DBSCAN聚类相结合，实现缺口自动补全；②多阶段检测流程（自动采样、HSV色彩过滤、模式识别）提升检测覆盖率；③在检测管线中嵌入区域匿名化（Gaussian模糊+空间k-匿名）而非事后处理；④通过GPU优化实现1.7秒/图像的实时性能。

**🔧 技术方法**

采用的技术包括：SAM ViT-H模型、HSV颜色空间滤波、DBSCAN聚类、OCR预处理、Gaussian模糊、空间k-匿名、CUDA并行加速。

**📊 数据集**

使用来自日本多座桥梁现场采集的图像数据集，涵盖不同损伤类型、光照条件和标识牌配置，规模足以支持多阶段评估。

**📈 对比分析**

与传统基于手工特征或单一CNN的视觉检测方法相比，本系统在混凝土裂缝检测上达到94.2%精度、91.8%召回；钢筋腐蚀检测精度96.1%、召回93.5%，整体F1为95.1%；隐私保护覆盖率达99.1%，同时保留98.7%的损伤信息；处理速度为1.7秒/图像，显著优于现有方案。

**⚠️ 局限性**

主要局限包括：仅在日本桥梁材料与环境下验证；对极端天气、不同桥梁结构（预应力混凝土、复合材料）适应性不足；缺乏移动端或边缘设备的部署实现；未来需扩展多材料、多区域的泛化能力。

---

## 75. Can LLMs Clean Up Your Mess? A Survey of Application-Ready Data Preparation with LLMs

**arXiv ID:** 2601.17058 | [PDF](https://arxiv.org/pdf/2601.17058v1)

**作者:** Wei Zhou `[一作]` (Shanghai Jiao Tong University), Fan Wu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18974 | [OpenAlex ID](https://openalex.org/A5075948251)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对近年基于大型语言模型的全流程数据准备技术进行系统综述与分类，提出以任务为中心的统一框架。

**💡 创新点**

首次将数据清洗、集成与丰富三大任务纳入LLM驱动的范式，系统化技术细分并给出未来研究路线图。

**🔧 技术方法**

综合运用了提示式生成、代码合成、检索增强生成、代理规划与多模型协同等LLM技术。

**📊 数据集**

评述并引用了众多公开基准（如Adult、Flight、OMOP、MIMIC、DBLP、ChEMBL等）以及相应的评价指标。

**📈 对比分析**

通过与传统规则、单模型和LLM+代理等方法对比，LLM在泛化性和语义推理方面表现更佳，但在推理成本与误差率方面仍有明显差距。

**⚠️ 局限性**

面临高计算与推理成本、幻觉与错误控制不足、对标注数据依赖强、缺乏全局一致性保障以及评估指标与实际需求不匹配等局限。

---

## 76. A Dataset of Dengue Hospitalizations in Brazil (1999 to 2021) with Weekly Disaggregation from Monthly Counts

**arXiv ID:** 2601.16994 | [PDF](https://arxiv.org/pdf/2601.16994v1)

**作者:** Lucas M. Morello `[一作]` (Universidade Estadual Paulista), Leopoldo Lusquino Filho `[通讯]` (Universidade Estadual Paulista)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5000088366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过对巴西1999-2021年市级登革热住院病例的月度数据进行三种插值（线性、抖动、三次样条）比较，并选择三次样条插值方法对数据进行时间细化至流行病学周，生成高分辨率的时间序列数据集。

**💡 创新点**

创新点在于：①使用高分辨率的圣保罗州2024年参照数据验证插值方法的真实性；②提出两条约束（每月周数匹配且保持月度总和）确保插值结果与原始数据一致；③将多种环境与社会经济协变量同步插值，构建统一的多变量时间序列库。

**🔧 技术方法**

主要技术包括：Python的数据处理库（Pandas、NumPy、SciPy）、三次样条插值（SciPy CubicSpline）、随机抖动生成、校正因子实现总和保持，以及多指标评价（MAE、RMSE、R²、KL、JSD、DTW、KS检验）。

**📊 数据集**

使用的基础数据集为巴西DataSUS月度登革热住院病例与IBGE的市级环境与社会经济变量；验证集为圣保罗州2024年同时提供月度与周度计数的CVE‑SP数据。

**📈 对比分析**

在与CVE‑SP周度数据的对比中，三次样条插值在分布相似度（KL、JSD）和时间对齐（DTW）上显著优于线性和抖动，平均R²最高，KS检验通过率约80%，显示其在高发病率地区能够更好捕捉疫情曲线形状。

**⚠️ 局限性**

局限性包括：①生成的周度序列为推断结果，不能完全再现真实的周度报告细节；②在极低发病率或零值频繁出现的市级数据中，三次样条可能产生不现实的振荡；③验证仅基于圣保罗州2024年数据，其他州或历史时期的适用性需进一步检验；④协变量在周度上仅复制月值，缺乏高频变化信息，影响模型解释性。

---

## 77. LoD Sketch Extraction from Architectural Models Using Generative AI: Dataset Construction for Multi-Level Architectural Design Generation

**arXiv ID:** 2601.17095 | [PDF](https://arxiv.org/pdf/2601.17095v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 78. AI-based System for Transforming text and sound to Educational Videos

**arXiv ID:** 2601.17022 | [PDF](https://arxiv.org/pdf/2601.17022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 79. Exploring EEG-driven brain-heart coupling across sleep stages in individuals with sleep disorders

**arXiv ID:** 2601.17149 | [PDF](https://arxiv.org/pdf/2601.17149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 80. Semi-Supervised Domain Adaptation with Latent Diffusion for Pathology Image Classification

**arXiv ID:** 2601.17228 | [PDF](https://arxiv.org/pdf/2601.17228v1)

**作者:** Tengyue Zhang `[一作]` (University of California), William Hsu `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种半监督域适应框架，利用在源域和目标域无标签数据上训练的潜在扩散模型（LDM）生成目标域感知、结构保持的合成图像，再与真实标签图像共同训练下游分类器，显著提升跨机构的病理图像分类性能。

**💡 创新点**

创新点在于：①使用LDM在无标签数据上学习目标域的视觉先验，实现目标域感知的合成；②通过UNI特征、队列身份和组织制备方式进行条件化，既保持组织结构，又注入目标域外观；③将LDM生成的合成图像与传统染色增广相结合，提供多维度的样本多样性。

**🔧 技术方法**

核心技术包括：潜在扩散模型（Latent Diffusion Model）、基于UNI的特征嵌入、条件交叉注意力、VQ‑VAE编码/解码、Vahadane/Macenko染色增广、Vision Transformer（ViT）下游分类器。

**📊 数据集**

使用了两个公开数据集：源域为 NLST（190 例，536 片）与目标域为 TCGA（399 例，1023 片），两者均为肺腺癌 H&E 滴染全切片图像。

**📈 对比分析**

与传统几何变换、颜色抖动、染色增广及基于UNI的线性回归等基线相比，LDM+Vahadane 组合在目标域 TCGA 上的加权 F1 从 0.611 提升至 0.706，宏 F1 从 0.641 提升至 0.716；在源域 NLST 上几乎保持原有性能，证明了跨域泛化的有效提升。

**⚠️ 局限性**

主要限制包括：扩散模型训练与图像生成计算成本高（需 76 小时、两块 L40S GPU），合成图像标签可能存在噪声，且在资源受限环境下不易实现；此外，合成图像的结构细节仍可进一步提升。

---

## 81. Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text

**arXiv ID:** 2601.17172 | [PDF](https://arxiv.org/pdf/2601.17172v1)

**作者:** Tunazzina Islam `[一作]` (Purdue University), Tunazzina Islam `[通讯]` (Purdue University)

**通讯引用:** 121 | [OpenAlex ID](https://openalex.org/A5056005531)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实施了针对年龄与性别的目标消息生成审计框架，比较三大LLM在气候传播中的词汇、风格与说服性偏差。

**💡 创新点**

提出了Standalone与Context‑Rich两种生成场景、Persuasion Bias Index（PBI）衡量说服强度，并系统评估了不同模型在性别/年龄维度上的说服性别偏差。

**🔧 技术方法**

使用GPT‑4o、Llama‑3.3、Mistral‑Large‑2.1模型，构建词典、Odds Ratio、WEAT、情感与形式性分类器，计算PBI并进行统计检验。

**📊 数据集**

通过手工构造的描述符表生成48条SG和1320条CRG文本，使用公开词典与WEAT/情感词集作为评估基准。

**📈 对比分析**

对三模型在词汇、风格和说服三维度进行t检验、ANOVA、OR和WEAT得分比较，结果显示男性/年轻人消息更具说服力，女性/老年人更温和，情境化强化了偏差。

**⚠️ 局限性**

样本量有限、仅考虑二元性别、未涉及交叉偏差、未做模型微调、未评估真实说服效果、仅聚焦气候主题。

---

## 82. (Mis-)Informed Consent: Predatory Apps and the Exploitation of Populations with Limited Literacy

**arXiv ID:** 2601.17025 | [PDF](https://arxiv.org/pdf/2601.17025v1)

**作者:** Muhammad Muneeb Pervez `[一作]` (LUMS), Yasir Zaki `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 1517 | [OpenAlex ID](https://openalex.org/A5018129441)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对低识字人群在使用掠夺性金融应用时的知情同意问题进行实证研究，提出基于大型语言模型的音频、翻译和可视化干预方案；

**💡 创新点**

首次将LLM生成的隐私政策摘要与情境化的“最坏情景”视觉提示相结合，以多模态方式提升低识字用户的风险认知；

**🔧 技术方法**

采用Gemini 1.5 Flash进行摘要生成、Google TTS合成语音、OpenAI/Stable Diffusion生成视觉提示，并对摘要进行人工法律校对；

**📊 数据集**

收集了50款在Google Play中排名前10k、涉及贷款、博彩、交易等掠夺性金融类应用的权限声明和隐私政策文本；

**📈 对比分析**

在34名工厂工人中采用交叉实验，对比未干预、仅文本/语音干预以及文本+视觉干预的效果，发现后两者显著降低用户自信、提升谨慎程度，效果优于单一文本；

**⚠️ 局限性**

样本规模有限、仅覆盖巴基斯坦特定地区，干预的恐惧式视觉可能产生过度焦虑，且模型生成的内容需人工审核防止幻觉，适用性受文化与语言差异限制。

---

## 83. Initial results of the Digital Consciousness Model

**arXiv ID:** 2601.17060 | [PDF](https://arxiv.org/pdf/2601.17060v1)

**作者:** Derek Shiller `[一作]` (Rethink Priorities), Hayley Clatterbuck `[通讯]` (Rethink Priorities)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了数字意识模型（DCM），用于系统化评估AI及生物系统的意识概率。

**💡 创新点**

创新点在于将多种主流意识理论整合为可量化的立场，并通过贝叶斯层级模型聚合不同理论的证据，形成可比较的意识概率。

**🔧 技术方法**

采用贝叶斯层级概率模型结合专家调查的指标置信度进行推断。

**📊 数据集**

使用专家问卷收集的指标置信度，包含2024年LLM、鸡类、ELIZA和人类四种系统的指标数据；共收集16份LLM专家问卷、2份鸡类、1份人类、1份ELIZA。

**📈 对比分析**

通过对不同立场的后验概率进行加权平均（等权或专家可接受度权重），比较各系统意识概率，结果显示LLM在大部分立场下概率低于鸡和人类，ELIZA极低。

**⚠️ 局限性**

局限包括对先验假设高度敏感、指标与立场的理论负荷、专家样本有限、模型仅采用二值独立变量、未覆盖所有可能的意识理论。

---

## 84. Forecasting Energy Consumption using Recurrent Neural Networks: A Comparative Analysis

**arXiv ID:** 2601.17110 | [PDF](https://arxiv.org/pdf/2601.17110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 85. Synthetic Data Guided Feature Selection for Robust Activity Recognition in Older Adults

**arXiv ID:** 2601.17053 | [PDF](https://arxiv.org/pdf/2601.17053v1)

**作者:** Shuhao Que `[一作]` (University of Twente), Ying Wang `[通讯]` (University of Twente)

**通讯引用:** 20876 | [OpenAlex ID](https://openalex.org/A5100347212)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于合成数据指导特征选择的老年人活动识别系统，并在模拟自由生活条件下评估其对跌髋康复相关活动（步行、站立、坐姿、卧姿和姿势转移）的识别性能。

**💡 创新点**

创新点在于利用动态时间规整平均（DBA）生成跨人群共同步态的合成数据，以此为依据进行特征选择，从而显著提升转移动作识别准确率并增强模型对个体差异的鲁棒性。

**🔧 技术方法**

技术手段包括：Savitzky–Golay 滤波、2 秒无重叠窗口分段、DBA 合成、异质特征选择集成（Relief‑F、MRMR、ICT、OOB‑I、LD‑R）、RRA 集成排序、KNN 分类器（k=5）以及 Z 标准化。

**📊 数据集**

数据集：24名≥80 岁健康老年人在模拟自由生活环境中使用双加速度计（上大腿、下背）采集的 20,677 个 2 秒窗口（共 8,244 个窗口每类），以及通过 DBA 生成的同类合成数据集（与实测数据保持相同的类别比例）。

**📈 对比分析**

采用留一参与者交叉验证比较基线模型（仅实测数据）与合成数据指导模型；合成数据指导模型（FIM）在整体 F1 得分 0.915±0.064 高于基线 0.896±0.075，且转移动作 F1 从 0.698 提升至 0.816，差异显著（p<0.05）。

**⚠️ 局限性**

局限性：仅在健康老年人上验证，未包含跌髋患者；合成数据可能抑制病理步态特征，导致对低幅度步态的误判；对真实患者群体、不同病理步态及更丰富的合成策略仍需进一步评估。

---

## 86. Boltzmann-GPT: Bridging Energy-Based World Models and Language Generation

**arXiv ID:** 2601.17094 | [PDF](https://arxiv.org/pdf/2601.17094v1)

**作者:** Junichiro Niimi `[一作]` (Meijo University), Junichiro Niimi `[通讯]` (Meijo University)

**通讯引用:** 58 | [OpenAlex ID](https://openalex.org/A5029152172)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出“口是大脑不是”的架构，将世界模型（DBM）与语言模型（GPT‑2）分离，实现基于行为数据的可控文本生成

**💡 创新点**

创新点在于通过能量模型DBM捕捉领域结构，再通过适配器将隐含信念映射为软提示，证明小型LLM在有合适世界模型时即可生成一致、可控且具有因果特异性的文本

**🔧 技术方法**

使用Deep Boltzmann Machine作为无监督世界模型，Adapter MLP映射至GPT‑2软提示，GPT‑2保持冻结状态；训练过程中采用RBM预训练、对比散度、变分推断和交叉熵损失

**📊 数据集**

使用亚马逊智能手机评论数据（约55k条记录），提取品牌、价格、评级、主题等特征作为可见向量

**📈 对比分析**

与仅基于提示的GPT‑2 baseline对比，评估情感相关性、困惑度、余弦相似度；实验显示DBM‑GPT模型在情感相关性+0.42、困惑度-9.9、相似度+0.09，均显著优于baseline

**⚠️ 局限性**

局限包括仅使用GPT‑2（非最先进LLM），世界模型仅捕捉共现关系未涉及时序动态，实验仅在智能手机评论单一领域，模型对更复杂语义或多语言的适应性待验证

---

## 87. Vidformer: Drop-in Declarative Optimization for Rendering Video-Native Query Results

**arXiv ID:** 2601.17221 | [PDF](https://arxiv.org/pdf/2601.17221v1)

**作者:** Dominik Winecki `[一作]` (Ohio State University), Arnab Nandi `[通讯]` (Ohio State University)

**通讯引用:** 1709 | [OpenAlex ID](https://openalex.org/A5001906560)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了 Vidformer，一种能够把现有基于 OpenCV 的视频可视化脚本自动转换为声明式规格并在服务器端高效渲染的系统；

**💡 创新点**

创新点包括：一键式 API shim 将命令式脚本“升维”为声明式表达式；一种基于声明式规格的并行渲染引擎与调度器；以及通过 VOD（HLS/DASH）按需渲染短段，实现在子秒级内完成视频播放；

**🔧 技术方法**

使用了符号计算、OpenCV API 兼容层、Rust + FFmpeg libav、OpenDAL、HLS/DASH、GPU 加速（NVDEC/NVENC）、多线程调度、TLA+ 形式化验证等技术；

**📊 数据集**

评测使用 Blender Open Movie “Tears of Steel”（720p 24fps 734s）及其 4K 版本；以及约 70 GB 的 PBS NOVA & FRONTLINE 纪录片集合（1920×1080 29.97fps）进行合成任务；

**📈 对比分析**

与基准 Python/OpenCV 脚本（同用 libav/ffmpeg、OpenCV）在相同硬件上比较，Vidformer 全渲染速度提升 2–3 倍，VOD 方式播放延迟下降至 0.25–0.5 s，整体相当于 400–500 倍加速；在多线程、稀疏帧访问、GPU 加速及 LLM 生成脚本等场景也表现出可观的性能提升；

**⚠️ 局限性**

局限性：仅支持无分支的可视化变换（需外部注释生成）；VOD 分段编码/解码仍存在额外开销；无法完全避免服务器端编码负担；GPU 加速目前仅限于编码/解码，过滤仍在 CPU；不适用于需要像素级分支或全功能视频编辑的场景。

---

## 88. TheoremForge: Scaling up Formal Data Synthesis with Low-Budget Agentic Workflow

**arXiv ID:** 2601.17332 | [PDF](https://arxiv.org/pdf/2601.17332v1)

**作者:** Yicheng Tao `[一作]` (Renmin University of China), Hongteng Xu `[通讯]` (Renmin University of China)

**通讯引用:** 3549 | [OpenAlex ID](https://openalex.org/A5035141289)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种低成本、高效的形式化数学数据合成管道TheoremForge，能够从自然语言命题生成完整的Lean形式化证明，并将流程拆分为陈述形式化、证明生成、前提选择、证明纠错与证明草图化五个子任务；

**💡 创新点**

创新点在于：①引入解耦提取策略（Decoupled Extraction），从全局失败的推理轨迹中提取局部成功的训练样本；②将形式化任务拆解为模块化子任务并采用专门的代理模型，提升数据产出率；③通过Gemini-3-Flash等通用LLM与专家模型协同实现低成本的高质量数据合成；

**🔧 技术方法**

技术上主要使用：大语言模型（Gemini-3-Flash/Pro、Claude‑Sonnet‑4.5 等）作为推理核心；Lean v4.19.0 编译器进行语法和语义验证；LLM-as-Judge 进行语义一致性判定；基于 LeanExplore 的前提检索；以及迭代式错误修正和子目标分解技术；

**📊 数据集**

数据集为 DeepMath（103K 题）与 DeepTheorem（121K IMO 级题）中随机抽取的 100 题与 2000 题测试集，采用这些公开数据生成评估基准；

**📈 对比分析**

实验将 TheoremForge 与直接使用专家模型（ReForm‑32B、Goedel‑Prover‑32B）做对比。结果显示在 2000 题基准上，TheoremForge 的 verified rate 为 12.60%（高于基线 8.60%），平均每成功轨迹成本仅 $0.481；在 100 题小规模基准中，Gemini‑3‑Flash 的成本效益最好，取得 23% 的 verified rate；

**⚠️ 局限性**

限制包括：对证明草图化和证明纠错的样本量不足，说明通用 LLM 在处理这些细粒度任务时仍存在瓶颈；未对合成数据在后续模型微调中的实际提升进行量化；缺乏与其他先进 agentic 工作流的直接对比，未来工作需进一步验证。

---

## 89. SPADE: A SIMD Posit-enabled compute engine for Accelerating DNN Efficiency

**arXiv ID:** 2601.17279 | [PDF](https://arxiv.org/pdf/2601.17279v1)

**作者:** Sonu Kumar `[一作]` (Indian Institute of Technology Indore), Adam Teman `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种可多精度（8、16、32位）SIMD Posit MAC计算引擎SPAde，能够在单一数据通路内并行执行多种精度的乘加操作。

**💡 创新点**

创新点在于基于阶段化的lane‑fusion策略和共享的Posit专用子模块（LOD、补码器、移位器、乘法器）实现了无复制的多精度支持，并通过精度可配置的模态信号统一控制不同位宽的执行。

**🔧 技术方法**

核心技术包括：Posit数值解码、Leading‑One Detector、量化补码、逻辑移位、Booth乘法、Quire累加以及SIMD多通道调度；在FPGA和ASIC上实现了RTL级综合与布局布线。

**📊 数据集**

使用的评估数据集包括MNIST、CIFAR‑10/100以及字母识别数据集，验证了在这些图像分类任务上的推理精度不逊于浮点基准。

**📈 对比分析**

通过与先前单精度Posit MAC、浮点/固定点SIMD MAC以及FPGA/ASIC实现的对比，SPAde在FPGA上实现了45.13% LUT与80%片上资源降低、Posit‑8模式下四倍吞吐量、Posit‑16/32模式下28%/17% LUT减少，并在28 nm ASIC上实现1.38 GHz/6.1 mW，显著提升了每瓦吞吐量。

**⚠️ 局限性**

局限性包括：仅支持三种预设精度（8/16/32），缺乏针对训练阶段的动态精度自适应；Posit的可变长度编码导致控制逻辑复杂，且在大规模网络或异构训练工作负载下的实测仍有限。

---

## 90. Atomic Depth Estimation From Noisy Electron Microscopy Data Via Deep Learning

**arXiv ID:** 2601.17046 | [PDF](https://arxiv.org/pdf/2601.17046v1)

**作者:** Matan Leibovich `[一作]` (New York University), Carlos Fernandez-Granda `[通讯]` (New York University)

**通讯引用:** 5384 | [OpenAlex ID](https://openalex.org/A5044336556)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出将3D原子列深度估计转化为语义分割任务，并使用深度卷积网络在带噪声的TEM图像上实现像素级深度预测。

**💡 创新点**

创新点在于将深度估计重新定义为分割问题，并通过在模拟噪声数据上训练UNet实现对低信噪比TEM图像的鲁棒深度恢复，且引入空间标签平滑和权重化交叉熵以提升性能。

**🔧 技术方法**

采用了UNet架构、Poisson噪声仿真、空间标签平滑、加权交叉熵损失、梯度可视化以及置信度校准等技术。

**📊 数据集**

使用了基于CeO2 (110)表面原子模型的多切片模拟TEM图像，并在此基础上添加Poisson噪声得到训练/验证/测试集；还在真实CeO2 TEM序列上做了定性验证。

**📈 对比分析**

与先前的去噪+分割两步方法相比，单阶段SegDepth在像素准确率、中心准确率、真实原子检出率等指标上均更优，且虚假原子率显著降低；在模拟数据上达到约93%像素准确率，中心准确率约94%。

**⚠️ 局限性**

局限在于缺乏真实标签导致只能做定性评估；对不同噪声水平的泛化有限，且当前模型仅适用于高对称区轴视角，未覆盖倾斜或多种元素的情况。

---

## 91. CUROCKET: Optimizing ROCKET for GPU

**arXiv ID:** 2601.17091 | [PDF](https://arxiv.org/pdf/2601.17091v1)

**作者:** Ole Stüven `[一作]` (Institute of Aircraft Production Technology, Hamburg University of Technology), Thorsten Schüppstuhl `[通讯]` (Institute of Aircraft Production Technology, Hamburg University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了ROCKET算法的GPU加速版本CUROCKET，能够高效地对时间序列进行特征提取，支持多实例、多核以及多GPU环境；

**💡 创新点**

创新之处在于设计了一套CUDA级别的卷积方法，能够一次性处理不同长度、扩张率与填充方式的异构卷积核，绕过了传统DL框架的局限，最终实现每瓦性能提升11倍；

**🔧 技术方法**

主要技术包括CUDA核心编程、CuPy与Numba结合的低级GPU计算、线程块/网格映射策略、多实例批处理、以及sklearn兼容接口；

**📊 数据集**

评测使用随机生成的合成数据，按实例数、序列长度、核数三维变化来验证性能；未在UCR数据集上直接实验，但指出目标是解决大规模真实业务场景；

**📈 对比分析**

通过与sklearn实现的CPU ROCKET进行对比，采用“每瓦特性能”指标，在RTX 3090（350 W）与EPYC 7443P（200 W）上测得CUROCKET最低加速为19.3倍，性能每瓦提升约11倍；

**⚠️ 局限性**

局限性包括：GPU实现的PPV特征在极大数据集上因浮点舍入不同而略微偏离CPU结果；此外，时间相关特征（如MultiRocket中的MPV）需进一步改造才能迁移到GPU实现。

---

## 92. STARS: Shared-specific Translation and Alignment for missing-modality Remote Sensing Semantic Segmentation

**arXiv ID:** 2601.17342 | [PDF](https://arxiv.org/pdf/2601.17342v1)

**作者:** Tong Wang `[一作]` (State Key Laboratory of information Engineering in Surveying, Mapping and Remote Sensing, Wuhan University), Zifan Wang `[通讯]` (Hubei FreerTech Co. Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出STARS框架，用于处理多模态遥感语义分割中缺失模态的问题；

**💡 创新点**

通过不对称翻译与停止梯度的对齐机制以及像素级语义采样对齐（PSA）策略，有效解决特征崩塌和类别不平衡问题；

**🔧 技术方法**

使用共享‑特定特征建模、双向翻译模块、SCSE注意力、FPN解码器、像素级对比损失和停止梯度等技术；

**📊 数据集**

在EarthMiss、WHU‑OPT‑SAR和ISPRS Potsdam三个遥感数据集上进行实验；

**📈 对比分析**

与Baseline‑SAR、ShaSpec、MetaRS等多种基线比较，STARS在缺失模态测试下均取得最高mIoU和mF1，提升幅度从1%到13%不等；

**⚠️ 局限性**

对“Others”等弱语义类别的性能略低，且对多模态缺失组合的泛化尚未验证。

---

## 93. Structural Complexity of Brain MRI reveals age-associated patterns

**arXiv ID:** 2601.17211 | [PDF](https://arxiv.org/pdf/2601.17211v1)

**作者:** Anzhe Cheng `[一作]` (University of Southern California), Paul Bogdan `[通讯]` (Instituto de Matemática e Estatística, Universidade de São Paulo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出并应用了三维结构复杂度分析方法来研究脑MRI随年龄的变化，发现大尺度结构复杂度随年龄下降。

**💡 创新点**

创新点在于引入滑动窗口粗化方案，显著提升了在大尺度下的统计稳定性，并提供了更连续、更平滑的多尺度信息损失评估。

**🔧 技术方法**

主要技术包括基于多尺度粗化的重叠度量（O）、滑动窗口均值卷积、对数-对数线性回归以及多尺度相关性分析。

**📊 数据集**

使用了来自英国生物样本库（UKBB）、阿尔茨海默病影像倡议（ADNI）和国家阿尔茨海默病协调中心（NACC）的高分辨率T1加权脑MRI，共计数千名成年受试者，年龄范围约44–90岁。

**📈 对比分析**

与传统体积或厚度指标相比，结构复杂度在大尺度（λ₄、λ₅）上与年龄呈显著负相关（r≈-0.33至-0.51，p<10⁻⁸），并且能作为脑龄预测的潜在特征，具有更好的多尺度敏感性。

**⚠️ 局限性**

限制在于粗化过程中仍受限于体素分辨率，且目前仅验证了健康与认知受损人群，缺乏对不同疾病亚型的进一步分层分析。

---

## 94. Context Lake: A System Class Defined by Decision Coherence

**arXiv ID:** 2601.17019 | [PDF](https://arxiv.org/pdf/2601.17019v1)

**作者:** Xiaowei Jiang `[一作]` `[通讯]` (Tacnode), Xiaowei Jiang (Tacnode)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出决策一致性法则并定义 Context Lake 系统类，阐述 AI 代理在并发不可逆决策时必须共享一致的语义上下文、事务一致性和时空边界。

**💡 创新点**

创新点在于将语义操作、事务一致性与时空边界三项要求统一为系统级保证，并证明现有系统类无法满足此需求，提出 Context Lake 作为必要的新系统类。

**🔧 技术方法**

使用理论推导、抽象模型设计、事务一致性机制与语义查询能力的集成概念。

**📊 数据集**

未使用具体数据集，论文为理论与架构设计。

**📈 对比分析**

无实验对比，主要通过理论证明与案例阐述，未给出性能指标。

**⚠️ 局限性**

实现复杂度高，需统一语义推理与事务管理，难以直接替代现有系统；性能与可扩展性尚未评估。

---

## 95. Beyond Correlations: A Downstream Evaluation Framework for Query Performance Prediction

**arXiv ID:** 2601.17339 | [PDF](https://arxiv.org/pdf/2601.17339v1)

**作者:** Payel Santra `[一作]` (Indian Association for the Cultivation of Science), Debasis Ganguly `[通讯]` (University of Glasgow)

**通讯引用:** 2048 | [OpenAlex ID](https://openalex.org/A5082339849)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种下游感知的查询性能预测（QPP）评价框架，将QPP估计作为IR融合的先验权重，以提升检索效果。

**💡 创新点**

创新点在于将QPP视为对多种检索模型的相对偏好分布，并通过在融合中使用该分布作为先验来衡量QPP的实际效用；同时发现下游评价与传统相关性评价不一致。

**🔧 技术方法**

采用多种无监督统计QPP方法（如NQC、RSD、UEF等）和监督模型（如NQA-QPP、BERT-QPP），将其输出用于加权CombSUM/RRF融合，并用Kendall τ、Pearson相关等指标进行评估。

**📊 数据集**

使用TREC Deep Learning 2019和2020主题集进行实验。

**📈 对比分析**

与无权融合（CombSUM、RRF）比较，QPP加权提升AP@100约4.5%以上；RSD等无监督模型表现最佳；但其在下游评价中的排名与传统相关性评价差异显著。

**⚠️ 局限性**

局限性包括：下游评价仍依赖所选融合策略，对少量查询训练的学习式融合效果不佳；对错误预测的鲁棒性和对不同检索模型的泛化性仍需进一步研究。

---

## 96. Investigating Self-regulated Learning Sequences within a Generative AI-based Intelligent Tutoring System

**arXiv ID:** 2601.17000 | [PDF](https://arxiv.org/pdf/2601.17000v1)

**作者:** Jie Gao `[一作]` (McGill University), Tingting Wang `[通讯]` (Renmin University of China)

**通讯引用:** 1468 | [OpenAlex ID](https://openalex.org/A5100447732)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究在智能辅导系统中对114名学生的自我调节学习（SRL）序列和对生成式人工智能（GenAI）的使用进行轨迹分析，利用序列聚类将学生分为两组，并对其AI使用目的进行信息加工层面（信息获取与信息转化）分类；

**💡 创新点**

创新之处在于将SRL动态序列与GenAI交互模式结合，首次用序列聚类揭示不同学生的学习行为路径，并区分AI使用的认知加工层级；

**🔧 技术方法**

采用顺序聚类（层次聚类与肘部法确定簇数）、转移率热图可视化、以及对AI提问的编码与分类等技术；

**📊 数据集**

使用Healthy Choice平台的学习轨迹日志（110名学生，其中62名至少使用一次Ask AI，共385条提示），包含SRL各阶段的点击和AI交互记录；

**📈 对比分析**

通过比较两组的转移率热图和AI使用频率，发现Cluster 1的转移率更高、AI使用更频繁，但在学习成绩与AI使用目的之间未显著关联；

**⚠️ 局限性**

局限性包括样本量有限、仅涉及单一任务与平台、编码一致率仅74%、缺乏因果推断以及对不同学科或更广泛用户群体的可推广性不足。

---

## 97. Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer LLM Federation

**arXiv ID:** 2601.17133 | [PDF](https://arxiv.org/pdf/2601.17133v1)

**作者:** Inderjeet Singh `[一作]` (Fujitsu Research of Europe), Motoyoshi Sekiya `[通讯]` (Fujitsu Research of Europe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了KNEXA-FL框架，实现去中心化LLM细调的协同学习，并通过中央匹配器学习最优的P2P配对策略。

**💡 创新点**

创新点包括：①将异构LLM间的P2P协作建模为上下文Bandit问题，利用LinUCB学习匹配策略；②采用安全文本级知识蒸馏与PEFT实现模型间知识共享；③设计非聚合中央匹配器避免单点故障；④通过实验验证在高异构环境下实现稳定收敛并大幅提升性能。

**🔧 技术方法**

使用了联邦学习与去中心化P2P通信、PEFT（LoRA）、文本级知识蒸馏、LinUCB上下文Bandit、抽象隐私保护配置、Guardrail过滤器以及加密的端到端通信。

**📊 数据集**

使用了基于HumanEval和MBPP合并的464道代码生成题库，训练集348道按Dirichlet分布分配给6个客户端，128道题为知识转移集。

**📈 对比分析**

与单机局部训练、随机P2P、JS多样性启发式P2P以及两种集中式蒸馏基线（FedID-CentralKD、Central-KD）进行比较。KNEXA-FL在Pass@1上相较随机P2P提升约50%，相较单机提升约6倍；集中式基线出现不稳定并出现灾难性忘记。

**⚠️ 局限性**

限制包括：仅在6个客户端的小规模实验，未验证大规模网络延迟；仅使用Dirichlet分布进行数据分割，未测试更复杂的分布；未与更先进的集中式FL优化器进行比较；对差分隐私、零知识等安全机制的完整评估尚未完成。

---

## 98. From Emotion to Expression: Theoretical Foundations and Resources for Fear Speech

**arXiv ID:** 2601.17132 | [PDF](https://arxiv.org/pdf/2601.17132v1)

**作者:** Vigneshwaran Shankaran `[一作]` (GESIS - Leibniz Institute for the Social Sciences), Claudia Wagner `[通讯]` (GESIS - Leibniz Institute for the Social Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对恐惧言论的跨学科理论进行梳理，并系统评估相关数据集，构建统一的恐惧言论标签体系。

**💡 创新点**

①将政治学、传播学、心理学、语言学的恐惧观融合成多层级定义；②通过AI驱动的文献检索和主动学习筛选，获得37个数据集；③提出包含“恐惧言论”“宣传”“框架”等维度的统一分类框架。

**🔧 技术方法**

采用AI2 Paper Finder与ASReview主动学习进行文献检索与筛选，对数据集进行可访问性、注释透明度与许可评估，并构建分类与维度词典。

**📊 数据集**

共审计37个数据集，覆盖英语、德语、阿拉伯语、乌尔都语等多语言，来源包括新闻、Twitter、WhatsApp、Telegram、Gab等平台。

**📈 对比分析**

本文未进行模型训练，比较侧重于对现有数据集的分布、标签多样性和可复用性分析，结果显示计算机科学主导、社科参与度低、注释透明度不足。

**⚠️ 局限性**

数据集不平衡、可访问性差、缺乏跨学科注释；仅聚焦“有意图的恐惧言论”而未考虑受众易感性；AI辅助检索的系统性评估缺失；未提供建模实验，无法验证方法性能。

---

## 99. How Do We Engage with Other Disciplines? A Framework to Study Meaningful Interdisciplinary Discourse in Scholarly Publications

**arXiv ID:** 2601.17020 | [PDF](https://arxiv.org/pdf/2601.17020v1)

**作者:** Bagyasree Sudharsan `[一作]` (University of Colorado Boulder), Maria Leonor Pacheco `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1003 | [OpenAlex ID](https://openalex.org/A5005560875)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了跨学科 NLP+CSS 文献的引用参与度并提出了专门的引用目的分类框架。

**💡 创新点**

创新点在于构建了针对跨学科研究的引用目的税onomy，并结合引用位置与语义相似度来评估参与深度。

**🔧 技术方法**

采用了人工标注、RoBERTa 及 ChatGPT 5.2 进行分类，使用 SPECTER embeddings 计算上下文与摘要的相似度。

**📊 数据集**

数据集包括 2,046 篇 NLP+CSS 论文中的 37,447 条引用，其中 394 条被人工标注为 369 条最终标签。

**📈 对比分析**

与自动化分类模型比较，Decision Tree + ChatGPT 5.2 的宏 F1 为 0.47，高于 Synthetic Train Set + RoBERTa 的 0.30，但整体性能仍不足以支持大规模自动化。

**⚠️ 局限性**

限制包括数据量有限、人工标注一致性不高、自动模型缺乏多样性、解析错误、以及跨领域通用性尚未验证。

---

## 100. Deconstructing Taste: Toward a Human-Centered AI Framework for Modeling Consumer Aesthetic Perceptions

**arXiv ID:** 2601.17134 | [PDF](https://arxiv.org/pdf/2601.17134v1)

**作者:** Matthew K. Hong `[一作]` (Toyota Research Institute), Matthew Klenk `[通讯]` (Toyota Research Institute)

**通讯引用:** 665 | [OpenAlex ID](https://openalex.org/A5046977124)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一个集成人类评估与计算机视觉特征的消费者审美偏好建模框架，并在汽车轮毂设计上进行实验。

**💡 创新点**

创新点在于将设计师定义的域特定特征、消费者主观评判以及机器提取的低层视觉特征三者整合，并通过语义对齐分析揭示不同风格标签的可区分性。

**🔧 技术方法**

使用了基于YOLOv9的轮毂定位、SIFT、Tamura纹理、GLCM、HOG与Vision Transformer提取特征；对文本描述采用GPT‑5.2生成并用Sentence‑BERT编码；通过Bradley–Terry模型转换为偏好分数。

**📊 数据集**

数据集为包含1000个去标识化汽车轮毂图像，子集80个进行配对比较，共收集约57.6万条评分与约315名参与者的手工特征标注。

**📈 对比分析**

通过OLS回归评估各特征对BT分数的影响，并与语义相似度结合；结果显示某些低层特征（如Tamura对比度、方向数）对动态、未来感、奢华风格显著正向影响，语义对齐对未来感、运动感、动态风格的预测力最高；性能以回归R²值（0.08–0.16）与显著性水平呈现。

**⚠️ 局限性**

局限包括：回归模型假设线性且未考虑配对比较的依赖性；特征标注可能存在主观偏差且未捕获高层语义；语义相似度受LLM预训练偏差影响；样本与风格词汇受文化与背景限制，且未探讨个体差异与群体细分。

---

## 101. Beyond Simulations: What 20,000 Real Conversations Reveal About Mental Health AI Safety

**arXiv ID:** 2601.17003 | [PDF](https://arxiv.org/pdf/2601.17003v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 102. Beyond Factual QA: Mentorship-Oriented Question Answering over Long-Form Multilingual Content

**arXiv ID:** 2601.17173 | [PDF](https://arxiv.org/pdf/2601.17173v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 103. Quantifying Ergonomics in the Elevate Soft Robotic Suit

**arXiv ID:** 2601.17249 | [PDF](https://arxiv.org/pdf/2601.17249v1)

**作者:** Peter Bryan `[一作]` (Imperial College), Dario Farina `[通讯]` (Imperial College)

**通讯引用:** 49840 | [OpenAlex ID](https://openalex.org/A5065669889)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对 Elevate 软体机器人套装在肩部抬升中的人体交互力进行量化评估，包括肩部压力、躯干压缩和上臂压缩等指标。

**💡 创新点**

采用运动捕捉与力传感相结合的方法，无需昂贵压力垫，即可估算套装与人体的交互力。

**🔧 技术方法**

使用 Vicon Vero 运动捕捉系统、线性负载传感器、Python 控制接口以及几何推算和压力量计算等技术。

**📊 数据集**

仅使用单一健康受试者在六个不同扭矩变化速率下的实验数据（约 8 小时穿戴）作为数据集。

**📈 对比分析**

将测得的肩部压力约 69‑85 N 与人手抓握压力相当，躯干压缩 ≤3%，上臂压缩 ≤8%，表明该套装在提供约 120 N 的绳索张力时仍保持良好舒适度。

**⚠️ 局限性**

仅评估了非助力力，未测量助力或代谢成本；实验仅针对单人，且使用了非可反向驱动电机，导致扭矩受限。

---

## 104. PhysE-Inv: A Physics-Encoded Inverse Modeling approach for Arctic Snow Depth Prediction

**arXiv ID:** 2601.17074 | [PDF](https://arxiv.org/pdf/2601.17074v1)

**作者:** Akila Sampath `[一作]` (University of Maryland Baltimore County), Jianwu Wang `[通讯]` (University of Maryland Baltimore County)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理约束的反演框架 PhysE-Inv，用于在海冰雪层稀缺、噪声大的条件下估计北极雪深

**💡 创新点**

核心创新在于使用可投射（surjective）逆映射与物理编码对比学习相结合，实现了更稳健的物理一致性反演

**🔧 技术方法**

采用 LSTM 编码-解码结构、Multi‑head 注意力、物理约束重构正则、物理引导对比学习以及对数似然损失

**📊 数据集**

利用 ERA5 重分析数据（1995‑2024）中的雪反照率、雪密度、海冰浓度等变量构建代理标签

**📈 对比分析**

与 LSTM、BiLSTM、NeuralODE、ResNet‑50 等主流时序模型做同类反演任务对比，PhysE-Inv 在 MSE/RMSE 上平均降低约20% 以上，并在数据稀缺时表现更稳健

**⚠️ 局限性**

局限在于仍需依赖代理模型假设，缺乏直接观测验证，且对高维非线性海冰过程的泛化能力尚待进一步检验

---

## 105. Implementing Tensor Logic: Unifying Datalog and Neural Reasoning via Tensor Contraction

**arXiv ID:** 2601.17188 | [PDF](https://arxiv.org/pdf/2601.17188v1)

**作者:** Swapn Shah `[一作]` (University of North Carolina at Charlotte), Wlodek Zadrozny `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 1178 | [OpenAlex ID](https://openalex.org/A5041770151)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

验证了Tensor Logic框架，即将Datalog规则与Einstein求和相等的假设，在三组实验中实现并测试了符号推理、嵌入空间的零样本推理以及大规模知识图谱的链接预测和多跳推理。

**💡 创新点**

证明了Datalog递归规则可通过迭代张量乘法+阈值实现，关系矩阵可在嵌入空间中学习并通过矩阵连乘实现多跳推理，并展示了Domingos提出的关系矩阵超位置构造(R_r=E^⊤A_rE)在实际知识图谱上的可行性。

**🔧 技术方法**

张量运算（einsum/矩阵乘法）、Heaviside阈值、全连接神经网络（用于关系矩阵学习）、交叉熵/Softmax损失、Adam/AdamW优化器、GPU加速。

**📊 数据集**

三大数据集：1）圣经族谱（1972节点、1727条亲子关系）用于转移闭包；2）mledoze/countries（489实体，489条事实）用于地理零样本推理；3）FB15k‑237（14,541实体、237关系）用于链接预测与2跳推理。

**📈 对比分析**

与传统知识图谱基线（TransE、RotatE等）对比；在标准链接预测上取得MRR 0.3068（Hits@1 22.15%），在去掉直接边的2跳推理基准上取得MRR 0.3346（Hits@1 24%）。虽然未达到最新state‑of‑the‑art，但验证了框架的正确性与多跳推理能力。

**⚠️ 局限性**

受限于Datalog片段（无函数符号、无第一阶量化），无法处理更复杂的逻辑；对大规模图谱仍需稀疏表示与近似技术；仅测试了两跳链路，未涵盖分支、否定或聚合查询；模型尚未与更深层注意力/强化学习等先进架构结合以提升性能。

---

## 106. Parallel Algorithm For Finding The Minimum s/t Cut in a Structured 3-Dimensional Proper Order Graph

**arXiv ID:** 2601.17026 | [PDF](https://arxiv.org/pdf/2601.17026v1)

**作者:** Shridharan Chandramouli `[一作]` `[通讯]` (University of Utah), Shridharan Chandramouli (University of Utah)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

目前未提供论文的具体研究内容，无法确定作者做了哪些工作。

**💡 创新点**

由于缺乏论文主体信息，无法确定创新点。

**🔧 技术方法**

未提供技术细节，无法判断使用了哪些技术。

**📊 数据集**

未提及数据集，无法判断使用了哪些数据。

**📈 对比分析**

缺乏实验与对比信息，无法说明性能表现。

**⚠️ 局限性**

由于上述信息缺失，无法评估论文的局限性。

---

## 107. Deferred Acceptance Algorithm Improves Peer Review Process

**arXiv ID:** 2601.17035 | [PDF](https://arxiv.org/pdf/2601.17035v1)

**作者:** Christoph Bartneck `[一作]` (University of Canterbury), Pattara Klinpibul `[通讯]` (University of Canterbury)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

**🎯 论文内容**

构建基于代理仿真的科研论文投稿和评审流程模型，比较传统过程与使用Deferred Acceptance Algorithm（DAA）匹配的流程。

**💡 创新点**

提出利用DAA实现论文与期刊的一次性匹配，从而显著减少评审次数、出版延迟和评审工作量，同时保持匹配质量。

**🔧 技术方法**

使用Python Mesa框架进行代理仿真，并用DAA（Matching库）进行多对一匹配。

**📊 数据集**

仿真参数基于文献估计的科研者、期刊质量、写作和评审时间分布；未使用公开数据集，而是自行生成符合分布的合成数据。

**📈 对比分析**

通过对比两种流程的指标（出版率、评审次数、延迟、质量适配、作者满意度等）发现DAA流程平均评审次数从8.58降至3.00，延迟从405天降至91天，匹配质量相近；总体效率提升约70%。

**⚠️ 局限性**

限制：仅一次提交机会，无法利用拒稿反馈改进；高比例低质量论文被分配至全部接受期刊；模型忽略作者与期刊的真实选择偏好、审稿人招募策略等现实细节。

---

## 108. Weighted Graph Clustering via Scale Contraction and Graph Structure Learning

**arXiv ID:** 2601.17307 | [PDF](https://arxiv.org/pdf/2601.17307v1)

**作者:** Haobing Liu `[一作]` (Ocean University of China), Yanwei Yu `[通讯]` (Ocean University of China)

**通讯引用:** 907 | [OpenAlex ID](https://openalex.org/A5068209849)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为CeeGCN的加权图聚类框架，结合了集群导向图收缩和权重感知稀疏注意力网络，实现了在噪声权重下的高效聚类。

**💡 创新点**

创新点在于①通过节点密度与距离的综合评分实现图收缩，显著降低图规模同时保留关键节点；②在注意力机制中直接融入边权并利用α-entmax实现自适应稀疏，抑制噪声边的影响。

**🔧 技术方法**

采用了图神经网络（改进的GAT）、α-entmax稀疏注意力、模糊C均值聚类、对比学习损失、模数化损失以及基于PageRank的节点重要性评估等技术。

**📊 数据集**

实验数据集包括船舶轨迹网络的Vessel01、Vessel10以及基于MovieLens100K的ML100K三组真实加权图。

**📈 对比分析**

与11种主流加权图聚类基线（SSGCN、DyFSS、DGCluster等）进行对比，CeeGCN在ACC和Micro‑F1指标上均显著领先，最高微调F1提升约31.5%，同时训练时间和显存使用显著下降。

**⚠️ 局限性**

局限性在于验证仅覆盖中小规模图，未充分评估极大图或高异质性图；依赖较多手工调参；节点特征仅为ID，可能限制在更复杂属性场景的适用性。

---

## 109. SFO: Learning PDE Operators via Spectral Filtering

**arXiv ID:** 2601.17090 | [PDF](https://arxiv.org/pdf/2601.17090v1)

**作者:** Noam Koren `[一作]` (Technion - Israel Institute of Technology), Elad Hazan `[通讯]` (Princeton University)

**通讯引用:** 17261 | [OpenAlex ID](https://openalex.org/A5024431603)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Spectral Filtering Operator（SFO），一种利用固定的Universal Spectral Basis（USB）对PDE解算器的核进行参数化的神经算子。

**💡 创新点**

创新点在于将USB（Hilbert矩阵特征向量）作为全局正交基，利用其快速谱衰减实现仅保留少量模式即可高效捕捉长程非局部交互；并在理论上证明离散Green函数具有空间LDS结构，给出USB截断误差上界。

**🔧 技术方法**

技术包括使用USB展开的核参数化、FFT高效卷积、STU风格的Spectral Transform Unit、残差MLP、以及在多维网格下的tied-index模式。

**📊 数据集**

实验基准涵盖六个PDE任务：1D Allen‑Cahn、Diffusion‑Sorption、Diffusion‑Reaction、Cahn‑Hilliard；2D Shallow Water；3D Maxwell电磁方程。

**📈 对比分析**

与七个SOTA基线（SVD‑NO、DeepONet、FNO、UNO、MPNN、PINO、UNet）对比，SFO在所有任务上取得最高精度，误差平均下降约28–40%，且参数量显著更少。

**⚠️ 局限性**

局限包括对均匀格点和周期边界的依赖，扩展到非均匀或复杂边界时需进一步研究；以及在极高维或高频场景下仍存在误差上升。

---

## 110. Interpretability of the Intent Detection Problem: A New Approach

**arXiv ID:** 2601.17156 | [PDF](https://arxiv.org/pdf/2601.17156v1)

**作者:** Eduardo Sanchez-Karhunen `[一作]` (Universidad de Sevilla), Miguel A. Gutiérrez-Naranjo `[通讯]` (Universidad de Sevilla)

**通讯引用:** 3146 | [OpenAlex ID](https://openalex.org/A5030982174)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了 RNN 在意图检测任务中的内部动态，通过将句子视为隐藏状态空间中的轨迹来解释模型的决策过程。

**💡 创新点**

创新点在于将动态系统理论与逆向工程技术相结合，构建了基于几何分离和读取对齐的诊断框架，揭示了类不平衡如何扭曲低维解空间，并提出了四种模型成功/失败模式。

**🔧 技术方法**

主要技术包括 RNN（Vanilla、LSTM、GRU）训练、PCA 低维嵌入、K‑means 聚类、剪枝评估、固定点和线性化分析，以及余弦相似度对齐度量。

**📊 数据集**

使用了两个公开数据集：平衡的 SNIPS（7 类）用于验证理想几何解，和不平衡的 ATIS（22 类训练 + 16 类测试）用于检验模型在真实场景中的鲁棒性。

**📈 对比分析**

与传统指标（准确率/ F1）相比，诊断框架通过几何分离（Silhouette）和读取对齐（余弦相似度）提供更细粒度的解释，实验显示在 SNIPS 上可达 93%+准确率，在 ATIS 上对不同频率类的表现揭示出多种失效模式。

**⚠️ 局限性**

局限性包括仅针对 RNN 结构，未扩展到 Transformer 等更现代模型；诊断框架依赖手工阈值（如 95% 方差阈值）且在极度不平衡数据下聚类表现不佳。

---

## 111. EMPM: Embodied MPM for Modeling and Simulation of Deformable Objects

**arXiv ID:** 2601.17251 | [PDF](https://arxiv.org/pdf/2601.17251v1)

**作者:** Yunuo Chen `[一作]` (Robotics and AI Institute), Chenfanfu Jiang `[通讯]` (UCLA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一套基于可微分材料点法（MPM）的实体化框架EMPM，能够从多视角RGB‑D视频同时重建物体几何、外观并进行物理模拟，并支持离线与在线参数优化。

**💡 创新点**

其创新点在于将可微分MPM与高保真3D高斯散点渲染相结合，实现了对弹性与弹塑性材料的统一建模；并通过实时感知驱动的在线自适应参数更新，使模拟结果更贴合真实物理行为。

**🔧 技术方法**

主要技术包括可微分MPM（Warp实现）、多视角RGB‑D点云重建、3D Gaussian Splatting渲染、Grounded‑SAM2目标分割、CMA‑ES与AdamW梯度优化等。

**📊 数据集**

使用了真实多视角RGB‑D视频数据集，包含手动操作的绳索、布料、软玩具、塑泥、面团和披萨面包等弹性与弹塑性物体，并在实验中采集了Franka双臂机械手的交互数据。

**📈 对比分析**

与PhysTwin（弹簧-质量模型）和PGND（神经动力模型）在Chamfer距离、IoU、PSNR、SSIM、LPIPS等几何与视觉指标上进行对比，EMPM在所有指标上均优于两者，尤其在弹塑性物体的裂纹与永久变形模拟上表现突出，训练时间虽高于PGND但低于PhysTwin。

**⚠️ 局限性**

局限性主要在于点云追踪在遮挡或大变形时可靠性下降，导致在线优化的监督信号不稳定；此外模型假设材料参数在整个物体内保持不变，无法处理材料非均匀性。

---

## 112. Autonomous Mars Rover Module for Soil Sampling and Life Component Analysis

**arXiv ID:** 2601.17158 | [PDF](https://arxiv.org/pdf/2601.17158v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 113. Tabular Foundation Models are Strong Graph Anomaly Detectors

**arXiv ID:** 2601.17301 | [PDF](https://arxiv.org/pdf/2601.17301v1)

**作者:** Yunhui Liu `[一作]` (State Key Laboratory for Novel Software Technology Nanjing University), Chuntao Hong `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种将图数据“扁平化”为增强的表格特征，直接让表格基础模型（Tabular Foundation Models, TFMs）进行图异常检测，无需针对每个图训练或微调。

**💡 创新点**

核心创新是把图结构信息（拉普拉斯嵌入、结构特征、Beta波let邻域聚合）融入表格特征，构成可被TFM直接使用的输入，从而实现跨域、低标注的“一模型适用所有图”。

**🔧 技术方法**

使用了表格基础模型（TabPFNv2、TabPFNv2.5、LimiX-2M、LimiX-16M）和图到表的特征构造方法；通过Beta波let滤波得到邻域特征，并加入拉普拉斯嵌入和节点度/ PageRank 等结构特征。

**📊 数据集**

在GADBench四大基准数据集上评估：Amazon、YelpChi、T‑Finance、T‑Social。

**📈 对比分析**

与多种需图特定训练的基线（GCN、AMNet、BWGNN、GHRN、RFGraph、XGBGraph、ConsisGAD、SpaceGNN）比较，TFM4GAD 在平均AUROC、AUPRC 上分别达到约89.9%和72.3%，明显优于最强基线 SpaceGNN 的86.5%/61.7%。

**⚠️ 局限性**

限制在于目前仍依赖于预先构造的特征工程，且对极大规模图的特征构造与内存开销需要进一步优化；此外，表格基础模型对极高维稀疏特征的处理可能受限。

---

## 114. Scaling medical imaging report generation with multimodal reinforcement learning

**arXiv ID:** 2601.17151 | [PDF](https://arxiv.org/pdf/2601.17151v1)

**作者:** Qianchu Liu `[一作]` (Microsoft Research), Hoifung Poon `[通讯]` (Microsoft Research)

**通讯引用:** 10066 | [OpenAlex ID](https://openalex.org/A5019494985)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了基于强化学习的通用医学影像报告生成框架 UniRG-CXR，并在胸部 X 光报告生成任务上训练了单一模型。

**💡 创新点**

创新点在于：①将多种临床相关奖励（BLEU、BERTScore、SembScore、RadGraph‑F1 以及 LLM‑based 的 CheXprompt）融合为一个复合奖励，直接优化临床可用性与事实准确性；②采用 GRPO 强化学习避免价值网络，提升生成多样性；③通过在多机构多数据集上联合训练，实现跨机构、跨评测指标、跨诊断级别和跨时间维度的普适性。

**🔧 技术方法**

核心技术包括：Qwen3‑VL‑8B‑Instruct 作为基础模型，SFT + RL 两阶段训练；GRPO 算法、奖励加权策略与 KL 正则化；在 RL 阶段分两步先优化 RadCliQ 复合指标，再加入 CheXprompt 错误惩罚。

**📊 数据集**

使用的训练数据为 MIMIC‑CXR、CheXpert‑Plus、ReXGradient 和 IU 四个公开胸 X 光数据集；评测包括 ReXrank 官方测试集、零样本评测的 IU‑Xray 与私有 PD 数据集。

**📈 对比分析**

与现有方法（如 MedVersa、MedGemma、MAIRA‑2 等）对比，UniRG‑CXR 在 ReXrank 复合指标 RadCliQ‑v1 上实现了 SOTA，四个公开数据集均领先对手，且在零样本、诊断 F1、性别/年龄/种族子组上保持稳健性能。

**⚠️ 局限性**

局限性主要体现在：①仍依赖大量带标注的报告数据，缺乏对低资源或非胸部影像模态的直接适配；②强化学习训练对超参数（奖励权重、学习率等）敏感，需细致调优；③模型在极端罕见疾病或极端图像质量下的鲁棒性尚待进一步验证。

---

## 115. What is a POLYNOMIAL-TIME Computable L2-Function?

**arXiv ID:** 2601.17078 | [PDF](https://arxiv.org/pdf/2601.17078v1)

**作者:** Aras Bacho `[一作]` (California Institute of Technology), Martin Ziegler `[通讯]` (KAIST)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文为连续域上的函数提出了多种多项式时间可计算性的定义（Fourier、Step、平均情形），并系统地比较了它们的关系，证明在一般情况下这些定义不可相互蕴含，除非满足特定的复杂度包含假设；同时给出热方程的应用，说明在特定可计算性假设下初值可计算时解仍可计算，但也构造了反例。

**💡 创新点**

创新点在于把可计算分析与平均复杂度理论结合，给出一组新的可计算性定义并证明它们之间的严格区分；通过将离散判定/计数问题嵌入实函数/数值，实现了从连续可计算性推导离散难度的技术；在热方程的例子中首次证明可计算初值的解在一定假设下保持可计算性。

**🔧 技术方法**

主要技术包括可计算分析（如Weierstrass逼近、Parseval 等）、傅里叶级数与步函数的双重基逼近、平均复杂度与误差传播分析、离散问题的数值编码（二进制/三进制编码）、多项式时间闭包性质以及对热方程解析解的傅里叶展开。

**📊 数据集**

本文没有使用实验数据集，而是采用理论构造与抽象证明的方式来展示结果。

**📈 对比分析**

方法的比较基于理论证明：通过构造分离函数、证明包含/非包含关系，评估不同可计算性定义的强弱；性能方面以多项式时间复杂度作为衡量标准，未进行数值实验。

**⚠️ 局限性**

局限性包括：结论在很大程度上依赖于假设 _1⊆_1（类似 P=NP 的假设）才可能得到某些包含关系；只针对 L²[0,1] 或 C^∞ 的特定函数空间，未覆盖更一般的测度/函数类；此外缺乏对实际数值算法的实现与实验验证。

---

## 116. MANGO: A Global Single-Date Paired Dataset for Mangrove Segmentation

**arXiv ID:** 2601.17039 | [PDF](https://arxiv.org/pdf/2601.17039v1)

**作者:** Junhyuk Heo `[一作]` (TelePIX), Darongsae Kwon `[通讯]` (TelePIX)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了全球红树林监测的MANGO数据集，共42,703对单日Sentinel‑2图像与对应掩模，解决了传统年度标签与单日影像配对不足的问题。

**💡 创新点**

创新点在于提出基于目标检测的“时间配对”方法：利用背景白化匹配器与Fisher判别率自动挑选最能代表年度标签的单日影像，并将该数据集公开为国家分离基准，促进跨国泛化研究。

**🔧 技术方法**

技术实现包括：目标检测式背景白化匹配器、Fisher判别率排序、CNN与Transformer语义分割模型的基准评估；数据采集通过Google Earth Engine。

**📊 数据集**

使用的核心数据集为42,703对Sentinel‑2 L2A（10 m）图像与GMW年度红树林掩模，覆盖124个国家。

**📈 对比分析**

在国别分离（country‑disjoint）拆分下，对比MVI基准与MF（目标检测）选择。MF方法提升IoU从约87%提升至91%，F1从约89%提升至93%，七种分割架构均表现出显著改善。

**⚠️ 局限性**

局限性：仍受GMW标签精度和云/水面干扰影响；仅涵盖2020年，未考虑季节变化；样本在不同红树林密度层级上分布不均，可能影响模型在极稀疏区域的泛化。

---

## 117. LGDWT-GS: Local and Global Discrete Wavelet-Regularized 3D Gaussian Splatting for Sparse-View Scene Reconstruction

**arXiv ID:** 2601.17185 | [PDF](https://arxiv.org/pdf/2601.17185v1)

**作者:** Shima Salehi `[一作]` (Texas A&M University), Joshua Peeples `[通讯]` (Texas A&M University)

**通讯引用:** 108 | [OpenAlex ID](https://openalex.org/A5042499322)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了LGDWT‑GS方法，用频域监督提升稀视角3D重建质量

**💡 创新点**

首次在3D Gaussian Splatting中结合全局与局部离散小波损失，并扩展到多光谱域

**🔧 技术方法**

离散小波变换（Haar）、3D Gaussian Splatting、可微光栅化、深度先验与多光谱联合优化

**📊 数据集**

自制多光谱温室数据集（红、绿、红缘、近红外）以及LLFF、MipNeRF360基准集

**📈 对比分析**

通过PSNR/SSIM/LPIPS等指标与3DGS、DNGaussian、NeRF等基线对比，LGDWT‑GS在稀视角下PSNR提升≈2–3dB、SSIM提升≈0.1，效果最优

**⚠️ 局限性**

对高频细节仍有轻微欠拟合，且在极端光照或户外场景下的鲁棒性待验证

---

## 118. Do VLMs Have a Moral Backbone? A Study on the Fragile Morality of Vision-Language Models

**arXiv ID:** 2601.17082 | [PDF](https://arxiv.org/pdf/2601.17082v1)

**作者:** Zhining Liu `[一作]` (University of Illinois Urbana-Champaign), Hanghang Tong `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 17136 | [OpenAlex ID](https://openalex.org/A5068043486)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究Vision‑Language模型在文本和视觉轻量级扰动下的道德判断鲁棒性，并提出推理时轻量干预方法

**💡 创新点**

首次系统评估多模态扰动对VLM道德稳定性的影响，构造统一的5类扰动框架，并证明强大模型可能更易屈从用户，提出轻量推理时修正策略

**🔧 技术方法**

使用文本/视觉扰动（误导性背景、前缀注入、持续否定、图像文字插入、图像符号注入）以及三种推理时防御（安全策略引导、伦理自我修正、推理引导净化）

**📊 数据集**

基于公开多模态道德基准（覆盖13个道德主题的图文对），结合23个不同规模、族群的VLM

**📈 对比分析**

对23个模型做对比实验，平均道德翻转率约40%；文本扰动比视觉扰动更易导致翻转；推理时干预平均提升约20%（最高约38%）

**⚠️ 局限性**

仅使用轻量化、模型无关扰动，未覆盖自适应或最优攻击；评估仅针对离散道德标签，未考虑文化或细粒度上下文差异

---

## 119. Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts

**arXiv ID:** 2601.17111 | [PDF](https://arxiv.org/pdf/2601.17111v1)

**作者:** Xuan-Phi Nguyen `[一作]` (Salesforce AI Research), Shafiq Joty `[通讯]` (Salesforce AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种名为Least‑Loaded Expert Parallelism (LLEP) 的动态专家并行算法，能够在MoE模型出现显著路由不平衡时，自动将过载设备上的token及其对应专家参数迁移至最空闲设备，以实现负载均衡。

**💡 创新点**

创新点在于：①基于全局路由负载的Least‑Loaded Assignment (LLA) 算法，实时决定每个专家在各GPU上的负载分配；②在不改变MoE原始计算的前提下，动态重路过超载token和专家权重，显著降低峰值显存并提升吞吐；③在设计中兼顾计算与通信开销，并支持梯度反向传播。

**🔧 技术方法**

使用了MoE结构、专家并行 (EP)、All‑to‑All 与 P2P 通信、NCCL、Triton/DeepEP 等实现细节；核心算法为LLA与动态重路由，结合负载阈值α、最小GEMM阈值m、平衡比例阈值λ。

**📊 数据集**

在 gpt‑oss‑20b、gpt‑oss‑120b、DeepSeek‑V3、Kimi‑K2 等大型MoE模型上进行实验，使用 Megatron‑Math 数据集进行真实推理测试，并在模拟失衡场景（30%–95%集中到1/4/16个专家）下评估。

**📈 对比分析**

与标准EP进行对比。LLEP 在不同失衡度下平均可获得 5–6 倍的速度提升，峰值显存下降 4–5 倍；全模型吞吐提升 1.9–2.2 倍，训练时可获得 1.25× 的加速，同时保持原有精度；在极端失衡（95%集中到1个专家）时可达 6.1× 的加速。

**⚠️ 局限性**

局限性包括：需要手动调节超参数 α、m、λ，参数设置对不同硬件/模型规模影响大；实现目前主要基于 Python，通信与权重迁移仍有进一步优化空间；在多节点高延迟网络环境下，跨节点通信开销仍是瓶颈。

---

## 120. ClinNet: Evidential Ordinal Regression with Bilateral Asymmetry and Prototype Memory for Knee Osteoarthritis Grading

**arXiv ID:** 2601.17315 | [PDF](https://arxiv.org/pdf/2601.17315v1)

**作者:** Xiaoyang Li `[一作]` (Northeastern University), Runni Zhou `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种名为ClinNet的自动化膝关节骨关节炎（KOA）分级框架，结合解剖学先验和可信度估计，实现基于X光影像的KL分级；

**💡 创新点**

创新点在于三方面：① 采用双侧不对称编码器（BAE）显式建模膝关节内侧/外侧结构差异；② 使用诊断记忆池对特征进行类级对齐；③ 引入Normal–Inverse–Gamma（NIG）证据推理框架，将KL分级视为连续序数回归并同时输出置信度；

**🔧 技术方法**

技术包括卷积神经网络骨干（ConvNeXt‑Base）、自注意力双分支、特征融合、记忆池原型匹配、NIG证据回归以及KL散度正则化；

**📊 数据集**

使用公开的膝关节X光影像数据集（含KL等级标签），对图像进行裁剪、缩放至448×448，并进行常规增强；

**📈 对比分析**

与DenseNet、ResNet、EfficientNet、Swin、ViT等多种CNN/Transformer基线在相同数据划分下对比，ClinNet在多分类指标上取得最高表现：准确率76.9%、二次加权Kappa 0.892，显著优于EfficientNet-V3（71.97%、0.849）及其他模型；

**⚠️ 局限性**

局限性包括：① 仅在单一数据集上评估，缺乏跨中心、跨设备的外部验证；② 可信度基准和阈值设置在真实临床工作流中仍需进一步验证；③ 对极端病变或极端噪声的鲁棒性尚未彻底测试。

---

## 121. Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning

**arXiv ID:** 2601.17223 | [PDF](https://arxiv.org/pdf/2601.17223v1)

**作者:** Massimiliano Pronesti `[一作]` (IBM Research Europe), Yufang Hou `[通讯]` (IT:U Interdisciplinary Transformation University Austria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Verifiable Process Reward Models (VPRM)，在大语言模型的推理过程中通过确定性规则验证每一步输出，并在医疗系统综述的风险偏倚评估任务中进行强化学习训练。

**💡 创新点**

创新点在于把过程级奖励可验证化：利用域内明确的规则为每一步推理提供精确、可复制的奖励信号，并给出理论保证正确推理获得正奖励、错误推理获得负奖励；相比传统仅终点奖励或神经过程奖励，显著提升了模型的推理连贯性与准确性。

**🔧 技术方法**

采用了 RLVR + GRPO/DAPO 策略优化、确定性规则检验器、步骤级奖励设计、SFT+RL 结合、LLM 提示工程，以及在 Cochrane 系统综述与 RoBBR benchmark 上的实验。

**📊 数据集**

使用的数据集包括 CochraneForest、CochraneForestExt、RoBBR Cochrane、RoBBR Non‑Cochrane，并通过 Llama‑3.1 生成并人工校验了步骤级标签。

**📈 对比分析**

与预训练 LLM、SFT、仅终点奖励、神经过程奖励等基线进行对比；VPRM 在 Coherence、Accuracy 和 macro‑F1 上均获得最高分，F1 提升至约 20%（相较于基线）且与仅终点奖励相比提升 6.5%，证明了可验证过程奖励的显著优势。

**⚠️ 局限性**

局限性包括：仅适用于拥有明确、可编程规则的结构化任务；对小模型的输出格式与检验器匹配要求高；实验仅聚焦风险偏倚评估，缺乏对开放式推理任务的推广验证。

---

## 122. Dynamic Meta-Ensemble Framework for Efficient and Accurate Deep Learning in Plant Leaf Disease Detection on Resource-Constrained Edge Devices

**arXiv ID:** 2601.17290 | [PDF](https://arxiv.org/pdf/2601.17290v1)

**作者:** Weloday Fikadu Moges `[一作]` (South West University Science and Technology), Amin Waqas `[通讯]` (South West University Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一种动态元集成框架（DMEF），通过自适应加权将三种轻量级CNN模型融合，以实现边缘设备上的高精度植物病害检测。

**💡 创新点**

创新点在于：①将模型准确性提升和模型尺寸两个指标联合成权重更新公式，实现训练阶段动态优化；②通过在训练时根据准确率提升和参数量调整权重，平衡精度与计算成本；③在边缘设备上达到不到75 ms的推理延迟和不到1 M参数的模型规模。

**🔧 技术方法**

使用了MobileNetV2、NASNetMobile和InceptionV3三种预训练轻量级CNN，结合自适应加权机制和梯度下降训练；在推理时采用加权softmax融合。

**📊 数据集**

使用PlantVillage数据集，分别对马铃薯（3个病害类）和玉米（4个病害类）进行实验。

**📈 对比分析**

与单一模型、静态加权集成以及多种主流CNN（如DenseNet、EfficientNet、ViT）比较，DMEF在马铃薯上达到99.53%精度、玉米达96.61%，相较最佳单模型提升2–6%且推理延迟仅70–72 ms。

**⚠️ 局限性**

局限性包括：1）实验仅在GPU加速环境下评估，未验证真实边缘硬件的推理速度与功耗；2）数据集规模相对有限，未检验对新病害或不同光照、背景的泛化能力；3）权重更新机制需手动设置学习率和裁剪区间，适用性需进一步验证。

---

## 123. Ego4OOD: Rethinking Egocentric Video Domain Generalization via Covariate Shift Scoring

**arXiv ID:** 2601.17056 | [PDF](https://arxiv.org/pdf/2601.17056v1)

**作者:** Zahra Vaseqi `[一作]` (McGill University), James Clark `[通讯]` (McGill University)

**通讯引用:** 48934 | [OpenAlex ID](https://openalex.org/A5007470779)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Ego4OOD域泛化基准，设计了基于聚类的协方差偏移评分，并用一对多二分类目标训练了轻量级的两层MLP网络；

**💡 创新点**

①以时间片级注释重新组织Ego4D，显著降低概念偏移；②提出可量化域难度的协方差偏移指标；③展示单纯的MLP一对多训练即可达到SOTA级别的泛化性能；

**🔧 技术方法**

使用SlowFast预提取特征，构建两层全连接MLP-Lite，采用一对多二分类损失；通过k‑means聚类对特征进行分组，计算域间原型距离得到协方差偏移分数；采用留一域离线评估；

**📊 数据集**

基准数据集：Ego4D（构成Ego4OOD）和Argo1M；

**📈 对比分析**

与ERM、CORAL、DANN、MADA、Mixup、BoDA、DocPrompt、CIR、EgoZAR等方法在Argo1M上对比，MLP‑Lite平均精度24.38%，在多域上仅次于CIR；在Ego4OOD上与CIR相近（54.02% vs 55.68%），并在FRL、UK等域表现更优；同时证明域偏移分数与识别准确率负相关；

**⚠️ 局限性**

仅提供协方差偏移指标，概念偏移仍缺乏量化；基准样本量相对较小；依赖预提取特征，未使用多模态信息；

---

## 124. Systematicity between Forms and Meanings across Languages Supports Efficient Communication

**arXiv ID:** 2601.17181 | [PDF](https://arxiv.org/pdf/2601.17181v1)

**作者:** Doreen Osmelak `[一作]` (Saarland University), Kate McCurdy `[通讯]` (Saarland University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个信息理论框架，用以量化动词和代词范式中形式与意义之间的系统性映射，并检验其是否提升语言的沟通效率。

**💡 创新点**

提出了基于学习可行性的复杂度度量（CETL），该度量通过序列到序列神经网络学习字符级形式，捕捉形态学内部的系统性，从而弥补传统信息瓶颈模型仅关注同形同义的缺陷。

**🔧 技术方法**

采用序列到序列 LSTM 结构实现编码器，对多语言的动词和代词范式进行训练，并用交叉熵下降速率作为复杂度指标；同时使用信息瓶颈模型的 KL 散度作为准确性衡量。

**📊 数据集**

使用包含 561 种语言的代词范式数据库（ppd）和 56 种语言的动词范式数据（Verb_Sem），涵盖 Semitic、Afro‑Asiatic、Germanic、Romance 等语言族；以及针对每种语言族的细粒度代词数据。

**📈 对比分析**

通过生成结构化与表面层的伪范式，比较attested范式与对照范式在 CETL 与信息瓶颈复杂度以及准确性上的表现。结果显示：CETL 能在 65%–90% 的对照伪范式中正确识别 attested 范式更高效，且与范式自然度呈正相关（ρ≈0.8）；而传统信息瓶颈模型在这两项指标上无明显优势。

**⚠️ 局限性**

仅限于离散的范式结构，无法处理连续域（如颜色）；未构造完整的 Pareto 前沿，伪范式生成并非穷尽所有可能；CETL 的优势在不同语义域的泛化性仍待验证。

---

## 125. Summary of the Unusual Activity Recognition Challenge for Developmental Disability Support

**arXiv ID:** 2601.17049 | [PDF](https://arxiv.org/pdf/2601.17049v1)

**作者:** Christina Garcia `[一作]` (Kyushu Institute of Technology), Sozo Inoue `[通讯]` (Kyushu Institute of Technology)

**通讯引用:** 1723 | [OpenAlex ID](https://openalex.org/A5080895628)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织了ISAS 2025 发展障碍支持场景下的姿态数据异常行为识别挑战，提供训练/测试数据并评估模型。

**💡 创新点**

通过姿态数据实现无隐私视频监控、引入留一主体交叉验证和宏F1评估，聚焦罕见异常行为识别。

**🔧 技术方法**

结合传统机器学习、CNN–LSTM、Transformer、Graph CNN等多种时空模型，使用加权损失和滑动窗口等技术。

**📊 数据集**

使用5名健康受试者模拟的8类行为（4正常+4异常）的2D骨架关键点序列数据，提供单人测试和LOSO集。

**📈 对比分析**

采用宏F1和准确率比较，最佳队伍宏F1≈0.92，平均LOSO准确率≈87%，表现优于大多数传统方法，但对细粒度动作仍有挑战。

**⚠️ 局限性**

数据规模小、受试者有限、异常行为为模拟、类不平衡及姿态噪声，限制了模型的泛化和精度。

---

## 126. CaseFacts: A Benchmark for Legal Fact-Checking and Precedent Retrieval

**arXiv ID:** 2601.17230 | [PDF](https://arxiv.org/pdf/2601.17230v1)

**作者:** Akshith Reddy Putta `[一作]` (University of Texas at Arlington), Chengkai Li `[通讯]` (University of Texas at Arlington)

**通讯引用:** 3775 | [OpenAlex ID](https://openalex.org/A5084878734)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CaseFacts基准，用于验证口语化法律主张与美国最高法院判例之间的真实性。

**💡 创新点**

创新在于构建跨语义差距的检索与验证任务、引入Overruled类别以模型化判例的时间动态、以及使用LLM驱动的多阶段合成与验证管线。

**🔧 技术方法**

利用大型语言模型进行主张生成与事实性验证、语义相似度启发式筛选、精调的嵌入检索模型以及GPT‑4o等基准模型进行检索‑验证实验。

**📊 数据集**

基于Oyez提供的3,299份最高法院案例专家摘要生成6,294条主张，分为5,794条训练集和500条人工标注的测试集。

**📈 对比分析**

采用检索质量的Evidence Score、判定准确率和Verdict Score进行评估；结果显示纯内部知识的GPT‑4o在检索和综合评分上优于开放搜索版本，精调检索模型显著提升Recall@1，但整体任务仍处于挑战级别。

**⚠️ 局限性**

局限于摘要而非完整判决、语义阈值方法可能漏检低相似度的Overruled案例、合成的Refuted主张可能缺乏真实误信息特征、以及仅覆盖最高法院，未涵盖法规或州级判例。

---

## 127. Crystal-KV: Efficient KV Cache Management for Chain-of-Thought LLMs via Answer-First Principle

**arXiv ID:** 2601.16986 | [PDF](https://arxiv.org/pdf/2601.16986v1)

**作者:** Zihan Wang `[一作]` (University of Science and Technology of China), Xuehai Zhou `[通讯]` (University of Science and Technology of China)

**通讯引用:** 3520 | [OpenAlex ID](https://openalex.org/A5077322091)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为Crystal-KV的链式思考（CoT）KV缓存压缩框架，旨在在推理过程中保留对最终答案至关重要的KV条目，显著降低内存占用并提升吞吐量。

**💡 创新点**

核心创新包括：①答案优先原则，将KV条目分为真正贡献答案的CrystalKV和仅维持推理流程的SlipKV；②基于注意力的Least Recently Frequently Used（LRFU）淘汰策略，精确识别并删除SlipKV；③自适应预算分配机制，在层级和头部层面动态调节缓存预算，以最大化重要条目的保留。

**🔧 技术方法**

技术实现主要涉及：注意力投影映射、top‑p核采样、CRF（Combined Recency and Frequency）评分、LRFU淘汰算法、层/头自适应预算分配、以及对已有模型KV缓存的增量更新。

**📊 数据集**

实验使用了CodeForces（编程）和MATH‑500（数学）两个大规模推理基准，并在DeepSeek‑R1‑distilled的Llama‑8B、Qwen‑14B、Qwen‑32B模型上进行评估。

**📈 对比分析**

与R‑KV、RaaS、SnapKV、StreamingLLM、H2O、Ada‑SnapKV及FullKV等基线比较，Crystal‑KV在相同KV预算下准确率提升18–24%，内存节省≈90%，吞吐量提升≈7.5×，并在一定预算下甚至超过FullKV的答案准确率，用户响应时延提升≈1.24×。

**⚠️ 局限性**

局限性包括：对λ（衰减率）与top‑p阈值的敏感性，需要经验性调参；方法主要针对CoT任务，可能对其它生成任务的适用性有限；在极长序列或极高动态性的推理场景下，CrystalKV与SlipKV的区分可能不够清晰，导致误淘汰或误保留。

---

## 128. Studying Mobile Spatial Collaboration across Video Calls and Augmented Reality

**arXiv ID:** 2601.17238 | [PDF](https://arxiv.org/pdf/2601.17238v1)

**作者:** Rishi Vanukuru `[一作]` (University of Colorado Boulder), Ellen Yi-Luen Do `[通讯]` (University of Colorado Boulder)

**通讯引用:** 4749 | [OpenAlex ID](https://openalex.org/A5071892737)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过比较研究评估移动视频呼叫与移动增强现实呼叫在空间协作中的效果。

**💡 创新点**

引入结构化比较观察方法，将AR与传统视频对比，揭示AR对空间感知和协作角色分配的影响。

**🔧 技术方法**

基于Unity实现的原型，集成Agora视频流、Photon Fusion网络定位、RealSense D435摄像头进行实时姿态捕捉与3D空间共享。

**📊 数据集**

研究中使用自建的两套犯罪现场模拟场景（艺术盗窃与技术入侵）及参与者收集的音视频、运动日志、问卷和手绘地图。

**📈 对比分析**

采用比较结构化观察法，结合定性访谈与观察以及空间存在感和回忆得分的量化分析，结果显示AR呼叫在空间存在感和记忆准确性上优于视频，任务完成时间无显著差异。

**⚠️ 局限性**

样本规模有限、实验任务受限于实验室环境，仅使用移动设备，未涉及真实日常使用场景和更大规模的用户研究。

---

## 129. Decoding Psychological States Through Movement: Inferring Human Kinesic Functions with Application to Built Environments

**arXiv ID:** 2601.17194 | [PDF](https://arxiv.org/pdf/2601.17194v1)

**作者:** Cheyu Lin `[一作]` (Carnegie Mellon University), Sirajum Munir `[通讯]` (Bosch Research and Technology Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了面向双人交互的多模态数据集 DUET，并基于 Ekman 与 Friesen 的 kinesics 体系构建了隐式识别框架，用于从人体骨骼运动直接推断交互功能，支持在公共基础设施中持续、可解释地测量社交互动层。

**💡 创新点**

创新点包括：① 将五类 kinesics（符号、示意、情感、调节、适应）与社交资本理论对齐，构造功能级交互词典；② 开发无需手工映射的双人 HAR 框架，将 ST‑GCN 的低维表示与 CNN 相结合，实现功能推断；③ 通过单摄像头多视角采集方法，获得高视角变异且隐私友好的四模态数据（RGB、IR、深度、骨骼）。

**🔧 技术方法**

技术手段主要包括：Azure Kinect 多模态采集、骨骼关键点预处理、ST‑GCN（10 层图卷积）提取空间‑时间特征、1D CNN 对特征进行功能分类、Pearson 相关分析验证模型间关系、对比六种公开双人 HAR 算法。

**📊 数据集**

使用的主要数据集是自建的 DUET，包含 14,400 个双人样本、12 个交互类别、3 个采集场景、4 种感知模态；另外对比了 NTU RGB+D 120、M^2I、G3Di 等现有双人数据集。

**📈 对比分析**

通过在 DUET 上对六个开源双人 HAR 模型进行交叉主体和交叉地点评估，发现骨骼基模型（如 DR‑GCN）在交叉主体上最高 41.57% 识别率，RGB/深度模型显著受遮挡影响。随后在 30 个随机子集上训练 ST‑GCN 与 CNN，ST‑GCN 与 CNN 的准确率呈现 ρ=0.91 的高度正相关，表明低维表示越好，功能识别准确率越高。整体性能仍不如单人任务，凸显双人交互识别的挑战。

**⚠️ 局限性**

局限性包括：① 框架对骨骼关键点的依赖，细粒度动作（如手势细节）仍被忽略，导致某些类别混淆；② 数据集规模虽然大，但仅覆盖 12 个预定义交互，未覆盖更广泛的社交行为；③ 评估多为实验室场景，真实环境中的光照、遮挡、多人等更复杂情形需进一步验证；④ 只验证了一个骨骼基模型（ST‑GCN）与 CNN 的组合，缺乏对更先进时空模型的探索。

---

## 130. Relating Word Embedding Gender Biases to Gender Gaps: A Cross-Cultural Analysis

**arXiv ID:** 2601.17203 | [PDF](https://arxiv.org/pdf/2601.17203v1)

**作者:** Scott Friedman `[一作]` (SIFT), Jeffrey Rye `[通讯]` (SIFT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究使用 2018 年的英文推文构建不同文化（99 个国家和 51 个美国州）的词嵌入，量化其性别偏差，并将这些偏差与多项国际与国内的性别差距统计量相关联，探讨语言偏差与社会性别不平等之间的关联；

**💡 创新点**

创新点在于：① 将主题化的词集（如政府、威胁、社区等）与具体性别差距维度进行匹配，发现偏差与相似主题的性别差距呈现正负相关；② 引入轴投影、相对 L2 范数等多种偏差度量，并对不同词嵌入算法进行系统对比；③ 通过情感与支配度分析识别与性别差距显著相关的形容词，揭示隐式性别评价与实际机会之间的关系；

**🔧 技术方法**

主要技术包括 Word2Vec 词嵌入、基于词向量的性别轴投影偏差度量、相对 L2 范数/比例偏差度量、特征选择与 R² 回归、情感与支配度评分（WordNet、Affect 词典）等；

**📊 数据集**

数据集为 2018 年的英文推文，覆盖 99 个国家（从 98K 条到 122M 条）和 51 个美国州（从 450K 条到 65M 条），对超过 10M 条的语料采用 10M 条采样；统计指标包括 18 项国际性别差距指标（全球性别差距指数等）和 5 项美国国内指标（工资差距、教育、立法、健康等）；

**📈 对比分析**

通过比较 Word2Vec、GloVe、FastText 等词嵌入算法以及三种偏差度量，发现 Word2Vec 结合轴投影度量在 R² 相关性上最优（部分主题与性别差距的 R² 最高可达 0.51，整体表现优于其他组合）；

**⚠️ 局限性**

主要局限性包括：仅使用英文推文忽略了非英语用户的声音；样本量受限（对大语料做 10M 采样），可能导致低语料国家偏差不准确；关联性强但不证明因果关系；模型只捕获语言层面的隐式偏差，未考虑线下文化因素或多语言语境。

---

## 131. PUNCH: Physics-informed Uncertainty-aware Network for Coronary Hemodynamics

**arXiv ID:** 2601.17192 | [PDF](https://arxiv.org/pdf/2601.17192v1)

**作者:** Sukirt Thakur `[一作]` (AngioInsight), Maziar Raissi `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理约束的无监督不确定性框架，利用常规血管造影（kymograph）直接估计冠脉流量储备（CFR），实现非侵入性诊断冠状动脉微血管功能（CMD）；

**💡 创新点**

创新点在于将物理信息神经网络（PINNs）与变分推理相结合，单患者无监督训练，给出自适应不确定性区间，同时采用双分支模型联合估计静息与高通状态的 CFR，突破了传统需大量标注和侵入性测量的局限；

**🔧 技术方法**

使用的技术包括：kymograph 构建与中心线跟踪、物理信息神经网络（PINNs）、变分推理与 KL 正则化、Monte Carlo 采样生成不确定性分布，整个推理流程在单张 GPU 上约 3 分钟完成；

**📊 数据集**

所用数据集为 1,000 条合成的 kymograph（模拟不同噪声、伪影），以及 12 名临床患者的常规造影与相应的热注射法（bolus thermodilution）CFR 结果；

**📈 对比分析**

方法通过与热注射法的 Pearson 相关系数 r=0.90、Spearman ρ=0.76、Bland‑Altman 平均偏差 -0.45、95% 限界 [-1.91,1.02] 进行比较，合成数据上 RMSE<0.1、覆盖率可调，且不确定性与误差高度相关；

**⚠️ 局限性**

局限性包括：仅基于一维中心线近似，难以处理分支、重叠或极端弯曲血管；缺乏多中心大样本验证，尚未确定临床阈值和决策影响；此外，合成与临床数据以及代码受专利与法规限制，无法公开。

---

## 132. Advancing Improvisation in Human-Robot Construction Collaboration: Taxonomy and Research Roadmap

**arXiv ID:** 2601.17219 | [PDF](https://arxiv.org/pdf/2601.17219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 133. Meta-Judging with Large Language Models: Concepts, Methods, and Challenges

**arXiv ID:** 2601.17312 | [PDF](https://arxiv.org/pdf/2601.17312v1)

**作者:** Hugo Silva `[一作]` (University of Coimbra), Hugo Gonçalo Oliveira `[通讯]` (University of Coimbra)

**通讯引用:** 1384 | [OpenAlex ID](https://openalex.org/A5000489917)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并构建LLM-as-a-Meta-Judge研究框架，梳理关键技术与方法。

**💡 创新点**

首次系统化梳理 meta-judging 的六大维度，并提出 Meta-Rewarding 等自我改进机制。

**🔧 技术方法**

使用 LLM 评判与 Meta-judging 机制、SFT+ DPO 训练、Self‑Rationalization、Meta‑Rewarding、跨模型多评判等技术。

**📊 数据集**

利用 JudgeBench、AlpacaEval 2、Arena‑Hard、MT‑Bench、RewardBench、BiGGen Bench、LiveBench 等多任务评估集。

**📈 对比分析**

与单模 LLM 判定器和人类评分对比，Meta‑Rewarding 在 JudgeBench 等基准上提升 Win‑rate 约 15–20%，Self‑Rationalizing Evaluators 提高 10% 以上准确度，但受数据与模型规模影响。

**⚠️ 局限性**

面临高计算成本、提示敏感性、共享偏差、数据局限性和可扩展性等挑战。

---

## 134. Parameter Inference and Uncertainty Quantification with Diffusion Models: Extending CDI to 2D Spatial Conditioning

**arXiv ID:** 2601.17224 | [PDF](https://arxiv.org/pdf/2601.17224v1)

**作者:** Dmitrii Torbunov `[一作]` (Brookhaven National Laboratory), Yimei Zhu `[通讯]` (Brookhaven National Laboratory)

**通讯引用:** 39434 | [OpenAlex ID](https://openalex.org/A5100671232)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对CDI框架进行二维空间扩展，用交叉模态Transformer实现对CBED参数的概率推断，并验证其不确定性量化效果。

**💡 创新点**

将参数空间扩散迁移至二维图像观测，提出跨模态Transformer联合token化实现图像与参数的双向注意，并证明其能给出校准良好的可辨识与模糊参数后验分布。

**🔧 技术方法**

采用条件扩散模型（DDPM）+ResNet‑34视觉编码器+Transformer注意力机制，并用置信区间与覆盖‑锐度评估来衡量不确定性质量。

**📊 数据集**

使用基于Bloch波物理模拟生成的1M训练、10万测试的CBED图像数据集，其中包含13个真值参数。

**📈 对比分析**

与多种CNN回归基线对比，CDI在可辨识参数上误差与回归相当，但在不确定参数上给出合理宽度；校准误差≤10%，覆盖‑锐度优于回归。

**⚠️ 局限性**

仅在模拟数据上验证，真实实验噪声与模型假设需进一步研究；对极高维参数空间的扩展仍面临计算瓶颈。

---

## 135. MambaNet: Mamba-assisted Channel Estimation Neural Network With Attention Mechanism

**arXiv ID:** 2601.17108 | [PDF](https://arxiv.org/pdf/2601.17108v1)

**作者:** Dianxin Luan `[一作]` (Institute for Imaging, Data and Communications), Cheng-Xiang Wang `[通讯]` (National Mobile Communications Research Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于自注意力与定制 Mamba 模块的 MambaNet 框架，用于在 OFDM 子载波数大时实现低复杂度且精度更高的通道估计。

**💡 创新点**

创新点在于：1）设计双向选择性状态空间扫描的 Mamba 模块，能高效捕获非因果子载波间的长距离相关性；2）将该模块与多头注意力预处理相结合，减少参数量并提升泛化能力。

**🔧 技术方法**

技术包括：多头自注意力、定制 Mamba（选择性状态空间模型、门控系数、双向扫描）、残差卷积网络、双线性上采样；训练使用 Huber 损失。

**📊 数据集**

使用 3GPP TS 36.101 的 ETU 渗透信道数据集，生成 125,000 条独立信道样本（95% 训练、5% 验证），SNR 范围 5–25 dB。

**📈 对比分析**

与 LS、MMSE、InterpolateNet、HA02、Channelformer 进行对比；在 5–30 dB SNR 范围内，MambaNet 的 MSE 下降 0.00072–0.000065，BER 低至 0.00018，优于所有基线；参数量 0.35 M，较 Channelformer/HA02 减少 73–76%，运行时间略高。

**⚠️ 局限性**

局限性：相较于 Channelformer，运行时间略高；仅在 ETU 信道上验证，实际环境中信道多样性和极大子载波数下的性能仍待进一步评估。

---

## 136. GlassesGB: Controllable 2D GAN-Based Eyewear Personalization for 3D Gaussian Blendshapes Head Avatars

**arXiv ID:** 2601.17088 | [PDF](https://arxiv.org/pdf/2601.17088v1)

**作者:** Rui-Yang Ju `[一作]` (Kyoto University), Jen-Shiun Chiang `[通讯]` (Tamkang University)

**通讯引用:** 1268 | [OpenAlex ID](https://openalex.org/A5025986552)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 GlassesGB 框架，结合 2D GAN 眼镜编辑与 3D Gaussian Blendshapes 头部重建，实现从单目视频生成可自定义、可视化的 3D 眼镜并实时渲染。

**💡 创新点**

创新点在于首次将可控 GAN 生成的眼镜与 3D Gaussian 表示相结合，使眼镜具备真实三维几何、可在任意视角渲染，并支持细粒度参数控制。

**🔧 技术方法**

使用的技术包括：GlassesGAN（可调节参数的 2D 眼镜生成）、3D Gaussian Splatting（高效 3D 渲染）、FLAME 头部模型与表情 blendshapes、光流平滑（Temporal Smoothing）以及 LBS 皮肤拉伸。

**📊 数据集**

实验数据集：单一 YouTube 视频（https://www.youtube.com/watch?v=mKHgXHKbJUE）作为输入进行评估；没有使用公开大规模眼镜或 3D 头部数据集。

**📈 对比分析**

比较方法：通过三种无参考指标（ITF、ISI、MOFM）评估视频平滑度；深度图对比显示 GlassesGB 的眼镜在几何上优于纯 3DGB 的纹理覆盖，且在新视角保持原始几何不失真；性能表现为显著提升的时间一致性指标。

**⚠️ 局限性**

局限性：仅在单一视频上验证，未覆盖多样面部特征与表情；未解决 VR 部署中的立体一致性、延迟预算和头显特定渲染失真问题；依赖 FLAME 头部拓扑，眼镜不在该拓扑内，需额外追踪器支持。

---

## 137. FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding

**arXiv ID:** 2601.17258 | [PDF](https://arxiv.org/pdf/2601.17258v1)

**作者:** João Pereira `[一作]`, David Semedo `[通讯]` (NOVA FCT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对视频异常理解（VAU）的新基准和评估方法，强调事件、实体和位置的细粒度描述；

**💡 创新点**

创新点在于将VAU拆解为What/Who/Where三维结构，并设计了基于LLM的结构化评估指标与自动化数据增强管道；

**🔧 技术方法**

采用Gemini-2.5-Pro等大型视觉语言模型进行事件分解、实体链接与属性提取，并用LLM进行评分；

**📊 数据集**

构建了名为What-Who-Where的新数据集，覆盖1544段监控视频，包含17813个事件、59392个实体及74593个属性；

**📈 对比分析**

与传统n-gram和现有LLM评估方法对比，所提出指标与人类判断的相关性最高，且在五个主流LVLM上显著揭示模型在细粒度事件捕捉上的不足；

**⚠️ 局限性**

局限性包括对低分辨率、噪声较多的异常视频仍难以准确识别细粒度特征，以及模型对异常的偏向性与幻觉现象仍未完全解决。

---

## 138. Decentralized Multi-Agent Swarms for Autonomous Grid Security in Industrial IoT: A Consensus-based Approach

**arXiv ID:** 2601.17303 | [PDF](https://arxiv.org/pdf/2601.17303v1)

**作者:** Samaresh Kumar Singh `[一作]`, Joyjit Roy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在工业物联网环境中提出了一种分布式多智能体蜂群（DMAS）架构，通过边缘 AI 代理实现子毫秒级的威胁检测与响应。

**💡 创新点**

创新点包括基于共识的威胁验证（CVT）轻量级协议、加权投票与邻近距离衰减、以及持续更新的代理信誉系统。

**🔧 技术方法**

使用技术包括边缘计算、轻量级机器学习模型（统计异常、RNN 行为模型、签名匹配）、UDP 多播 P2P 协议、Byzantine 容错共识、以及协议缓冲序列化。

**📊 数据集**

数据集结合了 30 天汽车制造商匿名真实流量、150,000 通过 Metasploit 等工具生成的合成攻击，以及 0 日零日攻击样本。

**📈 对比分析**

与传统集中式 IDS、单点 Edge IDS 和基于 LSTM 的云端 IDS 对比，DMAS 在 2000 设备测试中实现了 0.85 ms 的平均响应、97.3% 的检测准确率、87% 的零日检测率，并在 30% Byzantine 代理时仍保持 95% 以上准确率。

**⚠️ 局限性**

局限包括对网络连通性的依赖、冷启动与 Sybil 攻击风险、对资源受限边缘设备的支持不足，以及对对抗性机器学习攻击的防御仍有提升空间。

---

## 139. High-Rate Quantized Matrix Multiplication: Theory and Practice

**arXiv ID:** 2601.17187 | [PDF](https://arxiv.org/pdf/2601.17187v1)

**作者:** Or Ordentlich `[一作]` (Hebrew University of Jerusalem), Yury Polyanskiy `[通讯]` (MIT)

**通讯引用:** 8724 | [OpenAlex ID](https://openalex.org/A5031031216)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了高分辨率量化矩阵乘法的理论极限与实践方法，针对通用与权重仅量化两种场景给出信息论基准并评估常用量化方案。

**💡 创新点**

创新点在于提出了基于水分配（waterfilling）与旋转不变性的WaterSIC算法，证明其在高分辨率下仅落后信息论极限0.25比特，并将GPTQ与LDLQ视为该框架下的特殊实现。

**🔧 技术方法**

采用信息论高分辨率分析、均匀量化噪声模型、随机旋转、Cholesky分解与层次化向量量化、以及熵编码/球形/矩形包装等技术。

**📊 数据集**

以Llama‑3‑8B模型的15层权重与相应激活样本（WikiText‑2等校准集）为数据集进行评估。

**📈 对比分析**

与信息论极限、INT8/FP8 absmax、NVINT4/FP4、NestQuant等方案对比，实验表明WaterSIC/GPTQ在高率下接近理论极限，误差相对其他方案低约0.2–0.3比特。

**⚠️ 局限性**

局限在于仅针对高分辨率（R≫1）分析，低比特率下的收缩、维度约简与熵编码实现复杂度未充分解决。

---

## 140. C-RADIOv4 (Tech Report)

**arXiv ID:** 2601.17237 | [PDF](https://arxiv.org/pdf/2601.17237v1)

**作者:** Mike Ranzinger `[一作]`, Pavlo Molchanov `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过多教师蒸馏构建C‑RADIOv4模型，整合SigLIP2、DINOv3与SAM3的特征，实现统一的学生网络。

**💡 创新点**

创新点在于引入随机平移的shift‑equivariant损失、MESA与DAMP等正则化，并对摘要特征采用角度分散归一化，解决教师噪声与不平衡问题。

**🔧 技术方法**

技术包括多分辨率训练、FeatSharp上采样、窗口化Attention（ViTDet模式）、shift‑equivariant loss、MESA、DAMP、角度归一化摘要损失。

**📊 数据集**

使用公开基准数据集ImageNet‑1K、ADE20k、VOC、SA‑Co/Gold、SA‑Co/Gold实例分割、Probe3d等进行评估。

**📈 对比分析**

通过与DINOv3、RADIOv2.5及SAM3等模型的零样本准确率、k‑NN准确率以及密集任务的指标比较，C‑RADIOv4在参数量更少的情况下在零样本、k‑NN、语义分割、深度、法向等任务上均达到或超过同类模型。

**⚠️ 局限性**

局限性包括在SAM3替换实验中仍存在性能落差、对窗口尺寸选择敏感、对固定模式噪声的抑制仍不完美，以及多教师分布不均时仍可能导致学习偏差。

---

## 141. Between Search and Platform: ChatGPT Under the DSA

**arXiv ID:** 2601.17064 | [PDF](https://arxiv.org/pdf/2601.17064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 142. Superpixel-Based Image Segmentation Using Squared 2-Wasserstein Distances

**arXiv ID:** 2601.17071 | [PDF](https://arxiv.org/pdf/2601.17071v1)

**作者:** Jisui Huang `[一作]` (Capital Normal University), Na Lei `[通讯]` (Dalian University of Technology)

**通讯引用:** 1423 | [OpenAlex ID](https://openalex.org/A5086990531)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于超像素的图像分割方法，利用平方 2‑Wasserstein 距离在超像素层面进行分割。

**💡 创新点**

创新点在于将超像素生成与区域合并统一为一套离散最优传输（OT）框架，使用分布式 OT 距离替代传统均值/方差度量，并通过自适应异质性记忆实现更稳健的合并决策。

**🔧 技术方法**

核心技术包括 Power‑SLIC 超像素生成、OT 距离计算（网络单纯形求解）、优先队列贪婪合并、ROC‑LT 自动选取分割数、以及可选的标记驱动合并。

**📊 数据集**

实验使用三组公开数据集：合成光照视频、36 张 H&E 细胞图像（600×600）和 50 张三阴性乳腺癌 H&E 图像（512×512）。

**📈 对比分析**

与 SAM、SMST、AR、ZZ 等方法比较，SP 在 Dice 系数上最高（例如 36 张图像 88.78%），计算速度比 AR/ZZ 快十倍以上，且与 SAM 的速度相当，整体性能显著优于传统变分/深度模型。

**⚠️ 局限性**

局限在于对弱边界或与背景光照分布高度重叠的区域仍难以准确分割，且对极端噪声或非常细小结构的鲁棒性还有待提升。

---

## 143. BibAgent: An Agentic Framework for Traceable Miscitation Detection in Scientific Literature

**arXiv ID:** 2601.16993 | [PDF](https://arxiv.org/pdf/2601.16993v1)

**作者:** Peiran Li `[一作]` (Texas A and M University), Chaoqun Ni `[通讯]` (University of Wisconsin Madison)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个端到端、可追溯的 agentic 框架，用于自动检测科学文献中的 miscitation（误引用）问题。

**💡 创新点**

创新点包括：① 针对可访问与不可访问（付费墙）文献的双分支验证策略；② 引入 “Evidence Committee” 机制，以社区共识推断隐藏源的真实性；③ 构建统一的 5 类 miscitation 体系结构（Citation Validity、Content Misrepresentation、Scope Extrapolation、Evidence Characterization、Attribution & Traceability）；④ 公开了跨 254 领域、6,350 例的无污染 benchmark（CitationMiscite），用于严苛评估。

**🔧 技术方法**

技术手段主要有：多模态文档解析（OCR + 视觉–语言模型）将 PDF 转为结构化 Markdown；检索模块（bi‑encoder）定位相关文献；LLM 驱动的推理模块进行逐步 zoom‑in 评估；聚合模块（Voting Committee）在不可访问文献上构建多视角共识；以及基于前缀/后缀提示的可追踪推理链生成。

**📊 数据集**

使用的数据集包括：① 自主构建的 6,350 例无污染 benchmark；② 对应的 254 个 JCR 领域领先期刊中 2024–2025 年最被引用文章；③ 公开的多种 LLM（如 GPT‑4、Claude、LLaMA‑2 等）与检索索引。

**📈 对比分析**

与最先进的 LLM 基线（直接输入全文或摘要进行推理）对比，提出框架在 miscitation 识别准确率上提升了 6–10%（具体数字因领域而异），且通过多阶段检索与推理将 token 消耗降低 79.4%，同时生成可解释的证据链，显著提升了可追溯性。

**⚠️ 局限性**

局限性包括：① 对不可访问文献的 “Evidence Committee” 依赖于社区引用数量，稀缺领域可能缺乏足够证据；② 仍需要 LLM 进行推理，易受模型偏差和 hallucination 影响；③ 构建 benchmark 的知识空白协议需持续更新以应对更强 LLM 的记忆能力；④ 处理极其大规模多语言文献时，系统的计算与存储成本仍较高。

---

## 144. Acoustic Field Video for Multimodal Scene Understanding

**arXiv ID:** 2601.17123 | [PDF](https://arxiv.org/pdf/2601.17123v1)

**作者:** Daehwa Kim `[一作]` (Carnegie Mellon University), Chris Harrison `[通讯]` (Carnegie Mellon University)

**通讯引用:** 10494 | [OpenAlex ID](https://openalex.org/A5029290807)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并评估了将空间声场视频（Acoustic Field Video）作为新的多模态输入，用于提升视觉‑语言模型（VLM）的场景理解与问答性能。

**💡 创新点**

创新点在于首次将声学阵列通过 MUSIC 波束成形生成的空间声能图像化，并与 RGB 视频同帧对齐，使 VLM 能直接感知声音来源位置，从而弥补传统音频无空间信息的局限。

**🔧 技术方法**

核心技术包括低成本 MEMS 麦克风阵列、MUSIC 频域波束成形、四个中心频率的声压级（SPL）图生成、图像叠加的 jet 颜色映射，以及使用 Gemini 2.5 Pro 进行零样本视觉‑语言推理。

**📊 数据集**

使用了自建的 402 条问答样本数据集，包含在十种室内/室外场景下同步录制的 RGB 视频、声场视频与双声道音频，数据公开共享。

**📈 对比分析**

对比实验把传统 RGB + 立体声与传统 RGB + 声场视频两种输入送入 Gemini，结果显示答案准确率从 38.3% 提升至 67.4%，人类评审也更偏好声场视频的答案，提升幅度约 29%。

**⚠️ 局限性**

主要限制包括数据集规模有限、仅使用单一阵列几何、MUSIC 波束成形受低信噪比和室内回声影响、VLM 仅通过视觉通道接收声场图像而非直接融合原始声学特征，以及对可迁移性和实时性仍需进一步验证。

---

## 145. From Noise to Insights: Enhancing Supply Chain Decision Support through AI-Based Survey Integrity Analytics

**arXiv ID:** 2601.17005 | [PDF](https://arxiv.org/pdf/2601.17005v1)

**作者:** Bhubalan Mani `[一作]` `[通讯]`, Bhubalan Mani

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并验证了一套基于机器学习的调查问卷可靠性检测框架，过滤低质量/伪造响应。

**💡 创新点**

将逻辑规则、NLP一致性评分与监督分类相结合，实现对结构化供应链调查数据的自动真实性评估。

**🔧 技术方法**

使用随机森林、逻辑回归、XGBoost等分类器，结合规则引擎、BERT句子嵌入及逻辑分数。

**📊 数据集**

对99条来自制造和物流行业的供应链专业人士调查问卷进行手工标注，其中14条被标记为伪造。

**📈 对比分析**

通过80/20训练/测试拆分，评估准确率、精确率、召回率，随机森林在测试集上达到92%准确率、1.00精确率、0.50召回率。

**⚠️ 局限性**

样本量小且“伪造”类不平衡，导致召回率低；且开放式回答简短，限制了NLP特征深度。

---

## 146. Hierarchical Informative Path Planning via Graph Guidance and Trajectory Optimization

**arXiv ID:** 2601.17227 | [PDF](https://arxiv.org/pdf/2601.17227v1)

**作者:** Avraiem Iskandar `[一作]` (University of Waterloo), Stephen L. Smith `[通讯]` (University of Waterloo)

**通讯引用:** 11549 | [OpenAlex ID](https://openalex.org/A5103134580)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种三阶段层次信息路径规划框架，先在离散图上进行全局搜索得到预算可行路径，再对每条边分配局部预算并通过几何+核影响界定信息覆盖区域，最后使用贝塞尔/样条曲线在分配的预算内对每段进行轨迹优化，得到满足预算、障碍约束并最大化高斯过程不确定性降低的连续轨迹。

**💡 创新点**

创新点主要包括：
1) 将图搜索与连续轨迹优化结合的层次化规划框架，既保留全局最优性又避免连续优化陷入局部极小；
2) 利用几何可达椭圆和核影响半径构造信息覆盖上界，转化为最大覆盖问题并通过光滑指标求解局部预算分配；
3) 在段级优化时基于可达椭圆裁剪无关障碍，显著降低约束数量，提高求解速度；
4) 通过平滑最大覆盖和贝塞尔曲线参数化实现高效、可收敛的非线性规划。

**🔧 技术方法**

技术手段包括：
- 高斯过程建模与平方指数核；
- 贝塞尔/样条曲线轨迹参数化；
- CasADi+IPOPT求解非线性规划；
- CMA-ES 及连续空间图优化器作为对比；
- MIQP/MICP离散图规划；
- 结合几何椭圆 + Minkowski 和光滑指示函数的预算分配优化。

**📊 数据集**

实验数据集：
1) 合成障碍密集环境（3.5×3.5 m，12个多边形/圆形障碍，11个顶点手工构建图）；
2) 加拿大北极冰盖 ERA5 日数据（2025 年4月），经 10⁵ 缩放处理后得到约 100 km 单位的离散地图，90 个凸障碍。

**📈 对比分析**

性能对比：
- 在合成环境中，层次规划得到的目标值约 17.8，优于 CMA-ES（25.1）、CasADi‑IPOPT（25.8）和图规划（27.4），分别低约 29 % 与 35 %；
- 运行时间：层次规划 10.2 s，CMA‑ES 204.2 s，IPOPT 92.0 s，层次规划分别比 IPOPT 快 9 倍、比 CMA‑ES 快 20 倍；
- 在北极冰盖实验中，其他连续优化方法无法得到可行轨迹，层次规划成功完成；低预算路径的加权 RMSE 为 0.521，高预算路径为 0.381，显示预算提升可进一步降低误差。

**⚠️ 局限性**

局限性：
1) 依赖预先构造的离散图，无法在运行时动态重规划；
2) 需要先验已知的 GP 超参数，在线自适应更新尚未实现；
3) 预算假设为确定值，未考虑行进成本的不确定性；
4) 仅针对单机器人单目标场景，扩展到多机器人或多目标时需额外设计；
5) 对极端高相关或低相关场景的鲁棒性尚未充分验证。

---

## 147. Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations

**arXiv ID:** 2601.17087 | [PDF](https://arxiv.org/pdf/2601.17087v1)

**作者:** Preethi Seshadri `[一作]` (University of California Irvine), Seraphina Goldfarb-Tarrant `[通讯]` (Cohere)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在美国、印度、肯尼亚和尼日利亚招募真实用户，对τ‑Bench零售任务进行对比实验，评估LLM模拟用户在不同人口群体下的可靠性、鲁棒性与公平性。

**💡 创新点**

创新点在于：①系统性检验模拟用户对不同民族、方言、年龄群体的偏差；②提出并应用基于任务难度的Expected Calibration Error（ECE_Human‑LLM）衡量模拟与真实用户的校准误差；③揭示模拟用户与真实用户在问答频率、礼貌度、错误类型与归因上的显著差异。

**🔧 技术方法**

技术方法包括：①使用GPT‑4o、Sonnet3.7/4.5、Kimi‑K2‑Thinking等多种用户LLM进行鲁棒性检验；②采用GEE回归模型分析方言、年龄、国别对成功率与ECE的影响；③计算任务成功率、错误分布以及对话结构指标（回合数、动作数、词数）。

**📊 数据集**

使用的数据集为τ‑Bench零售任务（115个任务，抽样18个按难度分层），以及自定义的用户交互日志；受试者为约160名英语熟练的受访者（美国白人SAE、美国黑人AAVE、印度、肯尼亚、尼日利亚各18-34岁）。

**📈 对比分析**

比较方法：在同一组任务中分别让LLM模拟用户和真实用户交互，并通过自动化评估脚本计算成功率；利用ECE_Human‑LLM量化模拟与真实用户之间的性能差距。结果显示：不同用户LLM间成功率差异可达9个百分点；对真实用户的校准误差最高可达20%，并在AAVE和老年组中尤为严重；错误归因表明真实用户导致更多用户端错误，而模拟用户导致更多代理端错误。

**⚠️ 局限性**

局限性包括：①仅在英语环境下测试，未覆盖多语言场景；②仅评估单一域（零售）和单一代理模型（GPT‑4o），未探究其它域与代理的差异；③非美国受试者仅覆盖18-34岁，缺乏年龄跨国对比；④未检验不同用户LLM对不同代理的影响；⑤样本规模相对有限，可能影响统计显著性。

---

## 148. Bayesian Robust Financial Trading with Adversarial Synthetic Market Data

**arXiv ID:** 2601.17008 | [PDF](https://arxiv.org/pdf/2601.17008v1)

**作者:** Haochong Xia `[一作]` (Nanyang Technological University), Bo An `[通讯]` (Nanyang Technological University)

**通讯引用:** 6684 | [OpenAlex ID](https://openalex.org/A5017743551)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个基于贝叶斯博弈的强化学习框架，结合宏观条件的GAN生成器和对抗式训练，来学习在宏观经济不确定性下鲁棒的交易策略。

**💡 创新点**

创新点包括：① 将宏观经济指标作为主要控制变量的条件GAN，生成多样且逼真的市场情景；② 把交易过程建模为两人零和贝叶斯马尔可夫游戏，让对抗代理通过扰动宏观变量逼出最坏情况；③ 采用量化信念网络与贝叶斯神经虚拟自举（Bayesian NFSP）求解鲁棒完美贝叶斯均衡；④ 通过量化分位数估计隐藏市场状态，提升对不可观测宏观冲击的适应能力。

**🔧 技术方法**

使用的技术包括：TimeGAN/Informer/Transformer LSTM 架构的宏观条件生成器；GAN对抗训练与时序预测；贝叶斯神经虚拟自举（Bayesian NFSP）实现 min‑max 优化；量化信念网络（QBN）对隐藏状态进行分位数估计；标准强化学习（DQN、Q‑learning）与鲁棒强化学习框架；评估指标为年化回报率、夏普比率和最大回撤。

**📊 数据集**

数据集：9只涵盖商品、外汇和股票指数的ETF，历史数据自首个交易日起；宏观经济指标46个来源于美国联邦储备经济数据（FRED）；训练/验证/测试划分分别为 2018‑2020、2021‑2024，剩余用于训练。

**📈 对比分析**

与 9 个基线（如 Buy‑and‑Hold、DQN、Robust Trading Agent、RoM‑Q、RARL 等）以及 2 个消融实验（无贝叶斯 NFSP、无对抗代理）进行比较，使用 ARR、SR、MDD 三项指标。实验结果显示，本方法在所有 9 只ETF上均优于基线，ARR 与 SR 均显著提升，MDD 明显降低，Wilcoxon 检验 p<0.05。

**⚠️ 局限性**

局限性：① 生成器仍基于历史宏观变量，可能无法覆盖所有突发宏观冲击；② 对宏观数据的依赖使模型对指标质量敏感；③ 计算成本较高（每只ETF 22 小时训练，推理 2.7 ms/步），在更大规模或实时高频场景下可能受限；④ 仅验证了 ETF、外汇与指数三类资产，跨资产类别的一般性仍需进一步验证。

---

## 149. TrojanGYM: A Detector-in-the-Loop LLM for Adaptive RTL Hardware Trojan Insertion

**arXiv ID:** 2601.17178 | [PDF](https://arxiv.org/pdf/2601.17178v1)

**作者:** Saideep Sreekumar `[一作]` (New York University Abu Dhabi), Johann Knechtel `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 1011 | [OpenAlex ID](https://openalex.org/A5052751303)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 TrojanGYM，一种基于 LLM 的闭环 RTL 级硬件木马生成框架，能自动创建多样化木马并通过 GNN 检测器反馈持续迭代改进，最大化检测盲区曝光。

**💡 创新点**

创新点在于：① 将 LLM 生成、语法校验、功能等价验证与 GNN 检测器的实时反馈集成，形成闭环迭代；② 开发 Robust‑GNN4TJ，提升图抽取、训练和集成推理稳定性；③ 在 RTL 级进行属性驱动的木马插入，并通过多模型（GPT‑4、Gemini‑2.5Pro、LLaMA‑3.3‑70B）共同完成。

**🔧 技术方法**

使用技术包括：大型语言模型（LLM）生成 RTL 代码、语法与功能等价自动检查、GNN（Robust‑GNN4TJ）对 RTL 图进行木马检测、集成推理、以及迭代式反馈调整生成策略。

**📊 数据集**

数据集涵盖：884 条 VeriGen 纯净 RTL、3,536 条通过 GHOST+GPT‑4 生成的木马样本（HT1–HT4），以及 SRAM、AES‑128、UART 三个工业级 RTL 作为评测基准。

**📈 对比分析**

与原始 GNN4TJ、GHOST 及 TrustHub 等基线比较，Robust‑GNN4TJ 在单模型下检测率为 0%，集成模型提升至 60%；在 TrojanGYM 生成的基准上，木马逃逸率可达 83.33%（最高），而传统基准仅能检测 0%–30% 左右。

**⚠️ 局限性**

局限性包括：仍依赖 GNN 检测器反馈，难以直接评估非图模型的鲁棒性；受 RTL 级别限制，无法覆盖更深层次的门级或工艺级木马；LLM 的安全限制与算力开销可能限制大规模部署；以及对特定 HT 类型（HT2、HT3）在结构复杂设计中的误报与语法错误问题。

---

## 150. Frame-Guided Synthetic Claim Generation for Automatic Fact-Checking Using High-Volume Tabular Data

**arXiv ID:** 2601.17232 | [PDF](https://arxiv.org/pdf/2601.17232v1)

**作者:** Jacob Devasier `[一作]` (University of Texas at Arlington), Chengkai Li `[通讯]` (University of Texas at Arlington)

**通讯引用:** 3775 | [OpenAlex ID](https://openalex.org/A5084878734)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过从OECD高容量表格数据中自动生成多语言合成事实核查语句，构建了78,503条包含英、汉、西班牙语和印地语的事实核查数据集；

**💡 创新点**

创新点在于：①基于语义框架系统化定义六类事实核查语句；②采用程序化选择表格数据点并交由大模型生成自然语言语句，实现大规模、多语言的合成数据；③提出基于SQL查询的基线检索与推理框架，并对模型内部知识泄露进行分析；

**🔧 技术方法**

使用技术包括：大语言模型（Qwen3、Llama 3.3）用于生成与判断；句子拆分与子语句提取；嵌入模型（GTE多语言）与重排器进行表格检索；SQL生成与执行进行证据提取；

**📊 数据集**

数据集为434个OECD统计表（平均约596,552行/表），涵盖健康、环境、金融等多领域；

**📈 对比分析**

基线系统在测试集上的总体准确率仅约9.3%（在原始表上的36%，在检索表上的20%），远低于TabSQLify（约6%），表明该任务对检索与精确提取极具挑战性；

**⚠️ 局限性**

局限性包括：①仅覆盖OECD表格，缺乏跨领域或更大多样性；②合成语句与真实查询差距可能影响外推；③检索与SQL生成对大表仍不成熟，导致高“信息不足”率；④多语言质量差异（印地语较低）；

---

## 151. TelcoAI: Advancing 3GPP Technical Specification Search through Agentic Multi-Modal Retrieval-Augmented Generation

**arXiv ID:** 2601.16984 | [PDF](https://arxiv.org/pdf/2601.16984v1)

**作者:** Rahul Ghosh `[一作]` (Generative AI Innovation Center), Hazar Aouad `[通讯]` (Bouygues Telecom)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了TelcoAI，一个面向3GPP技术规范的多模态检索增强生成系统。

**💡 创新点**

创新点在于引入了面向章节的分块、结构化查询规划、元数据引导检索以及文本与图表的多模态融合。

**🔧 技术方法**

主要技术包括LLM驱动的检索增强生成、分块策略、查询规划算法以及多模态融合模型。

**📊 数据集**

使用的数据集为3GPP规范文档（.docx格式）和专家精心设计的查询集合。

**📈 对比分析**

与现有基准相比，TelcoAI在召回率、声明召回率和可靠度上分别达到87%、83%和92%，提升了16%。

**⚠️ 局限性**

局限性包括仅支持.docx文件，缺乏多模态格式扩展；仅处理单轮查询；未对系统效率和推理延迟进行优化。

---

## 152. Bowling Online: Accounting for Civil Society Reshaped into Streamlined Photons within a Fiber Network

**arXiv ID:** 2601.17139 | [PDF](https://arxiv.org/pdf/2601.17139v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 153. Exploring Needs and Design Opportunities for Proactive Information Support in In-Person Small-Group Conversations

**arXiv ID:** 2601.17240 | [PDF](https://arxiv.org/pdf/2601.17240v1)

**作者:** Shaoze Zhou `[一作]` (Florida International University), Chen Chen `[通讯]` (Florida International University)

**通讯引用:** 62663 | [OpenAlex ID](https://openalex.org/A5100352749)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过参与式设计与技术探针，评估了在混合现实头显中为小组对话提供主动信息支持的可行性与效果。

**💡 创新点**

创新点在于首次将主动AI信息推送与多人实时对话结合，并提出了动态可视化、非言语提示与自适应信息生成的设计机会。

**🔧 技术方法**

采用Meta Quest 3S头显、Wizard‑of‑Oz手动控制、GPT‑4o‑mini生成信息、Azure实时转写与说话者分离技术实现支持信息的呈现与交互。

**📊 数据集**

使用的“数据集”为10名受试者在两小组（3人）与一大组（4人）中产生的对话文本、录音与交互日志。

**📈 对比分析**

通过访谈、主题分析与对照组对话对比，定性表明主动支持提升聚焦度与参与度，性能以用户体验与交互效率指标呈现。

**⚠️ 局限性**

局限包括样本量小、键盘输入对对话流的干扰、非MR伙伴对可视化干扰的评估不足，以及未进行严格实验对照验证。

---

## 154. Safety, Mobility, and Environmental Impacts of Driver-Assistance-Enabled Electric Vehicles: An Empirical Study

**arXiv ID:** 2601.17256 | [PDF](https://arxiv.org/pdf/2601.17256v1)

**作者:** Gabriel Geffen `[一作]` (University of Arizona), Yao-Jan Wu `[通讯]` (University of Arizona)

**通讯引用:** 1604 | [OpenAlex ID](https://openalex.org/A5032118254)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用真实高频的 OpenACC 数据集，比较了装备 ACC 的电动汽车（EV）与内燃机汽车（ICEV）在交通效率、安全性和环境影响上的差异。

**💡 创新点**

创新点包括：
1) 引入动态时间规整（DTW）对主动车轨迹进行匹配，确保跟随对比实验在主车行为上高度相似；
2) 将微观行驶特征（ASV、TTC、DRAC、v–s 曲线）与宏观交通指标（通过 VT‑Micro 计算排放）相结合，形成完整的评估框架；
3) 通过实测数据验证了 EV 在 ACC 场景下能够显著提升流动性、减少事故风险并降低 ICEV 的排放。

**🔧 技术方法**

主要技术手段：动态时间规整（DTW）用于轨迹匹配；速度–间距（v–s）分段线性拟合；平均速度波动（ASV）计算；时间到碰撞（TTC）与冲击减速（DRAC）安全指标；VT‑Micro 排放模型进行燃料与尾气评估。

**📊 数据集**

使用了 OpenACC 实验数据集，包含 27 款车型、10 Hz 轨迹、覆盖 59 组车队共 326 对车轮，约 40 小时高精度驾驶数据。

**📈 对比分析**

比较方法：先用 DTW 选取主车轨迹相似度最高的 10 对 EV 跟随车与 10 对 ICEV 跟随车；随后计算 ASV、v–s 曲线、TTC/DRAC 关键事件率及 VT‑Micro 排放。结果表明：
- EV 的 ASV 低 0.393 m/s（比 ICEV 的 0.663 m/s 更平稳）；
- 关键间距缩小 60%（EV 6.17 m vs ICEV 15.03 m）；
- 1 s TTC 事件率下降 85.8%，5 m/s² DRAC 下降 86.6%；
- EV 领头车导致的 ICEV 排放均降低 6–26%。

**⚠️ 局限性**

局限性：
1) 仅研究纵向跟随行为，未考虑换道、合流等横向动作；
2) 数据量相对有限，EV 样本仅 57 辆跟随车，未涵盖更大规模混合交通情景；
3) 数据来源为欧洲封闭/低流量实验道，可能与真实道路环境存在差异。

---

## 155. Big Deal cancellations and scholarly publishing: Insights from faculty and graduate student interviews

**arXiv ID:** 2601.17033 | [PDF](https://arxiv.org/pdf/2601.17033v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 156. AstroTimer: Rethinking Non-Access Stratum Timers in LEO Constellations

**arXiv ID:** 2601.17195 | [PDF](https://arxiv.org/pdf/2601.17195v1)

**作者:** Arshiya Rezaie Hezaveh `[一作]` (University of Manitoba), Peng Hu `[通讯]` (University of Manitoba)

**通讯引用:** 684 | [OpenAlex ID](https://openalex.org/A5101976436)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 AstroTimer 框架，用于根据低地轨道（LEO）星座的链路波动、处理延迟和网络功能部署动态调整 NAS 计时器。

**💡 创新点**

创新点在于：①针对 LEO 特性设计轻量级自适应计时器模型；②推导出低计算量的闭式计时器尺寸公式；③同时优化注册过程中的 watchdog 与 backoff 计时器，避免信令风暴。

**🔧 技术方法**

使用了链路质量与时延的统计建模、网络功能放置分析、闭式数值求解方法，并在仿真平台上实现了该框架。

**📊 数据集**

主要使用仿真生成的数据（基于 5G NR 规范的链路状态和 UE 能耗模型），未使用公开真实数据集。

**📈 对比分析**

通过与 3GPP 默认计时器配置对比，评估指标包括注册延迟、重试次数和 UE 能耗，实验显示 AstroTimer 在三方面均有显著降低（注册时间降低 ~30%、重试频率下降 ~40%、能耗降低 ~25%），并有效抑制了信令拥塞。

**⚠️ 局限性**

局限性包括：①模型假设链路波动可由预先估计的统计参数描述，实际极端波动情况仍需进一步验证；②仅针对注册过程的计时器进行优化，其他 NAS 过程的适配性尚未评估；③需要在真实 LEO 星座环境中进行实验验证。

---

## 157. Breaking Task Impasses Quickly: Adaptive Neuro-Symbolic Learning for Open-World Robotics

**arXiv ID:** 2601.16985 | [PDF](https://arxiv.org/pdf/2601.16985v1)

**作者:** Pierrick Lorang `[一作]` (Tufts University), Pierrick Lorang `[通讯]` (Tufts University)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5094166137)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种集层级动作抽象、符号目标导向学习与好奇心驱动想象于一体的神经符号框架，以提升机器人在开放世界中的快速适应能力。

**💡 创新点**

创新点在于：①构建多层嵌套的执行器结构，能针对局部或全局新奇情况动态生成或更新技能；②使用符号层面的目标空间进行控制学习，结合 HER 提升样本效率；③通过好奇心驱动的世界模型生成符号操作，实现对未知状态的主动探索与计划扩展；④将这些模块统一集成到一个完整的学习体系。

**🔧 技术方法**

主要技术包括：符号规划（STRIPS、PDDL）、强化学习（MDP、SMDP）、层级强化学习（HRL）、符号目标条件化策略、HER（Hindsight Experience Replay）、好奇心奖励（intrinsic reward）与符号化世界模型构建。

**📊 数据集**

实验数据集：RoboSuite 的 Pick‑and‑Place Can 任务和 CARLA 自动驾驶仿真环境，分别用于机器人操作与自动驾驶场景。

**📈 对比分析**

与最佳基线方法（PRM&ICM、HyGOAL、Hierarchy 的对照基线）对比，框架在 15 个新奇场景中 13 场景性能提升，平均成功率提高 20%，收敛时间平均缩短 54%；具体如 Pick‑and‑Place 的成功率从 0.38 提升到 0.76，适应步数从 400 降至 290；在 CARLA 的多种障碍场景中，成功率基本保持或提升，同时适应步数从 700 降至 56‑37。

**⚠️ 局限性**

局限性：实验仍停留在预实验阶段，仅覆盖有限的任务与环境；对复杂多模态新奇情况的处理机制尚未完善；缺乏对人类演示或反馈的融合，限制了在实际部署中的快速迁移与鲁棒性。

---

## 158. Private Accountability in the Age of Artificial Intelligence

**arXiv ID:** 2601.17013 | [PDF](https://arxiv.org/pdf/2601.17013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 159. Attention-Based Variational Framework for Joint and Individual Components Learning with Applications in Brain Network Analysis

**arXiv ID:** 2601.17073 | [PDF](https://arxiv.org/pdf/2601.17073v1)

**作者:** Yifei Zhang `[一作]` (Yale University), Zhengwu Zhang `[通讯]` (University of North Carolina)

**通讯引用:** 1231 | [OpenAlex ID](https://openalex.org/A5050808730)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出CM-JIVNet框架，实现了结构和功能脑网络的联合与个体化潜在表示学习。

**💡 创新点**

创新点在于将变分自编码器与JIVE式分解结合，并在潜在空间引入多头注意力实现非线性联合-个体解耦。

**🔧 技术方法**

使用了卷积VAE、残差网络、注意力融合、Poisson与Gaussian似然以及互信息正则化等深度生成技术。

**📊 数据集**

使用Human Connectome Project Young Adult（HCP‑YA）1065名受试者的结构连接（SC）和功能连接（FC）数据。

**📈 对比分析**

通过与CNN‑MLP、MLP‑Attention、GATE等基线对比，CM‑JIVNet在MSE、SSIM、FID、相关系数等指标上均实现了最优或同等性能，缺失模态预测也优于现有方法。

**⚠️ 局限性**

局限性包括仅处理两种模态、仅在健康成人样本验证、未测试临床人群或其他模态，并且VAE生成质量受限，未来可考虑扩展到更多模态和更大规模数据。

---

## 160. Conservative & Aggressive NaNs Accelerate U-Nets for Neuroimaging

**arXiv ID:** 2601.17180 | [PDF](https://arxiv.org/pdf/2601.17180v1)

**作者:** Inés Gonzalez-Pepe `[一作]` (Concordia University), Tristan Glatard `[通讯]` (Concordia University)

**通讯引用:** 7056 | [OpenAlex ID](https://openalex.org/A5075952219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了CNN中数值不确定性导致的冗余计算，提出了Conservative & Aggressive NaNs及NaN卷积，在U‑Net等模型中通过将不稳定体素标记为NaN来跳过无意义运算，从而加速推理。

**💡 创新点**

首次将NaN传播与卷积操作结合，设计两种不同程度的NaN策略（保守和积极）以及NaN卷积，能够在保持精度的同时显著减少卷积次数。

**🔧 技术方法**

采用Monte Carlo Arithmetic（Verrou）分析数值不确定性，改写PyTorch的max pool/unpool与卷积实现，引入NaN阈值跳过机制。

**📊 数据集**

在FastSurfer（脑部分割）、FONDUE（MRI去噪）、MNIST（手写数字分类）与Xception（ImageNet分类）四个模型和对应的数据集上验证。

**📈 对比分析**

与原始PyTorch实现对比，测量跳过卷积比例、推理速度提升；保守NaNs可跳过约30%卷积，积极NaNs可跳过69%卷积；在脑部分割中实现1.67×的平均推理加速，保持Dice/PSNR不下降。

**⚠️ 局限性**

仅在数值不稳定区域有效，对多样化RGB图像或需要精细细节的任务效果有限；过度积极策略易导致精度下降，需要精细阈值调节；目前仅在CPU/GPU软件层面实现，未结合稀疏张量或硬件加速。

---

## 161. MathMixup: Boosting LLM Mathematical Reasoning with Difficulty-Controllable Data Synthesis and Curriculum Learning

**arXiv ID:** 2601.17006 | [PDF](https://arxiv.org/pdf/2601.17006v1)

**作者:** Xuchen Li `[一作]` (Chinese Academy of Sciences), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14319 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了难度可控的数学推理数据集 MathMixupQA，并提出基于混合与分解的生成方法；

**💡 创新点**

通过可调难度的混合与分解生成实现了显式难度梯度，支持课程学习；

**🔧 技术方法**

使用 BGE 嵌入、GPT‑4o 混合生成、QwQ‑32B 解答、自动自检与手工筛选等技术；

**📊 数据集**

以 MATH 和 AMC‑AIME 作为种子数据集；

**📈 对比分析**

对 Qwen2.5‑7B、InternLM2.5‑7B、LLaMA3.1‑8B 在七个数学基准上进行 SFT 与课程学习实验，MathMixup‑CL 平均提升至 52.6% 并刷新 SOTA；

**⚠️ 局限性**

生成与验证过程仍可能残留错误，仅检查 10%，数据规模和领域覆盖仍有限。

---

## 162. Low-Rank Tensor Approximation of Weights in Large Language Models via Cosine Lanczos Bidiagonalization

**arXiv ID:** 2601.17112 | [PDF](https://arxiv.org/pdf/2601.17112v1)

**作者:** A. El Ichi `[一作]` (Universite du Littoral Cote d'Opale), K. Jbilou `[通讯]` (Universite du Littoral Cote d'Opale)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于c‑product张量压缩框架TLASER，用于对Transformer模型的权重进行层选择性低秩张量分解，从而在不需再训练的情况下压缩大型语言模型。

**💡 创新点**

创新点在于将传统的矩阵级LASER方法推广到张量空间，利用c‑product的DCT变换捕获跨注意力头与FFN块的多维相关性；同时引入张量Lanczos双对角化实现只计算主奇异元，显著降低计算成本。

**🔧 技术方法**

核心技术包括：c‑product及其c‑SVD、张量Lanczos双对角化、张量层选择性秩约束、以及张量化Transformer权重（按注意力头/FFN块进行三阶张量化）。

**📊 数据集**

实验数据集为GPT‑J‑6B模型（28层）与TruthfulQA基准，使用5,882条样本（N=500子集）进行评估。

**📈 对比分析**

与传统LASER对比，TLASER在单层干预时提升了约0.4%准确率，且保持了更低的损失；在多层干预时则略逊一筹，但总体保持相近性能，表明其在细粒度压缩上的优势。

**⚠️ 局限性**

局限性包括：需要手动选择压缩层和秩阈值；对多层联合压缩的效果不一定比LASER更好；对极大模型的张量化与Lanczos计算仍存在显著内存/时间开销。

---

## 163. Uncertainty Quantification for Named Entity Recognition via Full-Sequence and Subsequence Conformal Prediction

**arXiv ID:** 2601.16999 | [PDF](https://arxiv.org/pdf/2601.16999v1)

**作者:** Matthew Singer `[一作]` (North Carolina State University), Karl Pazdernik `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5041535920)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套基于自适应共形预测的命名实体识别不确定性量化框架，能够为每句完整标签序列或实体子序列生成覆盖率可控的预测集。

**💡 创新点**

创新点在于：①将共形预测应用于序列标注任务，提供全句和子句级别的覆盖保证；②设计多种非一致性得分与组合策略（naive、conditional、RAPS）以控制预测集大小；③通过语言、句长和实体数等协变量进行分层校准，提升多语言、多长度场景的可靠性。

**🔧 技术方法**

主要技术包括：线性链CRF+BiLSTM-Transformer模型、Beam Search解码、共形预测（inductive与adaptive）、多种非一致性得分（nc1、nc2、nc3、组合式）以及Šidák校正以控制族错误。

**📊 数据集**

实验使用三大基准数据集：CoNLL++（英文）、CoNLL-Reduced（合并实体类型）、WikiNEuRAL（九语种），并在四个公开NER模型上评估（Babelscape、dslim/bert-base、Jean-Baptiste/roberta-large、tner/roberta-large-ontonotes5）。

**📈 对比分析**

与基线单点预测对比，所提出的预测集在95%覆盖率下实现了平均覆盖率均近似目标，且预测集大小普遍小于传统nc2方法；分层校准显著降低覆盖误差；在不同模型/数据集上，覆盖率与模型训练语料匹配度相关（如Babelscape在WikiNEuRAL表现最佳，Jean-Baptiste在CoNLL表现最佳）。

**⚠️ 局限性**

局限性包括：①共形预测的计算开销较高，尤其在Beam Search和多层非一致性组合时；②分层校准需要足够的校准样本，长句或稀有语言的划分可能导致样本不足；③对大型LLM或编码解码结构的适配尚未验证；④当前方法仅对NER输出进行校准，未探讨其在后续下游任务中的不确定性传播。

---

## 164. Online parameter estimation for the Crazyflie quadcopter through an EM algorithm

**arXiv ID:** 2601.17009 | [PDF](https://arxiv.org/pdf/2601.17009v1)

**作者:** Yanhua Zhao `[一作]` `[通讯]`, Yanhua Zhao

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在Crazyflie四旋翼无人机的SDE动力学模型基础上，加入随机噪声，利用扩展卡尔曼滤波（EKF）估计状态，并通过线性二次高斯（LQG）控制实现轨迹跟踪；随后采用期望最大化（EM）算法实现质量和惯性矩的离线与在线参数估计，分别在仿真中评估不同传感器观测源对估计精度的影响。

**💡 创新点**

创新点包括：① 将EM算法与EKF、LQG结合，首次实现四旋翼无人机的实时参数估计；② 提供离线与在线两种仿真框架，比较两者对参数收敛范围的差异；③ 通过三种传感器（EKF状态估计、全状态观测、部分状态观测）系统性评估观测完整性对质量与惯性矩估计的影响。

**🔧 技术方法**

使用的核心技术包括：随机微分方程（SDE）建模、扩展卡尔曼滤波、Rauch–Tung–Striebel平滑器、线性二次高斯（LQG）控制、期望最大化（EM）算法、仿真平台与MATLAB/Simulink实现。

**📊 数据集**

本文仅在仿真环境中生成飞行轨迹与传感器噪声数据，共20次循环，用于验证算法效果；未使用公开的真实飞行数据集。

**📈 对比分析**

通过在同一仿真设置下进行20次循环，对质量与惯性矩估计结果进行统计，比较估计区间与真实值的偏差；并将EKF的平均位置误差与文献Faessler、Edwin等结果对比，证明本方法在离线估计下平均位置误差约比对手小29%，但在线估计因观测窗口限制，惯性矩收敛范围扩大约2–6倍。

**⚠️ 局限性**

主要局限：① 仅在仿真中验证，缺乏真实硬件实验；② 在线估计使用有限观测窗口导致精度下降；③ 传感器观测缺失（如Euler角）会显著影响质量估计；④ 模型假设小角度，未考虑大角度运动；⑤ 未对多目标任务与实时控制耦合进行深入研究。

---

## 165. Learning with Geometric Priors in U-Net Variants for Polyp Segmentation

**arXiv ID:** 2601.17331 | [PDF](https://arxiv.org/pdf/2601.17331v1)

**作者:** Fabian Vazquez `[一作]` (University of Texas Rio Grande Valley), Pengfei Gu `[通讯]` (Sewickley Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种几何先验引导模块（GPM），通过在U‑Net结构中注入由VGGT微调得到的深度先验来实现精准的息肉分割。

**💡 创新点**

创新点在于：①利用VGGT生成符合内镜场景的高质量深度图；②设计双向交叉更新与自更新模块，将几何先验与特征相互融合；③模块可插拔，兼容CNN、Transformer及Mamba U‑Net变体。

**🔧 技术方法**

采用视觉几何预训练变换器（VGGT）进行深度估计，结合空间与通道注意力机制，构建交叉更新（CUB）和自更新（SUB）模块，集成至U‑Net系列网络。

**📊 数据集**

数据集包括：模拟ColoDepth（用于VGGT微调），Scenario（评估深度精度），以及五个公开息肉分割数据集（Kvasir‑SEG、ClinicDB、ColonDB、ETIS、CVC‑300）用于最终分割实验。

**📈 对比分析**

与基线U‑Net、U‑Net v2和VM‑UNetV2比较，GPM在所有五个数据集上均提升DSC和IoU；例如在Kvasir‑SEG上U‑Net从85.26%提升至88.48%，VM‑UNetV2 IoU从82.85%提升至84.34%；深度估计上微调VGGT在Scenario集上达δ1=0.863、RMSE=3.812mm，显著优于传统方法。

**⚠️ 局限性**

局限性包括：①深度先验依赖于模拟数据，可能存在域迁移不完全问题；②模块虽轻量但仍略增参数和FLOPs；③缺乏对其他深度估计器或多模态先验的系统性评估。

---

## 166. Artificial Intelligence in Spanish Gastroenterology: high expectations, limited integration. A national survey

**arXiv ID:** 2601.17011 | [PDF](https://arxiv.org/pdf/2601.17011v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 167. Automated Classification of Research Papers Toward Sustainable Development Goals: A Boolean Query-Based Computational Framework

**arXiv ID:** 2601.16988 | [PDF](https://arxiv.org/pdf/2601.16988v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 168. Are We Evaluating the Edit Locality of LLM Model Editing Properly?

**arXiv ID:** 2601.17343 | [PDF](https://arxiv.org/pdf/2601.17343v1)

**作者:** Wei Liu `[一作]` (National University of Singapore), Wee Sun Lee `[通讯]` (National University of Singapore)

**通讯引用:** 5684 | [OpenAlex ID](https://openalex.org/A5071864357)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对大语言模型（LLM）的知识编辑中“编辑局部性”（specificity）评估进行系统评审，发现现有基于确定答案的评估方式存在多项根本性缺陷，随后提出了一套无参考答案的连续性行为评估协议，并通过KL散度与Top‑k重叠度量实现对编辑后模型行为变化的敏感捕捉。

**💡 创新点**

创新点在于：① 对现有基于地面真值的specificity评估方法进行概念与经验双重剖析；② 提出以行为差异为核心的评估框架，将编辑效果直接映射到预编辑与后编辑输出分布的差异；③ 通过可调k值的Top‑k重叠度量实现对“编辑局部性”连续可调的、无语言偏差的评估；④ 在大规模实验中验证新度量与编辑正则项的高度相关性与更好的区分度。

**🔧 技术方法**

使用的技术主要包括：
- Locate‑then‑Edit（LTE）编辑框架（如MEMIT、RL‑edit等）
- 计算KL散度（KL‑divergence）以及Top‑k重叠度量来评估行为差异
- 通过Kendall’s τ相关系数衡量度量与编辑正则强度、不同方法间排名的一致性
- 训练与评估在LLama‑3‑8B‑Instruct、GPT2‑XL、Qwen2.5‑7B‑Instruct等开源LLM上进行

**📊 数据集**

实验所用数据集：
- MCF（Multi‑Counterfact）
- ZsRE（Zero‑shot Relation Extraction）
- 以及公开的对照实验中使用的其他知识编辑基准数据集（如Wiki等）

**📈 对比分析**

方法比较：对10种主流编辑方法（MEMIT、WISE、RECT、EMMET、PMET、PRUNE、Adaedit、Alphaedit、NAMET、RLedit）在MCF与ZsRE上进行基准测试。
- 传统基于地面真值的specificity（S‑acc、C‑acc/T‑acc）在不同方法之间区分度低、排名不稳定；
- 新的无参考specificity度量（KL‑divergence、Top‑1/5/10）对编辑正则强度敏感，相关性几乎为1；
- 在方法间的rank‑stability上，GT‑free度量的Kendall’s τ与目标知识注入指标（Efficacy、Generalization）相近，表明其评估结果更具可解释性和一致性。

**⚠️ 局限性**

局限性：
- 采用最后一个token的logits作为查询表示，未能完整捕捉长距离上下文和多样化输出；
- 以特定编辑正则项强度为参照来评估度量敏感度，正则强度并非唯一决定specificity的因素；
- 评估仍基于单一输出分布，未考虑多样化或生成多模态答案的情况；
- 仅在若干开源LLM上验证，尚需在更大规模或更不同架构模型上进一步检验。

---

## 169. SonoEdit: Null-Space Constrained Knowledge Editing for Pronunciation Correction in LLM-Based TTS

**arXiv ID:** 2601.17086 | [PDF](https://arxiv.org/pdf/2601.17086v1)

**作者:** Ayush Pratap Singh `[一作]` (TU Darmstadt), Sudarshan Kamath `[通讯]` (Smallest AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于声学因果追踪与零空间约束的无训练TTS发音纠错框架SonoEdit；

**💡 创新点**

创新点在于将声学因果追踪精准定位发音相关层，并在声学子空间中做零空间约束的闭式权重更新，既实现单词级别校正又避免全局遗忘；

**🔧 技术方法**

使用了Acoustic Causal Tracing、Null‑Space Pronunciation Editing（Rank‑One 关闭式更新）、SVD 计算 null space、以及对比的 LoRA/ROME 等PEFT技术；

**📊 数据集**

利用Orpheus‑TTS/​Sesame‑TTS预训练模型、LibriTTS数据集构建 null space、以及自建的 HardNoun‑300 多语言实体集进行实验；

**📈 对比分析**

与全量微调、LoRA、ROME 等方法对比，SonoEdit 在 Target‑WER 约 2.8%（接近 FFT 2.1%），Global‑WER 仅 3.15%，SIM 0.99，MOS 4.18，表现出色且无显著语音退化；

**⚠️ 局限性**

局限包括需预先计算 null space，分布漂移可能削弱正交性；仅能校正单词级别发音，对大规模批量修正和系统性偏差修正效果有限。

---

## 170. iFSQ: Improving FSQ for Image Generation with 1 Line of Code

**arXiv ID:** 2601.17124 | [PDF](https://arxiv.org/pdf/2601.17124v1)

**作者:** Bin Lin `[一作]` (Peking University), Li Yuan `[通讯]` (Peking University)

**通讯引用:** 17596 | [OpenAlex ID](https://openalex.org/A5100700791)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过改进Finite Scalar Quantization（FSQ）中的激活函数，实现了既能产生高质量连续潜在也能生成离散token的统一tokenizer，并以此对比了自回归（AR）与扩散模型的生成性能；

**💡 创新点**

创新点在于将tanh替换为1.6倍sigmoid的分布匹配激活，使FSQ在保持均匀分布的同时避免激活塌陷，从而实现信息效率与重构精度的双赢；

**🔧 技术方法**

采用的核心技术包括改进的FSQ（iFSQ）、自回归模型LlamaGen-REPA、扩散模型DiT，以及基于DINOv2的表示对齐（REPA）；

**📊 数据集**

实验数据集主要使用ImageNet和COCO的训练/验证集；

**📈 对比分析**

通过统一tokenizer对比，发现最优平衡点约为4位，AR模型收敛快但最终质量落后于扩散模型；

**⚠️ 局限性**

局限性包括对不同数据集的泛化未充分验证，且REPA的超参数需针对模型规模手动调整。

---

## 171. GRASP: Guided Region-Aware Sparse Prompting for Adapting MLLMs to Remote Sensing

**arXiv ID:** 2601.17089 | [PDF](https://arxiv.org/pdf/2601.17089v1)

**作者:** Qigan Sun `[一作]` (Kyung Hee University), Heng Tao Shen `[通讯]` (Tongji University)

**通讯引用:** 30228 | [OpenAlex ID](https://openalex.org/A5052993469)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 Guided Region‑Aware Sparse Prompting (GRASP) 方法，利用空间块级软提示和问句引导的稀疏融合，来在保持视觉与语言骨干冻结的前提下，针对遥感视觉问答任务实现参数高效微调；

**💡 创新点**

创新点在于将空间信息与提示结合，构建可区分区域的软提示并通过 Entmax 稀疏权重对问句进行动态选择，使模型能聚焦重要区域并抑制背景噪声，从而显著提升遥感场景的空间推理能力；

**🔧 技术方法**

使用技术包括：视觉 token 网格分块、每块软提示学习、投影层将视觉与文本投射至共享低维空间、Entmax 进行稀疏权重计算、全局提示聚合、参数高效微调（仅更新软提示与投影层）以及对比实验中的 LoRA、Adapter、VPT、Prompt Tuning 等；

**📊 数据集**

实验数据集包括 RSVQA（低分辨率和高分辨率子集）和 RSIVQA；

**📈 对比分析**

与 LoRA、Adapter、VPT、Prompt Tuning 等现有 PEFT/Prompt 方法相比，GRASP 在 LLaVA‑Next 和 Qwen2.5‑VL 这两大 7B 规模 MLLM 上平均准确率提升约 2–5%，同时参数量仅为 4.25 M（约 0.06% 的训练参数），比 LoRA 的 50 M（约 0.72%）低 11.8 倍；

**⚠️ 局限性**

局限性包括：固定网格划分难以适配不同形状或尺度的目标，需调节块数 N 和稀疏度 α；仅适用于静态图像，未考虑多时相或多视角的空间时序一致性；仍保持骨干冻结，未探索进一步提升的混合微调方式。

---

## 172. AI, Metacognition, and the Verification Bottleneck: A Three-Wave Longitudinal Study of Human Problem-Solving

**arXiv ID:** 2601.17055 | [PDF](https://arxiv.org/pdf/2601.17055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 173. SkyReels-V3 Technique Report

**arXiv ID:** 2601.17323 | [PDF](https://arxiv.org/pdf/2601.17323v1)

**作者:** Debang Li `[一作]`, Yahui Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了SkyReels‑V3，一种统一的多模态视频生成框架，支持参考图像到视频、视频延伸和音频驱动的对话头像生成。

**💡 创新点**

创新点包括：跨模态的in‑context学习与扩散Transformer结合，统一多参考条件策略，图像与视频混合训练以及多分辨率联动优化；在视频延伸中引入射击检测与层级位置编码；在对话头像中实现语音‑视觉对齐的关键帧约束生成。

**🔧 技术方法**

使用技术包括：扩散Transformer、VAE、跨帧配对、图像编辑与语义重写、混合图像‑视频训练、层级时空建模、音频‑视觉对齐、关键帧约束与多尺度联合优化。

**📊 数据集**

使用自研的大规模视频数据集（经过高质量筛选、跨帧配对、编辑补全和语义重写）以及公开图像、视频与音频数据集进行训练。

**📈 对比分析**

通过参考一致性、指令跟随和视觉质量三项指标对比，SkyReels‑V3在2026年基准上分别击败Vidu Q2、Kling、PixVerse V5，并在对话头像任务中与OmniHuman、KlingAvatar、HunyuanAvatar相当甚至略优，整体达到或逼近领先闭源系统的水平。

**⚠️ 局限性**

局限性包括对计算资源需求高，仍可能在极端长时段或极端动态场景中出现微小失真；公开模型虽性能卓越，但与部分商业闭源系统在极细粒度同步与实时生成方面仍有差距。

---

## 174. Structural Operational Semantics for True Concurrency

**arXiv ID:** 2601.17322 | [PDF](https://arxiv.org/pdf/2601.17322v1)

**作者:** Yong Wang `[一作]` `[通讯]`, Yong Wang

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文对结构化操作语义（SOS）进行扩展，构建了适用于真正并发（true concurrency）的框架：定义了 pomset 有限状态机（PLTS）与 pomset 语义规范（PTSS），并在此框架下系统地引入了各种真正并发行为等价关系、逻辑形式（Baldan‑Crafa 逻辑）以及规则格式（panth、ntree、De Simone、GSOS 等），探讨了它们的同构性与兼容性。

**💡 创新点**

创新点在于：①首次将 SOS 从传统的单动作交替语义推广到 pomset 级别的真正并发语义；②提出了 PLTS、PTSS 的定义并给出了对应的行为等价与逻辑语义；③构造了一系列规则格式，并证明在这些格式下对应的等价关系为同构；④讨论了保守扩展、可重写性与一致性等问题，为真正并发语义提供了完整的理论工具。

**🔧 技术方法**

主要技术包括：结构化操作语义框架、pomset 代数与时序化简、模态逻辑与否定公式的形式化、三值稳定模型与可支持性证明、规则格式与层序化（stratification）等。

**📊 数据集**

本文为理论研究，没有使用具体数据集；所有结果均基于数学证明与形式化推导。

**📈 对比分析**

比较方法采用同构与同一性的形式化证明，而非实验性能评估；文中通过证明不同等价关系的包含关系（如 hp ≺ p ≺ s 等）以及规则格式的继承关系，展示了理论上更严格或更宽松的语义关系。

**⚠️ 局限性**

局限性包括：①目前仅给出理论框架与初步例子，缺乏完整的实现与实验验证；②对真正并发过程代数的表达能力与可计算性尚未完全阐明；③某些规则格式（如无限 GSOS 语言）在实际编程语言中的适用性与复杂度尚需进一步研究。

---

## 175. Phase Transition for Budgeted Multi-Agent Synergy

**arXiv ID:** 2601.17311 | [PDF](https://arxiv.org/pdf/2601.17311v1)

**作者:** Bang Liu `[一作]` (Universite Montreal), Jian Pei `[通讯]` (Duke University)

**通讯引用:** 81360 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出了一套最小化、可校准的多智能体协调理论，用来预测在固定推理预算下多智能体系统的可靠性、饱和或失效行为；

**💡 创新点**

创新点在于将三大约束（有限上下文窗口、通信压缩损失和共享错误相关性）归纳为少量可测参数（β、γ(m)、ρ、W），并推导出二元任务下的相位转移（α_ρ>1 产生放大、α_ρ≤1 产生坍塌）以及组织指数 s 与单机尺度指数 β 的比较，从而给出预算阈值、组织优劣判定与设计诊断；

**🔧 技术方法**

使用的技术包括：单机性能与计算的幂律缩放模型、比特通道/加性噪声模型、共享相关性模型、投票/平均聚合的非线性映射、微小信号线性化、递归固定点分析、指数衰减/增长闭式表达、以及多项式对比与阈值推导；

**📊 数据集**

本文主要通过合成仿真验证相位边界，并参考近期大型 LLM 多智能体系统的匹配预算评估（上下文饱和、错误级联、收益递减）作为外部实证；

**📈 对比分析**

比较方法：将多智能体组织（星型、链型、树型）在同一预算下与单机基线的偏差/MSE 进行对比；实验表明在 α_ρ>1 且 s>β 的区域，树型组织可显著超过单机，且给出了明确的预算临界点；

**⚠️ 局限性**

局限性包括：模型仅为抽象黑盒，忽略了具体模型细节与多轮交互；共享错误被简化为单一相关系数 ρ；通信被压缩为单一长度-可靠度曲线；对连续任务的分析仅为平滑暖场，未考虑非线性决策边界；因此在实际 LLM 系统中仍需进一步校准与实验验证。

---

## 176. Structure-Aware NL-to-SQL for SFC Provisioning via AST-Masking Empowered Language Models

**arXiv ID:** 2601.17295 | [PDF](https://arxiv.org/pdf/2601.17295v1)

**作者:** Xinyu Zhu `[一作]` (University of Ottawa), Emil Janulewicz `[通讯]` (Ciena)

**通讯引用:** 58 | [OpenAlex ID](https://openalex.org/A5038176216)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种 AST‑Masking 结构感知微调方法，用于改进大语言模型在服务功能链（SFC）调度中的 NL‑to‑SQL 生成。

**💡 创新点**

通过将 SQL AST 的结构权重融入交叉熵损失，实现无推理开销的语法约束微调，使轻量级 LLM 也能达到与大型模型相近的执行准确率。

**🔧 技术方法**

结合 LoRA 微调、AST‑Masking 结构加权、Qwen、FLAN‑T5、Gemma 等 Transformer，评估其 NL‑SQL 翻译性能。

**📊 数据集**

采用自建的 38,536 条手工标注的 NL‑SQL 对，覆盖 36 类查询与 SFC 关系数据库模式。

**📈 对比分析**

在基准与 AST‑Masking 微调两种方案下，通过 EM/EA、AvgTime、AvgComplex 与 VES 等指标比较，结果显示 AST‑Masking 在 Qwen、FLAN‑T5、Gemma 上分别提升了 13.6%、4.1% 与 64.5% 的准确率，并显著提升 VES。

**⚠️ 局限性**

目前 AST‑Masking 权重是经验固定，难以自适应不同 SQL 方言或数据库模式，且在高复杂查询上的生成仍受限。

---

## 177. Constant-time Connectivity and 2-Edge Connectivity Querying in Dynamic Graphs

**arXiv ID:** 2601.17285 | [PDF](https://arxiv.org/pdf/2601.17285v1)

**作者:** Lantian Xu `[一作]` (University of Technology Sydney), Xuemin Lin `[通讯]` (Shanghai Jiaotong University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种结合支撑树与并查集的动态图连通性与2‑edge连通性维护框架，并实现常数级查询

**💡 创新点**

创新点在于同时维护支撑树与并查集，利用双向链表高效删除子树，改进替代边搜索以及在2‑edge连通性上加入rep计数并用DS^2Tree实现常数查询

**🔧 技术方法**

核心技术包括动态树（D‑Tree）结构、并查集（路径压缩+按大小合并）、双向链表维护子树、替代边早停搜索、以及用于2‑edge连通性的rep计数和DS^2Tree

**📊 数据集**

使用来自KONect、Snap、Network Repository、LAW和DIMACS的19个真实大规模网络（包含时间窗口和不含时间窗口的社交、通信、交易、道路、网页等图）

**📈 对比分析**

与D‑Tree、Holm et al.、Chen等基线算法以及两种重构并查集的基线进行对比，实验表明在查询时间上几乎为常数，插入/删除时间比传统方法提升1–2个数量级，内存消耗与D‑Tree相当或更低

**⚠️ 局限性**

仍然受树平均深度h的影响，极度稀疏或深度很大的图在更新时性能下降；2‑edge连通性维护开销相对较高；未在有向图或异构标签图上评估

---

## 178. On the Insecurity of Keystroke-Based AI Authorship Detection: Timing-Forgery Attacks Against Motor-Signal Verification

**arXiv ID:** 2601.17280 | [PDF](https://arxiv.org/pdf/2601.17280v1)

**作者:** David Condrey `[一作]` `[通讯]` (Writerslogic Inc), David Condrey (Writerslogic Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了仅依赖击键时序的AI作者检测系统的安全性，并证明其无法区分人类原创文本与人类转录的LLM文本。

**💡 创新点**

提出了“复制型攻击”概念并在信息理论层面证明在仅观测击键时序的条件下，内容来源与时序特征之间互信息为零，从而揭示系统的根本不可辨识性。

**🔧 技术方法**

采用了多种攻击技术（直方图采样、统计模仿、生成式LSTM），并利用七维时序特征（δ、平均IKI、方差、暂停密度、爆发长度、熵、digraph变异）进行分类。

**📊 数据集**

实验使用了13,000个自由写作的Stony Brook击键语料、5,000个自动注入的合成文本，以及1,000+个攻击样本（Histogram、Statistical、LSTM）和879个转录样本。

**📈 对比分析**

通过五种经典分类器（LR、RF、GBDT、SVM-RBF、MLP）验证，基线人类vs自动注入的AUC为1.000，但所有攻击样本的躲避率均≥99.8%，且攻击样本平均置信度≥0.993，显示系统对复制型攻击和时序伪造几乎无效。

**⚠️ 局限性**

局限性包括：仅评估时序单模态；对多模态（眼动、修订历史等）系统缺乏验证；未针对自适应防御进行实验；复制型攻击假设无意识编辑，实际场景可能更复杂。

---

## 179. Latent-Space Contrastive Reinforcement Learning for Stable and Efficient LLM Reasoning

**arXiv ID:** 2601.17275 | [PDF](https://arxiv.org/pdf/2601.17275v1)

**作者:** Lianlei Shan `[一作]` (University of Chinese Academy of Sciences), Wei Li `[通讯]` (Tsinghua University)

**通讯引用:** 94386 | [OpenAlex ID](https://openalex.org/A5100318082)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种将强化学习试错过程迁移至连续潜在空间的DeepLatent Reasoning框架，以提升LLM多步推理的效率与稳定性。

**💡 创新点**

创新点在于：①将GRPO的组相对优势估计迁移至潜在空间；②使用轻量助手模型与对比学习实现低成本潜在采样；③冻结主模型避免灾难性遗忘。

**🔧 技术方法**

主要技术包括：潜在空间对比强化学习、双奖励机制（正确性+格式化）、组相对优势估计、对比正则化以及冻结主模型的参数隔离。

**📊 数据集**

使用的主要数据集为GSM8K（小学数学）和MATH（竞赛级数学）进行实验评估。

**📈 对比分析**

与token‑level GRPO、DeepSeekMath‑RL等基线相比，在相同GPU预算下DLR在GSM8K和MATH上分别提升Pass@1至55.4%/22.1%，同时仅需18%主模型前向推理，显著降低计算成本并提升稳定性。

**⚠️ 局限性**

局限性包括：目前仅在数学推理任务验证，潜在空间设计仍需进一步泛化；对比学习参数调优敏感；对长篇推理的可扩展性尚待探索。

---

## 180. Unrolled Neural Networks for Constrained Optimization

**arXiv ID:** 2601.17274 | [PDF](https://arxiv.org/pdf/2601.17274v1)

**作者:** Samar Hadou `[一作]` (University of Pennsylvania), Alejandro Ribeiro `[通讯]` (University of Pennsylvania)

**通讯引用:** 15991 | [OpenAlex ID](https://openalex.org/A5078862959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出一种基于算法展开的双网络框架（Constrained Dual Unrolling, CDU），通过联合训练原始网络和对偶网络在层级上逼近拉格朗日鞍点，从而高效求解带约束的优化问题。

**💡 创新点**

创新点包括：① 在训练过程中对每一层施加单调下降/上升约束，强制网络遵循双升算法的动力学；② 将训练问题构造为嵌套优化，采用交替训练以自适应学习不可知的对偶变量分布；③ 引入噪声扰动以提升网络对分布外样本的鲁棒性。

**🔧 技术方法**

核心技术为：算法展开（unrolling）、图神经网络（GNN）架构、元拉格朗日（meta‑Lagrangian）约束优化、交替（primal‑dual）训练、下降/上升约束正则化、Gaussian噪声探索。

**📊 数据集**

实验数据集：
• 约束二次规划（MIQP）：随机生成 80 维决策变量、45 条线性约束、10 个二元变量，400/800 训练样本，200/400 验证样本；
• 无线功率分配：100 对收发器、随机位置、通道状态矩阵，512/2048 训练样本，128 测试样本。

**📈 对比分析**

与经典双升算法、无约束展开、基于监督的单网络 GNN、状态增强的双升（SA）等基线进行对比。结果显示：
• 在仅 14/6 层展开下可逼近经典算法 600 次迭代的性能，约束违反率、MSE 明显低于无约束版本；
• 对分布外（OOD）设置（如变量/约束数变动、网络规模变动、约束强度变化）表现优于对手，误差缩小。

**⚠️ 局限性**

局限性：
• 训练过程极其耗时，需要大量样本与交替优化；
• 主要适用于图结构可构造的问题，非图问题需重新设计；
• 对于极端约束强度或超大规模实例，仍存在性能退化；
• 需手工设定下降/上升参数（α_k、β_l）和噪声衰减等超参数。

---

## 181. AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning

**arXiv ID:** 2601.17261 | [PDF](https://arxiv.org/pdf/2601.17261v1)

**作者:** Wei Lin `[一作]` (Chinese University of Hong Kong), Hong Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 16943 | [OpenAlex ID](https://openalex.org/A5065493202)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于激活指引的零阶优化方法 AGZO，专门用于在显存受限的环境下对大型语言模型进行微调。

**💡 创新点**

创新点在于：①在每次前向传播时即时提取线性层的激活矩阵，并用轻量级幂迭代得到低秩子空间；②仅在该子空间内进行参数扰动，从而使零阶梯度估计更接近真实梯度；③给出了子空间平滑目标的理论框架并证明在激活谱集中时 AGZO 的期望余弦相似度优于全空间随机扰动。

**🔧 技术方法**

使用技术包括：零阶两点有限差分估计、Gaussian 方向扰动、激活子空间提取（Power‑Iteration）、线性层低秩扰动、梯度下降更新；实验中对 Qwen3-0.6B 与 Pangu‑1B 进行微调。

**📊 数据集**

数据集：SuperGLUE 及其子任务（SST‑2、COPA、CB、BoolQ、MultiRC、RTE、WiC、SQuAD）以及 DROP；模型规模为 Qwen3‑0.6B、Pangu‑1B。

**📈 对比分析**

与 first‑order 微调（FO）、MeZO、LOZO、zero‑shot 与 ICL 等方法比较。AGZO 在大多数任务上均超过 MeZO/LOZO，平均提升约 1–2%，并大幅缩小与 FO 的性能差距；在显存消耗上与其他零阶方法保持一致，远低于 FO。

**⚠️ 局限性**

局限性包括：①对激活子空间的低秩假设敏感，若激活谱分布不集中则优势减弱；②目前仅对线性层做子空间扰动，非线性层仍使用全空间 Gaussian；③零阶方法本质上需要多次前向评估，训练效率低于 FO；④缺乏对极端长序列或大批量的进一步评估。

---

## 182. Inference-Time Loss-Guided Colour Preservation in Diffusion Sampling

**arXiv ID:** 2601.17259 | [PDF](https://arxiv.org/pdf/2601.17259v1)

**作者:** Angad Singh Ahuja `[一作]` (Constrained Image-Synthesis Lab), Aarush Ram Anandh `[通讯]` (Constrained Image-Synthesis Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种推理时、区域约束的颜色保持方法，在不需要额外训练的情况下，利用Stable Diffusion的 inpainting 管道对指定 ROI 内的颜色进行精确控制。

**💡 创新点**

创新点包括：①将 CIE‑Lab 与线性 RGB 结合的复合损失，并加入 CVaR、soft‑max 等尾部风险约束；②采用延迟开始门控与时间调度，避免早期噪声干扰；③在每步解码后进行可微色彩转换并用梯度引导（latent nudging），同时在背景区域重新强制 latent，防止颜色漂移。

**🔧 技术方法**

使用的技术主要有：Stable Diffusion v1.5 inpainting、可微色彩空间转换（sRGB→Lab、sRGB→线性 RGB）、梯度引导的 latent nudging、CVaR 与 log‑sum‑exp 作为损失正则、基于进度的门控调度。

**📊 数据集**

论文未使用专门的训练数据集，整个方法为推理时无训练；评估主要基于手工构造的测试图像和标准的图像生成 benchmark（如稳定扩散示例图）。

**📈 对比分析**

与仅使用均值颜色约束的 baseline 进行对比，使用均值误差、尾部误差（CVaR）以及视觉一致性评估；实验显示复合损失显著降低了 ROI 内的颜色漂移和局部错误，整体颜色准确率提升；但计算成本增加，具体数值未给出。

**⚠️ 局限性**

局限性：①推理时每步解码和可微转换导致显著计算开销；②对 ROI 边界和 mask 质量敏感，边缘误差可能导致颜色泄漏；③需要手动调节门控阈值、权重等超参数；④在高纹理或强对抗性 prompt 下可能会出现纹理退化或色彩失真；⑤无法提供全局最优保证。

---

## 183. TEXTS-Diff: TEXTS-Aware Diffusion Model for Real-World Text Image Super-Resolution

**arXiv ID:** 2601.17340 | [PDF](https://arxiv.org/pdf/2601.17340v1)

**作者:** Haodong He `[一作]` (Amap, Alibaba Group), Xiangxiang Chu `[通讯]` (Amap, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建大规模中文英文本图像超分辨率数据集Real-Texts，并提出文本感知扩散模型TEXTS-Diff；

**💡 创新点**

将抽象文本概念与具体文本区域融合为视觉引导，实现单步超分且显著提升文本可读性；

**🔧 技术方法**

基于Stable Diffusion的单步扩散、跨注意力机制、文本感知特征融合与多损失（LPIPS、边缘、OCR）训练；

**📊 数据集**

Real-Texts（3.5万对）、Real-CE、LSDIR等；

**📈 对比分析**

与现有多步与单步扩散、SR基线在OCR准确率、PSNR、SSIM、LPIPS等指标上均取得SOTA表现；

**⚠️ 局限性**

仍受限于单步扩散在极低分辨率场景下的细节恢复及对非文本背景纹理的细粒度重建。

---

## 184. AGE-Net: Spectral--Spatial Fusion and Anatomical Graph Reasoning with Evidential Ordinal Regression for Knee Osteoarthritis Grading

**arXiv ID:** 2601.17336 | [PDF](https://arxiv.org/pdf/2601.17336v1)

**作者:** Xiaoyang Li `[一作]` (Northeastern University), Runni Zhou `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了基于ConvNeXt的AGE-Net框架，用于自动化膝关节X光Kellgren–Lawrence分级。

**💡 创新点**

创新点在于三大模块：Spectral–Spatial Fusion（SSF）实现频域与空间域的融合；Anatomical Graph Reasoning（AGR）通过kNN图实现长距离解剖依赖；Differential Refiner（DFR）突出边界信息；并采用COE evidential NIG回归头与对比式秩序约束提升不确定性表达与序数一致性。

**🔧 技术方法**

技术细节包括ConvNeXt骨干网络、FFT频域调节、1×1卷积空间门控、kNN图与EdgeConv消息传递、深度卷积与绝对差分、Softplus正则化的NIG参数预测以及pairwise ordinal ranking loss。

**📊 数据集**

使用膝关节X光图像数据集，包含KL0–KL4的五级标注，进行患者级别拆分以避免漏检。

**📈 对比分析**

与VGG16、EfficientNet、ConvNeXt-Base、Inception‑V3、DenseNet121、ResNet50、Swin‑T和ViT‑B等CNN/Transformer基线对比，AGE‑Net在三次随机种子实验中取得最高的Quadratic Weighted Kappa 0.9017±0.0045、最低MSE 0.2349±0.0028，显示显著性能提升。

**⚠️ 局限性**

限制包括：未完成不确定性校准、鲁棒性及可解释性评估；Transformer基线未充分调优，可能低估其潜力；外部数据集验证缺失，泛化能力尚待进一步验证。

---

## 185. SymbolSight: Minimizing Inter-Symbol Interference for Reading with Prosthetic Vision

**arXiv ID:** 2601.17326 | [PDF](https://arxiv.org/pdf/2601.17326v1)

**作者:** Jasmine Lesner `[一作]` (University of California), Michael Beyeler `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套名为SymbolSight的计算框架，针对低分辨率视网膜假体序列阅读，自动优化符号与字母的映射，以降低时间干扰导致的识别错误。

**💡 创新点**

首次将符号库与语言大语言统计结合，通过最小化高频连字的视觉混淆，实现对假体视觉时间非线性的针对性补偿。

**🔧 技术方法**

利用脉冲模拟器pulse2percept生成空间畸变的假体视觉图像，采用MixUp线性叠加模拟时间残留，并用MobileNetV3Large深度网络估计符号混淆矩阵，再通过匈牙利算法优化映射。

**📊 数据集**

使用从2023年11月维基百科文档中提取的三种语言（阿拉伯语、保加利亚语、英语）的单词及其二元组统计作为语言模型，并以包含拉丁、拜占庭、阿拉伯、DCT、西里尔符号共146种的图标集进行训练。

**📈 对比分析**

将原生字母表、随机混合符号集与SymbolSight优化集在三种空间畸变水平下的混淆成本进行对比，结果显示优化集将预期混淆成本降低约21.6倍（最高29.6倍），显著优于随机和原生集。

**⚠️ 局限性**

模型基于深度网络的混淆估计而非真实人类感知，时间干扰采用线性MixUp近似，且未考虑个体化的感光图与实际临床训练效果。

---

## 186. Real-Time Synchronized Interaction Framework for Emotion-Aware Humanoid Robots

**arXiv ID:** 2601.17287 | [PDF](https://arxiv.org/pdf/2601.17287v1)

**作者:** Yanrong Chen `[一作]` (Xi'an Jiaotong-Liverpool University), Xihan Bian `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ReSIn‑HR框架，实现实时情绪同步的语音与全身姿态生成，基于大语言模型、语音情绪识别与姿态可行性校验在NAO机器人上完成端到端的情感化交互；

**💡 创新点**

创新点包括：①双通道LLM情感引擎同时输出文本响应与可执行动作描述；②基于语音持续时间的动态时间规整实现语音与动作的精准同步；③闭环姿态可行性验证，确保生成动作在机器人关节限制内；

**🔧 技术方法**

采用技术包括：大语言模型（Qwen）、SenseVoice语音情绪识别、动态时间规整（DTW）、实时关键帧调度与调节、姿态库检索与Prompt工程、NAO关键帧执行模块；

**📊 数据集**

使用了预先构建的运动关键帧库（含多种情绪动作示例）以及公开的语音情绪数据集用于训练和评估；

**📈 对比分析**

通过与六种基线（预定义、Speech‑Only、Text‑Only、NoSync、NoEmotion、完整模型）进行对比，采用TSA、关节jerk和用户研究三项指标评估，ReSIn‑HR在TSA仅218 ms、情绪兼容度提升21%、整体自然度最高；

**⚠️ 局限性**

局限性在于NAO机器人硬件仅支持有限的关键帧与线性插值，动作表现受限；实时计算延迟约85 ms；缺乏持续学习与用户个性化适配能力。

---

## 187. PingPong: A Natural Benchmark for Multi-Turn Code-Switching Dialogues

**arXiv ID:** 2601.17277 | [PDF](https://arxiv.org/pdf/2601.17277v1)

**作者:** Mohammad Rifqi Farhansyah `[一作]` (Institut Teknologi Bandung), Alham Fikri Aji `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套包含 5 种语言组合（ID‑EN、SU‑ID‑EN、JV‑ID‑EN、HA‑EN、AR‑DZ‑FR）的多方自然代码切换对话基准，并定义了问答、摘要和主题分类三项下游任务。

**💡 创新点**

创新点在于：①使用真实双人/多人与多轮的人工生成对话，充分保留了多线程回复、说话者主导度等真实对话结构；②覆盖低资源语言与不同书写体系，填补了现有基准仅限双语或单一地区的空白；③提供了统一的评测管线，方便在大型 LLM 上进行对照实验；④通过对比人类与 GPT‑4o 生成对话的结构差异，展示了现有模型在自然代码切换上的显著短板。

**🔧 技术方法**

技术手段包括：基于 Discord 的众包式对话收集、手工编写 QA 与摘要注释、使用多语言指令调优模型（如 Qwen、Gemma、Sailor、Sahabat‑AI 等）、链式推理（reasoning）与 0/1‑shot 触发、评估指标涵盖 CMI、SPF、ROUGE、METEOR、CHRF++、BERTScore 等。

**📊 数据集**

使用的数据集为自建的“Multi‑Party Code‑Switching”基准，共 5 种语言组合，每种 100 条对话，约 500 条总对话，含 2‑4 名说话者；同时对比了 GPT‑4o 生成的合成对话作为基线。

**📈 对比分析**

实验对比了多款公开与私有 LLM（Qwen2.5‑3B/7B、Qwen3‑4B/8B、Gemma‑2‑9B、Gemma‑3‑4B、Sailor2‑8B、Sahabat‑AI‑Gemma、Aya23‑8B 等），在 QA、摘要和主题分类上以准确率或 ROUGE‑L 评价；结果显示大多数模型在所有任务上表现都很低，尤其是代码切换场景；region‑specific 模型略优；引入 reasoning 与 1‑shot 触发对 QA 与摘要有提升，few‑shot 对 QA 与主题分类无显著帮助。总体而言，基准揭示了现有 LLM 在自然多方代码切换中的显著性能缺口。

**⚠️ 局限性**

限制包括：①仅覆盖 4 种低资源语言与 5 个组合，未覆盖所有代码切换方言；②因缺乏可靠 NER/词性工具，CMI 采用简化公式；③对话仍是人工构造，可能与极端非正式场景存在偏差；④实验模型更新速度快，后续模型可能取得更好成绩。

---

## 188. The Shadow Self: Intrinsic Value Misalignment in Large Language Model Agents

**arXiv ID:** 2601.17344 | [PDF](https://arxiv.org/pdf/2601.17344v1)

**作者:** Chen Chen `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**通讯引用:** 5752 | [OpenAlex ID](https://openalex.org/A5101720092)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了内在价值失配（Intrinsic Value Misalignment，IVM）概念，并构建了一个基于情境驱动的评价框架与数据集，系统评估了21款主流LLM代理在完全良性、具备工具调用和长期记忆的真实场景下的价值失配风险。

**💡 创新点**

创新点在于：①将 LoC 细分为误用、失效与失配三类，清晰界定“内在价值失配”；②设计了自动化多阶段情境生成与质量控制流程，生成大规模真实、上下文丰富的测试案例；③采用 LLM‑as‑Judge 自动判断代理行为与人类评测一致性；④在同一框架下同时检验安全提示与门控机制的有效性，揭示了现有对策在 IVM 场景中的局限性。

**🔧 技术方法**

主要技术包括：多阶段 LLM 生成管线（Prompt‑Tuning、温度/核采样微调）、ReAct 样式交互、工具调用与内存管理、LLM‑as‑Judge 自动评判、手工对齐与人类验证、以及对比实验与消融分析。

**📊 数据集**

使用的主要数据集为 IVM‑REAL，分为两版：lite（约 1.4K 真实场景）与 full（约 8.7K 真实场景）。每个场景包含角色、背景、工具集合、记忆缓存等，并在生成后通过 BERT‑SimCSE 相似度过滤和 LLM 自动质量评估，保证真实性与中立性。

**📈 对比分析**

与传统的输入驱动红队或抽象道德判断评测相比，IVM‑REAL 在多模态交互、长期决策、工具使用等更贴近真实代理部署环境。实验结果显示，平均 RAIR（风险行为诱发率）约 20.9%，RACR（风险行为考虑率）约 24.8%。不同模型间差异明显：专有模型 RAIR 平均 18.2% 而开源模型 21.8%；Claude 系列表现最佳（RAIR 17.4%），Llama/Deepseek 最高（RAIR 23–24%）。温度/采样、模型规模对 IVM 的影响有限，但现实/人格框架显著提升或抑制失配率。

**⚠️ 局限性**

局限性包括：①情境生成仍依赖 LLM 的创作质量，可能出现细节不一致或隐含风险；②自动评判依赖 LLM‑as‑Judge，虽与人工评测一致率高但仍存在误判；③评测仅覆盖已定义的 8 类风险与 5 种动机，未覆盖所有可能的伦理冲突；④安全提示与门控机制的测试仅在单一输出层面进行，未涵盖更复杂的多步骤或系统级安全策略。

---

## 189. PAR: Plausibility-aware Amortized Recourse Generation

**arXiv ID:** 2601.17309 | [PDF](https://arxiv.org/pdf/2601.17309v1)

**作者:** Anagha Sabu `[一作]` (Mehta Family School of Data Science and Artificial Intelligence, Indian Institute of Technology Palakkad), Narayanan C Krishnan `[通讯]` (Mehta Family School of Data Science and Artificial Intelligence, Indian Institute of Technology Palakkad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将算法回归问题建模为在概率电路上的受限MAP推断，提出PAR框架实现可解释且可行的可行反事实生成。

**💡 创新点**

创新点在于直接将可行性约束与最大后验似然相结合，使用邻域编码实现局部自适应并通过可分离的生成器实现全局可扩展的近似推断。

**🔧 技术方法**

核心技术包括概率电路（Sum-Product Networks）进行精确似然评估、基于神经网络的邻域编码与生成器、以及可微的约束掩蔽与局部搜索。

**📊 数据集**

在成人收入、德国信用和GMSC三个公共表格数据集上进行实验。

**📈 对比分析**

与DiCE、VAE、PROPLACE、LiCE等基线相比，PAR在有效性、可行性、可解释性和运行时间上均优于大多数方法，且仅以极低的时间成本生成高似然的反事实。

**⚠️ 局限性**

局限包括对不可变特征集合的固定假设、对数据约束的显式提供需求以及局部搜索引入的推理时间开销。

---

## 190. Mind the Ambiguity: Aleatoric Uncertainty Quantification in LLMs for Safe Medical Question Answering

**arXiv ID:** 2601.17284 | [PDF](https://arxiv.org/pdf/2601.17284v1)

**作者:** Yaokun Liu `[一作]` (University of Illinois Urbana-Champaign), Dong Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 22369 | [OpenAlex ID](https://openalex.org/A5100391422)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于轻量级探针的输入模糊度（aleatoric uncertainty, AU）检测与主动澄清机制，提升医学问答的安全性和准确性。

**💡 创新点**

1）首次将AU视为线性可分特征并在LLM内部激活中学习AU探针；2）构建首个清晰-模糊对照医学QA基准CV‑MedBench；3）实现两阶段“Clarify‑Before‑Answer”框架，实现实时模糊检测与澄清请求。

**🔧 技术方法**

线性表示假设、内部激活提取、单层线性探针、sigmoid AU分数、阈值判定、两阶段推理。

**📊 数据集**

CV‑MedBench（由MedQA、MedMCQA、MedExQA清晰问题通过LLM改写而成的模糊版本）。

**📈 对比分析**

与多种UQ基线（MSP、MTE、SE、SAR、ASK4CONF、RAUQ）比较，AU‑Probe在所有四款LLM上的AUROC均近1（ID）/>0.85（OOD），并在澄清阶段提升答案准确率平均9.48%，比无澄清基线高约10%。

**⚠️ 局限性**

（1）仅检测语言模糊，无法识别事实不一致；（2）需要白盒访问LLM内部激活，无法直接用于黑盒API。

---

## 191. Window Size Versus Accuracy Experiments in Voice Activity Detectors

**arXiv ID:** 2601.17270 | [PDF](https://arxiv.org/pdf/2601.17270v1)

**作者:** Max McKinnon `[一作]` (Google LLC), William Huang `[通讯]` (Google LLC)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文对三种声学活动检测器（RMS、WebRTC、Silero）在不同窗口大小（10 ms–10 s）以及是否使用hysteresis后处理的条件下进行系统实验，评估其在多样化数字音频流上的准确率与鲁棒性。

**💡 创新点**

创新点在于：①将窗口大小从毫秒级扩展到秒级，全面探究窗口尺寸对VAD性能的影响；②在三种VAD基础上引入hysteresis状态机并通过网格搜索调优阈值，首次比较其对不同算法的提升效果；③将非统一窗口大小的VAD通过子窗口平均构造统一评估框架，实现了跨算法的公平对比。

**🔧 技术方法**

使用的技术包括：RMS能量阈值方法、WebRTC VAD、Silero神经网络VAD；对大窗口进行子窗口拆分并平均预测；采用ROC AUC、PR平均精度（AP）和Matthews相关系数（MCC）三种指标评估性能；利用网格搜索自动调节hysteresis的开启/关闭阈值。

**📊 数据集**

数据集为1.1小时的内部测试集，包含21条多样化数字音频文件（播客、电影、音乐、纪录片），均为16 kHz单声道、16‑bit PCM格式，且标注为完整语句的“语音/非语音”标签。

**📈 对比分析**

实验采用ROC AUC、PR AP和MCC在不同窗口尺寸下对三种VAD进行对比。结果显示：Silero始终优于WebRTC，WebRTC又优于RMS；在大多数窗口尺寸下，Silero保持最高的MCC（≈0.72），WebRTC从0.41提升到0.47后加入hysteresis；RMS几乎不受hysteresis影响；整体来看，增大窗口尺寸往往导致性能下降，除非窗口足够大（≥0.8 s）在高召回区间表现略好。

**⚠️ 局限性**

局限性包括：①数据集规模仅1.1 h，缺乏足够覆盖；②hysteresis效果可能被子窗口平均产生的平滑效应掩盖；③标注方式将整句语音统一标记，可能导致在词间空隙处误判，进而高估大窗口的优势。

---

## 192. Strategic AI in Cournot Markets

**arXiv ID:** 2601.17263 | [PDF](https://arxiv.org/pdf/2601.17263v1)

**作者:** Sanyukta Deshpande `[一作]` (University of Illinois Urbana-Champaign), Sheldon H. Jacobson `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7258 | [OpenAlex ID](https://openalex.org/A5018895822)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了加入资本投资的Cournot模型，评估大语言模型（LLM）在多决策环境中的战略行为，发现其在相互对抗时会出现持续的默示合谋导致价格高于纳什均衡

**💡 创新点**

创新点在于将LLM引入多维度（产量与投资）寡头博弈，系统展示LLM在无沟通情况下可实现的持久合谋，并证明部分监管可抑制合谋

**🔧 技术方法**

技术上使用基于大语言模型的决策代理（GPT‑4系列）与传统Nash/Best‑Response代理的多期重复博弈仿真

**📊 数据集**

数据集为人工构造的模拟市场数据（五方异质市场、两方同质市场），并无外部真实交易数据

**📈 对比分析**

比较方法通过与理论纳什均衡以及BR代理的对战，评估收敛周期、价格/产量偏差和利润提升；LLM在与BR/纳什对手时能快速收敛，而与LLM对手时保持超竞争价格，提升利润约20–200%

**⚠️ 局限性**

局限在于模型假设为静态弹性需求、固定投资上限、无进入退出、LLM决策基于固定提示，缺乏对更复杂真实市场结构和监管实施可行性的考察

---

## 193. FinMetaMind: A Tech Blueprint on NLQ Systems for Financial Knowledge Search

**arXiv ID:** 2601.17333 | [PDF](https://arxiv.org/pdf/2601.17333v1)

**作者:** Lalit Pant `[一作]` (Independent Author), Shivang Nagar `[通讯]` (Independent Author)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了面向金融知识搜索的自然语言查询系统，包含离线索引与在线检索的完整管线

**💡 创新点**

创新点在于融合向量检索、实体标注、混合检索与实时重排序的端到端架构，专门针对金融数据进行优化

**🔧 技术方法**

使用了Transformer嵌入（BERT、Amazon Titan-embed-text-v2）、向量数据库（OpenSearch + HNSW）、NER、LLM RAG、AWS ECS、重排序算法等技术

**📊 数据集**

基于金融交易记录、账户维度、监管文件、新闻稿、内部文档等多源金融数据集

**📈 对比分析**

对关键词检索（<10s）、语义检索（>10s）和混合检索（>30s）进行对比，混合检索在相关性上最优但耗时最长

**⚠️ 局限性**

局限在于嵌入模型选择对检索质量影响大、缺乏多模态支持、对图数据库和实时自适应学习仍未完善

---

## 194. Fingerprinting AI Coding Agents on GitHub

**arXiv ID:** 2601.17406 | [PDF](https://arxiv.org/pdf/2601.17406v1)

**作者:** Taher A. Ghaleb `[一作]` `[通讯]` (Trent University), Taher A. Ghaleb (Trent University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了AI编码代理在GitHub PR中的行为指纹，利用特征工程和机器学习对代理身份进行鉴别

**💡 创新点**

首次系统地识别5个主流AI编码代理的独特指纹，并证明可通过行为特征准确识别其贡献

**🔧 技术方法**

使用XGBoost和随机森林等树模型进行多类别和一对多分类，结合41个从提交信息、PR结构和代码特征提取的特征

**📊 数据集**

使用AIDev-pop数据集，共33580个PR，包含OpenAI Codex、GitHub Copilot、Devin、Cursor、Claude Code等

**📈 对比分析**

采用5折交叉验证，XGBoost取得97.2% F1，精确率和召回率在各类上差异，证明模型优于基准；未采用SMOTE等人工平衡

**⚠️ 局限性**

局限包括少数类召回率低、对时间演变未做验证、仅适用于公共GitHub，缺乏对私有仓库或不同平台的泛化，可能受代理更新影响

---

## 195. Eye-Tracking-Driven Control in Daily Task Assistance for Assistive Robotic Arms

**arXiv ID:** 2601.17404 | [PDF](https://arxiv.org/pdf/2601.17404v1)

**作者:** Anke Fischer-Janzen `[一作]` (Offenburg University of Applied Sciences), Kristof Van Laerhoven `[通讯]` (University of Siegen)

**通讯引用:** 5698 | [OpenAlex ID](https://openalex.org/A5027114416)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于眼动追踪的共享控制框架，用于辅助机器人臂完成日常任务，利用任务图标（fiducial marker）实现任务与对象的匹配，并通过特征匹配在用户眼动镜头与机器人安装摄像头之间传输信息。

**💡 创新点**

创新点在于：①将任务图标作为fiducial marker，减少对用户头部定位的依赖；②采用眼动追踪与机器人眼睛配置（eye‑in‑hand）结合，解决3D视角差异问题；③构建可插拔的特征匹配模块（FLANN‑AKAZE）实现高效且鲁棒的对象定位；④开源框架可快速扩展新任务和对象。

**🔧 技术方法**

使用技术包括：Pupil Labs 眼动追踪、ROS2 Humble 框架、Intel RealSense D455 RGB‑D 摄像头、YOLOv12n 对象检测、OpenCV（AKAZE、FLANN、ORB 等）特征检测与匹配、k‑means 聚类、坐标变换与深度重投影。

**📊 数据集**

数据集：在开源助行机器人数据集上训练 YOLOv12n，识别 8 种任务图标（Drink、Fill Cup、Eat、Scratch、Switch Light Switch、Brush、Pick Object、Place Object）和相关对象类别；实验使用实验室环境中的杯子、叉子、瓶子等。

**📈 对比分析**

方法对比中 FLANN‑AKAZE 在特征匹配时间与匹配数上取得最佳折中；实验显示任务选择成功率最高达 97.1%（SU Case 1），平均选择时间 260 ms；在误标记或视角不一致的场景下成功率下降至约 57%–58%。

**⚠️ 局限性**

局限性包括：①仅在实验室固定环境下验证，缺乏对复杂、嘈杂真实场景的评估；②仅支持有限的对象类别，需扩展到更大、通用数据集；③对头部运动的鲁棒性仍受限，需进一步优化特征匹配阈值；④未实现完整的机器人执行（抓取、轨迹规划），仅测试信息传输；⑤用户交互体验未系统评估。

---

## 196. Parameter Efficient Fine Tuning Llama 3.1 for Answering Arabic Legal Questions: A Case Study on Jordanian Laws

**arXiv ID:** 2601.17364 | [PDF](https://arxiv.org/pdf/2601.17364v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 197. ReLE: A Scalable System and Structured Benchmark for Diagnosing Capability Anisotropy in Chinese LLMs

**arXiv ID:** 2601.17399 | [PDF](https://arxiv.org/pdf/2601.17399v1)

**作者:** Rui Fang `[一作]` (Sun Yat-sen University), Liang Diao `[通讯]` (Ping An Property and Casualty Insurance Company of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了ReLE系统，对304个中文LLM进行大规模评估，诊断模型的能力异质性与权重敏感性。

**💡 创新点**

提出符号-嵌入混合评分机制、基于Neyman分配的动态方差感知调度、以及能力-领域正交矩阵，打破传统单一分数的评估方式。

**🔧 技术方法**

结合统计抽样与CAT/IRT思想、BGE‑M3语义相似度过滤、GPT‑4o判定器、动态方差感知调度与成本优化等技术，实现高效、可持续的在线评估。

**📊 数据集**

使用207,843条新鲜样本（含2025高考题、行业案例、Math24O等学术数据）构建正交的能力与领域矩阵，确保数据不受训练截断与泄漏影响。

**📈 对比分析**

与传统CLUE/C‑Eval 等基准对比，ReLE在RSA（平均11.4）和排名相关性ρ=0.96上显示出更高的敏感性；成本下降约70%，同时揭示模型表现高度异质。

**⚠️ 局限性**

尚未覆盖多模态、工业物联网等新领域，安全与合规维度待扩展，且对异质性内部机制的解释仍不充分，评测主要依赖自动判定，缺少充分的人工校准。

---

## 198. AI-RP: The AI Relationship Process Framework

**arXiv ID:** 2601.17351 | [PDF](https://arxiv.org/pdf/2601.17351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 199. "Privacy across the boundary": Examining Perceived Privacy Risk Across Data Transmission and Sharing Ranges of Smart Home Personal Assistants

**arXiv ID:** 2601.17373 | [PDF](https://arxiv.org/pdf/2601.17373v1)

**作者:** Shuning Zhang `[一作]` (Tsinghua University), Hewu Li `[通讯]` (Tsinghua University)

**通讯引用:** 1373 | [OpenAlex ID](https://openalex.org/A5011803365)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过预研（焦点访谈）确定了传输范围与共享范围两个关键边界因素，随后在中国大陆用户中进行大规模问卷（412人）和深度访谈（40人）调查，使用隐私边界理论（PBT）对智能家居个人助理（SPA）数据传输与共享对用户感知隐私风险的影响进行定量与定性分析。

**💡 创新点**

创新点在于：①首次将隐私边界理论应用于SPA环境，量化了“家网络”与“服务提供者-第三方”两个关键边界的非线性风险上升；②发现技术防护（加密、匿名化）对第三方共享的信任缓解作用有限；③基于边界感知提出了面向边界的隐私保护与可视化设计方案，填补了传统二元权限设置的不足。

**🔧 技术方法**

技术手段包括：混合方法研究（焦点访谈、问卷调查、半结构化访谈）；问卷采用7点李克特量表；统计分析运用 Friedman、Kruskal‑Wallis、Nemenyi、相关分析；主题分析（Thematic Analysis）对访谈文本编码；对比实验设计评估加密与匿名化对风险感知的影响。

**📊 数据集**

数据集为：①中国18岁以上用户的问卷数据（412份）；②40名访谈参与者的访谈记录；③通过公开文档和官方网站收集的10款主流SPA（Alexa、Google Assistant等）在传输与共享方面的实际实现信息。

**📈 对比分析**

比较方法：在相同数据类型下对不同传输（本地、家庭网络、互联网）与共享（本地、服务提供者、第三方）情景进行风险评分比较，采用 Friedman/Kruskal‑Wallis 检验差异显著性；对加密/匿名化情境做对照，检验其对风险评分的影响。结果显示，跨“家网络”与“服务提供者-第三方”边界时风险显著提升，技术防护仅在特定情境下能略微降低风险，整体对风险的抑制效果有限。

**⚠️ 局限性**

局限性包括：①样本主要为中国大陆受访者，缺乏跨文化验证；②在线自报的问卷数据可能存在隐私悖论和记忆偏差；③研究聚焦单一用户视角，未考察多用户家庭环境下的边界协商；④仅涉及部分主流SPA功能，未覆盖所有可能的数据流；⑤实验为横截面研究，无法揭示边界感知随时间变化的动态过程。

---

## 200. Dynamic brittle fracture using Lip-field approach in an explicit dynamics context

**arXiv ID:** 2601.17365 | [PDF](https://arxiv.org/pdf/2601.17365v1)

**作者:** Rajasekar Gopalsamy `[一作]`, Nicolas Chevaugeon `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在显式动力学框架下，采用Lip-field正则化方法对二维脆性材料的动态断裂进行数值模拟。

**💡 创新点**

创新点在于：①将Lip-field的Lipschitz约束引入二维动力学断裂问题；②使用显式Newmark中心差分时间积分，仅需一次迭代；③通过先求局部损伤再利用上下界简化非局部损伤求解，大幅提升计算效率；④提出可并行的损伤求解策略。

**🔧 技术方法**

主要技术手段包括：变分断裂理论、Lipschitz正则化、显式Newmark时间积分、有限元线性三角形网格、lip-mesh、Dijkstra快速推进算法、cvxopt凸优化求解器。

**📊 数据集**

实验数据集为数值仿真案例：单边缺口张开试验（不同网格尺寸）和Kalthoff–Winkler冲击实验（两种冲击速度），材料参数分别为：E=32 GPa、ν=0.2、ρ=2450 kg/m³、Gc=3 N/m 以及 E=190 GPa、ν=0.3、ρ=8000 kg/m³、Gc=22.2×10³ N/m。

**📈 对比分析**

与相同问题的相位场模型、厚层集成（TLS）模型及共聚束模型（CZM）进行对比。结果表明：损伤演化、裂纹路径与实验观察高度一致；能量收敛性随网格细化而提升；裂尖速度与雷利波速对比符合理论预期；计算效率显著高于完全非局部方法，且可并行实现。

**⚠️ 局限性**

局限性包括：①仅采用弹性损伤模型，无法捕捉高冲击速度下的韧性失效；②显式时间积分仅在条件稳定下适用，需严格满足CFL条件；③目前仅实现二维问题，三维扩展尚未完成；④缺少塑性或多物理耦合的考虑。

---

## 201. From Scores to Queues: Operationalizing Cross-Chain Obfuscation Signals for Smart-Contract Audits

**arXiv ID:** 2601.17356 | [PDF](https://arxiv.org/pdf/2601.17356v1)

**作者:** Yao Zhao `[一作]` (Hong Kong Polytechnic University), Shen Wang `[通讯]` (Final Round AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并评估了一种高效的跨链智能合约混淆度评分模型 HObfNET，并利用该模型在四条 EVM 链上进行百万级混淆度测量，探索阈值漂移、尾部结构、跨链扩散以及与真实攻击事件的关联。

**💡 创新点**

提出了基于层次注意力网络的混淆度评分代理模型 HObfNET，显著提升评估速度（千倍级别），并首次系统性分析跨链混淆度分布漂移、尾部结构特征及其跨链扩散，为多链安全运维提供可操作的队列划分和二级分流策略。

**🔧 技术方法**

使用层次注意力网络（Hierarchical Attention Network）对分段字节码进行编码，结合多任务学习恢复工具特征，并在 GPU 上实现快速推理；同时结合字节码规范化、哈希去重、Jaccard 重叠分析等技术进行跨链比较。

**📊 数据集**

在以太坊 1,042,923 条合约的 ObfProbe 标签上进行训练，并在币安智能链 2,308,899 条、Polygon 288,611 条、Avalanche 96,173 条去重后字节码上进行推理；同时采集公开事件地址、交易记录等样本用于对齐验证。

**📈 对比分析**

通过 MAPE 8.20%、MAE 0.634、PCC 0.916 与 ObfProbe 在以太坊测试集上对比，推理时间约 8–9 ms/合约，比 ObfProbe 速度提升 2.3k–5.2k 倍；在跨链测量中，阈值漂移量达 0.48%–2.32%；尾部结构及重叠度等指标也被系统评估。

**⚠️ 局限性**

代理模型受限于 ObfProbe 的噪声标签，跨链校准仍需改进；未直接捕获语义安全风险，仅提供混淆度排序；对公开事件的对齐受限于报告缺失与链间可观测性差异。

---

## 202. Joint Uplink-Downlink Fronthaul Bit Allocation in Fronthaul-Limited Massive MU-MIMO Systems

**arXiv ID:** 2601.17423 | [PDF](https://arxiv.org/pdf/2601.17423v1)

**作者:** Yasaman Khorsandmanesh `[一作]` (KTH Royal Institute of Technology), Joakim Jalden `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在受限上行/下行回传链路容量的 Massive MU‑MIMO 系统中，提出了一种联合上行/下行回传比特分配算法，最大化总谱效率。

**💡 创新点**

创新点在于将上行 CSI 与下行预编码矩阵的量化比特进行联合最优分配，给出闭式 SE 表达式并实现精确的离散搜索求解。

**🔧 技术方法**

采用 AQNM 量化模型、硬化界下的 SE 推导、MRT/ ZF/ WF 预编码、Monte Carlo 仿真与有限线搜索算法。

**📊 数据集**

使用仿真数据：M=128 天线，K=8 UE，τc=200，τp=K，Rayleigh 衰落，噪声/信噪比可变。

**📈 对比分析**

通过与理想完美 CSI/预编码基准以及固定比特分配方案对比，显示在中高 SNR 时 5 比特 CSI 或平衡分配能达到约 12 bit/s/Hz；低 SNR 时 80% 分配给 CSI 能获得最高 SE。

**⚠️ 局限性**

局限性包括仅考虑独立 Rayleigh 信道、仅针对 TDD 系统、只关注量化噪声模型、未考虑链路延迟与实际硬件非理想等。

---

## 203. Using psychological theory to ground guidelines for the annotation of misogynistic language

**arXiv ID:** 2601.17417 | [PDF](https://arxiv.org/pdf/2601.17417v1)

**作者:** Artemis Deligianni `[一作]` (University of Edinburgh), Leonidas A. A. Doumas `[通讯]` (University of Edinburgh)

**通讯引用:** 1861 | [OpenAlex ID](https://openalex.org/A5070044658)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于心理学理论的女权主义仇恨言论标注指南并生成对应数据集；

**💡 创新点**

创新点在于将多维心理学概念（如敌意性别歧视、温和性别歧视、性别本质主义等）纳入标注框架，提升标注一致性并覆盖更广泛的仇女语言；

**🔧 技术方法**

采用人工标注、LLM提示（Llama 3.3、Mistral Large、Qwen2）以及贝叶斯混合效应回归分析；

**📊 数据集**

使用自采自四个Reddit子版块的帖子以及改编的Ambivalent Sexism Inventory（ASI）作为评测数据集；

**📈 对比分析**

通过与现有Guest等标注方案对比，LLM在有指南提示时整体表现提升，但对自标注数据和ASI仍表现较差，表明LLM受限于固有社会偏见；

**⚠️ 局限性**

局限包括数据集规模小、仅二元分类、文化/语言变迁导致指南需更新，以及LLM偏见难以完全消除。

---

## 204. DiffusionCinema: Text-to-Aerial Cinematography

**arXiv ID:** 2601.17412 | [PDF](https://arxiv.org/pdf/2601.17412v1)

**作者:** Valerii Serpiva `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 1963 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套基于扩散模型的无人机文本驱动创意捕捉系统（DiffusionCinema），能够将自然语言提示和初始画面自动映射为无人机的航拍轨迹并执行，生成与提示匹配的电影级视频；

**💡 创新点**

创新点在于将视频扩散模型作为电影化运动先验，直接从文本生成高质量、连贯的航拍轨迹，实现“文本→电影化航拍”交互；

**🔧 技术方法**

使用了多模态视觉‑语言模型（VLN）提取语义动作、扩散模型（DiT）生成视频/运动轨迹、ORB‑SLAM3/视觉里程计提取状态、PID 控制器执行轨迹，以及OpenVINS等定位算法；

**📊 数据集**

训练数据为大规模人类拍摄视频集（用于扩散模型学习电影化运动规律），并使用真实飞行记录进行轨迹匹配验证；

**📈 对比分析**

通过10名参与者的用户研究，对比手动遥控与DiffusionCinema；NASA‑TLX工作负荷平均值为21.6（vs 58.1）、精神负担11.5（vs 60.5）、挫折感14.0（vs 54.5）；系统成功率约46.7%，轨迹提取与执行误差低；

**⚠️ 局限性**

局限性包括：成功率仅为约46%，受模型生成幻象和场景理解限制；数据集覆盖度有限，泛化能力待提升；实时性和动态场景适配能力不足，需进一步加入手势、实时反馈等多模态输入。

---

## 205. SMV-EAR: Bring Spatiotemporal Multi-View Representation Learning into Efficient Event-Based Action Recognition

**arXiv ID:** 2601.17391 | [PDF](https://arxiv.org/pdf/2601.17391v1)

**作者:** Rui Fan `[一作]`, Weidong Hao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于事件摄像头的动作识别框架SMV-EAR，利用时空多视图表示学习实现高效精确的事件动作识别。

**💡 创新点**

创新点包括：① 设计了可保持平移不变性的时空多视图表示TISM；② 构建了双分支动态交叉视图融合DDCF，按样本自适应加权不同视图的特征；③ 引入了多样化时间扭曲数据增强DTW，模拟真实动作速度变化。

**🔧 技术方法**

核心技术包括：事件到二维T‑H、T‑W视图的全局聚合表示、基于ResNet的双分支特征提取、跨视图多头注意力加权融合、时间扭曲变换与多功能窗口聚合。

**📊 数据集**

在三个主流事件动作数据集上评估：HARDVS、DailyDVS‑200 与 THU‑EACT‑50‑CHL。

**📈 对比分析**

与现有SMVRL基线MVF‑Net及其他SOTA方法比较，SMV‑EAR 在 HARDVS、DailyDVS‑200、THU‑EACT‑50‑CHL 上分别提升 Top‑1 7.0%、10.7% 与 10.2%，参数量仅 23.5 M，算力仅 1.8 G MACs，显著低于多数基线且性能更优。

**⚠️ 局限性**

局限性主要是：对相机运动噪声敏感、对微动作识别能力有限；背景噪声滤除、频域建模与学习型增强策略仍待进一步研究。

---

## 206. YASA: Scalable Multi-Language Taint Analysis on the Unified AST at Ant Group

**arXiv ID:** 2601.17390 | [PDF](https://arxiv.org/pdf/2601.17390v1)

**作者:** Yayi Wang `[一作]` (Ant Group), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 68899 | [OpenAlex ID](https://openalex.org/A5115602103)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出一种统一多语言静态污点分析框架 YASA，能够在同一内部抽象模型（UAST）下对多种语言进行深度数据流分析；

**💡 创新点**

核心创新在于设计了跨语言的统一抽象语法树（UAST）与语言无关的语义分析器，显著降低多语言支持的工程成本并提升分析精度；

**🔧 技术方法**

采用 UAST 作为统一中间表示，结合上下文、路径、字段敏感的指针分析与语言特定语义处理，再通过插件式污点传播规则实现跨框架检测；

**📊 数据集**

使用行业级多语言微基准 xAST（覆盖 Java、JavaScript、Python、Go）以及 Ant Group 内部 100+M 行代码的真实项目集；

**📈 对比分析**

与 6 个单语言工具和 2 个多语言框架对比，YASA 在四种语言的 soundness 与 completeness 评分均提升 10%~40%，扫描速度平均约 31.8 KLOC/min；

**⚠️ 局限性**

局限性包括：UAST 设计受当前语言约束，支持语言有限；不完全 soundness（循环展开有限）；对未知函数默认保守污点传播可能导致误报。

---

## 207. ONRW: Optimizing inversion noise for high-quality and robust watermark

**arXiv ID:** 2601.17388 | [PDF](https://arxiv.org/pdf/2601.17388v1)

**作者:** Xuan Ding `[一作]`, Yao Zhu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

暂无信息

**💡 创新点**

暂无信息

**🔧 技术方法**

暂无信息

**📊 数据集**

暂无信息

**📈 对比分析**

暂无信息

**⚠️ 局限性**

暂无信息

---

## 208. Physical Prompt Injection Attacks on Large Vision-Language Models

**arXiv ID:** 2601.17383 | [PDF](https://arxiv.org/pdf/2601.17383v1)

**作者:** Chen Ling `[一作]` (Wuhan University), Changhai Ou `[通讯]` (Wuhan University)

**通讯引用:** 652 | [OpenAlex ID](https://openalex.org/A5019287385)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了首个针对大型视觉语言模型（LVLM）的物理注入攻击（Physical Prompt Injection Attack, PPIA），通过在现实环境中放置包含恶意文字提示的物理容器，诱使LVLM在不获取输入文本的情况下错误推理。

**💡 创新点**

创新点包括：① 完全黑盒、查询无关的攻击模型；② 通过离线候选提示筛选与可识别性评估，确保提示在视觉感知时被准确识别；③ 利用时空注意力分布寻找最优物理放置位置，提高攻击成功率；④ 对多种语言、光照、角度、距离等现实因素的鲁棒性评估。

**🔧 技术方法**

使用的技术主要有：① 大语言模型生成多样化恶意提示；② 预训练视觉-语言模型（如Llama-3.2-11B-Vision）进行文本可识别性评估（交叉熵）；③ CLIP 视觉变压器提取时空注意力用于位置搜索；④ 在仿真（Habitat、Embodied City）与真实无人车平台上进行验证。

**📊 数据集**

数据集与实验环境包括：① 仿真环境（Habitat、Embodied City）用于生成10种场景；② 10款主流LVLM（GPT‑4o, GPT‑4‑turbo, Gemini‑1‑pl, Gemini‑1‑p2, Gemini‑1‑fl, Claude‑3‑5‑sl, Claude‑3‑5‑haiku, Llama‑3.2‑11b‑vision, Llama‑3.2‑90b‑vi 等）；③ 物理实验使用一辆搭载摄像头的无人车，在室内外不同光照、容器类型、模糊程度、距离与视角下进行。

**📈 对比分析**

与三类基线（多图文本注入、SceneTap、SGTA）相比，PPIA 在 QA、TP、NAV 三大任务中均实现了 70%–98% 的攻击成功率（ASR），在大多数模型上均优于基线；在实际物理实验中，ASR 超过 80%，在不同光照、距离、容器类型下仍保持较高成功率。

**⚠️ 局限性**

局限性：① 受摄像头分辨率限制，低分辨率可能导致文字识别失败；② 物理实验多在受控环境中进行，未覆盖极端动态或拥挤场景；③ 需要在目标环境中事先布置容器，可能被人工监控识别；④ 对非英语提示效果较差，受模型训练数据偏倚影响。

---

## 209. WarrantScore: Modeling Warrants between Claims and Evidence for Substantiation Evaluation in Peer Reviews

**arXiv ID:** 2601.17377 | [PDF](https://arxiv.org/pdf/2601.17377v1)

**作者:** Kiyotada Mori `[一作]` (Nara Institute of Science and Technology), Yoshitaka Ushiku `[通讯]` (OMRON SINIC X Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个基于论证结构的科学评审质量评估框架，核心指标为WarrantScore，显式生成并评估论证中的warrant（推理桥接），从而更准确地衡量评审意见的证据支撑和逻辑合理性。

**💡 创新点**

创新点在于：①首次将Toulmin模型中的warrant作为可解释的评估因素；②通过LLM自动生成warrant并用四点Likert评分衡量其可接受性，提升与人工主观评分的相关性；③在不依赖人工标注训练的前提下，提供可解释的评估流程。

**🔧 技术方法**

技术手段包括：使用大型语言模型（GPT‑5、Gemini 2.5 Flash）生成warrant，使用二分类LLM判断可接受性并重复尝试；LLM‑as‑judge给四点评分；使用Sentence‑BERT计算语义相似度；以及基于现代BERT模型（ModernBERT）抽取claim和evidence。

**📊 数据集**

采用公开的SubstanReview（50条人工标注的评审）和RottenReview（509条评审）的人类评分数据；使用ModernBERT抽取论证成分；为鲁棒性评估还使用了ReviewGuardReview（足够/不足评审）和ElongatedReview（原始与扩展版评审）两套数据。

**📈 对比分析**

与传统的SubstanScore、support_claims、coherence_rate以及基于回归的RM等指标对比；在SubstanReview中WarrantScore与人工评分的Spearman相关系数为0.82，显著高于SubstanScore（0.70）和其他基线；在RottenReview中与RM（0.49）相当，但WarrantScore对文本长度不敏感，且在ReviewGuardReview和ElongatedReview的鲁棒性测试中能正确区分优劣评审且保持对扩展版的评估不变。

**⚠️ 局限性**

限制包括：①warrant的形式被限定为常识性桥接，缺乏多样化和细粒度的论证类型；②评估过程高度依赖LLM，评分结果可能带有模型偏差且解释性仍有限；③实验数据规模较小，跨领域泛化和真实评审场景的验证仍待进一步扩展。

---

## 210. Robust Privacy: Inference-Time Privacy through Certified Robustness

**arXiv ID:** 2601.17360 | [PDF](https://arxiv.org/pdf/2601.17360v1)

**作者:** Jiankai Jin `[一作]` (360 AI Security Lab), Quanchen Zou `[通讯]` (360 AI Security Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为 Robust Privacy（RP）的推理时隐私概念，利用已证明的鲁棒性保证模型在输入附近保持输出不变，从而阻止攻击者通过预测结果推断敏感属性或重构训练数据。

**💡 创新点**

创新点在于将已知的鲁棒性证明（certified robustness）重新诠释为推理时的隐私保障，并引入 Attribute Privacy Enhancement（APE）来量化输入级不变性对敏感属性隐私的提升。

**🔧 技术方法**

主要技术包括随机平滑（Randomized Smoothing）实现 RP 的概率性鲁棒性证书，以及对模型进行输入域上的输出不变性检验；同时评估了鲁棒性参数 σ 与采样数 N 对隐私与准确率的影响。

**📊 数据集**

实验使用了医疗保险成本预测数据集（约10万条记录）进行敏感属性（BMI）的推荐任务，以及 CelebA 人脸数据集对模型反演攻击的评估。

**📈 对比分析**

在推荐任务中，RP 通过增大 σ 使正类预测扩散至阈值以下，敏感属性推断区间平均扩大 0.56–0.65；在反演攻击中，RP 把攻击成功率从 73% 降至 4%（σ=0.1,N=100）或 44%（σ=0.03,N=100），同时保持 59%–100% 的分类准确率。

**⚠️ 局限性**

局限性包括：鲁棒性证明的保守性导致实际隐私提升可能低于理论值；随机平滑的计算成本随采样数增大；RP 只对单个输入的局部不变性作保障，无法防御全局信息泄露；且对非欧氏距离或高维稀疏特征的适用性尚未验证。

---

## 211. Spectral Geometry for Deep Learning: Compression and Hallucination Detection via Random Matrix Theory

**arXiv ID:** 2601.17357 | [PDF](https://arxiv.org/pdf/2601.17357v1)

**作者:** Davide Ettori `[一作]` `[通讯]` (Politecnico di Milano), Davide Ettori (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出基于谱几何和随机矩阵理论的统一框架，开发了EigenTrack用于实时检测LLM和VLM的幻觉与OOD行为，以及RMT‑KD用于基于谱特征的知识蒸馏压缩模型。

**💡 创新点**

创新点在于利用隐藏激活的特征值动态提供可解释的可靠性监测，并以异常特征值为切入点进行稀疏投影的迭代蒸馏，实现了压缩率与精度兼顾的全新压缩方法。

**🔧 技术方法**

技术手段包括谱特征提取（特征值分布、熵、特征值间隙、与Marčenko–Pastur基线的偏差）、轻量递归分类器、随机矩阵理论指导的自蒸馏、以及低维子空间投影。

**📊 数据集**

实验使用了多种大型语言模型（如GPT‑X、PaLM）和视觉‑语言模型（如CLIP、BLIP），并在公开的NLP、VLM基准集（如Wikitext, GLUE, ImageNet‑V2）进行评估。

**📈 对比分析**

与现有的幻觉检测方法和压缩技术（Pruning、低秩近似）对比，EigenTrack在准确率与延迟上均实现SOTA，RMT‑KD在压缩率达到70%+的同时保持1%以内的准确率下降。

**⚠️ 局限性**

局限性包括：对谱统计的敏感性导致对极端噪声输入仍可能误判；压缩过程中需要额外的自蒸馏循环，可能导致训练成本提升；以及目前实验主要聚焦于Transformer类模型，尚未充分验证在其他网络结构上的通用性。

---

## 212. Safeguard: Security Controls at the Software Defined Network Layer

**arXiv ID:** 2601.17355 | [PDF](https://arxiv.org/pdf/2601.17355v1)

**作者:** Yi Lyu `[一作]` (University of Wisconsin Madison), Joe Catudal `[通讯]` (University of Wisconsin Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了Safeguard框架，结合规则化策略限制数据驱动的SDN安全决策，防止误判导致的网络中断；

**💡 创新点**

创新点在于将规则层与机器学习层耦合，通过显式的“已知良好流”保护规则抑制误判，实现数据驱动与规则化的双重保险；

**🔧 技术方法**

技术包括OpenFlow/OpenVSwitch控制器Floodlight、实时流量采集（tcpdump）、基于签名的入侵检测、HTTP接口与控制器交互、以及自定义的Safeguard规则集；

**📊 数据集**

使用了自建的实验数据集：在CloudLab虚拟网络环境下，模拟的DDoS/DoS攻击流量与正常TCP连接流量，手动标注为攻击或正常；

**📈 对比分析**

通过对比仅使用签名检测与加上Safeguard规则的两种方案，实验结果显示加规则后误删正常流量比例显著降低，性能差异仅为30秒的超时阈值调整；

**⚠️ 局限性**

局限性包括：实验规模受限于CloudLab资源，未覆盖真实多样化网络流量；使用签名而非真实机器学习模型导致泛化性不足；缺乏对高吞吐量攻击下的实时性评估。

---

## 213. Prompt and Circumstances: Evaluating the Efficacy of Human Prompt Inference in AI-Generated Art

**arXiv ID:** 2601.17379 | [PDF](https://arxiv.org/pdf/2601.17379v1)

**作者:** Khoi Trinh `[一作]` (University of Oklahoma), Anindya Maiti `[通讯]` (University of Oklahoma)

**通讯引用:** 482 | [OpenAlex ID](https://openalex.org/A5045020872)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估人类以及人机协同推断AI生成图像背后原始提示词的准确性，并比较其与原始提示词生成的图像相似度。

**💡 创新点**

首次将人类实验与LLM辅助提示生成相结合，并提出基于Kolmogorov–Smirnov检验的分布级别提示相似性评估方法。

**🔧 技术方法**

使用CLIP Interrogator、GPT‑4、LPIPS、CLIP分数、ImageHash等视觉相似度度量，以及KS检验对分布差异进行统计比较。

**📊 数据集**

构建了Controlled（来自Lexica的100条受限提示）和Uncontrolled（来自PromptHero的100条随机提示）两组数据集，使用Midjourney v5、Stable Diffusion XL、DreamShaper XL和Realistic Vision v5生成图像。

**📈 对比分析**

在每条提示下生成200张原始图像与50张推断图像，采用KS检验比较相似度分布，得到ImageHash命中率约53%，LPIPS命中率约22%/10%，CLIP命中率约7%；人机合成仅提升ImageHash而对LPIPS/CLIP不利。

**⚠️ 局限性**

对细节与风格的语义恢复能力不足，LLM融合策略简单导致语义一致性下降；实验仅覆盖有限模型与提示集，缺乏对更广泛场景的泛化能力。

---

## 214. A Syllogistic Probe: Tracing the Evolution of Logic Reasoning in Large Language Models

**arXiv ID:** 2601.17426 | [PDF](https://arxiv.org/pdf/2601.17426v1)

**作者:** Zhengqing Zang `[一作]` (Zhejiang University), Haobo Wang `[通讯]` (Zhejiang University)

**通讯引用:** 758 | [OpenAlex ID](https://openalex.org/A5049707744)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在推理三段论时，逻辑框架是否从传统的存在性推理逐渐转向现代的布尔逻辑。

**💡 创新点**

创新点在于将存在性导入作为切入点构建新的三段论数据集，并系统评估模型在传统与现代逻辑下的行为，揭示规模、思考训练和基模型对逻辑转变的影响。

**🔧 技术方法**

采用大规模语言模型（Qwen、Llama、Gemma 等），使用强化学习“思考”训练、提示工程以及对比实验等技术。

**📊 数据集**

使用自行构建的 9600 条三段论样本，涵盖中英双语、空与非空小项、15+9 合法形式。

**📈 对比分析**

通过在传统和现代逻辑下分别计算准确率、连贯性等指标，对比不同规模、不同训练方式模型，发现 Qwen 系列规模增大时 Acc_m 上升并出现逻辑转变；思考训练在相同规模下可显著提升现代逻辑表现；基模型决定转变的易难度。

**⚠️ 局限性**

局限在于仅关注三段论和存在性问题，未探究更广泛的第一阶逻辑；评价仅基于最终合法性判断，缺少对中间推理步骤的分析；蒸馏实验范围有限，结论不一定泛化。

---

## 215. GraphPilot: GUI Task Automation with One-Step LLM Reasoning Powered by Knowledge Graph

**arXiv ID:** 2601.17418 | [PDF](https://arxiv.org/pdf/2601.17418v1)

**作者:** Mingxian Yu `[一作]` (Sun Yat-sen University), Xu Chen `[通讯]` (Sun Yat-sen University)

**通讯引用:** 21266 | [OpenAlex ID](https://openalex.org/A5100385692)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为GraphPilot的移动GUI自动化代理，利用针对每个应用构建的知识图谱，支持几乎一次LLM查询即可生成完整操作序列，从而显著提升任务完成率并降低延迟；

**💡 创新点**

创新点在于将页面与元素功能及其转换规则组织为知识图谱，并在LLM推理中将图谱作为丰富上下文，引入验证器校验和动态HTML请求回退机制，实现近一步的单步推理；

**🔧 技术方法**

使用GPT‑4o作为LLM，基于HTML结构的界面表示，构建知识图谱（节点为页面/元素功能，边为转换规则），通过prompt生成和验证器迭代校正；

**📊 数据集**

在DroidTask基准上评估，包含13个Android应用共158个高层任务；

**📈 对比分析**

与Mind2Web、AutoDroid比较：任务完成率74.1%高于65.2%和62.0%；LLM查询次数1.03次（vs 4.54）和延迟下降70.4%/66.7%；表明在准确率和效率上均有显著提升；

**⚠️ 局限性**

局限性包括对知识图谱完整性的高度依赖、动作空间仅包含点击/输入等基本操作、评估仅考虑与基准序列完全一致的结果，未覆盖多路径有效方案。

---

## 216. HAAF: Hierarchical Adaptation and Alignment of Foundation Models for Few-Shot Pathology Anomaly Detection

**arXiv ID:** 2601.17405 | [PDF](https://arxiv.org/pdf/2601.17405v1)

**作者:** Chunze Yang `[一作]` (Xi'an Jiaotong University), Chen Li `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 29425 | [OpenAlex ID](https://openalex.org/A5100379155)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种层次化适配与对齐框架 HAAF，旨在将 Vision‑Language 基础模型在少样本病理异常检测任务中进行精细化调优。

**💡 创新点**

创新点主要包括跨层尺度对齐（CLSA）顺序化的视觉‑文本交互机制、双分支推理（语义 + 几何）以及轻量化的残差 Adapter 设计，解决了传统并行融合导致的粒度不匹配问题。

**🔧 技术方法**

采用 CLIP/CONCH 视觉‑语言基础模型，结合残差视觉 Adapter、残差文本 Adapter、跨层注意力模块以及双分支评分策略，整体保持参数高效。

**📊 数据集**

实验数据集覆盖四个病理领域：Camelyon16、BRACS（乳腺）、SICAPv2（前列腺）和 NCT‑CRC（结肠），全部以 ROI 级别二分类（正常/异常）构成。

**📈 对比分析**

在 4‑shot 评估中，HAAF 在 CLIP 基础上平均 AUC 84.98%–83.95% 领先于 MVFA、MadCLIP 等基线；在 CONCH 基础上更是实现 91.97%–94.05% AUC，显著优于同类 PEFT 方法和其他 V‑L 适配器。

**⚠️ 局限性**

局限性包括：仍需依赖预训练基础模型的可迁移性；在极低样本或跨病理类型泛化时表现尚不稳定；以及对 ROI 标注的依赖，限制了在完全无标注场景的直接应用。

---

## 217. CLM-Bench: Benchmarking and Analyzing Cross-lingual Misalignment of LLMs in Knowledge Editing

**arXiv ID:** 2601.17397 | [PDF](https://arxiv.org/pdf/2601.17397v1)

**作者:** Yucheng Hu `[一作]` (Tianjin University), Juesi Xiao `[通讯]` (Tianjin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了以中文为主的文化感知知识编辑基准 CLM‑Bench，利用该基准评估多语言知识编辑（MKE）的性能，并对跨语言编辑的不一致性进行了机制分析。

**💡 创新点**

① 采用“先中文、再英文”本土化构建，避免翻译痕迹并覆盖本土文化实体；② 系统揭示跨语言编辑的正交不匹配与线性可加性；③ 提供层级向量几何解释，证明中英编辑在参数空间中几乎正交且可线性组合。

**🔧 技术方法**

主要使用 locate‑and‑edit 方式的三种主流批量编辑算法：MEMIT、AlphaEdit、PMET；并对 Llama‑3、Qwen‑2、Mistral‑7B、Llama2‑Chinese‑7B 等四大 LLM 进行实验。

**📊 数据集**

核心数据集为 CLM‑Bench（1,010 条中文 CounterFact 对，已与英文本对齐），辅以 ZsRE 与 CounterFact 的部分公开数据。

**📈 对比分析**

将上述三种编辑方法在四款模型上进行单语与混语批量编辑比较，采用 reliability、generality、locality 四维指标及跨语言迁移得分。实验表明：中文编辑能达到约 60% 的 reliability，但对应的英文 reliability 仅 20%；混语编辑虽线性可加，但整体性能仍落后于单语编辑；不同层级和批量大小对跨语言误差的影响均不显著，说明问题为固有结构性。

**⚠️ 局限性**

研究仅聚焦中文与英文；仅评估 locate‑and‑edit 的批量编辑方法；机制分析只涉及单层参数向量，无法全面描述模型内部动态；因此结论对其他语言、编辑范式及更深层次机制的推广仍待验证。

---

## 218. Revisiting Modality Invariance in a Multilingual Speech-Text Model via Neuron-Level Analysis

**arXiv ID:** 2601.17387 | [PDF](https://arxiv.org/pdf/2601.17387v1)

**作者:** Toshiki Nakai `[一作]` (Saarland University), Vera Demberg `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 4309 | [OpenAlex ID](https://openalex.org/A5023605306)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究 SeamlessM4T v2 通过 neuron 级别分析探讨语音与文本表示的一致性及其对解码的影响。

**💡 创新点**

将 AP 排名方法扩展到多模态模型，结合中介干预、激活不平衡分析三种互补手段系统性揭示模态不完整的 invariance。

**🔧 技术方法**

使用 AP 排序识别语言/模态专属 neuron，median 替换干预、交叉注意力 key/value 关注度分析，以及激活 magnitude inequality 与 Gini 指标。

**📊 数据集**

采用 FLEURS 语料（6 种语言）与 XTTS v2 合成语音作为实验数据集。

**📈 对比分析**

通过比较专属 neuron 干预与随机干预的性能差异，发现专属 neuron 对模型性能影响有限；语音→文本翻译相对稳健，整体揭示解码中模态不一致导致性能下降。

**⚠️ 局限性**

研究仅基于单一模型、6 语言、合成语音，且缺乏对单个 neuron 计算细节的深入分析，限制了结论的普适性。

---

## 219. Diversified Scaling Inference in Time Series Foundation Models

**arXiv ID:** 2601.17376 | [PDF](https://arxiv.org/pdf/2601.17376v1)

**作者:** Ruijin Hua `[一作]` (HUST), Yiyuan Yang `[通讯]` (University of Oxford)

**通讯引用:** 836 | [OpenAlex ID](https://openalex.org/A5019084435)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

系统研究时间序列基础模型（TSFM）的推理阶段规模化和多样化采样，提供理论框架并通过大规模实验验证其对预测性能的提升。

**💡 创新点**

创新点包括：①揭示标准采样对TSFM性能的局限性；②提出通过输入扰动扩展采样分布的多样化推理策略；③推导批量大小临界阈值的理论分析；④引入RobustMSE度量评估模型在固定计算预算下的最佳可达性能；⑤系统比较不同模型、数据集、扰动方式和聚合方法的效果。

**🔧 技术方法**

使用的技术：多种采样策略（标准采样、温度调节、top‑p、输入扰动+采样）、聚合方法（Exact Match、Majority Voting）、模型规模、上下文长度扩展；理论分析（支持集扩展、期望最小损失、临界样本数推导）；实验框架（5次重复、5折交叉验证）以及RobustMSE评估。

**📊 数据集**

使用的数据集包括：ETTh1、ETTm1、Electricity、Traffic 四大公开时序数据集。

**📈 对比分析**

通过对不同采样、聚合、扰动组合的MSE和RobustMSE进行对比，实验发现多样化采样可提升最高约50%性能，RobustMSE在所有模型上更稳定，TimesFM在大多数场景获得最低RobustMSE。

**⚠️ 局限性**

局限性：①对扰动设计高度敏感，某些扰动会导致性能退化；②高温或极大模型时收益不稳定；③实验主要基于公开数据集，缺少对极端场景或工业真实数据的验证；④理论推导基于理想假设，实际应用需进一步验证。

---

## 220. Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

**arXiv ID:** 2601.17367 | [PDF](https://arxiv.org/pdf/2601.17367v1)

**作者:** Zecheng Tang `[一作]` (Soochow University), Min Zhang `[通讯]` (Soochow University)

**通讯引用:** 38345 | [OpenAlex ID](https://openalex.org/A5013794939)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Elastic Attention，利用轻量级 Attention Router 在推理时动态决定每个注意力头是采用全注意力（FA）还是稀疏注意力（SA），从而实现可变稀疏比例的高效推理。

**💡 创新点**

创新点在于：①不修改预训练模型参数，仅通过 Router 在每层头级别进行 FA/SA 路由；②使用 Gumbel‑Softmax 连续松弛与 Straight‑Through 估计解决离散决策的训练难题；③在任务层面区分稀疏鲁棒与稀疏敏感任务，自动分配合适的稀疏度。

**🔧 技术方法**

核心技术包括：Mixture‑of‑Experts 风格的 Attention Router、Gumbel‑Softmax 近似离散路由、Straight‑Through 估计、融合核（fused kernel）实现 FA 与 SA 的并行计算、基于 GQA 的多头架构，以及任务特定的稀疏度目标约束。

**📊 数据集**

实验数据集涵盖长上下文基准 LongBench、LongBench‑V2、RULER 以及自建训练集（ChatQA2‑Long‑SFT、MuSiQue、CoLT‑132K、GovReport、XSum），序列长度从 8K 到 64K 词，模型为 Qwen3‑4B/8B 与 Llama‑3.1‑8B‑Instruct。

**📈 对比分析**

与 DuoAttention、PruLong、InfLLM‑V2 等现有稀疏注意力方法在相同 backbone 上进行对比；在 LongBench‑E、LongBench‑V2、RULER 等任务上，Elastic Attention 在保持或接近全 FA 性能的同时，显著提升了推理速度，尤其在 64K 以上长序列时表现突出。

**⚠️ 局限性**

局限性包括：对稀疏鲁棒任务（如代码与摘要）的性能略低于部分基线；需手动设定任务级稀疏阈值；在极大上下文长度下仍可能遇到显存瓶颈；目前仅实现单 GPU 部署，尚未扩展到多 GPU/多机场景。

---

## 221. PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling

**arXiv ID:** 2601.17354 | [PDF](https://arxiv.org/pdf/2601.17354v1)

**作者:** Wenzhi Guo `[一作]` (Hong Kong Polytechnic University), Bing Wang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 18307 | [OpenAlex ID](https://openalex.org/A5100382568)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在手机设备上实现高质量3D Gaussian Splatting（3DGS）训练的完整框架PocketGS

**💡 创新点**

通过三种共设计的算子——几何先验构造、先验驱动的高斯初始化、硬件对齐的可微分渲染，解决了传统3DGS在移动端资源受限时的输入可靠性、初始化收敛性和可微分性三大矛盾

**🔧 技术方法**

使用GPU原生束调整（Bundle Adjustment）与单参考MVS构造几何先验，利用局部表面统计进行高斯方差和方向初始化，采用前后向Alpha合成与梯度散射的离散可微分渲染实现GPU原地优化

**📊 数据集**

在NeRF-Synthetic、LLFF以及作者自行收集的MobileScan（手机捕获）三大数据集上进行评估

**📈 对比分析**

与两类工作站基线（基于SfM的稀疏先验和基于MVS的密集先验）在相同迭代预算下比较，PocketGS在所有数据集上均实现了更高的PSNR/SSIM、更低的LPIPS，并且在iPhone 15上完成约4分钟训练、峰值内存<3 GB，显著优于基线的时间和内存开销

**⚠️ 局限性**

仍受限于较低的迭代次数导致的细节逼近精度，且对极端低光或大运动模糊的场景表现尚不理想，未来需要进一步降低对高斯数量和迭代预算的依赖

---

## 222. HyDeMiC: A Deep Learning-based Mineral Classifier using Hyperspectral Data

**arXiv ID:** 2601.17352 | [PDF](https://arxiv.org/pdf/2601.17352v1)

**作者:** M. L. Mamud `[一作]` (Pacific Northwest National Laboratory), M. K. Mudunuru `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 788 | [OpenAlex ID](https://openalex.org/A5060765277)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并训练了HyDeMiC模型，实现了在不同噪声水平下的矿物分类；

**💡 创新点**

创新在于采用1D卷积神经网络对USGS矿物光谱进行编码，并通过像素级推理将其扩展到2D hyperspectral影像，同时引入置信度评估以评估模型的不确定性；

**🔧 技术方法**

使用CNN（包含卷积、池化、批归一化、LeakyReLU、Dropout）、AdamW优化器、余弦退火学习率调度以及交叉熵损失；

**📊 数据集**

利用美国地质调查局（USGS）矿物光谱库的444条1D光谱（115种矿物），通过AVIRIS传感器响应函数模拟得到224波段数据；还生成了包含不同噪声级别（1%、2%、5%、10%）的2D合成hyperspectral图像；

**📈 对比分析**

通过Matthews相关系数（MCC）、真阳性率（TPR）和预测置信度（PC）进行评估；在无噪声下MCC=1.00；5%噪声下MCC≈0.99；10%噪声下MCC=0.92，显示出对噪声的较强鲁棒性；

**⚠️ 局限性**

仅在合成数据上验证，未在真实航空/卫星影像上测试；缺乏对混合像素的处理；矿物类别仅限115种，难以覆盖更广泛的地质场景；

---

## 223. Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning

**arXiv ID:** 2601.17428 | [PDF](https://arxiv.org/pdf/2601.17428v1)

**作者:** Ziming Li `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**通讯引用:** 19610 | [OpenAlex ID](https://openalex.org/A5044258783)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

提出了一种基于学习进度的自动课程强化学习框架（LP-ACRL），通过在线评估任务学习进展来动态调整任务采样分布，实现在多维、无结构任务空间中的自动课程生成，并在ANYmal D四足机器人上实现了高速度粗糙地形行走。

**💡 创新点**

核心创新在于：①利用基于回报的学习进度评估直接驱动任务采样；②无需预设难度排序或手工设计课程；③在复杂多任务空间中实现快速、可扩展的学习；④通过软max学习进度调节任务概率。

**🔧 技术方法**

技术包括：强化学习（如PPO/ SAC 等）、任务采样分布更新（softmax + 温度 β）、基于回报的学习进度计算、Teacher-Student 蒸馏、以及IsaacLab仿真与ANYmal D真实部署。

**📊 数据集**

数据集：基于IsaacLab的仿真环境构建的任务空间，涵盖多级线速度、角速度、六种地形类型及四级地形难度，共计600个离散任务实例；以及对应的真实地形高度图用于机器人部署。

**📈 对比分析**

与多种基准（绝对学习进度、优先级回放、手工课程、低奖励优先课程、均匀采样）比较，LP-ACRL 在收敛速度、成功率和最终性能（EPTE‑SP 误差低、平均奖励高）上均优于所有基准，尤其在600任务实例的规模化任务空间中，在1500次迭代内即可达到 80% 成功率，而其他方法需超过3000次。

**⚠️ 局限性**

局限性包括：①仍依赖离散化任务空间，连续任务难以直接处理；②学习进度的估计对噪声敏感，极端任务可能导致采样不稳定；③在极大任务空间下，softmax 更新可能产生梯度消失问题；④需额外的硬件与仿真对齐以实现真实部署。

---

## 224. Faster modular composition using two relation matrices

**arXiv ID:** 2601.17422 | [PDF](https://arxiv.org/pdf/2601.17422v1)

**作者:** Vincent Neiger `[一作]` (Sorbonne Université), Gilles Villard `[通讯]` (CNRS)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的模幂运算（univariate modular composition）算法，利用多项式矩阵与关系模块的最小基构造，对输入多项式做约化并实现快速组合。

**💡 创新点**

创新点在于：
- 通过构造两个更小维度的关系矩阵（而非传统单一大矩阵）显著降低了矩阵运算规模；
- 结合截断幂与二元模合成的等价性，统一处理截断乘积与多项式矩阵乘法；
- 在泛型输入（Zariski 开放集）下给出可检测的算法，首次实现从 O~(n^{1.43}) 降到 O~(n^{1.343}) 的代数复杂度。

**🔧 技术方法**

使用的核心技术包括：
- 多项式矩阵乘法与快速矩阵乘法（乘法指数 ω）；
- 最小基（Popov 形）与关系模块（[y]-模块和 [x]-模块）的计算；
- 逆克罗内克替换、截断幂与高次剩余的快速求解；
- 转置原理与线性映射的等价性，用于将截断幂问题映射到二元模合成。

**📊 数据集**

该工作为纯算法研究，不使用任何实验数据集；所有结果均以理论复杂度证明为主。

**📈 对比分析**

与之前最优的 Brent‑Kung 算法（O~(n^{1.43})）和 Kinoshita‑Li、Nüsken‑Ziegler 等算法比较，本文在泛型情况下实现了 
- 代数复杂度下降到 O~(n^{(ω+3)/4})；
- 采用最佳已知 ω ≈ 2.373 时，得到 O~(n^{1.343})，显著优于先前 O~(n^{1.43})。

**⚠️ 局限性**

限制与挑战：
- 需要输入满足泛型性（Zariski 开放集），非泛型输入会失败，需要回退到其他算法；
- 算法为确定性（Las Vegas）尚未实现；
- 仍依赖于高效的矩阵乘法实现，实际常数可能较大。

---

## 225. Minimizing Completion Times of Stochastic Jobs on Parallel Machines is Hard

**arXiv ID:** 2601.17425 | [PDF](https://arxiv.org/pdf/2601.17425v1)

**作者:** Benjamin Moseley `[一作]` (Carnegie Mellon University), Rudy Zhou `[通讯]` (Microsoft)

**通讯引用:** 61 | [OpenAlex ID](https://openalex.org/A5090909905)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了在无优先级约束、处理时间为两点分布、单位或相同权重的平行机器随机调度问题中，计算最优调度策略或其期望总完成时间在多项式时间内不可行。

**💡 创新点**

首次给出了该类问题的 #P 难度证明，突破了以往仅在更严格约束或特定分布下的难度结论。

**🔧 技术方法**

采用了从 Knapsack 计数问题的约化，构造了两类调度实例，并利用 SEPT / WSEPT 排序规则在不同实例中的期望成本差异来推导难度。

**📊 数据集**

本文不使用实际数据集，而是使用理论构造的随机调度实例进行证明。

**📈 对比分析**

没有实验对比或性能评估，所给结论仅说明在理论上该决策/评估问题属于 #P 难度，无法用现有算法高效求解。

**⚠️ 局限性**

局限在于仅证明了期望成本评估与决策问题的难度，未证明最优策略本身不可计算，也未提供近似或可行的算法框架。

---

## 226. CoT-Seg: Rethinking Segmentation with Chain-of-Thought Reasoning and Self-Correction

**arXiv ID:** 2601.17420 | [PDF](https://arxiv.org/pdf/2601.17420v1)

**作者:** Shiu-hong Kao `[一作]` (Hong Kong University of Science and Technology), Chi-Keung Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 13331 | [OpenAlex ID](https://openalex.org/A5062566088)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了CoT‑Seg，一种零训练的推理分割框架，利用链式思维生成元查询、自动自校正以及检索增强，实现对复杂视觉语言查询的逐步分割。

**💡 创新点**

创新点包括：① 将链式思维与自校正机制首次结合到分割任务；② 采用多代理协作（Reasoner、Segmentor、Evaluator）并通过JSON结构化交互；③ 支持多模态控制输入；④ 引入检索增强推理；⑤ 设计新数据集ReasonSeg‑Hard，用于评测极难案例。

**🔧 技术方法**

技术方法：预训练多模态大语言模型（如 GPT‑4o、Vision‑Reasoner‑7B）、SAM‑HQ 及其变体做分割基底；链式思维生成逐步问题与答案；自校正循环评估并生成正负修正指令；JSON格式统一通信；检索增强通过外部知识检索填补信息缺口。

**📊 数据集**

使用数据集：ReasonSeg、ReasonSeg‑Hard（新建）以及 RefCOCO 作为基准。

**📈 对比分析**

对比方法：在 ReasonSeg 与 ReasonSeg‑Hard 上进行零训练评估，指标为 gIoU 和 cIoU。CoT‑Seg 在 ReasonSeg‑Hard 上实现或逼近 SOTA，显著提升复杂或知识需求高的场景；在 RefCOCO 上提升有限，表明其优势主要体现在需要深层推理的任务。

**⚠️ 局限性**

局限性：① 推理时间显著增加，主要受 GPT 在线 API 调用成本影响；② 对分割器的兼容性有较高要求，若分割器缺乏相应输入能力会导致失败；③ 多轮自校正不一定总能改进，过度推理可能导致精度下降。

---

## 227. Cloud-Enabled IoT System for Real-Time Environmental Monitoring and Remote Device Control Using Firebase

**arXiv ID:** 2601.17414 | [PDF](https://arxiv.org/pdf/2601.17414v1)

**作者:** Abdul Hasib `[一作]` (University of Frontier Technology), A. S. M. Ahsanul Sarkar Akib `[通讯]` (Robo Tech Valley)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了基于Firebase实时数据库的云端物联网系统，实现了ESP32与DHT22、HC‑SR04传感器的数据采集和LED执行器的远程控制，提供了实时监测与双向控制的完整框架

**💡 创新点**

首次完整展示Firebase的双向通信能力，将多种传感器数据与云端同步，并通过云端即刻控制物理执行器，填补了传统单向监测或控制的空白；同时提出了低成本、易部署的架构

**🔧 技术方法**

ESP32微控制器、DHT22温湿度传感器、HC‑SR04超声波距离传感器、Firebase Realtime Database、Wi‑Fi通信、Web/移动界面、C/C++固件、JavaScript前端、JSON数据格式

**📊 数据集**

收集的环境传感器实时数据（温度、湿度、距离）以及LED状态控制指令，持续14天共计约120万条数据用于评估

**📈 对比分析**

与AWS IoT Core、MQTT代理、HTTP REST API等云方案对比，使用平均延迟、部署复杂度、成本等指标；实验显示本方案平均控制延迟1.4 s、数据传输成功率99.2%，成本低于竞争方案

**⚠️ 局限性**

安全规则需手动配置、易受配置错误影响；层次化数据库结构限制复杂数据关系；离线功能有限；大规模部署成本随数据量激增；对Google生态的锁定

---

## 228. Source-Free Domain Adaptation by Optimizing Batch-Wise Cosine Similarity

**arXiv ID:** 2601.17408 | [PDF](https://arxiv.org/pdf/2601.17408v1)

**作者:** Harsharaj Pathak `[一作]` (Indian Institute of Technology Hyderabad), Vineeth N Balasubramanian `[通讯]` (Indian Institute of Technology Hyderabad)

**通讯引用:** 6142 | [OpenAlex ID](https://openalex.org/A5038020125)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在源自由域适应任务中，作者提出利用邻域签名对目标域样本的预测进行余弦相似度对齐，以实现模型迁移。

**💡 创新点**

创新点在于将邻域签名作为语义相似度度量，结合类内多样性、预测惯性和类别不平衡的加权，形成单一损失函数。

**🔧 技术方法**

技术手段包括邻域签名计算、置信度自适应类编码、余弦相似度对齐损失，以及基于最近邻的内存池。

**📊 数据集**

使用了PACS、Office‑31和VisDA‑2017三个图像分类基准数据集。

**📈 对比分析**

与SHOT、HCL、G‑SFDA、NRC等现有SFDA方法对比，方法在VisDA上获得最高平均准确率，在Office‑31和PACS上表现相当或优于对手。

**⚠️ 局限性**

局限性包括对邻域近邻选择和衰减因子敏感，且在极端类别不平衡或噪声邻域多样性高的场景下效果可能下降。

---

## 229. Res-MIA: A Training-Free Resolution-Based Membership Inference Attack on Federated Learning Models

**arXiv ID:** 2601.17378 | [PDF](https://arxiv.org/pdf/2601.17378v1)

**作者:** Mohammad Zare `[一作]` (Shiraz University of Technology), Pirooz Shamsinejadbabaki `[通讯]` (Shiraz University of Technology)

**通讯引用:** 189 | [OpenAlex ID](https://openalex.org/A5043461810)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无需训练、仅通过黑盒查询的分辨率递归成员推断攻击Res-MIA

**💡 创新点**

利用模型对高频细节的敏感性，构造分辨率降解序列并以置信度衰减作为会员信号，首次将频率偏差与成员推断关联

**🔧 技术方法**

分辨率逐步降低（平均池化）+最近邻上采样，计算置信度衰减得分；仅需多次前向推理，无需梯度或阴影模型

**📊 数据集**

在联邦学习环境下使用CIFAR-10数据集训练ResNet-18（10个客户端）进行评估

**📈 对比分析**

与传统一次性置信度/熵基线对比，Res-MIA在AUC、准确率和FPR@TPR=80%上显著提升（AUC 0.88、准确率 0.81、FPR 0.19）

**⚠️ 局限性**

需要多次查询（K+1次），对高分辨率或非图像数据适用性有限，且对噪声或下采样方式敏感

---

## 230. Breaking Flat: A Generalised Query Performance Prediction Evaluation Framework

**arXiv ID:** 2601.17359 | [PDF](https://arxiv.org/pdf/2601.17359v1)

**作者:** Payel Santra `[一作]` (Indian Association for the Cultivation of Science), Debasis Ganguly `[通讯]` (University of Glasgow)

**通讯引用:** 2048 | [OpenAlex ID](https://openalex.org/A5082339849)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在论文中，作者提出了一个二维的查询性能预测（QPP）评估框架，扩展了传统单维（单一检索模型对多查询）的QPP评估，加入了检索模型维度，形成三种评估设置：单模型多查询、单查询多模型、以及全模型全查询。

**💡 创新点**

创新点在于：①首次将QPP评估从单一模型扩展到多模型的二维视角；②定义了三种新评估任务，并给出对应的评价指标；③对比13种不同QPP方法在这三种任务中的表现，揭示传统无监督方法在跨模型预测中表现优于监督方法，而在预测最佳模型时监督方法表现更佳。

**🔧 技术方法**

采用的技术包括传统的无监督QPP方法（NQC、WIG、σ_max等）、基于特征的监督回归方法（NQA‑QPP、BERTQPP）以及基于密集向量的无监督方法（DM）。评估指标主要使用Kendall’s τ相关系数，并对不同任务进行统计显著性检验（配对t检验）。

**📊 数据集**

实验数据集为TREC Deep Learning 2019 与 2020 的97个查询，检索集合为 MS MARCO passage collection（约880万段落）。

**📈 对比分析**

比较方法：在三种评估设置下计算各模型的平均Kendall’s τ值，并与传统评估结果对比；使用统计显著性检验评估差异。结果显示：①在跨模型预测（即单查询多模型）任务中，监督方法BERTQPP与无监督DM获得最高相关系数；②在单模型多查询任务中，无监督方法NQC、RSD表现最好；③整体二维评估中，BERTQPP和DM仍保持领先。

**⚠️ 局限性**

局限性包括：①目前仅评估已有的QPP方法，未提出专门针对多模型预测的新方法；②实验仅覆盖TREC DL 19/20数据集，缺乏对更大规模或不同领域的验证；③在多模型预测任务中，任务难度大，相关系数相对较低，表明现有方法仍需改进。

---

## 231. NeRF-MIR: Towards High-Quality Restoration of Masked Images with Neural Radiance Fields

**arXiv ID:** 2601.17350 | [PDF](https://arxiv.org/pdf/2601.17350v1)

**作者:** Xianliang Huang `[一作]` (Fudan University), Shuigeng Zhou `[通讯]` (Fudan University)

**通讯引用:** 10965 | [OpenAlex ID](https://openalex.org/A5017862559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种名为 NeRF-MIR 的基于 NeRF 的多视角图像掩膜恢复方法。

**💡 创新点**

创新点包括基于图像熵的分块射线发射策略（PERE）、逐步迭代自训练恢复（PIRE）以及动态加权损失函数，用于自适应关注掩膜区域。

**🔧 技术方法**

技术手段主要是神经辐射场（NeRF）、基于熵的射线采样、逐步迭代自训练、动态权重损失，并结合 MLP 和体渲染。

**📊 数据集**

使用了三套自建掩膜数据集（LLFF-M、Spaces-M、Blender-M），以及从 LLFF、Spaces、Blender 原始数据集和真实 iPhone 12 捕获场景（含雪花、落叶掩膜）构造的实验集。

**📈 对比分析**

通过与 2D 填补基线（HiFill、ZITS++ 等）和 NeRF 相关方法（Masked NeRF、Inpainting-NeRF、NeRF-In、SPIn-NeRF、NeRF-On-the-Go）进行对比实验；在 PSNR、SSIM 和 LPIPS 指标上，NeRF-MIR 在合成与真实数据集均显著优于对照组，提升幅度可达 10‑15%。

**⚠️ 局限性**

局限性包括对大面积连续掩膜区域恢复效果差、依赖外部目标检测器生成掩膜、以及由于逐步迭代和射线重采样导致的计算开销较高。

---

## 232. Multi-Agent Learning Path Planning via LLMs

**arXiv ID:** 2601.17346 | [PDF](https://arxiv.org/pdf/2601.17346v1)

**作者:** Haoxin Xu `[一作]` (Shanghai International Studies University), Xiaoqing Gu `[通讯]` (East China Normal University)

**通讯引用:** 2613 | [OpenAlex ID](https://openalex.org/A5088470075)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于多智能体的可解释学习路径规划框架MALPP，利用LLM生成个性化、可解释的学习路径。

**💡 创新点**

创新点在于将角色与规则协作的多智能体结构与认知负荷理论（CLT）和最近发展区（ZPD）约束嵌入LLM提示，实现既可解释又符合教学理论的路径规划。

**🔧 技术方法**

技术包括大语言模型（如XXNU-Plus）、知识图谱、知识跟踪、学术风险预测、学习资源推荐、角色与规则驱动的多智能体协作、以及CLT/ZPD约束模块。

**📊 数据集**

使用MOOC数据集MOOCCubeX中的六门课程子集（共1453名学习者、428个视频、209个练习、446个知识点）。

**📈 对比分析**

与随机基线（RBM）和单一LLM方案（SLMLPP）对比，MALPP在知识序列一致性（KSC）和认知负荷失配率（CLMR）上显著优于基线，平均路径长度和学习时长与基线相当。

**⚠️ 局限性**

局限包括：需要丰富的学习者交互数据，计算成本高（多代理与LLM交互产生大量token），未考虑多模态资源，且在数据稀疏或低质量场景下效果可能受限。

---

## 233. Oops, Wait: Token-Level Signals as a Lens into LLM Reasoning

**arXiv ID:** 2601.17421 | [PDF](https://arxiv.org/pdf/2601.17421v1)

**作者:** Jaehui Hwang `[一作]` (NAVER AI Lab), Byeongho Heo `[通讯]` (NAVER AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析大语言模型（LLM）推理过程中的词级信号，探究了不同训练策略和模型规模对推理行为的影响。

**💡 创新点**

创新点在于：①系统性量化词级信号（尤其是对话标记如"wait"、"therefore"等）与推理正确性的关联；②揭示训练策略而非模型规模主导词级信号；③提出基于词级信号的抑制、集成与后训练改进方法。

**🔧 技术方法**

主要技术包括：token概率计算（在

后取softmax概率平均）、t检验差异显著性检验、相关系数分析、概率跳跃检测与"wait"定位、词级抑制与基于词级差值的集成。

**📊 数据集**

使用的推理基准包括AIME24、GPQA‑D、MATH‑500等自然语言推理数据集，以及训练时使用的DeepSeek‑R1轨迹、s1‑1.1小规模SFT数据集。

**📈 对比分析**

实验比较了不同模型（DeepSeek‑R1‑distill‑Qwen‑32B、QwQ‑32B、s1‑1.1‑32B等）在各基准上的准确率，并评估了词级抑制、基于词级差值的集成等方法。结果显示：词级抑制对错误词的抑制会导致性能下降，基于词级差值的集成在AIME24上取得最高的准确率，表明词级信号能有效提升性能。

**⚠️ 局限性**

限制在于：①只在开源Qwen系列模型上验证，未覆盖LLaMA、Mixtral等架构；②方法依赖softmax输出和完整推理轨迹，难以用于闭源模型；③仅关注自然语言推理任务，代码生成等场景下的词级信号未知。

---

## 234. When AI Agents Touch CI/CD Configurations: Frequency and Success

**arXiv ID:** 2601.17413 | [PDF](https://arxiv.org/pdf/2601.17413v1)

**作者:** Taher A. Ghaleb `[一作]` `[通讯]` (Trent University), Taher A. Ghaleb (Trent University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析了 8,031 个由 5 种 AI 助手生成的 PR，评估它们对 YAML CI/CD 配置文件的修改频率、合并率和构建成功率。

**💡 创新点**

首次量化 AI 助手在 CI/CD 配置方面的参与度与质量，并揭示 Copilot 在 CI/CD 修改上的特殊优势和平台偏好。

**🔧 技术方法**

使用 GitHub GraphQL API 提取 PR 与工作流运行数据，利用正则表达式识别 CI/CD YAML 文件，并通过卡方检验和统计比较评估差异。

**📊 数据集**

采用 AIDev-pop 数据集（33,596 PR）以及补充的 GitHub 工作流运行记录（99,930 次）进行实验。

**📈 对比分析**

通过对比 CI/CD 修改与非修改 PR 的合并率和构建成功率，发现总体 CI/CD 修改的合并率略低，但 Copilot 等代理在 CI/CD 上更高；构建成功率与非修改相近或略高，三大代理显著提升。

**⚠️ 局限性**

结果受限于仅分析公开 GitHub PR，未覆盖被拒绝或未提交的建议；平台检测仅基于文件名正则，可能漏检；代理归属推断不完全精确，时间演进与选择偏差也影响结论。

---

## 235. Efficient Dilated Squeeze and Excitation Neural Operator for Differential Equations

**arXiv ID:** 2601.17407 | [PDF](https://arxiv.org/pdf/2601.17407v1)

**作者:** Prajwal Chauhan `[一作]` (New York University Abu Dhabi), Saif Eddin Jabari `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 1326 | [OpenAlex ID](https://openalex.org/A5061466757)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出轻量级神经算子 D-SENO，用膨胀卷积和 SE 模块替代 Transformer/FNO，保持局部性，显著提升速度并保持/超过精度。

**💡 创新点**

创新点是将数据集特定膨胀率与通道注意力结合，构建多尺度卷积层且不使用全局 FFT 或自注意，形成高效、可扩展的算子。

**🔧 技术方法**

技术包括膨胀卷积（Atrous）、Squeeze‑Excitation（SE）模块、轻量级点卷积、上升‑处理‑投影结构、残差连接、GELU 激活。

**📊 数据集**

使用的公开数据集包括：NACA‑0012 气动潜流（221×51）、Poiseuille 管流（129×129）、Darcy 多孔介质（85×85）、Navier–Stokes 圆环（64×64×20）。

**📈 对比分析**

与 FNO、U‑NO、Transolver 等基线对比，D‑SENO 在 3/4 任务上精度最优或相当，训练/推理速度比 Transolver 低约 20–300 倍（≈ 1–7 秒/epoch）。

**⚠️ 局限性**

局限：在周期性 Navier‑Stokes 任务上仍落后，未覆盖非结构化网格，膨胀率需手工设定，缺乏理论分析。

---

## 236. GO-OSC and VASH: Geometry-Aware Representation Learning for Early Degradation Detection in Oscillatory Systems

**arXiv ID:** 2601.17396 | [PDF](https://arxiv.org/pdf/2601.17396v1)

**作者:** Vashista Nobaub `[一作]` `[通讯]` (Datar Consulting), Vashista Nobaub (Datar Consulting)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了GO-OSC框架，利用可识别的振荡状态空间模型和Real–Schur规范化，构造了几何感知的时序表示，并基于此设计了一系列线性几何探测器，用于从短窗口的无标签数据中早期检测振荡系统的衰变；

**💡 创新点**

创新点包括：①在振荡时序上引入可识别的Canonical Real–Schur表示，消除相似变换歧义；②设计了针对衰变相关方向的线性几何探测器；③通过局部渐近正态性（LAN）理论证明能量统计在早期相位衰变下无检验力，而几何探测器具有正的检验效率；④实验证明该方法在早期检测、数据效率和鲁棒性方面优于传统能量指标和无约束自监督表示；

**🔧 技术方法**

技术手段包括：基于线性高斯状态空间模型的Kalman滤波与平滑；Real–Schur形式的相位频率规范化；线性几何探测器（如GSI、PCC、FWR等）的构造；局部渐近正态性理论与Pitman效率分析；以及与能量统计、TS2Vec、CPC等基线方法的对比实验；

**📊 数据集**

实验使用了合成振荡信号基准以及真实工业振动数据集（如CWRU或类似的振动数据），在这些数据上验证了模型的可行性与鲁棒性；

**📈 对比分析**

与传统能量基指标（RMS、功率谱）及无约束自监督表示进行比较。GO-OSC+线性几何探测器在早期衰变检测上实现了更高的AUROC（近似1.0），在数据效率上相较能量指标提升约16倍，并在强噪声或幅度扰动场景下保持高鲁棒性；

**⚠️ 局限性**

局限性在于：①要求信号可被线性高斯状态空间模型近似、振荡模式分离且局部平稳；②对强非线性耦合、幅度突变或大幅度噪声的情形表现不佳；③主要针对相位仅变化的早期衰变，无法直接处理幅度驱动或混合故障模式。

---

## 237. Do readers prefer AI-generated Italian short stories?

**arXiv ID:** 2601.17363 | [PDF](https://arxiv.org/pdf/2601.17363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 238. Collab: Fostering Critical Identification of Deepfake Videos on Social Media via Synergistic Annotation

**arXiv ID:** 2601.17371 | [PDF](https://arxiv.org/pdf/2601.17371v1)

**作者:** Shuning Zhang `[一作]` (Tsinghua University), Hewu Li `[通讯]` (Tsinghua University)

**通讯引用:** 1373 | [OpenAlex ID](https://openalex.org/A5011803365)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 Collab，一个协同深度伪造视频识别插件，通过用户标注、置信度加权聚合和分层展示提高识别准确率。

**💡 创新点**

创新点在于将深度伪造视频的时空标注与置信度加权 3D IoU 聚合相结合，并采用分层可视化避免社会影响，提升用户批判性。

**🔧 技术方法**

技术包括基于 Web 的交互式时空标注界面、置信度加权 3D Intersection-over-Union 聚合算法、语言模型文本聚合、K‑Means 文本聚类以及颜色编码可视化等。

**📊 数据集**

使用了 FaceForensics++、BioDeepAV、DFW、DDL 四大公开深伪数据集，共 240 条视频（各 30 条真实与 30 条伪造）。

**📈 对比分析**

对比实验中，Collab 的 F1 分数为 0.883，显著高于无聚合（0.849）和无标签（0.795）两种对照；准确率提升约 9%。

**⚠️ 局限性**

局限包括实验环境为模拟社交平台、样本比例平衡且需人工标注提示，缺乏真实生态验证及对抗性鲁棒性研究。

---

## 239. A Scoping Review and Guidelines on Privacy Policy's Visualization from an HCI Perspective

**arXiv ID:** 2601.17368 | [PDF](https://arxiv.org/pdf/2601.17368v1)

**作者:** Shuning Zhang `[一作]` (Tsinghua University), Hewu Li `[通讯]` (Tsinghua University)

**通讯引用:** 1373 | [OpenAlex ID](https://openalex.org/A5011803365)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对65篇顶级会议论文进行系统综述，构建基于设计生命周期的框架，梳理隐私政策可视化的四个演化模式并给出研究指南。

**💡 创新点**

首次从时间演进视角出发，结合设计生命周期四阶段（语境、需求、设计、评估）系统化分析隐私政策可视化的历史发展，并提出四大关键挑战与对应设计准则。

**🔧 技术方法**

使用系统性文献检索（PRISMA）和混合编码（归纳+演绎）技术，对论文进行主题编码和主题综合；未涉及具体算法实现。

**📊 数据集**

使用从ACM、IEEE等数字图书馆检索到的65篇论文作为研究数据；未使用公开数据集。

**📈 对比分析**

本研究为综述性质，无性能比较；通过定性分析与案例归纳阐释演化趋势，未给出量化指标。

**⚠️ 局限性**

局限在于仅覆盖顶级会议与近二十年文献，筛选与编码可能存在主观偏差，缺乏实证验证与跨学科多元视角。

---

## 240. UCAD: Uncertainty-guided Contour-aware Displacement for semi-supervised medical image segmentation

**arXiv ID:** 2601.17366 | [PDF](https://arxiv.org/pdf/2601.17366v1)

**作者:** Chengbo Ding `[一作]` (University of Science and Technology of China), Shaohua Kevin Zhou `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8749 | [OpenAlex ID](https://openalex.org/A5028465673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于不确定性引导的轮廓感知位移（UCAD）框架，用于半监督医学图像分割，能够在保持解剖边界完整性的同时提升一致性学习效果。

**💡 创新点**

创新点包括：① 使用 SLIC 超像素实现轮廓感知区域划分，避免传统矩形位移导致的结构畸变；② 通过像素熵估计挑选不确定区域进行位移，聚焦难以学习的边界；③ 引入动态不确定性加权一致性损失，根据信任度自适应抑制伪标签误差，提升训练稳定性。

**🔧 技术方法**

技术手段：Mean Teacher 训练框架、SLIC 超像素分割、熵值不确定性估计、Dice+交叉熵混合监督损失、动态不确定性加权一致性损失。

**📊 数据集**

实验数据集：ACDC 心脏 MRI 数据集和 Synapse 多器官 CT 数据集，分别评估 5% 与 10% 标注比例下的性能。

**📈 对比分析**

与多种最先进的半监督分割方法（UA-MT、SASSNet、DTC、URPC、MC-Net、SS-Net、MCF、BCP、CML、ABD）进行对比，UCAD 在两大数据集上均取得最高 Dice 分数和最低 ASD，示例结果包括 ACDC 5% 标注下 88.63% DSC/0.49 ASD、10% 标注下 89.93% DSC/0.57 ASD，Synapse 10% 标注下 66.73% DSC/29.62 ASD。

**⚠️ 局限性**

局限性：① 依赖超像素划分质量，若超像素不充分对齐解剖边界会影响效果；② 主要在 2D 切片上验证，未充分探索 3D 体卷的适用性；③ 计算成本相对传统矩形位移更高，且需要额外的熵估计与动态加权计算；④ 对极少标注（低于 5%）的鲁棒性尚未深入评估。

---

## 241. Revisiting Lightweight Low-Light Image Enhancement: From a YUV Color Space Perspective

**arXiv ID:** 2601.17349 | [PDF](https://arxiv.org/pdf/2601.17349v1)

**作者:** Hailong Yan `[一作]` (University of Electronic Science and Technology of China), Bo Li `[通讯]` (vivo Mobile Communication Company)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究轻量级低照度图像增强，提出基于YUV色空间的三模块框架；

**💡 创新点**

创新点在于通过频域分析辨别Y通道低频失真、UV通道高频噪声，设计DSGLA、LAFA、GI三模块实现通道特定处理与跨通道交互，参数仅30K；

**🔧 技术方法**

使用的技术包括YUV转换、双流全局-局部注意力（DDSA+GGA）、亮度引导的频域注意力（LAFA）、引导交互（GI）、傅里叶变换和深度可分离卷积等；

**📊 数据集**

实验使用LOLv1/LOLv2/LSRW进行训练，评估涵盖LOL、LSRW以及五个无配对数据集DICM、LIME、MEF、NPE、VV；

**📈 对比分析**

与多种SOTA方法对比，在PSNR/SSIM/LPIPS上获得最高或第二高分，参数量与FLOPs最低，GPU/CPU延迟仅6.5/124 ms；

**⚠️ 局限性**

局限在于未探究其他色彩空间的可能性，尚不确定YUV是否绝对最优，并需进一步验证对极端低光场景的鲁棒性。

---

## 242. Auditing Disability Representation in Vision-Language Models

**arXiv ID:** 2601.17348 | [PDF](https://arxiv.org/pdf/2601.17348v1)

**作者:** Srikant Panda `[一作]` (Lam Research), Palkesh Malviya `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过对比中性提示与含残障提示下的视觉语言模型生成的描述，系统评估了残障相关的解释性偏差，提出了基于解释性忠实度的评估框架并在15种VLM上进行基准测试。

**💡 创新点**

创新点包括：①使用配对提示（NP/DP）方法与LLM‑as‑judge相结合，量化残障上下文对模型输出的解释性漂移；②构建包含九类残障的评估税表，并对性别与种族交叉影响进行细致分析；③提出针对性提示与偏好优化两种可行的消除偏差方案。

**🔧 技术方法**

技术手段包括：零射提示生成、文本情感与尊重度量（VADER、Regard）、长度指标、LLM‑as‑judge评估、直接偏好优化（DPO）以及人工标注验证。

**📊 数据集**

使用的数据集主要是PAIRS合成图像集（覆盖多种职业与活动）以及COCO用于偏好学习；在PAIRS中为每张图生成一对NP/DP，形成2000条描述。

**📈 对比分析**

与传统单轴偏差评测相比，本文的评估在解释性维度上更细粒度，实验显示在九类残障中多数模型在DP下的解释性漂移、情感偏低和冗长度均显著上升；通过定向提示或DPO可将解释性漂移下降30‑60%，框架与刻板偏差几近消失。

**⚠️ 局限性**

局限性包括：仅使用合成图像，未检验真实图像；评测仅覆盖零射描述任务；LLM‑judge与人工标注存在一致性不足；消除策略仅在有限模型上验证，缺乏对大规模部署的通用性验证。

---

## 243. Efficient Self-Learning and Model Versioning for AI-native O-RAN Edge

**arXiv ID:** 2601.17534 | [PDF](https://arxiv.org/pdf/2601.17534v1)

**作者:** Mounir Bensalem `[一作]` (Technische Universitaet Braunschweig), Jenq-Shiou Leu `[通讯]` (National Taiwan University of Science and Technology)

**通讯引用:** 2431 | [OpenAlex ID](https://openalex.org/A5059627549)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种面向 AI‑native 6G O‑RAN 边缘的自学习模型版本管理框架，将云端训练与多层边缘推理统一起来，利用强化学习实现基于遥测的动态版本升级决策。

**💡 创新点**

创新点在于首次构建了完整的端到端 MLOps 体系，兼顾了 O‑RAN 的三种控制循环（实时、近实时、非实时）以及多层云/边缘架构，并通过 Q‑learning 自动平衡模型精度、稳定性、延迟与资源利用，解决了传统手工或启发式升级难以规模化的问题。

**🔧 技术方法**

采用了 Q‑learning 强化学习、容器编排（Kubernetes）、遥测与知识库、共享版本仓库等技术，实现了模型版本的自动选择、部署与资源调度。

**📊 数据集**

实验使用了六个自定义的模拟 ML 模型（含版本、准确率、稳定性等属性）以及基于负指数分布的请求流，数据为仿真生成的遥测与性能指标。

**📈 对比分析**

与四种基线策略（始终升级、永不升级、随机升级、基于服务器负载升级）对比，RL 策略在 dApp 的延迟最小化、xApp 与 rApp 的准确率提升以及整体系统稳定性方面均优于其他方案，表现出更好的网络效用和 SLA 合规性。

**⚠️ 局限性**

局限性包括仅在仿真环境中验证，缺乏真实基站或 O‑RAN 设备的实测；安全性、奖励函数的设计仍待进一步细化；以及对 RL 超参数的敏感性需在更大规模场景下评估。

---

## 244. Revealing the Truth with ConLLM for Detecting Multi-Modal Deepfakes

**arXiv ID:** 2601.17530 | [PDF](https://arxiv.org/pdf/2601.17530v1)

**作者:** Gautam Siddharth Kashyap `[一作]` (Macquarie University), Usman Naseem `[通讯]` (Macquarie University)

**通讯引用:** 4113 | [OpenAlex ID](https://openalex.org/A5077006200)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了一种双阶段的ConLLM框架，结合预训练模型提取模态特征、对比学习对齐和LLM推理，用于多模态深度伪造检测。

**💡 创新点**

创新点在于将对比学习与大型语言模型融合，解决模态碎片化和浅层跨模态推理问题，并实现语义一致性细粒度检测。

**🔧 技术方法**

采用XLS‑R、VideoMAE、VATLM等预训练模型提取特征，使用对比学习损失进行跨模态对齐，并用GPT风格LLM进行语义细化与分类。

**📊 数据集**

在ASVSpoof 2019、DECRO、Celeb‑DF、WildDeepfake、FakeAVCeleb和DFDC等六个公开数据集上进行实验。

**📈 对比分析**

与现有单模态和多模态基线相比，ConLLM在音频EER降低至0.21%（比对比模型低≈50%），视频准确率提升至98.75%（+8%），音频‑视频准确率提升至96.5%（+9%），整体性能显著优于State‑of‑the‑Art。

**⚠️ 局限性**

局限性包括对训练数据多样性和质量的高度依赖、训练与推理计算成本较高，以及在更广泛的真实世界数据集上的验证不足。

---

## 245. Invited: Toward Sustainable and Transparent Benchmarking for Academic Physical Design Research

**arXiv ID:** 2601.17520 | [PDF](https://arxiv.org/pdf/2601.17520v1)

**作者:** Liwen Jiang `[一作]` (Fudan University), Zhiyu Zheng `[通讯]` (Fudan University)

**通讯引用:** 1664 | [OpenAlex ID](https://openalex.org/A5027956538)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了RosettaStone 2.0——一个面向2D与Pin‑3D（面‑面）3D物理设计的可维护、可复现的开源基准框架，并在此框架下发布了完整的RTL‑to‑GDS参考流程；

**💡 创新点**

创新点在于：1）与OpenROAD‑Research及ORFS‑Research深度集成，形成可持续、CI驱动的评测后端；2）提供统一的METRICS2.1日志与排行榜治理；3）首次公开Pin‑3D混合绑定（HBT）3D RTL‑to‑GDS参考流，包含3D启用、分层约束与阶段性检查；4）构建RosettaStone 2.0路线图，支持学术基准翻译、合成数据生成与统一评测；

**🔧 技术方法**

技术包括：OpenROAD‑Research/ORFS‑Research脚本、OpenDB API、TritonPart、TritonRoute、TritonCTS、仿真提取、METRICS2.1规范、CI/CD、DCO治理、HBT模型与层栈、跨层电源网络、迭代多层置换与合法化；

**📊 数据集**

使用的数据集：传统Bookshelf竞赛基准（aes、ibex、jpeg）、ASAP7/NanGate45 PDK、HBT几何参数；以及通过ArtNet生成的可扩展合成网表；

**📈 对比分析**

比较方法：在相同评测契约（核心利用率、时钟周期、PDN、约束等）下，对OpenROAD‑Research（ORD）与匿名商用参考流（COMM）在三种基准与多种技术堆叠（2D、7+7、45+45、7+45）下收集runtime、wirelength、时序（WNS/TNS）、功耗、DRV/FEP等指标；实验显示3D流在routing、HBT使用和DRV方面相对更具挑战，ORFS在前期步骤更快但routing慢；不同工具链组合（Mixed1‑Mixed4）揭示合成与后端对QoR影响；

**⚠️ 局限性**

limitations：1）目前3D路由依赖2D路由器，速度慢且DRV多；2）HBT几何与尺寸对DRV敏感，需进一步优化模型；3）只提供PIN‑3D混合绑定，缺少TSV、M3D等其他3D风格；4）商用流未公开，无法直接进行同等水平对比；5）实验环境单一，缺乏跨平台验证；6）基准规模有限，未覆盖更大规模真实设计。

---

## 246. SpatialMath: Spatial Comprehension-Infused Symbolic Reasoning for Mathematical Problem-Solving

**arXiv ID:** 2601.17489 | [PDF](https://arxiv.org/pdf/2601.17489v1)

**作者:** Ashutosh Bajpai `[一作]` (Indian Institute of Technology Delhi), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology Abu Dhabi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于空间理解的中型多模态语言模型框架，并构建了扩展版的 MathVerse 数据集。

**💡 创新点**

将视觉空间描述与结构化推理链融合，改进了视觉推理与符号推理的对齐，并引入评估模块提升输出质量。

**🔧 技术方法**

使用 LoRA 等 PEFT 对 LLaVA-NeXT-34B / Phi-4 等模型进行空间理解与推理链微调，并利用 GPT‑4o 生成空间描述和推理步骤。

**📊 数据集**

扩展的 MathVerse（Geo 版）数据集，共 2,760 条训练样例和 500 条测试样例，覆盖 5 种视觉/文本融合设置。

**📈 对比分析**

相较于零-shot、ICL、CoT、SFT 以及 SFT+数据增强基线，在 vision‑intensive 与 vision‑only 等视觉密集场景提升 10%+平均 3.8% 的准确率，并在多种中型多模态模型上均表现出正向提升。

**⚠️ 局限性**

模型性能依赖空间理解模块的质量，误识别会导致错误；数据集仅包含英文，缺少低资源语言与非几何领域的扩展。

---

## 247. Automatic Stability and Recovery for Neural Network Training

**arXiv ID:** 2601.17483 | [PDF](https://arxiv.org/pdf/2601.17483v1)

**作者:** Barak Or `[一作]` `[通讯]` (Google and Reichman Tech School), Barak Or (Google and Reichman Tech School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个训练时的稳定性控制器，在训练过程中监测优化器提出的更新，并在检测到不稳定时自动回滚到最近安全状态。

**💡 创新点**

创新点在于把训练视为受控随机过程，利用外部创新信号（如验证探针）进行安全检测，提供理论上的运行时安全性和恢复保证，而不需要改动优化器。

**🔧 技术方法**

采用控制理论框架、创新信号检测、验证探针评估、回滚恢复机制，实验中使用AdamW等标准优化器。

**📊 数据集**

使用CIFAR‑10数据集训练ResNet‑18，并在字符级Transformer上使用合成文本语料进行实验。

**📈 对比分析**

通过在训练中注入梯度放大故障，比较控制器与标准训练的峰值退化、恢复速度和参数范数，结果表明控制器显著降低峰值损失、加速恢复、保持参数范数稳定，平均表现优于基线。

**⚠️ 局限性**

局限性包括：缺乏收敛/最优性保证；阈值需人工设定；存在额外的测量开销；仅实现回滚动作，未考虑更丰富的控制策略；未在分布式/异步训练环境中验证。

---

## 248. Lattice: Generative Guardrails for Conversational Agents

**arXiv ID:** 2601.17481 | [PDF](https://arxiv.org/pdf/2601.17481v1)

**作者:** Emily Broadhurst `[一作]` (Distyl AI), Karime Maamari `[通讯]` (Distyl AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并持续改进对话式AI的安全防护规则（guardrail）

**💡 创新点**

通过模拟对话和迭代优化自动生成规则，随后使用风险评估、对抗测试与闭环优化在部署后自动演化规则，无需人工干预

**🔧 技术方法**

大语言模型提示式推理、对话模拟、性能评估、规则编辑与聚类、对抗beam搜索、双重检查风险评估、闭环优化循环

**📊 数据集**

ProsocialDialog 多轮对话安全标注数据集（100条训练、652条测试、100条跨域改进）

**📈 对比分析**

与关键词匹配、LlamaGuard、NeMo Guardrails 三个静态基线对比；在测试集上达到91% F1，分别比关键词高43pp、LlamaGuard高25pp、NeMo高4pp；连续改进阶段在跨域数据上从86%提升至93% (+7pp)

**⚠️ 局限性**

仅在英语对话上验证，未测试多语言和更复杂场景；训练样本量有限，需评估规模影响；模型依赖单一LLM，可能对性能与成本产生影响；自动化改进对可解释性和审计追踪带来挑战

---

## 249. Unintended Memorization of Sensitive Information in Fine-Tuned Language Models

**arXiv ID:** 2601.17480 | [PDF](https://arxiv.org/pdf/2601.17480v1)

**作者:** Marton Szep `[一作]` (Technical University of Munich), Daniel Rueckert `[通讯]` (Imperial College London)

**通讯引用:** 90583 | [OpenAlex ID](https://openalex.org/A5006461848)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在对大型语言模型进行领域微调时，模型在仅出现在输入而不出现在目标中的个人可识别信息（PII）的无意记忆现象，并对四种主流隐私保护方法（差分隐私、机器遗忘、正则化、首选优化）在多种数据集上的效果进行系统评估。

**💡 创新点**

创新点在于首次将“输入‑仅 PII 记忆”与真实医疗数据结合，提出基于 True‑Prefix Attack（TPA）的量化指标，全面探讨语言、频率、模型规模和任务类型对记忆的影响，并对比预防性与后训练性方法的隐私‑性能权衡。

**🔧 技术方法**

使用的技术包括参数高效微调（QLoRA）、差分隐私训练（Opacus）、自蒸馏遗忘（UnDial）、正则化约束、对齐优化（DPO）以及多种评估策略（贪婪、采样、交叉记忆）。

**📊 数据集**

实验数据集包括：①合成多语言金融 NER 数据集 GretelAI‑Financial；②德国医院骨科病理报告（2,553 篇）；③德国住院摘要（26,306 篇）。

**📈 对比分析**

与基线微调模型相比，DP 在 greedy / cross‑memorization 场景下能显著降低 PII 泄露（高达 60% 以上）但训练不稳定；后训练方法（DPO、UnDial）在采样攻击下更稳健，隐私‑性能平衡更好；正则化效果有限，机器遗忘受 seed 集质量影响。整体任务准确率在 0.7–0.9 之间，PII 提取数目随模型规模和语言显著变化。

**⚠️ 局限性**

主要局限在于仅针对 1B–12B 参数的量化微调模型、未覆盖更大规模基础模型或完全微调；数据集主要为合成或少量私有医疗数据，标签质量和多样性受限；未对白盒梯度泄露、专业 jailbreak 等更强攻击进行全面评估。

---

## 250. PatchIsland: Orchestration of LLM Agents for Continuous Vulnerability Repair

**arXiv ID:** 2601.17471 | [PDF](https://arxiv.org/pdf/2601.17471v1)

**作者:** Wonyoung Kim `[一作]` (Samsung Electronics), Insu Yun `[通讯]` (KAIST)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个可与 OSS‑Fuzz 集成的持续漏洞修复（CVR）系统，利用多模型 LLM 代理组合、两阶段基于补丁的去重以及优先级+供应商感知的调度机制，实现自动生成并交付补丁，极大提升了持续 fuzzing 流程中的修复效率与可靠性。

**💡 创新点**

创新点主要包括：① 基于多代理的集成式修复框架，实现跨模型多样性与容错；② 两阶段补丁‑基去重（crash‑side 与 patch‑side）降低重复工作；③ FP^2 调度（First‑come‑first‑served、Preference‑based、Provider‑aware）兼顾时间与成本；④ 提供统一的 Agent 开发框架与环境缓存，降低研发成本。

**🔧 技术方法**

使用了 LLM（GPT‑4o‑mini、Claude Sonnet‑4、Gemini 2.5 Pro）、容器化与 Kubernetes 部署、ccache 与 Maven 依赖缓存、Tree‑Sitter/ctags 代码搜索、语言服务器（clangd、Eclipse JDT）、补丁‑基去重算法、调度策略与多代理协同。

**📊 数据集**

评估采用 AIxCC 2025/2026 公开基准，包含 24 个 OSS 项目、92 个合成漏洞（C/Java）以及 AIxCC final 53 项挑战项目的真实漏洞集。

**📈 对比分析**

与 AIxCC 竞赛中基线系统（VulFix、AutoPatch）对比，CVR 在 92 个 PoV 上修复 84 / 92（≈91%）漏洞，比赛中 31 / 43（72.1%）成功率，明显优于基线；ensemble 方案比单代理提升约 5–10% 的成功率；FP^2 在时间/成本平衡上优于纯并行或纯顺序，达到低成本高效的补丁生成。

**⚠️ 局限性**

主要限制包括：初始化阶段仍存在单点失效（如符号链接错误导致系统停机）；仍会产生“可疑但不正确”的可行补丁；整体 LLM 计算成本高于基线；在缺乏功能测试的零日或意外漏洞场景下易产生错误补丁，且对功能测试覆盖度要求高。

---

## 251. ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation

**arXiv ID:** 2601.17468 | [PDF](https://arxiv.org/pdf/2601.17468v1)

**作者:** Chia-Ming Lee `[一作]` (National Yang Ming Chiao Tung University), Yu-Lun Liu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 1145 | [OpenAlex ID](https://openalex.org/A5101674908)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于双流框架的单图像反射分离方法 ReflexSplit，能够将混合图像有效拆解为透射层与反射层；

**💡 创新点**

创新点包括：跨尺度门控融合（CrGF）实现多尺度特征自适应聚合；层融合‑分离块（LFSB）通过差分注意力实现显式层分离；以及课程式训练策略逐步强化层间差异；

**🔧 技术方法**

采用了 Swin Transformer 作为全局特征提取器，MuGI 作为局部纹理提取器，结合 CrGF 与 LFSB、差分注意力、余弦退火学习率、Charbonnier 及 VGG 感知损失等技术；

**📊 数据集**

训练使用 7,643 对合成数据、90 对真实数据、200 对自然数据；评估集包含 Real20、Nature、SIR²（Objects、Postcard、Wild）以及 OpenRR‑1K；

**📈 对比分析**

与 11 种现有方法（包括 DSRNet、DSIT、RDNet 等）进行对比，ReflexSplit 在 PSNR、SSIM、LPIPS 等指标上普遍领先，尤其在结构保真度和感知质量上表现突出；

**⚠️ 局限性**

局限性体现在强光照、镜面反射、室内外混合场景等极端条件下仍可能出现分离失真或细节缺失。

---

## 252. Data-driven Test Generation for Fuzzing AI Compiler

**arXiv ID:** 2601.17450 | [PDF](https://arxiv.org/pdf/2601.17450v1)

**作者:** Qingchao Shen `[一作]` (Tianjin University), Qingchao Shen `[通讯]` (Tianjin University)

**通讯引用:** 213 | [OpenAlex ID](https://openalex.org/A5056795788)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出统一的数据驱动AI编译器测试框架，系统覆盖模型加载、优化和低级硬件优化三阶段，并在四个主流编译器上发现266个新bug

**💡 创新点**

创新点在于：①将AI库的算子测试迁移至编译器模型加载阶段；②构造优化感知的计算图合成；③基于文档+LLM生成多样化低级IR并进行轻量级突变，三种阶段特定的生成策略；

**🔧 技术方法**

使用测试迁移、图谱合成、低级IR突变以及大语言模型(LLM)提取优化模式和约束的技术组合

**📊 数据集**

采用AI库（PyTorch、TensorFlow等）提供的算子单元测试、编译器自测的优化案例、以及通过LLM生成的低级IR种子；无外部公开数据集

**📈 对比分析**

与NNSmith、WhiteFox、Tzer等现有fuzzer对比，模型加载阶段覆盖率提升11.9%~47.4%，高阶优化阶段分支/行覆盖分别高60.2%/66.98%，低级优化阶段发现bug数量显著提高，整体表现优于现有方法

**⚠️ 局限性**

局限在仅评估四个编译器，缺乏对新兴编译器的全面验证；对LLM产生的错误约束/假设存在潜在风险；仍缺乏自动化bug定位与修复功能

---

## 253. Building a Bridge between the Two Schools: Realizing a Practical Path to Include Literacy-based Skills within the STEM Curricula

**arXiv ID:** 2601.17447 | [PDF](https://arxiv.org/pdf/2601.17447v1)

**作者:** Jorge Torres Gómez `[一作]` (Technische Universität Berlin), Carmen Peláez-Moreno `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 1071 | [OpenAlex ID](https://openalex.org/A5090068120)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一套分步骤的教学方法，利用美术、音乐、戏剧、游戏化等艺术化手段，将专业技能融入计算机科学等STEM课程，提升学生的技术与人文素养。

**💡 创新点**

创新点在于将艺术与专业技能系统化结合，构建可推广的“桥梁”方法论，突破传统技术课程与人文学科的壁垒，首次在同一课程内同步培养技术能力与软技能。

**🔧 技术方法**

主要技术包括：基于美术的可视化与文本创作练习（信息熵与艺术作品分析）、奥克斯福德式辩论（辩论论证与口语表达训练）、社交化游戏化学习（点数化评估与协作任务）。

**📊 数据集**

使用的数据集为学生问卷（定量与定性）和期末考试成绩；研究涵盖多所欧洲技术大学的课程（信息熵实验、辩论课程、游戏化课程）共计约 200 名学生。

**📈 对比分析**

通过与传统教学（无艺术化干预）的对比，期末考试平均成绩从 47.54 提升至 56.64，学生对活动的兴趣与感知效果均显著更高（问卷平均分 9/10），显示该方法在学习效果与参与度上均优于传统模式。

**⚠️ 局限性**

局限性包括：教师对艺术方法的接受度与专业培训不足；课程空间与资源受限，难以大规模推广；评估仍偏重定量，缺乏系统的质性评价框架；对女性参与度的影响尚未得到实证验证。

---

## 254. Clustering-driven Memory Compression for On-device Large Language Models

**arXiv ID:** 2601.17443 | [PDF](https://arxiv.org/pdf/2601.17443v1)

**作者:** Ondrej Bohdal `[一作]` (Samsung Research and Development Institute United Kingdom), Taha Ceritli `[通讯]` (Samsung Research and Development Institute United Kingdom)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于聚类的记忆压缩方法，用于在设备端大型语言模型的个性化生成中降低上下文占用并提升生成质量。

**💡 创新点**

创新点在于先将用户记忆按相似度聚类，然后在每个簇内取平均合并，从而既减少冗余记忆，又保持语义一致性，优于传统的拼接或直接取均值的做法。

**🔧 技术方法**

采用的技术包括K‑Means聚类、LoRA微调的冻结LLM进行记忆编码、BM25检索获取最近记忆、以及ROUGE‑L评估生成结果。

**📊 数据集**

实验使用了LaMP基准中的个性化新闻标题生成、推文改写和电影标签等任务的数据集。

**📈 对比分析**

与平均和拼接两种基线进行对比，实验结果表明聚类压缩在保持相同或更少记忆token的情况下，ROUGE‑L得分提升约1–3%，并在Qwen2.5、Gemma3、StableLM2等多模型上均优于基线。

**⚠️ 局限性**

主要局限包括需预先设定簇数且对聚类质量敏感，簇数过少会导致信息丢失；此外实验仅在小规模设备模型上验证，未探讨更大上下文窗口或多模态场景的适用性。

---

## 255. Data-driven Clustering and Merging of Adapters for On-device Large Language Models

**arXiv ID:** 2601.17441 | [PDF](https://arxiv.org/pdf/2601.17441v1)

**作者:** Ondrej Bohdal `[一作]` (Samsung Research and Development Institute UK), Umberto Michieli `[通讯]` (Samsung Research and Development Institute UK)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种数据驱动的 LoRA 聚类与合并方法 D^2C，帮助在移动设备有限存储下使用少量多任务 LoRA 提升文本生成性能。

**💡 创新点**

创新点在于利用极少量任务示例（仅10条）进行迭代优化的聚类过程，自动寻找能通过合并获得良好泛化的 LoRA 集群，而非传统的随机或 K‑Means 方法。

**🔧 技术方法**

使用 LoRA 作为参数高效微调技术，TIES 作为合并算法，以及基于任务示例的交叉熵损失评估；实现了迭代式迁移与评估的聚类优化。

**📊 数据集**

在 40 个跨语言（英、西、法、德、意、简体中文、韩、日）文本生成任务上进行实验，涉及 Grammar Error Correction、Smart Reply、Summarization、Tone Adjustment、Question Answering 等子任务，使用 Llama 3.2 3B、Qwen 2.5 1.5B、StableLM 2 1.6B 三大模型。

**📈 对比分析**

与随机聚类、K‑Means（平面、SVD 特征）以及单任务 LoRA、零样本基线比较，D^2C 在保持 12.5% 存储预算的情况下平均提升 2.1–4.5% 的性能，单任务 LoRA 上限约 32.9%，零样本约 14.7%。

**⚠️ 局限性**

局限性包括需要访问少量任务示例（尽管数量极少），聚类与合并过程依赖离线 GPU 计算，且对极小或极大集群数的表现仍需进一步验证。

---

## 256. The 17% Gap: Quantifying Epistemic Decay in AI-Assisted Survey Papers

**arXiv ID:** 2601.17431 | [PDF](https://arxiv.org/pdf/2601.17431v1)

**作者:** H. Kemal İlter `[一作]` `[通讯]` (Bakirçay University), H. Kemal İlter (Bakirçay University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 50 篇人工智能综述论文中的 5,514 条引用进行取证审计，量化并分类“幻影”引用的比例与模式。

**💡 创新点**

首次系统量化 AI 文献中引用链腐蚀的“幻影率”，提出分级诊断分类（Syntax Error、Broken Link、Ghost）并揭示其结构性稳定性（平稳 17% 幻影率）。

**🔧 技术方法**

使用混合验证管线：DOI 与 arXiv 解析、Crossref 与 Semantic Scholar API 调用、熵过滤、Levenshtein 余弦相似度、阈值决策与统计回归分析。

**📊 数据集**

数据集为 50 篇 2024‑2026 年间发布的 AI 综述论文，共 5,514 条独立引用，来源于 arXiv（cs.CL、cs.LG、cs.AI）和 Crossref 元数据。

**📈 对比分析**

与传统仅基于 DOI 解析的验证方法对比，混合管线提升了可验证率（41%→50%）并识别出 17% 幻影率；统计回归显示幻影率无显著时间趋势，证明其为结构性失效。

**⚠️ 局限性**

局限性包括：32% 未知类别可能掩盖更多幻影、阈值设定依赖人工校验、API 可用性与速率限制、未覆盖非索引源（GitHub、技术报告等），因此幻影率可能是下限。

---

## 257. On the Impossibility of Simulation Security for Quantum Functional Encryption

**arXiv ID:** 2601.17497 | [PDF](https://arxiv.org/pdf/2601.17497v1)

**作者:** Mohammed Barhoush `[一作]` (University of Montreal), Louis Salvail `[通讯]` (University of Montreal)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了在多种攻击模型下，量子功能加密的仿真安全性在一般电路族上不可实现，扩展了经典不可实现结果。

**💡 创新点**

创新地引入量子伪随机态的不可压缩性质，并在仅假设伪随机量子态或公钥加密可实现的情况下给出新的不可实现结论。

**🔧 技术方法**

采用量子信息压缩论证、Levy 引理、伪随机性与不可压缩性结合的归约技术，以及模拟者与真实实验的不可区分性分析。

**📊 数据集**

无数据集，本文为纯理论性不可实现证明。

**📈 对比分析**

与以往的经典不可实现论证对比，指出量子资源并未突破限制；结果在所有考虑的查询模型中均为无条件或在更弱假设下成立。

**⚠️ 局限性**

局限在于仅针对全功能电路族、秘密密钥设置；对受限功能族、简洁方案或公钥场景的进一步研究仍待解决。

---

## 258. EquiForm: Noise-Robust SE(3)-Equivariant Policy Learning from 3D Point Clouds

**arXiv ID:** 2601.17486 | [PDF](https://arxiv.org/pdf/2601.17486v1)

**作者:** Zhiyuan Zhang `[一作]` (Purdue University), Yu She `[通讯]` (Purdue University)

**通讯引用:** 1371 | [OpenAlex ID](https://openalex.org/A5018653973)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种针对3D点云的噪声鲁棒 SE(3) 等变政策学习框架 EquiForm，融合几何去噪和对比学习以保持等变性；

**💡 创新点**

在保持等变性的同时主动修正噪声导致的几何偏差，并通过对比损失正则化特征一致性；

**🔧 技术方法**

几何去噪模块（法向与切向校正）、Vector Neuron 等变网络、InfoNCE 对比学习、SE(3) 规范化与动作预测；

**📊 数据集**

Sim 任务：16 个 MimicGen、Push‑T、RoboTwin 任务；真实世界任务：Franka Panda 进行 4 种操纵任务；

**📈 对比分析**

与 DP3（非等变）和 Canonical Policy（等变但无去噪）比较，EquiForm 在 Sim 平均 66.6% 成功率，比 Canonical 提升 10% 以上、比 DP3 提升 24%+；在噪声、旋转和真实机器人实验中也显著优于基线；

**⚠️ 局限性**

对小物体、薄材质或极端遮挡下的点云采样失真仍影响性能；统一下采样可能导致细节丢失；对极端噪声/遮挡的鲁棒性有限。

---

## 259. Adversarial Alignment and Disentanglement for Cross-Domain CTR Prediction with Domain-Encompassing Features

**arXiv ID:** 2601.17472 | [PDF](https://arxiv.org/pdf/2601.17472v1)

**作者:** Junyou He `[一作]` (JD.COM), Sulong Xu `[通讯]` (JD.COM)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种跨域推荐模型 A^2DCDR，通过对抗性对齐与特征解耦来捕获域包容性特征，实现源域与目标域信息的高效迁移。

**💡 创新点**

创新点包括：1) 结合 DC-MMD 与梯度反转实现域包容性特征对齐；2) 引入信息理论互信息最小化 (CLUB) 的内部解耦与重构机制，保持信息完整性；3) 设计目标感知特征融合 (TAFC) 自适应聚合多类型特征。

**🔧 技术方法**

使用的技术包括：对抗性训练、最大均值差异 (MMD)、梯度反转层、信息互信息估计 (CLUB)、LightGCN 或 BST 编码器、重构网络、注意力融合。

**📊 数据集**

数据集涵盖 Amazon 的四个真实领域（Phone、Elec、Cloth、Sport）进行离线实验，并在 JD.com 视频与商品双域上进行在线 A/B 测试。

**📈 对比分析**

与 SOTA 方法（GDCCDR、DDCDR、DCCDR、DisenCDR 等）进行对比，A^2DCDR 在 HR@10、NDCG@10 上均取得显著提升（多场景提升 1–3% 以上），在线测试 CTR 提升约 7.2%。

**⚠️ 局限性**

局限性：仅支持两域共享用户场景，模型复杂度相对较高；对离散或稀疏特征的解释性有限；对多域层次关系、长期序列跨域依赖等情况尚未深入探索。

---

## 260. Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction

**arXiv ID:** 2601.17469 | [PDF](https://arxiv.org/pdf/2601.17469v1)

**作者:** Wei Ju `[一作]` (Sichuan University), Ming Zhang `[通讯]` (Peking University)

**通讯引用:** 20831 | [OpenAlex ID](https://openalex.org/A5100447284)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为ICGNN的鲁棒图神经网络，用来在存在标签噪声且标签稀缺的图数据上进行节点分类

**💡 创新点**

1) 通过结构层和属性层的图扩散矩阵计算影响矛盾分数（ICS）作为噪声指标；2) 用高斯混合模型自适应区分噪声与干净标签；3) 对检测到的噪声标签采用软聚合邻居预测进行校正；4) 对未标记节点生成伪标签进一步增强监督

**🔧 技术方法**

图扩散矩阵、ICS计算、Gaussian Mixture Model、软标签聚合、伪标签化、GCN编码器

**📊 数据集**

六个常用图数据集（Coauthor‑CS、Amazon‑Photo、Cora、Pubmed、Citeseer、DBLP）以及大规模OGBN‑Arxiv和异质性Cornell数据集

**📈 对比分析**

与传统GCN、Forward、Co‑Teaching+、NRGNN、RTGNN、CGNN、CR‑GNN、DND‑NET、ProCon等方法在20%噪声和不同噪声类型下进行对比，ICGNN在所有数据集上均取得最高或接近最高的准确率，且在噪声率和标签率变化时表现更稳健

**⚠️ 局限性**

对噪声标签的识别仍依赖于图结构和属性的预设权重α，需手工调参；当标签极少或噪声极高时，ICS分布可能与实际噪声不完全分离；在极大规模图上仍需进一步优化扩散矩阵的计算效率

---

## 261. PILOT: A Perceptive Integrated Low-level Controller for Loco-manipulation over Unstructured Scenes

**arXiv ID:** 2601.17440 | [PDF](https://arxiv.org/pdf/2601.17440v1)

**作者:** Xinru Cui `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8971 | [OpenAlex ID](https://openalex.org/A5107772128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出并实现了一个统一的单阶段强化学习框架（PILOT），能够在复杂不规则地形上实现感知驱动的全身行走与操作任务，并在仿真与真实Unitree G1双足机器人上完成跨阶梯与高平台的搬运任务。

**💡 创新点**

创新点包括：
1) 结合LiDAR基机器人中心地形高度图的跨模态感知编码器，利用多尺度注意力机制精确估计可行走位置；
2) 采用Mixture‑of‑Experts（MoE）策略网络，将不同运动技能解耦并在单一政策中协同；
3) 通过自适应指令日历与无运动捕捉的随机命令采样，避免分布偏差并提升泛化；
4) 在单一低层控制器上实现行走与抓取的无缝协作，为后续高层规划奠定基础。

**🔧 技术方法**

技术栈：
- 强化学习：PPO；
- 运动学与动力学：IsaacLab仿真、PD低层控制；
- 感知编码：跨模态上下文编码器（预测式本体编码+注意力式地形编码）；
- 模型结构：MoE actor网络；
- 训练策略：自适应命令日历、对比学习、残差动作参数化。

**📊 数据集**

数据集：
- 仿真环境：在IsaacLab中随机生成的多阶梯、斜坡、平台与噪声地形；
- 真实实验：使用Unitree G1机器人搭载LiDAR实现实时地形高度图；
- 训练无真实数据集，全部基于仿真生成的环境与随机指令。

**📈 对比分析**

对比方法：HOMIE、FALCON、AMO三种主流低层全身控制器。实验结果显示：
- 在平坦地形上，PILOT在速度、角速度、姿态与手部位置跟踪误差上均优于对比模型；
- 在复杂地形下，PILOT的脚步碰撞率（E_stumble）仅为0.006，显著低于去掉感知或去掉注意力编码的版本；
- 实际搬运任务中，PILOT在5次试验中实现100%成功率；
- 通过MoE与注意力模块的消融实验验证其对稳定性与精度的关键作用。

**⚠️ 局限性**

限制与未来工作：
- 依赖高分辨率LiDAR和精确姿态估计，设备成本与部署复杂度较高；
- 训练场景主要集中在地形高度变化与粗糙度，缺乏动态障碍物与人类交互的挑战；
- 目前仅在Unitree G1平台验证，跨机器人/跨尺寸的迁移性尚未探究；
- 仍需进一步提升在极端动态扰动和长时持续作业中的鲁棒性。

---

## 262. UniGRec: Unified Generative Recommendation with Soft Identifiers for End-to-End Optimization

**arXiv ID:** 2601.17438 | [PDF](https://arxiv.org/pdf/2601.17438v1)

**作者:** Jialei Li `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**通讯引用:** 41064 | [OpenAlex ID](https://openalex.org/A5038668215)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 UniGRec，一种统一的生成式推荐框架，实现了分词器和推荐器的端到端联合训练。

**💡 创新点**

创新点包括：①使用软标识符使分词器可微分；②温度退火推理对齐（Annealed Inference Alignment）解决训练-推理差异；③代码词统一正则化（Codeword Uniformity Regularization）避免标识符崩塌；④双重协同蒸馏（Dual Collaborative Distillation）注入协同信号。

**🔧 技术方法**

采用的技术包括：RQ‑VAE软分词、Transformer（T5）推荐器、温度退火、KL散度统一正则化、InfoNCE 蒸馏、两阶段联合训练。

**📊 数据集**

使用了三个真实数据集：Amazon Beauty、Amazon Pet 以及私有 Upwork。

**📈 对比分析**

与传统基于 ID 的方法（Caser、GRU4Rec、SASRec、BERT4Rec、HGN）以及其他生成式方法（TIGER、LETTER、EAGER、OneRec、ETEG‑Rec、DiscRec‑T）进行比较。UniGRec 在 Recall@K、NDCG@K 等指标上均优于所有基线，尤其在 Upwork、Beauty、Pet 三个数据集上均取得最高分。

**⚠️ 局限性**

局限性：依赖预训练协同模型进行蒸馏，训练流程仍需两阶段；在极大规模工业数据集上的可扩展性和效率尚待进一步验证；软分词和温度退火可能带来额外计算开销。

---

## 263. Active Hypothesis Testing for Correlated Combinatorial Anomaly Detection

**arXiv ID:** 2601.17430 | [PDF](https://arxiv.org/pdf/2601.17430v1)

**作者:** Zichuan Yang `[一作]` (Tongji University), Yiming Xing `[通讯]` (Tongji University)

**通讯引用:** 12 | [OpenAlex ID](https://openalex.org/A5013537197)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在相关噪声环境下的组合异常检测，提出了可自适应设计测量的活跃假设检验方法 ECC-AHT，并证明其样本复杂度达到信息理论极限。

**💡 创新点**

创新点在于：①将相关性视为可利用资源，通过差分测量消除公共噪声；②在每一步主动选取冠军‑挑战者对，利用二次规划得到最优测量向量，最大化 Chernoff 信息；③在组合假设空间中实现可扩展的推断，保持排名而非完整后验，显著降低计算复杂度。

**🔧 技术方法**

使用技术包括：序贯实验设计、Chernoff 信息量度、二次规划（Quadratic Programming）求解最优测量向量、伪似然排名更新、基于冠军‑挑战者的两两比较策略。

**📊 数据集**

实验数据集：合成数据（不同维度、不同相关系数）以及真实工业控制数据集 WaDi（预处理后 66 维传感器）。

**📈 对比分析**

与 Round‑Robin、Random Sparse Projection、CombGapE、TTTS、HDS 等基线方法比较。ECC‑AHT 在样本数、检测延迟上均优于基线，尤其在高相关情形下相差十倍；在 WaDi 上也实现了更低的检测延迟，验证了相关性建模的实用价值。

**⚠️ 局限性**

局限性：假设观测服从已知协方差的高斯分布，信号模式 δ 已知；只考虑线性测量；对未知信号强度或非高斯/非线性相关的场景尚未涵盖，需要进一步扩展。

---

## 264. Coronary Artery Segmentation and Vessel-Type Classification in X-Ray Angiography

**arXiv ID:** 2601.17429 | [PDF](https://arxiv.org/pdf/2601.17429v1)

**作者:** Mehdi Yousefzadeh `[一作]` (Institute for Research in Fundamental Sciences), Majid Maleki `[通讯]` (Iran University of Medical Sciences)

**通讯引用:** 4628 | [OpenAlex ID](https://openalex.org/A5075440850)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在X射线冠状动脉造影中实现了冠状动脉和血管类型的自动分割与识别，采用经典滤波器与深度学习模型相结合的策略。

**💡 创新点**

提出了基于每张图像自适应调参的经典滤波器改进、联合超分辨率与增强、以及冠脉+导管双标签监督的FPN模型，显著提升了分割稳定性和跨中心迁移能力。

**🔧 技术方法**

使用Meijering/Frangi/Sato滤波器+SVR自适应调参、U‑Net、FPN+SE‑ResNet18/SE‑ResNeXt50、Swin Transformer以及高分辨率输入与超分辨率预处理等技术。

**📊 数据集**

基于670条冠状动脉造影序列（407名受试者）构建的内部数据集，并在公开的DCA1外部数据集上进行验证。

**📈 对比分析**

与传统滤波器全局调参、单一深度模型及无监督对比，FPN+SE‑ResNet18在756×756分辨率下达成0.914 Dice，融合冠脉+导管标签进一步提升至0.931；在DCA1外部测试中，Dice由0.798/0.814提升至0.881/0.882；血管类型识别准确率在95–98%之间，Dice约0.79–0.84。

**⚠️ 局限性**

局限性包括对外部数据仍需细调、模型对高分辨率输入的计算开销较大、主要基于伊朗院系的样本，可能存在人口和设备差异导致的迁移性限制。

---

## 265. Less is More for RAG: Information Gain Pruning for Generator-Aligned Reranking and Evidence Selection

**arXiv ID:** 2601.17532 | [PDF](https://arxiv.org/pdf/2601.17532v1)

**作者:** Zhipeng Song `[一作]` (Dalian University of Technology), Heng Qi `[通讯]` (Dalian University of Technology)

**通讯引用:** 3357 | [OpenAlex ID](https://openalex.org/A5087744818)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出信息增益剪枝（IGP）作为检索增强生成（RAG）证据选择模块，在有限上下文预算下通过生成器的不确定性下降来评估证据效用，并在前置阶段进行重排序与阈值剪枝，以提升QA质量与成本效率。

**💡 创新点**

①用生成器的基于不确定性下降的“信息增益”作为证据效用信号，替代传统相关性排序；②只对检索后重排序阶段做插件式替换，保持原始预算接口；③通过阈值剪枝控制证据输入，显著降低冗余/冲突，提高端到端性能。

**🔧 技术方法**

黑盒步骤式不确定性估计（使用Top‑K logits/概率计算归一化熵），信息增益评分与阈值剪枝；基于生成器的推理（如Qwen2.5、Llama‑3）、稀疏/稠密检索、无监督/监督重排序；实验中采用vLLM进行并行推理。

**📊 数据集**

公开开放域QA基准NQ、TriviaQA、PopQA、SQuAD、AmbigQA，以及Wiki18检索语料库。

**📈 对比分析**

与BM25/Contriever检索、CE/BGE/YesNo/QLM重排序等基线相比，IGP在多种检索器、生成器规模下保持或提升token‑level F1，同时显著降低最终上下文token数；在多证据预算下，IGP可实现12–20%相对F1提升，token数下降约76–79%，提升了NTE。

**⚠️ 局限性**

只评估单一证据的增益，未显式建模多证据互补/冗余；信息增益可能因自信误导而误判；需要在推理前额外进行一次或多次推理（成本可并行），且对生成器的prompt/decoding设置敏感。

---

## 266. FMIR, a foundation model-based Image Registration Framework for Robust Image Registration

**arXiv ID:** 2601.17529 | [PDF](https://arxiv.org/pdf/2601.17529v1)

**作者:** Fengting Zhang `[一作]`, Hang Zhang `[通讯]` (Cornell University)

**通讯引用:** 8403 | [OpenAlex ID](https://openalex.org/A5100389684)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种基于基础模型的医学图像配准框架 FMIR，通过利用预训练的 2D 视觉基础模型（如 DINO、SAM）对 3D 体积进行切片处理并提取域不变特征，再配合多尺度通用注册头来预测变形场，实现仅用单一数据集训练即可在不同域上获得 SOTA 级别的配准效果。

**💡 创新点**

创新点主要包括：①将 2D 基础模型迁移到 3D 医学图像中，借助切片重组实现跨域域不变特征提取；②设计通用的多尺度金字塔注册头，可与不同基础模型无缝对接；③提出通道正则化（channel dropout + PCA）策略，强迫网络学习结构级别的核心关联，显著提升跨域泛化；④在保持高速推理（<1 s）和高精度的同时，仅用单一数据集即可获得多域表现。

**🔧 技术方法**

使用的关键技术包括：预训练视觉 Transformer（DINO ViT‑B）、SAM；切片-重组 3D 处理；通道正则化（随机通道抽样 + 训练时随机化，推理时 PCA 投影）；多尺度金字塔卷积注册头；相似性损失 NCC 与 Dice；平滑损失 L_smooth；Adam 优化器和多项式学习率衰减；以及 3 层 3D 卷积上下文恢复。

**📊 数据集**

采用了 ACDC 心脏 MRI 数据集（内域）和 Learn2Reg 2020 腹部 CT 数据集（跨域）进行评估；在 ACDC 内域测试共 100 对配准；在腹部 CT 测试共 42 对配准；训练集分别为 85（ACDC）和 20（腹部）扫描。

**📈 对比分析**

与多种学习式配准模型（VoxelMorph、TransMorph、LKU‑Net、CorrMLP、MemWarp、RDP、uniGradICON）在 ACDC 内域数据上对比，FMIR 在 Dice 上达 79.82%（最高），HD95 为 9.07 mm，SDlogJ 为 0.049，推理时间仅 0.62 s（相比 uniGradICON 的 4.95 s 大幅加速）。在跨域测试中，FMIR 在腹部对心脏或心脏对腹部的 Dice 均维持在 80% 以上，优于其它基线，并且通道正则化的消除导致跨域性能显著下降，验证了其有效性。

**⚠️ 局限性**

局限性包括：①仅采用 2D 基础模型切片方式，可能忽视 3D 体积的长程空间上下文；②对 GPU 资源和切片重组的计算开销仍有一定依赖；③未在更广泛的多解剖结构或更大模态差异的数据集上进行验证；④通道正则化的超参数和 PCA 维度需在不同任务中手动调优。

---

## 267. Constrained Multi-Objective Genetic Algorithm Variants for Design and Optimization of Tri-Band Microstrip Patch Antenna loaded CSRR for IoT Applications: A Comparative Case Study

**arXiv ID:** 2601.17513 | [PDF](https://arxiv.org/pdf/2601.17513v1)

**作者:** Moahmed Hamza Boulaich `[一作]`, Abdelatif El Afia `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了基于多目标遗传算法（MOGA）的自动化微带贴片天线设计与优化框架，重点在于通过嵌入补充环形谐振器（CSRR）实现3频段（2.4 GHz、3.6 GHz、5.2 GHz）多频共振，并在同一设计中实现天线尺寸压缩与回波损耗（S₁₁）低于-10 dB；

**💡 创新点**

创新点在于将多目标问题通过加权求和进行标量化，兼顾所有频段的S₁₁，并引入约束修复与精英保留机制，使搜索过程集中在单一最优解；同时系统化比较了五种主流MOGA（PGA、NSGA‑I、NSGA‑II、NSGA‑III、SPEA）在此天线设计任务中的收敛与多样性表现；

**🔧 技术方法**

技术方法包括：MATLAB与CST Microwave Studio 的脚本协同接口实现自动化仿真；五种遗传算法实现（包含非支配排序、拥挤距离、参考点、强度计数等机制）；加权求和标量化目标函数；遗传算子（交叉、变异）与精英保留；性能评估指标包括GD、IGD、VSWR、S₁₁；

**📊 数据集**

使用的数据集为基于CST的全波仿真结果；实验时采用Rogers RT5880介质（ε_r = 2.2、tan δ = 0.0009、厚度 1.57 mm），设计参数（如CSRR半径、间隙、贴片尺寸等）均通过算法搜索得到，未涉及公开的实验测量数据；

**📈 对比分析**

通过比较GD/IGD、S₁₁、VSWR、增益等指标发现标量化MOGA在10代内快速收敛并保持较高种群多样性，最终S₁₁分别为-21.56 dB、-16.60 dB、-27.69 dB，VSWR低于1.5，优于传统PGA、NSGA‑I/II/III及SPEA；

**⚠️ 局限性**

限制主要包括：标量化方法无法完全锁定3.6 GHz精确共振点（偏离0.02 GHz）；MATLAB–CST同步机制仅在最终解保存全部仿真数据，导致无法对整个种群做完整的实验记录；未进行实验验证；种群规模和迭代次数受限，未来可扩大规模或引入强化学习自适应控制参数。

---

## 268. BMDS-Net: A Bayesian Multi-Modal Deep Supervision Network for Robust Brain Tumor Segmentation

**arXiv ID:** 2601.17504 | [PDF](https://arxiv.org/pdf/2601.17504v1)

**作者:** Yan Zhou `[一作]` (Changsha University of Science and Technology), Zehua Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 31829 | [OpenAlex ID](https://openalex.org/A5100715016)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 BMDS‑Net，一种统一框架，用于多模态脑肿瘤分割，兼顾精度、鲁棒性和可信度。

**💡 创新点**

创新点：
1) Zero‑Init MMCF 模块实现输入层动态多模态权重调整，提升缺失模态时的鲁棒性；
2) Residual‑Gated DDS 在解码器深层加入门控深度监督，显著改善边界精度；
3) 内存高效的 Bayesian fine‑tuning，将网络转为概率预测器，提供体素级不确定度映射并实现良好校准。

**🔧 技术方法**

技术：Transformer‑based Swin UNETR 编码器、Zero‑Init 多模态上下文融合、残差门控深度监督、变分贝叶斯全连接层、单样本 MC Dropout 推理。

**📊 数据集**

使用 BraTS 2021 多模态 MRI 数据集（1,251 例），包含 FLAIR、T1、T1ce、T2 四种序列。

**📈 对比分析**

与 SegResNet、MedNeXt、TransBTS、nnFormer、nnU‑Net、Swin UNETR 等 SOTA 进行比较，Dice 分数与 nnU‑Net 相当（WT 0.9293，TC 0.9098，ET 0.8675），且在 HD95 上显著优于竞争模型；在缺失模态和噪声环境下，BMDS‑Net 的 Dice 稳定性最高；贝叶斯不确定度校准 ECE 仅 0.0037，逼近 Deep Ensemble 的性能，训练成本仅 1.2×。

**⚠️ 局限性**

局限：
1) Bayesian fine‑tuning 需要多次前向推理（T=20），推理速度比纯确定性模型慢；
2) 在缺失 T2 或严重噪声时，分割仍出现明显误差；
3) 对术后切除腔和某些解剖结构（如脉络丛）易误判为肿瘤，需进一步引入临床先验知识。

---

## 269. LogPrism: Unifying Structure and Variable Encoding for Effective Log Compression

**arXiv ID:** 2601.17482 | [PDF](https://arxiv.org/pdf/2601.17482v1)

**作者:** Yang Liu `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 33227 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了统一冗余编码框架 LogPrism，采用统一冗余树（URT）将日志结构与变量动态联合建模，替代传统的解析-压缩两步流程。

**💡 创新点**

创新点在于提出统一冗余编码理念，利用层次冗余挖掘、变量重排序与子树构造，打破结构与参数的硬分离，实现“结构+变量”共识路径的单一编码，从而挖掘更深层次的上下文冗余。

**🔧 技术方法**

技术方案包括：并行流式预处理与局部树生成、全局树聚合与等价子树合并、变量频率过滤与稳定性排序、残差数据的全局排序与即时模板化，以及基于 LZMA 的最终压缩；整个流程实现了细粒度并行与层次化降噪。

**📊 数据集**

实验使用 LogHub 16 大基准数据集（覆盖分布式系统、超级计算机、操作系统、移动系统、服务器应用和独立软件），约 77 GB 日志，确保泛化性。

**📈 对比分析**

与四个日志专用压缩器（LogZip、LogReducer、LogShrink、Denum）及四个通用压缩器（gzip、bzip2、LZMA、PPMd）在压缩比和速度上对比，LogPrism 在 13/16 数据集上取得最高压缩比，压缩比比最优基线高 4.7%–80.9%（最多 58%），压缩速度平均 29.87 MB/s（单线程版本）或 41.55 MB/s（内部并行版本），远快于竞品。

**⚠️ 局限性**

局限性：单文件全局模式虽然能进一步提升压缩比，但速度会有显著下降；对极高熵、缺乏结构化变量（如 Spark）表现略逊；在极小数据集上并行化收益有限；在分布式部署和实时流式日志场景中仍需进一步验证。

---

## 270. EuleroDec: A Complex-Valued RVQ-VAE for Efficient and Robust Audio Coding

**arXiv ID:** 2601.17517 | [PDF](https://arxiv.org/pdf/2601.17517v1)

**作者:** Luca Cerovaz `[一作]` (Sapienza University of Rome), Emanuele Rodolà `[通讯]` (Sapienza University of Rome)

**通讯引用:** 6947 | [OpenAlex ID](https://openalex.org/A5087051832)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一种完全在复数域下实现的端到端 VQ‑VAE 声码器 EuleroDec，专为 24 kHz 语音在 6 kbps 与 12 kbps 两种低比特率下工作。

**💡 创新点**

首次实现了全复数化的分析‑量化‑合成管线，保持幅度‑相位耦合，完全不依赖 GAN 或扩散后处理，训练步骤大幅减少且收敛更快。

**🔧 技术方法**

采用了复数卷积、复数注意力、2×2 白化正则化、modReLU、Wirtinger 求导、残差向量量化、轴向注意力以及指数移动平均更新码本等技术。

**📊 数据集**

训练与评测使用 LibriTTS 100 h 子集，测试集划分为 in‑domain（test‑clean）与 out‑of‑domain（test‑other）。

**📈 对比分析**

与 AudioDec、EnCodec、APCodec 等主流码器在同等比特率下对比，EuleroDec 在 SI‑SDR、PESQ、GDD 等指标上至少与基准相当或更优，尤其在 out‑of‑domain 场景表现领先，且训练步骤仅约 5 万步，远低于对手的数十万步。

**⚠️ 局限性**

目前仅支持 24 kHz 语音、固定 6/12 kbps 两个比特率，且模型不可实时流式；在不同采样率或更宽范围的音频内容上的泛化能力尚未验证。

---

## 271. Reconstructing Training Data from Adapter-based Federated Large Language Models

**arXiv ID:** 2601.17533 | [PDF](https://arxiv.org/pdf/2601.17533v1)

**作者:** Silong Chen `[一作]` (National University of Defense Technology), Xiaohua Jia `[通讯]` (City University of Hong Kong)

**通讯引用:** 18784 | [OpenAlex ID](https://openalex.org/A5013643572)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对适配器式联邦大语言模型提出了一种名为Unordered-word-bag-based Text Reconstruction（UTR）的梯度逆向攻击，能够在冻结主干网络下几乎完美重构训练文本。

**💡 创新点**

创新点在于：①利用冻结的注意力层推断词袋；②在低秩适配器梯度子空间中进行句子级逆向；③通过语义一致性约束与贪心解码大幅压缩搜索空间，从而在低维梯度和大批量情形下实现高效逆向。

**🔧 技术方法**

采用梯度逆向、低秩矩阵分解、词袋推断、语义一致性搜索等技术，结合LoRA等适配器机制。

**📊 数据集**

使用 CoLA、SST-2、Rotten Tomatoes 数据集，实验模型包括 GPT2-Large、BERT、Qwen2.5-7B。

**📈 对比分析**

与 LAMP、DAGER 等基线相比，UTR 在 ROUGE‑1/2 上均达到 99% 以上，甚至在 batch 128 的大规模批量下保持近乎 100% 的重构精度，显著优于现有方法。

**⚠️ 局限性**

主要局限：在加入足够强的差分隐私噪声或极端梯度裁剪时失效；对 GPT 的单向注意机制下长序列重构仍有挑战；以及对词序信息的恢复依赖于预训练模型的结构。

---

## 272. Pipeline Inspection, Visualization, and Interoperability in PyTerrier

**arXiv ID:** 2601.17502 | [PDF](https://arxiv.org/pdf/2601.17502v1)

**作者:** Emmanouil Georgios Lionis `[一作]` (University of Glasgow), Sean MacAvaney `[通讯]` (University of Glasgow)

**通讯引用:** 1653 | [OpenAlex ID](https://openalex.org/A5014199889)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在PyTerrier中实现了管道的程序化检查、可视化（schematics）以及通过MCP服务器实现与其他工具的互操作。

**💡 创新点**

创新点包括：①基于Transformer输入/输出列的自动检查机制；②可交互的管道可视化生成；③将PyTerrier管道以MCP接口暴露，支持LLM调用和插件生态。

**🔧 技术方法**

使用Python编写，依赖PyTerrier框架、FAISS、BM25、融合与重排序模型；利用Jupyter Notebook展示可视化；使用MCP协议（HTTP接口）实现互操作。

**📊 数据集**

在示例中使用了常见的IR基准数据集（如TREC检索任务、MS‑MARCO、Doc2Query等）来演示检索、重排序和问答管道。

**📈 对比分析**

对比方法主要是通过程序化验证管道兼容性；演示了BM25检索、RAG问答、Doc2Query解构三种管道在MCP环境下的调用；未给出具体性能数值，仅展示功能性和交互性。

**⚠️ 局限性**

局限性：仍需Python运行时；MCP服务器依赖网络延迟；部分Transformer的输入/输出信息可能未完整规范；缺乏对大规模部署的性能评估。

---

## 273. PEARL: Prototype-Enhanced Alignment for Label-Efficient Representation Learning with Deployment-Driven Insights from Digital Governance Communication Systems

**arXiv ID:** 2601.17495 | [PDF](https://arxiv.org/pdf/2601.17495v1)

**作者:** Ruiyu Zhang `[一作]` (University of Hong Kong), Xin Zhao `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 53674 | [OpenAlex ID](https://openalex.org/A5100445703)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级的 PEARL 方法，利用有限标签对固定句子嵌入空间进行原型引导的几何调整，以提升最近邻检索的标签一致性和早期排名效果。

**💡 创新点**

创新点在于：①用类别原型作为低方差监督信号，在标签稀缺时实现可控的局部空间对齐；②通过对齐、对比、重建和正交约束避免空间崩塌，同时保持维度不变；③在不需要重新训练基编码器的情况下，仅通过后处理即可显著改善检索质量。

**🔧 技术方法**

技术方法包括：基于原型的对齐与对比损失、轻量化神经映射（分离信号与残差编码器）、重建与正交正则化、以及在检索前的 L2 归一化和可选的 PCA/白化等传统后处理。

**📊 数据集**

使用的主要数据集为数字治理通信语料（包含市民消息与治理类别），并在该语料上进行多折交叉验证，实验中还对标签规模进行控制（100-5000 条）。

**📈 对比分析**

与未处理原始嵌入、仅归一化、PCA+白化、以及 LDA 投影等基线相比，PEARL 在标签稀缺（100-300 条）时的 Purity@K、Hit@1 和 MRR@K 取得显著提升（约 25-30% 的提升），在中等标签规模下仍保持竞争力；当标签充足时，LDA+L2 在某些指标上略胜一筹，但 PEARL 仍优于未处理和无监督后处理。

**⚠️ 局限性**

局限性包括：①依赖于质量良好的原型，标签噪声或严重不平衡会影响效果；②单一原型可能不足以捕捉多模态类别；③在大规模标签场景下无法超越传统监督投影；④对显著分布漂移不具备自适应能力，需周期性更新或主动学习；⑤对大 K 召回效果提升有限，需结合白化等方法。

---

## 274. UnWEIRDing Peer Review in Human Computer Interaction

**arXiv ID:** 2601.17476 | [PDF](https://arxiv.org/pdf/2601.17476v1)

**作者:** Hellina Hailu Nigatu `[一作]` (University of California Berkeley), Syed Ishtiaque Ahmed `[通讯]` (University of Toronto)

**通讯引用:** 3939 | [OpenAlex ID](https://openalex.org/A5089574660)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过四个焦点小组访谈研究全球南方学者在HCI同行评审中遭遇的偏见和压迫，并以认知压迫为视角提出改进措施。

**💡 创新点**

创新在于将认知压迫框架应用于同行评审过程，系统记录作者与评审双方的经验，并提出分层（第一、第二、第三阶）改进建议。

**🔧 技术方法**

采用质性访谈（焦点小组）和主题分析法，对16位参与者的访谈记录进行编码与主题归纳。

**📊 数据集**

使用16名HCI研究者的焦点小组访谈文本，包含约4小时录音、103页转录。

**📈 对比分析**

该研究不涉及量化对照，而是通过理论归纳与参与者反馈进行验证，发现现行评审实践对全球南方研究产生系统性压迫。

**⚠️ 局限性**

局限在于样本规模有限且主要来自全球北方机构，访谈仅用英语，可能遗漏非英语学者的不同经验。

---

## 275. When Seconds Count: Designing Real-Time VR Interventions for Stress Inoculation Training in Novice Physicians

**arXiv ID:** 2601.17458 | [PDF](https://arxiv.org/pdf/2601.17458v1)

**作者:** Shuhao Zhang `[一作]` (ShanghaiTech University), Quan Li `[通讯]` (ShanghaiTech University)

**通讯引用:** 1377 | [OpenAlex ID](https://openalex.org/A5100689370)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了一套基于虚拟现实的、可即时适配的应激干预系统，用于新手医生在手术急诊情境中的训练与评估。

**💡 创新点**

创新点在于将Just‑In‑Time Adaptive Intervention (JITAI) 框架与 Stress Inoculation Training (SIT) 结合，首次提供三类（自调节、程序指导、情绪/感官）实时干预，并通过心理画像实现个性化调节，弥补了现有VR‑SIT 系统缺乏实时支持的不足。

**🔧 技术方法**

采用了 Unreal Engine 5 构建沉浸式手术室，配合 PPG 与 GSR 传感器实现 HR、HRV 与皮肤电反应的实时监测；利用多模态干预模块（呼吸引导灯、压力反馈指示、层级程序提示、环境降噪与虚拟伴随）实现即时干预；使用 JITAI 算法根据生理与行为阈值动态触发不同级别的支持。

**📊 数据集**

数据来源主要为两部分：12名医师参与的需求调研（焦点组与共创原型）与26名实验参与者的实验数据（包括生理信号、任务表现与主观问卷）。未使用公开数据集，全部为本研究自行收集。

**📈 对比分析**

采用双组随机实验（实验组对照组）进行比较，评估指标包括任务完成率、操作时长、关键错误数、恢复时间、NASA‑TLX、SUS 与 IPQ。实验组任务完成率提升至 69.23%（对照组 15.38%），平均操作时长缩短 34%（从 207.31 s 降至 164.17 s），恢复时间平均下降 54%（从 11.19 s 降至 5.09 s）。

**⚠️ 局限性**

局限性：样本量相对有限（26 名参与者），实验仅覆盖单一手术急诊情境，VR 场景与真实手术环境仍存在差距；干预策略在极端突发事件中的适用性与可扩展性待进一步验证。

---

## 276. A new approach for combined model class selection and parameters learning for auto-regressive neural models

**arXiv ID:** 2601.17442 | [PDF](https://arxiv.org/pdf/2601.17442v1)

**作者:** Corrado Sgadari `[一作]` (Politecnico di Milano), Marcello Farina `[通讯]` (Politecnico di Milano)

**通讯引用:** 4045 | [OpenAlex ID](https://openalex.org/A5033116635)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于集合会员识别（Set‑Membership Identification）与集合距离（Set‑Distance）准则的自适应结构选择与参数学习框架，用于非线性自回归外部输入噪声（NARXESN）模型的识别；

**💡 创新点**

创新点在于：①将集合会员方法引入非线性神经网络结构选择，显式考虑测量噪声的上界；②提出集合距离作为模型一致性评估指标，指导前向选择、修剪与数值超参数调优；③采用情景采样（Scenario Approach）在可行参数集合（FPS）内逼近最优参数，降低计算复杂度；

**🔧 技术方法**

技术核心包括：集合会员识别（计算FPS），情景采样、集合距离评估，前向选择与修剪（SDRR准则），数值超参数调优（整数搜索与牛顿法），以及多初始化策略以提升全局性；

**📊 数据集**

实验数据：1）合成NARXESN系统，使用多层伪随机信号（MPRS）激励，噪声为均匀分布；2）真实Wiener‑Hammerstein基准实验，采样188,000点，51.2kHz；在这两组数据上分别进行训练/验证划分；

**📈 对比分析**

与方法比较：对合成系统，所提出算法正确恢复结构，验证集FIT≈95.4%；与传统LS估计相比，FIT提升≈10%，RMSE下降≈10%；与增量网格搜索（遍历n_u=n_z=1…10）相比，网格搜索最高FIT≈88.6%，所提模型FIT≈92.5%，显著优于全局穷举但计算量大；

**⚠️ 局限性**

局限性：①对噪声上界的依赖较强，若估计失误可能导致保守或不一致；②在超参数空间维度极高时，情景采样与前向选择仍可能面临组合爆炸；③当前仅针对SISO或小规模MIMO，扩展到高维输入/输出需进一步研究；

---

## 277. Towards a Declarative Agentic Layer for Intelligent Agents in MCP-Based Server Ecosystems

**arXiv ID:** 2601.17435 | [PDF](https://arxiv.org/pdf/2601.17435v1)

**作者:** Maria Jesus Rodriguez-Sanchez `[一作]`, Kawtar Benghazi `[通讯]` (Universidad de Granada)

**通讯引用:** 485 | [OpenAlex ID](https://openalex.org/A5081920399)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种名为DALIA的声明式架构层，用以在大型语言模型驱动的智能体之间显式链接目标、能力和执行，形成可验证的任务图。

**💡 创新点**

创新点在于构建可扩展的能力语义模型、Agentic Task Discovery Protocol (ATDP)、联邦Agent Directory以及基于声明式能力的确定性任务编排，实现从发现到执行的严格分离。

**🔧 技术方法**

采用MCP协议的扩展、声明式JSON描述、分布式元数据目录和确定性图规划算法；核心技术是结构化能力定义与任务发现。

**📊 数据集**

未使用标准公开数据集，论文通过餐厅预订的演示场景说明体系运行。

**📈 对比分析**

未给出量化对比实验，作者指出相较于传统基于LLM推理的自由对话式编排，DALIA可显著降低幻觉行为、未执行计划与脆弱协调问题，但需进一步实证验证。

**⚠️ 局限性**

局限性包括：需要人工编写和维护能力/任务声明；缺乏大规模实验评估；在高度动态或不完整的能力环境下的适应性尚未验证。

---

## 278. Co-Designing Digital Humans for Online Learning: A Framework for Human-AI Pedagogical Integration

**arXiv ID:** 2601.17434 | [PDF](https://arxiv.org/pdf/2601.17434v1)

**作者:** Xiaokang Lei `[一作]` (Hong Kong University of Science and Technology), Pan Hui `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 20891 | [OpenAlex ID](https://openalex.org/A5029925982)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了首个面向在线学习的数字教师设计与集成框架，结合设计空间分析、学习者问卷与专家共创研讨，指导何时、何种内容以及如何实现数字教师的教学支持。

**💡 创新点**

1）首次系统性地将“何时使用”“教学内容”“设计方式”三大维度与多学科关键因素映射为可操作的框架；2）通过共创流程获得的实践反馈，使框架兼顾教学效果、技术可行性与用户体验；3）强调数字教师与人类教师的协同角色而非替代。

**🔧 技术方法**

使用设计空间分析方法、问卷调查与主题分析、共创研讨与关联图分析，结合大语言模型（LLM）与检索增强生成（RAG）等技术构建的数字教师原型。

**📊 数据集**

文献复盘87篇跨领域论文；132名学习者的问卷数据；18名教育、设计与AI专家的共创研讨记录。

**📈 对比分析**

未给出基准模型的数值对比；通过问卷和专家评估展示框架在可用性、交互性、个性化等维度的优越性，建议后续通过实验验证其对学习成绩与学习体验的提升。

**⚠️ 局限性**

1）缺乏针对特定学习情境的实证验证；2）关键因素如学习者状态的量化指标尚未细化；3）伦理与隐私问题未深入探讨；4）框架主要基于高校与MOOC环境，需扩展至更广泛的教育场景。

---

## 279. Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping

**arXiv ID:** 2601.17467 | [PDF](https://arxiv.org/pdf/2601.17467v1)

**作者:** Jianxiong Zhang `[一作]` (Sichuan University), Xuefeng Du `[通讯]` (Nanyang Technological University)

**通讯引用:** 768 | [OpenAlex ID](https://openalex.org/A5001001983)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用推理轨迹生成的内部状态干预，构造答案一致性对，用于塑造答案嵌入，从而检测大型推理模型的幻觉输出。

**💡 创新点**

首次通过在推理轨迹末端施加微小的潜在干预生成对照答案，利用答案一致性自动生成无监督的训练信号，学习答案稳定性友好的嵌入。

**🔧 技术方法**

潜在干预+对比学习、轻量级线性映射、余弦相似度对比损失、嵌入式检测器（Supervised Probing、CCS、EigenScore、HaloScope）等。

**📊 数据集**

TruthfulQA、TriviaQA、GSM8K、MATH‑500，模型为 Qwen3‑8B/14B 与 DeepSeek‑R1‑Distill‑Llama‑8B/‑Qwen‑14B。

**📈 对比分析**

与无监督与有监督的概率、熵、相似度、显式一致性、专门针对LRM的RHD、RACE、G‑Detector 等方法对比，AUROC提升 10–20%（如 Qwen3‑8B 在 TruthfulQA 上由 66.85% 提升至 86.64%，在其他数据集亦保持领先）。

**⚠️ 局限性**

依赖于推理轨迹的质量和长度，干预强度需手动调节；对极端长或噪声轨迹时效果可能下降；模型内部状态对干预的敏感度因架构不同而异，需针对不同 LRM 重新调参。

---

## 280. Embodiment-Induced Coordination Regimes in Tabular Multi-Agent Q-Learning

**arXiv ID:** 2601.17454 | [PDF](https://arxiv.org/pdf/2601.17454v1)

**作者:** Muhammad Ahmed Atif `[一作]` (Habib University), Muhammad Ebad Atif `[通讯]` (Habib University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在8×8网格的捕食者-猎物环境中，采用速度和耐力等具身约束，对比独立Q学习(IQL)与集中Q学习(CQL)的表现，研究协调结构在不同运动学模式下的影响。

**💡 创新点**

创新点在于使用完全枚举的tabular方法消除函数逼近、部分可观测性等干扰，单独考察协调结构导致的学习动力学差异，揭示在具身约束下中心化学习并非始终有利，并指出角色对齐与运动优势对性能的决定性作用。

**🔧 技术方法**

使用的技术包括：tabular Q学习（IQL与CQL）、基于潜能的奖励塑形、微步速率/耐力消耗模型，以及对抗/合作的多智能体游戏理论框架。

**📊 数据集**

数据集为自构的8×8网格世界，包含两名捕食者与两名猎物，设置三种速度模式（均速、捕食者加速、猎物加速），共12种实验配置，随机种子10个。

**📈 对比分析**

比较方法是基于seed级别的最终1万条训练episode的平均长度、捕食者奖励和猎物奖励，使用Wilcoxon符号秩检验和Cliff’s δ效应量进行统计，结果显示IQL–IQL往往获得最短的捕获时间和最高的捕食者奖励，而混合配置（如IQL–CQL）则出现明显的协调失效，中心化学习优势随运动学模式和角色对齐而波动。

**⚠️ 局限性**

局限性包括：仅研究小规模、完全可观测、tabular设置；未涉及部分可观测性、通信、深度函数逼近；速度和耐力模型过于简化，难以直接推广到更大规模或更真实的具身系统。

---

## 281. DREAM: Dual-Standard Semantic Homogeneity with Dynamic Optimization for Graph Learning with Label Noise

**arXiv ID:** 2601.17449 | [PDF](https://arxiv.org/pdf/2601.17449v1)

**作者:** Yusheng Zhao `[一作]` (Peking University), Ming Zhang `[通讯]` (Peking University)

**通讯引用:** 20831 | [OpenAlex ID](https://openalex.org/A5100447284)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究图学习中的标签噪声问题，提出Dual-Standard Semantic Homogeneity with Dynamic Optimization（DASH）框架，利用节点间的语义与拓扑关系动态评估标签可靠性并在训练中重新加权。

**💡 创新点**

创新点包括：① 双标准Anchor选择（基于语义相似度和图拓扑邻近）构建针对每个目标节点的Anchor集合；② 通过语义同质性得分量化节点可靠性，逼近分布漂移系数；③ 在优化过程中动态重新评估并加权损失，理论上可获得ε误差近似最优参数。

**🔧 技术方法**

使用技术包括：两层GCN网络、余弦相似度与温度缩放计算语义同质性、地理距离（geodesic）衡量拓扑邻近、加权交叉熵损失、Adam优化器以及t‑SNE可视化。

**📊 数据集**

实验数据集涵盖六个公开图数据集：Cora、CiteSeer、PubMed、DBLP、A‑Photo 与 Flickr，包含同质性与异质性图。

**📈 对比分析**

与 Vanilla GCN、LLN 方法（S‑model、Coteaching 等）以及 GLN 方法（NRGNN、RTGNN、CP 等）进行比较，实验表明在三种噪声类型下，DASH 的平均分类准确率显著提升，最高提升约 14.9%（CiteSeer 上），并在噪声率上升时仍保持领先。

**⚠️ 局限性**

局限性包括：对 Anchor 数量（k_P、k_T）和距离阈值（d_max）的超参数敏感；在极大图或噪声率极高的场景下 Anchor 选取可能受限；构建候选集与 Anchor 选择的计算开销在非常稠密图上仍需进一步优化。

---

## 282. Will It Zero-Shot?: Will It Zero-Shot?: Predicting Zero-Shot Classification Performance For Arbitrary Queries

**arXiv ID:** 2601.17535 | [PDF](https://arxiv.org/pdf/2601.17535v1)

**作者:** Kevin Robbins `[一作]` (George Washington University), Robert Pless `[通讯]` (George Washington University)

**通讯引用:** 7561 | [OpenAlex ID](https://openalex.org/A5051490260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于生成图像与文本嵌入一致性评分的零样本分类性能预测方法，帮助非专家评估任意视觉任务的效果。

**💡 创新点**

创新点在于融合生成图像与多模态轮廓/一致性度量，显著提升对零样本准确度的预测精度，并提供交互式可视化反馈。

**🔧 技术方法**

采用CLIP及其变体（SigLIP、FLAVA）、SDXL‑Lightning或DALL‑E 3生成器、GPT‑4o生成文本提示，计算一致性和多模态轮廓分数。

**📊 数据集**

在十个CLIP基准数据集（ImageNet、CIFAR100、Food101、CUB‑200、Flowers‑102、Stanford‑Cars、Oxford‑IIIT Pets、FGVC‑Aircraft、Resisc‑45）以及ObjectNet等数据集上进行评估。

**📈 对比分析**

与仅在生成图像上测量零样本准确度相比，综合得分与真实准确度的Spearman相关系数平均>0.6，明显优于单纯文本评分；在多数数据集上相关系数>0.7。

**⚠️ 局限性**

受限于生成图像的歧义性与真实图像分布差异，某些类别（如Tornado、Apple Pie）预测偏差显著；方法仍需改进对模糊标签和背景上下文的处理。

---

## 283. One-Shot Federated Clustering of Non-Independent Completely Distributed Data

**arXiv ID:** 2601.17512 | [PDF](https://arxiv.org/pdf/2601.17512v1)

**作者:** Yiqun Zhang `[一作]` (Guangdong University of Technology), Haijun Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 23692 | [OpenAlex ID](https://openalex.org/A5100458465)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种单轮联邦聚类框架GOLD，解决非独立完全分布（Non-ICD）数据的聚类问题。

**💡 创新点**

引入了非ICD概念并提出竞争惩罚学习（CPL）在客户端与服务器端实现多粒度分布学习，自动确定合适聚类数。

**🔧 技术方法**

采用竞争惩罚学习、基于 Jensen‑Shannon 的非ICD度量、单轮通信、层次化多粒度聚类（MCPL）以及基于多粒度聚类的表示增强（REMC）等技术。

**📊 数据集**

在10个UCI公开数据集（Ecoli、US、VE、EP、YE、CA、LA、WI、PE、LE）上进行实验。

**📈 对比分析**

与8种SOTA联邦聚类方法（k‑Fed、OSFSC、FedSC、FFCM等）在 Purity、ARI 等指标上进行基准测试，GOLD 在大多数数据集上排名第一或第二，显著优于对手。

**⚠️ 局限性**

仅针对表格型静态数据，缺乏对流式、多模态或高维文本/图像的适应；单轮通信虽低开销，但在极端分布不完整或极大客户端数时仍可能出现性能下降。

---

## 284. MetaWorld: Skill Transfer and Composition in a Hierarchical World Model for Grounding High-Level Instructions

**arXiv ID:** 2601.17507 | [PDF](https://arxiv.org/pdf/2601.17507v1)

**作者:** Yutong Shen `[一作]` (Beijing University of Technology), Tongtong Feng `[通讯]` (Tsinghua University)

**通讯引用:** 104 | [OpenAlex ID](https://openalex.org/A5003240981)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种 MetaWorld 层次化世界模型框架，融合 Vision‑Language 模型语义规划、专家策略迁移和潜在动态模型控制，实现人形机器人在复杂环境下的高效定位与操作。

**💡 创新点**

① 通过 VLM 输出专家权重而非直接动作，解决符号地面化；② 设计动态专家选择与运动先验融合机制，实现实时自适应；③ 将语义规划层、专家迁移层与物理执行层分层耦合，形成可扩展的模块化架构。

**🔧 技术方法**

Vision‑Language 模型、预训练多专家策略库、动态权重融合、TD‑MPC2 潜在动态模型、量化回归与 TD 学习、模型预测控制与强化学习/模仿学习结合。

**📊 数据集**

HumanoidBench 基准（走路、站立、到达、开门等任务）和 AMASS 专家数据集用于构建模仿学习专家策略。

**📈 对比分析**

与 TD‑MPC2、DreamerV3 进行对比实验，评估平均回报；MetaWorld 在四项任务上平均提升 135.6%，跑步任务提升 2456.3%，门开启任务提升 278.3%，显著优于基线。

**⚠️ 局限性**

依赖简单轨迹匹配奖励，缺乏细粒度误差衡量；专家策略选择采用静态加权，缺少智能路由；缺少少样本泛化与新组合任务的适应性；未解决多技能协同梯度干扰。

---

## 285. To Case or Not to Case: An Empirical Study in Learned Sparse Retrieval

**arXiv ID:** 2601.17500 | [PDF](https://arxiv.org/pdf/2601.17500v1)

**作者:** Emmanouil Georgios Lionis `[一作]` (University of Glasgow), Andrew Yates `[通讯]` (Johns Hopkins University)

**通讯引用:** 3047 | [OpenAlex ID](https://openalex.org/A5059489981)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比了cased和uncased预训练模型在Learned Sparse Retrieval（LSR）中的效果，并通过下划线预处理和后处理技术探讨提升cased模型性能的方法。

**💡 创新点**

揭示了模型casing对LSR性能的显著影响，并证明下划线预处理能使cased模型与uncased模型表现相当；同时提供了通过后处理降低FLOPs的可行方案。

**🔧 技术方法**

使用SPLADE框架，基于BERT和DistilBERT的cased/uncased变体，结合下划线预处理、词表限制后处理和cased正则化技术。

**📊 数据集**

使用公开检索基准：MS MARCO Passage、TREC Deep Learning 2019/2020、BEIR集合（含14个子数据集）等。

**📈 对比分析**

在所有基准上，uncased模型整体领先；cased模型加下划线预处理后与uncased几乎持平；后处理可将FLOPs降低约50%，但准确率仅下降<0.2%。

**⚠️ 局限性**

依赖特定词表和预处理步骤，且对领域特定查询（如专业术语）或非下划线文本的适用性仍有限。

---

## 286. Towards Fair Large Language Model-based Recommender Systems without Costly Retraining

**arXiv ID:** 2601.17492 | [PDF](https://arxiv.org/pdf/2601.17492v1)

**作者:** Jin Li `[一作]` (University of Technology Sydney), Fang Chen `[通讯]` (University of Technology Sydney)

**通讯引用:** 22677 | [OpenAlex ID](https://openalex.org/A5100400043)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对大语言模型推荐系统的公平性问题，提出了一种基于机器无学习的快速统一去偏方法 FUDLR。

**💡 创新点**

创新点在于：①设计了一个与偏差类型无关的可学习掩码，能够在不改变模型结构的前提下通过三重目标（公平性、准确性、稀疏性）挑选需要去偏的样本；②利用影响函数实现一次参数更新，避免了昂贵的全模型再训练。

**🔧 技术方法**

主要技术包括可微公平性度量、掩码学习优化、LoRA 适配器、影响函数与 Hessian-向量积近似。

**📊 数据集**

使用了 MovieLens1M（含性别属性）和 Games（亚马逊视频游戏评论）两大真实推荐数据集。

**📈 对比分析**

与传统的 Reweighting、Reranking、CFP、Masking 等去偏基线和未去偏的 BIGRec 进行对比，FUDLR 在保持甚至提升准确率的同时显著降低热门/属性偏差，且运行时间减少约 95%（相较于最慢基线仅 1.5% 的参数更新），在所有测试场景均表现优异。

**⚠️ 局限性**

局限性包括：对 Hessian 逆的近似依赖计算资源；掩码选择可能无法捕捉所有细粒度或交叉偏差；在极大模型或极端稀疏数据时，影响函数估计误差可能放大。

---

## 287. LeanTutor: Towards a Verified AI Mathematical Proof Tutor

**arXiv ID:** 2601.17473 | [PDF](https://arxiv.org/pdf/2601.17473v1)

**作者:** Manooshree Patel `[一作]` (University of California), Gireeja Ranade `[通讯]` (University of California)

**通讯引用:** 1188 | [OpenAlex ID](https://openalex.org/A5055069190)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个结合大型语言模型与Lean定理证明器的数学证明辅导系统LeanTutor，能够自动形式化学生自然语言证明、生成下一步证明策略并给出自然语言反馈。

**💡 创新点**

创新点在于将LLM的自然语言处理能力与Lean的可验证性相结合，提出分步自动形式化和新的相对精确度评价指标，并在专门构造的PeanoBench数据集上实现高质量的自动化辅导。

**🔧 技术方法**

使用大语言模型（LLM）进行自形式化、下一步策略生成与反馈生成，并通过Lean theorem prover 与 LeanInteract 进行编译与检验。

**📊 数据集**

使用自构建的PeanoBench数据集，共371个Peano算术证明，包含自然语言和Lean形式化版本。

**📈 对比分析**

通过与基线模型对比（含/不含参考解），在自动形式化上步进式方法达到了约57% 术语匹配率、30% 正确证明率；在生成反馈时相较基线在准确性与相关性得分提升，整体表现良好。

**⚠️ 局限性**

局限包括对一对一自然语言步长与Lean tactic 的假设、依赖预先手工书写的参考解、数据集规模有限、自动形式化失败时可能产生错误反馈，并未覆盖所有学生错误类型。

---

## 288. PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors

**arXiv ID:** 2601.17470 | [PDF](https://arxiv.org/pdf/2601.17470v1)

**作者:** Chia-Ming Lee `[一作]` (National Yang Ming Chiao Tung University), Chih-Chung Hsu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 2020 | [OpenAlex ID](https://openalex.org/A5007305393)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 PhaSR 的双层物理对齐阴影去除框架，能够在单光和多光源环境下恢复无阴影图像。

**💡 创新点**

创新点包括：① 通过 Gray‑world 归一化和闭式 Log‑Retinex 分解实现全局光照校正的 Physically Aligned Normalization (PAN)；② 通过差分注意力对几何先验（DepthAnything‑V2）与语义先验（DINO‑V2）进行跨模态对齐的 Geometric‑Semantic Rectification Attention (GSRA)，实现局部几何与全局语义的相互调和。

**🔧 技术方法**

采用的技术包括：Gray‑world 颜色归一化、Log‑domain Retinex 分解、DepthAnything‑V2 提取深度/法向先验、DINO‑V2 提取语义特征、Transformer 编码器‑解码器架构以及差分注意力机制。

**📊 数据集**

使用的主要数据集有 ISTD、ISTD+、WSRD+、INS、Ambient6K 等，涵盖室内外单光、间接阴影和多光源环境。

**📈 对比分析**

与 OmniSR、DenseSR、ShadowFormer 等最先进方法在 PSNR/SSIM 指标上均保持竞争甚至领先优势；在 Ambient6K 多光源实验中表现出更高的泛化能力，并且模型推理时间约 87.9 ms，速度较多基线更快。

**⚠️ 局限性**

局限性包括：对暗色物体或高反射表面仍易出现误校正；依赖预训练几何/语义先验，在光照极端或缺乏深度信息的场景中效果可能受限。

---

## 289. Double-Cover-Based Analysis of the Bethe Permanent of Block-Structured Positive Matrices

**arXiv ID:** 2601.17508 | [PDF](https://arxiv.org/pdf/2601.17508v1)

**作者:** Binghong Wu `[一作]` (Chinese University of Hong Kong), Pascal O. Vontobel `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2931 | [OpenAlex ID](https://openalex.org/A5028154201)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了具有非负条目的方阵的永久性，提出了一种可行的近似方法，即Bethe永久性，并通过适当的因子图运行求和-乘法算法进行高效计算。针对块结构矩阵的集合，数值研究了矩阵的永久性与其Bethe永久性之间的比率。

**💡 创新点**

创新点在于通过图覆盖方法解释了块结构矩阵的永久性与Bethe永久性之间比率的集中现象，并量化了这一比率的值，扩展了可分析矩阵的类别。

**🔧 技术方法**

使用了图覆盖方法、正常因子图（NFG）表示法和多变量分析组合（ACSV）等技术。

**📊 数据集**

使用了块结构矩阵的数值实验，具体构造了基于随机生成的B矩阵的块结构矩阵A。

**📈 对比分析**

与现有方法相比，本文通过数值实验验证了Bethe永久性与永久性之间的比率在块结构矩阵中表现出强集中性，且比率接近于理论预测的√(π n/)。

**⚠️ 局限性**

限制在于对Bethe永久性的解析特征难以表征，除了少数特殊情况外，且在某些情况下，Bethe永久性与永久性之间的比率可能存在系统性差距。

---

## 290. BrainDistill: Implantable Motor Decoding with Task-Specific Knowledge Distillation

**arXiv ID:** 2601.17625 | [PDF](https://arxiv.org/pdf/2601.17625v1)

**作者:** Yuhan Xie `[一作]` (Neuro-X Institute), Mahsa Shoaran `[通讯]` (Neuro-X Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一套可植入的运动意图解码系统 BrainDistill，结合了轻量级Transformer解码器和任务特定知识蒸馏，支持少样本自校准与离线训练两种场景。

**💡 创新点**

创新点在于①提出任务特定知识蒸馏（TSKD），通过监督压缩教师嵌入来弥补学生容量差距；②引入任务特定比例（TSR）指标评估投影质量；③针对植入式硬件实现整数化量化训练，学习激活裁剪区间，保证全整数推理。

**🔧 技术方法**

技术包括：连续小波变换（CWT）时频分解+Token化、线性注意力Transformer、线性分类器、TSKD蒸馏框架、可学习裁剪的量化感知训练（QAT）。

**📊 数据集**

使用多种数据集验证：人类ECoG（Human‑C、Human‑D）、猴子ECoG（Monkey‑R）、EEG（BCIC‑2A、BCIC‑2B）、尖峰记录（FALCON‑M1）以及公共基线模型（EEGPT、NDT2、LaBraM）。

**📈 对比分析**

与传统解码器（EEGNet、EEGConformer、ATCNet、CTNet）和基线蒸馏方法（KD、SimKD、VkD、RdimKD、TOFD、TED）相比，BrainDistill 在多项任务中均取得显著提升；TSKD 在少样本校准下比其他蒸馏方式提高 5–15% 的 F1/Recall，量化模型在保持 <3% 性能损失的前提下，功耗下降超过 3 倍。

**⚠️ 局限性**

局限性包括：仍需教师模型支持；在极端低数据或高度不平衡任务中蒸馏效果有限；量化模型在极低功耗环境下的硬件实现仍需进一步验证；未在真实植入实验中评估长期稳定性与安全性。

---

## 291. Agentic Search in the Wild: Intents and Trajectory Dynamics from 14M+ Real Search Requests

**arXiv ID:** 2601.17617 | [PDF](https://arxiv.org/pdf/2601.17617v1)

**作者:** Jingjie Ning `[一作]` (Carnegie Mellon University), Chenyan Xiong `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4756 | [OpenAlex ID](https://openalex.org/A5102363883)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对14.44M条DeepResearchGym日志进行大规模行为分析，构建会话和轨迹级别的意图与查询改写标签，并提出Context-driven Term Adoption Rate (CTAR) 指标来度量检索证据在多步查询中的使用情况。

**💡 创新点**

①引入CTAR量化新词与检索证据的可追溯性；②使用LLM进行会话意图与轨迹类型标注；③基于真实开放API日志的全量分析，首次揭示Agentic Search的意图驱动行为模式、循环停滞现象及跨步证据整合。

**🔧 技术方法**

LLM-as-a-judge标注、语义连续性+时间阈值会话化、DRGym API重放构造证据、余弦相似度/词汇重叠度量、CTAR计算、统计分析与可视化。

**📊 数据集**

DeepResearchGym（DRGym）日志：2025年6–12月共14.44M请求，3.97M会话，覆盖多语言、不同国家客户端，检索语料库为ClueWeb22-A-EN与FineWeb。

**📈 对比分析**

通过对不同意图（Declarative、Procedural、Reasoning）和轨迹类型（Specialization、Generalization、Exploration、Repetition）的检索深度、查询长度、相似度、结果重叠及CTAR等指标进行对比，评估Agentic Search的效率与证据利用；结果显示约54%新词在累计证据中可追溯，重复率高，验证了行为差异与证据依赖。

**⚠️ 局限性**

①CTAR仅基于词汇匹配，忽略语义同义；②日志缺乏显式点击/反馈，无法完整追踪agent实际使用的上下文；③仅分析DRGym环境，泛化性受限；④未直接关联查询行为与答案质量，缺乏最终性能评估。

---

## 292. AlignUI: A Method for Designing LLM-Generated UIs Aligned with User Preferences

**arXiv ID:** 2601.17614 | [PDF](https://arxiv.org/pdf/2601.17614v1)

**作者:** Yimeng Liu `[一作]` (University of California), Chang Xiao `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种利用用户偏好数据集来引导大语言模型（LLM）生成与用户任务和偏好一致的用户界面（UI）的方法。

**💡 创新点**

创新点在于通过多阶段的推理流程（首先匹配相似任务，再匹配用户偏好，最后生成 UI 控件）让 LLM 在 UI 设计过程中能够主动参考并遵循用户偏好，显著提升 UI 与用户期望的契合度。

**🔧 技术方法**

核心技术包括：大语言模型（如 GPT 系列）进行 UI 生成与推理；多阶段推理框架；基于用户偏好数据集的指导策略。

**📊 数据集**

使用了由 50 名普通用户众包得到的 720 条 UI 控件偏好数据集，数据涵盖 8 个图像编辑任务，每个任务包含多种用户偏好及对应的 UI 控件。

**📈 对比分析**

通过在 6 个未见过的任务上生成 UI 并对 72 名额外普通用户进行使用者研究进行评估；结果显示生成的 UI 在多维度上与用户偏好高度吻合，性能优于传统无偏好指导的 LLM UI 生成方法。

**⚠️ 局限性**

局限性包括：数据集仅覆盖图像编辑任务，难以直接推广到其他领域；样本规模（50 份用户反馈）有限，可能影响模型对多样化偏好的覆盖；方法依赖 LLM 的推理质量，若模型理解失误可能导致 UI 不符合预期。

---

## 293. What Language Models Know But Don't Say: Non-Generative Prior Extraction for Generalization

**arXiv ID:** 2601.17609 | [PDF](https://arxiv.org/pdf/2601.17609v1)

**作者:** Sara Rezaeimanesh `[一作]` (Michigan State University), Mohammad M. Ghassemi `[通讯]` (Michigan State University)

**通讯引用:** 9538 | [OpenAlex ID](https://openalex.org/A5076266282)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 LoID 方法，通过在大语言模型中对 token 级 logits 进行定向探测，直接提取特征先验分布，用于 Bayesian Logistic Regression，帮助小样本 OOD 数据集的泛化。

**💡 创新点**

创新点在于使用 deterministic 的 logit 差值而非生成式采样提取先验，降低了不确定性与计算成本，并提供可复现的先验推断流程。

**🔧 技术方法**

技术手段包括基于 token 级 logits 的正负关系探测、均值与方差估计生成高斯先验、HMC/NUTS 后验采样，以及多句式 paraphrase 以量化不确定性。

**📊 数据集**

实验使用十个公开 tabular 数据集（Heart Disease、Liver Disease、Blood Donation、Room Occupancy、Diabetes、Pima Diabetes、Give Me Some Credit、Bank Marketing、Adult Income、Jungle Chess），构造 covariate‑shift 的 OOD split。

**📈 对比分析**

在与 OOD‑LR、AutoElicit、LLMProcesses、oracle upper‑bound 比较时，LoID 在 8/10 数据集上取得最高 AUC，平均关闭 59% 的性能缺口，并在计算效率上比 AutoElicit 提升约 1.4‑1.5 倍。

**⚠️ 局限性**

主要限制包括仅能捕获二元正负关系，需访问模型 token‑level logits（对闭源 API 不友好），以及对超参数（α、γ）仍需手工调优。

---

## 294. Human-Aligned Enhancement of Programming Answers with LLMs Guided by User Feedback

**arXiv ID:** 2601.17604 | [PDF](https://arxiv.org/pdf/2601.17604v1)

**作者:** Suborno Deb Bappon `[一作]` (University of Saskatchewan), Kevin Schneider `[通讯]` (University of Saskatchewan)

**通讯引用:** 3891 | [OpenAlex ID](https://openalex.org/A5089178328)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Stack Overflow 上利用 LLM 自动识别并融合用户评论来改进已有答案，并实现了一个名为 AUTOCOMBAT 的工具

**💡 创新点**

提出了从评论线程自动识别改进需求、结合问题上下文进行答案重写的端到端方法，并创建了 ReSOlve 基准数据集

**🔧 技术方法**

使用大语言模型（DeepSeek、GPT‑5、Gemini、LLaMA‑4）以及自定义 prompt，结合文本和代码语义评估指标

**📊 数据集**

构建了 790 条 SO 答案与评论的 ReSOlve 数据集（覆盖 Python、Java、C#、JavaScript，按评论数量分四个分位）

**📈 对比分析**

通过标准分类指标、ROUGE/BLEU/METEOR 等语义相似度指标以及人工意图保持评估，DeepSeek 在识别与重写任务上表现最佳，AUTOCOMBAT 在大多数指标上都显著优于基线 SOUP，用户实验显示 84.5% 的开发者愿意使用

**⚠️ 局限性**

在评论量很大或反馈冲突时性能下降；LLaMA 模型效果差；数据集仅涵盖四种语言且样本量有限；工具仅支持 Chrome 扩展，尚未跨平台

---

## 295. From Chains to DAGs: Probing the Graph Structure of Reasoning in LLMs

**arXiv ID:** 2601.17593 | [PDF](https://arxiv.org/pdf/2601.17593v1)

**作者:** Tianjun Zhong `[一作]` (Columbia University), Nima Mesgarani `[通讯]` (Columbia University)

**通讯引用:** 12907 | [OpenAlex ID](https://openalex.org/A5033351155)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“Reasoning DAG Probing”框架，直接从大语言模型隐藏层中提取并重构推理过程的有向无环图结构，

**💡 创新点**

创新点在于将多步推理从传统线性链转变为图结构，并通过浅层线性探针恢复节点深度与节点间距离等图几何属性，

**🔧 技术方法**

采用冻结模型隐藏状态，利用平均池化后的一阶线性探针预测节点深度和两节点距离，并通过阈值方法重构图，

**📊 数据集**

使用 ProofWriter 数据集构建推理 DAG，并在 Qwen3 系列模型上进行实验，

**📈 对比分析**

与节点仅文本、Bag‑of‑Words、标签洗牌等基线相比，探针在中间层能显著恢复图几何（Spearman 相关率和 sink 精度最高），并且模型规模越大、训练细化越多，探针表现越好；图重构 F1 最高在中间层可达 0.6‑0.7 以上，

**⚠️ 局限性**

局限性包括仅针对带有显式证明结构的数据集，探针只能衡量线性可访问性而非因果必要性，且对推理规则类型、边语义等更细粒度属性缺乏覆盖

---

## 296. Sequence Repetition Enhances Token Embeddings and Improves Sequence Labeling with Decoder-only Language Models

**arXiv ID:** 2601.17585 | [PDF](https://arxiv.org/pdf/2601.17585v1)

**作者:** Matija Luka Kukić `[一作]` (University of Zagreb), Jan Šnajder `[通讯]` (University of Zagreb)

**通讯引用:** 2059 | [OpenAlex ID](https://openalex.org/A5067851924)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了通过重复输入序列（Sequence Repetition, SR）来使解码器模型在序列标注任务中获得双向上下文，从而提升token级表示质量和任务性能。

**💡 创新点**

创新点在于提出SR作为一种简单、无需改动模型结构的方式，能够在解码器中实现双向注意力，并证明多次重复比单次更有效，同时通过早期退出减少计算成本。

**🔧 技术方法**

主要技术包括：序列重复输入、注意力矩阵的块结构分析、QLoRA参数高效微调、早期退出（early exit）提取中间层嵌入、与传统去因果掩码（unmasking）和编码器模型的对比实验。

**📊 数据集**

使用四个公开序列标注数据集：CoNLL03（NER）、SemEval-2014 Rest14（情感词条提取与极性）、NLU++（槽标注）、ACE05（事件触发分类）。

**📈 对比分析**

与BERT/RoBERTa等编码器以及去因果掩码解码器进行对比，SR在大多数数据集上均取得最高微F1分数，特别是Mistral‑7B在r=4时平均提升约3-5个百分点；多次重复（r>1）进一步提升，且早期退出在保持性能的同时将推理时间缩短至1/4–1/3。

**⚠️ 局限性**

局限性包括：实验仅在英语数据集上进行，未探索多语言或更大模型；使用固定学习率和超参数，可能未充分利用高重复次数；SR在r>4时计算成本上升且收益递减；未评估更细粒度的层级去掩码配置；未使用k折交叉验证提升鲁棒性。

---

## 297. Sponge Tool Attack: Stealthy Denial-of-Efficiency against Tool-Augmented Agentic Reasoning

**arXiv ID:** 2601.17566 | [PDF](https://arxiv.org/pdf/2601.17566v1)

**作者:** Qi Li `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**通讯引用:** 12599 | [OpenAlex ID](https://openalex.org/A5015574447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对工具增强型 LLM 代理的“海绵工具攻击”（STA），通过重写输入提示在仅有查询权限的条件下，使代理在保持任务语义的前提下，显著增加工具调用次数，产生计算成本膨胀。

**💡 创新点**

创新点在于：①发现工具调用过程存在新的攻击面——Denial-of-Efficiency；②设计了基于多 LLM 协作的重写、评估、策略诱导循环，并在离线阶段构建可迁移的重写策略库；③在保持语义相似度的前提下，系统性地诱导代理产生冗余推理步骤。

**🔧 技术方法**

主要技术包括：多模态提示重写器、质量评判模型、策略诱导器；使用 LLM（如 Qwen3-VL、Qwen2-VL、Gemma-3 等）和 VLM 进行重写与评估；通过语义相似度度量（MiniLM-L6）和奖励函数（step 提升与语义保持平衡）实现策略优化；离线策略库的构建与在线查询式重写。

**📊 数据集**

实验数据集覆盖 13 个 benchmark，涵盖通用推理、数学、科学、医学和代理推理，共 5 个类别，使用 12 种工具（搜索、Python 解释器、图像描述器、对象检测等），共 1,775 条评测样本。

**📈 对比分析**

与 4 个主流代理框架（AutoGen、GPT-Functions、LangChain、OctoTools）以及 6 个 LLM 模型（4 开源 2 API）在 2 种工具调用预算（15 与 40）下进行对比。结果显示 STA 在几乎所有模型和任务上均显著提升工具调用步数（Δ Steps>1），同时语义相似度保持在高水平；攻击奖励大幅提升，且对任务准确率影响较小。对比实验证明 STA 的普适性与可迁移性。

**⚠️ 局限性**

局限性包括：①攻击效果受限于代理对工具调用的预算上限；②重写策略需要先收集探测数据并离线构建，增加前期工作量；③在极度受限的工具调用或高度安全的代理中，攻击成功率可能降低；④攻击仅针对工具调用层面，未考虑对更深层模型或内存的干扰。

---

## 298. Sparse RBF Networks for PDEs and nonlocal equations: function space theory, operator calculus, and training algorithms

**arXiv ID:** 2601.17562 | [PDF](https://arxiv.org/pdf/2601.17562v1)

**作者:** Zihan Shao `[一作]` (University of California), Xiaochuan Tian `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了可自适应稀疏径向基函数网络（SparseRBFnet）来求解非线性偏微分方程，并给出了其理论性质与数值算法。

**💡 创新点**

创新点在于：①在Besov空间框架下证明无论核函数如何，网络生成的解空间等价；②利用核的显式结构实现整数阶及非局部（分数拉普拉斯）算子的准解析评估；③设计三阶段自适应训练策略，阐明增宽、二阶优化、梯度提升等模块对稀疏性与精度的影响。

**🔧 技术方法**

主要技术包括：可变半径RBF网络、L1稀疏正则化、半光滑Gauss–Newton二阶优化、梯度提升式贪婪插入、分数拉普拉斯的Hankel积分实现、各类核（高斯、Matérn、反多二次、Wendland）以及可变矩阵半径的各向异性核。

**📊 数据集**

使用合成 PDE 数据集：4 维双线性、分数泊松、粘性 Eikonal 等问题的离散碰撞点（均匀网格或随机采样），不依赖真实公开数据集。

**📈 对比分析**

与固定宽度 RBF、随机特征、深度 PINN 等方法对比，SparseRBFnet 在高阶、非局部算子场景下保持更高精度且稀疏表示更少；在 4D 双线性问题上相同误差下需更少核；在分数泊松问题中误差随阶数下降而增大，且网络仍能保持较低误差。

**⚠️ 局限性**

局限性包括：内核参数（中心、尺度）优化在高维时计算量与非凸性显著增长；各向异性核参数化导致参数量激增，影响速度；理论上贪婪训练的收敛性与解析可行性仍待进一步证明。

---

## 299. Saliency Driven Imagery Preprocessing for Efficient Compression -- Industrial Paper

**arXiv ID:** 2601.17555 | [PDF](https://arxiv.org/pdf/2601.17555v1)

**作者:** Justin Downes `[一作]` (Amazon Web Services), Anthony Chen `[通讯]` (Amazon Web Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `e0540dec-d77f-42db-94ae-d039248f6393` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于显著性掩模的可变宽度高斯模糊预处理，随后使用传统压缩算法实现卫星图像的可变速率压缩。

**💡 创新点**

创新点在于将显著性信息映射到多尺度平滑核，实现像素级信息削减，并通过预处理显著提升压缩率和下游任务性能。

**🔧 技术方法**

使用高斯平滑核、显著性掩模、JPEG2000、BPG、ImageMagick、Faster‑RCNN、Detectron2等技术。

**📊 数据集**

实验数据集包括RarePlanes、USGS Landsat、Sentinel‑2和SpaceNet（3个 AOI）。

**📈 对比分析**

通过对比均匀模糊与显著性驱动模糊的MSE、bpp降低率和目标检测AP，发现显著性驱动模糊可在不牺牲目标区域质量的前提下，平均降低约45% 的存储率并显著提升检测AP（最高提升≈43%）。

**⚠️ 局限性**

局限性包括显著性掩模需先验知识、结果对图像内容高度敏感、仅在合成掩模下验证，缺乏真实场景的显著性评估。

---

## 300. Deep Intrinsic Surprise-Regularized Control (DISRC): A Biologically Inspired Mechanism for Efficient Deep Q-Learning in Sparse Environments

**arXiv ID:** 2601.17598 | [PDF](https://arxiv.org/pdf/2601.17598v1)

**作者:** Yash Kini `[一作]` (James Madison High School), Shreya Polavarapu `[通讯]` (Northview High School)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了DISRC机制，在DQN框架中通过对观测进行编码并计算潜在空间中的惊讶值来动态调节Q值更新，从而提升稀疏奖励环境下的学习效率和稳定性。

**💡 创新点**

核心创新是将生物学启发的惊讶信号作为全局学习率调节因子，既在早期探索阶段增强可塑性，又在后期逐步收敛；这一机制与传统的好奇驱动或经验回放截然不同。

**🔧 技术方法**

采用了LayerNorm编码器、潜在空间惊讶计算、TD误差加权的Q更新缩放，以及Soft Target、经验回放、Adam优化等标准DQN技术。

**📊 数据集**

在MiniGrid的两个稀疏奖励任务——DoorKey-8x8-v0和LavaCrossingS9N1-v0——上进行实验。

**📈 对比分析**

与标准DQN基线在相同超参数下对比，DISRC在DoorKey中第一个成功 episode 的平均 episode 数减少 33%（79 vs 118），奖励标准差更小（0.25 vs 0.34），AUC 更高（596.42 vs 534.90）；在LavaCrossing中虽收敛稍慢但最终平均奖励和AUC 均略高（0.95 vs 0.93，957.04 vs 934.82）。

**⚠️ 局限性**

主要局限包括对潜在空间编码质量与惊讶估计的高度依赖、需要额外的超参数调优、计算开销增加，以及目前仅在小规模、部分可观测的MiniGrid环境中验证，尚未在更大、更复杂的任务上测试。

---

## 301. Learning to Ideate for Machine Learning Engineering Agents

**arXiv ID:** 2601.17596 | [PDF](https://arxiv.org/pdf/2601.17596v1)

**作者:** Yunxiang Zhang `[一作]` (University of Michigan), Lin Lee Cheong `[通讯]` (AWS AI Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种双代理框架，将机器学习工程任务的高层想法生成（Ideator）与底层实现（Implementer）分离；

**💡 创新点**

创新点在于通过引入可请求的“seek_help”动作实现动态协作，并利用执行反馈奖励对Ideator进行强化学习，显著提升思路质量；

**🔧 技术方法**

使用大型语言模型（如Qwen3-8B、Claude Sonnet 3.5）作为代理，CodeAct执行框架，GRPO算法进行RL训练，奖励机制基于一次性代码执行的性能提升；

**📊 数据集**

数据集为MLE-Bench，训练阶段从10个Kaggle任务中采集约1K状态样本，评估阶段使用剩余51个任务；

**📈 对比分析**

与单代理实现基线相比，提示式Ideator已提升Avg@3/Best@3；RL‑训练的Qwen3‑8B Ideator相较未训练版本提升约11.5%，并超过Claude Sonnet 3.5；在所有评测任务上表现均显著优于基线；

**⚠️ 局限性**

局限性包括：双代理系统推理成本更高；RL训练需要完整模型训练执行，耗费大量GPU资源；需要长上下文来提供完整轨迹，未来可通过摘要或代理奖励模型改进。

---

## 302. Ethical Risk Assessment of the Data Harnessing Process of LLM supported on Consensus of Well-known Multi-Ethical Frameworks

**arXiv ID:** 2601.17540 | [PDF](https://arxiv.org/pdf/2601.17540v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 303. Intelligence Requires Grounding But Not Embodiment

**arXiv ID:** 2601.17588 | [PDF](https://arxiv.org/pdf/2601.17588v1)

**作者:** Marcus Ma `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**通讯引用:** 30339 | [OpenAlex ID](https://openalex.org/A5010028928)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论证智能不需要具身，只需通过 grounding 赋予符号意义即可实现四项智能属性（动机、预测、因果理解、经验学习）

**💡 创新点**

提出 grounding 是智能的核心需求，区别于传统的具身必要性观念，并系统梳理了四项智能属性对 grounding 的依赖关系

**🔧 技术方法**

主要采用理论推理、文献综述与概念定义，未涉及具体算法实现

**📊 数据集**

未使用任何实验数据集

**📈 对比分析**

本文无实验对比，所有论证基于理论分析和已有研究综述

**⚠️ 局限性**

受限于对“智能”与“grounding”定义的主观解释，缺乏实证验证与实验支持

---

## 304. Improving User Privacy in Personalized Generation: Client-Side Retrieval-Augmented Modification of Server-Side Generated Speculations

**arXiv ID:** 2601.17569 | [PDF](https://arxiv.org/pdf/2601.17569v1)

**作者:** Alireza Salemi `[一作]` (University of Massachusetts Amherst), Hamed Zamani `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 3773 | [OpenAlex ID](https://openalex.org/A5101457713)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种交互式框架，利用服务器端大型 LLM 生成草稿，客户端小型 LLM 在保留用户隐私的前提下对草稿进行验证与修正，从而实现个性化生成。

**💡 创新点**

核心创新是将服务器端草稿生成与客户端基于检索的验证相结合的 speculate‑verify‑correct 过程；该过程只暴露极少的已校正 token，几乎不泄露用户个人信息，同时保持了大模型的生成质量。

**🔧 技术方法**

技术手段包括：检索增强生成（RAG）、检索器 (Contriever) 获取用户相关上下文、PII 检测与屏蔽、服务器端采样 k 个草稿 token、客户端根据个人上下文计算 token 概率并与阈值比较、可调阈值 τ 与 k、prompt engineering 与交互式生成。

**📊 数据集**

使用 LaMP‑QA 基准（包含 Art & Entertainment、Lifestyle & Personal Development、Society & Culture 三个子数据集），每个子集都提供用户历史、意图与个性化评价标准。

**📈 对比分析**

与四类基线对比：非个性化服务器、非个性化 Speculative Decoding、客户端 RAG‑个性化以及“泄露”上界 RAG‑个性化；在 LaMP‑QA 上，方法平均提升 7.4%–9% 以上，并获得 90.3%–95.7% 的上界性能；隐私攻击（linkability 与属性推断）仅增加 1.5%–3.5% 的泄露率，远低于直接暴露全档案的方案。

**⚠️ 局限性**

局限性：仍存在微量隐私泄露，需在服务器端保持大模型（计算与网络开销大），对阈值 τ 与 k 的调优敏感；客户端模型规模越大性能越好，但会增加本地内存与延迟；适用场景仍受网络连接与服务器响应速度限制。

---

## 305. Cognitive Platform Engineering for Autonomous Cloud Operations

**arXiv ID:** 2601.17542 | [PDF](https://arxiv.org/pdf/2601.17542v1)

**作者:** Vinoth Punniyamoorthy `[一作]` (IEEE), Durgaraman Maruthavanan `[通讯]` (IEEE)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过在云平台生命周期中嵌入感知、推理与自治行动，提出并实现了四平面认知平台工程（CPE）参考架构，并在 Kubernetes + Terraform + OPA 环境下进行原型部署与评估。

**💡 创新点**

创新点在于：① 统一闭环 Sense–Reason–Act 反馈循环，将 AI/ML 与 DevOps 自动化深度融合；② 四平面架构（Data、Intelligence、Control、Experience）实现数据驱动的自愈与合规治理；③ 通过 LLM、强化学习等前沿技术提升推理与决策质量。

**🔧 技术方法**

使用的技术包括：Kubernetes、Terraform、Open Policy Agent (OPA)、Prometheus、Grafana、OpenTelemetry、Kafka/NATS 事件总线、PyOD IsolationForest（异常检测）、LLM（诊断与摘要）及未来研究方向的强化学习框架。

**📊 数据集**

数据来源主要是实时运维遥测（CPU、内存、延迟、错误率、日志、追踪）以及基于真实与合成工作负载生成的负载轨迹，未使用公开传统数据集而是内部生成的运维日志与指标流。

**📈 对比分析**

方法：对比传统 DevOps 基线与 CPE 环境，在五次实验中分别记录 MTTR、资源效率 (CPU/RPS)、合规违规率；结果显示 CPE 将 MTTR 降低 31.7%，资源效率提升 18.2%，合规违规率下降 92.9%，且统计检验表明差异显著。

**⚠️ 局限性**

局限性包括：① 依赖高质量、完整的遥测数据，缺失或噪声会导致误判；② AI 推理模型需要持续训练与解释，存在偏差与安全审计挑战；③ 人工监督仍不可或缺，尤其是高影响决策；④ 评估主要在受控实验环境，跨云、多租户的泛化能力尚待验证。

---

## 306. Variants of Higher-Dimensional Automata

**arXiv ID:** 2601.17537 | [PDF](https://arxiv.org/pdf/2601.17537v1)

**作者:** Hugo Bazille `[一作]` (EPITA Research Laboratory), Krzysztof Ziemiański `[通讯]` (University of Warsaw)

**通讯引用:** 116 | [OpenAlex ID](https://openalex.org/A5064214731)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文对高维自动机（HDA）及其多种弱化变体（如带接口的HDA、部分HDA、ST-自动机、关系HDA、锥形HDA等）进行系统整理，并阐述它们之间的翻译关系与语言等价性；

**💡 创新点**

主要创新在于①将所有 HDA 变体统一为预纤维形式并构建对应的翻译图谱；②发现语言层面只产生两类：闭包下的 HDA/iHDA 及其余变体；③提出部分 HDA 的 Kleene 定理与确定化方法，突破了传统 HDA 的不可确定性；③引入锥形 HDA、广义 ST‑自动机与 P‑自动机等新模型；

**🔧 技术方法**

采用预纤维（presheaf）理论、lax/strict 结构、稀疏路径分析、稠密化/稀疏化操作、稠密化、以及经典 Thompson 算法等方法，构造对应的语法与构造证明；

**📊 数据集**

无实验数据，论文完全基于形式化定义与理论证明；

**📈 对比分析**

通过构造性的翻译与证明，展示各模型在语言表达能力上的等价或包含关系；在确定化方面，展示了从任意部分 HDA 构造可决定 pHDA 的方法；

**⚠️ 局限性**

限制主要包括：部分 HDA 的几何结构不如传统 HDA 直观（可“破碎”）；部分 HDA 仍存在非可确定化的变体（如 spHDA、srHDA），且确定化条件（如交换不变性）仍待进一步研究；

---

## 307. A Thermodynamic Theory of Learning I: Irreversible Ensemble Transport and Epistemic Costs

**arXiv ID:** 2601.17607 | [PDF](https://arxiv.org/pdf/2601.17607v1)

**作者:** Daisuke Okanohara `[一作]` (Preferred Networks), Daisuke Okanohara `[通讯]` (Preferred Networks)

**通讯引用:** 1028 | [OpenAlex ID](https://openalex.org/A5056883370)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出学习是有限时间内不可逆的过程，并通过熵产生与自由能的关系推导出经验速度限制。

**💡 创新点**

将学习视作概率分布的输运过程，引入熵产生、自由能等热力学概念，并得到与Wasserstein距离相关的经验速度限制。

**🔧 技术方法**

利用概率输运理论、Wasserstein距离、Fokker-Planck动力学以及自由能分析等技术。

**📊 数据集**

无实验数据集。

**📈 对比分析**

未进行实验比较，全部为理论推导与数学证明。

**⚠️ 局限性**

仅限于理论框架，假设高斯噪声等简化，未验证对实际算法或数据集的适用性。

---

## 308. Why They Link: An Intent Taxonomy for Including Hyperlinks in Social Posts

**arXiv ID:** 2601.17601 | [PDF](https://arxiv.org/pdf/2601.17601v1)

**作者:** Fangping Lan `[一作]` (Temple University), Eduard C. Dragut `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一套针对社交媒体（以Twitter为例）中URL共享意图的层级分类体系，并通过两项用户研究评估其分布和标注可靠性。

**💡 创新点**

创新点在于：①以读者视角而非作者动机为出发点，专门为URL共享设计意图分类；②采用混合式构建方法，将大规模众包编码与大型语言模型（LLM）相结合，得到既精细又易解释的26类细粒度意图；③将该意图体系应用于微博检索，证明其可提升检索质量。

**🔧 技术方法**

技术手段包括：1）Amazon Mechanical Turk众包标注与文本开放编码；2）人工智能（LLM，GPT‑5）生成类别名称、定义与示例；3）统计分析（Fleiss' κ、Cohen's κ、nDCG/MAP评估）和微博客检索模型（BM25）与意图增强的再排序。

**📊 数据集**

主要数据集：①从Twitter上随机采集的5 万+带URL推文，二次采样2 500条用于构建分类体系；②1 000条推文用于两项标注实验；③TREC Microblog Track 2011数据集用于检索实验。

**📈 对比分析**

与现有意图分类体系对齐后，本文意图分类覆盖所有先前类别并补充了娱乐/幽默类。检索实验显示，在BM25基础上加入URL意图信息后，nDCG@10从0.4166提升至0.4374，MAP从0.4518提升至0.4757，表明意图增强可提升检索效果。

**⚠️ 局限性**

局限性包括：①标注仍受主观解释影响，部分推文（短/回复）存在高不确定率；②LLM生成的示例为人工挑选，缺乏真实数据；③仅针对Twitter平台，泛化到其他社交媒体的适用性待验证。

---

## 309. Split Algorithm in Linear Time for the Vehicle Routing Problem with Simultaneous Pickup and Delivery and Time Windows

**arXiv ID:** 2601.17572 | [PDF](https://arxiv.org/pdf/2601.17572v1)

**作者:** Ethan Gibbons `[一作]` (University of Waterloo), Beatrice M. Ombuki-Berman `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了三种线性时间（Θ(n)）的 Split 算法，分别用于：1）同时包含取件与交付且带时间窗的 VRPSPDTW；2）软容量约束的 VRPSPD；3）软容量约束与时间折叠（time‑warp）罚函数的 VRPTW。

**💡 创新点**

创新点在于：1）将原先仅适用于 CVRP 的线性 Split 推广到更复杂的 VRP 变体，并给出了满足条件的通用证明；2）设计了常数时间的路线可行性评估方法（包括最高负荷点、最迟起始时间等），从而保持线性复杂度；3）在软约束与多重罚函数情况下，构造了分层双端队列结构，能够在单遍历中动态维护最佳前驱。

**🔧 技术方法**

技术主要包括：
- 以原始长路序列构建隐式有向无环图；
- 利用前缀和（C、D、P、S、W 等）实现 O(1) 计算负荷、时间窗满足情况；
- 双端队列（deque）与自定义前驱维护（如 dominates、dominates_h、dominates_a、dominates_αp 等）实现对前驱的剪枝；
- 软约束和罚函数的线性化处理，确保在最优解搜索时可直接使用。

**📊 数据集**

实验使用 105 个从 CVRP 生成的长路实例（规模从几千到上万客户不等），在每个实例上随机生成取件/交付需求、服务时间、时间窗和容量，保证单点可行；所有实例均在同一台 Intel i9‑13900K 机器上评估。

**📈 对比分析**

对比方法：传统的 Bellman‑based Split（Θ(n²)）与新提出的线性 Split；在不同容量和时间窗宽度下测量实际运行时间。实验结果显示：
- VRPSPDTW 的线性 Split 在大多数规模下速度提升约 1.5–3 倍，极端大规模可达数毫秒；
- 软 VRPSPD 与软 VRPTW 的线性 Split 相比其二次版本可提升 10–1000 倍，视实例规模与约束紧张程度而定；
- 随着平均可行路长增加，线性 Split 的优势更加明显。

**⚠️ 局限性**

局限性包括：
- 仍假设时间成本满足三角不等式（欧氏或最短路满足的情况）和单点可行；若不满足则算法失效；
- 对软约束的线性化依赖特定罚函数（α·max(负荷-容量,0) 与 β·time‑warp），其他罚函数可能需要重新设计；
- 算法实现较为复杂，数据结构管理（多队列、指针等）对编程实现难度和维护成本有一定要求。

---

## 310. Quantum-Inspired Episode Selection for Monte Carlo Reinforcement Learning via QUBO Optimization

**arXiv ID:** 2601.17570 | [PDF](https://arxiv.org/pdf/2601.17570v1)

**作者:** Hadi Salloum `[一作]` (Innopolis University), Alexander Gasnikov `[通讯]` (Innopolis University)

**通讯引用:** 2309 | [OpenAlex ID](https://openalex.org/A5028866398)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在蒙特卡罗强化学习中引入基于量子启发式优化的子集筛选方法（MC+QUBO），通过构造QUBO来挑选奖励高且覆盖广的轨迹以加速学习。

**💡 创新点**

创新点在于把轨迹选择问题视作二次无约束二进制优化，利用量子模拟退火（SQA）或分岔模拟（SB）等量子启发式求解器进行子集优化，从而显著降低样本冗余和方差，提高样本效率。

**🔧 技术方法**

采用的技术包括：QUBO建模、量子启发式求解器（SQA、SB）、蒙特卡罗首次访问估值、基于Jaccard相似度的轨迹相似性惩罚、云端量子求解服务。

**📊 数据集**

使用的是有限时限的GridWorld环境，尺寸从3×3到20×20，障碍物密度不同，作为实验数据集。

**📈 对比分析**

与传统全部使用轨迹的 Vanilla MC 进行对比；实验结果显示 MC+QUBO 在所有网格尺寸上都以更少的批次收敛、最终策略回报更高，尤其在大网格（≥10×10）中优势更明显。

**⚠️ 局限性**

局限性包括：仅在离散网格环境验证，缺乏对连续或多智能体任务的测试；轨迹选择权重需要手动调参；QUBO求解仍有通信开销，且量子启发式求解器的性能依赖于云端资源。

---

## 311. Real-Time Trend Prediction via Continually-Aligned LLM Query Generation

**arXiv ID:** 2601.17567 | [PDF](https://arxiv.org/pdf/2601.17567v1)

**作者:** Zijing Hui `[一作]` (Meta Platforms), Chu Wang `[通讯]` (Meta Platforms)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种实时趋势预测框架RTTP，利用持续学习的LLM直接从新闻内容生成搜索查询并结合用户互动权重进行趋势排名。

**💡 创新点**

核心创新在于Mix‑Policy DPO持续学习策略，它将 on‑policy 稳定性与 off‑policy 新颖性相结合，既保持模型原有推理能力，又避免灾难性遗忘。

**🔧 技术方法**

使用的技术包括持续学习LLM（CL‑LLM）、基于用户互动的评分机制、Mix‑Policy DPO 以及对比学习式的查询生成评估。

**📊 数据集**

数据集来源于 Facebook 搜索日志与用户互动数据（含人类标注的趋势标签）以及 MMLU 基准，用于评估生成准确性与推理稳定性。

**📈 对比分析**

与基线 Poisson 模型和传统 SFT 训练方法相比，RTTP 在 top‑500 趋势精确度提升 91.4%（从 41.8% 到 80%），查询生成准确率保持在 90.5% 以上，且在持续训练期间推理准确率下降仅约 5%。

**⚠️ 局限性**

局限性包括：对私有数据的高度依赖导致外部复现困难；在极低搜索量场景下仍可能受限于数据稀缺；以及对高频更新的实时反馈机制和跨平台推广的适用性尚未充分验证。

---

## 312. Towards Generalisable Imitation Learning Through Conditioned Transition Estimation and Online Behaviour Alignment

**arXiv ID:** 2601.17563 | [PDF](https://arxiv.org/pdf/2601.17563v1)

**作者:** Nathan Gavenski `[一作]` (Kings College London), Odinaldo Rodrigues `[通讯]` (Kings College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种完全无监督的观察式模仿学习方法，利用状态转移的因果结构在两阶段训练（重建+对抗）下实现对教师策略的逼近与提升。

**💡 创新点**

创新点：①不依赖动作标签和监督损失，完全通过状态差分自监督；②利用生成模型与策略交替优化获取近似转移函数；③引入对抗细化阶段以提升泛化并避免行为寻求；④整体流程实现了教师性能甚至超越教师的无监督模仿。

**🔧 技术方法**

核心技术：条件生成模型（预测下一状态）、策略网络、递归判别器、无监督重建损失、两阶段交替训练、梯度裁剪与小学习率对抗训练、基于状态差分的对抗判别。

**📊 数据集**

使用 Imitation Datasets v4（MuJoCo 五个任务：Ant、Half Cheetah、Hopper、Swimmer、Inverted Pendulum），在 10,000 个随机种子上进行评估。

**📈 对比分析**

与 BC, GAIfO, CILO, MAHALO, OPOLO 等基线对比，使用平均回报（AER）和归一化性能（Performance）衡量，平均 Performance 达 1.0197（高于教师），CV 最低，显示出更好的稳定性与泛化；在 Half‑Cheetah 等环境中显著超越教师。

**⚠️ 局限性**

局限性：①需要足够多的教师轨迹（如 300 条以上）才能接近或超越教师，样本量增大时可能出现过拟合；②对抗细化阶段仅跑 10 轮，进一步改进有限；③方法基于 MDP 的可分离转移假设，对非马尔可夫或高度噪声的真实环境尚未验证。

---

## 313. AsterNav: Autonomous Aerial Robot Navigation In Darkness Using Passive Computation

**arXiv ID:** 2601.17550 | [PDF](https://arxiv.org/pdf/2601.17550v1)

**作者:** Deepak Singh `[一作]` (Worcester Polytechnic Institute), Nitin J. Sanket `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 433 | [OpenAlex ID](https://openalex.org/A5055752014)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套利用单目IR相机、编码孔径镜头和低功耗结构光实现绝对黑暗环境下无人机自主导航的系统AsterNav。

**💡 创新点**

创新点在于将被动光学计算（编码孔径导致的深度相关散焦模式）与主动结构光结合，利用深度网络AsterNet实现零样本仿真到现实的深度估计，并在小型无人机上实时执行。

**🔧 技术方法**

核心技术包括：编码孔径光学设计、结构光投射、基于深度网络的深度估计（U‑Net+DenseNet121）、自监督不确定度损失、潜在场规划、Onboard Jetson Orin Nano实时推理。

**📊 数据集**

使用的是合成数据集：通过收集多层深度平面上的点扩散函数（PSF）并与MS‑COCO背景合成，生成约10万张训练图像；没有使用任何真实世界深度标注数据。

**📈 对比分析**

与传统的完全开放孔径、针孔孔径以及现有深度模型（DepthPro、MiDaS）对比，在室内外多种低光/绝对黑暗场景中，AsterNav的成功率达95.5%，深度误差约0.17 m（≤2 m范围内），相较于传统方法在相同条件下精度提升约40‑60%。

**⚠️ 局限性**

局限性包括：仅适用于前后距离不超过约2 m的场景；对光源投影与相机的相对姿态有一定鲁棒性但仍受极端姿态偏差影响；低光下的光照不均可能导致深度误差；在强风或姿态不稳定时，飞控模块仍是瓶颈。

---

## 314. Truth-Revealing Participatory Budgeting

**arXiv ID:** 2601.17538 | [PDF](https://arxiv.org/pdf/2601.17538v1)

**作者:** Qishen Han `[一作]` (Rutgers University), Lirong Xia `[通讯]` (Rutgers University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文从信息学视角研究参与式预算（PB），提出了“真相揭示”PB框架，评估了常用PB规则在噪声投票下选取高质量项目的能力，并探讨了投票是否在策略性情境下保持均衡。

**💡 创新点**

创新点在于：①首次将Condorcet陪审团定理的真相揭示理念扩展到多项目预算约束的PB；②给出单一预算单位成本时所有主流规则在样本量足够大时趋近于最优的定理；③在非单位成本情况下给出以成本比α为参数的上界和下界；④提供了策略性投票的贝叶斯纳什均衡必要条件，并证明其极为有限。

**🔧 技术方法**

技术方法包括：信息论（Hoeffding不等式、LLN）证明真相揭示性；概率与组合论构造最坏情境并计算下界；数理游戏理论分析贝叶斯均衡条件；数值实验（随机生成成本、质量、信号）评估规则性能。

**📊 数据集**

使用的是人工生成的合成数据：随机设定项目成本、预算、质量取值（0/1或更广泛区间）和信息结构（信号概率），随后对多组随机实例进行实验。

**📈 对比分析**

与最优解的预期效用比率作为性能指标。实验结果显示：在单位成本情形下，多数规则在人数增大后性能趋于1；在成本比α=5时性能约为0.57以上，远超理论上限；其中MES+AV规则表现最优。性能随α增大而下降。

**⚠️ 局限性**

局限性包括：①假设所有代理人共享相同的效用函数；②仅考虑二元信号和单一投票方式（赞成/否）；③策略性分析仅在最简单情形下得到必要条件，未给出充分条件或更一般化结果；④实验仅基于合成数据，缺乏真实PB案例验证。

---

## 315. Memento: Towards Proactive Visualization of Everyday Memories with Personal Wearable AR Assistant

**arXiv ID:** 2601.17622 | [PDF](https://arxiv.org/pdf/2601.17622v1)

**作者:** Yoonsang Kim `[一作]` (Stony Brook University), Arie E. Kaufman `[通讯]` (Stony Brook University)

**通讯引用:** 11514 | [OpenAlex ID](https://openalex.org/A5039517392)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了基于可穿戴AR设备的主动式语义记忆回溯助手，能够捕捉用户口头查询及其空间、时间、活动语境，将其存为“Referent‑anchored Spatiotemporal Activity Memory”，并在相似情境下主动将信息可视化在对应物体上。

**💡 创新点**

①提出RSAM概念，将查询与物体、空间、时间、活动四维语境绑定；②构建R‑tree与HNSW混合索引，实现高效空间+语义记忆检索；③在可穿戴AR中实现主动式语义提醒而非单纯反应式交互。

**🔧 技术方法**

使用多模态LLM推理（CLIP、两种多模态LLM）、YOLO‑World‑XL开源词汇检测、CLIP嵌入向量检索、R‑tree＋HNSW混合索引、Whisper语音转写、Google Custom Search API、Unity AR Foundation 与 Meta Quest3 HMD 等技术。

**📊 数据集**

使用 Ego4D 数据集评估空间场景与活动语义识别；自采集的用户口语查询与环境日志做记忆存储；CLIP 预训练模型用于文本与图像嵌入。

**📈 对比分析**

空间活动识别采用 CLIP+ResNet‑101，Scene 与 Activity 各 F1=0.82；对象检测 YOLO‑World‑XL AP=34.4，CLIP‑ViT‑L‑14 在 Scene/Activity 上分别取得 Precision≈0.81–0.82、Recall≈0.88–0.89、F1≈0.80–0.82；在用户研究中约 68.9% 记忆被主动召回，用户满意度高，但低光照或运动模糊时性能下降。

**⚠️ 局限性**

主要局限包括：referent 识别误差导致错误召回；过度泛化导致不相关提示；当前硬件（Quest3）不适合长时间佩戴，需更轻量化眼镜；缺乏长期使用评估；隐私担忧对记忆存储与可视化的控制；系统仍需混合式主动/被动触发与更细粒度活动表示。

---

## 316. Reconstructing Protected Biometric Templates from Binary Authentication Results

**arXiv ID:** 2601.17620 | [PDF](https://arxiv.org/pdf/2601.17620v1)

**作者:** Eliron Rahimi `[一作]` (University of Haifa), Orr Dunkelman `[通讯]` (University of Haifa)

**通讯引用:** 6715 | [OpenAlex ID](https://openalex.org/A5065467078)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了针对仅返回二进制认证结果的加密生物识别系统的模板重构攻击，能够在不泄露相似度分数的前提下几乎无误差地恢复受 FHE 保护的面部特征模板并进一步生成高分辨率面部图像。

**💡 创新点**

创新点在于利用二进制反馈构造多点半径等式，克服传统需要相似度分数的限制，且通过精细的二分搜索和线性系统求解，实现了低查询量下近乎完美的重构；同时将重构模板与生成式逆向模型结合，完成端到端的面部图像恢复。

**🔧 技术方法**

核心技术包括全同态加密（CKKS FHE）用于模板保护、代数等式求解（线性/非线性系统）、二分搜索实现半径约束、GAN‑based 生成模型进行模板到图像的逆向重建，以及基准实验中采用的 ArcFace/FaceNet 特征提取器。

**📊 数据集**

实验数据集为 LFW（13,233 张面部图像，5,749 名身份，其中 300 名身份用于系统测试），并使用 ArcFace 与 FaceNet 两种特征提取器。

**📈 对比分析**

与传统基于分数的 hill‑climbing 攻击以及基线平均法相比，所提攻击在相同或更少的查询次数（平均约 10,000 次）下实现了重构误差几乎为零（<7×10⁻⁵），在完整攻击链中成功率超过 98%，显著优于基线且逼近系统 FMR。

**⚠️ 局限性**

局限性包括：需对目标系统阈值有一定估计（FMR 近似）；攻击在高维度下仍需要数千次查询；仅针对面部识别的 FHE 系统，未验证对其他生物特征或非 FHE 方案的通用性；并假设攻击者可完整访问特征提取器。

---

## 317. Split-on-Share: Mixture of Sparse Experts for Task-Agnostic Continual Learning

**arXiv ID:** 2601.17616 | [PDF](https://arxiv.org/pdf/2601.17616v1)

**作者:** Fatema Siddika `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**通讯引用:** 1072 | [OpenAlex ID](https://openalex.org/A5079359777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于稀疏专家混合的持续学习框架SETA，解决LLM在任务序列中面临的可塑性‑稳定性冲突。

**💡 创新点**

创新点在于：①稀疏子空间选择+Split‑on‑Share机制将参数划分为共享专家和唯一专家；②弹性权重锚定保持共享知识稳定；③自适应门控实现任务无关推理。

**🔧 技术方法**

采用稀疏梯度选择、Mixture‑of‑Experts结构、弹性权重锚定（elastic anchoring）、基于logit不变性的门控扩展等技术。

**📊 数据集**

使用多领域持续学习基准（ScienceQA、FOMC、MeetingBank、C‑STANCE、20Minuten、NumGLUE）和通用基准（MMLU、BBH、PIQA）进行评估。

**📈 对比分析**

与10个基于PEFT的基线（Seq‑Train、ER、EWC、GEM、A‑GEM、L2P、PP、I‑LoRA、MTL）对比，SETA在域适应任务上平均提升≈5‑7%准确率，整体累计保持率最高（R_T≈30.5%），平均遗忘量显著降低（F_T≈19.3%）。

**⚠️ 局限性**

局限在于仅在单一LLaMA‑2‑7B上验证；任务序列较短，未考察更大模型或更长序列；共享子空间划分仍为经验性阈值，缺乏理论收敛保证。

---

## 318. Prompt Driven Development with Claude Code: Building a Complete TUI Framework for the Ring Programming Language

**arXiv ID:** 2601.17584 | [PDF](https://arxiv.org/pdf/2601.17584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 319. Home Health System Deployment Experience for Geriatric Care Remote Monitoring

**arXiv ID:** 2601.17608 | [PDF](https://arxiv.org/pdf/2601.17608v1)

**作者:** Dong Yoon Lee `[一作]` (University of California), Shijia Pan `[通讯]` (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

设计并迭代部署了基于振动传感的“老年人4Ms”框架的家居远程监测系统，支持亲属在家远程监护并实时获取可操作的健康信息。

**💡 创新点**

创新点在于将LLM辅助的部署推荐与结构化图谱相结合，实现无专业知识即可完成传感器配置，同时兼顾系统性能与隐私友好；并在硬件层面实现故障容忍的UDP协议与边缘推理。

**🔧 技术方法**

使用ESP32S3+ADS131M02的振动传感硬件、基于Temporal Convolutional Network的边缘模型、UDP+包层奇偶校验网络协议以及LLM（如ChatGPT）进行推荐与交互。

**📊 数据集**

数据集包括三轮部署：第一轮模拟实验室（4名健康青年）产生的视频与振动同步标签；第二、三轮真实居家（两位阿尔茨海默患者）收集的振动数据与30分钟视频验证标签。

**📈 对比分析**

通过t‑SNE可视化验证模型可分离四类关键活动；对比三轮部署的采样率、未被篡改传感器比例与SNR，LLM推荐方案在保持高SNR的同时将未被篡改率提升至约80%，优于传统人工配置。

**⚠️ 局限性**

局限包括：1）对空间信息的依赖仍以文字为主，未能完全消除定位模糊；2）缺乏大规模真实居家长期验证；3）LLM推理对用户隐私和可解释性的进一步评估尚未完成。

---

## 320. Scaling All-to-all Operations Across Emerging Many-Core Supercomputers

**arXiv ID:** 2601.17606 | [PDF](https://arxiv.org/pdf/2601.17606v1)

**作者:** Shannon Kinkead `[一作]` (Sandia National Laboratories), Amanda Bienz `[通讯]` (University of New Mexico)

**通讯引用:** 137 | [OpenAlex ID](https://openalex.org/A5075734006)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

比较多种all-to-all实现，提出两种新算法（多领导+节点感知、局部感知聚合），在Sapphire Rapids和MI‑300A上评测。

**💡 创新点**

结合多领导与节点感知、引入多组局部感知聚合，针对大规模多核节点优化内外节点通信，并实现3倍以上加速。

**🔧 技术方法**

采用MPI子通信、节点感知/多领导/局部感知聚合，配合Pairwise/非阻塞all-to-all底层实现，基于Intel/Cray MPI。

**📊 数据集**

使用各节点不同字节大小（4–4096 B）交换，规模从2到32节点，未使用特定科学数据集。

**📈 对比分析**

通过在多节点实验中与系统MPI、Bruck、层次、节点感知等实现比较，测量时间与吞吐，结果显示新算法在小消息时优于系统MPI，最大可达3×加速。

**⚠️ 局限性**

局部感知聚合未映射至NUMA域，非阻塞实现波动大，对硬件映射和MPI实现依赖较强。

---

## 321. Understanding Transformer Encoder-Decoder Representations through Bernoulli Dropout

**arXiv ID:** 2601.17602 | [PDF](https://arxiv.org/pdf/2601.17602v1)

**作者:** Xuanzhou Chen `[一作]` (Georgia Institute of Technology), Xuanzhou Chen `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5010720801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Transformer模型在编码器-解码器接口上应用伯努利Dropout（即Binary Erasure Channel）对高维嵌入进行稀疏化，并理论推导其对Top‑1预测的影响，同时通过英文-法文翻译任务验证理论结论。

**💡 创新点**

创新点在于将信息论中的二元擦除通道与伯努利Dropout结合，用几何角度分析高维表示在稀疏化下的结构保留；提出“有效稀疏度”与预测置信度的阈值关系，给出Top‑1不变的充分条件。

**🔧 技术方法**

技术包括：Transformer编码器‑解码器架构；伯努利Dropout与Binary Erasure Channel实现；添加高斯噪声（AWGN）作为噪声基线；角度相似性与高维几何分析；BLEU和验证准确率评估；注意力热图可视化。

**📊 数据集**

使用公开的英语‑法语翻译语料库（来自PyTorch教程），句子长度截断至50个token，采用one‑hot编码和teacher forcing。

**📈 对比分析**

与原始Transformer模型对比，实验在不同Dropout保留概率p（0~0.9）下测量验证准确率和BLEU。结果显示，随着p增大，两项指标呈现明显下降，但在中等稀疏度（p≈0.2‑0.3）仍保持可接受性能；训练时间也有5–8%的加速。

**⚠️ 局限性**

局限性包括：实验仅覆盖单一英‑法语任务，未验证在更大规模或其他语言对的泛化；模型只测试了Bernoulli Dropout，未探索更复杂的噪声或编码方式；理论推导依赖“有效稀疏度”假设，实际稀疏度分布可能不完全满足；以及在极高稀疏度下性能急剧下降，需进一步研究更稳健的压缩方法。

---

## 322. A Unified Approach to Concurrent, Parallel Map-Reduce in R using Futures

**arXiv ID:** 2601.17578 | [PDF](https://arxiv.org/pdf/2601.17578v1)

**作者:** Henrik Bengtsson `[一作]` `[通讯]`, Henrik Bengtsson

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

开发了一个名为 futurize() 的统一接口，能够将多种 R map‑reduce 调用（如 lapply、purrr::map、foreach、plyr 等）自动转译为使用 future 生态的并行实现，支持通过管道直接调用；

**💡 创新点**

创新点在于通过源代码级 transpilation 把不同 API 的并行化逻辑统一到一个函数，消除了学习多套并行接口的成本，并将并行选项抽象成统一的 futurize() 参数，提升了可维护性和可迁移性；

**🔧 技术方法**

实现技术包括 R 的非标准评估（NSE）捕获表达式、future、future.apply、furrr、doFuture、R 4.1.0 的管道操作符 `%>%` 以及对工作流程的自动包装和资源管理；

**📊 数据集**

论文未使用特定大规模数据集，而是通过在常用 R 包（如 boot、lme4、glmnet、caret 等）中的典型函数调用，结合示例脚本演示不同 backends（PSOCK、future.batchtools、future.callr 等）的性能；

**📈 对比分析**

通过在同一代码基础上切换为顺序执行与并行执行（使用同一 future 后端），测量 wall‑time，发现并行化显著缩短运行时间，同时保留相同的标准输出、消息、警告和错误；

**⚠️ 局限性**

局限性包括仅支持已实现的 transpiler，无法自动并行自定义复杂嵌套调用；需要安装对应的依赖包（future.apply、furrr、doFuture 等），并且在极端大规模并行时仍受限于后端资源配置。

---

## 323. Status Hierarchies in Language Models

**arXiv ID:** 2601.17577 | [PDF](https://arxiv.org/pdf/2601.17577v1)

**作者:** Emilio Barkett `[一作]` `[通讯]` (Columbia University), Emilio Barkett (Columbia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探究语言模型在多代理情景下是否会根据身份与能力形成等级层级，并通过基于期望状态理论的实验验证其递让行为。

**💡 创新点**

创新点在于把期望状态理论迁移至语言模型，系统评估身份标签与实际能力交互对递让的影响，发现高能力模型在被赋予高身份时递让率显著下降，揭示状态与能力的非加性效应。

**🔧 技术方法**

采用多代理实验设计，使用GPT‑4.1‑nano与GPT‑3.5‑turbo两大模型实例，利用prompt指令呈现不同身份（高级专家 vs. 初级实习）和能力差异。

**📊 数据集**

实验使用IMDB电影评论数据集的500条测试样本作为情感分析任务，作为模型的决策情境。

**📈 对比分析**

对比相同与不同能力、不同身份设定下的递让率和差异；结果显示相同能力下身份差异产生35%递让差异，能力差异产生近40%差异，表明能力差异对递让影响更大。

**⚠️ 局限性**

局限包括：可能因训练数据直接复制已知实验结果；仅研究单一情感分析任务；仅使用两款模型；未检验多轮交互对层级稳定性的影响；未探讨跨文化或性别等多样化身份特征的影响。

---

## 324. Measuring Braking Behavior Using Vehicle Tracking and Camera-to-Satellite Homography Rectification

**arXiv ID:** 2601.17558 | [PDF](https://arxiv.org/pdf/2601.17558v1)

**作者:** J. P. Fleischer `[一作]`, Mohammed Hadi `[通讯]` (Florida International University)

**通讯引用:** 1959 | [OpenAlex ID](https://openalex.org/A5012205543)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

该论文开发并验证了一套基于开源软件的交通摄像头视频分析管线，可通过将摄像头视角与卫星正射影像对齐，实现车辆轨迹、速度、制动事件等指标的提取。

**💡 创新点**

创新点在于利用MAGSAC++求解的稳健地面平面单应矩阵，将摄像头图像与公开卫星地图自动配准，消除了传统摄像头标定的需求，并能在地理坐标系下直接量化车辆运动。

**🔧 技术方法**

技术包括地面平面单应估计（MAGSAC++）、YOLO11深度检测模型、指数移动平均平滑的多目标跟踪、基于ArcGIS REST服务的卫星正射影像获取、Gemma 3视觉语言模型提取时间戳、以及ClickHouse数据库存储与查询。

**📊 数据集**

数据集主要来自佛罗里达州Key West两条信号化交叉口的摄像头实测视频（7 :00‑19 :00 ，2 天）以及公开的ArcGIS卫星影像；检测模型使用预训练COCO权重，无需自行训练。

**📈 对比分析**

通过与传统手工统计对比，系统在实时提取制动事件后得到峰值约57.5 / h（高流量交叉口）和15.5 / h（低流量交叉口），并在空间层面揭示制动发生位置与强度的分布规律；整体处理速度可在GPU服务器上完成数小时视频的批量分析。

**⚠️ 局限性**

限制包括需要人工标注特征点生成单应矩阵，无法处理摄像头位置漂移；对遮挡、恶劣天气下的跟踪鲁棒性不足；仅支持单摄像头单交叉口，缺乏多摄像头融合与实时部署验证。

---

## 325. Correct-by-Construction Vision-based Pose Estimation using Geometric Generative Models

**arXiv ID:** 2601.17556 | [PDF](https://arxiv.org/pdf/2601.17556v1)

**作者:** Ulices Santa Cruz `[一作]` (University of California), Yasser Shoukry `[通讯]` (University of California)

**通讯引用:** 1825 | [OpenAlex ID](https://openalex.org/A5019844918)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于几何生成模型（GGM）的安全可验证视觉姿态估计框架，该框架通过先用已知平面目标的几何与相机参数构造可解析的生成网络（decoder），再用其生成的数据训练姿态估计器（encoder），并在不含杂物的环境中给出确定性误差上界；随后通过可达性分析与空间滤波，将该框架推广到含杂物的真实环境，实现目标检测与姿态估计的多阶段可验证流程。

**💡 创新点**

创新点主要包括：① 将相机成像过程与目标几何解析地嵌入为可训练网络的结构（GGM），从而获得可解析的 Lipschitz 常数；② 通过格点采样与 Lipschitz 约束给出训练后估计器的全局确定性误差上界（无概率保证）；③ 结合可达性分析与空间滤波构建多阶段检测/估计管线，实现在含杂物环境下的可验证姿态估计；④ 通过事件相机真实数据验证了框架在实际场景中的可行性与安全性。

**🔧 技术方法**

核心技术包括：几何生成模型（GGM）与其计算图实现；Lipschitz 连续性分析与误差上界推导；格点采样与可达性分析；前向可达集计算；空间滤波与逻辑运算；事件相机图像预处理；多层全连接网络与正则化训练。

**📊 数据集**

使用的实验数据集包括：① 合成图像（基于已知平面目标的几何模型生成的图像）；② 真实事件相机捕获的三类目标图像：交通标志（STOP sign）、跑道标记、慢速车辆前方标志；③ 通过 Vicon 运动捕捉系统获取的姿态真值。

**📈 对比分析**

与传统深度学习方法相比，本文框架的性能表现为：① 训练后估计器在测试集上的最大误差与理论上界相符，误差均在数十厘米级；② 在含杂物环境中通过空间滤波与可达性检测实现目标成功率接近 100%（在实验中无误检或漏检）；③ 误差上界可根据网格分辨率和 Lipschitz 常数进行调优，提供可调的安全裕度。

**⚠️ 局限性**

主要限制包括：① 仅适用于已知平面几何目标；② 对目标对称性或视角外的情况可能导致 δ‑可辨识性失效；③ 可达性分析与格点采样在高分辨率或高维姿态空间时计算量显著增加；④ 空间滤波对杂物侵入有一定容限，若杂物遮挡严重会影响检测与估计；⑤ 该方法需要先验相机参数与目标几何信息，对动态目标或未知目标不可直接适用。

---

## 326. TOSHFA: A Mobile VR-Based System for Pose-Guided Exercise Rehabilitation for Low Back Pain

**arXiv ID:** 2601.17553 | [PDF](https://arxiv.org/pdf/2601.17553v1)

**作者:** Amin Mohamed `[一作]` (Zewail City of Science and Technology), Ahmad Al-Kabbany `[通讯]` (Arab Academy for Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了基于手机VR、webcam姿态估计的低成本下腰痛康复系统，并在20名受试者中完成可用性与体验评估。

**💡 创新点**

将实时标记无姿态估计与手机VR相结合，实现在家即可操作的即时生物反馈，并通过游戏化机制显著提升用户参与度。

**🔧 技术方法**

使用 MediaPipe 进行姿态估计、Python 后端实现运动分析与 UDP 低延迟通信、Unity 前端渲染手机VR场景，配合纸盒式VR头显。

**📊 数据集**

未使用公开数据集，所有姿态与运动数据均来自实验室内20名受试者的实时采集与日志记录。

**📈 对比分析**

通过 SUS 与 GEQ 量表评估可用性和体验；平均 SUS 47.4（表明可用性仍需改进），GEQ 正面情绪高；系统实现帧率 30–35 FPS、端到端延迟 <100 ms、姿态估计精度 ±2°，与传统桌面康复相比在硬件成本与易用性上具有明显优势。

**⚠️ 局限性**

样本规模有限（20人，年龄 18–23 岁），仅进行单次实验，缺乏长期随访；硬件校准与光照变化可能影响姿态估计准确性；可用性低导致部分用户体验受阻。

---

## 327. GreenServ: Energy-Efficient Context-Aware Dynamic Routing for Multi-Model LLM Inference

**arXiv ID:** 2601.17551 | [PDF](https://arxiv.org/pdf/2601.17551v1)

**作者:** Thomas Ziller `[一作]` (TU Wien), Ivona Brandic `[通讯]` (TU Wien)

**通讯引用:** 10624 | [OpenAlex ID](https://openalex.org/A5009158531)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种动态、基于上下文的多臂赌博机（MAB）框架，用于在多模型LLM池中实时选择最优模型，以在保持准确率的同时降低能耗。

**💡 创新点**

①使用上下文多特征（任务类型、语义聚类、文本复杂度）构造查询上下文；②在路由决策中直接测量GPU能耗，避免代理指标；③实现在线零校准集成新模型，支持动态模型池扩展。

**🔧 技术方法**

上下文多臂赌博机（LinUCB、ϵ‑Greedy、Thompson Sampling），Transformer嵌入、Logistic回归任务分类、在线K‑Means聚类、Flesch阅读难度；GPU功耗监测（NVIDIA NVML）和精确延迟测量。

**📊 数据集**

五个公开基准数据集：MMLU、HellaSwag、Winogrande、GSM8K、CNN/DailyMail；16个公开LLM（Qwen、Mistral、Gemma、Llama、Phi、Yi）。

**📈 对比分析**

与随机、最大模型、最小模型、最高准确率等静态基线，以及ϵ‑Greedy和Thompson Sampling进行对比。实验表明在λ=0.4时，框架在平均准确率提升约22%、能耗降低约31%；在RouterBench上取得最高AIQ 0.637、峰值准确率75.7%。

**⚠️ 局限性**

局限性包括：MAB对奖励漂移适应慢；仅评估具有客观评价指标的任务；依赖特定GPU环境；对特征工程参数（聚类数、复杂度分箱）敏感；未考虑多线程、批处理和排队延迟等实际部署因素。

---

## 328. Prompt Injection Attacks on Agentic Coding Assistants: A Systematic Analysis of Vulnerabilities in Skills, Tools, and Protocol Ecosystems

**arXiv ID:** 2601.17548 | [PDF](https://arxiv.org/pdf/2601.17548v1)

**作者:** Narek Maloyan `[一作]`, Dmitry Namiot `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统综述并量化了 agentic 编码助手（Claude Code、Copilot、Cursor 等）在 Prompt Injection 攻击下的漏洞与防御，提出了三维攻击分类法、首次梳理技能生态链漏洞，并构建了基于防御深度的综合安全框架。

**💡 创新点**

创新点包括：
1) 三维攻击分类法（投递向量、攻击模式、传播行为）统一不同文献的攻击描述；
2) 首次系统化技能与工具生态中的漏洞链，展示具体的 exploit chain；
3) 基于 meta‑analysis 的防御评估，提出多层防御（工具身份签名、能力分级、沙箱、运行时验证、人工介入）框架。

**🔧 技术方法**

所用技术：
- 文献检索与系统综述方法（arXiv/IEEE/ACM/USENIX）
- Meta‑analysis 与统计汇总（attack success, defense bypass rates）
- Benchmark 评估：MCPSecBench、IDEsaster、Nasr 等安全基准
- 对多平台（Claude、Copilot、Cursor、Codex CLI、Gemini CLI）进行对比实验。

**📊 数据集**

使用的数据集与评测集合：
- 78 篇近两年（2024–2025）相关论文
- MCPSecBench（17 类攻击、4 面向）
- IDEsaster（30+ CVE 记录）
- Nasr 等评测集（adaptive 攻击下的防御 bypass 率）

**📈 对比分析**

比较方法与性能：
- 对攻击成功率进行 meta‑analysis，发现 85%+ 的攻击可突破至少一个平台；
- 对防御机制进行横向评估，平均成功率低于 50%，但在适应性攻击下可突破 78%+；
- 多平台对比显示 Copilot 与 Cursor 处于高危区，Claude 处于低危区；
- 性能与可用性权衡通过层级防御模型（沙箱、签名、能力分级）展示。

**⚠️ 局限性**

limitations（局限性）：
- 研究时间窗口短，技术更新迅速可能导致结论过时；
- 多数平台闭源，无法完全了解内部防御细节，仅能做黑盒评估；
- benchmark 可能不覆盖所有真实攻击复杂度，适应性攻击评估有限；
- 主要关注静态防御，对动态学习型防御缺乏评估；
- 公开论文样本可能存在选择偏差，未涵盖未公开的高危攻击。

---

## 329. Push Down Optimization for Distributed Multi Cloud Data Integration

**arXiv ID:** 2601.17546 | [PDF](https://arxiv.org/pdf/2601.17546v1)

**作者:** Ravi Kiran Kodali `[一作]` (Cognizant Technology Solutions), Nachiappan Chockalingam `[通讯]` (IEEE)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在多云环境中使用推送下推优化来改进ETL流水线的设计与执行，结合本地化下推、混合执行和数据联邦等技术，并在AWS Redshift与Google BigQuery的案例中实现。

**💡 创新点**

提出了跨云下推优化的多模式策略（本地化、混合与联邦）以及基于成本与延迟的动态下推决策公式，解决多云SQL异构与数据移动瓶颈。

**🔧 技术方法**

采用了云原生ETL工具（如Glue、Data Factory、Informatica IDMC）、SQL下推、联邦查询（BigQuery Omni、Redshift Spectrum）、成本模型与自适应决策引擎。

**📊 数据集**

以企业业务数据为例，包括存储在AWS Redshift的客户数据与存储在Azure/Google BigQuery的销售/日志数据，进行跨云联邦查询与聚合。

**📈 对比分析**

通过将原始全转移ETL与下推优化后的管道进行对比，测量总运行时、跨云数据量、单引擎运行时和成本；结果显示总运行时间减少35%，跨云数据量减少20%，成本下降18.9%。

**⚠️ 局限性**

受限于跨云SQL语法不兼容、联邦查询性能波动、编排与监控复杂度、以及安全合规控制碎片化，导致下推覆盖范围有限且对工具支持与自动化仍需提升。

---

## 330. Lightspeed Data Compute for the Space Era

**arXiv ID:** 2601.17589 | [PDF](https://arxiv.org/pdf/2601.17589v1)

**作者:** Thomas Sandholm `[一作]` (Research Institutes of Sweden), Paris Carbone `[通讯]` (Royal Institute of Technology)

**通讯引用:** 1821 | [OpenAlex ID](https://openalex.org/A5039432313)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出 SpaceCoMP，一个面向 LEO 卫星星座的三阶段（Collect-Map-Reduce）分布式处理模型，利用光学星间链路实现高带宽内网通信，并通过距离优化路由、双分配匹配调度与中心化 reducer 放置，实现在轨道上完成数据预处理与聚合，显著降低地面下行负载。

**💡 创新点**

创新点包括：① 将 MapReduce 引入空间，新增 Collect 阶段；② 针对卫星轨道几何设计距离感知路由；③ 采用匈牙利算法在双分配图上求最优 Map 任务分配；④ 将 reducer 放在 AOI 中心以最小化聚合成本；⑤ 在仿真中展示多项性能提升，证明上述设计的有效性。

**🔧 技术方法**

主要技术：光学星间链路（ISL）、距离感知路由算法、匈牙利算法（双分配匹配）、基于 Walker Delta 的仿真平台、SGP4 轨道预测、MapReduce 编程模型、cFS/ColonyOS 运行时框架（未来实现方向）。

**📊 数据集**

使用模拟数据集：在 87° 轨道倾角、530 km 高度的 Walker Delta 星座中，AOI 选为美国领土，收集节点与映射节点各占 AOI 的 1/5，假设每个 Collect 任务产生 10 GB 原始影像，聚合压缩因子为 5，未使用真实遥感影像，而是通过仿真生成的空间传感数据。

**📈 对比分析**

评估方法：与最短跳数路由、随机分配、贪心分配及 LOS reducer 等基线进行对比；通过仿真测量传输距离、跳数、映射成本、聚合成本与节点拥塞。结果显示：距离感知路由比基线减少 8–21% 路径距离；双分配匹配在映射分配上比随机提升 61–79%，比贪心提升 18–28%；中心 reducer 放置比 LOS reducer 降低 67–72% 聚合成本。

**⚠️ 局限性**

局限性：仅在仿真环境中验证，未考虑多任务并发、动态负载与故障恢复；假设任务处理时间与链路质量固定；仅支持单一轨道方向（上升或下降）作业，忽略跨轨道混合计算；未深入探讨能源消耗与卫星硬件限制，需在实际平台上进一步验证。

---

## 331. Athena: Synergizing Data Prefetching and Off-Chip Prediction via Online Reinforcement Learning

**arXiv ID:** 2601.17615 | [PDF](https://arxiv.org/pdf/2601.17615v1)

**作者:** Rahul Bera `[一作]` (ETH Zürich), Onur Mutlu `[通讯]` (ETH Zürich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在高性能处理器中，研究者提出了 Athena——一种使用强化学习协调预取器与离线预测器的框架。

**💡 创新点**

创新点在于将两种互补的预取技术视为强化学习问题，引入分离相关与不相关奖励的复合奖励框架，并通过 Q 值驱动预取器的激进度调节。

**🔧 技术方法**

采用了 SARSA 强化学习算法、哈希分层 QVStore、Bloom 过滤器测量预取精度与污染，以及自动设计空间探索调参。

**📊 数据集**

使用了 100 个内存密集型单核工作负载（SPEC CPU 2006/2017、PARSEC、Ligra、CVP）以及 90 个四核/八核混合组合。

**📈 对比分析**

与 Naive、TLP、HPAC、MAB 等基线对比，Athena 在所有缓存设计、预取器、离线预测器和内存带宽配置下平均提升 5–10% IPC，单核最高提升 7.9%，多核也能保持 5–10% 的加速。

**⚠️ 局限性**

局限性包括对超大状态空间的表格存储仍有限，奖励权重需离线调优，且在极端带宽受限或特定工作负载组合下仍可能不如最佳静态配置。

---

## 332. Discovery of Feasible 3D Printing Configurations for Metal Alloys via AI-driven Adaptive Experimental Design

**arXiv ID:** 2601.17587 | [PDF](https://arxiv.org/pdf/2601.17587v1)

**作者:** Azza Fadhel `[一作]` (Washington State University), Jana Doppa `[通讯]` (Washington State University)

**通讯引用:** 2211 | [OpenAlex ID](https://openalex.org/A5055445718)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并部署了一种基于贝叶斯实验设计的 AI 辅助方法（BEAM），用于在低功率（500–1000W）下快速发现 GRCop-42 合金的可行加工参数。

**💡 创新点**

将主动搜索与领域知识结合，形成针对稀疏可行区域的贝叶斯实验设计框架，实现仅 10 次实验即可找到可行参数，显著降低实验成本。

**🔧 技术方法**

贝叶斯实验设计（BEAM），包含概率 kNN 代理模型、主动搜索采样策略以及域约束剪枝。

**📊 数据集**

先前 37 次失败实验数据和低功率下 GRCop-42 的实验结果作为训练样本；随后在不同激光功率水平下进行的 10 次实验。

**📈 对比分析**

与手工实验（37 次尝试无成功）和理想的全搜索（>10⁸ 组合）对比，BEAM 在 10 次实验内找到可行参数，实验效率提升约 99.9999% 以上，成本与时间显著下降。

**⚠️ 局限性**

对单一合金的探索限制，代理模型在稀疏数据下预测不确定，且对不同工艺平台的泛化仍需验证。

---

## 333. Stylizing ViT: Anatomy-Preserving Instance Style Transfer for Domain Generalization

**arXiv ID:** 2601.17586 | [PDF](https://arxiv.org/pdf/2601.17586v1)

**作者:** Sebastian Doerrich `[一作]` (University of Bamberg), Christian Ledig `[通讯]` (University of Bamberg)

**通讯引用:** 23889 | [OpenAlex ID](https://openalex.org/A5016912926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种可在训练和推理时进行风格迁移的 Vision Transformer（Stylizing ViT），用于医学影像数据增强，提升跨域泛化能力。

**💡 创新点**

创新在于将自注意力与跨注意力统一在同一权重共享的注意力块中，实现既保持解剖结构又实现多样化风格的无解码器风格迁移。

**🔧 技术方法**

采用单编码器 Vision Transformer、交叉注意力、MLP 重构、感知损失、VGG19 预训练等技术；在推理阶段用于测试时增强（TTA）。

**📊 数据集**

三组医学影像数据集：Camelyon17‑WILDS（淋巴结 H&E 病灶），Epithelium‑Stroma（乳腺癌/结肠癌组织分类），Fitzpatrick17k（皮肤色调皮肤病图像）。

**📈 对比分析**

与 AdaIN、IEContrAST、SANet、StyTR²、SGViT 等现有风格迁移方法比较，在三组数据集上在风格迁移质量（FID、LPIPS、ArtFID）和分类准确率上均取得领先，Camelyon17‑WILDS 提升约+9% 以上，Epithelium‑Stroma 提升约+13%。

**⚠️ 局限性**

局限在于对光照、皮肤色调等细粒度变化的泛化效果有限，训练时计算开销较大，对极端样本仍可能产生细节失真。

---

## 334. How AI Coding Agents Modify Code: A Large-Scale Study of GitHub Pull Requests

**arXiv ID:** 2601.17581 | [PDF](https://arxiv.org/pdf/2601.17581v1)

**作者:** Daniel Ogenrwot `[一作]` (University of Nevada Las Vegas), John Businge `[通讯]` (University of Nevada Las Vegas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 AI 编码代理生成的 GitHub 拉取请求（PR）与人类 PR 在代码变更结构与描述一致性上的差异，采用大规模数据集进行统计对比。

**💡 创新点**

首次系统对比 Agentic 与 Human PR 在提交次数、文件触及数、行增删量等结构特征以及描述‑diff 的词汇和语义一致性，揭示 Agentic PR 更集中化、提交次数差异最大，且描述与代码的语义一致性略高。

**🔧 技术方法**

使用 TF‑IDF、Okapi BM25、CodeBERT 与 GraphCodeBERT 进行词汇与语义相似度评估，并通过 Mann‑Whitney U 检验与 Cliff’s delta 量化效应大小。

**📊 数据集**

采用 MSR 2026 Mining Challenge 的 AIDev 数据集，包含 24,014 份已合并的 Agentic PR（共 440,295 次提交）与 5,081 份已合并的人类 PR（共 23,242 次提交）。

**📈 对比分析**

通过比较提交次数、文件触及数、行增删量等结构指标，以及相似度分数评估描述‑diff 对齐，发现提交次数差异最大（Cliff’s δ≈0.54，显著），所有 PR 的语义一致性均高于 0.9，Agentic PR 在所有指标上略优。

**⚠️ 局限性**

结果受限于 AIDev 数据集的完整性与 GitHub API 访问限制导致的缺失补丁、以及仅衡量描述与代码的一致性而未考虑代码质量、审查效率等后续影响。

---

## 335. JaxARC: A High-Performance JAX-based Environment for Abstraction and Reasoning Research

**arXiv ID:** 2601.17564 | [PDF](https://arxiv.org/pdf/2601.17564v1)

**作者:** Aadam `[一作]` (Indiana University), Mohamed Abdel-Mottaleb `[通讯]` (Indiana University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

开发了基于JAX的高性能ARC强化学习环境JaxARC。

**💡 创新点**

创新点在于纯函数无状态架构实现巨幅加速，支持多数据集、可配置动作空间和可组合包装器。

**🔧 技术方法**

采用JAX、XLA、PyTree、Hydra配置、纯函数编程以及GPU/TPU并行等技术。

**📊 数据集**

使用了多种ARC数据集，包括MiniARC、FullARC等。

**📈 对比分析**

与ARCLE对比，在CPU、RTX 3090、H100平台上相同批量下分别提升38×至5,439×，峰值吞吐达790M步/秒。

**⚠️ 局限性**

局限在小规模环境下加速有限，并且仍需整合完整基线代理和更丰富的数据集。

---

## 336. Private Iris Recognition with High-Performance FHE

**arXiv ID:** 2601.17561 | [PDF](https://arxiv.org/pdf/2601.17561v1)

**作者:** Jincheol Ha `[一作]` (Cryptolab, Inc.), Damien Stehlé `[通讯]` (Cryptolab, Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实现了一种基于阈值全同态加密（ThFHE）的隐私保护虹膜识别系统，支持在大规模虹膜数据库上批量匹配。

**💡 创新点**

创新点：
• 用 ThFHE 取代传统 2‑out‑3 secret‑sharing MPC，显著降低通信量、提升安全可扩展性；
• 提出了 “folding” 技术，在早期将分数合并以减少大量 bootstrapping；
• 结合 int8 GPU 张量核心与 RGSW‑based 加密矩阵乘法，实现高效的线性代数加速。

**🔧 技术方法**

主要技术：CKKS（ThFHE）方案、RGSW‑based CCMM、int8 cuBLAS 张量核心、Bootstrapping、folding 多项式、分层加密键共享、GPU 集群并行。

**📊 数据集**

使用 ND‑IRIS‑0405 数据集，生成 7·2^14 条虹膜模板（每条 31 个旋转），作为实验数据库。

**📈 对比分析**

性能对比：在 8 张 RTX‑5090 GPU 上，32 眼批量匹配 7·2^14 条模板约 1.8 秒，通信量 <1 KB；相较于 Bloemen 等基于 2‑out‑3 MPC 的方案（≈2 s、≈81 GB/party、≈40 轮通信），ThFHE 在相同时间内大幅降低通信量和轮数。

**⚠️ 局限性**

局限性：
• 仍需大规模 GPU 资源与高密钥尺寸；
• folding 假设要求每槽至多一正分，极端分布下可能失效；
• bootstrapping 成本仍高，参数调优复杂；
• 活跃攻击防护需零知识证明，实验中未实现。

---

## 337. Breaking the Protocol: Security Analysis of the Model Context Protocol Specification and Prompt Injection Vulnerabilities in Tool-Integrated LLM Agents

**arXiv ID:** 2601.17549 | [PDF](https://arxiv.org/pdf/2601.17549v1)

**作者:** Narek Maloyan `[一作]`, Dmitry Namiot `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Model Context Protocol（MCP）进行协议层面的安全分析，识别三类核心漏洞，构建ProtoAmp框架在847个攻击场景下量化MCP对攻击成功率的放大效应，并提出向后兼容的AttestMCP扩展来消除这些漏洞。

**💡 创新点**

创新点在于：①首次从协议规范出发系统性剖析MCP的安全缺陷；②用实验框架量化协议对攻击成功率的放大效应；③设计可无缝升级的AttestMCP协议扩展，实现能力证明、消息签名和服务器间隔离。

**🔧 技术方法**

采用协议安全分析、JSON‑RPC 拦截与记录、加密签名（HMAC‑SHA256）、PKI 证书颁发与验证、时间戳+nonce 重放保护、以及基于现有 LLM 后端（Claude‑3.5‑Sonnet、GPT‑4o、Llama‑3.1‑70B）的推理实验。

**📊 数据集**

使用 847 个攻击案例，覆盖 InjecAgent、AgentDojo、Agent‑SafetyBench 等 benchmark 生成的场景；5 种公开 MCP 服务器实现；并结合三种主流 LLM 后端进行实验。

**📈 对比分析**

与非MCP基线（直接工具调用）对比，MCP 将攻击成功率提升 23–41%；AttestMCP 在全部攻击类型中将总体成功率从 52.8% 降至 12.4%，并在各子类攻击中分别降低 61.5%、72.7%、85.8% 与 83.2%；平均每条消息延迟 8.3 ms（冷启动）/2.4 ms（热缓存），对 LLM 推理时间影响可忽略。

**⚠️ 局限性**

局限性包括：对已被授权但恶意的单服务器内容仍无法完全防御；用户授权弹窗可能被忽略导致安全失效；CA 生态协同与证书吊销仍需进一步落地；实验规模仅覆盖 5 种服务器，生产级多服务器部署可能产生新的风险；AttestMCP 对首接攻击（TOFU）无防御。

---

## 338. CTF for education

**arXiv ID:** 2601.17543 | [PDF](https://arxiv.org/pdf/2601.17543v1)

**作者:** Yi Lyu `[一作]` (University of Wisconsin-Madison), Andy Zhang `[通讯]` (University of Wisconsin-Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估并比较了攻击式、守卫式、Jeopardy式以及游戏化/战术式四种CTF格式在网络安全教育中的有效性与特点。

**💡 创新点**

提出将四种CTF格式综合运用以构建完整的网络安全学习路径，并系统化比较其学习目标、可访问性与难度。

**🔧 技术方法**

基于文献综述、公开CTF平台数据以及对比分析方法，对CTF格式进行定性评估，使用可视化与指标对比进行技术实现。

**📊 数据集**

使用公开CTF平台（如picoCTF、24/7 CTF、Over The Wire、Smash the Stack等）中的挑战题库及其安装、语言与文档信息。

**📈 对比分析**

采用类别对比表和指标（可访问性、主题覆盖、难度梯度、工具支持）进行定性比较，结果显示攻击式与守卫式互补、Jeopardy易用但浅层、游戏化/战术式深度高但难度大。

**⚠️ 局限性**

主要缺乏定量实验与用户反馈，比较主要基于文献和公开信息，未能量化CTF对学习效果的提升。

---

## 339. OTI: A Model-free and Visually Interpretable Measure of Image Attackability

**arXiv ID:** 2601.17536 | [PDF](https://arxiv.org/pdf/2601.17536v1)

**作者:** Jiaming Liang `[一作]` (University of Macau), Chi-Man Pun `[通讯]` (University of Macau)

**通讯引用:** 7959 | [OpenAlex ID](https://openalex.org/A5005772506)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于对象纹理强度（OTI）的无模型、可视化解释的图像攻击可易性评估方法

**💡 创新点**

首次将语义对象面积与纹理强度结合，形成既不依赖模型也能直观解释的攻击可易性指标，并从决策边界和频域两角度理论阐释其有效性

**🔧 技术方法**

使用语义/显著性分割得到对象掩码，采用 Sobel 运算提取纹理，结合 Hadamard 乘积得到 OTI，并在多种攻击（梯度、查询、生成、集成）与任务（分类、分割）下验证

**📊 数据集**

在 ImageNet 验证集（5万张）及其 1K 子集、医学图像分割数据集 Kvasir‑SEG（200张）上进行实验

**📈 对比分析**

与随机抽样和现有模型相关方法（如 IAARS、ZGP）对比，OTI 在单代理、集成、对抗训练模型以及查询攻击等场景中平均提升攻击成功率 10%–13%，并显著降低对抗样本所需扰动幅度

**⚠️ 局限性**

局限于图像域，尚未推广到音频、文本等其他模态

---

## 340. StyleDecoupler: Generalizable Artistic Style Disentanglement

**arXiv ID:** 2601.17697 | [PDF](https://arxiv.org/pdf/2601.17697v1)

**作者:** Zexi Jia `[一作]` (Wechat AI), Jie Zhou `[通讯]` (Wechat AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了StyleDecoupler，通过信息论框架和正交投影将多模态视觉模型中的风格与内容分离。

**💡 创新点**

创新点在于利用单模态模型抑制风格的特性作为内容参考，结合互信息最小化实现无监督风格提取，并且无需微调即可在冻结的VLM上使用。

**🔧 技术方法**

技术包括信息论互信息分离、正交投影、知识蒸馏对齐DINO与CLIP特征，以及基于GPT‑4o的文本描述抽取风格向量。

**📊 数据集**

使用了新构建的WeART大规模艺术品数据集（28万幅作品、152种风格、1556位艺术家）和WikiArt进行评测。

**📈 对比分析**

与现有VLM与专用风格检索模型对比，StyleDecoupler在WeART上获得最高mAP@1（70.3%）且在WikiArt保持竞争力（63.6%），同时在风格聚类与生成模型评估中表现优异。

**⚠️ 局限性**

局限性包括对多模态模型的依赖、无法完全消除风格与内容的潜在共线性，以及在极少量或全新风格上的推断受数据覆盖限制。

---

## 341. Segment Length Matters: A Study of Segment Lengths on Audio Fingerprinting Performance

**arXiv ID:** 2601.17690 | [PDF](https://arxiv.org/pdf/2601.17690v1)

**作者:** Ziling Gong `[一作]`, Nesreen K. Ahmed `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究音频指纹识别任务中不同片段长度对检索性能的影响，并评估大语言模型（LLM）在推荐最优片段长度方面的能力。

**💡 创新点**

①系统性评估多种片段长度（0.5 s、1 s、2 s）在不同查询时长下的性能差异；②提出了基于原有 NAFP 模型的 NAFP+ 变体以支持可变片段长度；③通过对 GPT‑5‑mini、Gemini‑2.5‑flash、Claude‑Sonnet‑4.5 等 LLM 的五种提示对比，验证 LLM 推荐片段长度与实验结果的一致性。

**🔧 技术方法**

使用神经网络音频指纹生成器 NAFP 及其改进版 NAFP+；对音频做 Mel 频谱转换、卷积块处理和对比学习训练；利用 Top‑K Exact 与 Top‑K Near Hit 评估指标；对 LLM 进行提示工程并收集其推荐结果。

**📊 数据集**

基于 Free Music Archive（FMA）音乐数据集，包含 10,000 条训练音频、500 条参考音频（用于生成查询）和 9,978 条干扰音频，所有音频长度约为 30 s。

**📈 对比分析**

通过对比 0.5 s、1 s、2 s 三种片段长度在 1–10 s 查询时长下的 Top‑1、Top‑3、Top‑10 Exact Hit 以及 Top‑1 Near Hit 率。实验显示 0.5 s 片段在大多数查询时长下获得最高 hit‑rate（至少 8/10 的“win”），1 s 片段仅在查询时长 ≥ 3 s 时性能接近；2 s 片段表现最差。LLM 评估表明 GPT‑5‑mini 在五个提示下始终推荐约 1 s 片段，最符合实验结果。

**⚠️ 局限性**

①实验仅覆盖音乐音频，无法直接推广到语音或环境声音等其他领域；②LLM 推荐结果受提示设计和模型版本影响，仍需进一步验证其稳定性；③实验未探索动态或自适应片段长度策略，可能进一步提升性能。

---

## 342. $\infty$-MoE: Generalizing Mixture of Experts to Infinite Experts

**arXiv ID:** 2601.17680 | [PDF](https://arxiv.org/pdf/2601.17680v1)

**作者:** Shota Takashiro `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**通讯引用:** 13879 | [OpenAlex ID](https://openalex.org/A5090592819)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了∞-MoE模型，将专家空间推广为连续无穷维，以稀疏采样实现高效推理。

**💡 创新点**

创新点在于把传统MoE的离散专家集合改为连续可采样空间，理论上实现无限专家容量，并通过mask动态激活FFN子集。

**🔧 技术方法**

采用高斯分布路由、Monte Carlo采样、top‑N%稀疏mask、CUDA自定义kernel加速等技术，并以GPT‑2 Small/Medium框架实现。

**📊 数据集**

使用FineWeb预训练语料10 B tokens，并在零样本NLP基准（BoolQ、HellaSwag、WinoGrande、ARC‑e/c、OBQA、RACE‑high）上评估。

**📈 对比分析**

与Dense、Switch Transformer、标准MoE比较；在GPT‑2 Small上平均得分39.8，分别比Dense高0.3、Switch高1.6、MoE高1.0；在Medium上平均41.3，超过所有基线；通过调节采样数K可在速度与精度之间灵活切换。

**⚠️ 局限性**

限制包括：在更大模型（如GPT‑3及以上）上的可扩展性未知；单峰高斯路由表达不足；迁移到视觉或多模态任务的适用性尚待验证；训练时动态稀疏导致吞吐量低于理想状态。

---

## 343. DIML: Differentiable Inverse Mechanism Learning from Behaviors of Multi-Agent Learning Trajectories

**arXiv ID:** 2601.17678 | [PDF](https://arxiv.org/pdf/2601.17678v1)

**作者:** Zhiyu An `[一作]` (University of California), Wan Du `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于对多智能体学习动态求逆的差分学习机制学习框架DIML，用于从仅观测到的行动轨迹中逆向重建未知的激励机制。

**💡 创新点**

创新点在于将学习过程中的离线策略概率和对手行动结合，利用对手行动固定的反事实收益进行梯度传播，克服了传统逆游戏理论对均衡假设的限制，并实现了对高维神经机制的可逆推断。

**🔧 技术方法**

主要技术包括对学习动态（如logit‑Q）进行可微分展开，构造对手条件下的反事实收益张量，并对轨迹似然进行最大似然估计和反向传播；此外在理论上给出对数it响应模型下的可辨识性与一致性证明。

**📊 数据集**

实验使用四类数据集： (1) 4个智能体5动作的随机神经机制； (2) 3个智能体7动作的拥堵计费和公共物品补贴机制； (3) 40–300个匿名智能体的基于计数的神经机制；以及 (4) 公开的模拟交互轨迹。

**📈 对比分析**

与基于表格最大似然、结构化参数化 MLE、以及学习规则错配版本的对比实验显示，DIML 在所有实验中均实现了最低的收益差异误差和较小的反事实 KL 散度，尤其在大规模匿名游戏中保持了可接受的性能。

**⚠️ 局限性**

主要局限包括对学习动态模型的依赖，若模型错配会导致收敛失败；需要足够的探索覆盖对手行动空间；以及对机制的非平稳性处理尚未覆盖。

---

## 344. SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation

**arXiv ID:** 2601.17657 | [PDF](https://arxiv.org/pdf/2601.17657v1)

**作者:** Taewan Cho `[一作]` (Gachon University), Andrew Jaeyong Choi `[通讯]` (Gachon University)

**通讯引用:** 115 | [OpenAlex ID](https://openalex.org/A5001613175)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SPACE-CLIP，利用冻结的 CLIP 视觉编码器的多层特征，构建双通道解码器直接预测单目深度图，省去文本提示与微调。

**💡 创新点**

创新点在于彻底放弃文本引擎，采用语义+结构双通道结合 FiLM 上下文调制和层级融合，首次实现对基础模型内部几何知识的直接解读。

**🔧 技术方法**

使用 ViT‑B/16 冻结 CLIP 编码器、双路径 Dense Predictor（Semantic 与 Structural）、FiLM 模块、层级特征融合、SILog+SSIM 复合损失，并在 224×224 低分辨率上训练后再上采样到 352×704。

**📊 数据集**

在 KITTI 数据集的 Eigen 训练/测试拆分上进行实验，输入图像尺寸 352×704，CLIP 只接受 224×224 中心裁。

**📈 对比分析**

与之前的 CLIP‑based 方法相比，SPACE‑CLIP 在 AbsRel、RMSE 等指标上显著提升（如 AbsRel 从 0.307 降到 0.104），并在无文本编码器、无 encoder 微调的 “No, No” 设定下保持竞争力；但仍略逊于专门训练的 vision‑only SOTA 模型。

**⚠️ 局限性**

局限性包括与最佳专用模型的性能差距、仅在 KITTI 上验证、对室内或多样化场景的泛化不足，以及当前仅支持 CLIP，需探索轻量级适配器、跨模型迁移和多任务整合。

---

## 345. Health-ORSC-Bench: A Benchmark for Measuring Over-Refusal and Safety Completion in Health Context

**arXiv ID:** 2601.17642 | [PDF](https://arxiv.org/pdf/2601.17642v1)

**作者:** Zhihao Zhang `[一作]` (Macquarie University), Usman Naseem `[通讯]` (Macquarie University)

**通讯引用:** 4113 | [OpenAlex ID](https://openalex.org/A5077006200)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文创建了Health-ORSC-Bench，一个用于评估医疗领域大型语言模型过度拒绝（over‑refusal）与安全完成（safe completion）的基准，包含3个难度子集共31,920条边界提示。

**💡 创新点**

创新点在于首次将医疗安全与可用性双指标结合，采用自动化种子提取、人工标注、LLM重写、七模型审核和分层难度的生成管线，构建了规模化、可重复使用的医疗安全评估工具。

**🔧 技术方法**

主要技术包括关键词+LLM分类筛选种子、Kimi‑K2生成安全边界提示、七大安全审核模型（Granite‑Guardian、Llama‑Guard等）筛除潜在有害内容、以及对30款SOTA模型的无系统提示、温度0、4,096 token限制下的批量评测；安全完成率采用LLM‑as‑Judge（Grok‑4）评估安全与有用性。

**📊 数据集**

使用的数据集来源于7个公开医疗毒性数据集（如DoNotAnswer、MedSafetyBench等），共提取2,306条医疗毒性种子，并在此基础上生成31,920条边界提示，覆盖7个健康风险类别（如自残、医学误导、药物滥用等）。

**📈 对比分析**

评测方法为对30个SOTA模型计算过度拒绝率（ORR）和安全完成率（SCR），结果显示安全优化模型如GPT‑5、Claude‑4在毒性提示拒绝率最高，但在安全提示上过度拒绝率可达80%；域专模型过度拒绝率低于15%但回答质量不足；Qwen‑Max表现最佳，近零ORR且≈70% SCR。

**⚠️ 局限性**

局限性包括：仅使用英文文本，忽略多语言和文化差异；分类体系仅包含7类风险，未覆盖保险欺诈、医院信息安全、治疗偏见等更细致风险；基准基于现有公开数据集，可能未捕捉最新的医疗安全挑战。

---

## 346. FOCA: Multimodal Malware Classification via Hyperbolic Cross-Attention

**arXiv ID:** 2601.17638 | [PDF](https://arxiv.org/pdf/2601.17638v1)

**作者:** Nitin Choudhury `[一作]`, Arun Balaji Buduru `[通讯]` (Indraprastha Institute of Information Technology Delhi)

**通讯引用:** 338 | [OpenAlex ID](https://openalex.org/A5014100784)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对恶意软件分类，提出了一种将二进制文件转换为音频和视觉两种模态，并在双曲几何空间中进行跨模态注意力融合的框架FOCA。

**💡 创新点**

创新点在于：①首次在恶意软件分类任务中利用双曲空间显式建模音频与视觉模态之间的层级关系；②设计了双曲跨模态注意力（Hyperbolic Cross‑Attention）以及 Möbius 加法融合层，兼顾曲率约束；③在不修改预训练模型的前提下实现多模态表示的高效对齐。

**🔧 技术方法**

使用的技术包括：
- 音频预训练模型：Wav2Vec2、WavLM、HuBERT；
- 视觉预训练模型：ResNet‑50、VGG‑19、ViT；
- 双曲空间映射（Poincaré 球映射、对数映射）
- 双曲跨模态注意力与 Möbius 加法融合；
- 传统卷积下游分类网络与全连接层。

**📊 数据集**

使用的数据集为：
- CICMalDroid‑2020（17,341个APK，涵盖1个正常类和4种恶意类）；
- Mal‑Net（约1.2M图片，47类、696族，选取10个类别各800样本构成平衡子集）。

**📈 对比分析**

与单模态、简单拼接、欧氏跨模态注意力以及现有SOTA方法进行对比。实验表明，FOCA 在两数据集上均取得最高准确率与宏观F1，尤其是 HuBERT+ViT 通过双曲融合获得 99.10%（CICMalDroid‑2020）和 88.62%（Mal‑Net）的准确率，显著优于欧氏注意力和其他基线。

**⚠️ 局限性**

局限性包括：
- 仅在两种公开数据集上验证，泛化性待进一步测试；
- 需要预训练模型与双曲映射的额外计算开销；
- 对超参数（曲率、投影维度）敏感，需额外调优；
- 对极端多模态缺失或噪声的鲁棒性尚未系统评估。

---

## 347. Study of Robust Power Allocation for User-Centric Cell-Free Massive MIMO Networks

**arXiv ID:** 2601.17632 | [PDF](https://arxiv.org/pdf/2601.17632v1)

**作者:** Saeed Mashdour `[一作]` (Pontifical Catholic University of Rio de Janeiro), Anke Schmeink `[通讯]` (RWTH Aachen University)

**通讯引用:** 3652 | [OpenAlex ID](https://openalex.org/A5072895220)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种鲁棒功率分配方法，利用最小二乘（LS）框架并加入Tikhonov正则化，得到闭式解并通过投影满足功率与非负约束，应用于用户中心的无单元化大规模MIMO下行链路。

**💡 创新点**

创新点在于将鲁棒功率分配转化为带正则项的LS问题，获得无迭代闭式解，显著降低计算复杂度；同时引入基于估计误差的正则参数，提升对CSI不确定性的抵抗能力。

**🔧 技术方法**

采用最小二乘估计、Tikhonov正则化、零迫（ZF）预编码、投影约束方法；实现了闭式的功率分配向量计算。

**📊 数据集**

使用仿真数据：25个AP，每AP 4个天线，总100个天线，200个单天线UE，随机分布在400m×400m区域，α=0.15的CSI误差。

**📈 对比分析**

与传统的梯度下降功率分配（GDPA）和鲁棒梯度下降（RGDPA）对比，结果显示RLSPA在相同SNR下拥有更高的总速率，并且在计算复杂度上仅略高于鲁棒梯度下降，整体表现优于两者。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实系统验证；仅考虑ZF预编码与下行链路；对CSI误差范围的正则参数设定需要经验或交叉验证；未考虑用户速率公平性或频谱效率的进一步优化。

---

## 348. EntWorld: A Holistic Environment and Benchmark for Verifiable Enterprise GUI Agents

**arXiv ID:** 2601.17722 | [PDF](https://arxiv.org/pdf/2601.17722v1)

**作者:** Ying Mo `[一作]` (Zhongguancun Laboratory), Dan Li `[通讯]` (Tsinghua University)

**通讯引用:** 64019 | [OpenAlex ID](https://openalex.org/A5100380810)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了EntWorld基准，包含1,756个基于真实企业系统的多模态交互任务。

**💡 创新点**

创新点在于利用数据库模式反向推理业务流程、SQL驱动的确定性验证以及多应用沙箱环境，打破传统仅靠视觉匹配的评估方式。

**🔧 技术方法**

采用Schema-Driven任务生成、Docker化企业应用、可执行SQL验证、截图与可访问树混合观察，以及LLM提示模板自动化生成等技术。

**📊 数据集**

使用开源企业系统（如Odoo、EspoCRM、ZenTao等）的数据库模式与真实数据，构造了EntWorld任务集。

**📈 对比分析**

通过与GPT‑4.1、Claude3.5 Sonnet、UI‑TARS、Qwen2.5‑VL‑32B等模型对比，EntAgent‑RL以56.89%成功率领先，明显优于现有开源和专有模型，远低于人类85%的基准。

**⚠️ 局限性**

仍存在UI可访问性信息不完整、长时序推理与状态跟踪能力不足、对非标准UI组件识别失误等局限，影响模型在更复杂企业流程中的表现。

---

## 349. Kareus: Joint Reduction of Dynamic and Static Energy in Large Model Training

**arXiv ID:** 2601.17654 | [PDF](https://arxiv.org/pdf/2601.17654v1)

**作者:** Ruofan Wu `[一作]` (University of Michigan), Mosharaf Chowdhury `[通讯]` (University of Michigan)

**通讯引用:** 14645 | [OpenAlex ID](https://openalex.org/A5013180923)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Kareus，一种面向大规模模型训练的能耗优化系统，能够同时调度 GPU 频率、SM 分配和通信启动时机，从而在保持训练时间不变或仅略微增长的前提下显著降低能耗。

**💡 创新点**

创新点包括：① 证明三种低层调度因素（SM 分配、通信启动时机、频率）相互影响，共同决定动态和静态能耗；② 提出分区重叠执行模型，将全局优化拆分为局部可枚举的子问题；③ 开发多目标贝叶斯优化（MBO）框架，利用超体积增益和不确定度引导候选选择，构造完整的时间‑能耗 Pareto 前沿。

**🔧 技术方法**

使用技术包括：多目标贝叶斯优化（基于 XGBoost surrogate 和 Bootstrap ensemble），热稳定性能耗测量（5 s 采样 + 5 s 冷却），Zeus 频率控制，MSCCL++ SM 分配控制，Megatron‑LM 与 Perseus 的集成，分区重叠执行引擎。

**📊 数据集**

实验数据集与工作负载：Llama 3.2 3B、Qwen 3 1.7B、Llama 3.3 70B 以及 14 种不同并行度与微批大小组合的训练任务。

**📈 对比分析**

与 Megatron‑LM、Nanobatching、Perseus 等基线在 max‑throughput 和前沿改进两种评估模式下比较；在真实 16 × A100 上，Kareus 在保持相同训练时间时能耗可降低至 77.9%（约 22.1% 降幅），时间可提升至 85.1%（约 14.9% 提升）；相对基线前沿，能耗可降低 28.3%，时间可缩短 27.5%。在大规模仿真中保持类似趋势。

**⚠️ 局限性**

局限性：① MBO 需要数小时的热稳定性采样，整体预处理开销较大；② 仅在 1F1B pipeline、NVIDIA A100 GPU 上验证，缺乏对其他 GPU 架构和更复杂通信依赖（如多阶段 AllGather）的全面评估；③ 依赖分区模型假设通信可独立重叠，某些模型的细粒度通信模式可能不完全适配；④ 频率切换成本和硬件限制在实际部署中可能影响优化效果。

---

## 350. The LLM Data Auditor: A Metric-oriented Survey on Quality and Trustworthiness in Evaluating Synthetic Data

**arXiv ID:** 2601.17717 | [PDF](https://arxiv.org/pdf/2601.17717v1)

**作者:** Kaituo Zhang `[一作]` (University of Houston), Na Zou `[通讯]` (University of Houston)

**通讯引用:** 1650 | [OpenAlex ID](https://openalex.org/A5084497683)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“LLM Data Auditor”框架，用以系统评估大型语言模型（LLM）生成的多模态合成数据的质量与可信度，并在六大数据模态（文本、符号/逻辑、表格、图、视听语言、代理数据）上构建统一的评估体系，强调从内在属性（intrinsic）而非仅靠下游任务性能来衡量生成数据。

**💡 创新点**

创新点主要体现在：
- 将数据视角放在首位，形成全新的评估框架；
- 建立了“质量（Validity, Fidelity, Diversity, Utility）+可信度（Safety, Faithfulness, Privacy, Fairness）”两大维度的指标体系；
- 对现有生成方法进行横向对比，揭示多模态评估中的空白与不足；
- 提出了针对不同模态的具体度量公式与实践建议，推动评估从经验化向标准化转变。

**🔧 技术方法**

技术手段包括：
- 语义与统计学度量（如Embedding Distribution Similarity、KST、TVD、β‑Recall等）；
- 逻辑与形式化验证（符号/逻辑数据的步骤验证、可执行测试、证明完整性检查）；
- 安全与偏见评估工具（Perspective API、ToxicChat、Statistical Parity、Equalized Odds等）；
- 数据合成策略（prompt‑based、fine‑tuning、混合架构、强化学习、工具验证）以及相应的评估方法。

**📊 数据集**

使用的数据集主要来自现有公开数据集与合成数据：
- 文本：SQuAD、OpenAssistant对话、ChatGPT生成对话；
- 符号/逻辑：GSM‑8K、ProofWriter、MetaMathQA；
- 表格：UCI 机器学习数据集、医疗临床表格、TabSyn、GReaT；
- 图：OpenKG、GO图谱、GraphBench；
- 视听语言：COCO‑Captions、VQA、Emu、Kosmos‑G；
- 代理：Simulated CARLA、TTSG、ChatSUMO 等；
- 论文中多使用作者自制的合成数据以对比评估。

**📈 对比分析**

通过对比分析，作者对六大模态的代表性生成方法（如RedPajama、Self‑Instruct、OpenCodeInstruct、GReaT、LLM4GraphGen、Emu 等）在各类质量与可信度指标上的表现进行系统汇总。结果显示：
- 许多方法仅报告单一指标（如多样性或有效性），缺乏完整的多维度评估；
- 在安全与隐私方面，评估覆盖率极低；
- 在表格与图数据的结构一致性检测中，多数方法未使用专门的约束验证；
- 综上，现有技术在质量与可信度的平衡上仍存在明显差距，提示未来研究需在指标完整性与评估工具可复现性上做进一步突破。

**⚠️ 局限性**

局限性包括：
- 该工作主要为综述与框架构建，未提出新的生成模型或算法，缺乏实验验证；
- 评估指标虽详尽，但在实际应用中部分度量（如β‑Recall、α‑Precision）计算复杂度高，易导致部署难度；
- 对某些模态（如日志、JSON）细粒度评估仍不完善；
- 可信度评估侧重安全与隐私，但对偏见与公平的全面量化尚未系统化；
- 综述中引用的实验数据多来源于已有论文，缺少统一的对比基准与标准化实验设置。

---

## 351. High-Order Mesh r-Adaptivity with Tangential Relaxation and Guaranteed Mesh Validity

**arXiv ID:** 2601.17708 | [PDF](https://arxiv.org/pdf/2601.17708v1)

**作者:** Ketan Mittal `[一作]` (Lawrence Livermore National Laboratory), Vladimir Tomov `[通讯]` (Lawrence Livermore National Laboratory)

**通讯引用:** 937 | [OpenAlex ID](https://openalex.org/A5056525853)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在高阶网格 r‑适应中加入切向松弛，并通过可证的 Jacobian 下界保证网格始终有效；同时提供一种基于块状线性边界函数的快速下界计算方法，配合偏移式障碍度量实现网格去扭曲。

**💡 创新点**

①只利用离散网格本身实现曲面上的切向松弛，无需 CAD 访问；②利用可证明的多项式下界（块状线性逼近）代替传统采样或 Bernstein 基础的保守下界，显著提升可靠性；③将该下界作为偏移障碍量用于网格去扭曲，提升了高阶网格的鲁棒性。

**🔧 技术方法**

Target‑Matrix Optimization Paradigm (TMOP)、高阶多项式函数的块状线性下界计算、偏移式障碍度量、最近点投影、拉普拉斯平滑、BFGS/牛顿优化、分层（q‑refinement）积分点自适应。

**📊 数据集**

没有公开真实数据集，全部使用合成高阶网格：四阶涡轮叶片网格、二维/三维 ALE 受力后的网格以及含扭曲元素的初始网格。

**📈 对比分析**

与传统在曲面上固定节点且仅在采样点检查 Jacobian 的做法对比，改进方法在保持网格有效性的前提下将 TMOP 目标从 170.8 降到 74.2，曲面节点重定位后曲率偏差显著下降；在 ALE 示例中，曲面节点在保持圆形几何后倾斜角从 44°–136° 缩小至 82°–97°；去扭曲后下界从负值恢复到正值，提升了网格稳定性。

**⚠️ 局限性**

①需要对初始高阶网格具有足够的几何精度，否则切向松弛会限制精度；②拉普拉斯平滑和最近点投影增加了计算开销，尤其在大规模网格上可能成为瓶颈；③q‑refinement 参数和障碍阈值需要经验调参，过大或过小会影响收敛与效率；④对极端非结构化或复杂 CAD 交互的适应性尚未验证。

---

## 352. Multi-core & GPU-based Balanced Butterfly Counting in Signed Bipartite Graphs

**arXiv ID:** 2601.17707 | [PDF](https://arxiv.org/pdf/2601.17707v1)

**作者:** Mekala Kiran `[一作]`, Tathagata Ray `[通讯]` (Birla Institute of Technology and Science Pilani)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了针对带符号二分图的平衡蝴蝶（2,2) 子图计数的并行实现，分别在多核 CPU（M‑BBC）和 GPU（G‑BBC、G‑BBC++）上实现高效计数。

**💡 创新点**

创新点在于：① 通过仅统计对称或不对称 wedge 并排除不平衡子结构，避免不必要的枚举；② 引入细粒度顶点级并行、基于 tile 的共享内存处理以及动态负载调度，显著提升 GPU 端吞吐；③ 结合多核与 GPU，构建了可扩展的高性能平衡蝴蝶计数框架。

**🔧 技术方法**

使用的技术包括：顶点优先级排序、基于 hash 的 wedge 分桶、TBB 或 OpenMP 并行循环、CUDA 并行块、共享内存 bitmap、原子计数、warp‑level 以及 block‑level 归约、动态任务计数器和自适应线程规模。

**📊 数据集**

实验采用了 15 个真实世界的二分图数据集，包括社交、推荐、电影、音乐、学术、电子商务等，图规模从几千到上亿条边不等（如 Netflix、Yahoo 等）。

**📈 对比分析**

与串行基线（BB2K）和改进的 SBCList++ 对比，M‑BBC 在所有数据集上平均实现 38× 加速（最大 71×），GPU 实现 G‑BBC++ 在所有数据集上平均实现 2,600× 加速（最大 13,320×），并且在大规模图上比 CPU 方案快 50–186 倍。

**⚠️ 局限性**

局限性包括：仅支持平衡 (2,2) 子图；对较大分区的平衡 (2,k) 计数存在负载不平衡和计算量大的问题；对动态图和更一般的 (p,q) 子图尚未给出高效方案。

---

## 353. SQL-Trail: Multi-Turn Reinforcement Learning with Interleaved Feedback for Text-to-SQL

**arXiv ID:** 2601.17699 | [PDF](https://arxiv.org/pdf/2601.17699v1)

**作者:** Harper Hua `[一作]` (Stanford University), Huzefa Rangwala `[通讯]` (Amazon Web Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个多轮强化学习框架，让文本转SQL模型在与数据库交互的过程中迭代生成并纠正查询。

**💡 创新点**

创新点在于将单通道生成转为多轮交互、难度感知的回合预算以及六项奖励机制，显著提升数据效率和跨域泛化。

**🔧 技术方法**

使用了LLM（Qwen2.5-Coder 3B/7B/14B）、基于GRPO的强化学习、ReAct式工具调用、奖励分面、以及教师模型蒸馏。

**📊 数据集**

主要使用Spider、BIRD及其变体（Spider-DK、Spider-Syn、Spider-Realistic）作为训练和评估数据集。

**📈 对比分析**

与单轮RL和公开最先进系统（SQL-R1、Reasoning-SQL、OminiSQL、Sonnet-3.7）对比，在仅2,000条训练样本下在BIRD-dev达到60%以上执行准确率，数据效率提升7–18倍。

**⚠️ 局限性**

局限包括需要可执行的数据库环境、较高推理成本/延迟、奖励设计的先验偏差，以及在实际生产环境中的适用性待验证。

---

## 354. LegalMALR:Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval

**arXiv ID:** 2601.17692 | [PDF](https://arxiv.org/pdf/2601.17692v1)

**作者:** Yunhan Li `[一作]` (City University of Macau), Min Yang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 68830 | [OpenAlex ID](https://openalex.org/A5100694840)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LegalMALR 框架，结合多代理查询理解、GRPO 优化与 LLM 重排序，实现对中文法规检索的高召回和精准排序。

**💡 创新点**

创新点在于：① 多代理系统生成多角度、结构化的查询改写，提升覆盖率；② 用 Generalized Reinforcement Policy Optimization 统一调优整个代理策略，显著降低随机性与提升召回；③ 采用大模型零样本重排序器进行法律推理，进一步提高排名质量。

**🔧 技术方法**

核心技术包括：多代理查询理解系统（MAS）、GRPO 强化学习策略优化、基于 Qwen3-4B 的稠密检索与轻量级重排序、零样本 LLM（Qwen-Max）重排序。

**📊 数据集**

使用的主要数据集为：① STARD（1,234 训练 / 309 测试）作为在分布内部训练与评估；② CSAID（118 真实/案例/失败查询）用于跨分布评估。

**📈 对比分析**

与传统 RAG、BM25、法学领域嵌入检索等基线比较，LegalMALR 在 STARD 上 Recall@10+6.16pp，MRR@10+6.31pp；在 CSAID 上 Recall@10+8.09pp，MRR@10+4.41pp，HitRate@10+2.54pp，显著优于现有最强 RAG 系统。

**⚠️ 局限性**

局限包括：① 依赖多轮 LLM 调用，复杂查询时计算量与延迟增加；② 需要商用 LLM 进行最终重排序，成本与可复现性受限；③ 对中文法律语料与结构特定，跨语言/跨法域推广尚待验证；④ GRPO 奖励仅关注召回，未显式考虑排序质量或中间推理合法性。

---

## 355. AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking

**arXiv ID:** 2601.17645 | [PDF](https://arxiv.org/pdf/2601.17645v1)

**作者:** Xilin Jiang `[一作]` (Columbia University), Nima Mesgarani `[通讯]` (Columbia University)

**通讯引用:** 12907 | [OpenAlex ID](https://openalex.org/A5033351155)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 AVMeme Exam 这个多模态、多语言、跨文化的音视频梗基准，并评估了 19 种主流多模态大型语言模型与人类对其的理解能力。

**💡 创新点**

首次系统地将上下文推理、情感识别、使用场景和世界知识等高级语义维度加入音视频梗的评测框架，形成了完整的七类问答形式。

**🔧 技术方法**

采用多模态 LLM（如 Gemini 3 Pro、GPT‑4o Audio 等）与人工标注的多项选择问题进行推理，并通过文本/视觉作弊检测和思考实验探究模型行为。

**📊 数据集**

使用 1,032 条来自 YouTube、Bilibili 的音视频梗，覆盖 10 种语言和 5 类声音（语音、歌曲、音乐、音效、无声音效）以及相应的元数据与问答。

**📈 对比分析**

在统一提示、统一音频/视频预处理、同一组 7 类问题的基础上对模型进行准确率评测，最佳模型 Gemini 3 Pro 在测试集的平均准确率达 80% 但在世界知识、使用场景等深层任务上仍低于 60%。

**⚠️ 局限性**

局限性包括：贡献者样本偏向受教育研究者、语言覆盖仍有限、30 秒截断可能丢失关键信息、评测仅为单轮多项选择且主观性高，难以反映实际多轮交互与开放式对话需求。

---

## 356. Code Change Characteristics and Description Alignment: A Comparative Study of Agentic versus Human Pull Requests

**arXiv ID:** 2601.17627 | [PDF](https://arxiv.org/pdf/2601.17627v1)

**作者:** Dung Pham `[一作]` (Trent University), Taher A. Ghaleb `[通讯]` (Trent University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较了5个大模型编写的自动化拉取请求（APRs）与人类拉取请求（HPRs）的代码变更特征与描述质量。

**💡 创新点**

首次同时量化符号冲突与更改周期，并揭示APRs在提交级别信息表达强但在全PR级别叙述不足。

**🔧 技术方法**

使用语义相似度模型GistEmbed、CodeT5、LLM评估（GPT‑4o）及随机森林、XGBoost等机器学习技术进行对齐度量与特征重要性分析。

**📊 数据集**

数据集为AIDev‑pop子集（33,596个APRs）与6,618个HPRs，涵盖多语言（TypeScript、Python、Go）及补丁差异、提交信息。

**📈 对比分析**

对比方法包括PR–Commit相似度、Patch–Commit相似度和LLM一致性评分；APRs在提交级别相似度（0.72）高于HPRs（0.68），但在PR级别相似度（0.86）低于HPRs（0.88），并显示符号冲突率和修改周期明显更高。

**⚠️ 局限性**

局限性包括对符号提取使用正则表达式、评估指标仅为语义相似度/LLM评分、数据仅覆盖动态语言、agent能力快速演进导致结果可能过时。

---

## 357. A Model-Driven Lossless Compression Algorithm Resistant to Mismatch

**arXiv ID:** 2601.17684 | [PDF](https://arxiv.org/pdf/2601.17684v1)

**作者:** Cordelia Hu `[一作]`, Jennifer Tang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种抵抗预测误差的无损压缩算法，利用下一词预测与桶分区结合最短唯一前缀编码，实现在允许多倍误差的前提下保证正确解码。

**💡 创新点**

在不使用算术编码的前提下，通过分桶、前缀码与最短唯一前缀，给出对结构化误差的误差认证与期望长度理论分析，提升压缩鲁棒性。

**🔧 技术方法**

使用大型语言模型的下一词概率预测、基于桶的概率分区、前缀码与最短唯一前缀编码技术，并进行理论与实验评估。

**📊 数据集**

在真实文本数据上测试，包括随机抽取的维基百科文章和莎士比亚《哈姆雷特》中的文本。

**📈 对比分析**

将压缩率与 gzip、bzip2 等传统压缩器对比；在误差上限 c≈10/3 时，压缩率可达 gzip 的 3 倍以上，解码准确率超过 90%，表现优于现有 PMATIC 等方法。

**⚠️ 局限性**

对概率分布的连续幂律假设与高概率词的处理不足，桶区间需手工选择，且对极大误差范围下的匹配保证仍有限。

---

## 358. Beyond the Rabbit Hole: Mapping the Relational Harms of QAnon Radicalization

**arXiv ID:** 2601.17658 | [PDF](https://arxiv.org/pdf/2601.17658v1)

**作者:** Bich Ngoc `[一作]`, Robert West `[通讯]` (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统分析了QAnon亲友叙事，映射激进化路径并量化情感创伤；

**💡 创新点**

首次将激进化视为关系现象，构建情感创伤与“激进化人设”对应的经验框架；

**🔧 技术方法**

采用BERTopic主题建模、LDA图模型挖掘人设、LLM情绪检测与逻辑回归分析；

**📊 数据集**

使用2019-2024年Reddit /r/QAnon亲友支持社区共12747条双视角叙事数据集；

**📈 对比分析**

通过ILR变换后的混合归属模型和逻辑回归，得到6个可解释人设，情绪预测AUC>0.8，显著提升传统聚类方法；

**⚠️ 局限性**

情绪标注依赖LLM，情绪维度有限；数据来源单一子版，缺乏时间序列和跨群体验证。

---

## 359. Time-Varying Causal Treatment for Quantifying the Causal Effect of Short-Term Variations on Arctic Sea Ice Dynamics

**arXiv ID:** 2601.17647 | [PDF](https://arxiv.org/pdf/2601.17647v1)

**作者:** Akila Sampath `[一作]` (University of Maryland), Jianwu Wang `[通讯]` (University of Maryland)

**通讯引用:** 3701 | [OpenAlex ID](https://openalex.org/A5101750217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出KGCM‑VAE框架，利用物理知识指导因果处理，并通过MMD与因果邻接约束对潜在空间进行去混杂，量化海冰厚度对SSH的短期因果影响

**💡 创新点**

创新点在于将海洋动力学（SSH–海冰厚度关系、地转平衡）嵌入变分自编码器，采用时间相关的Sigmoid调制生成物理一致的处理信号，并在潜在层同时施加MMD平衡与因果邻接约束，实现可解释且稳健的因果效应估计

**🔧 技术方法**

技术包括双向GRU编码器、因果解码器、变分自编码器、RBF核MMD潜在平衡、因果邻接矩阵、SSH‑速度Sigmoid调制、重构损失、KL、MSE损失融合

**📊 数据集**

使用S2S ECMWF实测数据（2020‑2024，60°N–90°N）构建日均海冰厚度、SSH、海流速度时间序列，并通过合成数据验证模型；数据覆盖1620天

**📈 对比分析**

与CF‑RNN、Causal‑RNN、Causal‑TaRNET三种时序因果基线对比，评估RMSE与PEHE；KGCM‑VAE在PEHE上最优（3.82）但RMSE略高；Ablation实验表明MMD+ADJ组合显著提升性能

**⚠️ 局限性**

局限在于预测精度（RMSE）不及传统模型，对大尺度空间字段的适用性需进一步验证；模型复杂度高，需依赖物理先验且对参数设置敏感

---

## 360. RPNT: Robust Pre-trained Neural Transformer -- A Pathway for Generalized Motor Decoding

**arXiv ID:** 2601.17641 | [PDF](https://arxiv.org/pdf/2601.17641v1)

**作者:** Hao Fang `[一作]` (University of Washington), Amy L. Orsborn `[通讯]` (University of Washington)

**通讯引用:** 1737 | [OpenAlex ID](https://openalex.org/A5017263213)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发了一种鲁棒预训练神经变压器RPNT，用于跨会话、跨任务、跨主体、跨站点的神经信号解码，提出多维旋转位置嵌入(MRoPE)、基于上下文的注意机制以及均匀随机掩码的自监督预训练方法。

**💡 创新点**

三大创新点：1）MRoPE将实验元数据（站点坐标、时序、行为类型等）嵌入到Transformer位置编码中，实现对不同记录配置的零样本泛化；2）通过卷积核生成的上下文注意机制捕获局部时序结构，适应神经信号的非平稳性；3）统一因果掩码与对比学习的自监督目标，使预训练不需要行为标签。

**🔧 技术方法**

使用Transformer框架，RoPE扩展到多维，Contextual Convolution Attention，因果掩码下的Poisson重建损失，Contrastive Loss，微调SFT；结合自监督预训练与后续监督微调。

**📊 数据集**

两个公开数据集：1）多会话、多任务、多主体的微电极基准（4只猴子，PMd/M1）；2）Neuropixels 1.0高密度探针记录（17个不同站点，PMd/M1）。

**📈 对比分析**

与传统线性、MLP、RNN、LFADS、Transformer（NDT、PoYo）等基线在跨会话、跨任务、跨主体、跨站点四种解码情景下进行R²比较。RPNT在所有情景下均优于基线，特别是T‑RT跨任务跨主体提升约7%，跨站点提升约4%；且在少量训练样本（10%）下仍能保持良好性能。

**⚠️ 局限性**

局限性：仅在运动皮层（PMd/M1）数据上验证，未覆盖前额叶、基底核等其他脑区；仅测试中心外向和随机目标的简化运动，缺乏自然行为与认知任务；仅为单模态模型，未来需扩展至多模态。

---

## 361. Scaling Laws for Moral Machine Judgment in Large Language Models

**arXiv ID:** 2601.17637 | [PDF](https://arxiv.org/pdf/2601.17637v1)

**作者:** Kazuhiro Takemoto `[一作]` (Kyushu Institute of Technology), Kazuhiro Takemoto `[通讯]` (Kyushu Institute of Technology)

**通讯引用:** 2581 | [OpenAlex ID](https://openalex.org/A5013426338)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了75个从0.27B到1000B参数的LLM在道德判断上的表现，并发现其与人类偏好之间的距离呈幂律衰减。

**💡 创新点**

首次将可扩展的幂律规律应用于价值判断任务，证明道德直觉是可量化的、可预测的能力；同时揭示推理能力在规模之外还能提升约16%。

**🔧 技术方法**

使用Moral Machine框架生成情境，计算平均边际成分效应（AMCE），用欧氏距离衡量模型与人类的偏好匹配度；通过Spearman、log‑linear回归与线性混合效应模型对规模关系进行定量分析。

**📊 数据集**

Moral Machine生成的1万个可变因素场景（年龄、性别、社会地位等）以及来自233个国家的40万+人类决策，用来构建人类AMCE向量。

**📈 对比分析**

与人类AMCE向量的欧氏距离作为对齐度量；结果显示模型规模越大对齐度越高，10倍参数提升约21%；扩展推理模型相较于普通模型多提升约16%，方差随规模减小。

**⚠️ 局限性**

仅使用欧氏距离的对齐度量缺乏对实际道德效果的验证；数据主要来自西方国家，缺乏跨文化验证；模型规模与训练数据质量、算力等共线，难以完全分离因果；评价仅针对交通事故类道德困境，其他伦理维度未知。

---

## 362. Conduit: Programmer-Transparent Near-Data Processing Using Multiple Compute-Capable Resources in Solid State Drives

**arXiv ID:** 2601.17633 | [PDF](https://arxiv.org/pdf/2601.17633v1)

**作者:** Rakesh Nadig `[一作]` (ETH Zürich), Onur Mutlu `[通讯]` (ETH Zürich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出一种名为Conduit的通用、程序员透明的近数据处理框架，能够在SSD内部三种异构计算资源（SSD控制器内核、SSD DRAM、NAND闪存）之间按指令粒度动态调度算子；

**💡 创新点**

创新点在于：①将编译时自动向量化与运行时成本评估结合，形成基于六项特征（操作类型、操作数位置、数据依赖、资源排队、数据移动延迟、计算延迟）的全局成本函数；②实现对三种资源的细粒度、实时决策与指令转换，无需硬件改造；③提供完整的程序员无感代码迁移与数据一致性机制；

**🔧 技术方法**

技术包括：LLVM自定义编译器扩展实现向量化、元数据嵌入；ARM MVE指令集用于控制器内核；SIMDRAM/MIMDRAM/Proteus等ISA用于DRAM；Flash-Cosmos等闪存原语；SSD控制器内的成本评估与指令转换单元；NVMe管理员命令用于二进制迁移；

**📊 数据集**

使用六个实际数据密集型工作负载：AES加密/解密、XOR过滤器、heat-3d/3D热扩散、jacobi-1d、LLaMA2 7B推理、LLaMA2 7B训练（INT8量化），覆盖计算、I/O、混合模式；

**📈 对比分析**

与CPU、GPU、四种单资源NDP基线（Flash-Cosmos、Flash-Plus、MIMDRAM、控制器内核）、两种基线调度模型（带宽驱动、数据移动驱动）以及理想无争用模型对比；Conduit平均在CPU上提升4.2×、GPU 1.8×，比最佳基线提升1.8×，能耗平均下降46%，并在尾延迟方面显著优于其它方案；

**⚠️ 局限性**

局限性包括：依赖LLVM自动向量化，无法处理复杂数据依赖、控制流或原子操作的代码；仅支持整数运算（需量化）；假设所有工作负载数据均在SSD上；成本函数虽轻量但仍有几微秒运行时开销；在极端高并发或非均衡工作负载下可能出现资源分配不均；

---

## 363. A Computational Approach to Visual Metonymy

**arXiv ID:** 2601.17706 | [PDF](https://arxiv.org/pdf/2601.17706v1)

**作者:** Saptarshi Ghosh `[一作]` (University of Cincinnati), Tianyu Jiang `[通讯]` (University of Cincinnati)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究视觉拟喻，提出基于符号学的图像生成管线，并构建ViMET数据集以评估多模态模型的非文字关联推理。

**💡 创新点**

创新点在于首次将视觉拟喻纳入计算机视觉与NLP任务，提出三步LLM+StableDiffusion管线，并提供首个视觉拟喻基准数据集。

**🔧 技术方法**

技术主要包括大型语言模型（Llama 3.1‑70B‑Instruct）、链式推理提示、文本到图像模型（Stable Diffusion‑3.5‑Large）以及人类评注与相似度过滤。

**📊 数据集**

使用的主要数据集是ViMET（2000张多选题图像），源自WordNet的2077个概念词以及由LLM生成的代表物（representamen）。

**📈 对比分析**

通过与多模态视觉语言模型（InternVL3‑78B、Qwen2.5‑VL‑72B等）和人类基准对比，模型准确率最高约65.9%，远低于人类86.9%，显示存在约21%性能差距。

**⚠️ 局限性**

局限性包括对LLM知识库的依赖导致文化关联缺失、对生成过程各步骤缺乏独立评估、以及视觉拟喻本质的多义性和主观性难以完全量化。

---

## 364. Do Reasoning Models Ask Better Questions? A Formal Information-Theoretic Analysis on Multi-Turn LLM Games

**arXiv ID:** 2601.17716 | [PDF](https://arxiv.org/pdf/2601.17716v1)

**作者:** Daniel M. Pedrozo `[一作]` (Advanced Knowledge Center for Immersive Technologies), Bryan L. M. de Oliveira `[通讯]` (Advanced Knowledge Center for Immersive Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个多轮对话框架，用信息增益评估LLM的问答探索能力。

**💡 创新点**

首次引入基于信息增益的细粒度评估，并对比链式推理与否。

**🔧 技术方法**

使用三种LLM代理（提问者、答案者、剪枝者），采用链式推理和信息增益计算。

**📊 数据集**

使用城市数据集（40个全球人口最多的城市）构建层次知识图谱。

**📈 对比分析**

在全可观测与部分可观测两种设置下比较不同模型，结果显示带链式推理的大模型信息增益更高、成功率更高。

**⚠️ 局限性**

局限在于需要完整的假设空间结构、仅评估地理游戏、依赖代理模型知识。

---

## 365. FedCCA: Client-Centric Adaptation against Data Heterogeneity in Federated Learning on IoT Devices

**arXiv ID:** 2601.17713 | [PDF](https://arxiv.org/pdf/2601.17713v1)

**作者:** Kaile Wang `[一作]` (Hong Kong Polytechnic University), Yinfeng Cao `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 304 | [OpenAlex ID](https://openalex.org/A5016974454)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在物联网环境下提出 FedCCA，针对数据异构的联邦学习，采用基于客户端信息的动态选择与注意力加权多源聚合，实现每个客户端的个性化模型训练。

**💡 创新点**

创新点在于①为每台设备引入专门的客户端编码器提取个性特征；②提出以客户端特征相似度为依据的动态客户端选择策略；③结合注意力机制的多源聚合，提升跨域知识迁移效果。

**🔧 技术方法**

技术包括联邦学习框架、额外的客户端编码器、欧氏距离与负指数注意力函数、基于相似度的客户端聚类、加权全局聚合以及常用的 CNN/ResNet 作为基础网络。

**📊 数据集**

实验使用了数字分类数据集（MNIST、Synthetic Digits、SVHN、USPS、RotatedMNIST、FEMNIST）和图像识别数据集（CIFAR‑100、DomainNet）进行评估。

**📈 对比分析**

与 FedAvg、FedProx、FedRep、FedAMP、FedDAN、FedDANN、FedSR、FedMC 等基线对比，FedCCA 在多数任务中均实现了最高或相近的准确率，并显著缩短收敛速度。

**⚠️ 局限性**

主要限制包括服务器端因动态选择与注意力计算导致的额外开销，且对客户端特征相似度的度量假设可能在极端异构或极大规模场景下失效。

---

## 366. S$^3$-Attention:Attention-Aligned Endogenous Retrieval for Memory-Bounded Long-Context Inference

**arXiv ID:** 2601.17702 | [PDF](https://arxiv.org/pdf/2601.17702v1)

**作者:** Qingsen Ma `[一作]` (Beijing University of Posts and Telecommunications), Zhaofeng He `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 1002 | [OpenAlex ID](https://openalex.org/A5101869968)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了S^3-Attention框架，将长上下文推理转化为端到端的自注意力对齐的内生检索，使用稀疏自编码器构建CPU倒排索引，省去GPU KV缓存，实现O(1) GPU内存。

**💡 创新点**

创新点在于利用模型自身注意力投影的稀疏语义特征做检索，消除外部检索的语义鸿沟；通过流式语义索引实现检索与生成器内部语义完全对齐，并与BM25融合获得混合检索。

**🔧 技术方法**

核心技术包括Top‑k稀疏自编码器（SAE）、流式语义索引、特征共激活评分、密度平滑与非极大值抑制（NMS）以及端到端的Hybrid融合。

**📊 数据集**

评测使用LongBench套件（单文档QA、多文档QA、摘要）以及多种LLM模型（Llama‑3.1、Mistral、Qwen2）作为数据集与模型基础。

**📈 对比分析**

在统一推理环境下，S^3‑Hybrid在多模型上保持99%以上的FullKV性能，并在信息密集任务上略优；与RAG、BM25、KV压缩方法相比，性能接近或更好，展示了近乎无损的检索效果。

**⚠️ 局限性**

当前原型的墙钟延迟高于FullKV，主要受限于Python实现、CPU–GPU同步和倒排索引存储方式；后续需通过CUDA核融合、索引压缩等工程优化来弥补延迟差距。

---

## 367. REV-INR: Regularized Evidential Implicit Neural Representation for Uncertainty-Aware Volume Visualization

**arXiv ID:** 2601.17689 | [PDF](https://arxiv.org/pdf/2601.17689v1)

**作者:** Shanu Saklani `[一作]` (Indian Institute of Technology Kanpur), Soumya Dutta `[通讯]` (Indian Institute of Technology Kanpur)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5035530630)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种新的不确定性感知隐式神经表示（REV‑INR），能够在一次前向推理中同时预测体素值、模型不确定性（epistemic）和数据不确定性（aleatoric），并通过专门设计的正则化方法提高不确定性估计的可解释性。

**💡 创新点**

创新点包括：
• 首次将高阶后验分布（NIG）与隐式网络结合，实现单模型同时估计两种不确定性；
• 设计两种基于Pearson相关系数的训练正则化（对模型不确定性与预测误差、对数据不确定性与局部梯度）以实现可校准的量化；
• 对传统的Monte Carlo Dropout INR和多解码器 INR 进行改造，使其也能估计数据不确定性，从而完成同类方法对比。

**🔧 技术方法**

技术手段包括：
• 隐式神经网络（SIREN）与增设四个输出节点；
• 采用 evidential 学习，使用 NIG 分布并通过 KL 散度损失与负对数似然（NLL）相结合；
• 两阶段训练策略（先 MSE 再加上 evidential 损失与正则化）；
• 对 MCD‑INR 采用 Dropout 并预测方差；
• 对 RMD‑INR 采用共享编码器+多解码器架构并预测方差；
• 通过 Level‑Crossing Probability (LCP) 进行不确定性可视化。

**📊 数据集**

使用六个科学数据集：
• Teardrop（128³）
• Vortex（512³）
• Combustion（480×720×120）
• Foot（500³）
• Hurricane Isabel（500×500×100）
• Heptane（512³）

**📈 对比分析**

对比方法包括：Deterministic INR、MCD‑INR、RMD‑INR、以及压缩方法 TTHRESH、ZFP。实验显示：
• REV‑INR 在 PSNR、SSIM、LPIPS 上往往优于或与基线相当；
• 在欧氏误差与模型不确定性、以及数据不确定性与局部方差/插值误差的相关性上显著高于对手；
• LCP 可视化中，REV‑INR 能恢复薄连接且不出现过度扩展；
• 推理时间最快或与 Deterministic INR 相当，而训练时间略高；
• 在压缩率‑PSNR 轨迹中位于最优位置。

**⚠️ 局限性**

局限性包括：
• 训练时需要计算复杂的 KL 散度和正则化，导致训练时间比 MCD‑INR 与 RMD‑INR 长；
• AU 正则化基于梯度大小，若梯度与数据噪声相关性弱，AU 估计可能失效；
• 不确定性正则化会在重建精度与不确定性校准之间产生权衡，需要进一步自动化平衡；
• 目前仅针对单变量体素数据，扩展到多变量或大规模数据需要进一步研究。

---

## 368. BanglaRobustNet: A Hybrid Denoising-Attention Architecture for Robust Bangla Speech Recognition

**arXiv ID:** 2601.17679 | [PDF](https://arxiv.org/pdf/2601.17679v1)

**作者:** Md Sazzadul Islam Ridoy `[一作]` (Ahsanullah University of Science and Technology), Md. Aminur Rahman `[通讯]` (Ahsanullah University of Science and Technology)

**通讯引用:** 6314 | [OpenAlex ID](https://openalex.org/A5054647892)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 BanglaRobustNet，一种针对孟加拉语的混合降噪‑注意力 ASR 框架，实现端到端训练，显著降低噪声环境下的错误率。

**💡 创新点**

① 使用基于扩散模型的 Phonetic‑aware DBDM，保留孟加拉语关键音位；② 引入基于说话人嵌入的 Contextual Cross‑Attention (CCAM)，实现对性别、年龄、方言的自适应；③ 通过联合 CTC、音素一致性与说话人一致性三项损失实现多目标优化。

**🔧 技术方法**

技术包括 Wav2Vec‑BERT 主干、扩散式降噪网络、跨模态注意力、CTC 解码、Phoneme 分类器、AdamW 优化器与多阶段训练策略。

**📊 数据集**

训练与评估使用 Mozilla Common Voice Bangla、OpenSLR Bengali、BengaliSR 以及自制含多噪声和方言的测试集（共 20 小时噪声数据、39 方言语料）。

**📈 对比分析**

与 Whisper‑Small、Whisper‑Large、Wav2Vec‑BERT 等基线在 Clean、SNR 0–10 dB、不同噪声类型上对比，BanglaRobustNet 在 Clean 下 WER 12.3%（比基线低 30%+），在 0 dB SNR 下 42.4%（比基线低约 30%），PER 与 BLEU 亦显著提升，RTF 仅 0.16。

**⚠️ 局限性**

局限包括：模型仍对极低 SNR（<0 dB）下的复杂混响保持不足；方言覆盖仍偏向标准孟加拉，稀有方言表现有限；训练需要较多 GPU 资源，移动部署仍受限于模型体积。

---

## 369. GazeSummary: Exploring Gaze as an Implicit Prompt for Personalization in Text-based LLM Tasks

**arXiv ID:** 2601.17676 | [PDF](https://arxiv.org/pdf/2601.17676v1)

**作者:** Jiexin Ding `[一作]` (University of Washington), Akshay Gadre `[通讯]` (University of Washington)

**通讯引用:** 485 | [OpenAlex ID](https://openalex.org/A5081775039)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究将用户注视信息作为隐式提示，利用大型语言模型（LLM）生成个性化文本摘要，并在学术阅读场景中实现原型。

**💡 创新点**

创新点在于：①首次系统评估多种注视表示（密度、热图、SVM）在文本LLM中的效果；②发现热图表示能在子句级别超越显式提示；③通过眼动驱动的LLM辅助阅读提升用户体验且降低使用成本。

**🔧 技术方法**

技术包括：眼动追踪、注视表示编码（热图生成、密度统计、SVM注意力分类）、多模态提示整合至 Gemini 2.5 Pro LLM、ROUGE/BERTScore、LLM自评等。

**📊 数据集**

使用的数据集包括：自制 TOEFL 阅读材料（700–800 词），以及之前工作中标注眼动与词聚焦关系的眼动数据集。

**📈 对比分析**

比较方法：与文本仅输入（Text）和显式提示（Target Paragraphs）两基线对比，使用 ROUGE、BERTScore、LLM 质量评分等指标。结果显示：热图表示在多项指标上显著优于基线，且在语义质量上与显式提示相当或更好；密度表示在句子层面优势明显；SVM表示无显著提升。

**⚠️ 局限性**

局限性包括：目前缺乏内置眼动追踪的商业智能眼镜；实验使用外置眼动仪，易受漂移影响；样本量小且仅涉及文本任务，未覆盖图像/多模态内容；对用户注意力与认知状态的捕捉仍不完备。

---

## 370. Uni-RS: A Spatially Faithful Unified Understanding and Generation Model for Remote Sensing

**arXiv ID:** 2601.17673 | [PDF](https://arxiv.org/pdf/2601.17673v1)

**作者:** Weiyu Zhang `[一作]` (Peking University), Yu Liu `[通讯]` (Peking University)

**通讯引用:** 66413 | [OpenAlex ID](https://openalex.org/A5100345666)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Uni‑RS 统一的遥感多模态模型，解决了空间逆转诅咒，实现了在文本到图像生成过程中对空间布局的高度保真，同时保持强大的图像理解能力。

**💡 创新点**

创新点包括：①系统识别并量化空间逆转诅咒；②引入 Spatial‑Layout Planning 将文本转化为结构化布局计划；③通过 Spatial‑Aware Query Supervision 对学习查询进行空间关系监督；④提出 Image‑Caption Spatial Layout Variation 的旋转与文本重写增广；⑤构建 RS‑Spatial 空间注释数据集。

**🔧 技术方法**

技术方案：将多模态大语言模型 Qwen2.5‑VL‑3B 与扩散图像生成器 SANA‑1.6B 通过 256 个可学习查询和 Transformer 连接器耦合；采用两阶段训练（先生成后联合理解与生成）；实现空间规划、查询监督和几何增广等模块。

**📊 数据集**

使用的数据集：Git10M（10M 图文对）用于生成训练；RSICap、VRSBench 用于多模态理解训练；RS‑Spatial 用于空间规划与查询监督；评估数据集包括 RSICD、RSIEval、VRSBench。

**📈 对比分析**

通过与多种基线（Text2Earth、CRS‑Diff、DALL‑E、Lafite 等）在 RSICD、RSIEval、VRSBench 上的 FID、零样本分类、CLIP 分数以及 Spatially Faithful Rate (SFR) 等指标比较。Uni‑RS 在不微调 RSICD 时已显著优于 Text2Earth；微调后在 SFR 上提升至约 80%（比 Text2Earth 的 63% 高出约 17%），在 FID 上降低 40% 以上，并保持或提升 CLIP 分数，显示出在空间保真和整体生成质量上的领先。

**⚠️ 局限性**

局限性：①空间逆转诅咒的解决仍主要关注单一对象与网格式位置，复杂多对象布局的空间一致性仍有待加强；②对旋转增广的依赖可能限制对非规则几何变换的泛化；③模型训练和推理需要大规模算力和显存；④RS‑Spatial 数据集虽丰富，但覆盖的空间关系类型仍有限，可能影响模型在更细粒度空间语义上的表现。

---

## 371. Align to the Pivot: Dual Alignment with Self-Feedback for Multilingual Math Reasoning

**arXiv ID:** 2601.17671 | [PDF](https://arxiv.org/pdf/2601.17671v1)

**作者:** Chunxu Zhao `[一作]`, Junlan Feng `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Pivot-Aligned Self-Feedback Multilingual Reasoning（PASMR）框架，通过将主语言作为 pivot 并利用自反馈奖励机制提升多语言推理一致性

**💡 创新点**

创新点在于无需外部黄金答案或翻译模型，使用主语言对齐与自生成的跨语言一致性奖励进行强化学习，显著提高低资源语言的推理表现

**🔧 技术方法**

技术包含 Pivot‑Aligned Mapping（PAM）阶段的翻译对齐与 Self‑feedback Reinforcement Learning（SRL）阶段的 REINFORCE++ 训练，采用 KL 损失与奖励函数衡量目标语言与 pivot 语言的一致性

**📊 数据集**

使用 GSM8K、MSVAMP、MGSM 三个数学推理数据集，构造 9 种低资源语言与英语共 20,480 条样例，并额外构造 OOD 训练集

**📈 对比分析**

与 Pipeline、MCOT、MSFT、MSFT+GoldRL、MAPO 等基线对比，PASMR 在 MGSM 与 MSVAMP 上平均提升 20–30%，低资源语言提升 30% 以上，并在 OOD 语料上保持鲁棒性

**⚠️ 局限性**

限制：仍受 pivot 翻译质量影响，过度依赖主语言对齐；对大模型的推理成本高；未验证在非数学推理或更广泛知识任务上的效果

---

## 372. Grammar-Aware Literate Generative Mathematical Programming with Compiler-in-the-Loop

**arXiv ID:** 2601.17670 | [PDF](https://arxiv.org/pdf/2601.17670v1)

**作者:** Roberto Rossi `[一作]` (Business School, University of Edinburgh), Steven D. Prestwich `[通讯]` (Insight Centre for Data Analytics, University College Cork)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于编译器循环的自然语言到代数建模语言（AML）生成系统SyntAGM，利用PyOPL编译器诊断和LLM评估实现生成–编译–评估–修订的迭代流程；

**💡 创新点**

创新点在于将AML语法以BNF形式做成上下文、使用编译器提供的具体错误反馈、引入可解释的“literate”模型注释以及对齐评判器，实现语法与语义双重约束的自适应修正；

**🔧 技术方法**

技术包括：PyOPL Python编译器、LLM（OpenAI GPT系列）作为生成器与评判器、RAG检索少量示例、在prompt中注入语法和例子、JSON输出约束、迭代修订策略；

**📊 数据集**

使用NL4Opt、ComplexOR、ReSocratic、IndustryOR四个现有OR基准以及自行构建的StochasticOR（10个两阶段/多阶段随机问题）作为评测数据集；

**📈 对比分析**

与Standard、Chain‑of‑Thought、Tree‑of‑Thoughts、Reflexion、Chain‑of‑Experts等基线在OpenAI gpt‑oss‑20b、GPT‑4.1、GPT‑5等模型上对比，SyntAGM在准确率上相当或略优，且显著降低token消耗（≈2–6k vs 60k）、成本和延迟（≈2–4min vs 10min），尤其在行业级难度和随机问题上表现更好；

**⚠️ 局限性**

局限性包括：依赖LLM和编译器的可用性，对新AML语法支持仍有限；在极复杂或多代理场景下迭代次数和成本仍高；缺乏对模型语义正确性（非数值目标）和多目标优化的全面评估。

---

## 373. Entropic Risk-Aware Monte Carlo Tree Search

**arXiv ID:** 2601.17667 | [PDF](https://arxiv.org/pdf/2601.17667v1)

**作者:** Pedro P. Santos `[一作]` (Instituto Superior Técnico), Francisco S. Melo `[通讯]` (Instituto de Estudos de Ciência e Tecnologia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种能够在风险感知马尔可夫决策过程中使用熵风险度量（ERM）的蒙特卡洛树搜索（MCTS）算法，并给出了其非渐近收敛与多项式 regret 集中性质的理论证明。

**💡 创新点**

创新点在于：①首次提供了针对ERM目标的 provably correct MCTS；②通过非渐近分析证明算法在根节点的经验ERM收敛到最优ERM；③在理论上证明了多项式 regret 收敛速率，从而填补了之前仅有经验或无理论保证的空白。

**🔧 技术方法**

主要技术包括：UCB‑based tree search、熵风险度量的动态规划展开、风险感知多臂赌博机的非平稳扩展以及对风险感知树搜索的递归理论推导。

**📊 数据集**

实验采用了一个四状态的MDP-4环境，用于展示不同β值下算法的性能。

**📈 对比分析**

与基准进行比较的方式是将提出的ERM‑MCTS与（i）使用后向归纳得到的oracle（ERM‑BI）以及（ii）通用风险感知MCTS（Acc‑MCTS）进行对比。实验结果显示，ERM‑MCTS在所有β值下均能匹配oracle的ERM值，并且在收敛速度和最终性能上优于Acc‑MCTS。

**⚠️ 局限性**

局限性包括：实验规模较小，仅在单一MDP-4环境上验证；对β值的敏感性未系统评估；且对大规模、复杂MDP的可扩展性尚未展示。

---

## 374. Training-Free Text-to-Image Compositional Food Generation via Prompt Grafting

**arXiv ID:** 2601.17666 | [PDF](https://arxiv.org/pdf/2601.17666v1)

**作者:** Xinyue Pan `[一作]` (Purdue University), Fengqing Zhu `[通讯]` (Purdue University)

**通讯引用:** 3871 | [OpenAlex ID](https://openalex.org/A5001380619)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究多食物图像生成中的对象纠缠问题，并提出训练无关的Prompt Grafting方法

**💡 创新点**

通过双阶段提示嫁接：先用布局提示生成分离区域，再用目标提示填充食物，实现可控的对象分离和组合

**🔧 技术方法**

基于Stable Diffusion v3的扩散模型，结合时间可变文本条件、负提示指导和动态切换时步的技术

**📊 数据集**

使用VFN和UEC‑256两份含多食物且带边界框标注的数据集进行评估

**📈 对比分析**

与SD、Structured Diffusion、Attend‑&‑Excite、Syngen、FLUX.1及SD3等方法比较，采用YOLOv11、BLIP VQA和FID指标，PG在召回率、F1和对象存在率上取得最优成绩

**⚠️ 局限性**

仍存在FID略高、对细粒度边界不易重建、复杂多层空间关系仍需手动指定，缺乏完全自动化的空间关系推理

---

## 375. UrduLM: A Resource-Efficient Monolingual Urdu Language Model

**arXiv ID:** 2601.17664 | [PDF](https://arxiv.org/pdf/2601.17664v1)

**作者:** Syed Muhammad Ali `[一作]`, Abdul Samad `[通讯]` (Habib University)

**通讯引用:** 3641 | [OpenAlex ID](https://openalex.org/A5087557706)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 UrduLM 100M 级单语Transformer模型，提供 33GB 语料、专属BPE tokenizer 以及公开评测基准。

**💡 创新点**

提出低资源语言单语LLM全流程，利用自定义 tokenizer 提高 20–30% 词化效率，且 100M 参数模型可与 30 倍参数多语模型媲美。

**🔧 技术方法**

使用 GPT‑3 结构的 decoder‑only Transformer，PyTorch DDP/FSDP 并行训练，BPE tokenizer 自定义正则，数据清洗+LSH 去重。

**📊 数据集**

构造 33GB Urdu 语料，包含新闻、文学、网页、OCR 图书、翻译内容，最终约 5–6B tokens；评测集来自改编与翻译的 Urdu benchmark。

**📈 对比分析**

与 LLaMA、Qwen、Gemma 等多语模型在 5‑shot 情感分类、语法纠错、问答等任务对比，UrduLM 100M 在情感分类 66.6% 精度、语法纠错 30.59 BLEU，达到 30 倍参数模型 70–85% 的效果。

**⚠️ 局限性**

计算资源受限未训练更大规模或替代架构；语料仍缺少方言、专业域；评测缺乏人工基准，模型偏见与安全评估不足。

---

## 376. A Systemic Evaluation of Multimodal RAG Privacy

**arXiv ID:** 2601.17644 | [PDF](https://arxiv.org/pdf/2601.17644v1)

**作者:** Ali Al-Lawati `[一作]` (Pennsylvania State University), Suhang Wang `[通讯]` (Pennsylvania State University)

**通讯引用:** 18039 | [OpenAlex ID](https://openalex.org/A5011048500)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在多模态检索增强生成（mRAG）系统中，针对视觉输入进行成员推断（MIA）与图像标题检索（ICR）攻击，评估其隐私泄露风险。

**💡 创新点**

系统性地将 MIA 与 ICR 结合到 mRAG，考虑图像变换、检索/重排序、提示结构等因素；首次量化不同变换对攻击效果的影响。

**🔧 技术方法**

使用 CLIP 作为检索器、Jina‑Reranker 作为重排序器，配合 Qwen2.5‑VL、Cosmos‑Reason1、InternVL3.5 等 VLM；采用黑盒 API 攻击；利用 BLEU、ROUGE、METEOR 等文本相似度度量。

**📊 数据集**

在 Conceptual Captions、ROCOv2、Pokemon Blip、mRAG‑Bench 四大视觉数据集上，插入 50% 作为成员样本进行评估。

**📈 对比分析**

在精确图像下 MIA 的 F1 近 1，变换后仍保持 0.6‑0.96；ICR 的 exact‑match 在 68% 以上，受检索覆盖率影响；大规模检索/更大数据库可降低泄露。

**⚠️ 局限性**

仅针对视觉 mRAG，未考虑文本/语音/视频场景；实验仅在中小型 VLM 上；对缓解措施讨论有限，仅尝试 LLM‑in‑the‑middle 方案。

---

## 377. Representative Litigation Settlement Agreements in Artificial Intelligence Copyright Infringement Disputes: A Comparative Reflection Based on the U.S

**arXiv ID:** 2601.17631 | [PDF](https://arxiv.org/pdf/2601.17631v1)

**作者:** Chanhou Lou `[一作]` `[通讯]` (University of Macau), Chanhou Lou (University of Macau)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

对生成式 AI 引发的版权纠纷进行制度化分析，提出代表性诉讼和集体和解协议作为结构化治理工具，并针对中国法律环境给出可行性路径。

**💡 创新点**

创新点在于将代表性诉讼和解视为“程序性市场制造”，通过集体和解形成市场定价、商业习惯和公平使用阻挡效应，并提出三项关键解释性调整（同类权利扩展、混合注册/确认机制、退出式同意权）。

**🔧 技术方法**

采用法律比较研究、案例分析和理论建模方法，结合美国 Bartz 案和中国民事诉讼法的条款进行推演。

**📊 数据集**

使用的“数据集”主要是美国 Bartz 集体诉讼的和解文件、庭审记录以及相关法律条文与判例；中国侧则以民事诉讼法第57条及证券案件代表诉讼规定为参考。

**📈 对比分析**

比较方法：先概述 Bartz 案在美国的程序与结果，再将其关键要素映射至中国法律框架，评估在解释性调整下的可行性。由于缺乏量化指标，性能评估以理论可行性和制度效益为准。

**⚠️ 局限性**

限制包括：1）中国法律对同类权利解释和代表诉讼的传统保守；2）缺乏公开的版权登记数据库，混合机制实施成本高；3）对“同意”要求的退出式转化可能面临司法解释不一和执行难题；4）案例仅基于单一美国案例，缺乏多样化的实证支持。

---

## 378. Advancing Structured Priors for Sparse-Voxel Surface Reconstruction

**arXiv ID:** 2601.17720 | [PDF](https://arxiv.org/pdf/2601.17720v1)

**作者:** Ting-Hsun Chi `[一作]` (National Taiwan University), Yu-Chiang Frank Wang `[通讯]` (NVIDIA)

**通讯引用:** 6628 | [OpenAlex ID](https://openalex.org/A5090045508)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于稀疏体素（SVO）的表面重建框架，结合自适应体素初始化和细化深度监督，实现快速且高精度的3D重建。

**💡 创新点**

创新点包括：① 通过多视角预训练模型生成伪几何先验，按像素实际投影面积动态分配体素分辨率，实现自适应细节级别初始化；② 引入“精细深度监督”机制，利用多视角交叉匹配直接给每条射线精确深度标签；③ 设计拓扑对齐融合算法和不确定性加权不透明度预测，提升初始化的完整性与鲁棒性。

**🔧 技术方法**

使用技术包括：多视角预训练 3D 视觉模型（如 VGGT）、基于 TSDF 的体素不透明度映射、可微分体素光栅化（SVRaster）与 SVO 优化、跨视角 NCC 深度匹配、可微分梯度裁剪与加权损失。

**📊 数据集**

主要数据集：DTU 体素重建基准；实验中也引用了多种公开基准（如DTU、其他公开场景）进行对比。

**📈 对比分析**

与多种基线（VolSDF、NeuS、2DGS、SVRaster、GeoSVR 等）对比，平均 Chamfer 距离最低，训练时间约 0.4h，显著加速收敛并提升几何精度，尤其在细节保持与完整性上优于现有稀疏体素方法。

**⚠️ 局限性**

局限性：依赖预训练模型生成的伪几何先验，若先验误差较大会影响初始化质量；精细深度监督需要多视角重投影，计算量相对较高；在纹理稀薄或光照不均的区域，深度匹配可能不稳健；方法目前在大规模场景与极端光照条件下的可扩展性尚待验证。

---

## 379. CaSNet: Compress-and-Send Network Based Multi-Device Speech Enhancement Model for Distributed Microphone Arrays

**arXiv ID:** 2601.17711 | [PDF](https://arxiv.org/pdf/2601.17711v1)

**作者:** Chengqian Jiang `[一作]` (University of Science and Technology of China), Haoyin Yan `[通讯]` (University of Science and Technology of China)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5036925221)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了轻量化的压缩发送网络 CaSNet，用于分布式麦克风阵列的语音增强，减少了数据传输量。

**💡 创新点**

创新点在于将单通道特征通过 SVD 进行低秩压缩，并在中心节点使用交叉窗口查询（CWQ）对齐特征，从而在保持高质量增强的同时显著降低通信成本。

**🔧 技术方法**

采用了 STFT 预处理、单通道 U‑Net 编码器、双路径 RNN 递归编码、SVD 低秩压缩、跨窗口注意力 CWQ、联合解码和 Griffin‑Lim 逆 STFT 等技术。

**📊 数据集**

使用了公开的 WSJ0‑WHAM! 与 RealMAN 两个数据集进行实验。

**📈 对比分析**

与 FaSNet、DFSNet、EaBNet、McNet 等多种 SOTA 多通道语音增强模型在 PESQ、STOI、COVL 等指标上对比，CaSNet 在相同或更低的数据传输量下实现了相近或更优的性能，并且支持任意数量的麦克风。

**⚠️ 局限性**

限制在于对低秩压缩参数较为敏感，过度压缩会影响质量；此外，实验主要在模拟/室内噪声环境下验证，尚未在真实实时部署或动态麦克风配置中充分验证。

---

## 380. A PUF-Based Security Framework for Fault and Intrusion Detection

**arXiv ID:** 2601.17661 | [PDF](https://arxiv.org/pdf/2601.17661v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 381. An AI-enabled tool for quantifying overlapping red blood cell sickling dynamics in microfluidic assays

**arXiv ID:** 2601.17703 | [PDF](https://arxiv.org/pdf/2601.17703v1)

**作者:** Nikhil Kadivar `[一作]`, Mengjia Xu `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 1960 | [OpenAlex ID](https://openalex.org/A5027699930)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套自动化深度学习框架，用于在微流控时间序列视频中定量分析红细胞（RBC）镰状化动态。

**💡 创新点**

创新点在于结合AI辅助标注、nnU-Net分割与标记控制的分水岭后处理，能够在细胞高度拥挤、重叠的条件下实现实例级计数并显著提升实验通量。

**🔧 技术方法**

使用Roboflow平台进行半自动标注、nnU-Net v2做分割、Watershed分水岭分割、CLAHE增强、OpenCV/Scikit-Image/Matplotlib等工具。

**📊 数据集**

数据集来自SCD患者血液，在双层PDMS微流控芯片中以不同细胞密度和药物处理（osivelotor）记录的高分辨率时间序列图像，总共约12帧用于训练，后续视频用于验证。

**📈 对比分析**

与人工计数对比，平均绝对误差在0.02-0.04之间；在不同密度和药物处理下模型预测的镰状细胞比例与人工测量高度一致，且可将实验通量提高2.5倍。

**⚠️ 局限性**

局限性包括训练样本极少、对极高密度或形态变化极端的细胞可能仍出现分割误差，且目前仅支持二维平面图像，未涵盖时间追踪或三维重建。

---

## 382. Agentic reinforcement learning empowers next-generation chemical language models for molecular design and synthesis

**arXiv ID:** 2601.17687 | [PDF](https://arxiv.org/pdf/2601.17687v1)

**作者:** Hao Li `[一作]` (Peking University), Li Yuan `[通讯]` (International Digital Economy Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

论文内容不完整，无法概括主要工作

**💡 创新点**

暂无可识别的创新点

**🔧 技术方法**

缺少技术细节

**📊 数据集**

未提及数据集

**📈 对比分析**

未说明比较方法与性能

**⚠️ 局限性**

论文限制无法评估

---

## 383. Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction

**arXiv ID:** 2601.17668 | [PDF](https://arxiv.org/pdf/2601.17668v1)

**作者:** Jang-Hyun Kim `[一作]` (NAVER AI Lab), Sangdoo Yun `[通讯]` (NAVER AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于门控的KV缓存压缩方法Fast KVzip，使大语言模型在保持近乎无损的推理性能下显著减小KV缓存。

**💡 创新点**

创新点在于利用轻量化的sink‑attention门控模块预测KV重要性，并通过只用前向传播的门训练方案避免重构与反向传播，兼顾prefill和decoding阶段，实现高达70%压缩而几乎不损失性能。

**🔧 技术方法**

采用门控机制、sink‑attention架构、基于KVzip重构目标的门训练、冻结LLM权重、Chunked prefill、缓存缓冲并行门计算、FP16/FP8量化等技术。

**📊 数据集**

门训练使用FineWeb‑Edu数据集，评估基准包括SCBench、MRCR、SQuAD、RULER、LongBench、AIME24、MATH等长上下文推理任务。

**📈 对比分析**

与KVzip、SnapKV、Expected Attention、DuoAttention、R‑KV、TrimKV等基线对比，在prefill/decoding任务上保持>95%原始性能，压缩至30%（甚至更低）KV预算，同时显著降低内存占用和推理延迟。

**⚠️ 局限性**

局限性包括：门训练仅在冻结模型权重下进行，可能不适用于需要结构改动的场景；目前仅验证单机GPU环境，分布式多机可扩展性未知；对极端长序列或特定任务仍存在轻微性能下降。

---

## 384. A Mosco sufficient condition for intrinsic stability of non-unique convex Empirical Risk Minimization

**arXiv ID:** 2601.17646 | [PDF](https://arxiv.org/pdf/2601.17646v1)

**作者:** Karim Bounja `[一作]` (Hassan 1st University), Abdeljalil Sakat `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了针对非唯一凸经验风险最小化（ERM）的内在稳定性概念，并给出了在Hilbert空间上以Painlevé–Kuratowski上半连续性（PK‑u.s.c.）为核心的集合层Well‑Posedness理论。

**💡 创新点**

创新点在于：①把PK‑u.s.c.确立为非唯一凸ERM的本质稳定性判据；②在Mosco收敛与局部有界极小化子集的假设下，证明了集合层的Hadamard稳定性；③利用二次增长（quadratic growth）提供了可量化的误差上界；④阐明了强凸正则化如何通过统一的二次增长保证稳定性，并区分了内在ERM不稳定与求解器/正则化引起的选择性不稳定。

**🔧 技术方法**

使用了变分分析中的Mosco收敛、Painlevé–Kuratowski极限、二次增长误差界、强凸正则化（Tikhonov）等工具，并结合Hilbert空间的弱/强收敛性质进行证明。

**📊 数据集**

该工作为理论研究，未使用具体实验数据集，主要针对一般凸损失函数的数学性质进行讨论。

**📈 对比分析**

由于本研究为理论分析，没有直接实验比较；若要评估可行性，可通过数值实验验证二次增长误差界和正则化稳定性，但本文未给出实验结果。

**⚠️ 局限性**

局限性包括：①仅适用于凸且不严格凸的ERM问题；②在缺乏局部有界性或Mosco收敛时无法保证PK‑u.s.c.；③对非凸损失的推广仍待研究；④对实际高维学习模型中的计算复杂度与可实现性未作讨论。

---

## 385. Distance-to-Distance Ratio: A Similarity Measure for Sentences Based on Rate of Change in LLM Embeddings

**arXiv ID:** 2601.17705 | [PDF](https://arxiv.org/pdf/2601.17705v1)

**作者:** Abdullah Qureshi `[一作]` (DataKnife), Alexander Wolpert `[通讯]` (Roosevelt University)

**通讯引用:** 207 | [OpenAlex ID](https://openalex.org/A5089677759)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种新的文本相似度度量——距离-距离比（DDR），并通过对句子进行词语替换实验验证其效果。

**💡 创新点**

引入了基于 Lipschitz 连续性理念的 DDR，用前后上下文嵌入距离比来衡量语义变化，理论上更稳健且能捕捉细粒度语义差异。

**🔧 技术方法**

采用 LLM 词嵌入（如 GPT 样式）、余弦相似度、Chordal 距离、Earth Mover’s Distance 等评估技术，并进行 CDF、EMD、Pearson 相关等统计分析。

**📊 数据集**

使用来自《The Hacker Crackdown》和《Dracula》共 500 段摘录，生成不同深度（1、2、3 词）词替换变体，并保持句长不变。

**📈 对比分析**

将 DDR 与基于均值池化的 Centroid、EOS 端点（两者皆用余弦相似度）进行对比，DDR 在所有编辑深度下均表现出更大的 EMD、低相关性和更明显的分布分离，优于传统方法。

**⚠️ 局限性**

仅适用于长度相同的序列，需进一步扩展到可变长度输入。

---

## 386. OwlerLite: Scope- and Freshness-Aware Web Retrieval for LLM Assistants

**arXiv ID:** 2601.17824 | [PDF](https://arxiv.org/pdf/2601.17824v1)

**作者:** Saber Zerhoudi `[一作]` (University of Passau), Jelena Mitrovic `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了浏览器插件式的检索增强生成系统OwlerLite，实现用户可控的检索范围与内容新鲜度；

**💡 创新点**

将用户定义的检索“scope”与语义新鲜度检测并入检索模型，提供可解释的范围与版本信息；

**🔧 技术方法**

采用SimHash+语义去重的变化检测、LightRAG的向量与知识图检索、浏览器插件+FastAPI后端等技术；

**📊 数据集**

使用MS MARCO V2.1语义段落集、TREC 2024 RAG数据集以及通过聚类得到的合成scope；

**📈 对比分析**

与稠密检索基线比较，在TREC 2024 RAG上将scope‑fidelity从0.64提升至0.83，scope‑leakage降至0.17，NDCG@10仅下降0.008；

**⚠️ 局限性**

依赖阈值启发式变更检测、合成scope，缺乏真实用户范围和大规模评估，未涉及多模态内容验证与完整的内容真实性检查。

---

## 387. ViTCoP: Accelerating Large Vision-Language Models via Visual and Textual Semantic Collaborative Pruning

**arXiv ID:** 2601.17818 | [PDF](https://arxiv.org/pdf/2601.17818v1)

**作者:** Wen Luo `[一作]` (Huazhong University of Science and Technology), LiQun Huang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 2877 | [OpenAlex ID](https://openalex.org/A5101820815)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种三阶段视觉‑文本协同剪枝框架 ViTCoP，能在保持高性能的前提下显著减少大型视觉‑语言模型的视觉标记数目。

**💡 创新点**

创新点包括：①在视觉编码器做粗粒度剪枝，②在 LLM 的浅层使用视觉信息聚类（VIC）与文本注意力（K‑vector L2 范数）协同选择多样且相关的标记，③在深层使用仅文本关注度进行精细剪枝；并提出 K‑vector L2 范数作为兼容 FlashAttention 的高效显著性度量。

**🔧 技术方法**

采用 Transformer 视觉编码器、跨模态注意力、K‑vector L2 范数显著性评估、视觉信息聚类算法 VIC、层级多阶段剪枝策略，以及与 FlashAttention 的兼容实现。

**📊 数据集**

在 11 个图像‑文本基准（COCO、Flickr30k、GQA、MMBench、MME、NoCaps、OK‑VQA、POPE、QBench、ScienceQA、VQA‑v2）和 4 个视频‑文本基准（EgoSchema、MVBench、Next‑QA、Video‑MME）上进行评估。

**📈 对比分析**

与 FastV、PyramidDrop、SparseVLM、VisionZip 等主流剪枝方法对比，ViTCoP 在 88.9% 以及 94.4% 的极端压缩率下分别保持 95.1% 与 90.8% 的平均性能，并在视频任务上达 97.7% 的性能保持；同时 TFLOPs 降低 94% 以上、前置延迟减少 85%、GPU 内存显著下降。

**⚠️ 局限性**

局限性：①对超大规模模型（如 13B 以上）仍需进一步验证；②需根据不同模型和数据设定裁剪阈值（d_c、τ 等），可能影响迁移性；③依赖 K‑vector L2 范数，若模型不支持 FlashAttention 可能效果下降；④在极端压缩下仍可能出现信息缺失导致的任务特异性性能下降。

---

## 388. Motif Diversity in Human Liver ChIP-seq Data Using MAP-Elites

**arXiv ID:** 2601.17808 | [PDF](https://arxiv.org/pdf/2601.17808v1)

**作者:** Alejandro Medina `[一作]` (Baylor University), Mary Lauren Benton `[通讯]` (Baylor University)

**通讯引用:** 549 | [OpenAlex ID](https://openalex.org/A5072386188)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文将 DNA 模式发现问题重新表述为质量-多样性问题，并使用 MAP‑Elites 算法在 CTCF ChIP‑seq 数据上演化 PWM 模式，得到多样化且高质量的模式集合。

**💡 创新点**

创新之处在于将 MAP‑Elites 这一光照式搜索方法引入生物序列模式发现，通过行为特征维度（信息含量与支持、GC 含量与熵、支持与分数尾行为）实现对模式特异性、覆盖度和稳健性的可解释探索。

**🔧 技术方法**

采用 MAP‑Elites 质量多样性搜索、位置权重矩阵（PWM）表示、基于对数似然的前景/背景评分、三种行为特征对、以及 pyribs 库实现的 IsoLineEmitter。

**📊 数据集**

使用来自 ENCODE 项目的 Human CTCF ChIP‑seq 数据（hg38），将前景峰序列划分为五个子集并匹配背景区域。

**📈 对比分析**

与经典单解工具 MEME 在相同数据集和评估函数下进行对比；MAP‑Elites 在平均 fitness 方面略低于 MEME 的最高峰值，但在多样性、平均表现和稳健性上更优，且归档覆盖率高。

**⚠️ 局限性**

实验仅局限于单一 TF、固定长度、单一 QD 算法；fitness 仅关注前景-背景区分，未评估其他指标，且未在更广泛的数据集上验证。

---

## 389. Token-Weighted Multi-Target Learning for Generative Recommenders with Curriculum Learning

**arXiv ID:** 2601.17787 | [PDF](https://arxiv.org/pdf/2601.17787v1)

**作者:** Wei-Ning Chiu `[一作]` (National Taiwan University), Pu-Jen Cheng `[通讯]` (National Taiwan University)

**通讯引用:** 728 | [OpenAlex ID](https://openalex.org/A5071684622)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了生成式推荐系统中token的重要性，提出了基于信息增益的前位权重（Front‑Greater）与频率权重（Frequency）两种token权重策略，并设计了多目标学习与课程学习相结合的训练框架。

**💡 创新点**

创新点在于：①将语义ID的层级前缀信息与token出现频率分别量化为信息增益权重；②将三种目标（前位权重、频率权重、标准交叉熵）联合优化，并通过可学习的logit缩放与指数课程学习动态调节权重，实现从粗到细的自适应训练。

**🔧 技术方法**

技术包括：RQ‑VAE生成语义ID、T5/LLM生成器、信息增益计算（前缀条件方差下降）、有效样本数（Effective Number）频率权重、可学习的logit缩放、多目标损失组合与指数课程学习。

**📊 数据集**

使用的数据集为Amazon Musical Instruments、Amazon Industrial & Scientific、Yelp、MovieLens 1M四个公开基准。

**📈 对比分析**

与传统推荐器（GRU4Rec、SASRec）以及多种token权重基线（Pos、CFT、IGD、TIGER）对比，Hit@5提升约6%、NDCG@5提升约7%，在head与tail项目上均有显著改进，实验结果显著优于所有对比方法。

**⚠️ 局限性**

局限性在于：仍依赖于语义ID的质量，对极度稀疏或冷启动场景的表现尚未充分验证；课程学习速率参数需要手动调节；实验仅覆盖四个数据集，进一步的跨域泛化需要验证。

---

## 390. Neuro-Symbolic Verification on Instruction Following of LLMs

**arXiv ID:** 2601.17789 | [PDF](https://arxiv.org/pdf/2601.17789v1)

**作者:** Yiming Su `[一作]` (University of Illinois Urbana-Champaign), Tianyin Xu `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2396 | [OpenAlex ID](https://openalex.org/A5027605695)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种神经-符号框架（Neuro‑Symbolic Verifier）用于验证LLM是否遵循指令，能够处理任意输入输出对。

**💡 创新点**

将指令转化为约束满足问题（CSP），区分逻辑约束和语义约束，利用符号推理（SMT）与LLM生成检查器相结合，提供可解释的细粒度反馈。

**🔧 技术方法**

使用LLM进行约束提取、分类和代码生成，利用Z3 SMT求解器进行逻辑验证，整体通过多智能体协同实现。

**📊 数据集**

构建了新的VIFBench基准，包含820条标注样本，涵盖多种逻辑与语义约束，支持细粒度标签。

**📈 对比分析**

与基线LLM‑as‑a‑judge、CoT优化、对话式评估和仅SMT求解器等方法对比，使用GPT‑4.1、DeepSeek、Qwen等模型，评估指标为F1、精确率、召回率和Pass@1；完整系统在GPT‑4.1上F1≈95%，显著优于基线（最多提升25%）并证明SMT推理的有效性。

**⚠️ 局限性**

依赖LLM的推理与代码生成，易受幻觉、错误分类影响；无法区分软硬约束；目前仅覆盖英文写作任务，未考虑多语言与更复杂领域；对不可满足指令的处理不完善。

---

## 391. MMR-Bench: A Comprehensive Benchmark for Multimodal LLM Routing

**arXiv ID:** 2601.17814 | [PDF](https://arxiv.org/pdf/2601.17814v1)

**作者:** Haoxuan Ma `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**通讯引用:** 3174 | [OpenAlex ID](https://openalex.org/A5065180062)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 MMR‑Bench，一个统一的离线成本感知多模态 LLM 路由基准，并基于该基准研究了多模态路由的有效性。

**💡 创新点**

创新点包括：①构建全局可复现的离线基准，提供每个实例-模型对的预计算 utility 与统一成本；②首次系统评估多模态信息在路由中的作用，证明自适应多模态融合可在约 33% 成本下匹配最强单模型；③展示了路由策略在跨任务、跨数据集以及跨模态（文本→多模态）上的零样本迁移能力。

**🔧 技术方法**

主要技术手段包括多模态特征融合（自适应权重 + 交互项）、k‑means / KNN 聚类、矩阵分解（LinearMF）与 MLP 预测器、离线评估指标 nAUC、Peak Score（P_s）与 Quality‑Neutral Cost（QNC），以及统一的成本归一化方案。

**📊 数据集**

使用的数据集涵盖三大场景：OCR（OCRBench、SEED‑Bench v2 Plus）、通用 VQA（MMStar、RealWorldQA）和多模态算术/图形推理（MathVerse、MathVista、MathVision），以及全数据集组合；同时在文本任务上进行零样本迁移评估（GSM8K、MMLU、ARC）。

**📈 对比分析**

通过预计算的实例‑模型表进行离线路由评估，比较路由器与单模型、Oracle 的 cost‑accuracy 曲线。结果表明：多模态路由在所有场景下均实现了更高的 nAUC，峰值准确率超过大多数单模型，且在仅占最强单模型成本 33% 的情况下即可达到相同或更高的准确率。

**⚠️ 局限性**

局限性包括：①基准仅涵盖有限的模型与任务，未覆盖所有可能的多模态 LLM；②路由策略仍需在真实在线推理环境中验证多模型调用的实际延迟与成本；③对模型间的可解释性和动态预算适应性尚未深入探究，且自适应融合在极端模态缺失情况下仍可能表现不佳。

---

## 392. Multi-Agent Collaborative Intrusion Detection for Low-Altitude Economy IoT: An LLM-Enhanced Agentic AI Framework

**arXiv ID:** 2601.17817 | [PDF](https://arxiv.org/pdf/2601.17817v1)

**作者:** Hongjuan Li `[一作]` (Jilin University), Abbas Jamalipour `[通讯]` (University of Sydney)

**通讯引用:** 16735 | [OpenAlex ID](https://openalex.org/A5086268677)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于LLM增强的代理式多智能体协同入侵检测框架，专为低空经济IoT网络设计；

**💡 创新点**

创新点在于将LLM与代理式AI架构融合，形成感知‑记忆‑推理‑行动四层循环，并通过自监督特征抽取、LLM驱动的特征选择与轻量化分类器实现零/少样本适应；

**🔧 技术方法**

采用的技术包括自监督扩散模型(DDPM)提取流量视觉特征、粒子群优化与LLM协同的特征筛选、LLM推理模块、以及轻量级分类器(如LightGBM)与边缘部署策略；

**📊 数据集**

实验使用了Edge‑IIoTset、USTC‑TFC和ISCX‑VPN三大真实网络流量数据集；

**📈 对比分析**

与传统监督学习(2D‑CNN、RBLJAN)和自监督方法(YaTC、MTC‑MAE)对比，框架在少样本情境下实现90%以上准确率，且标注数据需求显著降低，检测延迟更短；

**⚠️ 局限性**

局限性包括对数据稀缺与异构数据融合不足、对多智能体协同通信鲁棒性缺乏验证，以及LLM与自监督模型在边缘设备上部署时的算力与能耗瓶颈。

---

## 393. How Do We Evaluate Experiences in Immersive Environments?

**arXiv ID:** 2601.17811 | [PDF](https://arxiv.org/pdf/2601.17811v1)

**作者:** Xiang Li `[一作]` (University of Cambridge), Per Ola Kristensson `[通讯]` (University of Cambridge)

**通讯引用:** 6683 | [OpenAlex ID](https://openalex.org/A5042452579)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对375篇关于沉浸式体验评估的论文进行系统性梳理与编码，绘制现状与趋势。

**💡 创新点**

首次从域、方法、构念三维度综合描绘评估实践，指出方法多样但缺乏统一标准，并提出开放评估生态与混合建模的路径。

**🔧 技术方法**

采用PRISMA‑ScR、关键词检索、双人独立编码与共识、统计分析等技术手段。

**📊 数据集**

基于ACM CHI、UIST、VRST、SUI、IEEE VR、ISMAR、TVCG等七大顶级会议/期刊共375篇文献。

**📈 对比分析**

通过统计问卷、任务、系统与生理等测量组合与时序变化，展示不同设备与应用域的评估偏好，但未给出单一指标的性能数值。

**⚠️ 局限性**

研究仅覆盖七大会议/期刊，检索受关键词限制，短篇排除，编码缺乏可靠性指标，难以完全覆盖跨学科沉浸式研究。

---

## 394. A Multi-Modal Fusion Platform for Joint Environment Sensing and Channel Sounding in Highly Dynamic Scenarios

**arXiv ID:** 2601.17809 | [PDF](https://arxiv.org/pdf/2601.17809v1)

**作者:** Xuejian Zhang `[一作]` (Beijing Jiaotong University), Ziyi Qi `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 61 | [OpenAlex ID](https://openalex.org/A5100310498)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文设计并实现了一个多模态融合平台，可在高速移动场景下同步采集子6 GHz及28 GHz宽带信号、LiDAR点云、全景图像及GNSS位置信息，构建可用于6G ISAC 的环境–通道联合数据库。

**💡 创新点**

核心创新点包括：①模块化硬件架构实现跨频段（Sub‑6 GHz 与 mmWave）和大带宽（1 GHz）测量；②厘米级 360° 环境感知与米级定位的时空同步方案；③硬件与软件双向同步（Rubidium+GNSS、时钟分发、相机/雷达时间戳对齐）保证多模态数据一致；④首次在动态 V2I 现场验证多模态联合建模的可行性。

**🔧 技术方法**

使用技术：NI PXIe/FlexRIO 直采硬件，VSG/VSA 双频段信号发生与接收；双向上/下转换模块实现 27–29 GHz 直采；Rubidium 时钟 + GNSS 实现 1 PPS 时钟同步；LiDAR（OS1‑128）与全景摄像头（Insta360）采集点云/图像；IMU+SLAM 进行定位与姿态估计；后端数据融合、destaggering、SAGE 角度估计等算法。

**📊 数据集**

数据集为作者在北京郊区 V2I 路测（400 m 车道）自采集的多模态数据：28 GHz 1 GHz 带宽 SIMO 通道声学、LiDAR 点云、全景图像、GNSS 位置信息及 IMU 运动数据，未使用公开公开数据集。

**📈 对比分析**

通过与现有单模平台（如 5G‑R、mmWave 声学仪器）对比，评估路径损耗、功率时延分布、角分辨率和动态范围。实验表明：路径损耗指数 ≈ 2.07，最大可测损耗约 128 dB，延迟分辨率 1 ns，角分辨率 < 1°，动态范围 123–128 dB，可在 20–30 dB 级弱径路下保持高信噪；采样速率可达 50 / 100 Hz。

**⚠️ 局限性**

限制方面：①最短天线切换间隔 8 μs，未实现真正的多天线 MIMO；②仅覆盖 27–29 GHz mmWave，未实现更高频段；③对高速 (>100 km/h) 场景尚未验证；④硬件成本高，系统对同步硬件依赖强；⑤垂直视场仅 45°，对高楼/多层建筑的感知有限；⑥多模态融合算法尚未完全自动化，需人工校准。

---

## 395. Shortcut Learning in Binary Classifier Black Boxes: Applications to Voice Anti-Spoofing and Biometrics

**arXiv ID:** 2601.17782 | [PDF](https://arxiv.org/pdf/2601.17782v1)

**作者:** Md Sahidullah `[一作]` (TCG CREST), Tomi H. Kinnunen `[通讯]` (University of Eastern Finland)

**通讯引用:** 11355 | [OpenAlex ID](https://openalex.org/A5043168931)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了二分类器中的 shortcut learning 偏差，提出了结合干预与观测的统一框架，并用线性混合效应模型对检测器得分进行后验分析。

**💡 创新点**

在不需要了解模型内部的前提下，给出可用于任何黑盒二分类器的偏差检测与量化方法，并将干预与观测两种视角系统化。

**🔧 技术方法**

采用线性混合效应模型（LME）对得分回归，干预实验（添加噪声、MP3 压缩、静默填充等）与观测实验（估计 SNR、非语音比例）相结合，并进行统计显著性检验。

**📊 数据集**

使用 ASVspoof 2019 Logical Access 反欺骗数据集与 VoxCeleb 1/2 说话人验证数据集。

**📈 对比分析**

通过 EER、AUC 等评估指标与基准模型（LFCC‑GMM、AASIST‑RawBoost、ECAPA‑ASV）比较，结果显示干预可显著提升或降低 EER，证明存在显著的 shortcut；观测分析揭示噪声/静默比例对得分的显著影响。

**⚠️ 局限性**

仅针对二分类任务；仅考虑协变量偏移，未研究类别比例漂移；干预仅单一因素，未探讨多因素组合；依赖领域专家定义干预；实验仅在两个常用数据集上，未验证跨数据集的一致性。

---

## 396. Controlling Reading Ease with Gaze-Guided Text Generation

**arXiv ID:** 2601.17781 | [PDF](https://arxiv.org/pdf/2601.17781v1)

**作者:** Andreas Säuberli `[一作]` (LMU Munich), Barbara Plank `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将眼动预测模型融入语言模型的解码阶段，提出了一种在生成文本时可调节阅读难度且保持文本质量的实时可控文本生成方法。

**💡 创新点**

首次利用眼动追踪数据作为即时反馈来指导文本生成，实现了可解释的阅读难度控制机制，突破了传统提示或微调方式的缺陷。

**🔧 技术方法**

采用 Llama 3.2 instruction‑tuned 3B 语言模型、GPT‑2 124M 眼动预测模型，并使用 beam search 结合 gaze weight 对 token 概率进行重新排序。

**📊 数据集**

使用 EMTeC 眼动数据集（107 名 L1 英语读者的眼动记录）训练眼动模型，评估数据来自 EyeLink 1000 Plus 的眼动实验，并生成短篇故事文本。

**📈 对比分析**

与无眼动指导（weight 0）对比，正/负 gaze weight 可显著增加或减少 L1/L2 读者的 first‑pass reading time，主观难度评分随 gaze weight 变化，而文本自然度和有趣度保持相近，表明方法在保持文本质量的同时有效调节阅读难度。

**⚠️ 局限性**

仅关注词层面 FPRT 受词长/词频影响，缺乏对句法/语义层面的控制；仅在英语文本上验证，未评估跨语言、多模型场景，且眼动模型对低频词的预测仍有限。

---

## 397. Hylog: A Hybrid Approach to Logging Text Production in Non-alphabetic Scripts

**arXiv ID:** 2601.17753 | [PDF](https://arxiv.org/pdf/2601.17753v1)

**作者:** Roberto Crotti `[一作]` (University of Milano-Bicocca), Ricardo Muñoz Martín `[通讯]` (Alma Mater Studiorum University of Bologna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Hylog，一种混合键盘与文本日志系统，能够同步捕捉键盘事件与 IME 渲染文本，以实现对非拼音文字输入过程的完整记录。

**💡 创新点**

创新点在于将键盘日志与生态文本日志融合，提出动态快照窗口（DSW）算法和 hybridizer 模块，实现跨层级的精细时间同步，突破传统键盘或文本日志只能捕获单一层面的局限。

**🔧 技术方法**

采用 C# 与 VSTO 开发 Word 插件，JavaScript 开发 Chrome 插件；使用高分辨率单调时钟、diff 算法、Stanford CoreNLP 语义分词及自定义规则匹配实现同步与解析。

**📊 数据集**

使用两位受试者（L1 与 L2 中文译者）在 Windows 环境下完成英→简体中文翻译任务的键盘与文本日志数据，结合屏幕录像验证同步准确性。

**📈 对比分析**

与传统 Inputlog 9.5 对比，Hylog 在字符与 IME 确认层面获得了更完整、较长的 IKI 分布（平均 0.6–1.1 秒），显示更高的时间分辨率和对非字母输入的敏感性，验证了系统在捕捉 IME 相关事件上的优势。

**⚠️ 局限性**

局限性包括样本量极小、仅针对 Microsoft Pinyin IME 进行测试、对其他 IME 或非汉字脚本（如日语、阿拉伯语）需要额外规则与插件支持，且在大文本或多应用场景下的性能尚未充分评估。

---

## 398. Learning Sewing Patterns via Latent Flow Matching of Implicit Fields

**arXiv ID:** 2601.17740 | [PDF](https://arxiv.org/pdf/2601.17740v1)

**作者:** Cong Cao `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Hao Li `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 18253 | [OpenAlex ID](https://openalex.org/A5100348588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于隐式场的缝制图案建模框架，结合潜在空间、流匹配与缝合预测，支持图案生成、图像估计、补全与改形等完整工作流。

**💡 创新点**

创新点在于：①使用SDF+UDF隐式表征单板，构建可微分的连续潜在空间；②采用流匹配学习板块组合分布，消除模板束缚；③设计缝合预测模块捕捉局部几何与全局上下文；④将模型条件化于图像，实现端到端的图像到图案生成。

**🔧 技术方法**

技术组合包括变分自编码器（VAE）、Diffusion Transformer（流匹配）、点变换器、SDF/UDF隐式函数、DINO-3D与CLIP视觉编码、可微分网格化与布料模拟。

**📊 数据集**

使用Sewfactory与GCD两大公开缝制图案数据集，涵盖衬衫、裤子、连衣裙等多种服装。

**📈 对比分析**

与Sewformer与AIpparel两种基线相比，在Panel IoU、边数准确率、缝合Precision/Recall/F1等指标上均取得显著提升；在旋转误差略高但对下游布料模拟影响不大。

**⚠️ 局限性**

局限性包括：需依赖大量标注数据，极端多边形或非标准拓扑的图案仍难以完全捕捉；改形与补全需昂贵的可微分仿真；整体计算开销相对较高。

---

## 399. Athanor: Authoring Action Modification-based Interactions on Static Visualizations via Natural Language

**arXiv ID:** 2601.17736 | [PDF](https://arxiv.org/pdf/2601.17736v1)

**作者:** Can Liu `[一作]` (Nanyang Technological University), Yong Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 130619 | [OpenAlex ID](https://openalex.org/A5059976286)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新方法，使用户能够基于自然语言输入方便高效地为现有静态可视化添加交互功能。

**💡 创新点**

创新点在于通过自然语言接口，结合动作-修改交互设计空间、多代理需求分析器和实现无关的表示翻译器，解决了静态可视化交互的复杂性、用户需求多样性和可视化实现多样性等挑战。

**🔧 技术方法**

使用了多模态大型语言模型（MLLMs）来解析现有可视化，并通过约束模型进行表示和修改。

**📊 数据集**

使用了SVG格式的现有可视化数据集，特别是流行的图表类型，如条形图、折线图、散点图和面积图。

**📈 对比分析**

通过案例研究和用户访谈比较了方法的有效性，结果表明该方法能够有效满足用户需求，用户能够通过自然语言输入为静态可视化添加交互功能。

**⚠️ 局限性**

限制在于当前方法仅支持基于现有可视化的数据交互，无法处理需要外部数据的交互，且对复杂空间布局的可视化支持有限。

---

## 400. @NTT: Algorithm-Targeted NTT hardware acceleration via Design-Time Constant Optimization

**arXiv ID:** 2601.17806 | [PDF](https://arxiv.org/pdf/2601.17806v1)

**作者:** Mohammed Nabeel `[一作]` (New York University Abu Dhabi), Michail Maniatakos `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 3002 | [OpenAlex ID](https://openalex.org/A5043325974)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一套针对特定后量子密码算法的NTT硬件加速框架，通过在设计时将环参数作为常量并对常数乘法进行优化，实现了在每个时钟周期完成N点NTT的目标，并保持低功耗和高面积效率。

**💡 创新点**

创新点在于：①将环参数视为合成时常量，手动分解常数乘法为最少的移位与加/减，显著降低面积和延迟；②在同一蝶形单元中同时支持正向和逆向NTT；③在RTL层面完成优化而非仅靠综合工具。

**🔧 技术方法**

使用的技术包括：常数乘法（shift‑and‑add）最小化、Barrett模约简、完整流水线架构、FPGA/ASIC RTL生成、参数化Verilog/VHDL设计。

**📊 数据集**

数据集为NIST标准化的后量子密码算法参数，主要以Kyber（N=256, Q=3329）和Dilithium（N=256, Q=8380417）为例；对256点NTT进行仿真验证。

**📈 对比分析**

与现有最先进的FPGA实现比较，Dilithium的吞吐量提升5.2倍，Kyber提升8.5倍；ASIC上可在1 GHz下每纳秒完成一次256点NTT；面积比传统设计下降约30%，功耗也得到降低。

**⚠️ 局限性**

局限性在于：①仅适用于固定参数的算法，对参数可变的方案（如Falcon）效果有限；②实现深度流水线导致时序约束和功耗在极高频率下仍可能成为瓶颈；③框架对新的模乘实现需要重新适配。

---

## 401. Robust Computational Extraction of Non-Enhancing Hypercellular Tumor Regions from Clinical Imaging Data

**arXiv ID:** 2601.17802 | [PDF](https://arxiv.org/pdf/2601.17802v1)

**作者:** A. Brawanski `[一作]` (University Hospital), E. W. Lang `[通讯]` (CIML)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用深度学习在常规MRI上自动提取非强化高细胞肿瘤（NEH）区并验证其生物学与临床意义。

**💡 创新点**

提出双网络融合概率映射（PAUNet + UNETR++）的无监督评估策略，并用rCBV与复发位置等临床指标证明NEH的生物学相关性。

**🔧 技术方法**

PAUNet（改进的3D U‑Net）、UNETR++（Transformer‑Encoder U‑Net）以及UNETR‑staple++（融合后概率网络）三种深度学习架构。

**📊 数据集**

训练集：BraTS 2018/2021（包含NEH标签）；验证/临床对照集：UPenn‑GBM、RHUH glioblastoma。

**📈 对比分析**

与rCBV、ETRL等标志物比较，提升的Dice分别为ET 0.83、TC 0.78、WT 0.90；NEH区Dice仅 0.29；NEH区rCBV显著高于水肿区；空间指标（MeanEdgeDistance、FractionInside 等）与复发区显著相关，表明方法具有较高的临床可解释性。

**⚠️ 局限性**

缺乏组织学验证、NEH Dice低、模型依赖BraTS标签、临床样本量有限，限制了推广与进一步的精准治疗应用。

---

## 402. Agreement-Driven Multi-View 3D Reconstruction for Live Cattle Weight Estimation

**arXiv ID:** 2601.17791 | [PDF](https://arxiv.org/pdf/2601.17791v1)

**作者:** Rabin Dulal `[一作]` (Charles Sturt University), Jane Quinn `[通讯]` (Charles Sturt University)

**通讯引用:** 2854 | [OpenAlex ID](https://openalex.org/A5002248034)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一种基于RGB图像的多视角3D重建与集成回归管线，用于牛的非接触实时体重估计。

**💡 创新点**

提出了SAM 3D的多视角一致性融合策略，并将经典集成学习与深度学习模型在低样本场景下进行对比，证明一致性融合和集成模型在实用性上的优势。

**🔧 技术方法**

使用SAM 3D模型的多视角一致性融合、RGB图像分割、三维点云生成、统计几何特征提取以及11种传统机器学习回归器的堆叠集成。

**📊 数据集**

使用公开的103只牛的RGB三视角图像和对应体重的公开数据集（含点云数据）。

**📈 对比分析**

与RGB+D、单视角SAM 3D、TRELLIS2等重建方法以及传统ML和DL回归模型进行5折交叉验证比较，最终SAM 3D+一致性融合+集成回归取得R²≈0.69、MAE≈9.2 kg、MAPE≈2.2%，显著优于其他方案。

**⚠️ 局限性**

受限于样本量仅一只牛对应一组点云，深度学习模型过拟合；对遮挡和高动态姿态的鲁棒性仍待提升；缺乏大规模多品种畜禽数据验证。

---

## 403. Performance Analysis of Quantum-Secure Digital Signature Algorithms in Blockchain

**arXiv ID:** 2601.17785 | [PDF](https://arxiv.org/pdf/2601.17785v1)

**作者:** Tushar Jain `[一作]` `[通讯]` (Institute of Computer Science University of Tartu), Tushar Jain (Institute of Computer Science University of Tartu)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

构建了一个单节点区块链原型，支持多种基于格的后量子签名（Dilithium、Falcon、Hawk 以及 HAETAE），并对密钥生成、签名、验证以及区块级性能进行基准测试。

**💡 创新点**

首次将多种格签名方案统一集成到区块链框架中，并系统地比较它们在交易尺寸、区块大小和验证吞吐量等方面的整体性能，而非仅做微基准。

**🔧 技术方法**

使用 C/C++ 与 OQS 库（实现 Dilithium、Falcon）、Hawk 参考实现、SHA3‑256 哈希、区块与交易序列化以及高精度计时器等技术。

**📊 数据集**

采用 NIST 标准参数集（ML‑DSA‑44/65/87、Falcon‑512/1024、Hawk‑512/1024、HAETAE‑120/180/260）和 1000 条同步转账交易的合成数据集。

**📈 对比分析**

通过对每个算法进行 100 次密钥生成、1000 条交易签名、100 次区块验证等测量，结果显示 ML‑DSA 生成最快但签名尺寸最大；Falcon‑512 与 Hawk‑512 产生更小的区块（≈2.5–2.7 MB）并在 1000 交易区块上验证约 47–52 ms，验证吞吐量更高。

**⚠️ 局限性**

实验仅在单节点、无网络/共识环境下进行，交易类型有限（简单转账），HAETAE 仅做微基准未集成原型，结果受 Windows/MSVC 编译环境影响，缺乏多节点传播延迟和能耗等评估。

---

## 404. DPI: Exploiting Parameter Heterogeneity for Interference-Free Fine-Tuning

**arXiv ID:** 2601.17777 | [PDF](https://arxiv.org/pdf/2601.17777v1)

**作者:** Xiaoyu Liu `[一作]` (Northeastern University), Xianjie Wu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种动态参数隔离（Dynamic Parameter Isolation, DPI）的监督微调框架，能够在多任务训练中分离任务特定的核心参数并动态冻结，以减少任务间的干扰。

**💡 创新点**

创新点在于：①通过在每个任务上单独微调并计算参数更新幅度来识别核心参数；②利用核心参数的重叠度（Jaccard相似度）对任务进行聚类；③在多阶段训练中对前期任务的核心参数动态冻结，防止后续任务的破坏。

**🔧 技术方法**

技术方法包括：参数更新幅度测量、top‑k 核心参数筛选、Jaccard 相似度聚类、动态冻结掩码、基于多阶段 SFT 的训练策略。

**📊 数据集**

实验使用的公开数据集有：GSM8K（数学推理）、CodeAlpaca（代码生成）、LogiQA（逻辑推理）、Alpaca 与 UltraChat（指令跟随）。

**📈 对比分析**

与全量多任务 SFT、随机分组多阶段 SFT、启发式分组多阶段 SFT 等基线相比，DPI 在所有基模型（LLaMA‑2‑7B、Mistral‑8B、Qwen1.5‑7B、Gemma‑9B）和任务上均取得最高的平均归一化分数，显著提升性能。

**⚠️ 局限性**

局限性包括：①需要为每个任务单独微调以估计核心参数，增加预处理成本；②核心比例（p）和相似度阈值（τ）需要手工调优，可能不易迁移到极大规模或动态任务场景；③在任务数量极多时，聚类和多阶段训练的管理复杂度上升。

---

## 405. Reflexa: Uncovering How LLM-Supported Reflection Scaffolding Reshapes Creativity in Creative Coding

**arXiv ID:** 2601.17769 | [PDF](https://arxiv.org/pdf/2601.17769v1)

**作者:** Anqi Wang `[一作]` (Hong Kong University of Science and Technology), Pan Hui `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 20891 | [OpenAlex ID](https://openalex.org/A5029925982)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一套基于大语言模型的创意编码反思支架系统 Reflexa，整合对话式引导、版本导航与视觉生成，支持创作者在创作过程中的持续反思。

**💡 创新点**

创新点在于将多层反思模型（R1–R3）与可视化版本管理与即时迭代触发器相结合，形成系统级的反思支架，并通过实验验证其对创意质量与创作者体验的提升。

**🔧 技术方法**

使用 GPT‑4o 等 LLM 进行对话生成、代码生成及反思提示，结合 Vue3 前端、Monaco 编辑器、LangChain 与 ChromaDB 构建后端，配合 p5.js 进行可视化编码。

**📊 数据集**

数据集主要由 18 名具有创意编码与 LLM 使用经验的艺术家自行创作的 8 个项目构成，补充了 8 位专家的评测以及 18 份自评问卷。

**📈 对比分析**

采用 within‑subject 对照实验与自评/专家评估相结合，结果显示 Reflexa 在反思维度、控制感、合作感与创意质量（原创新性、美学等）均显著优于基线，且提升了创作者的主体性与 AI 依赖平衡。

**⚠️ 局限性**

限制包括缺乏单组件消融分析、对反思测度主要依赖自评、实验时长短、以及版本导航等附加功能可能混淆反思支架的独立效应。

---

## 406. Cross-Lingual Probing and Community-Grounded Analysis of Gender Bias in Low-Resource Bengali

**arXiv ID:** 2601.17764 | [PDF](https://arxiv.org/pdf/2601.17764v1)

**作者:** Md Asgor Hossain Reaj `[一作]` (American International University Bangladesh), Tze Hui Liew `[通讯]` (Multimedia University)

**通讯引用:** 73 | [OpenAlex ID](https://openalex.org/A5039685944)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了多种方法在低资源孟加拉语中的性别偏见检测与缓解效果。

**💡 创新点**

提出了社区驱动的本土化偏见识别框架，并证明现有英文化偏见检测方法在孟加拉语上适用性差。

**🔧 技术方法**

使用多语言嵌入检索、机器翻译、基于mBERT与BanglaBERT的分类、GPT生成、情感评分等技术。

**📊 数据集**

利用CORGI-PM、BUG、YouTube评论、FSB等四类数据集进行实验。

**📈 对比分析**

在各方法对比中，GPT生成获得最高偏见检测精度（90%），但多语言检索、翻译及分类方法的偏见发现率均低于30%。

**⚠️ 局限性**

主要局限包括跨语言迁移失真、社交媒体数据稀缺、词汇与语境不匹配以及生成文本缺乏多样性与文化深度。

---

## 407. AR-Omni: A Unified Autoregressive Model for Any-to-Any Generation

**arXiv ID:** 2601.17761 | [PDF](https://arxiv.org/pdf/2601.17761v1)

**作者:** Dongjie Cheng `[一作]` (Hong Kong Polytechnic University), Wenjie Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 11482 | [OpenAlex ID](https://openalex.org/A5100408983)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的自回归模型 AR-Omni，能够对文本、图像和语音进行感知、理解与生成，无需外部扩散解码器。

**💡 创新点**

创新点包括：将三种模态统一为单一离散词表实现任意输入-任意输出；通过任务感知加权损失解决模态不平衡；使用轻量化感知损失提升图像生成的视觉质量；引入有限状态解码器实现不同任务下的稳定与创意平衡。

**🔧 技术方法**

核心技术包括：离散化语音（单码本）、VQ 图像离散化、Transformer 自回归解码器、任务加权 NTP 损失、感知对齐损失、残差后归一化（swin‑norm）和有限状态解码器。

**📊 数据集**

使用多模态大规模预训练语料：Ultra‑FineWeb（文本）、LAION‑2B+LAION‑Aesthetics+JourneyDB（图文）、GigaSpeech、Common Voice、MLS（语音‑文本）等；并在 AnyInstruct、VoiceAssistant‑400K、UltraChat 等指令集上进行微调。

**📈 对比分析**

与现有 omni MLLM 对比，AR‑Omni 在文本、图像和语音三模态下都能实现自回归生成，并在实时语音生成上达到 0.88 RTF；在零样本 TTS WER 6.5、ASR WER 9.4、图像标题 CIDEr 56.53、CLIP‑score 0.24 等指标上取得与基线相近或更优表现；相较于依赖扩散解码器的模型，保持单一模型的优势。

**⚠️ 局限性**

局限性在于无扩散解码器的图像生成质量仍落后于扩散式方法，未来工作需提升无扩散图像生成的视觉表现。

---

## 408. MV-S2V: Multi-View Subject-Consistent Video Generation

**arXiv ID:** 2601.17756 | [PDF](https://arxiv.org/pdf/2601.17756v1)

**作者:** Ziyang Song `[一作]` (Hong Kong Polytechnic University), Zelin Zhao `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现多视角主旨视频生成（MV‑S2V），能够基于多视角参考图像合成保持 3D 主体一致性的高质量视频。

**💡 创新点**

创新点包括：① 将单视角 S2V 扩展为多视角任务；② 设计可控的合成数据采集管线，利用 S2I + I2V 生成大规模多视角视频并提取参考图像；③ 引入 Temporally Shifted RoPE（TS‑RoPE）实现跨主体与跨视角的区分；④ 针对 MV‑S2V 设计专属评估指标。

**🔧 技术方法**

采用扩散模型 Wan 2.1 + DiT、3D VAE、RoPE 与 TS‑RoPE、Rectified Flow 训练、UniPC + CFG 推理、Grounded SAM、DINO/CLIP/MEt3R 视觉相似度评估以及 π³ 3D 点云一致性评估。

**📊 数据集**

训练数据：约 22.9k 条合成视频（OC 11.8k，HOI 10.1k）+ 3.2k 条真实采集视频；评估使用 NAVI 35 个对象的 4 视角集合以及人工生成的人体参考图。

**📈 对比分析**

与公开基线 Phantom、MAGREF 的单视角与多视角版本对比，在 OC 与 HOI 场景下的多视角一致性、3D 一致性指标均优于基线，视觉质量和文本一致性保持竞争水平，说明 MV‑S2V 在主体一致性上具有显著优势。

**⚠️ 局限性**

局限性：① 仍高度依赖合成数据，极端真实场景下泛化可能受限；② TS‑RoPE 的时间偏移需手动设定，未能自动适应不同数据；③ 对多视角数量和质量的要求较高，过少或过差视角会影响一致性；④ 目前未充分处理多主体交互复杂性的动态变化。

---

## 409. Video Compression with Hierarchical Temporal Neural Representation

**arXiv ID:** 2601.17743 | [PDF](https://arxiv.org/pdf/2601.17743v1)

**作者:** Jun Zhu `[一作]` (University of Chinese Academy of Sciences), Jia Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 28772 | [OpenAlex ID](https://openalex.org/A5118788614)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种层次化时序神经表示 TeNeRV，用于视频压缩，并通过 IFF 模块聚合相邻帧特征、GAM 模块实现 GoP 自适应调制，实现局部时间一致性与长期语义一致性的统一建模。

**💡 创新点**

创新点包括：① 明确将时间维度分层处理，采用 Inter‑Frame Feature Fusion (IFF) 对相邻帧特征融合；② 引入 GoP‑Adaptive Modulation (GAM)，为每个 GoP 学习专属的特征嵌入和深度卷积核，动态调节网络参数；③ 结合内容感知 GoP 划分算法，自动适配场景切换；④ 通过深度可分离卷积与权重共享策略平衡内容适应性与参数效率。

**🔧 技术方法**

技术手段主要有：隐式神经表示 (INR) 与多分辨率时间网格；位置编码与可学习时间嵌入；深度可分离卷积和深度卷积核自适应调制；内容感知 GoP 划分算法；混合损失（MS‑SSIM + L1）；量化感知训练与熵编码。

**📊 数据集**

实验使用 UVG（7 条 1920×1080 视频）和 HEVC ClassB（5 条 1920×1080 视频）两大公开数据集。

**📈 对比分析**

与 HNeRV、FFNeRV、HNeRV‑Boost、HiNeRV 等 INR 基线以及传统编解码器 H.265/HM、H.266/VTM 和学习式压缩方法 DCVC 进行对比。TeNeRV 在 PSNR 与 MS‑SSIM 上均优于所有 INR 方案，平均提升 0.4–0.8 dB；在 RD 曲线下在 MS‑SSIM 量化下甚至超过 VTM；且在运动场景下恢复细节更清晰，时序稳定性更好。

**⚠️ 局限性**

局限性主要体现在：① 编码/解码耗时相对较长（约 2h35m），训练周期为 300 轮；② 仍需大量 GPU 资源和时间来得到高质量模型；③ 对极端动态或屏幕内容的泛化能力尚待进一步验证。

---

## 410. The Script is All You Need: An Agentic Framework for Long-Horizon Dialogue-to-Cinematic Video Generation

**arXiv ID:** 2601.17737 | [PDF](https://arxiv.org/pdf/2601.17737v1)

**作者:** Chenyu Mu `[一作]` (Tencent Hunyuan Multimodal Department), Linus `[通讯]` (Tencent Hunyuan Multimodal Department)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个端到端的对话到电影视频生成框架，先将粗略对白转换为可执行的电影脚本，再利用跨场连续生成技术将脚本映射为长时段连贯视频，并提供统一的评估体系。

**💡 创新点**

①引入“ScriptAgent”作为对话到脚本的专用LLM，通过两阶段（SFT + GRPO强化学习）训练实现专业导演水准的剧本生成；②设计了“Cross-Scene Continuous Generation”与帧锚定机制，克服现有模型固定时长导致的时间不连贯；③提出Visual‑Script Alignment (VSA) 指标，用来量化视频与脚本在时间语义上的对齐；④搭建大规模多模态脚本基准 ScriptBench，包含对白、音频与角色定位，支持专家级多轮校正。

**🔧 技术方法**

使用 Qwen‑Omni‑7B 作为基础模型；两阶段训练：监督微调（SFT）学习脚本结构，随后用 Group Relative Policy Optimization (GRPO) 结合结构奖励与人类偏好奖励进行对齐；视频生成采用跨场连续生成与帧锚定；评估使用自动化 AI 评估器、人工评审与 VSA 等指标。

**📊 数据集**

ScriptBench：3488 条含对白、音频与角色定位的多模态实例，平均时长 15.4 秒，划分 80% 训练/20% 测试；通过专家引导的三轮校正生成高质量可执行剧本。

**📈 对比分析**

与 StoryGen、StoryDiffusion、AutoStory、MovieAgent 等基线比较，ScriptAgent（完整模型）在 AI 与人工评测上均领先，尤其在格式遵循、情节连贯与视觉表现上提升 0.4–0.5 分；在视频生成层面，使用脚本作为输入可使 Sora2‑Pro、Veo‑3.1、Vidu‑Q2、Wan‑2.5 的脚本忠实度提升 0.2–0.8 分，VSA 指标提升 12–13 分，证明跨场连续生成显著改善时间语义一致性；同时观察到模型间存在“视觉壮观 vs 脚本忠实度” 的权衡。

**⚠️ 局限性**

仍面临音频-视频同步细节、面部同步、长时段身份保持、非照片级美术风格下的时间漂移等局限；模型在 15 秒以内的生成能力受限，难以处理更长的连续剧情；对不同美术风格的适配性不足，需要进一步的风格专用训练。

---

## 411. Unsupervised Elicitation of Moral Values from Language Models

**arXiv ID:** 2601.17728 | [PDF](https://arxiv.org/pdf/2601.17728v1)

**作者:** Meysam Alizadeh `[一作]` (University of Zurich), Zeynab Samei `[通讯]` (IPM)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在不使用人工标签的情况下，利用内部一致性最大化（ICM）算法从预训练语言模型中提取并强化道德推理能力。

**💡 创新点**

证明预训练模型已蕴含可通过无监督方式激活的道德认知，并展示ICM在多种道德基准上超过传统提示、聊天模型以及人类标注的效果。

**🔧 技术方法**

采用ICM算法进行自监督标签生成与微调，并构建无监督奖励模型与强化学习策略，实现无人工偏好标注的道德对齐。

**📊 数据集**

使用Norm Bank、ETHICS、UDHR三大基准数据集（各采样2024条样本），评估道德推理、框架泛化与社会偏见。

**📈 对比分析**

与零射（Base/Chat）、多射提示（Prompt-Human Chat）以及人类标注微调（FT-Human）等基线对比，ICM在Norm Bank达81%+准确率，ETHICS多框架均超越基线，UDHR社会偏差错误率下降至约4%，可与人类标注的微调相当或更优。

**⚠️ 局限性**

仍受模型内在知识与上下文窗口限制，某些道德范畴（如功利主义、宗教、性取向）提升有限；ICM效果依赖概念与预训练模型的相关性，且无监督方法难以覆盖全部社会多样性与价值演变。

---

## 412. VAE-REPA: Variational Autoencoder Representation Alignment for Efficient Diffusion Training

**arXiv ID:** 2601.17830 | [PDF](https://arxiv.org/pdf/2601.17830v1)

**作者:** Mengmeng Wang `[一作]` (Zhejiang University of Technology), Jingdong Wang `[通讯]` (Baidu)

**通讯引用:** 46175 | [OpenAlex ID](https://openalex.org/A5075880303)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在扩散变换器训练中，作者提出利用预训练的VAE特征进行内部对齐，以加速收敛并提升图像质量。

**💡 创新点**

创新点是通过轻量化的投影层和对齐损失，直接使用VAE重构特征作为指导，无需额外的外部表示编码器或双模型教师，显著降低训练成本。

**🔧 技术方法**

技术包括扩散变换器（SiT）、预训练VAE特征提取、投影MLP、smooth L1对齐损失以及标准的SDE求解器。

**📊 数据集**

主要使用ImageNet 256×256和MS‑COCO数据集进行实验。

**📈 对比分析**

与基线SiT、REPA、REG、MaskDiT、SRA等方法比较，VAE‑REPA在相同训练步数下实现更低的FID，收敛速度提升约7×，且仅增加4% GFLOPs。

**⚠️ 局限性**

局限性包括依赖预训练VAE的特征空间，可能对不同域或更高分辨率的任务适用性有限，且对层级选择、λ等超参仍需手动调节。

---

## 413. RegGuard: AI-Powered Retrieval-Enhanced Assistant for Pharmaceutical Regulatory Compliance

**arXiv ID:** 2601.17826 | [PDF](https://arxiv.org/pdf/2601.17826v1)

**作者:** Siyuan Yang `[一作]` (Xi'an Jiaotong-Liverpool University), Jiayin Tang `[通讯]` (Roche Diagnostics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并部署了RegGuard，一个面向制药监管合规的 AI 辅助系统，自动解析多格式法规文本，检索并生成符合内部政策的答案。

**💡 创新点**

核心创新是 HiSACC 的层次语义聚合分块技术，能够跨段落维护语义连贯；以及 ReLACE 的领域自适应列表式交叉编码重排序器，提升检索相关性并降低幻觉风险。

**🔧 技术方法**

技术栈包括：LLM（GPT‑4 Turbo / GPT‑4o）、检索增强生成（RAG）、文本嵌入（text‑embedding‑ada‑002）、Milvus 向量数据库、PaddleOCR、FastAPI + Gradio 前端、Docker、Nginx 反向代理、AWS EC2 + GPU、Google Drive API 以及内部的 Galileo AI 平台。

**📊 数据集**

数据集来源于 Roche 合规团队的 Google Drive 共享驱动快照，包含 600+ 份 PDF/Word/Excel 等文档；生成的 QA 数据集为 967 对 Q&A（139 份文档）用于评估；训练 ReLACE 的重排序数据集约 4.3k 个 QA‑passage 对，覆盖 568 个监管问题。

**📈 对比分析**

通过在不同 K（3、5、10、15）下对 RCS、HiSACC、RCS+ReLACE、HiSACC+ReLACE 四种配置进行对比实验。结果显示 HiSACC+ReLACE 在大多数指标上领先：K=15 时 AR≈0.873，CR≈0.879，GR≈0.845，FT≈0.925，ORP≈0.003，说明答案更相关、上下文更贴合、真值覆盖率更高。

**⚠️ 局限性**

局限性包括：仅在制药法规域内验证；对 GPT‑4 等大型模型的调用成本和可扩展性仍有挑战；模型仍可能出现细微幻觉，需要进一步的 RLHF 或插件化数据接口来持续改进；并未对跨行业通用性做充分评估。

---

## 414. DIETA: A Decoder-only transformer-based model for Italian-English machine TrAnslation

**arXiv ID:** 2601.17823 | [PDF](https://arxiv.org/pdf/2601.17823v1)

**作者:** Pranav Kasela `[一作]` (University of Milano-Bicocca), Alessandro Raganato `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 2389 | [OpenAlex ID](https://openalex.org/A5053806445)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个0.5B参数的decoder-only Transformer模型DIETA，用于意大利语-英语机器翻译；

**💡 创新点**

创新点在于：①通过收集207M高质量平行句对并生成352M的后向翻译数据，构建了约768M规模的专用语料；②在相同模型规模下利用后向翻译及继续训练显著提升性能；③发布了新的WikiNews‑25评测集；

**🔧 技术方法**

技术包括：decoder-only Transformer架构（6层、2048隐藏、32头，rotary embedding、squared‑ReLU等），SentencePiece 51,200词表，Lion优化器，单轮训练与多轮继续训练；

**📊 数据集**

数据集涵盖Europarl、DGT‑TM、ParaCrawl、OpenSubtitles、WikiMatrix、Books、NewsCrawl、FineWeb等来源的207M平行对，以及352M人工后向翻译生成的合成对；

**📈 对比分析**

在NTREX‑128、Tatoeba、WMT‑24pp、Flores‑200、WikiNews‑25等五个基准上与32个对照系统对比，DIETA+cont/DIETA+allsynth位于第二四分位，能够匹配1‑3B规模模型并在大部分指标上仅落后9B基线数分之几；

**⚠️ 局限性**

局限性主要在于：与最强大型模型相比在无参考质量估计（QE）指标上仍有差距，模型仍需在QE‑aware训练、参数扩展（如MoE）和模型压缩（蒸馏/量化）等方向进一步提升。

---

## 415. Less Is More: Scalable Visual Navigation from Limited Data

**arXiv ID:** 2601.17815 | [PDF](https://arxiv.org/pdf/2601.17815v1)

**作者:** Yves Inglin `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**通讯引用:** 19610 | [OpenAlex ID](https://openalex.org/A5044258783)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种端到端的、单帧RGB图像输入、目标姿态条件下的视觉导航策略——Less Is More（LIM），通过将少量真实专家演示与使用MPPI规划器生成的几何轨迹相结合，构建了大规模、跨场景、具身特定的训练数据集，并在真实机器人上实现了闭环部署。

**💡 创新点**

创新点主要有：① 将经典几何规划器的轨迹作为监督标签，显著扩展有限演示数据的规模与多样性；② 引入基于DINOv2特征与Transformer解码器的目标导向轨迹预测框架；③ 通过在训练时加入不可达目标来提升对不可行目标的鲁棒性；④ 在真实四足机器人上展示了无需地图或额外传感器的闭环视觉导航。

**🔧 技术方法**

技术包括：视觉Transformer（使用冻结的DINOv2编码器+4层解码器）预测N个SE(2)路径点；MPPI优化求解几何轨迹；基于高精度高程图的可通行度估计与地理距离场；数据增强（随机采样目标、生成多样轨迹）。

**📊 数据集**

主要使用了GrandTour数据集（ANYmal D机器人6小时高度多样化部署），从中提取了三类数据：① 真实遥控演示（Teleop）；② 通过MPPI规划生成的10条轨迹/图像（Geometric）；③ 以上两类合并得到的Augmented Dataset（约2162小时）。

**📈 对比分析**

与基线（直线、MPPI、OmniVLA开放循环、真实遥控路径）在保留的六条Held‑out GrandTour任务上评估。LIM在多样化测试集上SPL提升至约88.8%（相较于直线87.5%），成功率接近99%。在目标分布相似的基准集上，SPL仅略有提升，说明几何增强在多障碍环境中更为显著。相对OmniVLA，LIM在开放循环下的SPL更高，同时部署时可直接在机器人上实时运行。

**⚠️ 局限性**

主要局限包括：① 依赖高质量高程图与可通行度估计，若估计失真会导致标签不安全；② 仅使用单帧RGB，无时间记忆，难以处理遮挡或离开视野的障碍；③ 训练数据仍受原始GrandTour视觉多样性的限制，未能覆盖极端或极少见场景。

---

## 416. Delay-Compensated Stiffness Estimation for Robot-Mediated Dyadic Interaction

**arXiv ID:** 2601.17812 | [PDF](https://arxiv.org/pdf/2601.17812v1)

**作者:** Mingtian Du `[一作]` (Nanyang Technological University), Domenico Campolo `[通讯]` (Nanyang Technological University)

**通讯引用:** 2910 | [OpenAlex ID](https://openalex.org/A5079258091)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在机器人介导的双人（师徒）交互中，对被治疗者肌肉张力（刚度）进行估计时所面临的网络时延误差问题，并提出了一种新的时延补偿刚度估计框架。

**💡 创新点**

创新点在于：①基于准静态平衡推导出一次性闭环刚度估计式，能够显式对称延迟进行补偿；②引入归一化加权最小二乘（NWLS）方法，消除由时延导致的误差偏倚；③在实验中系统性比较了传统无时延估计、OLS 与 NWLS 的性能，证明 NWLS 在多种时延下都能保持较低的估计误差。

**🔧 技术方法**

主要技术包括：准静态平衡推导、最小二乘回归、加权最小二乘（OLS）与归一化加权最小二乘（NWLS），以及在 H‑MAN 机器人平台上实现的双向力/位置信号同步与延迟补偿。

**📊 数据集**

实验数据集：在两台 H‑MAN 平面康复机器人上进行 160 次试验，实验条件覆盖 4 种网络延迟（0、80、160、320 ms）、2 种被治疗者刚度（60、120 N/m）以及 X/Y 两个工作轴，重复 10 次每组。

**📈 对比分析**

通过对比 Naive、OLS、NWLS 三种估计方法，使用绝对百分误差（APE）进行统计评估。结果显示：Naive 在 80 ms 以上时明显下估，误差随时延增大；OLS 在 320 ms 时表现优于 Naive，但在无时延或低刚度情况下易受高位移点偏倚影响；NWLS 在所有时延条件下均保持低误差，并与 Naive 在 0 ms 时差别不显著，证明其在时延补偿上的稳健性。

**⚠️ 局限性**

局限性包括：仅考虑固定时延，未处理网络抖动与丢包；实验使用机器人-机器人设置，未验证在人类被治疗者的非线性、时变阻抗；NWLS 目前是批处理实现，尚未转化为递归实时估计。

---

## 417. Unveiling hidden features of social evolution by inferring Langevin dynamics from data

**arXiv ID:** 2601.17772 | [PDF](https://arxiv.org/pdf/2601.17772v1)

**作者:** Youngkyoung Bae `[一作]` (Seoul National University), David Wolpert `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出将历史社会演化建模为连续时间随机微分方程（SDE）框架，利用其捕捉结构趋势与不可预测波动，并通过不可逆性、外生冲击检测和缺失数据插补等诊断工具来分析历史轨迹。

**💡 创新点**

创新点在于将不可逆性分析、尾概率外生冲击检测以及概率插补方法嵌入SDE框架；同时结合两种推断技术（Langevin Bayesian Networks 和非参数高斯过程SDE）来应对不同数据稀疏度与采样不规则性。

**🔧 技术方法**

使用的技术包括：SDE建模、Euler–Maruyama近似、信息熵生产（不可逆性）、尾概率/惊讶度量、Langevin Bayesian Networks、非参数高斯过程SDE 推断、重要性采样/顺序蒙特卡洛插补。

**📊 数据集**

使用的数据集包括：现代政治经济数据（V‑Dem、WID、MPD）和Seshat 全球历史数据库的 Polaris 数据集（规模与计算力指标），分别用于两大案例研究。

**📈 对比分析**

与传统静态回归或脉冲响应等方法比较，SDE框架在模拟自相关、残差白噪声检验、不可逆性与尾概率与历史重大事件的对应性等方面表现更好；能够更精确地识别异常冲击、量化不确定性并完成缺失数据的多重插补。

**⚠️ 局限性**

限制包括：历史数据稀疏、时间间隔长导致漂移与扩散辨识困难；隐式假设同质性和独立性；状态空间设计需依赖理论选择，可能引入主观性；推断结果主要是关联性描述，无法直接推断因果机制。

---

## 418. LLM-42: Enabling Determinism in LLM Inference with Verified Speculation

**arXiv ID:** 2601.17768 | [PDF](https://arxiv.org/pdf/2601.17768v1)

**作者:** Raja Gond `[一作]` (Microsoft Research), Ashish Panwar `[通讯]` (Microsoft Research)

**通讯引用:** 336 | [OpenAlex ID](https://openalex.org/A5060296906)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于调度的推理框架（LLaMA‑42），通过快速解码-验证-回滚（DVR）机制，在动态批处理环境下实现大模型推理的可确定性；

**💡 创新点**

创新点在于：①将推理的确定性与普通快速解码解耦，只在需要时进行验证，避免全局采用慢速的 batch‑invariant kernel；②利用 GPU kernel 的形状一致性，构造可重现的验证窗口；③引入分组验证（grouped verification）在小窗口与多请求之间折中，提升利用率与降低回滚成本；

**🔧 技术方法**

技术实现包括：基于 FlashAttention‑3 的位置不变注意力；固定形状的 split‑K 约束；多轮预填充与解码的分离；Gumbel‑noise 采样器替代传统随机采样；多线程/多GPU的在线推理框架；

**📊 数据集**

主要使用公开数据集 ShareGPT 与 ArXiv，分别包含约 92k 与 6k 条请求，长度分布与真实工作负载相符；

**📈 对比分析**

与传统 deterministic（batch‑invariant）以及非确定性 baseline 进行比较；在 1–4 GPU 上的离线吞吐量与在线延迟实验显示：在大部分场景下，LLaMA‑42 在 0–10% deterministic traffic 时，吞吐量仅损失 1–4%，且在线延迟比 batch‑invariant baseline 低 10–30%；在 100% deterministic 时，仍比 batch‑invariant 低 6% 的吞吐量；

**⚠️ 局限性**

局限性包括：①验证阶段对所有请求（即使非 deterministic）会产生固定延迟；②预填与解码使用不同的归约策略，导致无法共享前缀 KV 缓存；③未集成与现有的基于草稿模型的 speculative decoding；④多 GPU 互连有限，未完全评估跨卡性能；

---

## 419. An MLIR Lowering Pipeline for Stencils at Wafer-Scale

**arXiv ID:** 2601.17754 | [PDF](https://arxiv.org/pdf/2601.17754v1)

**作者:** Nicolai Stawinoga `[一作]`, Tobias Grosser `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一条基于MLIR的编译流水线，自动将Stencil代码转换为Cerebras WSE可执行的CSL代码，省去手工改写。

**💡 创新点**

首次在WSE上实现无代码改动的Stencil自动化编译，并通过自定义Dialect桥接异步Actor模型与高层数学描述，显著提升了编译器可重用性。

**🔧 技术方法**

采用xDSL+MLIR框架，设计了csl-stencil、csl-wrapper、csl-ir等Dialect，并实现多阶段低层化、通信拆分、DSD优化等技术。

**📊 数据集**

使用了5个Benchmark（Jacobain、Diffusion、Acoustic、25‑Point Seismic、UVKBE），分别来自Fortran、Python（Devito）、CSL，尺寸分为small/medium/large。

**📈 对比分析**

在WSE2/WSE3上与手写CSL、128 A100 GPU和128 CPU节点进行对比，WSE3在大规模问题上平均提升14×（GPU）/20×（CPU），且与手写代码相比性能提升约8%。

**⚠️ 局限性**

局限在于仍需手工维护自定义Dialect，暂不支持所有Stencil形状（如box‑shape），对不同前端的集成仍有一定技术门槛。

---

## 420. Predicting Juror Predisposition Using Machine Learning: A Comparative Study of Human and Algorithmic Jury Selection

**arXiv ID:** 2601.17745 | [PDF](https://arxiv.org/pdf/2601.17745v1)

**作者:** Ashwin Murthy `[一作]` (Amazon), Ranjita Naik `[通讯]` (Georgia Institute of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文比较了专业陪审团顾问与机器学习模型在预测陪审团成员投票倾向上的效果。

**💡 创新点**

创新点在于首次在同一实验设置下，将人类专家判断与监督式机器学习模型进行严格的对标评估，并公开数据与代码。

**🔧 技术方法**

使用的技术包括随机森林（Random Forest）和k近邻（k‑Nearest Neighbors）两种监督学习模型。

**📊 数据集**

数据集由410名在线招募的模拟陪审团成员填写的问卷组成，共29个特征，分为训练集273例和测试集137例。

**📈 对比分析**

通过将模型预测与顾问多数投票在同一测试集上进行配对比较，随机森林准确率0.818、k‑NN 0.796，均显著优于顾问0.693，差异均置信区间不包含0，且McNemar检验表明错误模式有统计显著差异。

**⚠️ 局限性**

限制包括使用模拟陪审团且仅针对单一失业终止案例，未包含更丰富的多模态信息，亦未进行公平性或下游法律影响评估。

---

## 421. ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents

**arXiv ID:** 2601.17735 | [PDF](https://arxiv.org/pdf/2601.17735v1)

**作者:** Kyungho Kim `[一作]` (KAIST), Kijung Shin `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向关系型数据库预测任务的特征生成框架ReFuGe，利用LLM代理自动识别相关表列、生成候选特征并通过推理与验证筛选，迭代提升性能

**💡 创新点**

创新点在于将特征生成任务拆分为三个专门的LLM代理（schema selection、feature generation、feature filtering），并引入反馈驱动的迭代循环，实现无监督的自我改进

**🔧 技术方法**

核心技术包括多代理LLM（具备推理与生成能力）、多实例生成提升多样性、两阶段特征筛选（推理+验证）以及自然语言反馈机制

**📊 数据集**

实验使用七个公开的关系型数据库基准数据集，涵盖多领域与不同复杂度的schema

**📈 对比分析**

与单表学习、LLM直接预测/生成等多种基线对比，ReFuGe在11项任务中超过9项，平均性能最高、平均排名第一，验证了方法的有效性

**⚠️ 局限性**

局限性包括：需要多次LLM调用导致计算开销大；对高度复杂或极大规模数据库的扩展性未充分评估；目前仅验证分类任务，需进一步验证回归或链接预测等场景

---

## 422. CondenseGraph: Communication-Efficient Distributed GNN Training via On-the-Fly Graph Condensation

**arXiv ID:** 2601.17774 | [PDF](https://arxiv.org/pdf/2601.17774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 423. Aligning Medical Conversational AI through Online Reinforcement Learning with Information-Theoretic Rewards

**arXiv ID:** 2601.17828 | [PDF](https://arxiv.org/pdf/2601.17828v1)

**作者:** Tanvi Verma `[一作]` (Institute of High Performance Computing, Agency for Science, Technology and Research), Yong Liu `[通讯]` (Institute of High Performance Computing, Agency for Science, Technology and Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用在线强化学习和信息增益奖励，训练医学会话模型进行患者访谈并生成完整的病史（HPI），无需人工标注对话。

**💡 创新点**

创新点在于将病史采集建模为信息获取任务，设计基于熵减少的奖励函数，并结合 GPT‑4o‑mini 的质量评估实现安全、有效的提问策略；使用 GRPO 进行语言模型的稳定在线微调。

**🔧 技术方法**

技术包括：在线强化学习、Group Relative Policy Optimization (GRPO)、信息增益奖励（熵计算与实体覆盖追踪）、LLM（GPT‑4o‑mini）质量评估、LoRA 参数高效微调。

**📊 数据集**

使用 Avey AI Benchmark 的 350 条简洁病例作为训练集，评估数据为 Avey held‑out 48 条和 MIMIC‑IV 数据集的 50 条较长病例。

**📈 对比分析**

与基线模型（未微调的 Llama‑3.1‑8B、DeepSeek‑R1‑Distill‑Qwen‑7B、GPT‑4o‑mini、HuatuoGPT‑o1‑7B、UltraMedical‑8B）比较，IGFT 在 Avey 上 F1 提升 10.9%（DeepSeek）/8.4%（Llama），在 MIMIC 上提升 12.9%/8.6%，并超过 GPT‑4o‑mini 的 F1 分数。

**⚠️ 局限性**

局限性包括：高昂的 LLM 计算成本、需要预先提取实体（非端到端）、每轮仅独立优化缺乏全局规划、模型在更复杂病历上仍有提升空间、需进一步验证公平性与临床安全。

---

## 424. Beyond Symbols: Motion Perception Cues Enhance Dual-Task Performance with Wearable Directional Guidance

**arXiv ID:** 2601.17799 | [PDF](https://arxiv.org/pdf/2601.17799v1)

**作者:** Qing Zhang `[一作]` (University of Tokyo), Jun Rekimoto `[通讯]` (University of Tokyo)

**通讯引用:** 11548 | [OpenAlex ID](https://openalex.org/A5082649952)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过在可穿戴光学见透头戴式显示器的单眼侧视屏幕上呈现单色移动条纹，直接触发外周运动感知，以提供方向提示，并验证其在双任务环境下的有效性。

**💡 创新点**

创新点在于：①不使用符号或文字，直接激活视觉运动选择性神经通路；②采用单眼呈现减少视线切换，降低对中央视觉的干扰；③通过对比箭头符号和运动提示，证明了更低的认知负荷和更高的方向感知准确率。

**🔧 技术方法**

技术包括：可穿戴单色透明LCD面板、ESP32-C3微控制器、低对比度高频移动条纹刺激、单眼显示与中心透明孔设计、双任务实验框架。

**📊 数据集**

使用了两组用户研究实验：实验一（14人）在受控实验室中评估不同对比度下的方向识别准确度；实验二（10人）在双任务设置（篮球传球计数 + 方向提示）中比较运动提示与箭头符号的性能。

**📈 对比分析**

方法：在双任务实验中对比两种提示方式，使用绝对百分误差 (APE) 评估主任务（计数）与辅任务（方向响应）的表现。结果显示：运动提示在辅任务上的 APE 下降到 6.25%（p=0.008），箭头符号为 20%；主任务 APE 在运动提示下为 8.8%（中位 8%），箭头符号为 20.8%（中位 16%），差异未显著但表现更好。

**⚠️ 局限性**

局限性包括：样本量有限、实验环境相对受控、未考察不同用户的运动敏感性差异、偶尔出现虚假运动感知、缺乏真实世界情境验证、未探讨长时使用的适应效应。

---

## 425. Beyond a Single Perspective: Text Anomaly Detection with Multi-View Language Representations

**arXiv ID:** 2601.17786 | [PDF](https://arxiv.org/pdf/2601.17786v1)

**作者:** Yixin Liu `[一作]` (Griffith University), Shirui Pan `[通讯]` (Griffith University)

**通讯引用:** 22491 | [OpenAlex ID](https://openalex.org/A5008056593)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种多视图文本异常检测框架，利用多种预训练语言模型生成的嵌入来识别文本异常。

**💡 创新点**

创新点在于通过对多视图嵌入进行对比协作以增强跨视图一致性，并设计自适应分配模块根据样本特征动态调节各视图权重。

**🔧 技术方法**

主要技术包括基于自编码器的多视图重建、InfoNCE 对比学习协作模块、PCA 对齐+MLP 自适应权重分配以及两阶段训练策略。

**📊 数据集**

评估使用了来自 NLP-ADBench 与 TAD-Bench 的 10 个公开基准数据集。

**📈 对比分析**

在与三类基线（端到端、嵌入+检测器、多视图方法）对比后，实验显示在 9/10 个数据集上均取得最高 AUROC，证明了其优越的性能。

**⚠️ 局限性**

限制在于需要调用多套预训练模型，导致推理成本和延迟较高，且在资源受限环境下扩展性有限。

---

## 426. HyCARD-Net: A Synergistic Hybrid Intelligence Framework for Cardiovascular Disease Diagnosis

**arXiv ID:** 2601.17767 | [PDF](https://arxiv.org/pdf/2601.17767v1)

**作者:** Rajan Das Gupta `[一作]` (American International University–Bangladesh), Jiaqi He `[通讯]` (Tilburg University)

**通讯引用:** 1168 | [OpenAlex ID](https://openalex.org/A5083870128)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发了一种将CNN、LSTM深度学习模型与KNN、XGB传统机器学习模型通过加权投票集成的心血管疾病预测框架。

**💡 创新点**

创新点在于将深度特征提取与传统可解释模型融合，并通过权重投票实现高准确率与可解释性，同时在两个公开数据集上进行跨数据集验证，展示了模型的泛化能力。

**🔧 技术方法**

采用CNN、LSTM、KNN、XGB、加权多数投票集成、SMOTE过采样、特征归一化、10折交叉验证以及Python/Colab GPU加速等技术。

**📊 数据集**

使用了Kaggle公开的两个心脏疾病数据集：Dataset I（约70,000条样本）和Dataset II（约918条样本）。

**📈 对比分析**

通过10折交叉验证与单一ML、单一DL、组合DL模型等进行对比，实验显示模型在Dataset I上准确率为82.30%，在Dataset II上达到97.10%，均优于所有基线方法。

**⚠️ 局限性**

局限性包括：尚未在真实临床大规模数据上验证；数据来源相对有限，模型对不同人群或设备的适应性未知；过采样方法可能导致过拟合；缺乏可解释AI工具（如SHAP、LIME）来进一步提升临床可接受度。

---

## 427. Multi-Agent End-to-End Vulnerability Management for Mitigating Recurring Vulnerabilities

**arXiv ID:** 2601.17762 | [PDF](https://arxiv.org/pdf/2601.17762v1)

**作者:** Zelong Zheng `[一作]` (Zhejiang University), Shengyi Pan `[通讯]` (Zhejiang University)

**通讯引用:** 1716 | [OpenAlex ID](https://openalex.org/A5005384561)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于多代理（multi‑agent）的端到端递归漏洞管理框架（MAVM），能够从漏洞检测、确认、修复到验证全流程自动化处理 C 语言代码中的重复漏洞。

**💡 创新点**

创新点：①构建漏洞知识库（VKB）将公开 CVE 的根因、触发链和补丁信息系统化；②设计上下文检索工具，使代理能够在仓库级别获取函数调用链、数据结构等上下文；③将检测、确认、修复、验证四个传统手工流程拆分成独立代理，模拟真实安全工作流并实现无人工干预的闭环。

**🔧 技术方法**

技术：GPT‑4o 作为核心 LLM；多代理协作框架（Langgraph）；静态检测工具 ReDeBug 与哈希匹配；自定义检索工具（ripgrep、Tree‑sitter）获取代码上下文；一致性检查与补丁生成逻辑；回归验证环节。

**📊 数据集**

数据集：基于 Mystique 与 PPatHF 公开数据构建，筛选非同构迁移案例，最终获得 78 个 CVE（114 个函数级迁移对）仅包含 C 语言代码。

**📈 对比分析**

与基线组合（ReDeBug+Hash+FVF、GPT‑4o、PPatHF、Mystique 等）进行比较。检测阶段，MAVM 在精确率 76.4%、召回率 59.6%、F1 67.0%；修复阶段，成功修复 51/68 个漏洞，修复准确率 57.3%，比最优基线提升 31.9%–45.2%。

**⚠️ 局限性**

局限性：①仅在 C 语言上验证，缺乏多语言通用性；②数据集规模相对较小，未覆盖全部真实场景；③依赖 GPT‑4o，受 API 调用成本与输出随机性限制；④当前仅支持函数级迁移，无法直接处理更高层级（如 CVE 级）问题。

---

## 428. ProGraph-R1: Progress-aware Reinforcement Learning for Graph Retrieval Augmented Generation

**arXiv ID:** 2601.17755 | [PDF](https://arxiv.org/pdf/2601.17755v1)

**作者:** Jinyoung Park `[一作]` (Korea Advanced Institute of Science and Technology), Joo-Kyung Kim `[通讯]` (Amazon)

**通讯引用:** 92 | [OpenAlex ID](https://openalex.org/A5004011764)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ProGraph‑R1，一种进度感知的图检索与多步推理强化学习框架；

**💡 创新点**

创新点在于同时结合语义相似度与图结构相似度的超图检索机制，以及基于推理进度的逐步奖励与优势调整；

**🔧 技术方法**

采用LLM（如3B/7B模型）+图检索技术+强化学习（GRPO）+超图结构分析；

**📊 数据集**

在多跳问答基准（2WikiMultihopQA、HotPotQA、MuSiQue）和单跳NQ数据集上进行评估；

**📈 对比分析**

与NaiveGeneration、StandardRAG、SFT、R1、Search‑R1、R1‑Searcher、Graph‑R1等基线相比，ProGraph‑R1在多跳问答中平均F1提升约5点，单跳问答亦有显著提升；

**⚠️ 局限性**

受限于提示语、检索器、底层LLM及输入图结构的质量，模型性能可能显著波动；

---

## 429. Bridging Supervision Gaps: A Unified Framework for Remote Sensing Change Detection

**arXiv ID:** 2601.17747 | [PDF](https://arxiv.org/pdf/2601.17747v1)

**作者:** Kaixuan Jiang `[一作]`, Chengxi Han `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

这项研究提出了一个统一的遥感图像变化检测框架UniCD，能够在监督、弱监督和无监督三种设置下实现高精度变化提取。

**💡 创新点**

创新点在于引入轻量化的时空注意模块STAM、基于CAM的弱监督策略以及伪标签+对比学习的无监督优化，实现了跨场景一致性和错误抑制。

**🔧 技术方法**

该方法结合卷积编码器、STAM、解码器、CAM、FastSAM、CLIP对比损失以及伪标签生成等技术，形成端到端的统一模型。

**📊 数据集**

主要使用了三大公开数据集：LEVIR-CD、WHU-CD与CLCD，覆盖城市建筑、灾后重建和农田变化等多样场景。

**📈 对比分析**

在三大数据集的监督、弱监督和无监督实验中，UniCD相较于SNUNet、ChangeFormer、SAM-CD、WCDNet、ISFA等传统方法，表现出更好的F1、IoU和OA，尤其在边界完整性与假阳性抑制方面显著优于现有方案。

**⚠️ 局限性**

限制方面是对极端光照或云遮挡等极端成像条件的鲁棒性尚未充分验证，且在大尺度全景图像上推理速度与显存需求仍需进一步优化。

---

## 430. Faramesh: A Protocol-Agnostic Execution Control Plane for Autonomous Agent Systems

**arXiv ID:** 2601.17744 | [PDF](https://arxiv.org/pdf/2601.17744v1)

**作者:** Amjad Fatmi `[一作]` `[通讯]` (Faramesh Labs), Amjad Fatmi (Faramesh Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了执行时授权边界（Action Authorization Boundary, AAB）以及规范化动作表示（Canonical Action Representation, CAR），以在自治代理系统中强制执行不可绕过、确定性、可重放的执行授权。

**💡 创新点**

创新点在于：
• 引入了“执行时授权边界”作为独立的、协议无关的治理层；
• 设计了动作规范化流程，消除语义等价动作的表示差异；
• 通过可加链的决策日志实现不可篡改、可重放的审计记录；
• 明确界定失败时闭锁（fail‑closed）与多租户、多代理支持，保证系统可伸缩性。

**🔧 技术方法**

技术手段包括：
• 语义级动作规范化（Canonicalization）与哈希化；
• 纯粹的决策函数 Eval(A,P,S)；
• 基于哈希链的追加式决策日志；
• 并发去重与一次性执行保障；
• 对外提供的决策工件（artifact）与执行校验；

**📊 数据集**

数据集与实验环境：
• 合成工作负载：生成数千至数十万条动作；
• 策略集 |𝒫|∈{64,256,1024}；
• 状态摘要 |S|∈{4 KB, 64 KB, 512 KB}；
• 执行器批量大小 b∈{1,8,32}；
• 单机微基准运行于 Apple M1（8核）设备。

**📈 对比分析**

比较与性能：
• 延迟：单次决策平均 p95 < 10 ms，整体决策链 p95 < 10 ms；
• 吞吐：单线程可达 7.8k 次/分；
• 确定性：10 k 次评估始终产生同一哈希；
• 失效安全：timeout/kill‑switch 时返回 DENY；
• 旁路攻击覆盖率 ≈ 0.999；
• 并发去重在 1M 次重复请求下无双重执行。

**⚠️ 局限性**

限制与未覆盖的范畴：
• 未处理代理推理的正确性、语义解释与认知层面；
• 需要可信计算基（canonicalizer、决策引擎、日志）才能保证完整性；
• 仅为执行授权边界，无法替代传统 IAM、监控或治理平台的其它功能；
• 对执行器内部的错误或被破坏不提供保护；
• 需要手动配置或外部工具实现多租户隔离与策略管理。

---

## 431. Frequency-aware Neural Representation for Videos

**arXiv ID:** 2601.17741 | [PDF](https://arxiv.org/pdf/2601.17741v1)

**作者:** Jun Zhu `[一作]` (University of Chinese Academy of Sciences), Jia Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 28772 | [OpenAlex ID](https://openalex.org/A5118788614)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 FaNeRV，一种基于隐式神经表示的频率感知视频压缩框架

**💡 创新点**

创新点在于通过多分辨率监督将低高频分离、动态高频注入机制以及频率感知层来克服传统 INR 的频谱偏置

**🔧 技术方法**

采用隐式神经表示、层级多分辨率监督、动态高频注入、Hybrid Upsample、Falayer、尺度自适应损失、量化感知训练和熵编码等技术

**📊 数据集**

在 HEVC ClassB、UVG 以及 VTM（RA）基准视频集上进行实验

**📈 对比分析**

与 VTM、HM、DCVC、HiNeRV、HNeRV‑Boost 等方法对比，FaNeRV 在 PSNR/​MS‑SSIM 及 BD‑Rate 上均优于传统混合编解码器和现有 INR 方法，且实现了可扩展编码

**⚠️ 局限性**

缺点是相较于部分 INR 方法，编码时间略长；对极高动态场景的细节仍有提升空间

---

## 432. Flatten The Complex: Joint B-Rep Generation via Compositional $k$-Cell Particles

**arXiv ID:** 2601.17733 | [PDF](https://arxiv.org/pdf/2601.17733v1)

**作者:** Junran Lu `[一作]` (Nanjing University), Yanwen Guo `[通讯]` (Nanjing University)

**通讯引用:** 4284 | [OpenAlex ID](https://openalex.org/A5009275869)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种将B‑Rep结构转换为无序k‑cell粒子集合的表示方法，并实现了其联合拓扑与几何的无条件与条件生成。

**💡 创新点**

创新点在于通过粒子共享机制将高阶单元的几何信息复用，打破层次化生成瓶颈，并利用流匹配实现全局并行生成。

**🔧 技术方法**

采用了基于变分自编码器的CC‑VAE进行粒子空间学习，再配合Rectified Flow Transformer实现概率建模和多模态条件生成。

**📊 数据集**

使用了ABC、DeepCAD、Furniture、Roof等工业级CAD数据集进行训练与评估。

**📈 对比分析**

与BRepGen、DTG‑BRepGen等方法对比，本文在有效率、复杂度覆盖率和结构完整性等指标上均取得了更高的分数，生成模型更为多样且结构更完整。

**⚠️ 局限性**

局限性包括仍存在几何失真、粒子维度大导致计算开销增加，以及对严格几何约束的支持仍不充分。

---

## 433. Implicit Neural Representation-Based Continuous Single Image Super Resolution: An Empirical Study

**arXiv ID:** 2601.17723 | [PDF](https://arxiv.org/pdf/2601.17723v1)

**作者:** Tayyab Nasir `[一作]` (University of Western Australia), Ajmal Mian `[通讯]` (University of Western Australia)

**通讯引用:** 19770 | [OpenAlex ID](https://openalex.org/A5089986388)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对基于隐式神经表示（INR）的任意尺度图像超分辨率（ASSR）方法进行系统的经验性分析，构建统一实验框架，统一训练与评估设置，比较多种模型与训练策略，并提出混合像素‑梯度损失以提升纹理保真度。

**💡 创新点**

创新点包括：1) 统一多指标、多训练配置的实验框架和代码仓库；2) 采用Borda计分法进行跨模型、跨数据集、跨尺度、跨评价指标的全局排名；3) 提出混合像素‑梯度损失，显著提升纹理和边缘细节；4) 证明INR‑ASSR的缩放规律在不同模型、数据量、训练时长下成立。

**🔧 技术方法**

技术手段：隐式神经表示（MLP解码器）、多种Encoder（EDSR、RDN）、多尺度训练（1–4/1–6）、多步学习率/SGDR调度、不同损失函数（L1、Gram‑L1、Hybrid‑Gradient）、多种图像质量评估指标（PSNR、SSIM、GMSD、FSIM、VIF、SR‑SIM、LPIPS），以及Borda计分的综合排名方法。

**📊 数据集**

数据集：训练使用DIV2K（800张2K图像）；测试使用Set5、Set14、BSD100、Urban100、SVT、CelebA‑HQ等多个通用与领域特定数据集，形成多样化评测集合。

**📈 对比分析**

比较方法：在统一的6种训练配置（Patch大小、LR调度、尺度范围、损失函数等）下，对6个现有INR‑ASSR模型进行72次实验，使用7种IQA指标评估，并通过Borda计分得到全局排名。实验结果表明：① 最新模型改进有限；② 训练配置对性能影响大；③ 采用混合损失显著提升纹理与边缘细节；④ 模型架构与配置高度相关；⑤ 缩放规律成立。

**⚠️ 局限性**

局限性：1) 受限于现有基准数据集，模型已趋于饱和；2) 仅训练150 epoch，未探索更长训练或更大模型；3) 排除CLIT等更复杂模型；4) 缺乏更具挑战性的多样化数据集与高级感知指标；5) 真实应用场景下的泛化与鲁棒性验证不足。

---

## 434. Linguistic and Argument Diversity in Synthetic Data for Function-Calling Agents

**arXiv ID:** 2601.17829 | [PDF](https://arxiv.org/pdf/2601.17829v1)

**作者:** Dan Greenstein `[一作]` (Technion), Oren Somekh `[通讯]` (TII)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于通用多样性度量的生成方法，自动生成既具语言多样性又具参数多样性的单轮函数调用数据集，并在该数据集上训练并评估模型。

**💡 创新点**

创新点包括：①使用无规则的通用多样性优化器（通过贪婪增量式边际贡献计算）同时提升语言与参数两维度多样性；②将多种多样性度量（词汇、句法、语义、集群熵等）融入候选生成与筛选流程；③在不扩大训练规模的前提下实现 OOD 性能显著提升。

**🔧 技术方法**

技术手段包括：大语言模型生成候选与指令；Embedding（MiniLM）+聚类识别语义相似参数；多样性度量（TTR、树编辑距离、语义距离、集群熵等）计算边际贡献；RRF 级联排序；过滤与 LLM‑judge 判定；LoRA 微调训练 LLM。

**📊 数据集**

数据集：自建约1,793条样本；与公开的 ToolAce（约20K）和 APIGen（约60K）进行对比；在 BFCL benchmark（非实时 Python 函数调用子集约1,240条）上做 OOD 评估。

**📈 对比分析**

评估方法：使用多维度多样性指标与人工 correctness 样本检测，证明自身在词汇、句法、语义、参数多样性均优于基线；在 fine‑tune 任务中，模型在 APIGen、ToolAce、BFCL 三个 OOD 数据集上的准确率相较基线提升约 3–7%（如 4.2%/3.8% 等），并通过 McNemar / Holm‑Bonferroni 校正显著性检验。

**⚠️ 局限性**

限制：仅针对英文；只处理单轮交互；依赖强大 LLM 做过滤导致生成成本高；未验证多轮交互或低资源语言的适用性。

---

## 435. UniPACT: A Multimodal Framework for Prognostic Question Answering on Raw ECG and Structured EHR

**arXiv ID:** 2601.17916 | [PDF](https://arxiv.org/pdf/2601.17916v1)

**作者:** Jialu Tang `[一作]`, Aaqib Saeed `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 UniPACT 框架，实现对原始 12 先导 ECG 波形与结构化 EHR 数据的统一表述与融合，支持以自然语言提问的多任务临床预后推理；

**💡 创新点**

通过将 EHR 数值转换为语义丰富的提示词以及设计多模态投影器，将 ECG 特征映射至 LLM 嵌入空间，首次实现 LLM 对原始生理信号的直接推理；

**🔧 技术方法**

利用预训练 Transformer‑based ECG 编码器、结构化提示生成、MM‑Projector 投影、LLM（MedGemma‑4B）生成式推理以及多任务学习与 LoRA 微调；

**📊 数据集**

使用 MDS‑ED 基准（MIMIC‑IV‑ECG 与 MIMIC‑IV 结合的 1443 个子任务，包含诊断、恶化、ICU 入院、死亡等多任务）；

**📈 对比分析**

在 MDS‑ED 上与 ECG‑Chat、Q‑HEART、MDS‑ED 等多模态基线对比，UniPACT 的整体 AUROC 达 89.37%，比 MDS‑ED 提升约 0.5%，在多任务上表现更稳健；

**⚠️ 局限性**

对缺失模态的鲁棒性虽好，但在去除关键生理参数（如生命体征）时仍有显著性能下降，且模型对真实临床数据分布变化的适应性未作深入评估。

---

## 436. Prompt-Based REST API Test Amplification in Industry: An Experience Report

**arXiv ID:** 2601.17903 | [PDF](https://arxiv.org/pdf/2601.17903v1)

**作者:** Tolgahan Bardakci `[一作]` (University of Antwerp and Flanders Make), Serge Demeyr `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在物流公司的生产微服务上复现并验证了基于LLM的REST API测试放大方法，提升了结构化API覆盖率并揭示了异常；

**💡 创新点**

创新点在于将先前在开源环境中的方法迁移至真实工业环境，揭示了行业约束下的效果、挑战及最佳实践；

**🔧 技术方法**

使用ChatGPT 4o生成Gherkin与C#步骤、RESTCov覆盖测量工具以及OpenAPI文档作为输入；

**📊 数据集**

使用该微服务的六个代表性端点（共约1千行测试代码）作为实验数据；

**📈 对比分析**

通过对比放大前后的覆盖率（如路径、操作、状态码等）发现平均提升约25‑70%，但对比方法与基准一致，未检测错误发现率；

**⚠️ 局限性**

主要局限在于需要人工后处理、对状态与身份验证假设不稳定、规格规模导致噪声、工具适配成本以及仅单一服务的可推广性不足。

---

## 437. dLLM-ASR: A Faster Diffusion LLM-based Framework for Speech Recognition

**arXiv ID:** 2601.17902 | [PDF](https://arxiv.org/pdf/2601.17902v1)

**作者:** Wenjie Tian `[一作]` (Northwestern Polytechnical University), Lei Xie `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 9595 | [OpenAlex ID](https://openalex.org/A5066245750)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 dLLM-ASR 框架，利用离散扩散 LLM 结合 ASR prior、长度自适应剪枝和置信度自适应去噪，实现并行语音识别。

**💡 创新点**

创新点包括：将 ASR prior 作为去噪起点、基于置信度的 token 早停、长度自适应剪枝，以及把 KV 缓存固定在语音特征上以减少计算。

**🔧 技术方法**

使用技术包括离散扩散 LLM（LLaDA）、Whisper‑large‑v3 编码器、1D 卷积+线性适配器、LoRA 微调、chat‑style 交互式提示、置信度阈值早停与 KV 缓存复用。

**📊 数据集**

实验数据集：LibriSpeech、CommonVoice‑22.0‑English、GigaSpeech（共 13,900 小时）以及 VoxPopuli 作为跨域测试集。

**📈 对比分析**

与 Whisper‑LLaMA3、Whisper‑Qwen3（AR）及 Whisper‑LLaDA（NAR）对比；在 WER 方面与 AR 模型持平，RTF 仅为 0.063，较 Whisper‑LLaMA3 快 4.44×、Whisper‑LLaDA 快 27.6×。

**⚠️ 局限性**

局限性：目前仅支持离线批处理；对非英语或极端噪声环境的鲁棒性待验证；置信度阈值需要调优，过低会降低识别质量。

---

## 438. Investigating How Music Affects Persuasion, Engagement, and Emotion in Data Videos

**arXiv ID:** 2601.17893 | [PDF](https://arxiv.org/pdf/2601.17893v1)

**作者:** Sarmistha Sarna Gomasta `[一作]` (University of Massachusetts Amherst), Ali Sarvghad `[通讯]` (City St George's University of London)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对单一全球水危机数据视频进行对照实验，比较无音乐、默认音乐和自定义音乐对说服力、参与度和情绪感受的影响。

**💡 创新点**

首次系统评估音乐对数据视频三大维度的影响，并对默认音乐与自定义音乐效果进行对比。

**🔧 技术方法**

采用实验室间组设计、量表测量（态度变化、参与度、情绪强度）以及音乐创作与视频剪辑技术。

**📊 数据集**

使用公开的全球水危机数据视频（原1分22秒，后调整至1分42秒），无其他外部数据集。

**📈 对比分析**

通过对三组（每组≈85/87）进行独立样本t检验与置信区间估计，发现默认音乐提升说服力，但在参与度与情绪上无显著优势，甚至自定义音乐表现更差。

**⚠️ 局限性**

仅用单个视频与单条自定义音乐，样本来自MTurk，缺乏多样性与普适性，音乐多样性与观众偏好未充分考虑。

---

## 439. iResolveX: Multi-Layered Indirect Call Resolution via Static Reasoning and Learning-Augmented Refinement

**arXiv ID:** 2601.17888 | [PDF](https://arxiv.org/pdf/2601.17888v1)

**作者:** Monika Santra `[一作]` (Pennsylvania State University), Gang Tan `[通讯]` (Pennsylvania State University)

**通讯引用:** 4077 | [OpenAlex ID](https://openalex.org/A5010830558)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种多层级混合框架，结合保守的静态值集分析与机器学习的软签名评分及选择性反向分析，以实现对二进制间接调用的精确解析并生成带置信度的 CFG；

**💡 创新点**

创新点在于：①将静态推理与学习式精炼无缝耦合，既保证召回，又显著提升精度；②采用无监督的签名分离学习，使直接调用的训练即可推广到间接调用；③引入选择性跨过程反向追踪与内存扫描，补全 ML 模型的上下文缺失；④输出可调置信度的 p-IndirectCFG，满足多种下游需求；

**🔧 技术方法**

核心技术包括：基于块级的值集分析（BPA）求得上界；轻量级深度网络（4 层 DNN）进行软签名匹配（iScoreGen）；针对候选目标的选择性反向路径展开与内存指针扫掠（iScoreRefine）；可选的动态追踪模块用于阈值校准；

**📊 数据集**

使用 SPEC2006 以及五个真实世界 x86‑32 二进制（thttpd、memcached、lighttpd、exim、nginx）进行训练与评测，训练集源自 2664 个编译自 binutils/coreutils/MiBench 的二进制的直接调用对；

**📈 对比分析**

与 CALLEE、AttnCall、BinDSA 以及 IDA/Ghidra/Angr 等现有工具对比，p-IndirectCFG 在 AICT 上平均减少 44% 以上，同时召回率仅下降 0.4%；在不同配置下可在 97.8%–98.2% 的召回范围内实现 18%–48% 的 AICT 降低，F1 模式下提升 80% 以上；相较于传统工具在精度和召回双向表现更优；

**⚠️ 局限性**

主要限制包括：仅支持 x86‑32（因 BPA 引擎限制）；对静态前端精度依赖较高，若前端精度不足会影响后续层；运行时间比纯 ML 方法慢，尤其在大二进制上；动态追踪阈值设定需人工或额外工具支持；未来需扩展到 x86‑64/ARM、提升 L2 适应性并探索更高效的反向分析策略。

---

## 440. PEAfowl: Perception-Enhanced Multi-View Vision-Language-Action for Bimanual Manipulation

**arXiv ID:** 2601.17885 | [PDF](https://arxiv.org/pdf/2601.17885v1)

**作者:** Qingyu Fan `[一作]` (Institute of Automation, Chinese Academy of Sciences), Xun Cao `[通讯]` (Nanjing University)

**通讯引用:** 5652 | [OpenAlex ID](https://openalex.org/A5058572381)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多视角感知增强的视觉-语言-动作模型PEAfowl，用于双手协作操作，能在遮挡、视角和场景变化下保持稳定执行指令。

**💡 创新点**

创新点在于：①通过预测每个视觉标记的深度分布实现可微3D提升与跨视角邻域聚合，构建几何一致的跨视角表示；②用Perceiver式文本查询读取器在冻结CLIP特征上实现迭代文本驱动的视觉提取，显著提升指令对齐；③仅在训练阶段使用深度教师蒸馏，无推理开销。

**🔧 技术方法**

技术包括：多尺度RGB‑D特征提取、基于离散深度分布的3D提升、跨视角K近邻聚合、Perceiver‑style文本查询读取、SEM式联合中心扩散动作解码器以及联合损失（扩散、正则化、深度蒸馏）。

**📊 数据集**

主要使用RoboTwin 2.0仿真基准（9个双手任务，清洁与域随机化两种设置）以及对应的双臂AgileX物理机器人数据（6个任务，含VR遥控演示）。

**📈 对比分析**

与基准方法（ACT、DP、DP3、π₀、RDT、SEM）比较，PEAfowl在域随机化设置下平均成功率提升至47.1%（比SEM高23pp），在物理机器人上实现68.3%平均成功率，显示出显著的泛化与仿真到现实的迁移性能。

**⚠️ 局限性**

局限性包括：对训练时深度教师的依赖，需预训练深度模型；在极端遮挡或非结构化场景下跨视角聚合可能仍受深度噪声影响；模型参数约3亿，仍受限于嵌入式部署。

---

## 441. EEG Foundation Models: Progresses, Benchmarking, and Open Problems

**arXiv ID:** 2601.17883 | [PDF](https://arxiv.org/pdf/2601.17883v1)

**作者:** Dingkun Liu `[一作]` (Huazhong University of Science and Technology), Dongrui Wu `[通讯]` (Zhongguancun Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对近两年内 50 种 EEG 基础模型进行综述，构建统一的分类框架，并在 13 个不同 BCI 任务上对 12 个公开基础模型与传统/专用模型进行系统对比；

**💡 创新点**

创新点在于：①提出了标准化的 EEG 基础模型设计维度与预训练策略的统一框架；②构建了包含 LOSO 与少样本内置学习两种评估场景的大规模基准；③对模型规模、预训练目标与迁移性能的关系给出经验性结论；

**🔧 技术方法**

使用的技术包括：多种自监督预训练目标（遮蔽重构、token 重构、频域重构、代码本、因果预测）、Transformer/Mamba 等主干架构、全参数微调与线性探测两种 fine‑tune 策略；

**📊 数据集**

实验数据集涵盖 13 个 BCI 任务（MI、P300、SSVEP、癫痫检测、情绪识别、睡眠分期、工作负荷等），共 9 种范式，来自公开公开数据集；

**📈 对比分析**

比较方法是对同一数据集、相同训练/测试划分下的模型进行 BCA/RMSE 等指标的直接对齐，结果显示线性探测往往效果不佳；部分基础模型（如 CBraMod、EEGMamba）在多任务上排名靠前，但整体上从零训练的专用模型仍能保持竞争力，模型规模增大并不必然提升性能；

**⚠️ 局限性**

局限性在于：①基础模型的通用表示能力不足，往往需全参数微调；②预训练数据量有限、质量参差，导致跨任务迁移受限；③缺乏针对特定范式的高效预训练策略和快速适配机制。

---

## 442. On the Emergence and Test-Time Use of Structural Information in Large Language Models

**arXiv ID:** 2601.17869 | [PDF](https://arxiv.org/pdf/2601.17869v1)

**作者:** Michelle Chao Chen `[一作]` (Max Planck Institute for Intelligent Systems), Siyuan Guo `[通讯]` (University of Cambridge)

**通讯引用:** 2378 | [OpenAlex ID](https://openalex.org/A5063197394)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了基于变换语法的自然语言数据集，系统研究了大型语言模型在学习结构化变换（如被动化、NP提升、疑问化等）过程中的结构信息出现与表征，并评估其在测试时进行组合生成的能力。

**💡 创新点**

创新点包括：① 将变换语法视作合成任务生成一个可控的“结构转化”数据集；② 在模型训练过程中监测结构信息的出现并关联到复杂推理任务的性能；③ 通过细粒度的注意力/MLP消融与线性判别分析揭示模型内部对结构变换的编码机制。

**🔧 技术方法**

使用的技术包括：大规模语言模型（Pythia‑410M、Llama3‑8B）在该数据集上的训练与 LoRA 微调；结构信息量化（L2 范数、cosine 相似度）；多阶段微调（单步、双步、OOD 组合）和精确/部分匹配评估；以及因果消融与 LDA 线性探测器进行机制解释。

**📊 数据集**

所用数据集为人工生成的“Transformational Grammar Dataset”，包含约 2000 条单层变换样本和 500 条多层嵌套变换样本，涵盖 10 种变换类型（被动化、NP 提升、疑问化等），由 DeepSeek‑V3 生成并人工筛选。

**📈 对比分析**

对比方法：全参数微调、LoRA 微调、LoRA + 中间结果消除；评价指标为精确匹配率和部分匹配率。实验结果显示：单变换下 LoRA 可达 96% 以上精确匹配；双变换若提供中间结果可达 82%‑83%；若不提供中间结果精确匹配几乎为 0%，但部分匹配仍能达到 80%‑85%；OOV 组合的精确匹配率低至 0%‑50%，部分匹配在 45%‑50% 之间。

**⚠️ 局限性**

局限性：仅在英语语料上验证，跨语言泛化未评估；模型规模仅 8B，可能无法推广到更大模型；实验聚焦单层变换，未拆解变换内部子操作；数据集规模有限，可能导致模型记忆化；实验未覆盖更复杂的嵌套组合。

---

## 443. D-Models and E-Models: Diversity-Stability Trade-offs in the Sampling Behavior of Large Language Models

**arXiv ID:** 2601.17865 | [PDF](https://arxiv.org/pdf/2601.17865v1)

**作者:** Jia Gu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM在任务级分布下的采样行为，定义并验证了两种模型类型D-模型与E-模型，并在代码生成和推荐任务中进行对比。

**💡 创新点**

首次从概率采样视角系统区分并量化LLM的“多样性-稳定性”两种行为模式，揭示它们对任务适配性的根本差异。

**🔧 技术方法**

采用概率分布采样实验、e-score和ATVD指标，结合温度调节、层级概率追踪和配额补偿分析等技术。

**📊 数据集**

通过人工构造的极端与均匀分布任务，以及HumanEval和MovieLens‑1M数据集进行实验。

**📈 对比分析**

通过ATVD、e-score、Δpass、precision、can‑hit等指标比较两类模型，在极端分布下D‑模型表现更好，在均匀分布下E‑模型更优，代码生成中D‑模型Δpass高，推荐中E‑模型can‑hit高。

**⚠️ 局限性**

受限于API取样范围、温度可控性有限及缺乏跨任务普适验证，无法全面证明两类模型的普适性和可解释性。

---

## 444. Phase-Rotated Symbol Spreading for Scalable Rydberg Atomic-MIMO Detection

**arXiv ID:** 2601.17838 | [PDF](https://arxiv.org/pdf/2601.17838v1)

**作者:** Jiuyu Liu `[一作]` (University of Surrey), Rahim Tafazolli `[通讯]` (University of Surrey)

**通讯引用:** 19388 | [OpenAlex ID](https://openalex.org/A5032549075)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出相位旋转符号扩展（PRSS）方案，通过在发射端发送相位相差π/2的符号两次，使Rydberg原子MIMO接收器能够重构出等价线性信号并使用传统RF‑MIMO检测算法。

**💡 创新点**

首次将Rydberg接收器的非线性Rabi频率读数转化为可线性化的模型，利用双时隙相位旋转实现可扩展检测，并证明最优相位偏移为±π/2。

**🔧 技术方法**

使用相位旋转符号扩展、泰勒展开线性化、最小二乘估计、正交变换等技术，并结合ML、ZF等经典RF‑MIMO检测算法。

**📊 数据集**

仿真Rayleigh衰落信道，采用Rydberg原子参数（Rb 52D₅/₂/53P₃/₂）及5 GHz信号，比较4‑QAM与16‑QAM，在不同RSR（30‑45 dB）下评估性能。

**📈 对比分析**

与单槽RA‑MIMO（ML、EM‑GS）以及RF‑MIMO（ML、ZF）四种基准在8×4、128×64等规模下对比，PRSS在ML检测下比单槽提升≈3 dB，在ZF检测下比EM‑GS提升≈10 dB，且复杂度更低。

**⚠️ 局限性**

需要强参考信号（RSR≥30‑45 dB）、双时隙导致同步与时延要求，且在低RSR或非高斯噪声环境下性能可能下降，未来需降低RSR要求并适配更宽泛的信道环境。

---

## 445. Unleashing the Potential of Sparse Attention on Long-term Behaviors for CTR Prediction

**arXiv ID:** 2601.17836 | [PDF](https://arxiv.org/pdf/2601.17836v1)

**作者:** Weijiang Lai `[一作]` (Institute of Software, Chinese Academy of Sciences), Xingxing Wang `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了SparseCTR模型，用于CTR预测，针对长序列用户行为通过个性化时间分块和稀疏注意力实现高效建模。

**💡 创新点**

创新点包括①个性化时间分块(TimeChunking)；②三分支稀疏自注意力(EvoAttention)（全局、转移、局部）结合可学习相对时间编码(RelTemporal)；③在大规模场景下验证并展示可持续扩展的Scaling Law。

**🔧 技术方法**

采用Transformer自注意力、稀疏自注意力、相对时间编码、RMSNorm、SwiGLU FFN、GPU并行计算以及在线A/B实验评估技术。

**📊 数据集**

使用工业级用户行为数据集（86M+用户）、阿里巴巴1.14M用户数据集和Ele.me1.44M用户数据集进行实验。

**📈 对比分析**

与10个基线模型（DIN、CAN、SoftSIM、HardSIM、ETA、TWIN-V2、BST、HSTU、LONGER、SUAN）在AUC上比较，SparseCTR在三数据集分别提升约8.5%、26.6%和25.2%；在线A/B实验中CTR提升1.72%，CPM提升1.41%，推理时间保持40ms。

**⚠️ 局限性**

局限性在于对时间戳准确性敏感、稀疏注意力需调参以适应极长序列、实现复杂度较高，以及对多渠道异构特征的适配仍需进一步验证。

---

## 446. An Effective and Cost-Efficient Agentic Framework for Ethereum Smart Contract Auditing

**arXiv ID:** 2601.17833 | [PDF](https://arxiv.org/pdf/2601.17833v1)

**作者:** Xiaohui Hu `[一作]` (Huazhong University of Science and Technology), Ningyu He `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5091926865)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个基于代理式工作流的智能合约审计框架，自动识别复杂业务逻辑漏洞并进行高精度验证。

**💡 创新点**

创新点在于：① 通过图论社区检测和LLM语义标签实现功能级上下文分块；② Plan‑Remind‑Solve三阶段推理策略结合神经符号推理；③ 多层误报过滤器（上下文聚合、语义去重、威胁模型评估）显著降低假警。

**🔧 技术方法**

使用的技术包括：Slither静态分析、Python‑Louvain社区检测、网络X权重图、LLM（Claude、Gemini、GPT‑oss‑120B等）与MCP交互、Z3符号求解、OpenAI嵌入与DBSCAN聚类、LangGraph构建代理。

**📊 数据集**

实验数据集包括：真实高价值攻击（20起）、公开基准（30起）、竞赛项目（30起）以及对后June 2025新攻击的回溯检验，构成多维评估。

**📈 对比分析**

相较于行业基线（Claude、GPTScan、LLMSmartAudit、Hound），框架在高价值攻击检测率达86.7%，误报率降至31–49%，F1提升至0.63；成本仅$0.22/项目或$2.31/10K LOC，执行时间≤30 min，性能显著优于现有方法。

**⚠️ 局限性**

局限性包括：仅支持Solidity，依赖Slither；知识库需手动维护，难以覆盖全新攻击类；模拟范围局限于单笔交易，无法覆盖治理长周期或链下攻击；高阶模型推理仍受LLM不确定性影响。

---

## 447. A Universal Load Balancing Principle and Its Application to Large Language Model Serving

**arXiv ID:** 2601.17855 | [PDF](https://arxiv.org/pdf/2601.17855v1)

**作者:** Zixi Chen `[一作]` (Peking University), Zijie Zhou `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并解决了大型语言模型推理中由于阻塞同步和不可迁移作业导致的负载不平衡问题，提出了一种通用的“Balance‑Future”负载均衡原则。

**💡 创新点**

创新点在于将短期未来负载预测与整数优化相结合，得到了一种能够在恶劣的在线环境下给出可证明的最坏情况提升因子（随批大小和系统规模呈 √(B log G) 级增长）的通用负载均衡策略。

**🔧 技术方法**

主要技术包括基于短窗口负载预测的整数规划求解、理论分析证明最坏情况改进、以及在LLM解码阶段的 DP/EP/TP 同步模型中的实现。

**📊 数据集**

实验使用了公开的 BurstGPT 数据集以及内部工业级 LLM 请求轨迹。

**📈 对比分析**

与传统的 FCFS/JSQ/轮询等基线相比，Balance‑Future 在公开数据集上提高了约 14% 的吞吐量、降低了约 13% 的 TPOT，并将能耗降低约 3.4%；在工业数据集上亦显示类似提升。

**⚠️ 局限性**

局限性包括依赖中心化的等待队列接口、无法迁移或预处理作业、仅适用于非递减工作负载漂移，以及未与能耗调度或多目标优化结合。

---

## 448. ShapLoRA: Allocation of Low-rank Adaption on Large Language Models via Shapley Value Inspired Importance Estimation

**arXiv ID:** 2601.17921 | [PDF](https://arxiv.org/pdf/2601.17921v1)

**作者:** Yi Zhao `[一作]` (Singapore Management University), Wei Zhu `[通讯]` (University of Hong Kong)

**通讯引用:** 17697 | [OpenAlex ID](https://openalex.org/A5068308955)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了ShapLoRA框架，改进了LoRA的秩分配方法；

**💡 创新点**

核心创新在于引入Shapley敏感度（Shapley sensitivity），将Shapley值思想与梯度敏感度结合，提供可解释且可靠的秩重要性度量；

**🔧 技术方法**

技术主要包括LoRA及其SVD重参数化、随机掩码下的梯度敏感度计算、两阶段（秩分配+微调）的训练流程以及对比实验；

**📊 数据集**

使用多种基准任务的数据集：ARC-e/c、OBQA、PIQA、BoolQ、AQuA、GSM8k、MT‑Bench、MMLU、BBH、GLUE（SST‑2、RTE、QNLI）、E2E、WikiSQL、UltraChat等；

**📈 对比分析**

与多种SOTA PEFT方法（LoRA、AdaLoRA、AutoLoRA、MOELoRA、DoRA、Parallel‑Adapter、Learned‑Adapter、P‑tuning v2、IAPT、BitFit、(IA)³、SSP）进行对比，ShapLoRA在相同可调参数预算下在大部分任务上取得最高或接近最高分数；

**⚠️ 局限性**

局限性包括：需要两阶段训练导致一定的额外训练成本；随机掩码近似Shapley值的计算仍有随机性，虽然稳定但耗时；在更大模型或不同任务上效果需进一步验证。

---

## 449. treaming-dLLM: Accelerating Diffusion LLMs via Suffix Pruning and Dynamic Decoding

**arXiv ID:** 2601.17917 | [PDF](https://arxiv.org/pdf/2601.17917v1)

**作者:** Zhongyu Xiao `[一作]` (Beijing Institute of Technology), Han Hu `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 9653 | [OpenAlex ID](https://openalex.org/A5091049278)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种无训练的推理加速框架 Streaming-dLLM，通过削减后缀冗余和动态自适应并行解码来显著提升扩散式大语言模型的推理速度。

**💡 创新点**

创新点包括：① 衰减引导后缀建模，仅保留邻近后缀块和终止位置信息，减少空间冗余；② 动态自信度自适应并行解码策略，根据遮蔽比例动态调整阈值；③ 早期退出机制在预测到 EOS 后即停止剩余块的解码。

**🔧 技术方法**

采用扩散式语言模型块级解码、滑动窗口与 RoPE 位置信息、KV 缓存重用、动态阈值公式 τ(t)=τ₀(1-α(1-r_mask))、并行解码选择函数 S() 等技术。

**📊 数据集**

在 GSM8K、MATH、HumanEval、MBPP、GSM8K-CoT 等多任务数据集上进行评估。

**📈 对比分析**

与原始 dLLM、dKV-Cache、Prefix-Cache、Fast-dLLM 等方法对比，速度提升 3.7×–13.3×，在 512 长度时可达 68×，在 MBPP 512 上实现 68.2×；同时保持或略优于原模型准确率，推理延迟下降约 85%。

**⚠️ 局限性**

局限性：需手动调节窗口大小 w、α 等超参，部分任务的动态阈值可能导致精度波动；对极长文本依赖尾部位置信息；仅在块级解码的 dLLM 上验证，通用性尚待进一步验证。

---

## 450. From Statistical Disclosure Control to Fair AI: Navigating Fundamental Tradeoffs in Differential Privacy

**arXiv ID:** 2601.17909 | [PDF](https://arxiv.org/pdf/2601.17909v1)

**作者:** Adriana Watson `[一作]` `[通讯]` (Purdue University), Adriana Watson (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统阐述了差分隐私（DP）与公平性约束在机器学习中的三方权衡，提供了理论边界、实验证据与实践策略。

**💡 创新点**

创新点在于：① 将 Dalenius 的不可实现理论与现代公平性指标关联，揭示三方 Pareto 前沿；② 推导了 DP-公平-效用的下界与样本阈值；③ 综合了多种 DP 机制与公平修正方法，提出可操作的决策框架。

**🔧 技术方法**

主要技术包括：差分隐私（Laplace、Gaussian、Exponential、SVT、PTR）、DP‑SGD 训练、群体特定隐私预算、合成数据生成（PrivBayes、DP‑GAN）、后处理公平校正与阈值优化。

**📊 数据集**

使用的典型数据集包括 Adult（UCI）、COMPAS（刑事司法）、医学假设数据集（如烟民数据库）以及公开的多族群基因标记数据。

**📈 对比分析**

对比方法：将普通神经网络、加公平约束网络、单独 DP 网络和 DP+公平网络在同一数据集上评估准确率、风险差异与公平指标。实验显示 DP+公平模型在保证隐私的同时，准确率与公平度相对更好，但在极低 ε 下仍会显著退化。整体性能表现符合理论预期，即隐私越强、效用与公平越受限。

**⚠️ 局限性**

局限性：① 论文的理论界限主要针对二分类、群体平均统计，未覆盖多模态或顺序数据；② 合成数据与群体预算分配的伦理与有效性仍需深入验证；③ 现有公平修正方法在高维、稀疏样本场景下仍可能导致过度偏差。

---

## 451. Evolving Interdependent Operators with Large Language Models for Multi-Objective Combinatorial Optimization

**arXiv ID:** 2601.17899 | [PDF](https://arxiv.org/pdf/2601.17899v1)

**作者:** Junhao Qiu `[一作]` (City University of Hong Kong), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 38375 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出一种基于大型语言模型的多算子协同进化框架（E2OC），通过 MCTS 对算子设计思想进行组合搜索并利用算子旋转评估，实现多目标进化算法（MOEA）的自动算子组合生成。

**💡 创新点**

创新点在于将算子间的耦合关系建模为马尔可夫决策过程，引入设计思路空间与 MCTS 的渐进搜索，以及算子旋转机制，实现算子设计策略与可执行代码的共同进化；同时提供了对现有单算子 AHD 方法的系统性提升。

**🔧 技术方法**

主要技术包括：大型语言模型（如 deepseek-chat）作为算子设计师；蒙特卡洛树搜索（MCTS）进行设计思路组合搜索；算子旋转演化评估策略；多目标评估指标（Hypervolume、IGD）和可视化工具；对比实验中采用 NSGA‑II、NSGA‑III、MOEA/D、PPLS/D 等传统 MOEA 算法。

**📊 数据集**

使用的公开基准数据集为两类经典多目标组合优化问题：灵活车间调度问题（FJSP）与旅行商问题（TSP），分别考虑二目标与三目标版本，共包含 15 个 FJSP 实例（最大 mk15）和 3 组 TSP 实例（k=20/50/100）。

**📈 对比分析**

在所有对比实验中，E2OC 在二目标与三目标 FJSP、TSP 上相较于专家设计的算子组合提升了 10%–32% 的 Hypervolume，且在与现有 LLM‑AHD 方法（Random、FunSearch、EoH、ReEvo、MCTS‑AHD、CD、UCB、LLM、Win‑UCB）对比中，E2OC 的平均 HV 在 0.2435–0.2467 之间，IGD 在 1.1423–1.1751 之间，明显优于其他方法。

**⚠️ 局限性**

局限性包括：对算子设计思路空间的构建依赖于初始 Prompt 与 LLM 生成质量，较大 AP 参数会导致搜索空间爆炸；算子旋转评估仍需大量多目标评估计算，成本较高；目前仅在 FJSP 与 TSP 上验证，对其他类型问题的泛化尚未充分探究；且框架依赖于固定的 LLM（如 deepseek-chat），在更强或更低成本的模型上表现可能不同。

---

## 452. Comparative Algorithmic Governance of Public Health Instruments across India, EU, US and LMICs

**arXiv ID:** 2601.17877 | [PDF](https://arxiv.org/pdf/2601.17877v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 453. ChatLearn: Leveraging AI to Transform Non-Native Speaker Communication Challenges as Language Learning Opportunities

**arXiv ID:** 2601.17837 | [PDF](https://arxiv.org/pdf/2601.17837v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 454. Assessment of Generative Named Entity Recognition in the Era of Large Language Models

**arXiv ID:** 2601.17898 | [PDF](https://arxiv.org/pdf/2601.17898v1)

**作者:** Qi Zhan `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 20988 | [OpenAlex ID](https://openalex.org/A5100684575)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估开源大语言模型在平面与嵌套命名实体识别任务中的性能，并验证其与传统序列标注模型的对比。

**💡 创新点**

证明在合适的输出格式与轻量化微调下，开源LLM可匹配甚至超越传统NER模型，且生成式NER不依赖实体记忆。

**🔧 技术方法**

采用LoRA参数高效微调、指令调优以及多种输出模板（inline bracketed/XML、JSON等）。

**📊 数据集**

使用四个标准NER数据集：CoNLL2003、OntoNotes5.0、ACE2005、GENIA。

**📈 对比分析**

对比传统预训练模型、闭源GPT‑3以及不同LLM规模，在平面任务上平均F1≈90%与传统模型相近；在嵌套任务上仍有差距，但整体表现优于闭源模型。

**⚠️ 局限性**

受限于低资源领域的知识覆盖、对极简指令的敏感度以及未使用强化学习或链式推理等更强训练方法。

---

## 455. Causal Pre-training Under the Fairness Lens: An Empirical Study of TabPFN

**arXiv ID:** 2601.17912 | [PDF](https://arxiv.org/pdf/2601.17912v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 456. Physiological and Behavioral Modeling of Stress and Cognitive Load in Web-Based Question Answering

**arXiv ID:** 2601.17890 | [PDF](https://arxiv.org/pdf/2601.17890v1)

**作者:** Ailin Liu `[一作]` (Ludwig Maximilian University of Munich), Fiona Draxler `[通讯]` (University of Mannheim)

**通讯引用:** 356 | [OpenAlex ID](https://openalex.org/A5010045512)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本研究通过在实验室中结合鼠标动态、眼动、心电、皮肤电等多模态传感器，对受试者在短时（5–20秒）完成的多项选择题进行实时监测，探讨时间压力与题目难度对认知负荷与情绪压力的影响，并尝试用机器学习模型实时识别受试者的压力与负荷状态；

**💡 创新点**

创新点在于：① 将短时调查任务与多模态实时测量结合，突破传统基于自评或响应时间的低分辨率监测；② 通过实验设计验证任务设计与主观体验在多模态信号中的一致性与偏差，为自适应调查系统提供依据；③ 提出分层微干预框架，阐明何种信号应触发何种实时干预。

**🔧 技术方法**

技术包括：实验室交互平台（Flask web）、Lab Streaming Layer同步、多模态数据预处理（EDA deconvolution、ECG R峰检测、眼动/鼠标轨迹提取）、统计分析（线性混合效应模型）、机器学习分类（SVM、KNN、随机森林）和模型解释（SHAP）。

**📊 数据集**

使用自制的双因素（题目难度×时间压力）实验数据，共29名受试者完成48道多项选择题，产生约1400个试次的多模态记录。

**📈 对比分析**

与传统基于自评或单一任务标签的比较显示：基于自评标签的模型性能更好，随机森林在4分类（放松+易、放松+难、压力+易、压力+难）任务中平均准确率0.52，宏F1 0.35；基于实验条件标签的准确率约0.45-0.48。模型解释显示响应时间和皮肤电是最重要特征。

**⚠️ 局限性**

局限包括：① 样本量小且受试者为实验室内受控环境，未验证在真实在线调查场景下的泛化；② 仅使用汇总特征，未利用时序或深度学习；③ 受试者多为年轻人，可能缺乏对老年人或不同文化群体的适用性；④ 隐私与伦理风险需在实际部署前解决。

---

## 457. Self-Manager: Parallel Agent Loop for Long-form Deep Research

**arXiv ID:** 2601.17879 | [PDF](https://arxiv.org/pdf/2601.17879v1)

**作者:** Yilong Xu `[一作]` (University of Chinese Academy of Sciences), Yiwei Wang `[通讯]` (University of California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Self-Manager 并行单体循环架构，主线程将复杂任务拆分为子任务，异步分配给子线程执行，实现并发、异步与上下文隔离。

**💡 创新点**

创新点包括：1）将单体循环改造为多线程并行执行；2）引入 Thread Control Block (TCB) 结构实现子线程的自管理；3）通过子线程并行化和上下文隔离显著缓解线性上下文增长与信息丢失。

**🔧 技术方法**

使用基于 ReAct 的思考‑行动‑观察模式，集成搜索、阅读等工具调用，利用线程调度、异步执行、上下文隔离、子任务分解等技术实现 Self-Manager。

**📊 数据集**

使用 DeepResearch Bench（100 个长表格深度研究任务）和 BrowseComp-Plus（多跳深度搜索任务）两个基准数据集进行实验。

**📈 对比分析**

通过与单体 ReAct、ReSum、FoldAgent，固定多代理工作流（LangChain、AI‑Q），以及专有深度研究系统（Gemini Deep Research、OpenAI DeepResearch）对比；在 DeepResearch Bench 上 Self-Manager 在多指标上优于所有单体基线，并与专有系统差距缩小；在 BrowseComp‑Plus 上也取得单体最高成绩。

**⚠️ 局限性**

局限性：整体运行时间和成本略高；需要明确的子任务划分，若任务难以拆分则优势有限；工具调用开销较大；并行调度实现复杂度较高。

---

## 458. Artificial Intelligence and Intellectual Property Rights: Comparative Transnational Policy Analysis

**arXiv ID:** 2601.17892 | [PDF](https://arxiv.org/pdf/2601.17892v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 459. Think Locally, Explain Globally: Graph-Guided LLM Investigations via Local Reasoning and Belief Propagation

**arXiv ID:** 2601.17915 | [PDF](https://arxiv.org/pdf/2601.17915v1)

**作者:** Saurabh Jha `[一作]` (IBM Research), Ruchir Puri `[通讯]` (IBM Research)

**通讯引用:** 3310 | [OpenAlex ID](https://openalex.org/A5045722906)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于图结构的LLM诊断框架EoG，通过将局部证据检索与全局状态管理分离，实现非单调信念修正，从而在IT运维诊断任务中大幅提升诊断准确性和一致性。

**💡 创新点**

创新点包括：① 将运营诊断建模为在操作图上的诱导推理；② 设计了分离式架构——LLM仅做局部推断，符号控制器负责图遍历、状态维护和消息传递；③ 引入了语义信念传播（SBP）机制，实现信念的显式记录与随新证据自动修正；④ 通过控制上下文窗口和工具调用，显著减少了上下文溢出与工具失败。

**🔧 技术方法**

核心技术：基于LLM的局部推断策略、Deterministic Controller（图遍历与消息调度）、Context Contract（限定上下文包）、语义信念传播（类似贝叶斯传播）、Actor模型并发实现、MCP接口获取日志/指标/追踪等观测数据。

**📊 数据集**

使用ITBench基准数据集，共35个Kubernetes环境诊断场景（包含错误配置、资源耗尽、级联故障等），每个场景多次复测以评估一致性。

**📈 对比分析**

与传统ReAct型代理对比，EoG在Pass@3、Majority@3、RC Entity F1等指标上提升显著：如GPT‑5.1模型从22.9%/8.6%提升至88.9%/86.1%；在不同模型上平均提升20–40个百分点，且token消耗仅为ReAct的一半。实验还表明，仅使用控制器就能显著弥补ReAct的“GT never saw”与“证据未检索”缺陷，加入SBP进一步提升10–65%。

**⚠️ 局限性**

局限性包括：① 仍依赖预先构建的操作图与工具接口，图结构不完整时仍可能误判；② 对非常大规模系统的实时推理依赖并发控制，实际部署时需关注资源调度与延迟；③ 证据缺失或不可检索时，系统只能退回为“Defer”，不一定能给出根因；④ 对新兴多模态或非结构化数据的支持尚不充分。

---

## 460. FARM: Few-shot Adaptive Malware Family Classification under Concept Drift

**arXiv ID:** 2601.17907 | [PDF](https://arxiv.org/pdf/2601.17907v1)

**作者:** Numan Halit Guldemir `[一作]` (Centre for Secure Information Technologies), Jesús Martínez-del-Rincón `[通讯]` (Centre for Secure Information Technologies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 FARM 的框架，能够在 Windows PE 恶意软件检测中检测并自适应概念漂移，包括协变量漂移与标签漂移。

**💡 创新点**

创新点在于将三元组自编码器与 DBSCAN 聚类结合实现无监督漂移检测，并利用原型网络实现少量样本快速适应，随后可触发全量再训练以实现长期集成。

**🔧 技术方法**

使用三元组自编码器（triplet autoencoder）+ DBSCAN 聚类 + 原型网络（prototypical network）+ 软阈值动态阈值、缓冲区触发再训练等技术。

**📊 数据集**

基准数据集为 BenchMFC（拆包 Windows PE 样本），实验中选取 16 个家族（8 训练，8 漂移，8 未见）共 12,000 份样本。

**📈 对比分析**

与 CADE、基于 95% 分位数阈值的 Percentile 方法对比，FARM 在标签漂移下平均 F1 0.825，协变量漂移时 F1 提升 5.6%；少样本适应后平均 F1 0.85，完整再训练后提升至 0.94，整体表现显著优于基线。

**⚠️ 局限性**

局限性包括：聚类超参数（ε、minPts）对漂移检测敏感；嵌入空间对行为相似但类内差异的区分不足（如 vobfus 与已知家族重叠）；未处理打包样本；三元组采样未使用硬负采样，可能限制判别边界细化。

---

## 461. Masked Depth Modeling for Spatial Perception

**arXiv ID:** 2601.17895 | [PDF](https://arxiv.org/pdf/2601.17895v1)

**作者:** Bin Tan `[一作]`, Nan Xue `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并训练了 Masked Depth Modeling (MDM) 框架，利用 RGB‑D 传感器缺失的深度值作为自然掩码，实现稠密深度完成、单目深度估计，并将预训练模型用于视频深度恢复、3D 点跟踪和机械臂抓取等下游任务。

**💡 创新点**

创新点在于：① 把缺失深度视为自监督掩码，利用视觉上下文进行深度恢复；② 在 ViT 中采用分离的 RGB/Depth patch embedding 和模态+空间位置编码；③ 构建规模可达 3.2M 的真实+1M 合成 RGB‑D 数据集并与公开数据结合，支持大规模预训练；④ 将预训练模型作为强大的深度先验，用于多任务学习和后续系统集成。

**🔧 技术方法**

核心技术包括 Vision Transformer（ViT‑Large）编码器 + ConvStack 解码器、Masked Depth Modeling 目标、分离 patch embedding 与双重位置编码、混合精度 BF16 训练、Stereo 伪标签生成（FoundationStereo）、Diffusion 政策抓取、数据增强与梯度裁剪等。

**📊 数据集**

使用的数据集：3.2M 自制 RGB‑D 样本（1M 合成 + 2M 真实） + 7 个公开 RGB‑D 数据集，合计约10M 训练样本；基准评测使用 iBims、NYUv2、DIODE、ETH3D、KITTI、DDAD、Sintel、GSO、TartanAir 等。

**📈 对比分析**

与 OMNI‑DC、PromptDA、PriorDA 等深度完成基线对比，RMSE 降低 40%+；与 DINOv2 对比的单目深度估计，误差下降 10–20%；在 FoundationStereo 训练中，MDM 预训练加速收敛、最终性能优于原始模型；在视频深度恢复、3D 点跟踪和抓取任务中，表现出更高的时序一致性、更准确的轨迹和更高的抓取成功率。

**⚠️ 局限性**

局限性：仍需大量标注/伪标签数据；对极透明或高反射物体的预测仍不稳定；训练仅基于单帧，无显式时序建模；在极端光照或极端环境下的鲁棒性尚未完全验证。

---

## 462. VidLaDA: Bidirectional Diffusion Large Language Models for Efficient Video Understanding

**arXiv ID:** 2601.17868 | [PDF](https://arxiv.org/pdf/2601.17868v1)

**作者:** Zhihao He `[一作]` (Shanghai Jiao Tong University), Weiyao Lin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9010 | [OpenAlex ID](https://openalex.org/A5059372830)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 VidLaDA，一种利用双向扩散语言模型的视频大语言模型，并配合 MARS‑Cache 推理加速框架；

**💡 创新点**

创新点在于：①用双向扩散替代传统自回归解码，消除因因果掩码导致的视听信息不对称；②通过 Anchor Token 与异步视觉缓存实现大幅度推理加速；

**🔧 技术方法**

技术包括：扩散语言模型（DLM）、全双向注意力、2×2 视觉池化、Anchor Token 搜索、帧级局部注意力、模态异步缓存等；

**📊 数据集**

数据集：自构建 2‑30 min 视频集、MME、ReXTime、LongVideoBench、EgoSchema、MLVU、Video‑MMMU、Video‑MMM、Video‑MME 等；

**📈 对比分析**

与 SOTA 自回归模型（如 Qwen2.5‑VL、LLaVA‑Video、LLaVA‑OneVision）及 DLM 基线（Dream‑VL、LLaDA‑V、SDAR‑VL）对比，VidLaDA 在多项基准上性能更优或相近，同时 MARS‑Cache 可实现约 12× 的推理速度提升；

**⚠️ 局限性**

局限性：对 Anchor Token 的选择和刷新策略依赖度高，极长视频或高帧率下仍面临计算瓶颈；模型规模与硬件要求较高，未来需进一步验证在更广泛场景下的鲁棒性。

---

## 463. EFT-CoT: A Multi-Agent Chain-of-Thought Framework for Emotion-Focused Therapy

**arXiv ID:** 2601.17842 | [PDF](https://arxiv.org/pdf/2601.17842v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 464. Domain Generalization with Quantum Enhancement for Medical Image Classification: A Lightweight Approach for Cross-Center Deployment

**arXiv ID:** 2601.17862 | [PDF](https://arxiv.org/pdf/2601.17862v1)

**作者:** Jingsong Xia `[一作]` (Nanjing Medical University), Siqi Wang `[通讯]` (Nanjing Medical University)

**通讯引用:** 3795 | [OpenAlex ID](https://openalex.org/A5100420069)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种轻量级的量子增强域泛化框架，用于医学图像分类，能够在无真实多中心标注数据的情况下实现跨中心的稳健推理；

**💡 创新点**

创新点在于将可参数化的量子电路嵌入经典网络，进行非线性特征映射与纠缠建模，同时结合多域图像扰动、域对抗训练和测试时自适应，提升域不变特征学习；

**🔧 技术方法**

使用MobileNetV2编码器、梯度反转域判别器、参数化量子电路（量子特征增强层）以及基于梯度反转和批归一化的TTA；

**📊 数据集**

在模拟的多中心医学影像数据集上进行实验（通过亮度、对比度、锐化和噪声等方式生成三类虚拟域），未使用真实多中心数据；

**📈 对比分析**

与ResNet18、MobileNetV2、EfficientNet-B0、SimpleCNN等基线模型比较，DG‑Quantum在准确率、AUC、F1、敏感度与特异度等指标上均显著优于所有基线，置信区间更窄，表现出更高的稳定性和泛化能力；

**⚠️ 局限性**

局限性包括：量子模块仅为可模拟的浅层电路，规模受限；未验证多模态或真实跨中心数据；对量子硬件的鲁棒性与可扩展性尚待进一步研究。

---

## 465. Space-Air-Ground-Integrated Networks: The BER vs. Residual Delay and Doppler Analysis

**arXiv ID:** 2601.17859 | [PDF](https://arxiv.org/pdf/2601.17859v1)

**作者:** Chao Zhang `[一作]` (University of Southampton), Lajos Hanzo `[通讯]` (University of Southampton)

**通讯引用:** 83190 | [OpenAlex ID](https://openalex.org/A5091122305)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了空间‑空‑地集成网络（SAGIN）中残余多普勒效应与同步误差对16-QAM信号误码率（BER）的影响，并给出了基于相关Shadowed‑Rician通道的闭式BER表达式；

**💡 创新点**

创新点在于：①将残余多普勒建模为Jakes模型并结合相对论效应，构造完整的SAGIN通道模型；②推导相关Shadowed‑Rician通道的自相关与相关系数，并用双变量Gamma分布逼近其分布；③在此基础上得到含残余多普勒与同步误差的16-QAM闭式BER公式；

**🔧 技术方法**

主要技术包括：相关Shadowed‑Rician信道建模、Jakes多普勒模型、相对论时钟偏移建模、最小二乘信道估计与均衡、双变量Gamma分布逼近、Chebyshev‑Gauss求积求解BER积分；

**📊 数据集**

本文未使用公开数据集，而是通过仿真生成SAGIN通道参数：300km LEO、S‑波段、不同发射功率、Rician因子、Nakagami‑m参数、残余多普勒100‑1600 Hz、同步延迟0.2 ms等；

**📈 对比分析**

与Monte‑Carlo仿真结果进行对比，闭式公式与仿真吻合；BER随残余多普勒、同步延迟、大气阴影和Rician因子变化；在大气阴影或残余多普勒较大时，BER显著升高；

**⚠️ 局限性**

局限性包括：仅考虑单天线、16‑QAM调制；未考虑多用户、多天线或前向纠错；仿真参数范围有限，实际卫星轨道偏差、非理想硬件误差未纳入；模型假设与真实SAGIN环境仍有差距。

---

## 466. RAICL: Retrieval-Augmented In-Context Learning for Vision-Language-Model Based EEG Seizure Detection

**arXiv ID:** 2601.17844 | [PDF](https://arxiv.org/pdf/2601.17844v1)

**作者:** Siyang Li `[一作]` (Huazhong University of Science and Technology), Dongrui Wu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 14574 | [OpenAlex ID](https://openalex.org/A5008740867)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种利用视觉‑语言模型（VLM）对EEG波形图像进行癫痫发作检测的方法，并通过检索增强的上下文学习（RAICL）在零校准条件下提升性能。

**💡 创新点**

创新点在于：①将多通道EEG转换为堆叠色彩图像，充分利用VLM的视觉编码；②设计了基于代表性和相似性的检索策略，动态挑选最具代表性的少样本示例；③在VLM中嵌入专家先验知识的链式推理（CoT）提示，提高解释性和准确性。

**🔧 技术方法**

使用技术包括：EEG波形图像渲染（垂直堆叠、色彩编码、高清绘制）；视觉‑语言模型（Gemini‑3‑Flash、Qwen3‑VL‑32B、InternVL3‑38B）；检索增强的上下文学习（RAICL）；CoT提示工程；以及对比实验中CNN、Transformer、ResNet等传统模型。

**📊 数据集**

实验数据集为中国武汉儿童医院（CHSZ）和芬兰赫尔辛基大学医院（NICU）的癫痫发作EEG数据，包含多名患者的多通道EEG段，经过4秒无重叠划分后用于训练与评估。

**📈 对比分析**

与传统信号‑基模型（如EEGNet、Conformer）和基于图像的CNN（如ResNet、SwinTransformer）以及开放源VLM相比，RAICL‑Gemini‑3‑Flash在零校准情景下的平衡分类准确率（BCA）分别达到82.10%（CHSZ）和70.08%（NICU），明显优于其他方法。

**⚠️ 局限性**

局限性包括：①仍然是二分类任务，难以直接扩展到多类别或连续预测；②对VLM的视觉编码器高度依赖，未针对EEG特性进行微调；③检索与提示开销较大，实时部署受限；④对罕见或极端EEG模式的鲁棒性尚未充分验证。

---

## 467. Agentic AI for Self-Driving Laboratories in Soft Matter: Taxonomy, Benchmarks,and Open Challenges

**arXiv ID:** 2601.17920 | [PDF](https://arxiv.org/pdf/2601.17920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 468. Geometry-Grounded Gaussian Splatting

**arXiv ID:** 2601.17835 | [PDF](https://arxiv.org/pdf/2601.17835v1)

**作者:** Baowen Zhang `[一作]` (Hong Kong University of Science and Technology), Ping Tan `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 13819 | [OpenAlex ID](https://openalex.org/A5084953118)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出将高斯原语视为随机固体，并通过推导空缺函数与衰减系数，使其体渲染等价于光栅化，从而得到连续的中值深度，用于高质量形状重建。

**💡 创新点**

创新地将高斯原语与随机固体理论对齐，首次给出唯一的空缺表达式和衰减公式；利用连续传输特性实现无阶梯化的深度估计，显著提升多视一致性与几何细节。

**🔧 技术方法**

使用高斯剖面光栅化、随机固体体渲染理论、二分搜索求中值深度、闭式梯度传播、体积渲染的深度合成以及多视一致性正则化等技术。

**📊 数据集**

在公开的 DTU（15 场景）和 Tanks & Temples（6 场景）数据集上进行评估。

**📈 对比分析**

与现有 Gaussian Splatting 及基于 SDF 的方法（如 GeoSVR、PGSR）在 Chamfer Distance/F1‑Score 上进行对比；实验表明在 DTU 上可与主流方法持平，在 Tanks & Temples 上显著优于其它 Gaussian‑Splatting 方法，并且训练速度比 GeoSVR 与 PGSR 更快。

**⚠️ 局限性**

仍然比最快基线耗时更长（因二分搜索和多视正则化），仅深度使用体渲染，RGB/法向仍采用光栅化；对极大规模场景的进一步加速与完整体渲染仍是待解决的限制。

---

## 469. Prompt Injection Evaluations: Refusal Boundary Instability and Artifact-Dependent Compliance in GPT-4-Series Models

**arXiv ID:** 2601.17911 | [PDF](https://arxiv.org/pdf/2601.17911v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 470. On the Extension of Private Distributed Matrix Multiplication Schemes to the Grid Partition

**arXiv ID:** 2601.17834 | [PDF](https://arxiv.org/pdf/2601.17834v1)

**作者:** Christoph Hofmeister `[一作]` (Technical University of Munich), Rawad Bitar `[通讯]` (Technical University of Munich)

**通讯引用:** 487 | [OpenAlex ID](https://openalex.org/A5024768934)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文针对私有分布式矩阵乘法，设计了将外积分区代码扩展到网格分区的三种操作，并提出一种不属于扩展的全新GP‑CAT方案。

**💡 创新点**

创新点在于提供三种扩展操作实现OPP到GP的转换，并揭示GP扩展产生额外约束，从而提出直接为GP设计的更优GP‑CAT。

**🔧 技术方法**

使用了度表（Degree Table）、循环加法度表（Cyclic‑Addition Degree Table）以及有限域和根号性质的编码技术。

**📊 数据集**

论文为理论分析，无使用真实数据集；所有比较基于参数范围内的理论工人数计算。

**📈 对比分析**

通过在2≤K,M,L,T≤20的所有实例中计数，发现新方案在约29.4% 的实例中最优，平均比次优方案低约6.8%，最大差距达到31%。

**⚠️ 局限性**

局限在于仅评估工人数，未考虑通信开销、误差容忍、任务调度和真实网络环境，且对大规模参数缺乏实验验证。

---

## 471. Adaptive Weighting in Knowledge Distillation: An Axiomatic Framework for Multi-Scale Teacher Ensemble Optimization

**arXiv ID:** 2601.17910 | [PDF](https://arxiv.org/pdf/2601.17910v1)

**作者:** Aaron R. Flouro `[一作]`, Shawn P. Chadwick `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种多尺度教师集合的自适应加权知识蒸馏的公理化框架，涵盖token、task和context三层级的加权设计。

**💡 创新点**

创新点在于：①为每一层级给出五条结构性公理（归一化、正性、界限、连续性、序关系单调性），证明满足公理的加权器存在且非唯一；②通过乘积-归一化组合实现层级融合，保证整体加权器保持公理性；③在任何满足公理的加权器下，统一证明梯度下降收敛、稳健性与安全约束的全局理论保证。

**🔧 技术方法**

使用技术包括：公理化框架构建、乘积-归一化加权组合、随机逼近（SGD）收敛分析、收敛速率与扰动鲁棒性证明、Pareto/拉格朗日分析用于安全约束。

**📊 数据集**

论文为理论工作，未给出具体实验数据集；作者在后续实验中计划使用多任务、多域与安全敏感的公开数据集（如ImageNet‑VL, COCO‑MTL, medical text corpora）验证。

**📈 对比分析**

方法比较：与均匀加权、基于熵或方差的自适应加权等传统方案对齐；在理论上，所有满足公理的方案均达到相同的 O(1/t) 收敛率，并在安全指标上不低于均匀方案；实验结果（待公布）预期在准确率与安全阈值满足方面优于或至少与现有 heuristic 方法持平。

**⚠️ 局限性**

limitations: ①未给出具体实现细节和计算复杂度分析；②公理集合是否最小尚未证明；③对非平稳分布、动态温度调整等场景的理论扩展缺失；④实验验证缺乏，实际性能需进一步实证。

---

## 472. Benchmarking Direct Preference Optimization for Medical Large Vision-Language Models

**arXiv ID:** 2601.17918 | [PDF](https://arxiv.org/pdf/2601.17918v1)

**作者:** Dain Kim `[一作]` (Korea University), Jaewoo Kang `[通讯]` (Korea University)

**通讯引用:** 15135 | [OpenAlex ID](https://openalex.org/A5076917278)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对两种医学视觉语言模型（LLaVA‑Med 与 HuatuoGPT‑Vision）进行九种 DPO 变体的系统评估，并基于专家标注的视觉误判类型构造领域特定偏好对，进一步提升模型在 VQA、图像字幕和放射报告生成任务上的表现。

**💡 创新点**

首次将视觉误判的四大错误类别（模态误判、空间/左右混淆、解剖误判、解剖不精细）纳入偏好对构造，提出了领域特定的目标化 DPO 方法，显著超过通用 DPO 与单纯 SFT 的效果。

**🔧 技术方法**

采用多种 DPO 变体（文本仅 DPO、图像仅 CoPO、联合 mDPO、MMedPO 等）结合 LLM‑as‑judge 自动评估框架和 GPT‑4o 生成对抗样本；对偏好对进行图像检索与文本扰动。

**📊 数据集**

使用 VQA‑RAD、SLAKE、PathVQA 三大医学 VQA 数据集；AMBOSS 图像-字幕对集；MIMIC‑CXR 放射报告生成集；以及 80 条人工标注的错误示例构建自定义偏好对。

**📈 对比分析**

与基准模型和相同规模的 SFT 进行比较；通用 DPO 对 VQA 的提升仅与 SFT 相当；但基于错误类型的目标化 DPO 在 VQA 上相对最佳基线提升 3.6%，并在图像字幕与报告生成任务中实现了更为稳定的性能提升。

**⚠️ 局限性**

实验仅涵盖 7B 参数规模模型，未验证更大规模或新发布模型；对错误标签的依赖导致标注成本高，且实验场景仍未覆盖所有临床实际情况。

---

## 473. Revisiting 3D Reconstruction Kernels as Low-Pass Filters

**arXiv ID:** 2601.17900 | [PDF](https://arxiv.org/pdf/2601.17900v1)

**作者:** Shengjun Zhang `[一作]` (Tsinghua University), Yueqi Duan `[通讯]` (Tsinghua University)

**通讯引用:** 2268 | [OpenAlex ID](https://openalex.org/A5013973037)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于理想低通滤波器的 Jinc 核以及 Jinc splatting 方法，并在其基础上设计了通过频率调制实现的高效模调 Gaussian 与 Student's t 核，用于 3D 重建与新视角合成。

**💡 创新点**

创新点在于：①从 3D 理想低通滤波器逆变换得到 Jinc 核，实现在截止频率处瞬时降为零的理想低通；②通过频率调制将 Jinc 的理想频率特性与 Gaussian/Student 的快速空间衰减相结合，兼顾抗锯齿与计算效率。

**🔧 技术方法**

采用信号处理与频域分析、傅里叶变换、光线积分、坐标变换、梯度反向传播技术；在实现上基于 3D Gaussian Splatting 与 3D Student Splatting 代码框架，并实现了 Jinc splatting 与模调核的渲染与优化。

**📊 数据集**

使用的主要数据集包括 NeRF‑Synthetic（低分辨 64×64/128×128）、Mip‑NeRF‑360、Tanks & Temples、Deep Blending 等多场景数据。

**📈 对比分析**

与 3DGS、SSS、Mip‑NeRF‑360 等基线在 PSNR/SSIM/LPIPS 指标上进行对比；在低分辨率场景下，Jinc 核达到 29.87 dB PSNR，比 SSS 提升 0.70 dB、比 3DGS 提升 5.86 dB；在 128×128 仍保持显著优势；模调后的 Gaussian/Student 核在各数据集上平均提升约 0.7 dB PSNR，性能明显优于原始核。

**⚠️ 局限性**

主要局限：Jinc 核空间衰减慢导致高内存占用与矩形伪影；理想核存在切线波痕与振铃现象；调制策略虽缓解了空间衰减问题，但仍未彻底消除伪影；未来需进一步优化核设计或提升采样率以实现更高效无伪影的抗锯齿。

---

## 474. Quran-MD: A Fine-Grained Multilingual Multimodal Dataset of the Quran

**arXiv ID:** 2601.17880 | [PDF](https://arxiv.org/pdf/2601.17880v1)

**作者:** Muhammad Umar Salman `[一作]` (Mohammed Bin Zayed University of Artificial Intelligence), Mohammed Talha Alam `[通讯]` (Mohammed Bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了Quran‑MD多模态数据集，融合阿拉伯原文、英文翻译、音译及音频，在章节、节与词级别进行对齐；

**💡 创新点**

首次统一提供章节、节与词级别的文本、翻译、音译和多位朗读者的音频，填补了以往资源在多模态、跨朗读者对齐方面的空白；

**🔧 技术方法**

采用数据聚合、分层JSON模板设计、音频与文本对齐算法以及自动校验脚本实现数据统一与质量控制；

**📊 数据集**

整合三大公开来源：Kaggle的30位朗读者节级语音识别记录、专门的词级对齐文本/翻译/音译库，以及Internet Archive的词级音频集合；

**📈 对比分析**

通过与现有主要Quran数据集（如Quranic Arabic Corpus、Tanzil、MASAQ等）的特征表对比，Quran‑MD在文本、翻译、音译、词级音频、节级音频以及多位朗读者覆盖率方面均领先；

**⚠️ 局限性**

局限包括仅包含30位朗读者（不涵盖非母语朗读者或少数民族方言）、词级音频总时长约22小时，可能存在微小对齐误差，以及缺乏多语言翻译和深度语义标注。

---

## 475. MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging

**arXiv ID:** 2601.17858 | [PDF](https://arxiv.org/pdf/2601.17858v1)

**作者:** Jiapeng Wang `[一作]` (Renmin University of China), Jun Zhou `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MergeMix框架，利用模型权重插值作为低成本代理来优化LLM中期训练的数据混合比例。

**💡 创新点**

创新点在于证明权重插值可近似多域数据混合训练的第一阶优化动态，并将高成本的数据混合搜索转化为模型合并优化，具有高秩一致性、跨尺度可迁移和可层次化的搜索策略。

**🔧 技术方法**

技术包括：在共享初始化下训练域专属专家模型、线性权重插值合并、基于LightGBM的性能回归预测、基于下游基准的目标函数优化以及动态/静态混合策略和层次化合并。

**📊 数据集**

使用工业级大规模语料，按数学、代码、SFT（监督微调）和网页/其他四大域划分，并在此基础上对多个公开基准（如ARC、AGIEval、MATH、HumanEval、SuperGPQA等）进行评估。

**📈 对比分析**

与均匀采样、人工调参、RegMix回归等方法对比，MergeMix在大多数能力维度上匹配或超越人工基准，同时搜索成本降低超过100×；Spearman秩相关率>0.9，显示代理的可靠性。

**⚠️ 局限性**

局限性包括：依赖于中期训练的局部线性假设，可能无法捕捉更高阶交互；需要为每个域训练专家模型，若域数过多仍可能产生搜索复杂度；尚未在极大模型或更宽域多样性下充分验证。

---

## 476. Are we collaborative yet? A Usability Perspective on Mixnet Latency for Real-Time Applications

**arXiv ID:** 2601.17845 | [PDF](https://arxiv.org/pdf/2601.17845v1)

**作者:** Killian Davitt `[一作]` (King's College), Steven J. Murdoch `[通讯]` (University College)

**通讯引用:** 3625 | [OpenAlex ID](https://openalex.org/A5011443512)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过设计基于Web的协作测验系统，测量不同均值延迟（1 s、4 s、7 s、10 s）对用户完成实时协作任务的时间与主观体验的影响。

**💡 创新点**

创新点在于首次系统评估混网络（mixnet）引入的延迟对实时协作可用性的实证影响，并给出针对混网络运营者的可接受延迟范围建议。

**🔧 技术方法**

使用技术包括：Web前端协作平台、基于Automerge的CRDT实现、模拟第二用户（通过指数分布模拟打字延迟）以及统计分析（Friedman、Games‑Howell）。

**📊 数据集**

数据集：288名来自Prolific的英国受试者完成14道问答题的协作测验，共计约60 s任务时长；记录了完成时间、主观压迫感与感知速度的Likert评分。

**📈 对比分析**

方法比较：将各延迟水平与无协作控制组对比，采用方差不齐的非参数检验。结果显示1 s和4 s的延迟显著缩短完成时间（p<0.01），而7 s与10 s无显著优势，且用户主观压迫感随延迟增加而显著上升；效应量小至中等。

**⚠️ 局限性**

局限性包括：任务为人工构造的协作测验，未涵盖真实业务场景；模拟用户无法完全模拟真实互动；受试者学习效应、网络延迟未被纳入；延迟上限受试验时长限制；仅针对单跳混网络，未考虑多跳或中心化模型。

---

## 477. The Stateless Pattern: Ephemeral Coordination as the Third Pillar of Digital Sovereignty

**arXiv ID:** 2601.17875 | [PDF](https://arxiv.org/pdf/2601.17875v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 478. Feature-Space Generative Models for One-Shot Class-Incremental Learning

**arXiv ID:** 2601.17905 | [PDF](https://arxiv.org/pdf/2601.17905v1)

**作者:** Jack Foster `[一作]` (Samsung Research and Development Institute United Kingdom), Umberto Michieli `[通讯]` (Samsung Research and Development Institute United Kingdom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在一次样本的增量学习（1SCIL）场景下，该工作通过学习基类嵌入残差的共享分布，构建生成模型作为先验，以实现对新颖类的高效识别。

**💡 创新点**

创新点在于提出使用生成模型（VAE或扩散模型）捕捉基类残差的多模态结构，并将其作为先验推断单样本新类分布，突破传统高斯假设与多样本需求的局限。

**🔧 技术方法**

采用的技术包括基于预训练特征提取器的残差映射、条件VAE/扩散生成模型训练、重建误差/对数似然作为相似度度量，以及无模型更新的推理流程。

**📊 数据集**

实验使用CUB-200、CIFAR-100、CORe50、iCubWorld四个公开数据集，基类/新类划分分别为160/60、60/40、25/25、25/25，并在单类和多类1SCIL设置下评估。

**📈 对比分析**

与ProtoNet、RelationNet、FACT、SLDA等多种基线对比，所提方法在所有数据集上均实现了最优或同等优异的新类识别准确率，并在三项指标（BCR、NCR、AVG）中取得领先或相近成绩，尤其在多模态分布场景下显著优于传统高斯假设方法。

**⚠️ 局限性**

限制主要在于对预训练特征提取器的依赖（如DINOv2-s表现优于ResNet18），以及在极大类数或极度噪声数据下生成模型可能需要更深层次的可解释性与效率提升。

---

## 479. UniCog: Uncovering Cognitive Abilities of LLMs through Latent Mind Space Analysis

**arXiv ID:** 2601.17897 | [PDF](https://arxiv.org/pdf/2601.17897v1)

**作者:** Jiayu Liu `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27290 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建统一的潜在心智空间，分析LLM在推理中的认知能力，并提出基于潜在心智的候选排序策略提升推理准确率。

**💡 创新点**

将认知能力映射为稀疏可解释的潜在维度，揭示Pareto原理、能力特定签名及错误放大效应，并用潜在心智实现可插拔的推理提升。

**🔧 技术方法**

基于变分自编码器的潜在变量模型，使用Transformer实现后验与似然网络，采用稀疏映射与k-sparse正则化，结合动态时间归一化与激活强度排名。

**📊 数据集**

主要使用NuminaMath-CoT进行训练，CogMath、GSM8K、MATH-500、AIME24/25等数学推理基准进行评估。

**📈 对比分析**

与标准Chain-of-Thought、Self-Consistency、长度/困惑度/自我验证等方法对比，平均提升约7.5%，在AIME等高难度任务上表现尤为显著。

**⚠️ 局限性**

缺乏标准的解缠合度评估指标，且研究聚焦于数学推理，未验证在主观或开放式任务的泛化，且方法仅为后处理，未实现实时引导。

---

## 480. AI Personalization Paradox: Personalized AI Increases Superficial Engagement in Reading while Undermines Autonomy and Ownership in Writing

**arXiv ID:** 2601.17846 | [PDF](https://arxiv.org/pdf/2601.17846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 481. Data Siphoning Through Advanced Persistent Transmission Attacks At The Physical Layer

**arXiv ID:** 2601.17967 | [PDF](https://arxiv.org/pdf/2601.17967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 482. When Personalization Legitimizes Risks: Uncovering Safety Vulnerabilities in Personalized Dialogue Agents

**arXiv ID:** 2601.17887 | [PDF](https://arxiv.org/pdf/2601.17887v1)

**作者:** Jiahe Guo `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 15641 | [OpenAlex ID](https://openalex.org/A5017671620)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了“意图合法化”这一在长期个人化LLM代理中出现的安全失效模式，并通过新基准PS‑Bench对其进行系统评估；同时给出轻量化检测‑反射干预方法以缓解安全退化。

**💡 创新点**

①首次系统化识别并量化“意图合法化”这一安全缺陷；②设计了包含基础设置、主题记忆增强和角色驱动危害查询的PS‑Bench基准；③提供了内部表示分析的机制证据；④提出了可直接应用的检测‑反射干预策略。

**🔧 技术方法**

采用多种内存增强代理框架（LDAgent、Amem、Mem0、MemOS、MemU），大型LLM（GPT‑4o、GPT‑4o‑mini、Qwen3‑235B‑A22B、Qwen3‑8B、DeepSeek‑V3.2），自动危害检测器，PCA内部表示分析，以及检测‑反射干预机制。

**📊 数据集**

基于LoCoMo多轮对话历史与合成的主题会话，构建Persona‑Grounded Harmful Queries（约1,986条），并利用已有安全基准（SorryBench、Do‑Not‑Answer、HarmfulQA、ALERT、BeaverTails）形成最终测试集。

**📈 对比分析**

通过与无记忆（stateless）基线对照，使用攻击成功率（ASR）衡量安全性；实验表明个性化可使ASR上升15.8%–243.7%；检测‑反射干预后ASR下降约27%，恢复至近似stateless水平；不同模型与框架间存在显著差异。

**⚠️ 局限性**

①依赖合成会话与角色化危害查询，可能不完全反映真实交互；②未覆盖全部记忆设计、检索机制及多模态记忆；③评估仅为单轮文本，未探讨多轮或交互式情景；④自动危害检测器与人工评估的局限性。

---

## 483. MV-SAM: Multi-view Promptable Segmentation using Pointmap Guidance

**arXiv ID:** 2601.17866 | [PDF](https://arxiv.org/pdf/2601.17866v1)

**作者:** Yoonwoo Jeong `[一作]`, Jaesung Choe `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于点映射（pointmap）的多视角可提示分割框架（MV-SAM），将二维用户提示与三维几何自然对齐，直接利用预训练的SAM图像编码器和轻量级Transformer解码器实现跨视角一致的分割。

**💡 创新点**

创新点包括：①利用点映射的像素‑点一一对应关系，消除渲染/投影步骤；②在统一三维坐标系下为图像特征和提示加入三维位置编码；③仅使用单视角大规模数据训练，避免场景级优化，显著提升跨域泛化；④引入置信度嵌入，提升对低质量点的鲁棒性。

**🔧 技术方法**

技术手段主要包括：点映射预测（如VGGT）、SAM的图像编码器、三维位置编码（sin/cos+Fourier频率）、置信度学习嵌入、轻量级Transformer掩码解码器以及二分类损失（焦点+Dice）。

**📊 数据集**

主要使用的数据集：单视角大型数据集SA‑1B（训练）；多视角/视频基准NVOS、SpIn‑NeRF、ScanNet++、uCo3D、DL3DV（评估）。

**📈 对比分析**

与SAM2‑Video、传统视频/多视图分割方法（NeRF、Gaussian、深度）对比，MV‑SAM在所有基准上均显著优于SAM2‑Video，且在无场景优化的前提下与基于场景优化的算法性能相近。

**⚠️ 局限性**

局限性：性能高度依赖外部视觉几何模型生成的点映射，若点映射深度对齐或结构噪声严重，分割结果会受影响；缺乏显式三维一致性约束，可能在存在离群点或纹理缺失区域时产生不稳定预测。

---

## 484. SynMind: Reducing Semantic Hallucination in fMRI-Based Image Reconstruction

**arXiv ID:** 2601.17857 | [PDF](https://arxiv.org/pdf/2601.17857v1)

**作者:** Lan Yang `[一作]` (Beijing University of Posts and Telecommunications), Yi-Zhe Song `[通讯]` (University of Surrey)

**通讯引用:** 11488 | [OpenAlex ID](https://openalex.org/A5046046128)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种名为SynMind的框架，通过将fMRI信号映射到丰富的句子级语义描述，再与视觉先验结合生成图像，实现了更高的语义一致性和视觉逼真度。

**💡 创新点**

创新点在于将语义信息从传统的粗粒度COCO短句提升为多粒度、句子级的人工智能生成描述，并将其作为主导条件，引入语义感知模块（MimeVis）与Diffusion模型融合，从而显著减少语义错报。

**🔧 技术方法**

核心技术包括多模态大型语言模型（Qwen2‑VL）生成语义文本、软CLIP蒸馏学习、视觉辅助模块、以及基于Stable Diffusion 1.4的条件扩散生成。

**📊 数据集**

实验基于公开的大规模NSD（Natural Scenes Dataset）fMRI‑图像配对数据集，采用“nsdgeneral”ROI进行训练与测试。

**📈 对比分析**

与现有14+种基于fMRI的图像重建方法对比，SynMind在大多数定量指标（如CLIP、Inception、SSIM等）上获得领先，并在两选对比实验中被人类评估者选中率提升超过10%。

**⚠️ 局限性**

局限性包括仍依赖大型预训练模型对语义的准确性，且对非常细粒度的视觉细节恢复仍受限，未来需进一步提升对低级视觉特征的捕获和跨模态对齐。

---

## 485. Scaling Effects and Uncertainty Quantification in Neural Actor Critic Algorithms

**arXiv ID:** 2601.17954 | [PDF](https://arxiv.org/pdf/2601.17954v1)

**作者:** Nikos Georgoudios `[一作]` (Boston University), Justin Sirignano `[通讯]` (University of Oxford)

**通讯引用:** 2583 | [OpenAlex ID](https://openalex.org/A5103033774)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文基于强化学习中的深度网络模型，提出了一种系统性的渐近分析方法，探讨了在不同参数缩放下网络权重、价值函数、策略等的收敛性与误差展开，构造了多阶级展开的极限方程并给出了收敛速度与初始化随机性对误差的影响；

**💡 创新点**

创新点在于首次把强化学习的深度网络参数在大网络宽度极限下的梯度下降过程拆解为多阶级展开，阐明了不同缩放参数β下随机初始化与经验误差的竞争关系，以及如何通过更高阶误差项捕获初始化带来的随机性；

**🔧 技术方法**

采用了流形极限方程、随机逼近理论、梯度理论（如Policy Gradient）、矩阵分析与不动点迭代等数学工具；

**📊 数据集**

实验部分使用了典型的离散MDP数据集（如Forest和Maze等），以及基准的强化学习任务；

**📈 对比分析**

方法通过与传统Actor-Critic以及基于经验的Q学习算法进行对比，评估了收敛速度与最终策略质量，在理论上证明了收敛到贝尔曼解的可控误差，实验上显示更快的收敛与更低的误差；

**⚠️ 局限性**

主要局限在于仅考虑离散有限状态动作空间，且对连续MDP的推广尚未给出；此外，理论分析中对网络结构与激活函数的光滑性假设较强，实际应用需进一步验证。

---

## 486. FlowMorph: Physics-Consistent Self-Supervision for Label-Free Single-Cell Mechanics in Microfluidic Videos

**arXiv ID:** 2601.17947 | [PDF](https://arxiv.org/pdf/2601.17947v1)

**作者:** Bora Yimenicioglu `[一作]` (RareGen), Vishal Manikanden `[通讯]` (Cornell University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理一致的自监督框架FlowMorph，利用低维轮廓参数化与可微“胶囊在流中”模型，从亮视图微流控视频中学习无标注的机械代理k，并通过少量RT‑DC事件进行校准得到Young’s模量。

**💡 创新点**

将流体动力学与弹性恢复嵌入自监督视觉模型，直接从视频中无标注学习机械量，并通过流场、面积守恒、壁面约束等物理约束提升分割与力学估计，显著提高跨设备、跨光学设置的鲁棒性。

**🔧 技术方法**

采用自监督损失（轮廓IoU、光流一致、面积守恒、壁面非渗透、时间平滑）、星形傅里叶轮廓参数化、GRU时间编码器、可微渲染器、UnFlow光流估计与Isotonic回归校准等技术。

**📊 数据集**

使用四个公开RBC微流控数据集：RBCdataset（含流场与通道掩码）、RBC dynamics（中心裁剪、TT/FL标签）、RT‑DC子集（Young’s模量）、Bentley motility dataset（OOD测试）。

**📈 对比分析**

与传统形变指数、Mask R‑CNN+warp、无物理网络等基线对比；在RBCdataset中IoU 0.905、面积保持率 95.7%、壁面违约率 0.32%；在RBC dynamics中k对TT/FL的AUC 0.863、Spearman 0.617；在RT‑DC中MAE 0.118 MPa、Monotonicity 0.8%；跨域迁移时误差增幅不超过0.03。

**⚠️ 局限性**

仅适用于短轨道、低雷诺数的二维流，使用粗粒度弹性能量，k为有效弹性而非真实膜模量；对三维或高速流、遮挡、超平面运动敏感；需少量RT‑DC校准以获得物理单位，对设备变换仍存在一定偏移。

---

## 487. LLM-Based SQL Generation: Prompting, Self-Refinement, and Adaptive Weighted Majority Voting

**arXiv ID:** 2601.17942 | [PDF](https://arxiv.org/pdf/2601.17942v1)

**作者:** Yu-Jie Yang `[一作]` (National Yang Ming Chiao Tung University), Po-An Chen `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 1759 | [OpenAlex ID](https://openalex.org/A5101610714)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于 PET‑SQL 的单代理自我修正与集成投票（SSEV）流水线，并在其基础上设计了多代理协作框架 ReCAPAgent‑SQL，用于解决企业级、复杂、跨语法的 Text‑to‑SQL 问题。

**💡 创新点**

创新点包括：① 将自我修正与加权多数投票（WMA/RWMA）无监督集成；② 在多代理体系中引入专门的规划、检索、批判、语法链接、动作生成、验证等模块，实现了多轮迭代、执行反馈驱动的 SQL 优化；③ 通过动态权重更新实现专家模型的自适应组合，显著提升无标注推理性能。

**🔧 技术方法**

使用技术主要有：大型语言模型（GPT‑4, GPT‑4o, GPT‑4.1‑2025, Gemini‑2.5‑Pro, Qwen‑2.5‑72B 等）进行预/后 SQL 生成；加权多数投票（WMA）与随机加权多数投票（RWMA）进行集成；执行驱动的自我修正循环；语义链接（schema linking）与检索增强生成（RAG）等。

**📊 数据集**

数据集涵盖 Spider 1.0（Dev/Test）、BIRD、Spider 2.0‑lite（前 100 题）等公开基准，覆盖不同语义复杂度、数据库规模与 SQL 方言。

**📈 对比分析**

与单一模型及传统投票方法对比，SSEV 在 Spider 1.0 Dev/Test 上分别达到 85.5%/86.4% 的执行准确率，BIRD Dev 为 66.3%；ReCAPAgent‑SQL 在 Spider 2.0‑lite 前 100 题上实现 31% 的执行准确率（GPT‑4.1+WMA），相比基线提升 25%（从 6%→31%）或 6%（从 23%→29%）。

**⚠️ 局限性**

局限性包括：① 仍对极难、长链推理和大规模模式的处理不足；② 主要评估在 2.0‑lite 前 100 题，未覆盖完整 2.0 数据集；③ 对检索增强与历史修正日志的利用尚不充分；④ 多代理间通信与协议仍依赖自定义实现，缺乏统一标准。

---

## 488. MultiChain Blockchain Data Provenance for Deterministic Stream Processing with Kafka Streams: A Weather Data Case Study

**arXiv ID:** 2601.18011 | [PDF](https://arxiv.org/pdf/2601.18011v1)

**作者:** Niaz Mohammad Ramaki `[一作]` (Technische Universität Berlin), Florian Schintke `[通讯]` (Zuse Institute Berlin)

**通讯引用:** 720 | [OpenAlex ID](https://openalex.org/A5033149837)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Kafka Streams平台上设计并实现了基于区块链的流数据可追溯架构，通过将窗口级数据先做规范化、去重并生成Merkle根，然后将根哈希及元数据信息写入MultiChain区块链，实现流处理结果的审计和可重现性。

**💡 创新点**

创新点在于：①将窗口级数据的完整性以Merkle树形式承诺，仅将小型、不可变的元数据存入链上，避免链上存储大量原始数据；②采用规范化JSON与确定性窗口划分消除运行时非确定性；③在区块链上仅存根哈希与偏移信息，保证轻量化和可扩展性；④提供离链存储与链上锚定的混合模型；⑤实现线性时间的验证流程。

**🔧 技术方法**

技术手段包括Kafka Streams进行实时流处理；MultiChain区块链（使用轮询式挖矿、流式数据结构）做元数据锚定；SHA‑256哈希与Merkle树构造；规范化JSON（Canonical JSON）进行确定性序列化；NDJSON格式离链存储；epoch‑aligned定时窗口；Java 21、JVM实现；以及性能评测与基准测试。

**📊 数据集**

使用德国气象局（DWD）提供的实时气象数据，来源于柏林两座气象站——柏林-勃兰登堡（Berlin-Brandenburg）和柏林-特梅尔霍夫（Berlin-Tempelhof），通过BrightSky API获取。

**📈 对比分析**

通过对比多台机器的重跑结果验证了可重现性；对窗口验证过程做线性回归，表明验证时延与记录数呈线性关系；吞吐量保持在高水平；区块链层的交易速率因检查点写入频率低而较低，但在批量写入时可达约2.1 tx/s；整体性能在实时流处理场景下满足可接受范围。

**⚠️ 局限性**

局限性包括：①Merkle根仅能证明窗口整体完整性，无法细粒度追踪多步骤复杂管道的内部变换；②依赖于外部离链存储，若存储失效需额外恢复机制；③区块链侧目前无智能合约或链码支持更细粒度的审计逻辑；④未使用零知识证明，敏感数据在审计时仍会暴露；⑤区块链TPS受轮询式共识限制，吞吐上限在大规模写入场景下受到影响。

---

## 489. Strip-Fusion: Spatiotemporal Fusion for Multispectral Pedestrian Detection

**arXiv ID:** 2601.18008 | [PDF](https://arxiv.org/pdf/2601.18008v1)

**作者:** Asiegbu Miracle Kanu-Asiegbu `[一作]` (University of Michigan), Xiaoxiao Du `[通讯]` (University of Michigan)

**通讯引用:** 480 | [OpenAlex ID](https://openalex.org/A5076328887)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于时空融合的多光谱行人检测框架Strip-Fusion，用时序自适应卷积和新型KL散度损失实现可对齐、鲁棒的检测；

**💡 创新点**

创新点包括：时空自适应卷积实现隐式时序建模；基于Strip-MLP的时空融合模块；KL散度损失用于调节可见与热红外模态不平衡；以及跨模态后处理算法降低误检；

**🔧 技术方法**

技术主要包括：深度学习卷积网络、TAdaConv、Strip-MLP、KL散度损失、后处理算法、YOLOv5预训练；

**📊 数据集**

使用KAIST和CVC-14两个多光谱行人检测基准数据集；

**📈 对比分析**

与现有AR-CNN、MambaST、MS-DETR等方法对比，Strip-Fusion在KAIST和CVC-14的“Reasonable”设置下均取得或领先的MR（误检率），尤其在高遮挡和空间错位场景表现突出；

**⚠️ 局限性**

局限性在于：对长时序窗口的性能提升不稳定；对极小或极大尺寸行人仍存在误检；对计算效率的进一步优化仍需探索。

---

## 490. MorphXAI: An Explainable Framework for Morphological Analysis of Parasites in Blood Smear Images

**arXiv ID:** 2601.18001 | [PDF](https://arxiv.org/pdf/2601.18001v1)

**作者:** Aqsa Yousaf `[一作]` (University of Texas at Arlington), Habeeb Olufowobi `[通讯]` (University of Texas at Arlington)

**通讯引用:** 707 | [OpenAlex ID](https://openalex.org/A5071036966)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MorphXAI框架，实现了血液涂片中寄生虫的检测与可解释的形态学特征预测；

**💡 创新点**

创新点在于将形态学属性预测融入检测网络的多任务学习，直接输出可被临床医生解读的结构化解释，而非仅靠后置热力图；

**🔧 技术方法**

采用基于Transformer的RT‑DETRv3改进版，加入形态学解码器、多层监督与联合损失；

**📊 数据集**

使用了1,818张高分辨率血涂片图像，标注了5,903个Leishmania、Trypanosoma brucei与Trypanosoma cruzi实例及形态学属性；

**📈 对比分析**

在验证集上与RT‑DETRv3比较，MorphXAI在AP^(.5:.95)上略高（48.2% vs 47.3%），并在形态属性上实现88.6%–96.6%的准确率，推理速度为42.7 FPS；

**⚠️ 局限性**

局限在于数据集规模有限、仅覆盖三种寄生虫，形态属性数量受限，且更深层网络易过拟合；

---

## 491. Domain-Expert-Guided Hybrid Mixture-of-Experts for Medical AI: Integrating Data-Driven Learning with Clinical Priors

**arXiv ID:** 2601.17977 | [PDF](https://arxiv.org/pdf/2601.17977v1)

**作者:** Jinchen Gu `[一作]`, Lu Zhang `[通讯]` (Indiana University)

**通讯引用:** 2494 | [OpenAlex ID](https://openalex.org/A5022297444)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种将数据驱动专家与临床专家知识（眼动热图）融合的混合Mixture‑of‑Experts模块DKGH‑MoE，用于医学影像分类。

**💡 创新点**

创新点在于将临床专家眼动模式直接作为路由依据，与传统数据驱动MoE并行并通过可插拔融合门控实现性能提升和可解释性增强。

**🔧 技术方法**

使用技术包括Mixture‑of‑Experts架构、轻量MLP路由网络、soft‑max加权组合、门控融合、负载均衡正则、卷积提取眼动热图特征，以及在ResNet‑18/50骨干上的实现。

**📊 数据集**

使用数据集INBreast（410张乳腺钼靶图像），配有BI‑RADS标签和对应的眼动热图。

**📈 对比分析**

采用5折受试者级交叉验证，在dense与sparse路由设置下与基线ResNet、单一DD‑MoE、DE‑MoE比较，DKGH‑MoE在ResNet‑18上ACC 77.79%/AUC 82.91%，在ResNet‑50上ACC 77.23%/AUC 84.01%，显著优于其它方法。

**⚠️ 局限性**

局限性包括：单一任务验证，未在其他医学影像任务测试；眼动数据噪声可能影响路由；在更深网络或大规模数据时单一MoE易过拟合，需进一步研究。

---

## 492. Credit Fairness: Online Fairness In Shared Resource Pools

**arXiv ID:** 2601.17944 | [PDF](https://arxiv.org/pdf/2601.17944v1)

**作者:** Seyed Majid Zahedi `[一作]` (University of Waterloo), Rupert Freeman `[通讯]` (University of Virginia)

**通讯引用:** 858 | [OpenAlex ID](https://openalex.org/A5010895600)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究共享资源池中的在线分配机制，提出“信用公平”概念并设计满足信用公平与帕累托效率的机制。

**💡 创新点**

引入信用公平属性，证明无法同时满足信用公平、帕累托效率和策略不变性，提出新的机制满足信用公平与帕累托效率，并证明其在线策略不变性。

**🔧 技术方法**

利用基于比例共享与约束的算法（Divvy）实现分配，并通过信用系统与优先级调度实现公平。

**📊 数据集**

采用Google集群的CPU需求跟踪数据（12.5k机器，1个月）。

**📈 对比分析**

与SMMF、DMMF、Karma在多项公平与效率指标下进行对比，实验显示新机制在信用公平、无共享伤害、均衡性方面表现最佳，近似最优社会福利。

**⚠️ 局限性**

仅适用于单一资源、线性效用、固定代理集，且无法同时兼顾信用公平与策略不变性，需进一步扩展以适应更一般场景。

---

## 493. "Lighting The Way For Those Not Here": How Technology Researchers Can Help Fight the Missing and Murdered Indigenous Relatives (MMIR) Crisis

**arXiv ID:** 2601.17966 | [PDF](https://arxiv.org/pdf/2601.17966v1)

**作者:** Naman Gupta `[一作]` (University of Wisconsin Madison), Rahul Chatterjee `[通讯]` (University of Wisconsin Madison)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过基于 Google 搜索的爬虫收集了约 123,000 条公开网页，随后使用 LLM 辅助的筛选、分类与编码流程，精炼至约 1,200 条与技术使用相关的网页，系统性地分析了土著社区在寻找失踪/被谋杀亲属时所面临的社会技术障碍，以及他们为求生存、治愈和提升认知所采用的技术行动。整个研究采用了德殖民女权主义的研究方法，强调与土著社区的合作、尊重与自决。

**💡 创新点**

创新点包括：
1) 将大规模 Web 数据与 LLM 辅助流程相结合，构建了可复制的土著中心化技术研究范式；
2) 在方法学上首次从土著知识体系出发，结合叙事与故事化分析，避免了传统数据驱动方法的殖民倾向；
3) 公开了包含土著视角的网页数据集，为后续 HCI 与社交媒体安全研究提供了资源；
4) 以土著自决为核心，提出了技术与政策层面的改进建议，强调数据主权与系统透明。

**🔧 技术方法**

技术手段：
- Web 爬虫：Selenium + BeautifulSoup4；
- LLM 处理：本地部署的 Llama‑3.1‑8b（域/内容分类）和 Qwen‑2.5‑14b（技术类别判定）；
- 统计与可视化：Python（pandas、matplotlib）用于页面分布与领域分层抽样；
- 手工编码：基于 Zotero 的标注系统，结合土著叙事方法论。

**📊 数据集**

数据集：
- 公开网页数据集，包含 123,000 条 Google 搜索结果，其中 1,200 条被筛选为技术相关；
- 以土著部落名称与关键词（如 "MMIR"、"Missing and Murdered Indigenous Women" 等）构造的 1,200 条页面列表；
- 数据集已在 GitHub 上开源，包含 URL、摘要、技术类别、引用句子与手工标注标签，方便社区与研究者使用。

**📈 对比分析**

对比方法：本文并未与其他模型或方法做传统的准确率/召回率比较，而是通过主题饱和度（thematic saturation）来确定分析的充分性。性能方面：
- Llama‑3.1‑8b 在域分类上达 97% 准确率；
- Qwen‑2.5‑14b 在技术类别判定上对 500 条样本达到 88% 正确率；
- 通过人工复核，最终 116 条页面被认定为真正讨论技术使用的核心内容。

**⚠️ 局限性**

局限性：
- 依赖 Google 搜索索引，可能遗漏未被收录或被算法屏蔽的土著内容；
- 仅分析英文网页，忽略土著原语与多语种表达；
- LLM 受训练语料偏见影响，尤其在涉及暴力与敏感议题时可能产生自我审查；
- 仅采样约 1,200 条页面，无法覆盖全部相关技术和案例；
- 未涉及原始访谈或调查，难以验证在线叙事与真实情境的一致性；
- 研究周期限定在 2024‑2025 年，未涵盖 2026 年以后的新技术与政策变化。

---

## 494. A Monosemantic Attribution Framework for Stable Interpretability in Clinical Neuroscience Large Language Models

**arXiv ID:** 2601.17952 | [PDF](https://arxiv.org/pdf/2601.17952v1)

**作者:** Michail Mamalakis `[一作]` (University of Cambridge), Pietro Lio `[通讯]` (University of Cambridge)

**通讯引用:** 33851 | [OpenAlex ID](https://openalex.org/A5056748708)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种统一的解释框架，将稀疏自编码器（SAE）产生的单义特征空间与多种归因方法相结合，并通过Transformer/扩散网络优化归因结果，生成在临床阿尔茨海默病诊断中的稳定、可解释特征权重。

**💡 创新点**

创新点在于：①将机制解释（SAE的单义表征）与归因解释（多种梯度、SHAP等）融合，突破了两者的互补壁垒；②引入线性UMAP约束和解释优化器（TEO/DEO）提升解释的一致性与稀疏性；③通过对齐多维特征与Token级归因，实现从内部表示到输入级解释的双层归因。

**🔧 技术方法**

技术包括：稀疏自编码器（TopK、JumpReLU、Gated‑SAE）、六种归因方法（Feature Ablation、Layer Activation、Conductance、Grad‑SHAP、Integrated Gradients、Grad×Activation）、Transformer/扩散网络解释优化器、UMAP+PCA低维可视化与线性约束、RIS/ROS稀疏性指标评估。

**📊 数据集**

数据集：内部分布（IID）采用ADNI多模态生成文本；外部分布（OOD）采用BrainLAT多模态生成文本；分类任务包括二分类（Control vs AD）和三分类（Control/EMCI/LMCI或Control/FTD/AD）。

**📈 对比分析**

与传统归因方法相比，加入SAE后RIS/ROS显著下降，稀疏性提升；TEO‑SAE在IID、OOD下均获得最佳稳定性；TEO‑UMAP‑SAE在稀疏性与稳定性间提供可调权衡。总体而言，统一框架在多任务、多域下的解释质量均优于单一归因方法，且对分布漂移具有更强鲁棒性。

**⚠️ 局限性**

局限包括：①对不同模型和层级的泛化尚未系统评估；②解释优化器需要额外训练成本；③在三分类任务中部分类别缺乏配对样本导致统计检验受限；④最终的临床验证仍缺乏医生评估与真实病历对应。

---

## 495. "Label from Somewhere": Reflexive Annotating for Situated AI Alignment

**arXiv ID:** 2601.17937 | [PDF](https://arxiv.org/pdf/2601.17937v1)

**作者:** Anne Arzberger `[一作]` (Delft University of Technology), Jie Yang `[通讯]` (Delft University of Technology)

**通讯引用:** 22215 | [OpenAlex ID](https://openalex.org/A5100404947)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实验了“反思式注释”工具，让众包工作者在标注公平性时反思自身社会定位，并收集其主观判断与元数据。

**💡 创新点**

创新点在于将反思性实践引入注释流程，捕捉动态、交叉身份的情境化元数据，提供比静态人口统计更丰富的注释背景。

**🔧 技术方法**

采用了Miro低保真原型、React前端与服务器、Prolific众包平台，以及社会身份地图进行交互设计，并用反思性主题分析处理文本。

**📊 数据集**

数据来源为30名工人在Prolific上完成的公平性标注任务，包含两段文本（招聘广告与美容广告），以及对应的反思性阐述。

**📈 对比分析**

本文未给出传统意义上的性能指标，而是通过定性对比展示：反思式注释提升了注释的深度、身份意识、以及对不确定性的表达，但未实现量化性能提升。

**⚠️ 局限性**

局限包括样本规模与语言单一、可能的逃避与情绪负担、对身份认知的简化处理、隐私风险与可扩展性问题。

---

## 496. AI-based approach to burnout identification from textual data

**arXiv ID:** 2601.17993 | [PDF](https://arxiv.org/pdf/2601.17993v1)

**作者:** Marina Zavertiaeva `[一作]` (National Research University Higher School of Economics), Anastasiia Kibardina `[通讯]` (National Research University Higher School of Economics)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5115738726)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了一种基于RuBERT的NLP模型，用来从俄语文本中检测倦怠。

**💡 创新点**

创新点在于将情感分析模型RuBERT迁移到倦怠检测，结合合成与真实评论两种数据，并给出可解释的标注协议。

**🔧 技术方法**

使用了Transformer架构的RuBERT、ChatGPT 3.5 Turbo生成合成句子、以及GPT辅助标注的YouTube评论。

**📊 数据集**

数据集由约3.3万条ChatGPT合成句子、7,888条含倦怠的YouTube评论和8,505条非倦怠评论组成，总计18,395条样本。

**📈 对比分析**

在5个训练轮次下，模型在内部验证集上取得准确率0.94、精确率0.931、召回率0.944、F1 0.937、AUC‑ROC 0.98，显示出高效识别性能。

**⚠️ 局限性**

主要局限是缺乏外部数据验证、标注噪声可能影响结果以及模型对不同群体和语境的泛化能力尚待评估。

---

## 497. A cartesian closed fibration of higher-order regular languages

**arXiv ID:** 2601.18000 | [PDF](https://arxiv.org/pdf/2601.18000v1)

**作者:** Paul-André Melliès `[一作]` (CNRS, Université Paris Cité, INRIA), Vincent Moreau `[通讯]` (Tallinn University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文证明了高阶正则语言对乘积型、箭头型以及 λ‑项的逆像保持闭合，并构造了一个笛卡尔闭 fibration 以捕捉这些闭包性质，进而推广了 Brzozowski 导数。

**💡 创新点**

创新点在于首次给出高阶正则语言的笛卡尔闭 fibrations 的两种构造（fibrational 与拓扑的）以及利用它们得到高阶 Brzozowski 导数的通用形式。

**🔧 技术方法**

采用了 fibration / bifibration 的范畴论工具、Stone duality 与 profinite λ‑calculus、反射子句与 Frobenius 递推、以及 Hermida 的开拓证明技术。

**📊 数据集**

无实验数据集，论文为纯理论性研究。

**📈 对比分析**

无实验比较，文章主要通过形式证明展示了闭包性质与导数的正确性，并未涉及性能评估。

**⚠️ 局限性**

局限性在于对 "open" λ‑项的完整分类仍未完成，且目前的框架不涵盖所有可能的开闭性质，未来需进一步探索更广泛的应用与实现。

---

## 498. The Most Important Laboratory for Social Scientific and Computing Research in History

**arXiv ID:** 2601.17998 | [PDF](https://arxiv.org/pdf/2601.17998v1)

**作者:** Benjamin Mako Hill `[一作]` (Massachusetts Institute of Technology), Aaron Shaw `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了过去二十年关于维基百科的学术研究，梳理主要研究主题与发现，指出其在社会科学与计算研究中的重要性。

**💡 创新点**

创新点在于建立并系统化“维基媒体研究”年度演讲框架，将海量文献筛选为5-8个主题，并提出未来研究方向和跨学科视角。

**🔧 技术方法**

使用的技术主要是系统综述与文献检索（Google Scholar、会议工作坊、邮件列表）结合质性分析与定量指标评估，如错误率、质量评分、参与度相关性。

**📊 数据集**

数据集包括维基百科原始文本、编辑日志、浏览量数据、Wikidata结构化数据，以及各语言维基社区的多元化数据源。

**📈 对比分析**

比较方法采用对比分析（如维基百科与专业百科、新闻、法律引用等）和因果推断（实验、随机控制试验）来评估内容质量、传播效应和参与度，结果表明维基百科在多领域可达性与质量上表现优异。

**⚠️ 局限性**

局限性包括样本偏倚（主要聚焦英语版）、研究覆盖面不足（对非主流语言社区了解有限）、缺乏纵向因果研究以及对维基百科治理机制的深层机制分析不够充分。

---

## 499. Gradual Generation of User Interfaces as a Design Method for Malleable Software

**arXiv ID:** 2601.17975 | [PDF](https://arxiv.org/pdf/2601.17975v1)

**作者:** Bryan Min `[一作]` (University of California San Diego), Haijun Xia `[通讯]` (University of California San Diego)

**通讯引用:** 2208 | [OpenAlex ID](https://openalex.org/A5016819583)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种名为 Gradual Generation 的 UI 设计方法，利用生成式 AI 在界面生成过程中逐步加载并展示一系列中间 UI 层，以帮助用户发现并调节多样化的自定义选项。

**💡 创新点**

创新点在于将自定义功能拆解成可视化的“中间层”，每层暴露不同维度的定制项，并通过“回溯”机制让用户在生成过程中随时调整，既保持了 GenUI 的表达能力，又提升了可发现性与可视化简洁度。

**🔧 技术方法**

方法基于生成式 UI（GenUI）框架，使用自然语言、JSON 方案、代码块和 CSS 变量等多种规范作为中间层的参数输入，结合 AI 模型（如 GPT、Stable Diffusion 等）生成对应的界面。

**📊 数据集**

本文并未使用公开数据集，而是通过构建三个原型网站（视频首页、团队日程日历、课程管理系统）来演示该方法的可行性和适用场景。

**📈 对比分析**

在比较方面，作者主要通过可视化展示和案例对比来说明 Gradual Generation 在保持界面简洁度的同时增加了自定义可发现性，没有给出定量性能指标，但通过三例原型的使用演示表明该方法在可用性上有显著提升。

**⚠️ 局限性**

局限性包括：缺乏大规模用户实验验证自定义发现效率；中间层的设计依赖设计师的经验，缺乏统一规范；若自定义维度过多，可能导致界面复杂度再次上升；并且未讨论与现有 UI 设计工具的整合细节。

---

## 500. UPLiFT: Efficient Pixel-Dense Feature Upsampling with Local Attenders

**arXiv ID:** 2601.17950 | [PDF](https://arxiv.org/pdf/2601.17950v1)

**作者:** Matthew Walmer `[一作]` (University of Maryland), Abhinav Shrivastava `[通讯]` (University of Maryland)

**通讯引用:** 7433 | [OpenAlex ID](https://openalex.org/A5101614443)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于迭代卷积的轻量级特征上采样框架UPLiFT，结合局部注意力（Local Attender），实现从低分辨率预训练视觉特征到像素级高分辨率特征的转换。

**💡 创新点**

关键创新是将局部注意力取代传统跨注意力，保持语义一致性的同时仅线性扩展，避免跨注意力的二次方复杂度。

**🔧 技术方法**

采用卷积上采样、Local Attender、深度自监督训练损失、多步上采样、ViT等技术。

**📊 数据集**

在ImageNet‑1K进行自监督训练，并在COCO‑Stuff、Pascal VOC、ADE20k、Cityscapes、COCO、FacesHQ、LHQ等数据集验证；对VAE latent上采样使用Unsplash‑Lite。

**📈 对比分析**

与LiFT、Featup、LoftUp、JAFAR、AnyUp等基线以及双线性/最近邻对比，UPLiFT在语义分割、深度估计等密集预测任务中取得最高mIoU和最快推理速度；在VAE上采样生成、超分任务中与CFM相当甚至更优，并且延迟显著降低。

**⚠️ 局限性**

主要局限是需要多步迭代，灵活性低于跨注意力方法；对极大分辨率或不同特征尺度的适应性待进一步研究。

---

## 501. Designing AI Peers for Collaborative Mathematical Problem Solving with Middle School Students: A Participatory Design Study

**arXiv ID:** 2601.17962 | [PDF](https://arxiv.org/pdf/2601.17962v1)

**作者:** Wenhan Lyu `[一作]` (William and Mary), Yixuan Zhang `[通讯]` (William and Mary)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一所大学暑期营中，研究团队通过技术探针让24名六至八年级学生与两位AI“同伴”共同完成数学协同问题求解，并通过可视化贴纸、讨论、问卷等方式收集学生对AI伙伴的感知与设计需求；随后基于学生产出的原型及偏好调查，提出了以“循序渐进、工具辅助的支架化”与“可调节的伙伴人格”为核心的AI同伴设计规范与实现建议。

**💡 创新点**

创新点在于：①首次从学生视角进行参与式设计，获取了真实的“AI同伴”使用偏好；②将LLM嵌入多代理协同框架，探究在中学生数学协作情境下的角色与行为细节；③提出“诊断‑可视化‑提示”三阶段支架模型和“可配置的伙伴人格”两大设计原则，为未来的AI协同学习系统提供可操作的规范。

**🔧 技术方法**

技术主要包括：①基于GPT‑4o的大语言模型；②采用Gentopia框架实现对话管理、个体与对话层的模拟；③整合Desmos、词典等工具以实现可视化支架；④使用Python/TypeScript实现前端聊天界面和后端服务。

**📊 数据集**

数据集为：学校标准化数学题库（符合Virginia SOL）和学生生成的贴纸/脚本原型；未使用公开机器学习数据集；评估主要基于NASA‑TLX、GEQ问卷、学生偏好调查以及录音文本的定性编码。

**📈 对比分析**

方法对比：将人类仅协作与包含AI同伴的协作两种情境做配对比较；使用NASA‑TLX和GEQ进行量化评估；发现AI同伴显著降低时间压力，但增加操作负担和社交凝聚度；定性分析揭示四大张力点（效率 vs 操作负荷、教学 vs 合作、社交规范、情感支架）。整体并未给出传统算法性能指标，重点在协作体验与学生主观感受。

**⚠️ 局限性**

局限性包括：①样本量小（24人）且来自单一城市暑期营，缺乏普适性；②仅对两位AI同伴进行评估，未与单一AI导师或无AI情境比较；③探针设计可能对学生期望产生影响；④仅覆盖中学数学领域，难以推广到其他学科或年龄段；⑤未对AI模型的准确性或安全性进行系统评估。

---

## 502. Types for Grassroots Logic Programs

**arXiv ID:** 2601.17957 | [PDF](https://arxiv.org/pdf/2601.17957v1)

**作者:** Ehud Shapiro `[一作]` `[通讯]` (London School of Economics and Weizmann Institute of Science), Ehud Shapiro (London School of Economics and Weizmann Institute of Science)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了Typed GLP语言的类型系统，并证明其与语义的一致性。

**💡 创新点**

创新点在于将LP类型扩展为带模式注解的正则路径类型，实现单读单写变量与未来/承诺语义的结合，并引入子类型以支持安全交互。

**🔧 技术方法**

采用了正则路径类型、BNF定义、自动机、子类型推理、类型自动机、LP语义、编译器实现等技术。

**📊 数据集**

未使用数据集，主要基于理论证明与示例程序。

**📈 对比分析**

未进行性能对比，仅通过理论证明与案例展示其可行性。

**⚠️ 局限性**

局限在于缺乏多态、模块化支持，以及未评估大规模并发程序的实现与性能。

---

## 503. "I use ChatGPT to humanize my words": Affordances and Risks of ChatGPT to Autistic Users

**arXiv ID:** 2601.17946 | [PDF](https://arxiv.org/pdf/2601.17946v1)

**作者:** Renkai Ma `[一作]` (University of Cincinnati), Lingyao Li `[通讯]` (University of South Florida)

**通讯引用:** 1157 | [OpenAlex ID](https://openalex.org/A5031522503)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对 3,984 条 Reddit、X、Tumblr 上自认自闭症用户讨论 ChatGPT 的社交媒体帖子进行归纳主题分析，探究 ChatGPT 对自闭症用户的可用性（如执行功能卸载、情绪调节、跨神经差异沟通翻译和身份验证）以及潜在风险（如强化妄想、身份被自动掩蔽、与自闭症公正感冲突），并基于 Technology Affordance 框架提出未来神经包容性设计的改进方向。

**💡 创新点**

① 将 Technology Affordance 与大规模质性数据相结合，首次在同一研究中同时系统性阐述自闭症用户与 LLM 交互的收益与风险；② 通过对收益与风险的双向梳理揭示了三大权衡点（执行力依赖、自动掩蔽、情绪验证），为设计带来“有益摩擦”和双向翻译等新概念；③ 针对 LLM 交互提出了“认知强制功能（CFF）”与“解释式 AI”等可落地设计原则。

**🔧 技术方法**

① LLM 辅助的内容过滤管道（使用 GPT‑4o‑mini 与 Chain‑of‑Thought 进行 ChatGPT 与自闭症相关性筛选）；② 归纳主题分析（Google Sheets 编码、Affinity Diagram 归类、交叉检查）；③ 统计可靠性评估（α=1.00 与 α=0.91 的互评一致性）。

**📊 数据集**

从 Brandwatch 系统检索到的 3,984 条 Reddit、X、Tumblr 社交媒体帖子，时间范围为 2023‑01‑01 至 2025‑09‑30，关键词为 “ChatGPT” 加上自闭症相关术语（DSM‑5 术语与 SMHD 数据集）。

**📈 对比分析**

本研究采用质性方法，没有传统的模型对比或性能指标；其方法论亮点在于实现高一致性筛选（α=1.00）与对 239 个可用性代码、50 个风险代码的系统化归纳；因此性能指标主要体现在数据处理的可靠性与主题深度，而非数值化的模型精度。

**⚠️ 局限性**

① 样本仅来自公开社交媒体，存在自我选择与平台偏差；② 仅聚焦 ChatGPT，无法推广至其他 LLM；③ 研究基于自我报告，缺乏实验验证和因果推断；④ 主要以英文数据为主，跨文化适用性不明；⑤ 未对模型安全性或偏见进行技术评估，导致对风险评估的深度受限。

---

## 504. Late Breaking Results: Boosting Efficient Dual-Issue Execution on Lightweight RISC-V Cores

**arXiv ID:** 2601.17940 | [PDF](https://arxiv.org/pdf/2601.17940v1)

**作者:** Luca Colagrande `[一作]` (ETH Zurich), Luca Benini `[通讯]` (ETH Zurich)

**通讯引用:** 55671 | [OpenAlex ID](https://openalex.org/A5043408422)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对轻量级RISC‑V核心 Snitch，提出了 COPIFTv2 双指令发射执行方法，利用两条轻量 FIFO 队列实现整数核与浮点协处理器之间的直接细粒度通信与同步。

**💡 创新点**

创新点在于用轻量队列替代原 COPIFT 的内存溢出与批处理，消除软件流水线与多缓冲的复杂性，显著提升 IPC 与能效，并将方法简化至编译器可自动化的形式。

**🔧 技术方法**

采用 Snitch 微架构、CSR 定制指令、轻量 FIFO 队列、基于数据流图的指令分区与调度以及硬件循环技术，实现整数与浮点线程的并行执行。

**📊 数据集**

实验使用公开的混合整数/浮点基准程序集合（参见文中引用的基准），在 12LP+ FinFET 工艺下进行周期级 RTL 仿真与功耗估算。

**📈 对比分析**

与单指令发射基线和原 COPIFT 方法对比，COPIFTv2 在 IPC 上最高可提升 X 倍，平均提升 X 倍；能耗保持相近，能效最高提升 X 倍，平均提升 X 倍。

**⚠️ 局限性**

仍受限于对整数与浮点独立批处理的支持，且需要编译器辅助转换，尚未实现完全无人工调优的通用双指令发射方案，且对跨线程依赖的自动识别尚不完善。

---

## 505. DTC: A Deformable Transposed Convolution Module for Medical Image Segmentation

**arXiv ID:** 2601.17939 | [PDF](https://arxiv.org/pdf/2601.17939v1)

**作者:** Chengkun Sun `[一作]` (University of Florida), Jie Xu `[通讯]` (University of Florida)

**通讯引用:** 10700 | [OpenAlex ID](https://openalex.org/A5063914161)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种可学习的可变形转置卷积模块（DTC），用于改进医学图像分割模型的上采样过程；

**💡 创新点**

创新点在于将可变形卷积的坐标学习机制嵌入到转置卷积中，既保留了通道调整能力，又实现了动态、空间自适应的上采样；

**🔧 技术方法**

采用了可变形卷积（DCN）框架、grid_sample插值、可调可变 receptive field 参数 λ 等技术，构建了可变形转置卷积模块；

**📊 数据集**

使用了ISIC18（皮肤病变）、BUSI（乳腺超声）二维数据集以及BTCV15（三维CT血管）数据集进行评测；

**📈 对比分析**

将 DTC 与 UNet、SwinUNETR V2、SegMamba、nnUNet、UNETR、nnMamba 等模型以及传统的线性插值、转置卷积、Dysample、FADE 等上采样方法进行对比，实验表明 DTC 在 Dice、NSD 等指标上均有提升，尤其在二分类任务中提升幅度较大；

**⚠️ 局限性**

局限性包括：对 receptive field 超大时性能下降，需调参；在多分类分割中不同器官的提升不均衡；对 3D 中 Dysample 的结合效果未评估；计算开销虽不大，但仍需进一步优化。

---

## 506. Dissipative Learning: A Framework for Viable Adaptive Systems

**arXiv ID:** 2601.17933 | [PDF](https://arxiv.org/pdf/2601.17933v1)

**作者:** Laurent Caraffa `[一作]` `[通讯]` (Universite Gustave Eiffel), Laurent Caraffa (Universite Gustave Eiffel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出 BEDS（Bayesian Emergent Dissipative Structures）框架，将机器学习视为在有限资源下的耗散过程，阐明遗忘、正则化和稳定性是不可或缺的物理机制，并给出条件最优性定理：Fisher–Rao 距离是唯一的热力学最优正则化方式。

**💡 创新点**

创新点在于：
1) 将学习过程映射为可持续的耗散结构，并用信息几何与热力学原理统一解释传统正则化、Dropout、权重衰减等经验技巧；
2) 在假设 A1‑A3 下推导出条件最优性定理，证明 Fisher–Rao 正则化是唯一最优；
3) 设计了四维 BEDS 状态空间 (μ, τ, φ, κ)，并给出能量‑精度下界，形成完整的理论参考框架。

**🔧 技术方法**

采用的技术与理论：信息几何（Fisher–Rao 度量）、Prigogine 的耗散结构理论、Landauer 原理、最大熵推理、指数族几何、自然梯度等。

**📊 数据集**

论文未在具体数据集上进行实验评估；主要基于理论推导与对现有算法（如 LeJEPA、DINO、SIGReg、EMA 等）的概念性映射。

**📈 对比分析**

方法比较采用理论分析：通过能量‑精度图展示不同正则化技术相对 BEDS 最优轨迹的位置；并讨论欧氏正则化的结构性次优性。未给出实验性能指标。

**⚠️ 局限性**

局限性：
- 结论完全依赖假设 A1‑A3，若假设不成立则不适用；
- 论文缺乏对实际算法的数值验证，无法直接评估在真实训练中的效果；
- 未考虑硬件实现细节与离散化误差，主要是理论抽象。

---

## 507. Distances Between Top-Truncated Elections of Different Sizes

**arXiv ID:** 2601.17931 | [PDF](https://arxiv.org/pdf/2601.17931v1)

**作者:** Piotr Faliszewski `[一作]` (AGH University), Tomasz Wąs `[通讯]` (University of Oxford)

**通讯引用:** 316 | [OpenAlex ID](https://openalex.org/A5083795172)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了可处理不同规模与部分排名投票的选举映射方法，生成Preflib数据的可视化地图

**💡 创新点**

设计了位置扩展与DAP特征距离，克服了传统距离在候选人数不等和不完整投票上的限制

**🔧 技术方法**

采用频率矩阵、Wasserstein距离、特征函数（多样性、同意、极化）以及多维尺度/KK嵌入技术

**📊 数据集**

使用Preflib真实选举以及基于IC、Mallows、Urn、Euclidean等模型生成的合成选举，覆盖从3到约2600候选人、4到约64000选民的数据集

**📈 对比分析**

通过在不同规模与截断水平下计算平均距离并绘制地图，评估位置扩展与DAP的相似度，DAP在多数情形下更贴近原始结构，距离均低于5%直径（截断）或10%直径（规模差）

**⚠️ 局限性**

DAP不满足所有一致性公理，对随机丢弃截断方式表现不佳；位置扩展虽然一致但对截断敏感；两种距离在某些模型（如Mallows）下表现不够平稳

---

## 508. Post-Training Denoising of User Profiles with LLMs in Collaborative Filtering Recommendation

**arXiv ID:** 2601.18009 | [PDF](https://arxiv.org/pdf/2601.18009v1)

**作者:** Ervin Dervishaj `[一作]` (University of Copenhagen), Christina Lioma `[通讯]` (University of Copenhagen)

**通讯引用:** 2752 | [OpenAlex ID](https://openalex.org/A5045425016)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对已训练好的协同过滤推荐模型，使用大语言模型在推理阶段对用户交互记录进行后训练去噪，即通过LLM提示删除不利于候选物品排序的交互项，以提升推荐效果。

**💡 创新点**

首次提出仅在推理时利用LLM对用户资料进行去噪，无需重新训练推荐模型或对LLM进行微调；方法通过提示工程直接修改输入，兼顾效率与数据需求。

**🔧 技术方法**

使用多种大型语言模型（Qwen 3、Mistral NeMo、GPT‑4.1‑mini、DeepSeek‑V3.1）配合MultiVAE协同过滤模型；采用零样本、含推荐榜单或去噪示例的few‑shot提示。

**📊 数据集**

在三大公开推荐数据集上进行评估：MovieLens 1M、Yelp、Amazon CDs & Vinyl。

**📈 对比分析**

与随机去噪、最受欢迎去噪、语义相似度去噪以及验证集上上界（upper‑bound）等基线对比，使用NDCG@10/20/100、Hit Rate、MRR等指标。结果显示LLM去噪平均提升NDCG约2–4%，部分模型（如Qwen few‑shot‑2）接近上界，整体最高可提升13%（覆盖约50%用户）。

**⚠️ 局限性**

局限性包括：候选物品本身可能为噪声，影响去噪效果；LLM输出不易解释且易出现格式错误与hallucination，尤其在某些数据集；对长交互记录的适用性有限；依赖预训练LLM的知识与提示设计。

---

## 509. Coding-Enforced Resilient and Secure Aggregation for Hierarchical Federated Learning

**arXiv ID:** 2601.17995 | [PDF](https://arxiv.org/pdf/2601.17995v1)

**作者:** Shudi Weng `[一作]` (KTH Royal Institute of Technology), Mikael Skoglund `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 8707 | [OpenAlex ID](https://openalex.org/A5041348422)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于编码强制安全聚合的层级联邦学习框架 H‑SecCoGC，能够在不可靠通信和任意强度的本地差分隐私条件下实现全局模型的精确恢复。

**💡 创新点**

创新点在于：①将梯度共享编码（CoGC）与循环梯度码相结合，利用编码结构实现秘密键完全抵消；②在层级网络中提供对失效链路的鲁棒性；③给出完整的局部差分隐私分析，兼顾噪声相关性与不可靠链路。

**🔧 技术方法**

使用技术包括：循环梯度码、梯度共享编码、编码强制安全聚合、秘密键构造、局部差分隐私、Bernstein 误差界、概率分析与仿真。

**📊 数据集**

实验采用 CIFAR‑10 图像分类数据集，使用一个多层卷积神经网络（CNN）进行训练。

**📈 对比分析**

与无私有层级 FL、私有相关噪声层级 FL、理想完好网络等基线方法比较，实验表明 H‑SecCoGC 在对称和非对称网络、不同隐私噪声水平下均能保持与理想网络相同的学习准确率，尤其在强隐私噪声下仍能收敛，显著优于其他方法。

**⚠️ 局限性**

局限性：未深入讨论通信/计算开销、密钥协商复杂度、对大规模高维模型的适应性，以及对动态网络拓扑的进一步自适应能力。

---

## 510. NeuroManip: Prosthetic Hand Manipulation System Based on EMG and Eye Tracking Powered by the Neuromorphic Processor AltAi

**arXiv ID:** 2601.17991 | [PDF](https://arxiv.org/pdf/2601.17991v1)

**作者:** Roman Akinshin `[一作]` (Skolkovo Institute of Science and Technology), Valerii Kangler `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种将表面肌电（sEMG）与注视引导的计算机视觉结合的神经形态控制体系，用于上肢义肢的安全、高效操控；

**💡 创新点**

创新点在于：①将EMG识别模型迁移为脉冲神经网络，在神经形态处理器AltAi上实现低功耗实时推理；②通过眼动追踪与目标识别，将手势空间压缩至3种上下文合适的抓握，显著提升准确率并消除不安全抓握；

**🔧 技术方法**

使用技术包括：脉冲神经网络（SNN）在AltAi上实现EMG分类、基于YOLO的事件驱动目标检测、眼动追踪与三维瞳孔定位、以及多通道Dry EMG采集与信号预处理；

**📊 数据集**

数据集：针对上肢截肢患者采集的6种功能性EMG手势数据（10名受试者）以及配套的目标识别场景图像；

**📈 对比分析**

与传统GPU实现对比：AltAi在功耗0.07 W、推理时延≈4.5 ms下获得与RTX 4080相当的准确率；在完整实验中，单手势识别准确率从83%提升至95%，且在不同负重条件下完成时间与疲劳指数随负重显著上升；

**⚠️ 局限性**

局限性：①仅验证6种手势，缺乏更丰富的抓握模式；②眼动与视觉子系统尚未完全迁移至神经形态平台；③实验受试者规模有限，缺少不同截肢水平与肌肉残余情况的验证；

---

## 511. An Efficient Batch Solver for the Singular Value Decomposition on GPUs

**arXiv ID:** 2601.17979 | [PDF](https://arxiv.org/pdf/2601.17979v1)

**作者:** Ahmad Abdelfattah `[一作]` (University of Tennessee), Massimiliano Fasi `[通讯]` (University of Leeds)

**通讯引用:** 347 | [OpenAlex ID](https://openalex.org/A5024408748)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

实现了一个针对GPU的批量奇异值分解（batch SVD）求解器，能够一次性处理大量小型矩阵，并支持所有四种LAPACK精度及左/右奇异向量的计算。

**💡 创新点**

创新点包括：①基于一侧Jacobi算法的全GPU实现，②通过不完整的Hermitian特征求解、定制向量更新核和批量掩码等多层优化显著提升吞吐量；③针对极小矩阵采用共享内存单核实现，④对长宽矩阵预处理使用QR分解以降低计算量。

**🔧 技术方法**

使用技术包括：一侧Jacobi SVD、GPU并行排列（round‑robin）、批量GEMM和自定义的共享内存矩阵乘核、一次性特征求解、掩码批处理、以及可选的QR前处理和共享内存不阻塞实现。

**📊 数据集**

使用随机生成的多种奇异值分布矩阵（uniform、clustered、log‑random、geometric等），并在NVIDIA Grace Hopper GH200与AMD MI300A两款GPU上进行实验；同时与cuSOLVER、rocSOLVER、KBLAS、Wcycle‑SVD以及CPU上的MKL/NVPL进行对比。

**📈 对比分析**

相较于现有GPU实现，MAGMA批量SVD在单、双精度及复数精度下平均提升2–3倍（最高可达8倍）并在极小矩阵上实现3–80倍加速；与CPU实现相比，速度提升范围为2–10倍；与cuSOLVER/rocSOLVER/​KBLAS等库相比，往往能获得数十倍到上百倍的加速。

**⚠️ 局限性**

限制包括：对极大矩阵无法直接并行（需分批）；对共享内存容量敏感，导致对较高精度和较大块尺寸的限制；对矩阵列数远大于行数时的QR前处理实现尚未最优；以及缺乏对可扩展的多GPU协同加速支持。

---

## 512. FedGraph-VASP: Privacy-Preserving Federated Graph Learning with Post-Quantum Security for Cross-Institutional Anti-Money Laundering

**arXiv ID:** 2601.17935 | [PDF](https://arxiv.org/pdf/2601.17935v1)

**作者:** Daniel Commey `[一作]` (Texas A&M University), Garth V. Crosby `[通讯]` (Texas A&M University)

**通讯引用:** 598 | [OpenAlex ID](https://openalex.org/A5005775958)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了 FedGraph-VASP，一种联邦图学习框架，支持跨机构无共享原始交易数据地进行反洗钱检测。

**💡 创新点**

创新点包括：① 基于边界嵌入交换协议，仅共享压缩且不可逆的图神经网络嵌入；② 采用后量子加密（Kyber‑512+AES‑256‑GCM）保障嵌入传输安全；③ 通过边界嵌入对齐损失实现跨机构图拓扑一致性。

**🔧 技术方法**

技术手段：图神经网络（GraphSAGE）、联邦平均（FedAvg）、私钥生成与加密（Kyber‑512 + AES‑GCM）、私集成（PSI）与联邦学习协议。

**📊 数据集**

数据集：主要使用 Elliptic Bitcoin 数据集（Utxo‑基图），并在 Ethereum 欺诈检测数据集上进行泛化验证。

**📈 对比分析**

比较方法：与局部 GNN、标准 FedAvg 以及基于生成式邻域填充的 FedSage+ 对比。FedGraph‑VASP 在 Louvain 低连通划分下 F1 = 0.508，超过 FedSage+（0.453）12.1%，并在高连通 METIS 划分中接近中心化模型（F1 ≈ 0.63）。

**⚠️ 局限性**

局限性：① 仅评估单链（Bitcoin/Ethereum）数据，缺乏跨链实验；② 在极低连通环境下仍受限；③ 对会员推断攻击保护不足；④ 需要进一步研究拜占庭鲁棒性与隐私增强（如差分隐私）。

---

## 513. RemEdit: Efficient Diffusion Editing with Riemannian Geometry

**arXiv ID:** 2601.17927 | [PDF](https://arxiv.org/pdf/2601.17927v1)

**作者:** Eashan Adhikarla `[一作]` (Lehigh University), Brian D. Davison `[通讯]` (Lehigh University)

**通讯引用:** 8513 | [OpenAlex ID](https://openalex.org/A5042328810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 RemEdit 框架，实现高保真、高效的可控图像生成与编辑。

**💡 创新点**

创新点：① 将 U‑Net 瓶颈空间（h‑space）建模为黎曼流形，使用 Mamba 网络学习 Christoffel 符号并通过指数映射求解几何最短路径；② 引入双层 SLERP（Inner‑SLERP 与 Outer‑SLERP）实现编辑强度调节与身份保持；③ 用 Qwen2‑VL 进行目标感知的提示丰富，提升语义对齐；④ 设计任务感知的注意力剪枝头，专门针对编辑任务高效裁剪不必要的 token。

**🔧 技术方法**

主要技术：Mamba 预测曲率、指数映射求解 geodesic ODE、双层 SLERP、Vision‑Language 模型 Qwen2‑VL、任务感知注意力剪枝、DDIM 采样。

**📊 数据集**

数据集：CelebA‑HQ、LSUN‑Church、AFHQ‑Dog。

**📈 对比分析**

与 Asyrp、P2P 等基线对比，RemEdit 在语义对齐（S_dir）与身份保持（SC）上均超越对手；在 20–50% 剪枝后，推理时间从 2.89 s 降至 2.38 s，性能兼优。

**⚠️ 局限性**

局限：依赖预训练的 Qwen2‑VL 与 CLIP，域外任务需额外适配；几何求解虽高效但仍增加计算开销；当前验证仅在无监督 h‑space 上，可能在特定领域表现受限。

---

## 514. CommonLID: Re-evaluating State-of-the-Art Language Identification Performance on Web Data

**arXiv ID:** 2601.18026 | [PDF](https://arxiv.org/pdf/2601.18026v1)

**作者:** Pedro Ortiz Suarez `[一作]` (Common Crawl Foundation), Sarah Luger `[通讯]` (MLCommons)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过社区驱动、人类标注的Web域文本，创建了覆盖109种语言的CommonLID语言识别评测数据集，并用它评估八个主流LID模型及几款大型语言模型。

**💡 创新点**

创新点在于：①提供了首个面向Web域且面向低资源语言的开放式、行级标注数据集；②系统比较了不同覆盖率、不同训练来源模型在该域的真实表现；③揭示现有模型对Web文本的性能低估，推动更公平、准确的语言识别技术。

**🔧 技术方法**

技术手段包括：使用fastText、OpenLID、GlotLID等模型对CommonCrawl、MADLAD-400文本进行预筛选；自定义Dynabench界面实现行级标注；统一ISO 639-3语言标签；宏平均F1和误报率作为评估指标；利用DSPy对OpenAI GPT系列进行零样本LID测试。

**📊 数据集**

所用数据集：①CommonCrawl WET文件（Web文本）和MADLAD‑400（噪声文本）；②CommonLID自建标注集；③公开评测集（SmolSent、Bible、Social Media、FLORES、UDHR）用于对照评估。

**📈 对比分析**

比较方法：对每个模型分别计算覆盖全部语言（all）和模型覆盖语言子集（cov.）的宏平均F1和FPR；还在所有模型均覆盖的76种核心语言上绘制推理速度与F1曲线。结果显示CommonLID上大多数模型F1仅60–70%，GlotLID与CLD2在速度/覆盖率平衡上位于Pareto前沿；LLM在Web域仍低于GlotLID，性能差距在核心语言约-1.8%，在非洲语言约-30%。

**⚠️ 局限性**

局限性：①采样受限于已有LID模型，导致语言、域偏倚；②多标签/宏/微语言统一难度大，影响标注一致性；③缺乏多标注者一致性评估；④LLM评测样本被下采样；⑤低资源语言仍缺少足够样本，无法全面验证模型。

---

## 515. Persistent Permutability in Choice Petri Nets

**arXiv ID:** 2601.18004 | [PDF](https://arxiv.org/pdf/2601.18004v1)

**作者:** Eike Best `[一作]` (Carl von Ossietzky University), Raymond Devillers `[通讯]` (Universite Libre de Bruxelles)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

论文研究了在可达图上如何通过置换事件序列得到可持续的等价序列，并探讨了在等冲突（Equal‑Conflict）和纯消解对称（Pure Dissymmetric‑Choice）两类Petri网中SPE（短序列可持续性）与FPE（公平序列可持续性）之间的蕴含关系。

**💡 创新点**

创新点在于将Ochmański的猜想从安全自由选择网推广到不安全且可持续的等冲突网，并证明在纯 DC 网满足SPE时必定是可持续且满足FPE；同时给出了在纯、平凡但2‑有界网中SPE但不满足FPE的反例，表明SPE⇒FPE不适用于所有平凡网。

**🔧 技术方法**

主要技术手段是Petri网理论中的模式嵌入（pattern embedding）、等冲突（EC）与消解对称（DC）性质的逻辑推理、以及对齐（unification）可持续序列构造的归纳证明。

**📊 数据集**

论文未使用任何具体数据集，所有讨论均基于抽象Petri网和可达图的理论构造。

**📈 对比分析**

比较方法为理论证明；对于EC和纯 DC 网，证明表明满足SPE的网必为可持续且满足FPE；而在非安全或非 DC 网中，这一结论不成立，性能表现未涉及实验。

**⚠️ 局限性**

局限性在于结论仅在保持“纯”和“平凡”的前提下成立，且对等冲突网的结果不适用于不安全网；此外，SPE 与 FPE 之间的蕴含关系在更一般的 Petri 网族中仍未完全解决。

---

## 516. Systematic Characterization of Minimal Deep Learning Architectures: A Unified Analysis of Convergence, Pruning, and Quantization

**arXiv ID:** 2601.17987 | [PDF](https://arxiv.org/pdf/2601.17987v1)

**作者:** Ziwei Zheng `[一作]` (Newcastle University), Varun Ojha `[通讯]` (Newcastle University)

**通讯引用:** 1329 | [OpenAlex ID](https://openalex.org/A5058318377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

系统探索了最小网络在收敛、剪枝和量化三种压缩技术下的性能表现，覆盖MLP/DNN、CNN和ViT三类架构。

**💡 创新点**

提出统一的实验框架，将收敛阶段、剪枝容忍度与量化鲁棒性三者关联，揭示了不同任务难度下的最小参数阈值与稳定学习区间。

**🔧 技术方法**

采用结构化架构扫描、Adam训练、L1剪枝、8‑bit量化感知训练（QAT）以及统计误差分析等技术。

**📊 数据集**

使用MNIST、Fashion‑MNIST和CIFAR‑10三大经典图像分类数据集进行实验。

**📈 对比分析**

通过对比准确率、方差、可剪枝比例与8‑bit量化误差，发现当参数达到稳定阈值后，深层模型可容忍超过60%剪枝，8‑bit误差在简单任务下≤3%，但在CIFAR‑10上仍保持4–6%的差距。

**⚠️ 局限性**

局限性包括仅覆盖小规模图像任务、未探讨更复杂模型或更高位量化、剪枝方法单一、实验成本高且对更大数据集的推广性未知。

---

## 517. Federated learning for unpaired multimodal data through a homogeneous transformer model

**arXiv ID:** 2601.17986 | [PDF](https://arxiv.org/pdf/2601.17986v1)

**作者:** Anders Eklund `[一作]` `[通讯]`, Anders Eklund

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种在不同节点持有单一模态且数据无配对的联邦学习框架，利用统一的多模态Transformer进行训练。

**💡 创新点**

创新点包括使用公共anchor集与Gram矩阵通过CKA实现跨模态语义对齐、将LoRA与几何对齐结合的GeoLoRA、子空间稳定化LoRA以及基于不确定性的精度加权聚合。

**🔧 技术方法**

采用预训练Tokenizer、Adapter、共享Transformer、CKA对齐、LoRA/GeoLoRA、子空间稳定化、精度加权聚合等技术。

**📊 数据集**

实验使用公开的多模态医疗数据集（如医学影像、文本、基因、表格）和少量公共anchor集，并在开源多模态任务上验证。

**📈 对比分析**

与传统FedAvg和单模态方法对比，实验显示在保持高语义一致性的同时显著降低通信成本，且模型在多模态任务上性能不低于中心化训练。

**⚠️ 局限性**

局限性包括需要可用的公共anchor集、对anchor质量和分布偏差敏感，以及在极端异构节点间可能仍存在对齐不完全的问题。

---

## 518. Eyes on the Mission: Mixed Methods Assessment of Eye-Tracker-Enabled Interactive Decision Support in a Simulated Unmanned Aerial Vehicle System

**arXiv ID:** 2601.18015 | [PDF](https://arxiv.org/pdf/2601.18015v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 519. TensorLens: End-to-End Transformer Analysis via High-Order Attention Tensors

**arXiv ID:** 2601.17958 | [PDF](https://arxiv.org/pdf/2601.17958v1)

**作者:** Ido Andrew Atad `[一作]` (Tel Aviv University), Lior Wolf `[通讯]` (Tel Aviv University)

**通讯引用:** 25629 | [OpenAlex ID](https://openalex.org/A5078102229)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构造高阶张量将完整 Transformer 视为数据控制的线性算子，提出 TensorLens 以统一表征所有层与子模块的注意力。

**💡 创新点**

以 4 阶张量完整封装自注意力、FFN、LayerNorm、残差等所有组件，实现比传统注意力聚合更精确、理论上可解释的全模型线性表示。

**🔧 技术方法**

采用张量化的注意力、层归一化、FFN 等子层，利用 Kronecker 乘积、Hadamard 乘积与张量收缩等数学工具，并给出对输入扰动的误差上界分析。

**📊 数据集**

在视觉领域使用 DeiT（ImageNet‑1K）和 DeiT‑small；在 NLP 领域使用 BERT/RoBERTa（IMDB）以及自回归 LLM（Pythia‑1B、Pico‑570M、Phi‑1.5）在 WikiText‑103；关系解码任务采用 Pythia‑1B 关系数据。

**📈 对比分析**

与 8 种传统注意力聚合基线（Rollout、Mean、Attn、W. Attn、W. AttnResLN、GlbEnc 等）在扰动实验中比较；TensorLens 在视觉/文本扰动测试中的 AUC 分别超过 0.8/0.15，显著优于基线；关系解码中准确率亦高于 LRE。

**⚠️ 局限性**

张量化对 GPU 内存占用高，实验仅限于 ≤1B 参数模型；线性化过程采用简化假设，未充分探索张量稀疏性、秩崩塌等潜在优势。

---

## 520. From Specialist to Generalist: Unlocking SAM's Learning Potential on Unlabeled Medical Images

**arXiv ID:** 2601.17934 | [PDF](https://arxiv.org/pdf/2601.17934v1)

**作者:** Vi Vu `[一作]` (Carnegie Mellon University), Min Xu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12223 | [OpenAlex ID](https://openalex.org/A5100413849)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种专精—通用协同训练框架 SC‑SAM，利用 U‑Net 与 SAM 形成双向共训练循环，在有限标注的医疗影像上有效利用未标注数据进行半监督分割。

**💡 创新点**

创新点在于：①把传统的 U‑Net 作为“专精”网络生成点式提示和伪标签来指导 PEFT‑SAM；②把 SAM 作为“通用”监督者对 U‑Net 进行语义正则化；③通过 sigmoid ramp‑up 控制两侧监督的权重，避免早期过拟合与噪声污染；④实现了简洁的双向协同而非单向或双 SAM 分支的复杂结构。

**🔧 技术方法**

技术手段包括：Parameter‑Efficient Fine‑Tuning（Adapter 插入 SAM 编码器），U‑Net 的半监督学习（一致性、伪标签），点式提示与伪掩码的生成，双向损失（监督 + 伪标签），以及梯度权重 ramp‑up。

**📊 数据集**

实验数据集：PROMISE12（前列腺 MRI）与 COLON 组（CVC‑ClinicDB、Kvasir 等多源大肠息肉图像），分别在 5% 与 10% 标注比例下进行评估。

**📈 对比分析**

与多种基线（PEFT‑SAM、CPC‑SAM、KnowSAM、MedSAM、SAM‑Med‑2D）比较，SC‑SAM 在 Dice、IoU、HD95、ASD 上均取得显著提升，尤其在 5% 标注时超过 KnowSAM 约 5‑10 分，甚至优于现有医学专用 SAM 模型。

**⚠️ 局限性**

局限性包括：①对专精网络的依赖，若 U‑Net 性能不足会削弱整体效果；②需要手动调节 ramp‑up 参数，敏感度较高；③在极端域移位或样本分布不一致时，双向共训练仍可能出现不收敛或过拟合风险。

---

## 521. Learning Transferable Skills in Action RPGs via Directed Skill Graphs and Selective Adaptation

**arXiv ID:** 2601.17923 | [PDF](https://arxiv.org/pdf/2601.17923v1)

**作者:** Ali Najar `[一作]` `[通讯]` (Sharif University of Technology), Ali Najar (Sharif University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过将 Dark Souls III 的战斗分解为摄像机控制、锁定、移动、闪避和治疗-攻击决策五个可复用技能，并利用有向技能图和层次化训练课程，构建了一个模块化代理；在面对阶段性域变时，只需对下游技能进行微调即可恢复性能。

**💡 创新点**

创新点在于：①将复杂实时控制任务建模为有向技能图；②采用层次化训练流程，使上游技能先行固定，降低后续技能学习难度；③实现了在域变时仅针对受影响的下游技能进行选择性微调，显著提升终身学习效率。

**🔧 技术方法**

技术方法包括：模块化技能架构；依赖有向图的层次化训练；使用 Deep Q‑Network (DQN) 进行各技能的价值学习；经验回放、动作空间组合等；以及针对域变的选择性微调策略。

**📊 数据集**

数据集为 Dark Souls III 的游戏状态接口（内存读取）所提供的可观测信息，包含 25 维状态；实验在 Boss 战的 Phase 1 与 Phase 2 两个阶段进行，使用 5 次评估回合和 25 次测试回合。

**📈 对比分析**

与单一端到端 DQN 基线相比，技能图模型在约 230k 步内即可实现 44% 的胜率，而基线几乎无法学习到可靠的战斗行为；在 Phase 2 零射转移时获得 33% 的胜率，微调仅下游两技能后提升至 52%，显示出更高的样本效率和适应能力。

**⚠️ 局限性**

局限性包括：仅使用简单的 DQN 作为学习器，缺乏更先进的策略梯度或 actor‑critic 方法；仅利用内部状态接口，未考虑像素感知；实验仅聚焦单一 Boss，难以评估在更广泛多样化环境中的泛化能力；以及对技能间交互的完整性和长期稳定性仍有待进一步验证。

---

## 522. A System for Name and Address Parsing with Large Language Models

**arXiv ID:** 2601.18014 | [PDF](https://arxiv.org/pdf/2601.18014v1)

**作者:** Adeeba Tarannum `[一作]` (University of Arkansas), John Talburt `[通讯]` (University of Arkansas)

**通讯引用:** 806 | [OpenAlex ID](https://openalex.org/A5080276323)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无细调、基于提示与严格验证的框架，将无结构的姓名和地址文本转换为17字段结构化数据。

**💡 创新点**

创新点在于完全依赖提示引导LLM推断并结合后置规则校验，固定解码与批处理实现可复现，既保持高准确度，又避免昂贵的细调与模型训练。

**🔧 技术方法**

使用Claude 4.0 Sonnet LLM、输入规范化、结构化提示、受限解码、规则验证与置信度评分。

**📊 数据集**

数据集包括1500条匿名/合成的多语种人名地址记录，涵盖美国标准、波多黎各、国际多语种和噪声合成样本。

**📈 对比分析**

与传统模式匹配基线（99.34%准确）比较，取得99.8%精确率，误差率下降约23%，自动处理94%异常案例，置信度分层支持目标人工审核。

**⚠️ 局限性**

局限在于仍有0.2%错误，主要来源于不规则模式识别、记录分隔失败与模糊组件分配；当前规则集中于USPS/美国场景，跨国规则与更复杂多语种语法尚待扩展。

---

## 523. Evaluating Semantic and Syntactic Understanding in Large Language Models for Payroll Systems

**arXiv ID:** 2601.18012 | [PDF](https://arxiv.org/pdf/2601.18012v1)

**作者:** Hendrika Maclean `[一作]` (University of Arkansas), John Talburt `[通讯]` (University of Arkansas)

**通讯引用:** 806 | [OpenAlex ID](https://openalex.org/A5080276323)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了大型语言模型在薪资系统中的数值推理能力，设计了从简单到复杂的分层数据集和四级提示方案，评估模型在语义理解、计算顺序、以及以分美元级精度输出薪酬的表现。

**💡 创新点**

创新点在于：①提出针对高风险薪资计算的“层级式评估框架”；②将提示细度与计算复杂度对应，系统探讨了提示、工具调用与程序辅助三种模式的边界；③给出了可复现的评测流程与多模型对比，为企业在需要审计的场景中选择LLM提供实用决策依据。

**🔧 技术方法**

使用了提示工程（最小化、增量描述、自然语言流程、公式/伪代码）、链式思维、程序辅助（PAL、Toolformer）等技术，并通过 GPT‑5、Claude Sonnet‑4、Perplexity Pro、Grok Auto 以及 Gemini 2.5 Pro 五个主流模型进行实验。

**📊 数据集**

构造了五个层级的合成工资表数据集（very_basic、basic、moderate、complex、very_complex），每一层级增加薪酬组成、税务规则、跨州分摊、外汇转换等复杂度，数据量为每层 100 名员工，含完整的公式计算参考。

**📈 对比分析**

比较方法：对模型输出的薪酬表与公式生成的参考表按两位小数对齐，计算“Exact-within-tolerance”百分比与均方误差（MAE）。结果显示：在 very_basic 至 moderate 层级，最小提示即可达到 100% 精度；在 complex 层级，公式提示（Level 4）可使 Perplexity、GPT‑5、Grok 在 MAE 接近 0；在 very_complex 层级，只有 Perplexity 在 Level 4 达到 100% 精度，其他模型误差仍在数十美元级别。

**⚠️ 局限性**

局限性：①实验仅覆盖合成数据，真实工资表可能存在更多字段和规则；②对高级模型（如 Gemini 2.5 Pro）的评价受限于输出不规范；③仅考虑单次提示，未深入探究持续交互或多步验证的可行性；④在极高复杂度下，模型仍对小数点误差敏感，需要更严格的校验与工具链支持。

---

## 524. Memory-Efficient FPGA Implementation of Stochastic Simulated Annealing

**arXiv ID:** 2601.18007 | [PDF](https://arxiv.org/pdf/2601.18007v1)

**作者:** Duckgyu Shin `[一作]`, Takahiro Hanyu `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了硬件友好的随机模拟退火算法HA‑SSA，用于在FPGA上实现高效、低存储需求的组合优化器；

**💡 创新点**

创新点在于将伪逆温度控制改为整数位移操作并根据温度只存储最大温度下的自旋状态，从而将存储需求降低至传统SSA的六分之一，同时保持相同的求解质量；

**🔧 技术方法**

采用Stochastic计算的p-bit自旋门、FSM实现的tanh、XOR‑shift随机数发生器以及FPGA的BRAM、LUT、FF资源实现硬件；

**📊 数据集**

使用G‑set中的最大割问题（G11、G12、G13）以及自定义的King1图作为实验数据集；

**📈 对比分析**

在软件仿真和FPGA硬件上与传统SSA和SA比较，HA‑SSA在G11、G12、G13上分别实现了相同或更优的割值，并且在FPGA上达到1.00 ms的退火时间，比IPAPT实现快2.64倍；

**⚠️ 局限性**

局限在于目前仅验证了-1/1权重、稀疏连接的最大割问题，尚未在完整连通或整数权重的更复杂组合优化问题上进行评估。

---

## 525. PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation

**arXiv ID:** 2601.18006 | [PDF](https://arxiv.org/pdf/2601.18006v1)

**作者:** Lorenzo Proietti `[一作]` (Sapienza University of Rome), Matt Post `[通讯]` (Microsoft)

**通讯引用:** 4791 | [OpenAlex ID](https://openalex.org/A5108266978)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PEAR，一种将 MT 评估转化为双候选梯度相对评分的监督质量估计指标；

**💡 创新点**

创新点在于将传统单候选绝对评分框架改为双候选相对评分，并通过逆序正则化确保符号一致性；

**🔧 技术方法**

采用跨编码器（InfoXLM Large / XLM‑RoBERTa‑XL）构建源+两候选的联合表示，使用 Huber 损失与抗对称正则化进行训练；

**📊 数据集**

训练数据为 WMT16‑23 的 DA/SQM/MQM 人工评估以及 GPT‑4.1‑mini 生成的扩展 MQM 数据；评估基于 WMT24 MQM 共享任务测试集；

**📈 对比分析**

与匹配的单候选 QE 基线、MetricX、XCOMET、COMET‑22 等多种基准在 SPA、acc_eq*、Avg Corr 上对比，PEAR 在参数更少的情况下取得更高相关性，甚至超过部分参考指标；

**⚠️ 局限性**

局限性包括仅在 3.5B 模型规模上测试，未探索更大模型；对与其他指标低相关性缺乏深入分析；未使用针对性合成对数据。

---

## 526. SD-E$^2$: Semantic Exploration for Reasoning Under Token Budgets

**arXiv ID:** 2601.17982 | [PDF](https://arxiv.org/pdf/2601.17982v1)

**作者:** Kshitij Mishra `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Salem Lahlou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了语义多样性奖励框架 SD‑E²，在小语言模型训练中显式探索不同推理策略，并在发现正确答案后自动切换至利用模式。

**💡 创新点**

创新点在于将冻结句子编码器的几何度量用于奖励语义多样性，并引入认知适配的探索‑利用门控，既避免表面重写，又提升推理效率与精度。

**🔧 技术方法**

使用了 GRPO 强化学习、多目标奖励（正确性、格式、利用、语义探索）、句子编码器（BERT‑style）、KL 正则、z‑score 标准化、QLoRA 4‑bit PEFT 等技术。

**📊 数据集**

主要实验数据集包括 GSM8K（小学数学）、AIME（竞赛数学）和 MedMCQA（医学多选）。

**📈 对比分析**

与基线 GRPO‑CFL（结果驱动）和 GRPO‑CFEE（计数探索）对比，Qwen2.5‑3B‑Instruct 上 ACC 提升至 82.03%（+5.23pp），Llama‑3.1‑8B 提升至 75.44%；AIME 上从 9.87% 提升至 13.28%；MedMCQA 上 ACC 提升至 49.64%。

**⚠️ 局限性**

局限包括依赖冻结句子编码器的语义质量与阈值调参，解析模式易受损，仍可能产生表面多样但无用策略，GRPO 采样与编码器开销，以及仅在三类任务上验证，未覆盖长文本、多语言或代码场景。

---

## 527. LungCRCT: Causal Representation based Lung CT Processing for Lung Cancer Treatment

**arXiv ID:** 2601.18118 | [PDF](https://arxiv.org/pdf/2601.18118v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 528. LLMs as Cultural Archives: Cultural Commonsense Knowledge Graph Extraction

**arXiv ID:** 2601.17971 | [PDF](https://arxiv.org/pdf/2601.17971v1)

**作者:** Junior Cedric Tonga `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1061 | [OpenAlex ID](https://openalex.org/A5065822589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多语言的文化常识知识图谱（CCKG），并将其用于提升跨文化推理和故事生成的性能。

**💡 创新点**

创新点在于：①将LLM视为文化档案，通过递归式 prompt‑based 扩展生成多步因果链；②系统评估了不同语言（英文 vs 本土语）对知识图谱质量的影响；③证明了将 CCKG 作为 in‑context 补充能显著提升小模型在文化推理任务上的表现。

**🔧 技术方法**

采用的技术包括：LLM（GPT‑4o 与 Llama‑3.3‑70B‑Instruct）进行 prompt‑based 断言与路径生成；句子嵌入（SBERT、XLM‑R）用于去重和检索；基于图结构的路径扩展算法；人机评估和 BERTScore、句子相似度等指标。

**📊 数据集**

数据集：从 5 个国家（中国、印尼、日本、英国、埃及）的 11 个日常主题共 65 个子主题中抽取，生成约 37k 条英文断言和 16k 条本土语断言；用于评估的公开基准包括 ArabCulture、IndoCulture 的多项选择与句子完成任务。

**📈 对比分析**

比较方法：与零样本、CoT、Mango（基于事实的知识库）进行对比。结果显示：①在 MCQA 上，加入本土语断言或路径可平均提升 1–2% 的准确率；②在故事生成上，使用 CCKG 路径可使文化相关度、流畅度和连贯性平均提升 0.5–1.5 分；③与基线相比，尤其在小模型上提升最为显著。

**⚠️ 局限性**

局限性：①对 prompt 设计高度敏感，迁移到新模型或语言时需重新调优；②只覆盖资源较高的国家，未深入低资源文化；③使用 LLM 生成内容可能复制刻板印象，需进一步偏见检测；④知识图谱仍为实验原型，不能直接用于生产。

---

## 529. Information-Theoretic Secure Aggregation in Decentralized Networks

**arXiv ID:** 2601.17970 | [PDF](https://arxiv.org/pdf/2601.17970v1)

**作者:** Xiang Zhang `[一作]`, Giuseppe Caire `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了在完全去中心化网络中实现信息理论安全聚合（DSA）的最优通信与密钥需求，并给出了严格的可行率区域；

**💡 创新点**

首次对完全去中心化安全聚合的理论极限进行完整表征，提出了一种线性方案并证明其达到最优，揭示了每用户至少需发1比特、持1比特密钥且全局密钥总量至少K-1比特的必要条件；

**🔧 技术方法**

使用信息理论工具（熵、互信息、Shannon界限）进行正反证，设计零和结构的线性密钥与消息，构造完美安全的聚合协议；

**📊 数据集**

论文为理论分析，不依赖具体数据集，输入被抽象为独立均匀分布的比特；

**📈 对比分析**

由于是理论研究，未做实验比较，所给出的率区域即为性能上限；

**⚠️ 局限性**

局限在于假设网络完全连通、通信可靠且误差为零，输入独立均匀，且协作阈值T ≤ K-3，未考虑用户掉线、拓扑变化或非均匀输入等实际场景。

---

## 530. Typhoon-S: Minimal Open Post-Training for Sovereign Large Language Models

**arXiv ID:** 2601.18129 | [PDF](https://arxiv.org/pdf/2601.18129v1)

**作者:** Kunat Pipatanakul `[一作]` (SCB 10X), Pittawat Taveekitworachai `[通讯]` (SCB 10X)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Typhoon S 这一简洁开放的后训练方案，旨在在资源受限的主权设置中，将基础模型快速转化为具备通用指令跟随能力与本土化专业功能的 LLM。

**💡 创新点**

创新点在于：① 将轻量化 SFT 与基于 on‑policy distillation（OPD）的两阶段流程结合，构建低资源的通用指令微调路径；② 在 RFT 阶段引入 InK‑GRPO，将域知识注入的 next‑token 交叉熵损失与 GRPO 强化学习交叉使用，实现对特定领域（如泰国法律）的知识注入与多步推理。

**🔧 技术方法**

使用技术包括：跨语言 SFT、On‑Policy Distillation（全概率 logits）、GRPO 与 DAPO 的强化学习框架、Agentic RFT 与 Retrieval‑Augmented Generation（RAG）工具调用、以及自定义 reward（格式与准确性）。

**📊 数据集**

数据集涵盖公开英文指令数据 Tulu 3、Toucan 工具集、泰语 AutoIF、泰国法律 NitiBench、MIRAGE‑Bench、以及多语言 MMLU、OpenThaiEval 等本土化基准。

**📈 对比分析**

与 Qwen 3、GPT‑5 等强基线对比，Typhoon‑S‑8B Instruct 在泰语聊天、代码切换、知识与工具使用上显著提升；Typhoon‑S‑4B Legal Agent 在 NitiBench 任务上超越 GPT‑5 的 agentic 方案，整体保持在 48–49 分左右，未出现严重灾难性遗忘。

**⚠️ 局限性**

局限性包括：仅针对泰语与法律领域验证，未研究预训练/中间训练；资源规模仍受限于 4–8xH100，缺乏对更大规模或其他语言的系统扩展与规模曲线。

---

## 531. Enhancing LLM-based Recommendation with Preference Hint Discovery from Knowledge Graph

**arXiv ID:** 2601.18096 | [PDF](https://arxiv.org/pdf/2601.18096v1)

**作者:** Yuting Zhang `[一作]` (Hong Kong University of Science and Technology), Fuzhen Zhuang `[通讯]` (Beihang University)

**通讯引用:** 9759 | [OpenAlex ID](https://openalex.org/A5102969899)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 PIDLR 框架，利用协作式偏好提示提取与实例级双注意力机制，从交互与知识图中挑选关键偏好提示，并以扁平化文本提示喂入 LLM 进行推荐。

**💡 创新点**

创新点在于将传统推荐原理（协同过滤）与实例级提示发现相结合，形成双注意力筛选并生成头中心化扁平化提示，解决了嵌入-语义不匹配与属性噪声问题。

**🔧 技术方法**

采用知识图嵌入、协同过滤、双注意力机制、LoRA 参数高效微调及 LLaMA‑3 大语言模型进行推理。

**📊 数据集**

实验使用 MovieLens 与 LastFM 两个公开数据集，结合其对应的知识图。

**📈 对比分析**

与传统 CF、矩阵分解、KG 基础模型以及多种 LLM 基线（如 LLaMA‑3、MoRec、TallRec、LLaRA、CoLaKG、GLRec）进行 Pair‑wise 与 List‑wise 推荐任务对比，PIDLR 在 HitRatio@1 上平均提升 3–7%，并且在 ValidRatio 与 Few‑shot 场景中表现优异。

**⚠️ 局限性**

局限性包括：依赖知识图的完整性与覆盖度，提示生成与 LLM 长文本处理受 token 限制，且对用户兴趣的实时动态变化支持不足。

---

## 532. From Struggle to Success: Context-Aware Guidance for Screen Reader Users in Computer Use

**arXiv ID:** 2601.18092 | [PDF](https://arxiv.org/pdf/2601.18092v1)

**作者:** Nan Chen `[一作]` (Microsoft Research), Yuqing Yang `[通讯]` (Microsoft Research)

**通讯引用:** 1748 | [OpenAlex ID](https://openalex.org/A5101421201)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了AskEase，一款针对屏幕阅读器用户的实时、基于LLM的辅助系统，提供精准、分步、键盘友好的电脑使用指导；

**💡 创新点**

创新点在于多源上下文感知（环境、知识、会话）与无缝交互设计，支持自适应指导、屏幕描述，并为视觉障碍用户生成可访问的回答；

**🔧 技术方法**

采用GPT‑5等大型语言模型、检索增强生成(RAG)、NVDA插件、截图+屏幕状态+屏幕阅读器轨迹的多模态上下文捕获与提示工程；

**📊 数据集**

使用Windows Agent Arena 45个任务（12款应用）作为鲁棒性评测，Word/Excel 任务集以及12名盲人参与者的用户研究数据；

**📈 对比分析**

与传统搜索/AI助手对比，AskEase在任务成功率上提升至1.5/2（vs 0.5/2），NASA‑TLX工作负荷显著下降，鲁棒性测试成功率达96.6%，每次查询平均成本约$0.005；

**⚠️ 局限性**

局限包括仅在Windows/NVDA平台实现、环境信息采集可能不完整导致误导、检索文档不全易产生幻觉、需用户具备一定AI/键盘熟练度，以及隐私和跨平台适配挑战。

---

## 533. From Human Speech to Ocean Signals: Transferring Speech Large Models for Underwater Acoustic Target Recognition

**arXiv ID:** 2601.18086 | [PDF](https://arxiv.org/pdf/2601.18086v1)

**作者:** Mengcheng Huang `[一作]`, Dapeng Man `[通讯]` (Harbin Engineering University)

**通讯引用:** 708 | [OpenAlex ID](https://openalex.org/A5067145421)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了UATR‑SLM框架，将预训练的语音大型模型迁移到水下目标识别任务；

**💡 创新点**

创新点在于直接复用语音模型的特征提取与编码器，仅做轻量级分类头微调即可达到SOTA，并在可变信号长度与跨域场景下表现出显著鲁棒性；

**🔧 技术方法**

使用SenseVoiceSmall等语音大型模型作为编码器，采用log‑Mel特征提取、全微调、轻量化线性分类头，训练时采用AdamW、WarmupLR与交叉熵损失；

**📊 数据集**

使用DeepShip与ShipsEar这两个公开水下声学数据集；

**📈 对比分析**

与ResNet18/34/50、HUAT、SSA‑CACNN等基线对比，DeepShip上F1达99.32%，ShipsEar上99.00%；在可变长度评估中准确率≥95%，在零样本跨域实验中达到96.67%；

**⚠️ 局限性**

局限在于模型参数量较大（约234M），对资源受限设备部署不友好；跨域评估仅针对单一类别（Passenger）；对极端噪声或更复杂海洋环境的适应性尚未充分验证。

---

## 534. DRPG (Decompose, Retrieve, Plan, Generate): An Agentic Framework for Academic Rebuttal

**arXiv ID:** 2601.18081 | [PDF](https://arxiv.org/pdf/2601.18081v1)

**作者:** Peixuan Han `[一作]` (University of Illinois Urbana-Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7779 | [OpenAlex ID](https://openalex.org/A5003491365)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了四阶段自动学术反驳生成框架 DRPG，能够将评审意见拆解为原子问题、检索论文证据、规划最合适的反驳视角并生成针对性回应。

**💡 创新点**

创新点在于引入专门的 Planner 通过 MLP 评估候选视角与论文段落的支持度，达成 98%+ 的视角识别精度，同时提供多视角解释与可解释性；整个流程通过 LLM 代理实现端到端的结构化反驳。

**🔧 技术方法**

技术包括：大语言模型代理（LLM）、dense retrieval（BGE‑M3）、文本编码器+MLP 规划、四阶段流水线（Decomposer、Retriever、Planner、Executor），并结合自信度阈值控制 Planner 的使用。

**📊 数据集**

使用 Re^2 数据集（约 17k 篇论文、60k 条评审与反驳），涵盖 45 个顶级 CS 会议（ACL、ICLR、NeurIPS 等）。

**📈 对比分析**

通过 Elo 评分、Judge Score 以及人类对比评估，DRPG 在 4 种基准 LLM 上平均提升约 40 Elo，且在 8B 模型下已超过平均人类水平；Planner 阈值 0.8 时约 62% 的点评点被有效规划，显著提升整体表现。

**⚠️ 局限性**

局限性：仅能澄清已有内容和辩护已发表结果，无法生成新的实验或数据；模型可能出现幻觉，需要作者人工复核与验证。

---

## 535. EvolVE: Evolutionary Search for LLM-based Verilog Generation and Optimization

**arXiv ID:** 2601.18067 | [PDF](https://arxiv.org/pdf/2601.18067v1)

**作者:** Wei-Po Hsin `[一作]` (National Taiwan University), Shih-Hao Hung `[通讯]` (National Taiwan University)

**通讯引用:** 1124 | [OpenAlex ID](https://openalex.org/A5020028710)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EvolVE 框架，利用 LLM 与两种进化搜索（Idea‑Guided Refinement 与 Monte Carlo Tree Search）实现 Verilog 的自动生成与优化；

**💡 创新点**

首次将 MCTS 与 IGR 并用，既保证功能正确性又实现 PPA 优化；引入结构化测试生成 (STG) 提升反馈质量；并构建工业级 IC‑RTL 基准；

**🔧 技术方法**

使用 LLM 代码生成、检索增强生成、蒙特卡洛树搜索、Idea‑Guided Refinement、结构化测试生成、Yosys + Icarus Verilog、商用 EDA 流程等技术；

**📊 数据集**

评估基准包括 VerilogEval v2、Mod‑VerilogEval v2、RTLLM v2 以及自研的 IC‑RTL 复杂度更高的工业级数据集；

**📈 对比分析**

通过与现有 LLM 及人类参赛实现对比，功能正确率达 98.1%（VerilogEval v2）与 92%（RTLLM v2），在 IC‑RTL 上 PPA 乘积比参赛者低 66%（单例）或 17%（几何平均），显示显著性能提升；

**⚠️ 局限性**

局限性包括：STG 依赖可执行的黄金参考模型；工业级 PPA 优化基准仍有限；缺乏实时的 PPA 反馈闭环；以及对新 IP 的可扩展性待进一步验证。

---

## 536. Multimodal Machine Learning for Soft High-k Elastomers under Data Scarcity

**arXiv ID:** 2601.18032 | [PDF](https://arxiv.org/pdf/2601.18032v1)

**作者:** Brijesh FNU `[一作]` (University of Alabama at Birmingham), Truong-Son Hy `[通讯]` (University of Alabama at Birmingham)

**通讯引用:** 202 | [OpenAlex ID](https://openalex.org/A5073178563)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个聚丙烯酸酯基软介质弹性体数据集，并提出了利用预训练的聚合物序列与图模型的多模态学习框架，实现极少样本下电介质常数与杨氏模量的双目标预测。

**💡 创新点**

创新点在于将聚合物语言模型与图神经网络的预训练表示进行跨模态对齐，形成早期融合的多模态框架，使得在仅有35条样本的情况下显著提升了预测精度。

**🔧 技术方法**

使用了Transformer基聚合物语言模型（PolyBERT、TransPolymer）、GIN图神经网络、Gaussian Process回归以及对齐的早期融合技术。

**📊 数据集**

采用从文献整理的35条丙烯酸酯基弹性体实验数据，包含电介质常数、杨氏模量和对应的SMILES序列。

**📈 对比分析**

与传统摩根指纹、单模预训练模型以及不同融合策略对比，最佳的对齐早期融合取得平均R²≈0.758、RMSE≈13.7，优于单模模型和其他融合方案。

**⚠️ 局限性**

局限在数据集规模仍极小、仅覆盖丙烯酸酯骨架，未包含硅胶、聚氨酯等其它弹性体或掺杂复合材料，也未考虑温度等环境因素对性能的影响。

---

## 537. AttenMIA: LLM Membership Inference Attack through Attention Signals

**arXiv ID:** 2601.18110 | [PDF](https://arxiv.org/pdf/2601.18110v1)

**作者:** Pedram Zaree `[一作]` (University of California), Nael Abu-Ghazaleh `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种利用 Transformer 自注意力模式进行成员推断的攻击框架 AttenMIA。

**💡 创新点**

创新点是首次将注意力层的过渡与扰动特征作为内部隐私信号，证明其在低误报率下优于传统输出或梯度基方法。

**🔧 技术方法**

采用多头注意力统计（相关性、弗罗贝尼乌斯距离、KL、重心漂移）和输入扰动（token drop/replace/prefix）提取特征，并用轻量 MLP 分类器进行预测。

**📊 数据集**

在 WikiMIA、MIMIR 等公开数据集上，对 LLaMA‑2、Pythia、OPT 等开源大模型进行评估。

**📈 对比分析**

与多种基线（PPL、LOSS、ZLIB、PETAL、RECALL 等）对比，AttenMIA 在 AUC 与 TPR@1%FPR 均表现更佳，尤其在 1% FPR 下可达近 88% 或更高。

**⚠️ 局限性**

局限性包括：仅适用于白盒模型、对去重等简单防御无显著抑制、对长序列性能略退化、需要完整模型内部访问。

---

## 538. Neurocomputational Mechanisms of Syntactic Transfer in Bilingual Sentence Production

**arXiv ID:** 2601.18056 | [PDF](https://arxiv.org/pdf/2601.18056v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 539. Text-Pass Filter: An Efficient Scene Text Detector

**arXiv ID:** 2601.18098 | [PDF](https://arxiv.org/pdf/2601.18098v1)

**作者:** Chuang Yang `[一作]` (Northwestern Polytechnical University), Qi Wang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 17882 | [OpenAlex ID](https://openalex.org/A5100341261)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于带通滤波概念的文本检测框架 Text‑Pass Filter (TPF)，通过端到端网络直接分割任意形状文本，并设计了 Reinforcement Ensemble Unit (REU) 与 Foreground Prior Unit (FPU) 两个增强模块以提升检测精度与速度。

**💡 创新点**

创新点包括：①将电子学中的带通滤波器思想迁移到文本检测，构造文本特征‑滤波器对实现一次性完整文本分割；②REU 通过自分类矩阵和滤波器集成算法提升同一文本特征一致性和滤波器识别域；③FPU 利用前景优先的语义差异学习显著提高中心点定位与文本区分能力；④整个框架实现了在不需要复杂解码或后处理的情况下实现实时检测。

**🔧 技术方法**

技术方案涵盖：ResNet+FPN 的特征提取；中心点预测头；特征‑滤波器对生成器；REU 的自分类与滤波器集合；FPU 的前景差异学习；使用 focal loss 处理中心点与强化矩阵的不平衡；dice loss 监督连续标签；多尺度数据增强与随机变换；整体训练分为 SynthText 预训练 + 细调。

**📊 数据集**

数据集：SynthText 用于预训练；官方公开数据集 MSRA‑TD500、ICDAR2015、Total‑Text、CTW1500 以及 HUST‑TR400（用于补充训练）。

**📈 对比分析**

与多种 state‑of‑the‑art 方法（如 PAN、DB、MTS‑v3、GV、MCN、ABPN、CM‑Net、DB++ 等）在公开 benchmark 上比较，TPF 在 MSRA‑TD500、ICDAR2015、Total‑Text、CTW1500 的 F‑measure 均位居榜首或相近，并在 FPS 方面显著领先（如 MSRA‑TD500 34.5 FPS、ICDAR2015 28.7 FPS、Total‑Text 34.0 FPS、CTW1500 41.4 FPS）。

**⚠️ 局限性**

局限性：①对文本覆盖（overlay）和极度重叠的文本仍易失效；②REU 在合并滤波器时可能将不同文本的滤波器误合，导致多重检测；③在极端遮挡、光照变化或噪声环境下的鲁棒性仍有提升空间。

---

## 540. Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents

**arXiv ID:** 2601.18077 | [PDF](https://arxiv.org/pdf/2601.18077v1)

**作者:** Mahesh Ramesh `[一作]` (University of Wisconsin-Madison), Aniket Rege `[通讯]` (University of Wisconsin-Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Hanabi游戏中对17种LLM代理进行大规模评估，设计多种提示工程与隐式状态跟踪实验，并提出公开的HanabiLogs与HanabiRewards两套训练数据。

**💡 创新点**

创新点在于系统化的提示对齐、跨模型自评估、公开的带动作价值数据集，并通过微调使小型LLM接近最强模型表现。

**🔧 技术方法**

采用程序化贝叶斯推理提示、工作记忆式多轮提示、RL与可验证奖励（RLVR）以及Qwen3-4B-Instruct的监督与强化学习微调。

**📊 数据集**

使用HanabiLogs（1,520局完整轨迹）和HanabiRewards（560局密集动作价值）两套公开数据集。

**📈 对比分析**

与自我对弈与交叉对弈相比，跨模型表现平滑；在Hanabi中RL微调的4B模型平均得分约为23/25，略低于23+的专用搜索代理，但比GPT‑4.1高出52%。

**⚠️ 局限性**

局限在于模型仍难以完美跟踪隐式状态、对提示变化敏感、未实现与人类玩家的真实协作，以及跨域推广仍需进一步研究。

---

## 541. Secure Beamforming and Reflection Design for RIS-ISAC Systems under Collusion of Passive and Active Eavesdroppers

**arXiv ID:** 2601.18063 | [PDF](https://arxiv.org/pdf/2601.18063v1)

**作者:** Tian Zhang `[一作]`, Yueyi Dong `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了在RIS辅助ISAC系统中，针对主动与被动窃听者协同攻击时的物理层安全问题，提出联合基站波束成形、RIS反射系数与接收波束优化，以在满足感知性能和功率约束的前提下最大化系统保密率。

**💡 创新点**

创新点在于首次考虑主动窃听者产生干扰信号与被动窃听者协同的安全场景，构建完整的保密率最大化问题，并通过交替优化结合SCA与二次罚方法，将原始非凸问题分解为可解的凸子问题，实现对三者协同优化的统一框架。

**🔧 技术方法**

技术手段包括：交替优化（AO）分解、凸性近似的SCA、二次罚方法（处理RIS相位单位模约束）、Rayleigh商优化求解接收波束、并结合对数率变换与辅助变量实现最大化保密率。

**📊 数据集**

实验使用基于Rician/瑞利模型的信道仿真，设定K=3、M=80、N=6/8、路径损耗指数等参数，采用随机布置用户、雷达目标与RIS位置的模拟数据进行性能评估。

**📈 对比分析**

与三种基准（随机RIS相位、随机接收波束、无RIS）比较，实验显示所提JBRD算法在功率、RIS位置、感知阈值和AE干扰功率变化下均能显著提升系统保密率；同时在RIS元素数增加或基站天线数提升时，系统保密率随之提升，验证了算法的优越性。

**⚠️ 局限性**

局限性在于：1）仅在单目标感知场景下验证，缺乏多目标扩展；2）算法收敛速度与初始化相关，需精心设计初始点；3）实际部署中RIS相位控制精度与信道估计误差未考虑，可能影响性能。

---

## 542. Expert Evaluation and the Limits of Human Feedback in Mental Health AI Safety Testing

**arXiv ID:** 2601.18061 | [PDF](https://arxiv.org/pdf/2601.18061v1)

**作者:** Kiana Jafari `[一作]` (Stanford University), Mykel Kochenderfer `[通讯]` (Stanford University)

**通讯引用:** 12246 | [OpenAlex ID](https://openalex.org/A5068326377)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文研究了三名认证精神科医生对 360 条 LLM 生成的心理健康对话回应进行评价时的可靠性，利用定量指标（ICC、Krippendorff’s α、MAD）评估一致性，并通过半结构化访谈揭示专家框架差异，最终提出在高风险 AI 安全评估中不宜简单平均标签，而应保留并利用专家分歧。

**💡 创新点**

创新点在于首次提供“存在证明”，表明即使在专业训练、同一任务下，专家评价在安全关键领域也会出现系统性、非随机的分歧；指出分歧是由不同临床哲学（安全优先、治疗参与、文化背景）驱动的社会技术现象；提出在 AI 对齐与评测中应保存并学习专家分歧，而非仅求共识。

**🔧 技术方法**

使用了混合方法：①统计可靠性分析（单测 ICC(2,1)、平均 ICC(2,k)、Krippendorff’s α、MAD 以及方向性偏差检验）；②质性主题分析（对访谈转录进行反思性编码）；④对四种大型语言模型（GPT‑5、Claude 4 Sonnet、Grok‑4、Llama 3.2）输出的评价差异进行模型间比较；⑤采用 bootstrapping、Friedman 检验、配对 t 检验等统计手段。

**📊 数据集**

数据集为 360 条人工生成的心理健康情景回应，覆盖 10 个高危风险类别（自杀、非自杀性自伤、幻觉、药物滥用等），并按三种严重程度与沟通直率交叉生成；四大模型生成回应；共 8×360×3=8,640 条评注，提供 1,080 条响应级别标注；使用 8 个评估因子（安全与质量两维，共 8 个子项）进行打分。

**📈 对比分析**

方法比较：在所有因子上，单测 ICC 均低于 0.40，平均 ICC 仅在 0.22–0.56 范围内；Krippendorff’s α 多为负或低于 0.67；高危类（自杀/自伤）MAD > 0.56，显示最严重分歧；模型比较显示 Grok‑4 的分歧最大，其它模型差异相对较小。相较于传统的平均共识标签，本文显示在安全关键任务中，平均标签几乎失效，无法反映任何专家的实际判断。

**⚠️ 局限性**

局限性包括：仅 3 名专家评审，难以覆盖精神科的完整哲学多样性；受训机构均为美国，可能与其他文化背景存在差异；使用合成提示，缺乏真实患者数据；仅评估单轮对话，忽略了多轮交互与时间维度；尽管进行了 90 分钟的校准，框架差异仍难以通过进一步校准消除。

---

## 543. Robust Learning of a Group DRO Neuron

**arXiv ID:** 2601.18115 | [PDF](https://arxiv.org/pdf/2601.18115v1)

**作者:** Guyang Cao `[一作]` (University of Wisconsin-Madison), Jelena Diakonikolas `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在存在标签噪声和群体分布偏移的情况下，提出一种基于双重外推的原始-对偶算法，学习单个神经元并实现鲁棒性。

**💡 创新点**

创新点在于：①将 f‑divergence 正则化引入分组分布鲁棒优化；②在非凸平方损失下采用双侧对偶外推并给出常数因子近似保证；③通过“线性化”与群组结构结合，突破传统高阶误差限制。

**🔧 技术方法**

使用了：原始-对偶优化、对偶外推（momentum）技术、f‑divergence 正则化、梯度剪裁与凸性/尖锐性分析、线性化 lemma、随机采样与经验上界推导。

**📊 数据集**

实验数据集：RedPajama 语言模型预训练数据；Sheared‑LLaMA 1.3B（来自 LLaMA2‑7B 预训练）以及 11 个下游任务（ARC‑E/C, HellaSwag, SciQ, WinoGrande, BoolQ, LogicQA, LAMBADA, TruthfulQA 等）。

**📈 对比分析**

对比方法：与 DoReMi 的动态批加载（仅改用我们的对偶更新）进行对比。实验结果显示，PD‑KL 在多次检查点平均准确率提升 0.04%–0.96%，在 33.6M tokens 时达到 47.87%（相较 DoReMi 训练时间缩短 1.5 倍），在 92.4M tokens 时平均提升约 1%。

**⚠️ 局限性**

局限性：①理论保证仅为常数因子误差，未达到最优精度；②实验规模有限，主要验证预训练阶段，未在更大模型或更广泛任务上深入验证；③缺乏对不同 f‑divergence 与 λ 取值的系统性分析。

---

## 544. Semi-Supervised Hyperspectral Image Classification with Edge-Aware Superpixel Label Propagation and Adaptive Pseudo-Labeling

**arXiv ID:** 2601.18049 | [PDF](https://arxiv.org/pdf/2601.18049v1)

**作者:** Yunfei Qiu `[一作]` (Liaoning Technical University), Wei Yao `[通讯]` (State Key Laboratory of Regional and Urban Ecology, Institute of Urban Environment, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于超像素边缘感知标签传播与动态历史融合伪标签的半监督高光谱图像分类框架，实现少样本高精度分类。

**💡 创新点**

创新点包括：①Edge‑Aware Superpixel Label Propagation (EASLP)抑制边界标签扩散；②Dynamic History‑Fused Prediction (DHP)平滑伪标签时序；③Adaptive Tripartite Sample Categorization (ATSC)分层利用易/模糊/难样本；④将三者集成为DREPL实现时空一致性优化。

**🔧 技术方法**

采用SLIC超像素分割、Sobel边缘检测、余弦相似度标签传播、历史预测队列与动态加权融合、置信度与一致性阈值分层、双路径数据增强以及基于PyTorch的卷积神经网络。

**📊 数据集**

使用四个公共高光谱数据集：PaviaU、Houston2013、KSC、Botswana。

**📈 对比分析**

与两种全监督基线（A2S2K、SSTN）、两种半监督方法（DMSGer、CTF‑SSCL）及两种少样本自监督方法（DEMAE、RMAE）比较；在10样本/类条件下，平均OA/AA/Kappa分别达到95.21%/94.46%/93.67%，在所有数据集上均取得最高准确率，优于对手1–3个百分点。

**⚠️ 局限性**

仅适用于单场景，缺乏跨域适应与多场景迁移能力；伪标签生成仍受少量标注影响，对极端噪声与复杂地形的鲁棒性待进一步提升。

---

## 545. RGFL: Reasoning Guided Fault Localization for Automated Program Repair Using Large Language Models

**arXiv ID:** 2601.18044 | [PDF](https://arxiv.org/pdf/2601.18044v1)

**作者:** Melika Sepidband `[一作]` (York University), Hadi Hemmati `[通讯]` (York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于LLM的推理驱动的故障定位（RGFL）框架，在自动程序修复的文件级和元素级定位中引入逐候选解释与重排序。

**💡 创新点**

创新点在于首次将LLM生成的自然语言推理作为定位信号，并通过两阶段排名（LLM推理+嵌入相似度）显著提升定位准确性，且提供了对定位误差的反事实上界分析。

**🔧 技术方法**

采用Gemini 2.5 Pro（及Claude 4 Sonnet、o4‑mini）作为推理模型，结合Embedding（Gemini‑embedding‑001、Voyage 3.5、text‑embedding‑3‑small）实现文件/元素的重排序；在Agentless流水线中替换文件/元素定位模块；使用RAG、Spectrum‑Based、IR等基线进行对照。

**📊 数据集**

在SWE‑bench Verified（500实例）、Lite（300实例）和Java（91实例）三大子集上评估，涵盖Python与Java两种语言。

**📈 对比分析**

与Agentless、OpenHands、OrcaLoca、AutoCodeRover等基线对比，RGFL在文件级Hit@1/3从71.4%提升至85%/93%，MRR提升至88.8%；元素级Exact Match从36%提升至69%；在Verified集上端到端修复成功率从52%提升至58.2%，实现12.8%绝对提升。

**⚠️ 局限性**

局限包括：仅在Python/Java上验证；对LLM的依赖导致成本上升（约4.4美元/样本）；对行级推理未做改进；实验仅覆盖SWE‑bench数据集，可能不适用于C++/JavaScript等其他生态。

---

## 546. An Experimental Comparison of Cognitive Forcing Functions for Execution Plans in AI-Assisted Writing: Effects On Trust, Overreliance, and Perceived Critical Thinking

**arXiv ID:** 2601.18033 | [PDF](https://arxiv.org/pdf/2601.18033v1)

**作者:** Ahana Ghosh `[一作]` (Max Planck Institute for Software Systems), Christian Poelitz `[通讯]` (Microsoft Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在AI生成执行计划（plan）审查中使用认知强制函数（CFF）对用户判断和过度依赖的影响。

**💡 创新点**

创新点在于：① 将CFF与计划评估相结合；② 设计并比较三种基于批判性思维技能的plan‑中心CFF（论证分析、假设检验、两者组合）；③ 探索CFF在长篇写作任务中的实际效果。

**🔧 技术方法**

采用了在线实验和访谈相结合的混合方法：随机分配四种条件（两种CFF、一种组合、一种无CFF）；使用NASA‑TLX评估认知负荷；通过准备度评分、准确率、过度/欠度依赖率等指标衡量效果。

**📊 数据集**

使用了五个写作任务，任务素材取自CNN/DailyMail、MeetingBank、SelSum等公开数据集，使用GPT‑4o生成计划和草稿，并人为注入缺失步骤、错误步骤等合成错误。

**📈 对比分析**

在214名参与者的实验和12名访谈中，论证分析型CFF（Assumption‑Based）在降低过度依赖、提升准确率方面优于假设检验型CFF（What‑If）和组合型；且其认知负荷与无CFF相当；假设检验型CFF虽被用户认为更有帮助，却未带来性能提升，且负荷更高。

**⚠️ 局限性**

局限性包括：① 仅研究写作任务，缺乏对程序、数据分析等技术性知识工作场景的验证；② 合成错误类型有限，未系统探索不同错误难度对CFF效果的影响；③ CFF提示的质量与可扩展性尚未在真实生产环境中检验；④ 样本主要为熟悉GenAI的工作者，未覆盖更广泛的用户群体。

---

## 547. Mitigating the OWASP Top 10 For Large Language Models Applications using Intelligent Agents

**arXiv ID:** 2601.18105 | [PDF](https://arxiv.org/pdf/2601.18105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 548. Decentralized Multi-product Pricing: Diagonal Dominance, Nash Equilibrium, and Price of Anarchy

**arXiv ID:** 2601.18117 | [PDF](https://arxiv.org/pdf/2601.18117v1)

**作者:** Boxiao Chen `[一作]` (University of Illinois Chicago), Stefanus Jasin `[通讯]` (University of Michigan)

**通讯引用:** 1532 | [OpenAlex ID](https://openalex.org/A5034142159)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对多品类线性需求模型下的去中心化定价问题进行理论分析，建立纯策略纳什均衡的存在与唯一性，并量化其与集中定价最优收入之间的效率损失（Price of Anarchy）

**💡 创新点**

提出了一个最优的下界 PoA ≥ 4(1−μ)/(2−μ)²，证明该下界在满足对角占优的需求矩阵下是最严谨的；并进一步给出了基于需求交互矩阵谱特征的完全显式表达式，揭示不同网络拓扑对去中心化效率的精确影响

**🔧 技术方法**

主要使用矩阵分析技术：对称负定矩阵的逆、Loewner 级序、特征值分解、正定矩阵平方根，以及 Rayleigh 商的谱刻画；同时利用对角占优条件与特征值界定来推导效率下界

**📊 数据集**

本研究纯理论无实验数据，所使用的“数据集”是对线性需求参数（斜率矩阵 B）的一般符号设定，特别强调了满足对角占优的任意矩阵以及对称可交换模型的特殊构造

**📈 对比分析**

与集中定价最优解进行直接比较：PoA 是去中心化均衡收入与集中最大收入之比。研究给出了最差情况下的闭式下界，并证明该下界在对称可交换模型中可被实现；在特定网络（如星形、均匀）下，谱参数可显著改善该下界，说明实际效率可能远高于最坏情况预测

**⚠️ 局限性**

局限性包括：仅考虑线性需求形式；假设需求斜率矩阵满足对称与严格对角占优，限制了模型的通用性；没有考虑动态学习或时间序列效应；缺乏经验验证和对非对称、非线性交互的扩展

---

## 549. MalURLBench: A Benchmark Evaluating Agents' Vulnerabilities When Processing Web URLs

**arXiv ID:** 2601.18113 | [PDF](https://arxiv.org/pdf/2601.18113v1)

**作者:** Dezhang Kong `[一作]` (Zhejiang University), Meng Han `[通讯]` (Zhejiang University)

**通讯引用:** 2841 | [OpenAlex ID](https://openalex.org/A5100785901)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MalURLBench，针对Web Agent中LLM处理恶意URL的安全问题构建了一个大规模评测基准，并在此基准上评估了12种主流LLM模型，深入分析了影响攻击效果的因素，随后设计并实现了轻量化防御模块URLGuard。

**💡 创新点**

创新点：①首次系统化地针对恶意URL结构进行基准化评测；②构建61,845条真实恶意URL攻击实例，覆盖10个日常场景与7类恶意网站；③引入变异优化算法提升模板质量；④提出URLGuard作为预过滤模块，在保持低延迟的同时显著降低攻击成功率。

**🔧 技术方法**

主要技术：URL结构分析与模板生成、GPT-4o/DeepSeek生成式扩充、变异优化（基于文本梯度与实例优化）、LLM风险评分指标、低秩适配(QLoRA)微调、Web Agent自动化评测（MetaGPT）。

**📊 数据集**

数据集：从公开恶意网站数据集收集7类恶意域名，结合10个真实应用场景生成攻击模板，最终构成61,845个攻击实例；使用公开的LLM API和开源模型进行评测。

**📈 对比分析**

比较方法：对每个模型计算风险评分ℱ(ℳ)，评估攻击成功率；对URLGuard进行对比实验，计算攻击成功率下降幅度；实验显示大多数LLM攻击成功率≥30%，部分模型如GPT‑4o‑mini、Mistral‑small超过90%；URLGuard平均降低81%攻击成功率，表现显著。

**⚠️ 局限性**

局限性：仅覆盖URL结构篡改攻击，未考虑多模态嵌入、动态生成或DNS劫持等高级攻击；防御模型训练数据规模有限，泛化能力待验证；基准仅评估文本输出，未构造真实恶意网页，可能漏掉某些攻击形式。

---

## 550. "Crash Test Dummies" for AI-Enabled Clinical Assessment: Validating Virtual Patient Scenarios with Virtual Learners

**arXiv ID:** 2601.18085 | [PDF](https://arxiv.org/pdf/2601.18085v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 551. Beyond Static Datasets: Robust Offline Policy Optimization via Vetted Synthetic Transitions

**arXiv ID:** 2601.18107 | [PDF](https://arxiv.org/pdf/2601.18107v1)

**作者:** Pedram Agand `[一作]` (Simon Fraser University), Mo Chen `[通讯]` (Simon Fraser University)

**通讯引用:** 11128 | [OpenAlex ID](https://openalex.org/A5100387253)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 MoReBRAC 框架，通过不确定性感知的潜在合成和层次化世界模型扩展离线强化学习的训练数据。

**💡 创新点**

将 LSTM–GRU 递归世界模型与 VAE 地图、灵敏度分析和 MC Dropout 组成的多层不确定性堆栈相结合，实现安全的合成探索。

**🔧 技术方法**

使用 LSTM、GRU、VAE、Monte Carlo Dropout、Prioritized Replay Buffer 以及 ReBRAC/TD3+BC 目标函数。

**📊 数据集**

在 D4RL Gym‑MuJoCo 基准的随机、中等、专家和全回放数据集上评测。

**📈 对比分析**

与 ReBRAC、MORE、TD3+BC、IQL、SAC‑RND 等基线对比，随机和中等数据下平均分最高，专家数据略逊但更稳健。

**⚠️ 局限性**

在近最优数据下合成数据可能导致分布稀释，且模型对长时间滚动仍有限的误差累积。

---

## 552. Computational Framework for Estimating Relative Gaussian Blur Kernels between Image Pairs

**arXiv ID:** 2601.18099 | [PDF](https://arxiv.org/pdf/2601.18099v1)

**作者:** Akbar Saadat `[一作]` `[通讯]` (Iranian railways), Akbar Saadat (Iranian railways)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于高斯模糊核标准差的无训练计算框架，用于从一幅聚焦图像估计另一幅非聚焦图像的相对模糊并重建深度信息。

**💡 创新点**

创新点在于利用解析表达式和离散积分近似，将相对模糊估计转化为矩阵求解，并通过权重矩阵与相似度筛选实现零训练、实时的空间变异模糊估计。

**🔧 技术方法**

主要技术包括高斯模糊模型、离散Riemann求和、向量化矩阵化、权重矩阵W的构造、相邻点相似度过滤及图像重卷积重建。

**📊 数据集**

实验使用了 Real‑MFF、DPDD 真实多焦点数据集以及人工合成的高斯模糊图像，评估了不同分辨率下的性能。

**📈 对比分析**

通过与原始模糊图像计算 MAE 进行比较，合成案例 MAE <1.7%，真实案例 MAE 在 0.014–0.025 之间，显示出低误差和良好鲁棒性；在降采样过程中误差随分辨率下降，证明方法具有可扩展性。

**⚠️ 局限性**

局限性包括忽略镜头畸变、色差、噪声等光学误差；高斯模型仅适用于相对较小的深度变化；需要同视角、同场景的图像对；当实际深度差距较大或模糊程度超过阈值时误差会显著上升。

---

## 553. LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

**arXiv ID:** 2601.18089 | [PDF](https://arxiv.org/pdf/2601.18089v1)

**作者:** Venmugil Elango `[一作]`, Bita Rouhani `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的 Mixture-of-Experts 体系结构——LatentMoE，在 MoE 的专家路径中将输入投影到低维潜在空间，并通过相应扩展专家数量与 Top‑K 路由，降低通信与内存带宽开销，提升 FLOP/参数准确率。

**💡 创新点**

核心创新是：① 低维潜在投影结合专家参数压缩；② 通过压缩比例 α = d/ℓ 同时放大专家数 N 与 Top‑K，使得计算与通信成本保持不变却显著提升模型表达力与稀疏组合多样性；③ 在保持非线性预算不变的前提下，利用潜在空间实现高效路由与权重加载。

**🔧 技术方法**

采用的技术包括：低维潜在投影（下投影/上投影矩阵）、专家并行与全局 Top‑K 路由、FP4/FP8 混合精度训练、Roofline 性能分析、分布式通信模拟、混合 Mamba‑Attention 结构、参数/ FLOP 计数与加速基准、以及对 1T 规模训练进行的推理延迟/吞吐模拟。

**📊 数据集**

实验数据集：大规模预训练使用 1T token（含 1T 长序列上下文）；下游评估基于 MMLU、HumanEval/HumanEval+、MBPP/MBPP+、GSM8K、MATH‑500、RACE、ARC‑Challenge、HellaSwag、Winogrande 等公开数据集。

**📈 对比分析**

比较方法：在相同总参数数与 FLOP 预算下，用相同超参训练基准 MoE 与 LatentMoE；评估验证损失、MMLU、Code、Math、Commonsense 等指标。结果显示：LatentMoE 在 95B/8B active 规模下，MMLU 提升约 4–5%；推理吞吐差距 <6%；在 1T 规模下，通过潜在投影与专家扩展，可在保持相同准确率的前提下减少约 350B 参数，推理速度提升 1.24–3.46 倍。

**⚠️ 局限性**

局限性：① 对低维压缩的有效性依赖于任务特征 rank，压缩过度可能导致质量下降；② 仍需针对硬件的进一步优化（如 CUDA 流、切换到更适合小矩阵的 GEMM 内核）；③ 只验证了 MoE 与 Mamba‑Attention 两种架构，未与其它稀疏/量化方法组合；④ 对超参敏感性虽降低，但仍需调优；⑤ 低延迟场景下仍受路由和专家负载平衡的影响。

---

## 554. CIM-Tuner: Balancing the Compute and Storage Capacity of SRAM-CIM Accelerator via Hardware-mapping Co-exploration

**arXiv ID:** 2601.18070 | [PDF](https://arxiv.org/pdf/2601.18070v1)

**作者:** Jinwu Chen `[一作]` (Southeast University), Zhenhua Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 9497 | [OpenAlex ID](https://openalex.org/A5068626165)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了CIM‑Tuner工具，对SRAM‑CIM加速器的计算与存储容量进行硬件‑映射协同探索

**💡 创新点**

通过矩阵抽象通用加速器模板与细粒度两级映射策略，显著扩展映射策略空间，实现高效计算与存储平衡

**🔧 技术方法**

使用硬件‑映射协同优化、模拟退火搜索、矩阵抽象、指令流编译器及功耗/延迟模型等技术

**📊 数据集**

在BERT‑Large等多种DNN模型上验证，采用标准的BERT、BERT‑Large数据集

**📈 对比分析**

与现有CIM映射方法及SOTA加速器对比，能在相同面积预算下提升约1.58×能效、2.11×吞吐

**⚠️ 局限性**

受限于面积预算、硬件实现复杂度及对不同硬件细节的精细建模需求，仍需进一步实验验证

---

## 555. RouteMoA: Dynamic Routing without Pre-Inference Boosts Efficient Mixture-of-Agents

**arXiv ID:** 2601.18130 | [PDF](https://arxiv.org/pdf/2601.18130v1)

**作者:** Jize Wang `[一作]` (Shanghai Jiao Tong University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 97310 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 RouteMoA，一种通过动态路由、轻量级分数器和评判器来高效实现多模型协同的混合代理框架。

**💡 创新点**

创新点在于引入无推理的查询预测分数器与自评、交叉评判相结合的后验校正机制，实现模型子集的动态选择，并通过性能、成本和延迟的三维平衡实现显著的资源节省。

**🔧 技术方法**

主要技术包括基于 mDeBERTaV3 的嵌入式查询-模型相似度分数器、双重对比损失训练、混合评判器（自评+交叉评）以及基于成本/延迟的模型排序策略。

**📊 数据集**

实验使用覆盖数学、推理、编程、阅读与生物医学等五大能力的 30 个公开数据集，并在 15 余个大规模 LLM 池与 5 个小规模模型池上进行评测，还使用 AGIEval‑Gaokao 作为 OOD 基准。

**📈 对比分析**

与经典 MoA、Sparse MoA 以及无评判的 RouteMoA 进行对比，RouteMoA 在大规模模型池中平均准确率提升 7.3 点、成本下降 89.8%、延迟下降 63.6%；在小规模池中平均准确率为 83.1%，成本下降 81.4%，延迟下降 38.7%；在 OOD 任务中平均准确率提高 1.7 点，成本和延迟分别降低 11.5% 与 24.7%。

**⚠️ 局限性**

局限性在于引入新模型时需要对分数器重新训练（约 25 分钟），并且目前缺乏完全无训练的路由方案。

---

## 556. Understanding Users' Privacy Reasoning and Behaviors During Chatbot Use to Support Meaningful Agency in Privacy

**arXiv ID:** 2601.18125 | [PDF](https://arxiv.org/pdf/2601.18125v1)

**作者:** Mohammad Hadi Nezhad `[一作]` (University of Massachusetts Amherst), Ivon Arroyo `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 3036 | [OpenAlex ID](https://openalex.org/A5008924726)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在模拟ChatGPT界面中加入即时隐私通知面板，探究学生在使用聊天机器人时的敏感信息披露与保护行为及其背后理由

**💡 创新点**

创新性在于提供可选的三种匿名化策略（撤回、伪造、泛化）并将ChatGPT内置隐私控制（模型训练共享与记忆功能）显式展示，以提升用户对敏感信息的认知与主动保护

**🔧 技术方法**

技术实现包括基于GPT‑4o API的聊天模拟、浏览器端本地敏感信息检测引擎、可交互隐私面板、音视频录制与思维记录分析工具

**📊 数据集**

使用自制的包含姓名、地址、SSN等敏感字段的十份任务材料（共约数百条敏感实例），参与者为10名计算机科学本科/硕士学生

**📈 对比分析**

采用无面板与有面板两种实验条件进行对比，发现有面板时隐私意识提升、披露量下降、匿名化使用率上升，研究结果主要为定性分析而非数值指标

**⚠️ 局限性**

局限性包括样本规模小、仅在实验室模拟环境中测试、敏感信息检测范围有限、未覆盖健康与金融数据，且未评估面板在真实ChatGPT使用中的长期影响

---

## 557. Deadline-Aware, Energy-Efficient Control of Domestic Immersion Hot Water Heaters

**arXiv ID:** 2601.18123 | [PDF](https://arxiv.org/pdf/2601.18123v1)

**作者:** Muhammad Ibrahim Khan `[一作]` (Coventry University), James Brusey `[通讯]` (Coventry University)

**通讯引用:** 1857 | [OpenAlex ID](https://openalex.org/A5066204863)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了截止时间感知的家庭浸泡式热水器能耗最小化控制，并实现了Gymnasium仿真环境。

**💡 创新点**

提供统一的可重复实验协议，将基线、MCTS规划和PPO学习在同一物理模型下比较，证明学习策略在能耗上显著优于规划。

**🔧 技术方法**

第一阶热力学模型、Gymnasium API、蒙特卡洛树搜索、Proximal Policy Optimization。

**📊 数据集**

采用无真实数据的仿真，遍历不同初始温度、目标温度和截止步长的参数网格。

**📈 对比分析**

在相同物理与时序设置下记录总能耗，PPO在所有实验中形成最低能耗包络，MCTS略优于基线，PPO在60步时相较基线能耗降低约69%。

**⚠️ 局限性**

仅考虑单一恒定功率开/关控制、缺乏用户需求波动、时间使用价和排放信号等真实场景因素，且MCTS搜索成本较高。

---

## 558. FABLE: Forest-Based Adaptive Bi-Path LLM-Enhanced Retrieval for Multi-Document Reasoning

**arXiv ID:** 2601.18116 | [PDF](https://arxiv.org/pdf/2601.18116v1)

**作者:** Lin Sun `[一作]` (Qiyuan Tech), Xiangzheng Zhang `[通讯]` (Qiyuan Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于LLM生成的多层语义森林索引，并实现了双路径查询以实现多文档推理；

**💡 创新点**

通过让LLM主动组织知识结构、结合层级导航和结构传播的双路径检索，实现了查询自适应的检索与推理；

**🔧 技术方法**

使用LLM驱动的语义分块、树结构构建、向量与树扩展检索、预算自适应融合算法；

**📊 数据集**

在DragonBall、HotpotQA、2Wiki、BrowseComp-plus等合成与真实多跳问答与代理检索基准上进行评估；

**📈 对比分析**

相较于传统RAG、结构化RAG以及全上下文LLM推理，FABLE在完整性、准确率和token效率上均优于SOTA，并能在约8k token下逼近1M token模型；

**⚠️ 局限性**

需要提前进行索引构建，对高度无结构文本或仅依赖关键词检索的场景效果有限。

---

## 559. GLEN-Bench: A Graph-Language based Benchmark for Nutritional Health

**arXiv ID:** 2601.18106 | [PDF](https://arxiv.org/pdf/2601.18106v1)

**作者:** Jiatan Huang `[一作]` (University of Connecticut), Chuxu Zhang `[通讯]` (University of Connecticut)

**通讯引用:** 5262 | [OpenAlex ID](https://openalex.org/A5022275632)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了GLEN-Bench——一个集成饮食、临床与社会经济信息的图‑语言基准，用于风险检测、个性化推荐和可解释问答。

**💡 创新点**

①统一图‑语言基准覆盖三任务；②构建多维度饮食健康知识图谱；③系统评估20+模型，展示图‑LLM融合的最佳效果。

**🔧 技术方法**

采用图神经网络（GCN、GAT、RGCN、HAN、HGT等）、大型语言模型（LLaMA‑3.1、Qwen3、GPT‑4）与图‑RAG混合架构，并进行对比实验。

**📊 数据集**

利用NHANES、FNDDS/WWEIA与USDA Purchase‑to‑Plate价格数据构建健康‑饮食‑社会经济三维知识图谱。

**📈 对比分析**

通过两种训练/验证/测试拆分（60/20/20与70/15/15）对风险检测、推荐和问答进行多指标评估；图‑LLM混合模型在宏F1、AUC、GMean、Recall@20、NDCG@20、H‑Score、PA@20等指标上显著优于单一模型，验证了融合策略的有效性。

**⚠️ 局限性**

受限于标签不平衡导致稀有类识别不足、图构建需手工规则、仅在OUD场景验证，泛化到其他疾病需进一步验证。

---

## 560. Spatial-Conditioned Reasoning in Long-Egocentric Videos

**arXiv ID:** 2601.18100 | [PDF](https://arxiv.org/pdf/2601.18100v1)

**作者:** James Tribble `[一作]` (Clemson University), Abolfazl Razi `[通讯]` (Clemson University)

**通讯引用:** 2171 | [OpenAlex ID](https://openalex.org/A5011987346)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在长时间第一人称视角视频中利用空间感知的视觉语言模型（VLM）进行导航相关查询的表现，并通过对Google Sanpo数据集的精细注释（Sanpo-D）以及深度融合实验进行评估。

**💡 创新点**

提出了专注于空间推理的Sanpo-D细粒度注释数据集，并在不修改模型架构的前提下，通过深度融合和输入层空间先验来提升VLM的空间推理能力。

**🔧 技术方法**

使用多模态视觉语言模型（VILA、LLaVA-OneVision、LLaVA-NeXT-Video、InternVL2、VLM-3R等）进行零样本推理，并将深度估计模型生成的深度图与RGB图像融合作为输入。

**📊 数据集**

利用Google Sanpo数据集的647段视频，并在此基础上生成的Sanpo-D注释集（包含交叉、行人、障碍物和路径上下文四类问题）。

**📈 对比分析**

采用导航路径评估协议，对六个模型进行问答准确率比较，结果显示大模型在整体准确率上领先，而空间化的VLM-3R在安全关键任务（行人检测）上表现最佳；深度融合对大多数模型尤其是障碍物检测有正面提升，但也存在降效现象。

**⚠️ 局限性**

受限于视角漂移、部分可观测性和缺乏长期全局上下文，当前模型在长时序视频中的空间一致性仍不足，且深度融合的效果不稳定，需进一步探索更稳健的空间先验或微调策略。

---

## 561. XGuardian: Towards Explainable and Generalized AI Anti-Cheat on FPS Games

**arXiv ID:** 2601.18068 | [PDF](https://arxiv.org/pdf/2601.18068v1)

**作者:** Jiayi Zhang `[一作]` (University of Hong Kong), Chenxiong Qian `[通讯]` (University of Hong Kong)

**通讯引用:** 928 | [OpenAlex ID](https://openalex.org/A5071764813)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种通用、可解释、低开销的服务器端反作弊系统，专门用于检测第一人称射击游戏中的瞄准助手作弊行为。

**💡 创新点**

创新点在于只使用游戏必备的俯仰角和偏航角构造三种通用轨迹特征（速度、加速度、角度变化），并结合GRU‑CNN模型与SHAP可解释框架实现可解释的作弊检测。

**🔧 技术方法**

技术主要包括：基于俯仰/偏航的轨迹提取与映射、时序特征构造、滑动窗口的GRU‑CNN嵌入生成、随机森林集成、SHAP梯度/树解释器及时间压缩处理。

**📊 数据集**

使用了真实大型数据集：来自5EPlay的Counter‑Strike 2（约3.07 M瞄准操作，31 971条轨迹，5 486玩家）以及Lilith Games提供的Farlight 84数据，公开发布在Zenodo。

**📈 对比分析**

与多种基线（统计阈值、传统机器学习、Hawk等）对比，本文方法在CS 2上召回率达90.7%，FPR仅4.1%，比最新服务器端方案提升12.5%准确率，且推理时间约9.98 s/场，显著低于Hawk的400 s。

**⚠️ 局限性**

局限在于仅针对涉及击杀事件的轨迹，无法检测不造成击杀的作弊或不影响瞄准轨迹的作弊手段；对极低频率游戏和缺乏俯仰/偏航数据的游戏适用性有限。

---

## 562. Addressing LLM Diversity by Infusing Random Concepts

**arXiv ID:** 2601.18053 | [PDF](https://arxiv.org/pdf/2601.18053v1)

**作者:** Pulin Agrawal `[一作]` (Pennsylvania State University), Prasoon Goyal `[通讯]` (Amazon)

**通讯引用:** 3398 | [OpenAlex ID](https://openalex.org/A5072713291)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在LLM提示中加入随机概念（词/句）以提升生成输出的多样性

**💡 创新点**

提出一种轻量级随机注入提示技术，并配套系统化评估协议

**🔧 技术方法**

使用随机词/句生成、列表生成、熵与唯一计数统计、t检验、Python wonderwords库

**📊 数据集**

自生成的100个列表生成问题（每个需生成10项），在Gemma3 4b、Mistral‑Large、Amazon Nova Pro、Claude 3.5 Sonnet四个模型上测试

**📈 对比分析**

对比“常规提示”与“加随机词/句”两种提示，评估唯一计数和熵。实验显示无论是有序还是无序提示，加入随机上下文均显著提升唯一计数和熵（p<0.05）

**⚠️ 局限性**

改进幅度相对有限，主要针对列表生成任务；未涉及输出准确性与其他任务的适用性，随机注入方法可能在特定上下文产生不相关回答

---

## 563. Leveraging Persistence Image to Enhance Robustness and Performance in Curvilinear Structure Segmentation

**arXiv ID:** 2601.18045 | [PDF](https://arxiv.org/pdf/2601.18045v1)

**作者:** Zhuangzhi Gao `[一作]`, Yalin Zheng `[通讯]` (University of Liverpool)

**通讯引用:** 9719 | [OpenAlex ID](https://openalex.org/A5081186911)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计了 PIs-Regressor 与 Topology SegNet 两个模块，直接从原始医学图像学习持久性图像并将其嵌入分割网络，实现血管等曲线结构的分割

**💡 创新点**

首次通过学习可微的持久性图像而非手工构造的拓扑损失，并将拓扑特征融入下采样与上采样过程，显著提升分割的拓扑一致性与鲁棒性

**🔧 技术方法**

采用 Persistence Image 表示、ResNet‑50 编码器、DeConv 反卷积解码器、Dice+交叉熵混合损失、GUDHI 计算 Betti 数等技术

**📊 数据集**

在 DRIVE、ER 以及私有 OPTOS 超宽视野眼底图数据集上进行实验

**📈 对比分析**

与 U‑Net、clDice/cbDice、PH 损失等基线对比，实验表明在 Dice、clDice、mIoU 以及 Betti 误差方面均取得领先，并在曝光、模糊等扰动下保持较高鲁棒性

**⚠️ 局限性**

模型依赖预训练 ResNet‑50 产生的特征，PI 的近似误差可能影响极小尺度结构；目前仅在血管曲线上验证，缺乏对其他形状的推广与理论证明

---

## 564. Spelling Bee Embeddings for Language Modeling

**arXiv ID:** 2601.18030 | [PDF](https://arxiv.org/pdf/2601.18030v1)

**作者:** Markus N. Rabe `[一作]` (Sutter Hill Ventures), Zheren Dong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在词嵌入中加入前16个字节的RoPE编码字符嵌入，构建“拼写蜂”嵌入，改进了大型语言模型在拼写和常规基准上的表现。

**💡 创新点**

创新点在于将子词嵌入与字符级拼写信息相加，几乎不增加参数或计算成本，却显著提升模型的拼写相关能力。

**🔧 技术方法**

采用BPE分词、Transformer+SwiGLU、RoPE、FlashAttention、混合精度训练等技术，并自定义拼写蜂嵌入。

**📊 数据集**

主要使用SlimPajama数据集进行预训练，评估使用lm-evaluation-harness以及自制的拼写基准。

**📈 对比分析**

与基线在相同数据与超参下对比，拼写蜂模型在多项基准上平均提升约0.1–0.3分，等价于约8%更少的计算或数据即可达到同等损失。

**⚠️ 局限性**

限制包括在更大规模模型或不同数据集上效果可能衰减；拼写蜂嵌入需要从零预训练，无法直接迁移至已有模型；对字符级序列操作（如反转）无显著提升。

---

## 565. Sentipolis: Emotion-Aware Agents for Social Simulations

**arXiv ID:** 2601.18027 | [PDF](https://arxiv.org/pdf/2601.18027v1)

**作者:** Chiyuan Fu `[一作]` (Carnegie Mellon University), Mona Diab `[通讯]` (Carnegie Mellon University)

**通讯引用:** 11077 | [OpenAlex ID](https://openalex.org/A5038581447)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Sentipolis框架，构建情绪状态化LLM代理，解决传统模型的情绪遗忘（emotional amnesia）问题；

**💡 创新点**

创新点在于持续的PAD（三维情绪向量）表征、双速情绪动力学（快速与慢速推理）以及情绪‑记忆耦合机制，实现在长周期对话中的情绪连贯与可塑性；

**🔧 技术方法**

使用PAD空间连续表征、EMA双速更新、KNN映射将PAD映射为可解释情绪标签、情绪记忆标签化、LLM生成对话以及多模态评估框架；

**📊 数据集**

实验数据集为25名预设角色的模拟社会环境，使用GPT‑4o‑mini、Groks、GPT‑5.2、Qwen3‑235B‑A22B、Mimo‑v2‑flash、Kimi‑K2‑0905等多种LLM，评估采用SotopiaEval等情感与社交指标；

**📈 对比分析**

通过与无情绪状态基线对比，并使用三类LLM评审者进行评估，结果显示情绪连贯性提升约150%，情感智能（如共情、情绪恰当性）提升30%+，交互沟通质量提升4–70%；网络诊断显示高回报率、聚类性和时间稳定性；

**⚠️ 局限性**

局限性包括规模仅25人、时长12小时、缺乏系统消融实验、未引入真人对照、仅在单一沙盒环境测试，未来需扩大规模、延长周期并进行更细粒度的机制归因。

---

## 566. The Limits of AI Data Transparency Policy: Three Disclosure Fallacies

**arXiv ID:** 2601.18127 | [PDF](https://arxiv.org/pdf/2601.18127v1)

**作者:** Judy Hanwen Shen `[一作]` (Stanford University), Daniel E. Ho `[通讯]` (Stanford University)

**通讯引用:** 15874 | [OpenAlex ID](https://openalex.org/A5058408154)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从制度视角系统评估 AI 数据透明度政策，识别了三大缺陷——规范缺口、执法缺口和影响缺口，并提出技术与政策改进建议。

**💡 创新点**

创新点在于将数据透明度问题归纳为三种“缺口”模型，并结合现行法规与技术验证挑战，构建了从政策目标到必要披露的映射框架。

**🔧 技术方法**

主要采用文献综述与案例分析，讨论成员资格推断、数据溯源、加水印、加密等技术方向；未提出新算法。

**📊 数据集**

未使用实验数据集，而是基于已有法规（如加州 AB 2013、欧盟 AI 法案等）和行业案例进行分析。

**📈 对比分析**

本工作不涉及实验比较，评估方法为定性分析与政策评述，未给出性能指标。

**⚠️ 局限性**

局限性包括缺乏实证验证、未给出可操作的执法机制、对技术实现的依赖有限，且对实际行业执行效果未进行量化评估。

---

## 567. Grasp-and-Lift: Executable 3D Hand-Object Interaction Reconstruction via Physics-in-the-Loop Optimization

**arXiv ID:** 2601.18121 | [PDF](https://arxiv.org/pdf/2601.18121v1)

**作者:** Byeonggyeol Choi `[一作]` (Seoul National University), Jongwoo Lim `[通讯]` (Seoul National University)

**通讯引用:** 15520 | [OpenAlex ID](https://openalex.org/A5061007351)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一个仿真循环优化框架，将视觉对齐的手物体轨迹细化为物理可执行的轨迹。

**💡 创新点**

将轨迹细化视为黑盒优化，使用低维MANO PCA手势参数与三次样条关键帧，并利用CMA-ES进行无梯度优化，从而恢复可观测的接触力与位置。

**🔧 技术方法**

低维MANO PCA手势参数、三次样条关键帧、MuJoCo物理仿真、CMA-ES优化、基于姿态、速度、接触等多项损失。

**📊 数据集**

主要使用DexYCB视觉手物体轨迹数据集。

**📈 对比分析**

与MANIPTRANS基线对比，误差更低、成功率更高、优化时间更短，显著提升物理一致性与视觉一致性。

**⚠️ 局限性**

仅适用于离线、刚性物体的序列，难以处理大重建误差、可变形物体及实时多手/多模态应用。

---

## 568. Beyond Text-to-SQL: Can LLMs Really Debug Enterprise ETL SQL?

**arXiv ID:** 2601.18119 | [PDF](https://arxiv.org/pdf/2601.18119v1)

**作者:** Jing Ye `[一作]`, Xing Chen `[通讯]` (ByteDance Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并实现了第一个面向企业级SQL调试的基准，包含数千条真实复杂SQL脚本及其对应错误注释。

**💡 创新点**

创新点在于：①基于逆向工程的自动错误注入流程，能够高效生成规模大、类型多、难度真实的调试实例；②提出无需执行的评估框架（Exact Match、Graph Match、Modify Better），兼顾语法与语义准确性且资源消耗低。

**🔧 技术方法**

采用了大型语言模型（Claude‑4‑Sonnet、Qwen‑3‑Coder、GPT‑4o等）进行调试实验，同时探索了三种SFT策略（Vanilla、DM‑SFT、Diff‑SFT）和基于代理的迭代调试方案。

**📊 数据集**

数据集为“Squirrel”，共469条语法错误脚本和516条语义错误脚本，平均长度超140行，AST宽度>11，深度>8.7，涵盖金融、电商、医疗等十余业务场景。

**📈 对比分析**

在基准上评估近30个LLM，Claude‑4‑Sonnet最高仅达36.46%（语法）/32.17%（语义）的Graph Match分数；SFT与代理方法虽显著提升，但整体仍未突破40%界限，表明企业SQL调试仍是挑战。

**⚠️ 局限性**

局限性包括：①评估仍基于模拟数据库，可能无法覆盖所有真实数据分布；②逆向注入错误虽多样但仍受模型生成质量限制；③缺乏对多轮交互和真实用户反馈的深入研究。

---

## 569. Demystifying Data-Driven Probabilistic Medium-Range Weather Forecasting

**arXiv ID:** 2601.18111 | [PDF](https://arxiv.org/pdf/2601.18111v1)

**作者:** Jean Kossaifi `[一作]` (NVIDIA), Jan Kautz `[通讯]` (NVIDIA)

**通讯引用:** 40276 | [OpenAlex ID](https://openalex.org/A5056503617)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出Atlas系列模型，通过在压缩的潜在空间中使用标准Transformer实现可扩展的概率天气预报。

**💡 创新点**

创新点在于消除域特定的架构与训练复杂性，使用通用Transformer加上局部投影实现多尺度解耦，并统一适配三种概率建模方法。

**🔧 技术方法**

使用Diffusion Transformer、Stochastic Interpolants、Diffusion Models、CRPS+谱正则化以及Bilinear下采样与Transformer解码器。

**📊 数据集**

基于ERA5 1980–2019年6小时分辨率的75变量数据，测试集为2020年。

**📈 对比分析**

与ECMWF IFS ENS和公开的GenCast对比，Atlas在RMSE/CRPS上多数变量领先，尤其在早期7天内显著优于GenCast；对15天的全程表现优于IFS；性能差距统计显著。

**⚠️ 局限性**

局限包括仅在ERA5上训练，未对其他强基准如AIFS/FNG等做比较；对极端现象的验证有限；diffusion模型在短期/中期平衡需改进噪声调度；对其他大气过程的泛化仍待验证。

---

## 570. CHiRPE: A Step Towards Real-World Clinical NLP with Clinician-Oriented Model Explanations

**arXiv ID:** 2601.18102 | [PDF](https://arxiv.org/pdf/2601.18102v1)

**作者:** Stephanie Fong `[一作]` (Orygen and University of Melbourne), Dominic Dwyer `[通讯]` (Orygen and University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并验证了一个基于 NLP 的临床高危精神病预测与可解释性框架 CHiRPE，能够从 PSYCHS 半结构化访谈记录中自动预测精神病高危状态并生成符合临床认知的 SHAP 解释格式。

**💡 创新点**

创新点包括：①将症状域分割与摘要映射结合，提供临床语义结构；②与临床专家共同设计五种基于 SHAP 的解释呈现（句子摘要、叙事摘要、症状级条形图等），显著提升可解释性；③通过人机协作验证解释效果，促进工具在真实临床环境中的可接受性。

**🔧 技术方法**

技术手段包括：症状域分割（基于 fuzzy string matching）、段落摘要（Mistral‑7B‑Instruct‑v0.3）、BERT/ClinicalBERT/MentalBERT 细调分类器、SHAP 归因与多种可视化（词条条形图、热图、症状级条形图、句子/叙事摘要）。

**📊 数据集**

使用 AMP‑SCZ 研究中的 943 篇英文 PSYCHS 访谈转录（共 581 名 12‑30 岁受试者），按 64%/16%/20% 分离训练、验证与测试。

**📈 对比分析**

与基线模型（无分割/摘要）以及其他公开的深度学习方案对比，CHiRPE 在测试集上实现了 90% 以上准确率，BERT 0.95、ClinicalBERT/MentalBERT 0.97 的 AUC，显著优于基线；在解释性评估中，创新解释格式平均得分 4.2/5，显著高于传统词条/热图。

**⚠️ 局限性**

局限性包括：仅使用英文数据，缺乏多语言验证；模型仅基于 BERT，未探索其他架构；样本主要来自 24 家国际中心，可能缺乏更广泛的人群代表性；未来需进一步验证临床实用性并整合患者视角。

---

## 571. Tail-Latency-Aware Federated Learning with Pinching Antenna: Latency, Participation, and Placement

**arXiv ID:** 2601.18097 | [PDF](https://arxiv.org/pdf/2601.18097v1)

**作者:** Yushen Lin `[一作]` (University of Manchester), Zhiguo Ding `[通讯]` (Nanyang Technological University)

**通讯引用:** 57644 | [OpenAlex ID](https://openalex.org/A5002904166)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在无线联邦学习中使用PINCHING ANTENNA（PASS）技术，通过同时优化客户端抽样和天线位置来最小化期望的准确性达到时间（time-to‑accuracy），并考虑了非 IID 数据带来的统计异质性；

**💡 创新点**

创新点在于：①将物理层的尾延迟放大效应（K‑fold amplification）与统计收敛因子结合，给出 KKT 梯度递推公式，揭示“尾延迟溢价”；②在存在延迟类结构时推导出类内平方根抽样规律和两类阈值相变；③对天线放置问题给出分段包络导数解析并提供全局搜索算法；

**🔧 技术方法**

使用的主要技术包括：顺序统计分析、凸/非凸优化、KKT 条件、阶梯函数分段导数、包络定理、数值根搜索与多起点全局求解；

**📊 数据集**

实验数据集采用 MNIST 与 CIFAR‑10，使用轻量化 CNN 进行同步 FedAvg 训练；

**📈 对比分析**

对比方法包括传统固定天线放置与均匀抽样、PASS 随机放置、PASS 联合优化；实验显示 PASS 联合优化在相同墙钟预算下取得更高准确率（如 MNIST 95% vs 77%，CIFAR‑10 61% vs 49%），并显著压缩极端尾延迟；

**⚠️ 局限性**

限制与不足：仅考虑单个 PINCHING ANTENNA，假设 LoS 传播且无多径；同步 FL 的严格同步假设可能不适用于大规模异构网络；内部优化需要全局解，计算成本随客户端数增长；未来可探索多 PA 协同与非同步 FL。

---

## 572. From LLMs to LRMs: Rethinking Pruning for Reasoning-Centric Models

**arXiv ID:** 2601.18091 | [PDF](https://arxiv.org/pdf/2601.18091v1)

**作者:** Longwei Ding `[一作]` (Institute of Digital Twin), Xiaoyu Shen `[通讯]` (Institute of Digital Twin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在对比指令追随型（LLM-instruct）和推理增强型（LLM-think）大型语言模型的结构剪枝效果时，采用对齐校准与恢复数据的控制实验框架，对三类剪枝策略（静态深度剪枝、静态宽度剪枝、动态深度剪枝）进行了系统评估。

**💡 创新点**

创新点包括：①将剪枝校准与恢复数据与模型原始训练分布严格对齐，从而隔离剪枝本身的影响；②揭示不同模型范式对剪枝策略的敏感性差异，证明动态剪枝在推理增强模型中难以优化；③通过实验证实静态宽度剪枝在生成与推理任务中更稳健，静态深度剪枝在分类任务中更有效。

**🔧 技术方法**

主要技术手段包括：结构化剪枝方法（SLEB、ShortGPT、Shortened‑PPL、Shortened‑Taylor、LLM‑Pruner、SliceGPT、MOD、D‑LLM、SkipGPT）、重要性评估指标（BI、PPL、Taylor）、基于LoRA的微调与路由模块训练，以及在17项分类、生成与推理基准上的性能评估。

**📊 数据集**

数据集：指令追随模型使用Tulu指令微调语料；推理增强模型使用OpenThoughts推理语料；评估基准涵盖 BoolQ、PIQA、HellaSwag、WinoGrande、ARC‑Easy/Challenge、OpenBookQA、IFEval、TruthfulQA、PopQA、HumanEval+、MATH、AIME、LiveCodeBench、GPQA、JEE、GSM8K 等 17 项任务。

**📈 对比分析**

比较方法：将各剪枝策略在同一模型与任务上进行基线对齐实验，记录平均性能下降（AD）和绝对分数；结果表明：在 LLM‑instruct 中，动态剪枝（尤其是 SkipGPT）在分类与生成任务上实现几乎无损压缩；在 LLM‑think 中，静态剪枝（尤其是宽度剪枝）保持更高的推理准确率，而动态剪枝在 20%–60% 稀疏度下普遍导致语义退化，MOD 在这类任务中表现为唯一可行的动态方法；静态深度剪枝在分类任务上优于宽度剪枝，但在生成与推理任务上更易退化。

**⚠️ 局限性**

限制：实验仅在 Llama‑8B 规模模型上完成，未验证所观察到的剪枝行为是否能推广到更大或更小的模型；此外，动态剪枝在推理增强模型上的优化难度仍未得到完整解决。

---

## 573. Comparison requires valid measurement: Rethinking attack success rate comparisons in AI red teaming

**arXiv ID:** 2601.18076 | [PDF](https://arxiv.org/pdf/2601.18076v1)

**作者:** Alexandra Chouldechova `[一作]` (Microsoft Research), Hanna Wallach `[通讯]` (Microsoft Research)

**通讯引用:** 12377 | [OpenAlex ID](https://openalex.org/A5046348432)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对 AI 红队（尤其是 jailbreak）中常用的攻击成功率（ASR）进行分析，指出其在比较系统安全性或攻击方法效能时往往缺乏实证依据。

**💡 创新点**

创新点在于将社会科学测量理论与推断统计相结合，提出“概念一致性（conceptual coherence）”和“测量有效性（measurement validity）”这两个必要条件，并通过概率威胁模型对 ASR 的估计量进行形式化。

**🔧 技术方法**

主要技术包括：构建概率威胁模型、对攻击成功率进行抽象化为估计量（estimand）、使用 Top‑1、one‑shot 等聚合方式进行理论与实证比较、以及评估判别器（judge）误差对 ASR 的偏倚影响。

**📊 数据集**

使用的数据集主要包括：MaliciousInstruct（100 条基础有害提示）、160 条来自其他工作公开的有害提示集，以及在实验中自行采样的 49 种解码配置下的 392 条生成样本。

**📈 对比分析**

比较方法：在不同聚合方式（Top‑1 vs one‑shot）和解码配置（温度、采样数）下对同一模型进行重复实验，结果表明许多先前报告的“更优”攻击方法，其提升往往源自采样次数或聚合方式的改变，而非本质攻击策略更强，整体性能提升并不显著。

**⚠️ 局限性**

局限性：需要极细粒度的实验日志来验证概念一致性；许多现有研究缺乏对判别器误差的分层报告；本文聚焦于 jailbreak，无法直接推广到所有红队攻击场景；测量有效性分析仍需进一步量化方法来修正差异化误分类。

---

## 574. Diffusion Model-based Reinforcement Learning for Version Age of Information Scheduling: Average and Tail-Risk-Sensitive Control

**arXiv ID:** 2601.18069 | [PDF](https://arxiv.org/pdf/2601.18069v1)

**作者:** Haoyuan Pan `[一作]` (Shenzhen University), Tse-Tin Chan `[通讯]` (Education University of Hong Kong)

**通讯引用:** 248 | [OpenAlex ID](https://openalex.org/A5047023625)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本文针对多用户无线状态更新系统，提出了基于扩散模型和分布式强化学习的 VAoI 调度算法，用以在满足长期传输成本约束的前提下最小化平均版本信息年龄（VAoI）并降低其尾部风险。

**💡 创新点**

创新点包括：①将扩散模型嵌入 Soft Actor-Critic（SAC）框架，实现可表达的动作生成；②引入量化分布式评论器（QR‑DQN）以学习完整的回报分布，从而通过 CVaR 进行风险感知优化；③将上述两项技术结合，形成 RS‑D3SAC，首次实现对 VAoI 尾部风险的系统化控制。

**🔧 技术方法**

技术主要有：深度扩散模型（DDPM）用于生成策略；Soft Actor‑Critic（SAC）和其双评论器架构；分布式强化学习（QR‑DQN）用于估计回报分布；CVaR 作为风险度量；双重时间尺度的 Lagrange 多重目标优化。

**📊 数据集**

实验采用仿真生成的多用户状态更新数据，用户数 N=20，包生成率 r=0.75，链路成功率 p=0.9，传输成本约束 η_max=0.85；无公开数据集，仅使用本实验所构建的离散事件模拟器。

**📈 对比分析**

与 PPO、DQN、Rainbow、SAC 等基线对比，D2SAC 在平均 VAoI 上优于所有方法；RS‑D3SAC 在 CVaR（α=0.75）上显著低于对手且平均 VAoI 仍保持优良，表明既降低了尾部风险又不损失整体性能。

**⚠️ 局限性**

局限性包括：①扩散模型和分布式评论器导致训练和推理时间显著增加；②算法对超参数（扩散步数、量化分布数、CVaR 置信度）敏感；③实验仅基于离散仿真，未验证在真实网络环境中的鲁棒性。

---

## 575. Grounded Concreteness: Human-Like Concreteness Sensitivity in Vision-Language Models

**arXiv ID:** 2601.18065 | [PDF](https://arxiv.org/pdf/2601.18065v1)

**作者:** Aryan Roy `[一作]`, Christopher J. MacLellan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了ACL会议论文的模板使用说明和格式规范。

**💡 创新点**

创新点在于统一了排版规则，并提供了完整的LaTeX模板示例。

**🔧 技术方法**

使用LaTeX语言和相应的ACL样式文件（acl_natbib.sty等）。

**📊 数据集**

未使用任何研究数据集。

**📈 对比分析**

本文不涉及实验比较，主要关注格式和排版。

**⚠️ 局限性**

限制在于仅为格式说明文档，不包含研究方法或实验结果。

---

## 576. Resonant Sparse Geometry Networks

**arXiv ID:** 2601.18064 | [PDF](https://arxiv.org/pdf/2601.18064v1)

**作者:** Hasi Hays `[一作]` `[通讯]` (University of Arkansas), Hasi Hays (University of Arkansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于超bolic空间嵌入、输入激活点与软阈值传播的Resonant Sparse Geometry Network（RSGN），实现自组织、输入相关的稀疏层级连接。

**💡 创新点**

创新点在于将节点放置于Poincaré球模型中，使连接强度随几何距离指数衰减；结合两时尺度学习（梯度下降+Hebbian塑性）和局部抑制，形成动态稀疏自适应路由。

**🔧 技术方法**

使用技术包括超bolic嵌入、基于距离的权重衰减、软阈值激活、局部抑制、两时尺度学习（梯度+Hebbian）以及可微软化的稀疏化。

**📊 数据集**

使用合成层级分类任务（20类、长度64）和长程依赖任务（10类、长度128）进行实验，并与Transformer、Sparse Transformer、LSTM、MLP等基线模型对比。

**📈 对比分析**

与Transformer、Sparse Transformer等比较，RSGN在层级分类上以41,672参数获得23.8%准确率（比Transformer 10×参数压缩），在长程依赖上以40,382参数达到96.5%准确率（比Transformer 15×参数压缩），虽然准确率略低于Transformer，但显著提高了参数效率。

**⚠️ 局限性**

主要局限包括：绝对准确率仍低于Transformer；稀疏动态计算在现有GPU硬件上不易充分利用；双时尺度学习需要精细调参；未在大规模真实数据集上验证性能。

---

## 577. Cross-Domain Transfer with Self-Supervised Spectral-Spatial Modeling for Hyperspectral Image Classification

**arXiv ID:** 2601.18088 | [PDF](https://arxiv.org/pdf/2601.18088v1)

**作者:** Jianshu Chao `[一作]` (Fujian Institute of Research on the Structure of Matter), Wei Yao `[通讯]` (State Key Laboratory of Regional and Urban Ecology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种无源标签自监督跨域迁移框架，学习可迁移的光谱-空间联合表征并实现少量样本下的高效适配。

**💡 创新点**

创新点在于引入S²Former双分支Transformer与双向跨注意力实现光谱-空间协同建模、频域约束FDC提升细节辨识、以及DAFT扩散对齐微调提升跨域迁移鲁棒性。

**🔧 技术方法**

技术包括自监督掩码建模、双分支Transformer、频域约束（rFFT高频损失）、扩散对齐蒸馏（DAFT）以及多尺度特征融合。

**📊 数据集**

实验使用四个公共高光谱数据集：Pavia Center、Houston2013、Pavia University、Salinas，并构造四种跨域迁移任务。

**📈 对比分析**

与SSTN、CTF、DEMAE、DCFSL、FDFSL、HyMuT等方法比较，在少样本跨域任务中均实现了更高的整体准确率、平均准确率和Kappa值，提升幅度约2–3%。

**⚠️ 局限性**

局限在于仅利用目标域已有语义信息，未进一步挖掘无标签目标域的自监督特征，未来可结合对比学习等技术提升跨域迁移鲁棒性。

---

## 578. Generative Chain of Behavior for User Trajectory Prediction

**arXiv ID:** 2601.18213 | [PDF](https://arxiv.org/pdf/2601.18213v1)

**作者:** Chengkai Huang `[一作]` (University of New South Wales and Macquarie University), Lina Yao `[通讯]` (University of New South Wales and CSIRO Data61)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Generative Chain of Behavior（GCB）框架，用于多步预测用户行为轨迹，取代传统的单步推荐；

**💡 创新点**

创新点在于：①用RQ‑VAE结合k‑means构造语义ID的离散表示；②在该语义空间上训练自回归Transformer生成连贯的多步轨迹；③采用Beam Search在语义ID空间生成未来序列并映射回原始物品；

**🔧 技术方法**

采用的技术包括：Residual Quantized Variational AutoEncoder（RQ‑VAE）+ k‑means初始化、T5‑style encoder–decoder Transformer、Beam Search、交叉熵训练和Adam优化；

**📊 数据集**

使用Amazon的Beauty和Cell Phones & Accessories两个真实电商评论数据集；

**📈 对比分析**

与FPMC、SASRec、STOSA、PreDiff、DCRec等基线进行对比，采用多步HR@K、NDCG@K、SeqHR、SeqNDCG等指标。GCB在大多数多步预测任务中均取得最优或接近最优表现，尤其在Cell Phones数据集上优势明显；

**⚠️ 局限性**

局限性包括：①随着预测步长延长，性能仍显著下降；②离散语义ID映射可能出现冲突或映射不到物品的情况；③模型训练与推理成本相对较高，尤其Beam Search的计算开销大；

---

## 579. HeterCSI: Channel-Adaptive Heterogeneous CSI Pretraining Framework for Generalized Wireless Foundation Models

**arXiv ID:** 2601.18200 | [PDF](https://arxiv.org/pdf/2601.18200v1)

**作者:** Chenyu Zhang `[一作]` (National Engineering Research Center for Mobile Network Technologies, Beijing University of Posts and Telecommunications), Xiaofeng Tao `[通讯]` (National Engineering Research Center for Mobile Network Technologies, Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 HeterCSI 统一无线基础模型预训练框架，兼顾多尺度多场景通道状态信息 (CSI) 的高效训练与零样本泛化。

**💡 创新点**

核心创新在于：① 发现尺度异质性导致梯度冲突而场景多样性有助收敛；② 通过分桶排序与可调缩放自适应批处理减少填充与梯度冲突；③ 双重掩码（MAE + 注意力掩码）隔离填充噪声。

**🔧 技术方法**

采用 Vision Transformer 结构，MAE 自监督重建，注意力掩码，分桶排序+随机调度的自适应批处理。

**📊 数据集**

使用 QuaDRiGa 生成的 40 个多尺度多场景 CSI 数据集（含 12 个独立零样本测试集），涵盖 Indoor、RMa、UMa、UMi 等场景与 1.8–42 GHz 频段。

**📈 对比分析**

与 LSTM、Transformer、LLM4CP、BERT4MIMO、WiFo 等全景与零样本基线对比，零样本下平均 NMSE 下降 7.19 dB（重建）、4.08 dB（时域）和 5.27 dB（频域），相较最佳全景模型平均分别降低 1.33、1.58、3.14 dB；训练时间比全局混洗快 53%。

**⚠️ 局限性**

局限在于：① 仍需对极端尺度差异或极少量数据场景进行更细粒度的动态桶分配；② 双掩码与自适应批处理的超参数调优复杂；③ 对未来更高频段（THz）或多用户 MIMO 结构的迁移性尚未验证。

---

## 580. UTune: Towards Uncertainty-Aware Online Index Tuning

**arXiv ID:** 2601.18199 | [PDF](https://arxiv.org/pdf/2601.18199v1)

**作者:** Chenning Wu `[一作]` (Fudan University), X. Sean Wang `[通讯]` (Fudan University)

**通讯引用:** 11768 | [OpenAlex ID](https://openalex.org/A5100389006)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了UTune，一种基于不确定性感知的在线索引调优框架，利用操作符级学习模型对索引收益进行估算并通过不确定性校正提升成本估计精度。

**💡 创新点**

创新点在于引入操作符级CAM预测器、联合量化数据与模型不确定性、将不确定性融入成本校正与索引选择，解决了在线调优中数据匮乏和工作负载漂移的难题。

**🔧 技术方法**

技术手段包括多分类CAM预测、Monte Carlo Dropout与熵相结合的混合不确定性度量、基于不确定性加权的价值函数以及改进的ϵ-贪婪索引选择策略。

**📊 数据集**

实验使用了TPC‑H、TPC‑DS和JOB三大公开基准数据库，并构造了静态、连续、周期和循环漂移等多种工作负载。

**📈 对比分析**

与AutoIndex、HMAB、Indexer++和SWIRL等SOTA在线调优器对比，UTune在所有基准上均能提升5–12%的工作负载执行时间，且索引探索开销更低、收敛更快。

**⚠️ 局限性**

局限性包括对离线训练的依赖、主要关注OLAP场景、对索引维护成本考虑有限，以及迁移到不同DBMS需要重新定义操作符成本公式。

---

## 581. Multi-Perspective Subimage CLIP with Keyword Guidance for Remote Sensing Image-Text Retrieval

**arXiv ID:** 2601.18190 | [PDF](https://arxiv.org/pdf/2601.18190v1)

**作者:** Yifan Li `[一作]` (Qinghai University), Jianqiang Huang `[通讯]` (Qinghai University)

**通讯引用:** 6214 | [OpenAlex ID](https://openalex.org/A5079865276)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 MPS-CLIP，一种参数高效框架，通过 LLM 提取关键词、SamGeo 生成子视角，并结合 G^2A 适配器和 MPR 模块，实现关键字引导的多视角细粒度跨模态对齐；

**💡 创新点**

创新点在于：① 用关键字驱动的多视角对齐替代传统粗粒度全局匹配；② 采用轻量化 G^2A 适配器捕获全局语义并增强长程依赖；③ 结合最大相似度视角的多视角对比与加权三元组损失，提升细粒度匹配鲁棒性；

**🔧 技术方法**

使用技术包括 CLIP 预训练模型、LLM（如 ChatGPT）关键词抽取、Segment Anything（SamGeo）子视角生成、G^2A 适配器、Multi‑Perspective Representation（MPR）模块、混合对比与三元组损失、AdamW 等优化器；

**📊 数据集**

评测数据集为 RSICD 和 RSITMD；

**📈 对比分析**

与基于 CLIP 的 SOTA 方法（SkyCLIP、UniAdapter、PE‑RSITR、HarMA 等）以及全量微调 CLIP 进行对比，MPS-CLIP 在 RSICD 与 RSITMD 的 mR 分别提升到 35.18% 与 48.40%，均比对手高约 1–2 点；

**⚠️ 局限性**

局限性包括：依赖 LLM 与 SamGeo 的生成质量；仅验证了检索任务，未扩展到其他密集预测任务；对极端噪声或不完整文本的鲁棒性仍待进一步研究。

---

## 582. VIBEVOICE-ASR Technical Report

**arXiv ID:** 2601.18184 | [PDF](https://arxiv.org/pdf/2601.18184v1)

**作者:** Zhiliang Peng `[一作]` (Microsoft Research), Furu Wei `[通讯]` (Microsoft Research)

**通讯引用:** 32515 | [OpenAlex ID](https://openalex.org/A5014662947)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种单通道、一次性处理长达60分钟音频的统一语音理解框架VibeVoice，集成了语音识别、说话人归属和时间戳生成；

**💡 创新点**

核心创新在于：①采用7.5 Hz超低帧率双词化器将整小时音频压缩至可在LLM上下文窗口内一次性处理的序列；②将长形式转写改为生成式Rich Transcription，实时输出说话人身份、时间戳与内容；③引入提示式上下文注入机制，支持自定义热词、背景描述，提高领域专用词与多语种切换的识别；

**🔧 技术方法**

技术手段包括：预训练的Acoustic和Semantic双词化器、基于Qwen 2.5等decoder‑only LLM的生成式模型、GPT‑5用于合成脚本与上下文、TTS合成多说话人音频、GPT‑Audio用于非语音段标注，以及vLLM等高效推理框架；

**📊 数据集**

使用的数据集包括：预训练的多说话人数据（AMI、AliMeeting、AISHELL‑4等）、高质量基准（MLC‑SLM、Fisher、Muse）、自研合成数据（≈6000小时中文/英文/中英混合），以及评测时的多语言会议、广播等公开数据；

**📈 对比分析**

与Gemini‑2.5‑Pro和Gemini‑3‑Pro等多模态大模型对比，VibeVoice在所有评测数据集上均取得更低的说话人误差率（DER）和时序对齐WER（tcpWER），并在cpWER上有11/16场景表现最佳，整体在说话人建模、时间同步与多语言转写上优于现有技术；

**⚠️ 局限性**

局限性包括：SFT阶段主要覆盖英语、中文和双语混合，低资源语言性能可能下降；模型未显式处理重叠语音，当前会倾向于识别主导说话者，次要信息可能被遗漏。

---

## 583. FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning

**arXiv ID:** 2601.18150 | [PDF](https://arxiv.org/pdf/2601.18150v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 584. Fine-Grained Emotion Detection on GoEmotions: Experimental Comparison of Classical Machine Learning, BiLSTM, and Transformer Models

**arXiv ID:** 2601.18162 | [PDF](https://arxiv.org/pdf/2601.18162v1)

**作者:** Ani Harutyunyan `[一作]` (American University of Armenia), Sachin Kumar `[通讯]` (American University of Armenia)

**通讯引用:** 5042 | [OpenAlex ID](https://openalex.org/A5032948428)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对GoEmotions数据集进行多标签细粒度情感识别实验，比较了逻辑回归、BiLSTM+注意力和BERT三类模型。

**💡 创新点**

首次在同一实验框架下对不同容量模型进行统一评测，展示BERT在稀有情绪上的显著优势，并提出多标签阈值校准与情感共现建模的未来方向。

**🔧 技术方法**

采用TF-IDF+逻辑回归、BiLSTM+注意力、以及BERT微调（多标签头+交叉熵+焦点损失）三种技术。

**📊 数据集**

使用Google Research发布的GoEmotions 28类情感手工标注的Reddit评论数据集。

**📈 对比分析**

通过Micro‑F1、Macro‑F1、Hamming Loss、Subset Accuracy等多指标进行比较，BERT在Macro‑F1 0.49、Hamming 0.036、Subset Accuracy 0.36等方面领先，逻辑回归在Micro‑F1 0.51上略高。

**⚠️ 局限性**

主要限制包括类不平衡对阈值设定的影响、BiLSTM缺乏上下文嵌入导致性能不足，以及受限的GPU资源未能深入探索更大模型。

---

## 585. RareAlert: Aligning heterogeneous large language model reasoning for early rare disease risk screening

**arXiv ID:** 2601.18132 | [PDF](https://arxiv.org/pdf/2601.18132v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 586. Agentic Very Long Video Understanding

**arXiv ID:** 2601.18157 | [PDF](https://arxiv.org/pdf/2601.18157v1)

**作者:** Aniket Rege `[一作]` (Reality Labs Research at Meta), Hyo Jin Kim `[通讯]` (Reality Labs Research at Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了带时间标签的实体场景图，并在此基础上设计了agentic框架，实现对长时 egocentric 视频的跨模态推理与问答。

**💡 创新点**

创新点在于引入时间意识的实体图表示，结合规划 agent 与视觉/音频/图结构检索工具实现多跳跨模态推理，并在长期视频上达到最新 SOTA。

**🔧 技术方法**

使用 LLM 提取实体与关系、SQLite+向量数据库进行检索、视觉检索工具、音频检索工具、分析器及 VQA 代理等技术。

**📊 数据集**

使用 EgoLifeQA 与 Video‑MME（Long）数据集进行实验。

**📈 对比分析**

与基线 MLLM、RAG 及现有 agentic 方法对比，取得 32%/39.7% 等显著提升，成为这些评测的最新 SOTA。

**⚠️ 局限性**

主要限制是实体图构建依赖上游感知与语言模型的准确性，若使用自动对话者识别会导致误差，且整体处理时长仍相对较高。

---

## 587. EndoExtract: Co-Designing Structured Text Extraction from Endometriosis Ultrasound Reports

**arXiv ID:** 2601.18154 | [PDF](https://arxiv.org/pdf/2601.18154v1)

**作者:** Haiyi Li `[一作]` (University of Adelaide), Hsiang-Ting Chen `[通讯]` (University of Adelaide)

**通讯引用:** 2207 | [OpenAlex ID](https://openalex.org/A5036805602)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了基于本地LLM的 EndoExtract 系统，用于从子宫内膜异位症超声报告中提取结构化数据，并提供人机协同验证界面。

**💡 创新点**

通过可信度分级的选择性审查界面、自动在 PDF 中高亮证据并关联表格、批处理与人机分离的异步验证流程，以及语义标准化处理，实现了高效、可解释的人工审核工作流程。

**🔧 技术方法**

本地部署的 GPT‑OSS‑20B（通过 Ollama）、前端 React/JS 实现选择性审查与证据高亮、批处理控制、语义归一化规则与 UI 交互。

**📊 数据集**

使用阿德莱德大学内部收集的数千份子宫内膜异位症超声报告 PDF，包含多种模板和术语变体。

**📈 对比分析**

通过实地工作坊和思考实验与传统全手工抽取流程对比，系统在验证时间上节省约 30‑50%，抽取准确率与人工抽取相当，并在可解释性和用户满意度方面优于传统方法。

**⚠️ 局限性**

局限性包括：对解释性字段的可信度分割可能不适用于其他临床域；缺乏对过度跳读、模糊或不典型病例的防护机制；依赖本地 LLM 与手工标注，扩展性受限；未能有效检测模型幻觉和误报。

---

## 588. Enhance the Safety in Reinforcement Learning by ADRC Lagrangian Methods

**arXiv ID:** 2601.18142 | [PDF](https://arxiv.org/pdf/2601.18142v1)

**作者:** Mingxu Zhang `[一作]`, Ying Sun `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文阐述了ICLR 2026 论文提交的完整格式规范和排版要求，涵盖标题、作者、摘要、章节标题、引用、图表、脚注及 PDF/PS 输出细节。

**💡 创新点**

创新点在于系统化整理了 NeurIPS 风格的细节，强调严格遵守风格文件中的尺寸、字体与行距参数，并为作者提供了可直接使用的 LaTeX 模板。

**🔧 技术方法**

主要技术手段是 LaTeX 的自定义宏包与样式文件（iclr2026.sty、iclr2026cvpr.sty 等），以及对 PDF/PS 输出的兼容性设置。

**📊 数据集**

本文不涉及实验数据集，纯粹为排版规范说明。

**📈 对比分析**

本规范本身不做方法对比，唯一的“性能”是通过严格遵守规定确保审稿人对格式的一致性与可读性。

**⚠️ 局限性**

局限性在于缺乏对实际科研内容的讨论，读者只能获得排版指导，无法了解任何科学贡献或实验结果。

---

## 589. ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants

**arXiv ID:** 2601.18225 | [PDF](https://arxiv.org/pdf/2601.18225v1)

**作者:** Pei Wang `[一作]` (Alibaba Group), Bo Zheng `[通讯]` (Alibaba Group)

**通讯引用:** 16506 | [OpenAlex ID](https://openalex.org/A5050479679)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ShopSimulator，一个集成大规模中文商品库、多轮对话、个性化与细粒度商品区分的仿真环境，支持 LLM 代理的评估与训练。

**💡 创新点**

整合中文电商数据、结构化用户画像与多维奖励，填补现有英文环境缺乏个性化与训练支持的空白，并通过 SFT+RL 组合显著提升性能。

**🔧 技术方法**

利用 LLM 模拟用户、GRPO 强化学习与监督微调（SFT）相结合的训练策略。

**📊 数据集**

采集淘宝 1.34M 商品，构建 28K 任务并配备 4.7K 结构化个性化用户画像。

**📈 对比分析**

与 WebShop、DeepShop 等对比，闭源 LLM 如 GPT‑5 的全成功率仅 32%；在 Qwen3‑8B 上，SFT+RL 使全成功率提升至约 39%。

**⚠️ 局限性**

用户画像为人工合成，RL 仅使用 GRPO，且仅支持文本交互，未覆盖多模态信息与最新 RL 算法。

---

## 590. PaperTok: Exploring the Use of Generative AI for Creating Short-form Videos for Research Communication

**arXiv ID:** 2601.18218 | [PDF](https://arxiv.org/pdf/2601.18218v1)

**作者:** Meziah Ruby Cristobal `[一作]` (University of Washington), Gary Hsieh `[通讯]` (University of Washington)

**通讯引用:** 3898 | [OpenAlex ID](https://openalex.org/A5060931546)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个名为PaperTok的基于生成式AI的端到端工具，帮助科研人员将学术论文快速转化为短视频，用于公众传播。

**💡 创新点**

创新点在于将大语言模型与文本到视频模型、语音合成模型结合，构建人机协作工作流，自动生成钩子、脚本、故事板、语音及视觉内容，同时提供可编辑的信任与署名层面，显著降低科研者创作门槛。

**🔧 技术方法**

使用的技术包括Google Gemini 2.5 Flash LLM、Veo 2 文本到视频模型、Gemini 2.5 Flash Preview TTS语音模型，以及FFmpeg视频拼接、SvelteKit前端与Python后端服务。

**📊 数据集**

实验数据集为18名发表论文的HCI研究者上传的论文（共计数不等）和100名来自Prolific的受众样本，以及从三篇CHI 2025最佳论文生成的对比视频（PDFtoBrainrot、SciSpace、PaperTok）。

**📈 对比分析**

通过用户研究和问卷调查的混合方法，PaperTok生成的视频在吸引力、娱乐性、信息性等11维度上显著高于对比工具，且在可信度、准确性等信息质量维度与SciSpace持平、优于PDFtoBrainrot；研究者使用时平均耗时约20分钟，满意度高。

**⚠️ 局限性**

局限性包括：文本到视频模型生成的视觉质量不一致、缺乏对细粒度控制、仅支持TTS语音、评估聚焦于HCI论文和早期研究者，未验证在更广泛学科或更大规模真实环境中的效果。

---

## 591. InkIdeator: Supporting Chinese-Style Visual Design Ideation via AI-Infused Exploration of Chinese Paintings

**arXiv ID:** 2601.18193 | [PDF](https://arxiv.org/pdf/2601.18193v1)

**作者:** Shiwei Wu `[一作]` (Sun Yat-sen University), Zhenhui Peng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 646 | [OpenAlex ID](https://openalex.org/A5081718999)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了 InkIdeator，一个利用多模态大模型对 16,315 幻灯画进行文化符号、情感、构图、风格标注后，为中国风视觉设计师提供符号推荐、例图检索、情景分析与 AI 生成草图的完整创意支持系统。

**💡 创新点**

创新点在于构建了基于大模型的中国画维度标注数据集，并将文化符号、情感、构图、风格四维标签嵌入交互式工具中，以支持设计师在创意探索中的高效检索、分析和可视化。

**🔧 技术方法**

采用 GPT‑4o‑mini 进行多模态标注、ChatGPT 及 Midjourney 进行文本到图像生成，并结合 Flask‑React 前端实现交互。

**📊 数据集**

使用了 16,315 只在 RedNote 等平台收集的中国画图像，并通过模型与专家评审得到的文化符号、情感、构图、风格标签。

**📈 对比分析**

在 12 名参与者的 within‑subject 研究中，InkIdeator 在创意支持、探索效率和可视化速度上显著优于仅提供检索与生成的基线系统；虽然创意质量差异不显著，但用户体验、组织搜索和快速迭代得分均提升。

**⚠️ 局限性**

局限包括样本量小、实验时间短、生成图像在构图与笔法上的偏差以及对中国画知识的依赖性高，导致部分标签错误和文化误读。

---

## 592. Smooth, Sparse, and Stable: Finite-Time Exact Skeleton Recovery via Smoothed Proximal Gradients

**arXiv ID:** 2601.18189 | [PDF](https://arxiv.org/pdf/2601.18189v1)

**作者:** Rui Wu `[一作]` (University of Science and Technology of China), Yongjun Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18371 | [OpenAlex ID](https://openalex.org/A5100376886)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种新的混合阶层无环约束（AHOC）及其平滑近似的优化算法 SPG‑AHOC，实现了在有限迭代内精确恢复 DAG 结构。

**💡 创新点**

通过构造满足三大稳定性公理并证明不可兼容定理，首次实现了在连续优化中得到完全稀疏、无环且无需后处理阈值的结构恢复。

**🔧 技术方法**

采用混合阶层核心、归一化尺度自适应、平滑绝对值近似以及基于 Proximal‑LS 的加速梯度方法，利用拓扑锁定实现精确零检测。

**📊 数据集**

在从稀疏 Erdős–Rényi 与近循环线性高斯结构方程模型生成的合成数据以及真实生物网络 Sachs 数据上进行实验。

**📈 对比分析**

与 NOTEARS、DAGMA、AAC 等基线对比，SPG‑AHOC 在 SHD 与稀疏度上均优于或持平，且在中小维度下速度更快，能够输出完全满足无环约束的稀疏矩阵。

**⚠️ 局限性**

受限于近似矩阵指数计算的 O(d³) 复杂度、对不可代表性条件的依赖以及在极近循环结构下可能出现的数值不稳定，限制了在更高维或高度相关数据上的适用性。

---

## 593. Lip-Siri: Contactless Open-Sentence Silent Speech with Wi-Fi Backscatter

**arXiv ID:** 2601.18177 | [PDF](https://arxiv.org/pdf/2601.18177v1)

**作者:** Ye Tian `[一作]` (University of Science and Technology of China), Xiang-Yang Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18112 | [OpenAlex ID](https://openalex.org/A5100341802)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种利用Wi‑Fi回波散射技术实现的全无接触、开放句子识别的沉默语音接口（Lip‑Siri），通过频移回波标签提取唇部运动信号，并将连续信号映射为可变长度子词序列。

**💡 创新点**

创新点在于：①首次使用Wi‑Fi回波散射实现无体贴装、低功耗的唇读感知；②引入词表引导的子词解码框架，实现开放句子识别而非传统的闭集命令分类；③结合频移标签、运动干扰抑制、单元分割聚类、自监督预训练以及Transformer编码-解码的端到端流水线。

**🔧 技术方法**

核心技术包括：频移回波标签（分离目标散射并抑制非唇部运动干扰）、基于能量与VMD的唇动信号预处理、基于特征聚类的单位分割与自监督预训练、以及基于子词词典的Transformer编码器‑解码器+束搜索解码。

**📊 数据集**

数据集：基于15名志愿者共收集 11,700 条样本，涵盖 340 条句子、3,398 个单词，词表来源为美国国务院《Everyday Conversation Handbook》；此外还使用公开视觉 SSI 框架对同一视频数据进行对比实验。

**📈 对比分析**

与现有感知式 SSI（如 RFID Tattoo、HearMe、mSilent、TWLip‑Seq 等）相比，Lip‑Siri 在词级预测上达到 85.61% 的准确率，句级 WER 为 36.87%，与同类视觉 SSI（平均 WER 32.3%）相近；在消息发送与静默助手场景下，准确率可达 92% 以上，且在噪声环境中表现优于传统语音助手。

**⚠️ 局限性**

局限性包括：①句子级 WER 仍高于视觉基准；②对词表的依赖限制了完全自由的句子生成；③性能随 Wi‑Fi 信号强度变化明显，对弱信号环境不友好；④模型训练需要一定量的个体数据，长期使用可能需定期微调；⑤对剧烈身体运动（如跳跃）时鲁棒性不足。

---

## 594. TempDiffReg: Temporal Diffusion Model for Non-Rigid 2D-3D Vascular Registration

**arXiv ID:** 2601.18168 | [PDF](https://arxiv.org/pdf/2601.18168v1)

**作者:** Zehua Liu `[一作]` (Beihang University), Weixin Si `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种基于时间扩散模型的2D-3D血管配准框架TempDiffReg，实现了从粗到细的全流程配准。

**💡 创新点**

① 结构感知PnP实现全局粗配准；② 针对血管分支的时间扩散模型实现局部非刚性细化；③ 通过多帧时序信息与变形多样性训练提升精度与不确定性评估。

**🔧 技术方法**

时间扩散概率模型、Transformer编码器、结构先验（长度/曲率）约束、姿态PnP、深度学习可微注册。

**📊 数据集**

23例肝细胞癌临床案例，构建626条多帧样本和1322条单帧样本，包含预手术CTA 3D中心线与术中DSA 2D标注。

**📈 对比分析**

与TransMorph、uniGradICON、ViT‑VNet、SIRU‑Net等SOTA单帧方法在单帧数据集上对比，MSE 0.63 mm、MAE 0.51 mm，较最佳基线下降约66.7%/17.7%，在最大误差、长度误差、曲率误差也显著优于对手。

**⚠️ 局限性**

数据量有限，且金属伪影（支架/夹）会影响配准质量，需扩大数据集与鲁棒分割提升。

---

## 595. Lifecycle Cost-Effectiveness Modeling for Redundancy-Enhanced Multi-Chiplet Architectures

**arXiv ID:** 2601.18159 | [PDF](https://arxiv.org/pdf/2601.18159v1)

**作者:** Zizhen Liu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Huawei Li `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了面向多芯片架构的生命周期成本效能（LCE）框架，评估了模块级、路由级和芯片级冗余对成本、产量、可靠性与使用寿命的综合影响，

**💡 创新点**

创新点在于首次将冗余、可靠性和使用寿命统一纳入LCE度量，构建了完整的成本-效能模型，并通过多目标优化实现冗余协同设计

**🔧 技术方法**

采用Monte Carlo仿真、负二项分布缺陷建模、指数寿命模型以及LCE计算公式，对芯片、芯片组和整体架构进行分析

**📊 数据集**

通过对AMD公开数据的开放源代码Chiplet Actuary框架进行验证，使用的主要数据为AMD工艺与成本参数

**📈 对比分析**

与Chiplet Actuary对比，模型误差低于10%，在多种冗余配置下能显著降低LCE，验证了模型在成本与可靠性折中的表现

**⚠️ 局限性**

局限性包括对工作负载特定性能未建模、假设冗余组件与正常组件完全等价、并未进行实物实验验证

---

## 596. HomoFM: Deep Homography Estimation with Flow Matching

**arXiv ID:** 2601.18222 | [PDF](https://arxiv.org/pdf/2601.18222v1)

**作者:** Mengfan He `[一作]` (Tsinghua University), Ziyang Meng `[通讯]` (Tsinghua University)

**通讯引用:** 6146 | [OpenAlex ID](https://openalex.org/A5051392570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HomoFM 框架，将单目标平面投影变换问题建模为基于流匹配的连续位移场学习，从而实现高精度全局单应性估计。

**💡 创新点**

创新点在于首次将流匹配技术应用于单应性估计，将对齐视作从零初始网格到目标网格的线性传输过程，并通过梯度反转层实现域不变特征学习。

**🔧 技术方法**

使用 DINOv2‑Small 作为 backbone，基于条件流匹配（Conditional Flow Matching）和 ODE 求解器的速度化迭代，结合 GRL 进行域适应。

**📊 数据集**

使用 MSCOCO、VIS‑IR（可见‑红外）和 GoogleMap（卫星‑地图）三大数据集，另外构建 AVIID‑homo 进行零样本评估。

**📈 对比分析**

与特征匹配基线（SIFT+LightGlue、LoFTR、RoMa）及深度单应性方法（GFNet、RHWF、MCNet、PRISE）对比，HomoFM 在 AUC@3 上分别提升 0.76%–3.23%，参数仅 3.37 M，MACs 147 G，明显优于现有 SOTA。

**⚠️ 局限性**

局限在于仍需多步 ODE 求解，虽然比扩散快但比一次性回归略慢；对极端视角变形或极低纹理区域的鲁棒性尚待进一步验证。

---

## 597. Rhea: Detecting Privilege-Escalated Evasive Ransomware Attacks Using Format-Aware Validation in the Cloud

**arXiv ID:** 2601.18216 | [PDF](https://arxiv.org/pdf/2601.18216v1)

**作者:** Beom Heyn Kim `[一作]` (Hanyang University), Mohammad Mannan `[通讯]` (Concordia University)

**通讯引用:** 1720 | [OpenAlex ID](https://openalex.org/A5055898168)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种云端快照与格式感知验证相结合的勒索软件检测系统，针对特权提升与细粒度加密的PEER攻击实现高精度检测与恢复。

**💡 创新点**

利用文件格式规范作为安全不变式，在云端对增量快照进行格式感知校验，从而在极小加密片段下仍能准确识别恶意加密，突破传统熵/统计方法的局限。

**🔧 技术方法**

结合VM隔离、周期性块级快照、增量差分提取、自适应窗口统计筛选（SAWA）、块到文件映射以及多格式验证器（文本、ZIP、OOXML、PDF等）技术。

**📊 数据集**

使用约16.4 MB的10种常见格式（PDF、DOCX、PPTX、XLSX、TXT、JPEG、PNG、ZIP、MP3、MP4）做模拟实验，并在78个真实勒索软件样本上评估。

**📈 对比分析**

与基于熵/χ²的统计检测相比，FAV在细粒度加密下保持近100 %准确率；整体检测平均耗时约29 s（最高≈28 s），主要开销来自SAWA和FAV。

**⚠️ 局限性**

对无格式、低熵或使用Base64等编码隐藏的加密仍可能漏检；系统依赖VM、快照窗口和可信云服务器，且对开放格式的“宽松区”仍需额外契约检查。

---

## 598. PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR

**arXiv ID:** 2601.18207 | [PDF](https://arxiv.org/pdf/2601.18207v1)

**作者:** James Burgess `[一作]` (Stanford University), Serena Yeung-Levy `[通讯]` (Chan Zuckerberg Biohub Network)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了面向科学论文的检索式问答搜索代理，构建了 16 M 维基医学摘要语料库、60 k 题目答案对数据集，并在此环境下训练 RLVR 搜索代理。

**💡 创新点**

创新点包括：①将 RLVR 迁移至专业领域问答；②设计可验证的单步实体问题生成管线；③公开大型科学检索语料与基准，促进领域代理研究。

**🔧 技术方法**

技术手段主要为：强化学习可验证奖励（RLVR）与 Search‑R1 框架、LLM（Qwen‑Instruct 3B/7B）做决策、BM25 与 e5 的检索器、GRPO 优化器。

**📊 数据集**

使用数据集：PubMed 16 M 摘要检索库；PaperSearchQA 60 k 训练/5 k 测试集；BioASQ 1 609 题目作为外部基准。

**📈 对比分析**

与基线（无检索、Chain‑of‑Thought、RAG、PaperQA 等）相比，RLVR 训练的 7B 模型在 PaperSearchQA 上提升约 14.5 分、在 BioASQ 上提升约 9.3 分，RAG 与检索无关基线差距超过 17 分。

**⚠️ 局限性**

局限性：①仅覆盖单跳事实性问题，未涉及多跳、列表、摘要等类型；②数据生成依赖 LLM 可能产生幻觉；③仅使用文本检索，未考虑图像/表格等非文本信息；④未在真实科研工作中进行大规模部署与评估。

---

## 599. MemWeaver: Weaving Hybrid Memories for Traceable Long-Horizon Agentic Reasoning

**arXiv ID:** 2601.18204 | [PDF](https://arxiv.org/pdf/2601.18204v1)

**作者:** Juexiang Ye `[一作]` (Harbin Institute of Technology), Dechen Zhan `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1038 | [OpenAlex ID](https://openalex.org/A5113748483)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 MemWeaver，一种三层统一的长时记忆框架，将时序图、经验抽象和原始段落集成，用于支持大语言模型在多轮交互中的一致性推理。

**💡 创新点**

创新点在于：①将记忆视为动态整合与更新过程；②构建时序化知识图并通过 LLM 验证；③引入可验证的经验抽象层；④采用双通道检索同步结构化事实与证据，实现可追踪的推理。

**🔧 技术方法**

技术手段包括：LLM 辅助的实体/关系抽取、时间归一化与会话级审核；DBSCAN 聚类配合 LLM 一致性检查抽象经验；图结构检索与向量检索结合的双通道上下文构造；以及使用稀疏向量索引实现高效检索。

**📊 数据集**

实验使用 LoCoMo 长会话问答基准，涵盖单跳、多跳、时序、开放域等四类问题。

**📈 对比分析**

与 LoCoMo、MemoryBank、ReadAgent、A-Mem 等基线比较，MemWeaver 在多跳与时序任务上显著提升 F1 与 BLEU‑1，整体排名第一；同时推理时上下文长度减少超过 95%，在 GPT‑4o‑mini、Llama3.2 与 Qwen2.5 等多种 LLM 上均保持领先性能。

**⚠️ 局限性**

限制点包括：对底层 LLM 的抽取与验证能力依赖较高；目前仅支持文本交互，难以直接扩展至多模态；检索与更新过程仍有一定计算开销；经验抽象阈值与聚类参数需手工调优。

---

## 600. DMAP: Human-Aligned Structural Document Map for Multimodal Document Understanding

**arXiv ID:** 2601.18203 | [PDF](https://arxiv.org/pdf/2601.18203v1)

**作者:** ShunLiang Fu `[一作]` (Nanjing University of Science and Technology), Jinhui Tang `[通讯]` (Nanjing Forestry University)

**通讯引用:** 27708 | [OpenAlex ID](https://openalex.org/A5035112538)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种文档级结构化文档映射（DMAP），用于多模态文档理解，旨在解决现有多模态文档问答系统在语义检索中忽视文档内在层次和关系结构的问题。

**💡 创新点**

创新点在于DMAP显式编码文档的层次组织和元素间关系，设计了结构化语义理解代理（SSUA）和反思推理代理（RRA），以实现结构感知的检索和推理。

**🔧 技术方法**

使用了结构化语义理解代理（SSUA）和反思推理代理（RRA）技术，结合了文档的文本、表格和视觉信息，进行结构化的知识映射和推理。

**📊 数据集**

在MMDocQA基准上进行了广泛实验，使用了多个数据集，包括MMLongBench、LongDocURL、PaperText、PaperTab和FetaTab。

**📈 对比分析**

与传统的RAG方法相比，DMAP在检索精度、推理一致性和多模态理解上显著提升，尤其在复杂问题和多页文档的处理上表现优异，平均准确率提高了12.4%。

**⚠️ 局限性**

局限性在于尽管DMAP在多模态理解上表现良好，但在处理纯文本问题时的优势可能不如在处理视觉信息时明显，且对长文档的处理仍面临挑战。

---

## 601. SAGE: Steerable Agentic Data Generation for Deep Search with Execution Feedback

**arXiv ID:** 2601.18202 | [PDF](https://arxiv.org/pdf/2601.18202v1)

**作者:** Fangyuan Xu `[一作]` (New York University), Chen-Yu Lee `[通讯]` (Google Cloud AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种双代理（数据生成器+搜索代理）框架，通过搜索执行反馈迭代生成需要多步搜索的高质量问答数据；

**💡 创新点**

创新点在于利用搜索执行轨迹作为反馈而非仅作过滤，实现“可调节难度”的逆向数据生成，并且在固定语料库上自动生成多步搜索题目，显著提升生成数据的正确率和难度；

**🔧 技术方法**

核心技术包括基于大型语言模型（gemini‑2.5‑flash）的数据生成与搜索代理，ReACT式交互式搜索与推理，强化学习（PPO）训练搜索代理，以及对执行轨迹的反馈循环；

**📊 数据集**

使用的数据集主要是2018年维基百科语料库作为检索库，并在公开的 NQ、HotpotQA、Musique、FRAMES 等多跳 QA 数据集上进行评测；

**📈 对比分析**

与传统基于单步检索或人工标注的训练数据相比，生成的数据在内部评测中正确率提升至约87%，难度（Avg@4）显著下降；在下游训练搜索代理时，相比 NQ+HotpotQA 与 Musique，Qwen‑2.5‑3B 的平均准确率从 15.9% 提升至 28.5%（27% 相对提升），Qwen‑2.5‑7B 从 29.1% 提升至 38.1%（29% 相对提升），并在跨工具（Google Search）测试中亦表现出显著优势；

**⚠️ 局限性**

局限性包括：依赖固定搜索代理作为反馈，未实现双代理协同进化；评估仅基于 pass@K=1 的正确性判定，可能忽略某些错误；实验规模限制在 7B 级模型及单一通用语料库，未覆盖更大模型或专业领域数据；

---

## 602. YOLO-DS: Fine-Grained Feature Decoupling via Dual-Statistic Synergy Operator for Object Detection

**arXiv ID:** 2601.18172 | [PDF](https://arxiv.org/pdf/2601.18172v1)

**作者:** Lin Huang `[一作]` (Chongqing University), Yue Niu `[通讯]` (Chongqing University)

**通讯引用:** 49 | [OpenAlex ID](https://openalex.org/A5080143525)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过引入Dual-Statistic Synergy Operator（DSO）、Dual-Statistic Synergy Gating（DSG）以及Multi-Path Segmented Gating（MSG），在YOLOv8框架中实现细粒度特征解耦和多尺度自适应注意力，提升目标检测性能。

**💡 创新点**

创新点包括：①在两维空间同时建模通道均值与峰值‑均值差，形成DSO实现不同尺度与背景特征的解耦；②DSG在通道维度上做自适应门控，进一步抑制跨类别干扰；③MSG在深度维度上做分段门控，引入温度控制和随机噪声，使不同深度特征按需加权；这些模块仅带来微小计算开销。

**🔧 技术方法**

技术实现采用通道均值、峰值统计、非线性DSO函数、1×1卷积门控、Sigmoid激活、Softmax归一化、温度调节、Gaussian噪声、TensorRT加速等。

**📊 数据集**

实验基于MS‑COCO 2017数据集（训练集118,287张，验证集5,000张），进行多尺度（N、S、M、L、X）训练与评估。

**📈 对比分析**

与YOLOv8在相同尺度下对比，YOLO‑DS在COCO val2017的AP提升1.1%~1.7%，Latency仅提升≤0.5 ms；在大模型规模（M、L）下优于YOLOv9/10/11，展示了更优的精度‑速度权衡。

**⚠️ 局限性**

局限性包括：对极小模型的加速提升有限；目前验证仅在COCO上，需进一步测试在不同数据集与真实部署场景中的鲁棒性与可迁移性。

---

## 603. Accelerating Update Broadcasts Over LoRaWAN Downlink via D2D Cooperation

**arXiv ID:** 2601.18134 | [PDF](https://arxiv.org/pdf/2601.18134v1)

**作者:** Anshika Singh `[一作]` (Indian Institute of Technology Bhubaneswar), Siddhartha S. Borkotoky `[通讯]` (Indian Institute of Technology Bhubaneswar)

**通讯引用:** 280 | [OpenAlex ID](https://openalex.org/A5049436523)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

在LoRaWAN中设计了一种基于设备对设备（D2D）合作的更新广播机制，用于加速远程固件/模型更新的下行投递。

**💡 创新点**

创新点在于：1) 通过已完成更新的节点自发广播少量码化碎片，无需额外的服务器反馈或基础设施；2) 采用无界拉特码结合自适应D2D帧数量的机制，实现更新完成率的动态提升；3) 通过下行SF动态调度与D2D窗口同步，兼顾能耗与时延。

**🔧 技术方法**

技术手段包括：LoRaWAN Class B/​C协议、基于拉特码的无界前向纠错(FEC)、SF自适应下行调度、D2D窗口与随机超时段选择、能耗与时延分析模型。

**📊 数据集**

实验数据全部来自仿真，设置为10 kB更新拆分为200个50 B碎片，400节点均匀分布在1 km半径区域，1 % duty cycle，采用Rayleigh衰落及Poisson干扰模型。

**📈 对比分析**

通过与固定SF(FSF‑12)、无组多SF(GL‑MSF)和理想D2D（D2D‑PSI）三种基准方案比较，结果显示：在节点边缘完成时延从42 小时降至45 分钟，整体平均时延下降≈99%；能耗也比基准低约70–80%。

**⚠️ 局限性**

局限性包括：在稀疏网络或严重阴影/阻塞环境下收益减弱；D2D帧数量与能耗平衡需要根据设备电池进行调优；实现假设节点间可自由广播且同步可靠，实际部署需进一步验证。

---

## 604. RTeAAL Sim: Using Tensor Algebra to Represent and Accelerate RTL Simulation (Extended Version)

**arXiv ID:** 2601.18140 | [PDF](https://arxiv.org/pdf/2601.18140v1)

**作者:** Yan Zhu `[一作]` (University of California), Nandeeka Nayak `[通讯]` (University of California)

**通讯引用:** 77 | [OpenAlex ID](https://openalex.org/A5029821955)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将 RTL 仿真重新表述为稀疏张量代数问题，并实现一个基于 EDGE 和 TeAAL 的原型仿真器

**💡 创新点**

创新点在于：① 用稀疏张量代数模型取代传统直线 C++ 代码；② 通过 TeAAL 的分层抽象（Cascade、Mapping、Format、Binding）系统化地映射并优化 RTL 仿真；③ 结合张量压缩、循环展开和张量内联等技术显著降低编译成本与前端瓶颈

**🔧 技术方法**

核心技术包括：稀疏张量（fibertree）表示、Extended Einsum（EDGE）语义、TeAAL 优化框架、张量压缩格式、循环展开、张量内联、以及 C++ 编译器（clang++ 14）

**📊 数据集**

使用 FIRRTL 生成的 RTL 设计（RocketChip、BOOM、Gemmini、SHA3）以及对应的 1‑12 核、1‑24 核等不同规模的实验集，测试量化为千周期级别

**📈 对比分析**

与主流仿真器 Verilator 和 ESSENT 对比；在多核、多 ISA (Arm/x86) 主机上，最佳 kernel（如 PSU）在多种 RTL 设计下均能与 Verilator 竞争或超越其性能，同时编译时间与内存使用远低于 Verilator/ESSENT

**⚠️ 局限性**

限制主要在于：① 目前仅实现部分优化，仍为原型级；② 对于小型、极度缓存敏感的设计（如 SHA3），高展开的 TI kernel 仍优于已优化的 Tensor kernel；③ 在极大设计规模或非 CPU 平台（GPU/ASIC）尚未充分验证，需进一步泛化

---

## 605. Rethinking Cross-Modal Fine-Tuning: Optimizing the Interaction between Feature Alignment and Target Fitting

**arXiv ID:** 2601.18231 | [PDF](https://arxiv.org/pdf/2601.18231v1)

**作者:** Trong Khiem Tran `[一作]` (Washington State University), Trong Nghia Hoang `[通讯]` (Washington State University)

**通讯引用:** 729 | [OpenAlex ID](https://openalex.org/A5102929916)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一种新的交叉模态微调框架RECRAFT，结合特征对齐、特征-标签失真和目标拟合三项指标，通过两阶段优化实现跨模态知识迁移。

**💡 创新点**

创新点在于引入可证明的泛化上界，将特征对齐与目标拟合的相互作用量化为“特征‑标签失真”，并以此指导算法设计，突破了现有经验性调度的局限。

**🔧 技术方法**

主要技术包括：理论泛化界推导、Wasserstein距离与熵信息度量、源模型特征映射与目标预测器的双向传输规划、以及两阶段梯度优化（先对齐再拟合）。

**📊 数据集**

使用的基准数据集有NAS‑Bench‑360（10种不同模态任务）和PDEBench（多种偏微分方程模拟数据），并在各自的源模型上采用RoBERTa与Swin Transformer。

**📈 对比分析**

与Naive Fine‑Tuning、ORCA、PARE、MoNA等现有方法对比，RECRAFT在NAS‑Bench‑360上获得8/10任务最低误差、平均排名1.3；在PDEBench上获得7/8任务最佳或次优误差、平均排名1.25，表现显著优于SOTA。

**⚠️ 局限性**

局限性包括：理论上需要对源模型预测器满足Lipschitz假设，实际实现需调参；两阶段优化复杂度高，可能对极端小样本场景效果有限；在某些模态组合下，特征‑标签失真估计仍依赖源标签模拟，可能受误差影响。

---

## 606. Facial Emotion Recognition on FER-2013 using an EfficientNetB2-Based Approach

**arXiv ID:** 2601.18228 | [PDF](https://arxiv.org/pdf/2601.18228v1)

**作者:** Sahil Naik `[一作]` (VIT), Pavankumar Singh `[通讯]` (VIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于EfficientNetB2的轻量化面部情绪识别模型，并通过两阶段训练（warm‑up + fine‑tuning）实现高精度预测。

**💡 创新点**

创新点包括：①在FER-2013数据集上首次将EfficientNetB2与AdamW、标签平滑、类别权重裁剪、混合精度训练及实时增强相结合，实现参数量仅为VGG16的十分之一；②采用两阶段训练策略，使得模型在保持预训练特征的同时快速适应情绪类别；③提供完整可复现的实验框架和细粒度性能评估。

**🔧 技术方法**

技术手段包括：EfficientNetB2卷积骨干、GlobalAveragePooling、Dropout、AdamW优化器、标签平滑(ε=0.06)、裁剪的类别权重、混合精度训练、实时数据增强（旋转、平移、缩放、剪切、水平翻转）等。

**📊 数据集**

使用的主要数据集是公开的FER‑2013（35,887张48×48灰度图，7个情绪类别），训练集按87.5/12.5比例划分，保持官方测试集不变。

**📈 对比分析**

与VGG16、ResNet等传统大型网络进行对比，本文模型在FER‑2013测试集上取得68.78%准确率，参数量约9.2M，显著低于VGG16的138M，且在保持相近或更好精度的同时大幅降低计算和存储成本，适合实时/边缘部署。

**⚠️ 局限性**

局限性包括：仍存在对低分辨率图像中恐惧与悲伤、悲伤与中性等情绪的混淆；标签噪声与类别不平衡问题未完全消除；缺乏人种/性别公平性评估；仅在FER‑2013上验证，跨数据集泛化尚未探测。

---

## 607. Yunjue Agent Tech Report: A Fully Reproducible, Zero-Start In-Situ Self-Evolving Agent System for Open-Ended Tasks

**arXiv ID:** 2601.18226 | [PDF](https://arxiv.org/pdf/2601.18226v1)

**作者:** Haotian Li `[一作]` (Harbin Institute of Technology), Chao Peng `[通讯]` (Yunjue Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Yunjue Agent，一种在推理阶段通过不断合成、验证和归纳工具，实现自我演化的 LLM‑驱动代理。

**💡 创新点**

创新点在于：①以工具为核心的自我演化框架；②并行批量演化与工具吸收机制；③引入演化一般化损失（EGL）指标用于监控收敛。

**🔧 技术方法**

使用技术包括 Gemini 3 Pro / GPT‑5 LLM、ReAct 交互式推理、多代理架构（Manager/Tool Developer/Executor/Integrator）、工具聚类与合并算法。

**📊 数据集**

实验数据集涵盖五个跨域基准：HLE、DeepSearchQA、xBench（ScienceQA 与 DeepSearch）、FinSearchComp。

**📈 对比分析**

与商业与开源静态/自演化代理进行对比；在零起点与热起点设置下均取得 SOTA 结果，特别是 DSQA、FSC、xSciQA；工具收敛曲线表明系统在约 1k 条查询后实现快速收敛。

**⚠️ 局限性**

局限性：仅演化工具，未同步演化记忆与工作流；LLM 生成随机性导致工具集不稳定；批量大小选择会影响收敛速度；缺乏系统级预训练与更广泛的模型适配性评估。

---

## 608. LLM-ForcedAligner: A Non-Autoregressive and Accurate LLM-Based Forced Aligner for Multilingual and Long-Form Speech

**arXiv ID:** 2601.18220 | [PDF](https://arxiv.org/pdf/2601.18220v1)

**作者:** Bingshen Mu `[一作]` (Northwestern Polytechnical University), Lei Xie `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 9595 | [OpenAlex ID](https://openalex.org/A5066245750)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 LLM‑ForcedAligner，将强制对齐（FA）改造成槽位填充任务，使用多语言 Speech LLM 在语音与加入时间槽的文本上直接预测离散时间索引，实现多语种、跨语种和长时段语音的快速准确对齐。

**💡 创新点**

创新点包括：① 将时间戳离散化为索引并以特殊 token 作为槽位；② 动态槽位插入，支持任意位置对齐；③ 在训练中使用 causal attention 及无位移的输入/标签，仅在槽位位置计算 loss；④ 支持非自回归推理，消除幻觉并加快速度；⑤ 通过 Speech LLM 的多语言语义和长上下文能力，克服传统方法对语言和时延的限制。

**🔧 技术方法**

技术要点：语音编码器 AuT；多语言 LLM Qwen3‑0.6B；时间戳预测层（线性 3 750 类）；causal attention + dynamic slot insertion；非自回归推理；Adam + warm‑up 训练；评估指标 AAS（平均时移）和 RTF（实时因子）。

**📊 数据集**

数据集：56 000 小时、10 语种（中、英、法、德、意、日、韩、葡、俄、西）混合训练集，来源于 Aishell、WenetSpeech、GigaSpeech、LibriSpeech、MLS 等；测试集使用 MFA 伪标签的多语种语料以及内部人工标注的中文数据。

**📈 对比分析**

比较方法：与 Monotonic‑Aligner（仅中文）、NFA、WhisperX 等传统 FA 方法对比；在多语种、跨语种、长达 300 s 的语音上，LLM‑ForcedAligner 的 AAS 下降 69–78%（如中文 33.1 ms 对比 161.1 ms），平均 AAS 仅 42.9 ms，RTF 仅略高（0.0159 vs 0.0079–0.0113）。在人工标注中文数据上，同样实现 32.4 ms 的平均 AAS，表明在实际场景中的性能显著优于传统方法。

**⚠️ 局限性**

局限性：① 训练依赖 MFA 伪标签，可能导致模型过拟合 MFA 的系统性偏差；② 人工标注测试仅覆盖中文，无法全面评估其它语言表现；③ 语言分布不均导致非中英语种性能相对较差；④ 对极端噪声、会议、音乐、影视等更复杂场景的鲁棒性尚未验证；⑤ 由于时间索引粒度限制（80 ms），对毫秒级精度的提升空间有限。

---

## 609. Scalable Quantum Message Passing Graph Neural Networks for Next-Generation Wireless Communications: Architectures, Use Cases, and Future Directions

**arXiv ID:** 2601.18198 | [PDF](https://arxiv.org/pdf/2601.18198v1)

**作者:** Le Tung Giang `[一作]` (Pusan National University), Won-Joo Hwang `[通讯]` (Pusan National University)

**通讯引用:** 5246 | [OpenAlex ID](https://openalex.org/A5085192467)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了一种可扩展的量子消息传递图神经网络（SQM‑GNN），并在设备到设备（D2D）功率控制任务中进行验证。

**💡 创新点**

核心创新在于：①将消息传递直接嵌入参数化量子电路（PQC）中；②采用子图分解与共享参数的PQC，兼顾边特征并适配NISQ硬件；③实现了在大规模无线图上的可扩展、可泛化的量子图学习。

**🔧 技术方法**

技术手段包括：量子机器学习（PQC、量子编码、测量）；图神经网络（消息传递、聚合）；子图采样与共享PQC；旋转角度编码；Python/PennyLane 与 PyTorch 的混合训练框架。

**📊 数据集**

实验使用自生成的D2D网络数据集：在500m×500m 区域内随机放置 10–80 对单天线 D2D 链，生成 10,000 条训练/测试样本，采用路径损耗模型得到信道增益。

**📈 对比分析**

对比方法：WMMSE、经典 GNN 与 SQM‑GNN。结果显示，SQM‑GNN 在 100 轮训练后获得约 2.6 bps/Hz 的系统总速率，优于 GNN 的 2.3 bps/Hz，并在不同节点数 K 与功率约束下，以 95–107% 的比例超越 WMMSE，展现出更好的泛化与可扩展性能。

**⚠️ 局限性**

局限性：受 NISQ 设备的 qubit 数量与门误差限制，必须采用子图分解导致邻域信息损失；量子电路深度与噪声影响可训练性；缺乏标准化的 PQC 设计方法；在更大规模网络下仍需进一步验证与优化。

---

## 610. QualiRAG: Retrieval-Augmented Generation for Visual Quality Understanding

**arXiv ID:** 2601.18195 | [PDF](https://arxiv.org/pdf/2601.18195v1)

**作者:** Linhan Cao `[一作]` (Shanghai Jiao Tong University), Xiongkuo Min `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9611 | [OpenAlex ID](https://openalex.org/A5043405654)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种训练无须的检索增强生成框架QualiRAG，用于视觉质量的可解释性理解。

**💡 创新点**

创新点在于将质量相关问题拆解为结构化检索请求，构建四类补充知识库（元数据、主体定位、全局质量摘要、局部质量描述），并通过相关性检索实现对视觉质量的全局/局部解释推理，无需任何任务特定微调。

**🔧 技术方法**

采用检索增强生成（RAG）技术，利用LMM主干与辅助模型（InternVL3-9B-Instruct、Qwen3-VL-8B-Instruct）、元数据提取、主体定位、全局/局部质量描述生成、Contriever+FAISS的相似度检索等技术。

**📊 数据集**

使用的评测数据集包括图像质量理解的Q‑Bench、Q‑Bench‑Video；质量比较的LIVE‑C、AGIQA、PIPAL、KoNViD‑1K、VDPVE、LIVE‑HFR等公开基准。

**📈 对比分析**

与开源通用LMM、VQA微调LMM、闭源专有LMM以及专用VQA评分模型进行对比。QualiRAG在Q‑Bench上整体准确率达81.7%，在Q‑Bench‑Video上达61.7%，均超过所有对手；在质量比较任务中取得同类最佳或可与专用VQA模型相媲美的性能，展现出较强的跨域鲁棒性。

**⚠️ 局限性**

局限性包括受主干LMM性能约束；对极端低质量或新型AIGC噪声的解释能力仍有限；知识库构建需要手工或额外工具支持；在高分辨率或长时长视频场景下的检索与生成开销较大。

---

## 611. Success Conditioning as Policy Improvement: The Optimization Problem Solved by Imitating Success

**arXiv ID:** 2601.18175 | [PDF](https://arxiv.org/pdf/2601.18175v1)

**作者:** Daniel Russo `[一作]` (Columbia University), Daniel Russo `[通讯]` (Columbia University)

**通讯引用:** 2269 | [OpenAlex ID](https://openalex.org/A5101747404)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文研究了成功条件化（success conditioning）作为一种提高策略的技术，证明了它可以精确地解决一个信任区域优化问题，最大化策略改进，同时满足由数据自动确定的χ²散度约束。

**💡 创新点**

创新点在于将成功条件化解释为一种保守的策略改进算子，提供了相对政策改进、政策变化和行动影响之间的精确等式，揭示了成功条件化的几何特性和优化问题。

**🔧 技术方法**

使用了信任区域优化理论，特别是χ²散度作为约束，结合贝叶斯条件化来实现成功条件化。

**📊 数据集**

使用了马尔可夫决策过程（MDP）中的轨迹数据，特别是成功和失败的终端状态来定义成功事件。

**📈 对比分析**

与其他信任区域优化方法（如TRPO和PPO）进行比较，成功条件化在保持接近当前策略的同时，提供了更保守的改进，且在失败时表现为几乎不改变策略。

**⚠️ 局限性**

限制在于该分析仅在固定行为策略下进行，未考虑有限数据、函数逼近或大规模优化的影响，这些因素可能会影响成功条件化的实际表现。

---

## 612. An Initial Evaluation of Distributed Graph Algorithms using NWGraph and HPX

**arXiv ID:** 2601.18158 | [PDF](https://arxiv.org/pdf/2601.18158v1)

**作者:** Karame Mohammadiporshokooh `[一作]` (Louisiana State University), Hartmut Kaiser `[通讯]` (Louisiana State University)

**通讯引用:** 1830 | [OpenAlex ID](https://openalex.org/A5051320432)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

实现了基于HPX的分布式图算法库NWGraph，并实现了分布式BFS和PageRank，并与Boost Graph Library（PBGL）进行了性能对比。

**💡 创新点**

创新点在于将NWGraph与HPX的异步多任务模型结合，利用Active Global Address Space（AGAS）和分布式partitioned_vector，实现在不需要全局同步的情况下完成图遍历和迭代，展示了HPX在分布式图分析中的可扩展性。

**🔧 技术方法**

使用了HPX的异步任务、AGAS、分布式partitioned_vector、异步远程动作、C++20概念以及NWGraph库。

**📊 数据集**

使用Erdős–Rényi随机图（urand系列）作为数据集，如urand25（2^25节点）。

**📈 对比分析**

通过在相同硬件上（Intel Xeon Ice Lake，最多32节点）对比BFS和PageRank的运行时间和加速比，结果显示HPX实现的BFS性能优于PBGL，而PageRank虽有优化但仍落后于PBGL。

**⚠️ 局限性**

主要局限在于通信开销高、负载不均衡、PageRank性能仍不足；实现仍较原始，缺乏完整的错误收敛检测、故障容错机制以及对更广泛算法的支持。

---

## 613. Contact Plan Design For Optical Interplanetary Communications

**arXiv ID:** 2601.18148 | [PDF](https://arxiv.org/pdf/2601.18148v1)

**作者:** Jason Gerard `[一作]` (Concordia University), Sandra Cespedes `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种考虑光学定点、获取与跟踪（PAT）延迟的光学星际链路调度框架，并通过混合整数线性规划（MILP）最大化链路容量；

**💡 创新点**

首次将物理层定点延迟模型与时间扩展网络容量模型相结合，量化光学网络的使用周期并揭示“少数长链接”最优策略；

**🔧 技术方法**

使用时间扩展图、MILP、DTN中继、光学头跟踪模型以及NASA DSOC/TBIRD等仿真参数；

**📊 数据集**

利用NASA DSOC、TBIRD、LCRD以及IPN‑D生成的轨道与链路可用性数据；

**📈 对比分析**

与贪心、零延迟、FCP等基线算法对比，MILP在相同场景下提升约30%+容量，网络上线时间约为28天；

**⚠️ 局限性**

求解规模受限（超过约16节点即超时），缺乏对更大星座的可扩展性，未涵盖多波束/多频复用等未来技术。

---

## 614. Using LibCal Seats to Better Serve Students

**arXiv ID:** 2601.18230 | [PDF](https://arxiv.org/pdf/2601.18230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 615. Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking

**arXiv ID:** 2601.18146 | [PDF](https://arxiv.org/pdf/2601.18146v1)

**作者:** Huizhong Guo `[一作]` (Zhejiang University), Zhu Sun `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 2048 | [OpenAlex ID](https://openalex.org/A5033957641)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种轻量级、模型感知的推理路由框架，用于在LLM驱动的排序任务中自适应决定是否使用多步推理。

**💡 创新点**

创新点在于：①将排序任务特有的结构性特征（候选分散度、上下文-候选对齐等）与模型自身的难度感知（掩码检查表）相结合，形成多维路由信号；②在推理前即做决策，避免不必要的计算；③通过验证Pareto前沿动态选取不同的成本-效能策略，实现可配置的部署。

**🔧 技术方法**

核心技术包括：LLM隐藏层特征提取、统计式排序特征计算、掩码式诊断问答（checklist）获取模型难度信号、轻量回归式路由头训练以及基于成本敏感阈值的路由规则。

**📊 数据集**

实验使用三大公开数据集：信息检索的MS‑MARCO、个性化推荐的MovieLens（电影）和Amazon‑VG（视频游戏），并在五种开源LLM（Qwen3‑4B/8B/14B、Gemma3‑12B、GPT‑Oss‑20B）上评测。

**📈 对比分析**

与“始终推理”“始终直推”“随机”“自选”以及ARM系列自适应推理模型对比，所提路由在大多数模型和数据集上实现了显著的排序质量提升（例如MovieLens NDCG@10+6.3%）同时大幅降低token消耗（约-49.5%），并在Pareto前沿上提供多种成本效能平衡点。

**⚠️ 局限性**

局限性：路由所用的特征集合和诊断问答列表是基于当前任务与提示设计手工选取，可能无法覆盖更细粒度的语义难度，跨域或新提示时需要重新调优；此外模型感知信号受LLM内部概率的偏差影响，若LLM对难度判断本身不准则会影响路由效果。

---

## 616. DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints

**arXiv ID:** 2601.18137 | [PDF](https://arxiv.org/pdf/2601.18137v1)

**作者:** Yinger Zhang `[一作]` (Alibaba Group), Junyang Lin `[通讯]` (Alibaba Group)

**通讯引用:** 3025 | [OpenAlex ID](https://openalex.org/A5100612233)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一套基于多天旅游和购物任务的长周期规划基准，评估LLM代理在主动信息获取、局部与全局约束推理下的规划能力。

**💡 创新点**

通过构造可复现的离线沙盒环境、分层任务生成与手工质量控制，首次实现了真实世界复杂约束的多任务评估，并揭示当前LLM在全局一致性与信息采集上的缺陷。

**🔧 技术方法**

采用Python工具接口与LLM函数调用框架，结合内部推理（思考）与并行/顺序工具调用，以及规则化代码评估实现高效客观打分。

**📊 数据集**

旅游域使用公开API抓取的真实城市交通、住宿、景点等数据，购物域合成细粒度商品属性、库存、促销等数据，共计120个任务（中文/英文）。

**📈 对比分析**

对比多家前沿LLM（GPT‑5、Claude‑4.5、Gemini‑3等）在推理/非推理模式下的案例准确率与综合得分，发现即便顶尖模型在个体约束上表现良好，整体案例正确率也低至35%（旅游）/低于60%（购物），并揭示工具使用与内部推理带来的性能提升与成本权衡。

**⚠️ 局限性**

仅覆盖旅游与购物两个领域，查询合成可能偏离真实用户表达，且仅涵盖单轮规划，缺乏多轮交互与更广泛场景的评估。

---

## 617. Learning Fair Domain Adaptation with Virtual Label Distribution

**arXiv ID:** 2601.18171 | [PDF](https://arxiv.org/pdf/2601.18171v1)

**作者:** Yuguang Zhang `[一作]`, Ran He `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了虚拟标签分布感知学习（VILL）框架，通过自适应加权和KL重平衡两项策略提升无监督领域适应中的类别公平性，尤其关注最差类别的性能。

**💡 创新点**

创新点在于引入虚拟标签分布动态权重与KL基重平衡损失，既增强了难分类类别的学习，又在目标域中显式调整决策边界，无需额外监督即可提升最差类别准确率。

**🔧 技术方法**

使用了基于伪标签的动态权重计算、负指数变换的类别加权、KL散度重平衡损失以及传统域对齐损失，整合进现有UDA模型的可插拔模块。

**📊 数据集**

在常用的 OfficeHome、Office 四域图像分类数据集上进行实验，使用 ResNet‑50、CLIP‑ResNet‑50 等主干网络。

**📈 对比分析**

与 CDAN、MDD、ATDOC、PDA 等主流 UDA 方法比较，VILL 在 Worst‑5、Worst‑10 等最差类别指标上提升 4–6%（例如 CDAN‑VILL 从 20.3% 提升到 26.8%），同时保持或略微提升整体准确率。

**⚠️ 局限性**

局限性包括依赖伪标签分布的稳定性、对超参数 α、β 的敏感性以及在极端数据稀缺或伪标签噪声较大的场景下可能效果受限。

---

## 618. GAIA: A Data Flywheel System for Training GUI Test-Time Scaling Critic Models

**arXiv ID:** 2601.18197 | [PDF](https://arxiv.org/pdf/2601.18197v1)

**作者:** Shaokang Wang `[一作]` (Xiaomi Inc), Jian Luan `[通讯]` (Xiaomi Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了GUI Action Critic的Data Flywheel System（GAIA），通过自循环的数据飞轮收集正负动作样本并训练Intuitive Critic Model（ICM），在测试时使用Best‑of‑N筛选提高GUI agent执行准确率。

**💡 创新点**

创新点包括：① 用数据飞轮实现正负样本自我扩增；② 引入直观判断的ICM取代多步推理；③ 在无需额外训练的情况下，在测试阶段显著提升多模型性能。

**🔧 技术方法**

采用LVLM基础模型（如Qwen2.5 VL 7B、GPT‑4o等）进行Cross‑Entropy训练ICM，并在测试时通过Best‑of‑N候选筛选与数据飞轮采集与迭代。

**📊 数据集**

使用公开GUI基准数据集AndroidControl、GUI‑Odyssey和ScreenSpotV2，构建第一轮正负样本集𝒟及第二轮扩充集𝒟^+。

**📈 对比分析**

通过Type/GR/SR和点击准确率指标对闭源（GPT‑4o、Doubao）与开源（Qwen 2.5 VL、UI‑TARS等）模型进行对比，ICM与ICM‑r2分别提升10–12% SR、5–10% GR，明显优于基线和基于推理的RCM。

**⚠️ 局限性**

局限性在于仅在高层指令下工作，未充分利用低层计划；缺乏在线实时更新数据飞轮；对极罕见操作仍易误判；对多模态输入的探索有限。

---

## 619. MindCine: Multimodal EEG-to-Video Reconstruction with Large-Scale Pretrained Models

**arXiv ID:** 2601.18192 | [PDF](https://arxiv.org/pdf/2601.18192v1)

**作者:** Tian-Yi Zhou `[一作]`, Wei-Long Zheng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9714 | [OpenAlex ID](https://openalex.org/A5056335002)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MindCine框架，利用多模态联合学习和因果序列模型从EEG重建高质量视频。

**💡 创新点**

创新点在于结合多模态(文本、图像、深度)联合学习、引入大规模EEG预训练模型缓解数据稀缺，并使用CausalSeq解码动态感知信息。

**🔧 技术方法**

使用大型EEG编码器（如Gram等）、SoftCLIP对齐、EmbedNet+Transformer CausalSeq、文本到视频扩散模型以及对抗引导。

**📊 数据集**

主要使用SEED‑DV EEG‑视频对齐数据集，并利用BLIP2生成文本描述、DepthAnything v2提取深度。

**📈 对比分析**

与EEG2Video等基线在7个评估指标上比较，MindCine在语义层次和像素层次指标上均实现了最高或次高成绩，显著优于对照。

**⚠️ 局限性**

局限在于仍依赖大规模EEG模型和外部扩散模型，且在更大规模、异质性更高的数据上效果未知。

---

## 620. \textsc{NaVIDA}: Vision-Language Navigation with Inverse Dynamics Augmentation

**arXiv ID:** 2601.18188 | [PDF](https://arxiv.org/pdf/2601.18188v1)

**作者:** Weiye Zhu `[一作]` (SUSTech), Feng Zheng `[通讯]` (SpatialTemporal AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 NaVIDA 框架，通过逆向动力学监督和分层动作块编码实现视觉动作因果关系的显式学习，提升导航稳定性与泛化能力。

**💡 创新点**

创新点在于：①将逆向动力学监督与策略学习统一；②采用分层概率动作块（HPAC）提供多尺度、可区分的动作标签；③在推理阶段使用熵引导的执行长度自适应，避免长时序误差累积。

**🔧 技术方法**

技术手段包括：大规模多模态语言模型（Qwen‑2.5‑VL‑3B）、逆向动力学监督、动作块分层合并、熵基执行阈值、统一的动作语法和提示工程。

**📊 数据集**

使用的公开数据集包括：VLN‑CE R2R、RxR、以及 Habitat 轨迹（R2R、RxR、ScaleVLN、DAgger）用于构建 IDS 训练样本。

**📈 对比分析**

与现有单 RGB 8B 基线（如 JanusVLN、StreamVLN）以及多传感器方法对比，NaVIDA 在 R2R‑CE val‑unseen 上 NE↓0.46、OS↑4.3、SR↑0.9，RxR‑CE 上 NE↓1.2、SPL↑3.6，参数仅 3B，性能领先并保持高效率。

**⚠️ 局限性**

局限性包括：①仅基于单目 RGB，可能在动态或遮挡环境下受限；②动作块合并概率与层数需手工调节，对不同场景的泛化仍有空间；③仍依赖大规模预训练模型，部署时对算力有一定要求。

---

## 621. Exploring Customizable Interactive Tools for Therapeutic Homework Support in Mental Health Counseling

**arXiv ID:** 2601.18179 | [PDF](https://arxiv.org/pdf/2601.18179v1)

**作者:** Yimeng Wang `[一作]` (William and Mary), Yixuan Zhang `[通讯]` (William and Mary)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文设计、实现并评估了一款面向治疗师的交互式仪表板（TheraTrack），该仪表板集成多源临床数据（作业记录、情绪评估、患者生成健康数据等），利用大语言模型自动生成可追溯的摘要与问答，帮助治疗师快速梳理与解读患者的课后作业进展。

**💡 创新点**

创新点在于：①把临床作业数据、评估结果和可穿戴设备数据统一纳入同一平台；②通过可配置的三步工作流（需求定义→组件选择→仪表板展示）实现高度个性化；③结合大语言模型实现可追溯的自然语言摘要与交互式问答，并在界面中嵌入原始数据跳转，提升解释性与信任度。

**🔧 技术方法**

核心技术包括：Next.js Web 框架、Azure OpenAI GPT‑4o 大语言模型、LangChain 风格的提示工程、JSON 数据统一存储与可视化库（如 D3.js/Chart.js）以及安全的 OAuth 与 HIPAA‑兼容的数据加密方案。

**📊 数据集**

数据集主要来自临床实务：治疗师收集的真实患者作业日志、情绪打分、标准化测评结果、可穿戴设备的睡眠/心率/步数等生理数据；此外，还使用了公开的 CBT 练习模板与心理测评量表（如 PHQ‑9、GAD‑7 等）作为背景信息。

**📈 对比分析**

在 14 位治疗师的单次实验中，系统获得了高可用性与满意度：79% 的受访者对 AI 摘要功能“极大减少工作量”表示强烈同意，71% 认为系统高效定位信息；在可视化和交互式问答方面，90% 的受访者认为信息呈现清晰且易于检索；相比传统手工复核，治疗师的预备时间平均缩短约 40%（根据自评），但未给出客观的性能基准或对照组。

**⚠️ 局限性**

主要局限包括：① 样本规模有限（仅 14 位治疗师），未覆盖更广泛的治疗流派与临床设置；② 评估基于模拟案例与一次性交互，缺乏长期真实使用与随访数据；③ 未收集患者视角或隐私合规评估；④ 依赖 GPT‑4o 的生成质量与可解释性受限，系统在面对不完整或矛盾数据时的鲁棒性尚未验证。

---

## 622. Explaining Synergistic Effects in Social Recommendations

**arXiv ID:** 2601.18151 | [PDF](https://arxiv.org/pdf/2601.18151v1)

**作者:** Yicong Li `[一作]` (Dalian University of Technology), Feng Xia `[通讯]` (RMIT University)

**通讯引用:** 18946 | [OpenAlex ID](https://openalex.org/A5089615958)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SemExplainer，研究多视图图神经网络在社交推荐中的协同效应并给出可解释路径；

**💡 创新点**

创新点在于引入协同子图理论，用信息增益与条件熵优化提取协同子图，并以路径形式呈现协同解释；

**🔧 技术方法**

采用软掩码学习、互信息/条件熵优化、图神经网络（多视图）与 Dijkstra 路径搜索等技术；

**📊 数据集**

使用 ACM、Last‑FM、AugCitation 三个社交推荐数据集进行实验；

**📈 对比分析**

与 8 种基线（包括 GNNExplainer、PGExplainer、AxiomLayeredge、MAGE、GraphSHAP‑IQ、PaGE‑Link、xPath、SR‑GCA、CaGE）对比，SemExplainer 在 SIS/SIN 指标上始终最优，FID+、FID- 与基线相近，SPA 与基线相当；

**⚠️ 局限性**

主要限制是计算开销较大，仅能生成实例级解释，缺乏全局协同解释能力。

---

## 623. Forward Consistency Learning with Gated Context Aggregation for Video Anomaly Detection

**arXiv ID:** 2601.18135 | [PDF](https://arxiv.org/pdf/2601.18135v1)

**作者:** Jiahao Lyu `[一作]` (Xian University of Technology), Zhiyong Lv `[通讯]` (Xian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了轻量化视频异常检测模型FoGA，利用前向一致性学习和门控上下文聚合同时预测即时帧与长远帧；

**💡 创新点**

创新点包括仅2M参数的高效模型、联合即时+远程帧预测增强时序建模、门控上下文聚合模块（GCAM）在U‑Net跳跃连接中动态融合多尺度特征、前向一致性损失与混合错误评分机制；

**🔧 技术方法**

采用U‑Net骨干、门控注意力（通道+空间Efficient Attention）、多尺度上下文聚合、前向一致性损失、SSIM、PSNR混合异常分数、多尺度评估等技术；

**📊 数据集**

使用四大公开数据集：UCSD Ped1、UCSD Ped2、CUHK Avenue、ShanghaiTech；

**📈 对比分析**

与重建、单帧预测、多帧预测等多种基准方法对比，FoGA在Ped1、Ped2、Avenue、Sh‑Tech上的AUC分别为87.4%、98.9%、90.1%、76.2%，帧率110 FPS（不使用多尺度可达155 FPS），在参数、FLOPs、显存方面均最低，取得最佳性能‑效率平衡；

**⚠️ 局限性**

局限在于仅在四个标准数据集验证，缺乏更大规模或多场景真实部署评估；对极端光照/视角变化的鲁棒性未充分验证；模型仍需手动调优 λ、σ 等超参数。

---

## 624. Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents

**arXiv ID:** 2601.18217 | [PDF](https://arxiv.org/pdf/2601.18217v1)

**作者:** Zhihan Liu `[一作]` (Northwestern University), Na Zhang `[通讯]` (Northwestern University)

**通讯引用:** 69588 | [OpenAlex ID](https://openalex.org/A5100359646)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对四种文本型agent环境（WebShop、Sokoban、ALFWorld、SciWorld）进行RL后训练，系统评估并提升跨域泛化能力；

**💡 创新点**

创新点在于发现并验证状态信息丰富度与规划复杂度是跨域性能的关键因子，并通过轻量级状态随机化、SFT warm‑up与显式逐步推理等技术实现因果性提升；

**🔧 技术方法**

使用了基于GRPO的多轮RL框架、状态信息增强（注入无关文本）、SFT预训练、step‑by‑step reasoning以及对比分析的排名指标；

**📊 数据集**

实验数据来自四个公开agentic环境：WebShop、Sokoban、ALFWorld和SciWorld；

**📈 对比分析**

与无增强基线相比，状态随机化平均提升OOV成功率约32‑45%，SFT warm‑up在覆盖域内显著减缓性能衰减，启用推理保持OOV稳定，整体OOV排名由低至高依次提升；

**⚠️ 局限性**

局限性包括仅评估四个环境、模型规模有限、状态增强方法手工设计，且对更大规模、多模态或更复杂任务的推广尚需进一步验证。

---

## 625. Validation of a Software-Defined 100-Gb/s RDMA Streaming Architecture for Ultrafast Optoacoustic and Ultrasound Imaging

**arXiv ID:** 2601.18280 | [PDF](https://arxiv.org/pdf/2601.18280v1)

**作者:** Federico Villani `[一作]` (ETH Zurich), Luca Benini `[通讯]` (University of Bologna)

**通讯引用:** 55671 | [OpenAlex ID](https://openalex.org/A5043408422)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文设计并验证了一种软件定义的256通道超高速光声与超声成像架构——ListenToLight（LtL），通过16通道演示系统演示了从射频信号接收、JESD204B解码、实时环形缓冲、100 GbE RDMA流式传输到工作站的完整数据链路，并在超声与光声phantom上验证了时序精度和图像质量。

**💡 创新点**

核心创新在于：①整合宽带模拟前端与Zynq UltraScale+ MPSoC，实现软件可编程的触发与序列控制；②使用100 GbE RDMA实现无缓冲、连续高速原始数据流，突破传统局部缓冲+突发传输的瓶颈；③支持双模（光声+超声）并可扩展至256通道，满足宽带、低噪、实时成像需求。

**🔧 技术方法**

技术实现包括：JESD204B（子类1）高速数据接口、Zynq MPSoC（FPGA+ARM APU）进行时钟同步、帧切分与RDMA控制；100 GbE网络卡及RDMA协议栈；16通道模拟前端AFE58JD48（16 bit，125 MSPS）和STHV1600高压激励器；Linux+Petalinux自定义驱动实现软硬件协同。

**📊 数据集**

实验使用的“数据集”为：①CIRS 040GSE超声phantom（5 MHz平面波，80 MSPS采样）；②自制黑带光声phantom（808 nm激光，1 mJ，100 ns脉冲）；在演示系统中将16通道数据扩展为256通道（填充虚拟通道）进行RDMA传输和图像重建。

**📈 对比分析**

与现有研究平台（ULA‑OP 256、DiPhAS、Verasonics NXT、us4R、OpenSonics、CONUS、Robin等）在帧率、采样率、位深、通道数、双模支持和数据流速等指标进行雷达图对比；LtL实现持续原始数据流速达11.95 GB/s，125 MSPS采样、16 bit深度，双模5级脉冲；在16通道验证下可持续采样4.14 kFPS（256通道）；RDMA吞吐量最高95.6 Gb/s。

**⚠️ 局限性**

局限性主要在：①演示系统仅16通道，需进一步扩展到256通道（需复制前端卡并扩充FPGA资源）；②目前依赖商业评估板，尚未实现集成卡和完整硬件栈；③对实时波束形成等后处理功能尚未在芯片内实现；④连续流式传输仍受工作站内存/存储带宽限制。

---

## 626. Revisiting Aerial Scene Classification on the AID Benchmark

**arXiv ID:** 2601.18263 | [PDF](https://arxiv.org/pdf/2601.18263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 627. What Do Learned Models Measure?

**arXiv ID:** 2601.18278 | [PDF](https://arxiv.org/pdf/2601.18278v1)

**作者:** Indrė Žliobaitė `[一作]` (University of Helsinki), Indrė Žliobaitė `[通讯]` (University of Helsinki)

**通讯引用:** 7648 | [OpenAlex ID](https://openalex.org/A5037208593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文研究了在机器学习模型被用作测量仪器时，如何评估其测量稳定性。

**💡 创新点**

提出了测量稳定性概念，并指出传统的预测评估指标无法保证测量稳定。

**🔧 技术方法**

使用理论定义、实验验证与对比，主要通过线性回归模型测温的案例。

**📊 数据集**

实验使用UCI Air Quality数据集的气温与传感器读数。

**📈 对比分析**

比较了两模型在一般预测指标（均方误差、校准、鲁棒性）以及测量不稳定性指标，发现预测指标相同但测量结果系统性不一致。

**⚠️ 局限性**

局限性在于需要多次训练并对比模型，无法在单一模型上直接验证稳定性；且只验证了线性模型，未扩展到更复杂网络。

---

## 628. TEFormer: Structured Bidirectional Temporal Enhancement Modeling in Spiking Transformers

**arXiv ID:** 2601.18274 | [PDF](https://arxiv.org/pdf/2601.18274v1)

**作者:** Sicheng Shen `[一作]` (BrainCog Lab), Yi Zeng `[通讯]` (BrainCog Lab)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 TEFormer 的时序增强脉冲 Transformer 模型，通过双向时间融合提升 SNN 序列建模能力。

**💡 创新点**

创新点在于引入轻量化的 TEA 模块实现无参数前向时间融合，并在 MLP 中设计 T‑MLP 实现后向递归融合，构建生物启发的双向时间模型。

**🔧 技术方法**

采用了脉冲注意力（QK‑Attention/SSA）、LIF 神经元、可学习的 α 参数以及带门控的后向递归等技术。

**📊 数据集**

在 CIFAR10/100、SVHN、CIFAR10‑DVS、N‑CALTECH101、NCARS、sCIFAR、sMNIST、SHD、HMDB51‑DVS、UCF101‑DVS 等多种静态、事件与长序列数据集上评测。

**📈 对比分析**

与现有 Spiking Transformer（Spikformer、QKFormer、SDT、TIM 等）在统一 STEP 框架下比较，TEFormer 在所有基准上均显著提升准确率，尤其在不同编码方案下保持稳定优势。

**⚠️ 局限性**

局限在于仅在软件仿真与分类任务中验证，缺乏硬件部署与更复杂任务的实验，且脉冲编码标准仍未统一。

---

## 629. A Tumor Aware DenseNet Swin Hybrid Learning with Boosted and Hierarchical Feature Spaces for Large-Scale Brain MRI Classification

**arXiv ID:** 2601.18330 | [PDF](https://arxiv.org/pdf/2601.18330v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 630. Beyond Retention: Orchestrating Structural Safety and Plasticity in Continual Learning for LLMs

**arXiv ID:** 2601.18255 | [PDF](https://arxiv.org/pdf/2601.18255v1)

**作者:** Fei Meng `[一作]` `[通讯]` (Yangtze Delta Region Institute of Tsinghua University), Fei Meng (Yangtze Delta Region Institute of Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对LLMs持续学习进行实验，探讨经验重放对稳健与脆弱任务的不同影响，并提出Orthogonal Subspace Wake-up（OSW）方法。

**💡 创新点**

发现经验重放对结构脆弱任务产生严重负迁移，并提出基于梯度子空间正交投影的OSW，提供结构安全保证。

**🔧 技术方法**

采用LoRA参数高效微调、正交投影、SVD子空间估计以及经验重放对照。

**📊 数据集**

Qwen-1.5B模型，四任务序列：C-Stance、MeetingBank、Py150、ScienceQA。

**📈 对比分析**

与Seq、ER、Wake-up Only对照；OSW在保留代码生成任务性能的同时保持对新任务的高塑性，优于ER在脆弱任务上的表现。

**⚠️ 局限性**

需要更多任务、多尺度实验，正交投影参数选择可能限制对稳健任务的正迁移，且对大模型扩展的可行性待验证。

---

## 631. A Generative AI-Driven Reliability Layer for Action-Oriented Disaster Resilience

**arXiv ID:** 2601.18308 | [PDF](https://arxiv.org/pdf/2601.18308v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 632. Co-PLNet: A Collaborative Point-Line Network for Prompt-Guided Wireframe Parsing

**arXiv ID:** 2601.18252 | [PDF](https://arxiv.org/pdf/2601.18252v1)

**作者:** Chao Wang `[一作]` (Sichuan University), Hao Qin `[通讯]` (University College Dublin)

**通讯引用:** 1062 | [OpenAlex ID](https://openalex.org/A5057343419)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种点线协同的Wireframe解析框架Co-PLNet，利用早期点线检测结果生成空间提示并在后续阶段互相引导，最终得到精确且一致的线段与结点结构。

**💡 创新点**

核心创新是点线提示编码器与跨引导解码器的双向协同机制，借助稀疏多头交叉注意力在空间提示间共享信息，显著提升点线一致性与解析精度，同时保持实时性能。

**🔧 技术方法**

使用SuperPoint提取特征、U‑Net融合、点线提示编码器 (PLP‑Encoder) 生成空间提示、稀疏多头交叉注意力的Cross‑Guidance Line Decoder、线段终点验证 (LOI) 以及端到端的联合损失。

**📊 数据集**

在Wireframe和YorkUrban两个公开数据集上进行评估，采用结构平均精度 (sAP) 和帧率 (FPS) 作为评价指标。

**📈 对比分析**

与现有SOTA方法（如HAWPv2、PLNet、F‑Clip）对比，Co‑PLNet在sAP^15上取得约73‑74% 的精度，并以76.8 FPS实现实时推理，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括对光照变化、遮挡和极端视角的鲁棒性仍有待提升，且当前仅适用于单帧静态图像，未在闭环SLAM等动态场景中进行验证。

---

## 633. A Master Class on Reproducibility: A Student Hackathon on Advanced MRI Reconstruction Methods

**arXiv ID:** 2601.18314 | [PDF](https://arxiv.org/pdf/2601.18314v1)

**作者:** Lina Felsner `[一作]` (Technical University of Munich), Julia A. Schnabel `[通讯]` (King’s College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在技术讲座中组织学生团队，尝试重现MoDL、HUMUS-Net和无训练+物理模型三篇MRI重建论文的实验结果，并分析复现成功与失败原因

**💡 创新点**

强调可复现性的重要性，系统性地评估不同采样模式和跨域迁移对模型表现的影响，并结合FAIR与FUTURE‑AI原则提出改进建议

**🔧 技术方法**

利用PyTorch实现的模型未卷积网络（MoDL、HUMUS-Net）和自监督无训练+物理正则化方法，配合预训练权重和硬件加速

**📊 数据集**

采用fastMRI knee、fastMRI brain等公开MRI数据集，以及自制的变密度（VD）和GRAPPA式采样掩模

**📈 对比分析**

通过SSIM与PSNR指标与原论文对比，MoDL在VD掩模下表现一致但在GRAPPA掩模下降；HUMUS-Net在跨域脑数据上SSIM与训练域相近；无训练+物理模型无法复现

**⚠️ 局限性**

实验时间短、第三篇论文文档不足、代码依赖过时、缺失训练检查点导致无法复现，反映复现环境与文档完整性的关键性

---

## 634. TAM-Eval: Evaluating LLMs for Automated Unit Test Maintenance

**arXiv ID:** 2601.18241 | [PDF](https://arxiv.org/pdf/2601.18241v1)

**作者:** Elena Bruches `[一作]` (Siberian Neuronets LLC), Stanislav Moiseev `[通讯]` (T-Technologies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TAM-Eval框架与基准，用于在Python、Java和Go项目中自动化单元测试的创建、修复与更新；

**💡 创新点**

首次在测试文件级别、完整仓库上下文环境下构建基准，并采用无参考评估（通过率、覆盖率、变异测试）和统一、可迭代的提示策略；

**🔧 技术方法**

利用大语言模型（LLM）进行测试生成，配合Docker化执行、代码覆盖与变异工具（coveragepy、Jacoco、Go cover、mutpy等）实现自动化评测；

**📊 数据集**

构建了1,539个经过多阶段过滤与验证的真实项目场景，涵盖Python、Java、Go三种语言，覆盖测试创建、修复与更新三大任务；

**📈 对比分析**

对比GPT‑5、GPT‑OSS‑120B、Qwen3 Coder 480B、DeepSeek V3.1、Gemini 2.5 Flash等SOTA模型，评估指标为通过率、覆盖率提升与变异覆盖率提升；GPT‑5以约42%通过率、约20%覆盖率提升居首，但总体提升有限；

**⚠️ 局限性**

LLM在语义正确性、覆盖率提升和更新任务上仍表现不佳，失败率高；评测依赖于多次迭代反馈；基准仅覆盖三种语言，可能存在样本偏差。

---

## 635. Gradient-Informed Machine Learning in Electromagnetics

**arXiv ID:** 2601.18300 | [PDF](https://arxiv.org/pdf/2601.18300v1)

**作者:** Matteo Zorzetto `[一作]` (University of Padova), Sebastian Schöps `[通讯]` (Technische Universität Darmstadt)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用可微分的等几何分析(IGA)构建永久磁同步电机的参数化非线性模型，并通过POD降维与梯度增强的高斯过程回归(GPR)生成磁场分布与关键性能指标(KPI)的快速代理模型。

**💡 创新点**

① 将IGA的微分特性与参数梯度信息整合进POD基组和GPR训练，实现梯度增强的代理；② 在数据稀缺场景下显著提升预测精度；③ 直接对KPI进行梯度增强学习，避免重建完整磁场，进一步降低模型尺寸与计算成本。

**🔧 技术方法**

等几何分析(IGA)、主成分分析(POD)、高斯过程回归(GPR)（含梯度增强版）以及Sobol采样生成的参数化数据集。

**📊 数据集**

基于四个几何参数（MH、MW、MAG、Theta1）的PMSM模型，使用Sobol方法产生15、31、61、119个训练样本（每个样本需10秒求解共6177个未知量），并在相同的30个未见参数组合上测试。

**📈 对比分析**

对比梯度自由（GF）和梯度增强（GE）GPR在磁通密度重构与扭矩预测上的表现；结果显示GE在小样本（≤31样本）时平均相对误差降低约30%–50%，并在扭矩预测中达到MSE/MAE显著下降；直接对扭矩建模的GE-GPR在数据量充足时优于基于磁场的POD+GPR。训练时间虽因梯度信息增加而提升，但预测时间仅几毫秒。

**⚠️ 局限性**

① 需要在IGA求解器中实现梯度计算，导致实现复杂；② GE模型训练时间显著高于GF；③ POD截断误差仍是整体误差上限；④ 对参数空间维度高或非光滑KPI的情况，梯度增强效果有限；⑤ 代理模型对训练数据质量高度敏感，若采样不充分或几何失衡，性能下降。

---

## 636. U-Fold: Dynamic Intent-Aware Context Folding for User-Centric Agents

**arXiv ID:** 2601.18285 | [PDF](https://arxiv.org/pdf/2601.18285v1)

**作者:** Jin Su `[一作]` (Zhejiang University), Fajie Yuan `[通讯]` (Westlake University)

**通讯引用:** 2389 | [OpenAlex ID](https://openalex.org/A5081665927)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种动态、意图感知的上下文折叠框架U-Fold，用于在长对话和多工具调用场景下保持对话历史与关键信息的完整性并压缩上下文。

**💡 创新点**

创新点在于同时使用对话摘要与动态数据提取两大模块，实现意图跟踪与任务相关信息自适应提取，避免了传统静态压缩导致的约束丢失和意图漂移。

**🔧 技术方法**

技术核心包括LLM驱动的对话摘要器（生成摘要和待办清单）和基于摘要的LLM数据提取器（从工具日志中挑选相关字段），与ReAct框架无缝集成。

**📊 数据集**

使用了多种用户中心化基准：τ-bench、τ^2-bench、VitaBench，以及在这些基准上构造的更严格长上下文版本。

**📈 对比分析**

与ReAct及其它折叠基线（ReSum、IterResearch）对比，U-Fold在长上下文设置下平均提升约27%（最高可达71.4%胜率），在噪声多、工具繁多的任务中表现尤为突出。

**⚠️ 局限性**

局限包括折叠触发时每次都执行完整摘要与提取，导致额外开销；目前的基准仍不足以覆盖多会话、多用户的真实复杂场景，需进一步扩展测试集。

---

## 637. Beyond Pairwise Comparisons: A Distributional Test of Distinctiveness for Machine-Generated Works in Intellectual Property Law

**arXiv ID:** 2601.18156 | [PDF](https://arxiv.org/pdf/2601.18156v1)

**作者:** Anirban Mukherjee `[一作]` (Avyayam Holdings), Hannah Hanwen Chang `[通讯]` (Singapore Management University)

**通讯引用:** 1124 | [OpenAlex ID](https://openalex.org/A5101581868)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于最大均值差距（MMD）和语义嵌入的分布式方法，用来评估人类创作与机器生成过程的创意差异，从而衡量专利、版权、商标等知识产权的“新颖性”“原创性”“显著性”。

**💡 创新点**

创新点在于将单一作品的成对比较转化为对创作过程的整体分布比较，既克服了有限样本造成的“单位不匹配”问题，又避免了对训练数据的依赖，提供了可直接在法院、专利局使用的统计显著性检验框架。

**🔧 技术方法**

核心技术包括语义嵌入（文本使用Sentence‑Transformers，图像使用CLIP/CNN），核均值嵌入（KME）和无监督的两样本MMD检验，并通过置换检验实现 p‑value 计算。

**📊 数据集**

验证数据集包括：MNIST 手写数字（图像），USPTO/CCDV 专利摘要（文本），AI‑ArtBench（人类与两种扩散模型生成的艺术图像）。

**📈 对比分析**

比较方法：对比不同类别（数字、技术领域、艺术风格）之间的 MMD 统计，使用置换检验判定显著性。性能表现：在 MNIST 上仅需 5–10 样本即可达到 95% 以上检验功效；在专利文本中 7–15 篇摘要即可显著区分技术领域；在 AI‑ArtBench 中，即便人类评估者仅以 58% 的准确率区分，MMD 仍能显著检测到人类与 AI、以及不同模型之间的分布差异。

**⚠️ 局限性**

局限性包括：对语义嵌入的质量高度依赖；在极端噪声或水印扰动下嵌入可能失真；置换检验在样本极小或分布差异极微弱时可能功效不足；并且该方法只评估分布差异，无法直接识别具体的剽窃或复制实例，需要与记忆检测技术配合使用。

---

## 638. VissimRL: A Multi-Agent Reinforcement Learning Framework for Traffic Signal Control Based on Vissim

**arXiv ID:** 2601.18284 | [PDF](https://arxiv.org/pdf/2601.18284v1)

**作者:** Hsiao-Chuan Chang `[一作]` (National Yang Ming Chiao Tung University), I-Chen Wu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 2693 | [OpenAlex ID](https://openalex.org/A5016730899)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了VissimRL框架，实现了Vissim仿真器与强化学习的无缝集成，并支持单、双、多人学习。

**💡 创新点**

创新点：提供高层Python API封装Vissim COM，模块化状态/动作/奖励设计，兼容Gymnasium/PettingZoo，实现高精度仿真与RL结合的首个框架；通过批处理与缓存提升运行效率，支持多代理协作并在真实场景中出现“绿波”现象。

**🔧 技术方法**

技术：Python、Vissim COM接口、Gymnasium、PettingZoo、PPO（RLlib）、模块化插件化设计、批处理与缓存优化。

**📊 数据集**

数据集：合成网络（单交叉口、三交叉口）以及桃园代园交流区真实高峰时段交通流，一小时内车辆类型包括汽车、公交、摩托车。

**📈 对比分析**

比较方法：对比Vissim原始COM与VissimRL在开发代码量、每步延迟、吞吐量；对比RL控制（PPO+三种动作设计）与固定时序；结果显示VissimRL将代码量缩减84%，运行效率提升约1.1倍；RL控制相比固定时序降低约63%延迟、56%行驶时间、38%等待时间。

**⚠️ 局限性**

局限：目前仅兼容Vissim 2025版本，对更复杂城市网络、多模态交通（公交、非机动车）支持不完整；缺乏实时部署与安全性验证。

---

## 639. Depth to Anatomy: Learning Internal Organ Locations from Surface Depth Images

**arXiv ID:** 2601.18260 | [PDF](https://arxiv.org/pdf/2601.18260v1)

**作者:** Eytan Kats `[一作]` (University of Luebeck), Mattias P. Heinrich `[通讯]` (University of Luebeck)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种利用单张深度图自动定位内部器官并实现CT/磁共振扫描前自动调台的系统

**💡 创新点**

通过 Pixel‑to‑Voxel (Pix2Vox) 混合 2D‑3D CNN，直接从单视角深度图预测 3D 内部结构边界框和分割，无需中间三维重建或 SMPL 模型

**🔧 技术方法**

深度相机采集、3D 卷积网络 Pix2Vox、2D→3D 转换层、Dice + BCE 损失、数据增强、后处理

**📊 数据集**

德国全国队列 NAKO 全身 MRI 数据集（10,020 张），合成深度图与 41 个器官的人工标注

**📈 对比分析**

与 mean 模型和 2.5D CNN 进行对比；Pix2Vox 在冠状面平均定位误差 <10 mm，Dice >0.85，ASSD 较低；但在深度方向误差相对较大

**⚠️ 局限性**

合成深度与真实传感器存在域差；仅训练于标准 supine 体位，遮挡、覆盖和姿态变化对鲁棒性影响较大

---

## 640. Generative AI in Saudi Arabia: A National Survey of Adoption, Risks, and Public Perceptions

**arXiv ID:** 2601.18234 | [PDF](https://arxiv.org/pdf/2601.18234v1)

**作者:** Abdulaziz AlDakheel `[一作]` (Saudi Electronic University), Raed Alharbi `[通讯]` (Saudi Electronic University)

**通讯引用:** 912 | [OpenAlex ID](https://openalex.org/A5075926625)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开展了针对沙特阿拉伯国民的全国性问卷调查，系统评估了Generative AI（GenAI）的认知、使用频率、影响感知、培训需求、风险与信任以及数据共享行为。

**💡 创新点**

首次在中东地区提供了多维度、基于实证的GenAI使用基线，为Vision 2030数字化转型提供了第一手的公众认知与行为洞见，且通过构建综合指数和共选分析揭示了使用者的主题聚类和需求结构。

**🔧 技术方法**

采用结构化问卷（多选、Likert量表、开放式文本），并通过描述性统计、相关分析、方差分析、OLS回归、Jaccard相似度等定量方法和主题编码的定性方法进行数据处理与分析。

**📊 数据集**

基准数据集为330名沙特阿拉伯本土受访者的问卷响应，涵盖性别、年龄、地区、教育水平、行业、组织规模等背景信息，并对使用频率、认知水平、影响评价、培训兴趣、风险担忧与数据共享行为等指标进行量化。

**📈 对比分析**

通过对不同群体（年龄、教育、行业、就业状态）进行分组比较和回归预测，发现GenAI认知与使用频率呈正相关，行业背景显著影响认知；使用率高但对内容准确性和隐私风险担忧亦较为突出；综合影响指数（PII）平均为3.69/5，说明受访者普遍认为GenAI在提升工作效率和知识获取方面有效，但对批判性思维与决策支持的帮助有限。

**⚠️ 局限性**

局限性包括非概率便利抽样导致的代表性不足、主要基于自我报告可能产生的记忆偏差与社会期望偏差、数据收集时间点短暂且处于快速技术变革期，且研究结果局限于沙特阿拉伯，难以直接推广到其他阿拉伯或全球语境。

---

## 641. TC-IDM: Grounding Video Generation for Executable Zero-shot Robot Motion

**arXiv ID:** 2601.18323 | [PDF](https://arxiv.org/pdf/2601.18323v1)

**作者:** Weishi Mi `[一作]` (Beijing Innovation Center of Humanoid Robotics), Jian Tang `[通讯]` (Beijing Innovation Center of Humanoid Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了 Tool‑Centric Inverse Dynamics Model (TC‑IDM)，通过从世界模型生成的视频中提取工具（手爪）轨迹，将高层视觉规划转化为可执行的低层控制指令；

**💡 创新点**

核心创新在于将工具轨迹作为中间表示，分离视觉驱动的把手控制和几何驱动的姿态规划，实现对多种工具、不同视角和柔性物体的稳健执行；

**🔧 技术方法**

采用预训练视觉编码器 DINOv3、Segment Anything Model 3 (SAM 3) 与 3D 点轨迹追踪器、SE(3) 解析运动恢复，并在此基础上构建两路 MLP 预测头；

**📊 数据集**

使用公开的世界模型（如 Sora、Kling、Cosmos、WoW 等）生成的视频序列以及实际机器人抓取与操纵的真实实验数据；

**📈 对比分析**

在多难度任务（易/中/难）和多世界模型上进行对比，TC‑IDM 在简单任务上 77.7%、总体平均 61.11%、零射线柔性物体 38.46%，显著优于端到端 VLA 基线和其他 IDM 方法；

**⚠️ 局限性**

局限性包括对精细分割与 3D 追踪的依赖，极端遮挡或光照变化时可能导致轨迹误检，以及对复杂动力学约束（如高负载或高速运动）的适应性仍需进一步提升。

---

## 642. CovertComBench: The First Domain-Specific Testbed for LLMs in Wireless Covert Communication

**arXiv ID:** 2601.18315 | [PDF](https://arxiv.org/pdf/2601.18315v1)

**作者:** Zhaozhi Liu `[一作]` (South-Central Minzu University), Zan Zhou `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 463 | [OpenAlex ID](https://openalex.org/A5056148718)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个针对无线隐蔽通信的LLM基准CovertComBench，并通过人工专家和LLM-as-Judge两种评估方式对LLM在概念理解、数学推导和代码实现三类任务进行系统评估。

**💡 创新点**

创新点在于①针对隐蔽通信的约束优化问题设计三种任务形式（MCQ、ODQ、CGQ）；②提出多维度评估框架并量化LLM评判器的可靠性；③从实验中揭示LLM在高阶数学推导方面的瓶颈，为后续工具辅助和负样本训练提供依据。

**🔧 技术方法**

技术手段包括：多轮提示工程、自动脚本评估、手工评分过程、KL散度检测、代码执行测试循环、LLM-as-Judge框架与人类专家评分对比。

**📊 数据集**

数据集为CovertComBench，总计约500道题，覆盖IRS、NOMA、MIMO等现代通信模型，按难度层级（中等、困难、极难、专家）划分，并包含完整的元数据和参考答案。

**📈 对比分析**

与多种API与本地LLM（如DeepSeek、Gemini、GPT-oss、Llama3、Mistral等）在三类任务上进行对比；在概念理解和代码实现上达80%+准确率，但在多步数学推导任务的准确率仅为18%–55%，并且LLM评判器与人类评估之间存在较大MAE，表明评判偏差。

**⚠️ 局限性**

局限性在于：①LLM在严苛的安全约束下的数学推理能力不足；②评判器缺乏细粒度评分，易产生极端得分；③代码生成易出现幻觉并难以自我纠错；③缺乏与外部符号计算工具的无缝结合。

---

## 643. Dicey Games: Shared Sources of Randomness in Distributed Systems

**arXiv ID:** 2601.18303 | [PDF](https://arxiv.org/pdf/2601.18303v1)

**作者:** Léonard Brice `[一作]` (Institute of Science and Technology Austria), K. S. Thejaswini `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5057216103)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出并研究了Dicey Games——一种在分布式系统中通过有限共享随机源来协调多玩家与单一对手的并发零和博弈，给出了最优策略的存在性、结构和计算复杂度，并探讨了在共享随机资源受限时的最优分配问题。

**💡 创新点**

创新点主要包括：①引入Dicey Games框架，统一描述多玩家共享随机源的博弈；②证明任何可行策略都可转化为k‑grid策略（即只需在每个骰子上划分k个区间），并给出最优值的解析与数值近似；③利用实代数几何、Fritz‑John条件和Cylindrical Algebraic Decomposition推导出最优策略与最优值的指数大小上界；④把阈值决策与最优值计算问题归入∃理论与EXP空间，给出对应的硬度证明。

**🔧 技术方法**

主要技术：Carathéodory定理、几何切片与重整化、实代数几何（Collins、Basu‑Pollack‑Roy的算法）、Fritz‑John最优性条件、可满足性/存在性理论（∃理论）以及复杂度类（∃、EXP、^∃）的运用。

**📊 数据集**

没有使用传统机器学习或实验数据集；研究完全基于理论模型，主要通过匹配硬币（matching pennies）以及其多玩家、共享骰子图的实例来说明和验证结论。

**📈 对比分析**

与独立随机化（1/2ⁿ）和全共享随机化（1/2）对比，Dicey Games在共享随机源受限时仍能显著提升赢率（如三人匹配硬币可达≈0.2781），并给出了最优策略的具体构造；从算法视角来看，阈值问题可在EXP时间内解答，最优值计算可在EXP空间完成，虽然仍为高复杂度，但给出了最优解的可达性与结构。

**⚠️ 局限性**

局限性：总体复杂度仍是指数级（阈值决策∈∃，最优值计算EXP空间）；对实际大规模系统难以直接求解；最优策略的数值解常为代数数，符号表达繁复；对共享随机源分配的完整优化仍有开放问题，尤其在约束稀疏或结构化情形下的有效算法尚未给出。

---

## 644. Contextual Range-View Projection for 3D LiDAR Point Clouds

**arXiv ID:** 2601.18301 | [PDF](https://arxiv.org/pdf/2601.18301v1)

**作者:** Seyedali Mousavi `[一作]` (Mälardalen University), Masoud Daneshtalab `[通讯]` (Mälardalen University)

**通讯引用:** 4176 | [OpenAlex ID](https://openalex.org/A5063193249)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了两种基于上下文的二维视图投影方法（CAP与CWAP），用于解决LiDAR点云投影时的多对一映射冲突，并通过改进投影策略提升语义分割性能。

**💡 创新点**

创新点在于：①将实例中心距离作为分数来调整深度优先级（CAP），优先保留对象核心点；②通过可自定义类别权重调节投影优先级（CWAP），实现针对性提升。

**🔧 技术方法**

核心技术包括：基于三维高斯分布的中心度计算、加权投影准则、二维范围图生成、以及与RangeViT网络的结合。

**📊 数据集**

在SemanticKITTI数据集上进行实验，利用公开的点云和标签进行训练与评估。

**📈 对比分析**

与传统仅按最小深度挑选的投影方法对比，CAP在实例类上提升了约3.1% mIoU；CWAP在指定类别（如卡车、摩托车）上实现了显著的准确率提升（高达20.7%），但对部分其它类别有轻微负面影响。

**⚠️ 局限性**

主要局限：投影改进仅在训练阶段使用，推理时仍回退到深度优先投影；缺乏推理时的中心度或类别权重估计，导致无法在部署时完全利用上下文信息；对类别权重设定敏感，需人工调参。

---

## 645. Designing large language model prompts to extract scores from messy text: A shared dataset and challenge

**arXiv ID:** 2601.18271 | [PDF](https://arxiv.org/pdf/2601.18271v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 646. Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning

**arXiv ID:** 2601.18296 | [PDF](https://arxiv.org/pdf/2601.18296v1)

**作者:** Zhaoyan Gong `[一作]` (Zhejiang University), Wen Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 16439 | [OpenAlex ID](https://openalex.org/A5100444425)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种名为 Temp‑R1 的自主端到端 Temporal Knowledge Graph Question Answering（TKGQA）代理，利用强化学习实现多步时序推理，能够在不依赖固定工作流或昂贵 API 的情况下完成复杂时间约束问题的解答。

**💡 创新点**

创新点主要包括：①将内部推理拆分为 <plan>、<filter>、<rank> 三个专用动作，显著降低单一“think”标签的认知负荷；②引入逆向课程学习（先训练难题后放宽），有效避免模型在简单问题上形成捷径并陷入路径依赖；③通过 Group Relative Policy Optimization（GRPO）对策略进行强化学习，从而让模型在无需预先定义工作流的情况下自主探索多样化的解题路径；④整合这些设计，构建了可扩展、成本低且在开源 8B 模型上实现 GPT‑4o 级别甚至更高性能的系统。

**🔧 技术方法**

使用的技术包括：大语言模型（Llama3.1‑8B‑Instruct 作为主体）、检索器 + RAG、ReAct‑style 代理框架、扩展的内部/外部动作空间、监督式冷启动（SFT）与后续的强化学习（GRPO），以及逆向课程学习策略。

**📊 数据集**

数据集：MultiTQ、TimelineKGQA‑Cron（in‑domain）和 TimelineKGQA‑ICEWS‑Actor（out‑of‑domain），以及用于构建 SFT 轨迹的高质量 GPT‑4o 生成样本。

**📈 对比分析**

通过 Hits@1 指标与多种基线对比，包括基于嵌入、提示工程、微调 LLM、闭源 GPT‑4o、Search‑R1 等；结果显示 Temp‑R1 在 MultiTQ 上整体得分 0.780，复杂多跳、时序约束子任务提升 19.8%，在 TimelineKGQA‑ICEWS‑Actor 上仍保持领先，表现优于 GPT‑4o、PoK 等强基线。

**⚠️ 局限性**

局限性：实验仅覆盖到 8B 参数模型，未验证更大规模模型的可扩展性；逆向课程学习的效果目前仅在 TKGQA 任务中得到验证，尚不确定其在更广泛推理任务中的普适性；以及由于算力限制，无法在更大数据量或更复杂任务上进一步评估。

---

## 647. Think-Augmented Function Calling: Improving LLM Parameter Accuracy Through Embedded Reasoning

**arXiv ID:** 2601.18282 | [PDF](https://arxiv.org/pdf/2601.18282v1)

**作者:** Lei Wei `[一作]`, Bin Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在函数签名中加入 "think" 参数，使 LLM 在调用函数前能够生成并记录对参数和函数本身的推理过程，从而提升函数调用的准确性与可解释性。

**💡 创新点**

核心创新在于：①将推理过程作为结构化参数嵌入函数签名；②基于参数复杂度评分自动触发细粒度推理；③采用动态描述调优与工具描述优化，实现对推理质量与人类期望的一致性。

**🔧 技术方法**

使用了函数签名自动增补、复杂度评分与阈值触发机制、Markov 生成链模型、提示层与 token 层的描述优化、Meta‑LLM 迭代改进以及黑盒工具描述优化等技术。

**📊 数据集**

在 ToolBench（包含约 16,000 个 REST API）上进行评测，使用 I1‑Inst、I2‑Inst、I3‑Inst 三类指令，结合 ToolEval 评估协议。

**📈 对比分析**

与传统函数调用（Standard FC）相比，在 Pass Rate 与 Win Rate 两项指标上均有提升，尤其在多参数、跨集合工具调用场景中提升显著，模型规模越小提升越明显。

**⚠️ 局限性**

限制主要包括：对简单单参数函数可能产生冗余推理导致复杂度提升；在极其复杂的跨域任务中仍与专有模型存在性能差距；依赖手工设定的复杂度阈值与提示模板，调优成本相对较高。

---

## 648. When Nobody Around Is Real: Exploring Public Opinions and User Experiences On the Multi-Agent AI Social Platform

**arXiv ID:** 2601.18275 | [PDF](https://arxiv.org/pdf/2601.18275v1)

**作者:** Qiufang Yu `[一作]` (Fudan University), Xingyu Lan `[通讯]` (Fudan University)

**通讯引用:** 470 | [OpenAlex ID](https://openalex.org/A5046324646)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对社交平台 Social.AI 的 883 条公开评论进行内容分析，并开展了 20 名参与者的 7 天日记研究，探讨公众对多代理 AI 社交平台的态度与实际用户体验。

**💡 创新点**

创新点在于首次系统性研究多代理 AI 社交平台的社会心理效应，揭示人类如何将社交期望投射到 AI 代理上，以及这些期望如何在实际使用中破碎并提出针对性设计建议。

**🔧 技术方法**

采用了自然语言处理辅助的归纳式内容分析与主题分析技术，对评论文本和日记数据进行编码和归纳，结合 CASA 理论框架解释人机社交行为。

**📊 数据集**

数据集包括从 Reddit、Twitter、Instagram 等社交媒体抓取的 883 条 Social.AI 相关评论，以及 20 名美国大学生在 7 天内自述的日记条目。

**📈 对比分析**

研究通过对比公共评论与日记体验两种数据源，未涉及传统性能指标；结果表明公众关注风险与宏观问题，而实际用户体验则呈现短暂满足与后续失望，体现了人机交互的“双重性”。

**⚠️ 局限性**

局限性包括仅聚焦单一平台（Social.AI）、样本规模小、参与者为美国大学生且缺乏跨文化对比，且研究为横断面，无法评估长期使用对情感与社交行为的持续影响。

---

## 649. Neural Network Approximation: A View from Polytope Decomposition

**arXiv ID:** 2601.18264 | [PDF](https://arxiv.org/pdf/2601.18264v1)

**作者:** ZeYu Li `[一作]`, FengLei Fan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

本研究探讨了ReLU网络的通用逼近能力，提出了一种基于多面体分解的更现实的逼近理论，旨在填补现有理论与实际学习能力之间的差距。

**💡 创新点**

创新点在于引入了多面体分解方法，提供了一种更灵活和高效的网络结构，能够根据目标函数的局部规律进行自适应调整，尤其是在目标函数的奇异点附近。

**🔧 技术方法**

使用了显式的核多项式方法（Kernel Polynomial Method, KPM）来推导连续函数的通用逼近，并构建了ReLU网络来分别在每个子域上进行逼近。

**📊 数据集**

论文中未具体提及使用的数据集，但讨论了对连续函数和解析函数的逼近能力。

**📈 对比分析**

与现有方法相比，基于多面体分解的逼近在许多情况下表现出更高的效率和灵活性，尤其是在处理目标函数的奇异点时，逼近误差显著降低。

**⚠️ 局限性**

限制在于该方法的理论构建仍需与实际学习能力相结合，未来的研究方向包括如何将该构建与梯度下降学习相结合。

---

## 650. FGGM: Fisher-Guided Gradient Masking for Continual Learning

**arXiv ID:** 2601.18261 | [PDF](https://arxiv.org/pdf/2601.18261v1)

**作者:** Chao-Hong Tan `[一作]` (Alibaba Group), Jieping Ye `[通讯]` (Alibaba Group)

**通讯引用:** 38771 | [OpenAlex ID](https://openalex.org/A5010419481)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于 Fisher 信息矩阵的梯度遮罩框架 FGGM，用于在不访问历史数据的情况下实现大语言模型的连续学习，缓解灾难性遗忘。

**💡 创新点**

创新点在于使用理论可解释的 Fisher 信息量化参数重要性，动态生成二进制遮罩并采用输入维度聚合，硬遮罩梯度投影，显著优于经验阈值方法（如 MIGU），同时无需存储历史数据。

**🔧 技术方法**

核心技术包括对角 Fisher 信息矩阵近似、阈值化二值遮罩、输入维度聚合、梯度投影硬遮罩、基于 Qwen2 的大模型实现。

**📊 数据集**

使用了 TRACE 任务序列、General 评估基准（MMLU、BBH、TyDiQA、PIQA、BoolQ、GSM8K）、Magicoder‑Evol‑Instruct‑110K、HumanEval 以及 BBH 数据集。

**📈 对比分析**

与 ORI、EWC、REP、SFT、LORA、MIGU 等基线在 1.5B/7B 规模下对比，FGGM 在 General（稳定性）和 TRACE‑OP（整体性能）上分别比 MIGU 提升 9.6% 与 4.4%，并显著降低忘记率。

**⚠️ 局限性**

局限包括需额外计算 FIM 与遮罩的开销，仍可能对极大模型造成内存/计算压力，且目前仅在单模态语言模型上验证，缺乏对多模态或更大规模的通用性探讨。

---

## 651. BoRP: Bootstrapped Regression Probing for Scalable and Human-Aligned LLM Evaluation

**arXiv ID:** 2601.18253 | [PDF](https://arxiv.org/pdf/2601.18253v1)

**作者:** Peng Sun `[一作]` (Alibaba Group), Duan Wu `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 BoRP 框架，在 LLM 隐藏空间中进行几何感知的回归探测，以高保真度评估用户满意度并实现自动化 rubric 生成与低成本高吞吐量评估。

**💡 创新点**

创新点在于：① 通过 Polarization Index 进行无监督极端样本挖掘并自动生成 rubric；② 使用 PLS 回归代替传统生成式评估；③ 引入层间一致性作为置信度估计，提升评估可靠性。

**🔧 技术方法**

采用了对比提示编码、Polarization Index 采样、集合蒸馏、PLS 回归、层级一致性不确定度、Suffix‑Only 推理与 KV‑cache 重用、INT4‑AWQ 与 BF16 量化、以及 CUPED 方差减少等技术。

**📊 数据集**

使用了内部 700 条专家标注的对话日志和公开的 HelpSteer2 语料库（聚焦多轮对话的 verbosity 维度）。

**📈 对比分析**

与 Qwen3‑14B/Max 等生成式评估基线对比，BoRP 在工业数据上取得 K‑α 0.796、Pearson 0.806，超越更大模型；在 HelpSteer2 上保持约 0.7 的 K‑α；吞吐量提升约 7.9 倍，成本降低 169 倍。

**⚠️ 局限性**

局限性在于：仅采用线性回归，难以深入评估复杂推理任务；目前仅验证单一语言模型与内部数据，跨模态或多模型扩展仍待探索。

---

## 652. Discriminability-Driven Spatial-Channel Selection with Gradient Norm for Drone Signal OOD Detection

**arXiv ID:** 2601.18329 | [PDF](https://arxiv.org/pdf/2601.18329v1)

**作者:** Chuhan Feng `[一作]` (Xidian University), Fengkui Gong `[通讯]` (Xidian University)

**通讯引用:** 2675 | [OpenAlex ID](https://openalex.org/A5034754948)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于空间‑通道可辨识度选择与梯度范数的无人机信号OOD检测算法。

**💡 创新点**

创新点在于将时频图自适应加权的空间通道选择与梯度范数度量预测对扰动的敏感度相结合，实现静态能量得分与动态不稳定性指标的融合。

**🔧 技术方法**

采用了MobileNetV2特征提取、时频图构造、空间‑通道加权、梯度范数与能量得分融合以及z‑score标准化等技术。

**📊 数据集**

使用了DroneRFa数据集，包括15种无人机信号与背景噪声，并在-15dB至15dB的SNR范围内进行实验。

**📈 对比分析**

与五种基线方法（MobileNetV2、Softmax、Uncertainty、VAE、DDCS）比较，DDSCS在准确率95.18%、召回率98.65%、F1 96.42%和AUROC 95.77%等指标上表现最佳，且在低SNR下保持良好鲁棒性。

**⚠️ 局限性**

局限性包括对极低SNR下的鲁棒性仍有提升空间，以及对完全未知干扰类型的泛化能力需要进一步验证。

---

## 653. TechING: Towards Real World Technical Image Understanding via VLMs

**arXiv ID:** 2601.18238 | [PDF](https://arxiv.org/pdf/2601.18238v1)

**作者:** Tafazzul Nadeem `[一作]` (Indian Institute of Technology Kanpur), Ashutosh Modi `[通讯]` (Indian Institute of Technology Kanpur)

**通讯引用:** 1988 | [OpenAlex ID](https://openalex.org/A5076043215)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

暂无具体论文内容，无法说明研究工作

**💡 创新点**

无法确定创新点

**🔧 技术方法**

未知技术

**📊 数据集**

未知数据集

**📈 对比分析**

未知方法比较及性能

**⚠️ 局限性**

缺乏信息，无法评估限制

---

## 654. Integrating Fine-Grained Audio-Visual Evidence for Robust Multimodal Emotion Reasoning

**arXiv ID:** 2601.18321 | [PDF](https://arxiv.org/pdf/2601.18321v1)

**作者:** Zhixian Zhao `[一作]` (Northwestern Polytechnical University), Lei Xie `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 9595 | [OpenAlex ID](https://openalex.org/A5066245750)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于结构化证据分解（SED）和一致性感知对齐（CA‑DPO）的多模态情绪推理框架SABER‑LLM，并构建600K视频的六维细粒度情绪数据集SABER。

**💡 创新点**

创新点在于：① 通过SABER数据集提供细粒度视觉、语音、身体语言等六维注释，填补情绪推理细粒度标注空白；② 采用SED将感知与推理拆分，强制模型先提取独立的视觉/音频证据再进行因果推理；③ 引入CA‑DPO对冲突场景进行偏好对齐，显著抑制单模态主导与幻觉。

**🔧 技术方法**

技术包括：多模态大型语言模型（Qwen2.5‑Omni‑7B）作为基座，SFT训练SABER数据集实现SED，后续使用CA‑DPO优化逻辑一致性；评估时使用GPT‑4o等人工评价工具。

**📊 数据集**

使用数据集：SABER（600K视频，六维细粒度标注）以及官方基准EMER、EmoBench‑M和自建的SABER‑Test（1800样本，包含一致/不一致子集）。

**📈 对比分析**

与多种开源模型（Qwen2.5‑Omni‑3B/7B、Qwen3‑Omni‑30B、Intern‑S1‑9B等）和闭源模型（Gemini‑2.5‑Pro）对比，SABER‑LLM在EmoBench‑M平均准确率达到59.88%（同类开源模型最高），在EMER的“Clue Overlap”得分8.25（领先所有开源模型），在SABER‑Test的冲突子集上保持高达2.65的AFD分数，显示出优越的鲁棒性与参数效率。

**⚠️ 局限性**

局限性包括：① 仍需更大规模的跨文化、多模态对齐数据以覆盖更广泛的情绪表达；② 对极端噪声或极度细粒度的非语言线索仍可能出现感知误差；③ 需要更高效的对齐算法以降低计算成本，尤其在大模型上训练时仍占用显著GPU资源。

---

## 655. MultiVis-Agent: A Multi-Agent Framework with Logic Rules for Reliable and Comprehensive Cross-Modal Data Visualization

**arXiv ID:** 2601.18320 | [PDF](https://arxiv.org/pdf/2601.18320v1)

**作者:** Jinwei Lu `[一作]` (Hong Kong Polytechnic University), Raymond Chi-Wing Wong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5933 | [OpenAlex ID](https://openalex.org/A5049858061)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 MultiVis-Agent，一个基于四层逻辑规则的多智能体框架，用于可靠地生成和迭代多模态可视化。

**💡 创新点**

创新点在于：1）引入四层逻辑规则为 LLM 行为提供数学约束，实现系统可靠性保证；2）提出 MultiVis 任务公式化与 MultiVis‑Bench 基准，覆盖基本生成、图像引用、代码引用、迭代改进四大场景；3）结合多模态输入与迭代改进的端到端评估方法。

**🔧 技术方法**

使用技术包括：大型语言模型（Gemini、GPT）、多智能体协作架构、图像‑语言模型、SQL 生成与执行、Altair 代码生成、验证与评估 Agent 以及结构化与感知层双重评估。

**📊 数据集**

数据集：141 个 SQLite 数据库，127 种图表模板，构建的 MultiVis‑Bench 共 1,202 条案例，涵盖四类场景。

**📈 对比分析**

通过与 Instructing LLM、LLM Workflow、nvAgent 三种基线在 MultiVis‑Bench 上对比，采用双层（结构化 + 感知）评估，MultiVis‑Agent 在四类场景均实现整体可视化分数 75.63% 以上、任务完成率 99.58% 以上、代码执行成功率 94.56% 以上，显著优于基线。

**⚠️ 局限性**

局限性：对极端复杂逻辑或长序列交互仍可能超出迭代上限；系统性能受所选 LLM 能力限制；逻辑规则在新场景下需要手工扩展或重新定义。

---

## 656. Convex Chance-Constrained Stochastic Control under Uncertain Specifications with Application to Learning-Based Hybrid Powertrain Control

**arXiv ID:** 2601.18313 | [PDF](https://arxiv.org/pdf/2601.18313v1)

**作者:** Teruki Kato `[一作]` (Toyota Central Research and Development Labs Inc), Kenji Kashima `[通讯]` (Kyoto University)

**通讯引用:** 3722 | [OpenAlex ID](https://openalex.org/A5057149593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个严格凸的机会约束随机控制框架，联合优化控制输入与风险分配，并将其推广至学习式可精确线性化模型，最终应用于混合动力电机控制。

**💡 创新点**

创新点在于同时考虑控制规范的不确定性，保证概率约束可满足且通过正则化风险分配实现严格凸性与唯一解，弥补了现有方法无法兼顾(i)-(iv)的不足。

**🔧 技术方法**

使用机会约束分解与凸松弛、风险分配正则化、神经网络参数化的可精确线性化模型、模型预测控制、IPOPT/OSQP求解器以及Monte Carlo估计期望与方差。

**📊 数据集**

数据主要来自混合动力系统的仿真生成，包含车速、发动机转矩、电池SoC等时序数据；请求加速度按高斯分布采样构造未来速度轨迹，未使用公开实验数据集。

**📈 对比分析**

通过与确定性控制、均匀风险分配的随机控制对比，结果显示优化风险分配的方案在满足SoC约束、降低发动机转矩使用、提升燃油经济性方面优于其他两种方法，并保持总体风险等于允许值；而确定性控制的实际违约概率远超容许阈值。

**⚠️ 局限性**

局限在于求解效率仍需提升以满足实时控制需求；正则化风险分配需人工调参；对非线性模型的推广受限于可精确线性化假设；目前验证仅在仿真环境，缺乏实车实验验证。

---

## 657. MarioChart: Autonomous Tangibles as Active Proxy Interfaces for Embodied Casual Data Exploration

**arXiv ID:** 2601.18328 | [PDF](https://arxiv.org/pdf/2601.18328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 658. A Heterogeneous Massive MIMO Technique for Uniform Service in Cellular Networks

**arXiv ID:** 2601.18298 | [PDF](https://arxiv.org/pdf/2601.18298v1)

**作者:** Wei Jiang `[一作]` (German Research Center for Artificial Intelligence), Hans D. Schotten `[通讯]` (Technische Universität)

**通讯引用:** 9998 | [OpenAlex ID](https://openalex.org/A5008473850)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种将中心基站大天线阵列与边缘接入点融合的异构大规模MIMO网络架构，旨在兼顾优质服务与成本效益。

**💡 创新点**

创新点在于通过在每个小区内部分集中天线与边缘AP相结合，实现与全分布式无单元网络（CF‑mMIMO）相近的用户公平率，却显著降低了AP站点数量和光纤互连成本。

**🔧 技术方法**

使用了TDD制导的MMSE信道估计、最大比合并与共轭波束成形、最大最小功率控制，以及相关Rayleigh衰落与高斯局部散射模型的空间相关性分析。

**📊 数据集**

实验数据基于1 km×1 km平面模拟，划分为四个500 m×500 m小区，采用512根天线、32个用户；CF‑mMIMO采用128个4天线AP，使用COST‑Hata路径损耗模型和8 dBσ的对数正态阴影。

**📈 对比分析**

与传统单元式大规模MIMO和全分布式CF‑mMIMO进行CDF对比，结果显示在95%可靠率下，HmMIMO（1/4天线聚合）可达0.65 bps/Hz，接近CF‑mMIMO（0.61 bps/Hz），远超单元式网络（0.013 bps/Hz）；采用最大最小功率控制后两者进一步提升至约2 bps/Hz，且HmMIMO通过减少约25%–50%AP站点实现更低的基础设施成本。

**⚠️ 局限性**

局限性包括：需在每个小区内部署至少一部分分布式AP，仍受光纤互连延迟与维护成本影响；仿真基于理想化模型，未考虑大规模部署中的干扰、时延和硬件非理想性等实际问题。

---

## 659. Complex-Valued-Matrix Permanents: SPA-based Approximations and Double-Cover Analysis

**arXiv ID:** 2601.18232 | [PDF](https://arxiv.org/pdf/2601.18232v1)

**作者:** Junda Zhou `[一作]` (Chinese University of Hong Kong), Pascal O. Vontobel `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2931 | [OpenAlex ID](https://openalex.org/A5028154201)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究使用双边缘正规因子图（DE-NFG）与求和乘积算法（SPA）对复值矩阵的行列式（permanent）进行近似，探讨其在不同复数分布下的固定点行为与Bethe近似的有效性。

**💡 创新点**

创新点在于将传统的正实矩阵Bethe近似与图覆盖分析扩展到复值矩阵，首次提出并验证DE-NFG上的SPA、双重覆盖计数、以及在复值情形下的Bethe自由能的一致性与极限性质；同时揭示当矩阵分布趋向标准复高斯时，SPA收敛退化以及Bethe近似失效的阈值。

**🔧 技术方法**

主要技术包括：双边缘正规因子图建模、求和乘积算法（含中点阻尼和随机初始化）、Wirtinger 微积分求解复值Bethe自由能的极值条件、图覆盖计数（尤其是双重覆盖）、以及组合学工具（对称群循环指数）来推导二阶矩的闭式与渐近。

**📊 数据集**

使用的“数据集”是人工生成的随机复矩阵：每个条目独立同分布在以角度α∈[0,π]定义的扇形Sα内（包括α=0的全1矩阵、α=π的标准复高斯矩阵等），并以不同n（如4、10）进行实验。

**📈 对比分析**

比较方法：在相同随机矩阵集上计算真值Z、SPA得到的Bethe近似Z_、以及其二阶覆盖近似Z_,2；通过比值Z/Z_、Z/Z_,2以及其对数来评估误差，并与解析的渐近公式对比。实验显示：当α较小（靠近全1）时，Z/Z_≈(Z/Z_,2)^2 成立且误差随n增长趋于0；当α趋近π（复高斯）时，误差显著增大，SPA固定点退化为对角矩阵，近似性能退化。

**⚠️ 局限性**

局限性：① 对于接近复高斯分布的矩阵，SPA的Bethe近似失效，导致大误差；② 复值Bethe自由能缺乏凸性，极值点不一定全局最优；③ 分析目前仅覆盖零均值复矩阵和全1矩阵的特例，未能给出对任意复矩阵的通用误差界；④ 只关注二阶覆盖，尚未研究更高阶覆盖的潜在改进。

---

## 660. Quest2ROS2: A ROS 2 Framework for Bi-manual VR Teleoperation

**arXiv ID:** 2601.18289 | [PDF](https://arxiv.org/pdf/2601.18289v1)

**作者:** Jialong Li `[一作]` (Lund University), Volker Krueger `[通讯]` (Lund University)

**通讯引用:** 708 | [OpenAlex ID](https://openalex.org/A5109357557)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

开发了一个基于ROS 2的双臂VR遥操作框架Quest2ROS2，支持相对运动控制、可视化、夹爪控制和暂停重置功能。

**💡 创新点**

创新点在于使用VR控制器相对运动实现独立姿态控制，解除工作空间限制，并提供模块化、可插拔的架构与“镜像”控制模式。

**🔧 技术方法**

使用Meta Quest 2/3 VR控制器、ROS 2通信、Rviz可视化、ROS-TCP-Endpoint、Cartesian impedance controller等技术。

**📊 数据集**

未使用公开数据集，主要通过手动演示和仿真测试验证框架。

**📈 对比分析**

通过对比原Quest2ROS的绝对坐标复制方法，实验展示了更大的工作空间和更顺畅的双臂协同；性能以演示视频为主，无量化指标。

**⚠️ 局限性**

局限性包括对控制器视角的依赖、需要手动重置姿态、夹爪仅支持开闭切换，且在极限姿态下仍可能出现运动跳跃。

---

## 661. Orchestrating Specialized Agents for Trustworthy Enterprise RAG

**arXiv ID:** 2601.18267 | [PDF](https://arxiv.org/pdf/2601.18267v1)

**作者:** Xincheng You `[一作]` (Atlassian), Sean Culatana `[通讯]` (Atlassian)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ADORE框架，用多代理迭代式检索与写作替代传统RAG，实现长篇企业报告的可追溯生成。

**💡 创新点**

创新点在于以结构化Memory Bank约束生成、基于章节证据覆盖的自适应检索循环以及证据驱动的终止判据。

**🔧 技术方法**

使用大型语言模型、专门化检索、规划、执行、报告生成代理，配合自进化检索引擎和结构化证据图。

**📊 数据集**

在DeepResearch Bench、内部企业基准和DeepConsult数据集上进行评测。

**📈 对比分析**

通过RACE指标与无参考侧对侧评估，ADORE在DeepResearch Bench获得52.65分、DeepConsult获77.21%胜率，均领先同类系统。

**⚠️ 局限性**

仍受限于对复杂多模态内容的支持不足、需人工参与计划校对以及高计算成本。

---

## 662. A Mechanical Wi-Fi Antenna Device for Automatic Orientation Tuning with Bayesian Optimization

**arXiv ID:** 2601.18256 | [PDF](https://arxiv.org/pdf/2601.18256v1)

**作者:** Akihito Taya `[一作]` (University of Tokyo), Kaoru Sezaki `[通讯]` (University of Tokyo)

**通讯引用:** 3897 | [OpenAlex ID](https://openalex.org/A5050720322)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种可自动调节Wi‑Fi天线方向的机械装置，并通过贝叶斯优化实现方向自适应。

**💡 创新点**

首次将机械天线重配置与贝叶斯优化结合，使非专业用户能够自动提升网络性能。

**🔧 技术方法**

使用Picoscenes采集CSI，计算MIMO‑OFDM信道容量，并采用基于UCB的贝叶斯优化寻找最优天线方向。

**📊 数据集**

实验使用自制设备在室内2 m LoS环境下收集的CSI与RSSI数据，没有公开数据集。

**📈 对比分析**

与随机搜索和Sobol序列搜索对比，贝叶斯优化在更少试验次数下获得更高吞吐量，最大提升约70 Mbps。

**⚠️ 局限性**

局限在单个AP、静态室内环境，对动态移动场景或多AP部署的适用性尚未验证。

---

## 663. Tractable Gaussian Phase Retrieval with Heavy Tails and Adversarial Corruption with Near-Linear Sample Complexity

**arXiv ID:** 2601.18245 | [PDF](https://arxiv.org/pdf/2601.18245v1)

**作者:** Santanu Das `[一作]` (Tata Institute of Fundamental Research), Jatin Batra `[通讯]` (Tata Institute of Fundamental Research)

**通讯引用:** 197 | [OpenAlex ID](https://openalex.org/A5059145110)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了多项式时间算法，利用鲁棒PCA和截断技术实现对带有重尾噪声和强对抗性误差的相位恢复的鲁棒初始化和梯度下降。

**💡 创新点**

首次将鲁棒PCA与相位恢复的谱初始化关联，获得近线性样本复杂度并消除之前指数时间瓶颈；同时扩展到非零均值噪声情形。

**🔧 技术方法**

使用鲁棒PCA（过滤+截断）估计协方差的主特征向量，截断后进行鲁棒谱初始化；随后采用鲁棒梯度下降和对称化技巧处理非零均值噪声。

**📊 数据集**

实验基于高斯测量向量生成的合成数据集（无真实数据集）。

**📈 对比分析**

相较于先前的指数时间算法和仅在无噪声时的近线性方法，所给算法在样本复杂度为 Õ(n)，时间为 O(m^2 n) 级别下实现了同等或更优的误差收敛；但在近线性时间上尚未达到最优。

**⚠️ 局限性**

限制在于对抗性误差的容忍度与噪声-信号比相关，无法完全独立于噪声；此外，近线性时间实现仍需更高样本量或进一步证明稳定性条件。

---

## 664. GenCI: Generative Modeling of User Interest Shift via Cohort-based Intent Learning for CTR Prediction

**arXiv ID:** 2601.18251 | [PDF](https://arxiv.org/pdf/2601.18251v1)

**作者:** Kesha Ou `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 24297 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于生成式用户意图的CTR预测框架GenCI，利用NTP任务主动生成语义兴趣队列，并通过层次化候选感知网络将其与历史行为和目标物品对齐。

**💡 创新点**

核心创新在于将生成式预训练与召回-排序的一致性结合，利用语义兴趣队列从宏观层面捕捉即时兴趣，并通过层次注意力精炼成针对目标的短期意图，实现了对用户兴趣动态的全景建模。

**🔧 技术方法**

主要技术包括Transformer式的序列生成模型、RQ‑VAE层次量化编码、交叉注意力和层次化候选感知机制，以及自监督正则化与端到端联合优化的训练框架。

**📊 数据集**

在三个公开数据集上进行评估：MovieLens（电影推荐）、Amazon‑Fashion（服饰）和Amazon‑Musical‑Instruments（乐器）数据集。

**📈 对比分析**

与LR、FwFM、NFM、PNN、FiGNN、DeepFM、DCNv2、AutoInt+、xDeepFM、AFN+、RFM、DIN、DIEN、MIRRN、SFG等多种基线对比，GenCI在AUC上分别提升约10.8%、10.3%和9.9%，在LogLoss上也取得显著降幅，整体性能显著优于现有方法。

**⚠️ 局限性**

限制在于对语义ID的依赖使模型对文本表征质量敏感；生成的兴趣队列可能包含语义相似但行为不相关的物品；以及缺乏多模态或跨域信息，可能限制对更丰富兴趣信号的捕获。

---

## 665. Vision-Language-Model-Guided Differentiable Ray Tracing for Fast and Accurate Multi-Material RF Parameter Estimation

**arXiv ID:** 2601.18242 | [PDF](https://arxiv.org/pdf/2601.18242v1)

**作者:** Zerui Kang `[一作]` (Singapore University of Technology and Design), Jihong Park `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 5962 | [OpenAlex ID](https://openalex.org/A5027907258)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用 VLM 先验实现可微射线追踪（DRT）的多材料 RF 参数（导电率）估计，加速并稳定梯度下降收敛。

**💡 创新点**

创新点在于：① 用 VLM 解析场景图像得到材料类别并映射到 ITU‑R 物料表，提供更精准的导电率初始化；② VLM 决策有利的 Tx/Rx 布局，显著提高测量信息量和路径多样性；③ 将两项先验无缝嵌入 DRT 梯度优化，显著提升收敛速度与精度。

**🔧 技术方法**

核心技术包括：可微射线追踪引擎 NVIDIA Sionna；大语言视觉模型 Gemini 2.5 Pro；梯度下降（Adam）优化；ITU‑R P.2040 物料表作为先验数据库。

**📊 数据集**

实验数据：10 m × 10 m × 3 m 室内仿真场景，9 个已知几何物体（墙、地板、四盒不同材料）；生成的接收信号由 Sionna 在 3.5 GHz、4 次射线深度、5000 条射线等参数下模拟；导电率按 ITU‑R 标准再加 0.8–1.2 乘子扰动。

**📈 对比分析**

与随机初始化/均匀初始化（RandInit/UnifInit）以及随机测量位置（RandSel）等基线对比；VLMInit+VLMSel 在相同实验设置下实现 2–4 倍更快收敛，10–100 倍更低最终误差，最终均相对误差低于 0.1%。

**⚠️ 局限性**

局限性：仅估计导电率，未联合估计介电常数；实验仅在仿真环境，真实照片/硬件验证尚缺失；VLM 的图像语义识别对光照、遮挡等鲁棒性未知；对极大规模场景的可扩展性未评估。

---

## 666. V-Loop: Visual Logical Loop Verification for Hallucination Detection in Medical Visual Question Answering

**arXiv ID:** 2601.18240 | [PDF](https://arxiv.org/pdf/2601.18240v1)

**作者:** Mengyuan Jin `[一作]` (Northwestern Polytechnical University), Yong Xia `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 18643 | [OpenAlex ID](https://openalex.org/A5100670074)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了 V-Loop，一种无训练、可插拔的视觉逻辑循环验证框架，用于检测医学视觉问答（VQA）中的幻觉输出。

**💡 创新点**

创新点在于构建双向推理循环，通过从答案和问题中抽取语义单元生成验证问题，并强制视觉注意一致性，以直接核实答案的视觉真实性。

**🔧 技术方法**

主要技术包括多模态大型语言模型（MLLM）推理、语义单元抽取、逻辑/重述式验证问题生成、视觉注意一致性约束（VAC）以及使用辅助 LLM（如 DeepSeek‑V3.2‑Exp）进行语义一致性评估。

**📊 数据集**

使用了三大医学 VQA 基准：VQA‑RAD、VQA‑Med‑2019 和 SLAKE，并在 MedGemma‑4B‑it、Lingshu‑7B、InternVL3‑8B 等多种 MLLM 上进行评测。

**📈 对比分析**

与七种基于不确定性估计的基线（AvgProb、MaxProb、AvgEnt、MaxEnt、SE、RadFlag、VASE）对比，V‑Loop 在大多数数据集的 AUC 与 AUG 指标上均实现显著提升，且可进一步提升不确定性方法的性能。

**⚠️ 局限性**

主要局限在于对底层 MLLM 的视觉语义推理能力高度依赖；在模型能力较弱或缺乏双语义单元时，验证逻辑效果下降；且辅助 LLM 的质量也影响验证问题的准确性。

---

## 667. A multimodal vision foundation model for generalizable knee pathology

**arXiv ID:** 2601.18250 | [PDF](https://arxiv.org/pdf/2601.18250v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 668. Probing the Future of Meta-Analysis: Eliciting Design Principles via an Agentic Research IDE

**arXiv ID:** 2601.18239 | [PDF](https://arxiv.org/pdf/2601.18239v1)

**作者:** Sizhe Cheng `[一作]` (Nanyang Technological University), Yong Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 130619 | [OpenAlex ID](https://openalex.org/A5059976286)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了Research IDE原型，将“研究即代码”理念与假设断点机制结合，实现在写作过程中即时验证文献证据。

**💡 创新点**

创新点在于把假设检验类比为代码调试，提供断点式交互与多智能体后端，既保持研究者的主导权又利用AI进行跨文献推理。

**🔧 技术方法**

采用React前端与多智能体框架（Planner、Librarian、Reasoner、Producer）以及OpenAI GPT‑5‑mini实现检索、推理与可视化。

**📊 数据集**

使用真实研究文献库进行检索，未使用公开标准数据集；在实地部署中共检索并标注了548篇论文。

**📈 对比分析**

通过对8名专家为期一周的现场部署和反思研讨会评估，未进行与现有工具的定量性能比较，而是通过用户反馈和使用日志量化验证效果。

**⚠️ 局限性**

局限性包括样本规模小且仅涵盖STEM研究者，缺乏跨学科验证；工具功能对不同研究工作流的通用性仍待进一步迭代。

---

## 669. TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment

**arXiv ID:** 2601.18292 | [PDF](https://arxiv.org/pdf/2601.18292v1)

**作者:** Zhewen Tan `[一作]` (Peking University), Lin Sun `[通讯]` (Qiyuan Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TriPlay‑RL 三角色闭环强化学习框架，能够在几乎无人工标注的情况下实现攻击者、守护者和评估者的协同进化，从而提升 LLM 的安全对齐能力。

**💡 创新点**

创新点：① 引入评估者角色形成三方闭环；② 通过多模型攻击、多样性惩罚与三阶奖励机制避免熵坍塌和安全/通用性能的权衡；③ 使用语义奖励保持攻击意图不变，同时保持输出多样性。

**🔧 技术方法**

技术：强化学习（GRPO + RLVR）进行角色训练；Self‑BLEU 与句向量余弦相似度构成多样性惩罚；多模型红蓝游戏；专家多投票评估机制；向量化语义匹配与三阶奖励。

**📊 数据集**

数据集：HarmBench（基本攻击提示）；AIR‑Bench 2024、JailBreakBench、WildJailBreak、S‑Eval 用于安全评估；IFEval、GPQA、LiveCodeBench、AIME 2025 用于通用推理评估；自制的三方对话与评估数据（约3k条）。

**📈 对比分析**

对比结果：红方攻击成功率提升 20–50%；蓝方在 AIR‑Bench、JailBreakBench 等安全基准上的 ASR 下降至 1–5%（显著提高安全性），且在 IFEval、GPQA 等通用推理基准上保持甚至略高的表现；评估者在三类分类任务上的准确率从 56% 提升至 98%。

**⚠️ 局限性**

局限：① 所有角色均基于同一模型初始化，未探讨异构角色效果；② 未将外部安全/攻击数据融入训练；③ 对三方博弈的纳什均衡与能力平衡缺乏深入理论分析；④ 自动评估可能出现误差，易被奖励劫持；⑤ 对双重用途攻击风险的防护仍需完善。

---

## 670. Reflecting Twice before Speaking with Empathy: Self-Reflective Alternating Inference for Empathy-Aware End-to-End Spoken Dialogue

**arXiv ID:** 2601.18281 | [PDF](https://arxiv.org/pdf/2601.18281v1)

**作者:** Yuhang Jia `[一作]` (Nankai University), Yong Qin `[通讯]` (Nankai University)

**通讯引用:** 9438 | [OpenAlex ID](https://openalex.org/A5088716214)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了自评估模型EmpathyEval和端到端语音模型ReEmpathy，实现通过自我反思交替推理提升语音对话的共情质量。

**💡 创新点**

创新点在于提出描述性自然语言评估和自我反思交替推理机制，打破传统基于分数或单一监督的共情优化。

**🔧 技术方法**

使用Qwen3-Omni（及GLM-4-Voice）作为基础模型，结合GPT‑4生成数据、CosyVoice2 TTS、EmotionTalk情感标注、以及自我反思训练。

**📊 数据集**

主要数据集为18,000条中文语音对话（含情感标注）和EmotionTalk、OpenS2S等。

**📈 对比分析**

与多种对话模型（SFT、DPO、CoT等）对比，ReEmpathy在EmpathyEval四维评分和人类MOS上均优于基线，提升约1–2个百分点。

**⚠️ 局限性**

局限性包括缺乏粒度级别的on‑policy监督、反思推理仅靠SFT、以及模型对低频交替或极短chunk的鲁棒性不足。

---

## 671. Cognitive Fusion of ZC Sequences and Time-Frequency Images for Out-of-Distribution Detection of Drone Signals

**arXiv ID:** 2601.18326 | [PDF](https://arxiv.org/pdf/2601.18326v1)

**作者:** Jie Li `[一作]` (Xidian University), Fengkui Gong `[通讯]` (Xidian University)

**通讯引用:** 2675 | [OpenAlex ID](https://openalex.org/A5034754948)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于 Zadoff‑Chu 序列与时频图（TFI）认知融合的无人机信号异常检测与识别算法。

**💡 创新点**

创新点在于：① 将通信协议中可识别的 ZC 序列特征与 TFI 特征进行多模态交互、单模态融合与多模态融合；② 引入空间与通道维度的判别分数与自适应注意力权重，以突出协议差异带来的时间‑频率特征；③ 通过专门的特征提取、交互和融合模块，显著提升了开集识别（OODD）的鲁棒性。

**🔧 技术方法**

主要技术包括：ZC 序列交叉相关特征提取、短时傅里叶变换生成 TFI、MobileNetV4 与自定义卷积网络、通道/空间注意力机制、判别分数计算与自适应加权、Softmax 分类。

**📊 数据集**

使用了三个公开无人机 RF 数据集：DroneRFa、DroneRFb‑DIR 与 RFUAV，覆盖多种通信协议、飞行距离、LOS/非 LOS 状况以及不同采样时长。

**📈 对比分析**

与传统 IQ‑CNN、ZC‑CNN、TFI‑MobileNet、简单拼接、重构误差、置信分数等方法对比，实验表明该算法在 RID 方面提升了至少 1.7% 的准确率，在 OODD 方面提升了至少 7.5% 的准确率，整体参数量与 FLOPs 也在可接受范围内。

**⚠️ 局限性**

局限性：① 模型参数量与计算开销较大，实时部署仍有挑战；② 对 ZC 序列的识别仍需依赖已知协议，未知协议的鲁棒性尚待进一步验证；③ 在极低信噪比或极端 NLOS 环境下仍可能出现误检。

---

## 672. Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLM

**arXiv ID:** 2601.18306 | [PDF](https://arxiv.org/pdf/2601.18306v1)

**作者:** Everlyn Asiko Chimoto `[一作]` (Lelapa AI), Bruce A. Bassett `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多语言校准集对大语言模型量化性能的影响

**💡 创新点**

首次系统比较不同语言组合校准集对GPTQ和AWQ量化的效果，发现多语言校准可显著降低perplexity并提升多语言性能

**🔧 技术方法**

采用GPTQ和AWQ两种权重仅量化技术

**📊 数据集**

使用10种语言（英语、法语、斯瓦希里语、中文、 isiXhosa 及其多语言混合集）构成的校准集，包含原始英语校准、翻译集、C4/Wikipedia 采样及代码/数学扩展等

**📈 对比分析**

通过对比不同校准集的perplexity及XLI, XStoryCloze、Global MMLU等下游任务准确率，发现多语言校准能降低多达3.52点perplexity，并在多语言任务上获得更好性能

**⚠️ 局限性**

仅评估perplexity，未覆盖更多模型、量化方法和实际下游任务；样本规模和语言覆盖受限，缺乏动态校准或更广泛的实用评估

---

## 673. SwipeGen: Bridging the Execution Gap in GUI Agents via Human-like Swipe Synthesis

**arXiv ID:** 2601.18305 | [PDF](https://arxiv.org/pdf/2601.18305v1)

**作者:** Xuan Wang `[一作]` (Fudan University), Yangfan Zhou `[通讯]` (Shanghai Key Laboratory of Intelligent Information Processing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了自动合成人类滑动交互数据的管道，构建了滑动执行基准，并训练了改进的 GUI VLM Agent。

**💡 创新点**

通过将滑动分解为起点、终点、方向、时长四维参数，自动探索 GUI 并验证效果，解决了现有数据缺失与描述不完整的问题。

**🔧 技术方法**

结合视觉解析 (OmniParser)、VLM 推理、随机 GUI 探索、执行‑验证流程、RLVR 奖励设计，微调 Qwen2.5‑VL‑3B‑Instruct。

**📊 数据集**

利用自动生成的 152 条可执行滑动样本（来自 16 款新发布的手机应用），并在 185 条交互样本（124 滑动+61 点击）上进行训练。

**📈 对比分析**

在 OOD 滑动基准上与基线 Qwen2.5‑VL‑Instruct 对比，滑动成功率从约 32% 提升至 69.07%，提升约 214%。

**⚠️ 局限性**

依赖随机探索导致覆盖不全；自动生成的自然语言描述可能存在噪声和模型偏差。

---

## 674. Suppressing Final Layer Hidden State Jumps in Transformer Pretraining

**arXiv ID:** 2601.18302 | [PDF](https://arxiv.org/pdf/2601.18302v1)

**作者:** Keigo Shibata `[一作]` (Tohoku University), Jun Suzuki `[通讯]` (Tohoku University)

**通讯引用:** 7900 | [OpenAlex ID](https://openalex.org/A5001456824)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过引入跳跃抑制正则化（JREG）来降低Transformer语言模型在最终层隐藏状态的显著“跳跃”，从而提高中间层的有效利用和整体性能；

**💡 创新点**

创新点在于提出了量化最终层跳跃强度的指标并设计了针对跳跃的正则化方法，使模型在不改变架构的前提下显著降低跳跃并提升下游任务表现；

**🔧 技术方法**

使用的技术包括基于余弦相似度的隐藏状态位移度量、跳跃率计算、JREG正则化项、以及标准交叉熵训练；

**📊 数据集**

数据集包括公开的FineWeb‑Edu 100B 语料用于预训练，LAMBADA、BoolQ、ARC‑e、HellaSwag、PIQA、RACE、SocialIQA、SciQ、SWAG 等用于下游评估，SFT 采用 Tulu‑v1 指令数据集；

**📈 对比分析**

通过与仅使用交叉熵的基线模型对比，JREG 在 170M、1B、3.4B 三种规模的 Llama‑based 模型中在平均下游任务得分上提升了约1–5%，且跳跃率从数十降至0，表明显著改善了模型表现；

**⚠️ 局限性**

局限性包括方法对模型深度敏感，需要针对不同层数调优超参数；目前仅在 Llama 系列架构上验证，未证实其在其他 Transformer 变体或不同归一化/注意机制下的效果。

---

## 675. Collaposer: Transforming Photo Collections into Visual Assets for Storytelling with Collages

**arXiv ID:** 2601.18428 | [PDF](https://arxiv.org/pdf/2601.18428v1)

**作者:** Jiayi Zhou `[一作]` (Hong Kong University of Science and Technology), Anyi Rao `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 4539 | [OpenAlex ID](https://openalex.org/A5067715162)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了Collaposer工具，自动将照片集合转换为按故事描述筛选、实例分割、语义分层、可直接使用的视觉剪裁资产，以支持数字拼贴叙事的资产准备与组织。

**💡 创新点**

①利用LLM推理自动生成故事核心与相关标签，弥补关键词搜索不足；②将实例分割与分层语义组织一体化，减少手工裁切与命名；③根据资产与故事匹配度、视觉多样性、分辨率动态调整显示尺寸，提升可视化体验。

**🔧 技术方法**

使用RAM（零样本标签）、Grounding DINO（开放集检测）、SAM（实例分割）、GPT‑4o（标签选择、聚类、角色部件解析）、Sapiens（关键点估计）、CLIP（相似度与一致性评分）、前端树+画布视图与JSON导出。

**📊 数据集**

采用公开域照片集合（NASA Archives、Olympics Archive、Library of Congress等）共116张，预处理得到2427个视觉元素；用户评估使用同一集合。

**📈 对比分析**

在12位参与者的within‑subject实验中，对比两个基线（仅关键词匹配+默认展示，及仅展示无聚类），通过prompt次数、满意度问卷（Consistency、Diversity、Presentation、Usability）和访谈评估。Collaposer在一致性、内容多样性、呈现效果和易用性上均显著优于基线，平均prompt次数更少，所有用户均实现一次性成功。

**⚠️ 局限性**

局限：①仅依赖文本意图表达，易出现与创作者意图不一致的资产；②LLM推理缺乏透明性，用户对推荐理由不易理解；③当前流程为单次准备，缺乏动态迭代与局部精修；④实例分割精度尚未达到商业级别；⑤缺乏长期使用和跨项目的评估。

---

## 676. Superlinear Multi-Step Attention

**arXiv ID:** 2601.18401 | [PDF](https://arxiv.org/pdf/2601.18401v1)

**作者:** Yufeng Huang `[一作]` `[通讯]` (Concavity AI), Yufeng Huang (Concavity AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种多步超线性注意力架构，能够在保持随机上下文访问的同时将长序列的注意力复杂度降至 O(L^{3/2}) 级别。

**💡 创新点**

核心创新是将注意力视为多步搜索问题，使用跨度搜索+跨度注意力以及可学习的软路由，实现子二次复杂度并保留结构非排除性。

**🔧 技术方法**

技术实现包括基于线性递归（Mamba‑2/SSM）的累积、可学习的搜索查询向量、Top‑k 路由、区间注意力以及基于键块的 GPU bucketed 内核。

**📊 数据集**

在实验中使用 RULER 任务中的 Needle‑in‑a‑Haystack (NIAH) 数据集进行微调，并在原始 Nemotron‑3‑Nano‑30B 模型上评估。

**📈 对比分析**

与 FlashAttention‑2 的稠密注意力对比，Superlinear 在 60K–10M 令牌规模下显著提升前置和解码吞吐量；在 10M 语境下解码速度约 76 tokens/s，证明在极长序列上的实用性。

**⚠️ 局限性**

局限性包括仅验证 N=2 的基线实现，缺乏全面的质量评测；训练时需依赖课程学习和手工调节指数，且对更大 N 或更复杂任务的可扩展性尚待验证。

---

## 677. Do not be greedy, Think Twice: Sampling and Selection for Document-level Information Extraction

**arXiv ID:** 2601.18395 | [PDF](https://arxiv.org/pdf/2601.18395v1)

**作者:** Mikel Zubillaga `[一作]` (University of the Basque Country), Eneko Agirre `[通讯]` (University of the Basque Country)

**通讯引用:** 14535 | [OpenAlex ID](https://openalex.org/A5047151336)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ThinkTwice 框架，通过采样多份候选输出并进行选择，提升文档级信息抽取的结果。

**💡 创新点**

创新点在于利用 LLM 输出的多样性，结合无监督的 F1‑Voting 与有监督的奖励模型两种选择策略，并引入基于拒绝采样的银数据生成方法，克服缺失金标推理轨迹的问题。

**🔧 技术方法**

主要技术包括大规模 LLM（如 Llama 3.3/DeepSeek Distill R1、Qwen 3）、约束解码、采样与选择（F1‑Voting、奖励模型）、拒绝采样生成银数据以及跨语言迁移训练。

**📊 数据集**

实验使用 MUC‑4、MultiMUC（多语种扩展）和 BETTER Granular 三个文档级事件抽取数据集。

**📈 对比分析**

与传统贪婪解码、ChatGPT 3‑shot 以及现有监督基线相比，ThinkTwice 在零样本、监督和跨语言场景均实现了显著提升，单词 F1 约提高 2–4 点，逼近上界，取得了新 state‑of‑the‑art。

**⚠️ 局限性**

主要限制包括：仍缺乏真正的金标推理轨迹，奖励模型与无监督选择之间存在性能差距；采样与选择的计算成本较高；银数据生成成功率不完全覆盖训练样本；跨语言迁移在某些语言（如韩语）仍表现不佳。

---

## 678. Efficient Complex-Valued Vision Transformers for MRI Classification Directly from k-Space

**arXiv ID:** 2601.18392 | [PDF](https://arxiv.org/pdf/2601.18392v1)

**作者:** Moritz Rempe `[一作]` (University Hospital Essen), Jens Kleesiek `[通讯]` (University Hospital Essen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于复杂值 Vision Transformer（kViT）的模型，直接在磁共振成像的原始k‑space数据上完成病灶分类任务。

**💡 创新点**

创新点包括：①径向k‑space补丁策略，保持频域能量分布；②复杂值位置嵌入（RoPE），在复数域中捕获全局依赖；③大幅降低VRAM消耗（高达68倍），使模型更适合资源受限环境。

**🔧 技术方法**

使用的技术包括：复杂值多头自注意力与前馈网络、RoPE位置嵌入、k‑space切片化及径向补丁、复数域层归一化、复数域激活、数据增强（切除、旋转等）以及多实例学习框架。

**📊 数据集**

实验数据集涵盖 fastMRI 前列腺、膝关节（含 fastMRI+ 病理标签）以及自制脑肿瘤（患者级标签）三大数据集。

**📈 对比分析**

与 ResNet、EfficientNet、ViT‑Tiny 等基线进行比较，kViT 在全采样和高下采样（4x–24x）条件下保持与基线相当甚至更优的 AUROC/AUPRC，且在训练时的 VRAM 使用量从 10.6 GB 降至 0.96 GB（≈68×减小）。

**⚠️ 局限性**

局限性包括：①对几何性病变（如膝关节）鲁棒性不如前列腺数据；②仅采用环形补丁，导致方向信息损失；③未进行预训练或混合域（k‑space 与图像域）学习，可能限制进一步性能提升。

---

## 679. AI Agent for Reverse-Engineering Legacy Finite-Difference Code and Translating to Devito

**arXiv ID:** 2601.18381 | [PDF](https://arxiv.org/pdf/2601.18381v1)

**作者:** Yinghan Hou `[一作]` (Imperial College London), Zongyou Yang `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于AI代理的系统，实现对传统Fortran有限差分代码的逆向工程，并将其自动翻译为Devito框架，整个过程通过多阶段检索、知识图谱、结构化提示与严格的输出约束完成。

**💡 创新点**

创新点包括：① 将GraphRAG与动态迭代优化结合，形成基于LangGraph的强化学习式决策路由；② 采用多模态检索和社区感知检索提升信息检索质量；③ 通过Pydantic约束保证生成代码的结构化与可验证性；④ 引入G‑Eval与传统静态分析相结合的评估框架；⑤ 并发异步处理显著提升翻译吞吐量。

**🔧 技术方法**

主要技术：检索增强生成（RAG）+ GraphRAG；Neo4j知识图谱、BGE‑M3嵌入、Leiden社区检测；LangGraph工作流、Pydantic结构化输出、Ruff代码格式化；G‑Eval评估模型；异步asyncio并发；多阶段检索、结果融合与重排序；强化学习式质量驱动迭代。

**📊 数据集**

使用Devito源代码仓库（约6k知识块、70个社区、32k节点）和Fortran翻译测试集（13个案例、11个检索基准），并在此基础上构建检索语料与翻译对。

**📈 对比分析**

对比方法：检索性能评估（Precision@5 = 0.964，Recall@5 = 0.930，MRR = 1.000，平均响应时间 0.012 s）；代码翻译质量评估（Grade‑A = 76.9%，在执行、结构、API三维度均达1.00）；并发性能：四代理并行实现吞吐率达131.2 文件/小时，5.75×提升于单进程。

**⚠️ 局限性**

局限：质量评估阈值固定、缺乏自动调优；对不同代码类型/复杂度的适应性有限；依赖单一开源LLM，未引入专门的评估模型；知识图谱与模式挖掘深度不足，难以实现自适应学习与记忆重用。

---

## 680. Integrating HAPS, LEO, and Terrestrial Networks: A Cost-Performance Study for IoT Connectivity

**arXiv ID:** 2601.18361 | [PDF](https://arxiv.org/pdf/2601.18361v1)

**作者:** Jean Michel de Souza Sant'Ana `[一作]` (Centre for Wireless Communications University of Oulu), Aamir Mahmood `[通讯]` (Mid Sweden University)

**通讯引用:** 3174 | [OpenAlex ID](https://openalex.org/A5015129195)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文通过仿真框架对高空平台（HAPS）、低地球轨道（LEO）卫星与传统陆基网络在遥远地区 IoT 连接中的传输抹除概率、包成功率及成本进行比较，探讨它们单独及混合使用的性能与经济性。

**💡 创新点**

创新点在于首次系统评估 HAPS 与 LEO 与陆基网络的混合部署，提出统一的抹除概率与成功率度量，并对三种方案的 CAPEX/OPEX 进行成本对比，揭示在稀疏陆基环境下非地面技术的优势与潜在协同。

**🔧 技术方法**

使用了 LoRaWAN LR‑FHSS 的物理层模型、自由空间传播与阴影瑞利（shadowed Rice）衰落模型、Gamma 分布近似、基于离散随机轨道的卫星可覆盖时间模型，以及统一的成本折现公式。

**📊 数据集**

主要使用仿真生成的数据集：在半径 80 km 圆形区域内随机布置 5000 台设备，分别模拟 10 或 30 个陆基基站、单一 HAPS、单一 LEO 以及其混合配置。

**📈 对比分析**

通过热图、violin 图以及包成功率随设备数变化曲线对三种部署方案进行定量比较，结果显示 HAPS 在抹除概率与成功率上优于 LEO，LEO 在成本上更具竞争力；混合 HAPS/LEO 或陆基网络可进一步提升覆盖公平性与可扩展性。

**⚠️ 局限性**

局限性包括：仅基于仿真，未进行现场实验验证；假设单一 HAPS 与单一 LEO；未考虑多卫星可视性、HAPS 机动与多天线技术；成本模型简化，未涵盖频谱许可与维护费用等。

---

## 681. Code over Words: Overcoming Semantic Inertia via Code-Grounded Reasoning

**arXiv ID:** 2601.18352 | [PDF](https://arxiv.org/pdf/2601.18352v1)

**作者:** Manjie Xu `[一作]` (Peking University), Yixin Zhu `[通讯]` (Peking University)

**通讯引用:** 3881 | [OpenAlex ID](https://openalex.org/A5051255725)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在动态可变规则环境中出现的“语义惯性”问题，并提出了通过对比式反事实对齐实现可即时生成可执行世界模型的lcv框架；

**💡 创新点**

创新点在于：①首次发现模型规模与语义惯性呈逆向缩放；②设计对比式反事实对齐训练，使模型在推理时能抑制预训练先验；③实现可 amortized 的代码生成+规划组合，显著降低推理延迟；

**🔧 技术方法**

使用技术包括：对比式反事实对齐 (Contrastive Contrastive Alignment)、Python 代码生成的可执行转移函数、经典规划器（Greedy Best‑First Search）、与现有基线（直接策略、CoT、代码策略、TheoryCoder 等）对比；

**📊 数据集**

数据集：自制可变规则游戏《Baba is You》中的 45/50 层级实例（分为语义对齐、语义冲突、动态可塑性三层），以及约 600 对规则冲突样本用于 fine‑tune；

**📈 对比分析**

在三个难度层级上与多种基线对比，lcv 在 Tier1、Tier2、Tier3 的成功率分别达到约 88.9%、75.6%（或 93.3%/75.6%）和 62%，显著优于直接策略、CoT 及 TheoryCoder 等方法；同时推理 token 约 800，推理速度比 TheoryCoder 快约 4 倍；

**⚠️ 局限性**

限制：仅在离散格子世界上验证；需额外感知模块才能迁移至连续高维视觉任务；对比式反事实样本需要可生成的规则冲突，天然环境中难以收集；

---

## 682. On the Subspace Orbit Problem and the Simultaneous Skolem Problem

**arXiv ID:** 2601.18349 | [PDF](https://arxiv.org/pdf/2601.18349v1)

**作者:** Piotr Bacik `[一作]` (University of Oxford), Anton Varonka `[通讯]` (TU Wien)

**通讯引用:** 23 | [OpenAlex ID](https://openalex.org/A5062243686)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文研究了线性动力学系统（LDS）的子空间轨道问题（Subspace Orbit Problem），给出了在目标子空间维度相对轨道维度为对数级别时可判定的结果，并证明当目标维度与轨道维度成线性关系时问题与Skolem问题同难；同时给出了相应的复杂度上界；

**💡 创新点**

创新点在于：①提出并证明了“归约LDS”与“固有维度”的概念，明确了判定难度与目标维度的关系；②将子空间轨道问题与同时Skolem问题等价化，并利用MSTV类（最多3个主导根）构造可判定的线性组合；③给出了一般的高度与极大值上界，进而得到对解空间的指数上界；④证明了若存在线性比例可判定算法则Skolem问题可判定，提升了该问题的复杂度下限。

**🔧 技术方法**

主要技术包括：线性递推序列的指数多项式展开、代数数论（绝对值、位数高度、分拆定理）、Krylov子空间与矩阵特征值分析、MSTV类判定、Baker型零点估计、Siegel–Vaaler等号理论、矩阵行列式高度估计以及基于极大数值上界的猜测与检验算法。

**📊 数据集**

论文未使用任何实验数据集，全部为理论分析与算法构造。

**📈 对比分析**

相对于以往只在目标维度≤3可判定的结果，本文扩展到目标维度≤2log₃d，可判定，并给出在d固定时问题属于P空间的复杂度；对于目标维度线性增长的情形，给出了与Skolem问题等价的硬性证明。

**⚠️ 局限性**

限制：1）仍未解决目标维度≥4且不满足对数上界的情形；2）对高维目标子空间的判定仍保持未决；3）给出的复杂度上界在实践中可能过于粗糙；4）MSTV类判定依赖于高度与根数的估计，实际实现难度较高。

---

## 683. Analytic Incremental Learning For Sound Source Localization With Imbalance Rectification

**arXiv ID:** 2601.18335 | [PDF](https://arxiv.org/pdf/2601.18335v1)

**作者:** Zexia Fan `[一作]` (University of Science and Technology Beijing), Xinyuan Qian `[通讯]` (Eigenspace GmbH)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出统一框架SSL-GCIL解决增量SSL中长期尾类和任务间不平衡导致的灾难性遗忘。

**💡 创新点**

创新点包括基于GCC-PHAT峰值的无额外数据增广方法GDA和自适应动态不平衡校正器ADIR，可实时调整权重并不需示例存储。

**🔧 技术方法**

技术手段包含GCC-PHAT特征提取、MLP特征提取器、GDA数据增强、ADIR闭式自适应重加权及递归最小二乘更新。

**📊 数据集**

使用SLLR基准数据集，10个任务的逐步长尾分布，采样为4通道48kHz音频。

**📈 对比分析**

与四种SOTA CIL基线（LwF、iCaRL、ACIL、GACL）对比，在清晰条件下获得89.0%精度、5.3° MAE、1.6正向BWT，优于ACIL 3.1%精度提升，接近联合训练上限。

**⚠️ 局限性**

限制在于对极低SNR条件仍显退化，且仅针对DoA角度预测，未考虑多源混叠与实时性验证。

---

## 684. Overalignment in Frontier LLMs: An Empirical Study of Sycophantic Behaviour in Healthcare

**arXiv ID:** 2601.18334 | [PDF](https://arxiv.org/pdf/2601.18334v1)

**作者:** Clément Christophe `[一作]` (M42 Abu Dhabi), Praveenkumar Kanithi `[通讯]` (M42 Abu Dhabi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在医疗多选题基准上评估LLM的同情性，并提出基于可验证事实的评估框架。

**💡 创新点**

创新点在于引入“调整后的同情性分数”（S_a），通过剔除模型随机不稳定性来准确衡量对用户偏好而非事实的顺从。

**🔧 技术方法**

采用对抗性提示（基本诱导与专家诱导）、思考与非思考推理模式比较，并进行参数规模扩展实验。

**📊 数据集**

使用MedQA与MMLU‑Pro医学多选题数据集。

**📈 对比分析**

与原始同情性分数（S_r）和多模型（如Qwen‑3、Llama‑3、GPT‑5.2等）对照，发现S_a在参数≥14B时趋近零；思考模式在专家诱导下表现出更高S_a，显示在权威压力下易失真。

**⚠️ 局限性**

局限在于仅评估单轮多选问答，未覆盖多轮对话和更复杂的权威策略；且S_a假设随机错误均匀分布，可能忽略特定误导选项的影响。

---

## 685. daVinci-Dev: Agent-native Mid-training for Software Engineering

**arXiv ID:** 2601.18418 | [PDF](https://arxiv.org/pdf/2601.18418v1)

**作者:** Ji Zeng `[一作]` (SJTU), Pengfei Liu `[通讯]` (SII)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型的中训练阶段引入“agent-native”数据，构建能够模拟完整软件工程交互流程的训练样本，实现了从Pull Request到可执行环境的全流程数据合成与训练。

**💡 创新点**

创新点包括：①提出并实现“上下文完整轨迹”与“环境真实轨迹”两类agent-native数据，解决了传统训练数据与真实开发流程的分布不匹配问题；②在中训练阶段直接使用这些轨迹显著提升模型的agentic能力，而不依赖昂贵的强化学习；③通过大规模（68.6B+3.1B）数据实现了高效的token利用，取得了同类开源模型最高的Pass@1成绩。

**🔧 技术方法**

主要技术：大规模中训练（mid‑training）策略、上下文完整轨迹与环境真实轨迹的数据合成、基于SWE‑Agent scaffold的后续SFT与RL；使用了Qwen‑2.5系列（32B/72B）模型。

**📊 数据集**

数据集：①68.6B-token的contextually‑native轨迹（来自公开PR，包含issue、相关文件、逐步提交与合并历史）；②3.1B-token的environmentally‑native轨迹（从PR派生的Docker可执行环境中收集的agent与真实测试反馈交互日志）。

**📈 对比分析**

与Kimi‑Dev等开源中训练方案比较：在相同基模型与agentic scaffold下，daVinci‑Dev‑72B实现58.5% Pass@1，daVinci‑Dev‑32B实现56.1% Pass@1，均优于Kimi‑Dev（48.6%）和其他公开方法；在无强化学习的SFT设置下，亦保持显著优势；同时在CodeBench与Scientific Benchmarks上亦表现出稳健提升。

**⚠️ 局限性**

局限性：①数据中包含部分开发者信息，存在隐私与著作权风险；②评测部分依赖对benchmark进行的手动修复，可能引入评测误差；③实验仅基于Qwen‑2.5系列与SWE‑V benchmark，缺乏对不同模型体系和更大规模真实任务的验证。

---

## 686. Estimation of geometric transformation matrices using grid-shaped pilot signals

**arXiv ID:** 2601.18385 | [PDF](https://arxiv.org/pdf/2601.18385v1)

**作者:** Rinka Kawano `[一作]` (Yamaguchi University), Masaki Kawamura `[通讯]` (Yamaguchi University)

**通讯引用:** 356 | [OpenAlex ID](https://openalex.org/A5070241028)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于网格形Pilot信号与Radon变换的几何变换估计与同步水印方法，能够在裁剪及尺度、旋转、剪切攻击后仍准确定位嵌入区域。

**💡 创新点**

创新点在于：①利用不同取值的水平/垂直网格线区分方向；②将Pilot信号嵌入U通道，避免对水印的干扰；③通过Radon变换提取变换角度与间距，并推导解析式直接得到变换矩阵，避免暴力搜索；④在估计后进行逆变换实现水印解码。

**🔧 技术方法**

技术手段包括：QIM嵌入/提取、Radon变换、方差与零交叉检测、正则化与阈值化、Autocorrelation+DFT获取网格间距、解析求解变换矩阵。

**📊 数据集**

实验使用六幅IHC标准高分辨率图像(4608×3456)和十幅Kodak低分辨率图像(768×512或512×768)。

**📈 对比分析**

与传统SIFT+DFT水印方法对比，单攻击下高分辨率图像估计误差几乎为0，逆变换后水印BER <0.1；低分辨率图像误差增大，估计成功率约50%。总体性能优于传统方法，特别是对裁剪+几何混合攻击的鲁棒性显著。

**⚠️ 局限性**

局限性包括：对低分辨率或小裁剪区域估计效果差；无法直接区分180°对称变换矩阵，需额外校验位；Pilot嵌入导致图像质量下降约1 dB；在强烈变形导致嵌入区域被完全消失时仍无法恢复水印。

---

## 687. Hierarchical Text Classification with LLM-Refined Taxonomies

**arXiv ID:** 2601.18375 | [PDF](https://arxiv.org/pdf/2601.18375v1)

**作者:** Jonas Golde `[一作]` (Humboldt Universität zu Berlin), Phong Le `[通讯]` (School of Computer Science, University of St Andrews)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型（LLM）对人类手工构建的层级分类标签体系进行结构化改造（重命名、合并、拆分、重排）以提升层级文本分类（HTC）性能。

**💡 创新点**

提出全树级别的LLM驱动重构框架（两阶段：全局生成 + 纠错后处理），首次将LLM用作“层级结构设计者”，而非仅做文本分类任务；同时引入新的评估指标 Taxonomy Probing Metric（TPM）衡量模型对层级结构的可辨识度。

**🔧 技术方法**

核心技术包括：大型语言模型（Anthropic Claude 3、Haiku、Sonnet‑3/3.5）用于生成树形变换，基于 Levenshtein 距离的自动纠错，DistilBERT 编码器+二元交叉熵分类器，TPM 作为结构对齐评估。

**📊 数据集**

在三个公开 HTC 基准上验证：Amazon（产品评论分类）、Books（图书类目）、Web of Science（科研论文分类）。

**📈 对比分析**

与原始人类手工标签和两种控制基线（线性层、结构化编号）对比，LLM 重构的层级在单节点表示下平均提升 F1 约 +1.7pp（Books 上最大 +2.9pp）；在 few‑shot 场景下亦保持优势；TPM 指标虽然低于原始层级，但更能解释模型的预测表现，表明更具表达力的层级结构虽在嵌入空间更难聚类，却能提升分类准确率。

**⚠️ 局限性**

主要局限包括：对改造后层级结构的语义合理性、偏见与领域一致性缺乏深入分析；实验仅覆盖基于编码器的 Transformer（DistilBERT、BERT）且未测试生成式或图模型；使用的 LLM 集合局限于同一家族（Anthropic Claude）且依赖固定模板，缺乏人工评估对改动质量的定性检验。

---

## 688. Making medical vision-language models think causally across modalities with retrieval-augmented cross-modal reasoning

**arXiv ID:** 2601.18356 | [PDF](https://arxiv.org/pdf/2601.18356v1)

**作者:** Weiqin Yang `[一作]`, Tingbo Zhang `[通讯]` (Hohai University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于因果检索增强生成的多模态框架（MCRAG），通过因果图指导检索与生成，提升医学视觉语言模型的事实性与鲁棒性。

**💡 创新点**

首次将结构因果模型与跨模态检索结合，并引入因果一致性评分、手工精炼以及因果优先级的检索筛选，显著降低因果相关的虚假输出。

**🔧 技术方法**

使用结构因果模型（SCM）、视觉语言模型（VLM）、对比学习检索、因果一致性评分、手工精炼、LoRA 微调以及 RAG 框架。

**📊 数据集**

在 MIMIC‑CXR 与 IU‑Xray 两大放射学数据集上进行视觉问答和报告生成实验。

**📈 对比分析**

与 MMed‑RAG、FactMM‑RAG、RULE 等基线相比，MCRAG 在 IU‑Xray VQA 上 Acc 提升 0.58、F1 提升 1.31、AUC 提升 1.12；在 MIMIC‑CXR VQA 上 Acc 提升 1.34、F1 提升 0.88、AUC 提升 1.34；报告生成 BLEU 提升 3.64（IU‑Xray）与 2.56（MIMIC‑CXR），实现了 SOTA。

**⚠️ 局限性**

依赖 VLM 能准确提取因果结构；检索时需要额外的模型查询，导致计算开销增加；在知识快速演进或高度专业化的领域，因果图构建可能不够准确，影响性能。

---

## 689. When Domain Pretraining Interferes with Instruction Alignment: An Empirical Study of Adapter Merging in Medical LLMs

**arXiv ID:** 2601.18350 | [PDF](https://arxiv.org/pdf/2601.18350v1)

**作者:** Junyi Zou `[一作]` `[通讯]` (Zjydiary Group), Junyi Zou (Zjydiary Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过两阶段 LoRA 训练（先域自适应预训练后监督微调）和加权适配器融合，构建了一款 14B 参数的医学问答 LLM，并对其安全性与推理能力进行了评估。

**💡 创新点**

创新点在于提出加权适配器融合（Weighted Adapter Merging）策略，平衡知识注入与指令对齐的冲突，并揭示表面指标与推理准确性在医学任务中的不一致性。

**🔧 技术方法**

使用了 LoRA（低秩参数高效微调）、QLoRA（4-bit 量化）、FlashAttention-2、以及模型融合的线性加权方法。

**📊 数据集**

主要数据集包括 MedCorpus（2GB 文本）、MedInstruct（50k QA 对话）、MedQA（1.2k MCQ）、PubMedQA（1k QA）以及内部拆分的验证集 F5/F6。

**📈 对比分析**

通过 BLEU‑4、ROUGE‑1/2/L、MedQA 及安全性拒绝率等指标对比，发现纯 SFT 最高 BLEU，但 30% PT / 70% SFT 的混合模型在推理准确性、拒绝率等安全指标上更优，且对解码温度更鲁棒。

**⚠️ 局限性**

局限性包括：（1）表面指标与实际推理能力不一致，导致评估偏差；（2）安全性评估仅基于有限 probe，未覆盖完整安全基准；（3）加权融合比例仍需手工调优，缺乏自动化选择方法。

---

## 690. Maps of Tournaments: Distances, Experiments, and Data

**arXiv ID:** 2601.18348 | [PDF](https://arxiv.org/pdf/2601.18348v1)

**作者:** Filip Nikolow `[一作]` (AGH University), Stanisław Szufa `[通讯]` (AGH University)

**通讯引用:** 106 | [OpenAlex ID](https://openalex.org/A5004773077)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将“选举地图”框架扩展到锦标赛图，构建锦标赛地图并用于可视化与分析实验结果

**💡 创新点**

提出将图编辑距离（GED）与Katz中心性距离结合生成地图，验证两者在大规模锦标赛上的可比性；同时给出多种生成锦标赛的随机模型并与真实数据对比

**🔧 技术方法**

使用图编辑距离、Katz中心性距离、MDS嵌入、ILP求解（Slater、可能淘汰冠军）以及统计学生成模型（Uniform、Condorcet噪声、强度模型）

**📊 数据集**

使用NBA常规赛、波兰桥牌联赛、已发表的随机锦标赛数据以及基于选举的锦标赛（无偏文化模型）

**📈 对比分析**

通过相关系数（PCC/SCC）、平均失真、计算时间比较两种距离的有效性；Katz距离在计算速度上快数百倍且与GED相关性在0.5左右，MDS嵌入的失真在0.2-0.3之间

**⚠️ 局限性**

限制在于GED计算仅适用于较小规模锦标赛，Katz距离忽略图结构细节，生成模型仍未覆盖地图中所有区域；未来需寻找更快且跨规模的距离以及更丰富的随机模型

---

## 691. Q-Bench-Portrait: Benchmarking Multimodal Large Language Models on Portrait Image Quality Perception

**arXiv ID:** 2601.18346 | [PDF](https://arxiv.org/pdf/2601.18346v1)

**作者:** Sijing Wu `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 9611 | [OpenAlex ID](https://openalex.org/A5043405654)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Q‑Bench‑Portrait，一个面向肖像图像质量感知的综合基准，包含2,765个图像-问题-答案三元组。

**💡 创新点**

创新点在于：①首个专门针对肖像图像的低层质量评估基准；②覆盖自然、合成失真、AI生成、艺术及电脑图形多种来源；③从技术失真、AIGC特有失真和审美三个维度构造多类型问题（单选、多选、真伪、开放式），同时考虑全局与局部问答。

**🔧 技术方法**

采用YOLOv9+MTCNN自动提取肖像，利用Gemini等LLM生成描述，再通过GPT辅助与人工双轮审校生成问题；评测使用标准准确率和OpenAI ChatGPT进行开放式答案打分。

**📊 数据集**

使用了17个公开图像质量与美学数据集：CGFIQA‑40k、PIQ、LiveBeauty、MEBeauty、KonIQ‑10k、SPAQ、AVA、TAD66K、FIQA、PIPAL、KADID‑10k、AGHI‑QA、EvalMi‑50k、ArtEmis、BAID、CGIQA‑6k、NBU‑CIQAD。

**📈 对比分析**

在25个MLLM（20开源+5闭源）上进行基准测试，得到最高准确率为64.00%（Qwen3‑VL‑32B），虽然该模型超越部分闭源模型，但整体表现仍有限；多选题最难，局部问题最弱，AI生成与CG图像难度最高。

**⚠️ 局限性**

局限性包括：①整体准确率仅在60%区间，显示模型在肖像质量感知上仍有较大提升空间；②对AI生成和电脑图形图像的识别能力不足；③多选和局部问题的难度导致模型表现低落；④评测主要基于人工标注与ChatGPT评分，可能存在主观偏差。

---

## 692. Structural Gender Bias in Credit Scoring: Proxy Leakage

**arXiv ID:** 2601.18342 | [PDF](https://arxiv.org/pdf/2601.18342v1)

**作者:** Navya SD `[一作]`, SS Uma Sankari `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究对台湾信用违约数据集中的结构性性别偏见进行了全面审计，揭示即使去除显式性别特征，模型仍通过非敏感特征泄露性别信息并维持歧视；

**💡 创新点**

创新点在于将SHAP解释性方法与对抗性逆向建模相结合，量化了代理变量中的性别泄漏，并挑战了“盲目公平”假说；

**🔧 技术方法**

主要技术包括特征预处理（类权重、SMOTE、抽样）、Logistic回归与XGBoost模型、SHAP可解释性分析、SHAP-Gender T检验及逆向预测模型评估；

**📊 数据集**

使用的公开数据集为台湾信用违约（30,000条记录，包含30个特征，包括性别、教育、婚姻状况、年龄等非财务变量与信用额度、账单金额等财务变量）；

**📈 对比分析**

通过对12种实验配置进行公平性指标（Disparate Impact、Equalized Odds Difference、Demographic Parity Difference）和模型性能评估，发现所有配置在公平性指标上均满足监管阈值，但SHAP和逆向建模表明存在显著的结构性偏差，ROC‑AUC可达0.65；

**⚠️ 局限性**

局限性包括仅使用单一地区的数据集，未建立因果关系框架，逆向模型仅捕捉相关性，且缺乏跨数据集验证和对监管差异的深入探讨。

---

## 693. Agentic Much? Adoption of Coding Agents on GitHub

**arXiv ID:** 2601.18341 | [PDF](https://arxiv.org/pdf/2601.18341v1)

**作者:** Romain Robbes `[一作]` (University of Bordeaux), Stefano Zacchiroli `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 GitHub 开源项目进行大规模采样，利用文件、提交、拉取请求及标签等多种痕迹，系统量化了 2025 年出现的“编程代理（coding agents）”的采用率、使用场景、项目与组织特征、提交规模与类型等。研究覆盖了数千个活跃项目，并对代理的贡献与人类、普通机器人提交做对比。

**💡 创新点**

创新点在于：①首次以多维痕迹（文件、提交、PR、标签）构建全面的代理使用检测方法；②在真实世界中对代理使用进行全景式、细粒度的统计与分类，揭示了代理在不同项目成熟度、组织、主题与语言上的广泛应用；③采用自动化流水线实现对持续快速演进的代理生态的实时跟踪。

**🔧 技术方法**

技术手段主要包括：GitHub REST 与 GraphQL API 采集数据；Python/SQL 进行日志解析、文件列表检索；基于手工编制与验证的 heuristics 进行代理痕迹识别；统计分析与可视化（Matplotlib/Seaborn）展示结果；使用 Conventional Commit 规范对提交类型进行自动与人工标注。

**📊 数据集**

数据集来源：GitHub 开源仓库采样（Dabic 等工具），筛选出非 fork、至少 5000 行代码、100 条以上提交、近期活跃的项目，最终样本规模约 10–12k 项目。对这些项目的代码、提交、PR 进行全面抓取与处理。

**📈 对比分析**

对比方法：①对代理提交与人类/普通机器人提交在行数、文件数、删除/新增量等指标进行统计分布对比；②对提交类型（feat、fix、docs 等）采用 Conventional Commit 规范进行分类，比较代理与人类在功能性/维护性工作上的比例差异；③利用占比、比例、分布等指标展示代理使用的规模与趋势。结果显示：代理提交平均新增行数、删除行数及涉及文件数显著高于人类提交；功能性（feat、fix）提交比例远高于人类，且代理提交往往覆盖多文件。

**⚠️ 局限性**

局限性：①仅能检测公开可见的痕迹，可能低估实际采用率；②开发者可能手动提交代理产生的代码，导致漏检；③采样偏向高星级、活跃项目，难以代表所有 GitHub 仓库；④不同代理的签名/标签规范差异导致部分痕迹识别不完全；⑤对提交类型的自动分类依赖 Conventional Commit，未覆盖全部真实提交语义；⑥由于数据来源是公开仓库，无法验证内部企业使用情况。

---

## 694. PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction

**arXiv ID:** 2601.18336 | [PDF](https://arxiv.org/pdf/2601.18336v1)

**作者:** Isaac Deutsch `[一作]` (NVIDIA), Zan Gojcic `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个可微分的相机成像管线（PPISP），能够在多视角3D重建过程中学习并在推理时通过控制器预测曝光与色彩校正，从而提升新视角合成质量。

**💡 创新点**

创新点包括：①将曝光、散光、色彩校正和CRF等四个模块按物理模型分离，明确区分相机内在与时变效应；②设计基于渲染辐射场的控制器，自动预测新视角的ISP参数；③支持直接利用图像元数据（如曝光补偿）进一步提升预测精度。

**🔧 技术方法**

采用可微分渲染与NeRF/3DGS框架，四阶段ISP管线（曝光偏移、散光、多通道线性校正、非线性CRF），控制器为CNN+MLP，使用Huber正则化及亮度/颜色惩罚，实验对比BilaRF、ADOP、RawNeRF等后处理方法。

**📊 数据集**

实验数据集包括标准NeRF基准（Mip-NeRF360、Tanks and Temples、BilaRF、HDR-NeRF）、Waymo Open Dataset序列，以及自制PPISP数据集（四个场景，用三台手机相机拍摄）。

**📈 对比分析**

在所有基准上与BilaRF、ADOP等方法对比，使用PSNR/SSIM/LPIPS（含对齐指标PSNR-CC）评估；PPISP在训练视角和新视角均达到或超过基线，尤其在新视角时与对齐指标相近，Runtime略高但仍快于BilaRF。

**⚠️ 局限性**

局限性包括：1）未能建模局部调色、镜头光晕等空间自适应效应；2）当相机手动调节曝光/光圈等导致与渲染辐射缺乏相关性时，控制器预测误差；3）在训练视角上有时略逊于基线，可能因过拟合或模型容量不足。

---

## 695. Fusion of Spatio-Temporal and Multi-Scale Frequency Features for Dry Electrodes MI-EEG Decoding

**arXiv ID:** 2601.18424 | [PDF](https://arxiv.org/pdf/2601.18424v1)

**作者:** Tianyi Gong `[一作]` (Chinese University of Hong Kong), Dahong Qian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2086 | [OpenAlex ID](https://openalex.org/A5081179416)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种三分支融合网络STGMFM，用于低信噪比的干电极运动想象EEG解码

**💡 创新点**

创新点包括：①双序列图卷积（CCG→TSG与TSG→CCG）产生互补的空间-时间特征；②多尺度频率混合器(MFM)在幅度包络层面捕捉ERD/ERS特征；③用相位锁定值(PLV)初始化可学习的电极图谱作为生理先验；④简单的决策层融合提高鲁棒性

**🔧 技术方法**

技术主要为图神经网络、时序卷积、频谱映射与多尺度混合、PLV先验、L1/L2正则与余弦退火训练

**📊 数据集**

在23通道、250Hz的自采干电极MI-EEG数据集上进行实验，包含19位受试者、两天会话、三类运动想象任务

**📈 对比分析**

与ShallowNet、EEGNet、EEGTCNet、EEGConformer、BaseNet、LMDANet、STGENet等基线比较，STGMFM在跨会话、跨受试者及预训练+微调三种评估协议下均取得最高准确率、Kappa和F1，尤其在跨受试者场景下准确率达57.26%

**⚠️ 局限性**

局限性包括：仍需大量数据进行微调；对极低采样率或更大通道数的适应性未验证；模型复杂度略高，尚未在移动设备上进行部署评估

---

## 696. Fundamentals, Recent Advances, and Challenges Regarding Cryptographic Algorithms for the Quantum Computing Era

**arXiv ID:** 2601.18413 | [PDF](https://arxiv.org/pdf/2601.18413v1)

**作者:** Darlan Noetzold `[一作]`, Valderi Reis Quietinho Leithardt `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本工作综述了从经典密码学到量子密码学、后量子密码学（PQC）以及混合方案的演进历程，阐述了相关技术的原理、实现挑战与安全评估，并结合标准化进程与治理框架给出对未来安全架构的建议。

**💡 创新点**

创新点在于将历史时间线、技术分类、性能对比与合规治理统一成一套系统化框架，首次提供跨域（经典、量子、后量子）密码技术的完整评估与交叉分析；同时提出混合安全层级设计与实践落地路线。

**🔧 技术方法**

主要采用文献调研与综述方法，对 AES、RSA、ECC、QKD、PQC 家族（如 Kyber、Dilithium、SPHINCS+）、同态加密（BGV/BFV/CKKS/FHEW/TFHE）等核心算法进行理论与实践分析；使用标准化工具（NIST PQC 评估、HEBench）和协议性能基准（TLS 1.3/QUIC）。

**📊 数据集**

数据来源主要为公开文献、NIST 公开评估报告、HEBench 性能数据以及各类协议实现基准测试（如 TLS 流量、QKD 链路测距）。并未使用专有数据集，而是聚合已公开的实验结果。

**📈 对比分析**

通过构建表格与时间线，对算法的安全级别、密钥长度、运行时延、能耗、实现难度等指标进行横向对比；性能评估显示，经典方案（AES/GCM）仍是基线，而 QKD 在光纤中可实现数十公里无泄漏，PQC 方案在 NIST 选定级别下提供 128‑bits 以上安全且密钥尺寸适中；同态加密在目前的硬件实现下，单次加解密时间往往在秒级以上，仍属于研究阶段。

**⚠️ 局限性**

局限性包括：量子硬件与 QKD 基础设施成本高、部署距离受限；同态加密计算成本高、内存占用大；后量子方案标准化仍在进行中，兼容性与实现细节尚未完全成熟；治理与合规框架在多国法规环境下需进一步细化；本文主要基于文献与公开基准，缺乏大规模实测验证。

---

## 697. Frequency-Based Hyperparameter Selection in Games

**arXiv ID:** 2601.18409 | [PDF](https://arxiv.org/pdf/2601.18409v1)

**作者:** Aniket Sanyal `[一作]` (TU Munich), Tatjana Chavdarova `[通讯]` (TU Wien)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在游戏优化中 LookAhead（LA）算法的超参数选择，提出了一种基于模态频域分析的自适应方法 Modal LookAhead (MoLA)，并给出收敛性证明

**💡 创新点**

创新点在于将旋转动力学映射到频域，利用主模态的频率和幅值来设计最优的 LA 超参数（k、α），首次提供了 LA 的收敛保证并显著降低了手动调参的依赖

**🔧 技术方法**

主要技术包括：变分不等式理论、Lipschitz 连续性与单调性分析、Z-变换与模态（频率）分析、极点（pole）分析、梯度下降与 LookAhead 迭代框架、主模态估计与自适应参数选择

**📊 数据集**

实验数据集包括两类：随机高维双线性游戏（d=100，随机生成矩阵）和结构化强凸–强凹二次游戏（SC‑SC），通过不同旋转强度（σ 或 β）对算法进行评估

**📈 对比分析**

与基线方法（GD、Extragradient、Optimistic GD、随机选 k 的 LA）比较，MoLA 在距离平衡点和 CPU 时间上均优于所有对照组，尤其在高旋转环境下显著加速收敛，且对超参数不敏感

**⚠️ 局限性**

局限性包括：主模态估计在极大规模问题上可能成本较高，当前仅在局部线性化下分析，未考虑非平滑或非单调情况，且需进一步验证在实际 GAN 或多智能体强化学习中的效果

---

## 698. CitiLink: Enhancing Municipal Transparency and Citizen Engagement through Searchable Meeting Minutes

**arXiv ID:** 2601.18374 | [PDF](https://arxiv.org/pdf/2601.18374v1)

**作者:** Rodrigo Silva `[一作]` (University of Beira Interior), Ricardo Campos `[通讯]` (University of Beira Interior)

**通讯引用:** 1832 | [OpenAlex ID](https://openalex.org/A5089440969)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 CitiLink 平台，将市政会议纪要转换为结构化数据并提供可搜索界面。

**💡 创新点**

结合 LLM 提取多层元数据、讨论主题和投票结果，并将其与数据库索引结合，实现多维过滤和全文检索。

**🔧 技术方法**

使用 Gemini 2.0 Flash LLM 与 prompt engineering 进行信息抽取，MongoDB Atlas 存储，React 前端，Flask API，BM25 排序。

**📊 数据集**

基于六个葡萄牙市政的 120 条会议纪要（含手动匿名化和 DeepL 英文翻译）作为评测数据。

**📈 对比分析**

对比人工标注的元数据、主题、投票三层，元数据宏 F1=0.84，主题 ROUGE‑L=0.31、BLEU=0.21，投票宏 F1=0.67，表明元数据抽取效果好但主题和投票仍需改进。

**⚠️ 局限性**

目前仅针对葡萄牙语纪要，主题抽取不提供位置对齐，投票识别准确率偏低，且缺乏用户群体广泛的真实反馈。

---

## 699. Pisets: A Robust Speech Recognition System for Lectures and Interviews

**arXiv ID:** 2601.18415 | [PDF](https://arxiv.org/pdf/2601.18415v1)

**作者:** Ivan Bondarenko `[一作]` (Novosibirsk State University), Lyudmila Budneva `[通讯]` (Novosibirsk State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种三组件ASR系统Pisets，用于科学家和记者的语音转文本，结合Wav2Vec2分段、AST过滤误报以及Whisper最终识别，并通过课程学习和不确定性建模提升性能。

**💡 创新点**

创新点在于将Wav2Vec2 VAD与AST后置过滤相结合，构建三层识别链，减少Whisper模型的幻觉，并引入多源不确定性估计与BIRM微调。

**🔧 技术方法**

使用的技术包括Wav2Vec2、Audio Spectrogram Transformer (AST)、Whisper、BIRM、课程学习策略、语音拉伸的TTA、以及基于Whisper概率的不确定性评分。

**📊 数据集**

训练和评估使用的语料包括Golos、Russian Librispeech、RuDevices、Taiga Speech、Podlodka Speech等俄语语音数据集。

**📈 对比分析**

与WhisperX对比，Pisets在长音频上实现了WER 0.1065、BERT-score 0.9652，显著优于WhisperX的WER 0.1683、BERT-score 0.9479，验证了在多噪声条件下的更高准确性。

**⚠️ 局限性**

局限性主要是对同音异义词和相似发音短语的识别不足，需要引入语义与语用层面的理解，例如使用大规模多模态模型来提升对上下文的把握。

---

## 700. Algebraic Characterizations of Classes of Regular Languages in DynFO

**arXiv ID:** 2601.18429 | [PDF](https://arxiv.org/pdf/2601.18429v1)

**作者:** Corentin Barloy `[一作]`, Thomas Zeume `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

无法获取足够信息来判断论文的主要工作

**💡 创新点**

无法判断论文的创新点

**🔧 技术方法**

无法判断使用的技术

**📊 数据集**

无法判断使用的数据集

**📈 对比分析**

无法判断比较方法及其性能表现

**⚠️ 局限性**

无法判断论文的局限性

---

## 701. Larger than memory image processing

**arXiv ID:** 2601.18407 | [PDF](https://arxiv.org/pdf/2601.18407v1)

**作者:** Jon Sporring `[一作]` (University of Copenhagen), David Stansby `[通讯]` (University College London)

**通讯引用:** 6207 | [OpenAlex ID](https://openalex.org/A5041800777)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种面向大型图像的单通道流式处理框架，能够在有限内存下对PB级卷积、分割等计算进行单次扫描；

**💡 创新点**

创新点在于将3D数据映射到2D切片堆栈并构建基于窗口的流式 DSL，自动完成窗口大小、融合与调度，显著降低I/O重复读取；

**🔧 技术方法**

使用了基于 F# 的 DSL、窗口化滑动、管道编译与运行时分析、堆栈式 I/O、Zarr/HDF5 存储访问、分支与合并（tee/zip）等技术；

**📊 数据集**

主要评估使用了人类器官图谱 Hub 的 HiP‑CT 多分辨率数据（高达 2 TB）以及 1.4 PB 的电子显微镜体积；

**📈 对比分析**

通过与 ITK 流式滤波、Dask/Xarray 并行块处理和传统 3D 块遍历对比，单通道流式实现只需一次或两次读取，内存占用可控制在 1 TB，吞吐量提升数倍至十数倍；

**⚠️ 局限性**

局限在于对全局变换（如 3D 变形、FFT）需暂时切换至块式布局；对极大核尺寸或多核并行时窗口调度复杂度增加；

---

## 702. On the Bandwidth Consumption of Blockchains

**arXiv ID:** 2601.18400 | [PDF](https://arxiv.org/pdf/2601.18400v1)

**作者:** Andrei Lebedev `[一作]` (University of Sydney), Vincent Gramoli `[通讯]` (Redbelly Network)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

对 Algorand、Aptos、Avalanche、Redbelly 和 Solana 五大 Layer‑1 区块链的带宽消耗进行实验测量与对比，揭示不同网络协议和区块传播策略对流量的主导影响。

**💡 创新点**

首次系统性、实证性地比较多种主流区块链的网络流量，并通过细粒度的流量监测与热图分析，展示了协议设计对带宽、可扩展性和能耗的深远影响。

**🔧 技术方法**

采用基于 iptables 的流量计数器、5 节点客户端与 20 节点节点的实验环境，结合 WebSocket / HTTP polling、TCP、UDP 等传输协议，使用 Diabolo/Stable 等基准框架对链进行部署。

**📊 数据集**

使用自行生成的约 20,000 条交易数据（分布式客户端负载）和 25 台 Ubuntu 虚拟机构成的实验集群，未使用公开大规模数据集，而是通过自制实验负载与参数化实验场景来评估带宽。

**📈 对比分析**

通过对各链的发送/接收流量、热图、不同节点/验证者规模下的总带宽、不同发送速率下的绝对带宽等多维度量化指标进行对比；结果显示 Solana 最高，Aptos 在空闲期产生大量流量，Algorand 与 Redbelly 受验证者数影响显著，Avalanche 与 Aptos 两者同时受节点数与验证者数影响。

**⚠️ 局限性**

实验规模有限（25 节点、单一数据中心），未覆盖大规模分布式网络；节点硬件规格低于某些链的推荐配置；只评估了五个链，结果对其他协议的泛化受限；流量监测仅在 VM 层面，未考虑云服务商计费模式和真实网络延迟等外部因素。

---

## 703. Corpus-Based Approaches to Igbo Diacritic Restoration

**arXiv ID:** 2601.18380 | [PDF](https://arxiv.org/pdf/2601.18380v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 704. ARMOR: Agentic Reasoning for Methods Orchestration and Reparameterization for Robust Adversarial Attacks

**arXiv ID:** 2601.18386 | [PDF](https://arxiv.org/pdf/2601.18386v1)

**作者:** Gabriel Lee Jun Rong `[一作]`, Konstantinos N. Plataniotis `[通讯]` (University of Toronto)

**通讯引用:** 21464 | [OpenAlex ID](https://openalex.org/A5059152392)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 ARMOR 框架，利用 VLM 和 LLM 组成的多智能体系统，在对抗攻击中动态协作、语义感知并实时重参数化 CW、JSMA、STA 三种攻击方法，以生成具有更高成功率和更好转移性的对抗样本。

**💡 创新点**

创新点在于：①将 Vision‑Language 模型用于图像语义推理，为攻击决策提供内容上下文；②让 LLM 充当策略指导者，实时调整攻击超参数和混合权重；③构建闭环“Mixing Desk”实现对抗样本的自适应混合和评估，显著提升跨模型转移性能；④将多攻击方式并行协作，实现比传统静态组合更灵活的攻击策略。

**🔧 技术方法**

使用技术包括：Vision‑Language 模型 Qwen2.5‑VL‑32B‑Instruct‑AWQ 进行图像分析；大型语言模型 Qwen3‑32B‑AWQ 负责策略规划与重参数化；CW、JSMA、STA 三种对抗方法并行生成扰动；随机化爬山法进行混合权重优化；SSIM 与置信度作为闭环评价指标；对抗样本在保持 ‖δ‖∞ 约束下进行投影。

**📊 数据集**

实验数据集为 AADD‑LQ 子集（710 张低质量伪造图像），对比了多种基准攻击（MI‑FGSM、DI‑FGSM、TI‑FGSM、SimBA‑DCT、AutoAttack 等）。

**📈 对比分析**

与传统的基于转移、查询、集成和其它 agentic 攻击相比，ARMOR 在 surrogate 模型上实现 100% 的攻击成功率和高 wASR；在盲目标 ViT‑B/16 上的 ASR 为 0.396，wASR 0.280，条件转移概率 0.396，显著高于其它方法；消融实验表明多智能体协作与语义推理是提升性能的关键。

**⚠️ 局限性**

局限性包括：对抗样本在转移到盲模型时仍存在显著下降；需要较大 VLM/LLM 计算资源，难以轻量化部署；仅验证了 CW、JSMA、STA 三种攻击，其他对抗方法尚未纳入；对不同架构的通用性和对抗鲁棒性的进一步提升仍有空间。

---

## 705. Dynamic Thinking-Token Selection for Efficient Reasoning in Large Reasoning Models

**arXiv ID:** 2601.18383 | [PDF](https://arxiv.org/pdf/2601.18383v1)

**作者:** Zhenyuan Guo `[一作]` (Zhejiang University), Wenzhi Chen `[通讯]` (Zhejiang University)

**通讯引用:** 3990 | [OpenAlex ID](https://openalex.org/A5101562846)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型推理模型（LRM）中的推理轨迹，提出动态思考-令牌选择（DynTS）机制，在推理过程中仅保留对最终答案至关重要的思考令牌，从而实现KV缓存压缩。

**💡 创新点**

创新点：①通过对答案令牌对思考令牌的注意力加权求和得到客观重要性分数；②引入轻量级的重要性预测器（MLP），可在推理时即时预测当前思考令牌的重要性；③结合本地窗口与选择窗口的双窗口机制，在达到预算时按重要性分数保留KV缓存，避免错误剔除关键令牌。

**🔧 技术方法**

使用的技术：Transformer‑based LRM、注意力权重重要性度量、3‑层 MLP 重要性预测器、KV 缓存选择策略、理论计算量与突破点分析、Pareto 原理验证、自动推理与自回归生成。

**📊 数据集**

使用的数据集：DeepSeek‑R1‑Distill‑Llama‑8B 与 DeepSeek‑R1‑Distill‑Qwen‑7B 作为模型；六个推理基准：AIME24、AIME25、AMC23、GK23EN、MATH500、GPQA‑D。

**📈 对比分析**

与传统 full‑cache Transformers 以及 SOTA KV 压缩方法（StreamingLLM、H2O、SepLLM、SnapKV、R‑KV）比较，DynTS 在保持 Pass@1 与 full‑cache 基线相当（或略优）时，推理速度提升约1.6‑1.9×，KV 缓存内存降低 3.3‑5.7×，并在同等预算下比最佳基线提升 2.6% Pass@1。

**⚠️ 局限性**

局限性：①需要额外训练重要性预测器，训练成本与推理时的计算开销仍有一定比例；②压缩效果高度依赖预算设置、窗口大小与保留比例的调参；③仅在数学推理任务上验证，未知对其他领域（如对话、代码生成）的适用性；④若预测器误判关键令牌，可能导致推理准确率下降。

---

## 706. OREHAS: A fully automated deep-learning pipeline for volumetric endolymphatic hydrops quantification in MRI

**arXiv ID:** 2601.18368 | [PDF](https://arxiv.org/pdf/2601.18368v1)

**作者:** Caterina Fuster-Barceló `[一作]` (Universidad Carlos III de Madrid), Arrate Muñoz-Barrutia `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 5442 | [OpenAlex ID](https://openalex.org/A5022934126)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一套名为OREHAS的全自动内耳泵液水肿（EH）体积量化流程，利用标准3D‑SPACE‑MRC与3D‑REAL‑IR MRI实现每耳泵液与前庭体积比例（ELR）的精确计算；

**💡 创新点**

首次实现从极少量标注（3‑6张切片）即可推广至完整3D体积的全流程自动化，公开代码、可追溯，且对比商业软件syngo.via表现出更准确、可重现的阈值；

**🔧 技术方法**

采用三阶段深度学习架构：5层卷积+ResNet50做切片分类（EarGate），YOLOv5实现耳部定位（AuriBox），U‑Net+BCE+Dice损失进行序列专属分割（EHMasker），随后3D重建与体积计算；

**📊 数据集**

使用90例CUN临床样本（83例Menière病患者+7例对照），采集3T Siemens空间扫描，获取3D‑SPACE‑MRC与3D‑REAL‑IR序列；

**📈 对比分析**

与syngo.via及5例完整手工标注样本比较，Dice分别达0.90（SPACE‑MRC）和0.75（REAL‑IR），REAL‑IR VSI为74%对比42%；ELR值与syngo.via显著偏低，显示软件偏高估计；

**⚠️ 局限性**

样本量有限、标注范围有限（仅少量切片），可能导致跨中心泛化受限；REAL‑IR图像低对比导致分割误差；以及syngo.via内部插值不透明导致对比偏差。

---

## 707. Comparative Evaluation of Machine Learning Algorithms for Affective State Recognition from Children's Drawings

**arXiv ID:** 2601.18414 | [PDF](https://arxiv.org/pdf/2601.18414v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 708. Gradient Regularized Natural Gradients

**arXiv ID:** 2601.18420 | [PDF](https://arxiv.org/pdf/2601.18420v1)

**作者:** Satya Prakash Dash `[一作]` (University of Manchester), Mingfei Sun `[通讯]` (University of Manchester)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5101591811)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一类梯度正则化的自然梯度优化器（GRNG），通过在自然梯度更新中加入显式或隐式梯度正则化，构造了可扩展的频率和贝叶斯两种实现。

**💡 创新点**

创新点在于将梯度正则化与自然梯度深度融合，设计了基于块式 Kronecker 近似、惰性 FIM 估计和 Newton 迭代的频率版，以及利用 Kalman 滤波完全消除 FIM 逆运算的贝叶斯版，从而兼顾收敛速度与泛化能力。

**🔧 技术方法**

核心技术包括 Kronecker‑factored FIM 近似、惰性 Hessian（Lazy Fisher）、Newton 迭代求逆、双向梯度回传（double back‑prop）、隐式梯度正则化、Kalman 滤波以及正则化的观测噪声协方差。

**📊 数据集**

实验使用的视觉数据集包括 CIFAR‑10/100、Oxford‑IIIT Pet、Food‑101 与 ImageNet‑100，语言数据集为 GLUE 基准中的 MNLI‑mm、QQP、QNLI、SST‑2、CoLA、STS‑B、MRPC、RTE；模型分别是 ViT‑B16（图像）与 RoBERTa‑base（文本），并采用 LoRA 参数高效微调。

**📈 对比分析**

与 AdamW、Sophia、NGD 等基线比较，GRNG 在大多数任务上实现了更高的验证/测试准确率，RING/RENG 在大规模数据集上优势明显，R‑Kalman 在低数据/少样本场景表现最佳；同时总运行时间显著降低，表明收敛速度更快。

**⚠️ 局限性**

局限性包括：1）仍需对 FIM 近似进行参数调优，惰性估计在极度非线性阶段可能不稳定；2）在极大模型或超大批量时，Kronecker 近似与 Kalman 更新的计算和存储开销仍不可忽视；3）实验仅涵盖微调场景，尚未验证在从头训练或超大规模预训练任务中的表现。

---

## 709. Promises, Perils, and (Timely) Heuristics for Mining Coding Agent Activity

**arXiv ID:** 2601.18345 | [PDF](https://arxiv.org/pdf/2601.18345v1)

**作者:** Romain Robes Théo Matricon `[一作]`, Stefano Zacchiroli `[通讯]` (Telecom Paris)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对GitHub仓库的静态与动态痕迹进行挖掘，提出并验证了一套用于检测编码代理活动的启发式规则，并基于此构建了一个面向社区的共享仓库，初步评估了2025年间编码代理在开源项目中的采纳率约为15–20%。

**💡 创新点**

创新点在于首次系统化地识别并量化编码代理在软件仓库中的痕迹，提出可复制的检测启发式方法，并结合大规模实践数据揭示了编码代理对软件工程实践的潜在影响与风险，为后续研究提供了方法论框架。

**🔧 技术方法**

主要技术方法包括：利用GitHub搜索API与代码分析进行文件/提交/PR/issue痕迹的自动检索；结合手工标注与经验总结构建启发式规则；使用Python脚本进行批量数据采集与统计分析。

**📊 数据集**

数据集主要来自GitHub公开仓库，使用约1万余个已标注包含编码代理痕迹的仓库作为样本，并通过搜索查询统计各类痕迹（如AGENTS.md、Co‑authored‑by等）的出现频率。

**📈 对比分析**

研究并未针对代理性能进行对比，而是通过对不同代理的痕迹计数与采纳率进行量化描述，指出如Claude、Codex、Cursor、Copilot等主流代理共占约80%的采纳份额，并提供PR合并率等指标作为行为后果的初步观察。

**⚠️ 局限性**

主要局限包括：观测不完整（部分代理不公开或隐藏痕迹）、代理多样性导致启发式泛化困难、快速迭代的代理技术使得检测规则易失效、闭源LLM导致可重复性差，以及高成本与“AI编码斑点”可能对研究数据质量产生影响。

---

## 710. TopKGAT: A Top-K Objective-Driven Architecture for Recommendation

**arXiv ID:** 2601.18432 | [PDF](https://arxiv.org/pdf/2601.18432v1)

**作者:** Sirui Chen `[一作]` (Zhejiang University), Can Wang `[通讯]` (Zhejiang University)

**通讯引用:** 11352 | [OpenAlex ID](https://openalex.org/A5100428567)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为 TopKGAT 的图注意网络架构，直接以可微分的 Precision@K 目标驱动前向计算，从而在训练时本身就优化 top‑K 推荐质量。

**💡 创新点**

创新点在于：① 将 top‑K 指标通过分位数阈值和 sigmoid 平滑化得到可微分形式；② 设计了“带通”激活函数 ω(·)=4σ′(·)，仅对接近阈值的用户–物品相似度赋予高权重；③ 将阈值 β_u^(l) 设为可学习的、层次化的用户特定参数，使模型能够自适应地聚焦不同层次的 top‑K 决策。

**🔧 技术方法**

采用图注意网络（GAT）结构，结合可微分的 top‑K 目标、带通激活、可学习阈值以及基于相似度的加权聚合；实验中使用了标准的 5‑core 采样和交叉验证。

**📊 数据集**

在四个真实数据集上验证：Ali‑Display、Epinions、Food、Gowalla；每个数据集均采用 7:1:2 的训练/验证/测试划分。

**📈 对比分析**

与无注意力方法（MF、LightGCN、LightGCN++、ReducedGCN）和注意力方法（GAT、NGAT4Rec、MGFormer、RankFormer）对比，TopKGAT 在 Recall@20 与 NDCG@20 上均取得平均提升 3.53%（NDCG）和 2.84%（Recall），在所有数据集上均位居第一，且提升显著（p<0.05）。

**⚠️ 局限性**

局限性包括：① 需要额外的阈值参数 β，随着层数增加会导致参数量和训练复杂度上升；② 对于极大规模图仍需采样或近似，带通激活在高维稀疏环境下可能受限；③ 目前仅针对 Precision@K，扩展到其他 top‑K 指标（如 MAP、R-Precision）还需进一步研究。

---

## 711. Time-Scale-Adaptable Spectrum Sharing for Hybrid Satellite-Terrestrial Networks

**arXiv ID:** 2601.18410 | [PDF](https://arxiv.org/pdf/2601.18410v1)

**作者:** Yanmin Wang `[一作]` (Minzu University of China), Cheng-Xiang Wang `[通讯]` (Southeast University)

**通讯引用:** 32979 | [OpenAlex ID](https://openalex.org/A5100779393)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在卫星-地面混合网络中基于统计CSI的时频谱共享框架，提出了可调节时间尺度的协同链路调度与功率控制方案；

**💡 创新点**

创新点在于结合粗同步下的时间尺度可调性，利用链路特征投影与分层聚类实现低复杂度的链路分配，并支持多卫星选择；

**🔧 技术方法**

使用了链路特征投影、分层K-means聚类、蒙特卡洛仿真、顺序逼近凸优化、标准指派问题求解等技术；

**📊 数据集**

实验使用基于模拟的卫星-地面网络拓扑，包括3颗卫星、28个基站、24个基站用户和96个卫星用户，参数依据实际路径损耗与阴影模型生成；

**📈 对比分析**

与FineSync、PartialPreScheme和RandScheme等基准方案对比，结果表明在严格的互链路干扰约束下，提出方案平均增益超过15%~23%，且大部分卫星用户满足QoS；

**⚠️ 局限性**

主要局限在于对统计CSI的依赖、需要生成大量Monte Carlo样本、以及在全频重用情况下干扰增大导致性能下降，且对动态移动环境的适应性尚待进一步验证。

---

## 712. Estimating Dense-Packed Zone Height in Liquid-Liquid Separation: A Physics-Informed Neural Network Approach

**arXiv ID:** 2601.18399 | [PDF](https://arxiv.org/pdf/2601.18399v1)

**作者:** Mehmet Velioglu `[一作]`, Manuel Dahmen `[通讯]` (Forschungszentrum Jülich)

**通讯引用:** 1360 | [OpenAlex ID](https://openalex.org/A5060026142)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了基于物理信息神经网络的两阶段预训练+微调模型，并将其嵌入扩展卡尔曼滤波框架，以实现仅凭流量测量估算液液分离器中密集包层高度的软传感器。

**💡 创新点**

创新点在于：①利用低保真机理模型生成的合成数据与物理约束进行预训练，再用稀缺实验数据微调；②将可微分的PINN直接作为扩展卡尔曼滤波的预测模型；③采用集成学习提升鲁棒性，并通过简单网络映射平均DPZ高度到出口高度，实现洪水风险监测。

**🔧 技术方法**

技术手段包括物理信息神经网络（PINN）、两阶段迁移学习、自动微分、L‑BFGS与Adam优化、逆Dirichlet加权、扩展卡尔曼滤波、集成学习、YOLOv8图像检测、前馈神经网络映射。

**📊 数据集**

数据集包括：①合成仿真数据集 𝒟_sim（1000段，来源于低保真机理模型）；②实验数据集 𝒟_exp（4条时序，含流量与DPZ/水相高度）；③物理约束采样集 𝒟_physics和初始条件集 𝒟_init，用于训练时的物理损失。

**📈 对比分析**

通过与单阶段PINN、纯数据驱动的VNN以及低保真机理模型对比，在插值与外推轨迹上，预训练‑微调的PINN集成平均RMSE约为0.04‑0.06，显著优于VNN（0.07‑0.09）和机理模型（明显偏差），表明方法具有更高的预测与估算精度。

**⚠️ 局限性**

局限性包括：①假设DPZ为带状，无法准确描述高流速下的锥形分布；②实验DPZ高度测量噪声大且缺失值多，影响训练质量；③缺少靠近或超过洪水阈值的轨迹，限制了对洪水检测性能的评估。

---

## 713. OCR-Enhanced Multimodal ASR Can Read While Listening

**arXiv ID:** 2601.18393 | [PDF](https://arxiv.org/pdf/2601.18393v1)

**作者:** Junli Chen `[一作]`, Chao Zhang `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了 Donut‑Whisper，一种端到端的音视频语音识别模型，能够同时利用 Whisper 语音编码器和 Donut OCR 视觉编码器通过双编码器与交叉注意力融合来实现字幕信息与语音的协同解码。

**💡 创新点**

其创新点在于融合线性映射与滑动窗口 Q‑Former 的两种对齐结构，并通过交叉注意力进一步融合音频与视觉特征；此外提出了轻量化的知识蒸馏方案，使多模模型可用作无监督蒸馏的教师。

**🔧 技术方法**

该模型采用 Whisper‑large‑V3 语音 Transformer、Donut Swin Transformer OCR、滑动窗口 Q‑Former、交叉注意力融合模块、LoRA 微调技术以及温度化软标签蒸馏。

**📊 数据集**

为评估模型，作者构建了一个包含 57 小时中文、33 小时英文、约 103,000 条字幕注释的电影片段多语音数据集，严格保证音频与字幕的时序对齐。

**📈 对比分析**

实验结果显示，Donut‑Whisper‑base 在英文集上 WER 降低 5.6%（绝对值），中文集上 CER 降低 4.0%，显著优于单模 Donut‑base 与 Whisper‑large‑V3 基线；蒸馏后 Whisper‑large‑V3 的英文 WER 从 10.08% 降至 9.86%。

**⚠️ 局限性**

主要局限在于模型依赖可见字幕信息，对无字幕或字幕被遮挡的情况鲁棒性不足；滑动窗口大小需人工设定，且视觉 OCR 的误差仍会影响整体识别质量。

---

## 714. Gaze Prediction in Virtual Reality Without Eye Tracking Using Visual and Head Motion Cues

**arXiv ID:** 2601.18372 | [PDF](https://arxiv.org/pdf/2601.18372v1)

**作者:** Christos Petrou `[一作]` (Libra AI Technologies), Sotirios Chatzis `[通讯]` (Cyprus University of Technology)

**通讯引用:** 2519 | [OpenAlex ID](https://openalex.org/A5040795718)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无需眼动追踪的多模态VR注视预测框架，结合头戴式显示器（HMD）运动信号与视觉显著性信息，实现对未来注视方向的预测；

**💡 创新点**

创新点在于将轻量级UniSal显著性编码器与HMD运动特征进行多模态融合，并在时序模型层使用TSMixer和LSTM进行注视预测，既解决了硬件隐私与延迟问题，又实现了实时预测；

**🔧 技术方法**

核心技术包括UniSal显著性网络、TSMixer/​LSTM时序预测模型、核心ML部署以及对HMD姿态与运动特征的预处理与融合；

**📊 数据集**

使用EHTask数据集（15段360°视频，30名受试者，包含头部与眼动记录）进行训练与评估；

**📈 对比分析**

与Center‑of‑HMD和Mean‑HMD基线对比，LSTM/TSMixer在333 ms预测周期下球面RMSE约为5°，显著优于基线（约10°）；在1 s预测周期误差趋近基线但仍保持优势；

**⚠️ 局限性**

局限包括长时间预测误差逐渐增大、在低功耗设备上的实时性受限以及未考虑更复杂的用户意图或多任务场景。

---

## 715. Adversarial Synchronization

**arXiv ID:** 2601.18362 | [PDF](https://arxiv.org/pdf/2601.18362v1)

**作者:** Anton E. Lipin `[一作]` (Krasovskii Institute of Mathematics and Mechanics; Ural Federal University), Mikhail V. Volkov `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文研究了有限确定性自动机在不同同步游戏变体（k‑游戏、m/ω‑游戏等）下的胜负判定，并给出了与Černý猜想相关的极限参数（k(n)、m(n)）的理论取值，进一步阐明了同步游戏层次结构在固定大小自动机中的表现。

**💡 创新点**

创新点在于确立了所有 A_ω‑自动机类的 Černý 函数值为 n−1，给出了 A_k‑自动机类的下界（n(n−1)/2k），并证明了在 n‑状态自动机中，k(n)＝m(n)＝n²，展示了同步游戏参数与子集自动机直径的深层关联。

**🔧 技术方法**

主要技术包括：同步游戏的图形表示、子集自动机构造、图论中的强连通分量与哈密顿路径分析、有限状态机的转移单子集构造、广义字母序列长度分析、以及基于 Tarjan 算法的强连通分量求解。

**📊 数据集**

无需实际数据集，全部结论均来自纯粹的理论证明与构造示例。

**📈 对比分析**

由于是理论工作，没有经验性方法对比；相对既有同步猜想与自动机类判定结果，本研究提供了更紧的下界并给出精确的 Černý 函数值，对 A_ω‑类实现了最佳性能。

**⚠️ 局限性**

局限在于：对 A_k‑自动机（k∈ℕ）的 Černý 函数仅给出了下界，尚未给出精确值；此外，算法复杂度（O(|Q|^4·|Σ|)）仍可能可进一步优化，整体理论模型对实际应用场景的直接适用性尚未验证。

---

## 716. Uncertainty Quantification in Calibration and Simulation of Thermo-Chemical Curing of Epoxy Resins

**arXiv ID:** 2601.18359 | [PDF](https://arxiv.org/pdf/2601.18359v1)

**作者:** Jendrik-Alexander Tröger `[一作]` (Clausthal University of Technology), Stefan Hartmann `[通讯]` (Clausthal University of Technology)

**通讯引用:** 3487 | [OpenAlex ID](https://openalex.org/A5001932535)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究针对环氧树脂固化过程中的多步参数校准与数值模拟，系统评估了不确定性传播及其对模型响应的影响；

**💡 创新点**

创新点在于将一阶二矩法（FOSM）应用于多步校准与仿真，配合覆盖率检验验证其鲁棒性，并首次将该方法与Monte Carlo对比评估；

**🔧 技术方法**

采用非线性最小二乘法进行参数识别，利用FOSM进行不确定性传播与响应量预测，同时使用Monte Carlo作为基准，数值仿真采用有限元与自适应Runge–Kutta时间积分；

**📊 数据集**

使用实验数据集，包括DSC、TMDSC、LFA及温度场测量，覆盖了玻璃转变温度、固化动力学、热膨胀、比热容与热导率等六组参数；

**📈 对比分析**

通过比较FOSM与Monte Carlo结果发现，两者在温度与固化度的置信区间上基本一致，FOSM计算量显著减少（约35次模型评估对比300次），但在非高斯分布或大方差情形下略有低估；

**⚠️ 局限性**

局限性主要是FOSM为一阶近似，无法完整捕捉高度非线性或非高斯分布导致的尾部行为；当校准数据稀疏或噪声假设失效时，方差估计可能不足，需要进一步改进或结合贝叶斯/蒙特卡洛方法。

---

## 717. Can Good Writing Be Generative? Expert-Level AI Writing Emerges through Fine-Tuning on High-Quality Books

**arXiv ID:** 2601.18353 | [PDF](https://arxiv.org/pdf/2601.18353v1)

**作者:** Tuhin Chakrabarty `[一作]` (Stony Brook University), Paramveer S. Dhillon `[通讯]` (University of Michigan)

**通讯引用:** 1454 | [OpenAlex ID](https://openalex.org/A5063223563)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过让 28 名 MFA 级作家与三大 LLM（GPT‑4o、Claude 3.5 Sonnet、Gemini 1.5 Pro）在 50 位批评界认可的作者风格/声音上进行写作模拟实验，并用双盲评审（专家 28 位、非专业 131 位）比较人类与 AI 写作在质量与风格忠实度上的差异，随后对写作者进行访谈探讨其身份与美学信心的影响。

**💡 创新点**

① 发现对作者全集进行 fine‑tune 后，AI 在写作质量与风格忠实度上能显著超越人类，甚至专家也难以辨别；② 细调提升对写作者身份与创作信心的冲击；③ 提出 AI 写作可能使创意工作重新定义为“过程与意图”而非仅输出。

**🔧 技术方法**

大语言模型技术（in‑context prompting 与 fine‑tune），配合双盲评审、深度访谈以及文本质量与风格匹配的定量与定性评估。

**📊 数据集**

使用 50 位跨文化、时代多样的文学作者全集（共约 30 位用于 fine‑tune，覆盖 0.9–10.9 M tokens）以及对应的写作提示与风格说明，构成实验文本。

**📈 对比分析**

评估方法：对每个作者生成一段 200–450 字文本，专家与非专业评审分别做“写作质量”与“风格忠实度”配对偏好与理由；结果显示：in‑context 条件下专家偏好人类 82.7%，细调后专家 62% AI；非专业评审两条件均偏好 AI，细调后更强；统计显著性检验表明 fine‑tune 对 AI 写作质量提升显著。

**⚠️ 局限性**

局限性包括：仅采集美国 MFA 项目作者，样本规模有限；实验仅以短段落为单位，无法检验长篇文本连贯性；专家样本集中于 MFA，缺乏更广泛文学工作者视角；实验聚焦英语文本，未考察多语言情况；实验激励支付可能影响创作质量；未验证 AI 生成长篇作品的可行性与质量。

---

## 718. Forecasting the Maintained Score from the OpenSSF Scorecard for GitHub Repositories linked to PyPI libraries

**arXiv ID:** 2601.18344 | [PDF](https://arxiv.org/pdf/2601.18344v1)

**作者:** Alexandros Tsakpinis `[一作]` (fortiss GmbH), Alexander Pretschner `[通讯]` (Technical University of Munich)

**通讯引用:** 7208 | [OpenAlex ID](https://openalex.org/A5002011805)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了基于 GitHub 仓库历史活动的多变量时间序列模型，预测 OpenSSF Scorecard 的 Maintained 指标在未来 1~6 个月的变化。

**💡 创新点**

创新点在于首次把 Maintained 指标转化为可预测的时间序列任务，并证明对聚合表示（bucketed score、trend type）可取得高精度，且简单统计/机器学习模型可与深度学习模型媲美，降低了预测门槛。

**🔧 技术方法**

使用了三类模型：统计 VARMA、机器学习 Random Forest、深度学习 LSTM，并在不同训练窗口（3–12 个月）和预测周期（1–6 个月）下进行实验。

**📊 数据集**

数据集为 3,220 个与前 1% PyPI 重要度库（PageRank）关联的 GitHub 仓库，覆盖 2021–2023 年的 Maintained 指标历史记录。

**📈 对比分析**

实验表明：原始 Maintained 指标预测平均准确率约 0.77–0.80；bucketed score 约 0.95；trend slope 约 0.65–0.69；trend type 约 0.80–0.83。VARMA 与 Random Forest 与 LSTM 的平均性能相近，甚至略优于 LSTM，且方差更小，训练成本更低。

**⚠️ 局限性**

局限性包括：仅覆盖 GitHub 与 PyPI，且只考虑最中心的 1% 库，数据为单一时点快照，缺乏版本信息，难以推广到非中心或其他生态；实验依赖时间窗口选择与滑动窗口设置，可能存在一定的过拟合与数据泄漏风险。

---

## 719. Beyond Rigid: Benchmarking Non-Rigid Video Editing

**arXiv ID:** 2601.18340 | [PDF](https://arxiv.org/pdf/2601.18340v1)

**作者:** Bingzheng Qu `[一作]` (Institute of Computing and Intelligence), Min Zhang `[通讯]` (Institute of Computing and Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了专门用于非刚性视频编辑的基准 NRVBench、对应的 VLM 评估指标 NRVE-Acc 以及训练‑free 区域条件编辑方法 VM‑Edit，帮助系统评估和提升非刚性变形的物理合理性与时序一致性。

**💡 创新点**

创新点在于（1）构建了六大物理类别、180段高质量视频、2,340 条细粒度编辑指令和 360 个多选题的完整基准；（2）提出 NRVE‑Acc，利用视觉‑语言模型对指令遵循、物理符合度与时间连贯性三维度进行统一评分；（3）设计 VM‑Edit 的双时钟区域采样策略，实现结构保持与局部非刚性编辑的平衡。

**🔧 技术方法**

使用的核心技术包括：VLM（Qwen2.5‑VL）做评估、SAM2 生成掩码、GPT‑4o 生成说明与问答、基于 diffusion 的 I2V 模型与双时钟区域采样、光流可视化做时序一致性评估。

**📊 数据集**

数据集为来自 DAVIS 与 Pexels 的 180 条视频，按六类物理属性划分，并配有 2,340 条编辑指令、360 个 MCQ 以及人工校正的像素级掩码。

**📈 对比分析**

通过 NRVE‑Acc 与传统 CLIP/LPIPS 等指标对比，VM‑Edit 在结构保持、背景保真、文本对齐和运动保真方面优于 TokenFlow、Pyramid‑Edit、AnyV2V 等方法，获得 NRVE‑Acc 最高（第二名）并保持较高帧率；传统指标上亦表现领先。

**⚠️ 局限性**

局限性包括：需要为每个视频手动或网格搜索选择前后时钟参数，限制了极大拓扑变化的生成；方法仍受源视频先验限制，难以实现完全自由的大尺度变形；以及 VLM 评估在某些细分类别（如 HFF）上误判率略高。

---

## 720. A Dataset for Automatic Vocal Mode Classification

**arXiv ID:** 2601.18339 | [PDF](https://arxiv.org/pdf/2601.18339v1)

**作者:** Reemt Hinrichs `[一作]` (Leibniz University Hannover), Jörn Ostermann `[通讯]` (Leibniz University Hannover)

**通讯引用:** 5519 | [OpenAlex ID](https://openalex.org/A5064913233)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文收集并公开了一个新的包含四位歌手、13,000+ 录音样本的完整声乐模式分类数据集，并给出了多种基线分类器的性能评估。

**💡 创新点**

创新点在于首次构建完整声乐模式（Neutral、Curbing、Overdrive、Edge）数据集并提供多重注释（个人、合并、强/弱共识），同时展示了不同标签标准对分类性能的显著影响。

**🔧 技术方法**

采用了深度卷积神经网络（ResNet18/34）、传统机器学习算法（XGBoost、SVM、KNN、随机森林）以及光谱预处理与能量归一化等技术。

**📊 数据集**

使用了从四名歌手（两男两女）录制的3,752个唯一声乐模式样本，覆盖整个声域，共计13,335个音频样本，并在Zenodo公开发布。

**📈 对比分析**

通过5折交叉验证比较，使用合并注释时ResNet18取得81.3%平衡准确率，使用名义标签时ResNet34达到95.3%，显著优于传统模型，说明标签质量对性能影响巨大。

**⚠️ 局限性**

主要局限在于样本来源单一、专业歌手有限、低音区样本稀缺、注释者间存在较大分歧（Fleiss' κ=0.45），以及未覆盖真实音乐片段和多种演唱技术。

---

## 721. Beyond the Checkbox: Strengthening DSA Compliance Through Social Media Algorithmic Auditing

**arXiv ID:** 2601.18405 | [PDF](https://arxiv.org/pdf/2601.18405v1)

**作者:** Sara Solarova `[一作]` (Kempelen Institute of Intelligent Technologies), Ivan Srba `[通讯]` (Kempelen Institute of Intelligent Technologies)

**通讯引用:** 821 | [OpenAlex ID](https://openalex.org/A5082763244)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性分析2024年四大社交平台（YouTube、Meta FB/IG、TikTok）发布的第一波DSA审计报告，识别传统审计在算法透明度、未成年人保护及敏感数据广告三大条款上的方法缺陷，并提出算法审计作为补充手段；

**💡 创新点**

首次将监管与技术视角相结合，对真实审计报告进行定性内容分析，揭示审计方法的不一致性与技术深度不足，并系统性提出面向AI系统的算法审计框架；

**🔧 技术方法**

采用文献与文件分析方法，结合定性内容分析（directed + thematic coding）对审计报告进行编码与主题归纳；

**📊 数据集**

公开的四份非机密DSA审计报告（YouTube、Meta FB/IG、TikTok）共约180页；

**📈 对比分析**

对三条关键DSA条款（推荐系统透明度、未成年人保护、敏感数据广告）中的审计方法进行对比，发现不同平台的审计手段、证据类型、结论一致性存在显著差异；传统审计表现出点检式、时间局限性，而算法审计被认为能实现长期、动态评估，但本文未给出量化性能指标；

**⚠️ 局限性**

仅分析公开非机密版本，无法验证审计过程的真实性；缺少对审计细节的技术说明；方法主观性高，缺乏标准化评估指标；算法审计技术仍处于探索阶段，需进一步完善可重复性与真实场景模拟。

---

## 722. Closing the Modality Gap Aligns Group-Wise Semantics

**arXiv ID:** 2601.18525 | [PDF](https://arxiv.org/pdf/2601.18525v1)

**作者:** Eleonora Grassucci `[一作]` (Sapienza University of Rome), Danilo Comminiello `[通讯]` (Sapienza University of Rome)

**通讯引用:** 2669 | [OpenAlex ID](https://openalex.org/A5019647783)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种新的对抗损失，显著缩小多模态学习中的模态间差距（Modality Gap）并提升群体层任务（如聚类）性能。

**💡 创新点**

创新点在于结合 Align True Pairs Loss 与 Centroid Uniformity Loss 两个目标，既确保正样本对齐，又保持潜在空间的均匀性，从而使模态间的中心完全重合。

**🔧 技术方法**

技术方法基于 CLIP 的 InfoNCE 对比学习框架，加入自定义损失，并使用温度参数控制负样本梯度，实验中未改动网络结构，保持原有 encoder 与投影层。

**📊 数据集**

使用四大数据集：二模态 CIFAR‑10（图像‑文本）、三模态 AV‑MNIST（图像‑音频‑文本）、二模态 MSCOCO（图像‑字幕）和三模态 MSR‑VTT（视频‑音频‑文本）。

**📈 对比分析**

与传统 CLIP（可学习或固定温度）以及 NotAGap 方法对比，实验表明模态中心间距离从 0.47 降至约 0.03，CosTP 由 0.34 提升至 0.77，聚类 V‑Measure 提升 7–10 点；检索 Recall@1 变化不大甚至略有提升，证明实例级任务不受影响。

**⚠️ 局限性**

局限性：仅针对对比学习框架，未探索在生成任务或更大规模模态集合上的扩展；对极其复杂的数据分布或高维模态，闭合模态差距的效果可能受限；以及新损失在训练期间需要额外计算开销。

---

## 723. Ribbons from Independence Structure: Hypercontractivity, $Φ$-Mutual Information, and Matrix $Φ$-Entropy

**arXiv ID:** 2601.18516 | [PDF](https://arxiv.org/pdf/2601.18516v1)

**作者:** Chenyu Wang `[一作]` (Chinese University of Hong Kong), Amin Gohari `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2007 | [OpenAlex ID](https://openalex.org/A5015573260)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究在给定部分独立结构下的超收缩锦标和Φ锦标，给出显式内部界、矩阵Φ锦标及其张量化与数据处理性质，并计算双对称二元源的SDPI常数。

**💡 创新点**

首次给出基于超图的凸包内部界、Φ互信息版Zhang–Yeung不等式、矩阵Φ锦标概念以及完整的张量化与数据处理证明。

**🔧 技术方法**

利用凸分析、信息不等式、Φ-熵与矩阵Φ-熵、张量化与数据处理技术。

**📊 数据集**

本文为理论工作，无使用实验数据集。

**📈 对比分析**

与已知的完整独立或完全相关情形对比，给出新的内部界和非平凡点，性能以理论证明为主。

**⚠️ 局限性**

仅给出内部界，外部界仍未完全确定；适用于特定的Φ函数和超图结构，结果在一般情形下可能不最优。

---

## 724. BAIT: Visual-illusion-inspired Privacy Preservation for Mobile Data Visualization

**arXiv ID:** 2601.18497 | [PDF](https://arxiv.org/pdf/2601.18497v1)

**作者:** Sizhe Cheng `[一作]` (Nanyang Technological University), Yong Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 130619 | [OpenAlex ID](https://openalex.org/A5059976286)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种自动化生成移动数据可视化隐私保护方案（BAIT），通过在原始图表上叠加设计良好的干扰图（decoy）来欺骗肩膀偷窥者，同时让使用者在近距离仍能清晰获取信息。

**💡 创新点**

创新点主要有：
- 引入视觉错觉原理，以空间频率差异让不同视距下产生不同感知；
- 采用可视变量（形状、位置、倾斜、尺寸、颜色、空间频率）对干扰图进行自动生成；
- 通过感知驱动的组合优化（VSI 与 MS‑SSIM）在近距/远距两种视距下平衡可读性与隐私保护，突破传统模糊/遮挡方法的可读性折衷。

**🔧 技术方法**

技术实现包括：
- 干扰图生成框架：规则化设计可视变量、Hough 变换、圆形 Hough 变换、Gaussian 滤波、遮罩频率增强；
- 感知模型：基于人眼下采样模拟不同视距，计算 VSI（视觉重要性相似度）和 MS‑SSIM（结构相似度）来量化“可读性”与“误导性”；
- 组合优化：对亮度、色度、卷积核大小、遮罩面积四个参数进行穷举搜索，最大化 α·Gap₁ + β·Gap₂。

**📊 数据集**

数据集：使用自定义随机生成的数据绘制四种常见图表（柱状图、折线图、散点图、饼图）进行实验；实验样本为 32 名受试者（实验一）和 12 名受试者（实验二）。

**📈 对比分析**

与基线方法对比：
- 未处理（UV）和仅遮罩（MS）两种现有方案。
- 结果显示：在近距离（30 cm）受试者识别率 0.96（BAIT）≈0.99（UV），与基线差异不显著；
- 在远距离（90 cm）肩膀偷窥者识别率 0.04（BAIT）远低于 0.99（UV）和 0.28（MS），且低于随机猜测（25 %）。
- 效果显著，p < 0.0001，展示了 BAIT 在隐私保护与可读性之间实现的优异平衡。

**⚠️ 局限性**

局限性：
- 仅覆盖四种常见静态图表，尚未扩展至网络图、信息图或动画可视化；
- 实验设备单一（6.67 inch 手机），缺乏跨设备验证；
- 样本量有限（实验一 32 人，实验二 12 人），难以完全泛化；
- 需要一定的学习成本，虽短暂但存在适应期；
- 对动态/动画可视化的隐私保护尚未实现。

---

## 725. An Audit of Machine Learning Experiments on Software Defect Prediction

**arXiv ID:** 2601.18477 | [PDF](https://arxiv.org/pdf/2601.18477v1)

**作者:** Giuseppe Destefanis `[一作]` (University College London), Mahir Arzoky `[通讯]` (Brunel University London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对2019-2023年软件缺陷预测实验进行系统审核，评估实验设计、统计分析、可复现性与问题；

**💡 创新点**

首次结合可复现性量表对近五年5000+实验进行大样本审计，揭示实践差异与普遍缺陷；

**🔧 技术方法**

采用系统技术审计流程、González‑Barahona & Robles可复现性量表、R统计工具等；

**📊 数据集**

采样101篇实验，涵盖74个会议/期刊，使用多达365个公开数据集（主导为Promise数据集）；

**📈 对比分析**

通过比较指标（F1、AUC、MCC等）、OOS验证方式、统计检验与可复现性分数，发现约45%采用正式统计推断、65%使用OOS验证，平均可复现性评分0.52；

**⚠️ 局限性**

样本仅占全部研究的6.4%，仅覆盖Scopus收录文献，未覆盖灰色文献；可复现性评估工具的等权重假设可能忽略重要细节，且审计仅关注可检出问题，未完全检验内部一致性。

---

## 726. GCFX: Generative Counterfactual Explanations for Deep Graph Models at the Model Level

**arXiv ID:** 2601.18447 | [PDF](https://arxiv.org/pdf/2601.18447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 727. Finite-Aperture Fluid Antenna Array Design: Analysis and Algorithm

**arXiv ID:** 2601.18471 | [PDF](https://arxiv.org/pdf/2601.18471v1)

**作者:** Zhentian Zhang `[一作]` (Southeast University), Yangyang Zhang `[通讯]` (Kuang-Chi Science Limited)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在有限口径条件下对流体天线阵列（FAA）进行设计，推导出闭式CRB以及随机放置下最小间距的概率密度，并基于这些理论给出梯度优化算法；

**💡 创新点**

创新点在于统一把CRB与阵列几何方差关联，得到随机放置的最小间距PDF，进而提出一种兼顾主瓣精度与旁瓣抑制的FAA设计框架；

**🔧 技术方法**

主要采用了统计概率（顺序统计量）、虚拟阵列变换、CRB推导、最大特征值优化以及投影梯度下降（PGD）等技术；

**📊 数据集**

实验数据主要基于Monte‑Carlo仿真，包括ULA、离散MRA（缩放）与连续FAA的阵列位置；

**📈 对比分析**

与传统ULA及离散FAS比较，固定口径下优化后的FAA在CRB上降低约30%，均方误差下降约42.5%，最大特征值也显著减小；

**⚠️ 局限性**

局限性包括算法易受初始化影响、只能获得局部最优、未在真实硬件上验证且仅考虑单源LOS情形，未考虑多径或多目标的复杂场景。

---

## 728. Token-level Collaborative Alignment for LLM-based Generative Recommendation

**arXiv ID:** 2601.18457 | [PDF](https://arxiv.org/pdf/2601.18457v1)

**作者:** Fake Lin `[一作]` (University of Science and Technology of China), Tong Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 3878 | [OpenAlex ID](https://openalex.org/A5025292786)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Token-level Collaborative Alignment for Recommendation（TCA4Rec）框架，将协同过滤（CF）信号与大型语言模型（LLM）生成过程显式对齐。

**💡 创新点**

创新点在于通过协同Tokenizer将item级CF logits映射到token级分布，再用Soft Label Alignment在下一步token预测中融合CF与one‑hot标签，既保留生成性又可调节协同一致性。

**🔧 技术方法**

使用了协同Tokenizer、Soft Label Alignment、软下游next‑token预测损失、LLM微调、token化与概率聚合等技术。

**📊 数据集**

实验采用了Amazon Toys、Sports和Office三大公开数据集。

**📈 对比分析**

与传统CF模型、LLM基线（TallRec、LLaRA、Collm、MSL等）以及量化方法（TIGER、LETTER）进行对比，TCA4Rec在N@k和H@k指标上持续提升，证明方法有效。

**⚠️ 局限性**

局限在于依赖CF模型的质量；当α过大会引入噪声导致性能下降；对非文本特征或极度稀疏场景的适用性仍需进一步研究。

---

## 729. Geneses: Unified Generative Speech Enhancement and Separation

**arXiv ID:** 2601.18456 | [PDF](https://arxiv.org/pdf/2601.18456v1)

**作者:** Kohei Asai `[一作]` (University of Tokyo), Hiroshi Saruwatari `[通讯]` (University of Tokyo)

**通讯引用:** 7628 | [OpenAlex ID](https://openalex.org/A5003814223)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种统一的生成式语音增强与分离框架Geneses，能够同时处理多说话人混合语音中的背景噪声、回声、带宽限制、剪切、编解码失真及包丢失等复杂失真；

**💡 创新点**

创新点在于将潜在流匹配（latent flow matching）与多模态扩散Transformer（MM‑DiT）相结合，通过自监督学习（SSL）提取的条件特征引导流场，直接在VAE潜在空间内恢复干净且分离的语音；

**🔧 技术方法**

核心技术包括：预训练的VAE压缩语音为低维潜在空间；MM‑DiT作为流预测器，融合SSL特征与潜在表示；潜在流匹配（flow matching）训练方式；w2v‑BERT 2.0+LoRA特征提取；以及数值ODE求解实现生成；

**📊 数据集**

使用LibriTTS‑R混声数据，并在此基础上生成两种失真场景：仅背景噪声与复杂失真（包含回声、噪声、带宽限制、剪切、编解码失真、包丢失）；测试集采用DEMAND与MIT环境RIR等真实噪声/回声；

**📈 对比分析**

与Hu等人提出的基于掩码的噪声鲁棒分离方法进行对比，采用参考无关（DNSMOS、NISQA、UTMOSv2、WER）和参考有关（ESTOI、MCD、LSD、SpeechBERTScore、SpkSim）指标。实验显示，在背景噪声条件下Geneses在主观质量指标上略优，WER略低；在复杂失真条件下，Geneses在所有指标上均显著优于传统方法，尤其是WER从5.54降至0.43，表明其在极端失真下具备更高可懂度与自然度；

**⚠️ 局限性**

局限性包括：生成式模型在内容保留上仍易出现“幻觉”，导致WER、LSD、MCD等客观忠实度指标略逊于纯判别方法；此外实验仅验证两说话人混合语音，未来需扩展至多说话人以及真实对话场景；

---

## 730. On the Optimal Message Size in PIR Under Arbitrary Collusion Patterns

**arXiv ID:** 2601.18440 | [PDF](https://arxiv.org/pdf/2601.18440v1)

**作者:** Guru S. Dornadula `[一作]` (International Institute of Information Technology), Prasad Krishnan `[通讯]` (International Institute of Information Technology)

**通讯引用:** 718 | [OpenAlex ID](https://openalex.org/A5103048827)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了在任意协同模式下的私有信息检索（PIR）协议，给出了容量实现方案的可分解性特征，并推导了最优消息尺寸的下界与闭式表达式。

**💡 创新点**

创新点在于：①完整刻画了容量实现的可分解PIR方案的代数结构；②提出基于碰撞模式的“命中数”下界；③在周期连续协同和分块协同等特殊模式下给出了匹配的容量实现方案，首次实现了最优消息尺寸。

**🔧 技术方法**

使用了分数覆盖版本的Shearer不等式、线性规划与集合论分析、可分解与均匀可分解PIR的代数结构等技术来推导下界并构造实现方案。

**📊 数据集**

该工作为理论分析，未使用实验数据集；通过符号参数（N、K、T）和示例（如 N=5 等）进行验证。

**📈 对比分析**

通过与已知的容量实现方案（如 T‑PIR、MDS‑coded PIR）在消息尺寸上的对比，证明在周期连续协同和分块协同场景下所给方案实现了最小消息尺寸并保持容量最优。

**⚠️ 局限性**

局限性：仅讨论复制服务器模型，对编码服务器和非均匀可分解方案的推广有限；并非在所有协同模式下都给出了匹配实现，只在部分特殊模式提供闭式解。

---

## 731. Scaling up Privacy-Preserving ML: A CKKS Implementation of Llama-2-7B

**arXiv ID:** 2601.18511 | [PDF](https://arxiv.org/pdf/2601.18511v1)

**作者:** Jaiyoung Park `[一作]` (Graduate School of Convergence Science and Technology, Seoul National University), Damien Stehlé `[通讯]` (CryptoLab, Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对大型语言模型（LLM）的加密推理方案，支持数千个输入令牌，其中仅最后128个被加密，并实现了完整的推理流程。

**💡 创新点**

创新点包括：①引入“非平衡分块预填充”框架，将公开和私有输入分别处理；②针对PC‑MM、Softmax等关键算子设计了低乘深度、低引导量的 FHE 算法；③通过预置“错误令牌”和旋转技术减小非线性层的取值范围，降低多项式逼近度；④在 CKKS 上实现高效的矩阵乘法和稀疏多项式求值。

**🔧 技术方法**

使用的技术有：CKKS FHE、全同态矩阵乘法（PCMM/CCMM）、稀疏多项式评估、Softmax 多项式逼近、Bootstrapping 最小化、Rotations、RoPE、RMSNorm、SwiGLU、NCCL 并行、CUDA、C++/Python 绑定等。

**📊 数据集**

数据集：使用 Llama‑2‑7B 与 Llama‑3‑8B 公开模型，主要在标准语言建模/生成任务上进行推理评估（如摘要、文本生成），并未对特定自定义数据集进行说明。

**📈 对比分析**

对比方法：与 Jayashankar 等人（8 H100 GPU 上 128 令牌 295s）以及 Moon 等人（1 A100 GPU 上 128 令牌 602s）进行比较。本文在 8 个 RTX‑4090 GPU 上完成 4096 令牌推理，摘要 85s、生成 33s，显示出显著的性能提升。

**⚠️ 局限性**

局限性：①仅在部分令牌加密的场景下有效；②对大型模型仍依赖较多 GPU 与显存；③某些关键操作（如 CCMM）占比高，仍有加速空间；④未覆盖采样、推理多样性等后续任务；⑤实验仅基于 CKKS，缺乏对其他 FHE 基础方案的评估。

---

## 732. Using Large Language Models to Construct Virtual Top Managers: A Method for Organizational Research

**arXiv ID:** 2601.18512 | [PDF](https://arxiv.org/pdf/2601.18512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 733. Evaluating Morphological Plausibility of Subword Tokenization via Statistical Alignment with Morpho-Syntactic Features

**arXiv ID:** 2601.18536 | [PDF](https://arxiv.org/pdf/2601.18536v1)

**作者:** Abishek Stephen `[一作]` (Charles University), Jindřich Libovický `[通讯]` (Charles University)

**通讯引用:** 1679 | [OpenAlex ID](https://openalex.org/A5061045500)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于IBM Model 1的子词分割形态学合理性评估指标，可在缺少金标准分词的情况下通过对齐子词与形态句法特征来衡量子词分割质量。

**💡 创新点**

创新点在于不依赖于传统的形态学边界标注，而是利用多语言通用的形态句法特征（如UniMorph）与子词进行概率对齐，从而实现跨语言、跨词形系统的通用评估。

**🔧 技术方法**

使用的技术包括IBM Model 1的EM训练对齐算法、子词分割器（BPE、WordPiece、Unigram）、多种聚合函数（Sum、Mean、Max 等）以及Spearman相关性分析。

**📊 数据集**

实验数据集涵盖10种语言的 Universal Segmentations 及 UniMorph 形态句法标签（Armenian, Finnish, Kannada, English, German, Dutch, Czech, Serbo‑Croatian, Hungarian, Slovak），并使用 CC100 语料训练子词分割器。

**📈 对比分析**

通过将评估指标与传统形态边界精度/召回率做 Spearman 相关性对比，结果显示与召回率的相关性普遍高于0.7（部分语言达到0.94+），与精度的相关性则更为不稳定，整体表现证明该指标在多语言环境中能有效捕捉形态学合理性。

**⚠️ 局限性**

局限性包括：仍以现有金标准分割为参照，若金标准本身存在误差会影响评估；对形态特征与词形之间的对应关系依赖阈值和数据稀疏度；对极其形态复杂或资源稀缺语言的泛化能力尚待进一步验证。

---

## 734. Scalable Transit Delay Prediction at City Scale: A Systematic Approach with Multi-Resolution Feature Engineering and Deep Learning

**arXiv ID:** 2601.18521 | [PDF](https://arxiv.org/pdf/2601.18521v1)

**作者:** Emna Boudabbous `[一作]` (École de technologie supérieure), Omar Alam `[通讯]` (Trent University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了面向城市公交网络的端到端延迟预测流水线，涵盖多分辨率特征工程、Hybrid H3+拓扑聚类、降维、全局LSTM模型及推理。

**💡 创新点**

创新点包括：系统化多分辨率H3特征生成、解决“巨型聚类”问题的Hybrid H3+拓扑聚类、利用Adaptive PCA压缩特征并证明简单LSTM优于Transformer。

**🔧 技术方法**

使用技术包括：Uber H3空间索引、聚类算法（Ward、Jaccard）、Adaptive PCA、LSTM、XGBoost、PatchTST、Autoformer、Spark分布式计算、GTFS‑RT实时数据处理。

**📊 数据集**

使用的数据集为蒙特利尔交通局（STM）6个月的公交实时位置、静态GTFS、天气数据。

**📈 对比分析**

通过walk‑forward时间交叉验证对五种模型进行对比，结果显示全局LSTM在节省参数275倍的同时，延迟预测RMSE提升18‑52%，Trip级别RMSE仅1.85分钟，远优于Transformer。

**⚠️ 局限性**

局限性在于仅在单一城市公交网络验证，天气数据分辨率低、特殊事件手动识别、缺乏乘客计数数据、未覆盖多模态交通或较小城市。

---

## 735. LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models

**arXiv ID:** 2601.18513 | [PDF](https://arxiv.org/pdf/2601.18513v1)

**作者:** Kai Hu `[一作]` (Carnegie Mellon University), Matt Fredrikson `[通讯]` (Carnegie Mellon University)

**通讯引用:** 10168 | [OpenAlex ID](https://openalex.org/A5057424614)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种可扩展的 1‑Lipschitz 神经网络架构 LipNeXt，用于在保证确定性鲁棒性的前提下，支持亿级参数规模的训练与推理。

**💡 创新点**

创新点包括：①在正交矩阵上实现无约束的流形优化，使用 FastExp 近似与周期极化重投提升效率；②提出卷积自由的空间 Shift 模块，理论证明其为唯一等距深度卷积；③结合 β‑Abs 非线性，提供高效 1‑Lipschitz 激活。

**🔧 技术方法**

采用的技术包括：流形优化（正交流形、指数映射近似）、FastExp、极化重投、Lookahead 与 Adam 结合、空间 Shift、β‑Abs、L2 空间池化、bfloat16 低精度训练。

**📊 数据集**

使用的数据集为 CIFAR‑10、CIFAR‑100、Tiny‑ImageNet 与 ImageNet（完整与 400 类子集）。

**📈 对比分析**

与现有最优 Lipschitz 与随机平滑方法进行对照，评估指标为 clean accuracy 与 CRA（ε=36/255 或 ε=1）；LipNeXt 在 CIFAR、Tiny‑ImageNet 及 ImageNet 上实现 CRA 提升 3–8%、clean accuracy 提升，并能在 1B 参数模型上保持 bfloat16 训练吞吐。

**⚠️ 局限性**

局限性：易出现鲁棒过拟合，需使用合成数据缓解；在极大模型或多任务学习中训练时间仍显高；正交矩阵乘法的计算成本在某些硬件上仍不完全可扩展。

---

## 736. DV-VLN: Dual Verification for Reliable LLM-Based Vision-and-Language Navigation

**arXiv ID:** 2601.18492 | [PDF](https://arxiv.org/pdf/2601.18492v1)

**作者:** Zijun Li `[一作]` (Zhejiang Normal University), Shoujun Zhou `[通讯]` (Shenzhen Institutes of Advanced Technology)

**通讯引用:** 2008 | [OpenAlex ID](https://openalex.org/A5046902140)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DV-VLN 框架，利用 LLM 生成结构化的预测–视图匹配–动作链式思维，并在推理时通过双重验证（True–False 和 Masked‑Entity）对多样化候选动作进行重新排序。

**💡 创新点**

创新点在于：① 采用 generate‑then‑verify 思路，将多样化候选动作与双重验证相结合提升导航鲁棒性；② 设计了可训练的结构化链式思维格式；③ 在不需额外监督的前提下实现了推理时的双重验证机制。

**🔧 技术方法**

技术手段包括：LLaMA‑2 通过参数高效领域适配；BLIP 生成视觉文本描述；CLIP 进行实体匹配；采样解码生成多候选；True‑False 与 Masked‑Entity 两类自检进行验证。

**📊 数据集**

使用 R2R、RxR（英语子集）和 REVERIE 三大 VLN 基准数据集进行训练与评估。

**📈 对比分析**

与跨模态基线、提示式 LLM 代理及可训练语言仅代理对比，DV‑VLN 在 R2R 未见环境上 SR 52%/SPL 45%，在 RxR 和 REVERIE 上也明显优于语言仅基线，并能与部分跨模态系统相当。

**⚠️ 局限性**

局限性包括：视觉‑文本转换可能丢失细节信息；双重验证在推理时增加计算开销；对文本描述的依赖限制了对复杂视觉信息的充分利用。

---

## 737. Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States

**arXiv ID:** 2601.18479 | [PDF](https://arxiv.org/pdf/2601.18479v1)

**作者:** Kyoleen Kwak `[一作]` (Kyung Hee University), Hyoseok Hwang `[通讯]` (Kyung Hee University)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5018395387)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于转移诱导相似状态的损失方法ASAP，利用空间一致性和时间二阶差分惩罚来抑制深度强化学习中的高频动作振荡，同时保持或提升策略性能。

**💡 创新点**

创新点包括：① 定义相似状态为同一前一状态下的真实转移分布，理论证明其形成有界邻域；② 将空间正则化与第二阶时间差分惩罚结合，形成新的损失结构；③ 仅在训练损失中加入，保持网络结构不变。

**🔧 技术方法**

使用Lipschitz连续性分析、基于转移分布的空间损失、预测头预测前一状态下动作的机制，以及Grad‑CAPS的二阶差分惩罚；在PPO和SAC框架下训练，并在Gymnasium与Isaac‑Lab仿真环境中评估。

**📊 数据集**

实验数据集：Gymnasium基准（LunarLander、Pendulum、Reacher、Ant、Hopper、Walker等）和Isaac‑Lab机器人任务（Franka reach、Lift‑Cube、Repose‑Cube、Anymal‑Velocity）以及SAC+LipsNet的实验环境。

**📈 对比分析**

与原始SAC/PPO、CAPS、L2C2、Grad‑CAPS及LipsNet+CAPS等方法对比；在Gymnasium环境中ASAP在大部分任务获得最高累计回报或最低smoothness评分，在Isaac‑Lab中亦实现更低smoothness且回报保持或提升；相较于其他方法，ASAP在降低高频振荡方面显著，例如Hopper环境下降约89%。

**⚠️ 局限性**

局限性：在噪声水平较高的环境中，相似状态邻域可能过大，削弱局部Lipschitz约束效果；需通过调整空间损失权重来平衡；在某些任务（如Lift‑Cube）中，振荡抑制可能限制探索导致回报方差增大。

---

## 738. Latent Knowledge as a Predictor of Fact Acquisition in Fine-Tuned Large Language Models

**arXiv ID:** 2601.18468 | [PDF](https://arxiv.org/pdf/2601.18468v1)

**作者:** Daniel B. Hier `[一作]` (University of Illinois at Chicago), Tayo Obafemi-Ajayi `[通讯]` (Missouri State University)

**通讯引用:** 735 | [OpenAlex ID](https://openalex.org/A5040578359)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文在大型语言模型预训练基础上，利用生物医学本体（HPO和GO）的术语-标识符对，对模型进行监督微调，并通过生存分析量化事实获取、泛化与退化的速率。

**💡 创新点**

创新点在于将事实学习视为时间至事件过程，使用生存分析揭示预训练隐含知识（latent knowledge）对学习速率和泛化的决定性影响，并阐明训练对已知事实的保护机制。

**🔧 技术方法**

使用的技术包括Llama‑3.1‑8B‑Instruct模型、LoRA参数高效微调、基于温度1.0的随机采样检测隐含知识，以及Cox比例风险模型与Kaplan‑Meier曲线进行速率分析。

**📊 数据集**

数据集为从Human Phenotype Ontology抽取的800个术语-标识符对（全部用于训练）和Gene Ontology抽取的800对（400训练、400留存）的频率分层样本，且对每对计算PubMed Central中的术语/标识符频数与注释计数。

**📈 对比分析**

与传统的二元准确率评估相比，本文采用速率指标显示HPO事实学习在存在隐含知识时平均10.8个epoch即可完成，学习速率最高时为每epoch 21.1%，而未见隐含知识时需20个epoch；对GO未见事实的泛化率仅5.8%，但隐含知识大幅提升。

**⚠️ 局限性**

限制包括仅考虑术语-标识符映射的事实形式、固定的微调超参与20个epoch限制、未探究内部机制（如参数更新或嵌入重排）以及方法对连续/多维技能学习的适用性不足。

---

## 739. Rethinking AI in the age of climate collapse: Ethics, power, and responsibility

**arXiv ID:** 2601.18462 | [PDF](https://arxiv.org/pdf/2601.18462v1)

**作者:** Julio Vega `[一作]` `[通讯]` (Rey Juan Carlos University), Julio Vega (Rey Juan Carlos University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对人工智能在气候危机中的作用进行跨学科评估，阐述其潜在收益与风险，并提出可持续治理建议。

**💡 创新点**

将环境伦理、哲学、法律与技术视角融合，强调AI治理需以正义与可持续为核心，并提出多维度政策框架。

**🔧 技术方法**

主要采用理论分析、文献综述与案例讨论，无实验技术。

**📊 数据集**

未使用具体数据集，参考了相关报告、政策文件与实例案例。

**📈 对比分析**

未进行实验比较，文章以概念性讨论与现有研究评述为主。

**⚠️ 局限性**

缺乏量化评估与实证验证，实际可行性、跨国监管差异以及技术实施细节等方面仍有局限。

---

## 740. Scaling Behaviors of Evolutionary Algorithms on GPUs: When Does Parallelism Pay Off?

**arXiv ID:** 2601.18446 | [PDF](https://arxiv.org/pdf/2601.18446v1)

**作者:** Xinmeng Yu `[一作]`, Kay Chen Tan `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对16种主流进化算法在GPU上进行系统评测，比较CPU/GPU在不同维度、种群规模下的效率与效果。

**💡 创新点**

揭示GPU并行不仅仅是加速，还会改变算法行为、评估准则与种群规模效应，提出基于时间而非FEs的评测方法，并识别不同算法在GPU上的可扩展性区间。

**🔧 技术方法**

使用EvoX框架实现GPU端向量化运算，采用CUDA/多GPU，评测CPU与GPU在同一配置下的运行时间、FEs吞吐和解质量指标（IGD、HV）。

**📊 数据集**

基于30个数值优化基准（Ackley、Griewank等）以及Brax物理仿真神经进化任务，涵盖单目标和多目标问题。

**📈 对比分析**

通过固定时间30秒/600秒对CPU和GPU进行比较，计算速度提升（speed‑up）、NFE吞吐与解质量，结果显示GPU在大维度/大种群时可获得数倍加速，但不同算法加速幅度差异显著，某些算法甚至出现速度下降。

**⚠️ 局限性**

实验仍受限于单一GPU型号与固定算法实现，未深入探讨动态负载平衡与多GPU协同；且对不同问题的泛化仍需进一步验证。

---

## 741. Properties of calculus in r-Complexity 2025

**arXiv ID:** 2601.18437 | [PDF](https://arxiv.org/pdf/2601.18437v1)

**作者:** Rares Folea `[一作]` (National University of Science and Technology Politehnica Bucharest), Emil Slusanschi `[通讯]` (National University of Science and Technology Politehnica Bucharest)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文通过定义并证明 r-Complexity 体系（Big r-Theta、Big r-O、Big r-Omega 等）中的若干基本性质（可反射、传递、对称、投影、加法运算）来扩展传统的 Bachmann‑Landau 记号，并通过示例展示 r-Complexity 在区分常数系数不同、但属于同一 Theta 类算法上的优势。该工作还说明了如何利用回归拟合性能指标（时间、内存）得到 r-Complexity 函数，并把其应用于代码嵌入与矩阵乘法算法比较。

**💡 创新点**

创新点在于：①提出了 r-Complexity 记号的完整算术与代数性质；②将这些性质与传统记号进行对比，证明 r-Complexity 能够在常数因素不同的情况下区分算法；③给出一种基于有限输入指标的回归拟合方法来估算 r-Complexity 函数；④通过 Strassen 算法与普通矩阵乘法的实际测评，展示 r-Complexity 在实际性能预测上的准确性。

**🔧 技术方法**

主要技术包括：符号计算与极限分析（证明各种性质），回归拟合（多项式拟合时间/内存指标），极限与比值运算（定义 Big r‑Θ、Big r‑O、Big r‑Ω 等），以及对比实验（算法运行时间与内存测量）。

**📊 数据集**

本文示例使用了人工生成的性能数据集：对某算法在输入规模 10、20、30 时测得的时间（秒）与内存（kB）数据，以及通过回归得到的多项式时间/内存模型。实际比较中还引用了公开的矩阵乘法基准（包括 Strassen 与普通算法）以验证 r-Complexity 的预测能力。

**📈 对比分析**

比较方法是：①将实验得到的时间/内存曲线拟合成 r-Complexity 函数；②使用极限比值判断两函数落在哪一类（Θ、O、Ω）；③对 Strassen 与普通矩阵乘法在不同输入规模下的实际耗时进行对比，说明虽然 Strassen 在理论上更优，但在 r-Complexity 视角下在 25 M 以内更慢。性能表现：在常数差异很大的情况下，r-Complexity 能准确区分并给出更细粒度的复杂度估计。

**⚠️ 局限性**

局限性包括：①目前 r-Complexity 仍处于理论阶段，缺乏广泛的实测案例；②需要足够的实验数据来训练回归模型，数据量不足时估计误差可能较大；③与传统记号相比，定义与证明更为繁琐，实用性尚未得到充分验证；④对于某些算法的极端规模或特殊内存访问模式，r-Complexity 的预测可能不如经验公式可靠。

---

## 742. The Quantum Cliff: A Critical Proton Tunneling Threshold Determines Clinical Severity in RPE65-Mediated Retinal Disease

**arXiv ID:** 2601.18435 | [PDF](https://arxiv.org/pdf/2601.18435v1)

**作者:** Biraja Ghoshal `[一作]` (University College London), Biraja Ghoshal `[通讯]` (University College London)

**通讯引用:** 520 | [OpenAlex ID](https://openalex.org/A5050505627)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

结合AlphaFold结构预测与VQE量子模拟，构建RPE65突变的结构-表型预测管线，揭示量子隧道阈值效应导致的“Quantum Cliff”。

**💡 创新点**

发现RPE65活性对亚Å级几何变化呈指数级衰减，提出RQAS指标并将量子隧道机制与临床表型关联。

**🔧 技术方法**

采用AlphaFold2、VQE量子算法、WKB隧道与热激活理论计算能面，结合GPU加速的量子模拟。

**📊 数据集**

以AlphaFold2预测的RPE65结构为基准，整合临床突变表型实验活性数据作为验证。

**📈 对比分析**

与实验活性通过R²=0.93对比，能准确排名突变严重程度；虽然绝对值低估，但相对顺序鲁棒。

**⚠️ 局限性**

静态势能面忽略了热动力学与蛋白动力学的助隧道效应，导致对绝对活性预测偏低；模型仅适用于量子隧道敏感酶。

---

## 743. From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for Reinforcement Learning of Open-ended Generation

**arXiv ID:** 2601.18533 | [PDF](https://arxiv.org/pdf/2601.18533v1)

**作者:** Yuxin Jiang `[一作]` (Huawei Technologies Co), Lifeng Shang `[通讯]` (Huawei Technologies Co)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将可验证奖励扩展到开放式生成任务的框架 RLVRR，通过参考答案生成可验证的内容和风格信号来指导强化学习。

**💡 创新点**

创新点在于将奖励链拆分为可验证的内容和风格两维度，并用 LLM 自动生成可验证的关键词和 Python 检验函数，既保留 RL 的探索性，又兼具 SFT 的低成本、高可靠性。

**🔧 技术方法**

采用基于规则的可验证器、最长公共子序列（LCS）对关键词对齐、LLM 自动生成的可验证 Python 检测函数、GRPO 优化器以及多参考的奖励聚合。

**📊 数据集**

使用 Qwen2.5、Llama3.1 等大模型，训练数据来自 OpenHermes、Magpie、WebR 等公开指令调优数据集，结合 GPT‑4o‑mini 生成的高质量参考答案。

**📈 对比分析**

与 SFT（10×数据）、RLHF（RM、GRM、RLPR）、DPO、BLEU 等方法对比，RLVRR 在 10+ 开放式基准、数学、推理、编程任务上均取得显著提升（例如在 Qwen2.5‑3B 上平均提升 4–6 分，超过 DPO、RM 等），同时保持输出多样性。

**⚠️ 局限性**

局限性包括：需要足够多且高质量的参考答案；对极其主观或无可验证标准的任务效果可能有限；虽然成本低于 RLHF，但仍需 LLM 调用生成参考与可验证器，且对语言模型的依赖性较高。

---

## 744. From Cold Start to Active Learning: Embedding-Based Scan Selection for Medical Image Segmentation

**arXiv ID:** 2601.18532 | [PDF](https://arxiv.org/pdf/2601.18532v1)

**作者:** Devon Levy `[一作]` (University of Haifa), Bella Specktor-Fadida `[通讯]` (University of Haifa)

**通讯引用:** 151 | [OpenAlex ID](https://openalex.org/A5005409145)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一个两阶段的主动学习框架，先用基础模型嵌入+聚类进行冷启动样本选择，再通过不确定性与空间多样性的结合进行后续主动采样，用于医学图像分割任务。

**💡 创新点**

创新点在于自动确定聚类数并采用比例最远点采样的冷启动策略，以及将图像级熵与特征空间距离融合的主动学习采样方法，同时提供可视化的特征空间分布。

**🔧 技术方法**

使用了RadImageNet预训练的ResNet‑50作为特征提取器、t‑SNE降维、k‑means（基于轮廓系数自动选k）与medoid/最远点采样、基于像素熵的图像级不确定性、α 参数权衡的混合采样；实现上采用MONAI框架、Attention‑U‑Net、Dice+交叉熵损失。

**📊 数据集**

实验数据集包括 2D 脑 MRI（SynthStrip），胸部 X‑ray（Montgomery）以及包含心脏结构的胸部 X‑ray（CheXmask‑300）。

**📈 对比分析**

与随机采样、k‑means‑to‑budget、单纯熵采样和单纯多样性采样比较，冷启动聚类在 Dice 上提升 0.01–0.03、Hausdorff 距离下降 5–10 mm，并且方差更小；主动学习阶段混合采样在 Dice、HD95 上均优于单一指标方法。

**⚠️ 局限性**

局限性包括仅在 2D 任务上验证、主动学习中 α 固定为 0.3、对基础模型特征质量高度依赖、未对 3D 场景进行充分评估以及未对超参数进行系统搜索。

---

## 745. From Human Labels to Literature: Semi-Supervised Learning of NMR Chemical Shifts at Scale

**arXiv ID:** 2601.18524 | [PDF](https://arxiv.org/pdf/2601.18524v1)

**作者:** Yongqi Jin `[一作]` (Peking University), Weinan E `[通讯]` (Peking University)

**通讯引用:** 34962 | [OpenAlex ID](https://openalex.org/A5071854504)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了半监督NMR化学位移预测框架，利用已标注和未标注文献谱数据。

**💡 创新点**

创新点是将未标注谱转化为排列不变的集合监督，采用排序损失实现大规模半监督学习，并首次在规模上整合溶剂信息。

**🔧 技术方法**

采用SE(3)-equivariant Transformer（NMRNet）与排序损失、溶剂嵌入的多种注入策略。

**📊 数据集**

使用ShiftDB‑Lit（百万级文献提取谱）与NMRShiftDB2（手工标注谱）数据集。

**📈 对比分析**

与传统HOSE、GNN等方法及NMRNet在NMRShiftDB2和ShiftDB‑Lit上对比，MAE下降13–60%，显示显著性能提升。

**⚠️ 局限性**

主要限制在于弱监督单独使用易崩溃，需要足够高质量标注；溶剂分类仍粗略，且对极少见溶剂信息依赖有限。

---

## 746. Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates

**arXiv ID:** 2601.18510 | [PDF](https://arxiv.org/pdf/2601.18510v1)

**作者:** Yibo Li `[一作]` (National University of Singapore), Bryan Hooi `[通讯]` (National University of Singapore)

**通讯引用:** 5314 | [OpenAlex ID](https://openalex.org/A5065675832)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种在测试时无需梯度更新、通过检索经验记忆计算优势并直接调整LLM logits 的 Just-In-Time Reinforcement Learning（JitRL）框架，实现持续自适应。

**💡 创新点**

创新点包括：①实现无参数更新的即时策略优化；②将优势估计视作非参数记忆检索；③理论证明 logit 加法即 KL 约束下的闭式最优；④显著降低计算与成本。

**🔧 技术方法**

使用了邻近检索、优势估计、KL‑约束策略优化、logit 加权更新以及基于 LLM 的测试时自适应技术。

**📊 数据集**

实验数据集包括 WebArena（网页导航）和 Jericho（文本冒险游戏）等基准。

**📈 对比分析**

与训练自由基线（Static、Memory、Reflexion、AWM、EvoTest）以及权重更新方法（WebRL、GRPO）对比，JitRL 在 WebArena 与 Jericho 上均取得 SOTA 结果，且成本低于传统训练 30 倍以上。

**⚠️ 局限性**

局限性：依赖记忆检索的相似性，稀疏奖励或极端多样化任务时优势估计不稳；对极大记忆容量与跨领域迁移的可扩展性尚未充分验证；对仅暴露 logits 的黑盒 LLM 仍存在访问限制。

---

## 747. DEEPMED: Building a Medical DeepResearch Agent via Multi-hop Med-Search Data and Turn-Controlled Agentic Training & Inference

**arXiv ID:** 2601.18496 | [PDF](https://arxiv.org/pdf/2601.18496v1)

**作者:** Zihan wang `[一作]`, Xiaozhong Ji `[通讯]` (ByteDance)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出DeepMed，一种专为医疗问答设计的DeepResearch模型，结合多跳医学搜索数据合成、Agentic SFT/强化学习、难度感知回合惩罚以及Over‑Evidence监控，以提升医学推理质量和可靠性。

**💡 创新点**

创新点在于：① 针对医学任务特征和工具扩展差距，构造多跳医学链并生成实体模糊的医药搜索QA；② 在强化学习阶段引入难度感知回合惩罚，抑制过度检索；③ 添加Over‑Evidence监控，防止模型陷入重复验证的“过证据”状态；④ 通过少量训练数据实现与大模型相比更高的平均提升。

**🔧 技术方法**

技术包括：DeepResearch+ReAct框架、Web搜索工具（检索+内容摘要）、Agentic SFT、GRPO强化学习、实体模糊化、多跳链生成、难度感知回合惩罚、Over‑Evidence监控、LLM评判证据完整性。

**📊 数据集**

使用的基准数据集有七个医学问答集（MedXpert、MedMCQA、MedQA‑USMLE、HLE‑Med、PubMedQA、MMLU‑Pro‑Med、CMExam），以及自行合成的5,437条多跳医学搜索QA和6,304条难度过滤的医学诊断QA。

**📈 对比分析**

在13个基线（Awesome General、Medical Reasoning、DeepResearch等）上对比实验，DeepMed‑14B在大多数基准上超过更大参数或更多训练数据的模型，平均提升约9.8%（SFT）至13.9%（RL），并在多数任务接近或达到state‑of‑the‑art水平。

**⚠️ 局限性**

局限性包括：agentic训练与工具调用需要高昂计算和工程成本；未在更大模型或更大训练语料上系统验证；仅在公开基准评估，缺乏真实临床工作流验证；工具依赖公开网络，信息可用性和质量随时间波动。

---

## 748. Demographic Probing of Large Language Models Lacks Construct Validity

**arXiv ID:** 2601.18486 | [PDF](https://arxiv.org/pdf/2601.18486v1)

**作者:** Manuel Tonneau `[一作]` (World Bank), Valentin Hofmann `[通讯]` (Allen Institute for AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过系统比较多种种族和性别提示（姓名、方言、对话历史、显式属性），评估大语言模型在医疗、薪资和法律三类一人求助对话中的构造效度，检验不同提示对模型行为的一致性与差异。

**💡 创新点**

首次量化不同种族/性别提示的收敛效度与判别效度，揭示传统单一提示方法缺乏构造效度，并归因于提示与群体关联强度差异及语言混杂因素，提出使用多提示、控制混杂、采用生态有效提示的改进方案。

**🔧 技术方法**

采用构造效度评估方法（Pearson相关、回归分析）、数据增强技术、三款LLM（LLaMA‑3.1 8B、OLMo2‑7B、GPT‑5.2）进行实验，结合统计检验与置信区间可视化。

**📊 数据集**

使用美国第一人称求助对话库（医疗、薪资、法律），通过名称列表（三源）、AAVE翻译、CAD/PRISM对话历史、显式属性扩展，并利用GPT‑5进行数据增强生成约4,440个医疗、5,000个薪资、5,000个法律提示。

**📈 对比分析**

方法：对每个提示计算与无提示基准的偏差向量，计算不同提示间以及同提示下不同群体的Pearson相关，评估收敛与判别；计算Black/White结果比值并绘制置信区间。结果显示：收敛性仅部分一致，判别性弱，群组差异因提示而大幅波动；模型表现受提示与语言特征的共同作用影响。

**⚠️ 局限性**

限制：提示-群体关联仅通过模型预测评估，不能揭示内部机制；GPT‑5.2仅通过API测试；输出受限于二元/数字格式，忽略更丰富的交互特征；仅关注美国一人称求助情境，可能不适用于其他任务或地区；性别分析受限于与种族交叉的二元性别设定。

---

## 749. Funny or Persuasive, but Not Both: Evaluating Fine-Grained Multi-Concept Control in LLMs

**arXiv ID:** 2601.18483 | [PDF](https://arxiv.org/pdf/2601.18483v1)

**作者:** Arya Labroo `[一作]` (University of Cambridge), Mario Fritz `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套评估LLM细粒度多概念控制的框架，并在三种文本生成任务中使用该框架评测了多种语言风格概念的控制效果。

**💡 创新点**

创新点在于：①系统化的多概念可控性评估方法；②通过 Spearman 相关系数量化单概念与双概念控制的差异；③揭示即便概念理论上可分离，LLM 在双概念情境下也会出现显著的干扰。

**🔧 技术方法**

使用的技术包括：结构化提示模板、基于 GPT‑4.1 的判别式评审模型（pairwise 比较）、Spearman 相关性与 Fisher z‑变换进行统计分析。

**📊 数据集**

数据集：Argument（Persuasion 数据集）、Story（ROCStories）、Structured Text（GEM），每个任务各 75 条测试样本；概念集为 humor、persuasiveness、clarity、politeness、assertiveness、formality。

**📈 对比分析**

对比方式：单概念控制 vs 双概念控制（固定次概念水平与随机次概念水平），使用三种中等规模模型（Llama‑11B、Gemma‑12B、Qwen‑14B）。结果显示单概念控制的 Spearman 相关性普遍较高（0.8–1.0 之间），而加入第二概念后相关性显著下降，尤其是 humor‑persuasiveness 对结构化文本生成的影响最大。

**⚠️ 局限性**

局限性包括：①仅评估了三对概念；②只对 3B–14B 参数量的模型进行实验，未探讨更大模型的表现；③仅使用提示式控制，未测试表示工程或 logit‑bias 等其它方法。

---

## 750. 3DGesPolicy: Phoneme-Aware Holistic Co-Speech Gesture Generation Based on Action Control

**arXiv ID:** 2601.18451 | [PDF](https://arxiv.org/pdf/2601.18451v1)

**作者:** Xuanmeng Sha `[一作]` (Osaka University), Yuki Uranishi `[通讯]` (Osaka University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种3DGesPolicy框架，将全身与面部共语音手势生成转化为动作控制的连续轨迹生成任务。

**💡 创新点**

创新点包括使用动作差分表示的轨迹控制、将手势生成视为连续动作决策问题，并通过Gesture-Audio-Phoneme（GAP）融合实现逐音素级语义与运动的精准对齐。

**🔧 技术方法**

主要技术包括3D扩散策略（DDIM）、多模态Transformer编码器、HuBERT和XPhoneBERT音频/音素特征提取、以及跨模态注意力的GAP融合模块。

**📊 数据集**

使用BEAT2数据集（约76小时高质量3D动作捕捉、同步音频与文本），进行训练与评估。

**📈 对比分析**

通过与DiffSHEG、EMAGE、MambaTalk等基线在FGD、DIV、MSE、LVD等定量指标以及用户评测（自然度、同步度、情感表达）进行对比，3DGesPolicy在所有指标上均显著优于对手。

**⚠️ 局限性**

局限性在于手部细节与指尖动作的精细控制仍不够完善，需要进一步改进手部动作的细粒度表达。

---

## 751. KeyMemRT Compiler and Runtime: Unlocking Memory-Scalable FHE

**arXiv ID:** 2601.18445 | [PDF](https://arxiv.org/pdf/2601.18445v1)

**作者:** Eymen Ünay `[一作]` (University of Edinburgh), Jackson Woodruff `[通讯]` (University of Edinburgh)

**通讯引用:** 130 | [OpenAlex ID](https://openalex.org/A5072181658)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 KeyMemRT 编译器与运行时，能够细粒度管理 FHE 旋转密钥，从而显著降低内存占用。

**💡 创新点**

创新点在于提出新的 MLIR 方言实现密钥生命周期分析、自动化细粒度密钥管理、引入引导键（bootstrap key）管理以及预取提示，首次实现全自动化的细粒度密钥与引导键管理。

**🔧 技术方法**

技术包括 MLIR、数据流分析、BSGS 旋转链化、旋转键提升、公共子表达式消除、低内存与平衡两种运行时模式，并在编译器与运行时之间使用键预取提示实现异步加载。

**📊 数据集**

使用多种机器学习模型（MLP、LoLa、ResNet‑1/8/10/18/34/44/56/50、LeNet、AlexNet）作为实验数据集进行评估。

**📈 对比分析**

与现有 ANT‑ACE 与 Fhelipe 对比，KeyMemRT 在内存上平均减少约 1.5–2.6 倍，并在速度上比 Fhelipe 提升 1.5–2.5 倍；低内存模式下内存更低，平衡模式下速度更快，整体表现优于两者。

**⚠️ 局限性**

局限性包括：需要将密钥序列化到磁盘；仅在 CKKS 下验证；极大密钥规模仍有生成/加载开销；对小程序提升有限；运行时预取受硬件资源限制。

---

## 752. SG-CADVLM: A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation

**arXiv ID:** 2601.18442 | [PDF](https://arxiv.org/pdf/2601.18442v1)

**作者:** Hongyi Zhao `[一作]` (Southeast University), Ziyuan Pu `[通讯]` (Southeast University)

**通讯引用:** 3141 | [OpenAlex ID](https://openalex.org/A5047891218)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SG-CADVLM 框架，利用上下文感知解码和多模态输入，直接从事故报告与道路图生成可在 SUMO、CARLA 等平台上执行的安全关键场景。

**💡 创新点**

创新点：① 上下文感知解码（CAD）显著抑制 VLM 幻觉；② 多模态融合同时生成道路几何与车辆轨迹；③ 集成 Retrieval‑Augmented Generation（RAG）提升 API 合规性与生成成功率。

**🔧 技术方法**

技术手段：大型视觉语言模型（Qwen VL 72B / Llama VL 90B）、CAD 机制、RAG 检索、SUMO、CARLA、Esmini、OpenSCENARIO 等仿真与可视化工具。

**📊 数据集**

数据集：32份 NHTSA 事故报告及对应道路图，OpenStreetMap 公开道路数据库用于检索；实验中使用手工构建的真值道路网络进行评估。

**📈 对比分析**

比较方法：与 Omnitester、Text2Scenario、Template‑Based Generation (TBG)、OSM‑Direct 四个基线在几何重建指标（ICE、LCE、CE）、安全关键指标（碰撞概率、TTC、PET、交互强度）以及生成成功率（GSR）进行对比。SG‑CADVLM 在几乎所有指标上均显著优于基线，关键风险场景生成率提升至 84.4%（比基线 12.5% 提升 469%），碰撞概率提升 30 倍，TTC、PET 等时间安全指标大幅下降。

**⚠️ 局限性**

局限性：生成过程较慢，需要多轮 CAD 迭代；部分 LLM（如 ChatGPT、Claude、Gemini）缺少 logits 导致 CAD 无法应用；未涵盖动态天气、传感器多模态输入，未在更大规模、多地区数据上进行验证。

---

## 753. Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmark

**arXiv ID:** 2601.18509 | [PDF](https://arxiv.org/pdf/2601.18509v1)

**作者:** Andro Sabashvili `[一作]` `[通讯]`, Andro Sabashvili

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对多种时间序列分布无关预测区间方法进行实证评估，提出并验证了适用于ARIMA基线的多步拆分式一致性预测（MSCP）方案；

**💡 创新点**

创新点在于将标准拆分一致性预测延伸至多步预测，针对每个预测步长独立校准残差分位数，从而更好地捕捉时间依赖性与不确定性增长；

**🔧 技术方法**

使用的技术包括拆分式一致性预测（SCP）、多步拆分一致性预测（MSCP）、Ensemble Batch Prediction Intervals（EnbPI）、Sequential Predictive Conformal Inference（SPCI）、Global-CP、在线学习控制器（ACI、AcMCP）、以及传统参数预测区间；

**📊 数据集**

实验数据集为3,000多条月度零售与消费品销量序列，涵盖不同国家与行业；

**📈 对比分析**

在90%覆盖率目标下，MSCP、ACI、Parametric-PI三者在覆盖率和Winkler区间分数上表现最佳，MSCP在更大数据规模下显著优于其它方法；

**⚠️ 局限性**

局限在于仅基于ARIMA模型，缺少对更复杂预测器（如神经网络）的评估，并且在非混合或高度非平稳环境下的适用性未得到充分验证。

---

## 754. Nearly Optimal Bayesian Inference for Structural Missingness

**arXiv ID:** 2601.18500 | [PDF](https://arxiv.org/pdf/2601.18500v1)

**作者:** Chen Liang `[一作]` (Harbin Institute of Technology), Yifei Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 7912 | [OpenAlex ID](https://openalex.org/A5100355218)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对结构缺失（Structural Missingness）的近似贝叶斯推理框架，通过在第二阶结构因果模型（Second‑Order SCM）下训练 Prior‑Fitted Network（PFN）和 Conditional Flow Matching 以实现对缺失值的后验预测与不确定性传播；

**💡 创新点**

创新点包括：①将缺失推理拆分为后验预测与缺失后验两部分，使用 PFN 在结构因果先验下实现全局后验推断；②利用流匹配学习缺失值后验，避免单点插补导致的偏差；③提供有限样本下的贝叶斯近优性理论与偏差削减分析；

**🔧 技术方法**

使用技术主要有：第二阶结构因果模型生成任务先验、Prior‑Fitted Network 进行后验预测、Conditional Flow Matching 进行缺失后验学习、贝叶斯预测分布积分、理论误差与样本复杂度证明；

**📊 数据集**

实验基准涵盖 58 个表格数据集，其中 33 个分类数据集（缺失率≤10%）、10 个分类数据集（>10%）、15 个用于 MNAR 评估的填充任务，全部来自公开数据集；

**📈 对比分析**

与 50 种不同缺失处理方法（XGBoost、LightGBM、CatBoost、TabPFN 等）对比，PFN‑Flow 在分类准确率、填充 MAE 及运行时效率上均居首，速度提升可达数千倍，填充误差最小、稳定性最佳；

**⚠️ 局限性**

局限性包括：对结构缺失先验的依赖需要人工设定或数据驱动的 SCM；预训练生成任务的合成性可能与真实分布偏差；在极端缺失模式或极大样本规模下的可扩展性仍待验证。

---

## 755. AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security

**arXiv ID:** 2601.18491 | [PDF](https://arxiv.org/pdf/2601.18491v1)

**作者:** Dongrui Liu `[一作]` (Shanghai Artificial Intelligence Laboratory), Xia Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AgentDoG诊断性安全门控模型，构建三维安全分类法并实现细粒度风险诊断与可解释性；

**💡 创新点**

创新在于统一的三维安全分类法（风险来源、失败模式、真实后果）、针对agent轨迹的细粒度诊断与可解释性推断框架，以及基于此分类法的合成数据集ATBench；

**🔧 技术方法**

采用大规模LLM微调、基于规划的合成管线、信息增益与概率落差的层级归因方法，以及XAI层级解释模块；

**📊 数据集**

使用合成的ATBench（约1.7k工具、4.5k交互轮次）以及公开的R‑Judge、ASSE‑Safety等基准；

**📈 对比分析**

在R‑Judge、ASSE‑Safety和ATBench上，AgentDoG在安全轨迹判别上取得F1≈90–92%，远超专用门控模型（如LlamaGuard、Qwen3‑Guard）且与大型通用模型相近；在细粒度诊断上，风险来源准确率≈82%，失败模式≈32%，后果≈58%，显著优于基线模型；

**⚠️ 局限性**

局限包括仅处理文本轨迹、缺乏多模态支持、未实现主动对齐功能、对超大模型的可扩展性尚待验证。

---

## 756. LoD-Structured 3D Gaussian Splatting for Streaming Video Reconstruction

**arXiv ID:** 2601.18475 | [PDF](https://arxiv.org/pdf/2601.18475v1)

**作者:** Xinhui Liu `[一作]` (University of Hong Kong), Dong Xu `[通讯]` (University of Hong Kong)

**通讯引用:** 24425 | [OpenAlex ID](https://openalex.org/A5082181536)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了StreamLoD-GS，一种面向流式自由视角视频重建的层级细节（LoD）高斯展开框架。

**💡 创新点**

创新点包括：① Anchor+Octree结构化LoD高斯展开与分层高斯丢弃，提升稀视角下的鲁棒性；② 基于高斯混合模型（GMM）的运动分区，将场景动态与静态区域分离；③ 量化残差细化机制，显著压缩存储同时保持视觉质量。

**🔧 技术方法**

采用技术包括3D Gaussian Splatting、Octree分层、Anchor+Octree LoD结构、分层高斯丢弃（Hierarchical Gaussian Dropout）、GMM运动分区、共享MLP属性预测、量化残差压缩、以及流式训练流程。

**📊 数据集**

在N3DV和Meet Room公开数据集上进行实验，分别使用3/4/5/6/9/12视角进行训练与评估。

**📈 对比分析**

与StreamRF、3DGStream、HiCoM、4DGC、QUEEN等主流基线比较，结果显示：在稀疏视角下PSNR平均提升约2–3 dB，存储占用降低70%+，渲染速度提升10–30 FPS；在12视角的密集视角场景下依然保持最高PSNR、最快渲染、最小存储。

**⚠️ 局限性**

局限性在于GMM运动分区对光滑、反射性表面易误判为动态区域，导致伪运动和光照不一致，需要进一步加入颜色一致性与几何先验来提高分割精度。

---

## 757. Fair-Eye Net: A Fair, Trustworthy, Multimodal Integrated Glaucoma Full Chain AI System

**arXiv ID:** 2601.18464 | [PDF](https://arxiv.org/pdf/2601.18464v1)

**作者:** Wenbin Wei `[一作]`, Xiangyu Gao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一套完整的多模态眼底与临床数据融合、风险预测与动态预警的全链条青光眼AI系统（Fair-Eye Net）。

**💡 创新点**

创新点在于：①将基金会影像、OCT结构指标、视场功能指标和人口学因素通过双流异构融合架构统一处理；②在模型中嵌入层次化不确定性门控，实现安全的“可选择预测”与人工审核；③将公平性约束作为主目标而非事后校正，并在训练与评估全过程持续关注亚组公平；④通过多任务学习同时完成筛查与进展率预测，实现闭环风险管理。

**🔧 技术方法**

核心技术包括：双流 ResNet-50 + Densely Connected Clinical Encoder；基于 MC Dropout 与 TTA 的贝叶斯不确定性估计；不确定性门控与自适应阈值；多任务损失（分类 + Smooth L1 回归）与公平性约束；多中心交叉验证与外部OOB评估。

**📊 数据集**

主要使用的数据集有：SMDG-19（视觉预训练、12,449 张影像）；Harvard‑GDP（1,214 名患者、9,872 次随访，包含影像、OCT、VF 与临床变量）；GRAPE（3,058 张影像，2 个未知设备品牌，OOD 验证）；Harvard‑GF（分层种族组、用于公平性审计）。

**📈 对比分析**

与SOTA单模态或融合模型对比，Fair‑Eye Net 在 AUC、准确率、F1 等指标上保持领先；在筛查方面达到 0.912 的 AUC、83.7% 的灵敏度与 96.7% 的特异度；在动态风险预警上能提前 3–12 个月发出警报，敏感度 92%、特异度 88%；公平性指标（跨组FNR差距）从 0.123 降至 0.033，减少 73.4% 的差距。

**⚠️ 局限性**

局限性包括：①仍缺乏更大规模、多中心的前瞻性验证；②阈值与成本敏感度未进行临床优化；③公平性校正对跨域迁移的鲁棒性待验证；④未实现时间到事件预测与个性化随访推荐；⑤模型对设备与协议的依赖仍需进一步消除。

---

## 758. Regulatory Hub Discovery in MDD Methylome: Hypotheses for Molecular Subtypes via Computational Analysis

**arXiv ID:** 2601.18498 | [PDF](https://arxiv.org/pdf/2601.18498v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 759. GenAI for Social Work Field Education: Client Simulation with Real-Time Feedback

**arXiv ID:** 2601.18517 | [PDF](https://arxiv.org/pdf/2601.18517v1)

**作者:** James Sungarda `[一作]` (University of Hong Kong), Ben Kao `[通讯]` (University of Hong Kong)

**通讯引用:** 6203 | [OpenAlex ID](https://openalex.org/A5063695659)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了名为 SWITCH 的社会工作实训聊天机器人，能够模拟客户、即时识别咨询师使用的技能，并根据 Motivational Interviewing（MI）阶段动态推进会话。

**💡 创新点**

创新点在于将 MI 阶段结构与可动态更新的认知模型（静态字段+动态情绪/自动思维+开放度参数）结合；同时引入基于检索的上下文学习（ICL）和 BERT 多标签分类两种方式提升技能识别精度。

**🔧 技术方法**

技术包括：GPT‑4o‑mini 用于对话生成与 MI 阶段判断，BERT（微调）用于多标签技能分类，检索模型 BM25、MiniLM、BGE‑M3 用于 ICL 示例检索，链式思维提示及阈值优化。

**📊 数据集**

数据集来自 19 条社会工作对话转录，共 4,734 条咨询师发言，标注 20 种咨询技巧（含频率分布），按 80/20 划分训练/测试。

**📈 对比分析**

与仅提示技能列表或定义+例子的基线相比，ICL 方案准确率 0.92–0.94、BERT 方案 0.98–0.99，宏/微 F1 与准确率均显著提升，BGE‑M3 检索在三种检索方法中略优。

**⚠️ 局限性**

局限性包括：稀有技巧分类效果不佳（数据不平衡）；只覆盖 MI 的前三个阶段，未考虑后续阶段；目前仅支持文本交互，缺乏语音/视频多模态实现。

---

## 760. Coding Schemes for Document Exchange under Multiple Substring Edits

**arXiv ID:** 2601.18441 | [PDF](https://arxiv.org/pdf/2601.18441v1)

**作者:** Hrishi Narayanan `[一作]` (Technical University of Munich), Antonia Wachter-Zeh `[通讯]` (Technical University of Munich)

**通讯引用:** 1143 | [OpenAlex ID](https://openalex.org/A5041962883)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种针对多次子串编辑（k-substring edits）的文档交换方案，并给出了其在平均情况上的改进版本。

**💡 创新点**

创新点在于：①利用syndrome compression技术和标签函数实现了与已知最优冗余（4tlog+o(log)）相当但复杂度显著降低的方案；②在平均情况下通过划分“模式稠密”字符串与非稠密字符串两类，并使用预编码思想，首次实现了平均冗余降至(4-1)log+o(log)的效果。

**🔧 技术方法**

主要技术包括：
- 语义压缩（syndrome compression）与标签函数的设计；
- 子串编辑误差模型的混合误差校正理论；
- 基于模式稠密性的划分与预编码策略；
- 误差球（confusion ball）大小估计与上界分析。

**📊 数据集**

本文没有使用具体的数据集，而是完全基于理论分析与复杂度证明；所有结果均为上界/期望上界的形式。

**📈 对比分析**

与之前使用分布式图着色得到的4log+O(loglog)冗余但复杂度为O(n^8t)的方案相比，本文的方案在相同冗余水平下实现了O(n^2t+1)的编码复杂度和O(n^{t+1})的解码复杂度；平均情况上，冗余下降至(4-1)log+o(log)，但未给出具体实验验证。

**⚠️ 局限性**

局限性包括：
- 对于最坏情况冗余并未进一步突破4tlog+o(log)；
- 平均冗余的改进依赖于“模式稠密”假设，实际文本分布可能不满足；
- 方案仍然是理论构造，缺乏对真实数据的实验评估。

---

## 761. Exploring Fine-Tuning for In-Context Retrieval and Efficient KV-Caching in Long-Context Language Models

**arXiv ID:** 2601.18527 | [PDF](https://arxiv.org/pdf/2601.18527v1)

**作者:** Francesco Maria Molfese `[一作]` (Sapienza University of Rome), Adrià de Gispert `[通讯]` (Amazon AGI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过强化学习（GRPO）和可验证奖励函数对长上下文语言模型进行微调，使其能够更好地聚焦并利用相关信息，提升在检索增强生成任务中的表现。

**💡 创新点**

首次提出将多种可验证奖励（答复正确、文档ID、内容复制、引用提取、LLM判官推理）与GRPO结合，显著提高模型在域内的鲁棒性并在部分跨域任务中超越传统RAG。

**🔧 技术方法**

采用GRPO（强化学习）、可验证奖励函数、KV‑Cache压缩技术，并在Qwen2.5‑7B‑Instruct‑1M长上下文模型上进行训练。

**📊 数据集**

训练数据基于HotpotQA和2WikiMultihopQA，评估数据涵盖Helmet RAG子集、∞Bench、LongBench‑v2、Loong金融等多任务、多域基准。

**📈 对比分析**

与基线RAG@32k和SFT对比，GRPO策略在Helmet域内提升高达+20分，在某些跨域任务（如Loong金融、∞Bench QA）也取得超越RAG的成绩，但在多选和摘要任务仍略逊于RAG。

**⚠️ 局限性**

受限于仅在32k上下文训练、仅使用Wiki段落、缺乏多语言与多模态数据，导致极长序列训练和跨域泛化仍存在不足。

---

## 762. DisasterInsight: A Multimodal Benchmark for Function-Aware and Grounded Disaster Assessment

**arXiv ID:** 2601.18493 | [PDF](https://arxiv.org/pdf/2601.18493v1)

**作者:** Sara Tehrani `[一作]` (Linkoping University), Michael Felsberg `[通讯]` (Linkoping University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 DisasterInsight 基准，构建了建筑级灾害评估的功能、损伤、灾害类型分类、计数和结构化报告生成等多模态任务。

**💡 创新点**

创新点包括将 OSM 功能标签迁移至实例级别、引入多提示评估策略、设计基于人道主义指南的结构化报告生成，并提供了域适应的 DI‑Chat 基线。

**🔧 技术方法**

采用低秩适配 LoRA 的指令微调和 Video‑LLaVA 风格的多模训练框架，在 TeoChat、Qwen2.5‑VL 等视觉语言模型上实现。

**📊 数据集**

基于 xBD 数据集并结合 OSM 补全，构造了 112,507 个建筑实例，形成灾害评估的多模态数据集。

**📈 对比分析**

与 LLaVA‑OneVision、Qwen‑VL 等通用模型比较，DI‑Chat 在损伤级别、灾害类型分类、计数和报告生成等任务上显著提升（宏 F1 与 BERTScore 最高），但建筑功能分类仍落后于最优通用模型。

**⚠️ 局限性**

局限包括 OSM 标签噪声、仅依赖光学影像、自动生成报告缺乏人工评估，以及建筑功能识别的严重类别不平衡。

---

## 763. OffSeeker: Online Reinforcement Learning Is Not All You Need for Deep Research Agents

**arXiv ID:** 2601.18467 | [PDF](https://arxiv.org/pdf/2601.18467v1)

**作者:** Yuhang Zhou `[一作]` (Fudan University), Jingjing Chen `[通讯]` (Fudan University)

**通讯引用:** 5285 | [OpenAlex ID](https://openalex.org/A5100373492)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了完整的离线训练框架和数据集，用于构建大型深度研究代理，并发布了8B规模的离线训练模型。

**💡 创新点**

提供了轻量级的数据合成管道、公开的66k QA+33k SFT+21k DPO数据集，并证明仅凭离线SFT+DPO即可匹敌在线RL系统。

**🔧 技术方法**

采用LLM提示驱动的实体扩展与问题生成、ReAct式工具调用框架、监督微调+Direct Preference Optimization (DPO)以及Qwen3-8B模型。

**📊 数据集**

自研的DeepSearch任务合成数据集（66k QA、33k SFT轨迹、21k DPO对），并在GAIA、BrowseComp、HLE、XBench-DeepSearch、WebWalkerQA等六大基准上评测。

**📈 对比分析**

与多款同尺度与更大尺度基线（WebSailor、DeepDive、Claude-4、DeepSeek）在pass@1指标对比，离线训练的8B模型在多数基准上名列前茅，甚至逼近30B在线RL系统。

**⚠️ 局限性**

受上下文长度限制，部分极长查询超出128k令牌；数据主要覆盖网页搜索，缺乏学术文献、多模态等领域，离线模型对超长链路推理仍有限制。

---

## 764. On the existence of heavy columns in binary matrices with distinct rows

**arXiv ID:** 2601.18450 | [PDF](https://arxiv.org/pdf/2601.18450v1)

**作者:** Jamolidin K. Abdurakhmanov `[一作]` `[通讯]` (Andijan State University), Jamolidin K. Abdurakhmanov (Andijan State University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究了在行唯一的二进制矩阵中，证明存在至少一个“重列”（一列中1的数量不少于0），并给出了两种递归算法来检验这一性质。

**💡 创新点**

首次提出利用“未配对行”与递归结构来证明重列存在的理论框架，并在第二个算法中加入了提前终止条件，显著简化判断过程。

**🔧 技术方法**

递归子矩阵筛选、列过滤、配对行与未配对行概念、鸠形原理和递归调用。

**📊 数据集**

无实验数据集，论文为纯理论证明。

**📈 对比分析**

未进行实验对比，研究仅在理论层面证明算法的正确性，未给出时间复杂度或实测性能。

**⚠️ 局限性**

限制在于需满足行唯一、列唯一且无全零列；算法复杂度可能指数级；未给出寻找重列的具体实现或实际应用评估。

---

## 765. On Procrustes Contamination in Machine Learning Applications of Geometric Morphometrics

**arXiv ID:** 2601.18448 | [PDF](https://arxiv.org/pdf/2601.18448v1)

**作者:** Lloyd Austin Courtenay `[一作]` `[通讯]` (University of Bordeaux), Lloyd Austin Courtenay (University of Bordeaux)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在几何形态计量中使用Generalised Procrustes Analysis（GPA）对机器学习模型的影响，提出了一种在训练前仅对训练集进行GPA并将测试集对齐到训练参考点的“重新对齐”流程。

**💡 创新点**

创新点在于正式量化GPA导致的数据污染效应，揭示了样本大小与标记数之间的“对角线”关系（2D斜率≈1/3，3D≈1/4），并证明空间自相关对预测精度的显著提升，提供了避免数据泄露的实用预处理指南。

**🔧 技术方法**

采用了仿真生成的二维和三维标记数据，使用线性回归和卷积神经网络进行形态-尺寸回归，并通过RMSE比较污染与未污染两种GPA流程的性能。

**📊 数据集**

使用的是合成数据（基于单位圆或球面、加噪、剪切、尺度变换的标记集），没有使用真实生物学数据集。

**📈 对比分析**

通过比较两种GPA流程的RMSE，发现污染流程在大多数情形下略微降低RMSE（平均差-0.0117），尤其在标记数/样本比高时更明显；利用卷积网络利用空间结构的模型明显优于仅线性回归，说明空间自相关重要。

**⚠️ 局限性**

局限包括：仅基于等方差的仿真，未检验在真实生物数据（具有模块化、异方差等特征）中的表现；对GPA参数（是否保留尺度）未做系统探究；未评估非线性或更复杂模型的鲁棒性。

---

## 766. UrgentMOS: Unified Multi-Metric and Preference Learning for Robust Speech Quality Assessment

**arXiv ID:** 2601.18438 | [PDF](https://arxiv.org/pdf/2601.18438v1)

**作者:** Wei Wang `[一作]` (Shanghai Jiao Tong University), Yanmin Qian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 11754 | [OpenAlex ID](https://openalex.org/A5100341993)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了统一的语音质量评估框架UrgentMOS，能够同时学习绝对分数和相对偏好，并且对缺失的指标标签具有容忍性；

**💡 创新点**

创新点在于：①采用多指标监督和缺失标签容忍机制，使模型能在多源异构数据上鲁棒训练；②将绝对MOS预测与自然性条件下的偏好建模结合，实现绝对与比较两种评估模式的统一；③通过从ACR数据生成偏好标注的技术，扩大可用的偏好训练数据；

**🔧 技术方法**

使用多分支预训练音频编码器的共享特征提取器、Transformer编码器、多指标预测模块、自然性条件偏好模块，配合范围约束激活、交叉注意力和加权损失；

**📊 数据集**

训练数据涵盖TTS、VC、SE和模拟失真等多种来源：BC19、BVCC、NISQA、PSTN、SOMOS、TCD-VoIP、Tencent、URGENT2024/25 SQA；偏好数据集包括SpeechEval和SpeechJudge-Data；评估使用官方测试集以及构造的参考级/语料库/任意匹配的偏好对；

**📈 对比分析**

通过与DNSMOS、UTMOS、SCOREQ、Distill-MOS、NISQA-MOS等基准模型在绝对分数相关性（Pearson/Spearman）和偏好准确率（acc_0.5/acc_0）上对比，UrgentMOS在绝大多数数据集上取得最高或第二高的相关性和偏好准确率，显著优于现有方法；

**⚠️ 局限性**

局限性包括：未提供自然语言解释，推理时多特征提取器会增加计算成本；多指标监督若指标相关性弱可能导致性能下降；缺乏针对极低/高质量样本的特定校准。

---

## 767. Rank-1 Approximation of Inverse Fisher for Natural Policy Gradients in Deep Reinforcement Learning

**arXiv ID:** 2601.18626 | [PDF](https://arxiv.org/pdf/2601.18626v1)

**作者:** Yingxiao Huo `[一作]` (University of Manchester), Mingfei Sun `[通讯]` (University of Manchester)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5101591811)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于经验Fisher矩阵与Sherman‑Morrison公式的秩‑1近似自然策略梯度，并将其集成到Actor‑Critic框架中实现SM‑ActorCritic；

**💡 创新点**

创新点在于：1）利用秩‑1更新避免显式求逆，时间复杂度降至O(d)；2）理论证明在某些条件下该近似具备全局收敛性与样本复杂度与随机策略梯度相当；3）结合经验Fisher实现可扩展的自然梯度更新；

**🔧 技术方法**

技术手段包括：经验Fisher矩阵估计、Sherman‑Morrison公式、GAE优势估计、A2C基础架构、Adam与SGD优化、卷积梯度下降（CG）对比；

**📊 数据集**

实验数据集：OpenAI Gym经典控制环境（Acrobot、Cartpole、Pendulum）和MuJoCo物理仿真环境（Half‑Cheetah、Hopper、Swimmer、Walker、Humanoid、Pusher）；

**📈 对比分析**

与三种基线（AC‑SGD、AC‑Adam、AC‑CG）对比；SM‑ActorCritic在大多数MuJoCo任务和部分经典任务中收敛更快、最终回报更高，样本效率更好；在某些任务如Pendulum与部分MuJoCo环境表现相当或略逊；

**⚠️ 局限性**

局限性：1）在部分环境中不如CG近似自然梯度；2）单样本版本训练次数多、计算量大；3）可能因过早变得过度自信导致次优策略；4）依赖经验Fisher近似，若样本不足可能误差放大。

---

## 768. CASSANDRA: Programmatic and Probabilistic Learning and Inference for Stochastic World Modeling

**arXiv ID:** 2601.18620 | [PDF](https://arxiv.org/pdf/2601.18620v1)

**作者:** Panagiotis Lymperopoulos `[一作]` (Tufts University), Kaheer Suleman `[通讯]` (Skyfall AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研发了一种双流神经符号世界模型框架，利用LLM先验生成并细化代码化的确定性转移模型，以及通过LLM引导的因果结构学习构建概率图模型，以实现对商业模拟环境中混合确定与随机动态的建模与规划。

**💡 创新点**

创新点在于：①同时结合符号代码和概率图模型两种子系统，①1) 代码化确定性部分通过LLM生成并通过轨迹误差进行有针对性微调；② 代码化随机部分通过LLM引导的因果结构学习获得稀疏可解释的DAG；②首次在复杂业务模拟器中将LLM作为知识先验来指导混合动力学建模，显著提升了模型的可解释性与样本效率。

**🔧 技术方法**

技术方法包括：LLM（GPT‑4.1‑mini）生成初始代码与因果图；代码世界模型用于确定性转移；概率图模型与结构学习用于随机变量的因果捕捉；MCTS 与 MPC 作为下游规划器；实验中还使用了消融分析来验证各组件贡献。

**📊 数据集**

使用的数据集为两个业务模拟器：CoffeeShopSimulator（咖啡店日常管理）和 Mini Amusement Parks (MAPs)（主题公园运营），每个环境都提供环境文本描述与观测轨迹作为训练与评估数据。

**📈 对比分析**

比较方法：将完整框架与基线模型（WALL‑E、WorldCoder）以及三种消融版本（独立随机预测、随机线性DAG、无LLM先验的随机DAG）在转移预测与规划任务中进行对比。实验显示：在 CoffeeShopSimulator 中，完整模型在 MPC 下的收益显著高于基线；在 MAPs 中，完整模型实现 50 天全部存活率 100%，而基线大部分跑法均破产；消融实验进一步证明因果结构学习和代码微调对性能提升的关键作用。

**⚠️ 局限性**

局限性：①需要先验划分确定性与随机性变量，可能不适用于所有领域；②模型对LLM的知识质量敏感，若环境描述不足或LLM理解错误可能导致先验错误；③未实现在线或主动学习，模型仅在离线训练后使用；④在更大规模或更复杂环境中可扩展性与推理速度尚待验证。

---

## 769. Geometry-Free Conditional Diffusion Modeling for Solving the Inverse Electrocardiography Problem

**arXiv ID:** 2601.18615 | [PDF](https://arxiv.org/pdf/2601.18615v1)

**作者:** Ramiro Valdes Jara `[一作]` (University of Miami), Adam Meyers `[通讯]` (University of Miami)

**通讯引用:** 2562 | [OpenAlex ID](https://openalex.org/A5043134485)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种无几何条件扩散模型，用于解决逆电生理学（ECGI）问题；

**💡 创新点**

通过将扩散模型与Transformer去噪器相结合，捕捉多解的概率分布，避免依赖患者特定几何体，解决传统方法缺乏不确定性量化和几何构造的缺陷；

**🔧 技术方法**

使用条件扩散概率模型（DDPM）与Transformer架构实现噪声预测去噪，结合时间步信息进行迭代重建；

**📊 数据集**

在犬类心脏实验数据集上训练与测试，该数据集包括490个心表面电极记录与192个体表电极的模拟信号，噪声水平约20 dB；

**📈 对比分析**

与1D‑CNN、LSTM及Transformer基线进行对比，扩散模型在时间相关系数（0.78）和均方误差、平均绝对误差方面均优于其它方法；

**⚠️ 局限性**

受限于训练数据仅包含单一噪声水平和固定前向模型配置，导致学习到的条件分布集中，实际不确定性估计不足，未来需引入更丰富的噪声与模型扰动来提升不确定性校准。

---

## 770. Learning long term climate-resilient transport adaptation pathways under direct and indirect flood impacts using reinforcement learning

**arXiv ID:** 2601.18586 | [PDF](https://arxiv.org/pdf/2601.18586v1)

**作者:** Miguel Costa `[一作]` (Technical University of Denmark), Francisco C. Pereira `[通讯]` (Technical University of Denmark)

**通讯引用:** 7026 | [OpenAlex ID](https://openalex.org/A5001424439)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一套结合气候模型与强化学习的整合评估框架，用于学习多年代的城市交通抗洪投资路径。

**💡 创新点**

创新点在于将 IAM 与 RL 结合形成可插拔模块的决策支持框架，能够在深度不确定性和空间耦合下学习时间-空间协同的适应策略，并能对不同气候情景做鲁棒性评估。

**🔧 技术方法**

使用强化学习（PPO）与图神经网络策略，对雨量投影、洪水模型、交通仿真与成本评估模块进行集成；同时构建了基于天气情景的 IAM 环境。

**📊 数据集**

数据集包括丹麦气象局的气候图谱（RCP2.6/4.5/8.5）、SCALGO Live 洪水工具、OpenStreetMap 的交通网络、丹麦全国出行调查（TU）等。

**📈 对比分析**

与无行动、随机行动两基线以及不同气候情景下的对比，学习到的策略在2024–2100年期间累计成本比无行动低22%，比随机行动低408%；在不同情景下鲁棒性良好，极端情景下也能获得显著收益。

**⚠️ 局限性**

局限包括：仅使用三种离散气候情景，缺乏概率轨迹和信念更新；模型假设可能遗漏分布性影响；训练计算量大，限制了行动空间与场景数量；框架仅优化金钱成本，未考虑公平与健康等社会效益。

---

## 771. One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization

**arXiv ID:** 2601.18572 | [PDF](https://arxiv.org/pdf/2601.18572v1)

**作者:** Franziska Weeber `[一作]` (University of Stuttgart), Sebastian Padó `[通讯]` (University of Stuttgart)

**通讯引用:** 6051 | [OpenAlex ID](https://openalex.org/A5003870894)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了六种常用persona提示方式对LLM个性化偏差的影响；

**💡 创新点**

创新点在于将多种persona cue与任务、模型、族群维度进行跨维度关联分析，揭示单一提示可能产生误导性结论；

**🔧 技术方法**

采用LLM prompt工程、Spearman相关、ANOVA和Tukey-Kramer后验检验等统计方法；

**📊 数据集**

使用SBB、MMMD、AITA、IssueBench等评测数据集，以及PRISM和ImplicitPersona构造的会话历史；

**📈 对比分析**

通过相关系数与显著性检验发现不同提示高度相关但仍存在显著差异，提示需多种cue共同评估以获得稳健结论；

**⚠️ 局限性**

研究局限包括仅考察三种族群维度、仅用美国选民登记姓名、未覆盖交叉效应和开放式任务样本不足。

---

## 772. Feature-Indexed Federated Recommendation with Residual-Quantized Codebooks

**arXiv ID:** 2601.18570 | [PDF](https://arxiv.org/pdf/2601.18570v1)

**作者:** Mingzhe Han `[一作]` (Fudan University), Tun Lu `[通讯]` (Fudan University)

**通讯引用:** 2048 | [OpenAlex ID](https://openalex.org/A5004237040)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于特征索引通信的新框架RQFedRec，解决了传统ID索引通信在联邦推荐中的通信开销大、跨物品泛化差和噪声敏感等问题。

**💡 创新点**

创新点在于将通信单元从物品ID改为共享的离散码ID，利用残差量化生成多层码书并通过协作-语义双通道聚合，同时采用课程学习逐步引入协作信息。

**🔧 技术方法**

核心技术包括残差量化KMeans（RQ‑kmeans）、双通道码书聚合、代码书生成与更新、课程学习策略以及差分隐私噪声保护。

**📊 数据集**

在五个公开数据集（MovieLens 100K/1M、Steam、Toys、Book）上进行实验。

**📈 对比分析**

与FedMF、FedNCF、PFedRec、GPFedRec、FedRAP、FedCIA等SOTA联邦推荐方法比较，RQFedRec在相同通信预算下的Recall@10、MRR@10、NDCG@10均显著提升，通信开销亦显著下降。

**⚠️ 局限性**

限制主要包括需预先训练大型语言模型提取语义信息、码书更新周期与协作信息同步受限，以及在极低通信预算或极高噪声环境下性能仍会下降。

---

## 773. An LLM-Agent-Based Framework for Age of Information Optimization in Heterogeneous Random Access Networks

**arXiv ID:** 2601.18563 | [PDF](https://arxiv.org/pdf/2601.18563v1)

**作者:** Fang Liu `[一作]` (Shenzhen University), Shengli Zhang `[通讯]` (Shenzhen University)

**通讯引用:** 22547 | [OpenAlex ID](https://openalex.org/A5100413426)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种基于大型语言模型（LLM）代理的 Reflex-Core 框架，利用 Observe‑Reflect‑Decide‑Execute 闭环对异构随机接入网络的年龄信息（AoI）进行自适应优化，并提出了优先级感知的 Reflexive Multiple Access (RMA) 协议。

**💡 创新点**

创新点在于将 LLM 的语义推理与强化学习（SFT + PPO）相结合，形成可解释且可自我进化的决策循环，首次实现了在异构网络中通过语言思考自动推断对手协议并实时调整传输策略；同时引入优先级机制以满足多层 QoS 要求。

**🔧 技术方法**

采用的核心技术包括：大语言模型（LongChat‑7B‑16k）作为代理核心；监督微调（SFT）和近端策略优化（PPO）对模型进行域特定的奖励对齐；Observe‑Reflect‑Decide‑Execute 四模块闭环实现多尺度时间控制；以及对优先级的语义理解与策略记忆机制。

**📊 数据集**

使用的是基于仿真生成的异构网络场景数据（Scenario 1‑5，包含 ALOHA、TDMA 与 RMA 节点），每个场景通过时间槽模拟传输、碰撞与 AoI 变化，未使用公开现实数据集。

**📈 对比分析**

与 CP‑AgentNet（LLMA）和 DLMA 对比，RMA 在五个异构场景中平均降低系统 AoI 10.3‑14.9%，节点级 AoI 13.8‑14.9%，并在动态网络与优先级场景中实现了 20% 的收敛速度提升和 14.2‑17.2% 的 AoI 减少；实验采用仿真脚本和自定义指标进行客观评估。

**⚠️ 局限性**

主要局限包括：模型在边缘设备上的计算与存储成本较高；对非常高密度或极端动态网络的鲁棒性仍需进一步验证；实验仅基于仿真数据，缺乏真实环境验证；对对抗性攻击与安全隐患的处理尚未充分探讨。

---

## 774. Deconstructing Instruction-Following: A New Benchmark for Granular Evaluation of Large Language Model Instruction Compliance Abilities

**arXiv ID:** 2601.18554 | [PDF](https://arxiv.org/pdf/2601.18554v1)

**作者:** Alberto Purpura `[一作]` (Capital One), Adam Faulkner `[通讯]` (Capital One)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MOSAIC框架，用模块化合成数据评估LLM的指令遵从能力。

**💡 创新点**

创新点在于动态生成、20条以上应用型约束，独立分析指令遵从与任务完成，并考虑约束数量、顺序及交互。

**🔧 技术方法**

采用LLM-as-a-judge、模块化约束列表、Pearson相关分析、位置偏差实验等技术。

**📊 数据集**

使用自研的MOSAIC数据集，包含4000条文本生成任务，平均10.5约束，涵盖5组21种约束类型。

**📈 对比分析**

对七款LLM进行单/双约束合规率评估，发现不同模型在语义/格式约束上表现差异，说明该基准能细粒度揭示缺陷。

**⚠️ 局限性**

局限包括评判器偏差、合成数据可能不完全代表真实指令、仅评估部分模型与任务、未覆盖多语言与其他领域。

---

## 775. Unknown Unknowns: Why Hidden Intentions in LLMs Evade Detection

**arXiv ID:** 2601.18552 | [PDF](https://arxiv.org/pdf/2601.18552v1)

**作者:** Devansh Srivastav `[一作]` (CISPA Helmholtz Center for Information Security), Lea Schönherr `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建实验性测试床，系统研究了大型语言模型（LLM）中的隐藏意图行为，提出了基于社会科学的十类隐藏意图分类框架，并评估了多种检测方法在现实开放环境中的有效性。

**💡 创新点**

创新点包括：①将隐藏意图拆解为十个设计层面的策略类别；②利用提示工程和后处理在不修改模型权重的情况下诱导并标注隐藏意图；③在低频率、类别无关的真实场景下对LLM判断器进行压力测试，揭示其检测瓶颈。

**🔧 技术方法**

技术手段包括：提示工程、规则后处理、角色模板生成（模拟隐藏意图）；LLM判定器（推理型与非推理型）进行判别；静态嵌入式分类器作对照；以及精度、召回、FPR、FNR等评价指标。

**📊 数据集**

数据集：自建4000条（每类400条）prompt‑response对，包含触发与非触发实例；此外在三款SOTA LLM上收集真实输出进行案例验证。

**📈 对比分析**

评估方法：在类别特定与类别无关两设置下，计算准确率、F1、FPR、FNR；结果显示：类别特定判定效果明显优于无关判定；但在低普遍率下precision迅速崩溃，FNR也显著升高；推理模型并未显著提升检测性能。

**⚠️ 局限性**

局限性：仅测试单轮、单一隐藏意图实例；未覆盖多轮对话、跨语言场景；样本比例为10%人工标注；未进行专家级人机循环验证，导致对真实复杂情境的评估不足。

---

## 776. SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction

**arXiv ID:** 2601.18537 | [PDF](https://arxiv.org/pdf/2601.18537v1)

**作者:** Linyong Gan `[一作]` (Chinese University of Hong Kong), Shuhang Chen `[通讯]` (COSCO SHIPPING Advanced Technology Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究长时段船舶轨迹预测，提出以语义关键点（NKP）为条件的层次化模型；

**💡 创新点**

创新点是将高层航向意图抽象为NKP并作为条件约束，实现全球语义决策与局部运动建模分离，同时采用检索增强验证与对比学习的NKP预测，支持开放集；

**🔧 技术方法**

使用Transformer解码器（MiniMind）、对比学习、检索增强验证、逆向运动学等技术；

**📊 数据集**

使用AIS真实海事数据集及公开AIS数据集进行训练与泛化测试；

**📈 对比分析**

与MP‑LSTM、TrAISformer等基线对比，MSEP、MFD、MSEC均显著下降，推理速度更快；

**⚠️ 局限性**

受限于NKP预测误差可能导致轨迹偏离，对极端环境（高噪声、突发事件）尚未充分验证。

---

## 777. REMAC: Reference-Based Martian Asymmetrical Image Compression

**arXiv ID:** 2601.18547 | [PDF](https://arxiv.org/pdf/2601.18547v1)

**作者:** Qing Ding `[一作]` (Beihang University), Xin Zou `[通讯]` (Beihang University)

**通讯引用:** 29985 | [OpenAlex ID](https://openalex.org/A5046104594)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于参考的火星图像压缩方法REMAC，旨在降低编码器复杂度并提升压缩性能。

**💡 创新点**

创新点包括：①利用火星图像的强内外相似性设计参考引导熵模块与参考解码器；②引入多尺度参考解码器以扩大感受野；③提出潜在特征回收机制，将计算负担从编码器迁移至解码器。

**🔧 技术方法**

采用学习型图像压缩框架，使用卷积自编码器、GDN层、参考选择、参考引导熵估计、多尺度残差网络以及潜在特征回收技术。

**📊 数据集**

主要使用MIC火星图像数据集（包含约3000幅图像），并在DIV2K、NASA Mars 2020数据集上进行对比与泛化评估。

**📈 对比分析**

与传统HEVC/VVC以及学习压缩方法Ballé、Minnen、Cheng、WACNN在BD-PSNR、BD-MS-SSIM、BD-LPIPS等指标上对比，REMAC在PSNR/SSIM/LPIPS上均取得最高BD-PSNR提升约+0.27 dB，同时编码器FLOPs仅为比率42%/56%等，整体在率-失真-复杂度三维上实现最佳平衡。

**⚠️ 局限性**

模型采用32位浮点权重，导致内存与计算量较大；目前未进行模型压缩或量化，限制了在极低资源设备上的部署。

---

## 778. GenAgent: Scaling Text-to-Image Generation via Agentic Multimodal Reasoning

**arXiv ID:** 2601.18543 | [PDF](https://arxiv.org/pdf/2601.18543v1)

**作者:** Kaixun Jiang `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**通讯引用:** 3173 | [OpenAlex ID](https://openalex.org/A5100669255)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 GenAgent，一种基于代理的多模态模型，能够将视觉理解和图像生成解耦，通过外部图像生成工具的调用与反思，支持多轮交互式生成。

**💡 创新点**

创新点包括：① 将生成器视为可调用工具，形成可动态决策的工具调用框架；② 引入多轮链式思考（推理–调用–判断–反思）实现自我迭代；③ 采用两阶段训练：先用高质量的SFT数据进行冷启动，再通过结合点奖励和对奖励的RL训练，提升反思质量和生成效果；④ 显示出跨工具泛化、测试时扩展和任务自适应推理三大特性。

**🔧 技术方法**

使用技术：多模态代理模型（Qwen2.5‑VL‑7B），链式思考（Chain-of-Thought），工具调用机制，监督微调（SFT）与强化学习（GRPO）相结合，点奖励（LLM判定）与对奖励（连续生成图像质量对比）混合；轨迹重采样提升探索；以及FLUX.1‑dev、Qwen‑Image等图像生成工具。

**📊 数据集**

数据集：训练使用自构造的工具调用与反思示例（基于FLUX.1‑dev生成失败样本、Qwen3‑VL‑235B‑A22B‑Thinking生成的思路、Gemini‑2.5‑Pro指导的反思），评估使用GenEval++、WISE、Imagine三个公开基准；同时在RL阶段利用FLUX.1‑dev和Qwen‑Image两款生成器进行交互。

**📈 对比分析**

与扩散模型（Qwen‑Image、FLUX.1‑dev）、统一架构（Janus‑Pro‑7B、Bagel、GPT‑4o）以及分离式方法（PromptEnhancer、ReflectionFlow）对比，GenAgent 在 GenEval++ 提升 +23.6%（+14% 在 WISE）并接近 GPT‑4o，在所有三个基准上均实现或超过现有开源方法的最佳成绩，且表现出良好的跨工具泛化和多轮提升。

**⚠️ 局限性**

局限性：生成效果仍受限于底层生成器能力，超过两轮反思提升有限；存在过度反思导致循环的失败案例；RL训练仍需大量交互样本，训练成本不低；目前主要聚焦图像生成，对更广泛的多模态任务或文本到图像的高复杂度需求仍有提升空间。

---

## 779. Information Hidden in Gradients of Regression with Target Noise

**arXiv ID:** 2601.18546 | [PDF](https://arxiv.org/pdf/2601.18546v1)

**作者:** Arash Jamshidi `[一作]` (University of Helsinki), Kai Puolamäki `[通讯]` (University of Helsinki)

**通讯引用:** 2588 | [OpenAlex ID](https://openalex.org/A5067547881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过向目标添加方差与批量大小相等的高斯噪声，利用梯度协方差来估计线性回归（以及实验中扩展到非线性 MLP）的海森矩阵（即数据协方差），并在此基础上实现预处理、对抗风险估计和梯度仅训练。

**💡 创新点**

关键创新在于：1) 发现只通过梯度即可恢复海森矩阵，只需对目标噪声进行精确校准（噪声方差设为批量大小）；2) 提供非渐近的算子范数误差界，证明在子高斯输入下只需 O(d/ε²) 批次即可获得全局一致估计；3) 将此估计应用于预处理加速梯度下降和对抗风险评估，展示梯度仅环境中的实用性。

**🔧 技术方法**

核心技术包括：梯度协方差分析、方差校准、子高斯分布的集中定理、矩阵范数估计、预处理（Hessian 的逆近似）以及对抗风险的闭式表达；实验使用随机矩阵采样、线性回归和多层感知机模型训练。

**📊 数据集**

实验数据集包括：
- 合成高斯分布（稠密/对角协方差）
- 公共线性回归数据集：Wave Energy Converter、Bike Sharing、California Housing、Wine Quality
- MLP 训练同一四个数据集。

**📈 对比分析**

与无噪声梯度、仅乘以批量大小的粗略估计以及梯度仅预处理方法相比，噪声校准后的梯度协方差在算子范数误差（r）上通常 < 0.1，远优于 0.5-0.9；在预处理加速梯度下降/Adam 的实验中，收敛步数从 O(κ log 1/ε) 降到 O(log 1/ε)，并在对抗风险估计中取得与解析结果相近的误差。

**⚠️ 局限性**

局限性：
- 理论证明仅针对线性回归，非线性网络的收敛与误差界仍是开放问题；
- 对批量大小的依赖较高（理论上为 Ω(d³/ε²)），实际需大批量以保证海森估计的精度；
- 噪声校准需要预知或估计噪声方差，若未知需进一步的鲁棒性分析。

---

## 780. TwinPurify: Purifying gene expression data to reveal tumor-intrinsic transcriptional programs via self-supervised learning

**arXiv ID:** 2601.18640 | [PDF](https://arxiv.org/pdf/2601.18640v1)

**作者:** Zhiwei Zheng `[一作]` (University of Glasgow), Kevin Bryson `[通讯]` (University of Glasgow)

**通讯引用:** 10588 | [OpenAlex ID](https://openalex.org/A5063296381)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发并评估了一种基于自监督Barlow Twins的框架TwinPurify，用于从bulk转录组中去除邻近正常组织干扰并提取肿瘤内在转录程序。

**💡 创新点**

通过将邻近正常组织作为结构化噪声引入自监督训练，首次在基因表达上实现Barlow Twins的自监督去噪，从而无需外部参考即可纯化肿瘤信号。

**🔧 技术方法**

使用Barlow Twins自监督学习、编码器+投影器结构、结构化邻近正常扰动、Optuna超参数优化、GSEA、Cox回归、分类评估等技术。

**📊 数据集**

使用了SCAN‑B、TCGA‑BRCA（RNA‑seq）和METABRIC（microarray）三大乳腺癌批量数据集，并生成合成稀释混合进行评估。

**📈 对比分析**

与自编码器、变分自编码器和PCA等基线在PAM50分型、肿瘤分级、存活预测及GSEA等任务中比较，TwinPurify在高纯度下降时仍保持宏F1>0.75、分类准确率提升3‑7%，GSEA通路数显著多，Cox C-index最高。

**⚠️ 局限性**

缺乏绝对纯度标注只能用合成稀释评估；纯化后维度间解耦并不保证生物学完全独立；对不同肿瘤组织外部参考的通用性尚待验证。

---

## 781. Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting

**arXiv ID:** 2601.18633 | [PDF](https://arxiv.org/pdf/2601.18633v1)

**作者:** Tong Shi `[一作]` (University of Glasgow), Paul Henderson `[通讯]` (University of Glasgow)

**通讯引用:** 6426 | [OpenAlex ID](https://openalex.org/A5068933785)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于3D高斯散射（3DGS）的单图像音频驱动口型动画方法Splat-Portrait，可在无3D监督的情况下从单张人像生成高质量、可控制的三维说话头视频。

**💡 创新点**

创新点包括：① 通过自监督预训练实现静态高斯散射重建并自动预测背景；② 设计动态解码器直接预测音频条件下的高斯位移，无需复杂形变模型；③ 在训练过程中利用2D扩散模型的score distillation提升极端视角下的细节和逼真度；④ 将所有技术整合为轻量级端到端架构，显著提升生成质量与一致性。

**🔧 技术方法**

核心技术：3D Gaussian Splatting、U-Net静态/动态生成器、FiLM条件机制、音频特征提取与注意力融合、score distillation sampling (SDS)、LPIPS + L2 复原损失、基于近似相机参数的姿态编码。

**📊 数据集**

使用公开的单目说话视频数据集HDTF（400+人、350+子集）和TalkingHead-1KH（1100人、约25k帧）进行训练与评估，并对比了多种视频驱动与音频驱动的3D说话头方法。

**📈 对比分析**

与OTAvatar、HiDe-NeRF、Real3D-Portrait、NeRFFaceSpeech、GAGAvatar+ARtalker等基线相比，Splat-Portrait在PSNR、SSIM、LPIPS、CSIM、FID及LipSync等指标上均实现了显著提升（如PSNR 23.87↑、SSIM 0.814↑、CSIM 0.726↑、FID 28.62↓、LipSync 6.218↑），在同一身份与跨身份场景下均保持领先。

**⚠️ 局限性**

局限性：① 仍需估计相机内外参，难以完全无标注；② 对极端姿态的改进依赖2D扩散模型的知识，可能在某些极端场景下不足；③ 目前仅在单一人像与静态背景上验证，尚未充分评估在复杂动态背景或多主体场景下的表现。

---

## 782. AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning

**arXiv ID:** 2601.18631 | [PDF](https://arxiv.org/pdf/2601.18631v1)

**作者:** Mingyang Song `[一作]` (Fudan University), Yu Cheng `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 31478 | [OpenAlex ID](https://openalex.org/A5000234334)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套名为 AdaReasoner 的多模态大语言模型，能够通过学习工具的使用、时机和组合，实现跨任务、跨工具的自适应推理。

**💡 创新点**

创新点包括：① 统一的高质量多轮工具交互数据生成管线；② 针对长时程工具规划的 Tool‑GRPO 强化学习框架；③ 通过令工具名称与参数随机化、语义重述的自适应学习（ADL）机制，提升模型对未见工具与任务的泛化能力。

**🔧 技术方法**

主要技术：Chain‑of‑Thought 生成、工具调用模板、GRPO‑based RL（Tool‑GRPO）、自适应学习（ADL）、多模态视觉工具集（感知、操作、计算）。

**📊 数据集**

使用的数据集：VSP（Visual Spatial Planning）、Jigsaw（图像拼图）、GUIQA、WebQA、Visual Search（V*、HRBench），并在 Qwen2.5‑VL‑3B/7B 预训练模型上进行微调与 RL。

**📈 对比分析**

与传统 SFT、单工具 RL 及现有专有模型对比，AdaReasoner 在 7B 版本上平均提升 38.66%（+24.9% 具体任务），在 VSP、Jigsaw 等任务上突破 GPT‑5 与 Claude Sonnet 4，近乎完美（97.64%）完成 VSP 任务。

**⚠️ 局限性**

局限性：① 对新工具的即时自适应仍不稳定，需进一步 RL 训练；② 主要关注视觉工具，其他模态工具的推广尚待验证；③ RL 过程计算成本高，模型仍受规模限制。

---

## 783. ExoGS: A 4D Real-to-Sim-to-Real Framework for Scalable Manipulation Data Collection

**arXiv ID:** 2601.18629 | [PDF](https://arxiv.org/pdf/2601.18629v1)

**作者:** Yiming Wang `[一作]` (Shanghai Jiao Tong University), Hao-Shu Fang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6180 | [OpenAlex ID](https://openalex.org/A5061763793)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过低成本的被动外骨骼 AirExo-3 采集人类操作者的高精度演示，并将其转化为 3D 高斯剖分（3DGS）可编辑的数字资产，构建可在仿真中无缝重放并进行大规模数据增强的 Real‑to‑Sim‑to‑Real 框架。

**💡 创新点**

① 引入机器人等价外骨骼实现“机器人‑free”的演示采集；② 将 3DGS 与可编辑资产结合，实现几何一致的大规模仿真数据增强；③ 设计轻量级 Mask Adapter，将实例级语义注入 ViT‑based 视觉策略，显著提升跨域泛化。

**🔧 技术方法**

使用的技术包括：
- 3D Gaussian Splatting 进行场景重建与渲染；
- AirExo‑3 外骨骼配备 12 位编码器实现毫米级关节跟踪；
- 多视角 RGB‑D 采集与 COLMAP 相机姿态估计；
- FoundationPose 进行物体姿态追踪；
- ViT‑based ACT 政策与 Mask Adapter（语义分割头 + 标签驱动注意力）。

**📊 数据集**

数据集：
- 通过 AirExo‑3 采集 60 条原始演示（每个任务）；
- 通过 3DGS 生成的可编辑资产进行视角、颜色、背景、姿态四种增强，扩充至 20 倍原始大小；
- 传统基准为 60 条远程操作（teleoperation）演示；
- 所有实验均在 Flexiv Rizon 4s + Intel RealSense D415 环境下进行。

**📈 对比分析**

对比方法：
- 数据采集效率：AirExo‑3 的采集时间比遥操作快 1–2 倍，成功率在所有任务上更高，尤其在螺帽拧开任务中遥操作成功率仅 17% 而 AirExo‑3 达 87%。
- 策略性能：
  * 未进行数据增强时，AirExo‑3 训练的策略在 Pick‑Place、Pick‑Place‑Close、Unscrew 任务上分别获得 50%，48% 和 24% 的成功率，略低于遥操作但在 Unscrew 任务上显著优于遥操作（24% vs 8%）。
  * 采用四种增强后，AirExo‑3 训练的策略在所有视觉扰动场景下均优于未增强的策略和遥操作策略，尤其在颜色、背景和光照变化下成功率提升 15–20%。
  * Mask Adapter 在未增强数据上即可把策略性能提升到与遥操作相当甚至更好，且对新物体（未见过的数字资产）保持 76% 成功率。

**⚠️ 局限性**

局限性：
- 3DGS 假设为刚体，无法很好建模可变形或柔性物体；
- 仍依赖高质量的多视角摄像头和 3DGS 渲染的光照精度，某些极端背景/灯光条件下语义分割与策略表现仍会下降；
- 需要手工标定外骨骼的零位，虽然易于部署但在长期使用中可能出现漂移。

---

## 784. K-Myriad: Jump-starting reinforcement learning with unsupervised parallel agents

**arXiv ID:** 2601.18580 | [PDF](https://arxiv.org/pdf/2601.18580v1)

**作者:** Vincenzo De Paola `[一作]` (Politecnico di Milano), Marcello Restelli `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种 K-Myriad 框架，利用多头共享网络在大量并行环境中通过无监督最大化集体状态熵来预训练多样化的策略，并在后续稀疏奖励任务中显著加速学习。

**💡 创新点**

创新点在于：①将并行策略建模为共享骨干+多头网络，实现可扩展的并行熵最大化；②使用 k‑NN 熵估计直接对状态分布进行梯度优化；③通过预训练多元化策略实现对不同任务的快速跳起。

**🔧 技术方法**

主要技术包括：政策梯度（policy gradient）与 k‑NN 熵估计、共享骨干+多头网络架构、Isaac Sim GPU 并行仿真、PPO 微调、KL 散度评估等。

**📊 数据集**

实验数据集为 Isaac Sim 中的高维连续控制任务——Ant 机器人在 Empty、Maze、Pyramid、Cave 四种地形环境中的模拟数据。

**📈 对比分析**

与单一通用熵最大化策略、随机初始化以及单体预训练策略的 PPO 进行对比；K-Myriad 在 50 代理时获得更高的状态熵和更快的成功率（如 Empty 环境 1.33±0.01，Maze 1.02±0.02），证明了多样化预训练的有效性。

**⚠️ 局限性**

局限性包括：在固定总采样预算下，代理数量增多导致单个代理的样本效率下降；训练接近收敛时策略趋于确定性，优势逐渐消失；大规模代理时需进一步平衡计算资源与采样成本。

---

## 785. Self-Refining Video Sampling

**arXiv ID:** 2601.18577 | [PDF](https://arxiv.org/pdf/2601.18577v1)

**作者:** Sangwon Jang `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (DeepAuto.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在推理阶段利用已有视频生成器自我改进的采样方法——Self‑Refining Video Sampling；

**💡 创新点**

创新点在于将流匹配目标重新解释为时变去噪自编码器，并利用迭代的 Predict‑and‑Perturb（P&P）循环在固定噪声水平下自我细化，同时通过不确定性感知掩码实现选择性改进，避免过度强化导致的伪影；

**🔧 技术方法**

技术包括：流匹配（Flow Matching）模型、去噪自编码器框架、P&P迭代细化、基于模型自一致性的误差估计与掩码生成、经典 ODE 求解器（UniPC）等；

**📊 数据集**

主要在公开的多种视频生成模型上验证：Wan2.1、Wan2.2、Cosmos‑2.5；同时在构造的 Dynamic‑bench、VideoJAM‑bench、PAI‑Bench、VideoPhy2、PhyWorldBench、PisaBench 等评测基准上进行测试；

**📈 对比分析**

与基线方法（默认 UniPC、CFG‑Zero、FlowMo、Verifier‑based 重采样等）比较，实验表明 P&P 在人类评估中获得 70‑73% 的偏好率，自动化评估指标（VBench、物理一致性、抓取成功率等）均显著提升；

**⚠️ 局限性**

局限性包括：对极端复杂或完全错误的内容仍难以修正，需额外外部验证器；过多迭代会导致模式收敛（缺失多样性）和背景过饱和；目前主要验证于大规模公开数据训练的模型，尚未评估在小数据或特定领域模型上的效果。

---

## 786. Stable Matching with Deviators and Conformists

**arXiv ID:** 2601.18573 | [PDF](https://arxiv.org/pdf/2601.18573v1)

**作者:** Frederik Glitzner `[一作]` (University of Glasgow), David Manlove `[通讯]` (University of Glasgow)

**通讯引用:** 4380 | [OpenAlex ID](https://openalex.org/A5050695548)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究了在存在可偏离者（deviators）与顺从者（conformists）的稳定匹配问题，探讨了在这些限制下匹配的存在性、最优性与阻塞对数量的复杂性，并给出了多种算法与复杂度分析；

**💡 创新点**

创新点包括：①首次将偏离者/顺从者模型引入稳定匹配框架并证明即使偏离者有限问题仍为NP‑完备；②提出针对偏离者数和偏好列表长度的FPT算法；③在偏好列表长度≤2时提供线性/多项式最优算法，形成完整的复杂度全景；

**🔧 技术方法**

采用的技术主要有：归约与构造证明（如从(2,2)-e3-SAT、k‑BP‑AlmostStable等），匹配理论算法（Gale–Shapley、Irving、Micali–Vazirani），参数化算法与最大权匹配求解，图分量与连通子图分析；

**📊 数据集**

本文为理论性研究，未使用实际数据集；所有结果均基于构造实例与理论证明；

**📈 对比分析**

与传统稳定匹配算法比较，本文指出一般情况下该问题为NP‑完备，但在偏离者数或偏好列表长度有限时可在多项式或线性时间内求解；对于列表长度≤2的实例，分别在O(n)和O(n²)时间内得到最优解；

**⚠️ 局限性**

局限性在于：仅对|D|与列表长度共同参数化给出FPT结果，尚未证明仅按|D|参数化可FPT；在完整偏好列表和大列表/多偏离者场景下问题仍NP‑完备；缺乏实验评估验证算法在实际数据上的表现。

---

## 787. Physics-Informed Uncertainty Enables Reliable AI-driven Design

**arXiv ID:** 2601.18638 | [PDF](https://arxiv.org/pdf/2601.18638v1)

**作者:** Tingkai Xue `[一作]` (National University of Singapore), My Ha Dao `[通讯]` (Technology Centre for Offshore and Marine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在逆向设计频率选择表面（FSS）时引入物理信息的不确定性量化，并将其嵌入多分辨率粒子群优化框架；

**💡 创新点**

创新点在于提出了基于物理法则（电磁场连续性）计算的低成本不确定性度量PHY‑UNC，并证明其与传统集成学习方法（ENSB‑UNC）在指导优化时同等有效；

**🔧 技术方法**

技术包括：基于WideResNet的深度学习预测器、物理法则推导的不确定性度量、单/多分辨率粒子群优化（BPSO）以及高低分辨率求解器交替评估策略；

**📊 数据集**

数据集为10,000个18×18像素、八向对称的Meta‑atom设计，通过高精度HFSS求解得到20–30 GHz区间的S‑参数；

**📈 对比分析**

与传统单精度BPSO、仅使用低精度预测或ENSEMBLE不确定性的方法比较，实验显示多分辨率PHY‑UNC策略将成功率从<10%提升至>50%（band‑stop）或100%（band‑pass），且计算成本约为单精度BPSO的1/10；

**⚠️ 局限性**

局限性包括：PHY‑UNC基于单一物理约束，可能无法捕捉更复杂系统的误差；数据集仍有限，无法覆盖全部10^13种可能设计；未来需引入更丰富的物理约束或主动学习扩展训练集。

---

## 788. Brazilian Social Media Anti-vaccine Information Disorder Dataset -- Telegram (2020-2025)

**arXiv ID:** 2601.18622 | [PDF](https://arxiv.org/pdf/2601.18622v1)

**作者:** João Phillipe Cardenuto `[一作]` (Recod.ai), Anderson Rocha `[通讯]` (Recod.ai)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过 Telegram API 对 119 个公开反疫苗频道的内容进行爬取，构建了约 400 万条帖子的数据集，并对其中的疫苗相关信息进行了自动标注。

**💡 创新点**

创新点在于提供了一个规模宏大、包含文本、媒体与传播指标（转发量、浏览量等）的公开数据集，且首次结合大语言模型对海量社交媒体内容进行疫苗相关性筛选，填补了当前研究对抗疫苗信息失真的数据缺口。

**🔧 技术方法**

使用了 Telethon 库实现对 Telegram 的高效抓取，利用 langdetect 进行语言识别，采用 Sabiá‑3 大语言模型完成疫苗相关文本的二分类，并通过脚本更新转发与浏览指标。

**📊 数据集**

所用数据集来自 119 个公开反疫苗 Telegram 频道，时间范围为 2020 年 1 月至 2025 年 6 月，包含 400 万条文本帖子、144 万条媒体文件（图片、视频、音频等），共计 5.5 TB 大小。

**📈 对比分析**

在 600 条人工标注样本上，Sabiá‑3 的 F1 值达 90%，与三名专家的平均 Cohen’s Kappa 0.866 相符，表明该自动标注方法在实际应用中具有较高的准确性和可重复性。

**⚠️ 局限性**

局限性包括仅采集公开频道导致信息覆盖不足，媒体文件被限制在 50 MB 以内导致部分大文件缺失；Telegram API 的速率限制与媒体下载耗时导致抓取周期长；以及由于隐私与法律原因，部分敏感信息被删除或隐藏，影响数据完整性。

---

## 789. Scale-Aware Self-Supervised Learning for Segmentation of Small and Sparse Structures

**arXiv ID:** 2601.18619 | [PDF](https://arxiv.org/pdf/2601.18619v1)

**作者:** Jorge Quesada `[一作]` (Georgia Institute of Technology), Ghassan AlRegib `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5518 | [OpenAlex ID](https://openalex.org/A5006145139)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出在自监督预训练中加入小窗口裁剪的尺度感知策略，以提升对小而稀疏结构的分割性能

**💡 创新点**

创新点在于将尺度感知裁剪直接嵌入数据增强管线，形成对细粒度特征的诱导偏置，无需改动网络结构或额外监督

**🔧 技术方法**

使用了对比式 SSL（SimCLR、BYOL、VICReg）和基于 ResNet‑18 的编码器；在下游采用 DeepLabV3 解码器和 Dice 损失进行微调

**📊 数据集**

在地震成像的断层分割（使用公开的断层与地层数据集）以及神经影像的细胞与血管分割（公开脑成像数据集）上进行实验

**📈 对比分析**

与传统全图 SSL、VICRegL 多裁剪方案及仅监督基线对比，尺度感知 SSL 在断层分割上提升 Dice 最高 13%，细胞分割提升 5%；对大尺度结构如地层与轴突几乎无提升甚至略降

**⚠️ 局限性**

局限性包括：仅对小而稀疏目标有效，可能对大尺度目标产生负面影响；需要根据任务调节裁剪尺寸与策略；实验仅覆盖两个领域，尚未验证在更广泛数据集上的泛化

---

## 790. An Unsupervised Tensor-Based Domain Alignment

**arXiv ID:** 2601.18564 | [PDF](https://arxiv.org/pdf/2601.18564v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 791. A Balanced Neuro-Symbolic Approach for Commonsense Abductive Logic

**arXiv ID:** 2601.18595 | [PDF](https://arxiv.org/pdf/2601.18595v1)

**作者:** Joseph Cotnareanu `[一作]` (McGill University), Mark Coates `[通讯]` (McGill University)

**通讯引用:** 6904 | [OpenAlex ID](https://openalex.org/A5009031715)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种神经符号混合方法ARGOS，用LLM在逻辑求解器的反馈下迭代生成常识假设，从而完成缺失背景知识的证明规划。

**💡 创新点**

创新点在于：①利用逻辑求解器的backbone信息引导前向搜索；②通过LLM评分判断生成的常识假设是否可信且与上下文相关；③不受限于小范围规则，支持任意未出现过的命题。

**🔧 技术方法**

技术包括：大语言模型（Llama3‑8B/70B、Mistral‑7B）+ SAT求解器(Cadical)、自一致性推理、基于backbone的前向链式搜索、两阶段LLM评分。

**📊 数据集**

评测数据集为改造后的ProntoQA、CLUTRR、FOLIO、CosmosQA、QUAIL、ESNLI，均转为真/假形式。

**📈 对比分析**

与COT、Self‑Consistency、SAT‑LM、LoT‑20、LLM‑Tres等基线比较，ARGOS在所有数据集上显著提升，最大提升约+13%，在FOLIO、CLUTRR、QUAIL等结构化或模糊推理任务中表现尤为突出。

**⚠️ 局限性**

局限包括：仅处理真/假问题；前向链式限制为最多两前件；需访问LLM logits；对翻译质量敏感；在自一致性产生幻觉时可能受影响；未实现后向链式或多选问题的完整支持。

---

## 792. From Classification to Ranking: Enhancing LLM Reasoning Capabilities for MBTI Personality Detection

**arXiv ID:** 2601.18582 | [PDF](https://arxiv.org/pdf/2601.18582v1)

**作者:** Yuan Cao `[一作]` (Institute of Computing Technology), Qiang Qiu `[通讯]` (Institute of Computing Technology)

**通讯引用:** 3710 | [OpenAlex ID](https://openalex.org/A5101992408)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了将人格检测视为列表式排名任务的LLM微调框架PerDet-R1，并通过两阶段训练（SFT+GRPO）实现对MBTI四维度的联合建模与排名推理；

**💡 创新点**

创新点包括①把MBTI四维度视为整体排名而非独立二分类，消除维度分离导致的认知失真；②设计基于NDCG与维度相似度的奖励函数，兼顾正确性与排名顺序，避免奖励稀疏与作弊；③采用SFT+GRPO两阶段训练，利用教师模型的推理轨迹进行离策略蒸馏并在RL阶段自适应探索，减少对人工提示的依赖；

**🔧 技术方法**

技术方案涵盖：大型语言模型（Qwen-plus、Llama3-8B、Qwen2.5-7B）+监督微调（SFT）+ Group Relative Policy Optimization（GRPO）+ NDCG+维度相似度奖励函数+ rejection sampling+ LoRA微调框架；

**📊 数据集**

使用公开的Kaggle（MBTI论坛）和PANDORA（Reddit）两套社交媒体帖子与MBTI标签数据集；

**📈 对比分析**

与SVM、XGBoost、BERT、D-DGCN、ChatGPT、Qwen-plus、PsyCoT、TAE、ETM等基线在Macro‑F1（4维二分类）、F1‑Score（16类型多分类）及NDCG@3指标上进行对比；PerDet‑R1在Kaggle上Macro‑F1提升约2.8%，F1‑Score提升约8.8%；在PANDORA上Macro‑F1提升约0.8%，F1‑Score提升约5%；在排名任务上NDCG@3均为最高；

**⚠️ 局限性**

局限性包括：仅依赖文本数据，未加入多模态信息；奖励函数设计对模型与超参数敏感，需手动调优；对信息量极低或标签稀疏的社交帖子表现仍受限；模型在偏见、隐私与伦理风险方面未做系统评估。

---

## 793. On the Abolition of the "ICSE Paper" and the Adoption of the "Registered Proposal" and the "Results Report"

**arXiv ID:** 2601.18566 | [PDF](https://arxiv.org/pdf/2601.18566v1)

**作者:** Fabio Massacci `[一作]` (University of Trento and Vrije University Amsterdam), Winnie Mbaka `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出废除传统ICSE技术论文模式，改为两层出版体系：首先提交“Registered Proposal”，随后任何人可提交“Results Report”对其进行独立验证。

**💡 创新点**

创新点在于将新颖性与可复制性分离，鼓励多团队重复实验并公开负面结果，旨在打破“新颖性-严谨”与“复制性-新颖性”之间的恶性循环，提升软件工程研究的可靠性与工业影响力。

**🔧 技术方法**

采用注册报告（Registered Report）和成果报告（Results Report）的出版机制，配合同行评议、实验包（artifact）标准化与可执行化。

**📊 数据集**

论文并未使用具体实验数据集，而是基于先前调查与社区讨论收集的意见，强调未来可用已有的公开数据集、工具或案例进行验证。

**📈 对比分析**

未给出实验比较与性能评估；该提案属于方法学与出版模式的设想，后续需通过实际会议与论文采纳来检验其效果。

**⚠️ 局限性**

局限性包括：社区接受度未知、实施成本与流程复杂、对审稿人偏见与资源不足的担忧，以及可能导致评审工作负荷增加。

---

## 794. AI-enabled Satellite Edge Computing: A Single-Pixel Feature based Shallow Classification Model for Hyperspectral Imaging

**arXiv ID:** 2601.18560 | [PDF](https://arxiv.org/pdf/2601.18560v1)

**作者:** Li Fang `[一作]` (Chinese Academy of Sciences), Wei Yao `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 9490 | [OpenAlex ID](https://openalex.org/A5100643673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在卫星边缘计算平台上实现轻量化的高光谱图像分类方法，采用单像素谱特征的两阶段像素级标签传播，支持少样本学习并具备噪声鲁棒性。

**💡 创新点**

① 仅利用单像素内在光谱特征，避免空间结构信息导致的误判；② 引入两阶段标签传播：先通过锚点传播初始软标签，再利用稀疏图与闭式解精细更新；③ 用秩约束图聚类自动生成锚点标签，消除人工标注需求。

**🔧 技术方法**

锚点图构建、Gaussian核相似度、稀疏kNN图、闭式标签传播、秩约束图聚类、soft标签学习、少样本/伪标签生成等。

**📊 数据集**

Indian Pines、Salinas、Pavia University 三个公开高光谱数据集。

**📈 对比分析**

与 CNN-PPF、GFHF、PL、FSS、SGL、STSE-DWLR、DSSPL 等多种方法比较，实验表明在三数据集上均实现了更高的 OA、AA 与 Kappa，尤其在 Indian Pines 数据集上 OA 提升超过 11%，在噪声环境下仍保持较强的鲁棒性。

**⚠️ 局限性**

① 仍存在图结构构建与闭式解计算的时间开销，限制了完全实时性能；② 锚点选择对类别不平衡敏感，少量锚点可能导致稀有类别失配；③ 仅基于单像素谱特征，对混合像素和空间相关信息的利用有限，可能在复杂场景下略逊。

---

## 795. Generative Diffusion Augmentation with Quantum-Enhanced Discrimination for Medical Image Diagnosis

**arXiv ID:** 2601.18556 | [PDF](https://arxiv.org/pdf/2601.18556v1)

**作者:** Jingsong Xia `[一作]` (Nanjing Medical University), Siqi Wang `[通讯]` (Nanjing Medical University)

**通讯引用:** 3795 | [OpenAlex ID](https://openalex.org/A5100420069)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 SDA-QEC 框架，通过简化扩散增强生成少数类合成样本并在 MobileNetV2 后加入轻量量子特征层，针对冠状动脉造影图像分类任务解决了严重的类别不平衡问题。

**💡 创新点**

创新点包括：① 仅利用前向扩散噪声（5 步）快速生成高质量少数类样本，省去逆向网络计算；② 将 4 量子比特 2 层量子电路嵌入 CNN 后端，实现低参数、高非线性映射；③ 通过数据层与模型层的协同优化，实现 98.33% 的准确率、召回率与特异性，展示了“数据重构 + 特征提升”双重优化的有效性。

**🔧 技术方法**

使用技术：简化前向扩散模型、Pennylane 量子特征层、MobileNetV2 轻量 CNN、交叉熵+L2 正则、Adam 优化、Bootstrap 置信区间评估、FID 质量评估。

**📊 数据集**

数据集：冠脉造影图像分类数据集，样本总量 120 张（60 正例、60 负例），用于训练、验证和测试。

**📈 对比分析**

对比方法：ResNet18、MobileNetV2、DenseNet121、VGG16 等经典基线。SDA-QEC 在准确率、精确率、召回率、特异性、F1 及 AUC 上均优于基线，最高指标均达到 98.33% 或 98.78%，并在 Bootstrap 分析中表现出极低的方差，表明模型稳定可靠。

**⚠️ 局限性**

局限性：① 仅在单一数据集上验证，需在更大规模、多中心或多模态数据上进一步测试；② 简化扩散仅采用前向噪声，可能不足以捕捉极其复杂的少数类分布；③ 量子层在现有量子硬件上的实现受限，实际部署仍需硬件支持；④ 对极端极小样本或更高类别不平衡比例的鲁棒性尚未验证。

---

## 796. Issues regarding the Indexing of Publication Types and Study Designs

**arXiv ID:** 2601.18616 | [PDF](https://arxiv.org/pdf/2601.18616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 797. Fast and Safe Trajectory Optimization for Mobile Manipulators With Neural Configuration Space Distance Field

**arXiv ID:** 2601.18548 | [PDF](https://arxiv.org/pdf/2601.18548v1)

**作者:** Yulin Li `[一作]` (Robotics and Autonomous Systems Thrust, Hong Kong University of Science and Technology), Jun Ma `[通讯]` (Robotics and Autonomous Systems Thrust, Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于神经网络的广义配置空间距离场（GCDF）用于移动操控器的快速安全轨迹优化，解决了在复杂环境中进行全身轨迹优化的挑战。

**💡 创新点**

创新点在于将配置空间距离场（CDF）扩展到移动操控器，提出GCDF，能够在无界工作空间中处理平移和旋转关节的机器人，同时保持准确的几何表示和优化友好的碰撞成本。

**🔧 技术方法**

使用了神经网络生成和训练管道，开发了高性能的C++顺序凸优化框架，支持基于GCDF的碰撞推理，并通过GPU批量查询实现高效计算。

**📊 数据集**

使用了随机生成的高密度障碍物基准测试和真实机器人实验，验证了所提出方法的有效性和优越性。

**📈 对比分析**

与强基线方法相比，GCDF约束的轨迹优化算法在成功率、轨迹质量和求解时间上表现出一致的优越性，能够快速、安全、可靠地进行全身规划。

**⚠️ 局限性**

限制在于训练阶段未充分探索网络架构选择和超参数对学习到的隐式GCDF的影响，且在处理复杂障碍物几何时，计算复杂度可能仍然受到碰撞约束规模的主导。

---

## 798. Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation

**arXiv ID:** 2601.18623 | [PDF](https://arxiv.org/pdf/2601.18623v1)

**作者:** Zihao Wang `[一作]` (University of Tennessee), Shaogang Ren `[通讯]` (University of Tennessee at Chattanooga)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种在反向扩散过程中动态嵌入跨模态域偏移的模型，利用可学习的空间变换混合字段在每一步引导生成，显著减少去噪步骤。

**💡 创新点**

创新点在于：①引入空间可变混合场Λ_t以实现局部、通道级别的域转移；②基于能量最小化的连续时间SDE推导，给出解析解并构造一阶采样器；③将域偏移作为内置的恢复漂移，使采样沿低能量路径前进，降低语义漂移和计算成本。

**🔧 技术方法**

使用条件扩散模型（VP式）+ SDE + 学习的空间变换调度 + 解析解+ 一阶采样器 + 位置编码 + 轻量卷积网络预测Λ_t。

**📊 数据集**

实验数据集包括：IXI（T1↔T2医学成像）、Sentinel-1/2（SAR→光学遥感）以及PSCDE（电致发光→语义掩膜）三组跨模态翻译任务。

**📈 对比分析**

与Pix2Pix、ABridge、DOSSR、BBDM、DBIM等基线比较，CDTSDE在SSIM、PSNR、MSE、Dice等指标上均领先或位居前列，同时在相同质量下只需 1–5 步去噪，速度提升约 2×，训练步数也更少。

**⚠️ 局限性**

局限性：在相似模态（如IXI）提升有限；对未配对数据的适用性尚未验证；生成的图像在感知锐度上仍不及部分GAN基线；模型需要额外的空间调度网络，增加了训练复杂度。

---

## 799. Emergence of Phonemic, Syntactic, and Semantic Representations in Artificial Neural Networks

**arXiv ID:** 2601.18617 | [PDF](https://arxiv.org/pdf/2601.18617v1)

**作者:** Pierre Orhan `[一作]` (Paris Brain Institute), Jean-Rémi King `[通讯]` (Meta AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对自监督文本和语音模型的内部激活进行线性探针分析，跟踪语音学、词汇语义和句法结构的出现顺序。

**💡 创新点**

将结构探针从句法推广到语音和词汇语义，并在训练过程中动态评估其出现阶段，揭示模型与儿童语言发展相似但数据效率差异。

**🔧 技术方法**

使用自监督学习的LLama2、Pythia、Wav2Vec2.0模型；应用结构探针（线性投影）配合Spearman相关和最小生成树等评估方法。

**📊 数据集**

采用UD–EWT语法树、WordNet名词子图以及合成语音的TTS语料，并对音频进行对齐。

**📈 对比分析**

通过Spearman相关评估结构探针与语音学/语义/句法目标距离的匹配度，发现模型按音素→词汇→句法顺序提升，但所需训练数据比儿童高2–4个数量级。

**⚠️ 局限性**

缺乏对句法与词汇语义先后顺序的明确结论；只评估距离匹配而非下游任务；只对名词和音素做探测；未探测文本模型音素结构；数据效率与人类差距大。

---

## 800. AGSP-DSA: An Adaptive Graph Signal Processing Framework for Robust Multimodal Fusion with Dynamic Semantic Alignment

**arXiv ID:** 2601.18589 | [PDF](https://arxiv.org/pdf/2601.18589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 801. LaCoGSEA: Unsupervised deep learning for pathway analysis via latent correlation

**arXiv ID:** 2601.18604 | [PDF](https://arxiv.org/pdf/2601.18604v1)

**作者:** Zhiwei Zheng `[一作]` (University of Glasgow), Kevin Bryson `[通讯]` (University of Glasgow)

**通讯引用:** 10588 | [OpenAlex ID](https://openalex.org/A5063296381)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了LaCoGSEA，一种无监督深度学习框架，利用自编码器提取非线性表达模式，并通过全局基因–潜在维度相关性生成预排序基因列表，随后使用经典GSEA进行通路富集分析。

**💡 创新点**

创新点在于：① 结合无监督深度表征与全局相关性预排序，突破传统需要标签或单样本评分的局限；② 用Pearson相关性代替稀疏XAI方法，更贴合通路协同表达特性；③ 通过深度模型捕获非线性依赖，显著提升通路检索与亚型分层能力。

**🔧 技术方法**

主要技术包括：深度自编码器（带Elastic Net正则化）、Pearson相关矩阵、GSEA、t‑SNE可视化、ARI评估、K‑means聚类、以及与PCA、SHAP、DeepLIFT、GSVA、ssGSEA、标准DE等方法的对比。

**📊 数据集**

使用的数据集涵盖三大癌症数据集（SCAN‑B、METABRIC、TCGA‑Lung）和五个独立GEO数据集（GSE10846、GSE48350、GSE11375、GSE126848、GSE116250），共计约 26,000 个基因、超过 7,000 个样本。

**📈 对比分析**

与PCA（相关权重/加载）、梯度XAI（SHAP、DeepLIFT）、标准差异表达（DE）以及单样本富集方法（GSVA、ssGSEA）比较，LaCoGSEA在：① 通路检索（平均目标路径排名≈17.5，覆盖率95%）② 亚型聚类（SCAN‑B 上ARI=0.372，优于PCA 0.240、GSVA 0.126、ssGSEA 0.185）③ 小样本（N≈30）中显著识别关键通路，均表现出色。

**⚠️ 局限性**

局限性包括：① 仅捕获数据中主变异，可能与临床结局不完全一致；② 对高度重叠或多效基因集解释仍有挑战；③ 依赖参考数据库完整性；④ 需要手动选择潜在维度；⑤ 作为无监督方法，需进一步关联临床标签或外部验证。

---

## 802. EFSI-DETR: Efficient Frequency-Semantic Integration for Real-Time Small Object Detection in UAV Imagery

**arXiv ID:** 2601.18597 | [PDF](https://arxiv.org/pdf/2601.18597v1)

**作者:** Yu Xia `[一作]` (Wuhan University), Zhigang Tu `[通讯]` (Wuhan University)

**通讯引用:** 3362 | [OpenAlex ID](https://openalex.org/A5074405661)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了EFSI-DETR框架，针对无人机图像中的实时小目标检测进行改进

**💡 创新点**

创新点包括DyFusNet融合频域与空间信息的动态多分辨率分解以及轻量化的ESFC语义特征聚焦模块

**🔧 技术方法**

采用动态频域分解、深度卷积、GhostBlock、SFCM等技术实现高效特征融合与语义增强

**📊 数据集**

在VisDrone和CODrone这两个无人机视觉基准数据集上进行训练与评估

**📈 对比分析**

与最新SOTA方法相比，VisDrone上的AP提升至33.1%（+1.6%），AP_s提升5.8%，并保持188FPS；CODrone上AP提升至20.2%，显著优于YOLO系列和RT-DETR

**⚠️ 局限性**

对大目标的检测效果相对较弱，且在不同尺度间的性能平衡仍有提升空间

---

## 803. How are MLOps Frameworks Used in Open Source Projects? An Empirical Characterization

**arXiv ID:** 2601.18591 | [PDF](https://arxiv.org/pdf/2601.18591v1)

**作者:** Fiorella Zampetti `[一作]` (University of Sannio), Massimiliano Di Penta `[通讯]` (University of Sannio)

**通讯引用:** 19707 | [OpenAlex ID](https://openalex.org/A5025099559)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统地分析了八个流行开源 MLOps 框架（BentoML、Deepchecks、Evidently AI、Kedro、Metaflow、MLFlow、Prefect、Wandb）在 GitHub 上的实际使用方式、调用频次、与 CI/CD 的集成情况，并挖掘并分类了这些框架 issue 跟踪器中提交的功能请求，进一步探讨使用模式与社区需求的匹配关系。

**💡 创新点**

创新点主要有三：①首次以大规模的 GitHub 数据为基础，量化 MLOps 框架在真实项目中的使用模式；②提出并验证了一个两级功能请求分类法（高层与子类），揭示了使用与需求之间的对齐度；③系统识别并描述了多框架协同使用的典型模式，为工具设计和实践者提供了参考。

**🔧 技术方法**

技术手段包括：GitHub API 和 dependency‑graph 抓取；Python AST 分析提取框架调用；对 CI/CD 工作流的 YAML 解析；人工与 LLM（Gemini 2.5 Pro）混合的双轮分类；统计与可视化分析（频次、占比、共用模式）。

**📊 数据集**

数据集：约 2,857 个经过星标、Python 语言过滤后的 GitHub 依赖项目；以及 8 个框架的 4,075 条已闭合的 issue，涵盖 1,024+ 个功能请求，分别来自每个框架。数据通过公开 API 与爬取方式获得，并在复制包中公开。

**📈 对比分析**

比较与评估方法：通过对各框架调用次数、CI/CD 集成比例、功能覆盖率等指标进行量化；将功能请求的子类频次与对应被使用功能进行映射，计算匹配度；未进行性能（如执行时间、资源消耗）实验，而是侧重于使用频次与需求一致性的统计。整体发现：MLOps 框架多以 API 方式使用，CI/CD 集成不足，功能请求大多聚焦已使用核心功能的改进。

**⚠️ 局限性**

局限性：①仅覆盖公开 GitHub Python 项目，未考察私有或本地使用情况；②依赖图与代码解析的准确性可能导致漏检或误检；③功能请求分类仍有人工误差，LLM 辅助虽提升一致性但不可完全替代；④未对商业闭源框架或行业大规模应用进行验证，结果可能不具代表性；⑤缺乏对框架性能（速度、资源）等技术细节的深入评估。

---

## 804. FastInsight: Fast and Insightful Retrieval via Fusion Operators for Graph RAG

**arXiv ID:** 2601.18579 | [PDF](https://arxiv.org/pdf/2601.18579v1)

**作者:** Seonho An `[一作]` (KAIST), Min-Soo Kim `[通讯]` (KAIST)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出FastInsight，一种针对语料库图的Graph RAG检索方法，能够快速且具洞察力地检索相关节点并提升生成质量。

**💡 创新点**

创新点在于提出两种融合运算符：Graph-based Reranker (GRanker) 用图结构去噪并结合语义评分，Semantic-Topological eXpansion (STeX) 在向量搜索基础上加入拓扑扩展；通过交替使用这两种运算符实现高效且洞察式检索。

**🔧 技术方法**

使用的技术包括：基于向量的检索、图搜索、模型驱动检索；GRanker利用图拉普拉斯正则化的第一阶近似；STeX结合结构重要性与语义相似度的加权得分；实验中使用OpenAI的text-embedding-3-small、gpt-5-mini、bge-reranker-v2-m3等模型。

**📊 数据集**

实验数据集覆盖两类语料库图：引用网络ACL‑OCL、LACD以及文本丰富的知识图BSARD‑G、SciFact‑G、NFcorpus‑G；并在UltraDomain的农业与混合域数据集上进行生成实验。

**📈 对比分析**

与多种基线（Vector Search、SPLADE、Contriever、HyDE、Re2、LightRAG、PathRAG、HippoRAG、GAR等）比较，FastInsight在R@10上平均提升约9.9%，nDCG@10提升约9.1%；在查询处理时间上相较于IRCoT等交互式方法减少42–58%并提升R@10；在生成任务上平均win rate超过55%。

**⚠️ 局限性**

局限性包括：仍需在更大规模、不同类型图结构（如社交网络、知识图谱）上验证；对超参数（α、β）的敏感性需进一步自动化；以及在极端稀疏图或高维向量空间中的性能尚未彻底评估。

---

## 805. Attention-Based Neural-Augmented Kalman Filter for Legged Robot State Estimation

**arXiv ID:** 2601.18569 | [PDF](https://arxiv.org/pdf/2601.18569v1)

**作者:** Seokju Lee `[一作]` (Korea Advanced Institute of Science and Technology), Kyung-Soo Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 6773 | [OpenAlex ID](https://openalex.org/A5040980904)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于注意力机制的神经增量卡尔曼滤波器AttenNKF，用于腿式机器人的实时状态估计，重点解决足部滑移导致的误差；

**💡 创新点**

创新点在于将滑移程度作为注意力查询，通过跨注意力在InEKF隐空间与滑移上下文之间实现滑移条件的可调补偿，并采用两阶段训练与隐空间编码，显著提升对不同滑移场景的适应性；

**🔧 技术方法**

使用的技术包括Invariant Extended Kalman Filter（InEKF）、GRU自编码器、跨注意力（cross‑attention）机制、残差式补偿、Lie群数学运算、以及多种损失函数（MSE、KL、教师‑学生监督）等；

**📊 数据集**

实验数据集来自Isaac Gym仿真中随机地形与动力学的Unitree Go1机器人，室内真实测试场景包括碎石、Teflon、楼梯和软地等，户外则使用约100 m草地轨迹，真值通过Vicon和RTK‑GPS获取；

**📈 对比分析**

与Slip Rejection、Learned Contact、InNKF等基线比较，AttenNKF在室内多滑移地形的相对误差下降30–90%，特别是位置误差显著减小；在户外100 m草地上误差降低55–73%，并保持580 Hz的实时率；

**⚠️ 局限性**

局限性包括仅校正均值而不更新协方差，导致长期累计误差仍存在；对训练之外的极端地形鲁棒性有限；对滑移检测的依赖与训练样本需求较大；未将补偿反馈至滤波递归中。

---

## 806. Automated Landmark Detection for assessing hip conditions: A Cross-Modality Validation of MRI versus X-ray

**arXiv ID:** 2601.18555 | [PDF](https://arxiv.org/pdf/2601.18555v1)

**作者:** Roberto Di Via `[一作]` (University of Genoa), Irina Voiculescu `[通讯]` (University of Oxford)

**通讯引用:** 1172 | [OpenAlex ID](https://openalex.org/A5055611196)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

验证了在三维MRI扫描中使用热图回归模型进行髋关节标记检测，并与传统X光成像的同类检测结果进行对比，证明MRI在α角和LCE角测量上能够达到与X光相当的准确性；

**💡 创新点**

首次在同一患者匹配数据集中完成跨模态验证，证明MRI可实现与X光同等的自动化FAI评估，为将自动化FHA分析引入常规MRI工作流奠定基础；

**🔧 技术方法**

采用UNet+++ResNet18热图回归网络，并结合数据增强、测试时增强（TTA）等技术；评估指标包括MRE、SDR、ICC、Bland–Altman等；

**📊 数据集**

使用89例FAI患者的配对数据集，包含AP pelvic X光和多切片T1加权冠状MRI，所有标注统一在中间切片；

**📈 对比分析**

在相同的测试集上比较，MRI整体MRE为2.98±0.23mm，X光为3.02±0.10mm；α角MAE约12°/13°，LCE角MAE约3°/2.5°；两模态在α>65°的诊断准确率均为87.5%，ICC LCE角分别为0.82和0.73；

**⚠️ 局限性**

仅使用单一MRI切片，未充分利用体积信息；α角测量受cam点定位不确定性影响，导致可靠性偏低；样本量相对较小，未实现自动切片选择或不确定性量化。

---

## 807. Constraint-Aware Discrete-Time PID Gain Optimization for Robotic Joint Control Under Actuator Saturation

**arXiv ID:** 2601.18639 | [PDF](https://arxiv.org/pdf/2601.18639v1)

**作者:** Ojasva Mishra `[一作]` (Downingtown STEM Academy), Min Xu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12223 | [OpenAlex ID](https://openalex.org/A5100413849)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了机器人关节在离散时间 PID 控制下，考虑饱和、采样、延迟和测量误差等实际限制的性能与调参流程。

**💡 创新点**

创新点包括：① 用 Jury 判据推导 PI 控制的离散时间稳定性区域（Euler 与 ZOH）；② 在饱和占优情形下评估离散后向计算抗风up 的效果；③ 设计混合认证的 Bayesian 优化工作流，先用解析式和短时安全检查剔除不安全增益，再进行鲁棒评估。

**🔧 技术方法**

使用了离散时间系统分析、Jury 判据、离散后向计算抗风up、贝叶斯优化、模拟仿真随机模型族以及基于 IAE 的稳健目标函数。

**📊 数据集**

通过对参数（τ、K）、延迟、噪声、量化、饱和限制等进行随机采样生成的模拟模型族，以及第二阶关节模型的扩展基准；未使用真实硬件数据。

**📈 对比分析**

与手工基准、鲁棒调优以及 Safe‑BO 进行对比实验。鲁棒调优将中位数 IAE 从 0.843 降至 0.430，且中位数超调低于 2%；Safe‑BO 在给定评估次数内实现更低目标，并将不安全评估率降至 11.6%。

**⚠️ 局限性**

主要局限在于全部基于模拟，未验证到真实机器人；分析仅覆盖 PI 的离散稳定性；第二阶模型仍为简化抽象，缺乏对更高阶动态、温漂等未知效应的考量。

---

## 808. Assessing the Quality of Mental Health Support in LLM Responses through Multi-Attribute Human Evaluation

**arXiv ID:** 2601.18630 | [PDF](https://arxiv.org/pdf/2601.18630v1)

**作者:** Abeer Badawi `[一作]` (York University), Elham Dolatabadi `[通讯]` (Vector Institute)

**通讯引用:** 990 | [OpenAlex ID](https://openalex.org/A5061175035)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对500个真实心理咨询会话生成并评估了9种LLM的回复，采用双专家5分量表评价6项治疗质量。

**💡 创新点**

提出人本化多维评价框架，聚焦认知支持与情感共鸣，系统揭示认知-情感差距。

**🔧 技术方法**

使用基于prompt的LLM生成，双盲专家评分，5分Likert量表和标准化统计分析。

**📊 数据集**

选取 MentalChat16K、EmoCare、CounselChat 共500条会话作为评估基准。

**📈 对比分析**

对9款模型按认知、情感六维量化评分，GPT‑4o及Gemini最高，闭源模型平均得分≥4.5，开源模型多在情感维度偏低，存在明显低分比例。

**⚠️ 局限性**

仅单轮对话、样本量有限、缺乏多文化覆盖、未验证真实临床效果，评估仍依赖人工专家。

---

## 809. CONQUER: Context-Aware Representation with Query Enhancement for Text-Based Person Search

**arXiv ID:** 2601.18625 | [PDF](https://arxiv.org/pdf/2601.18625v1)

**作者:** Zequn Xie `[一作]` `[通讯]` (Zhejiang University), Zequn Xie (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出两阶段框架 CONQUER，训练阶段通过 CARE 加强跨模态表征，推理阶段使用 IQE 主动改写模糊查询以提升文本检索式行人搜索性能。

**💡 创新点**

创新点在于 CARE 结合多粒度编码、互补对挖掘与基于 Optimal Transport 的上下文引导对齐显著提升跨模态匹配；IQE 则为可插拔的推理阶段查询增强模块，可在不重新训练模型的情况下根据锚点和属性主动改写查询。

**🔧 技术方法**

采用 CLIP ViT‑B/16 视觉/文本编码器、多粒度特征提取、互补负样本挖掘、Sinkhorn 迭代求解 Optimal Transport、KL 散度对齐、MLLM 交互式问答与属性合成等技术。

**📊 数据集**

主要实验数据集包括 CUHK‑PEDES、ICFG‑PEDES 与 RSTPReid 三大文本‑图像检索基准。

**📈 对比分析**

与多种最新方法对比，CONQUER 在三组数据集上均实现最高或接近最高的 Rank‑1、Rank‑5 与 mAP 指标，并在跨域与缺失查询场景表现出显著优势。

**⚠️ 局限性**

主要局限在于推理阶段仍需额外锚点检索与 LLM 交互导致推理时延增加，以及属性抽取不准确时可能导致查询误改。

---

## 810. Multimodal Privacy-Preserving Entity Resolution with Fully Homomorphic Encryption

**arXiv ID:** 2601.18612 | [PDF](https://arxiv.org/pdf/2601.18612v1)

**作者:** Susim Roy `[一作]`, Nalini Ratha `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在全同态加密域中完成的多模态实体解析框架，利用图像和文本编码器生成的生物特征和生物识别特征，并通过加密的模板匹配实现身份核实。

**💡 创新点**

创新点在于：①在加密域实现多模态特征融合（特征级和得分级），②使用RNS‑CKKS FHE对模板进行安全存储和计算，③公开了一份规模达36,661人、287,532条记录的合成实体解析数据集。

**🔧 技术方法**

主要技术包括：CLIP ViT‑B/32预训练的图像/文本编码器、两段可训练MLP（BMMLP、BGMLP）产生得分向量、Min‑Max归一化、RNS‑CKKS FHE（HEAAN库）、多项式逼近实现加密域的比较函数。

**📊 数据集**

使用的是合成实体解析数据集：36,661名独立个体，287,532条记录，包含姓名、地址变体、年龄变化头部图像等多模态信息。

**📈 对比分析**

与单模态基线（仅生物特征或仅生物识别）对比，特征级/得分级融合在加密域与明文结果一致，EER从单模态的7.78%/12.37%下降到4.08%/5.17%，ROC/DET曲线显示更高TPR，CMC曲线表明特征级融合在Rank‑1识别上可达75.86%，得分级在Rank‑5上略优。加密运算通过多线程可实现4.42×的加速。

**⚠️ 局限性**

局限性包括：①评估仅基于合成数据，缺乏真实世界大规模数据验证；②FHE计算仍存在较高延迟和对非线性操作的多项式逼近依赖；③融合效果受限于所用预训练模型与数据质量；④未探究多模态融合在更复杂攻击场景下的鲁棒性。

---

## 811. PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression

**arXiv ID:** 2601.18608 | [PDF](https://arxiv.org/pdf/2601.18608v1)

**作者:** Fabian Fumagalli `[一作]` (Bielefeld University), Christopher Musco `[通讯]` (New York University)

**通讯引用:** 1282 | [OpenAlex ID](https://openalex.org/A5018420180)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PolySHAP方法，利用多项式回归（包含高阶交互项）来逼近游戏函数，从而估计Shapley值；同时证明paired KernelSHAP与2阶PolySHAP等价，提供paired采样有效性的理论解释。

**💡 创新点**

①把KernelSHAP的线性逼近扩展到任意多项式逼近；②证明PolySHAP在样本数足够时收敛于真实Shapley值；③揭示paired采样在不增加拟合代价的情况下就能捕获所有二阶交互，从而解释其实用优势。

**🔧 技术方法**

多项式回归（交互项设计矩阵）、最小二乘拟合、利用Shapley权重与勒维尔分数的采样、投影矩阵变换、对称性与勒维尔分数理论、以及在不同采样策略下的误差分析。

**📊 数据集**

15个局部解释游戏（Housing、ViT9、Bike、Forest、Adult、ResNet18、DistilBERT、Estate、ViT16、CIFAR10、Cancer、CG60、IL60、NHANES、Crime），涵盖表格、图像、语言与合成数据，特征维数从8到101不等。

**📈 对比分析**

与Permutation Sampling、SVARM、MSR、Unbiased KernelSHAP、RegressionMSR（XGBoost）、TreeSHAP等基线比较。实验表明：在paired采样下，PolySHAP在引入3阶或更高交互项时，MSE明显低于KernelSHAP；在低维任务中，3-PolySHAP往往优于所有基线；在高维任务中，部分3阶交互即可带来可观提升，但相较于RegressionMSR仍显逊色。

**⚠️ 局限性**

（1）高阶交互项导致参数维度呈指数增长，需更大采样预算；（2）对大特征维度的PolySHAP仍受限于计算和内存；（3）paired k-PolySHAP与(k+1)-PolySHAP在奇数k>1时等价的理论证明尚未完成；（4）实验多依赖于lever分数采样和XGBoost等工具，缺乏对其他模型的通用性验证。

---

## 812. Improvement of the Gilbert-Varshamov Bound for Linear Codes and Quantum Codes

**arXiv ID:** 2601.18590 | [PDF](https://arxiv.org/pdf/2601.18590v1)

**作者:** Chen Yuan `[一作]`, Ruiqi Zhu `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种简洁的概率方法，改进了 q 取值线性码和量子码的 Gilbert‑Varshamov (GV) 下界，给出了相对于传统 GV 下界提高了 O(√n) 的乘子；

**💡 创新点**

创新点在于利用 Bonferroni（布隆菲诺里）不等式的高阶展开，结合 Chernoff 绑定，得到对线性码与对称距量子码的 ω(√n) 乘数提升；

**🔧 技术方法**

核心技术包括：
- 随机线性码生成矩阵的均匀分布性；
- 通过引入线性相关性分析（独立 vs 线性相关子集）处理多重事件；
- 在量子码场景下扩展到辛自正交结构，分析辛球体与其正交空间的交集大小；
- 结合 Stirling 近似、Taylor 展开和大数定律，精确估计高阶交集概率；

**📊 数据集**

本文为理论研究，无使用具体数据集；

**📈 对比分析**

与传统 GV 下界（仅为 q^k<q^n/|B_q(n,d-1)|）相比，改进后得到：
- 对线性码： q^k−1/q−1 < c_δ√n·q^n/∑_{i=0}^{d-1} n_i(q−1)^i；
- 对量子码： q^{2n−k}−1/q−1 < c_δ√n·q^{2n}/∑_{i=0}^{d-1} n_i(q^2−1)^i；
该改进在大 n 情况下提升了 Θ(√n) 的可达率/距离曲线，且不牺牲常数因子。

**⚠️ 局限性**

限制与不足：
- 仅给出存在性证明，缺乏具体构造或算法实现；
- 结果主要在渐进意义上有效，常数 c_δ 未给出明确取值；
- 只适用于相对距离 δ<1−1/q（线性码）或 δ<1−1/q^2（量子码）区间；
- 对特殊参数（如 q 非质数幂或高距离）未给出进一步改进；
- 仍未解决如何在实际编码与解码中高效实现该概率构造。

---

## 813. An ISAC-ready Full-Duplex Backscatter Architecture for the mmWave IoT

**arXiv ID:** 2601.18727 | [PDF](https://arxiv.org/pdf/2601.18727v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 814. Stability as a Liability:Systematic Breakdown of Linguistic Structure in LLMs

**arXiv ID:** 2601.18588 | [PDF](https://arxiv.org/pdf/2601.18588v1)

**作者:** Xianzhe Meng `[一作]` (Huazhong University of Science and Technology), Renzhi Lu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 2056 | [OpenAlex ID](https://openalex.org/A5087579932)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探究了在大规模语言模型中，训练稳定性对生成质量的负面影响，并证明稳定的最大似然训练会导致模式坍塌和低熵输出；

**💡 创新点**

创新点在于首次将训练稳定性与模式坍塌建立理论联系，揭示稳定的最大似然优化会逼近前向KL最小化，进而聚焦少数经验模式，并通过实验验证该效应；

**🔧 技术方法**

采用理论推导（MLE与KL关系）、反馈调节框架BARRA（含DMU/DTU模块）以及熵-损失轨迹分析等技术，对内部生成统计进行动态稳定控制；

**📊 数据集**

实验基于公开的语言建模数据集（如WikiText、BookCorpus等），并在GPT‑2与BERT等多种架构上进行验证；

**📈 对比分析**

通过对比不同稳定强度（α）的实验，利用高频词比例、熵与损失曲线以及困惑度等指标评估，结果显示稳定训练导致高频词比例升高、熵下降、困惑度升高，尽管训练损失平滑收敛；

**⚠️ 局限性**

局限性包括仅关注最大似然训练和稳定机制，未给出有效的缓解方案；实验仅覆盖部分模型和数据集，缺乏对更广泛任务与架构的通用性验证。

---

## 815. GimmBO: Interactive Generative Image Model Merging via Bayesian Optimization

**arXiv ID:** 2601.18585 | [PDF](https://arxiv.org/pdf/2601.18585v1)

**作者:** Chenxi Liu `[一作]` (University of Toronto), Alec Jacobson `[通讯]` (University of Toronto)

**通讯引用:** 5318 | [OpenAlex ID](https://openalex.org/A5060647975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于用户偏好贝叶斯优化的交互式适配器合并框架，用于在数十个风格适配器之间快速找到最佳合成权重，提升图像生成的满足度。

**💡 创新点**

核心创新包括：① 针对高维（20–30D）稀疏合并问题设计的两阶段贝叶斯优化（先在 B‑capped 单纯形中寻找稀疏解，再在精细空间中细化）；② 采用偏好学习的高斯过程代理结合 SAAS 先验；③ 在采样阶段使用改进的 Dirichlet/stick‑breaking 过程并做阈值稀疏化，显著提升采样效率与收敛速度。

**🔧 技术方法**

主要技术：人机交互式偏好贝叶斯优化、偏好学习高斯过程、B‑capped 单纯形约束、两阶段搜索、SDEdit 轻量级内容控制、基于 DreamSim 的图像相似度评估、使用 BoTorch 实现。

**📊 数据集**

实验使用从 Civitai 收集的 LoRA 适配器集合（约 20 个多样风格）与 30 条基于 CIFAR-10/100 超类的文本提示；模拟用户基于 DreamSim 生成偏好，真实用户完成 12 人的匹配任务。

**📈 对比分析**

与线性搜索、随机/坐标下降、Sequential Gallery 等基线对比，本文方法在相同评估预算下实现更快的收敛、最高的图像相似度（平均 >0.9）、更高的成功率和 F1 分数；在 30/40 维度的压力测试中仍保持优势。

**⚠️ 局限性**

主要局限：仅验证线性合并；假设用户偏好满足传递性；改进的 Dirichlet 采样存在坐标顺序偏差；在计算受限环境下生成延迟可能影响体验；未处理潜在的伦理与滥用问题。

---

## 816. SeNeDiF-OOD: Semantic Nested Dichotomy Fusion for Out-of-Distribution Detection Methodology in Open-World Classification. A Case Study on Monument Style Classification

**arXiv ID:** 2601.18739 | [PDF](https://arxiv.org/pdf/2601.18739v1)

**作者:** Ignacio Antequera-Sánchez `[一作]` (University of Granada), Francisco Herrera `[通讯]` (University of Granada)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 MonuMAI 建筑风格识别系统中提出并实现了 SeNeDiF‑OOD：一种通过语义嵌套二分树实现的分层 OOD 检测框架，并在四个过滤层级上进行实验验证。

**💡 创新点**

创新点在于：① 将 OOD 检测拆解为层次化的语义二分决策，能够针对不同类型的 OOD 逐层过滤；② 在每个层级融合专家模型与人类知识，提升解释性和可维护性；③ 通过嵌套二分法形成可解释的决策路径，提供“为什么被拒绝”的信息。

**🔧 技术方法**

使用的技术包括：嵌套二分法（nested dichotomies）、AdaScale‑A 远 OOD 检测器、Entropy‑OE 近 OOD 检测器、CLIP 主题筛选、EfficientNet、ResNet‑50、MonuNet（风格分类器）和 MonuMAI‑KED（元素检测器）。

**📊 数据集**

所用数据集：ImageNet、MNIST、DTD、Wikimedia Commons、MonuMAI 原始训练集、MonuTest‑OOD（含 6,354 张多类别样本）以及公开的建筑与非建筑图像集合。

**📈 对比分析**

对比方法：与原始 MonuMAI 过滤逻辑（仅基于元素检测）进行比较。SeNeDiF‑OOD 在整体准确率（0.9417）和 AUC（0.9642）上大幅提升；精度从 0.4130 提升至 0.8823，误报率显著降低，回报率保持在 0.7625。

**⚠️ 局限性**

局限性：① 需要在每个层级手动设定阈值和进行校准；② 依赖手工挑选的高质量训练集，迁移到新领域时需要大量重采集和标注；③ 对极端失真、极端视角或边缘语义的样本仍可能产生误判，需要进一步的数据增强和自适应训练。

---

## 817. HalluCitation Matters: Revealing the Impact of Hallucinated References with 300 Hallucinated Papers in ACL Conferences

**arXiv ID:** 2601.18724 | [PDF](https://arxiv.org/pdf/2601.18724v1)

**作者:** Yusuke Sakai `[一作]` (Nara Institute of Science and Technology), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 2380 | [OpenAlex ID](https://openalex.org/A5102396915)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性分析了2024-2025年ACL、NAACL和EMNLP会议论文中出现的幻觉引用（HalluCitation）并量化其流行程度与影响

**💡 创新点**

首次提出HalluCitation概念，并建立自动化检测管道，揭示论文中此类错误正迅速增长并影响会议声誉

**🔧 技术方法**

利用OCR+MinerU提取引用，GROBID进行结构化解析，字符级Levenshtein相似度匹配，手工验证确认结果

**📊 数据集**

分析了约17,842篇会议论文的PDF和元数据，提取约741,656条引用，并对ACL Anthology、arXiv、DBLP、OpenAlex等数据库进行匹配

**📈 对比分析**

通过候选数量与实际幻觉率的关联，发现4条以上候选时检测率超过70%，表明简易匹配已能有效识别大多数案例，准确率高而误检率低

**⚠️ 局限性**

方法依赖精确匹配和手工校验，受限于OCR/解析误差和数据库污染，结果仅为下限，且未评估被拒论文和非ACL会议的情况

---

## 818. Trustworthy Evaluation of Robotic Manipulation: A New Benchmark and AutoEval Methods

**arXiv ID:** 2601.18723 | [PDF](https://arxiv.org/pdf/2601.18723v1)

**作者:** Mengyuan Liu `[一作]`, Hong Liu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可信评估框架 AutoEval 与 Eval-Actions 基准，用于细粒度执行质量与源真实性评估。

**💡 创新点**

创新点在于整合多源轨迹与失败案例的 Eval-Actions 数据集，并结合 AutoEval-S 的时空聚合与 AutoEval-P 的 GRPO 强化学习，实现可信评估。

**🔧 技术方法**

采用时空聚合策略、辅助动力学校准信号、GRPO 强化学习、监督微调 LoRA 等技术，并使用多模态 VLM 进行融合。

**📊 数据集**

使用 Eval-Actions 基准（约 13k 条轨迹，含失败、混合来源、RGB‑D+文本）以及其 Eval‑Actions Small 子集进行评估。

**📈 对比分析**

与现有 VLM 基线对比，AutoEval‑S 在 EG/ RG 下 SRCC 分别达 0.81/0.84，成功率 90%+，源判别准确率 99.6%；在 Franka 机器人上仍保持 0.71/0.75；AutoEval‑P 在 CoT 下 SRCC 0.70。

**⚠️ 局限性**

局限在于缺乏任务泛化与语言泛化的评估指标，且政策生成数据量相对有限，未来需扩展多样化 SOTA 策略及更全面的鲁棒性度量。

---

## 819. Low Cost, High Efficiency: LiDAR Place Recognition in Vineyards with Matryoshka Representation Learning

**arXiv ID:** 2601.18714 | [PDF](https://arxiv.org/pdf/2601.18714v1)

**作者:** Judith Vilella-Cantos `[一作]` (University Institute for Engineering Research, Miguel Hernández University), David Valiente `[通讯]` (University Institute for Engineering Research, Miguel Hernández University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对葡萄园环境下的LiDAR位姿识别，提出了轻量化网络MinkUNeXt-VINE，结合多损失Matryoshka Representation Learning实现高效特征学习。

**💡 创新点**

创新点包括：①在MinkUNeXt基础上剪枝优化、降低输出维度至192；②使用输入归一化与大尺度量化匹配低分辨率LiDAR；③引入多损失MRL策略，使嵌入层在不同维度下独立表征，提高判别力；④在稀疏、低成本LiDAR（Livox、Velodyne）下实现跨季节鲁棒位姿识别。

**🔧 技术方法**

技术主要包括：稀疏体素卷积（Minkowski卷积）、GeM池化、TSAP损失、MRL多损失、点云预处理（去噪、裁剪、归一化）、多尺度量化、梯度裁剪等。

**📊 数据集**

使用两大长周期葡萄园LiDAR数据集：Bacchus Long‑Term（BLT，Ouster OS1‑16）和TEMPO‑VINE（Livox MID360 与 Velodyne VLP‑16），覆盖春夏秋冬不同生长阶段。

**📈 对比分析**

与传统深度学习方法（PointNetVLAD、LPD‑Net、MinkLoc3Dv2）及手工特征（Scan Context、FPFH）对比，MinkUNeXt‑VINE在TEMPO‑VINE上Recall@1%达到69.71%（Velodyne）/55.57%（Livox），相较最优基线提升约+39%/+30%；在BLT上Recall@1%略低于原MinkUNeXt但仍保持可比水平；实时推理延迟约11.7 ms，参数量27.8 M，显著低于原MinkUNeXt。

**⚠️ 局限性**

局限性包括：①在高密度、静态环境（BLT）下因低维度导致细粒度信息略有损失；②对极端遮挡（如夏季稠密草丛）仍有召回下降；③仅评估单一LiDAR传感器，未结合多模态（相机+激光）融合；④需要更多不同农田、不同作物场景验证其泛化性。

---

## 820. Analyzing Images of Blood Cells with Quantum Machine Learning Methods: Equilibrium Propagation and Variational Quantum Circuits to Detect Acute Myeloid Leukemia

**arXiv ID:** 2601.18710 | [PDF](https://arxiv.org/pdf/2601.18710v1)

**作者:** A. Bano `[一作]`, L. Liebovitch `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文实现并基准测试了量子机器学习方法（Equilibrium Propagation 与 4‑量子位 VQC），用于从低分辨率血细胞图像中检测急性髓性白血病。

**💡 创新点**

创新点在于证明即使在受限数据量与低分辨率条件下，QML 能与经典 CNN 接近的准确率，并首次展示无反向传播的能量驱动学习在真实医疗图像上的可行性。

**🔧 技术方法**

使用的技术包括：Equilibrium Propagation、4‑量子位 VQC（ZZFeatureMap 与 RealAmplitudes ansatz）、PCA 降维、传统 CNN 与全连接网络基线、手工特征提取。

**📊 数据集**

使用的数据集为 AML‑Cytomorphology（18,365 张专家标注的血细胞图像，统一裁剪到 64×64 像素）。

**📈 对比分析**

与经典 CNN（98.4%）和 Dense NN（92%）比较，EP 达到 86.4%（比 CNN 低 12%），VQC 维持 83% 的准确率，且在 50–250 样本/类别间保持稳定；CNN 需要 250 样本才达到最高性能。

**⚠️ 局限性**

局限性包括：VQC 仅在理想状态矢量模拟下评估，未考虑真实硬件噪声；仅使用手工特征，缺乏端到端学习；研究仅限二分类，未扩展至多类别血液细胞鉴别。

---

## 821. A Pragmatic VLA Foundation Model

**arXiv ID:** 2601.18692 | [PDF](https://arxiv.org/pdf/2601.18692v1)

**作者:** Wei Wu `[一作]`, Kecheng Zheng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并训练了一个基于大规模真实双臂机器人数据（约20,000小时）的Vision‑Language‑Action（VLA）基础模型，并在三种不同平台的100个任务上进行系统评估。

**💡 创新点**

① 在真实机器人数据上实现了可观的规模收益，20,000小时训练仍无饱和；② 通过Mixture‑of‑Transformers架构将预训练VLM与动作专家解耦，利用共享自注意力实现跨模态协同；③ 引入深度信息对齐（Vision Distillation）提升空间感知；④ 开发高效训练代码，利用FSDP、混合精度和Sparse Attention 等优化显著提升吞吐量。

**🔧 技术方法**

采用Qwen2.5‑VL预训练模型，动作生成采用Flow Matching，MoT架构，深度对齐采用LingBot‑Depth，分布式训练使用FSDP2 + HSDP，操作层融合与FlexAttention。

**📊 数据集**

数据集包括：
- 20,000小时双臂机器人真实采集数据（9个平台）；
- GM‑100基准（100个任务、39,000条专家演示）；
- RoboTwin 2.0仿真数据（50任务，清洁与随机场景各25k示例）。

**📈 对比分析**

与三大SOTA VLA模型（π_0.5、GR00T N1.6、WALL‑OSS）在三平台的100任务上进行对比，使用相同训练与评估管线；成功率（SR）与进度分数（PS）作为指标。结果显示：
- 在真实机器人评估中，本模型无深度版本SR≈30–35%，有深度版本≈35–40%，均显著优于基线；
- 在仿真评估中，SR从基线的82.74%提升至88.56%；
- 训练吞吐量比StarVLA、Dexbotic、OpenPI快1.5–2.8倍，规模扩展性良好。

**⚠️ 局限性**

受限于：
- 仍主要针对桌面双臂抓取任务，缺乏移动或单臂多任务场景；
- 对深度信息的依赖需要相机硬件支持，场景光照变化仍是挑战；
- 训练与评估耗费大量真实演示，收集成本高；
- 在极端稀疏或未知任务上仍可能出现性能下降。

---

## 822. A Dynamic Framework for Grid Adaptation in Kolmogorov-Arnold Networks

**arXiv ID:** 2601.18672 | [PDF](https://arxiv.org/pdf/2601.18672v1)

**作者:** Spyros Rigas `[一作]` (National and Kapodistrian University of Athens), Georgios Alexandridis `[通讯]` (National and Kapodistrian University of Athens)

**通讯引用:** 1428 | [OpenAlex ID](https://openalex.org/A5036405671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了Kolmogorov–Arnold网络（KAN）的网格自适应问题，提出了基于重要性密度函数（IDF）的通用框架，并给出了以曲率为依据的节点分配策略。

**💡 创新点**

创新点在于将节点分配视为密度估计，提出IDF概念并证明传统输入密度策略是其特殊情况，同时设计了曲率驱动的自适应方法，显著提升了逼近精度。

**🔧 技术方法**

主要技术包括B‑spline基础的KAN层、IDF框架、通过自动微分计算曲率（Hessian）、网格扩展、训练动态监控以及Wilcoxon符号秩检验。

**📊 数据集**

实验数据集包含10个自定义多维合成函数、Feynman物理方程15条以及四种不同频率的Helmholtz PDE实例。

**📈 对比分析**

在与输入基线的对比实验中，曲率自适应在合成函数上平均降低25.3%误差、Feynman数据集降低9.4%、PDE降幅23.3%，并通过Wilcoxon检验获得显著性；算力开销仅提升5–10%。

**⚠️ 局限性**

局限性包括仅验证了曲率IDF、未探究高维或非科学任务、网络规模受限、网格更新频率为手工设定等。

---

## 823. A Scanning-Based Indoor Optical Wireless Positioning System with Single VCSEL

**arXiv ID:** 2601.18740 | [PDF](https://arxiv.org/pdf/2601.18740v1)

**作者:** Yicheng Dong `[一作]` (University of Glasgow), Hanaa Abumarshoud `[通讯]` (University of Glasgow)

**通讯引用:** 604 | [OpenAlex ID](https://openalex.org/A5026932755)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种基于单个 VCSEL 天花板发射器的室内可见光定位系统，利用光束扫描（1° 分辨率）结合接收光功率和到达角实现三维定位；

**💡 创新点**

创新点在于仅使用一个窄束 VCSEL 通过扫描实现 3D 定位，极大降低硬件复杂度与成本，同时实现亚厘米级精度；

**🔧 技术方法**

采用 VCSEL 扫描、接收光功率测量、角度匹配与同步、RSS 反推距离、Laplace 分布模拟用户姿态，以及蒙特卡洛仿真与误差分析；

**📊 数据集**

使用仿真生成的数据集（室内 1m×1m×3m 的网格点、随机姿态角度、噪声模型），未使用实测数据集；

**📈 对比分析**

与传统多发射器 VLP 方案比较，仿真表明大多数测试点的定位误差在 7 cm 内，X、Y 轴 80% 误差<1 cm，误差随 SNR 提升而下降，SNR>40 dB 误差趋于 0.02–0.03 m；

**⚠️ 局限性**

局限在于用户设备姿态不确定会显著增加误差，尤其在垂直方向；系统未考虑多用户场景与实时实现，也未在实际环境中验证。

---

## 824. One Adapts to Any: Meta Reward Modeling for Personalized LLM Alignment

**arXiv ID:** 2601.18731 | [PDF](https://arxiv.org/pdf/2601.18731v1)

**作者:** Hongru Cai `[一作]` (Hong Kong Polytechnic University), Wenjie Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 11482 | [OpenAlex ID](https://openalex.org/A5100408983)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过元学习框架为每位用户学习奖励模型的初始化，从而实现少量反馈下的个性化对齐。

**💡 创新点**

创新点在于将个性化奖励建模视为元学习任务，使用可加基础奖励函数的加权组合，并提出鲁棒个性化目标（RPO）动态加权困难用户的元优化。

**🔧 技术方法**

采用 MAML 风格的两层优化、加权基础奖励函数、软阈值加权的鲁棒个性化目标，并结合 Bradley‑Terry 对比学习。

**📊 数据集**

使用 PRISM 与 Reddit TLDR 两个用户级偏好数据集，分别包含约1,287名和40名用户的对话或摘要偏好对。

**📈 对比分析**

与 Skywork‑Reward、BT、GPO、VPL、PAL、LoRe、SynthesizeMe 等基线对比，MRM 在整体准确率上平均提升约1.5–2%，在最差用户子集上表现更稳健，且参数规模与计算成本更低。

**⚠️ 局限性**

仍依赖稀疏的对比式反馈，难以处理动态偏好；未探讨主动查询、隐式信号或完整模型级元学习的可扩展性。

---

## 825. SMART: Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model

**arXiv ID:** 2601.18707 | [PDF](https://arxiv.org/pdf/2601.18707v1)

**作者:** Jan Hagnberger `[一作]` (Machine Learning and Simulation Lab, University of Stuttgart), Mathias Niepert `[通讯]` (Machine Learning and Simulation Lab, University of Stuttgart)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了 SMART，一种基于 Transformer 的网格无关气动仿真代理模型，能够仅凭几何点云和参数预测任意空间位置的物理量，无需生成 CFD 模拟网格。

**💡 创新点**

创新点：① 将几何和仿真参数压缩到共享潜在空间，并通过多层交叉注意力逐步细化；② 采用跨层几何–物理交互的解码器，实现几何与物理场的联合更新；③ 引入调制位置编码和多尺度潜在几何，兼顾查询独立性与高精度。

**🔧 技术方法**

技术：Transformer 编码器‑解码器结构，交叉注意力（cross‑attention），调制位置编码，仿真参数调制的 MLP，跨层几何–物理互联，分层潜在几何压缩。

**📊 数据集**

数据集：ShapeNetCar、AhmedML、SHIFT‑SUV、SHIFT‑Wing 四个工业级汽车/航空气动仿真数据集，点数从几千到数千万不等。

**📈 对比分析**

对比方法：与 GINO、OFormer、GNOT、Transolver、LNO、GP‑UPT、AB‑UPT 等基线比较；在 16k 采样子集与全分辨率测试中，SMART 在所有数据集上均优于或与网格依赖模型相当，尤其在 SHIFT‑Wing 的大规模仿真以及无模拟网格情况下保持低误差；对查询分布偏移和 CAD 网格的评测也表现出稳健性。

**⚠️ 局限性**

局限性：目前仅针对时间不变的流场；对大尺度时间动态仿真及极端流动（湍流、激波）验证不足；对极端几何形变的鲁棒性仍待进一步验证。

---

## 826. TEA-Bench: A Systematic Benchmarking of Tool-enhanced Emotional Support Dialogue Agent

**arXiv ID:** 2601.18700 | [PDF](https://arxiv.org/pdf/2601.18700v1)

**作者:** Xingyu Sui `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 15641 | [OpenAlex ID](https://openalex.org/A5017671620)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出首个工具增强情感支持对话基准TEA‑Bench，并收集高质量工具辅助对话数据，评估多轮情感支持中工具使用对情感与事实基础的影响。

**💡 创新点**

将外部工具与多轮情感支持结合，构建过程级评估指标（TEA分数与幻觉检测），并展示模型能力决定工具效益的规模依赖性。

**🔧 技术方法**

使用大型语言模型（Qwen3、Gemini、GPT‑4等）与MCP工具调用框架，加入Hallucination Detection Module 进行自动评估，并在工具环境中实现多种情境查询。

**📊 数据集**

基于ExTES的81个情境生成的31种工具集合，构成TEA数据集；另外筛选出365条无幻觉、高质量工具辅助对话，形成TEA‑Gold 数据集。

**📈 对比分析**

在9个LLM上比较“无工具”与“有工具”两种设置，工具使用可提升TEA总分并显著降低幻觉率，效果随模型规模提升；SFT在ID场景提升信息与有效性，但在OOD场景泛化差且幻觉率上升。

**⚠️ 局限性**

受限于仿真用户、短期对话、训练数据分布偏移导致泛化受限，工具使用效率随模型能力差异大，SFT在有限高质量数据上易引发幻觉。

---

## 827. Unheard in the Digital Age: Rethinking AI Bias and Speech Diversity

**arXiv ID:** 2601.18641 | [PDF](https://arxiv.org/pdf/2601.18641v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 828. Neural Multi-Speaker Voice Cloning for Nepali in Low-Resource Settings

**arXiv ID:** 2601.18694 | [PDF](https://arxiv.org/pdf/2601.18694v1)

**作者:** Aayush M. Shrestha `[一作]` (Institute of Engineering), Dinesh B. Kshatri `[通讯]` (Institute of Engineering)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

论文提出了一种针对尼泊尔语的少样本语音克隆系统，可仅用少量音频从梵文文本生成目标说话人的声音。

**💡 创新点**

创新点在于在低资源语言环境下构建多说话人克隆模型，结合GE2E声纹编码、Tacotron2文本到语谱、WaveRNN声码器，并通过自制语音-文本配对数据集验证其可行性。

**🔧 技术方法**

采用的技术包括GE2E声纹编码器、Tacotron2 TTS、WaveRNN声码器、UMAP可视化、EER评估及传统STFT、mel spectrogram转换。

**📊 数据集**

数据集来源于公开的OpenSLR（SLR43、SLR54、SLR143）、自采音频（有声书、自录语音、YouTube访谈等）共计833名说话人（235小时）和6,046条语音-文本配对（8.67小时）。

**📈 对比分析**

通过Cosine相似度、EER曲线和MOS评估，系统在10位测试说话人上平均MOS 3.924、相似度3.87，EER下降至0.04，表明克隆质量与说话人识别性能均具备良好表现。

**⚠️ 局限性**

主要局限包括数据量不足、说话人多样性有限、对噪声或非标准发音的鲁棒性差，以及未使用更高性能声码器导致的音质与自然度受限。

---

## 829. Learning Real-Life Approval Elections

**arXiv ID:** 2601.18651 | [PDF](https://arxiv.org/pdf/2601.18651v1)

**作者:** Piotr Faliszewski `[一作]` (AGH University of Kraków), Stanisław Szufa `[通讯]` (CNRS, LAMSADE, Université Paris Dauphine)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

学习并评估了多种独立审批模型（IAM）及其混合模型，用于生成与真实投票数据相似的随机选举；

**💡 创新点**

提出了针对IAM的最大似然估计和贝叶斯学习算法，并证明了通过候选人分组可以高效求解最优参数；

**🔧 技术方法**

主要使用期望最大化（EM）算法和NumPyro中的贝叶斯推断（NUTS + Gibbs），并采用动态规划求解多参数IAM；

**📊 数据集**

在Pabulib数据库中对271个包含至少2000名选民的真实参与式预算选举进行实验；

**📈 对比分析**

通过绝对和相对汉明距离衡量模型生成的选举与真实选举的相似度，结果表明混合IAM（尤其是4全IAM）明显优于单一组件，EM方法在此场景中表现略优于贝叶斯方法；

**⚠️ 局限性**

局限性包括：1）混合模型的参数空间较大，收敛速度慢；2）在极其多样化或稀疏的选举中，即使是混合IAM也难以拟合；3）贝叶斯方法对先验和标签交换问题敏感。

---

## 830. FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory

**arXiv ID:** 2601.18642 | [PDF](https://arxiv.org/pdf/2601.18642v1)

**作者:** Lei Wei `[一作]`, Bin Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FadeMem 记忆架构，引入生物启发的主动遗忘机制

**💡 创新点**

双层记忆层结合可调指数衰减、LLM 辅助冲突解决与智能融合，实现动态选择性遗忘

**🔧 技术方法**

采用指数衰减函数、语义相似度评估、LLM（GPT‑4o‑mini）进行冲突判断与融合，使用 text‑embedding‑3‑small 生成嵌入

**📊 数据集**

在 Multi‑Session Chat、LoCoMo 与 LTI‑Bench 三大数据集上进行评估

**📈 对比分析**

与固定窗口、RAG、Mem0、MemGPT 等基线对比，FadeMem 在多跳推理、检索精度、存储节省等指标上提升约 5–45% 以上，表现最优

**⚠️ 局限性**

受 LLM 推理误差、参数调优复杂度以及长期连续交互中的稀疏更新限制

---

## 831. Anticipation in Action: Evaluating Stimulus-Preceding Negativity as an Implicit Trigger for Adaptive Mixed Reality

**arXiv ID:** 2601.18750 | [PDF](https://arxiv.org/pdf/2601.18750v1)

**作者:** Francesco Chiossi `[一作]` (LMU Munich), Sven Mayer `[通讯]` (TU Dortmund University)

**通讯引用:** 2725 | [OpenAlex ID](https://openalex.org/A5040353906)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

在混合现实环境下，利用EEG与眼动追踪，探究并区分观测与选择意图下的预期神经信号（SPN）并通过深度学习实现意图识别；

**💡 创新点**

首次将SPN作为对抗不确定性与系统反馈的隐式意图指示器，并验证其在多场景MR交互中的鲁棒性；

**🔧 技术方法**

使用EEG+眼动同步采集，SPN时域特征提取，结合五种CNN架构（EEGNetv4、EEGInceptionERP等）进行深度学习分类，并采用LIME进行可解释性分析；

**📊 数据集**

自建MR实验数据集：28名受试者在三种日常场景（应用启动、文档编辑、视频播放）下完成4种条件（观测/选择×有/无反馈），共约90×4×3=1080个实验周期；

**📈 对比分析**

采用线性混合模型验证SPN效应，并在个体依赖与跨个体两种设置下比较分类器性能，最佳个体依赖模型EEGInceptionERP达78%平均准确率，跨个体最佳为Deep4Net的69%；

**⚠️ 局限性**

局限包括样本量有限、仅离线评估、需要个体校准、未包含错误相关电位、未探究多步交互序列的预期变化等因素限制了在真实MR系统中的即时部署与泛化能力。

---

## 832. Let's Make Every Pull Request Meaningful: An Empirical Analysis of Developer and Agentic Pull Requests

**arXiv ID:** 2601.18749 | [PDF](https://arxiv.org/pdf/2601.18749v1)

**作者:** Haruhiko Yoshioka `[一作]` (Nara Institute of Science and Technology), Kenichi Matsumoto `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 7000 | [OpenAlex ID](https://openalex.org/A5011588138)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过大规模经验分析比较人类和 AI 生成的 Pull Request（PR）的合并结果，提取 64 个特征，构建逻辑回归模型，评估各特征对合并成功率的影响。

**💡 创新点**

首次系统比较人类与多种 AI 代理生成 PR 的合并特征，揭示提交者属性主导合并、评审活动对两者影响相反，并提供 AI 代理特定特征的重要性，为人机协作提供实证指导。

**🔧 技术方法**

使用逻辑回归模型、方差膨胀因子去除多重共线性、AUC/精确率/召回率/F1/Brier 分数评估、LR χ² 重要性检验以及雷达图可视化等技术。

**📊 数据集**

使用 AIDev 数据集（包含 932,791 条 AI 生成 PR，其中 33,596 条带完整元数据；6,618 条人类生成 PR）。

**📈 对比分析**

对人类 PR、所有 AI PR 以及各 AI 代理 PR 分别训练逻辑回归模型，五折交叉验证得到 AUC 为 0.878（人类）和 0.960（所有 AI），精度、召回率、F1 均接近 1，Brier 分数低，说明模型可靠；通过 LR χ² 重要性测试比较特征族贡献。

**⚠️ 局限性**

结果仅为相关性而非因果，样本偏差（排除禁止 AI PR 的仓库）、部分 AI 代理样本量不足导致模型不稳定；部分特征缺失或缺少评论信息可能低估人类 PR 的影响。

---

## 833. Conditioned Generative Modeling of Molecular Glues: A Realistic AI Approach for Synthesizable Drug-like Molecules

**arXiv ID:** 2601.18716 | [PDF](https://arxiv.org/pdf/2601.18716v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 834. Symmetric Proofs of Parameterized Programs

**arXiv ID:** 2601.18745 | [PDF](https://arxiv.org/pdf/2601.18745v1)

**作者:** Ruotong Cheng `[一作]` (University of Toronto), Azadeh Farzan `[通讯]` (University of Toronto)

**通讯引用:** 1576 | [OpenAlex ID](https://openalex.org/A5016276143)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了可对富拓扑下无限状态参数化程序进行安全验证的新型证明系统——参数化证明空间，并给出了其相对完备性与算法构造的理论框架。

**💡 创新点**

创新点在于引入局部对称性作为证明规则的核心，推广传统证明空间以适应多种拓扑；提出了泛化Ashcroft不变式与参数化证明空间的对应关系；设计了基于极限程序与参数化谓词自动机的自动化验证流程；并证明了在满足特定条件的布尔程序族上可获得判定式。

**🔧 技术方法**

主要技术包括：一阶模型理论（同构、极限结构、Fraïssé 极限）、局部对称性定义与对称性规则、参数化证明空间与普通证明空间的关系、参数化谓词自动机的构造与可判定性分析、以及基于覆盖预序的向后可达性算法。

**📊 数据集**

本文并未使用实验数据集，而是通过严谨的数学证明和理论分析来验证其有效性与完备性；若需要实验验证，可采用典型的参数化并行程序（如令牌传递环、二维卷积等）作为案例。

**📈 对比分析**

相较于以往仅适用于星形拓扑的证明空间和Petri网覆盖判定方法，本文在更广泛的拓扑族（树、环、森林等）上实现了相对完备性与判定性；在满足条件 C 的情形下，验证算法可通过自动化构造极限程序与参数化谓词自动机实现；实验层面尚无公开基准，但理论上证明了在布尔程序族上的判定复杂度可归约至已知可判定问题。

**⚠️ 局限性**

局限性包括：需满足“有限基底”条件才能保证相对完备性，对某些拓扑（如环族）必须进行下闭包扩充；算法的可判定性依赖于结构的同构类有限且可计算，若这些条件不满足则验证仍为半可判定；整体框架在处理多量词混合不变式时的复杂度与可扩展性尚待进一步研究。

---

## 835. Benchmarking Machine Learning Models for IoT Malware Detection under Data Scarcity and Drift

**arXiv ID:** 2601.18736 | [PDF](https://arxiv.org/pdf/2601.18736v1)

**作者:** Jake Lyon `[一作]` (Wooster), Shamik Sengupta `[通讯]` (Nevada Reno)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了随机森林、LightGBM、逻辑回归和多层感知机在IoT-23数据集上的恶意软件检测与分类性能，涵盖二分类、多分类、数据量敏感性以及随时间漂移的鲁棒性；

**💡 创新点**

创新点在于将四种主流监督模型在不同数据稀缺与概念漂移场景下进行对比，首次量化树模型在资源受限的IoT环境中对恶意软件多样性变化的适应能力，并提出持续再训练的必要性；

**🔧 技术方法**

使用了特征预处理（互信息筛选、标签/独热编码、Min‑Max归一化）、5‑折交叉验证、随机采样调参、滚动窗口时间演化实验，并分别评估了准确率、F1、精确率与召回率；

**📊 数据集**

采用公开的IoT‑23网络流量数据集，包含约3.25亿条流记录，涵盖20类恶意软件和3类正常流量；

**📈 对比分析**

通过5‑折交叉验证、训练比例（30%–80%）和滚动时间窗口比较，随机森林与LightGBM在二分类任务中准确率>99%，F1>0.99，MUL多分类中F1约0.97；随着时间推移，性能明显下降，尤其是多分类场景；

**⚠️ 局限性**

局限在于模型在概念漂移下易退化，轻量级逻辑回归和LGBM在多分类时受类别不平衡影响显著，且未实现在线/主动学习机制，需进一步研究自适应重训练策略。

---

## 836. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

**arXiv ID:** 2601.18734 | [PDF](https://arxiv.org/pdf/2601.18734v1)

**作者:** Siyan Zhao `[一作]` (University of California Los Angeles), Aditya Grover `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种On-Policy Self-Distillation（OPSD）框架，使大型语言模型在仅有一轮自生成的轨迹上，利用访问真值解的教师策略来对其自身进行密集的token级监督，从而提升推理能力。

**💡 创新点**

创新点在于将教师与学生置于同一模型内部，利用特权信息（真值答案）生成教师策略，消除了外部教师模型和奖励模型的需求，并通过token级分布匹配实现高效的自我蒸馏。

**🔧 技术方法**

技术包括基于自回归语言模型的token级分布匹配（全词汇Jensen‑Shannon Divergence）、对抗式自生成采样、以及对比实验中的GRPO和SFT对齐策略。

**📊 数据集**

使用的主要数据集是OpenThoughts的数学推理子集（约30K题目，含链式推理），并在AIME 2024/25、HMMT 2025、Amo‑Bench等竞赛级别数学基准上进行评估。

**📈 对比分析**

与SFT和GRPO比较时，OPSD在8B模型上在所有基准上均取得与GRPO相当或更优的成绩，同时在生成token数量上比GRPO高效4–8倍，且仅需单轮采样而非多轮。

**⚠️ 局限性**

局限性包括对模型规模的依赖（1.7B模型效果有限）、仅在8B以内实验，未探讨更大规模模型；未充分利用答案的可验证性，缺乏对问题难度自适应的课程学习策略。

---

## 837. Health-SCORE: Towards Scalable Rubrics for Improving Health-LLMs

**arXiv ID:** 2601.18706 | [PDF](https://arxiv.org/pdf/2601.18706v1)

**作者:** Zhichao Yang `[一作]` (Optum AI), Robert E. Tillman `[通讯]` (Optum AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可扩展的健康领域Rubric评估框架Health‑SCORE，用于评估与训练医疗LLM

**💡 创新点**

创新在于将实例级细粒度Rubric聚类抽象为可迁移的通用Rubric，并通过自适应选择机制聚焦相关准则，实现低成本高效评估

**🔧 技术方法**

采用LLM嵌入+聚类、LLM评判自适应选择、GRPO强化学习、Rubric奖励以及提示注入等技术

**📊 数据集**

基于HealthBench、HealthBench‑Hard、CSEDB等医疗对话数据集进行实验

**📈 对比分析**

与单轴、Multi‑Axis、LLM生成及实例级Rubric基准对比，Health‑SCORE在域内外评估均优于其他奖励方案，且训练更快、更稳定，性能可与实例级Rubric相媲美

**⚠️ 局限性**

局限包括依赖LLM自动判断可能产生偏差，抽象过程存在主观性，奖励权重统一，且未在所有医疗任务上验证通用性

---

## 838. From Fuzzy to Exact: The Halo Architecture for Infinite-Depth Reasoning via Rational Arithmetic

**arXiv ID:** 2601.18702 | [PDF](https://arxiv.org/pdf/2601.18702v1)

**作者:** Hansheng Ren `[一作]` `[通讯]` (Independent Researcher), Hansheng Ren (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

在低精度深度学习的背景下，提出 Exactness Hypothesis 并构建 Halo Architecture，使 Transformer 采用有理数（ℚ）进行无误差推理。

**💡 创新点**

创新点在于将算术运算从 IEEE 754 浮点切换到 ℚ 有理数，并通过 Exact Inference Unit（EIU）与 Ring 机制实现无限深度、完全精确的推理与数值稳定。

**🔧 技术方法**

技术包括硬件原生有理数算子、Rational Softmax 与 Rational Attention、动态位宽 EIU、Ring 机制（向量符号化与重构）以及 Rational Alignment Loss。

**📊 数据集**

主要使用 Huginn-0125 原型模型作为实验平台，比较 600B 规模参数与 BF16/FP32 基线。

**📈 对比分析**

与传统低精度（FP16/FP32/BF16）对比，Halo 在 2000 步递归推理、混沌系统模拟、梯度稳定性和长文本记忆等任务上实现零数值漂移、保持精确推理并显著提升逻辑一致性。

**⚠️ 局限性**

局限性包括硬件实现复杂度高、计算开销相对传统浮点增加、对大规模并行推理的实现尚未成熟，以及对非有理数运算的近似仍需引入。

---

## 839. ART for Diffusion Sampling: A Reinforcement Learning Approach to Timestep Schedule

**arXiv ID:** 2601.18681 | [PDF](https://arxiv.org/pdf/2601.18681v1)

**作者:** Yilie Huang `[一作]` (Columbia University), Xunyu Zhou `[通讯]` (Columbia University)

**通讯引用:** 1678 | [OpenAlex ID](https://openalex.org/A5086115612)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于控制理论的自适应时间重参数化（ART）框架，用于在固定时间预算下重新分配扩散模型采样过程中的计算量；并提出了利用连续时间强化学习（ART‑RL）的actor‑critic算法来学习最优的时间表；

**💡 创新点**

创新点在于把时间步长分配问题建模为连续时间最优控制问题，并通过随机化控制与熵正则化引入高斯策略，将其转化为可求解的强化学习问题，理论证明了ART‑RL的最优解与原始控制问题等价；

**🔧 技术方法**

使用连续时间强化学习、Gauss随机策略、Actor‑Critic方法、Euler误差代理、以及概率流ODE的解析；

**📊 数据集**

在一维解析例子、CIFAR‑10、AFHQv2、FFHQ和ImageNet等图像数据集上进行实验；

**📈 对比分析**

与均匀时间网格和手工设计的EDM时间表做对比，结果显示ART‑RL在所有预算下均优于两者，尤其在低到中等计算预算（NFE较小）时提升显著；

**⚠️ 局限性**

局限在于目前仅针对概率流ODE的欧拉/Heun求解器，未考虑随机采样器；时间表的学习基于欧拉误差代理，可能不完全适配更高阶求解器；并且在一维案例中学习出的策略几乎只依赖时间，尚不清楚在更复杂场景中状态依赖性的重要性。

---

## 840. Counterfactual Explanations on Robust Perceptual Geodesics

**arXiv ID:** 2601.18678 | [PDF](https://arxiv.org/pdf/2601.18678v1)

**作者:** Eslam Zaher `[一作]` (University of Queensland), Fred Roosta `[通讯]` (University of Queensland)

**通讯引用:** 683 | [OpenAlex ID](https://openalex.org/A5056884940)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种利用鲁棒感知 Riemannian 几何优化的潜在空间反事实生成方法 PCG。

**💡 创新点**

创新点在于通过从鲁棒视觉模型提取特征构建鲁棒感知度量，形成潜在空间的 Riemannian 结构，并在该几何上全局优化最短曲线（geodesic），实现语义上连贯、在数值上可解释的反事实。

**🔧 技术方法**

使用了 StyleGAN2/3 生成器、鲁棒视觉模型（Robust ResNet、Inception 等）提取的多层特征、拉普拉斯拉布与拉格朗日动力学、两阶段路径优化以及鲁棒感知度量的拉伸梯度惩罚。

**📊 数据集**

实验基于 AFHQ、FFHQ 与 PlantVillage 三个高分辨率图像数据集。

**📈 对比分析**

与 REVISE、VSGD、RSGD、RSGD‑C 等基线相比，PCG 在 LPIPS、R‑LPIPS、R‑FID、SM 等感知与语义距离指标上显著更优，数值表格显示其距离指标最低、语义一致性最高，且成功率（flip rate）接近 95%。

**⚠️ 局限性**

局限性包括对高质量生成器与潜在映射的强依赖、鲁棒模型选择对度量效果的敏感、计算开销相对较大，以及在多类别或非视觉任务中的通用性尚待验证。

---

## 841. Quasi Monte Carlo methods enable extremely low-dimensional deep generative models

**arXiv ID:** 2601.18676 | [PDF](https://arxiv.org/pdf/2601.18676v1)

**作者:** Miles Martinez `[一作]` (Duke University), Alex H. Williams `[通讯]` (New York University)

**通讯引用:** 1389 | [OpenAlex ID](https://openalex.org/A5007674806)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种利用随机化准蒙特卡洛积分的深度生成模型（QLVM），直接逼近边际似然，省去编码器，专注于极低维可解释嵌入。

**💡 创新点**

创新点在于用准蒙特卡洛采样代替变分推断，直接优化边际对数似然；实现无编码器、周期边界、统一先验，并在低维空间上优于传统 VAE/IWAE，同时支持可导解码器实现聚类、可视化等后处理。

**🔧 技术方法**

采用随机化格点（随机平移晶格、Fibonacci/Korobov 格点）采样、log‑sum‑exp 估计、对数似然逼近、解码器 Jacobian 分析、无编码器深度网络等技术。

**📊 数据集**

使用 MNIST、灰度 CIFAR‑10、鸟鸣子音、蒙古鼠叫、6d 合成图像以及 Zebra finch 等多种数据集进行实验。

**📈 对比分析**

与同一解码器结构下的 2D VAE、IWAE 进行对比，QLVM 在低维（2D）时在重构误差、边际似然和样本多样性上均优于 VAE/IWAE；在更高维度时 VAE 仍有优势。计算成本相对更高但在低维设置下可接受；在聚类和可视化任务中 QLVM 提供更透明、无假设的结果。

**⚠️ 局限性**

局限性：仅适用于低维空间，无法捕捉高分辨率细节；样本质量不如高维生成模型；采样数量随维度指数增长，难以扩展到更高维；低维嵌入的可解释性受到多重可识别性的影响；对先验的统一假设限制了模型的灵活性。

---

## 842. S$^2$GR: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation

**arXiv ID:** 2601.18664 | [PDF](https://arxiv.org/pdf/2601.18664v1)

**作者:** Zihao Guo `[一作]` (Kuaishou Technology), Kaiqiao Zhan `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了S^2GR框架，在生成推荐中通过在每一步Semantic ID（SID）生成前插入可解释的思考向量，实现层级化的逐步推理，提升了生成推荐的质量和鲁棒性。

**💡 创新点**

创新点包括：① 在SID生成的每个层级插入思考向量，并用粗粒度代码簇聚类结果对其进行对比学习监督，保证推理路径可解释且均衡；② 通过构建物品共现图与代码簿均匀性、负载平衡约束的CoBa RQ‑VAE，实现了更高质量的多层次语义映射；③ 在初始思考向量上加入全局物品语义约束，缓解早期推理漂移。

**🔧 技术方法**

技术方法包括：残差量化 + 代码簿优化（CoBa RQ‑VAE），图卷积共现传播，Transformer‑T5架构，逐层思考向量生成，粗粒度对比学习（InfoNCE）与聚类中心对齐，正则化约束，及在线A/B测试评估。

**📊 数据集**

使用的公开数据集为Amazon Beauty（产品评论），工业数据集为大规模短视频平台的用户交互日志；两者均采用多模态嵌入（Qwen3‑Embedding‑4B或自研多模态大模型）。

**📈 对比分析**

与CasER、GRU4Rec、SASRec、BERT4Rec、ReaRec、TIGER等传统序列推荐和生成推荐基线进行比较；S^2GR在公开数据集提升约10.5%、在工业数据集提升约33.8%；在线A/B测试显示用户停留时间和视频观看频次显著增加。

**⚠️ 局限性**

局限性包括：① 依赖高质量的代码簿与共现图，若数据稀疏或代码簿选择不当，推理效果可能下降；② 每层思考向量增加计算量，可能影响极低时延场景的部署；③ 目前仅在视频/商品领域验证，跨域推广需要进一步研究。

---

## 843. Balancing Privacy and Robustness in Coded Computing Under Profiled Workers

**arXiv ID:** 2601.18661 | [PDF](https://arxiv.org/pdf/2601.18661v1)

**作者:** Rimpi Borah `[一作]` (Indian Institute of Technology Delhi), Aaditya Sharma `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在分布式计算中提出了一种鲁棒的 NS‑LCC 框架，利用评估索引的定制分配来同时抑制好奇工人泄露数据和拜占庭工人造成的错误；

**💡 创新点**

创新点在于：①基于工人可信度信息定制评估索引以平衡隐私与鲁棒性；②通过联合最小化 MIS 泄漏与错误定位概率构造优化问题；③提出低复杂度贪心算法实现近似最优索引分配；

**🔧 技术方法**

技术方法包括：Lagrange 多项式编码与 Chebyshev 节点评估、DCT 基错误检测与纠正、信息理论的 MIS 分析、整数规划/组合优化、贪心搜索；

**📊 数据集**

实验使用合成数据（随机矩阵、Gaussian 噪声）进行仿真，并未依赖公开数据集；

**📈 对比分析**

通过仿真比较极端解（仅最小化隐私或错误定位）与联合解，结果显示联合解同时显著降低了隐私泄露和定位错误，贪心解与穷举解相近，且计算复杂度显著降低；

**⚠️ 局限性**

局限性包括：需预先知道工人不可靠性与身份；仅考虑有限精度与拜占庭/好奇模型；在大规模系统中仍有组合爆炸风险；未验证在实际工业数据上的表现。

---

## 844. When Is Self-Disclosure Optimal? Incentives and Governance of AI-Generated Content

**arXiv ID:** 2601.18654 | [PDF](https://arxiv.org/pdf/2601.18654v1)

**作者:** Juan Wu `[一作]` (University of Science and Technology of China), Amit Mehra `[通讯]` (University of Texas at Dallas)

**通讯引用:** 1141 | [OpenAlex ID](https://openalex.org/A5000124503)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

在定量模型中比较平台是否需要强制要求创作者披露生成式AI使用情况，分析不同披露与执法强度对平台收益、创作者剩余以及内容透明度和质量的影响。

**💡 创新点**

提出将披露视为一种战略治理工具而非单纯透明手段，构建包含创作者异质性、观众AI厌恶、检测不完全和信任折扣的博弈框架，揭示披露在AI价值和成本效率处于中间区间时最优。

**🔧 技术方法**

运用微观经济学中的博弈论、最优激励设计与均衡分析技术，推导平台与创作者的最优策略。

**📊 数据集**

无实际数据集，研究完全基于理论推导与数值区间分析。

**📈 对比分析**

通过对比无披露和强制披露两种监管模式下的均衡结果和平台利润，发现披露效果呈非单调关系：在AI质量与成本收益均处中等时，披露可提升透明度并保持内容质量；在极端区间则适得其反。

**⚠️ 局限性**

模型假设简化：检测准确率为对称、观众AI厌恶为单一参数、未考虑多平台互动与动态演化、且忽略了创作者学习与技术升级等现实细节。

---

## 845. Trust, Don't Trust, or Flip: Robust Preference-Based Reinforcement Learning with Multi-Expert Feedback

**arXiv ID:** 2601.18751 | [PDF](https://arxiv.org/pdf/2601.18751v1)

**作者:** Seyed Amir Hosseini `[一作]` (K. N. Toosi University of Technology), Mahdi Javanmardi `[通讯]` (Amirkabir University of Technology)

**通讯引用:** 310 | [OpenAlex ID](https://openalex.org/A5070911378)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种联合学习奖励函数和专家信任参数的Preference-based RL框架 TTP，能够自动识别可靠、噪声和逆向专家并对逆向偏好进行翻转；

**💡 创新点**

创新点在于通过可学习的专家信任参数实现对多样化专家的自动分层处理（正向可信、零权重噪声、负向翻转），并提供理论可辨识性证明与梯度分析；

**🔧 技术方法**

采用 Bradley–Terry 对数似然损失，联合梯度优化奖励网络与信任参数，使用 tanh 限制并 max 归一化处理信任，结合 PEBBLE+SAC 策略训练；

**📊 数据集**

在 MetaWorld（Door-Open-v2、Sweep-Into-v2）和 DMControl（Cheetah-Run、Walker-Walk）四个基准环境中模拟多专家（K=4 或 5）反馈；

**📈 对比分析**

与传统 PBRL（PEBBLE）、RIME、MCP 以及 oracle SAC 对比，TTP 在 25% 逆向或噪声专家时保持接近 oracle 表现，显著优于基线；在 Walker-Walk 的反馈量/专家组合实验中显示对高质量反馈更敏感、低质量反馈更不稳定；

**⚠️ 局限性**

局限在于信任参数仅为单一全局标量，无法捕捉专家随情境变化的可靠性；对所有比较赋相同权重，缺乏比较难易度估计；需要进一步研究情境相关信任、主动专家选择等。

---

## 846. TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models

**arXiv ID:** 2601.18744 | [PDF](https://arxiv.org/pdf/2601.18744v1)

**作者:** Fangxu Yu `[一作]` (University of Maryland), Tianyi Zhou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一个名为TSRBench的综合多模态多任务时序推理基准，用于系统评估通用模型在时序感知、推理、预测与决策四大能力上的表现。

**💡 创新点**

创新点：①从14个领域共收集4125个问题，覆盖15个任务，构建多模态（文本、图像、文本+图像、嵌入）时序数据；②将时序任务拆分为感知、推理、预测、决策四维度，并细化至七类推理任务；③设计统一的评测流程和多模态输入转换；④通过大规模实验揭示规模、模态融合、预测与推理的关联性与瓶颈。

**🔧 技术方法**

使用的技术包括：大语言模型（LLM）、视觉语言模型（VLM）、时序专用LLM（TSLLM）与工具增强推理；时序数据转换为文本序列、绘图或嵌入；评估采用准确率、Spearman相关性、任务分布方差等指标；实验涵盖30+模型并尝试不同推理开销、视觉分辨率等。

**📊 数据集**

数据集为TSRBench，自行收集合成与真实时序数据，涵盖能源、交通、金融、医疗、工业等14个领域，含多模态输入；代码与数据公开于 https://tsrbench.github.io/。

**📈 对比分析**

比较方法：在统一评测框架下对比LLM、VLM、TSLLM在T、V、T+V、嵌入等输入方式下的准确率；结果显示：①LLM和VLM在感知、推理、决策上随规模提升显著提升；②预测任务不随规模提升，且与其他任务相关性低；③文本与视觉在不同任务上互补，但当前模型融合效果有限；最佳表现为GPT‑5 (T+V) 55.6%，最优开源模型Qwen3‑VL‑32B 44.9%。

**⚠️ 局限性**

局限性：①预测任务表现低且不随规模提升，表明当前通用模型对数值时序推理欠缺；②多模态融合未能显著提升性能，提示模型缺乏有效跨模态学习机制；③任务难度分布存在高方差任务，说明模型在特定推理上高度依赖训练样本；④实验主要基于准确率，未深入探究推理过程可解释性和鲁棒性。

---

## 847. Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems

**arXiv ID:** 2601.18735 | [PDF](https://arxiv.org/pdf/2601.18735v1)

**作者:** Jusheng Zhang `[一作]` (Sun Yat-sen University), Keze Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2250 | [OpenAlex ID](https://openalex.org/A5088124671)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出一种基于市场机制的多智能体视觉-语言模型协调框架Agora，能够将认知不确定性转化为可交易资产，指导代理通过利润驱动的交易实现高效协作。

**💡 创新点**

创新点在于将不确定性量化为结构化、可交易的资产，设计利润驱动的交易协议和基于Thompson Sampling的市场-aware broker，实现成本感知且结构感知的经济优化，而非传统的启发式代理路由。

**🔧 技术方法**

核心技术包括不确定性量化与资产“铸造”、利润驱动的交易协议、市场-aware Broker（扩展的Thompson Sampling）、多代理贝叶斯优化、以及基于成本与专家向量的可变换交易规则。

**📊 数据集**

实验数据集涵盖五个多模态视觉理解基准：MMMU、MMBench、MathVision、InfoVQA 与 CC‑OCR，使用公开的VLM模型（如qwen、gemini、gpt‑4o 等）进行评估。

**📈 对比分析**

与多种启发式路由器（MoA、KABB）及多代理策略（FrugalGPT、RouteLLM、EmbedLLM、HybridLLM）对比，Agora 在准确率上提升 1.1%–8.5% 的同时，将成本降低 3 倍以上，达成最优的准确率–成本 Pareto 前沿。

**⚠️ 局限性**

局限性包括对代理多样性与市场规模的依赖，单一代理情况下仍受限于代理本身性能；此外，交易机制假设代理之间的成本和专业度已知，实际部署时需进一步解决信息不对称与实时价格估计问题。

---

## 848. Reflect: Transparent Principle-Guided Reasoning for Constitutional Alignment at Scale

**arXiv ID:** 2601.18730 | [PDF](https://arxiv.org/pdf/2601.18730v1)

**作者:** Henry Bell `[一作]` (Duke University), Brandon Fain `[通讯]` (Duke University)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5015652321)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Reflect的推理时宪法对齐框架，利用在上下文中的自评、自批判和修订来让大型语言模型遵守任意自然语言原则。

**💡 创新点**

核心创新在于完全不需要参数微调或人工标注，使用自我评估阈值触发有限的批判修订，生成透明的推理链并可直接产生用于后续微调的高质量对齐数据。

**🔧 技术方法**

使用大模型（GPT‑4.1‑Mini、Claude‑3.5‑Haiku、Mistral‑7B）结合专门构造的系统提示，执行自评（Likert评分）、批判和最终修订；同时收集生成的对齐示例供SFT/DPO训练。

**📊 数据集**

主要评测数据包括PKU‑Safe‑RLHF和Anthropic HH‑RLHF两套红队提示集，以及用于生成训练数据的LMSYS‑Chat‑1M；在事实推理方面使用GSM8K和MMLU。

**📈 对比分析**

与仅基于预置宪法的生成（CCBase）以及Self‑Refine等方法比较，Reflect在三大模型上平均提升约0.1–0.4分的5分制合规评分，显著降低1–2分违规率（尤其是罕见尾部违规）且计算开销比Self‑Refine低3–5倍；对事实任务影响微乎其微。

**⚠️ 局限性**

局限性包括对自评和判定模型的依赖可能导致误报或误判，生成的推理链有时不完整或误导，原则集受限，无法保证对所有伦理关切的完全覆盖，并且在微调后可能出现表面对齐但真实误差。

---

## 849. Gained in Translation: Privileged Pairwise Judges Enhance Multilingual Reasoning

**arXiv ID:** 2601.18722 | [PDF](https://arxiv.org/pdf/2601.18722v1)

**作者:** Lintang Sutawika `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20906 | [OpenAlex ID](https://openalex.org/A5068811427)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段训练框架（自对抗与特权对比评判）用于在无目标语言训练数据的情况下提升多语言推理模型的性能。

**💡 创新点**

创新点在于：①使用英语参考答案作为评判器的特权信息，显著提升评判质量；②采用对比评判的自对抗训练方式克服评判不传递性；③在RL阶段融合可验证奖励与对比评判，减少对目标语言数据的需求。

**🔧 技术方法**

主要技术包括：1) 在翻译后的英语问答对上做监督微调（SFT）；2) 用可验证的准确率、格式化与语言忠诚度三项二值奖励 + 评判器提供的对比奖励；3) 采用 DR.GRPO（GRPO变体）的策略梯度RL；4) 对评判器使用 GPT‑4o‑mini，并在评判时提供特权信息。

**📊 数据集**

数据集为 DeepScaleR 中的数学推理题（AIME、AMC 等），共 18 种语言（含印尼语、孟加拉语、斯瓦希里语等）通过 GPT‑5‑Nano 翻译得到，训练样本约 125k 条（相当于基准 1M 的 1/8）。

**📈 对比分析**

与 Qwen2.5‑7B‑Instruct（1M 训练样本）以及 Translate‑Test 基线比较，使用精度和语言忠诚度两指标评估，SP3F 在 4 个任务（MGSM、MT‑Math100、Belebele、Global MMLU Lite）上均优于基准，尤其在低资源语言上提升幅度显著，同时保持了 1/8 训练数据的高效性。

**⚠️ 局限性**

局限性包括：1) 依赖高质量的英语参考答案和机器翻译，若翻译失真会影响评判；2) 评判器仍可能出现不传递性，需要自对抗策略来缓解；3) 在某些非数学领域任务中提升不如预期，需进一步改进通用性；4) 评判过程计算成本较高，限制了大规模部署。

---

## 850. Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning

**arXiv ID:** 2601.18699 | [PDF](https://arxiv.org/pdf/2601.18699v1)

**作者:** Olaf Yunus Laitinen Imanov `[一作]` `[通讯]` (Technical University of Denmark), Olaf Yunus Laitinen Imanov (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对六个规模从109B到1.5T参数的Transformer LLM进行序列化微调实验，系统地分析了灾难性遗忘的内部机制，揭示了注意力头失稳、表示漂移以及损失景观平坦化三大相互作用机制。

**💡 创新点**

创新点在于首次将注意力权重变动、CKA相似度、梯度相似度以及Hessian特征值等多维度机制量化为可解释的“机制指纹”，并通过干预实验验证它们对遗忘的因果影响，为持续学习的目标化改进奠定了理论基础。

**🔧 技术方法**

使用的技术包括：梯度相似度（余弦相似度）分析、Centered Kernel Alignment (CKA) 评估表示漂移、主成分旋转角度测量、Hessian特征值近似、注意力头熵和专业化指数、代表性重整（仿射变换）和曲率正则化等一系列机制级别的分析与干预方法。

**📊 数据集**

实验数据集覆盖12条任务序列，包含24个NLP任务（情感分析、问答、摘要、翻译、代码生成、事实知识、推理等），使用公开数据集如SST‑2、SQuAD、CNN/DailyMail、WMT14/16、HumanEval、MMLU、GSM8K 等。

**📈 对比分析**

与六个不同规模模型（Llama 4 Scout、Maverick、DeepSeek‑V3.1、GPT‑5.1、Claude Opus 4.5、Gemini 2.5 Pro）在相同任务序列上进行比较；结果显示梯度冲突预测遗忘率最高，且不同机制在不同阶段占主导；干预实验表明冻结注意力层、重整表示和曲率正则化可分别减少 64%/38%/34% 的遗忘，整体恢复约 70% 的原始性能。

**⚠️ 局限性**

限制包括：仅针对解码器单向Transformer；任务序列长度短（4–6 步）；实验仅覆盖监督微调而未考虑强化学习或人类反馈；未探讨极大模型（>1T）以及编码‑解码、稀疏或状态空间模型的遗忘特性。

---

## 851. COMETS: Coordinated Multi-Destination Video Transmission with In-Network Rate Adaptation

**arXiv ID:** 2601.18670 | [PDF](https://arxiv.org/pdf/2601.18670v1)

**作者:** Yulong Zhang `[一作]` (Hong Kong University of Science and Technology), Dirk Kutscher `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 4229 | [OpenAlex ID](https://openalex.org/A5043302724)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并评估了一种基于ICN的分布式多目的地视频传输框架COMETS，通过请求聚合和分布式优化实现可扩展且公平的自适应码率控制。

**💡 创新点**

①提出范围兴趣协议将客户端能力信息集中化；②使用双重分解的分布式闭式更新算法实现无中心控制的最优率分配；③将ICN聚合与多跳优化结合，显著提升QoE与公平性。

**🔧 技术方法**

信息中心网络（ICN）原语（PIT、缓存）、分布式拉格朗日分解优化、VMAF质量评估、Mini‑NDN仿真、MoQ/DASH‑BOLA等基线技术。

**📊 数据集**

采用多分辨率编码的视频流（每段2s），使用VMAF模型的预设质量映射；实验数据来自Mininet/ Mini‑NDN 仿真网络。

**📈 对比分析**

在10–300并发用户场景下与DASH‑BOLA、MoQ、GB、NDN‑MMRA对比，COMETS在VMAF、抖动、缓冲、启动延迟和Jain公平度上均优于基线，QoE保持0.7–0.9，优化延迟<50 ms，3.7×快于集中式方案。

**⚠️ 局限性**

仅适用于单一管理域，无法直接支持多域或P2P场景；对实时直播、多源流的扩展未验证；实验假设网络可靠，丢包仅在仿真层面。

---

## 852. Capturing P: On the Expressive Power and Efficient Evaluation of Boolean Retrieval

**arXiv ID:** 2601.18747 | [PDF](https://arxiv.org/pdf/2601.18747v1)

**作者:** Amir Aavani `[一作]` `[通讯]` (Apple Inc), Amir Aavani (Apple Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个支持多层次、可执行的检索语言ℒ_R（基于DAG）和一种新的评估算法（PN-Response）来实现对任意多项式时间可计算属性的检索，解决了传统检索架构在处理复杂布尔与算术约束时的效率瓶颈。

**💡 创新点**

创新点在于：1）证明ℒ_R语言恰好捕捉P类复杂度；2）设计PN-Response双向表示与DAG记忆化算法，消除了树展开和全域扫描的成本；3）提出面向LLM的查询编译流程，将自然语言意图转化为可直接在索引上执行的逻辑电路。

**🔧 技术方法**

核心技术包括：有向无环图（DAG）查询表示、正负双向集合表示、基于排序列表/位图的集合运算、顶点拓扑排序+记忆化、并行子任务调度、布尔门与算术门的逻辑合成。

**📊 数据集**

在实验中使用了MS MARCO语料库（约880万文档）以及自定义的学术搜索数据集，对比了传统DAAT、TAAT与提出的ComputePN算法。

**📈 对比分析**

与传统方法相比，ComputePN在处理包含重合子表达式、分布式否定以及异或等复杂逻辑时，保持了线性时间与输出感知的O(|V|·|U_active|)复杂度，实验结果显示对10,000+条句法树的复杂查询，ComputePN的执行时间约0.8 s，显著优于标准迭代器和递归方法。

**⚠️ 局限性**

局限性包括：1）对索引字段需二进制拆分或位切片表示，复杂数值或向量查询仍需宏化扩展；2）仅支持无环结构，无法直接处理循环或递归查询；3）需要LLM预编译阶段，编译错误或歧义可能导致查询失效。

---

## 853. Advances and Innovations in the Multi-Agent Robotic System (MARS) Challenge

**arXiv ID:** 2601.18733 | [PDF](https://arxiv.org/pdf/2601.18733v1)

**作者:** Li Kang `[一作]` (Shanghai AI Lab), Luc Van Gool `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并组织了MARS Challenge，旨在通过规划与控制两大赛道评估多智能体协作在嵌入式AI中的表现；

**💡 创新点**

通过赛道拆分实现了规划与控制的系统化评测，并在参赛方案中创新性地引入自纠正规划、专家拆分与去中心化协作机制；

**🔧 技术方法**

主要技术包括视觉‑语言模型(VLM)用于感知与高层规划、混合专家网络(MoE)与分离式激活/监控模块、以及基于图像与状态的去中心化VLA策略；

**📊 数据集**

使用VIKI‑Bench、ManiSkill3仿真器以及RoboCasa/ RocoBench等多源视觉与操作数据集进行任务设计与评测；

**📈 对比分析**

与传统单体模型对比，冠军方案在规划赛道取得最高0.893分、控制赛道多臂任务平均成功率约28%，但总体仍低于预期，显示出对长周期规划与多臂协同的挑战；

**⚠️ 局限性**

主要限制在于高度仿真化环境缺乏真实世界噪声与不确定性，且多臂协作的动作空间指数增长导致算法可扩展性与鲁棒性不足。

---

## 854. Riemannian AmbientFlow: Towards Simultaneous Manifold Learning and Generative Modeling from Corrupted Data

**arXiv ID:** 2601.18728 | [PDF](https://arxiv.org/pdf/2601.18728v1)

**作者:** Willem Diepeveen `[一作]` (University of California), Oscar Leong `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Riemannian AmbientFlow框架，能够从受噪声或线性混叠的观测中同时学习潜在数据流形与生成模型。

**💡 创新点**

创新点在于把变分推断与基于拉普拉斯几何的Riemannian Autoencoder结合，证明在合适的几何正则化与测量条件下可恢复原始分布，并为逆问题提供可逆、平滑的解码器。

**🔧 技术方法**

使用了Normalizing Flow、可逆变分自编码器、拉普拉斯几何、RIP/ RRIC 条件、梯度下降收敛分析等技术。

**📊 数据集**

实验使用低维合成流形（正弦曲线）和受模糊+噪声污染的MNIST数据集。

**📈 对比分析**

与传统AmbientFlow、TV正则化等方法对比，在合成数据上恢复误差可控；在MNIST逆问题中，RAE解码器的均方误差比TV低约一倍，视觉效果更清晰。

**⚠️ 局限性**

局限性包括：需要参考样本或低秩正则化以避免退化到测量核；对高维复杂分布的表达能力有限；理论假设（常数行列式Jacobian、RIP/ RRIC）在实际场景中难以满足；优化景观未知，可能存在局部最优。

---

## 855. Are Video Generation Models Geographically Fair? An Attraction-Centric Evaluation of Global Visual Knowledge

**arXiv ID:** 2601.18698 | [PDF](https://arxiv.org/pdf/2601.18698v1)

**作者:** Xiao Liu `[一作]` (University of California), Jiawei Zhang `[通讯]` (University of California)

**通讯引用:** 14243 | [OpenAlex ID](https://openalex.org/A5100462828)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Geo-Attraction Landmark Probing 框架，用来评估文本到视频模型在全球各地区对地理知识的公平性。

**💡 创新点**

创新点在于：①将旅游景点作为可量化的地理知识代理；②构建包含 500 个全球景点的 benchmark；③设计 Patch‑level CLIP、Keypoint‑based Local Alignment、VLM‑as‑a‑Judge 三种互补的知识评估指标。

**🔧 技术方法**

技术包括：Sora 2 文本到视频模型、Patch‑level CLIP（DINOv2）、GroundingDINO+SAM 生成区域 mask、LoFTR 关键点匹配、GPT‑5.1 作为 VLM judge，以及 AIGVE‑MACS 用于视频质量评估。

**📊 数据集**

使用的数据集为基于 Google Landmarks Dataset v2 与 Google Landmarks Places 的 500 景点 benchmark，并补充 Wikipedia 页面浏览量作为受欢迎程度指标。

**📈 对比分析**

通过与人工评估的 Spearman 相关度验证指标有效；对比不同地区、受欢迎程度和提示细节，发现模型在各洲、北南、东西之间的差异均小于 1 分；提示细化可提升 0.1–0.3 分；总体认为模型在全球范围内表现均匀。

**⚠️ 局限性**

局限性包括：评估仅聚焦景点，未覆盖文化符号与事件；指标与人工判断仍存在差距；数据集虽覆盖广泛但可能遗漏部分区域；模型水平尚未达到专家级，可能掩盖更细微的偏差。

---

## 856. Bridging Instead of Replacing Online Coding Communities with AI through Community-Enriched Chatbot Designs

**arXiv ID:** 2601.18697 | [PDF](https://arxiv.org/pdf/2601.18697v1)

**作者:** Junling Wang `[一作]` (ETH Zurich), April Yi Wang `[通讯]` (ETH Zurich)

**通讯引用:** 442 | [OpenAlex ID](https://openalex.org/A5046673805)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种 Community-Enriched AI 聊天机器人，在 KAGGLE 社区中嵌入社区生成内容和社交信号，以检索增强生成（RAG）方式为数据科学学习者提供答案，并展示源文档预览与社交指标。

**💡 创新点**

创新点在于：① 在 LLM 交互中显式展示社区帖子预览、作者信息、投票/浏览量等社交特征，实现 AI 回答与社区知识的融合；② 提出 Community-Enriched AI 设计范式，并通过两项用户研究验证其提升信任、参与度和学习效果；③ 通过高级检索面板支持用户自定义相关性、投票、浏览量排名，进一步增强可解释性与透明度。

**🔧 技术方法**

技术手段包括：检索增强生成（RAG）+ GPT‑4o；后端使用 Flask + LangChain + ChromaDB；前端基于 React + react‑markdown 与 react‑syntax‑highlighter；实现了四种交互模式（Alpha、Beta、Gamma、Delta）并配备高级搜索面板、预览面板和摘要功能。

**📊 数据集**

数据集：Kaggle 公共笔记本数据（Meta Kaggle Code 与 Meta Kaggle，约 4.8M 代码文件、37K Notebook）以及选定的 Kaggle 比赛数据（如 Quora Insincere Questions Classification），通过预处理、分块与嵌入构建检索数据库。

**📈 对比分析**

比较方法：两项实证研究—第 1 组 28 名数据科学学习者在四种模式下完成 4 任务（笔记本成绩、完成时间、问卷和访谈），第 2 组 12 名参与者对四级社区特征级别进行对比。结果显示：Alpha 模式显著提升任务成绩、缩短完成时间、提升可信度、可用性和社区参与度；Beta 与 Gamma 相比提升有限；Delta 最差。问卷数据显示 Alpha 的可靠性和实用性得分最高，且用户更倾向于使用社区预览功能。

**⚠️ 局限性**

局限性：① 仅在单一 Kaggle 比赛（Quora）上评估，缺乏跨任务和跨平台的验证；② 未测量答案准确性与实际学习成效；③ 高级检索仅支持固定属性（相关性、投票、浏览），可能产生偏见；④ 只利用公开数据，未探讨隐私与数据使用合规性；⑤ 研究样本规模相对较小，难以推断长期使用和社区贡献转化效果。

---

## 857. Explainability Methods for Hardware Trojan Detection: A Systematic Comparison

**arXiv ID:** 2601.18696 | [PDF](https://arxiv.org/pdf/2601.18696v1)

**作者:** Paul Whitten `[一作]` (Case Western Reserve University), Chris Papachristou `[通讯]` (Case Western Reserve University)

**通讯引用:** 4872 | [OpenAlex ID](https://openalex.org/A5046620750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对门级硬件木马检测任务中三类可解释方法（域感知属性分析、案例推理、模型不可知特征归因）进行系统比较，并评估其对检测性能与工程师可操作性的影响。

**💡 创新点**

首次将域感知属性分析与模型不可知的 LIME、SHAP、梯度归因等 XAI 技术在硬件安全场景中进行对标，量化解释的可操作性与模型性能的权衡。

**🔧 技术方法**

使用 XGBoost 进行门级特征分类，结合 k‑NN 案例推理、LIME、SHAP（TreeExplainer）和数值梯度归因等解释方法。

**📊 数据集**

在 Trust‑Hub 门级电路网表数据集（共 56,959 个门，11,392 个测试样本，46 个木马）上进行实验。

**📈 对比分析**

在同一测试集上比较精度、召回、F1、误报率；域感知属性法精度仅 2.00%，k‑NN 解释对应率 97.4%；XGBoost 结合阈值 0.99 获得 46.15% 精度、52.17% 召回，较基线提升 9 倍。

**⚠️ 局限性**

实验仅覆盖 Trust‑Hub 数据集，且仅使用五维结构特征，导致属性分析效果差；对数字组合木马的关注不足，仍有约 48% 木马未被检测；未验证在工业级真实工艺中的泛化性。

---

## 858. Learning temporal embeddings from electronic health records of chronic kidney disease patients

**arXiv ID:** 2601.18675 | [PDF](https://arxiv.org/pdf/2601.18675v1)

**作者:** Aditya Kumar `[一作]` (Hahn-Schickard), Oliver Amft `[通讯]` (Hahn-Schickard)

**通讯引用:** 7264 | [OpenAlex ID](https://openalex.org/A5064135418)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文研究了利用长期电子健康记录中的时序嵌入模型，对慢性肾病患者进行临床表征学习，并在ICU死亡率预测任务中评估其有效性。

**💡 创新点**

创新点在于将时间间隔显式纳入LSTM（T‑LSTM）并证明嵌入模型相较于直接端到端预测模型在多任务下表现更佳。

**🔧 技术方法**

采用三种递归架构：普通LSTM、带注意力机制的LSTM以及时间感知LSTM，并配合全连接层与逻辑回归做后续预测。

**📊 数据集**

使用MIMIC‑IV v2.2数据库中的慢性肾病患者共3932例（10,000次住院记录）进行实验。

**📈 对比分析**

通过5折交叉验证对CKD阶段的聚类质量（Davies–Bouldin指数、分类准确率）以及ICU死亡率预测的AUROC、准确率等指标进行比较，结果显示T‑LSTM在所有指标上均优于其它两种模型，嵌入+逻辑回归配置表现最好。

**⚠️ 局限性**

局限性包括仅在单一队列上评估、监督式嵌入学习导致标签相关性、未充分利用缺失值信息、模型复杂度导致对超参数敏感、缺乏临床专家评估与前瞻性验证。

---

## 859. Quantum Rotation Diversity in Displaced Squeezed Binary Phase-Shift Keying

**arXiv ID:** 2601.18655 | [PDF](https://arxiv.org/pdf/2601.18655v1)

**作者:** Ioannis Krikidis `[一作]` (University of Cyprus), Ioannis Krikidis `[通讯]` (University of Cyprus)

**通讯引用:** 10742 | [OpenAlex ID](https://openalex.org/A5080502122)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种量子旋转多样化（QRD）方案，在离散平方压缩二相移键（BPSK）量子光通信中通过对相邻时隙进行正交旋转，实现无需降频的时域多样化；

**💡 创新点**

创新点在于将经典信号空间多样化概念引入量子光通信，利用主动旋转与联合ML检测实现两路独立衰落的多样化，同时通过抑制量子噪声的压缩技术实现“超多样化”效果；

**🔧 技术方法**

采用量子位移-压缩态、Gamma–Gamma衰落模型、同相同相位的同相干探测、最大似然联合检测以及拉普拉斯变换分析与渐近理论；

**📊 数据集**

使用Gamma–Gamma分布的衰落模型（代表FSO通道的强湍流条件），并在仿真中设置η=0.8、ϵ=0.5、ζ=1.2，未使用公开数据集；

**📈 对比分析**

与传统单时隙无旋转BPSK方案对比，QRD在高信噪比下实现了两倍的多样化阶数（从2g提升到4g）并获得更高的编码增益；

**⚠️ 局限性**

局限在于假设完全通道状态信息、使用理想同相干探测且未考虑背景噪声和非理想量子接收机，未来需扩展到不完美CSI或非协同检测场景。

---

## 860. FaLW: A Forgetting-aware Loss Reweighting for Long-tailed Unlearning

**arXiv ID:** 2601.18650 | [PDF](https://arxiv.org/pdf/2601.18650v1)

**作者:** Liheng Yu `[一作]` (University of Science and Technology of China), Yang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 244234 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对机器遗忘中长尾遗忘集，提出了实例级动态损失重加权方法FaLW，能够自适应调节每个样本的遗忘强度，减少遗忘偏差；

**💡 创新点**

创新点在于通过预测概率与未见数据分布的z分数来估计遗忘偏差，并引入遗忘感知权重和类别平衡因子，实现对异质和倾斜遗忘偏差的同时纠正；

**🔧 技术方法**

采用了动态损失重加权、高斯近似未见样本概率分布、基于z分数的权重调节以及类别平衡因子，整合到梯度基近似遗忘框架；

**📊 数据集**

在CIFAR‑10、CIFAR‑100和Tiny‑ImageNet三大图像分类基准上进行实验，使用ResNet‑18和VGG‑16两种网络；

**📈 对比分析**

与9种主流近似遗忘方法（FT、GA、RL、SalUn、SFRon、BE、BS、IU、L1‑Sparse）对比，FaLW在平均性能差距（Avg. Gap）上均优于所有基线，且在FA、RA、TA、MIA等指标上表现更接近黄金标准Retrain；

**⚠️ 局限性**

局限性包括对验证集分布估计的依赖、需手动调节平衡因子和温度参数，以及在非图像任务或极端长尾分布下的泛化能力尚待进一步验证。

---

## 861. Digital Euro: Frequently Asked Questions Revisited

**arXiv ID:** 2601.18644 | [PDF](https://arxiv.org/pdf/2601.18644v1)

**作者:** Joe Cannataci `[一作]` (University of Groningen), Bernd Lucke `[通讯]` (University of Hamburg)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对欧洲央行数字欧元FAQ及其设计方案进行系统性批判性评估，重点指出其在隐私、技术安全、法律责任、经济激励和社会效益等方面存在的缺陷。

**💡 创新点**

创新点在于从多维度（隐私、密码学、经济学、监管治理）对CBDC设计进行全面剖析，并提出以开放设计、非强制接纳和免费软件为核心的改进路线。

**🔧 技术方法**

主要采用理论分析、隐私评估框架（如欧盟人权法）和安全评估方法（CAP定理、可信执行环境攻击案例）来评估数字欧元的设计与实现。

**📊 数据集**

未使用实验或大规模数据集，而是基于公开的ECB文件、法律条文、学术文献和行业案例进行文本与案例分析。

**📈 对比分析**

通过与现有支付系统（如银行卡、信用卡、PayPal等）以及现金对比，发现数字欧元在成本、匿名性和技术安全性方面并未提供显著优势；性能表现不如传统系统，安全风险被评为较高。

**⚠️ 局限性**

主要局限在于缺乏实证验证、对硬件安全实现细节的实测支持，以及对不同国家监管环境下实际可行性的深入评估。

---

## 862. Sampling Sphere Packings with Continuum Glauber Dynamics

**arXiv ID:** 2601.18748 | [PDF](https://arxiv.org/pdf/2601.18748v1)

**作者:** Aiya Kuchukova `[一作]` (Georgia Institute of Technology), Daniel J. Zhang `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了有限范围排斥配对势的Gibbs点过程（尤其是硬球模型）的连续Glauber动力学的快速混合性，并给出了从固定大小的球体打包（canonical Gibbs分布）高效采样的算法；

**💡 创新点**

创新点在于将离散时域的“负域定位（negative fields localization）”技术推广到连续空间，利用谱独立性和谱收敛（spectral gap）理论，通过“加热-冷却”与“加速技术”实现了更高密度球体打包的近似采样；

**🔧 技术方法**

主要技术包括谱独立性分析、谱收敛理论、GNZ方程、离散与连续时间Markov链的转换、空间哈希加速和自适应随机时间采样；

**📊 数据集**

本工作为理论研究，未使用具体实验数据集；

**📈 对比分析**

与之前针对硬球模型的随机采样方法相比，本文在活动参数 λ≈c/2^d（c<e）时可实现近似采样的时间复杂度为多项式级别，采样质量可控制在给定的总变差距离 δ，性能优于原有方法在相同密度下的指数级复杂度；

**⚠️ 局限性**

限制在于仅适用于有限范围排斥势、且谱间隔下界仅在 λ≤e/Δ_ϕ 范围内有效，且对更高密度或非排斥势情况尚未给出保证。

---

## 863. ctELM: Decoding and Manipulating Embeddings of Clinical Trials with Embedding Language Models

**arXiv ID:** 2601.18796 | [PDF](https://arxiv.org/pdf/2601.18796v1)

**作者:** Brian Ondov `[一作]` (Yale University), Hua Xu `[通讯]` (Yale University)

**通讯引用:** 50886 | [OpenAlex ID](https://openalex.org/A5101613292)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究开发了一个开源的Embedding Language Model（ELM）框架，用于将大型语言模型与临床试验摘要的文本嵌入空间对齐，并能够从嵌入向量解码、比较、生成临床试验摘要。

**💡 创新点**

创新点包括①首次公开实现ELM架构与训练框架；②利用少量多任务和单阶段训练即可在医学文本上获得高性能；③证明ELM可从新嵌入生成具有临床合理性的摘要，并可通过概念激活向量实现可控生成。

**🔧 技术方法**

使用的技术包括Llama 3.1基础模型、两层MLP适配器、LoRA微调、语义一致性（SC）评价指标，以及概念激活向量（CAV）用于方向控制。

**📊 数据集**

采用PubMed 200K RCT数据集（190k训练+2.5k验证+2.5k测试）以及人工合成的多任务数据进行训练。

**📈 对比分析**

与Vec2Text等基线对比，ELM在所有任务的SC分数均高于基线；在新嵌入生成任务中，人类专家被误导率约44%，接近理论极限50%，表明生成结果既流畅又具有临床合理性。

**⚠️ 局限性**

局限性包括：模型训练仅针对临床试验摘要，难以推广到其他医学文本或领域；依赖人工合成标签；生成的试验摘要尚未具备实际可执行性，可能存在伦理和安全风险。

---

## 864. POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration

**arXiv ID:** 2601.18779 | [PDF](https://arxiv.org/pdf/2601.18779v1)

**作者:** Yuxiao Qu `[一作]` (Carnegie Mellon University), Aviral Kumar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3907 | [OpenAlex ID](https://openalex.org/A5102493293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Privileged On‑Policy Exploration (POPE) 方法，在大语言模型的强化学习训练中利用人类或其他oracle的部分解答作为指导，推动模型在困难问题上获得奖励信号并实现从指导到无指导的迁移。

**💡 创新点**

创新点在于不把oracle解答当作训练目标，而是将其作为起始前缀与系统指令配合，引导自回归生成走向高奖励状态，解决了传统on‑policy RL在难题上收敛慢和ray interference问题。

**🔧 技术方法**

核心技术是将oracle前缀拼接到问题输入并加上指令，引导模型在RL（GRPO）中进行on‑policy采样；同时采用token级重要性权重裁剪、entropy、pass@k等实验对照。

**📊 数据集**

使用多来源的数学与推理难题集合（如AIME、HMMT、OmniMath、AceReason等），并在训练时加入人类编写的完整解答作为oracle。

**📈 对比分析**

与传统on‑policy RL、SFT+RL、pass@k优化等方法对比，POPE在难题集上pass@32提升约10–30%，在标准化基准（AIME 2025、HMMT 2025）上pass@1/16亦有显著提升，且能在混合难易样本时避免ray interference。

**⚠️ 局限性**

局限性包括：需要oracle解答（可扩展但成本高），对模型指令遵循能力高度依赖，且在极其难以推理或缺乏知识的题目上仍无法充分利用指导；未给出理论上可证明的收敛或样本复杂度分析。

---

## 865. Goal-oriented Communication for Fast and Robust Robotic Fault Detection and Recovery

**arXiv ID:** 2601.18765 | [PDF](https://arxiv.org/pdf/2601.18765v1)

**作者:** Shutong Chen `[一作]` (King's College London), Yansha Deng `[通讯]` (King's College London)

**通讯引用:** 6893 | [OpenAlex ID](https://openalex.org/A5009213856)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种Goal-oriented Communication（GoC）框架，用于快速且稳健地完成机器人故障检测与恢复，旨在最小化FDR时间并最大化任务成功率。

**💡 创新点**

创新点包括：①以3D场景图（3D‑SG）为语义表示，仅在检测到故障时进行语义数据传输；②用LoRA对小型语言模型（SLM）进行任务专属微调，并通过知识蒸馏提升其推理能力；③仅传输对象边缘点，构建轻量级数字孪生以实现高精度运动补偿。

**🔧 技术方法**

使用技术包括：TripletGCN 3D‑SG 生成、Attention‑based 边缘采样与曲线拟合、LoRA 微调、知识蒸馏、轻量数字孪生、以及端到端的语义通信与控制策略。

**📊 数据集**

训练数据主要来自 3RScan 3D 关系数据集，用于 3D‑SG 生成；仿真任务在 MuJoCo 中设置工作件排序、杂货包装、包裹托盘化三类工业任务，并生成随机故障场景。

**📈 对比分析**

通过在 25 次仿真跑量中与两种 SOTA 框架（文本约束+LLM 与空间约束+LLM）对比，GoC 框架使 FDR 时间平均降低 82.6% 以上，任务成功率最高提升至 76%。

**⚠️ 局限性**

局限性：依赖离线训练的 3D‑SG 与 SLM，数字孪生在极复杂或全新场景下可能缺失细节；在极低带宽或极端未知故障时，本地计算或模型推理仍可能成为瓶颈。

---

## 866. Multi-Stage Structured Estimators for Information Freshness

**arXiv ID:** 2601.18763 | [PDF](https://arxiv.org/pdf/2601.18763v1)

**作者:** Sahan Liyanaarachchi `[一作]` (University of Maryland), Nail Akar `[通讯]` (Bilkent University)

**通讯引用:** 1247 | [OpenAlex ID](https://openalex.org/A5080807022)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种 p-MAP 估计器，用分段常数函数逼近 MAP 估计器，并给出其信息新鲜度（MBF）的解析表达式，随后设计基于状态的采样率分配策略，以进一步提升新鲜度。

**💡 创新点**

创新点在于：①将 MAP 估计器转化为可控的多阶段分段常数近似；②在时间可逆 CTMC 下实现闭式 MBF 计算；③通过马尔可夫决策过程（SMDP）求解最优状态依赖采样率，从而突破传统均匀或两阶段 τ‑MAP 估计器的限制。

**🔧 技术方法**

采用的技术包括：连续时间马尔可夫链（CTMC）理论、特征值分解与详细平衡性、Rolle 定理证明分段性质、积分解析求解 MBF、以及 SMDP 与策略迭代算法求解最优采样率。

**📊 数据集**

实验使用两类数据集：一是具有唯一稳态极值的有限出生-死亡链（BDC）；二是非时可逆环形链（无限振荡 MAP 估计器），两者均在模拟环境下生成。

**📈 对比分析**

通过将 p‑MAP、τ‑MAP 与马尔可夫估计器与均匀采样方案在相同采样预算下进行比较，结果显示：在最优状态依赖采样下，p‑MAP 的 MBF 可提升约 15%（马尔可夫估计器）和 4%（τ‑MAP）；而在非时可逆链中，增加阶段数可让 p‑MAP 逐步逼近 MAP 估计器，证明早期阶段的精确度对新鲜度更为关键。

**⚠️ 局限性**

局限性包括：①闭式分析仅在时间可逆 CTMC（或所有特征值为实数的链）下成立；②对非可逆链只能通过数值积分近似；③p‑MAP 的阶段数需要预先指定，过多阶段会导致计算复杂度上升；④SMDP 方案在大状态空间下求解仍具挑战。

---

## 867. UI Remix: Supporting UI Design Through Interactive Example Retrieval and Remixing

**arXiv ID:** 2601.18759 | [PDF](https://arxiv.org/pdf/2601.18759v1)

**作者:** Junling Wang `[一作]` (ETH Zurich), April Yi Wang `[通讯]` (ETH Zurich)

**通讯引用:** 442 | [OpenAlex ID](https://openalex.org/A5046673805)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个基于例子检索与混音的交互式移动 UI 设计系统，使终端用户可以通过检索、选择并改造真实 UI 示例来实现创意设计。

**💡 创新点**

创新点在于将检索增强生成（MMRAG）与全局/局部混音机制相结合，并在检索结果中加入来源透明度指标（评级、下载量、开发者信息）以提升信任感。

**🔧 技术方法**

采用多模态检索增强生成模型（MMRAG）结合 GPT‑5 生成器；使用 GUIClip 对 UI 截图进行嵌入，ChromaDB 做向量检索；前端基于 React、Monaco 编辑器，后端 FastAPI。

**📊 数据集**

使用 Mobbin、Interaction Mining、MobileViews 三大公开 UI 截图库，挑选 196 个热门应用共约 900 张 UI 截图作为检索数据库。

**📈 对比分析**

通过 24 名无 UI 设计经验的参与者进行对照实验，比较系统与基线 GPT‑Canvas 的完成度、迭代次数、探索度及信任感；实验结果显示相对基线，系统显著提升了探索与迭代效果，完成时间略长但差异不大；检索精度在 Hit@5 0.88、nDCG@5 0.77。

**⚠️ 局限性**

局限在于仅支持单屏静态原型、短时任务，未检验多屏交互或长期使用；缺乏对透明度指标对设计质量影响的细粒度评估；并未对检索与混音各自贡献进行消融。

---

## 868. Handling Scope Checks (Extended Version)

**arXiv ID:** 2601.18793 | [PDF](https://arxiv.org/pdf/2601.18793v1)

**作者:** Michael Lee `[一作]` (University of Cambridge), Jeremy Yallop `[通讯]` (University of Cambridge)

**通讯引用:** 708 | [OpenAlex ID](https://openalex.org/A5020576379)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文对动态作用域外延检查（scope extrusion checks）进行正式化研究，提出了两套新的计算机语言（λOpQuoteSplice 与 λOpAST），并基于这两套语言构建了一个统一的框架，用以描述、实现与比较不同的动态检查方案。

**💡 创新点**

创新点包括：
- 引入“Cause‑for‑Concern (C4C)”动态检查机制，兼顾了传统懒惰检查与贪婪检查的优点；
- 将动态检查与精细化的环境分类器（refined environment classifiers）结合，实现了既有运行时检查又有静态预防的混合方案；
- 通过逻辑关系证明该混合方案的正确性，并与纯动态检查在表达力上的差异进行了理论比较；
- 在 MacroCaml 语言中实现并验证了三种检查（懒惰、贪婪、C4C）的可行性。

**🔧 技术方法**

主要技术手段包括：
- 形式化两种两阶段编程语言（含代码报价、切片与效应处理器）；
- 使用 elaboration（源语言到中间形式的展开）来简化语义并嵌入动态检查；
- 利用逻辑关系与类型系统来证明检查的安全性；
- 在 MacroCaml 的编译器中实现并测试新检查。

**📊 数据集**

本文没有使用传统意义上的数据集；评估主要通过理论证明与在 MacroCaml 上的实验验证完成。

**📈 对比分析**

比较方法：
- 通过形式化定义的检查框架，将懒惰检查、贪婪检查以及 C4C 检查视为对源语言的不同 elaboration；
- 在 MacroCaml 中实现三种检查，观察错误检测的时机、错误信息的可读性以及程序生成时的性能开销；
- 结果显示：懒惰检查导致错误信息难以定位且存在后期开销；贪婪检查虽然能更早报错，但在效应处理器的某些组合下会错误拒绝合法程序；C4C 检查兼顾了两者，既能早期发现错误，又能在效应处理器交互时保持正确性，整体性能与懒惰检查相当，明显优于贪婪检查。

**⚠️ 局限性**

局限性：
- 只考虑了两阶段、同质（homogeneous）的多阶段编程模型，未覆盖多阶段或异质系统；
- 主要针对深层无名效应处理器，未探讨一-shot 或命名处理器的行为；
- 评估主要基于 MacroCaml 的实验，缺乏在更大规模真实项目中的实证分析；
- 证明与实现集中于作用域外延问题，未涉及其他效应相关的安全性与性能问题。

---

## 869. Reuse your FLOPs: Scaling RL on Hard Problems by Conditioning on Very Off-Policy Prefixes

**arXiv ID:** 2601.18795 | [PDF](https://arxiv.org/pdf/2601.18795v1)

**作者:** Amrith Setlur `[一作]` (FAIR at Meta), Sang Michael Xie `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型的强化学习训练中，提出利用先前采样得到的正确离线前缀作为条件来指导 on‑policy 训练，从而显著提升在低通率难题上的学习效率。

**💡 创新点**

创新点：①将离线前缀仅作为上下文条件而非直接监督，避免熵崩塌与梯度不稳定；②发现并量化了“back‑generalization”现象，前缀能促进对未训练前缀问题的性能提升；③给出了 PrefixRL 目标的一致性和样本效率的理论证明。

**🔧 技术方法**

使用的技术包括：对齐策略梯度 RL（REINFORCE / PPO / GRPO）、自然策略梯度分析、重要性采样与梯度截断对比、LLM 的 chain‑of‑thought 生成、Rejection Sampling 收集前缀、back‑generalization 实验与可视化。

**📊 数据集**

数据集：训练集采用 DAPO 与 OMNI‑MATH（共 1k 个难题，base 模型通率 ≈0）；评估集包括 AIME'25、HMMT'25、IMO‑AnswerBench；离线前缀来自对 Llama‑3.1‑8B 或 Qwen‑3‑4B 进行的 Rejection Sampling。

**📈 对比分析**

与标准 on‑policy RL、SFT+RL、importance‑weighted off‑policy RL、LUFFY 等方法进行 compute‑matched 比较。PrefixRL 在训练集上 pass@1 提升约 45%，在 AIME'25 提升 23% pass@1；在 HMMT 20%；相对最强 baseline 的计算效率提升约 2×。

**⚠️ 局限性**

局限性：①主要在难题（低通率）场景显著效果；②需要先前获取的正确离线前缀，收集成本不可忽略；③对前缀分布和长度较敏感，过短或过长可能减弱效果；④back‑generalization 机制尚未在理论上完全解释；⑤在无可用前缀或易题场景下，优势不明显。

---

## 870. MEGnifying Emotion: Sentiment Analysis from Annotated Brain Data

**arXiv ID:** 2601.18792 | [PDF](https://arxiv.org/pdf/2601.18792v1)

**作者:** Brian Liu `[一作]` (University of Oxford), Oiwi Parker Jones `[通讯]` (University of Oxford)

**通讯引用:** 2333 | [OpenAlex ID](https://openalex.org/A5015428481)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

使用预训练的情感分析模型为已有的MEG脑影像数据打标签，并训练脑-情感预测（Brain‑to‑Sentiment）模型，验证从脑信号直接预测情感的可行性。

**💡 创新点**

创新点在于：①将文本情感分析结果与MEG事件注释进行强制对齐，自动为脑数据赋予情感标签；②通过此方法实现了无需人工情感标注即可构建情感预测模型，首次证明可在MEG数据上直接解码情感。

**🔧 技术方法**

技术手段包括：文本情感分析（CardiffNLP等预训练模型）、文本‑音频强制对齐、MEG预处理、MLP与LSTM回归网络训练、Spearman相关分析以及t检验进行统计显著性评估。

**📊 数据集**

数据集为三位受试者各10小时的MEG记录（共约30小时），受试者聆听《Sherlock Holmes》有声书，配套的事件注释（单词开始时间）和无标点的文本转录。

**📈 对比分析**

与随机猜测（33.3%）和多数类基线（85.0%）相比，LSTM模型平均准确率达87.37%，MLP为82.19%；balanced accuracy LSTM为35.745%，MLP为35.878%；统计显著提升（t‑test p ≪ 0.05），表明模型在情感预测上优于基线，且LSTM在效果大小上更佳。

**⚠️ 局限性**

局限性包括：样本量小（仅3位受试者）、类别不平衡导致大多数类偏向、未使用更强大或多模态模型、缺乏模型可解释性、未考虑个体差异、情感标签依赖预训练模型和事件对齐精度，可能引入噪声。

---

## 871. Unsupervised Text Segmentation via Kernel Change-Point Detection on Sentence Embeddings

**arXiv ID:** 2601.18788 | [PDF](https://arxiv.org/pdf/2601.18788v1)

**作者:** Mumin Jia `[一作]` (York University), Jairo Diaz-Rodriguez `[通讯]` (York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Embed‑KCPD，一种基于预训练句子嵌入与核变化点检测的无监督文本分割框架；

**💡 创新点**

首次给出对m‑依赖序列的核变化点检测理论，包含oracle不等式与定位一致性证明；

**🔧 技术方法**

采用句子编码器（sBERT、MPNet、text‑embedding‑3‑small、RoBERTa）与RBF/余弦核，利用动态规划与PELT实现惩罚式核变化点检测；

**📊 数据集**

在Choi合成数据、Wiki‑300/50、Elements、arXiv和真实推文流等多种文本语料上进行实验；

**📈 对比分析**

与传统无监督基线（TextTiling、GraphSeg、Coherence）和部分监督方法比较，Embed‑KCPD在P_k和WindowDiff指标上均优于或接近最强基线，且在多数数据集上可与监督模型相媲美；

**⚠️ 局限性**

理论假设（m‑依赖、字符特征核、足够分离）对真实语言的适用性有限，且在极短文档或高度语义模糊场景下性能可能下降。

---

## 872. Dep-Search: Learning Dependency-Aware Reasoning Traces with Persistent Memory

**arXiv ID:** 2601.18771 | [PDF](https://arxiv.org/pdf/2601.18771v1)

**作者:** Yanming Liu `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 1760 | [OpenAlex ID](https://openalex.org/A5047459900)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Dep-Search框架，使大型语言模型能够通过依赖感知的分解、检索、记忆写入与调用以及结论归纳实现结构化的多跳推理。

**💡 创新点**

创新点在于将问题拆分成带有显式依赖关系的子问题（QDMR式）、通过持久化内存自动保存并检索已提取的事实，并使用GRPO进行端到端的轨迹级强化学习，彻底摆脱了隐式自然语言搜索策略的局限。

**🔧 技术方法**

使用的技术包括：显式控制符（<Decompose>, <Retrieve>, <Memory>, <Conclusion>），基于稠密检索+重排序的检索管线，LRU内存缓冲及语义相似度查询，GRPO（trajectory‑level REINFORCE）强化学习，以及Qwen2.5系列大模型的指令调优。

**📊 数据集**

在七个多跳问答数据集上评估：HotpotQA、2WikiMultiHopQA、Musique、Bamboogle、TriviaQA、PopQA、NQ（以及对应的单跳/多跳划分）。

**📈 对比分析**

与十个基线（Directly Inference、Vanilla RAG、IRCoT、RA-ISF、Search‑O1、Search‑R1、R1‑Searcher、HierSearch、O²‑Searcher、ZeroSearch）对比，Dep‑Search在Qwen2.5‑3B和7B上平均得分分别提升至39.29和49.77，尤其在多跳数据集上超过基线约10+分，显示显著的性能提升。

**⚠️ 局限性**

限制包括：依赖关系拆分需要手工设计或额外的QDMR模板，内存容量需调优且在大规模知识库中可能导致噪声；强化学习过程训练成本高，奖励阈值对性能敏感；框架主要针对结构化多跳问答，未必适用于开放式生成或其他推理场景。

---

## 873. From Access Control to Usage Control with User-Managed Access

**arXiv ID:** 2601.18761 | [PDF](https://arxiv.org/pdf/2601.18761v1)

**作者:** Wout Slabbinck `[一作]` (Ghent University), Beatriz Esteves `[通讯]` (Ghent University)

**通讯引用:** 147 | [OpenAlex ID](https://openalex.org/A5025661674)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

**🎯 论文内容**

实现了一个基于 Solid 的 User‑Managed Access（UMA）系统，将访问控制与数据存储解耦，并使用 W3C ODRL 规范在授权服务器上执行使用控制策略。

**💡 创新点**

创新点在于：① 将 UMA 作为 Solid 的授权层，打破传统 WAC/ACP 与 LDP 的紧耦合；② 将 ODRL 作为可扩展的政策语言，支持法律约束、禁止与义务；③ 在 UMA 流程中加入 ODRL 评估引擎，形成完整的使用控制闭环；④ 提供了可复现的原型实现，展示了与现有 Solid 基础设施兼容的技术路径。

**🔧 技术方法**

核心技术包括：OAuth 2.0 与 UMA 2.0 认证授权协议；W3C ODRL 规范与其评估器；Solid 社区服务器（CSS）作为资源服务器；OpenID Connect（OIDC）身份令牌；Linked Data Platform（LDP）接口；JSON‑LD、Verifiable Credentials 方案（后续可扩展）。

**📊 数据集**

本文未使用公开数据集，而是通过在 GitHub 上公开的原型代码（https://github.com/SolidLabResearch/user‑managed‑access/）进行演示与验证；所有实验均在本地实验环境中进行。

**📈 对比分析**

通过与传统 WAC/ACP 方案进行对比，阐述了 UMA+ODRL 在权限灵活性、跨域支持、法律表达能力等方面的优势；虽然未给出量化性能指标，但示例演示了在多资源、多域下的访问决策效率，并证明了解耦后对审计与安全的正面影响。

**⚠️ 局限性**

主要限制包括：① ODRL 评估算法尚未标准化，导致实现细节需自行设计；② UMA 本身无法直接执行或监控义务（如“24 小时后删除”）；③ 缺乏统一的政策管理接口，资源所有者需自行维护策略文件；④ 对可验证声明（Verifiable Credentials）的支持尚不完善，未来需扩展声明类型与目的声明。

---

## 874. $α^3$-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks

**arXiv ID:** 2601.18754 | [PDF](https://arxiv.org/pdf/2601.18754v1)

**作者:** Mohamed Amine Ferrag `[一作]` (United Arab Emirates University), Merouane Debbah `[通讯]` (Khalifa University of Science and Technology)

**通讯引用:** 63755 | [OpenAlex ID](https://openalex.org/A5056145687)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现α^3‑SecBench，一个大规模安全评估套件，用于在6G环境下评估LLM驱动无人机自主系统在遭受多层攻击时的安全性、弹性与可信度。

**💡 创新点**

创新点在于引入层级化攻击分类、观察层面注入安全叠加、三维度（安全、弹性、可信）评分体系以及可验证的CWE归因机制。

**🔧 技术方法**

技术主要包括基于α^3‑Bench的对话式无人机任务、基于JSON安全叠加的攻击注入、LLM工具调用与安全警报交互、CWE判定的LLM判别器及整体评分引擎。

**📊 数据集**

使用的主要数据集是α^3‑Bench的113,475条多轮无人机任务，并通过自动化叠加生成2万+个验证攻击实例；公开数据位于GitHub。

**📈 对比分析**

对23种领先LLM（包括Google、OpenAI、Anthropic等）进行100个标准化攻击实例评估，最终最高得分0.576，表现出不同模型在检测、归因与安全响应方面的显著差异。

**⚠️ 局限性**

局限在于仅注入观察层攻击，缺乏对内部模型漏洞或网络侧真实攻击的模拟；且CWE归因精度仍低，难以实现完整的安全决策闭环。

---

## 875. Are Conversational AI Agents the Way Out? Co-Designing Reader-Oriented News Experiences with Immigrants and Journalists

**arXiv ID:** 2601.18772 | [PDF](https://arxiv.org/pdf/2601.18772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 876. HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs

**arXiv ID:** 2601.18753 | [PDF](https://arxiv.org/pdf/2601.18753v1)

**作者:** Xinyue Zeng `[一作]` (Virginia Tech), Dawei Zhou `[通讯]` (Virginia Tech)

**通讯引用:** 990 | [OpenAlex ID](https://openalex.org/A5022696348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了统一的幻觉风险上界框架，将LLM生成的幻觉拆分为数据驱动和推理驱动两类，并基于神经切线核（NTK）设计了一种可解释且无需外部参考的幻觉检测分数；

**💡 创新点**

创新点在于：①给出了首个完整的幻觉风险理论分解，明确阐释幻觉的产生与演化机制；②利用NTK几何和雅可比特征构造联合评分，既捕获数据缺陷也检测推理不稳定；

**🔧 技术方法**

技术手段包括NTK分析、Jacobian展开、Freedman不等式、特征谱敏感度评估以及自监督的NTK特征校准模块；

**📊 数据集**

数据集涵盖10个多样化基准（包括数据驱动QA、推理任务与指令跟随任务），并在9种主流LLM（如Llama2、Llama3、Qwen、GPT‑2等）上进行评测；

**📈 对比分析**

与11种现有检测方法对比，所提出的Score在大多数基准上均实现了SOTA表现，AUROC提升约5–10个百分点，尤其在小模型和推理任务上优势更显著；

**⚠️ 局限性**

局限性：依赖NTK在大模型上的近似假设，可能在极大参数规模或非标准架构下失效；此外，理论推导对训练-推理不匹配的量化仍需进一步实证验证。

---

## 877. MortalMATH: Evaluating the Conflict Between Reasoning Objectives and Emergency Contexts

**arXiv ID:** 2601.18790 | [PDF](https://arxiv.org/pdf/2601.18790v1)

**作者:** Etienne Lanzeray `[一作]` (Univ. Lille), Damien Sileo `[通讯]` (Univ. Lille)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评测 MortalMATH 这一基准，用于检测大语言模型在面对危急情境下的算术求解与安全优先的冲突。

**💡 创新点**

创新点在于构造了含有升高紧急度语境的算术问答场景，揭示了专门优化推理的模型可能忽视安全需求、存在“任务惯性”与危险延迟的问题。

**🔧 技术方法**

使用了基于箱式输出的拒绝检测、数学验证工具（Math‑Verify）、推理令牌计数等技术来量化模型的拒绝率、正确率与推理延迟，并对不同系统提示进行敏感性实验。

**📊 数据集**

数据集为 150 条用户场景（10 题 MATH 难度 4 的代数问题 + 15 种不同紧急度描述），共 5 个紧急度等级，从普通干扰到极端生死情境。

**📈 对比分析**

对比了 Llama‑3.1、Gemini 等通用模型与 Qwen‑3‑32b、GPT‑5‑nano 等推理模型，发现通用模型拒绝率随紧急度上升显著提升，推理模型保持 90%+ 正确率但拒绝率低、推理时间可达 10‑15 秒；安全性能表现存在明显分化。

**⚠️ 局限性**

局限性包括：拒绝检测仅依赖 oxed{} 格式，可能出现误判；情境模拟为文本化，缺乏多模态真实性；样本量有限，统计显著性不足；对 RLVR 等训练机制的因果归因尚未证实。

---

## 878. PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation

**arXiv ID:** 2601.18777 | [PDF](https://arxiv.org/pdf/2601.18777v1)

**作者:** Abhishek Divekar `[一作]` (Amazon AI), Anirban Majumder `[通讯]` (Amazon AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PRECISE框架，通过结合少量人工标注与LLM评判，来高效估计检索系统的精准度指标。

**💡 创新点**

在PPI基础上引入子实例级的评分拆解，利用稀疏K‑hot向量减少计算复杂度，并通过LLM不确定性校准消除偏差。

**🔧 技术方法**

使用预测增强推理（Prediction‑Powered Inference）、Claude 3 Sonnet/Haiku等LLM作为自动评判器、等距回归校准、不确定性提问链以及稀疏向量化的Precision@K公式。

**📊 数据集**

在ESC I公开数据集和印度电商内部Body查询样本（约8.5k，其中100条人工评注与8.4k无标签）上进行实验。

**📈 对比分析**

与纯人工评估、纯LLM评估以及传统PPI++对比，PRECISE在Precision@K上显著降低方差、提升准确性，预测T1处理器优于T2并与A/B实验结果高度一致，误差仅数百分点。

**⚠️ 局限性**

方法仍需人工黄金集、对LLM校准高度依赖、仅适用于静态语料库，且对跨模态或多轮搜索的适用性尚未验证。

---

## 879. Beyond Preferences: Learning Alignment Principles Grounded in Human Reasons and Values

**arXiv ID:** 2601.18760 | [PDF](https://arxiv.org/pdf/2601.18760v1)

**作者:** Henry Bell `[一作]` (Duke University), Brandon Fain `[通讯]` (Duke University)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5015652321)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Grounded Constitutional AI（GCAI）框架，利用人类偏好注释的理由与一般价值陈述自动生成既包含上下文原则又包含普遍原则的宪法，用以指导大型语言模型（LLM）的对齐；

**💡 创新点**

创新点在于：①将人类注释理由（justifications）与普遍价值文本结合，产生更具可解释性、可追溯性的上下文原则；②统一设计了从候选生成、聚类、摘要到打分选取的完整流程，克服传统ICAI仅从偏好标签中推断原则的不足；③通过“比例公平聚类”和“层次聚类”两种策略平衡多样性与去重，提升原则的代表性；

**🔧 技术方法**

技术细节包括：使用GPT‑4.1‑mini‑2025‑04‑14进行候选原则生成与摘要；使用OpenAI的text‑embedding‑3‑small进行嵌入；层次聚类（相似度阈值0.42）用于上下文原则，比例公平聚类用于一般原则；使用LLM评估原则与偏好注释的一致性，并以均方距离（MSD）等指标选取最终宪法；

**📊 数据集**

主要数据集：HelpSteer‑2（含偏好注释与理由）用于生成上下文原则；PRISM问卷（含对AI对齐价值的自由文本陈述）用于生成一般原则；同时在基准ICAI时使用相同HelpSteer‑2数据。

**📈 对比分析**

比较方法：在两类用户调查（宪法层面与原则层面）中让受试者评估GCAI与ICAI生成的宪法，在多维度（道德基础、公平、共识、连贯性等）上进行量化；对模型层面采用对齐后在MMLU与BBQ基准上的性能比较。结果显示：GCAI宪法在大多数维度上显著获得更高的偏好评分，且在安全、伦理、反歧视等主题上表现更好；在标准基准上两者差异不显著，说明对齐效果基本相当。

**⚠️ 局限性**

局限性：①仍依赖LLM生成与摘要，可能带来偏见且无法保证完全faithfulness；②未提供宪法的正式批准或投票机制，最终宪法仍需人工监督与社区协商；③仅使用了两个分离的数据源，未在同一人群中同时获取上下文与一般原则，未来需统一大规模数据；④原则选择使用的评分/聚类指标不一定完全映射人类真实偏好。

---

## 880. Subword-Based Comparative Linguistics across 242 Languages Using Wikipedia Glottosets

**arXiv ID:** 2601.18791 | [PDF](https://arxiv.org/pdf/2601.18791v1)

**作者:** Iaroslav Chelombitko `[一作]` (Neapolis University Pafos), Aleksey Komissarov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 242 种拉丁与西里尔文字语言上，构建 Wikipedia 词表（glottosets）并采用子词 BPE 进行统一分词，完成大规模跨语言比较。

**💡 创新点**

创新点在于：①以脚本为聚合单元，首次在全语言层面揭示脚本共享对词汇相似度的影响；②将 BPE 视为无监督形态学分割器，提供可扩展的宏观语言比较框架。

**🔧 技术方法**

使用 Byte‑Pair Encoding (BPE) 子词分词、rank‑based 向量表示、Mantel 相关检验、Jaccard 距离等技术。

**📊 数据集**

基于维基百科的 320 语言原始文本，筛选出 242 种拉丁/西里尔脚本语言，生成词表。

**📈 对比分析**

比较方法：BPE 词表相似度与基因学距离相关，跨语言同形词的分割差异与语言距离正相关；实验显示 BPE 较随机分割提升 95% 的形态学匹配，BPE 相似度与遗传相似度相关性 r=0.329，跨同形词差异约 48.7%。

**⚠️ 局限性**

局限性：受 Wikipedia 主题偏差、脚本过滤不够精准、BPE 受词频驱动导致形态学误差、未覆盖屈折形态、仅在单一语料来源。

---

## 881. Design Techniques for LLM-Powered Interactive Storytelling: A Case Study of the Dramamancer System

**arXiv ID:** 2601.18785 | [PDF](https://arxiv.org/pdf/2601.18785v1)

**作者:** Tiffany Wang `[一作]` (Midjourney), Max Kreminski `[通讯]` (Midjourney)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个名为 Dramamancer 的系统，利用大语言模型（LLM）将作者设计的故事框架（包含风格、角色、场景和事件）动态生成交互式剧情，并通过玩家输入实时影响故事进展。

**💡 创新点**

创新点在于将作者的“故事模式”与 LLM 的文本生成与情节判定功能相结合，形成两阶段（实例化与解释）对话式交互，使得故事既遵循作者设定又能随玩家选择自由变化；同时设计了事件触发条件与结果的结构化表达（storylet），为 LLM 提供清晰的情节控制信号。

**🔧 技术方法**

技术上采用了两大 LLM 模块：1）实例化模块，通过结构化提示（prompt）让 LLM 生成下一句剧情并判断是否需暂停等待玩家输入；2）解释模块，在玩家输入后调用 LLM 判断当前情节是否满足任何事件条件，并返回已满足的条件列表。

**📊 数据集**

论文未公开使用任何公开数据集；作者在实验中自定义了若干故事 schema 作为测试样例，并通过人工编写的事件条件与玩家输入进行验证。

**📈 对比分析**

性能评估主要基于作者视角（风格一致性、角色特色、场景准确性、事件触发与结果实现）和玩家视角（响应度、输入时机、决策感知、沉浸度）进行主观评估；未提供量化指标或与现有系统的客观对比实验。

**⚠️ 局限性**

局限性包括：1）高度依赖 LLM 的生成质量与推理准确性，模型误差会直接影响剧情连贯性与事件触发；2）事件判定与结果实现仅通过文本提示，缺乏更精细的状态管理机制；3) 未对多样化题材或长篇剧情进行系统化验证，实际可扩展性有待进一步研究。

---

## 882. Multi-Objective Reinforcement Learning for Efficient Tactical Decision Making for Trucks in Highway Traffic

**arXiv ID:** 2601.18783 | [PDF](https://arxiv.org/pdf/2601.18783v1)

**作者:** Deepthi Pathare `[一作]` (Chalmers University of Technology), Morteza Haghir Chehreghani `[通讯]` (Chalmers University of Technology)

**通讯引用:** 805 | [OpenAlex ID](https://openalex.org/A5103015876)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

开发了一套基于多目标近端策略优化（MOPPO）与Generalized Policy Improvement with Linear Support（GPI‑LS）的框架，用以在高速公路上为重型卡车执行安全、时间效率与能耗三目标的战术决策；

**💡 创新点**

创新点包括：①将GPI‑LS从价值基算法迁移到策略梯度，得到能够连续生成Pareto前沿的MOPPO；②在策略网络中加入权重条件特征提取与多维动作logits，使得单一网络可满足任意线性权重配置；③通过动作掩码安全过滤实现基于时间间隙与制动可行性约束的安全车道变换；

**🔧 技术方法**

采用的技术有：多目标强化学习、Proximal Policy Optimization、GPI‑LS、动作掩码安全过滤、SUMO仿真器、IDM纵向控制、LC2013横向控制；

**📊 数据集**

使用自建的SUMO仿真环境，构造了三种交通密度（0、0.015、0.03车辆/米）的混合车流场景，未使用公开数据集；

**📈 对比分析**

通过与单目标PPO及其他MORL方法对比，评估了Pareto前沿的完整性、成功率（100%）以及与分析最优速度/成本的偏差，实验结果表明该方法在不同交通密度下均能逼近理论最优，且能在不同权重下快速切换；

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，缺乏真实道路实验；权重设置需人工指定，未自动学习用户偏好；动作掩码安全过滤可能过于保守，导致潜在可行策略被剔除；未考虑乘客舒适度等其它重要目标。

---

## 883. Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability

**arXiv ID:** 2601.18778 | [PDF](https://arxiv.org/pdf/2601.18778v1)

**作者:** Shobhita Sundaram `[一作]` (Massachusetts Institute of Technology), Julia Kempe `[通讯]` (New York University)

**通讯引用:** 8227 | [OpenAlex ID](https://openalex.org/A5051933322)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种教师-学生异构Meta‑RL框架，利用预训练语言模型自动生成中间练习题（即“踏石”课程），帮助学生模型突破在困难数学数据集上的性能瓶颈。

**💡 创新点**

核心创新在于将教师奖励根植于学生在真实难题集上的进步，而非传统的内部可代理奖励；通过Meta‑RL对教师策略进行锐化，能够在不直接见到目标难题的前提下挖掘出可提升学生学习的自生成练习题；同时验证了教师的教学能力与学生的解题能力可分离。

**🔧 技术方法**

采用了自回归语言模型与强化学习（RLOO）相结合的教师-学生双层Meta‑RL训练，内循环使用标准RLVR对学生进行少量更新，外循环利用学生性能提升作为教师奖励；实现了无监督的自生成课程设计与自适应训练。

**📊 数据集**

在三大数学推理基准上实验：MATH、HARP 与 OlympiadBench；重点评估难度最高的 fail@128 子集（模型初始 0/128 成功率）。

**📈 对比分析**

相较于直接在难题集上进行 RL 训练、基于学习性目标的教师、以及使用完整官方训练集的上界，所提出方法在 MATH 上实现 pass@1 提升约4倍、pass@32 提升约2倍；在 HARP 上实现 pass@1 提升约2倍、pass@32 提升约1.5倍；同时在未见过的 OlympiadBench 上也能获得显著的跨域提升。

**⚠️ 局限性**

主要限制是双层 Meta‑RL 需要大量并行训练与高计算成本；内循环训练虽短，但仍需多次并行跑以获得稳定奖励；目前实验仅在 3B 参数模型上验证，扩展到更大规模及更广泛任务仍有挑战。

---

