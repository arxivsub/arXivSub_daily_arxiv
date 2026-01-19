# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-19 | 今日论文总数: 331

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. PRISM: Personalized Recommendation via Information Synergy Module

**arXiv ID:** 2601.10944 | [PDF](https://arxiv.org/pdf/2601.10944v1)

**作者:** Xinyi Zhang `[一作]` (Imperial College London), Zhongxuan Han `[通讯]` (Zhejiang University)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5044163944)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可插拔的多模态序列推荐框架PRISM，能够把多模态信息细粒度拆分为唯一、冗余和协同三种成分，并通过用户偏好动态加权融合，从而提升推荐精度。

**💡 创新点**

①系统化地将多模态交互拆解为三类信息（唯一、冗余、协同）；②引入信息理论驱动的三重损失（专用性、冗余抑制、协同激励）训练专家网络；③在交互专家层之后加入兴趣感知自适应融合层，实现用户级别的可解释加权。

**🔧 技术方法**

使用Mixture-of-Experts架构的多层感知机来分别学习三类交互专家，采用三种专门的损失函数（triplet、余弦相似度等）进行训练；通过MLP实现自适应加权；在整体模型中与已有SR骨干（SASRec、STOSA、InDiRec等）无缝集成。

**📊 数据集**

在四个公开数据集上评测：Amazon Home、Beauty、Sports 和 Yelp，均采用5-core预处理，使用leave-one-out划分，并报告Recall@10/20 和 NDCG@10/20。

**📈 对比分析**

与传统SR、基线多模态推荐以及专注于唯一/冗余的分离方法进行对比。PRISM在所有数据集和指标上均优于基线，最大提升约70% N@10（STOSA+PRISM），在InDiRec基线上平均提升约10% Recall/NCG，证明了其强大的性能和广泛的适用性。

**⚠️ 局限性**

当前模型仅考虑图像和文本两种模态，对缺失模态的鲁棒性未做深入探讨；调参较为敏感，尤其是协同与唯一损失的权重；对大规模多模态场景的实时性和可扩展性还有待进一步验证。

---

## 2. A Unified 3D Object Perception Framework for Real-Time Outside-In Multi-Camera Systems

**arXiv ID:** 2601.10819 | [PDF](https://arxiv.org/pdf/2601.10819v1)

**作者:** Yizhou Wang `[一作]` (NVIDIA Corporation), Sujit Biswas `[通讯]` (NVIDIA Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一套适用于工业设施的静态多摄像头 3D 目标感知与追踪框架，将 Sparse4D 从车载“inside‑out”迁移到 “outside‑in”，实现统一的世界坐标空间、可视化加权嵌入与时空查询。

**💡 创新点**

关键创新包括 (a) 用绝对世界坐标替代惯性运动对齐、构建多视角变形聚合； (b) 视觉可见度加权的 Occlusion‑Aware ReID 模块； (c) 采用 NVIDIA COSMOS 的文本驱动风格迁移实现 Sim2Real 数据增强； (d) 针对 MSDA 的 FP16 TensorRT 插件实现 2.15× 的加速。

**🔧 技术方法**

采用 Transformer‑based sparse query 方案、Deformable Convolution、Multi‑Scale Deformable Aggregation、视觉可见度网络、CNN backbone（ResNet‑101）、FP16 TensorRT 插件以及 COSMOS 风格迁移。

**📊 数据集**

主要在 AI City Challenge 2025 多摄像头跟踪数据集上训练和评测，并利用 COSMOS 生成的模拟风格数据进行增广。

**📈 对比分析**

在 AI City Challenge 2025 leaderboard 的在线 camera‑only 任务中，HOTA 提升至 45.22，超出现有在线基准 13.59 分；在不同 GPU 上，优化后的 MSDA 使单张 GPU 可支持 64 条流，速度提升 2.15×。

**⚠️ 局限性**

限制包括仍依赖大量标注数据（尽管已通过 COSMOS 缩小域差），在极端遮挡或极低光照下可见度估计不够精确；以及对同步时序和相机标定精度仍有一定要求。

---

## 3. Multi-Stage Patient Role-Playing Framework for Realistic Clinical Interactions

**arXiv ID:** 2601.10951 | [PDF](https://arxiv.org/pdf/2601.10951v1)

**作者:** Shijie Jiang `[一作]` (Jilin University), Ruihong Zhao `[通讯]` (Jilin University)

**通讯引用:** 3544 | [OpenAlex ID](https://openalex.org/A5103214318)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了首个基于真实胃肠科临床会诊记录构建的中文病人模拟数据集Ch-PatientSim，并研发了三阶段无训练的多阶段病人角色扮演（MSPRP）框架来提升LLM在病人模拟中的人格一致性、事实准确性、自然度和上下文相关性。

**💡 创新点**

①首次利用真实临床对话创建多维人格病人模拟数据集，解决了现有LLM生成对话缺乏真实性和情感表达的问题；②提出的MSPRP框架将生成过程拆分为基本信息生成、沟通风格注入、表达一致性调节三阶段，显著提升模拟病人行为的个性化和真实感。

**🔧 技术方法**

多维人格结构建模（五维：个性、情绪、病史回忆、医学理解、语言流畅度）；无训练的三阶段控制机制；使用LLM（如Qwen2.5、GLM-4、GPT-4o-mini等）进行基础与增强生成；自动评测指标（BLEU、ROUGE、METEOR、BERTScore）及模型评测（Persona Consistency、Factual Consistency、Naturalness、Contextual Relevance）。

**📊 数据集**

Ch-PatientSim：591例胃肠科门诊真实会诊数据，含病历信息、五维人格标签和对话转录；通过LLM少量示例生成和人工审核实现数据增强。

**📈 对比分析**

与多种LLM基线（Qwen、GLM、Llama、Internlm、DeepSeek、GPT-4o-mini）对比，MSPRP提升了自动评测指标（BLEU、ROUGE、METEOR、BERTScore）和人工评测指标（Persona Consistency、Factual Consistency、Naturalness、Contextual Relevance），在所有指标上均显著优于基线，尤其在自然度和个性一致性上提升超过10%。

**⚠️ 局限性**

目前仍受限于中文医疗数据规模有限，模型在处理极端情绪或罕见病史时可能仍表现不佳；MSPRP为无训练方案，无法进一步利用大规模自监督或强化学习提升效果；缺乏跨科室、跨语言的通用性验证。

---

## 4. Do You Trust Me? Cognitive-Affective Signatures of Trustworthiness in Large Language Models

**arXiv ID:** 2601.10719 | [PDF](https://arxiv.org/pdf/2601.10719v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 5. Unifying Speech Recognition, Synthesis and Conversion with Autoregressive Transformers

**arXiv ID:** 2601.10770 | [PDF](https://arxiv.org/pdf/2601.10770v1)

**作者:** Runyuan Cai `[一作]` (AutoArk AI), Xiaodong Zeng `[通讯]` (AutoArk AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种名为General-Purpose Audio（GPA）的统一自回归大型语言模型框架，能够在单一模型中完成文本转语音（TTS）、语音识别（ASR）和声源转换（VC）三大核心语音任务，且通过任务指令实现无缝切换。

**💡 创新点**

创新点在于：①将语音、文本和控制信号统一映射到共享的离散音频/文本子标记空间，实现所有任务的统一序列化处理；②采用BiCodec和GLM两种Tokenizer共同提供声学与语义信息；③采用纯自回归架构和联合多任务训练，消除跨任务的架构碎片化并提升跨任务知识迁移；④提供可在边缘设备部署的0.3B小模型与3B大模型两种规模。

**🔧 技术方法**

核心技术包括：自回归Transformer LLM骨干（Qwen3风格）、BiCodec声学-语义双流离散化、GLM ASR预训练Tokenizer、指令驱动的任务切换、联合多任务训练（使用一体化的next-token预测目标）、高效的流式推理与并发处理、边缘化优化策略。

**📊 数据集**

训练数据主要来自约1百万小时的公开语料（Emilia等）和自有多样化数据集，预训练后再用约20万小时的标注语音-文本对进行监督微调，文本标签通过多模型共识（Whisper、Faster-Whisper、SeamlessM4T等）产生，使用VAD、说话人分离、动态范围压缩等预处理。

**📈 对比分析**

在TTS、ASR以及VC等任务上与多阶段NAR和一阶段AR基线进行对比。0.3B模型在TTS上CER/WER低于大多数小型基线且Sim略低于专用模型；3B模型在TTS与ASR上分别接近或优于大型同类模型。流式推理指标显示在多并发请求下，GPA-0.3B的TTFC/TTFT、RTF等保持在可接受范围，显示出良好的吞吐量与延迟性能。

**⚠️ 局限性**

局限性包括：①共享单一模型可能导致在极度专业化任务上性能受限；②纯自回归推理随序列长度线性增长，长音频或超低延迟场景仍需进一步优化；③小型模型（0.3B）在ASR准确率上仍落后于专用大模型，表明容量限制；④缺乏针对长音频的高效剪枝或并行化机制。

---

## 6. "My Brother Is a School Principal, Earns About $80,000 Per Year... But When the Kids See Me, 'Wow, Uncle, You Have 1500 Followers on TikTok!'": A Study of Blind TikTokers' Alternative Professional Development Experiences

**arXiv ID:** 2601.10956 | [PDF](https://arxiv.org/pdf/2601.10956v1)

**作者:** Yao Lyu `[一作]` (University of Michigan), John M. Carroll `[通讯]` (Pennsylvania State University)

**通讯引用:** 36655 | [OpenAlex ID](https://openalex.org/A5054610664)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对60名盲人TikTok用户进行访谈，分析其在该平台上的职业发展动机、策略与面临的挑战，提出“替代性职业发展”框架；

**💡 创新点**

首次将社交媒体视为盲人职业发展的替代路径，强调平台可塑性、身份重塑与收入多元化；

**🔧 技术方法**

采用半结构化访谈与主题分析法，结合社交媒体与残障研究理论；

**📊 数据集**

60名盲人TikTok用户的访谈记录（涵盖年龄、性别、视觉残障类型、职业与收入情况等）；

**📈 对比分析**

此研究为质性探索，未做数值比较或性能评估，而是通过案例分析与主题编码呈现用户经验与平台影响；

**⚠️ 局限性**

样本仅限于活跃TikTok用户，缺乏跨平台及时间跨度的普遍性；平台更新快速，结果可能随技术迭代变化；

---

## 7. ARC Prize 2025: Technical Report

**arXiv ID:** 2601.10904 | [PDF](https://arxiv.org/pdf/2601.10904v1)

**作者:** François Chollet `[一作]`, Bryan Landers `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了ARC-AGI 2025竞赛与ARC-AGI-2基准进展，聚焦 refinement loop 与知识覆盖的限制；

**💡 创新点**

提出 refinement loop 作为 AGI 进展核心，展示零预训练小模型与进化程序综合的突破；

**🔧 技术方法**

采用测试时训练、进化程序合成、零预训练深度学习、链式思维优化等技术；

**📊 数据集**

使用 ARC-AGI-2 数据集（400 公训、120 半私评、120 私评）进行评估；

**📈 对比分析**

在 ARC-AGI-2 私评上最高得分为 24%，仍远低于人类 100% 的表现，体现显著差距；

**⚠️ 局限性**

局限在于对知识覆盖高度依赖、易出现过拟合、模型规模和推理效率仍不足。

---

## 8. Analytic Bijections for Smooth and Interpretable Normalizing Flows

**arXiv ID:** 2601.10774 | [PDF](https://arxiv.org/pdf/2601.10774v1)

**作者:** Mathis Gerdes `[一作]` (Institute of Physics, University of Amsterdam), Miranda C. N. Cheng `[通讯]` (University of Amsterdam)

**通讯引用:** 1606 | [OpenAlex ID](https://openalex.org/A5089449248)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了三类可解析的全局光滑单变量双射（立方有理、双曲正弦、立方多项式），并将其作为耦合层中仿射或样条变换的替代方案；同时设计了基于径向分解的径向流和角度可变的 Fourier 径向流。

**💡 创新点**

创新点在于同时满足全局光滑、无界域、解析可逆、可控导数以及局部与全局可变形的特性，构建了可直接插拔的高表达力双射，并提出了利用径向分解实现参数直接可训练、极具可解释性的径向流架构。

**🔧 技术方法**

采用共轭映射与 Cardano 公式得到解析逆、softplus/ sigmoid 进行参数约束、JAX 实现计算图；将这些双射嵌入耦合层、径向层以及角度可变 Fourier 径向层，实现可训练的正则化正则流。

**📊 数据集**

在一维震荡多峰、二维旋涡、二维高斯混合以及 20×20 ϕ⁴ 物理场等数据集上进行实验验证。

**📈 对比分析**

与仿射、样条、残差流等基线相比，解析双射在耦合层中可获得更低的 KL / 更高 ESS；径向流在参数量降低 1000× 的同时实现相当的性能，整体在多维目标上优于传统基线。

**⚠️ 局限性**

局限在于径向流仅适用于低维或具有明显径向结构的分布，高维场景下参数量仍较大；对更复杂多模或高维空间的适应性需要进一步探索。

---

## 9. Selecting Language Models for Social Science: Start Small, Start Open, and Validate

**arXiv ID:** 2601.10926 | [PDF](https://arxiv.org/pdf/2601.10926v1)

**作者:** Dustin S. Stoltz `[一作]` (Lehigh University), Sanuj Kumar `[通讯]` (New Mexico State University)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5114044325)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对社会科学研究者在选择大型语言模型时的四个关键维度（模型开放性、模型足迹、训练数据、模型架构与微调），并给出从小型、开放模型出发、构建专属验证集的实践框架。

**💡 创新点**

创新点在于将可验证性、可靠性、可复制性与可再现性作为评价标准，强调“从小型、开放”与“验证”而非仅凭预训练基准；同时系统梳理了模型开放性细分（权重开放、源代码开放）以及如何通过量化、上下文长度、硬件资源等考虑降低模型足迹。

**🔧 技术方法**

采用的技术包括Transformer架构基础、量化与稀疏化、微调（instruction、preference、reasoning）、检索增强生成（RAG）以及构建任务专属验证集和多维度评估指标。

**📊 数据集**

使用的数据主要是公开可获取的大规模语料（Common Crawl、The Pile/The Common Pile、Zyphra Zyda、Dolma 等），以及研究者自行标注的微调与验证数据集；讨论了合成数据与真实数据的权衡。

**📈 对比分析**

比较方法侧重于构建专属基准并以可靠性、可复制性为核心评估；文中指出在现有公开基准下较小模型往往与大型模型相当或优于其在特定任务上的表现，强调模型大小与能耗、硬件可达性的权衡；没有给出统一的性能数值，而是提供了选型指南。

**⚠️ 局限性**

局限性包括：缺乏普适的社会科学任务基准；对模型开放度的定义仍存在争议；数据泄露、版权与偏见问题仍未彻底解决；小型开放模型在极端复杂任务上的表现不一定能满足需求；实践中需要人工构建验证集，工作量大。

---

## 10. FrankenMotion: Part-level Human Motion Generation and Composition

**arXiv ID:** 2601.10909 | [PDF](https://arxiv.org/pdf/2601.10909v1)

**作者:** Chuqiao Li `[一作]` (Tübingen AI Center), Gerard Pons-Moll `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 13916 | [OpenAlex ID](https://openalex.org/A5076908763)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

建立了细粒度时间对齐的体部级文本注释数据集，并提出可按序列、原子动作、体部级别层次控制的扩散式文本到运动生成模型。

**💡 创新点**

① 通过LLM推理自动生成细粒度、异步的体部级文本注释；② 设计层次化文本条件的扩散模型，实现同时控制体部、动作和序列；③ 能生成未见过组合动作的能力。

**🔧 技术方法**

使用Deepseek‑R1 LLM生成注释；CLIP文本编码+PCA降维；Transformer‑based diffusion网络；beta掩码训练；SMPL姿态表示；多层次文本嵌入融合。

**📊 数据集**

结合KIT‑ML、BABEL、HumanML3D等公开数据，扩展得到约39小时、138k注释的新数据集，包含序列、原子动作、体部三级标注。

**📈 对比分析**

与STMC、DART、UniMotion等基线对齐训练后比较；在语义正确性（R‑Precision、M2T）和逼真度（FID、diversity）指标上均显著优于基线，平均部分语义正确率、动作/序列一致性提升，FID更低。

**⚠️ 局限性**

目前只能生成数秒级短序列，无法一次性生成分钟级长序列；长时结构建模仍需改进。

---

## 11. Secure Data Bridging in Industry 4.0: An OPC UA Aggregation Approach for Including Insecure Legacy Systems

**arXiv ID:** 2601.10929 | [PDF](https://arxiv.org/pdf/2601.10929v1)

**作者:** Dalibor Sain `[一作]` (Josef Ressel Centre for Intelligent and Secure Industrial Automation), Stefan Huber `[通讯]` (Josef Ressel Centre for Intelligent and Secure Industrial Automation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了SigmaServer，一种基于TCP层的聚合服务器，用于在工业4.0网络中安全桥接不安全的遗留设备与安全的OPC UA系统。

**💡 创新点**

创新点在于通过为每个遗留设备分配独立的TCP端口，避免命名空间污染，同时提供完整的OPC UA命名空间映射，并实现低延迟、低资源占用的安全聚合方案。

**🔧 技术方法**

技术实现采用C++/Open62541库构建多线程安全客户端与服务器，使用OPC UA安全策略Basic256Sha256，以及Modbus/TCP协议兼容性。

**📊 数据集**

数据集来自JRC ISIA实验平台，包括三台PLC（OPC UA）和一台Raspberry Pi（Modbus/TCP）模拟的遗留设备，反映实际工业生产环境。

**📈 对比分析**

通过与OPC Foundation Console Aggregation Server的对比实验，评估了端到端延迟（<2.6 ms，平均21 µs内部延迟）、CPU占用（0.75–3.16%）和内存使用（6–18 MiB）等指标，显示SigmaServer在资源占用和延迟方面优于现有方案。

**⚠️ 局限性**

局限性包括仅支持读取操作，无法阻止来自遗留区内部的攻击，需要补充IDS/蜜罐等安全措施；同时聚合延迟受客户端轮询频率影响，未来可通过发布订阅同步降低此延迟。

---

## 12. Neuro-Symbolic Activation Discovery: Transferring Mathematical Structures from Physics to Ecology for Parameter-Efficient Neural Networks

**arXiv ID:** 2601.10740 | [PDF](https://arxiv.org/pdf/2601.10740v1)

**作者:** Anas Hajbi `[一作]` `[通讯]` (Mohammed VI Polytechnic University), Anas Hajbi (Mohammed VI Polytechnic University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用遗传程序发现可解释的数学公式并将其作为激活函数注入神经网络，实现了域特定的神经符号激活；

**💡 创新点**

提出Neuro‑Symbolic Activation Discovery框架，首次发现并验证“几何转移现象”，证明物理域激活可迁移至生态域，显著提升参数效率；

**🔧 技术方法**

主要技术包括遗传程序（GP）进行符号回归、PyTorch自定义激活模块、批归一化保证训练稳定以及AUC/ log10(Params)的效率评估指标；

**📊 数据集**

实验使用的公开数据集为粒子物理的HIGGS、生态地理的Forest Cover以及文本分类的Spambase；

**📈 对比分析**

在相同轻量网络结构下对比ReLU、GELU、SiLU和Hybrid，结果显示：HIGGS轻量模型准确率0.718，Hybrid效率0.215；Forest Cover轻量模型Hybrid Transfer准确率82.4%、参数仅为5.5×，效率最高0.240；Spambase轻量模型Hybrid Specialist准确率92.0%，效率0.256；

**⚠️ 局限性**

局限性包括仅测试了浅层网络、GP结果受种子随机性影响、转移仅验证连续域间成功而离散域失败、仅涉及三种领域缺乏更广泛验证、GP计算开销与公式可解释性仍有限。

---

## 13. BYOL: Bring Your Own Language Into LLMs

**arXiv ID:** 2601.10804 | [PDF](https://arxiv.org/pdf/2601.10804v1)

**作者:** Syed Waqas Zamir `[一作]` (Microsoft AI for Good Research Lab), Juan Lavista Ferres `[通讯]` (Microsoft AI for Good Research Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 BYOL（Bring Your Own Language）框架，系统性地把低资源与极低资源语言纳入 LLM 生态，提供语言资源分层、数据清洗扩展、持续预训练、指令微调和模型融合的完整技术链。

**💡 创新点**

创新点在于：①基于真实 web‑scale 文本构建四层语言资源分类，为每种语言量身定制集成路径；②整合多源清洗、人工翻译与自动后编辑的全栈数据增强管道；③通过权重空间模型融合实现专家模型与多语言基础模型的无缝融合；④为极低资源语言引入翻译‑测试路径，显著降低对大规模文本的依赖。

**🔧 技术方法**

采用的核心技术包括：持续预训练（Continual Pretraining）、监督微调（Supervised Finetuning）、模型融合（Weight‑Space Merging）、轮回翻译评估（RTTBench）、NMT 与 LLM 后编辑、语料清洗与人工翻译、以及基于 LLM 的句子对齐与翻译。

**📊 数据集**

使用的数据集有：
- 经过 Common Crawl 处理的多语言语料库（用来划分资源层级与构建训练集），
- 真实低资源语言文本（Chichewa、Māori）与人工翻译对（FLORES‑200、Belebele、MGSM 等），
- 合成数据（通过最佳 MT 系统将英语语料翻译成目标语言），
- 极低资源语言 Inuktitut 的手工标注对（Nunavut Hansard、儿童书籍、新闻等），
- 公开多模态评测基准（ARC‑Easy/Hard、MGSM、XCOPA、StoryCloze、PIQA、HellaSwag、XNLI‑2.0、XWinograd、TruthfulQA）。

**📈 对比分析**

评测方式：在 12 项多语言基准上与强大多语言 LLM（Gemma‑3、Llama‑3.1、Qwen‑3、GPT‑4o 等）对比；在 Chichewa 与 Māori 上，4B 版 BYOL 模型平均提升约 12%（比 27B‑IT 更快且更优），12B 版与 GPT‑4o 在 MultiWikiQA 上性能相当；在 Inuktitut 上，自己训练的 NMT 系统 BLEU +4，翻译‑测试路径使 LLM 在极低资源语言上的准确率提升约 14%。

**⚠️ 局限性**

局限性包括：①安全与偏见仍依赖于基础模型的英文对齐，无法完全覆盖低资源语言特有的文化与规范；②在极低资源场景下仍需高质量翻译与后编辑，成本高昂；③模型融合在极端低资源语言直接微调时仍不可行；④整体方法对数据量、语言特征的依赖较大，扩展到更广泛语言族群仍需进一步研究。

---

## 14. Collaborative Continuum Robots: A Survey

**arXiv ID:** 2601.10721 | [PDF](https://arxiv.org/pdf/2601.10721v1)

**作者:** Xinyu Li `[一作]`, Ke Wu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了协作连续机器人（Collaborative Continuum Robots, CCRs）的研究现状，包括结构设计、建模、运动规划和控制等方面。

**💡 创新点**

创新点在于首次系统地定义了CCR的三种协作模式（分离协作、辅助协作、并行协作），并将现有研究按模式、技术维度进行分类、对比，进一步提出了挑战与未来研究方向。

**🔧 技术方法**

主要采用文献综述与统计分析技术，利用Web of Science和Scopus数据库进行文献检索，构建论文数量、研究主题与协作模式的关系图表。

**📊 数据集**

使用的数据集为检索到的CCR相关文献集合，涵盖2018‑2025年的公开论文与会议报告。

**📈 对比分析**

通过对比不同协作模式下的研究热点和论文数量，定性评价各模式的技术成熟度与应用范围；未给出具体实验性能数据，而是以文献数量、技术创新度等指标进行比较。

**⚠️ 局限性**

局限性：①综述依赖公开文献，缺乏统一实验评估与量化指标；②对每种模式的细节实现与性能分析相对粗略；③未涉及实时系统实现与工业部署的实际验证。

---

## 15. AI-Guided Human-In-the-Loop Inverse Design of High Performance Engineering Structures

**arXiv ID:** 2601.10859 | [PDF](https://arxiv.org/pdf/2601.10859v1)

**作者:** Dat Quoc Ha `[一作]` (Massachusetts Institute of Technology), Josephine V. Carstensen `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 626 | [OpenAlex ID](https://openalex.org/A5080283907)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了AI协助的Human-in-the-Loop拓扑优化框架HiTopAI，通过机器学习预测用户偏好区域并指导结构修改

**💡 创新点**

将人机交互转化为图像分割任务，用U‑Net预测并推荐用户关注区域，显著降低迭代次数

**🔧 技术方法**

U‑Net深度学习分割、骨架化（skeletonization）获取结构图、SIMP与Heaviside投影的拓扑优化算法以及人工合成标签生成

**📊 数据集**

使用TopoDiff 30k 2D拓扑优化设计（经过滤、上采样、数据增强）构成的109k 带分割标签的数据集

**📈 对比分析**

与传统单纯TO比较，在L‑bracket结构上实现39%屈曲抗力提升，仅增加4%设计时间；对最复杂节点案例提升制造性；模型IOU平均0.58

**⚠️ 局限性**

标签基于自动骨架化，存在噪声与偏差；模型仅预测单一区域；缺乏真实人类交互数据验证

---

## 16. Steering Language Models Before They Speak: Logit-Level Interventions

**arXiv ID:** 2601.10960 | [PDF](https://arxiv.org/pdf/2601.10960v1)

**作者:** Hyeseon An `[一作]` (Yonsei University), Yo-Sub Han `[通讯]` (Yonsei University)

**通讯引用:** 1428 | [OpenAlex ID](https://openalex.org/A5077698683)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练、推理时的logit层干预方法，通过统计token的z分数表来引导LLM在可读性、礼貌程度和毒性等输出特征上实现可控生成。

**💡 创新点**

创新点在于：1）仅在所有自回归模型共享的最终logit层进行干预，保持模型架构无关；2）使用token级别的z标准化log‑odds作为可解释的控制信号；3）不需要额外训练或微调，且对内部隐藏状态不做任何扰动。

**🔧 技术方法**

主要技术包括：token级统计得分（log‑odds + Dirichlet平滑 + z标准化）、候选集过滤（top‑k）、基于z分数的logit偏置（top‑m）以及softmax重新分布。

**📊 数据集**

实验数据集：OSE（阅读难度等级）、WikiPol（礼貌程度）、RealTox（毒性评估）。

**📈 对比分析**

与prompt‑based基线对比，方法在三大任务上实现了平均+47%准确率、50× F1提升，且保持或提升了模型的语义连贯性与安全性，证明了其在无训练情境下的优越性能。

**⚠️ 局限性**

局限性：需访问预训练LLM并在推理阶段额外计算，适用于新的输出特征时必须重新生成token级统计表；对极其稀疏或高度上下文依赖的特征（如毒性）仍存在挑战。

---

## 17. Beyond Accuracy: A Stability-Aware Metric for Multi-Horizon Forecasting

**arXiv ID:** 2601.10863 | [PDF](https://arxiv.org/pdf/2601.10863v1)

**作者:** Chutian Ma `[一作]` (Causify), Paul Smith `[通讯]` (Causify)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

介绍并实现了一个多期预测准确度与连贯度（AC）评分，结合能量得分和能量距离衡量准确性和稳定性，并将其作为可微分目标训练SARIMA模型。

**💡 创新点**

将准确度与时间一致性统一为可调权重的AC评分，并使用其作为训练损失，首次在概率多期预测中同时优化准确性和稳定性。

**🔧 技术方法**

能量得分、能量距离、加权权重函数、可微分SARIMA、PyTorch自动微分、AdamW优化。

**📊 数据集**

M4 Hourly 数据集的 50 条时序子集。

**📈 对比分析**

与传统一阶最大似然估计训练的SARIMA进行对比，AC-优化模型在75.4%降低垂直波动率，稳定度提升55%且多期准确度平均提升4.8%。

**⚠️ 局限性**

仍未区分“合理”与“无谓”修订，且仅验证在SARIMA上，未测试更复杂模型或其他预测任务。

---

## 18. PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis

**arXiv ID:** 2601.10945 | [PDF](https://arxiv.org/pdf/2601.10945v1)

**作者:** K Lokesh `[一作]` (Indian Institute of Technology Jodhpur), Anand Mishra `[通讯]` (Indian Institute of Technology Jodhpur)

**通讯引用:** 1282 | [OpenAlex ID](https://openalex.org/A5075621328)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过两个交互式视觉语言模型（DocVLM 与 PatientVLM）生成临床诊断对话，并在此对话数据上微调 DocVLM，以实现基于图像+对话的医疗诊断。

**💡 创新点**

创新点在于：①使用双模型交互模拟真实医生‑病患对话，避免单一 LLM 产生的角色模糊与文本局限；②在对话生成过程中引入真实诊断标签作为 PatientVLM 的隐式输入，保证症状描述的医学一致性；③利用生成的对话数据实现对 VLM 的对话条件微调，显著提升诊断准确性与可解释性。

**🔧 技术方法**

技术主要包括：视觉语言模型（InternVL3、Gemma3、Qwen2.5‑VL、MedGemma3、mPLUG‑Owl3 等）；对话生成使用结构化提示；对 DocVLM 的对话条件微调采用 LoRA 方式；评估使用标准的生成损失（交叉熵）和诊断准确率/ F1 计算。

**📊 数据集**

数据集为 MedMNIST v2 上的四个公开医学图像分类数据集：DermaMNIST、PneumoniaMNIST、RetinaMNIST、PathMNIST；训练集用于生成对话，测试集用于评估诊断性能。

**📈 对比分析**

方法与传统图像‑仅预测、CLIP/MedCLIP 等模型、以及直接零样本/提示/微调 VLM 进行对比。PCDF 在所有四个数据集上均实现了显著提升，平均 F1 提升约 11.5 分，某些场景最高可达 +37.2 分；在零样本下的对话生成同样优于链式推理（CoT）提示。

**⚠️ 局限性**

局限性包括：①临床验证样本有限，需更大规模、多样性评估；②部分生成问题过于技术化，患者可读性不足；③当前仅支持英文，缺乏多语言支持；④依赖 VLM 的预训练质量，若模型偏差会影响对话真实性。

---

## 19. Multi-Agent Taint Specification Extraction for Vulnerability Detection

**arXiv ID:** 2601.10865 | [PDF](https://arxiv.org/pdf/2601.10865v1)

**作者:** Jonah Ghebremichael `[一作]` (North Carolina State University), Alexandros Kapravelos `[通讯]` (North Carolina State University)

**通讯引用:** 1789 | [OpenAlex ID](https://openalex.org/A5041544321)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种多智能体系统，利用大型语言模型（LLM）与传统静态程序分析相结合，自动提取JavaScript包的污点源、汇、调用边以及第三方库的流量摘要，从而提升CodeQL等静态分析工具的漏洞检测效果。

**💡 创新点**

创新点在于：①使用TICR（污点信息驱动的调用解析）仅针对可能构成漏洞路径的未解析调用进行推断，显著减少LLM调用量；②将LLM作为有状态的推理代理，按需探测代码并记录推理链；③通过需求驱动的流量摘要验证，仅对实际触发的第三方调用进行深入分析；④采用外部谓词机制将污点规范模块化，避免重新编译查询。

**🔧 技术方法**

技术手段包括：静态调用图构建（CodeQL）、LLM推理（GPT‑5、GPT‑5‑mini）、多智能体架构、双向污点传播扩展规则、TICR算法、需求驱动的流量摘要验证、外部谓词接口以及多跑聚合与置信度校准。

**📊 数据集**

使用了两大数据集：Brito等人整理的957条已验证npm漏洞（筛选后得到172条）以及从Libraries.io挑选的102个热门npm包，用于评估已知漏洞检出率与发现未知漏洞。

**📈 对比分析**

评估指标主要为召回率与LLM调用成本。相较于CodeQL基线（在172条漏洞中召回率为0%），所提系统在已知漏洞中实现约65.4% 召回（115/176），并在未知包中发现4条新漏洞；同时通过TICR和需求驱动验证将LLM推理量分别压缩至94.5%与93.2%。

**⚠️ 局限性**

局限性包括：仅支持JavaScript/Node.js；只能覆盖CodeQL支持的37类CWE；对大型包（>1M token）不适用；LLM非确定性导致需多跑聚合；无法完全覆盖回调/异步传播等动态特性；依赖CodeQL的污点语义与防御器（sanitizer）实现；对非包内调用的精确性不足；需人工审计验证结果。

---

## 20. Towards Reliable ML Feature Engineering via Planning in Constrained-Topology of LLM Agents

**arXiv ID:** 2601.10820 | [PDF](https://arxiv.org/pdf/2601.10820v1)

**作者:** Himanshu Thakur `[一作]` (Meta), Jay Katukuri `[通讯]` (JPMorgan Chase & Co.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于规划器的受限拓扑多代理框架，自动完成机器学习特征工程中的多步骤代码生成。

**💡 创新点**

创新点在于将团队环境建模为有向图，让LLM规划器动态调度代理并利用下游失败信息回溯修正上游结果，同时支持人机交互。

**🔧 技术方法**

核心技术包括LLM规划器、基于图约束的代理调度、上下文感知提示、失败自检与重试机制，以及人与代理协同的交互。

**📊 数据集**

使用了自研的PySpark多轮仓库级别基准数据集，共10个任务，涵盖特征脚本、单元测试和配置文件的生成。

**📈 对比分析**

与固定顺序和随机顺序两种基线比较，采用pass@3指标，实验显示本框架平均0.833，显著高于顺序0.600和随机0.333。

**⚠️ 局限性**

局限包括对固定提示的依赖、下游验证延迟、数据集规模有限、缺乏长期记忆以及对不同编码标准的适配性不足。

---

## 21. Verified Design of Robotic Autonomous Systems using Probabilistic Model Checking

**arXiv ID:** 2601.10720 | [PDF](https://arxiv.org/pdf/2601.10720v1)

**作者:** Atef Azaiez `[一作]` (Norwegian University of Life Sciences), Alireza David Anisi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出一种在机器人自主系统（RAS）概念研究阶段引入形式化验证层的方法，先生成设计变体，利用概率模型检查（PMC）筛选满足安全可靠性属性的方案，然后将验证结果映射为可量化分数，结合多准则决策（MCDM）得到验证设计（Verified Design）。

**💡 创新点**

创新点在于：1）首次将PMC直接应用于概念选择阶段，实现“正确构造”的设计过滤；2）构建可逆映射函数将验证结果转化为可比较分数；3）将形式化验证与传统MCDM方法无缝结合，形成端到端的验证+评估流程；4）在农业机器人用例中演示完整实践，展示方法可行性。

**🔧 技术方法**

主要技术包括：概率模型检查（PRISM）、参数化离散时间马尔可夫链（DTMC）建模、PCTL属性编写、Python脚本自动化调用PRISM、逆向映射与分数归一化、MCDM（平均评分法）等。

**📊 数据集**

使用的“数据集”为人工设定的设计变体组合（感知、运动规划、操控三类子系统的多种实现方案），以及对应的转移概率表。未使用公开实验数据，而是基于仿真/文档估计的概率。

**📈 对比分析**

对比方法：将所有变体通过PMC过滤后，仅保留满足阈值的设计；随后对剩余变体按可验证准则评分并计算平均值，选取评分最高的前三个作为验证设计。性能方面，方法显著降低了不满足安全/可靠性要求的方案数量，并提供了客观评分支持决策，优于传统经验或经验性MCDM。

**⚠️ 局限性**

局限性：1）依赖准确的转移概率估计，若概率误差大则可能误判；2）状态爆炸问题仍可能限制模型规模；3）只覆盖可验证的准则，未对成本、能耗等非形式化准则做正式分析；4）方法需要人工映射与脚本编写，缺乏完整自动化工具；5）在实验用例中仅评估12个变体，扩展到更大设计空间需进一步验证。

---

## 22. Neural Induction of Finite-State Transducers

**arXiv ID:** 2601.10918 | [PDF](https://arxiv.org/pdf/2601.10918v1)

**作者:** Michael Ginn `[一作]` (University of Colorado), Mans Hulden `[通讯]` (New College of Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种利用循环神经网络学习的隐藏状态空间来自动构造无权有限状态转导器（FST）的方法，适用于未对齐的字符串对；

**💡 创新点**

创新点在于：①使用真实噪声数据而非小型语法；②提取FST而非FSA；③在RNN中引入自定义转导训练目标与谱正则化，促使隐藏状态呈现有限状态特性；

**🔧 技术方法**

核心技术包括：CRPAlign对齐算法、单层Elman RNN的转导训练目标、谱正则化、k‑means状态聚类、基于阈值的状态分裂与FSA最小化；

**📊 数据集**

实验使用了SIGMORPHON 2020的形态变形（24种语言）、Grapheme‑to‑Phoneme（15种语言）以及历史正则化（7种语言）数据集；

**📈 对比分析**

与基线OSTIA、DD‑OSTIA以及人类专家手工FST相比，本文方法在绝大多数数据集上均取得更高的准确率，尤其在形态变形任务中接近或达到专家级性能；

**⚠️ 局限性**

局限性包括：只能处理满足马尔可夫假设的无权任务，难以处理右侧依赖或双向上下文；对大规模数据的计算资源需求仍较高；

---

## 23. Resource-Bounded Martin-Löf Type Theory: Compositional Cost Analysis for Dependent Types

**arXiv ID:** 2601.10772 | [PDF](https://arxiv.org/pdf/2601.10772v1)

**作者:** Mirco A. Mannucci `[一作]` (HoloMathics), Corey Thuro `[通讯]` (University of Maryland)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

扩展资源限制的类型理论到Martin‑Löf类型理论，实现在类型中显式声明与输入尺寸相关的计算成本上界。

**💡 创新点**

引入依赖边界的Π型与Σ型、大小索引递归族、资源索引宇宙与格模态，并证明成本安全性、可解释性和初始性。

**🔧 技术方法**

利用依赖类型理论、格模态、资源格格 Lattice、预射范畴语义模型、群组化（groupoid‑valued presheaves）以及 CwF 框架。

**📊 数据集**

未使用传统数据集，而以向量运算、二分查找、归并排序、矩阵乘法等算法为案例研究。

**📈 对比分析**

通过类型推导给出 O(n)、O(log n) 等精确上界，证明评估成本不超过该上界；与手工分析相比，能在类型层面自动完成复杂度验证。

**⚠️ 局限性**

仅支持显式写入的多项式/对数形式边界；缺乏自动推断；仅处理直观的同伦结构，未覆盖更高阶同伦类型和更复杂的随机访问实现。

---

## 24. Reasoning Distillation for Lightweight Automated Program Repair

**arXiv ID:** 2601.10987 | [PDF](https://arxiv.org/pdf/2601.10987v1)

**作者:** Aanand Balasubramanian `[一作]` (Purdue University), Sashank Silwal `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对轻量级程序修复模型，通过在学生模型训练中加入符号推理监督，以提升bug类型分类的准确率。

**💡 创新点**

创新点在于将大型教师模型生成的结构化符号推理标签作为额外监督信号，利用知识蒸馏的方式在不增大模型规模的前提下增强小模型的语义理解。

**🔧 技术方法**

主要技术包括CodeT5-small学生模型、联合损失（标签交叉熵+符号推理交叉熵）、大型语言模型作为教师生成符号标签，以及对推理标签的严格校验。

**📊 数据集**

使用的数据集为IntroClass基准，包含284个短C程序，经过过滤后得到227个训练样本和57个验证样本，所有样本均带有单一bug类型标签。

**📈 对比分析**

比较方法为在同一训练设定下分别训练仅标签监督模型和加入符号推理监督模型；验证集结果显示准确率从0.491提升到0.544，宏F1从0.213提升到0.249，符号推理标签的macro F1为0.545。

**⚠️ 局限性**

局限性包括数据稀疏导致对少见bug类型提升有限，符号标签难以捕捉细粒度错误导致部分误分类，以及JSON式监督在低数据环境下难以学习。

---

## 25. Digital Metabolism: Decoupling Logic from Facts via Regenerative Unlearning -- Towards a Pure Neural Logic Core

**arXiv ID:** 2601.10810 | [PDF](https://arxiv.org/pdf/2601.10810v1)

**作者:** Mengmeng Peng `[一作]` (Northwestern Polytechnical University), He Sun `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 8686 | [OpenAlex ID](https://openalex.org/A5010862241)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对大语言模型进行有针对性的事实遗忘，构建了 Regenerative Logic‑Core Protocol (RLCP)，实现逻辑核心的纯净化和知识存储的解耦。

**💡 创新点**

创新点在于提出“数字代谢”理论，将事实遗忘与逻辑推理分离，并通过对抗性梯度反转实现权重层级的“代谢”，无需改动模型结构即可获得纯粹的推理核心。

**🔧 技术方法**

使用的技术包括对抗性梯度反转、RAG 适配、KL 正则、双流训练框架以及深层梯度反转，核心算法为 RLCP。

**📊 数据集**

实验使用 Qwen2.5‑0.5B 模型和 15 条高频城市‑国家事实集进行训练，评估数据集为 GSM8K 数学推理任务。

**📈 对比分析**

与原始模型、Just‑RAG 和 Unlikelihood 三种基线相比，RLCP 模型在不损失 RAG 召回的前提下将事实检索准确率降至 0%（近随机），同时在 GSM8K 上显著提升了链式推理结构，表现优于基线。

**⚠️ 局限性**

局限性包括跨域证明不充分、样本规模与模型规模有限、可能存在非代谢导致的输出风格改变、线性探针无法捕获所有信息以及理论与实践的差距。

---

## 26. BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search

**arXiv ID:** 2601.11037 | [PDF](https://arxiv.org/pdf/2601.11037v1)

**作者:** Shiyu Liu `[一作]` (Meituan Inc.), Jinsong Su `[通讯]` (Xiamen University)

**通讯引用:** 3902 | [OpenAlex ID](https://openalex.org/A5066326238)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 BAPO 框架，结合边界感知奖励与自适应奖励调制，提升代理式检索模型在多跳问答中的可靠性。

**💡 创新点**

创新点是引入群组边界感知奖励与自适应奖励调制机制，解决 RL 训练仅关注准确率导致模型忽略 “我不知道” 的问题。

**🔧 技术方法**

采用 GRPO 强化学习、ReAct 推理+检索环境、IDK 奖励机制以及自适应奖励调制器。

**📊 数据集**

使用 HotpotQA、Muisque、2WikiMultiHopQA、Bamboogle 四大多跳 QA 数据集，训练集包含 5000 条多跳样本。

**📈 对比分析**

与 Search‑R1、ReSearch、GRPO、TIR Prompt 等基线对比，BAPO 在可靠性指标上平均提升约 15.8 分，精度提升约 11.8%，准确率略有下降，但整体可靠性显著提升。

**⚠️ 局限性**

局限性包括仅在知识检索任务上验证，未探究在其他推理场景的通用性；实验规模仅至 14B 参数；使用本地 RAG，未覆盖真实 Web 搜索的噪声和动态性。

---

## 27. ZPD Detector: Data Selection via Capability-Difficulty Alignment for Large Language Models

**arXiv ID:** 2601.10986 | [PDF](https://arxiv.org/pdf/2601.10986v1)

**作者:** Bo Yang `[一作]` (Zhejiang University), Shijian Li `[通讯]` (Zhejiang University)

**通讯引用:** 7004 | [OpenAlex ID](https://openalex.org/A5103196339)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出并实现了ZPD Detector，一种基于模型能力与样本难度匹配的动态数据选择框架，帮助在有限数据预算下显著提升大型语言模型的训练效果。

**💡 创新点**

创新点包括：① 将教育理论Zone of Proximal Development（ZPD）应用于LLM数据选择；② 结合难度校准、基于IRT的一维能力估计和ZPDScore，实现动态、能力感知的样本筛选；③ 通过对比分析验证ZPD选样在不同数据比例下的优势。

**🔧 技术方法**

核心技术包括：① 平均token‑级NLL作为原始难度估计；② 错误反馈校准公式；③ 1PL Rasch模型对难度与模型能力进行联合建模；④ 基于p(1‑p)的ZPDScore用于衡量模型在样本上的不确定性；⑤ LoRA微调、精度/EM评估。

**📊 数据集**

使用的主要数据集为MedQA、GSM8K（公开问答基准）以及可控难度的合成数据集AgriQA（Synthetic）。

**📈 对比分析**

与Random、PPL、IFD、AlpaGasus、Data Whisperer等基线方法在1%–15%数据预算下进行对比；ZPD Detector在所有模型与任务上均达到或超过全量训练性能，明显优于其他方法，尤其在synthetic数据上表现最突出。

**⚠️ 局限性**

限制包括：① 仅使用单维能力估计，无法捕捉多维技能；② 校准和能力估计依赖固定的提示/解码设置，可能受任务或格式变化影响；③ 计算难度与校准需要额外前向推理，增加选择开销；④ 在数据分布漂移或噪声较大的情况下，难度估计误差可能导致选择失效。

---

## 28. Unit-Consistent (UC) Adjoint for GSD and Backprop in Deep Learning Applications

**arXiv ID:** 2601.10873 | [PDF](https://arxiv.org/pdf/2601.10873v1)

**作者:** Jeffrey Uhlmann `[一作]` (University of Missouri), Jeffrey Uhlmann `[通讯]` (University of Missouri)

**通讯引用:** 22035 | [OpenAlex ID](https://openalex.org/A5074812706)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了单元一致的伴随算子（UC adjoint）以及相应的UC梯度下降方法，解决了深度网络中因正向非线性同质性导致的参数尺度对优化的影响。

**💡 创新点**

创新点在于将尺度不变性要求迁移到反向传播和优化几何层面，构造了对任意对角尺度变换不变的伴随算子与预处理梯度的更新规则。

**🔧 技术方法**

采用了矩阵的可逆性、单位一致逆（unit-consistent inverse）、canonical decomposition、正交规范化等线性代数技术，并将其推广到卷积、偏置、残差连接及状态优化器（Momentum、Adam）等常见网络操作。

**📊 数据集**

本文未使用具体数据集，主要为理论推导与算法设计。

**📈 对比分析**

没有实验对比，缺乏数值验证；若在后续工作中加入实验，预计在保持相同尺度下可与标准SGD、BatchNorm、Adam 等方法实现等价或更优的收敛性能。

**⚠️ 局限性**

局限性包括：理论仍需在实际训练中验证其数值稳定性；对非同质激活函数或复杂归一化层的适用性待研究；实现与计算开销相对传统方法可能更高。

---

## 29. Multi-Agent Formation Navigation Using Diffusion-Based Trajectory Generation

**arXiv ID:** 2601.10725 | [PDF](https://arxiv.org/pdf/2601.10725v1)

**作者:** Hieu Do Quang `[一作]` (Hanoi University of Science and Technology), Quoc Van Tran `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于扩散模型的领导者-跟随者阵型控制框架，利用扩散策略预测领导者中点的动作序列并通过分布式距离控制器使跟随者维持阵型；

**💡 创新点**

创新点在于将扩散模型用于多代理长时域路径规划，预测整段动作序列以获得平滑连续轨迹，同时采用领导者中点的单体预测避免独立领导者冲突；

**🔧 技术方法**

核心技术包括扩散概率模型（DDPM）+CNN‑U‑Net架构的动作生成网络、FiLM调制、递归视窗控制、距离约束阵型跟踪控制；

**📊 数据集**

使用1878条手工生成的路径演示（通过基于势场的PAC收集）作为训练数据，测试集为100个包含3–5个圆形障碍的随机环境；

**📈 对比分析**

与MPPI和PAC基线对比，扩散方案在轨迹平滑、能耗和控制力度上优于MPPI，在成功率（62%）略低于PAC（100%），但在运动质量指标（曲率、冲击）方面最优；

**⚠️ 局限性**

局限主要在：①对训练数据的分布依赖强，导致在狭窄或极度混乱环境中碰撞率高；②CNN结构的感受野有限，导致对长距离约束识别不足，缺乏尖锐转弯所需的曲率。

---

## 30. Action Shapley: A Training Data Selection Metric for World Model in Reinforcement Learning

**arXiv ID:** 2601.10905 | [PDF](https://arxiv.org/pdf/2601.10905v1)

**作者:** Rajat Ghosh `[一作]` (Nutanix Inc), Debojyoti Dutta `[通讯]` (Nutanix Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并验证了一种名为 Action Shapley 的训练数据价值评估指标，用于在模型驱动强化学习（MBRL）中挑选高效的训练样本，并给出了随机化动态算法以降低传统 Shapley 计算的指数复杂度。

**💡 创新点**

创新点在于将 Shapley 值首次应用到强化学习环境模型训练数据的价值评估，并设计了基于阈值剪枝的随机化算法，使计算从 𝒪(2ⁿ) 降到 𝒪(ε) 的最佳情况，显著提升了计算效率。

**🔧 技术方法**

技术手段包括 Shapley 值理论、随机化动态剪枝算法、RBF 神经网络世界模型、SAC‑PID 与 PPO‑PID 两种强化学习策略，以及自编码器预训练。

**📊 数据集**

实验使用五个真实场景数据集：VM 右尺寸、负载平衡、数据库调优、Kubernetes（k8s）管理和数据中心冷却，每个数据集由若干时间序列（每条 1440 点）构成，涵盖 CPU 利用率、写入速率、延迟和温度等指标。

**📈 对比分析**

通过与基线（使用所有可用训练数据）对比，Action Shapley 在四个案例中显著提升累计奖励（最高可达 79% 的收益提升），并在实验中实现了 80% 以上的计算效率提升，验证了其优越性。

**⚠️ 局限性**

局限性包括：仍需人工设定错误边界（ε）和阈值 θ；在部分数据稀疏或高噪声场景下，Action Shapley 的优势不明显；算法的随机性导致在极少数案例中收益差异不显著。

---

## 31. Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration

**arXiv ID:** 2601.10744 | [PDF](https://arxiv.org/pdf/2601.10744v1)

**作者:** Sen Wang `[一作]` (East China Normal University), Xin Tan `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了长期记忆具身探索（LMEE）框架，并构建了LMEE-Bench基准，包括多目标导航与基于记忆的问答两大任务，旨在统一代理的探索认知与决策能力；

**💡 创新点**

核心创新在于通过多目标导航动态构建事件记忆库，并设计MemoryExplorer模型，该模型结合强化学习与多任务奖励函数，实现主动记忆检索与前沿探索；

**🔧 技术方法**

技术上采用多模态大语言模型（如Qwen2.5-VL-7B）与CLIP特征匹配实现记忆检索，利用GRPO进行强化微调，奖励函数融合动作准确度、前沿预测、答案精度与输出格式完整性；

**📊 数据集**

使用的主要数据集为HM3DSem生成的LMEE-Bench（约1982任务、377k条记录）和GOAT-Bench的Val Unseen子集，用于评估多目标导航与记忆问答性能；

**📈 对比分析**

与Explore‑EQA、3D‑Mem和RA‑Mem等基线对比，MemoryExplorer在LMEE-Bench的SPL/答案得分显著提升，GOAT‑Bench成功率提高至46.4%且SPL提升至28.03，表明主动记忆检索显著增强了探索效率；

**⚠️ 局限性**

局限性包括仅支持单轮工具调用、受限于离线训练与可用算力、在某些复杂环境下仍可能出现记忆检索错误或答案幻觉，且对连续动作的支持仍有限。

---

## 32. Redefining Machine Simultaneous Interpretation: From Incremental Translation to Human-Like Strategies

**arXiv ID:** 2601.11002 | [PDF](https://arxiv.org/pdf/2601.11002v1)

**作者:** Qianen Zhang `[一作]` (Chinese University of Hong Kong Shenzhen), Satoshi Nakamura `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于大语言模型的同步机器翻译框架，扩展了传统 READ/WRITE 策略，引入了句子切割、部分摘要、删减与代词化四种人类口译策略；

**💡 创新点**

创新点在于将人类口译中的高阶操作（Sentence_Cut、Partial_Summarization、Drop、Pronominalization）融入动作空间，并通过动作级统计指导模型在推理时动态选择；

**🔧 技术方法**

技术实现主要采用 decoder‑only LLM（如 Qwen3‑8B 与 GPT‑4o）与基于提示的动作驱动推理、以及 latency‑aware TTS 管道计算 TTS‑based LAAL；

**📊 数据集**

使用了 ACL60/60 与 MUST‑C 这两个英语-中文、英语-德语、英语-日语的并行语料库，包含原始、salami 分段与动作适配的目标；

**📈 对比分析**

与多种基线（mBART50 微调、TransLLaMA、few‑shot prompting、Salami、传统 action‑free 推理）对比，动作驱动推理在 BLEU/COMET 上均优于参考/Salami，且 LAAL 延迟更低，表现出最优的质量‑延迟平衡；

**⚠️ 局限性**

局限在于仅针对文本到文本的同步翻译，未直接处理语音输入，且动作策略需手工设计，未来可探索端到端语音-动作联合建模与自动化动作学习。

---

## 33. Where to Touch, How to Contact: Hierarchical RL-MPC Framework for Geometry-Aware Long-Horizon Dexterous Manipulation

**arXiv ID:** 2601.10930 | [PDF](https://arxiv.org/pdf/2601.10930v1)

**作者:** Zhixian Xie `[一作]` (Arizona State University), Wanxin Jin `[通讯]` (Arizona State University)

**通讯引用:** 406 | [OpenAlex ID](https://openalex.org/A5017350249)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种分层的 RL–MPC 框架，用于几何感知的长时程非抓取操纵任务。

**💡 创新点**

创新点在于提出“接触意图”接口，将接触位置与对象子目标分离，让高层 RL 负责几何与运动规划，低层 MPC 负责接触动力学；同时引入三组件对象中心化观测与双分支网络。

**🔧 技术方法**

使用了 PPO 进行高层策略训练，配合 ComFree‑MPC 的接触隐式优化，以及基于点云的关键点表示与双分支 PointNet++ 结构。

**📊 数据集**

数据集包括在仿真中随机生成的 12 种字母（训练 6 种、测试 6 种）和立方体任务，关键点数为 256；实验后置 3D 打印物体用于真实世界验证。

**📈 对比分析**

与端到端 RL 及 HACMan 层次结构比较，实验显示在 40 倍更少的高层决策步骤（≈15K vs 600K）内即可达到 100% 成功率，且在模拟/真实转移、外力、摩擦和执行器扰动下保持 100% 成功，显著优于基线。

**⚠️ 局限性**

局限性包括对高精度物体姿态估计的强依赖，以及使用离散关键点集合导致的多接触点时动作空间爆炸。

---

## 34. RobuMTL: Enhancing Multi-Task Learning Robustness Against Weather Conditions

**arXiv ID:** 2601.10921 | [PDF](https://arxiv.org/pdf/2601.10921v1)

**作者:** Tasneem Shaffee `[一作]` (Brown University), Sherief Reda `[通讯]` (Brown University)

**通讯引用:** 4530 | [OpenAlex ID](https://openalex.org/A5015719218)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现RobuMTL框架，利用动态选择层次化LoRA专家以提升多任务学习在恶劣天气下的鲁棒性；

**💡 创新点**

将MoE路由思想与LoRA结合，提出单次路由的DMLS和MEPF加权融合，避免传统MoE多层路由导致的不稳定与任务冲突，并支持混合扰动下的专家集合；

**🔧 技术方法**

使用Swin‑Tiny骨干网络、层次化低秩LoRA、CNN+SE的动态专家选择器DMLS、加权融合模块MEPF，以及MoE启发式专家聚合技术；

**📊 数据集**

在PASCAL VOC与NYUD‑v2两大多任务基准上，并通过对噪声、模糊、雨雪等五种人工扰动生成的数据集进行训练与评估；

**📈 对比分析**

与单任务、全微调MTL、MTLoRA、MTLMoE以及多项SOTA方法对比，RobuMTL在PASCAL平均提升2.8%鲁棒性、NYUD‑v2提升9.7%，在混合扰动下提升44.4%，且参数与计算量相对友好；

**⚠️ 局限性**

对真实气象环境的泛化尚待验证，专家数量与扰动类型需人工预设，推理时仍需额外的路由与融合开销。

---

## 35. SwiftKV: An Edge-Oriented Attention Algorithm and Multi-Head Accelerator for Fast, Efficient LLM Decoding

**arXiv ID:** 2601.10953 | [PDF](https://arxiv.org/pdf/2601.10953v1)

**作者:** Junming Zhang `[一作]` (Huazhong University of Science and Technology), Xiangshui Miao `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 10364 | [OpenAlex ID](https://openalex.org/A5100607859)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SwiftKV Attention及其多头加速器SwiftKV-MHA，专为边缘加速器设计的单通道低延迟注意力算法与多头并行解码硬件。

**💡 创新点**

创新点在于：①单通道流水线单遍注意力算法，避免打分量化、块软max及第二遍遍历；②使用32位定点实现高精度注意力，重用低位整数GEMV硬件；③SKV处理单元同时支持高精度注意力与低精度GEMV；④实现Decoder‑Specialized RoPE，显著减少键的重新编码。

**🔧 技术方法**

技术包括：定点数计算（FXP32 Q15.17），按token流水线计算，2^x LUT近似指数，硬件共享MAC阵列，单元分区KV缓存，GEMV与Attention共享算子，FPGA实现（Xilinx U55C）与HBM。

**📊 数据集**

使用LLaMA2‑7B与ChatGLM‑6B两大模型进行推理验证，并在PG‑19数据集上抽取长度512序列进行Top‑k准确率评估。

**📈 对比分析**

与Flash Attention、Streaming Attention等基线在相同FPGA平台下比较，SwiftKV Attention实现7.16×速度提升、Attention延迟下降13.48×；SwiftKV‑MHA在LLaMA‑2‑7B和ChatGLM‑6B上，单token延迟12.3 ms，速度81.5 tokens/s，比SOTA提升17.4%，能耗效率提升1.98×。

**⚠️ 局限性**

局限性在于目前仅针对6B–10B规模模型，硬件资源消耗仍高（占用约50% DSP），对更大规模模型或更高精度量化的适配尚未验证。

---

## 36. MMedExpert-R1: Strengthening Multimodal Medical Reasoning via Domain-Specific Adaptation and Clinical Guideline Reinforcement

**arXiv ID:** 2601.10949 | [PDF](https://arxiv.org/pdf/2601.10949v1)

**作者:** Meidan Ding `[一作]` (Shenzhen University), Linlin Shen `[通讯]` (Shenzhen University)

**通讯引用:** 10908 | [OpenAlex ID](https://openalex.org/A5019313200)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MMedExpert-R1框架，解决医学多模态模型在复杂临床推理任务中表现欠佳的问题。

**💡 创新点**

创新点在于结合领域特定适配（DSA）、指南基础优势（GBA）与冲突感知融合（TIES-Merging），在强化学习中引入多视角优势函数，显著提升多专科推理能力。

**🔧 技术方法**

使用LoRA适配技术进行专科专家训练，改进的GRPO（GBA）作为优势函数，冲突感知融合算法，构建MMedExpert数据集以及MedEvalKit评测框架。

**📊 数据集**

使用了自构建的MMedExpert数据集（约3.9K多模态推理样本，划分为成人医学、儿科、脑与感官、肿瘤与手术四大子集），并在MedXpert-MM、PMC‑VQA、OmniMedVQA、GMAI‑MM等公开基准上进行评测。

**📈 对比分析**

通过与多种SOTA MedVLM（如MedVLM‑R1、Med‑R1、MedVLThinker、LLaVA‑Med、HuatuoGPT‑V、BiMediX2、Lingshu）在2B和7B参数规模下对比，7B版MMedExpert‑R1在MedXpert‑MM 27.50、PMC‑VQA 56.78、OmniMedVQA 83.03、GMAI‑MM 52.10等指标均实现或逼近最优，显著优于竞争者。

**⚠️ 局限性**

局限性包括专科细粒度不足（仅覆盖四大子科），评测范围主要集中在现有基准，未覆盖更多任务如病历摘要、纵向比较等；未来可细化子领域、扩展评测集以进一步验证性能。

---

## 37. Multivariate LSTM-Based Forecasting for Renewable Energy: Enhancing Climate Change Mitigation

**arXiv ID:** 2601.10961 | [PDF](https://arxiv.org/pdf/2601.10961v1)

**作者:** Farshid Kamrani `[一作]` (Carleton University), Kristen Schell `[通讯]` (Carleton University)

**通讯引用:** 386 | [OpenAlex ID](https://openalex.org/A5088558963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并应用一种多变量长短期记忆网络（M-LSTM）对阿尔伯塔省PV发电进行时序预测，以提升日间与实时能源调度的准确性和可靠性。

**💡 创新点**

创新点在于：①将邻近地区的历史发电数据联合输入，实现跨区域的时间空间关联建模；②针对日夜交替的光伏发电，采用先验知识在暗时段手动设为零，减少无意义预测；③将预测结果直接嵌入经济调度（ED）流程，量化其对燃煤/燃气发电量、CO₂排放、负荷切断和系统成本的影响。

**🔧 技术方法**

采用双层ReLU激活的LSTM网络，后接Dropout和全连接层；训练使用Adam优化器、MSE损失；数据归一化、重尺度处理；对比基线模型使用K-means聚类和月均值预测。

**📊 数据集**

使用阿尔伯塔省电力系统运营商（AESO）公开的全年每小时PV发电数据（8760条样本），涉及三大规划区，共三条特征。

**📈 对比分析**

与K-means和月均值预测对比，M-LSTM在NMAE上最低（0.46），经济调度成本最低（$43,490），天然气发电量减少58%（从263.7 MW降至108.7 MW），CO₂排放降低58%（从53,267 kg降至21,957 kg），负荷切断完全消失，系统可靠性显著提升。

**⚠️ 局限性**

局限性包括：仅验证单一RES（PV）且仅在阿尔伯塔省三区；未探讨模型对其他可再生能源或更大范围区域的泛化能力；模型训练与部署的实时性与可扩展性未做详细评估；缺乏对不同超参数组合、不同深度网络结构的系统性比较。

---

## 38. Children's Expectations, Engagement, and Evaluation of an LLM-enabled Spherical Visualization Platform in the Classroom

**arXiv ID:** 2601.11060 | [PDF](https://arxiv.org/pdf/2601.11060v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 39. Beyond Max Tokens: Stealthy Resource Amplification via Tool Calling Chains in LLM Agents

**arXiv ID:** 2601.10955 | [PDF](https://arxiv.org/pdf/2601.10955v1)

**作者:** Kaiyu Zhou `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**通讯引用:** 5752 | [OpenAlex ID](https://openalex.org/A5101720092)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型代理的工具调用环节提出了一种隐蔽的、多轮经济型拒绝服务攻击，能在保持任务完成正确性的前提下，显著拉高模型的生成长度与系统资源消耗。

**💡 创新点**

创新点在于：①将攻击重点从单轮提示或RAG上下文转移到工具层；②设计了可在保持MCP协议兼容的前提下进行文本字段和返回策略编辑的通用恶意模板；③采用蒙特卡罗树搜索（MCTS）自动化优化模板，使攻击在多轮交互中最大化资源消耗并保持高成功率。

**🔧 技术方法**

主要技术包括：MCP协议工具服务器的文本可编辑模板、MCTS优化器、蒙特卡罗树搜索的两阶段评估、基于文本的校准序列与段索引控制多轮生成。

**📊 数据集**

使用了两个公开基准数据集：ToolBench（105工具+261查询）和BFCL（80工具+203查询），并在六种主流LLM（Qwen‑3‑32B、Llama‑3.3‑70B‑Instruct、Llama‑DeepSeek‑70B、Mistral‑Large、Seed‑32B、GLM‑4.5‑Air）上进行评估。

**📈 对比分析**

与无攻击基线和现有单轮攻击Overthink对比，攻击在所有模型和数据集上平均token长度提升至≈60k–90k（最多×658），ASR保持≈80–96%，能耗提升约100–560×，GPU KV缓存占用提升至35–74%，并使并发吞吐量下降≈50%。

**⚠️ 局限性**

局限性包括：①需要对工具服务器（MCP）进行直接篡改，攻击者需具备服务器控制权；②依赖特定的协议规范与工具实现，跨协议迁移需要额外工作；③在模型或任务对多轮交互不敏感时，攻击效果会受限。

---

## 40. One Model, Many Behaviors: Training-Induced Effects on Out-of-Distribution Detection

**arXiv ID:** 2601.10836 | [PDF](https://arxiv.org/pdf/2601.10836v1)

**作者:** Gerhard Krumpl `[一作]` (Graz University of Technology), Horst Possegger `[通讯]` (Graz University of Technology)

**通讯引用:** 4086 | [OpenAlex ID](https://openalex.org/A5039382695)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在固定 ResNet‑50 和 ImageNet 的条件下，系统评估了 56 种不同训练策略下 21 种 post‑hoc OOD 检测方法，探究训练方法与 OOD 性能的关系。

**💡 创新点**

首次揭示 ID 准确率与 OOD 性能呈非单调的“先升后降”趋势，并发现模型与检测方法的交互项解释了 20%+ 的方差，说明训练策略对 OOD 检测具有决定性影响。

**🔧 技术方法**

使用 21 种 post‑hoc 检测技术（如 MSP、KNN、GRAM、SCALE 等），对 56 个训练策略（数据增强、对抗训练、对比学习等）进行 AUROC/FPR95 评估，并采用三因素 ANOVA 分析方差来源。

**📊 数据集**

ID 数据集为 ImageNet，OOD 测试集包括 8 个不同类别（近似 OOD：SSB‑Hard、NINCO；远程 OOD：iNaturalist、Textures、OpenImage‑O；极端 OOD：MNIST、Fashion‑MNIST；合成 OOD：NINCO synthetic），覆盖语义距离从近到极端的多样性。

**📈 对比分析**

通过全排列评估比较，模型增强型方法（SCALE、NNGuide、ASH）和统计方法（GRAM、fDBD）在多种训练策略下表现最稳健；高 ID 准确率并不保证更好 OOD，性能会随训练策略波动。

**⚠️ 局限性**

实验仅局限于 ResNet‑50 结构，无法直接推广到 Transformer 等其他网络；此外仅考虑无 OOD 训练，未来需要在多模型、多结构以及更广泛训练策略下进一步验证。

---

## 41. Multi-Artifact Analysis of Self-Admitted Technical Debt in Scientific Software

**arXiv ID:** 2601.10850 | [PDF](https://arxiv.org/pdf/2601.10850v1)

**作者:** Eric L. Melin `[一作]` (Boise State University, Oak Ridge National Laboratory), Addi Malviya-Thakur `[通讯]` (Oak Ridge National Laboratory, University of Tennessee)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在科学软件中自我承认的技术债（SATD），构建跨 artifact 的多源数据集，并训练分类器识别并量化科学债务。

**💡 创新点**

提出将科学债务作为独立 SATD 类别，首次创建涵盖代码注释、提交信息、拉取请求、问题追踪等四种 artifact 的多源 SATD 数据集，并证明传统 SATD 类别无法覆盖此类债务。

**🔧 技术方法**

采用 Transformer（Falcon3‑3B‑Instruct）多任务学习框架，结合伪标注、主动学习与语义清洗技术进行数据扩增和模型训练。

**📊 数据集**

融合 SATDAUG、CppSATD、Awon 等公开数据集，并在 DOE CASS 项目中扩展 23 个科学软件项目的多 artifact 样本，最终形成 115,524 条标注实例。

**📈 对比分析**

通过与多种 Transformer、BERT、CodeBERT 等模型对比，Falcon3‑3B‑Instruct 在宏 F1 0.8255、科学债务 F1 0.6353 上表现最佳；传统模型在识别科学债务时误分类率高达 74%，表明需加入新类别。

**⚠️ 局限性**

科学债务样本极为稀少导致分类性能受限；外部数据集可能带来标签噪声；实验仅覆盖 23 个公开 CASS 项目，缺乏跨领域、跨语言的验证。

---

## 42. What Matters in Data Curation for Multimodal Reasoning? Insights from the DCVLR Challenge

**arXiv ID:** 2601.10922 | [PDF](https://arxiv.org/pdf/2601.10922v1)

**作者:** Yosub Shin `[一作]` (University of Hawai'i at Manoa), Igor Molybog `[通讯]` (University of Hawai'i at Manoa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在NeurIPS 2025 DCVLR 挑战框架下的多模态推理数据策划，系统评估了难度筛选、数据规模、以及多样性/合成增强对性能的影响。

**💡 创新点**

证明在固定训练协议下，难度筛选是提升推理准确率的关键因素；数据规模增大后仅能降低方差，无法显著提升平均得分；常用的多样性与合成增强在此 regime 下无效甚至有害。

**🔧 技术方法**

使用 Qwen2.5‑VL‑7B‑Instruct 的固定 fine‑tune 方案，基于多次随机解码评估样本难度；实现聚类、类别平衡等多样性策略，并混合 Walton 与 CoSyn‑400k 合成数据。

**📊 数据集**

以 Walton Multimodal Cold Start 为基底，挑选 1k 难度筛选样本；对比 10k Walton、LiveXivTQA 等 benchmark；使用 DCVLR 提供的 10 个多模态推理基准进行评估。

**📈 对比分析**

在同一训练与评估管线下，比较不同难度阈值、样本规模、混合比例等；结果显示中等难度样本最优，1k 样本即可达到与 10k 相当的整体得分，而多样性/CoSyn 方案未提升性能，甚至略有下降。

**⚠️ 局限性**

未尝试改变训练算法或超参；多样性实验仅覆盖常见方法；结论仅适用于固定训练协议与当前基准，缺乏跨模型、跨任务的普适性验证。

---

## 43. Self-Augmented Mixture-of-Experts for QoS Prediction

**arXiv ID:** 2601.11036 | [PDF](https://arxiv.org/pdf/2601.11036v1)

**作者:** Kecheng Cai `[一作]`, Xia Chen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种自增式混合专家模型（SA‑MoE），用于在用户‑服务交互稀疏的情境下预测服务质量（QoS）指标；

**💡 创新点**

创新点在于：①自补填机制——利用模型自身的预测结果对缺失条目进行迭代填充；②伪标签策略——将部分预测结果作为额外监督信号；③在MoE框架中实现专家协作与迭代细化，从而显著提升稀疏环境下的泛化能力；

**🔧 技术方法**

采用低秩矩阵分解提取协同特征、双塔特征融合（ID、侧信息、MF嵌入）以及神经Mixture‑of‑Experts（MoE）预测网络，并结合自增补填与伪标签训练流程；

**📊 数据集**

使用公开的WS‑DREAM数据集，包括响应时间（RT）和吞吐量（TP）两项QoS指标；

**📈 对比分析**

在多种稀疏率（2.5%、5%、7.5%、10%）下，与UIPCC、PMF、NFMF、RAHN、GraphMF、TAN等基线模型对比，SA‑MoE在响应时间任务的MAE/RMSE均明显优于基线，吞吐量任务中在高稀疏度下亦表现更佳；

**⚠️ 局限性**

局限性包括：①在吞吐量预测中补填可能引入较大噪声导致性能波动；②模型依赖预先完成的矩阵分解，难以实时适应动态的用户/服务集；③伪标签误差可能在训练中累积影响最终精度。

---

## 44. Unified Optimization of Source Weights and Transfer Quantities in Multi-Source Transfer Learning: An Asymptotic Framework

**arXiv ID:** 2601.10779 | [PDF](https://arxiv.org/pdf/2601.10779v1)

**作者:** Qingyue Zhang `[一作]` (Tsinghua University), Shao-Lun Huang `[通讯]` (Tsinghua University)

**通讯引用:** 1898 | [OpenAlex ID](https://openalex.org/A5088293566)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一优化源任务权重与转移量的理论框架UOWQ，用于多源迁移学习与多任务学习。

**💡 创新点**

创新点在于将源权重与转移量联合最优化，利用Kullback–Leibler基准的泛化误差进行渐近分析，证明在权重最优时使用全部源样本始终最优，并给出闭式解与凸优化求解方法。

**🔧 技术方法**

核心技术包括：参数估计视角、KL基泛化误差、极限正态性、Fisher信息矩阵、二次规划求解权重、动态样本权重更新、深度模型梯度下降。

**📊 数据集**

实验数据集为DomainNet和Office‑Home两大跨域图像分类基准。

**📈 对比分析**

与多种基线（H‑ensemble、MCW、MADA、WADN、OTQMS、SGDA等）以及多任务权重方法（MGDA‑UB、GradNorm、PCGrad、CAGrad等）对比，UOWQ在10‑shot迁移任务上平均提升约1.3–1.4%，在多任务学习中比最佳基线提升0.4–0.7个百分点，且保持了较好的鲁棒性和计算效率。

**⚠️ 局限性**

局限包括：仅考虑同一参数空间的任务；对极端源与目标差异仍需经验验证；权重更新依赖目标样本的初始估计，少样本时可能不稳定；理论推导基于大样本渐近近似，有限样本下可能偏差。

---

## 45. SurfSLAM: Sim-to-Real Underwater Stereo Reconstruction For Real-Time SLAM

**arXiv ID:** 2601.10814 | [PDF](https://arxiv.org/pdf/2601.10814v1)

**作者:** Onur Bagoren `[一作]` (University of Michigan), Katherine A. Skinner `[通讯]` (University of Michigan)

**通讯引用:** 425 | [OpenAlex ID](https://openalex.org/A5002924029)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 sim‑to‑real 的水下立体视差估计训练框架，并基于此与 IMU、DVL、气压计融合的实时 SLAM 系统 SurfSLAM，用于在复杂水下环境（如船舶残骸）中实现精确定位与稠密重建。

**💡 创新点**

创新点包括：①完整的训练时数据增强管道（水柱、折射光斑、方向光、光照、悬浮颗粒），①自监督细化与 Occam 先验相结合的损失函数，②将高精度的立体深度作为全局注册因子与声学‑惯性轨迹融合。

**🔧 技术方法**

采用了 Transformer‑backbone 的 DEFOM‑Stereo（ViT‑S/L）模型，结合自监督稀疏重投影、SSIM、Warp、Occam 以及平滑正则化；在 SLAM 端使用 iSAM2 轨迹优化、GICP 视觉配准和 Huber 鲁棒因子。

**📊 数据集**

使用了两个新数据集：UWSim（通过 Omniverse RTX 生成的带水下渲染的合成数据）和 SUDS（真实船舶残骸的 24000+ 立体图像、地面真值、轨迹）。

**📈 对比分析**

与多种基准（FoundationStereo、DEFOM‑Stereo、RAFT‑Stereo、SVIn2、ORB‑SLAM3、DROID‑SLAM 等）比较，SurfSLAM 在立体视差精度（EPE、BP‑X）和 SLAM 跟踪误差（APE）以及稠密地图完成度上均优于现有方法，特别是在低纹理和水柱干扰场景下表现突出。

**⚠️ 局限性**

主要限制是对高度浊度环境下的地面真值重建困难；系统仍依赖视觉全局配准，遇到长时间无纹理或大尺度漂移时可能需要更稳健的束平衡或三维声波传感器补偿。

---

## 46. Two Complexity Results on Spanning-Tree Congestion Problems

**arXiv ID:** 2601.10881 | [PDF](https://arxiv.org/pdf/2601.10881v1)

**作者:** Sunny Atalig `[一作]` (University of California at Riverside), Gregory Zhu `[通讯]` (University of California at Riverside)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明了在最大度为3的图上，生成最小拥塞生成树（STC）问题仍为NP‑hard，并给出一种基于cactus表示与动态规划的线性时间算法，在K‑边连通图中判断其拥塞是否为K。

**💡 创新点**

创新点在于：①首次完成了从最大度为8到最大度为3的NP‑hard性完整划分；②提出了利用cactus结构和“hub”概念将全局拥塞约束转化为局部可递归求解的技术；③实现了对K‑边连通图的O(m)决策算法，克服了传统方法中K‑cut非层状的难题。

**🔧 技术方法**

主要技术包括：
- cacti（cactus）表示法用于捕捉K‑cut的结构；
- 基本K‑cut的层次化（laminar）划分；
- 定义“hub”节点并构建递归关系，形成动态规划；
- 通过根化图和递归合并，最终判断全图是否存在拥塞为K的生成树。

**📊 数据集**

本文为理论算法研究，未使用实验数据集；所有结果均通过严谨证明获得。

**📈 对比分析**

与现有方法相比，传统的STC NP‑hard性证明多基于无度数限制的图；本文的算法在K‑边连通图上实现线性时间（O(m)），显著提升了决策阶段的效率；同时，NP‑hard性证明填补了度数3至7区间的空白。

**⚠️ 局限性**

局限性包括：
- 线性时间算法仅适用于决策版本（判断拥塞是否≤K），而非直接构造最小拥塞生成树；
- 对一般图（非K‑边连通）仍保持NP‑hard，算法无法直接推广；
- 仅提供理论复杂度，缺乏实验验证和对实际网络实例的评估。

---

## 47. Adaptive Sliding Mode Control for Vehicle Platoons with State-Dependent Friction Uncertainty

**arXiv ID:** 2601.10724 | [PDF](https://arxiv.org/pdf/2601.10724v1)

**作者:** Rishabh Dev Yadav `[一作]` (International Institute of Information Technology), Rishabh Dev Yadav `[通讯]` (International Institute of Information Technology)

**通讯引用:** 129 | [OpenAlex ID](https://openalex.org/A5070582236)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种新的自适应滑模控制器，用于在未知摩擦力下实现多车队车辆的速度跟踪和安全间距保持。

**💡 创新点**

创新点在于：①采用两阶段控制（运动学层+自适应滑模层）能够同时处理不确定惯性参数和状态相关摩擦；②不需要先验摩擦模型或参数，利用自适应增益实时估计不确定性；③通过Lyapunov分析证明闭环系统实现统一最终有界(UUB)。

**🔧 技术方法**

技术方法包括：自适应滑模控制、滑移面设计与自适应增益更新、Lyapunov稳定性证明、Gazebo仿真环境与TurtleBot3模型、饱和函数替代符号函数以降低抖动。

**📊 数据集**

数据集方面：未使用公开数据集，而是在Gazebo中自建四象限摩擦场（μ1=0.1，μ2=0.13）和速度破坏器，生成的路径为figure‑eight 轨迹，仿真涉及三台机器人。

**📈 对比分析**

与传统自适应滑模控制进行对比，采用RMS误差和间距误差作为评价指标。结果显示，在摩擦变化较大的第三象限中，所提控制器的轨迹误差和间距误差分别下降约30%和25%，整体表现优于标准方法。

**⚠️ 局限性**

局限性包括：仅在仿真中验证，未在真实机器人平台上实验；参数选择仍需经验调优；对更大规模车队或不同机器人平台的可扩展性尚未验证。

---

## 48. A PAC-Bayesian Analysis of Channel-Induced Degradation in Edge Inference

**arXiv ID:** 2601.10915 | [PDF](https://arxiv.org/pdf/2601.10915v1)

**作者:** Yangshuo He `[一作]` (Zhejiang University), Jingge Zhu `[通讯]` (University of Melbourne)

**通讯引用:** 494 | [OpenAlex ID](https://openalex.org/A5007978311)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对在无噪声环境下训练、在带噪声无线边缘网络中部署的神经网络性能下降进行PAC‑Bayes分析，并给出了通用误差上界；

**💡 创新点**

创新点在于将通道模型视为增广网络中的随机层，构造通道惩罚项，推导针对通道失真最紧的PAC‑Bayes界，并基于该界提出无需即时信道信息的通道感知训练算法；

**🔧 技术方法**

主要技术包括PAC‑Bayes理论、增广网络建模、熵与KL散度分析、对BEC与Rayleigh通道的闭式误差项求解、变分推理（高斯后验）以及随机梯度下降；

**📊 数据集**

实验数据集为MNIST（FCN‑4、CNN‑4）和CIFAR‑10（CNN‑9）；

**📈 对比分析**

与传统经验风险最小化（ERM）对比，利用0‑1损失的期望风险评估，通道感知算法在各类通道（BEC、Rayleigh）下的风险显著低于ERM，并且给出的上界对实际风险具有可观的保守性；

**⚠️ 局限性**

局限性包括：上界在某些场景下相对松散；复杂衰落模型下通道惩罚难以解析；算法依赖已知通道统计且需要近似梯度范数K，可能在大模型或高度非线性网络中效果有限。

---

## 49. ICONIC-444: A 3.1-Million-Image Dataset for OOD Detection Research

**arXiv ID:** 2601.10802 | [PDF](https://arxiv.org/pdf/2601.10802v1)

**作者:** Gerhard Krumpl `[一作]` (Graz University of Technology), Horst Possegger `[通讯]` (Graz University of Technology)

**通讯引用:** 4086 | [OpenAlex ID](https://openalex.org/A5039382695)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个针对工业图像分类的、可无污染划分ID/OOD的大规模数据集ICONIC-444，并在其上对22种后置OOD检测方法做基线评估

**💡 创新点**

创新点包括：①提供3.1M张高分辨率、无ID污染的工业图像和4层级的OOD难度；②设计了四个覆盖不同复杂度的基准任务；③首次引入FPR99作为衡量高TPR下误检率的指标，强调实际应用中的安全性

**🔧 技术方法**

采用了ResNet18与Compact Transformer (CCT) 两种架构，使用OpenOOD库中的22种后置OOD检测方法（基于softmax、logit、特征空间、模型增强等）

**📊 数据集**

使用ICONIC-444数据集（含四个ID任务：Almond、Wheat、Kernels、Food-grade）以及公开数据集和合成OOD样本做验证和测试

**📈 对比分析**

实验表明，特征层方法（如GRAM、ATS、KNN）在大多数OOD类别上优于传统softmax/Logit方法，但在near‑和far‑OOD场景下FPR仍高达30%+，即使是最优方法在FPR99上也未能满足安全阈值；更复杂网络并未显著提升性能

**⚠️ 局限性**

局限性：①数据集虽然无污染，但仍属于单一工业环境，泛化到其他领域受限；②FPR99指标对样本量要求高，需大量评估样本；③基线方法多为后置且不再训练，可能无法充分挖掘模型潜力；④当前方法在near‑OOD细粒度任务上仍表现不佳

---

## 50. Hidden-in-Plain-Text: A Benchmark for Social-Web Indirect Prompt Injection in RAG

**arXiv ID:** 2601.10923 | [PDF](https://arxiv.org/pdf/2601.10923v1)

**作者:** Haoze Guo `[一作]` (University of Wisconsin), Ziqi Wei `[通讯]` (University of Wisconsin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个针对 Web 版 RAG 的间接提示注入与检索中毒的可复现基准与测试平台。

**💡 创新点**

首次将 HTML/Markdown 清洗、Unicode 规范化和归属限定提示结合起来，对社交网络网页载体进行系统评估。

**🔧 技术方法**

使用 BM25 与稠密检索器、Llama‑3、Mistral 等 LLM，结合 DOMPurify 清洗、NFKC 规范化以及引用限定的 prompt。

**📊 数据集**

自建 6,200 条社交网页样本，覆盖隐藏标签、off‑screen CSS、alt 文本、ARIA、零宽字符等载体，另包含 PDF/SVG。

**📈 对比分析**

通过 ASR、ΔMRR@10、ΔnDCG@10、回答率、延迟等指标比较四种防御组合，发现全部防御组合在保持近乎无延迟的同时将 ASR 降至 <5%。

**⚠️ 局限性**

仅评估静态载体，未覆盖 JS 渲染、OCR 噪声或完全知情的提示攻击，且防御可能影响检索召回率。

---

## 51. Bridging Psychological Safety and Skill Guidance: An Adaptive Robotic Interview Coach

**arXiv ID:** 2601.10824 | [PDF](https://arxiv.org/pdf/2601.10824v1)

**作者:** Wanqi Zhang `[一作]` (University of Tennessee), Marielle Santos `[通讯]` (University of Tennessee)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5113028390)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过三阶段迭代设计，构建了基于人本治疗（PCT）的社交机器人面试教练，力求在提供心理安全与技能指导之间实现动态平衡。

**💡 创新点**

创新点包括揭示“安全–指导缺口”和“支架悖论”，并提出Agency‑Driven Interaction Layer及Adaptive Scaffolding Ecosystem，使机器人能够根据用户情绪与学习状态实时切换从“抚慰者”到“指导者”的角色。

**🔧 技术方法**

技术上使用Mist II社交机器人与OpenAI GPT‑4 Realtime API相结合，前端采用React.js/Node.js Web应用与WebRTC实现低延迟的多模态交互，并通过脚本化指令控制机器人表情与动作。

**📊 数据集**

实验数据来源于8名大学生的面试模拟记录，包括RoSAS、B–L RI、MASI量表以及访谈转录文本；未使用公开的大规模语料库或外部数据集。

**📈 对比分析**

采用within‑subjects和between‑subjects对比设计，比较了心理安全、焦虑水平、认知负荷与用户满意度。结果显示Agency‑Driven模式在保持温暖与信任的同时，显著降低社交与沟通焦虑，并优于单纯PCT或强支架策略。

**⚠️ 局限性**

局限性主要在于样本量小、实验时间短、且评估主要基于自我报告量表，缺乏客观面试评分与生理指标，未来需扩大样本、引入长期跟踪与更精确的客观测量。

---

## 52. Classification of Chest XRay Diseases through image processing and analysis techniques

**arXiv ID:** 2601.10913 | [PDF](https://arxiv.org/pdf/2601.10913v1)

**作者:** Santiago Martínez Novoa `[一作]` (Universidad de los Andes), Jeremias Kramer `[通讯]` (Universidad de los Andes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并比较了三种深度学习模型（DenseNet121、EfficientNetV2、ViT）在 CheXpert 数据集上对 5 种常见肺部疾病（No Finding、Cardiomegaly、Edema、Pneumothorax、Pleural Effusion）的多分类任务，评估了不同学习率对模型性能的影响。

**💡 创新点**

创新点在于：①将五个临床重要且数据量充足的病种聚焦为统一分类目标，②系统比较了 CNN 与 Transformer 架构在同一数据集与任务上的表现，③通过实验验证学习率微调对预训练模型迁移学习的关键性，④提供了开源 Web 应用实现以便快速部署。

**🔧 技术方法**

使用的技术包括：预训练模型迁移学习（DenseNet121、EfficientNetV2、ViT），交叉熵损失 + Adam 优化器，图像预处理与增强（尺寸调整、水平/垂直翻转、归一化），学习率调节（0.01 与 0.0001）以及标准评估指标（准确率、精确率、召回率、F1）。

**📊 数据集**

数据集为公开的 CheXpert，原始 224,316 张胸部 X‑ray，作者筛选并平衡了 5 类病种（每类 8,193 张训练样本，5,595 张验证与测试），只保留正例以便聚焦分类任务。

**📈 对比分析**

在相同实验设置下，学习率 0.0001 通常优于 0.01；DenseNet121 与 EfficientNetV2 在 5 类任务上取得最高的 F1（约 0.73–0.76），ViT 在 0.01 下几乎无法收敛，0.0001 时仍落后于两者。整体结论是较低学习率更稳定，预训练 CNN 结构在小样本场景中更具鲁棒性。

**⚠️ 局限性**

局限性包括：①仅关注 5 种病种，无法覆盖 CheXpert 所含的 14 种观察；②ViT 需要更大数据量或更精细的超参调优；③缺乏模型可解释性与临床专家评估；④仅尝试了两种学习率，其他超参（批大小、正则化等）未系统探索。

---

## 53. Can Vision-Language Models Understand Construction Workers? An Exploratory Study

**arXiv ID:** 2601.10835 | [PDF](https://arxiv.org/pdf/2601.10835v1)

**作者:** Hieu Bui `[一作]` (Villanova University), Arash Tavakoli `[通讯]` (Villanova University)

**通讯引用:** 335 | [OpenAlex ID](https://openalex.org/A5066478442)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在不进行域适配的前提下，评估了三种通用视觉语言模型（GPT‑4o、Florence 2、LLaVA‑1.5）在静态施工现场图像中识别工人动作与情绪的能力。

**💡 创新点**

创新点在于首次用通用VLM对建筑工地行为与情绪进行零样本评估，并构建了基于1000张图像的多标签动作/情绪标注基准。

**🔧 技术方法**

技术主要采用VLM的零样本推断（Direct Generation）和标准分类评估指标（精确率、召回率、F1、准确率）。

**📊 数据集**

使用的实验数据集为从公开资源（Unsplash、Roboflow）收集的1000张施工现场图像，并重新标注了10个动作与10个情绪标签。

**📈 对比分析**

通过对比精确率、召回率、F1和准确率，GPT‑4o在动作识别上平均F1≈0.756、准确率≈0.799，情绪识别上F1≈0.712、准确率≈0.773，明显优于Florence 2和LLaVA‑1.5。

**⚠️ 局限性**

主要局限包括仅使用单帧静态图像忽视时间上下文、数据规模有限且类别分布不均、模型未进行建筑域微调、情绪识别易混淆且缺乏多模态输入。

---

## 54. Fundamental Limits of Quantum Semantic Communication via Sheaf Cohomology

**arXiv ID:** 2601.10958 | [PDF](https://arxiv.org/pdf/2601.10958v1)

**作者:** Christo Kurisummoottil Thomas `[一作]` (Worcester Polytechnic Institute), Mingzhe Chen `[通讯]` (University of Miami)

**通讯引用:** 17440 | [OpenAlex ID](https://openalex.org/A5072241033)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了基于层同调的量子语义通信框架，并证明了语义对齐的最小通信率等于第一层同调群的维数。

**💡 创新点**

创新点在于将量子情境性和纠缠资源映射为层同调障碍，从而给出语义对齐的量化上限，并揭示纠缠与语义信息的等价性。

**🔧 技术方法**

主要技术包括量子希尔伯特空间层同调、完全正映射、超密编码以及纠缠协助的通道容量推导。

**📊 数据集**

论文未使用具体实验数据集，全部以理论模型和数学证明为依据。

**📈 对比分析**

由于是理论推导，没有经验比较；作者通过构造通道容量和层同调维数的等式，说明在给定噪声下量子方案可优于经典。

**⚠️ 局限性**

主要局限在于假设预共享纠缠、无生成成本，且未考虑噪声下的量子操作误差与实现可行性。

---

## 55. IMU-based Real-Time Crutch Gait Phase and Step Detections in Lower-Limb Exoskeletons

**arXiv ID:** 2601.10832 | [PDF](https://arxiv.org/pdf/2601.10832v1)

**作者:** Anis R. Shakkour `[一作]` (Tel Aviv University), Avishai Sintov `[通讯]` (Tel Aviv University)

**通讯引用:** 647 | [OpenAlex ID](https://openalex.org/A5030565378)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在单个扶手 IMU 上实现步态阶段与步数检测，支持下肢外骨骼/假肢同步

**💡 创新点**

仅用单一低成本 IMU 并结合有限状态机提高准确性，消除力传感硬件与推迟控制

**🔧 技术方法**

Temporal Convolutional Network（TCN）+ FSM，且对比 LSTM 与 Transformer

**📊 数据集**

训练集来自四名健康成年人扶手步行，测试集包括一名健康者与一名完全下肢瘫痪的外骨骼用户

**📈 对比分析**

对 TCN、LSTM、Transformer 进行精度与推理时延对比，TCN 在 PC 与 Jetson AGX Orin 上均以 94% 成功率和最低延迟表现最佳

**⚠️ 局限性**

步态启动识别不足、辅助阶段与行走阶段区分不充分，需要更多启动样本和多样化辅助数据

---

## 56. Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse

**arXiv ID:** 2601.11042 | [PDF](https://arxiv.org/pdf/2601.11042v1)

**作者:** Chi Zhang `[一作]` (Shandong University), Zhumin Chen `[通讯]` (Shandong University)

**通讯引用:** 4932 | [OpenAlex ID](https://openalex.org/A5050947285)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于谱分析的序列知识编辑稳定化方法，保护预训练权重的主奇异子空间，避免模型通用能力崩溃。

**💡 创新点**

首次揭示序列编辑导致模型崩溃的根本原因是主奇异方向被破坏，并设计了 plug‑and‑play 框架 REVIVE，利用能量阈值过滤对主奇异方向的更新干扰，从而实现长周期编辑的稳定性。

**🔧 技术方法**

使用奇异值分解（SVD）对参数更新进行谱分解，构建 Dominant Subspace Protection（DSP）策略；通过能量阈值识别主子空间，并在更新时仅保留低能量方向的改动。

**📊 数据集**

在 CounterFact 与 ZsRE 两大事实编辑基准上进行实验，并用 GLUE 评估通用能力，实验覆盖 10,000 次及 20,000 次连续编辑。

**📈 对比分析**

与 MEMIT、PRUNE、RECT、AlphaEdit、NSE 等现有方法对比，REVIVE 在编辑成功率、语义一致性及通用性能上均实现显著提升，尤其在 10k/20k 编辑后仍保持约 80% 以上 GLUE 分数。

**⚠️ 局限性**

仅针对 FFN 层进行谱分析与保护，子空间划分依赖能量阈值且可能非最优；未扩展到注意力等其他模块，且在极端编辑情形下的鲁棒性仍需进一步验证。

---

## 57. Efficient LLR-Domain Decoding of ABS+ Polar Codes

**arXiv ID:** 2601.10808 | [PDF](https://arxiv.org/pdf/2601.10808v1)

**作者:** Mikhail Chernikov `[一作]`, Peter Trifonov `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了基于 LLR 的 ABS+ 极化码 SCL 解码器，实现了低复杂度硬件友好的实现

**💡 创新点**

首次将 ABS+ 极化码的解码过程迁移到 LLR 域，去除了冗余运算并利用中间值复用，显著降低算术运算量

**🔧 技术方法**

采用 LLR 近似、min‑sum 近似、递归解码、路径度量更新、以及对中间 LLR 的共享与复用等技术

**📊 数据集**

使用 AWGN 信道、BPSK 调制的 (1024,512) 极化码作为实验数据集

**📈 对比分析**

通过与传统 Arikan 极化码在 SC 与 CRC‑辅助 SCL 解码下的 FER 对比，ABS+ LLR 解码器在 2–2.5 dB Eb/N0 范围内实现 0.25 dB 的 FER 提升，并在相同算术操作数下显著降低所需列表大小

**⚠️ 局限性**

对概率域解码器的近似导致微小性能损失（≤0.05 dB），并且未对极限性能和更大码长的稳健性作进一步验证

---

## 58. Cooperative UAVs for Remote Data Collection under Limited Communications: An Asynchronous Multiagent Learning Framework

**arXiv ID:** 2601.10849 | [PDF](https://arxiv.org/pdf/2601.10849v1)

**作者:** Cuong Le `[一作]` (National University of Singapore), Thang X. Vu `[通讯]` (University of Luxembourg)

**通讯引用:** 3677 | [OpenAlex ID](https://openalex.org/A5032104368)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出异步多智能体强化学习框架 AQMIX，用于在有限通信条件下的无人机协同数据采集问题，解决能量效率与任务完成时间优化；

**💡 创新点**

创新点在于将轨迹规划问题建模为 Dec-POSMDP，设计可在异步决策环境下保持蒙提的 QMIX 结构，并引入状态下采样降低维度；

**🔧 技术方法**

技术包括 Dec-POSMDP 建模、异步 QMIX 算法、混合网络与超网络、GRU 策略网络、凸优化求解带宽分配；

**📊 数据集**

数据集为仿真生成的 UAV 监控区域网格（8×8、10×10、15×15、20×20）和 Poisson 分布的传感节点与数据需求；

**📈 对比分析**

与独立学习、同步 QMIX 以及启发式划分算法对比，AQMIX 在能量效率、任务完成时间和收集数据比例上均优于基线，表现出更高的鲁棒性和更快的收敛；

**⚠️ 局限性**

局限在于仍依赖仿真，未解决实际 UAV 通信协议设计与实时干扰建模，且在高 UAV 数量或极端通信范围下性能下降。

---

## 59. Adaptive Privacy Budgeting

**arXiv ID:** 2601.10866 | [PDF](https://arxiv.org/pdf/2601.10866v1)

**作者:** Yuting Liang `[一作]` (University of Toronto), Ke Yi `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5846 | [OpenAlex ID](https://openalex.org/A5009196125)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套在广义差分隐私（geo‑privacy）框架下的自适应隐私预算与隐私节省方法，构建了可在多查询、多用户、多维数据上动态分配预算的滤波器与迭代剔除算法；

**💡 创新点**

创新点在于将隐私滤波器与停止时间理论结合，得到对任意度量空间下的可预期预算保证；同时通过迭代剔除将不重要数据的隐私成本降至零，实现隐私节省；并将该框架统一应用于范围计数、核密度估计与最近邻查询；

**🔧 技术方法**

主要技术包括：geo‑privacy 的隐私过滤器与连续化（CGP）过滤器；适应性隐私计数的停止时间与马尔可夫性质；对异构数据组件的统一度量与一致性计数；以及基于 1‑Lipschitz 函数的非交互/交互迭代剔除模板；

**📊 数据集**

实验数据集：MNIST 手写数字集、纽约机动车碰撞数据集（NYMVC）和北京 T‑Drive 轨迹数据集；

**📈 对比分析**

与传统基线方法（BM）对比，PM（带隐私节省）在单查询时保持相近的误差水平，但显著保留更多预算；在多查询场景下，PM 方法利用累积的隐私节省显著提升计数、核密度和最近邻的精度，实验显示误差下降可达 20%–30% 以上；

**⚠️ 局限性**

局限性包括：需预先设定 1‑Lipschitz 重要性函数且其评估误差受噪声控制；迭代剔除在样本极少或噪声过大时可能失效；预算划分策略与参数（c、ν_low/ν_high）对性能影响大，需要手动调优；以及在高维空间中噪声放大导致的偏差增大。

---

## 60. Self-learned representation-guided latent diffusion model for breast cancer classification in deep ultraviolet whole surface images

**arXiv ID:** 2601.10917 | [PDF](https://arxiv.org/pdf/2601.10917v1)

**作者:** Pouya Afshin `[一作]` (Georgia State University), Dong Hye Ye `[通讯]` (Georgia State University)

**通讯引用:** 7284 | [OpenAlex ID](https://openalex.org/A5068927047)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于自监督学习嵌入引导的潜在扩散模型，生成深紫外荧光扫描显微镜图像的高质量合成补丁，用于乳腺保留手术中的病理切缘评估。

**💡 创新点**

创新点在于将Fine‑tuned DINO的SSL嵌入作为条件，引导LDM产生细胞级别的结构化图像，显著提升合成样本的真实性和诊断相关性。

**🔧 技术方法**

使用了DINO自监督学习、潜在扩散模型（LDM）以及Vision Transformer（ViT）进行补丁级分类与WSI级聚合。

**📊 数据集**

使用了来自密歇根医学院的142份深紫外全切面图像（58良性、84恶性），共提取172,984块400×400补丁。

**📈 对比分析**

与基于类别条件的DDPM、LDM以及无合成基线进行5折交叉验证比较；SSL引导LDM在WSI级别实现了96.47%准确率、96.46%灵敏度和96.36%特异性，优于其他方法。

**⚠️ 局限性**

局限包括合成数据仅占总量5%，对不同光照或样本多样性的鲁棒性未充分验证，以及潜在空间训练对计算资源需求较高。

---

## 61. HOSL: Hybrid-Order Split Learning for Memory-Constrained Edge Training

**arXiv ID:** 2601.10940 | [PDF](https://arxiv.org/pdf/2601.10940v1)

**作者:** Aakriti `[一作]` (Rochester Institute of Technology), Haibo Yang `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1527 | [OpenAlex ID](https://openalex.org/A5013868893)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种混合阶分割学习框架（Hybrid-Order Split Learning），将客户端采用零阶优化、服务器采用一阶优化，实现边缘设备的低内存高效训练。

**💡 创新点**

创新点在于：① 通过在客户端使用零阶梯度估计消除反向传播和激活存储，显著降低内存占用；② 在服务器使用一阶SGD保持快速收敛，二者结合在理论和实验上实现了收敛速率仅依赖客户端参数维度 d_c；③ 证明了在非凸目标下该混合策略收敛速率为 𝒪(√(d_c/TQ))。

**🔧 技术方法**

技术手段包括：分割模型到客户端/服务器；客户端使用MeZO式零阶梯度估计（多点估计 Q）；服务器使用标准一阶SGD；在通信中仅交换激活和标量损失；理论分析基于 L‑smooth、梯度无偏与方差界限。

**📊 数据集**

使用 OPT 系列语言模型（125M 与 1.3B）进行全参数和 LoRA 细调；在 GLUE 与 SuperGLUE 的六个任务（SST‑2、CB、WIC、WSC、BoolQ、RTE）上评估。

**📈 对比分析**

与两种基线对比：FO‑FO（全一阶）和 ZO‑ZO（全零阶）。实验显示：在保持 1.38–3.7 倍的客户端 GPU 内存减少（相较 FO‑FO）同时准确率仅比 FO‑FO 低 0.41%–4.23%，并比 ZO‑ZO 提升 0.69%–15.55% 的准确率，证明了兼顾性能与内存的优势。

**⚠️ 局限性**

主要局限是：客户端需要多次前向传播（2Q 次）导致训练时间增长；实验仅针对单一客户端，未考虑多客户端或数据异构的联邦学习场景。

---

## 62. CTHA: Constrained Temporal Hierarchical Architecture for Stable Multi-Agent LLM Systems

**arXiv ID:** 2601.10738 | [PDF](https://arxiv.org/pdf/2601.10738v1)

**作者:** Percy Jardine `[一作]` `[通讯]`, Percy Jardine

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Constrained Temporal Hierarchical Architecture（CTHA），在多时序层级代理中引入消息合约、权限多样性约束和仲裁器以恢复层间协调稳定性。

**💡 创新点**

创新点在于将层间通信投影到结构化流形上，形成三种约束机制（消息合约、权限多样性约束、仲裁器），从而实现安全、可解释且高效的多层决策。

**🔧 技术方法**

使用了大型语言模型（DeepSeek‑V3.2‑Speciale、Kimi‑2、Qwen‑3‑32B、GLM‑4.6‑9B）作为不同时间尺度的子代理；引入了分层映射（pre/post/residual）、约束投影函数、结构化 JSON schema 验证、权威多面体投影以及轻量级 Transformer 生成的仲裁器。

**📊 数据集**

在九个基准上进行评估，包括 ToolBench、WebArena、SWE‑Bench、τ²‑Bench、AgentBench、ALFWorld、HotpotQA、GAIA 与 SafetyBench 等多任务和安全测试集。

**📈 对比分析**

与单尺度代理、多代理系统及无约束层级架构比较，CTHA 在大多数任务上获得 5–7% 的平均提升，失败级联率下降 78% 以上，安全攻击成功率降低 76%，同时保持 1.12× 的计算成本。

**⚠️ 局限性**

局限性包括固定的四层层级结构、仲裁器训练数据分布局限、对极简任务的额外开销以及上下文窗口长度限制，未来可通过自适应层级学习、不同流形约束及多代理扩展进一步提升。

---

## 63. Asymmetric Encoding-Decoding Schemes for Lossless Data Compression

**arXiv ID:** 2601.10991 | [PDF](https://arxiv.org/pdf/2601.10991v1)

**作者:** Hirosuke Yamamoto `[一作]`, Ken-ichi Iwata `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

提出了一种新的无失真数据压缩编码方案——非对称编码-解码方案（AEDS），并探讨其在不同状态数下对码长和冗余的影响。

**💡 创新点**

创新点在于：①将tANS的后向编码/前向解码结构推广为更一般的AEDS；②证明在特定条件下（根节点子树权重>0.61803或>0.56984）仅用2或5个状态即可比Huffman码取得更短的平均码长；③给出AEDS（包括其子类sAEDS）的平均码长上界，并证明最优sAEDS与tANS在状态数N趋大时平均码长与源熵的误差为O(1/N)。

**🔧 技术方法**

使用的技术包括：编码/解码状态转移函数、前缀码、码树（如Huffman码树和分段码树）、信息理论分析（熵、相对熵、冗余上界）、数学证明与定理推导。

**📊 数据集**

本文为理论研究，无使用具体数据集；所有结论基于i.i.d. 源概率分布与码树结构推导。

**📈 对比分析**

比较方法：将AEDS的平均码长和冗余与Huffman码、tANS以及理论最优码长（熵）进行对比。结果显示：在满足子树权重大于阈值时，AEDS可在仅2或5个状态下实现比Huffman码更短的码长；整体上最优sAEDS与tANS的平均码长与熵差距随状态数N以O(1/N)收敛，远优于传统tANS在有限N下的表现。

**⚠️ 局限性**

限制与不足：①对给定N的AEDS不一定是最优的；②设计与实现相对复杂，尤其是状态转移函数与码集的选取；③对极端分布或状态数较少时的性能提升有限；④本文主要讨论理论上限，实际实现中的细节（如硬件/软件优化、实时性）未作深入探讨。

---

## 64. Change And Cover: Last-Mile, Pull Request-Based Regression Test Augmentation

**arXiv ID:** 2601.10942 | [PDF](https://arxiv.org/pdf/2601.10942v1)

**作者:** Zitong Zhou `[一作]` (University of California Los Angeles), Michael Pradel `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型的 PR 级回归测试增量方法，自动补全 PR 中未被现有测试覆盖的代码行。

**💡 创新点**

创新点在于：①将 PR 上下文、动态测试上下文和迭代反馈融入 LLM 生成测试；②自动将生成测试与项目已有测试套件无缝对齐；③通过覆盖分析、PR 文本抽取和 AST 合并实现精准集成。

**🔧 技术方法**

采用 GPT‑4o‑mini 等大型语言模型，结合代码覆盖分析、PR 内容抽取、动态调用图、LLM 生成与迭代反馈以及 AST 级别的合并技术。

**📊 数据集**

在 SciPy、Qiskit、Pandas 三个大规模 Python 开源项目共 145 条 PR 上进行实验，并针对 30 条 PR 做进一步定性评估与提交验证。

**📈 对比分析**

与无上下文、无反馈、仅 LLM 上下文等对照组比较，平均覆盖提升 13 行，测试通过率提升 25%，30% 的 PR 实现 100% 补丁覆盖，单 PR 平均成本约 0.12 美元。

**⚠️ 局限性**

局限性包括：仅支持 Python 项目；对项目构建与环境依赖敏感；LLM 可能生成不符合细节的测试；缺乏对 Bug PR 的主动检测；对非公开或未收录于训练集的项目泛化能力未知。

---

## 65. Massively Multilingual Joint Segmentation and Glossing

**arXiv ID:** 2601.10925 | [PDF](https://arxiv.org/pdf/2601.10925v1)

**作者:** Michael Ginn `[一作]` (University of Colorado Boulder), Alexis Palmer `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1192 | [OpenAlex ID](https://openalex.org/A5069931383)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了PolyGloss模型，实现一次推理同时生成形态分割和词间注释。

**💡 创新点**

提出联合训练和交织输出格式以确保分割与注释对齐，并证明按词错误率和对齐分数提升显著。

**🔧 技术方法**

采用ByT5字节级序列到序列模型，结合低秩适配和RL调优等技术。

**📊 数据集**

使用改进后的GlossLM语料库，新增91k条实例并加入Fieldwork、IMTVault等多源IGT数据，覆盖2077种语言。

**📈 对比分析**

与GlossLM、LLM ICL和多种单语基线对比，PolyGloss在词间注释MER低至0.234、分割F1达0.862、对齐分数1.000，优于对齐要求的现有模型。

**⚠️ 局限性**

模型未对词形级别进行细化，未标准化注释规范，且在极少语料或高方言差异时表现有限。

---

## 66. Optimisation of complex product innovation processes based on trend models with three-valued logic

**arXiv ID:** 2601.10768 | [PDF](https://arxiv.org/pdf/2601.10768v1)

**作者:** Nina Bočková `[一作]` (Prague University of Economics and Business), Mirko Dohnal `[通讯]` (Brno University of Technology)

**通讯引用:** 1232 | [OpenAlex ID](https://openalex.org/A5085999321)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出基于三值趋势（增加、减少、保持）的趋势模型，用来表征和优化复杂产品创新过程，并通过一个跨国企业子公司知识转移的案例进行验证。

**💡 创新点**

创新点在于：①用极简的三值趋势量化方式替代传统数值、模糊或统计方法，降低信息需求；②引入趋势一致性与约束关系剔除机制，将模型约束转化为可优化的删减关系；③利用转移图完整描绘所有可能的未来/历史情景与转移路径。

**🔧 技术方法**

技术方法包括：三值逻辑趋势模型、相关矩阵转趋势规则、主观启发式知识编码、组合优化（最小删减关系/最小相关系数和）、以及基于情景集构建的有向转移图。

**📊 数据集**

案例使用了跨国企业子公司管理与知识转移相关的七个变量（PI、SD、EE、SS、SA、SE、KS）所构成的主观/相关数据集，未公开具体数据来源。

**📈 对比分析**

与传统统计/模糊集方法对比，趋势模型不需要数值参数，可直接获得所有可能的情景与转移，理论上覆盖更完整；但本文未给出定量性能指标，仅通过案例说明模型能产生可解释的终端情景。

**⚠️ 局限性**

局限性包括：①模型高度依赖主观启发式和相关矩阵的一致性，若数据噪声大或相关性不稳定，模型可能无解；②当变量和关系数目增大时，组合优化求解复杂度高；③缺乏在真实大规模数据集上的实证验证；④未考虑模型随时间更新或不确定性动态调整的问题。

---

## 67. PruneRAG: Confidence-Guided Query Decomposition Trees for Efficient Retrieval-Augmented Generation

**arXiv ID:** 2601.11024 | [PDF](https://arxiv.org/pdf/2601.11024v1)

**作者:** Shuguang Jiao `[一作]` (Harbin Institute of Technology), Lina Yao `[通讯]` (UNSW)

**通讯引用:** 15749 | [OpenAlex ID](https://openalex.org/A5052731721)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于置信度引导的查询分解树（PruneRAG），用于在检索增强生成（RAG）中减少证据遗忘并提升推理效率。

**💡 创新点**

创新点包括：①自适应节点扩展策略，可根据当前查询和检索上下文动态控制树宽深；②置信度引导剪枝机制，用生成答案的置信分数决定是否接受答案或继续拆分；③细粒度检索机制，在无法进一步拆分时提取实体级别锚点进行精准检索；④引入证据遗忘率（EFR）指标，用以量化检索后证据未被有效利用的情况。

**🔧 技术方法**

技术核心包括：多层次查询分解模块、基于token概率的置信度计算、树结构的前向扩展与后向回溯聚合、以及与外部检索器（FAISS + BGE-large-en-v1.5）结合的检索策略。

**📊 数据集**

实验使用的多跳问答数据集包括 HotpotQA、2WikiMultihopQA、Musique、Bamboogle、GPQA；单跳数据集有 Natural Questions 与 TriviaQA，用以评估方法的泛化能力。

**📈 对比分析**

与链式RAG（React、Search-o1、Self-RAG、MemoRAG）和树式RAG（ConTReGen、RAG-Star、ProbTree）等基线相比，PruneRAG 在 EM/F1 上平均提升 5–6% ，证据遗忘率平均下降 20% 以上，并且推理时延下降约 4.9 倍，检索次数更少，整体效率显著提高。

**⚠️ 局限性**

局限性主要包括：①对置信度阈值和树深度的超参数仍需经验调优；②在极大规模知识库或多模态场景下，细粒度检索可能引入检索成本；③若生成模型本身的语义推理能力不足，置信度评估可能失真，导致错误剪枝。

---

## 68. High-Order Lie Derivatives from Taylor Series in the ADTAYL Package

**arXiv ID:** 2601.10828 | [PDF](https://arxiv.org/pdf/2601.10828v1)

**作者:** Nedialko S. Nedialkov `[一作]` (McMaster University), John D. Pryce `[通讯]` (Cardiff University)

**通讯引用:** 2443 | [OpenAlex ID](https://openalex.org/A5064076305)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文利用Taylor级数数值方法高效计算非线性系统的高阶李导数，避免了符号方法导致的表达式膨胀；

**💡 创新点**

创新点在于将李导数与Taylor系数对齐，只需两行MATLAB代码即可求解标量、向量、余向量场的高阶李导数；

**🔧 技术方法**

核心技术是基于自适应Taylor级数自动微分的ADOL-C实现，并利用Jacobian逆/前向变换求取导数；

**📊 数据集**

采用抓取式滑轮系统（gantry crane）作为实验数据集，包含状态变量z、ϕ、ẋ、ϕ̇；

**📈 对比分析**

与MATLAB Symbolic Math Toolbox和ADOL-C比较，本文方法在10阶时速度提升约1000倍，符号表达式构造时间显著下降；

**⚠️ 局限性**

局限性包括对较大阶数（k>50）时的平方复杂度，以及对高维系统时的数值误差可能随阶数迅速放大。

---

## 69. Balanced allocation: considerations from large scale service environments

**arXiv ID:** 2601.10874 | [PDF](https://arxiv.org/pdf/2601.10874v1)

**作者:** Amer Diwan `[一作]` (Google), Eli Upfal `[通讯]` (Brown University)

**通讯引用:** 15506 | [OpenAlex ID](https://openalex.org/A5028869858)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e`

**🎯 论文内容**

本文研究了在大规模云服务中使用 d‑way 平衡分配（Power of d Choices）处理突发请求、不同优先级任务以及噪声信息时的表现。

**💡 创新点**

创新点在于在动态负载、突发、高优先级以及信息不准确的多种实际场景下，证明了双指数衰减仍然成立，并给出了恢复时间、优先级分布与噪声误差的严谨理论分析。

**🔧 技术方法**

主要技术手段包括离散事件仿真、差分方程与流式极限分析、马尔科夫链稳定性与Lyapunov漂移、以及贝塔–恩斯理论推导。

**📊 数据集**

实验数据来自在 1000 个队列、λ=0.95 的仿真模型，进行 30 次独立运行，并在不同 d、突发次数、优先级策略和信息噪声参数下收集平均/最大队列深度。

**📈 对比分析**

与随机（d=1）对比，通过平均/最大队列深度、恢复时间和优先级延迟等指标评估，发现 d=2、3、4 能显著降低队列深度、加快恢复，并在优先级分配上保持较低的高优先级延迟。

**⚠️ 局限性**

限制主要在于模型假设（Poisson 到达、指数服务、单层负载平衡）以及仅在可解析的参数范围内证明，未覆盖多层分布式调度、网络时延、以及真实云平台的测量验证。

---

## 70. Energy-Efficient Omnidirectional Locomotion for Wheeled Quadrupeds via Predictive Energy-Aware Nominal Gait Selection

**arXiv ID:** 2601.10723 | [PDF](https://arxiv.org/pdf/2601.10723v1)

**作者:** Xu Yang `[一作]` (Tsinghua University), Yilin Mo `[通讯]` (Tsinghua University)

**通讯引用:** 8564 | [OpenAlex ID](https://openalex.org/A5018443722)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一套分层控制框架，利用能量预测网络挑选最省能量的本体步态，再用残差强化学习（RL）对本体步态进行微调，以实现轮式四足机器人在各向全向运动中的能量优化与速度跟踪。

**💡 创新点**

创新点主要包括：① 设计了可预测1 s能耗的功率预测网络，可实时评估不同步态的能耗；② 构建了包含驱动、行走与疾跑三种步态的扩展步态库，并将轮速嵌入本体步态；③ 将预测的步态选择与残差RL策略组合，实现了既能保留本体步态的结构化约束，又能通过RL学习到细致的补偿动作；④ 采用课程学习方式平衡探索与能耗最优的训练过程。

**🔧 技术方法**

技术手段包括：深度神经网络（预测器、估计器和RL策略）；分层控制架构；使用IsaacGym并行仿真训练；基于IMU和编码器的观测；使用PD和PID控制实现关节与轮速跟踪；在Jetson Xavier NX上实现实时部署；能源估计采用关节力矩×速度。

**📊 数据集**

数据集：主要在IsaacGym中生成的 4096 机器人并行仿真数据，覆盖多种速度指令和随机扰动；随后在实际 Unitree Go1 机器人上收集少量真实传感器数据验证模型；未使用公开数据集，全部为作者自行生成或收集。

**📈 对比分析**

比较方法包括基线（无步态或预测）、PMTG（固定疾跑步态+RL控制）、无轮速预测、无步态切换等 ablation。实验显示：相较于基线和PMTG，能耗下降约30‑35%；在速度跟踪任务中误差均低于对照组；在随机推力扰动下成功恢复率至少高出10‑20%，并在最大扰动 0.9 m/s 下恢复率仍超过 90%。

**⚠️ 局限性**

局限性：训练依赖大量并行仿真，计算成本高；能量预测网络在极端地形或滑移情况时可能失效；实验多聚焦平坦或轻微不平坦地面，未验证在复杂崎岖、湿滑环境下的鲁棒性；真实部署仍需进一步调优对模型误差和硬件延迟的容忍度。

---

## 71. Line-based Event Preprocessing: Towards Low-Energy Neuromorphic Computer Vision

**arXiv ID:** 2601.10742 | [PDF](https://arxiv.org/pdf/2601.10742v1)

**作者:** Amélie Gruel `[一作]` (University of Bordeaux), Sylvain Saïghi `[通讯]` (University of Bordeaux)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对事件相机产生的事件数据进行基于线检测的预处理，使得随后脉冲神经网络（SNN）分类器所需的突触操作量显著下降，同时保持甚至提升分类准确率。

**💡 创新点**

首次提出端到端的线基事件预处理机制，将事件数据转化为可直接输入SNN的线特征，显著降低理论能耗并在多种策略下实现高效能与较高准确率的双赢。

**🔧 技术方法**

使用脉冲神经网络（LIF模型、Winner‑Takes‑All抑制）、线检测的稀疏突触模式、NEST+PyNN仿真、以及与卷积层和全连接隐藏层的对比实验。

**📊 数据集**

实验所用数据集包括 PokerDVS、N‑MNIST 以及 DVS128 Gesture 三个事件视觉基准集。

**📈 对比分析**

通过准确率、突触事件数、理论效率（准确率/突触事件）三指标进行比较；在最佳效率参数化下，线预处理大部分策略可将效率提升 2–7 倍，且多数策略保持或提升准确率；在 DVS128 Gesture 上的累积策略同时提升准确率与效率。

**⚠️ 局限性**

局限性包括：仅使用简单的全连接分类器，缺乏对更复杂任务的验证；预处理机制尚未在真实硬件上实现；精度仍低于现有最先进方法；需要针对不同数据集调参；对极端动态场景的鲁棒性尚未评估。

---

## 72. AdaMARP: An Adaptive Multi-Agent Interaction Framework for General Immersive Role-Playing

**arXiv ID:** 2601.11007 | [PDF](https://arxiv.org/pdf/2601.11007v1)

**作者:** Zhenhua Xu `[一作]` (Zhejiang University), Yabiao Wang `[通讯]` (Tencent)

**通讯引用:** 8280 | [OpenAlex ID](https://openalex.org/A5028731909)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AdaMARP框架，结合环境感知的消息格式和离散动作的场景管理器，实现多角色动态配音、场景切换与角色即刻加入的角色扮演系统；

**💡 创新点**

创新点在于：①将思考、动作、环境与台词四要素交错嵌入消息中，实现更沉浸式叙事；②设计可控的场景管理器（动作空间{init, add, change, next, finish}），实现多角色协作与情节推进；③构建AdaRPSet与AdaSMSet两大数据集并推出AdaptiveBench轨迹级评测。

**🔧 技术方法**

采用监督式微调技术：对Actor模型和Scene Manager分别使用AdaRPSet和AdaSMSet进行全参数微调，并通过结构化提示与离散动作控制实现交互；

**📊 数据集**

使用AdaRPSet（提取自文学文本与LLM合成的20主题剧情）与AdaSMSet（基于合成轨迹的动作与理由标注），以及在AdaptiveBench上生成的模拟对话轨迹进行评估；

**📈 对比分析**

与多款商业与开源LLM（GPT‑4o‑mini、GPT‑5‑Chat、Claude Sonnet 4.5、Qwen3‑14B等）以及现有通用角色扮演基线（BeyondDialogue、Crab、CoSER）进行对比；实验显示8B Actor模型在AdaptiveBench上平均分约8.7/10，超过多款大型商用模型；14B Scene Manager平均分约8.4/10，已逼近或超越Claude Sonnet 4.5；

**⚠️ 局限性**

局限性在于：1）数据集依赖人工与LLM生成的场景与角色，可能存在偏差；2）评测仍停留在模拟轨迹层面，缺乏真实用户交互验证；3）对极端动态情节的鲁棒性尚待进一步验证。

---

## 73. Tail-Aware Data Augmentation for Long-Tail Sequential Recommendation

**arXiv ID:** 2601.10933 | [PDF](https://arxiv.org/pdf/2601.10933v1)

**作者:** Yizhou Dang `[一作]` (Northeastern University), Xingwei Wang `[通讯]` (Northeastern University)

**通讯引用:** 9913 | [OpenAlex ID](https://openalex.org/A5100326915)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种针对长尾序列推荐的 Tail‑Aware 数据增强方法 TADA，能够在不损失头部性能的前提下显著提升尾部推荐效果。

**💡 创新点**

创新点包括设计两种面向长尾的增强操作 T‑Substitute 与 T‑Insert，结合基于线性模型的关联与共现候选集，并通过跨序列混合产生多样化增强样本。

**🔧 技术方法**

主要技术为线性协同矩阵学习候选集、贝塔分布混合策略、表示层级混合以及多层级的序列增强与交叉增强。

**📊 数据集**

在 Toys、Beauty 与 Sports 三个 5‑core 数据集上进行实验验证。

**📈 对比分析**

与多种基线（CITIES、LOAM、MELT、CMR、RepPad、SASRec 等）对比，TADA 在尾部指标提升显著，整体/头部性能保持或提升，平均提升幅度在 30%–45% 之间。

**⚠️ 局限性**

缺点是对预先构建的候选集和超参数敏感，且未考虑多模态信息，仍需在更复杂场景中进一步验证。

---

## 74. Reasoning Models Generate Societies of Thought

**arXiv ID:** 2601.10825 | [PDF](https://arxiv.org/pdf/2601.10825v1)

**作者:** Junsol Kim `[一作]` (Google), James Evans `[通讯]` (University of Chicago)

**通讯引用:** 15197 | [OpenAlex ID](https://openalex.org/A5071261828)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 DeepSeek‑R1、QwQ‑32B 等高性能推理模型的生成过程进行细致分析，揭示其推理轨迹内部存在类似“思维社会”的多方位对话结构，并证明这种内部社会化对话是提高推理准确率的关键机制；同时利用可解释性方法（稀疏自编码器、特征驱动的激活干预）证明了对话相关特征的激活能显著提升推理表现，并在强化学习实验中显示对话式训练能加速模型学习。

**💡 创新点**

创新点包括：①首次将“社会化思维”框架引入 LLM 推理解释，发现推理模型通过内部多视角对话实现自我验证与纠错；②利用 LLM‑as‑judge 识别对话行为、情感角色与隐含视角，并量化其对推理准确率的直接与间接作用；③通过稀疏自编码器特征驱动的对话激活，系统性地证明对话特征可被刻意增强并提升推理表现；④在强化学习实验中验证仅奖励准确率即可自发出现对话行为，并证明对话式预训练能显著加速学习。

**🔧 技术方法**

技术手段：①基于 Gemini‑2.5‑Pro 的 LLM‑as‑judge 对话与情感角色、认知行为、隐含视角进行自动编码；②稀疏自编码器（32,768 维）在 DeepSeek‑R1‑Llama‑8B 的第 15 层残差流上训练，用于识别与对话相关的特征并进行激活添加；③强化学习采用 PPO（Verl 框架）和自定义奖励（准确率×0.9 + 格式×0.1）；④对隐含视角进行人格（BFI‑10）与专业领域特征的嵌入与多样性计算；⑤使用 UMAP、Shannon 熵等指标评估特征多样性。

**📊 数据集**

数据集：8,262 个跨领域推理任务（BigBench Hard、GPQA、MATH、MMLU‑Pro、IFEval、MUSR）；Countdown 算术推理 1,024 题；政治误信息检测 23,299 条事实核查标题；此外用于训练稀疏自编码器的 SlimPajama‑3B 语料；使用 Intelligence Squared Debates Corpus 评估 LLM‑as‑judge 对话识别的准确性。

**📈 对比分析**

比较方法：在相同问题集上将推理模型与其基线指令调优版本（DeepSeek‑V3、Qwen‑2.5‑32B‑IT 等）以及 GPT‑4/Claude 等标准模型做对比；在强化学习实验中比较基线、对话式预训练和单语者预训练三种策略；评估指标包括：推理准确率、对话行为频率、人格与专业多样性分数、特征覆盖率与熵、RL 训练步数内的准确率提升速度。实验结果表明：①推理模型在对话行为和情感角色上显著高于基线；②对话激活特征 30939 的正向驱动可使 Countdown 准确率从 27% 提升至 55%；③对话式预训练模型在 RL 训练中比单语者预训练快 1.5–2 倍、最终准确率更高。

**⚠️ 局限性**

局限性：①实验主要基于 3B–70B 规模模型，尚未验证在更大规模模型上的可迁移性；②对话行为和人格多样性的评估高度依赖 LLM‑as‑judge，可能受生成质量影响；③稀疏自编码器的特征解释仍属于黑箱，缺乏可解释性阈值的严格定义；④强化学习实验仅在简单算术任务和误信息检测两类任务上验证，难以确认对话机制在更复杂情境中的普适性；⑤对话特征激活可能在实际应用中引入生成不连贯或语义漂移的风险。

---

## 75. Too Helpful to Be Safe: User-Mediated Attacks on Planning and Web-Use Agents

**arXiv ID:** 2601.10758 | [PDF](https://arxiv.org/pdf/2601.10758v1)

**作者:** Fengchao Chen `[一作]` (Monash University), Carsten Rudolph `[通讯]` (Monash University)

**通讯引用:** 1393 | [OpenAlex ID](https://openalex.org/A5086593836)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了用户中介攻击场景，系统评估了12款商业LLM代理在无安全请求、软安全请求和硬安全请求下对未验证内容的处理和执行行为。

**💡 创新点**

创新点在于提出用户中介攻击框架，将攻击者通过诱导用户转发恶意内容映射为代理输入，并揭示代理安全优先级缺失。

**🔧 技术方法**

使用了黑盒实验、沙盒环境、URL验证与任务执行等测试手段，量化了绕过率并评估代理安全机制的触发条件。

**📊 数据集**

构造了合成URL变体、促销广告以及公开恶意URL和旅行预订链接等测试用例，作为实验数据。

**📈 对比分析**

通过对比12款代理在三种安全请求情景下的环境约束和常识约束绕过率，结果显示无请求时绕过率>92%，软请求降至54.7%，硬请求降至7%，验证了安全检查被动激活的问题。

**⚠️ 局限性**

研究局限在于仅覆盖12款代理，实验周期短且未涵盖后续更新；缺乏用户侧防御措施，实验数据主要为合成，实际环境适用性待进一步验证。

---

## 76. Future Optical Flow Prediction Improves Robot Control & Video Generation

**arXiv ID:** 2601.10781 | [PDF](https://arxiv.org/pdf/2601.10781v1)

**作者:** Kanchana Ranasinghe `[一作]` (Salesforce AI Research), Juan Carlos Niebles `[通讯]` (Salesforce AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种统一的 Vision‑Language‑Diffusion 框架，用语言指令预测稠密未来光流，并将其应用于机器人控制和文本驱动的视频生成；

**💡 创新点**

创新点在于：① 将 VLM 与扩散模型联合，用光流生成作为桥梁实现跨域通用性；② 通过相机运动补偿和光流解耦获得干净的监督信号；③ 在大规模网络人类视频上预训练，显著提升跨域泛化；④ 将预测光流直接映射到控制策略与视频合成两大任务，形成闭环；

**🔧 技术方法**

核心技术包括：Qwen2.5‑VL 视觉‑语言 Transformer、Flux.1 VAE 编码/解码光流、OmniGen‑style DiT 扩散 Transformer（时空注意力），以及相机运动估计（同伦+深度特征）与相对光流计算；用于机器人的是扩散策略网络，视频生成使用 Go‑with‑the‑Flow；

**📊 数据集**

使用的数据集包括 Something‑Something‑V2、EgoDex、DROID、SSv2、CALVIN、RoboTwin 2.0 以及海量带字幕的网络人类活动视频；

**📈 对比分析**

与基线对比：在 CALVIN ABC→D 零样本长序任务中，成功率0.787、平均长度4.48，略优于 DreamVLA；在 RoboTwin 2.0 平均成功率68.6% 超越 VPP 的 61.8%；在 SSv2 视频生成中，FVD、LPIPS 等指标均优于 CogVideoX，展示了更好的运动控制和生成质量；

**⚠️ 局限性**

局限性包括：模型对文本提示高度敏感，轻微改动可导致错误预测；参数量约 7B，推理需高显存（≥24GB），难以实时部署；目前缺乏对细粒度语言差异的鲁棒性；

---

## 77. SonicBench: Dissecting the Physical Perception Bottleneck in Large Audio Language Models

**arXiv ID:** 2601.11039 | [PDF](https://arxiv.org/pdf/2601.11039v1)

**作者:** Yirong Sun `[一作]` (Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative Institute of Digital Twin EIT), Xiaoyu Shen `[通讯]` (Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative Institute of Digital Twin EIT)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SonicBench，一个基于心理物理学的音频物理属性评测基准，覆盖12个音频物理属性，采用绝对判断与相对比较两种任务形式；

**💡 创新点**

创新点在于：①系统性、统一化的物理属性分类与任务设计；②结合可控生成工具箱生成严格对齐的音频刺激；③通过对比识别与比较任务，揭示模型在物理感知与关系推理上的差距；

**🔧 技术方法**

采用可控生成算法、心理物理阈值校准、线性探针评估、对比实验以及多模型对齐与解码分析；

**📊 数据集**

使用自建的2400条问答音频对（每个属性100条识别+100条比较），覆盖Spectral & Amplitude、Temporal、Spatial & Environment、Timbre、Scene‑Level五个维度；

**📈 对比分析**

与人类、12类大型模型（LALM、LARM、OLM）进行零样本对照，结果显示人类平均准确率91%，最佳模型Qwen3‑Omni仅达72%，绝大多数模型接近随机；线性探针表明编码器已捕获物理信息，差距主要来自对齐/解码阶段；

**⚠️ 局限性**

限制包括：探针仅为线性分类器，可能低估编码器潜力；数据集以可验证标签为主，缺乏“野外”多样性；评测仅使用英文提示，未检验多语言鲁棒性。

---

## 78. Approximately Optimal Global Planning for Contact-Rich SE(2) Manipulation on a Graph of Reachable Sets

**arXiv ID:** 2601.10827 | [PDF](https://arxiv.org/pdf/2601.10827v1)

**作者:** Simin Liu `[一作]` (Carnegie Mellon University), Tao Pang `[通讯]` (Robotics and AI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于互相可达集（MRS）的图搜索框架，用于在 SE(2) 空间实现接触丰富的全局操控规划。

**💡 创新点**

创新点在于引入面向对象的离散决策空间——互相可达集，将低层接触模式与高层运动原语折中，显著降低组合复杂度并实现近似最优搜索。

**🔧 技术方法**

技术方法包括：离线构建凸近似 MRS、构造 MRS 图并使用图的凸集（GCS）求最短路径、采用 CQDC‑MPC 进行接触感知轨迹优化、使用 RRT‑Connect 进行碰撞避免规划、拟合 GCS 边/点成本以及多路径采样以提升鲁棒性。

**📊 数据集**

数据集与实验平台：在双臂 KUKA iiwa‑7 系统上测试圆柱物体，生成 250 条随机起止姿态查询，并在硬件上执行开环轨迹，评估离线构图时间、在线查询时间与成功率。

**📈 对比分析**

与现有 ContactRRT 的对比实验表明：任务成本下降 61%，成功率提升至 91%（相比 82%），接触变化比降低 25%，查询时间约 40 秒（低于 62 秒）；离线构图耗时 1.7 小时；消融实验显示路径采样与成本拟合分别提升成功率约 7.5% 与成本约 6%。

**⚠️ 局限性**

局限性：若起止姿态落在图覆盖之外会失败；凸近似与离散化可能引入不可达状态；转移段的可达性假设对极端姿态敏感；重抓阶段易出现数值漂移导致失败；目前仅在 SE(2) 下验证，扩展到 SE(3) 需新方法。

---

## 79. A Concise Agent is Less Expert: Revealing Side Effects of Using Style Features on Conversational Agents

**arXiv ID:** 2601.10809 | [PDF](https://arxiv.org/pdf/2601.10809v1)

**作者:** Young-Min Cho `[一作]` (University of Pennsylvania), Lyle Ungar `[通讯]` (University of Pennsylvania)

**通讯引用:** 30279 | [OpenAlex ID](https://openalex.org/A5044944954)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了对话代理中多风格控制的交叉风格侧效应，系统评估12种常用风格特征对彼此的因果影响，并提出CASSE数据集。

**💡 创新点**

首次系统量化风格控制的交叉侧效应，揭示风格特征深度耦合，提供多目标偏差补偿实验，并公开CASSE。

**🔧 技术方法**

采用Llama3、Qwen3等LLM生成合成对话，使用LLM-as-Judge双人比较评估风格，构造因果矩阵，应用Prompt Intervention与Contrastive Activation Addition（CAA）进行侧效应缓解。

**📊 数据集**

从ACL Anthology 2023-2025筛选127篇对话代理论文获取12种风格特征；使用LMSYS-Chat-1M与DailyDialog生成12,200条合成回复；CASSE包含风格评分。

**📈 对比分析**

通过win‑rate比较风格影响，发现例如简洁导致专业度下降；在缓解实验中Prompt Intervention恢复侧效应，但往往削弱主风格；Steering Intervention也能恢复但更易损失主风格。

**⚠️ 局限性**

仅基于短合成回复，缺乏真实用户交互；未评估大型模型；侧效应缓解仅使用简单方法，未尝试更复杂的多目标优化或微调。

---

## 80. Bidirectional Human-Robot Communication for Physical Human-Robot Interaction

**arXiv ID:** 2601.10796 | [PDF](https://arxiv.org/pdf/2601.10796v1)

**作者:** Junxiang Wang `[一作]` (Carnegie Mellon University), Zackory Erickson `[通讯]` (Carnegie Mellon University)

**通讯引用:** 627 | [OpenAlex ID](https://openalex.org/A5075365855)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了BRIDGE系统，实现了用户通过自然语言实时修改机器人在物理辅助任务中的轨迹（位置、速度、力），并在每次用户发话后给予口头反馈（确认或澄清问题），从而实现双向沟通。

**💡 创新点**

创新点在于：①将大型语言模型（LLM）用于即时解释用户命令并生成轨迹修改指令；②通过简洁的YAML表示实现低延迟输出；③在LLM生成的同时产生口头确认或澄清问题，增强系统透明度和交互性；④在轨迹修改中同时支持全局、局部（关节点）和单点的三种作用域。

**🔧 技术方法**

技术手段包括：大型语言模型GPT‑4.1；Azure 语音识别服务；自定义YAML轨迹与修改格式；高斯衰减模型与势场方法用于局部速度/力和位置调整；Stretch 3移动操作机器人；Python/ROS实现实时规划与控制。

**📊 数据集**

数据来源为18名老年志愿者在实验室完成的三种辅助任务（抓取、喂食、擦洗）的语音指令与机器人轨迹记录；未使用公开数据集，而是通过用户研究收集的交互日志和问卷结果。

**📈 对比分析**

与两种对照方案对比：①单向通信（可修改轨迹但无反馈），②无修改基线（监听但不改变轨迹）。实验结果显示：• 任务成功率与用户控制感与两种可修改方案相当；• 双向反馈方案在互动性、透明度、用户对机器人理解度等六项Likert量表上显著高于单向方案；• 平均响应时间约1.7 s（LLM 1.3 s + 规划 0.4 s），满足实时交互需求。

**⚠️ 局限性**

局限性：仅在单次实验中评估，未验证长期部署下的反馈策略；用户始终静止，无法覆盖动态策略或自由移动；目前仅支持口头交流，未尝试触觉或手势等多模态交互；LLM模型与推理成本对资源受限机器人平台的适配仍需进一步研究。

---

## 81. AFLL: Real-time Load Stabilization for MMO Game Servers Based on Circular Causality Learning

**arXiv ID:** 2601.10998 | [PDF](https://arxiv.org/pdf/2601.10998v1)

**作者:** Shinsuk Kang `[一作]` (Sogang University), Youngjae Kim `[通讯]` (Sogang University)

**通讯引用:** 3250 | [OpenAlex ID](https://openalex.org/A5100458491)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了一套实时自适应学习系统AFLL，用来学习并控制MMO游戏服务器中的循环因果反馈，提前预防负载峰值。

**💡 创新点**

创新点在于：
• 第一次在MMO服务器中实时学习并调节循环因果回路；
• 通过梯度下降的线性回归学习每种消息类型的负载贡献；
• 将学习与控制并行化，利用预测缓存、增量统计缓存和后台学习线程实现0ms的学习开销；
• 采用概率阻塞与预测阈值相结合的四步决策，既保证游戏关键消息，又在负载峰值前主动抑制。

**🔧 技术方法**

技术主要包括：
• 梯度下降（back‑propagation）学习负载权重；
• 预测缓存（O(k)→O(1)）和增量统计缓存（O(n×200)→O(1)）提升计算效率；
• 并行学习与控制（后台线程、无锁队列）；
• 统计学方法（线性回归、窗口滑动、动量）；
• 线程阻塞与负载阈值控制。

**📊 数据集**

数据集：在模拟环境中使用 1,000 并发玩家的虚拟客户端，生成与真实 MMO 行为相似的移动、攻击、技能等事件，持续 30 分钟的实验。没有使用公开数据集。

**📈 对比分析**

比较方法：对比 Learning OFF（固定权重）与 Learning ON（实时学习）三次独立实验。评价指标包括 CPU 时间、线程竞争率、负载评分、性能峰值频率、阻塞率等。
性能提升：
• CPU 时间平均下降 48.3%（从 6,143ms/s 降到 3,177ms/s）；
• 线程竞争率从 60.36% 降至 21.51%；
• 负载峰值超过 50ms 的情况下降 99.7%；
• 学习开销仅 38ms/s（约 1.3%），远低于节省的 2,966ms/s。可重复性优异，CV 仅 0.54%。

**⚠️ 局限性**

限制与不足：
• 仅学习消息传输导致的负载，忽略了 AI、逻辑、计时等非消息负载；
• 只在单机环境验证，未验证多服或跨区域的分布式场景；
• 学习窗口固定为 1 秒，可能无法完全捕捉更快或更慢的反馈周期；
• 对极端网络延迟或大规模玩家突发事件的鲁棒性尚未评估；
• 在安全性要求高的系统中直接丢包可能导致不可接受的后果，需要额外的安全保障措施。

---

## 82. Toward Adaptive Grid Resilience: A Gradient-Free Meta-RL Framework for Critical Load Restoration

**arXiv ID:** 2601.10973 | [PDF](https://arxiv.org/pdf/2601.10973v1)

**作者:** Zain ul Abdeen `[一作]` (Virginia Tech), Ming Jin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对分布式电网极端事件后关键负荷恢复，提出了基于元学习的无梯度强化学习框架（MGF‑RL）。

**💡 创新点**

创新点在于将一阶元更新与进化策略相结合，构建梯度无关的元强化学习方法，既消除二阶导数的计算成本，又实现对多变恢复任务的快速适应。

**🔧 技术方法**

主要技术包括：ES‑RL（进化策略）作为任务内优化；一阶元更新（MAML 近似）用于跨任务知识迁移；OpenDSS 仿真环境与线性功率流模型用于评估；深度策略网络表示决策。

**📊 数据集**

使用的数据集：IEEE‑13 与 IEEE‑123 变电站仿真网络；NREL 风光预测历史数据生成不同的可再生发电与误差场景；构造多任务负荷恢复案例。

**📈 对比分析**

与 ES‑RL、MAML‑RL、AC‑RL、MPC 等基线进行对比；在 SAIDI、恢复速度、奖励收敛等指标上，MGF‑RL 均优于对手，恢复率提升 27–41%，并能在控制时限内实现 90% 负荷恢复。

**⚠️ 局限性**

局限性包括：假设拓扑已重新配置、负荷需求恒定；仅针对离散时间恢复，缺少对动态负荷变化与拓扑变更的处理；在极端大规模系统下的实际部署尚未验证。

---

## 83. LogicLens: Leveraging Semantic Code Graph to explore Multi Repository large systems

**arXiv ID:** 2601.10773 | [PDF](https://arxiv.org/pdf/2601.10773v1)

**作者:** Niko Usai `[一作]` (Sourcesense), Raffaele Camanzo `[通讯]` (Sourcesense)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LogicLens，结合多仓库语义图与对话式代理，帮助开发者在大型软件系统中快速定位、理解和排查问题。

**💡 创新点**

创新点在于：① 将 AST 解析得到的结构图与 LLM 生成的自然语言语义节点融合，构建跨仓库的域实体、操作和工作流三维语义图；② 采用 ReAct+GraphRAG 架构，让代理动态选择工具检索不同层级的子图，从而实现跨项目的高层次查询和技术细节探究。

**🔧 技术方法**

技术手段包括：AST 解析（Tree‑sitter）、LLM 生成代码与项目摘要（GPT‑5/Claude 4.5）、图数据库与 GraphRAG 检索、ReAct 代理与多工具链（Projects、Entities、Codes、Graph Query、Source）。

**📊 数据集**

使用真实企业多仓库订单管理系统作为实验对象，并构造了 30 题专家级评测问答集，涵盖事实查询、跨源链接、预测分析等多维度问题。

**📈 对比分析**

与基线向量检索系统（Qdrant + n8n）对比，利用人类评估的三维指标（准确性、完整性、连贯性）。LogicLens 在准确性达到 69.5% 高分，连贯性 52.2% 高分，完整性显著提升（从 0% 到 26% 高分、低分从 43.48% 降至 8.7%）。

**⚠️ 局限性**

局限性包括：目前仅支持 Java/Python/Go/TypeScript 四种语言；只利用源代码，未纳入配置文件、API 规范、CI/CD、容器与部署信息；完整性仍不理想，需进一步完善跨语言和跨工件的知识融合。

---

## 84. Towards Tensor Network Models for Low-Latency Jet Tagging on FPGAs

**arXiv ID:** 2601.10801 | [PDF](https://arxiv.org/pdf/2601.10801v1)

**作者:** Alberto Coppi `[一作]` (University of Padua), Simone Montangero `[通讯]` (University of Padua)

**通讯引用:** 8575 | [OpenAlex ID](https://openalex.org/A5003672540)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并实现了基于张量网络（MPS和TTN）的低延迟喷射标记模型，并将其部署到FPGA上，完成数据嵌入、训练、后训练量化与硬件合成。

**💡 创新点**

首次证明张量网络能够在满足HL‑LHC L1触发器严格延迟和资源约束的前提下，提供可解释且与深度学习相当的喷射标记性能；利用量子互信息对特征进行分组，实现模型压缩与精度优化；同时提供两种硬件实现路径（HLS和VHDL）。

**🔧 技术方法**

张量网络（MPS、TTN）、固定点后训练量化、FPGA固件合成（Vitis HLS、Vivado VHDL）、量子互信息特征分析与资源感知合成。

**📊 数据集**

公开的低级粒子特征喷射数据集（每粒子 pT、E_rel、ΔR）来自LHC实验。

**📈 对比分析**

与现有深度学习模型（transformer、MLP‑Mixer）在分类准确率和AUC上对比，取得8、16、32个粒子分别约66%、72.5%和77.1%的准确率；在XCVU13P FPGA上实现子微秒级延迟（<1µs），符合L1触发器要求。

**⚠️ 局限性**

高精度模型在FPGA上DSP资源紧张，量化后性能对MPS更敏感；VHDL实现复杂；当前未包含完整的预处理和后处理流水线，未来需进一步优化量化技术和端到端FPGA集成。

---

## 85. Haptic Light-Emitting Diodes: Miniature, Luminous Tactile Actuators

**arXiv ID:** 2601.11043 | [PDF](https://arxiv.org/pdf/2601.11043v1)

**作者:** Max Linnander `[一作]` (University of California), Yon Visell `[通讯]` (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文提出并实现了Haptic Light-Emitting Diodes (HLEDs)，一种将光脉冲直接转化为热膨胀驱动弹性膜的微型光热气体驱动致动器。

**💡 创新点**

创新点在于将光源与光热吸收层集成在微型封闭腔体内，实现低电压下毫米级位移与0.4 N以上力的光致机械转换，同时实现可视化光输出。

**🔧 技术方法**

技术包括表面贴装LED、石墨光吸收层、PDMS弹性膜、铝/PS多层腔体以及有限元仿真和力/位移测量。

**📊 数据集**

数据集：实验使用的光功率、电流、脉冲持续时间等参数的测量数据，心理学实验收集的七名受试者的强度评价。

**📈 对比分析**

通过与其他微型致动器（电磁、静电、压电等）对比，HLED在0.5–200 Hz频率下实现1 mm位移、440 mN力，响应时间5–100 ms，显示出更高的力/位移比和更低的驱动电压。

**⚠️ 局限性**

局限性包括热量累积导致的热漂移、光吸收率仅约30%、工作效率仅约0.08%以及对高温光吸收层材料的耐久性要求。

---

## 86. Combating Spurious Correlations in Graph Interpretability via Self-Reflection

**arXiv ID:** 2601.11021 | [PDF](https://arxiv.org/pdf/2601.11021v1)

**作者:** Kecheng Cai `[一作]`, Chao Peng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种无训练、可迭代自我反思的图解释框架，通过多轮将上游模块产生的边重要性掩码反馈回自身，逐步过滤掉与任务无关的结构，提升解释质量与分类性能。

**💡 创新点**

创新点在于将 LLM 中的自我反思概念迁移至图学习，构建了乘法更新的递归掩码机制，并在此基础上提出一致性正则化的微调目标，证明了最优掩码可保持一致性。

**🔧 技术方法**

技术包括 L2X 解释框架、图神经网络（GIN、PNA、LE 等）、自我反思迭代掩码、互信息约束、ℓ1 一致性损失以及实验评估指标（准确率、AUC）。

**📊 数据集**

主要在 Spurious‑Motif 合成基准上测试，并在 BA‑2Motifs、Mutag、MolBACE、MolBBBP、MolHIV 等公开图数据集上进行验证。

**📈 对比分析**

与现有解释方法（GNNExplainer、PGExplainer、DIR、GSAT 等）相比，迭代自我反思框架在 Spurious‑Motif 上提升了 2–5% 的准确率、AUC，微调版本更显著，尤其在高比例诱导偏差（b=0.9）场景下显著压缩训练-测试误差。

**⚠️ 局限性**

局限性包括：对极高迭代次数会出现性能下降；需要先行训练好的解释模型；仅在离线环境下评估，未探索在线或 RL‑驱动的自适应策略；在无明显伪相关的真实数据集上增益有限。

---

## 87. Matching High-Dimensional Geometric Quantiles for Test-Time Adaptation of Transformers and Convolutional Networks Alike

**arXiv ID:** 2601.11022 | [PDF](https://arxiv.org/pdf/2601.11022v1)

**作者:** Sravan Danda `[一作]` (BITS Pilani), Snehanshu Saha `[通讯]` (BITS Pilani)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种与网络架构无关的测试时适应（TTA）方法，通过在预训练分类器前添加装饰器网络，将测试图像映射到训练分布，并使用几何分位数损失在特征空间中匹配分布。

**💡 创新点**

创新点包括：
① 在高维特征空间中使用几何分位数来衡量分布相似度，避免仅靠均值和方差；
② 引入量化损失并证明在“良好初始化”下与配对MSE等价，可对齐类别条件；
③ 设计可在小批量训练下高效估计的复合可分离损失，并利用记忆池实现控制变量（SVRG）来降低梯度方差；
④ 该方法对CNN和Transformer均适用，突破了以往仅适用于BN或特定网络结构的限制。

**🔧 技术方法**

技术实现包括：
- 几何分位数理论与量化损失；
- 以卷积/反卷积为主的装饰器网络；
- 记忆池与控制变量（SVRG）实现的小批量优化；
- 训练中加入批归一化统计正则化；
- 在多种网络架构（ResNet、CCT、CVT、ViT‑Lite）上评估。

**📊 数据集**

实验数据集：
- CIFAR10-C、CIFAR100-C、TinyImageNet-C（各自的干净版本为 CIFAR10、CIFAR100、TinyImageNet）。
- 以 5 级污染强度为主测试环境。

**📈 对比分析**

性能对比：
- 与直接使用预训练模型（Baseline）对比，量化损失方法在所有网络上显著提升准确率（最高提升约 25%）。
- 与 SOTA 方法 SoTTA、TENT、CoTTA 等对比，CNN 上略低于 SoTTA，但在 Transformer 架构上超越或接近 SoTTA，提升 6‑15% 左右。整体表现稳定，特别是在更复杂网络上表现更好。

**⚠️ 局限性**

局限性：
- 对 CNN 模型的提升仍略逊于现有 SOTA；
- 需要额外的装饰器网络和记忆池，增加模型复杂度；
- 对超参数（量化数、批量大小）敏感；
- 理论证明依赖于“良好初始化”，实际可能不总满足；
- 目前仅针对静态 covariate shift，未处理动态分布漂移或更复杂的目标分布。

---

## 88. Is open robotics innovation a threat to international peace and security?

**arXiv ID:** 2601.10877 | [PDF](https://arxiv.org/pdf/2601.10877v1)

**作者:** Ludovic Righetti `[一作]` (New York University), Vincent Boulanin `[通讯]` (Stockholm International Peace Research Institute)

**通讯引用:** 124 | [OpenAlex ID](https://openalex.org/A5023984953)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

分析开放式机器人创新对国际和平与安全的双重使用风险，并提出四项可操作的风险治理路线图。

**💡 创新点**

首次系统性提出针对机器人领域的开放性风险评估与治理框架，并结合跨学科经验制定教育、激励、监管与红线等具体措施。

**🔧 技术方法**

基于风险管理方法（风险识别、评估与缓解）和现有技术（开源软件、机器学习模型等）构建风险评估工具与治理策略。

**📊 数据集**

未使用特定实验数据集，主要引用公开文献与案例（如乌克兰冲突、ISIS无人机）进行说明。

**📈 对比分析**

通过对比现有核武器、化学武器等领域的监管经验，评估机器人领域缺乏指导的现状，建议引入分层风险门控与合规评估。

**⚠️ 局限性**

局限在于缺乏实证验证与量化指标，主要为理论性建议，未来需通过案例研究与工具实现进行评估。

---

## 89. Gamifying Cyber Governance: A Virtual Escape Room to Transform Cybersecurity Policy Education

**arXiv ID:** 2601.10852 | [PDF](https://arxiv.org/pdf/2601.10852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 90. ORBITFLOW: SLO-Aware Long-Context LLM Serving with Fine-Grained KV Cache Reconfiguration

**arXiv ID:** 2601.10729 | [PDF](https://arxiv.org/pdf/2601.10729v1)

**作者:** Xinyue Ma `[一作]` (POSTECH), Myeongjae Jeon `[通讯]` (POSTECH)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于 SLO 的长上下文 LLM 服务系统，能够动态地细粒度重构 KV 缓存的 GPU 与主机内存布局，从而在严格的延迟目标下显著提升吞吐量与服务质量。

**💡 创新点**

创新点包括：①使用轻量 ILP 求解器在运行时根据请求长度与 GPU 带宽自适应决定每层 KV 的放置；②引入 Token Deposit 与 Pause‑Resume 机制在内存压力或 SLO 触发时平滑延迟并避免请求被完全丢弃；③通过在单个 GPU 上实现按请求分层的 KV 分配，并在多 GPU 环境下共享同一规划结果，提升了系统的可扩展性。

**🔧 技术方法**

主要技术手段有：轻量 ILP 优化、动态 KV 管理器、Token Deposit 缓冲、Pause‑Resume 预取/延迟机制、单线程异步求解与预调度、分布式 Tensor‑Parallel 支持。

**📊 数据集**

使用的数据集为基于 ShareGPT 生成的合成请求轨迹（请求长度最高 400k token），模型采用 LLaMA3-8B 与 LLaMA3-70B，并在单 GPU（RTX A5000）与多 GPU（4× RTX A6000）硬件上评估。

**📈 对比分析**

与 DeepSpeed‑Inference、FlexGen、FlexGen+、SLO‑aware Offloading、Dynamic Heuristic 等基线对比，ORBITFLOW 在 TPOT 与 TBT SLO 达成率分别提升高达 66% 与 48%，P95 TBT 延迟降低 38%，吞吐量提升 3.3×，且整体运行时开销不到 1%。

**⚠️ 局限性**

局限性包括：①搜索空间仅考虑等距 offloading，可能错过更优的非均匀布局；②目前仅支持 GPU‑CPU 内存迁移，未涉及 SSD 或更大规模多机扩展；③依赖准确的计算与带宽估计，若系统状态与模型推断时差异大，规划效果会下降；④在极端高并发或极长上下文时仍需更精细的调度与资源分配策略。

---

## 91. Struggling to Connect: A Researchers' Reflection on Networking in Software Engineering

**arXiv ID:** 2601.10907 | [PDF](https://arxiv.org/pdf/2601.10907v1)

**作者:** Shalini Chakraborty `[一作]` (University of Bayreuth), Shalini Chakraborty `[通讯]` (University of Bayreuth)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5085515034)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析并讨论软件工程研究者在不同地理、移民、语言、性别、个性等维度下的网络机会与障碍，并提出社区行动方案

**💡 创新点**

提出“专家声音”报告框架和可视化网络结构的建议，强调网络是系统性结构问题而非个人短板

**🔧 技术方法**

无技术或方法

**📊 数据集**

无数据集

**📈 对比分析**

无比较方法，未进行性能评估

**⚠️ 局限性**

缺乏实证数据支持，主要为反思性描述，难以量化网络不平等

---

## 92. M3DDM+: An improved video outpainting by a modified masking strategy

**arXiv ID:** 2601.11048 | [PDF](https://arxiv.org/pdf/2601.11048v1)

**作者:** Takuya Murakawa `[一作]` (Nagoya Institute of Technology), Toru Tamaki `[通讯]` (Nagoya Institute of Technology)

**通讯引用:** 2030 | [OpenAlex ID](https://openalex.org/A5068412717)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 M3DDM+，一种改进的训练方法，通过在所有帧上统一掩码方向和宽度来提升视频 outpainting 的生成质量。

**💡 创新点**

创新点在于识别并纠正 M3DDM 训练与推理期间掩码策略的不匹配，通过统一掩码来降低模型对跨帧信息的过度依赖，从而显著减少模糊和时间不一致问题。

**🔧 技术方法**

采用隐层扩散模型（Latent Diffusion Model）与 3D U‑Net、VAE 编码/解码、跨帧注意力以及细化掩码策略等技术，结合 fine‑tuning 进一步提升效果。

**📊 数据集**

训练使用 WebVid 子集（10,000 条视频），评估在 DAVIS 和无运动版本 DAVIS Static 数据集上进行。

**📈 对比分析**

通过 MSE、PSNR、SSIM、LPIPS、FVD 和自定义的 BMSE 指标与原 M3DDM 进行对比，结果显示 M3DDM+ 在大掩码比例和无相机运动等信息受限场景中显著提升了所有指标，尤其是降低了模糊度（BMSE）并提升了视觉质量和时间一致性。

**⚠️ 局限性**

局限性包括：仍需要 256×256 分辨率的 VAE 编码，推理仍需 GPU 计算资源；对极大掩码或复杂动态场景的适应性可能有限；且在极端信息缺失时仍可能出现细节损失。

---

## 93. AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts

**arXiv ID:** 2601.11044 | [PDF](https://arxiv.org/pdf/2601.11044v1)

**作者:** Keyu Li `[一作]` (Shanghai Jiao Tong University), Pengfei Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10150 | [OpenAlex ID](https://openalex.org/A5100355001)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AgencyBench benchmark，评估自主代理在真实世界长时序任务中的能力。

**💡 创新点**

构建 32 场真实场景、138 任务，平均需 1M token、90 次工具调用；提供自动化评估框架，包括用户模拟代理和 Docker 沙箱。

**🔧 技术方法**

采用 LLM 主体的代理框架、工具套件、用户模拟代理、Docker 隔离环境、可执行评估脚本、LLM 作为评判者等技术。

**📊 数据集**

任务数据由 20 位专家收集，包含查询、交付物、评估标准，形成 138 个任务。

**📈 对比分析**

通过与 9 种 LLM（闭源与开源）在 AgencyBench 上对比，闭源模型平均得分 48.4% 远高于开源 32.1%，并分析自纠、资源效率和工具使用差异。

**⚠️ 局限性**

仅评估软件/数字任务，未覆盖物理/机器人；模型选择有限；基于 LLM 评判的可靠性仍需进一步验证。

---

## 94. Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data

**arXiv ID:** 2601.11038 | [PDF](https://arxiv.org/pdf/2601.11038v1)

**作者:** Xuanming Zhang `[一作]` (Columbia University), Dan Roth `[通讯]` (Oracle AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“Anytime Reasoning”评估框架和 Anytime Index 指标，并提出一种基于 LLM 自生成对比偏好数据的推理时自提升方法。

**💡 创新点**

创新在于量化模型在不同令牌预算下的推理质量曲线，并利用无监督的偏好对比来提升中间推理质量。

**🔧 技术方法**

采用 Chain-of-Thought (CoT)、自我生成的偏好对比数据提示以及基于 AUC 的 Anytime Index 计算等技术。

**📊 数据集**

实验使用 NaturalPlan（行程规划）、AIME 2024（数学推理）和 GPQA-Diamond（科学问答）三大基准数据集。

**📈 对比分析**

与标准 CoT、LEAP 原则学习以及仅使用高质量示例的 PDP(+) 进行对比，PDP 在三大基准上平均提升 Anytime Index 约 7–10%，并显著提高最终准确率和中间推理质量。

**⚠️ 局限性**

实验范围有限，仅覆盖规划/数理/科学三类任务；未与 Tree-of-Thoughts、Self-Consistency 等更先进提示方式进行比较；且只在推理时使用偏好提示，未将偏好监督纳入模型训练。

---

## 95. Sparse Data Tree Canopy Segmentation: Fine-Tuning Leading Pretrained Models on Only 150 Images

**arXiv ID:** 2601.10931 | [PDF](https://arxiv.org/pdf/2601.10931v1)

**作者:** David Szczecina `[一作]` (University of Waterloo), Lincoln Linlin Xu `[通讯]` (University of Calgary)

**通讯引用:** 2799 | [OpenAlex ID](https://openalex.org/A5034166335)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在仅有150张稀疏树冠图像的条件下，对YOLOv11、Mask R‑CNN、DeepLabv3、Swin‑UNet和DINOv2进行微调，评估其在遥感树冠分割任务中的表现。

**💡 创新点**

证明在极低数据量场景下，卷积基模型的空间先验和大规模预训练显著优于Transformer架构，揭示了实例分割与语义分割任务差异对模型性能的影响。

**🔧 技术方法**

采用预训练权重微调、数据增强、像素级分割头和实例化后处理等技术，并比较不同模型的mAP和像素准确率。

**📊 数据集**

使用Solafune Tree Canopy Detection比赛提供的150张RGB航空图像数据集，包含5个分辨率等级。

**📈 对比分析**

通过对比5种模型的验证像素准确率与测试加权mAP，YOLOv11和Mask R‑CNN在加权mAP上分别达到0.281和0.219，远高于DeepLabv3、Swin‑UNet和DINOv2。

**⚠️ 局限性**

Transformer模型在小样本上过拟合，缺乏足够的先验导致实例检测性能差，且当前实验未探索混合CNN–ViT或更高级微调策略，限制了对更复杂场景的适用性。

---

## 96. A Survey of Real-Time Support, Analysis, and Advancements in ROS 2

**arXiv ID:** 2601.10722 | [PDF](https://arxiv.org/pdf/2601.10722v1)

**作者:** Daniel Casini `[一作]` (Scuola Superiore Sant'Anna), Harun Teper `[通讯]` (TU Dortmund University)

**通讯引用:** 100 | [OpenAlex ID](https://openalex.org/A5064144075)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述 ROS 2 在实时性方面的研究现状，梳理了调度机制、通信层、分析方法、定制执行器、系统级改进、工具与实验等多维度成果。

**💡 创新点**

提出了多套统一的分类法（调度器类型、系统模型、时序指标、执行时间模型、到达模型、DDS 通信、OS 模型）和针对 DDS 与同步器的专门表格，全面归纳并对比了现有研究，填补了学术界与工业界对 ROS 2 实时性能理解的空白。

**🔧 技术方法**

综合利用实时系统理论（响应时间分析、离散时间模型、可观测性分析）、模型检查（UPPAAL 时序自动机）、DDS 性能模型（FastDDS/Connext 线程模型）、自定义执行器实现（固定优先级、EDF、锁无竞争等）以及大规模实验测评工具（ros2_tracing、LTTng、eBPF 等）来评估与验证 ROS 2 的实时特性。

**📊 数据集**

引用并对比了多种典型应用与数据集：Turtlebot3 导航栈、Autoware 自动驾驶软件、机器人赛车（TUM Autonomous Motorsport）以及工业机器人仿真（CARLA）等；通过这些真实系统案例展示了不同方法的适用范围与效果。

**📈 对比分析**

通过表格与实验数据对比，作者展示了各类分析/验证方法在不同指标（E2E 反应时间、数据年龄、时延抖动、单/多线程执行器可调度性）上的优劣；实验表明改进执行器或系统级调度可将端到端延迟降低 30%–90%，但仍受限于 DDS 传输模式与 OS 预emption 级别。

**⚠️ 局限性**

局限性在于：①综述基于已有文献，无法覆盖最新未公开工作；②对多分布式系统的跨节点同步与通信延迟建模仍不够完整；③缺乏统一的基准平台与数据集，导致不同研究间的可比性受限；④对深度学习/大数据场景下的实时性问题探讨不足。

---

## 97. Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation

**arXiv ID:** 2601.10880 | [PDF](https://arxiv.org/pdf/2601.10880v1)

**作者:** Chongcong Jiang `[一作]` (University of Central Florida), Yu Tian `[通讯]` (University of Central Florida)

**通讯引用:** 539 | [OpenAlex ID](https://openalex.org/A5103270744)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

在33个多模态医学图像数据集上对SAM3进行全参数微调，构建了可通过纯文本提示完成二维与三维医学图像分割的Medical SAM3模型。

**💡 创新点**

①全参数微调而非仅使用轻量化适配；②通过文本-视觉语义对齐消除对空间提示的依赖；③在统一的二维表示下实现跨模态、跨维度的分割；④构建大规模文本‑图像‑掩码配对语料库。

**🔧 技术方法**

使用SAM3基础架构、层级学习率衰减、集合预测的交叉匹配损失、文本-视觉语义对齐蒸馏以及多任务训练策略。

**📊 数据集**

33个公开数据集，覆盖10种成像模态（CT、MRI、X光、超声、内镜、视网膜、皮肤病、组织病理等），共约77 000张图像和263 000个掩码。

**📈 对比分析**

与原始SAM3在10个内部验证集和7个外部未见集进行Dice与IoU比较，Medical SAM3平均Dice提升约23个百分点、IoU提升约24个百分点，尤其在小、薄或低对比度结构上表现显著提升。

**⚠️ 局限性**

计算量大且高分辨率训练耗时；仍依赖二维平面表示未充分利用三维连续性；文本提示局限于原子概念，未考虑同义词或属性组合；缺乏多中心验证与不确定性估计。

---

## 98. FAConvLSTM: Factorized-Attention ConvLSTM for Efficient Feature Extraction in Multivariate Climate Data

**arXiv ID:** 2601.10914 | [PDF](https://arxiv.org/pdf/2601.10914v1)

**作者:** Francis Ndikum Nji `[一作]`, Jianwu Wang `[通讯]` (University of Maryland)

**通讯引用:** 3701 | [OpenAlex ID](https://openalex.org/A5101750217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出FAConvLSTM模型，用于高分辨率多变量气候数据的高效特征提取与时空表示学习。

**💡 创新点**

创新点包括：①使用1×1瓶颈和共享深度卷积分解门计算，显著降低通道复杂度；②多尺度膨胀深度卷积与SE通道重校正，捕获多尺度物理过程；③稀疏轴向注意力在时间上定期注入遥感连接信息；④子空间嵌入头结合时间自注意力，生成紧凑、可解释的时序嵌入。

**🔧 技术方法**

技术栈：瓶颈投影、共享深度卷积、膨胀卷积、Squeeze‑and‑Excitation、融合门与Peephole、轴向注意力、子空间嵌入、时间多头自注意力、空间拉普拉斯平滑正则。

**📊 数据集**

主要数据集：ERA5 高分辨率气候重分析数据（含多变量空间时间张量）。

**📈 对比分析**

在ERA5上与CNN、CNN‑LSTM、ConvLSTM三种基线进行聚类与重构指标对比，FAConvLSTM在Silhouette、DB、CH、RMSE、Var、I‑CD等指标上均表现最佳，尤其在群聚密度和遥感连接方面显著优于基线。

**⚠️ 局限性**

局限性：①轴向注意力仅稀疏使用，可能无法捕获所有非局部依赖；②子空间维度需手工设定，影响表达丰富度；③在更大尺度或不同任务上仍需验证其通用性和计算成本；④模型对极端事件的鲁棒性尚未全面评估。

---

## 99. LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning

**arXiv ID:** 2601.10775 | [PDF](https://arxiv.org/pdf/2601.10775v1)

**作者:** Tommaso Felice Banfi `[一作]` (Politecnico di Milano), Sashenka Gamage `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于大型语言模型的离散博弈推理框架，结合了熵引导的上下文检索和自适应链式推理，展示了在井字棋中的应用。

**💡 创新点**

创新点在于利用模型的 token‑级熵动态决定检索例子的数量与推理路径的分支，既实现了零样本推理，又显著降低了查询成本。

**🔧 技术方法**

核心技术包括自监督的棋盘向量编码器（带对比损失）、检索增强推理（RAG）、熵引导的自适应链式推理以及基于 LLaMA‑7B 的 LLM 生成。

**📊 数据集**

使用的“数据集”是井字棋的全部棋盘状态（约 20% 用于检索），并借助 Minimax 表获取最优动作作为监督。

**📈 对比分析**

与基线（无上下文、固定上下文、单/多路径 CoT 等）对比，熵引导自适应 CoT 在 100 场游戏中平均得分提升至 +9.5%，平均查询数仅 48，远低于全树 CoT 的 188 次。

**⚠️ 局限性**

局限性包括仅在小型确定性棋盘上验证、对更大或部分可观测环境的可扩展性未知、对 token 熵作为不确定性指标的假设及检索数据库的完整性依赖。

---

## 100. When Personalization Misleads: Understanding and Mitigating Hallucinations in Personalized LLMs

**arXiv ID:** 2601.11000 | [PDF](https://arxiv.org/pdf/2601.11000v1)

**作者:** Zhongxiang Sun `[一作]` (Renmin University of China), Jun Xu `[通讯]` (Renmin University of China)

**通讯引用:** 13586 | [OpenAlex ID](https://openalex.org/A5020766468)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了个性化LLM导致的事实性幻觉问题，并提出了在推理时动态调节个性化的FPPS框架。

**💡 创新点**

创新点是识别并消除个性化与事实知识的表示耦合，并在保持个性化的同时提升事实准确率。

**🔧 技术方法**

采用了表示偏移定位器、事实纠缠探测器和自适应知识导向模块，并实现了三种调节模式（硬、软、混合）。

**📊 数据集**

使用了新构建的PFQABench（包含个性化与事实问答），并在多种公开LLaMA和Qwen模型上评估。

**📈 对比分析**

与四种主流个性化策略（PAG、DPL、RAG、LLM-TRSR）对比，FPPS-M在整体得分上提升约50%，显著提升事实准确率，同时保持个性化性能。

**⚠️ 局限性**

局限在于仅在开放权重模型上验证，对闭源API不适用，且需进一步扩展到更大规模和多样化的模型与更丰富的评测数据。

---

## 101. Your One-Stop Solution for AI-Generated Video Detection

**arXiv ID:** 2601.11035 | [PDF](https://arxiv.org/pdf/2601.11035v1)

**作者:** Long Ma `[一作]` (University of Science and Technology of China), Zhen Bi `[通讯]` (Huzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了AIGVDBench基准，包含31种生成模型、44万视频的高质量数据集，并系统评估了四类AI生成视频检测方法。

**💡 创新点**

创新点包括：① 采用属性平衡采样算法实现语义与属性均衡的提示集合；② 提供规模最大、质量最高的生成视频基准；③ 通过跨模型训练/测试矩阵揭示生成模型质量与检测难度不直接相关；④ 对视频分类、图像检测、视频检测和多模态大语言模型四种检测范式进行了深入对比与新发现。

**🔧 技术方法**

使用了属性平衡采样算法、视频统一预处理（H.264压缩、帧采样、尺寸归一化）、多类检测模型（I3D、Effort、DeCoF、Deepseek‑VL‑2等）以及VBench评估框架。

**📊 数据集**

基准数据集AIGVDBench：20个开源模型、11个闭源模型，约440,000个视频（约14M帧），来源于OpenVid-HD、VBench评测集、官方演示等公开与闭源渠道。

**📈 对比分析**

采用VBench框架，先用Open‑Sora训练基线模型，再对33个检测器进行1,500+评估。结果显示所有四类方法均存在提升空间，闭源模型更难检测，Effort和ForgenLens在开闭源上表现较好，VLM性能整体落后但Deepseek‑VL‑2在闭源模型上表现突出。

**⚠️ 局限性**

局限性：① 仍未涵盖深度伪造面部；② 统一压缩可能掩盖低级特征导致偏差；③ 评估仅聚焦文本‑到‑视频场景，未覆盖其他生成任务；④ 跨模型评估受检测器与生成模型匹配程度影响；⑤ VLM仅输出标签，缺乏概率，影响性能评估。

---

## 102. DialDefer: A Framework for Detecting and Mitigating LLM Dialogic Deference

**arXiv ID:** 2601.10896 | [PDF](https://arxiv.org/pdf/2601.10896v1)

**作者:** Parisa Rabbani `[一作]` (University of Illinois Urbana-Champaign), Dilek Hakkani-Tür `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 9732 | [OpenAlex ID](https://openalex.org/A5068709817)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DialDefer 框架，通过在同一信息下比较事实询问与说话者评估两种对话框架，评估大型语言模型在第三方判断中的“对话式递让”偏差。

**💡 创新点**

创新点在于首次量化并揭示对话框架对模型判断的方向性偏移，提出 Dialogic Deference Score（DDS）指标，展示其与传统准确率的差异，并揭示人类 vs. AI 的说话者属性对偏差的驱动作用。

**🔧 技术方法**

使用四种 LLM（GPT‑4o、GPT‑4o‑mini、Gemma‑3‑12B、Qwen‑2.5‑7B）进行实验，并通过 Prompt 设计、SFT 与 DPO 等对齐与校准技术来尝试减轻偏差。

**📊 数据集**

构建统一基准（九个 QA、社交推理、专业知识与主观建议数据集）共 3,244 条样本，并收集 Reddit r/AIO 真实对话 280 条，用于检验自然对话中的偏差放大。

**📈 对比分析**

与传统准确率对比，计算 DDS 并在四模型、九域上比较；发现准确率变化 <2pp，但 DDS 可达 ±87pp；SFT 在大多数域提升准确率 +22pp、DDS 减少 24pp，但在 r/AIO 上往往过度偏向怀疑。

**⚠️ 局限性**

局限在于仅针对英文西方语境，未覆盖多语言与跨文化情境；实验对话极简，缺乏多轮交互；r/AIO 数据样本量小且类别不平衡，真值以社区共识为准；对齐干预难以在保持准确率的同时消除 DDS，需要更精细的校准策略。

---

## 103. Effects of Different Attention Mechanisms Applied on 3D Models in Video Classification

**arXiv ID:** 2601.10854 | [PDF](https://arxiv.org/pdf/2601.10854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 104. Mugi: Value Level Parallelism For Efficient LLMs

**arXiv ID:** 2601.10823 | [PDF](https://arxiv.org/pdf/2601.10823v1)

**作者:** Daniel Price `[一作]` (University of Central Florida), Di Wu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8543 | [OpenAlex ID](https://openalex.org/A5011648245)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种将值层并行（VLP）扩展到Transformer LLM的非线性近似与小批量异步BF16‑INT4 GEMM的架构，并将非线性计算与GEMM共享同一计算阵列，完成完整LLM工作负载的加速。

**💡 创新点**

创新点在于：①采用输入近似和值中心化的VLP非线性近似方法；②将VLP与分组查询注意力（GQA）、权重/KV缓存量化（WOQ/KVQ）结合，支持异步量化的BF16‑INT4 GEMM；③通过共享阵列实现非线性与GEMM的资源复用，显著降低面积与碳排放。

**🔧 技术方法**

核心技术包括值层并行（VLP）、时序订阅与转换、滑动窗口LUT、输入近似、异步量化（WOQ/KVQ）、分组查询注意力（GQA）以及利用单节点/多节点NoC的阵列共享设计。

**📊 数据集**

实验在多种LLM模型上进行：LLaMA2（7B/13B/70B）、Whisper Tiny、SwinV2、ViViT等，使用公开的HuggingFace Transformers数据集进行100次推理并收集中间张量。

**📈 对比分析**

与Carat、Systolic、SIMD、FIGNA、Tensor Core等基线比较，单节点上吞吐率提升至2.07×，能效提升3.11×；非线性操作吞吐率提升45×，能效提升668×；整体碳排放降低1.45×（运行）和1.48×（嵌入）。

**⚠️ 局限性**

局限性包括：尚未支持层归一化、RoPE、MoE、多模态等操作；使用离线预计算LUT，输入分布漂移可能影响精度；需要进一步验证在线LUT自适应与更大规模模型的适用性。

---

## 105. Realistic Curriculum Reinforcement Learning for Autonomous and Sustainable Marine Vessel Navigation

**arXiv ID:** 2601.10911 | [PDF](https://arxiv.org/pdf/2601.10911v1)

**作者:** Zhang Xiaocai `[一作]`, Zhang Wenbin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并验证了一种基于课程强化学习的可持续海上船舶导航框架，集成了数据驱动的模拟环境、扩散模型生成动态航迹以及燃料消耗预测模块。

**💡 创新点**

创新点包括：①将课程学习与强化学习耦合，逐步提升任务难度；②在高保真AIS+扩散模型仿真环境中引入图像化状态表示；③利用XGBoost对燃料消耗进行精准预测并融入多目标奖励函数。

**🔧 技术方法**

所用技术：课程强化学习（CRL）、PPO算法、扩散模型（DM）用于合成航迹、XGBoost燃料预测、卷积神经网络+全连接网络的混合状态处理。

**📊 数据集**

使用数据集：公开AIS船舶轨迹数据、行业合作的海上燃料消耗记录、印度洋海域的海流速度与方向月度分布。

**📈 对比分析**

方法对比：与CL‑ABDDQN、CL‑A2C、DDPG三种基线模型在三种航行实例中进行累计奖励、燃料消耗和安全评分的比较，CRL在累计奖励最高、燃料消耗最低、碰撞安全评分最低；课程学习提升了收敛速度与训练稳定性。

**⚠️ 局限性**

局限性：依赖行业内部燃料数据，仿真环境的可迁移性有限，缺乏实船实验验证，对极端天气或不可预知事件的鲁棒性尚未评估。

---

## 106. A Quantum-Driven Evolutionary Framework for Solving High-Dimensional Sharpe Ratio Portfolio Optimization

**arXiv ID:** 2601.11029 | [PDF](https://arxiv.org/pdf/2601.11029v1)

**作者:** Mingyang Yu `[一作]` (Nankai University), Jing Xu `[通讯]` (Nankai University)

**通讯引用:** 9323 | [OpenAlex ID](https://openalex.org/A5017604391)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种基于量子混合差分进化（QHDE）的高维Sharpe比率投资组合优化算法。

**💡 创新点**

创新点在于三大改进的融合：良点集–混沌逆学习初始化、量子进化策略以及动态精英池与Cauchy–Gaussian混合扰动，显著提升全局搜索与局部收敛能力。

**🔧 技术方法**

主要技术包括量子启发式差分进化（DE）、混沌映射、良点集初始化、Cauchy‑Gaussian混合扰动及精英池策略。

**📊 数据集**

使用的数据集为CEC 2020/2022 benchmark函数（高维多峰约束优化）以及中国沪深300指数的20、40、60、80只股票的收盘价构成的实盘组合。

**📈 对比分析**

与七个先进算法（SASS、COLSHADE、sCMAES、HSEPSO、NDSOT、DE、ADE）在上述 benchmark 和四规模投资组合上比较，QHDE平均提升约 29–73% 的性能，收敛最快、最稳健，且在约束满足度上表现最佳。

**⚠️ 局限性**

局限性包括对参数调优依赖经验、缺乏理论收敛性证明，以及仅验证单目标Sharpe比率，未覆盖多目标或动态交易场景。

---

## 107. AJAR: Adaptive Jailbreak Architecture for Red-teaming

**arXiv ID:** 2601.10971 | [PDF](https://arxiv.org/pdf/2601.10971v1)

**作者:** Yipu Dou `[一作]` (Southeast University), Wang Yang `[通讯]` (Southeast University)

**通讯引用:** 28972 | [OpenAlex ID](https://openalex.org/A5100322731)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了AJAR框架，用于对大型语言模型代理进行红队评估，实现协议驱动的认知编排，支持多轮动态工具调用与自适应攻击。

**💡 创新点**

创新点：① 引入 Model Context Protocol (MCP) 将攻击逻辑模块化为可插拔服务，解耦攻击与执行循环；② 通过 Auditor Agent 实现状态回溯、规划与策略执行；③ 系统化探讨“Agentic Gap”，揭示工具使用对安全的双刃效应。

**🔧 技术方法**

技术栈：Petri 运行时、MCP 协议、X‑Teaming 算法、LLM 评判器、工具调用模拟、强化学习式规划、动态回溯与重置。

**📊 数据集**

数据集与模型：使用 HarmBench 的高危查询集进行实验，模型为 DeepSeek V3.2（Auditor 与 Target 同为该模型），并参考 OpenRT 基准进行框架对比。

**📈 对比分析**

评估方法：在文本仅对话配置与工具增强配置两种环境下，执行相同攻击路径，比较成功率、拒绝率及交互步数；实验表明工具使用降低 Persona‑based 攻击成功率，但开启了间接代码注入的新通道；但未进行大规模量化，性能评估仍为定性。

**⚠️ 局限性**

局限性：① 仅完成架构可行性验证，缺乏大规模量化评估；② 研究聚焦单一模型与单一攻击算法，未验证对更强模型或多算法的适用性；③ 仅处理文本与模拟工具，未覆盖多模态或真实环境的安全挑战。

---

## 108. On the Entropy of a Random Geometric Graph

**arXiv ID:** 2601.10778 | [PDF](https://arxiv.org/pdf/2601.10778v1)

**作者:** Praneeth Kumar Vippathalla `[一作]` (University of Oxford), Mihai-Alin Badiu `[通讯]` (University of Oxford)

**通讯引用:** 695 | [OpenAlex ID](https://openalex.org/A5026106230)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了硬随机几何图（Hard RGG）在单位立方体和单位环面上的熵的渐近行为，并给出了上界、下界及在特定范围下的精确结果。

**💡 创新点**

首次在硬RGG中给出熵的闭式或渐近式表达，证明其为 dm log m 级别，并揭示结构熵在高维下占主导。

**🔧 技术方法**

采用Warren定理、卷积几何、Crofton细胞、Poisson化和熵不等式等工具推导上、下界。

**📊 数据集**

无真实数据集，论文为理论分析，使用随机点的均匀分布模型。

**📈 对比分析**

与Erdős–Rényi、SBM等已知模型的熵对比，发现硬RGG熵量级为 dm log m，显著高于软RGG或ER的情况。

**⚠️ 局限性**

主要限制是仅对固定连接半径r及r≤1/4的情况给出严格结论，且证明技术复杂，实际压缩方案尚未给出。

---

## 109. Chatting with Confidants or Corporations? Privacy Management with AI Companions

**arXiv ID:** 2601.10754 | [PDF](https://arxiv.org/pdf/2601.10754v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 110. Building AI Agents to Improve Job Referral Requests to Strangers

**arXiv ID:** 2601.10726 | [PDF](https://arxiv.org/pdf/2601.10726v1)

**作者:** Ross Chu `[一作]` (University of California Berkeley), Yuting Huang `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Blind在线社区开发AI代理，帮助求职者改写求职推荐请求以提升获得员工推荐的概率。

**💡 创新点**

创新点在于将奖励模型、检索增强生成(RAG)与LLM编辑相结合，针对弱请求显著提升成功率且不降低强请求。

**🔧 技术方法**

使用的技术包括句子变换器奖励模型、低秩适配LoRA微调、检索器挑选高质量示例、解释器提供句子重要性评分以及大语言模型进行文本改写。

**📊 数据集**

数据集为2024年2月29日至11月17日在Blind平台“Jobs & Referrals”频道收集的1.75万条帖子和2.71万条评论，去除认证信息的遮掩令模型聚焦文本质量。

**📈 对比分析**

对比方法基于预训练句子变换器的AUROC 0.681、准确率0.63；实验显示RAG工作流将弱请求成功率提升14%，而基本流程对强请求导致下降。

**⚠️ 局限性**

局限在于缺乏外部验证与真实转化率测量、模型预测可能与实际获得推荐不一致、平台演化与技术行业局限导致泛化性差。

---

## 111. Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs

**arXiv ID:** 2601.11019 | [PDF](https://arxiv.org/pdf/2601.11019v1)

**作者:** Xinwei Wu `[一作]` (Tianjin University), Kaifu Zhang `[通讯]` (Alibaba International Digital Commerce)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用稀疏自编码器(SAE)识别并验证了一组“翻译启动”特征，证明其对翻译质量和真实性的因果影响，并将该特征作为内在信号用于数据选择，从而显著提升微调效率并降低hallucination。

**💡 创新点**

提出了三阶段框架（高频召回 → 特征影响向量 → PCA一致性筛选）以系统定位任务特定特征，并首次将SAE与因果干预结合，展示了可直接转化为高效数据筛选策略的能力。

**🔧 技术方法**

技术手段包括：稀疏自编码器（Google公开的SAE）、JumpReLU激活、特征影响向量计算、PCA一致性度量、因果干预（放大/消除特征）、COMET与LLM-as-Judge评估hallucination。

**📊 数据集**

数据集主要为WMT24++（en-zh、en-ar、en-ru、en-ja 共约1k句），以及100k条en-zh平行句用于微调；实验模型包括Gemma-2-2B-IT、Gemma-2-9B-IT、LLaMA-3.1-1B-IT、LLaMA-3.2-8B-IT。

**📈 对比分析**

与随机选取、高质量样本、最高损失样本等策略比较。机制选取在Gemma系列模型中实现与完整数据相当的COMET并将hallucination率降至最低；在LLaMA跨家族迁移时无显著提升。实验显示在少量数据下（仅20%–50%）即可超越全数据集的性能，验证了高数据效率。

**⚠️ 局限性**

局限性：所识别的特征及其转移性仅在同一模型家族内显著，对不同架构（如Gemma→LLaMA）不具备可迁移性；研究聚焦翻译任务，其他任务的可推广性未知；特征阈值和PCA阈值需手工设置；因果干预的计算成本相对较高。

---

## 112. Pruning as Evolution: Emergent Sparsity Through Selection Dynamics in Neural Networks

**arXiv ID:** 2601.10765 | [PDF](https://arxiv.org/pdf/2601.10765v1)

**作者:** Zubair Shah `[一作]` (Hamad Bin Khalifa University), Noaman Khan `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将神经网络的参数组视为进化种群，提出基于种群选择的无显式剪枝过程，并在训练中联合更新参数和种群质量，最后按种群质量阈值进行剪枝。

**💡 创新点**

创新点在于把稀疏化视作自然的进化灭绝过程：不需要预设阈值或剪枝计划，稀疏度由梯度贡献驱动的相对适应度自发产生，并给出了理论分析与可实现的演化动力学。

**🔧 技术方法**

使用复制器、归一化增长和选择‑突变三种演化动力学，利用梯度贡献 |∂ℒ/∂p_i|（加正则化）作为适应度，结合标准SGD训练实现参数和种群质量的协同演化。

**📊 数据集**

在MNIST数据集上，使用784–512–256–10的多层感知机，对隐藏层的512+256个神经元进行实验。

**📈 对比分析**

对比三种演化动力学，在35%–50%稀疏率下评估测试准确率：基线≈98%；35%时≈95.5%；40%≈92.9%；45%≈92.1%；50%≈88.3–88.6%；变异动力学在最高稀疏率下略优。结果表明演化过程能自发产生可接受的稀疏-精度折衷。

**⚠️ 局限性**

局限性：仅在简单的MLP+MNIST上验证，未对深度网络、卷积/Transformer或大规模数据集进行测试；种群质量逼近零时数值不稳定；梯度异质性和层级学习率未作深度调优；未考虑滤波/块级结构化剪枝或多目标适应度。

---

## 113. Streaming Stochastic Submodular Maximization with On-Demand User Requests

**arXiv ID:** 2601.10901 | [PDF](https://arxiv.org/pdf/2601.10901v1)

**作者:** Honglian Wang `[一作]` (KTH Royal Institute of Technology), Aristides Gionis `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 19730 | [OpenAlex ID](https://openalex.org/A5022164041)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了面向新闻推荐的流式随机子模最大化问题，并给出了基于上限估计的几种低内存算法；

**💡 创新点**

通过将问题归约到划分马尔可夫约束下的子模最大化，首次实现了在不知访问次数的情形下的可竞争算法；

**🔧 技术方法**

采用子模函数与分区马尔可夫约束的在线流式贪婪算法、阈值替换策略和并行猜测技巧；

**📊 数据集**

在六个真实数据集（如MovieLens、Amazon、LastFM等）上进行实验验证；

**📈 对比分析**

与传统离线贪婪、Sieve、基于匹配的算法等基线对比，实验显示所提算法在覆盖率、内存和响应时间上均优于基线；

**⚠️ 局限性**

对访问次数的上限估计要求较高，且在估计误差较大时竞争比可能下降，且对极端稀疏场景下的性能尚未充分验证。

---

## 114. Can Instructed Retrieval Models Really Support Exploration?

**arXiv ID:** 2601.10936 | [PDF](https://arxiv.org/pdf/2601.10936v1)

**作者:** Piyush Maheshwari `[一作]` (University of Massachusetts), Hamed Zamani `[通讯]` (University of Massachusetts)

**通讯引用:** 3773 | [OpenAlex ID](https://openalex.org/A5101457713)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对指令式检索模型在基于种子文档的探索性搜索中的表现进行了评估，重点关注多维度相关性和指令遵循能力。

**💡 创新点**

创新点在于首次将专家标注的多面向探索性检索数据集CSFCube用于同时衡量排名相关性与指令遵循，并探讨不同指令变体对模型敏感性的影响。

**🔧 技术方法**

采用了指令式检索模型（如GritLM‑7B、gpt‑4o_prp）以及对大型语言模型的两种提示方式（Pairwise Ranking Prompting与单点提示），并将它们作为重排名器与基线检索器组合使用。

**📊 数据集**

使用的主要数据集是CSFCube，包含50个多面向查询及约800k计算机科学文献，且每个查询在不同维度（Background、Method、Result）下都有相关性标注。

**📈 对比分析**

通过与基线稠密检索模型（Specter2、SciNCL、otAspire）以及人工基准进行对比，发现指令式检索在NDCG@20上有显著提升，但在p‑MRR（指令遵循）上表现不稳定，部分模型甚至出现反直觉的指令响应。

**⚠️ 局限性**

局限性包括：当前指令式检索模型对指令细微差异不敏感，且在长时间回忆导向的探索任务中缺乏足够的指令响应灵活性，整体性能仍落后于人工评估。

---

## 115. Japanese AI Agent System on Human Papillomavirus Vaccination: System Design

**arXiv ID:** 2601.10718 | [PDF](https://arxiv.org/pdf/2601.10718v1)

**作者:** Junyu Liu `[一作]` (Kyoto University), Tomoki Aoyama `[通讯]` (Kyoto University)

**通讯引用:** 6452 | [OpenAlex ID](https://openalex.org/A5047803577)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并评估了一套双向人工智能代理系统，既能通过对话界面向公众提供经验证的 HPV 疫苗信息，又能自动生成针对医疗机构的社会媒体和聊天数据分析报告。

**💡 创新点**

创新点在于将 Retrieval‑Augmented Generation (RAG) 与 ReAct 多工具协同架构结合，允许单一控制器在对话中动态选择并整合学术论文、政府文件、新闻报道和社交媒体等多源信息；同时构建了端到端的自动报告生成流程，提供可视化和可引用的多语言报告。

**🔧 技术方法**

核心技术包括：
- LLM（如 LlamaIndex/Claude）做文本生成和评估;
- 向量数据库（Qdrant）存储多源知识并实现语义检索;
- ReAct 代理架构实现多工具调用与双层引用校验;
- 机器学习工具进行情绪分析、立场分类、主题建模和误信息检测；
- PDF 生成与双语排版。

**📊 数据集**

使用了 139,939 篇文档，分别来自 PubMed 论文、WHO 与日本厚生劳动省官方文件、日语新闻与推特/X 社交媒体帖子，以及 139 条公开聊天记录；这些数据被嵌入 2048 维向量并存入 Qdrant。

**📈 对比分析**

评估方法：
- 对话质量：单轮和多轮评估，使用 LLM 评判员与人工专家交叉验证，得分均在 4.8‑4.98/5，显示高相关性、准确性和专业性。
- 报告质量：对四个时间段生成的报告分别评估完整性、正确性、帮助性以及引用有效性，所有维度均在 4.0‑5.0/5，引用正确率平均 4.33/5（新闻）和 4.08/5（论文）。

**⚠️ 局限性**

局限性包括：
- 社交媒体样本仅来自日本推特，可能低估老年人和非数字用户的观点；
- LLM 评估可能存在偏差，尤其在工具选择上有多种有效方案时；
- 模拟用户数据规模有限，缺乏真实部署和长期用户反馈；
- 仅在日语/英语双语环境下验证，跨国、跨语言迁移需进一步验证。

---

## 116. A Differential Geometry and Algebraic Topology Based Public-Key Cryptographic Algorithm in Presence of Quantum Adversaries

**arXiv ID:** 2601.10883 | [PDF](https://arxiv.org/pdf/2601.10883v1)

**作者:** Andrea Rondelli `[一作]` `[通讯]`, Andrea Rondelli

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于Calabi‑Yau流形切线束的非对称公钥加密算法Z‑Sigil，并证明其可逆性与完整性，同时给出量子安全性分析。

**💡 创新点**

① 采用连续几何/拓扑结构（Calabi‑Yau流形、切线束群oid）作为密钥空间；② 通过块级串行加密抑制量子并行攻击；③ 结合谱正则化、GUE随机矩阵等高级数学工具增强安全性。

**🔧 技术方法**

微分几何、复Kähler/Calabi‑Yau 理论、切线束与群oid 运算、Hilbert–Schmidt 算子与谱 ζ‑函数正则化、随机 GUE 矩阵、量子查询模型与 Grover 下界。

**📊 数据集**

本研究为理论工作，没有使用公开数据集；未来计划在数值模拟中选取不同维度的 Calabi‑Yau 参数进行实验。

**📈 对比分析**

通过与 RSA 的结构类比以及理论分析给出 Grover 攻击下的指数搜索成本，说明在量子时代仍具安全性；未提供实测运行时间或加密速度基准。

**⚠️ 局限性**

缺乏实现与实验评估，计算复杂度与实际性能未知；参数选择需在数值模拟中确定，尚未找到最优操作点；对抗攻击的完整性证明仍待进一步细化。

---

## 117. SecMLOps: A Comprehensive Framework for Integrating Security Throughout the MLOps Lifecycle

**arXiv ID:** 2601.10848 | [PDF](https://arxiv.org/pdf/2601.10848v1)

**作者:** Xinrui Zhang `[一作]` (Carleton University), Rongxing Lu `[通讯]` (Queen's University)

**通讯引用:** 28750 | [OpenAlex ID](https://openalex.org/A5070447777)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e0540dec-d77f-42db-94ae-d039248f6393` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SecMLOps框架，将安全措施嵌入完整的MLOps生命周期，并在基于VLPD的行人检测系统中进行了实证验证。

**💡 创新点**

创新点在于基于PTPGC模型构建了完整的安全角色与阶段性活动，系统化映射ML特有威胁并给出针对性对策，首次在案例中演示其可行性与效果。

**🔧 技术方法**

使用了DevSecOps实践、STRIDE威胁建模、CI/CD自动化、安全训练技术（差分隐私、联邦学习）、对抗训练与模型蒸馏、异常检测与实时监控等多种技术组合。

**📊 数据集**

采用CityPersons（CityScapes派生）数据集进行行人检测实验。

**📈 对比分析**

通过对比无防御基线模型，利用DP、FGSM、DF及其组合攻击，并用log‑average miss rate（laMR）衡量性能，SecMLOps在攻击下显著降低laMR，仅在正常情况下略有性能损失。

**⚠️ 局限性**

局限性包括仅在单一模型与数据集上评估，攻击场景相对有限，缺乏大规模部署、跨任务验证以及成本与资源消耗的系统评估。

---

## 118. "I'm Constantly Getting Comments Like, 'Oh, You're Blind. You're Like the Only Woman That I Stand a Chance With.'": A Study of Blind TikTokers' Intersectional Experiences of Gender and Sexuality

**arXiv ID:** 2601.10957 | [PDF](https://arxiv.org/pdf/2601.10957v1)

**作者:** Yao Lyu `[一作]` (University of Michigan), John M. Carroll `[通讯]` (Pennsylvania State University)

**通讯引用:** 36655 | [OpenAlex ID](https://openalex.org/A5054610664)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究盲人女性及 LGBTQ+ TikTok 创作者的交叉边缘化经历，并通过访谈揭示平台的社会技术影响。

**💡 创新点**

创新点在于将交叉体验视为基础设施化，提出“基础设施工作”概念来解释平台设计导致的多维边缘化。

**🔧 技术方法**

使用主题分析与基础设施理论相结合的方法来整理和解释访谈数据。

**📊 数据集**

使用 41 位盲人女性/ LGBTQ+ TikTok 用户的访谈数据，覆盖 18-62 岁、不同视力状态与教育背景。

**📈 对比分析**

以主题对比方式评估不同边缘化层面，发现算法和设计对身份表达的压制与强化，未采用传统机器学习性能指标。

**⚠️ 局限性**

局限在于样本仅限于自我标识为盲人的用户，且研究团队非盲人/ LGBTQ+，可能影响对多重身份细微差异的把握。

---

## 119. Efficient Protein Optimization via Structure-aware Hamiltonian Dynamics

**arXiv ID:** 2601.11012 | [PDF](https://arxiv.org/pdf/2601.11012v1)

**作者:** Jiahao Wang `[一作]` (Shanghai Jiao Tong University), Shuangjia Zheng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 4671 | [OpenAlex ID](https://openalex.org/A5075817762)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建一种结合Hamiltonian动力学与结构感知的贝叶斯优化框架（HADES）用于蛋白质定向进化。

**💡 创新点**

创新点在于：①用Hamiltonian Monte Carlo在连续序列空间高效跳跃采样并加入虚拟壁垒保证一维约束；②采用两阶段编码器-解码器结构先学习结构变异（RMSD）再学习功能，形成平滑的结构-功能潜在空间；③利用模型集成不确定性进行UCB选择，兼顾探索与利用。

**🔧 技术方法**

技术包括：贝叶斯优化、Hamiltonian动力学（Leapfrog、Metropolis）、高斯过程不确定性估计、ESMFold结构预测、编码器-解码器网络、UCB采样和虚拟壁垒实现。

**📊 数据集**

使用两大公开组合突变数据集：GB1（约16万条样本）和PhoQ（约14万条样本）。

**📈 对比分析**

与ESM‑zs、BO、CMA‑ES、AdaLead、PEX、EvoPlay及其Langevin版本对比，HADES在累计最大适应度、平均适应度和fDiv上均为最佳（GB1全部跑到最优，PhoQ显著超越对手）。

**⚠️ 局限性**

局限性包括：依赖ESMFold的结构预测精度；Hamiltonian采样与虚拟壁垒实现复杂且计算开销大；在极度稀疏的高适应度区域仍可能收敛慢；多目标优化尚未覆盖。

---

## 120. RidgeWalker: Perfectly Pipelined Graph Random Walks on FPGAs

**arXiv ID:** 2601.11057 | [PDF](https://arxiv.org/pdf/2601.11057v1)

**作者:** Hongshi Tan `[一作]` (National University of Singapore), Bingsheng He `[通讯]` (National University of Singapore)

**通讯引用:** 20655 | [OpenAlex ID](https://openalex.org/A5039946576)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了基于FPGA的RidgeWalker加速器，利用Markov性质将随机游走拆分为无状态细粒度任务，支持异步流水线和零气泡调度，显著提升随机图遍历吞吐量；

**💡 创新点**

创新点在于：①利用无状态任务拆分实现完全异步流水线，打破传统的顺序调度瓶颈；②基于排队论的反馈驱动调度器实现零气泡、动态负载平衡；③结合HBM通道独立访问和128个非阻塞请求的异步访问引擎，实现内存访问延迟彻底隐藏；

**🔧 技术方法**

采用CSR图存储、异步流水线架构、零气泡调度器（排队论驱动）、AXI‑Stream通信、HBM通道并行、FPGA可重配置架构、ThundeRiNG随机数生成器；

**📊 数据集**

使用多种真实世界图数据集（规模从数十万到数百万顶点的稀疏图），并在多种GRW算法（URW、PPR、DeepWalk、Node2Vec、MetaPath）上进行实验；

**📈 对比分析**

与最先进的FPGA实现（FastRW、LightRW）以及GPU实现对比。实验显示，RidgeWalker在FPGA上平均提升7.0×，最高71.0×；在GPU上平均提升8.1×，最高22.9×；随机访问带宽利用率高达88%；

**⚠️ 局限性**

局限性包括：①对大规模高带宽HBM的FPGA平台依赖；②需要对不同采样算法重新编译/映射，更新成本高；③主要针对CSR格式的静态图，动态图更新支持有限；④调度器的延迟受设计参数影响，极端不均衡工作负载下仍可能出现轻微瓶颈。

---

## 121. EncodeRec: An Embedding Backbone for Recommendation Systems

**arXiv ID:** 2601.10837 | [PDF](https://arxiv.org/pdf/2601.10837v1)

**作者:** Guy Hadad `[一作]` (Ben-Gurion University of the Negev), Bracha Shapira `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 14966 | [OpenAlex ID](https://openalex.org/A5086920790)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出EncodeRec，一种针对推荐系统优化的嵌入骨干网络；

**💡 创新点**

创新点在于利用标题与描述的对比学习，将结构化元数据对齐为高分辨率、面向领域的嵌入；

**🔧 技术方法**

主要技术包括冻结预训练语言模型、共享编码器、基于InfoNCE的元数据对比损失、L2归一化；

**📊 数据集**

使用Amazon 2023子数据集的商品标题、描述与属性，实验覆盖Beauty、Toys、Sports、Video Games、Baby Products等域；

**📈 对比分析**

在UniSRec序列推荐和TIGER生成式推荐的Recall@10和NDCG@10上，相比BERT、Sentence‑T5、BLaIR和通用EmbeddingGemma等基线，EncodeRec提升了5–26%（UniSRec）和4–9%（TIGER），并消除了Semantic ID冲突；

**⚠️ 局限性**

局限性包括未结合用户交互信号、未扩展到多模态特征、以及大规模模型实验受算力限制。

---

## 122. OpFML: Pipeline for ML-based Operational Forecasting

**arXiv ID:** 2601.11046 | [PDF](https://arxiv.org/pdf/2601.11046v1)

**作者:** Shahbaz Alvi `[一作]` (CMCC Foundation), Pasquale Schiano `[通讯]` (CMCC Foundation)

**通讯引用:** 276 | [OpenAlex ID](https://openalex.org/A5052668024)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个可配置、可适配的机器学习预测流水线，用于周期性预测每日火灾危险指数（FDI），并在南意大利与葡萄牙两地区进行了演示与验证。

**💡 创新点**

首次在操作层面实现了完整的端到端ML预测系统，采用TOML配置实现多数据源、变量预处理与模型切换的高度可定制化流水线。

**🔧 技术方法**

使用Python实现；核心算法为卷积长短时记忆网络（ConvLSTM）；数据存取通过CMCC DDS；预处理通过自定义函数库完成；流水线容器化后在Docker/Kubernetes上部署。

**📊 数据集**

训练数据来源于2009‑2021年的火灾记录和13个火灾预测因子（NDVI、LST、温度、湿度、降水、风速、DEM、坡度、道路/水道距离、人口密度、土地利用等）；预测使用MODIS、GFS、WRF_2km等实时遥感与气象预报。

**📈 对比分析**

通过将预测的FDI热度图与VIIRS及MODIS燃烧区产品的火灾事件进行空间对比，展示了两地区预测结果与实际火灾的相符性；未给出定量性能指标，性能评估计划在后续工作中开展。

**⚠️ 局限性**

模型泛化性在不同地区尚未充分验证；对日间FDI与实际火灾事件的相关性评估受火灾检测不确定性与随机性限制；缺乏完整的定量性能评价与数据缺失处理机制的细化。

---

## 123. Constant Metric Scaling in Riemannian Computation

**arXiv ID:** 2601.10992 | [PDF](https://arxiv.org/pdf/2601.10992v1)

**作者:** Kisung You `[一作]` (Baruch), Kisung You `[通讯]` (Baruch)

**通讯引用:** 873 | [OpenAlex ID](https://openalex.org/A5012261592)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

讨论并系统整理了在黎曼流形上对度量进行恒定缩放时，哪些几何量会变化、哪些保持不变，阐明了该操作对计算方法的影响。

**💡 创新点**

明确区分了度量缩放对测量量（范数、距离、体积、梯度大小）和几何结构（Levi‑Civita 连接、测地线、指数/对数映射、平行移动）的不同影响，提供了一个完整的参考框架。

**🔧 技术方法**

主要采用黎曼几何的基础理论（度量、连接、测地线、指数映射等）进行推导与证明，利用常数缩放的简单代数关系得到结果。

**📊 数据集**

未使用任何具体数据集；论文为理论性说明性工作。

**📈 对比分析**

未进行实验比较；作者通过解析推导说明在梯度下降等优化算法中，常数度量缩放等价于步长的全局缩放，性能影响仅体现在数值大小而非算法结构。

**⚠️ 局限性**

局限性在于仅讨论常数度量缩放；若缩放因子随点变化，连接、测地线等几何量会改变，需要进一步研究。

---

## 124. AVP-Pro: An Adaptive Multi-Modal Fusion and Contrastive Learning Approach for Comprehensive Two-Stage Antiviral Peptide Identification

**arXiv ID:** 2601.11028 | [PDF](https://arxiv.org/pdf/2601.11028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 125. Transient learning dynamics drive escape from sharp valleys in Stochastic Gradient Descent

**arXiv ID:** 2601.10962 | [PDF](https://arxiv.org/pdf/2601.10962v1)

**作者:** Ning Yang `[一作]` (Peking University), Yuhai Tu `[通讯]` (Flatiron Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析随机梯度下降（SGD）在训练初期的探索性动力学，构建二维阱模型解释其偏好平坦最小值的机制，并与实际神经网络实验相结合。

**💡 创新点**

提出“临时冻结”机制：SGD中的各向异性噪声产生有效势能，促使网络在早期频繁跨阱跳跃并倾向于平坦阱；随着能量壁增大，探索停止，噪声强度决定冻结时间，从而决定最终落入哪个阱。

**🔧 技术方法**

使用梯度噪声模拟（Langevin方程）、Kramers逃逸率、准稳态分析、Hessian谱和Jaccard相似度等理论工具；在MNIST、CIFAR‑10上训练多层感知机和卷积网络，采集训练/测试误差、平坦度指标和参数空间投影。

**📊 数据集**

主要使用MNIST（1000样本子集）进行实验，补充材料中使用CIFAR‑10作为验证。

**📈 对比分析**

对不同学习率和批大小组合进行多次独立训练，比较训练损失、测试准确率和平坦度。结果显示：噪声强度越大，冻结时间越长，最终得到的解平坦度更高、测试准确率更好；理论预测的平坦阱占优概率与实验相符。

**⚠️ 局限性**

局限性：实验仅在简易网络和有限数据集上验证；二维阱模型在高维参数空间的可推广性尚未完全证明；假设噪声协方差与海森矩阵成正比，实际情况可能更复杂；仅关注早期探索阶段，长期稳定性和泛化机制仍需进一步研究。

---

## 126. Predicting Biased Human Decision-Making with Large Language Models in Conversational Settings

**arXiv ID:** 2601.11049 | [PDF](https://arxiv.org/pdf/2601.11049v1)

**作者:** Stephen Pilli `[一作]` (University), Vivek Nallur `[通讯]` (University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在聊天机器人对话中设计六种经典决策任务，探究大型语言模型（LLM）是否能预测并再现人类在对话情境下的偏见决策（框架效应、现状偏差），并评估对话复杂度对认知负荷及偏见的影响。

**💡 创新点**

创新点在于：①首次系统性检验LLM在对话环境中对人类偏见的再现与预测能力；②揭示对话复杂度可显著放大框架效应但对现状偏差影响有限；③比较不同LLM（GPT‑4、GPT‑5、开源模型）及三种人类化提示水平的表现；④通过消融与扰动实验剖析对话记忆与算术成分对模型预测的作用。

**🔧 技术方法**

使用技术包括：大型语言模型推理（GPT‑4.1、GPT‑5、GPT‑4.1‑mini、GPT‑5‑mini、GPT‑OSS‑120B、Llama4、Qwen3）；三层人类化提示（HL1‑HL3）；NASA‑TLX 认知负荷测量；统计方法（Cohen's h、效应量、z‑score、Spearman相关、分类准确率）；以及对话重构与模型预测的自动化脚本。

**📊 数据集**

数据集为 1648 名 Prolific 参与者完成的六个决策任务（Risky‑Choice、Attribute、Goal 框架效应；Budget Allocation、Investment、College Jobs 现状偏差），并提供两种先导对话（Simple 与 Complex）。LLM 在预测时仅使用这些对话文本与人口统计信息，未使用任何专门训练的偏见数据集。

**📈 对比分析**

比较方法：将 LLM 的个体预测准确率与人类选择进行对比；计算样本层偏见效应（Cohen’s h）与人类的差异；计算人类与 LLM 在对话复杂度下效应变化（z‑score）的 Spearman 相关。结果显示 GPT‑4.1 在偏见再现上达到约 75% 的一致率，且在对话复杂度影响下相关性最高（ρ≈0.77）；GPT‑5 与大多数开源模型表现显著逊色；加入对话上下文显著提升预测准确率，尤其对 Goal‑Framing 与 Investment 决策。

**⚠️ 局限性**

局限性包括：仅检验框架效应与现状偏差，未验证对其他偏见的普适性；对话复杂度作为认知负荷的代理可能遗漏情绪、疲劳等因素；使用自我报告的 NASA‑TLX 而非实时生理指标；提示过度拟合导致 HL3 在某些任务中产生过度偏见；样本来源单一，可能缺乏跨文化代表性；LLM 可能仅通过统计关联再现偏见，缺乏对底层认知机制的真正模拟。

---

## 127. IDDR-NGP: Incorporating Detectors for Distractor Removal with Instant Neural Radiance Field

**arXiv ID:** 2601.11030 | [PDF](https://arxiv.org/pdf/2601.11030v1)

**作者:** Xianliang Huang `[一作]` (Fudan University), Shuigeng Zhou `[通讯]` (Fudan University)

**通讯引用:** 10965 | [OpenAlex ID](https://openalex.org/A5017862559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Instant‑NGP与2D目标检测器的端到端3D场景干扰物去除方法（IDDR‑NGP），可同时处理雪、碎片、叶子等多类型干扰物。

**💡 创新点**

创新点包括①首次将2D检测器与Instant‑NGP结合，直接在隐式3D表示上去除干扰物；②引入多分辨率Hash编码、LPIPS感知损失和多视角补偿损失，提升多视角一致性与细节恢复；③构建新型带干扰物的多视角数据集，并提供完整标注。

**🔧 技术方法**

使用技术包括Instant Neural Radiance Field（Instant‑NGP）、多分辨率Hash编码、YOLOv5/FCOS检测器、LPIPS感知损失、Multi‑View Compensation Loss（MVCL）以及PyTorch实现的训练框架。

**📊 数据集**

使用的数据集包括：DTU、LLFF合成场景（加入雪、叶、碎片等干扰物）和真实场景（Apple iPhone 12拍摄的20–30张含/不含碎片的图像），并自行构建标注丰富的干扰物数据集。

**📈 对比分析**

与现有雪去除方法（Weather removal、SnowFormer、HCDW‑Net）、IDDR‑Inpainting、IDDR‑NeRF等进行对比；实验显示IDDR‑NGP在PSNR/SSIM/LPIPS上均优于对比方法，尤其在多视角一致性和多类型干扰物去除方面表现突出，PSNR提升约10–20%。

**⚠️ 局限性**

局限性在于：①只能去除在多视角中出现的可见干扰物，无法处理所有视角均出现的静态大面积干扰物；②对检测器的准确性高度依赖，检测误差会直接影响去除效果；③目前仍需依赖检测器，缺乏完全无检测器的自动化去除方案。

---

## 128. PEMNet: Towards Autonomous and Enhanced Environment-Aware Mobile Networks

**arXiv ID:** 2601.11025 | [PDF](https://arxiv.org/pdf/2601.11025v1)

**作者:** Lei Li `[一作]` (Chinese University of Hong Kong), Tsung-Hui Chang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 8742 | [OpenAlex ID](https://openalex.org/A5064271996)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于标准测量的感知嵌入映射（PEM）框架，联合表征基站覆盖区域内的无线传播特性（如角度功率谱APS、时延功率谱DPS）与网格级流量需求，实现对环境的细粒度感知并支持自适应网络优化。

**💡 创新点**

创新点在于①将信道与流量信息共嵌入同一空间网格，弥补单一信道或流量映射的局限；②利用现有MR、MDT、RSRP等标准测量数据，避免额外硬件与高昂测量成本；③在PEM之上构建PEMNet，直接驱动PHY、MAC、网络层的协同优化。

**🔧 技术方法**

技术主要包括：基于多波束RSRP的信道特征恢复（利用稀疏重构、NeRF等）、图信号/空间插值方法补全缺失网格、基于结构化预测（GPR、ConvLSTM、GNN）构建流量时空模型、以及基于PEM的多用户协调波束成形与接收波束设计。

**📊 数据集**

实验数据主要来自：①仿真产生的4路径多径信道和高斯混合模型的流量分布；②真实网络中收集的MR/MDT RSRP、UE连接日志、调度与QoS统计；未使用公开大规模数据集，而是结合标准测量和仿真场景。

**📈 对比分析**

与传统完整CSI协调波束成形、无协调单元波束（PCBF）以及仅基于宏观信道映射的方法相比，PEMNet在有效总率、干扰抑制和资源利用率上取得显著提升（如ESR提升10–30%）；同时显著降低了信道估计开销和互联负荷。

**⚠️ 局限性**

局限性包括：①PEM依赖MR/MDT的覆盖和定位精度，稀疏测量会导致信道/流量信息缺失；②仅提供大尺度信道统计，无法捕获细粒度小尺度衰落；③需要持续自适应更新，受环境变化与数据刷新频率限制；④对多模态融合与跨站协作等高级功能尚未实现。

---

## 129. Modeling Multi-Party Interaction in Couples Therapy: A Multi-Agent Simulation Approach

**arXiv ID:** 2601.10970 | [PDF](https://arxiv.org/pdf/2601.10970v1)

**作者:** Canwen Wang `[一作]` (Carnegie Mellon University), Haiyi Zhu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3243 | [OpenAlex ID](https://openalex.org/A5051842323)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一个多模态、多智能体模拟系统，用于训练夫妻治疗师，让受训者在低风险环境中体验和练习处理配偶间的需求‑撤退循环及六个治疗阶段；

**💡 创新点**

创新点在于首次将需求‑撤退动态与六个核心治疗阶段融入多智能体对话，并通过阶段控制器和情绪反馈实现逼真、非线性的多方交互；

**🔧 技术方法**

技术包括大语言模型（LLM）驱动的双人虚拟患者、阶段基准控制器、情绪识别与多模态输出（文本+语音）、React前端与Flask+WebSocket后端实现实时交互；

**📊 数据集**

数据集为七个公开的夫妻治疗对话记录（约1,621个发言）用于验证阶段与循环出现频率，并在用户评估中使用了两套基于真实案例的情境（抑郁与不忠）；

**📈 对比分析**

通过21名美国持证夫妻治疗师的双条件对照实验（实验系统 vs 基线系统），使用分层GLS回归与配对t检验验证，实验系统在阶段识别、需求‑撤退识别、代理响应真实感和整体真实感上均显著提升（p<0.001）；

**⚠️ 局限性**

局限性包括仅覆盖15分钟短时会话，未提供系统化反馈或长期案例追踪，模型对更深层关系动力学与多周期治疗的适应性尚未验证。

---

## 130. CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs

**arXiv ID:** 2601.11047 | [PDF](https://arxiv.org/pdf/2601.11047v1)

**作者:** Yuanxiang Liu `[一作]` (Zhejiang University), Wen Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 16439 | [OpenAlex ID](https://openalex.org/A5100444425)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出CoG框架，实现无训练的KG问答推理，利用蓝图引导和失效自适应纠错；

**💡 创新点**

创新点在于融合关系蓝图指导的全局结构约束与失效感知的反向修正机制，解决传统方法的误差级联和结构失配问题；

**🔧 技术方法**

技术包括离线关系蓝图抽取与检索、在线蓝图适配与软约束重排序、LLM驱动的失效诊断与局部回溯、以及基于句子编码的检索索引；

**📊 数据集**

使用WebQSP、CWQ、GrailQA三个多跳KGQA基准数据集；

**📈 对比分析**

与现有LLM和KG增强基线相比，CoG在Hits@1上分别在CWQ、WebQSP、GrailQA上提升约3-5%，在GPT‑4底层上达77.8/89.7，且在调用次数和token消耗上更优；

**⚠️ 局限性**

局限包括对KG完整性的依赖、蓝图库覆盖度不足时的指导失效、回溯过程可能带来的延迟以及蓝图未实现在线动态更新。

---

## 131. WenetSpeech-Wu: Datasets, Benchmarks, and Models for a Unified Chinese Wu Dialect Speech Processing Ecosystem

**arXiv ID:** 2601.11027 | [PDF](https://arxiv.org/pdf/2601.11027v1)

**作者:** Chengyou Wang `[一作]` (Northwestern Polytechnical University), Lei Xie `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 9595 | [OpenAlex ID](https://openalex.org/A5066245750)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 8,000 小时的多维度吴语语料库 WenetSpeech-Wu，发布首个吴语评测基准 WenetSpeech-Wu-Bench，并训练了一系列强大的开源模型（ASR、TTS、统一语音理解与指令式 TTS）。

**💡 创新点**

提供首个大规模、跨八大吴方言、含转写、翻译、情感、属性等多维度标签的数据集与标准化基准，搭建完整的吴语语音处理生态，显著提升各任务性能。

**🔧 技术方法**

自动化数据采集与 VAD、DNSMOS+SNR 筛选、双模型 ROVER 融合转写、Lexicon+Qwen3 翻译、SenseVoice/Emo2Vec+Qwen3+DeepSeek-R1+Gemini 评估情感、语音特征提取、LoRA 参数高效微调、CPT+SFT+SS-SFT 多阶段 TTS 训练、Step-Audio2 框架统一语音理解等技术。

**📊 数据集**

WenetSpeech-Wu（约 8,000 小时）与其基准测试集（ASR 9.75h、AST 4.4h、情感/属性 1,000 样本、TTS 242 句、指令式 TTS 20+ 样本），以及公开的 MagicData‑Shanghai 等小数据集。

**📈 对比分析**

与公开开源和商业基准（Paraformer、SenseVoice、Whisper、Qwen3‑ASR、Tencent、Gemini、CosyVoice2 等）在 CER、BLEU、分类准确率、IMOS/SMOS/AMOS 等指标上对比，WenetSpeech‑Wu 模型在 ASR CER 降至约 8%/7%，AST BLEU 提升数分，情感/属性识别准确率提升显著，TTS 在 CER、IMOS、SMOS、AMOS 上均显著优于基线。

**⚠️ 局限性**

方言与域分布不均衡、自动注释可能带噪声、模型基线非最优、未覆盖所有吴方言细变种，未来需平衡数据、加强人工标注并探索更专业化的模型与训练策略。

---

## 132. Crane Lowering Guidance Using a Attachable Camera Module for Driver Vision Support

**arXiv ID:** 2601.11026 | [PDF](https://arxiv.org/pdf/2601.11026v1)

**作者:** HyoJae Kang `[一作]` (Hanyang University), Min-Sung Kang `[通讯]` (Hanyang University)

**通讯引用:** 1999 | [OpenAlex ID](https://openalex.org/A5006778364)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发并验证了一种可吸附在吊装物体侧面的摄像机模块，能够实时采集垂直向下的图像，并利用激光指示器与图像处理算法绘制落点指导线，为塔吊操作员提供下放位置参考。

**💡 创新点**

创新点在于：①将摄像机模块通过吸盘直接附着在吊装物体侧面，实现对吊载“盲区”图像的即时获取；②结合激光指示点与霍夫变换的线条识别，利用图像畸变预测落点；③支持多模块并行工作，从不同角度提供落点指引。

**🔧 技术方法**

技术手段包括：单板计算机（Raspberry Pi 4B）+ Arducam摄像头+ 532 nm激光指针；图像预处理（灰度化、Gaussian模糊）+ Canny边缘检测 + Hough变换提取水平/对角线；HSV颜色空间+形态学过滤检测激光点；WiFi实时传输到主机PC；硬件使用3D打印轻量化框架和吸盘固定。

**📊 数据集**

实验数据来源于室内测试场景：在墙面上投射激光点的框架物体上安装模块，记录实时视频图像并验证指导线绘制；未使用公开数据集。

**📈 对比分析**

实验验证：在1 m间隔的墙面上，模块可在5 m距离内稳定接收视频并实时展示指导线；虽未与现有塔吊视觉系统进行定量对比，但显示在室内环境下功能正常，表明在5 m以内的吊装高度可提供有效落点指导。

**⚠️ 局限性**

局限性：①实验未在真实吊装现场验证，缺乏风光照等外部环境影响；②激光指针与物体表面水平距离、吸盘附着误差会导致落点误差；③仅适用于具有明显边缘的矩形物体，对圆柱等无边缘形状不适用；④多模块数量受主机接收能力限制。

---

## 133. From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models

**arXiv ID:** 2601.11020 | [PDF](https://arxiv.org/pdf/2601.11020v1)

**作者:** Youmi Ma `[一作]` (Institute of Science Tokyo), Naoaki Okazaki `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 3953 | [OpenAlex ID](https://openalex.org/A5066940046)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在LLM中遮蔽检索头并利用对比学习训练模型，提升其长上下文处理能力

**💡 创新点**

将检索头遮蔽产生的对比样本作为Direct Preference Optimization（DPO）的训练信号，验证并强化检索头在长上下文中的关键作用

**🔧 技术方法**

使用Needle‑in‑a‑Haystack检索头识别、检索头遮蔽、Contrastive Response Generation以及DPO等技术

**📊 数据集**

主要使用公开对话数据集LMSYS‑Chat‑1M（并在WildChat、Guru‑RL‑92K上验证）

**📈 对比分析**

与Smaller‑Model、Win‑Lose‑Pair、Non‑Retrieval‑Mask、Random‑Mask等基线比较，Llama‑3.1在128K上下文提升约+2.28分，Cite +70%、Re‑rank +32%；Qwen3也有提升，Olmo‑3提升有限

**⚠️ 局限性**

仅对检索头集中式模型有效，对检索头分布广泛的模型效果有限；目前仅在≤8B模型验证，未评估更大规模模型

---

## 134. Backdoor Attacks on Multi-modal Contrastive Learning

**arXiv ID:** 2601.11006 | [PDF](https://arxiv.org/pdf/2601.11006v1)

**作者:** Simi D Kuniyilh `[一作]`, Rita Machacy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对比研究了多模态对比学习中的后门攻击及其防御方法，系统梳理了攻击维度、触发器类型、目标领域与攻击目标，并提出了多模态后门攻击的新视角。

**💡 创新点**

创新点在于提供了从攻击阶段、触发器类型、目标域和攻击目标四个维度构建的系统分类框架，深入探讨了对比学习在联邦、图与多模态场景下的脆弱性，并比较了多种后门攻击与防御的效果。

**🔧 技术方法**

使用了信息对比损失、二层触发器优化、联邦聚合（BAGEL）、结构扰动、图像文本对齐等对比学习框架与攻击实现技术，以及 CleanCLIP 与 CleanerCLIP 等基于相似度过滤与语义对抗增强的防御方法。

**📊 数据集**

主要在公开数据集上评估，如 ImageNet、COCO、WIDER、CIFAR、图数据集和 CLIP 语料，结合多模态图像‑文本配对进行实验。

**📈 对比分析**

通过对比攻击成功率（ASR）、干净准确率下降、跨任务迁移等指标，发现攻击可达 90%+ 的成功率且对干净性能影响微乎其微；防御方法能将成功率降至随机水平，同时保持干净任务的性能。

**⚠️ 局限性**

局限性包括：防御依赖可检测的异常特征，易被自适应攻击规避；CleanCLIP 对触发器特征分布假设较强；CleanerCLIP 的对抗生成与语义分解增加计算成本，并在语义结构弱的域效果有限。

---

## 135. NAACL: Noise-AwAre Verbal Confidence Calibration for LLMs in RAG Systems

**arXiv ID:** 2601.11004 | [PDF](https://arxiv.org/pdf/2601.11004v1)

**作者:** Jiayu Liu `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10457 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究检索增强生成（RAG）环境下大语言模型的置信度校准问题，并提出基于噪声感知规则的自训练框架 NAACL。

**💡 创新点**

通过定义三条噪声感知校准规则并构造带噪声检索数据，利用自监督生成高质量推理轨迹进行细粒度置信度校准。

**🔧 技术方法**

采用 RAG、LLM 微调（LoRA）、自监督数据生成与过滤、规则驱动的推理注释以及 ECE/AUROC 等评价指标。

**📊 数据集**

HotpotQA、Natural Questions、StrategyQA、Bamboogle 以及 Wiki 检索语料。

**📈 对比分析**

与 Vanilla、CoT、Ensemble、Label‑only SFT 等基线对比，平均 ECE 降低约10–11%，AUROC 明显提升，单推理通道即可达到最佳效果。

**⚠️ 局限性**

仅在 7B–8B 开源模型上验证，噪声合成可能不完全匹配真实检索误差；实验仅覆盖短答题场景，难以直接推广到长文本或更大模型。

---

## 136. Exact Constraint Enforcement in Physics-Informed Extreme Learning Machines using Null-Space Projection Framework

**arXiv ID:** 2601.10999 | [PDF](https://arxiv.org/pdf/2601.10999v1)

**作者:** Rishi Mishra `[一作]` (Indian Institute of Technology Madras), Ganapathy Krishnamurthi `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 5244 | [OpenAlex ID](https://openalex.org/A5048838919)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 Null‑Space Projected Physics‑Informed Extreme Learning Machine (NP‑PIELM)，通过在系数空间进行线性投影实现边界/初始条件的精确满足，将受限优化转化为无约束最小二乘求解；

**💡 创新点**

核心创新在于利用系数空间的零空间投影实现对线性约束的完全满足，消除了罚项、拉格朗日乘子以及问题特定的构造，且可应用于任意几何、混合边界和耦合系统，维度无关；

**🔧 技术方法**

技术手段包括随机隐藏层的极限学习机、系数空间的零空间投影与重新参数化、基于Tensor‑Product Bernstein基函数的空间-时间离散、最小二乘/伪逆求解以及稀疏/正则化技术；

**📊 数据集**

使用的“数据集”是若干经典的合成 PDE 基准问题：1D稳态对流扩散反应、1D时变对流扩散、二维 Poisson 方程（混合边界）、二维不规则花形域上的热传导、二维稳态 Stokes 流；

**📈 对比分析**

通过与传统使用罚项的 PIELM 进行对比，NP‑PIELM 在 1D‑2D 典型问题上实现了机器精度误差（10⁻¹⁵‑10⁻¹⁰），边界条件在离散点上完全满足，计算时间仅需 0.07–19 秒，展示了高效的单次训练和优越的精度；

**⚠️ 局限性**

局限性包括仅针对线性 PDE，目前未扩展到非线性或逆问题；大规模系统的条件数分析不足；对极高维或极复杂几何的鲁棒性尚未充分验证；

---

## 137. Evaluating 21st-Century Competencies in Postsecondary Curricula with Large Language Models: Performance Benchmarking and Reasoning-Based Prompting Strategies

**arXiv ID:** 2601.10983 | [PDF](https://arxiv.org/pdf/2601.10983v1)

**作者:** Zhen Xu `[一作]` (Columbia University), Renzhe Yu `[通讯]` (Columbia University)

**通讯引用:** 956 | [OpenAlex ID](https://openalex.org/A5047810054)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究利用7000余份高校课程文件与三种21世纪能力框架，构建对齐评估基准，评估零射击大型语言模型（LLM）在能力映射任务中的表现，并提出基于推理的Curricular CoT提示策略来提升模型的教育推理能力。

**💡 创新点**

创新点在于：①在多种课程文档类型（课程描述、教学目标、学习活动、教学日程、学习管理系统记录）与三大能力框架上构建大规模人工标注基准；②系统评估零射击LLM在不同粒度任务中的性能；③提出Curricular CoT结构化提示策略，通过先提取教学要素再进行能力判断，显著缓解模型过度推断和信息检索困难。

**🔧 技术方法**

主要技术包括大语言模型推理（GPT‑4o、GPT‑3.5‑turbo、Llama‑3‑70B、Llama‑3‑8B），多种提示方式（ZERO、DEF、CQA、CQ、QA、A），以及多层次分类粒度评估（5‑级到二分类）和回归分析以探究文档类型、框架和模型规模对性能的影响。

**📊 数据集**

数据集由Open Syllabus、两所美国高校通识目录和Canvas LMS日志抽取的五类课程文档共200份，每份与38项能力进行对齐，生成7600个标注对，涵盖O*NET、EU Key Competences、ESDC Success Model三种框架。

**📈 对比分析**

与零射击基线相比，Curricular CoT在二分类任务中平均提升≈2–4 %准确率，5级分类下可将过度推断的误差减小约10 %；开放权重模型在粗粒度任务中与专有模型相近，表明具备成本效益；但所有模型在细粒度任务中仍低于人类标注水平。

**⚠️ 局限性**

局限性包括：①基准样本量有限，未涵盖所有学科与多样化机构；②提示设计未经过交叉验证或迭代优化；③仅评估单一文档类型，未探究多源整合效果；④Curricular CoT依赖中间摘要质量，模型差异导致噪声；⑤缺乏公开共享的高质量教学文档数据集，限制了研究的可复现性与推广。

---

## 138. Non-uniformly Stable Common Independent Sets

**arXiv ID:** 2601.11153 | [PDF](https://arxiv.org/pdf/2601.11153v1)

**作者:** Naoyuki Kamiyama `[一作]` `[通讯]` (Institute of Mathematics for Industry Kyushu University), Naoyuki Kamiyama (Institute of Mathematics for Industry Kyushu University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文提出了在含有平等偏好的稳定匹配问题的图论推广——即在两个互不相交的集合上寻找满足非均匀稳定性的公共独立集合，并给出了多项式时间的判定与构造算法。

**💡 创新点**

创新点包括：① 将传统稳定匹配中的平衡稳定性（super‑stability 与 strong‑stability 的统一形式）推广到 matroid 约束下；② 通过构造“critical subset”“shortcut arc”等图论工具，首次实现了在两个 matroid 之间的非均匀稳定性判定；③ 通过将原问题拆分为基于 matroid 的子问题，并使用 matroid 同构与交换性质，保证算法的多项式时间复杂度。

**🔧 技术方法**

核心技术主要是 matroid 理论（基、环、闭包、基本环、同构与余余运算）、图论中的简单有向环与 shortcut arc 概念、以及对非均匀稳定性的定义所需的偏好等价/弱优先关系的利用；算法层面采用迭代构造、最大公共独立集与关键子集求解，并在每一步通过循环检测与环消去来维护稳定性。

**📊 数据集**

该工作为纯理论算法，不依赖任何实际数据集；所有证明与算法分析均在抽象的 matroid 与偏好关系上完成。

**📈 对比分析**

算法的运行时间为多项式（与元素集大小 |E| 成多项式关系），并在每一步通过子 matroid 的基础运算与最大公共独立集求解实现；与以往的稳定匹配（无平等、无 matroid 约束）以及仅考虑 super‑stability/strong‑stability 的结果相比，本算法在更广泛的约束与偏好模型下仍保持多项式时间，证明了其可行性。

**⚠️ 局限性**

局限性：① 仅处理两个 matroid 的公共独立集问题，无法直接扩展到多于两个 matroid 的情形；② 对于偏好中的平等（ties）仍需满足特定的偏好结构（例如转移闭包），若偏好更复杂可能不适用；③ 算法实现复杂，实际工程化难度较高。

---

## 139. Context-aware Graph Causality Inference for Few-Shot Molecular Property Prediction

**arXiv ID:** 2601.11135 | [PDF](https://arxiv.org/pdf/2601.11135v1)

**作者:** Van Thuy Hoang `[一作]` (Catholic University of Korea), O-Joun Lee `[通讯]` (Catholic University of Korea)

**通讯引用:** 638 | [OpenAlex ID](https://openalex.org/A5069645890)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种上下文感知的因果推断框架 CaMol，用于在少样本分子属性预测中自动发现并利用关键子结构。

**💡 创新点**

创新点在于：①构建包含功能团、分子与属性的上下文图来引导因果子结构发现；②设计可学习的原子遮罩策略分离因果与噪声子结构；③使用基于功能团的分布干预实现后门调整，使模型能有效剔除混杂效应。

**🔧 技术方法**

主要技术包括：图神经网络（EGIN）、可学习的遮罩机制（Gumbel–Sigmoid）、因果结构模型（SCM）与后门调整、元学习优化（MAML）以及对抗正则化（KL 与 invariance 损失）。

**📊 数据集**

实验使用 MoleculeNet 上的 Tox21、SIDER、MUV、ToxCast、PCBA、ClinTox 等六个少样本属性预测数据集；解释性实验使用 Benzene、Alkane Carbonyl、Fluoride Carbonyl 三个有真值子结构的数据集。

**📈 对比分析**

与传统元学习方法（MAML、ProtoNet 等）和基于上下文的对抗学习方法（Pin‑Tuning、PAR、GS‑Meta 等）进行对比，CaMol 在所有 10/5/1‑shot 任务中均实现了 4–7% 的 ROC‑AUC 提升，并在解释性指标（Fidelity、JSD）上显著优于对手。

**⚠️ 局限性**

主要局限包括：①对功能团划分和前置知识的依赖，若功能团定义不准确可能影响性能；②模型在极度稀疏或高度异质的数据场景下仍需进一步验证；③训练复杂度相较于简单 GNN 较高，需要更大的计算资源。

---

## 140. Differentially Private Subspace Fine-Tuning for Large Language Models

**arXiv ID:** 2601.11113 | [PDF](https://arxiv.org/pdf/2601.11113v1)

**作者:** Lele Zheng `[一作]` (Xidian University), Yulong Shen `[通讯]` (Xidian University)

**通讯引用:** 5999 | [OpenAlex ID](https://openalex.org/A5043356063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种双阶段的差分隐私子空间微调框架DP‑SFT，先用梯度轨迹SVD构造低维任务特定子空间，再在该子空间内注入DP Gaussian噪声并投影回原空间进行参数更新。

**💡 创新点**

创新点在于：①将DP噪声仅投射到任务相关的低维子空间，显著降低噪声幅度；②子空间可在公共数据上构造并迁移到私有任务，几乎不消耗隐私预算；③两阶段流程实现了高效、稳定的DP微调。

**🔧 技术方法**

使用技术包括：梯度裁剪、Gaussian机制、SVD子空间构造、子空间投影与逆投影、Adam/SGD优化；对比LoRA‑DP、Adapter‑DP等参数高效DP微调方法。

**📊 数据集**

实验数据集为SST‑2、IMDB、QNLI、MNLI，采用RoBERTa‑base作为基础模型。

**📈 对比分析**

与Full‑tuning、Full‑DP、LoRA‑DP、Adapter‑DP等基线比较，在ε=4和ε=1两种隐私预算下，DP‑SFT在所有数据集上均达成接近非DP的准确率，提升幅度最高可达13%+，显著优于其他DP微调方案。

**⚠️ 局限性**

局限性：①若子空间构造时使用带噪声的私有数据，子空间质量下降，导致性能衰减；②子空间迁移依赖任务相似度，对完全不同任务的迁移效果未知；③仍受模型规模和DP预算约束，在极端隐私（ε→0）下效果有限。

---

## 141. Shaping a Quantum-Resistant Future: Strategies for Post-Quantum PKI

**arXiv ID:** 2601.11104 | [PDF](https://arxiv.org/pdf/2601.11104v1)

**作者:** Grazia D'Onghia `[一作]` (Politecnico di Torino), Antonio Lioy `[通讯]` (Politecnico di Torino)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出了面向量子安全的公共密钥基础设施迁移路线图，探讨了X.509证书、CRL和OCSP的后量子实现方案。

**💡 创新点**

创新点在于将混合、组合和并行证书链等多种过渡形式系统化，并结合实际实现评估其对证书大小、验证速度和兼容性的影响。

**🔧 技术方法**

使用NIST标准化的后量子算法（Falcon、Dilithium、Kyber、SPHINCS+等），结合OpenSSL、mbedTLS、OQS库以及自研的Hybrid/Composite证书实现。

**📊 数据集**

实验数据来自在OpenSSL 1.0.2/1.1.1、mbedTLS 2.4.2、CFSSL等开源工具上生成的混合证书、CRL和OCSP响应，尺寸范围从数十KB到数百KB。

**📈 对比分析**

通过比较传统RSA/ECC与混合/组合后量子证书在大小、链验证耗时以及OCSP处理时间上的差异，发现后量子证书显著增大了链长度和网络延迟，但验证速度在大多数场景仍可接受。

**⚠️ 局限性**

局限性包括缺乏对OCSP交易耗时的详细测量、对移动设备等资源受限环境的评估不足，以及在不同实现间的兼容性差异尚未彻底验证。

---

## 142. Towards Quantum-Resistant Trusted Computing: Architectures for Post-Quantum Integrity Verification Techniques

**arXiv ID:** 2601.11095 | [PDF](https://arxiv.org/pdf/2601.11095v1)

**作者:** Grazia D'Onghia `[一作]` (Politecnico di Torino), Antonio Lioy `[通讯]` (Politecnico di Torino)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文设计并提出一种多层次、可迁移的可信计算体系架构，将后量子密码（PQC）集成到固件安全启动、测量启动和远程证明等完整可信链中。

**💡 创新点**

创新点在于：①将PQC算法与传统可信根（RoT）/物理/软件TPM耦合，形成适用于ARM TrustZone的fTPM和x86混合方案；②提出混合签名策略（经典ECDSA + PQ包装），兼顾现有硬件与后量子安全；③为固件保护制定完整的密钥生命周期和状态管理方案。

**🔧 技术方法**

使用的技术包括：NIST标准的后量子签名算法（ML‑DSA、SLH‑DSA、LMS、XMSS等）、量子安全哈希函数（SHA‑384/512、SHA‑3‑512）、TPM/物理TPM扩展、ARM TrustZone、Keylime验证框架与liboqs库、Linux IMA模块配置。

**📊 数据集**

本文没有使用公开数据集；评估基于算法性能指标（签名/验证速度、密钥/签名尺寸、资源占用）以及对典型固件镜像、测量日志的模拟测试。

**📈 对比分析**

比较方法主要是对不同PQC算法在验证时间、签名大小、资源消耗等维度进行量化，对比经典ECDSA/PKCS#1。实验结果显示，ML‑DSA在签名速度与尺寸上相对最优，LMS/XMSS在固件安全启动中具有可接受的性能，SLH‑DSA虽安全级别最高但签名尺寸巨大且生成慢。

**⚠️ 局限性**

局限性包括：①缺乏大规模实测性能评估和硬件实现细节；②状态化哈希签名（LMS/XMSS）对状态管理与安全存储提出挑战；③在混合签名方案中，TPM扩展实现复杂度高；④尚未对量子攻击模型的完整评估与对策提供细化方案。

---

## 143. MiCA: A Mobility-Informed Causal Adapter for Lightweight Epidemic Forecasting

**arXiv ID:** 2601.11089 | [PDF](https://arxiv.org/pdf/2601.11089v1)

**作者:** Suhan Guo `[一作]` (Nanjing University), Furao Shen `[通讯]` (Nanjing University)

**通讯引用:** 1799 | [OpenAlex ID](https://openalex.org/A5036608458)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种轻量级的“Mobility‑Informed Causal Adapter (MiCA)”模块，能够将基于因果发现的移动性信息注入时间序列预测模型；

**💡 创新点**

创新点在于通过因果推断得到定向、带权重的移动性先验，再通过双层门控的残差混合方式软性引入，使轻量模型在噪声和数据稀缺的流行病场景下既能利用空间关联又保持鲁棒性；

**🔧 技术方法**

使用PCMCI因果发现、加权指数衰减的图构建、两层门控（全局门与边缘门）残差混合、轻量化的RAM‑pruned PatchTST或DLinear作为时序骨干；

**📊 数据集**

在四个真实流行病数据集上验证：COVID‑19 事件（发病与死亡，日数据）、Influenza（周数据）、Dengue（周数据），涉及49个美国州或27个巴西市；

**📈 对比分析**

与传统统计模型（AR、ARMA）、循环模型（RNN、GRU、LSTM）以及专门的时空模型（DCRNN、STGCN、ColaGNN）对比，MiCA在轻量骨干上平均降低RMSE约5.6%、MAE约9.4%，在多数数据集上甚至超越参数量更大的时空模型；

**⚠️ 局限性**

局限在于：需要足够的历史移动性数据进行因果发现；门控机制与权重需手动调参；在极度稀缺或极噪声的流行病记录中，因果图可能不稳定，导致性能波动。

---

## 144. PhysRVG: Physics-Aware Unified Reinforcement Learning for Video Generative Models

**arXiv ID:** 2601.11087 | [PDF](https://arxiv.org/pdf/2601.11087v1)

**作者:** Qiyuan Zhang `[一作]` (Zhejiang University), Changqing Zou `[通讯]` (Zhejiang Lab)

**通讯引用:** 2920 | [OpenAlex ID](https://openalex.org/A5100604564)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于强化学习的物理感知视频生成框架，利用物理约束直接指导生成过程。

**💡 创新点**

创新点在于：①设计了物理基准奖励函数（Trajectory Offset 与 Collision Detection）；②引入 Mimicry‑Discovery Cycle 通过在学习初期使用 Flow Matching 稳定 RL，逐步过渡到物理探索；③构建了包含 700 条刚体运动视频的新基准，用 IoU/TO 量化物理一致性。

**🔧 技术方法**

技术包括：Transformer‑based Video Diffusion（Flow Matching）、GRPO 强化学习、SAM2 生成运动掩码、Mimicry‑Discovery Cycle 训练策略、以及多阶段 V2V 微调。

**📊 数据集**

使用的数据集：公开视频集（Panda‑70M、InternVid、WebVid‑10M）、自研竞赛与游戏视频，以及 700 条手工标注的刚体运动视频（冲击、摆动、自由落体、滚动）。

**📈 对比分析**

与 VBench、VideoPhy‑2 及传统 V2V/ I2V 基线对比，本文模型在 IoU、TO 上显著优于其他方法，同时保持优秀的视觉质量（VBench 分数高于同类模型）。

**⚠️ 局限性**

局限性：主要关注刚体运动，对柔体或复杂交互效果缺乏覆盖；奖励设计对标注质量高度依赖；训练过程复杂、需要大规模计算；在人类主体偏置的场景中仍可能出现不符合物理的生成。

---

## 145. Clustering High-dimensional Data: Balancing Abstraction and Representation Tutorial at AAAI 2026

**arXiv ID:** 2601.11160 | [PDF](https://arxiv.org/pdf/2601.11160v1)

**作者:** Claudia Plant `[一作]` (University of Vienna), Christian Böhm `[通讯]` (University of Vienna)

**通讯引用:** 26243 | [OpenAlex ID](https://openalex.org/A5077864331)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并比较了传统K‑means、子空间聚类以及深度聚类（包括自编码器、对比学习和生成式方法）在高维数据（如交通标识图像）上的效果，提出未来混合方法的研究方向。

**💡 创新点**

通过系统对比多种聚类技术并引入子空间与深度方法的混合框架，强调在不同尺度上灵活平衡抽象与表征的必要性，提出多表征与层次化学习的思路。

**🔧 技术方法**

采用K‑means、Sub‑K‑Means、PCA、深度自编码器（如DEC、IDEC、DeepECT）、对比学习网络（如CC）、变分自编码器（VaDE）以及生成式对抗网络（ClusterGAN）等技术。

**📊 数据集**

主要使用德国交通标志基准集（GTSB）中约3,000维像素的子集进行实验，并以该数据集为基准进行性能评估。

**📈 对比分析**

对比实验显示：K‑means NMI≈0.28，PCA+K‑means≈0.3，Autoencoder+K‑means≈0.56，DEC≈0.58，IDEC≈0.60，DeepECT≈0.80；生成式方法与对比学习在保持抽象与表征平衡时也取得较好效果。

**⚠️ 局限性**

局限性包括：对高维数据的训练成本高、可解释性差、缺乏自动化的抽象/表征权衡机制、对不同聚类尺度的适配不足，以及对真实多模态或极稀疏数据的评估有限。

---

## 146. Deep GraphRAG: A Balanced Approach to Hierarchical Retrieval and Adaptive Integration

**arXiv ID:** 2601.11144 | [PDF](https://arxiv.org/pdf/2601.11144v1)

**作者:** Yuejie Li `[一作]` (Ant Group), Chengjun Mao `[通讯]` (Ant Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Deep GraphRAG框架，实现了层次化的全局-局部检索和动态beam‑search重排，用于提升检索增强生成的准确性与效率。

**💡 创新点**

创新点包括三阶段层次检索策略、Beam搜索优化的动态重排模块以及针对多目标奖励的动态加权GRPO（DW‑GRPO）强化学习方法。

**🔧 技术方法**

使用技术有图神经网络层次聚类、向量相似度检索、Beam搜索、LLM（Qwen2.5‑72B、1.5B）以及DW‑GRPO强化学习。

**📊 数据集**

实验数据集为Natural Questions和HotpotQA。

**📈 对比分析**

与本地检索、全局搜索及Drift Search基线相比，Deep GraphRAG在EM‑Total上实现了44.69%（NQ）和45.44%（HotpotQA）的最高分，同时显著降低了检索延迟。

**⚠️ 局限性**

局限在于对综合问题的局部细节检索仍有欠缺，层次化总结可能导致细粒度事实被模糊或忽略。

---

## 147. Learning Quadrupedal Locomotion for a Heavy Hydraulic Robot Using an Actuator Model

**arXiv ID:** 2601.11143 | [PDF](https://arxiv.org/pdf/2601.11143v1)

**作者:** Minho Lee `[一作]` (Korea Advanced Institute of Science and Technology), Jemin Hwangbo `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5314 | [OpenAlex ID](https://openalex.org/A5080397455)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个基于物理的解析液压执行器模型，并用其训练RL行走控制器，使300kg级四足机器人实现1 m/s高速稳步行走。

**💡 创新点**

创新点在于用简化的液压动力学公式加入阻力与冲击校正项，实现超低计算成本（<1 µs）且在数据稀缺情况下仍能高精度预测扭矩。

**🔧 技术方法**

使用技术包括解析液压模型、PPO强化学习、400并行仿真环境、估计器网络以及位置/扭矩PID控制。

**📊 数据集**

数据集来自真实机器人20 秒的行走记录（采样1000 Hz）与对应仿真数据，用以训练并对比模型。

**📈 对比分析**

与MLP、LSTM、GRU基准相比，模型的RMSE/MAPE显著更低（约3‑5% vs 20‑60%），训练时间缩短到三分之一，行走速度与稳定性均有提升。

**⚠️ 局限性**

局限在于未验证极端速度、急剧方向变化、长期耐久性以及跨平台的泛化能力。

---

## 148. FlashLabs Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning

**arXiv ID:** 2601.11141 | [PDF](https://arxiv.org/pdf/2601.11141v1)

**作者:** Tanyu Chen `[一作]` (FlashLabs), Yi Shi `[通讯]` (FlashLabs)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 Chroma 1.0，一个开源、实时、端到端的语音对话模型，能够在低延迟下实现高保真个性化语音克隆并具备良好的推理与对话能力。

**💡 创新点**

创新点包括 1:2 的文本-音频交织生成调度、将语音理解与生成紧密耦合的流式架构、将 Backbone 与 Decoder 分离以提升推理速度，以及仅用 4B 参数即可实现高质量语音克隆。

**🔧 技术方法**

采用了语音分词器、神经音频编解码器（EnCodec、RVQ）、Qwen2‑Audio 编码、CSM‑1B 参考音频嵌入、基于 LLaMA 的 Backbone 与轻量化 Decoder、Causal CNN 音频解码器，并实现了 24kHz 采样率的高保真输出。

**📊 数据集**

训练使用自制的 LLM‑TTS 生成语音对话数据，评估则主要基于 CommonVoice、URO‑Bench 以及与 ElevenLabs 等商业系统的主观/客观比较。

**📈 对比分析**

在零样本语音克隆上相较于人类基线提升 10.96% 的 speaker similarity，实时因子 RTF 为 0.43，首音频令牌延迟 TTFT 仅 146.87 ms；与 ElevenLabs 比较，语音相似度基本持平但自然度略低，整体性能与当前主流系统相当。

**⚠️ 局限性**

局限性包括仅支持英文、训练数据为合成语音、缺乏多语言与多方言支持、无法批量推理，以及对极端口音与噪声环境的鲁棒性尚待验证。

---

## 149. A Defender-Attacker-Defender Model for Optimizing the Resilience of Hospital Networks to Cyberattacks

**arXiv ID:** 2601.11129 | [PDF](https://arxiv.org/pdf/2601.11129v1)

**作者:** Stephan Helfrich `[一作]` (Karlsruhe Institute of Technology), Emilia Grass `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 313 | [OpenAlex ID](https://openalex.org/A5007275023)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种防御者-攻击者-防御者优化模型，用于评估并提升医院网络在网络攻击中的韧性，模型将攻击图与医院服务容量的相互依赖关系结合，并考虑时间维度的损失、恢复与抵御指标。

**💡 创新点**

创新点在于：①将攻击图驱动的威胁建模直接映射到医院服务容量衰减；②首次在同一框架中同时评估预防与应急策略；③将长期恢复和抵御能力纳入目标函数，实现对攻击后完整时间窗的韧性量化。

**🔧 技术方法**

使用技术包括：混合整数线性规划（MILP）与列/约束生成求解器、攻击图生成工具、基于可利用性分数的攻击成本建模，以及基于欧盟健康系统韧性框架的多目标加权目标函数。

**📊 数据集**

实验数据来自德国巴登-符腾堡州的公共医院网络，包含62家医院、58种手术类别、约20,611个攻击图节点和144,158条边，攻击图依据公开的IT基础设施与漏洞数据库构建。

**📈 对比分析**

通过案例研究和灵敏度分析将模型与无预防措施基线对比，结果显示在第一阶段重点投入合作协议与备份容量可将损失约85%、恢复时间显著缩短，模型求解在完整实例上耗时约5天，体现出可行但计算量大。

**⚠️ 局限性**

主要局限包括：对大规模攻击图的求解可扩展性不足；模型仅考虑单一攻击者预算与单一攻击场景；未考虑攻击蔓延与资产间的级联失效；以及缺乏多玩家博弈视角。

---

## 150. Learn Before Represent: Bridging Generative and Contrastive Learning for Domain-Specific LLM Embeddings

**arXiv ID:** 2601.11124 | [PDF](https://arxiv.org/pdf/2601.11124v1)

**作者:** Xiaoyu Liang `[一作]` (Zhejiang University), Xincheng Zhou `[通讯]` (Peking University)

**通讯引用:** 659 | [OpenAlex ID](https://openalex.org/A5100937167)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 Learn Before Represent (LBR) 两阶段框架，先通过信息瓶颈约束的生成学习注入垂直领域知识，再用生成精炼的对比学习进行语义对齐。

**💡 创新点**

核心创新是将信息瓶颈机制与生成学习结合，既保留自回归注意力架构，又压缩输入语义，解决生成与对比学习的目标冲突，避免表示崩塌。

**🔧 技术方法**

技术包括信息瓶颈约束的自回归生成学习（IB‑GL），生成精炼对比学习（GR‑CL），以及在保持因果注意力下的压缩标记与掩码设计。

**📊 数据集**

实验使用医学、化学、代码三大垂直领域数据集（共计约300k样本），并在MTEB检索任务上评估。

**📈 对比分析**

与传统 LLM+GL、LLM+CL、LLM+GL+CL 等基线对比，LBR 在三领域 R@10 和 NDCG@10 上均显著领先，最高可达 87.9% 的平均分，排名第三（小模型亦能超越大型基线）。

**⚠️ 局限性**

局限性包括：仅验证实体密集型领域，未覆盖法律、金融等推理强度高的场景；目前仅注入事实知识，缺乏高阶推理；压缩标记长度固定，可能不适用于信息密度差异大的输入。

---

## 151. ReCreate: Reasoning and Creating Domain Agents Driven by Experience

**arXiv ID:** 2601.11100 | [PDF](https://arxiv.org/pdf/2601.11100v1)

**作者:** Zhezheng Hao `[一作]` (Zhejiang University), Jiawei Chen `[通讯]` (Zhejiang University)

**通讯引用:** 4186 | [OpenAlex ID](https://openalex.org/A5100362810)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于交互经验的自动化域级代理创建框架 ReCreate。

**💡 创新点**

创新点在于用交互轨迹、评估日志等白盒证据驱动 scaffold 更新，并引入经验存储、推理-创建协同和层级更新机制。

**🔧 技术方法**

主要技术包括 LLM 代理架构、经验检索与索引、推理-创建路由、域级抽象更新；采用 GPT 系列模型作为任务代理和 ReCreate‑Agent。

**📊 数据集**

在软件工程、数据科学、数学和数字助手四大领域的 13 个基准数据集（如 Django、SymPy、NumPy、AppWorld 等）上进行评测。

**📈 对比分析**

与人工设计的 scaffold、Self‑Evolve、Agent Generation 方法对比，ReCreate 在平均分上提升 5–7%，在所有领域均优于基线，且成本比 ADAS 降低 36–82%。

**⚠️ 局限性**

局限性包括仅优化文本/代码层面的 scaffold，未处理执行环境或底层系统定制；且未对基础模型进行微调。

---

## 152. ABC-Bench: Benchmarking Agentic Backend Coding in Real-World Development

**arXiv ID:** 2601.11077 | [PDF](https://arxiv.org/pdf/2601.11077v1)

**作者:** Jie Yang `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ABC-Bench评测框架和ABC-Pipeline任务生成流水线，用于评估LLM驱动的后端软件工程代理在完整生命周期（代码编辑、环境配置、部署、端到端测试）中的表现。

**💡 创新点**

创新点在于：①实现了从真实GitHub仓库自动抽取完整后端开发任务，覆盖8种语言、19种框架；②首次将环境配置与容器化部署纳入评测循环，真正检验服务能否在生产‑级环境中启动并通过外部API测试；③构建了可扩展的流水线（ABC-Pipeline）实现任务批量生成，显著降低人工成本。

**🔧 技术方法**

技术手段包括：大型语言模型与多轮交互代理（OpenHands、GPT‑5、Claude Sonnet 4.5等）、Docker容器化构建与部署、自动化API测试套件、GitHub仓库分析与代码补丁生成（masking策略）以及交互回合统计与性能分析。

**📊 数据集**

数据集：从2,000个MIT许可的后端开源仓库中抽取，生成224个完整任务，涵盖8种编程语言（Python、Go、JavaScript、Java、Ruby、C#、PHP、Rust）与19个主流框架，任务中92个专注于自主环境配置。

**📈 对比分析**

比较方法：使用pass@1（单次尝试通过率）作为主指标，每个模型-代理组合在每个任务上执行3次；结果显示最强模型Claude Sonnet 4.5达63.2% pass@1，DeepSeek‑V3.2约50%，小模型如Qwen3‑8B不到10%。进一步拆分发现环境构建阶段（S_1）是主要瓶颈，S_2（功能测试）表现较好；交互回合数与性能呈正相关（r≈0.87）。

**⚠️ 局限性**

局限性：环境配置与容器化部署仍是主要障碍；不同语言/框架间表现差异显著，尤其是Rust；模型规模受限导致部分任务被S_1过滤；评测依赖手动验证的测试套件，可能忽略部分细节；缺乏对多轮推理与自适应决策的深入研究。

---

## 153. LSTM VS. Feed-Forward Autoencoders for Unsupervised Fault Detection in Hydraulic Pumps

**arXiv ID:** 2601.11163 | [PDF](https://arxiv.org/pdf/2601.11163v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 154. A3D: Adaptive Affordance Assembly with Dual-Arm Manipulation

**arXiv ID:** 2601.11076 | [PDF](https://arxiv.org/pdf/2601.11076v1)

**作者:** Jiaqi Liang `[一作]` (Peking University), Ruihai Wu `[通讯]` (Peking University)

**通讯引用:** 196 | [OpenAlex ID](https://openalex.org/A5086096450)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

学习双臂家具装配的自适应支撑与稳定化 affordance，并通过交互反馈实时调整支撑策略；

**💡 创新点**

提出点级 affordance 与交互上下文适配模块，使得能在多形状零件上通用预测最佳支撑点，并在双臂协作中动态优化策略；

**🔧 技术方法**

采用 PointNet++ 提取点云特征，cVAE 生成支撑方向，分支 Affordance‑Proposal‑Scoring 架构，结合交互上下文注意力融合，实现对形状与动力学的感知与自适应；

**📊 数据集**

基于扩展的 FurnitureBench（IsaacGym）数据集，包含 50+ 多样化零件、8 类家具以及 4 种装配任务，既在仿真中也在真实双臂机器人上进行实验；

**📈 对比分析**

与 Random、Heuristic、DP3、LLM‑Guided 基线及消融实验对比，在四个任务中的成功率约 70–80%，相较基线提升 10–20%；在真实实验中分别取得 11/15、12/15、9/15 的成功率；

**⚠️ 局限性**

受限于运动规划，RRTConnect 有时无法找到可行轨迹，导致失败；未来需开发更稳健的运动规划或运动细化策略以提升实地鲁棒性。

---

## 155. Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs

**arXiv ID:** 2601.11061 | [PDF](https://arxiv.org/pdf/2601.11061v1)

**作者:** Lecheng Yan `[一作]` (Southern University of Science and Technology), Chris Lee `[通讯]` (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 RLVR 训练中伪奖励如何激活 LLM 的记忆捷径，揭示了 Anchor-Adapter 机制，并提出基于 MLP 键的可逆调控方法。

**💡 创新点**

发现并验证了“Perplexity Paradox”，定位功能锚层（L18-20）与结构适配器层（L21+）的分离机制，并提供可调节的神经机制干预手段。

**🔧 技术方法**

使用 Path Patching、Logit Lens、JSD、Neural Differential Equations、线性探测、AUC 识别以及神经关键激活尺度调控等多种解释与干预技术。

**📊 数据集**

评估数据集包括 Qwen2.5-Math 7B 及其 RLVR 调优模型，基准集包含 MATH-500、MinervaMath、LiveMathBench、AIME2024/25 等数学推理数据。

**📈 对比分析**

通过与原始模型对比和多维度指标（准确率、PPL、AUC、JSD）评估，伪奖励在泄漏集上提升至约 90% 以上准确率，非泄漏集保持约 70%；干预可恢复基线性能。

**⚠️ 局限性**

仅针对 Qwen2.5-Math，未验证其他 LLM；机制推断依赖多重工具交叉验证；调控方法在大模型中实现成本高；未针对如何消除泄漏数据本身给出完整方案。

---

## 156. SoLA-Vision: Fine-grained Layer-wise Linear Softmax Hybrid Attention

**arXiv ID:** 2601.11164 | [PDF](https://arxiv.org/pdf/2601.11164v1)

**作者:** Ruibang Li `[一作]` (Chinese Academy of Sciences), Weiming Hu `[通讯]` (ShanghaiTech University)

**通讯引用:** 23571 | [OpenAlex ID](https://openalex.org/A5114549594)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SoLA‑Vision，一种全注意力骨干网络，采用线性注意力与 softmax 注意力的细粒度层级混合，并引入 Hidden State Bridge 机制来提升长距离建模与效率平衡。

**💡 创新点**

创新点包括：① 对线性与 softmax 注意力的有效范围进行理论分析，证明仅需少量全局 softmax 层即可弥补信息衰减；② 设计了稀疏层级混合策略（如在 Stage3 采用 LLSLLS），显著提升精度；③ 引入 Hidden State Bridge 在高分辨率线性层与后续 softmax 层之间桥接高层特征。

**🔧 技术方法**

使用技术包括 WKV 线性注意力、标准 multi‑head softmax 注意力、层级化阶段设计、采样+门控的 Hidden State Bridge、全局与局部窗口自适应混合、以及大规模预训练与细粒度调优。

**📊 数据集**

实验数据集涵盖 ImageNet‑1K（分类）、COCO‑2017（目标检测）和 ADE20K（语义分割），并在 100‑类子集上进行层级混合实验。

**📈 对比分析**

通过与 ResNet、DeiT、Swin、Vim、VRWKV、Mamba、MambaVision 等主流模型对比，SoLA‑Vision 在 ImageNet‑1K 上达到 79.8% / 82.9% / 84.1% 的 Top‑1，检测 AP^b 43.8 / 46.6 / 47.5，分割 mIoU 44.7 / 48.1 / 50.5，均在相同或更低 FLOPs、参数量下优于现有线性、窗口化与混合模型。

**⚠️ 局限性**

局限性包括：① 层级混合位置与比例仍需人工设计，缺乏自动化搜索或学习策略；② 对极高分辨率输入的扩展仍有限；③ Hidden State Bridge 机制的理论解释和鲁棒性尚未深入；④ 在某些任务或模型规模下，softmax 层数的增减对性能的影响仍需进一步验证。

---

## 157. Vertex ordering characterizations of interval r-graphs

**arXiv ID:** 2601.11158 | [PDF](https://arxiv.org/pdf/2601.11158v1)

**作者:** Indrajit Paul `[一作]` (University of Calcutta), Ashok Kumar Das `[通讯]` (University of Calcutta)

**通讯引用:** 23991 | [OpenAlex ID](https://openalex.org/A5011213078)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对一般r-分区图的间隔图性质给出了两种顶点排序表征，并据此提出了相应的禁用模式，实现了interval r-graphs的完全结构性描述。

**💡 创新点**

提出了“广义间隔排序”和“r-间隔排序”两种全新的顶点排序，并证明它们与interval r-graphs之间的等价关系；同时给出了有限集合的禁用模式，完成了该类图的完全描述。

**🔧 技术方法**

主要使用了图论中的顶点排序、区间模型、邻接矩阵的“几乎连续1”特性，以及组合逻辑推理。

**📊 数据集**

无实验数据集，全部为理论证明与图示说明。

**📈 对比分析**

未进行实验对比或性能评估，文中仅给出了理论等价性与构造证明。

**⚠️ 局限性**

仅提供理论框架，缺乏实现细节与复杂度分析；未验证在实际应用场景中的可行性和效率。

---

## 158. Sensing Mutual Information for Communication Signal with Deterministic Pilots and Random Data Payloads

**arXiv ID:** 2601.11149 | [PDF](https://arxiv.org/pdf/2601.11149v1)

**作者:** Lei Xie `[一作]` (Southeast University), Shenghui Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 3957 | [OpenAlex ID](https://openalex.org/A5025467937)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了带有确定性导频和随机数据载荷的通信波形在集成感知与通信（ISAC）系统中的感知互信息（SMI），并给出了闭式近似表达式和基于 ADMM 的预编码优化方法。

**💡 创新点**

创新点在于首次对混合导频/数据波形的 SMI 进行随机矩阵理论推导，得到可解析的近似式，并在此基础上设计了兼顾感知和通信约束的预编码器。

**🔧 技术方法**

使用了随机矩阵理论（RMT）求解 SMI 近似、梯度投影（GP）与 ADMM 交替求解预编码优化、以及基于 SDP 的半正定规划。

**📊 数据集**

实验采用毫米波 28 GHz 环境下的模拟数据（N_t=4、N_r=6、L≥8），无公开数据集，仅进行 Monte‑Carlo 仿真。

**📈 对比分析**

通过与 Empirical Monte‑Carlo 结果以及感知导向、通信导向、时分共享、波束合成等基线进行对比，所提方法在给定通信速率下实现了更高的 SMI，且在不同 L 值下逼近理论近似。

**⚠️ 局限性**

局限性包括：近似表达式仅在大 L 下收敛；模型假设单目标且仅考虑 AWGN；算法复杂度高，未在真实硬件或多目标场景中验证。

---

## 159. Optimized Algorithms for Text Clustering with LLM-Generated Constraints

**arXiv ID:** 2601.11118 | [PDF](https://arxiv.org/pdf/2601.11118v1)

**作者:** Chaoqi Jia `[一作]` (RMIT University), Kok-Leong Ong `[通讯]` (RMIT University)

**通讯引用:** 2340 | [OpenAlex ID](https://openalex.org/A5013793366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用大语言模型生成的集合式约束（must‑link和cannot‑link）来改进短文本聚类的完整框架；

**💡 创新点**

创新点在于：①将传统的成对约束转化为集合式约束，显著降低LLM查询次数；②针对LLM产生的不确定约束，设计硬/软约束分层处理与惩罚机制；③提出基于本地搜索的CL约束匹配算法；

**🔧 技术方法**

技术包括LLM（以In-Context Learning方式）生成约束；距离/coreset方法用于候选集筛选；k-means++、soft‑penalty、局部搜索及最大匹配等聚类算法；

**📊 数据集**

使用五个真实短文本数据集：tweet、banking77、clinc（I/D）、GoEmo及其它；嵌入模型采用Instructor‑large和E5；

**📈 对比分析**

与四类基线（FSC、COP、PCK、CKM++、BH‑KM）以及无约束k‑means相比，实验显示：①约束生成查询数比FSC低20×；②生成约束准确率提升12%+；③聚类ACC、NMI、ARI均超过基线，尤其在约束比例20%时表现最佳；

**⚠️ 局限性**

局限包括：①LLM的质量依赖于所选模型，低质量LLM会削弱效果；②CL约束匹配算法时间复杂度较高（O(k^9/2 + nk^4)），在极大数据规模下可能不可行；③对极高约束比例时，错误约束比例上升导致性能下降。

---

## 160. Sparing User Time with a Socially-Aware Independent Metaverse Avatar

**arXiv ID:** 2601.11115 | [PDF](https://arxiv.org/pdf/2601.11115v1)

**作者:** Theofanis P. Raptis `[一作]` (Institute of Informatics and Telematics), Andrea Passarella `[通讯]` (Institute of Informatics and Telematics)

**通讯引用:** 11391 | [OpenAlex ID](https://openalex.org/A5018610428)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了将独立头像（自治代理）整合进元宇宙社交网络的计算模型，并通过启发式算法优化用户闲置时间，从而提升社交效率。

**💡 创新点**

创新点在于将进化人类社会学的“自我网络”与元宇宙头像代理交互框架相结合，证明问题为NP‑hard并给出基于排序请求的启发式求解；同时量化头像代理在时间和社交成本上的收益。

**🔧 技术方法**

使用线性规划、整数规划、冲突图建模、社交成本函数，以及启发式排序调度算法；同时用理论分析证明 NP‑hard；仿真使用 MATLAB。

**📊 数据集**

数据集为自建的 10,000 份 ego‑network 模拟数据，规模在 68、126、170 位联系人；并通过不同参数组合（冲突率、头像时间、截止期限、γ 等）进行实验。

**📈 对比分析**

通过对比非头像（non‑A）与头像（A）两种场景的社交成本、每个 alter 的成本分布及对参数敏感度进行实验；结果显示头像方案可将社交成本降低 75–95%，并在冲突、时间限制下保持优势。

**⚠️ 局限性**

局限包括：仅考虑单一用户的头像代理，未建模多用户交互；仿真使用人工生成的网络而非真实社交数据；启发式解未证明全局最优；对头像行为复杂度和信任等社交心理因素未深入研究。

---

## 161. Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis

**arXiv ID:** 2601.11102 | [PDF](https://arxiv.org/pdf/2601.11102v1)

**作者:** Shangbo Yuan `[一作]` (University of Electronic Science and Technology of China), Na Zhao `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 8176 | [OpenAlex ID](https://openalex.org/A5040897632)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种结合图平滑模块和局部几何学习模块的点云分类与分割方法（GSPoint）。

**💡 创新点**

创新点在于：① 通过对称邻接修正与von Neumann核实现图结构的平滑，解决边界稀疏与节点噪声问题；② 利用可学习的形状特征和圆柱坐标变换提取局部几何信息，显著提升特征表达能力。

**🔧 技术方法**

采用图卷积、对称邻接归一化、多步图平滑、von Neumann核、MLP学习特征、局部协方差特征、圆柱坐标变换等技术。

**📊 数据集**

使用了 ModelNet40、ScanObjectNN、ShapeNetPart、S3DIS 等公开数据集进行评估。

**📈 对比分析**

与 PointNet++、PointMLP、PointNeXt、PointWavelet、DuGREAT 等主流方法对比，GSPoint 在 ModelNet40 的 mAcc/ OA 达到 91.5%/94.5%，ScanObjectNN 达到 86.4%/88.1%，ShapeNetPart 的 Cls.mIoU/Ins.mIoU 达到 85.6%/87.2%，S3DIS 的 mIoU/ OA 达到 71.5%/91.2%，整体性能优于或与最先进方法持平。

**⚠️ 局限性**

局限性包括：图平滑计算开销较大，对参数（α、T）敏感；未在极大规模或完全未知类别的点云上进行广泛测试；实现复杂，推理速度相对慢。

---

## 162. CoDance: An Unbind-Rebind Paradigm for Robust Multi-Subject Animation

**arXiv ID:** 2601.11096 | [PDF](https://arxiv.org/pdf/2601.11096v1)

**作者:** Shuai Tan `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**通讯引用:** 33631 | [OpenAlex ID](https://openalex.org/A5078109015)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种能够在任意数量、任意类型、任意空间布局的角色图像动画框架，可用单个未对齐的驱动姿态序列为多主体图像生成逼真、可控的舞蹈视频。

**💡 创新点**

核心创新是 Unbind‑Rebind 两阶段策略：Unbind 模块通过随机平移/缩放姿态并在特征层做扰动，破除姿态与参考图像的严格空间绑定，学习位置无关的运动语义；Rebind 模块利用文本提示和目标主体掩模实现语义与空间重绑定，使模型能够在空间错位的情况下准确定位并动画指定角色。

**🔧 技术方法**

技术手段包括：基于 Diffusion Transformer（DiT）的条件扩散网络，使用 VAE 对图像进行潜在编码；Pose Shift Encoder（3D 卷积）提取姿态特征；文本编码采用 umT5；掩模编码器使用 SAM 提取的主体掩模；LoRA 进行微调；混合数据训练策略将动画数据与大规模文本‑视频数据交替训练，以提升文本理解与语义绑定能力。

**📊 数据集**

训练数据来源于公开的 TikTok 与 Fashion 视频数据集，并自行采集约 1,200 条 TikTok‑style 视频；为强化文本与空间绑定，额外加入 10,000 条文本‑视频样本和 20 条多主体视频；评测使用单主体的 TikTok 与 Fashion 数据集以及多主体的 Follow‑Your‑Pose‑V2 benchmark 和自建的 20 条多主体舞蹈视频 benchmark。

**📈 对比分析**

在 Follow‑Your‑Pose‑V2 和自建 benchmark 上与 AnimateAnyone、MusePose、ControlNeXt、MimicMotion、UniAnimate、Animate‑X、StableAnimator、UniAnimateDiT 等 SOTA 单主体方法进行对比，采用 LPIPS、PSNR/SSIM、FVD 等指标。实验显示该方法在多主体场景下显著优于所有基线，在身份保持、运动一致性和视觉真实性方面取得最高分，且在单一姿态驱动多主体的挑战性设置中依旧保持较好表现。

**⚠️ 局限性**

局限性包括：仍需依赖精确的主体掩模（若掩模质量差会影响动画质量）；对极端姿态变化或非常大规模的多主体场景可能仍有误差；模型训练仅基于单主体视频，虽然能在多主体场景中工作，但在极端分布差异下可能需要进一步多样化训练数据。

---

## 163. Soft Bayesian Context Tree Models for Real-Valued Time Series

**arXiv ID:** 2601.11079 | [PDF](https://arxiv.org/pdf/2601.11079v1)

**作者:** Shota Saito `[一作]` (Gunma University), Toshiyasu Matsushima `[通讯]` (Waseda University)

**通讯引用:** 460 | [OpenAlex ID](https://openalex.org/A5110471799)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Soft-BCT模型，用软（概率）分割上下文空间来建模实值时间序列，并给出了对应的变分推断学习算法；

**💡 创新点**

创新点在于将传统硬阈值的BCT-AR扩展为软分割形式，允许不同节点使用不同阈值，并通过变分推断实现近似MAP估计；

**🔧 技术方法**

主要技术包括多层软max回归来决定路径、基于变分推断的概率推断、以及MAP估计和后验预测的递归求解；

**📊 数据集**

实验使用了人工生成数据（四种不同模型）以及若干真实世界时间序列数据（如Genz、Wind等公开数据集）；

**📈 对比分析**

与BCT-AR进行对比，使用均方误差评价；在人工数据上性能略逊于BCT-AR，而在大多数真实数据上表现相当或略优；

**⚠️ 局限性**

局限性包括：在硬阈值生成的数据上无法提升；需要手动设置或搜索阈值超参数；当M>2时多分类logistic回归的变分求解尚无高效算法，导致计算复杂度较高。

---

## 164. Fairness in Healthcare Processes: A Quantitative Analysis of Decision Making in Triage

**arXiv ID:** 2601.11065 | [PDF](https://arxiv.org/pdf/2601.11065v1)

**作者:** Rachmadita Andreswari `[一作]` (Humboldt-Universität zu Berlin), Jan Mendling `[通讯]` (Humboldt-Universität zu Berlin)

**通讯引用:** 25084 | [OpenAlex ID](https://openalex.org/A5062764959)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

分析MIMIC-IV ED紧急科triage流程中的公平性，利用过程挖掘与统计检验识别不同敏感属性对时间、重复、偏差和决策结果的影响。

**💡 创新点**

首次将组织正义理论（分配、程序、互动）与过程挖掘指标相结合，构建公平性评估框架，并在真实医疗事件日志上系统评估多属性偏差。

**🔧 技术方法**

使用过程挖掘技术（Heuristic Miner、Token Replay）、非参数统计检验（Kruskal–Wallis、卡方检验）以及效应量评估（ε²、Cramér’s V）进行数据分析。

**📊 数据集**

基于公开的MIMIC-IV ED事件日志（MIMICEL），共7,422,277条记录、413,893个案例。

**📈 对比分析**

通过比较不同敏感属性在各ESI级别上的显著性与效应量，发现保险和语言对决策与偏差影响最大，年龄在低严重度时影响时间；整体显示公平差异存在但效应量多为小到中等。

**⚠️ 局限性**

缺乏患者临床和社会经济详细信息，未考虑医疗工作者负荷和疾病共病，结果仅为统计差异而非因果关系，且仅基于单一机构的数据。

---

## 165. Cross-Modal Attention Network with Dual Graph Learning in Multimodal Recommendation

**arXiv ID:** 2601.11151 | [PDF](https://arxiv.org/pdf/2601.11151v1)

**作者:** Ji Dai `[一作]` (Beijing University of Posts and Telecommunications), Can Zhao `[通讯]` (Aviation Data Communication Corporation)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种跨模态递归注意网络（CRANE），通过双图结构融合用户行为与物品多模态信息，实现更精准的个性化推荐。

**💡 创新点**

核心创新包括递归跨模态注意力机制（RCA）实现高阶模态融合、对称双图（用户-物品交互图与物品-物品语义图）以及对用户多模态特征的显式构建。

**🔧 技术方法**

采用图卷积网络（GCN）进行结构传播，递归跨模态注意力进行模态对齐，结合自监督对比学习（InfoNCE）对协同视图与语义视图进行对齐。

**📊 数据集**

在四个公开亚马逊数据集（Baby、Sports、Clothing、Electronics）上进行实验，使用预提取的视觉（ResNet50）和文本（BERT）特征。

**📈 对比分析**

与10种基线（BPR、LightGCN、VBPR、MMGCN、SLMRec、LATTICE、FREEDOM、LGMRec、DGAVE、LPIC）比较，CRANE平均提升Recall@20约5%，在大规模Electronics数据集上实现最高Recall@20。

**⚠️ 局限性**

主要限制在于RCA的全连接注意力导致理论O(N²)复杂度，对极大规模物品库（N>10⁶）可能超出GPU内存，需进一步优化稀疏化或近似方法。

---

## 166. Patterns of Bot Participation and Emotional Influence in Open-Source Development

**arXiv ID:** 2601.11138 | [PDF](https://arxiv.org/pdf/2601.11138v1)

**作者:** Matteo Vaccargiu `[一作]` (University of Cagliari), Giuseppe Destefanis `[通讯]` (University College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对十个以太坊开源仓库的 bot 参与模式和情绪影响进行定量分析，结合讨论生命周期与 27 类细粒度情绪分类；

**💡 创新点**

首次将讨论时间分布与情绪动态相结合，揭示 bot 与人类在参与时序、响应速度及情绪取向上的显著差异，并量化 bot 引入后情绪转变的幅度；

**🔧 技术方法**

采用三步 bot 检测框架（规则 + Isolation Forest + 手工校验）、RoBERTa-Base-GoEmotions 情绪分类、Jensen–Shannon Divergence 计算情绪差异，并使用 Kolmogorov‑Smirnov、Wilcoxon、Mann‑Whitney 等非参数检验；

**📊 数据集**

基于 36,875 位贡献者（105 名确认 bot）和 10 个 Ethereum 项目的 issue、PR、commit、comment 数据集，包含 50k+ bot 评论与 181k+ 人类评论；

**📈 对比分析**

通过比较参与时间分布（KS 与 Cliff’s Delta）、响应时间中位数、情绪 JSD 与单情绪概率差异，结果显示 bot 在 PR 中快速持续参与、issue 中响应慢且集中于后期，情绪上 bot 更中性，且其评论后人类评论情绪多样化且正向情绪显著提升，统计显著且效应大；

**⚠️ 局限性**

局限性包括仅聚焦十个以太坊项目，bot 比例低且未区分类型，情绪模型可能受限于预训练数据，生命周期归一化可能掩盖短长线程差异，未建立因果关系，且仅使用公开 GitHub 数据，可能缺乏对其他生态系统的普适性。

---

## 167. The Big Ban Theory: A Pre- and Post-Intervention Dataset of Online Content Moderation Actions

**arXiv ID:** 2601.11128 | [PDF](https://arxiv.org/pdf/2601.11128v1)

**作者:** Aldo Cerulli `[一作]` (IIT-CNR), Stefano Cresci `[通讯]` (IIT-CNR)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建并发布了名为“The Big Ban Theory”的大规模基于干预的数据集，涵盖25种不同类型的内容审核干预（如社区禁令、隔离、内容删除、迁移等），并为每个干预提供干预前后各三个月的用户活动记录（共339,125名用户、约39.0 M条评论）。

**💡 创新点**

创新点在于①提供统一、可跨干预、跨平台对齐的四维切片（内/外、前/后）结构；②覆盖多样化的干预类型与时间跨度；③使用标准化JSON格式并严格伪匿名化，遵循FAIR原则，显著提升了研究复现性和可比性；④为多种研究任务（效果评估、公平性检验、预测建模等）提供了可直接使用的基础数据。

**🔧 技术方法**

技术手段包括：从Reddit（torrent文件）和Voat（公开仓库）获取原始数据；采用bot过滤（时间戳相同的多条评论判定为机器人）与手工校验；通过字段映射与统一命名实现跨年份、跨平台的字段标准化；使用哈希加密对所有ID、用户名及文本中的ID进行伪匿名化；按时间窗口对齐三个月前后数据；最终以行分隔JSON文件形式发布。

**📊 数据集**

使用的数据集为公开的Reddit数据（2015‑2023年）与Voat公共数据，结合25个审核干预事件，构成一份包含339,125名用户、39,028,732条评论的综合数据集。

**📈 对比分析**

该数据集可用于做干预前后活动量、活跃用户数、情感倾向、毒性度等指标的差异检验；支持Interrupted Time Series、Difference‑in‑Differences等准实验方法；亦可用于构建预测模型（如预测用户流失、迁移、行为改变）。报告中通过对比每个干预的pre/post评论/活跃度，发现大多数干预后两项指标均下降，说明数据可捕捉到干预效果；具体性能指标（如显著性水平、预测准确率）需研究者自行实验。

**⚠️ 局限性**

局限性包括：①平台与干预类型偏向Reddit和硬性禁令/隔离；②缺乏独立对照组，因果推断受限；③对禁令干预后空间内数据缺失，只能通过外部空间数据推断；④仅保留预干预活跃度≥10条的用户，低活跃用户被排除；⑤Bot过滤不完全，残留机器人可能影响结果；⑥数据跨年份，平台政策与用户行为演化导致时间漂移；⑦隐私与伦理约束限制对原始文本内容的深入分析。

---

## 168. AI Twin: Enhancing ESL Speaking Practice through AI Self-Clones of a Better Me

**arXiv ID:** 2601.11103 | [PDF](https://arxiv.org/pdf/2601.11103v1)

**作者:** Minju Park `[一作]` (University of British Columbia), Dongwook Yoon `[通讯]` (University of British Columbia)

**通讯引用:** 1430 | [OpenAlex ID](https://openalex.org/A5028316272)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估 AI Twin 系统，该系统在 ESL 口语练习中对学习者的发言进行重新表述，并以学习者的克隆声音播放，从而保持对话流畅并提供隐式反馈。

**💡 创新点**

创新点在于将 Ideal L2 Self（理想第二语言自我）理论与自我克隆（self‑clone）技术结合，利用学习者的“更好自我”语音模型来提升情感支持与动机，同时通过隐式重述取代传统显式纠错，减少对话中断。

**🔧 技术方法**

使用的技术包括：大型语言模型（LLM）用于生成更流畅的重述文本；语音识别（ASR）捕捉学习者语音；基于 ElevenLabs 的即时语音克隆与语音合成技术，用以生成学习者的克隆声音；以及对话生成模型为 AI 对话者提供回应。

**📊 数据集**

数据来源为实验中 20 名韩国成人 ESL 学习者产生的实时对话与音频样本，学习者提供的 30 秒语音样本用于生成克隆声音；未使用公开语料库，所有数据均为实验自生成。

**📈 对比分析**

通过 within‑subject 设计，对比 Explicit Feedback（显式纠错）、AI Proxy（重述+非个人化声音）和 AI Twin（重述+个人化克隆声音）。采用情绪、认知、行为三维参与问卷及半结构化访谈评估。结果显示：AI Twin 在情绪参与上显著优于显式纠错，且与 AI Proxy 相当；认知和行为维度差异不显著；实验未涉及长期学习成效评估。

**⚠️ 局限性**

限制包括：仅进行一次简短会话，未评估真实学习效果；样本仅为韩国成人 ESL 学习者，难以推广到不同文化/语言背景；技术依赖 LLM/ASR/语音合成，在低资源语言下可用性受限；实验规模小，未检验长期动机和持续使用情况。

---

## 169. Integrity Shield A System for Ethical AI Use & Authorship Transparency in Assessments

**arXiv ID:** 2601.11093 | [PDF](https://arxiv.org/pdf/2601.11093v1)

**作者:** Ashish Raj Shekhar `[一作]` (Arizona State University), Vivek Gupta `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了Integrity*Shield，一种在评估PDF中嵌入结构感知、项级不可见水印的系统，用以阻止和检测大语言模型完成试卷；

**💡 创新点**

创新点在于直接作用于PDF底层，结合无形文本、字形重映和遮罩技术，并通过LLM驱动的规划器为不同题型定制水印方案，既能高效阻断AI答题（91‑94%）又能可靠恢复作者身份（89‑93%），提供可解释的 AI 介入信号；

**🔧 技术方法**

使用的技术包括 PDF 水印引擎、LLM规划器、无形文本注入、CMap 字形重映、离屏遮罩、以及基于水印签名的作者身份检索与校准服务；

**📊 数据集**

实验数据集为30份涵盖 STEM、文学与医学推理的公开考试PDF，构建10份基准试卷用于评估；

**📈 对比分析**

与三种基线（ICW、GlyphPert、TrapDoc）对比，采用预防-ASR和检测指标，Integrity*Shield在四大商业 MLLM 上均实现了90‑94% 的阻断率和89‑93% 的检索率；

**⚠️ 局限性**

局限性包括仅在10份试卷和固定前沿 MLLM 上验证，需在机构掌控下使用 PDF，水印鲁棒性随模型与解析管道演进可能衰减，且作者身份得分仅为预警信号，不能直接判定违规。

---

## 170. Efficient Multilingual Name Type Classification Using Convolutional Networks

**arXiv ID:** 2601.11090 | [PDF](https://arxiv.org/pdf/2601.11090v1)

**作者:** Davor Lauc `[一作]` (University of Zagreb), Davor Lauc `[通讯]` (University of Zagreb)

**通讯引用:** 74 | [OpenAlex ID](https://openalex.org/A5075965117)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在CPU上高效运行的多语种姓名类型分类CNN模型Onomas‑CNN X，能够同时识别姓名所属语言和实体类型。

**💡 创新点**

创新点包括深度可分离卷积、并行多核卷积分支、学习权重的三种池化组合、分层分类结构以及针对CPU的量化与内存布局优化。

**🔧 技术方法**

采用卷积神经网络、深度可分离卷积、注意力池化、分层分类、焦点损失、INT8量化等技术。

**📊 数据集**

使用自建的1.5亿条姓名实例数据集，涵盖104种语言、四类实体，来源包括Wikidata、ORCID、Common Crawl等公开语料。

**📈 对比分析**

与XLM‑RoBERTa、FastText等基线对比，Onomas‑CNN X在随机测试中准确率92.1%（仅比XLM‑RoBERTa低0.8%），单核吞吐量2813样本/秒，比XLM‑RoBERTa快46倍，能耗降低46倍。

**⚠️ 局限性**

对低资源语言的表现仍有差距，分层分类先预测语言可能失败，词表固定难以即时适应新名字，缺乏跨语言语义共享，仍不及大规模预训练模型在跨语言迁移上的优势。

---

## 171. Assesing the Viability of Unsupervised Learning with Autoencoders for Predictive Maintenance in Helicopter Engines

**arXiv ID:** 2601.11154 | [PDF](https://arxiv.org/pdf/2601.11154v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 172. Visual Marker Search for Autonomous Drone Landing in Diverse Urban Environments

**arXiv ID:** 2601.11078 | [PDF](https://arxiv.org/pdf/2601.11078v1)

**作者:** Jiaohong Yao `[一作]` (Macquarie University), Yuankai Qi `[通讯]` (Macquarie University)

**通讯引用:** 4378 | [OpenAlex ID](https://openalex.org/A5070842891)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了基于AirSim的多样化城市环境仿真数据集，并评估了三种导航策略在视觉标记搜索和着陆任务中的表现。

**💡 创新点**

提供了系统的、可控的多环境仿真评估平台，结合多样的灯光、天气和城市布局，对标记搜索策略进行全面比较，揭示了环境因素对性能的影响。

**🔧 技术方法**

使用AirSim+Unreal Engine 4模拟环境，利用RGB和深度相机实现标记检测与障碍规避，采用启发式 Spiral/Zigzag 轨迹和基于深度感知的强化学习（PPO）策略。

**📊 数据集**

共966个仿真情景，涵盖 ModernCity、PostSoviet 和 UrbanDistrict 三个城市地图，并通过多时段、多天气条件生成161个独特的标记‑无人机组合。

**📈 对比分析**

通过成功率、导航误差、路径长度权衡（SPL）、碰撞率和误检率等指标比较 Spiral、Zigzag（2D/3D）与 E2E‑RL；3D 版本与 RL 在成功率和碰撞率上表现优于 2D 版，RL 虽成功率低但路径更高效、碰撞率为零。

**⚠️ 局限性**

RL 策略受限于 2D 深度感知导致垂直搜索不足，整体成功率低；基于轨迹的方法受制于预设路径，容易被障碍物或遮挡干扰；未考虑动态障碍、实时 GPS 与惯性导航协同，且模型仅在仿真中验证，缺乏真实世界测试。

---

## 173. Bridging Cognitive Neuroscience and Graph Intelligence: Hippocampus-Inspired Multi-View Hypergraph Learning for Web Finance Fraud

**arXiv ID:** 2601.11073 | [PDF](https://arxiv.org/pdf/2601.11073v1)

**作者:** Rongkun Cui `[一作]` (Tongji University), Qi Zhang `[通讯]` (Tongji University)

**通讯引用:** 9729 | [OpenAlex ID](https://openalex.org/A5100360250)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在网络金融场景中提出一种基于海马回路启发的多视角超图学习模型，用以检测在线欺诈交易。

**💡 创新点**

创新点包括：① 跨视角不一致感知模块，量化同一交易在不同视角下的身份、特征和标签差异；② CA1灵感的新颖性感知超图学习模块，通过局部方差自适应加权，增强对长尾稀疏欺诈样本的感知。

**🔧 技术方法**

采用超图构建、多视角嵌入、GNN消息传递与新颖性加权、注意力融合等技术，整体实现端到端的超图学习与欺诈判别。

**📊 数据集**

使用六个真实网络金融欺诈数据集：公开的S-FFSD、Sparkov，以及四个私人月度数据集（Private‑1~4）。

**📈 对比分析**

与15个SOTA基线（传统、深度、图神经网络等）在AUC、F1、AP三个指标上对比，平均提升分别为6.42%、9.74%和39.14%，在极度不平衡的数据集上也显著优于其他模型。

**⚠️ 局限性**

局限性：仅在网络金融交易数据上验证，尚未推广到其他欺诈场景（如假新闻、身份盗窃）；模型对超图构建参数（窗口大小、视角数）敏感，需进一步自动化调优。

---

## 174. H-AIM: Orchestrating LLMs, PDDL, and Behavior Trees for Hierarchical Multi-Robot Planning

**arXiv ID:** 2601.11063 | [PDF](https://arxiv.org/pdf/2601.11063v1)

**作者:** Haishan Zeng `[一作]` (University of Chinese Academy of Sciences), Peng Li `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 19949 | [OpenAlex ID](https://openalex.org/A5100432789)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一套名为H-AIM的分层多机器人规划框架，将LLM、PDDL规划与行为树结合，实现从自然语言指令到异构机器人团队的长时序任务自动规划与执行。

**💡 创新点**

采用三阶段级联架构，首次将LLM的语义解析、经典规划器的搜索与行为树的实时控制无缝衔接，并通过共享黑板支持动态异构队列协作。

**🔧 技术方法**

依托LLM（如GPT‑4o等）、PDDL符号规划器FastDownward、行为树编译器、LLM驱动的语义验证与合并、共享黑板通信等技术。

**📊 数据集**

新建MACE‑THOR基准，包含42个复杂家庭任务，涵盖8个室内布局，区分并行独立与时间依赖两类任务。

**📈 对比分析**

与最新基线LaMMA‑P（GPT‑4o）对比，在MACE‑THOR上任务成功率从12%提升至55%，目标条件召回率从32%提升至72%；并行任务SR 0.71/ GCR 0.88，时间依赖任务SR 0.38/ GCR 0.62。

**⚠️ 局限性**

假设环境完全可观测、仅在仿真验证，缺乏视觉感知与部分可观测下的鲁棒性，且性能受所用LLM推理能力限制。

---

## 175. GMM-COMET: Continual Source-Free Universal Domain Adaptation via a Mean Teacher and Gaussian Mixture Model-Based Pseudo-Labeling

**arXiv ID:** 2601.11161 | [PDF](https://arxiv.org/pdf/2601.11161v1)

**作者:** Pascal Schlachter `[一作]` (University of Stuttgart), Bin Yang `[通讯]` (University of Stuttgart)

**通讯引用:** 40107 | [OpenAlex ID](https://openalex.org/A5085036036)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在连续目标域流中实现无源通用域适应的模型自适应方法。

**💡 创新点**

创新点在于将GMM伪标签与Mean Teacher框架结合，并加入源一致性损失和学生‑教师一致性损失，显著抑制长期漂移。

**🔧 技术方法**

采用ResNet‑50 backbone，Gaussian Mixture Model伪标签，Mean Teacher EMA，contrastive loss、entropy loss、source consistency loss 和 student‑teacher consistency loss 等技术。

**📊 数据集**

在 DomainNet、CIFAR‑10‑C 与 CIFAR‑100‑C 三个公开数据集上进行实验。

**📈 对比分析**

与多种现有的SF‑UniDA与TTA方法比较，GMM‑COMET 在 PDA、ODA、OPDA 三种类别移位下在大多数域和类别组合上均优于或不劣于基准，持续提升性能。

**⚠️ 局限性**

在源类别已知且类别不变的简单场景下提升有限，且对动态类别迁移和未知类的处理仍有局限，未来需进一步扩展。

---

## 176. Theoretically and Practically Efficient Resistance Distance Computation on Large Graphs

**arXiv ID:** 2601.11159 | [PDF](https://arxiv.org/pdf/2601.11159v1)

**作者:** Yichun Yang `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7036 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出两种基于Lanczos迭代的算法——Lanczos Iteration（全局近线性时间）和Lanczos Push（局部子线性时间），用于高效近似无向图的电阻距离（Resistance Distance）。

**💡 创新点**

创新点在于：①将Lanczos方法用于电阻距离的矩阵函数求解，从而把对条件数κ的依赖从O(κ)降低到O(√κ)；②设计了一种子集Lanczos递推（subset Lanczos recurrence）实现局部求解，时间复杂度从O(κ³/ε²)降至O(κ².⁷⁵/ε)，并在理论与实验上证明其优越性。

**🔧 技术方法**

主要技术包括Lanczos迭代、Chebyshev多项式逼近、子集Lanczos递推（approximate matrix‑vector multiplication）、随机游走估计、谱分析与误差传播分析。

**📊 数据集**

实验使用八个真实网络数据集（4个社交网络、4个道路/基础设施网络）以及合成的Erdős–Rényi与Barabási–Albert图，覆盖从10²到10⁶顶点、1/ε从10⁻¹到10⁻⁵的多种规模与精度。

**📈 对比分析**

与Power Method、Laplacian Solver、TP、TPC、GEER、FastRD等六种最先进方法相比，Lanczos Iteration 在社交网络上速度提升5–10×、在道路网络上提升100×；Lanczos Push 在所有数据集上都比所有局部方法快5–50×，尤其在高κ道路网络上表现尤为突出。

**⚠️ 局限性**

局限性：理论时间复杂度仍包含上界常数C₁、C₂（实际表现更好但上界不紧）；局部算法依赖于对特征值区间的假设，且在某些极端图或大κ场景下可能需要调优阈值ε；虽然已证明Ω(κ)的下界，但上界与下界仍有显著差距，进一步优化仍是开放问题。

---

## 177. Konflux: Optimized Function Fusion for Serverless Applications

**arXiv ID:** 2601.11156 | [PDF](https://arxiv.org/pdf/2601.11156v1)

**作者:** Niklas Kowallik `[一作]` (TU Berlin), David Bermbach `[通讯]` (TU Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了 Konflux 框架，用于在本地模拟 FaaS 环境，能够对所有可能的函数融合配置进行穷举实验并收集成本与延迟数据。

**💡 创新点**

创新点在于：①通过本地化模拟显著降低了实验成本与时间；②实现了对融合配置空间的完整探索，提供了基于不同计费模型的最优融合策略；③基于实验结果提出新的融合启发式规则。

**🔧 技术方法**

使用 Golang 编写框架，Docker 容器化执行函数，利用 DAG 结构表示调用关系，结合 AWS Lambda 与 Google Cloud Run 的计费模型进行成本计算；采用 Lucas–Lehmer 生成均衡 CPU 负载。

**📊 数据集**

使用四个基于前人工作改编的示例 FaaS 应用，内部采用统一的计算密集型工作负载（LLT）进行实验；未使用公开大型真实数据集，而是以合成任务为主。

**📈 对比分析**

通过与未融合、默认资源配置的基线服务器无服务器部署进行对比，结果显示在成本与延迟上分别实现了约 40% 与 89% 的提升；在不同 α 权重和计费模型下，发现最优融合配置的差异与规律。

**⚠️ 局限性**

局限性包括：①仅支持内部函数调用的分析，无法处理外部网络连接；②实验规模受限，最多可覆盖 7 个函数，扩展性差；③未覆盖极少数事件与高负载场景，结果在真实大规模应用中的适用性待验证；④成本与时间随着函数数量呈指数增长。

---

## 178. Shape-morphing programming of soft materials on complex geometries via neural operator

**arXiv ID:** 2601.11126 | [PDF](https://arxiv.org/pdf/2601.11126v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 179. Do We Always Need Query-Level Workflows? Rethinking Agentic Workflow Generation for Multi-Agent Systems

**arXiv ID:** 2601.11147 | [PDF](https://arxiv.org/pdf/2601.11147v1)

**作者:** Zixu Wang `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种低成本的多智能体工作流生成框架，利用自我预测加少量校准执行的方式替代传统的完整执行评估；

**💡 创新点**

创新点在于证明查询级工作流生成并非必要，并提出通过自我预测与少量校准相结合的 surrogate 评估，显著降低 token 消耗同时保持性能；

**🔧 技术方法**

技术上使用大语言模型（Qwen-Plus 作为执行者、Qwen3-8B 作为优化器），实现自我预测与少样本校准、MCTS 风格搜索以及分阶段评估；

**📊 数据集**

实验数据集涵盖多跳推理（DROP、HotpotQA）、数学推理（GSM8K、MATH）、程序合成（HumanEval、MBPP）等六个 benchmark；

**📈 对比分析**

与现有任务级方法 Aflow、AgentPrune 以及查询级方法 ScoreFlow 对比，平均性能下降仅 0.61%，token 使用量从 54% 至 83% 下降，显示在不显著损失性能的前提下实现了显著成本压缩；

**⚠️ 局限性**

局限性包括自我预测仍受模型偏差影响，对极端少样本校准的稳定性尚未完全验证，并且在极大规模数据集上仍需进一步评估其可扩展性与泛化能力。

---

## 180. When "Likers'' Go Private: Engagement With Reputationally Risky Content on X

**arXiv ID:** 2601.11140 | [PDF](https://arxiv.org/pdf/2601.11140v1)

**作者:** Yuwei Chuai `[一作]` (University of Luxembourg), Nicolas Pröllochs `[通讯]` (JLU Giessen)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过观察性 Difference‑in‑Differences 分析与问卷实验，研究将点赞信息设为私有后，高声誉风险内容的点赞量是否会发生变化。

**💡 创新点**

创新点在于将平台层面隐私改动视为自然实验，结合量化的 DiD 与用户意愿调查，揭示了公开与私有点赞对行为与意愿的差距。

**🔧 技术方法**

主要技术包括 Difference‑in‑Differences、负二项回归、HonestDiD 估计、等效性检验、配对差异分析以及基于 Pew 数据的重加权。

**📊 数据集**

数据集包含 154,122 条来自 1068 个账号（政治、极端、科技、娱乐等）的帖子与点赞/转发记录，以及 203 名美国 Twitter 用户的问卷调查数据。

**📈 对比分析**

通过比较高低声誉风险账号以及点赞与转发的 DiD，结果显示平台级点赞量无显著提升；问卷实验显示在私有条件下的自愿点赞率略有提高，但群体平均差异不显著，表明意愿与行为存在缺口。

**⚠️ 局限性**

局限包括：① 关注的指标为聚合层面，难以捕捉单个用户行为细节；② 参与度高度集中在高活跃/机器人账号，可能掩盖普通用户的细微变化；③ 问卷自报可能存在社会期望偏差；④ 研究仅针对单一平台和文化语境，结果可推广性有限。

---

## 181. FSL-BDP: Federated Survival Learning with Bayesian Differential Privacy for Credit Risk Modeling

**arXiv ID:** 2601.11134 | [PDF](https://arxiv.org/pdf/2601.11134v1)

**作者:** Sultan Amed `[一作]` (Indian Institute of Management), Sayantan Banerjee `[通讯]` (Indian Statistical Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种联邦生存学习框架（FSL-BDP），在不集中存储个人借款人数据的情况下，联合多家金融机构学习时间到违约模型，并通过贝叶斯差分隐私（BDP）为模型更新提供正式隐私保证。

**💡 创新点**

创新点：①首次将贝叶斯差分隐私引入联邦生存分析；②使用离散时间生存模型区分早期与晚期违约，提供更具业务价值的风险时序信息；③发现隐私机制在联邦部署中的效能排名与中心化基准相反，强调需要在目标架构中评估隐私方案。

**🔧 技术方法**

技术：联邦学习（FedAvg）、离散时间生存模型、全连接神经网络、梯度裁剪与高斯噪声注入、贝叶斯DP（基于留一梯度估计与RDP合成）、RDP转(ε,δ)隐私计数、跨客户端加权聚合。

**📊 数据集**

数据集：美国P2P平台LendingClub（约160万笔）、美国SBA小企业贷款（约75万笔）、欧洲P2P平台Bondora（约22.8万笔），按自然地理边界划分为27、32、3个客户端。

**📈 对比分析**

比较方法：在中心化、联邦、无隐私、经典DP、BDP三种设置下，用C-index和Integrated Brier Score（IBS）评估训练集、测试集和OOT。结果显示：联邦训练无显著损失；在联邦下BDP的C-index普遍高于经典DP，IBS更低；在中心化下经典DP优于BDP，说明隐私机制在不同部署模式下表现逆转。

**⚠️ 局限性**

局限性：①假设所有客户端每轮全量参与且同步；②仅考虑水平联邦，未覆盖垂直或混合联邦；③未对分布漂移进行长期监控与周期性重训练；④对贝叶斯DP的蒙特卡洛估计与噪声调参依赖经验，可能影响实际部署；⑤预算管理与更新频率受隐私预算限制，尚无完整运维方案。

---

## 182. Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning

**arXiv ID:** 2601.11109 | [PDF](https://arxiv.org/pdf/2601.11109v1)

**作者:** Shaofeng Yin `[一作]` (University of California), Haiwen Feng `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个通过交互式生成-执行-渲染-比较-修正循环的视觉逆图形编码代理，能从单张图片或文本指令零样本重建或编辑3D/4D/2D场景。

**💡 创新点**

创新点在于将分析-合成循环与多模态推理结合，构建技能库和可扩展上下文记忆，使代理实现任务无关、模型无关的零样本逆图形重建。

**🔧 技术方法**

技术包括大语言模型代理、工具辅助的生成与验证、可编程的图形引擎（Blender）、基于上下文窗口的记忆管理以及可视化检验工具。

**📊 数据集**

使用了BlenderGym、SlideBench和自研的BlenderBench三大基准集进行评估。

**📈 对比分析**

与一键式基线及无记忆版BlenderAlchemy对比，平均提升BlenderGym 35.3%、SlideBench 117.2%以及BlenderBench 124.7%，尤其在小模型上显著弥补差距。

**⚠️ 局限性**

局限在于受到底层VLM能力和检验工具精度限制，长程推理会耗尽上下文窗口，需要更高级的记忆机制和更细粒度的几何检验。

---

## 183. More Human or More AI? Visualizing Human-AI Collaboration Disclosures in Journalistic News Production

**arXiv ID:** 2601.11072 | [PDF](https://arxiv.org/pdf/2601.11072v1)

**作者:** Amber Kusters `[一作]` (Centrum Wiskunde and Informatica), Abdallah El Ali `[通讯]` (Centrum Wiskunde and Informatica)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过共创会议生成了四种可视化披露方案（文本披露、基于角色的时间线、基于任务的时间线、聊天机器人），并在实验室中对32名受试者进行眼动跟踪、问卷和访谈实验，评估这些方案在新闻制作中展示人类–AI协作比例与细节的效果及其对读者感知的影响。

**💡 创新点**

创新点在于：①提出并系统化了人类–AI协作披露的设计空间；②首次将四种不同视觉化方法与协作比例关联起来进行对比研究；③揭示了披露设计会主动塑造读者对AI贡献的感知，并在不同协作情境下产生“放大”或“抑制”效果。

**🔧 技术方法**

技术方法包括：①共创设计（N=10）收集 69 份设计想法；②使用 JavaScript/React 开发四个原型；③在实验中结合眼动追踪（Tobii Pro Fusion）、累计链接混合效应模型（CLMM）和线性混合效应模型（LMM）进行定量分析；④半结构化访谈进行定性主题分析。

**📊 数据集**

数据集与材料：共创会议的设计草稿；实验使用四篇改编自 BBC 的新闻文本（分别为主要由人类撰写和主要由AI撰写的两篇），共 16 条刺激材料；受试者样本 N=32，包含多种 AI 文学素养水平。

**📈 对比分析**

比较方法：在 4×2（披露类型 × 协作比例）的受试内设计中，评估“人类–AI协作感知”“清晰度”“信息量”“任务步骤理解”等 Likert 量表，以及眼动指标（注视时长、注视次数、扫视次数）。实验结果显示：文本披露在所有指标上最低；聊天机器人在信息深度上最高但易用性最差；基于角色与基于任务的时间线在清晰度和任务步骤理解上最佳；不同披露会显著影响读者对 AI 角色的认知，形成“放大”或“抑制”效应。

**⚠️ 局限性**

局限性包括：①仅评估四种原型，未覆盖所有设计空间；②只针对文本新闻，未涉及多模态内容；③实验在实验室环境进行，生态有效性有限；④未邀请记者参与，缺乏工作流程可行性验证；⑤样本规模有限，且仅为一次性实验，无法观察长期曝光效应。

---

## 184. VLAgents: A Policy Server for Efficient VLA Inference

**arXiv ID:** 2601.11250 | [PDF](https://arxiv.org/pdf/2601.11250v1)

**作者:** Tobias Jülg `[一作]` (University of Technology Nuremberg), Wolfram Burgard `[通讯]` (University of Technology Nuremberg)

**通讯引用:** 68791 | [OpenAlex ID](https://openalex.org/A5084499878)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 VLAgents，面向 Vision‑Language‑Action 模型的统一、可扩展的策略服务器，支持在同机共享内存或跨机网络调用。

**💡 创新点**

创新点在于：① 将 Gymnasium‑style 接口与数据感知压缩（JPEG）结合；② 通过 RPyC 实现无缝的零拷贝共享内存与网络切换；③ 统一支持多种模型（OpenVLA、π_0、Diffusion Policy 等），降低集成门槛。

**🔧 技术方法**

技术包括 Python、GPyTorch/NumPy、RPyC、共享内存（POSIX / mmap）、JPEG 编码、Slurm CLI、Gymnasium API、视频录制工具。

**📊 数据集**

使用了 Maniskill‑3 仿真环境与 Robot Control Stack（RCS）实验平台进行评估，涉及四款真实机器人手臂和 MuJoCo 模拟，未公开专门的数据集。

**📈 对比分析**

通过与 OpenVLA、OpenPi、LeRobot 三大主流服务器在本机和网络环境下的往返时延（RTT）对比，VLAgents 在网络上达 220 Hz、局域网延迟 0.3 ms，速度是传统方案的约三倍，显著提升了并行仿真与真实硬件推理效率。

**⚠️ 局限性**

局限性包括：对非 Vision‑Language‑Action 任务的适配尚未充分验证；压缩方法依赖 JPEG，可能在极低延迟或高分辨率场景下仍有瓶颈；需要在不同机器间保证共享内存与网络协议的兼容性。

---

## 185. Epistemic Control and the Normativity of Machine Learning-Based Science

**arXiv ID:** 2601.11202 | [PDF](https://arxiv.org/pdf/2601.11202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 186. FactCorrector: A Graph-Inspired Approach to Long-Form Factuality Correction of Large Language Models

**arXiv ID:** 2601.11232 | [PDF](https://arxiv.org/pdf/2601.11232v1)

**作者:** Javier Carnerero-Cano `[一作]` (IBM Research Europe), Elizabeth Daly `[通讯]` (IBM Research Europe)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种后置纠错方法 FactCorrector，用结构化的事实性反馈对 LLM 生成的长文本进行改正，并构建了新的 Veli5 评测基准。

**💡 创新点**

创新点在于：① 结合 FactReasoner 的图模型，将检索到的证据与事实单元的蕴含/矛盾关系整体聚合；② 通过迭代式精细化模型将反馈嵌入提示，显著提升纠错准确率；③ 创建 Veli5 数据集，实现大规模、系统化的长文本事实性评估。

**🔧 技术方法**

技术方法包括：Atomizer、Reviser、Retriever、Evaluator 的 FactReasoner 流程；基于图模型的概率推理；结构化反馈与原始响应拼接的 refinement 模型（任意指令跟随 LLM）；可选的 LoRA SFT 训练；以及对比实验所用的多种公开 LLM。

**📊 数据集**

使用的数据集有：Veli5（来自 ELI5 的 17,522 例子，包含人工和人工+LLM 合成的答案）、Biographies（183 条 ChatGPT 生成的传记段落）、AskHistorians（200 条 r/AskHistorians 题解）和 Conflicts（100 条已知真实的命题），以及公开的标准长文本事实性数据集。

**📈 对比分析**

与 CRITIC、RAC、LLM1/LLM2 等后置纠错方法以及 LoRA SFT 进行对比。FactCorrector 在所有模型（Llama、Mixtral、Granite、GPT‑OSS）和数据集上均实现了显著提升：F1@K、精度、召回率、可验证性与完整性均得到正向增益，平均 F1@K 的提升约 0.3，且在多模型、多数据集上表现出强鲁棒性。

**⚠️ 局限性**

局限性包括：① 对 Atomizer/ Reviser 的提示敏感，分词粒度限制在一次性句子级；② Retriever 的查询生成与上下文质量对结果影响大；③ Evaluator 的关系抽取需要多轮 LLM 调用，计算开销高（O(n·m)）；④ 仅在开放源模型上验证，未覆盖闭源大模型；⑤ 仅支持单次修正迭代，无法处理更细粒度的多层次纠错。

---

## 187. Language of Thought Shapes Output Diversity in Large Language Models

**arXiv ID:** 2601.11227 | [PDF](https://arxiv.org/pdf/2601.11227v1)

**作者:** Shaoyang Xu `[一作]` (Singapore University of Technology and Design), Wenxuan Zhang `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5100629634)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在大语言模型的中间推理阶段控制使用的思考语言（从英语切换到多种非英语语言或混合多语言），提升模型生成文本的多样性。

**💡 创新点**

发现不同思考语言对应模型思维空间的不同几何区域，且使用非英语或混合多语言的思考方式能系统性地提高多样性，为多样性控制提供了结构化的新的轴向。

**🔧 技术方法**

利用语言控制前缀对Qwen3系列和DeepSeek等LLM进行多语言推理，实施单语采样与混合语采样，采用Distinct Score、Similarity Score和质量评估，并用文化多样性熵评估多文化覆盖度。

**📊 数据集**

使用NoveltyBench、Infinity-Chat评估多样性；Blend、WVS评估文化知识与价值多样性；并在15种不同语言的提示和思考语言上进行实验。

**📈 对比分析**

与英语采样、高温采样、显式多样性请求以及多语言提示等方法比较，混合语言采样在Distinct Score上平均提升约8–10点，在文化多样性熵上显著优于其他策略。

**⚠️ 局限性**

一方面跨语言对齐可能削弱非英语语言所带来的多样性；另一方面文化多样性评估仅基于熵，未能覆盖真实部署场景中对特定文化价值的约束。

---

## 188. Game Accessibility Through Shared Control for People With Upper-Limb Impairments

**arXiv ID:** 2601.11218 | [PDF](https://arxiv.org/pdf/2601.11218v1)

**作者:** Sergio Mascetti `[一作]` (Università degli Studi di Milano), Dragan Ahmetovic `[通讯]` (Università degli Studi di Milano)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过构建GamePals框架，研究了对上肢障碍者在玩Rocket League时使用人类合作与部分自动化两种共享控制方式的可行性和体验，并对两种方式进行了对比评估。

**💡 创新点**

创新点在于（1）提出了可通用的GamePals共享控制框架，支持任意第三方游戏的双人/多人共享控制；（2）首次在同一实验中同时比较人类合作与部分自动化的实际使用效果；（3）发现并归纳了共享控制中的“协作混淆”现象，为后续设计提供了指导。

**🔧 技术方法**

采用了Xbox控制器、Xbox Adaptive Controller、BakkesMod游戏状态读取、基于Necto的游戏代理、输入重映射、命令解释器与仲裁器等技术实现共享控制；实验使用Rocket League 5分钟单局。

**📊 数据集**

数据集包括13名上肢障碍参与者的游戏日志、屏幕录像、音频记录、问卷调查结果和半结构化访谈文本。

**📈 对比分析**

通过对比人类合作与部分自动化的玩家自评工作负荷、投入度、满意度，以及定性分析玩家的协作策略与误解情况，结果显示两者都能使参与者完成游戏，但部分自动化更能提升自主性，且在高难度控制上表现更佳；人类合作则提供更丰富的沟通与策略协商。

**⚠️ 局限性**

主要局限包括样本规模有限、仅选用单一游戏（Rocket League）、实验仅使用一名人类副驾驶且缺乏长时间跟踪，且部分自动化代理缺乏即时交互与自适应能力，导致部分玩家出现协作混淆。

---

## 189. FAQ: Mitigating Quantization Error via Regenerating Calibration Data with Family-Aware Quantization

**arXiv ID:** 2601.11200 | [PDF](https://arxiv.org/pdf/2601.11200v1)

**作者:** Haiyang Xiao `[一作]` (Alibaba Cloud Computing), Yuewei Zhang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FAQ 框架，通过在同一 LLM 家族中使用更强大的“长辈同辈”模型重生成高质量校准数据，提升 PTQ 的量化效果。

**💡 创新点**

创新点在于利用家族先验与链式推理生成校准样本，并通过专家指导的样本竞争与规范化，形成数据驱动的 PTQ 优化方法，首次把模型家族知识注入量化校准。

**🔧 技术方法**

使用技术包括后训练量化（PTQ）、大语言模型 Prompting 与 CoT 生成、激活分布对齐、样本质量评估与选择、以及与 GPTQ、AWQ、SPQR、GPTAQ 等现有量化算法的无缝结合。

**📊 数据集**

实验数据集涵盖语言建模（Wikitext2、C4、LAMBADA）、12项通用推理与多语言任务（ARC、BoolQ、Hellaswag 等）、数学/代码基准（AIME、MATH‑500、LiveCodeBench），以及 Qwen3 系列 dense 与 MoE 模型。

**📈 对比分析**

在 INT8 与 INT4 两种位宽下，FAQ 作为插件式提升，在多种 PTQ 方法上平均减少 28.5% 的准确率损失；在 perplexity、推理任务和专业基准上均表现出显著提升，且提升幅度随模型规模与稀疏性不同而保持一致。

**⚠️ 局限性**

局限性包括对重生成策略（温度、候选数等）的敏感性；在极端 INT4 场景下偶有边缘任务性能波动；需要同族大模型作为教师，可能对跨家族或跨架构的通用性有待验证。

---

## 190. SD-RAG: A Prompt-Injection-Resilient Framework for Selective Disclosure in Retrieval-Augmented Generation

**arXiv ID:** 2601.11199 | [PDF](https://arxiv.org/pdf/2601.11199v1)

**作者:** Aiman Al Masoud `[一作]` (University of Pavia), Antonino Nocera `[通讯]` (University of Pavia)

**通讯引用:** 1203 | [OpenAlex ID](https://openalex.org/A5085793134)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在检索增强生成（RAG）中实现选择性披露的系统SD‑RAG，先对检索到的内容进行去敏化，再交给生成模型，避免提示注入导致的敏感信息泄露。

**💡 创新点**

创新点在于：1) 将安全与隐私约束的执行从生成阶段分离，放在检索后去敏化阶段；2) 通过图结构将自然语言的动态隐私约束编码进语料库，实现细粒度、可扩展的约束绑定；3) 采用提取式与改写式两种生成式去敏化范式，并配合重排序机制提升约束匹配。

**🔧 技术方法**

核心技术包括：检索增强生成、基于文本嵌入的语义相似度计算、图模型的节点/边表示、自然语言处理的生成式去敏化（prompt‑engineering）、重排序算法（平均/加权相似度）以及LLM微调/元提示策略。

**📊 数据集**

使用作者自行构造的合成数据集——20篇包含半敏感信息的短文，每篇配有约束、问答对，约12,600词。

**📈 对比分析**

与单体“全量约束”式基线对比（Qwen2.5‑7B、Llama‑3‑8B），在隐私得分上提升最高58%，在对抗提示注入时仍保持较高隐私；完整性得分略低（约10%）。实验表明提取式去敏化在隐私上最强，改写式更快。

**⚠️ 局限性**

主要局限：① 依赖未被毒化的检索语料；② 仅在小型开源LLM上验证，缺乏大模型/闭源系统的评估；③ 假设攻击者无对语料库先验知识，未研究多轮查询的去识别攻击。

---

## 191. TimeMar: Multi-Scale Autoregressive Modeling for Unconditional Time Series Generation

**arXiv ID:** 2601.11184 | [PDF](https://arxiv.org/pdf/2601.11184v1)

**作者:** Xiangyu Xu `[一作]` (East China Normal University), Jilin Hu `[通讯]` (East China Normal University)

**通讯引用:** 1035 | [OpenAlex ID](https://openalex.org/A5020559625)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了结构解耦的多尺度自回归生成框架TimeMAR，用于无条件时间序列生成。

**💡 创新点**

创新点包括：双路径VQ‑VAE实现趋势与季节分离，粗到细自回归生成策略，以及粗季节引导的细节重建。

**🔧 技术方法**

使用技术包括多尺度VQ‑VAE、双路径编码、频域注意力机制、粗细季节引导以及GPT风格Transformer自回归模型。

**📊 数据集**

实验数据集涵盖六个，真实数据有Stocks、ETTh、Energy、fMRI，合成数据有Sines和MuJoCo。

**📈 对比分析**

与GAN、VAE、扩散模型及SDformer等基线对比，TimeMAR在Discriminative Score和Context‑FID指标上均表现显著优于其它模型，且在小模型(<6M参数)时与大型模型相当，并能保持长序列生成的稳定性。

**⚠️ 局限性**

局限性在于对时序的分解仍需经验式阈值，且模型目前仅支持无条件生成，缺乏条件生成、缺失值填补等更广泛的应用功能。

---

## 192. Sample-Near-Optimal Agnostic Boosting with Improved Running Time

**arXiv ID:** 2601.11265 | [PDF](https://arxiv.org/pdf/2601.11265v1)

**作者:** Arthur da Cunha `[一作]` (Aarhus University), Andrea Paudice `[通讯]` (Aarhus University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出一种新的无偏(agnostic)提升算法，该算法在样本复杂度上接近理论下界，并且在样本量固定时具有多项式时间复杂度。

**💡 创新点**

创新点在于首次将近乎最优样本复杂度与多项式运行时间结合，解决了以往仅样本复杂度优秀但调用弱学习器指数级别的缺陷；同时通过引入“pruning”技术将AdaBoost产生的大量弱学习器压缩到与VC维度相关的规模。

**🔧 技术方法**

技术方法包括：
- 将数据划分为训练/验证两部分，先用弱学习器生成多组弱模型，再在验证集上寻找最优组合；
- 采用AdaBoost变体产生具有统一优势θ的弱模型，并利用margin理论实现高边际损失零点；
- 通过统一收敛和Bernstein偏差不等式证明所得到组合的风险与最佳可达风险差距可控制；
- 采用基类和其对偶VC维度对算法调用次数和运行时间进行分析；
- 使用多次弱学习器调用与放大技术保证在随机子样本上获得所需优势。

**📊 数据集**

论文主要是理论研究，没有给出具体的数据集或实验评估；所有结论均基于概率与VC理论分析。

**📈 对比分析**

与之前的几种无偏提升方法相比：
- 样本复杂度与已知下界几乎一致（O*(1/θ²)），
- 与此前实现多项式样本但指数时间的算法相比，改进为多项式时间；
- 与一些基于不同弱学习器定义的高效提升算法相比，本文在样本复杂度上更接近最优，但仍保留对弱学习器优势θ的依赖。

**⚠️ 局限性**

局限性包括：
- 仍需假设弱学习器在给定参数下具有足够优势θ；
- 计算复杂度与基类的对偶VC维度相关，若该维度较大会导致较高的计算量；
- 只给出理论保证，未在实测数据集上验证实际性能；
- 对弱学习器的实现细节依赖于可访问的随机种子与重抽样，实际应用中可能需要额外实现工作。

---

## 193. Rate-Distortion-Perception Tradeoff for the Gray-Wyner Problem

**arXiv ID:** 2601.11257 | [PDF](https://arxiv.org/pdf/2601.11257v1)

**作者:** Yu Yang `[一作]` (Southern University of Science and Technology), Lin Zhou `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 2975 | [OpenAlex ID](https://openalex.org/A5058899630)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文提出并证明了在Gray-Wyner多终端压缩模型中，加入失真与感知约束后第一阶渐近下的速率-失真-感知（RDP）可达区域；通过随机圆形移位操作实现感知约束并消除公共随机性，证明了确定性编码即可实现最优分布。

**💡 创新点**

创新点在于：①将RDP理论从点到点扩展到Gray-Wyner多终端网络；②引入条件RDP函数与公共信息互信息结构，揭示感知约束下的分离结构；③使用随机圆形移位直接嵌入编码/解码，简化感知分析；④证明公共随机性在第一阶上可被消除，确定性编码仍最优。

**🔧 技术方法**

主要技术包括：随机编码与典型集论证；随机圆形移位算子与感知约束的联合处理；条件RDP函数与信息量取值的最小化；源模拟技术消除公共随机性；典型性误差分析与收敛性证明。

**📊 数据集**

该工作为纯理论分析，未使用具体数据集；所有结论基于有限字母系统的记忆无关源模型。

**📈 对比分析**

与现有的Gray‑Wyner失真区域对比，本文在感知约束下给出完整的速率三元组；当感知阈值趋于无穷时可恢复传统失真区域；实验与数值对比未给出，仅通过理论证明实现最优。

**⚠️ 局限性**

局限性包括：仅给出第一阶渐近结果；仅适用于有限字母系统，未讨论Gaussian或连续模型；缺乏二阶/有限块长度分析；实现复杂度及实际编码器设计未给出。

---

## 194. Language-Agnostic Visual Embeddings for Cross-Script Handwriting Retrieval

**arXiv ID:** 2601.11248 | [PDF](https://arxiv.org/pdf/2601.11248v1)

**作者:** Fangke Chen `[一作]` (Zhejiang University), Yining Chen `[通讯]` (Zhejiang University)

**通讯引用:** 4009 | [OpenAlex ID](https://openalex.org/A5066483084)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量化的非对称双编码器框架，用于跨语言的语义不变手写体检索。

**💡 创新点**

创新点包括：利用部分冻结的多语文本编码器作为语义锚点，结合实例级对比和类别级语义一致性损失，构建统一且对书写风格与语言无关的嵌入空间；同时采用“先合成后实景”两阶段训练与量化加速实现边缘设备部署。

**🔧 技术方法**

技术主要有：双塔（视觉+文本）结构、InfoNCE 双向对比损失、类别一致性对齐损失、部分冻结 DistilBERT 文本编码器、MobileNetV3-Small 视觉编码器、量化（int8）推理。

**📊 数据集**

使用的主要数据集为英语（IAM）、中文（HWDB1.0）以及通过合成字体生成的跨语言样本，评估涵盖 In‑Domain 与 Out‑of‑Domain（字体风格迁移）情景。

**📈 对比分析**

与28个基线（OCR、嵌入式方法、VLLM 等）以及 Qwen、InternVL 等大型 VLLM 进行对比；在同语言检索中实现 Acc@1 最高 0.8605，跨语言检索 Acc@1 最高 90.98%；参数量和延迟分别比大型 VLLM 降低 10‑100 倍，性能保持接近或优于基线。

**⚠️ 局限性**

局限性主要是：仍依赖手工标注的语义 ID；在完全野外手写数据上的评估有限；跨语言性能主要基于英语、中文、西班牙三种语言，其他语言的泛化尚未验证。

---

## 195. Image-Text Knowledge Modeling for Unsupervised Multi-Scenario Person Re-Identification

**arXiv ID:** 2601.11243 | [PDF](https://arxiv.org/pdf/2601.11243v1)

**作者:** Zhiqi Pang `[一作]` (Harbin Institute of Technology), Gaurav Sharma `[通讯]` (University of Rochester)

**通讯引用:** 17435 | [OpenAlex ID](https://openalex.org/A5100705959)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对未标注的多场景行人重识别任务（UMS-ReID），提出了三阶段的图像-文本知识建模框架 ITKM，用场景嵌入、跨场景分离损失、群组级与实例级异构匹配以及动态文本表示更新，构建单一模型同时兼容可见-红外、服装变化、分辨率变化等多种场景。

**💡 创新点**

创新点包括：①首次定义并解决多场景行人重识别任务；②在 CLIP 视觉语言模型上引入场景嵌入与多场景分离损失，提升模型对不同场景的自适应；③设计群组级异构匹配（CHM）和实例级异构匹配（IHM）获取可靠的跨模态正样本；④提出动态文本表示更新（DRU）保持图像与文本监督的一致性。

**🔧 技术方法**

主要技术：预训练 CLIP 模型（图像编码器+文本编码器）；Transformer 前端双分支加场景嵌入；聚类+对比损失（同场景与异构对比）；多场景分离损失；动态文本表示更新；群组级与实例级异构匹配算法；整体采用无监督学习框架。

**📊 数据集**

使用公开多场景数据集：SYSU-MM01（可见-红外），LTCC（服装变化），MLR-CUHK03（低分辨率-高分辨率）进行评估；在这三大数据集上进行单场景与多场景联合训练。

**📈 对比分析**

与现有无监督传统（UT-ReID）和无监督场景专属（USS-ReID）方法相比，ITKM 在单场景训练下已接近或优于场景专属方法；在多场景联合训练时，ITKM(S) 与 ITKM(M) 的 Rank‑1 与 mAP 均明显高于对比方法（如 SDCL、TokenMatcher 等），显示出良好的跨场景泛化与性能提升。

**⚠️ 局限性**

限制与待改进：①实验仅在三大数据集上验证，未涉及更大规模或多种跨模态组合；②模型参数与训练规模未做最优调优，可能存在进一步提升空间；③对极端场景（如极低分辨率、极大光照变化）效果仍待评估。

---

## 196. How DDAIR you? Disambiguated Data Augmentation for Intent Recognition

**arXiv ID:** 2601.11234 | [PDF](https://arxiv.org/pdf/2601.11234v1)

**作者:** Galo Castillo-López `[一作]` (Universite Paris-Saclay), Gaël de Chalendar `[通讯]` (Universite Paris-Saclay)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种迭代式数据增强方法 DDAIR，用于在意图识别任务中检测并重新生成 LLM 生成的含糊示例，提升合成数据质量。

**💡 创新点**

创新点在于结合句子变换器（sentence transformers）与 LLM，自动识别与目标意图不匹配的合成句子，并通过多轮重新生成来降低含糊率；方法可在少样本场景下迭代改进。

**🔧 技术方法**

技术包括：基于 BGE、MPNet、MiniLM-L6 的句子编码器计算意图中心向量，判定含糊示例；使用 Mistral‑7B 与 Llama‑3‑8B 进行 in‑context 学习式生成；多轮检测‑重生成迭代；BERTBASE 进行意图分类微调。

**📊 数据集**

实验数据集涵盖 BANKING77、CLINC150 以及 MPGT（多标签医学对话）三大意图检测语料。

**📈 对比分析**

与未做去歧义处理的基线相比，DDAIR 在 BANKING77、CLINC150 的 2‑shot 与 5‑shot 设置下平均宏 F1 提升 1–3 分；在 MPGT 上，采用去歧义策略可提升 3–6 分；Silhouette 系数与含糊率均显著下降。

**⚠️ 局限性**

局限性包括：依赖句子编码器质量；含糊判定基于中心向量，易受离群点影响；迭代过程增加 LLM 调用次数，计算成本上升；仅在少样本设置验证，1‑shot 情况下效果未知。

---

## 197. Operator learning on domain boundary through combining fundamental solution-based artificial data and boundary integral techniques

**arXiv ID:** 2601.11222 | [PDF](https://arxiv.org/pdf/2601.11222v1)

**作者:** Haochen Wu `[一作]` (Chinese Academy of Sciences), Benzhuo Lu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 2744 | [OpenAlex ID](https://openalex.org/A5112452186)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于边界数据的神经算子学习框架MAD‑BNO，用于线性PDE的解算子学习。

**💡 创新点**

创新点在于只使用边界Dirichlet–Neumann数据训练网络，并通过MATHEMATICAL ARTIFICIAL DATA (MAD) 生成精确物理一致的合成数据，消除对内部采样与实验/数值数据的需求。

**🔧 技术方法**

技术包括边界积分表述、基于基本解的合成数据生成、全连接线性网络（400×400）以及传统数值积分方法。

**📊 数据集**

使用由基本解线性组合构造的合成边界条件数据集，包含数千个训练样本并在不同几何与频率下测试。

**📈 对比分析**

与PI‑DeepONet和MAD‑DeepONet比较，MAD‑BNO在Laplace、Poisson和Helmholtz三类问题上训练时间缩短约70‑95%，准确率提升1‑2个数量级，尤其在高频/复杂边界时表现更稳健。

**⚠️ 局限性**

局限在于仅适用于具有已知基本解的线性PDE，且推断阶段仍需显式边界/体积分，需借助FMM或其他加速；3D积分与高维积分实现尚未完成。

---

## 198. Policy-Based Deep Reinforcement Learning Hyperheuristics for Job-Shop Scheduling Problems

**arXiv ID:** 2601.11189 | [PDF](https://arxiv.org/pdf/2601.11189v1)

**作者:** Sofiene Lassoued `[一作]` (South Westphalia University of Applied Sciences), Andreas Schwung `[通讯]` (South Westphalia University of Applied Sciences)

**通讯引用:** 1142 | [OpenAlex ID](https://openalex.org/A5025397538)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于策略的深度强化学习超启发式框架，用于求解作业车间调度问题（JSSP），并在框架中加入动作预过滤和承诺机制。

**💡 创新点**

创新点包括：①利用时序彩色 Petri 网的 guard 进行动作预过滤，确保低级启发式只在可行动作上评估；②引入承诺（commitment）机制，在固定步数内保持同一启发式，提升信用分配和学习稳定性；③结合超启发式与可解释的低级启发式，提供多层次可解释性。

**🔧 技术方法**

使用的技术：Proximal Policy Optimization（PPO）强化学习算法；时序彩色 Petri 网模型用于环境建模与动作遮蔽；策略网络（policy network）与价值网络；动作预过滤与承诺机制实现；Petri 网可视化用于解释决策。

**📊 数据集**

实验数据集为标准 Taillard benchmark，包含 15×15 至 100×20 的作业数/机台数实例（共 8 组，每组 10 个实例）。

**📈 对比分析**

对比方法包括：传统调度规则（FIFO、SPT、SPS、MTWR 等）、经典元启发式（Tabu、GA、遗传+SA 等）和最近的学习方法（GIN、GAM、DGERD、MPPO）。在所有实例上，本方法平均最小化 makespan 为 2860 步，较最佳单一启发式提升约 1.6%，较所有启发式平均提升约 4%，并在大多数实例上击败所有比较算法。

**⚠️ 局限性**

局限性：①动作概率在训练早期易出现主导，导致多样性探索受限；②承诺长度的选择需经验调参，过短或过长均可能降低性能；③仍依赖手工设计的低级启发式，无法完全自适应所有场景；④策略网络本身仍为黑盒，整体决策过程仍存在一定的不可解释性。

---

## 199. TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech

**arXiv ID:** 2601.11178 | [PDF](https://arxiv.org/pdf/2601.11178v1)

**作者:** Girish A. Koushik `[一作]`, Diptesh Kanojia `[通讯]` (Association for the Advancement of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

制定并说明了AAAI会议提交的匿名提交和正式稿件的格式化规范。

**💡 创新点**

将匿名提交的要求与正式稿件统一，并对元数据清理和版权声明进行了详细说明，确保稿件匿名性与合规性。

**🔧 技术方法**

使用LaTeX排版、metadata‑cleaning工具、aaai2026.sty与aaai2026.bst文件等。

**📊 数据集**

无具体数据集，内容为格式化指引。

**📈 对比分析**

无实验对比，主要是对比匿名与正式稿件的差异；未提供性能指标。

**⚠️ 局限性**

适用范围仅限AAAI 2026会议，缺乏对其他会议或软件的支持；若使用非支持软件则可能出现排版错误。

---

## 200. Rank4Gen: RAG-Preference-Aligned Document Set Selection and Ranking

**arXiv ID:** 2601.11273 | [PDF](https://arxiv.org/pdf/2601.11273v1)

**作者:** Yongqi Fan `[一作]` (East China University of Science and Technology), Tong Ruan `[通讯]` (East China University of Science and Technology)

**通讯引用:** 1163 | [OpenAlex ID](https://openalex.org/A5005820786)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向生成器的检索排序模型Rank4Gen，并构建了PRISM数据集

**💡 创新点**

创新点在于将排序目标从传统检索相关性转向下游生成质量，并对不同生成器进行条件化的偏好建模

**🔧 技术方法**

使用双阶段训练：先进行基于相关性的监督微调，再通过Direct Preference Optimization（DPO）实现生成器偏好对齐，并采用双模式推理（ID与ID+内容）

**📊 数据集**

利用多语料库（HotpotQA、2WikiMultiHopQA、MUSIQUE、MS MARCO、CRUD-RAG）构建PRISM，覆盖141k问答，随后在五个RAG基准上评估

**📈 对比分析**

与点式、列表式、集合式等传统与LLM基排序方法对比，Rank4Gen在五个基准上在EM/F1上均实现或接近最佳效果，尤其在不同生成器间表现更稳定

**⚠️ 局限性**

主要局限包括仅使用PRISM子集PRISM_13k训练、偏好优化可能导致EM下降、构造过程依赖已标注正负文档以及生成器元信息的自动生成可能存在小幅幻觉

---

## 201. X-Distill: Cross-Architecture Vision Distillation for Visuomotor Learning

**arXiv ID:** 2601.11269 | [PDF](https://arxiv.org/pdf/2601.11269v1)

**作者:** Maanping Shao `[一作]` (Tsinghua University), Huazhe Xu `[通讯]` (Tsinghua University)

**通讯引用:** 2655 | [OpenAlex ID](https://openalex.org/A5049093671)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过将大型Vision Transformer DINOv2的知识蒸馏到轻量级ResNet-18，实现了在数据稀缺场景下高效的视觉编码器；

**💡 创新点**

创新点在于跨架构蒸馏，将ViT的开放世界语义知识与CNN的本征偏置结合，形成兼具泛化与样本效率的编码器；

**🔧 技术方法**

采用冻结的DINOv2教师与MSE蒸馏损失训练ResNet-18，再联合扩散策略头进行端到端微调；

**📊 数据集**

蒸馏阶段使用ImageNet-1K，后续下游任务在MetaWorld、Adroit、DexArt三大模拟基准以及5个真实桌面操作任务中进行；

**📈 对比分析**

与ResNet从零训练、ViT预训练、Depth-Anything、Theia等同参数视觉基线以及PointNet-DP3、π₀等高级模型相比，X‑Distill在34个模拟任务和5个真实任务上均实现了最高的成功率，尤其在OOD和长周期任务上表现突出；

**⚠️ 局限性**

局限在于仅采用特征级蒸馏，未对中间层对齐或多模态教师进行探索，且在数据量大或动态任务中的可扩展性待进一步验证。

---

## 202. Skill-Aware Diffusion for Generalizable Robotic Manipulation

**arXiv ID:** 2601.11266 | [PDF](https://arxiv.org/pdf/2601.11266v1)

**作者:** Aoshen Huang `[一作]` (Shandong University), Wei Zhang `[通讯]` (Shandong University)

**通讯引用:** 48232 | [OpenAlex ID](https://openalex.org/A5100675809)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种 Skill-Aware Diffusion (SADiff) 模型，利用技能级信息对机器人操作任务进行通用建模，并生成对象中心的运动流。

**💡 创新点**

创新点在于通过可学习的技能 token 构建技能感知编码模块，将技能信息显式注入扩散过程，并提出技能检索变换策略以利用技能特定轨迹先验改进 2D 运动流到 3D 行动的映射。

**🔧 技术方法**

使用扩散模型生成运动流、技能感知编码、技能检索变换，以及基于 IsaacSim 的仿真与真实硬件实验技术。

**📊 数据集**

采用了 IsaacSkill 高保真数据集，该数据集包含多种基础机器人技能，以及其他公开仿真/实物数据。

**📈 对比分析**

与传统通过扩大数据或网络规模提升泛化的基线进行比较，实验结果显示 SADiff 在多任务仿真与真实环境下具有更高的成功率和更好的泛化性能，尤其在新任务与不同对象尺寸的迁移中表现突出。

**⚠️ 局限性**

局限性包括对极端复杂或非标任务仍需大量技能示例；扩散模型的推理速度相对较慢，实时性能有限；以及缺乏对动态环境中鲁棒性的全面评估。

---

## 203. Beyond Model Scaling: Test-Time Intervention for Efficient Deep Reasoning

**arXiv ID:** 2601.11252 | [PDF](https://arxiv.org/pdf/2601.11252v1)

**作者:** Qianyue Wang `[一作]` (South China University of Technology), Mingkui Tan `[通讯]` (South China University of Technology)

**通讯引用:** 14075 | [OpenAlex ID](https://openalex.org/A5032352025)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种交互式推理框架Think-with-Me，利用过渡连词作为外部反馈的插入点，使大型推理模型能够在推理过程中主动暂停并根据外部（人类或LLM代理）反馈决定是否继续或终止，从而显著减少过度思考和过度越过。

**💡 创新点**

创新点在于将多源外部反馈与推理过程结合，采用过渡连词触发点并通过GRPO训练使模型具备对反馈做出即时决策的能力，实现动态推理长度控制。

**🔧 技术方法**

使用了Group Relative Policy Optimization（GRPO）强化学习配合LoRA微调、基于合理性和完整性评估的外部反馈、特殊标记包装反馈以及信息论分析等技术。

**📊 数据集**

在多任务、多难度的数据集上进行实验，包括MATH、GSM8K、AIME24/25、GPQA Diamond、LiveCodeBench，并在IFEval、SEP、LitBench等安全/创意任务上进行案例验证。

**📈 对比分析**

与多种基线（Qwen2.5-72B-Instruct、Qwen2.5-Math-72B、DeepSeek-R1-7B/14B/32B、QwQ-32B、DEER、SEAL、Speculative Thinking等）在8K/32K窗口下对比，实验显示在8K窗口下准确率与传统模型相当甚至提升（AIME24提升7.19%），同时思考长度平均缩短81%或以上。

**⚠️ 局限性**

局限性包括对外部反馈质量和时延敏感，特别是人工反馈需人工投入；在极难或高风险任务中鲁棒性有限，且过渡连词触发策略需针对不同任务进一步微调；大规模部署时需保证反馈代理的一致性和可扩展性。

---

## 204. MultiCaption: Detecting disinformation using multilingual visual claims

**arXiv ID:** 2601.11220 | [PDF](https://arxiv.org/pdf/2601.11220v1)

**作者:** Rafael Martins Frade `[一作]` (University of Santiago de Compostela), Arkaitz Zubiaga `[通讯]` (Queen Mary University of London)

**通讯引用:** 6571 | [OpenAlex ID](https://openalex.org/A5071220716)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了MultiCaption数据集并在其上对多语言视觉声明对比任务进行实验。

**💡 创新点**

首次公开64语言、11088对视觉声明对抗的多语言数据集，并利用claim‑post链接和自扩展等策略大幅提升样本规模。

**🔧 技术方法**

采用Transformer分类器、NLI模型以及大语言模型（如Mistral、Llama‑3等），并用GPT‑5生成对抗样本进行数据扩充与模型微调。

**📊 数据集**

主要使用基于MultiClaim的MultiCaption（11,088对）和COSMOS作为对比基准。

**📈 对比分析**

在MultiCaption测试集上，Fine‑tuned Mistral取得F1≈0.912，mDeBERTa≈0.90；在COSMOS上Fine‑tuned NLI和Transformer表现低，Fine‑tuned LLM略优；多语言训练相较于单语训练提升约8%。

**⚠️ 局限性**

存在LLM标注误差、仅基于文本缺乏视觉信息、需原始上下文且语言分布不均衡等限制。

---

## 205. SDFLoRA: Selective Dual-Module LoRA for Federated Fine-tuning with Heterogeneous Clients

**arXiv ID:** 2601.11219 | [PDF](https://arxiv.org/pdf/2601.11219v1)

**作者:** Zhikang Shen `[一作]` (Zhejiang University), Jianhai Chen `[通讯]` (Zhejiang University)

**通讯引用:** 2324 | [OpenAlex ID](https://openalex.org/A5015845819)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SDFLoRA框架，在联邦学习中对大语言模型进行LoRA参数高效微调，解决rank异质性与隐私保护问题。

**💡 创新点**

创新点在于将LoRA适配器拆分为全局共享模块与本地私有模块，仅对全局模块进行选择性堆叠聚合并注入DP噪声，从而兼顾可迁移知识与个性化，提升对rank异质性和DP噪声的鲁棒性。

**🔧 技术方法**

使用技术包括LoRA、选择性堆叠聚合、低秩重压缩、差分隐私DP‑SGD、联邦学习（FedAvg、FedProx等）和基于梯度裁剪的隐私机制。

**📊 数据集**

实验数据集为GLUE基准（QNLI、RTE、QQP、MNLI、SST‑2），采用冻结的LLaMA‑7B模型作为backbone。

**📈 对比分析**

与零填充、FedAvg、传统堆叠等基线在rank异质性场景下对比，SDFLoRA在所有任务均提升1–6%准确率；在引入DP时实现更优的utility‑privacy平衡，优于无结构聚合方法。

**⚠️ 局限性**

局限性包括对rank预算、客户端规模和数据非IID程度的敏感性；局部模块不共享可能限制全局模型的知识聚合；对其他适配器族的泛化和动态模块分配机制尚未深入探索。

---

## 206. Performance Analysis of Cell-Free Massive MIMO under Imperfect LoS Phase Tracking

**arXiv ID:** 2601.11179 | [PDF](https://arxiv.org/pdf/2601.11179v1)

**作者:** Noor Ul Ain `[一作]` (Technical University of Berlin), Sławomir Stańczak `[通讯]` (Fraunhofer Heinrich Hertz Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究在细粒度LoS相位跟踪误差存在的情况下，细胞自由大规模MIMO网络的信道估计与波束成形，并给出性能下限与上限。

**💡 创新点**

提出统一的Rician衰落模型，将相位误差建模为均匀分布并引入相位噪声惩罚因子；推导线性MMSE估计器、虚拟上行模型、中心化与分布式MMSE波束成形器，能连续覆盖从完美相位到完全未知相位的两极情况。

**🔧 技术方法**

线性MMSE估计、虚拟上行近似、中心化/分布式MMSE波束成形、使用-忘记(UatF)与乐观极限(OER)速率下限/上限分析、Monte‑Carlo仿真。

**📊 数据集**

无公开数据集，采用仿真：L=100个AP，N=4天线，K=40单天线UE，1000 m×1000 m 区域，5 GHz载波，Rician因子κ∈{1,5,20,100}，相位误差δ∈{0°,15°,30°,45°,90°,180°}。

**📈 对比分析**

通过与两极情况（完全已知相位与完全未知相位）以及不同κ值的CDF/平均SE对比，显示即使相位误差较大（δ≈45°）仍能获得显著性能提升；中心化波束成形对相位误差更稳健，分布式波束成形在高κ或大δ时性能衰减更明显。

**⚠️ 局限性**

局限：相位误差假设为均匀分布且不随时间演化；仅考虑上行链路；未对硬件失真、频率偏移或移动性产生的相位漂移做细致建模；仿真仅覆盖理想化环境，缺乏真实测量验证。

---

## 207. Proving Circuit Functional Equivalence in Zero Knowledge

**arXiv ID:** 2601.11173 | [PDF](https://arxiv.org/pdf/2601.11173v1)

**作者:** Sirui Shen `[一作]` (Centrum Wiskunde en Informatica), Chenglu Jin `[通讯]` (Centrum Wiskunde en Informatica)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了基于零知识证明的组合等价检查框架 ZK‑CEC，能够在不泄露秘密设计结构的前提下，正式验证第三方 IP 的功能正确性。

**💡 创新点**

创新点在于首次将零知识证明与形式化验证结合，提出了一套通用的秘密公式属性检查蓝图，并针对组合等价问题设计了四个子协议，克服了传统 ZKUNSAT 在秘密公式下的可靠性与效率双重瓶颈。

**🔧 技术方法**

核心技术包括 VOLE‑基础零知识证明、基于多项式的 CNF 归约、分布式分辨率证明验证、压缩优化以及 EMP‑toolkit 与 NTL 的实现支撑。

**📊 数据集**

使用了 37 个组合电路数据集，涵盖算术算子、控制逻辑、错误检测模块、加密组件（AES、SM4、Ascon、Present）及投票电路等多种典型实现。

**📈 对比分析**

与原始 ZKUNSAT 对比，实验显示在压缩优化后，总证明时间平均提升约 2.88 倍，且能够在实际时间内完成 AES S‑Box 的验证，证明了方法在中等规模设计上的可行性和高效性。

**⚠️ 局限性**

局限性主要体现在分辨率证明规模急剧增长导致的内存和时间瓶颈，当前框架在处理非常大规模设计时会出现内存溢出或超时。

---

## 208. Noisy Graph Patterns via Ordered Matrices

**arXiv ID:** 2601.11171 | [PDF](https://arxiv.org/pdf/2601.11171v1)

**作者:** Jules Wulms `[一作]` (TU Eindhoven), Bettina Speckmann `[通讯]` (TU Eindhoven)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种利用良序化邻接矩阵与 Moran's I 重新排序来检测和可视化图中的噪声模式（Clique、Biclique、Star），并通过 Ring Motif 简化展示噪声水平。

**💡 创新点**

创新点在于：1）将噪声模式定义为矩阵中满足局部 Moran's I 阈值的矩形子块；2）通过矩阵重新排序把图结构显式化；3）提出 Ring Motif 形状和力导向布局，直接在图的高层结构上编码噪声。

**🔧 技术方法**

技术包括：①使用 Moran's I 作为矩阵重新排序的目标函数（转化为 TSP 问题）；②基于局部重加权 Moran's I 的子矩阵噪声判定；③候选模式枚举与动态规划选择最大权重独立集；④力导向布局四力（旋转、吸引、排斥、引力）生成 Ring Motif。

**📊 数据集**

实验使用三类真实数据集：FLT（脑功能网络，29 节点），ZKC（Zachary Karate Club，34 节点），SCH（学校社交网络，242 节点）。

**📈 对比分析**

与传统无排序的矩阵和现有的图形化简法相比，本文方法在同一噪声阈值下能捕捉到更少噪声、覆盖率更高的模式；计算时间上排序耗时最多但仍在秒级；模式枚举和选择在毫秒级完成，布局在秒级收敛。

**⚠️ 局限性**

局限包括：①排序步骤依赖外部 TSP 求解器，对大规模图可能不可行；②噪声阈值参数 σ、τ 需要人工调优；③对稀疏图的参数设置敏感；④Ring Motif 的布局仍较刚性，链接可视化在高重叠时可能混乱。

---

## 209. How Do Technological Prototypes in the Food Industry Impact People's Perception? Insights from the MUSAE "GROW, COOK, CODE" Final Exhibition

**arXiv ID:** 2601.11169 | [PDF](https://arxiv.org/pdf/2601.11169v1)

**作者:** Francesco Semeraro `[一作]` (University of Manchester), Angelo Cangelosi `[通讯]` (University of Manchester)

**通讯引用:** 10048 | [OpenAlex ID](https://openalex.org/A5091768977)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对MUSAE“GROW, COOK, CODE”展览中11个技术原型的访客调查，评估了这些原型对人们健康、信任、饮食习惯及环境意识的影响。

**💡 创新点**

创新点在于将Design Futures Art-driven (DFA) 方法与艺术家-中小企业合作相结合，开发面向食品行业的技术原型，并首次系统性地通过公开展览与访客交互评估其社会认知效应。

**🔧 技术方法**

使用的方法包括5点李克特量表问卷、开放式主题分析（thematic analysis）以及对不同维度（健康、信任、饮食、环境）的聚合评分。

**📊 数据集**

数据集为24份完整问卷（10男14女，平均年龄34岁）和21份开放式回答，收集自2025年6月在塞尔维亚贝尔格莱德科学宫举办的展览。

**📈 对比分析**

对比方法为对各维度问卷题目进行平均分计算，结果显示所有维度中位数均为正值，且健康维度影响最大；主题分析显示77%评论为正面。

**⚠️ 局限性**

局限性包括样本量有限、受访者多为城市访客、评价仅为短期感知，未考察原型在实际生活中的长期效果与可持续性。

---

## 210. Scalable Music Cover Retrieval Using Lyrics-Aligned Audio Embeddings

**arXiv ID:** 2601.11262 | [PDF](https://arxiv.org/pdf/2601.11262v1)

**作者:** Joanne Affolter `[一作]` (Deezer Research), Frédéric Kaplan `[通讯]` (EPFL)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LIVI方法，用歌词信息对音乐封面检索进行投影学习，兼顾准确性和效率。

**💡 创新点**

创新点在于用已训练好的歌词嵌入空间作为监督，只训练音频编码器即可对齐音频与歌词特征，省去转录步骤。

**🔧 技术方法**

技术包括Whisper语音识别的编码器、句子级多语言文本嵌入（gte‑multilingual‑base）、注意力池化和投影网络，结合点对点与几何保持损失进行训练。

**📊 数据集**

使用Discogs‑VI、Covers80、SHS100k等公开和内部数据集进行训练与评估，并对比官方版本检索基线。

**📈 对比分析**

与字面转录、原始Whisper、ByteCover2等基线对比，LIVI在Covers80、SHS100k、Discogs‑VI上均达到或超过HR@1、MAP@10，且在Discogs‑VI上显著优于音频基线。

**⚠️ 局限性**

局限在于只能处理含足量人声的曲目，完全依赖歌词；未针对任务微调文本嵌入；预训练的声学检测为专有，影响完整复现。

---

## 211. Latent Dynamics Graph Convolutional Networks for model order reduction of parameterized time-dependent PDEs

**arXiv ID:** 2601.11259 | [PDF](https://arxiv.org/pdf/2601.11259v1)

**作者:** Lorenzo Tomada `[一作]` (SISSA), Gianluigi Rozza `[通讯]` (SISSA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种全新的无编码器的潜在动力学图卷积网络（LD-GCN）用于参数化时变偏微分方程的模型顺序化（ROM）；

**💡 创新点**

创新点在于将潜在动力学与图卷积解码器相结合，既保留了时序因果性又充分利用几何先验，并通过无编码器架构实现参数无关的低维表征与零射击预测；

**🔧 技术方法**

使用的技术包括图神经网络（GCN）、潜在动力学网络（node）、显式欧拉时间积分、损失函数的误差与正则化项以及对数据的归一化；

**📊 数据集**

实验数据集基于高精度有限元（FEniCS、RBniCS）生成的四个数值基准：带物理/几何参数的扩散-对流方程、叶片驱动腔流、纳维-斯托克斯腔流以及Coandă效应的分岔问题；

**📈 对比分析**

与传统GCA和LDnet进行对比，LD-GCN在相同参数化与时空范围下误差平均降低约一阶（最高相对误差从≈10⁻¹降至≈5×10⁻³），且参数量约为对方的50%，在时间与参数外推以及零射击预测方面表现优异；

**⚠️ 局限性**

局限性包括：网络参数随网格节点数线性增长，难以扩展到极大网格；需要固定网格连通性与节点数；目前不支持不同初始条件或外力场的变形；未来工作需引入多精度/多分辨率、运算误差估计与更一般的时间可变信号处理。

---

## 212. Knowledge is Not Enough: Injecting RL Skills for Continual Adaptation

**arXiv ID:** 2601.11258 | [PDF](https://arxiv.org/pdf/2601.11258v1)

**作者:** Pingzhi Tang `[一作]` (Peking University), Muhan Zhang `[通讯]` (Peking University)

**通讯引用:** 4714 | [OpenAlex ID](https://openalex.org/A5071515223)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种名为PaST的框架，用来在大型语言模型（LLM）中通过从源领域抽取的技能向量（Skill Vector）实现知识更新后逻辑推理能力的迁移，从而克服知识截止（knowledge cutoff）问题。

**💡 创新点**

创新点在于发现并利用SFT（监督微调）与RL（强化学习）产生的参数更新几乎正交的性质，提出通过对源领域SFT与RL模型参数差值提取领域无关的技能向量，并在目标领域轻量SFT后线性注入该向量，以无需在目标领域执行昂贵的RL即可提升推理与执行能力。

**🔧 技术方法**

核心技术包括参数向量算术（Task Vector）提取、正交性分析、线性技能注入、迭代技能细化、以及基于RL的技能学习（如GRPO、PPO）。

**📊 数据集**

使用的数据集涵盖：SQuAD（短段落知识检索）、LooGLE（长上下文QA）、ToolBench（工具使用和执行）。

**📈 对比分析**

与现有知识更新方法（如SEAL、SFT+synthetic、SFT+GPT-4.1等）以及不同技能注入策略（顺序微调、预注入、后置组合）进行对比；在SQuAD上PaST提升至56.9%准确率，比SEAL高+9.9点；在LooGLE上实现8.0点绝对精度提升；在ToolBench跨域工具使用中平均提高10.3%成功率，显著优于SFT基线。

**⚠️ 局限性**

限制包括：实验仅覆盖QA和工具使用两类任务，缺乏更广泛的源-目标转移场景；技能向量系数λ固定为1，可能不适用于所有情境；在不同模型规模或架构下正交性是否保持仍待验证。

---

## 213. On Known APNs

**arXiv ID:** 2601.11247 | [PDF](https://arxiv.org/pdf/2601.11247v1)

**作者:** Valérie Gillot ad Philippe Langevin `[一作]` `[通讯]`, Valérie Gillot ad Philippe Langevin

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过构造新的不变性、可扩展性判据以及回溯算法，对 6 位映射进行分类，系统地验证了已知的 14 个 CCZ‑类的完整性，并在此基础上探索了两层谱级的函数。

**💡 创新点**

创新点在于：① 提出基于第四阶谱时刻的全新不变量；② 设计了判定 (m,m‑k) 向量函数是否可扩展的可扩展性判据；③ 结合回溯搜索与不变量过滤，实现了对 6‑bit APN 函数空间的全量枚举。

**🔧 技术方法**

主要技术手段包括：布尔函数的 Walsh 变换、谱时刻分析、Lift by Restriction 不变量、EA/CCZ 等价判定、以及基于位操作的并行回溯搜索。

**📊 数据集**

使用的实验数据集为全部已知的 6‑bit 映射（共 2⁶⁰ 6‑bit 函数），其中包含 716 个 EA‑类、14 个 CCZ‑类、534 个三次函数等。

**📈 对比分析**

比较方法：通过计算不变量（如 (2,0)、(p,q) 等）在各类中的取值分布，并与已知分类结果对照；性能表现为：对 6‑bit 函数的全量不变量计算仅需 18 秒，回溯扩展搜索在多核环境下完成 1,362,046 个子空间的扩展，产生 37 个新的分量类。

**⚠️ 局限性**

局限性：① 对两层谱级（α,β）函数的全搜索仍非常耗时，未能完全枚举；② 仅在 6‑bit 维度验证，尚未推广至更高维；③ 现有算法对计算量敏感，若扩展到更大空间需进一步优化并行策略。

---

## 214. LLM-Assisted Pseudo-Relevance Feedback

**arXiv ID:** 2601.11238 | [PDF](https://arxiv.org/pdf/2601.11238v1)

**作者:** David Otero `[一作]` (Universidade da Coruña), Javier Parapar `[通讯]` (Universidade da Coruña)

**通讯引用:** 1833 | [OpenAlex ID](https://openalex.org/A5046723532)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在RM3前加入LLM判别过滤伪相关文档，提升检索效果

**💡 创新点**

创新点是利用LLM的判别能力来去噪PRF集合，避免生成式模型的幻觉

**🔧 技术方法**

采用RM3、MonoT5或Llama 3.1‑8B‑Instruct进行二值相关判断，并在此基础上计算RM3

**📊 数据集**

在AP88‑89、ROBUST04、MSMARCO（DL‑19/DL‑20）和WT10G等标准IR集合上评测

**📈 对比分析**

与QLD、RM3、MonoT5 rerank、MonoT5F、MonoT5F+RM3 w/prob、LLMF+RM3、LLMF+RM3 w/prob和oracle比较，显著提升AP@1000和NDCG@100

**⚠️ 局限性**

主要局限在LLM判别质量、计算开销、对叙述信息依赖以及oracle距离仍显著

---

## 215. Bio-inspired fine-tuning for selective transfer learning in image classification

**arXiv ID:** 2601.11235 | [PDF](https://arxiv.org/pdf/2601.11235v1)

**作者:** Ana Davila `[一作]` (Nagoya University), Yasuhisa Hasegawa `[通讯]` (Nagoya University)

**通讯引用:** 6401 | [OpenAlex ID](https://openalex.org/A5067104092)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于进化算法的自适应微调方法BioTune，用以自动决定冻结哪些层及其学习率，从而提升迁移学习效果。

**💡 创新点**

创新点在于将层冻结与学习率调整联合建模为优化问题，并通过改进的遗传算法结合种群动量实现高效搜索。

**🔧 技术方法**

采用进化算法（GA+PSO动量）、梯度下降、数据分层评估等技术来寻找最佳微调配置。

**📊 数据集**

在九个图像分类数据集（MNIST、SVHN、CIFAR‑10、Flowers‑102、FGVC‑Aircraft、DTD、ISIC2020等）以及四种CNN架构上进行实验。

**📈 对比分析**

与全微调、线性探测、正则化、逐层解冻、AutoRGN、LoRA等方法对比，BioTune在大多数任务上实现了0.3–9.7%的准确率提升，参数使用量更低。

**⚠️ 局限性**

局限包括仅在CNN上验证，未覆盖ViT等新型架构，且进化搜索仍需额外计算，未来需探索更高效的搜索策略和跨任务适用性。

---

## 216. Adaptive Monitoring of Stochastic Fire Front Processes via Information-seeking Predictive Control

**arXiv ID:** 2601.11231 | [PDF](https://arxiv.org/pdf/2601.11231v1)

**作者:** Savvas Papaioannou `[一作]` (KIOS Research and Innovation Centre of Excellence), Marios M. Polycarpou `[通讯]` (KIOS Research and Innovation Centre of Excellence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一套基于移动无人机的火线自适应监测框架，集成了感知、递归贝叶斯估计与预测控制，目标是最小化在有限感知范围内对火前沿状态不确定性的累积误差。

**💡 创新点**

创新点主要包括：① 将感知、估计与控制统一归入随机最优控制（SOC）框架；② 针对随机非线性椭圆型火线扩散模型提出递归贝叶斯估计（支持多目标Poisson测量）；③ 将非线性SOC问题转换为有限时限马尔可夫决策过程，并采用基于下置信界（LCB）的自适应搜索实现非贪婪信息寻求策略；④ 通过风险加权方差度量（RWD）评估火前沿的不确定性，实现风险感知规划。

**🔧 技术方法**

技术与方法包括：粒子滤波（SIR）实现递归贝叶斯估计；下置信界（LCB）和UCB1框架的自适应策略搜索；有限时限MDP建模；风险加权分散（RWD）度量；离散控制动作集与代理动力学；对火前沿的椭圆增长模型与环境随机性（风向、风速、燃料扩散率）的概率建模；Poisson点过程测量建模。

**📊 数据集**

实验使用仿真数据：3km×3km 方形场景被离散成10×10网格，20个火前沿顶点、随机风向/风速/燃料扩散率分布以及风险图；在此仿真环境下进行50次Monte Carlo试验，评估不同规划时限和基线方法的性能。

**📈 对比分析**

与三种基线（无限感知范围静止、有限感知范围静止、随机控制）以及不同规划时限（T=1、3、5）进行对比，性能指标为25步时长内每个顶点的RMSE。结果显示：非贪婪策略（T=3、5）显著降低估计误差，特别是在高风险区域；相较于贪婪策略（T=1）和随机/静止策略，整体误差下降约30–50%（数值取决于试验），表明信息寻求与非贪婪规划有效提升监测精度。

**⚠️ 局限性**

局限性包括：① 代理动力学假设为确定性且动作离散化，真实无人机需考虑动力学约束与连续动作；② 计算复杂度高，尤其是粒子滤波与多策略回放在大时限下资源占用大；③ 依赖已知的风险图与环境参数，实际部署需在线获取或预估；④ 仅在仿真中验证，缺乏真实火灾现场数据与鲁棒性评估。

---

## 217. ATATA: One Algorithm to Align Them All

**arXiv ID:** 2601.11194 | [PDF](https://arxiv.org/pdf/2601.11194v1)

**作者:** Boyi Pang `[一作]` (Harbin Institute of Technology), Evgeny Burnaev `[通讯]` (Applied AI Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于Rectified Flow的联合推理算法，可在同一潜在空间内同步生成结构对齐的图像、视频和3D模型；

**💡 创新点**

核心创新是对潜在空间中线段进行速度引导的联合传输，并通过anchor‑velocity平滑正则化保证过渡平滑，仅改动推理循环而不需要额外训练；

**🔧 技术方法**

使用Rectified Flow模型（FLUX、Trellis、WAN等）及其速度场，结合线段采样、线性回归、anchor‑velocity校正与平滑调度；

**📊 数据集**

采用A3D短提示数据集、相似物体对样例、各种视频场景集合，并借助Grounded SAM、Depth Anything、CLIP、GPTEval等评测工具进行实验；

**📈 对比分析**

与编辑式方法（RF‑Inversion、VACE、Lucy‑edit、MVEdit）和联合生成方法（A3D、MatchDiffusion）对比，使用DIFT、Depth MAE、CLIP、DINO、VLM、GPTEval等指标，结果显示在图像、视频、3D上均与SOTA持平或超越；视频指标最高，3D速度提升百倍，整体质量接近SOTA；

**⚠️ 局限性**

对纹理与几何细节的把握仍不及基于SDS的A3D，且依赖结构化潜在表示，极端多样性或大规模场景下对齐效果仍需进一步提升。

---

## 218. DOREMI: Optimizing Long Tail Predictions in Document-Level Relation Extraction

**arXiv ID:** 2601.11190 | [PDF](https://arxiv.org/pdf/2601.11190v1)

**作者:** Laura Menotti `[一作]` (University of Padova), Gianmaria Silvello `[通讯]` (University of Padova)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DOREMI，一种通过迭代主动学习和模型间不一致度选择来提升文档级关系抽取中长尾关系性能的框架；

**💡 创新点**

创新点在于将模型间不一致度作为主动学习采样准则，最小化人工标注量而显著提升稀有关系的精确度和泛化能力；

**🔧 技术方法**

采用多模型集成、概率不一致度计算、阈值筛选、以及 BERT/Transformer 等现有 DocRE 模型的微调；

**📊 数据集**

使用 DocRED 与其改进版 Re-DocRED 作为数据集，分别在两者的稀有关系上进行实验；

**📈 对比分析**

与当前最优标签去噪方法 UGDRE 以及 ATLOP、DREEAM 等基线对比，结果显示 DOREMI 在整体与长尾 F1 分别提升约 +0.7% 与 +5%，在极端长尾上精确率提升高达 +83%；

**⚠️ 局限性**

局限性包括：需人工标注虽极少但仍不可完全自动化；对超参数（阈值、采样量）敏感，调优成本高；且对极少样本关系的召回提升有限。

---

## 219. From Knots to Knobs: Towards Steerable Collaborative Filtering Using Sparse Autoencoders

**arXiv ID:** 2601.11182 | [PDF](https://arxiv.org/pdf/2601.11182v1)

**作者:** Martin Spišák `[一作]` (Recombee), Rodrigo Alves `[通讯]` (Faculty of Information Technology Czech Technical University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在协同过滤推荐中插入稀疏自编码器（SAE），将密集用户嵌入转换为稀疏可解释的“控制钮”，实现推荐的可调节性。

**💡 创新点**

创新点在于首次将SAE嵌入CFAE，并通过自监督学习得到单义词概念对应的神经元，实现可解释性与可调节性的统一。

**🔧 技术方法**

使用的技术包括协同过滤自编码器（ELSA、MultVAE）、稀疏自编码器（Basic SAE、TopK SAE）以及基于标签的概念-神经元映射。

**📊 数据集**

使用的数据集为MovieLens 25M和Million Song Dataset（MSD），采用二进制隐式反馈。

**📈 对比分析**

与原始CFAE比较，TopK SAE在保持Recall@20和nDCG@20近乎不下降的前提下，实现了大约90%+的性能保持，且稀疏程度可调；实验表明可通过激活神经元显著偏移推荐主题。

**⚠️ 局限性**

局限性包括对仅采用二进制反馈、有限标签信息的依赖；仅使用了简单的SAE变体，未探索更深层次或辅助损失；并非所有CFAE架构均适用于稀疏重建，尤其是基于变分的模型表现不佳。

---

## 220. Effects of Introducing Synaptic Scaling on Spiking Neural Network Learning

**arXiv ID:** 2601.11261 | [PDF](https://arxiv.org/pdf/2601.11261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 221. The Growing Gains and Pains of Iterative Web Corpora Crawling: Insights from South Slavic CLASSLA-web 2.0 Corpora

**arXiv ID:** 2601.11170 | [PDF](https://arxiv.org/pdf/2601.11170v1)

**作者:** Taja Kuzman Pungeršek `[一作]`, Nikola Ljubešić `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建并发布了CLASSLA‑web 2.0，涵盖七种南斯拉夫语言的17亿词、3800万文本，并对每条文本进行体裁、主题与词法标注。

**💡 创新点**

创新点包括：两年后再次爬取顶级域得到更多新内容；引入新闻主题标签；提出基于URL重叠的加权线性模型快速估算文本重叠；以及对高频域进行人工筛查，显著降低自动生成文本。

**🔧 技术方法**

技术手段包括MaCoCu爬虫、trigram + CLD2 + Naive Bayes HBS语言识别、jusText与onion去除冗余、X‑GENRE与新闻主题分类器、CLASSLA‑Stanza词法标注，以及MinHash近似重复检测和线性回归估算重叠。

**📊 数据集**

使用的数据集为CLASSLA‑web 2.0语料（七种语言），与先前的CLASSLA‑web 1.0版本进行对比；同时参考了cc100、mC4、OSCAR、HPLT、FineWeb2等公开多语言语料。

**📈 对比分析**

对比方法通过MinHash检测近似重复、计算URL与文本重叠并用加权线性回归预测重叠率，相关系数0.908；结果显示2.0版本比1.0增加约57%词量、46%文本量，且约80%内容为新文本，表明网页内容快速更新。

**⚠️ 局限性**

局限性包括：自动生成与低质量文本比例上升，需人工核查高频域；内容重叠测量仍依赖耗时的近似重复检测；HBS互通语言的识别准确率仍有提升空间。

---

## 222. Metabolomic Biomarker Discovery for ADHD Diagnosis Using Interpretable Machine Learning

**arXiv ID:** 2601.11283 | [PDF](https://arxiv.org/pdf/2601.11283v1)

**作者:** Nabil Belacel `[一作]` (Digital Technology Research Center), Mohamed Rachid Boulassel `[通讯]` (Sultan Qaboos University)

**通讯引用:** 4956 | [OpenAlex ID](https://openalex.org/A5022122700)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过结合尿液代谢组学与可解释机器学习，构建并验证了基于间隔学习的Closest Resemblance（CR）分类器，用于精准诊断儿童ADHD。

**💡 创新点**

创新点在于采用区间特征学习与嵌入式贪婪特征选择，既提升了分类准确率又保证了模型的透明度与低计算成本。

**🔧 技术方法**

使用了CR分类器（PROMETHEE相似度、区间学习）、贪婪嵌入式特征选择、留一交叉验证，并与随机森林、k近邻等传统模型进行对比。

**📊 数据集**

利用公开儿童ADHD尿液代谢组数据（98例，60个代谢物），包括52例ADHD患者与46例健康对照。

**📈 对比分析**

在LOOCV下，CR在60特征时AUC 0.96、准确率 95.9%，使用14特征后AUC 0.978、准确率 97.9%，训练时间 <0.01s，显著优于RF（AUC 0.91）和kNN等。

**⚠️ 局限性**

局限在于样本量有限、仅单一队列、缺乏外部验证，需进一步检验跨人群泛化和长期稳定性。

---

## 223. From SERPs to Sound: How Search Engine Result Pages and AI-generated Podcasts Interact to Influence User Attitudes on Controversial Topics

**arXiv ID:** 2601.11282 | [PDF](https://arxiv.org/pdf/2601.11282v1)

**作者:** Junjie Wang `[一作]` (Delft University of Technology), Ujwal Gadiraju `[通讯]` (Delft University of Technology)

**通讯引用:** 3669 | [OpenAlex ID](https://openalex.org/A5038081564)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对 483 名受试者的对照实验，考察了搜索引擎结果页（SERPs）与 AI 生成的播客在信息序列（SERP‑first vs Podcast‑first）、观点偏向（支持、中立、反对）以及议题争议度（中度 vs 高度）下，对用户态度变化的影响。

**💡 创新点**

创新点在于首次系统探讨多模态信息（视觉 SERPs 与听觉播客）在不同顺序下的交互效应，以及观点偏向与争议度如何调节这一效应，为多模态搜索系统的负责任设计提供经验依据。

**🔧 技术方法**

研究方法包括 2×3×2 的实验设计、基于 NotebookLM 生成观点偏向的播客和 SERP 概览、使用差分-差分混合模型（piecewise DiD）分析态度随时间的变化，并通过自评量表评估 AI 文识、智识谦逊、认知需求和用户投入等个体差异。

**📊 数据集**

数据集来源于公开的观点标注 SERP 结果（ProCon+NotebookLM）以及利用 NotebookLM 生成的音频内容，涵盖六个中度与高度争议的议题（如手机辐射、社交网络、肥胖、移民、枪支控制、堕胎）。

**📈 对比分析**

比较结果显示，Podcast‑first 顺序相较于 SERP‑first 能略微提升态度变化（β≈-0.38，p=0.034，Cohen's d≈-0.10），且在观点偏向与争议度交互中表现出统计显著但效应量微小的差异，表明多模态序列确实产生影响，但整体效应较小。

**⚠️ 局限性**

局限性包括：效应量普遍偏小；仅覆盖六个议题，缺乏更广泛的主题覆盖；依赖自我报告的态度与投入量表，可能存在社会期望偏差；实验样本为美国 18–50 岁的 Crowdsourcing 受试者，外推性受限；以及仅考虑两种媒介，未覆盖更丰富的多模态组合。

---

## 224. Reasoning in Trees: Improving Retrieval-Augmented Generation for Multi-Hop Question Answering

**arXiv ID:** 2601.11255 | [PDF](https://arxiv.org/pdf/2601.11255v1)

**作者:** Yuling Shi `[一作]` (Shanghai Jiao Tong University), Xiaodong Gu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2317 | [OpenAlex ID](https://openalex.org/A5033286111)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于推理树的检索增强生成框架RT‑RAG，显式将多跳问答拆解为树形结构并自底向上检索合成答案。

**💡 创新点**

通过结构化实体分析、共识驱动的树选取、动态叶节点判定以及拒绝采样等机制，构建可解释且稳健的层次推理路径，显著抑制误差传播和幻觉。

**🔧 技术方法**

采用LLM进行问题拆解、树生成与检索，结合检索增强生成（RAG）、查询重写、投票式答案筛选与共识树选择等技术实现。

**📊 数据集**

在MuSiQue、2WikiMQA和HotpotQA三大多跳问答基准上进行实验。

**📈 对比分析**

与ChainRAG、LongRAG、HippoRAG、Self‑Ask等SOTA方法对比，平均提升约7% F1和6% EM，在所有基准上均取得最优成绩。

**⚠️ 局限性**

对树深度和拆解质量仍有依赖，过深或错误的树会导致计算开销增大；在极度稀疏或非结构化知识场景下，树生成与检索仍面临挑战；此外需要额外的候选树生成与共识计算，增加实现复杂度。

---

## 225. FTDMamba: Frequency-Assisted Temporal Dilation Mamba for Unmanned Aerial Vehicle Video Anomaly Detection

**arXiv ID:** 2601.11254 | [PDF](https://arxiv.org/pdf/2601.11254v1)

**作者:** Cheng-Zhuang Liu `[一作]` (Anhui University), Bin Luo `[通讯]` (Anhui University)

**通讯引用:** 10833 | [OpenAlex ID](https://openalex.org/A5100372676)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于频率辅助时间扩张Mamba的无人机视频异常检测框架FTDMamba。

**💡 创新点**

创新点在于引入频率解耦时空相关模块分离多源运动，并结合时间膨胀Mamba实现多尺度时间与空间建模。

**🔧 技术方法**

使用的技术包括一维/二维FFT频率分解、Wiener–Khinchin自相关、Mamba结构的状态空间模型以及多尺度时间膨胀扫描。

**📊 数据集**

实验使用了现有的Drone‑Anomaly、UIT‑ADrone数据集以及新构建的多运动无人机异常检测数据集MUVAD。

**📈 对比分析**

与多种基准方法对比，FTDMamba在Micro‑AUC、Macro‑AUC和EER指标上均达成SOTA表现，尤其在动态背景场景中显著优于先前方法。

**⚠️ 局限性**

主要局限包括对极端噪声和帧缺失的鲁棒性仍有限，且模型深度与实时性能之间存在折衷。

---

## 226. Democratizing planetary-scale analysis: An ultra-lightweight Earth embedding database for accurate and flexible global land monitoring

**arXiv ID:** 2601.11183 | [PDF](https://arxiv.org/pdf/2601.11183v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 227. T$^\star$: Progressive Block Scaling for MDM Through Trajectory Aware RL

**arXiv ID:** 2601.11214 | [PDF](https://arxiv.org/pdf/2601.11214v1)

**作者:** Hanchen Xia `[一作]` (Shanghai Academy of AI for Science), Siyu Zhu `[通讯]` (Fudan University)

**通讯引用:** 2833 | [OpenAlex ID](https://openalex.org/A5013549550)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于TraceRL的逐步块大小扩展训练课程，用于掩码扩散语言模型的并行推理。

**💡 创新点**

创新点在于通过轨迹感知强化学习实现块大小的渐进扩张，并利用块边界移位缓解边界效应，从而在保持推理精度的同时提升并行度。

**🔧 技术方法**

技术包括TraceRL（PPO式轨迹感知强化学习）、掩码扩散语言模型（MDM）以及块式推理策略。

**📊 数据集**

数据集使用了8K高质量数学题（Openr1math）以及MATH500、GSM8K、AIME24等评测数据。

**📈 对比分析**

与直接在大块上训练的TraceRL基线相比，本文方法在MATH500、GSM8K、AIME24上保持甚至提升准确率，稳定性更好，尤其在块大小从4→8→16时不出现性能崩溃。

**⚠️ 局限性**

限制在于对更大块（B≥64）未进行扩展，残留的性能下降未完全消除，并且对初始高质量“cold-start”阶段仍有需求。

---

## 228. VidLeaks: Membership Inference Attacks Against Text-to-Video Models

**arXiv ID:** 2601.11210 | [PDF](https://arxiv.org/pdf/2601.11210v1)

**作者:** Li Wang `[一作]` (Shandong University), Shanqing Guo `[通讯]` (Shandong University)

**通讯引用:** 1349 | [OpenAlex ID](https://openalex.org/A5084460856)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了文本到视频模型的成员推断攻击，并提出稀疏‑时序攻击框架

**💡 创新点**

创新点在于设计了针对视频稀疏记忆与时序动态的两种攻击信号：Sparse Reconstruction Fidelity（SRF）与 Temporal Generative Stability（TGS），并在三种黑盒威胁模型下实现无监督到监督的完整攻击链

**🔧 技术方法**

技术手段包括关键帧提取、Top‑K CLIP 相似度匹配、TGS 通过多次生成的方差计算、统计异常检测与线性融合、无监督评分以及多查询策略

**📊 数据集**

使用公开可得的 T2V 模型训练集（AnimateDiff、Mira、InstructVideo 所用的 WebVid‑10M、MiraData）作为成员样本，Panda‑70M 作为非成员样本；同时使用 Gemini Pro、Doubao 等公共视频字幕模型作为代理文本生成工具

**📈 对比分析**

与传统帧级 CLIP 相似度、视频级 VideoCLIP 相似度等基线相比，SRF+TGS 在三种威胁模型下均显著提升 AUC（Query‑only 最高可达 82.92% / 97.01%），TPR@1%FPR 也大幅提高，证明攻击效果显著优于现有方法

**⚠️ 局限性**

局限性包括仅评估三种模型；未覆盖更大规模或最新的 T2V 系统；对代理字幕质量的依赖未作深入探讨；防御实验仅采用轻量化参数扰动，缺乏全面的 DP 或数据过滤等更稳健的安全策略

---

## 229. LoRA as Oracle

**arXiv ID:** 2601.11207 | [PDF](https://arxiv.org/pdf/2601.11207v1)

**作者:** Marco Arazzi `[一作]` (University of Pavia), Antonino Nocera `[通讯]` (University of Pavia)

**通讯引用:** 1203 | [OpenAlex ID](https://openalex.org/A5085793134)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于LoRA的后训练模型审计框架LoRAcle，利用低秩适配器对模型进行轻量化微调，进而实现后门检测与成员推断。

**💡 创新点**

创新点在于将LoRA视为“oracle”，通过分析其更新的几何特征（能量、对齐度、轨迹混沌）在无需原始训练数据或完整模型重训练的前提下，区分已训练样本与未知样本，并检测隐藏的后门。

**🔧 技术方法**

核心技术包括LoRA参数高效微调、低秩更新统计、能量与混沌指标计算、软聚类专家模型融合、代理样本生成（针对后门）、以及排名/能量基统计决策。

**📊 数据集**

实验使用MNIST、CIFAR‑10、CIFAR‑100、GTSRB四个视觉数据集，模型分别为ResNet18、VGG19、DenseNet和Vision Transformer。

**📈 对比分析**

与现有后门与成员推断方法对比，LoRAcle在四个数据集和四种模型上实现了接近或超过90%的会员推断准确率，以及在多种后门攻击（BadNets、Blended、WaNet）下Top‑3检测准确率超过90%，且显著降低了GPU功耗和显存占用。

**⚠️ 局限性**

局限性包括对Vision Transformer的召回率偏低、对低毒化率或隐蔽后门的检测灵敏度不足、代理样本生成方式较为粗糙，以及实验仅在小规模数据集和固定微调设置下验证，尚未验证在大规模预训练语料上的可扩展性。

---

## 230. X-raying the arXiv: A Large-Scale Analysis of arXiv Submissions' Source Files

**arXiv ID:** 2601.11385 | [PDF](https://arxiv.org/pdf/2601.11385v1)

**作者:** Giovanni Apruzzese `[一作]` (University of Liechtenstein), Aurore Fass `[通讯]` (Inria Centre at Université Côte d’Azur)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2015-2025年约60万份arXiv提交的源文件进行纵向分析，量化并识别其中不必要的冗余数据以及可能泄露的敏感信息；

**💡 创新点**

首次系统性评估arXiv源文件的冗余比例与信息泄露风险，并提出可扩展的自动化检测工具与改进建议；

**🔧 技术方法**

使用文件解析、内容去重、文本分类、统计分析等技术，对源文件进行结构化提取与量化评估；

**📊 数据集**

研究基于约60万份arXiv源文件的公开数据集，覆盖2015-2025年所有提交；

**📈 对比分析**

通过与PDF生成所需文件量对比，展示平均27%源文件为冗余，累计超580GB；并与行业通用的文件压缩与清理方法做对比，证明现行做法效果不足；

**⚠️ 局限性**

局限性在于仅关注文件内容与大小，对作者上传行为、平台政策与实际使用场景影响未做深入探讨，所提工具尚未在arXiv正式部署。

---

## 231. Offline Reinforcement-Learning-Based Power Control for Application-Agnostic Energy Efficiency

**arXiv ID:** 2601.11352 | [PDF](https://arxiv.org/pdf/2601.11352v1)

**作者:** Akhilesh Raj `[一作]` (Vanderbilt University), Solomon Bekele Abera `[通讯]` (Argonne National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并评估了一种基于离线强化学习的CPU功率控制器，实现了对任意应用和任意硬件平台的能效优化。

**💡 创新点**

创新点包括：①使用离线RL（CQL）训练模型，避免在线训练的成本与风险；②将应用进度心跳与硬件性能计数器作为状态，构建与应用无关的进度指标；③在未知应用上实现自适应调节，提升泛化能力。

**🔧 技术方法**

技术手段：离线强化学习（CQL）+Q网络；RAPL功率限制与GEOPM接口；PAPI硬件计数器采集；自定义心跳进度测量；基准集评测。

**📊 数据集**

数据集：约1291条（state, action, reward, next state）样本，来源于STREAM和NPB基准在随机功率限制下采样，包含进度、功率、IPC、STL、CMR等五维状态。

**📈 对比分析**

与基线（无功率限制、固定功率、全局/应用特定PI控制器、DVFS RL、ondemand）比较；结果显示平均节能约20%，性能下降约7–8%，ED²P显著降低，整体优于大多数基线。

**⚠️ 局限性**

局限性：仅在CPU多核节点、可测心跳的迭代式应用上表现良好；对非迭代或小规模应用（如NPB‑CG、NPB‑BT）控制不够稳定；需要离线收集数据，无法动态适应实时变化；目前未扩展到GPU或大规模集群。

---

## 232. Enhancing Vision Language Models with Logic Reasoning for Situational Awareness

**arXiv ID:** 2601.11322 | [PDF](https://arxiv.org/pdf/2601.11322v1)

**作者:** Pavana Pradeep `[一作]`, Suya Yu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种利用传统计算机视觉（TCV）与显式逻辑推理相结合的视听语言模型（VLM）细化调优与可解释推理框架，用于交通、武术及人类动作监测的情境感知；

**💡 创新点**

创新点在于：①通过TCV提取的细粒度代理活动与VLM主任务活动之间的逻辑一致性约束，形成智能（定向）调优选择；②在推理时实时执行一致性检查，为VLM输出提供可靠性判断与解释；③实现与多种VLM架构（MiniGPT‑4、Video‑LLaMA、Video‑Mamba等）的兼容。

**🔧 技术方法**

技术包括：多模态VLM（如MiniGPT‑4、MiniGPT‑4‑Video、X‑CLIP、Video‑LLaMA、Video‑Mamba、Video‑MAE）、YOLO对象检测与跟踪、基于SMT的逻辑推理（Z3）、自适应批次调优与一致性驱动的数据选择。

**📊 数据集**

使用三类公开数据集：TU_DAT（交通事故视频）、Taekwondo（跆拳道动作视频）和Kinetics‑100（100类人类动作），分别包含数百至数千段视频。

**📈 对比分析**

与无定向随机调优或基于准确率的调优对比，定向调优在三大数据集上均显著提升准确率（最高约+10%）并且一致性改进因子（CIF）提升10%~20%；调优时间与无定向相近，推理时解释成本仅略高，平均≈2s。

**⚠️ 局限性**

局限性包括：需额外运行辅助VLM，导致推理资源消耗增加；代理活动与逻辑约束需人工或LLM手工构造，复杂场景下需额外训练；对TCV错误敏感，若TCV误检会导致一致性错误。

---

## 233. FORESTLLM: Large Language Models Make Random Forest Great on Few-shot Tabular Learning

**arXiv ID:** 2601.11311 | [PDF](https://arxiv.org/pdf/2601.11311v1)

**作者:** Zhihan Yang `[一作]` (National University of Singapore), Chenyu You `[通讯]` (Stony Brook University)

**通讯引用:** 4171 | [OpenAlex ID](https://openalex.org/A5076320750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在少样本表格学习中，提出一种利用大型语言模型（LLM）仅在训练阶段进行结构设计的决策森林框架，构建语义驱动的树结构并在叶子节点采用一次性推理生成稳定预测。

**💡 创新点**

创新点包括：① 将LLM作为离线模型设计师，仅用于指导树分裂和叶子标签合成；② 引入语义分裂准则，利用标注与未标注数据的语义一致性来选择分裂；③ 采用一次性上下文推理实现叶子预测，消除传统树模型对稀疏样本的依赖；④ 通过无须LLM推理的轻量化森林实现高效、可解释的推断。

**🔧 技术方法**

核心技术：大规模预训练语言模型（GPT‑4o）、自然语言提示（prompt）与上下文学习、语义推理、半监督树分裂、随机森林集成、叶子级别的离线推理与统计平均。

**📊 数据集**

实验使用了 30+ 公开表格数据集，包括 Adult、Bank、Blood、Credit‑g、Cultivars、NHANES、Car、CDC Diabetes、Heart、Communities、Myocardial、Breast‑W、Gallstone、Infrared_Thermography_Temperature 等，覆盖二分类、多分类与回归任务。

**📈 对比分析**

与传统树模型（CART、RF、XGBoost）、深度模型（MLP、ElasticNet、LogReg）以及近期 LLM‑驱动方法（LIFT、TabLLM、TP‑BERTa、FeatLLM、TabPFN 等）对比，在 4、8、16 shot 的少样本设置下，平均 AUC 与 NRMSE 均达到或超过现有最佳方法，表现出显著的优势。

**⚠️ 局限性**

局限性：① 依赖昂贵的 LLM 训练时推理，无法在资源受限环境下直接部署；② 对数值推理仍不如传统回归模型，需进一步改进；③ 需要足够未标注数据以构建语义特征，数据稀缺时优势受限；④ 由于 LLM 仅在训练阶段使用，模型的可解释性虽好，但无法利用 LLM 的实时自适应能力。

---

## 234. One LLM to Train Them All: Multi-Task Learning Framework for Fact-Checking

**arXiv ID:** 2601.11293 | [PDF](https://arxiv.org/pdf/2601.11293v1)

**作者:** Malin Astrid Larsson `[一作]` (Factiverse), Vinay Setty `[通讯]` (Factiverse)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一个统一的多任务学习框架，利用开源LLM（如Qwen3-4B）通过QLoRA自适应层一次性完成命题检测、证据重排序与立场判定三大核心任务。

**💡 创新点**

创新点包括①首次在解码器LLM上联合微调三项任务；②系统比较了CLS、CLM与Instruction‑Tuning三种任务头，提出最优配置；③通过实验揭示损失权重、任务顺序和模型规模对性能的影响，并给出可操作的实践指南。

**🔧 技术方法**

技术手段包括参数高效微调（QLoRA）、分类/因果语言建模/指令调优任务头、混合任务批次、损失加权、任务调度策略以及低秩自适应网络。

**📊 数据集**

使用的数据集包括CheckThat! 2024（英文子集）用于命题检测、改造的检索+答案数据集用于证据重排序，以及Politifact、Snopes、FullFact等来源的四类立场检测数据。

**📈 对比分析**

实验将MTL模型与零/少量提示、单任务微调和非LLM基线（XLM‑RoBERTa‑Large、BGE‑reranker）对比。CLS‑MTL在宏F1上相较零/少量提示提升分别为44%、54%和31%；总体性能优于提示基线，且计算、存储与能耗更低。

**⚠️ 局限性**

局限性包括：多任务模型不总能超越单任务微调，需要手工调节权重与顺序；对长句、多段证据和隐含推理的错误率仍较高；训练数据分布偏斜导致误判；模型对不同语言和领域的迁移能力待进一步验证。

---

## 235. Efficient On-Board Processing of Oblique UAV Video for Rapid Flood Extent Mapping

**arXiv ID:** 2601.11290 | [PDF](https://arxiv.org/pdf/2601.11290v1)

**作者:** Vishisht Sharma `[一作]` (Ghent University), Pieter Simoens `[通讯]` (Ghent University)

**通讯引用:** 4218 | [OpenAlex ID](https://openalex.org/A5001314520)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并验证了Temporal Token Reuse（TTR）框架，用以在无人机边缘设备上加速视频语义分割，利用时间冗余动态跳过不变图像块的计算；

**💡 创新点**

将图像块视为令牌，并使用轻量余弦相似度检测静态块，缓存其深度特征并在后续帧中复用，融合SegBlocks的空间稀疏处理，实现时空自适应计算；

**🔧 技术方法**

Patch‑based CNN（ResNet/EfficientNet）、SegBlocks与BlockCopy/BlockSkip、余弦相似度阈值、深度特征缓存及轻量级Token Reuse机制；

**📊 数据集**

Floodwater（新建高分辨率滞洪视频集）、FloodNet、UAVid、A2D2；

**📈 对比分析**

在GTX 1080 Ti与Jetson Orin Nano等平台上与SegFormer、MobileNetV2、BiSeNet等基线对比，TTR在Floodwater上将FPS从30提升至50（≈1.7×），mIoU仅下降0.4%；在UAVid、A2D2等也获得1.6–1.7×加速，mIoU下降≤0.5%；整体提升显著，保持高精度；

**⚠️ 局限性**

仅在图像平面工作，未实现正射纠正与地理坐标投影；阈值固定，缺乏自适应；对极小动态物体检测仍有挑战；需进一步整合实时正射、学习阈值策略与更复杂场景适配。

---

## 236. SUG-Occ: An Explicit Semantics and Uncertainty Guided Sparse Learning Framework for Real-Time 3D Occupancy Prediction

**arXiv ID:** 2601.11396 | [PDF](https://arxiv.org/pdf/2601.11396v1)

**作者:** Hanlin Wu `[一作]` (University of Tokyo), Manabu Tsukada `[通讯]` (University of Tokyo)

**通讯引用:** 1069 | [OpenAlex ID](https://openalex.org/A5067716610)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种实时3D语义占用预测框架SUG-Occ，利用语义与不确定性引导的稀疏学习实现高效占用推断。

**💡 创新点**

创新点包括基于语义与深度不确定性筛选投影、显式距离编码、超交叉稀疏卷积、级联稀疏完成以及OCR掩码解码器。

**🔧 技术方法**

使用语义/不确定性引导的Lift‑Splat‑Shoot、显式距离编码、超交叉稀疏卷积、生成上采样、自适应裁剪、OCR上下文掩码解码、查询去噪和EMA等技术。

**📊 数据集**

在SemanticKITTI数据集上进行评测。

**📈 对比分析**

与MonoScene、TPVFormer、VoxFormer、OccFormer、SparseOcc、ProtoOcc等方法对比，mIoU提升至14.91（比ProtoOcc高7.4%），FPS提升至10.1（比ProtoOcc高57.8%），实现最优性能与实时性。

**⚠️ 局限性**

仅使用单帧相机输入，未利用时序或多车协同信息，限制了进一步提升。

---

## 237. Show me the evidence: Evaluating the role of evidence and natural language explanations in AI-supported fact-checking

**arXiv ID:** 2601.11387 | [PDF](https://arxiv.org/pdf/2601.11387v1)

**作者:** Greta Warren `[一作]` (University of Copenhagen), Isabelle Augenstein `[通讯]` (University of Copenhagen)

**通讯引用:** 4483 | [OpenAlex ID](https://openalex.org/A5018976680)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种将证据与自然语言解释相结合的LLM事实核查系统，探究其对用户决策的影响。

**💡 创新点**

创新在于同时提供证据文档和不同类型的自然语言解释，并评估其对用户信任和过度依赖的影响。

**🔧 技术方法**

使用Qwen2.5-14B-Instruct生成预测、置信度和解释，并通过人机交互实验进行评估。

**📊 数据集**

采用DRUID数据集中的事实核查主张及其证据文档。

**📈 对比分析**

通过3×2×2实验设计对比三种解释类型、AI准确性与置信度，发现证据是最重要的信息来源，解释提高了用户对不确定性的判断，整体表现优于仅提供置信度。

**⚠️ 局限性**

局限在于缺乏源信息的证据文档、样本为非专业的众包工作者、实验环境受限于人工合成界面。

---

## 238. Minimizing the Cost of EFx Allocations

**arXiv ID:** 2601.11372 | [PDF](https://arxiv.org/pdf/2601.11372v1)

**作者:** Eva Deltl `[一作]` `[通讯]` (TU Clausthal), Eva Deltl (TU Clausthal)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在成本约束下求解满足EFx公平且成本最小的分配问题，给出了其NP‑hard性、可多项式核化以及若干参数化可行算法；

**💡 创新点**

首次将EFx公平与成本最小化结合，证明两人已NP‑hard、W[1]‑hard并给出对项数的多项式核化与对有限代价模型的逼近结果；

**🔧 技术方法**

采用参数化复杂度分析、核化技术、归约与逼近不可逼近证明，以及在特定成本模型下的多项式时间算法；

**📊 数据集**

未使用任何真实数据集，全部为理论构造与证明；

**📈 对比分析**

论文未给出实验或实证比较，主要以理论证明和算法复杂度分析为主；

**⚠️ 局限性**

局限性包括：仅适用于所有代理共享同一加性价值函数，成本模型一般情况下仍为NP‑hard且不可逼近，且缺乏实验验证。

---

## 239. Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs

**arXiv ID:** 2601.11369 | [PDF](https://arxiv.org/pdf/2601.11369v1)

**作者:** Marcantonio Bracale Syrnikov `[一作]` (Sapienza University of Rome), Daniele Nardi `[通讯]` (VU Amsterdam)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在重复 Cournot 市场中验证并比较三种治理模式（无治理、宪法式提示、基于治理图的 Institutional AI），证明后者能显著降低企业间的默契协同与市场划分。

**💡 创新点**

首次提出将多代理 LLM 的协同对齐视作机制设计问题，用可发布的治理图（manifest）与可审计日志实现外部可执行的、可追溯的监管框架，突破提示级别的“禁令”易被规避的局限。

**🔧 技术方法**

使用治理图（ABDICO 语法）、Oracle‑Controller 解释器、可变参数政策程序、加密摘要与不可篡改日志，以及线性需求的 Cournot 竞赛环境。

**📊 数据集**

基于公开的两商品两公司 Cournot 实验设置，成本参数为 {40,50}，采用 GPT‑4o、GPT‑3.5‑turbo、Gemini‑2.5‑Flash、Grok‑4‑Fast 等六种模型配置（共 90 次运行/条件）。

**📈 对比分析**

通过对比三种治理条件下的 collusion tier、HHI、CV 等指标，并用 Welch t 检验、比例检验及配对置换检验，发现 Institutional 模式将平均 collusion tier 从约 3.1 降至 1.8，严重协同率由 50% 降至 5.6%，效果显著（Cohen’s d > 1.2）。

**⚠️ 局限性**

实验局限在于仅测试两公司两商品的简化 Cournot 环境，缺乏更大规模、多样化信息与交互的情境；治理阈值固定易被游戏；未验证跨域迁移（如谈判、信息扩散）和模型自适应训练的兼容性。

---

## 240. InterPUF: Distributed Authentication via Physically Unclonable Functions and Multi-party Computation for Reconfigurable Interposers

**arXiv ID:** 2601.11368 | [PDF](https://arxiv.org/pdf/2601.11368v1)

**作者:** Ishraq Tashdid `[一作]` (University of Central Florida), Sazadur Rahman `[通讯]` (University of Central Florida)

**通讯引用:** 189 | [OpenAlex ID](https://openalex.org/A5089960356)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出 InterPUF，一种将差分延迟 PUF 嵌入可重配置中介器并通过多方计算实现分布式根信任的低开销芯片组身份验证框架。

**💡 创新点**

创新点在于：①把 PUF 直接集成到可重配置中介器的路由网格中，形成全系统根信任；②采用无中心验证的 MPC，消除单点故障；③通过黄金无参考自检、挑战加密与哈希化的链路，提升建模抵抗与最小信任；④实现极低的面积（<0.23%）和功耗（<0.072%）开销。

**🔧 技术方法**

使用的技术包括：差分路由延迟 PUF、可重配置网格路由、SHA‑256 与 HKDF 哈希、Yao 加密电路与 OT 的两方 MPC、Majority 计票错误校正、PyPUF 模拟、RTL 合成与功耗分析。

**📊 数据集**

使用的实验数据集为：基于 PyPUF 生成的 5 个合成芯片实例，8000 条 CRP 进行机器学习攻击训练，3000 条 CRP 用于测试；并在多种 SoC 基准（CVA6、NVDLA、RISC‑V 等）上进行面积、功耗与时延仿真。

**📈 对比分析**

方法比较：与 SECT‑HI、PQC‑HI 等现有方案对比；InterPUF 仅占 0.05–0.23% 的面积与 0.005–0.072% 的功耗，且认证时延仅为 2 ns（PUF）+ <32 ns（哈希）+ 5–30 µs（MPC），模型攻击准确率约 47%（随机水平）。在这些指标上，InterPUF 显著优于传统集中式或重计算方案。

**⚠️ 局限性**

局限性：目前仅在 RTL 与仿真层面验证，缺乏硅片与 FPGA 实验；未充分测试极端 PVT、老化与环境应力；对攻击者假设为半诚实；若存在极端侧信道或故障注入，仍需进一步补强。

---

## 241. Think-Clip-Sample: Slow-Fast Frame Selection for Video Understanding

**arXiv ID:** 2601.11359 | [PDF](https://arxiv.org/pdf/2601.11359v1)

**作者:** Wenhui Tan `[一作]` (Renmin University of China), Zhenbo Luo `[通讯]` (Xiaomi Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无训练的框架TCS，用于提升多模态大语言模型在长视频理解中的性能，主要通过多查询推理和片段级慢-快采样两大模块；

**💡 创新点**

创新点在于（1）多查询推理：让LLM根据不同视角生成多条查询，提升检索多样性；（2）片段级慢-快采样：先识别高相似度片段再分配更多帧，平衡局部细节与全局语境；

**🔧 技术方法**

使用技术包括：LLM生成查询、CLIP-ViT-Large进行查询-帧相似度计算、Gaussian平滑、峰值阈值检测、慢-快采样策略以及与基线MLLM（Qwen2‑VL‑7B、MiMo‑VL‑7B）的无缝集成；

**📊 数据集**

实验数据集涵盖三大长视频理解基准：MLVU（2593问）、LongVideoBench（1337问）和VideoMME（短、中、长各900问，总计2700问）；

**📈 对比分析**

与传统帧采样方法（AKS、Q‑Frame）及长视频LLM（Video‑XL、LongVILA）对比，TCS在三大基准上平均提升3-4%准确率，最高可达6.9%，并在相同帧预算下将推理时间降低50%；

**⚠️ 局限性**

局限性包括：依赖CLIP/ VLM进行相似度评分，可能忽略音频或字幕信息；阈值α和快帧比例需要手动调节；在极长视频或极低帧数预算下仍可能出现覆盖不足或细节丢失；

---

## 242. Assessing Building Heat Resilience Using UAV and Street-View Imagery with Coupled Global Context Vision Transformer

**arXiv ID:** 2601.11357 | [PDF](https://arxiv.org/pdf/2601.11357v1)

**作者:** Steffen Knoblauch `[一作]` (Heidelberg University), Alexander Zipf `[通讯]` (Heidelberg University)

**通讯引用:** 10754 | [OpenAlex ID](https://openalex.org/A5091474205)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一套跨视角融合的机器学习框架，利用无人机与街景图像提取建筑热相关属性，并与热红外数据关联，识别热暴露不平等。

**💡 创新点**

创新点在于首次将无人机与街景双模态图像通过Coupled Global Context Vision Transformer（CGCViT）进行跨视角融合，显著提升建筑属性分类精度并揭示其与热暴露的关联。

**🔧 技术方法**

采用了CGCViT、双模态交叉视角学习、全局查询token、自注意力机制、交叉熵与focal loss训练，并用Kruskal‑Wallis、Pearson相关分析评估属性与热红外的关系。

**📊 数据集**

数据集包含Panoramax街景全景图、OpenAerialMap无人机正射影像、Geofabrik建筑轮廓及HotSat‑1热红外影像，共覆盖约4,965栋住宅建筑。

**📈 对比分析**

在UAV‑only、SV‑only和双模态融合三种设置下进行5折空间交叉验证；融合模型在多项属性的加权F1上提升0–9.3%，并显示预测属性与HotSat‑1 TIR值呈显著负相关。

**⚠️ 局限性**

局限性包括墙体材质识别仍较困难、样本受限于可观测视角（如狭窄人行道与被遮挡建筑），以及在不同城市密度与建筑风格下的推广性尚未验证。

---

## 243. Unlocking the Potentials of Retrieval-Augmented Generation for Diffusion Language Models

**arXiv ID:** 2601.11342 | [PDF](https://arxiv.org/pdf/2601.11342v1)

**作者:** Chuanyue Yu `[一作]` (Nankai University), Ziwei Zhang `[通讯]` (Beihang University)

**通讯引用:** 5156 | [OpenAlex ID](https://openalex.org/A5100442619)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了将离散扩散语言模型（DLM）与检索增强生成（RAG）结合的可行性，发现 DLM 在 RAG 框架下易出现语义漂移（RSD）问题，并针对这一问题提出了 SPREAD 方法；

**💡 创新点**

创新点包括①正式定义并量化 DLM 在 RAG 中的语义漂移指标 RSD；②提出了基于查询相关性的动态去噪策略（Query‑Token Semantic Similarity + Relevance‑Guided Token Selection），通过在每一步去噪时引入查询语义约束，显著抑制语义漂移并提升答案精度；

**🔧 技术方法**

采用的技术包括离散扩散语言模型（LLaDA、Dream）、检索增强生成框架（基于 NV‑Embed‑v2 的向量检索）、查询‑词语语义相似度评估（内部隐藏状态、余弦相似度 + Sigmoid映射）、自适应 token 选择策略、以及对生成过程的多维度评估（Precision、RSD、Copy Rate、Redundancy 等）；

**📊 数据集**

使用了六个开放域问答基准数据集：Natural Questions、TriviaQA、HotpotQA、MuSiQue、MultiHop‑RAG、UltraDomain；

**📈 对比分析**

与多种基线去噪策略（confidence、entropy、random 等）在 LLaDA 与 Dream 两种 DLM 上进行对比，主要评估指标为 Precision、Recall、RSD、Copy Rate 等；实验结果显示 SPREAD 在 Precision 上平均提升 15‑31%，RSD 下降 10‑61%，同时保持 Recall 与基线相近，且计算开销仅微增；

**⚠️ 局限性**

局限性主要是：①性能高度依赖检索到的上下文质量，若检索不佳则无法充分发挥作用；②实验仅覆盖问答任务，未验证在更开放式生成（如故事创作）等场景的有效性。

---

## 244. Neural Chain-of-Thought Search: Searching the Optimal Reasoning Path to Enhance Large Language Models

**arXiv ID:** 2601.11340 | [PDF](https://arxiv.org/pdf/2601.11340v1)

**作者:** Guoming Ling `[一作]` (Sun Yat-sen University), Liang Lin `[通讯]` (Sun Yat-sen University)

**通讯引用:** 33732 | [OpenAlex ID](https://openalex.org/A5100412937)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种Neural Chain-of-Thought Search (NCoTS)，将推理过程视为动态搜索，在关键决策点主动寻找最优思考路径；

**💡 创新点**

创新点在于把思考步骤抽象为可枚举的操作集合，并采用双因子启发式（准确性潜能与进度估计）进行一跳前瞻搜索，显著提升推理准确率与效率；

**🔧 技术方法**

使用教师蒸馏得到的路径潜能估计器、基于隐藏状态的进度估计器，以及KV缓存的一跳前瞻搜索算法，结合LLM的文本生成；

**📊 数据集**

在AMC23、ARC-C、GPQA、GSM8K四大公开推理基准上进行评估，并用LogicQA、Math500、AIME22-25、HumanEval等数据集训练估计器；

**📈 对比分析**

与标准采样、NoWait、AdaptThink、ThinkPrune、Laser等六个基线比较，平均提升约3.5%准确率、减少约22%生成长度，效率指标η提升至1.5-1.6，显示出优越性能；

**⚠️ 局限性**

局限性包括：思考令牌集合固定、主要针对英文STEM推理；教师监督限制探索范围；仅采用局部一跳前瞻，难以处理极长或复杂全局规划；需进一步扩展至多语种或创造性任务。

---

## 245. F-Actor: Controllable Conversational Behaviour in Full-Duplex Models

**arXiv ID:** 2601.11329 | [PDF](https://arxiv.org/pdf/2601.11329v1)

**作者:** Maike Züfle `[一作]` (Karlsruhe Institute of Technology), Tsz Kin Lam `[通讯]` (NatWest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了F‑Actor——首个可按指令控制的全双工语音对话模型，支持说话者声音、话题、回声/打断、对话开启等多维度自适应；采用单阶段训练，只微调LLM，冻结音频编码器，使用2000小时数据即可训练；同时将文本流与音频流同步生成，提高语音与文本一致性。

**💡 创新点**

1）公开了第一个可指令跟随的全双工语音模型；2）仅微调LLM、冻结音频编码器，实现低成本高效训练；3）将说话者嵌入与文本指令前缀统一拼接，实现多维度行为控制；4）通过词级对齐和音频延迟，显著提升语音与文本同步与一致性。

**🔧 技术方法**

采用Llama‑3.2‑1B‑Instruct作为LLM骨干；NanoCodec（FSQ）语音编码器，独立嵌入层和多头线性解码器；说话者嵌入来自ECAPA‑TDNN；指令前缀与说话者嵌入拼接；使用VAD、Parakeet等检测回声/打断；单阶段训练，梯度累积，音频延迟与词级对齐。

**📊 数据集**

Behavior‑SD（2164h）合成多轮双说话者对话数据，包含叙事结构、回声和打断注释；使用Kaldi强制对齐获得文本-语音对齐；在此基础上构造指令前缀并重写叙事。

**📈 对比分析**

与Behavior‑SD原始对话、Moshi、dGLSM进行对照。评估指标包括语音文本perplexity、UTMOS、WER、说话时间平衡、回声/打断相关系数、说话者一致性、对话开启准确率。最佳配置实现perplexity≈21.45、WER≈7.6%、UTMOS≈3.4‑3.5，回声相关系数0.54、打断相关系数0.25，开启准确率99%。相较基线性能有显著提升，但仍低于Oracle（UTMOS3.78、WER4.5%）。

**⚠️ 局限性**

1）数据为合成语音，真实性有限；2）只能使用限定的52个说话者，无法实现声纹克隆；3）打断捕捉准确性不足，相关系数仅0.25；4）RVQ编码器训练导致语音质量下降；5）需精细对齐与音频延迟，实现成本较高；6）长对话或多轮持续性表现仍受限。

---

## 246. Can Small Agent Collaboration Beat a Single Big LLM?

**arXiv ID:** 2601.11327 | [PDF](https://arxiv.org/pdf/2601.11327v1)

**作者:** Agata Żywot `[一作]` (University of Amsterdam), Maarten de Rijke `[通讯]` (University of Amsterdam)

**通讯引用:** 29323 | [OpenAlex ID](https://openalex.org/A5031439294)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在 GAIA 基准上，利用工具增强的较小语言模型是否能匹配或超越大型单体模型的能力

**💡 创新点**

证明工具辅助带来的提升远大于纯规模扩展，且明确了不同规模模型在显式思考与工具使用下的优势与劣势

**🔧 技术方法**

采用改进的 Agentic‑Reasoning 框架，整合 Web‑Search、Coding、Mind‑Map 三种工具，并实验不同模型规模（4B–32B）与思考策略（无、规划者、完整）

**📊 数据集**

使用 GAIA benchmark（L1–L3 三级难度，共 165 题）进行评测

**📈 对比分析**

与单体无工具版本对比，发现 4B‑Instruct+Agentic 在 GAIA 上达 18.18% ACC，超过 32B 无工具的 12.73%；工具使用显著提升小模型性能，显式思考在某些规模与难度下会导致性能下降，整体提升幅度在 5–25% 之间

**⚠️ 局限性**

仅在 GAIA 上评估，工具实现和检索质量有限，未覆盖其他 agentic 基准，且对动态思考策略和模型不确定性的探索仍不足

---

## 247. SAMannot: A Memory-Efficient, Local, Open-source Framework for Interactive Video Instance Segmentation based on SAM2

**arXiv ID:** 2601.11301 | [PDF](https://arxiv.org/pdf/2601.11301v1)

**作者:** Gergely Dinya `[一作]` (Pázmány Péter Catholic University), Anna Gelencsér-Horváth `[通讯]` (Pázmány Péter Catholic University)

**通讯引用:** 12 | [OpenAlex ID](https://openalex.org/A5084032474)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本工作开发了开源本地框架 SAMannot，将 SAM2 集成到人机交互视频实例分割与跟踪工作流中，并实现了低内存、可视化友好的注释界面。

**💡 创新点**

创新点包括对 SAM2 的内存优化滑动窗口、持久化实例 ID 映射、自动骨架化提示生成与检查点锁定机制，从而在中等显存 GPU 上实现长序列实时推理和自动化传播。

**🔧 技术方法**

主要技术包括 PyTorch+SAM2 推理、Tkinter GUI、异步多线程推理、骨架化点提示、分块处理与 GPU 资源监控。

**📊 数据集**

评估数据集使用 DAVIS 2017 与 LVOS 两个公开视频分割基准的子集，并在动物行为跟踪实验中进行验证。

**📈 对比分析**

在同一任务下与 SAM2 基线对比，使用 IoU、Dice 与像素准确率衡量，平均 IoU 0.918、Dice 0.951、PixelAcc 0.989，显著降低显存占用的同时保持高精度。

**⚠️ 局限性**

局限性在于高密度提示会导致分割碎片；受 SAM2 对点提示数量的内在限制，遮挡或光照极端条件下精度可能下降；需要至少 6GB VRAM 的显卡。

---

## 248. Seek and You Shall Find: Design & Evaluation of a Context-Aware Interactive Search Companion

**arXiv ID:** 2601.11287 | [PDF](https://arxiv.org/pdf/2601.11287v1)

**作者:** Markus Bink `[一作]` (Neu-Ulm University of Applied Sciences), David Elsweiler `[通讯]` (University of Regensburg)

**通讯引用:** 2073 | [OpenAlex ID](https://openalex.org/A5041921674)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了一个可嵌入搜索引擎页面的交互式搜索伴侣，通过右侧侧边栏提供上下文提示，帮助用户澄清信息需求、改进查询、探索结果并降低偏见。

**💡 创新点**

创新点在于将微型学习提示与用户行为实时结合，形成轻量级、无缝集成的“提示-执行”机制，避免传统长篇教程的认知负荷，同时实现按需展示而非持续弹窗。

**🔧 技术方法**

技术实现主要采用前端侧边栏组件、基于用户交互事件（查询提交、停留时间、点击结果）触发的逻辑，并提供可点击的查询建议与“了解更多”展开面板。

**📊 数据集**

实验数据来源于170名流利英语的健康/医疗搜索任务参与者，使用Prolific平台招募，任务涵盖多种医学主题（如益生菌、抗氧化剂等）。

**📈 对比分析**

通过对照实验将基线10蓝链系统与伴侣系统进行比较；结果显示伴侣组查询数增加约75%、结果页数翻倍，且在难度较高的任务中准确率略有提升，整体准确率与基线相近。

**⚠️ 局限性**

局限包括：提示内容为预设而非实时LLM生成、仅在单一健康领域测试、易任务效果不佳、可能导致认知过载或过度改写查询，以及实验环境缺乏真实网络搜索场景。

---

## 249. Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning

**arXiv ID:** 2601.11393 | [PDF](https://arxiv.org/pdf/2601.11393v1)

**作者:** Haomiao Tang `[一作]` (Affiliation 1), Shu-Tao Xia `[通讯]` (Affiliation 2)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

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

信息不足

---

## 250. Hyperparameter Optimization of Constraint Programming Solvers

**arXiv ID:** 2601.11389 | [PDF](https://arxiv.org/pdf/2601.11389v1)

**作者:** Hedieh Haddad `[一作]` (University of Luxembourg), Pascal Bouvry `[通讯]` (University of Luxembourg)

**通讯引用:** 10202 | [OpenAlex ID](https://openalex.org/A5058311932)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Probe and Solve算法（PSA），一种在固定时间预算下对约束求解器进行自动超参数优化的两阶段框架，集成于CPMpy库并首次支持ACE求解器；

**💡 创新点**

创新点在于将贝叶斯优化与Hamming距离搜索嵌入PSA，提供可插拔的时间管理、初始化、演化和停止策略模块，显著提升不同求解器的解质量；

**🔧 技术方法**

技术主要包括贝叶斯优化（Gaussian Process + acquisition function）、Hamming距离搜索、动态/静态时间分配、Luby与几何时间演化、以及解决阶段的目标剪枝；

**📊 数据集**

数据集为114个来自XCSP3 benchmark的组合优化和满足性实例，分别在ACE（150k+配置空间）和Choco（136k+配置空间）上进行实验；

**📈 对比分析**

通过与默认配置以及PSA-BO/PSA-Hamming两种配置的对比，实验显示PSA-BO在ACE上提升25.4%实例，在Choco上提升38.6%实例，且总比Hamming搜索和默认设置优越；

**⚠️ 局限性**

局限性包括仅在两款求解器上验证、超参数固定比例（20%探测），未考虑多目标优化或自适应策略，且对实例特征的利用不足。

---

## 251. RITA: A Tool for Automated Requirements Classification and Specification from Online User Feedback

**arXiv ID:** 2601.11362 | [PDF](https://arxiv.org/pdf/2601.11362v1)

**作者:** Manjeshwar Aniruddh Mallya `[一作]` (Lero Research Ireland Centre for Software), Jacek Dąbrowski `[通讯]` (Lero Research Ireland Centre for Software)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并展示了一款名为RITA的端到端工具，整合轻量级开源大语言模型，对线上用户反馈进行请求分类、非功能需求识别以及自然语言需求规范和用户故事的自动生成，并可直接将生成的需求导入Jira；

**💡 创新点**

将先前独立评估的LLM反馈分析技术整合到一个统一工作流中，并实现了与主流项目管理工具Jira的无缝对接，首次提供从原始反馈到可直接使用的需求工件的完整链路；

**🔧 技术方法**

利用React构建前端、FastAPI编写后端、SQLite存储中间结果、Ollama托管轻量级LLM进行推理，并通过Jira API导出需求；

**📊 数据集**

收集来自应用商店评论、调查问卷和问题报告的文本数据，支持txt、csv、xls等常见格式，未使用特定公开数据集；

**📈 对比分析**

目前尚未提供系统性量化对比；论文计划通过与人工分析比较测量时间效率、识别数量及人工修订工作量，并收集用户访谈与问卷反馈评估可用性与效果；

**⚠️ 局限性**

模型准确性与一致性不足，生成的需求有时不符合结构化标准，工具稳定性不完全，Jira导出格式与工业规范不完全匹配；

---

## 252. AstroReason-Bench: Evaluating Unified Agentic Planning across Heterogeneous Space Planning Problems

**arXiv ID:** 2601.11354 | [PDF](https://arxiv.org/pdf/2601.11354v1)

**作者:** Weiyi Wang `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Fudan University)

**通讯引用:** 17130 | [OpenAlex ID](https://openalex.org/A5044665993)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AstroReason‑Bench benchmark suite，用于评估 LLM 代理在多样化空间规划问题（SatNet、Revisit、Regional Coverage、Stereo Imaging、Latency）中的适应性与性能。

**💡 创新点**

创新点在于将多类空间规划任务统一到单一物理引擎与 agent‑centric 接口；通过高保真 SGP4、爬坡动力学和资源模型，提供真实物理约束的评测环境，并展示 LLM 代理在面对复杂组合约束时的零样本推理与工具调用能力。

**🔧 技术方法**

采用 LLM 代理（Claude、Gemini、DeepSeek 等）结合 ReAct+Python API、MCP 语义工具；物理层使用 SGP4 轨道传播、三角形速度曲线爬坡模型、功率与存储资源管理；评估基于 Python 接口与 JSON‑MCP 的双模交互。

**📊 数据集**

使用公开的 TLE 数据、全球城市数据库（40k+城市）、以及程序化生成的 4 天时域（2025‑07‑17~21）的场景；共五类任务与对应的目标与约束；实验包含 150 条完整任务模拟。

**📈 对比分析**

与传统优化基线（MILP、RL、贪婪启发式、模拟退火）比较；LLM 代理在组合约束任务（Stereo、Latency）上取得 0–18% 覆盖率，优于基线；但在搜索密集任务（SatNet、Revisit）上仍落后（U_rms≈0.53‑0.59 vs MILP 0.30；M_gap≈18‑20h vs SA 13.6h）。

**⚠️ 局限性**

局限包括：只测试 Flash‑class LLM，缺乏大模型与更复杂工作流；计算资源与专用优化器不匹配；样本量有限，缺乏置信区间；未覆盖系统设计与深空轨道规划；代理在空间推理与探索上仍显不足。

---

## 253. Beer-Lambert Autoencoder for Unsupervised Stain Representation Learning and Deconvolution in Multi-immunohistochemical Brightfield Histology Images

**arXiv ID:** 2601.11336 | [PDF](https://arxiv.org/pdf/2601.11336v1)

**作者:** Mark Eastwood `[一作]`, Fayyaz Minhas `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于U-Net编码器和可学习Beer‑Lambert解码器的无监督多染色体素分离模型，可在RGB mIHC图像中分离超过3种染色物质，并生成稀疏浓度图。

**💡 创新点**

将物理模型与深度学习相结合，联合学习染色基矩阵与浓度，并通过熵、重叠、颜色一致性等多重无监督损失实现低混叠、锐利分离；首次实现对K>3染色的稳定分离。

**🔧 技术方法**

U‑Net编码器、可学习的Beer‑Lambert解码器、VGG‑19感知重建损失、熵损失、top‑k 重叠损失、颜色一致性约束与稀疏掩膜损失；整体训练为无监督。

**📊 数据集**

共32,547张960×960像素的colorectal cancer WSI补丁，染色包含5种染色剂（H、CDX2、MUC2、MUC5、CD8），来自128张WSI。

**📈 对比分析**

与传统基于矩阵的Beer‑Lambert分离方法对比，利用余弦相似度衡量染色通道交叉，结果显示本方法显著降低通道间混叠，重建误差更低，且生成的单通道渲染更清晰。

**⚠️ 局限性**

学习得到的染色基矩阵仅针对特定面板和队列，跨实验室迁移需微调；掩膜损失仅针对稀有染色且依赖手工阈值；方法不保证绝对定量，仅提供相对分离。

---

## 254. ProjecTA: A Semi-Humanoid Robotic Teaching Assistant with In-Situ Projection for Guided Tours

**arXiv ID:** 2601.11328 | [PDF](https://arxiv.org/pdf/2601.11328v1)

**作者:** Hanqing Zhou `[一作]` (Southern University of Science and Technology), Pengcheng An `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 730 | [OpenAlex ID](https://openalex.org/A5014119112)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估了一款名为ProjecTA的半人形教学助手机器人，利用头戴投影仪、手部动作和语音，为参观者在工作坊中提供现场投影式的引导式学习体验。

**💡 创新点**

将机器人投影技术与手势与语音多模态协同相结合，在移动学习环境中实现对目标物体的即时、空间化信息投射，显著降低了学习者的外在认知负荷。

**🔧 技术方法**

结合ROS、MQTT、Jetson Orin Nano、Aurzen头戴投影机、6‑DOF手臂与20‑DOF手掌动作，配合LLM（GPT‑4o）脚本生成与多模态协同调度。

**📊 数据集**

通过专家工作坊与现场身体实验收集设备信息、学习要点、手势库等内部资源，实验采用24名初学者对照实验。

**📈 对比分析**

采用双盲、交叉设计的混合方法研究，将ProjecTA与基准屏幕显示系统进行对照；结果显示ProjecTA在外在认知负荷、可用性、可视化效用和多模态互补性上均显著优于屏幕版本，且测验成绩相当。

**⚠️ 局限性**

实验样本规模有限，投影受光照、表面不平整和遮挡影响，机器人仅支持单向交互，且在更广泛的学习场景中的适用性尚待验证。

---

## 255. GENPACK: KPI-Guided Multi-Objective Genetic Algorithm for Industrial 3D Bin Packing

**arXiv ID:** 2601.11325 | [PDF](https://arxiv.org/pdf/2601.11325v1)

**作者:** Dheeraj Poolavaram `[一作]` (Augsburg University of Applied Sciences), Sebastian Dorn `[通讯]` (Augsburg University of Applied Sciences)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结合构造启发式、遗传算法和后处理的工业级三维箱装算法（GENPACK），通过多KPI加权实现对空间利用、稳定性、支撑和平衡的统一优化；

**💡 创新点**

①将KPI直接融入多目标适应度，提供可调节的工业评价标准；②采用层级染色体表示与专用交叉/变异算子，保持可行性并高效搜索；③构建完整的三阶段流水线（启发式→GA细化→后处理），实现高质量、可部署的包装方案；

**🔧 技术方法**

层级染色体表示、KPI加权适应度、专用交叉/变异算子、最大矩形启发式、方向压缩后处理、欧几里得中心平衡等技术；

**📊 数据集**

BED‑BPP真实工业订单数据集（1500个欧式托盘订单），包含真实尺寸与重量信息；

**📈 对比分析**

与七类基线（极点、最大矩形、Sisyphus、O3D‑BPP‑PCT、GOPT、Extreme Point）比较，GENPACK在绝对/相对密度、表面支撑、平衡等KPI上均显著优于其他方法；平均运行时间约29 s/单订单（比启发式快但慢于纯学习方法），但实现的空间利用率提升35%且稳定性提升15–20%；

**⚠️ 局限性**

仅支持轴对齐长方体，KPI加权需人工调参，遗传算子和后处理仍有计算量；对动态或不确定订单、非轴对齐物品的适应性有限。

---

## 256. Context-Aware Semantic Segmentation via Stage-Wise Attention

**arXiv ID:** 2601.11310 | [PDF](https://arxiv.org/pdf/2601.11310v1)

**作者:** Antoine Carreaud `[一作]` (Ecole Polytechnique Federale de Lausanne), Adrien Gressin `[通讯]` (University of Applied Sciences Western Switzerland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了 CASWiT，这是一种双分支 Swin Transformer 体系，在 UHR RGB 航测影像语义分割中通过跨尺度交叉注意力实现高分辨率细节与低分辨率全局上下文的融合。

**💡 创新点**

创新点在于：①阶段级跨尺度交叉注意力与可学习门控实现早期上下文注入；②提出基于 SimMIM 的自监督预训练方案，采用 75% 高分辨率随机遮掩与 50% 低分辨率中心遮掩共同训练；③将两种分支共享同一 Swin 架构并在每个阶段交互，提升细粒度与全局语义一致性。

**🔧 技术方法**

使用技术包括 Swin Transformer 编码器、跨尺度交叉注意力块、可学习门控残差、UPerNet 解码器、SimMIM‑style 自监督预训练、以及 mIoU、mF1、mBIoU 等多维度评估指标。

**📊 数据集**

数据集包括：FLAIR‑HUB（RGB 512×512 UHR）、URUR（5120×5120 UHR）以及 SWISSIMAGE（无标签 orthophoto，约 1067 Gpx）用于预训练。

**📈 对比分析**

在 FLAIR‑HUB RGB‑only UHR 协议下，CASWiT 取得 65.83% mIoU（相较基线提升约 2.7%），mF1 78.22%，mBIoU 36.90%；在 URUR 上得到 49.1% mIoU（比前沿模型提升约 0.9%），同时保持与其他 UHR 模型相近的显存占用。

**⚠️ 局限性**

局限性包括：对极细小目标的边界恢复仍有提升空间；预训练所需的大量无标签数据对数据获取成本有一定影响；在多模态（NIR、DSM 等）输入下的性能提升尚未充分验证。

---

## 257. "Can You Tell Me?": Designing Copilots to Support Human Judgement in Online Information Seeking

**arXiv ID:** 2601.11284 | [PDF](https://arxiv.org/pdf/2601.11284v1)

**作者:** Markus Bink `[一作]` (Neu-Ulm University of Applied Sciences), David Elsweiler `[通讯]` (University of Regensburg)

**通讯引用:** 2073 | [OpenAlex ID](https://openalex.org/A5041921674)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并测试了一种基于大型语言模型的对话式 copilot，用来帮助用户在信息检索时进行评价和批判性思考，而不是直接给出答案。

**💡 创新点**

创新点在于将教育性辅导与搜索界面融合，采用 Socratic 交互式提问和数字素养策略（如侧向阅读、SIFT 框架）来 scaffold 用户的评价过程，强调元认知反思而非即时回答。

**🔧 技术方法**

核心技术是使用 Gemini‑2.5‑flash LLM 进行检索增强生成（RAG）生成 AI 概览，以及与用户对话的交互式辅导；搜索结果由 Brave Search API 提供；所有交互通过聊天窗口嵌入在传统 SERP 界面中。

**📊 数据集**

实验数据来自 261 名 Prolific 平台受试者，使用三类预设的保健问题（抗氧化剂、褪黑激素、牵引疗法）以及对应的 Cochrane Review 作为真值；每个受试者在三种界面条件（10‑blue‑links、ai‑overview、copilot）中随机分配。

**📈 对比分析**

对比方法为三组间的随机对照实验，使用逻辑回归和线性回归检验答案正确率、置信度、页面浏览数和查询次数；结果显示 copilot 并未显著提升答案正确率或搜索参与度，虽然用户在聊天中投入更多时间并表现出更高的元认知参与度。

**⚠️ 局限性**

限制包括：仅测试三类医学问题，缺乏长期跟踪；人为构造错误的 AI 概览可能影响外推；仅使用 Gemini‑2.5‑flash，未检验其他模型；受试者来自 Prolific，技术熟练度高，生态效度有限；实验为单次会话，无法评估数字素养的持续性。

---

## 258. Polar Orbit Decoding: Universal Parallel Soft Decoding via Automorphism Orbits

**arXiv ID:** 2601.11373 | [PDF](https://arxiv.org/pdf/2601.11373v1)

**作者:** Pin-Jing Li `[一作]` (National Yang Ming Chiao Tung University), Yu-Chih Huang `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 2547 | [OpenAlex ID](https://openalex.org/A5082244371)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Polar Orbit Decoding (𝖯𝖮𝖣) 并实现了对任意二进制线性块码的并行软判决解码；

**💡 创新点**

创新点在于利用块码自同构群的轨道产生相同动态冻结约束的多条解码轨迹，无需重新设计冻结集即可并行解码；

**🔧 技术方法**

采用极化变换、极化子码、自动同构群的基和强生成集（BSGS）以及 Schreier–Sims 算法构造轨道；

**📊 数据集**

实验使用扩展 BCH 码（16,7）、（64,16）、（64,36）以及扩展 Golay 码（24,12）；

**📈 对比分析**

与 ML、SC、SCL、HD 等传统解码方法对比，𝖯𝖮𝖣 在相同有效列表大小下逼近 ML 性能，同时显著降低解码延迟；

**⚠️ 局限性**

受限于自同构群的大小与轨道构造的计算复杂度，某些块码的可用轨道数有限，导致并行解码的灵活性受到一定限制。

---

## 259. Evaluating LLM Behavior in Hiring: Implicit Weights, Fairness Across Groups, and Alignment with Human Preferences

**arXiv ID:** 2601.11379 | [PDF](https://arxiv.org/pdf/2601.11379v1)

**作者:** Morgane Hoffmann `[一作]` (Malt), Charles Pebereau `[通讯]` (Malt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型在招聘场景中的决策逻辑进行实验性审计，揭示其对技能、经验、平台声誉等核心信号的加权方式。

**💡 创新点**

创新点在于将劳动力经济学的因果设计（全因子实验）与LLM评分结合，系统评估模型对不同属性的权重及其与人类招聘者的差异。

**🔧 技术方法**

使用Gemini 2.0 Flash进行评分，并通过OLS回归、交互效应分析和稳健性检验来量化模型的隐含权重。

**📊 数据集**

合成数据来源于欧洲大型自由职业平台的真实自由职业者档案与项目描述，按全因子设计生成10,800个自由职业者档案和16个项目简介，共计172,800个评分对。

**📈 对比分析**

通过将LLM评分与人类招聘者在相同实验设计下的评分进行直接对比，评估两者在属性加权上的一致性；模型在绝对评分上表现出高解释力（R²≈0.90），但在相对排名情境下表现略有差异。

**⚠️ 局限性**

局限包括仅评估单一模型（Gemini 2.0 Flash）、受实验设计与提示敏感性限制、对不同语言与行业的外推性不足，以及未对模型进行公平性校准。

---

## 260. Human Factors in Immersive Analytics

**arXiv ID:** 2601.11365 | [PDF](https://arxiv.org/pdf/2601.11365v1)

**作者:** Yi Li `[一作]` (TU Wien), Tim Dwyer `[通讯]` (Monash University)

**通讯引用:** 5784 | [OpenAlex ID](https://openalex.org/A5008149778)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并规划了关于人因学在沉浸式分析中的研讨会，阐述目标、议程、参与者、活动等内容。

**💡 创新点**

将认知、物理、协作等人因系统性纳入沉浸式分析研究议程，倡导跨学科合作与评估方法的革新。

**🔧 技术方法**

采用线上预研、Google Drive协作、现场小组讨论、视频展示等技术手段，未使用特定实验技术。

**📊 数据集**

无实验数据集，仅使用研讨会相关材料和预先提交的演示视频。

**📈 对比分析**

未进行实验比较，主要通过专家评议和工作坊讨论形成共识。

**⚠️ 局限性**

受限于参与人数、缺乏实证评估，且工作坊产出依赖参与者质量，无法提供客观指标。

---

## 261. How Much Would a Clinician Edit This Draft? Evaluating LLM Alignment for Patient Message Response Drafting

**arXiv ID:** 2601.11344 | [PDF](https://arxiv.org/pdf/2601.11344v1)

**作者:** Parker Seegmiller `[一作]` (Dartmouth), Sarah Masud Preum `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

无法确定论文的具体内容

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

## 262. Cutting Corners on Uncertainty: Zonotope Abstractions for Stream-based Runtime Monitoring

**arXiv ID:** 2601.11358 | [PDF](https://arxiv.org/pdf/2601.11358v1)

**作者:** Bernd Finkbeiner `[一作]` (CISPA Helmholtz Center for Information Security), Paul Kröger `[通讯]` (Carl von Ossietzky Universität)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5069467441)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在存在传感器噪声和校准误差时，基于流式规范的在线运行时监控问题，提出一种利用Zonotope抽象域进行符号推理并实现有限内存监控的算法。

**💡 创新点**

创新点在于：①首次将Zonotope作为抽象域引入一般流式监控，能够在符号推理的基础上实现对无限状态空间的有限记忆逼近；②提出了在每个时间步对监控状态进行Zonotope over‑approximation 的通用框架；③系统评估了多种现有Zonotope近似策略在监控精度与误报率上的差异，揭示了当前方法在处理校准误差与随机噪声时的局限。

**🔧 技术方法**

主要技术包括：符号（affine）算术表示、ISO 5725 误差模型（常量校准误差 + 随机测量误差）、Zonotope 近似与压缩算法（adaptive、combastel、girard、methA、scott、pca、interval hull），以及基于三元触发比较运算符的鲁棒判定。

**📊 数据集**

使用了两组工业机器人轨迹数据集：①受限机器人（二维坐标系中受限区域内运动）和 ②全向机器人（可自由二维运动，测量加速度与方向），通过随机行走产生 200 或 1000 条测量序列并加入噪声来构造实验输入。

**📈 对比分析**

对不同 Zono 近似策略的比较采用了曼哈顿 Hausdorff 距离评估误差、误报率（false‑positive rate）以及随时间变化的平均误差曲线。实验表明：在受限机器人中，七种策略误差极小、误报率低；在全向机器人中，误差随时间显著累积，误报率从 0.025 至 0.065 之间变化，表明策略选择对监控性能影响显著。

**⚠️ 局限性**

主要局限性包括：①现有 Zono 近似方法忽略未来状态外推对误差传播的影响，导致误差持续累积；②对校准误差与随机误差的区分不足，削弱了精度提升；③算法对复杂规范的适用性仍有限，需进一步研究针对监控特定需求的专用近似方法。

---

## 263. Distributed Control Barrier Functions for Safe Multi-Vehicle Navigation in Heterogeneous USV Fleets

**arXiv ID:** 2601.11335 | [PDF](https://arxiv.org/pdf/2601.11335v1)

**作者:** Tyler Paine `[一作]`, Michael Benjamin `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出在异构无人水面舰队与有人艇并行作业时，利用分布式控制屏障函数（CBF）实现实时碰撞规避；

**💡 创新点**

创新点在于采用对抗性最坏情况假设、分布式控制过滤器并将COLREGS行为与CBF结合，以无需共享意图的方式实现安全保障；

**🔧 技术方法**

主要技术包括基于CBF的控制过滤器、二阶（一阶）无人艇动力学模型、线性规划求解最坏输入、以及使用Slack变量保证可行性；

**📊 数据集**

实验数据来源于100,000次Monte‑Carlo模拟碰撞实验以及四艘异构USV（Heron、BlueBoat、WAM‑V）和一艘有人艇在圆形“争斗”任务中的实地运行；

**📈 对比分析**

通过与单纯COLREGS行为、单纯CBF过滤以及两者结合三种策略比较，结果显示CBF可完全消除碰撞、COLREGS可显著降低近碰事件并提升效率，二者结合在安全性和效率上均优于单一方案；

**⚠️ 局限性**

局限性包括：对最坏情况假设导致过度保守、对动力学模型与环境扰动的简化、以及在车辆过近或模型不完整时需Slack扩展；

---

## 264. Automation and Reuse Practices in GitHub Actions Workflows: A Practitioner's Perspective

**arXiv ID:** 2601.11299 | [PDF](https://arxiv.org/pdf/2601.11299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 265. Membership Inference on LLMs in the Wild

**arXiv ID:** 2601.11314 | [PDF](https://arxiv.org/pdf/2601.11314v1)

**作者:** Jiatong Yi `[一作]` (Chinese University of Hong Kong), Yanyang Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 458 | [OpenAlex ID](https://openalex.org/A5101595086)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对严格黑盒大语言模型的会员推断攻击框架——SimMIA。

**💡 创新点**

创新点在于采用逐词采样与语义相似度打分相结合的软评分机制，显著提升了对黑盒模型的检测鲁棒性。

**🔧 技术方法**

主要技术包括 word‑by‑word 采样、基于词嵌入的余弦相似度打分、相对聚合以及可插拔的前缀策略。

**📊 数据集**

使用了 WikiMIA、MIMIR 以及新构建的 WikiMIA‑25 基准，并在多款公开模型和最新专有模型（Gemini‑2.5‑Flash、GPT‑5‑Chat 等）上进行实验。

**📈 对比分析**

与十余种灰盒与黑盒基线对比，SimMIA 在黑盒设置下平均提升 AUC 约 15–20 分，甚至在某些模型上超过最佳灰盒方法。

**⚠️ 局限性**

主要局限是需要多次生成采样，导致查询开销和成本高，缺乏对单次推断的高效性。

---

## 266. OpenACM: An Open-Source SRAM-Based Approximate CiM Compiler

**arXiv ID:** 2601.11292 | [PDF](https://arxiv.org/pdf/2601.11292v1)

**作者:** Yiqi Zhou `[一作]` (Nanjing University of Science and Technology), Zhiqiang Xiao `[通讯]` (The 58th Research Institute of China Electronics Technology Group Corporation)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个开源的 SRAM 基于近似 Compute-in-Memory 编译器 OpenACM，可根据应用误差容忍度自动生成准确度可调的乘法器并完成物理布局。

**💡 创新点**

首次将准确度可配置的近似乘法器（4‑2 压缩器与对数乘法器）嵌入 DCiM 编译器；完全开源且支持变异感知 SRAM 分析；使用 OpenROAD 与 FreePDK45 实现无专有工具的全流程。

**🔧 技术方法**

采用 4‑2 压缩器近似乘法器、对数乘法器、SRAM macro 生成、变异感知 Monte Carlo + Importance Sampling、OpenROAD/FreePDK45 物理设计、OpenYield、Python 脚本自动化。

**📊 数据集**

用卷积神经网络（ResNet‑18）、图像融合与边缘检测实验、以及标准图像（Lake、Mandril 等）做 PSNR 测试；数据集包括 ImageNet ILSVRC2012。

**📈 对比分析**

通过后布局比较面积、延迟、功耗；相较于 Exact/OpenC2，Log‑our 在 64×32 乘法器可节能 64%，面积 33%‑51% 缩减；在 ResNet‑18 维持 Top‑1 0.680（略高于 exact）且功耗大幅降低。

**⚠️ 局限性**

目前尚未自动生成 SRAM 布局，浮点支持缺失；仅支持单核 PE，未提供完整 DSE 引擎；对大规模芯片的时序与互连尚未验证。

---

## 267. FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster for Time-Series Forecasting

**arXiv ID:** 2601.11350 | [PDF](https://arxiv.org/pdf/2601.11350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 268. Joint Antenna Rotation and IRS Beamforming for Multi-User Uplink Communications

**arXiv ID:** 2601.11291 | [PDF](https://arxiv.org/pdf/2601.11291v1)

**作者:** Guoying Zhang `[一作]` (Shanghai Jiao Tong University), Penghui Huang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1749 | [OpenAlex ID](https://openalex.org/A5056609940)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出在多用户上行链路中同时优化可旋转天线的方位角、接收波束和IRS相位调制，实现更好的覆盖与容量；

**💡 创新点**

创新点在于将天线旋转角度纳入IRS辅助系统的联合优化框架，并针对近场/远场混合传播模型给出高效的投影梯度上升、MMSE波束和FP+RCG相位优化方案；

**🔧 技术方法**

使用投影梯度上升（PGA）处理角度非光滑性、最小均方误差（MMSE）求解接收波束、分式规划（FP）配合Riemannian共轭梯度（RCG）求解IRS相位；

**📊 数据集**

实验使用仿真参数（28 GHz、4×4 BS、8×8 IRS、4个单天线用户、不同用户数/IRS元件/发射功率）进行数值评估，无真实数据集；

**📈 对比分析**

与固定天线+IRS和仅天线旋转两种基线比较；在用户数、IRS元件、功率变化下，所提RA+IRS方案平均提升1.24–2.97 bps/Hz（≈24.6%），在IRS元件增大时优势进一步扩大；

**⚠️ 局限性**

局限包括：近场模型与近似仍可能导致性能偏差；算法复杂度随IRS规模增长；仅考虑上行链路和理想CSI；在高度干扰或大规模系统中收益可能降低。

---

## 269. The Mini Wheelbot Dataset: High-Fidelity Data for Robot Learning

**arXiv ID:** 2601.11394 | [PDF](https://arxiv.org/pdf/2601.11394v1)

**作者:** Henrik Hose `[一作]` (RWTH Aachen University), Sebastian Trimpe `[通讯]` (RWTH Aachen University)

**通讯引用:** 2409 | [OpenAlex ID](https://openalex.org/A5023990842)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 Mini Wheelbot 的大规模高频运动学数据集，并提供了示例脚本演示模型学习、状态估计和时间序列分类等应用。

**💡 创新点**

创新点在于：①首次为一个不稳定、非线性、欠驱动的单轮机器人发布完整的 1 kHz 传感器同步数据；②覆盖多硬件实例、不同地面与控制策略（PRBS、MPC、RL）以保证数据多样性；③配套 Python 包和开源示例代码，方便社区快速复现与基准对比。

**🔧 技术方法**

采用的技术包括：1 kHz 同步记录、三轴 IMU、光学运动捕捉、视频日志；示例中使用 MLP 预测动力学、基于运动捕捉的姿态估计器、时间序列 Transformer 分类器。

**📊 数据集**

使用的主要数据集为公开的 Mini Wheelbot 数据集（约 13 百万状态转移），包含传感器读数、估计姿态、地面真值、视频等信息。

**📈 对比分析**

方法对比基于：①对 MLP 自回归预测与真实轨迹的均方误差；②姿态估计器与运动捕捉真值的欧拉角误差；③Transformer 分类器在不同序列长度下的准确率。实验显示 MLP 能在 50 步自回归中保持较低误差，姿态估计器误差在几度以内，分类准确率随序列长度提升至约 90 %+。

**⚠️ 局限性**

局限性包括：数据仅来自单一机器人平台，缺乏 LiDAR/视觉信息；控制策略覆盖有限，未包含高级高层任务；数据主要聚焦低层动力学，缺乏长期任务评估；可能对不同硬件硬件版本的泛化能力有限。

---

## 270. Reward Modeling for Scientific Writing Evaluation

**arXiv ID:** 2601.11374 | [PDF](https://arxiv.org/pdf/2601.11374v1)

**作者:** Furkan Şahinuç `[一作]` (Technical University of Darmstadt), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 25133 | [OpenAlex ID](https://openalex.org/A5027450194)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练了专门用于科学写作评估的奖励模型SciRM和SciRM-Ref，采用两阶段强化学习优化评估偏好和推理能力。

**💡 创新点**

创新点在于引入基于评估宪章的上下文化奖励，利用两阶段GRPO训练提升对多方面评估规范的遵循与自我修正。

**🔧 技术方法**

技术包括大语言模型（Qwen2.5‑7B）与LoRA微调、GRPO强化学习、双阶段奖励函数与自我反思机制。

**📊 数据集**

数据集包括来自RevUtil、Related Work评估、Paper Review、Novelty Alignment、Revision Alignment等多任务的科学写作文本，涵盖多维度打分规则。

**📈 对比分析**

与多种开源LLM、基准评测模型对比，SciRM‑Ref在四项任务上平均得分最高，尤其在需要强推理的Novelty Alignment任务上超过基线。

**⚠️ 局限性**

限制在于只训练7B模型，评估数据多为二元打分，缺乏更细粒度评估数据；大型模型与更丰富数据集可进一步提升性能。

---

## 271. Walk based Laplacians for Modeling Diffusion on Complex Networks

**arXiv ID:** 2601.11338 | [PDF](https://arxiv.org/pdf/2601.11338v1)

**作者:** Francesca Arrigo `[一作]` (University of Strathclyde), Fabio Durastante `[通讯]` (University of Pisa)

**通讯引用:** 310 | [OpenAlex ID](https://openalex.org/A5018974435)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种统一的图神经网络框架，能通过学习任意图卷积核实现链路预测；

**💡 创新点**

通过自适应多跳卷积核，自动捕获不同层级图结构信息，并兼容多种现有 GNN 方法；

**🔧 技术方法**

使用图卷积、内积边嵌入、多跳消息传递，基于 PyTorch 实现；

**📊 数据集**

在合成网络以及 Cora、Citeseer、Pubmed 等真实图数据集上进行实验；

**📈 对比分析**

与传统矩阵分解、节点嵌入及 GCN/GraphSAGE/GAT 等方法对比，取得更高的 AUC/精度，表现为 state‑of‑the‑art；

**⚠️ 局限性**

计算复杂度较高，训练参数多，难以直接扩展到极大规模或动态图网络

---

## 272. Information Theoretic Perspective on Representation Learning

**arXiv ID:** 2601.11334 | [PDF](https://arxiv.org/pdf/2601.11334v1)

**作者:** Deborah Pereg `[一作]` `[通讯]` (Istituto Dalle Molle di Studi sull'Intelligenza Artificiale), Deborah Pereg (Istituto Dalle Molle di Studi sull'Intelligenza Artificiale)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文从信息论角度提出了表示学习的理论框架，定义了表示率、表示容量与压缩输出的率失真关系，并推导了对应的可达界与逆界；

**💡 创新点**

创新点在于将经典信息论的源/信道容量与率失真理论直接映射到深度学习中的最后层嵌入空间，阐明了在回归任务中表示率必须满足的熵与互信息约束；

**🔧 技术方法**

主要技术手段为信息论的相对熵、AEP、互信息分析、Lipschitz 连续性假设以及对齐的可逆/单射映射证明；

**📊 数据集**

论文并未给出具体实验数据集，理论中引用了典型的图像/序列场景（如MNIST、ResNet-50、UNet等）作为示例；

**📈 对比分析**

由于缺乏实验验证，本文未与其他方法进行性能比较，只提供了理论上“可达率”和“容量”边界；

**⚠️ 局限性**

主要局限在于：①假设映射为可逆或单射，实际网络可能出现冲突；②仅作理论推导，缺乏经验评估；③未考虑非 i.i.d. 或高维稀疏数据的实际影响。

---

## 273. Idea First, Code Later: Disentangling Problem Solving from Code Generation in Evaluating LLMs for Competitive Programming

**arXiv ID:** 2601.11332 | [PDF](https://arxiv.org/pdf/2601.11332v1)

**作者:** Sama Hadhoud `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Alham Fikri Aji `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 2868 | [OpenAlex ID](https://openalex.org/A5112924039)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了以自然语言编辑报告为中间产物的竞争性编程评测流程，分别评估模型的算法推理能力与代码实现质量；

**💡 创新点**

将问题求解与实现拆分开来，用编辑报告作为可转移的“计划”抽象；引入金手册编辑报告来量化推理瓶颈，利用LLM做评判者以大规模评测编辑报告质量；

**🔧 技术方法**

利用大语言模型生成编辑报告与代码，采用单射一轮提示；使用LLM-as-a-judge对编辑报告进行结构化标签；对代码使用ICPC风格的编译-运行判题器；

**📊 数据集**

收集了83道ICPC风格问题（来自七场比赛，2017-2025），每道问题配有题设、金手册编辑报告和完整官方测试集；

**📈 对比分析**

对19个LLM（闭源与开源、推理型与通用型）在三种条件下（无编辑、生成编辑、金手册编辑）进行pass@1与虚拟排名百分位的评估；结果显示金手册编辑带来显著提升（≈30%），生成编辑提升有限（≈10-15%），实现仍是主要瓶颈；跨模型编辑转移实验表明强推理模型的编辑可提升弱实现模型的性能，部分情况下可超越自身完整流程；

**⚠️ 局限性**

数据量有限（83题），专家评审覆盖单一比赛与单一评审；仅在C++环境下评测，未考虑多语言、多样化题型；采用单次无采样的固定提示，未探究更复杂的交互式或工具使用场景；

---

## 274. XChoice: Explainable Evaluation of AI-Human Alignment in LLM-based Constrained Choice Decision Making

**arXiv ID:** 2601.11286 | [PDF](https://arxiv.org/pdf/2601.11286v1)

**作者:** Weihong Qi `[一作]` (Indiana University Bloomington), Haewoon Kwak `[通讯]` (Indiana University Bloomington)

**通讯引用:** 14141 | [OpenAlex ID](https://openalex.org/A5061923992)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出XChoice框架，通过恢复并对比人类与LLM在约束决策中的机制参数来评估人工智能与人类的对齐情况。

**💡 创新点**

创新点在于把对齐问题转化为机制对比，利用结构化的决策模型和可解释参数实现细粒度诊断和稳健性检验。

**🔧 技术方法**

主要技术包括约束优化建模、逆优化参数估计、余弦相似度和平均绝对偏差对齐度量、鲁棒性不变性分析以及检索增强生成（RAG）干预。

**📊 数据集**

使用美国时间使用调查（ATUS）数据作为人类基准，对应的日常时间分配决策进行建模与对比。

**📈 对比分析**

与传统的准确率或F1等结果匹配指标相比，XChoice在识别模型与人类在不同属性和子群体中的权重偏差方面表现更好，且在RAG干预后能显著提升对齐度。

**⚠️ 局限性**

局限性包括仅适用于可表述为约束优化的问题、对模型形式和归一化敏感、可能受误设和未观测变量影响，且子群体分析若缺乏上下文可能导致刻板印象或误导。

---

## 275. QUPID: A Partitioned Quantum Neural Network for Anomaly Detection in Smart Grid

**arXiv ID:** 2601.11500 | [PDF](https://arxiv.org/pdf/2601.11500v1)

**作者:** Hoang M. Ngo `[一作]` (University of Florida), My T. Thai `[通讯]` (University of Florida)

**通讯引用:** 8765 | [OpenAlex ID](https://openalex.org/A5005663679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种分区量子神经网络QUPID，能够在智能电网中实现高效且鲁棒的异常检测，并通过R-QUPID进一步提供可证明的对抗性防护；

**💡 创新点**

创新点包括：①使用分区方案显著降低量子计算规模；②引入复杂值编码提升特征表达；③首次正式证明量子噪声可放大差分隐私并增强对抗鲁棒性；

**🔧 技术方法**

采用量子变分电路、幅度编码、旋转与纠缠层、去极化噪声模拟以及经典梯度下降和ReLU激活等技术；

**📊 数据集**

使用Oak Ridge National Laboratory的ICS小规模智能电网数据集，包含15个场景、119个特征、6类标签；

**📈 对比分析**

与四种传统深度学习基线（MLP、InceptionNet、MTL-LSTM、HQ-DNN）及FTTransformer对比，在15个场景、7项指标（准确率、精确率、召回率、F1、ROC‑AUC、MCC、G‑Mean）上QUPID均表现优于或相近其余模型，且在FGSM和PGD攻击下R‑QUPID保持更高鲁棒性；

**⚠️ 局限性**

限制主要在于：实验仅基于小规模模拟数据，量子模拟器和噪声模型未在真实量子硬件上验证；

---

## 276. BoxMind: Closed-loop AI strategy optimization for elite boxing validated in the 2024 Olympics

**arXiv ID:** 2601.11492 | [PDF](https://arxiv.org/pdf/2601.11492v1)

**作者:** Kaiwen Wang `[一作]` (Tsinghua University), Ji Wu `[通讯]` (Tsinghua University)

**通讯引用:** 5267 | [OpenAlex ID](https://openalex.org/A5029547618)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了BoxMind闭环AI专家系统，对拳击比赛视频进行原子拳击事件识别，生成18层次技术战术指标，并基于图模型预测比赛结果与推荐策略，最终在2024巴黎奥运会上帮助中国拳击队夺得五枚奖牌。

**💡 创新点**

创新点在于：① 将拳击动作细化为时空属性原子事件并聚合成可解释的18指标；② 在BoxerGraph中结合可学习的时间变latent嵌入与显式指标进行融合预测；③ 利用预测模型可微分梯度直接生成对手专属可执行战术建议，实现从观察到因果优化的闭环。

**🔧 技术方法**

主要技术包括4D‑Humans姿态估计、UV‑map增强跟踪、TCN与Pose‑Region Guided原子事件检测、图神经网络+多任务MLP预测、梯度优化策略推荐。

**📊 数据集**

使用了自建的BoxingWeb/BoxingStudio人工标注数据（80场回合/10.9K原子事件），扩展至BoxingWeb‑Full（651场比赛/119小时视频）以及与中国国家队训练视频（2,240回合）构成的多源大规模数据集。

**📈 对比分析**

与传统的Glicko、Elo、WHR评级体系进行对比，BoxMind在BoxerGraph‑80KG测试集上准确率为69.8%（提升约9.5%），奥运赛事实验证中预测策略的F1值为0.601，略优于人类专家平均0.467，且标准差更小；在闭环案例中指标提升明显且与冠军表现对应。

**⚠️ 局限性**

局限性包括：① 目前仅为赛前/赛后离线系统，缺乏实时推理能力；② 依赖人工标注的原子事件数据，标注成本高；③ 仅验证于拳击，可迁移性需要进一步验证；④ 梯度策略虽可执行但未涵盖战术变化的时间序列决策。

---

## 277. Extractive summarization on a CMOS Ising machine

**arXiv ID:** 2601.11491 | [PDF](https://arxiv.org/pdf/2601.11491v1)

**作者:** Ziqing Zeng `[一作]` (University of Minnesota), Sachin S. Sapatnekar `[通讯]` (University of Minnesota)

**通讯引用:** 15302 | [OpenAlex ID](https://openalex.org/A5068714995)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了完整的硬件感知工作流，在CMOS耦合振荡器Ising机上实现提取式摘要。

**💡 创新点**

通过引入平衡局域场与耦合项的改进Ising模型、随机舍入与分解策略，解决低精度整数权重与大规模问题。

**🔧 技术方法**

使用ILP→QUBO→Ising映射、偏置平衡、随机舍入、分解、迭代精炼，并在COBI CMOS Ising机上执行。

**📊 数据集**

采用CNN/DailyMail（20/50句段落）和XSum（100句段落）数据集进行实验。

**📈 对比分析**

与软件Tabu搜索和穷举基线在TTS和ETS指标下比较，COBI实现3–4.5× TTS加速、能耗降低2–3个数量级，摘要质量保持归一化目标≥0.9。

**⚠️ 局限性**

对高精度需求的解仍存在误差，COBI硬件的随机性与量化误差导致与Tabu相比精度略低，且仅支持有限位整数权重。

---

## 278. Low-Rank Key Value Attention

**arXiv ID:** 2601.11471 | [PDF](https://arxiv.org/pdf/2601.11471v1)

**作者:** James O'Neill `[一作]` (Intercom), Fergal Reid `[通讯]` (Intercom)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种低秩KV适配(LRKV)机制，用于在Transformer预训练中显著减少KV缓存占用，同时保持多头注意力的表达力。

**💡 创新点**

创新点在于将每个头的KV投影拆解为共享全秩基底加上低秩、头特异性残差，从而在不牺牲头级多样性的前提下实现显著内存压缩，并提供可调的效率-表达力连续曲线。

**🔧 技术方法**

采用的技术包括多头注意力、低秩因子分解、RMSNorm、可测量的注意力双线性形式、核PCA和有效秩分析等；同时将LRKV与现有的MQA、GQA、MLA等方法在同一框架下实现对齐。

**📊 数据集**

实验使用FineWeb‑Edu大规模文本进行无监督预训练，并在SmolTalk、MMLU、GSM8K等混合指令集上进行中训练，随后在ARC‑Easy、ARC‑Challenge、MMLU、GSM8K、HumanEval等标准基准上评估。

**📈 对比分析**

在2.5B参数规模下，LRKV在预训练阶段以52.6% KV缓存相对标准多头注意力实现0.719 BPB的最佳语言建模表现；在中训练后在5个下游任务中获得最高综合得分37.9%，相较于标准MHA、GQA、MQA、MLA均有显著提升，且训练计算更高效。

**⚠️ 局限性**

局限性包括仅评估了解码器单向Transformer并限于最多2.5B规模；对不同硬件、极长上下文或编码-解码架构的适用性尚未验证；以及对多层/多头不同低秩分配策略的探索仍不足。

---

## 279. MHA2MLA-VLM: Enabling DeepSeek's Economical Multi-Head Latent Attention across Vision-Language Models

**arXiv ID:** 2601.11464 | [PDF](https://arxiv.org/pdf/2601.11464v1)

**作者:** Xiaoran Fan `[一作]` (Fudan University), Tao Gui `[通讯]` (Fudan University)

**通讯引用:** 4787 | [OpenAlex ID](https://openalex.org/A5058353652)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将现有的基于 MHA/GQA 的视觉语言模型迁移到 DeepSeek 的 Multi‑Head Latent Attention（MLA）架构，实现 KV 缓存大幅压缩并保持原模型性能。

**💡 创新点**

提出模态自适应部分 RoPE（MKL）与模态分离低秩 SVD（MD‑SVD）相结合的压缩方案，并引入参数高效微调（PEFT）和量化技术，显著降低 KV 内存占用且兼容现有量化方法。

**🔧 技术方法**

使用部分 RoPE、低秩键值联合压缩、MD‑SVD、PEFT、KV 量化与剪枝等技术实现模型迁移与压缩。

**📊 数据集**

在 LLaVA‑1.5、LLaVA‑NeXT、Qwen2.5‑VL 三大视觉语言模型上，以公开的视觉语言任务数据集（如 VQA、Caption 等）进行微调与评估。

**📈 对比分析**

与原始模型、GQA、Cache 剪枝和量化方法对比，KV 缓存可压缩 94% 以上，同时保持与原始模型相近甚至更优的性能；在 62.5% 压缩率下平均得分 68.75，明显优于 H₂O、TOVA 等剪枝方案。

**⚠️ 局限性**

仍需一定量的微调数据和时间，且对不同模态频率选择的精细调优有待进一步改进，深层层次跨模态干扰在极端压缩下可能导致轻微性能下降。

---

## 280. Interactive Narrative Analytics: Bridging Computational Narrative Extraction and Human Sensemaking

**arXiv ID:** 2601.11459 | [PDF](https://arxiv.org/pdf/2601.11459v1)

**作者:** Brian Keith `[一作]` `[通讯]` (Catholic University of Northern), Brian Keith (Catholic University of Northern)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并定义了交互式叙事分析（INA）作为一种将计算叙事提取与交互式可视化相结合的跨学科研究领域，阐述其核心组成与挑战。

**💡 创新点**

首次将叙事提取、可视化、语义交互、知识集成以及评估框架统一为一个系统性研究范式，并强调人机协同在叙事理解中的核心作用。

**🔧 技术方法**

采用自然语言处理、机器学习、图算法、可视化界面技术、语义交互模型以及外部知识图谱等多种技术手段。

**📊 数据集**

文中未给出具体实验数据集，主要以新闻、社交媒体、科学文献等大型文本集合为典型场景做示例说明。

**📈 对比分析**

未进行实验比较，作者仅讨论了现有方法的局限与未来评估框架，未给出性能指标。

**⚠️ 局限性**

主要限制包括缺乏标准化评估指标、可扩展性与实时交互的技术瓶颈、叙事解释的主观性与跨领域迁移困难，以及对误导性叙事的检测挑战。

---

## 281. Indoor Neutral-Host Networks Over Shared Spectrum and Shared Infrastructure: A Comparison Study of Real-World Deployments

**arXiv ID:** 2601.11457 | [PDF](https://arxiv.org/pdf/2601.11457v1)

**作者:** Joshua Roy Palathinkal `[一作]` (University of Notre Dame), Monisha Ghosh `[通讯]` (University of Notre Dame)

**通讯引用:** 2539 | [OpenAlex ID](https://openalex.org/A5101557097)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在三处不同建筑环境中实地测量，比较了CBRS频段下的中立宿主网络与传统MNO宏站以及Wi‑Fi在室内外覆盖、PHY层吞吐量、上行功率和端到端性能的差异。

**💡 创新点**

创新点在于首次系统评估中立宿主在共享频谱上显著降低室内穿墙损耗、提升上行调制阶层和功率效率，并将其与Wi‑Fi在相同回程条件下进行端到端对比，揭示中立宿主的优劣势。

**🔧 技术方法**

使用QualiPoc与SigCap测量工具获取LTE/NR PHY指标（RSRP/SS‑RSRP、吞吐量、调制、TX功率），Wi‑Fi RSSI和TX功率；通过HTTP GET/PUT 与 ICMP ping 测试收集端到端吞吐量和RTT；搭建CBRS小基站、MNO宏站和Wi‑Fi AP 进行对比实验。

**📊 数据集**

数据集包含254,728个LTE/NR测量点和126,790个Wi‑Fi beacon 点，涵盖RSRP/SS‑RSRP、吞吐量、调制阶层、TX功率、HTTP上传成功率等多维度指标，覆盖三种建筑环境（零售店、办公楼、仓库）。

**📈 对比分析**

通过分位点统计、归一化吞吐量、调制使用比例、UE TX功率、HTTP上传成功率、RTT 等指标进行对比；结果显示中立宿主室内RSRP比MNO高约30 dB，UL吞吐量和64‑QAM使用率显著优于MNO，端到端DL吞吐量与MNO相当甚至略优，但UL吞吐量和延迟低于Wi‑Fi。

**⚠️ 局限性**

局限性包括：部署场景有限（仅三站点）、Wi‑Fi与NH使用不同回程导致延迟偏差、未系统评估跨频段切换与协议栈优化，缺乏大规模、多租户环境下的普适性验证。

---

## 282. Topology-Guaranteed Image Segmentation: Enforcing Connectivity, Genus, and Width Constraints

**arXiv ID:** 2601.11409 | [PDF](https://arxiv.org/pdf/2601.11409v1)

**作者:** Wenxiao Li `[一作]` (Beijing Normal University), Jun Liu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种宽度感知拓扑能量，利用平滑形态学梯度对持久同调关键点进行处理，使分割结果既保持拓扑一致性，又保留结构宽度；

**💡 创新点**

创新点在于把宽度信息显式嵌入持久同调能量中，构造平滑的形态学梯度约束关键点，解决传统PH忽略几何宽度、导致单像素连通或错误孔洞的问题；

**🔧 技术方法**

技术包括：持久同调（PH）、平滑形态学梯度、非局部软阈值动态（NLSTD）变分模型、AdamW优化、深度卷积网络（UNet、DeepLabV3+、SegFormer、UNet++）的损失正则化；

**📊 数据集**

使用的数据集有：合成图像、MNIST、ISICDM（皮肤病变/膀胱壁）、Massachusetts Roads、ISBI神经元电镜分割；

**📈 对比分析**

与无拓扑约束、仅PH约束进行对比，评估指标包括准确率、Dice、IoU、BDIoU、HD95、clDice、Betti数。结果显示，WT能量在保持拓扑正确性的同时，宽度更准确，整体指标均优于PH及无约束方法；

**⚠️ 局限性**

局限性包括：PH计算耗时长，无法直接在推理阶段高效使用；需要手动指定拓扑通道和宽度信息；仅处理单通道拓扑，未充分利用多尺度结构；对复杂管状结构仍存在挑战。

---

## 283. Efficient Channel Autoencoders for Wideband Communications leveraging Walsh-Hadamard interleaving

**arXiv ID:** 2601.11407 | [PDF](https://arxiv.org/pdf/2601.11407v1)

**作者:** Cel Thys `[一作]` (Katholieke Universiteit Leuven), Sofie Pollin `[通讯]` (Katholieke Universiteit Leuven)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

开发了基于Walsh–Hadamard（WH）互插转换的端到端自动编码器（AE）架构，用于实现高效宽带通信并对系统能耗和误码率进行评估。

**💡 创新点**

创新点在于将WH域互插转换与深度学习AE结合，形成一种硬件友好的编码调制方案，能够在不重新设计算法的前提下，实现与传统Polar码相近的误码性能，同时在系统能效上提升约29%。

**🔧 技术方法**

使用的主要技术包括Walsh–Hadamard变换（FWHT/IFWHT层）、时间互插转换、端到端神经网络AE（全连接网络）、BCE损失、Adam优化、硬件能耗建模（包括基带运算功耗与转换器功耗）等。

**📊 数据集**

数据集：论文使用随机生成的二进制比特流作为输入进行仿真，未采用公开数据集，仅依赖 Monte Carlo 生成的随机比特序列。

**📈 对比分析**

通过与时间互插AE、CNN‑AE、Polar 码（L=2/4/8）、LDPC 等基线进行比较，评估阈值SNR、系统功耗和能效（bit/J）。WH‑AE在阈值SNR上仅比Polar低0.14 dB，能效比TI‑AE提升29%，与CNN‑AE提升可达4.8倍。

**⚠️ 局限性**

局限性包括：结果基于仿真与理论功耗模型，缺乏实际硬件实现验证；仅在短块长度（n=32）且AWGN信道下测试，未验证在更大块长或更复杂信道环境下的性能；能耗估计假设线性缩放与特定硬件平台，可能不完全适用于不同实现。

---

## 284. SME-YOLO: A Real-Time Detector for Tiny Defect Detection on PCB Surfaces

**arXiv ID:** 2601.11402 | [PDF](https://arxiv.org/pdf/2601.11402v1)

**作者:** Meng Han `[一作]` (Henan University), Meng Han `[通讯]` (Henan University)

**通讯引用:** 1245 | [OpenAlex ID](https://openalex.org/A5053334015)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向极小 PCB 表面缺陷的轻量级 YOLOv11n 变体 SME‑YOLO。

**💡 创新点**

创新点包括：使用 Normalized Wasserstein Distance Loss 缓解 IoU 对微小目标定位误差敏感；引入 Efficient Upsampling Convolution Block 提升上采样细节恢复；设计 Multi‑Scale Focused Attention 模块针对 PCB 缺陷主尺度进行多尺度聚焦。

**🔧 技术方法**

技术包括：YOLOv11n 框架、NWDLoss、EUCB、MSFA、二维高斯分布 Wasserstein 距离、深度可分离卷积、通道混合与注意力机制。

**📊 数据集**

使用北京大学智能机器人开放实验室发布的 PKU‑PCB 数据集（1386 张高分辨率图像，6 种缺陷类型）。

**📈 对比分析**

在同一实验环境下与 YOLOv5、v8、v10、v11 等主流模型对比，SME‑YOLO 在 mAP@0.5 上提升至 0.950（比 YOLOv11 提升 2.2%），Precision 提升 4%，Recall 提升 1.8%。

**⚠️ 局限性**

局限性：仅在 PKU‑PCB 数据集验证，缺陷类别和样本规模有限；在复杂背景、强反射、动态光照或油尘污染等极端条件下未系统评估。

---

## 285. Factored Value Functions for Graph-Based Multi-Agent Reinforcement Learning

**arXiv ID:** 2601.11401 | [PDF](https://arxiv.org/pdf/2601.11401v1)

**作者:** Ahmed Rashwan `[一作]` (University of Bath), Lisa Kreusser `[通讯]` (Monumo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Diffusion Value Function（DVF）作为图结构MARL的分解价值函数，并基于其实现Diffusion A2C（DA2C）与学习式DropEdge GNN（LD‑GNN）来学习分布式算法。

**💡 创新点**

创新点在于将奖励通过时间折扣与图距离衰减的扩散算子统一建模，保证DVF在无穷视角下收敛、可分解为全局价值，并可用GNN高效估计；同时在DA2C中把DVF作为优势函数，LD‑GNN学习稀疏消息传递。

**🔧 技术方法**

使用图神经网络（GNN）实现价值函数与演员，结合优势演员‑批评家框架、TD学习和扩散算子；对比全局、局部与邻域批评家。

**📊 数据集**

在三类基准任务上评测：火灾扑救（Firefighting）、向量图着色（Vector Graph Colouring）与无线功率控制（Transmit Power Control），每个任务对应多种图结构与规模（数十到数千节点，甚至10,000节点的OOD）。

**📈 对比分析**

相较于REINFORCE、IA2C、NA2C、MAA2C等批评家，DA2C在所有任务上取得最优或相近最佳奖励，平均提升高达11%，并在不同消息惩罚、图拓扑与规模下保持稳定，实验显著优于基准。

**⚠️ 局限性**

局限在于假设影响图固定且已知，扩散衰减可能削弱长距离协作；目前仅在集中训练/分散执行框架下验证，缺乏完全分散训练与时间变异图的处理。

---

## 286. On the Probability of First Success in Differential Evolution: Hazard Identities and Tail Bounds

**arXiv ID:** 2601.11499 | [PDF](https://arxiv.org/pdf/2601.11499v1)

**作者:** Dimitar Nedanovski `[一作]` (Sofia University St. Kliment Ohridski), Dimitar Pilev `[通讯]` (University of Chemical Technology and Metallurgy)

**通讯引用:** 94 | [OpenAlex ID](https://openalex.org/A5029810158)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过条件危害（hazard）框架分析差分进化（尤其是自适应变体 L‑SHADE）的首次命中时间，并给出分布无关的存活概率与尾部上界；提出可检验的“见证事件”来为每一步成功概率提供确定性下界；利用 Kaplan‑Meier 生存分析对 CEC2017 基准集进行实证验证，识别出三种不同的成功模式。

**💡 创新点**

①将存活概率写成条件第一次命中概率的乘积，从而得到无假设的尾部上界；②为 L‑SHADE 引入基于采样规则、种群结构的可观测见证事件，使得每一步的危害下界仅由理论常数和经验频率组成；③将理论常数与经验频率分离，解释常数危害下界为何往往保守，并在实验中验证其有效性。

**🔧 技术方法**

条件危害分析、存活概率乘积公式、分布无关尾部上界；可观测见证事件构造；基于采样规则的概率下界计算；Kaplan‑Meier 生存分析；对比常数危害模型；实验验证（统计量、聚类系数等）。

**📊 数据集**

CEC2017 基准函数（10 维），在 51 次独立运行、预算 10000d 评估次数的设置下进行评估。

**📈 对比分析**

与传统的常数危害上界进行对比；通过 Kaplan‑Meier 曲线识别聚类（burst）与几何尾部两种模式；实验结果显示，实际后期危害率往往比理论上界小 1–2 个数量级，常数危害模型在聚类成功时显得过于保守；在易成功函数上，几何尾部模型较好；在难解函数上，存活概率高于预算限制。

**⚠️ 局限性**

①见证事件出现的频率在不同函数/阶段差异大，导致理论下界有时极其保守；②分析主要针对 Morse 函数，非 Morse 或高度多模态问题的推广有限；③需要已知参数范围（F、CR）和存档大小等假设，实际实现中可能不完全满足；④整体上仍未给出精确的算法参数选择指南，仅提供诊断框架。

---

## 287. PRISM-CAFO: Prior-conditioned Remote-sensing Infrastructure Segmentation and Mapping for CAFOs

**arXiv ID:** 2601.11451 | [PDF](https://arxiv.org/pdf/2601.11451v1)

**作者:** Oishee Bintey Hoque `[一作]` (University of Virginia), Abhijin Adiga `[通讯]` (University of Virginia)

**通讯引用:** 1155 | [OpenAlex ID](https://openalex.org/A5019703184)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种基于先验基础的遥感图像分析流程，先用域自适应YOLOv8检测CAFO关键设施，再利用SAM2生成高精度实例掩码，并通过几何过滤得到结构化设施候选，随后融合这些掩码与深度视觉特征与工程化领域先验（面积比例、相邻关系、县级畜种比例）构建多模态特征，送入掩码引导空间注意力与掩码注意力池化的分类器实现CAFO类型预测并提供可解释的掩码级归因；

**💡 创新点**

创新点在于：①将检测-分割-分类三步串联为先验驱动的端到端管道；②利用领域调优YOLO与SAM2生成可靠设施掩码并通过几何过滤实现结构化设施识别；③设计掩码引导空间注意力与掩码注意力池化，既提升准确率又提供可解释的对象与关系级归因；

**🔧 技术方法**

技术包括域自适应YOLOv8、SAM2实例分割、几何函数过滤、面积比例与相邻度等工程化先验、掩码引导空间注意力模块、掩码注意力池化、Transformer/CNN视觉编码器（如Swin‑B、EfficientNet‑B3等）与线性分类器；

**📊 数据集**

使用自建的38,000+ 高分辨率NAIP图像补丁数据集（涵盖20州、4种畜种和负样本）以及通过人工校正和YOLO生成的约130k高质量设施掩码；

**📈 对比分析**

与多种基线（CLIP、DINOv2、RemoteCLIP、ViT、ResNet等）相比，加入先验后模型在随机拆分上提升F1最高15%，在空间拆分（跨州）上提升高达15%，尤其在奶牛和牛肉类的准确率上显著提升；

**⚠️ 局限性**

局限在于：①Transformer视觉模型在本任务仍不如CNN；②跨区域泛化需要足够的正样本，表明需要半监督或主动学习；③方法依赖于人工标注的设施框与掩码，标注成本仍较高；

---

## 288. IMS: Intelligent Hardware Monitoring System for Secure SoCs

**arXiv ID:** 2601.11447 | [PDF](https://arxiv.org/pdf/2601.11447v1)

**作者:** Wadid Foudhaili `[一作]` (University of Luebeck), Saleh Mulhem `[通讯]` (University of Luebeck)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种基于机器学习的智能硬件监控系统（IMS），实时检测SoC AXI总线的协议违规和DoS攻击。

**💡 创新点**

首次将深度学习模型嵌入AXI监控，采用低精度量化与稀疏剪枝实现资源友好；同时公开了完整的AXI攻击与正常数据集，填补了公共数据缺口。

**🔧 技术方法**

使用多层感知器（MLP）并通过QKeras量化、剪枝优化后，用HLS4ML转换为HDL，在FPGA上实现；同时采用SMOTE等数据增强和特征工程（PCA、相关性分析）提升模型效果。

**📊 数据集**

基于RISC‑V SoC在Xilinx ZCU104平台收集的16,383条正常交易与3,242条合成攻击交易，构成公开的AXI安全数据集。

**📈 对比分析**

与传统静态分析工具（XRAY、eXpect）及硬件监控单元（TMU、DD‑MPU）对比，IMS在98.7%准确率、>99% AUC、0.23% DSP、0.70% FF、1.566 ms延迟、2.5 M推理/秒的指标下表现优异，且实时检测率近100%。

**⚠️ 局限性**

目前仅覆盖AXI4的DoS相关攻击，未针对零日或其他协议通道的攻击；需要预先训练的模型和攻击样本，对未知攻击可能存在检测盲区。

---

## 289. When Are Two Scores Better Than One? Investigating Ensembles of Diffusion Models

**arXiv ID:** 2601.11444 | [PDF](https://arxiv.org/pdf/2601.11444v1)

**作者:** Raphaël Razafindralambo `[一作]` (Université Côte d’Azur), Pierre-Alexandre Mattei `[通讯]` (Université Côte d’Azur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并系统评估在无监督扩散模型中使用各种集成方法（Deep Ensemble、Monte Carlo Dropout、Mixture of Experts、随机森林等）对生成质量与训练目标的影响，提供理论分析并对比多种聚合规则。

**💡 创新点**

①首次系统对多种集成聚合策略在扩散模型中的效果进行评估；②揭示在扩散模型中平均分数不等价于几何平均分布（PoE），解释训练目标与感知质量的脱节；③通过理论证明（Jensen不等式、PoE非可交换性）解释实验现象。

**🔧 技术方法**

使用score‑based diffusion模型、DDSM损失、Deep Ensemble、Monte Carlo Dropout、算术/几何/中位/占优势/交替采样/专家混合等聚合规则；随机森林用于表格数据生成；理论分析聚合对目标损失与分布的影响。

**📊 数据集**

CIFAR‑10 (32×32)、FFHQ (256×256) 图像数据集；Iris 表格数据集用于随机森林实验。

**📈 对比分析**

通过 FID、KID、DDSM 损失等指标对比单模型与各类集成方法的性能。实验发现：Deep Ensemble 能显著降低 DDSM 损失，但对 FID/KID 提升有限；Mixture of Experts 在 FFHQ 上可略优于最佳单模型；随机森林中的 dominant 聚合在 Iris 上显著降低 Wasserstein 距离。整体来看，集成效果多为边际提升。

**⚠️ 局限性**

①集成在扩散模型中收益有限，成本高；②平均分数并未提升感知质量，原因在于训练目标与 FID/KID 的不一致；③MC Dropout 对性能无益甚至下降；③理论上平均分数不等价于 PoE，说明直接平均不等同于采样混合分布；④未探索更高级的加权/提升策略，限制了潜在收益。

---

## 290. GenDA: Generative Data Assimilation on Complex Urban Areas via Classifier-Free Diffusion Guidance

**arXiv ID:** 2601.11440 | [PDF](https://arxiv.org/pdf/2601.11440v1)

**作者:** Francisco Giral `[一作]` (Universidad Politécnica de Madrid), Soledad Le Clainche `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 2424 | [OpenAlex ID](https://openalex.org/A5084006208)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

基于稀疏传感器观测，使用生成式扩散模型在无结构网格上实现城市风场的高分辨率重建

**💡 创新点**

将classifier‑free guidance解释为后验重构机制，结合多尺度图网络实现对几何依赖的物理先验学习，且无需在新几何或网格分辨率下重新训练

**🔧 技术方法**

多尺度图扩散模型、CFG（classifier‑free guidance）、CFD（RANS）仿真数据训练、无结构网格消息传递

**📊 数据集**

英国布里斯托尔城市邻域的RANS仿真数据，六个高度切片，约30万节点、170万条边的无结构网格

**📈 对比分析**

与MeshGraphNet、其多尺度变体以及低成本SVD（LCSVD）基线对比，RRMSE下降25‑57%，SSIM提升23‑33%，在稀疏传感器（约0.1% 覆盖）下仍保持较高结构和方向一致性

**⚠️ 局限性**

仅限二维平面，未考虑时间动态、非平稳性和更大规模网格；对移动或多层三维流场的扩展仍待验证

---

## 291. A Practical Guide to Establishing Technical Debt Management

**arXiv ID:** 2601.11430 | [PDF](https://arxiv.org/pdf/2601.11430v1)

**作者:** Marion Wiese `[一作]` `[通讯]`, Marion Wiese

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文通过对三家企业团队实践的总结，提出一套可落地的技术债务管理流程与最佳实践，帮助团队识别、预防、记录、优先级排序、偿还技术债务，并可视化跟踪进度。

**💡 创新点**

创新点在于将学术研究成果转化为实用指南，明确区分所有团队共用的“最佳实践”与可选的“nice-to-haves”，并给出针对不同团队情况的可配置模板与指标。

**🔧 技术方法**

主要使用的技术包括问题跟踪工具（Jira、Azure DevOps、GitLab）进行债务记录与管理，静态代码分析工具（SonarCloud、NDepend）辅助识别债务，以及 Power BI/Tableau 等商业智能工具进行可视化监控。

**📊 数据集**

本文并未采用公开数据集，而是基于三支来自不同行业企业的实际团队（共计约 3 组团队）的真实工单与债务数据进行案例分析与流程验证。

**📈 对比分析**

由于缺乏客观实验对比，本文没有传统意义上的性能评估；通过专家评审和团队反馈证明，所提出流程在实际落地后能够提升债务可见性、降低偿还成本、促进团队协作，但缺少量化效益指标。

**⚠️ 局限性**

局限性包括：仅聚焦单团队层面，未覆盖跨团队或全公司规模的推广与治理；对工具定制与可视化的技术实现依赖具体平台，迁移成本可能较高；缺乏长期数据验证与对比实验来量化效益。

---

## 292. The Great March 100: 100 Detail-oriented Tasks for Evaluating Embodied AI Agents

**arXiv ID:** 2601.11421 | [PDF](https://arxiv.org/pdf/2601.11421v1)

**作者:** Ziyu Wang `[一作]` (Shanghai Jiao Tong University), Yong-Lu Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2108 | [OpenAlex ID](https://openalex.org/A5031174631)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Great March 100（GM‑100）机器人学习奥林匹克基准，包含100个精心设计的任务及其数据集，并在两台不同机器人平台上进行数据采集与评估。

**💡 创新点**

创新点在于：① 基于人-物交互原语与物体适配性系统化扩展任务空间，覆盖稀有与长尾行为；② 采用LLM（Qwen3）与人工双重过滤生成高质量可执行任务；③ 引入多指标评估（SR、PSR、动作预测误差）以细粒度对比模型表现。

**🔧 技术方法**

技术方法包括：LLM任务生成与可执行性打分、手工筛选、远程遥控数据采集、强化学习与扩散策略基准模型（DP、π₀、π₀.₅、GR00T）训练、动作预测误差计算与可视化。

**📊 数据集**

使用的数据集为13,000+条遥控轨迹，覆盖100个任务，在Agilex Cobot Magic和Dobot Xtrainer两台机器人上收集，公开发布于https://rhos.ai/research/gm‑100。

**📈 对比分析**

评估方法通过Success Rate、Partial Success Rate及动作预测误差对基线模型进行比较；结果显示π₀.₅在大多数任务中取得最高PSR与最低误差，DP表现最差，整体SR仍低，体现任务挑战性。

**⚠️ 局限性**

局限性包括：训练数据量有限导致SR普遍偏低、硬件与环境差异影响公平性、评价体系仍以实验室条件为主、任务生成仍需人工干预、缺乏长期持续评测与公开验证机制。

---

## 293. Qihe: A General-Purpose Static Analysis Framework for Verilog

**arXiv ID:** 2601.11408 | [PDF](https://arxiv.org/pdf/2601.11408v1)

**作者:** Qinlin Chen `[一作]` (Nanjing University), Yue Li `[通讯]` (Nanjing University)

**通讯引用:** 14316 | [OpenAlex ID](https://openalex.org/A5100387753)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

开发了 Qihe——第一个针对 Verilog 的通用静态分析框架，并在其基础上实现了 20 个应用级分析（bug 检测、漏洞识别、程序理解等）。

**💡 创新点**

创新点在于：①首次构建完整的硬件静态分析基础设施（前端、IR、22 个基本分析）；②通过“雪崩式”依赖构建来实现复杂分析；③设计了可扩展的 IR 与属性系统以满足不同分析需求；④实现了基于 Java 注解的分析管理器，自动发现、注册并按依赖顺序执行分析。

**🔧 技术方法**

使用的技术包括：Java 实现、前端将 Verilog 转换为自定义的 Qihe IR（包含层次结构与三地址码），多种基本分析（寄存器、时钟、重置、控制流、数据流、并发、位向量算术等），以及基于这些分析构建的高级分析；框架内部采用图分析、循环检测、常量传播、符号执行等方法。

**📊 数据集**

数据集：20 个真实开源硬件项目（SoC、加密模块、AXI 协议实现等），平均 GitHub stars 1.5k，最大项目 1.8M 行 Verilog；同时收集了 9 个未公开的缺陷（已由开发者验证）和 18 个通过基准库/工业专家注入的缺陷。

**📈 对比分析**

与传统 Verilog linter（Slang、Verible 等）对比，Qihe 在 20 个客户端分析中发现了 9 个原始项目的未知缺陷、18 个 linter 无法检测的缺陷和 16 个安全漏洞；在 RISC‑V SoC 上，完成全部 20+22 个分析只需约 5 分钟、40 GB 内存；相比手工写分析，单个客户端平均仅需 300 行 Java 代码，显著降低开发成本。

**⚠️ 局限性**

局限性：①对非可综合代码支持有限；②分析结果对不同综合工具的实现差异不完全可保真，导致某些分析可能不完全 Sound；③仍需人工维护基本分析的精度与互相依赖的调优；④缺少对加密/IP 版块的完整可视化支持，需要进一步完善模块签名推断。

---

## 294. ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models

**arXiv ID:** 2601.11404 | [PDF](https://arxiv.org/pdf/2601.11404v1)

**作者:** Linqing Zhong `[一作]` (Beihang University), Guanghui Ren `[通讯]` (AgiBot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在动作空间进行推理的通用机器人策略框架 ACoT‑VLA，并通过显式动作推理器（EAR）和隐式动作推理器（IAR）为策略提供精细的动作引导，显著提升了多种仿真与真实世界任务的成功率。

**💡 创新点**

创新点在于将 Chain‑of‑Thought 思维从传统的语言/视觉中间表示转移到直接的动作空间；通过 EAR 生成粗粒度动作参考轨迹、IAR 提取 VLM 隐含的动作先验，实现了更精确、更具运动语义的中间推理，从而弥合语义‑运动的鸿沟。

**🔧 技术方法**

使用了预训练的 Vision‑Language 模型（SigLIP+Gemma 2B）作为特征提取器；EAR 采用轻量级 Transformer 结合跨模态注意力生成参考动作；IAR 用可学习的查询与跨层注意力提取隐式动作先验；动作预测头采用流匹配（flow‑matching）训练的扩散模型；整个框架在 BFloat16 下在 8 张 H100 GPU 上训练。

**📊 数据集**

主要数据集包括 LIBERO、LIBERO‑Plus 与 VLABench 的仿真演示集；真实世界实验使用自采集的 AgiBot G1 与 AgileX 机器人执行擦污、倒水、开放集拾取等任务。

**📈 对比分析**

在 LIBERO 上取得 98.5% 成功率（相比基线 π_0.5 提升 1.6%），在 LIBERO‑Plus 上达 84.1%（在多种扰动下分别提升 11.6%、16.3% 和 12.5%），在 VLABench 上获得 47.4% 的进度/意图得分，整体优于现有最先进方法。

**⚠️ 局限性**

局限性包括：1) 对 VLM 预训练数据的依赖，若语义或视觉特征偏差仍可能影响动作推理；2) 需要在训练阶段使用大量标注演示；3) 当前的动作参考长度和步长设置需手工调参，难以自适应不同任务；4) 在高度动态或极端未知环境下的鲁棒性仍待进一步验证。

---

## 295. Wetland mapping from sparse annotations with satellite image time series and temporal-aware segment anything model

**arXiv ID:** 2601.11400 | [PDF](https://arxiv.org/pdf/2601.11400v1)

**作者:** Shuai Yuan `[一作]` (University of Hong Kong), Peng Gong `[通讯]` (University of Hong Kong)

**通讯引用:** 64484 | [OpenAlex ID](https://openalex.org/A5059264917)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 WetSAM，一个基于 SAM 的双分支框架，用稀疏点标签和卫星影像时间序列实现湿地高精度分割。

**💡 创新点**

创新点在于：① 将 SAM 与层级多尺度适配器和动态时序聚合模块结合；② 通过时间约束的区域生长生成稠密伪标签；③ 双向一致性正则化促使时序与空间分支互相对齐，从而在弱监督下实现空间连贯与语义准确。

**🔧 技术方法**

使用的技术包括：层级多尺度适配器、时间位置编码、趋势-残差分解的动态时序聚合、基于注意力的高频事件提取、区域生长伪标签生成、双向预测对齐损失、SAM 基础模型及其冻结编码器。

**📊 数据集**

实验数据集为 Sentinel‑2 Level‑2A 两年时间序列，覆盖全球八个湿地区域（约 5000 km²/区），共采集 2000–4000 个稀疏点标签，涵盖水面、沼泽、泥滩等四类。

**📈 对比分析**

与 SAM、SAM 2、DINO‑SAM、PSPNet、SegNext、CRGNet、UTAE 等基线对比，WetSAM 在所有八区平均 F1 得分 85.58%，明显优于其他方法（最高仅 83.15% 的 SAM 2）。

**⚠️ 局限性**

局限性包括：仅使用光学影像易受云遮蔽影响；对高噪声标签敏感；冻结的 SAM 编码器限制了对遥感域的自适应；未集成 SAR 等多模态数据，难以实现全天气湿地监测。

---

## 296. Coding Schemes for the Noisy Torn Paper Channel

**arXiv ID:** 2601.11501 | [PDF](https://arxiv.org/pdf/2601.11501v1)

**作者:** Frederik Walter `[一作]` (Technical University of Munich), Antonia Wachter-Zeh `[通讯]` (Technical University of Munich)

**通讯引用:** 1143 | [OpenAlex ID](https://openalex.org/A5041962883)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了针对噪声破裂纸通道（Noisy Torn Paper Channel）的两种编码方案，能够在收到无序、噪声碎片的情况下重新组装并纠正错误，从而实现 DNA 存储等长时序数据的可靠恢复。

**💡 创新点**

创新点在于：① 将传统的静态标记与基于数据的局部敏感哈希（LSH）相结合，构建可容忍替换误差的重组指针；② 采用多层嵌套哈希编码与 LDPC 外码相结合的结构，在无需显式索引的情况下实现高码率；③ 通过大规模仿真展示在不同破碎率 α 与替换概率 p_s 下，成功率可超过 99%，并与理论容量及现有方案对比。

**🔧 技术方法**

主要技术包括：LDPC 低密度奇偶校验码、De Bruijn 序列作为索引、静态 001 标记、局部敏感哈希（LSH）以及基于束搜索（beam search）的重组与错误纠正算法。

**📊 数据集**

使用的是模拟数据集：在不同 α（0.05,0.07,0.10）和 p_s（0.4%，0.9%，1.8%，5%）组合下，利用 IEEE 802.16e WiMAX 与 IEEE 802.22 WRAN 标准 LDPC 码作为外码，进行仿真验证。

**📈 对比分析**

方法对比：在同一信道条件下比较静态标记、Stride 1、Stride 2（LSH）以及块（Block）四种标记策略。结果表明：当 p_s 较高（≥1.8%）时，静态标记鲁棒性更强；当 p_s 较低（≤0.9%）时，Stride 2 的 LSH 能显著降低误码率。总体而言，两种方案的码率均优于理论容量，并在所有仿真场景中实现了 >99% 的成功率。

**⚠️ 局限性**

局限性主要包括：① 计算资源瓶颈，解码失败主要因束搜索耗尽资源；② 仅考虑替换错误和碎片化，未涵盖插入/删除错误，限制了对真实 DNA 存储噪声模型的适用性；③ 实验数据为仿真，缺乏真实 DNA 读写实验的验证。

---

## 297. The Poisoned Apple Effect: Strategic Manipulation of Mediated Markets via Technology Expansion of AI Agents

**arXiv ID:** 2601.11496 | [PDF](https://arxiv.org/pdf/2601.11496v1)

**作者:** Eilam Shapira `[一作]` (Technion Israel Institute of Technology), Moshe Tennenholtz `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究 AI 代理技术扩展对经济博弈（议价、谈判、说服）中的策略互动和监管结果的影响，发现了“毒苹果”效应，说明技术可被用作监管操纵工具。

**💡 创新点**

创新点在于首次系统性揭示技术可被作为“毒苹果”投机性威胁，迫使监管者改变市场设计以获取利益；并证明静态监管框架易被利用，提出需要动态、适应性市场设计。

**🔧 技术方法**

采用博弈论的元游戏框架，计算 Nash 均衡，模拟技术加入后市场环境和监管目标（公平/效率）的优化过程。

**📊 数据集**

使用 GLEE 数据集，包含 13 种最先进 LLM 在 1,320 个配置下产生的 580,000+ 策略决策，用于构建 50,000+ 的元游戏样本。

**📈 对比分析**

通过比较技术扩展前后均衡收益与监管指标的变化，对比公平和效率目标下的结果；实验显示约 1/3 的收益逆转即使新技术未被使用，也可显著提升某方收益；整体上技术扩展往往对公平目标产生负面影响，需动态调优。

**⚠️ 局限性**

局限性：实验仅覆盖三类非合作博弈，未涉及多方或动态博弈；模型假设代理仅一次选择技术，未模拟长期演化；结果高度依赖 GLEE 数据集和 LLM 表现，可能不完全泛化。

---

## 298. CTest-Metric: A Unified Framework to Assess Clinical Validity of Metrics for CT Report Generation

**arXiv ID:** 2601.11488 | [PDF](https://arxiv.org/pdf/2601.11488v1)

**作者:** Vanshali Sharma `[一作]` (Northwestern University), Ulas Bagci `[通讯]` (Northwestern University)

**通讯引用:** 9288 | [OpenAlex ID](https://openalex.org/A5030188696)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了统一评估框架 CTest‑Metric，用于系统性评估 CT 放射学报告生成指标的写作风格鲁棒性、合成错误敏感度和与专家判断的相关性。

**💡 创新点**

三模组评估方法首次将写作风格可泛化、合成错误注入、专家相关性三项指标标准化，并公开框架与部分数据实现可复现。

**🔧 技术方法**

利用 LLM 进行报告改写与错误注入、传统 NLG 指标（BLEU/ROUGE/METEOR/BERTScore‑F1）与临床效用指标（F1‑RadGraph、RaTEScore、GREEN Score、CRG）以及 Spearman 相关性分析。

**📊 数据集**

使用公开的 CT‑RATE 胸部 CT 数据集及其对应的放射学报告，随机挑选 175 个冲突案例进行专家评分。

**📈 对比分析**

通过对重写差异、错误等级变化以及与专家评分的相关系数对指标进行比较，发现 GREEN Score 与专家评分相关性最高（ρ≈0.70），CRG 与专家评分呈负相关。

**⚠️ 局限性**

局限性包括仅基于单一数据集，LLM 改写与错误注入未独立验证，且评估多样性和临床推广性待进一步验证。

---

## 299. Health Facility Location in Ethiopia: Leveraging LLMs to Integrate Expert Knowledge into Algorithmic Planning

**arXiv ID:** 2601.11479 | [PDF](https://arxiv.org/pdf/2601.11479v1)

**作者:** Yohai Trabelsi `[一作]` (John A. Paulson School of Engineering and Applied Sciences Harvard University), Milind Tambe `[通讯]` (John A. Paulson School of Engineering and Applied Sciences Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出一种将大型语言模型与子模最大化相结合的混合框架，用于在埃塞俄比亚卫生设施升级规划中同时优化人群覆盖率和专家意见的一致性。

**💡 创新点**

创新点在于将语言模型的自然语言反馈映射为可量化的对齐指标，并在迭代过程中通过α–β参数实现理论覆盖保证与专家建议的可控权衡，同时支持在线预算动态更新。

**🔧 技术方法**

使用技术包括子模函数的贪心与“Guided Greedy”算法、LLM（Gemini）生成的专家对齐函数、提示优化（prompt gradient descent）以及多目标约束求解。

**📊 数据集**

数据集为2026年埃塞俄比亚各地区的人口预测、两小时步行可达性地图以及由Gemini生成的20条包含冲突的专家建议句子。

**📈 对比分析**

通过与纯贪心和量化反馈两种基线对比，实验显示语言反馈版本在保持1‑e^{-αβ}覆盖下显著提升对齐分数，α增大提升覆盖率但降低对齐，整体性能可在两项指标间权衡。

**⚠️ 局限性**

限制包括对子模目标的依赖、对LLM提示质量和推理准确度的敏感、未覆盖更复杂的空间/预算约束以及仅在埃塞俄比亚三地区验证，泛化性待进一步验证。

---

## 300. Isotropy-Optimized Contrastive Learning for Semantic Course Recommendation

**arXiv ID:** 2601.11427 | [PDF](https://arxiv.org/pdf/2601.11427v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 301. Predict the Retrieval! Test time adaptation for Retrieval Augmented Generation

**arXiv ID:** 2601.11443 | [PDF](https://arxiv.org/pdf/2601.11443v1)

**作者:** Xin Sun `[一作]` (Chinese Academy of Sciences), Liang Wang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 42306 | [OpenAlex ID](https://openalex.org/A5115602506)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种在推理时对检索增强生成（RAG）模型进行动态参数自适应的方法，利用检索结果内部的前后缀关系作为自监督信号，实现无标签、无额外训练数据的实时模型优化。

**💡 创新点**

创新点在于：①把检索段落拆分为前缀-后缀对，模型学习从前缀预测后缀，从而更好捕捉领域特定语言模式；②通过自监督目标在推理阶段即时更新参数，解决传统RAG在专业领域的分布偏移问题；③该方法无需访问源域数据或额外标注，极大提升隐私友好性与实用性。

**🔧 技术方法**

核心技术包括：检索增强生成（RAG）框架、测试时适应（TTA）与自监督学习、prefix-suffix预测任务、AdamW优化、梯度累积与裁剪；实现基于Llama、ChatGLM等大语言模型。

**📊 数据集**

使用了CRAG（覆盖金融、体育、音乐、电影、开放领域的2706条问答），以及医学领域的PubMedQA（1000条）和BioASQ（500条）数据集；检索使用BM25或MiniLM重排。

**📈 对比分析**

与naive‑RAG、Chain‑of‑Thought、In‑Context Learning以及Ret‑Robust、RAAT、Self‑RAG等基线进行对比。实验显示在6个专业领域平均提升3.7%（最高单域提升19.4%），并在医学领域达到10.8%–25.0%的显著改进；相比CoT的推理时间几乎减半。

**⚠️ 局限性**

局限性：①推理时仍需额外计算，尤其多对前后缀时会增加1.75–2.60 s的平均延迟；②对检索质量高度依赖，若检索段落不相关，适应效果有限；③在极端领域偏移或极短检索文本时，模型更新效果可能不足。

---

## 302. Map2Thought: Explicit 3D Spatial Reasoning via Metric Cognitive Maps

**arXiv ID:** 2601.11442 | [PDF](https://arxiv.org/pdf/2601.11442v1)

**作者:** Xiangjun Gao `[一作]` (Hong Kong University of Science and Technology), Youngkyoon Jang `[通讯]` (Huawei Noah’s Ark Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Map2Thought 框架，实现 3D 视觉‑语言模型的显式可解释空间推理

**💡 创新点**

创新点在于将离散网格与连续度量空间融合成 Metric‑CogMap，并通过 Cog‑CoT 进行可追踪的链式几何推理；同时设计了面向视频的高效图谱构建流水线

**🔧 技术方法**

核心技术包括：预训练 2D/3D 感知模型（CLIP‑ViT、CUT3R 等）、跨模态注意力融合、基于 Covisibility Map 的多视角检测与追踪、SAM2 语义分割、对齐至真实尺度的全局坐标转换，以及 Cog‑CoT 的确定性几何运算

**📊 数据集**

使用 ScanNet、ScanNet++、ARKitScenes 进行训练，构建 Metric‑CogMap；在 VSI‑Bench 上进行评测

**📈 对比分析**

与多种基准（包括 GPT‑4o、Gemini‑1.5‑Pro、InternVL2、LongViLA、VLM‑3R、VG‑LLM‑8B 等）对比，Map2Thought 在 50% 训练数据下达成 59.9% 精度，几乎等同 100% 训练的 60.9%；在 10%/25% 数据下分别超越同类方法 5.3%/4.8%；在 Abs.Dist、Room Size 等对度量空间敏感任务上表现尤为突出

**⚠️ 局限性**

主要局限在于 Metric‑CogMap 的构造质量；若检测/分割误差较大，Rel.Dist/Rel.Dir 的性能会受影响；此外，模型依赖于预训练 3D 视觉基础模型的精度，未来可通过更高质量的感知与地图生成提升鲁棒性

---

## 303. Hierarchical Orthogonal Residual Spread for Precise Massive Editing in Large Language Models

**arXiv ID:** 2601.11441 | [PDF](https://arxiv.org/pdf/2601.11441v1)

**作者:** Xiaojie Gu `[一作]` (Independent Researcher), Andi Zhang `[通讯]` (University of Manchester)

**通讯引用:** 2082 | [OpenAlex ID](https://openalex.org/A5077911588)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的大规模模型编辑方法 HORSE，能够在不重新训练模型的前提下快速、精准地更新预训练语言模型的知识。

**💡 创新点**

创新点在于引入层级正交残差传播机制，将编辑残差在 Transformer 的各层正交化并实时调整权重，从而显著降低新旧知识冲突，并通过 token‑level 超网络实现高效、稳定的参数更新。

**🔧 技术方法**

使用技术包括：残差矩阵优化、层级正交传播、基于 MLP 的超网络、token‑level 训练、以及正则化约束来保证编辑与原知识的兼容。

**📊 数据集**

在两大数据集 zsRE（问答对）和 CounterFact（对抗事实）上对 GPT‑J、LLaMA‑2‑7B 和 Mistral‑7B 进行实验。

**📈 对比分析**

与 FT、LoRA、MEMIT、MEND、PMET、MALMEN、EMMET 等方法比较，HORSE 在大多数指标上实现平均 +6.26% 的提升，尤其在 Specificity 上提升 +10.12%，并且编辑速度最快（100 条样本仅需 8.85 秒）。

**⚠️ 局限性**

局限性包括：在个别单指标上不一定领先、仅在两大数据集上评估、对极大规模编辑的泛化性和对模型其它下游任务的潜在副作用仍需进一步验证。

---

## 304. Sociotechnical Challenges of Machine Learning in Healthcare and Social Welfare

**arXiv ID:** 2601.11417 | [PDF](https://arxiv.org/pdf/2601.11417v1)

**作者:** Tyler Reinmund `[一作]` (University of Oxford), Marina Jirotka `[通讯]` (University of Oxford)

**通讯引用:** 4568 | [OpenAlex ID](https://openalex.org/A5023741875)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过实地研究、文献综述和共设计工作坊，提出了针对医疗与社会福利场景的机器学习社会技术挑战框架与过程模型。

**💡 创新点**

创新点在于将社会技术挑战分为11类并关联到机器学习支持的护理路径，以及提出三条导致挑战出现的动态过程（设计决定、约束应对、脚本偏离）。

**🔧 技术方法**

采用的技术包括定性田野调查、案例研究、系统文献回顾和共设计工作坊，并以技术结构理论为理论支撑。

**📊 数据集**

未使用公开数据集，主要基于一所英国社会福利组织在跌倒预防项目中的实际部署数据与访谈记录。

**📈 对比分析**

本文并未进行性能比较或量化评估，而是侧重对挑战的归纳与解释，未提供模型性能指标。

**⚠️ 局限性**

局限性包括样本来源单一（仅一所机构）、缺乏定量验证、框架尚处于初步阶段，后续需在更广泛场景中检验与细化。

---

## 305. Space-Optimal, Computation-Optimal, Topology-Agnostic, Throughput-Scalable Causal Delivery through Hybrid Buffering

**arXiv ID:** 2601.11487 | [PDF](https://arxiv.org/pdf/2601.11487v1)

**作者:** Paulo Sérgio Almeida `[一作]` `[通讯]` (INESC TEC and University of Minho), Paulo Sérgio Almeida (INESC TEC and University of Minho)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的混合缓冲策略（Sender Permission to Send + FIFO），实现了拓扑无关、空间最优、计算最优且吞吐量可扩展的因果交付算法。

**💡 创新点**

创新点在于将因果交付的控制权交给发送方（SPS）并与接收方的 FIFO 缓冲结合，克服了纯发送方缓冲算法的吞吐量瓶颈；同时设计了滑动数组/滑动映射数据结构，实现了 O(1) 的元数据管理和计算复杂度。

**🔧 技术方法**

使用基于滑动数组/滑动映射的轻量级数据结构，采用唯一全局时钟生成消息 ID，结合确认（ACK）与许可（Permit）消息实现协议；算法在无可靠网络环境下仍能保证安全性与活性。

**📊 数据集**

该工作主要是理论分析与算法设计，并未使用具体实验数据集；未来工作计划在大规模微服务环境中进行性能评估。

**📈 对比分析**

通过与传统接收方缓冲算法（RST、KS、Newtop）以及纯发送方缓冲算法（MF、Cykas）在空间开销、计算时间、吞吐量和活性等维度进行对比；实验结果表明该算法在元数据占用和计算开销上保持常数级，吞吐量可与网络延迟无关，优于现有算法。

**⚠️ 局限性**

局限性在于该算法并非延迟最优，因果交付延迟相对较高；在极低延迟需求或高网络丢包场景下，需进一步优化许可传输和重传策略；另外，算法的实现复杂度较高，实际部署需要仔细处理并发与异常情况。

---

## 306. Generative Scenario Rollouts for End-to-End Autonomous Driving

**arXiv ID:** 2601.11475 | [PDF](https://arxiv.org/pdf/2601.11475v1)

**作者:** Rajeev Yasarla `[一作]` (Qualcomm AI Research), Hong Cai `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Generative Scenario Rollouts 框架，将 Vision‑Language‑Action 模型的规划、预测与场景生成融合，生成语言对齐的未来交通场景。

**💡 创新点**

创新点：①将 VLA 模型与自动回归场景滚动联合训练；②引入 Rollout‑Consistency 损失与 GRPO 强化学习以保持时序一致性与安全性；③通过 VQA 与场景描述实现多步动作的语言对齐。

**🔧 技术方法**

使用 VAE 规划头、LLM（Qwen2.5VL 或 ORION）生成隐含 token，KL 一致性损失、GRPO 强化学习，以及视觉+语言多模态编码。

**📊 数据集**

数据集：Bench2Drive（闭环评测）和 nuScenes（开环评测），并利用 ChatB2D、DriveLM‑nuScenes 生成 VQA 语料。

**📈 对比分析**

在 Bench2Drive 闭环评测中 Driving Score 提升 15.7（Qwen）或 4.16（ORION），Success Rate 提升 26.2/5.5%；在 nuScenes 开环评测中 L2 误差降至 0.31（-67.7%），碰撞率 0.14（-76.7%），零样本跨数据集迁移表现显著。

**⚠️ 局限性**

限制：依赖大量多模态标注，回归滚动对计算资源要求高；模型在极端稀疏场景仍可能漂移；未在真实车辆上验证，安全性需进一步评估。

---

## 307. Learning Semantic-Geometric Task Graph-Representations from Human Demonstrations

**arXiv ID:** 2601.11460 | [PDF](https://arxiv.org/pdf/2601.11460v1)

**作者:** Franziska Herbert `[一作]` (Technical University of Darmstadt), Georgia Chalvatzaki `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 750 | [OpenAlex ID](https://openalex.org/A5026055366)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用人类双手演示学习语义-几何任务图谱，预测未来动作、对象及对象运动，并将其迁移到物理双手机器人进行在线动作选择。

**💡 创新点**

提出联合编码语义关系与几何演化的语义-几何任务图谱，并使用MPNN+Transformer解码器实现多时序预测，突破了单纯语义或单纯几何模型在高变异任务上的局限。

**🔧 技术方法**

消息传递神经网络（MPNN）编码器、Transformer解码器、RoPE时序编码、动作-对象-运动三分支预测头、动作分块与时间加权集成等技术。

**📊 数据集**

KIT Bimacs（5个烹饪+4个工作坊任务，6名受试者）和自制Ours(Bimacs)（4个烹饪相关任务，4名受试者）数据集，亦使用10次机器人演示做迁移评估。

**📈 对比分析**

与Dreher、Lagamtzis、Transformer、Decoder-Only等基线对比；在动作/对象预测准确率、运动RMSE上MPNN表现最佳，尤其在高动作/对象变异任务；机器人上迁移后成功率达90%，动作匹配率99%/100%。

**⚠️ 局限性**

需要在机器人上微调以补偿运动差异；仅在简单任务上验证，复杂任务时预定义原语受限；动作/运动频率差异导致记忆化，需进一步改进；数据量和多模态输入不足。

---

## 308. Inter-patient ECG Arrhythmia Classification with LGNs and LUTNs

**arXiv ID:** 2601.11433 | [PDF](https://arxiv.org/pdf/2601.11433v1)

**作者:** Wout Mommen `[一作]` (imec), Piet Wambacq `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 8910 | [OpenAlex ID](https://openalex.org/A5059069030)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究提出并实现了可在心电图异常分类的低功耗逻辑门网络（LGN）和查找表网络（LUTN）。

**💡 创新点**

创新点包括在MIT‑BIH数据集上首次在交叉患者模式下应用LGN/LUTN、使用基于MUX布尔方程的LUT训练方法、采用速率编码提升LGN性能，以及提出更优的特征提取预处理。

**🔧 技术方法**

主要技术包括可微分逻辑门网络、可微分查找表网络、速率编码、基于MUX的LUT训练、FPGA实现及相关功耗估算。

**📊 数据集**

使用了MIT‑BIH心律失常数据库进行四分类实验，并在MNIST/Fashion‑MNIST上验证LUTN训练方法。

**📈 对比分析**

与现有SVM、SNN、CNN等基线相比，LGN/LUTN在交叉患者模式下达到了94%以上准确率、最高0.683的jκ指标，仅耗费几千FLOPs，FPGA功耗约5–7 mW，显示出显著的功耗与资源优势。

**⚠️ 局限性**

局限包括对极端类不平衡的处理不足、深层6‑LUTN训练不稳定、速率编码在FPGA上导致功耗/延迟上升，以及对小样本或少量患者的泛化仍待验证。

---

## 309. Forcing and Diagnosing Failure Modes of Fourier Neural Operators Across Diverse PDE Families

**arXiv ID:** 2601.11428 | [PDF](https://arxiv.org/pdf/2601.11428v1)

**作者:** Lennon Shikhman `[一作]` `[通讯]`, Lennon Shikhman

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在多种 PDE 家族（非线性薛定谔、泊松、Navier–Stokes、Black–Scholes、Kuramoto–Sivashinsky）中构建并执行一系列系统的压力测试，评估 Fourier Neural Operator（FNO）的鲁棒性和失效模式。

**💡 创新点**

提出了完整的压力测试框架和失效模式图谱，系统地揭示参数、边界、分辨率、长期递归与输入扰动对 FNO 性能的影响，并通过大量模型实验量化失效程度。

**🔧 技术方法**

使用 FNO 作为基准模型，并与 DeepONet 对比；训练时采用 L2 损失、GPU 加速；压力测试包括参数/系数偏移、边界/终值变动、分辨率外推、长时递归滚动、输入扰动；采用频谱误差分析、误差衰减因子等指标。

**📊 数据集**

为五类 PDE 生成训练和测试数据，涵盖不同参数范围、边界/终值分布、网格分辨率，最终共训练 1,000 个模型。

**📈 对比分析**

通过计算每个压力测试下的误差衰减因子（out‑of‑distribution 误差 / in‑distribution 误差）以及频谱误差分布，比较不同模型在相同 PDE 上的鲁棒性。结果显示：参数/边界偏移可导致误差提升数倍，分辨率外推主要影响高频误差，长期滚动导致误差指数放大，输入扰动对 FNO 稳定；DeepONet 在分辨率外推方面表现稍好。

**⚠️ 局限性**

模型依赖于训练数据分布，无法实现对超出范围的参数或极端边界的泛化；受限于固定 Fourier 模式数的光谱偏差；长时递归易累积误差，未能保持物理守恒或动力学不稳定性；实验仅覆盖五类 PDE，其他类型或几何变形尚未验证。

---

## 310. Learning-Based Shrinking Disturbance-Invariant Tubes for State- and Input-Dependent Uncertainty

**arXiv ID:** 2601.11426 | [PDF](https://arxiv.org/pdf/2601.11426v1)

**作者:** Abdelrahman Ramadan `[一作]` (Queen’s University), Sidney Givigi `[通讯]` (Queen’s University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于高斯过程学习的管束模型预测控制框架，通过构造收缩的扰动不变管道，实现对状态与输入相关不确定性的安全约束；

**💡 创新点**

创新点包括：① 将高斯过程后验置信椭圆固定并外接为多面体，避免学习-控制循环中的圆环依赖；② 在升维空间中引入图约束，将不变集的寻找转化为单调外向内迭代，保证收敛与嵌套；③ 采用两时标策略，使得每个学习阶段的管道可递归收缩；

**🔧 技术方法**

使用的技术主要包括高斯过程回归、置信椭圆到多面体的外近似、升维图约束与外向内固定点迭代、可测选择定理以及MPC管束收缩；

**📊 数据集**

实验采用二维双积分器的仿真数据，构造的扰动包括速度依赖阻力、输入相关驱动以及过程噪声；

**📈 对比分析**

与传统最坏情况固定边界方法对比，实验结果显示在同一安全域下约束收敛速度提升22.9倍，整体保守性下降55.4%，同时仍保持硬约束满足；

**⚠️ 局限性**

局限性在于：① 仅处理无时间相关的扰动，未考虑色噪声；② 需要先验或估计 Lipschitz 常数以保证统一安全；③ 实际可测选择策略的实现仍为挑战。

---

## 311. On the Virtual Network Embedding polytope

**arXiv ID:** 2601.11419 | [PDF](https://arxiv.org/pdf/2601.11419v1)

**作者:** Amal Benhamiche `[一作]` (Orange Research), Alexis Schneider `[通讯]` (Université Sorbonne Paris Nord)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `40105733-5154-44cd-8090-a8cab9e64b07` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文对虚拟网络嵌入(VNE)问题的多面体进行了首次研究，提出并证明了一组新的有效不等式可完整刻画单条虚拟边在路径型基底网络上的嵌入多面体，

**💡 创新点**

创新点在于设计了流离场（flow departure）与流连续性（flow continuity）两类不等式，并通过自定义的流分解算法证明了其在路径案例下可获得多面体完全描述，

**🔧 技术方法**

主要技术包括基于双向流模型的整数规划表述、流分解理论以及线性松弛强化的多面体分析，

**📊 数据集**

实验使用了100个14节点22条边的Erdős–Rényi随机虚拟网络，以及Intellifiber、Uninett、TW等70–75节点的真实基础网络，并对节点容量稀疏情形进行多场景测试，

**📈 对比分析**

与仅使用基本流模型相比，加入流离场不等式可将CPLEX求解时间缩短约2–3倍（在小型真实网络上可达20倍），并显著减少分支界限节点数与线性松弛误差；流连续性不等式在稀疏容量场景下亦有提升，但因约束数量庞大导致整体求解时间略增；

**⚠️ 局限性**

当前研究仅证明了路径型基底网络的完整性，对更复杂的树型或更大子图的情形尚未给出理论保证，且流连续性不等式规模过大，若直接使用会影响求解效率，需要进一步的分支裁剪或启发式选取策略。

---

## 312. Convergence Properties of Good Quantum Codes for Classical Communication

**arXiv ID:** 2601.11498 | [PDF](https://arxiv.org/pdf/2601.11498v1)

**作者:** Alptug Aytekin `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**通讯引用:** 13727 | [OpenAlex ID](https://openalex.org/A5021132487)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究量子通道上实现容量的经典信息传输码的输出分布收敛性质，证明最佳输出分布唯一性并给出 vanishing 与 non‑vanishing 误差情形下的收敛定理。

**💡 创新点**

首次将经典通道的输出统计收敛结果推广到量子通道，证明量子容量上最佳输出态唯一且给出针对强逆性通道的二阶相反推导，实现了对量子良好码输出分布的渐近一致性。

**🔧 技术方法**

采用量子相对熵、数据处理不等式、Donald 恒等式、量子超曲性结果、以及量子版本的黄金公式和 Fano 不等式，结合量子通道的强逆性与可加性。

**📊 数据集**

本研究为理论分析，无需外部实验数据集。

**📈 对比分析**

通过相对熵距离衡量输出分布与最优分布的差距，证明其 1/n 归零速率，并在强逆性通道下给出二阶收敛边界，性能与经典结果相当。

**⚠️ 局限性**

对非强逆性或非可加通道、非确定性编码、平均误差标准下的结果尚未完整证明，且对实际量子硬件噪声模型的适用性未作验证。

---

## 313. Exploring LLM Features in Predictive Process Monitoring for Small-Scale Event-Logs

**arXiv ID:** 2601.11468 | [PDF](https://arxiv.org/pdf/2601.11468v1)

**作者:** Alessandro Padella `[一作]` (Università degli Studi di Padova), Marlon Dumas `[通讯]` (University of Tartu)

**通讯引用:** 29617 | [OpenAlex ID](https://openalex.org/A5085212075)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在数据稀缺环境下，利用大型语言模型（LLM）构建预测过程监控框架，并在多个关键绩效指标（KPI）上进行实验；

**💡 创新点**

提出了LLM基准化推理方法（beta‑learners），证明LLM利用内在语义知识和推理机制；

**🔧 技术方法**

使用Gemini 2.5 Flash LLM、prompting技术、语义哈希、β‑learner抽象、Good‑Turing估计等；

**📊 数据集**

采用三份业务事件日志（BPI12、BacA、Hospital）以及相关KPI（总耗时、活动出现率）；

**📈 对比分析**

与CatBoost、PGTNet等传统模型对比，LLM在仅100条训练样本时在总耗时和活动出现率上表现相当或更优；在语义哈希后性能下降，表明LLM利用语义；

**⚠️ 局限性**

受限于LLM上下文长度、对大型日志的适用性不明，且对模型内部机制解释仍不完全透明；

---

## 314. The unreasonable effectiveness of pattern matching

**arXiv ID:** 2601.11432 | [PDF](https://arxiv.org/pdf/2601.11432v1)

**作者:** Gary Lupyan `[一作]` (University of Wisconsin), Blaise Agüera y Arcas `[通讯]` (Google)

**通讯引用:** 11458 | [OpenAlex ID](https://openalex.org/A5044698998)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在面对大量随机替换、几乎全无语义信息的“Jabberwocky”式文本时，能够通过结构模式匹配恢复原意，并通过多轮实验验证该能力。

**💡 创新点**

创新点在于将“看似无意义文本”视为可通过语法、形态和上下文模式恢复的压缩信息，提出并验证LLM在极度扭曲文本中的“去模糊”能力，挑战传统的“鹦鹉式模仿”与“数据库式存储”类比。

**🔧 技术方法**

采用多种大型语言模型（GPT‑4、ChatGPT o3、Gemini 2.5 Pro 等）进行翻译任务，并利用词嵌入（OpenAI text‑embedding‑3‑large）计算原文与模型输出的相似度，作为评估指标。

**📊 数据集**

数据集主要是将已公开文本（莎士比亚作品、新闻稿、社交媒体帖子、播客转录、学生论文等）通过随机字符串替换生成的“Jabberwockified”版本；包括新近发布的 ESPN 与 Reddit 片段，确保测试文本不在模型预训练语料中。

**📈 对比分析**

评估方法为：①对每条“Jabberwockified”文本让模型生成翻译；②计算生成文本与原文的词嵌入相似度；③在多条文本上统计相似度分布。结果显示，平均相似度在 0.75–0.90 之间，证明模型在未见过的文本上也能保持较高的意义恢复精度。

**⚠️ 局限性**

局限性包括：
1) 模型仍高度依赖海量训练数据，低语境或极度随机化的文本仍可能出现误译；
2) 解释机制仍不完整，难以完全区分是模式匹配还是其他隐式学习策略；
3) 在高度专业化或跨语种文本中，结构模式匹配的泛化能力尚待进一步验证。

---

## 315. Relational Linearity is a Predictor of Hallucinations

**arXiv ID:** 2601.11429 | [PDF](https://arxiv.org/pdf/2601.11429v1)

**作者:** Yuetian Lu `[一作]` (Center for Information and Language Processing), Hinrich Schütze `[通讯]` (Center for Information and Language Processing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在面对合成实体（模型无法回答的实体）时的幻觉现象，并提出关系的线性度是导致幻觉的关键因素。

**💡 创新点**

创新点在于：①把关系线性度（Δcos）与幻觉率关联起来，提供了一个新的幻觉预测指标；②构建了六种线性/非线性关系的合成实体数据集，用于系统性评估。

**🔧 技术方法**

采用了线性关系探测（平均差向量）、余弦相似度差 Δcos 作为线性度指标；使用四个指令调优的大型语言模型（Gemma‑7B‑IT、Llama‑3.1‑8B‑Instruct、Mistral‑7B‑Instruct‑v0.3、Qwen2.5‑7B‑Instruct）和自动评估工具对模型输出进行分类。

**📊 数据集**

使用了自制的 SyntheticRelations 数据集（6 个关系，每个关系 1000 个合成实体）以及公开的 LRE (语言关系嵌入) 数据集来计算线性度。

**📈 对比分析**

在合成数据上，四个模型的幻觉率与 Δcos 显著正相关（Pearson r≈0.78‑0.82），表明更线性的关系导致更高幻觉率；在自然数据上则呈负相关，说明关系线性度是一个重要的预测指标。

**⚠️ 局限性**

局限性包括：仅使用平均差向量的平移模型，未考虑更一般的线性映射；实验仅覆盖 6 种关系，合成实体可能受输出空间先验影响；结果仅为相关性，未进行因果干预。

---

## 316. PubMed-OCR: PMC Open Access OCR Annotations

**arXiv ID:** 2601.11425 | [PDF](https://arxiv.org/pdf/2601.11425v1)

**作者:** Hunter Heidenreich `[一作]` (Roots ai), Ben Elliott `[通讯]` (Roots ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文构建了 PubMedOCR 语料库，基于 PubMed Central 开放获取 PDF 直接进行 OCR，生成段落、行、词级的坐标框并以统一 JSON 格式发布。

**💡 创新点**

创新点在于跳过 PDF/XML 对齐，采用 OCR‑first 方法获得更高的文本回收率，并提供粒度细化的布局标注（段落、行、词）供布局感知建模和基于坐标的 QA 使用。

**🔧 技术方法**

技术实现使用 Google Cloud Vision OCR 对 150 DPI 图像进行识别，并通过基于行高阈值的聚类手段从词框重构行框，随后聚合生成段落框。

**📊 数据集**

所使用的数据集为 PubMed Central Open Access（PMCOA）中的 209.5 k 篇论文，共 1.5 M 页，包含约 1.3 B 词、164 M 行和 61 M 段落。

**📈 对比分析**

与 OCR‑IDL 等现有 OCR 资源相比，PubMedOCR 在词与行标注数量上分别提升约 10 倍和 4 倍，且在科学文章中提供更细粒度的布局信息，虽然论文未给出模型性能指标，但说明了数据规模与标注细度的优势。

**⚠️ 局限性**

局限性包括仅依赖单一 OCR 引擎、行标注的聚类方法可能引入排版误差、缺少字符级框和数学表达式/表格结构标注，以及受 PMCOA 许可和期刊分布限制。

---

## 317. New Adaptive Mechanism for Large Neighborhood Search using Dual Actor-Critic

**arXiv ID:** 2601.11414 | [PDF](https://arxiv.org/pdf/2601.11414v1)

**作者:** Shaohua Yu `[一作]` (Nanjing University of Science and Technology), Jakob Puchinger `[通讯]` (EM Normandie Business School)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出基于Dual Actor-Critic的Adaptive Large Neighborhood Search（DAC-ALNS），通过GNN提取实例特征并分别为销毁和修复阶段设计独立策略，显著提升CVRP和VRPTW的搜索效率与泛化能力。

**💡 创新点**

创新点在于：①将销毁与修复视为两独立但耦合的MDP，使用Dual Actor-Critic架构充分利用两者的相互影响；②采用GNN构建可迁移的状态表示，实现不同规模与类型实例的无监督迁移；③提出针对ALNS的专门奖励设计与可扩展的权重调整机制。

**🔧 技术方法**

技术手段包括：双Actor-Critic强化学习框架、图神经网络（GCN+注意力池化）、基于策略梯度的权重更新、以及专门的销毁/修复操作集（随机、最坏、贪婪、惩罚等）。

**📊 数据集**

数据集使用经典的CVRP（Sets A,B,P）与VRPTW（Solomon的C,R,RC集合），按客户数分为小/中/大三组，并对部分集合做in‑distribution与out‑of‑distribution迁移实验。

**📈 对比分析**

与传统ALNS以及单一Actor‑Critic的AC-ALNS进行对比，实验显示DAC-ALNS在Best/Avg Gap上均优于两者，尤其在大规模或时间窗约束强的实例中优势更为明显；尽管在某些小规模实例上性能相当，运行时间略高但总体可接受。

**⚠️ 局限性**

局限性包括：①算法的推理与决策开销相对传统ALNS更大，可能限制极大规模实时应用；②实验仅覆盖CVRP与VRPTW，未验证在更复杂的VRP变体（异构车队、动态需求、电动车等）上的表现；③对不同类型操作集的鲁棒性及超参数敏感性尚未系统评估。

---

## 318. Understanding Help Seeking for Digital Privacy, Safety, and Security

**arXiv ID:** 2601.11398 | [PDF](https://arxiv.org/pdf/2601.11398v1)

**作者:** Kurt Thomas `[一作]` (Google), Nina Taft `[通讯]` (Google)

**通讯引用:** 7178 | [OpenAlex ID](https://openalex.org/A5061868730)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过混合定性与LLM微调方法，构建大规模管道，从1.1B Reddit帖子中识别并标注约300万条数字隐私、安全与安全相关的求助信息。

**💡 创新点**

创新点在于将人工编码定义的求助范畴与LoRA微调的Gemini模型结合，实现高精度（≈93%）自动识别与多主题情绪标注，并提供公开数据集。

**🔧 技术方法**

使用的技术包括定性编码、提示工程、LoRA微调的Gemini 1.5/2.0 Flash LLM、情绪分类器以及多标签主题识别。

**📊 数据集**

使用的数据集为2021-2024年1.1B Reddit原始帖子以及手工标注的2,000条求助与750条多标签主题黄金样本。

**📈 对比分析**

与零样本提示学习相比，LoRA微调模型在帮助求助识别上F1提升至90.7%，在主题识别上F1为92.0%，整体精确率、召回率均在90%以上。

**⚠️ 局限性**

局限性包括爬取范围可能不完整、模型误差可能不均匀、黄金样本存在偏差、无法评估误检/漏检对统计结果的影响、且仅覆盖公开Reddit内容。

---

## 319. Latent Space Inference via Paired Autoencoders

**arXiv ID:** 2601.11397 | [PDF](https://arxiv.org/pdf/2601.11397v1)

**作者:** Emma Hart `[一作]`, Matthias Chung `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种基于配对自编码器（PAIR）的潜在空间推断（LSI）框架，用于在观测不完整或不一致的情况下解决逆问题，主要应用于医学 CT 成像和地震全波形反演。

**💡 创新点**

创新点在于：①将观测空间与参数空间的自编码器通过可学习的潜在映射耦合；②在潜在空间中进行优化（LSI）以重建缺失或损坏的数据，并进一步估计模型参数；③不需要显式求解前向模型或其雅可比，只依赖已训练的自编码器和映射，显著提升了缺失数据下的重建质量。

**🔧 技术方法**

核心技术包括：配对自编码器（两个独立的 Encoder‑Decoder 结构）、潜在映射学习、潜在空间优化（L‑BFGS）、线性/非线性正则化、概率变分配对自编码器（VPAE）以及基于残差网络的卷积架构。

**📊 数据集**

使用的数据集包括：① 2DeteCT 医学 CT 样本（约 4900 训练样本，100 测试样本，125 OOD 样本）；② 2000 个地震波速模型和对应的 30 条源/接收通道数据。

**📈 对比分析**

与基线方法（单一 PAIR、完整数据下的端到端 Encoder‑Decoder、以及在缺失数据上直接训练的 Encoder‑Decoder）进行比较。实验表明：在完整数据时，单一 PAIR 的性能略优；但在随机缺失角度或块缺失角度时，PAIR+LSI 在相对重建误差（RRE）和 SSIM 指标上均优于其他方法，尤其在 OOD 以及高度缺失（>70%）的情形下差距更明显。

**⚠️ 局限性**

局限性包括：① 需要先训练配对自编码器，训练成本高；② 对潜在映射的线性假设在某些高度非线性问题中可能不足；③ 对极端缺失（接近 100%）时仍会出现模糊或不准确的重建；④ 目前仅验证了两类应用，缺乏更广泛的通用性评估。

---

## 320. Validating Search Query Simulations: A Taxonomy of Measures

**arXiv ID:** 2601.11412 | [PDF](https://arxiv.org/pdf/2601.11412v1)

**作者:** Andreas Konstantin Kruff `[一作]` (TH Köln), Philipp Schaer `[通讯]` (TH Köln)

**通讯引用:** 923 | [OpenAlex ID](https://openalex.org/A5087564658)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个针对查询模拟验证的分层分类法（taxonomy），并通过实验验证了各类指标之间的关系，进一步发布了支持该分类法的Python库。

**💡 创新点**

创新点在于（1）系统梳理并归纳了查询模拟验证的多维度指标与维度，形成可操作的分层分类法；（2）用实证分析（因子分析、相关与互信息）验证分类法的合理性；（3）开源了完整的度量实现库。

**🔧 技术方法**

使用了文本相似度度量（Jaccard、Cosine、BERT、WordNet）、传统IR性能指标（nDCG、MAP、MRR等）、SERP重叠度量（Jaccard、RBO）以及基本查询统计量；对这些指标进行因子分析、Pearson/Kendall相关系数和互信息计算。

**📊 数据集**

实验基于四个公开查询模拟数据集：Sim4IA 2025、UQV100、UQV子集（TREC Common Core 2017）和DL seed queries（Deep Learning Tracks 2021/2022），涵盖真实查询、模拟查询和文档/相关性标注。

**📈 对比分析**

比较方法是对同一查询的真实与模拟版本计算上述所有指标，并利用因子分析和相关矩阵探测指标之间的内在关系；实验结果显示传统IR指标高度冗余，语义相似度与SERP重叠度量提供互补信息，整体验证框架有效。

**⚠️ 局限性**

局限包括（1）并未覆盖所有列举的度量，部分指标需要特定数据（如相关性判断、人工评估）而实验数据缺乏；（2）仅基于一对一的最优模拟查询，可能忽略多样化查询生成的影响；（3）验证多聚焦于查询层面，未深入探讨整个交互会话的模拟真实性。

---

## 321. ShapeR: Robust Conditional 3D Shape Generation from Casual Captures

**arXiv ID:** 2601.11514 | [PDF](https://arxiv.org/pdf/2601.11514v1)

**作者:** Yawar Siddiqui `[一作]` (Meta Reality Labs Research), Jakob Engel `[通讯]` (Meta Reality Labs Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出ShapeR，一种多模态条件的直流模型，用于从随意捕获的序列生成完整的三维形状。

**💡 创新点**

创新点在于融合稀疏SLAM点云、姿态图像和文本描述的多模态输入，并通过直流匹配在VecSet潜在空间中生成精细的度量三维形状，同时采用两阶段课程学习和实时增强实现对嘈杂场景的鲁棒性。

**🔧 技术方法**

技术主要包括视觉-惯性SLAM、3D实例检测、DINOv2图像特征、ResNet3D稀疏卷积编码、T5文本编码、CLIP文本嵌入、直流匹配变换器以及基于VAE的VecSet潜在空间。

**📊 数据集**

使用了超过60万网格的人工合成对象数据集进行预训练，Aria合成环境进行微调，并在自建的7个场景、178个对象的野外数据集上进行评估。

**📈 对比分析**

与多种单视角、视图融合以及场景级重建方法对比，ShapeR在Chamfer距离上提升约2.7倍，能够在遮挡、噪声和视角变化下实现完整、度量精确的三维重建。

**⚠️ 局限性**

局限在于仍需依赖SLAM产生的稀疏点云，对极端低纹理或完全遮挡的对象性能有限；且生成过程计算量大，实时性尚需提升。

---

## 322. Applying Formal Methods Tools to an Electronic Warfare Codebase (Experience report)

**arXiv ID:** 2601.11510 | [PDF](https://arxiv.org/pdf/2601.11510v1)

**作者:** Letitia W. Li `[一作]` (BAE Systems), Robert B. Ross `[通讯]` (BAE Systems)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在电子战系统的C++代码库中，对多款开源形式化方法工具进行评估与应用，并结合工程师视角给出可用性改进建议。

**💡 创新点**

首次系统性地从工程师角度比较不同形式化工具的可用性与漏洞覆盖，提出统一术语、自动化注释生成和库代码隔离等改进方案。

**🔧 技术方法**

使用 CBMC、ESBMC、Clang Analyzer、CN、Frama‑Clang、Infer、IKOS、KLEE 等工具，结合 CodeChecker 与 CI/CD 流水线集成实现静态分析。

**📊 数据集**

以现有电子战应用的 C++ 代码库和安全缺陷层级树为数据集，未使用公开标准数据集；主要以库代码（如 spdlog、C++ 标准库）为测试案例。

**📈 对比分析**

通过比较工具覆盖的漏洞类、注释要求、CI/CD 集成耗时等指标；发现单一工具无法覆盖全部漏洞，IKOS 检测最严格但耗时最高；多工具组合可显著提升检测覆盖率。

**⚠️ 局限性**

受限于工具文档不统一、缺乏自动化注释、库代码报告噪声、缺少解析器分析支持，以及安全漏洞分类不一致等因素。

---

## 323. ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes

**arXiv ID:** 2601.11508 | [PDF](https://arxiv.org/pdf/2601.11508v1)

**作者:** Emily Steiner `[一作]` (Stanford University), Iro Armeni `[通讯]` (Stanford University)

**通讯引用:** 3034 | [OpenAlex ID](https://openalex.org/A5014426007)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了稀疏时间间隔室内4D语义实例分割（4DSIS）任务，给出统一端到端的 ReScene4D 方法，并提出兼顾分割质量与时间一致性的 t‑mAP 评价指标。

**💡 创新点**

创新点包括：① 将现有 3DSIS 的 transformer 查询框架扩展到 4D 并消除后期匹配步骤；② 设计了三种跨时间信息共享策略（对比学习、时空序列化、时空掩模）实现实例身份在稀疏扫描中的一致性；③ 研发了专门针对 4DSIS 的 t‑mAP 评价，兼顾 IoU 约束与身份连续性。

**🔧 技术方法**

使用 Mask3D 基础 transformer 结构，结合 Sparse UNet 或 PTv3（Sonata/Concerto）自监督编码器，采用跨时序 transformer query、对比损失、时空序列化与掩模等技术。

**📊 数据集**

在 3RScan（多时间室内扫描）与 ScanNet（单帧）混合训练数据集上评估；使用 3RScan 生成的长度为 2 的时间序列作为测试。

**📈 对比分析**

与 Mask4D、Mask4Former、Mask3D+语义/几何匹配 baseline 对比，ReScene4D 在 t‑mAP 由 38.9（Mask4Former）提升至 66.8（Concerto+对比+序列化），标准 mAP 亦从 21.9 提升至 81.9，显著优于所有基线，并在时间一致性上取得明显进步。

**⚠️ 局限性**

局限性主要在于数据集规模与多样性不足，3RScan 只包含有限的变化实例且注释不完整，导致对极少见变化和跨场景泛化的能力受限；同时方法在极端稀疏或大幅变化场景下仍有待进一步验证。

---

## 324. Industry Influence in High-Profile Social Media Research

**arXiv ID:** 2601.11507 | [PDF](https://arxiv.org/pdf/2601.11507v1)

**作者:** Joseph Bak-Coleman `[一作]`, Carl T. Bergstrom `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

**🎯 论文内容**

系统性地利用公开数据库（OpenAlex、Altmetrics、期刊竞争利益声明和编辑/审稿人公开简历）构建了高影响力社交媒体研究论文语料库，识别并量化作者、编辑和审稿人与Meta、X、Google、Microsoft等企业之间的雇佣、合作与资助关联，并评估这些关联对论文影响力、引用率、政策引用和公共舆论的影响。

**💡 创新点**

首次将行业关联与研究产出、评审过程、以及学术与公众影响力进行全链路关联分析，并揭示行业关系高度集中、披露率低、且与研究主题偏向（尤其在错误信息研究中）呈现显著关联的现象。

**🔧 技术方法**

采用贝叶斯二项回归、Gini系数与Lorenz曲线衡量行业资金分配不平衡；利用图谱聚类（bibliographic coupling network）识别五大研究主题；使用Altmetric与引用数据比较行业关联与独立研究的影响力；通过手工验证与公开简历交叉核对确认行业关联。

**📊 数据集**

公开数据集包括：OpenAlex论文及作者信息、Altmetrics在线关注度、期刊竞争利益政策细则、编辑与审稿人公开简历、以及行业公开资助与合作项目列表。

**📈 对比分析**

比较结果显示，行业关联论文在Altmetric得分与学术引用率上平均高约两倍；行业关联论文在政策文件、社交媒体、新闻报道和维基百科引用上显著多于独立论文；主题偏向分析发现，错误信息共享研究与行业关联比例高，平台动力学研究则相对较低；但整体影响力的差异并不完全由行业资源决定，仍需进一步探究。

**⚠️ 局限性**

局限包括：仅覆盖高影响力期刊论文，可能低估低影响力期刊的行业影响；依赖公开披露数据，无法捕捉未公开的资金、股权、咨询等关联；缺乏因果分析，无法确定行业关联是导致高影响力还是高影响力吸引行业资助；以及对主题偏向的分析仅限宏观层面，未深入细粒度的研究框架与结果偏差。

---

## 325. UniX: Unifying Autoregression and Diffusion for Chest X-Ray Understanding and Generation

**arXiv ID:** 2601.11522 | [PDF](https://arxiv.org/pdf/2601.11522v1)

**作者:** Ruiheng Zhang `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**通讯引用:** 28857 | [OpenAlex ID](https://openalex.org/A5060042752)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了UniX模型，实现胸部X光图像的理解与生成统一处理。

**💡 创新点**

核心创新在于将理解与生成解耦为自回归分支和扩散分支，并通过跨模态自注意力实现动态协同。

**🔧 技术方法**

使用自回归Transformer、VAE‑编码的扩散模型、跨模态自注意力以及三阶段训练策略。

**📊 数据集**

在MIMIC‑CXR数据集上进行理解训练，在ChexGenBench（MIMIC‑CXR图像‑报告对）上进行生成训练。

**📈 对比分析**

相较于LLM‑CXR，UniX在Micro‑F1上提升46.1%，在FD‑RadDino上提升24.2%，仅占参数的1/8（1.5B vs 12B），并匹配单任务模型性能。

**⚠️ 局限性**

仍受限于数据规模与多模态一致性评估，模型在不同病理类别的细粒度生成表现有待进一步验证。

---

## 326. Empirical Coordination over Markov Channel with Independent Source

**arXiv ID:** 2601.11520 | [PDF](https://arxiv.org/pdf/2601.11520v1)

**作者:** Mengyuan Zhao `[一作]` (KTH Royal Institute of Technology), Tobias J. Oechtering `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 2548 | [OpenAlex ID](https://openalex.org/A5079492269)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在马尔科夫信道上使用严格因果编码和非因果解码的联合源-信道编码，并给出了可实现的联合分布的单字母内外界限。

**💡 创新点**

创新点在于提出了“输入驱动马尔科夫典型性”概念，利用该典型性直接处理马尔科夫信道的记忆结构，并首次将协调编码框架推广到非记忆信道之外。

**🔧 技术方法**

采用的信息论工具包括典型性分析、AEP、封装与打包引理、辅助随机变量构造以及单字母互信息约束的推导。

**📊 数据集**

本工作为理论研究，无使用具体数据集，所有结论均在离散有限字母集合上证明。

**📈 对比分析**

与传统DMC协调编码结果对比，作者证明了在马尔科夫信道情形下的内外界限收敛到相应的DMC极限；文中未给出数值仿真或性能评估。

**⚠️ 局限性**

主要限制包括：外界限一般比内界宽松；假设信道满足唯一循环类、不可约且无周期性；仅考虑严格因果编码与非因果解码，未讨论反馈或更通用的编码策略。

---

## 327. MetaboNet: The Largest Publicly Available Consolidated Dataset for Type 1 Diabetes Management

**arXiv ID:** 2601.11505 | [PDF](https://arxiv.org/pdf/2601.11505v1)

**作者:** Miriam K. Wolff `[一作]` (Replica Health), Sam F. Royston `[通讯]` (Replica Health)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并统一了21个公开及受限的T1D管理数据集，生成统一格式的MetaboNet资源，提供可直接下载的公共子集与受限子集的处理脚本。

**💡 创新点**

通过跨来源标准化采样、统一单位和特征名称，整合了多达3135名患者、1228人年数据，显著扩大了数据规模和多样性，为T1D算法评估与泛化提供可重复的基准。

**🔧 技术方法**

使用了数据清洗、5分钟重采样、缺失值保留、重叠时间段去重、质量保障检测以及Python开源处理管道等技术，保证数据一致性与可重复性。

**📊 数据集**

集成了Loop观测、OpenAPS Commons、T1DEXI、T1DEXIP、Tidepool Donation等公开数据，以及CC BY 4.0授权的数据（如AZT1D、BrisT1D、Shanghai T1DM等），共21个来源。

**📈 对比分析**

提供血糖预测基准，比较基线、线性和非线性模型在30分钟预测窗口的RMSE，结果显示数据量越大预测误差越低；还展示了TIR与TITR关系以及跨人群糖监测指标的散点分析。

**⚠️ 局限性**

存在缺失值与零值歧义、少数族裔代表不足、跨来源处理原则差异导致残留不一致，以及部分数据需通过DUA申请获取。

---

## 328. Building Production-Ready Probes For Gemini

**arXiv ID:** 2601.11516 | [PDF](https://arxiv.org/pdf/2601.11516v1)

**作者:** János Kramár `[一作]` (Google DeepMind), Arthur Conmy `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在前沿大型语言模型中开发并评估激活探测器，用于检测网络攻击类提示并应对长上下文、多轮对话等生产环境分布漂移；

**💡 创新点**

提出 MultiMax 与 Max‑of‑Rolling‑Means Attention 这两种新探测器架构以提升长上下文泛化；使用 AlphaEvolve 自动进化搜索更优探测器；将探测器与 LLM 级联，实现低成本高准确率的监控；

**🔧 技术方法**

激活探测技术、Attention、EMA、MLP、MultiMax、Max‑of‑Rolling‑Means、AlphaEvolve 进化搜索、级联分类器、统计显著性检验、分布偏移评估等；

**📊 数据集**

多种网络攻击与非攻击提示数据集，包括短上下文/长上下文、multi‑turn、hard negatives、overtriggering、pre‑existing jailbreaks、adaptive red‑teaming 等；

**📈 对比分析**

与 Gemini 2.5 Flash/Pro、线性 probe、EMA、Attention 等基线对比，发现 MultiMax、Max‑of‑Rolling‑Means 与 AlphaEvolve 在测试误差上优于基线，且与 LLM 级联在成本‑准确率曲线上表现更佳；

**⚠️ 局限性**

对长上下文训练成本高、仍难以完全抵御自适应攻击、误差区间宽、仅评估输入监控、未覆盖所有层激活、对其他领域缺乏验证等限制。

---

## 329. Do explanations generalize across large reasoning models?

**arXiv ID:** 2601.11517 | [PDF](https://arxiv.org/pdf/2601.11517v1)

**作者:** Koyena Pal `[一作]` (Northeastern University), Chandan Singh `[通讯]` (Microsoft Research)

**通讯引用:** 3275 | [OpenAlex ID](https://openalex.org/A5017514239)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型推理模型（LRM）产生的链式思考（CoT）解释在不同模型间的泛化能力，并提出基于句子级别的集成方法提升一致性。

**💡 创新点**

首次定义跨模型一致性度量并证明CoT能在不同LRM间传递行为，提出简单集成策略并与人类偏好及强化学习后训练关联，展示一致性与解释可传递性的内在联系。

**🔧 技术方法**

采用多种CoT生成方式（空白、默认贪婪/采样、转移、集成），利用一致性与准确率评估模型性能，并开展人类用户研究及RL后训练实验。

**📊 数据集**

使用医学推理基准 MedCalc‑Bench 与通用推理基准 Instruction Induction（含 20 任务）进行实验。

**📈 对比分析**

在五大 LRM（NRR、OpenT、OSS、QwQ、DAPO）上评估，集成 CoT 使跨模型一致性从 25% 提升至 66%，在 Instruction Induction 上提升至 62%；人类评测显示一致性与用户偏好高度相关；RL 后训练显著提升一致性，但准确率提升不均衡。

**⚠️ 局限性**

局限包括：解释虽提升一致性但可能同向传递错误；一致性与准确率不一定同步；实验仅覆盖有限模型与任务，易受共性偏差影响，需进一步验证在更广泛场景与人类用户中的泛化效果。

---

## 330. How Long Is a Piece of String? A Brief Empirical Analysis of Tokenizers

**arXiv ID:** 2601.11518 | [PDF](https://arxiv.org/pdf/2601.11518v1)

**作者:** Jonathan Roberts `[一作]` (University of Cambridge), Samuel Albanie `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

系统性比较前沿LLM的分词器，在八大文本领域（论文、技术文档、代码、π数列、表情符号、UUID、金融表格、网页）以及英文高频词和随机词集上评估字符压缩率、词压缩率和上下文极限。

**💡 创新点**

①跨模型、跨领域的压缩率对比；②挑战并驳斥“一 token ≈ 0.75 词”的经验法则；③提出以字符数或等效 Llama‑3 essay token 为统一长度度量，揭示模型原生 token 计数的局限性。

**🔧 技术方法**

使用子词分词训练与分割理论，统计 token/字符/词的压缩比，采用 50 次确定性采样求平均压缩率与标准误；绘制误差条、置信区间，并将上下文极限转换为字符或等效 token 进行可视化对比。

**📊 数据集**

8 领域专用语料库（Paul Graham 论文、技术 arXiv 论文、Python 代码、π 前 10 万位、Emoji 集、UUID 词典、FinTabNet 金融表格、10 个热门网页）；英文高频词表（Google Trillion‑Word Corpus 10000 词）及 WordNet 随机词集。

**📈 对比分析**

通过统计不同 tokenizer 在相同文本样本上的 token 数、字符压缩率和词压缩率，计算标准误并绘制 95% 置信区间；将模型报告的原生 token 上下文极限转换为字符极限，再转为 Llama‑3 essay token 极限，展示模型间实际可处理文本长度差异。

**⚠️ 局限性**

限制：样本规模相对有限，仅评估 10 种 tokenizer；实验集中在文本领域，对非文本输入（如图像、音频）未做评估；未考虑 tokenizer 在动态推理成本、延迟等实际使用指标上的表现。

---

## 331. Capacity Constraints Make Admissions Processes Less Predictable

**arXiv ID:** 2601.11513 | [PDF](https://arxiv.org/pdf/2601.11513v1)

**作者:** Evan Dong `[一作]` (Cornell University), Sarah Dean `[通讯]` (Cornell University)

**通讯引用:** 7208 | [OpenAlex ID](https://openalex.org/A5034027561)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在容量受限的招生决策中，申请者池的变化会导致传统机器学习模型预测效果下降，并通过理论与纽约市高中匹配系统实证验证了这一结论。

**💡 创新点**

创新点在于提出了“不稳定性（instability）”和“可变性（variability）”两种衡量招录决策对申请者池依赖性的理论指标，并证明只有满足极小不稳定性且可变性为1的选择函数才可被标准机器学习模型完全表示。

**🔧 技术方法**

使用了理论分析（定义不稳定性、可变性、替代性、顺序化组合等），以及基于机器学习的预测模型（逻辑回归、梯度提升机）结合分位阈值决策。

**📊 数据集**

数据来源为纽约市教育局2021-2022学年高中匹配系统的真实申请与录取记录，并通过自建模拟器生成不同选择函数和不同申请者池下的对照录取结果。

**📈 对比分析**

与传统固定阈值或无容量约束的模型相比，针对不同不稳定性和可变性的选择函数，机器学习模型在应用于未来或混合池时表现出更快的准确率下降，尤其是不稳定性为1且可变性>1的决策更易出现误差。

**⚠️ 局限性**

局限性包括：仅评估了1-不稳定性的真实招生决策；对极端不可逆变化的实际影响仍不充分；实验基于纽约市高中系统，推广到人类主导或更复杂的招聘场景仍需进一步研究。

---

